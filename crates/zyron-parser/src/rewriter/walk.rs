//! Walking a parsed statement to rename what it references.
//!
//! Most rewrites are a rename: a relation moved, a function was renamed, a
//! column was renamed. Doing that on the parsed statement rather than on its
//! text is what keeps a rename out of string literals and comments, and what
//! lets an alias that happens to match the old name stay untouched.
//!
//! The walk covers every position a user object can put a name in: FROM
//! items, joins, subqueries in FROM and in expressions, projections, WHERE,
//! GROUP BY, HAVING, QUALIFY, ORDER BY, window partitions and frames, set
//! operation branches, and CTEs

use crate::ast::{Expr, FunctionArg, SelectItem, SelectStatement, Statement, TableRef};

/// What a walk is renaming
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RenameTarget {
    /// A relation named in FROM, in a join, or as a qualifier on a column
    Relation,
    /// A function called in an expression or in FROM
    Function,
    /// A column, wherever it is referenced unqualified or qualified
    Column,
}

/// Renames every occurrence of one name in a statement, returning how many
/// places changed.
///
/// The comparison is case insensitive because SQL identifiers fold, and the
/// replacement is written exactly as given
pub fn rename(statement: &mut Statement, target: RenameTarget, from: &str, to: &str) -> usize {
    let mut walker = Renamer {
        target,
        from,
        to,
        changed: 0,
    };
    walker.statement(statement);
    walker.changed
}

/// The query a statement carries, which is what a rename walks.
///
/// Every statement that holds a user-authored query holds exactly one, so
/// this is an option rather than a list
pub fn query_of(statement: &mut Statement) -> Option<&mut SelectStatement> {
    match statement {
        Statement::Select(select) => Some(select.as_mut()),
        Statement::CreateView(view) => Some(view.query.as_mut()),
        Statement::CreateMaterializedView(view) => Some(view.query.as_mut()),
        Statement::CreateStreamingJob(job) => Some(&mut job.query),
        Statement::Insert(insert) => match &mut insert.source {
            crate::ast::InsertSource::Query(query) => Some(query.as_mut()),
            crate::ast::InsertSource::Values(_) => None,
        },
        _ => None,
    }
}

struct Renamer<'a> {
    target: RenameTarget,
    from: &'a str,
    to: &'a str,
    changed: usize,
}

impl Renamer<'_> {
    /// Whether a name is the one being renamed, comparing the last part of a
    /// qualified name as well so `sales.warehouse` matches `warehouse`
    fn matches(&self, name: &str) -> bool {
        if name.eq_ignore_ascii_case(self.from) {
            return true;
        }
        match name.rsplit_once('.') {
            Some((_, last)) => last.eq_ignore_ascii_case(self.from),
            None => false,
        }
    }

    /// Replaces a name, keeping any schema qualifier it carried
    fn replace(&mut self, name: &mut String) {
        if !self.matches(name) {
            return;
        }
        match name.rsplit_once('.') {
            Some((prefix, _)) if !self.to.contains('.') => {
                *name = format!("{prefix}.{}", self.to);
            }
            _ => *name = self.to.to_string(),
        }
        self.changed += 1;
    }

    fn statement(&mut self, statement: &mut Statement) {
        if let Some(query) = query_of(statement) {
            self.select(query);
        }
    }

    fn select(&mut self, select: &mut SelectStatement) {
        if let Some(with) = select.with.as_mut() {
            for cte in with.ctes.iter_mut() {
                self.select(cte.query.as_mut());
            }
        }
        for item in select.projections.iter_mut() {
            match item {
                SelectItem::Expr(expr, _) => self.expr(expr),
                SelectItem::QualifiedWildcard(name) => {
                    if self.target == RenameTarget::Relation {
                        self.replace(name);
                    }
                }
                SelectItem::Wildcard => {}
            }
        }
        for from in select.from.iter_mut() {
            self.table_ref(from);
        }
        if let Some(predicate) = select.where_clause.as_mut() {
            self.expr(predicate);
        }
        for expr in select.group_by.iter_mut() {
            self.expr(expr);
        }
        if let Some(sets) = select.group_by_sets.as_mut() {
            match sets {
                crate::ast::GroupBySets::Rollup(exprs) | crate::ast::GroupBySets::Cube(exprs) => {
                    for expr in exprs.iter_mut() {
                        self.expr(expr);
                    }
                }
                crate::ast::GroupBySets::GroupingSets(groups) => {
                    for group in groups.iter_mut() {
                        for expr in group.iter_mut() {
                            self.expr(expr);
                        }
                    }
                }
            }
        }
        if let Some(having) = select.having.as_mut() {
            self.expr(having);
        }
        if let Some(qualify) = select.qualify.as_mut() {
            self.expr(qualify);
        }
        for op in select.set_ops.iter_mut() {
            self.select(op.right.as_mut());
        }
        for order in select.order_by.iter_mut() {
            self.expr(&mut order.expr);
        }
        if let Some(limit) = select.limit.as_mut() {
            self.expr(limit);
        }
        if let Some(offset) = select.offset.as_mut() {
            self.expr(offset);
        }
    }

    fn table_ref(&mut self, table: &mut TableRef) {
        match table {
            TableRef::Table { name, .. } => {
                if self.target == RenameTarget::Relation {
                    self.replace(name);
                }
            }
            TableRef::Join(join) => {
                self.table_ref(&mut join.left);
                self.table_ref(&mut join.right);
                if let Some(asof) = join.asof.as_mut() {
                    self.expr(&mut asof.condition);
                }
                match &mut join.condition {
                    crate::ast::JoinCondition::On(on) => self.expr(on),
                    crate::ast::JoinCondition::Using(columns) => {
                        if self.target == RenameTarget::Column {
                            for column in columns.iter_mut() {
                                self.replace(column);
                            }
                        }
                    }
                    crate::ast::JoinCondition::Natural | crate::ast::JoinCondition::None => {}
                }
            }
            TableRef::Subquery { query, .. } => self.select(query.as_mut()),
            TableRef::Lateral { subquery } => self.table_ref(subquery.as_mut()),
            TableRef::TableFunction(call) => {
                if self.target == RenameTarget::Function {
                    self.replace(&mut call.name);
                }
                for arg in call.args.iter_mut() {
                    self.function_arg(arg);
                }
            }
            TableRef::ExternalInline(_) => {}
            TableRef::Unnest(unnest) => {
                for array in unnest.arrays.iter_mut() {
                    self.expr(array);
                }
            }
            TableRef::Flatten(flatten) => self.expr(&mut flatten.input),
            TableRef::Pivot(pivot) => {
                self.table_ref(&mut pivot.input);
                for agg in pivot.aggregates.iter_mut() {
                    if self.target == RenameTarget::Function {
                        self.replace(&mut agg.function);
                    }
                    self.expr(&mut agg.argument);
                }
                self.expr(&mut pivot.pivot_column);
            }
            TableRef::Unpivot(unpivot) => {
                self.table_ref(&mut unpivot.input);
                if self.target == RenameTarget::Column {
                    for column in unpivot.value_columns.iter_mut() {
                        self.replace(column);
                    }
                    self.replace(&mut unpivot.name_column);
                    for item in unpivot.items.iter_mut() {
                        for column in item.columns.iter_mut() {
                            self.replace(column);
                        }
                    }
                }
            }
        }
    }

    fn function_arg(&mut self, arg: &mut FunctionArg) {
        match arg {
            FunctionArg::Unnamed(expr) => self.expr(expr),
            FunctionArg::Named { value, .. } => self.expr(value),
            FunctionArg::Wildcard => {}
        }
    }

    fn expr(&mut self, expr: &mut Expr) {
        match expr {
            Expr::Identifier(name) => {
                if self.target == RenameTarget::Column {
                    self.replace(name);
                }
            }
            Expr::QualifiedIdentifier { table, column } => match self.target {
                RenameTarget::Relation => self.replace(table),
                RenameTarget::Column => self.replace(column),
                RenameTarget::Function => {}
            },
            Expr::Literal(_) | Expr::Parameter(_) => {}
            // The parameter names one element rather than a column of a
            // relation, so a column rename does not reach it
            Expr::Lambda { body, .. } => self.expr(body),
            Expr::Collate { expr, .. } => self.expr(expr),
            Expr::BinaryOp { left, right, .. } => {
                self.expr(left);
                self.expr(right);
            }
            Expr::UnaryOp { expr, .. } => self.expr(expr),
            Expr::IsNull { expr, .. } => self.expr(expr),
            Expr::InList { expr, list, .. } => {
                self.expr(expr);
                for item in list.iter_mut() {
                    self.expr(item);
                }
            }
            Expr::Between {
                expr, low, high, ..
            } => {
                self.expr(expr);
                self.expr(low);
                self.expr(high);
            }
            Expr::Like { expr, pattern, .. } | Expr::ILike { expr, pattern, .. } => {
                self.expr(expr);
                self.expr(pattern);
            }
            Expr::Function { name, args, .. } => {
                if self.target == RenameTarget::Function {
                    self.replace(name);
                }
                for arg in args.iter_mut() {
                    self.function_arg(arg);
                }
            }
            Expr::Cast { expr, .. } => self.expr(expr),
            Expr::Case {
                operand,
                conditions,
                else_result,
            } => {
                if let Some(operand) = operand.as_mut() {
                    self.expr(operand);
                }
                for when in conditions.iter_mut() {
                    self.expr(&mut when.condition);
                    self.expr(&mut when.result);
                }
                if let Some(else_result) = else_result.as_mut() {
                    self.expr(else_result);
                }
            }
            Expr::Nested(inner) => self.expr(inner),
            Expr::Subquery(query) | Expr::AnySubquery { query } | Expr::AllSubquery { query } => {
                self.select(query.as_mut())
            }
            Expr::InSubquery { expr, query, .. } => {
                self.expr(expr);
                self.select(query.as_mut());
            }
            Expr::Exists { query, .. } => self.select(query.as_mut()),
            Expr::WindowFunction {
                function,
                partition_by,
                order_by,
                ..
            } => {
                self.expr(function);
                for expr in partition_by.iter_mut() {
                    self.expr(expr);
                }
                for order in order_by.iter_mut() {
                    self.expr(&mut order.expr);
                }
            }
            Expr::ArrayConstructor(items) => {
                for item in items.iter_mut() {
                    self.expr(item);
                }
            }
            Expr::ArraySubscript { array, index } => {
                self.expr(array);
                self.expr(index);
            }
            Expr::JsonAccess { left, right, .. }
            | Expr::JsonContains { left, right, .. }
            | Expr::JsonExists { left, right, .. }
            | Expr::VectorDistance { left, right, .. } => {
                self.expr(left);
                self.expr(right);
            }
            Expr::MatchAgainst { columns, query, .. } => {
                if self.target == RenameTarget::Column {
                    for column in columns.iter_mut() {
                        self.replace(column);
                    }
                }
                self.expr(query);
            }
            Expr::TemporalRef { inner, .. } => self.expr(inner),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parse;

    fn only(sql: &str) -> Statement {
        let mut stmts = parse(sql).unwrap_or_else(|e| panic!("`{sql}` did not parse, {e}"));
        stmts.remove(0)
    }

    #[test]
    fn test_renames_a_relation_in_from() {
        let mut stmt = only("SELECT a FROM warehouse WHERE a > 1");
        assert_eq!(
            rename(&mut stmt, RenameTarget::Relation, "warehouse", "compute"),
            1
        );
        let rendered = format!("{stmt:?}");
        assert!(rendered.contains("compute"), "{rendered}");
        assert!(!rendered.contains("warehouse"), "{rendered}");
    }

    #[test]
    fn test_renames_a_relation_through_a_join_and_a_subquery() {
        let mut stmt =
            only("SELECT w.a FROM warehouse w JOIN (SELECT b FROM warehouse) s ON w.a = s.b");
        let changed = rename(&mut stmt, RenameTarget::Relation, "warehouse", "compute");
        assert_eq!(changed, 2, "both the join side and the subquery");
    }

    #[test]
    fn test_renames_a_relation_inside_a_cte_and_a_set_operation() {
        let mut stmt = only(
            "WITH c AS (SELECT a FROM warehouse) SELECT a FROM c \
             UNION SELECT a FROM warehouse",
        );
        assert_eq!(
            rename(&mut stmt, RenameTarget::Relation, "warehouse", "compute"),
            2
        );
    }

    #[test]
    fn test_a_qualifier_is_renamed_but_a_column_of_the_same_name_is_not() {
        let mut stmt = only("SELECT warehouse.id FROM warehouse");
        assert_eq!(
            rename(&mut stmt, RenameTarget::Relation, "warehouse", "compute"),
            2
        );
        let mut stmt = only("SELECT warehouse FROM t");
        assert_eq!(
            rename(&mut stmt, RenameTarget::Relation, "warehouse", "compute"),
            0,
            "a bare column reference is not a relation"
        );
    }

    #[test]
    fn test_renames_a_function_call() {
        let mut stmt = only("SELECT old_fn(a, b) FROM t");
        assert_eq!(
            rename(&mut stmt, RenameTarget::Function, "old_fn", "new_fn"),
            1
        );
        let rendered = format!("{stmt:?}");
        assert!(rendered.contains("new_fn"), "{rendered}");
    }

    #[test]
    fn test_renames_a_column_in_a_predicate_and_a_projection() {
        let mut stmt = only("SELECT old_col FROM t WHERE old_col > 1 ORDER BY old_col");
        assert_eq!(
            rename(&mut stmt, RenameTarget::Column, "old_col", "new_col"),
            3
        );
    }

    #[test]
    fn test_a_schema_qualifier_is_kept() {
        let mut stmt = only("SELECT a FROM sales.warehouse");
        assert_eq!(
            rename(&mut stmt, RenameTarget::Relation, "warehouse", "compute"),
            1
        );
        let rendered = format!("{stmt:?}");
        assert!(rendered.contains("sales.compute"), "{rendered}");
    }

    #[test]
    fn test_a_string_literal_is_left_alone() {
        let mut stmt = only("SELECT 'warehouse' FROM t");
        assert_eq!(
            rename(&mut stmt, RenameTarget::Relation, "warehouse", "compute"),
            0
        );
        assert_eq!(
            rename(&mut stmt, RenameTarget::Column, "warehouse", "compute"),
            0
        );
    }

    #[test]
    fn test_a_view_body_is_walked() {
        let mut stmt = only("CREATE VIEW v AS SELECT a FROM warehouse");
        assert_eq!(
            rename(&mut stmt, RenameTarget::Relation, "warehouse", "compute"),
            1
        );
    }

    #[test]
    fn test_nothing_matching_changes_nothing() {
        let mut stmt = only("SELECT a FROM t");
        let before = format!("{stmt:?}");
        assert_eq!(
            rename(&mut stmt, RenameTarget::Relation, "warehouse", "compute"),
            0
        );
        assert_eq!(before, format!("{stmt:?}"));
    }
}
