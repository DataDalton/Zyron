//! Desugars MERGE into UPDATE, DELETE and INSERT statements that execute
//! sequentially inside one transaction.
//!
//! The rewrite leans on correlated subqueries, which the engine executes in
//! every expression position. A MATCHED action becomes a statement over the
//! target gated by EXISTS against the source, a NOT MATCHED INSERT becomes
//! INSERT INTO target SELECT FROM source gated by NOT EXISTS against the
//! target. Assignment values are wrapped in scalar subqueries over the
//! source so they can reference source columns, and a source that matches
//! one target row twice surfaces as the scalar subquery cardinality error,
//! mirroring the SQL rule that MERGE may not affect a row twice.
//!
//! Sequential execution inside one transaction is equivalent to single
//! snapshot MERGE semantics only under guardrails, which are enforced here
//! rather than silently diverging
//! - at most one WHEN MATCHED and one WHEN NOT MATCHED clause
//! - UPDATE assignments may not touch columns the ON condition references,
//!   so updating a row never changes whether it is matched
//! - WHEN MATCHED DELETE cannot combine with WHEN NOT MATCHED INSERT,
//!   because the insert would resurrect rows the delete just matched
//!
//! Execution order is UPDATE, DELETE, INSERT. With the guardrails above,
//! updated rows stay matched so the trailing NOT EXISTS insert skips them,
//! which makes the common upsert shape exact.

use crate::ast::{
    Assignment, BinaryOperator, DeleteStatement, Expr, InsertSource, InsertStatement, MergeAction,
    MergeClause, MergeStatement, SelectItem, SelectStatement, SoftDeleteSelectMode, Statement,
    TableRef, UpdateStatement,
};
use zyron_common::{Result, ZyronError};

/// Minimal SELECT wrapper, everything defaulted except the given pieces
fn bare_select(
    projections: Vec<SelectItem>,
    from: Vec<TableRef>,
    where_clause: Option<Box<Expr>>,
) -> SelectStatement {
    SelectStatement {
        with: None,
        distinct: false,
        distinct_on: Vec::new(),
        projections,
        from,
        where_clause,
        group_by: Vec::new(),
        group_by_sets: None,
        having: None,
        qualify: None,
        set_ops: Vec::new(),
        order_by: Vec::new(),
        limit: None,
        offset: None,
        fetch: None,
        for_clause: None,
        soft_delete_mode: SoftDeleteSelectMode::Default,
    }
}

fn and(a: Expr, b: Expr) -> Expr {
    Expr::BinaryOp {
        left: Box::new(a),
        op: BinaryOperator::And,
        right: Box::new(b),
    }
}

/// ON condition combined with an optional clause condition
fn match_predicate(on: &Expr, clause_cond: &Option<Expr>) -> Expr {
    match clause_cond {
        Some(c) => and(on.clone(), c.clone()),
        None => on.clone(),
    }
}

/// Column names the ON condition mentions with the target table qualifier,
/// plus every unqualified identifier, which conservatively counts as a
/// possible target column reference
fn on_target_columns(on: &Expr, target: &str, out: &mut Vec<String>) {
    match on {
        Expr::Identifier(name) => out.push(name.to_lowercase()),
        Expr::QualifiedIdentifier { table, column } => {
            if table.eq_ignore_ascii_case(target) {
                out.push(column.to_lowercase());
            }
        }
        Expr::BinaryOp { left, right, .. } => {
            on_target_columns(left, target, out);
            on_target_columns(right, target, out);
        }
        Expr::UnaryOp { expr, .. } | Expr::Nested(expr) | Expr::Cast { expr, .. } => {
            on_target_columns(expr, target, out);
        }
        Expr::IsNull { expr, .. } => on_target_columns(expr, target, out),
        Expr::Between {
            expr, low, high, ..
        } => {
            on_target_columns(expr, target, out);
            on_target_columns(low, target, out);
            on_target_columns(high, target, out);
        }
        Expr::InList { expr, list, .. } => {
            on_target_columns(expr, target, out);
            for e in list {
                on_target_columns(e, target, out);
            }
        }
        Expr::Function { args, .. } => {
            for a in args {
                if let crate::ast::FunctionArg::Unnamed(e)
                | crate::ast::FunctionArg::Named { value: e, .. } = a
                {
                    on_target_columns(e, target, out);
                }
            }
        }
        _ => {}
    }
}

/// Rewrites MERGE into 0 to 2 plain statements, in execution order
pub fn desugar_merge(stmt: &MergeStatement) -> Result<Vec<Statement>> {
    let mut matched: Option<(&Option<Expr>, &MergeAction)> = None;
    let mut not_matched: Option<(&Option<Expr>, &MergeAction)> = None;
    for clause in &stmt.clauses {
        match clause {
            MergeClause::WhenMatched { condition, action } => {
                if matched.is_some() {
                    return Err(ZyronError::ParseError(
                        "MERGE supports at most one WHEN MATCHED clause".to_string(),
                    ));
                }
                matched = Some((condition, action));
            }
            MergeClause::WhenNotMatched { condition, action } => {
                if not_matched.is_some() {
                    return Err(ZyronError::ParseError(
                        "MERGE supports at most one WHEN NOT MATCHED clause".to_string(),
                    ));
                }
                not_matched = Some((condition, action));
            }
        }
    }

    let has_matched_delete = matches!(matched, Some((_, MergeAction::Delete)));
    let has_insert = matches!(not_matched, Some((_, MergeAction::Insert { .. })));
    if has_matched_delete && has_insert {
        return Err(ZyronError::ParseError(
            "MERGE combining WHEN MATCHED DELETE with WHEN NOT MATCHED INSERT needs \
             single snapshot semantics, run the DELETE and the INSERT as separate \
             statements"
                .to_string(),
        ));
    }

    let mut out: Vec<Statement> = Vec::new();

    if let Some((cond, action)) = matched {
        match action {
            MergeAction::Update(assignments) => {
                let mut on_cols = Vec::new();
                on_target_columns(&stmt.on, &stmt.target, &mut on_cols);
                for a in assignments {
                    if on_cols.contains(&a.column.to_lowercase()) {
                        return Err(ZyronError::ParseError(format!(
                            "MERGE UPDATE may not assign column {} because the ON \
                             condition references it, updating it would change \
                             whether the row is matched",
                            a.column
                        )));
                    }
                }
                let pred = match_predicate(&stmt.on, cond);
                // each value becomes a scalar subquery over the source so it
                // can reference source columns, a doubly matching source row
                // fails with the subquery cardinality error as MERGE requires
                let rewritten: Vec<Assignment> = assignments
                    .iter()
                    .map(|a| Assignment {
                        column: a.column.clone(),
                        value: Expr::Subquery(Box::new(bare_select(
                            vec![SelectItem::Expr(a.value.clone(), None)],
                            vec![stmt.source.clone()],
                            Some(Box::new(pred.clone())),
                        ))),
                    })
                    .collect();
                let gate = Expr::Exists {
                    query: Box::new(bare_select(
                        vec![SelectItem::Expr(
                            Expr::Literal(crate::ast::LiteralValue::Integer(1)),
                            None,
                        )],
                        vec![stmt.source.clone()],
                        Some(Box::new(pred)),
                    )),
                    negated: false,
                };
                out.push(Statement::Update(Box::new(UpdateStatement {
                    table: stmt.target.clone(),
                    assignments: rewritten,
                    where_clause: Some(Box::new(gate)),
                    returning: None,
                })));
            }
            MergeAction::Delete => {
                let pred = match_predicate(&stmt.on, cond);
                let gate = Expr::Exists {
                    query: Box::new(bare_select(
                        vec![SelectItem::Expr(
                            Expr::Literal(crate::ast::LiteralValue::Integer(1)),
                            None,
                        )],
                        vec![stmt.source.clone()],
                        Some(Box::new(pred)),
                    )),
                    negated: false,
                };
                out.push(Statement::Delete(Box::new(DeleteStatement {
                    table: stmt.target.clone(),
                    where_clause: Some(Box::new(gate)),
                    returning: None,
                    hard: false,
                })));
            }
            MergeAction::DoNothing => {}
            MergeAction::Insert { .. } => {
                return Err(ZyronError::ParseError(
                    "WHEN MATCHED cannot INSERT".to_string(),
                ));
            }
        }
    }

    if let Some((cond, action)) = not_matched {
        match action {
            MergeAction::Insert { columns, values } => {
                // NOT EXISTS over the target keeps only source rows without a
                // match, the clause condition narrows the source rows
                let anti = Expr::Exists {
                    query: Box::new(bare_select(
                        vec![SelectItem::Expr(
                            Expr::Literal(crate::ast::LiteralValue::Integer(1)),
                            None,
                        )],
                        vec![TableRef::Table {
                            name: stmt.target.clone(),
                            alias: None,
                            as_of: None,
                        }],
                        Some(Box::new(stmt.on.clone())),
                    )),
                    negated: true,
                };
                let where_clause = match cond {
                    Some(c) => and(anti, c.clone()),
                    None => anti,
                };
                let select = bare_select(
                    values
                        .iter()
                        .map(|v| SelectItem::Expr(v.clone(), None))
                        .collect(),
                    vec![stmt.source.clone()],
                    Some(Box::new(where_clause)),
                );
                out.push(Statement::Insert(Box::new(InsertStatement {
                    table: stmt.target.clone(),
                    columns: columns.clone(),
                    source: InsertSource::Query(Box::new(select)),
                    on_conflict: None,
                    returning: None,
                })));
            }
            MergeAction::DoNothing => {}
            MergeAction::Update(_) | MergeAction::Delete => {
                return Err(ZyronError::ParseError(
                    "WHEN NOT MATCHED supports only INSERT or DO NOTHING".to_string(),
                ));
            }
        }
    }

    Ok(out)
}
