//! AST-to-bound-plan conversion with name resolution and type checking.
//!
//! The binder converts parsed AST nodes (string-based names) into bound nodes
//! (OID-based references with resolved types). This is the semantic analysis phase
//! that validates column existence, type compatibility, and resolves ambiguity.

use std::collections::HashMap;
use std::sync::Arc;
use zyron_catalog::{Catalog, ColumnId, NameResolver, TableEntry, TableId};
use zyron_common::{Result, TypeId, ZyronError};
use zyron_parser::ast::*;

// ---------------------------------------------------------------------------
// Resolved column reference
// ---------------------------------------------------------------------------

/// Uniquely identifies a column within a query scope.
/// table_idx is a local index into BindContext.tables, not the catalog TableId.
/// This correctly handles self-joins and subquery aliases.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ColumnRef {
    pub table_idx: usize,
    pub column_id: ColumnId,
    pub type_id: TypeId,
    pub nullable: bool,
}

// ---------------------------------------------------------------------------
// Bound table and column definitions
// ---------------------------------------------------------------------------

/// Metadata for a table (or subquery) registered in a bind scope.
#[derive(Debug, Clone)]
pub struct BoundTableRef {
    pub table_idx: usize,
    pub table_id: Option<TableId>,
    pub alias: String,
    pub columns: Vec<BoundColumnDef>,
    pub entry: Option<Arc<TableEntry>>,
}

/// Column definition within a bound table scope.
#[derive(Debug, Clone, PartialEq)]
pub struct BoundColumnDef {
    pub column_id: ColumnId,
    pub name: String,
    pub type_id: TypeId,
    pub nullable: bool,
    pub ordinal: u16,
}

// ---------------------------------------------------------------------------
// Bind context (scope for name resolution)
// ---------------------------------------------------------------------------

/// Scope for name resolution. Tracks visible tables and their columns.
/// A new context is pushed for each subquery.
#[derive(Debug)]
pub struct BindContext {
    pub tables: Vec<BoundTableRef>,
    pub ctes: HashMap<String, BoundCte>,
    pub outer: Option<Box<BindContext>>,
}

impl BindContext {
    pub fn new() -> Self {
        Self {
            tables: Vec::new(),
            ctes: HashMap::new(),
            outer: None,
        }
    }

    pub fn with_outer(outer: Box<BindContext>) -> Self {
        Self {
            tables: Vec::new(),
            ctes: HashMap::new(),
            outer: Some(outer),
        }
    }
}

/// A bound CTE storing the bound query and output column schema.
#[derive(Debug, Clone)]
pub struct BoundCte {
    pub name: String,
    pub columns: Vec<BoundColumnDef>,
    pub query: Box<BoundSelect>,
}

// ---------------------------------------------------------------------------
// Bound expressions
// ---------------------------------------------------------------------------

/// Type-resolved expression. Replaces string identifiers with ColumnRef.
/// PartialEq is implemented manually because Subquery/Exists/InSubquery
/// variants contain BoundSelect which has Arc<TableEntry> (no PartialEq).
/// Those variants compare as structurally unequal, which is conservative
/// and correct for the optimizer's change-detection logic.
#[derive(Debug, Clone)]
pub enum BoundExpr {
    ColumnRef(ColumnRef),
    Literal {
        value: LiteralValue,
        type_id: TypeId,
    },
    BinaryOp {
        left: Box<BoundExpr>,
        op: BinaryOperator,
        right: Box<BoundExpr>,
        type_id: TypeId,
    },
    UnaryOp {
        op: UnaryOperator,
        expr: Box<BoundExpr>,
        type_id: TypeId,
    },
    IsNull {
        expr: Box<BoundExpr>,
        negated: bool,
    },
    InList {
        expr: Box<BoundExpr>,
        list: Vec<BoundExpr>,
        negated: bool,
    },
    Between {
        expr: Box<BoundExpr>,
        low: Box<BoundExpr>,
        high: Box<BoundExpr>,
        negated: bool,
    },
    Like {
        expr: Box<BoundExpr>,
        pattern: Box<BoundExpr>,
        negated: bool,
    },
    ILike {
        expr: Box<BoundExpr>,
        pattern: Box<BoundExpr>,
        negated: bool,
    },
    Function {
        name: String,
        args: Vec<BoundExpr>,
        return_type: TypeId,
        distinct: bool,
    },
    AggregateFunction {
        name: String,
        args: Vec<BoundExpr>,
        distinct: bool,
        return_type: TypeId,
    },
    Cast {
        expr: Box<BoundExpr>,
        target_type: TypeId,
    },
    Case {
        operand: Option<Box<BoundExpr>>,
        conditions: Vec<BoundWhen>,
        else_result: Option<Box<BoundExpr>>,
        type_id: TypeId,
    },
    Nested(Box<BoundExpr>),
    Subquery {
        plan: Box<BoundSelect>,
        type_id: TypeId,
    },
    Exists {
        plan: Box<BoundSelect>,
        negated: bool,
    },
    InSubquery {
        expr: Box<BoundExpr>,
        plan: Box<BoundSelect>,
        negated: bool,
    },
    WindowFunction {
        function: Box<BoundExpr>,
        partition_by: Vec<BoundExpr>,
        order_by: Vec<BoundOrderBy>,
        frame: Option<WindowFrame>,
        type_id: TypeId,
    },
    Parameter {
        index: usize,
        type_id: TypeId,
    },
}

impl PartialEq for BoundExpr {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (BoundExpr::ColumnRef(a), BoundExpr::ColumnRef(b)) => a == b,
            (
                BoundExpr::Literal {
                    value: v1,
                    type_id: t1,
                },
                BoundExpr::Literal {
                    value: v2,
                    type_id: t2,
                },
            ) => v1 == v2 && t1 == t2,
            (
                BoundExpr::BinaryOp {
                    left: l1,
                    op: o1,
                    right: r1,
                    type_id: t1,
                },
                BoundExpr::BinaryOp {
                    left: l2,
                    op: o2,
                    right: r2,
                    type_id: t2,
                },
            ) => o1 == o2 && t1 == t2 && l1 == l2 && r1 == r2,
            (
                BoundExpr::UnaryOp {
                    op: o1,
                    expr: e1,
                    type_id: t1,
                },
                BoundExpr::UnaryOp {
                    op: o2,
                    expr: e2,
                    type_id: t2,
                },
            ) => o1 == o2 && t1 == t2 && e1 == e2,
            (
                BoundExpr::IsNull {
                    expr: e1,
                    negated: n1,
                },
                BoundExpr::IsNull {
                    expr: e2,
                    negated: n2,
                },
            ) => n1 == n2 && e1 == e2,
            (
                BoundExpr::InList {
                    expr: e1,
                    list: l1,
                    negated: n1,
                },
                BoundExpr::InList {
                    expr: e2,
                    list: l2,
                    negated: n2,
                },
            ) => n1 == n2 && e1 == e2 && l1 == l2,
            (
                BoundExpr::Between {
                    expr: e1,
                    low: lo1,
                    high: hi1,
                    negated: n1,
                },
                BoundExpr::Between {
                    expr: e2,
                    low: lo2,
                    high: hi2,
                    negated: n2,
                },
            ) => n1 == n2 && e1 == e2 && lo1 == lo2 && hi1 == hi2,
            (
                BoundExpr::Like {
                    expr: e1,
                    pattern: p1,
                    negated: n1,
                },
                BoundExpr::Like {
                    expr: e2,
                    pattern: p2,
                    negated: n2,
                },
            ) => n1 == n2 && e1 == e2 && p1 == p2,
            (
                BoundExpr::ILike {
                    expr: e1,
                    pattern: p1,
                    negated: n1,
                },
                BoundExpr::ILike {
                    expr: e2,
                    pattern: p2,
                    negated: n2,
                },
            ) => n1 == n2 && e1 == e2 && p1 == p2,
            (
                BoundExpr::Function {
                    name: n1,
                    args: a1,
                    return_type: r1,
                    distinct: d1,
                },
                BoundExpr::Function {
                    name: n2,
                    args: a2,
                    return_type: r2,
                    distinct: d2,
                },
            ) => n1 == n2 && d1 == d2 && r1 == r2 && a1 == a2,
            (
                BoundExpr::AggregateFunction {
                    name: n1,
                    args: a1,
                    distinct: d1,
                    return_type: r1,
                },
                BoundExpr::AggregateFunction {
                    name: n2,
                    args: a2,
                    distinct: d2,
                    return_type: r2,
                },
            ) => n1 == n2 && d1 == d2 && r1 == r2 && a1 == a2,
            (
                BoundExpr::Cast {
                    expr: e1,
                    target_type: t1,
                },
                BoundExpr::Cast {
                    expr: e2,
                    target_type: t2,
                },
            ) => t1 == t2 && e1 == e2,
            (
                BoundExpr::Case {
                    operand: o1,
                    conditions: c1,
                    else_result: e1,
                    type_id: t1,
                },
                BoundExpr::Case {
                    operand: o2,
                    conditions: c2,
                    else_result: e2,
                    type_id: t2,
                },
            ) => t1 == t2 && o1 == o2 && c1 == c2 && e1 == e2,
            (BoundExpr::Nested(a), BoundExpr::Nested(b)) => a == b,
            (
                BoundExpr::Parameter {
                    index: i1,
                    type_id: t1,
                },
                BoundExpr::Parameter {
                    index: i2,
                    type_id: t2,
                },
            ) => i1 == i2 && t1 == t2,
            (
                BoundExpr::WindowFunction {
                    function: f1,
                    partition_by: p1,
                    order_by: o1,
                    frame: fr1,
                    type_id: t1,
                },
                BoundExpr::WindowFunction {
                    function: f2,
                    partition_by: p2,
                    order_by: o2,
                    frame: fr2,
                    type_id: t2,
                },
            ) => t1 == t2 && f1 == f2 && p1 == p2 && o1 == o2 && fr1 == fr2,
            // Subquery variants: conservatively unequal (contain Arc<TableEntry>).
            (BoundExpr::Subquery { .. }, BoundExpr::Subquery { .. }) => false,
            (BoundExpr::Exists { .. }, BoundExpr::Exists { .. }) => false,
            (BoundExpr::InSubquery { .. }, BoundExpr::InSubquery { .. }) => false,
            _ => false,
        }
    }
}

impl BoundExpr {
    /// Returns the output TypeId of this expression.
    pub fn type_id(&self) -> TypeId {
        match self {
            BoundExpr::ColumnRef(cr) => cr.type_id,
            BoundExpr::Literal { type_id, .. } => *type_id,
            BoundExpr::BinaryOp { type_id, .. } => *type_id,
            BoundExpr::UnaryOp { type_id, .. } => *type_id,
            BoundExpr::IsNull { .. } => TypeId::Boolean,
            BoundExpr::InList { .. } => TypeId::Boolean,
            BoundExpr::Between { .. } => TypeId::Boolean,
            BoundExpr::Like { .. } => TypeId::Boolean,
            BoundExpr::ILike { .. } => TypeId::Boolean,
            BoundExpr::Function { return_type, .. } => *return_type,
            BoundExpr::AggregateFunction { return_type, .. } => *return_type,
            BoundExpr::Cast { target_type, .. } => *target_type,
            BoundExpr::Case { type_id, .. } => *type_id,
            BoundExpr::Nested(inner) => inner.type_id(),
            BoundExpr::Subquery { type_id, .. } => *type_id,
            BoundExpr::Exists { .. } => TypeId::Boolean,
            BoundExpr::InSubquery { .. } => TypeId::Boolean,
            BoundExpr::WindowFunction { type_id, .. } => *type_id,
            BoundExpr::Parameter { type_id, .. } => *type_id,
        }
    }

    /// Returns true if this expression can produce NULL.
    pub fn nullable(&self) -> bool {
        match self {
            BoundExpr::ColumnRef(cr) => cr.nullable,
            BoundExpr::Literal { value, .. } => matches!(value, LiteralValue::Null),
            BoundExpr::IsNull { .. } => false,
            BoundExpr::Exists { .. } => false,
            BoundExpr::InSubquery { .. } => false,
            _ => true,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct BoundWhen {
    pub condition: BoundExpr,
    pub result: BoundExpr,
}

#[derive(Debug, Clone, PartialEq)]
pub struct BoundOrderBy {
    pub expr: BoundExpr,
    pub asc: bool,
    pub nulls_first: bool,
}

// ---------------------------------------------------------------------------
// Bound statements
// ---------------------------------------------------------------------------

/// Top-level bound statement.
#[derive(Debug, Clone)]
pub enum BoundStatement {
    Select(BoundSelect),
    Insert(BoundInsert),
    Update(BoundUpdate),
    Delete(BoundDelete),
}

#[derive(Debug, Clone)]
pub struct BoundSelect {
    pub projections: Vec<BoundSelectItem>,
    pub from: Vec<BoundFromItem>,
    pub where_clause: Option<BoundExpr>,
    pub group_by: Vec<BoundExpr>,
    pub having: Option<BoundExpr>,
    pub order_by: Vec<BoundOrderBy>,
    pub limit: Option<BoundExpr>,
    pub offset: Option<BoundExpr>,
    pub distinct: bool,
    pub set_ops: Vec<BoundSetOp>,
    pub ctes: Vec<BoundCte>,
    pub output_schema: Vec<BoundColumnDef>,
}

#[derive(Debug, Clone)]
pub enum BoundSelectItem {
    Expr(BoundExpr, Option<String>),
    AllColumns(usize),
    Wildcard,
}

#[derive(Debug, Clone)]
pub enum BoundFromItem {
    BaseTable {
        table_idx: usize,
        table_id: TableId,
        entry: Arc<TableEntry>,
    },
    Join {
        left: Box<BoundFromItem>,
        join_type: JoinType,
        right: Box<BoundFromItem>,
        condition: BoundJoinCondition,
    },
    Subquery {
        table_idx: usize,
        query: Box<BoundSelect>,
    },
}

#[derive(Debug, Clone)]
pub enum BoundJoinCondition {
    On(BoundExpr),
    Using(Vec<ColumnId>),
    Natural,
    None,
}

#[derive(Debug, Clone)]
pub struct BoundSetOp {
    pub op: SetOpType,
    pub all: bool,
    pub right: Box<BoundSelect>,
}

#[derive(Debug, Clone)]
pub struct BoundInsert {
    pub table_id: TableId,
    pub table_entry: Arc<TableEntry>,
    pub target_columns: Vec<ColumnId>,
    pub source: BoundInsertSource,
    pub returning: Option<Vec<BoundSelectItem>>,
}

#[derive(Debug, Clone)]
pub enum BoundInsertSource {
    Values(Vec<Vec<BoundExpr>>),
    Query(Box<BoundSelect>),
}

#[derive(Debug, Clone)]
pub struct BoundUpdate {
    pub table_id: TableId,
    pub table_entry: Arc<TableEntry>,
    pub assignments: Vec<BoundAssignment>,
    pub where_clause: Option<BoundExpr>,
    pub returning: Option<Vec<BoundSelectItem>>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct BoundAssignment {
    pub column_id: ColumnId,
    pub value: BoundExpr,
}

#[derive(Debug, Clone)]
pub struct BoundDelete {
    pub table_id: TableId,
    pub table_entry: Arc<TableEntry>,
    pub where_clause: Option<BoundExpr>,
    pub returning: Option<Vec<BoundSelectItem>>,
}

// ---------------------------------------------------------------------------
// Aggregate function names
// ---------------------------------------------------------------------------

const AGGREGATE_FUNCTIONS: &[&str] = &["count", "sum", "avg", "min", "max"];

fn is_aggregate_function(name: &str) -> bool {
    AGGREGATE_FUNCTIONS
        .iter()
        .any(|&f| f.eq_ignore_ascii_case(name))
}

// ---------------------------------------------------------------------------
// Binder
// ---------------------------------------------------------------------------

/// Converts parsed AST into bound plan with resolved OIDs and type information.
#[allow(dead_code)]
pub struct Binder<'a> {
    resolver: NameResolver,
    catalog: &'a Catalog,
    next_table_idx: usize,
}

impl<'a> Binder<'a> {
    pub fn new(resolver: NameResolver, catalog: &'a Catalog) -> Self {
        Self {
            resolver,
            catalog,
            next_table_idx: 0,
        }
    }

    fn alloc_table_idx(&mut self) -> usize {
        let idx = self.next_table_idx;
        self.next_table_idx += 1;
        idx
    }

    /// Main entry point. Dispatches to statement-specific binders.
    pub async fn bind(&mut self, stmt: Statement) -> Result<BoundStatement> {
        match stmt {
            Statement::Select(s) => {
                let mut ctx = BindContext::new();
                let bound = self.bind_select(&mut ctx, &s).await?;
                Ok(BoundStatement::Select(bound))
            }
            Statement::Insert(s) => {
                let bound = self.bind_insert(&s).await?;
                Ok(BoundStatement::Insert(bound))
            }
            Statement::Update(s) => {
                let bound = self.bind_update(&s).await?;
                Ok(BoundStatement::Update(bound))
            }
            Statement::Delete(s) => {
                let bound = self.bind_delete(&s).await?;
                Ok(BoundStatement::Delete(bound))
            }
            other => Err(ZyronError::PlanError(format!(
                "unsupported statement type for planning: {:?}",
                std::mem::discriminant(&other)
            ))),
        }
    }

    // -----------------------------------------------------------------------
    // SELECT binding
    // -----------------------------------------------------------------------

    fn bind_select<'b>(
        &'b mut self,
        ctx: &'b mut BindContext,
        stmt: &'b SelectStatement,
    ) -> std::pin::Pin<Box<dyn std::future::Future<Output = Result<BoundSelect>> + 'b>> {
        Box::pin(async move {
            // Bind CTEs first
            let mut bound_ctes = Vec::new();
            if let Some(with) = &stmt.with {
                for cte in &with.ctes {
                    let mut cte_ctx = BindContext::new();
                    let cte_query = self.bind_select(&mut cte_ctx, &cte.query).await?;
                    let cte_columns = cte_query.output_schema.clone();
                    let bound_cte = BoundCte {
                        name: cte.name.clone(),
                        columns: cte_columns,
                        query: Box::new(cte_query),
                    };
                    ctx.ctes.insert(cte.name.clone(), bound_cte.clone());
                    bound_ctes.push(bound_cte);
                }
            }

            // Bind FROM clause
            let from = self.bind_from(ctx, &stmt.from).await?;

            // Bind WHERE
            let where_clause = if let Some(expr) = &stmt.where_clause {
                Some(self.bind_expr(ctx, expr).await?)
            } else {
                None
            };

            // Bind GROUP BY
            let mut group_by = Vec::with_capacity(stmt.group_by.len());
            for e in &stmt.group_by {
                group_by.push(self.bind_expr(ctx, e).await?);
            }

            // Bind HAVING
            let having = if let Some(expr) = &stmt.having {
                Some(self.bind_expr(ctx, expr).await?)
            } else {
                None
            };

            // Bind projections
            let projections = self.bind_select_items(ctx, &stmt.projections).await?;

            // Build output schema from projections
            let output_schema = self.build_output_schema(ctx, &projections);

            // Bind ORDER BY
            let mut order_by = Vec::with_capacity(stmt.order_by.len());
            for o in &stmt.order_by {
                order_by.push(self.bind_order_by(ctx, o).await?);
            }

            // Bind LIMIT / OFFSET
            let limit = if let Some(expr) = &stmt.limit {
                Some(self.bind_expr(ctx, expr).await?)
            } else {
                None
            };
            let offset = if let Some(expr) = &stmt.offset {
                Some(self.bind_expr(ctx, expr).await?)
            } else {
                None
            };

            // Bind set operations
            let mut set_ops = Vec::new();
            for set_op in &stmt.set_ops {
                let mut right_ctx = BindContext::new();
                let right = self.bind_select(&mut right_ctx, &set_op.right).await?;
                set_ops.push(BoundSetOp {
                    op: set_op.op,
                    all: set_op.all,
                    right: Box::new(right),
                });
            }

            Ok(BoundSelect {
                projections,
                from,
                where_clause,
                group_by,
                having,
                order_by,
                limit,
                offset,
                distinct: stmt.distinct,
                set_ops,
                ctes: bound_ctes,
                output_schema,
            })
        }) // end Box::pin
    }

    // -----------------------------------------------------------------------
    // FROM clause binding
    // -----------------------------------------------------------------------

    async fn bind_from(
        &mut self,
        ctx: &mut BindContext,
        from: &[TableRef],
    ) -> Result<Vec<BoundFromItem>> {
        let mut items = Vec::with_capacity(from.len());
        for table_ref in from {
            let item = self.bind_table_ref(ctx, table_ref).await?;
            items.push(item);
        }
        Ok(items)
    }

    fn bind_table_ref<'b>(
        &'b mut self,
        ctx: &'b mut BindContext,
        table_ref: &'b TableRef,
    ) -> std::pin::Pin<Box<dyn std::future::Future<Output = Result<BoundFromItem>> + 'b>> {
        Box::pin(async move {
            match table_ref {
                TableRef::Table { name, alias, .. } => {
                    // Check if this is a CTE reference
                    let display_name = alias.as_deref().unwrap_or(name);
                    if let Some(cte) = ctx.ctes.get(name).cloned() {
                        let idx = self.alloc_table_idx();
                        let bound_table = BoundTableRef {
                            table_idx: idx,
                            table_id: None,
                            alias: display_name.to_string(),
                            columns: cte.columns.clone(),
                            entry: None,
                        };
                        ctx.tables.push(bound_table);
                        return Ok(BoundFromItem::Subquery {
                            table_idx: idx,
                            query: cte.query.clone(),
                        });
                    }

                    // Parse optional schema qualifier (schema.table)
                    let (schema_name, table_name) = if let Some(dot_pos) = name.find('.') {
                        (Some(&name[..dot_pos]), &name[dot_pos + 1..])
                    } else {
                        (None, name.as_str())
                    };

                    let entry = self.resolver.resolve_table(schema_name, table_name).await?;
                    let idx = self.alloc_table_idx();

                    let columns: Vec<BoundColumnDef> = entry
                        .columns
                        .iter()
                        .map(|c| BoundColumnDef {
                            column_id: c.id,
                            name: c.name.clone(),
                            type_id: c.type_id,
                            nullable: c.nullable,
                            ordinal: c.ordinal,
                        })
                        .collect();

                    let bound_table = BoundTableRef {
                        table_idx: idx,
                        table_id: Some(entry.id),
                        alias: display_name.to_string(),
                        columns,
                        entry: Some(Arc::clone(&entry)),
                    };
                    ctx.tables.push(bound_table);

                    Ok(BoundFromItem::BaseTable {
                        table_idx: idx,
                        table_id: entry.id,
                        entry,
                    })
                }
                TableRef::Join(join_ref) => {
                    let left = self.bind_table_ref(ctx, &join_ref.left).await?;
                    let right = self.bind_table_ref(ctx, &join_ref.right).await?;
                    let condition = self.bind_join_condition(ctx, &join_ref.condition).await?;
                    Ok(BoundFromItem::Join {
                        left: Box::new(left),
                        join_type: join_ref.join_type,
                        right: Box::new(right),
                        condition,
                    })
                }
                TableRef::Subquery { query, alias } => {
                    let mut sub_ctx = BindContext::new();
                    let bound_query = self.bind_select(&mut sub_ctx, query).await?;
                    let idx = self.alloc_table_idx();
                    let columns = bound_query.output_schema.clone();
                    let bound_table = BoundTableRef {
                        table_idx: idx,
                        table_id: None,
                        alias: alias.clone(),
                        columns,
                        entry: None,
                    };
                    ctx.tables.push(bound_table);
                    Ok(BoundFromItem::Subquery {
                        table_idx: idx,
                        query: Box::new(bound_query),
                    })
                }
                TableRef::Lateral { subquery } => {
                    // Treat LATERAL as a regular subquery for now
                    self.bind_table_ref(ctx, subquery).await
                }
            }
        }) // end Box::pin
    }

    async fn bind_join_condition(
        &mut self,
        ctx: &BindContext,
        condition: &JoinCondition,
    ) -> Result<BoundJoinCondition> {
        match condition {
            JoinCondition::On(expr) => {
                let bound = self.bind_expr(ctx, expr).await?;
                Ok(BoundJoinCondition::On(bound))
            }
            JoinCondition::Using(cols) => {
                // Resolve column names to IDs from the last two tables in context
                let mut ids = Vec::with_capacity(cols.len());
                for col_name in cols {
                    // Find the column in any of the tables in scope
                    let mut found = false;
                    for table in &ctx.tables {
                        if let Some(c) = table.columns.iter().find(|c| c.name == *col_name) {
                            ids.push(c.column_id);
                            found = true;
                            break;
                        }
                    }
                    if !found {
                        return Err(ZyronError::ColumnNotFound(col_name.clone()));
                    }
                }
                Ok(BoundJoinCondition::Using(ids))
            }
            JoinCondition::Natural => Ok(BoundJoinCondition::Natural),
            JoinCondition::None => Ok(BoundJoinCondition::None),
        }
    }

    // -----------------------------------------------------------------------
    // Expression binding
    // -----------------------------------------------------------------------

    fn bind_expr<'b>(
        &'b mut self,
        ctx: &'b BindContext,
        expr: &'b Expr,
    ) -> std::pin::Pin<Box<dyn std::future::Future<Output = Result<BoundExpr>> + 'b>> {
        Box::pin(async move {
            match expr {
                Expr::Identifier(name) => {
                    let cr = self.resolve_column(ctx, name)?;
                    Ok(BoundExpr::ColumnRef(cr))
                }
                Expr::QualifiedIdentifier { table, column } => {
                    let cr = self.resolve_qualified_column(ctx, table, column)?;
                    Ok(BoundExpr::ColumnRef(cr))
                }
                Expr::Literal(lit) => {
                    let type_id = literal_type(lit);
                    Ok(BoundExpr::Literal {
                        value: lit.clone(),
                        type_id,
                    })
                }
                Expr::BinaryOp { left, op, right } => {
                    let left_bound = self.bind_expr(ctx, left).await?;
                    let right_bound = self.bind_expr(ctx, right).await?;
                    let type_id =
                        infer_binary_type(op, left_bound.type_id(), right_bound.type_id())?;
                    Ok(BoundExpr::BinaryOp {
                        left: Box::new(left_bound),
                        op: *op,
                        right: Box::new(right_bound),
                        type_id,
                    })
                }
                Expr::UnaryOp { op, expr: inner } => {
                    let bound = self.bind_expr(ctx, inner).await?;
                    let type_id = match op {
                        UnaryOperator::Not => TypeId::Boolean,
                        UnaryOperator::Minus => bound.type_id(),
                    };
                    Ok(BoundExpr::UnaryOp {
                        op: *op,
                        expr: Box::new(bound),
                        type_id,
                    })
                }
                Expr::IsNull {
                    expr: inner,
                    negated,
                } => {
                    let bound = self.bind_expr(ctx, inner).await?;
                    Ok(BoundExpr::IsNull {
                        expr: Box::new(bound),
                        negated: *negated,
                    })
                }
                Expr::InList {
                    expr: inner,
                    list,
                    negated,
                } => {
                    let bound_expr = self.bind_expr(ctx, inner).await?;
                    let mut bound_list = Vec::with_capacity(list.len());
                    for e in list {
                        bound_list.push(self.bind_expr(ctx, e).await?);
                    }
                    Ok(BoundExpr::InList {
                        expr: Box::new(bound_expr),
                        list: bound_list,
                        negated: *negated,
                    })
                }
                Expr::Between {
                    expr: inner,
                    low,
                    high,
                    negated,
                } => {
                    let bound_expr = self.bind_expr(ctx, inner).await?;
                    let bound_low = self.bind_expr(ctx, low).await?;
                    let bound_high = self.bind_expr(ctx, high).await?;
                    Ok(BoundExpr::Between {
                        expr: Box::new(bound_expr),
                        low: Box::new(bound_low),
                        high: Box::new(bound_high),
                        negated: *negated,
                    })
                }
                Expr::Like {
                    expr: inner,
                    pattern,
                    negated,
                } => {
                    let bound_expr = self.bind_expr(ctx, inner).await?;
                    let bound_pattern = self.bind_expr(ctx, pattern).await?;
                    Ok(BoundExpr::Like {
                        expr: Box::new(bound_expr),
                        pattern: Box::new(bound_pattern),
                        negated: *negated,
                    })
                }
                Expr::ILike {
                    expr: inner,
                    pattern,
                    negated,
                } => {
                    let bound_expr = self.bind_expr(ctx, inner).await?;
                    let bound_pattern = self.bind_expr(ctx, pattern).await?;
                    Ok(BoundExpr::ILike {
                        expr: Box::new(bound_expr),
                        pattern: Box::new(bound_pattern),
                        negated: *negated,
                    })
                }
                Expr::Function {
                    name,
                    args,
                    distinct,
                } => {
                    let mut bound_args = Vec::with_capacity(args.len());
                    for a in args {
                        let e = match a {
                            FunctionArg::Unnamed(e) => e,
                            FunctionArg::Named { value, .. } => value,
                        };
                        bound_args.push(self.bind_expr(ctx, e).await?);
                    }

                    let arg_types: Vec<TypeId> = bound_args.iter().map(|a| a.type_id()).collect();

                    if is_aggregate_function(name) {
                        let return_type = infer_aggregate_type(name, &arg_types)?;
                        Ok(BoundExpr::AggregateFunction {
                            name: name.to_lowercase(),
                            args: bound_args,
                            distinct: *distinct,
                            return_type,
                        })
                    } else {
                        let return_type = infer_function_type(name, &arg_types);
                        Ok(BoundExpr::Function {
                            name: name.clone(),
                            args: bound_args,
                            return_type,
                            distinct: *distinct,
                        })
                    }
                }
                Expr::Cast {
                    expr: inner,
                    data_type,
                } => {
                    let bound = self.bind_expr(ctx, inner).await?;
                    let target_type = data_type.to_type_id();
                    Ok(BoundExpr::Cast {
                        expr: Box::new(bound),
                        target_type,
                    })
                }
                Expr::Case {
                    operand,
                    conditions,
                    else_result,
                } => {
                    let bound_operand = if let Some(e) = operand.as_ref() {
                        Some(self.bind_expr(ctx, e).await?)
                    } else {
                        None
                    };
                    let mut bound_conditions = Vec::with_capacity(conditions.len());
                    for wc in conditions {
                        bound_conditions.push(BoundWhen {
                            condition: self.bind_expr(ctx, &wc.condition).await?,
                            result: self.bind_expr(ctx, &wc.result).await?,
                        });
                    }
                    let bound_else = if let Some(e) = else_result.as_ref() {
                        Some(self.bind_expr(ctx, e).await?)
                    } else {
                        None
                    };

                    // Result type is the type of the first THEN clause
                    let type_id = bound_conditions
                        .first()
                        .map(|w| w.result.type_id())
                        .unwrap_or(TypeId::Null);

                    Ok(BoundExpr::Case {
                        operand: bound_operand.map(Box::new),
                        conditions: bound_conditions,
                        else_result: bound_else.map(Box::new),
                        type_id,
                    })
                }
                Expr::Nested(inner) => {
                    let bound = self.bind_expr(ctx, inner).await?;
                    Ok(BoundExpr::Nested(Box::new(bound)))
                }
                Expr::Subquery(query) => {
                    let mut sub_ctx = BindContext::new();
                    // Copy outer scope for correlated subquery support
                    for table in &ctx.tables {
                        sub_ctx.tables.push(table.clone());
                    }
                    let bound = Box::pin(self.bind_select(&mut sub_ctx, query)).await?;
                    let type_id = bound
                        .output_schema
                        .first()
                        .map(|c| c.type_id)
                        .unwrap_or(TypeId::Null);
                    Ok(BoundExpr::Subquery {
                        plan: Box::new(bound),
                        type_id,
                    })
                }
                Expr::Exists { query, negated } => {
                    let mut sub_ctx = BindContext::new();
                    for table in &ctx.tables {
                        sub_ctx.tables.push(table.clone());
                    }
                    let bound = Box::pin(self.bind_select(&mut sub_ctx, query)).await?;
                    Ok(BoundExpr::Exists {
                        plan: Box::new(bound),
                        negated: *negated,
                    })
                }
                Expr::InSubquery {
                    expr: inner,
                    query,
                    negated,
                } => {
                    let bound_expr = self.bind_expr(ctx, inner).await?;
                    let mut sub_ctx = BindContext::new();
                    for table in &ctx.tables {
                        sub_ctx.tables.push(table.clone());
                    }
                    let bound_query = Box::pin(self.bind_select(&mut sub_ctx, query)).await?;
                    Ok(BoundExpr::InSubquery {
                        expr: Box::new(bound_expr),
                        plan: Box::new(bound_query),
                        negated: *negated,
                    })
                }
                Expr::WindowFunction {
                    function,
                    partition_by,
                    order_by,
                    frame,
                } => {
                    let bound_func = self.bind_expr(ctx, function).await?;
                    let mut bound_partition = Vec::with_capacity(partition_by.len());
                    for e in partition_by {
                        bound_partition.push(self.bind_expr(ctx, e).await?);
                    }
                    let mut bound_order = Vec::with_capacity(order_by.len());
                    for o in order_by {
                        bound_order.push(self.bind_order_by(ctx, o).await?);
                    }
                    let type_id = bound_func.type_id();
                    Ok(BoundExpr::WindowFunction {
                        function: Box::new(bound_func),
                        partition_by: bound_partition,
                        order_by: bound_order,
                        frame: frame.clone(),
                        type_id,
                    })
                }
                Expr::Parameter(idx) => Ok(BoundExpr::Parameter {
                    index: *idx,
                    type_id: TypeId::Null, // Resolved at execution time
                }),
                // JSON, array, vector operators: bind as generic binary ops for now
                Expr::JsonAccess { left, right, .. }
                | Expr::JsonContains { left, right, .. }
                | Expr::JsonExists { left, right, .. }
                | Expr::VectorDistance { left, right, .. } => {
                    let bound_left = self.bind_expr(ctx, left).await?;
                    let _bound_right = self.bind_expr(ctx, right).await?;
                    Ok(BoundExpr::Function {
                        name: "json_op".to_string(),
                        args: vec![bound_left, _bound_right],
                        return_type: TypeId::Jsonb,
                        distinct: false,
                    })
                }
                Expr::ArrayConstructor(elements) => {
                    let mut bound_elements = Vec::with_capacity(elements.len());
                    for e in elements {
                        bound_elements.push(self.bind_expr(ctx, e).await?);
                    }
                    Ok(BoundExpr::Function {
                        name: "array".to_string(),
                        args: bound_elements,
                        return_type: TypeId::Array,
                        distinct: false,
                    })
                }
                Expr::ArraySubscript { array, index } => {
                    let bound_array = self.bind_expr(ctx, array).await?;
                    let bound_index = self.bind_expr(ctx, index).await?;
                    Ok(BoundExpr::Function {
                        name: "array_subscript".to_string(),
                        args: vec![bound_array, bound_index],
                        return_type: TypeId::Null, // Element type unknown without full type system
                        distinct: false,
                    })
                }
                Expr::AnySubquery { query } | Expr::AllSubquery { query } => {
                    let mut sub_ctx = BindContext::new();
                    let bound_query = Box::pin(self.bind_select(&mut sub_ctx, query)).await?;
                    let type_id = bound_query
                        .output_schema
                        .first()
                        .map(|c| c.type_id)
                        .unwrap_or(TypeId::Null);
                    Ok(BoundExpr::Subquery {
                        plan: Box::new(bound_query),
                        type_id,
                    })
                }
                Expr::MatchAgainst { columns, query, .. } => {
                    // Bind as a function call with the match columns and query
                    let bound_query = self.bind_expr(ctx, query).await?;
                    let mut args = Vec::with_capacity(columns.len() + 1);
                    for col_name in columns {
                        let cr = self.resolve_column(ctx, col_name)?;
                        args.push(BoundExpr::ColumnRef(cr));
                    }
                    args.push(bound_query);
                    Ok(BoundExpr::Function {
                        name: "match_against".to_string(),
                        args,
                        return_type: TypeId::Float64,
                        distinct: false,
                    })
                }
            }
        }) // end Box::pin for bind_expr
    }

    // -----------------------------------------------------------------------
    // Column resolution
    // -----------------------------------------------------------------------

    /// Resolves an unqualified column name by searching all tables in scope.
    fn resolve_column(&self, ctx: &BindContext, name: &str) -> Result<ColumnRef> {
        let mut found: Option<ColumnRef> = None;
        for table in &ctx.tables {
            for col in &table.columns {
                if col.name == name {
                    if found.is_some() {
                        return Err(ZyronError::PlanError(format!(
                            "ambiguous column reference: {}",
                            name
                        )));
                    }
                    found = Some(ColumnRef {
                        table_idx: table.table_idx,
                        column_id: col.column_id,
                        type_id: col.type_id,
                        nullable: col.nullable,
                    });
                }
            }
        }

        // Check outer scope for correlated references
        if found.is_none() {
            if let Some(outer) = &ctx.outer {
                return self.resolve_column(outer, name);
            }
        }

        found.ok_or_else(|| ZyronError::ColumnNotFound(name.to_string()))
    }

    /// Resolves a qualified column reference (table.column).
    fn resolve_qualified_column(
        &self,
        ctx: &BindContext,
        table: &str,
        column: &str,
    ) -> Result<ColumnRef> {
        for bound_table in &ctx.tables {
            if bound_table.alias == table {
                for col in &bound_table.columns {
                    if col.name == column {
                        return Ok(ColumnRef {
                            table_idx: bound_table.table_idx,
                            column_id: col.column_id,
                            type_id: col.type_id,
                            nullable: col.nullable,
                        });
                    }
                }
                return Err(ZyronError::ColumnNotFound(format!("{}.{}", table, column)));
            }
        }

        // Check outer scope
        if let Some(outer) = &ctx.outer {
            return self.resolve_qualified_column(outer, table, column);
        }

        Err(ZyronError::TableNotFound(table.to_string()))
    }

    // -----------------------------------------------------------------------
    // SELECT item binding
    // -----------------------------------------------------------------------

    async fn bind_select_items(
        &mut self,
        ctx: &BindContext,
        items: &[SelectItem],
    ) -> Result<Vec<BoundSelectItem>> {
        let mut result = Vec::with_capacity(items.len());
        for item in items {
            match item {
                SelectItem::Expr(expr, alias) => {
                    let bound = self.bind_expr(ctx, expr).await?;
                    result.push(BoundSelectItem::Expr(bound, alias.clone()));
                }
                SelectItem::Wildcard => {
                    result.push(BoundSelectItem::Wildcard);
                }
                SelectItem::QualifiedWildcard(table_name) => {
                    if let Some(table) = ctx.tables.iter().find(|t| t.alias == *table_name) {
                        result.push(BoundSelectItem::AllColumns(table.table_idx));
                    } else {
                        return Err(ZyronError::TableNotFound(table_name.clone()));
                    }
                }
            }
        }
        Ok(result)
    }

    // -----------------------------------------------------------------------
    // Output schema construction
    // -----------------------------------------------------------------------

    fn build_output_schema(
        &self,
        ctx: &BindContext,
        projections: &[BoundSelectItem],
    ) -> Vec<BoundColumnDef> {
        let mut schema = Vec::new();
        for (ordinal, item) in projections.iter().enumerate() {
            match item {
                BoundSelectItem::Expr(expr, alias) => {
                    let name = alias.clone().unwrap_or_else(|| {
                        if let BoundExpr::ColumnRef(cr) = expr {
                            // Find column name from context
                            for table in &ctx.tables {
                                if table.table_idx == cr.table_idx {
                                    if let Some(col) =
                                        table.columns.iter().find(|c| c.column_id == cr.column_id)
                                    {
                                        return col.name.clone();
                                    }
                                }
                            }
                            format!("col{}", ordinal)
                        } else {
                            format!("col{}", ordinal)
                        }
                    });
                    schema.push(BoundColumnDef {
                        column_id: ColumnId(ordinal as u16),
                        name,
                        type_id: expr.type_id(),
                        nullable: expr.nullable(),
                        ordinal: ordinal as u16,
                    });
                }
                BoundSelectItem::Wildcard => {
                    for table in &ctx.tables {
                        for col in &table.columns {
                            schema.push(col.clone());
                        }
                    }
                }
                BoundSelectItem::AllColumns(table_idx) => {
                    if let Some(table) = ctx.tables.iter().find(|t| t.table_idx == *table_idx) {
                        for col in &table.columns {
                            schema.push(col.clone());
                        }
                    }
                }
            }
        }
        schema
    }

    // -----------------------------------------------------------------------
    // ORDER BY binding
    // -----------------------------------------------------------------------

    async fn bind_order_by(
        &mut self,
        ctx: &BindContext,
        order: &OrderByExpr,
    ) -> Result<BoundOrderBy> {
        let bound_expr = self.bind_expr(ctx, &order.expr).await?;
        Ok(BoundOrderBy {
            expr: bound_expr,
            asc: order.asc.unwrap_or(true),
            nulls_first: order.nulls_first.unwrap_or(false),
        })
    }

    // -----------------------------------------------------------------------
    // INSERT binding
    // -----------------------------------------------------------------------

    async fn bind_insert(&mut self, stmt: &InsertStatement) -> Result<BoundInsert> {
        let (schema_name, table_name) = if let Some(dot_pos) = stmt.table.find('.') {
            (Some(&stmt.table[..dot_pos]), &stmt.table[dot_pos + 1..])
        } else {
            (None, stmt.table.as_str())
        };

        let entry = self.resolver.resolve_table(schema_name, table_name).await?;

        // Resolve target columns
        let target_columns = if stmt.columns.is_empty() {
            // All columns in table order
            entry.columns.iter().map(|c| c.id).collect()
        } else {
            let mut ids = Vec::with_capacity(stmt.columns.len());
            for col_name in &stmt.columns {
                let col = self.resolver.resolve_column(&entry, col_name)?;
                ids.push(col.id);
            }
            ids
        };

        // Register table in a temporary context for RETURNING
        let mut ctx = BindContext::new();
        let idx = self.alloc_table_idx();
        let columns: Vec<BoundColumnDef> = entry
            .columns
            .iter()
            .map(|c| BoundColumnDef {
                column_id: c.id,
                name: c.name.clone(),
                type_id: c.type_id,
                nullable: c.nullable,
                ordinal: c.ordinal,
            })
            .collect();
        ctx.tables.push(BoundTableRef {
            table_idx: idx,
            table_id: Some(entry.id),
            alias: table_name.to_string(),
            columns,
            entry: Some(Arc::clone(&entry)),
        });

        let source = match &stmt.source {
            InsertSource::Values(rows) => {
                let mut bound_rows = Vec::with_capacity(rows.len());
                for row in rows {
                    let mut bound_row = Vec::with_capacity(row.len());
                    for e in row {
                        bound_row.push(self.bind_expr(&ctx, e).await?);
                    }
                    bound_rows.push(bound_row);
                }
                BoundInsertSource::Values(bound_rows)
            }
            InsertSource::Query(query) => {
                let mut sub_ctx = BindContext::new();
                let bound_query = self.bind_select(&mut sub_ctx, query).await?;
                BoundInsertSource::Query(Box::new(bound_query))
            }
        };

        let returning = if let Some(items) = &stmt.returning {
            Some(self.bind_select_items(&ctx, items).await?)
        } else {
            None
        };

        Ok(BoundInsert {
            table_id: entry.id,
            table_entry: entry,
            target_columns,
            source,
            returning,
        })
    }

    // -----------------------------------------------------------------------
    // UPDATE binding
    // -----------------------------------------------------------------------

    async fn bind_update(&mut self, stmt: &UpdateStatement) -> Result<BoundUpdate> {
        let (schema_name, table_name) = if let Some(dot_pos) = stmt.table.find('.') {
            (Some(&stmt.table[..dot_pos]), &stmt.table[dot_pos + 1..])
        } else {
            (None, stmt.table.as_str())
        };

        let entry = self.resolver.resolve_table(schema_name, table_name).await?;

        // Register table in context
        let mut ctx = BindContext::new();
        let idx = self.alloc_table_idx();
        let columns: Vec<BoundColumnDef> = entry
            .columns
            .iter()
            .map(|c| BoundColumnDef {
                column_id: c.id,
                name: c.name.clone(),
                type_id: c.type_id,
                nullable: c.nullable,
                ordinal: c.ordinal,
            })
            .collect();
        ctx.tables.push(BoundTableRef {
            table_idx: idx,
            table_id: Some(entry.id),
            alias: table_name.to_string(),
            columns,
            entry: Some(Arc::clone(&entry)),
        });

        let mut assignments = Vec::with_capacity(stmt.assignments.len());
        for a in &stmt.assignments {
            let col = self.resolver.resolve_column(&entry, &a.column)?;
            let value = self.bind_expr(&ctx, &a.value).await?;
            assignments.push(BoundAssignment {
                column_id: col.id,
                value,
            });
        }

        let where_clause = if let Some(expr) = &stmt.where_clause {
            Some(self.bind_expr(&ctx, expr).await?)
        } else {
            None
        };

        let returning = if let Some(items) = &stmt.returning {
            Some(self.bind_select_items(&ctx, items).await?)
        } else {
            None
        };

        Ok(BoundUpdate {
            table_id: entry.id,
            table_entry: entry,
            assignments,
            where_clause,
            returning,
        })
    }

    // -----------------------------------------------------------------------
    // DELETE binding
    // -----------------------------------------------------------------------

    async fn bind_delete(&mut self, stmt: &DeleteStatement) -> Result<BoundDelete> {
        let (schema_name, table_name) = if let Some(dot_pos) = stmt.table.find('.') {
            (Some(&stmt.table[..dot_pos]), &stmt.table[dot_pos + 1..])
        } else {
            (None, stmt.table.as_str())
        };

        let entry = self.resolver.resolve_table(schema_name, table_name).await?;

        // Register table in context
        let mut ctx = BindContext::new();
        let idx = self.alloc_table_idx();
        let columns: Vec<BoundColumnDef> = entry
            .columns
            .iter()
            .map(|c| BoundColumnDef {
                column_id: c.id,
                name: c.name.clone(),
                type_id: c.type_id,
                nullable: c.nullable,
                ordinal: c.ordinal,
            })
            .collect();
        ctx.tables.push(BoundTableRef {
            table_idx: idx,
            table_id: Some(entry.id),
            alias: table_name.to_string(),
            columns,
            entry: Some(Arc::clone(&entry)),
        });

        let where_clause = if let Some(expr) = &stmt.where_clause {
            Some(self.bind_expr(&ctx, expr).await?)
        } else {
            None
        };

        let returning = if let Some(items) = &stmt.returning {
            Some(self.bind_select_items(&ctx, items).await?)
        } else {
            None
        };

        Ok(BoundDelete {
            table_id: entry.id,
            table_entry: entry,
            where_clause,
            returning,
        })
    }
}

// ---------------------------------------------------------------------------
// Type inference helpers
// ---------------------------------------------------------------------------

/// Infers the TypeId for a literal value.
fn literal_type(lit: &LiteralValue) -> TypeId {
    match lit {
        LiteralValue::Integer(_) => TypeId::Int64,
        LiteralValue::Float(_) => TypeId::Float64,
        LiteralValue::String(_) => TypeId::Varchar,
        LiteralValue::Boolean(_) => TypeId::Boolean,
        LiteralValue::Null => TypeId::Null,
    }
}

/// Infers the result type of a binary operation.
fn infer_binary_type(op: &BinaryOperator, left: TypeId, right: TypeId) -> Result<TypeId> {
    match op {
        // Comparison operators always produce Boolean
        BinaryOperator::Eq
        | BinaryOperator::Neq
        | BinaryOperator::Lt
        | BinaryOperator::Gt
        | BinaryOperator::LtEq
        | BinaryOperator::GtEq => Ok(TypeId::Boolean),

        // Logical operators produce Boolean
        BinaryOperator::And | BinaryOperator::Or => Ok(TypeId::Boolean),

        // Concatenation produces Varchar
        BinaryOperator::Concat => Ok(TypeId::Varchar),

        // Arithmetic operators: promote to the wider numeric type
        BinaryOperator::Plus
        | BinaryOperator::Minus
        | BinaryOperator::Multiply
        | BinaryOperator::Divide
        | BinaryOperator::Modulo => Ok(promote_numeric(left, right)),
    }
}

/// Promotes two numeric types to their common supertype.
fn promote_numeric(left: TypeId, right: TypeId) -> TypeId {
    if left == right {
        return left;
    }

    // If either side is NULL, use the other
    if left == TypeId::Null {
        return right;
    }
    if right == TypeId::Null {
        return left;
    }

    // Float64 absorbs everything
    if left == TypeId::Float64 || right == TypeId::Float64 {
        return TypeId::Float64;
    }
    if left == TypeId::Float32 || right == TypeId::Float32 {
        return TypeId::Float64;
    }
    if left == TypeId::Decimal || right == TypeId::Decimal {
        return TypeId::Decimal;
    }

    // Integer promotion: pick the wider type
    let left_rank = integer_rank(left);
    let right_rank = integer_rank(right);
    if left_rank >= right_rank { left } else { right }
}

fn integer_rank(t: TypeId) -> u8 {
    match t {
        TypeId::Int8 | TypeId::UInt8 => 1,
        TypeId::Int16 | TypeId::UInt16 => 2,
        TypeId::Int32 | TypeId::UInt32 => 3,
        TypeId::Int64 | TypeId::UInt64 => 4,
        TypeId::Int128 | TypeId::UInt128 => 5,
        _ => 4, // Default to Int64 rank for non-integer types
    }
}

/// Infers the return type of an aggregate function.
fn infer_aggregate_type(name: &str, arg_types: &[TypeId]) -> Result<TypeId> {
    let lower = name.to_lowercase();
    match lower.as_str() {
        "count" => Ok(TypeId::Int64),
        "sum" => {
            if let Some(&t) = arg_types.first() {
                if t.is_floating_point() {
                    Ok(TypeId::Float64)
                } else if t == TypeId::Decimal {
                    Ok(TypeId::Decimal)
                } else {
                    Ok(TypeId::Int64)
                }
            } else {
                Ok(TypeId::Int64)
            }
        }
        "avg" => Ok(TypeId::Float64),
        "min" | "max" => {
            if let Some(&t) = arg_types.first() {
                Ok(t)
            } else {
                Ok(TypeId::Null)
            }
        }
        _ => Err(ZyronError::PlanError(format!(
            "unknown aggregate function: {}",
            name
        ))),
    }
}

/// Infers the return type of a scalar function.
/// Returns Null for unknown functions (resolved at execution time).
fn infer_function_type(name: &str, arg_types: &[TypeId]) -> TypeId {
    let lower = name.to_lowercase();
    match lower.as_str() {
        "abs" | "ceil" | "ceiling" | "floor" | "round" | "trunc" | "truncate" => {
            arg_types.first().copied().unwrap_or(TypeId::Float64)
        }
        "length" | "char_length" | "character_length" | "octet_length" => TypeId::Int64,
        "lower" | "upper" | "trim" | "ltrim" | "rtrim" | "substring" | "replace" | "concat" => {
            TypeId::Varchar
        }
        "now" | "current_timestamp" => TypeId::TimestampTz,
        "current_date" => TypeId::Date,
        "current_time" => TypeId::Time,
        "coalesce" => arg_types.first().copied().unwrap_or(TypeId::Null),
        "nullif" => arg_types.first().copied().unwrap_or(TypeId::Null),
        "greatest" | "least" => arg_types.first().copied().unwrap_or(TypeId::Null),
        _ => arg_types.first().copied().unwrap_or(TypeId::Null),
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_literal_type_inference() {
        assert_eq!(literal_type(&LiteralValue::Integer(42)), TypeId::Int64);
        assert_eq!(literal_type(&LiteralValue::Float(3.14)), TypeId::Float64);
        assert_eq!(
            literal_type(&LiteralValue::String("hello".to_string())),
            TypeId::Varchar
        );
        assert_eq!(literal_type(&LiteralValue::Boolean(true)), TypeId::Boolean);
        assert_eq!(literal_type(&LiteralValue::Null), TypeId::Null);
    }

    #[test]
    fn test_binary_type_inference() {
        // Comparisons produce Boolean
        assert_eq!(
            infer_binary_type(&BinaryOperator::Eq, TypeId::Int64, TypeId::Int64).unwrap(),
            TypeId::Boolean
        );
        assert_eq!(
            infer_binary_type(&BinaryOperator::Lt, TypeId::Float64, TypeId::Int32).unwrap(),
            TypeId::Boolean
        );

        // Arithmetic promotes types
        assert_eq!(
            infer_binary_type(&BinaryOperator::Plus, TypeId::Int32, TypeId::Int64).unwrap(),
            TypeId::Int64
        );
        assert_eq!(
            infer_binary_type(&BinaryOperator::Multiply, TypeId::Int64, TypeId::Float64).unwrap(),
            TypeId::Float64
        );

        // Concatenation produces Varchar
        assert_eq!(
            infer_binary_type(&BinaryOperator::Concat, TypeId::Varchar, TypeId::Varchar).unwrap(),
            TypeId::Varchar
        );
    }

    #[test]
    fn test_aggregate_type_inference() {
        assert_eq!(
            infer_aggregate_type("count", &[TypeId::Int32]).unwrap(),
            TypeId::Int64
        );
        assert_eq!(
            infer_aggregate_type("sum", &[TypeId::Int32]).unwrap(),
            TypeId::Int64
        );
        assert_eq!(
            infer_aggregate_type("sum", &[TypeId::Float64]).unwrap(),
            TypeId::Float64
        );
        assert_eq!(
            infer_aggregate_type("avg", &[TypeId::Int32]).unwrap(),
            TypeId::Float64
        );
        assert_eq!(
            infer_aggregate_type("min", &[TypeId::Varchar]).unwrap(),
            TypeId::Varchar
        );
        assert_eq!(
            infer_aggregate_type("max", &[TypeId::Int64]).unwrap(),
            TypeId::Int64
        );
    }

    #[test]
    fn test_numeric_promotion() {
        assert_eq!(promote_numeric(TypeId::Int32, TypeId::Int64), TypeId::Int64);
        assert_eq!(
            promote_numeric(TypeId::Int64, TypeId::Float64),
            TypeId::Float64
        );
        assert_eq!(
            promote_numeric(TypeId::Int32, TypeId::Float32),
            TypeId::Float64
        );
        assert_eq!(promote_numeric(TypeId::Null, TypeId::Int64), TypeId::Int64);
        assert_eq!(promote_numeric(TypeId::Int64, TypeId::Null), TypeId::Int64);
    }

    #[test]
    fn test_is_aggregate_function() {
        assert!(is_aggregate_function("count"));
        assert!(is_aggregate_function("COUNT"));
        assert!(is_aggregate_function("Sum"));
        assert!(is_aggregate_function("avg"));
        assert!(is_aggregate_function("min"));
        assert!(is_aggregate_function("max"));
        assert!(!is_aggregate_function("length"));
        assert!(!is_aggregate_function("lower"));
    }
}
