//! Query planner for Zyron.
//!
//! Converts parsed SQL ASTs into optimized physical execution plans.
//! Pipeline: Parse -> Bind -> Logical Plan -> Optimize -> Physical Plan.

pub mod binder;
pub mod bound_predicate_sql;
pub mod cost;
pub mod explain;
pub mod lake_predicate;
pub mod logical;
pub mod optimizer;
pub mod physical;
pub mod statistics;

pub use binder::{BindContext, Binder, BoundStatement, BoundStreamingJob};
pub use cost::{CostModel, PlanCost};
pub use explain::{
    ACTUAL_AUX_SLOTS, ActualMetrics, ExplainFormat, ExplainNode, ExplainOptions, NodeMetrics,
    aux_labels, millis_parts,
};
pub use logical::LogicalPlan;
pub use optimizer::Optimizer;
pub use physical::PhysicalPlan;

use std::sync::Arc;

use zyron_catalog::{Catalog, DatabaseId, TableEntry};
use zyron_common::Result;
use zyron_parser::Statement;

/// Binds a table's CHECK constraint predicates (against a canonical table_idx of
/// 0) so an executor-internal write path that lacks a bound statement (FK
/// cascade, branch merge) can still enforce them. CHECK predicates reference
/// only the table's own columns, so the resolver scope is irrelevant.
pub async fn bind_table_check_constraints(
    catalog: &Catalog,
    entry: &TableEntry,
) -> Result<Vec<binder::BoundExpr>> {
    let resolver = catalog.resolver(DatabaseId(1), vec!["public".to_string()]);
    let mut binder = Binder::new(resolver, catalog);
    binder.bind_check_constraints(entry).await
}

/// Binds one predicate against a table (at a canonical table_idx of 0), so a
/// maintenance command carrying a WHERE clause evaluates it through the same
/// expression machinery a query would.
pub async fn bind_table_predicate(
    catalog: &Catalog,
    entry: &TableEntry,
    expr: &zyron_parser::ast::Expr,
) -> Result<binder::BoundExpr> {
    let resolver = catalog.resolver(DatabaseId(1), vec!["public".to_string()]);
    let mut binder = Binder::new(resolver, catalog);
    binder.bind_table_predicate(entry, expr).await
}

/// Binds the DEFAULT expression of each named column, so an
/// executor-internal write path with no bound statement can fill a column
/// the way INSERT does. A column with no default yields a NULL literal of
/// its own type, which is what SET DEFAULT means for it.
///
/// A default is a constant or a volatile call with no column references, so
/// it binds in an empty scope. One that fails to parse or bind is a real
/// catalog error and is reported rather than quietly becoming NULL.
pub async fn bind_column_defaults(
    catalog: &Catalog,
    entry: &TableEntry,
    columns: &[zyron_catalog::ColumnId],
) -> Result<Vec<(zyron_catalog::ColumnId, binder::BoundExpr)>> {
    let resolver = catalog.resolver(DatabaseId(1), vec!["public".to_string()]);
    let mut binder = Binder::new(resolver, catalog);
    binder.bind_column_defaults(entry, columns).await
}

/// One row-security predicate for a table. `permissive` predicates within a
/// table are OR'd together then AND'd with the user filter; non-permissive
/// (restrictive) predicates are AND'd.
#[derive(Debug, Clone)]
pub struct RowPredicate {
    pub sql: String,
    pub permissive: bool,
}

/// Supplies RLS / ABAC / row-ownership predicates for a table given the
/// session's role context. Implemented by the connection layer over the
/// SecurityManager so zyron-planner does not depend on zyron-auth.
pub trait RowSecurityProvider: Send + Sync {
    /// Predicates to enforce for `table_id`. Empty means no row security.
    fn row_predicates(&self, table_id: u32) -> Vec<RowPredicate>;
    /// True when `table_id` has any row-security policy. Used to fail closed
    /// for query shapes where per-table injection is not performed.
    fn has_row_security(&self, table_id: u32) -> bool;
}

/// Plans a parsed SQL statement into an optimized physical execution plan.
/// Internal/admin path: no row security is injected.
pub async fn plan(
    catalog: &Catalog,
    database_id: DatabaseId,
    search_path: Vec<String>,
    stmt: Statement,
    peers: Option<&zyron_common::PeerRegistry>,
) -> Result<PhysicalPlan> {
    plan_with_security(catalog, database_id, search_path, stmt, None, peers).await
}

/// Plans a statement, injecting RLS/ABAC/row-ownership predicates from the
/// provider (user-facing query path).
pub async fn plan_with_security(
    catalog: &Catalog,
    database_id: DatabaseId,
    search_path: Vec<String>,
    stmt: Statement,
    security: Option<Arc<dyn RowSecurityProvider>>,
    peers: Option<&zyron_common::PeerRegistry>,
) -> Result<PhysicalPlan> {
    let resolver = catalog.resolver(database_id, search_path);
    let mut binder = Binder::new(resolver, catalog);
    if let Some(sec) = security {
        binder.set_row_security(sec);
    }
    let bound = binder.bind(stmt).await?;
    let logical = logical::builder::build_logical_plan(&bound)?;
    let optimized = Optimizer::new(catalog).optimize(logical)?;
    let physical = physical::builder::build_physical_plan(optimized, catalog, peers)?;
    Ok(physical)
}

/// Plans a statement for EXPLAIN output.
/// Returns the physical plan alongside the explain options for rendering.
pub async fn plan_for_explain(
    catalog: &Catalog,
    database_id: DatabaseId,
    search_path: Vec<String>,
    stmt: Statement,
    options: ExplainOptions,
    peers: Option<&zyron_common::PeerRegistry>,
) -> Result<(PhysicalPlan, ExplainOptions)> {
    let resolver = catalog.resolver(database_id, search_path);
    let mut binder = Binder::new(resolver, catalog);
    let bound = binder.bind(stmt).await?;
    let logical = logical::builder::build_logical_plan(&bound)?;
    let optimized = Optimizer::new(catalog).optimize(logical)?;
    let physical = physical::builder::build_physical_plan(optimized, catalog, peers)?;
    Ok((physical, options))
}
