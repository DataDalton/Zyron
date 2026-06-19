//! Query planner for ZyronDB.
//!
//! Converts parsed SQL ASTs into optimized physical execution plans.
//! Pipeline: Parse -> Bind -> Logical Plan -> Optimize -> Physical Plan.

pub mod binder;
pub mod cost;
pub mod explain;
pub mod logical;
pub mod optimizer;
pub mod physical;
pub mod statistics;

pub use binder::{BindContext, Binder, BoundStatement, BoundStreamingJob};
pub use cost::{CostModel, PlanCost};
pub use explain::{ExplainFormat, ExplainNode, ExplainOptions};
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
) -> Result<PhysicalPlan> {
    plan_with_security(catalog, database_id, search_path, stmt, None).await
}

/// Plans a statement, injecting RLS/ABAC/row-ownership predicates from the
/// provider (user-facing query path).
pub async fn plan_with_security(
    catalog: &Catalog,
    database_id: DatabaseId,
    search_path: Vec<String>,
    stmt: Statement,
    security: Option<Arc<dyn RowSecurityProvider>>,
) -> Result<PhysicalPlan> {
    let resolver = catalog.resolver(database_id, search_path);
    let mut binder = Binder::new(resolver, catalog);
    if let Some(sec) = security {
        binder.set_row_security(sec);
    }
    let bound = binder.bind(stmt).await?;
    let logical = logical::builder::build_logical_plan(&bound)?;
    let optimized = Optimizer::new(catalog).optimize(logical)?;
    let physical = physical::builder::build_physical_plan(optimized, catalog)?;
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
) -> Result<(PhysicalPlan, ExplainOptions)> {
    let resolver = catalog.resolver(database_id, search_path);
    let mut binder = Binder::new(resolver, catalog);
    let bound = binder.bind(stmt).await?;
    let logical = logical::builder::build_logical_plan(&bound)?;
    let optimized = Optimizer::new(catalog).optimize(logical)?;
    let physical = physical::builder::build_physical_plan(optimized, catalog)?;
    Ok((physical, options))
}
