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

use zyron_catalog::{Catalog, DatabaseId};
use zyron_common::Result;
use zyron_parser::Statement;

/// Plans a parsed SQL statement into an optimized physical execution plan.
/// This is the main entry point for query planning.
pub async fn plan(
    catalog: &Catalog,
    database_id: DatabaseId,
    search_path: Vec<String>,
    stmt: Statement,
) -> Result<PhysicalPlan> {
    let resolver = catalog.resolver(database_id, search_path);
    let mut binder = Binder::new(resolver, catalog);
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
