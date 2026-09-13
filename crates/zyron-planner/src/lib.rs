//! Query planner for Zyron.
//!
//! Converts parsed SQL ASTs into optimized physical execution plans.
//! Pipeline: Parse -> Bind -> Logical Plan -> Optimize -> Physical Plan.

pub mod binder;
pub mod bound_predicate_sql;
pub mod cluster_expr;
pub mod cost;
pub mod explain;
pub mod lake_predicate;
pub mod logical;
pub mod optimizer;
pub mod physical;
pub mod statistics;
pub mod work_estimate;

pub use binder::{BindContext, Binder, BoundStatement, BoundStreamingJob};
pub use cost::{CostModel, PlanCost};
pub use explain::{
    ACTUAL_AUX_SLOTS, ActualMetrics, ExplainFormat, ExplainNode, ExplainOptions, NodeMetrics,
    aux_labels, millis_parts,
};
pub use logical::LogicalPlan;
pub use optimizer::Optimizer;
pub use optimizer::rules::predicate_pushdown::collect_column_refs;
pub use physical::{ClusterFitDetail, PhysicalPlan};

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
    let resolver = catalog.resolver(DatabaseId(1), zyron_catalog::default_search_path());
    let mut binder = Binder::new(resolver, catalog);
    binder.bind_check_constraints(entry).await
}

/// Binds a table's STORED generated columns, for a write path that builds
/// its operator directly rather than through a planned statement. A
/// referential action rewriting a child row still has to recompute what
/// that row generates.
pub async fn bind_table_generated_columns(
    catalog: &Catalog,
    entry: &TableEntry,
) -> Result<Vec<binder::BoundGeneratedColumn>> {
    let resolver = catalog.resolver(DatabaseId(1), zyron_catalog::default_search_path());
    let mut binder = Binder::new(resolver, catalog);
    binder.bind_generated_columns(entry).await
}

/// Binds one predicate against a table (at a canonical table_idx of 0), so a
/// maintenance command carrying a WHERE clause evaluates it through the same
/// expression machinery a query would.
pub async fn bind_table_predicate(
    catalog: &Catalog,
    entry: &TableEntry,
    expr: &zyron_parser::ast::Expr,
) -> Result<binder::BoundExpr> {
    let resolver = catalog.resolver(DatabaseId(1), zyron_catalog::default_search_path());
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
    let resolver = catalog.resolver(DatabaseId(1), zyron_catalog::default_search_path());
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

pub mod change_scan;

/// What the planner needs to know about a table's change data feed.
///
/// Resolving `LATEST`, refusing a range retention has reclaimed, costing a
/// change scan and telling EXPLAIN how many change files it will open all
/// need the feed itself, which lives in the CDC layer. Implemented there and
/// installed once at startup, the same way the pressure controller and the
/// media presign secret are, so no planning call site has to carry it
pub trait ChangeFeedFacts: Send + Sync {
    /// True when the table records changes right now
    fn feed_enabled(&self, table_id: u32) -> bool;
    /// The oldest and newest commit version the feed still holds
    fn version_range(&self, table_id: u32) -> Option<(u64, u64)>;
    /// The highest commit version retention has reclaimed, zero when none
    fn purge_floor(&self, table_id: u32) -> u64;
    /// The last version at or before a timestamp
    fn version_at_timestamp(&self, table_id: u32, timestamp: i64) -> u64;
    /// Records the window holds, from the feed's per-version counters
    fn rows_in_window(&self, table_id: u32, from_exclusive: u64, to_inclusive: u64) -> u64;
    /// Change files the window opens, and the ones its bounds prune
    fn files_for_window(
        &self,
        table_id: u32,
        from_exclusive: u64,
        to_inclusive: u64,
    ) -> (usize, usize);
    /// The columns the feed records, empty when it records every column
    fn recorded_columns(&self, table_id: u32) -> Vec<u16>;
    /// False when the feed records one row per update rather than two
    fn before_image(&self, table_id: u32) -> bool;
}

/// The change feed facts for one catalog, None before its server installs
/// them.
///
/// Keyed by the directory the catalog's storage writes under, because a
/// process can hold more than one node's catalog. A test binary builds
/// several servers, and each has its own feeds. A single provider would give
/// every one of them the first server's answers.
///
/// A planning path with none resolves a change scan's bounds as written and
/// leaves the resolved range for the operator, which is what an internal plan
/// built before the server finished starting sees
pub fn change_feed_facts_for(catalog: &Catalog) -> Option<Arc<dyn ChangeFeedFacts>> {
    let data_dir = catalog.data_dir()?;
    CHANGE_FEED_FACTS
        .get()?
        .read()
        .ok()?
        .iter()
        .find(|(at, _)| at == data_dir)
        .map(|(_, facts)| Arc::clone(facts))
}

/// Installs the change feed facts for one catalog's data directory.
///
/// Installing again for the same directory replaces what was there, which is
/// what a server restarting inside one process does
pub fn install_change_feed_facts_for(
    data_dir: std::path::PathBuf,
    facts: Arc<dyn ChangeFeedFacts>,
) {
    let registry = CHANGE_FEED_FACTS.get_or_init(|| std::sync::RwLock::new(Vec::new()));
    // A poisoned registry would leave every later plan without facts, so the
    // lock is taken through the guard a panicking writer left behind
    let mut held = match registry.write() {
        Ok(held) => held,
        Err(poisoned) => poisoned.into_inner(),
    };
    match held.iter_mut().find(|(at, _)| *at == data_dir) {
        Some(slot) => slot.1 = facts,
        None => held.push((data_dir, facts)),
    }
}

#[allow(clippy::type_complexity)]
static CHANGE_FEED_FACTS: std::sync::OnceLock<
    std::sync::RwLock<Vec<(std::path::PathBuf, Arc<dyn ChangeFeedFacts>)>>,
> = std::sync::OnceLock::new();

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
    plan_for_session(
        catalog,
        database_id,
        search_path,
        stmt,
        security,
        peers,
        None,
    )
    .await
}

/// Plans a statement for one session, so a bare name reaches that session's
/// temporary tables before the search path.
///
/// Every internal planning path passes None for `temp_tables`: a view, a
/// materialized view or a pipeline is defined once and read by whoever runs
/// it, so resolving one through some session's temporary namespace would
/// give two sessions different definitions of the same object.
pub async fn plan_for_session(
    catalog: &Catalog,
    database_id: DatabaseId,
    search_path: Vec<String>,
    stmt: Statement,
    security: Option<Arc<dyn RowSecurityProvider>>,
    peers: Option<&zyron_common::PeerRegistry>,
    temp_tables: Option<Arc<zyron_catalog::SessionTempTables>>,
) -> Result<PhysicalPlan> {
    let resolver = catalog.resolver_for_session(database_id, search_path, temp_tables);
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
    plan_for_explain_for_session(
        catalog,
        database_id,
        search_path,
        stmt,
        options,
        peers,
        None,
    )
    .await
}

/// Plans a statement for EXPLAIN in one session, so a bare name reaches that
/// session's temporary tables before the search path.
#[allow(clippy::too_many_arguments)]
pub async fn plan_for_explain_for_session(
    catalog: &Catalog,
    database_id: DatabaseId,
    search_path: Vec<String>,
    stmt: Statement,
    options: ExplainOptions,
    peers: Option<&zyron_common::PeerRegistry>,
    temp_tables: Option<Arc<zyron_catalog::SessionTempTables>>,
) -> Result<(PhysicalPlan, ExplainOptions)> {
    let resolver = catalog.resolver_for_session(database_id, search_path, temp_tables);
    let mut binder = Binder::new(resolver, catalog);
    let bound = binder.bind(stmt).await?;
    let logical = logical::builder::build_logical_plan(&bound)?;
    let optimized = Optimizer::new(catalog).optimize(logical)?;
    let physical = physical::builder::build_physical_plan(optimized, catalog, peers)?;
    Ok((physical, options))
}
