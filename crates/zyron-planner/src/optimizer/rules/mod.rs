//! Optimization rules for the query planner.

mod constant_folding;
mod join_reorder;
mod predicate_pushdown;
mod projection_pushdown;
mod subquery_decorrelate;

pub use constant_folding::ConstantFolding;
pub use join_reorder::JoinReorder;
pub use predicate_pushdown::PredicatePushdown;
pub use projection_pushdown::ProjectionPushdown;
pub use subquery_decorrelate::SubqueryDecorrelate;
