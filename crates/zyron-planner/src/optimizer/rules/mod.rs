//! Optimization rules for the query planner.

mod constant_folding;
pub mod encoding_pushdown;
mod index_advisor;
mod join_reorder;
pub mod parallel_plan;
mod predicate_pushdown;
mod projection_pushdown;
mod subquery_decorrelate;

pub use constant_folding::ConstantFolding;
pub use encoding_pushdown::EncodingPushdown;
pub use index_advisor::{IndexAdvisor, IndexRecommendation};
pub use join_reorder::JoinReorder;
pub use parallel_plan::{compute_worker_count, should_parallelize};
pub use predicate_pushdown::PredicatePushdown;
pub use projection_pushdown::ProjectionPushdown;
pub use subquery_decorrelate::SubqueryDecorrelate;
