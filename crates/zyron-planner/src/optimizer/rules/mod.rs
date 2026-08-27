//! Optimization rules for the query planner.

use crate::logical::LogicalPlan;
use std::sync::Arc;

/// Rebuilds a rewritten child plan slot, reusing the original Arc when the
/// rewrite left the child unchanged. Shared by rules whose rewrite functions
/// return None for an untouched subtree
pub(crate) fn rebuilt_child(
    original: &Arc<LogicalPlan>,
    rewritten: Option<LogicalPlan>,
) -> Arc<LogicalPlan> {
    match rewritten {
        Some(p) => Arc::new(p),
        None => Arc::clone(original),
    }
}

mod constant_folding;
pub mod encoding_pushdown;
mod index_advisor;
mod join_reorder;
pub mod parallel_plan;
mod predicate_pushdown;
mod projection_pushdown;

pub use constant_folding::ConstantFolding;
pub use encoding_pushdown::EncodingPushdown;
pub use index_advisor::{IndexAdvisor, IndexRecommendation};
pub use join_reorder::JoinReorder;
pub use parallel_plan::{compute_worker_count, should_parallelize};
pub use predicate_pushdown::PredicatePushdown;
pub use projection_pushdown::ProjectionPushdown;
