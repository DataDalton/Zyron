//! Rule-based query optimizer.
//!
//! Applies transformation rules in a fixed-point loop until no rule
//! produces a change, or the maximum iteration count is reached.

pub mod cardinality;
pub mod rules;

use crate::logical::LogicalPlan;
use zyron_catalog::Catalog;
use zyron_common::Result;

/// Trait for a single optimization rule.
pub trait OptimizationRule: Send + Sync {
    /// Human-readable name for debugging.
    fn name(&self) -> &str;

    /// Applies the rule to a logical plan, returning a transformed plan.
    /// Returns None if the rule does not apply.
    fn apply(&self, plan: &LogicalPlan, catalog: &Catalog) -> Option<LogicalPlan>;
}

/// Applies a sequence of optimization rules in a fixed-point loop.
pub struct Optimizer<'a> {
    catalog: &'a Catalog,
    rules: Vec<Box<dyn OptimizationRule>>,
    max_iterations: usize,
}

impl<'a> Optimizer<'a> {
    pub fn new(catalog: &'a Catalog) -> Self {
        let rules: Vec<Box<dyn OptimizationRule>> = vec![
            Box::new(rules::ConstantFolding),
            Box::new(rules::PredicatePushdown),
            Box::new(rules::ProjectionPushdown),
            Box::new(rules::SubqueryDecorrelate),
            Box::new(rules::JoinReorder::new()),
            Box::new(rules::EncodingPushdown),
            Box::new(rules::IndexAdvisor::new()),
        ];
        Self {
            catalog,
            rules,
            max_iterations: 10,
        }
    }

    /// Runs all rules repeatedly until no rule produces a change.
    ///
    /// Trivial plan shapes that no rule can improve (a bare `Insert(Values)`
    /// or `Values` row source) bypass the rule loop entirely, sparing
    /// `INSERT INTO t VALUES (...)` queries the per-row tree walks that
    /// `ConstantFolding` and friends would otherwise do for nothing.
    pub fn optimize(&self, plan: LogicalPlan) -> Result<LogicalPlan> {
        if !plan_is_optimizable(&plan) {
            return Ok(plan);
        }
        let mut current = plan;
        for _ in 0..self.max_iterations {
            let mut changed = false;
            for rule in &self.rules {
                if let Some(new_plan) = rule.apply(&current, self.catalog) {
                    current = new_plan;
                    changed = true;
                }
            }
            if !changed {
                break;
            }
        }
        Ok(current)
    }
}

/// Returns true when at least one rule could meaningfully alter the plan.
///
/// Conservative: returns true for any shape that contains a Filter, Project,
/// Join, Aggregate, Sort, Distinct, SetOp, Subquery, or Scan — i.e. anything
/// where predicate/projection/encoding/index-advisor logic could fire.
/// Returns false for plans whose entire body is `Insert{Values}` or `Values`
/// without further structure on top.
fn plan_is_optimizable(plan: &LogicalPlan) -> bool {
    match plan {
        // Pure value-producer with no upstream structure.
        LogicalPlan::Values { .. } => false,
        // INSERT-from-VALUES has no filter/project/scan to rewrite. INSERT
        // with a SELECT source still walks into the source plan below.
        LogicalPlan::Insert { source, .. } => match source.as_ref() {
            LogicalPlan::Values { .. } => false,
            other => plan_is_optimizable(other),
        },
        _ => true,
    }
}
