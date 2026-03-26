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
    pub fn optimize(&self, plan: LogicalPlan) -> Result<LogicalPlan> {
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
