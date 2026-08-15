//! Join reordering optimization rule.
//!
//! Uses dynamic programming over subsets for up to 10 relations.
//! Falls back to a greedy heuristic for larger join graphs.
//! Reorders joins to minimize estimated total cost.
//! DP stores compact (left_mask, right_mask) split descriptors instead of
//! full LogicalPlan trees, deferring plan construction to a single pass
//! after the optimal ordering is found. This eliminates O(2^n) intermediate
//! tree clones during enumeration.

use crate::binder::BoundExpr;
use crate::cost::{CostModel, PlanCost};
use crate::logical::{JoinCondition, LogicalPlan};
use crate::optimizer::OptimizationRule;
use std::sync::Arc;
use zyron_catalog::Catalog;
use zyron_parser::ast::JoinType;

/// Maximum number of relations for DP join enumeration.
const MAX_DP_RELATIONS: usize = 10;

pub struct JoinReorder {
    cost_model: CostModel,
}

impl JoinReorder {
    pub fn new() -> Self {
        Self {
            cost_model: CostModel::default(),
        }
    }
}

impl OptimizationRule for JoinReorder {
    fn name(&self) -> &str {
        "join_reorder"
    }

    fn apply(&self, plan: &LogicalPlan, catalog: &Catalog) -> Option<LogicalPlan> {
        if !has_join(plan) {
            return None;
        }
        let reordered = reorder_joins(plan, &self.cost_model, catalog);
        if reordered != *plan {
            Some(reordered)
        } else {
            None
        }
    }
}

/// Returns true if the plan tree contains any Join node.
fn has_join(plan: &LogicalPlan) -> bool {
    match plan {
        LogicalPlan::Join { .. } => true,
        other => other.children().iter().any(|c| has_join(c)),
    }
}

fn reorder_joins(plan: &LogicalPlan, cost_model: &CostModel, catalog: &Catalog) -> LogicalPlan {
    match plan {
        LogicalPlan::Join { .. } => {
            // Collect all base relations and join predicates
            let mut relations = Vec::new();
            let mut predicates = Vec::new();
            flatten_join_tree(plan, &mut relations, &mut predicates);

            if relations.len() < 2 {
                return plan.clone();
            }

            // Recursively optimize children first
            let relations: Vec<LogicalPlan> = relations
                .into_iter()
                .map(|r| reorder_joins(&r, cost_model, catalog))
                .collect();

            if relations.len() > MAX_DP_RELATIONS {
                greedy_join_order(relations, predicates, cost_model, catalog)
            } else {
                dp_join_order(relations, predicates, cost_model, catalog)
            }
        }
        // Recurse into other plan nodes
        LogicalPlan::Filter { predicate, child } => LogicalPlan::Filter {
            predicate: predicate.clone(),
            child: Arc::new(reorder_joins(child, cost_model, catalog)),
        },
        LogicalPlan::Project {
            expressions,
            aliases,
            child,
            output_table_idx,
        } => LogicalPlan::Project {
            expressions: expressions.clone(),
            aliases: aliases.clone(),
            child: Arc::new(reorder_joins(child, cost_model, catalog)),
            output_table_idx: *output_table_idx,
        },
        LogicalPlan::Aggregate {
            group_by,
            aggregates,
            child,
        } => LogicalPlan::Aggregate {
            group_by: group_by.clone(),
            aggregates: aggregates.clone(),
            child: Arc::new(reorder_joins(child, cost_model, catalog)),
        },
        LogicalPlan::Sort { order_by, child } => LogicalPlan::Sort {
            order_by: order_by.clone(),
            child: Arc::new(reorder_joins(child, cost_model, catalog)),
        },
        LogicalPlan::Limit {
            limit,
            offset,
            child,
        } => LogicalPlan::Limit {
            limit: *limit,
            offset: *offset,
            child: Arc::new(reorder_joins(child, cost_model, catalog)),
        },
        LogicalPlan::Distinct { child } => LogicalPlan::Distinct {
            child: Arc::new(reorder_joins(child, cost_model, catalog)),
        },
        LogicalPlan::SetOp {
            op,
            all,
            left,
            right,
        } => LogicalPlan::SetOp {
            op: *op,
            all: *all,
            left: Arc::new(reorder_joins(left, cost_model, catalog)),
            right: Arc::new(reorder_joins(right, cost_model, catalog)),
        },
        other => other.clone(),
    }
}

/// Flattens a left-deep or bushy inner join tree into base relations and predicates.
fn flatten_join_tree(
    plan: &LogicalPlan,
    relations: &mut Vec<LogicalPlan>,
    predicates: &mut Vec<BoundExpr>,
) {
    match plan {
        LogicalPlan::Join {
            left,
            right,
            join_type: JoinType::Inner,
            condition,
        } => {
            flatten_join_tree(left, relations, predicates);
            flatten_join_tree(right, relations, predicates);
            if let JoinCondition::On(expr) = condition {
                predicates.push(expr.clone());
            }
        }
        LogicalPlan::Join {
            left,
            right,
            join_type: JoinType::Cross,
            ..
        } => {
            flatten_join_tree(left, relations, predicates);
            flatten_join_tree(right, relations, predicates);
        }
        other => {
            relations.push(other.clone());
        }
    }
}

/// Compact DP entry storing the split decision (which subsets to join)
/// and the resulting cost, without storing LogicalPlan trees.
struct DpEntry {
    left_mask: usize,
    right_mask: usize,
    cumulative_cost: f64,
    plan_cost: PlanCost,
}

/// DP-based join ordering for small number of relations.
/// Stores compact split descriptors (left_mask, right_mask) instead of full
/// LogicalPlan trees. The plan is constructed once after the optimal ordering
/// is determined, eliminating all intermediate tree clones during enumeration.
fn dp_join_order(
    relations: Vec<LogicalPlan>,
    predicates: Vec<BoundExpr>,
    cost_model: &CostModel,
    catalog: &Catalog,
) -> LogicalPlan {
    let n = relations.len();
    let table_size = 1usize << n;

    // Flat Vec indexed by bitmask. None = not yet computed.
    let mut dp: Vec<Option<DpEntry>> = Vec::with_capacity(table_size);
    dp.resize_with(table_size, || None);

    // Initialize single relations with pre-computed costs (no plan stored).
    for i in 0..n {
        let mask = 1usize << i;
        let plan_cost = cost_model.estimate_plan_cost(&relations[i], catalog);
        let total = plan_cost.total();
        dp[mask] = Some(DpEntry {
            left_mask: 0,
            right_mask: 0,
            cumulative_cost: total,
            plan_cost,
        });
    }

    // Enumerate subsets of increasing size.
    for size in 2..=n {
        for mask in 1..table_size {
            if (mask as u32).count_ones() as usize != size {
                continue;
            }

            // Enumerate all non-empty proper subsets of mask.
            let mut sub = (mask - 1) & mask;
            while sub > 0 {
                let complement = mask & !sub;
                if complement > 0 && sub < complement {
                    if let (Some(left_entry), Some(right_entry)) = (&dp[sub], &dp[complement]) {
                        let join_cost = cost_model
                            .cost_hash_join(&left_entry.plan_cost, &right_entry.plan_cost);
                        let total = left_entry.cumulative_cost
                            + right_entry.cumulative_cost
                            + join_cost.total();

                        let is_better = match &dp[mask] {
                            Some(entry) => total < entry.cumulative_cost,
                            None => true,
                        };
                        if is_better {
                            dp[mask] = Some(DpEntry {
                                left_mask: sub,
                                right_mask: complement,
                                cumulative_cost: total,
                                plan_cost: join_cost,
                            });
                        }
                    }
                }
                sub = (sub - 1) & mask;
            }
        }
    }

    let total_mask = table_size - 1;
    if dp[total_mask].is_some() {
        let mut used = vec![false; predicates.len()];
        let plan = reconstruct_plan(total_mask, &dp, &relations, &predicates, &mut used);
        // A predicate no join claimed reads one input alone, or reads across
        // a pair this ordering never joins directly. It still has to run, so
        // it sits above the tree rather than being dropped
        let leftover: Vec<BoundExpr> = predicates
            .iter()
            .enumerate()
            .filter(|(i, _)| !used[*i])
            .map(|(_, p)| p.clone())
            .collect();
        if leftover.is_empty() {
            plan
        } else {
            LogicalPlan::Filter {
                predicate: super::predicate_pushdown::combine_conjuncts(leftover),
                child: Arc::new(plan),
            }
        }
    } else {
        build_left_deep(relations, predicates)
    }
}

/// Recursively builds a LogicalPlan from the DP split descriptors.
/// Only called once after the optimal ordering is found.
fn reconstruct_plan(
    mask: usize,
    dp: &[Option<DpEntry>],
    relations: &[LogicalPlan],
    predicates: &[BoundExpr],
    used: &mut [bool],
) -> LogicalPlan {
    let Some(entry) = dp[mask].as_ref() else {
        // Every mask reached here was filled by the enumeration above, so
        // this cannot happen. Rebuilding the relations left-deep keeps a
        // wrong plan from becoming a panic if that ever stops holding
        return build_left_deep(relations.to_vec(), predicates.to_vec());
    };

    // Base case: single relation (left_mask == 0 means this is a leaf)
    if entry.left_mask == 0 {
        let bit = mask.trailing_zeros() as usize;
        return relations[bit].clone();
    }

    let left = reconstruct_plan(entry.left_mask, dp, relations, predicates, used);
    let right = reconstruct_plan(entry.right_mask, dp, relations, predicates, used);

    let condition = find_join_predicate(
        predicates,
        used,
        entry.left_mask,
        entry.right_mask,
        relations,
    );

    LogicalPlan::Join {
        left: Arc::new(left),
        right: Arc::new(right),
        join_type: JoinType::Inner,
        condition,
    }
}

/// Greedy join ordering for large number of relations.
/// Repeatedly joins the pair with the lowest estimated cost.
/// Caches PlanCost per relation to avoid O(n^2) redundant cost calls.
fn greedy_join_order(
    mut relations: Vec<LogicalPlan>,
    predicates: Vec<BoundExpr>,
    cost_model: &CostModel,
    catalog: &Catalog,
) -> LogicalPlan {
    // Pre-compute costs for all relations.
    let mut costs: Vec<PlanCost> = relations
        .iter()
        .map(|r| cost_model.estimate_plan_cost(r, catalog))
        .collect();

    while relations.len() > 1 {
        let mut best_i = 0;
        let mut best_j = 1;
        let mut best_cost = f64::MAX;

        for i in 0..relations.len() {
            for j in (i + 1)..relations.len() {
                let join_cost = cost_model.cost_hash_join(&costs[i], &costs[j]);
                let total = join_cost.total();
                if total < best_cost {
                    best_cost = total;
                    best_i = i;
                    best_j = j;
                }
            }
        }

        let right = relations.remove(best_j);
        let right_cost = costs.remove(best_j);
        let left = relations.remove(best_i);
        let left_cost = costs.remove(best_i);

        let join_cost = cost_model.cost_hash_join(&left_cost, &right_cost);
        let join = LogicalPlan::Join {
            left: Arc::new(left),
            right: Arc::new(right),
            join_type: JoinType::Inner,
            condition: JoinCondition::Cross,
        };
        relations.push(join);
        costs.push(join_cost);
    }

    let mut plan = relations.remove(0);
    for pred in predicates {
        plan = LogicalPlan::Filter {
            predicate: pred,
            child: Arc::new(plan),
        };
    }
    plan
}

/// Finds a join predicate that references columns from both subsets.
/// The table indices a set of relations produces, as a bitmask join can be
/// tested against.
///
/// Uses the same walk predicate pushdown uses, so a derived table is
/// reported by the index its enclosing query addresses it by rather than by
/// the scans inside it, and the two rules cannot disagree about which side a
/// column belongs to.
fn tables_of(mask: usize, relations: &[LogicalPlan]) -> Vec<usize> {
    let mut out = Vec::new();
    for (i, relation) in relations.iter().enumerate() {
        if mask & (1usize << i) != 0 {
            out.extend(super::predicate_pushdown::collect_table_indices(relation));
        }
    }
    out
}

/// Takes the predicates that belong on this join, marking them used.
///
/// A predicate belongs here when every column it reads is available from the
/// two sides and at least one column comes from each, which is what makes it
/// a join condition rather than a filter on one input. Anything else is left
/// for another level or for the caller to apply above the tree.
///
/// Each predicate is claimed once. Handing the same one to every level, as a
/// first-available choice does, both repeats work and silently drops every
/// other predicate, so a three table join would return a cross product.
fn find_join_predicate(
    predicates: &[BoundExpr],
    used: &mut [bool],
    left_mask: usize,
    right_mask: usize,
    relations: &[LogicalPlan],
) -> JoinCondition {
    let left_tables = tables_of(left_mask, relations);
    let right_tables = tables_of(right_mask, relations);

    let mut chosen: Vec<BoundExpr> = Vec::new();
    for (i, predicate) in predicates.iter().enumerate() {
        if used[i] {
            continue;
        }
        let refs = super::predicate_pushdown::collect_column_refs(predicate);
        if refs.is_empty() {
            continue;
        }
        let touches_left = refs.iter().any(|r| left_tables.contains(&r.table_idx));
        let touches_right = refs.iter().any(|r| right_tables.contains(&r.table_idx));
        let fully_covered = refs
            .iter()
            .all(|r| left_tables.contains(&r.table_idx) || right_tables.contains(&r.table_idx));
        if touches_left && touches_right && fully_covered {
            used[i] = true;
            chosen.push(predicate.clone());
        }
    }

    if chosen.is_empty() {
        JoinCondition::Cross
    } else {
        JoinCondition::On(super::predicate_pushdown::combine_conjuncts(chosen))
    }
}

/// Builds a left-deep join tree in the given order.
fn build_left_deep(mut relations: Vec<LogicalPlan>, predicates: Vec<BoundExpr>) -> LogicalPlan {
    let mut plan = relations.remove(0);
    for relation in relations {
        plan = LogicalPlan::Join {
            left: Arc::new(plan),
            right: Arc::new(relation),
            join_type: JoinType::Inner,
            condition: JoinCondition::Cross,
        };
    }
    // Apply predicates as filters
    for pred in predicates {
        plan = LogicalPlan::Filter {
            predicate: pred,
            child: Arc::new(plan),
        };
    }
    plan
}
