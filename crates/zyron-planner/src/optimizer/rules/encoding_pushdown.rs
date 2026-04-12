//! Encoding pushdown optimization rule.
//!
//! Analyzes filter predicates on scan nodes to determine which columnar
//! encoding optimizations can be applied. Produces EncodingHint annotations
//! that the physical plan builder uses to adjust cost estimates for
//! encoded columnar scans.

use crate::binder::BoundExpr;
use crate::logical::LogicalPlan;
use crate::optimizer::OptimizationRule;
use zyron_catalog::Catalog;
use zyron_parser::ast::BinaryOperator;

// ---------------------------------------------------------------------------
// Encoding hint
// ---------------------------------------------------------------------------

/// Describes which encoding-level optimizations a predicate supports.
/// The physical builder uses these hints to estimate encoded scan costs
/// and decide whether a columnar scan is cheaper than a heap scan.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct EncodingHint {
    /// Predicate can be evaluated against zone map min/max values.
    pub zone_map_applicable: bool,
    /// Equality predicate can use bloom filter for fast rejection.
    pub bloom_filter_applicable: bool,
    /// Equality on dictionary-encoded column: look up code, scan codes array.
    pub dictionary_lookup: bool,
    /// Range predicate on RLE-sorted column: binary search on run values.
    pub rle_binary_search: bool,
    /// Range predicate on FastLanes column: check FoR base + delta bounds.
    pub fastlanes_bounds_check: bool,
}

impl EncodingHint {
    /// Returns true if any encoding optimization applies.
    pub fn any_applicable(&self) -> bool {
        self.zone_map_applicable
            || self.bloom_filter_applicable
            || self.dictionary_lookup
            || self.rle_binary_search
            || self.fastlanes_bounds_check
    }

    /// Merges another hint into this one (union of applicable optimizations).
    pub fn merge(&mut self, other: &EncodingHint) {
        self.zone_map_applicable |= other.zone_map_applicable;
        self.bloom_filter_applicable |= other.bloom_filter_applicable;
        self.dictionary_lookup |= other.dictionary_lookup;
        self.rle_binary_search |= other.rle_binary_search;
        self.fastlanes_bounds_check |= other.fastlanes_bounds_check;
    }

    /// Estimates the skip_rate for cost model parameters based on which
    /// optimizations are applicable. Returns a fraction (0.0 to 1.0).
    pub fn estimated_skip_rate(&self) -> f64 {
        let mut rate = 0.0_f64;
        if self.zone_map_applicable {
            rate += 0.3; // Zone maps typically skip ~30% of segments
        }
        if self.bloom_filter_applicable {
            rate += 0.2; // Bloom filters add ~20% skip for selective equality
        }
        rate.min(0.9) // Cap at 90% skip rate
    }
}

// ---------------------------------------------------------------------------
// Predicate analysis
// ---------------------------------------------------------------------------

/// Analyzes a predicate expression and returns encoding hints describing
/// which encoded evaluation strategies are applicable.
pub fn analyze_predicate(predicate: &BoundExpr) -> EncodingHint {
    match predicate {
        // Equality: zone map + bloom filter + dictionary lookup
        BoundExpr::BinaryOp {
            op: BinaryOperator::Eq,
            ..
        } => EncodingHint {
            zone_map_applicable: true,
            bloom_filter_applicable: true,
            dictionary_lookup: true,
            rle_binary_search: false,
            fastlanes_bounds_check: false,
        },
        // Range: zone map + RLE binary search + FastLanes bounds check
        BoundExpr::BinaryOp {
            op:
                BinaryOperator::Lt | BinaryOperator::Gt | BinaryOperator::LtEq | BinaryOperator::GtEq,
            ..
        } => EncodingHint {
            zone_map_applicable: true,
            bloom_filter_applicable: false,
            dictionary_lookup: false,
            rle_binary_search: true,
            fastlanes_bounds_check: true,
        },
        // IN list: equality-like, supports bloom filter + dictionary
        BoundExpr::InList { negated: false, .. } => EncodingHint {
            zone_map_applicable: true,
            bloom_filter_applicable: true,
            dictionary_lookup: true,
            rle_binary_search: false,
            fastlanes_bounds_check: false,
        },
        // BETWEEN: range-like, supports zone map + RLE + FastLanes
        BoundExpr::Between { negated: false, .. } => EncodingHint {
            zone_map_applicable: true,
            bloom_filter_applicable: false,
            dictionary_lookup: false,
            rle_binary_search: true,
            fastlanes_bounds_check: true,
        },
        // AND: merge hints from both sides
        BoundExpr::BinaryOp {
            left,
            op: BinaryOperator::And,
            right,
            ..
        } => {
            let mut hint = analyze_predicate(left);
            hint.merge(&analyze_predicate(right));
            hint
        }
        // OR: only zone map applies (both branches might match different segments)
        BoundExpr::BinaryOp {
            left,
            op: BinaryOperator::Or,
            right,
            ..
        } => {
            let left_hint = analyze_predicate(left);
            let right_hint = analyze_predicate(right);
            EncodingHint {
                zone_map_applicable: left_hint.zone_map_applicable
                    && right_hint.zone_map_applicable,
                bloom_filter_applicable: false,
                dictionary_lookup: false,
                rle_binary_search: false,
                fastlanes_bounds_check: false,
            }
        }
        // Inequality (!=): zone map can only skip if min == max == value (rare).
        // Not worth inflating the estimated skip rate for this case.
        BoundExpr::BinaryOp {
            op: BinaryOperator::Neq,
            ..
        } => EncodingHint {
            zone_map_applicable: false,
            bloom_filter_applicable: false,
            dictionary_lookup: false,
            rle_binary_search: false,
            fastlanes_bounds_check: false,
        },
        _ => EncodingHint::default(),
    }
}

// ---------------------------------------------------------------------------
// Optimization rule
// ---------------------------------------------------------------------------

/// Encoding pushdown rule: annotates logical scan nodes with encoding hints
/// derived from filter predicates. The physical builder reads these hints
/// to adjust cost estimates for columnar-encoded scans.
pub struct EncodingPushdown;

impl OptimizationRule for EncodingPushdown {
    fn name(&self) -> &str {
        "encoding_pushdown"
    }

    /// Walks the plan tree looking for Filter -> Scan patterns.
    /// When found, analyzes the filter predicate and stores encoding hints
    /// on the scan node via the encoding_hints field.
    fn apply(&self, plan: &LogicalPlan, _catalog: &Catalog) -> Option<LogicalPlan> {
        self.transform(plan)
    }
}

impl EncodingPushdown {
    fn transform(&self, plan: &LogicalPlan) -> Option<LogicalPlan> {
        match plan {
            // Filter -> Scan: analyze predicate for encoding hints
            LogicalPlan::Filter { predicate, child } => {
                if let LogicalPlan::Scan {
                    table_id,
                    table_idx,
                    columns,
                    alias,
                    encoding_hints: _,
                    as_of,
                } = child.as_ref()
                {
                    let hint = analyze_predicate(predicate);
                    if hint.any_applicable() {
                        let new_scan = LogicalPlan::Scan {
                            table_id: *table_id,
                            table_idx: *table_idx,
                            columns: columns.clone(),
                            alias: alias.clone(),
                            encoding_hints: Some(hint),
                            as_of: as_of.clone(),
                        };
                        return Some(LogicalPlan::Filter {
                            predicate: predicate.clone(),
                            child: Box::new(new_scan),
                        });
                    }
                }

                // Recurse into child
                if let Some(new_child) = self.transform(child) {
                    return Some(LogicalPlan::Filter {
                        predicate: predicate.clone(),
                        child: Box::new(new_child),
                    });
                }
                None
            }
            // Recurse into single-child nodes
            LogicalPlan::Project {
                expressions,
                aliases,
                child,
            } => self.transform(child).map(|new_child| LogicalPlan::Project {
                expressions: expressions.clone(),
                aliases: aliases.clone(),
                child: Box::new(new_child),
            }),
            LogicalPlan::Sort { order_by, child } => {
                self.transform(child).map(|new_child| LogicalPlan::Sort {
                    order_by: order_by.clone(),
                    child: Box::new(new_child),
                })
            }
            LogicalPlan::Limit {
                limit,
                offset,
                child,
            } => self.transform(child).map(|new_child| LogicalPlan::Limit {
                limit: *limit,
                offset: *offset,
                child: Box::new(new_child),
            }),
            LogicalPlan::Distinct { child } => {
                self.transform(child)
                    .map(|new_child| LogicalPlan::Distinct {
                        child: Box::new(new_child),
                    })
            }
            LogicalPlan::Aggregate {
                group_by,
                aggregates,
                child,
            } => self
                .transform(child)
                .map(|new_child| LogicalPlan::Aggregate {
                    group_by: group_by.clone(),
                    aggregates: aggregates.clone(),
                    child: Box::new(new_child),
                }),
            // Recurse into two-child nodes
            LogicalPlan::Join {
                left,
                right,
                join_type,
                condition,
            } => {
                let new_left = self.transform(left);
                let new_right = self.transform(right);
                if new_left.is_some() || new_right.is_some() {
                    Some(LogicalPlan::Join {
                        left: Box::new(new_left.unwrap_or_else(|| left.as_ref().clone())),
                        right: Box::new(new_right.unwrap_or_else(|| right.as_ref().clone())),
                        join_type: join_type.clone(),
                        condition: condition.clone(),
                    })
                } else {
                    None
                }
            }
            LogicalPlan::SetOp {
                op,
                all,
                left,
                right,
            } => {
                let new_left = self.transform(left);
                let new_right = self.transform(right);
                if new_left.is_some() || new_right.is_some() {
                    Some(LogicalPlan::SetOp {
                        op: op.clone(),
                        all: *all,
                        left: Box::new(new_left.unwrap_or_else(|| left.as_ref().clone())),
                        right: Box::new(new_right.unwrap_or_else(|| right.as_ref().clone())),
                    })
                } else {
                    None
                }
            }
            // DML nodes: recurse into child/source
            LogicalPlan::Insert {
                table_id,
                target_columns,
                source,
            } => self
                .transform(source)
                .map(|new_source| LogicalPlan::Insert {
                    table_id: *table_id,
                    target_columns: target_columns.clone(),
                    source: Box::new(new_source),
                }),
            LogicalPlan::Update {
                table_id,
                assignments,
                child,
            } => self.transform(child).map(|new_child| LogicalPlan::Update {
                table_id: *table_id,
                assignments: assignments.clone(),
                child: Box::new(new_child),
            }),
            LogicalPlan::Delete { table_id, child } => {
                self.transform(child).map(|new_child| LogicalPlan::Delete {
                    table_id: *table_id,
                    child: Box::new(new_child),
                })
            }
            // Leaf nodes: no transformation
            LogicalPlan::Scan { .. }
            | LogicalPlan::Values { .. }
            | LogicalPlan::GraphAlgorithm { .. } => None,
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_equality_hint() {
        let pred = BoundExpr::BinaryOp {
            left: Box::new(BoundExpr::Literal {
                value: zyron_parser::ast::LiteralValue::Integer(1),
                type_id: zyron_common::TypeId::Int64,
            }),
            op: BinaryOperator::Eq,
            right: Box::new(BoundExpr::Literal {
                value: zyron_parser::ast::LiteralValue::Integer(2),
                type_id: zyron_common::TypeId::Int64,
            }),
            type_id: zyron_common::TypeId::Boolean,
        };
        let hint = analyze_predicate(&pred);
        assert!(hint.zone_map_applicable);
        assert!(hint.bloom_filter_applicable);
        assert!(hint.dictionary_lookup);
        assert!(!hint.rle_binary_search);
    }

    #[test]
    fn test_range_hint() {
        let pred = BoundExpr::BinaryOp {
            left: Box::new(BoundExpr::Literal {
                value: zyron_parser::ast::LiteralValue::Integer(1),
                type_id: zyron_common::TypeId::Int64,
            }),
            op: BinaryOperator::Gt,
            right: Box::new(BoundExpr::Literal {
                value: zyron_parser::ast::LiteralValue::Integer(100),
                type_id: zyron_common::TypeId::Int64,
            }),
            type_id: zyron_common::TypeId::Boolean,
        };
        let hint = analyze_predicate(&pred);
        assert!(hint.zone_map_applicable);
        assert!(!hint.bloom_filter_applicable);
        assert!(hint.rle_binary_search);
        assert!(hint.fastlanes_bounds_check);
    }

    #[test]
    fn test_and_merges_hints() {
        let eq = BoundExpr::BinaryOp {
            left: Box::new(BoundExpr::Literal {
                value: zyron_parser::ast::LiteralValue::Integer(1),
                type_id: zyron_common::TypeId::Int64,
            }),
            op: BinaryOperator::Eq,
            right: Box::new(BoundExpr::Literal {
                value: zyron_parser::ast::LiteralValue::Integer(2),
                type_id: zyron_common::TypeId::Int64,
            }),
            type_id: zyron_common::TypeId::Boolean,
        };
        let range = BoundExpr::BinaryOp {
            left: Box::new(BoundExpr::Literal {
                value: zyron_parser::ast::LiteralValue::Integer(1),
                type_id: zyron_common::TypeId::Int64,
            }),
            op: BinaryOperator::Lt,
            right: Box::new(BoundExpr::Literal {
                value: zyron_parser::ast::LiteralValue::Integer(100),
                type_id: zyron_common::TypeId::Int64,
            }),
            type_id: zyron_common::TypeId::Boolean,
        };
        let and = BoundExpr::BinaryOp {
            left: Box::new(eq),
            op: BinaryOperator::And,
            right: Box::new(range),
            type_id: zyron_common::TypeId::Boolean,
        };
        let hint = analyze_predicate(&and);
        assert!(hint.zone_map_applicable);
        assert!(hint.bloom_filter_applicable);
        assert!(hint.dictionary_lookup);
        assert!(hint.rle_binary_search);
        assert!(hint.fastlanes_bounds_check);
    }

    #[test]
    fn test_default_hint_not_applicable() {
        let hint = EncodingHint::default();
        assert!(!hint.any_applicable());
    }

    #[test]
    fn test_skip_rate() {
        let hint = EncodingHint {
            zone_map_applicable: true,
            bloom_filter_applicable: true,
            dictionary_lookup: false,
            rle_binary_search: false,
            fastlanes_bounds_check: false,
        };
        let rate = hint.estimated_skip_rate();
        assert!((rate - 0.5).abs() < 0.001); // 0.3 + 0.2
    }
}
