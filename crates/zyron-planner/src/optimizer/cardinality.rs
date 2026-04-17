//! Cardinality estimation for query optimization.
//!
//! CardinalityEstimator provides selectivity and cardinality estimates using
//! catalog statistics. The estimation hierarchy for each predicate type is:
//! 1. MCV (most common values) frequency lookup
//! 2. Histogram bucket interpolation
//! 3. 1/NDV (number of distinct values)
//! 4. Default heuristic constants

use crate::binder::BoundExpr;
use crate::cost::{
    DEFAULT_EQUALITY_SELECTIVITY, DEFAULT_IN_LIST_SELECTIVITY, DEFAULT_LIKE_SELECTIVITY,
    DEFAULT_NULL_SELECTIVITY, DEFAULT_RANGE_SELECTIVITY,
};
use crate::statistics::histogram::EquiHeightHistogram;
use zyron_catalog::{ColumnId, ColumnStats, TableStats};
use zyron_parser::ast::{BinaryOperator, JoinType, LiteralValue, UnaryOperator};

// ---------------------------------------------------------------------------
// Cardinality estimator
// ---------------------------------------------------------------------------

/// Central cardinality estimation engine using catalog statistics.
pub struct CardinalityEstimator;

impl CardinalityEstimator {
    /// Estimates selectivity of a predicate (fraction of rows matching, 0.0 to 1.0).
    /// Uses MCV list, histogram interpolation, NDV, or heuristic defaults.
    pub fn estimate_selectivity(
        predicate: &BoundExpr,
        table_stats: Option<&TableStats>,
        column_stats: Option<&[ColumnStats]>,
    ) -> f64 {
        match predicate {
            // AND: independence assumption
            BoundExpr::BinaryOp {
                left,
                op: BinaryOperator::And,
                right,
                ..
            } => {
                let left_sel = Self::estimate_selectivity(left, table_stats, column_stats);
                let right_sel = Self::estimate_selectivity(right, table_stats, column_stats);
                Self::estimate_and(left_sel, right_sel, None)
            }
            // OR: inclusion-exclusion
            BoundExpr::BinaryOp {
                left,
                op: BinaryOperator::Or,
                right,
                ..
            } => {
                let left_sel = Self::estimate_selectivity(left, table_stats, column_stats);
                let right_sel = Self::estimate_selectivity(right, table_stats, column_stats);
                Self::estimate_or(left_sel, right_sel)
            }
            // NOT
            BoundExpr::UnaryOp {
                op: UnaryOperator::Not,
                expr,
                ..
            } => 1.0 - Self::estimate_selectivity(expr, table_stats, column_stats),
            // Equality: col = literal
            BoundExpr::BinaryOp {
                left,
                op: BinaryOperator::Eq,
                right,
                ..
            } => Self::estimate_eq_predicate(left, right, column_stats),
            // Inequality: col != literal
            BoundExpr::BinaryOp {
                left,
                op: BinaryOperator::Neq,
                right,
                ..
            } => 1.0 - Self::estimate_eq_predicate(left, right, column_stats),
            // Range: col < literal, col > literal, col <= literal, col >= literal
            BoundExpr::BinaryOp {
                left,
                op:
                    op @ (BinaryOperator::Lt
                    | BinaryOperator::Gt
                    | BinaryOperator::LtEq
                    | BinaryOperator::GtEq),
                right,
                ..
            } => Self::estimate_range_predicate(left, right, op, column_stats),
            // IS NULL / IS NOT NULL
            BoundExpr::IsNull { expr, negated } => {
                if let Some(col_id) = extract_column_id(expr) {
                    if let Some(stats) = column_stats {
                        if let Some(cs) = stats.iter().find(|s| s.column_id == col_id) {
                            return if *negated {
                                1.0 - cs.null_fraction
                            } else {
                                cs.null_fraction
                            };
                        }
                    }
                }
                if *negated {
                    1.0 - DEFAULT_NULL_SELECTIVITY
                } else {
                    DEFAULT_NULL_SELECTIVITY
                }
            }
            // IN list
            BoundExpr::InList {
                expr,
                list,
                negated,
            } => {
                let base = if let Some(col_id) = extract_column_id(expr) {
                    let eq_sel = Self::estimate_equality(col_id, column_stats);
                    (list.len() as f64 * eq_sel).min(1.0)
                } else {
                    DEFAULT_IN_LIST_SELECTIVITY
                };
                if *negated { 1.0 - base } else { base }
            }
            // BETWEEN
            BoundExpr::Between { expr, negated, .. } => {
                let base = if let Some(col_id) = extract_column_id(expr) {
                    Self::estimate_range_for_column(col_id, column_stats)
                } else {
                    DEFAULT_RANGE_SELECTIVITY
                };
                if *negated { 1.0 - base } else { base }
            }
            // LIKE / ILIKE
            BoundExpr::Like { negated, .. } | BoundExpr::ILike { negated, .. } => {
                if *negated {
                    1.0 - DEFAULT_LIKE_SELECTIVITY
                } else {
                    DEFAULT_LIKE_SELECTIVITY
                }
            }
            // Boolean literal
            BoundExpr::Literal {
                value: LiteralValue::Boolean(true),
                ..
            } => 1.0,
            BoundExpr::Literal {
                value: LiteralValue::Boolean(false),
                ..
            } => 0.0,
            // Default
            _ => 0.5,
        }
    }

    /// Estimates join cardinality using NDV of join columns.
    /// result = left_rows * right_rows / max(left_ndv, right_ndv)
    pub fn estimate_join_cardinality(
        left_rows: f64,
        right_rows: f64,
        left_ndv: f64,
        right_ndv: f64,
        join_type: &JoinType,
    ) -> f64 {
        let max_ndv = left_ndv.max(right_ndv).max(1.0);
        let base = left_rows * right_rows / max_ndv;

        match join_type {
            JoinType::Cross => (left_rows * right_rows).max(1.0),
            JoinType::Left => base.max(left_rows),
            JoinType::Right => base.max(right_rows),
            JoinType::Full => base.max(left_rows.max(right_rows)),
            JoinType::Inner => base.max(1.0),
        }
    }

    // -----------------------------------------------------------------------
    // Internal estimation methods
    // -----------------------------------------------------------------------

    /// AND selectivity with optional correlation adjustment.
    /// At corr=0 (independent): left * right.
    /// At corr=1 (fully correlated): min(left, right).
    /// Interpolates linearly between the two extremes.
    fn estimate_and(left: f64, right: f64, correlation: Option<f64>) -> f64 {
        if let Some(corr) = correlation {
            let abs_corr = corr.abs().min(1.0);
            let independent = left * right;
            let correlated = left.min(right);
            independent + abs_corr * (correlated - independent)
        } else {
            // Independence assumption
            left * right
        }
    }

    /// OR selectivity using inclusion-exclusion.
    fn estimate_or(left: f64, right: f64) -> f64 {
        left + right - left * right
    }

    /// Estimates equality selectivity for a comparison expression.
    fn estimate_eq_predicate(
        left: &BoundExpr,
        right: &BoundExpr,
        column_stats: Option<&[ColumnStats]>,
    ) -> f64 {
        // Check left = literal
        if let Some(col_id) = extract_column_id(left) {
            if let Some(literal_bytes) = extract_literal_bytes(right) {
                return Self::estimate_equality_with_value(col_id, &literal_bytes, column_stats);
            }
            return Self::estimate_equality(col_id, column_stats);
        }
        // Check literal = right
        if let Some(col_id) = extract_column_id(right) {
            if let Some(literal_bytes) = extract_literal_bytes(left) {
                return Self::estimate_equality_with_value(col_id, &literal_bytes, column_stats);
            }
            return Self::estimate_equality(col_id, column_stats);
        }
        DEFAULT_EQUALITY_SELECTIVITY
    }

    /// Estimates range selectivity for a comparison expression.
    fn estimate_range_predicate(
        left: &BoundExpr,
        right: &BoundExpr,
        op: &BinaryOperator,
        column_stats: Option<&[ColumnStats]>,
    ) -> f64 {
        if let Some(col_id) = extract_column_id(left) {
            if let Some(literal_bytes) = extract_literal_bytes(right) {
                return Self::estimate_range_with_histogram(
                    col_id,
                    &literal_bytes,
                    op,
                    column_stats,
                );
            }
            return Self::estimate_range_for_column(col_id, column_stats);
        }
        if let Some(col_id) = extract_column_id(right) {
            if let Some(literal_bytes) = extract_literal_bytes(left) {
                // Flip the operator direction: literal < col becomes col > literal
                let flipped = match op {
                    BinaryOperator::Lt => BinaryOperator::Gt,
                    BinaryOperator::Gt => BinaryOperator::Lt,
                    BinaryOperator::LtEq => BinaryOperator::GtEq,
                    BinaryOperator::GtEq => BinaryOperator::LtEq,
                    other => other.clone(),
                };
                return Self::estimate_range_with_histogram(
                    col_id,
                    &literal_bytes,
                    &flipped,
                    column_stats,
                );
            }
        }
        DEFAULT_RANGE_SELECTIVITY
    }

    /// Equality selectivity using MCV then 1/NDV fallback.
    fn estimate_equality(column_id: ColumnId, column_stats: Option<&[ColumnStats]>) -> f64 {
        if let Some(stats) = column_stats {
            if let Some(cs) = stats.iter().find(|s| s.column_id == column_id) {
                // Use average MCV frequency as a reasonable estimate
                if !cs.most_common_freqs.is_empty() {
                    let avg_freq: f64 = cs.most_common_freqs.iter().sum::<f64>()
                        / cs.most_common_freqs.len() as f64;
                    return avg_freq;
                }
                if cs.distinct_count > 0 {
                    return 1.0 / cs.distinct_count as f64;
                }
            }
        }
        DEFAULT_EQUALITY_SELECTIVITY
    }

    /// Equality selectivity using MCV lookup for a specific value,
    /// falling back to (1 - mcv_total_freq) / (ndv - mcv_count).
    fn estimate_equality_with_value(
        column_id: ColumnId,
        value: &[u8],
        column_stats: Option<&[ColumnStats]>,
    ) -> f64 {
        if let Some(stats) = column_stats {
            if let Some(cs) = stats.iter().find(|s| s.column_id == column_id) {
                // Direct MCV lookup on borrowed slices (no allocation)
                for (i, mcv_val) in cs.most_common_values.iter().enumerate() {
                    if mcv_val.as_slice() == value {
                        return cs.most_common_freqs[i];
                    }
                }

                // Value not in MCV: use non-MCV frequency
                if cs.distinct_count > 0 {
                    let mcv_count = cs.most_common_values.len() as u64;
                    let mcv_freq: f64 = cs.most_common_freqs.iter().sum();
                    let remaining_ndv = cs.distinct_count.saturating_sub(mcv_count).max(1);
                    return (1.0 - mcv_freq) / remaining_ndv as f64;
                }
            }
        }
        DEFAULT_EQUALITY_SELECTIVITY
    }

    /// Range selectivity using histogram bucket interpolation.
    fn estimate_range_with_histogram(
        column_id: ColumnId,
        value: &[u8],
        op: &BinaryOperator,
        column_stats: Option<&[ColumnStats]>,
    ) -> f64 {
        if let Some(stats) = column_stats {
            if let Some(cs) = stats.iter().find(|s| s.column_id == column_id) {
                if let Some(hist) = &cs.histogram {
                    // Build EquiHeightHistogram from catalog Histogram
                    if let Some(eqh) = build_equi_height_from_catalog(hist) {
                        return match op {
                            BinaryOperator::Lt | BinaryOperator::LtEq => {
                                eqh.estimate_range_selectivity(None, Some(value))
                            }
                            BinaryOperator::Gt | BinaryOperator::GtEq => {
                                eqh.estimate_range_selectivity(Some(value), None)
                            }
                            _ => DEFAULT_RANGE_SELECTIVITY,
                        };
                    }
                }
            }
        }
        DEFAULT_RANGE_SELECTIVITY
    }

    /// Range selectivity using NDV-based estimate when no histogram is available.
    fn estimate_range_for_column(column_id: ColumnId, column_stats: Option<&[ColumnStats]>) -> f64 {
        if let Some(stats) = column_stats {
            if let Some(cs) = stats.iter().find(|s| s.column_id == column_id) {
                if cs.histogram.is_some() || cs.distinct_count > 0 {
                    return DEFAULT_RANGE_SELECTIVITY;
                }
            }
        }
        DEFAULT_RANGE_SELECTIVITY
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Extracts a ColumnId from a BoundExpr if it is a column reference.
fn extract_column_id(expr: &BoundExpr) -> Option<ColumnId> {
    match expr {
        BoundExpr::ColumnRef(cr) => Some(cr.column_id),
        BoundExpr::Nested(inner) => extract_column_id(inner),
        _ => None,
    }
}

/// Extracts literal value bytes from a BoundExpr for histogram lookups.
fn extract_literal_bytes(expr: &BoundExpr) -> Option<Vec<u8>> {
    match expr {
        BoundExpr::Literal { value, .. } => match value {
            LiteralValue::Integer(v) => Some(v.to_be_bytes().to_vec()),
            LiteralValue::Float(v) => Some(v.to_be_bytes().to_vec()),
            LiteralValue::String(v) => Some(v.as_bytes().to_vec()),
            LiteralValue::Boolean(v) => Some(vec![*v as u8]),
            LiteralValue::Null => None,
            LiteralValue::Interval(i) => Some(i.to_le_bytes().to_vec()),
        },
        BoundExpr::Nested(inner) => extract_literal_bytes(inner),
        _ => None,
    }
}

/// Converts a catalog Histogram into an EquiHeightHistogram for selectivity estimation.
fn build_equi_height_from_catalog(hist: &zyron_catalog::Histogram) -> Option<EquiHeightHistogram> {
    if hist.bounds.is_empty() || hist.counts.is_empty() {
        return None;
    }

    let total_rows: u64 = hist.counts.iter().sum();
    if total_rows == 0 {
        return None;
    }

    // Catalog histogram stores bounds and counts directly.
    // Estimate distinct counts per bucket from total counts (assume uniform distribution).
    let total_distinct = hist.bounds.len().saturating_sub(1) as u64;
    let per_bucket_distinct = if hist.counts.is_empty() {
        1
    } else {
        (total_distinct / hist.counts.len() as u64).max(1)
    };

    let distinct_counts = vec![per_bucket_distinct; hist.counts.len()];

    Some(EquiHeightHistogram {
        bounds: hist.bounds.clone(),
        row_counts: hist.counts.clone(),
        distinct_counts,
        total_rows,
        col_min: None,
        col_max: None,
    })
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::binder::ColumnRef;
    use zyron_catalog::{Histogram, TableId};
    use zyron_common::TypeId;

    fn make_column_ref(col_id: u16) -> BoundExpr {
        BoundExpr::ColumnRef(ColumnRef {
            table_idx: 0,
            column_id: ColumnId(col_id),
            type_id: TypeId::Int64,
            nullable: false,
        })
    }

    fn make_literal_int(val: i64) -> BoundExpr {
        BoundExpr::Literal {
            value: LiteralValue::Integer(val),
            type_id: TypeId::Int64,
        }
    }

    fn make_eq(left: BoundExpr, right: BoundExpr) -> BoundExpr {
        BoundExpr::BinaryOp {
            left: Box::new(left),
            op: BinaryOperator::Eq,
            right: Box::new(right),
            type_id: TypeId::Boolean,
        }
    }

    fn make_stats_with_ndv(col_id: u16, ndv: u64) -> Vec<ColumnStats> {
        vec![ColumnStats {
            table_id: TableId(1),
            column_id: ColumnId(col_id),
            null_fraction: 0.05,
            distinct_count: ndv,
            avg_width: 8,
            histogram: None,
            most_common_values: vec![],
            most_common_freqs: vec![],
        }]
    }

    fn make_stats_with_mcv(
        col_id: u16,
        values: Vec<Vec<u8>>,
        freqs: Vec<f64>,
        ndv: u64,
    ) -> Vec<ColumnStats> {
        vec![ColumnStats {
            table_id: TableId(1),
            column_id: ColumnId(col_id),
            null_fraction: 0.0,
            distinct_count: ndv,
            avg_width: 8,
            histogram: None,
            most_common_values: values,
            most_common_freqs: freqs,
        }]
    }

    #[test]
    fn test_equality_with_ndv() {
        let pred = make_eq(make_column_ref(0), make_literal_int(42));
        let stats = make_stats_with_ndv(0, 100);
        let sel = CardinalityEstimator::estimate_selectivity(&pred, None, Some(&stats));
        // 1/100 = 0.01
        assert!((sel - 0.01).abs() < 0.001);
    }

    #[test]
    fn test_equality_with_mcv_hit() {
        let value_bytes = 42i64.to_be_bytes().to_vec();
        let stats = make_stats_with_mcv(0, vec![value_bytes], vec![0.25], 100);
        let pred = make_eq(make_column_ref(0), make_literal_int(42));
        let sel = CardinalityEstimator::estimate_selectivity(&pred, None, Some(&stats));
        // MCV frequency for value 42 should be 0.25
        assert!((sel - 0.25).abs() < 0.001);
    }

    #[test]
    fn test_equality_with_mcv_miss() {
        let value_bytes = 1i64.to_be_bytes().to_vec();
        let stats = make_stats_with_mcv(0, vec![value_bytes], vec![0.30], 100);
        let pred = make_eq(make_column_ref(0), make_literal_int(42));
        let sel = CardinalityEstimator::estimate_selectivity(&pred, None, Some(&stats));
        // Non-MCV: (1 - 0.30) / (100 - 1) = 0.70/99 ~= 0.00707
        assert!(sel > 0.005 && sel < 0.01, "sel = {}", sel);
    }

    #[test]
    fn test_and_selectivity() {
        let left = make_eq(make_column_ref(0), make_literal_int(1));
        let right = make_eq(make_column_ref(1), make_literal_int(2));
        let pred = BoundExpr::BinaryOp {
            left: Box::new(left),
            op: BinaryOperator::And,
            right: Box::new(right),
            type_id: TypeId::Boolean,
        };
        // No stats: both default to 0.1, AND = 0.01
        let sel = CardinalityEstimator::estimate_selectivity(&pred, None, None);
        assert!((sel - 0.01).abs() < 0.001);
    }

    #[test]
    fn test_or_selectivity() {
        let left = BoundExpr::Literal {
            value: LiteralValue::Boolean(false),
            type_id: TypeId::Boolean,
        };
        let right = BoundExpr::Literal {
            value: LiteralValue::Boolean(true),
            type_id: TypeId::Boolean,
        };
        let pred = BoundExpr::BinaryOp {
            left: Box::new(left),
            op: BinaryOperator::Or,
            right: Box::new(right),
            type_id: TypeId::Boolean,
        };
        let sel = CardinalityEstimator::estimate_selectivity(&pred, None, None);
        assert!((sel - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_join_cardinality_inner() {
        let rows = CardinalityEstimator::estimate_join_cardinality(
            1000.0,
            500.0,
            100.0,
            50.0,
            &JoinType::Inner,
        );
        // 1000 * 500 / max(100, 50) = 5000
        assert!((rows - 5000.0).abs() < 1.0);
    }

    #[test]
    fn test_join_cardinality_left_outer() {
        let rows = CardinalityEstimator::estimate_join_cardinality(
            1000.0,
            10.0,
            100.0,
            5.0,
            &JoinType::Left,
        );
        // base = 1000 * 10 / 100 = 100, max(100, 1000) = 1000
        assert!(rows >= 1000.0);
    }

    #[test]
    fn test_is_null_selectivity() {
        let pred = BoundExpr::IsNull {
            expr: Box::new(make_column_ref(0)),
            negated: false,
        };
        let stats = make_stats_with_ndv(0, 100);
        let sel = CardinalityEstimator::estimate_selectivity(&pred, None, Some(&stats));
        assert!((sel - 0.05).abs() < 0.001); // null_fraction = 0.05
    }

    #[test]
    fn test_not_selectivity() {
        let pred = BoundExpr::UnaryOp {
            op: UnaryOperator::Not,
            expr: Box::new(BoundExpr::Literal {
                value: LiteralValue::Boolean(true),
                type_id: TypeId::Boolean,
            }),
            type_id: TypeId::Boolean,
        };
        let sel = CardinalityEstimator::estimate_selectivity(&pred, None, None);
        assert!((sel - 0.0).abs() < 0.001);
    }

    #[test]
    fn test_range_with_histogram() {
        let bounds: Vec<Vec<u8>> = (0..11u64)
            .map(|i| (i * 100).to_be_bytes().to_vec())
            .collect();
        let counts = vec![100u64; 10];
        let stats = vec![ColumnStats {
            table_id: TableId(1),
            column_id: ColumnId(0),
            null_fraction: 0.0,
            distinct_count: 1000,
            avg_width: 8,
            histogram: Some(Histogram {
                num_buckets: 10,
                bounds,
                counts,
            }),
            most_common_values: vec![],
            most_common_freqs: vec![],
        }];

        let pred = BoundExpr::BinaryOp {
            left: Box::new(make_column_ref(0)),
            op: BinaryOperator::Lt,
            right: Box::new(make_literal_int(500)),
            type_id: TypeId::Boolean,
        };
        let sel = CardinalityEstimator::estimate_selectivity(&pred, None, Some(&stats));
        // Should be roughly 0.5 (half the range)
        assert!(sel > 0.2 && sel < 0.8, "range sel = {}", sel);
    }
}
