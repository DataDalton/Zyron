//! Choosing cluster keys from measurement.
//!
//! Frequency and selectivity counting only: which columns the workload
//! filters on, how often, and how many distinct values each holds. Nothing
//! here learns or predicts, and a proposal it produces still has to pass the
//! gate in `feedback` before a single file is rewritten.
//!
//! Per-column strategy follows the measured shape of the column rather than
//! one curve for everything, because Z-order degrades when dimensions have
//! very different cardinalities and loses to a sort for single-column range
//! predicates. The rules, in order:
//!
//! * over 90% null: dropped, there is nothing to order by
//! * near-unique (`ndv/rows >= 0.5`) or very high cardinality (`> 2^20`):
//!   RangePartition, so a range predicate reads one run of files
//! * temporal and mostly range-queried: RangePartition, same reason
//! * low cardinality (`<= 256`): BitInterleave, cheap and it leaves room for
//!   the other dimensions
//! * moderate (`<= 65536`): SpaceFilling, whose better locality is worth its
//!   more expensive transform at that size
//! * anything else: RangePartition, the safe default
//!
//! Bootstrap exists so a table is never unordered while it has no
//! observations: declared keys, then the primary key, then the leading
//! columns of the most selective unique constraint, then a temporal column,
//! then nothing.

use zyron_common::TypeId;

use crate::feedback::PredicateClass;
use crate::manifest::{ClusterKey, ClusterStrategy, ManifestFile};
use crate::predicate::{CompareOp, LakePredicate, LakeValue};
use crate::workload::{
    TERM_BYTES_CONSIDERED, TERM_BYTES_SKIPPED, TERM_EQUALITY, TERM_JOIN_KEY, TERM_RANGE,
    TERM_ROWS_MATCHED, TERM_ROWS_SCANNED, WorkloadObserver, column_term,
};

/// Cardinality at or below which interleaving is the cheap right answer.
const LOW_CARDINALITY: u64 = 256;
/// Cardinality at or below which Hilbert's locality pays for its cost.
const MODERATE_CARDINALITY: u64 = 65_536;
/// Above this a column is treated as effectively unique.
const HIGH_CARDINALITY: u64 = 1 << 20;
/// Distinct-to-row ratio at which a column is near-unique.
const NEAR_UNIQUE_RATIO: f64 = 0.5;
/// A column this null is not worth ordering by.
const MAX_NULL_FRACTION: f64 = 0.9;

/// What the writer and the observer between them know about one column.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ColumnEvidence {
    pub column_id: u32,
    pub type_id: TypeId,
    /// Distinct values, exact from the writer's per-file statistics
    pub ndv: u64,
    pub row_count: u64,
    pub null_fraction: f64,
    /// Decayed weight of equality predicates on this column
    pub equality_weight: f64,
    /// Decayed weight of range predicates on this column
    pub range_weight: f64,
    /// Decayed weight of joins whose equality key reached this column.
    ///
    /// Separate from `equality_weight` because a join key carries no
    /// constant, so it prunes file pairs rather than files, and because it
    /// only pays off when the other side is ordered by its half of the key
    /// too
    pub join_weight: f64,
}

impl ColumnEvidence {
    /// How much of the workload touches this column at all.
    ///
    /// Joins count at full weight beside filters. A join key genuinely
    /// prunes: with both sides ordered by it, a file on one side only has
    /// to be read against the files on the other whose key ranges overlap
    /// it, and the rest of the pairs are rejected from the manifest with no
    /// IO. That is the same kind of saving an equality filter buys, at the
    /// granularity of file pairs rather than files
    pub fn total_weight(&self) -> f64 {
        self.equality_weight.max(0.0) + self.range_weight.max(0.0) + self.join_weight.max(0.0)
    }

    /// Whether joins are the main reason to order by this column.
    ///
    /// Decides the curve rather than the ranking: a join needs the two
    /// sides' key ranges to be comparable, and only a contiguous range per
    /// file gives that
    fn is_join_key(&self) -> bool {
        self.join_weight > 0.0
            && self.join_weight >= self.equality_weight.max(0.0) + self.range_weight.max(0.0)
    }

    fn is_temporal(&self) -> bool {
        matches!(
            self.type_id,
            TypeId::Date | TypeId::Time | TypeId::Timestamp | TypeId::TimestampTz
        )
    }
}

/// The curve a column's measured shape asks for, or None when the column is
/// not worth ordering by at all.
pub fn choose_strategy(evidence: &ColumnEvidence) -> Option<ClusterStrategy> {
    if evidence.null_fraction > MAX_NULL_FRACTION {
        return None;
    }
    if evidence.ndv <= 1 {
        // One value orders nothing
        return None;
    }
    let ratio = if evidence.row_count == 0 {
        0.0
    } else {
        evidence.ndv as f64 / evidence.row_count as f64
    };
    if ratio >= NEAR_UNIQUE_RATIO || evidence.ndv > HIGH_CARDINALITY {
        return Some(ClusterStrategy::RangePartition);
    }
    // A column the workload mostly joins on needs contiguous ranges,
    // whatever its cardinality says. The interleaving and space-filling
    // curves scatter a single key across the file set on purpose, which is
    // right for a multi-column filter and wrong here: the two sides' ranges
    // stop being comparable, so no file pair can be rejected and the join
    // reads everything against everything
    if evidence.is_join_key() {
        return Some(ClusterStrategy::RangePartition);
    }
    if evidence.is_temporal() && evidence.range_weight >= evidence.equality_weight {
        return Some(ClusterStrategy::RangePartition);
    }
    if evidence.ndv <= LOW_CARDINALITY {
        return Some(ClusterStrategy::BitInterleave);
    }
    if evidence.ndv <= MODERATE_CARDINALITY {
        return Some(ClusterStrategy::SpaceFilling);
    }
    Some(ClusterStrategy::RangePartition)
}

/// Proposes cluster keys from measurement, anchors first.
///
/// Anchors are the columns an operator pinned, and they keep their declared
/// order at the front whatever the evidence says: Hybrid mode means the
/// operator knows something the measurements do not, and the planner fills
/// in around that rather than overruling it.
///
/// Everything after them is ordered by measured weight, heaviest first, so
/// the column the workload filters on most becomes the leading free key.
/// A column no query touches is not proposed at all.
pub fn propose(evidence: &[ColumnEvidence], anchors: &[u32], max_keys: usize) -> Vec<ClusterKey> {
    let mut keys: Vec<ClusterKey> = Vec::with_capacity(max_keys.min(evidence.len()));

    for anchor in anchors {
        let Some(column) = evidence.iter().find(|c| c.column_id == *anchor) else {
            continue;
        };
        // A pinned column is used even when the evidence would have dropped
        // it, with the closest defensible curve
        let strategy = choose_strategy(column).unwrap_or(ClusterStrategy::RangePartition);
        keys.push(ClusterKey {
            column_id: *anchor,
            strategy,
            param: 0,
        });
    }

    let mut ranked: Vec<&ColumnEvidence> = evidence
        .iter()
        .filter(|c| !anchors.contains(&c.column_id) && c.total_weight() > 0.0)
        .collect();
    ranked.sort_by(|a, b| {
        b.total_weight()
            .total_cmp(&a.total_weight())
            .then_with(|| a.column_id.cmp(&b.column_id))
    });

    for column in ranked {
        if keys.len() >= max_keys {
            break;
        }
        let Some(strategy) = choose_strategy(column) else {
            continue;
        };
        keys.push(ClusterKey {
            column_id: column.column_id,
            strategy,
            param: 0,
        });
    }
    keys.truncate(max_keys);
    keys
}

/// What a table clusters by before it has any observations.
///
/// Declared keys win outright. Otherwise the primary key, then the leading
/// columns of the most selective unique constraint, then a temporal column,
/// then nothing. This also makes constraint enforcement cheap, since a
/// primary key that is a cluster key means min/max eliminates nearly every
/// file before any bloom probe.
pub fn bootstrap(
    declared: &[ClusterKey],
    primary_key: &[u32],
    unique_keys: &[Vec<u32>],
    temporal: &[u32],
    max_keys: usize,
) -> Vec<ClusterKey> {
    let take = |columns: &[u32]| -> Vec<ClusterKey> {
        columns
            .iter()
            .take(max_keys)
            .map(|column_id| ClusterKey {
                column_id: *column_id,
                strategy: ClusterStrategy::RangePartition,
                param: 0,
            })
            .collect()
    };

    if !declared.is_empty() {
        let mut keys = declared.to_vec();
        keys.truncate(max_keys);
        return keys;
    }
    if !primary_key.is_empty() {
        return take(primary_key);
    }
    // The most selective unique constraint is the narrowest one: fewer
    // columns means each one carries more of the key
    if let Some(narrowest) = unique_keys
        .iter()
        .filter(|k| !k.is_empty())
        .min_by_key(|k| k.len())
    {
        return take(narrowest);
    }
    if !temporal.is_empty() {
        return take(&temporal[..1]);
    }
    Vec::new()
}

/// Builds one evidence record per schema column from the manifest's
/// per-file statistics and the observer's counters.
///
/// Statistics aggregate per file rather than merging across files.
/// Distinct counts are not mergeable from stored estimates, and per-file
/// shape is the right question anyway: a curve orders rows inside a file
/// and separates one file from the next, so what decides the curve is how
/// many distinct values a typical file holds. A column with ten values
/// reads as ten whether it sits in one file or a thousand.
///
/// Files are weighted by their size, so a large file's shape counts for
/// more than a small one's, matching how the gate scores skip rate.
/// A column no file carries statistics for produces no evidence rather
/// than evidence of zero, because those are different states.
pub fn evidence_from_manifest(
    manifest: &ManifestFile,
    observer: &WorkloadObserver,
    table_id: u32,
    now: u16,
) -> Vec<ColumnEvidence> {
    let mut out = Vec::with_capacity(manifest.schema.columns.len());
    for column in &manifest.schema.columns {
        let mut weight = 0f64;
        let mut ndv_sum = 0f64;
        let mut rows_sum = 0f64;
        let mut nulls_sum = 0f64;
        for entry in &manifest.entries {
            let Some(stats) = entry.stats_for(column.id) else {
                continue;
            };
            let Some(ndv) = stats.ndv else {
                continue;
            };
            // A zero-byte file would drop out of a size weighting entirely,
            // so it counts as one byte rather than as nothing
            let file_weight = entry.size_bytes.max(1) as f64;
            weight += file_weight;
            ndv_sum += file_weight * ndv as f64;
            rows_sum += file_weight * entry.row_count as f64;
            nulls_sum += file_weight * stats.bounds.null_count as f64;
        }
        if weight == 0.0 {
            continue;
        }
        let row_count = (rows_sum / weight).round() as u64;
        let null_fraction = if row_count == 0 {
            0.0
        } else {
            (nulls_sum / weight) / row_count as f64
        };
        out.push(ColumnEvidence {
            column_id: column.id,
            type_id: column.type_id,
            ndv: (ndv_sum / weight).round() as u64,
            row_count,
            null_fraction: null_fraction.clamp(0.0, 1.0),
            equality_weight: observer.score(table_id, column_term(column.id, TERM_EQUALITY), now),
            range_weight: observer.score(table_id, column_term(column.id, TERM_RANGE), now),
            join_weight: observer.score(table_id, column_term(column.id, TERM_JOIN_KEY), now),
        });
    }
    out
}

/// The byte-weighted skip rate the engine actually measured for scans
/// filtering on one column, or None when no such scan has run.
///
/// This is the ground truth the gate checks its replay against. A replay
/// that disagrees with it is a model that has drifted, and a drifted model
/// does not get to drive a rewrite.
pub fn measured_skip_rate(
    observer: &WorkloadObserver,
    table_id: u32,
    column_id: u32,
    now: u16,
) -> Option<f64> {
    let considered = observer.score(table_id, column_term(column_id, TERM_BYTES_CONSIDERED), now);
    if considered <= 0.0 {
        return None;
    }
    let skipped = observer.score(table_id, column_term(column_id, TERM_BYTES_SKIPPED), now);
    Some((skipped / considered).clamp(0.0, 1.0))
}

/// The fraction of scanned rows the workload's predicates on one column
/// actually returned, or None when no scan has finished.
///
/// This is what places a replay probe. Skip rate says how well the layout
/// serves a predicate, selectivity says how much of the table it wants,
/// and only the second one identifies where the real constants sit.
pub fn measured_selectivity(
    observer: &WorkloadObserver,
    table_id: u32,
    column_id: u32,
    now: u16,
) -> Option<f64> {
    let scanned = observer.score(table_id, column_term(column_id, TERM_ROWS_SCANNED), now);
    if scanned <= 0.0 {
        return None;
    }
    let matched = observer.score(table_id, column_term(column_id, TERM_ROWS_MATCHED), now);
    Some((matched / scanned).clamp(0.0, 1.0))
}

/// Builds the gate's predicate classes from measured evidence.
///
/// The observer records which columns are compared and how, never the
/// constants a query used, so replay has to supply one. A stand-in picked
/// arbitrarily is worse than useless: a probe an order of magnitude more
/// selective than the real workload scores a skip rate the workload never
/// sees, and the gate rightly refuses to trust it.
///
/// So the constant is placed rather than guessed. Measured selectivity
/// says what fraction of the table the workload's predicates on this
/// column keep, and the probe is the value at that quantile of the
/// column's observed range. The gate's own model check then compares the
/// probe's replayed skip rate against the measured one, and a probe that
/// cannot reproduce what was measured is refused, which is correct.
pub fn predicate_classes(
    manifest: &ManifestFile,
    evidence: &[ColumnEvidence],
    observer: &WorkloadObserver,
    table_id: u32,
    now: u16,
) -> Vec<PredicateClass> {
    let mut classes = Vec::new();
    for column in evidence {
        let measured = measured_skip_rate(observer, table_id, column.column_id, now);
        // With no finished scan to learn from, the probe splits the range,
        // which is the least assuming placement available
        let selectivity =
            measured_selectivity(observer, table_id, column.column_id, now).unwrap_or(0.5);
        let Some(value) = probe_value(manifest, column.column_id, selectivity) else {
            continue;
        };
        for (weight, op) in [
            (column.equality_weight, CompareOp::Eq),
            (column.range_weight, CompareOp::Lt),
        ] {
            if weight <= 0.0 {
                continue;
            }
            classes.push(PredicateClass {
                predicate: LakePredicate::Compare {
                    column_id: column.column_id,
                    op,
                    value: value.clone(),
                },
                weight,
                measured_skip_rate: measured,
            });
        }
    }
    classes
}

/// The value at one quantile of a column's observed range.
///
/// Numeric ranges interpolate, which places the probe where the data
/// actually is rather than where its file boundaries happen to fall. Types
/// with no arithmetic fall back to the quantile of the per-file bounds,
/// which is coarser but still a value the data holds.
fn probe_value(manifest: &ManifestFile, column_id: u32, quantile: f64) -> Option<LakeValue> {
    let q = quantile.clamp(0.0, 1.0);
    let mut low: Option<&LakeValue> = None;
    let mut high: Option<&LakeValue> = None;
    let mut bounds: Vec<&LakeValue> = Vec::new();
    for entry in &manifest.entries {
        let Some(stats) = entry.stats_for(column_id) else {
            continue;
        };
        if let Some(min) = stats.bounds.min.as_ref() {
            bounds.push(min);
            if low
                .map(|l| min.compare(l) == Some(std::cmp::Ordering::Less))
                .unwrap_or(true)
            {
                low = Some(min);
            }
        }
        if let Some(max) = stats.bounds.max.as_ref() {
            bounds.push(max);
            if high
                .map(|h| max.compare(h) == Some(std::cmp::Ordering::Greater))
                .unwrap_or(true)
            {
                high = Some(max);
            }
        }
    }
    if let (Some(low), Some(high)) = (low, high) {
        if let Some(interpolated) = interpolate(low, high, q) {
            return Some(interpolated);
        }
    }
    if bounds.is_empty() {
        return None;
    }
    bounds.sort_by(|a, b| a.compare(b).unwrap_or(std::cmp::Ordering::Equal));
    bounds.dedup_by(|a, b| a.compare(b) == Some(std::cmp::Ordering::Equal));
    let index = ((bounds.len() - 1) as f64 * q).round() as usize;
    bounds.get(index).map(|v| (*v).clone())
}

/// Linear interpolation between two values of the same numeric variant.
/// None for anything without arithmetic, which falls back to the bounds
fn interpolate(low: &LakeValue, high: &LakeValue, q: f64) -> Option<LakeValue> {
    Some(match (low, high) {
        (LakeValue::Int(a), LakeValue::Int(b)) => LakeValue::Int(a + ((b - a) as f64 * q) as i64),
        (LakeValue::UInt(a), LakeValue::UInt(b)) => {
            LakeValue::UInt(a + ((b - a) as f64 * q) as u64)
        }
        (LakeValue::Int128(a), LakeValue::Int128(b)) => {
            LakeValue::Int128(a + ((b - a) as f64 * q) as i128)
        }
        (LakeValue::UInt128(a), LakeValue::UInt128(b)) => {
            LakeValue::UInt128(a + ((b - a) as f64 * q) as u128)
        }
        (LakeValue::Float(a), LakeValue::Float(b)) => LakeValue::Float(a + (b - a) * q),
        _ => return None,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn column(column_id: u32, ndv: u64, rows: u64) -> ColumnEvidence {
        ColumnEvidence {
            column_id,
            type_id: TypeId::Int64,
            ndv,
            row_count: rows,
            null_fraction: 0.0,
            equality_weight: 1.0,
            range_weight: 0.0,
            join_weight: 0.0,
        }
    }

    #[test]
    fn test_strategy_follows_the_measured_shape_of_the_column() {
        // Near unique: a range predicate should read one run
        assert_eq!(
            choose_strategy(&column(0, 900, 1_000)),
            Some(ClusterStrategy::RangePartition)
        );
        // Very high cardinality, whatever the ratio
        assert_eq!(
            choose_strategy(&column(0, (1 << 20) + 1, 100_000_000)),
            Some(ClusterStrategy::RangePartition)
        );
        // Low cardinality leaves room for the other dimensions
        assert_eq!(
            choose_strategy(&column(0, 12, 1_000_000)),
            Some(ClusterStrategy::BitInterleave)
        );
        // Moderate: Hilbert's locality is worth its cost
        assert_eq!(
            choose_strategy(&column(0, 5_000, 10_000_000)),
            Some(ClusterStrategy::SpaceFilling)
        );
        // Between moderate and high, the safe default
        assert_eq!(
            choose_strategy(&column(0, 100_000, 100_000_000)),
            Some(ClusterStrategy::RangePartition)
        );
    }

    #[test]
    fn test_a_column_with_nothing_to_order_by_is_dropped() {
        let mut mostly_null = column(0, 500, 1_000_000);
        mostly_null.null_fraction = 0.95;
        assert_eq!(choose_strategy(&mostly_null), None);

        // One distinct value orders nothing
        assert_eq!(choose_strategy(&column(0, 1, 1_000_000)), None);
        assert_eq!(choose_strategy(&column(0, 0, 1_000_000)), None);
    }

    #[test]
    fn test_a_temporal_column_queried_by_range_takes_range_partition() {
        let mut ts = column(0, 5_000, 10_000_000);
        ts.type_id = TypeId::Timestamp;
        ts.range_weight = 10.0;
        ts.equality_weight = 1.0;
        assert_eq!(
            choose_strategy(&ts),
            Some(ClusterStrategy::RangePartition),
            "a time range must read one run of files"
        );

        // The same column queried by equality falls back to the size rule
        ts.range_weight = 0.0;
        ts.equality_weight = 10.0;
        assert_eq!(choose_strategy(&ts), Some(ClusterStrategy::SpaceFilling));
    }

    #[test]
    fn test_keys_are_ordered_by_what_the_workload_filters_on() {
        let mut heavy = column(1, 10, 1_000_000);
        heavy.equality_weight = 100.0;
        let mut light = column(2, 10, 1_000_000);
        light.equality_weight = 1.0;
        let untouched = ColumnEvidence {
            equality_weight: 0.0,
            range_weight: 0.0,
            ..column(3, 10, 1_000_000)
        };

        let keys = propose(&[light, heavy, untouched], &[], 4);
        assert_eq!(keys.len(), 2, "a column no query touches is not proposed");
        assert_eq!(keys[0].column_id, 1, "the heaviest column leads");
        assert_eq!(keys[1].column_id, 2);
    }

    #[test]
    fn test_anchors_keep_their_place_and_their_column() {
        let mut heavy = column(1, 10, 1_000_000);
        heavy.equality_weight = 100.0;
        let mut anchored = column(9, 900, 1_000);
        anchored.equality_weight = 0.0;
        anchored.range_weight = 0.0;

        let keys = propose(&[heavy, anchored], &[9], 4);
        assert_eq!(keys[0].column_id, 9, "the pinned column leads regardless");
        assert_eq!(keys[0].strategy, ClusterStrategy::RangePartition);
        assert_eq!(keys[1].column_id, 1, "measurement fills in after it");

        // An anchor the evidence would have dropped is still used
        let mut null_heavy = column(9, 900, 1_000);
        null_heavy.null_fraction = 0.99;
        let keys = propose(&[null_heavy], &[9], 4);
        assert_eq!(keys.len(), 1);
        assert_eq!(keys[0].column_id, 9);
    }

    #[test]
    fn test_max_keys_bounds_the_proposal() {
        let evidence: Vec<ColumnEvidence> = (0..8)
            .map(|i| {
                let mut c = column(i, 10, 1_000_000);
                c.equality_weight = (8 - i) as f64;
                c
            })
            .collect();
        let keys = propose(&evidence, &[], 3);
        assert_eq!(keys.len(), 3);
        assert_eq!(keys[0].column_id, 0, "heaviest first");
    }

    #[test]
    fn test_bootstrap_prefers_what_the_table_already_declares() {
        let declared = vec![ClusterKey {
            column_id: 7,
            strategy: ClusterStrategy::BitInterleave,
            param: 0,
        }];
        let keys = bootstrap(&declared, &[1], &[vec![2]], &[3], 4);
        assert_eq!(keys, declared, "a declared key is not second-guessed");

        // Then the primary key
        let keys = bootstrap(&[], &[1, 2], &[vec![5]], &[3], 4);
        assert_eq!(keys.len(), 2);
        assert_eq!(keys[0].column_id, 1);

        // Then the narrowest unique constraint, which is the most selective
        let keys = bootstrap(&[], &[], &[vec![5, 6, 7], vec![9]], &[3], 4);
        assert_eq!(keys.len(), 1);
        assert_eq!(keys[0].column_id, 9);

        // Then a temporal column
        let keys = bootstrap(&[], &[], &[], &[3, 4], 4);
        assert_eq!(keys.len(), 1);
        assert_eq!(keys[0].column_id, 3);

        // Then nothing, rather than an arbitrary column
        assert!(bootstrap(&[], &[], &[], &[], 4).is_empty());
    }

    /// A join is a reason to order a table by a column, and it has to
    /// outrank a column nothing touches. Before joins were observed, a
    /// table joined on a column a thousand times a minute looked exactly
    /// like a table nobody read
    #[test]
    fn test_a_join_key_is_proposed_over_a_column_nothing_touches() {
        let mut joined = column(7, 5_000, 10_000);
        joined.equality_weight = 0.0;
        joined.join_weight = 4.0;
        let mut filtered = column(3, 5_000, 10_000);
        filtered.equality_weight = 1.0;
        let mut untouched = column(9, 5_000, 10_000);
        untouched.equality_weight = 0.0;

        let keys = propose(&[untouched, filtered, joined], &[], 4);
        assert_eq!(
            keys.first().map(|k| k.column_id),
            Some(7),
            "the heaviest signal leads, and a join key is a signal"
        );
        assert!(
            keys.iter().all(|k| k.column_id != 9),
            "a column with no weight of any kind is still not proposed"
        );
    }

    /// A join needs the two sides' key ranges to be comparable, and only a
    /// contiguous range per file gives that. The interleaving curves are
    /// right for a multi-column filter and wrong here: they scatter one key
    /// across the file set on purpose, so no file pair can be rejected
    #[test]
    fn test_a_join_key_takes_contiguous_ranges_whatever_its_cardinality() {
        // Low cardinality, which without a join would interleave
        let mut low = column(1, 8, 100_000);
        low.equality_weight = 1.0;
        low.join_weight = 0.0;
        assert_eq!(choose_strategy(&low), Some(ClusterStrategy::BitInterleave));

        low.join_weight = 2.0;
        assert_eq!(
            choose_strategy(&low),
            Some(ClusterStrategy::RangePartition),
            "a column the workload mostly joins on has to keep comparable ranges"
        );

        // Moderate cardinality, which without a join would take a space
        // filling curve
        let mut moderate = column(2, 5_000, 100_000);
        moderate.equality_weight = 1.0;
        assert_eq!(
            choose_strategy(&moderate),
            Some(ClusterStrategy::SpaceFilling)
        );
        moderate.join_weight = 2.0;
        assert_eq!(
            choose_strategy(&moderate),
            Some(ClusterStrategy::RangePartition)
        );
    }

    /// A column joined once and filtered constantly is a filter column.
    /// The curve follows whichever signal dominates, because a filter on
    /// several columns is served by interleaving and a join is not
    #[test]
    fn test_a_column_that_is_mostly_filtered_keeps_its_filter_curve() {
        let mut mixed = column(1, 8, 100_000);
        mixed.equality_weight = 10.0;
        mixed.join_weight = 1.0;
        assert_eq!(
            choose_strategy(&mixed),
            Some(ClusterStrategy::BitInterleave),
            "one join against ten filters does not make it a join key"
        );
    }

    /// A column nothing joins on is unaffected, which is what keeps this
    /// from moving every existing layout
    #[test]
    fn test_join_weight_of_zero_changes_nothing() {
        let mut plain = column(1, 8, 100_000);
        plain.equality_weight = 1.0;
        plain.join_weight = 0.0;
        assert_eq!(plain.total_weight(), 1.0);
        assert_eq!(
            choose_strategy(&plain),
            Some(ClusterStrategy::BitInterleave)
        );
    }
}
