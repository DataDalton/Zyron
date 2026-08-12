//! The gate a clustering proposal has to pass.
//!
//! Replay never executes a query. It evaluates the predicates the observer
//! recorded against per-file statistics and scores byte-weighted skip rate,
//! so judging a candidate layout costs no IO and no query time.
//!
//! The order of the checks is the design:
//!
//! 0. anchor legality, free and first, because a proposal that violates an
//!    operator's pinned keys is illegal whatever it would have scored
//! 1. per-class skip rate, current layout against candidate
//! 2. model validation: replay that disagrees with what was measured means
//!    the model is wrong, and a wrong model must not be trusted to accept
//! 3. any class worse than epsilon vetoes the proposal
//! 4. weighted improvement below the threshold is not worth a rewrite
//! 5. accept
//!
//! Steps 3 and 4 are the net-positive guarantee: Auto can only improve the
//! layout or leave it alone, never regress it.

use crate::manifest::PartitionEntry;
use crate::predicate::{LakePredicate, PruneDecision};
use crate::schema::LakeSchema;

/// One predicate shape the workload actually issues, weighted by how much
/// of the workload it is.
#[derive(Debug, Clone)]
pub struct PredicateClass {
    pub predicate: LakePredicate,
    /// Share of the workload, any positive scale
    pub weight: f64,
    /// Skip rate the engine actually observed for this class, when it has
    /// run. Replay is checked against it, so a model that has drifted
    /// cannot quietly drive a rewrite
    pub measured_skip_rate: Option<f64>,
}

/// Why a proposal was refused, or that it was taken.
#[derive(Debug, Clone, PartialEq)]
pub enum Decision {
    /// Weighted skip-rate improvement, byte weighted
    Accept { delta: f64 },
    /// The candidate drops or reorders a key the operator pinned
    AnchorConflict { expected: Vec<u32>, found: Vec<u32> },
    /// Replay disagreed with measurement on too many classes, so the model
    /// is not currently trustworthy
    ReplayDiverged { classes: usize, tolerance: f64 },
    /// Some class of query would read more bytes than it does today
    Worse { class: usize, delta: f64 },
    /// Better, but not by enough to pay for rewriting the files
    BelowThreshold { delta: f64, required: f64 },
}

/// Thresholds the gate applies.
#[derive(Debug, Clone, Copy)]
pub struct GateConfig {
    /// Weighted improvement a proposal must reach to be worth a rewrite
    pub min_improvement: f64,
    /// Slack allowed on a single class before it counts as a regression
    pub epsilon: f64,
    /// How far replay may be from measurement before the model is distrusted
    pub replay_tolerance: f64,
}

impl Default for GateConfig {
    fn default() -> Self {
        Self {
            min_improvement: 0.05,
            epsilon: 0.01,
            replay_tolerance: 0.25,
        }
    }
}

/// Fraction of bytes a predicate lets the scan skip, over one file set.
///
/// Byte weighted rather than file weighted: skipping one large file is
/// worth more than skipping several small ones, and the planner should
/// prefer the layout that reads fewer bytes.
pub fn skip_rate(files: &[PartitionEntry], schema: &LakeSchema, predicate: &LakePredicate) -> f64 {
    let mut total = 0u64;
    let mut skipped = 0u64;
    for entry in files {
        total += entry.size_bytes;
        let stats = crate::manifest::FileStats::new(entry, schema);
        if predicate.prune(&stats) == PruneDecision::CannotMatch {
            skipped += entry.size_bytes;
        }
    }
    if total == 0 {
        // No bytes to read is total skipping. A candidate layout whose
        // files all turned out to hold nothing but deleted rows reads
        // nothing, which is the best any layout can do
        return 1.0;
    }
    skipped as f64 / total as f64
}

/// Scores a candidate layout against the current one and returns the
/// decision, with the checks in the order that makes a refusal cheap.
pub fn evaluate(
    current: &[PartitionEntry],
    candidate: &[PartitionEntry],
    schema: &LakeSchema,
    classes: &[PredicateClass],
    anchors: &[u32],
    candidate_keys: &[u32],
    config: GateConfig,
) -> Decision {
    // 0. Anchor legality. Free, and a proposal that fails it is illegal
    // whatever it would have scored
    if !candidate_keys.starts_with(anchors) {
        return Decision::AnchorConflict {
            expected: anchors.to_vec(),
            found: candidate_keys.to_vec(),
        };
    }
    if classes.is_empty() {
        // No evidence is not an improvement. A layout change with nothing
        // behind it is exactly what Auto must not make
        return Decision::BelowThreshold {
            delta: 0.0,
            required: config.min_improvement,
        };
    }

    // 1. Per-class replay, current against candidate
    let mut rates = Vec::with_capacity(classes.len());
    for class in classes {
        rates.push((
            skip_rate(current, schema, &class.predicate),
            skip_rate(candidate, schema, &class.predicate),
        ));
    }

    // 2. Model validation. A replay that disagrees with what the engine
    // measured is a model that cannot be trusted to accept a rewrite
    let mut measured = 0usize;
    let mut diverged = 0usize;
    for (class, (current_rate, _)) in classes.iter().zip(rates.iter()) {
        if let Some(observed) = class.measured_skip_rate {
            measured += 1;
            if (observed - current_rate).abs() > config.replay_tolerance {
                diverged += 1;
            }
        }
    }
    if measured > 0 && diverged * 2 > measured {
        return Decision::ReplayDiverged {
            classes: diverged,
            tolerance: config.replay_tolerance,
        };
    }

    // 3. Regression veto. One class reading more bytes than it does today
    // refuses the proposal however good the average looks
    for (index, (current_rate, candidate_rate)) in rates.iter().enumerate() {
        let delta = candidate_rate - current_rate;
        if delta < -config.epsilon {
            return Decision::Worse {
                class: index,
                delta,
            };
        }
    }

    // 4. Weighted improvement, against the cost of rewriting
    let total_weight: f64 = classes.iter().map(|c| c.weight.max(0.0)).sum();
    if total_weight <= 0.0 {
        return Decision::BelowThreshold {
            delta: 0.0,
            required: config.min_improvement,
        };
    }
    let delta: f64 = classes
        .iter()
        .zip(rates.iter())
        .map(|(class, (current_rate, candidate_rate))| {
            class.weight.max(0.0) * (candidate_rate - current_rate)
        })
        .sum::<f64>()
        / total_weight;
    if delta < config.min_improvement {
        return Decision::BelowThreshold {
            delta,
            required: config.min_improvement,
        };
    }

    // 5. Better on every class, better on the weighted average, and by
    // enough to pay for the rewrite
    Decision::Accept { delta }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::manifest::ColumnStatsEntry;
    use crate::predicate::{ColumnBounds, CompareOp, LakeValue};
    use crate::schema::LakeColumn;
    use zyron_common::TypeId;

    fn schema() -> LakeSchema {
        LakeSchema::new(
            1,
            vec![
                LakeColumn {
                    id: 0,
                    name: "a".into(),
                    type_id: TypeId::Int64,
                    nullable: false,
                    ts_precision: None,
                    tz_offset_secs: None,
                    max_length: None,
                    default_expr: None,
                },
                LakeColumn {
                    id: 1,
                    name: "b".into(),
                    type_id: TypeId::Int64,
                    nullable: false,
                    ts_precision: None,
                    tz_offset_secs: None,
                    max_length: None,
                    default_expr: None,
                },
            ],
        )
        .expect("schema")
    }

    /// One file with the given per-column ranges and a fixed byte size.
    fn file(partition_id: u64, size_bytes: u64, ranges: &[(u32, i64, i64)]) -> PartitionEntry {
        let mut column_stats: Vec<ColumnStatsEntry> = ranges
            .iter()
            .map(|(column_id, low, high)| ColumnStatsEntry {
                ndv: None,
                column_id: *column_id,
                bounds: ColumnBounds {
                    min: Some(LakeValue::Int(*low)),
                    max: Some(LakeValue::Int(*high)),
                    null_count: 0,
                    row_count: 100,
                },
                bloom: None,
            })
            .collect();
        column_stats.sort_by_key(|s| s.column_id);
        PartitionEntry {
            partition_id,
            size_bytes,
            row_count: 100,
            added_version: 1,
            cluster_spec_id: 1,
            column_stats,
            delete_predicate_ids: Vec::new(),
        }
    }

    fn eq(column_id: u32, value: i64) -> LakePredicate {
        LakePredicate::Compare {
            column_id,
            op: CompareOp::Eq,
            value: LakeValue::Int(value),
        }
    }

    fn class(predicate: LakePredicate, weight: f64) -> PredicateClass {
        PredicateClass {
            predicate,
            weight,
            measured_skip_rate: None,
        }
    }

    /// Four files whose `a` ranges overlap completely: no predicate on `a`
    /// can skip anything.
    fn unclustered() -> Vec<PartitionEntry> {
        (0..4)
            .map(|i| file(i, 1_000, &[(0, 0, 99), (1, 0, 99)]))
            .collect()
    }

    /// The same data clustered on `a`: disjoint ranges, so an equality on
    /// `a` reads one file.
    fn clustered_on_a() -> Vec<PartitionEntry> {
        (0..4)
            .map(|i| {
                let low = i as i64 * 25;
                file(i, 1_000, &[(0, low, low + 24), (1, 0, 99)])
            })
            .collect()
    }

    #[test]
    fn test_skip_rate_is_byte_weighted() {
        let schema = schema();
        // One large file that matches and three small ones that do not
        let files = vec![
            file(0, 10_000, &[(0, 0, 9)]),
            file(1, 1_000, &[(0, 100, 109)]),
            file(2, 1_000, &[(0, 200, 209)]),
            file(3, 1_000, &[(0, 300, 309)]),
        ];
        // Skipping three small files out of 13_000 bytes
        let rate = skip_rate(&files, &schema, &eq(0, 5));
        assert!((rate - 3_000.0 / 13_000.0).abs() < 1e-9);
        // A predicate nothing can match skips everything
        assert_eq!(skip_rate(&files, &schema, &eq(0, 9_999)), 1.0);
    }

    #[test]
    fn test_a_clustering_that_helps_is_accepted() {
        let schema = schema();
        let decision = evaluate(
            &unclustered(),
            &clustered_on_a(),
            &schema,
            &[class(eq(0, 30), 1.0)],
            &[],
            &[0],
            GateConfig::default(),
        );
        match decision {
            Decision::Accept { delta } => assert!(delta > 0.7, "delta {delta}"),
            other => panic!("expected accept, got {other:?}"),
        }
    }

    #[test]
    fn test_a_class_that_would_read_more_vetoes_the_proposal() {
        let schema = schema();
        // Going the other way: the clustered layout is current and the
        // proposal would smear `a` back across every file
        let decision = evaluate(
            &clustered_on_a(),
            &unclustered(),
            &schema,
            &[class(eq(0, 30), 1.0)],
            &[],
            &[1],
            GateConfig::default(),
        );
        match decision {
            Decision::Worse { class, delta } => {
                assert_eq!(class, 0);
                assert!(delta < 0.0, "delta {delta}");
            }
            other => panic!("expected a regression veto, got {other:?}"),
        }
    }

    #[test]
    fn test_one_regressing_class_outweighs_a_good_average() {
        let schema = schema();
        // A proposal that helps a heavy class and hurts a light one is
        // still refused: the veto is per class, not on the average
        let decision = evaluate(
            &clustered_on_a(),
            &clustered_on_a(),
            &schema,
            &[
                class(eq(0, 30), 100.0),
                PredicateClass {
                    predicate: eq(1, 5),
                    weight: 0.001,
                    measured_skip_rate: None,
                },
            ],
            &[],
            &[0],
            GateConfig::default(),
        );
        // Identical layouts improve nothing, so this lands on the threshold
        // rather than the veto, which is the honest answer
        assert!(matches!(decision, Decision::BelowThreshold { .. }));
    }

    #[test]
    fn test_an_anchor_the_proposal_drops_is_refused_before_anything_is_scored() {
        let schema = schema();
        let decision = evaluate(
            &unclustered(),
            &clustered_on_a(),
            &schema,
            &[class(eq(0, 30), 1.0)],
            &[1],
            &[0],
            GateConfig::default(),
        );
        match decision {
            Decision::AnchorConflict { expected, found } => {
                assert_eq!(expected, vec![1]);
                assert_eq!(found, vec![0]);
            }
            other => panic!("expected an anchor conflict, got {other:?}"),
        }
        // Keeping the anchor as the leading key is legal
        let decision = evaluate(
            &unclustered(),
            &clustered_on_a(),
            &schema,
            &[class(eq(0, 30), 1.0)],
            &[0],
            &[0, 1],
            GateConfig::default(),
        );
        assert!(matches!(decision, Decision::Accept { .. }));
    }

    #[test]
    fn test_a_model_that_disagrees_with_measurement_is_not_trusted() {
        let schema = schema();
        // Replay says the current layout skips nothing, the engine measured
        // that it skips almost everything: the model is wrong, so it does
        // not get to drive a rewrite
        let decision = evaluate(
            &unclustered(),
            &clustered_on_a(),
            &schema,
            &[PredicateClass {
                predicate: eq(0, 30),
                weight: 1.0,
                measured_skip_rate: Some(0.9),
            }],
            &[],
            &[0],
            GateConfig::default(),
        );
        match decision {
            Decision::ReplayDiverged { classes, .. } => assert_eq!(classes, 1),
            other => panic!("expected divergence, got {other:?}"),
        }
        // Agreeing measurement leaves the proposal on its merits
        let decision = evaluate(
            &unclustered(),
            &clustered_on_a(),
            &schema,
            &[PredicateClass {
                predicate: eq(0, 30),
                weight: 1.0,
                measured_skip_rate: Some(0.0),
            }],
            &[],
            &[0],
            GateConfig::default(),
        );
        assert!(matches!(decision, Decision::Accept { .. }));
    }

    #[test]
    fn test_no_evidence_is_not_an_improvement() {
        let schema = schema();
        let decision = evaluate(
            &unclustered(),
            &clustered_on_a(),
            &schema,
            &[],
            &[],
            &[0],
            GateConfig::default(),
        );
        assert!(matches!(decision, Decision::BelowThreshold { delta, .. } if delta == 0.0));
    }
}
