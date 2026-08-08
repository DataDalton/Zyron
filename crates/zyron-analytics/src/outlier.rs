// Outlier detection methods: ZSCORE, IQR, MAD, and Isolation Forest.
// Each method operates over a numeric slice and returns either per-row
// scores or boolean flags. All methods are bounded-memory.

use crate::value::AnalyticsValue;
use zyron_common::PreHashMap;

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum OutlierDecision {
    Inlier,
    Outlier,
    // A NaN input cannot be ranked against a threshold so it is flagged as a
    // distinct non-inlier state rather than silently classified as an inlier
    Missing,
}

impl OutlierDecision {
    pub fn is_outlier(&self) -> bool {
        matches!(self, OutlierDecision::Outlier)
    }
}

// ===== Z-score =====
pub struct ZScoreEvaluator {
    pub mean: f64,
    pub stddev: f64,
}

impl ZScoreEvaluator {
    pub fn fit(values: &[f64]) -> Self {
        // Two-pass form: sum-then-variance. Each pass is a tight reduction
        // the compiler can auto-vectorise. Welford-style single-pass is
        // numerically more stable but has a serial dependency between mean
        // and m2 that blocks vectorisation, so it ends up several times
        // slower in practice for the data sizes the analytics engine
        // handles. Two-pass precision is more than adequate when the data
        // mean is not catastrophically larger than the variance (the
        // regime z-scoring is meaningful in to begin with).
        if values.is_empty() {
            return Self {
                mean: 0.0,
                stddev: 0.0,
            };
        }
        // Pass 1: count and sum non-NaN values
        let mut sum = 0.0f64;
        let mut n: u64 = 0;
        for &x in values {
            if !x.is_nan() {
                sum += x;
                n += 1;
            }
        }
        if n == 0 {
            return Self {
                mean: 0.0,
                stddev: 0.0,
            };
        }
        let mean = sum / n as f64;
        // Pass 2: sum of squared deviations
        let mut var_sum = 0.0f64;
        for &x in values {
            if !x.is_nan() {
                let d = x - mean;
                var_sum += d * d;
            }
        }
        let var = var_sum / n as f64;
        Self {
            mean,
            stddev: var.sqrt(),
        }
    }

    pub fn score(&self, x: f64) -> f64 {
        if self.stddev == 0.0 {
            0.0
        } else {
            (x - self.mean) / self.stddev
        }
    }
}

pub fn zscore(values: &[f64]) -> Vec<f64> {
    let z = ZScoreEvaluator::fit(values);
    values.iter().map(|v| z.score(*v)).collect()
}

// ===== IQR =====
pub fn iqr_outlier(values: &[f64], multiplier: f64) -> Vec<OutlierDecision> {
    if values.is_empty() {
        return Vec::new();
    }
    let mut sorted: Vec<f64> = values.iter().copied().filter(|v| !v.is_nan()).collect();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let q1 = quantile(&sorted, 0.25);
    let q3 = quantile(&sorted, 0.75);
    let iqr = q3 - q1;
    let lo = q1 - multiplier * iqr;
    let hi = q3 + multiplier * iqr;
    values
        .iter()
        .map(|v| {
            if v.is_nan() {
                OutlierDecision::Missing
            } else if *v < lo || *v > hi {
                OutlierDecision::Outlier
            } else {
                OutlierDecision::Inlier
            }
        })
        .collect()
}

fn quantile(sorted: &[f64], q: f64) -> f64 {
    if sorted.is_empty() {
        return 0.0;
    }
    let pos = q * (sorted.len() as f64 - 1.0);
    let lo = pos.floor() as usize;
    let hi = pos.ceil() as usize;
    let lo_v = sorted[lo];
    let hi_v = sorted[hi.min(sorted.len() - 1)];
    let frac = pos - lo as f64;
    lo_v + (hi_v - lo_v) * frac
}

// ===== MAD detector (Modified Z-score) =====
pub struct MadDetector {
    pub median: f64,
    pub mad: f64,
}

impl MadDetector {
    pub fn fit(values: &[f64]) -> Self {
        // Single working buffer: filter-collect non-NaN values once, sort
        // for the median, then mutate the same buffer in place into the
        // deviations array and sort again. Avoids the second N-sized
        // allocation that the naive form pays for.
        let mut buf: Vec<f64> = values.iter().copied().filter(|v| !v.is_nan()).collect();
        buf.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let median = quantile(&buf, 0.5);
        for x in buf.iter_mut() {
            *x = (*x - median).abs();
        }
        buf.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let mad = quantile(&buf, 0.5);
        Self { median, mad }
    }

    // Modified z-score using the 0.6745 scaling factor for normal data
    pub fn score(&self, x: f64) -> f64 {
        if self.mad == 0.0 {
            0.0
        } else {
            0.6745 * (x - self.median) / self.mad
        }
    }
}

pub fn mad_outlier(values: &[f64], threshold: f64) -> Vec<OutlierDecision> {
    let d = MadDetector::fit(values);
    values
        .iter()
        .map(|v| {
            if v.is_nan() {
                OutlierDecision::Missing
            } else if d.score(*v).abs() > threshold {
                OutlierDecision::Outlier
            } else {
                OutlierDecision::Inlier
            }
        })
        .collect()
}

// ===== Isolation Forest =====
// Standard isolation tree: at each split, choose a random feature and a
// random split value within the bounding box of the partition. Path length
// to a leaf is the anomaly signal.
pub struct IsolationForest {
    pub trees: Vec<IsolationTree>,
    pub sample_size: usize,
    pub n_features: usize,
}

pub struct IsolationTree {
    pub root: TreeNode,
    pub max_depth: u32,
}

pub enum TreeNode {
    Leaf {
        size: u32,
    },
    Internal {
        feature: usize,
        threshold: f64,
        left: Box<TreeNode>,
        right: Box<TreeNode>,
    },
}

// XorShift64 PRNG for deterministic, dependency free randomness
struct Xorshift64 {
    state: u64,
}

impl Xorshift64 {
    fn new(seed: u64) -> Self {
        Self {
            state: if seed == 0 { 0xdeadbeefcafebabe } else { seed },
        }
    }
    fn next(&mut self) -> u64 {
        let mut x = self.state;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.state = x;
        x
    }
    fn next_f64(&mut self) -> f64 {
        let bits = self.next() >> 11;
        bits as f64 / (1u64 << 53) as f64
    }
    fn next_in_range(&mut self, lo: usize, hi_exclusive: usize) -> usize {
        if hi_exclusive <= lo {
            return lo;
        }
        let span = (hi_exclusive - lo) as u64;
        lo + (self.next() % span) as usize
    }
}

impl IsolationTree {
    fn build(
        rows: &[Vec<f64>],
        indices: Vec<usize>,
        depth: u32,
        max_depth: u32,
        rng: &mut Xorshift64,
    ) -> TreeNode {
        if depth >= max_depth || indices.len() <= 1 {
            return TreeNode::Leaf {
                size: indices.len() as u32,
            };
        }
        let n_features = rows[indices[0]].len();
        if n_features == 0 {
            return TreeNode::Leaf {
                size: indices.len() as u32,
            };
        }
        let feature = rng.next_in_range(0, n_features);
        let mut lo = f64::INFINITY;
        let mut hi = f64::NEG_INFINITY;
        for &i in &indices {
            let v = rows[i][feature];
            if v < lo {
                lo = v;
            }
            if v > hi {
                hi = v;
            }
        }
        if lo == hi {
            return TreeNode::Leaf {
                size: indices.len() as u32,
            };
        }
        let threshold = lo + rng.next_f64() * (hi - lo);
        let mut left = Vec::new();
        let mut right = Vec::new();
        for i in indices {
            if rows[i][feature] < threshold {
                left.push(i);
            } else {
                right.push(i);
            }
        }
        TreeNode::Internal {
            feature,
            threshold,
            left: Box::new(Self::build(rows, left, depth + 1, max_depth, rng)),
            right: Box::new(Self::build(rows, right, depth + 1, max_depth, rng)),
        }
    }

    fn path_length(&self, row: &[f64]) -> f64 {
        Self::walk(&self.root, row, 0)
    }

    fn walk(node: &TreeNode, row: &[f64], depth: u32) -> f64 {
        match node {
            TreeNode::Leaf { size } => depth as f64 + c_factor(*size as usize),
            TreeNode::Internal {
                feature,
                threshold,
                left,
                right,
            } => {
                if row[*feature] < *threshold {
                    Self::walk(left, row, depth + 1)
                } else {
                    Self::walk(right, row, depth + 1)
                }
            }
        }
    }
}

// Average path length adjustment for an unsuccessful BST search
fn c_factor(n: usize) -> f64 {
    if n <= 1 {
        return 0.0;
    }
    let n = n as f64;
    2.0 * (n - 1.0).ln() + 0.5772156649 - 2.0 * (n - 1.0) / n
}

impl IsolationForest {
    pub fn fit(rows: &[Vec<f64>], n_trees: usize, sample_size: usize, seed: u64) -> Self {
        let mut rng = Xorshift64::new(seed);
        let n_features = rows.first().map(|r| r.len()).unwrap_or(0);
        let max_depth = (sample_size.max(2) as f64).log2().ceil() as u32;
        let mut trees = Vec::with_capacity(n_trees);
        for _ in 0..n_trees {
            // PreHashMap used as a set: u64 hash from the index avoids
            // re-hashing inside the table. Indices are pre-mixed so contiguous
            // values (0,1,2,...) land in spread-out buckets.
            let mut chosen: PreHashMap<u64, usize> = PreHashMap::default();
            let target = sample_size.min(rows.len());
            while chosen.len() < target && !rows.is_empty() {
                let idx = rng.next_in_range(0, rows.len());
                let key = zyron_common::fx_finalize(idx as u64);
                chosen.insert(key, idx);
            }
            let indices: Vec<usize> = chosen.into_values().collect();
            let root = IsolationTree::build(rows, indices, 0, max_depth, &mut rng);
            trees.push(IsolationTree { root, max_depth });
        }
        Self {
            trees,
            sample_size,
            n_features,
        }
    }

    // Anomaly score in [0, 1]. Closer to 1 means more anomalous.
    pub fn score(&self, row: &[f64]) -> f64 {
        if self.trees.is_empty() {
            return 0.0;
        }
        let avg: f64 =
            self.trees.iter().map(|t| t.path_length(row)).sum::<f64>() / self.trees.len() as f64;
        let cn = c_factor(self.sample_size);
        if cn == 0.0 {
            return 0.0;
        }
        2f64.powf(-avg / cn)
    }

    // Apply contamination threshold: rows with score >= threshold are
    // marked as outliers. Threshold is the score at the (1-c) quantile.
    pub fn predict(&self, rows: &[Vec<f64>], contamination: f64) -> Vec<OutlierDecision> {
        let scores: Vec<f64> = rows.iter().map(|r| self.score(r)).collect();
        let mut sorted = scores.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let cutoff_idx =
            ((1.0 - contamination).clamp(0.0, 1.0) * sorted.len() as f64).floor() as usize;
        let threshold = sorted
            .get(cutoff_idx.min(sorted.len().saturating_sub(1)))
            .copied()
            .unwrap_or(0.5);
        scores
            .into_iter()
            .map(|s| {
                if s >= threshold {
                    OutlierDecision::Outlier
                } else {
                    OutlierDecision::Inlier
                }
            })
            .collect()
    }
}

// ===== AnalyticsValue helpers =====
pub fn extract_numeric(values: &[AnalyticsValue]) -> Vec<f64> {
    values.iter().filter_map(|v| v.as_f64()).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn zscore_marks_far_values() {
        let xs = [1.0, 2.0, 3.0, 4.0, 5.0, 100.0];
        let scores = zscore(&xs);
        assert!(scores[5].abs() > 2.0);
        assert!(scores[2].abs() < 1.0);
    }

    #[test]
    fn iqr_flags_outliers() {
        let xs = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 100.0];
        let r = iqr_outlier(&xs, 1.5);
        assert_eq!(r[7], OutlierDecision::Outlier);
        assert_eq!(r[3], OutlierDecision::Inlier);
    }

    #[test]
    fn mad_handles_zero_mad() {
        let xs = [5.0; 10];
        let r = mad_outlier(&xs, 3.5);
        for d in r {
            assert_eq!(d, OutlierDecision::Inlier);
        }
    }

    #[test]
    fn nan_flagged_as_missing() {
        let xs = [1.0, 2.0, 3.0, f64::NAN, 5.0, 100.0];
        let iqr = iqr_outlier(&xs, 1.5);
        assert_eq!(iqr[3], OutlierDecision::Missing);
        let mad = mad_outlier(&xs, 3.5);
        assert_eq!(mad[3], OutlierDecision::Missing);
    }

    #[test]
    fn isolation_forest_scores_anomalies_higher() {
        let mut rows = Vec::new();
        for i in 0..200 {
            rows.push(vec![i as f64 * 0.01, (i as f64 * 0.01).sin()]);
        }
        // Inject anomalies far from the cluster
        rows.push(vec![100.0, 100.0]);
        rows.push(vec![-50.0, 50.0]);
        let forest = IsolationForest::fit(&rows, 32, 64, 42);
        let normal = forest.score(&rows[10]);
        let outlier = forest.score(&rows[200]);
        assert!(outlier > normal, "outlier {} <= normal {}", outlier, normal);
    }
}
