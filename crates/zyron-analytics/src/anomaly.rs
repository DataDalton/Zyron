#![allow(non_snake_case)]
// Unified anomaly and drift detection
// detectAnomalies routes row data through the isolation forest or the
// single series detectors, detectAnomaliesTimeseries scores residuals
// from a fitted baseline, detectDrift compares two samples with KS,
// chi square, or PSI

use crate::ml::transforms::{ksStatistic, psi};
use crate::outlier::{IsolationForest, MadDetector, ZScoreEvaluator, iqr_outlier};
use crate::predictive::seasonalDecompose;
use crate::stats_tests::{kolmogorovQ, regularizedUpperGamma};
use zyron_common::error::{Result, ZyronError};

const ISOLATION_TREES: usize = 100;
const ISOLATION_SAMPLE: usize = 256;
const ISOLATION_SEED: u64 = 42;
const DRIFT_BINS: usize = 10;
const DRIFT_PSI_THRESHOLD: f64 = 0.25;
const DRIFT_P_THRESHOLD: f64 = 0.05;

#[derive(Debug, Clone)]
pub struct AnomalyParams {
    // isolation forest share of rows expected to be outliers
    pub contamination: f64,
    // zscore and mad flag when the absolute score exceeds this
    pub threshold: f64,
    // iqr fence multiplier
    pub multiplier: f64,
}

impl Default for AnomalyParams {
    fn default() -> Self {
        Self {
            contamination: 0.1,
            threshold: 3.0,
            multiplier: 1.5,
        }
    }
}

fn validateRows(rows: &[Vec<f64>]) -> Result<usize> {
    if rows.is_empty() {
        return Err(ZyronError::InvalidParameter {
            name: "rows".to_string(),
            value: "empty input".to_string(),
        });
    }
    let p = rows[0].len();
    if p == 0 {
        return Err(ZyronError::InvalidParameter {
            name: "rows".to_string(),
            value: "rows have no columns".to_string(),
        });
    }
    for r in rows {
        if r.len() != p {
            return Err(ZyronError::InvalidParameter {
                name: "rows".to_string(),
                value: "rows have inconsistent dimensions".to_string(),
            });
        }
        if r.iter().any(|v| !v.is_finite()) {
            return Err(ZyronError::InvalidParameter {
                name: "rows".to_string(),
                value: "rows contain non finite values".to_string(),
            });
        }
    }
    Ok(p)
}

fn singleColumn(rows: &[Vec<f64>], p: usize, method: &str) -> Result<Vec<f64>> {
    if p != 1 {
        return Err(ZyronError::InvalidParameter {
            name: "rows".to_string(),
            value: format!(
                "method '{}' operates on a single column, got {} columns",
                method, p
            ),
        });
    }
    Ok(rows.iter().map(|r| r[0]).collect())
}

fn quantileSorted(sorted: &[f64], q: f64) -> f64 {
    if sorted.is_empty() {
        return 0.0;
    }
    let pos = q * (sorted.len() as f64 - 1.0);
    let lo = pos.floor() as usize;
    let hi = (pos.ceil() as usize).min(sorted.len() - 1);
    let frac = pos - lo as f64;
    sorted[lo] + (sorted[hi] - sorted[lo]) * frac
}

/// Per row anomaly detection, returns (index, isAnomaly, score)
/// Methods
///   isolation_forest uses every column, contamination sets the flag rate
///   zscore, iqr, mad expect a single column
pub fn detectAnomalies(
    rows: &[Vec<f64>],
    method: &str,
    params: &AnomalyParams,
) -> Result<Vec<(usize, bool, f64)>> {
    let p = validateRows(rows)?;
    let lower = method.to_ascii_lowercase();
    match lower.as_str() {
        "isolation_forest" => {
            if !(params.contamination > 0.0 && params.contamination < 1.0) {
                return Err(ZyronError::InvalidParameter {
                    name: "contamination".to_string(),
                    value: format!("must be in (0, 1), got {}", params.contamination),
                });
            }
            let forest =
                IsolationForest::fit(rows, ISOLATION_TREES, ISOLATION_SAMPLE, ISOLATION_SEED);
            // Scored once: the decisions come from the same score vector,
            // scoring per row again would double the whole walk
            let scores = forest.score_all(rows);
            let decisions = IsolationForest::decisions_from_scores(&scores, params.contamination);
            Ok(scores
                .into_iter()
                .enumerate()
                .map(|(i, s)| (i, decisions[i].is_outlier(), s))
                .collect())
        }
        "zscore" => {
            if !(params.threshold > 0.0) {
                return Err(ZyronError::InvalidParameter {
                    name: "threshold".to_string(),
                    value: format!("must be positive, got {}", params.threshold),
                });
            }
            let values = singleColumn(rows, p, "zscore")?;
            let z = ZScoreEvaluator::fit(&values);
            Ok(values
                .iter()
                .enumerate()
                .map(|(i, v)| {
                    let s = z.score(*v);
                    (i, s.abs() > params.threshold, s)
                })
                .collect())
        }
        "iqr" => {
            if !(params.multiplier > 0.0) {
                return Err(ZyronError::InvalidParameter {
                    name: "multiplier".to_string(),
                    value: format!("must be positive, got {}", params.multiplier),
                });
            }
            let values = singleColumn(rows, p, "iqr")?;
            let decisions = iqr_outlier(&values, params.multiplier);
            let mut sorted = values.clone();
            sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            let q1 = quantileSorted(&sorted, 0.25);
            let q3 = quantileSorted(&sorted, 0.75);
            let lo = q1 - params.multiplier * (q3 - q1);
            let hi = q3 + params.multiplier * (q3 - q1);
            Ok(values
                .iter()
                .enumerate()
                .map(|(i, v)| {
                    // score is the distance beyond the nearest fence, 0 inside
                    let score = if *v < lo {
                        v - lo
                    } else if *v > hi {
                        v - hi
                    } else {
                        0.0
                    };
                    (i, decisions[i].is_outlier(), score)
                })
                .collect())
        }
        "mad" => {
            if !(params.threshold > 0.0) {
                return Err(ZyronError::InvalidParameter {
                    name: "threshold".to_string(),
                    value: format!("must be positive, got {}", params.threshold),
                });
            }
            let values = singleColumn(rows, p, "mad")?;
            let d = MadDetector::fit(&values);
            Ok(values
                .iter()
                .enumerate()
                .map(|(i, v)| {
                    let s = d.score(*v);
                    (i, s.abs() > params.threshold, s)
                })
                .collect())
        }
        other => Err(ZyronError::InvalidParameter {
            name: "method".to_string(),
            value: format!(
                "'{}' is not supported, supported methods are isolation_forest, zscore, iqr, mad",
                other
            ),
        }),
    }
}

#[derive(Debug, Clone)]
pub struct TimeseriesAnomalyParams {
    // moving_average trailing window length
    pub window: usize,
    // seasonal decomposition period
    pub period: usize,
    // ewma smoothing factor
    pub alpha: f64,
    // z score cutoff on the residuals
    pub threshold: f64,
}

impl Default for TimeseriesAnomalyParams {
    fn default() -> Self {
        Self {
            window: 10,
            period: 7,
            alpha: 0.3,
            threshold: 3.0,
        }
    }
}

/// Time series anomaly detection via residual z scores against a fitted
/// baseline. Methods are seasonal, moving_average, ewma. Positions where
/// the baseline is not yet defined return score 0 and no flag
pub fn detectAnomaliesTimeseries(
    values: &[f64],
    method: &str,
    params: &TimeseriesAnomalyParams,
) -> Result<Vec<(usize, bool, f64)>> {
    let n = values.len();
    if n == 0 {
        return Err(ZyronError::InvalidParameter {
            name: "values".to_string(),
            value: "empty series".to_string(),
        });
    }
    if values.iter().any(|v| !v.is_finite()) {
        return Err(ZyronError::InvalidParameter {
            name: "values".to_string(),
            value: "series contains non finite values".to_string(),
        });
    }
    if !(params.threshold > 0.0) {
        return Err(ZyronError::InvalidParameter {
            name: "threshold".to_string(),
            value: format!("must be positive, got {}", params.threshold),
        });
    }
    let lower = method.to_ascii_lowercase();
    // residual per position with a defined-baseline mask
    let (residuals, defined): (Vec<f64>, Vec<bool>) = match lower.as_str() {
        "moving_average" => {
            let w = params.window;
            if w < 2 || w >= n {
                return Err(ZyronError::InvalidParameter {
                    name: "window".to_string(),
                    value: format!("must be in 2..{}, got {}", n, w),
                });
            }
            let mut res = vec![0.0f64; n];
            let mut def = vec![false; n];
            let mut running: f64 = values[..w].iter().sum();
            for t in w..n {
                res[t] = values[t] - running / w as f64;
                def[t] = true;
                running += values[t] - values[t - w];
            }
            (res, def)
        }
        "ewma" => {
            if !(params.alpha > 0.0 && params.alpha < 1.0) {
                return Err(ZyronError::InvalidParameter {
                    name: "alpha".to_string(),
                    value: format!("must be in (0, 1), got {}", params.alpha),
                });
            }
            let mut res = vec![0.0f64; n];
            let mut def = vec![false; n];
            let mut level = values[0];
            for t in 1..n {
                res[t] = values[t] - level;
                def[t] = true;
                level = params.alpha * values[t] + (1.0 - params.alpha) * level;
            }
            (res, def)
        }
        "seasonal" => {
            if params.period < 2 || n < params.period * 2 {
                return Err(ZyronError::InvalidParameter {
                    name: "period".to_string(),
                    value: format!(
                        "seasonal method needs period >= 2 and at least 2 periods of data, got period {} with {} points",
                        params.period, n
                    ),
                });
            }
            let comp = seasonalDecompose(values, params.period);
            (comp.residual, vec![true; n])
        }
        "prophet" | "arima" => {
            return Err(ZyronError::InvalidParameter {
                name: "method".to_string(),
                value: format!(
                    "'{}' has no supported implementation, supported methods are seasonal, moving_average, ewma",
                    lower
                ),
            });
        }
        other => {
            return Err(ZyronError::InvalidParameter {
                name: "method".to_string(),
                value: format!(
                    "'{}' is not supported, supported methods are seasonal, moving_average, ewma",
                    other
                ),
            });
        }
    };
    // z score the residuals over the defined positions
    let mut sum = 0.0f64;
    let mut count = 0u64;
    for i in 0..n {
        if defined[i] {
            sum += residuals[i];
            count += 1;
        }
    }
    if count == 0 {
        return Ok((0..n).map(|i| (i, false, 0.0)).collect());
    }
    let mean = sum / count as f64;
    let mut var_sum = 0.0f64;
    for i in 0..n {
        if defined[i] {
            let d = residuals[i] - mean;
            var_sum += d * d;
        }
    }
    let std = (var_sum / count as f64).sqrt();
    Ok((0..n)
        .map(|i| {
            if !defined[i] {
                return (i, false, 0.0);
            }
            let score = if std > 0.0 {
                (residuals[i] - mean) / std
            } else {
                0.0
            };
            (i, score.abs() > params.threshold, score)
        })
        .collect())
}

#[derive(Debug, Clone)]
pub struct DriftResult {
    pub statistic: f64,
    // None for PSI, which has no p value
    pub pValue: Option<f64>,
    pub drifted: bool,
}

/// Distribution drift between a reference sample and a current sample
/// Methods and drift conventions
///   ks_test flags when the two sample KS p value is below 0.05
///   chi_square flags when the binned chi square p value is below 0.05
///   psi flags when the population stability index exceeds 0.25
pub fn detectDrift(reference: &[f64], current: &[f64], method: &str) -> Result<DriftResult> {
    let ref_vals: Vec<f64> = reference
        .iter()
        .copied()
        .filter(|v| v.is_finite())
        .collect();
    let cur_vals: Vec<f64> = current.iter().copied().filter(|v| v.is_finite()).collect();
    if ref_vals.is_empty() || cur_vals.is_empty() {
        return Err(ZyronError::InvalidParameter {
            name: "samples".to_string(),
            value: "reference and current must both contain finite values".to_string(),
        });
    }
    let lower = method.to_ascii_lowercase();
    match lower.as_str() {
        "ks_test" => {
            let d = ksStatistic(&ref_vals, &cur_vals);
            let na = ref_vals.len() as f64;
            let nb = cur_vals.len() as f64;
            let en = (na * nb / (na + nb)).sqrt();
            let p = kolmogorovQ((en + 0.12 + 0.11 / en) * d);
            Ok(DriftResult {
                statistic: d,
                pValue: Some(p),
                drifted: p < DRIFT_P_THRESHOLD,
            })
        }
        "chi_square" => {
            let (chi2, p) = chiSquareBinned(&ref_vals, &cur_vals, DRIFT_BINS);
            Ok(DriftResult {
                statistic: chi2,
                pValue: Some(p),
                drifted: p < DRIFT_P_THRESHOLD,
            })
        }
        "psi" => {
            let (ref_counts, cur_counts) = binCounts(&ref_vals, &cur_vals, DRIFT_BINS);
            let s = psi(&cur_counts, &ref_counts);
            Ok(DriftResult {
                statistic: s,
                pValue: None,
                drifted: s > DRIFT_PSI_THRESHOLD,
            })
        }
        other => Err(ZyronError::InvalidParameter {
            name: "method".to_string(),
            value: format!(
                "'{}' is not supported, supported methods are ks_test, chi_square, psi",
                other
            ),
        }),
    }
}

// Equal width histogram counts over the combined range of both samples
fn binCounts(reference: &[f64], current: &[f64], bins: usize) -> (Vec<u64>, Vec<u64>) {
    let mut min = f64::INFINITY;
    let mut max = f64::NEG_INFINITY;
    for v in reference.iter().chain(current.iter()) {
        if *v < min {
            min = *v;
        }
        if *v > max {
            max = *v;
        }
    }
    let mut ref_counts = vec![0u64; bins];
    let mut cur_counts = vec![0u64; bins];
    if max <= min {
        ref_counts[0] = reference.len() as u64;
        cur_counts[0] = current.len() as u64;
        return (ref_counts, cur_counts);
    }
    let width = (max - min) / bins as f64;
    let index = |v: f64| (((v - min) / width) as usize).min(bins - 1);
    for &v in reference {
        ref_counts[index(v)] += 1;
    }
    for &v in current {
        cur_counts[index(v)] += 1;
    }
    (ref_counts, cur_counts)
}

// Chi square statistic and p value over equal width bins of the combined
// range. Expected counts come from the reference proportions scaled to
// the current total, floored at 0.5 so mass in bins the reference never
// saw registers as large but finite drift
fn chiSquareBinned(reference: &[f64], current: &[f64], bins: usize) -> (f64, f64) {
    let (ref_counts, cur_counts) = binCounts(reference, current, bins);
    let ref_total: u64 = ref_counts.iter().sum();
    let cur_total: u64 = cur_counts.iter().sum();
    let mut chi2 = 0.0f64;
    let mut used = 0usize;
    for i in 0..bins {
        if ref_counts[i] == 0 && cur_counts[i] == 0 {
            continue;
        }
        used += 1;
        let expected = (ref_counts[i] as f64 / ref_total as f64 * cur_total as f64).max(0.5);
        let observed = cur_counts[i] as f64;
        let d = observed - expected;
        chi2 += d * d / expected;
    }
    let df = used.saturating_sub(1).max(1) as f64;
    let p = regularizedUpperGamma(df / 2.0, chi2 / 2.0);
    (chi2, p)
}

#[cfg(test)]
mod tests {
    use super::*;
    use zyron_common::Xoshiro256pp;

    #[test]
    fn isolationForestScoresPlantedOutlierAboveClean() {
        let mut rng = Xoshiro256pp::fromSeed(61);
        let mut rows: Vec<Vec<f64>> = (0..150)
            .map(|_| vec![rng.nextNormal(), rng.nextNormal()])
            .collect();
        rows.push(vec![50.0, -50.0]);
        let out = detectAnomalies(&rows, "isolation_forest", &AnomalyParams::default())
            .expect("isolation forest");
        let planted = out[150].2;
        let clean_max = out[..150].iter().map(|r| r.2).fold(f64::MIN, f64::max);
        assert!(
            planted > clean_max,
            "planted {} vs clean max {}",
            planted,
            clean_max
        );
        assert!(out[150].1, "planted outlier not flagged");
    }

    #[test]
    fn zscoreFlagsSpike() {
        let mut rows: Vec<Vec<f64>> = (0..100).map(|i| vec![(i % 5) as f64]).collect();
        rows.push(vec![500.0]);
        let out = detectAnomalies(&rows, "zscore", &AnomalyParams::default()).expect("zscore");
        assert!(out[100].1);
        assert!(!out[10].1);
    }

    #[test]
    fn unknownMethodErrors() {
        let rows = vec![vec![1.0], vec![2.0]];
        assert!(detectAnomalies(&rows, "dbscan", &AnomalyParams::default()).is_err());
    }

    #[test]
    fn ewmaFlagsSpike() {
        let mut values: Vec<f64> = (0..120).map(|i| (i as f64 * 0.1).sin()).collect();
        values[80] = 25.0;
        let out = detectAnomaliesTimeseries(&values, "ewma", &TimeseriesAnomalyParams::default())
            .expect("ewma");
        assert!(out[80].1, "spike not flagged, score = {}", out[80].2);
        let flagged: usize = out.iter().filter(|r| r.1).count();
        assert!(flagged <= 3, "flagged {} points", flagged);
    }

    #[test]
    fn prophetErrorsClearly() {
        let values: Vec<f64> = (0..50).map(|i| i as f64).collect();
        let err =
            detectAnomaliesTimeseries(&values, "prophet", &TimeseriesAnomalyParams::default());
        match err {
            Err(ZyronError::InvalidParameter { name, value }) => {
                assert_eq!(name, "method");
                assert!(value.contains("prophet"));
                assert!(value.contains("ewma"));
            }
            other => panic!("expected InvalidParameter, got {:?}", other.map(|_| ())),
        }
    }

    #[test]
    fn ksDriftFlagsShiftPassesIdentical() {
        let mut rng = Xoshiro256pp::fromSeed(71);
        let reference: Vec<f64> = (0..400).map(|_| rng.nextNormal()).collect();
        let shifted: Vec<f64> = reference.iter().map(|v| v + 1.5).collect();
        let drift = detectDrift(&reference, &shifted, "ks_test").expect("ks drift");
        assert!(drift.drifted);
        assert!(drift.pValue.map(|p| p < 0.01).unwrap_or(false));
        let same = detectDrift(&reference, &reference, "ks_test").expect("ks same");
        assert!(
            !same.drifted,
            "identical samples flagged, p = {:?}",
            same.pValue
        );
    }

    #[test]
    fn chiSquarePValueSanity() {
        let mut rng = Xoshiro256pp::fromSeed(72);
        let reference: Vec<f64> = (0..500).map(|_| rng.nextNormal()).collect();
        let current: Vec<f64> = (0..500).map(|_| rng.nextNormal()).collect();
        let same = detectDrift(&reference, &current, "chi_square").expect("chi same");
        assert!(
            same.pValue.map(|p| p > 0.01).unwrap_or(false),
            "p = {:?}",
            same.pValue
        );
        let shifted: Vec<f64> = reference.iter().map(|v| v + 2.0).collect();
        let drift = detectDrift(&reference, &shifted, "chi_square").expect("chi drift");
        assert!(drift.drifted);
        assert!(drift.statistic > same.statistic);
    }

    #[test]
    fn psiDriftThreshold() {
        let mut rng = Xoshiro256pp::fromSeed(73);
        let reference: Vec<f64> = (0..500).map(|_| rng.nextNormal()).collect();
        let shifted: Vec<f64> = reference.iter().map(|v| v + 2.0).collect();
        let drift = detectDrift(&reference, &shifted, "psi").expect("psi drift");
        assert!(drift.drifted);
        assert!(drift.pValue.is_none());
        let same = detectDrift(&reference, &reference, "psi").expect("psi same");
        assert!(!same.drifted);
    }
}
