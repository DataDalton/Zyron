#![allow(non_snake_case)]
// Reproducible feature transformations stored alongside features
// At inference time the same transform is replayed on incoming rows

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Transform {
    /// Pass-through, no transform
    Identity,
    /// Standard score, (x - mean) / std
    Standardize { mean: f64, std: f64 },
    /// Min-max scaling to [0,1]
    MinMax { min: f64, max: f64 },
    /// Robust scaling with median and IQR
    Robust { median: f64, iqr: f64 },
    /// Equal-width binning, returns bin index
    BinEqualWidth { min: f64, width: f64, bins: u32 },
    /// Quantile binning
    BinQuantile { edges: Vec<f64> },
    /// One-hot, value is the categorical level mapped to its column index
    OneHot { levels: Vec<String> },
    /// Hashing trick, modulo bucket count, hash via fxMix
    Hash { buckets: u32 },
    /// Logarithm base e of (x + offset)
    Log { offset: f64 },
    /// Polynomial expansion up to degree d, returns multiple values
    Polynomial { degree: u32 },
}

impl Transform {
    pub fn applyScalar(&self, x: f64) -> f64 {
        match self {
            Transform::Identity => x,
            Transform::Standardize { mean, std } => {
                if *std == 0.0 {
                    0.0
                } else {
                    (x - mean) / std
                }
            }
            Transform::MinMax { min, max } => {
                let range = max - min;
                if range == 0.0 { 0.0 } else { (x - min) / range }
            }
            Transform::Robust { median, iqr } => {
                if *iqr == 0.0 {
                    0.0
                } else {
                    (x - median) / iqr
                }
            }
            Transform::BinEqualWidth { min, width, bins } => {
                if *width <= 0.0 {
                    0.0
                } else {
                    (((x - min) / width).floor() as i64).clamp(0, (*bins - 1) as i64) as f64
                }
            }
            Transform::BinQuantile { edges } => {
                let mut bin = 0usize;
                for e in edges {
                    if x > *e {
                        bin += 1;
                    }
                }
                bin as f64
            }
            Transform::Log { offset } => (x + offset).max(f64::MIN_POSITIVE).ln(),
            Transform::Polynomial { degree } => {
                // Apply scalar returns highest-degree only; full expansion via applyVec
                x.powi(*degree as i32)
            }
            Transform::OneHot { levels: _ } | Transform::Hash { buckets: _ } => x,
        }
    }

    pub fn applyText(&self, text: &str) -> Vec<f64> {
        match self {
            Transform::OneHot { levels } => {
                let mut out = vec![0.0f64; levels.len()];
                if let Some(idx) = levels.iter().position(|l| l == text) {
                    out[idx] = 1.0;
                }
                out
            }
            Transform::Hash { buckets } => {
                let mut h: u64 = 0x9E3779B97F4A7C15;
                for b in text.as_bytes() {
                    h = zyron_common::fx_mix(h, *b as u64);
                }
                let bucket = (h % (*buckets as u64)) as usize;
                let mut out = vec![0.0f64; *buckets as usize];
                out[bucket] = 1.0;
                out
            }
            _ => Vec::new(),
        }
    }

    pub fn applyVec(&self, x: f64) -> Vec<f64> {
        match self {
            Transform::Polynomial { degree } => (1..=*degree).map(|d| x.powi(d as i32)).collect(),
            _ => vec![self.applyScalar(x)],
        }
    }
}

/// Pipeline of transforms applied in order to a column
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct TransformPipeline {
    pub steps: Vec<Transform>,
}

impl TransformPipeline {
    pub fn new() -> Self {
        Self { steps: Vec::new() }
    }

    pub fn push(&mut self, t: Transform) {
        self.steps.push(t);
    }

    pub fn applyScalar(&self, x: f64) -> f64 {
        let mut v = x;
        for t in &self.steps {
            v = t.applyScalar(v);
        }
        v
    }
}

/// Fit a Standardize transform from a column slice
pub fn fitStandardize(values: &[f64]) -> Transform {
    if values.is_empty() {
        return Transform::Standardize {
            mean: 0.0,
            std: 1.0,
        };
    }
    let mean: f64 = values.iter().sum::<f64>() / values.len() as f64;
    let mut var = 0.0f64;
    for &v in values {
        let d = v - mean;
        var += d * d;
    }
    let std = if values.len() > 1 {
        (var / (values.len() - 1) as f64).sqrt()
    } else {
        1.0
    };
    Transform::Standardize {
        mean,
        std: if std == 0.0 { 1.0 } else { std },
    }
}

pub fn fitMinMax(values: &[f64]) -> Transform {
    if values.is_empty() {
        return Transform::MinMax { min: 0.0, max: 1.0 };
    }
    let mut mn = f64::INFINITY;
    let mut mx = f64::NEG_INFINITY;
    for &v in values {
        if v < mn {
            mn = v;
        }
        if v > mx {
            mx = v;
        }
    }
    Transform::MinMax { min: mn, max: mx }
}

/// Date features: returns (year, month, day, dow, hour, isWeekend, dayOfYear, weekOfYear, quarter)
pub fn dateFeatures(timestampMs: i64) -> [f64; 9] {
    use crate::value::MS_PER_DAY;
    let days = timestampMs.div_euclid(MS_PER_DAY) as i64;
    let (y, m, d) = civilFromDays(days);
    let dow = (((days % 7) + 4 + 7) % 7) as i32; // Unix epoch was Thursday
    let dayOfYear = dayOfYear(y, m, d);
    let weekOfYear = (dayOfYear - 1) / 7 + 1;
    let quarter = (m - 1) / 3 + 1;
    let secondsOfDay = (timestampMs.rem_euclid(MS_PER_DAY)) as i64 / 1000;
    let hour = (secondsOfDay / 3600) as i32;
    let isWeekend = if dow == 5 || dow == 6 { 1.0 } else { 0.0 };
    [
        y as f64,
        m as f64,
        d as f64,
        dow as f64,
        hour as f64,
        isWeekend,
        dayOfYear as f64,
        weekOfYear as f64,
        quarter as f64,
    ]
}

fn civilFromDays(days: i64) -> (i32, u32, u32) {
    let z = days + 719_468;
    let era = if z >= 0 { z } else { z - 146_096 } / 146_097;
    let doe = (z - era * 146_097) as u64;
    let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146_096) / 365;
    let mut y = yoe as i64 + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    if m <= 2 {
        y += 1;
    }
    (y as i32, m as u32, d as u32)
}

fn dayOfYear(y: i32, m: u32, d: u32) -> u32 {
    let monthDays = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31];
    let mut days: u32 = 0;
    for i in 0..(m as usize - 1) {
        days += monthDays[i];
    }
    days += d;
    let leap = (y % 4 == 0 && y % 100 != 0) || (y % 400 == 0);
    if leap && m > 2 {
        days += 1;
    }
    days
}

/// Polynomial features over multiple columns up to a given degree
/// Returns row-major output with all monomials up to total degree d
pub fn polynomialFeatures(features: &[f64], degree: u32) -> Vec<f64> {
    if degree == 0 {
        return vec![1.0];
    }
    let mut out: Vec<f64> = features.to_vec();
    if degree >= 2 {
        for i in 0..features.len() {
            for j in i..features.len() {
                out.push(features[i] * features[j]);
            }
        }
    }
    if degree >= 3 {
        for i in 0..features.len() {
            for j in i..features.len() {
                for k in j..features.len() {
                    out.push(features[i] * features[j] * features[k]);
                }
            }
        }
    }
    out
}

/// Drift detection via Population Stability Index between two distributions
/// Both inputs are histogram-bin counts of equal length
pub fn psi(actual: &[u64], expected: &[u64]) -> f64 {
    debug_assert_eq!(actual.len(), expected.len());
    let aTotal: u64 = actual.iter().sum();
    let eTotal: u64 = expected.iter().sum();
    if aTotal == 0 || eTotal == 0 {
        return 0.0;
    }
    let mut s = 0.0f64;
    for i in 0..actual.len() {
        let a = (actual[i] as f64 / aTotal as f64).max(1e-6);
        let e = (expected[i] as f64 / eTotal as f64).max(1e-6);
        s += (a - e) * (a / e).ln();
    }
    s
}

/// Two-sample Kolmogorov-Smirnov D statistic on sorted samples
pub fn ksStatistic(a: &[f64], b: &[f64]) -> f64 {
    if a.is_empty() || b.is_empty() {
        return 0.0;
    }
    let mut aSorted: Vec<f64> = a.to_vec();
    aSorted.sort_by(|x, y| x.partial_cmp(y).unwrap_or(std::cmp::Ordering::Equal));
    let mut bSorted: Vec<f64> = b.to_vec();
    bSorted.sort_by(|x, y| x.partial_cmp(y).unwrap_or(std::cmp::Ordering::Equal));
    let mut i = 0usize;
    let mut j = 0usize;
    let mut d = 0.0f64;
    let na = aSorted.len() as f64;
    let nb = bSorted.len() as f64;
    while i < aSorted.len() && j < bSorted.len() {
        let av = aSorted[i];
        let bv = bSorted[j];
        if av < bv {
            i += 1;
        } else if bv < av {
            j += 1;
        } else {
            i += 1;
            j += 1;
        }
        let cdfA = i as f64 / na;
        let cdfB = j as f64 / nb;
        let g = (cdfA - cdfB).abs();
        if g > d {
            d = g;
        }
    }
    d
}

/// Lag features: produces wide rows with values at offsets
pub fn lagFeatures(values: &[f64], lags: &[usize]) -> HashMap<usize, Vec<f64>> {
    let mut out = HashMap::new();
    for &l in lags {
        let mut col = Vec::with_capacity(values.len());
        for i in 0..values.len() {
            col.push(if i >= l { values[i - l] } else { f64::NAN });
        }
        out.insert(l, col);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn standardizeFitMatches() {
        let xs = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let t = fitStandardize(&xs);
        match t {
            Transform::Standardize { mean, std } => {
                assert!((mean - 3.0).abs() < 1e-12);
                assert!(std > 1.0);
            }
            _ => panic!(),
        }
    }

    #[test]
    fn polynomialFeaturesDegreeTwo() {
        let f = vec![2.0, 3.0];
        let out = polynomialFeatures(&f, 2);
        assert_eq!(out, vec![2.0, 3.0, 4.0, 6.0, 9.0]);
    }

    #[test]
    fn psiDetectsShift() {
        let a: Vec<u64> = vec![100, 100, 100, 100];
        let b: Vec<u64> = vec![10, 90, 90, 10];
        let s1 = psi(&a, &a);
        let s2 = psi(&a, &b);
        assert!(s1 < 1e-9);
        assert!(s2 > 0.1, "psi = {}", s2);
    }

    #[test]
    fn ksDetectsShift() {
        let a: Vec<f64> = (0..100).map(|i| i as f64).collect();
        let b: Vec<f64> = (50..150).map(|i| i as f64).collect();
        let d = ksStatistic(&a, &b);
        assert!(d > 0.4, "ks = {}", d);
    }

    #[test]
    fn dateFeaturesShape() {
        let f = dateFeatures(1_700_000_000_000);
        assert_eq!(f.len(), 9);
    }
}
