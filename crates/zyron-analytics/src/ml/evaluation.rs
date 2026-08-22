#![allow(non_snake_case)]
// Cross-cutting evaluation utilities
// Confusion matrix, ROC, PR curve, calibration, fairness reports

use std::collections::HashMap;

#[derive(Debug, Clone)]
pub struct ConfusionMatrix {
    pub k: usize,
    pub counts: Vec<u64>,
}

impl ConfusionMatrix {
    pub fn newSquare(k: usize) -> Self {
        Self {
            k,
            counts: vec![0u64; k * k],
        }
    }

    pub fn build(yTrue: &[u32], yPred: &[u32], k: usize) -> Self {
        let mut cm = Self::newSquare(k);
        for (t, p) in yTrue.iter().zip(yPred.iter()) {
            let ti = (*t as usize).min(k - 1);
            let pi = (*p as usize).min(k - 1);
            cm.counts[ti * k + pi] += 1;
        }
        cm
    }

    pub fn accuracy(&self) -> f64 {
        let mut diag = 0u64;
        let mut total = 0u64;
        for i in 0..self.k {
            diag += self.counts[i * self.k + i];
            for j in 0..self.k {
                total += self.counts[i * self.k + j];
            }
        }
        if total == 0 {
            0.0
        } else {
            diag as f64 / total as f64
        }
    }

    pub fn precision(&self, cls: usize) -> f64 {
        let mut tp = 0u64;
        let mut fp = 0u64;
        for i in 0..self.k {
            let v = self.counts[i * self.k + cls];
            if i == cls {
                tp = v;
            } else {
                fp += v;
            }
        }
        let denom = (tp + fp) as f64;
        if denom > 0.0 { tp as f64 / denom } else { 0.0 }
    }

    pub fn recall(&self, cls: usize) -> f64 {
        let mut tp = 0u64;
        let mut fnc = 0u64;
        for j in 0..self.k {
            let v = self.counts[cls * self.k + j];
            if j == cls {
                tp = v;
            } else {
                fnc += v;
            }
        }
        let denom = (tp + fnc) as f64;
        if denom > 0.0 { tp as f64 / denom } else { 0.0 }
    }

    pub fn f1(&self, cls: usize) -> f64 {
        let p = self.precision(cls);
        let r = self.recall(cls);
        if p + r > 0.0 {
            2.0 * p * r / (p + r)
        } else {
            0.0
        }
    }
}

/// ROC curve points (fpr, tpr) sorted by threshold descending
pub fn rocCurve(yTrue: &[u32], yScore: &[f64]) -> Vec<(f64, f64)> {
    let mut pairs: Vec<(f64, u32)> = yTrue
        .iter()
        .zip(yScore.iter())
        .map(|(y, s)| (*s, *y))
        .collect();
    pairs.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
    let totalPos: u64 = yTrue.iter().filter(|&&y| y == 1).count() as u64;
    let totalNeg: u64 = yTrue.iter().filter(|&&y| y == 0).count() as u64;
    let mut tp = 0u64;
    let mut fp = 0u64;
    let mut out = Vec::with_capacity(pairs.len() + 1);
    out.push((0.0, 0.0));
    for (_, y) in &pairs {
        if *y == 1 {
            tp += 1;
        } else {
            fp += 1;
        }
        let fpr = if totalNeg > 0 {
            fp as f64 / totalNeg as f64
        } else {
            0.0
        };
        let tpr = if totalPos > 0 {
            tp as f64 / totalPos as f64
        } else {
            0.0
        };
        out.push((fpr, tpr));
    }
    out
}

pub fn rocAuc(yTrue: &[u32], yScore: &[f64]) -> f64 {
    let curve = rocCurve(yTrue, yScore);
    let mut auc = 0.0f64;
    for w in curve.windows(2) {
        auc += (w[1].0 - w[0].0) * (w[0].1 + w[1].1) * 0.5;
    }
    auc
}

/// Brier score for probabilistic binary predictions
pub fn brierScore(yTrue: &[u32], yScore: &[f64]) -> f64 {
    if yTrue.is_empty() {
        return 0.0;
    }
    let mut s = 0.0f64;
    for (y, p) in yTrue.iter().zip(yScore.iter()) {
        let d = *p - (*y as f64);
        s += d * d;
    }
    s / yTrue.len() as f64
}

/// Demographic parity, P(yhat=1 | group=g) per group
pub fn demographicParity(yPred: &[u32], group: &[u32]) -> HashMap<u32, f64> {
    let mut counts: HashMap<u32, (u64, u64)> = HashMap::new();
    for (p, g) in yPred.iter().zip(group.iter()) {
        let entry = counts.entry(*g).or_insert((0, 0));
        entry.1 += 1;
        if *p == 1 {
            entry.0 += 1;
        }
    }
    let mut out = HashMap::new();
    for (g, (pos, total)) in counts {
        let r = if total > 0 {
            pos as f64 / total as f64
        } else {
            0.0
        };
        out.insert(g, r);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn confusionAccuracyMatches() {
        let yt = vec![0, 1, 0, 1, 0, 1];
        let yp = vec![0, 1, 0, 0, 0, 1];
        let cm = ConfusionMatrix::build(&yt, &yp, 2);
        assert!((cm.accuracy() - 5.0 / 6.0).abs() < 1e-12);
    }

    #[test]
    fn rocAucPerfect() {
        let yt = vec![0, 0, 1, 1];
        let ys = vec![0.1, 0.2, 0.8, 0.9];
        let auc = rocAuc(&yt, &ys);
        assert!((auc - 1.0).abs() < 1e-9, "auc = {}", auc);
    }

    #[test]
    fn brierScoreZeroForPerfect() {
        let yt = vec![0, 1];
        let ys = vec![0.0, 1.0];
        assert!(brierScore(&yt, &ys) < 1e-12);
    }
}
