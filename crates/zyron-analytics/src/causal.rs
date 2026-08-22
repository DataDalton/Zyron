#![allow(non_snake_case)]
// Causal inference estimators
// Propensity score, ATE (inverse propensity weighting), ATT, DID
// Bootstrap CIs use the shared numeric::bootstrapCi infrastructure

use crate::ml::logisticRegression as logreg;
use crate::ml::{Hyperparameters, ModelConfig, ModelType, TrainingData};
use crate::numeric::KahanSum;
use zyron_common::error::{Result, ZyronError};

/// Causal estimate with point estimate and bootstrap percentile CI
#[derive(Debug, Clone)]
pub struct CausalEstimate {
    pub estimate: f64,
    pub lowerCi: f64,
    pub upperCi: f64,
    pub nObs: u64,
}

/// Per-row propensity scores for assignment to treatment given covariates
pub fn propensityScore(treatment: &[f64], covariates: &[f64], p: usize) -> Result<Vec<f64>> {
    let n = treatment.len();
    if n == 0 {
        return Ok(Vec::new());
    }
    if covariates.len() != n * p {
        return Err(ZyronError::InvalidParameter {
            name: "covariates".to_string(),
            value: format!("expected {}*{} entries, got {}", n, p, covariates.len()),
        });
    }
    let names: Vec<String> = (0..p).map(|j| format!("x{}", j)).collect();
    let mut hp = Hyperparameters::new();
    hp.setF64("learning_rate", 0.1);
    hp.setF64("max_epochs", 200.0);
    hp.setF64("lambda", 0.001);
    let config = ModelConfig {
        modelType: ModelType::LogisticRegression,
        featureColumns: names,
        targetColumn: Some("treatment".into()),
        hyperparameters: hp,
    };
    let data = TrainingData::new(covariates, treatment, n, p);
    let model = logreg::train(&config, &data)?;
    let mut probs = vec![0.0f64; n];
    for i in 0..n {
        probs[i] = logreg::predictProbability(&model, &covariates[i * p..i * p + p])
            .clamp(1e-3, 1.0 - 1e-3);
    }
    Ok(probs)
}

/// ATE via inverse propensity weighting estimator
/// Returns (1/n) sum_i [ T_i Y_i / p_i - (1-T_i) Y_i / (1 - p_i) ]
pub fn ate(outcome: &[f64], treatment: &[f64], covariates: &[f64], p: usize) -> Result<f64> {
    let n = outcome.len();
    if n == 0 {
        return Ok(0.0);
    }
    if treatment.len() != n {
        return Err(ZyronError::InvalidParameter {
            name: "treatment".to_string(),
            value: "length mismatch".to_string(),
        });
    }
    let propensity = propensityScore(treatment, covariates, p)?;
    let mut sum = KahanSum::new();
    for i in 0..n {
        let t = treatment[i];
        let y = outcome[i];
        let pi = propensity[i];
        let term = (t * y) / pi - ((1.0 - t) * y) / (1.0 - pi);
        sum.add(term);
    }
    Ok(sum.value() / n as f64)
}

/// Bootstrap CI on the ATE estimator
pub fn ateWithCi(
    outcome: &[f64],
    treatment: &[f64],
    covariates: &[f64],
    p: usize,
    bootstrap: usize,
    alpha: f64,
    seed: u64,
) -> Result<CausalEstimate> {
    let n = outcome.len();
    let estimate = ate(outcome, treatment, covariates, p)?;
    if bootstrap == 0 || n == 0 {
        return Ok(CausalEstimate {
            estimate,
            lowerCi: estimate,
            upperCi: estimate,
            nObs: n as u64,
        });
    }
    // Parallel bootstrap (F9). Split the B replicates across worker threads,
    // each fed an independent stream derived by long-jumping the seed
    // state per thread. forkStream() returns the post-jump state but does
    // not advance the parent, so jump explicitly between forks
    let nThreads = std::thread::available_parallelism()
        .map(|p| p.get())
        .unwrap_or(1)
        .min(bootstrap.max(1));
    let perThread = (bootstrap + nThreads - 1) / nThreads;
    let mut baseRng = zyron_common::Xoshiro256pp::fromSeed(seed);
    let mut estimates: Vec<f64> = std::thread::scope(|scope| -> Vec<f64> {
        let mut handles = Vec::with_capacity(nThreads);
        for t in 0..nThreads {
            let mut threadRng = baseRng;
            baseRng.longJump();
            let count = perThread.min(bootstrap.saturating_sub(t * perThread));
            let outRef = outcome;
            let treatRef = treatment;
            let covRef = covariates;
            let h = scope.spawn(move || {
                let mut bO = vec![0.0f64; n];
                let mut bT = vec![0.0f64; n];
                let mut bC = vec![0.0f64; n * p];
                let mut local = Vec::with_capacity(count);
                for _ in 0..count {
                    for i in 0..n {
                        let idx = threadRng.nextRange(n as u64) as usize;
                        bO[i] = outRef[idx];
                        bT[i] = treatRef[idx];
                        for j in 0..p {
                            bC[i * p + j] = covRef[idx * p + j];
                        }
                    }
                    if let Ok(e) = ate(&bO, &bT, &bC, p) {
                        local.push(e);
                    }
                }
                local
            });
            handles.push(h);
        }
        let mut out = Vec::with_capacity(bootstrap);
        for h in handles {
            if let Ok(mut v) = h.join() {
                out.append(&mut v);
            }
        }
        out
    });
    if estimates.is_empty() {
        return Ok(CausalEstimate {
            estimate,
            lowerCi: estimate,
            upperCi: estimate,
            nObs: n as u64,
        });
    }
    estimates.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let lo = ((alpha / 2.0) * estimates.len() as f64).floor() as usize;
    let hi = (((1.0 - alpha / 2.0) * estimates.len() as f64) - 1.0)
        .ceil()
        .max(0.0) as usize;
    let lo = lo.min(estimates.len() - 1);
    let hi = hi.min(estimates.len() - 1);
    Ok(CausalEstimate {
        estimate,
        lowerCi: estimates[lo],
        upperCi: estimates[hi],
        nObs: n as u64,
    })
}

/// ATT estimator, average treatment effect on the treated
pub fn att(outcome: &[f64], treatment: &[f64], covariates: &[f64], p: usize) -> Result<f64> {
    let n = outcome.len();
    if n == 0 {
        return Ok(0.0);
    }
    let propensity = propensityScore(treatment, covariates, p)?;
    let mut numerator = KahanSum::new();
    let mut denomTreated = 0u64;
    let mut weightedControl = KahanSum::new();
    let mut weightSum = KahanSum::new();
    for i in 0..n {
        let t = treatment[i];
        let y = outcome[i];
        let pi = propensity[i];
        if t == 1.0 {
            numerator.add(y);
            denomTreated += 1;
        } else {
            let w = pi / (1.0 - pi);
            weightedControl.add(w * y);
            weightSum.add(w);
        }
    }
    if denomTreated == 0 {
        return Err(ZyronError::ExecutionError(
            "no treated observations".to_string(),
        ));
    }
    if weightSum.value() == 0.0 {
        return Ok(numerator.value() / denomTreated as f64);
    }
    let treatedMean = numerator.value() / denomTreated as f64;
    let controlMean = weightedControl.value() / weightSum.value();
    Ok(treatedMean - controlMean)
}

/// Difference-in-differences estimator
/// (mean(Y | T=1, post=1) - mean(Y | T=1, post=0))
/// minus (mean(Y | T=0, post=1) - mean(Y | T=0, post=0))
pub fn diffInDiff(outcome: &[f64], treatment: &[f64], post: &[f64]) -> Result<f64> {
    let n = outcome.len();
    if n == 0 || treatment.len() != n || post.len() != n {
        return Err(ZyronError::InvalidParameter {
            name: "did_inputs".to_string(),
            value: "length mismatch".to_string(),
        });
    }
    let mut sum = [[0.0f64; 2]; 2];
    let mut cnt = [[0u64; 2]; 2];
    for i in 0..n {
        let ti = if treatment[i] != 0.0 { 1 } else { 0 };
        let pi = if post[i] != 0.0 { 1 } else { 0 };
        sum[ti][pi] += outcome[i];
        cnt[ti][pi] += 1;
    }
    for ti in 0..2 {
        for pi in 0..2 {
            if cnt[ti][pi] == 0 {
                return Err(ZyronError::ExecutionError(format!(
                    "empty cell treatment={} post={}",
                    ti, pi
                )));
            }
        }
    }
    let mean = |t: usize, p: usize| sum[t][p] / cnt[t][p] as f64;
    let did = (mean(1, 1) - mean(1, 0)) - (mean(0, 1) - mean(0, 0));
    Ok(did)
}

#[cfg(test)]
mod tests {
    use super::*;
    use zyron_common::Xoshiro256pp;

    #[test]
    fn propensityProducesValidProbs() {
        let mut rng = Xoshiro256pp::fromSeed(13);
        let n = 500;
        let mut cov = Vec::with_capacity(n * 2);
        let mut treat = Vec::with_capacity(n);
        for _ in 0..n {
            let x1 = rng.nextNormal();
            let x2 = rng.nextNormal();
            cov.push(x1);
            cov.push(x2);
            let z = 0.5 * x1 - 0.3 * x2;
            let p = 1.0 / (1.0 + (-z).exp());
            treat.push(if rng.nextF64() < p { 1.0 } else { 0.0 });
        }
        let probs = propensityScore(&treat, &cov, 2).unwrap();
        for p in probs {
            assert!(p > 0.0 && p < 1.0);
        }
    }

    #[test]
    fn ateRecoversTreatmentEffect() {
        let mut rng = Xoshiro256pp::fromSeed(31);
        let n = 1500;
        let mut cov = Vec::with_capacity(n * 2);
        let mut treat = Vec::with_capacity(n);
        let mut out = Vec::with_capacity(n);
        let trueEffect = 1.0;
        for _ in 0..n {
            let x1 = rng.nextNormal();
            let x2 = rng.nextNormal();
            cov.push(x1);
            cov.push(x2);
            let z = 0.5 * x1 - 0.3 * x2;
            let p = 1.0 / (1.0 + (-z).exp());
            let t = if rng.nextF64() < p { 1.0 } else { 0.0 };
            treat.push(t);
            // Y = trueEffect*T + 0.4*x1 + 0.2*x2 + noise
            out.push(trueEffect * t + 0.4 * x1 + 0.2 * x2 + 0.3 * rng.nextNormal());
        }
        let est = ate(&out, &treat, &cov, 2).unwrap();
        assert!((est - trueEffect).abs() < 0.4, "ate = {}", est);
    }

    #[test]
    fn diffInDiffComputesProperly() {
        let outcome = vec![1.0, 2.0, 1.0, 4.0];
        let treatment = vec![0.0, 0.0, 1.0, 1.0];
        let post = vec![0.0, 1.0, 0.0, 1.0];
        let did = diffInDiff(&outcome, &treatment, &post).unwrap();
        // (4-1) - (2-1) = 2.0
        assert!((did - 2.0).abs() < 1e-12);
    }
}
