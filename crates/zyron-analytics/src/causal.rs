#![allow(non_snake_case)]
// Causal inference estimators
// Propensity score, ATE (inverse propensity weighting), ATT, DID
// Bootstrap CIs use the shared numeric::bootstrapCi infrastructure

use crate::ml::logisticRegression as logreg;
use crate::ml::{Hyperparameters, ModelConfig, ModelType, TrainingData};
use crate::numeric::{KahanSum, columnStandardize};
use crate::stats_tests::{normalCdf, normalQuantile};
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

// ===== Causal impact via local level state space model =====

#[derive(Debug, Clone)]
pub struct CausalImpactPoint {
    // position in the time ordered series
    pub index: usize,
    pub actual: f64,
    pub predicted: f64,
    pub effect: f64,
    pub lower: f64,
    pub upper: f64,
    pub cumulativeEffect: f64,
    pub cumulativeLower: f64,
    pub cumulativeUpper: f64,
}

fn empiricalQuantile(sorted: &[f64], q: f64) -> f64 {
    if sorted.is_empty() {
        return 0.0;
    }
    let pos = q * (sorted.len() as f64 - 1.0);
    let lo = pos.floor() as usize;
    let hi = (pos.ceil() as usize).min(sorted.len() - 1);
    let frac = pos - lo as f64;
    sorted[lo] + (sorted[hi] - sorted[lo]) * frac
}

// Kalman filter for the local level model with observation variance 1 and
// level variance q, diffuse initialization. Returns the final filtered
// level, the one step residuals, and the concentrated log likelihood
fn localLevelFilter(y: &[f64], q: f64) -> (f64, Vec<f64>, f64) {
    let mut a = y[0];
    let mut p_var = 1e7f64;
    let mut residuals = Vec::with_capacity(y.len().saturating_sub(1));
    let mut sum_ln_f = 0.0f64;
    let mut sum_sq = 0.0f64;
    for &obs in y.iter().skip(1) {
        let f = p_var + 1.0;
        let v = obs - a;
        residuals.push(v);
        sum_ln_f += f.ln();
        sum_sq += v * v / f;
        let gain = p_var / f;
        a += gain * v;
        p_var = p_var * (1.0 - gain) + q;
    }
    let m = residuals.len() as f64;
    let sigma2 = if m > 0.0 { sum_sq / m } else { 0.0 };
    let ll = if sigma2 > 0.0 {
        -0.5 * (sum_ln_f + m * sigma2.ln())
    } else {
        // a perfect fit dominates every alternative
        f64::INFINITY
    };
    (a, residuals, ll)
}

/// Intervention effect on a univariate series
/// A local level state space model is fit on the pre period by maximum
/// concentrated likelihood over a grid of signal to noise ratios, the
/// fitted level forecasts the post period, and the interval comes from
/// the empirical quantiles of the pre period one step residuals with the
/// cumulative band widening by the square root of the horizon
pub fn causalImpact(
    outcome: &[f64],
    timestamps: &[i64],
    interventionIndex: usize,
) -> Result<Vec<CausalImpactPoint>> {
    let n = outcome.len();
    if n == 0 || timestamps.len() != n {
        return Err(ZyronError::InvalidParameter {
            name: "outcome".to_string(),
            value: "outcome and timestamps must be nonempty and equal length".to_string(),
        });
    }
    if outcome.iter().any(|v| !v.is_finite()) {
        return Err(ZyronError::InvalidParameter {
            name: "outcome".to_string(),
            value: "outcome contains non finite values".to_string(),
        });
    }
    if interventionIndex < 3 || interventionIndex >= n {
        return Err(ZyronError::InvalidParameter {
            name: "intervention_index".to_string(),
            value: format!(
                "needs at least 3 pre period points and 1 post period point, got index {} with {} points",
                interventionIndex, n
            ),
        });
    }
    // time order the series before splitting pre and post
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by_key(|&i| timestamps[i]);
    let y: Vec<f64> = order.iter().map(|&i| outcome[i]).collect();
    let pre = &y[..interventionIndex];

    let mut best_level = pre[pre.len() - 1];
    let mut best_residuals: Vec<f64> = Vec::new();
    let mut best_ll = f64::NEG_INFINITY;
    for step in 0..25 {
        // log spaced grid from 1e-6 to 1e2
        let q = 10f64.powf(-6.0 + 8.0 * step as f64 / 24.0);
        let (level, residuals, ll) = localLevelFilter(pre, q);
        if ll > best_ll {
            best_ll = ll;
            best_level = level;
            best_residuals = residuals;
        }
    }
    let mut sorted_res = best_residuals;
    sorted_res.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let q_lo = empiricalQuantile(&sorted_res, 0.025);
    let q_hi = empiricalQuantile(&sorted_res, 0.975);

    let mut out = Vec::with_capacity(n - interventionIndex);
    let mut cumulative = 0.0f64;
    for (h, idx) in (interventionIndex..n).enumerate() {
        let actual = y[idx];
        let predicted = best_level;
        let effect = actual - predicted;
        cumulative += effect;
        let half_width = 0.5 * (q_hi - q_lo) * ((h + 1) as f64).sqrt();
        out.push(CausalImpactPoint {
            index: idx,
            actual,
            predicted,
            effect,
            lower: effect + q_lo,
            upper: effect + q_hi,
            cumulativeEffect: cumulative,
            cumulativeLower: cumulative - half_width,
            cumulativeUpper: cumulative + half_width,
        });
    }
    Ok(out)
}

// ===== Counterfactual estimation =====

#[derive(Debug, Clone)]
pub struct CounterfactualRow {
    pub index: usize,
    pub observed: f64,
    pub counterfactual: f64,
    // treated minus control outcome for this unit
    pub effect: f64,
}

/// Per row counterfactual outcome from the nearest neighbor in the
/// opposite treatment group, distances measured in standardized feature
/// space
pub fn counterfactual(
    observed: &[f64],
    treatmentFlag: &[f64],
    features: &[Vec<f64>],
) -> Result<Vec<CounterfactualRow>> {
    let n = observed.len();
    if n == 0 || treatmentFlag.len() != n || features.len() != n {
        return Err(ZyronError::InvalidParameter {
            name: "counterfactual_inputs".to_string(),
            value: "observed, treatment, and features must be nonempty and equal length"
                .to_string(),
        });
    }
    let p = features[0].len();
    if p == 0 {
        return Err(ZyronError::InvalidParameter {
            name: "features".to_string(),
            value: "feature rows have no columns".to_string(),
        });
    }
    let mut flat = Vec::with_capacity(n * p);
    for r in features {
        if r.len() != p {
            return Err(ZyronError::InvalidParameter {
                name: "features".to_string(),
                value: "feature rows have inconsistent dimensions".to_string(),
            });
        }
        if r.iter().any(|v| !v.is_finite()) {
            return Err(ZyronError::InvalidParameter {
                name: "features".to_string(),
                value: "feature rows contain non finite values".to_string(),
            });
        }
        flat.extend_from_slice(r);
    }
    let (mean, std) = columnStandardize(&flat, n, p);
    let standardized: Vec<Vec<f64>> = (0..n)
        .map(|i| {
            (0..p)
                .map(|j| (flat[i * p + j] - mean[j]) / std[j])
                .collect()
        })
        .collect();
    let treated: Vec<usize> = (0..n).filter(|&i| treatmentFlag[i] != 0.0).collect();
    let control: Vec<usize> = (0..n).filter(|&i| treatmentFlag[i] == 0.0).collect();
    if treated.is_empty() || control.is_empty() {
        return Err(ZyronError::InvalidParameter {
            name: "treatment".to_string(),
            value: "both treatment groups must be nonempty".to_string(),
        });
    }
    let sq_dist = |a: &[f64], b: &[f64]| -> f64 {
        let mut s = 0.0f64;
        for j in 0..p {
            let d = a[j] - b[j];
            s += d * d;
        }
        s
    };
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        let is_treated = treatmentFlag[i] != 0.0;
        let pool = if is_treated { &control } else { &treated };
        let mut best_idx = pool[0];
        let mut best_d = f64::INFINITY;
        for &j in pool {
            let d = sq_dist(&standardized[i], &standardized[j]);
            if d < best_d {
                best_d = d;
                best_idx = j;
            }
        }
        let cf = observed[best_idx];
        let effect = if is_treated {
            observed[i] - cf
        } else {
            cf - observed[i]
        };
        out.push(CounterfactualRow {
            index: i,
            observed: observed[i],
            counterfactual: cf,
            effect,
        });
    }
    Ok(out)
}

// ===== Propensity score matching =====

/// Greedy nearest neighbor matching of treated to control units on the
/// propensity score, skipping candidates beyond the caliper
/// Returns (treatedIndex, controlIndex, scoreDifference) pairs, each
/// control matched at most once
pub fn propensityMatch(
    treated: &[Vec<f64>],
    control: &[Vec<f64>],
    caliper: f64,
) -> Result<Vec<(usize, usize, f64)>> {
    if treated.is_empty() || control.is_empty() {
        return Err(ZyronError::InvalidParameter {
            name: "groups".to_string(),
            value: "treated and control groups must both be nonempty".to_string(),
        });
    }
    if !(caliper > 0.0 && caliper.is_finite()) {
        return Err(ZyronError::InvalidParameter {
            name: "caliper".to_string(),
            value: format!("must be a positive finite number, got {}", caliper),
        });
    }
    let p = treated[0].len();
    if p == 0 {
        return Err(ZyronError::InvalidParameter {
            name: "groups".to_string(),
            value: "feature rows have no columns".to_string(),
        });
    }
    let n_treated = treated.len();
    let n_control = control.len();
    let mut covariates = Vec::with_capacity((n_treated + n_control) * p);
    let mut treatment = Vec::with_capacity(n_treated + n_control);
    for r in treated.iter().chain(control.iter()) {
        if r.len() != p {
            return Err(ZyronError::InvalidParameter {
                name: "groups".to_string(),
                value: "feature rows have inconsistent dimensions".to_string(),
            });
        }
        covariates.extend_from_slice(r);
    }
    treatment.extend(std::iter::repeat(1.0).take(n_treated));
    treatment.extend(std::iter::repeat(0.0).take(n_control));
    let scores = propensityScore(&treatment, &covariates, p)?;
    let treated_scores = &scores[..n_treated];
    let control_scores = &scores[n_treated..];
    let mut matched = vec![false; n_control];
    let mut pairs = Vec::new();
    for (ti, &ts) in treated_scores.iter().enumerate() {
        let mut best_ci = None;
        let mut best_diff = f64::INFINITY;
        for (ci, &cs) in control_scores.iter().enumerate() {
            if matched[ci] {
                continue;
            }
            let diff = (ts - cs).abs();
            if diff < best_diff {
                best_diff = diff;
                best_ci = Some(ci);
            }
        }
        if let Some(ci) = best_ci {
            if best_diff <= caliper {
                matched[ci] = true;
                pairs.push((ti, ci, best_diff));
            }
        }
    }
    Ok(pairs)
}

// ===== A/B test analysis =====

#[derive(Debug, Clone)]
pub struct AbVariantStats {
    pub variant: String,
    pub n: u64,
    pub mean: f64,
    // difference in means against the first variant, None on the control
    pub lift: Option<f64>,
    pub ciLower: Option<f64>,
    pub ciUpper: Option<f64>,
    pub pValue: Option<f64>,
}

/// Pairwise comparison of every variant against the first observed one
/// metricType proportion uses the pooled two proportion z test,
/// continuous uses the Welch t statistic with the Welch Satterthwaite
/// degrees of freedom, p values via the Wallace normal approximation of
/// the t tail. Confidence intervals are 95 percent on the difference
pub fn abTestAnalysis(
    variants: &[String],
    outcomes: &[f64],
    metricType: &str,
) -> Result<Vec<AbVariantStats>> {
    let n = variants.len();
    if n == 0 || outcomes.len() != n {
        return Err(ZyronError::InvalidParameter {
            name: "variants".to_string(),
            value: "variants and outcomes must be nonempty and equal length".to_string(),
        });
    }
    let lower = metricType.to_ascii_lowercase();
    let is_proportion = match lower.as_str() {
        "proportion" => true,
        "continuous" => false,
        other => {
            return Err(ZyronError::InvalidParameter {
                name: "metric_type".to_string(),
                value: format!(
                    "'{}' is not supported, supported metric types are proportion, continuous",
                    other
                ),
            });
        }
    };
    if outcomes.iter().any(|v| !v.is_finite()) {
        return Err(ZyronError::InvalidParameter {
            name: "outcomes".to_string(),
            value: "outcomes contain non finite values".to_string(),
        });
    }
    if is_proportion && outcomes.iter().any(|v| *v != 0.0 && *v != 1.0) {
        return Err(ZyronError::InvalidParameter {
            name: "outcomes".to_string(),
            value: "proportion metric requires 0 or 1 outcomes".to_string(),
        });
    }
    // group in first appearance order, first group is the control
    let mut groups: Vec<(String, Vec<f64>)> = Vec::new();
    for i in 0..n {
        match groups.iter_mut().find(|(name, _)| *name == variants[i]) {
            Some((_, vals)) => vals.push(outcomes[i]),
            None => groups.push((variants[i].clone(), vec![outcomes[i]])),
        }
    }
    let stats = |vals: &[f64]| -> (f64, f64, f64) {
        let cnt = vals.len() as f64;
        let mean = vals.iter().sum::<f64>() / cnt;
        let var = if vals.len() > 1 {
            vals.iter().map(|v| (v - mean) * (v - mean)).sum::<f64>() / (cnt - 1.0)
        } else {
            0.0
        };
        (cnt, mean, var)
    };
    let (n0, mean0, var0) = stats(&groups[0].1);
    let z95 = normalQuantile(0.975);
    let mut out = Vec::with_capacity(groups.len());
    out.push(AbVariantStats {
        variant: groups[0].0.clone(),
        n: groups[0].1.len() as u64,
        mean: mean0,
        lift: None,
        ciLower: None,
        ciUpper: None,
        pValue: None,
    });
    for (name, vals) in groups.iter().skip(1) {
        let (n1, mean1, var1) = stats(vals);
        let diff = mean1 - mean0;
        let (se, p) = if is_proportion {
            let se_ci = (mean0 * (1.0 - mean0) / n0 + mean1 * (1.0 - mean1) / n1).sqrt();
            let pooled = (mean0 * n0 + mean1 * n1) / (n0 + n1);
            let se_z = (pooled * (1.0 - pooled) * (1.0 / n0 + 1.0 / n1)).sqrt();
            let z = if se_z > 0.0 { diff / se_z } else { 0.0 };
            (se_ci, 2.0 * (1.0 - normalCdf(z.abs())))
        } else {
            let se = (var0 / n0 + var1 / n1).sqrt();
            let t = if se > 0.0 { diff / se } else { 0.0 };
            // Welch Satterthwaite degrees of freedom
            let df_num = (var0 / n0 + var1 / n1) * (var0 / n0 + var1 / n1);
            let df_den = if n0 > 1.0 && n1 > 1.0 {
                (var0 / n0) * (var0 / n0) / (n0 - 1.0) + (var1 / n1) * (var1 / n1) / (n1 - 1.0)
            } else {
                0.0
            };
            let df = if df_den > 0.0 { df_num / df_den } else { 1.0 };
            // Wallace normal approximation of the t tail at df
            let z_equiv = t.abs() * (1.0 - 1.0 / (4.0 * df)) / (1.0 + t * t / (2.0 * df)).sqrt();
            (se, 2.0 * (1.0 - normalCdf(z_equiv)))
        };
        out.push(AbVariantStats {
            variant: name.clone(),
            n: vals.len() as u64,
            mean: mean1,
            lift: Some(diff),
            ciLower: Some(diff - z95 * se),
            ciUpper: Some(diff + z95 * se),
            pValue: Some(p.clamp(0.0, 1.0)),
        });
    }
    Ok(out)
}

/// Required sample size per variant for a two proportion test
/// baselineConversion is the control rate, mde the absolute minimum
/// detectable effect, power and alpha the usual operating levels
pub fn abTestSampleSize(baselineConversion: f64, mde: f64, power: f64, alpha: f64) -> Result<f64> {
    let p1 = baselineConversion;
    let p2 = baselineConversion + mde;
    if !(p1 > 0.0 && p1 < 1.0) {
        return Err(ZyronError::InvalidParameter {
            name: "baseline_conversion".to_string(),
            value: format!("must be in (0, 1), got {}", p1),
        });
    }
    if mde == 0.0 || !(p2 > 0.0 && p2 < 1.0) {
        return Err(ZyronError::InvalidParameter {
            name: "mde".to_string(),
            value: format!(
                "must be nonzero and keep baseline + mde inside (0, 1), got {}",
                mde
            ),
        });
    }
    if !(power > 0.0 && power < 1.0) {
        return Err(ZyronError::InvalidParameter {
            name: "power".to_string(),
            value: format!("must be in (0, 1), got {}", power),
        });
    }
    if !(alpha > 0.0 && alpha < 1.0) {
        return Err(ZyronError::InvalidParameter {
            name: "alpha".to_string(),
            value: format!("must be in (0, 1), got {}", alpha),
        });
    }
    let z_alpha = normalQuantile(1.0 - alpha / 2.0);
    let z_power = normalQuantile(power);
    let p_bar = (p1 + p2) / 2.0;
    let term = z_alpha * (2.0 * p_bar * (1.0 - p_bar)).sqrt()
        + z_power * (p1 * (1.0 - p1) + p2 * (1.0 - p2)).sqrt();
    Ok((term * term) / (mde * mde))
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

    #[test]
    fn causalImpactDetectsStepChange() {
        let mut rng = Xoshiro256pp::fromSeed(81);
        let n = 80usize;
        let split = 50usize;
        let mut outcome = Vec::with_capacity(n);
        let mut timestamps = Vec::with_capacity(n);
        for i in 0..n {
            let base = if i < split { 10.0 } else { 20.0 };
            outcome.push(base + 0.2 * rng.nextNormal());
            timestamps.push(i as i64 * 1000);
        }
        let points = causalImpact(&outcome, &timestamps, split).expect("impact");
        assert_eq!(points.len(), n - split);
        let last = points.last().expect("post period rows");
        assert!(last.cumulativeEffect > 0.0);
        assert!(
            last.cumulativeLower > 0.0,
            "cumulative lower = {}",
            last.cumulativeLower
        );
        let avg_effect = last.cumulativeEffect / (n - split) as f64;
        assert!(
            (avg_effect - 10.0).abs() < 1.5,
            "avg effect = {}",
            avg_effect
        );
        for p in &points {
            assert!(p.effect > 5.0, "point effect = {}", p.effect);
            assert!(p.lower <= p.effect && p.effect <= p.upper);
        }
    }

    #[test]
    fn causalImpactValidatesInterventionIndex() {
        let outcome = vec![1.0, 2.0, 3.0, 4.0];
        let ts = vec![0i64, 1, 2, 3];
        assert!(causalImpact(&outcome, &ts, 2).is_err());
        assert!(causalImpact(&outcome, &ts, 4).is_err());
    }

    #[test]
    fn counterfactualMatchesOppositeGroup() {
        // treated units mirror control units in feature space
        let observed = vec![10.0, 12.0, 5.0, 6.0];
        let flags = vec![1.0, 1.0, 0.0, 0.0];
        let features = vec![vec![1.0], vec![2.0], vec![1.05], vec![2.05]];
        let rows = counterfactual(&observed, &flags, &features).expect("cf");
        assert!((rows[0].counterfactual - 5.0).abs() < 1e-12);
        assert!((rows[0].effect - 5.0).abs() < 1e-12);
        assert!((rows[2].counterfactual - 10.0).abs() < 1e-12);
        assert!((rows[2].effect - 5.0).abs() < 1e-12);
    }

    #[test]
    fn propensityMatchRespectsCaliper() {
        let mut rng = Xoshiro256pp::fromSeed(91);
        let treated: Vec<Vec<f64>> = (0..30)
            .map(|_| vec![1.0 + 0.3 * rng.nextNormal(), 0.5 * rng.nextNormal()])
            .collect();
        let control: Vec<Vec<f64>> = (0..60)
            .map(|_| vec![0.3 * rng.nextNormal(), 0.5 * rng.nextNormal()])
            .collect();
        let pairs = propensityMatch(&treated, &control, 0.2).expect("match");
        let mut seen = std::collections::HashSet::new();
        for (ti, ci, diff) in &pairs {
            assert!(*ti < 30 && *ci < 60);
            assert!(*diff <= 0.2);
            assert!(seen.insert(*ci), "control {} matched twice", ci);
        }
        assert!(propensityMatch(&treated, &control, 0.0).is_err());
    }

    #[test]
    fn abTestAnalysisProportionDetectsLift() {
        let mut variants = Vec::new();
        let mut outcomes = Vec::new();
        for i in 0..1000 {
            variants.push("control".to_string());
            outcomes.push(if i % 10 == 0 { 1.0 } else { 0.0 });
        }
        for i in 0..1000 {
            variants.push("test".to_string());
            outcomes.push(if i % 5 == 0 { 1.0 } else { 0.0 });
        }
        let stats = abTestAnalysis(&variants, &outcomes, "proportion").expect("ab");
        assert_eq!(stats.len(), 2);
        assert!(stats[0].lift.is_none());
        let lift = stats[1].lift.expect("lift");
        assert!((lift - 0.1).abs() < 1e-9);
        assert!(stats[1].pValue.expect("p") < 0.001);
        assert!(stats[1].ciLower.expect("ci") > 0.0);
    }

    #[test]
    fn abTestSampleSizePlausibleRange() {
        let n = abTestSampleSize(0.10, 0.02, 0.8, 0.05).expect("sample size");
        assert!(n > 3000.0 && n < 5000.0, "n = {}", n);
        assert!(abTestSampleSize(0.0, 0.02, 0.8, 0.05).is_err());
        assert!(abTestSampleSize(0.99, 0.02, 0.8, 0.05).is_err());
    }
}
