#![allow(non_snake_case)]
// Binary logistic regression
// Mini-batch SGD with stable sigmoid and L2 regularization
// Outputs P(y=1 | x) in [0,1]

use crate::ml::f64Kernels::{axpy, dot, log1pExp, sigmoid};
use crate::ml::{Hyperparameters, ModelConfig, ModelData, ModelMetrics, ModelType, TrainedModel, TrainingData};
use crate::numeric::{KahanSum, columnStandardize};
use zyron_common::Xoshiro256pp;
use zyron_common::error::{Result, ZyronError};

pub fn train(config: &ModelConfig, data: &TrainingData) -> Result<TrainedModel> {
    if data.n == 0 {
        return Err(ZyronError::InvalidParameter {
            name: "training_data".to_string(),
            value: "empty".to_string(),
        });
    }
    if data.p == 0 {
        return Err(ZyronError::InvalidParameter {
            name: "feature_count".to_string(),
            value: "zero".to_string(),
        });
    }
    let lambda = config.hyperparameters.getF64Or("lambda", 0.0).max(0.0);
    let lr = config.hyperparameters.getF64Or("learning_rate", 0.1);
    let epochs = config.hyperparameters.getUsizeOr("max_epochs", 100);
    let batchSize = config.hyperparameters.getUsizeOr("batch_size", 1024).max(1).min(data.n);
    let tol = config.hyperparameters.getF64Or("tolerance", 1e-5);
    let seed = config.hyperparameters.getU64Or("seed", 42);
    let standardize = config.hyperparameters.getBoolOr("standardize", true);

    for &y in data.ys {
        if !(y == 0.0 || y == 1.0) {
            return Err(ZyronError::InvalidParameter {
                name: "target".to_string(),
                value: format!("expected binary 0/1, found {}", y),
            });
        }
    }

    let (mean, std, xs) = if standardize {
        let (mean, std) = columnStandardize(data.xs, data.n, data.p);
        let mut xs = vec![0.0f64; data.n * data.p];
        for i in 0..data.n {
            for j in 0..data.p {
                xs[i * data.p + j] = (data.xs[i * data.p + j] - mean[j]) / std[j];
            }
        }
        (mean, std, xs)
    } else {
        (vec![0.0; data.p], vec![1.0; data.p], data.xs.to_vec())
    };

    let p = data.p;
    let mut weights = vec![0.0f64; p + 1];
    let mut grad = vec![0.0f64; p + 1];
    let mut order: Vec<usize> = (0..data.n).collect();
    let mut rng = Xoshiro256pp::fromSeed(seed);

    // Scratch buffer for per-row scaled errors. Indexes directly into
    // the source xs by `order[idx..end]`, avoiding the gather-copy of
    // rows into a contiguous batch buffer. axpy is dispatched once at
    // the top of the inner loop and called as a fn-pointer per row,
    // saving the OnceLock probe each call
    let mut batchErrors = vec![0.0f64; batchSize];
    let axpyFn = crate::ml::f64Kernels::resolveAxpyFn();
    let dotFn = crate::ml::f64Kernels::resolveDot();

    let mut prevLoss = f64::INFINITY;
    for _epoch in 0..epochs {
        rng.shuffle(&mut order);
        let mut idx = 0;
        while idx < data.n {
            let end = (idx + batchSize).min(data.n);
            let batchN = end - idx;
            let invN = 1.0f64 / batchN as f64;
            for (slot, &i) in order[idx..end].iter().enumerate() {
                let row = &xs[i * p..i * p + p];
                let z = unsafe { dotFn(row.as_ptr(), weights[..p].as_ptr(), p) } + weights[p];
                let err = (sigmoid(z) - data.ys[i]) * invN;
                batchErrors[slot] = err;
            }
            for v in grad.iter_mut() {
                *v = 0.0;
            }
            // Accumulate sum_i err[i] * xs[order[i]] into grad. Indexing
            // through order avoids the row copy from the previous form
            for (slot, &i) in order[idx..end].iter().enumerate() {
                let err = batchErrors[slot];
                if err == 0.0 {
                    continue;
                }
                let row = &xs[i * p..i * p + p];
                unsafe {
                    axpyFn(err, row.as_ptr(), grad[..p].as_mut_ptr(), p);
                }
            }
            let mut bias = 0.0f64;
            for k in 0..batchN {
                bias += batchErrors[k];
            }
            grad[p] = bias;
            if lambda > 0.0 {
                for j in 0..p {
                    grad[j] += lambda * weights[j];
                }
            }
            axpy(-lr, &grad[..p], &mut weights[..p]);
            weights[p] -= lr * grad[p];
            idx = end;
        }
        // Convergence loss estimated on a deterministic stride sample of
        // up to 4096 rows. Full-data loss for the convergence check adds
        // O(n*p) per epoch and a stride estimate is more than accurate
        // enough for delta-based stopping
        let lossN = data.n.min(4096);
        let stride = (data.n / lossN).max(1);
        let mut loss = KahanSum::new();
        let mut sampled = 0usize;
        let mut i = 0usize;
        while i < data.n {
            let row = &xs[i * p..i * p + p];
            let z = dot(row, &weights[..p]) + weights[p];
            loss.add(log1pExp(z) - data.ys[i] * z);
            sampled += 1;
            i += stride;
        }
        let mLoss = loss.value() / sampled.max(1) as f64;
        if (prevLoss - mLoss).abs() < tol {
            break;
        }
        prevLoss = mLoss;
    }

    let mut model = TrainedModel::new(String::new(), ModelType::LogisticRegression);
    model.featureColumns = config.featureColumns.clone();
    model.targetColumn = config.targetColumn.clone();
    model.featureMean = mean;
    model.featureStd = std;
    model.weights = weights;
    model.hyperparameters = config.hyperparameters.clone();
    model.trainingRows = data.n as u64;
    model.metrics = computeClassificationMetrics(&model, &xs, data.ys).intoMap();
    let _ = ModelData::None;
    Ok(model)
}

pub fn predictProbability(model: &TrainedModel, features: &[f64]) -> f64 {
    let p = model.featureColumns.len();
    debug_assert_eq!(features.len(), p);
    let mut z = 0.0f64;
    for j in 0..p {
        let std = model.featureStd.get(j).copied().unwrap_or(1.0);
        let mean = model.featureMean.get(j).copied().unwrap_or(0.0);
        let denom = if std == 0.0 { 1.0 } else { std };
        let v = (features[j] - mean) / denom;
        z += model.weights[j] * v;
    }
    z += model.weights[p];
    sigmoid(z)
}

pub fn predictClass(model: &TrainedModel, features: &[f64], threshold: f64) -> f64 {
    if predictProbability(model, features) >= threshold {
        1.0
    } else {
        0.0
    }
}

pub fn predictBatchProbability(model: &TrainedModel, xs: &[f64], n: usize, out: &mut [f64]) {
    let p = model.featureColumns.len();
    debug_assert_eq!(xs.len(), n * p);
    debug_assert_eq!(out.len(), n);
    for i in 0..n {
        out[i] = predictProbability(model, &xs[i * p..i * p + p]);
    }
}

fn computeClassificationMetrics(model: &TrainedModel, xs: &[f64], ys: &[f64]) -> ModelMetrics {
    let p = model.featureColumns.len();
    let n = ys.len();
    if n == 0 {
        return ModelMetrics::default();
    }
    let mut tp = 0u64;
    let mut tn = 0u64;
    let mut fp = 0u64;
    let mut fnCount = 0u64;
    let mut logLossSum = KahanSum::new();
    for i in 0..n {
        let row = &xs[i * p..i * p + p];
        let mut z = 0.0f64;
        for j in 0..p {
            z += model.weights[j] * row[j];
        }
        z += model.weights[p];
        let prob = sigmoid(z);
        let yhat = if prob >= 0.5 { 1.0 } else { 0.0 };
        let y = ys[i];
        if y == 1.0 && yhat == 1.0 {
            tp += 1;
        } else if y == 0.0 && yhat == 0.0 {
            tn += 1;
        } else if y == 0.0 && yhat == 1.0 {
            fp += 1;
        } else {
            fnCount += 1;
        }
        let pClamped = prob.clamp(1e-12, 1.0 - 1e-12);
        logLossSum.add(-(y * pClamped.ln() + (1.0 - y) * (1.0 - pClamped).ln()));
    }
    let mut m = ModelMetrics::default();
    let total = (tp + tn + fp + fnCount) as f64;
    m.accuracy = Some((tp + tn) as f64 / total);
    let precDenom = (tp + fp) as f64;
    m.precision = Some(if precDenom > 0.0 { tp as f64 / precDenom } else { 0.0 });
    let recallDenom = (tp + fnCount) as f64;
    m.recall = Some(if recallDenom > 0.0 { tp as f64 / recallDenom } else { 0.0 });
    let p = m.precision.unwrap();
    let r = m.recall.unwrap();
    m.f1Score = Some(if p + r > 0.0 { 2.0 * p * r / (p + r) } else { 0.0 });
    m.logLoss = Some(logLossSum.value() / n as f64);
    m
}

#[cfg(test)]
mod tests {
    use super::*;

    fn buildSeparableData(n: usize, seed: u64) -> (Vec<f64>, Vec<f64>) {
        let mut rng = Xoshiro256pp::fromSeed(seed);
        let mut xs = Vec::with_capacity(n * 2);
        let mut ys = Vec::with_capacity(n);
        for _ in 0..n {
            let cls = (rng.nextRange(2)) as f64;
            let mu = if cls == 1.0 { 2.0 } else { -2.0 };
            xs.push(mu + 0.7 * rng.nextNormal());
            xs.push(mu + 0.7 * rng.nextNormal());
            ys.push(cls);
        }
        (xs, ys)
    }

    #[test]
    fn convergesOnSeparableData() {
        let (xs, ys) = buildSeparableData(2000, 42);
        let mut config = ModelConfig::new(
            ModelType::LogisticRegression,
            vec!["x1".to_string(), "x2".to_string()],
        );
        config.targetColumn = Some("y".to_string());
        config.hyperparameters.setF64("learning_rate", 0.1);
        config.hyperparameters.setF64("max_epochs", 100.0);
        let data = TrainingData::new(&xs, &ys, ys.len(), 2);
        let model = train(&config, &data).unwrap();
        let acc = model.metrics.get("accuracy").copied().unwrap();
        assert!(acc > 0.9, "accuracy = {}", acc);
        // Predicted probabilities should be in [0, 1]
        for i in 0..50 {
            let row = &xs[i * 2..i * 2 + 2];
            let p = predictProbability(&model, row);
            assert!((0.0..=1.0).contains(&p));
        }
    }

    #[test]
    fn rejectsNonBinaryTarget() {
        let xs = vec![0.0, 0.0];
        let ys = vec![0.5];
        let config = ModelConfig::new(ModelType::LogisticRegression, vec!["x".to_string()]);
        let data = TrainingData::new(&xs, &ys, 1, 2);
        let result = train(&config, &data);
        assert!(result.is_err());
    }
}
