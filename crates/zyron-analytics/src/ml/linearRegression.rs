#![allow(non_snake_case)]
// Linear regression
// Closed-form normal equations with Cholesky for p <= 1024
// Mini-batch SGD fallback for higher dimensions
// Optional ridge regularization (lambda)

use crate::ml::f64Kernels::{addInPlace, axpy, dot, scaleInPlace};
use crate::ml::{Hyperparameters, ModelConfig, ModelData, ModelMetrics, ModelType, TrainedModel, TrainingData};
use crate::numeric::{KahanSum, choleskySolve, columnStandardize};
use zyron_common::Xoshiro256pp;
use zyron_common::error::{Result, ZyronError};

const CLOSED_FORM_LIMIT: usize = 1024;

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
    let standardize = config.hyperparameters.getBoolOr("standardize", true);
    let useClosedForm = data.p + 1 <= CLOSED_FORM_LIMIT
        && config.hyperparameters.getStr("optimizer").unwrap_or("auto") != "sgd";
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

    let weights = if useClosedForm {
        trainClosedForm(&xs, data.ys, data.n, data.p, lambda)?
    } else {
        trainSgd(&xs, data.ys, data.n, data.p, &config.hyperparameters)?
    };

    // Bake the standardization transform into the weights so inference
    // is a single dot product with no per-feature divide.
    // standardized = (raw - mean) / std
    // y = sum_j w_j * (raw_j - mean_j) / std_j + b
    //   = sum_j (w_j / std_j) * raw_j + (b - sum_j w_j * mean_j / std_j)
    let mut bakedWeights = vec![0.0f64; data.p + 1];
    let mut biasOffset = 0.0f64;
    for j in 0..data.p {
        let s = if std[j] == 0.0 { 1.0 } else { std[j] };
        bakedWeights[j] = weights[j] / s;
        biasOffset += weights[j] * mean[j] / s;
    }
    bakedWeights[data.p] = weights[data.p] - biasOffset;

    let mut model = TrainedModel::new(String::new(), ModelType::LinearRegression);
    model.featureColumns = config.featureColumns.clone();
    model.targetColumn = config.targetColumn.clone();
    // featureMean and featureStd are zeroed/identity so inference paths
    // that consult them on legacy models do not double-apply the transform.
    // Models persisted to disk before this change still have the original
    // mean/std and the per-feature path corrects the math; freshly trained
    // models go through the prebaked fast path below
    model.featureMean = vec![0.0; data.p];
    model.featureStd = vec![1.0; data.p];
    model.weights = bakedWeights;
    model.hyperparameters = config.hyperparameters.clone();
    model.trainingRows = data.n as u64;
    model.metrics = computeRegressionMetrics(&model, data.xs, data.ys).intoMap();
    Ok(model)
}

fn trainClosedForm(xs: &[f64], ys: &[f64], n: usize, p: usize, lambda: f64) -> Result<Vec<f64>> {
    // Build augmented design with intercept column at index p.
    // A = X^T X is symmetric so accumulate only the upper triangle then
    // mirror to the lower at the end. The inner loop is half the size
    // and the compiler vectorises it cleanly. Also accumulate the
    // intercept row+column once per outer pass rather than per j inside
    let pAug = p + 1;
    let mut a = vec![0.0f64; pAug * pAug];
    let mut b = vec![0.0f64; pAug];
    for i in 0..n {
        let row = &xs[i * p..i * p + p];
        let yi = ys[i];
        for j in 0..p {
            let rj = row[j];
            // Upper triangle accumulation: k in [j..p)
            let base = j * pAug;
            for k in j..p {
                a[base + k] += rj * row[k];
            }
            // Intercept column at position p
            a[base + p] += rj;
            b[j] += rj * yi;
        }
        a[p * pAug + p] += 1.0;
        b[p] += yi;
    }
    if lambda > 0.0 {
        for j in 0..p {
            a[j * pAug + j] += lambda;
        }
    }
    // Mirror upper triangle to lower so Cholesky sees a complete matrix
    for j in 0..pAug {
        for k in 0..j {
            a[j * pAug + k] = a[k * pAug + j];
        }
    }
    choleskySolve(&mut a, &mut b, pAug)?;
    Ok(b)
}

fn trainSgd(
    xs: &[f64],
    ys: &[f64],
    n: usize,
    p: usize,
    hp: &Hyperparameters,
) -> Result<Vec<f64>> {
    let lr = hp.getF64Or("learning_rate", 0.01);
    let lambda = hp.getF64Or("lambda", 0.0);
    let epochs = hp.getUsizeOr("max_epochs", 200);
    let batchSize = hp.getUsizeOr("batch_size", 1024).max(1).min(n);
    let tol = hp.getF64Or("tolerance", 1e-5);
    let seed = hp.getU64Or("seed", 42);

    let mut weights = vec![0.0f64; p + 1];
    let mut grad = vec![0.0f64; p + 1];
    let mut order: Vec<usize> = (0..n).collect();
    let mut rng = Xoshiro256pp::fromSeed(seed);

    let mut prevLoss = f64::INFINITY;
    for _ in 0..epochs {
        rng.shuffle(&mut order);
        let mut idx = 0;
        while idx < n {
            let end = (idx + batchSize).min(n);
            let batchN = end - idx;
            for v in grad.iter_mut() {
                *v = 0.0;
            }
            for &i in &order[idx..end] {
                let row = &xs[i * p..i * p + p];
                let pred = dot(row, &weights[..p]) + weights[p];
                let err = pred - ys[i];
                axpy(err / batchN as f64, row, &mut grad[..p]);
                grad[p] += err / batchN as f64;
            }
            // L2 penalty on non-intercept weights
            if lambda > 0.0 {
                for j in 0..p {
                    grad[j] += lambda * weights[j];
                }
            }
            // weights -= lr * grad
            axpy(-lr, &grad[..p], &mut weights[..p]);
            weights[p] -= lr * grad[p];
            idx = end;
        }
        let mut s = KahanSum::new();
        for i in 0..n {
            let row = &xs[i * p..i * p + p];
            let pred = dot(row, &weights[..p]) + weights[p];
            let err = pred - ys[i];
            s.add(err * err);
        }
        let loss = s.value() / n as f64;
        if (prevLoss - loss).abs() < tol {
            break;
        }
        prevLoss = loss;
    }
    let _ = addInPlace; // silence unused warning if not used elsewhere
    let _ = scaleInPlace;
    Ok(weights)
}

pub fn predict(model: &TrainedModel, features: &[f64]) -> f64 {
    let p = model.featureColumns.len();
    debug_assert_eq!(features.len(), p);
    // Fast path for freshly-trained models with baked-in transform:
    // mean is all zeros, std is all ones, so prediction is one dot
    // product plus a bias. Detect via featureStd values
    let prebaked = model.featureStd.iter().all(|s| *s == 1.0)
        && model.featureMean.iter().all(|m| *m == 0.0);
    if prebaked {
        return crate::ml::f64Kernels::dot(features, &model.weights[..p]) + model.weights[p];
    }
    // Legacy path for older serialized models that still carry the
    // per-feature mean/std transform separately
    let mut s = 0.0f64;
    for j in 0..p {
        let std = model.featureStd.get(j).copied().unwrap_or(1.0);
        let mean = model.featureMean.get(j).copied().unwrap_or(0.0);
        let denom = if std == 0.0 { 1.0 } else { std };
        s += model.weights[j] * (features[j] - mean) / denom;
    }
    s + model.weights[p]
}

pub fn predictBatch(model: &TrainedModel, xs: &[f64], n: usize, out: &mut [f64]) {
    let p = model.featureColumns.len();
    debug_assert_eq!(xs.len(), n * p);
    debug_assert_eq!(out.len(), n);
    let prebaked = model.featureStd.iter().all(|s| *s == 1.0)
        && model.featureMean.iter().all(|m| *m == 0.0);
    if prebaked {
        crate::ml::f64Kernels::rowMajorMatvec(
            xs,
            &model.weights[..p],
            model.weights[p],
            n,
            p,
            out,
        );
        return;
    }
    for i in 0..n {
        out[i] = predict(model, &xs[i * p..i * p + p]);
    }
}

fn computeRegressionMetrics(model: &TrainedModel, xs: &[f64], ys: &[f64]) -> ModelMetrics {
    let p = model.featureColumns.len();
    let n = ys.len();
    if n == 0 {
        return ModelMetrics::default();
    }
    let mut yMean = 0.0f64;
    for &y in ys {
        yMean += y;
    }
    yMean /= n as f64;
    let mut ssRes = 0.0f64;
    let mut absSum = 0.0f64;
    let mut ssTot = 0.0f64;
    for i in 0..n {
        let row = &xs[i * p..i * p + p];
        let mut pred = 0.0f64;
        for j in 0..p {
            pred += model.weights[j] * row[j];
        }
        pred += model.weights[p];
        let err = pred - ys[i];
        ssRes += err * err;
        absSum += err.abs();
        let dy = ys[i] - yMean;
        ssTot += dy * dy;
    }
    let mut m = ModelMetrics::default();
    m.rmse = Some((ssRes / n as f64).sqrt());
    m.mae = Some(absSum / n as f64);
    m.rSquared = if ssTot > 0.0 {
        Some(1.0 - ssRes / ssTot)
    } else {
        Some(0.0)
    };
    m
}

#[cfg(test)]
mod tests {
    use super::*;

    fn buildLinearData(n: usize, slope: f64, intercept: f64, noise: f64, seed: u64) -> (Vec<f64>, Vec<f64>) {
        let mut rng = Xoshiro256pp::fromSeed(seed);
        let mut xs = Vec::with_capacity(n);
        let mut ys = Vec::with_capacity(n);
        for i in 0..n {
            let x = i as f64 * 0.1;
            xs.push(x);
            ys.push(slope * x + intercept + noise * rng.nextNormal());
        }
        (xs, ys)
    }

    #[test]
    fn closedFormRecoversCoefficients() {
        let (xs, ys) = buildLinearData(2000, 2.0, 3.0, 0.01, 1);
        let mut config = ModelConfig::new(ModelType::LinearRegression, vec!["x".to_string()]);
        config.targetColumn = Some("y".to_string());
        let data = TrainingData::new(&xs, &ys, xs.len(), 1);
        let model = train(&config, &data).unwrap();
        let r2 = model.metrics.get("r_squared").copied().unwrap();
        assert!(r2 > 0.99, "r2 = {}", r2);
        // Predict at known x values
        let yhat0 = predict(&model, &[0.0]);
        let yhat10 = predict(&model, &[10.0]);
        assert!((yhat0 - 3.0).abs() < 0.5, "intercept estimate: {}", yhat0);
        assert!((yhat10 - 23.0).abs() < 0.5, "slope+intercept: {}", yhat10);
    }

    #[test]
    fn ridgeRegularizes() {
        let (xs, ys) = buildLinearData(100, 2.0, 3.0, 1.0, 7);
        let mut config = ModelConfig::new(ModelType::LinearRegression, vec!["x".to_string()]);
        config.hyperparameters.setF64("lambda", 1.0);
        let data = TrainingData::new(&xs, &ys, xs.len(), 1);
        let _ = train(&config, &data).unwrap();
    }

    #[test]
    fn sgdConverges() {
        let (xs, ys) = buildLinearData(500, 2.0, 3.0, 0.05, 99);
        let mut config = ModelConfig::new(ModelType::LinearRegression, vec!["x".to_string()]);
        config.hyperparameters.setStr("optimizer", "sgd");
        config.hyperparameters.setF64("learning_rate", 0.05);
        config.hyperparameters.setF64("max_epochs", 200.0);
        let data = TrainingData::new(&xs, &ys, xs.len(), 1);
        let model = train(&config, &data).unwrap();
        let r2 = model.metrics.get("r_squared").copied().unwrap();
        assert!(r2 > 0.95, "sgd r2 = {}", r2);
    }
}

#[allow(dead_code)]
fn _useImports() {
    // Keep import refs alive across feature gates
    let _ = ModelData::None;
}
