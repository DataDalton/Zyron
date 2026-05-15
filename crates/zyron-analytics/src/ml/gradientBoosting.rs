#![allow(non_snake_case)]
// Gradient boosting on regression trees
// Regression: squared loss, leaves fit residuals directly
// Binary classification: log-odds residuals against logistic loss

use crate::ml::decisionTree::{TreeBuildConfig, predictTree, trainTree};
use crate::ml::f64Kernels::sigmoid;
use crate::ml::{
    ModelConfig, ModelData, ModelMetrics, ModelType, TrainedModel, TreeNode, TrainingData,
};
use zyron_common::error::{Result, ZyronError};

pub fn train(config: &ModelConfig, data: &TrainingData) -> Result<TrainedModel> {
    if data.n == 0 || data.p == 0 {
        return Err(ZyronError::InvalidParameter {
            name: "training_data".to_string(),
            value: "empty".to_string(),
        });
    }
    let regression = matches!(config.modelType, ModelType::GradientBoostingRegression);
    if !regression {
        for &y in data.ys {
            if !(y == 0.0 || y == 1.0) {
                return Err(ZyronError::InvalidParameter {
                    name: "target".to_string(),
                    value: format!("expected binary 0/1 for boosting classifier, got {}", y),
                });
            }
        }
    }

    let nTrees = config.hyperparameters.getUsizeOr("n_trees", 100).max(1);
    let learningRate = config
        .hyperparameters
        .getF64Or("learning_rate", 0.1)
        .max(1e-4);
    let mut treeHp = config.hyperparameters.clone();
    if !treeHp.values.contains_key("max_depth") {
        treeHp.setF64("max_depth", 3.0);
    }
    let buildConfig = TreeBuildConfig::fromHp(&treeHp, true);

    let baseScore: f64 = if regression {
        data.ys.iter().sum::<f64>() / data.n as f64
    } else {
        let p = (data.ys.iter().sum::<f64>() / data.n as f64).clamp(1e-6, 1.0 - 1e-6);
        (p / (1.0 - p)).ln()
    };

    let mut current = vec![baseScore; data.n];
    let mut residuals = vec![0.0f64; data.n];
    let mut trees: Vec<Vec<TreeNode>> = Vec::with_capacity(nTrees);
    // Independent RNG stream per boosting iteration. longJump advances
    // the state by 2^192 calls so successive trees are guaranteed not to
    // share substream prefixes (additive offsets share short runs and
    // give correlated splits across early iterations)
    let mut seedRng = zyron_common::Xoshiro256pp::fromSeed(buildConfig.seed);

    for _t in 0..nTrees {
        if regression {
            for i in 0..data.n {
                residuals[i] = data.ys[i] - current[i];
            }
        } else {
            for i in 0..data.n {
                let prob = sigmoid(current[i]);
                residuals[i] = data.ys[i] - prob;
            }
        }
        let resData = TrainingData::new(data.xs, &residuals, data.n, data.p);
        let mut cfg = buildConfig.clone();
        cfg.seed = seedRng.nextU64();
        seedRng.longJump();
        let tree = trainTree(&resData, &cfg, None, None)?;
        for i in 0..data.n {
            let pred = predictTree(&tree, data.row(i));
            current[i] += learningRate * pred;
        }
        trees.push(tree);
    }

    let mut model = TrainedModel::new(String::new(), config.modelType);
    model.featureColumns = config.featureColumns.clone();
    model.targetColumn = config.targetColumn.clone();
    model.featureMean = vec![0.0; data.p];
    model.featureStd = vec![1.0; data.p];
    model.weights = Vec::new();
    model.data = ModelData::BoostedTrees {
        baseScore,
        learningRate,
        trees: trees.clone(),
    };
    model.hyperparameters = config.hyperparameters.clone();
    model.trainingRows = data.n as u64;
    if regression {
        model.metrics = computeRegressionMetrics(baseScore, learningRate, &trees, data).intoMap();
    } else {
        model.metrics = computeClassificationMetrics(baseScore, learningRate, &trees, data).intoMap();
    }
    Ok(model)
}

pub fn predictRegression(
    baseScore: f64,
    learningRate: f64,
    trees: &[Vec<TreeNode>],
    features: &[f64],
) -> f64 {
    let mut s = baseScore;
    for t in trees {
        s += learningRate * predictTree(t, features);
    }
    s
}

pub fn predictProbability(
    baseScore: f64,
    learningRate: f64,
    trees: &[Vec<TreeNode>],
    features: &[f64],
) -> f64 {
    sigmoid(predictRegression(baseScore, learningRate, trees, features))
}

fn computeRegressionMetrics(
    base: f64,
    lr: f64,
    trees: &[Vec<TreeNode>],
    data: &TrainingData,
) -> ModelMetrics {
    let n = data.n;
    if n == 0 {
        return ModelMetrics::default();
    }
    let mut yMean = 0.0f64;
    for &y in data.ys {
        yMean += y;
    }
    yMean /= n as f64;
    let mut ssRes = 0.0f64;
    let mut absSum = 0.0f64;
    let mut ssTot = 0.0f64;
    for i in 0..n {
        let pred = predictRegression(base, lr, trees, data.row(i));
        let err = pred - data.ys[i];
        ssRes += err * err;
        absSum += err.abs();
        let d = data.ys[i] - yMean;
        ssTot += d * d;
    }
    let mut m = ModelMetrics::default();
    m.rmse = Some((ssRes / n as f64).sqrt());
    m.mae = Some(absSum / n as f64);
    m.rSquared = Some(if ssTot > 0.0 { 1.0 - ssRes / ssTot } else { 0.0 });
    m
}

fn computeClassificationMetrics(
    base: f64,
    lr: f64,
    trees: &[Vec<TreeNode>],
    data: &TrainingData,
) -> ModelMetrics {
    let n = data.n;
    if n == 0 {
        return ModelMetrics::default();
    }
    let mut correct = 0u64;
    let mut tp = 0u64;
    let mut fp = 0u64;
    let mut fnCount = 0u64;
    let mut logLoss = 0.0f64;
    for i in 0..n {
        let prob = predictProbability(base, lr, trees, data.row(i));
        let yhat = if prob >= 0.5 { 1.0 } else { 0.0 };
        let y = data.ys[i];
        if yhat == y {
            correct += 1;
        }
        if y == 1.0 && yhat == 1.0 {
            tp += 1;
        } else if y == 0.0 && yhat == 1.0 {
            fp += 1;
        } else if y == 1.0 && yhat == 0.0 {
            fnCount += 1;
        }
        let pc = prob.clamp(1e-12, 1.0 - 1e-12);
        logLoss += -(y * pc.ln() + (1.0 - y) * (1.0 - pc).ln());
    }
    let mut m = ModelMetrics::default();
    m.accuracy = Some(correct as f64 / n as f64);
    let pd = (tp + fp) as f64;
    let p = if pd > 0.0 { tp as f64 / pd } else { 0.0 };
    let rd = (tp + fnCount) as f64;
    let r = if rd > 0.0 { tp as f64 / rd } else { 0.0 };
    m.precision = Some(p);
    m.recall = Some(r);
    m.f1Score = Some(if p + r > 0.0 { 2.0 * p * r / (p + r) } else { 0.0 });
    m.logLoss = Some(logLoss / n as f64);
    m
}

#[cfg(test)]
mod tests {
    use super::*;
    use zyron_common::Xoshiro256pp;

    #[test]
    fn fitsRegressionWell() {
        let mut rng = Xoshiro256pp::fromSeed(11);
        let n = 600;
        let mut xs = Vec::with_capacity(n * 2);
        let mut ys = Vec::with_capacity(n);
        for _ in 0..n {
            let x1 = rng.nextNormal();
            let x2 = rng.nextNormal();
            xs.push(x1);
            xs.push(x2);
            ys.push(2.0 * x1 + 0.5 * x2 + 0.1 * rng.nextNormal());
        }
        let data = TrainingData::new(&xs, &ys, n, 2);
        let mut config = ModelConfig::new(
            ModelType::GradientBoostingRegression,
            vec!["x1".into(), "x2".into()],
        );
        config.hyperparameters.setF64("n_trees", 50.0);
        config.hyperparameters.setF64("learning_rate", 0.1);
        config.hyperparameters.setF64("max_depth", 3.0);
        let model = train(&config, &data).unwrap();
        let r2 = model.metrics.get("r_squared").copied().unwrap();
        assert!(r2 > 0.85, "gbm r2 = {}", r2);
    }
}
