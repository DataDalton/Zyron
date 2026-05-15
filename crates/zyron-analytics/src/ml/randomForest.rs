#![allow(non_snake_case)]
// Random forest, bagged decision trees with feature subsampling
// Parallel training with std::thread::scope, no rayon dep

use crate::ml::decisionTree::{
    TreeBuildConfig, detectNumClasses, predictTree, trainTree,
};
use crate::ml::{
    ModelConfig, ModelData, ModelMetrics, ModelType, TrainedModel, TreeNode, TrainingData,
};
use zyron_common::Xoshiro256pp;
use zyron_common::error::{Result, ZyronError};

pub fn train(config: &ModelConfig, data: &TrainingData) -> Result<TrainedModel> {
    if data.n == 0 || data.p == 0 {
        return Err(ZyronError::InvalidParameter {
            name: "training_data".to_string(),
            value: "empty".to_string(),
        });
    }
    let regression = matches!(config.modelType, ModelType::RandomForestRegression);
    let nClasses = if !regression {
        Some(detectNumClasses(data.ys))
    } else {
        None
    };

    let nTrees = config.hyperparameters.getUsizeOr("n_trees", 50).max(1);
    let bootstrapFrac = config.hyperparameters.getF64Or("bootstrap_frac", 1.0).clamp(0.1, 1.0);
    let seed = config.hyperparameters.getU64Or("seed", 42);

    let mut treeConfigs = Vec::with_capacity(nTrees);
    let mut rng = Xoshiro256pp::fromSeed(seed);
    for _ in 0..nTrees {
        let mut hp = config.hyperparameters.clone();
        hp.setF64("seed", rng.nextU64() as f64);
        if !hp.values.contains_key("max_features") {
            hp.setF64("feature_subsample", 1.0);
        }
        treeConfigs.push(TreeBuildConfig::fromHp(&hp, regression));
    }

    let mut sampleSets: Vec<Vec<usize>> = Vec::with_capacity(nTrees);
    let m = ((data.n as f64) * bootstrapFrac).max(1.0) as usize;
    for _ in 0..nTrees {
        let mut s = Vec::with_capacity(m);
        for _ in 0..m {
            s.push(rng.nextRange(data.n as u64) as usize);
        }
        sampleSets.push(s);
    }

    // Parallel training
    let trees: Vec<Vec<TreeNode>> = std::thread::scope(|scope| {
        let mut handles = Vec::with_capacity(nTrees);
        for t in 0..nTrees {
            let cfg = treeConfigs[t].clone();
            let samples = sampleSets[t].clone();
            let dataRef = data;
            let nClassesT = nClasses;
            let h = scope.spawn(move || -> Result<Vec<TreeNode>> {
                trainTree(dataRef, &cfg, Some(&samples), nClassesT)
            });
            handles.push(h);
        }
        let mut out: Vec<Vec<TreeNode>> = Vec::with_capacity(nTrees);
        for h in handles {
            match h.join() {
                Ok(Ok(tree)) => out.push(tree),
                Ok(Err(e)) => return Err(e),
                Err(_) => {
                    return Err(ZyronError::ExecutionError(
                        "tree training thread panicked".to_string(),
                    ))
                }
            }
        }
        Ok(out)
    })?;

    let mut model = TrainedModel::new(String::new(), config.modelType);
    model.featureColumns = config.featureColumns.clone();
    model.targetColumn = config.targetColumn.clone();
    model.featureMean = vec![0.0; data.p];
    model.featureStd = vec![1.0; data.p];
    model.weights = if let Some(k) = nClasses {
        vec![k as f64]
    } else {
        Vec::new()
    };
    model.data = ModelData::Forest { trees: trees.clone() };
    model.hyperparameters = config.hyperparameters.clone();
    model.trainingRows = data.n as u64;
    if regression {
        model.metrics = computeRegressionMetrics(&trees, data).intoMap();
    } else {
        model.metrics =
            computeClassificationMetrics(&trees, data, nClasses.unwrap_or(2)).intoMap();
    }
    Ok(model)
}

pub fn predictRegression(trees: &[Vec<TreeNode>], features: &[f64]) -> f64 {
    if trees.is_empty() {
        return 0.0;
    }
    let mut s = 0.0f64;
    for t in trees {
        s += predictTree(t, features);
    }
    s / trees.len() as f64
}

pub fn predictClassification(trees: &[Vec<TreeNode>], features: &[f64], nClasses: usize) -> f64 {
    if trees.is_empty() {
        return 0.0;
    }
    let k = nClasses.max(2);
    // Stack-allocated vote tally for the common case of <= 16 classes,
    // heap fallback for higher class counts. Avoids one Vec allocation
    // per prediction, which matters for batch predict workloads
    if k <= 16 {
        let mut votes = [0u32; 16];
        let limit = k.min(16);
        for t in trees {
            let v = predictTree(t, features);
            let cls = (v.round() as usize).min(limit - 1);
            votes[cls] += 1;
        }
        let mut bestCls = 0usize;
        let mut bestCnt = 0u32;
        for i in 0..limit {
            if votes[i] > bestCnt {
                bestCnt = votes[i];
                bestCls = i;
            }
        }
        return bestCls as f64;
    }
    let mut votes = vec![0u32; k];
    for t in trees {
        let v = predictTree(t, features);
        let cls = (v.round() as usize).min(votes.len() - 1);
        votes[cls] += 1;
    }
    let mut bestCls = 0usize;
    let mut bestCnt = 0u32;
    for (i, &v) in votes.iter().enumerate() {
        if v > bestCnt {
            bestCnt = v;
            bestCls = i;
        }
    }
    bestCls as f64
}

fn computeRegressionMetrics(trees: &[Vec<TreeNode>], data: &TrainingData) -> ModelMetrics {
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
        let pred = predictRegression(trees, data.row(i));
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
    trees: &[Vec<TreeNode>],
    data: &TrainingData,
    nClasses: usize,
) -> ModelMetrics {
    let n = data.n;
    if n == 0 {
        return ModelMetrics::default();
    }
    let mut correct = 0u64;
    let mut tp = 0u64;
    let mut fp = 0u64;
    let mut fnCount = 0u64;
    for i in 0..n {
        let pred = predictClassification(trees, data.row(i), nClasses);
        let yhat = pred as usize;
        let y = data.ys[i] as usize;
        if yhat == y {
            correct += 1;
        }
        if nClasses == 2 {
            if y == 1 && yhat == 1 {
                tp += 1;
            } else if y == 0 && yhat == 1 {
                fp += 1;
            } else if y == 1 && yhat == 0 {
                fnCount += 1;
            }
        }
    }
    let mut m = ModelMetrics::default();
    m.accuracy = Some(correct as f64 / n as f64);
    if nClasses == 2 {
        let pd = (tp + fp) as f64;
        let p = if pd > 0.0 { tp as f64 / pd } else { 0.0 };
        let rd = (tp + fnCount) as f64;
        let r = if rd > 0.0 { tp as f64 / rd } else { 0.0 };
        m.precision = Some(p);
        m.recall = Some(r);
        m.f1Score = Some(if p + r > 0.0 { 2.0 * p * r / (p + r) } else { 0.0 });
    }
    m
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn classifiesSeparableData() {
        let mut rng = Xoshiro256pp::fromSeed(7);
        let n = 600;
        let mut xs = Vec::with_capacity(n * 2);
        let mut ys = Vec::with_capacity(n);
        for _ in 0..n {
            let cls = rng.nextRange(2) as f64;
            let mu = if cls == 1.0 { 3.0 } else { -3.0 };
            xs.push(mu + 0.5 * rng.nextNormal());
            xs.push(mu + 0.5 * rng.nextNormal());
            ys.push(cls);
        }
        let data = TrainingData::new(&xs, &ys, n, 2);
        let mut config = ModelConfig::new(
            ModelType::RandomForestClassification,
            vec!["x".into(), "y".into()],
        );
        config.hyperparameters.setF64("n_trees", 16.0);
        config.hyperparameters.setF64("max_depth", 6.0);
        let model = train(&config, &data).unwrap();
        let acc = model.metrics.get("accuracy").copied().unwrap();
        assert!(acc > 0.95, "rf accuracy = {}", acc);
    }
}
