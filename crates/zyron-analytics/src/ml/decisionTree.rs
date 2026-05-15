#![allow(non_snake_case)]
// Decision tree, histogram-based CART
// Classification: Gini impurity, multi-class with up to 64 labels
// Regression: variance reduction (squared error)
// Histogram bins per numeric feature, single quantile-bin pass at root,
// per-node bin accumulation for split candidates

use crate::ml::{
    Hyperparameters, ModelConfig, ModelData, ModelMetrics, ModelType, TrainedModel, TrainingData,
    TreeNode,
};
use zyron_common::Xoshiro256pp;
use zyron_common::error::{Result, ZyronError};

const MAX_BINS: usize = 64;

/// Split criterion
#[derive(Debug, Clone, Copy)]
pub enum SplitCriterion {
    Gini,
    Mse,
}

#[derive(Debug, Clone)]
pub struct TreeBuildConfig {
    pub maxDepth: usize,
    pub minSamplesLeaf: usize,
    pub maxFeatures: Option<usize>,
    pub criterion: SplitCriterion,
    pub seed: u64,
    pub featureSubsample: bool,
}

impl TreeBuildConfig {
    pub fn fromHp(hp: &Hyperparameters, regression: bool) -> Self {
        let criterion = if regression {
            SplitCriterion::Mse
        } else {
            SplitCriterion::Gini
        };
        let maxFeatures = hp.values.get("max_features").map(|v| (*v as usize).max(1));
        Self {
            maxDepth: hp.getUsizeOr("max_depth", 8).max(1),
            minSamplesLeaf: hp.getUsizeOr("min_samples_leaf", 5).max(1),
            maxFeatures,
            criterion,
            seed: hp.getU64Or("seed", 42),
            featureSubsample: hp.getBoolOr("feature_subsample", false),
        }
    }
}

/// Bin layout per feature, holds sorted edges and the assigned bin index per row
struct FeatureBinning {
    edges: Vec<f64>,
    rowBin: Vec<u8>,
}

fn buildBinsForFeature(values: &[f64], rowIndices: &[usize]) -> FeatureBinning {
    let n = rowIndices.len();
    let mut sorted: Vec<f64> = rowIndices.iter().map(|&i| values[i]).collect();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let mut edges = Vec::with_capacity(MAX_BINS - 1);
    if n > MAX_BINS {
        for k in 1..MAX_BINS {
            let idx = (k * n) / MAX_BINS;
            edges.push(sorted[idx.min(n - 1)]);
        }
        edges.dedup_by(|a, b| (*a - *b).abs() < 1e-15);
    } else {
        let mut uniq: Vec<f64> = Vec::new();
        for &v in &sorted {
            if uniq.last().map(|u| *u != v).unwrap_or(true) {
                uniq.push(v);
            }
        }
        for w in uniq.windows(2) {
            edges.push(0.5 * (w[0] + w[1]));
        }
    }
    let mut rowBin = vec![0u8; values.len()];
    for &i in rowIndices {
        let v = values[i];
        let bin = match edges
            .binary_search_by(|e| e.partial_cmp(&v).unwrap_or(std::cmp::Ordering::Equal))
        {
            Ok(idx) => idx + 1,
            Err(idx) => idx,
        };
        rowBin[i] = bin.min(MAX_BINS - 1) as u8;
    }
    FeatureBinning { edges, rowBin }
}

/// Train a tree, return a flat node vector and (for classification) the
/// number of classes. Regression trees set leaves to mean target value;
/// classification trees set leaves to the majority class index encoded as
/// a f64
pub fn trainTree(
    data: &TrainingData,
    config: &TreeBuildConfig,
    sampleIndices: Option<&[usize]>,
    nClassesIfClassification: Option<usize>,
) -> Result<Vec<TreeNode>> {
    let allIndices: Vec<usize> = match sampleIndices {
        Some(s) => s.to_vec(),
        None => (0..data.n).collect(),
    };
    if allIndices.is_empty() {
        return Err(ZyronError::InvalidParameter {
            name: "training_data".to_string(),
            value: "empty after sampling".to_string(),
        });
    }
    // Lazy bin construction (F6). Allocate only the slots; each feature's
    // bin array is populated the first time chooseFeatures picks it. With
    // feature subsampling (random forest path) most features are never
    // touched, saving O(n) memory and time per skipped feature
    let mut binnings: Vec<Option<FeatureBinning>> = Vec::with_capacity(data.p);
    for _ in 0..data.p {
        binnings.push(None);
    }

    let mut nodes: Vec<TreeNode> = Vec::new();
    let mut rng = Xoshiro256pp::fromSeed(config.seed);
    let mut workspace = SplitWorkspace::new(nClassesIfClassification.unwrap_or(2));

    let rootIdx = nodes.len() as i32;
    nodes.push(TreeNode::leaf(0.0));
    splitNode(
        &mut nodes,
        rootIdx,
        &allIndices,
        data,
        &mut binnings,
        config,
        &mut rng,
        &mut workspace,
        0,
        nClassesIfClassification,
    );
    Ok(nodes)
}

fn ensureBinning(
    binnings: &mut [Option<FeatureBinning>],
    data: &TrainingData,
    featureIdx: usize,
    allIndices: &[usize],
) {
    if binnings[featureIdx].is_none() {
        let column: Vec<f64> = (0..data.n)
            .map(|i| data.xs[i * data.p + featureIdx])
            .collect();
        binnings[featureIdx] = Some(buildBinsForFeature(&column, allIndices));
    }
}

/// Reusable scratch buffers for split finding (F7). Keeping these around
/// the recursion avoids ~3-4 small allocations per node, which adds up
/// for trees with hundreds of internal nodes
struct SplitWorkspace {
    binCountU64: Vec<u64>,
    binSum: Vec<f64>,
    binSumSq: Vec<f64>,
    classBinCounts: Vec<u64>,
    leftClass: Vec<u64>,
    totalClass: Vec<u64>,
    nClasses: usize,
}

impl SplitWorkspace {
    fn new(nClasses: usize) -> Self {
        let k = nClasses.max(2);
        Self {
            binCountU64: Vec::new(),
            binSum: Vec::new(),
            binSumSq: Vec::new(),
            classBinCounts: Vec::new(),
            leftClass: vec![0u64; k],
            totalClass: vec![0u64; k],
            nClasses: k,
        }
    }

    fn resizeForBins(&mut self, nBins: usize) {
        self.binCountU64.clear();
        self.binCountU64.resize(nBins, 0);
        self.binSum.clear();
        self.binSum.resize(nBins, 0.0);
        self.binSumSq.clear();
        self.binSumSq.resize(nBins, 0.0);
    }

    fn resizeForClassBins(&mut self, nBins: usize, k: usize) {
        self.nClasses = k.max(2);
        self.classBinCounts.clear();
        self.classBinCounts.resize(nBins * self.nClasses, 0);
        self.leftClass.clear();
        self.leftClass.resize(self.nClasses, 0);
        self.totalClass.clear();
        self.totalClass.resize(self.nClasses, 0);
        self.binCountU64.clear();
        self.binCountU64.resize(nBins, 0);
    }
}

#[allow(clippy::too_many_arguments)]
fn splitNode(
    nodes: &mut Vec<TreeNode>,
    nodeIdx: i32,
    indices: &[usize],
    data: &TrainingData,
    binnings: &mut [Option<FeatureBinning>],
    config: &TreeBuildConfig,
    rng: &mut Xoshiro256pp,
    workspace: &mut SplitWorkspace,
    depth: usize,
    nClassesIfClassification: Option<usize>,
) {
    if indices.len() < 2 * config.minSamplesLeaf || depth >= config.maxDepth {
        nodes[nodeIdx as usize] =
            TreeNode::leaf(leafValue(indices, data, nClassesIfClassification));
        return;
    }
    let candidateFeatures = chooseFeatures(data.p, config, rng);
    let mut bestGain = 0.0f64;
    let mut bestFeature: i32 = -1;
    let mut bestThreshold = 0.0f64;
    let mut bestLeft: Vec<usize> = Vec::new();
    let mut bestRight: Vec<usize> = Vec::new();

    let parentImpurity = nodeImpurity(indices, data, config.criterion, nClassesIfClassification);
    for &fIdx in &candidateFeatures {
        ensureBinning(binnings, data, fIdx, indices);
        let bin = match binnings[fIdx].as_ref() {
            Some(b) => b,
            None => continue,
        };
        if bin.edges.is_empty() {
            continue;
        }
        let result = bestSplitOnFeature(
            indices,
            data,
            fIdx,
            bin,
            config.criterion,
            config.minSamplesLeaf,
            nClassesIfClassification,
            parentImpurity,
            workspace,
        );
        if let Some((gain, threshold, left, right)) = result {
            if gain > bestGain {
                bestGain = gain;
                bestFeature = fIdx as i32;
                bestThreshold = threshold;
                bestLeft = left;
                bestRight = right;
            }
        }
    }
    if bestFeature < 0 || bestGain <= 0.0 {
        nodes[nodeIdx as usize] =
            TreeNode::leaf(leafValue(indices, data, nClassesIfClassification));
        return;
    }
    let leftChildIdx = nodes.len() as i32;
    nodes.push(TreeNode::leaf(0.0));
    let rightChildIdx = nodes.len() as i32;
    nodes.push(TreeNode::leaf(0.0));
    nodes[nodeIdx as usize] = TreeNode {
        featureIdx: bestFeature,
        threshold: bestThreshold,
        left: leftChildIdx,
        right: rightChildIdx,
        leafValue: 0.0,
    };
    splitNode(
        nodes,
        leftChildIdx,
        &bestLeft,
        data,
        binnings,
        config,
        rng,
        workspace,
        depth + 1,
        nClassesIfClassification,
    );
    splitNode(
        nodes,
        rightChildIdx,
        &bestRight,
        data,
        binnings,
        config,
        rng,
        workspace,
        depth + 1,
        nClassesIfClassification,
    );
}

fn chooseFeatures(p: usize, config: &TreeBuildConfig, rng: &mut Xoshiro256pp) -> Vec<usize> {
    let take = match config.maxFeatures {
        Some(k) => k.min(p),
        None => {
            if config.featureSubsample {
                ((p as f64).sqrt().ceil() as usize).max(1).min(p)
            } else {
                p
            }
        }
    };
    if take >= p {
        return (0..p).collect();
    }
    let mut all: Vec<usize> = (0..p).collect();
    rng.shuffle(&mut all);
    all.truncate(take);
    all
}

#[allow(clippy::too_many_arguments)]
fn bestSplitOnFeature(
    indices: &[usize],
    data: &TrainingData,
    _featureIdx: usize,
    bin: &FeatureBinning,
    criterion: SplitCriterion,
    minLeaf: usize,
    nClasses: Option<usize>,
    parentImpurity: f64,
    workspace: &mut SplitWorkspace,
) -> Option<(f64, f64, Vec<usize>, Vec<usize>)> {
    let n = indices.len();
    let nBins = bin.edges.len() + 1;
    if nBins < 2 {
        return None;
    }

    let mut bestGain = 0.0f64;
    let mut bestEdge = 0usize;
    match criterion {
        SplitCriterion::Mse => {
            workspace.resizeForBins(nBins);
            let binCount = &mut workspace.binCountU64;
            let binSum = &mut workspace.binSum;
            let binSumSq = &mut workspace.binSumSq;
            for &i in indices {
                let b = bin.rowBin[i] as usize;
                let y = data.ys[i];
                binCount[b] += 1;
                binSum[b] += y;
                binSumSq[b] += y * y;
            }
            let mut leftCount = 0u64;
            let mut leftSum = 0.0f64;
            let mut leftSumSq = 0.0f64;
            let totalCount: u64 = binCount.iter().sum();
            let totalSum: f64 = binSum.iter().sum();
            let totalSumSq: f64 = binSumSq.iter().sum();
            for b in 0..(nBins - 1) {
                leftCount += binCount[b];
                leftSum += binSum[b];
                leftSumSq += binSumSq[b];
                let rightCount = totalCount - leftCount;
                if (leftCount as usize) < minLeaf || (rightCount as usize) < minLeaf {
                    continue;
                }
                let leftImp = leftSumSq - leftSum * leftSum / leftCount as f64;
                let rightSum = totalSum - leftSum;
                let rightImp = (totalSumSq - leftSumSq) - rightSum * rightSum / rightCount as f64;
                let weightedImp = (leftImp + rightImp) / n as f64;
                let gain = parentImpurity - weightedImp;
                if gain > bestGain {
                    bestGain = gain;
                    bestEdge = b;
                }
            }
        }
        SplitCriterion::Gini => {
            let k = nClasses.unwrap_or(2).max(2);
            workspace.resizeForClassBins(nBins, k);
            let counts = &mut workspace.classBinCounts;
            let binCount = &mut workspace.binCountU64;
            let leftClass = &mut workspace.leftClass;
            let totalClass = &mut workspace.totalClass;
            for &i in indices {
                let b = bin.rowBin[i] as usize;
                let cls = (data.ys[i] as usize).min(k - 1);
                counts[b * k + cls] += 1;
                binCount[b] += 1;
            }
            for b in 0..nBins {
                for c in 0..k {
                    totalClass[c] += counts[b * k + c];
                }
            }
            let total: u64 = totalClass.iter().sum();
            let mut leftCount = 0u64;
            for b in 0..(nBins - 1) {
                for c in 0..k {
                    leftClass[c] += counts[b * k + c];
                }
                leftCount += binCount[b];
                let rightCount = total - leftCount;
                if (leftCount as usize) < minLeaf || (rightCount as usize) < minLeaf {
                    continue;
                }
                let mut leftImp = 1.0f64;
                let mut rightImp = 1.0f64;
                for c in 0..k {
                    let lp = leftClass[c] as f64 / leftCount as f64;
                    leftImp -= lp * lp;
                    let rc = totalClass[c] - leftClass[c];
                    let rp = rc as f64 / rightCount as f64;
                    rightImp -= rp * rp;
                }
                let weightedImp =
                    (leftCount as f64 * leftImp + rightCount as f64 * rightImp) / total as f64;
                let gain = parentImpurity - weightedImp;
                if gain > bestGain {
                    bestGain = gain;
                    bestEdge = b;
                }
            }
        }
    }
    if bestGain <= 0.0 {
        return None;
    }
    let threshold = bin.edges[bestEdge];
    let mut left = Vec::with_capacity(n / 2);
    let mut right = Vec::with_capacity(n / 2);
    for &i in indices {
        let b = bin.rowBin[i] as usize;
        if b <= bestEdge {
            left.push(i);
        } else {
            right.push(i);
        }
    }
    Some((bestGain, threshold, left, right))
}

fn nodeImpurity(
    indices: &[usize],
    data: &TrainingData,
    criterion: SplitCriterion,
    nClasses: Option<usize>,
) -> f64 {
    if indices.is_empty() {
        return 0.0;
    }
    match criterion {
        SplitCriterion::Mse => {
            let mut sum = 0.0f64;
            let mut sumSq = 0.0f64;
            for &i in indices {
                let y = data.ys[i];
                sum += y;
                sumSq += y * y;
            }
            let n = indices.len() as f64;
            sumSq - sum * sum / n
        }
        SplitCriterion::Gini => {
            let k = nClasses.unwrap_or(2).max(2);
            let mut counts = vec![0u64; k];
            for &i in indices {
                let cls = (data.ys[i] as usize).min(k - 1);
                counts[cls] += 1;
            }
            let n = indices.len() as f64;
            let mut imp = 1.0f64;
            for c in counts {
                let p = c as f64 / n;
                imp -= p * p;
            }
            imp * indices.len() as f64
        }
    }
}

fn leafValue(indices: &[usize], data: &TrainingData, nClasses: Option<usize>) -> f64 {
    if indices.is_empty() {
        return 0.0;
    }
    if let Some(k) = nClasses {
        let mut counts = vec![0u64; k];
        for &i in indices {
            let cls = (data.ys[i] as usize).min(k - 1);
            counts[cls] += 1;
        }
        let mut bestC = 0usize;
        let mut bestCnt = 0u64;
        for (c, &cnt) in counts.iter().enumerate() {
            if cnt > bestCnt {
                bestCnt = cnt;
                bestC = c;
            }
        }
        bestC as f64
    } else {
        let mut s = 0.0f64;
        for &i in indices {
            s += data.ys[i];
        }
        s / indices.len() as f64
    }
}

pub fn predictTree(nodes: &[TreeNode], features: &[f64]) -> f64 {
    let mut idx = 0i32;
    while idx >= 0 && (idx as usize) < nodes.len() {
        let n = nodes[idx as usize];
        if n.isLeaf() {
            return n.leafValue;
        }
        let v = features[n.featureIdx as usize];
        idx = if v <= n.threshold { n.left } else { n.right };
    }
    0.0
}

pub fn train(config: &ModelConfig, data: &TrainingData) -> Result<TrainedModel> {
    if data.n == 0 || data.p == 0 {
        return Err(ZyronError::InvalidParameter {
            name: "training_data".to_string(),
            value: "empty".to_string(),
        });
    }
    let regression = matches!(config.modelType, ModelType::DecisionTreeRegression);
    let nClasses = if !regression {
        Some(detectNumClasses(data.ys))
    } else {
        None
    };
    let buildConfig = TreeBuildConfig::fromHp(&config.hyperparameters, regression);
    let nodes = trainTree(data, &buildConfig, None, nClasses)?;

    let mut model = TrainedModel::new(String::new(), config.modelType);
    model.featureColumns = config.featureColumns.clone();
    model.targetColumn = config.targetColumn.clone();
    model.featureMean = vec![0.0; data.p];
    model.featureStd = vec![1.0; data.p];
    model.weights = Vec::new();
    model.data = ModelData::Tree {
        nodes: nodes.clone(),
    };
    model.hyperparameters = config.hyperparameters.clone();
    model.trainingRows = data.n as u64;
    if regression {
        model.metrics = computeRegressionMetrics(&nodes, data).intoMap();
    } else {
        model.metrics = computeClassificationMetrics(&nodes, data, nClasses.unwrap()).intoMap();
    }
    Ok(model)
}

pub fn detectNumClasses(ys: &[f64]) -> usize {
    let mut maxCls = 0usize;
    for &y in ys {
        let c = y as usize;
        if c > maxCls {
            maxCls = c;
        }
    }
    maxCls + 1
}

fn computeRegressionMetrics(nodes: &[TreeNode], data: &TrainingData) -> ModelMetrics {
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
        let pred = predictTree(nodes, data.row(i));
        let err = pred - data.ys[i];
        ssRes += err * err;
        absSum += err.abs();
        let dy = data.ys[i] - yMean;
        ssTot += dy * dy;
    }
    let mut m = ModelMetrics::default();
    m.rmse = Some((ssRes / n as f64).sqrt());
    m.mae = Some(absSum / n as f64);
    m.rSquared = Some(if ssTot > 0.0 {
        1.0 - ssRes / ssTot
    } else {
        0.0
    });
    m
}

fn computeClassificationMetrics(
    nodes: &[TreeNode],
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
        let pred = predictTree(nodes, data.row(i));
        let yhat = pred.round() as usize;
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
        m.f1Score = Some(if p + r > 0.0 {
            2.0 * p * r / (p + r)
        } else {
            0.0
        });
    }
    m
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn classifiesIrisLikeData() {
        // Synthetic 3-class data with separable means
        let mut rng = Xoshiro256pp::fromSeed(1);
        let mut xs: Vec<f64> = Vec::new();
        let mut ys: Vec<f64> = Vec::new();
        for cls in 0..3u64 {
            for _ in 0..150 {
                let mu = cls as f64 * 5.0;
                xs.push(mu + 0.5 * rng.nextNormal());
                xs.push(mu + 0.5 * rng.nextNormal());
                xs.push(mu + 0.5 * rng.nextNormal());
                xs.push(mu + 0.5 * rng.nextNormal());
                ys.push(cls as f64);
            }
        }
        let data = TrainingData::new(&xs, &ys, ys.len(), 4);
        let mut config = ModelConfig::new(
            ModelType::DecisionTreeClassification,
            vec!["a".into(), "b".into(), "c".into(), "d".into()],
        );
        config.hyperparameters.setF64("max_depth", 6.0);
        let model = train(&config, &data).unwrap();
        let acc = model.metrics.get("accuracy").copied().unwrap();
        assert!(acc > 0.9, "tree accuracy = {}", acc);
    }

    #[test]
    fn regressesLinearTrend() {
        let mut rng = Xoshiro256pp::fromSeed(2);
        let n = 500;
        let mut xs = Vec::with_capacity(n);
        let mut ys = Vec::with_capacity(n);
        for i in 0..n {
            let x = i as f64 * 0.1;
            xs.push(x);
            ys.push(2.0 * x + 3.0 + 0.05 * rng.nextNormal());
        }
        let data = TrainingData::new(&xs, &ys, n, 1);
        let mut config = ModelConfig::new(ModelType::DecisionTreeRegression, vec!["x".into()]);
        config.hyperparameters.setF64("max_depth", 8.0);
        let model = train(&config, &data).unwrap();
        let r2 = model.metrics.get("r_squared").copied().unwrap();
        assert!(r2 > 0.95, "tree r2 = {}", r2);
    }
}
