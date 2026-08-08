#![allow(non_snake_case)]
// Inference engine and server-wide model cache
// Predicts on cached deserialized models, dispatches per ModelType

use crate::ml::decisionTree::predictTree;
use crate::ml::f64Kernels::{dot, rowMajorMatvec, sigmoid};
use crate::ml::gradientBoosting::{predictProbability as gbmProb, predictRegression as gbmReg};
use crate::ml::kmeans::predictCluster;
use crate::ml::knn;
use crate::ml::linearRegression::predict as linPredict;
use crate::ml::logisticRegression::predictProbability as logProb;
use crate::ml::randomForest::{predictClassification as rfClass, predictRegression as rfReg};
use crate::ml::{ModelData, ModelType, TrainedModel};
use parking_lot::RwLock;
use std::collections::HashMap;
use std::sync::{Arc, OnceLock};
use zyron_common::error::{Result, ZyronError};

/// Single-row prediction
pub fn predictOne(model: &TrainedModel, features: &[f64]) -> f64 {
    match model.modelType {
        ModelType::LinearRegression => linPredict(model, features),
        ModelType::LogisticRegression => logProb(model, features),
        ModelType::DecisionTreeClassification | ModelType::DecisionTreeRegression => {
            if let ModelData::Tree { nodes } = &model.data {
                predictTree(nodes, features)
            } else {
                f64::NAN
            }
        }
        ModelType::RandomForestRegression => {
            if let ModelData::Forest { trees } = &model.data {
                rfReg(trees, features)
            } else {
                f64::NAN
            }
        }
        ModelType::RandomForestClassification => {
            if let ModelData::Forest { trees } = &model.data {
                let k = model.weights.first().copied().unwrap_or(2.0) as usize;
                rfClass(trees, features, k)
            } else {
                f64::NAN
            }
        }
        ModelType::GradientBoostingRegression => {
            if let ModelData::BoostedTrees {
                baseScore,
                learningRate,
                trees,
            } = &model.data
            {
                gbmReg(*baseScore, *learningRate, trees, features)
            } else {
                f64::NAN
            }
        }
        ModelType::GradientBoostingClassification => {
            if let ModelData::BoostedTrees {
                baseScore,
                learningRate,
                trees,
            } = &model.data
            {
                gbmProb(*baseScore, *learningRate, trees, features)
            } else {
                f64::NAN
            }
        }
        ModelType::KMeans => {
            if let ModelData::KMeans { k, nFeatures } = &model.data {
                predictCluster(&model.weights, *k, *nFeatures, features) as f64
            } else {
                f64::NAN
            }
        }
        ModelType::KnnRegression => {
            if let ModelData::Knn { x, y, k, nFeatures } = &model.data {
                knn::predictReg(x, y, *k, *nFeatures, features)
            } else {
                f64::NAN
            }
        }
        ModelType::KnnClassification => {
            if let ModelData::Knn { x, y, k, nFeatures } = &model.data {
                knn::predictClass(x, y, *k, *nFeatures, features) as f64
            } else {
                f64::NAN
            }
        }
    }
}

/// Batch prediction, fills out[]
/// Linear and logistic models take a vectorized gemv fast path. Tree-based
/// and KNN models still loop per-row but only pay the dispatch cost once
/// per batch via the cached InferenceHandle (F11, F12)
pub fn predictBatch(
    model: &TrainedModel,
    xs: &[f64],
    n: usize,
    p: usize,
    out: &mut [f64],
) -> Result<()> {
    debug_assert_eq!(xs.len(), n * p);
    debug_assert_eq!(out.len(), n);
    match model.modelType {
        ModelType::LinearRegression => {
            // Apply per-feature standardization, then gemv: out = X_std * w + b
            let w = &model.weights[..p];
            let b = model.weights[p];
            // For unstandardised features we could call rowMajorMatvec directly
            let allUnit = model.featureStd.iter().all(|s| *s == 1.0)
                && model.featureMean.iter().all(|m| *m == 0.0);
            if allUnit {
                rowMajorMatvec(xs, w, b, n, p, out);
                return Ok(());
            }
            for i in 0..n {
                let row = &xs[i * p..i * p + p];
                let mut z = 0.0f64;
                for j in 0..p {
                    let std = model.featureStd.get(j).copied().unwrap_or(1.0);
                    let mean = model.featureMean.get(j).copied().unwrap_or(0.0);
                    let denom = if std == 0.0 { 1.0 } else { std };
                    z += w[j] * (row[j] - mean) / denom;
                }
                out[i] = z + b;
            }
        }
        ModelType::LogisticRegression => {
            let w = &model.weights[..p];
            let b = model.weights[p];
            for i in 0..n {
                let row = &xs[i * p..i * p + p];
                let mut z = 0.0f64;
                for j in 0..p {
                    let std = model.featureStd.get(j).copied().unwrap_or(1.0);
                    let mean = model.featureMean.get(j).copied().unwrap_or(0.0);
                    let denom = if std == 0.0 { 1.0 } else { std };
                    z += w[j] * (row[j] - mean) / denom;
                }
                out[i] = sigmoid(z + b);
            }
        }
        _ => {
            // Parallelize across row chunks for tree/forest/GBM/KMeans
            // /KNN models. Each row is independent so trivially shardable.
            // Below the threshold the thread-spawn overhead dominates; at
            // 4096+ rows the parallel speedup amortizes
            const PARALLEL_THRESHOLD: usize = 4096;
            let nThreads = std::thread::available_parallelism()
                .map(|p| p.get())
                .unwrap_or(1);
            if n < PARALLEL_THRESHOLD || nThreads < 2 {
                for i in 0..n {
                    out[i] = predictOne(model, &xs[i * p..i * p + p]);
                }
                return Ok(());
            }
            let chunkRows = (n + nThreads - 1) / nThreads;
            let worker_failed = std::thread::scope(|scope| {
                let mut starts: Vec<usize> = Vec::with_capacity(nThreads);
                let mut s = 0usize;
                while s < n {
                    starts.push(s);
                    s += chunkRows;
                }
                // Split the output into disjoint mutable chunks aligned
                // with the input chunks so threads write to disjoint
                // memory without locks
                let chunks: Vec<&mut [f64]> = out.chunks_mut(chunkRows).collect();
                let mut handles = Vec::with_capacity(chunks.len());
                for (chunkIdx, outChunk) in chunks.into_iter().enumerate() {
                    let start = starts[chunkIdx];
                    let rowCount = outChunk.len();
                    let xsSlice = &xs[start * p..(start + rowCount) * p];
                    let h = scope.spawn(move || {
                        for i in 0..rowCount {
                            outChunk[i] = predictOne(model, &xsSlice[i * p..i * p + p]);
                        }
                    });
                    handles.push(h);
                }
                // A panicked worker leaves its output chunk stale or zero. Detect
                // any join error so the whole batch fails rather than returning
                // partial output as success
                let mut failed = false;
                for h in handles {
                    if h.join().is_err() {
                        failed = true;
                    }
                }
                failed
            });
            if worker_failed {
                return Err(ZyronError::ExecutionError(
                    "predictBatch worker thread panicked".to_string(),
                ));
            }
        }
    }
    let _ = dot;
    Ok(())
}

/// Server-wide cache of trained models keyed by name
pub struct ModelCache {
    models: RwLock<HashMap<String, Arc<TrainedModel>>>,
    version: std::sync::atomic::AtomicU64,
}

impl ModelCache {
    pub fn new() -> Self {
        Self {
            models: RwLock::new(HashMap::new()),
            version: std::sync::atomic::AtomicU64::new(0),
        }
    }

    pub fn install(&self, name: String, model: TrainedModel) {
        // Invalidate any stale ANN index built from a previous model with
        // the same name. Without this a CREATE MODEL that replaces an
        // existing KNN model would serve predictions from the old HNSW
        // index until process restart
        crate::ml::annKnn::knnAnnCache().remove(&name);
        let mut g = self.models.write();
        g.insert(name, Arc::new(model));
        self.version
            .fetch_add(1, std::sync::atomic::Ordering::Release);
    }

    pub fn invalidate(&self, name: &str) {
        let mut g = self.models.write();
        g.remove(name);
        self.version
            .fetch_add(1, std::sync::atomic::Ordering::Release);
        // Drop the auto-built ANN index for this model so a future
        // CREATE MODEL with the same name does not serve stale results
        crate::ml::annKnn::knnAnnCache().remove(name);
    }

    pub fn get(&self, name: &str) -> Option<Arc<TrainedModel>> {
        self.models.read().get(name).cloned()
    }

    pub fn names(&self) -> Vec<String> {
        self.models.read().keys().cloned().collect()
    }

    pub fn len(&self) -> usize {
        self.models.read().len()
    }

    pub fn isEmpty(&self) -> bool {
        self.models.read().is_empty()
    }

    pub fn version(&self) -> u64 {
        self.version.load(std::sync::atomic::Ordering::Acquire)
    }
}

impl Default for ModelCache {
    fn default() -> Self {
        Self::new()
    }
}

static MODEL_CACHE: OnceLock<Arc<ModelCache>> = OnceLock::new();

/// Process-wide singleton cache, lazily initialized
pub fn modelCache() -> Arc<ModelCache> {
    MODEL_CACHE
        .get_or_init(|| Arc::new(ModelCache::new()))
        .clone()
}

/// Cached batch evaluator that resolves the model handle once. The
/// `predictFn` pointer is set at resolve time so per-call match dispatch
/// is paid once per batch, not per row. For KNN models that have
/// promoted to ANN, the resolved index handle is cached here too so the
/// ANN cache HashMap lookup happens once per resolve, not once per query
pub struct InferenceHandle {
    pub model: Arc<TrainedModel>,
    pub cacheVersion: u64,
    predictFn: fn(&TrainedModel, &[f64]) -> f64,
    annIndex: Option<Arc<crate::ml::annKnn::KnnAnnIndex>>,
}

impl InferenceHandle {
    pub fn resolve(name: &str) -> Option<Self> {
        let cache = modelCache();
        let model = cache.get(name)?;
        let predictFn = predictFnForType(model.modelType);
        // For KNN models that exceed the ANN promotion threshold, resolve
        // (and build on miss) the HNSW index here so per-call predictOne
        // does not pay the ANN cache lookup per row
        let annIndex = match model.modelType {
            ModelType::KnnRegression | ModelType::KnnClassification => {
                if let ModelData::Knn { x, y, k, nFeatures } = &model.data {
                    crate::ml::annKnn::resolveOrBuildAnn(name, x, y, y.len(), *nFeatures, *k)
                } else {
                    None
                }
            }
            _ => None,
        };
        Some(Self {
            model,
            cacheVersion: cache.version(),
            predictFn,
            annIndex,
        })
    }

    #[inline]
    pub fn predictOne(&self, features: &[f64]) -> f64 {
        // Fast path for KNN with resolved ANN: skip the dispatch and call
        // the index directly. Saves both the fn-pointer indirect call and
        // the ANN cache HashMap probe per row
        if let Some(ann) = &self.annIndex {
            return match self.model.modelType {
                ModelType::KnnRegression => ann.searchRegression(features),
                ModelType::KnnClassification => ann.searchClassification(features) as f64,
                _ => (self.predictFn)(&self.model, features),
            };
        }
        (self.predictFn)(&self.model, features)
    }

    pub fn predictBatch(&self, xs: &[f64], n: usize, p: usize, out: &mut [f64]) -> Result<()> {
        predictBatch(&self.model, xs, n, p, out)
    }
}

fn predictFnForType(t: ModelType) -> fn(&TrainedModel, &[f64]) -> f64 {
    match t {
        ModelType::LinearRegression => predictLinear,
        ModelType::LogisticRegression => predictLogistic,
        ModelType::DecisionTreeClassification | ModelType::DecisionTreeRegression => {
            predictDecisionTree
        }
        ModelType::RandomForestRegression => predictRandomForestRegression,
        ModelType::RandomForestClassification => predictRandomForestClassification,
        ModelType::GradientBoostingRegression => predictGbmRegression,
        ModelType::GradientBoostingClassification => predictGbmClassification,
        ModelType::KMeans => predictKMeans,
        ModelType::KnnRegression => predictKnnRegression,
        ModelType::KnnClassification => predictKnnClassification,
    }
}

fn predictLinear(m: &TrainedModel, f: &[f64]) -> f64 {
    linPredict(m, f)
}
fn predictLogistic(m: &TrainedModel, f: &[f64]) -> f64 {
    logProb(m, f)
}
fn predictDecisionTree(m: &TrainedModel, f: &[f64]) -> f64 {
    if let ModelData::Tree { nodes } = &m.data {
        predictTree(nodes, f)
    } else {
        f64::NAN
    }
}
fn predictRandomForestRegression(m: &TrainedModel, f: &[f64]) -> f64 {
    if let ModelData::Forest { trees } = &m.data {
        rfReg(trees, f)
    } else {
        f64::NAN
    }
}
fn predictRandomForestClassification(m: &TrainedModel, f: &[f64]) -> f64 {
    if let ModelData::Forest { trees } = &m.data {
        let k = m.weights.first().copied().unwrap_or(2.0) as usize;
        rfClass(trees, f, k)
    } else {
        f64::NAN
    }
}
fn predictGbmRegression(m: &TrainedModel, f: &[f64]) -> f64 {
    if let ModelData::BoostedTrees {
        baseScore,
        learningRate,
        trees,
    } = &m.data
    {
        gbmReg(*baseScore, *learningRate, trees, f)
    } else {
        f64::NAN
    }
}
fn predictGbmClassification(m: &TrainedModel, f: &[f64]) -> f64 {
    if let ModelData::BoostedTrees {
        baseScore,
        learningRate,
        trees,
    } = &m.data
    {
        gbmProb(*baseScore, *learningRate, trees, f)
    } else {
        f64::NAN
    }
}
fn predictKMeans(m: &TrainedModel, f: &[f64]) -> f64 {
    if let ModelData::KMeans { k, nFeatures } = &m.data {
        predictCluster(&m.weights, *k, *nFeatures, f) as f64
    } else {
        f64::NAN
    }
}
fn predictKnnRegression(m: &TrainedModel, f: &[f64]) -> f64 {
    if let ModelData::Knn { x, y, k, nFeatures } = &m.data {
        // Promote to HNSW when training set is above threshold. The
        // cache returns None below threshold, falling back to brute force
        let n = y.len();
        if let Some(ann) = crate::ml::annKnn::resolveOrBuildAnn(&m.modelId, x, y, n, *nFeatures, *k)
        {
            return ann.searchRegression(f);
        }
        knn::predictReg(x, y, *k, *nFeatures, f)
    } else {
        f64::NAN
    }
}
fn predictKnnClassification(m: &TrainedModel, f: &[f64]) -> f64 {
    if let ModelData::Knn { x, y, k, nFeatures } = &m.data {
        let n = y.len();
        if let Some(ann) = crate::ml::annKnn::resolveOrBuildAnn(&m.modelId, x, y, n, *nFeatures, *k)
        {
            return ann.searchClassification(f) as f64;
        }
        knn::predictClass(x, y, *k, *nFeatures, f) as f64
    } else {
        f64::NAN
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ml::{Hyperparameters, ModelConfig, TrainingData};

    #[test]
    fn cacheRoundTrip() {
        let cache = modelCache();
        let mut m = TrainedModel::new("test_m".to_string(), ModelType::LinearRegression);
        m.featureColumns = vec!["a".into()];
        m.featureMean = vec![0.0];
        m.featureStd = vec![1.0];
        m.weights = vec![2.0, 1.0];
        cache.install("test_m".to_string(), m);
        let h = InferenceHandle::resolve("test_m").unwrap();
        let p = h.predictOne(&[3.0]);
        assert!((p - 7.0).abs() < 1e-12, "p = {}", p);
        cache.invalidate("test_m");
        assert!(InferenceHandle::resolve("test_m").is_none());
    }

    #[test]
    fn dispatchLinearRegression() {
        let xs: Vec<f64> = (0..100).map(|i| i as f64 * 0.1).collect();
        let ys: Vec<f64> = xs.iter().map(|x| 2.0 * x + 3.0).collect();
        let mut config = ModelConfig::new(ModelType::LinearRegression, vec!["x".into()]);
        config.targetColumn = Some("y".into());
        config.hyperparameters = Hyperparameters::new();
        let data = TrainingData::new(&xs, &ys, xs.len(), 1);
        let model = crate::ml::linearRegression::train(&config, &data).unwrap();
        let p = predictOne(&model, &[5.0]);
        assert!((p - 13.0).abs() < 0.5, "predict = {}", p);
    }
}
