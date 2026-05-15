#![allow(non_snake_case)]
// Auto-promoted ANN-backed KNN
//
// When a KNN model is trained on more than ANN_PROMOTION_THRESHOLD rows
// the inference path builds an HNSW index from the stored training
// matrix on first use and serves all subsequent queries through ANN
// search. The index is cached server-wide keyed by model name so a
// single build amortizes across the lifetime of the model in cache.
//
// On-disk model format is unchanged. ModelData::Knn keeps storing the
// raw Vec<f64> training matrix; HNSW is reconstructed lazily on first
// query of a model, which costs sub-second build time even at 1M rows
// and removes the need to version a binary HNSW serialization format

use std::collections::HashMap;
use std::sync::{Arc, OnceLock};

use parking_lot::RwLock;
use zyron_search::vector::ann_index::AnnIndex;
use zyron_search::vector::{DistanceMetric, HnswConfig, VectorSearch};

/// Training-set size at which KNN auto-promotes to HNSW. Below this
/// brute force runs in microseconds and the index build overhead is
/// not amortized; above it, HNSW is asymptotically log(n)
pub const ANN_PROMOTION_THRESHOLD: usize = 10_000;

/// Per-model HNSW handle. Holds the index plus the label vector that
/// HNSW does not see (it only knows row indices as VectorId values)
pub struct KnnAnnIndex {
    pub index: AnnIndex,
    pub labels: Vec<f64>,
    pub nFeatures: usize,
    pub k: usize,
}

impl KnnAnnIndex {
    /// Build an HNSW from a row-major (n, p) f64 training matrix.
    /// The HNSW operates on f32 since that is what the existing kernel
    /// dispatches to. f64 -> f32 precision drop on features is acceptable
    /// for ANN search (which is itself an approximation)
    pub fn build(
        xs: &[f64],
        ys: &[f64],
        n: usize,
        p: usize,
        k: usize,
        modelHash: u32,
    ) -> Result<Self, String> {
        if n == 0 || p == 0 {
            return Err("empty training data".to_string());
        }
        debug_assert_eq!(xs.len(), n * p);
        debug_assert_eq!(ys.len(), n);

        // One contiguous f32 buffer holding all rows in row-major order.
        // Cheaper than n separate Vec<f32> allocations (~50% less heap
        // pressure on the build path) and gives the HNSW builder slices
        // pointing into a single arena
        let mut f32Buf: Vec<f32> = Vec::with_capacity(n * p);
        for &v in xs {
            f32Buf.push(v as f32);
        }
        let vectors: Vec<(u64, &[f32])> = (0..n)
            .map(|i| (i as u64, &f32Buf[i * p..i * p + p]))
            .collect();

        // Use Euclidean distance to match brute-force KNN semantics.
        // ANN_PROMOTION_THRESHOLD * 8 floats fits in typical L3 cache so
        // the build path is memory-bandwidth bound, not CPU
        let mut config = HnswConfig::default();
        config.metric = DistanceMetric::Euclidean;

        let index = AnnIndex::build(&vectors, modelHash, 0, 0, config)
            .map_err(|e| format!("ann build: {}", e))?;

        Ok(Self {
            index,
            labels: ys.to_vec(),
            nFeatures: p,
            k,
        })
    }

    pub fn searchRegression(&self, query: &[f64]) -> f64 {
        QUERY_BUF.with(|cell| {
            let mut buf = cell.borrow_mut();
            buf.clear();
            buf.reserve(query.len());
            for &v in query {
                buf.push(v as f32);
            }
            match self.index.search(&buf, self.k, 0) {
                Ok(results) if !results.is_empty() => {
                    let mut s = 0.0f64;
                    for (idx, _dist) in &results {
                        s += self.labels[*idx as usize];
                    }
                    s / results.len() as f64
                }
                _ => 0.0,
            }
        })
    }

    pub fn searchClassification(&self, query: &[f64]) -> u32 {
        QUERY_BUF.with(|cell| {
            let mut buf = cell.borrow_mut();
            buf.clear();
            buf.reserve(query.len());
            for &v in query {
                buf.push(v as f32);
            }
            match self.index.search(&buf, self.k, 0) {
                Ok(results) if !results.is_empty() => {
                    // Stack-allocated tally for the common nClasses <= 16
                    // case. Falls back to HashMap only when results are
                    // labelled with very high class indices
                    let mut votes = [0u32; 16];
                    let mut over = HashMap::<u32, u32>::new();
                    for (idx, _dist) in &results {
                        let cls = self.labels[*idx as usize] as u32;
                        if (cls as usize) < 16 {
                            votes[cls as usize] += 1;
                        } else {
                            *over.entry(cls).or_insert(0) += 1;
                        }
                    }
                    let mut best = 0u32;
                    let mut bestCnt = 0u32;
                    for i in 0..16 {
                        if votes[i] > bestCnt {
                            bestCnt = votes[i];
                            best = i as u32;
                        }
                    }
                    for (cls, cnt) in over {
                        if cnt > bestCnt {
                            bestCnt = cnt;
                            best = cls;
                        }
                    }
                    best
                }
                _ => 0,
            }
        })
    }
}

thread_local! {
    /// Per-thread reusable buffer for the f32 query downcast. Avoids one
    /// Vec<f32> allocation per ANN search call
    static QUERY_BUF: std::cell::RefCell<Vec<f32>> = std::cell::RefCell::new(Vec::new());
}

/// Server-wide HNSW cache keyed by model name. First inference call on
/// a promoted KNN model builds the index and stashes it here; subsequent
/// calls reuse it. On DROP MODEL or cache invalidation the entry can be
/// cleared via remove(name)
pub struct KnnAnnCache {
    entries: RwLock<HashMap<String, Arc<KnnAnnIndex>>>,
}

impl KnnAnnCache {
    pub fn new() -> Self {
        Self {
            entries: RwLock::new(HashMap::new()),
        }
    }

    pub fn get(&self, name: &str) -> Option<Arc<KnnAnnIndex>> {
        self.entries.read().get(name).cloned()
    }

    pub fn install(&self, name: String, index: Arc<KnnAnnIndex>) {
        self.entries.write().insert(name, index);
    }

    pub fn remove(&self, name: &str) {
        self.entries.write().remove(name);
    }

    pub fn len(&self) -> usize {
        self.entries.read().len()
    }
}

impl Default for KnnAnnCache {
    fn default() -> Self {
        Self::new()
    }
}

static KNN_ANN_CACHE: OnceLock<Arc<KnnAnnCache>> = OnceLock::new();

/// Process-wide ANN-KNN cache
pub fn knnAnnCache() -> Arc<KnnAnnCache> {
    KNN_ANN_CACHE
        .get_or_init(|| Arc::new(KnnAnnCache::new()))
        .clone()
}

/// Deterministic hash of a model name, used as the HNSW indexId so two
/// distinct models never share the same internal index identifier
pub fn modelNameHash(name: &str) -> u32 {
    let mut h: u64 = 0x9E3779B97F4A7C15;
    for &b in name.as_bytes() {
        h = zyron_common::fx_mix(h, b as u64);
    }
    (h ^ (h >> 32)) as u32
}

/// Resolves an ANN index for the given KNN model, building it lazily on
/// first call. Returns None when the training set is below the
/// promotion threshold and brute force should be used instead
pub fn resolveOrBuildAnn(
    modelName: &str,
    xs: &[f64],
    ys: &[f64],
    n: usize,
    p: usize,
    k: usize,
) -> Option<Arc<KnnAnnIndex>> {
    if n < ANN_PROMOTION_THRESHOLD {
        return None;
    }
    let cache = knnAnnCache();
    if let Some(existing) = cache.get(modelName) {
        if existing.nFeatures == p && existing.k == k && existing.labels.len() == n {
            return Some(existing);
        }
    }
    // Build outside the cache write lock to avoid blocking other models
    let hash = modelNameHash(modelName);
    let built = KnnAnnIndex::build(xs, ys, n, p, k, hash).ok()?;
    let arc = Arc::new(built);
    cache.install(modelName.to_string(), Arc::clone(&arc));
    Some(arc)
}
