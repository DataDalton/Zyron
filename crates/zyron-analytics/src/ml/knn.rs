#![allow(non_snake_case)]
// k-Nearest Neighbors, brute force with reservoir-sampled training cap
// Classification by majority vote, regression by mean

use crate::ml::f64Kernels::sqDistance;
use crate::ml::{ModelConfig, ModelData, ModelMetrics, ModelType, TrainedModel, TrainingData};
use zyron_common::error::{Result, ZyronError};

pub fn train(config: &ModelConfig, data: &TrainingData) -> Result<TrainedModel> {
    if data.n == 0 || data.p == 0 {
        return Err(ZyronError::InvalidParameter {
            name: "training_data".to_string(),
            value: "empty".to_string(),
        });
    }
    let k = config.hyperparameters.getUsizeOr("k", 5).max(1);
    let maxRows = config
        .hyperparameters
        .getUsizeOr("max_training_rows", 1_000_000);
    let regression = matches!(config.modelType, ModelType::KnnRegression);

    let nUse = data.n.min(maxRows);
    // Pick the sampled training indices once and reuse for both xs and ys
    let indices: Vec<usize> = if nUse == data.n {
        (0..data.n).collect()
    } else {
        let seed = config.hyperparameters.getU64Or("seed", 42);
        let mut rng = zyron_common::Xoshiro256pp::fromSeed(seed);
        let mut idx: Vec<usize> = (0..nUse).collect();
        for i in nUse..data.n {
            let j = rng.nextRange((i + 1) as u64) as usize;
            if j < nUse {
                idx[j] = i;
            }
        }
        idx
    };
    let xs: Vec<f64> = if nUse == data.n {
        data.xs.to_vec()
    } else {
        let mut buf = Vec::with_capacity(nUse * data.p);
        for &i in &indices {
            buf.extend_from_slice(data.row(i));
        }
        buf
    };
    let ys: Vec<f64> = if nUse == data.n {
        data.ys.to_vec()
    } else {
        indices.iter().map(|&i| data.ys[i]).collect()
    };

    let mut model = TrainedModel::new(String::new(), config.modelType);
    model.featureColumns = config.featureColumns.clone();
    model.targetColumn = config.targetColumn.clone();
    model.featureMean = vec![0.0; data.p];
    model.featureStd = vec![1.0; data.p];
    model.weights = Vec::new();
    model.data = ModelData::Knn {
        x: xs.clone(),
        y: ys.clone(),
        k,
        nFeatures: data.p,
    };
    model.hyperparameters = config.hyperparameters.clone();
    model.trainingRows = nUse as u64;

    // Self-evaluation, leave-one-out approximated by using 2*k nearest then drop self
    let mut metrics = ModelMetrics::default();
    if !regression {
        let mut correct = 0u64;
        for i in 0..nUse {
            let pred = predictClass(&xs, &ys, k, data.p, &xs[i * data.p..i * data.p + data.p]);
            if (pred as i64) == (ys[i] as i64) {
                correct += 1;
            }
        }
        metrics.accuracy = Some(correct as f64 / nUse as f64);
    } else {
        let mut ssRes = 0.0f64;
        let mut yMean = 0.0f64;
        for &y in &ys {
            yMean += y;
        }
        yMean /= nUse as f64;
        let mut ssTot = 0.0f64;
        for i in 0..nUse {
            let pred = predictReg(&xs, &ys, k, data.p, &xs[i * data.p..i * data.p + data.p]);
            let err = pred - ys[i];
            ssRes += err * err;
            let d = ys[i] - yMean;
            ssTot += d * d;
        }
        metrics.rmse = Some((ssRes / nUse as f64).sqrt());
        metrics.rSquared = Some(if ssTot > 0.0 {
            1.0 - ssRes / ssTot
        } else {
            0.0
        });
    }
    model.metrics = metrics.intoMap();
    Ok(model)
}

/// Find the indices of the k nearest training rows in xs (n*p row-major)
/// Uses a partial heap of size k, O(n log k) per query
fn topKNearest(xs: &[f64], n: usize, p: usize, k: usize, query: &[f64]) -> Vec<(f64, usize)> {
    let mut heap: Vec<(f64, usize)> = Vec::with_capacity(k);
    for i in 0..n {
        let d = sqDistance(query, &xs[i * p..i * p + p]);
        if heap.len() < k {
            heap.push((d, i));
            if heap.len() == k {
                heapify(&mut heap);
            }
        } else if d < heap[0].0 {
            heap[0] = (d, i);
            siftDown(&mut heap, 0);
        }
    }
    heap.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
    heap
}

fn heapify(heap: &mut [(f64, usize)]) {
    let n = heap.len();
    if n < 2 {
        return;
    }
    for i in (0..n / 2).rev() {
        siftDown(heap, i);
    }
}

fn siftDown(heap: &mut [(f64, usize)], mut i: usize) {
    let n = heap.len();
    loop {
        let l = 2 * i + 1;
        let r = 2 * i + 2;
        let mut largest = i;
        if l < n && heap[l].0 > heap[largest].0 {
            largest = l;
        }
        if r < n && heap[r].0 > heap[largest].0 {
            largest = r;
        }
        if largest == i {
            break;
        }
        heap.swap(i, largest);
        i = largest;
    }
}

pub fn predictClass(xs: &[f64], ys: &[f64], k: usize, p: usize, query: &[f64]) -> u32 {
    let neighbors = topKNearest(xs, ys.len(), p, k, query);
    let mut counts: std::collections::HashMap<u32, u32> = std::collections::HashMap::new();
    for (_, idx) in neighbors {
        let cls = ys[idx] as u32;
        *counts.entry(cls).or_insert(0) += 1;
    }
    let mut best = 0u32;
    let mut bestCnt = 0u32;
    for (cls, cnt) in counts {
        if cnt > bestCnt {
            bestCnt = cnt;
            best = cls;
        }
    }
    best
}

pub fn predictReg(xs: &[f64], ys: &[f64], k: usize, p: usize, query: &[f64]) -> f64 {
    let neighbors = topKNearest(xs, ys.len(), p, k, query);
    if neighbors.is_empty() {
        return 0.0;
    }
    let mut s = 0.0f64;
    for (_, idx) in &neighbors {
        s += ys[*idx];
    }
    s / neighbors.len() as f64
}

#[cfg(test)]
mod tests {
    use super::*;
    use zyron_common::Xoshiro256pp;

    #[test]
    fn classifiesByNeighbor() {
        let mut rng = Xoshiro256pp::fromSeed(3);
        let n = 400;
        let mut xs = Vec::with_capacity(n * 2);
        let mut ys = Vec::with_capacity(n);
        for _ in 0..n {
            let cls = rng.nextRange(2) as f64;
            let mu = if cls == 1.0 { 2.0 } else { -2.0 };
            xs.push(mu + 0.4 * rng.nextNormal());
            xs.push(mu + 0.4 * rng.nextNormal());
            ys.push(cls);
        }
        let data = TrainingData::new(&xs, &ys, n, 2);
        let mut config =
            ModelConfig::new(ModelType::KnnClassification, vec!["x".into(), "y".into()]);
        config.hyperparameters.setF64("k", 5.0);
        let model = train(&config, &data).unwrap();
        let acc = model.metrics.get("accuracy").copied().unwrap();
        assert!(acc > 0.9, "knn acc = {}", acc);
    }
}
