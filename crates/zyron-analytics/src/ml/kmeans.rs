#![allow(non_snake_case)]
// K-means clustering, Lloyd's algorithm with kmeans++ seeding
// Centroids stored row-major in TrainedModel.weights
// ModelData::KMeans holds k and feature dimension for indexing

use crate::ml::f64Kernels::{normSquared, resolveDot, sqDistance};
use crate::ml::{ModelConfig, ModelData, ModelMetrics, ModelType, TrainedModel, TrainingData};
use zyron_common::Xoshiro256pp;
use zyron_common::error::{Result, ZyronError};

pub fn train(config: &ModelConfig, data: &TrainingData) -> Result<TrainedModel> {
    if data.n == 0 || data.p == 0 {
        return Err(ZyronError::InvalidParameter {
            name: "training_data".to_string(),
            value: "empty".to_string(),
        });
    }
    let k = config.hyperparameters.getUsizeOr("k", 8).max(1);
    if k > data.n {
        return Err(ZyronError::InvalidParameter {
            name: "k".to_string(),
            value: format!("k={} exceeds n={}", k, data.n),
        });
    }
    let maxIter = config.hyperparameters.getUsizeOr("max_iter", 100).max(1);
    let tol = config.hyperparameters.getF64Or("tolerance", 1e-4);
    let seed = config.hyperparameters.getU64Or("seed", 42);
    let mut rng = Xoshiro256pp::fromSeed(seed);

    // Precompute |x|^2 once per training set (F3). Lloyd's identity
    // |x-c|^2 = |x|^2 - 2 x.c + |c|^2 lets the per-pair work drop to
    // one dot product and a constant lookup
    let mut xNormSq = vec![0.0f64; data.n];
    for i in 0..data.n {
        xNormSq[i] = normSquared(data.row(i));
    }

    let mut centroids = kmeansPpInit(data, k, &mut rng, &xNormSq);
    let mut centroidNormSq = vec![0.0f64; k];
    refreshCentroidNorms(&centroids, k, data.p, &mut centroidNormSq);
    let mut assignments = vec![0u32; data.n];
    // Reusable scratch buffers for the centroid update step, allocated
    // once and zeroed at the top of each iteration. Replaces the prior
    // form which allocated fresh Vecs every iteration
    let mut newCentroids = vec![0.0f64; k * data.p];
    let mut counts = vec![0u64; k];
    let mut prevInertia = f64::INFINITY;
    let mut iter = 0usize;
    while iter < maxIter {
        let inertia = assignAndComputeInertia(
            data,
            &centroids,
            k,
            &mut assignments,
            &xNormSq,
            &centroidNormSq,
        );
        let shifted = updateCentroidsInto(
            data,
            &assignments,
            &mut centroids,
            k,
            &mut newCentroids,
            &mut counts,
        );
        if shifted {
            refreshCentroidNorms(&centroids, k, data.p, &mut centroidNormSq);
        }
        if !shifted {
            break;
        }
        if (prevInertia - inertia).abs() < tol {
            break;
        }
        prevInertia = inertia;
        iter += 1;
    }
    let inertia = assignAndComputeInertia(
        data,
        &centroids,
        k,
        &mut assignments,
        &xNormSq,
        &centroidNormSq,
    );

    let mut model = TrainedModel::new(String::new(), ModelType::KMeans);
    model.featureColumns = config.featureColumns.clone();
    model.targetColumn = None;
    model.featureMean = vec![0.0; data.p];
    model.featureStd = vec![1.0; data.p];
    model.weights = centroids;
    model.data = ModelData::KMeans {
        k,
        nFeatures: data.p,
    };
    model.hyperparameters = config.hyperparameters.clone();
    model.trainingRows = data.n as u64;
    let mut metrics = ModelMetrics::default();
    metrics.inertia = Some(inertia);
    metrics.silhouette = Some(silhouetteEstimate(data, &assignments, k));
    model.metrics = metrics.intoMap();
    Ok(model)
}

fn kmeansPpInit(
    data: &TrainingData,
    k: usize,
    rng: &mut Xoshiro256pp,
    xNormSq: &[f64],
) -> Vec<f64> {
    let p = data.p;
    let dotFn = resolveDot();
    let mut centroids = Vec::with_capacity(k * p);
    let firstIdx = rng.nextRange(data.n as u64) as usize;
    centroids.extend_from_slice(data.row(firstIdx));
    let mut centroidNormSq = normSquared(&centroids[0..p]);

    let mut minDist = vec![0.0f64; data.n];
    for i in 0..data.n {
        let xdotc = unsafe { dotFn(data.row(i).as_ptr(), centroids[0..p].as_ptr(), p) };
        let d = xNormSq[i] - 2.0 * xdotc + centroidNormSq;
        minDist[i] = if d > 0.0 { d } else { 0.0 };
    }
    for _ in 1..k {
        let total: f64 = minDist.iter().sum();
        if total <= 0.0 {
            let idx = rng.nextRange(data.n as u64) as usize;
            centroids.extend_from_slice(data.row(idx));
        } else {
            let r = rng.nextF64() * total;
            let mut cum = 0.0;
            let mut chosen = data.n - 1;
            for i in 0..data.n {
                cum += minDist[i];
                if cum >= r {
                    chosen = i;
                    break;
                }
            }
            centroids.extend_from_slice(data.row(chosen));
        }
        let lastStart = centroids.len() - p;
        centroidNormSq = normSquared(&centroids[lastStart..lastStart + p]);
        for i in 0..data.n {
            let xdotc = unsafe {
                dotFn(
                    data.row(i).as_ptr(),
                    centroids[lastStart..lastStart + p].as_ptr(),
                    p,
                )
            };
            let d = xNormSq[i] - 2.0 * xdotc + centroidNormSq;
            let d = if d > 0.0 { d } else { 0.0 };
            if d < minDist[i] {
                minDist[i] = d;
            }
        }
    }
    centroids
}

fn refreshCentroidNorms(centroids: &[f64], k: usize, p: usize, out: &mut [f64]) {
    for c in 0..k {
        let cstart = c * p;
        out[c] = normSquared(&centroids[cstart..cstart + p]);
    }
}

fn assignAndComputeInertia(
    data: &TrainingData,
    centroids: &[f64],
    k: usize,
    assignments: &mut [u32],
    xNormSq: &[f64],
    centroidNormSq: &[f64],
) -> f64 {
    let p = data.p;
    let mut inertia = 0.0f64;
    // Resolve the SIMD dot kernel once, then call the fn pointer directly
    // from the inner loop. Avoids the OnceLock::get_or_init touch per
    // (row, centroid) pair which is otherwise n*k function calls
    let dotFn = resolveDot();
    for i in 0..data.n {
        let row = data.row(i);
        let xn = xNormSq[i];
        let mut bestC = 0usize;
        let mut bestD = f64::INFINITY;
        for c in 0..k {
            let cstart = c * p;
            let xdotc =
                unsafe { dotFn(row.as_ptr(), centroids[cstart..cstart + p].as_ptr(), p) };
            let d = xn - 2.0 * xdotc + centroidNormSq[c];
            if d < bestD {
                bestD = d;
                bestC = c;
            }
        }
        assignments[i] = bestC as u32;
        if bestD > 0.0 {
            inertia += bestD;
        }
    }
    inertia
}

fn updateCentroidsInto(
    data: &TrainingData,
    assignments: &[u32],
    centroids: &mut [f64],
    k: usize,
    newCentroids: &mut [f64],
    counts: &mut [u64],
) -> bool {
    for v in newCentroids.iter_mut() {
        *v = 0.0;
    }
    for c in counts.iter_mut() {
        *c = 0;
    }
    for i in 0..data.n {
        let c = assignments[i] as usize;
        counts[c] += 1;
        let cstart = c * data.p;
        let row = data.row(i);
        for j in 0..data.p {
            newCentroids[cstart + j] += row[j];
        }
    }
    let mut shifted = false;
    for c in 0..k {
        if counts[c] == 0 {
            continue;
        }
        let cstart = c * data.p;
        for j in 0..data.p {
            let v = newCentroids[cstart + j] / counts[c] as f64;
            if (centroids[cstart + j] - v).abs() > 1e-12 {
                shifted = true;
            }
            centroids[cstart + j] = v;
        }
    }
    shifted
}

/// Cheap silhouette score, sampled for large n
fn silhouetteEstimate(data: &TrainingData, assignments: &[u32], k: usize) -> f64 {
    let n = data.n;
    if n < 2 || k < 2 {
        return 0.0;
    }
    let cap = 1024usize.min(n);
    let stride = (n / cap).max(1);
    let mut total = 0.0f64;
    let mut samples = 0usize;
    for i in (0..n).step_by(stride) {
        let cluster = assignments[i] as usize;
        let mut a = 0.0f64;
        let mut aCount = 0u64;
        let mut bMin = f64::INFINITY;
        for c in 0..k {
            let mut sum = 0.0f64;
            let mut cnt = 0u64;
            for j in (0..n).step_by(stride) {
                if i == j {
                    continue;
                }
                if assignments[j] as usize == c {
                    sum += sqDistance(data.row(i), data.row(j)).sqrt();
                    cnt += 1;
                }
            }
            if cnt == 0 {
                continue;
            }
            let avg = sum / cnt as f64;
            if c == cluster {
                a = avg;
                aCount = cnt;
            } else if avg < bMin {
                bMin = avg;
            }
        }
        if aCount == 0 || !bMin.is_finite() {
            continue;
        }
        let s = (bMin - a) / a.max(bMin);
        total += s;
        samples += 1;
    }
    if samples == 0 {
        0.0
    } else {
        total / samples as f64
    }
}

pub fn predictCluster(centroids: &[f64], k: usize, p: usize, features: &[f64]) -> usize {
    let mut bestC = 0usize;
    let mut bestD = f64::INFINITY;
    for c in 0..k {
        let cstart = c * p;
        let d = sqDistance(features, &centroids[cstart..cstart + p]);
        if d < bestD {
            bestD = d;
            bestC = c;
        }
    }
    bestC
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn discoversTwoClusters() {
        let mut rng = Xoshiro256pp::fromSeed(5);
        let mut xs = Vec::new();
        let mut ys = Vec::new();
        for _ in 0..200 {
            xs.push(-3.0 + 0.4 * rng.nextNormal());
            xs.push(-3.0 + 0.4 * rng.nextNormal());
            ys.push(0.0);
        }
        for _ in 0..200 {
            xs.push(3.0 + 0.4 * rng.nextNormal());
            xs.push(3.0 + 0.4 * rng.nextNormal());
            ys.push(1.0);
        }
        let data = TrainingData::new(&xs, &ys, ys.len(), 2);
        let mut config = ModelConfig::new(ModelType::KMeans, vec!["x".into(), "y".into()]);
        config.hyperparameters.setF64("k", 2.0);
        let model = train(&config, &data).unwrap();
        let inertia = model.metrics.get("inertia").copied().unwrap();
        assert!(inertia.is_finite());
        assert!(model.weights.len() == 4);
        // After training, the two centroids should be near (-3,-3) and (3,3)
        let c0 = &model.weights[0..2];
        let c1 = &model.weights[2..4];
        let near0 = (c0[0] + 3.0).abs() < 0.5 && (c0[1] + 3.0).abs() < 0.5;
        let near1 = (c1[0] - 3.0).abs() < 0.5 && (c1[1] - 3.0).abs() < 0.5;
        let alt0 = (c0[0] - 3.0).abs() < 0.5 && (c0[1] - 3.0).abs() < 0.5;
        let alt1 = (c1[0] + 3.0).abs() < 0.5 && (c1[1] + 3.0).abs() < 0.5;
        assert!((near0 && near1) || (alt0 && alt1), "centroids = {:?}", model.weights);
    }
}
