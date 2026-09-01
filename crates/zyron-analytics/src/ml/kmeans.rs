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
            let xdotc = unsafe { dotFn(row.as_ptr(), centroids[cstart..cstart + p].as_ptr(), p) };
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

// Seed used when a caller does not supply one, keeps repeated runs stable
pub const DEFAULT_KMEANS_SEED: u64 = 42;

#[derive(Debug, Clone)]
pub struct KmeansClusterOutput {
    // per row (clusterId, euclidean distance to that centroid)
    pub assignments: Vec<(usize, f64)>,
    pub centroids: Vec<Vec<f64>>,
}

fn flattenValidated(rows: &[Vec<f64>]) -> Result<(Vec<f64>, usize, usize)> {
    let n = rows.len();
    if n == 0 {
        return Err(ZyronError::InvalidParameter {
            name: "rows".to_string(),
            value: "empty input".to_string(),
        });
    }
    let p = rows[0].len();
    if p == 0 {
        return Err(ZyronError::InvalidParameter {
            name: "rows".to_string(),
            value: "rows have no columns".to_string(),
        });
    }
    let mut xs = Vec::with_capacity(n * p);
    for r in rows {
        if r.len() != p {
            return Err(ZyronError::InvalidParameter {
                name: "rows".to_string(),
                value: "rows have inconsistent dimensions".to_string(),
            });
        }
        if r.iter().any(|v| !v.is_finite()) {
            return Err(ZyronError::InvalidParameter {
                name: "rows".to_string(),
                value: "rows contain non finite values".to_string(),
            });
        }
        xs.extend_from_slice(r);
    }
    Ok((xs, n, p))
}

fn trainOnRows(
    xs: &[f64],
    n: usize,
    p: usize,
    k: usize,
    maxIter: usize,
    seed: u64,
) -> Result<TrainedModel> {
    let ys = vec![0.0f64; n];
    let data = TrainingData::new(xs, &ys, n, p);
    let mut config = ModelConfig::new(
        ModelType::KMeans,
        (0..p).map(|j| format!("f{}", j)).collect(),
    );
    config.hyperparameters.setF64("k", k as f64);
    config
        .hyperparameters
        .setF64("max_iter", maxIter.max(1) as f64);
    config.hyperparameters.setF64("seed", seed as f64);
    train(&config, &data)
}

/// Clusters rows and returns per row assignment with distance plus the
/// fitted centroids
pub fn kmeansCluster(
    rows: &[Vec<f64>],
    k: usize,
    maxIter: usize,
    seed: u64,
) -> Result<KmeansClusterOutput> {
    let (xs, n, p) = flattenValidated(rows)?;
    if k == 0 {
        return Err(ZyronError::InvalidParameter {
            name: "k".to_string(),
            value: "k must be at least 1".to_string(),
        });
    }
    let model = trainOnRows(&xs, n, p, k, maxIter, seed)?;
    let fitted_k = match model.data {
        ModelData::KMeans { k, .. } => k,
        _ => {
            return Err(ZyronError::ExecutionError(
                "kmeans training returned unexpected model data".to_string(),
            ));
        }
    };
    let centroids_flat = &model.weights;
    let mut assignments = Vec::with_capacity(n);
    for r in rows {
        let c = predictCluster(centroids_flat, fitted_k, p, r);
        let cstart = c * p;
        let dist = sqDistance(r, &centroids_flat[cstart..cstart + p]).sqrt();
        assignments.push((c, dist));
    }
    let centroids = centroids_flat
        .chunks(p)
        .take(fitted_k)
        .map(|c| c.to_vec())
        .collect();
    Ok(KmeansClusterOutput {
        assignments,
        centroids,
    })
}

/// Fitted centroids only
pub fn kmeansCentroids(
    rows: &[Vec<f64>],
    k: usize,
    maxIter: usize,
    seed: u64,
) -> Result<Vec<Vec<f64>>> {
    Ok(kmeansCluster(rows, k, maxIter, seed)?.centroids)
}

/// Inertia per candidate k for elbow selection
/// Each k takes the best of three seeded restarts so the curve stays
/// close to the optimal nonincreasing shape
pub fn kmeansElbow(rows: &[Vec<f64>], kRange: &[usize]) -> Result<Vec<(usize, f64)>> {
    let (xs, n, p) = flattenValidated(rows)?;
    if kRange.is_empty() {
        return Err(ZyronError::InvalidParameter {
            name: "k_range".to_string(),
            value: "empty k range".to_string(),
        });
    }
    let mut out = Vec::with_capacity(kRange.len());
    for &k in kRange {
        if k == 0 || k > n {
            return Err(ZyronError::InvalidParameter {
                name: "k_range".to_string(),
                value: format!("k must be in 1..={}, got {}", n, k),
            });
        }
        let mut best = f64::INFINITY;
        for offset in 0..3u64 {
            let model = trainOnRows(&xs, n, p, k, 100, DEFAULT_KMEANS_SEED + offset)?;
            let inertia = model.metrics.get("inertia").copied().ok_or_else(|| {
                ZyronError::ExecutionError("kmeans training produced no inertia metric".to_string())
            })?;
            if inertia < best {
                best = inertia;
            }
        }
        out.push((k, best));
    }
    Ok(out)
}

/// Assigns rows to the nearest of the given centroids
pub fn kmeansPredict(rows: &[Vec<f64>], centroids: &[Vec<f64>]) -> Result<Vec<usize>> {
    let (_, _, p) = flattenValidated(rows)?;
    if centroids.is_empty() {
        return Err(ZyronError::InvalidParameter {
            name: "centroids".to_string(),
            value: "empty centroid list".to_string(),
        });
    }
    let mut flat = Vec::with_capacity(centroids.len() * p);
    for c in centroids {
        if c.len() != p {
            return Err(ZyronError::InvalidParameter {
                name: "centroids".to_string(),
                value: format!(
                    "centroid dimension {} does not match row dimension {}",
                    c.len(),
                    p
                ),
            });
        }
        if c.iter().any(|v| !v.is_finite()) {
            return Err(ZyronError::InvalidParameter {
                name: "centroids".to_string(),
                value: "centroids contain non finite values".to_string(),
            });
        }
        flat.extend_from_slice(c);
    }
    Ok(rows
        .iter()
        .map(|r| predictCluster(&flat, centroids.len(), p, r))
        .collect())
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
        assert!(
            (near0 && near1) || (alt0 && alt1),
            "centroids = {:?}",
            model.weights
        );
    }

    fn twoBlobRows(seed: u64) -> Vec<Vec<f64>> {
        let mut rng = Xoshiro256pp::fromSeed(seed);
        let mut rows = Vec::with_capacity(240);
        for _ in 0..120 {
            rows.push(vec![
                -4.0 + 0.3 * rng.nextNormal(),
                -4.0 + 0.3 * rng.nextNormal(),
            ]);
        }
        for _ in 0..120 {
            rows.push(vec![
                4.0 + 0.3 * rng.nextNormal(),
                4.0 + 0.3 * rng.nextNormal(),
            ]);
        }
        rows
    }

    #[test]
    fn kmeansClusterStableAcrossRunsWithSameSeed() {
        let rows = twoBlobRows(9);
        let a = kmeansCluster(&rows, 2, 100, DEFAULT_KMEANS_SEED).expect("first run");
        let b = kmeansCluster(&rows, 2, 100, DEFAULT_KMEANS_SEED).expect("second run");
        assert_eq!(a.centroids, b.centroids);
        for (x, y) in a.assignments.iter().zip(b.assignments.iter()) {
            assert_eq!(x.0, y.0);
            assert!((x.1 - y.1).abs() < 1e-12);
        }
    }

    #[test]
    fn kmeansElbowInertiaNonincreasing() {
        let rows = twoBlobRows(19);
        let curve = kmeansElbow(&rows, &[1, 2, 3, 4]).expect("elbow");
        assert_eq!(curve.len(), 4);
        for i in 1..curve.len() {
            assert!(
                curve[i].1 <= curve[i - 1].1 + 1e-9,
                "inertia rose from k={} ({}) to k={} ({})",
                curve[i - 1].0,
                curve[i - 1].1,
                curve[i].0,
                curve[i].1
            );
        }
        // the drop from 1 to 2 clusters dominates on two blob data
        assert!(curve[1].1 < curve[0].1 * 0.2);
    }

    #[test]
    fn kmeansPredictMatchesNearestCentroid() {
        let centroids = vec![vec![-4.0, -4.0], vec![4.0, 4.0]];
        let rows = vec![vec![-3.5, -4.2], vec![3.9, 4.4], vec![-4.0, -3.9]];
        let preds = kmeansPredict(&rows, &centroids).expect("predict");
        assert_eq!(preds, vec![0, 1, 0]);
        let bad = kmeansPredict(&rows, &[vec![1.0]]);
        assert!(bad.is_err());
    }
}
