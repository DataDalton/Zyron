//! Data profiling and automatic parameter tuning for vector indexes.
//!
//! DataProfile is computed once from a sample of vectors at build time. All HNSW
//! and IVF-PQ parameters become functions of the profile, eliminating hardcoded
//! constants.

use super::distance::{distWithFn, resolveDistFn};
use super::types::DistanceMetric;

/// Characterizes a vector dataset from a small sample, driving all index
/// parameter computations. Computed in O(sample^2) pairwise distance calls.
#[derive(Debug, Clone)]
pub struct DataProfile {
    /// Total vector count in the dataset.
    pub n: usize,
    /// Vector dimensionality.
    pub d: u16,
    /// Distance spread: (p90 - p10) / median of pairwise distances.
    /// Low values (0.05) indicate hard datasets. High values (0.5+) indicate easy datasets.
    pub distSpread: f32,
    /// Approximate intrinsic dimensionality, clamped to [2.0, d].
    pub intrinsicDim: f32,
    /// Whether the distance distribution is bimodal (clustered data).
    pub isClustered: bool,
    /// Number of CPU cores available for parallel operations.
    pub numCores: usize,
}

impl DataProfile {
    /// Builds a profile without the O(sample^2) pairwise distance scan.
    /// Fills n, d, and numCores (the only fields build/build_owned consume,
    /// via parallelThreshold) and leaves the distribution fields at neutral
    /// defaults. Callers that need distSpread/intrinsicDim/isClustered for
    /// parameter auto-tuning must use compute instead.
    pub fn lightweight(totalN: usize, d: u16) -> Self {
        let numCores = std::thread::available_parallelism()
            .map(|p| p.get())
            .unwrap_or(1);
        Self {
            n: totalN,
            d,
            distSpread: 0.5,
            intrinsicDim: d as f32,
            isClustered: false,
            numCores,
        }
    }

    /// Computes a DataProfile from a sample of vectors.
    /// Samples min(1000, N) vectors and measures pairwise distance distribution.
    pub fn compute(vectors: &[&[f32]], totalN: usize, d: u16, metric: DistanceMetric) -> Self {
        let numCores = std::thread::available_parallelism()
            .map(|p| p.get())
            .unwrap_or(1);

        if vectors.is_empty() {
            return Self {
                n: totalN,
                d,
                distSpread: 0.5,
                intrinsicDim: d as f32,
                isClustered: false,
                numCores,
            };
        }

        // Sample up to 1000 vectors using random sampling (xorshift RNG) to
        // avoid stride patterns that systematically miss clusters between
        // sampling points.
        let sampleSize = vectors.len().min(1000);
        let sample: Vec<&[f32]> = if vectors.len() <= sampleSize {
            vectors.to_vec()
        } else {
            let mut rng: u64 = 0x9E3779B97F4A7C15u64.wrapping_mul(vectors.len() as u64 | 1);
            if rng == 0 {
                rng = 1;
            }
            let mut picks: Vec<usize> = Vec::with_capacity(sampleSize);
            let mut seen = std::collections::HashSet::with_capacity(sampleSize);
            while picks.len() < sampleSize {
                rng ^= rng << 13;
                rng ^= rng >> 7;
                rng ^= rng << 17;
                let idx = (rng as usize) % vectors.len();
                if seen.insert(idx) {
                    picks.push(idx);
                }
            }
            picks.into_iter().map(|i| vectors[i]).collect()
        };

        // Detect unit-sphere normalized data by checking sample vector norms.
        // When data is normalized and the requested metric is Euclidean, profiling
        // switches to Cosine distance to characterize angular geometry accurately.
        // Euclidean on the unit sphere satisfies euclidean = sqrt(2 * cosine), which
        // compresses bimodal distributions and hides cluster structure.
        let isNormalized = if sampleSize >= 10 {
            let checkCount = sampleSize.min(50);
            let norms: Vec<f32> = sample[..checkCount]
                .iter()
                .map(|v| v.iter().map(|x| x * x).sum::<f32>().sqrt())
                .collect();
            let meanNorm = norms.iter().sum::<f32>() / norms.len() as f32;
            let variance =
                norms.iter().map(|n| (n - meanNorm).powi(2)).sum::<f32>() / norms.len() as f32;
            meanNorm > 0.95 && meanNorm < 1.05 && variance.sqrt() < 0.05
        } else {
            false
        };

        let profileMetric = if isNormalized && metric == DistanceMetric::Euclidean {
            DistanceMetric::Cosine
        } else {
            metric
        };

        // Compute all pairwise distances in the sample.
        let pairCount = sampleSize * (sampleSize - 1) / 2;
        let mut distances = Vec::with_capacity(pairCount);

        let profileDistFn = resolveDistFn(profileMetric);
        for i in 0..sampleSize {
            for j in (i + 1)..sampleSize {
                let dist = distWithFn(profileDistFn, sample[i], sample[j]);
                if dist.is_finite() {
                    distances.push(dist);
                }
            }
        }

        if distances.is_empty() {
            return Self {
                n: totalN,
                d,
                distSpread: 0.5,
                intrinsicDim: d as f32,
                isClustered: false,
                numCores,
            };
        }

        distances.sort_unstable_by(|a, b| a.total_cmp(b));
        let len = distances.len();

        let p10 = distances[len / 10];
        let median = distances[len / 2];
        let p90 = distances[len * 9 / 10];

        let distSpread = if median > 1e-10 {
            ((p90 - p10) / median).clamp(0.01, 2.0)
        } else {
            0.01
        };

        // Intrinsic dimensionality estimate: ~1/distSpread^2, clamped to [2, d].
        let intrinsicDim = (1.0 / (distSpread * distSpread)).clamp(2.0, d as f32);

        // Detect clustering via two complementary heuristics.
        //
        // Heuristic 1 (quartile spread): if p75-p25 is much wider than p10-p25,
        // the distribution is skewed in a way consistent with bimodal (clustered)
        // data. Works when intra-cluster pairs form a visible fraction (few clusters).
        //
        // Heuristic 2 (low-percentile gap): if the median distance is much larger
        // than the 2nd percentile, there is a concentration of close pairs (intra-
        // cluster) distinct from the bulk (inter-cluster). Works even when clusters
        // are numerous and intra-cluster pairs are a small fraction of all pairs.
        let isClustered = if len >= 20 {
            let p25 = distances[len / 4];
            let p75 = distances[len * 3 / 4];
            let lowerSpread = p25 - p10;
            let upperSpread = p75 - p25;
            let quartileTest = lowerSpread > 1e-10 && upperSpread / lowerSpread > 3.0;

            let p1 = distances[len / 100];
            let gapTest = p1 > 1e-10 && median / p1 > 3.0;

            quartileTest || gapTest
        } else {
            false
        };

        Self {
            n: totalN,
            d,
            distSpread,
            intrinsicDim,
            isClustered,
            numCores,
        }
    }

    // -----------------------------------------------------------------------
    // HNSW parameter formulas
    // -----------------------------------------------------------------------

    /// Maximum connections per node per layer.
    /// Computed from intrinsicDim (effective dimensionality of the data).
    /// Structured data (low intrinsicDim) has low local dimensionality and
    /// needs fewer connections to represent neighborhoods accurately.
    /// Unstructured data (high intrinsicDim) spreads mass across many
    /// directions and needs more connections per node.
    ///
    /// Formula: m = 13 + intrinsicDim * 0.5 + log2(N / 100_000) * 2.
    /// The intrinsicDim term sizes the graph for the data's local geometry.
    /// The log(N) term grows m with dataset size so HNSW's required edge
    /// count for constant recall is satisfied at any scale.
    /// The floor of 16 prevents degenerate m on trivially-separable data.
    /// The upper bound is the vector's own dimensionality since there can
    /// be no more meaningful orthogonal connections than directions in
    /// the embedding space.
    pub fn hnswM(&self) -> u16 {
        let m_base = 13.0 + self.intrinsicDim as f64 * 0.5;
        let n_scale = (self.n as f64 / 100_000.0).max(1.0).log2() * 2.0;
        let m = (m_base + n_scale).round();
        let upper = (self.d as f64).max(16.0);
        m.clamp(16.0, upper) as u16
    }

    /// Build-time beam width.
    /// Scales with m and data difficulty (1/distSpread). Harder distance
    /// distributions need deeper search during build to find quality neighbors.
    ///
    /// Formula: m * (8 + difficulty), clamped [2*m, 1000].
    pub fn hnswEfConstruction(&self, m: u16) -> u16 {
        let difficulty = 1.0 / self.distSpread as f64;
        let ef = m as f64 * (8.0 + difficulty);
        ef.round().clamp(2.0 * m as f64, 1000.0) as u16
    }

    /// Query-time beam width.
    /// Scales with m, data difficulty, and dataset size. Larger graphs and
    /// harder distance distributions need more candidates to find the true
    /// top-k. The formula has no upper cap so it remains valid at any scale.
    ///
    /// Formula: 2 * m * (1 + difficulty) * sqrt(N / 10000)
    /// Floor: 2*m, the minimum meaningful beam width.
    pub fn hnswEfSearch(&self, m: u16) -> u16 {
        let difficulty = 1.0 / self.distSpread as f64;
        let nScale = (self.n as f64 / 10_000.0).sqrt().max(1.0);
        let ef = 2.0 * m as f64 * (1.0 + difficulty) * nScale;
        ef.round().max(2.0 * m as f64) as u16
    }

    /// Parallel build threshold. Below this count, single-threaded build is used.
    /// Formula: max(10000, 50000 / cores).
    pub fn parallelThreshold(&self) -> usize {
        (50_000 / self.numCores.max(1)).max(10_000)
    }

    // -----------------------------------------------------------------------
    // IVF-PQ parameter formulas
    // -----------------------------------------------------------------------

    /// Number of Voronoi partitions.
    /// Formula: clamp(sqrt(N) * (clustered ? 1.5 : 0.85), 16, 65536).
    /// The clustered branch oversamples past the sqrt(N) baseline because
    /// structure lets extra centroids pay off in sharper partition boundaries.
    /// The uniform branch undersamples because extra centroids add probe cost
    /// without recall gain when points are spread evenly across the space.
    pub fn ivfNumCentroids(&self) -> u32 {
        let base = (self.n as f64).sqrt();
        let scaled = if self.isClustered {
            base * 1.5
        } else {
            base * 0.85
        };
        scaled.round().clamp(16.0, 65536.0) as u32
    }

    /// Number of PQ sub-vector segments.
    /// Picks the divisor of d closest to the target sub-dim ratio.
    /// Target subDim = clamp(d/16, 4, 8). Fallback finds nearest valid divisor
    /// using distance-to-target ranking rather than just "largest <= numSub".
    pub fn ivfNumSubvectors(&self) -> u16 {
        let d = self.d as usize;
        let targetSubDim = (d / 16).clamp(4, 8);
        let targetNumSub = d / targetSubDim;
        // Try exact match first.
        if d % targetNumSub == 0 {
            return targetNumSub as u16;
        }
        // Find divisor of d nearest to targetNumSub.
        let mut bestDivisor = 1usize;
        let mut bestDiff = targetNumSub;
        for candidate in 1..=d {
            if d % candidate == 0 {
                let diff = (candidate as i32 - targetNumSub as i32).unsigned_abs() as usize;
                if diff < bestDiff {
                    bestDiff = diff;
                    bestDivisor = candidate;
                }
            }
        }
        bestDivisor as u16
    }

    /// Number of partitions probed at query time.
    /// Formula: clamp(sqrt(centroids) * difficulty^1.3, 1, centroids).
    /// Difficulty exponent above 1.0 means hard data (low distSpread) probes
    /// aggressively since nearest neighbors are spread across many cells.
    /// No artificial cap below centroids: on pathological data (uniform random)
    /// probing all cells is correct since IVF partitioning provides no locality
    /// benefit. On well-clustered data (difficulty ~2-3) probes ~20-30% of cells.
    pub fn ivfNumProbes(&self, numCentroids: u32) -> u16 {
        let difficulty = 1.0 / self.distSpread as f64;
        let probes = (numCentroids as f64).sqrt() * difficulty.powf(1.3);
        probes.round().clamp(1.0, numCentroids as f64) as u16
    }

    /// Number of residual centroids for ARQ stage 2.
    /// Formula: clamp(sqrt(N) / (distSpread * 2 + 0.5), 32, 512).
    /// Inverted vs prior: hard data (low distSpread) needs MORE residual
    /// centroids for quantization accuracy, easy data gets fewer.
    pub fn ivfResidualCentroids(&self) -> usize {
        let rc = (self.n as f64).sqrt() / (self.distSpread as f64 * 2.0 + 0.5);
        rc.round().clamp(32.0, 512.0) as usize
    }

    /// K-means iteration limit. Most convergence happens in the first 10
    /// iterations, so the cap is kept low for build speed.
    /// Formula: clamp(8 + ln(N)/2, 8, 20).
    pub fn kmeansMaxIters(&self) -> usize {
        let iters = 8.0 + (self.n.max(2) as f64).ln() / 2.0;
        iters.round().clamp(8.0, 20.0) as usize
    }

    // -----------------------------------------------------------------------
    // Serialization
    // -----------------------------------------------------------------------

    /// Serializes to bytes: 4(n) + 2(d) + 4(distSpread) + 4(intrinsicDim) + 1(isClustered) + 1(numCores) = 16 bytes.
    pub fn toBytes(&self) -> Vec<u8> {
        let mut buf = Vec::with_capacity(16);
        buf.extend_from_slice(&(self.n as u32).to_le_bytes());
        buf.extend_from_slice(&self.d.to_le_bytes());
        buf.extend_from_slice(&self.distSpread.to_le_bytes());
        buf.extend_from_slice(&self.intrinsicDim.to_le_bytes());
        buf.push(if self.isClustered { 1 } else { 0 });
        buf.push(self.numCores.min(255) as u8);
        buf
    }

    /// Deserializes from bytes produced by toBytes.
    pub fn fromBytes(data: &[u8]) -> Option<Self> {
        if data.len() < 16 {
            return None;
        }
        let n = u32::from_le_bytes([data[0], data[1], data[2], data[3]]) as usize;
        let d = u16::from_le_bytes([data[4], data[5]]);
        let distSpread = f32::from_le_bytes([data[6], data[7], data[8], data[9]]);
        let intrinsicDim = f32::from_le_bytes([data[10], data[11], data[12], data[13]]);
        let isClustered = data[14] != 0;
        let numCores = data[15] as usize;

        Some(Self {
            n,
            d,
            distSpread,
            intrinsicDim,
            isClustered,
            numCores: if numCores == 0 {
                std::thread::available_parallelism()
                    .map(|p| p.get())
                    .unwrap_or(1)
            } else {
                numCores
            },
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn makeSample(n: usize, d: usize, seed: u64) -> Vec<Vec<f32>> {
        let mut rng = seed;
        (0..n)
            .map(|_| {
                (0..d)
                    .map(|_| {
                        rng ^= rng << 13;
                        rng ^= rng >> 7;
                        rng ^= rng << 17;
                        (rng as f32) / (u64::MAX as f32) * 2.0 - 1.0
                    })
                    .collect()
            })
            .collect()
    }

    #[test]
    fn profileComputeBasic() {
        let vecs = makeSample(200, 128, 42);
        let slices: Vec<&[f32]> = vecs.iter().map(|v| v.as_slice()).collect();
        let profile = DataProfile::compute(&slices, 100_000, 128, DistanceMetric::Euclidean);

        assert_eq!(profile.n, 100_000);
        assert_eq!(profile.d, 128);
        assert!(profile.distSpread > 0.0);
        assert!(profile.intrinsicDim >= 2.0);
        assert!(profile.intrinsicDim <= 128.0);
    }

    #[test]
    fn profileComputeEmpty() {
        let profile = DataProfile::compute(&[], 0, 64, DistanceMetric::Cosine);
        assert_eq!(profile.n, 0);
        assert_eq!(profile.distSpread, 0.5);
    }

    #[test]
    fn hnswParamsScaleWithN() {
        let vecs = makeSample(200, 128, 42);
        let slices: Vec<&[f32]> = vecs.iter().map(|v| v.as_slice()).collect();

        let small = DataProfile::compute(&slices, 1_000, 128, DistanceMetric::Euclidean);
        let large = DataProfile::compute(&slices, 10_000_000, 128, DistanceMetric::Euclidean);

        // Larger N should produce larger m.
        assert!(large.hnswM() >= small.hnswM());
    }

    #[test]
    fn ivfParamsReasonable() {
        let vecs = makeSample(200, 128, 42);
        let slices: Vec<&[f32]> = vecs.iter().map(|v| v.as_slice()).collect();
        let profile = DataProfile::compute(&slices, 1_000_000, 128, DistanceMetric::Euclidean);

        let centroids = profile.ivfNumCentroids();
        assert!(centroids >= 16);
        assert!(centroids <= 65536);

        let subs = profile.ivfNumSubvectors();
        assert!(128 % subs as usize == 0);

        let probes = profile.ivfNumProbes(centroids);
        assert!(probes >= 1);
        assert!(probes as u32 <= centroids / 2);
    }

    #[test]
    fn profileSerializationRoundTrip() {
        let profile = DataProfile {
            n: 500_000,
            d: 256,
            distSpread: 0.15,
            intrinsicDim: 44.4,
            isClustered: true,
            numCores: 16,
        };

        let bytes = profile.toBytes();
        assert_eq!(bytes.len(), 16);

        let restored = DataProfile::fromBytes(&bytes).expect("valid deserialization");
        assert_eq!(restored.n, profile.n);
        assert_eq!(restored.d, profile.d);
        assert!((restored.distSpread - profile.distSpread).abs() < 1e-6);
        assert!((restored.intrinsicDim - profile.intrinsicDim).abs() < 1e-3);
        assert_eq!(restored.isClustered, profile.isClustered);
        assert_eq!(restored.numCores, profile.numCores);
    }

    /// Generates clustered vectors normalized to the unit sphere.
    fn makeNormalizedClusteredSample(
        n: usize,
        d: usize,
        numClusters: usize,
        stdDev: f32,
        seed: u64,
    ) -> Vec<Vec<f32>> {
        let mut rng = seed;
        let nextF32 = |state: &mut u64| -> f32 {
            *state ^= *state << 13;
            *state ^= *state >> 7;
            *state ^= *state << 17;
            (*state as f32) / (u64::MAX as f32) * 2.0 - 1.0
        };
        let centers: Vec<Vec<f32>> = (0..numClusters)
            .map(|_| (0..d).map(|_| nextF32(&mut rng) * 2.0).collect())
            .collect();
        (0..n)
            .map(|_| {
                let ci = (rng as usize) % numClusters;
                rng ^= rng << 13;
                rng ^= rng >> 7;
                rng ^= rng << 17;
                let mut v: Vec<f32> = (0..d)
                    .map(|dim| centers[ci][dim] + nextF32(&mut rng) * stdDev)
                    .collect();
                let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-10);
                for x in v.iter_mut() {
                    *x /= norm;
                }
                v
            })
            .collect()
    }

    #[test]
    fn clusterDetectionNormalizedEuclidean() {
        let vecs = makeNormalizedClusteredSample(500, 64, 16, 0.15, 99);
        let slices: Vec<&[f32]> = vecs.iter().map(|v| v.as_slice()).collect();
        let profile = DataProfile::compute(&slices, 500, 64, DistanceMetric::Euclidean);
        assert!(
            profile.isClustered,
            "should detect clusters on normalized data with Euclidean metric"
        );
    }

    #[test]
    fn clusterDetectionNormalizedCosine() {
        let vecs = makeNormalizedClusteredSample(500, 64, 16, 0.15, 99);
        let slices: Vec<&[f32]> = vecs.iter().map(|v| v.as_slice()).collect();
        let profile = DataProfile::compute(&slices, 500, 64, DistanceMetric::Cosine);
        assert!(
            profile.isClustered,
            "should detect clusters on normalized data with Cosine metric"
        );
    }

    #[test]
    fn normalizedUniformNotClustered() {
        let mut rng = 42u64;
        let vecs: Vec<Vec<f32>> = (0..500)
            .map(|_| {
                let mut v: Vec<f32> = (0..64)
                    .map(|_| {
                        rng ^= rng << 13;
                        rng ^= rng >> 7;
                        rng ^= rng << 17;
                        (rng as f32) / (u64::MAX as f32) * 2.0 - 1.0
                    })
                    .collect();
                let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-10);
                for x in v.iter_mut() {
                    *x /= norm;
                }
                v
            })
            .collect();
        let slices: Vec<&[f32]> = vecs.iter().map(|v| v.as_slice()).collect();
        let profile = DataProfile::compute(&slices, 500, 64, DistanceMetric::Euclidean);
        assert!(
            !profile.isClustered,
            "uniform random normalized data should not be detected as clustered"
        );
    }

    #[test]
    fn distSpreadReasonableForNormalizedClusters() {
        let vecs = makeNormalizedClusteredSample(500, 64, 16, 0.15, 99);
        let slices: Vec<&[f32]> = vecs.iter().map(|v| v.as_slice()).collect();
        let profile = DataProfile::compute(&slices, 500, 64, DistanceMetric::Euclidean);
        assert!(
            profile.distSpread > 0.15,
            "distSpread should be reasonable for normalized clusters, got {}",
            profile.distSpread
        );
    }
}
