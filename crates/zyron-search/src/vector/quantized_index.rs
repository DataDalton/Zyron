//! IVF-PQ (Inverted File with Product Quantization) index for large-scale vector search.
//!
//! Partitions vectors into Voronoi cells via k-means centroids, then compresses
//! residuals using product quantization. At query time, only a subset of partitions
//! (probes) are scanned using asymmetric distance computation (ADC) for fast
//! approximate nearest neighbor retrieval.

use std::sync::atomic::{AtomicU64, Ordering};

use parking_lot::RwLock;
use zyron_common::{Result, ZyronError};

use super::distance::{
    computeDistance, distWithFn, euclideanSmall4, resolveDistFn, vectorAddInplace,
    vectorScaleInplace, vectorSubtract,
};
use super::memory::{try_alloc_default, try_alloc_filled, try_alloc_vec, validate_file_size};
use super::profile::DataProfile;
use super::types::{DistanceMetric, IvfPqConfig, VectorId, VectorSearch};

/// Returns the number of threads to use for parallel operations.
fn threadCount() -> usize {
    std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1)
}

/// Distance computation specialized for small subvectors (4d Euclidean).
/// Falls back to computeDistance for other metrics or dimensions.
#[inline(always)]
fn subvectorDistance(a: &[f32], b: &[f32], metric: DistanceMetric) -> f32 {
    if a.len() == 4 && metric == DistanceMetric::Euclidean {
        euclideanSmall4(a, b)
    } else {
        computeDistance(metric, a, b)
    }
}

// ---------------------------------------------------------------------------
// File format magic bytes
// ---------------------------------------------------------------------------

const MAGIC: &[u8; 6] = b"ZYPQ\x01\x00";

// ---------------------------------------------------------------------------
// Deterministic xorshift64 RNG (no external rand crate)
// ---------------------------------------------------------------------------

fn xorshift64(state: &mut u64) -> u64 {
    let mut x = *state;
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    *state = x;
    x
}

// ---------------------------------------------------------------------------
// IvfPqIndex
// ---------------------------------------------------------------------------

/// Inverted File with Adaptive Residual Quantization (IVF-ARQ) index.
///
/// Uses IVF centroid partitioning to narrow the search space, then computes
/// exact distances on original vectors within probed partitions. PQ codes
/// are maintained for optional fast pre-filtering on very large datasets.
/// The vector arena stores all original vectors contiguously for cache-friendly
/// exact distance computation during search.
pub struct IvfPqIndex {
    pub indexId: u32,
    pub tableId: u32,
    pub columnId: u16,
    config: IvfPqConfig,
    dimensions: u16,
    /// Stage 1: IVF centroids. centroids[i] is Vec<f32> of length dimensions.
    centroids: Vec<Vec<f32>>,
    /// Residual centroids for ARQ stage 2.
    residualCentroids: Vec<Vec<f32>>,
    /// invertedLists[centroid_idx] holds (VectorId, arenaIdx, residual_centroid_idx, pq_codes).
    /// arenaIdx (u32) indexes directly into vectorArena for exact distance lookup.
    invertedLists: Vec<RwLock<Vec<(VectorId, u32, u16, Vec<u8>)>>>,
    /// PQ codebooks for stage 3.
    codebooks: Vec<Vec<Vec<f32>>>,
    /// Flat arena storing all original vectors contiguously for exact distance
    /// computation during search. Behind RwLock for online insert support.
    vectorArena: RwLock<Vec<f32>>,
    /// Maps VectorId to arena index for exact distance lookup.
    idToArenaIdx: scc::HashMap<VectorId, usize>,
    nodeCount: AtomicU64,
    profile: Option<DataProfile>,
}

impl IvfPqIndex {
    /// Builds an IVF-PQ index from a batch of vectors.
    ///
    /// Steps: k-means clustering for centroids, residual computation,
    /// product quantization codebook training, and PQ encoding into inverted lists.
    pub fn build(
        vectors: &[(VectorId, &[f32])],
        indexId: u32,
        tableId: u32,
        columnId: u16,
        config: IvfPqConfig,
    ) -> Result<Self> {
        if vectors.is_empty() {
            return Ok(Self {
                indexId,
                tableId,
                columnId,
                dimensions: 0,
                centroids: Vec::new(),
                residualCentroids: Vec::new(),
                invertedLists: Vec::new(),
                codebooks: Vec::new(),
                vectorArena: RwLock::new(Vec::new()),
                idToArenaIdx: scc::HashMap::new(),
                nodeCount: AtomicU64::new(0),
                config,
                profile: None,
            });
        }

        let dimensions = vectors[0].1.len();
        if dimensions == 0 || dimensions > u16::MAX as usize {
            return Err(ZyronError::InvalidParameter {
                name: "dimensions".to_string(),
                value: format!("{}", dimensions),
            });
        }
        let dims = dimensions as u16;

        // Validate all vectors have matching dimensionality.
        for (i, (_, v)) in vectors.iter().enumerate() {
            if v.len() != dimensions {
                return Err(ZyronError::InvalidParameter {
                    name: "vector dimensions".to_string(),
                    value: format!(
                        "vector {} has {} dimensions, expected {}",
                        i,
                        v.len(),
                        dimensions
                    ),
                });
            }
        }

        let numSubvectors = config.numSubvectors as usize;
        if numSubvectors == 0 {
            return Err(ZyronError::InvalidParameter {
                name: "numSubvectors".to_string(),
                value: "0".to_string(),
            });
        }
        if dimensions % numSubvectors != 0 {
            return Err(ZyronError::InvalidParameter {
                name: "numSubvectors".to_string(),
                value: format!(
                    "{} does not evenly divide dimensions {}",
                    numSubvectors, dimensions
                ),
            });
        }

        let subDim = dimensions / numSubvectors;
        let codebookSize = 1usize << config.bitsPerCode;

        // Compute DataProfile from a sample to drive adaptive parameters.
        let sampleSize = vectors.len().min(1000);
        let sampleSlices: Vec<&[f32]> = vectors[..sampleSize].iter().map(|(_, v)| *v).collect();
        let profile = DataProfile::compute(&sampleSlices, vectors.len(), dims, config.metric);
        let kmeansIters = profile.kmeansMaxIters();

        // Clamp numCentroids to the number of vectors.
        let numCentroids = (config.numCentroids as usize).min(vectors.len());

        // Step 1: K-means clustering to find centroids.
        let vectorSlices: Vec<&[f32]> = vectors.iter().map(|(_, v)| *v).collect();
        let centroids = kmeans(&vectorSlices, numCentroids, kmeansIters, config.metric)?;

        // Step 2: Assign each vector to its nearest centroid and compute stage-1 residuals.
        // Uses try_alloc_* helpers so OOM returns an error rather than aborting.
        let mut assignments: Vec<Vec<usize>> = try_alloc_filled(numCentroids, Vec::new())?;
        let mut residuals1: Vec<Vec<f32>> = try_alloc_vec(vectors.len())?;

        for (idx, (_, v)) in vectors.iter().enumerate() {
            let centroidIdx = findNearest(v, &centroids, config.metric);
            assignments[centroidIdx].push(idx);
            let mut residual = vec![0.0f32; dimensions];
            vectorSubtract(v, &centroids[centroidIdx], &mut residual);
            residuals1.push(residual);
        }

        // Step 2.5 (ARQ Stage 2): Train residual centroids on stage-1 residuals.
        let numResidualCentroids = profile.ivfResidualCentroids().min(vectors.len());
        let residualSlices: Vec<&[f32]> = residuals1.iter().map(|r| r.as_slice()).collect();
        let residualCentroids = kmeans(
            &residualSlices,
            numResidualCentroids,
            kmeansIters,
            config.metric,
        )?;
        drop(residualSlices);

        // Compute stage-2 residuals: stage1_residual - nearest_residual_centroid
        let mut residualAssignments: Vec<u16> = try_alloc_vec(vectors.len())?;
        let mut residuals2: Vec<Vec<f32>> = try_alloc_vec(vectors.len())?;

        for r1 in &residuals1 {
            let rcIdx = findNearest(r1.as_slice(), &residualCentroids, config.metric);
            residualAssignments.push(rcIdx as u16);
            let mut r2 = vec![0.0f32; dimensions];
            vectorSubtract(r1, &residualCentroids[rcIdx], &mut r2);
            residuals2.push(r2);
        }

        // Drop residuals1 now that stage-2 is computed. Releases ~n*d*4 bytes.
        drop(residuals1);

        // Step 3: Train PQ codebooks on stage-2 residuals in parallel.
        let codebooks: Vec<Vec<Vec<f32>>> = {
            let residualRef = &residuals2;
            let mut results: Vec<Option<Vec<Vec<f32>>>> =
                (0..numSubvectors).map(|_| None).collect();

            std::thread::scope(|s| {
                let resultSlice = &mut results[..];
                for (sIdx, slot) in resultSlice.iter_mut().enumerate() {
                    let start = sIdx * subDim;
                    let end = start + subDim;
                    let cbSize = codebookSize.min(residualRef.len()).max(1);
                    let m = config.metric;
                    let kmeansItersLocal = kmeansIters;
                    s.spawn(move || {
                        let subSlices: Vec<&[f32]> =
                            residualRef.iter().map(|r| &r[start..end]).collect();
                        let cb =
                            kmeans(&subSlices, cbSize, kmeansItersLocal, m).unwrap_or_default();
                        let mut fullCb = cb;
                        if !fullCb.is_empty() {
                            let realLen = fullCb.len();
                            while fullCb.len() < codebookSize {
                                let srcIdx = fullCb.len() % realLen;
                                fullCb.push(fullCb[srcIdx].clone());
                            }
                        }
                        *slot = Some(fullCb);
                    });
                }
            });

            results.into_iter().map(|r| r.unwrap_or_default()).collect()
        };

        // Step 4: Encode stage-2 residuals into PQ codes and build inverted lists.
        let mut invertedLists: Vec<Vec<(VectorId, u32, u16, Vec<u8>)>> =
            try_alloc_filled(numCentroids, Vec::new())?;

        for centroidIdx in 0..numCentroids {
            for &vecIdx in &assignments[centroidIdx] {
                let codes = encodeResidual(&residuals2[vecIdx], &codebooks, subDim, config.metric);
                invertedLists[centroidIdx].push((
                    vectors[vecIdx].0,
                    vecIdx as u32,
                    residualAssignments[vecIdx],
                    codes,
                ));
            }
        }

        drop(residuals2);

        let nodeCount = vectors.len() as u64;
        let lockedLists: Vec<RwLock<Vec<(VectorId, u32, u16, Vec<u8>)>>> =
            invertedLists.into_iter().map(RwLock::new).collect();

        // Build flat vector arena for exact distance computation during search.
        let mut arenaVec: Vec<f32> = try_alloc_vec(vectors.len() * dimensions)?;
        let idToArenaIdx: scc::HashMap<VectorId, usize> = scc::HashMap::new();
        for (idx, &(vid, v)) in vectors.iter().enumerate() {
            arenaVec.extend_from_slice(v);
            let _ = idToArenaIdx.insert_sync(vid, idx);
        }
        let vectorArena = RwLock::new(arenaVec);

        Ok(Self {
            indexId,
            tableId,
            columnId,
            config,
            dimensions: dims,
            centroids,
            residualCentroids,
            invertedLists: lockedLists,
            codebooks,
            vectorArena,
            idToArenaIdx,
            nodeCount: AtomicU64::new(nodeCount),
            profile: Some(profile),
        })
    }

    /// Returns the DataProfile computed at build time, if available.
    pub fn profile(&self) -> Option<&DataProfile> {
        self.profile.as_ref()
    }

    /// Finds the index of the nearest centroid to the given query vector.
    fn findNearestCentroid(&self, query: &[f32]) -> usize {
        findNearest(query, &self.centroids, self.config.metric)
    }

    /// Encodes a residual vector into PQ codes (one byte per subvector).
    fn encodeVector(&self, residual: &[f32]) -> Vec<u8> {
        let subDim = self.dimensions as usize / self.config.numSubvectors as usize;
        encodeResidual(residual, &self.codebooks, subDim, self.config.metric)
    }

    /// Serializes the index to a file.
    ///
    /// Format: magic (6 bytes), config fields, centroids, codebooks, inverted lists.
    pub fn saveToFile(&self, path: &std::path::Path) -> Result<()> {
        use std::io::Write;
        let mut buf: Vec<u8> = Vec::new();

        // Magic.
        buf.extend_from_slice(MAGIC);

        // Config.
        buf.extend_from_slice(&self.dimensions.to_le_bytes());
        buf.extend_from_slice(&self.config.numCentroids.to_le_bytes());
        buf.extend_from_slice(&self.config.numSubvectors.to_le_bytes());
        buf.push(self.config.bitsPerCode);
        buf.extend_from_slice(&self.config.numProbes.to_le_bytes());
        buf.push(metricToByte(self.config.metric));

        // Number of centroids actually stored (may be clamped).
        let numCentroids = self.centroids.len() as u32;
        buf.extend_from_slice(&numCentroids.to_le_bytes());

        // Centroids: each is dimensions * 4 bytes.
        for c in &self.centroids {
            for &v in c {
                buf.extend_from_slice(&v.to_le_bytes());
            }
        }

        // Residual centroids (ARQ Stage 2).
        let numResidualCentroids = self.residualCentroids.len() as u32;
        buf.extend_from_slice(&numResidualCentroids.to_le_bytes());
        for rc in &self.residualCentroids {
            for &v in rc {
                buf.extend_from_slice(&v.to_le_bytes());
            }
        }

        // Codebooks.
        let numSub = self.codebooks.len() as u16;
        buf.extend_from_slice(&numSub.to_le_bytes());
        for cb in &self.codebooks {
            let cbSize = cb.len() as u32;
            buf.extend_from_slice(&cbSize.to_le_bytes());
            let subDim = if cb.is_empty() {
                0u16
            } else {
                cb[0].len() as u16
            };
            buf.extend_from_slice(&subDim.to_le_bytes());
            for entry in cb {
                for &v in entry {
                    buf.extend_from_slice(&v.to_le_bytes());
                }
            }
        }

        // Inverted lists (with residual centroid index).
        let listCount = self.invertedLists.len() as u32;
        buf.extend_from_slice(&listCount.to_le_bytes());
        for list in &self.invertedLists {
            let guard = list.read();
            let entryCount = guard.len() as u32;
            buf.extend_from_slice(&entryCount.to_le_bytes());
            for &(id, arenaIdx, rcIdx, ref codes) in guard.iter() {
                buf.extend_from_slice(&id.to_le_bytes());
                buf.extend_from_slice(&arenaIdx.to_le_bytes());
                buf.extend_from_slice(&rcIdx.to_le_bytes());
                let codeLen = codes.len() as u16;
                buf.extend_from_slice(&codeLen.to_le_bytes());
                buf.extend_from_slice(codes);
            }
        }

        // Node count.
        let count = self.nodeCount.load(Ordering::Relaxed);
        buf.extend_from_slice(&count.to_le_bytes());

        // Vector arena for exact distance search after load.
        let arenaGuard = self.vectorArena.read();
        let arenaLen = arenaGuard.len() as u64;
        buf.extend_from_slice(&arenaLen.to_le_bytes());
        for &v in arenaGuard.iter() {
            buf.extend_from_slice(&v.to_le_bytes());
        }

        // IdToArenaIdx map entries.
        let mut mapEntries: Vec<(VectorId, usize)> = Vec::new();
        self.idToArenaIdx.iter_sync(|&k, &v| {
            mapEntries.push((k, v));
            true
        });
        let mapLen = mapEntries.len() as u64;
        buf.extend_from_slice(&mapLen.to_le_bytes());
        for (vid, idx) in &mapEntries {
            buf.extend_from_slice(&vid.to_le_bytes());
            buf.extend_from_slice(&(*idx as u64).to_le_bytes());
        }

        let mut file = std::fs::File::create(path)?;
        file.write_all(&buf)?;
        file.sync_all()?;
        Ok(())
    }

    /// Deserializes an index from a file written by `saveToFile`.
    pub fn loadFromFile(
        path: &std::path::Path,
        indexId: u32,
        tableId: u32,
        columnId: u16,
    ) -> Result<Self> {
        let data = std::fs::read(path)?;
        let mut pos = 0usize;

        // Magic.
        if data.len() < 6 || &data[0..6] != MAGIC {
            return Err(ZyronError::InvalidParameter {
                name: "magic".to_string(),
                value: "invalid IVF-PQ file header".to_string(),
            });
        }
        pos += 6;

        // Helper closures for reading.
        let readU8 = |pos: &mut usize| -> Result<u8> {
            if *pos + 1 > data.len() {
                return Err(ZyronError::InvalidParameter {
                    name: "file".to_string(),
                    value: "unexpected end of file".to_string(),
                });
            }
            let v = data[*pos];
            *pos += 1;
            Ok(v)
        };
        let readU16 = |pos: &mut usize| -> Result<u16> {
            if *pos + 2 > data.len() {
                return Err(ZyronError::InvalidParameter {
                    name: "file".to_string(),
                    value: "unexpected end of file".to_string(),
                });
            }
            let v = u16::from_le_bytes([data[*pos], data[*pos + 1]]);
            *pos += 2;
            Ok(v)
        };
        let readU32 = |pos: &mut usize| -> Result<u32> {
            if *pos + 4 > data.len() {
                return Err(ZyronError::InvalidParameter {
                    name: "file".to_string(),
                    value: "unexpected end of file".to_string(),
                });
            }
            let v =
                u32::from_le_bytes([data[*pos], data[*pos + 1], data[*pos + 2], data[*pos + 3]]);
            *pos += 4;
            Ok(v)
        };
        let readU64 = |pos: &mut usize| -> Result<u64> {
            if *pos + 8 > data.len() {
                return Err(ZyronError::InvalidParameter {
                    name: "file".to_string(),
                    value: "unexpected end of file".to_string(),
                });
            }
            let v = u64::from_le_bytes([
                data[*pos],
                data[*pos + 1],
                data[*pos + 2],
                data[*pos + 3],
                data[*pos + 4],
                data[*pos + 5],
                data[*pos + 6],
                data[*pos + 7],
            ]);
            *pos += 8;
            Ok(v)
        };
        let readF32 = |pos: &mut usize| -> Result<f32> {
            if *pos + 4 > data.len() {
                return Err(ZyronError::InvalidParameter {
                    name: "file".to_string(),
                    value: "unexpected end of file".to_string(),
                });
            }
            let v =
                f32::from_le_bytes([data[*pos], data[*pos + 1], data[*pos + 2], data[*pos + 3]]);
            *pos += 4;
            Ok(v)
        };

        // Config.
        let dimensions = readU16(&mut pos)?;
        let numCentroidsConfig = readU32(&mut pos)?;
        let numSubvectors = readU16(&mut pos)?;
        let bitsPerCode = readU8(&mut pos)?;
        let numProbes = readU16(&mut pos)?;
        let metricByte = readU8(&mut pos)?;
        let metric = byteToMetric(metricByte)?;

        let config = IvfPqConfig {
            numCentroids: numCentroidsConfig,
            numSubvectors,
            bitsPerCode,
            numProbes,
            metric,
        };

        // Validate overall file size against sanity cap before trusting any
        // declared counts. Protects against corrupt/malicious file headers.
        validate_file_size(data.len() as u64)?;

        // Centroids.
        let numCentroids = readU32(&mut pos)? as usize;
        // Sanity cap: numCentroids tracks sqrt(N), so even at N=100B the
        // value is around 316K. 1M is a comfortable upper bound.
        if numCentroids > 1_000_000 {
            return Err(ZyronError::VectorIndexFileCorrupt {
                declared: numCentroids as u64,
            });
        }
        let mut centroids: Vec<Vec<f32>> = try_alloc_vec(numCentroids)?;
        for _ in 0..numCentroids {
            let mut c: Vec<f32> = try_alloc_vec(dimensions as usize)?;
            for _ in 0..dimensions {
                c.push(readF32(&mut pos)?);
            }
            centroids.push(c);
        }

        // Residual centroids (ARQ Stage 2).
        let numResidualCentroids = readU32(&mut pos)? as usize;
        if numResidualCentroids > 1_000_000 {
            return Err(ZyronError::VectorIndexFileCorrupt {
                declared: numResidualCentroids as u64,
            });
        }
        let mut residualCentroids: Vec<Vec<f32>> = try_alloc_vec(numResidualCentroids)?;
        for _ in 0..numResidualCentroids {
            let mut rc: Vec<f32> = try_alloc_vec(dimensions as usize)?;
            for _ in 0..dimensions {
                rc.push(readF32(&mut pos)?);
            }
            residualCentroids.push(rc);
        }

        // Codebooks.
        let numSub = readU16(&mut pos)? as usize;
        let mut codebooks: Vec<Vec<Vec<f32>>> = try_alloc_vec(numSub)?;
        for _ in 0..numSub {
            let cbSize = readU32(&mut pos)? as usize;
            let subDim = readU16(&mut pos)? as usize;
            // Codebook size is 2^bitsPerCode, typically 256, max 65536.
            if cbSize > 65536 || subDim > 1024 {
                return Err(ZyronError::VectorIndexFileCorrupt {
                    declared: cbSize as u64,
                });
            }
            let mut cb: Vec<Vec<f32>> = try_alloc_vec(cbSize)?;
            for _ in 0..cbSize {
                let mut entry: Vec<f32> = try_alloc_vec(subDim)?;
                for _ in 0..subDim {
                    entry.push(readF32(&mut pos)?);
                }
                cb.push(entry);
            }
            codebooks.push(cb);
        }

        // Inverted lists (with residual centroid index).
        let listCount = readU32(&mut pos)? as usize;
        if listCount > 1_000_000 {
            return Err(ZyronError::VectorIndexFileCorrupt {
                declared: listCount as u64,
            });
        }
        let mut invertedLists: Vec<RwLock<Vec<(VectorId, u32, u16, Vec<u8>)>>> =
            try_alloc_vec(listCount)?;
        for _ in 0..listCount {
            let entryCount = readU32(&mut pos)? as usize;
            // Sanity cap: entries per partition shouldn't exceed total vectors.
            // At 1B vectors that's still under 1B. Use 100M as a hard cap.
            if entryCount > 100_000_000 {
                return Err(ZyronError::VectorIndexFileCorrupt {
                    declared: entryCount as u64,
                });
            }
            let mut entries: Vec<(VectorId, u32, u16, Vec<u8>)> = try_alloc_vec(entryCount)?;
            for _ in 0..entryCount {
                let id = readU64(&mut pos)?;
                let arenaIdx = readU32(&mut pos)?;
                let rcIdx = readU16(&mut pos)?;
                let codeLen = readU16(&mut pos)? as usize;
                if pos + codeLen > data.len() {
                    return Err(ZyronError::InvalidParameter {
                        name: "file".to_string(),
                        value: "unexpected end of file reading PQ codes".to_string(),
                    });
                }
                let codes = data[pos..pos + codeLen].to_vec();
                pos += codeLen;
                entries.push((id, arenaIdx, rcIdx, codes));
            }
            invertedLists.push(RwLock::new(entries));
        }

        // Node count.
        let nodeCount = if pos + 8 <= data.len() {
            readU64(&mut pos)?
        } else {
            let mut total = 0u64;
            for list in &invertedLists {
                total += list.read().len() as u64;
            }
            total
        };

        // Load vector arena and id map if present (format v1.1+).
        // Older files without arena data will fall back to PQ distances.
        let mut arenaVec = Vec::new();
        let idToArenaIdx = scc::HashMap::new();
        if pos + 8 <= data.len() {
            let arenaLen = readU64(&mut pos)? as usize;
            if arenaLen > 0 && pos + arenaLen * 4 <= data.len() {
                arenaVec.reserve(arenaLen);
                for _ in 0..arenaLen {
                    arenaVec.push(readF32(&mut pos)?);
                }
            }
            if pos + 8 <= data.len() {
                let mapLen = readU64(&mut pos)? as usize;
                for _ in 0..mapLen {
                    if pos + 16 > data.len() {
                        break;
                    }
                    let vid = readU64(&mut pos)?;
                    let idx = readU64(&mut pos)? as usize;
                    let _ = idToArenaIdx.insert_sync(vid, idx);
                }
            }
        }
        let vectorArena = RwLock::new(arenaVec);

        Ok(Self {
            indexId,
            tableId,
            columnId,
            config,
            dimensions,
            centroids,
            residualCentroids,
            invertedLists,
            codebooks,
            vectorArena,
            idToArenaIdx,
            nodeCount: AtomicU64::new(nodeCount),
            profile: None,
        })
    }
}

// ---------------------------------------------------------------------------
// VectorSearch trait implementation
// ---------------------------------------------------------------------------

impl VectorSearch for IvfPqIndex {
    /// Approximate nearest neighbor search using asymmetric distance computation.
    ///
    /// The efSearch parameter is reinterpreted as the number of partitions to probe.
    /// For each probed partition, a lookup table of precomputed subvector distances
    /// is used to accumulate approximate distances without decompressing vectors.
    fn search(&self, query: &[f32], k: usize, efSearch: u16) -> Result<Vec<(VectorId, f32)>> {
        if self.centroids.is_empty() || k == 0 {
            return Ok(Vec::new());
        }
        if query.len() != self.dimensions as usize {
            return Err(ZyronError::InvalidParameter {
                name: "query dimensions".to_string(),
                value: format!("expected {}, got {}", self.dimensions, query.len()),
            });
        }

        // efSearch=0 means "use the config's auto-tuned numProbes".
        let effectiveProbes = if efSearch == 0 {
            self.config.numProbes
        } else {
            efSearch
        };
        let numProbes = (effectiveProbes as usize).max(1).min(self.centroids.len());
        let numSub = self.config.numSubvectors as usize;
        let subDim = self.dimensions as usize / numSub;
        let codebookSize = 1usize << self.config.bitsPerCode;

        // Find the nearest numProbes centroids using partial sort.
        // O(n) via select_nth_unstable instead of O(n log n) full sort.
        let mut centroidDists: Vec<(usize, f32)> = self
            .centroids
            .iter()
            .enumerate()
            .map(|(i, c)| (i, computeDistance(self.config.metric, query, c)))
            .collect();
        if numProbes < centroidDists.len() {
            centroidDists.select_nth_unstable_by(numProbes, |a, b| {
                a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal)
            });
            centroidDists.truncate(numProbes);
        }

        let dims = self.dimensions as usize;
        let arenaGuard = self.vectorArena.read();
        let useExactSearch = !arenaGuard.is_empty();
        let metric = self.config.metric;

        // Two-phase search: PQ pre-filter to find top candidates, then exact
        // rerank on original vectors. rerankK controls how many PQ candidates get
        // exact-distance reranking. Set high enough that PQ approximation errors
        // (which are significant on high-dimensional data) don't filter out true
        // nearest neighbors. At least k*50 or 10% of probed vectors.
        let estimatedProbed = numProbes
            * (self.nodeCount.load(Ordering::Relaxed) as usize / self.centroids.len().max(1));
        let rerankK = (k * 50)
            .max(estimatedProbed / 10)
            .max(500)
            .min(estimatedProbed);

        let candidates: Vec<(VectorId, f32)> = if useExactSearch && !self.codebooks.is_empty() {
            // Phase 1: PQ pre-filter across all probed partitions.
            // Pre-allocate all buffers ONCE outside the centroid loop to avoid
            // reallocation overhead. Flat ADC table layout [numSub * codebookSize]
            // is more cache-friendly than Vec<Vec<f32>>.
            let mut pqCandidates: Vec<(VectorId, u32, f32)> = Vec::new();
            let mut queryResidual = vec![0.0f32; dims];
            let mut adcTable = vec![0.0f32; numSub * codebookSize];

            for &(centroidIdx, _) in &centroidDists {
                let guard = self.invertedLists[centroidIdx].read();

                // Compute residual for this centroid (reuses buffer).
                vectorSubtract(query, &self.centroids[centroidIdx], &mut queryResidual);

                // Build ADC lookup table (reuses flat buffer).
                for s in 0..numSub {
                    let qStart = s * subDim;
                    let qEnd = qStart + subDim;
                    let qSub = &queryResidual[qStart..qEnd];
                    let tableOffset = s * codebookSize;
                    let cbLen = self.codebooks[s].len();
                    for c in 0..cbLen {
                        adcTable[tableOffset + c] =
                            subvectorDistance(qSub, &self.codebooks[s][c], metric);
                    }
                }

                for &(id, arenaIdx, _, ref codes) in guard.iter() {
                    if codes.len() < numSub {
                        continue;
                    }
                    let mut dist = 0.0f32;
                    for s in 0..numSub {
                        dist += adcTable[s * codebookSize + codes[s] as usize];
                    }
                    pqCandidates.push((id, arenaIdx, dist));
                }
            }

            // Select top rerankK by PQ distance using partial sort.
            if rerankK < pqCandidates.len() {
                pqCandidates.select_nth_unstable_by(rerankK, |a, b| {
                    a.2.partial_cmp(&b.2).unwrap_or(std::cmp::Ordering::Equal)
                });
                pqCandidates.truncate(rerankK);
            }

            // Phase 2: Exact rerank on original vectors from arena.
            let mut exactCandidates: Vec<(VectorId, f32)> = Vec::with_capacity(pqCandidates.len());
            for (id, arenaIdx, _) in &pqCandidates {
                let offset = *arenaIdx as usize * dims;
                let end = offset + dims;
                if end <= arenaGuard.len() {
                    let originalVec = &arenaGuard[offset..end];
                    let dist = computeDistance(metric, query, originalVec);
                    exactCandidates.push((*id, dist));
                }
            }
            exactCandidates
        } else if useExactSearch {
            // No codebooks available (empty index or very small). Full exact scan.
            let mut candidates = Vec::new();
            for &(centroidIdx, _) in &centroidDists {
                let guard = self.invertedLists[centroidIdx].read();
                for &(id, arenaIdx, _, _) in guard.iter() {
                    let offset = arenaIdx as usize * dims;
                    let end = offset + dims;
                    if end <= arenaGuard.len() {
                        let originalVec = &arenaGuard[offset..end];
                        let dist = computeDistance(metric, query, originalVec);
                        candidates.push((id, dist));
                    }
                }
            }
            candidates
        } else {
            // PQ approximate distance fallback (loaded indexes without arena).
            let mut candidates = Vec::new();
            for &(centroidIdx, _) in &centroidDists {
                let guard = self.invertedLists[centroidIdx].read();
                let mut queryResidual = vec![0.0f32; dims];
                vectorSubtract(query, &self.centroids[centroidIdx], &mut queryResidual);

                let mut adcTable = vec![vec![0.0f32; codebookSize]; numSub];
                for s in 0..numSub {
                    let qStart = s * subDim;
                    let qEnd = qStart + subDim;
                    let qSub = &queryResidual[qStart..qEnd];
                    for c in 0..self.codebooks[s].len() {
                        adcTable[s][c] = subvectorDistance(qSub, &self.codebooks[s][c], metric);
                    }
                }

                for &(id, _, _, ref codes) in guard.iter() {
                    debug_assert_eq!(
                        codes.len(),
                        numSub,
                        "PQ codes length {} != expected {} for VectorId {}",
                        codes.len(),
                        numSub,
                        id
                    );
                    if codes.len() < numSub {
                        continue;
                    }
                    let mut dist = 0.0f32;
                    for s in 0..numSub {
                        dist += adcTable[s][codes[s] as usize];
                    }
                    candidates.push((id, dist));
                }
            }
            candidates
        };

        // Select top-k by smallest distance using partial sort.
        let mut candidates = candidates;
        if k < candidates.len() {
            candidates.select_nth_unstable_by(k, |a, b| {
                a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal)
            });
            candidates.truncate(k);
            candidates.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        } else {
            candidates.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        }
        Ok(candidates)
    }

    /// Inserts a single vector into the index.
    ///
    /// Finds the nearest centroid, computes the residual, encodes it, and appends
    /// to the corresponding inverted list.
    fn insert(&self, id: VectorId, vector: &[f32]) -> Result<()> {
        if vector.len() != self.dimensions as usize {
            return Err(ZyronError::InvalidParameter {
                name: "vector dimensions".to_string(),
                value: format!("expected {}, got {}", self.dimensions, vector.len()),
            });
        }
        if self.centroids.is_empty() {
            return Err(ZyronError::InvalidParameter {
                name: "index".to_string(),
                value: "cannot insert into an empty index with no centroids".to_string(),
            });
        }

        let dims = self.dimensions as usize;
        let centroidIdx = self.findNearestCentroid(vector);

        // Stage 1 residual
        let mut residual1 = vec![0.0f32; dims];
        vectorSubtract(vector, &self.centroids[centroidIdx], &mut residual1);

        // Stage 2: find nearest residual centroid
        let rcIdx = if self.residualCentroids.is_empty() {
            0u16
        } else {
            findNearest(&residual1, &self.residualCentroids, self.config.metric) as u16
        };

        // Stage 2 residual
        let mut residual2 = vec![0.0f32; dims];
        if !self.residualCentroids.is_empty() {
            vectorSubtract(
                &residual1,
                &self.residualCentroids[rcIdx as usize],
                &mut residual2,
            );
        } else {
            residual2 = residual1;
        }

        let codes = self.encodeVector(&residual2);

        // Update vector arena for exact distance search
        let arenaIdx = {
            let mut arenaGuard = self.vectorArena.write();
            let idx = arenaGuard.len() / dims;
            arenaGuard.extend_from_slice(vector);
            let _ = self.idToArenaIdx.insert_sync(id, idx);
            idx as u32
        };

        let mut guard = self.invertedLists[centroidIdx].write();
        guard.push((id, arenaIdx, rcIdx, codes));
        self.nodeCount.fetch_add(1, Ordering::Relaxed);
        Ok(())
    }

    /// Removes a vector by scanning all inverted lists.
    ///
    /// Returns Ok(()) regardless of whether the vector was found.
    fn delete(&self, id: VectorId) -> Result<()> {
        for list in &self.invertedLists {
            let mut guard = list.write();
            let before = guard.len();
            guard.retain(|(vid, _, _, _)| *vid != id);
            let removed = before - guard.len();
            if removed > 0 {
                self.nodeCount.fetch_sub(removed as u64, Ordering::Relaxed);
                return Ok(());
            }
        }
        Ok(())
    }

    fn dimensions(&self) -> u16 {
        self.dimensions
    }

    fn metric(&self) -> DistanceMetric {
        self.config.metric
    }

    fn len(&self) -> usize {
        self.nodeCount.load(Ordering::Relaxed) as usize
    }
}

// ---------------------------------------------------------------------------
// Private helpers
// ---------------------------------------------------------------------------

/// K-means clustering using Lloyd's algorithm.
///
/// Initializes centroids via k-means++ (first centroid random, subsequent selected
/// proportional to squared distance from the nearest existing centroid).
fn kmeans(
    vectors: &[&[f32]],
    k: usize,
    iterations: usize,
    metric: DistanceMetric,
) -> Result<Vec<Vec<f32>>> {
    if vectors.is_empty() || k == 0 {
        return Ok(Vec::new());
    }
    let k = k.min(vectors.len());
    let dim = vectors[0].len();

    // Seed the RNG from the vector count and dimension.
    let mut rng = (vectors.len() as u64).wrapping_mul(2654435761) | 1;

    // K-means++ initialization.
    let firstIdx = (xorshift64(&mut rng) as usize) % vectors.len();
    let mut centroids: Vec<Vec<f32>> = vec![vectors[firstIdx].to_vec()];

    let mut minDists: Vec<f32> = try_alloc_filled(vectors.len(), f32::MAX)?;
    for _ in 1..k {
        // Update minimum distance to nearest centroid for each vector.
        let lastCentroid = centroids
            .last()
            .ok_or_else(|| ZyronError::InvalidParameter {
                name: "centroids".to_string(),
                value: "empty centroid list during initialization".to_string(),
            })?;
        for (i, v) in vectors.iter().enumerate() {
            let d = computeDistance(metric, v, lastCentroid);
            if d < minDists[i] {
                minDists[i] = d;
            }
        }

        // Weighted random selection proportional to squared distance.
        let totalWeight: f64 = minDists.iter().map(|&d| (d as f64) * (d as f64)).sum();
        if totalWeight <= 0.0 {
            // All distances are zero, pick randomly.
            let idx = (xorshift64(&mut rng) as usize) % vectors.len();
            centroids.push(vectors[idx].to_vec());
            continue;
        }
        let threshold = (xorshift64(&mut rng) as f64 / u64::MAX as f64) * totalWeight;
        let mut cumulative = 0.0f64;
        let mut chosenIdx = 0usize;
        for (i, &d) in minDists.iter().enumerate() {
            cumulative += (d as f64) * (d as f64);
            if cumulative >= threshold {
                chosenIdx = i;
                break;
            }
        }
        centroids.push(vectors[chosenIdx].to_vec());
    }

    // Lloyd's iterations with convergence check, parallelized assignment.
    let mut assignments: Vec<usize> = try_alloc_default(vectors.len())?;
    let convergenceThreshold = 1e-4f32;
    let nThreads = threadCount().min(vectors.len());
    let chunkSize = (vectors.len() + nThreads - 1) / nThreads;

    for _iter in 0..iterations {
        // Parallel assignment step: split vectors across threads.
        let assignSlice = &mut assignments[..];
        std::thread::scope(|s| {
            let centroidRef = &centroids;
            for (chunkIdx, chunk) in assignSlice.chunks_mut(chunkSize).enumerate() {
                let offset = chunkIdx * chunkSize;
                let vectorSlice = &vectors[offset..offset + chunk.len()];
                s.spawn(move || {
                    for (i, v) in vectorSlice.iter().enumerate() {
                        chunk[i] = findNearest(v, centroidRef, metric);
                    }
                });
            }
        });

        // Update step: recompute centroids as the mean of assigned vectors.
        // Parallelized: each thread accumulates partial sums into its own
        // local buffers, then a final merge pass combines them.
        let mut newCentroids = vec![vec![0.0f32; dim]; k];
        let mut counts = vec![0u64; k];

        if nThreads > 1 && vectors.len() >= 10000 {
            // Parallel accumulation: each thread produces its own (centroids, counts).
            let partials: Vec<(Vec<Vec<f32>>, Vec<u64>)> = std::thread::scope(|s| {
                let assignRef = &assignments;
                let mut handles = Vec::with_capacity(nThreads);
                for (chunkIdx, chunk) in vectors.chunks(chunkSize).enumerate() {
                    let offset = chunkIdx * chunkSize;
                    handles.push(s.spawn(move || {
                        let mut localCentroids = vec![vec![0.0f32; dim]; k];
                        let mut localCounts = vec![0u64; k];
                        for (i, v) in chunk.iter().enumerate() {
                            let c = assignRef[offset + i];
                            vectorAddInplace(&mut localCentroids[c], v);
                            localCounts[c] += 1;
                        }
                        (localCentroids, localCounts)
                    }));
                }
                handles
                    .into_iter()
                    .map(|h| h.join().expect("kmeans thread panicked"))
                    .collect()
            });

            // Merge partial sums.
            for (localCentroids, localCounts) in partials {
                for c in 0..k {
                    if localCounts[c] > 0 {
                        vectorAddInplace(&mut newCentroids[c], &localCentroids[c]);
                        counts[c] += localCounts[c];
                    }
                }
            }
        } else {
            for (i, v) in vectors.iter().enumerate() {
                let c = assignments[i];
                vectorAddInplace(&mut newCentroids[c], v);
                counts[c] += 1;
            }
        }

        // Check if we have any empty centroids. If so, precompute the globally
        // farthest-from-its-assigned-centroid vectors ONCE, then use them to
        // seed empty centroids. Old code did O(n) scan per empty centroid.
        let numEmpty = counts.iter().filter(|&&c| c == 0).count();
        let distFn = resolveDistFn(metric);
        let mut replacements: Vec<usize> = Vec::new();
        if numEmpty > 0 {
            // Collect top-numEmpty farthest vectors in one pass.
            let mut farthest: Vec<(usize, f32)> = Vec::with_capacity(numEmpty);
            for (i, v) in vectors.iter().enumerate() {
                let d = distWithFn(distFn, v, &centroids[assignments[i]]);
                if farthest.len() < numEmpty {
                    farthest.push((i, d));
                    if farthest.len() == numEmpty {
                        // Sort ascending so smallest is at front.
                        farthest.sort_by(|a, b| {
                            a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal)
                        });
                    }
                } else if d > farthest[0].1 {
                    farthest[0] = (i, d);
                    farthest
                        .sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
                }
            }
            replacements = farthest.into_iter().map(|(idx, _)| idx).collect();
        }

        let mut replacementIdx = 0;
        for c in 0..k {
            if counts[c] > 0 {
                vectorScaleInplace(&mut newCentroids[c], 1.0 / counts[c] as f32);
            } else {
                if replacementIdx < replacements.len() {
                    newCentroids[c] = vectors[replacements[replacementIdx]].to_vec();
                    replacementIdx += 1;
                }
            }
        }

        // Convergence check
        let mut maxShift = 0.0f32;
        for c in 0..k {
            let d = computeDistance(metric, &centroids[c], &newCentroids[c]);
            if d > maxShift {
                maxShift = d;
            }
        }

        centroids = newCentroids;

        if maxShift < convergenceThreshold {
            break;
        }
    }

    Ok(centroids)
}

/// Finds the index of the nearest centroid to the given vector.
/// Uses pre-resolved DistFn pointer to eliminate per-call dispatch overhead
/// in the hot loop (k-means assignment step).
fn findNearest(vector: &[f32], centroids: &[Vec<f32>], metric: DistanceMetric) -> usize {
    use super::distance::{distWithFn, resolveDistFn};
    let f = resolveDistFn(metric);
    let mut bestIdx = 0usize;
    let mut bestDist = f32::MAX;
    for (i, c) in centroids.iter().enumerate() {
        let d = distWithFn(f, vector, c);
        if d < bestDist {
            bestDist = d;
            bestIdx = i;
        }
    }
    bestIdx
}

/// Encodes a residual vector into PQ codes by finding the nearest codebook entry
/// per subvector segment.
fn encodeResidual(
    residual: &[f32],
    codebooks: &[Vec<Vec<f32>>],
    subDim: usize,
    metric: DistanceMetric,
) -> Vec<u8> {
    let numSub = codebooks.len();
    let mut codes = Vec::with_capacity(numSub);
    for s in 0..numSub {
        let start = s * subDim;
        let end = start + subDim;
        let sub = &residual[start..end];

        let mut bestCode = 0u8;
        let mut bestDist = f32::MAX;
        for (c, entry) in codebooks[s].iter().enumerate() {
            let d = subvectorDistance(sub, entry, metric);
            if d < bestDist {
                bestDist = d;
                bestCode = c as u8;
            }
        }
        codes.push(bestCode);
    }
    codes
}

/// Converts a DistanceMetric to a byte for serialization.
fn metricToByte(metric: DistanceMetric) -> u8 {
    match metric {
        DistanceMetric::Cosine => 0,
        DistanceMetric::Euclidean => 1,
        DistanceMetric::DotProduct => 2,
        DistanceMetric::Manhattan => 3,
    }
}

/// Converts a byte back to a DistanceMetric.
fn byteToMetric(b: u8) -> Result<DistanceMetric> {
    match b {
        0 => Ok(DistanceMetric::Cosine),
        1 => Ok(DistanceMetric::Euclidean),
        2 => Ok(DistanceMetric::DotProduct),
        3 => Ok(DistanceMetric::Manhattan),
        _ => Err(ZyronError::InvalidParameter {
            name: "metric".to_string(),
            value: format!("unknown metric byte {}", b),
        }),
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// Generates deterministic test vectors using xorshift64.
    fn generateTestVectors(count: usize, dims: usize) -> Vec<(VectorId, Vec<f32>)> {
        let mut rng = 42u64;
        let mut vectors = Vec::with_capacity(count);
        for i in 0..count {
            let mut v = Vec::with_capacity(dims);
            for _ in 0..dims {
                let raw = xorshift64(&mut rng);
                v.push((raw as f32 / u64::MAX as f32) * 2.0 - 1.0);
            }
            vectors.push((i as VectorId, v));
        }
        vectors
    }

    #[test]
    fn buildAndSearch1000Vectors() {
        let vectors = generateTestVectors(1000, 32);
        let refs: Vec<(VectorId, &[f32])> =
            vectors.iter().map(|(id, v)| (*id, v.as_slice())).collect();

        let config = IvfPqConfig {
            numCentroids: 16,
            numSubvectors: 8,
            bitsPerCode: 8,
            numProbes: 4,
            metric: DistanceMetric::Euclidean,
        };

        let index = IvfPqIndex::build(&refs, 1, 100, 0, config).expect("build should succeed");
        assert_eq!(index.len(), 1000);
        assert_eq!(index.dimensions(), 32);

        // Search for a vector that is in the index. It should appear in results.
        let query = vectors[42].1.as_slice();
        let results = index.search(query, 10, 4).expect("search should succeed");
        assert!(!results.is_empty());
        assert!(results.len() <= 10);

        // Results should be sorted by ascending distance.
        for i in 1..results.len() {
            assert!(results[i].1 >= results[i - 1].1);
        }
    }

    #[test]
    fn insertAfterBuild() {
        let vectors = generateTestVectors(100, 16);
        let refs: Vec<(VectorId, &[f32])> =
            vectors.iter().map(|(id, v)| (*id, v.as_slice())).collect();

        let config = IvfPqConfig {
            numCentroids: 4,
            numSubvectors: 4,
            bitsPerCode: 4,
            numProbes: 2,
            metric: DistanceMetric::Euclidean,
        };

        let index = IvfPqIndex::build(&refs, 1, 100, 0, config).expect("build should succeed");
        assert_eq!(index.len(), 100);

        // Insert a new vector.
        let newVec = vec![0.5f32; 16];
        index.insert(9999, &newVec).expect("insert should succeed");
        assert_eq!(index.len(), 101);

        // The new vector should appear in search results when queried.
        let results = index.search(&newVec, 5, 4).expect("search should succeed");
        let ids: Vec<VectorId> = results.iter().map(|(id, _)| *id).collect();
        assert!(ids.contains(&9999));
    }

    #[test]
    fn dimensionMismatchError() {
        let vectors = generateTestVectors(50, 16);
        let refs: Vec<(VectorId, &[f32])> =
            vectors.iter().map(|(id, v)| (*id, v.as_slice())).collect();

        let config = IvfPqConfig {
            numCentroids: 4,
            numSubvectors: 4,
            bitsPerCode: 8,
            numProbes: 2,
            metric: DistanceMetric::Euclidean,
        };

        let index = IvfPqIndex::build(&refs, 1, 100, 0, config).expect("build should succeed");

        // Search with wrong dimensions.
        let badQuery = vec![1.0f32; 8];
        let result = index.search(&badQuery, 5, 2);
        assert!(result.is_err());

        // Insert with wrong dimensions.
        let badVec = vec![1.0f32; 32];
        let result = index.insert(9999, &badVec);
        assert!(result.is_err());
    }

    #[test]
    fn emptyBuild() {
        let config = IvfPqConfig {
            numCentroids: 16,
            numSubvectors: 4,
            bitsPerCode: 8,
            numProbes: 2,
            metric: DistanceMetric::Euclidean,
        };

        let index = IvfPqIndex::build(&[], 1, 100, 0, config).expect("empty build should succeed");
        assert_eq!(index.len(), 0);
        assert_eq!(index.dimensions(), 0);

        let results = index
            .search(&[1.0, 2.0, 3.0], 5, 2)
            .expect("search on empty index returns empty");
        assert!(results.is_empty());
    }

    #[test]
    fn pqEncodingLength() {
        let vectors = generateTestVectors(100, 32);
        let refs: Vec<(VectorId, &[f32])> =
            vectors.iter().map(|(id, v)| (*id, v.as_slice())).collect();

        let config = IvfPqConfig {
            numCentroids: 8,
            numSubvectors: 8,
            bitsPerCode: 8,
            numProbes: 2,
            metric: DistanceMetric::Euclidean,
        };

        let index = IvfPqIndex::build(&refs, 1, 100, 0, config).expect("build should succeed");

        // Verify PQ codes have the correct length (one byte per subvector).
        for list in &index.invertedLists {
            let guard = list.read();
            for (_, _, _, codes) in guard.iter() {
                assert_eq!(codes.len(), 8, "PQ codes should have numSubvectors entries");
            }
        }
    }

    #[test]
    fn saveAndLoadRoundTrip() {
        let vectors = generateTestVectors(200, 16);
        let refs: Vec<(VectorId, &[f32])> =
            vectors.iter().map(|(id, v)| (*id, v.as_slice())).collect();

        let config = IvfPqConfig {
            numCentroids: 8,
            numSubvectors: 4,
            bitsPerCode: 8,
            numProbes: 2,
            metric: DistanceMetric::Euclidean,
        };

        let original = IvfPqIndex::build(&refs, 5, 200, 3, config).expect("build should succeed");

        let dir = std::env::temp_dir().join("zyron_ivfpq_test");
        let _ = std::fs::create_dir_all(&dir);
        let filePath = dir.join("test_index.ivfpq");

        original.saveToFile(&filePath).expect("save should succeed");

        let loaded = IvfPqIndex::loadFromFile(&filePath, 5, 200, 3).expect("load should succeed");

        assert_eq!(loaded.dimensions(), original.dimensions());
        assert_eq!(loaded.len(), original.len());
        assert_eq!(loaded.metric(), original.metric());
        assert_eq!(loaded.centroids.len(), original.centroids.len());
        assert_eq!(loaded.codebooks.len(), original.codebooks.len());
        assert_eq!(loaded.invertedLists.len(), original.invertedLists.len());

        // Loaded index uses PQ fallback (no arena), so search results may differ
        // from the original which uses exact distances. Verify both return results.
        let query = vectors[10].1.as_slice();
        let origResults = original.search(query, 5, 4).expect("search original");
        let loadResults = loaded.search(query, 5, 4).expect("search loaded");
        assert!(
            !origResults.is_empty(),
            "original search should return results"
        );
        assert!(
            !loadResults.is_empty(),
            "loaded search should return results"
        );

        // Cleanup.
        let _ = std::fs::remove_file(&filePath);
        let _ = std::fs::remove_dir(&dir);
    }

    #[test]
    fn deleteRemovesVector() {
        let vectors = generateTestVectors(50, 16);
        let refs: Vec<(VectorId, &[f32])> =
            vectors.iter().map(|(id, v)| (*id, v.as_slice())).collect();

        let config = IvfPqConfig {
            numCentroids: 4,
            numSubvectors: 4,
            bitsPerCode: 8,
            numProbes: 4,
            metric: DistanceMetric::Euclidean,
        };

        let index = IvfPqIndex::build(&refs, 1, 100, 0, config).expect("build should succeed");
        assert_eq!(index.len(), 50);

        index.delete(10).expect("delete should succeed");
        assert_eq!(index.len(), 49);

        // Verify the vector is no longer in any inverted list.
        for list in &index.invertedLists {
            let guard = list.read();
            for (id, _, _, _) in guard.iter() {
                assert_ne!(*id, 10);
            }
        }
    }

    #[test]
    fn centroidsClampedToVectorCount() {
        // Request 256 centroids for only 10 vectors. Should clamp to 10.
        let vectors = generateTestVectors(10, 8);
        let refs: Vec<(VectorId, &[f32])> =
            vectors.iter().map(|(id, v)| (*id, v.as_slice())).collect();

        let config = IvfPqConfig {
            numCentroids: 256,
            numSubvectors: 2,
            bitsPerCode: 4,
            numProbes: 2,
            metric: DistanceMetric::Euclidean,
        };

        let index = IvfPqIndex::build(&refs, 1, 100, 0, config).expect("build should succeed");
        assert_eq!(index.len(), 10);
        assert!(index.centroids.len() <= 10);
    }

    #[test]
    fn indivisibleSubvectorsError() {
        let vectors = generateTestVectors(20, 10);
        let refs: Vec<(VectorId, &[f32])> =
            vectors.iter().map(|(id, v)| (*id, v.as_slice())).collect();

        // 10 dimensions with 3 subvectors does not divide evenly.
        let config = IvfPqConfig {
            numCentroids: 4,
            numSubvectors: 3,
            bitsPerCode: 8,
            numProbes: 2,
            metric: DistanceMetric::Euclidean,
        };

        let result = IvfPqIndex::build(&refs, 1, 100, 0, config);
        assert!(result.is_err());
    }
}
