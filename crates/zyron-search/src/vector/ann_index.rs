//! HNSW (Hierarchical Navigable Small World) approximate nearest neighbor index.
//!
//! Build path is completely lock-free: operates on raw Vec directly with zero
//! synchronization overhead. Uses a generation-based visited marker to avoid
//! per-search allocations. After construction, the index is wrapped in RwLock
//! for concurrent read access (read locks are free when no writer).

use std::cell::RefCell;
use std::cmp::Reverse;
use std::collections::BinaryHeap;
use std::io::{Read, Write};
use std::sync::atomic::{AtomicBool, AtomicU16, AtomicU64, Ordering};

use parking_lot::RwLock;
use zyron_common::{Result, ZyronError};

// Thread-local reusable visited buffer, generation counter, and last-used
// indexId. Eliminates per-query allocation of vec![0u32; N] which is 4MB at
// 1M nodes. Tracks indexId so querying a different index on the same thread
// properly resets the generation counter instead of carrying stale marks.
thread_local! {
    static SEARCH_VISITED: RefCell<(Vec<u32>, u32, u32)> =
        RefCell::new((Vec::new(), 0, u32::MAX));
}

use super::distance::{
    computeDistance, computeQuantizationBounds, euclideanQuantized, quantizeArena, quantizeVector,
};
use super::memory::{try_alloc_default, try_alloc_vec, validate_file_size};
use super::profile::DataProfile;
use super::types::{DistanceMetric, HnswConfig, VectorId, VectorSearch};

// ---------------------------------------------------------------------------
// Distance-ordered entry for BinaryHeap (f32 does not implement Ord)
// ---------------------------------------------------------------------------

#[derive(Clone, Copy)]
struct DistEntry {
    dist: f32,
    idx: usize,
}

impl PartialEq for DistEntry {
    fn eq(&self, other: &Self) -> bool {
        self.dist.total_cmp(&other.dist) == std::cmp::Ordering::Equal && self.idx == other.idx
    }
}

impl Eq for DistEntry {}

impl PartialOrd for DistEntry {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for DistEntry {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.dist.total_cmp(&other.dist)
    }
}

// ---------------------------------------------------------------------------
// Xorshift64 RNG (no external rand crate)
// ---------------------------------------------------------------------------

#[inline]
fn xorshift64(state: &mut u64) -> u64 {
    let mut x = *state;
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    *state = x;
    x
}

/// Generates a random level using the HNSW probability distribution.
/// Takes a mutable RNG state directly (no atomics needed during build).
#[inline]
fn randomLevelFrom(rng: &mut u64, m: u16) -> usize {
    let _ = xorshift64(rng);
    let r = xorshift64(rng);
    let uniform = ((r & 0x000F_FFFF_FFFF_FFFF) as f64 + 1.0) / ((1u64 << 52) as f64);
    let ml = (m.max(2) as f64).ln();
    (-uniform.ln() / ml).floor() as usize
}

// ---------------------------------------------------------------------------
// Free-function search layer (no &self, operates on slices directly)
// ---------------------------------------------------------------------------

/// Returns the vector slice for a node from the flat arena.
#[inline]
fn arenaSlice(arena: &[f32], dims: usize, nodeIdx: usize) -> &[f32] {
    let offset = nodeIdx * dims;
    &arena[offset..offset + dims]
}

/// Reusable search state to avoid per-search allocations during build.
/// Heaps and output buffer are allocated once and cleared between searches.
struct SearchState {
    candidates: BinaryHeap<Reverse<DistEntry>>,
    results: BinaryHeap<DistEntry>,
    output: Vec<(usize, f32)>,
}

impl SearchState {
    fn new(capacity: usize) -> Self {
        Self {
            candidates: BinaryHeap::with_capacity(capacity),
            results: BinaryHeap::with_capacity(capacity),
            output: Vec::with_capacity(capacity),
        }
    }

    fn clear(&mut self) {
        self.candidates.clear();
        self.results.clear();
        self.output.clear();
    }
}

/// Beam search at a single layer using reusable search state (zero allocation).
/// Results are written to state.output sorted by distance ascending.
/// Takes a resolved DistFn to avoid per-call dispatch overhead.
fn searchLayerReuse(
    metric: DistanceMetric,
    dims: usize,
    query: &[f32],
    entryIdx: usize,
    ef: usize,
    layer: usize,
    nodes: &[HnswNode],
    arena: &[f32],
    visitedGen: &mut Vec<u32>,
    currentGen: u32,
    state: &mut SearchState,
) {
    state.clear();
    let entryDist = computeDistance(metric, query, arenaSlice(arena, dims, entryIdx));

    state.candidates.push(Reverse(DistEntry {
        dist: entryDist,
        idx: entryIdx,
    }));
    state.results.push(DistEntry {
        dist: entryDist,
        idx: entryIdx,
    });
    visitedGen[entryIdx] = currentGen;

    while let Some(Reverse(current)) = state.candidates.pop() {
        if state.results.len() >= ef {
            let farthestResult = state
                .results
                .peek()
                .map(|e| e.dist)
                .unwrap_or(f32::INFINITY);
            if current.dist > farthestResult {
                break;
            }
        }

        let neighborIndices = if layer < nodes[current.idx].connections.len() {
            &nodes[current.idx].connections[layer]
        } else {
            continue;
        };

        // Prefetch-aware neighbor loop: look ahead to the next unvisited
        // neighbor's vector data while computing distance on the current one.
        let neighborCount = neighborIndices.len();
        for (nPos, &neighborIdx) in neighborIndices.iter().enumerate() {
            let ni = neighborIdx as usize;
            if ni >= nodes.len() || visitedGen[ni] == currentGen {
                continue;
            }
            visitedGen[ni] = currentGen;

            // Prefetch next unvisited neighbor's full vector into L1 cache.
            // Issue prefetch hints across every 64 bytes (cache line) to cover
            // the entire vector, not just the first element.
            #[cfg(target_arch = "x86_64")]
            {
                if nPos + 1 < neighborCount {
                    let nextNi = neighborIndices[nPos + 1] as usize;
                    if nextNi < nodes.len() && visitedGen[nextNi] != currentGen {
                        let prefetchOffset = nextNi * dims;
                        if prefetchOffset + dims <= arena.len() {
                            unsafe {
                                use std::arch::x86_64::*;
                                let base = arena.as_ptr().add(prefetchOffset) as *const i8;
                                // 128d * 4 bytes = 512 bytes = 8 cache lines.
                                let cacheLines = (dims * 4 + 63) / 64;
                                for cl in 0..cacheLines {
                                    _mm_prefetch(base.add(cl * 64), _MM_HINT_T0);
                                }
                            }
                        }
                    }
                }
            }

            let dist = computeDistance(metric, query, arenaSlice(arena, dims, ni));
            let farthestResult = state
                .results
                .peek()
                .map(|e| e.dist)
                .unwrap_or(f32::INFINITY);

            if dist < farthestResult || state.results.len() < ef {
                state.candidates.push(Reverse(DistEntry { dist, idx: ni }));
                state.results.push(DistEntry { dist, idx: ni });
                if state.results.len() > ef {
                    state.results.pop();
                }
            }
        }
    }

    // Drain results into sorted output
    state.output.clear();
    state
        .output
        .extend(state.results.drain().map(|e| (e.idx, e.dist)));
    state.output.sort_unstable_by(|a, b| a.1.total_cmp(&b.1));
}

/// Allocating version of beam search (for the concurrent search path where
/// we can't reuse state across calls).
fn searchLayerFn(
    metric: DistanceMetric,
    dims: usize,
    query: &[f32],
    entryIdx: usize,
    ef: usize,
    layer: usize,
    nodes: &[HnswNode],
    arena: &[f32],
    visitedGen: &mut Vec<u32>,
    currentGen: u32,
) -> Vec<(usize, f32)> {
    let mut state = SearchState::new(ef + 16);
    searchLayerReuse(
        metric, dims, query, entryIdx, ef, layer, nodes, arena, visitedGen, currentGen, &mut state,
    );
    std::mem::take(&mut state.output)
}

/// Beam search at layer 0 using u8 quantized distances for fast candidate
/// ranking. Returns (nodeIdx, quantizedDist) pairs. The caller must recompute
/// exact f32 distances on the returned candidates for final ranking.
fn searchLayerQuantized(
    dims: usize,
    queryQuantized: &[u8],
    entryIdx: usize,
    ef: usize,
    nodes: &[HnswNode],
    quantizedArena: &[u8],
    visitedGen: &mut Vec<u32>,
    currentGen: u32,
) -> Vec<(usize, u32)> {
    let mut candidates: BinaryHeap<Reverse<(u32, usize)>> = BinaryHeap::with_capacity(ef + 16);
    let mut results: BinaryHeap<(u32, usize)> = BinaryHeap::with_capacity(ef + 16);

    let entrySlice = &quantizedArena[entryIdx * dims..(entryIdx + 1) * dims];
    let entryDist = euclideanQuantized(queryQuantized, entrySlice);

    candidates.push(Reverse((entryDist, entryIdx)));
    results.push((entryDist, entryIdx));
    visitedGen[entryIdx] = currentGen;

    while let Some(Reverse((currentDist, currentIdx))) = candidates.pop() {
        if results.len() >= ef {
            let farthest = results.peek().map(|&(d, _)| d).unwrap_or(u32::MAX);
            if currentDist > farthest {
                break;
            }
        }

        let neighborIndices = if !nodes[currentIdx].connections.is_empty() {
            &nodes[currentIdx].connections[0]
        } else {
            continue;
        };

        for &neighborIdx in neighborIndices {
            let ni = neighborIdx as usize;
            if ni >= nodes.len() || visitedGen[ni] == currentGen {
                continue;
            }
            visitedGen[ni] = currentGen;

            let nStart = ni * dims;
            let nEnd = nStart + dims;
            if nEnd > quantizedArena.len() {
                continue;
            }
            let dist = euclideanQuantized(queryQuantized, &quantizedArena[nStart..nEnd]);
            let farthest = results.peek().map(|&(d, _)| d).unwrap_or(u32::MAX);

            if dist < farthest || results.len() < ef {
                candidates.push(Reverse((dist, ni)));
                results.push((dist, ni));
                if results.len() > ef {
                    results.pop();
                }
            }
        }
    }

    // Return unsorted - caller reranks by exact distance and sorts there.
    // Avoids redundant sort since quantized ordering will be discarded.
    results.drain().map(|(d, idx)| (idx, d)).collect()
}

/// Heuristic neighbor selection (HNSW paper Algorithm 4).
/// Takes resolved DistFn for zero-overhead distance calls.
fn selectNeighborsHeuristic(
    candidates: &[(usize, f32)],
    m: usize,
    nodes: &[HnswNode],
    arena: &[f32],
    dims: usize,
    metric: DistanceMetric,
) -> Vec<u32> {
    let mut sorted: Vec<(usize, f32)> = candidates.to_vec();
    sorted.sort_unstable_by(|a, b| a.1.total_cmp(&b.1));

    let mut selected: Vec<u32> = Vec::with_capacity(m);
    let mut rejected: Vec<u32> = Vec::new();

    for &(idx, dist) in &sorted {
        if selected.len() >= m {
            break;
        }
        if nodes[idx].deleted.load(Ordering::Relaxed) {
            continue;
        }

        let mut dominated = false;
        let candidateVec = arenaSlice(arena, dims, idx);
        for &sel in &selected {
            let interDist =
                computeDistance(metric, candidateVec, arenaSlice(arena, dims, sel as usize));
            if interDist < dist {
                dominated = true;
                break;
            }
        }

        if !dominated {
            selected.push(idx as u32);
        } else {
            rejected.push(idx as u32);
        }
    }

    for &idx in &rejected {
        if selected.len() >= m {
            break;
        }
        selected.push(idx);
    }

    selected
}

/// Simple closest-m neighbor selection for pruning.
fn selectNeighborsSimple(candidates: &[(usize, f32)], m: usize, nodes: &[HnswNode]) -> Vec<u32> {
    let mut sorted: Vec<(usize, f32)> = candidates.to_vec();
    sorted.sort_unstable_by(|a, b| a.1.total_cmp(&b.1));

    let mut selected = Vec::with_capacity(m);
    for &(idx, _) in &sorted {
        if selected.len() >= m {
            break;
        }
        if !nodes[idx].deleted.load(Ordering::Relaxed) {
            selected.push(idx as u32);
        }
    }
    selected
}

// ---------------------------------------------------------------------------
// HNSW graph node
// ---------------------------------------------------------------------------

struct HnswNode {
    id: VectorId,
    connections: Vec<Vec<u32>>,
    deleted: AtomicBool,
}

// ---------------------------------------------------------------------------
// Concurrent build (shared graph with per-node locking)
// ---------------------------------------------------------------------------
// Multiple threads insert into a single shared graph simultaneously. Each
// node has its own RwLock around its connection list so threads only block
// on the specific neighbors they're modifying. Produces identical quality
// to single-thread build when the same parameters and pruning are used.
//
// The build path is structured around four invariants:
//   1. Pruning uses the HNSW Algorithm 4 heuristic (selectNeighborsBuild),
//      which keeps diverse long-range connections and is required for high
//      recall at any scale.
//   2. Every insert uses the configured efConstruction in full. A constant
//      ef across the build matches what hnswlib and the HNSW paper
//      describe, and keeps late inserts from settling for locally-nearby
//      but globally-weak neighbors.
//   3. Vectors are inserted in the order the caller provided. Preserving
//      input order keeps temporal locality intact for callers that stage
//      data by cluster, and avoids poor graph quality caused by shuffling
//      or Z-order remapping on high-dimensional data.
//   4. m and efConstruction are capped by the auto-tuner so the heuristic
//      domination check does not reject within-cluster candidates.

struct BuildNode {
    id: VectorId,
    level: usize,
    connections: parking_lot::RwLock<Vec<Vec<u32>>>,
}

/// Beam search over BuildNode graph. Takes a per-node read lock to copy
/// the neighbor list for one layer, then releases the lock before doing
/// distance computations on those neighbors. This minimizes lock hold time
/// so concurrent searches don't block each other.
fn searchLayerBuild(
    metric: DistanceMetric,
    dims: usize,
    query: &[f32],
    entryIdx: usize,
    ef: usize,
    layer: usize,
    nodes: &[BuildNode],
    arena: &[f32],
    visitedGen: &mut [u32],
    currentGen: u32,
    state: &mut SearchState,
    neighborBuf: &mut Vec<u32>,
) {
    state.clear();
    let entryDist = computeDistance(metric, query, arenaSlice(arena, dims, entryIdx));

    state.candidates.push(Reverse(DistEntry {
        dist: entryDist,
        idx: entryIdx,
    }));
    state.results.push(DistEntry {
        dist: entryDist,
        idx: entryIdx,
    });
    visitedGen[entryIdx] = currentGen;

    while let Some(Reverse(current)) = state.candidates.pop() {
        if state.results.len() >= ef {
            let farthest = state
                .results
                .peek()
                .map(|e| e.dist)
                .unwrap_or(f32::INFINITY);
            if current.dist > farthest {
                break;
            }
        }

        neighborBuf.clear();
        {
            let conns = nodes[current.idx].connections.read();
            if layer < conns.len() {
                neighborBuf.extend_from_slice(&conns[layer]);
            } else {
                continue;
            }
        }

        for &neighborIdx in neighborBuf.iter() {
            let ni = neighborIdx as usize;
            if ni >= nodes.len() || visitedGen[ni] == currentGen {
                continue;
            }
            visitedGen[ni] = currentGen;

            let dist = computeDistance(metric, query, arenaSlice(arena, dims, ni));
            let farthest = state
                .results
                .peek()
                .map(|e| e.dist)
                .unwrap_or(f32::INFINITY);

            if dist < farthest || state.results.len() < ef {
                state.candidates.push(Reverse(DistEntry { dist, idx: ni }));
                state.results.push(DistEntry { dist, idx: ni });
                if state.results.len() > ef {
                    state.results.pop();
                }
            }
        }
    }

    state.output.clear();
    state
        .output
        .extend(state.results.drain().map(|e| (e.idx, e.dist)));
    state.output.sort_unstable_by(|a, b| a.1.total_cmp(&b.1));
}

/// HNSW Algorithm 4 heuristic neighbor selection for BuildNode. Same as
/// selectNeighborsHeuristic but works on BuildNode (no deleted flag check
/// since nodes aren't deleted during build).
fn selectNeighborsBuild(
    candidates: &[(usize, f32)],
    m: usize,
    arena: &[f32],
    dims: usize,
    metric: DistanceMetric,
) -> Vec<u32> {
    let mut sorted: Vec<(usize, f32)> = candidates.to_vec();
    sorted.sort_unstable_by(|a, b| a.1.total_cmp(&b.1));

    let mut selected: Vec<u32> = Vec::with_capacity(m);
    let mut rejected: Vec<u32> = Vec::new();

    for &(idx, dist) in &sorted {
        if selected.len() >= m {
            break;
        }
        let mut dominated = false;
        let candidateVec = arenaSlice(arena, dims, idx);
        for &sel in &selected {
            let interDist =
                computeDistance(metric, candidateVec, arenaSlice(arena, dims, sel as usize));
            if interDist < dist {
                dominated = true;
                break;
            }
        }
        if !dominated {
            selected.push(idx as u32);
        } else {
            rejected.push(idx as u32);
        }
    }

    // Fallback: if not enough non-dominated candidates, add dominated ones
    // to meet the m target. This ensures the graph stays well-connected.
    for &idx in &rejected {
        if selected.len() >= m {
            break;
        }
        selected.push(idx);
    }

    selected
}

/// Concurrent HNSW build. All threads insert into a shared BuildNode graph
/// with per-node RwLock. Uses heuristic pruning, full efConstruction, and
/// input-order insertion, matching the correctness choices of the
/// single-thread path so graph quality does not depend on thread count.
fn concurrentBuild(
    vectors: &[(VectorId, &[f32])],
    dims: usize,
    config: &HnswConfig,
    nThreads: usize,
    indexId: u32,
) -> Result<(Vec<HnswNode>, Vec<f32>, Vec<(VectorId, usize)>, u64, u16)> {
    let n = vectors.len();
    let m = config.m as usize;
    let metric = config.metric;
    let efConstruction = config.efConstruction as usize;

    // Pre-allocate arena with vectors in input order.
    let mut arena: Vec<f32> = try_alloc_vec(n * dims)?;
    for &(_, vec) in vectors {
        if vec.len() != dims {
            return Err(ZyronError::InvalidParameter {
                name: "vector.len()".to_string(),
                value: format!("expected {} dimensions, got {}", dims, vec.len()),
            });
        }
        arena.extend_from_slice(vec);
    }

    // Pre-compute levels using a deterministic RNG seeded from indexId.
    let mut rng = (indexId as u64)
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1);
    if rng == 0 {
        rng = 1;
    }
    let mut buildNodes: Vec<BuildNode> = Vec::with_capacity(n);
    for &(id, _) in vectors {
        let level = randomLevelFrom(&mut rng, config.m);
        // Each connection list is allocated to its maximum capacity up front
        // so concurrent inserts never have to grow the backing buffer. The
        // cap is m*2 at layer 0 and m above, matching the heuristic-pruning
        // limit applied later when edges overflow.
        let conns: Vec<Vec<u32>> = (0..=level)
            .map(|l| {
                let cap = if l == 0 { m * 2 } else { m };
                Vec::with_capacity(cap)
            })
            .collect();
        buildNodes.push(BuildNode {
            id,
            level,
            connections: parking_lot::RwLock::new(conns),
        });
    }

    let entryPoint = AtomicU64::new(0);
    let maxLayer = AtomicU16::new(buildNodes[0].level as u16);

    // Sequential bootstrap: insert the first few nodes one at a time so
    // the graph has a starting structure before concurrent threads begin.
    // Small bootstrap (just enough to establish upper layers).
    let bootstrapCount = n.min(100);

    // Shared insert function usable from both bootstrap and concurrent phase.
    // Returns nothing. Mutates shared state via locks and atomics.
    let insertOne = |insertIdx: usize,
                     visitedGen: &mut Vec<u32>,
                     currentGen: &mut u32,
                     searchState: &mut SearchState,
                     neighborBuf: &mut Vec<u32>,
                     pruneBuf: &mut Vec<(usize, f32)>| {
        if insertIdx == 0 {
            return; // node 0 is already the entry point
        }
        let vec = arenaSlice(&arena, dims, insertIdx);
        let level = buildNodes[insertIdx].level;

        // Load maxLayer before entryPoint. The CAS update writes maxLayer then
        // entryPoint, so loading in reverse order preserves the invariant that
        // entryPoint has at least currentMaxLayer levels. Otherwise a reader
        // could observe the new maxLayer with the old entryPoint and start a
        // search at a layer the entry node doesn't have.
        let currentMaxLayer = maxLayer.load(Ordering::Acquire) as usize;
        let mut ep = entryPoint.load(Ordering::Acquire) as usize;

        // Greedy descent through upper layers (above this node's level).
        if level < currentMaxLayer {
            for l in (level + 1..=currentMaxLayer).rev() {
                *currentGen = currentGen.wrapping_add(1);
                if *currentGen == 0 {
                    *currentGen = 1;
                    visitedGen.fill(0);
                }
                searchLayerBuild(
                    metric,
                    dims,
                    vec,
                    ep,
                    1,
                    l,
                    &buildNodes,
                    &arena,
                    visitedGen,
                    *currentGen,
                    searchState,
                    neighborBuf,
                );
                if let Some(&(closest, _)) = searchState.output.first() {
                    ep = closest;
                }
            }
        }

        // Connect at each layer from min(level, currentMaxLayer) down to 0.
        let topConnectLayer = level.min(currentMaxLayer);
        for l in (0..=topConnectLayer).rev() {
            *currentGen = currentGen.wrapping_add(1);
            if *currentGen == 0 {
                *currentGen = 1;
                visitedGen.fill(0);
            }
            searchLayerBuild(
                metric,
                dims,
                vec,
                ep,
                efConstruction,
                l,
                &buildNodes,
                &arena,
                visitedGen,
                *currentGen,
                searchState,
                neighborBuf,
            );
            let candidates = &searchState.output;
            if let Some(&(closest, _)) = candidates.first() {
                ep = closest;
            }

            let maxConn = if l == 0 { m * 2 } else { m };
            let neighbors = selectNeighborsBuild(candidates, maxConn, &arena, dims, metric);

            // Write own connections at this layer. Reuse the pre-allocated Vec
            // to avoid heap alloc/free churn during concurrent build.
            {
                let mut conns = buildNodes[insertIdx].connections.write();
                if l < conns.len() {
                    conns[l].clear();
                    conns[l].extend_from_slice(&neighbors);
                }
            }

            // Add bidirectional edges with overflow pruning via heuristic.
            // Reuses the pre-allocated connection Vec (clear+extend) instead of
            // replacing it (drop old + alloc new) to avoid concurrent heap churn.
            for &neighborIdx in &neighbors {
                let ni = neighborIdx as usize;
                let mut conns = buildNodes[ni].connections.write();
                if l < conns.len() {
                    conns[l].push(insertIdx as u32);
                    if conns[l].len() > maxConn {
                        pruneBuf.clear();
                        pruneBuf.extend(conns[l].iter().map(|&cidx| {
                            let d = computeDistance(
                                metric,
                                arenaSlice(&arena, dims, ni),
                                arenaSlice(&arena, dims, cidx as usize),
                            );
                            (cidx as usize, d)
                        }));
                        let pruned = selectNeighborsBuild(pruneBuf, maxConn, &arena, dims, metric);
                        conns[l].clear();
                        conns[l].extend_from_slice(&pruned);
                    }
                }
            }
        }

        // Update entry point if this node has a higher level via CAS loop.
        if level as u16 > maxLayer.load(Ordering::Acquire) {
            loop {
                let currentMax = maxLayer.load(Ordering::Acquire);
                if (level as u16) <= currentMax {
                    break;
                }
                if maxLayer
                    .compare_exchange(
                        currentMax,
                        level as u16,
                        Ordering::AcqRel,
                        Ordering::Relaxed,
                    )
                    .is_ok()
                {
                    entryPoint.store(insertIdx as u64, Ordering::Release);
                    break;
                }
            }
        }
    };

    // Bootstrap phase: sequential inserts 1..bootstrapCount.
    {
        let mut visitedGen: Vec<u32> = try_alloc_default(n)?;
        let mut currentGen: u32 = 0;
        let mut searchState = SearchState::new(efConstruction + 16);
        let mut neighborBuf: Vec<u32> = Vec::with_capacity(m * 2 + 16);
        let mut pruneBuf: Vec<(usize, f32)> = Vec::with_capacity(m * 2 + 16);
        for insertIdx in 1..bootstrapCount {
            insertOne(
                insertIdx,
                &mut visitedGen,
                &mut currentGen,
                &mut searchState,
                &mut neighborBuf,
                &mut pruneBuf,
            );
        }
    }

    // Concurrent phase: threads race to grab indices via atomic counter.
    let nextInsert = AtomicU64::new(bootstrapCount as u64);
    let insertOneRef = &insertOne;
    std::thread::scope(|s| -> Result<()> {
        let mut handles = Vec::with_capacity(nThreads);
        for threadIdx in 0..nThreads {
            let nextRef = &nextInsert;
            handles.push(s.spawn(move || -> Result<()> {
                let mut visitedGen: Vec<u32> = try_alloc_default(n)?;
                let mut currentGen: u32 = (threadIdx as u32 + 1).wrapping_mul(1_000_003);
                let mut searchState = SearchState::new(efConstruction + 16);
                let mut neighborBuf: Vec<u32> = Vec::with_capacity(m * 2 + 16);
                let mut pruneBuf: Vec<(usize, f32)> = Vec::with_capacity(m * 2 + 16);
                loop {
                    let insertIdx = nextRef.fetch_add(1, Ordering::AcqRel) as usize;
                    if insertIdx >= n {
                        break;
                    }
                    insertOneRef(
                        insertIdx,
                        &mut visitedGen,
                        &mut currentGen,
                        &mut searchState,
                        &mut neighborBuf,
                        &mut pruneBuf,
                    );
                }
                Ok(())
            }));
        }
        for h in handles {
            h.join().expect("concurrent build thread panicked")?;
        }
        Ok(())
    })?;

    // Finalize: convert BuildNode -> HnswNode.
    let ep = entryPoint.load(Ordering::Acquire);
    let ml = maxLayer.load(Ordering::Acquire);
    let mut hnswNodes: Vec<HnswNode> = Vec::with_capacity(n);
    let mut idPairs: Vec<(VectorId, usize)> = Vec::with_capacity(n);
    for (i, bnode) in buildNodes.into_iter().enumerate() {
        let conns = bnode.connections.into_inner();
        idPairs.push((bnode.id, i));
        hnswNodes.push(HnswNode {
            id: bnode.id,
            connections: conns,
            deleted: AtomicBool::new(false),
        });
    }

    Ok((hnswNodes, arena, idPairs, ep, ml))
}

// ---------------------------------------------------------------------------
// AnnIndex
// ---------------------------------------------------------------------------

pub struct AnnIndex {
    pub indexId: u32,
    pub tableId: u32,
    pub columnId: u16,
    config: HnswConfig,
    dimensions: u16,
    nodes: RwLock<Vec<HnswNode>>,
    vectorArena: RwLock<Vec<f32>>,
    /// u8 scalar-quantized vector arena for fast approximate distance during
    /// beam search. Each dimension is quantized to [0, 255] using per-dimension
    /// min/max scaling. 4x less memory bandwidth than f32 arena.
    quantizedArena: RwLock<Vec<u8>>,
    /// Per-dimension min values for dequantization.
    quantMins: Vec<f32>,
    /// Per-dimension scale factors: 255 / (max - min).
    quantScales: Vec<f32>,
    idMap: scc::HashMap<VectorId, usize>,
    entryPoint: AtomicU64,
    maxLayer: AtomicU16,
    nodeCount: AtomicU64,
    rngCounter: AtomicU64,
    profile: Option<DataProfile>,
}

impl AnnIndex {
    pub fn new(
        indexId: u32,
        tableId: u32,
        columnId: u16,
        dimensions: u16,
        config: HnswConfig,
    ) -> Self {
        let rngSeed = (indexId as u64)
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1);
        Self {
            indexId,
            tableId,
            columnId,
            config,
            dimensions,
            nodes: RwLock::new(Vec::new()),
            vectorArena: RwLock::new(Vec::new()),
            quantizedArena: RwLock::new(Vec::new()),
            quantMins: Vec::new(),
            quantScales: Vec::new(),
            idMap: scc::HashMap::new(),
            entryPoint: AtomicU64::new(u64::MAX),
            maxLayer: AtomicU16::new(0),
            nodeCount: AtomicU64::new(0),
            rngCounter: AtomicU64::new(rngSeed),
            profile: None,
        }
    }

    /// Parallel batch build. Partitions vectors across CPU cores, builds
    /// independent sub-graphs in parallel (zero synchronization), then merges
    /// by concatenating and cross-linking. Uses adaptive efConstruction that
    /// decreases as each sub-graph grows. Computes a DataProfile from a
    /// sample of vectors to drive cross-linking and adaptive parameters.
    pub fn build(
        vectors: &[(VectorId, &[f32])],
        indexId: u32,
        tableId: u32,
        columnId: u16,
        config: HnswConfig,
    ) -> Result<Self> {
        if vectors.is_empty() {
            return Ok(Self::new(indexId, tableId, columnId, 0, config));
        }
        let dimensions = vectors[0].1.len() as u16;
        let dims = dimensions as usize;
        let n = vectors.len();

        // Compute DataProfile from a sample to drive all adaptive parameters.
        let sampleSize = n.min(1000);
        let sampleSlices: Vec<&[f32]> = vectors[..sampleSize].iter().map(|(_, v)| *v).collect();
        let profile = DataProfile::compute(&sampleSlices, n, dimensions, config.metric);

        let nThreads = std::thread::available_parallelism()
            .map(|p| p.get())
            .unwrap_or(1)
            .min(n);
        let parallelThreshold = profile.parallelThreshold();

        if n < parallelThreshold || nThreads <= 1 {
            return Self::buildSingleThreadWithProfile(
                vectors, indexId, tableId, columnId, config, profile,
            );
        }

        // Concurrent build: all threads insert into a shared graph with
        // per-node RwLock. Same pruning (heuristic) and same params as
        // single-thread, just parallelized. Produces equivalent graph
        // quality to single-thread at a fraction of the wall time.
        let (mergedNodes, mergedArena, mergedIdPairs, globalEntryPoint, globalMaxLayer) =
            concurrentBuild(vectors, dims, &config, nThreads, indexId)?;
        let totalNodes = mergedNodes.len();

        let idMap: scc::HashMap<VectorId, usize> = scc::HashMap::new();
        for &(vid, idx) in &mergedIdPairs {
            let _ = idMap.insert_sync(vid, idx);
        }

        let (qMins, qMaxs) = computeQuantizationBounds(&mergedArena, dims);
        let (quantized, qScales) = quantizeArena(&mergedArena, dims, &qMins, &qMaxs);

        let rngSeed = (indexId as u64)
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1);
        Ok(Self {
            indexId,
            tableId,
            columnId,
            config,
            dimensions,
            nodes: RwLock::new(mergedNodes),
            vectorArena: RwLock::new(mergedArena),
            quantizedArena: RwLock::new(quantized),
            quantMins: qMins,
            quantScales: qScales,
            idMap,
            entryPoint: AtomicU64::new(globalEntryPoint),
            maxLayer: AtomicU16::new(globalMaxLayer),
            nodeCount: AtomicU64::new(totalNodes as u64),
            rngCounter: AtomicU64::new(rngSeed.wrapping_add(totalNodes as u64)),
            profile: Some(profile),
        })
    }

    /// Single-threaded build for small inputs or single-core systems.
    fn buildSingleThreadWithProfile(
        vectors: &[(VectorId, &[f32])],
        indexId: u32,
        tableId: u32,
        columnId: u16,
        config: HnswConfig,
        profile: DataProfile,
    ) -> Result<Self> {
        let dimensions = vectors[0].1.len() as u16;
        let dims = dimensions as usize;
        let m = config.m as usize;
        let efConstruction = config.efConstruction as usize;
        let metric = config.metric;

        let n = vectors.len();
        let mut arena: Vec<f32> = try_alloc_vec(n * dims)?;
        let mut nodes: Vec<HnswNode> = try_alloc_vec(n)?;
        let mut visitedGen: Vec<u32> = try_alloc_default(n)?;
        let mut currentGen: u32 = 0;
        let mut entryPoint: u64 = u64::MAX;
        let mut maxLayer: u16 = 0;
        let mut rng = (indexId as u64)
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1);
        if rng == 0 {
            rng = 1;
        }
        let mut idVec: Vec<(VectorId, usize)> = try_alloc_vec(n)?;
        let mut searchState = SearchState::new(efConstruction + 16);
        let mut pruneBuf: Vec<(usize, f32)> = Vec::with_capacity(m * 2 + 16);

        for &(id, vec) in vectors {
            if vec.len() != dims {
                return Err(ZyronError::InvalidParameter {
                    name: "vector.len()".to_string(),
                    value: format!("expected {} dimensions, got {}", dims, vec.len()),
                });
            }

            let level = randomLevelFrom(&mut rng, config.m);
            let newIdx = nodes.len();
            nodes.push(HnswNode {
                id,
                connections: vec![Vec::new(); level + 1],
                deleted: AtomicBool::new(false),
            });
            arena.extend_from_slice(vec);
            idVec.push((id, newIdx));

            if entryPoint == u64::MAX {
                entryPoint = newIdx as u64;
                maxLayer = level as u16;
                continue;
            }

            if visitedGen.len() < nodes.len() {
                visitedGen.resize(nodes.len(), 0);
            }
            let mut ep = entryPoint as usize;
            let currentMaxLayer = maxLayer as usize;

            // Use full efConstruction for every insert (no adaptive decay).
            let _ = newIdx; // no longer used
            let ef = config.efConstruction as usize;

            if level < currentMaxLayer {
                for l in (level + 1..=currentMaxLayer).rev() {
                    currentGen = currentGen.wrapping_add(1);
                    if currentGen == 0 {
                        currentGen = 1;
                        visitedGen.fill(0);
                    }
                    searchLayerReuse(
                        metric,
                        dims,
                        vec,
                        ep,
                        1,
                        l,
                        &nodes,
                        &arena,
                        &mut visitedGen,
                        currentGen,
                        &mut searchState,
                    );
                    if let Some(&(closest, _)) = searchState.output.first() {
                        ep = closest;
                    }
                }
            }

            let topConnectLayer = level.min(currentMaxLayer);
            for l in (0..=topConnectLayer).rev() {
                currentGen = currentGen.wrapping_add(1);
                if currentGen == 0 {
                    currentGen = 1;
                    visitedGen.fill(0);
                }
                searchLayerReuse(
                    metric,
                    dims,
                    vec,
                    ep,
                    ef,
                    l,
                    &nodes,
                    &arena,
                    &mut visitedGen,
                    currentGen,
                    &mut searchState,
                );
                let candidates = &searchState.output;
                if let Some(&(closest, _)) = candidates.first() {
                    ep = closest;
                }

                let maxConn = if l == 0 { m * 2 } else { m };
                let neighbors =
                    selectNeighborsHeuristic(&candidates, maxConn, &nodes, &arena, dims, metric);
                nodes[newIdx].connections[l] = neighbors.clone();

                for &neighborIdx in &neighbors {
                    let ni = neighborIdx as usize;
                    if l < nodes[ni].connections.len() {
                        nodes[ni].connections[l].push(newIdx as u32);
                        if nodes[ni].connections[l].len() > maxConn {
                            pruneBuf.clear();
                            pruneBuf.extend(nodes[ni].connections[l].iter().map(|&cidx| {
                                let d = computeDistance(
                                    metric,
                                    arenaSlice(&arena, dims, ni),
                                    arenaSlice(&arena, dims, cidx as usize),
                                );
                                (cidx as usize, d)
                            }));
                            // Prune with the HNSW Algorithm 4 heuristic so
                            // long-range connections are preserved when a
                            // node overflows maxConn.
                            let pruned = selectNeighborsHeuristic(
                                &pruneBuf, maxConn, &nodes, &arena, dims, metric,
                            );
                            nodes[ni].connections[l].clear();
                            nodes[ni].connections[l].extend_from_slice(&pruned);
                        }
                    }
                }
            }

            if level as u16 > maxLayer {
                entryPoint = newIdx as u64;
                maxLayer = level as u16;
            }
        }

        let idMap: scc::HashMap<VectorId, usize> = scc::HashMap::new();
        for (vid, idx) in idVec {
            let _ = idMap.insert_sync(vid, idx);
        }

        let (qMins, qMaxs) = computeQuantizationBounds(&arena, dims);
        let (quantized, qScales) = quantizeArena(&arena, dims, &qMins, &qMaxs);

        Ok(Self {
            indexId,
            tableId,
            columnId,
            config,
            dimensions,
            nodes: RwLock::new(nodes),
            vectorArena: RwLock::new(arena),
            quantizedArena: RwLock::new(quantized),
            quantMins: qMins,
            quantScales: qScales,
            idMap,
            entryPoint: AtomicU64::new(entryPoint),
            maxLayer: AtomicU16::new(maxLayer),
            nodeCount: AtomicU64::new(vectors.len() as u64),
            rngCounter: AtomicU64::new(rng),
            profile: Some(profile),
        })
    }

    /// Lock-free random level for online inserts (uses atomic counter).
    fn randomLevel(&self) -> usize {
        let counter = self.rngCounter.fetch_add(1, Ordering::Relaxed);
        let mut seed = counter
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        if seed == 0 {
            seed = 1;
        }
        randomLevelFrom(&mut seed, self.config.m)
    }

    /// Returns the DataProfile computed at build time, if available.
    pub fn profile(&self) -> Option<&DataProfile> {
        self.profile.as_ref()
    }

    /// Inserts a single vector into the HNSW graph (online path, uses locks).
    fn insertInternal(&self, id: VectorId, vector: &[f32]) -> Result<()> {
        let dims = self.dimensions as usize;
        if vector.len() != dims {
            return Err(ZyronError::InvalidParameter {
                name: "vector.len()".to_string(),
                value: format!("expected {} dimensions, got {}", dims, vector.len()),
            });
        }

        let level = self.randomLevel();
        let m = self.config.m as usize;
        let efConstruction = self.config.efConstruction as usize;

        let node = HnswNode {
            id,
            connections: vec![Vec::new(); level + 1],
            deleted: AtomicBool::new(false),
        };

        let metric = self.config.metric;
        let mut nodesGuard = self.nodes.write();
        let mut arenaGuard = self.vectorArena.write();
        let newIdx = nodesGuard.len();
        nodesGuard.push(node);
        arenaGuard.extend_from_slice(vector);

        // Keep quantized arena in sync with vectorArena. Without this, online
        // inserts leave stale/garbage data in the quantized arena and cause
        // incorrect beam search distances.
        if !self.quantMins.is_empty() && self.quantMins.len() == dims {
            let mut quantGuard = self.quantizedArena.write();
            let mut quantBuf = vec![0u8; dims];
            quantizeVector(vector, &self.quantMins, &self.quantScales, &mut quantBuf);
            quantGuard.extend_from_slice(&quantBuf);
        }

        let _ = self.idMap.insert_sync(id, newIdx);

        let currentEntry = self.entryPoint.load(Ordering::Acquire);

        if currentEntry == u64::MAX {
            self.entryPoint.store(newIdx as u64, Ordering::Release);
            self.maxLayer.store(level as u16, Ordering::Release);
            self.nodeCount.fetch_add(1, Ordering::Relaxed);
            return Ok(());
        }

        // Visited marker for this insert. try_alloc_default reserves the
        // buffer through Vec::try_reserve_exact and returns an error on OOM.
        let mut visitedGen: Vec<u32> = try_alloc_default(nodesGuard.len())?;
        let mut currentGen: u32 = 0;

        let mut ep = currentEntry as usize;
        let currentMaxLayer = self.maxLayer.load(Ordering::Acquire) as usize;

        if level < currentMaxLayer {
            for l in (level + 1..=currentMaxLayer).rev() {
                currentGen = currentGen.wrapping_add(1);
                if currentGen == 0 {
                    currentGen = 1;
                    visitedGen.fill(0);
                }
                let result = searchLayerFn(
                    metric,
                    dims,
                    vector,
                    ep,
                    1,
                    l,
                    &nodesGuard,
                    &arenaGuard,
                    &mut visitedGen,
                    currentGen,
                );
                if let Some(&(closest, _)) = result.first() {
                    ep = closest;
                }
            }
        }

        let topConnectLayer = level.min(currentMaxLayer);
        for l in (0..=topConnectLayer).rev() {
            currentGen = currentGen.wrapping_add(1);
            if currentGen == 0 {
                currentGen = 1;
                visitedGen.fill(0);
            }
            let candidates = searchLayerFn(
                metric,
                dims,
                vector,
                ep,
                efConstruction,
                l,
                &nodesGuard,
                &arenaGuard,
                &mut visitedGen,
                currentGen,
            );

            if let Some(&(closest, _)) = candidates.first() {
                ep = closest;
            }

            let maxConn = if l == 0 { m * 2 } else { m };
            let neighbors = selectNeighborsHeuristic(
                &candidates,
                maxConn,
                &nodesGuard,
                &arenaGuard,
                dims,
                metric,
            );

            nodesGuard[newIdx].connections[l] = neighbors.clone();

            for &neighborIdx in &neighbors {
                let ni = neighborIdx as usize;
                if l < nodesGuard[ni].connections.len() {
                    nodesGuard[ni].connections[l].push(newIdx as u32);
                    if nodesGuard[ni].connections[l].len() > maxConn {
                        let connWithDist: Vec<(usize, f32)> = nodesGuard[ni].connections[l]
                            .iter()
                            .map(|&cidx| {
                                let d = computeDistance(
                                    metric,
                                    arenaSlice(&arenaGuard, dims, ni),
                                    arenaSlice(&arenaGuard, dims, cidx as usize),
                                );
                                (cidx as usize, d)
                            })
                            .collect();
                        let pruned = selectNeighborsSimple(&connWithDist, maxConn, &nodesGuard);
                        nodesGuard[ni].connections[l] = pruned;
                    }
                }
            }
        }

        if level > currentMaxLayer {
            self.entryPoint.store(newIdx as u64, Ordering::Release);
            self.maxLayer.store(level as u16, Ordering::Release);
        }

        self.nodeCount.fetch_add(1, Ordering::Relaxed);
        Ok(())
    }

    // -----------------------------------------------------------------------
    // Persistence
    // -----------------------------------------------------------------------

    pub fn saveToFile(&self, path: &std::path::Path) -> Result<()> {
        let nodesGuard = self.nodes.read();
        let arenaGuard = self.vectorArena.read();
        let mut file = std::fs::File::create(path)?;
        let dims = self.dimensions as usize;

        file.write_all(b"ZYVEC\x01")?;
        file.write_all(&self.dimensions.to_le_bytes())?;
        file.write_all(&(nodesGuard.len() as u64).to_le_bytes())?;

        file.write_all(&self.config.m.to_le_bytes())?;
        file.write_all(&self.config.efConstruction.to_le_bytes())?;
        file.write_all(&self.config.efSearch.to_le_bytes())?;
        let metricByte = match self.config.metric {
            DistanceMetric::Cosine => 0u8,
            DistanceMetric::Euclidean => 1u8,
            DistanceMetric::DotProduct => 2u8,
            DistanceMetric::Manhattan => 3u8,
        };
        file.write_all(&[metricByte])?;

        file.write_all(&self.entryPoint.load(Ordering::Relaxed).to_le_bytes())?;
        file.write_all(&self.maxLayer.load(Ordering::Relaxed).to_le_bytes())?;

        for (i, node) in nodesGuard.iter().enumerate() {
            file.write_all(&node.id.to_le_bytes())?;
            let deleted = if node.deleted.load(Ordering::Relaxed) {
                1u8
            } else {
                0u8
            };
            file.write_all(&[deleted])?;

            let vecStart = i * dims;
            for j in 0..dims {
                file.write_all(&arenaGuard[vecStart + j].to_le_bytes())?;
            }

            file.write_all(&(node.connections.len() as u16).to_le_bytes())?;
            for layer in &node.connections {
                file.write_all(&(layer.len() as u32).to_le_bytes())?;
                for &conn in layer {
                    file.write_all(&conn.to_le_bytes())?;
                }
            }
        }

        file.flush()?;
        Ok(())
    }

    pub fn loadFromFile(
        path: &std::path::Path,
        indexId: u32,
        tableId: u32,
        columnId: u16,
    ) -> Result<Self> {
        let mut file = std::fs::File::open(path)?;

        let mut magic = [0u8; 6];
        file.read_exact(&mut magic)?;
        if &magic != b"ZYVEC\x01" {
            return Err(ZyronError::InvalidParameter {
                name: "magic".to_string(),
                value: "invalid HNSW file header".to_string(),
            });
        }

        let mut buf2 = [0u8; 2];
        let mut buf8 = [0u8; 8];

        file.read_exact(&mut buf2)?;
        let dimensions = u16::from_le_bytes(buf2);

        file.read_exact(&mut buf8)?;
        let nodeCount = u64::from_le_bytes(buf8);

        file.read_exact(&mut buf2)?;
        let m = u16::from_le_bytes(buf2);
        file.read_exact(&mut buf2)?;
        let efConstruction = u16::from_le_bytes(buf2);
        file.read_exact(&mut buf2)?;
        let efSearch = u16::from_le_bytes(buf2);

        let mut metricBuf = [0u8; 1];
        file.read_exact(&mut metricBuf)?;
        let metric = match metricBuf[0] {
            0 => DistanceMetric::Cosine,
            1 => DistanceMetric::Euclidean,
            2 => DistanceMetric::DotProduct,
            3 => DistanceMetric::Manhattan,
            other => {
                return Err(ZyronError::InvalidParameter {
                    name: "metric".to_string(),
                    value: format!("unknown metric byte {other}"),
                });
            }
        };

        let config = HnswConfig {
            m,
            efConstruction,
            efSearch,
            metric,
        };

        file.read_exact(&mut buf8)?;
        let entryPoint = u64::from_le_bytes(buf8);
        file.read_exact(&mut buf2)?;
        let maxLayer = u16::from_le_bytes(buf2);

        // Validate the declared arena size before allocating. A corrupt or
        // malicious file with an inflated nodeCount is rejected by
        // validate_file_size when the computed size exceeds the sanity cap.
        let declared_arena_bytes = (nodeCount as u64)
            .saturating_mul(dimensions as u64)
            .saturating_mul(4);
        let declared_nodes_bytes = (nodeCount as u64).saturating_mul(128); // approx HnswNode size
        let total_declared = declared_arena_bytes.saturating_add(declared_nodes_bytes);
        validate_file_size(total_declared)?;

        let mut nodes: Vec<HnswNode> = try_alloc_vec(nodeCount as usize)?;
        let mut vectorArena: Vec<f32> = try_alloc_vec(nodeCount as usize * dimensions as usize)?;
        let idMap = scc::HashMap::new();

        for i in 0..nodeCount as usize {
            file.read_exact(&mut buf8)?;
            let id = u64::from_le_bytes(buf8);

            let mut deletedBuf = [0u8; 1];
            file.read_exact(&mut deletedBuf)?;
            let deleted = deletedBuf[0] != 0;

            let mut buf4 = [0u8; 4];
            for _ in 0..dimensions {
                file.read_exact(&mut buf4)?;
                vectorArena.push(f32::from_le_bytes(buf4));
            }

            file.read_exact(&mut buf2)?;
            let numLayers = u16::from_le_bytes(buf2) as usize;
            // Sanity cap: HNSW layers should never exceed ~32 (log probability).
            if numLayers > 64 {
                return Err(ZyronError::VectorIndexFileCorrupt {
                    declared: numLayers as u64,
                });
            }
            let mut connections: Vec<Vec<u32>> = try_alloc_vec(numLayers)?;
            for _ in 0..numLayers {
                let mut buf4c = [0u8; 4];
                file.read_exact(&mut buf4c)?;
                let numConns = u32::from_le_bytes(buf4c) as usize;
                // Sanity cap per layer: m * 2 maxes out around 256, allow some slack.
                if numConns > 4096 {
                    return Err(ZyronError::VectorIndexFileCorrupt {
                        declared: numConns as u64,
                    });
                }
                let mut conns: Vec<u32> = try_alloc_vec(numConns)?;
                for _ in 0..numConns {
                    file.read_exact(&mut buf4c)?;
                    conns.push(u32::from_le_bytes(buf4c));
                }
                connections.push(conns);
            }

            let _ = idMap.insert_sync(id, i);
            nodes.push(HnswNode {
                id,
                connections,
                deleted: AtomicBool::new(deleted),
            });
        }

        // Build quantized arena from loaded vectors for fast search.
        let dims = dimensions as usize;
        let (qMins, qMaxs) = computeQuantizationBounds(&vectorArena, dims);
        let (quantized, qScales) = quantizeArena(&vectorArena, dims, &qMins, &qMaxs);

        let rngSeed = (indexId as u64)
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1);
        Ok(Self {
            indexId,
            tableId,
            columnId,
            config,
            dimensions,
            nodes: RwLock::new(nodes),
            vectorArena: RwLock::new(vectorArena),
            quantizedArena: RwLock::new(quantized),
            quantMins: qMins,
            quantScales: qScales,
            idMap,
            entryPoint: AtomicU64::new(entryPoint),
            maxLayer: AtomicU16::new(maxLayer),
            nodeCount: AtomicU64::new(nodeCount),
            rngCounter: AtomicU64::new(rngSeed.wrapping_add(nodeCount)),
            profile: None,
        })
    }
}

// ---------------------------------------------------------------------------
// VectorSearch trait implementation (concurrent path, uses read locks)
// ---------------------------------------------------------------------------

impl VectorSearch for AnnIndex {
    fn search(&self, query: &[f32], k: usize, efSearch: u16) -> Result<Vec<(VectorId, f32)>> {
        let dims = self.dimensions as usize;
        if query.len() != dims {
            return Err(ZyronError::InvalidParameter {
                name: "query.len()".to_string(),
                value: format!("expected {} dimensions, got {}", dims, query.len()),
            });
        }

        let ep = self.entryPoint.load(Ordering::Acquire);
        if ep == u64::MAX {
            return Ok(Vec::new());
        }

        // Read locks are free when no writer (single atomic load on parking_lot)
        let nodesGuard = self.nodes.read();
        let arenaGuard = self.vectorArena.read();
        if nodesGuard.is_empty() {
            return Ok(Vec::new());
        }

        let metric = self.config.metric;
        let nodeCount = nodesGuard.len();
        let mut currentEp = ep as usize;
        let currentMaxLayer = self.maxLayer.load(Ordering::Acquire) as usize;

        // Use thread-local visited buffer to avoid per-query allocation.
        // The generation counter provides O(1) reset between searches.
        // We track the last-used indexId so switching indexes on the same
        // thread properly resets state instead of carrying stale marks.
        let thisIndexId = self.indexId;
        let result = SEARCH_VISITED.with(|cell| {
            let mut borrow = cell.borrow_mut();
            let (ref mut visitedGen, ref mut currentGen, ref mut lastIndexId) = *borrow;

            // Reset if switching to a different index, or if the buffer has
            // grown way beyond what's needed (index shrunk significantly).
            if *lastIndexId != thisIndexId {
                visitedGen.clear();
                visitedGen.resize(nodeCount, 0);
                *currentGen = 0;
                *lastIndexId = thisIndexId;
            } else if visitedGen.len() < nodeCount {
                visitedGen.resize(nodeCount, 0);
            }

            for l in (1..=currentMaxLayer).rev() {
                *currentGen = currentGen.wrapping_add(1);
                if *currentGen == 0 {
                    *currentGen = 1;
                    visitedGen.fill(0);
                }
                let found = searchLayerFn(
                    metric,
                    dims,
                    query,
                    currentEp,
                    1,
                    l,
                    &nodesGuard,
                    &arenaGuard,
                    visitedGen,
                    *currentGen,
                );
                if let Some(&(closest, _)) = found.first() {
                    currentEp = closest;
                }
            }

            // efSearch=0 means "use the config's auto-tuned value".
            let effectiveEf = if efSearch == 0 {
                self.config.efSearch
            } else {
                efSearch
            };
            let ef = (effectiveEf as usize).max(k);
            *currentGen = currentGen.wrapping_add(1);
            if *currentGen == 0 {
                *currentGen = 1;
                visitedGen.fill(0);
            }

            // Use quantized beam search for layer 0 if available. Only valid for
            // Euclidean metric because the u8 quantization and euclideanQuantized
            // distance function preserve Euclidean ordering. For other metrics
            // (Cosine, DotProduct, Manhattan), fall through to the f32 path.
            let quantGuard = self.quantizedArena.read();
            if !quantGuard.is_empty()
                && !self.quantMins.is_empty()
                && metric == DistanceMetric::Euclidean
            {
                // Quantize query vector once.
                let mut queryQ = vec![0u8; dims];
                quantizeVector(query, &self.quantMins, &self.quantScales, &mut queryQ);

                let quantResults = searchLayerQuantized(
                    dims,
                    &queryQ,
                    currentEp,
                    ef,
                    &nodesGuard,
                    &quantGuard,
                    visitedGen,
                    *currentGen,
                );
                // Return indices with placeholder distances (reranked below).
                quantResults
                    .into_iter()
                    .map(|(idx, d)| (idx, d as f32))
                    .collect()
            } else {
                searchLayerFn(
                    metric,
                    dims,
                    query,
                    currentEp,
                    ef,
                    0,
                    &nodesGuard,
                    &arenaGuard,
                    visitedGen,
                    *currentGen,
                )
            }
        });

        // Rerank with exact f32 distances on ALL ef candidates.
        // Quantized ordering differs from exact ordering, so we must compute
        // exact distance on every candidate, then sort and truncate to k.
        // Breaking early at k in quantized order would miss true neighbors.
        let mut output: Vec<(VectorId, f32)> = Vec::with_capacity(result.len());
        for (idx, _approxDist) in &result {
            if !nodesGuard[*idx].deleted.load(Ordering::Relaxed) {
                let exactDist = computeDistance(metric, query, arenaSlice(&arenaGuard, dims, *idx));
                output.push((nodesGuard[*idx].id, exactDist));
            }
        }
        output.sort_unstable_by(|a, b| a.1.total_cmp(&b.1));
        output.truncate(k);

        Ok(output)
    }

    fn insert(&self, id: VectorId, vector: &[f32]) -> Result<()> {
        self.insertInternal(id, vector)
    }

    fn delete(&self, id: VectorId) -> Result<()> {
        let idx = self.idMap.read_sync(&id, |_, &v| v);
        match idx {
            Some(i) => {
                let nodesGuard = self.nodes.read();
                if i < nodesGuard.len() {
                    nodesGuard[i].deleted.store(true, Ordering::Release);
                    self.nodeCount.fetch_sub(1, Ordering::Relaxed);
                }
                Ok(())
            }
            None => Err(ZyronError::InvalidParameter {
                name: "id".to_string(),
                value: format!("vector id {} not found", id),
            }),
        }
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
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn makeConfig() -> HnswConfig {
        HnswConfig {
            m: 16,
            efConstruction: 100,
            efSearch: 64,
            metric: DistanceMetric::Euclidean,
        }
    }

    #[test]
    fn buildAndSearchKnownVector() {
        let dims = 128;
        let mut vectors: Vec<(VectorId, Vec<f32>)> = Vec::with_capacity(100);
        for i in 0..100u64 {
            let v: Vec<f32> = (0..dims).map(|d| (i * dims + d) as f32 * 0.01).collect();
            vectors.push((i, v));
        }

        let refs: Vec<(VectorId, &[f32])> =
            vectors.iter().map(|(id, v)| (*id, v.as_slice())).collect();
        let index = AnnIndex::build(&refs, 1, 1, 0, makeConfig()).expect("build");

        let query = &vectors[42].1;
        let results = index.search(query, 1, 64).expect("search");
        assert!(!results.is_empty());
        assert_eq!(results[0].0, 42);
    }

    #[test]
    fn insertDeleteVerifyExclusion() {
        let dims = 32usize;
        let index = AnnIndex::new(2, 1, 0, dims as u16, makeConfig());

        for i in 0..10u64 {
            let v: Vec<f32> = (0..dims).map(|d| (i * 100 + d as u64) as f32).collect();
            index.insert(i, &v).expect("insert");
        }
        assert_eq!(index.len(), 10);

        for i in 0..5u64 {
            index.delete(i).expect("delete");
        }
        assert_eq!(index.len(), 5);

        let query: Vec<f32> = (0..dims).map(|d| d as f32).collect();
        let results = index.search(&query, 10, 64).expect("search");
        for (id, _) in &results {
            assert!(*id >= 5, "deleted vector id {} appeared in results", id);
        }
    }

    #[test]
    fn dimensionMismatchError() {
        let index = AnnIndex::new(3, 1, 0, 64, makeConfig());
        let valid: Vec<f32> = vec![1.0; 64];
        index.insert(1, &valid).expect("valid insert");

        let wrong: Vec<f32> = vec![1.0; 32];
        assert!(index.insert(2, &wrong).is_err());
    }

    #[test]
    fn emptyIndexSearchReturnsEmpty() {
        let index = AnnIndex::new(4, 1, 0, 16, makeConfig());
        let query = vec![0.0f32; 16];
        let results = index.search(&query, 5, 64).expect("search empty");
        assert!(results.is_empty());
    }

    #[test]
    fn saveLoadRoundTrip() {
        let dims = 32usize;
        let config = HnswConfig {
            m: 8,
            efConstruction: 50,
            efSearch: 32,
            metric: DistanceMetric::Euclidean,
        };

        let mut vectors: Vec<(VectorId, Vec<f32>)> = Vec::with_capacity(20);
        for i in 0..20u64 {
            let v: Vec<f32> = (0..dims)
                .map(|d| (i * 100 + d as u64) as f32 * 0.1)
                .collect();
            vectors.push((i, v));
        }

        let refs: Vec<(VectorId, &[f32])> =
            vectors.iter().map(|(id, v)| (*id, v.as_slice())).collect();
        let index = AnnIndex::build(&refs, 10, 2, 1, config).expect("build");

        let query = &vectors[5].1;
        let resultsBefore = index.search(query, 3, 32).expect("search before");

        let tmpDir = std::env::temp_dir();
        let filePath = tmpDir.join("zyron_hnsw_test.bin");
        index.saveToFile(&filePath).expect("save");

        let loaded = AnnIndex::loadFromFile(&filePath, 10, 2, 1).expect("load");
        let resultsAfter = loaded.search(query, 3, 32).expect("search after");

        assert_eq!(resultsBefore.len(), resultsAfter.len());
        for (before, after) in resultsBefore.iter().zip(resultsAfter.iter()) {
            assert_eq!(before.0, after.0);
            assert!((before.1 - after.1).abs() < 1e-6);
        }

        let _ = std::fs::remove_file(&filePath);
    }

    #[test]
    fn buildEmptyVectors() {
        let index = AnnIndex::build(&[], 5, 1, 0, makeConfig()).expect("empty build");
        assert_eq!(index.len(), 0);
    }
}
