//! HNSW (Hierarchical Navigable Small World) approximate nearest neighbor index.
//!
//! Edges live in a single flat Vec<u32> arena indexed via per-node offsets so
//! the build path does not pay for nested Vec allocations. Concurrent build
//! shares one BuildGraph with per-node Mutex and atomic edge cells with a
//! seqlock version counter for lock-free reads. Search uses a generation-counter
//! VisitedSet (dense Vec<u32> per node) for O(1) visited checks.

use std::cell::RefCell;
use std::cmp::Reverse;
use std::collections::BinaryHeap;
use std::io::{Read, Write};
use std::sync::atomic::{AtomicBool, AtomicU16, AtomicU32, AtomicU64, Ordering};

use parking_lot::RwLock;
use zyron_common::{Result, ZyronError};

thread_local! {
    /// Thread-local reusable visited set. Sized per query by calling prepare
    /// on the VisitedSet before each search.
    static SEARCH_VISITED: RefCell<VisitedSet> = RefCell::new(VisitedSet::new());
    /// Thread-local reusable beam search state so per-layer searches within a
    /// query and subsequent queries on the same thread reuse the same heaps.
    static SEARCH_STATE: RefCell<SearchState> = RefCell::new(SearchState::new(128));
    /// Thread-local scratch buffer for quantizing the query vector on the
    /// search path. Avoids a per-query allocation of a dims-sized Vec<u8>.
    static SEARCH_QUANT_BUF: RefCell<Vec<u8>> = RefCell::new(Vec::new());
}

use super::distance::{
    DistFn, computeQuantizationBounds, distWithFn, euclideanQuantized, quantizeArena,
    quantizeVector, resolveDistFn,
};
use super::memory::{try_alloc_vec, validate_file_size};
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

/// Random level drawn from the HNSW geometric distribution using the given
/// mutable RNG state. No atomics during build.
#[inline]
fn randomLevelFrom(rng: &mut u64, m: u16) -> usize {
    let _ = xorshift64(rng);
    let r = xorshift64(rng);
    let uniform = ((r & 0x000F_FFFF_FFFF_FFFF) as f64 + 1.0) / ((1u64 << 52) as f64);
    let ml = (m.max(2) as f64).ln();
    (-uniform.ln() / ml).floor() as usize
}

// ---------------------------------------------------------------------------
// Generation-counter visited set
// ---------------------------------------------------------------------------
// Dense per-node u32 buffer plus a generation counter. prepare bumps the
// counter, so clearing is O(1) until the counter wraps (then we zero the
// buffer). Cache-friendly at small N (the buffer fits in L2), simple, and
// has no fill-up edge cases.

struct VisitedSet {
    marks: Vec<u32>,
    epoch: u32,
}

impl VisitedSet {
    fn new() -> Self {
        Self {
            marks: Vec::new(),
            epoch: 0,
        }
    }

    /// Ensures the buffer can hold max_nodes entries and rolls the generation
    /// counter forward. Resets the buffer when the counter wraps to zero.
    fn prepare(&mut self, max_nodes: usize) {
        let needed = max_nodes.max(1);
        if self.marks.len() < needed {
            self.marks.resize(needed, 0);
        }
        self.epoch = self.epoch.wrapping_add(1);
        if self.epoch == 0 {
            self.epoch = 1;
            for m in self.marks.iter_mut() {
                *m = 0;
            }
        }
    }

    /// Marks idx as visited. Returns true on first observation in the current
    /// generation, false if already visited or out of range.
    #[inline]
    fn insert(&mut self, idx: u32) -> bool {
        let i = idx as usize;
        if i >= self.marks.len() {
            return false;
        }
        if self.marks[i] == self.epoch {
            false
        } else {
            self.marks[i] = self.epoch;
            true
        }
    }
}

// ---------------------------------------------------------------------------
// Flat arena helpers
// ---------------------------------------------------------------------------

#[inline]
fn arenaSlice(arena: &[f32], dims: usize, nodeIdx: usize) -> &[f32] {
    let offset = nodeIdx * dims;
    &arena[offset..offset + dims]
}

/// Reusable heap and output buffer for beam search.
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

/// Reusable heap and output buffer for u32 quantized beam search. Keeping
/// distances as u32 avoids the precision loss of a u32 to f32 cast above
/// 2^24, which occurs at roughly dims * 255^2 > 16.7M (dims >= 259).
struct SearchStateQ {
    candidates: BinaryHeap<Reverse<(u32, usize)>>,
    results: BinaryHeap<(u32, usize)>,
    output: Vec<(usize, u32)>,
}

impl SearchStateQ {
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

// ---------------------------------------------------------------------------
// NodeStore: flat arena holding all node connections for a finalized index
// ---------------------------------------------------------------------------
// Per-node slot layout for a node at level L:
//   layer 0: [len: u32, e_0, ..., e_{2m-1}]          -> (2m + 1) slots
//   layer l: [len: u32, e_0, ..., e_{m-1}]           -> (m + 1) slots  (1 <= l <= L)
// node_offsets[i] is the starting index in conn_data for node i.
// Total per-node size is (2m + 1) + L * (m + 1) u32 slots.

pub(crate) struct NodeStore {
    ids: Vec<VectorId>,
    levels: Vec<u8>,
    deleted: Vec<AtomicBool>,
    conn_data: Vec<u32>,
    node_offsets: Vec<u64>,
    m: u32,
}

impl NodeStore {
    fn new(m: u32) -> Self {
        Self {
            ids: Vec::new(),
            levels: Vec::new(),
            deleted: Vec::new(),
            conn_data: Vec::new(),
            node_offsets: Vec::new(),
            m,
        }
    }

    fn with_capacity(m: u32, node_cap: usize, conn_cap: usize) -> Result<Self> {
        let mut conn_data: Vec<u32> = Vec::new();
        if conn_cap > 0 {
            conn_data.try_reserve_exact(conn_cap).map_err(|_| {
                ZyronError::MemoryAllocationFailed {
                    bytes: (conn_cap as u64).saturating_mul(4),
                }
            })?;
        }
        let mut ids: Vec<VectorId> = Vec::new();
        if node_cap > 0 {
            ids.try_reserve_exact(node_cap)
                .map_err(|_| ZyronError::MemoryAllocationFailed {
                    bytes: (node_cap as u64).saturating_mul(8),
                })?;
        }
        Ok(Self {
            ids,
            levels: Vec::with_capacity(node_cap),
            deleted: Vec::with_capacity(node_cap),
            conn_data,
            node_offsets: Vec::with_capacity(node_cap),
            m,
        })
    }

    #[inline]
    pub(crate) fn len(&self) -> usize {
        self.ids.len()
    }

    #[inline]
    fn slots_per_node(m: u32, level: u8) -> usize {
        let mu = m as usize;
        (2 * mu + 1) + (level as usize) * (mu + 1)
    }

    #[inline]
    fn layer_offset(&self, node: usize, layer: usize) -> usize {
        let base = self.node_offsets[node] as usize;
        let mu = self.m as usize;
        if layer == 0 {
            base
        } else {
            base + (2 * mu + 1) + (layer - 1) * (mu + 1)
        }
    }

    #[inline]
    fn layer_cap(&self, layer: usize) -> usize {
        let mu = self.m as usize;
        if layer == 0 { 2 * mu } else { mu }
    }

    /// Edges at the given layer. Empty slice when the layer is above the node's level.
    #[inline]
    fn neighbors(&self, node: usize, layer: usize) -> &[u32] {
        if layer > self.levels[node] as usize {
            return &[];
        }
        let off = self.layer_offset(node, layer);
        let len = self.conn_data[off] as usize;
        &self.conn_data[off + 1..off + 1 + len]
    }

    /// Appends a new node with empty edge slots. Returns the new index.
    fn push_node(&mut self, id: VectorId, level: u8) -> Result<usize> {
        let slots = Self::slots_per_node(self.m, level);
        let base = self.conn_data.len() as u64;
        self.conn_data.try_reserve_exact(slots).map_err(|_| {
            ZyronError::MemoryAllocationFailed {
                bytes: (slots as u64).saturating_mul(4),
            }
        })?;
        for _ in 0..slots {
            self.conn_data.push(0);
        }
        let idx = self.ids.len();
        self.ids.push(id);
        self.levels.push(level);
        self.deleted.push(AtomicBool::new(false));
        self.node_offsets.push(base);
        Ok(idx)
    }

    fn set_neighbors(&mut self, node: usize, layer: usize, edges: &[u32]) {
        let cap = self.layer_cap(layer);
        let off = self.layer_offset(node, layer);
        let n = edges.len().min(cap);
        self.conn_data[off] = n as u32;
        for i in 0..n {
            self.conn_data[off + 1 + i] = edges[i];
        }
    }

    /// Appends one edge. Returns false when the slot is already at capacity.
    fn push_neighbor(&mut self, node: usize, layer: usize, edge: u32) -> bool {
        let cap = self.layer_cap(layer);
        let off = self.layer_offset(node, layer);
        let len = self.conn_data[off] as usize;
        if len >= cap {
            return false;
        }
        self.conn_data[off + 1 + len] = edge;
        self.conn_data[off] = (len + 1) as u32;
        true
    }
}

// ---------------------------------------------------------------------------
// Free-function search layers over NodeStore
// ---------------------------------------------------------------------------

/// Beam search at a single layer using reusable search state and a hash
/// visited set. Output is written to state.output sorted by distance ascending.
/// Takes a pre-resolved DistFn so the hot path has no per-call metric dispatch.
fn searchLayerReuse(
    distFn: DistFn,
    dims: usize,
    query: &[f32],
    entryIdx: usize,
    ef: usize,
    layer: usize,
    store: &NodeStore,
    arena: &[f32],
    visited: &mut VisitedSet,
    state: &mut SearchState,
) {
    state.clear();
    let entryDist = distWithFn(distFn, query, arenaSlice(arena, dims, entryIdx));
    state.candidates.push(Reverse(DistEntry {
        dist: entryDist,
        idx: entryIdx,
    }));
    state.results.push(DistEntry {
        dist: entryDist,
        idx: entryIdx,
    });
    visited.insert(entryIdx as u32);

    let nnodes = store.len();
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
        let neighbors = store.neighbors(current.idx, layer);
        if neighbors.is_empty() {
            continue;
        }

        let nlen = neighbors.len();
        for nPos in 0..nlen {
            let neighborIdx = neighbors[nPos];
            let ni = neighborIdx as usize;
            if ni >= nnodes || !visited.insert(neighborIdx) {
                continue;
            }

            #[cfg(target_arch = "x86_64")]
            {
                if nPos + 1 < nlen {
                    let nextNi = neighbors[nPos + 1] as usize;
                    if nextNi < nnodes {
                        // nextNi < nnodes implies nextNi * dims + dims <= arena.len()
                        // since arena holds exactly nnodes * dims f32s.
                        let prefetchOffset = nextNi * dims;
                        unsafe {
                            use std::arch::x86_64::*;
                            let base = arena.as_ptr().add(prefetchOffset) as *const i8;
                            let cacheLines = (dims * 4 + 63) / 64;
                            for cl in 0..cacheLines {
                                _mm_prefetch(base.add(cl * 64), _MM_HINT_T0);
                            }
                        }
                    }
                }
            }

            let dist = distWithFn(distFn, query, arenaSlice(arena, dims, ni));
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

/// Beam search at layer 0 over the u8 quantized arena. Returns unsorted pairs
/// for the caller to rerank with exact f32 distances.
fn searchLayerQuantized(
    dims: usize,
    queryQuantized: &[u8],
    entryIdx: usize,
    ef: usize,
    store: &NodeStore,
    quantizedArena: &[u8],
    visited: &mut VisitedSet,
) -> Vec<(usize, u32)> {
    let mut candidates: BinaryHeap<Reverse<(u32, usize)>> = BinaryHeap::with_capacity(ef + 16);
    let mut results: BinaryHeap<(u32, usize)> = BinaryHeap::with_capacity(ef + 16);

    let entrySlice = &quantizedArena[entryIdx * dims..(entryIdx + 1) * dims];
    let entryDist = euclideanQuantized(queryQuantized, entrySlice);

    candidates.push(Reverse((entryDist, entryIdx)));
    results.push((entryDist, entryIdx));
    visited.insert(entryIdx as u32);

    let nnodes = store.len();
    while let Some(Reverse((curDist, curIdx))) = candidates.pop() {
        if results.len() >= ef {
            let farthest = results.peek().map(|&(d, _)| d).unwrap_or(u32::MAX);
            if curDist > farthest {
                break;
            }
        }
        let neighbors = store.neighbors(curIdx, 0);
        for &neighborIdx in neighbors {
            let ni = neighborIdx as usize;
            if ni >= nnodes || !visited.insert(neighborIdx) {
                continue;
            }
            let ns = &quantizedArena[ni * dims..(ni + 1) * dims];
            let dist = euclideanQuantized(queryQuantized, ns);
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

    results.drain().map(|(d, idx)| (idx, d)).collect()
}

// ---------------------------------------------------------------------------
// Neighbor selection
// ---------------------------------------------------------------------------

/// Heuristic neighbor selection (HNSW Algorithm 4). Keeps diverse long-range
/// connections by skipping candidates dominated by an already selected one.
/// Input must be sorted ascending by distance. Results are written into
/// selectOut and rejectBuf is used as scratch space; both buffers are
/// cleared at entry so the caller can reuse them across calls with no
/// heap allocation in the steady state.
fn selectNeighborsHeuristic(
    sortedCandidates: &[(usize, f32)],
    m: usize,
    store: &NodeStore,
    arena: &[f32],
    dims: usize,
    distFn: DistFn,
    selectOut: &mut Vec<u32>,
    rejectBuf: &mut Vec<u32>,
) {
    selectOut.clear();
    rejectBuf.clear();

    for &(idx, dist) in sortedCandidates {
        if selectOut.len() >= m {
            break;
        }
        if store.deleted[idx].load(Ordering::Relaxed) {
            continue;
        }

        let mut dominated = false;
        let candidateVec = arenaSlice(arena, dims, idx);
        for &sel in selectOut.iter() {
            let interDist = distWithFn(distFn, candidateVec, arenaSlice(arena, dims, sel as usize));
            if interDist < dist {
                dominated = true;
                break;
            }
        }

        if !dominated {
            selectOut.push(idx as u32);
        } else {
            rejectBuf.push(idx as u32);
        }
    }

    for &idx in rejectBuf.iter() {
        if selectOut.len() >= m {
            break;
        }
        selectOut.push(idx);
    }
}

/// Closest-m selection without the domination check. Input must be sorted
/// ascending by distance. Results written into selectOut.
fn selectNeighborsSimple(
    sortedCandidates: &[(usize, f32)],
    m: usize,
    store: &NodeStore,
    selectOut: &mut Vec<u32>,
) {
    selectOut.clear();
    for &(idx, _) in sortedCandidates {
        if selectOut.len() >= m {
            break;
        }
        if !store.deleted[idx].load(Ordering::Relaxed) {
            selectOut.push(idx as u32);
        }
    }
}

/// HNSW Algorithm 4 heuristic on the concurrent build path. Matches
/// selectNeighborsHeuristic but does not check a deleted flag because nodes
/// are never deleted during build.
fn selectNeighborsBuild(
    sortedCandidates: &[(usize, f32)],
    m: usize,
    arena: &[f32],
    dims: usize,
    distFn: DistFn,
    selectOut: &mut Vec<u32>,
    rejectBuf: &mut Vec<u32>,
) {
    selectOut.clear();
    rejectBuf.clear();

    for &(idx, dist) in sortedCandidates {
        if selectOut.len() >= m {
            break;
        }
        let mut dominated = false;
        let candidateVec = arenaSlice(arena, dims, idx);
        for &sel in selectOut.iter() {
            let interDist = distWithFn(distFn, candidateVec, arenaSlice(arena, dims, sel as usize));
            if interDist < dist {
                dominated = true;
                break;
            }
        }
        if !dominated {
            selectOut.push(idx as u32);
        } else {
            rejectBuf.push(idx as u32);
        }
    }

    for &idx in rejectBuf.iter() {
        if selectOut.len() >= m {
            break;
        }
        selectOut.push(idx);
    }
}

// ---------------------------------------------------------------------------
// BuildGraph: shared flat arena used during concurrent build
// ---------------------------------------------------------------------------
// Mirrors NodeStore's slot layout but each edge cell is an AtomicU32 and each
// node has its own RwLock so writers serialise for a given node while parallel
// writers proceed across different nodes. Readers copy neighbor slices under
// the read lock so distance computation happens outside the critical section.

struct BuildGraph {
    ids: Vec<VectorId>,
    levels: Vec<u8>,
    conn_data: Vec<AtomicU32>,
    node_offsets: Vec<u64>,
    /// Writer serialization (only one writer per node at a time).
    write_locks: Vec<parking_lot::Mutex<()>>,
    /// Seqlock-style version counter per node. Writers bump the counter
    /// before and after their store sequence so readers can detect a
    /// concurrent write and retry.
    versions: Vec<AtomicU32>,
    m: u32,
}

impl BuildGraph {
    fn new(m: u32, levels: &[u8]) -> Result<Self> {
        let n = levels.len();
        let mut node_offsets: Vec<u64> = Vec::with_capacity(n);
        let mut total: u64 = 0;
        for &lvl in levels {
            node_offsets.push(total);
            total = total.saturating_add(NodeStore::slots_per_node(m, lvl) as u64);
        }
        let total_usize = total as usize;

        let mut conn_data: Vec<AtomicU32> = Vec::new();
        if total_usize > 0 {
            conn_data.try_reserve_exact(total_usize).map_err(|_| {
                ZyronError::MemoryAllocationFailed {
                    bytes: total.saturating_mul(4),
                }
            })?;
        }
        for _ in 0..total_usize {
            conn_data.push(AtomicU32::new(0));
        }

        let mut write_locks: Vec<parking_lot::Mutex<()>> = Vec::with_capacity(n);
        let mut versions: Vec<AtomicU32> = Vec::with_capacity(n);
        for _ in 0..n {
            write_locks.push(parking_lot::Mutex::new(()));
            versions.push(AtomicU32::new(0));
        }

        Ok(Self {
            ids: Vec::with_capacity(n),
            levels: levels.to_vec(),
            conn_data,
            node_offsets,
            write_locks,
            versions,
            m,
        })
    }

    #[inline]
    fn layer_cap(&self, layer: usize) -> usize {
        let mu = self.m as usize;
        if layer == 0 { 2 * mu } else { mu }
    }

    #[inline]
    fn layer_offset(&self, node: usize, layer: usize) -> usize {
        let base = self.node_offsets[node] as usize;
        let mu = self.m as usize;
        if layer == 0 {
            base
        } else {
            base + (2 * mu + 1) + (layer - 1) * (mu + 1)
        }
    }

    /// Snapshots the edge list at this layer using a seqlock pattern. No
    /// lock acquisition: the reader loads the version counter, copies edges
    /// via per-element Relaxed atomic loads, then re-reads the version and
    /// retries if a writer raced. Each individual u32 load is atomic so no
    /// torn reads occur on any architecture.
    fn read_neighbors(&self, node: usize, layer: usize, out: &mut Vec<u32>) {
        out.clear();
        if layer > self.levels[node] as usize {
            return;
        }
        let off = self.layer_offset(node, layer);
        let cap = self.layer_cap(layer);
        loop {
            let v1 = self.versions[node].load(Ordering::Acquire);
            if v1 & 1 != 0 {
                std::hint::spin_loop();
                continue;
            }
            let len = self.conn_data[off].load(Ordering::Acquire) as usize;
            if len > cap {
                std::hint::spin_loop();
                continue;
            }
            out.clear();
            out.reserve(len);
            for i in 0..len {
                out.push(self.conn_data[off + 1 + i].load(Ordering::Relaxed));
            }
            let v2 = self.versions[node].load(Ordering::Acquire);
            if v1 == v2 {
                return;
            }
            out.clear();
        }
    }

    /// Publishes a new edge list at this layer. Caller must hold this node's
    /// write_locks Mutex. Updates the version counter so concurrent readers
    /// can detect the write and retry their snapshot.
    fn write_neighbors_locked(&self, node: usize, layer: usize, edges: &[u32]) {
        let off = self.layer_offset(node, layer);
        let cap = self.layer_cap(layer);
        let n = edges.len().min(cap);
        // Bump version to odd to signal a writer is active.
        self.versions[node].fetch_add(1, Ordering::Release);
        for i in 0..n {
            self.conn_data[off + 1 + i].store(edges[i], Ordering::Relaxed);
        }
        self.conn_data[off].store(n as u32, Ordering::Release);
        // Bump version back to even to mark the write committed.
        self.versions[node].fetch_add(1, Ordering::Release);
    }

    /// Appends a single edge to the layer's slot if there is room. Returns
    /// the previous length on success, or None if the slot was full. Caller
    /// must hold this node's write_locks Mutex.
    fn try_append_edge_locked(&self, node: usize, layer: usize, edge: u32) -> Option<usize> {
        let off = self.layer_offset(node, layer);
        let cap = self.layer_cap(layer);
        let currentLen = self.conn_data[off].load(Ordering::Relaxed) as usize;
        if currentLen >= cap {
            return None;
        }
        self.versions[node].fetch_add(1, Ordering::Release);
        self.conn_data[off + 1 + currentLen].store(edge, Ordering::Relaxed);
        self.conn_data[off].store((currentLen + 1) as u32, Ordering::Release);
        self.versions[node].fetch_add(1, Ordering::Release);
        Some(currentLen)
    }
}

/// Beam search at a single layer over a BuildGraph. Reads neighbor lists
/// under the per-node read lock and computes distances outside the lock.
fn searchLayerBuild(
    distFn: DistFn,
    dims: usize,
    query: &[f32],
    entryIdx: usize,
    ef: usize,
    layer: usize,
    graph: &BuildGraph,
    arena: &[f32],
    visited: &mut VisitedSet,
    state: &mut SearchState,
    neighborBuf: &mut Vec<u32>,
) {
    state.clear();
    let entryDist = distWithFn(distFn, query, arenaSlice(arena, dims, entryIdx));
    state.candidates.push(Reverse(DistEntry {
        dist: entryDist,
        idx: entryIdx,
    }));
    state.results.push(DistEntry {
        dist: entryDist,
        idx: entryIdx,
    });
    visited.insert(entryIdx as u32);

    let nnodes = graph.levels.len();

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

        graph.read_neighbors(current.idx, layer, neighborBuf);
        if neighborBuf.is_empty() {
            continue;
        }

        let nlen = neighborBuf.len();
        for nPos in 0..nlen {
            let neighborIdx = neighborBuf[nPos];
            let ni = neighborIdx as usize;
            if ni >= nnodes || !visited.insert(neighborIdx) {
                continue;
            }

            #[cfg(target_arch = "x86_64")]
            {
                if nPos + 1 < nlen {
                    let nextNi = neighborBuf[nPos + 1] as usize;
                    if nextNi < nnodes {
                        // arena holds exactly nnodes * dims f32s, so nextNi < nnodes
                        // guarantees the prefetch range is in bounds.
                        let prefetchOffset = nextNi * dims;
                        unsafe {
                            use std::arch::x86_64::*;
                            let base = arena.as_ptr().add(prefetchOffset) as *const i8;
                            let cacheLines = (dims * 4 + 63) / 64;
                            for cl in 0..cacheLines {
                                _mm_prefetch(base.add(cl * 64), _MM_HINT_T0);
                            }
                        }
                    }
                }
            }

            let dist = distWithFn(distFn, query, arenaSlice(arena, dims, ni));
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

/// Quantized beam search at a single layer over a BuildGraph for Euclidean
/// metric. Distances are computed against the u8 quantized arena, giving 4x
/// less memory bandwidth than f32 distances. Distances are kept as u32
/// throughout so the heap ordering is precision-preserving for any realistic
/// dims (a direct u32 to f32 cast loses precision above 2^24). The caller
/// reranks the output with exact f32 distances before neighbor selection.
fn searchLayerBuildQuantized(
    dims: usize,
    queryQuantized: &[u8],
    entryIdx: usize,
    ef: usize,
    layer: usize,
    graph: &BuildGraph,
    quantizedArena: &[u8],
    visited: &mut VisitedSet,
    state: &mut SearchStateQ,
    neighborBuf: &mut Vec<u32>,
) {
    state.clear();
    let entryQ = &quantizedArena[entryIdx * dims..(entryIdx + 1) * dims];
    let entryDist = euclideanQuantized(queryQuantized, entryQ);
    state.candidates.push(Reverse((entryDist, entryIdx)));
    state.results.push((entryDist, entryIdx));
    visited.insert(entryIdx as u32);

    let nnodes = graph.levels.len();

    while let Some(Reverse((curDist, curIdx))) = state.candidates.pop() {
        if state.results.len() >= ef {
            let farthest = state.results.peek().map(|&(d, _)| d).unwrap_or(u32::MAX);
            if curDist > farthest {
                break;
            }
        }

        graph.read_neighbors(curIdx, layer, neighborBuf);
        if neighborBuf.is_empty() {
            continue;
        }

        let nlen = neighborBuf.len();
        for nPos in 0..nlen {
            let neighborIdx = neighborBuf[nPos];
            let ni = neighborIdx as usize;
            if ni >= nnodes || !visited.insert(neighborIdx) {
                continue;
            }

            #[cfg(target_arch = "x86_64")]
            {
                if nPos + 1 < nlen {
                    let nextNi = neighborBuf[nPos + 1] as usize;
                    if nextNi < nnodes {
                        let prefetchOffset = nextNi * dims;
                        unsafe {
                            use std::arch::x86_64::*;
                            let base = quantizedArena.as_ptr().add(prefetchOffset) as *const i8;
                            let cacheLines = (dims + 63) / 64;
                            for cl in 0..cacheLines {
                                _mm_prefetch(base.add(cl * 64), _MM_HINT_T0);
                            }
                        }
                    }
                }
            }

            let nQ = &quantizedArena[ni * dims..(ni + 1) * dims];
            let dist = euclideanQuantized(queryQuantized, nQ);
            let farthest = state.results.peek().map(|&(d, _)| d).unwrap_or(u32::MAX);

            if dist < farthest || state.results.len() < ef {
                state.candidates.push(Reverse((dist, ni)));
                state.results.push((dist, ni));
                if state.results.len() > ef {
                    state.results.pop();
                }
            }
        }
    }

    state.output.clear();
    state
        .output
        .extend(state.results.drain().map(|(d, idx)| (idx, d)));
    state.output.sort_unstable_by_key(|&(_, d)| d);
}

// ---------------------------------------------------------------------------
// Concurrent build (shared BuildGraph with per-node locking)
// ---------------------------------------------------------------------------
// Build requirements for high graph quality regardless of thread count:
//   1. Pruning uses the HNSW Algorithm 4 heuristic (selectNeighborsBuild) so
//      long-range connections survive overflow pruning.
//   2. Every insert uses the full efConstruction so late inserts do not
//      settle for locally-nearby but globally-weak neighbors.
//   3. Vectors are inserted in caller-provided order so temporal locality
//      stays intact for clustered data.
//   4. m and efConstruction are chosen by the auto-tuner against the
//      profile so the heuristic does not reject valid neighbors.

fn concurrentBuild(
    vectors: &[(VectorId, &[f32])],
    dims: usize,
    config: &HnswConfig,
    nThreads: usize,
    indexId: u32,
) -> Result<(
    NodeStore,
    Vec<f32>,
    Vec<u8>,
    Vec<f32>,
    Vec<f32>,
    Vec<(VectorId, usize)>,
    u64,
    u16,
)> {
    let n = vectors.len();
    let m_u = config.m as usize;
    let m_u32 = config.m as u32;
    let metric = config.metric;
    let distFn = resolveDistFn(metric);
    let efConstruction = config.efConstruction as usize;

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

    // Quantize upfront for Euclidean so the build's beam search can use cheap
    // u8 distances instead of f32. Other metrics keep the f32 path because
    // u8 quantization does not preserve their relative ordering.
    let useQuantizedBuild = metric == DistanceMetric::Euclidean;
    let (qMins, qMaxs) = computeQuantizationBounds(&arena, dims);
    let (quantizedArena, qScales) = quantizeArena(&arena, dims, &qMins, &qMaxs);

    let mut rng = (indexId as u64)
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1);
    if rng == 0 {
        rng = 1;
    }
    let mut levels: Vec<u8> = Vec::with_capacity(n);
    for _ in 0..n {
        let lvl = randomLevelFrom(&mut rng, config.m);
        levels.push(lvl.min(u8::MAX as usize) as u8);
    }

    let mut graph = BuildGraph::new(m_u32, &levels)?;
    for &(id, _) in vectors {
        graph.ids.push(id);
    }

    let entryPoint = AtomicU64::new(0);
    let maxLayer = AtomicU16::new(levels[0] as u16);

    let bootstrapCount = n.min(100);

    let insertOne = |insertIdx: usize,
                     visited: &mut VisitedSet,
                     searchState: &mut SearchState,
                     searchStateQ: &mut SearchStateQ,
                     neighborBuf: &mut Vec<u32>,
                     pruneBuf: &mut Vec<(usize, f32)>,
                     queryQuantBuf: &mut Vec<u8>,
                     selectBuf: &mut Vec<u32>,
                     rejectBuf: &mut Vec<u32>,
                     pruneOut: &mut Vec<u32>| {
        if insertIdx == 0 {
            return;
        }
        let vec = arenaSlice(&arena, dims, insertIdx);
        let level = levels[insertIdx] as usize;

        // Quantize the query vector once per insert when the build path is
        // running u8 distances. Cost is amortized across every layer search
        // performed for this insert.
        if useQuantizedBuild {
            if queryQuantBuf.len() != dims {
                queryQuantBuf.resize(dims, 0);
            }
            quantizeVector(vec, &qMins, &qScales, queryQuantBuf);
        }

        let currentMaxLayer = maxLayer.load(Ordering::Acquire) as usize;
        let mut ep = entryPoint.load(Ordering::Acquire) as usize;

        if level < currentMaxLayer {
            for l in (level + 1..=currentMaxLayer).rev() {
                visited.prepare(n);
                if useQuantizedBuild {
                    searchLayerBuildQuantized(
                        dims,
                        queryQuantBuf,
                        ep,
                        1,
                        l,
                        &graph,
                        &quantizedArena,
                        visited,
                        searchStateQ,
                        neighborBuf,
                    );
                    if let Some(&(closest, _)) = searchStateQ.output.first() {
                        ep = closest;
                    }
                } else {
                    searchLayerBuild(
                        distFn,
                        dims,
                        vec,
                        ep,
                        1,
                        l,
                        &graph,
                        &arena,
                        visited,
                        searchState,
                        neighborBuf,
                    );
                    if let Some(&(closest, _)) = searchState.output.first() {
                        ep = closest;
                    }
                }
            }
        }

        let topConnectLayer = level.min(currentMaxLayer);
        for l in (0..=topConnectLayer).rev() {
            visited.prepare(n);
            if useQuantizedBuild {
                searchLayerBuildQuantized(
                    dims,
                    queryQuantBuf,
                    ep,
                    efConstruction,
                    l,
                    &graph,
                    &quantizedArena,
                    visited,
                    searchStateQ,
                    neighborBuf,
                );
                // Rerank quantized candidates with exact f32 distances so
                // selectNeighborsBuild's domination check operates on real
                // distances, preserving graph quality. Results land in
                // searchState.output sorted ascending by exact distance.
                searchState.output.clear();
                searchState.output.reserve(searchStateQ.output.len());
                for &(idx, _) in searchStateQ.output.iter() {
                    let d = distWithFn(distFn, vec, arenaSlice(&arena, dims, idx));
                    searchState.output.push((idx, d));
                }
                searchState
                    .output
                    .sort_unstable_by(|a, b| a.1.total_cmp(&b.1));
            } else {
                searchLayerBuild(
                    distFn,
                    dims,
                    vec,
                    ep,
                    efConstruction,
                    l,
                    &graph,
                    &arena,
                    visited,
                    searchState,
                    neighborBuf,
                );
            }
            let candidates = &searchState.output;
            if let Some(&(closest, _)) = candidates.first() {
                ep = closest;
            }

            let maxConn = if l == 0 { m_u * 2 } else { m_u };
            // searchState.output is already sorted ascending by distance.
            selectNeighborsBuild(
                candidates, maxConn, &arena, dims, distFn, selectBuf, rejectBuf,
            );

            {
                let _g = graph.write_locks[insertIdx].lock();
                graph.write_neighbors_locked(insertIdx, l, selectBuf);
            }

            // Snapshot the chosen neighbors before the reverse-edge loop so
            // mutating selectBuf inside the loop (the prune path writes to
            // pruneOut and does not touch selectBuf) stays valid. The copy
            // is small: at most 2m entries.
            neighborBuf.clear();
            neighborBuf.extend_from_slice(selectBuf);

            for &neighborIdx in neighborBuf.iter() {
                let ni = neighborIdx as usize;
                if l > graph.levels[ni] as usize {
                    continue;
                }
                let _g = graph.write_locks[ni].lock();
                if graph
                    .try_append_edge_locked(ni, l, insertIdx as u32)
                    .is_none()
                {
                    // Slot is full. Snapshot current edges, prune via heuristic.
                    let cap = if l == 0 { m_u * 2 } else { m_u };
                    let off = graph.layer_offset(ni, l);
                    let currentLen = graph.conn_data[off].load(Ordering::Relaxed) as usize;
                    pruneBuf.clear();
                    pruneBuf.reserve(currentLen + 1);
                    let ni_vec = arenaSlice(&arena, dims, ni);
                    for i in 0..currentLen {
                        let cidx = graph.conn_data[off + 1 + i].load(Ordering::Relaxed) as usize;
                        let d = distWithFn(distFn, ni_vec, arenaSlice(&arena, dims, cidx));
                        pruneBuf.push((cidx, d));
                    }
                    let d_new = distWithFn(distFn, ni_vec, arenaSlice(&arena, dims, insertIdx));
                    pruneBuf.push((insertIdx, d_new));
                    // Sort in place so selectNeighborsBuild can skip its
                    // internal sort and operate on the existing buffer.
                    pruneBuf.sort_unstable_by(|a, b| a.1.total_cmp(&b.1));
                    selectNeighborsBuild(pruneBuf, cap, &arena, dims, distFn, pruneOut, rejectBuf);
                    graph.write_neighbors_locked(ni, l, pruneOut);
                }
            }
        }

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

    {
        let mut visited = VisitedSet::new();
        let mut searchState = SearchState::new(efConstruction + 16);
        let mut searchStateQ = SearchStateQ::new(efConstruction + 16);
        let mut neighborBuf: Vec<u32> = Vec::with_capacity(m_u * 2 + 16);
        let mut pruneBuf: Vec<(usize, f32)> = Vec::with_capacity(m_u * 2 + 16);
        let mut queryQuantBuf: Vec<u8> = Vec::with_capacity(dims);
        let mut selectBuf: Vec<u32> = Vec::with_capacity(m_u * 2 + 16);
        let mut rejectBuf: Vec<u32> = Vec::with_capacity(m_u * 2 + 16);
        let mut pruneOut: Vec<u32> = Vec::with_capacity(m_u * 2 + 16);
        for insertIdx in 1..bootstrapCount {
            insertOne(
                insertIdx,
                &mut visited,
                &mut searchState,
                &mut searchStateQ,
                &mut neighborBuf,
                &mut pruneBuf,
                &mut queryQuantBuf,
                &mut selectBuf,
                &mut rejectBuf,
                &mut pruneOut,
            );
        }
    }

    let nextInsert = AtomicU64::new(bootstrapCount as u64);
    let insertOneRef = &insertOne;
    std::thread::scope(|s| -> Result<()> {
        let mut handles = Vec::with_capacity(nThreads);
        for _ in 0..nThreads {
            let nextRef = &nextInsert;
            handles.push(s.spawn(move || -> Result<()> {
                let mut visited = VisitedSet::new();
                let mut searchState = SearchState::new(efConstruction + 16);
                let mut searchStateQ = SearchStateQ::new(efConstruction + 16);
                let mut neighborBuf: Vec<u32> = Vec::with_capacity(m_u * 2 + 16);
                let mut pruneBuf: Vec<(usize, f32)> = Vec::with_capacity(m_u * 2 + 16);
                let mut queryQuantBuf: Vec<u8> = Vec::with_capacity(dims);
                let mut selectBuf: Vec<u32> = Vec::with_capacity(m_u * 2 + 16);
                let mut rejectBuf: Vec<u32> = Vec::with_capacity(m_u * 2 + 16);
                let mut pruneOut: Vec<u32> = Vec::with_capacity(m_u * 2 + 16);
                loop {
                    let insertIdx = nextRef.fetch_add(1, Ordering::AcqRel) as usize;
                    if insertIdx >= n {
                        break;
                    }
                    insertOneRef(
                        insertIdx,
                        &mut visited,
                        &mut searchState,
                        &mut searchStateQ,
                        &mut neighborBuf,
                        &mut pruneBuf,
                        &mut queryQuantBuf,
                        &mut selectBuf,
                        &mut rejectBuf,
                        &mut pruneOut,
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

    // Materialise NodeStore from BuildGraph by copying atomic cells into plain u32.
    let ep = entryPoint.load(Ordering::Acquire);
    let ml = maxLayer.load(Ordering::Acquire);

    let total_slots = graph.conn_data.len();
    let mut store = NodeStore::with_capacity(m_u32, n, total_slots)?;
    let mut idPairs: Vec<(VectorId, usize)> = Vec::with_capacity(n);
    for i in 0..n {
        let idx = store.push_node(graph.ids[i], graph.levels[i])?;
        idPairs.push((graph.ids[i], idx));
        let max_l = graph.levels[i] as usize;
        for l in 0..=max_l {
            let src_off = graph.layer_offset(i, l);
            let dst_off = store.layer_offset(idx, l);
            let len = graph.conn_data[src_off].load(Ordering::Relaxed) as usize;
            store.conn_data[dst_off] = len as u32;
            for j in 0..len {
                store.conn_data[dst_off + 1 + j] =
                    graph.conn_data[src_off + 1 + j].load(Ordering::Relaxed);
            }
        }
    }

    Ok((
        store,
        arena,
        quantizedArena,
        qMins,
        qScales,
        idPairs,
        ep,
        ml,
    ))
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
    nodes: RwLock<NodeStore>,
    vectorArena: RwLock<Vec<f32>>,
    /// u8 scalar-quantized vector arena. Each dimension is quantized to
    /// [0, 255] using per-dimension min/max scaling, giving 4x lower memory
    /// bandwidth than the f32 arena for the beam search hot path.
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
        let m = config.m as u32;
        Self {
            indexId,
            tableId,
            columnId,
            config,
            dimensions,
            nodes: RwLock::new(NodeStore::new(m)),
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

    /// Parallel batch build. Profiles a sample of the input to drive auto-tuned
    /// parameters, then runs a single concurrent build over a shared flat
    /// BuildGraph. Small inputs fall back to the single-thread path.
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

        let sampleSize = n.min(1000);
        let sampleSlices: Vec<&[f32]> = vectors[..sampleSize].iter().map(|(_, v)| *v).collect();
        let profile = DataProfile::compute(&sampleSlices, n, dimensions, config.metric);

        // Use 75% of available cores for the build so the rest of the
        // database (query executor, WAL writer, connection acceptor, and
        // background workers) keeps making progress while an index builds.
        // Scales: 4 cores -> 3 workers, 8 -> 6, 16 -> 12, 32 -> 24.
        let nThreads = std::thread::available_parallelism()
            .map(|p| (p.get() * 3 / 4).max(1))
            .unwrap_or(1)
            .min(n);
        let parallelThreshold = profile.parallelThreshold();

        if n < parallelThreshold || nThreads <= 1 {
            return Self::buildSingleThreadWithProfile(
                vectors, indexId, tableId, columnId, config, profile,
            );
        }

        let (
            store,
            mergedArena,
            mergedQuantized,
            qMins,
            qScales,
            mergedIdPairs,
            globalEntryPoint,
            globalMaxLayer,
        ) = concurrentBuild(vectors, dims, &config, nThreads, indexId)?;
        let totalNodes = store.len();

        let idMap: scc::HashMap<VectorId, usize> = scc::HashMap::new();
        for &(vid, idx) in &mergedIdPairs {
            let _ = idMap.insert_sync(vid, idx);
        }

        let rngSeed = (indexId as u64)
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1);
        Ok(Self {
            indexId,
            tableId,
            columnId,
            config,
            dimensions,
            nodes: RwLock::new(store),
            vectorArena: RwLock::new(mergedArena),
            quantizedArena: RwLock::new(mergedQuantized),
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
        let m_u32 = config.m as u32;
        let efConstruction = config.efConstruction as usize;
        let metric = config.metric;
        let distFn = resolveDistFn(metric);

        let n = vectors.len();
        let mut arena: Vec<f32> = try_alloc_vec(n * dims)?;
        // Preallocate the conn_data arena to the expected upper bound assuming
        // every node gets one level. Real builds draw smaller levels so the
        // Vec will end up shorter but never reallocates during the hot path.
        let preconn = n.saturating_mul(NodeStore::slots_per_node(m_u32, 1));
        let mut store = NodeStore::with_capacity(m_u32, n, preconn)?;
        let mut visited = VisitedSet::new();
        let mut entryPoint: u64 = u64::MAX;
        let mut maxLayer: u16 = 0;
        let mut rng = (indexId as u64)
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1);
        if rng == 0 {
            rng = 1;
        }
        let mut idVec: Vec<(VectorId, usize)> = Vec::with_capacity(n);
        let mut searchState = SearchState::new(efConstruction + 16);
        let mut pruneBuf: Vec<(usize, f32)> = Vec::with_capacity(m * 2 + 16);
        let mut selectBuf: Vec<u32> = Vec::with_capacity(m * 2 + 16);
        let mut rejectBuf: Vec<u32> = Vec::with_capacity(m * 2 + 16);
        let mut pruneOut: Vec<u32> = Vec::with_capacity(m * 2 + 16);
        let mut neighborBuf: Vec<u32> = Vec::with_capacity(m * 2 + 16);

        for &(id, vec) in vectors {
            if vec.len() != dims {
                return Err(ZyronError::InvalidParameter {
                    name: "vector.len()".to_string(),
                    value: format!("expected {} dimensions, got {}", dims, vec.len()),
                });
            }

            let level = randomLevelFrom(&mut rng, config.m).min(u8::MAX as usize) as u8;
            let newIdx = store.push_node(id, level)?;
            arena.extend_from_slice(vec);
            idVec.push((id, newIdx));

            if entryPoint == u64::MAX {
                entryPoint = newIdx as u64;
                maxLayer = level as u16;
                continue;
            }

            let mut ep = entryPoint as usize;
            let currentMaxLayer = maxLayer as usize;
            let ef = config.efConstruction as usize;

            let currentNodeCount = store.len();
            if (level as usize) < currentMaxLayer {
                for l in (level as usize + 1..=currentMaxLayer).rev() {
                    visited.prepare(currentNodeCount);
                    searchLayerReuse(
                        distFn,
                        dims,
                        vec,
                        ep,
                        1,
                        l,
                        &store,
                        &arena,
                        &mut visited,
                        &mut searchState,
                    );
                    if let Some(&(closest, _)) = searchState.output.first() {
                        ep = closest;
                    }
                }
            }

            let topConnectLayer = (level as usize).min(currentMaxLayer);
            for l in (0..=topConnectLayer).rev() {
                visited.prepare(currentNodeCount);
                searchLayerReuse(
                    distFn,
                    dims,
                    vec,
                    ep,
                    ef,
                    l,
                    &store,
                    &arena,
                    &mut visited,
                    &mut searchState,
                );
                // Snapshot candidates before mutating store to satisfy the borrow checker.
                let candidates: Vec<(usize, f32)> = searchState.output.clone();
                if let Some(&(closest, _)) = candidates.first() {
                    ep = closest;
                }

                let maxConn = if l == 0 { m * 2 } else { m };
                // searchState.output is already sorted ascending.
                selectNeighborsHeuristic(
                    &candidates,
                    maxConn,
                    &store,
                    &arena,
                    dims,
                    distFn,
                    &mut selectBuf,
                    &mut rejectBuf,
                );
                store.set_neighbors(newIdx, l, &selectBuf);
                neighborBuf.clear();
                neighborBuf.extend_from_slice(&selectBuf);

                for &neighborIdx in neighborBuf.iter() {
                    let ni = neighborIdx as usize;
                    if l > store.levels[ni] as usize {
                        continue;
                    }
                    if !store.push_neighbor(ni, l, newIdx as u32) {
                        // Slot is full. Collect current edges and prune with
                        // Algorithm 4 heuristic so long-range connections survive.
                        pruneBuf.clear();
                        let ni_vec = arenaSlice(&arena, dims, ni);
                        {
                            let edges = store.neighbors(ni, l);
                            pruneBuf.reserve(edges.len() + 1);
                            for &cidx in edges {
                                let d = distWithFn(
                                    distFn,
                                    ni_vec,
                                    arenaSlice(&arena, dims, cidx as usize),
                                );
                                pruneBuf.push((cidx as usize, d));
                            }
                        }
                        let d_new = distWithFn(distFn, ni_vec, arenaSlice(&arena, dims, newIdx));
                        pruneBuf.push((newIdx, d_new));
                        pruneBuf.sort_unstable_by(|a, b| a.1.total_cmp(&b.1));
                        selectNeighborsHeuristic(
                            &pruneBuf,
                            maxConn,
                            &store,
                            &arena,
                            dims,
                            distFn,
                            &mut pruneOut,
                            &mut rejectBuf,
                        );
                        store.set_neighbors(ni, l, &pruneOut);
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
            nodes: RwLock::new(store),
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

    /// Level drawn atomically for online inserts.
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

    pub fn profile(&self) -> Option<&DataProfile> {
        self.profile.as_ref()
    }

    /// Online single-vector insert. Acquires write locks on the store and
    /// vector arena for the duration of the call.
    fn insertInternal(&self, id: VectorId, vector: &[f32]) -> Result<()> {
        let dims = self.dimensions as usize;
        if vector.len() != dims {
            return Err(ZyronError::InvalidParameter {
                name: "vector.len()".to_string(),
                value: format!("expected {} dimensions, got {}", dims, vector.len()),
            });
        }

        let level_usize = self.randomLevel().min(u8::MAX as usize);
        let level = level_usize as u8;
        let m = self.config.m as usize;
        let efConstruction = self.config.efConstruction as usize;
        let metric = self.config.metric;
        let distFn = resolveDistFn(metric);

        let mut storeGuard = self.nodes.write();
        let mut arenaGuard = self.vectorArena.write();
        let newIdx = storeGuard.push_node(id, level)?;
        arenaGuard.extend_from_slice(vector);

        if !self.quantMins.is_empty() && self.quantMins.len() == dims {
            let mut quantGuard = self.quantizedArena.write();
            SEARCH_QUANT_BUF.with(|cell| {
                let mut quantBuf = cell.borrow_mut();
                if quantBuf.len() != dims {
                    quantBuf.resize(dims, 0);
                }
                quantizeVector(vector, &self.quantMins, &self.quantScales, &mut quantBuf);
                quantGuard.extend_from_slice(&quantBuf);
            });
        }

        let _ = self.idMap.insert_sync(id, newIdx);

        let currentEntry = self.entryPoint.load(Ordering::Acquire);
        if currentEntry == u64::MAX {
            self.entryPoint.store(newIdx as u64, Ordering::Release);
            self.maxLayer.store(level as u16, Ordering::Release);
            self.nodeCount.fetch_add(1, Ordering::Relaxed);
            return Ok(());
        }

        // Hoist visited set and SearchState across all layer searches so
        // per-layer calls reuse the same backing storage.
        let mut visited = VisitedSet::new();
        let mut searchState = SearchState::new(efConstruction + 16);
        let mut selectBuf: Vec<u32> = Vec::with_capacity(m * 2 + 16);
        let mut rejectBuf: Vec<u32> = Vec::with_capacity(m * 2 + 16);
        let mut pruneOut: Vec<u32> = Vec::with_capacity(m * 2 + 16);
        let mut neighborBuf: Vec<u32> = Vec::with_capacity(m * 2 + 16);
        let mut connWithDist: Vec<(usize, f32)> = Vec::with_capacity(m * 2 + 16);
        let mut ep = currentEntry as usize;
        let currentMaxLayer = self.maxLayer.load(Ordering::Acquire) as usize;
        let currentNodeCount = storeGuard.len();

        if (level as usize) < currentMaxLayer {
            for l in (level as usize + 1..=currentMaxLayer).rev() {
                visited.prepare(currentNodeCount);
                searchLayerReuse(
                    distFn,
                    dims,
                    vector,
                    ep,
                    1,
                    l,
                    &storeGuard,
                    &arenaGuard,
                    &mut visited,
                    &mut searchState,
                );
                if let Some(&(closest, _)) = searchState.output.first() {
                    ep = closest;
                }
            }
        }

        let topConnectLayer = (level as usize).min(currentMaxLayer);
        for l in (0..=topConnectLayer).rev() {
            visited.prepare(currentNodeCount);
            searchLayerReuse(
                distFn,
                dims,
                vector,
                ep,
                efConstruction,
                l,
                &storeGuard,
                &arenaGuard,
                &mut visited,
                &mut searchState,
            );
            // Snapshot the sorted output before store mutation.
            let candidates: Vec<(usize, f32)> = searchState.output.clone();

            if let Some(&(closest, _)) = candidates.first() {
                ep = closest;
            }

            let maxConn = if l == 0 { m * 2 } else { m };
            // searchState.output is already sorted ascending by distance.
            selectNeighborsHeuristic(
                &candidates,
                maxConn,
                &storeGuard,
                &arenaGuard,
                dims,
                distFn,
                &mut selectBuf,
                &mut rejectBuf,
            );
            storeGuard.set_neighbors(newIdx, l, &selectBuf);
            neighborBuf.clear();
            neighborBuf.extend_from_slice(&selectBuf);

            for &neighborIdx in neighborBuf.iter() {
                let ni = neighborIdx as usize;
                if l > storeGuard.levels[ni] as usize {
                    continue;
                }
                if !storeGuard.push_neighbor(ni, l, newIdx as u32) {
                    connWithDist.clear();
                    {
                        let edges = storeGuard.neighbors(ni, l);
                        connWithDist.reserve(edges.len());
                        let ni_vec = arenaSlice(&arenaGuard, dims, ni);
                        for &cidx in edges {
                            let d = distWithFn(
                                distFn,
                                ni_vec,
                                arenaSlice(&arenaGuard, dims, cidx as usize),
                            );
                            connWithDist.push((cidx as usize, d));
                        }
                    }
                    connWithDist.sort_unstable_by(|a, b| a.1.total_cmp(&b.1));
                    selectNeighborsSimple(&connWithDist, maxConn, &storeGuard, &mut pruneOut);
                    storeGuard.set_neighbors(ni, l, &pruneOut);
                }
            }
        }

        if level as u16 > currentMaxLayer as u16 {
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
        let storeGuard = self.nodes.read();
        let arenaGuard = self.vectorArena.read();
        let mut file = std::fs::File::create(path)?;
        let dims = self.dimensions as usize;

        file.write_all(b"ZYVEC\x01")?;
        file.write_all(&self.dimensions.to_le_bytes())?;
        file.write_all(&(storeGuard.len() as u64).to_le_bytes())?;

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

        for i in 0..storeGuard.len() {
            file.write_all(&storeGuard.ids[i].to_le_bytes())?;
            let deleted = if storeGuard.deleted[i].load(Ordering::Relaxed) {
                1u8
            } else {
                0u8
            };
            file.write_all(&[deleted])?;

            let vecStart = i * dims;
            for j in 0..dims {
                file.write_all(&arenaGuard[vecStart + j].to_le_bytes())?;
            }

            let level = storeGuard.levels[i] as usize;
            let numLayers = (level + 1) as u16;
            file.write_all(&numLayers.to_le_bytes())?;
            for l in 0..=level {
                let edges = storeGuard.neighbors(i, l);
                file.write_all(&(edges.len() as u32).to_le_bytes())?;
                for &conn in edges {
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
        let mut buf4 = [0u8; 4];
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

        // Validate declared sizes before allocating anything sized by file input.
        let declared_arena_bytes = (nodeCount as u64)
            .saturating_mul(dimensions as u64)
            .saturating_mul(4);
        // Worst-case per node: (2m + 1) + 32 * (m + 1) slots at 4 bytes each.
        let max_slots_per_node = (2 * m as u64 + 1) + 32 * (m as u64 + 1);
        let declared_conn_bytes = (nodeCount as u64)
            .saturating_mul(max_slots_per_node)
            .saturating_mul(4);
        let total_declared = declared_arena_bytes.saturating_add(declared_conn_bytes);
        validate_file_size(total_declared)?;

        let m_u32 = m as u32;
        let mut store = NodeStore::with_capacity(m_u32, nodeCount as usize, 0)?;
        let mut vectorArena: Vec<f32> = try_alloc_vec(nodeCount as usize * dimensions as usize)?;
        let idMap = scc::HashMap::new();

        for i in 0..nodeCount as usize {
            file.read_exact(&mut buf8)?;
            let id = u64::from_le_bytes(buf8);

            let mut deletedBuf = [0u8; 1];
            file.read_exact(&mut deletedBuf)?;
            let deleted = deletedBuf[0] != 0;

            for _ in 0..dimensions {
                file.read_exact(&mut buf4)?;
                vectorArena.push(f32::from_le_bytes(buf4));
            }

            file.read_exact(&mut buf2)?;
            let numLayers = u16::from_le_bytes(buf2) as usize;
            if numLayers == 0 || numLayers > 64 {
                return Err(ZyronError::VectorIndexFileCorrupt {
                    declared: numLayers as u64,
                });
            }
            let level = (numLayers - 1) as u8;
            let idx = store.push_node(id, level)?;
            debug_assert_eq!(idx, i);
            store.deleted[i].store(deleted, Ordering::Relaxed);

            for l in 0..numLayers {
                file.read_exact(&mut buf4)?;
                let numConns = u32::from_le_bytes(buf4) as usize;
                if numConns > 4096 {
                    return Err(ZyronError::VectorIndexFileCorrupt {
                        declared: numConns as u64,
                    });
                }
                let mut edges: Vec<u32> = Vec::with_capacity(numConns);
                for _ in 0..numConns {
                    file.read_exact(&mut buf4)?;
                    edges.push(u32::from_le_bytes(buf4));
                }
                store.set_neighbors(i, l, &edges);
            }

            let _ = idMap.insert_sync(id, i);
        }

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
            nodes: RwLock::new(store),
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
// VectorSearch trait implementation
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

        let storeGuard = self.nodes.read();
        let arenaGuard = self.vectorArena.read();
        if storeGuard.len() == 0 {
            return Ok(Vec::new());
        }

        let metric = self.config.metric;
        let distFn = resolveDistFn(metric);
        let mut currentEp = ep as usize;
        let currentMaxLayer = self.maxLayer.load(Ordering::Acquire) as usize;

        let effectiveEf = if efSearch == 0 {
            self.config.efSearch
        } else {
            efSearch
        };
        let ef = (effectiveEf as usize).max(k);

        let nodeCount = storeGuard.len();
        let result = SEARCH_VISITED.with(|vcell| {
            SEARCH_STATE.with(|scell| {
                let mut visited = vcell.borrow_mut();
                let mut state = scell.borrow_mut();

                for l in (1..=currentMaxLayer).rev() {
                    visited.prepare(nodeCount);
                    searchLayerReuse(
                        distFn,
                        dims,
                        query,
                        currentEp,
                        1,
                        l,
                        &storeGuard,
                        &arenaGuard,
                        &mut visited,
                        &mut state,
                    );
                    if let Some(&(closest, _)) = state.output.first() {
                        currentEp = closest;
                    }
                }

                visited.prepare(nodeCount);

                // Quantized Euclidean search is only valid when the stored
                // arena and bounds match the query metric. Other metrics
                // fall through to the exact f32 path.
                let quantGuard = self.quantizedArena.read();
                if !quantGuard.is_empty()
                    && !self.quantMins.is_empty()
                    && metric == DistanceMetric::Euclidean
                {
                    let quantResults = SEARCH_QUANT_BUF.with(|cell| {
                        let mut queryQ = cell.borrow_mut();
                        if queryQ.len() != dims {
                            queryQ.resize(dims, 0);
                        }
                        quantizeVector(query, &self.quantMins, &self.quantScales, &mut queryQ);
                        searchLayerQuantized(
                            dims,
                            &queryQ,
                            currentEp,
                            ef,
                            &storeGuard,
                            &quantGuard,
                            &mut visited,
                        )
                    });
                    quantResults
                        .into_iter()
                        .map(|(idx, d)| (idx, d as f32))
                        .collect()
                } else {
                    searchLayerReuse(
                        distFn,
                        dims,
                        query,
                        currentEp,
                        ef,
                        0,
                        &storeGuard,
                        &arenaGuard,
                        &mut visited,
                        &mut state,
                    );
                    std::mem::take(&mut state.output)
                }
            })
        });

        // Rerank every candidate with exact f32 distance then trim to k.
        // Quantized ordering is approximate, so a top-k cut at quantized
        // distance would miss true neighbors.
        let mut output: Vec<(VectorId, f32)> = Vec::with_capacity(result.len());
        for (idx, _approxDist) in &result {
            if !storeGuard.deleted[*idx].load(Ordering::Relaxed) {
                let exactDist = distWithFn(distFn, query, arenaSlice(&arenaGuard, dims, *idx));
                output.push((storeGuard.ids[*idx], exactDist));
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
                let storeGuard = self.nodes.read();
                if i < storeGuard.len() {
                    storeGuard.deleted[i].store(true, Ordering::Release);
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

    #[test]
    fn visitedSetBasic() {
        let mut vs = VisitedSet::new();
        vs.prepare(64);
        assert!(vs.insert(1));
        assert!(!vs.insert(1));
        assert!(vs.insert(2));
        // Out-of-range index returns false without touching the buffer.
        assert!(!vs.insert(1 << 20));
        vs.prepare(64);
        // Generation rolled forward, prior marks invalidated.
        assert!(vs.insert(1));
        assert!(vs.insert(2));
    }
}
