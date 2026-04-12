//! Graph algorithms with SIMD-accelerated array operations for PageRank.
//!
//! Provides BFS, shortest path (BFS/Dijkstra), PageRank, connected components,
//! community detection (label propagation), and betweenness centrality (Brandes).

use std::cmp::Reverse;
use std::collections::{BinaryHeap, HashMap, VecDeque};
use std::sync::OnceLock;

use zyron_common::{Result, ZyronError};

use super::schema::NodeId;
use super::storage::CompactGraph;

// ---------------------------------------------------------------------------
// SIMD function pointer types and dispatch
// ---------------------------------------------------------------------------

type FillFn = unsafe fn(data: *mut f64, value: f64, len: usize);
type DiffMaxFn = unsafe fn(a: *const f64, b: *const f64, len: usize) -> f64;

static FILL_FN: OnceLock<FillFn> = OnceLock::new();
static DIFF_MAX_FN: OnceLock<DiffMaxFn> = OnceLock::new();

fn getFillFn() -> FillFn {
    *FILL_FN.get_or_init(selectFillFn)
}

fn getDiffMaxFn() -> DiffMaxFn {
    *DIFF_MAX_FN.get_or_init(selectDiffMaxFn)
}

// ---------------------------------------------------------------------------
// Fill selection
// ---------------------------------------------------------------------------

#[cfg(target_arch = "x86_64")]
fn selectFillFn() -> FillFn {
    if is_x86_feature_detected!("avx512f") {
        fillAvx512
    } else if is_x86_feature_detected!("avx2") {
        fillAvx2
    } else {
        fillFallback
    }
}

#[cfg(target_arch = "aarch64")]
fn selectFillFn() -> FillFn {
    fillNeon
}

#[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
fn selectFillFn() -> FillFn {
    fillFallback
}

// ---------------------------------------------------------------------------
// DiffMax selection
// ---------------------------------------------------------------------------

#[cfg(target_arch = "x86_64")]
fn selectDiffMaxFn() -> DiffMaxFn {
    if is_x86_feature_detected!("avx512f") {
        diffMaxAvx512
    } else if is_x86_feature_detected!("avx2") {
        diffMaxAvx2
    } else {
        diffMaxFallback
    }
}

#[cfg(target_arch = "aarch64")]
fn selectDiffMaxFn() -> DiffMaxFn {
    diffMaxNeon
}

#[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
fn selectDiffMaxFn() -> DiffMaxFn {
    diffMaxFallback
}

// ---------------------------------------------------------------------------
// Fill implementations
// ---------------------------------------------------------------------------

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
#[allow(unsafe_op_in_unsafe_fn)]
unsafe fn fillAvx512(data: *mut f64, value: f64, len: usize) {
    use std::arch::x86_64::*;
    let vec = _mm512_set1_pd(value);
    let chunks = len / 8;
    for c in 0..chunks {
        _mm512_storeu_pd(data.add(c * 8), vec);
    }
    for i in (chunks * 8)..len {
        *data.add(i) = value;
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[allow(unsafe_op_in_unsafe_fn)]
unsafe fn fillAvx2(data: *mut f64, value: f64, len: usize) {
    use std::arch::x86_64::*;
    let vec = _mm256_set1_pd(value);
    let chunks = len / 4;
    for c in 0..chunks {
        _mm256_storeu_pd(data.add(c * 4), vec);
    }
    for i in (chunks * 4)..len {
        *data.add(i) = value;
    }
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
#[allow(unsafe_op_in_unsafe_fn)]
unsafe fn fillNeon(data: *mut f64, value: f64, len: usize) {
    use std::arch::aarch64::*;
    let vec = vdupq_n_f64(value);
    let chunks = len / 2;
    for c in 0..chunks {
        vst1q_f64(data.add(c * 2), vec);
    }
    for i in (chunks * 2)..len {
        *data.add(i) = value;
    }
}

unsafe fn fillFallback(data: *mut f64, value: f64, len: usize) {
    for i in 0..len {
        *data.add(i) = value;
    }
}

// ---------------------------------------------------------------------------
// DiffMax implementations (max absolute difference between two arrays)
// ---------------------------------------------------------------------------

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
#[allow(unsafe_op_in_unsafe_fn)]
unsafe fn diffMaxAvx512(a: *const f64, b: *const f64, len: usize) -> f64 {
    use std::arch::x86_64::*;
    let signMask = _mm512_set1_pd(f64::from_bits(0x7FFF_FFFF_FFFF_FFFF));
    let mut maxVec = _mm512_setzero_pd();
    let chunks = len / 8;
    for c in 0..chunks {
        let off = c * 8;
        let va = _mm512_loadu_pd(a.add(off));
        let vb = _mm512_loadu_pd(b.add(off));
        let diff = _mm512_sub_pd(va, vb);
        let absDiff = _mm512_and_pd(diff, signMask);
        maxVec = _mm512_max_pd(maxVec, absDiff);
    }
    let mut result = _mm512_reduce_max_pd(maxVec);
    for i in (chunks * 8)..len {
        let d = (*a.add(i) - *b.add(i)).abs();
        if d > result {
            result = d;
        }
    }
    result
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[allow(unsafe_op_in_unsafe_fn)]
unsafe fn diffMaxAvx2(a: *const f64, b: *const f64, len: usize) -> f64 {
    use std::arch::x86_64::*;
    let signMask = _mm256_set1_pd(f64::from_bits(0x7FFF_FFFF_FFFF_FFFF));
    let mut maxVec = _mm256_setzero_pd();
    let chunks = len / 4;
    for c in 0..chunks {
        let off = c * 4;
        let va = _mm256_loadu_pd(a.add(off));
        let vb = _mm256_loadu_pd(b.add(off));
        let diff = _mm256_sub_pd(va, vb);
        let absDiff = _mm256_and_pd(diff, signMask);
        maxVec = _mm256_max_pd(maxVec, absDiff);
    }
    // Horizontal max of 4 f64 values
    let hi = _mm256_extractf128_pd(maxVec, 1);
    let lo = _mm256_castpd256_pd128(maxVec);
    let pairMax = _mm_max_pd(lo, hi);
    let swapped = _mm_unpackhi_pd(pairMax, pairMax);
    let scalarMax = _mm_max_pd(pairMax, swapped);
    let mut result = _mm_cvtsd_f64(scalarMax);
    for i in (chunks * 4)..len {
        let d = (*a.add(i) - *b.add(i)).abs();
        if d > result {
            result = d;
        }
    }
    result
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
#[allow(unsafe_op_in_unsafe_fn)]
unsafe fn diffMaxNeon(a: *const f64, b: *const f64, len: usize) -> f64 {
    use std::arch::aarch64::*;
    let mut maxVec = vdupq_n_f64(0.0);
    let chunks = len / 2;
    for c in 0..chunks {
        let off = c * 2;
        let va = vld1q_f64(a.add(off));
        let vb = vld1q_f64(b.add(off));
        let diff = vsubq_f64(va, vb);
        let absDiff = vabsq_f64(diff);
        maxVec = vmaxq_f64(maxVec, absDiff);
    }
    let mut result = vmaxvq_f64(maxVec);
    for i in (chunks * 2)..len {
        let d = (*a.add(i) - *b.add(i)).abs();
        if d > result {
            result = d;
        }
    }
    result
}

unsafe fn diffMaxFallback(a: *const f64, b: *const f64, len: usize) -> f64 {
    let mut result = 0.0f64;
    for i in 0..len {
        let d = (*a.add(i) - *b.add(i)).abs();
        if d > result {
            result = d;
        }
    }
    result
}

// ---------------------------------------------------------------------------
// SIMD helper: fill and diffMax safe wrappers
// ---------------------------------------------------------------------------

/// Fills a mutable f64 slice with a constant value using SIMD when available.
fn simdFill(data: &mut [f64], value: f64) {
    if data.is_empty() {
        return;
    }
    let f = getFillFn();
    unsafe { f(data.as_mut_ptr(), value, data.len()) };
}

/// Returns the maximum absolute difference between two f64 slices.
/// Both slices must have the same length.
fn simdDiffMax(a: &[f64], b: &[f64]) -> f64 {
    debug_assert_eq!(a.len(), b.len());
    let len = a.len().min(b.len());
    if len == 0 {
        return 0.0;
    }
    let f = getDiffMaxFn();
    unsafe { f(a.as_ptr(), b.as_ptr(), len) }
}

// ---------------------------------------------------------------------------
// 1. PageRank
// ---------------------------------------------------------------------------

/// Computes PageRank scores for all nodes in the graph using iterative
/// contribution-push with SIMD-accelerated array fill and convergence check.
///
/// Returns a list of (NodeId, rank) pairs sorted by dense index.
/// The damping factor must be in [0.0, 1.0].
pub fn pagerank(
    graph: &CompactGraph,
    damping: f64,
    iterations: usize,
) -> Result<Vec<(NodeId, f64)>> {
    if damping < 0.0 || damping > 1.0 {
        return Err(ZyronError::GraphAlgorithmError(
            "damping factor must be in [0.0, 1.0]".to_string(),
        ));
    }

    let n = graph.node_count;
    if n == 0 {
        return Ok(Vec::new());
    }

    let nf = n as f64;
    let baseRank = (1.0 - damping) / nf;
    let initRank = 1.0 / nf;

    let mut rank = vec![0.0f64; n];
    simdFill(&mut rank, initRank);
    let mut newRank = vec![0.0f64; n];

    for _ in 0..iterations {
        simdFill(&mut newRank, baseRank);

        // Scatter contributions from each source node to its neighbors
        for src in 0..n {
            let deg = graph.out_degree(src as u32);
            if deg == 0 {
                continue;
            }
            let contribution = damping * rank[src] / deg as f64;
            for &dst in graph.neighbors(src as u32) {
                newRank[dst as usize] += contribution;
            }
        }

        // Check convergence before swapping
        let maxDiff = simdDiffMax(&rank, &newRank);
        std::mem::swap(&mut rank, &mut newRank);
        if maxDiff < 1e-10 {
            break;
        }
    }

    let mut result = Vec::with_capacity(n);
    for i in 0..n {
        result.push((graph.to_node_id(i as u32), rank[i]));
    }
    Ok(result)
}

// ---------------------------------------------------------------------------
// 2. Shortest Path (BFS for unweighted, Dijkstra for weighted)
// ---------------------------------------------------------------------------

/// Wrapper for f64 values in a BinaryHeap that implements Ord via total_cmp.
#[derive(Clone, Copy, PartialEq)]
struct OrdF64(f64);

impl Eq for OrdF64 {}

impl PartialOrd for OrdF64 {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for OrdF64 {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.0.total_cmp(&other.0)
    }
}

/// Finds the shortest path from source to target in the graph.
/// Uses BFS for unweighted graphs, Dijkstra for weighted graphs.
/// Returns None if no path exists.
pub fn shortest_path(
    graph: &CompactGraph,
    source: NodeId,
    target: NodeId,
) -> Result<Option<Vec<NodeId>>> {
    let srcIdx = graph.to_index(source).ok_or_else(|| {
        ZyronError::GraphAlgorithmError(format!("source node not found: {source}"))
    })?;
    let tgtIdx = graph.to_index(target).ok_or_else(|| {
        ZyronError::GraphAlgorithmError(format!("target node not found: {target}"))
    })?;

    if srcIdx == tgtIdx {
        return Ok(Some(vec![source]));
    }

    if graph.weights.is_some() {
        dijkstraPath(graph, srcIdx, tgtIdx)
    } else {
        bfsPath(graph, srcIdx, tgtIdx)
    }
}

/// BFS-based shortest path for unweighted graphs.
fn bfsPath(graph: &CompactGraph, srcIdx: u32, tgtIdx: u32) -> Result<Option<Vec<NodeId>>> {
    let n = graph.node_count;
    let mut parent: Vec<u32> = vec![u32::MAX; n];
    let mut visited = vec![false; n];
    let mut queue = VecDeque::new();

    visited[srcIdx as usize] = true;
    queue.push_back(srcIdx);

    while let Some(current) = queue.pop_front() {
        for &neighbor in graph.neighbors(current) {
            if !visited[neighbor as usize] {
                visited[neighbor as usize] = true;
                parent[neighbor as usize] = current;
                if neighbor == tgtIdx {
                    return Ok(Some(reconstructPath(graph, &parent, srcIdx, tgtIdx)));
                }
                queue.push_back(neighbor);
            }
        }
    }

    Ok(None)
}

/// Dijkstra's algorithm for weighted graphs.
fn dijkstraPath(graph: &CompactGraph, srcIdx: u32, tgtIdx: u32) -> Result<Option<Vec<NodeId>>> {
    let n = graph.node_count;
    let mut dist: Vec<f64> = vec![f64::INFINITY; n];
    let mut parent: Vec<u32> = vec![u32::MAX; n];
    let mut visited = vec![false; n];

    dist[srcIdx as usize] = 0.0;

    // Min-heap: (Reverse(distance), node_index)
    let mut heap: BinaryHeap<(Reverse<OrdF64>, u32)> = BinaryHeap::new();
    heap.push((Reverse(OrdF64(0.0)), srcIdx));

    while let Some((Reverse(OrdF64(d)), u)) = heap.pop() {
        if u == tgtIdx {
            return Ok(Some(reconstructPath(graph, &parent, srcIdx, tgtIdx)));
        }
        if visited[u as usize] {
            continue;
        }
        visited[u as usize] = true;

        if d > dist[u as usize] {
            continue;
        }

        let weights = graph.neighbor_weights(u);
        for (i, &neighbor) in graph.neighbors(u).iter().enumerate() {
            let edgeWeight = match &weights {
                Some(w) => w[i],
                None => 1.0,
            };
            let newDist = d + edgeWeight;
            if newDist < dist[neighbor as usize] {
                dist[neighbor as usize] = newDist;
                parent[neighbor as usize] = u;
                heap.push((Reverse(OrdF64(newDist)), neighbor));
            }
        }
    }

    Ok(None)
}

/// Reconstructs a path from source to target using the parent array.
fn reconstructPath(graph: &CompactGraph, parent: &[u32], srcIdx: u32, tgtIdx: u32) -> Vec<NodeId> {
    let mut path = Vec::new();
    let mut current = tgtIdx;
    while current != srcIdx {
        path.push(graph.to_node_id(current));
        current = parent[current as usize];
    }
    path.push(graph.to_node_id(srcIdx));
    path.reverse();
    path
}

// ---------------------------------------------------------------------------
// 3. BFS (level-limited)
// ---------------------------------------------------------------------------

/// Performs a level-limited BFS from the source node.
/// Returns (node_id, depth) pairs for each reachable node within the given
/// depth (inclusive). The source node has depth 0.
pub fn bfs(graph: &CompactGraph, source: NodeId, maxDepth: u32) -> Result<Vec<(NodeId, u32)>> {
    let srcIdx = graph.to_index(source).ok_or_else(|| {
        ZyronError::GraphAlgorithmError(format!("source node not found: {source}"))
    })?;

    let n = graph.node_count;
    let mut visited = vec![false; n];
    let mut result: Vec<(NodeId, u32)> = Vec::new();
    let mut queue: VecDeque<(u32, u32)> = VecDeque::new(); // (node_idx, depth)

    visited[srcIdx as usize] = true;
    result.push((source, 0));
    queue.push_back((srcIdx, 0));

    while let Some((current, depth)) = queue.pop_front() {
        if depth >= maxDepth {
            continue;
        }
        let nextDepth = depth + 1;
        for &neighbor in graph.neighbors(current) {
            if !visited[neighbor as usize] {
                visited[neighbor as usize] = true;
                result.push((graph.to_node_id(neighbor), nextDepth));
                queue.push_back((neighbor, nextDepth));
            }
        }
    }

    Ok(result)
}

// ---------------------------------------------------------------------------
// 4. Connected Components (union-find)
// ---------------------------------------------------------------------------

/// Finds connected components treating the graph as undirected.
/// Uses union-find with path compression and union by rank.
/// Returns a list of components, where each component is a list of NodeIds.
pub fn connected_components(graph: &CompactGraph) -> Result<Vec<Vec<NodeId>>> {
    let n = graph.node_count;
    if n == 0 {
        return Ok(Vec::new());
    }

    let mut parent: Vec<u32> = (0..n as u32).collect();
    let mut rank: Vec<u8> = vec![0; n];

    // Process all edges as undirected
    for src in 0..n {
        for &dst in graph.neighbors(src as u32) {
            union(&mut parent, &mut rank, src as u32, dst);
        }
    }

    // Group nodes by their root representative
    let mut components: HashMap<u32, Vec<NodeId>> = HashMap::new();
    for i in 0..n {
        let root = find(&mut parent, i as u32);
        components
            .entry(root)
            .or_default()
            .push(graph.to_node_id(i as u32));
    }

    Ok(components.into_values().collect())
}

/// Finds the root of the set containing x, with path compression.
fn find(parent: &mut [u32], x: u32) -> u32 {
    let mut root = x;
    while parent[root as usize] != root {
        root = parent[root as usize];
    }
    // Path compression
    let mut current = x;
    while parent[current as usize] != root {
        let next = parent[current as usize];
        parent[current as usize] = root;
        current = next;
    }
    root
}

/// Unions the sets containing x and y, using union by rank.
fn union(parent: &mut [u32], rank: &mut [u8], x: u32, y: u32) {
    let rootX = find(parent, x);
    let rootY = find(parent, y);
    if rootX == rootY {
        return;
    }
    if rank[rootX as usize] < rank[rootY as usize] {
        parent[rootX as usize] = rootY;
    } else if rank[rootX as usize] > rank[rootY as usize] {
        parent[rootY as usize] = rootX;
    } else {
        parent[rootY as usize] = rootX;
        rank[rootX as usize] += 1;
    }
}

// ---------------------------------------------------------------------------
// 5. Community Detection (label propagation)
// ---------------------------------------------------------------------------

/// Detects communities using label propagation.
/// Each node starts with its own label and iteratively adopts the most
/// frequent label among its neighbors. Runs for at most 20 iterations.
/// The graph is treated as undirected by building a symmetric adjacency view.
pub fn community_detection(graph: &CompactGraph) -> Result<Vec<Vec<NodeId>>> {
    let n = graph.node_count;
    if n == 0 {
        return Ok(Vec::new());
    }

    // Build undirected adjacency lists from the directed CSR
    let mut adjList: Vec<Vec<u32>> = vec![Vec::new(); n];
    for src in 0..n {
        for &dst in graph.neighbors(src as u32) {
            adjList[src].push(dst);
            adjList[dst as usize].push(src as u32);
        }
    }
    // Deduplicate neighbor lists
    for neighbors in &mut adjList {
        neighbors.sort_unstable();
        neighbors.dedup();
    }

    let mut labels: Vec<u32> = (0..n as u32).collect();
    let maxIterations = 20usize;

    // Pre-allocated frequency buffer indexed by label. Avoids per-node
    // HashMap allocation (would be 1M allocations per iteration at scale).
    let mut freq: Vec<u32> = vec![0u32; n];
    let mut touched: Vec<u32> = Vec::with_capacity(64);

    for _ in 0..maxIterations {
        let mut changed = false;

        for node in 0..n {
            let neighbors = &adjList[node];
            if neighbors.is_empty() {
                continue;
            }

            // Count label frequencies among neighbors using the flat buffer
            touched.clear();
            for &nbr in neighbors {
                let lbl = labels[nbr as usize];
                if freq[lbl as usize] == 0 {
                    touched.push(lbl);
                }
                freq[lbl as usize] += 1;
            }

            // Find the most frequent label (tie-break by smallest label)
            let mut bestLabel = labels[node];
            let mut bestCount = 0u32;
            for &lbl in &touched {
                let count = freq[lbl as usize];
                if count > bestCount || (count == bestCount && lbl < bestLabel) {
                    bestLabel = lbl;
                    bestCount = count;
                }
            }

            // Reset only the touched entries (not the whole array)
            for &lbl in &touched {
                freq[lbl as usize] = 0;
            }

            if bestLabel != labels[node] {
                labels[node] = bestLabel;
                changed = true;
            }
        }

        if !changed {
            break;
        }
    }

    // Group nodes by label
    let mut communities: HashMap<u32, Vec<NodeId>> = HashMap::new();
    for i in 0..n {
        communities
            .entry(labels[i])
            .or_default()
            .push(graph.to_node_id(i as u32));
    }

    Ok(communities.into_values().collect())
}

// ---------------------------------------------------------------------------
// 6. Betweenness Centrality (Brandes' algorithm)
// ---------------------------------------------------------------------------

/// Computes betweenness centrality for all nodes using Brandes' algorithm.
/// Runs BFS from each node to compute shortest path counts and accumulate
/// dependency scores. Complexity: O(V * E).
/// Returns (NodeId, centrality) pairs for all nodes.
pub fn betweenness_centrality(graph: &CompactGraph) -> Result<Vec<(NodeId, f64)>> {
    let n = graph.node_count;
    if n == 0 {
        return Ok(Vec::new());
    }

    // Build undirected adjacency for Brandes
    let mut adjList: Vec<Vec<u32>> = vec![Vec::new(); n];
    for src in 0..n {
        for &dst in graph.neighbors(src as u32) {
            adjList[src].push(dst);
            adjList[dst as usize].push(src as u32);
        }
    }
    for neighbors in &mut adjList {
        neighbors.sort_unstable();
        neighbors.dedup();
    }

    let mut centrality = vec![0.0f64; n];

    for s in 0..n {
        let mut stack: Vec<u32> = Vec::new();
        let mut predecessors: Vec<Vec<u32>> = vec![Vec::new(); n];
        let mut sigma = vec![0.0f64; n]; // number of shortest paths
        let mut dist: Vec<i64> = vec![-1; n];
        let mut delta = vec![0.0f64; n];

        sigma[s] = 1.0;
        dist[s] = 0;
        let mut queue = VecDeque::new();
        queue.push_back(s as u32);

        // BFS phase
        while let Some(v) = queue.pop_front() {
            stack.push(v);
            let vDist = dist[v as usize];
            for &w in &adjList[v as usize] {
                // First visit to w
                if dist[w as usize] < 0 {
                    dist[w as usize] = vDist + 1;
                    queue.push_back(w);
                }
                // w is on a shortest path from s through v
                if dist[w as usize] == vDist + 1 {
                    sigma[w as usize] += sigma[v as usize];
                    predecessors[w as usize].push(v);
                }
            }
        }

        // Accumulation phase (back-propagate dependencies)
        while let Some(w) = stack.pop() {
            for &v in &predecessors[w as usize] {
                let fraction = (sigma[v as usize] / sigma[w as usize]) * (1.0 + delta[w as usize]);
                delta[v as usize] += fraction;
            }
            if w as usize != s {
                centrality[w as usize] += delta[w as usize];
            }
        }
    }

    // For undirected graphs, each pair is counted twice
    for val in &mut centrality {
        *val /= 2.0;
    }

    let mut result = Vec::with_capacity(n);
    for i in 0..n {
        result.push((graph.to_node_id(i as u32), centrality[i]));
    }
    Ok(result)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// Helper to build a CompactGraph from a list of unweighted edges.
    fn buildUnweighted(edges: &[(u64, u64)]) -> CompactGraph {
        let edgeData: Vec<(NodeId, NodeId, Option<f64>)> =
            edges.iter().map(|&(a, b)| (a, b, None)).collect();
        CompactGraph::build(&edgeData)
    }

    /// Helper to build a CompactGraph from weighted edges.
    fn buildWeighted(edges: &[(u64, u64, f64)]) -> CompactGraph {
        let edgeData: Vec<(NodeId, NodeId, Option<f64>)> =
            edges.iter().map(|&(a, b, w)| (a, b, Some(w))).collect();
        CompactGraph::build(&edgeData)
    }

    // -----------------------------------------------------------------------
    // PageRank tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_pagerank_cycle_graph() {
        // 4-node directed cycle: A->B->C->D->A
        // All nodes should have equal rank (1/4 each).
        let graph = buildUnweighted(&[(1, 2), (2, 3), (3, 4), (4, 1)]);
        let result = pagerank(&graph, 0.85, 100).expect("pagerank should succeed");
        assert_eq!(result.len(), 4);

        let expectedRank = 0.25;
        for (_, rank) in &result {
            assert!(
                (rank - expectedRank).abs() < 1e-6,
                "expected rank ~{expectedRank}, got {rank}"
            );
        }
    }

    #[test]
    fn test_pagerank_empty_graph() {
        let graph = CompactGraph::build(&[]);
        let result = pagerank(&graph, 0.85, 10).expect("pagerank should succeed");
        assert!(result.is_empty());
    }

    #[test]
    fn test_pagerank_invalid_damping() {
        let graph = buildUnweighted(&[(1, 2)]);
        assert!(pagerank(&graph, -0.1, 10).is_err());
        assert!(pagerank(&graph, 1.1, 10).is_err());
    }

    // -----------------------------------------------------------------------
    // Shortest path tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_shortest_path_chain() {
        // Chain: A->B->C->D
        let graph = buildUnweighted(&[(1, 2), (2, 3), (3, 4)]);
        let path = shortest_path(&graph, 1, 4)
            .expect("should succeed")
            .expect("path should exist");
        assert_eq!(path, vec![1, 2, 3, 4]);
    }

    #[test]
    fn test_shortest_path_disconnected() {
        // Two separate edges with no connection between groups
        let graph = buildUnweighted(&[(1, 2), (3, 4)]);
        let result = shortest_path(&graph, 1, 4).expect("should succeed");
        assert!(result.is_none());
    }

    #[test]
    fn test_shortest_path_same_node() {
        let graph = buildUnweighted(&[(1, 2)]);
        let path = shortest_path(&graph, 1, 1)
            .expect("should succeed")
            .expect("path should exist");
        assert_eq!(path, vec![1]);
    }

    #[test]
    fn test_shortest_path_weighted() {
        // Weighted graph: direct path 1->3 costs 10, path 1->2->3 costs 3
        let graph = buildWeighted(&[(1, 2, 1.0), (2, 3, 2.0), (1, 3, 10.0)]);
        let path = shortest_path(&graph, 1, 3)
            .expect("should succeed")
            .expect("path should exist");
        assert_eq!(path, vec![1, 2, 3]);
    }

    #[test]
    fn test_shortest_path_node_not_found() {
        let graph = buildUnweighted(&[(1, 2)]);
        assert!(shortest_path(&graph, 1, 999).is_err());
        assert!(shortest_path(&graph, 999, 1).is_err());
    }

    // -----------------------------------------------------------------------
    // BFS tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_bfs_star_depth_1() {
        // Star graph: hub 0 -> 1, 2, 3, 4, 5
        let graph = buildUnweighted(&[(0, 1), (0, 2), (0, 3), (0, 4), (0, 5)]);
        let result = bfs(&graph, 0, 1).expect("bfs should succeed");

        // Should contain hub (0) at depth 0 and all direct neighbors at depth 1
        let depthOf = |id: u64| result.iter().find(|(n, _)| *n == id).map(|(_, d)| *d);
        assert_eq!(depthOf(0), Some(0));
        for nodeId in 1..=5u64 {
            assert_eq!(
                depthOf(nodeId),
                Some(1),
                "node {nodeId} should be at depth 1"
            );
        }
        assert_eq!(result.len(), 6);
    }

    #[test]
    fn test_bfs_depth_limited() {
        // Chain: 1->2->3->4->5
        let graph = buildUnweighted(&[(1, 2), (2, 3), (3, 4), (4, 5)]);
        let result = bfs(&graph, 1, 2).expect("bfs should succeed");

        // Depth 0: node 1, depth 1: node 2, depth 2: node 3
        let depthOf = |id: u64| result.iter().find(|(n, _)| *n == id).map(|(_, d)| *d);
        assert_eq!(depthOf(1), Some(0));
        assert_eq!(depthOf(2), Some(1));
        assert_eq!(depthOf(3), Some(2));
        assert_eq!(depthOf(4), None);
        assert_eq!(depthOf(5), None);
    }

    #[test]
    fn test_bfs_node_not_found() {
        let graph = buildUnweighted(&[(1, 2)]);
        assert!(bfs(&graph, 999, 5).is_err());
    }

    // -----------------------------------------------------------------------
    // Connected components tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_connected_components_two_triangles() {
        // Triangle 1: 1-2-3, Triangle 2: 4-5-6
        let graph = buildUnweighted(&[(1, 2), (2, 3), (3, 1), (4, 5), (5, 6), (6, 4)]);
        let mut components = connected_components(&graph).expect("should succeed");
        assert_eq!(components.len(), 2);

        // Sort components by size then by first element for deterministic comparison
        for c in &mut components {
            c.sort_unstable();
        }
        components.sort_unstable_by_key(|c| c[0]);

        assert_eq!(components[0], vec![1, 2, 3]);
        assert_eq!(components[1], vec![4, 5, 6]);
    }

    #[test]
    fn test_connected_components_single() {
        // All nodes connected through one chain
        let graph = buildUnweighted(&[(1, 2), (2, 3), (3, 4)]);
        let components = connected_components(&graph).expect("should succeed");
        assert_eq!(components.len(), 1);
        assert_eq!(components[0].len(), 4);
    }

    #[test]
    fn test_connected_components_empty() {
        let graph = CompactGraph::build(&[]);
        let components = connected_components(&graph).expect("should succeed");
        assert!(components.is_empty());
    }

    // -----------------------------------------------------------------------
    // Community detection tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_community_detection_two_cliques() {
        // Two cliques (1,2,3) and (4,5,6) connected by a single edge 3->4
        let graph = buildUnweighted(&[
            // Clique 1
            (1, 2),
            (2, 1),
            (1, 3),
            (3, 1),
            (2, 3),
            (3, 2),
            // Bridge
            (3, 4),
            // Clique 2
            (4, 5),
            (5, 4),
            (4, 6),
            (6, 4),
            (5, 6),
            (6, 5),
        ]);
        let communities = community_detection(&graph).expect("should succeed");

        // With strong internal connections, label propagation should find
        // at most 2 communities. The bridge node may join either side.
        assert!(
            communities.len() <= 3,
            "expected at most 3 communities, got {}",
            communities.len()
        );
        // All 6 nodes should be present across all communities
        let totalNodes: usize = communities.iter().map(|c| c.len()).sum();
        assert_eq!(totalNodes, 6);
    }

    #[test]
    fn test_community_detection_empty() {
        let graph = CompactGraph::build(&[]);
        let communities = community_detection(&graph).expect("should succeed");
        assert!(communities.is_empty());
    }

    // -----------------------------------------------------------------------
    // Betweenness centrality tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_betweenness_centrality_bridge() {
        // Graph: 1-2-3, node 2 is the bridge
        // 1->2, 2->3 (treated as undirected)
        let graph = buildUnweighted(&[(1, 2), (2, 3)]);
        let result = betweenness_centrality(&graph).expect("should succeed");

        // Node 2 should have the highest centrality (it's on the only path 1<->3)
        let centralityMap: HashMap<NodeId, f64> = result.into_iter().collect();

        let c1 = centralityMap[&1];
        let c2 = centralityMap[&2];
        let c3 = centralityMap[&3];

        assert!(
            c2 > c1 && c2 > c3,
            "bridge node 2 should have highest centrality: c1={c1}, c2={c2}, c3={c3}"
        );
    }

    #[test]
    fn test_betweenness_centrality_complete_graph() {
        // Complete graph on 4 nodes: all centralities should be equal
        let graph = buildUnweighted(&[(1, 2), (1, 3), (1, 4), (2, 3), (2, 4), (3, 4)]);
        let result = betweenness_centrality(&graph).expect("should succeed");
        assert_eq!(result.len(), 4);

        let firstCentrality = result[0].1;
        for (_, c) in &result {
            assert!(
                (c - firstCentrality).abs() < 1e-9,
                "all nodes in a complete graph should have equal centrality"
            );
        }
    }

    #[test]
    fn test_betweenness_centrality_empty() {
        let graph = CompactGraph::build(&[]);
        let result = betweenness_centrality(&graph).expect("should succeed");
        assert!(result.is_empty());
    }

    // -----------------------------------------------------------------------
    // SIMD helper tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_simd_fill() {
        let mut data = vec![0.0f64; 100];
        simdFill(&mut data, 3.14);
        for val in &data {
            assert!((val - 3.14).abs() < 1e-15);
        }
    }

    #[test]
    fn test_simd_fill_empty() {
        let mut data: Vec<f64> = Vec::new();
        simdFill(&mut data, 1.0); // should not panic
    }

    #[test]
    fn test_simd_diff_max() {
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        let b = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 7.0];
        let result = simdDiffMax(&a, &b);
        assert!(
            (result - 3.0).abs() < 1e-15,
            "max diff should be 3.0, got {result}"
        );
    }

    #[test]
    fn test_simd_diff_max_identical() {
        let a = vec![1.0, 2.0, 3.0];
        let result = simdDiffMax(&a, &a);
        assert!(result.abs() < 1e-15);
    }
}
