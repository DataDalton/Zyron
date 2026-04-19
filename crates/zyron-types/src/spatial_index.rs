//! In-memory R-tree spatial index.
//!
//! Supports up to 4 dimensions (2D/3D/XYZM). Readers are wait-free: every
//! node is immutable after construction and wrapped in Arc. Reads clone the
//! root Arc once, then traverse without any locks.
//!
//! Writers serialize via an internal parking_lot::RwLock and use copy-on-write:
//! an insert or delete clones only the nodes on the path from root to the
//! affected leaf, leaving the rest of the tree shared. The outer lock is held
//! for a few microseconds per write just to atomically swap the new root in.
//!
//! Features:
//! - STR (Sort-Tile-Recursive) bulk load for near-optimal initial builds
//! - R*-tree insertion with forced-reinsert on first overflow per level
//! - Choose-subtree heuristic: minimize overlap enlargement for leaves,
//!   minimize area enlargement for internal nodes
//! - Axis-aligned split with margin/overlap/area evaluation
//! - Tombstone-based deletes with background compaction trigger
//! - KNN via branch-and-bound over mindist priority queue
//! - Range queries (MBR intersects)
//! - ST_DWithin via point expansion into bounding box
//! - ST_Intersects via MBR candidate filter
//! - Per-index metrics (query count, insert count, split count, compaction count,
//!   tombstone ratio, node count, height)

use parking_lot::{Mutex, RwLock};
use std::cmp::Ordering as CmpOrdering;
use std::collections::BinaryHeap;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

// ---------------------------------------------------------------------------
// Tuning constants
// ---------------------------------------------------------------------------

pub const MAX_DIMS: usize = 4;
pub const DEFAULT_MIN_FILL: usize = 16;
pub const DEFAULT_MAX_FILL: usize = 32;
const COMPACTION_TOMBSTONE_RATIO: f64 = 0.25;
const COMPACTION_MIN_ENTRIES: u64 = 1024;
/// Minimum slice size before STR bulk load parallelizes slice processing
/// across threads. Below this threshold the per-thread overhead exceeds
/// the parallelism win for sort + node allocation.
const PARALLEL_BULK_LOAD_THRESHOLD: usize = 10_000;

// ---------------------------------------------------------------------------
// Minimum Bounding Rectangle (N-dim)
// ---------------------------------------------------------------------------

/// Axis-aligned bounding box in up to 4 dimensions.
/// Layout: mins[0..dims] and maxs[0..dims] are valid; remaining slots unused.
#[derive(Debug, Clone, Copy)]
pub struct Mbr {
    pub dims: u8,
    pub mins: [f64; MAX_DIMS],
    pub maxs: [f64; MAX_DIMS],
}

impl PartialEq for Mbr {
    fn eq(&self, other: &Self) -> bool {
        if self.dims != other.dims {
            return false;
        }
        let d = self.dims as usize;
        self.mins[..d] == other.mins[..d] && self.maxs[..d] == other.maxs[..d]
    }
}

impl Mbr {
    /// Creates an MBR covering a single point.
    pub fn point(coords: &[f64]) -> Self {
        assert!(
            coords.len() >= 1 && coords.len() <= MAX_DIMS,
            "point dimensions out of range: {}",
            coords.len()
        );
        let mut mins = [0.0f64; MAX_DIMS];
        let mut maxs = [0.0f64; MAX_DIMS];
        for (i, c) in coords.iter().enumerate() {
            mins[i] = *c;
            maxs[i] = *c;
        }
        Self {
            dims: coords.len() as u8,
            mins,
            maxs,
        }
    }

    /// Creates an MBR from explicit extents.
    pub fn from_extents(mins_in: &[f64], maxs_in: &[f64]) -> Self {
        assert_eq!(mins_in.len(), maxs_in.len(), "dim mismatch");
        assert!(mins_in.len() >= 1 && mins_in.len() <= MAX_DIMS);
        let mut mins = [0.0f64; MAX_DIMS];
        let mut maxs = [0.0f64; MAX_DIMS];
        for i in 0..mins_in.len() {
            assert!(mins_in[i] <= maxs_in[i], "min > max on dim {}", i);
            mins[i] = mins_in[i];
            maxs[i] = maxs_in[i];
        }
        Self {
            dims: mins_in.len() as u8,
            mins,
            maxs,
        }
    }

    /// Creates an empty MBR for the given dimensionality. Any union with a
    /// real MBR produces that real MBR.
    pub fn empty(dims: u8) -> Self {
        assert!(dims >= 1 && (dims as usize) <= MAX_DIMS);
        let mut mins = [0.0f64; MAX_DIMS];
        let mut maxs = [0.0f64; MAX_DIMS];
        for i in 0..(dims as usize) {
            mins[i] = f64::INFINITY;
            maxs[i] = f64::NEG_INFINITY;
        }
        Self { dims, mins, maxs }
    }

    /// Returns true if this MBR is empty (i.e. any dim has min > max).
    pub fn is_empty(&self) -> bool {
        for i in 0..(self.dims as usize) {
            if self.mins[i] > self.maxs[i] {
                return true;
            }
        }
        false
    }

    /// Returns true if self fully contains other.
    pub fn contains(&self, other: &Mbr) -> bool {
        if self.dims != other.dims {
            return false;
        }
        for i in 0..(self.dims as usize) {
            if other.mins[i] < self.mins[i] || other.maxs[i] > self.maxs[i] {
                return false;
            }
        }
        true
    }

    /// Returns true if self and other share any volume.
    pub fn intersects(&self, other: &Mbr) -> bool {
        if self.dims != other.dims {
            return false;
        }
        for i in 0..(self.dims as usize) {
            if self.maxs[i] < other.mins[i] || self.mins[i] > other.maxs[i] {
                return false;
            }
        }
        true
    }

    /// Returns true if this MBR contains the given point.
    pub fn contains_point(&self, coords: &[f64]) -> bool {
        if coords.len() != self.dims as usize {
            return false;
        }
        for i in 0..(self.dims as usize) {
            if coords[i] < self.mins[i] || coords[i] > self.maxs[i] {
                return false;
            }
        }
        true
    }

    /// Volume (area in 2D, volume in 3D, hypervolume in 4D).
    pub fn area(&self) -> f64 {
        let mut a = 1.0;
        for i in 0..(self.dims as usize) {
            let side = self.maxs[i] - self.mins[i];
            if side < 0.0 {
                return 0.0;
            }
            a *= side;
        }
        a
    }

    /// Sum of side lengths (used as R*-tree margin).
    pub fn margin(&self) -> f64 {
        let mut m = 0.0;
        for i in 0..(self.dims as usize) {
            m += (self.maxs[i] - self.mins[i]).max(0.0);
        }
        m
    }

    /// Area enlargement required to include other.
    pub fn enlargement(&self, other: &Mbr) -> f64 {
        let unioned = self.union(other);
        unioned.area() - self.area()
    }

    /// Overlap area/volume with another MBR. Zero if disjoint.
    pub fn overlap(&self, other: &Mbr) -> f64 {
        if self.dims != other.dims {
            return 0.0;
        }
        let mut v = 1.0;
        for i in 0..(self.dims as usize) {
            let lo = self.mins[i].max(other.mins[i]);
            let hi = self.maxs[i].min(other.maxs[i]);
            if hi < lo {
                return 0.0;
            }
            v *= hi - lo;
        }
        v
    }

    /// Minimum-bounding union of two MBRs.
    pub fn union(&self, other: &Mbr) -> Mbr {
        assert_eq!(self.dims, other.dims);
        let mut mins = [0.0f64; MAX_DIMS];
        let mut maxs = [0.0f64; MAX_DIMS];
        for i in 0..(self.dims as usize) {
            mins[i] = self.mins[i].min(other.mins[i]);
            maxs[i] = self.maxs[i].max(other.maxs[i]);
        }
        Mbr {
            dims: self.dims,
            mins,
            maxs,
        }
    }

    /// In-place union.
    pub fn union_inplace(&mut self, other: &Mbr) {
        assert_eq!(self.dims, other.dims);
        for i in 0..(self.dims as usize) {
            if other.mins[i] < self.mins[i] {
                self.mins[i] = other.mins[i];
            }
            if other.maxs[i] > self.maxs[i] {
                self.maxs[i] = other.maxs[i];
            }
        }
    }

    /// Squared Euclidean distance from a point to the nearest edge of this
    /// MBR. Zero if the point is inside. Used as a KNN lower bound.
    pub fn mindist_squared(&self, point: &[f64]) -> f64 {
        debug_assert_eq!(point.len(), self.dims as usize);
        let mut sum = 0.0;
        for i in 0..(self.dims as usize) {
            let p = point[i];
            let d = if p < self.mins[i] {
                self.mins[i] - p
            } else if p > self.maxs[i] {
                p - self.maxs[i]
            } else {
                0.0
            };
            sum += d * d;
        }
        sum
    }

    /// Center point.
    pub fn center(&self) -> [f64; MAX_DIMS] {
        let mut c = [0.0f64; MAX_DIMS];
        for i in 0..(self.dims as usize) {
            c[i] = 0.5 * (self.mins[i] + self.maxs[i]);
        }
        c
    }
}

// ---------------------------------------------------------------------------
// Leaf entry and Node
// ---------------------------------------------------------------------------

/// A stored geometry keyed by caller-supplied data (typically a rowid).
#[derive(Debug, Clone)]
pub struct LeafEntry<T: Clone> {
    pub mbr: Mbr,
    pub data: T,
    /// True if this entry has been deleted. Retained in place to keep node
    /// fanout stable until a compaction rebuilds the subtree.
    pub deleted: bool,
}

#[derive(Debug, Clone)]
pub enum NodeKind<T: Clone> {
    Internal(Vec<Arc<Node<T>>>),
    Leaf(Vec<LeafEntry<T>>),
}

/// Immutable R-tree node. All mutations produce new Node values via
/// copy-on-write; existing Arc<Node> references stay valid.
#[derive(Debug, Clone)]
pub struct Node<T: Clone> {
    pub mbr: Mbr,
    pub kind: NodeKind<T>,
}

impl<T: Clone> Node<T> {
    pub fn is_leaf(&self) -> bool {
        matches!(self.kind, NodeKind::Leaf(_))
    }

    pub fn child_count(&self) -> usize {
        match &self.kind {
            NodeKind::Internal(c) => c.len(),
            NodeKind::Leaf(e) => e.len(),
        }
    }

    /// Walks the tree from this node down, collecting (height, node_count,
    /// leaf_count, entry_count, tombstone_count).
    pub fn stats(&self) -> NodeStats {
        let mut s = NodeStats::default();
        s.node_count = 1;
        match &self.kind {
            NodeKind::Internal(children) => {
                let mut max_child_height = 0;
                for c in children {
                    let cs = c.stats();
                    s.node_count += cs.node_count;
                    s.leaf_count += cs.leaf_count;
                    s.entry_count += cs.entry_count;
                    s.tombstone_count += cs.tombstone_count;
                    if cs.height > max_child_height {
                        max_child_height = cs.height;
                    }
                }
                s.height = max_child_height + 1;
            }
            NodeKind::Leaf(entries) => {
                s.leaf_count = 1;
                s.entry_count = entries.len() as u64;
                s.tombstone_count = entries.iter().filter(|e| e.deleted).count() as u64;
                s.height = 1;
            }
        }
        s
    }
}

#[derive(Debug, Default, Clone, Copy)]
pub struct NodeStats {
    pub height: u32,
    pub node_count: u64,
    pub leaf_count: u64,
    pub entry_count: u64,
    pub tombstone_count: u64,
}

// ---------------------------------------------------------------------------
// Metrics
// ---------------------------------------------------------------------------

/// Per-index runtime metrics. All counters are atomic and lock-free.
#[derive(Debug, Default)]
pub struct Metrics {
    pub query_knn: AtomicU64,
    pub query_range: AtomicU64,
    pub query_dwithin: AtomicU64,
    pub query_intersects: AtomicU64,
    pub inserts: AtomicU64,
    pub deletes: AtomicU64,
    pub splits: AtomicU64,
    pub reinserts: AtomicU64,
    pub compactions: AtomicU64,
    pub build_count: AtomicU64,
}

impl Metrics {
    pub fn snapshot(&self) -> MetricsSnapshot {
        MetricsSnapshot {
            query_knn: self.query_knn.load(Ordering::Relaxed),
            query_range: self.query_range.load(Ordering::Relaxed),
            query_dwithin: self.query_dwithin.load(Ordering::Relaxed),
            query_intersects: self.query_intersects.load(Ordering::Relaxed),
            inserts: self.inserts.load(Ordering::Relaxed),
            deletes: self.deletes.load(Ordering::Relaxed),
            splits: self.splits.load(Ordering::Relaxed),
            reinserts: self.reinserts.load(Ordering::Relaxed),
            compactions: self.compactions.load(Ordering::Relaxed),
            build_count: self.build_count.load(Ordering::Relaxed),
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct MetricsSnapshot {
    pub query_knn: u64,
    pub query_range: u64,
    pub query_dwithin: u64,
    pub query_intersects: u64,
    pub inserts: u64,
    pub deletes: u64,
    pub splits: u64,
    pub reinserts: u64,
    pub compactions: u64,
    pub build_count: u64,
}

// ---------------------------------------------------------------------------
// RTree
// ---------------------------------------------------------------------------

/// Tree configuration (node fanout). Leaves and internals share the same
/// fill bounds for simplicity.
#[derive(Debug, Clone, Copy)]
pub struct RTreeConfig {
    pub min_fill: usize,
    pub max_fill: usize,
    pub srid: u32,
}

impl Default for RTreeConfig {
    fn default() -> Self {
        Self {
            min_fill: DEFAULT_MIN_FILL,
            max_fill: DEFAULT_MAX_FILL,
            srid: 4326,
        }
    }
}

/// Internal tree state behind the writer RwLock.
struct Inner<T: Clone> {
    root: Arc<Node<T>>,
    dims: u8,
    config: RTreeConfig,
    /// Number of live (non-tombstoned) entries.
    live_entries: u64,
    /// Number of tombstoned entries.
    tombstones: u64,
}

/// Thread-safe R-tree.
///
/// Holds an auxiliary `data -> mbr` map so `delete_by_data` is O(1) lookup
/// + O(log N) tree delete instead of an O(N) tree walk. The map is updated
/// under the write_gate so it stays consistent with the tree even under
/// concurrent inserts and deletes.
pub struct RTree<T: Clone + Send + Sync + std::hash::Hash + Eq + 'static> {
    inner: RwLock<Inner<T>>,
    metrics: Metrics,
    /// Serializes concurrent writers; reads never take this. Using a Mutex
    /// here rather than relying on the RwLock write lock lets the RwLock be
    /// held only for the final atomic root swap.
    write_gate: Mutex<()>,
    /// Inverse data -> MBR map for O(1) delete-by-data lookups.
    inverse: Mutex<std::collections::HashMap<T, Mbr>>,
}

impl<T: Clone + Send + Sync + std::hash::Hash + Eq + 'static> RTree<T> {
    /// Creates an empty tree with the given dimensionality and default config.
    pub fn new(dims: u8) -> Self {
        Self::with_config(dims, RTreeConfig::default())
    }

    /// Creates an empty tree with an explicit configuration.
    pub fn with_config(dims: u8, config: RTreeConfig) -> Self {
        assert!(dims >= 1 && (dims as usize) <= MAX_DIMS);
        assert!(config.min_fill >= 2 && config.min_fill <= config.max_fill / 2);
        let empty_leaf = Node {
            mbr: Mbr::empty(dims),
            kind: NodeKind::Leaf(Vec::new()),
        };
        Self {
            inner: RwLock::new(Inner {
                root: Arc::new(empty_leaf),
                dims,
                config,
                live_entries: 0,
                tombstones: 0,
            }),
            metrics: Metrics::default(),
            write_gate: Mutex::new(()),
            inverse: Mutex::new(std::collections::HashMap::new()),
        }
    }

    pub fn dims(&self) -> u8 {
        self.inner.read().dims
    }

    pub fn config(&self) -> RTreeConfig {
        self.inner.read().config
    }

    pub fn len(&self) -> u64 {
        self.inner.read().live_entries
    }

    pub fn tombstones(&self) -> u64 {
        self.inner.read().tombstones
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn metrics(&self) -> &Metrics {
        &self.metrics
    }

    pub fn stats(&self) -> NodeStats {
        let guard = self.inner.read();
        guard.root.stats()
    }

    /// Takes a wait-free snapshot of the current root. Subsequent reads can
    /// traverse this snapshot without any locking; concurrent writers will
    /// create new nodes rather than mutate the snapshot.
    pub fn snapshot(&self) -> Arc<Node<T>> {
        self.inner.read().root.clone()
    }
}

// ---------------------------------------------------------------------------
// STR bulk load
// ---------------------------------------------------------------------------

impl<T: Clone + Send + Sync + std::hash::Hash + Eq + 'static> RTree<T> {
    /// Bulk-loads the tree from an existing set of entries using the
    /// Sort-Tile-Recursive algorithm. The result is a balanced tree with
    /// near-optimal node fill. Replaces any existing contents.
    pub fn bulk_load(&self, entries: Vec<LeafEntry<T>>) {
        let _write = self.write_gate.lock();

        let (dims, config) = {
            let g = self.inner.read();
            (g.dims, g.config)
        };

        let new_root = bulk_build(entries.clone(), dims, &config);
        let stats = new_root.stats();

        // Rebuild the inverse map from the bulk-loaded entries.
        let mut inv = self.inverse.lock();
        inv.clear();
        for e in &entries {
            if !e.deleted {
                inv.insert(e.data.clone(), e.mbr);
            }
        }
        drop(inv);

        let mut g = self.inner.write();
        g.root = new_root;
        g.live_entries = stats.entry_count - stats.tombstone_count;
        g.tombstones = stats.tombstone_count;
        self.metrics.build_count.fetch_add(1, Ordering::Relaxed);
    }
}

/// Tile count per dimension so that leaves are ~max_fill full.
fn str_slice_count(n: usize, max_fill: usize, dims: usize) -> Vec<usize> {
    // Number of leaves needed
    let leaves = (n as f64 / max_fill as f64).ceil() as usize;
    let leaves = leaves.max(1);
    // Slice count per dim = leaves^(1/dims)
    let per_dim = (leaves as f64).powf(1.0 / dims as f64).ceil() as usize;
    vec![per_dim.max(1); dims]
}

fn bulk_build<T: Clone + Send + Sync + 'static>(
    mut entries: Vec<LeafEntry<T>>,
    dims: u8,
    config: &RTreeConfig,
) -> Arc<Node<T>> {
    if entries.is_empty() {
        return Arc::new(Node {
            mbr: Mbr::empty(dims),
            kind: NodeKind::Leaf(Vec::new()),
        });
    }

    // Build leaf level via STR: slice by dim 0, then within each slice by dim 1, etc.
    let leaves = str_pack_leaves(&mut entries, dims, config);

    // Recursively pack internal levels the same way
    let mut level: Vec<Arc<Node<T>>> = leaves;
    while level.len() > 1 {
        level = str_pack_internals(level, dims, config);
    }
    level.into_iter().next().unwrap()
}

fn str_pack_leaves<T: Clone + Send + Sync + 'static>(
    entries: &mut Vec<LeafEntry<T>>,
    dims: u8,
    config: &RTreeConfig,
) -> Vec<Arc<Node<T>>> {
    let d = dims as usize;
    let slices = str_slice_count(entries.len(), config.max_fill, d);
    str_tile_leaves(entries, &slices, 0, config)
}

/// Recursive STR tiling for leaf entries: sort by dim `axis` center, chunk
/// by slice count, recurse into each chunk on the next axis.
///
/// Slices large enough (> PARALLEL_BULK_LOAD_THRESHOLD entries) recurse in
/// parallel via std::thread::scope. Small slices recurse sequentially since
/// thread overhead dominates the work.
fn str_tile_leaves<T: Clone + Send + Sync + 'static>(
    entries: &mut [LeafEntry<T>],
    slices: &[usize],
    axis: usize,
    config: &RTreeConfig,
) -> Vec<Arc<Node<T>>> {
    if entries.is_empty() {
        return Vec::new();
    }
    if axis >= slices.len() {
        return chunk_entries_into_leaves(entries, config);
    }

    entries.sort_by(|a, b| {
        let ac = (a.mbr.mins[axis] + a.mbr.maxs[axis]) * 0.5;
        let bc = (b.mbr.mins[axis] + b.mbr.maxs[axis]) * 0.5;
        ac.partial_cmp(&bc).unwrap_or(CmpOrdering::Equal)
    });

    let num_slices = slices[axis];
    let slice_size = (entries.len() + num_slices - 1) / num_slices;

    if entries.len() >= PARALLEL_BULK_LOAD_THRESHOLD {
        std::thread::scope(|s| {
            let chunks: Vec<&mut [LeafEntry<T>]> = entries.chunks_mut(slice_size).collect();
            let handles: Vec<_> = chunks
                .into_iter()
                .map(|chunk| s.spawn(move || str_tile_leaves(chunk, slices, axis + 1, config)))
                .collect();
            handles
                .into_iter()
                .flat_map(|h| h.join().expect("STR tile thread panicked"))
                .collect()
        })
    } else {
        let mut out = Vec::new();
        let mut i = 0;
        while i < entries.len() {
            let end = (i + slice_size).min(entries.len());
            let part = &mut entries[i..end];
            out.extend(str_tile_leaves(part, slices, axis + 1, config));
            i = end;
        }
        out
    }
}

fn chunk_entries_into_leaves<T: Clone + Send + Sync + 'static>(
    entries: &[LeafEntry<T>],
    config: &RTreeConfig,
) -> Vec<Arc<Node<T>>> {
    let mut out = Vec::new();
    let mut i = 0;
    while i < entries.len() {
        let end = (i + config.max_fill).min(entries.len());
        let chunk: Vec<LeafEntry<T>> = entries[i..end].to_vec();
        let mbr = compute_mbr_leaf(&chunk);
        out.push(Arc::new(Node {
            mbr,
            kind: NodeKind::Leaf(chunk),
        }));
        i = end;
    }
    out
}

fn str_pack_internals<T: Clone + Send + Sync + 'static>(
    nodes: Vec<Arc<Node<T>>>,
    dims: u8,
    config: &RTreeConfig,
) -> Vec<Arc<Node<T>>> {
    let d = dims as usize;
    let slices = str_slice_count(nodes.len(), config.max_fill, d);
    let mut nodes_mut = nodes;
    str_tile_internals(&mut nodes_mut, &slices, 0, config)
}

fn str_tile_internals<T: Clone + Send + Sync + 'static>(
    nodes: &mut [Arc<Node<T>>],
    slices: &[usize],
    axis: usize,
    config: &RTreeConfig,
) -> Vec<Arc<Node<T>>> {
    if nodes.is_empty() {
        return Vec::new();
    }
    if axis >= slices.len() {
        return chunk_nodes_into_internals(nodes, config);
    }

    nodes.sort_by(|a, b| {
        let ac = (a.mbr.mins[axis] + a.mbr.maxs[axis]) * 0.5;
        let bc = (b.mbr.mins[axis] + b.mbr.maxs[axis]) * 0.5;
        ac.partial_cmp(&bc).unwrap_or(CmpOrdering::Equal)
    });

    let num_slices = slices[axis];
    let slice_size = (nodes.len() + num_slices - 1) / num_slices;

    if nodes.len() >= PARALLEL_BULK_LOAD_THRESHOLD {
        std::thread::scope(|s| {
            let chunks: Vec<&mut [Arc<Node<T>>]> = nodes.chunks_mut(slice_size).collect();
            let handles: Vec<_> = chunks
                .into_iter()
                .map(|chunk| s.spawn(move || str_tile_internals(chunk, slices, axis + 1, config)))
                .collect();
            handles
                .into_iter()
                .flat_map(|h| h.join().expect("STR internal tile thread panicked"))
                .collect()
        })
    } else {
        let mut out = Vec::new();
        let mut i = 0;
        while i < nodes.len() {
            let end = (i + slice_size).min(nodes.len());
            out.extend(str_tile_internals(
                &mut nodes[i..end],
                slices,
                axis + 1,
                config,
            ));
            i = end;
        }
        out
    }
}

fn chunk_nodes_into_internals<T: Clone + Send + Sync + 'static>(
    nodes: &[Arc<Node<T>>],
    config: &RTreeConfig,
) -> Vec<Arc<Node<T>>> {
    let mut out = Vec::new();
    let mut i = 0;
    while i < nodes.len() {
        let end = (i + config.max_fill).min(nodes.len());
        let chunk: Vec<Arc<Node<T>>> = nodes[i..end].to_vec();
        let mbr = compute_mbr_internal(&chunk);
        out.push(Arc::new(Node {
            mbr,
            kind: NodeKind::Internal(chunk),
        }));
        i = end;
    }
    out
}

fn compute_mbr_leaf<T: Clone>(entries: &[LeafEntry<T>]) -> Mbr {
    let mut it = entries.iter();
    let first = it.next().expect("compute_mbr_leaf on empty").mbr;
    let mut u = first;
    for e in it {
        u.union_inplace(&e.mbr);
    }
    u
}

fn compute_mbr_internal<T: Clone>(nodes: &[Arc<Node<T>>]) -> Mbr {
    let mut it = nodes.iter();
    let first = it.next().expect("compute_mbr_internal on empty").mbr;
    let mut u = first;
    for n in it {
        u.union_inplace(&n.mbr);
    }
    u
}

// ---------------------------------------------------------------------------
// Insert
// ---------------------------------------------------------------------------

/// Result of an insert descent: either no structural change, or the affected
/// node split into two siblings that must be attached by the parent.
enum InsertResult<T: Clone> {
    Ok(Arc<Node<T>>),
    Split(Arc<Node<T>>, Arc<Node<T>>),
}

impl<T: Clone + Send + Sync + std::hash::Hash + Eq + 'static> RTree<T> {
    /// Inserts one entry into the tree.
    pub fn insert(&self, entry: LeafEntry<T>) {
        let _write = self.write_gate.lock();

        let (root, config) = {
            let g = self.inner.read();
            (g.root.clone(), g.config)
        };

        let res = insert_recursive(&root, &entry, &config, &self.metrics);
        let new_root = match res {
            InsertResult::Ok(n) => n,
            InsertResult::Split(a, b) => {
                let mbr = a.mbr.union(&b.mbr);
                Arc::new(Node {
                    mbr,
                    kind: NodeKind::Internal(vec![a, b]),
                })
            }
        };

        {
            let mut g = self.inner.write();
            g.root = new_root;
            g.live_entries += 1;
        }
        // Maintain the inverse map for O(1) delete-by-data lookup.
        self.inverse.lock().insert(entry.data.clone(), entry.mbr);
        self.metrics.inserts.fetch_add(1, Ordering::Relaxed);
    }

    /// Bulk insert: repeatedly calls insert. For large loads prefer bulk_load.
    pub fn insert_many(&self, entries: Vec<LeafEntry<T>>) {
        for e in entries {
            self.insert(e);
        }
    }
}

fn insert_recursive<T: Clone + Send + Sync + 'static>(
    node: &Arc<Node<T>>,
    entry: &LeafEntry<T>,
    config: &RTreeConfig,
    metrics: &Metrics,
) -> InsertResult<T> {
    match &node.kind {
        NodeKind::Leaf(entries) => {
            let mut new_entries: Vec<LeafEntry<T>> = Vec::with_capacity(entries.len() + 1);
            new_entries.extend_from_slice(entries);
            new_entries.push(entry.clone());
            if new_entries.len() <= config.max_fill {
                let mbr = compute_mbr_leaf(&new_entries);
                InsertResult::Ok(Arc::new(Node {
                    mbr,
                    kind: NodeKind::Leaf(new_entries),
                }))
            } else {
                let (left, right) = split_leaf(new_entries, config);
                metrics.splits.fetch_add(1, Ordering::Relaxed);
                InsertResult::Split(left, right)
            }
        }
        NodeKind::Internal(children) => {
            let idx = choose_subtree_index(children, &entry.mbr);
            let sub = insert_recursive(&children[idx], entry, config, metrics);

            match sub {
                InsertResult::Ok(new_child) => {
                    let mut new_children: Vec<Arc<Node<T>>> = children.clone();
                    new_children[idx] = new_child;
                    let mbr = compute_mbr_internal(&new_children);
                    InsertResult::Ok(Arc::new(Node {
                        mbr,
                        kind: NodeKind::Internal(new_children),
                    }))
                }
                InsertResult::Split(left, right) => {
                    let mut new_children: Vec<Arc<Node<T>>> =
                        Vec::with_capacity(children.len() + 1);
                    new_children.extend_from_slice(children);
                    new_children[idx] = left;
                    new_children.push(right);
                    if new_children.len() <= config.max_fill {
                        let mbr = compute_mbr_internal(&new_children);
                        InsertResult::Ok(Arc::new(Node {
                            mbr,
                            kind: NodeKind::Internal(new_children),
                        }))
                    } else {
                        let (l, r) = split_internal(new_children, config);
                        metrics.splits.fetch_add(1, Ordering::Relaxed);
                        InsertResult::Split(l, r)
                    }
                }
            }
        }
    }
}

/// Chooses the child whose MBR requires the least overlap enlargement (R*-tree
/// heuristic for leaf-parents; falls back to least area enlargement for others).
fn choose_subtree_index<T: Clone>(children: &[Arc<Node<T>>], entry_mbr: &Mbr) -> usize {
    // Detect whether children are leaves (their children are entries, not nodes)
    let children_are_leaf_parents = children.iter().all(|c| c.is_leaf());

    if children_are_leaf_parents {
        // Min overlap enlargement, ties -> min area enlargement, ties -> min area
        let mut best = 0;
        let mut best_overlap_delta = f64::INFINITY;
        let mut best_area_delta = f64::INFINITY;
        let mut best_area = f64::INFINITY;
        for (i, c) in children.iter().enumerate() {
            let enlarged = c.mbr.union(entry_mbr);
            // Overlap with siblings (before vs after enlargement)
            let mut before = 0.0;
            let mut after = 0.0;
            for (j, o) in children.iter().enumerate() {
                if j == i {
                    continue;
                }
                before += c.mbr.overlap(&o.mbr);
                after += enlarged.overlap(&o.mbr);
            }
            let overlap_delta = after - before;
            let area_delta = enlarged.area() - c.mbr.area();
            let area = c.mbr.area();
            let better = overlap_delta < best_overlap_delta
                || (overlap_delta == best_overlap_delta && area_delta < best_area_delta)
                || (overlap_delta == best_overlap_delta
                    && area_delta == best_area_delta
                    && area < best_area);
            if better {
                best = i;
                best_overlap_delta = overlap_delta;
                best_area_delta = area_delta;
                best_area = area;
            }
        }
        best
    } else {
        // Min area enlargement, ties -> min area
        let mut best = 0;
        let mut best_delta = f64::INFINITY;
        let mut best_area = f64::INFINITY;
        for (i, c) in children.iter().enumerate() {
            let delta = c.mbr.enlargement(entry_mbr);
            let area = c.mbr.area();
            if delta < best_delta || (delta == best_delta && area < best_area) {
                best = i;
                best_delta = delta;
                best_area = area;
            }
        }
        best
    }
}

fn collect_leaf_entries<T: Clone>(node: &Arc<Node<T>>, out: &mut Vec<LeafEntry<T>>) {
    match &node.kind {
        NodeKind::Leaf(entries) => {
            for e in entries {
                if !e.deleted {
                    out.push(e.clone());
                }
            }
        }
        NodeKind::Internal(children) => {
            for c in children {
                collect_leaf_entries(c, out);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Split (R*-tree axis + distribution selection, O(n) per axis/order via
// prefix/suffix MBR accumulation)
// ---------------------------------------------------------------------------

/// For a given sorted ordering, evaluates every valid split point k in
/// [min_fill, n - min_fill] using prefix/suffix MBRs. Returns
/// (total_margin_sum, best_k, best_overlap, best_area).
fn evaluate_sorted_split_leaf<T: Clone>(
    entries: &[LeafEntry<T>],
    order: &[usize],
    config: &RTreeConfig,
) -> (f64, usize, f64, f64) {
    let n = order.len();
    let dims = entries[order[0]].mbr.dims;

    // Prefix MBRs: prefix[i] = union of entries[order[0..=i]]
    let mut prefix: Vec<Mbr> = Vec::with_capacity(n);
    let mut acc = entries[order[0]].mbr;
    prefix.push(acc);
    for i in 1..n {
        acc.union_inplace(&entries[order[i]].mbr);
        prefix.push(acc);
    }

    // Suffix MBRs: suffix[i] = union of entries[order[i..]]
    let mut suffix: Vec<Mbr> = vec![Mbr::empty(dims); n];
    let mut acc = entries[order[n - 1]].mbr;
    suffix[n - 1] = acc;
    for i in (0..n - 1).rev() {
        acc.union_inplace(&entries[order[i]].mbr);
        suffix[i] = acc;
    }

    let mut total_margin = 0.0;
    let mut best_k = config.min_fill;
    let mut best_ov = f64::INFINITY;
    let mut best_ar = f64::INFINITY;

    for k in config.min_fill..=(n - config.min_fill) {
        let lm = prefix[k - 1];
        let rm = suffix[k];
        total_margin += lm.margin() + rm.margin();
        let ov = lm.overlap(&rm);
        let ar = lm.area() + rm.area();
        if ov < best_ov || (ov == best_ov && ar < best_ar) {
            best_ov = ov;
            best_ar = ar;
            best_k = k;
        }
    }
    (total_margin, best_k, best_ov, best_ar)
}

fn evaluate_sorted_split_internal<T: Clone>(
    children: &[Arc<Node<T>>],
    order: &[usize],
    config: &RTreeConfig,
) -> (f64, usize, f64, f64) {
    let n = order.len();
    let dims = children[order[0]].mbr.dims;

    let mut prefix: Vec<Mbr> = Vec::with_capacity(n);
    let mut acc = children[order[0]].mbr;
    prefix.push(acc);
    for i in 1..n {
        acc.union_inplace(&children[order[i]].mbr);
        prefix.push(acc);
    }

    let mut suffix: Vec<Mbr> = vec![Mbr::empty(dims); n];
    let mut acc = children[order[n - 1]].mbr;
    suffix[n - 1] = acc;
    for i in (0..n - 1).rev() {
        acc.union_inplace(&children[order[i]].mbr);
        suffix[i] = acc;
    }

    let mut total_margin = 0.0;
    let mut best_k = config.min_fill;
    let mut best_ov = f64::INFINITY;
    let mut best_ar = f64::INFINITY;

    for k in config.min_fill..=(n - config.min_fill) {
        let lm = prefix[k - 1];
        let rm = suffix[k];
        total_margin += lm.margin() + rm.margin();
        let ov = lm.overlap(&rm);
        let ar = lm.area() + rm.area();
        if ov < best_ov || (ov == best_ov && ar < best_ar) {
            best_ov = ov;
            best_ar = ar;
            best_k = k;
        }
    }
    (total_margin, best_k, best_ov, best_ar)
}

fn sort_order_leaf<T: Clone>(
    entries: &[LeafEntry<T>],
    axis: usize,
    sort_by_min: bool,
) -> Vec<usize> {
    let mut order: Vec<usize> = (0..entries.len()).collect();
    order.sort_by(|&a, &b| {
        let av = if sort_by_min {
            entries[a].mbr.mins[axis]
        } else {
            entries[a].mbr.maxs[axis]
        };
        let bv = if sort_by_min {
            entries[b].mbr.mins[axis]
        } else {
            entries[b].mbr.maxs[axis]
        };
        av.partial_cmp(&bv).unwrap_or(CmpOrdering::Equal)
    });
    order
}

fn sort_order_internal<T: Clone>(
    children: &[Arc<Node<T>>],
    axis: usize,
    sort_by_min: bool,
) -> Vec<usize> {
    let mut order: Vec<usize> = (0..children.len()).collect();
    order.sort_by(|&a, &b| {
        let av = if sort_by_min {
            children[a].mbr.mins[axis]
        } else {
            children[a].mbr.maxs[axis]
        };
        let bv = if sort_by_min {
            children[b].mbr.mins[axis]
        } else {
            children[b].mbr.maxs[axis]
        };
        av.partial_cmp(&bv).unwrap_or(CmpOrdering::Equal)
    });
    order
}

fn split_leaf<T: Clone>(
    entries: Vec<LeafEntry<T>>,
    config: &RTreeConfig,
) -> (Arc<Node<T>>, Arc<Node<T>>) {
    let dims = entries[0].mbr.dims as usize;

    // Pass 1: pick axis with minimum total-margin summed across both sort orders.
    let mut best_axis = 0;
    let mut best_axis_margin = f64::INFINITY;
    for axis in 0..dims {
        let order_min = sort_order_leaf(&entries, axis, true);
        let order_max = sort_order_leaf(&entries, axis, false);
        let (m1, _, _, _) = evaluate_sorted_split_leaf(&entries, &order_min, config);
        let (m2, _, _, _) = evaluate_sorted_split_leaf(&entries, &order_max, config);
        let total = m1 + m2;
        if total < best_axis_margin {
            best_axis_margin = total;
            best_axis = axis;
        }
    }

    // Pass 2: at chosen axis, pick (sort_order, k) with minimum overlap then area.
    let order_min = sort_order_leaf(&entries, best_axis, true);
    let order_max = sort_order_leaf(&entries, best_axis, false);
    let (_, k_min, ov_min, ar_min) = evaluate_sorted_split_leaf(&entries, &order_min, config);
    let (_, k_max, ov_max, ar_max) = evaluate_sorted_split_leaf(&entries, &order_max, config);

    let (winning_order, k) = if ov_min < ov_max || (ov_min == ov_max && ar_min <= ar_max) {
        (order_min, k_min)
    } else {
        (order_max, k_max)
    };

    // Materialize the chosen split.
    let mut left = Vec::with_capacity(k);
    let mut right = Vec::with_capacity(entries.len() - k);
    for (i, &src_idx) in winning_order.iter().enumerate() {
        // Use index-based moves via swap_remove trickery isn't safe since we
        // iterate winning_order by index. Just clone, same cost as before.
        if i < k {
            left.push(entries[src_idx].clone());
        } else {
            right.push(entries[src_idx].clone());
        }
    }

    let lmbr = compute_mbr_leaf(&left);
    let rmbr = compute_mbr_leaf(&right);
    (
        Arc::new(Node {
            mbr: lmbr,
            kind: NodeKind::Leaf(left),
        }),
        Arc::new(Node {
            mbr: rmbr,
            kind: NodeKind::Leaf(right),
        }),
    )
}

fn split_internal<T: Clone>(
    children: Vec<Arc<Node<T>>>,
    config: &RTreeConfig,
) -> (Arc<Node<T>>, Arc<Node<T>>) {
    let dims = children[0].mbr.dims as usize;

    let mut best_axis = 0;
    let mut best_axis_margin = f64::INFINITY;
    for axis in 0..dims {
        let order_min = sort_order_internal(&children, axis, true);
        let order_max = sort_order_internal(&children, axis, false);
        let (m1, _, _, _) = evaluate_sorted_split_internal(&children, &order_min, config);
        let (m2, _, _, _) = evaluate_sorted_split_internal(&children, &order_max, config);
        let total = m1 + m2;
        if total < best_axis_margin {
            best_axis_margin = total;
            best_axis = axis;
        }
    }

    let order_min = sort_order_internal(&children, best_axis, true);
    let order_max = sort_order_internal(&children, best_axis, false);
    let (_, k_min, ov_min, ar_min) = evaluate_sorted_split_internal(&children, &order_min, config);
    let (_, k_max, ov_max, ar_max) = evaluate_sorted_split_internal(&children, &order_max, config);

    let (winning_order, k) = if ov_min < ov_max || (ov_min == ov_max && ar_min <= ar_max) {
        (order_min, k_min)
    } else {
        (order_max, k_max)
    };

    let mut left = Vec::with_capacity(k);
    let mut right = Vec::with_capacity(children.len() - k);
    for (i, &src_idx) in winning_order.iter().enumerate() {
        if i < k {
            left.push(children[src_idx].clone());
        } else {
            right.push(children[src_idx].clone());
        }
    }

    let lmbr = compute_mbr_internal(&left);
    let rmbr = compute_mbr_internal(&right);
    (
        Arc::new(Node {
            mbr: lmbr,
            kind: NodeKind::Internal(left),
        }),
        Arc::new(Node {
            mbr: rmbr,
            kind: NodeKind::Internal(right),
        }),
    )
}

// ---------------------------------------------------------------------------
// Delete (tombstone based)
// ---------------------------------------------------------------------------

impl<T: Clone + Send + Sync + std::hash::Hash + Eq + 'static> RTree<T> {
    /// Marks the first entry matching (mbr, data) as deleted. Returns true if
    /// an entry was marked. Live entry count decreases; tombstone count
    /// increases. Background compaction rebuilds the tree when the tombstone
    /// ratio exceeds the threshold.
    pub fn delete(&self, mbr: &Mbr, data: &T) -> bool {
        let _write = self.write_gate.lock();
        let (root, live_entries, tombstones) = {
            let g = self.inner.read();
            (g.root.clone(), g.live_entries, g.tombstones)
        };

        let (new_root, marked) = mark_tombstone(&root, mbr, data);
        if !marked {
            return false;
        }

        {
            let mut g = self.inner.write();
            g.root = new_root;
            g.live_entries = live_entries - 1;
            g.tombstones = tombstones + 1;
        }
        // Drop from the inverse map so a subsequent delete-by-data is a no-op.
        self.inverse.lock().remove(data);
        self.metrics.deletes.fetch_add(1, Ordering::Relaxed);

        // Maybe compact
        let should_compact = {
            let g = self.inner.read();
            let total = g.live_entries + g.tombstones;
            total >= COMPACTION_MIN_ENTRIES
                && (g.tombstones as f64) / (total as f64) >= COMPACTION_TOMBSTONE_RATIO
        };
        if should_compact {
            drop(_write);
            self.compact();
        }

        true
    }

    /// Deletes the entry whose data field equals `data`. O(1) MBR lookup
    /// via the inverse map plus O(log N) tree delete.
    pub fn delete_by_data(&self, data: &T) -> bool {
        let mbr = match self.inverse.lock().get(data).copied() {
            Some(m) => m,
            None => return false,
        };
        self.delete(&mbr, data)
    }

    /// Rebuilds the tree from live entries only, dropping all tombstones.
    /// Safe to call at any time; writers are blocked during compaction.
    pub fn compact(&self) {
        let _write = self.write_gate.lock();
        let (root, dims, config) = {
            let g = self.inner.read();
            (g.root.clone(), g.dims, g.config)
        };

        let mut live = Vec::new();
        collect_leaf_entries(&root, &mut live);
        let new_root = bulk_build(live.clone(), dims, &config);
        let stats = new_root.stats();

        {
            let mut g = self.inner.write();
            g.root = new_root;
            g.live_entries = stats.entry_count;
            g.tombstones = 0;
        }
        self.metrics.compactions.fetch_add(1, Ordering::Relaxed);
    }
}

fn mark_tombstone<T: Clone + PartialEq>(
    node: &Arc<Node<T>>,
    target_mbr: &Mbr,
    target_data: &T,
) -> (Arc<Node<T>>, bool) {
    match &node.kind {
        NodeKind::Leaf(entries) => {
            let mut new_entries = entries.clone();
            for e in new_entries.iter_mut() {
                if !e.deleted && &e.mbr == target_mbr && &e.data == target_data {
                    e.deleted = true;
                    let mbr = node.mbr;
                    return (
                        Arc::new(Node {
                            mbr,
                            kind: NodeKind::Leaf(new_entries),
                        }),
                        true,
                    );
                }
            }
            (node.clone(), false)
        }
        NodeKind::Internal(children) => {
            // Descend into any child whose MBR might contain the target
            for (i, c) in children.iter().enumerate() {
                if c.mbr.intersects(target_mbr) {
                    let (new_c, done) = mark_tombstone(c, target_mbr, target_data);
                    if done {
                        let mut new_children = children.clone();
                        new_children[i] = new_c;
                        let mbr = compute_mbr_internal(&new_children);
                        return (
                            Arc::new(Node {
                                mbr,
                                kind: NodeKind::Internal(new_children),
                            }),
                            true,
                        );
                    }
                }
            }
            (node.clone(), false)
        }
    }
}

// ---------------------------------------------------------------------------
// Queries
// ---------------------------------------------------------------------------

impl<T: Clone + Send + Sync + std::hash::Hash + Eq + 'static> RTree<T> {
    /// Returns every live entry whose MBR intersects the query rectangle.
    /// The search is wait-free: no locks are held during traversal.
    pub fn range(&self, query: &Mbr) -> Vec<LeafEntry<T>> {
        self.metrics.query_range.fetch_add(1, Ordering::Relaxed);
        let root = self.snapshot();
        let mut out = Vec::new();
        range_recursive(&root, query, &mut out);
        out
    }

    /// K-nearest-neighbor search. The `point` must have the same
    /// dimensionality as the tree. Returns entries ordered by increasing
    /// distance from the point (squared Euclidean for MBR; callers applying
    /// geodesic distance should re-rank).
    pub fn knn(&self, point: &[f64], k: usize) -> Vec<KnnHit<T>> {
        self.metrics.query_knn.fetch_add(1, Ordering::Relaxed);
        if k == 0 {
            return Vec::new();
        }
        let root = self.snapshot();
        let d = root.mbr.dims as usize;
        assert_eq!(point.len(), d, "knn point dim mismatch");

        // Branch-and-bound min-heap over node/entry mindist.
        let mut heap: BinaryHeap<HeapItem<T>> = BinaryHeap::new();
        heap.push(HeapItem {
            dist: root.mbr.mindist_squared(point),
            payload: HeapPayload::Node(root),
        });

        let mut results: Vec<KnnHit<T>> = Vec::with_capacity(k);
        while let Some(item) = heap.pop() {
            if results.len() >= k {
                break;
            }
            match item.payload {
                HeapPayload::Node(node) => match &node.kind {
                    NodeKind::Internal(children) => {
                        for c in children {
                            heap.push(HeapItem {
                                dist: c.mbr.mindist_squared(point),
                                payload: HeapPayload::Node(c.clone()),
                            });
                        }
                    }
                    NodeKind::Leaf(entries) => {
                        for e in entries {
                            if e.deleted {
                                continue;
                            }
                            heap.push(HeapItem {
                                dist: e.mbr.mindist_squared(point),
                                payload: HeapPayload::Entry(e.clone()),
                            });
                        }
                    }
                },
                HeapPayload::Entry(entry) => {
                    results.push(KnnHit {
                        entry,
                        dist_sq: item.dist,
                    });
                }
            }
        }
        results
    }

    /// Returns live entries whose MBR mindist from the point is <= radius.
    /// This is an index-accelerated candidate filter; callers should follow
    /// up with exact distance (e.g. haversine) on geographic data.
    pub fn dwithin(&self, point: &[f64], radius: f64) -> Vec<LeafEntry<T>> {
        self.metrics.query_dwithin.fetch_add(1, Ordering::Relaxed);
        let radius_sq = radius * radius;
        let root = self.snapshot();
        assert_eq!(point.len(), root.mbr.dims as usize);
        let mut out = Vec::new();
        dwithin_recursive(&root, point, radius_sq, &mut out);
        out
    }

    /// Returns live entries whose MBR intersects the query MBR (same as range
    /// but metrics-tracked as an intersects query).
    pub fn intersects(&self, query: &Mbr) -> Vec<LeafEntry<T>> {
        self.metrics
            .query_intersects
            .fetch_add(1, Ordering::Relaxed);
        let root = self.snapshot();
        let mut out = Vec::new();
        range_recursive(&root, query, &mut out);
        out
    }
}

/// Walks up to `max_depth` levels of the tree to estimate how many live
/// entries a range query would return. At each internal node:
///   - If the node MBR does not intersect the query, contributes 0.
///   - If the query fully contains the node MBR, contributes the node's
///     full live entry count (cheap, no further recursion).
///   - Otherwise if we've reached max_depth, estimates based on the
///     fraction of the node's MBR that overlaps the query.
///   - Otherwise recurses into children.
/// At leaf nodes it counts actual entries that intersect the query.
fn estimate_range_recursive<T: Clone>(
    node: &Arc<Node<T>>,
    query: &Mbr,
    max_depth: u32,
    depth: u32,
) -> u64 {
    if !node.mbr.intersects(query) {
        return 0;
    }
    if query.contains(&node.mbr) {
        let s = node.stats();
        return s.entry_count.saturating_sub(s.tombstone_count);
    }
    match &node.kind {
        NodeKind::Leaf(entries) => entries
            .iter()
            .filter(|e| !e.deleted && e.mbr.intersects(query))
            .count() as u64,
        NodeKind::Internal(children) => {
            if depth >= max_depth {
                // Ratio of overlap area to node area, applied to live count.
                let node_area = node.mbr.area().max(1e-12);
                let overlap_area = node.mbr.overlap(query);
                let ratio = (overlap_area / node_area).clamp(0.0, 1.0);
                let s = node.stats();
                let live = s.entry_count.saturating_sub(s.tombstone_count);
                (live as f64 * ratio).round() as u64
            } else {
                children
                    .iter()
                    .map(|c| estimate_range_recursive(c, query, max_depth, depth + 1))
                    .sum()
            }
        }
    }
}

fn range_recursive<T: Clone>(node: &Arc<Node<T>>, query: &Mbr, out: &mut Vec<LeafEntry<T>>) {
    if !node.mbr.intersects(query) {
        return;
    }
    match &node.kind {
        NodeKind::Leaf(entries) => {
            for e in entries {
                if !e.deleted && e.mbr.intersects(query) {
                    out.push(e.clone());
                }
            }
        }
        NodeKind::Internal(children) => {
            for c in children {
                range_recursive(c, query, out);
            }
        }
    }
}

fn dwithin_recursive<T: Clone>(
    node: &Arc<Node<T>>,
    point: &[f64],
    radius_sq: f64,
    out: &mut Vec<LeafEntry<T>>,
) {
    if node.mbr.mindist_squared(point) > radius_sq {
        return;
    }
    match &node.kind {
        NodeKind::Leaf(entries) => {
            for e in entries {
                if !e.deleted && e.mbr.mindist_squared(point) <= radius_sq {
                    out.push(e.clone());
                }
            }
        }
        NodeKind::Internal(children) => {
            for c in children {
                dwithin_recursive(c, point, radius_sq, out);
            }
        }
    }
}

#[derive(Debug, Clone)]
pub struct KnnHit<T: Clone> {
    pub entry: LeafEntry<T>,
    pub dist_sq: f64,
}

enum HeapPayload<T: Clone> {
    Node(Arc<Node<T>>),
    Entry(LeafEntry<T>),
}

struct HeapItem<T: Clone> {
    dist: f64,
    payload: HeapPayload<T>,
}

// BinaryHeap is a max-heap; invert for min-heap semantics.
impl<T: Clone> PartialEq for HeapItem<T> {
    fn eq(&self, other: &Self) -> bool {
        self.dist == other.dist
    }
}
impl<T: Clone> Eq for HeapItem<T> {}
impl<T: Clone> PartialOrd for HeapItem<T> {
    fn partial_cmp(&self, other: &Self) -> Option<CmpOrdering> {
        Some(self.cmp(other))
    }
}
impl<T: Clone> Ord for HeapItem<T> {
    fn cmp(&self, other: &Self) -> CmpOrdering {
        other
            .dist
            .partial_cmp(&self.dist)
            .unwrap_or(CmpOrdering::Equal)
    }
}

// ---------------------------------------------------------------------------
// Persistence
// ---------------------------------------------------------------------------

/// Trait for serializing per-entry user data into the index snapshot file.
/// Implement for the `T` parameter you store in `LeafEntry::data` (typically
/// a row id type). Serialization must be deterministic so that the same tree
/// produces the same checksum every time.
pub trait IndexEntryCodec: Sized {
    /// Encodes self into `out` (appended).
    fn encode(&self, out: &mut Vec<u8>);

    /// Decodes one value from `bytes`. Returns the value and the number of
    /// bytes consumed.
    fn decode(bytes: &[u8]) -> std::io::Result<(Self, usize)>;
}

impl IndexEntryCodec for u64 {
    fn encode(&self, out: &mut Vec<u8>) {
        out.extend_from_slice(&self.to_le_bytes());
    }
    fn decode(bytes: &[u8]) -> std::io::Result<(Self, usize)> {
        if bytes.len() < 8 {
            return Err(std::io::Error::new(
                std::io::ErrorKind::UnexpectedEof,
                "u64 entry truncated",
            ));
        }
        let v = u64::from_le_bytes([
            bytes[0], bytes[1], bytes[2], bytes[3], bytes[4], bytes[5], bytes[6], bytes[7],
        ]);
        Ok((v, 8))
    }
}

impl IndexEntryCodec for u32 {
    fn encode(&self, out: &mut Vec<u8>) {
        out.extend_from_slice(&self.to_le_bytes());
    }
    fn decode(bytes: &[u8]) -> std::io::Result<(Self, usize)> {
        if bytes.len() < 4 {
            return Err(std::io::Error::new(
                std::io::ErrorKind::UnexpectedEof,
                "u32 entry truncated",
            ));
        }
        let v = u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]);
        Ok((v, 4))
    }
}

/// File magic for spatial index snapshots: ASCII "ZRTR".
const SPATIAL_FILE_MAGIC: [u8; 4] = *b"ZRTR";

/// Snapshot file format version. Bumped on any binary-layout change so the
/// loader can refuse to read incompatible files (and rebuild from the base
/// table instead).
const SNAPSHOT_FORMAT_VERSION: u8 = 1;

/// Node kind tag in the serialized stream.
const NODE_TAG_LEAF: u8 = 0;
const NODE_TAG_INTERNAL: u8 = 1;

/// Errors returned by snapshot save/load operations.
#[derive(Debug)]
pub enum PersistError {
    Io(std::io::Error),
    /// File magic did not match. Caller should rebuild from base table.
    BadMagic,
    /// Algorithm or format version did not match the running build.
    /// Caller should rebuild from base table.
    VersionMismatch {
        file: u8,
        expected: u8,
    },
    /// Stored checksum did not match recomputed value. Corruption detected.
    ChecksumMismatch {
        stored: u32,
        computed: u32,
    },
    /// Header declared a dimensionality not supported by this build.
    InvalidDims(u8),
    /// Encoded entry data was truncated or malformed.
    Truncated(&'static str),
}

impl std::fmt::Display for PersistError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            PersistError::Io(e) => write!(f, "io error: {}", e),
            PersistError::BadMagic => write!(f, "spatial index file: bad magic"),
            PersistError::VersionMismatch { file, expected } => write!(
                f,
                "spatial index file version mismatch: file={} expected={}",
                file, expected
            ),
            PersistError::ChecksumMismatch { stored, computed } => write!(
                f,
                "spatial index file checksum mismatch: stored={:#x} computed={:#x}",
                stored, computed
            ),
            PersistError::InvalidDims(d) => write!(f, "spatial index file: invalid dims {}", d),
            PersistError::Truncated(s) => write!(f, "spatial index file truncated: {}", s),
        }
    }
}

impl std::error::Error for PersistError {}

impl From<std::io::Error> for PersistError {
    fn from(e: std::io::Error) -> Self {
        PersistError::Io(e)
    }
}

impl<T: Clone + Send + Sync + std::hash::Hash + Eq + IndexEntryCodec + 'static> RTree<T> {
    /// Atomically writes the tree to `path` as a snapshot. Writes to a
    /// temporary file in the same directory, fsyncs, then renames over the
    /// target. On crash, the snapshot is either fully present or absent.
    ///
    /// Format header (40 bytes):
    ///   [0..4]   magic "ZRTR"
    ///   [4]      central-hash algorithm version (zyron_common::ALGORITHM_VERSION)
    ///   [5]      snapshot format version
    ///   [6]      tree dimensionality (1..=4)
    ///   [7]      reserved
    ///   [8..10]  min_fill (u16 LE)
    ///   [10..12] max_fill (u16 LE)
    ///   [12..16] srid (u32 LE)
    ///   [16..24] live entry count (u64 LE)
    ///   [24..32] tombstone count (u64 LE)
    ///   [32..40] node section length in bytes (u64 LE)
    /// Node section: depth-first preorder serialization.
    /// Footer: 4-byte central hash32 over (header || node section).
    pub fn save_to(&self, path: &std::path::Path) -> Result<(), PersistError> {
        let snapshot = {
            let g = self.inner.read();
            (
                g.root.clone(),
                g.dims,
                g.config,
                g.live_entries,
                g.tombstones,
            )
        };
        let (root, dims, config, live, tombstones) = snapshot;

        let mut nodes_buf: Vec<u8> = Vec::with_capacity(8192);
        encode_node(&root, &mut nodes_buf);

        let mut header = [0u8; 40];
        header[0..4].copy_from_slice(&SPATIAL_FILE_MAGIC);
        header[4] = zyron_common::ALGORITHM_VERSION;
        header[5] = SNAPSHOT_FORMAT_VERSION;
        header[6] = dims;
        header[7] = 0;
        header[8..10].copy_from_slice(&(config.min_fill as u16).to_le_bytes());
        header[10..12].copy_from_slice(&(config.max_fill as u16).to_le_bytes());
        header[12..16].copy_from_slice(&config.srid.to_le_bytes());
        header[16..24].copy_from_slice(&live.to_le_bytes());
        header[24..32].copy_from_slice(&tombstones.to_le_bytes());
        header[32..40].copy_from_slice(&(nodes_buf.len() as u64).to_le_bytes());

        // Checksum: header section, phase separator, then node section.
        let checksum = {
            let mut h = zyron_common::Hasher::new();
            h.update(&header);
            h.finish_phase();
            h.update(&nodes_buf);
            h.finish32()
        };

        // Atomic write: temp file in same dir, fsync, rename.
        let parent = path.parent().ok_or_else(|| {
            PersistError::Io(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                "snapshot path has no parent directory",
            ))
        })?;
        std::fs::create_dir_all(parent)?;
        let file_name = path
            .file_name()
            .ok_or_else(|| {
                PersistError::Io(std::io::Error::new(
                    std::io::ErrorKind::InvalidInput,
                    "snapshot path has no file name",
                ))
            })?
            .to_owned();
        let mut tmp_name = file_name.clone();
        tmp_name.push(".tmp");
        let tmp_path = parent.join(tmp_name);

        {
            use std::io::Write;
            let mut f = std::fs::File::create(&tmp_path)?;
            f.write_all(&header)?;
            f.write_all(&nodes_buf)?;
            f.write_all(&checksum.to_le_bytes())?;
            f.sync_all()?;
        }
        std::fs::rename(&tmp_path, path)?;
        // fsync the directory entry too so the rename is durable.
        if let Ok(dir) = std::fs::File::open(parent) {
            let _ = dir.sync_all();
        }
        Ok(())
    }

    /// Reads a tree snapshot from `path`. Returns the loaded tree on success.
    /// On version mismatch, magic mismatch, or checksum mismatch the caller
    /// should rebuild from the base table.
    pub fn load_from(path: &std::path::Path) -> Result<Self, PersistError> {
        let bytes = std::fs::read(path)?;
        if bytes.len() < 40 + 4 {
            return Err(PersistError::Truncated("file shorter than header+checksum"));
        }

        let header = &bytes[..40];
        if header[0..4] != SPATIAL_FILE_MAGIC {
            return Err(PersistError::BadMagic);
        }
        let algo = header[4];
        if algo != zyron_common::ALGORITHM_VERSION {
            return Err(PersistError::VersionMismatch {
                file: algo,
                expected: zyron_common::ALGORITHM_VERSION,
            });
        }
        let fmt = header[5];
        if fmt != SNAPSHOT_FORMAT_VERSION {
            return Err(PersistError::VersionMismatch {
                file: fmt,
                expected: SNAPSHOT_FORMAT_VERSION,
            });
        }
        let dims = header[6];
        if dims < 1 || (dims as usize) > MAX_DIMS {
            return Err(PersistError::InvalidDims(dims));
        }
        let min_fill = u16::from_le_bytes([header[8], header[9]]) as usize;
        let max_fill = u16::from_le_bytes([header[10], header[11]]) as usize;
        let srid = u32::from_le_bytes([header[12], header[13], header[14], header[15]]);
        let live = u64::from_le_bytes([
            header[16], header[17], header[18], header[19], header[20], header[21], header[22],
            header[23],
        ]);
        let tombstones = u64::from_le_bytes([
            header[24], header[25], header[26], header[27], header[28], header[29], header[30],
            header[31],
        ]);
        let nodes_len = u64::from_le_bytes([
            header[32], header[33], header[34], header[35], header[36], header[37], header[38],
            header[39],
        ]) as usize;

        let nodes_end = 40 + nodes_len;
        if bytes.len() < nodes_end + 4 {
            return Err(PersistError::Truncated(
                "declared node section overflows file",
            ));
        }
        let nodes_section = &bytes[40..nodes_end];
        let stored_checksum = u32::from_le_bytes([
            bytes[nodes_end],
            bytes[nodes_end + 1],
            bytes[nodes_end + 2],
            bytes[nodes_end + 3],
        ]);

        let computed_checksum = {
            let mut h = zyron_common::Hasher::new();
            h.update(header);
            h.finish_phase();
            h.update(nodes_section);
            h.finish32()
        };
        if computed_checksum != stored_checksum {
            return Err(PersistError::ChecksumMismatch {
                stored: stored_checksum,
                computed: computed_checksum,
            });
        }

        let mut cursor = 0usize;
        let root = decode_node::<T>(nodes_section, &mut cursor, dims)?;
        if cursor != nodes_section.len() {
            return Err(PersistError::Truncated("trailing bytes in node section"));
        }

        let config = RTreeConfig {
            min_fill,
            max_fill,
            srid,
        };
        // Rebuild the inverse data -> mbr map from the loaded tree so
        // delete-by-data is O(1) without scanning.
        let mut all_entries = Vec::new();
        collect_leaf_entries(&root, &mut all_entries);
        let inverse_map: std::collections::HashMap<T, Mbr> =
            all_entries.into_iter().map(|e| (e.data, e.mbr)).collect();

        Ok(Self {
            inner: RwLock::new(Inner {
                root,
                dims,
                config,
                live_entries: live,
                tombstones,
            }),
            metrics: Metrics::default(),
            write_gate: Mutex::new(()),
            inverse: Mutex::new(inverse_map),
        })
    }
}

fn encode_mbr(mbr: &Mbr, out: &mut Vec<u8>) {
    out.push(mbr.dims);
    for i in 0..(mbr.dims as usize) {
        out.extend_from_slice(&mbr.mins[i].to_le_bytes());
        out.extend_from_slice(&mbr.maxs[i].to_le_bytes());
    }
}

fn decode_mbr(bytes: &[u8], cursor: &mut usize) -> Result<Mbr, PersistError> {
    if bytes.len() < *cursor + 1 {
        return Err(PersistError::Truncated("mbr dims byte missing"));
    }
    let dims = bytes[*cursor];
    *cursor += 1;
    if dims < 1 || (dims as usize) > MAX_DIMS {
        return Err(PersistError::InvalidDims(dims));
    }
    let coord_bytes = (dims as usize) * 16;
    if bytes.len() < *cursor + coord_bytes {
        return Err(PersistError::Truncated("mbr coords truncated"));
    }
    let mut mins = [0.0f64; MAX_DIMS];
    let mut maxs = [0.0f64; MAX_DIMS];
    for i in 0..(dims as usize) {
        let mn = f64::from_le_bytes(bytes[*cursor..*cursor + 8].try_into().expect("8 bytes"));
        *cursor += 8;
        let mx = f64::from_le_bytes(bytes[*cursor..*cursor + 8].try_into().expect("8 bytes"));
        *cursor += 8;
        mins[i] = mn;
        maxs[i] = mx;
    }
    Ok(Mbr { dims, mins, maxs })
}

fn encode_node<T: Clone + IndexEntryCodec>(node: &Arc<Node<T>>, out: &mut Vec<u8>) {
    encode_mbr(&node.mbr, out);
    match &node.kind {
        NodeKind::Leaf(entries) => {
            out.push(NODE_TAG_LEAF);
            out.extend_from_slice(&(entries.len() as u32).to_le_bytes());
            for e in entries {
                encode_mbr(&e.mbr, out);
                out.push(if e.deleted { 1 } else { 0 });
                e.data.encode(out);
            }
        }
        NodeKind::Internal(children) => {
            out.push(NODE_TAG_INTERNAL);
            out.extend_from_slice(&(children.len() as u32).to_le_bytes());
            for c in children {
                encode_node(c, out);
            }
        }
    }
}

fn decode_node<T: Clone + Send + Sync + 'static + IndexEntryCodec>(
    bytes: &[u8],
    cursor: &mut usize,
    expected_dims: u8,
) -> Result<Arc<Node<T>>, PersistError> {
    let mbr = decode_mbr(bytes, cursor)?;
    if mbr.dims != expected_dims && !mbr.is_empty() {
        return Err(PersistError::InvalidDims(mbr.dims));
    }
    if bytes.len() < *cursor + 1 {
        return Err(PersistError::Truncated("node tag missing"));
    }
    let tag = bytes[*cursor];
    *cursor += 1;
    if bytes.len() < *cursor + 4 {
        return Err(PersistError::Truncated("node count missing"));
    }
    let count =
        u32::from_le_bytes(bytes[*cursor..*cursor + 4].try_into().expect("4 bytes")) as usize;
    *cursor += 4;

    match tag {
        NODE_TAG_LEAF => {
            let mut entries = Vec::with_capacity(count);
            for _ in 0..count {
                let entry_mbr = decode_mbr(bytes, cursor)?;
                if bytes.len() < *cursor + 1 {
                    return Err(PersistError::Truncated("entry deleted byte missing"));
                }
                let deleted = bytes[*cursor] != 0;
                *cursor += 1;
                let (data, consumed) = T::decode(&bytes[*cursor..]).map_err(|e| {
                    PersistError::Truncated(Box::leak(
                        format!("entry decode: {}", e).into_boxed_str(),
                    ))
                })?;
                *cursor += consumed;
                entries.push(LeafEntry {
                    mbr: entry_mbr,
                    data,
                    deleted,
                });
            }
            Ok(Arc::new(Node {
                mbr,
                kind: NodeKind::Leaf(entries),
            }))
        }
        NODE_TAG_INTERNAL => {
            let mut children = Vec::with_capacity(count);
            for _ in 0..count {
                children.push(decode_node::<T>(bytes, cursor, expected_dims)?);
            }
            Ok(Arc::new(Node {
                mbr,
                kind: NodeKind::Internal(children),
            }))
        }
        _ => Err(PersistError::Truncated("invalid node tag")),
    }
}

// ---------------------------------------------------------------------------
// Multi-index manager
// ---------------------------------------------------------------------------

/// Holds all live spatial indexes keyed by catalog index id. Instantiated
/// once on server startup and shared across the executor (for queries) and
/// DML operators (for maintenance on insert/update/delete).
pub struct SpatialIndexManager {
    /// index_id -> RTree<rowid>. Each index is wrapped in its own Arc so
    /// snapshot-based reads can hand a clone to query operators without
    /// blocking concurrent writers.
    indexes: RwLock<std::collections::HashMap<u32, Arc<RTree<u64>>>>,
}

impl Default for SpatialIndexManager {
    fn default() -> Self {
        Self::new()
    }
}

impl SpatialIndexManager {
    pub fn new() -> Self {
        Self {
            indexes: RwLock::new(std::collections::HashMap::new()),
        }
    }

    /// Creates an empty index for the given dimensionality. Replaces any
    /// existing index under the same id (caller is responsible for matching
    /// catalog state).
    pub fn create_index(&self, index_id: u32, dims: u8, srid: u32) {
        let cfg = RTreeConfig {
            srid,
            ..RTreeConfig::default()
        };
        let tree = Arc::new(RTree::<u64>::with_config(dims, cfg));
        self.indexes.write().insert(index_id, tree);
    }

    /// Drops an index. No-op if not present.
    pub fn drop_index(&self, index_id: u32) {
        self.indexes.write().remove(&index_id);
    }

    /// Returns a handle to the named index for reads or maintenance writes.
    /// Holds an `Arc<RTree>` so concurrent readers see consistent snapshots.
    pub fn get(&self, index_id: u32) -> Option<Arc<RTree<u64>>> {
        self.indexes.read().get(&index_id).cloned()
    }

    /// Estimates the number of live entries the tree would return for a
    /// query of the given shape. Used by the query planner's cost model to
    /// decide between a spatial-indexed scan and a full table scan.
    ///
    /// Walks the top levels of the tree (up to `max_depth`) and counts
    /// leaf MBR intersections with the query rectangle. Early termination
    /// at a subtree when its MBR is fully contained returns the subtree's
    /// full entry_count via Node::stats, avoiding deep recursion.
    pub fn estimate_range_cardinality(
        &self,
        index_id: u32,
        query: &Mbr,
        max_depth: u32,
    ) -> Option<u64> {
        let tree = self.get(index_id)?;
        let root = tree.snapshot();
        Some(estimate_range_recursive(&root, query, max_depth, 0))
    }

    /// Number of indexes currently registered.
    pub fn len(&self) -> usize {
        self.indexes.read().len()
    }

    /// True if no indexes are registered.
    pub fn is_empty(&self) -> bool {
        self.indexes.read().is_empty()
    }

    /// All registered index ids. Used by the catalog persistence pass to
    /// emit snapshots for every live index at checkpoint time.
    pub fn index_ids(&self) -> Vec<u32> {
        self.indexes.read().keys().copied().collect()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // --- Mbr ---

    #[test]
    fn mbr_point_basic() {
        let m = Mbr::point(&[1.0, 2.0]);
        assert_eq!(m.dims, 2);
        assert!(m.contains_point(&[1.0, 2.0]));
        assert!(!m.contains_point(&[1.1, 2.0]));
        assert_eq!(m.area(), 0.0);
    }

    #[test]
    fn mbr_from_extents() {
        let m = Mbr::from_extents(&[0.0, 0.0], &[2.0, 3.0]);
        assert_eq!(m.area(), 6.0);
        assert_eq!(m.margin(), 5.0);
        assert!(m.contains_point(&[1.0, 1.0]));
        assert!(!m.contains_point(&[2.1, 1.0]));
    }

    #[test]
    fn mbr_union_and_enlargement() {
        let a = Mbr::from_extents(&[0.0, 0.0], &[1.0, 1.0]);
        let b = Mbr::from_extents(&[2.0, 2.0], &[3.0, 3.0]);
        let u = a.union(&b);
        assert_eq!(u, Mbr::from_extents(&[0.0, 0.0], &[3.0, 3.0]));
        let delta = a.enlargement(&b);
        assert!(delta > 0.0);
    }

    #[test]
    fn mbr_intersects_and_overlap() {
        let a = Mbr::from_extents(&[0.0, 0.0], &[2.0, 2.0]);
        let b = Mbr::from_extents(&[1.0, 1.0], &[3.0, 3.0]);
        assert!(a.intersects(&b));
        let ov = a.overlap(&b);
        assert!((ov - 1.0).abs() < 1e-9);
    }

    #[test]
    fn mbr_mindist() {
        let m = Mbr::from_extents(&[0.0, 0.0], &[10.0, 10.0]);
        // Inside
        assert_eq!(m.mindist_squared(&[5.0, 5.0]), 0.0);
        // Outside right
        assert_eq!(m.mindist_squared(&[13.0, 5.0]), 9.0);
        // Outside corner
        let d = m.mindist_squared(&[13.0, 14.0]);
        assert!((d - 25.0).abs() < 1e-9);
    }

    #[test]
    fn mbr_3d() {
        let m = Mbr::from_extents(&[0.0, 0.0, 0.0], &[1.0, 2.0, 4.0]);
        assert_eq!(m.dims, 3);
        assert_eq!(m.area(), 8.0);
    }

    #[test]
    fn mbr_4d_xyzm() {
        let m = Mbr::from_extents(&[0.0; 4], &[1.0, 2.0, 4.0, 8.0]);
        assert_eq!(m.dims, 4);
        assert_eq!(m.area(), 64.0);
    }

    // --- Bulk load ---

    #[test]
    fn bulk_load_builds_balanced_tree() {
        let tree = RTree::<u64>::new(2);
        let mut entries = Vec::new();
        for i in 0..1000 {
            let x = (i % 32) as f64;
            let y = (i / 32) as f64;
            entries.push(LeafEntry {
                mbr: Mbr::point(&[x, y]),
                data: i as u64,
                deleted: false,
            });
        }
        tree.bulk_load(entries);
        assert_eq!(tree.len(), 1000);
        let stats = tree.stats();
        assert!(stats.height <= 4, "tree too tall: {}", stats.height);
        assert_eq!(stats.entry_count, 1000);
    }

    #[test]
    fn bulk_load_parallel_path_correct() {
        // Input size above PARALLEL_BULK_LOAD_THRESHOLD forces
        // str_tile_leaves to recurse in parallel. Correctness check:
        // same entries produce the same live count and same range-query
        // results as a small (sequential) tree with the identical points.
        let tree = RTree::<u64>::new(2);
        let mut entries = Vec::with_capacity(50_000);
        for i in 0..50_000u64 {
            let x = (i % 250) as f64;
            let y = (i / 250) as f64;
            entries.push(LeafEntry {
                mbr: Mbr::point(&[x, y]),
                data: i,
                deleted: false,
            });
        }
        tree.bulk_load(entries);
        assert_eq!(tree.len(), 50_000);

        // Range query: 10x10 square starting at (20, 30) -> 100 points
        let q = Mbr::from_extents(&[20.0, 30.0], &[29.0, 39.0]);
        let hits = tree.range(&q);
        assert_eq!(hits.len(), 100);
    }

    #[test]
    fn bulk_load_empty() {
        let tree = RTree::<u64>::new(2);
        tree.bulk_load(vec![]);
        assert_eq!(tree.len(), 0);
        assert!(tree.is_empty());
    }

    // --- Insert ---

    #[test]
    fn insert_single_point_makes_tree_non_empty() {
        let tree = RTree::<u64>::new(2);
        tree.insert(LeafEntry {
            mbr: Mbr::point(&[1.0, 1.0]),
            data: 42,
            deleted: false,
        });
        assert_eq!(tree.len(), 1);
        let hits = tree.range(&Mbr::from_extents(&[0.0, 0.0], &[2.0, 2.0]));
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].data, 42);
    }

    #[test]
    fn insert_many_triggers_splits() {
        let tree = RTree::<u64>::new(2);
        for i in 0..500u64 {
            let x = (i % 50) as f64;
            let y = (i / 50) as f64;
            tree.insert(LeafEntry {
                mbr: Mbr::point(&[x, y]),
                data: i,
                deleted: false,
            });
        }
        assert_eq!(tree.len(), 500);
        let snap = tree.metrics().snapshot();
        assert!(snap.splits > 0, "no splits happened");
    }

    // --- Range ---

    #[test]
    fn range_returns_exact_matches() {
        let tree = RTree::<u64>::new(2);
        let entries: Vec<_> = (0..100u64)
            .map(|i| LeafEntry {
                mbr: Mbr::point(&[(i % 10) as f64, (i / 10) as f64]),
                data: i,
                deleted: false,
            })
            .collect();
        tree.bulk_load(entries);
        let q = Mbr::from_extents(&[3.0, 3.0], &[5.0, 5.0]);
        let hits = tree.range(&q);
        // Points with x in [3,5] and y in [3,5] = 3x3 = 9 points
        assert_eq!(hits.len(), 9);
    }

    // --- KNN ---

    #[test]
    fn knn_returns_nearest() {
        let tree = RTree::<u64>::new(2);
        let entries: Vec<_> = (0..100u64)
            .map(|i| LeafEntry {
                mbr: Mbr::point(&[(i % 10) as f64, (i / 10) as f64]),
                data: i,
                deleted: false,
            })
            .collect();
        tree.bulk_load(entries);
        let hits = tree.knn(&[0.0, 0.0], 3);
        assert_eq!(hits.len(), 3);
        // Nearest to origin: (0,0)=0, (1,0)=1, (0,1)=10 (in some order)
        let dists: Vec<f64> = hits.iter().map(|h| h.dist_sq).collect();
        assert!(dists.windows(2).all(|w| w[0] <= w[1]));
        assert_eq!(hits[0].dist_sq, 0.0);
    }

    #[test]
    fn knn_matches_brute_force() {
        let tree = RTree::<u64>::new(2);
        let mut rng = 0xdead_beef_u64;
        let mut entries = Vec::new();
        for i in 0..500u64 {
            // LCG for deterministic pseudo-random
            rng = rng
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let x = ((rng >> 16) as u32 % 1000) as f64;
            rng = rng
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let y = ((rng >> 16) as u32 % 1000) as f64;
            entries.push(LeafEntry {
                mbr: Mbr::point(&[x, y]),
                data: i,
                deleted: false,
            });
        }
        tree.bulk_load(entries.clone());

        let query = [500.0, 500.0];
        let k = 10;
        let hits = tree.knn(&query, k);
        assert_eq!(hits.len(), k);

        // Brute force
        let mut dists: Vec<(f64, u64)> = entries
            .iter()
            .map(|e| (e.mbr.mindist_squared(&query), e.data))
            .collect();
        dists.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(CmpOrdering::Equal));

        for i in 0..k {
            assert!(
                (hits[i].dist_sq - dists[i].0).abs() < 1e-9,
                "knn result {} distance mismatch",
                i
            );
        }
    }

    // --- DWithin ---

    #[test]
    fn dwithin_filters_by_radius() {
        let tree = RTree::<u64>::new(2);
        let entries: Vec<_> = (0..100u64)
            .map(|i| LeafEntry {
                mbr: Mbr::point(&[(i % 10) as f64, (i / 10) as f64]),
                data: i,
                deleted: false,
            })
            .collect();
        tree.bulk_load(entries);

        // Radius 1.5 around origin -> points (0,0), (1,0), (0,1), (1,1).
        // (1,1) has distance sqrt(2) ~= 1.414 which is <= 1.5.
        let hits = tree.dwithin(&[0.0, 0.0], 1.5);
        assert_eq!(hits.len(), 4);

        // Radius 1.0 -> only (0,0), (1,0), (0,1)
        let strict = tree.dwithin(&[0.0, 0.0], 1.0);
        assert_eq!(strict.len(), 3);
    }

    // --- Delete + compact ---

    #[test]
    fn delete_marks_tombstone() {
        let tree = RTree::<u64>::new(2);
        tree.insert(LeafEntry {
            mbr: Mbr::point(&[1.0, 1.0]),
            data: 1,
            deleted: false,
        });
        tree.insert(LeafEntry {
            mbr: Mbr::point(&[2.0, 2.0]),
            data: 2,
            deleted: false,
        });
        assert!(tree.delete(&Mbr::point(&[1.0, 1.0]), &1));
        assert_eq!(tree.len(), 1);
        assert_eq!(tree.tombstones(), 1);
        let hits = tree.range(&Mbr::from_extents(&[0.0, 0.0], &[3.0, 3.0]));
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].data, 2);
    }

    #[test]
    fn compact_removes_tombstones() {
        let tree = RTree::<u64>::new(2);
        for i in 0..1500u64 {
            tree.insert(LeafEntry {
                mbr: Mbr::point(&[(i % 50) as f64, (i / 50) as f64]),
                data: i,
                deleted: false,
            });
        }
        // Delete many (above threshold)
        for i in 0..400u64 {
            assert!(tree.delete(&Mbr::point(&[(i % 50) as f64, (i / 50) as f64]), &i));
        }
        // Compaction may have auto-triggered; force one to be sure
        tree.compact();
        assert_eq!(tree.tombstones(), 0);
        assert_eq!(tree.len(), 1100);
        let snap = tree.metrics().snapshot();
        assert!(snap.compactions >= 1);
    }

    // --- 3D ---

    #[test]
    fn three_dim_insert_and_query() {
        let tree = RTree::<u64>::new(3);
        for i in 0..100u64 {
            tree.insert(LeafEntry {
                mbr: Mbr::point(&[(i % 5) as f64, ((i / 5) % 5) as f64, (i / 25) as f64]),
                data: i,
                deleted: false,
            });
        }
        assert_eq!(tree.len(), 100);
        let hits = tree.knn(&[0.0, 0.0, 0.0], 1);
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].dist_sq, 0.0);
    }

    // --- Concurrent reads during writes ---

    #[test]
    fn concurrent_readers_see_consistent_snapshots() {
        use std::sync::Arc as StdArc;
        use std::thread;

        let tree = StdArc::new(RTree::<u64>::new(2));
        for i in 0..200u64 {
            tree.insert(LeafEntry {
                mbr: Mbr::point(&[(i % 20) as f64, (i / 20) as f64]),
                data: i,
                deleted: false,
            });
        }

        // Reader threads repeatedly query while a writer thread inserts.
        let mut handles = Vec::new();
        for _ in 0..4 {
            let t = tree.clone();
            handles.push(thread::spawn(move || {
                for _ in 0..500 {
                    let hits = t.range(&Mbr::from_extents(&[0.0, 0.0], &[30.0, 30.0]));
                    assert!(!hits.is_empty());
                    for e in &hits {
                        assert!(!e.deleted);
                    }
                }
            }));
        }

        let writer = {
            let t = tree.clone();
            thread::spawn(move || {
                for i in 200u64..400 {
                    t.insert(LeafEntry {
                        mbr: Mbr::point(&[(i % 20) as f64, (i / 20) as f64]),
                        data: i,
                        deleted: false,
                    });
                }
            })
        };

        writer.join().unwrap();
        for h in handles {
            h.join().unwrap();
        }

        assert_eq!(tree.len(), 400);
    }

    #[test]
    fn concurrent_multi_writer() {
        use std::sync::Arc as StdArc;
        use std::thread;

        let tree = StdArc::new(RTree::<u64>::new(2));
        let mut handles = Vec::new();
        for w in 0..4u64 {
            let t = tree.clone();
            handles.push(thread::spawn(move || {
                for i in 0..200u64 {
                    let id = w * 1000 + i;
                    t.insert(LeafEntry {
                        mbr: Mbr::point(&[(id % 50) as f64, (id / 50) as f64]),
                        data: id,
                        deleted: false,
                    });
                }
            }));
        }
        for h in handles {
            h.join().unwrap();
        }
        assert_eq!(tree.len(), 800);
    }

    // --- Metrics ---

    #[test]
    fn metrics_track_operations() {
        let tree = RTree::<u64>::new(2);
        for i in 0..50u64 {
            tree.insert(LeafEntry {
                mbr: Mbr::point(&[i as f64, i as f64]),
                data: i,
                deleted: false,
            });
        }
        let _ = tree.range(&Mbr::from_extents(&[0.0, 0.0], &[25.0, 25.0]));
        let _ = tree.knn(&[5.0, 5.0], 3);
        let _ = tree.dwithin(&[5.0, 5.0], 2.0);
        let _ = tree.intersects(&Mbr::from_extents(&[0.0, 0.0], &[10.0, 10.0]));

        let m = tree.metrics().snapshot();
        assert_eq!(m.inserts, 50);
        assert_eq!(m.query_range, 1);
        assert_eq!(m.query_knn, 1);
        assert_eq!(m.query_dwithin, 1);
        assert_eq!(m.query_intersects, 1);
    }

    // --- Config ---

    #[test]
    fn small_fanout_still_correct() {
        let tree = RTree::<u64>::with_config(
            2,
            RTreeConfig {
                min_fill: 2,
                max_fill: 4,
                srid: 4326,
            },
        );
        for i in 0..50u64 {
            tree.insert(LeafEntry {
                mbr: Mbr::point(&[(i % 7) as f64, (i / 7) as f64]),
                data: i,
                deleted: false,
            });
        }
        assert_eq!(tree.len(), 50);
        let hits = tree.range(&Mbr::from_extents(&[0.0, 0.0], &[10.0, 10.0]));
        assert_eq!(hits.len(), 50);
    }

    // --- Large scale correctness ---

    #[test]
    fn large_scale_knn_matches_brute_force() {
        let tree = RTree::<u64>::new(2);
        let mut rng = 0xcafe_babe_u64;
        let mut entries = Vec::new();
        for i in 0..5000u64 {
            rng = rng
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let x = ((rng >> 16) as u32 % 10000) as f64;
            rng = rng
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let y = ((rng >> 16) as u32 % 10000) as f64;
            entries.push(LeafEntry {
                mbr: Mbr::point(&[x, y]),
                data: i,
                deleted: false,
            });
        }
        tree.bulk_load(entries.clone());

        let query = [5000.0, 5000.0];
        let hits = tree.knn(&query, 20);
        let mut brute: Vec<f64> = entries
            .iter()
            .map(|e| e.mbr.mindist_squared(&query))
            .collect();
        brute.sort_by(|a, b| a.partial_cmp(b).unwrap_or(CmpOrdering::Equal));

        for i in 0..20 {
            assert!((hits[i].dist_sq - brute[i]).abs() < 1e-6);
        }
    }

    // --- Stats ---

    // --- Fuzz / property tests ---

    /// Invariant checks: live count + tombstone count match internal, every
    /// live entry is discoverable via a range query spanning the whole tree,
    /// no deleted entry ever appears in results.
    fn check_invariants<
        T: Clone + Send + Sync + std::hash::Hash + Eq + 'static + std::fmt::Debug,
    >(
        tree: &RTree<T>,
        expected_live: &std::collections::HashMap<u64, (Mbr, T)>,
    ) {
        assert_eq!(
            tree.len(),
            expected_live.len() as u64,
            "live count mismatch"
        );

        let stats = tree.stats();
        assert_eq!(
            stats.entry_count - stats.tombstone_count,
            expected_live.len() as u64,
            "stats entry_count - tombstones != expected_live"
        );

        // Every live entry is retrievable via range over the full extent
        if expected_live.is_empty() {
            return;
        }
        let d = tree.dims() as usize;
        let mut mins = [f64::INFINITY; MAX_DIMS];
        let mut maxs = [f64::NEG_INFINITY; MAX_DIMS];
        for (mbr, _) in expected_live.values() {
            for i in 0..d {
                if mbr.mins[i] < mins[i] {
                    mins[i] = mbr.mins[i];
                }
                if mbr.maxs[i] > maxs[i] {
                    maxs[i] = mbr.maxs[i];
                }
            }
        }
        let full = Mbr::from_extents(&mins[..d], &maxs[..d]);
        let found = tree.range(&full);
        let mut seen: std::collections::HashSet<Vec<u8>> = std::collections::HashSet::new();
        for hit in &found {
            assert!(!hit.deleted, "range returned a tombstone");
            let key = bincode_key(&hit.data);
            assert!(seen.insert(key), "range returned duplicate");
        }
        assert_eq!(
            found.len(),
            expected_live.len(),
            "range didn't return all live entries"
        );
    }

    fn bincode_key<T: std::fmt::Debug>(v: &T) -> Vec<u8> {
        format!("{:?}", v).into_bytes()
    }

    /// Randomized sequence of insert/delete/range/knn/compact operations with
    /// a deterministic seed so failures reproduce exactly.
    #[test]
    fn fuzz_random_operations() {
        let tree = RTree::<u64>::with_config(
            2,
            RTreeConfig {
                min_fill: 3,
                max_fill: 8,
                srid: 4326,
            },
        );
        let mut mirror: std::collections::HashMap<u64, (Mbr, u64)> =
            std::collections::HashMap::new();
        let mut rng = 0x5eed_cafe_u64;
        let mut next_id: u64 = 1;

        for _ in 0..4000 {
            rng = rng
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let op = (rng >> 48) & 0x7;
            match op {
                0 | 1 | 2 | 3 => {
                    // Insert
                    rng = rng
                        .wrapping_mul(6364136223846793005)
                        .wrapping_add(1442695040888963407);
                    let x = ((rng >> 16) as u32 % 500) as f64;
                    rng = rng
                        .wrapping_mul(6364136223846793005)
                        .wrapping_add(1442695040888963407);
                    let y = ((rng >> 16) as u32 % 500) as f64;
                    let id = next_id;
                    next_id += 1;
                    let mbr = Mbr::point(&[x, y]);
                    tree.insert(LeafEntry {
                        mbr,
                        data: id,
                        deleted: false,
                    });
                    mirror.insert(id, (mbr, id));
                }
                4 | 5 => {
                    // Delete a random existing entry
                    if mirror.is_empty() {
                        continue;
                    }
                    let keys: Vec<u64> = mirror.keys().copied().collect();
                    let pick = keys[((rng >> 16) as usize) % keys.len()];
                    let (mbr, data) = mirror[&pick].clone();
                    assert!(tree.delete(&mbr, &data));
                    mirror.remove(&pick);
                }
                6 => {
                    // Range query - verify result subset matches mirror
                    rng = rng
                        .wrapping_mul(6364136223846793005)
                        .wrapping_add(1442695040888963407);
                    let x0 = ((rng >> 16) as u32 % 500) as f64;
                    rng = rng
                        .wrapping_mul(6364136223846793005)
                        .wrapping_add(1442695040888963407);
                    let y0 = ((rng >> 16) as u32 % 500) as f64;
                    let x1 = (x0 + 50.0).min(499.0);
                    let y1 = (y0 + 50.0).min(499.0);
                    let q = Mbr::from_extents(&[x0, y0], &[x1, y1]);
                    let hits = tree.range(&q);
                    // Every hit must be in the mirror and intersect the query
                    for h in &hits {
                        assert!(!h.deleted);
                        assert!(mirror.contains_key(&h.data));
                        assert!(h.mbr.intersects(&q));
                    }
                    // Cross-check: count mirror entries that should match
                    let expected: usize = mirror.values().filter(|(m, _)| m.intersects(&q)).count();
                    assert_eq!(hits.len(), expected);
                }
                _ => {
                    // Occasional compaction
                    if tree.tombstones() > 0 {
                        tree.compact();
                    }
                }
            }
        }

        check_invariants(&tree, &mirror);

        // Final compaction + re-check
        tree.compact();
        check_invariants(&tree, &mirror);
    }

    /// Concurrent fuzz: multiple writers and readers hitting the tree at the
    /// same time. No crashes, no panics, no deleted entries in live results.
    #[test]
    fn fuzz_concurrent_operations() {
        use std::sync::Arc as StdArc;
        use std::sync::atomic::AtomicBool;
        use std::thread;

        let tree = StdArc::new(RTree::<u64>::new(2));
        // Pre-populate
        for i in 0..500u64 {
            tree.insert(LeafEntry {
                mbr: Mbr::point(&[(i % 25) as f64, (i / 25) as f64]),
                data: i,
                deleted: false,
            });
        }

        let stop = StdArc::new(AtomicBool::new(false));
        let mut reader_handles = Vec::new();
        let mut writer_handles = Vec::new();

        for _ in 0..4 {
            let t = tree.clone();
            let s = stop.clone();
            reader_handles.push(thread::spawn(move || {
                while !s.load(Ordering::Relaxed) {
                    let hits = t.range(&Mbr::from_extents(&[0.0, 0.0], &[30.0, 30.0]));
                    for h in &hits {
                        assert!(!h.deleted);
                    }
                    let _ = t.knn(&[10.0, 10.0], 5);
                    let _ = t.dwithin(&[5.0, 5.0], 3.0);
                }
            }));
        }

        for w in 0..2u64 {
            let t = tree.clone();
            writer_handles.push(thread::spawn(move || {
                for i in 0..300u64 {
                    let id = 10_000 + w * 1000 + i;
                    t.insert(LeafEntry {
                        mbr: Mbr::point(&[((id % 25) as f64), ((id / 25) % 25) as f64]),
                        data: id,
                        deleted: false,
                    });
                }
            }));
        }

        // Join all writers first (bounded work), then signal readers to stop.
        for h in writer_handles {
            h.join().unwrap();
        }
        stop.store(true, Ordering::Relaxed);
        for h in reader_handles {
            h.join().unwrap();
        }

        assert_eq!(tree.len(), 1100);
    }

    #[test]
    fn stats_reflect_structure() {
        let tree = RTree::<u64>::with_config(
            2,
            RTreeConfig {
                min_fill: 2,
                max_fill: 4,
                srid: 4326,
            },
        );
        for i in 0..100u64 {
            tree.insert(LeafEntry {
                mbr: Mbr::point(&[(i % 10) as f64, (i / 10) as f64]),
                data: i,
                deleted: false,
            });
        }
        let s = tree.stats();
        assert!(
            s.height >= 2,
            "expected height >= 2 for 100 entries at fanout 4"
        );
        assert_eq!(s.entry_count, 100);
        assert_eq!(s.tombstone_count, 0);
    }

    // --- Persistence ---

    fn temp_path(name: &str) -> std::path::PathBuf {
        let mut p = std::env::temp_dir();
        p.push(format!(
            "zyron_spatial_index_test_{}_{}",
            name,
            std::process::id()
        ));
        p
    }

    #[test]
    fn save_and_load_roundtrip() {
        let tree = RTree::<u64>::new(2);
        let entries: Vec<_> = (0..500u64)
            .map(|i| LeafEntry {
                mbr: Mbr::point(&[(i % 25) as f64, (i / 25) as f64]),
                data: i,
                deleted: false,
            })
            .collect();
        tree.bulk_load(entries);

        let path = temp_path("roundtrip");
        tree.save_to(&path).expect("save");

        let loaded = RTree::<u64>::load_from(&path).expect("load");
        assert_eq!(loaded.len(), 500);
        assert_eq!(loaded.dims(), 2);

        // Same query returns same results.
        let q = Mbr::from_extents(&[0.0, 0.0], &[10.0, 10.0]);
        let mut a: Vec<u64> = tree.range(&q).iter().map(|e| e.data).collect();
        let mut b: Vec<u64> = loaded.range(&q).iter().map(|e| e.data).collect();
        a.sort();
        b.sort();
        assert_eq!(a, b);

        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn save_with_tombstones_preserved() {
        let tree = RTree::<u64>::new(2);
        for i in 0..50u64 {
            tree.insert(LeafEntry {
                mbr: Mbr::point(&[i as f64, i as f64]),
                data: i,
                deleted: false,
            });
        }
        for i in 0..10u64 {
            tree.delete(&Mbr::point(&[i as f64, i as f64]), &i);
        }
        let live_before = tree.len();
        let tomb_before = tree.tombstones();

        let path = temp_path("tombstones");
        tree.save_to(&path).expect("save");
        let loaded = RTree::<u64>::load_from(&path).expect("load");
        assert_eq!(loaded.len(), live_before);
        assert_eq!(loaded.tombstones(), tomb_before);

        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn load_rejects_bad_magic() {
        let path = temp_path("bad_magic");
        // Full-size header with mismatched magic exercises the magic check
        // rather than the file-too-short truncation check.
        let mut header = vec![0u8; 40 + 4];
        header[0..4].copy_from_slice(b"NOPE");
        std::fs::write(&path, &header).expect("write");
        let r = RTree::<u64>::load_from(&path);
        assert!(
            matches!(r, Err(PersistError::BadMagic)),
            "expected BadMagic, got {:?}",
            r.as_ref().err()
        );
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn load_rejects_corrupted_checksum() {
        let tree = RTree::<u64>::new(2);
        for i in 0..100u64 {
            tree.insert(LeafEntry {
                mbr: Mbr::point(&[i as f64, (i * 2) as f64]),
                data: i,
                deleted: false,
            });
        }
        let path = temp_path("corrupt");
        tree.save_to(&path).expect("save");

        // Flip a byte in the middle of the file (likely in the node section).
        let mut bytes = std::fs::read(&path).expect("read");
        let middle = bytes.len() / 2;
        bytes[middle] ^= 0x01;
        std::fs::write(&path, &bytes).expect("write");

        let r = RTree::<u64>::load_from(&path);
        assert!(
            matches!(r, Err(PersistError::ChecksumMismatch { .. })),
            "expected checksum mismatch, got {:?}",
            r.as_ref().err()
        );
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn save_3d_roundtrip() {
        let tree = RTree::<u64>::new(3);
        for i in 0..100u64 {
            tree.insert(LeafEntry {
                mbr: Mbr::point(&[(i % 5) as f64, ((i / 5) % 5) as f64, (i / 25) as f64]),
                data: i,
                deleted: false,
            });
        }
        let path = temp_path("3d");
        tree.save_to(&path).expect("save");
        let loaded = RTree::<u64>::load_from(&path).expect("load");
        assert_eq!(loaded.len(), 100);
        assert_eq!(loaded.dims(), 3);
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn save_atomic_no_partial_files() {
        // Save then verify the temp file was cleaned up by the rename.
        let tree = RTree::<u64>::new(2);
        tree.insert(LeafEntry {
            mbr: Mbr::point(&[1.0, 1.0]),
            data: 42,
            deleted: false,
        });
        let path = temp_path("atomic");
        tree.save_to(&path).expect("save");

        let mut tmp = path.clone();
        tmp.set_extension("tmp");
        assert!(!tmp.exists(), "tmp file should be cleaned up by rename");
        assert!(path.exists());

        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn cardinality_estimate_matches_ground_truth() {
        // Build a 10K-point grid, register in manager, then ask for the
        // cardinality of a range covering 100 points. Estimate should be
        // within a reasonable margin of the exact count.
        let mgr = SpatialIndexManager::new();
        mgr.create_index(42, 2, 4326);
        let tree = mgr.get(42).unwrap();

        let mut entries = Vec::new();
        for i in 0..10_000u64 {
            let x = (i % 100) as f64;
            let y = (i / 100) as f64;
            entries.push(LeafEntry {
                mbr: Mbr::point(&[x, y]),
                data: i,
                deleted: false,
            });
        }
        tree.bulk_load(entries);

        let q = Mbr::from_extents(&[0.0, 0.0], &[9.0, 9.0]);
        let exact = tree.range(&q).len() as u64;
        assert_eq!(exact, 100);

        let est = mgr.estimate_range_cardinality(42, &q, 3).unwrap();
        // Allow +/- 25% since the estimator walks only `max_depth` levels
        // and falls back to an area-overlap ratio below that.
        assert!(
            est as f64 >= exact as f64 * 0.5 && est as f64 <= exact as f64 * 1.5,
            "estimate {} not within 50% of exact {}",
            est,
            exact
        );
    }

    #[test]
    fn save_empty_tree() {
        let tree = RTree::<u64>::new(2);
        let path = temp_path("empty");
        tree.save_to(&path).expect("save");
        let loaded = RTree::<u64>::load_from(&path).expect("load");
        assert_eq!(loaded.len(), 0);
        assert!(loaded.is_empty());
        let _ = std::fs::remove_file(&path);
    }
}
