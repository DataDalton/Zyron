//! Graph storage using Compressed Sparse Row (CSR) format for cache-friendly
//! traversal. The CSR representation is built on-demand from edge data and
//! cached for reuse across algorithm invocations.

use std::collections::HashMap;
use std::sync::Arc;

use parking_lot::RwLock;

use super::schema::NodeId;

/// Compressed Sparse Row (CSR) representation for cache-friendly graph traversal.
/// Node IDs are remapped to dense indices [0, node_count) for contiguous memory access.
pub struct CompactGraph {
    /// Number of nodes in the graph.
    pub node_count: usize,
    /// Offsets into the `edges` array. offsets[i] is the start index of
    /// node i's outgoing edges. Length: node_count + 1.
    /// offsets[node_count] == edges.len().
    pub offsets: Vec<u32>,
    /// Contiguous array of target node indices (dense-remapped).
    pub edges: Vec<u32>,
    /// Optional edge weights, parallel to edges[].
    pub weights: Option<Vec<f64>>,
    /// Maps dense index back to original NodeId.
    pub index_to_node_id: Vec<NodeId>,
    /// Maps original NodeId to dense index.
    pub node_id_to_index: HashMap<NodeId, u32>,
}

impl CompactGraph {
    /// Builds a CSR graph from a list of edges.
    /// Each edge is (from_node, to_node, optional_weight).
    /// Nodes are automatically discovered from the edge list and remapped
    /// to dense indices for contiguous memory access.
    pub fn build(edge_data: &[(NodeId, NodeId, Option<f64>)]) -> Self {
        if edge_data.is_empty() {
            return Self {
                node_count: 0,
                offsets: vec![0],
                edges: Vec::new(),
                weights: None,
                index_to_node_id: Vec::new(),
                node_id_to_index: HashMap::new(),
            };
        }

        // Collect unique node IDs
        let mut node_set: Vec<NodeId> = Vec::new();
        for (from, to, _) in edge_data {
            node_set.push(*from);
            node_set.push(*to);
        }
        node_set.sort_unstable();
        node_set.dedup();

        let node_count = node_set.len();

        // Build bidirectional mappings
        let mut node_id_to_index: HashMap<NodeId, u32> = HashMap::with_capacity(node_count);
        let mut index_to_node_id: Vec<NodeId> = Vec::with_capacity(node_count);
        for (idx, &nid) in node_set.iter().enumerate() {
            node_id_to_index.insert(nid, idx as u32);
            index_to_node_id.push(nid);
        }

        // Check if any edge has a weight
        let has_weights = edge_data.iter().any(|(_, _, w)| w.is_some());

        // Remap edges and sort by source
        let mut remapped: Vec<(u32, u32, Option<f64>)> = Vec::with_capacity(edge_data.len());
        for (from, to, weight) in edge_data {
            let src = node_id_to_index[from];
            let dst = node_id_to_index[to];
            remapped.push((src, dst, *weight));
        }
        remapped.sort_unstable_by_key(|&(src, dst, _)| (src, dst));

        // Build offsets and edges arrays
        let mut offsets: Vec<u32> = vec![0u32; node_count + 1];
        let mut edges: Vec<u32> = Vec::with_capacity(remapped.len());
        let mut weights: Vec<f64> = if has_weights {
            Vec::with_capacity(remapped.len())
        } else {
            Vec::new()
        };

        // Count edges per source node
        for &(src, _, _) in &remapped {
            offsets[src as usize + 1] += 1;
        }

        // Prefix sum to build offsets
        for i in 1..=node_count {
            offsets[i] += offsets[i - 1];
        }

        // Fill edges and weights
        for &(_, dst, weight) in &remapped {
            edges.push(dst);
            if has_weights {
                weights.push(weight.unwrap_or(1.0));
            }
        }

        Self {
            node_count,
            offsets,
            edges,
            weights: if has_weights { Some(weights) } else { None },
            index_to_node_id,
            node_id_to_index,
        }
    }

    /// Returns the number of edges in the graph.
    pub fn edge_count(&self) -> usize {
        self.edges.len()
    }

    /// Returns the out-degree of a node given its dense index.
    #[inline]
    pub fn out_degree(&self, node_idx: u32) -> u32 {
        self.offsets[node_idx as usize + 1] - self.offsets[node_idx as usize]
    }

    /// Returns an iterator over the neighbors of a node (by dense index).
    #[inline]
    pub fn neighbors(&self, node_idx: u32) -> &[u32] {
        let start = self.offsets[node_idx as usize] as usize;
        let end = self.offsets[node_idx as usize + 1] as usize;
        &self.edges[start..end]
    }

    /// Returns the edge weights for a node's outgoing edges.
    /// Returns None if the graph has no weights.
    #[inline]
    pub fn neighbor_weights(&self, node_idx: u32) -> Option<&[f64]> {
        self.weights.as_ref().map(|w| {
            let start = self.offsets[node_idx as usize] as usize;
            let end = self.offsets[node_idx as usize + 1] as usize;
            &w[start..end]
        })
    }

    /// Translates a NodeId to its dense index, if present.
    #[inline]
    pub fn to_index(&self, node_id: NodeId) -> Option<u32> {
        self.node_id_to_index.get(&node_id).copied()
    }

    /// Translates a dense index back to its original NodeId.
    #[inline]
    pub fn to_node_id(&self, idx: u32) -> NodeId {
        self.index_to_node_id[idx as usize]
    }
}

/// Cache for CSR graph representations, invalidated on DML changes to
/// the backing node/edge tables.
pub struct CsrCache {
    cache: RwLock<HashMap<String, Arc<CompactGraph>>>,
}

impl CsrCache {
    /// Creates a new empty CSR cache.
    pub fn new() -> Self {
        Self {
            cache: RwLock::new(HashMap::new()),
        }
    }

    /// Returns the cached CSR for the given graph schema name, if available.
    pub fn get(&self, schema_name: &str) -> Option<Arc<CompactGraph>> {
        self.cache.read().get(schema_name).cloned()
    }

    /// Stores a CSR in the cache for the given graph schema name.
    pub fn insert(&self, schema_name: String, graph: Arc<CompactGraph>) {
        self.cache.write().insert(schema_name, graph);
    }

    /// Invalidates the cached CSR for the given graph schema name.
    /// Called when DML operations modify the backing tables.
    pub fn invalidate(&self, schema_name: &str) {
        self.cache.write().remove(schema_name);
    }

    /// Clears all cached CSR representations.
    pub fn clear(&self) {
        self.cache.write().clear();
    }
}

impl Default for CsrCache {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_empty_graph() {
        let graph = CompactGraph::build(&[]);
        assert_eq!(graph.node_count, 0);
        assert_eq!(graph.edge_count(), 0);
        assert_eq!(graph.offsets.len(), 1);
    }

    #[test]
    fn test_simple_chain() {
        // A -> B -> C
        let edges = vec![(1u64, 2u64, None), (2, 3, None)];
        let graph = CompactGraph::build(&edges);

        assert_eq!(graph.node_count, 3);
        assert_eq!(graph.edge_count(), 2);

        let a = graph.to_index(1).expect("node 1");
        let b = graph.to_index(2).expect("node 2");
        let c = graph.to_index(3).expect("node 3");

        assert_eq!(graph.out_degree(a), 1);
        assert_eq!(graph.neighbors(a), &[b]);

        assert_eq!(graph.out_degree(b), 1);
        assert_eq!(graph.neighbors(b), &[c]);

        assert_eq!(graph.out_degree(c), 0);
        assert_eq!(graph.neighbors(c).len(), 0);
    }

    #[test]
    fn test_weighted_graph() {
        let edges = vec![
            (10u64, 20u64, Some(1.5)),
            (10, 30, Some(2.0)),
            (20, 30, Some(0.5)),
        ];
        let graph = CompactGraph::build(&edges);

        assert_eq!(graph.node_count, 3);
        assert!(graph.weights.is_some());

        let n10 = graph.to_index(10).expect("node 10");
        let weights = graph.neighbor_weights(n10).expect("weights");
        assert_eq!(weights.len(), 2);
    }

    #[test]
    fn test_node_id_roundtrip() {
        let edges = vec![(100u64, 200u64, None), (200, 300, None)];
        let graph = CompactGraph::build(&edges);

        for &nid in &[100u64, 200, 300] {
            let idx = graph.to_index(nid).expect("index");
            assert_eq!(graph.to_node_id(idx), nid);
        }
    }

    #[test]
    fn test_csr_cache() {
        let cache = CsrCache::new();

        assert!(cache.get("test_graph").is_none());

        let graph = Arc::new(CompactGraph::build(&[(1u64, 2u64, None)]));
        cache.insert("test_graph".to_string(), graph.clone());

        assert!(cache.get("test_graph").is_some());
        assert_eq!(cache.get("test_graph").expect("cached").node_count, 2);

        cache.invalidate("test_graph");
        assert!(cache.get("test_graph").is_none());
    }

    #[test]
    fn test_star_graph() {
        // Hub node 0 connected to nodes 1..5
        let edges: Vec<(u64, u64, Option<f64>)> = (1..=5).map(|i| (0u64, i, None)).collect();
        let graph = CompactGraph::build(&edges);

        assert_eq!(graph.node_count, 6);
        let hub = graph.to_index(0).expect("hub");
        assert_eq!(graph.out_degree(hub), 5);

        // Leaf nodes have no outgoing edges
        for leaf_id in 1..=5u64 {
            let leaf = graph.to_index(leaf_id).expect("leaf");
            assert_eq!(graph.out_degree(leaf), 0);
        }
    }
}
