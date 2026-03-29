//! Column-level data lineage graph.
//!
//! Tracks how data flows between columns across tables, views,
//! materialized views, and pipelines. Supports forward tracing
//! (where does this column flow to?), backward tracing (where does
//! this column come from?), and impact analysis (all downstream
//! nodes affected by a change).

use zyron_common::Result;

/// The kind of object a lineage node belongs to.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum LineageNodeKind {
    Table,
    View,
    MaterializedView,
    Pipeline,
}

impl std::fmt::Display for LineageNodeKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let label = match self {
            LineageNodeKind::Table => "Table",
            LineageNodeKind::View => "View",
            LineageNodeKind::MaterializedView => "MaterializedView",
            LineageNodeKind::Pipeline => "Pipeline",
        };
        write!(f, "{}", label)
    }
}

/// A node in the lineage graph representing a specific column
/// within a database object.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct LineageNode {
    pub kind: LineageNodeKind,
    pub objectName: String,
    pub columnName: String,
}

impl LineageNode {
    /// Creates a new lineage node.
    pub fn new(kind: LineageNodeKind, objectName: &str, columnName: &str) -> Self {
        Self {
            kind,
            objectName: objectName.to_string(),
            columnName: columnName.to_string(),
        }
    }

    /// Returns the lookup key for this node: "kind:object:column".
    pub fn toKey(&self) -> String {
        format!("{}:{}:{}", self.kind, self.objectName, self.columnName)
    }

    /// Returns the object-level key: "kind:object" (without column).
    fn objectKey(&self) -> String {
        format!("{}:{}", self.kind, self.objectName)
    }
}

/// Describes how data is transformed between lineage nodes.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum TransformType {
    /// Column is passed through without modification.
    Direct,
    /// Column participates in an aggregate function (SUM, COUNT, etc.).
    Aggregated,
    /// Column is used in a WHERE/HAVING filter condition.
    Filtered,
    /// Column is derived from a computed expression.
    Computed,
    /// Column is involved in a JOIN condition.
    Joined,
}

impl std::fmt::Display for TransformType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let label = match self {
            TransformType::Direct => "Direct",
            TransformType::Aggregated => "Aggregated",
            TransformType::Filtered => "Filtered",
            TransformType::Computed => "Computed",
            TransformType::Joined => "Joined",
        };
        write!(f, "{}", label)
    }
}

/// A directed edge in the lineage graph from source column to
/// target column, annotated with the type of transformation applied.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LineageEdge {
    pub source: LineageNode,
    pub target: LineageNode,
    pub transformType: TransformType,
}

/// Bidirectional lineage graph supporting forward and backward
/// traversal of column-level data flow.
///
/// Uses a two-level index: column-level edges in forward/reverse maps
/// for O(1) single-column lookup, plus an object-level index mapping
/// "kind:object" to its column keys for O(1) object-level removal
/// (no full-table scan).
pub struct LineageGraph {
    /// Maps source node key ("kind:object:column") to edges flowing out.
    forward: scc::HashMap<String, Vec<LineageEdge>>,
    /// Maps target node key ("kind:object:column") to edges flowing in.
    reverse: scc::HashMap<String, Vec<LineageEdge>>,
    /// Maps object key ("kind:object") to column keys registered under it.
    /// Enables O(1) lookup of all columns for removeEdgesFor.
    objectIndex: scc::HashMap<String, Vec<String>>,
}

impl LineageGraph {
    /// Creates an empty lineage graph.
    pub fn new() -> Self {
        Self {
            forward: scc::HashMap::new(),
            reverse: scc::HashMap::new(),
            objectIndex: scc::HashMap::new(),
        }
    }

    /// Adds a lineage edge from source to target with the given
    /// transform type. Updates forward, reverse, and object index.
    pub fn addEdge(
        &self,
        source: LineageNode,
        target: LineageNode,
        transformType: TransformType,
    ) -> Result<()> {
        let edge = LineageEdge {
            source: source.clone(),
            target: target.clone(),
            transformType,
        };

        let sourceKey = source.toKey();
        let targetKey = target.toKey();
        let sourceObjKey = source.objectKey();
        let targetObjKey = target.objectKey();

        // Add to forward map (source -> edges).
        let edgeClone = edge.clone();
        let fwdEntry = self.forward.entry_sync(sourceKey.clone());
        match fwdEntry {
            scc::hash_map::Entry::Occupied(mut occ) => {
                occ.get_mut().push(edgeClone);
            }
            scc::hash_map::Entry::Vacant(vac) => {
                vac.insert_entry(vec![edgeClone]);
            }
        }

        // Add to reverse map (target -> edges).
        let revEntry = self.reverse.entry_sync(targetKey.clone());
        match revEntry {
            scc::hash_map::Entry::Occupied(mut occ) => {
                occ.get_mut().push(edge);
            }
            scc::hash_map::Entry::Vacant(vac) => {
                vac.insert_entry(vec![edge]);
            }
        }

        // Update object index for both source and target.
        indexColumnKey(&self.objectIndex, &sourceObjKey, sourceKey);
        indexColumnKey(&self.objectIndex, &targetObjKey, targetKey);

        Ok(())
    }

    /// Removes all lineage edges for the given object (across all
    /// columns), in both forward and reverse maps. Uses the object
    /// index for O(1) key lookup instead of full-table scan.
    pub fn removeEdgesFor(&self, kind: &LineageNodeKind, objectName: &str) -> Result<()> {
        let objKey = format!("{}:{}", kind, objectName);

        // Get all column keys for this object from the index.
        let columnKeys: Vec<String> = self
            .objectIndex
            .read_sync(&objKey, |_k, v| v.clone())
            .unwrap_or_default();

        // Remove forward entries and clean up corresponding reverse entries.
        for colKey in &columnKeys {
            let edges: Vec<LineageEdge> = self
                .forward
                .read_sync(colKey, |_k, v| v.clone())
                .unwrap_or_default();

            for edge in &edges {
                let targetKey = edge.target.toKey();
                let entry = self.reverse.entry_sync(targetKey);
                if let scc::hash_map::Entry::Occupied(mut occ) = entry {
                    occ.get_mut()
                        .retain(|e| !(e.source.kind == *kind && e.source.objectName == objectName));
                }
            }
            let _ = self.forward.remove_sync(colKey);
        }

        // Remove reverse entries and clean up corresponding forward entries.
        for colKey in &columnKeys {
            let edges: Vec<LineageEdge> = self
                .reverse
                .read_sync(colKey, |_k, v| v.clone())
                .unwrap_or_default();

            for edge in &edges {
                let sourceKey = edge.source.toKey();
                let entry = self.forward.entry_sync(sourceKey);
                if let scc::hash_map::Entry::Occupied(mut occ) = entry {
                    occ.get_mut()
                        .retain(|e| !(e.target.kind == *kind && e.target.objectName == objectName));
                }
            }
            let _ = self.reverse.remove_sync(colKey);
        }

        // Remove the object from the index.
        let _ = self.objectIndex.remove_sync(&objKey);

        Ok(())
    }

    /// Traces forward from a node: returns all edges where this node
    /// is the source. Answers "where does this column flow to?"
    pub fn traceForward(&self, node: &LineageNode) -> Vec<LineageEdge> {
        let key = node.toKey();
        self.forward
            .read_sync(&key, |_k, v| v.clone())
            .unwrap_or_default()
    }

    /// Traces backward from a node: returns all edges where this node
    /// is the target. Answers "where does this column come from?"
    pub fn traceBackward(&self, node: &LineageNode) -> Vec<LineageEdge> {
        let key = node.toKey();
        self.reverse
            .read_sync(&key, |_k, v| v.clone())
            .unwrap_or_default()
    }

    /// Performs impact analysis: finds all downstream nodes reachable
    /// from the given column via transitive forward edges. Uses BFS
    /// to traverse the graph.
    pub fn impactAnalysis(
        &self,
        kind: &LineageNodeKind,
        objectName: &str,
        columnName: &str,
    ) -> Vec<LineageNode> {
        let startNode = LineageNode::new(kind.clone(), objectName, columnName);
        let mut visited = hashbrown::HashSet::new();
        let mut result = Vec::new();
        let mut queue = std::collections::VecDeque::new();

        visited.insert(startNode.toKey());
        queue.push_back(startNode);

        while let Some(current) = queue.pop_front() {
            let edges = self.traceForward(&current);
            for edge in edges {
                let targetKey = edge.target.toKey();
                if visited.insert(targetKey) {
                    result.push(edge.target.clone());
                    queue.push_back(edge.target);
                }
            }
        }

        result
    }

    /// Returns the total number of forward edges in the graph.
    pub fn edgeCount(&self) -> usize {
        let mut count = 0;
        self.forward.iter_sync(|_k, v| {
            count += v.len();
            true
        });
        count
    }
}

impl Default for LineageGraph {
    fn default() -> Self {
        Self::new()
    }
}

/// Add a column key to the object index if not already present.
fn indexColumnKey(objectIndex: &scc::HashMap<String, Vec<String>>, objKey: &str, colKey: String) {
    let entry = objectIndex.entry_sync(objKey.to_string());
    match entry {
        scc::hash_map::Entry::Occupied(mut occ) => {
            let list = occ.get_mut();
            if !list.contains(&colKey) {
                list.push(colKey);
            }
        }
        scc::hash_map::Entry::Vacant(vac) => {
            vac.insert_entry(vec![colKey]);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tableNode(table: &str, column: &str) -> LineageNode {
        LineageNode::new(LineageNodeKind::Table, table, column)
    }

    fn viewNode(view: &str, column: &str) -> LineageNode {
        LineageNode::new(LineageNodeKind::View, view, column)
    }

    fn mvNode(mv: &str, column: &str) -> LineageNode {
        LineageNode::new(LineageNodeKind::MaterializedView, mv, column)
    }

    fn pipelineNode(pipe: &str, column: &str) -> LineageNode {
        LineageNode::new(LineageNodeKind::Pipeline, pipe, column)
    }

    #[test]
    fn test_add_edge_and_trace_forward() {
        let graph = LineageGraph::new();
        let src = tableNode("users", "email");
        let tgt = viewNode("user_emails", "email");

        graph
            .addEdge(src.clone(), tgt.clone(), TransformType::Direct)
            .expect("add edge");

        let edges = graph.traceForward(&src);
        assert_eq!(edges.len(), 1);
        assert_eq!(edges[0].target, tgt);
        assert_eq!(edges[0].transformType, TransformType::Direct);
    }

    #[test]
    fn test_add_edge_and_trace_backward() {
        let graph = LineageGraph::new();
        let src = tableNode("orders", "amount");
        let tgt = mvNode("order_summary", "total_amount");

        graph
            .addEdge(src.clone(), tgt.clone(), TransformType::Aggregated)
            .expect("add edge");

        let edges = graph.traceBackward(&tgt);
        assert_eq!(edges.len(), 1);
        assert_eq!(edges[0].source, src);
        assert_eq!(edges[0].transformType, TransformType::Aggregated);
    }

    #[test]
    fn test_trace_empty() {
        let graph = LineageGraph::new();
        let node = tableNode("missing", "col");
        assert!(graph.traceForward(&node).is_empty());
        assert!(graph.traceBackward(&node).is_empty());
    }

    #[test]
    fn test_multiple_edges_same_source() {
        let graph = LineageGraph::new();
        let src = tableNode("users", "id");
        let tgt1 = viewNode("active_users", "user_id");
        let tgt2 = mvNode("user_stats", "user_id");

        graph
            .addEdge(src.clone(), tgt1.clone(), TransformType::Direct)
            .expect("add");
        graph
            .addEdge(src.clone(), tgt2.clone(), TransformType::Filtered)
            .expect("add");

        let edges = graph.traceForward(&src);
        assert_eq!(edges.len(), 2);
    }

    #[test]
    fn test_multiple_edges_same_target() {
        let graph = LineageGraph::new();
        let src1 = tableNode("orders", "amount");
        let src2 = tableNode("orders", "tax");
        let tgt = mvNode("revenue", "total");

        graph
            .addEdge(src1.clone(), tgt.clone(), TransformType::Computed)
            .expect("add");
        graph
            .addEdge(src2.clone(), tgt.clone(), TransformType::Computed)
            .expect("add");

        let edges = graph.traceBackward(&tgt);
        assert_eq!(edges.len(), 2);
    }

    #[test]
    fn test_impact_analysis_single_hop() {
        let graph = LineageGraph::new();
        let src = tableNode("users", "email");
        let tgt = viewNode("emails", "address");

        graph
            .addEdge(src.clone(), tgt.clone(), TransformType::Direct)
            .expect("add");

        let impacted = graph.impactAnalysis(&LineageNodeKind::Table, "users", "email");
        assert_eq!(impacted.len(), 1);
        assert_eq!(impacted[0], tgt);
    }

    #[test]
    fn test_impact_analysis_transitive() {
        let graph = LineageGraph::new();
        let src = tableNode("users", "email");
        let mid = viewNode("user_view", "email");
        let dst = mvNode("email_report", "email");

        graph
            .addEdge(src.clone(), mid.clone(), TransformType::Direct)
            .expect("add");
        graph
            .addEdge(mid.clone(), dst.clone(), TransformType::Direct)
            .expect("add");

        let impacted = graph.impactAnalysis(&LineageNodeKind::Table, "users", "email");
        assert_eq!(impacted.len(), 2);
        assert!(impacted.contains(&mid));
        assert!(impacted.contains(&dst));
    }

    #[test]
    fn test_impact_analysis_no_downstream() {
        let graph = LineageGraph::new();
        let impacted = graph.impactAnalysis(&LineageNodeKind::Table, "orphan", "col");
        assert!(impacted.is_empty());
    }

    #[test]
    fn test_impact_analysis_branching() {
        let graph = LineageGraph::new();
        let src = tableNode("orders", "amount");
        let branch1 = viewNode("daily_totals", "total");
        let branch2 = mvNode("monthly_totals", "total");
        let leaf = pipelineNode("reporting", "amount");

        graph
            .addEdge(src.clone(), branch1.clone(), TransformType::Aggregated)
            .expect("add");
        graph
            .addEdge(src.clone(), branch2.clone(), TransformType::Aggregated)
            .expect("add");
        graph
            .addEdge(branch1.clone(), leaf.clone(), TransformType::Direct)
            .expect("add");

        let impacted = graph.impactAnalysis(&LineageNodeKind::Table, "orders", "amount");
        assert_eq!(impacted.len(), 3);
        assert!(impacted.contains(&branch1));
        assert!(impacted.contains(&branch2));
        assert!(impacted.contains(&leaf));
    }

    #[test]
    fn test_remove_edges_for() {
        let graph = LineageGraph::new();
        let src = tableNode("users", "id");
        let tgt = viewNode("v1", "user_id");
        let other = tableNode("orders", "id");
        let otherTgt = viewNode("v1", "order_id");

        graph
            .addEdge(src.clone(), tgt.clone(), TransformType::Direct)
            .expect("add");
        graph
            .addEdge(other.clone(), otherTgt.clone(), TransformType::Joined)
            .expect("add");

        graph
            .removeEdgesFor(&LineageNodeKind::View, "v1")
            .expect("remove");

        let forwardFromSrc = graph.traceForward(&src);
        assert!(forwardFromSrc.is_empty());

        let forwardFromOther = graph.traceForward(&other);
        assert!(forwardFromOther.is_empty());
    }

    #[test]
    fn test_edge_count() {
        let graph = LineageGraph::new();
        assert_eq!(graph.edgeCount(), 0);

        graph
            .addEdge(
                tableNode("t1", "c1"),
                viewNode("v1", "c1"),
                TransformType::Direct,
            )
            .expect("add");
        graph
            .addEdge(
                tableNode("t1", "c2"),
                viewNode("v1", "c2"),
                TransformType::Direct,
            )
            .expect("add");

        assert_eq!(graph.edgeCount(), 2);
    }

    #[test]
    fn test_lineage_node_key_format() {
        let node = tableNode("users", "email");
        assert_eq!(node.toKey(), "Table:users:email");

        let mv = LineageNode::new(LineageNodeKind::MaterializedView, "mv1", "col");
        assert_eq!(mv.toKey(), "MaterializedView:mv1:col");
    }

    #[test]
    fn test_transform_type_display() {
        assert_eq!(TransformType::Direct.to_string(), "Direct");
        assert_eq!(TransformType::Aggregated.to_string(), "Aggregated");
        assert_eq!(TransformType::Filtered.to_string(), "Filtered");
        assert_eq!(TransformType::Computed.to_string(), "Computed");
        assert_eq!(TransformType::Joined.to_string(), "Joined");
    }

    #[test]
    fn test_lineage_node_equality() {
        let a = tableNode("users", "email");
        let b = tableNode("users", "email");
        let c = tableNode("users", "name");
        assert_eq!(a, b);
        assert_ne!(a, c);
    }

    #[test]
    fn test_impact_analysis_avoids_cycles() {
        let graph = LineageGraph::new();
        let a = viewNode("v1", "col");
        let b = viewNode("v2", "col");

        graph
            .addEdge(a.clone(), b.clone(), TransformType::Direct)
            .expect("add");
        graph
            .addEdge(b.clone(), a.clone(), TransformType::Direct)
            .expect("add");

        let impacted = graph.impactAnalysis(&LineageNodeKind::View, "v1", "col");
        assert_eq!(impacted.len(), 1);
        assert_eq!(impacted[0], b);
    }
}
