//! Graph pattern query types and compilation to relational join plans.
//!
//! Cypher-like pattern matching is represented as a sequence of alternating
//! node and edge elements. Patterns are compiled into join conditions
//! that the existing relational query engine can execute on the backing
//! node and edge tables.

use super::schema::{GraphSchema, LabelId};

/// Direction of an edge in a graph pattern.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EdgeDirection {
    /// From left node to right node (->)
    Outgoing,
    /// From right node to left node (<-)
    Incoming,
    /// Either direction (--)
    Undirected,
}

/// A single element in a graph match pattern.
#[derive(Debug, Clone)]
pub enum PatternElement {
    /// A node pattern: (variable:Label)
    Node {
        /// Optional variable binding name
        variable: Option<String>,
        /// Optional label filter
        label: Option<String>,
    },
    /// An edge pattern: -[variable:Label]-> or -[*min..max]->
    Edge {
        /// Optional variable binding name
        variable: Option<String>,
        /// Optional edge label filter
        label: Option<String>,
        /// Direction of the edge
        direction: EdgeDirection,
        /// Minimum number of hops (for variable-length paths)
        min_hops: u32,
        /// Maximum number of hops (for variable-length paths)
        max_hops: u32,
    },
}

impl PatternElement {
    /// Creates a simple node pattern with a variable and optional label.
    pub fn node(variable: Option<String>, label: Option<String>) -> Self {
        Self::Node { variable, label }
    }

    /// Creates a single-hop edge pattern.
    pub fn edge(variable: Option<String>, label: Option<String>, direction: EdgeDirection) -> Self {
        Self::Edge {
            variable,
            label,
            direction,
            min_hops: 1,
            max_hops: 1,
        }
    }

    /// Creates a variable-length path pattern.
    pub fn variable_length_edge(
        variable: Option<String>,
        label: Option<String>,
        direction: EdgeDirection,
        min_hops: u32,
        max_hops: u32,
    ) -> Self {
        Self::Edge {
            variable,
            label,
            direction,
            min_hops,
            max_hops,
        }
    }

    /// Returns true if this element represents a variable-length path.
    pub fn is_variable_length(&self) -> bool {
        matches!(self, Self::Edge { min_hops, max_hops, .. } if *min_hops != *max_hops || *max_hops > 1)
    }
}

/// A column or expression to return from a graph query.
#[derive(Debug, Clone)]
pub struct GraphReturnItem {
    /// Variable name from the pattern (e.g., "a", "b", "r")
    pub variable: String,
    /// Optional property access (e.g., "name" in a.name)
    pub property: Option<String>,
    /// Optional alias for the output column
    pub alias: Option<String>,
}

/// A full graph match pattern with optional WHERE and RETURN clauses.
#[derive(Debug, Clone)]
pub struct GraphPattern {
    /// Alternating node/edge elements forming the pattern
    pub elements: Vec<PatternElement>,
    /// Optional WHERE clause filter (stored as raw string for now,
    /// bound during query planning)
    pub where_clause: Option<String>,
    /// Columns to return
    pub return_items: Vec<GraphReturnItem>,
    /// Whether this is an OPTIONAL MATCH (left join semantics)
    pub optional: bool,
}

impl GraphPattern {
    /// Creates a new graph pattern.
    pub fn new(elements: Vec<PatternElement>) -> Self {
        Self {
            elements,
            where_clause: None,
            return_items: Vec::new(),
            optional: false,
        }
    }

    /// Returns the number of hops in the pattern.
    /// Each edge element contributes its max_hops.
    pub fn total_max_hops(&self) -> u32 {
        self.elements
            .iter()
            .filter_map(|e| match e {
                PatternElement::Edge { max_hops, .. } => Some(*max_hops),
                _ => None,
            })
            .sum()
    }
}

/// A reference to a table scan in a compiled graph query.
#[derive(Debug, Clone)]
pub struct TableScanRef {
    /// Catalog table ID of the backing table
    pub table_id: u32,
    /// Alias for this scan (from the pattern variable)
    pub alias: String,
    /// Optional label filter (translated to a WHERE predicate on label_id)
    pub label_filter: Option<LabelId>,
}

/// A join condition linking two scans in a compiled graph query.
#[derive(Debug, Clone)]
pub struct JoinCondition {
    /// Left side: (table alias, column name)
    pub left: (String, String),
    /// Right side: (table alias, column name)
    pub right: (String, String),
}

/// A compiled graph query expressed as relational operations on backing tables.
/// The existing relational query engine executes this.
#[derive(Debug, Clone)]
pub struct CompiledGraphQuery {
    /// Table scans for each node and edge in the pattern
    pub table_scans: Vec<TableScanRef>,
    /// Join conditions linking edges to nodes
    pub join_conditions: Vec<JoinCondition>,
    /// Optional WHERE filter predicate (raw SQL string)
    pub filter_predicate: Option<String>,
    /// Projected return items
    pub return_items: Vec<GraphReturnItem>,
    /// Whether this uses LEFT JOIN (OPTIONAL MATCH)
    pub optional: bool,
}

/// Compiles a graph pattern into a relational query plan using the graph schema
/// to resolve label names to table IDs.
///
/// For a pattern like (a:Person)-[:KNOWS]->(b:Person):
/// - a -> scan person_nodes table with alias "a"
/// - [:KNOWS] -> scan knows_edges table with alias "e0"
/// - b -> scan person_nodes table with alias "b"
/// - Join: e0.from_node = a.node_id AND e0.to_node = b.node_id
///
/// Variable-length paths (a)-[:KNOWS*1..3]->(b) expand to multiple
/// join plans combined with UNION:
/// - Hop 1: a -> e0 -> b
/// - Hop 2: a -> e0 -> mid0 -> e1 -> b
/// - Hop 3: a -> e0 -> mid0 -> e1 -> mid1 -> e2 -> b
pub fn compile_pattern(
    pattern: &GraphPattern,
    schema: &GraphSchema,
) -> zyron_common::Result<Vec<CompiledGraphQuery>> {
    let mut results = Vec::new();

    // Separate nodes and edges from the pattern
    let mut nodes: Vec<&PatternElement> = Vec::new();
    let mut edges: Vec<&PatternElement> = Vec::new();

    for element in &pattern.elements {
        match element {
            PatternElement::Node { .. } => nodes.push(element),
            PatternElement::Edge { .. } => edges.push(element),
        }
    }

    // For simple single-hop patterns, compile directly
    if edges.len() == 1 && !edges[0].is_variable_length() {
        let query = compile_single_hop(pattern, schema)?;
        results.push(query);
        return Ok(results);
    }

    // For variable-length paths, expand to multiple hop counts
    if edges.len() == 1 {
        if let PatternElement::Edge {
            min_hops, max_hops, ..
        } = edges[0]
        {
            for hop_count in *min_hops..=*max_hops {
                let query = compile_fixed_hops(pattern, schema, hop_count)?;
                results.push(query);
            }
            return Ok(results);
        }
    }

    // Multi-edge patterns: compile each edge segment
    let query = compile_multi_edge(pattern, schema)?;
    results.push(query);

    Ok(results)
}

/// Compiles a single-hop pattern into one relational query.
fn compile_single_hop(
    pattern: &GraphPattern,
    schema: &GraphSchema,
) -> zyron_common::Result<CompiledGraphQuery> {
    compile_fixed_hops(pattern, schema, 1)
}

/// Compiles a pattern with a fixed number of hops into one relational query.
fn compile_fixed_hops(
    pattern: &GraphPattern,
    schema: &GraphSchema,
    hop_count: u32,
) -> zyron_common::Result<CompiledGraphQuery> {
    let mut scans = Vec::new();
    let mut joins = Vec::new();

    // Extract the edge element (first edge in the pattern)
    let edge_elem = pattern
        .elements
        .iter()
        .find(|e| matches!(e, PatternElement::Edge { .. }));

    let (edge_label, direction) = match edge_elem {
        Some(PatternElement::Edge {
            label, direction, ..
        }) => (label.as_deref(), *direction),
        _ => (None, EdgeDirection::Outgoing),
    };

    // Resolve edge label to table ID
    let edge_table_id = if let Some(label_name) = edge_label {
        let edge = schema.get_edge_label(label_name).ok_or_else(|| {
            zyron_common::ZyronError::GraphQueryError(format!(
                "edge label '{}' not found in graph schema '{}'",
                label_name, schema.name
            ))
        })?;
        edge.edge_table_id
    } else if let Some(first_edge) = schema.edge_labels.first() {
        first_edge.edge_table_id
    } else {
        return Err(zyron_common::ZyronError::GraphQueryError(
            "graph schema has no edge labels".to_string(),
        ));
    };

    // Get the start and end node patterns
    let start_node = pattern
        .elements
        .iter()
        .find(|e| matches!(e, PatternElement::Node { .. }));
    let end_node = pattern
        .elements
        .iter()
        .rev()
        .find(|e| matches!(e, PatternElement::Node { .. }));

    let start_alias = match start_node {
        Some(PatternElement::Node { variable, .. }) => {
            variable.clone().unwrap_or_else(|| "n_start".to_string())
        }
        _ => "n_start".to_string(),
    };

    let end_alias = match end_node {
        Some(PatternElement::Node { variable, .. }) => {
            variable.clone().unwrap_or_else(|| "n_end".to_string())
        }
        _ => "n_end".to_string(),
    };

    // Resolve start node label to table ID
    let start_label_filter = resolve_node_label(start_node, schema);
    let end_label_filter = resolve_node_label(end_node, schema);

    let start_table_id = resolve_node_table_id(start_node, schema);
    let end_table_id = resolve_node_table_id(end_node, schema);

    // Add start node scan
    scans.push(TableScanRef {
        table_id: start_table_id,
        alias: start_alias.clone(),
        label_filter: start_label_filter,
    });

    // Add edge scans and intermediate node scans for multi-hop
    let mut prev_alias = start_alias;

    for hop in 0..hop_count {
        let edge_alias = format!("e{}", hop);

        scans.push(TableScanRef {
            table_id: edge_table_id,
            alias: edge_alias.clone(),
            label_filter: None,
        });

        // Join: edge.from_node = previous_node.node_id
        let (from_col, to_col) = match direction {
            EdgeDirection::Outgoing => ("from_node", "to_node"),
            EdgeDirection::Incoming => ("to_node", "from_node"),
            EdgeDirection::Undirected => ("from_node", "to_node"),
        };

        joins.push(JoinCondition {
            left: (prev_alias.clone(), "node_id".to_string()),
            right: (edge_alias.clone(), from_col.to_string()),
        });

        if hop < hop_count - 1 {
            // Add intermediate node
            let mid_alias = format!("mid{}", hop);
            scans.push(TableScanRef {
                table_id: start_table_id,
                alias: mid_alias.clone(),
                label_filter: None,
            });
            joins.push(JoinCondition {
                left: (edge_alias, to_col.to_string()),
                right: (mid_alias.clone(), "node_id".to_string()),
            });
            prev_alias = mid_alias;
        } else {
            // Join to end node
            joins.push(JoinCondition {
                left: (edge_alias, to_col.to_string()),
                right: (end_alias.clone(), "node_id".to_string()),
            });
        }
    }

    // Add end node scan
    scans.push(TableScanRef {
        table_id: end_table_id,
        alias: end_alias,
        label_filter: end_label_filter,
    });

    Ok(CompiledGraphQuery {
        table_scans: scans,
        join_conditions: joins,
        filter_predicate: pattern.where_clause.clone(),
        return_items: pattern.return_items.clone(),
        optional: pattern.optional,
    })
}

/// Compiles a multi-edge pattern (e.g., (a)-[:R1]->(b)-[:R2]->(c)).
fn compile_multi_edge(
    pattern: &GraphPattern,
    schema: &GraphSchema,
) -> zyron_common::Result<CompiledGraphQuery> {
    let mut scans = Vec::new();
    let mut joins = Vec::new();
    let mut edge_idx = 0u32;
    let mut prev_node_alias: Option<String> = None;

    for element in &pattern.elements {
        match element {
            PatternElement::Node { variable, label } => {
                let alias = variable
                    .clone()
                    .unwrap_or_else(|| format!("n{}", scans.len()));
                let table_id = if let Some(label_name) = label {
                    schema
                        .get_node_label(label_name)
                        .map(|nl| nl.node_table_id)
                        .unwrap_or(0)
                } else if let Some(first) = schema.node_labels.first() {
                    first.node_table_id
                } else {
                    0
                };
                let label_filter = label
                    .as_ref()
                    .and_then(|name| schema.get_node_label(name).map(|nl| nl.label_id));

                scans.push(TableScanRef {
                    table_id,
                    alias: alias.clone(),
                    label_filter,
                });
                prev_node_alias = Some(alias);
            }
            PatternElement::Edge {
                label, direction, ..
            } => {
                let edge_alias = format!("e{}", edge_idx);
                edge_idx += 1;

                let edge_table_id = if let Some(label_name) = label {
                    schema
                        .get_edge_label(label_name)
                        .map(|el| el.edge_table_id)
                        .unwrap_or(0)
                } else if let Some(first) = schema.edge_labels.first() {
                    first.edge_table_id
                } else {
                    0
                };

                scans.push(TableScanRef {
                    table_id: edge_table_id,
                    alias: edge_alias.clone(),
                    label_filter: None,
                });

                let (from_col, to_col) = match direction {
                    EdgeDirection::Outgoing => ("from_node", "to_node"),
                    EdgeDirection::Incoming => ("to_node", "from_node"),
                    EdgeDirection::Undirected => ("from_node", "to_node"),
                };

                // Join edge.from_node = previous_node.node_id
                if let Some(ref prev) = prev_node_alias {
                    joins.push(JoinCondition {
                        left: (prev.clone(), "node_id".to_string()),
                        right: (edge_alias.clone(), from_col.to_string()),
                    });
                }

                // The next node element will join to this edge's to_node
                // Store the edge alias and to_col for the next node
                prev_node_alias = Some(format!("{}:{}", edge_alias, to_col));
            }
        }
    }

    // Fix up joins where prev_node_alias contains an edge reference
    // (from the edge -> next_node connection)
    let mut final_joins = Vec::new();
    let mut pending_edge_join: Option<(String, String)> = None;

    for scan in &scans {
        if let Some((edge_ref, to_col)) = pending_edge_join.take() {
            if !scan.alias.contains(':') {
                final_joins.push(JoinCondition {
                    left: (edge_ref, to_col),
                    right: (scan.alias.clone(), "node_id".to_string()),
                });
            }
        }

        // Check if this alias contains a pending edge reference
        if scan.alias.contains(':') {
            let parts: Vec<&str> = scan.alias.splitn(2, ':').collect();
            pending_edge_join = Some((parts[0].to_string(), parts[1].to_string()));
        }
    }

    // Rebuild joins with the corrected ones
    let all_joins = joins.into_iter().chain(final_joins.into_iter()).collect();

    Ok(CompiledGraphQuery {
        table_scans: scans,
        join_conditions: all_joins,
        filter_predicate: pattern.where_clause.clone(),
        return_items: pattern.return_items.clone(),
        optional: pattern.optional,
    })
}

/// Resolves a node pattern's label to a LabelId if specified.
fn resolve_node_label(node: Option<&PatternElement>, schema: &GraphSchema) -> Option<LabelId> {
    match node {
        Some(PatternElement::Node {
            label: Some(name), ..
        }) => schema.get_node_label(name).map(|nl| nl.label_id),
        _ => None,
    }
}

/// Resolves a node pattern's label to its backing table ID.
fn resolve_node_table_id(node: Option<&PatternElement>, schema: &GraphSchema) -> u32 {
    match node {
        Some(PatternElement::Node {
            label: Some(name), ..
        }) => schema
            .get_node_label(name)
            .map(|nl| nl.node_table_id)
            .unwrap_or(0),
        _ => schema
            .node_labels
            .first()
            .map(|nl| nl.node_table_id)
            .unwrap_or(0),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::schema::{GraphSchema, PropertyDef};

    fn test_schema() -> GraphSchema {
        let mut schema = GraphSchema::new("social".to_string(), 1);
        let person_id = schema.add_node_label(
            "Person".to_string(),
            vec![PropertyDef {
                name: "name".to_string(),
                type_id: zyron_common::TypeId::Varchar,
                nullable: false,
            }],
            200,
        );
        let company_id = schema.add_node_label("Company".to_string(), vec![], 202);
        let _ = schema.add_edge_label("KNOWS".to_string(), person_id, person_id, vec![], 204, true);
        let _ = schema.add_edge_label(
            "WORKS_AT".to_string(),
            person_id,
            company_id,
            vec![],
            206,
            true,
        );
        schema
    }

    #[test]
    fn test_single_hop_compilation() {
        let schema = test_schema();
        let pattern = GraphPattern::new(vec![
            PatternElement::node(Some("a".to_string()), Some("Person".to_string())),
            PatternElement::edge(None, Some("KNOWS".to_string()), EdgeDirection::Outgoing),
            PatternElement::node(Some("b".to_string()), Some("Person".to_string())),
        ]);

        let queries = compile_pattern(&pattern, &schema).expect("compile");
        assert_eq!(queries.len(), 1);

        let q = &queries[0];
        assert_eq!(q.table_scans.len(), 3); // a, edge, b
        assert_eq!(q.join_conditions.len(), 2);
    }

    #[test]
    fn test_variable_length_expansion() {
        let schema = test_schema();
        let pattern = GraphPattern::new(vec![
            PatternElement::node(Some("a".to_string()), Some("Person".to_string())),
            PatternElement::variable_length_edge(
                None,
                Some("KNOWS".to_string()),
                EdgeDirection::Outgoing,
                1,
                3,
            ),
            PatternElement::node(Some("b".to_string()), Some("Person".to_string())),
        ]);

        let queries = compile_pattern(&pattern, &schema).expect("compile");
        // Should produce 3 queries: 1-hop, 2-hop, 3-hop
        assert_eq!(queries.len(), 3);

        // 1-hop: start + edge + end = 3 scans
        assert_eq!(queries[0].table_scans.len(), 3);
        // 2-hop: start + edge + mid + edge + end = 5 scans
        assert_eq!(queries[1].table_scans.len(), 5);
        // 3-hop: start + edge + mid + edge + mid + edge + end = 7 scans
        assert_eq!(queries[2].table_scans.len(), 7);
    }

    #[test]
    fn test_pattern_element_helpers() {
        let node = PatternElement::node(Some("x".to_string()), None);
        assert!(!node.is_variable_length());

        let edge = PatternElement::edge(None, None, EdgeDirection::Outgoing);
        assert!(!edge.is_variable_length());

        let var_edge =
            PatternElement::variable_length_edge(None, None, EdgeDirection::Outgoing, 1, 3);
        assert!(var_edge.is_variable_length());
    }
}
