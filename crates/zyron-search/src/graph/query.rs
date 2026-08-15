//! Graph pattern query types and compilation to relational join plans.
//!
//! Cypher-like pattern matching is represented as a sequence of alternating
//! node and edge elements. Patterns are compiled into join conditions
//! that the existing relational query engine can execute on the backing
//! node and edge tables.

use zyron_common::ZyronError;

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
    /// Optional WHERE clause over pattern variables, as written. Compiling
    /// the pattern resolves it into `CompiledGraphQuery::filters`, so a
    /// clause it cannot resolve fails the compile rather than travelling on
    /// as text nothing applies
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

/// How a filter compares a pattern property to a value.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FilterOp {
    Eq,
    Neq,
    Lt,
    Lte,
    Gt,
    Gte,
}

impl FilterOp {
    /// The operator's SQL spelling, longest first so `<=` is not read as `<`.
    const SPELLINGS: [(&'static str, FilterOp); 7] = [
        ("<=", FilterOp::Lte),
        (">=", FilterOp::Gte),
        ("<>", FilterOp::Neq),
        ("!=", FilterOp::Neq),
        ("=", FilterOp::Eq),
        ("<", FilterOp::Lt),
        (">", FilterOp::Gt),
    ];

    pub fn as_sql(&self) -> &'static str {
        match self {
            FilterOp::Eq => "=",
            FilterOp::Neq => "<>",
            FilterOp::Lt => "<",
            FilterOp::Lte => "<=",
            FilterOp::Gt => ">",
            FilterOp::Gte => ">=",
        }
    }
}

/// A literal a filter compares against.
#[derive(Debug, Clone, PartialEq)]
pub enum FilterValue {
    Integer(i64),
    Float(f64),
    Text(String),
    Boolean(bool),
    Null,
}

/// One `variable.property <op> value` term of a pattern's WHERE clause.
///
/// The term is resolved rather than carried as text: a consumer applies it
/// against the scan the variable named, the way it applies a join condition,
/// with no second parse and no chance of a predicate being carried along and
/// silently not applied.
#[derive(Debug, Clone, PartialEq)]
pub struct GraphFilter {
    /// Pattern variable, which is the alias of the scan this term filters
    pub variable: String,
    /// Property on that variable, which is the column name
    pub property: String,
    pub op: FilterOp,
    pub value: FilterValue,
}

/// Resolves a pattern's WHERE clause against the scans the pattern produced.
///
/// A term naming a variable the pattern does not bind is refused: it would
/// otherwise filter nothing and read as a narrower match than it was.
fn resolve_filters(
    pattern: &GraphPattern,
    scans: &[TableScanRef],
) -> zyron_common::Result<Vec<GraphFilter>> {
    let Some(text) = &pattern.where_clause else {
        return Ok(Vec::new());
    };
    let filters = parse_where_clause(text)?;
    for filter in &filters {
        if !scans.iter().any(|s| s.alias == filter.variable) {
            let bound: Vec<&str> = scans.iter().map(|s| s.alias.as_str()).collect();
            return Err(ZyronError::PlanError(format!(
                "graph WHERE names variable \"{}\", which the pattern does not bind. Bound: {}",
                filter.variable,
                bound.join(", ")
            )));
        }
    }
    Ok(filters)
}

/// Parses a pattern's WHERE text into terms over pattern variables.
///
/// Accepts `variable.property <op> literal` terms joined by AND, which is
/// what a pattern filter is: the joins are already expressed as join
/// conditions, so a term here always names one variable's property. Anything
/// else is refused by name, because a filter that parsed to nothing would
/// return the unfiltered pattern and look like it had matched more.
pub fn parse_where_clause(text: &str) -> zyron_common::Result<Vec<GraphFilter>> {
    let mut out = Vec::new();
    for term in split_conjuncts(text) {
        let term = term.trim();
        if term.is_empty() {
            continue;
        }
        out.push(parse_filter_term(term)?);
    }
    if out.is_empty() {
        return Err(ZyronError::PlanError(format!(
            "graph WHERE clause \"{}\" has no conditions",
            text.trim()
        )));
    }
    Ok(out)
}

/// Splits on AND at the top level, leaving AND inside a quoted literal alone.
fn split_conjuncts(text: &str) -> Vec<&str> {
    let bytes = text.as_bytes();
    let mut parts = Vec::new();
    let mut start = 0usize;
    let mut i = 0usize;
    let mut in_quote = false;
    while i < bytes.len() {
        let c = bytes[i];
        if c == b'\'' {
            in_quote = !in_quote;
            i += 1;
            continue;
        }
        if !in_quote
            && (c == b'a' || c == b'A')
            && text[i..].len() >= 3
            && text[i..i + 3].eq_ignore_ascii_case("and")
            && (i == 0 || bytes[i - 1].is_ascii_whitespace())
            && (i + 3 == bytes.len() || bytes[i + 3].is_ascii_whitespace())
        {
            parts.push(&text[start..i]);
            i += 3;
            start = i;
            continue;
        }
        i += 1;
    }
    parts.push(&text[start..]);
    parts
}

fn parse_filter_term(term: &str) -> zyron_common::Result<GraphFilter> {
    let refuse = |reason: &str| {
        ZyronError::PlanError(format!(
            "graph WHERE term \"{}\" {}. A term is variable.property compared to a literal, terms joined by AND",
            term, reason
        ))
    };
    // Longest spelling first, and the operator is found outside quotes so a
    // value containing one is not mistaken for the comparison
    let mut split: Option<(usize, usize, FilterOp)> = None;
    let bytes = term.as_bytes();
    let mut i = 0usize;
    let mut in_quote = false;
    'scan: while i < bytes.len() {
        if bytes[i] == b'\'' {
            in_quote = !in_quote;
            i += 1;
            continue;
        }
        if !in_quote {
            for (spelling, op) in FilterOp::SPELLINGS {
                if term[i..].starts_with(spelling) {
                    split = Some((i, i + spelling.len(), op));
                    break 'scan;
                }
            }
        }
        i += 1;
    }
    let Some((lhs_end, rhs_start, op)) = split else {
        return Err(refuse("has no comparison operator"));
    };

    let lhs = term[..lhs_end].trim();
    let Some((variable, property)) = lhs.split_once('.') else {
        return Err(refuse("does not name variable.property on its left"));
    };
    let variable = variable.trim();
    let property = property.trim();
    if variable.is_empty() || property.is_empty() || property.contains('.') {
        return Err(refuse("does not name variable.property on its left"));
    }

    let rhs = term[rhs_start..].trim();
    let value = parse_filter_value(rhs).ok_or_else(|| refuse("compares against no literal"))?;
    Ok(GraphFilter {
        variable: variable.to_string(),
        property: property.to_string(),
        op,
        value,
    })
}

fn parse_filter_value(text: &str) -> Option<FilterValue> {
    if text.len() >= 2 && text.starts_with('\'') && text.ends_with('\'') {
        // Two quotes are one literal quote, the SQL escape
        return Some(FilterValue::Text(
            text[1..text.len() - 1].replace("''", "'"),
        ));
    }
    if text.eq_ignore_ascii_case("null") {
        return Some(FilterValue::Null);
    }
    if text.eq_ignore_ascii_case("true") {
        return Some(FilterValue::Boolean(true));
    }
    if text.eq_ignore_ascii_case("false") {
        return Some(FilterValue::Boolean(false));
    }
    if let Ok(v) = text.parse::<i64>() {
        return Some(FilterValue::Integer(v));
    }
    if let Ok(v) = text.parse::<f64>() {
        if v.is_finite() {
            return Some(FilterValue::Float(v));
        }
    }
    None
}

/// A compiled graph query expressed as relational operations on backing tables.
/// The existing relational query engine executes this.
#[derive(Debug, Clone)]
pub struct CompiledGraphQuery {
    /// Table scans for each node and edge in the pattern
    pub table_scans: Vec<TableScanRef>,
    /// Join conditions linking edges to nodes
    pub join_conditions: Vec<JoinCondition>,
    /// WHERE terms, resolved to the scan alias each one filters. Empty when
    /// the pattern carries no filter
    pub filters: Vec<GraphFilter>,
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

    let filters = resolve_filters(pattern, &scans)?;
    Ok(CompiledGraphQuery {
        table_scans: scans,
        join_conditions: joins,
        filters,
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

    let filters = resolve_filters(pattern, &scans)?;
    Ok(CompiledGraphQuery {
        table_scans: scans,
        join_conditions: all_joins,
        filters,
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

    /// A pattern's WHERE clause is resolved to terms bound to the scan each
    /// one filters. Carrying it as text would let a consumer drop it and
    /// return the unfiltered pattern, which reads as a wider match rather
    /// than as a failure.
    #[test]
    fn test_a_where_clause_resolves_to_terms_bound_to_the_scans() {
        let schema = test_schema();
        let mut pattern = GraphPattern::new(vec![
            PatternElement::node(Some("a".to_string()), Some("Person".to_string())),
            PatternElement::edge(None, Some("KNOWS".to_string()), EdgeDirection::Outgoing),
            PatternElement::node(Some("b".to_string()), Some("Person".to_string())),
        ]);
        pattern.where_clause =
            Some("a.name = 'Ada' AND b.age >= 21 AND a.active <> false".to_string());

        let queries = compile_pattern(&pattern, &schema).expect("compile");
        assert_eq!(
            queries[0].filters,
            vec![
                GraphFilter {
                    variable: "a".to_string(),
                    property: "name".to_string(),
                    op: FilterOp::Eq,
                    value: FilterValue::Text("Ada".to_string()),
                },
                GraphFilter {
                    variable: "b".to_string(),
                    property: "age".to_string(),
                    op: FilterOp::Gte,
                    value: FilterValue::Integer(21),
                },
                GraphFilter {
                    variable: "a".to_string(),
                    property: "active".to_string(),
                    op: FilterOp::Neq,
                    value: FilterValue::Boolean(false),
                },
            ]
        );

        // No clause is no terms, not an error
        let mut plain = pattern.clone();
        plain.where_clause = None;
        assert!(
            compile_pattern(&plain, &schema).expect("compile")[0]
                .filters
                .is_empty()
        );
    }

    /// A clause the compiler cannot resolve fails the compile. Accepting it
    /// and filtering nothing would answer a different question than asked.
    #[test]
    fn test_a_where_clause_that_cannot_be_resolved_fails_the_compile() {
        let schema = test_schema();
        let mut pattern = GraphPattern::new(vec![
            PatternElement::node(Some("a".to_string()), Some("Person".to_string())),
            PatternElement::edge(None, Some("KNOWS".to_string()), EdgeDirection::Outgoing),
            PatternElement::node(Some("b".to_string()), Some("Person".to_string())),
        ]);
        for (clause, expected) in [
            // Names a variable the pattern does not bind
            ("z.name = 'Ada'", "does not bind"),
            // Not a comparison at all
            ("a.name", "no comparison operator"),
            // Compares two properties, which is a join the pattern expresses
            ("a.name = b.name", "compares against no literal"),
            // No property on the left
            ("name = 'Ada'", "variable.property"),
            ("", "no conditions"),
        ] {
            pattern.where_clause = Some(clause.to_string());
            let err = compile_pattern(&pattern, &schema)
                .expect_err(&format!("clause {clause:?} must be refused"));
            assert!(
                err.to_string().contains(expected),
                "clause {clause:?} reported {err}"
            );
        }
    }

    /// The AND split and the operator scan both run outside quotes, so a
    /// literal containing either is one value rather than two terms.
    #[test]
    fn test_a_literal_containing_and_or_an_operator_stays_one_value() {
        let filters = parse_where_clause("a.label = 'sales and marketing'").expect("parse");
        assert_eq!(filters.len(), 1);
        assert_eq!(
            filters[0].value,
            FilterValue::Text("sales and marketing".to_string())
        );

        let filters = parse_where_clause("a.expr = 'x >= y'").expect("parse");
        assert_eq!(filters[0].op, FilterOp::Eq);
        assert_eq!(filters[0].value, FilterValue::Text("x >= y".to_string()));

        // Two quotes are one quote
        let filters = parse_where_clause("a.name = 'O''Hara'").expect("parse");
        assert_eq!(filters[0].value, FilterValue::Text("O'Hara".to_string()));

        // A word merely containing "and" does not split the clause
        let filters = parse_where_clause("a.brand = 'x' AND a.hand = 2").expect("parse");
        assert_eq!(filters.len(), 2);
        assert_eq!(filters[1].property, "hand");
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
