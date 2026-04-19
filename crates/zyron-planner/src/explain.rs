//! EXPLAIN query plan output.
//!
//! Builds a tree representation of a physical plan with estimated costs,
//! and optionally merges actual execution metrics from EXPLAIN ANALYZE.
//! Supports text, JSON, and YAML output formats.

use crate::cost::PlanCost;
use crate::physical::PhysicalPlan;
use std::fmt::Write;

// ---------------------------------------------------------------------------
// Options
// ---------------------------------------------------------------------------

/// Controls what information EXPLAIN includes in its output.
#[derive(Debug, Clone)]
pub struct ExplainOptions {
    /// Execute the query and collect runtime metrics.
    pub analyze: bool,
    /// Show cost estimates (default: true).
    pub costs: bool,
    /// Show buffer hit/miss counts (requires ANALYZE).
    pub buffers: bool,
    /// Show per-operator timing (requires ANALYZE).
    pub timing: bool,
    /// Output format.
    pub format: ExplainFormat,
}

impl Default for ExplainOptions {
    fn default() -> Self {
        Self {
            analyze: false,
            costs: true,
            buffers: false,
            timing: true,
            format: ExplainFormat::Text,
        }
    }
}

/// Output format for EXPLAIN results.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExplainFormat {
    Text,
    Json,
    Yaml,
}

impl ExplainFormat {
    /// Parses a format name string (case-insensitive).
    pub fn from_str(s: &str) -> Self {
        match s.to_lowercase().as_str() {
            "json" => ExplainFormat::Json,
            "yaml" => ExplainFormat::Yaml,
            _ => ExplainFormat::Text,
        }
    }
}

// ---------------------------------------------------------------------------
// Actual metrics
// ---------------------------------------------------------------------------

/// Runtime metrics collected during EXPLAIN ANALYZE execution.
#[derive(Debug, Clone)]
pub struct ActualMetrics {
    pub rows: u64,
    pub elapsed_ms: f64,
    pub batches: u64,
}

// ---------------------------------------------------------------------------
// Explain node
// ---------------------------------------------------------------------------

/// Tree node representing one operator in the EXPLAIN output.
#[derive(Debug, Clone)]
pub struct ExplainNode {
    /// Operator name (e.g., "SeqScan", "HashJoin").
    pub operator_name: String,
    /// Key-value detail pairs (e.g., ("table", "orders"), ("predicate", "id > 5")).
    pub details: Vec<(String, String)>,
    /// Estimated cost from the planner.
    pub estimated_cost: Option<PlanCost>,
    /// Actual runtime metrics (populated by EXPLAIN ANALYZE).
    pub actual_metrics: Option<ActualMetrics>,
    /// Child operator nodes.
    pub children: Vec<ExplainNode>,
}

impl ExplainNode {
    /// Builds an ExplainNode tree from a PhysicalPlan.
    pub fn from_physical_plan(plan: &PhysicalPlan) -> Self {
        match plan {
            PhysicalPlan::SeqScan {
                table_id,
                columns,
                predicate,
                cost,
                ..
            } => {
                let mut details = vec![
                    ("table_id".to_string(), format!("{}", table_id.0)),
                    ("columns".to_string(), format!("{}", columns.len())),
                ];
                if predicate.is_some() {
                    details.push(("filter".to_string(), "yes".to_string()));
                }
                Self {
                    operator_name: "SeqScan".to_string(),
                    details,
                    estimated_cost: Some(*cost),
                    actual_metrics: None,
                    children: Vec::new(),
                }
            }
            PhysicalPlan::IndexScan {
                table_id,
                index_id,
                cost,
                ..
            } => Self {
                operator_name: "IndexScan".to_string(),
                details: vec![
                    ("table_id".to_string(), format!("{}", table_id.0)),
                    ("index_id".to_string(), format!("{}", index_id.0)),
                ],
                estimated_cost: Some(*cost),
                actual_metrics: None,
                children: Vec::new(),
            },
            PhysicalPlan::Filter {
                predicate: _,
                child,
                cost,
            } => Self {
                operator_name: "Filter".to_string(),
                details: Vec::new(),
                estimated_cost: Some(*cost),
                actual_metrics: None,
                children: vec![Self::from_physical_plan(child)],
            },
            PhysicalPlan::Project {
                expressions,
                child,
                cost,
                ..
            } => Self {
                operator_name: "Project".to_string(),
                details: vec![("columns".to_string(), format!("{}", expressions.len()))],
                estimated_cost: Some(*cost),
                actual_metrics: None,
                children: vec![Self::from_physical_plan(child)],
            },
            PhysicalPlan::NestedLoopJoin {
                left,
                right,
                join_type,
                cost,
                ..
            } => Self {
                operator_name: "NestedLoopJoin".to_string(),
                details: vec![("join_type".to_string(), format!("{:?}", join_type))],
                estimated_cost: Some(*cost),
                actual_metrics: None,
                children: vec![
                    Self::from_physical_plan(left),
                    Self::from_physical_plan(right),
                ],
            },
            PhysicalPlan::HashJoin {
                left,
                right,
                join_type,
                cost,
                ..
            } => Self {
                operator_name: "HashJoin".to_string(),
                details: vec![("join_type".to_string(), format!("{:?}", join_type))],
                estimated_cost: Some(*cost),
                actual_metrics: None,
                children: vec![
                    Self::from_physical_plan(left),
                    Self::from_physical_plan(right),
                ],
            },
            PhysicalPlan::MergeJoin {
                left,
                right,
                join_type,
                cost,
                ..
            } => Self {
                operator_name: "MergeJoin".to_string(),
                details: vec![("join_type".to_string(), format!("{:?}", join_type))],
                estimated_cost: Some(*cost),
                actual_metrics: None,
                children: vec![
                    Self::from_physical_plan(left),
                    Self::from_physical_plan(right),
                ],
            },
            PhysicalPlan::HashAggregate {
                group_by,
                aggregates,
                child,
                cost,
            } => Self {
                operator_name: "HashAggregate".to_string(),
                details: vec![
                    ("groups".to_string(), format!("{}", group_by.len())),
                    ("aggregates".to_string(), format!("{}", aggregates.len())),
                ],
                estimated_cost: Some(*cost),
                actual_metrics: None,
                children: vec![Self::from_physical_plan(child)],
            },
            PhysicalPlan::SortAggregate {
                group_by,
                aggregates,
                child,
                cost,
            } => Self {
                operator_name: "SortAggregate".to_string(),
                details: vec![
                    ("groups".to_string(), format!("{}", group_by.len())),
                    ("aggregates".to_string(), format!("{}", aggregates.len())),
                ],
                estimated_cost: Some(*cost),
                actual_metrics: None,
                children: vec![Self::from_physical_plan(child)],
            },
            PhysicalPlan::Sort {
                child, limit, cost, ..
            } => {
                let mut details = Vec::new();
                if let Some(l) = limit {
                    details.push(("top_n".to_string(), format!("{}", l)));
                }
                Self {
                    operator_name: "Sort".to_string(),
                    details,
                    estimated_cost: Some(*cost),
                    actual_metrics: None,
                    children: vec![Self::from_physical_plan(child)],
                }
            }
            PhysicalPlan::Limit {
                limit,
                offset,
                child,
                cost,
            } => {
                let mut details = Vec::new();
                if let Some(l) = limit {
                    details.push(("limit".to_string(), format!("{}", l)));
                }
                if let Some(o) = offset {
                    details.push(("offset".to_string(), format!("{}", o)));
                }
                Self {
                    operator_name: "Limit".to_string(),
                    details,
                    estimated_cost: Some(*cost),
                    actual_metrics: None,
                    children: vec![Self::from_physical_plan(child)],
                }
            }
            PhysicalPlan::HashDistinct { child, cost } => Self {
                operator_name: "HashDistinct".to_string(),
                details: Vec::new(),
                estimated_cost: Some(*cost),
                actual_metrics: None,
                children: vec![Self::from_physical_plan(child)],
            },
            PhysicalPlan::SetOp {
                op,
                all,
                left,
                right,
                cost,
            } => Self {
                operator_name: format!("{:?}", op),
                details: vec![("all".to_string(), format!("{}", all))],
                estimated_cost: Some(*cost),
                actual_metrics: None,
                children: vec![
                    Self::from_physical_plan(left),
                    Self::from_physical_plan(right),
                ],
            },
            PhysicalPlan::Insert {
                table_id,
                source,
                cost,
                ..
            } => Self {
                operator_name: "Insert".to_string(),
                details: vec![("table_id".to_string(), format!("{}", table_id.0))],
                estimated_cost: Some(*cost),
                actual_metrics: None,
                children: vec![Self::from_physical_plan(source)],
            },
            PhysicalPlan::Values { rows, cost, .. } => Self {
                operator_name: "Values".to_string(),
                details: vec![("rows".to_string(), format!("{}", rows.len()))],
                estimated_cost: Some(*cost),
                actual_metrics: None,
                children: Vec::new(),
            },
            PhysicalPlan::Update {
                table_id,
                child,
                cost,
                ..
            } => Self {
                operator_name: "Update".to_string(),
                details: vec![("table_id".to_string(), format!("{}", table_id.0))],
                estimated_cost: Some(*cost),
                actual_metrics: None,
                children: vec![Self::from_physical_plan(child)],
            },
            PhysicalPlan::Delete {
                table_id,
                child,
                cost,
            } => Self {
                operator_name: "Delete".to_string(),
                details: vec![("table_id".to_string(), format!("{}", table_id.0))],
                estimated_cost: Some(*cost),
                actual_metrics: None,
                children: vec![Self::from_physical_plan(child)],
            },
            // Parallel plan variants
            PhysicalPlan::ParallelSeqScan {
                table_id,
                columns,
                num_workers,
                cost,
                ..
            } => Self {
                operator_name: "ParallelSeqScan".to_string(),
                details: vec![
                    ("table_id".to_string(), format!("{}", table_id.0)),
                    ("columns".to_string(), format!("{}", columns.len())),
                    ("workers".to_string(), format!("{}", num_workers)),
                ],
                estimated_cost: Some(*cost),
                actual_metrics: None,
                children: Vec::new(),
            },
            PhysicalPlan::ParallelHashJoin {
                left,
                right,
                join_type,
                num_workers,
                cost,
                ..
            } => Self {
                operator_name: "ParallelHashJoin".to_string(),
                details: vec![
                    ("join_type".to_string(), format!("{:?}", join_type)),
                    ("workers".to_string(), format!("{}", num_workers)),
                ],
                estimated_cost: Some(*cost),
                actual_metrics: None,
                children: vec![
                    Self::from_physical_plan(left),
                    Self::from_physical_plan(right),
                ],
            },
            PhysicalPlan::Gather {
                child,
                num_workers,
                cost,
            } => Self {
                operator_name: "Gather".to_string(),
                details: vec![("workers".to_string(), format!("{}", num_workers))],
                estimated_cost: Some(*cost),
                actual_metrics: None,
                children: vec![Self::from_physical_plan(child)],
            },
            PhysicalPlan::Repartition {
                child,
                num_partitions,
                cost,
                ..
            } => Self {
                operator_name: "Repartition".to_string(),
                details: vec![("partitions".to_string(), format!("{}", num_partitions))],
                estimated_cost: Some(*cost),
                actual_metrics: None,
                children: vec![Self::from_physical_plan(child)],
            },
            PhysicalPlan::Broadcast {
                child,
                num_workers,
                cost,
            } => Self {
                operator_name: "Broadcast".to_string(),
                details: vec![("workers".to_string(), format!("{}", num_workers))],
                estimated_cost: Some(*cost),
                actual_metrics: None,
                children: vec![Self::from_physical_plan(child)],
            },

            PhysicalPlan::FulltextScan {
                table_id,
                index_id,
                columns,
                cost,
                ..
            } => Self {
                operator_name: "FulltextScan".to_string(),
                details: vec![
                    ("table_id".to_string(), format!("{}", table_id.0)),
                    ("index_id".to_string(), format!("{}", index_id.0)),
                    ("columns".to_string(), format!("{}", columns.len())),
                ],
                estimated_cost: Some(*cost),
                actual_metrics: None,
                children: Vec::new(),
            },
            PhysicalPlan::VectorScan {
                table_id,
                index_id,
                columns,
                cost,
                k,
                ..
            } => Self {
                operator_name: "VectorScan".to_string(),
                details: vec![
                    ("table_id".to_string(), format!("{}", table_id.0)),
                    ("index_id".to_string(), format!("{}", index_id.0)),
                    ("columns".to_string(), format!("{}", columns.len())),
                    ("k".to_string(), format!("{}", k)),
                ],
                estimated_cost: Some(*cost),
                actual_metrics: None,
                children: Vec::new(),
            },
            PhysicalPlan::SpatialScan {
                table_id,
                index_id,
                columns,
                kind,
                cost,
                ..
            } => {
                let kind_str = match kind {
                    super::physical::SpatialScanKind::Knn { k, .. } => format!("knn(k={})", k),
                    super::physical::SpatialScanKind::DWithin { radius_meters, .. } => {
                        format!("dwithin(radius={:.1}m)", radius_meters)
                    }
                    super::physical::SpatialScanKind::Range { .. } => "range".to_string(),
                };
                Self {
                    operator_name: "SpatialScan".to_string(),
                    details: vec![
                        ("table_id".to_string(), format!("{}", table_id.0)),
                        ("index_id".to_string(), format!("{}", index_id.0)),
                        ("columns".to_string(), format!("{}", columns.len())),
                        ("kind".to_string(), kind_str),
                    ],
                    estimated_cost: Some(*cost),
                    actual_metrics: None,
                    children: Vec::new(),
                }
            }
            PhysicalPlan::GraphAlgorithm {
                schema_name,
                algorithm,
                cost,
                ..
            } => Self {
                operator_name: "GraphAlgorithm".to_string(),
                details: vec![
                    ("schema".to_string(), schema_name.clone()),
                    ("algorithm".to_string(), format!("{:?}", algorithm)),
                ],
                estimated_cost: Some(*cost),
                actual_metrics: None,
                children: Vec::new(),
            },
            PhysicalPlan::Window {
                window_exprs,
                child,
                cost,
                ..
            } => Self {
                operator_name: "Window".to_string(),
                details: vec![("functions".to_string(), format!("{}", window_exprs.len()))],
                estimated_cost: Some(*cost),
                actual_metrics: None,
                children: vec![Self::from_physical_plan(child)],
            },
        }
    }

    /// Merges actual execution metrics into this node and its children.
    /// Metrics are matched by tree position (pre-order traversal).
    pub fn merge_metrics_flat(&mut self, metrics: &[(u64, f64, u64)]) {
        let mut idx = 0;
        self.merge_metrics_recursive(metrics, &mut idx);
    }

    fn merge_metrics_recursive(&mut self, metrics: &[(u64, f64, u64)], idx: &mut usize) {
        if *idx < metrics.len() {
            let (rows, elapsed_ms, batches) = metrics[*idx];
            self.actual_metrics = Some(ActualMetrics {
                rows,
                elapsed_ms,
                batches,
            });
            *idx += 1;
        }
        for child in &mut self.children {
            child.merge_metrics_recursive(metrics, idx);
        }
    }

    /// Renders the explain output in the specified format.
    pub fn render(&self, options: &ExplainOptions) -> String {
        match options.format {
            ExplainFormat::Text => self.to_text(options),
            ExplainFormat::Json => self.to_json(options),
            ExplainFormat::Yaml => self.to_yaml(options),
        }
    }

    // -----------------------------------------------------------------------
    // Text format
    // -----------------------------------------------------------------------

    fn to_text(&self, options: &ExplainOptions) -> String {
        let mut output = String::new();
        self.write_text_node(&mut output, options, 0);
        output
    }

    fn write_text_node(&self, output: &mut String, options: &ExplainOptions, depth: usize) {
        let indent = if depth == 0 {
            String::new()
        } else {
            format!("{}-> ", "  ".repeat(depth))
        };

        let _ = write!(output, "{}{}", indent, self.operator_name);

        // Details
        for (key, value) in &self.details {
            let _ = write!(output, " {}={}", key, value);
        }

        // Estimated cost
        if options.costs {
            if let Some(cost) = &self.estimated_cost {
                let _ = write!(
                    output,
                    " (cost={:.2} rows={:.0})",
                    cost.total(),
                    cost.row_count
                );
            }
        }

        // Actual metrics (ANALYZE)
        if options.analyze {
            if let Some(actual) = &self.actual_metrics {
                let _ = write!(output, " (actual rows={}", actual.rows);
                if options.timing {
                    let _ = write!(output, " time={:.3}ms", actual.elapsed_ms);
                }
                let _ = write!(output, ")");
            }
        }

        let _ = writeln!(output);

        for child in &self.children {
            child.write_text_node(output, options, depth + 1);
        }
    }

    // -----------------------------------------------------------------------
    // JSON format
    // -----------------------------------------------------------------------

    fn to_json(&self, options: &ExplainOptions) -> String {
        let mut output = String::new();
        self.write_json_node(&mut output, options, 0);
        let _ = writeln!(output);
        output
    }

    fn write_json_node(&self, output: &mut String, options: &ExplainOptions, depth: usize) {
        let pad = "  ".repeat(depth);
        let _ = writeln!(output, "{}{{", pad);
        let _ = writeln!(output, "{}  \"operator\": \"{}\",", pad, self.operator_name);

        // Details
        if !self.details.is_empty() {
            let _ = write!(output, "{}  \"details\": {{", pad);
            for (i, (key, value)) in self.details.iter().enumerate() {
                if i > 0 {
                    let _ = write!(output, ", ");
                }
                let _ = write!(output, "\"{}\": \"{}\"", key, value);
            }
            let _ = writeln!(output, "}},");
        }

        // Estimated cost
        if options.costs {
            if let Some(cost) = &self.estimated_cost {
                let _ = writeln!(output, "{}  \"estimated_cost\": {:.2},", pad, cost.total());
                let _ = writeln!(
                    output,
                    "{}  \"estimated_rows\": {:.0},",
                    pad, cost.row_count
                );
            }
        }

        // Actual metrics
        if options.analyze {
            if let Some(actual) = &self.actual_metrics {
                let _ = writeln!(output, "{}  \"actual_rows\": {},", pad, actual.rows);
                if options.timing {
                    let _ = writeln!(
                        output,
                        "{}  \"actual_time_ms\": {:.3},",
                        pad, actual.elapsed_ms
                    );
                }
            }
        }

        // Children
        if self.children.is_empty() {
            let _ = writeln!(output, "{}  \"children\": []", pad);
        } else {
            let _ = writeln!(output, "{}  \"children\": [", pad);
            for (i, child) in self.children.iter().enumerate() {
                child.write_json_node(output, options, depth + 2);
                if i < self.children.len() - 1 {
                    let _ = write!(output, ",");
                }
                let _ = writeln!(output);
            }
            let _ = writeln!(output, "{}  ]", pad);
        }

        let _ = write!(output, "{}}}", pad);
    }

    // -----------------------------------------------------------------------
    // YAML format
    // -----------------------------------------------------------------------

    fn to_yaml(&self, options: &ExplainOptions) -> String {
        let mut output = String::new();
        self.write_yaml_node(&mut output, options, 0);
        output
    }

    fn write_yaml_node(&self, output: &mut String, options: &ExplainOptions, depth: usize) {
        let pad = "  ".repeat(depth);
        let _ = writeln!(output, "{}operator: {}", pad, self.operator_name);

        for (key, value) in &self.details {
            let _ = writeln!(output, "{}{}: {}", pad, key, value);
        }

        if options.costs {
            if let Some(cost) = &self.estimated_cost {
                let _ = writeln!(output, "{}estimated_cost: {:.2}", pad, cost.total());
                let _ = writeln!(output, "{}estimated_rows: {:.0}", pad, cost.row_count);
            }
        }

        if options.analyze {
            if let Some(actual) = &self.actual_metrics {
                let _ = writeln!(output, "{}actual_rows: {}", pad, actual.rows);
                if options.timing {
                    let _ = writeln!(output, "{}actual_time_ms: {:.3}", pad, actual.elapsed_ms);
                }
            }
        }

        if !self.children.is_empty() {
            let _ = writeln!(output, "{}children:", pad);
            for child in &self.children {
                let _ = write!(output, "{}  - operator: {}\n", pad, child.operator_name);
                for (key, value) in &child.details {
                    let _ = writeln!(output, "{}    {}: {}", pad, key, value);
                }
                if options.costs {
                    if let Some(cost) = &child.estimated_cost {
                        let _ = writeln!(output, "{}    estimated_cost: {:.2}", pad, cost.total());
                        let _ =
                            writeln!(output, "{}    estimated_rows: {:.0}", pad, cost.row_count);
                    }
                }
                if options.analyze {
                    if let Some(actual) = &child.actual_metrics {
                        let _ = writeln!(output, "{}    actual_rows: {}", pad, actual.rows);
                        if options.timing {
                            let _ = writeln!(
                                output,
                                "{}    actual_time_ms: {:.3}",
                                pad, actual.elapsed_ms
                            );
                        }
                    }
                }
                if !child.children.is_empty() {
                    let _ = writeln!(output, "{}    children:", pad);
                    for grandchild in &child.children {
                        grandchild.write_yaml_node(output, options, depth + 4);
                    }
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn make_simple_plan() -> ExplainNode {
        ExplainNode {
            operator_name: "HashJoin".to_string(),
            details: vec![("join_type".to_string(), "Inner".to_string())],
            estimated_cost: Some(PlanCost {
                io_cost: 10.0,
                cpu_cost: 50.0,
                row_count: 1000.0,
            }),
            actual_metrics: None,
            children: vec![
                ExplainNode {
                    operator_name: "SeqScan".to_string(),
                    details: vec![("table_id".to_string(), "1".to_string())],
                    estimated_cost: Some(PlanCost {
                        io_cost: 5.0,
                        cpu_cost: 20.0,
                        row_count: 5000.0,
                    }),
                    actual_metrics: None,
                    children: Vec::new(),
                },
                ExplainNode {
                    operator_name: "IndexScan".to_string(),
                    details: vec![("table_id".to_string(), "2".to_string())],
                    estimated_cost: Some(PlanCost {
                        io_cost: 2.0,
                        cpu_cost: 10.0,
                        row_count: 200.0,
                    }),
                    actual_metrics: None,
                    children: Vec::new(),
                },
            ],
        }
    }

    #[test]
    fn test_text_output() {
        let node = make_simple_plan();
        let options = ExplainOptions::default();
        let text = node.render(&options);
        assert!(text.contains("HashJoin"));
        assert!(text.contains("SeqScan"));
        assert!(text.contains("IndexScan"));
        assert!(text.contains("cost="));
        assert!(text.contains("rows="));
    }

    #[test]
    fn test_text_no_costs() {
        let node = make_simple_plan();
        let options = ExplainOptions {
            costs: false,
            ..Default::default()
        };
        let text = node.render(&options);
        assert!(text.contains("HashJoin"));
        assert!(!text.contains("cost="));
    }

    #[test]
    fn test_text_with_analyze() {
        let mut node = make_simple_plan();
        node.actual_metrics = Some(ActualMetrics {
            rows: 982,
            elapsed_ms: 3.2,
            batches: 5,
        });
        let options = ExplainOptions {
            analyze: true,
            ..Default::default()
        };
        let text = node.render(&options);
        assert!(text.contains("actual rows=982"));
        assert!(text.contains("time=3.200ms"));
    }

    #[test]
    fn test_json_output() {
        let node = make_simple_plan();
        let options = ExplainOptions {
            format: ExplainFormat::Json,
            ..Default::default()
        };
        let json = node.render(&options);
        assert!(json.contains("\"operator\": \"HashJoin\""));
        assert!(json.contains("\"children\""));
    }

    #[test]
    fn test_yaml_output() {
        let node = make_simple_plan();
        let options = ExplainOptions {
            format: ExplainFormat::Yaml,
            ..Default::default()
        };
        let yaml = node.render(&options);
        assert!(yaml.contains("operator: HashJoin"));
        assert!(yaml.contains("children:"));
    }

    #[test]
    fn test_merge_metrics() {
        let mut node = make_simple_plan();
        let metrics = vec![(1000, 5.0, 10), (5000, 3.0, 8), (200, 1.5, 4)];
        node.merge_metrics_flat(&metrics);
        assert_eq!(node.actual_metrics.as_ref().map(|m| m.rows), Some(1000));
        assert_eq!(
            node.children[0].actual_metrics.as_ref().map(|m| m.rows),
            Some(5000)
        );
        assert_eq!(
            node.children[1].actual_metrics.as_ref().map(|m| m.rows),
            Some(200)
        );
    }

    #[test]
    fn test_explain_format_from_str() {
        assert_eq!(ExplainFormat::from_str("json"), ExplainFormat::Json);
        assert_eq!(ExplainFormat::from_str("JSON"), ExplainFormat::Json);
        assert_eq!(ExplainFormat::from_str("yaml"), ExplainFormat::Yaml);
        assert_eq!(ExplainFormat::from_str("text"), ExplainFormat::Text);
        assert_eq!(ExplainFormat::from_str("anything"), ExplainFormat::Text);
    }
}
