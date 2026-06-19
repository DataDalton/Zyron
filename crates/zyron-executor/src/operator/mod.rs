//! Operator trait and execution batch types for the volcano-style pull executor.
//!
//! Each operator implements the Operator trait, producing ExecutionBatch results
//! one batch at a time. Operators form a tree where each node pulls data from
//! its children on demand.

pub mod aggregate;
pub mod analytics_table_fn;
pub mod branch_write;
pub mod column_scan;
pub mod distinct;
pub mod filter;
pub mod fk;
pub mod fts_scan;
pub mod gapfill;
pub mod graph_scan;
pub mod join;
pub mod limit;
pub mod modify;
pub mod project;
pub mod scan;
pub mod setop;
pub mod sort;
pub mod spatial_scan;
pub mod vector_scan;
pub mod window;

use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;

use zyron_common::Result;
use zyron_storage::TupleId;

use crate::batch::{ColumnBuilder, DataBatch};
use crate::column::ScalarValue;
use crate::context::ExecutionContext;
use zyron_planner::logical::LogicalColumn;

/// Enforces column-level security on a result batch: classification clearance
/// and masking. Columns the session role lacks clearance for are masked when
/// a masking policy exists, otherwise NULLed (deny). Internal queries (no
/// security context) are returned unchanged. Single source of truth shared by
/// every scan operator so heap and columnar reads enforce identical policy.
pub(crate) fn apply_column_security(
    ctx: &ExecutionContext,
    table_id: u32,
    output_columns: &[LogicalColumn],
    batch: DataBatch,
) -> DataBatch {
    let sc = match &ctx.security_context {
        Some(s) => s,
        None => return batch,
    };
    let sm = match &ctx.security_manager {
        Some(s) => s,
        None => return batch,
    };
    let n = batch.num_rows;
    let mut cols = Vec::with_capacity(batch.columns.len());
    for (i, col) in batch.columns.iter().enumerate() {
        if i >= output_columns.len() {
            cols.push(col.clone());
            continue;
        }
        let col_id = output_columns[i].column_id.0;
        let cleared = sm
            .classification_store
            .check_clearance(sc.clearance, table_id, col_id);
        let mut probe = String::new();
        let has_mask = sm.masking_policy_store.apply_masking(
            table_id,
            col_id,
            "",
            &sc.effective_roles,
            &mut probe,
        );
        if cleared && !has_mask {
            cols.push(col.clone());
            continue;
        }
        let mut b = ColumnBuilder::new(col.type_id, n);
        for r in 0..n {
            let v = col.get_scalar(r);
            let masked_text = if let ScalarValue::Utf8(s) = &v {
                let mut buf = String::new();
                if sm.masking_policy_store.apply_masking(
                    table_id,
                    col_id,
                    s,
                    &sc.effective_roles,
                    &mut buf,
                ) {
                    Some(buf)
                } else {
                    None
                }
            } else {
                None
            };
            if let Some(m) = masked_text {
                b.push(&ScalarValue::Utf8(m));
            } else if cleared {
                b.push(&v);
            } else {
                b.push(&ScalarValue::Null);
            }
        }
        cols.push(b.finish());
    }
    DataBatch::new(cols)
}

/// Boxed future returned by Operator::next().
pub type OperatorResult<'a> =
    Pin<Box<dyn Future<Output = Result<Option<ExecutionBatch>>> + Send + 'a>>;

/// A batch of rows produced by an operator, optionally carrying tuple IDs
/// for DML operators that need to identify specific rows in heap storage.
pub struct ExecutionBatch {
    /// Columnar batch containing the row data.
    pub batch: DataBatch,
    /// Optional tuple IDs for each row, used by UPDATE and DELETE operators
    /// to locate the source tuples in heap pages.
    pub tuple_ids: Option<Vec<TupleId>>,
    /// Optional (columnar_file_id, sys_rowid) per row for rows served from a
    /// .zyr segment. UPDATE and DELETE route these to the columnar patch log
    /// instead of the heap. Aligned 1:1 with batch rows when present.
    pub columnar_locators: Option<Vec<(u64, u64)>>,
}

impl ExecutionBatch {
    /// Creates a new ExecutionBatch without tuple IDs.
    pub fn new(batch: DataBatch) -> Self {
        Self {
            batch,
            tuple_ids: None,
            columnar_locators: None,
        }
    }

    /// Creates a new ExecutionBatch with associated tuple IDs.
    pub fn with_tuple_ids(batch: DataBatch, tuple_ids: Vec<TupleId>) -> Self {
        Self {
            batch,
            tuple_ids: Some(tuple_ids),
            columnar_locators: None,
        }
    }

    /// Creates a batch whose rows are columnar-resident, each carrying its
    /// (file_id, sys_rowid) locator for the DML patch path.
    pub fn with_columnar_locators(batch: DataBatch, locators: Vec<(u64, u64)>) -> Self {
        Self {
            batch,
            tuple_ids: None,
            columnar_locators: Some(locators),
        }
    }

    /// Returns the number of rows in this batch.
    pub fn num_rows(&self) -> usize {
        self.batch.num_rows
    }
}

/// Pull-based operator trait for the volcano execution model.
///
/// Each call to next() returns the next batch of rows, or None when exhausted.
/// Operators are composed into a tree, with leaf operators (scans) reading from
/// storage and interior operators (filter, project, join) transforming data
/// from their children.
///
/// Uses boxed futures for dyn-compatible async dispatch.
pub trait Operator: Send {
    /// Returns the next batch of rows, or None if the operator is exhausted.
    fn next(&mut self) -> OperatorResult<'_>;
}

// ---------------------------------------------------------------------------
// OperatorMetrics - per-operator stats for EXPLAIN ANALYZE
// ---------------------------------------------------------------------------

/// Per-operator metrics collected during query execution.
/// Shared via Arc so the executor can read metrics after the operator
/// tree is drained.
#[derive(Debug)]
pub struct OperatorMetrics {
    /// Display name for the operator (e.g. "SeqScan", "HashJoin").
    pub name: String,
    /// Total rows produced by this operator.
    pub rows_produced: AtomicU64,
    /// Total wall-clock time spent in this operator's next() calls, in nanoseconds.
    pub elapsed_ns: AtomicU64,
    /// Number of times next() was called.
    pub batches: AtomicU64,
    /// Metrics from child operators (forms a tree for display).
    pub children: Vec<Arc<OperatorMetrics>>,
}

impl OperatorMetrics {
    pub fn new(name: &str) -> Arc<Self> {
        Arc::new(Self {
            name: name.to_string(),
            rows_produced: AtomicU64::new(0),
            elapsed_ns: AtomicU64::new(0),
            batches: AtomicU64::new(0),
            children: Vec::new(),
        })
    }

    pub fn with_children(name: &str, children: Vec<Arc<OperatorMetrics>>) -> Arc<Self> {
        Arc::new(Self {
            name: name.to_string(),
            rows_produced: AtomicU64::new(0),
            elapsed_ns: AtomicU64::new(0),
            batches: AtomicU64::new(0),
            children,
        })
    }

    /// Returns elapsed time in fractional milliseconds.
    pub fn elapsed_ms(&self) -> f64 {
        self.elapsed_ns.load(Ordering::Relaxed) as f64 / 1_000_000.0
    }

    /// Formats the metrics tree for display (EXPLAIN ANALYZE output).
    pub fn format_tree(&self, indent: usize) -> String {
        let mut out = String::new();
        let prefix = " ".repeat(indent);
        out.push_str(&format!(
            "{}{} (rows={}, time={:.3}ms, batches={})\n",
            prefix,
            self.name,
            self.rows_produced.load(Ordering::Relaxed),
            self.elapsed_ms(),
            self.batches.load(Ordering::Relaxed),
        ));
        for child in &self.children {
            out.push_str(&child.format_tree(indent + 2));
        }
        out
    }
}

/// Wrapper operator that collects timing and row count metrics around
/// an inner operator. Used by the executor when analyze mode is enabled.
pub struct MetricsOperator {
    inner: Box<dyn Operator>,
    metrics: Arc<OperatorMetrics>,
}

impl MetricsOperator {
    pub fn new(inner: Box<dyn Operator>, metrics: Arc<OperatorMetrics>) -> Self {
        Self { inner, metrics }
    }

    pub fn metrics(&self) -> &Arc<OperatorMetrics> {
        &self.metrics
    }
}

impl Operator for MetricsOperator {
    fn next(&mut self) -> OperatorResult<'_> {
        Box::pin(async move {
            let start = Instant::now();
            let result = self.inner.next().await;
            let elapsed = start.elapsed().as_nanos() as u64;
            self.metrics
                .elapsed_ns
                .fetch_add(elapsed, Ordering::Relaxed);
            self.metrics.batches.fetch_add(1, Ordering::Relaxed);

            if let Ok(Some(ref eb)) = result {
                self.metrics
                    .rows_produced
                    .fetch_add(eb.num_rows() as u64, Ordering::Relaxed);
            }

            result
        })
    }
}
