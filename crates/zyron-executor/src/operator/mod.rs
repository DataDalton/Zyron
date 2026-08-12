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
pub mod doc_fetch;
pub mod filter;
pub mod fk;
pub mod foreign_scan;
pub mod fts_scan;
pub mod gapfill;
pub mod graph_scan;
pub mod join;
pub mod lake_scan;
pub mod limit;
pub mod lock_rows;
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

use zyron_common::{Result, RowLocator};
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

/// A batch of rows produced by an operator, optionally carrying one storage
/// locator per row so DML operators can address the source rows regardless
/// of which store holds them.
pub struct ExecutionBatch {
    /// Columnar batch containing the row data.
    pub batch: DataBatch,
    /// Per row storage locator, aligned 1:1 with batch rows when present.
    /// Pass through operators (filter, limit) slice this vector with the
    /// same mask they apply to the batch, without caring which store the
    /// rows live in.
    pub locators: Option<Vec<RowLocator>>,
}

impl ExecutionBatch {
    /// Creates a new ExecutionBatch without locators.
    pub fn new(batch: DataBatch) -> Self {
        Self {
            batch,
            locators: None,
        }
    }

    /// Creates a batch with one locator per row.
    pub fn with_locators(batch: DataBatch, locators: Vec<RowLocator>) -> Self {
        Self {
            batch,
            locators: Some(locators),
        }
    }

    /// Creates a batch of heap resident rows from their tuple ids.
    pub fn with_tuple_ids(batch: DataBatch, tuple_ids: Vec<TupleId>) -> Self {
        let locators = tuple_ids.into_iter().map(TupleId::locator).collect();
        Self {
            batch,
            locators: Some(locators),
        }
    }

    /// Creates a batch of columnar resident rows from (file_id, sys_rowid)
    /// pairs for the DML patch path.
    pub fn with_columnar_locators(batch: DataBatch, pairs: Vec<(u64, u64)>) -> Self {
        let locators = pairs
            .into_iter()
            .map(|(file_id, sys_rowid)| RowLocator::Columnar { file_id, sys_rowid })
            .collect();
        Self {
            batch,
            locators: Some(locators),
        }
    }

    /// Returns the number of rows in this batch.
    pub fn num_rows(&self) -> usize {
        self.batch.num_rows
    }

    /// Heap tuple ids when the batch carries locators and every row is heap
    /// resident. An empty locator vector counts as heap so zero row batches
    /// take the heap no-op path.
    pub fn heap_ids(&self) -> Option<Vec<TupleId>> {
        let locs = self.locators.as_ref()?;
        let mut out = Vec::with_capacity(locs.len());
        for l in locs {
            out.push(TupleId::from_locator(*l)?);
        }
        Some(out)
    }

    /// Columnar (file_id, sys_rowid) pairs when the batch carries locators
    /// and every row is columnar resident. Empty means not columnar.
    pub fn columnar_pairs(&self) -> Option<Vec<(u64, u64)>> {
        let locs = self.locators.as_ref()?;
        if locs.is_empty() {
            return None;
        }
        let mut out = Vec::with_capacity(locs.len());
        for l in locs {
            out.push(l.columnar_pair()?);
        }
        Some(out)
    }

    /// Classifies the batch by the storage tier of its locators. None when
    /// the batch carries no locators or mixes tiers, which no DML producer
    /// emits: the index scan breaks batches on a kind change, the hybrid
    /// scan drains one tier before the other, and the FK gather returns one
    /// batch per tier. An empty locator vector counts as Heap so zero row
    /// batches take the heap no-op path
    pub fn tier(&self) -> Option<BatchTier> {
        let locs = self.locators.as_ref()?;
        let mut tier: Option<BatchTier> = None;
        for l in locs {
            let t = match l {
                RowLocator::Heap { .. } => BatchTier::Heap,
                RowLocator::Columnar { .. } => BatchTier::Columnar,
                RowLocator::Lake { .. } => BatchTier::Lake,
            };
            match tier {
                None => tier = Some(t),
                Some(prev) if prev != t => return None,
                Some(_) => {}
            }
        }
        Some(tier.unwrap_or(BatchTier::Heap))
    }
}

/// Storage tier a locator-bearing batch addresses. DML batches are tier
/// homogeneous by construction, so the tier is a batch-level property and
/// the DML operators dispatch on it exhaustively: a new RowLocator variant
/// fails to compile until every dispatch site handles it
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BatchTier {
    Heap,
    Columnar,
    Lake,
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
// IndexScanStats - IO counters shared by every index-driven scan
// ---------------------------------------------------------------------------

/// The table and index IO counters for one scan driven by an index.
///
/// Every index-driven scan has the same accounting shape: the index search
/// runs to completion before any row is fetched, so the entries it examined
/// are known up front, and the fetch loop then resolves locators to rows by
/// reading pages. Four operators do this (B+tree, fulltext, vector and
/// spatial) and writing the six lines four times is how two copies of a
/// rule drift apart, so it is written once here.
///
/// Both counters are resolved when the scan is built and held for its
/// lifetime, which keeps the registry lookup off the batch path.
pub(crate) struct IndexScanStats {
    table: Option<Arc<zyron_common::TableIOStats>>,
    index: Option<Arc<zyron_common::IndexIOStats>>,
}

impl IndexScanStats {
    /// Records one index scan being initiated on the table and the index,
    /// plus the index entries the search examined.
    ///
    /// `entries_examined` is what the search produced before visibility or
    /// any remaining predicate is applied, so the gap between it and the
    /// rows eventually fetched is entries that pointed at a row this
    /// snapshot could not see.
    pub(crate) fn open(
        ctx: &ExecutionContext,
        table_id: u32,
        index_id: u32,
        entries_examined: usize,
    ) -> Self {
        let table = ctx.table_io_stats_for(table_id);
        if let Some(stats) = &table {
            stats.record_idx_scan();
        }
        let index = ctx.index_io_stats_for(index_id);
        if let Some(stats) = &index {
            stats.record_scan();
            stats.record_batch(entries_examined as u64, 0);
        }
        Self { table, index }
    }

    /// Records one batch of fetched rows and the bytes of table data read
    /// to fetch them. Called once per batch, never once per row.
    #[inline]
    pub(crate) fn record_batch(&self, rows: u64, bytes: u64) {
        if let Some(stats) = &self.table {
            stats.record_idx_batch(rows, bytes);
        }
        if let Some(stats) = &self.index {
            stats.record_batch(0, rows);
        }
    }
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
    /// Operator-specific counters. Fixed width and inline, so filling one
    /// is a relaxed add with no allocation and no trait change, and what
    /// each slot means is resolved at render time by the operator's name
    pub aux: [AtomicU64; AUX_SLOTS],
    /// Metrics from child operators (forms a tree for display).
    pub children: Vec<Arc<OperatorMetrics>>,
}

/// Auxiliary counter slots per operator.
pub const AUX_SLOTS: usize = 4;

/// Data files a scan's manifest listed.
pub const AUX_FILES_CONSIDERED: usize = 0;
/// Data files statistics excluded before any byte was read.
pub const AUX_FILES_PRUNED: usize = 1;
/// Bytes those files held in total.
pub const AUX_BYTES_CONSIDERED: usize = 2;
/// Bytes the pruned files held, the IO the predicate saved.
pub const AUX_BYTES_PRUNED: usize = 3;

/// Rows a peer returned for a foreign scan. Shares slot 0 with the file
/// count because no operator reports both, and the labels an operator's
/// name selects are what give a slot its meaning.
pub const AUX_ROWS_FETCHED: usize = 0;
/// Milliseconds the round trip to the peer took, the part of a foreign
/// scan's cost that no local tuning changes.
pub const AUX_REMOTE_MS: usize = 1;

impl OperatorMetrics {
    pub fn new(name: &str) -> Arc<Self> {
        Arc::new(Self {
            name: name.to_string(),
            rows_produced: AtomicU64::new(0),
            elapsed_ns: AtomicU64::new(0),
            batches: AtomicU64::new(0),
            aux: Default::default(),
            children: Vec::new(),
        })
    }

    /// Sets one auxiliary counter. Used for a quantity the operator knows
    /// outright rather than accumulates, such as how many files a scan's
    /// statistics excluded before it opened any of them
    #[inline]
    pub fn set_aux(&self, slot: usize, value: u64) {
        if let Some(counter) = self.aux.get(slot) {
            counter.store(value, Ordering::Relaxed);
        }
    }

    /// Reads one auxiliary counter.
    #[inline]
    pub fn aux(&self, slot: usize) -> u64 {
        self.aux
            .get(slot)
            .map(|c| c.load(Ordering::Relaxed))
            .unwrap_or(0)
    }

    pub fn with_children(name: &str, children: Vec<Arc<OperatorMetrics>>) -> Arc<Self> {
        Arc::new(Self {
            name: name.to_string(),
            rows_produced: AtomicU64::new(0),
            elapsed_ns: AtomicU64::new(0),
            batches: AtomicU64::new(0),
            aux: Default::default(),
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
