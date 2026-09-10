//! Operator trait and execution batch types for the volcano-style pull executor.
//!
//! Each operator implements the Operator trait, producing ExecutionBatch results
//! one batch at a time. Operators form a tree where each node pulls data from
//! its children on demand.

pub mod aggregate;
pub mod analytics_table_fn;
pub mod asof_join;
pub mod branch_write;
pub mod column_scan;
pub mod distinct;
pub mod doc_fetch;
pub mod expand_rows;
pub mod filter;
pub mod fk;
pub mod foreign_scan;
pub mod fts_scan;
pub mod gapfill;
pub mod grace;
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
pub mod view_write;
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
    // Columns this pass hands back untouched. A path a scan read out of a
    // variant column survives only when the column itself did, because a
    // masked or withheld document must not be readable one field at a time
    let mut untouched: Vec<u16> = Vec::new();
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
            untouched.push(col_id);
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
    let resolved = batch
        .resolved
        .into_iter()
        .filter(|r| untouched.contains(&r.column_id))
        .collect();
    DataBatch::new(cols).with_resolved(resolved)
}

/// A value an aggregate hands back out of a column's cells, NULLed when
/// the session is not cleared for that column or a masking policy covers
/// it.
///
/// MIN, MAX and SUM all expose something derived from actual values, so
/// they answer to the same column level policy a row scan enforces. COUNT
/// exposes no value and does not come through here
pub(crate) fn expose_column_value(
    ctx: &ExecutionContext,
    table_id: u32,
    column_id: Option<zyron_catalog::ColumnId>,
    value: ScalarValue,
) -> ScalarValue {
    let Some(cid) = column_id else {
        return value;
    };
    let Some((sc, sm)) = ctx
        .security_context
        .as_ref()
        .zip(ctx.security_manager.as_ref())
    else {
        return value;
    };
    let cleared = sm
        .classification_store
        .check_clearance(sc.clearance, table_id, cid.0);
    let mut probe = String::new();
    let has_mask =
        sm.masking_policy_store
            .apply_masking(table_id, cid.0, "", &sc.effective_roles, &mut probe);
    if cleared && !has_mask {
        value
    } else {
        ScalarValue::Null
    }
}

/// One running answer for a metadata pushdown aggregate.
///
/// Shared by the two tiers that answer aggregates from statistics, because
/// both fold the same four kinds and both fall back to folding rows for
/// the files their statistics do not describe. Two copies of this would be
/// two chances for a statistics answer and a scan answer to disagree
pub(crate) enum MetaAcc {
    Count(i64),
    MinMax(Option<ScalarValue>),
    /// Running total and whether any non-null value reached it, because a
    /// SUM over no rows is NULL rather than zero
    Sum {
        total: i128,
        any: bool,
    },
}

impl MetaAcc {
    /// A fresh accumulator per aggregate, in spec order
    pub(crate) fn for_specs(specs: &[zyron_planner::physical::MetaAggSpec]) -> Vec<MetaAcc> {
        use zyron_planner::physical::MetaAggKind;
        specs
            .iter()
            .map(|s| match s.kind {
                MetaAggKind::CountStar | MetaAggKind::CountCol => MetaAcc::Count(0),
                MetaAggKind::Min | MetaAggKind::Max => MetaAcc::MinMax(None),
                MetaAggKind::Sum => MetaAcc::Sum {
                    total: 0,
                    any: false,
                },
            })
            .collect()
    }

    /// Keeps the smaller or larger of what is held and what arrives.
    /// A NULL is not a candidate, MIN and MAX both ignore them
    pub(crate) fn fold_minmax(cur: &mut Option<ScalarValue>, v: ScalarValue, want_max: bool) {
        if matches!(v, ScalarValue::Null) {
            return;
        }
        match cur {
            None => *cur = Some(v),
            Some(c) => {
                if let Some(ord) = v.partial_cmp(c) {
                    let take = if want_max {
                        ord == std::cmp::Ordering::Greater
                    } else {
                        ord == std::cmp::Ordering::Less
                    };
                    if take {
                        *cur = Some(v);
                    }
                }
            }
        }
    }

    /// The scalar this accumulator answers with, on the aggregate's
    /// declared type and before column policy.
    ///
    /// A sum accumulates at 128 bits whatever the column's width, exactly
    /// as the row path's does, so the same narrowing the row path applies
    /// on the way out is applied here. Skipping it lands a 128-bit value
    /// in a builder typed for the declared width, which silently stores
    /// nothing
    pub(crate) fn finish(self, target: zyron_common::TypeId) -> Result<ScalarValue> {
        let raw = match self {
            MetaAcc::Count(c) => ScalarValue::Int64(c),
            MetaAcc::MinMax(m) => m.unwrap_or(ScalarValue::Null),
            MetaAcc::Sum { total, any } => {
                if any {
                    ScalarValue::Int128(total)
                } else {
                    ScalarValue::Null
                }
            }
        };
        crate::operator::aggregate::coerce_aggregate_scalar(raw, target)
    }
}

/// Folds every row an operator produces into metadata aggregate
/// accumulators, for the files whose statistics could not answer.
///
/// `proj_idx` maps each spec to the column of the batch holding its
/// target, or None for COUNT(*), which needs no column
pub(crate) async fn fold_rows_into_meta_accs(
    mut op: Box<dyn Operator>,
    specs: &[zyron_planner::physical::MetaAggSpec],
    proj_idx: &[Option<usize>],
    accs: &mut [MetaAcc],
) -> Result<()> {
    use zyron_planner::physical::MetaAggKind;
    while let Some(eb) = op.next().await? {
        let b = &eb.batch;
        for (si, spec) in specs.iter().enumerate() {
            match (&spec.kind, &mut accs[si]) {
                (MetaAggKind::CountStar, MetaAcc::Count(c)) => {
                    *c += b.num_rows as i64;
                }
                (MetaAggKind::CountCol, MetaAcc::Count(c)) => {
                    if let Some(ci) = proj_idx[si] {
                        let col = &b.columns[ci];
                        for r in 0..b.num_rows {
                            if !col.is_null(r) {
                                *c += 1;
                            }
                        }
                    }
                }
                (MetaAggKind::Min, MetaAcc::MinMax(m)) | (MetaAggKind::Max, MetaAcc::MinMax(m)) => {
                    if let Some(ci) = proj_idx[si] {
                        let want_max = spec.kind == MetaAggKind::Max;
                        let col = &b.columns[ci];
                        for r in 0..b.num_rows {
                            if !col.is_null(r) {
                                MetaAcc::fold_minmax(m, col.get_scalar(r), want_max);
                            }
                        }
                    }
                }
                (MetaAggKind::Sum, MetaAcc::Sum { total, any }) => {
                    if let Some(ci) = proj_idx[si] {
                        let col = &b.columns[ci];
                        for r in 0..b.num_rows {
                            if col.is_null(r) {
                                continue;
                            }
                            let v = col.get_scalar(r).to_i128().ok_or_else(|| {
                                zyron_common::ZyronError::ExecutionError(
                                    "metadata aggregate: SUM over a column that does not add exactly"
                                        .into(),
                                )
                            })?;
                            *total = total.checked_add(v).ok_or_else(|| {
                                zyron_common::ZyronError::ExecutionError(
                                    "SUM overflowed its 128-bit accumulator".to_string(),
                                )
                            })?;
                            *any = true;
                        }
                    }
                }
                _ => {}
            }
        }
    }
    Ok(())
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
/// Pass-through that polls cancellation and the statement deadline before
/// every batch it hands upward. Wrapped around the inputs of blocking
/// operators (joins, aggregates, sorts, set ops, windows): their
/// consume-everything loops would otherwise run to completion after a
/// cancel, because only sources poll while producing and a source that is
/// another operator's buffered output never polls. Costs one relaxed atomic
/// load per batch when no deadline is set.
pub(crate) struct CancelPollOperator {
    child: Box<dyn Operator>,
    ctx: std::sync::Arc<crate::context::ExecutionContext>,
    batches: u32,
}

impl CancelPollOperator {
    pub(crate) fn wrap(
        child: Box<dyn Operator>,
        ctx: &std::sync::Arc<crate::context::ExecutionContext>,
    ) -> Box<dyn Operator> {
        Box::new(Self {
            child,
            ctx: std::sync::Arc::clone(ctx),
            batches: 0,
        })
    }
}

impl Operator for CancelPollOperator {
    fn next(&mut self) -> OperatorResult<'_> {
        Box::pin(async move {
            self.ctx.check_cancelled()?;
            // The compute between operator awaits never touches a tokio leaf
            // resource, so the runtime's cooperative budget never trips and a
            // long pipeline can hog its worker for minutes, starving timers
            // and the accept loop. A periodic explicit yield keeps the
            // runtime scheduling everything else
            self.batches = self.batches.wrapping_add(1);
            if self.batches % 16 == 0 {
                tokio::task::yield_now().await;
            }
            self.child.next().await
        })
    }
}

/// Convention for operators with unbounded compute per `next()` call: the
/// executor's compute never touches a tokio leaf resource, so the runtime's
/// cooperative budget never trips and a long loop can hog its worker and
/// outlive a cancel. Blocking operators get a `CancelPollOperator` wrapped
/// around their inputs at build time; an operator whose OUTPUT phase also
/// loops without bound (a spill merge, a join product) additionally holds a
/// poll context, checks `check_cancelled` per batch, and calls
/// `tokio::task::yield_now` every 16 batches or every 64 inner steps. See
/// `SortOperator` and `NestedLoopJoinOperator` for the two shapes.
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
pub const AUX_SLOTS: usize = 6;

/// Data files a scan's manifest listed
pub const AUX_FILES_CONSIDERED: usize = 0;
/// Data files statistics excluded before any byte was read
pub const AUX_FILES_PRUNED: usize = 1;
/// Bytes those files held in total
pub const AUX_BYTES_CONSIDERED: usize = 2;
/// Bytes the pruned files held, the IO the predicate saved
pub const AUX_BYTES_PRUNED: usize = 3;
/// Index files a lake scan read to address its rows. Zero means it read
/// none, which is what an unindexed column and a declined index both look
/// like from the outside
pub const AUX_INDEX_FILES_READ: usize = 4;
/// Rows a secondary index addressed for a lake scan
pub const AUX_INDEX_ROWS_ADDRESSED: usize = 5;

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

    /// Formats the metrics tree for display (EXPLAIN ANALYZE output).
    pub fn format_tree(&self, indent: usize) -> String {
        let mut out = String::new();
        let prefix = " ".repeat(indent);
        let (ms, frac) = zyron_planner::millis_parts(self.elapsed_ns.load(Ordering::Relaxed));
        out.push_str(&format!(
            "{}{} (rows={}, time={}.{:03}ms, batches={})\n",
            prefix,
            self.name,
            self.rows_produced.load(Ordering::Relaxed),
            ms,
            frac,
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
