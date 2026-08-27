//! Runner that drives a Zyron-to-Zyron streaming job end to end.
//!
//! A runner owns a background thread that repeatedly polls the source CDF,
//! applies an optional predicate and projection, and pushes the surviving
//! rows through a ZyronRowSink. The runner is opened with a StreamingJobSpec
//! that carries only plain data, intentionally avoiding any dependency on
//! the planner so this crate stays a leaf. The caller is responsible for
//! translating a BoundStreamingJob into a StreamingJobSpec at dispatch time.
//!
//! The runner stops when its AtomicBool stop flag is set. On any pipeline
//! error, it writes the job status to Failed through the catalog and exits.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::thread::JoinHandle;
use std::time::Duration;

use parking_lot::Mutex as PlMutex;
use zyron_catalog::schema::{CatalogStreamingWriteMode, StreamingJobStatus};
use zyron_catalog::{Catalog, StreamingJobEntry, StreamingJobId};
use zyron_common::{Result, TypeId, ZyronError};

use crate::interval_join_runner::{
    IntervalJoinEngine, IntervalJoinEngineConfig, JoinSide, temporal_left_null_right,
};
use crate::row_codec::{
    CompiledExpr, StreamValue, compile_expr, decode_row, encode_projected_row, encode_row,
    eval_compiled,
};
use crate::sink_connector::ZyronRowSink;
use crate::source_connector::{CdfChange, ZyronTableSource};
use crate::upsert_sink::ZyronUpsertSink;

// -----------------------------------------------------------------------------
// Join runtime
// -----------------------------------------------------------------------------

/// Live join state a joining runner drives each poll cycle. Built from the
/// spec's JoinSpec at spawn, so a job whose right side has no feed fails at
/// spawn instead of silently running unjoined.
pub(crate) enum JoinRuntime {
    Interval {
        right: ZyronTableSource,
        engine: IntervalJoinEngine,
        cfg: IntervalJoinConfig,
        /// Highest event time seen per side, the punctual watermark inputs.
        left_high: i64,
        right_high: i64,
    },
    Temporal {
        right: ZyronTableSource,
        cfg: TemporalJoinConfig,
        /// Right-table image keyed by encoded primary key, maintained from
        /// the right CDF: inserts and update postimages upsert, deletes
        /// remove.
        cache: std::collections::HashMap<Vec<u8>, Vec<StreamValue>>,
    },
}

impl JoinRuntime {
    /// Builds the runtime for a spec's join, resolving the right-side feed.
    pub(crate) fn build(
        join: &JoinSpec,
        cdc_registry: &Arc<zyron_cdc::CdfRegistry>,
    ) -> Result<Self> {
        match join {
            JoinSpec::Interval(cfg) => Ok(JoinRuntime::Interval {
                right: ZyronTableSource::new(cfg.right_source_table_id, Arc::clone(cdc_registry))?,
                engine: IntervalJoinEngine::new(IntervalJoinEngineConfig {
                    left_types: cfg.left_types.clone(),
                    right_types: cfg.right_types.clone(),
                    left_key_ordinals: cfg.left_key_ordinals.clone(),
                    right_key_ordinals: cfg.right_key_ordinals.clone(),
                    left_event_ordinal: cfg.left_event_time_ordinal,
                    right_event_ordinal: cfg.right_event_time_ordinal,
                    within_us: cfg.within_us,
                    join_kind: cfg.join_kind,
                }),
                cfg: cfg.clone(),
                left_high: i64::MIN,
                right_high: i64::MIN,
            }),
            JoinSpec::Temporal(cfg) => Ok(JoinRuntime::Temporal {
                right: ZyronTableSource::new(cfg.right_table_id, Arc::clone(cdc_registry))?,
                cfg: cfg.clone(),
                cache: std::collections::HashMap::new(),
            }),
        }
    }

    /// Runs one join step: consumes the right side's new changes, feeds the
    /// left batch through, and returns the combined rows encoded on the
    /// join's output schema.
    fn step(&mut self, left: Vec<CdfChange>) -> Result<Vec<CdfChange>> {
        match self {
            JoinRuntime::Interval {
                right,
                engine,
                cfg,
                left_high,
                right_high,
            } => {
                // A joined row synthesizes its commit stamp from the newest
                // input seen this cycle, so downstream ordering stays sane
                let mut version = 0u64;
                let mut timestamp = 0i64;

                let right_records = right.read_batch(RUNNER_BATCH)?;
                for rec in &right_records {
                    if !matches!(
                        rec.change_type,
                        zyron_cdc::ChangeType::Insert | zyron_cdc::ChangeType::UpdatePostimage
                    ) {
                        continue;
                    }
                    version = version.max(rec.commit_version);
                    timestamp = timestamp.max(rec.commit_timestamp);
                    let row = decode_row(&rec.row_data, &cfg.right_types)?;
                    if let Some(v) = row.get(cfg.right_event_time_ordinal as usize) {
                        if let Ok(t) = v.as_i64() {
                            *right_high = (*right_high).max(t);
                        }
                    }
                    engine.feed_row(JoinSide::Right, row)?;
                }
                for rec in &left {
                    if !matches!(
                        rec.change_type,
                        zyron_cdc::ChangeType::Insert | zyron_cdc::ChangeType::UpdatePostimage
                    ) {
                        continue;
                    }
                    version = version.max(rec.commit_version);
                    timestamp = timestamp.max(rec.commit_timestamp);
                    let row = decode_row(&rec.row_data, &cfg.left_types)?;
                    if let Some(v) = row.get(cfg.left_event_time_ordinal as usize) {
                        if let Ok(t) = v.as_i64() {
                            *left_high = (*left_high).max(t);
                        }
                    }
                    engine.feed_row(JoinSide::Left, row)?;
                }

                // Punctual watermark: the slower side bounds how far time
                // has provably advanced on both inputs
                let wm = (*left_high).min(*right_high);
                if wm > i64::MIN {
                    engine.advance_watermark(wm);
                }

                let mut out = Vec::new();
                for row in engine.pop_emissions() {
                    out.push(CdfChange {
                        commit_version: version,
                        commit_timestamp: timestamp,
                        change_type: zyron_cdc::ChangeType::Insert,
                        row_data: encode_row(&row, &cfg.output_types)?,
                        primary_key_data: Vec::new(),
                    });
                }
                Ok(out)
            }
            JoinRuntime::Temporal { right, cfg, cache } => {
                // Maintain the right-table image first, so a left row looks
                // up the newest state the feed has published
                let right_records = right.read_batch(RUNNER_BATCH)?;
                let pk_types: Vec<TypeId> = cfg
                    .right_pk_ordinals
                    .iter()
                    .map(|&o| cfg.right_types[o as usize])
                    .collect();
                for rec in &right_records {
                    match rec.change_type {
                        zyron_cdc::ChangeType::Insert | zyron_cdc::ChangeType::UpdatePostimage => {
                            let row = decode_row(&rec.row_data, &cfg.right_types)?;
                            let key_values: Vec<StreamValue> = cfg
                                .right_pk_ordinals
                                .iter()
                                .map(|&o| row[o as usize].clone())
                                .collect();
                            let key = encode_row(&key_values, &pk_types)?;
                            cache.insert(key, row);
                        }
                        zyron_cdc::ChangeType::Delete => {
                            let row = decode_row(&rec.row_data, &cfg.right_types)?;
                            let key_values: Vec<StreamValue> = cfg
                                .right_pk_ordinals
                                .iter()
                                .map(|&o| row[o as usize].clone())
                                .collect();
                            let key = encode_row(&key_values, &pk_types)?;
                            cache.remove(&key);
                        }
                        _ => {}
                    }
                }

                let right_width = cfg.right_types.len();
                let mut out = Vec::with_capacity(left.len());
                for rec in &left {
                    if !matches!(
                        rec.change_type,
                        zyron_cdc::ChangeType::Insert | zyron_cdc::ChangeType::UpdatePostimage
                    ) {
                        continue;
                    }
                    let row = decode_row(&rec.row_data, &cfg.left_types)?;
                    // A NULL lookup key matches nothing, so it takes the
                    // outer-miss path directly instead of encoding a null
                    // key that could collide with a real one
                    let null_key = cfg
                        .left_key_ordinals
                        .iter()
                        .any(|&o| matches!(row.get(o as usize), Some(StreamValue::Null)));
                    let hit = if null_key {
                        None
                    } else {
                        let key_values: Vec<StreamValue> = cfg
                            .left_key_ordinals
                            .iter()
                            .map(|&o| row[o as usize].clone())
                            .collect();
                        cache.get(&encode_row(&key_values, &pk_types)?)
                    };
                    let combined = match (hit, cfg.join_kind) {
                        (Some(right_row), _) => {
                            let mut c = Vec::with_capacity(row.len() + right_row.len());
                            c.extend(row.iter().cloned());
                            c.extend(right_row.iter().cloned());
                            c
                        }
                        (None, StreamingJoinKind::Left) => {
                            temporal_left_null_right(&row, right_width)
                        }
                        (None, _) => continue,
                    };
                    out.push(CdfChange {
                        commit_version: rec.commit_version,
                        commit_timestamp: rec.commit_timestamp,
                        change_type: zyron_cdc::ChangeType::Insert,
                        row_data: encode_row(&combined, &cfg.output_types)?,
                        primary_key_data: Vec::new(),
                    });
                }
                Ok(out)
            }
        }
    }
}

// -----------------------------------------------------------------------------
// Sink dispatch
// -----------------------------------------------------------------------------

/// Runtime sink wrapper. The runner's write path branches on this enum so a
/// single run loop can drive either Append or Upsert semantics without
/// duplicating the poll, filter, and project scaffolding.
pub enum RunnerSink {
    Append(ZyronRowSink),
    Upsert(ZyronUpsertSink),
    // -----------------------------------------------------------------------------
    // Remote variant dispatches to a ZyronSinkAdapter trait object, letting
    // the runner push rows to a remote Zyron instance over the PG wire
    // protocol through a concrete client in the zyron-wire crate.
    Remote(Arc<dyn crate::sink_connector::ZyronSinkAdapter>),
}

impl RunnerSink {
    pub(crate) async fn write_batch(&self, records: Vec<CdfChange>) -> Result<()> {
        match self {
            RunnerSink::Append(s) => s.write_batch(records),
            RunnerSink::Upsert(s) => s.write_batch(records),
            RunnerSink::Remote(adapter) => adapter.write_batch(records).await,
        }
    }

    /// Writes anything the sink is still holding.
    ///
    /// A remote sink coalesces small batches into larger transactions, so it
    /// can be holding rows that `write_batch` accepted but has not sent. The
    /// runner calls this wherever the stream goes quiet or stops, which is
    /// what bounds how long a row can wait. The local sinks write through and
    /// have nothing to drain.
    pub(crate) async fn flush(&self) -> Result<()> {
        match self {
            RunnerSink::Append(_) | RunnerSink::Upsert(_) => Ok(()),
            RunnerSink::Remote(adapter) => adapter.flush().await,
        }
    }

    /// Whether a batch `write_batch` accepted is committed when the call
    /// returns. The local sinks write transactionally per batch, so ingest
    /// progress can be acknowledged right after a write. The remote sink
    /// coalesces, its acknowledgment has to wait for a flush.
    pub(crate) fn writes_through(&self) -> bool {
        matches!(self, RunnerSink::Append(_) | RunnerSink::Upsert(_))
    }
}

// ---------------------------------------------------------------------------
// StreamingJobSpec
// ---------------------------------------------------------------------------

/// Self-contained description of a streaming job's filter and project work.
/// Built by the wire layer from a BoundStreamingJob. Kept free of planner
/// types so this crate does not pull in the planner dependency.
#[derive(Debug, Clone)]
pub struct StreamingJobSpec {
    pub source_table_id: u32,
    pub target_table_id: u32,
    pub write_mode: CatalogStreamingWriteMode,
    pub projections: Vec<ExprSpec>,
    pub predicate: Option<ExprSpec>,
    /// Column types of the source table in ordinal order. Required by the
    /// row decoder to turn CDF row_data into StreamValue per column.
    pub source_types: Vec<TypeId>,
    /// Column types of the target table in ordinal order. Required by the
    /// row encoder to repack projection outputs into a tuple the target
    /// heap can accept.
    pub target_types: Vec<TypeId>,
    // -----------------------------------------------------------------------------
    // Target-table ordinals for the primary-key columns. Populated only when
    // write_mode is Upsert, empty otherwise. Used by ZyronUpsertSink to
    // derive the PK lookup key from each decoded source row. The wire layer
    // resolves BoundStreamingJob.target_pk_columns to ordinals here.
    pub target_pk_ordinals: Vec<u16>,
    // -----------------------------------------------------------------------------
    // Aggregate pipeline. When None, the runner runs the filter+project loop.
    // When Some, the runner uses the aggregating loop that assigns events to
    // windows, maintains per (window, key) accumulators, and emits finalized
    // rows as watermarks close windows.
    pub aggregate: Option<AggregateSpec>,
    // -----------------------------------------------------------------------------
    // Join pipeline. When set, the runner drives the stream-stream interval
    // join or stream-table temporal join instead of the plain filter+project
    // loop. Populated by the wire lowering step from the bound plan.
    pub join: Option<JoinSpec>,
}

/// Runner-visible JOIN shape. Picks between the stream-stream interval
/// engine and the stream-table temporal engine. The wire layer builds one
/// of these from a BoundStreamingJoinSpec.
#[derive(Debug, Clone)]
pub enum JoinSpec {
    Interval(IntervalJoinConfig),
    Temporal(TemporalJoinConfig),
}

/// Streaming-join row-match semantics. Mirrors the planner's
/// BoundStreamingJoinType but lives in the streaming crate so the runner
/// does not depend on the planner.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StreamingJoinKind {
    Inner,
    Left,
    Right,
    Full,
}

/// Interval-join runtime parameters. Column ordinals are in the source's
/// own column order on each side. Type lists describe the decoded row shape
/// for the left and right CDFs, and the combined output schema written to
/// the sink.
#[derive(Debug, Clone)]
pub struct IntervalJoinConfig {
    pub left_source_table_id: u32,
    pub right_source_table_id: u32,
    pub left_types: Vec<TypeId>,
    pub right_types: Vec<TypeId>,
    pub output_types: Vec<TypeId>,
    /// Multi-column equi-key ordinals on the left side. Matches
    /// right_key_ordinals pairwise.
    pub left_key_ordinals: Vec<u16>,
    pub right_key_ordinals: Vec<u16>,
    pub left_event_time_ordinal: u16,
    pub right_event_time_ordinal: u16,
    pub within_us: i64,
    pub watermark: crate::watermark::WatermarkStrategy,
    pub join_kind: StreamingJoinKind,
}

/// Temporal-join runtime parameters. The right side is a Zyron table looked
/// up per left-side row by primary key. pk_ordinals are within the right
/// table's own column order.
#[derive(Debug, Clone)]
pub struct TemporalJoinConfig {
    pub left_source_table_id: u32,
    pub right_table_id: u32,
    pub left_types: Vec<TypeId>,
    pub right_types: Vec<TypeId>,
    pub output_types: Vec<TypeId>,
    /// Multi-column equi-key ordinals on the left side. For temporal joins
    /// every column maps to the right side's primary-key columns in order.
    pub left_key_ordinals: Vec<u16>,
    pub right_pk_ordinals: Vec<u16>,
    pub left_event_time_ordinal: u16,
    /// Only Inner and Left are meaningful. Right and Full are rejected at
    /// bind time.
    pub join_kind: StreamingJoinKind,
}

// ---------------------------------------------------------------------------
// AggregateSpec: windowed aggregate configuration
// ---------------------------------------------------------------------------

/// Instructions for the aggregating runner. Built by the wire lowering step
/// from BoundAggregateSpec on the bound streaming job.
#[derive(Debug, Clone)]
pub struct AggregateSpec {
    /// Window shape. One of Tumbling, Hopping, Session. The runner builds the
    /// matching assigner from this enum.
    pub window_type: AggWindowType,
    /// Source-column ordinal carrying the event time value.
    pub event_time_ordinal: u16,
    /// Normalization for the event time column. Microseconds is the canonical
    /// runner unit, everything else is scaled on read.
    pub event_time_scale: EventTimeScale,
    /// Source-column ordinals that form the grouping key.
    pub group_by_ordinals: Vec<u16>,
    /// Ordered list of aggregates computed per window per key. Output rows
    /// contain the group-by columns followed by one column per item in this
    /// list, followed by window_start and window_end.
    pub aggregations: Vec<AggregateItem>,
    /// Watermark strategy. The runner advances the watermark on each event
    /// and on every batch boundary.
    pub watermark: crate::watermark::WatermarkStrategy,
    /// Policy for events that arrive after their window has closed.
    pub late_data_policy: crate::late_data::LateDataPolicy,
}

/// Runner-visible window type. Mirrors window::WindowType but avoids leaking
/// the assigner construction into the spec consumer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AggWindowType {
    Tumbling { size_ms: i64 },
    Hopping { size_ms: i64, slide_ms: i64 },
    Session { gap_ms: i64 },
}

/// One aggregated output. input_ordinal is None for COUNT(*). For everything
/// else the runner reads the source column at the ordinal and feeds it to
/// the accumulator resolved from the function name.
#[derive(Debug, Clone)]
pub struct AggregateItem {
    pub function: String,
    pub input_ordinal: Option<u16>,
    pub input_type: TypeId,
}

/// How to normalize the source event time column to microseconds.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EventTimeScale {
    Microseconds,
    Milliseconds,
    Seconds,
}

impl EventTimeScale {
    /// Converts a raw i64 event-time value into microseconds.
    #[inline]
    pub fn to_micros(&self, raw: i64) -> i64 {
        match self {
            EventTimeScale::Microseconds => raw,
            EventTimeScale::Milliseconds => raw.saturating_mul(1_000),
            EventTimeScale::Seconds => raw.saturating_mul(1_000_000),
        }
    }
}

// ---------------------------------------------------------------------------
// ExprSpec
// ---------------------------------------------------------------------------

/// Minimal expression tree for streaming-job filter and project evaluation.
/// Used in place of the planner's BoundExpr so this crate has no planner
/// dependency. The wire layer's lower_bsj_to_spec function is responsible
/// for lowering BoundExpr into ExprSpec.
#[derive(Debug, Clone)]
pub enum ExprSpec {
    /// Literal scalar. Streaming predicates primarily compare against
    /// integer or boolean literals.
    LiteralBool(bool),
    LiteralI64(i64),
    LiteralF64(f64),
    LiteralString(String),
    /// Reference to a source column by ordinal. The wire layer's
    /// lower_bsj_to_spec resolves the source BoundExpr::ColumnRef into the
    /// source table's column index here.
    ColumnRef {
        ordinal: u16,
    },
    /// Binary operation. Evaluated by row_codec::eval_expr against a decoded
    /// StreamValue row.
    BinaryOp {
        op: BinaryOpKind,
        left: Box<ExprSpec>,
        right: Box<ExprSpec>,
    },
    /// Unary not.
    Not(Box<ExprSpec>),
}

/// Binary operators understood by the runner's evaluator.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BinaryOpKind {
    Eq,
    NotEq,
    Lt,
    LtEq,
    Gt,
    GtEq,
    And,
    Or,
    Add,
    Sub,
    Mul,
    Div,
}

// ---------------------------------------------------------------------------
// StreamJobHandle
// ---------------------------------------------------------------------------

/// Handle to a running streaming job thread. Dropping the handle does not
/// stop the thread. Callers must call StreamJobManager::stop_job.
pub struct StreamJobHandle {
    pub stop_flag: Arc<AtomicBool>,
    thread: Option<JoinHandle<()>>,
}

impl StreamJobHandle {
    /// Signals the thread to stop and waits for it to exit.
    pub fn stop(&mut self) {
        self.stop_flag.store(true, Ordering::Release);
        if let Some(handle) = self.thread.take() {
            let _ = handle.join();
        }
    }
}

// ---------------------------------------------------------------------------
// Runner loop
// ---------------------------------------------------------------------------

/// Fixed poll batch size. Tuned small enough to return control to the stop
/// check often, large enough to amortize CDF query overhead.
const RUNNER_BATCH: usize = 1024;

/// Sleep between polls when the source has no new records. Balances latency
/// against idle CPU usage.
const RUNNER_IDLE_MS: u64 = 100;

/// Drives the source into the sink until the stop flag is set or a pipeline
/// error occurs. A non-fatal empty poll sleeps for RUNNER_IDLE_MS.
/// Fatal errors write the job status to Failed through the catalog.
fn run_loop(
    entry: StreamingJobEntry,
    spec: StreamingJobSpec,
    source: ZyronTableSource,
    mut join_runtime: Option<JoinRuntime>,
    sink: RunnerSink,
    catalog: Arc<Catalog>,
    stop_flag: Arc<AtomicBool>,
) {
    // Joined rows carry the join's combined output schema, which is what
    // the predicate and projections were bound against, so the post-join
    // decode uses it instead of the left source's own shape
    let mut spec = spec;
    if let Some(rt) = &join_runtime {
        spec.source_types = match rt {
            JoinRuntime::Interval { cfg, .. } => cfg.output_types.clone(),
            JoinRuntime::Temporal { cfg, .. } => cfg.output_types.clone(),
        };
    }
    // Build a single runtime per thread for async catalog updates.
    let rt = match tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
    {
        Ok(r) => r,
        Err(e) => {
            tracing::error!(job_id = entry.id.0, "failed to build runtime: {e}");
            return;
        }
    };

    loop {
        if stop_flag.load(Ordering::Acquire) {
            // Anything staged for coalescing is written before the runner
            // goes away, so a stop never strands a partial batch
            let _ = rt.block_on(async { sink.flush().await });
            break;
        }

        // Pause handling. A paused job sleeps and re-checks status until
        // resumed or stopped.
        let current_status = catalog.get_streaming_job_by_id(entry.id).map(|j| j.status);
        match current_status {
            Some(StreamingJobStatus::Paused) => {
                std::thread::sleep(Duration::from_millis(RUNNER_IDLE_MS));
                continue;
            }
            Some(StreamingJobStatus::Failed) => break,
            None => break,
            _ => {}
        }

        // Poll a batch from the source.
        let records = match source.read_batch(RUNNER_BATCH) {
            Ok(v) => v,
            Err(e) => {
                mark_failed(&rt, &catalog, entry.id, format!("source error: {e}"));
                break;
            }
        };

        // A joining job runs its join step even on an empty left batch: the
        // right side may have new rows or the watermark may release buffered
        // outer emissions
        let records = match join_runtime.as_mut() {
            Some(join) => match join.step(records) {
                Ok(v) => v,
                Err(e) => {
                    mark_failed(&rt, &catalog, entry.id, format!("join error: {e}"));
                    break;
                }
            },
            None => records,
        };

        if records.is_empty() {
            // The source has nothing more right now, so anything the sink is
            // still coalescing has no later batch to join and goes out now
            let _ = rt.block_on(async { sink.flush().await });
            std::thread::sleep(Duration::from_millis(RUNNER_IDLE_MS));
            continue;
        }

        // Apply predicate and projection. apply_filter_project uses a fast
        // passthrough when projections are ColumnRef in source order and
        // there is no predicate, otherwise it decodes, evaluates, and
        // re-encodes per row via row_codec.
        let filtered = match apply_filter_project(&records, &spec) {
            Ok(v) => v,
            Err(e) => {
                mark_failed(&rt, &catalog, entry.id, format!("expression error: {e}"));
                break;
            }
        };

        if filtered.is_empty() {
            continue;
        }

        // Write to sink. RunnerSink::write_batch is async to support Remote
        // adapters, so block on the current-thread runtime owned by the
        // runner thread.
        if let Err(e) = rt.block_on(async { sink.write_batch(filtered).await }) {
            mark_failed(&rt, &catalog, entry.id, format!("sink error: {e}"));
            break;
        }
    }
}

/// Runs a runner body under a panic guard. A panic inside the body is caught
/// and the job is transitioned to Failed in the catalog, matching how an Err
/// return is handled, instead of unwinding the runner thread silently. The
/// body is given a runtime so it can update the catalog if it returns normally.
fn run_guarded<F>(entry: &StreamingJobEntry, catalog: &Arc<Catalog>, body: F)
where
    F: FnOnce(),
{
    // The runner owns its source, sink, and catalog handle. On a panic the
    // body's state is dropped and only the failed-status update runs after,
    // so asserting unwind safety here is sound. AssertUnwindSafe<F> implements
    // FnOnce, so it is passed directly as the catch_unwind body.
    if std::panic::catch_unwind(std::panic::AssertUnwindSafe(body)).is_err() {
        let rt = match tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
        {
            Ok(r) => r,
            Err(e) => {
                tracing::error!(
                    job_id = entry.id.0,
                    "runner panicked and the failed-status runtime build failed: {e}"
                );
                return;
            }
        };
        mark_failed(&rt, catalog, entry.id, "runner thread panicked".to_string());
    }
}

/// Transitions the job to Failed in the catalog. Swallows update errors
/// because the runner is already on the exit path and has no recourse.
fn mark_failed(
    rt: &tokio::runtime::Runtime,
    catalog: &Arc<Catalog>,
    id: StreamingJobId,
    reason: String,
) {
    tracing::error!(job_id = id.0, reason = %reason, "streaming job failed");
    let res = rt.block_on(async {
        catalog
            .update_streaming_job_status(id, StreamingJobStatus::Failed, Some(reason))
            .await
    });
    if let Err(e) = res {
        tracing::error!(job_id = id.0, "failed to persist failed status: {e}");
    }
}

// ---------------------------------------------------------------------------
// Filter and project
// ---------------------------------------------------------------------------

/// Applies the job spec's predicate and projections to a batch of CDF rows.
/// For each record the runner decodes row_data into StreamValue per source
/// column, evaluates the predicate, evaluates each projection, then
/// re-encodes the projected values into a new tuple that matches the target
/// schema. The primary_key_data is preserved unchanged.
fn apply_filter_project(records: &[CdfChange], spec: &StreamingJobSpec) -> Result<Vec<CdfChange>> {
    // Fast path: no predicate, projections are ColumnRef in source order, and
    // source_types matches target_types. Copy input rows straight through.
    let is_identity =
        spec.predicate.is_none()
            && spec.source_types == spec.target_types
            && spec.projections.iter().enumerate().all(
                |(i, p)| matches!(p, ExprSpec::ColumnRef { ordinal } if *ordinal as usize == i),
            )
            && spec.projections.len() == spec.source_types.len();
    if is_identity {
        return Ok(records.to_vec());
    }

    // Compile once per record batch: literals materialize a single time, so
    // per-row evaluation borrows every leaf and the projected values encode
    // straight into the output tuple without an intermediate vector
    let compiled_predicate = spec.predicate.as_ref().map(compile_expr);
    let compiled_projections: Vec<CompiledExpr> =
        spec.projections.iter().map(compile_expr).collect();

    let mut out = Vec::with_capacity(records.len());
    for rec in records {
        let row = decode_row(&rec.row_data, &spec.source_types)?;

        if let Some(pred) = &compiled_predicate {
            match eval_compiled(pred, &row)?.as_ref() {
                StreamValue::Bool(true) => {}
                _ => continue,
            }
        }

        let new_bytes = encode_projected_row(&compiled_projections, &row, &spec.target_types)?;

        out.push(CdfChange {
            commit_version: rec.commit_version,
            commit_timestamp: rec.commit_timestamp,
            change_type: rec.change_type,
            row_data: new_bytes,
            primary_key_data: rec.primary_key_data.clone(),
        });
    }
    Ok(out)
}

// ---------------------------------------------------------------------------
// StreamJobManager extension
// ---------------------------------------------------------------------------

use crate::job::StreamJobManager;

impl StreamJobManager {
    /// Spawns a Zyron-to-Zyron streaming runner thread. The caller supplies
    /// the bind-time spec, a reconstructed security context, and shared
    /// handles to the storage and CDC subsystems. The runner registers
    /// itself in the manager's handle map keyed by StreamingJobId so
    /// stop_job can find and terminate it.
    #[allow(clippy::too_many_arguments)]
    pub fn spawn_zyron_table_job(
        &self,
        entry: StreamingJobEntry,
        spec: StreamingJobSpec,
        security_ctx: zyron_auth::SecurityContext,
        catalog: Arc<Catalog>,
        heap_for_target: Arc<zyron_storage::HeapFile>,
        cdc_registry: Arc<zyron_cdc::CdfRegistry>,
        txn_manager: Arc<zyron_storage::txn::TransactionManager>,
        wal: Arc<zyron_wal::WalWriter>,
        security_manager: Arc<zyron_auth::SecurityManager>,
        io_stats: Arc<zyron_common::TableIOStatsRegistry>,
    ) -> Result<()> {
        // Build source and sink. Branch on the write mode so UPSERT jobs are
        // driven by ZyronUpsertSink and APPEND jobs stay on ZyronRowSink.
        let source = ZyronTableSource::new(spec.source_table_id, Arc::clone(&cdc_registry))?;
        // A joining job resolves its right-side feed here, so a missing feed
        // fails the spawn loudly instead of the runner silently producing
        // unjoined rows
        // The aggregating loop does not consume a join, so the combination
        // is refused loudly instead of silently dropping the join
        if spec.join.is_some() && spec.aggregate.is_some() {
            return Err(ZyronError::StreamingError(
                "a streaming job cannot combine a join with a windowed aggregate".into(),
            ));
        }
        let join_runtime = match &spec.join {
            Some(join) => Some(JoinRuntime::build(join, &cdc_registry)?),
            None => None,
        };
        let ctx_arc = Arc::new(PlMutex::new(security_ctx));
        let sink = match spec.write_mode {
            CatalogStreamingWriteMode::Upsert => {
                let upsert = ZyronUpsertSink::new(
                    spec.target_table_id,
                    spec.target_pk_ordinals.clone(),
                    spec.target_types.clone(),
                    Arc::clone(&catalog),
                    heap_for_target,
                    txn_manager,
                    wal,
                    Arc::clone(&ctx_arc),
                    security_manager,
                    io_stats,
                )?;
                RunnerSink::Upsert(upsert)
            }
            CatalogStreamingWriteMode::Append => RunnerSink::Append(ZyronRowSink::new(
                spec.target_table_id,
                spec.write_mode,
                Arc::clone(&catalog),
                heap_for_target,
                txn_manager,
                Arc::clone(&ctx_arc),
                security_manager,
                io_stats,
            )),
        };

        let stop_flag = Arc::new(AtomicBool::new(false));
        let stop_for_thread = Arc::clone(&stop_flag);
        let entry_for_thread = entry.clone();
        let spec_for_thread = spec.clone();
        let catalog_for_thread = Arc::clone(&catalog);

        let thread = std::thread::Builder::new()
            .name(format!("zyron-stream-{}", entry.id.0))
            .spawn(move || {
                let guard_entry = entry_for_thread.clone();
                let guard_catalog = Arc::clone(&catalog_for_thread);
                run_guarded(&guard_entry, &guard_catalog, move || {
                    if let Some(agg) = spec_for_thread.aggregate.clone() {
                        crate::agg_runner::run_aggregating_loop(
                            entry_for_thread,
                            spec_for_thread.source_types.clone(),
                            agg,
                            source,
                            sink,
                            catalog_for_thread,
                            stop_for_thread,
                        );
                    } else {
                        run_loop(
                            entry_for_thread,
                            spec_for_thread,
                            source,
                            join_runtime,
                            sink,
                            catalog_for_thread,
                            stop_for_thread,
                        );
                    }
                });
            })
            .map_err(|e| {
                ZyronError::StreamingError(format!("failed to spawn runner thread: {e}"))
            })?;

        let handle = StreamJobHandle {
            stop_flag,
            thread: Some(thread),
        };
        self.register_handle(entry.id, handle);
        Ok(())
    }

    /// Signals the runner for a given streaming job id to stop and waits for
    /// the thread to exit. Returns an error if no handle is registered.
    pub fn stop_job(&self, id: StreamingJobId) -> Result<()> {
        let mut handle = self.take_handle(id).ok_or_else(|| {
            ZyronError::StreamingError(format!("no running job with id {}", id.0))
        })?;
        handle.stop();
        Ok(())
    }
}

// Runner end-to-end coverage lives at the integration level in
// zyron-server/tests. Unit tests for the decoded-row filter and project path
// live in row_codec.rs.

#[cfg(test)]
mod remote_sink_tests {
    use super::*;
    use crate::source_connector::CdfChange;
    use std::sync::Mutex as StdMutex;

    struct MockAdapter {
        calls: StdMutex<Vec<Vec<CdfChange>>>,
        fail: bool,
    }

    #[async_trait::async_trait]
    impl crate::sink_connector::ZyronSinkAdapter for MockAdapter {
        async fn write_batch(&self, records: Vec<CdfChange>) -> Result<()> {
            if self.fail {
                return Err(ZyronError::StreamingError("mock failure".into()));
            }
            self.calls.lock().unwrap().push(records);
            Ok(())
        }
        async fn flush(&self) -> Result<()> {
            Ok(())
        }
        async fn shutdown(&self) -> Result<()> {
            Ok(())
        }
    }

    fn mk_change(v: u64) -> CdfChange {
        CdfChange {
            commit_version: v,
            commit_timestamp: 0,
            change_type: zyron_cdc::ChangeType::Insert,
            row_data: vec![0],
            primary_key_data: vec![],
        }
    }

    #[tokio::test]
    async fn runner_sink_remote_delegates_to_adapter() {
        let mock = Arc::new(MockAdapter {
            calls: StdMutex::new(Vec::new()),
            fail: false,
        });
        let sink =
            RunnerSink::Remote(mock.clone() as Arc<dyn crate::sink_connector::ZyronSinkAdapter>);
        sink.write_batch(vec![mk_change(1), mk_change(2)])
            .await
            .unwrap();
        let calls = mock.calls.lock().unwrap();
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].len(), 2);
        assert_eq!(calls[0][0].commit_version, 1);
    }

    #[tokio::test]
    async fn runner_remote_adapter_errors_propagate() {
        let mock = Arc::new(MockAdapter {
            calls: StdMutex::new(Vec::new()),
            fail: true,
        });
        let sink = RunnerSink::Remote(mock as Arc<dyn crate::sink_connector::ZyronSinkAdapter>);
        let res = sink.write_batch(vec![mk_change(1)]).await;
        assert!(res.is_err());
    }
}

// ---------------------------------------------------------------------------
// External source and sink spawn methods
// ---------------------------------------------------------------------------
//
// These build on the shared run-loop shape. For external endpoints the
// per-row filter and project work runs on already-decoded rows, there is no
// CDF envelope to preserve. Scheduled mode sleeps between passes based on
// the catalog entry's schedule_cron. Watch and OneShot share a tight loop
// that pages through the source.

use crate::external_sink::ExternalRowSink;
use crate::external_source::ExternalTableSource;
use std::time::Instant;
use zyron_catalog::schema::ExternalMode;

/// Drives external-to-external streaming pipelines. Applies the spec's
/// predicate and projections to decoded rows, writes the surviving rows to
/// the external sink. One-shot mode exits when the source reports
/// exhausted. Watch mode loops until stopped.
fn run_external_loop(
    entry: StreamingJobEntry,
    spec: StreamingJobSpec,
    source: Arc<ExternalTableSource>,
    sink: Arc<ExternalRowSink>,
    catalog: Arc<Catalog>,
    stop_flag: Arc<AtomicBool>,
    mode: ExternalMode,
    schedule_cron: Option<String>,
) {
    let rt = match tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
    {
        Ok(r) => r,
        Err(e) => {
            tracing::error!(job_id = entry.id.0, "failed to build runtime: {e}");
            return;
        }
    };

    // Compile schedule once. Falls back to a fixed interval parsed from the
    // string when cron parsing fails, supporting strings like "60s", "5m".
    let schedule = schedule_cron.as_deref().and_then(parse_schedule);

    loop {
        if stop_flag.load(Ordering::Acquire) {
            // Best effort on the way out. An unflushed batch keeps its
            // objects unacknowledged, so a restart re-reads them
            if rt.block_on(async { sink.flush().await }).is_ok() {
                if let Err(e) = source.commit_progress() {
                    tracing::warn!(
                        job_id = entry.id.0,
                        "ingest progress commit on stop failed: {e}"
                    );
                }
            }
            break;
        }

        let current_status = catalog.get_streaming_job_by_id(entry.id).map(|j| j.status);
        match current_status {
            Some(StreamingJobStatus::Paused) => {
                std::thread::sleep(Duration::from_millis(RUNNER_IDLE_MS));
                continue;
            }
            Some(StreamingJobStatus::Failed) => break,
            None => break,
            _ => {}
        }

        // Read one batch from the source.
        let rows = match source.read_batch(RUNNER_BATCH) {
            Ok(v) => v,
            Err(e) => {
                mark_failed(&rt, &catalog, entry.id, format!("source error: {e}"));
                break;
            }
        };

        if !rows.is_empty() {
            let filtered = match apply_external_filter_project(&rows, &spec) {
                Ok(v) => v,
                Err(e) => {
                    mark_failed(&rt, &catalog, entry.id, format!("expression error: {e}"));
                    break;
                }
            };
            if !filtered.is_empty() {
                if let Err(e) = rt.block_on(async { sink.write_batch(filtered).await }) {
                    mark_failed(&rt, &catalog, entry.id, format!("sink error: {e}"));
                    break;
                }
            }
        }

        match mode {
            ExternalMode::OneShot => {
                if source.exhausted() {
                    // Completing with rows still coalesced in the sink would
                    // report success for rows that never landed, so a flush
                    // failure fails the job instead of finishing it
                    if let Err(e) = rt.block_on(async { sink.flush().await }) {
                        mark_failed(&rt, &catalog, entry.id, format!("sink flush error: {e}"));
                        break;
                    }
                    if let Err(e) = source.commit_progress() {
                        mark_failed(&rt, &catalog, entry.id, format!("progress error: {e}"));
                        break;
                    }
                    // No Completed status exists, transition to Paused to
                    // indicate the runner finished cleanly.
                    let _ = rt.block_on(async {
                        catalog
                            .update_streaming_job_status(
                                entry.id,
                                StreamingJobStatus::Paused,
                                Some("one-shot completed".to_string()),
                            )
                            .await
                    });
                    break;
                }
                if rows.is_empty() {
                    // The source has nothing more right now, so anything the sink is
                    // still coalescing has no later batch to join and goes out now.
                    // Once the flush lands, the delivered objects are acknowledged
                    // in the ingest progress log. A failure leaves them
                    // unacknowledged and both retry on the next idle tick
                    match rt.block_on(async { sink.flush().await }) {
                        Ok(()) => {
                            if let Err(e) = source.commit_progress() {
                                tracing::warn!(
                                    job_id = entry.id.0,
                                    "ingest progress commit failed: {e}"
                                );
                            }
                        }
                        Err(e) => tracing::warn!(
                            job_id = entry.id.0,
                            "sink flush failed, ingest progress not committed: {e}"
                        ),
                    }
                    std::thread::sleep(Duration::from_millis(RUNNER_IDLE_MS));
                }
            }
            ExternalMode::Watch => {
                if rows.is_empty() {
                    // The source has nothing more right now, so anything the sink is
                    // still coalescing has no later batch to join and goes out now.
                    // Once the flush lands, the delivered objects are acknowledged
                    // in the ingest progress log. A failure leaves them
                    // unacknowledged and both retry on the next idle tick
                    match rt.block_on(async { sink.flush().await }) {
                        Ok(()) => {
                            if let Err(e) = source.commit_progress() {
                                tracing::warn!(
                                    job_id = entry.id.0,
                                    "ingest progress commit failed: {e}"
                                );
                            }
                        }
                        Err(e) => tracing::warn!(
                            job_id = entry.id.0,
                            "sink flush failed, ingest progress not committed: {e}"
                        ),
                    }
                    std::thread::sleep(Duration::from_millis(RUNNER_IDLE_MS));
                }
            }
            ExternalMode::Scheduled => {
                if source.exhausted() {
                    match rt.block_on(async { sink.flush().await }) {
                        Ok(()) => {
                            if let Err(e) = source.commit_progress() {
                                tracing::warn!(
                                    job_id = entry.id.0,
                                    "ingest progress commit failed: {e}"
                                );
                            }
                        }
                        Err(e) => tracing::warn!(
                            job_id = entry.id.0,
                            "sink flush failed, ingest progress not committed: {e}"
                        ),
                    }
                    // Wait until next scheduled tick or stop.
                    let delay = schedule
                        .as_ref()
                        .map(|s| s.next_delay())
                        .unwrap_or(Duration::from_secs(60));
                    let start = Instant::now();
                    while start.elapsed() < delay {
                        if stop_flag.load(Ordering::Acquire) {
                            break;
                        }
                        std::thread::sleep(Duration::from_millis(RUNNER_IDLE_MS));
                    }
                }
            }
        }
    }
}

/// Parsed schedule for the scheduled-mode runner. Supports cron strings
/// through the cron crate, and simple interval strings like "60s", "5m",
/// "2h", "1d" through a local parser.
enum Schedule {
    Cron(cron::Schedule),
    Interval(Duration),
}

impl Schedule {
    fn next_delay(&self) -> Duration {
        match self {
            Schedule::Cron(c) => {
                let now = chrono_now();
                match c.upcoming(chrono::Utc).next() {
                    Some(next) => {
                        let secs = (next.timestamp() - now).max(1) as u64;
                        Duration::from_secs(secs)
                    }
                    None => Duration::from_secs(60),
                }
            }
            Schedule::Interval(d) => *d,
        }
    }
}

fn chrono_now() -> i64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs() as i64)
        .unwrap_or(0)
}

fn parse_schedule(s: &str) -> Option<Schedule> {
    use std::str::FromStr;
    if let Ok(sched) = cron::Schedule::from_str(s) {
        return Some(Schedule::Cron(sched));
    }
    parse_interval(s).map(Schedule::Interval)
}

fn parse_interval(s: &str) -> Option<Duration> {
    let s = s.trim();
    if s.is_empty() {
        return None;
    }
    let (num_part, unit) = s.split_at(s.len().saturating_sub(1));
    let n: u64 = num_part.trim().parse().ok()?;
    match unit {
        "s" => Some(Duration::from_secs(n)),
        "m" => Some(Duration::from_secs(n * 60)),
        "h" => Some(Duration::from_secs(n * 3600)),
        "d" => Some(Duration::from_secs(n * 86400)),
        _ => None,
    }
}

/// Applies the spec's predicate and projections to already-decoded rows.
fn apply_external_filter_project(
    rows: &[Vec<StreamValue>],
    spec: &StreamingJobSpec,
) -> Result<Vec<Vec<StreamValue>>> {
    // Identity passthrough shortcut.
    let is_identity =
        spec.predicate.is_none()
            && spec.projections.iter().enumerate().all(
                |(i, p)| matches!(p, ExprSpec::ColumnRef { ordinal } if *ordinal as usize == i),
            )
            && spec.projections.len() == spec.source_types.len();
    if is_identity {
        return Ok(rows.to_vec());
    }

    // Compile once per call so predicate evaluation borrows its leaves. The
    // projected output rows are owned, so each projected value pays exactly
    // one clone when it leaves the borrowed evaluation
    let compiled_predicate = spec.predicate.as_ref().map(compile_expr);
    let compiled_projections: Vec<CompiledExpr> =
        spec.projections.iter().map(compile_expr).collect();

    let mut out = Vec::with_capacity(rows.len());
    for row in rows {
        if let Some(pred) = &compiled_predicate {
            match eval_compiled(pred, row)?.as_ref() {
                StreamValue::Bool(true) => {}
                _ => continue,
            }
        }
        let mut projected = Vec::with_capacity(compiled_projections.len());
        for p in &compiled_projections {
            projected.push(eval_compiled(p, row)?.into_owned());
        }
        out.push(projected);
    }
    Ok(out)
}

impl StreamJobManager {
    /// Spawns an external-source to Zyron-table runner. The sink consumes
    /// encoded rows, so the runner encodes each projected row through the
    /// target_types before handing it to the sink.
    #[allow(clippy::too_many_arguments)]
    pub fn spawn_external_to_zyron_job(
        &self,
        entry: StreamingJobEntry,
        spec: StreamingJobSpec,
        source: Arc<ExternalTableSource>,
        sink: RunnerSink,
        mode: ExternalMode,
        schedule_cron: Option<String>,
        catalog: Arc<Catalog>,
    ) -> Result<()> {
        let stop_flag = Arc::new(AtomicBool::new(false));
        let stop_for_thread = Arc::clone(&stop_flag);
        let entry_for_thread = entry.clone();
        let spec_for_thread = spec.clone();
        let catalog_for_thread = Arc::clone(&catalog);
        let source_for_thread = Arc::clone(&source);

        let thread = std::thread::Builder::new()
            .name(format!("zyron-ext-in-{}", entry.id.0))
            .spawn(move || {
                let guard_entry = entry_for_thread.clone();
                let guard_catalog = Arc::clone(&catalog_for_thread);
                run_guarded(&guard_entry, &guard_catalog, move || {
                    run_external_to_zyron_loop(
                        entry_for_thread,
                        spec_for_thread,
                        source_for_thread,
                        sink,
                        catalog_for_thread,
                        stop_for_thread,
                        mode,
                        schedule_cron,
                    );
                });
            })
            .map_err(|e| ZyronError::StreamingError(format!("spawn external-in: {e}")))?;
        self.register_handle(
            entry.id,
            StreamJobHandle {
                stop_flag,
                thread: Some(thread),
            },
        );
        Ok(())
    }

    /// Spawns a Zyron-table to external-sink runner. Uses the existing
    /// ZyronTableSource for CDF reads, decodes each row, then buffers into
    /// the external sink.
    #[allow(clippy::too_many_arguments)]
    pub fn spawn_zyron_to_external_job(
        &self,
        entry: StreamingJobEntry,
        spec: StreamingJobSpec,
        source: ZyronTableSource,
        sink: Arc<ExternalRowSink>,
        catalog: Arc<Catalog>,
    ) -> Result<()> {
        let stop_flag = Arc::new(AtomicBool::new(false));
        let stop_for_thread = Arc::clone(&stop_flag);
        let entry_for_thread = entry.clone();
        let spec_for_thread = spec.clone();
        let catalog_for_thread = Arc::clone(&catalog);
        let sink_for_thread = Arc::clone(&sink);

        let thread = std::thread::Builder::new()
            .name(format!("zyron-ext-out-{}", entry.id.0))
            .spawn(move || {
                let guard_entry = entry_for_thread.clone();
                let guard_catalog = Arc::clone(&catalog_for_thread);
                run_guarded(&guard_entry, &guard_catalog, move || {
                    run_zyron_to_external_loop(
                        entry_for_thread,
                        spec_for_thread,
                        source,
                        sink_for_thread,
                        catalog_for_thread,
                        stop_for_thread,
                    );
                });
            })
            .map_err(|e| ZyronError::StreamingError(format!("spawn external-out: {e}")))?;
        self.register_handle(
            entry.id,
            StreamJobHandle {
                stop_flag,
                thread: Some(thread),
            },
        );
        Ok(())
    }

    /// Spawns an external-to-external runner.
    #[allow(clippy::too_many_arguments)]
    pub fn spawn_external_to_external_job(
        &self,
        entry: StreamingJobEntry,
        spec: StreamingJobSpec,
        source: Arc<ExternalTableSource>,
        sink: Arc<ExternalRowSink>,
        mode: ExternalMode,
        schedule_cron: Option<String>,
        catalog: Arc<Catalog>,
    ) -> Result<()> {
        let stop_flag = Arc::new(AtomicBool::new(false));
        let stop_for_thread = Arc::clone(&stop_flag);
        let entry_for_thread = entry.clone();
        let spec_for_thread = spec.clone();
        let catalog_for_thread = Arc::clone(&catalog);
        let source_for_thread = Arc::clone(&source);
        let sink_for_thread = Arc::clone(&sink);

        let thread = std::thread::Builder::new()
            .name(format!("zyron-ext-ext-{}", entry.id.0))
            .spawn(move || {
                let guard_entry = entry_for_thread.clone();
                let guard_catalog = Arc::clone(&catalog_for_thread);
                run_guarded(&guard_entry, &guard_catalog, move || {
                    run_external_loop(
                        entry_for_thread,
                        spec_for_thread,
                        source_for_thread,
                        sink_for_thread,
                        catalog_for_thread,
                        stop_for_thread,
                        mode,
                        schedule_cron,
                    );
                });
            })
            .map_err(|e| ZyronError::StreamingError(format!("spawn ext-ext: {e}")))?;
        self.register_handle(
            entry.id,
            StreamJobHandle {
                stop_flag,
                thread: Some(thread),
            },
        );
        Ok(())
    }
}

fn run_external_to_zyron_loop(
    entry: StreamingJobEntry,
    spec: StreamingJobSpec,
    source: Arc<ExternalTableSource>,
    sink: RunnerSink,
    catalog: Arc<Catalog>,
    stop_flag: Arc<AtomicBool>,
    mode: ExternalMode,
    schedule_cron: Option<String>,
) {
    let rt = match tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
    {
        Ok(r) => r,
        Err(e) => {
            tracing::error!(job_id = entry.id.0, "failed to build runtime: {e}");
            return;
        }
    };
    let schedule = schedule_cron.as_deref().and_then(parse_schedule);

    loop {
        if stop_flag.load(Ordering::Acquire) {
            // Best effort on the way out. An unflushed batch keeps its
            // objects unacknowledged, so a restart re-reads them
            if rt.block_on(async { sink.flush().await }).is_ok() {
                if let Err(e) = source.commit_progress() {
                    tracing::warn!(
                        job_id = entry.id.0,
                        "ingest progress commit on stop failed: {e}"
                    );
                }
            }
            break;
        }
        let current_status = catalog.get_streaming_job_by_id(entry.id).map(|j| j.status);
        match current_status {
            Some(StreamingJobStatus::Paused) => {
                std::thread::sleep(Duration::from_millis(RUNNER_IDLE_MS));
                continue;
            }
            Some(StreamingJobStatus::Failed) => break,
            None => break,
            _ => {}
        }

        let rows = match source.read_batch(RUNNER_BATCH) {
            Ok(v) => v,
            Err(e) => {
                mark_failed(&rt, &catalog, entry.id, format!("source error: {e}"));
                break;
            }
        };

        if !rows.is_empty() {
            let filtered = match apply_external_filter_project(&rows, &spec) {
                Ok(v) => v,
                Err(e) => {
                    mark_failed(&rt, &catalog, entry.id, format!("expression error: {e}"));
                    break;
                }
            };
            // Encode rows into CdfChange envelopes for the Zyron sink.
            let mut changes = Vec::with_capacity(filtered.len());
            for row in filtered {
                let bytes = match crate::row_codec::encode_row(&row, &spec.target_types) {
                    Ok(b) => b,
                    Err(e) => {
                        mark_failed(&rt, &catalog, entry.id, format!("encode error: {e}"));
                        return;
                    }
                };
                changes.push(CdfChange {
                    commit_version: 0,
                    commit_timestamp: 0,
                    change_type: zyron_cdc::ChangeType::Insert,
                    row_data: bytes,
                    primary_key_data: Vec::new(),
                });
            }
            if !changes.is_empty() {
                if let Err(e) = rt.block_on(async { sink.write_batch(changes).await }) {
                    mark_failed(&rt, &catalog, entry.id, format!("sink error: {e}"));
                    break;
                }
            }
            // A write-through sink has committed this batch, so the objects
            // it drained can be acknowledged now. The coalescing sink's
            // acknowledgment waits for its flush on the idle tick
            if sink.writes_through() {
                if let Err(e) = source.commit_progress() {
                    mark_failed(&rt, &catalog, entry.id, format!("progress error: {e}"));
                    break;
                }
            }
        }

        match mode {
            ExternalMode::OneShot => {
                if source.exhausted() {
                    // Completing with rows still coalesced in the sink would
                    // report success for rows that never landed, so a flush
                    // failure fails the job instead of finishing it
                    if let Err(e) = rt.block_on(async { sink.flush().await }) {
                        mark_failed(&rt, &catalog, entry.id, format!("sink flush error: {e}"));
                        break;
                    }
                    if let Err(e) = source.commit_progress() {
                        mark_failed(&rt, &catalog, entry.id, format!("progress error: {e}"));
                        break;
                    }
                    let _ = rt.block_on(async {
                        catalog
                            .update_streaming_job_status(
                                entry.id,
                                StreamingJobStatus::Paused,
                                Some("one-shot completed".to_string()),
                            )
                            .await
                    });
                    break;
                }
                if rows.is_empty() {
                    // The source has nothing more right now, so anything the sink is
                    // still coalescing has no later batch to join and goes out now.
                    // Once the flush lands, the delivered objects are acknowledged
                    // in the ingest progress log. A failure leaves them
                    // unacknowledged and both retry on the next idle tick
                    match rt.block_on(async { sink.flush().await }) {
                        Ok(()) => {
                            if let Err(e) = source.commit_progress() {
                                tracing::warn!(
                                    job_id = entry.id.0,
                                    "ingest progress commit failed: {e}"
                                );
                            }
                        }
                        Err(e) => tracing::warn!(
                            job_id = entry.id.0,
                            "sink flush failed, ingest progress not committed: {e}"
                        ),
                    }
                    std::thread::sleep(Duration::from_millis(RUNNER_IDLE_MS));
                }
            }
            ExternalMode::Watch => {
                if rows.is_empty() {
                    // The source has nothing more right now, so anything the sink is
                    // still coalescing has no later batch to join and goes out now.
                    // Once the flush lands, the delivered objects are acknowledged
                    // in the ingest progress log. A failure leaves them
                    // unacknowledged and both retry on the next idle tick
                    match rt.block_on(async { sink.flush().await }) {
                        Ok(()) => {
                            if let Err(e) = source.commit_progress() {
                                tracing::warn!(
                                    job_id = entry.id.0,
                                    "ingest progress commit failed: {e}"
                                );
                            }
                        }
                        Err(e) => tracing::warn!(
                            job_id = entry.id.0,
                            "sink flush failed, ingest progress not committed: {e}"
                        ),
                    }
                    std::thread::sleep(Duration::from_millis(RUNNER_IDLE_MS));
                }
            }
            ExternalMode::Scheduled => {
                if source.exhausted() {
                    match rt.block_on(async { sink.flush().await }) {
                        Ok(()) => {
                            if let Err(e) = source.commit_progress() {
                                tracing::warn!(
                                    job_id = entry.id.0,
                                    "ingest progress commit failed: {e}"
                                );
                            }
                        }
                        Err(e) => tracing::warn!(
                            job_id = entry.id.0,
                            "sink flush failed, ingest progress not committed: {e}"
                        ),
                    }
                    let delay = schedule
                        .as_ref()
                        .map(|s| s.next_delay())
                        .unwrap_or(Duration::from_secs(60));
                    let start = Instant::now();
                    while start.elapsed() < delay {
                        if stop_flag.load(Ordering::Acquire) {
                            break;
                        }
                        std::thread::sleep(Duration::from_millis(RUNNER_IDLE_MS));
                    }
                }
            }
        }
    }
}

fn run_zyron_to_external_loop(
    entry: StreamingJobEntry,
    spec: StreamingJobSpec,
    source: ZyronTableSource,
    sink: Arc<ExternalRowSink>,
    catalog: Arc<Catalog>,
    stop_flag: Arc<AtomicBool>,
) {
    let rt = match tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
    {
        Ok(r) => r,
        Err(e) => {
            tracing::error!(job_id = entry.id.0, "failed to build runtime: {e}");
            return;
        }
    };

    loop {
        if stop_flag.load(Ordering::Acquire) {
            let _ = rt.block_on(async { sink.flush().await });
            break;
        }
        let current_status = catalog.get_streaming_job_by_id(entry.id).map(|j| j.status);
        match current_status {
            Some(StreamingJobStatus::Paused) => {
                std::thread::sleep(Duration::from_millis(RUNNER_IDLE_MS));
                continue;
            }
            Some(StreamingJobStatus::Failed) => break,
            None => break,
            _ => {}
        }

        let records = match source.read_batch(RUNNER_BATCH) {
            Ok(v) => v,
            Err(e) => {
                mark_failed(&rt, &catalog, entry.id, format!("source error: {e}"));
                break;
            }
        };
        if records.is_empty() {
            // The source has nothing more right now, so anything the sink is
            // still coalescing has no later batch to join and goes out now
            let _ = rt.block_on(async { sink.flush().await });
            std::thread::sleep(Duration::from_millis(RUNNER_IDLE_MS));
            continue;
        }

        // Decode each CDF row into StreamValue, apply filter + project.
        let mut decoded: Vec<Vec<StreamValue>> = Vec::with_capacity(records.len());
        for rec in &records {
            match crate::row_codec::decode_row(&rec.row_data, &spec.source_types) {
                Ok(row) => decoded.push(row),
                Err(e) => {
                    mark_failed(&rt, &catalog, entry.id, format!("decode error: {e}"));
                    return;
                }
            }
        }
        let filtered = match apply_external_filter_project(&decoded, &spec) {
            Ok(v) => v,
            Err(e) => {
                mark_failed(&rt, &catalog, entry.id, format!("expression error: {e}"));
                break;
            }
        };

        if !filtered.is_empty() {
            if let Err(e) = rt.block_on(async { sink.write_batch(filtered).await }) {
                mark_failed(&rt, &catalog, entry.id, format!("sink error: {e}"));
                break;
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Local CDF source to RunnerSink spawn
// ---------------------------------------------------------------------------

impl StreamJobManager {
    /// Spawns a runner that reads from a local ZyronTableSource and writes
    /// into a pre-built RunnerSink. Used when the sink is a Remote adapter
    /// rather than a local ZyronRowSink/ZyronUpsertSink, so the caller has
    /// already constructed the sink and does not need the heap or txn
    /// dependencies that spawn_zyron_table_job requires.
    pub fn spawn_zyron_source_to_runner_sink_job(
        &self,
        entry: StreamingJobEntry,
        spec: StreamingJobSpec,
        source: ZyronTableSource,
        sink: RunnerSink,
        catalog: Arc<Catalog>,
        cdc_registry: Arc<zyron_cdc::CdfRegistry>,
    ) -> Result<()> {
        // The aggregating loop does not consume a join, so the combination
        // is refused loudly instead of silently dropping the join
        if spec.join.is_some() && spec.aggregate.is_some() {
            return Err(ZyronError::StreamingError(
                "a streaming job cannot combine a join with a windowed aggregate".into(),
            ));
        }
        let join_runtime = match &spec.join {
            Some(join) => Some(JoinRuntime::build(join, &cdc_registry)?),
            None => None,
        };
        let stop_flag = Arc::new(AtomicBool::new(false));
        let stop_for_thread = Arc::clone(&stop_flag);
        let entry_for_thread = entry.clone();
        let spec_for_thread = spec.clone();
        let catalog_for_thread = Arc::clone(&catalog);

        let thread = std::thread::Builder::new()
            .name(format!("zyron-src-runner-{}", entry.id.0))
            .spawn(move || {
                let guard_entry = entry_for_thread.clone();
                let guard_catalog = Arc::clone(&catalog_for_thread);
                run_guarded(&guard_entry, &guard_catalog, move || {
                    if let Some(agg) = spec_for_thread.aggregate.clone() {
                        crate::agg_runner::run_aggregating_loop(
                            entry_for_thread,
                            spec_for_thread.source_types.clone(),
                            agg,
                            source,
                            sink,
                            catalog_for_thread,
                            stop_for_thread,
                        );
                    } else {
                        run_loop(
                            entry_for_thread,
                            spec_for_thread,
                            source,
                            join_runtime,
                            sink,
                            catalog_for_thread,
                            stop_for_thread,
                        );
                    }
                });
            })
            .map_err(|e| ZyronError::StreamingError(format!("spawn src-runner: {e}")))?;
        self.register_handle(
            entry.id,
            StreamJobHandle {
                stop_flag,
                thread: Some(thread),
            },
        );
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Remote Zyron source spawn
// ---------------------------------------------------------------------------

impl StreamJobManager {
    /// Spawns a runner that reads from a remote Zyron publication through a
    /// ZyronSourceAdapter trait object and writes into a local RunnerSink.
    /// The adapter's run method owns the pull loop. on_batch forwards each
    /// batch through the spec's filter and project stages into the sink on
    /// the runner's tokio runtime.
    #[allow(clippy::too_many_arguments)]
    pub fn spawn_remote_source_to_zyron_job(
        &self,
        entry: StreamingJobEntry,
        spec: StreamingJobSpec,
        source: Arc<dyn crate::source_connector::ZyronSourceAdapter>,
        sink: RunnerSink,
        catalog: Arc<Catalog>,
        start_lsn: u64,
    ) -> Result<()> {
        let stop_flag = Arc::new(AtomicBool::new(false));
        let stop_for_thread = Arc::clone(&stop_flag);
        let entry_for_thread = entry.clone();
        let spec_for_thread = spec.clone();
        let catalog_for_thread = Arc::clone(&catalog);

        let thread = std::thread::Builder::new()
            .name(format!("zyron-remote-src-{}", entry.id.0))
            .spawn(move || {
                let guard_entry = entry_for_thread.clone();
                let guard_catalog = Arc::clone(&catalog_for_thread);
                run_guarded(&guard_entry, &guard_catalog, move || {
                    run_remote_source_to_zyron_loop(
                        entry_for_thread,
                        spec_for_thread,
                        source,
                        sink,
                        catalog_for_thread,
                        stop_for_thread,
                        start_lsn,
                    );
                });
            })
            .map_err(|e| ZyronError::StreamingError(format!("spawn remote-src: {e}")))?;
        self.register_handle(
            entry.id,
            StreamJobHandle {
                stop_flag,
                thread: Some(thread),
            },
        );
        Ok(())
    }
}

fn run_remote_source_to_zyron_loop(
    entry: StreamingJobEntry,
    spec: StreamingJobSpec,
    source: Arc<dyn crate::source_connector::ZyronSourceAdapter>,
    sink: RunnerSink,
    catalog: Arc<Catalog>,
    stop_flag: Arc<AtomicBool>,
    start_lsn: u64,
) {
    let rt = match tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
    {
        Ok(r) => r,
        Err(e) => {
            tracing::error!(job_id = entry.id.0, "failed to build runtime: {e}");
            return;
        }
    };

    // The adapter's run loop is async and owns its own pull cadence. Route
    // every batch through filter+project, then into the sink. Errors from
    // the callback bubble back out through the adapter's return value.
    let sink_arc: Arc<RunnerSink> = Arc::new(sink);
    let sink_for_cb = Arc::clone(&sink_arc);
    let spec_for_cb = spec.clone();
    let catalog_for_cb = Arc::clone(&catalog);
    let entry_id = entry.id;

    // The callback pushes filtered batches into the sink using a fresh
    // per-callback current-thread runtime. block_on on the adapter's runtime
    // would deadlock since the adapter itself is driven on that runtime.
    let on_batch: Box<dyn Fn(Vec<CdfChange>) -> Result<()> + Send + Sync> =
        Box::new(move |records: Vec<CdfChange>| -> Result<()> {
            if records.is_empty() {
                return Ok(());
            }
            let filtered = apply_filter_project(&records, &spec_for_cb)?;
            if filtered.is_empty() {
                return Ok(());
            }
            // Use a local runtime here. The adapter's async context already holds
            // one runtime, and calling block_on from inside a runtime panics.
            let local_rt = tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()
                .map_err(|e| {
                    ZyronError::StreamingError(format!(
                        "remote-source callback runtime build failed: {e}"
                    ))
                })?;
            local_rt.block_on(async { sink_for_cb.write_batch(filtered).await })?;
            let _ = &catalog_for_cb;
            let _ = entry_id;
            Ok(())
        });

    let res = rt.block_on(async { source.run(start_lsn, on_batch, stop_flag).await });
    if let Err(e) = res {
        mark_failed(&rt, &catalog, entry.id, format!("remote source: {e}"));
    }
    let _ = &sink_arc;
}
