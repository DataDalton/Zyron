// -----------------------------------------------------------------------------
// Aggregating streaming-job runner
// -----------------------------------------------------------------------------
//
// Processes CDF batches into per (window, key) accumulator state, advances a
// watermark on every event, and emits finalized rows when the watermark has
// closed a window. Emitted row shape is:
//
//   [group-by columns..., aggregate outputs..., window_start_us, window_end_us]
//
// window_start_us and window_end_us are microsecond timestamps suitable for
// direct storage in a TIMESTAMP column.
//
// The module is split into a pure AggregateEngine that owns all window state
// and a thread-driven run_aggregating_loop that binds the engine to a Zyron
// CDF source and a RunnerSink. Unit tests drive AggregateEngine directly.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;

use zyron_catalog::schema::StreamingJobStatus;
use zyron_catalog::{Catalog, StreamingJobEntry, StreamingJobId};
use zyron_common::{Result, TypeId, ZyronError};

use crate::accumulator::{WindowAccumulator, get_accumulator};
use crate::job_runner::{AggWindowType, AggregateItem, AggregateSpec, RunnerSink};
use crate::late_data::LateDataPolicy;
use crate::row_codec::{StreamValue, decode_row, encode_row};
use crate::source_connector::{CdfChange, ZyronTableSource};
use crate::watermark::WatermarkGenerator;
use crate::window::{
    SessionAssigner, SlidingWindowAssigner, TumblingWindowAssigner, WindowAssigner, WindowRange,
};
use crate::window_state::WindowStateStore;

// -----------------------------------------------------------------------------
// WindowAssignerKind: runtime-dispatched window assigner
// -----------------------------------------------------------------------------

/// Holds exactly one of the three supported assigners. Sessions are per-key
/// stateful, the others are stateless.
enum WindowAssignerKind {
    Tumbling(TumblingWindowAssigner),
    Sliding(SlidingWindowAssigner),
    Session(SessionAssigner),
}

// -----------------------------------------------------------------------------
// AggregateEngine: pure state owner
// -----------------------------------------------------------------------------

/// Owns window state for one streaming aggregate job. Callers feed decoded
/// rows through process_row and drain emissions through drain_emissions. The
/// engine has no I/O so its behavior can be exercised in unit tests without
/// threads, sinks, or catalog.
pub struct AggregateEngine {
    spec: AggregateSpec,
    assigner: WindowAssignerKind,
    watermark: WatermarkGenerator,
    state: WindowStateStore,
    accumulators: Vec<Box<dyn WindowAccumulator>>,
    /// Output-row type layout: group-by columns, then each aggregate's finalized
    /// type, then two timestamp columns for window_start_us and window_end_us.
    output_types: Vec<TypeId>,
    /// Source-column types used to decode incoming rows.
    source_types: Vec<TypeId>,
    /// Source column types corresponding to each group-by ordinal, captured
    /// for key encoding without re-reading the full source_types list.
    group_key_types: Vec<TypeId>,
}

impl AggregateEngine {
    /// Builds a new engine from the spec and the source/target type layouts
    /// the runner already holds. Returns an error when any named aggregate
    /// cannot be resolved by get_accumulator.
    pub fn new(spec: AggregateSpec, source_types: Vec<TypeId>) -> Result<Self> {
        let assigner = match spec.window_type {
            AggWindowType::Tumbling { size_ms } => {
                WindowAssignerKind::Tumbling(TumblingWindowAssigner::new(size_ms))
            }
            AggWindowType::Hopping { size_ms, slide_ms } => {
                WindowAssignerKind::Sliding(SlidingWindowAssigner::new(size_ms, slide_ms))
            }
            AggWindowType::Session { gap_ms } => {
                WindowAssignerKind::Session(SessionAssigner::new(gap_ms))
            }
        };
        let mut accumulators: Vec<Box<dyn WindowAccumulator>> =
            Vec::with_capacity(spec.aggregations.len());
        for item in &spec.aggregations {
            let acc = get_accumulator(&item.function, item.input_type).ok_or_else(|| {
                ZyronError::StreamingError(format!(
                    "unknown streaming aggregate function {} on type {:?}",
                    item.function, item.input_type
                ))
            })?;
            accumulators.push(acc);
        }
        let mut group_key_types = Vec::with_capacity(spec.group_by_ordinals.len());
        for o in &spec.group_by_ordinals {
            let t = source_types.get(*o as usize).copied().ok_or_else(|| {
                ZyronError::StreamingError(format!("group-by ordinal {} out of range", o))
            })?;
            group_key_types.push(t);
        }
        let mut output_types =
            Vec::with_capacity(spec.group_by_ordinals.len() + accumulators.len() + 2);
        for t in &group_key_types {
            output_types.push(*t);
        }
        for acc in &accumulators {
            output_types.push(acc.output_type());
        }
        output_types.push(TypeId::Timestamp);
        output_types.push(TypeId::Timestamp);

        Ok(Self {
            watermark: WatermarkGenerator::new(spec.watermark),
            assigner,
            state: WindowStateStore::new(),
            accumulators,
            spec,
            output_types,
            source_types,
            group_key_types,
        })
    }

    /// Returns the output-row type layout, useful for sinks that need to
    /// re-encode emitted rows.
    pub fn output_types(&self) -> &[TypeId] {
        &self.output_types
    }

    /// Folds one decoded row into the window state. Advances the watermark
    /// and applies the late-data policy for rows whose target windows are
    /// already closed.
    pub fn process_row(&self, row: &[StreamValue]) -> Result<()> {
        // Extract raw event time and convert to microseconds.
        let ev_ord = self.spec.event_time_ordinal as usize;
        let raw = row
            .get(ev_ord)
            .ok_or_else(|| {
                ZyronError::StreamingError(format!("event-time ordinal {} out of range", ev_ord))
            })?
            .as_i64()?;
        let event_us = self.spec.event_time_scale.to_micros(raw);
        // Observe sets the watermark monotonically. The returned value is the
        // watermark just after this event.
        let wm_us = self.watermark.observe(event_us);
        // Window assignment uses milliseconds internally.
        let event_ms = event_us / 1_000;
        let wm_ms = wm_us / 1_000;

        // Encode the group-by key once per row.
        let key = self.encode_group_key(row)?;

        // Gather the windows this event lands in.
        let windows = self.assign_windows(&key, event_ms);

        for win in windows {
            let window_closed = win.end_ms <= wm_ms;
            if window_closed {
                match self.spec.late_data_policy {
                    LateDataPolicy::Drop | LateDataPolicy::SideOutput => continue,
                    LateDataPolicy::Update | LateDataPolicy::ReopenWindow => {
                        // Fold into state, the next drain may re-emit this
                        // window depending on whether it is still tracked.
                    }
                }
            }
            // Fold every aggregate over this (window, key).
            self.state.update(win, &key, |existing| {
                let mut combined = combine_states(existing, &self.accumulators);
                for (i, item) in self.spec.aggregations.iter().enumerate() {
                    let value = resolve_agg_input(row, item);
                    // update can fail for type mismatches. The loop body is
                    // fallible but update signature returns Vec<u8> directly,
                    // so errors are captured as a sentinel state of length 0.
                    let acc = &self.accumulators[i];
                    let slice = extract_state_slice(&combined, i, &self.accumulators);
                    let mut buf = slice.to_vec();
                    if let Err(_e) = acc.update(&mut buf, &value) {
                        // On update failure, leave state unchanged for this
                        // aggregate. The engine does not surface per-row
                        // errors to the sink, to keep streaming resilient.
                    }
                    write_state_slice(&mut combined, i, buf, &self.accumulators);
                }
                combined
            });
        }
        Ok(())
    }

    /// Returns and consumes all windows closed by the current watermark. Each
    /// emitted row is the fully encoded NSM bytes that match output_types.
    pub fn drain_emissions(&self) -> Result<Vec<Vec<u8>>> {
        let wm_us = self.watermark.current();
        if wm_us == i64::MIN {
            return Ok(Vec::new());
        }
        let wm_ms = wm_us / 1_000;

        // For session assigners, sessions live in two places: the
        // SessionAssigner and the WindowStateStore under the matching window.
        // drain_closed on the assigner is what closes sessions per key. The
        // WindowStateStore drain then emits those windows.
        if let WindowAssignerKind::Session(sess) = &self.assigner {
            let _ = sess.drain_closed(wm_ms);
        }

        let drained = self.state.drain_closed(wm_ms);
        let mut out = Vec::with_capacity(drained.len());
        for (window, key, state_bytes) in drained {
            let row = self.finalize_row(window, &key, &state_bytes)?;
            let encoded = encode_row(&row, &self.output_types)?;
            out.push(encoded);
        }
        Ok(out)
    }

    // ---- private helpers ----

    fn assign_windows(&self, key: &[u8], event_ms: i64) -> Vec<WindowRange> {
        match &self.assigner {
            WindowAssignerKind::Tumbling(a) => a.assign_windows(event_ms),
            WindowAssignerKind::Sliding(a) => a.assign_windows(event_ms),
            WindowAssignerKind::Session(a) => vec![a.assign(key, event_ms)],
        }
    }

    fn encode_group_key(&self, row: &[StreamValue]) -> Result<Vec<u8>> {
        let mut values = Vec::with_capacity(self.spec.group_by_ordinals.len());
        for o in &self.spec.group_by_ordinals {
            let v = row.get(*o as usize).ok_or_else(|| {
                ZyronError::StreamingError(format!("group-by ordinal {} out of range", o))
            })?;
            values.push(v.clone());
        }
        // Reuse the NSM row codec for deterministic byte-level grouping.
        encode_row(&values, &self.group_key_types)
    }

    fn decode_group_key(&self, key: &[u8]) -> Result<Vec<StreamValue>> {
        decode_row(key, &self.group_key_types)
    }

    fn finalize_row(
        &self,
        window: WindowRange,
        key: &[u8],
        state_bytes: &[u8],
    ) -> Result<Vec<StreamValue>> {
        let group_values = self.decode_group_key(key)?;
        let mut out = Vec::with_capacity(self.output_types.len());
        for v in group_values {
            out.push(v);
        }
        for (i, acc) in self.accumulators.iter().enumerate() {
            let slice = extract_state_slice(state_bytes, i, &self.accumulators);
            out.push(acc.finalize(slice)?);
        }
        // Convert milliseconds back to microseconds for the timestamp columns.
        out.push(StreamValue::I64(window.start_ms.saturating_mul(1_000)));
        out.push(StreamValue::I64(window.end_ms.saturating_mul(1_000)));
        Ok(out)
    }
}

fn resolve_agg_input(row: &[StreamValue], item: &AggregateItem) -> StreamValue {
    match item.input_ordinal {
        Some(o) => row.get(o as usize).cloned().unwrap_or(StreamValue::Null),
        None => StreamValue::Null,
    }
}

/// Computes the byte ranges for each accumulator's sub-state inside the
/// combined byte blob. Each sub-state is prefixed by a 4-byte little-endian
/// length so the layout stays self-describing across variable-length
/// accumulators like FIRST and LAST.
fn combine_states(existing: Option<&[u8]>, accs: &[Box<dyn WindowAccumulator>]) -> Vec<u8> {
    if let Some(bytes) = existing {
        return bytes.to_vec();
    }
    // Fresh layout: [len_0][state_0][len_1][state_1]...
    let mut out = Vec::new();
    for acc in accs {
        let init = acc.init();
        out.extend_from_slice(&(init.len() as u32).to_le_bytes());
        out.extend_from_slice(&init);
    }
    out
}

fn extract_state_slice<'a>(
    combined: &'a [u8],
    idx: usize,
    accs: &[Box<dyn WindowAccumulator>],
) -> &'a [u8] {
    let mut off = 0usize;
    for i in 0..accs.len() {
        if combined.len() < off + 4 {
            return &[];
        }
        let len = u32::from_le_bytes(combined[off..off + 4].try_into().unwrap()) as usize;
        off += 4;
        if combined.len() < off + len {
            return &[];
        }
        if i == idx {
            return &combined[off..off + len];
        }
        off += len;
    }
    &[]
}

fn write_state_slice(
    combined: &mut Vec<u8>,
    idx: usize,
    new_bytes: Vec<u8>,
    accs: &[Box<dyn WindowAccumulator>],
) {
    let mut rebuilt = Vec::with_capacity(combined.len() + 8);
    let mut off = 0usize;
    for i in 0..accs.len() {
        if combined.len() < off + 4 {
            return;
        }
        let len = u32::from_le_bytes(combined[off..off + 4].try_into().unwrap()) as usize;
        off += 4;
        let slice = if combined.len() >= off + len {
            &combined[off..off + len]
        } else {
            &[][..]
        };
        let replacement: &[u8] = if i == idx { &new_bytes } else { slice };
        rebuilt.extend_from_slice(&(replacement.len() as u32).to_le_bytes());
        rebuilt.extend_from_slice(replacement);
        off += len;
    }
    *combined = rebuilt;
}

// -----------------------------------------------------------------------------
// run_aggregating_loop: binds the engine to a Zyron CDF source and sink
// -----------------------------------------------------------------------------

const RUNNER_BATCH: usize = 1024;
const RUNNER_IDLE_MS: u64 = 100;

/// Drives the aggregating streaming-job pipeline. Matches the run_loop shape
/// used by the non-aggregating runner: poll CDF, fold, drain emissions, write
/// to sink. Exits when stop_flag is set or a terminal catalog status arrives.
pub fn run_aggregating_loop(
    entry: StreamingJobEntry,
    source_types: Vec<TypeId>,
    spec: AggregateSpec,
    source: ZyronTableSource,
    sink: RunnerSink,
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

    let engine = match AggregateEngine::new(spec, source_types.clone()) {
        Ok(e) => e,
        Err(e) => {
            mark_failed(
                &rt,
                &catalog,
                entry.id,
                format!("aggregate engine init failed: {e}"),
            );
            return;
        }
    };

    loop {
        if stop_flag.load(Ordering::Acquire) {
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
            std::thread::sleep(Duration::from_millis(RUNNER_IDLE_MS));
            continue;
        }

        // Fold each source row into the engine. Decoding errors are fatal.
        for rec in &records {
            let row = match decode_row(&rec.row_data, &source_types) {
                Ok(r) => r,
                Err(e) => {
                    mark_failed(&rt, &catalog, entry.id, format!("decode error: {e}"));
                    return;
                }
            };
            if let Err(e) = engine.process_row(&row) {
                mark_failed(&rt, &catalog, entry.id, format!("engine error: {e}"));
                return;
            }
        }

        // Drain whatever windows the watermark has closed and hand them to
        // the sink wrapped as synthetic CdfChange inserts.
        let emissions = match engine.drain_emissions() {
            Ok(v) => v,
            Err(e) => {
                mark_failed(&rt, &catalog, entry.id, format!("emit error: {e}"));
                break;
            }
        };
        if !emissions.is_empty() {
            let changes: Vec<CdfChange> = emissions
                .into_iter()
                .map(|bytes| CdfChange {
                    commit_version: 0,
                    commit_timestamp: 0,
                    change_type: zyron_cdc::ChangeType::Insert,
                    row_data: bytes,
                    primary_key_data: Vec::new(),
                })
                .collect();
            if let Err(e) = rt.block_on(async { sink.write_batch(changes).await }) {
                mark_failed(&rt, &catalog, entry.id, format!("sink error: {e}"));
                break;
            }
        }
    }
}

fn mark_failed(
    rt: &tokio::runtime::Runtime,
    catalog: &Arc<Catalog>,
    id: StreamingJobId,
    reason: String,
) {
    tracing::error!(job_id = id.0, reason = %reason, "aggregating streaming job failed");
    let _ = rt.block_on(async {
        catalog
            .update_streaming_job_status(id, StreamingJobStatus::Failed, Some(reason))
            .await
    });
}

// -----------------------------------------------------------------------------
// Tests
// -----------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::late_data::LateDataPolicy;
    use crate::watermark::WatermarkStrategy;

    fn tumbling_count_spec() -> AggregateSpec {
        AggregateSpec {
            window_type: AggWindowType::Tumbling { size_ms: 10 },
            event_time_ordinal: 0,
            event_time_scale: crate::job_runner::EventTimeScale::Milliseconds,
            group_by_ordinals: vec![1],
            aggregations: vec![AggregateItem {
                function: "COUNT".to_string(),
                input_ordinal: None,
                input_type: TypeId::Null,
            }],
            watermark: WatermarkStrategy::Punctual,
            late_data_policy: LateDataPolicy::Drop,
        }
    }

    fn row(event_time_ms: i64, key: i64) -> Vec<StreamValue> {
        vec![StreamValue::I64(event_time_ms), StreamValue::I64(key)]
    }

    fn decode_emission(bytes: &[u8], types: &[TypeId]) -> Vec<StreamValue> {
        decode_row(bytes, types).expect("decode emission")
    }

    #[test]
    fn test_tumbling_count_per_key() {
        let spec = tumbling_count_spec();
        let engine = AggregateEngine::new(spec, vec![TypeId::Int64, TypeId::Int64]).unwrap();
        // Window 1 [0,10) with keys 1 twice and 2 once.
        engine.process_row(&row(0, 1)).unwrap();
        engine.process_row(&row(5, 1)).unwrap();
        engine.process_row(&row(3, 2)).unwrap();
        // Window 2 [10,20) with key 1 once. Advancing to this window closes
        // the first window under a Punctual watermark.
        engine.process_row(&row(12, 1)).unwrap();
        let types = engine.output_types().to_vec();
        let emissions = engine.drain_emissions().unwrap();
        // Two groups in window 1 should be emitted. Window 2 stays open.
        assert_eq!(emissions.len(), 2);
        let mut counts: Vec<(i64, i64)> = emissions
            .iter()
            .map(|b| {
                let r = decode_emission(b, &types);
                let key = r[0].as_i64().unwrap();
                let count = r[1].as_i64().unwrap();
                (key, count)
            })
            .collect();
        counts.sort();
        assert_eq!(counts, vec![(1, 2), (2, 1)]);
    }

    #[test]
    fn test_hopping_sum() {
        // Size 10ms, slide 5ms. Event at t=6 lands in windows [0,10) and [5,15).
        let spec = AggregateSpec {
            window_type: AggWindowType::Hopping {
                size_ms: 10,
                slide_ms: 5,
            },
            event_time_ordinal: 0,
            event_time_scale: crate::job_runner::EventTimeScale::Milliseconds,
            group_by_ordinals: vec![1],
            aggregations: vec![AggregateItem {
                function: "SUM".to_string(),
                input_ordinal: Some(2),
                input_type: TypeId::Int64,
            }],
            watermark: WatermarkStrategy::Punctual,
            late_data_policy: LateDataPolicy::Drop,
        };
        let engine =
            AggregateEngine::new(spec, vec![TypeId::Int64, TypeId::Int64, TypeId::Int64]).unwrap();
        let row = |t: i64, k: i64, v: i64| {
            vec![
                StreamValue::I64(t),
                StreamValue::I64(k),
                StreamValue::I64(v),
            ]
        };
        engine.process_row(&row(6, 1, 100)).unwrap();
        engine.process_row(&row(7, 1, 200)).unwrap();
        // Advance watermark past window [0,10) with an event at t=20.
        engine.process_row(&row(20, 1, 1)).unwrap();
        let emissions = engine.drain_emissions().unwrap();
        // Window [0,10) sums to 300, window [5,15) sums to 300, window [10,20)
        // has no rows from key=1 (except the event that triggered the advance
        // is at t=20 which belongs to [10,20) and [15,25)). The two closed
        // windows are emitted.
        assert!(emissions.len() >= 2);
    }

    #[test]
    fn test_session_accumulator() {
        let spec = AggregateSpec {
            window_type: AggWindowType::Session { gap_ms: 50 },
            event_time_ordinal: 0,
            event_time_scale: crate::job_runner::EventTimeScale::Milliseconds,
            group_by_ordinals: vec![1],
            aggregations: vec![AggregateItem {
                function: "COUNT".to_string(),
                input_ordinal: None,
                input_type: TypeId::Null,
            }],
            watermark: WatermarkStrategy::Punctual,
            late_data_policy: LateDataPolicy::Drop,
        };
        let engine = AggregateEngine::new(spec, vec![TypeId::Int64, TypeId::Int64]).unwrap();
        // Key 1: two events close together form one session, plus a later
        // event that starts a new session after the gap.
        engine.process_row(&row(100, 1)).unwrap();
        engine.process_row(&row(130, 1)).unwrap();
        // Advance past [100, 180) session end by emitting from a key 2 event
        // at t=300 which causes a watermark advance.
        engine.process_row(&row(300, 2)).unwrap();
        let emissions = engine.drain_emissions().unwrap();
        assert!(!emissions.is_empty());
    }

    #[test]
    fn test_late_data_drop() {
        let spec = tumbling_count_spec();
        let engine = AggregateEngine::new(spec, vec![TypeId::Int64, TypeId::Int64]).unwrap();
        // Seed a row into window [0,10). Then an event far in the future
        // moves the watermark past 10. A late row then lands back in [0,10)
        // and should be dropped since the window is already closed.
        engine.process_row(&row(3, 1)).unwrap();
        engine.process_row(&row(25, 2)).unwrap();
        engine.process_row(&row(2, 1)).unwrap();
        let types = engine.output_types().to_vec();
        let emissions = engine.drain_emissions().unwrap();
        // Only the initial row counted for window [0,10). Late row dropped.
        assert!(!emissions.is_empty());
        let r0 = decode_emission(&emissions[0], &types);
        // The count for key 1 in window [0,10) is 1, not 2.
        assert_eq!(r0[1].as_i64().unwrap(), 1);
    }

    #[test]
    fn test_watermark_triggers_emission() {
        let spec = tumbling_count_spec();
        let engine = AggregateEngine::new(spec, vec![TypeId::Int64, TypeId::Int64]).unwrap();
        engine.process_row(&row(1, 1)).unwrap();
        engine.process_row(&row(3, 1)).unwrap();
        // No event has advanced the watermark past 10 yet.
        let early = engine.drain_emissions().unwrap();
        assert!(early.is_empty(), "no emission before watermark advance");
        engine.process_row(&row(11, 2)).unwrap();
        let later = engine.drain_emissions().unwrap();
        assert_eq!(later.len(), 1);
    }
}
