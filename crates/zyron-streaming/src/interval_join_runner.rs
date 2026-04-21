// -----------------------------------------------------------------------------
// Interval-join engine for streaming jobs
// -----------------------------------------------------------------------------
//
// Row-level interval join over decoded StreamValue rows. Supports multi-column
// equi-keys, Inner / Left / Right / Full semantics, and deterministic outer
// emission on watermark advance. The engine is pure state, no I/O, so the unit
// tests drive it directly.
//
// Input rows are tagged with their side (Left or Right) and an event time in
// microseconds. The engine buffers both sides keyed on a byte-encoded multi-key
// and emits combined output rows whenever a match is found. Unmatched rows
// whose event_time + within_us has fallen below the current watermark get
// flushed with the opposite side filled as Null, subject to the configured
// StreamingJoinKind.
//
// Output row shape: left columns in source order, followed by right columns in
// source order. Callers encode the output through row_codec::encode_row against
// the combined types list.

use std::collections::HashMap;

use zyron_common::{Result, TypeId, ZyronError};

use crate::job_runner::StreamingJoinKind;
use crate::row_codec::{encode_row, StreamValue};

// -----------------------------------------------------------------------------
// Side
// -----------------------------------------------------------------------------

/// Identifies which input stream a row came from.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum JoinSide {
    Left,
    Right,
}

// -----------------------------------------------------------------------------
// BufferedRow
// -----------------------------------------------------------------------------

/// Row held in per-side state while the engine waits for a potential match.
/// matched flips to true the first time the row pairs with an opposite row.
/// The outer-emission scan at watermark time uses matched to decide whether
/// the row should be padded with Nulls or simply dropped.
#[derive(Debug, Clone)]
struct BufferedRow {
    values: Vec<StreamValue>,
    event_us: i64,
    matched: bool,
}

// -----------------------------------------------------------------------------
// IntervalJoinEngine
// -----------------------------------------------------------------------------

/// Multi-key interval-join engine. The caller drives it by handing in rows
/// through feed_row and draining matches out of pop_emissions. Watermark
/// advances are announced through advance_watermark, which may append more
/// rows to the emission queue when outer semantics flush unmatched candidates.
pub struct IntervalJoinEngine {
    left_key_ordinals: Vec<u16>,
    right_key_ordinals: Vec<u16>,
    left_event_ordinal: u16,
    right_event_ordinal: u16,
    left_types: Vec<TypeId>,
    right_types: Vec<TypeId>,
    within_us: i64,
    join_kind: StreamingJoinKind,
    left_state: HashMap<Vec<u8>, Vec<BufferedRow>>,
    right_state: HashMap<Vec<u8>, Vec<BufferedRow>>,
    emissions: Vec<Vec<StreamValue>>,
    watermark_us: i64,
}

/// Configuration shape the runner passes in. Mirrors the subset of
/// IntervalJoinConfig the engine cares about. Kept separate so the engine
/// can be tested without dragging in a full runner spec.
#[derive(Debug, Clone)]
pub struct IntervalJoinEngineConfig {
    pub left_types: Vec<TypeId>,
    pub right_types: Vec<TypeId>,
    pub left_key_ordinals: Vec<u16>,
    pub right_key_ordinals: Vec<u16>,
    pub left_event_ordinal: u16,
    pub right_event_ordinal: u16,
    pub within_us: i64,
    pub join_kind: StreamingJoinKind,
}

impl IntervalJoinEngine {
    pub fn new(cfg: IntervalJoinEngineConfig) -> Self {
        Self {
            left_key_ordinals: cfg.left_key_ordinals,
            right_key_ordinals: cfg.right_key_ordinals,
            left_event_ordinal: cfg.left_event_ordinal,
            right_event_ordinal: cfg.right_event_ordinal,
            left_types: cfg.left_types,
            right_types: cfg.right_types,
            within_us: cfg.within_us,
            join_kind: cfg.join_kind,
            left_state: HashMap::new(),
            right_state: HashMap::new(),
            emissions: Vec::new(),
            watermark_us: i64::MIN,
        }
    }

    /// Output-row type layout: left types followed by right types.
    pub fn output_types(&self) -> Vec<TypeId> {
        let mut out = Vec::with_capacity(self.left_types.len() + self.right_types.len());
        out.extend(self.left_types.iter().copied());
        out.extend(self.right_types.iter().copied());
        out
    }

    /// Feeds a decoded row into the engine. Writes zero or more matched rows
    /// to the emission queue. Unmatched rows stay buffered until a matching
    /// row arrives on the opposite side or the watermark passes their
    /// event_time + within_us.
    pub fn feed_row(&mut self, side: JoinSide, values: Vec<StreamValue>) -> Result<()> {
        let (event_ord, key_ords, key_types) = match side {
            JoinSide::Left => (
                self.left_event_ordinal,
                self.left_key_ordinals.clone(),
                self.key_types_left(),
            ),
            JoinSide::Right => (
                self.right_event_ordinal,
                self.right_key_ordinals.clone(),
                self.key_types_right(),
            ),
        };
        let event_us = values
            .get(event_ord as usize)
            .ok_or_else(|| {
                ZyronError::StreamingError(format!(
                    "interval-join event-time ordinal {} out of range",
                    event_ord
                ))
            })?
            .as_i64()?;
        let key = encode_key(&values, &key_ords, &key_types)?;

        // Probe the opposite side for matches within the window.
        let matched_count = self.probe_and_emit(side, &key, event_us, &values);

        // Buffer this row. An inner join still buffers so a later opposite
        // row can match with this one.
        let row = BufferedRow {
            values,
            event_us,
            matched: matched_count > 0,
        };
        match side {
            JoinSide::Left => self.left_state.entry(key).or_default().push(row),
            JoinSide::Right => self.right_state.entry(key).or_default().push(row),
        }
        Ok(())
    }

    /// Advances the watermark in microseconds. Rows on either side whose
    /// event_time + within_us has fallen at or below the watermark become
    /// eligible for outer emission, then are evicted from state. Inner-only
    /// semantics evict without emitting.
    pub fn advance_watermark(&mut self, wm_us: i64) {
        if wm_us <= self.watermark_us {
            return;
        }
        self.watermark_us = wm_us;
        let cutoff = wm_us.saturating_sub(self.within_us);
        // Left side.
        let kinds = self.join_kind;
        let emit_left_nulls = matches!(kinds, StreamingJoinKind::Left | StreamingJoinKind::Full);
        let emit_right_nulls = matches!(kinds, StreamingJoinKind::Right | StreamingJoinKind::Full);
        let right_width = self.right_types.len();
        let left_width = self.left_types.len();

        let mut emissions = std::mem::take(&mut self.emissions);
        // Walk left buffer, flush and drop rows older than cutoff.
        self.left_state.retain(|_, rows| {
            rows.retain(|r| {
                if r.event_us <= cutoff {
                    if !r.matched && emit_left_nulls {
                        let mut combined =
                            Vec::with_capacity(r.values.len() + right_width);
                        combined.extend(r.values.iter().cloned());
                        for _ in 0..right_width {
                            combined.push(StreamValue::Null);
                        }
                        emissions.push(combined);
                    }
                    false
                } else {
                    true
                }
            });
            !rows.is_empty()
        });
        self.right_state.retain(|_, rows| {
            rows.retain(|r| {
                if r.event_us <= cutoff {
                    if !r.matched && emit_right_nulls {
                        let mut combined = Vec::with_capacity(left_width + r.values.len());
                        for _ in 0..left_width {
                            combined.push(StreamValue::Null);
                        }
                        combined.extend(r.values.iter().cloned());
                        emissions.push(combined);
                    }
                    false
                } else {
                    true
                }
            });
            !rows.is_empty()
        });
        self.emissions = emissions;
    }

    /// Drains all currently queued joined rows. Callers typically run this
    /// after every feed_row and after every advance_watermark.
    pub fn pop_emissions(&mut self) -> Vec<Vec<StreamValue>> {
        std::mem::take(&mut self.emissions)
    }

    /// Encodes every queued emission as a row-codec byte string against the
    /// combined type list. Convenience wrapper for runners that feed the sink.
    pub fn drain_encoded(&mut self) -> Result<Vec<Vec<u8>>> {
        let rows = self.pop_emissions();
        let types = self.output_types();
        let mut out = Vec::with_capacity(rows.len());
        for r in rows {
            out.push(encode_row(&r, &types)?);
        }
        Ok(out)
    }

    // ---- private helpers ----

    fn key_types_left(&self) -> Vec<TypeId> {
        self.left_key_ordinals
            .iter()
            .map(|o| self.left_types[*o as usize])
            .collect()
    }

    fn key_types_right(&self) -> Vec<TypeId> {
        self.right_key_ordinals
            .iter()
            .map(|o| self.right_types[*o as usize])
            .collect()
    }

    /// Probes the opposite-side buffer for rows that share the same key and
    /// fall within +/- within_us of the probe event. Emits one combined row
    /// per match and marks the buffered row as matched. Returns the number
    /// of emitted rows so the caller knows whether the probe row itself
    /// matched anything for outer bookkeeping.
    fn probe_and_emit(
        &mut self,
        side: JoinSide,
        key: &[u8],
        probe_us: i64,
        probe_values: &[StreamValue],
    ) -> usize {
        let mut emitted = 0usize;
        match side {
            JoinSide::Left => {
                if let Some(rows) = self.right_state.get_mut(key) {
                    for r in rows.iter_mut() {
                        if (probe_us - r.event_us).abs() <= self.within_us {
                            let mut combined =
                                Vec::with_capacity(probe_values.len() + r.values.len());
                            combined.extend(probe_values.iter().cloned());
                            combined.extend(r.values.iter().cloned());
                            self.emissions.push(combined);
                            r.matched = true;
                            emitted += 1;
                        }
                    }
                }
            }
            JoinSide::Right => {
                if let Some(rows) = self.left_state.get_mut(key) {
                    for r in rows.iter_mut() {
                        if (probe_us - r.event_us).abs() <= self.within_us {
                            let mut combined =
                                Vec::with_capacity(r.values.len() + probe_values.len());
                            combined.extend(r.values.iter().cloned());
                            combined.extend(probe_values.iter().cloned());
                            self.emissions.push(combined);
                            r.matched = true;
                            emitted += 1;
                        }
                    }
                }
            }
        }
        emitted
    }
}

/// Encodes the key columns from a row as a byte string using the row codec.
/// The deterministic byte layout makes HashMap lookups stable across sides.
fn encode_key(
    row: &[StreamValue],
    ordinals: &[u16],
    types: &[TypeId],
) -> Result<Vec<u8>> {
    let mut key_values = Vec::with_capacity(ordinals.len());
    for o in ordinals {
        let v = row.get(*o as usize).ok_or_else(|| {
            ZyronError::StreamingError(format!(
                "interval-join key ordinal {} out of range",
                o
            ))
        })?;
        key_values.push(v.clone());
    }
    encode_row(&key_values, types)
}

// -----------------------------------------------------------------------------
// Temporal-join helper: flush left with Nulls on lookup miss
// -----------------------------------------------------------------------------

/// Builds the null-right-padded combined row for a temporal Left-outer miss.
/// Callers pass the left row plus the combined width so the engine does not
/// need to retain the type list at hand for this one case.
pub fn temporal_left_null_right(
    left_row: &[StreamValue],
    right_width: usize,
) -> Vec<StreamValue> {
    let mut out = Vec::with_capacity(left_row.len() + right_width);
    out.extend(left_row.iter().cloned());
    for _ in 0..right_width {
        out.push(StreamValue::Null);
    }
    out
}

// -----------------------------------------------------------------------------
// Tests
// -----------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn engine_for(kind: StreamingJoinKind, within_us: i64) -> IntervalJoinEngine {
        IntervalJoinEngine::new(IntervalJoinEngineConfig {
            left_types: vec![TypeId::Int64, TypeId::Int64, TypeId::Int64],
            right_types: vec![TypeId::Int64, TypeId::Int64],
            left_key_ordinals: vec![0],
            right_key_ordinals: vec![0],
            left_event_ordinal: 1,
            right_event_ordinal: 1,
            within_us,
            join_kind: kind,
        })
    }

    fn row3(k: i64, t: i64, v: i64) -> Vec<StreamValue> {
        vec![StreamValue::I64(k), StreamValue::I64(t), StreamValue::I64(v)]
    }
    fn row2(k: i64, t: i64) -> Vec<StreamValue> {
        vec![StreamValue::I64(k), StreamValue::I64(t)]
    }

    #[test]
    fn test_inner_match_within_window() {
        let mut e = engine_for(StreamingJoinKind::Inner, 5_000);
        e.feed_row(JoinSide::Left, row3(1, 0, 10)).unwrap();
        e.feed_row(JoinSide::Right, row2(1, 2_000)).unwrap();
        let emissions = e.pop_emissions();
        assert_eq!(emissions.len(), 1);
    }

    #[test]
    fn test_inner_drops_outside_window() {
        let mut e = engine_for(StreamingJoinKind::Inner, 1_000);
        e.feed_row(JoinSide::Left, row3(1, 0, 10)).unwrap();
        e.feed_row(JoinSide::Right, row2(1, 5_000)).unwrap();
        assert!(e.pop_emissions().is_empty());
    }

    #[test]
    fn test_interval_left_outer_emits_nulls_on_watermark() {
        let mut e = engine_for(StreamingJoinKind::Left, 1_000);
        // Left row with no matching right row. After advancing the watermark
        // past 0 + 1_000, the left row must be emitted with right-side Nulls.
        e.feed_row(JoinSide::Left, row3(1, 0, 42)).unwrap();
        assert!(e.pop_emissions().is_empty());
        e.advance_watermark(5_000);
        let emissions = e.pop_emissions();
        assert_eq!(emissions.len(), 1);
        // Left columns preserved, right columns Null-filled.
        let combined = &emissions[0];
        assert_eq!(combined.len(), 5);
        assert!(matches!(combined[0], StreamValue::I64(1)));
        assert!(matches!(combined[3], StreamValue::Null));
        assert!(matches!(combined[4], StreamValue::Null));
    }

    #[test]
    fn test_interval_full_outer_both_sides() {
        let mut e = engine_for(StreamingJoinKind::Full, 1_000);
        // Unmatched left.
        e.feed_row(JoinSide::Left, row3(1, 0, 10)).unwrap();
        // Unmatched right with a different key.
        e.feed_row(JoinSide::Right, row2(2, 100)).unwrap();
        e.advance_watermark(5_000);
        let emissions = e.pop_emissions();
        assert_eq!(emissions.len(), 2, "both sides should flush unmatched");
    }

    #[test]
    fn test_multi_key_interval_join() {
        let mut e = IntervalJoinEngine::new(IntervalJoinEngineConfig {
            left_types: vec![TypeId::Int64, TypeId::Int64, TypeId::Int64],
            right_types: vec![TypeId::Int64, TypeId::Int64, TypeId::Int64],
            // Two-column equi-key over (col0, col1).
            left_key_ordinals: vec![0, 1],
            right_key_ordinals: vec![0, 1],
            left_event_ordinal: 2,
            right_event_ordinal: 2,
            within_us: 5_000,
            join_kind: StreamingJoinKind::Inner,
        });
        // Matching pair: same (key_a, key_b), different value cols.
        e.feed_row(
            JoinSide::Left,
            vec![StreamValue::I64(7), StreamValue::I64(9), StreamValue::I64(0)],
        )
        .unwrap();
        e.feed_row(
            JoinSide::Right,
            vec![StreamValue::I64(7), StreamValue::I64(9), StreamValue::I64(1_000)],
        )
        .unwrap();
        let emissions = e.pop_emissions();
        assert_eq!(emissions.len(), 1);
        // Non-matching second column must not emit.
        e.feed_row(
            JoinSide::Right,
            vec![StreamValue::I64(7), StreamValue::I64(42), StreamValue::I64(2_000)],
        )
        .unwrap();
        assert!(e.pop_emissions().is_empty());
    }

    #[test]
    fn test_temporal_left_outer_lookup_miss() {
        // Temporal Left-outer: on a missed lookup the runner emits the left
        // row with Nulls in the right-side slots. The helper builds exactly
        // that shape for the runner to encode and push into the sink.
        let left_row = vec![StreamValue::I64(42), StreamValue::I64(1_000)];
        let combined = temporal_left_null_right(&left_row, 3);
        assert_eq!(combined.len(), 5);
        assert!(matches!(combined[0], StreamValue::I64(42)));
        assert!(matches!(combined[2], StreamValue::Null));
        assert!(matches!(combined[3], StreamValue::Null));
        assert!(matches!(combined[4], StreamValue::Null));
    }

    #[test]
    fn test_self_join_matches_delta() {
        // Self-join is realized by feeding the same stream as both Left and
        // Right. The engine itself is side-agnostic, so matches come out
        // normally even when rows share a source.
        let mut e = engine_for(StreamingJoinKind::Inner, 60_000_000);
        e.feed_row(JoinSide::Left, row3(1, 0, 10)).unwrap();
        e.feed_row(JoinSide::Right, row2(1, 5_000)).unwrap();
        let emissions = e.pop_emissions();
        assert_eq!(emissions.len(), 1);
    }
}
