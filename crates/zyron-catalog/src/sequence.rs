//! Live sequence manager.
//!
//! Each sequence has one `LiveSequence` holding the immutable definition and a
//! small in-memory block cursor. `nextval` hands out values from the cached
//! block without touching storage. When the block is exhausted it reserves the
//! next block by bumping the durable `reserved` high-water and persisting the
//! sequence entry, so a crash skips at most `cache` values rather than reusing
//! any. Refills are serialized by an async gate so the persisted `reserved`
//! advances monotonically.

use crate::ids::SchemaId;
use crate::schema::SequenceEntry;
use parking_lot::Mutex;
use zyron_common::{Result, ZyronError};

/// In-memory block cursor. `last` is the most recent value handed out.
/// `cached_until` is the highest (for ascending) or lowest (for descending)
/// value that may be returned without reserving a new block; it mirrors the
/// durable `reserved`.
#[derive(Debug, Clone, Copy)]
struct SeqCursor {
    last: i64,
    cached_until: i64,
}

/// The window installed by a refill: the new baseline and the new durable
/// high-water. `new_last` is set as the cursor baseline so the first value
/// returned after install is `new_last + increment`.
struct RefillPlan {
    new_last: i64,
    new_cached_until: i64,
}

/// Live state for one sequence. Parameters are immutable for the life of the
/// object; ALTER SEQUENCE replaces the whole `LiveSequence`.
pub struct LiveSequence {
    pub id: u32,
    pub schema_id: u32,
    pub name: String,
    pub increment: i64,
    pub min_value: i64,
    pub max_value: i64,
    pub start: i64,
    pub cache: i64,
    pub cycle: bool,
    cursor: Mutex<SeqCursor>,
    /// Serializes refills so the persisted `reserved` advances in order. Held
    /// only on the slow refill path, not per value.
    refill_gate: tokio::sync::Mutex<()>,
}

impl LiveSequence {
    /// Builds a live sequence from a persisted entry. The cursor starts
    /// exhausted at `reserved`, so the first `nextval` reserves a block. With
    /// a freshly created entry (`reserved == start - increment`) that block
    /// begins at `start`.
    pub fn from_entry(entry: &SequenceEntry) -> Self {
        Self {
            id: entry.id,
            schema_id: entry.schema_id.0,
            name: entry.name.clone(),
            increment: entry.increment,
            min_value: entry.min_value,
            max_value: entry.max_value,
            start: entry.start,
            cache: entry.cache.max(1),
            cycle: entry.cycle,
            cursor: Mutex::new(SeqCursor {
                last: entry.reserved,
                cached_until: entry.reserved,
            }),
            refill_gate: tokio::sync::Mutex::new(()),
        }
    }

    /// Serializes refills. The caller holds this across the persist of a
    /// reserved block so storage writes do not reorder.
    pub async fn lock_refill(&self) -> tokio::sync::MutexGuard<'_, ()> {
        self.refill_gate.lock().await
    }

    /// Hands out the next value from the cached block. Returns `None` when the
    /// block is exhausted and a refill is required.
    pub fn try_next(&self) -> Option<i64> {
        let mut c = self.cursor.lock();
        let cand = c.last.checked_add(self.increment)?;
        let within = if self.increment > 0 {
            cand <= c.cached_until
        } else {
            cand >= c.cached_until
        };
        if within {
            c.last = cand;
            Some(cand)
        } else {
            None
        }
    }

    /// Computes the next block to reserve from the current cursor state.
    /// Returns the entry to persist (with the bumped `reserved`) and the plan
    /// to install once the persist succeeds. Errors when an ascending sequence
    /// is at its maximum (or descending at its minimum) and `cycle` is off.
    pub fn plan_refill(&self) -> Result<(SequenceEntry, RefillSlot)> {
        let cur = *self.cursor.lock();
        let ascending = self.increment > 0;
        let reserved = cur.cached_until;

        let first = reserved.checked_add(self.increment);
        let exhausted = match first {
            None => true,
            Some(f) => {
                if ascending {
                    f > self.max_value
                } else {
                    f < self.min_value
                }
            }
        };

        let plan = if exhausted {
            if !self.cycle {
                let bound = if ascending { "maximum" } else { "minimum" };
                return Err(ZyronError::Internal(format!(
                    "nextval: sequence '{}' reached its {bound} value",
                    self.name
                )));
            }
            let wrap_start = if ascending {
                self.min_value
            } else {
                self.max_value
            };
            let new_last = wrap_start.saturating_sub(self.increment);
            let top = self.block_top(new_last);
            RefillPlan {
                new_last,
                new_cached_until: top,
            }
        } else {
            let top = self.block_top(reserved);
            RefillPlan {
                new_last: reserved,
                new_cached_until: top,
            }
        };

        let entry = self.to_entry(plan.new_cached_until);
        Ok((entry, RefillSlot { plan }))
    }

    /// Installs the cursor window from a plan without handing out a value.
    /// Used by setval, which positions the sequence but does not consume.
    pub fn install_window(&self, slot: RefillSlot) {
        let mut c = self.cursor.lock();
        c.last = slot.plan.new_last;
        c.cached_until = slot.plan.new_cached_until;
    }

    /// Installs a reserved block after its entry is durably persisted and
    /// returns the first value of the block.
    pub fn install_refill(&self, slot: RefillSlot) -> Result<i64> {
        self.install_window(slot);
        self.try_next().ok_or_else(|| {
            ZyronError::Internal(format!(
                "sequence '{}' produced an empty reserved block",
                self.name
            ))
        })
    }

    /// The durable high-water currently reserved. ALTER SEQUENCE reads this to
    /// preserve position when no RESTART is requested.
    pub fn current_reserved(&self) -> i64 {
        self.cursor.lock().cached_until
    }

    /// Builds the entry to persist for setval and the cursor state to install.
    /// `is_called` true makes `value` the current value (next is `value +
    /// increment`); false makes the next `nextval` return `value`.
    pub fn plan_setval(&self, value: i64, is_called: bool) -> Result<(SequenceEntry, RefillSlot)> {
        if value < self.min_value || value > self.max_value {
            return Err(ZyronError::Internal(format!(
                "setval: value {value} is out of range for sequence '{}'",
                self.name
            )));
        }
        let new_last = if is_called {
            value
        } else {
            value.saturating_sub(self.increment)
        };
        let plan = RefillPlan {
            new_last,
            new_cached_until: value,
        };
        let entry = self.to_entry(value);
        Ok((entry, RefillSlot { plan }))
    }

    /// Highest value (ascending) or lowest (descending) reachable in one block
    /// starting from `baseline`, clamped to the sequence bounds.
    fn block_top(&self, baseline: i64) -> i64 {
        let span = self.increment.saturating_mul(self.cache);
        let top = baseline.saturating_add(span);
        if self.increment > 0 {
            top.min(self.max_value)
        } else {
            top.max(self.min_value)
        }
    }

    /// Materializes the durable entry with the given `reserved` high-water.
    pub fn to_entry(&self, reserved: i64) -> SequenceEntry {
        SequenceEntry {
            id: self.id,
            schema_id: SchemaId(self.schema_id),
            name: self.name.clone(),
            increment: self.increment,
            min_value: self.min_value,
            max_value: self.max_value,
            start: self.start,
            cache: self.cache,
            cycle: self.cycle,
            reserved,
        }
    }
}

/// Opaque handle carrying the planned cursor window from `plan_refill` /
/// `plan_setval` to `install_refill`, so the install cannot drift from the
/// persisted entry.
pub struct RefillSlot {
    plan: RefillPlan,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn entry(
        increment: i64,
        min: i64,
        max: i64,
        start: i64,
        cache: i64,
        cycle: bool,
    ) -> SequenceEntry {
        SequenceEntry {
            id: 1,
            schema_id: SchemaId(2),
            name: "s".to_string(),
            increment,
            min_value: min,
            max_value: max,
            start,
            cache,
            cycle,
            reserved: start - increment,
        }
    }

    // Drives nextval through plan_refill + install_refill without storage.
    fn next(seq: &LiveSequence) -> Result<i64> {
        if let Some(v) = seq.try_next() {
            return Ok(v);
        }
        let (_persist, slot) = seq.plan_refill()?;
        seq.install_refill(slot)
    }

    #[test]
    fn first_value_is_start_then_increments() {
        let seq = LiveSequence::from_entry(&entry(1, 1, 1000, 1, 10, false));
        assert_eq!(next(&seq).unwrap(), 1);
        assert_eq!(next(&seq).unwrap(), 2);
        assert_eq!(next(&seq).unwrap(), 3);
    }

    #[test]
    fn block_cache_spans_multiple_values_then_refills() {
        let seq = LiveSequence::from_entry(&entry(1, 1, 1000, 1, 3, false));
        // First block reserves up to 3; persisted reserved tracks the top.
        let (e1, slot1) = {
            assert!(seq.try_next().is_none());
            seq.plan_refill().unwrap()
        };
        assert_eq!(e1.reserved, 3);
        assert_eq!(seq.install_refill(slot1).unwrap(), 1);
        assert_eq!(seq.try_next().unwrap(), 2);
        assert_eq!(seq.try_next().unwrap(), 3);
        assert!(seq.try_next().is_none());
    }

    #[test]
    fn ascending_exhaustion_without_cycle_errors() {
        let seq = LiveSequence::from_entry(&entry(1, 1, 2, 1, 5, false));
        assert_eq!(next(&seq).unwrap(), 1);
        assert_eq!(next(&seq).unwrap(), 2);
        assert!(next(&seq).is_err());
    }

    #[test]
    fn ascending_cycle_wraps_to_min() {
        let seq = LiveSequence::from_entry(&entry(1, 1, 3, 1, 2, true));
        assert_eq!(next(&seq).unwrap(), 1);
        assert_eq!(next(&seq).unwrap(), 2);
        assert_eq!(next(&seq).unwrap(), 3);
        assert_eq!(next(&seq).unwrap(), 1);
    }

    #[test]
    fn descending_counts_down_and_cycles() {
        let seq = LiveSequence::from_entry(&entry(-1, 1, 3, 3, 2, true));
        assert_eq!(next(&seq).unwrap(), 3);
        assert_eq!(next(&seq).unwrap(), 2);
        assert_eq!(next(&seq).unwrap(), 1);
        assert_eq!(next(&seq).unwrap(), 3);
    }

    #[test]
    fn setval_called_sets_current_value() {
        let seq = LiveSequence::from_entry(&entry(1, 1, 1000, 1, 10, false));
        let (_e, slot) = seq.plan_setval(50, true).unwrap();
        seq.install_window(slot);
        assert_eq!(next(&seq).unwrap(), 51);
    }

    #[test]
    fn setval_not_called_returns_value_next() {
        let seq = LiveSequence::from_entry(&entry(1, 1, 1000, 1, 10, false));
        let (_e, slot) = seq.plan_setval(50, false).unwrap();
        seq.install_window(slot);
        assert_eq!(next(&seq).unwrap(), 50);
    }
}
