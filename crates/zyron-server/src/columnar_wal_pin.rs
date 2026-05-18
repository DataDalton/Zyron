//! WAL retention pin for the columnar registry.
//!
//! A fold's registry entry is made durable only every
//! `registry_persist_every` segments (the rest update the catalog cache
//! only, O(1) per fold). Between durable persists the segment is
//! reconstructed at startup from the WAL CompactionEnd record, so that
//! record must not be reclaimed by a checkpoint until the registry is
//! durable. A checkpoint advances independently of the fold cadence, so
//! without a pin it can delete WAL segments holding CompactionEnd records
//! for not-yet-persisted folds; recovery would then lose those segments and
//! the next fold could reuse the file id and overwrite a live .zyr.
//!
//! This tracks, per table, the LSN of the oldest CompactionEnd not yet
//! covered by a durable registry persist. The WAL retention hook returns the
//! minimum across tables, so `cleanup_old_segments` never drops a segment a
//! recovery would still need. A durable persist releases the table's pin.

use std::collections::HashMap;
use std::sync::{Mutex, OnceLock};

/// Process-global pin. One database instance per process, mirroring
/// `ColumnarPatchManager::global`.
pub struct ColumnarWalPin {
    /// table_id -> LSN of the oldest CompactionEnd not yet durably persisted.
    pins: Mutex<HashMap<u32, u64>>,
}

static GLOBAL: OnceLock<ColumnarWalPin> = OnceLock::new();

impl ColumnarWalPin {
    pub fn global() -> &'static ColumnarWalPin {
        GLOBAL.get_or_init(|| ColumnarWalPin {
            pins: Mutex::new(HashMap::new()),
        })
    }

    /// Records that `lsn` is a CompactionEnd for `table_id` whose registry
    /// entry is cache-only (not yet durable). Keeps the oldest such LSN; a
    /// later cache-only fold does not move the pin forward.
    pub fn note(&self, table_id: u32, lsn: u64) {
        let mut g = self.pins.lock().unwrap_or_else(|e| e.into_inner());
        g.entry(table_id).or_insert(lsn);
    }

    /// Releases the pin for `table_id` after a durable registry persist: every
    /// prior CompactionEnd for the table is now reflected in durable storage.
    pub fn release(&self, table_id: u32) {
        let mut g = self.pins.lock().unwrap_or_else(|e| e.into_inner());
        g.remove(&table_id);
    }

    /// The minimum pinned LSN across all tables, or None when nothing is
    /// pinned (the checkpoint may then reclaim up to its own LSN).
    pub fn min_retained(&self) -> Option<u64> {
        let g = self.pins.lock().unwrap_or_else(|e| e.into_inner());
        g.values().copied().min()
    }
}
