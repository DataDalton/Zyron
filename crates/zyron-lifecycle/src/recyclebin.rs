//! Recycle bin: a configurable grace window during which DELETE / DROP TABLE
//! / TRUNCATE remain recoverable (via WAL + versioning time travel). After
//! the window the purge worker finalizes the removal. Erasure/legal-hold
//! removals intentionally bypass the recycle window.

use zyron_catalog::schema::TableEntry;

use crate::ttl::now_micros;

/// Recycle window in seconds for a table (0 = disabled, immediate purge).
pub fn recycle_window_seconds(entry: &TableEntry) -> i64 {
    entry.lifecycle.recycle_window_seconds.max(0)
}

/// Whether a row/table soft-removed at `removed_at_us` is still within the
/// recoverable window as of now.
pub fn within_window(entry: &TableEntry, removed_at_us: i64) -> bool {
    let w = recycle_window_seconds(entry);
    if w == 0 {
        return false;
    }
    removed_at_us + w.saturating_mul(1_000_000) > now_micros()
}

/// Whether a removal whose grace started at `removed_at_us` may now be
/// physically finalized (window elapsed). Erasure forces this true regardless.
pub fn finalizable(entry: &TableEntry, removed_at_us: i64, forced_by_erasure: bool) -> bool {
    if forced_by_erasure {
        return true;
    }
    !within_window(entry, removed_at_us)
}
