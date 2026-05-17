//! TTL expiry computation. The cutoff is `now - ttl`; per-row retention
//! compares a per-row column against `now` instead.

use std::time::{SystemTime, UNIX_EPOCH};

use zyron_catalog::schema::TableEntry;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TtlAction {
    Delete,
    Archive,
    Anonymize,
}

impl TtlAction {
    pub fn from_u8(v: u8) -> TtlAction {
        match v {
            1 => TtlAction::Archive,
            2 => TtlAction::Anonymize,
            _ => TtlAction::Delete,
        }
    }
}

/// Current wall-clock time in microseconds since the Unix epoch.
pub fn now_micros() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_micros() as i64)
        .unwrap_or(0)
}

/// How a table's TTL is evaluated.
#[derive(Debug, Clone)]
pub enum TtlMode {
    /// No TTL configured.
    None,
    /// Fixed interval: a row expires when `ttl_column + ttl_seconds < now`.
    Interval {
        column_id: u32,
        ttl_seconds: i64,
        action: TtlAction,
    },
    /// Per-row retention: a row expires when `retention_column < now`.
    PerRow { column_id: u32, action: TtlAction },
}

/// Resolves the TTL mode for a table from its lifecycle config.
pub fn ttl_mode(entry: &TableEntry) -> TtlMode {
    let lc = &entry.lifecycle;
    let action = TtlAction::from_u8(lc.ttl_action);
    if lc.retention_column_id != 0 {
        return TtlMode::PerRow {
            column_id: lc.retention_column_id,
            action,
        };
    }
    if lc.ttl_column_id != 0 && lc.ttl_seconds > 0 {
        return TtlMode::Interval {
            column_id: lc.ttl_column_id,
            ttl_seconds: lc.ttl_seconds,
            action,
        };
    }
    TtlMode::None
}

/// Returns true when a row whose retention timestamp (micros) has the given
/// value is expired as of `now_us`.
pub fn is_expired(mode: &TtlMode, row_ts_micros: i64, now_us: i64) -> bool {
    match mode {
        TtlMode::None => false,
        TtlMode::Interval { ttl_seconds, .. } => {
            row_ts_micros + ttl_seconds.saturating_mul(1_000_000) < now_us
        }
        TtlMode::PerRow { .. } => row_ts_micros < now_us,
    }
}
