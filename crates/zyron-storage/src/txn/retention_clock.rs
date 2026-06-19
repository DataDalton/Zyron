//! Wall-clock to WAL-LSN sample log for time-based time-travel retention.
//!
//! Time-based retention keeps row versions whose delete committed within a
//! window of now. Versions are dated by commit LSN, so the vacuum worker needs
//! to translate "now minus the window" into a floor LSN. This records periodic
//! (timestamp, LSN) samples and answers that translation with a binary search.
//! Resolution is the sample interval (the vacuum cycle), which is far finer
//! than the day-scale windows it serves, so per-transaction timestamps are not
//! needed. The log is bounded by pruning samples older than the longest window.

use std::path::{Path, PathBuf};
use std::sync::RwLock;

use zyron_common::{Result, ZyronError};

/// Append-only (wall-clock micros, WAL LSN) samples, strictly increasing in
/// both. Recorded by the vacuum worker once per cycle and persisted at
/// checkpoint so retention resumes immediately after restart.
pub struct RetentionClock {
    samples: RwLock<Vec<(u64, u64)>>,
}

impl RetentionClock {
    pub fn new() -> Self {
        Self {
            samples: RwLock::new(Vec::new()),
        }
    }

    /// Records a sample. Skipped when not strictly newer than the last in both
    /// timestamp and LSN, so the series stays monotonic and a stalled clock or
    /// idle WAL does not insert duplicates.
    pub fn record(&self, ts_micros: u64, lsn: u64) {
        let mut s = self.samples.write().unwrap_or_else(|e| e.into_inner());
        match s.last() {
            Some(&(last_ts, last_lsn)) if ts_micros <= last_ts || lsn < last_lsn => {}
            _ => s.push((ts_micros, lsn)),
        }
    }

    /// Returns the floor LSN for a cutoff timestamp: the LSN of the newest
    /// sample at or before the cutoff. Returns 0 when no sample is that old, so
    /// the caller retains everything older than the recorded history (the safe
    /// over-retain direction) rather than reclaiming undated versions.
    pub fn lsn_at(&self, cutoff_micros: u64) -> u64 {
        let s = self.samples.read().unwrap_or_else(|e| e.into_inner());
        let idx = s.partition_point(|&(ts, _)| ts <= cutoff_micros);
        if idx == 0 { 0 } else { s[idx - 1].1 }
    }

    /// Drops samples older than `min_micros`, keeping the newest sample at or
    /// before it so a cutoff just inside the retained window still resolves.
    pub fn prune_before(&self, min_micros: u64) {
        let mut s = self.samples.write().unwrap_or_else(|e| e.into_inner());
        let idx = s.partition_point(|&(ts, _)| ts <= min_micros);
        let drop_to = idx.saturating_sub(1);
        if drop_to > 0 {
            s.drain(0..drop_to);
        }
    }

    /// Number of retained samples.
    pub fn len(&self) -> usize {
        self.samples.read().unwrap_or_else(|e| e.into_inner()).len()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    fn file_path(dir: &Path) -> PathBuf {
        dir.join(".zyretention")
    }

    /// Persists the samples with an atomic rename so a crash mid-write never
    /// leaves a torn file.
    pub fn persist(&self, dir: &Path) -> Result<()> {
        let s = self.samples.read().unwrap_or_else(|e| e.into_inner());
        let mut buf = Vec::with_capacity(8 + s.len() * 16);
        buf.extend_from_slice(&(s.len() as u64).to_le_bytes());
        for &(ts, lsn) in s.iter() {
            buf.extend_from_slice(&ts.to_le_bytes());
            buf.extend_from_slice(&lsn.to_le_bytes());
        }
        drop(s);
        let path = Self::file_path(dir);
        let tmp = path.with_extension("zyretention.tmp");
        {
            use std::io::Write;
            let mut f = std::fs::File::create(&tmp).map_err(ZyronError::Io)?;
            f.write_all(&buf).map_err(ZyronError::Io)?;
            f.sync_all().map_err(ZyronError::Io)?;
        }
        std::fs::rename(&tmp, &path).map_err(ZyronError::Io)?;
        Ok(())
    }

    /// Loads persisted samples. A missing file is a clean no-op.
    pub fn load(&self, dir: &Path) -> Result<()> {
        let path = Self::file_path(dir);
        let data = match std::fs::read(&path) {
            Ok(d) => d,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(()),
            Err(e) => return Err(ZyronError::Io(e)),
        };
        if data.len() < 8 {
            return Ok(());
        }
        let count = u64::from_le_bytes(data[0..8].try_into().unwrap()) as usize;
        let mut samples = Vec::with_capacity(count);
        let mut off = 8;
        for _ in 0..count {
            if off + 16 > data.len() {
                break;
            }
            let ts = u64::from_le_bytes(data[off..off + 8].try_into().unwrap());
            let lsn = u64::from_le_bytes(data[off + 8..off + 16].try_into().unwrap());
            samples.push((ts, lsn));
            off += 16;
        }
        *self.samples.write().unwrap_or_else(|e| e.into_inner()) = samples;
        Ok(())
    }
}

impl Default for RetentionClock {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lsn_at_finds_newest_sample_at_or_before_cutoff() {
        let c = RetentionClock::new();
        c.record(100, 10);
        c.record(200, 25);
        c.record(300, 40);
        // Cutoff before the first sample: nothing that old, floor 0.
        assert_eq!(c.lsn_at(50), 0);
        // Exact and between samples resolve to the newest at-or-before.
        assert_eq!(c.lsn_at(200), 25);
        assert_eq!(c.lsn_at(250), 25);
        assert_eq!(c.lsn_at(999), 40);
    }

    #[test]
    fn record_skips_non_monotonic() {
        let c = RetentionClock::new();
        c.record(100, 10);
        c.record(90, 20); // older timestamp, skipped
        c.record(110, 5); // lower lsn, skipped
        c.record(120, 30); // accepted
        assert_eq!(c.len(), 2);
        assert_eq!(c.lsn_at(120), 30);
    }

    #[test]
    fn prune_keeps_floor_resolvable() {
        let c = RetentionClock::new();
        for i in 1..=10u64 {
            c.record(i * 100, i * 10);
        }
        c.prune_before(550);
        // A cutoff inside the retained window still resolves to a real floor.
        assert!(c.lsn_at(560) >= 50);
        assert!(c.len() < 10);
    }

    #[test]
    fn persist_load_round_trip() {
        let dir = tempfile::TempDir::new().unwrap();
        let c = RetentionClock::new();
        c.record(100, 10);
        c.record(200, 25);
        c.persist(dir.path()).unwrap();
        let loaded = RetentionClock::new();
        loaded.load(dir.path()).unwrap();
        assert_eq!(loaded.lsn_at(250), 25);
        assert_eq!(loaded.len(), 2);
    }
}
