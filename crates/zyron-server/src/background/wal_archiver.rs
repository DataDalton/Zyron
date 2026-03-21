//! WAL segment archiver.
//!
//! Copies completed WAL segments to an archive directory for backup and
//! point-in-time recovery. Enforces a retention policy by deleting
//! archived segments beyond the configured count.

use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, OnceLock};
use std::thread::{self, JoinHandle};
use std::time::Duration;

use tracing::{debug, error, info};

/// Configuration for the WAL archiver.
#[derive(Debug, Clone)]
pub struct WalArchiverConfig {
    /// Source WAL directory.
    pub wal_dir: PathBuf,
    /// Destination archive directory.
    pub archive_dir: PathBuf,
    /// Maximum archived segments to retain (default 100).
    pub retention_count: usize,
    /// Check interval in seconds (default 30).
    pub interval_secs: u64,
}

/// WAL archiver statistics.
pub struct WalArchiverStats {
    pub segments_archived: AtomicU64,
    pub segments_purged: AtomicU64,
}

impl WalArchiverStats {
    fn new() -> Self {
        Self {
            segments_archived: AtomicU64::new(0),
            segments_purged: AtomicU64::new(0),
        }
    }
}

/// Background worker that archives completed WAL segments.
pub struct WalArchiver {
    shutdown: Arc<AtomicBool>,
    waker: Arc<OnceLock<thread::Thread>>,
    thread: Option<JoinHandle<()>>,
    stats: Arc<WalArchiverStats>,
}

impl WalArchiver {
    /// Starts the WAL archiver thread.
    pub fn start(config: WalArchiverConfig) -> Self {
        let shutdown = Arc::new(AtomicBool::new(false));
        let waker = Arc::new(OnceLock::new());
        let stats = Arc::new(WalArchiverStats::new());

        let thread_shutdown = Arc::clone(&shutdown);
        let thread_waker = Arc::clone(&waker);
        let thread_stats = Arc::clone(&stats);

        let handle = thread::Builder::new()
            .name("zyron-wal-archiver".into())
            .spawn(move || {
                let _ = thread_waker.set(thread::current());
                Self::archiver_loop(&config, &thread_shutdown, &thread_stats);
            })
            .expect("failed to spawn WAL archiver thread");

        Self {
            shutdown,
            waker,
            thread: Some(handle),
            stats,
        }
    }

    /// Main archiver loop.
    fn archiver_loop(config: &WalArchiverConfig, shutdown: &AtomicBool, stats: &WalArchiverStats) {
        let interval = Duration::from_secs(config.interval_secs);

        // Create archive directory if it does not exist
        if let Err(e) = std::fs::create_dir_all(&config.archive_dir) {
            error!(
                "Failed to create archive directory {}: {}",
                config.archive_dir.display(),
                e
            );
            return;
        }

        loop {
            thread::park_timeout(interval);

            if shutdown.load(Ordering::Acquire) {
                return;
            }

            // List WAL segment files (sorted by name for chronological order)
            let mut wal_files = match Self::list_wal_segments(&config.wal_dir) {
                Ok(files) => files,
                Err(e) => {
                    debug!("Failed to list WAL segments: {}", e);
                    continue;
                }
            };
            wal_files.sort();

            // List already-archived segments
            let archived = match Self::list_wal_segments(&config.archive_dir) {
                Ok(files) => files.into_iter().collect::<std::collections::HashSet<_>>(),
                Err(_) => std::collections::HashSet::new(),
            };

            // Archive new segments (skip the last one, which may still be active)
            let archivable = if wal_files.len() > 1 {
                &wal_files[..wal_files.len() - 1]
            } else {
                &[]
            };

            let mut newly_archived = 0u64;
            for filename in archivable {
                if archived.contains(filename) {
                    continue;
                }
                let src = config.wal_dir.join(filename);
                let dst = config.archive_dir.join(filename);
                let tmp = config.archive_dir.join(format!("{}.tmp", filename));
                // Atomic copy: write to temp file, then rename.
                // Rename is atomic on the same filesystem, preventing
                // partial files if the archiver crashes mid-copy.
                if let Err(e) = std::fs::copy(&src, &tmp) {
                    error!("Failed to copy {} to temp: {}", filename, e);
                    let _ = std::fs::remove_file(&tmp);
                    continue;
                }
                if let Err(e) = std::fs::rename(&tmp, &dst) {
                    error!("Failed to rename temp to {}: {}", filename, e);
                    let _ = std::fs::remove_file(&tmp);
                    continue;
                }
                newly_archived += 1;
            }

            if newly_archived > 0 {
                stats
                    .segments_archived
                    .fetch_add(newly_archived, Ordering::Relaxed);
                info!("Archived {} WAL segments", newly_archived);
            }

            // Enforce retention policy: delete oldest archived segments
            Self::enforce_retention(&config.archive_dir, config.retention_count, stats);
        }
    }

    /// Lists WAL segment filenames in a directory (files matching the segment pattern).
    fn list_wal_segments(dir: &PathBuf) -> std::io::Result<Vec<String>> {
        let mut segments = Vec::new();
        for entry in std::fs::read_dir(dir)? {
            let entry = entry?;
            if entry.file_type()?.is_file() {
                if let Some(name) = entry.file_name().to_str() {
                    // WAL segments are named with numeric IDs (e.g., "00000001.wal").
                    // Validate the stem is all digits to avoid matching unrelated .wal files.
                    if name.ends_with(".wal") {
                        let stem = name.trim_end_matches(".wal");
                        if !stem.is_empty() && stem.chars().all(|c| c.is_ascii_digit()) {
                            segments.push(name.to_string());
                        }
                    }
                }
            }
        }
        Ok(segments)
    }

    /// Deletes the oldest archived segments to stay within retention_count.
    fn enforce_retention(archive_dir: &PathBuf, retention_count: usize, stats: &WalArchiverStats) {
        let mut archived = match Self::list_wal_segments(archive_dir) {
            Ok(files) => files,
            Err(_) => return,
        };

        if archived.len() <= retention_count {
            return;
        }

        archived.sort();
        let to_delete = archived.len() - retention_count;
        let mut purged = 0u64;

        for filename in &archived[..to_delete] {
            let path = archive_dir.join(filename);
            if let Err(e) = std::fs::remove_file(&path) {
                debug!("Failed to purge archived segment {}: {}", filename, e);
            } else {
                purged += 1;
            }
        }

        if purged > 0 {
            stats.segments_purged.fetch_add(purged, Ordering::Relaxed);
            debug!("Purged {} archived WAL segments", purged);
        }
    }

    /// Returns a reference to archiver statistics.
    pub fn stats(&self) -> &Arc<WalArchiverStats> {
        &self.stats
    }

    /// Gracefully shuts down the archiver thread.
    pub fn shutdown(&mut self) {
        self.shutdown.store(true, Ordering::Release);
        if let Some(t) = self.waker.get() {
            t.unpark();
        }
        if let Some(handle) = self.thread.take() {
            let _ = handle.join();
        }
    }
}

impl Drop for WalArchiver {
    fn drop(&mut self) {
        if self.thread.is_some() {
            self.shutdown();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_values() {
        let config = WalArchiverConfig {
            wal_dir: PathBuf::from("/data/wal"),
            archive_dir: PathBuf::from("/archive/wal"),
            retention_count: 50,
            interval_secs: 15,
        };
        assert_eq!(config.retention_count, 50);
        assert_eq!(config.interval_secs, 15);
    }

    #[test]
    fn test_stats_initial() {
        let stats = WalArchiverStats::new();
        assert_eq!(stats.segments_archived.load(Ordering::Relaxed), 0);
        assert_eq!(stats.segments_purged.load(Ordering::Relaxed), 0);
    }
}
