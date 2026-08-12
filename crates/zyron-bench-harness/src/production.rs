//! The configuration the engine ships with. Benchmarks get this and
//! nothing else.
//!
//! A suite that configures the engine by hand measures an engine nobody
//! runs. That was not hypothetical here: every suite built its own
//! `WalWriterConfig` literal, and because a Rust struct literal has to
//! name every field, none of them ever inherited a default. They were
//! hand-written, copied and drifted, so WAL rings spanned sixty-four fold
//! and fsync was off everywhere while the server ships with it on.
//!
//! Every value below is read from the same `Default` implementations the
//! server reads, never restated, so changing a shipping default moves the
//! benchmarks with it rather than leaving them behind.
//!
//! Only the directories differ, because a run needs its own, and a path
//! is a location rather than a tunable. There is no way to ask this
//! module for anything else. A suite whose subject is a setting sweeps a
//! range of values that includes the production one, which is a different
//! test and says so.

use std::path::{Path, PathBuf};

use zyron_buffer::BufferPoolConfig;
use zyron_common::config::StorageConfig;
use zyron_storage::DiskManagerConfig;
use zyron_storage::columnar::CompactionConfig;
use zyron_wal::WalWriterConfig;

/// The WAL configuration the server runs, pointed at `wal_dir`
pub fn wal_config(wal_dir: impl AsRef<Path>) -> WalWriterConfig {
    WalWriterConfig {
        wal_dir: wal_dir.as_ref().to_path_buf(),
        ..WalWriterConfig::default()
    }
}

/// The buffer pool the server runs
pub fn buffer_pool_config() -> BufferPoolConfig {
    BufferPoolConfig {
        num_frames: StorageConfig::default().buffer_pool_pages,
    }
}

/// The disk manager the server runs, pointed at `data_dir`
pub fn disk_config(data_dir: impl AsRef<Path>) -> DiskManagerConfig {
    DiskManagerConfig {
        data_dir: data_dir.as_ref().to_path_buf(),
        fsync_enabled: StorageConfig::default().fsync_enabled,
    }
}

/// The fold configuration the server runs, pointed at `columnar_dir`.
///
/// Encoding threads and fsync in particular: suites ran one thread
/// against the four the server uses, so every fold number described a
/// single-threaded encoder that nobody ships.
///
/// A functional test that has to fold fewer rows than the server's
/// trigger asks for sets `min_rows` on the result, which is a trigger
/// threshold rather than a tunable and is visible where it is done. A
/// benchmark leaves every field alone.
pub fn compaction_config(columnar_dir: impl AsRef<Path>) -> CompactionConfig {
    CompactionConfig {
        columnar_dir: columnar_dir.as_ref().to_path_buf(),
        ..CompactionConfig::default()
    }
}

/// A data and WAL directory pair under one root, named the way the server
/// names them
pub fn data_and_wal_dirs(root: &Path) -> (PathBuf, PathBuf) {
    let storage = StorageConfig::default();
    let name = |p: &Path, fallback: &str| {
        p.file_name()
            .map(|n| n.to_owned())
            .unwrap_or_else(|| std::ffi::OsString::from(fallback))
    };
    let data = root.join(name(&storage.data_dir, "data"));
    let wal = data.join(name(&storage.wal_dir, "wal"));
    (data, wal)
}

/// Creates both directories and returns them
pub fn create_dirs(root: &Path) -> std::io::Result<(PathBuf, PathBuf)> {
    let (data, wal) = data_and_wal_dirs(root);
    std::fs::create_dir_all(&data)?;
    std::fs::create_dir_all(&wal)?;
    Ok((data, wal))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Field by field against the shipping defaults. A suite cannot fork a
    /// value without changing the server's default, which is the point
    #[test]
    fn test_the_harness_hands_out_the_shipping_configuration() {
        let root = Path::new("/tmp/zyron-bench");
        let (data, wal_dir) = data_and_wal_dirs(root);

        let wal = wal_config(&wal_dir);
        let shipped = WalWriterConfig::default();
        assert_eq!(wal.ring_buffer_capacity, shipped.ring_buffer_capacity);
        assert_eq!(wal.segment_size, shipped.segment_size);
        assert_eq!(
            wal.fsync_enabled, shipped.fsync_enabled,
            "the server ships with fsync on, so a benchmark measures it on"
        );
        assert_eq!(wal.wal_dir, wal_dir, "only the location differs");

        let storage = StorageConfig::default();
        assert_eq!(buffer_pool_config().num_frames, storage.buffer_pool_pages);
        let disk = disk_config(&data);
        assert_eq!(disk.fsync_enabled, storage.fsync_enabled);
        assert_eq!(disk.data_dir, data);

        let fold = compaction_config(data.join("columnar"));
        let shipped_fold = CompactionConfig::default();
        assert_eq!(fold.max_encoding_threads, shipped_fold.max_encoding_threads);
        assert_eq!(fold.fsync_enabled, shipped_fold.fsync_enabled);
        assert_eq!(fold.min_rows, shipped_fold.min_rows);
        assert_eq!(fold.max_rows_per_file, shipped_fold.max_rows_per_file);
        assert_eq!(fold.exact_encoding, shipped_fold.exact_encoding);
        assert_eq!(
            fold.oltp_p99_threshold_us,
            shipped_fold.oltp_p99_threshold_us
        );
        assert_eq!(fold.check_interval_ms, shipped_fold.check_interval_ms);
    }
}
