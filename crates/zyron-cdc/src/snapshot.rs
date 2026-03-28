//! Initial snapshot export for new replication slots.
//!
//! When a consumer creates a replication slot, it needs the current table
//! state as a baseline before streaming incremental changes. This module
//! exports a consistent point-in-time snapshot of table data.

use std::fs::{self, File};
use std::io::{Read, Write};
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};
use zyron_common::{Result, ZyronError};
use zyron_wal::Lsn;

// ---------------------------------------------------------------------------
// TableSnapshotInfo
// ---------------------------------------------------------------------------

/// Information about a single table's snapshot.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TableSnapshotInfo {
    pub table_id: u32,
    pub table_name: String,
    pub row_count: u64,
    pub snapshot_file: PathBuf,
}

// ---------------------------------------------------------------------------
// SnapshotExport
// ---------------------------------------------------------------------------

/// A complete snapshot export containing data from one or more tables.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SnapshotExport {
    pub snapshot_lsn: u64,
    pub table_snapshots: Vec<TableSnapshotInfo>,
}

impl SnapshotExport {
    /// Creates a new snapshot export at the given LSN.
    pub fn new(snapshot_lsn: Lsn) -> Self {
        Self {
            snapshot_lsn: snapshot_lsn.0,
            table_snapshots: Vec::new(),
        }
    }

    /// Adds a table snapshot to the export.
    pub fn add_table(
        &mut self,
        table_id: u32,
        table_name: String,
        row_count: u64,
        snapshot_file: PathBuf,
    ) {
        self.table_snapshots.push(TableSnapshotInfo {
            table_id,
            table_name,
            row_count,
            snapshot_file,
        });
    }

    /// Writes the export metadata to a JSON manifest file using atomic rename.
    pub fn write_manifest(&self, data_dir: &Path) -> Result<PathBuf> {
        let snap_dir = data_dir.join("snapshots");
        fs::create_dir_all(&snap_dir).map_err(|e| {
            ZyronError::CdcSnapshotFailed(format!("failed to create snapshot directory: {e}"))
        })?;

        let manifest_path = snap_dir.join(format!("snapshot_{}.json", self.snapshot_lsn));
        let tmp_path = snap_dir.join(format!("snapshot_{}.json.tmp", self.snapshot_lsn));
        let data = serde_json::to_vec(self).map_err(|e| {
            ZyronError::CdcSnapshotFailed(format!("failed to serialize manifest: {e}"))
        })?;

        {
            let mut file = File::create(&tmp_path)?;
            file.write_all(&data)?;
            file.sync_all()?;
        }

        fs::rename(&tmp_path, &manifest_path)?;
        Ok(manifest_path)
    }

    /// Reads a snapshot export from a manifest file.
    pub fn read_manifest(path: &Path) -> Result<Self> {
        let mut file = File::open(path)?;
        let mut data = Vec::new();
        file.read_to_end(&mut data)?;

        serde_json::from_slice(&data).map_err(|e| {
            ZyronError::CdcSnapshotFailed(format!("failed to parse snapshot manifest: {e}"))
        })
    }

    /// Total number of rows across all table snapshots.
    pub fn total_rows(&self) -> u64 {
        self.table_snapshots.iter().map(|t| t.row_count).sum()
    }
}

// ---------------------------------------------------------------------------
// SnapshotReader
// ---------------------------------------------------------------------------

/// Reads exported snapshot files row-by-row.
pub struct SnapshotReader {
    snapshot_file: PathBuf,
}

impl SnapshotReader {
    pub fn new(snapshot_file: PathBuf) -> Self {
        Self { snapshot_file }
    }

    /// Reads all rows from the snapshot file as serialized tuple byte arrays.
    /// Each row is stored as [u32 len][row_bytes...].
    pub fn read_all_rows(&self) -> Result<Vec<Vec<u8>>> {
        /// 64 MB per-record safety limit to prevent corrupt length fields
        /// from causing unbounded allocations.
        const MAX_RECORD_SIZE: usize = 64 * 1024 * 1024;

        let mut file = File::open(&self.snapshot_file)?;
        let file_len = file.metadata()?.len();
        let mut rows = Vec::new();
        let mut offset: u64 = 0;

        while offset + 4 <= file_len {
            let mut len_buf = [0u8; 4];
            file.read_exact(&mut len_buf)?;
            let row_len = u32::from_le_bytes(len_buf) as usize;

            if row_len > MAX_RECORD_SIZE {
                return Err(ZyronError::CdcSnapshotFailed(format!(
                    "record size {row_len} exceeds maximum {MAX_RECORD_SIZE} at offset {offset}"
                )));
            }

            if offset + 4 + row_len as u64 > file_len {
                break;
            }

            let mut row_data = vec![0u8; row_len];
            file.read_exact(&mut row_data)?;
            rows.push(row_data);

            offset += 4 + row_len as u64;
        }

        Ok(rows)
    }

    /// Returns the path to the snapshot file.
    pub fn file_path(&self) -> &Path {
        &self.snapshot_file
    }
}

// ---------------------------------------------------------------------------
// Cleanup
// ---------------------------------------------------------------------------

/// Removes snapshot files after the consumer has loaded them.
pub fn cleanup_snapshot(export: &SnapshotExport) -> Result<()> {
    for info in &export.table_snapshots {
        if info.snapshot_file.exists() {
            fs::remove_file(&info.snapshot_file)?;
        }
    }
    Ok(())
}

/// Writes rows to a snapshot file in length-prefixed format.
pub fn write_snapshot_rows(path: &Path, rows: &[Vec<u8>]) -> Result<u64> {
    let parent = path.parent().ok_or_else(|| {
        ZyronError::CdcSnapshotFailed("snapshot file has no parent directory".into())
    })?;
    fs::create_dir_all(parent).map_err(|e| {
        ZyronError::CdcSnapshotFailed(format!("failed to create snapshot directory: {e}"))
    })?;

    let mut file = File::create(path)?;
    let mut count: u64 = 0;

    for row in rows {
        let len = row.len() as u32;
        file.write_all(&len.to_le_bytes())?;
        file.write_all(row)?;
        count += 1;
    }

    file.sync_all()?;
    Ok(count)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[test]
    fn test_snapshot_export_manifest() {
        let tmp = TempDir::new().unwrap();

        let mut export = SnapshotExport::new(Lsn(5000));
        export.add_table(1, "users".into(), 100, tmp.path().join("snap_users.dat"));
        export.add_table(2, "orders".into(), 50, tmp.path().join("snap_orders.dat"));

        assert_eq!(export.total_rows(), 150);

        let manifest_path = export.write_manifest(tmp.path()).unwrap();
        assert!(manifest_path.exists());

        let loaded = SnapshotExport::read_manifest(&manifest_path).unwrap();
        assert_eq!(loaded.snapshot_lsn, 5000);
        assert_eq!(loaded.table_snapshots.len(), 2);
        assert_eq!(loaded.total_rows(), 150);
    }

    #[test]
    fn test_write_and_read_snapshot_rows() {
        let tmp = TempDir::new().unwrap();
        let path = tmp.path().join("snap_test.dat");

        let rows = vec![vec![1u8, 2, 3], vec![4, 5, 6, 7], vec![8]];

        let count = write_snapshot_rows(&path, &rows).unwrap();
        assert_eq!(count, 3);

        let reader = SnapshotReader::new(path);
        let loaded = reader.read_all_rows().unwrap();
        assert_eq!(loaded.len(), 3);
        assert_eq!(loaded[0], vec![1, 2, 3]);
        assert_eq!(loaded[1], vec![4, 5, 6, 7]);
        assert_eq!(loaded[2], vec![8]);
    }

    #[test]
    fn test_cleanup_snapshot() {
        let tmp = TempDir::new().unwrap();
        let path = tmp.path().join("snap_cleanup.dat");

        write_snapshot_rows(&path, &[vec![1, 2, 3]]).unwrap();
        assert!(path.exists());

        let mut export = SnapshotExport::new(Lsn(1000));
        export.add_table(1, "test".into(), 1, path.clone());

        cleanup_snapshot(&export).unwrap();
        assert!(!path.exists());
    }

    #[test]
    fn test_snapshot_reader_missing_file() {
        let reader = SnapshotReader::new(PathBuf::from("/nonexistent/file.dat"));
        assert!(reader.read_all_rows().is_err());
    }
}
