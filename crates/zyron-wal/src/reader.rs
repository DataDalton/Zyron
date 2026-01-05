//! WAL reader for log replay and recovery.

use crate::record::{LogRecord, LogRecordType, Lsn};
use crate::segment::{LogSegment, SegmentHeader, SegmentId};
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use zyron_common::{Result, ZyronError};

/// WAL reader for scanning and replaying log records.
pub struct WalReader {
    /// Directory containing WAL segments.
    wal_dir: PathBuf,
    /// Cached segment files sorted by ID.
    segments: BTreeMap<SegmentId, PathBuf>,
}

impl WalReader {
    /// Creates a new WAL reader.
    pub async fn new(wal_dir: &Path) -> Result<Self> {
        let mut segments = BTreeMap::new();

        if wal_dir.exists() {
            let mut dir = tokio::fs::read_dir(wal_dir).await?;
            while let Some(entry) = dir.next_entry().await? {
                let path = entry.path();

                if path.extension().map(|e| e == "wal").unwrap_or(false) {
                    // Extract segment ID from filename
                    if let Some(stem) = path.file_stem() {
                        if let Ok(id) = stem.to_string_lossy().parse::<u32>() {
                            segments.insert(SegmentId(id), path);
                        }
                    }
                }
            }
        }

        Ok(Self {
            wal_dir: wal_dir.to_path_buf(),
            segments,
        })
    }

    /// Returns the number of segment files.
    pub fn segment_count(&self) -> usize {
        self.segments.len()
    }

    /// Returns the first segment ID, if any.
    pub fn first_segment_id(&self) -> Option<SegmentId> {
        self.segments.keys().next().copied()
    }

    /// Returns the last segment ID, if any.
    pub fn last_segment_id(&self) -> Option<SegmentId> {
        self.segments.keys().last().copied()
    }

    /// Opens a segment by ID.
    pub async fn open_segment(&self, segment_id: SegmentId) -> Result<LogSegment> {
        let path = self.segments.get(&segment_id).ok_or_else(|| {
            ZyronError::WalCorrupted {
                lsn: 0,
                reason: format!("segment {} not found", segment_id),
            }
        })?;

        LogSegment::open(path).await
    }

    /// Scans all records starting from the given LSN.
    ///
    /// Returns a vector of all records from start_lsn onwards.
    pub async fn scan_from(&self, start_lsn: Lsn) -> Result<Vec<LogRecord>> {
        let mut results = Vec::new();
        let segment_id = SegmentId(start_lsn.segment_id());
        let start_offset = start_lsn.offset();

        // Find starting segment
        let mut current_segment_id = Some(segment_id);
        let mut is_first_segment = true;

        while let Some(seg_id) = current_segment_id {
            if !self.segments.contains_key(&seg_id) {
                break;
            }

            let mut segment = self.open_segment(seg_id).await?;

            // Bulk read entire segment data into memory
            let data = segment.read_all_data().await?;
            segment.close().await?;

            if !data.is_empty() {
                // Parse all records from memory buffer (zero-copy via Bytes slicing)
                let segment_records = LogRecord::parse_all(data)?;

                // For first segment, skip records before start_offset
                if is_first_segment && start_offset > SegmentHeader::SIZE as u32 {
                    let skip_bytes = (start_offset - SegmentHeader::SIZE as u32) as usize;
                    let mut byte_offset = 0;
                    for record in segment_records {
                        if byte_offset >= skip_bytes {
                            results.push(record);
                        } else {
                            byte_offset += record.size_on_disk();
                        }
                    }
                } else {
                    results.extend(segment_records);
                }
            }

            is_first_segment = false;

            // Move to next segment
            current_segment_id = Some(seg_id.next());
        }

        Ok(results)
    }

    /// Scans all records in the WAL.
    ///
    /// Returns a vector of all records with checksum verification.
    pub async fn scan_all(&self) -> Result<Vec<LogRecord>> {
        let mut results = Vec::new();

        let first_id = match self.first_segment_id() {
            Some(id) => id,
            None => return Ok(results),
        };

        let mut current_segment_id = Some(first_id);

        while let Some(seg_id) = current_segment_id {
            if !self.segments.contains_key(&seg_id) {
                break;
            }

            let mut segment = self.open_segment(seg_id).await?;

            // Bulk read entire segment data into memory
            let data = segment.read_all_data().await?;
            segment.close().await?;

            if !data.is_empty() {
                // Parse all records from memory buffer (zero-copy via Bytes slicing)
                let segment_records = LogRecord::parse_all(data)?;
                results.extend(segment_records);
            }

            // Move to next segment
            current_segment_id = Some(seg_id.next());
        }

        Ok(results)
    }

    /// Finds the last checkpoint in the WAL.
    pub async fn find_last_checkpoint(&self) -> Result<Option<(Lsn, LogRecord)>> {
        let mut last_checkpoint: Option<(Lsn, LogRecord)> = None;

        for record in self.scan_all().await? {
            if record.record_type == LogRecordType::CheckpointEnd {
                last_checkpoint = Some((record.lsn, record));
            }
        }

        Ok(last_checkpoint)
    }

    /// Collects all active transactions at the given LSN.
    pub async fn find_active_transactions(&self, at_lsn: Lsn) -> Result<Vec<u32>> {
        let mut active = std::collections::HashSet::new();

        for record in self.scan_all().await? {
            if record.lsn > at_lsn {
                break;
            }

            match record.record_type {
                LogRecordType::Begin => {
                    active.insert(record.txn_id);
                }
                LogRecordType::Commit | LogRecordType::Abort => {
                    active.remove(&record.txn_id);
                }
                _ => {}
            }
        }

        Ok(active.into_iter().collect())
    }

    /// Refreshes the list of segment files.
    pub async fn refresh(&mut self) -> Result<()> {
        self.segments.clear();

        let mut dir = tokio::fs::read_dir(&self.wal_dir).await?;
        while let Some(entry) = dir.next_entry().await? {
            let path = entry.path();

            if path.extension().map(|e| e == "wal").unwrap_or(false) {
                if let Some(stem) = path.file_stem() {
                    if let Ok(id) = stem.to_string_lossy().parse::<u32>() {
                        self.segments.insert(SegmentId(id), path);
                    }
                }
            }
        }

        Ok(())
    }
}

/// Synchronous WAL reader for replay without async overhead.
///
/// Uses standard library I/O for maximum throughput during recovery.
/// All file reads are performed synchronously with no await points.
pub struct SyncWalReader {
    segments: BTreeMap<SegmentId, PathBuf>,
}

impl SyncWalReader {
    /// Creates a new synchronous WAL reader.
    pub fn new_sync(wal_dir: &Path) -> Result<Self> {
        let mut segments = BTreeMap::new();

        if wal_dir.exists() {
            for entry in std::fs::read_dir(wal_dir)? {
                let entry = entry?;
                let path = entry.path();

                if path.extension().map(|e| e == "wal").unwrap_or(false) {
                    if let Some(stem) = path.file_stem() {
                        if let Ok(id) = stem.to_string_lossy().parse::<u32>() {
                            segments.insert(SegmentId(id), path);
                        }
                    }
                }
            }
        }

        Ok(Self { segments })
    }

    /// Returns the number of segment files.
    pub fn segment_count(&self) -> usize {
        self.segments.len()
    }

    /// Returns the first segment ID, if any.
    pub fn first_segment_id(&self) -> Option<SegmentId> {
        self.segments.keys().next().copied()
    }

    /// Returns the last segment ID, if any.
    pub fn last_segment_id(&self) -> Option<SegmentId> {
        self.segments.keys().last().copied()
    }

    /// Scans all records in the WAL synchronously.
    ///
    /// Bulk reads entire segments and parses all records without async overhead.
    /// Returns a vector of all records with checksum verification.
    /// Pre-allocates result vector based on estimated record count from segment sizes.
    #[inline]
    pub fn scan_all_sync(&self) -> Result<Vec<LogRecord>> {
        use crate::segment::SyncLogSegment;

        let first_id = match self.first_segment_id() {
            Some(id) => id,
            None => return Ok(Vec::new()),
        };

        // Pre-calculate total data size to estimate record count
        // Average record is ~64 bytes (header + small payload)
        let mut total_bytes: usize = 0;
        for path in self.segments.values() {
            if let Ok(metadata) = std::fs::metadata(path) {
                total_bytes += metadata.len() as usize;
            }
        }
        let estimated_records = total_bytes / 64;
        let mut results = Vec::with_capacity(estimated_records);

        let mut current_segment_id = Some(first_id);

        while let Some(seg_id) = current_segment_id {
            if let Some(path) = self.segments.get(&seg_id) {
                let mut segment = SyncLogSegment::open_sync(path)?;

                // Bulk read entire segment data into memory
                let data = segment.read_all_data_sync()?;

                if !data.is_empty() {
                    // Parse all records from memory buffer
                    let segment_records = LogRecord::parse_all(data)?;
                    results.extend(segment_records);
                }

                // Move to next segment
                current_segment_id = Some(seg_id.next());
            } else {
                break;
            }
        }

        Ok(results)
    }

    /// Scans records starting from the given LSN synchronously.
    /// Pre-allocates result vector based on estimated record count.
    #[inline]
    pub fn scan_from_sync(&self, start_lsn: Lsn) -> Result<Vec<LogRecord>> {
        use crate::segment::SyncLogSegment;

        let segment_id = SegmentId(start_lsn.segment_id());
        let start_offset = start_lsn.offset();

        // Estimate records from remaining segments
        let mut total_bytes: usize = 0;
        for (&seg_id, path) in &self.segments {
            if seg_id.0 >= segment_id.0 {
                if let Ok(metadata) = std::fs::metadata(path) {
                    total_bytes += metadata.len() as usize;
                }
            }
        }
        let estimated_records = total_bytes / 64;
        let mut results = Vec::with_capacity(estimated_records);

        let mut current_segment_id = Some(segment_id);
        let mut is_first_segment = true;

        while let Some(seg_id) = current_segment_id {
            if let Some(path) = self.segments.get(&seg_id) {
                let mut segment = SyncLogSegment::open_sync(path)?;

                // Bulk read entire segment data into memory
                let data = segment.read_all_data_sync()?;

                if !data.is_empty() {
                    // Parse all records from memory buffer
                    let segment_records = LogRecord::parse_all(data)?;

                    // For first segment, skip records before start_offset
                    if is_first_segment && start_offset > SegmentHeader::SIZE as u32 {
                        let skip_bytes = (start_offset - SegmentHeader::SIZE as u32) as usize;
                        let mut byte_offset = 0;
                        for record in segment_records {
                            if byte_offset >= skip_bytes {
                                results.push(record);
                            } else {
                                byte_offset += record.size_on_disk();
                            }
                        }
                    } else {
                        results.extend(segment_records);
                    }
                }

                is_first_segment = false;
                current_segment_id = Some(seg_id.next());
            } else {
                break;
            }
        }

        Ok(results)
    }
}

/// Recovery manager for replaying WAL during startup.
pub struct RecoveryManager {
    reader: WalReader,
}

impl RecoveryManager {
    /// Creates a new recovery manager.
    pub async fn new(wal_dir: &Path) -> Result<Self> {
        let reader = WalReader::new(wal_dir).await?;
        Ok(Self { reader })
    }

    /// Performs recovery, returning the redo and undo information.
    pub async fn recover(&self) -> Result<RecoveryResult> {
        // Find the last checkpoint
        let checkpoint = self.reader.find_last_checkpoint().await?;

        let start_lsn = if let Some((lsn, _)) = checkpoint {
            lsn
        } else if let Some(first_id) = self.reader.first_segment_id() {
            Lsn::new(first_id.0, SegmentHeader::SIZE as u32)
        } else {
            return Ok(RecoveryResult::empty());
        };

        // Scan from checkpoint and collect records
        let mut redo_records = Vec::new();
        let mut active_txns = std::collections::HashMap::new();
        let mut committed_txns = std::collections::HashSet::new();
        let mut aborted_txns = std::collections::HashSet::new();

        for record in self.reader.scan_from(start_lsn).await? {
            let txn_id = record.txn_id;

            match record.record_type {
                LogRecordType::Begin => {
                    active_txns.insert(txn_id, record.lsn);
                }
                LogRecordType::Commit => {
                    active_txns.remove(&txn_id);
                    committed_txns.insert(txn_id);
                }
                LogRecordType::Abort => {
                    active_txns.remove(&txn_id);
                    aborted_txns.insert(txn_id);
                }
                LogRecordType::Insert
                | LogRecordType::Update
                | LogRecordType::Delete
                | LogRecordType::FullPage => {
                    redo_records.push(record);
                }
                _ => {}
            }
        }

        // Filter redo records: only committed transactions
        let redo_records: Vec<_> = redo_records
            .into_iter()
            .filter(|r| committed_txns.contains(&r.txn_id))
            .collect();

        // Undo records: from active (uncommitted) transactions
        let undo_txns: Vec<_> = active_txns.keys().copied().collect();

        Ok(RecoveryResult {
            redo_records,
            undo_txns,
            last_lsn: self.reader.last_segment_id().map(|id| Lsn::new(id.0, 0)),
        })
    }
}

/// Result of WAL recovery.
#[derive(Debug)]
pub struct RecoveryResult {
    /// Records to redo (from committed transactions).
    pub redo_records: Vec<LogRecord>,
    /// Transaction IDs to undo (uncommitted at crash).
    pub undo_txns: Vec<u32>,
    /// Last LSN found in the WAL.
    pub last_lsn: Option<Lsn>,
}

impl RecoveryResult {
    /// Creates an empty recovery result.
    pub fn empty() -> Self {
        Self {
            redo_records: Vec::new(),
            undo_txns: Vec::new(),
            last_lsn: None,
        }
    }

    /// Returns true if there's nothing to recover.
    pub fn is_empty(&self) -> bool {
        self.redo_records.is_empty() && self.undo_txns.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::writer::{WalWriter, WalWriterConfig};
    use bytes::Bytes;
    use tempfile::tempdir;

    async fn create_test_wal() -> (WalWriter, tempfile::TempDir) {
        let dir = tempdir().unwrap();
        let config = WalWriterConfig {
            wal_dir: dir.path().to_path_buf(),
            segment_size: LogSegment::DEFAULT_SIZE,
            fsync_enabled: true,
            group_commit_size: 1,
        };
        let writer = WalWriter::new(config).await.unwrap();
        (writer, dir)
    }

    #[tokio::test]
    async fn test_wal_reader_empty() {
        let dir = tempdir().unwrap();
        let reader = WalReader::new(dir.path()).await.unwrap();
        assert_eq!(reader.segment_count(), 0);
        assert!(reader.first_segment_id().is_none());
    }

    #[tokio::test]
    async fn test_wal_reader_with_segments() {
        let (writer, dir) = create_test_wal().await;

        // Write some records
        writer.log_begin(1).await.unwrap();
        writer.log_commit(1, Lsn::INVALID).await.unwrap();
        writer.close().await.unwrap();

        // Read back
        let reader = WalReader::new(dir.path()).await.unwrap();
        assert_eq!(reader.segment_count(), 1);
        assert_eq!(reader.first_segment_id(), Some(SegmentId::FIRST));
    }

    #[tokio::test]
    async fn test_wal_reader_scan_all() {
        let (writer, dir) = create_test_wal().await;

        // Write multiple transactions
        for i in 1..=5 {
            let begin = writer.log_begin(i).await.unwrap();
            writer.log_insert(i, begin, Bytes::from(format!("data{}", i))).await.unwrap();
            writer.log_commit(i, Lsn::INVALID).await.unwrap();
        }
        writer.close().await.unwrap();

        // Scan all records
        let reader = WalReader::new(dir.path()).await.unwrap();
        let records = reader.scan_all().await.unwrap();

        // Should have 15 records (3 per transaction * 5 transactions)
        assert_eq!(records.len(), 15);
    }

    #[tokio::test]
    async fn test_wal_reader_find_active_transactions() {
        let (writer, dir) = create_test_wal().await;

        // Transaction 1: committed
        let lsn1 = writer.log_begin(1).await.unwrap();
        let lsn2 = writer.log_insert(1, lsn1, Bytes::new()).await.unwrap();
        writer.log_commit(1, lsn2).await.unwrap();

        // Transaction 2: started but not committed
        let lsn3 = writer.log_begin(2).await.unwrap();
        let lsn4 = writer.log_insert(2, lsn3, Bytes::new()).await.unwrap();

        writer.close().await.unwrap();

        // Find active transactions at the end
        let reader = WalReader::new(dir.path()).await.unwrap();
        let active = reader.find_active_transactions(lsn4).await.unwrap();

        assert_eq!(active.len(), 1);
        assert!(active.contains(&2));
    }

    #[tokio::test]
    async fn test_wal_reader_find_checkpoint() {
        let (writer, dir) = create_test_wal().await;

        writer.log_begin(1).await.unwrap();
        writer.log_checkpoint_begin().await.unwrap();
        writer.log_checkpoint_end(Bytes::from_static(b"checkpoint")).await.unwrap();
        writer.log_commit(1, Lsn::INVALID).await.unwrap();
        writer.close().await.unwrap();

        let reader = WalReader::new(dir.path()).await.unwrap();
        let checkpoint = reader.find_last_checkpoint().await.unwrap();

        assert!(checkpoint.is_some());
        let (_, record) = checkpoint.unwrap();
        assert_eq!(record.record_type, LogRecordType::CheckpointEnd);
    }

    #[tokio::test]
    async fn test_recovery_manager_empty() {
        let dir = tempdir().unwrap();
        let recovery = RecoveryManager::new(dir.path()).await.unwrap();
        let result = recovery.recover().await.unwrap();

        assert!(result.is_empty());
    }

    #[tokio::test]
    async fn test_recovery_manager_committed_txns() {
        let (writer, dir) = create_test_wal().await;

        // Committed transaction
        let begin = writer.log_begin(1).await.unwrap();
        let insert = writer.log_insert(1, begin, Bytes::from_static(b"data")).await.unwrap();
        writer.log_commit(1, insert).await.unwrap();
        writer.close().await.unwrap();

        // Recover
        let recovery = RecoveryManager::new(dir.path()).await.unwrap();
        let result = recovery.recover().await.unwrap();

        // Should have 1 redo record (the insert)
        assert_eq!(result.redo_records.len(), 1);
        assert_eq!(result.redo_records[0].record_type, LogRecordType::Insert);
        assert!(result.undo_txns.is_empty());
    }

    #[tokio::test]
    async fn test_recovery_manager_uncommitted_txn() {
        let (writer, dir) = create_test_wal().await;

        // Uncommitted transaction
        let begin = writer.log_begin(1).await.unwrap();
        writer.log_insert(1, begin, Bytes::from_static(b"data")).await.unwrap();
        // No commit!
        writer.close().await.unwrap();

        // Recover
        let recovery = RecoveryManager::new(dir.path()).await.unwrap();
        let result = recovery.recover().await.unwrap();

        // The insert should NOT be in redo (transaction not committed)
        assert!(result.redo_records.is_empty());
        // Transaction 1 should be in undo list
        assert_eq!(result.undo_txns.len(), 1);
        assert!(result.undo_txns.contains(&1));
    }
}
