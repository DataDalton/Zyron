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
    pub fn new(wal_dir: &Path) -> Result<Self> {
        let mut segments = BTreeMap::new();

        if wal_dir.exists() {
            for entry in std::fs::read_dir(wal_dir)? {
                let entry = entry?;
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
    pub fn open_segment(&self, segment_id: SegmentId) -> Result<LogSegment> {
        let path = self.segments.get(&segment_id).ok_or_else(|| {
            ZyronError::WalCorrupted {
                lsn: 0,
                reason: format!("segment {} not found", segment_id),
            }
        })?;

        LogSegment::open(path)
    }

    /// Creates an iterator over all records starting from the given LSN.
    pub fn scan_from(&self, start_lsn: Lsn) -> Result<WalIterator<'_>> {
        WalIterator::new(self, start_lsn)
    }

    /// Creates an iterator over all records.
    pub fn scan_all(&self) -> Result<WalIterator<'_>> {
        let start_lsn = if let Some(first_id) = self.first_segment_id() {
            Lsn::new(first_id.0, SegmentHeader::SIZE as u32)
        } else {
            Lsn::FIRST
        };

        self.scan_from(start_lsn)
    }

    /// Finds the last checkpoint in the WAL.
    pub fn find_last_checkpoint(&self) -> Result<Option<(Lsn, LogRecord)>> {
        let mut last_checkpoint: Option<(Lsn, LogRecord)> = None;

        for record in self.scan_all()? {
            let record = record?;
            if record.record_type == LogRecordType::CheckpointEnd {
                last_checkpoint = Some((record.lsn, record));
            }
        }

        Ok(last_checkpoint)
    }

    /// Collects all active transactions at the given LSN.
    pub fn find_active_transactions(&self, at_lsn: Lsn) -> Result<Vec<u32>> {
        let mut active = std::collections::HashSet::new();

        for record in self.scan_all()? {
            let record = record?;

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
    pub fn refresh(&mut self) -> Result<()> {
        self.segments.clear();

        for entry in std::fs::read_dir(&self.wal_dir)? {
            let entry = entry?;
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

/// Iterator over WAL records across multiple segments.
pub struct WalIterator<'a> {
    reader: &'a WalReader,
    current_segment: Option<LogSegment>,
    current_segment_id: Option<SegmentId>,
    current_offset: u32,
}

impl<'a> WalIterator<'a> {
    /// Creates a new iterator starting from the given LSN.
    fn new(reader: &'a WalReader, start_lsn: Lsn) -> Result<Self> {
        let segment_id = SegmentId(start_lsn.segment_id());
        let offset = start_lsn.offset();

        let current_segment = if reader.segments.contains_key(&segment_id) {
            Some(reader.open_segment(segment_id)?)
        } else {
            None
        };

        Ok(Self {
            reader,
            current_segment,
            current_segment_id: Some(segment_id),
            current_offset: offset,
        })
    }

    /// Advances to the next segment.
    fn advance_segment(&mut self) -> Result<bool> {
        // Close current segment
        if let Some(ref mut segment) = self.current_segment {
            segment.close()?;
        }

        // Find next segment
        let next_id = self.current_segment_id.map(|id| id.next());

        if let Some(id) = next_id {
            if self.reader.segments.contains_key(&id) {
                self.current_segment = Some(self.reader.open_segment(id)?);
                self.current_segment_id = Some(id);
                self.current_offset = SegmentHeader::SIZE as u32;
                return Ok(true);
            }
        }

        self.current_segment = None;
        self.current_segment_id = None;
        Ok(false)
    }
}

impl Iterator for WalIterator<'_> {
    type Item = Result<LogRecord>;

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            let segment = match self.current_segment.as_mut() {
                Some(s) => s,
                None => return None,
            };

            // Check if we've reached the end of this segment
            if self.current_offset >= segment.write_offset() {
                match self.advance_segment() {
                    Ok(true) => continue,
                    Ok(false) => return None,
                    Err(e) => return Some(Err(e)),
                }
            }

            // Try to read a record
            match segment.read_at(self.current_offset) {
                Ok(record) => {
                    self.current_offset += record.size_on_disk() as u32;
                    return Some(Ok(record));
                }
                Err(e) => {
                    // If we hit a corrupted record, try to skip to next segment
                    if matches!(e, ZyronError::WalCorrupted { .. }) {
                        match self.advance_segment() {
                            Ok(true) => continue,
                            Ok(false) => return None,
                            Err(e) => return Some(Err(e)),
                        }
                    }
                    return Some(Err(e));
                }
            }
        }
    }
}

/// Recovery manager for replaying WAL during startup.
pub struct RecoveryManager {
    reader: WalReader,
}

impl RecoveryManager {
    /// Creates a new recovery manager.
    pub fn new(wal_dir: &Path) -> Result<Self> {
        let reader = WalReader::new(wal_dir)?;
        Ok(Self { reader })
    }

    /// Performs recovery, returning the redo and undo information.
    pub fn recover(&self) -> Result<RecoveryResult> {
        // Find the last checkpoint
        let checkpoint = self.reader.find_last_checkpoint()?;

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

        for record in self.reader.scan_from(start_lsn)? {
            let record = record?;
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

    fn create_test_wal() -> (WalWriter, tempfile::TempDir) {
        let dir = tempdir().unwrap();
        let config = WalWriterConfig {
            wal_dir: dir.path().to_path_buf(),
            segment_size: LogSegment::DEFAULT_SIZE,
            fsync_enabled: true,
            group_commit_size: 1,
        };
        let writer = WalWriter::new(config).unwrap();
        (writer, dir)
    }

    #[test]
    fn test_wal_reader_empty() {
        let dir = tempdir().unwrap();
        let reader = WalReader::new(dir.path()).unwrap();
        assert_eq!(reader.segment_count(), 0);
        assert!(reader.first_segment_id().is_none());
    }

    #[test]
    fn test_wal_reader_with_segments() {
        let (writer, dir) = create_test_wal();

        // Write some records
        writer.log_begin(1).unwrap();
        writer.log_commit(1, Lsn::INVALID).unwrap();
        writer.close().unwrap();

        // Read back
        let reader = WalReader::new(dir.path()).unwrap();
        assert_eq!(reader.segment_count(), 1);
        assert_eq!(reader.first_segment_id(), Some(SegmentId::FIRST));
    }

    #[test]
    fn test_wal_reader_scan_all() {
        let (writer, dir) = create_test_wal();

        // Write multiple transactions
        for i in 1..=5 {
            let begin = writer.log_begin(i).unwrap();
            writer.log_insert(i, begin, Bytes::from(format!("data{}", i))).unwrap();
            writer.log_commit(i, Lsn::INVALID).unwrap();
        }
        writer.close().unwrap();

        // Scan all records
        let reader = WalReader::new(dir.path()).unwrap();
        let records: Vec<_> = reader.scan_all().unwrap().collect();

        // Should have 15 records (3 per transaction * 5 transactions)
        assert_eq!(records.len(), 15);
    }

    #[test]
    fn test_wal_reader_find_active_transactions() {
        let (writer, dir) = create_test_wal();

        // Transaction 1: committed
        let lsn1 = writer.log_begin(1).unwrap();
        let lsn2 = writer.log_insert(1, lsn1, Bytes::new()).unwrap();
        writer.log_commit(1, lsn2).unwrap();

        // Transaction 2: started but not committed
        let lsn3 = writer.log_begin(2).unwrap();
        let lsn4 = writer.log_insert(2, lsn3, Bytes::new()).unwrap();

        writer.close().unwrap();

        // Find active transactions at the end
        let reader = WalReader::new(dir.path()).unwrap();
        let active = reader.find_active_transactions(lsn4).unwrap();

        assert_eq!(active.len(), 1);
        assert!(active.contains(&2));
    }

    #[test]
    fn test_wal_reader_find_checkpoint() {
        let (writer, dir) = create_test_wal();

        writer.log_begin(1).unwrap();
        writer.log_checkpoint_begin().unwrap();
        writer.log_checkpoint_end(Bytes::from_static(b"checkpoint")).unwrap();
        writer.log_commit(1, Lsn::INVALID).unwrap();
        writer.close().unwrap();

        let reader = WalReader::new(dir.path()).unwrap();
        let checkpoint = reader.find_last_checkpoint().unwrap();

        assert!(checkpoint.is_some());
        let (_, record) = checkpoint.unwrap();
        assert_eq!(record.record_type, LogRecordType::CheckpointEnd);
    }

    #[test]
    fn test_recovery_manager_empty() {
        let dir = tempdir().unwrap();
        let recovery = RecoveryManager::new(dir.path()).unwrap();
        let result = recovery.recover().unwrap();

        assert!(result.is_empty());
    }

    #[test]
    fn test_recovery_manager_committed_txns() {
        let (writer, dir) = create_test_wal();

        // Committed transaction
        let begin = writer.log_begin(1).unwrap();
        let insert = writer.log_insert(1, begin, Bytes::from_static(b"data")).unwrap();
        writer.log_commit(1, insert).unwrap();
        writer.close().unwrap();

        // Recover
        let recovery = RecoveryManager::new(dir.path()).unwrap();
        let result = recovery.recover().unwrap();

        // Should have 1 redo record (the insert)
        assert_eq!(result.redo_records.len(), 1);
        assert_eq!(result.redo_records[0].record_type, LogRecordType::Insert);
        assert!(result.undo_txns.is_empty());
    }

    #[test]
    fn test_recovery_manager_uncommitted_txn() {
        let (writer, dir) = create_test_wal();

        // Uncommitted transaction
        let begin = writer.log_begin(1).unwrap();
        writer.log_insert(1, begin, Bytes::from_static(b"data")).unwrap();
        // No commit!
        writer.close().unwrap();

        // Recover
        let recovery = RecoveryManager::new(dir.path()).unwrap();
        let result = recovery.recover().unwrap();

        // The insert should NOT be in redo (transaction not committed)
        assert!(result.redo_records.is_empty());
        // Transaction 1 should be in undo list
        assert_eq!(result.undo_txns.len(), 1);
        assert!(result.undo_txns.contains(&1));
    }
}
