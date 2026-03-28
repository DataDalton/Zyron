//! Change Data Feed (CDF) storage for per-table row-level change tracking.
//!
//! Each CDF-enabled table gets a separate .zycdf file that records every
//! insert, update (preimage + postimage), delete, schema change, and truncate.
//! Records are length-prefixed with checksums for crash recovery.
//!
//! File format (aligned with .zyr columnar pattern):
//!   File header: magic (8) + format_version (u32) + table_id (u32) +
//!                header_checksum (u32) = 20 bytes
//!   Records:     [u32 record_len][record_bytes][u32 checksum] ...
//!
//! table_id is stored once in the file header (not per-record), matching
//! the .zyr pattern where per-file metadata lives in the header. On read,
//! table_id is populated from the header into each ChangeRecord.
//!
//! Concurrency model: a single Mutex covers both the file writer and the
//! in-memory index. Writes are serialized. Reads snapshot the index under
//! the lock, then perform file I/O outside the lock.

use std::fs::{self, File, OpenOptions};
use std::io::{BufWriter, Read, Write};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use parking_lot::Mutex;
use scc::HashMap as SccHashMap;
use serde::{Deserialize, Serialize};
use zyron_common::{Result, ZyronError};

/// Maximum allowed record size (64 MB). Prevents OOM from corrupt files.
const MAX_RECORD_SIZE: u64 = 64 * 1024 * 1024;

/// File header: magic (8) + format_version (4) + table_id (4) + header_checksum (4) = 20 bytes.
const FILE_HEADER_SIZE: usize = 20;
const FILE_MAGIC: &[u8; 8] = b"ZYCDF\0\0\0";
const FORMAT_VERSION: u32 = 1;

// ---------------------------------------------------------------------------
// ChangeType
// ---------------------------------------------------------------------------

/// Type of change captured in a CDF record.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[repr(u8)]
pub enum ChangeType {
    Insert = 0,
    UpdatePreimage = 1,
    UpdatePostimage = 2,
    Delete = 3,
    SchemaChange = 4,
    Truncate = 5,
}

impl ChangeType {
    pub(crate) fn from_u8(v: u8) -> Result<Self> {
        match v {
            0 => Ok(Self::Insert),
            1 => Ok(Self::UpdatePreimage),
            2 => Ok(Self::UpdatePostimage),
            3 => Ok(Self::Delete),
            4 => Ok(Self::SchemaChange),
            5 => Ok(Self::Truncate),
            _ => Err(ZyronError::CdcDecoderError(format!(
                "unknown change type: {v}"
            ))),
        }
    }
}

// ---------------------------------------------------------------------------
// ChangeRecord
// ---------------------------------------------------------------------------

/// A single change record stored in a .zycdf file.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChangeRecord {
    pub change_type: ChangeType,
    pub commit_version: u64,
    pub commit_timestamp: i64,
    pub table_id: u32,
    pub txn_id: u32,
    pub schema_version: u32,
    pub row_data: Vec<u8>,
    pub primary_key_data: Vec<u8>,
    pub is_last_in_txn: bool,
}

// Per-record on-disk format (no table_id, that's in the file header):
//   [u32 record_len][record_bytes][u32 checksum]
//
// Record binary layout (all little-endian):
//   change_type:      u8
//   commit_version:   u64  (8 bytes)
//   commit_timestamp: i64  (8 bytes)
//   txn_id:           u32  (4 bytes)
//   schema_version:   u32  (4 bytes)
//   is_last_in_txn:   u8
//   row_data_len:     u32  (4 bytes)
//   row_data:         [u8; row_data_len]
//   pk_data_len:      u32  (4 bytes)
//   primary_key_data: [u8; pk_data_len]
// Fixed header = 34 bytes (was 38 before removing table_id).

const RECORD_FRAME_PREFIX: usize = 4; // u32 length prefix
const RECORD_FRAME_SUFFIX: usize = 4; // u32 checksum
const BINARY_FIXED_HEADER: usize = 34;

impl ChangeRecord {
    /// Serializes to packed binary format (table_id omitted, stored in file header).
    fn serialize(&self) -> Vec<u8> {
        let total = BINARY_FIXED_HEADER + self.row_data.len() + self.primary_key_data.len();
        let mut buf = Vec::with_capacity(total);
        buf.push(self.change_type as u8);
        buf.extend_from_slice(&self.commit_version.to_le_bytes());
        buf.extend_from_slice(&self.commit_timestamp.to_le_bytes());
        // table_id NOT written: stored once in file header
        buf.extend_from_slice(&self.txn_id.to_le_bytes());
        buf.extend_from_slice(&self.schema_version.to_le_bytes());
        buf.push(if self.is_last_in_txn { 1 } else { 0 });
        buf.extend_from_slice(&(self.row_data.len() as u32).to_le_bytes());
        buf.extend_from_slice(&self.row_data);
        buf.extend_from_slice(&(self.primary_key_data.len() as u32).to_le_bytes());
        buf.extend_from_slice(&self.primary_key_data);
        buf
    }

    /// Deserializes from packed binary format. table_id is supplied from the
    /// file header, not from the record bytes.
    fn deserialize(data: &[u8], table_id: u32) -> Result<Self> {
        if data.len() < BINARY_FIXED_HEADER {
            return Err(ZyronError::CdcDecoderError(
                "change record too short for header".into(),
            ));
        }
        let mut off = 0usize;

        let change_type = ChangeType::from_u8(data[off])?;
        off += 1;

        let commit_version = u64::from_le_bytes(
            data[off..off + 8]
                .try_into()
                .map_err(|_| ZyronError::CdcDecoderError("bad commit_version".into()))?,
        );
        off += 8;

        let commit_timestamp = i64::from_le_bytes(
            data[off..off + 8]
                .try_into()
                .map_err(|_| ZyronError::CdcDecoderError("bad commit_timestamp".into()))?,
        );
        off += 8;

        let txn_id = u32::from_le_bytes(
            data[off..off + 4]
                .try_into()
                .map_err(|_| ZyronError::CdcDecoderError("bad txn_id".into()))?,
        );
        off += 4;

        let schema_version = u32::from_le_bytes(
            data[off..off + 4]
                .try_into()
                .map_err(|_| ZyronError::CdcDecoderError("bad schema_version".into()))?,
        );
        off += 4;

        let is_last_in_txn = data[off] != 0;
        off += 1;

        if off + 4 > data.len() {
            return Err(ZyronError::CdcDecoderError("truncated row_data_len".into()));
        }
        let row_data_len = u32::from_le_bytes(
            data[off..off + 4]
                .try_into()
                .map_err(|_| ZyronError::CdcDecoderError("bad row_data_len".into()))?,
        ) as usize;
        off += 4;

        if off + row_data_len > data.len() {
            return Err(ZyronError::CdcDecoderError("truncated row_data".into()));
        }
        let row_data = data[off..off + row_data_len].to_vec();
        off += row_data_len;

        if off + 4 > data.len() {
            return Err(ZyronError::CdcDecoderError("truncated pk_data_len".into()));
        }
        let pk_data_len = u32::from_le_bytes(
            data[off..off + 4]
                .try_into()
                .map_err(|_| ZyronError::CdcDecoderError("bad pk_data_len".into()))?,
        ) as usize;
        off += 4;

        if off + pk_data_len > data.len() {
            return Err(ZyronError::CdcDecoderError(
                "truncated primary_key_data".into(),
            ));
        }
        let primary_key_data = data[off..off + pk_data_len].to_vec();

        Ok(Self {
            change_type,
            commit_version,
            commit_timestamp,
            table_id,
            txn_id,
            schema_version,
            row_data,
            primary_key_data,
            is_last_in_txn,
        })
    }

    /// Reads version and timestamp from the first 17 bytes of a serialized
    /// record without deserializing the variable-length fields.
    fn peek_version_timestamp(data: &[u8]) -> Result<(u64, i64)> {
        if data.len() < 17 {
            return Err(ZyronError::CdcDecoderError("too short for peek".into()));
        }
        let version = u64::from_le_bytes(
            data[1..9]
                .try_into()
                .map_err(|_| ZyronError::CdcDecoderError("bad peek version".into()))?,
        );
        let timestamp = i64::from_le_bytes(
            data[9..17]
                .try_into()
                .map_err(|_| ZyronError::CdcDecoderError("bad peek timestamp".into()))?,
        );
        Ok((version, timestamp))
    }
}

// ---------------------------------------------------------------------------
// File header
// ---------------------------------------------------------------------------

/// Writes the .zycdf file header: magic + format_version + table_id + checksum.
fn write_file_header(w: &mut impl Write, table_id: u32) -> Result<()> {
    let mut header = [0u8; FILE_HEADER_SIZE];
    header[0..8].copy_from_slice(FILE_MAGIC);
    header[8..12].copy_from_slice(&FORMAT_VERSION.to_le_bytes());
    header[12..16].copy_from_slice(&table_id.to_le_bytes());
    let checksum = zyron_wal::data_checksum(&header[0..16]);
    header[16..20].copy_from_slice(&checksum.to_le_bytes());
    w.write_all(&header)?;
    Ok(())
}

/// Reads and validates the .zycdf file header. Returns the table_id.
fn read_file_header(data: &[u8]) -> Result<u32> {
    if data.len() < FILE_HEADER_SIZE {
        return Err(ZyronError::CdcDecoderError(
            "file too small for header".into(),
        ));
    }
    if &data[0..8] != FILE_MAGIC {
        return Err(ZyronError::CdcDecoderError("bad CDF magic bytes".into()));
    }
    let _version = u32::from_le_bytes(data[8..12].try_into().unwrap_or([0; 4]));
    let table_id = u32::from_le_bytes(data[12..16].try_into().unwrap_or([0; 4]));
    let stored_checksum = u32::from_le_bytes(data[16..20].try_into().unwrap_or([0; 4]));
    let computed = zyron_wal::data_checksum(&data[0..16]);
    if stored_checksum != computed {
        return Err(ZyronError::CdcDecoderError(
            "CDF header checksum mismatch".into(),
        ));
    }
    Ok(table_id)
}

/// Writes a single framed record (len + data + checksum) into a writer.
fn write_record(w: &mut impl Write, data: &[u8], checksum: u32) -> Result<u64> {
    let len = data.len() as u32;
    w.write_all(&len.to_le_bytes())?;
    w.write_all(data)?;
    w.write_all(&checksum.to_le_bytes())?;
    Ok(RECORD_FRAME_PREFIX as u64 + data.len() as u64 + RECORD_FRAME_SUFFIX as u64)
}

// ---------------------------------------------------------------------------
// CdfIndexEntry - flat, cache-friendly, 24 bytes
// ---------------------------------------------------------------------------

#[derive(Clone, Copy)]
struct CdfIndexEntry {
    version: u64,
    timestamp: i64,
    offset: u64,
}

// ---------------------------------------------------------------------------
// Inner state protected by a single Mutex
// ---------------------------------------------------------------------------

struct CdfInner {
    writer: Option<BufWriter<File>>,
    index: Vec<CdfIndexEntry>,
    file_size: u64,
}

// ---------------------------------------------------------------------------
// ChangeDataFeed
// ---------------------------------------------------------------------------

/// Per-table change data feed with append-only file storage.
pub struct ChangeDataFeed {
    pub table_id: u32,
    enabled: bool,
    pub retention_days: u32,
    file_path: PathBuf,
    inner: Mutex<CdfInner>,
    record_count_atomic: AtomicU64,
    file_size_atomic: AtomicU64,
}

impl ChangeDataFeed {
    /// Opens or creates a CDF file. Validates file header, performs crash
    /// recovery by verifying checksums and truncating at the last valid record.
    pub fn open(data_dir: &Path, table_id: u32, retention_days: u32) -> Result<Self> {
        let cdf_dir = data_dir.join("cdf");
        fs::create_dir_all(&cdf_dir).map_err(|e| {
            ZyronError::CdcStreamError(format!("failed to create cdf directory: {e}"))
        })?;

        let file_path = cdf_dir.join(format!("{table_id:08}.zycdf"));
        let mut index: Vec<CdfIndexEntry> = Vec::new();
        let mut valid_end: u64 = FILE_HEADER_SIZE as u64;

        if file_path.exists() {
            let mut data = Vec::new();
            File::open(&file_path)?.read_to_end(&mut data)?;

            if data.len() >= FILE_HEADER_SIZE {
                let header_table_id = read_file_header(&data)?;
                if header_table_id != table_id {
                    return Err(ZyronError::CdcDecoderError(format!(
                        "CDF file table_id mismatch: header has {header_table_id}, expected {table_id}"
                    )));
                }

                let file_len = data.len() as u64;
                let mut offset = FILE_HEADER_SIZE as u64;

                while offset + (RECORD_FRAME_PREFIX + RECORD_FRAME_SUFFIX) as u64 <= file_len {
                    let o = offset as usize;
                    if o + 4 > data.len() {
                        break;
                    }
                    let record_len =
                        u32::from_le_bytes(data[o..o + 4].try_into().unwrap_or([0; 4])) as u64;

                    if record_len > MAX_RECORD_SIZE {
                        break;
                    }

                    let total =
                        RECORD_FRAME_PREFIX as u64 + record_len + RECORD_FRAME_SUFFIX as u64;
                    if offset + total > file_len {
                        break;
                    }

                    let record_start = o + RECORD_FRAME_PREFIX;
                    let record_end = record_start + record_len as usize;
                    let crc_end = record_end + RECORD_FRAME_SUFFIX;

                    if crc_end > data.len() {
                        break;
                    }

                    let record_data = &data[record_start..record_end];
                    let stored_crc =
                        u32::from_le_bytes(data[record_end..crc_end].try_into().unwrap_or([0; 4]));
                    let computed_crc = zyron_wal::data_checksum(record_data);
                    if stored_crc != computed_crc {
                        break;
                    }

                    match ChangeRecord::peek_version_timestamp(record_data) {
                        Ok((version, timestamp)) => {
                            index.push(CdfIndexEntry {
                                version,
                                timestamp,
                                offset,
                            });
                        }
                        Err(_) => break,
                    }

                    valid_end = offset + total;
                    offset = valid_end;
                }

                if valid_end < file_len {
                    let file = OpenOptions::new().write(true).open(&file_path)?;
                    file.set_len(valid_end)?;
                }
            } else {
                // File exists but too small for header, overwrite.
                valid_end = 0;
            }
        } else {
            valid_end = 0;
        }

        // Create file with header if it doesn't exist or was too small.
        if valid_end == 0 {
            let mut file = File::create(&file_path)?;
            write_file_header(&mut file, table_id)?;
            file.sync_all()?;
            valid_end = FILE_HEADER_SIZE as u64;
        }

        let record_count = index.len() as u64;
        Ok(Self {
            table_id,
            enabled: true,
            retention_days,
            file_path,
            inner: Mutex::new(CdfInner {
                writer: None,
                index,
                file_size: valid_end,
            }),
            record_count_atomic: AtomicU64::new(record_count),
            file_size_atomic: AtomicU64::new(valid_end),
        })
    }

    pub fn is_enabled(&self) -> bool {
        self.enabled
    }

    pub fn enable(&mut self) {
        self.enabled = true;
    }

    pub fn disable(&mut self) {
        self.enabled = false;
    }

    fn open_writer(inner: &mut CdfInner, path: &Path) -> Result<()> {
        if inner.writer.is_none() {
            let file = OpenOptions::new().create(true).append(true).open(path)?;
            inner.writer = Some(BufWriter::new(file));
        }
        Ok(())
    }

    /// Appends a single change record.
    pub fn append_change(&self, record: &ChangeRecord) -> Result<()> {
        if !self.enabled {
            return Ok(());
        }

        let data = record.serialize();
        let checksum = zyron_wal::data_checksum(&data);

        let mut inner = self.inner.lock();
        Self::open_writer(&mut inner, &self.file_path)?;

        let offset = inner.file_size;
        let writer = inner
            .writer
            .as_mut()
            .ok_or_else(|| ZyronError::CdcStreamError("CDF writer not initialized".into()))?;
        let bytes_written = write_record(writer, &data, checksum)?;
        writer.flush()?;

        inner.file_size += bytes_written;
        inner.index.push(CdfIndexEntry {
            version: record.commit_version,
            timestamp: record.commit_timestamp,
            offset,
        });

        self.file_size_atomic
            .store(inner.file_size, Ordering::Release);
        self.record_count_atomic
            .store(inner.index.len() as u64, Ordering::Release);

        Ok(())
    }

    /// Appends multiple change records in a single lock acquisition.
    pub fn append_batch(&self, records: &[ChangeRecord]) -> Result<()> {
        if !self.enabled || records.is_empty() {
            return Ok(());
        }

        // Pre-serialize outside the lock: one contiguous buffer for all records.
        let mut batch_buf: Vec<u8> = Vec::with_capacity(records.len() * 64);
        let mut meta: Vec<(usize, u64, i64)> = Vec::with_capacity(records.len());

        for record in records {
            let start = batch_buf.len();
            let data = record.serialize();
            let checksum = zyron_wal::data_checksum(&data);
            batch_buf.extend_from_slice(&(data.len() as u32).to_le_bytes());
            batch_buf.extend_from_slice(&data);
            batch_buf.extend_from_slice(&checksum.to_le_bytes());
            meta.push((start, record.commit_version, record.commit_timestamp));
        }

        let mut inner = self.inner.lock();
        Self::open_writer(&mut inner, &self.file_path)?;

        if inner.writer.is_none() {
            return Err(ZyronError::CdcStreamError(
                "CDF writer not initialized".into(),
            ));
        }

        let writer = inner.writer.as_mut().unwrap();
        writer.write_all(&batch_buf)?;
        writer.flush()?;

        let base_offset = inner.file_size;
        inner.index.reserve(records.len());
        for &(start, version, timestamp) in &meta {
            inner.index.push(CdfIndexEntry {
                version,
                timestamp,
                offset: base_offset + start as u64,
            });
        }
        inner.file_size += batch_buf.len() as u64;

        self.file_size_atomic
            .store(inner.file_size, Ordering::Release);
        self.record_count_atomic
            .store(inner.index.len() as u64, Ordering::Release);

        Ok(())
    }

    /// Queries change records by version range [start_version, end_version].
    pub fn query_changes(&self, start_version: u64, end_version: u64) -> Result<Vec<ChangeRecord>> {
        let offsets = {
            let inner = self.inner.lock();
            let mut result = Vec::new();
            for entry in &inner.index {
                if entry.version >= start_version && entry.version <= end_version {
                    result.push(entry.offset);
                }
            }
            result
        };

        self.read_records_bulk(&offsets)
    }

    /// Queries change records by timestamp range [start_ts, end_ts].
    pub fn query_changes_by_time(&self, start_ts: i64, end_ts: i64) -> Result<Vec<ChangeRecord>> {
        let offsets = {
            let inner = self.inner.lock();
            let mut result = Vec::new();
            for entry in &inner.index {
                if entry.timestamp >= start_ts && entry.timestamp <= end_ts {
                    result.push(entry.offset);
                }
            }
            result
        };

        self.read_records_bulk(&offsets)
    }

    /// Returns the maximum version at or below the given timestamp.
    pub fn max_version_before_time(&self, cutoff_ts: i64) -> Option<u64> {
        let inner = self.inner.lock();
        let mut max_ver: Option<u64> = None;
        for entry in inner.index.iter().rev() {
            if entry.timestamp <= cutoff_ts {
                max_ver = Some(entry.version);
                break;
            }
        }
        max_ver
    }

    /// Bulk-reads records by reading the entire file once and parsing at offsets.
    /// table_id is populated from self.table_id (from the file header).
    fn read_records_bulk(&self, offsets: &[u64]) -> Result<Vec<ChangeRecord>> {
        if offsets.is_empty() {
            return Ok(Vec::new());
        }

        let mut file_data = Vec::new();
        File::open(&self.file_path)?.read_to_end(&mut file_data)?;

        let mut results = Vec::with_capacity(offsets.len());
        for &offset in offsets {
            let o = offset as usize;
            if o + 4 > file_data.len() {
                return Err(ZyronError::CdcDecoderError(format!(
                    "offset {offset} beyond file end"
                )));
            }
            let record_len =
                u32::from_le_bytes(file_data[o..o + 4].try_into().unwrap_or([0; 4])) as usize;
            let data_start = o + RECORD_FRAME_PREFIX;
            let data_end = data_start + record_len;
            if data_end + RECORD_FRAME_SUFFIX > file_data.len() {
                return Err(ZyronError::CdcDecoderError(format!(
                    "record at offset {offset} truncated"
                )));
            }

            let record_data = &file_data[data_start..data_end];
            let stored_crc = u32::from_le_bytes(
                file_data[data_end..data_end + 4]
                    .try_into()
                    .unwrap_or([0; 4]),
            );
            let computed_crc = zyron_wal::data_checksum(record_data);
            if stored_crc != computed_crc {
                return Err(ZyronError::CdcDecoderError(format!(
                    "checksum mismatch at offset {offset}"
                )));
            }

            results.push(ChangeRecord::deserialize(record_data, self.table_id)?);
        }

        Ok(results)
    }

    /// Purges records with commit_version < min_version.
    pub fn purge_before_version(&self, min_version: u64) -> Result<u64> {
        let mut inner = self.inner.lock();

        let keep_offsets: Vec<u64> = inner
            .index
            .iter()
            .filter(|e| e.version >= min_version)
            .map(|e| e.offset)
            .collect();

        let old_count = inner.index.len() as u64;
        let purged = old_count.saturating_sub(keep_offsets.len() as u64);

        if purged == 0 {
            return Ok(0);
        }

        // Read retained records from file in bulk.
        let records = if !keep_offsets.is_empty() {
            let mut file_data = Vec::new();
            File::open(&self.file_path)?.read_to_end(&mut file_data)?;

            let mut result = Vec::with_capacity(keep_offsets.len());
            for &offset in &keep_offsets {
                let o = offset as usize;
                if o + 4 > file_data.len() {
                    break;
                }
                let record_len =
                    u32::from_le_bytes(file_data[o..o + 4].try_into().unwrap_or([0; 4])) as usize;
                let data_start = o + RECORD_FRAME_PREFIX;
                let data_end = data_start + record_len;
                if data_end > file_data.len() {
                    break;
                }
                result.push(ChangeRecord::deserialize(
                    &file_data[data_start..data_end],
                    self.table_id,
                )?);
            }
            result
        } else {
            Vec::new()
        };

        // Close writer before rewrite.
        inner.writer = None;

        // Write to temp with header, rename.
        let tmp_path = self.file_path.with_extension("zycdf.tmp");
        let mut new_index: Vec<CdfIndexEntry> = Vec::with_capacity(records.len());
        let mut new_file_size: u64 = FILE_HEADER_SIZE as u64;

        {
            let tmp = File::create(&tmp_path)?;
            let mut writer = BufWriter::new(tmp);

            write_file_header(&mut writer, self.table_id)?;

            for record in &records {
                let data = record.serialize();
                let checksum = zyron_wal::data_checksum(&data);
                let offset = new_file_size;
                new_file_size += write_record(&mut writer, &data, checksum)?;
                new_index.push(CdfIndexEntry {
                    version: record.commit_version,
                    timestamp: record.commit_timestamp,
                    offset,
                });
            }

            writer.flush()?;
            writer.get_ref().sync_all()?;
        }

        fs::rename(&tmp_path, &self.file_path)?;

        inner.index = new_index;
        inner.file_size = new_file_size;

        self.file_size_atomic
            .store(new_file_size, Ordering::Release);
        self.record_count_atomic
            .store(inner.index.len() as u64, Ordering::Release);

        Ok(purged)
    }

    pub fn record_count(&self) -> u64 {
        self.record_count_atomic.load(Ordering::Acquire)
    }

    pub fn file_size_bytes(&self) -> u64 {
        self.file_size_atomic.load(Ordering::Acquire)
    }
}

// ---------------------------------------------------------------------------
// CdfRegistry
// ---------------------------------------------------------------------------

/// Global registry of change data feeds, one per CDF-enabled table.
pub struct CdfRegistry {
    feeds: SccHashMap<u32, Arc<ChangeDataFeed>>,
    data_dir: PathBuf,
}

impl CdfRegistry {
    pub fn new(data_dir: PathBuf) -> Self {
        Self {
            feeds: SccHashMap::new(),
            data_dir,
        }
    }

    pub fn enable_for_table(
        &self,
        table_id: u32,
        retention_days: u32,
    ) -> Result<Arc<ChangeDataFeed>> {
        let feed = Arc::new(ChangeDataFeed::open(
            &self.data_dir,
            table_id,
            retention_days,
        )?);
        let _ = self.feeds.insert_sync(table_id, feed.clone());
        Ok(feed)
    }

    pub fn disable_for_table(&self, table_id: u32, purge: bool) -> Result<()> {
        if let Some((_, _feed)) = self.feeds.remove_sync(&table_id) {
            if purge {
                let file_path = self
                    .data_dir
                    .join("cdf")
                    .join(format!("{table_id:08}.zycdf"));
                if file_path.exists() {
                    fs::remove_file(&file_path)?;
                }
            }
        }
        Ok(())
    }

    pub fn get_feed(&self, table_id: u32) -> Option<Arc<ChangeDataFeed>> {
        let mut result = None;
        self.feeds.read_sync(&table_id, |_k, v| {
            result = Some(v.clone());
        });
        result
    }

    pub fn remove_table(&self, table_id: u32) -> Result<()> {
        self.disable_for_table(table_id, true)
    }

    pub fn list_feeds(&self) -> Vec<(u32, u64, u64, u32)> {
        let mut result = Vec::new();
        self.feeds.iter_sync(|table_id, feed| {
            result.push((
                *table_id,
                feed.record_count(),
                feed.file_size_bytes(),
                feed.retention_days,
            ));
            true
        });
        result
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    fn make_record(version: u64, ts: i64, change_type: ChangeType) -> ChangeRecord {
        ChangeRecord {
            change_type,
            commit_version: version,
            commit_timestamp: ts,
            table_id: 1,
            txn_id: 100,
            schema_version: 1,
            row_data: vec![1, 2, 3, 4],
            primary_key_data: vec![1],
            is_last_in_txn: true,
        }
    }

    #[test]
    fn test_change_type_roundtrip() {
        for v in 0..=5u8 {
            let ct = ChangeType::from_u8(v).unwrap();
            assert_eq!(ct as u8, v);
        }
        assert!(ChangeType::from_u8(99).is_err());
    }

    #[test]
    fn test_change_record_serde() {
        let record = make_record(1, 1000, ChangeType::Insert);
        let data = record.serialize();
        let decoded = ChangeRecord::deserialize(&data, 1).unwrap();
        assert_eq!(decoded.commit_version, 1);
        assert_eq!(decoded.commit_timestamp, 1000);
        assert_eq!(decoded.change_type, ChangeType::Insert);
        assert_eq!(decoded.row_data, vec![1, 2, 3, 4]);
        assert_eq!(decoded.table_id, 1);
    }

    #[test]
    fn test_file_header_roundtrip() {
        let mut buf = Vec::new();
        write_file_header(&mut buf, 42).unwrap();
        assert_eq!(buf.len(), FILE_HEADER_SIZE);
        let table_id = read_file_header(&buf).unwrap();
        assert_eq!(table_id, 42);
    }

    #[test]
    fn test_file_header_corruption_detected() {
        let mut buf = Vec::new();
        write_file_header(&mut buf, 42).unwrap();
        buf[14] ^= 0xFF; // corrupt table_id byte
        assert!(read_file_header(&buf).is_err());
    }

    #[test]
    fn test_open_and_append() {
        let tmp = TempDir::new().unwrap();
        let feed = ChangeDataFeed::open(tmp.path(), 1, 30).unwrap();

        let record = make_record(1, 1000, ChangeType::Insert);
        feed.append_change(&record).unwrap();
        assert_eq!(feed.record_count(), 1);
        assert!(feed.file_size_bytes() > FILE_HEADER_SIZE as u64);

        let results = feed.query_changes(0, 10).unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].commit_version, 1);
        assert_eq!(results[0].table_id, 1);
    }

    #[test]
    fn test_append_batch() {
        let tmp = TempDir::new().unwrap();
        let feed = ChangeDataFeed::open(tmp.path(), 1, 30).unwrap();

        let records: Vec<ChangeRecord> = (1..=5)
            .map(|i| make_record(i, i as i64 * 1000, ChangeType::Insert))
            .collect();
        feed.append_batch(&records).unwrap();
        assert_eq!(feed.record_count(), 5);

        let results = feed.query_changes(2, 4).unwrap();
        assert_eq!(results.len(), 3);
    }

    #[test]
    fn test_query_by_time() {
        let tmp = TempDir::new().unwrap();
        let feed = ChangeDataFeed::open(tmp.path(), 1, 30).unwrap();

        let records: Vec<ChangeRecord> = (1..=5)
            .map(|i| make_record(i, i as i64 * 1000, ChangeType::Insert))
            .collect();
        feed.append_batch(&records).unwrap();

        let results = feed.query_changes_by_time(2000, 4000).unwrap();
        assert_eq!(results.len(), 3);
    }

    #[test]
    fn test_crash_recovery_truncates_torn_write() {
        let tmp = TempDir::new().unwrap();

        {
            let feed = ChangeDataFeed::open(tmp.path(), 1, 30).unwrap();
            feed.append_change(&make_record(1, 1000, ChangeType::Insert))
                .unwrap();
        }

        // Append garbage to simulate torn write.
        let file_path = tmp.path().join("cdf").join("00000001.zycdf");
        {
            let mut f = OpenOptions::new().append(true).open(&file_path).unwrap();
            f.write_all(&[0xFF; 20]).unwrap();
        }

        let feed = ChangeDataFeed::open(tmp.path(), 1, 30).unwrap();
        assert_eq!(feed.record_count(), 1);
        let results = feed.query_changes(0, 10).unwrap();
        assert_eq!(results.len(), 1);
    }

    #[test]
    fn test_purge_before_version() {
        let tmp = TempDir::new().unwrap();
        let feed = ChangeDataFeed::open(tmp.path(), 1, 30).unwrap();

        let records: Vec<ChangeRecord> = (1..=10)
            .map(|i| make_record(i, i as i64 * 1000, ChangeType::Insert))
            .collect();
        feed.append_batch(&records).unwrap();
        assert_eq!(feed.record_count(), 10);

        let purged = feed.purge_before_version(6).unwrap();
        assert_eq!(purged, 5);
        assert_eq!(feed.record_count(), 5);

        let results = feed.query_changes(1, 10).unwrap();
        assert_eq!(results.len(), 5);
        assert_eq!(results[0].commit_version, 6);
    }

    #[test]
    fn test_purge_nothing_returns_zero() {
        let tmp = TempDir::new().unwrap();
        let feed = ChangeDataFeed::open(tmp.path(), 1, 30).unwrap();

        let records: Vec<ChangeRecord> = (5..=10)
            .map(|i| make_record(i, i as i64 * 1000, ChangeType::Insert))
            .collect();
        feed.append_batch(&records).unwrap();

        let purged = feed.purge_before_version(1).unwrap();
        assert_eq!(purged, 0);
        assert_eq!(feed.record_count(), 6);
    }

    #[test]
    fn test_disabled_feed_skips_append() {
        let tmp = TempDir::new().unwrap();
        let mut feed = ChangeDataFeed::open(tmp.path(), 1, 30).unwrap();
        feed.disable();

        feed.append_change(&make_record(1, 1000, ChangeType::Insert))
            .unwrap();
        assert_eq!(feed.record_count(), 0);
    }

    #[test]
    fn test_registry_enable_disable() {
        let tmp = TempDir::new().unwrap();
        let registry = CdfRegistry::new(tmp.path().to_path_buf());

        let feed = registry.enable_for_table(42, 30).unwrap();
        feed.append_change(&make_record(1, 1000, ChangeType::Insert))
            .unwrap();

        assert!(registry.get_feed(42).is_some());
        assert!(registry.get_feed(99).is_none());

        registry.disable_for_table(42, false).unwrap();
        assert!(registry.get_feed(42).is_none());
    }

    #[test]
    fn test_registry_remove_table_purges_file() {
        let tmp = TempDir::new().unwrap();
        let registry = CdfRegistry::new(tmp.path().to_path_buf());

        let feed = registry.enable_for_table(42, 30).unwrap();
        feed.append_change(&make_record(1, 1000, ChangeType::Insert))
            .unwrap();

        let file_path = tmp.path().join("cdf").join("00000042.zycdf");
        assert!(file_path.exists());

        registry.remove_table(42).unwrap();
        assert!(!file_path.exists());
    }

    #[test]
    fn test_update_preimage_postimage_pair() {
        let tmp = TempDir::new().unwrap();
        let feed = ChangeDataFeed::open(tmp.path(), 1, 30).unwrap();

        let pre = ChangeRecord {
            change_type: ChangeType::UpdatePreimage,
            commit_version: 5,
            commit_timestamp: 5000,
            table_id: 1,
            txn_id: 200,
            schema_version: 1,
            row_data: vec![10, 20],
            primary_key_data: vec![1],
            is_last_in_txn: false,
        };
        let post = ChangeRecord {
            change_type: ChangeType::UpdatePostimage,
            commit_version: 5,
            commit_timestamp: 5000,
            table_id: 1,
            txn_id: 200,
            schema_version: 1,
            row_data: vec![10, 30],
            primary_key_data: vec![1],
            is_last_in_txn: true,
        };

        feed.append_batch(&[pre, post]).unwrap();
        let results = feed.query_changes(5, 5).unwrap();
        assert_eq!(results.len(), 2);
        assert_eq!(results[0].change_type, ChangeType::UpdatePreimage);
        assert_eq!(results[1].change_type, ChangeType::UpdatePostimage);
    }

    #[test]
    fn test_max_version_before_time() {
        let tmp = TempDir::new().unwrap();
        let feed = ChangeDataFeed::open(tmp.path(), 1, 30).unwrap();

        let records: Vec<ChangeRecord> = (1..=5)
            .map(|i| make_record(i, i as i64 * 1000, ChangeType::Insert))
            .collect();
        feed.append_batch(&records).unwrap();

        assert_eq!(feed.max_version_before_time(3000), Some(3));
        assert_eq!(feed.max_version_before_time(5000), Some(5));
        assert_eq!(feed.max_version_before_time(500), None);
    }

    #[test]
    fn test_reopen_reads_header() {
        let tmp = TempDir::new().unwrap();

        {
            let feed = ChangeDataFeed::open(tmp.path(), 42, 30).unwrap();
            feed.append_change(&make_record(1, 1000, ChangeType::Insert))
                .unwrap();
        }

        // Reopen and verify table_id comes from header.
        let feed = ChangeDataFeed::open(tmp.path(), 42, 30).unwrap();
        assert_eq!(feed.record_count(), 1);
        let results = feed.query_changes(0, 10).unwrap();
        assert_eq!(results[0].table_id, 42);
    }

    #[test]
    fn test_table_id_mismatch_detected() {
        let tmp = TempDir::new().unwrap();
        let cdf_dir = tmp.path().join("cdf");
        fs::create_dir_all(&cdf_dir).unwrap();

        // Manually write a file header for table 42 at the path where
        // table 99 would look. This simulates a corrupted or misplaced file.
        let wrong_path = cdf_dir.join("00000099.zycdf");
        {
            let mut f = File::create(&wrong_path).unwrap();
            write_file_header(&mut f, 42).unwrap();
            f.sync_all().unwrap();
        }

        // Opening as table 99 detects that the header says table 42.
        let result = ChangeDataFeed::open(tmp.path(), 99, 30);
        assert!(result.is_err());
    }
}
