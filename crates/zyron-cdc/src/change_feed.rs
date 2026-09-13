//! Change Data Feed (CDF) storage for per-table row-level change tracking.
//!
//! A feed-enabled table owns a directory of change segments plus a manifest.
//! Writes append to the open segment. When the open segment reaches its size
//! target it is sealed. Its records are compressed with the feed's codec and a
//! summary trailer is written naming the version range, the timestamp range,
//! the change kinds present and the record count. The manifest holds one
//! summary per sealed segment, so a read over a version window opens only the
//! segments whose range overlaps it and a pending-row count is a lookup rather
//! than a scan.
//!
//! ```text
//! cdf/00000042/
//!     manifest.zycdm          feed configuration, segment summaries, counters
//!     000000000001.zycdf      sealed segment
//!     000000000002.zycdf      open segment, appended to
//! ```
//!
//! Segment layout. A format envelope header whose extension carries the table
//! id and the sequence, then the records. An open segment holds them framed as
//! `[u32 len][record][u32 checksum]` so a torn tail truncates cleanly. A sealed
//! segment holds one compressed block of that same frame stream, then a
//! trailer the format checksums itself, which is why the kind is registered
//! with the own-trailer framing rather than the envelope's body checksum.
//!
//! Concurrency. One Mutex covers the open segment's writer and the in-memory
//! index. Writes serialize. A read snapshots what it needs under the lock and
//! then does its file I/O outside it

use std::collections::{HashMap, HashSet};
use std::fs::{self, File, OpenOptions};
use std::io::{BufWriter, Read, Write};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

use parking_lot::Mutex;
use scc::HashMap as SccHashMap;
use serde::{Deserialize, Serialize};
use zyron_common::format::FormatKind;
use zyron_common::format::envelope::{self, ENVELOPE_HEADER_LEN};
use zyron_common::{Result, ZyronError};

// Record and header checksums use the canonical hot-path hash, which has
// no runtime dispatch and no lane setup, so the per-row insert path pays
// only the mixing itself
use zyron_common::checksum::hot::hot_hash32;

use crate::format::{CHANGE_FEED_MANIFEST_FORMAT_VERSION, CHANGE_FEED_SEGMENT_FORMAT_VERSION};

/// The largest record a segment read accepts, so a corrupt length field
/// cannot ask for unbounded memory
const MAX_RECORD_SIZE: u64 = 64 * 1024 * 1024;

/// Bytes the open segment reaches before it seals and a new one opens.
///
/// A read prunes by whole segment, so a smaller target prunes more finely and
/// costs more files to open for a wide range. A larger one is the reverse
pub const SEGMENT_TARGET_BYTES: u64 = 4 * 1024 * 1024;

/// Per-segment header extension, table id and segment sequence
const SEGMENT_EXTENSION_LEN: usize = 8;

/// Byte offset of the first record in a segment
const SEGMENT_BODY_OFFSET: u64 = (ENVELOPE_HEADER_LEN + SEGMENT_EXTENSION_LEN) as u64;

/// Bytes at the tail of a sealed segment, length, checksum and marker
const TRAILER_SUFFIX_LEN: usize = 12;

/// The record kinds the manifest file holds, the first byte of each
/// record's body, a whole description of the feed and a delta of what
/// changed since the record before it
const MANIFEST_WHOLE: u8 = 0;
const MANIFEST_DELTA: u8 = 1;

/// Bytes of the per record header extension of a manifest record, which
/// carries the body length so the records of one file can be told apart
const MANIFEST_LENGTH_EXTENSION: usize = 4;

/// Delta records appended after a whole one before the next record lays
/// the file down whole again, which bounds what a reopen replays
const MANIFEST_COMPACT_AFTER: u32 = 64;

/// Marks the end of a sealed segment's trailer
const TRAILER_MARKER: [u8; 4] = *b"ZCDT";

/// Per-format envelope flag, this segment is sealed and carries a trailer
const FLAG_SEALED: u32 = 1 << 8;

/// Per-format envelope flag mask holding the codec the sealed body uses
const FLAG_CODEC_SHIFT: u32 = 9;
const FLAG_CODEC_MASK: u32 = 0b11 << FLAG_CODEC_SHIFT;

/// Per-format envelope flag, the sealed body is column-sliced, see
/// [`crate::segment_columns`]. Without it a sealed body is the frames
/// compressed as one block behind their plain length
const FLAG_COLUMNAR: u32 = 1 << 11;

const RECORD_FRAME_PREFIX: usize = 4;
const RECORD_FRAME_SUFFIX: usize = 4;

// Record binary layout (all little-endian):
//   change_type:      u8
//   commit_version:   u64
//   commit_timestamp: i64
//   txn_id:           u64
//   change_ordinal:   u64
//   schema_version:   u32
//   flags:            u8
//   row_data_len:     u32 + bytes
//   pk_data_len:      u32 + bytes
const BINARY_FIXED_HEADER: usize = 46;

/// Record flag, this record is the last one its transaction produced
const RECORD_FLAG_LAST_IN_TXN: u8 = 1 << 0;

/// Record flag, the row bytes hold the feed's column subset rather than the
/// table's full layout, so they decode through the projected layout
const RECORD_FLAG_PROJECTED: u8 = 1 << 1;

// ---------------------------------------------------------------------------
// ChangeType
// ---------------------------------------------------------------------------

/// Type of change captured in a CDF record.
///
/// The discriminants are the codes `zyron_common::change_kind` declares, so
/// what a record carries on disk is the same byte a planner prunes on and an
/// executor renders from
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[repr(u8)]
pub enum ChangeType {
    Insert = zyron_common::CHANGE_TYPE_INSERT,
    UpdatePreimage = zyron_common::CHANGE_TYPE_UPDATE_PREIMAGE,
    UpdatePostimage = zyron_common::CHANGE_TYPE_UPDATE_POSTIMAGE,
    Delete = zyron_common::CHANGE_TYPE_DELETE,
    SchemaChange = zyron_common::CHANGE_TYPE_SCHEMA_CHANGE,
    Truncate = zyron_common::CHANGE_TYPE_TRUNCATE,
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

    /// The name `table_changes` puts in `_change_type`
    pub fn label(self) -> &'static str {
        zyron_common::change_type_label(self as u8).unwrap_or("unknown")
    }

    /// Resolves a `_change_type` literal, for a predicate that names one
    pub fn from_label(text: &str) -> Option<Self> {
        zyron_common::change_type_code(text).and_then(|code| Self::from_u8(code).ok())
    }

    /// Bit this kind occupies in a segment summary's change-kind mask
    #[inline]
    fn mask_bit(self) -> u8 {
        zyron_common::change_type_bit(self as u8)
    }
}

// ---------------------------------------------------------------------------
// CdfCodec
// ---------------------------------------------------------------------------

/// Compression applied to a sealed segment's record block
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[repr(u8)]
pub enum CdfCodec {
    None = 0,
    #[default]
    Lz4 = 1,
    Zstd = 2,
}

impl CdfCodec {
    pub fn from_u8(v: u8) -> Result<Self> {
        match v {
            0 => Ok(Self::None),
            1 => Ok(Self::Lz4),
            2 => Ok(Self::Zstd),
            _ => Err(ZyronError::CdcDecoderError(format!(
                "unknown change feed compression code: {v}"
            ))),
        }
    }

    /// Resolves the word an `ALTER TABLE` option carries
    pub fn from_name(name: &str) -> Result<Self> {
        match name.to_ascii_lowercase().as_str() {
            "none" | "off" => Ok(Self::None),
            "lz4" => Ok(Self::Lz4),
            "zstd" => Ok(Self::Zstd),
            other => Err(ZyronError::Internal(format!(
                "cdf_compression accepts 'none', 'lz4' or 'zstd', not '{other}'"
            ))),
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            CdfCodec::None => "none",
            CdfCodec::Lz4 => "lz4",
            CdfCodec::Zstd => "zstd",
        }
    }

    pub(crate) fn compress(self, input: &[u8]) -> Result<Vec<u8>> {
        match self {
            CdfCodec::None => Ok(input.to_vec()),
            CdfCodec::Lz4 => Ok(lz4_flex::compress(input)),
            CdfCodec::Zstd => zstd::stream::encode_all(input, 3).map_err(|e| {
                ZyronError::CdcStreamError(format!("change segment zstd compression failed: {e}"))
            }),
        }
    }

    /// Decodes into `out`, answering with how many bytes were written
    pub(crate) fn decompress_into(self, input: &[u8], out: &mut [u8]) -> Result<usize> {
        match self {
            CdfCodec::None => {
                if input.len() != out.len() {
                    return Err(ZyronError::CdcDecoderError(
                        "change segment raw block does not fit where it was asked for".into(),
                    ));
                }
                out.copy_from_slice(input);
                Ok(input.len())
            }
            CdfCodec::Lz4 => lz4_flex::block::decompress_into(input, out).map_err(|e| {
                ZyronError::CdcDecoderError(format!("change segment lz4 block did not decode: {e}"))
            }),
            CdfCodec::Zstd => zstd::bulk::decompress_to_buffer(input, out).map_err(|e| {
                ZyronError::CdcDecoderError(format!(
                    "change segment zstd block did not decode: {e}"
                ))
            }),
        }
    }

    pub(crate) fn decompress(self, input: &[u8], expected_len: usize) -> Result<Vec<u8>> {
        match self {
            CdfCodec::None => Ok(input.to_vec()),
            CdfCodec::Lz4 => lz4_flex::decompress(input, expected_len).map_err(|e| {
                ZyronError::CdcDecoderError(format!("change segment lz4 block did not decode: {e}"))
            }),
            CdfCodec::Zstd => zstd::stream::decode_all(input).map_err(|e| {
                ZyronError::CdcDecoderError(format!(
                    "change segment zstd block did not decode: {e}"
                ))
            }),
        }
    }
}

// ---------------------------------------------------------------------------
// FeedConfig
// ---------------------------------------------------------------------------

/// Everything `ALTER TABLE t SET (...)` can say about a feed
#[derive(Debug, Clone, PartialEq)]
pub struct FeedConfig {
    pub enabled: bool,
    /// How long a change is kept, in microseconds. Zero keeps changes until
    /// a byte cap or an explicit purge removes them
    pub retention_micros: i64,
    /// Column ids the feed records, plus the table's key columns. None
    /// records every column
    pub columns: Option<Vec<u16>>,
    /// False records one row per update instead of two, which halves an
    /// update-heavy feed and makes update_preimage unavailable
    pub before_image: bool,
    pub codec: CdfCodec,
    /// For a branch's feed, the version of the table's own feed the branch
    /// was taken at. Changes at or below it are the table's, read from the
    /// table's feed, and changes above it on the branch are here. Zero for
    /// the table's own feed
    pub branch_point: u64,
    /// The source version the feed began recording at, above which its
    /// changes lie. Zero for a feed that recorded from the table's first
    /// change. A lake table's log holds commits from before its feed was
    /// turned on, and those are not the feed's
    pub first_version: u64,
}

/// Microseconds in a day, for the day count a feed's retention used to be
/// written as
pub const MICROS_PER_DAY: i64 = 24 * 60 * 60 * 1_000_000;

impl Default for FeedConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            retention_micros: 7 * MICROS_PER_DAY,
            columns: None,
            before_image: true,
            codec: CdfCodec::Lz4,
            branch_point: 0,
            first_version: 0,
        }
    }
}

impl FeedConfig {
    /// The configuration a table carrying only a day count describes
    pub fn from_retention_days(days: u32) -> Self {
        Self {
            retention_micros: days as i64 * MICROS_PER_DAY,
            ..Self::default()
        }
    }

    /// Retention rendered as whole days, for a caller that reports in days
    pub fn retention_days(&self) -> u32 {
        (self.retention_micros / MICROS_PER_DAY).max(0) as u32
    }

    /// Whether the feed records this column
    pub fn records_column(&self, column_id: u16) -> bool {
        match &self.columns {
            None => true,
            Some(list) => list.contains(&column_id),
        }
    }

    fn write_into(&self, buf: &mut Vec<u8>) {
        buf.push(u8::from(self.enabled));
        buf.extend_from_slice(&self.retention_micros.to_le_bytes());
        match &self.columns {
            None => buf.push(0),
            Some(list) => {
                buf.push(1);
                buf.extend_from_slice(&(list.len() as u32).to_le_bytes());
                for id in list {
                    buf.extend_from_slice(&id.to_le_bytes());
                }
            }
        }
        buf.push(u8::from(self.before_image));
        buf.push(self.codec as u8);
        buf.extend_from_slice(&self.branch_point.to_le_bytes());
        buf.extend_from_slice(&self.first_version.to_le_bytes());
    }

    fn read_from(cursor: &mut ByteCursor<'_>) -> Result<Self> {
        let enabled = cursor.u8()? != 0;
        let retention_micros = cursor.i64()?;
        let columns = if cursor.u8()? == 0 {
            None
        } else {
            let count = cursor.u32()? as usize;
            let mut list = Vec::with_capacity(count);
            for _ in 0..count {
                list.push(cursor.u16()?);
            }
            Some(list)
        };
        let before_image = cursor.u8()? != 0;
        let codec = CdfCodec::from_u8(cursor.u8()?)?;
        let branch_point = cursor.u64()?;
        let first_version = cursor.u64()?;
        Ok(Self {
            enabled,
            retention_micros,
            columns,
            before_image,
            codec,
            branch_point,
            first_version,
        })
    }
}

// ---------------------------------------------------------------------------
// ByteCursor
// ---------------------------------------------------------------------------

/// Bounds-checked little-endian reader for the manifest, the trailer and
/// the index a derived source keeps
pub(crate) struct ByteCursor<'a> {
    data: &'a [u8],
    off: usize,
}

impl<'a> ByteCursor<'a> {
    pub(crate) fn new(data: &'a [u8]) -> Self {
        Self { data, off: 0 }
    }

    fn take(&mut self, n: usize) -> Result<&'a [u8]> {
        if self.off + n > self.data.len() {
            return Err(ZyronError::CdcDecoderError(format!(
                "change feed metadata is truncated, {n} bytes needed at offset {}",
                self.off
            )));
        }
        let slice = &self.data[self.off..self.off + n];
        self.off += n;
        Ok(slice)
    }

    pub(crate) fn u8(&mut self) -> Result<u8> {
        Ok(self.take(1)?[0])
    }

    pub(crate) fn u16(&mut self) -> Result<u16> {
        let b = self.take(2)?;
        Ok(u16::from_le_bytes([b[0], b[1]]))
    }

    pub(crate) fn u32(&mut self) -> Result<u32> {
        let b = self.take(4)?;
        Ok(u32::from_le_bytes([b[0], b[1], b[2], b[3]]))
    }

    pub(crate) fn u64(&mut self) -> Result<u64> {
        let b = self.take(8)?;
        let mut wide = [0u8; 8];
        wide.copy_from_slice(b);
        Ok(u64::from_le_bytes(wide))
    }

    pub(crate) fn i64(&mut self) -> Result<i64> {
        Ok(self.u64()? as i64)
    }

    /// Whether every byte has been read
    pub(crate) fn is_empty(&self) -> bool {
        self.off >= self.data.len()
    }
}

// ---------------------------------------------------------------------------
// ChangeRecord
// ---------------------------------------------------------------------------

/// A single change record stored in a change segment
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChangeRecord {
    pub change_type: ChangeType,
    pub commit_version: u64,
    pub commit_timestamp: i64,
    pub table_id: u32,
    pub txn_id: u64,
    /// Position within the commit, so an update's two rows sort together and
    /// a consumer orders deterministically inside one version
    pub change_ordinal: u64,
    pub schema_version: u32,
    pub row_data: Vec<u8>,
    pub primary_key_data: Vec<u8>,
    pub is_last_in_txn: bool,
    /// True when `row_data` holds the feed's column subset rather than the
    /// table's full layout
    pub projected: bool,
}

impl ChangeRecord {
    /// A record with the fields a caller that is not projecting sets, and
    /// defaults for the rest
    pub fn new(
        change_type: ChangeType,
        commit_version: u64,
        commit_timestamp: i64,
        table_id: u32,
        txn_id: u64,
        schema_version: u32,
        row_data: Vec<u8>,
    ) -> Self {
        Self {
            change_type,
            commit_version,
            commit_timestamp,
            table_id,
            txn_id,
            change_ordinal: 0,
            schema_version,
            row_data,
            primary_key_data: Vec::new(),
            is_last_in_txn: false,
            projected: false,
        }
    }

    /// Bytes this record occupies in packed binary form, so a caller can
    /// size a buffer for many records exactly instead of growing it
    #[inline]
    fn serialized_len(&self) -> usize {
        BINARY_FIXED_HEADER + self.row_data.len() + self.primary_key_data.len()
    }

    /// Deserializes from packed binary format. table_id is supplied from the
    /// segment header, not from the record bytes
    fn deserialize(data: &[u8], table_id: u32) -> Result<Self> {
        let mut cursor = ByteCursor::new(data);
        let change_type = ChangeType::from_u8(cursor.u8()?)?;
        let commit_version = cursor.u64()?;
        let commit_timestamp = cursor.i64()?;
        let txn_id = cursor.u64()?;
        let change_ordinal = cursor.u64()?;
        let schema_version = cursor.u32()?;
        let flags = cursor.u8()?;
        let row_len = cursor.u32()? as usize;
        let row_data = cursor.take(row_len)?.to_vec();
        let pk_len = cursor.u32()? as usize;
        let primary_key_data = cursor.take(pk_len)?.to_vec();
        Ok(Self {
            change_type,
            commit_version,
            commit_timestamp,
            table_id,
            txn_id,
            change_ordinal,
            schema_version,
            row_data,
            primary_key_data,
            is_last_in_txn: flags & RECORD_FLAG_LAST_IN_TXN != 0,
            projected: flags & RECORD_FLAG_PROJECTED != 0,
        })
    }

    /// Reads version and timestamp from a serialized record without
    /// deserializing its variable-length fields
    fn peek_version_timestamp(data: &[u8]) -> Result<(u64, i64)> {
        if data.len() < 17 {
            return Err(ZyronError::CdcDecoderError(
                "change record is too short to carry a version".into(),
            ));
        }
        let mut version = [0u8; 8];
        version.copy_from_slice(&data[1..9]);
        let mut timestamp = [0u8; 8];
        timestamp.copy_from_slice(&data[9..17]);
        Ok((
            u64::from_le_bytes(version),
            i64::from_le_bytes(timestamp) as i64,
        ))
    }

    /// The change kind a serialized record carries, without decoding it
    #[inline]
    fn peek_change_type(data: &[u8]) -> Result<ChangeType> {
        if data.is_empty() {
            return Err(ZyronError::CdcDecoderError(
                "change record is empty".to_string(),
            ));
        }
        ChangeType::from_u8(data[0])
    }

    /// Reads the schema epoch from a serialized record without
    /// deserializing its variable-length fields
    fn peek_schema_version(data: &[u8]) -> Result<u32> {
        if data.len() < 37 {
            return Err(ZyronError::CdcDecoderError(
                "change record is too short to carry a schema epoch".into(),
            ));
        }
        let mut epoch = [0u8; 4];
        epoch.copy_from_slice(&data[33..37]);
        Ok(u32::from_le_bytes(epoch))
    }

    /// Reads the transaction id from a serialized record without
    /// deserializing its variable-length fields
    fn peek_txn_id(data: &[u8]) -> Result<u64> {
        if data.len() < 25 {
            return Err(ZyronError::CdcDecoderError(
                "change record is too short to carry a transaction id".into(),
            ));
        }
        let mut txn_id = [0u8; 8];
        txn_id.copy_from_slice(&data[17..25]);
        Ok(u64::from_le_bytes(txn_id))
    }
}

/// What every change of one write shares
#[derive(Debug, Clone, Copy)]
pub struct ChangeHeader {
    pub commit_version: u64,
    pub commit_timestamp: i64,
    pub txn_id: u64,
    pub schema_version: u32,
}

/// The layout the rows of one write were encoded under, so the feed can
/// take them apart into columns when the segment seals. None for rows the
/// writer cannot describe, which stay framed
#[derive(Debug, Clone, Copy)]
pub struct RowLayout<'a> {
    /// The physical columns in tuple order, the table's layout at the
    /// write's epoch or the feed's subset of it
    pub columns: Option<&'a [zyron_catalog::PhysicalColumn]>,
    /// Whether the rows hold the feed's column subset
    pub projected: bool,
}

impl RowLayout<'_> {
    /// Rows the writer does not describe
    pub const UNKNOWN: RowLayout<'static> = RowLayout {
        columns: None,
        projected: false,
    };
}

/// One change of a write, with its bytes borrowed from the writer's own
/// tuples, so recording a change copies the row once, into the feed's
/// buffer, rather than into a record of its own first
#[derive(Debug, Clone, Copy)]
pub struct RowChange<'a> {
    pub change_type: ChangeType,
    pub row_data: &'a [u8],
    pub primary_key_data: &'a [u8],
    pub is_last_in_txn: bool,
    pub projected: bool,
}

impl<'a> RowChange<'a> {
    /// A change of a row written whole, with no key bytes of its own
    pub fn of(change_type: ChangeType, row_data: &'a [u8], is_last_in_txn: bool) -> Self {
        Self {
            change_type,
            row_data,
            primary_key_data: &[],
            is_last_in_txn,
            projected: false,
        }
    }

    /// A change whose row bytes hold the feed's column subset rather than
    /// the table's full layout
    pub fn projected(change_type: ChangeType, row_data: &'a [u8], is_last_in_txn: bool) -> Self {
        Self {
            change_type,
            row_data,
            primary_key_data: &[],
            is_last_in_txn,
            projected: true,
        }
    }

    /// Bytes the change occupies once framed
    #[inline]
    fn framed_len(&self) -> usize {
        RECORD_FRAME_PREFIX
            + BINARY_FIXED_HEADER
            + self.row_data.len()
            + self.primary_key_data.len()
            + RECORD_FRAME_SUFFIX
    }

    #[inline]
    fn flag_byte(&self) -> u8 {
        let mut flags = 0u8;
        if self.is_last_in_txn {
            flags |= RECORD_FLAG_LAST_IN_TXN;
        }
        if self.projected {
            flags |= RECORD_FLAG_PROJECTED;
        }
        flags
    }
}

/// What a scan needs and nothing it does not. The metadata columns and the
/// encoded row, with no copy of the row and no allocation per record
#[derive(Debug, Clone, Copy)]
pub struct ChangeRecordRef<'a> {
    pub change_type: ChangeType,
    pub commit_version: u64,
    pub commit_timestamp: i64,
    pub table_id: u32,
    pub txn_id: u64,
    pub change_ordinal: u64,
    pub schema_version: u32,
    pub is_last_in_txn: bool,
    /// True when the row bytes hold the feed's column subset
    pub projected: bool,
    pub row_data: &'a [u8],
}

impl<'a> ChangeRecordRef<'a> {
    /// Reads the fixed header and takes the row bytes as a slice
    fn of(data: &'a [u8], table_id: u32) -> Result<Self> {
        if data.len() < BINARY_FIXED_HEADER {
            return Err(ZyronError::CdcDecoderError(
                "change record is shorter than its header".into(),
            ));
        }
        let change_type = ChangeType::from_u8(data[0])?;
        let read_u64 = |at: usize| {
            let mut wide = [0u8; 8];
            wide.copy_from_slice(&data[at..at + 8]);
            u64::from_le_bytes(wide)
        };
        let commit_version = read_u64(1);
        let commit_timestamp = read_u64(9) as i64;
        let txn_id = read_u64(17);
        let change_ordinal = read_u64(25);
        let mut epoch = [0u8; 4];
        epoch.copy_from_slice(&data[33..37]);
        let schema_version = u32::from_le_bytes(epoch);
        let flags = data[37];
        let mut row_len_bytes = [0u8; 4];
        row_len_bytes.copy_from_slice(&data[38..42]);
        let row_len = u32::from_le_bytes(row_len_bytes) as usize;
        if 42 + row_len > data.len() {
            return Err(ZyronError::CdcDecoderError(
                "change record row bytes run past the record".into(),
            ));
        }
        Ok(Self {
            change_type,
            commit_version,
            commit_timestamp,
            table_id,
            txn_id,
            change_ordinal,
            schema_version,
            is_last_in_txn: flags & RECORD_FLAG_LAST_IN_TXN != 0,
            projected: flags & RECORD_FLAG_PROJECTED != 0,
            row_data: &data[42..42 + row_len],
        })
    }
}

// ---------------------------------------------------------------------------
// SegmentSummary
// ---------------------------------------------------------------------------

/// What one segment holds, as the manifest records it
#[derive(Debug, Clone, PartialEq)]
pub struct SegmentSummary {
    pub seq: u64,
    pub min_version: u64,
    pub max_version: u64,
    pub min_timestamp: i64,
    pub max_timestamp: i64,
    /// One bit per `ChangeType` present in the segment
    pub change_type_mask: u8,
    pub record_count: u64,
    pub bytes: u64,
    pub codec: CdfCodec,
    pub sealed: bool,
    /// The lowest schema epoch a record of the segment was written under,
    /// `u32::MAX` while the segment is empty. The table's layout for that
    /// epoch stays recorded for as long as the segment holds the record
    pub min_epoch: u32,
}

impl SegmentSummary {
    fn empty(seq: u64, codec: CdfCodec) -> Self {
        Self {
            seq,
            min_version: u64::MAX,
            max_version: 0,
            min_timestamp: i64::MAX,
            max_timestamp: i64::MIN,
            change_type_mask: 0,
            record_count: 0,
            bytes: SEGMENT_BODY_OFFSET,
            codec,
            sealed: false,
            min_epoch: u32::MAX,
        }
    }

    fn observe(&mut self, version: u64, timestamp: i64, change_type: ChangeType, epoch: u32) {
        self.min_version = self.min_version.min(version);
        self.max_version = self.max_version.max(version);
        self.min_timestamp = self.min_timestamp.min(timestamp);
        self.max_timestamp = self.max_timestamp.max(timestamp);
        self.change_type_mask |= change_type.mask_bit();
        self.record_count += 1;
        self.min_epoch = self.min_epoch.min(epoch);
    }

    fn is_empty(&self) -> bool {
        self.record_count == 0
    }

    /// Whether any record in this segment can satisfy the range
    fn overlaps(&self, range: &ChangeRange) -> bool {
        if self.is_empty() {
            return false;
        }
        if self.max_version < range.start_version || self.min_version > range.end_version {
            return false;
        }
        if self.max_timestamp < range.start_timestamp || self.min_timestamp > range.end_timestamp {
            return false;
        }
        match range.change_types {
            Some(mask) => self.change_type_mask & mask != 0,
            None => true,
        }
    }

    fn write_into(&self, buf: &mut Vec<u8>) {
        buf.extend_from_slice(&self.seq.to_le_bytes());
        buf.extend_from_slice(&self.min_version.to_le_bytes());
        buf.extend_from_slice(&self.max_version.to_le_bytes());
        buf.extend_from_slice(&self.min_timestamp.to_le_bytes());
        buf.extend_from_slice(&self.max_timestamp.to_le_bytes());
        buf.push(self.change_type_mask);
        buf.extend_from_slice(&self.record_count.to_le_bytes());
        buf.extend_from_slice(&self.bytes.to_le_bytes());
        buf.push(self.codec as u8);
        buf.push(u8::from(self.sealed));
        buf.extend_from_slice(&self.min_epoch.to_le_bytes());
    }

    fn read_from(cursor: &mut ByteCursor<'_>) -> Result<Self> {
        Ok(Self {
            seq: cursor.u64()?,
            min_version: cursor.u64()?,
            max_version: cursor.u64()?,
            min_timestamp: cursor.i64()?,
            max_timestamp: cursor.i64()?,
            change_type_mask: cursor.u8()?,
            record_count: cursor.u64()?,
            bytes: cursor.u64()?,
            codec: CdfCodec::from_u8(cursor.u8()?)?,
            sealed: cursor.u8()? != 0,
            min_epoch: cursor.u32()?,
        })
    }
}

// ---------------------------------------------------------------------------
// VersionCount
// ---------------------------------------------------------------------------

/// How many records one commit version wrote, and how many the feed had
/// written before it.
///
/// `prior` counts from the feed's creation and a purge never reduces it, so
/// the difference between two entries is the number of records between two
/// versions whether or not anything below them was reclaimed
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct VersionCount {
    pub version: u64,
    pub records: u64,
    pub prior: u64,
    pub first_timestamp: i64,
}

/// The lookups a per version record index answers, over entries ascending
/// by version whose `prior` counts ascend with them.
///
/// A feed keeps such an index over the records it wrote, and a source whose
/// changes are derived from a store of its own keeps one over the records
/// each of its versions yields, so a stream position means the same count
/// on either and the same answers come from one place
pub mod version_index {
    use super::VersionCount;

    /// Records at or below `version`, None when the index is empty. A
    /// version below everything indexed still counts what lies beneath it,
    /// because `prior` is absolute
    pub fn records_at_or_below(versions: &[VersionCount], version: u64) -> Option<u64> {
        if versions.is_empty() {
            return None;
        }
        Some(
            match versions.binary_search_by(|e| e.version.cmp(&version)) {
                Ok(at) => versions[at].prior + versions[at].records,
                Err(0) => versions[0].prior,
                Err(at) => versions[at - 1].prior + versions[at - 1].records,
            },
        )
    }

    /// Records before the record at `ordinal` within `version`. Ordinals
    /// number a version's records from zero in the order they were
    /// written, so this is the version's `prior` plus the ordinal
    pub fn records_before(versions: &[VersionCount], version: u64, ordinal: u64) -> u64 {
        match versions.binary_search_by(|e| e.version.cmp(&version)) {
            Ok(at) => versions[at].prior + ordinal.min(versions[at].records),
            Err(0) => versions.first().map(|v| v.prior).unwrap_or(0),
            Err(at) => versions[at - 1].prior + versions[at - 1].records,
        }
    }

    /// The version and ordinal of the record at which exactly `count`
    /// records had been written, None when the count is zero or names a
    /// record below everything indexed
    pub fn cursor_at_count(versions: &[VersionCount], count: u64) -> Option<(u64, u64)> {
        if count == 0 {
            return None;
        }
        let at = versions.partition_point(|entry| entry.prior + entry.records < count);
        let entry = versions.get(at)?;
        if entry.prior >= count {
            return None;
        }
        Some((entry.version, count - entry.prior - 1))
    }

    /// The version at which exactly `count` records had been written, the
    /// inverse of `records_at_or_below`. A count above everything indexed
    /// answers with the newest version, which is the furthest a position
    /// can legitimately reach
    pub fn version_at_count(versions: &[VersionCount], count: u64) -> u64 {
        if count == 0 || versions.is_empty() {
            return 0;
        }
        // The last version whose records are wholly at or below the count.
        // `prior` ascends with the index, so this is a binary search rather
        // than a walk
        let at = versions.partition_point(|entry| entry.prior + entry.records <= count);
        match at {
            // Below the end of the first retained version. A count at or past
            // everything reclaimed names the version before the first
            // retained one, so a reader resuming there reads that version
            // from the record the count names, and a count inside what was
            // reclaimed names nothing a read can reach
            0 if count >= versions[0].prior => versions[0].version.saturating_sub(1),
            0 => 0,
            _ => versions[at - 1].version,
        }
    }

    /// The version holding the record at `consumed + max_rows`, where a
    /// read that resumes after `consumed` records and takes at most
    /// `max_rows` more ends, or the version before it when that count is
    /// exactly the end of that one. None when fewer than that many records
    /// lie past the position, or when the cut would land on the newest
    /// version, which bounds nothing
    pub fn bounded_cut(versions: &[VersionCount], consumed: u64, max_rows: u64) -> Option<u64> {
        let target = consumed.saturating_add(max_rows);
        let at = versions.partition_point(|entry| entry.prior + entry.records < target);
        let entry = versions.get(at)?;
        let cut = if entry.prior >= target {
            versions[at.checked_sub(1)?].version
        } else {
            entry.version
        };
        if versions.last().is_some_and(|last| last.version <= cut) {
            return None;
        }
        Some(cut)
    }

    /// The first entry strictly above `version`
    pub fn first_after(versions: &[VersionCount], version: u64) -> Option<VersionCount> {
        let at = versions.partition_point(|e| e.version <= version);
        versions.get(at).copied()
    }
}

/// Where one transaction's records lie in the feed, the first and the last
/// version it wrote at. A bounded read ends at a version no transaction
/// writes across, which is what these answer
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TxnSpan {
    pub txn_id: u64,
    pub first: u64,
    pub last: u64,
}

// ---------------------------------------------------------------------------
// ChangeRange
// ---------------------------------------------------------------------------

/// The window a read asks for, after the planner resolved its bounds
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ChangeRange {
    pub start_version: u64,
    pub end_version: u64,
    pub start_timestamp: i64,
    pub end_timestamp: i64,
    /// One bit per accepted `ChangeType`. None accepts every kind
    pub change_types: Option<u8>,
}

impl Default for ChangeRange {
    fn default() -> Self {
        Self::everything()
    }
}

impl ChangeRange {
    /// Every record the feed holds
    pub fn everything() -> Self {
        Self {
            start_version: 0,
            end_version: u64::MAX,
            start_timestamp: i64::MIN,
            end_timestamp: i64::MAX,
            change_types: None,
        }
    }

    /// A closed version window
    pub fn versions(start: u64, end: u64) -> Self {
        Self {
            start_version: start,
            end_version: end,
            ..Self::everything()
        }
    }

    /// A closed timestamp window
    pub fn timestamps(start: i64, end: i64) -> Self {
        Self {
            start_timestamp: start,
            end_timestamp: end,
            ..Self::everything()
        }
    }

    /// Narrows to the given change kinds
    pub fn with_change_types(mut self, kinds: &[ChangeType]) -> Self {
        let mut mask = 0u8;
        for kind in kinds {
            mask |= kind.mask_bit();
        }
        self.change_types = Some(mask);
        self
    }

    #[inline]
    fn admits(&self, version: u64, timestamp: i64, change_type: ChangeType) -> bool {
        version >= self.start_version
            && version <= self.end_version
            && timestamp >= self.start_timestamp
            && timestamp <= self.end_timestamp
            && self
                .change_types
                .is_none_or(|mask| mask & change_type.mask_bit() != 0)
    }
}

// ---------------------------------------------------------------------------
// ReadPlan
// ---------------------------------------------------------------------------

/// The segments a read will open, resolved before any file is touched.
///
/// `EXPLAIN` reports the count, and a range predicate that prunes every
/// segment produces an empty plan without a single file open
#[derive(Debug, Clone)]
pub struct ReadPlan {
    /// Sequence numbers of the segments whose summaries overlap the range
    pub segments: Vec<u64>,
    /// Segments the range excluded
    pub pruned: usize,
    /// Records the surviving segments hold, an upper bound on the answer
    pub candidate_records: u64,
}

// ---------------------------------------------------------------------------
// Segment file
// ---------------------------------------------------------------------------

/// Writes a segment's envelope header plus its extension into a writer
fn write_segment_header(w: &mut impl Write, table_id: u32, seq: u64, flags: u32) -> Result<()> {
    let mut extension = [0u8; SEGMENT_EXTENSION_LEN];
    extension[0..4].copy_from_slice(&table_id.to_le_bytes());
    extension[4..8].copy_from_slice(&(seq as u32).to_le_bytes());
    let header = envelope::encode_header(
        FormatKind::ChangeFeedSegment,
        CHANGE_FEED_SEGMENT_FORMAT_VERSION,
        flags,
        &extension,
    );
    w.write_all(&header)?;
    w.write_all(&extension)?;
    Ok(())
}

/// What a segment header says about the file behind it
struct SegmentHeader {
    table_id: u32,
    sealed: bool,
    codec: CdfCodec,
    columnar: bool,
}

/// Reads and validates a segment's envelope header
fn read_segment_header(bytes: &[u8]) -> Result<SegmentHeader> {
    let (header, extension) = envelope::decode_header(bytes).map_err(|e| {
        ZyronError::CdcDecoderError(format!("change segment header did not decode, {e}"))
    })?;
    if header.kind != FormatKind::ChangeFeedSegment {
        return Err(ZyronError::CdcDecoderError(format!(
            "expected a change feed segment, found a {} file",
            header.kind
        )));
    }
    if header.version != CHANGE_FEED_SEGMENT_FORMAT_VERSION {
        return Err(ZyronError::CdcDecoderError(format!(
            "change segment is at format version {}, this binary writes and reads {}. \
             Upgrade through a release that still reads {} to move it forward first",
            header.version, CHANGE_FEED_SEGMENT_FORMAT_VERSION, header.version
        )));
    }
    if extension.len() < SEGMENT_EXTENSION_LEN {
        return Err(ZyronError::CdcDecoderError(
            "change segment header carries no table id".into(),
        ));
    }
    let mut table = [0u8; 4];
    table.copy_from_slice(&extension[0..4]);
    Ok(SegmentHeader {
        table_id: u32::from_le_bytes(table),
        sealed: header.flags & FLAG_SEALED != 0,
        codec: CdfCodec::from_u8(((header.flags & FLAG_CODEC_MASK) >> FLAG_CODEC_SHIFT) as u8)?,
        columnar: header.flags & FLAG_COLUMNAR != 0,
    })
}

/// Walks a framed record stream, handing each record's bytes to a visitor.
///
/// Stops at the first frame that does not check out, reporting how many bytes
/// were valid. An open segment's tail can be torn by a crash, and the caller
/// truncates there rather than reading past it
fn walk_frames(body: &[u8], mut visit: impl FnMut(&[u8]) -> Result<()>) -> Result<(usize, usize)> {
    let mut off = 0usize;
    let mut count = 0usize;
    while off + RECORD_FRAME_PREFIX + RECORD_FRAME_SUFFIX <= body.len() {
        let mut len_bytes = [0u8; 4];
        len_bytes.copy_from_slice(&body[off..off + 4]);
        let record_len = u32::from_le_bytes(len_bytes) as u64;
        if record_len > MAX_RECORD_SIZE {
            break;
        }
        let total = RECORD_FRAME_PREFIX as u64 + record_len + RECORD_FRAME_SUFFIX as u64;
        if off as u64 + total > body.len() as u64 {
            break;
        }
        let data_start = off + RECORD_FRAME_PREFIX;
        let data_end = data_start + record_len as usize;
        let record = &body[data_start..data_end];
        let mut crc_bytes = [0u8; 4];
        crc_bytes.copy_from_slice(&body[data_end..data_end + 4]);
        if u32::from_le_bytes(crc_bytes) != hot_hash32(record) {
            break;
        }
        visit(record)?;
        count += 1;
        off += total as usize;
    }
    Ok((off, count))
}

/// Walks frames from `offset`, handing each record's bytes to the visitor
/// until it answers false or the frames run out, and answers with the
/// offset of the frame after the last one visited. A frame that fails its
/// length or checksum ends the walk the way the end of the frames does
fn walk_frames_from(
    body: &[u8],
    offset: usize,
    mut visit: impl FnMut(&[u8]) -> Result<bool>,
) -> Result<usize> {
    let mut off = offset;
    while off + RECORD_FRAME_PREFIX + RECORD_FRAME_SUFFIX <= body.len() {
        let mut len_bytes = [0u8; 4];
        len_bytes.copy_from_slice(&body[off..off + 4]);
        let record_len = u32::from_le_bytes(len_bytes) as u64;
        if record_len > MAX_RECORD_SIZE {
            return Ok(body.len());
        }
        let total = RECORD_FRAME_PREFIX as u64 + record_len + RECORD_FRAME_SUFFIX as u64;
        if off as u64 + total > body.len() as u64 {
            return Ok(body.len());
        }
        let data_start = off + RECORD_FRAME_PREFIX;
        let data_end = data_start + record_len as usize;
        let record = &body[data_start..data_end];
        let mut crc_bytes = [0u8; 4];
        crc_bytes.copy_from_slice(&body[data_end..data_end + 4]);
        if u32::from_le_bytes(crc_bytes) != hot_hash32(record) {
            return Ok(body.len());
        }
        off += total as usize;
        if !visit(record)? {
            return Ok(off);
        }
    }
    Ok(off)
}

/// Where each version's first frame begins in a run of frames, ascending.
///
/// Frames are written in version order, so the run a read starting at a
/// version wants is everything from that version's first frame on
fn frame_version_starts(frames: &[u8]) -> Vec<(u64, usize)> {
    let mut starts: Vec<(u64, usize)> = Vec::new();
    let mut off = 0usize;
    while off + RECORD_FRAME_PREFIX + RECORD_FRAME_SUFFIX <= frames.len() {
        let mut len_bytes = [0u8; 4];
        len_bytes.copy_from_slice(&frames[off..off + 4]);
        let record_len = u32::from_le_bytes(len_bytes) as usize;
        let data_start = off + RECORD_FRAME_PREFIX;
        let total = RECORD_FRAME_PREFIX + record_len + RECORD_FRAME_SUFFIX;
        if record_len as u64 > MAX_RECORD_SIZE || off + total > frames.len() {
            break;
        }
        if let Ok((version, _)) =
            ChangeRecord::peek_version_timestamp(&frames[data_start..data_start + record_len])
        {
            if starts.last().is_none_or(|(last, _)| *last != version) {
                starts.push((version, off));
            }
        }
        off += total;
    }
    starts
}

/// Frames a record at its own version and ordinal, the shape a rewrite of
/// records already held takes
fn frame_into(buf: &mut Vec<u8>, record: &ChangeRecord, ordinal: u64) {
    let header = ChangeHeader {
        commit_version: record.commit_version,
        commit_timestamp: record.commit_timestamp,
        txn_id: record.txn_id,
        schema_version: record.schema_version,
    };
    let change = RowChange {
        change_type: record.change_type,
        row_data: &record.row_data,
        primary_key_data: &record.primary_key_data,
        is_last_in_txn: record.is_last_in_txn,
        projected: record.projected,
    };
    frame_row(buf, &header, record.commit_version, &change, ordinal);
}

/// Frames one change at `version`, in the same bytes a record of it would
/// take. The length prefix is written after the body, whose size is known
/// once it is encoded, and the change serializes straight into the shared
/// buffer rather than into a buffer of its own
fn frame_row(
    buf: &mut Vec<u8>,
    header: &ChangeHeader,
    version: u64,
    change: &RowChange<'_>,
    ordinal: u64,
) {
    let start = buf.len();
    buf.extend_from_slice(&0u32.to_le_bytes());
    let body_start = buf.len();
    buf.push(change.change_type as u8);
    buf.extend_from_slice(&version.to_le_bytes());
    buf.extend_from_slice(&header.commit_timestamp.to_le_bytes());
    buf.extend_from_slice(&header.txn_id.to_le_bytes());
    buf.extend_from_slice(&ordinal.to_le_bytes());
    buf.extend_from_slice(&header.schema_version.to_le_bytes());
    buf.push(change.flag_byte());
    buf.extend_from_slice(&(change.row_data.len() as u32).to_le_bytes());
    buf.extend_from_slice(change.row_data);
    buf.extend_from_slice(&(change.primary_key_data.len() as u32).to_le_bytes());
    buf.extend_from_slice(change.primary_key_data);
    let body_len = (buf.len() - body_start) as u32;
    buf[start..body_start].copy_from_slice(&body_len.to_le_bytes());
    let checksum = hot_hash32(&buf[body_start..]);
    buf.extend_from_slice(&checksum.to_le_bytes());
}

/// Frames one record from its fields, the form every record takes in a
/// segment's framed body
#[allow(clippy::too_many_arguments)]
pub(crate) fn frame_parts(
    buf: &mut Vec<u8>,
    change_type: u8,
    version: u64,
    timestamp: i64,
    txn_id: u64,
    ordinal: u64,
    epoch: u32,
    flags: u8,
    row: &[u8],
    key: &[u8],
) {
    let start = buf.len();
    buf.extend_from_slice(&0u32.to_le_bytes());
    let body_start = buf.len();
    buf.push(change_type);
    buf.extend_from_slice(&version.to_le_bytes());
    buf.extend_from_slice(&timestamp.to_le_bytes());
    buf.extend_from_slice(&txn_id.to_le_bytes());
    buf.extend_from_slice(&ordinal.to_le_bytes());
    buf.extend_from_slice(&epoch.to_le_bytes());
    buf.push(flags);
    buf.extend_from_slice(&(row.len() as u32).to_le_bytes());
    buf.extend_from_slice(row);
    buf.extend_from_slice(&(key.len() as u32).to_le_bytes());
    buf.extend_from_slice(key);
    let body_len = (buf.len() - body_start) as u32;
    buf[start..body_start].copy_from_slice(&body_len.to_le_bytes());
    let checksum = hot_hash32(&buf[body_start..]);
    buf.extend_from_slice(&checksum.to_le_bytes());
}

/// Walks a run of frames that must all check out, which a sealed body's
/// do, reporting a frame that does not rather than stopping short of it
pub(crate) fn walk_frames_exact(body: &[u8], visit: impl FnMut(&[u8]) -> Result<()>) -> Result<()> {
    let (valid, _) = walk_frames(body, visit)?;
    if valid != body.len() {
        return Err(ZyronError::CdcDecoderError(format!(
            "a change segment's frames end at byte {valid} of {}",
            body.len()
        )));
    }
    Ok(())
}

/// Makes a rename durable by syncing the containing directory. Windows has
/// no directory handle to sync, metadata durability rides on the volume
#[cfg(not(windows))]
pub(crate) fn sync_parent_dir(path: &Path) -> Result<()> {
    if let Some(parent) = path.parent() {
        File::open(parent)?.sync_all()?;
    }
    Ok(())
}

#[cfg(windows)]
pub(crate) fn sync_parent_dir(_path: &Path) -> Result<()> {
    Ok(())
}

// ---------------------------------------------------------------------------
// Inner state
// ---------------------------------------------------------------------------

/// The feed's counters as they stood when the segment being written was
/// last made durable, which is what a manifest written at any later instant
/// records.
///
/// The counters count every record the moment it lands, and the manifest is
/// written without waiting for the segment to be durable, at a seal, a
/// purge, a configuration change. A manifest holding counts of records that
/// a crash then takes with the segment's unsynced tail, or a reopen that
/// counts again what the manifest already counted, both move every
/// position the feed replicates as a count. So the manifest carries the
/// counts as of this mark, and a reopen counts the frames past it from
/// the files, which hold exactly what survived
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
struct SyncMark {
    /// The segment the mark is in
    seq: u64,
    /// Records written to the feed at the mark
    records_written: u64,
    /// The newest version at the mark and how many of its records had
    /// landed, so the version index is cut to the mark when written
    last_version: u64,
    last_records: u64,
    /// Records of the marked segment the counts cover, so a reopen counts
    /// the segment's frames from that one on, whether it finds the segment
    /// raw or sealed
    segment_records: u64,
}

/// The counters a batch of records advances as it is staged, taken before
/// the batch so a write that fails leaves the feed counting exactly what
/// its file holds
struct StageMark {
    open: SegmentSummary,
    versions: usize,
    last_records: Option<u64>,
    records_written: u64,
    ordinal_version: u64,
    next_ordinal: u64,
}

struct CdfInner {
    config: FeedConfig,
    /// Sealed segments, ascending by sequence
    sealed: Vec<SegmentSummary>,
    /// The segment appends land in
    open: SegmentSummary,
    /// The open segment's frames, held beside the file so a read of the
    /// newest changes never reopens the file. A segment seals at a few
    /// megabytes, which bounds what this holds, and it starts over at each
    /// seal
    open_frames: Vec<u8>,
    /// Where each version's first frame begins in `open_frames`, ascending,
    /// so a read starting at a version copies out only what follows it
    open_version_starts: Vec<(u64, usize)>,
    /// Held for the runs of one transaction's records an append collects,
    /// so a batch of any size allocates nothing to carry them
    span_scratch: Vec<(u64, u64, u64)>,
    /// True once the feed's files were removed, so a sealer that finishes
    /// afterwards writes nothing into a directory that is gone or reused
    closed: bool,
    /// The open segment's file, held open across appends. Appended to
    /// directly, since an append is written whole and forced out at once,
    /// and a buffer in front of it would copy every frame a second time
    /// on the way to the same write
    writer: Option<File>,
    /// True while a failed append may have left bytes past `open.bytes` in
    /// the open segment's file that could not be cut off, so the next
    /// append cuts them before it writes rather than framing records after
    /// a partial one
    torn_tail: bool,
    /// Bytes of the open segment already forced to durable storage
    synced_bytes: u64,
    /// Where in the log the oldest append not yet synced to its file was
    /// recorded, None once a sync covers every logged append. The log keeps
    /// its segments from here so the bytes can be put back after a crash
    logged_from: Option<zyron_wal::Lsn>,
    /// The position of the newest logged append's last record, so a
    /// manifest written while nothing is pending can say the log holds
    /// nothing this feed needs
    logged_through: u64,
    /// The log position replay starts from, as the manifest recorded it.
    /// A logged append below it is in a synced file already, and one at or
    /// above it is laid back into the segment file before the feed reopens
    replay_from: u64,
    /// The counters as of the last durable point, what a manifest records
    synced_mark: SyncMark,
    /// Closed segments whose raw file was not made durable through its end,
    /// which a rotation leaves until the next sync or the seal that
    /// replaces the file. The mark never moves past a segment listed here
    closed_unsynced: Vec<u64>,
    /// One entry per commit version the feed holds, ascending
    versions: Vec<VersionCount>,
    /// Records written since the feed was created, never reduced by a purge
    records_written: u64,
    /// Records a purge has removed
    records_purged: u64,
    /// Highest commit version a purge has reclaimed. A position at or below
    /// it names changes the feed no longer holds, which is what makes a
    /// stream reading from there stale rather than short
    purge_floor: u64,
    /// Sequence the next segment takes
    next_seq: u64,
    /// The version the open segment is numbering ordinals within
    ordinal_version: u64,
    /// Ordinal the next record of `ordinal_version` takes
    next_ordinal: u64,
    /// Bumped by every purge or compaction that rewrites a segment, so a
    /// reader that resolved a plan before a rewrite retries instead of
    /// reading a file that moved underneath it
    rewrite_epoch: u64,
    /// The first version each transaction with records in the feed wrote
    /// at, kept until the transaction is known to have ended. A read that
    /// must not move past an unfinished transaction's changes reads the
    /// lowest of these rather than walking the records to find it
    in_flight: HashMap<u64, u64>,
    /// Purges a byte cap made ahead of retention, which is what tells a
    /// stream reading from the feed's oldest change that the next one
    /// would take its changes
    cap_purges: u64,
    /// Every transaction with records in the feed, in the order their first
    /// records landed, so the spans ascend by first version
    spans: Vec<TxnSpan>,
    /// Where each transaction's span sits in `spans`
    span_index: HashMap<u64, usize>,
    /// The layouts rows have been written under, by schema epoch and
    /// whether the row holds the feed's column subset, which is what lets
    /// a seal slice a segment's rows into columns
    layouts: crate::segment_columns::RowLayouts,
    /// Numbers each manifest record encoded, so records encoded under the
    /// lock reach the file in the order they describe the feed in
    manifest_gen: std::sync::atomic::AtomicU64,
    /// Orders the records onto the file
    manifest_gate: Arc<ManifestGate>,
    /// What the records on disk describe so far
    chain: ManifestChain,
}

/// The form a segment's file takes on disk
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SegmentForm {
    /// Frames appended as they came, the segment still open or closed and
    /// not yet sealed
    Open,
    /// Sealed with the frames compressed as one block
    SealedFrames,
    /// Sealed with the records sliced into columns
    SealedColumns,
}

/// A manifest record encoded and not yet written
struct PendingManifest {
    generation: u64,
    bytes: Vec<u8>,
    /// True for a record that lays the file down whole, false for one
    /// appended after the records already there
    whole: bool,
    gate: Arc<ManifestGate>,
    /// True once the record reached the file, so a record dropped
    /// unwritten is reported to the gate as a gap in the chain
    done: bool,
}

impl Drop for PendingManifest {
    fn drop(&mut self) {
        if !self.done {
            self.gate.finish(self.generation, true);
        }
    }
}

/// Orders the manifest's records onto the file by generation.
///
/// Records are encoded under the feed's lock, in generation order, and
/// each delta describes what changed since the record encoded before it,
/// so the file has to hold them in that order. A writer whose turn has not
/// come waits for the records before its own, and a record dropped
/// unwritten leaves a gap the next record encoded closes by laying the
/// file down whole
struct ManifestGate {
    state: parking_lot::Mutex<ManifestGateState>,
    turn: parking_lot::Condvar,
}

struct ManifestGateState {
    /// The generation of the last record written or given up on
    written: u64,
    /// Generations given up on ahead of their turn
    skipped: Vec<u64>,
    /// True while a record since the last whole one never reached the
    /// file, which leaves the chain on disk short of what the records
    /// after it describe
    gap: bool,
}

impl ManifestGate {
    fn new(written: u64) -> Self {
        Self {
            state: parking_lot::Mutex::new(ManifestGateState {
                written,
                skipped: Vec::new(),
                gap: false,
            }),
            turn: parking_lot::Condvar::new(),
        }
    }

    /// Waits until every record before `generation` is written or given
    /// up on
    fn wait_turn(&self, generation: u64) {
        let mut state = self.state.lock();
        while state.written.saturating_add(1) < generation {
            self.turn.wait(&mut state);
        }
    }

    /// Records that `generation` reached the file, or was given up on when
    /// `gap` is true, and lets the next writer go
    fn finish(&self, generation: u64, gap: bool) {
        let mut state = self.state.lock();
        if gap {
            state.gap = true;
        }
        if generation == state.written.saturating_add(1) {
            state.written = generation;
            // Records given up on ahead of their turn are passed over now
            // that their turn has come
            while let Some(at) = state
                .skipped
                .iter()
                .position(|skipped| *skipped == state.written.saturating_add(1))
            {
                state.skipped.swap_remove(at);
                state.written = state.written.saturating_add(1);
            }
        } else if generation > state.written {
            state.skipped.push(generation);
        }
        self.turn.notify_all();
    }

    /// Whether a record since the last whole one was given up on, which
    /// the next record encoded answers by being a whole one
    fn take_gap(&self) -> bool {
        std::mem::take(&mut self.state.lock().gap)
    }
}

/// What the manifest's records on disk describe of the feed so far, so a
/// delta record carries what changed since the record before it
#[derive(Default)]
struct ManifestChain {
    /// Entries of the version index the last record encoded reaches
    versions_encoded: usize,
    /// Sealed segments added or replaced since the last record encoded
    sealed_dirty: Vec<u64>,
    /// Spans opened or extended since the last record encoded, by their
    /// position in the span list
    spans_dirty: Vec<usize>,
    /// Delta records encoded since the last whole one
    deltas_since_whole: u32,
    /// True when the next record has to be a whole one, because the index
    /// was pruned or rebuilt, or no record exists yet
    whole_next: bool,
}

/// What one feed holds at one instant, read under its lock
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FeedBoundary {
    /// The newest version the feed holds, zero when it is empty
    pub latest: u64,
    /// Records written since the feed was created
    pub records: u64,
    /// The lowest version any transaction that has not ended wrote at, None
    /// when every record belongs to an ended transaction
    pub first_open: Option<u64>,
}

/// The counters every manifest record carries, as read back
struct ManifestCounters {
    config: FeedConfig,
    records_written: u64,
    records_purged: u64,
    purge_floor: u64,
    next_seq: u64,
    open_seq: u64,
    mark: SyncMark,
    replay_from: u64,
}

impl CdfInner {
    /// The log position replay starts from as things stand. The oldest
    /// logged append not yet synced when there is one, otherwise the
    /// position past the newest logged append, which every logged append
    /// is below
    fn replay_from_now(&self) -> u64 {
        match self.logged_from {
            Some(lsn) => lsn.0,
            None => self.logged_through.saturating_add(1),
        }
    }

    /// The counters as they stand before a batch is staged
    fn mark(&self) -> StageMark {
        StageMark {
            open: self.open.clone(),
            versions: self.versions.len(),
            last_records: self.versions.last().map(|v| v.records),
            records_written: self.records_written,
            ordinal_version: self.ordinal_version,
            next_ordinal: self.next_ordinal,
        }
    }

    /// Takes a staged batch back off the open segment, the frames past
    /// `base` and the version starts past `starts_base`, and puts the
    /// counters back to `mark`, for bytes that never reached the file
    fn unstage(&mut self, mark: StageMark, base: usize, starts_base: usize) {
        self.open_frames.truncate(base);
        self.open_version_starts.truncate(starts_base);
        self.restore(mark);
    }

    /// Puts the counters back to `mark`, for a batch whose bytes never
    /// reached the file
    fn restore(&mut self, mark: StageMark) {
        self.open = mark.open;
        self.versions.truncate(mark.versions);
        if let (Some(last), Some(records)) = (self.versions.last_mut(), mark.last_records) {
            last.records = records;
        }
        self.records_written = mark.records_written;
        self.ordinal_version = mark.ordinal_version;
        self.next_ordinal = mark.next_ordinal;
    }

    /// Forgets the transactions `ended` says are over and answers with the
    /// lowest version the rest wrote at
    fn first_open_version(&mut self, ended: &dyn Fn(u64) -> bool) -> Option<u64> {
        self.in_flight.retain(|txn_id, _| !ended(*txn_id));
        self.in_flight.values().copied().min()
    }

    /// What the feed holds right now, with the unfinished transactions
    /// pruned to the ones still open
    fn boundary(&mut self, ended: &dyn Fn(u64) -> bool) -> FeedBoundary {
        FeedBoundary {
            latest: self.versions.last().map(|v| v.version).unwrap_or(0),
            records: self.records_written,
            first_open: self.first_open_version(ended),
        }
    }

    /// Records written before the record at `ordinal` within `version`,
    /// counted from the feed's creation
    fn records_before(&self, version: u64, ordinal: u64) -> u64 {
        version_index::records_before(&self.versions, version, ordinal)
    }

    /// The version and ordinal of the record at which the feed had written
    /// exactly `count` records, None when the count is zero or names a
    /// record retention has reclaimed
    fn cursor_at_count(&self, count: u64) -> Option<(u64, u64)> {
        version_index::cursor_at_count(&self.versions, count)
    }

    /// Every segment, sealed then open, for the paths that walk all of them
    fn all_segments(&self) -> Vec<SegmentSummary> {
        let mut all = Vec::with_capacity(self.sealed.len() + 1);
        all.extend(self.sealed.iter().cloned());
        if !self.open.is_empty() {
            all.push(self.open.clone());
        }
        all
    }

    fn summary_for(&self, seq: u64) -> Option<&SegmentSummary> {
        if self.open.seq == seq && !self.open.is_empty() {
            return Some(&self.open);
        }
        self.sealed.iter().find(|s| s.seq == seq)
    }

    /// Records the feed wrote at or below `version`, counted from creation.
    /// With every version purged, everything written lies below any version
    fn records_at_or_below(&self, version: u64) -> u64 {
        version_index::records_at_or_below(&self.versions, version).unwrap_or(self.records_written)
    }

    /// The first version strictly above `version`, with its timestamp
    fn first_after(&self, version: u64) -> Option<VersionCount> {
        version_index::first_after(&self.versions, version)
    }

    /// The counters as they stand, taken as the mark once the open segment
    /// is durable through its end
    fn take_mark(&self) -> SyncMark {
        SyncMark {
            seq: self.open.seq,
            records_written: self.records_written,
            last_version: self.versions.last().map(|v| v.version).unwrap_or(0),
            last_records: self.versions.last().map(|v| v.records).unwrap_or(0),
            segment_records: self.open.record_count,
        }
    }

    fn observe_version(&mut self, version: u64, timestamp: i64) {
        match self.versions.last_mut() {
            Some(last) if last.version == version => {
                last.records += 1;
            }
            _ => {
                let prior = self.records_written;
                self.versions.push(VersionCount {
                    version,
                    records: 1,
                    prior,
                    first_timestamp: timestamp,
                });
            }
        }
        self.records_written += 1;
    }

    /// Extends a transaction's span over `first..=last`, opening one at
    /// its first record
    fn observe_span(&mut self, txn_id: u64, first: u64, last: u64) {
        match self.span_index.get(&txn_id) {
            Some(&at) => {
                if let Some(span) = self.spans.get_mut(at) {
                    span.last = span.last.max(last);
                }
                self.chain.spans_dirty.push(at);
            }
            None => {
                self.span_index.insert(txn_id, self.spans.len());
                self.chain.spans_dirty.push(self.spans.len());
                self.spans.push(TxnSpan {
                    txn_id,
                    first,
                    last,
                });
            }
        }
    }

    /// Replaces the open segment's in-memory frames with what its file now
    /// holds, after a seal, a rewrite or a recovery replaced the file
    fn reset_open_frames(&mut self, frames: Vec<u8>) {
        self.open_version_starts = frame_version_starts(&frames);
        self.open_frames = frames;
    }

    /// The open segment's frames from the first frame of `from_version` on,
    /// copied out under the lock so a read walks them without the file
    fn open_frames_from(&self, from_version: u64) -> Vec<u8> {
        let at = self
            .open_version_starts
            .partition_point(|(version, _)| *version < from_version);
        let offset = self
            .open_version_starts
            .get(at)
            .map(|(_, offset)| *offset)
            .unwrap_or(self.open_frames.len());
        self.open_frames[offset..].to_vec()
    }

    /// The version the next record lands at or above, so versions ascend in
    /// the order records are written. A batch whose version is below the
    /// newest one already here, which a statement that logged before another
    /// but reached the feed after it produces, is written at that newest
    /// version instead, so the per-version index stays a prefix count and a
    /// reader never finds a version below one it has already passed
    fn floor(&self) -> u64 {
        self.versions.last().map(|v| v.version).unwrap_or(0)
    }

    /// Frames one change onto the open segment's bytes and counts it,
    /// answering with the version it landed at. A preimage on a feed that
    /// keeps one image is not recorded. Ordinals are assigned under the
    /// lock because they number a commit's records, and two batches of one
    /// commit must not both start at zero.
    ///
    /// The frame is written where the segment keeps it, so the bytes the
    /// file takes and the bytes a reader walks are the same bytes. A batch
    /// that fails to reach the file cuts them back off
    fn stage(
        &mut self,
        header: &ChangeHeader,
        change: &RowChange<'_>,
        floor: &mut u64,
    ) -> Option<u64> {
        if !self.config.before_image && change.change_type == ChangeType::UpdatePreimage {
            return None;
        }
        let version = header.commit_version.max(*floor);
        *floor = version;
        let ordinal = self.next_ordinal_for(version);
        // A version already open in the segment keeps its first frame, a
        // batch only ever continues the newest one
        let at = self.open_frames.len();
        if self
            .open_version_starts
            .last()
            .is_none_or(|(last, _)| *last != version)
        {
            self.open_version_starts.push((version, at));
        }
        frame_row(&mut self.open_frames, header, version, change, ordinal);
        self.open.observe(
            version,
            header.commit_timestamp,
            change.change_type,
            header.schema_version,
        );
        self.observe_version(version, header.commit_timestamp);
        Some(version)
    }

    /// Rebuilds the span index after the spans were replaced or trimmed.
    /// The positions a delta would name moved, so the next manifest record
    /// is a whole one
    fn reindex_spans(&mut self) {
        self.span_index = self
            .spans
            .iter()
            .enumerate()
            .map(|(at, span)| (span.txn_id, at))
            .collect();
        self.chain.spans_dirty.clear();
        self.chain.whole_next = true;
    }

    /// The version a read that resumes after `consumed` records and takes
    /// at most `max_rows` more ends at, None when fewer than that many
    /// records lie past the position.
    ///
    /// The cut lands on the version holding the record at that count and
    /// then moves past every transaction that wrote inside the read and
    /// again beyond it, so the read hands over whole transactions
    fn bounded_cut(&self, from_exclusive: u64, consumed: u64, max_rows: u64) -> Option<u64> {
        let cut = version_index::bounded_cut(&self.versions, consumed, max_rows)?;
        Some(self.extend_cut(from_exclusive, cut))
    }

    /// Moves `cut` past every transaction that wrote inside
    /// `(from_exclusive, cut]` and again beyond it.
    ///
    /// A transaction that began before the read's start is wholly before it:
    /// the read that ended there moved past everything such a transaction
    /// wrote, and one still open held that read short of its first record.
    /// So only the spans that open inside the read are walked, and each one
    /// that reaches past the cut moves it
    fn extend_cut(&self, from_exclusive: u64, mut cut: u64) -> u64 {
        let start = self
            .spans
            .partition_point(|span| span.first <= from_exclusive);
        for span in &self.spans[start..] {
            if span.first > cut {
                break;
            }
            if span.last > cut {
                cut = span.last;
            }
        }
        cut
    }

    /// The transactions whose first record lies in `(from_exclusive,
    /// to_inclusive]`, in the order their first records landed. A
    /// transaction that began before the read's start is wholly before it,
    /// for the reason `extend_cut` gives
    fn txns_opening_in(&self, from_exclusive: u64, to_inclusive: u64) -> Vec<u64> {
        let start = self
            .spans
            .partition_point(|span| span.first <= from_exclusive);
        self.spans[start..]
            .iter()
            .take_while(|span| span.first <= to_inclusive)
            .map(|span| span.txn_id)
            .collect()
    }

    /// The lowest and highest version a transaction wrote at in this feed,
    /// None for one with no records here
    fn span_of(&self, txn_id: u64) -> Option<(u64, u64)> {
        self.span_index
            .get(&txn_id)
            .and_then(|at| self.spans.get(*at))
            .map(|span| (span.first, span.last))
    }

    /// Assigns the next ordinal within a commit version
    #[inline]
    fn next_ordinal_for(&mut self, version: u64) -> u64 {
        if self.ordinal_version != version {
            self.ordinal_version = version;
            self.next_ordinal = 0;
        }
        let ordinal = self.next_ordinal;
        self.next_ordinal += 1;
        ordinal
    }
}

// ---------------------------------------------------------------------------
// CompactionView
// ---------------------------------------------------------------------------

/// The open segment's frames as a read took them under the feed's lock,
/// from the first version the read wants on
struct OpenTail {
    seq: u64,
    frames: Vec<u8>,
}

/// The frames of one segment. The open segment's come from the tail a read
/// took under the lock, a sealed segment's from its file. None for a file
/// too short to hold a header
fn segment_frames_in(
    dir: &Path,
    table_id: u32,
    seq: u64,
    open: Option<&OpenTail>,
) -> Result<Option<Vec<u8>>> {
    if let Some(open) = open {
        return Ok(Some(open.frames.clone()));
    }
    let _load = zyron_common::profile::scope(zyron_common::profile::Phase::ChangeScanLoad);
    let path = ChangeDataFeed::segment_path_in(dir, seq);
    let mut bytes = Vec::new();
    {
        let _read = zyron_common::profile::scope(zyron_common::profile::Phase::ChangeScanRead);
        let mut file = File::open(&path)?;
        // Sized from the file so the read lands in one allocation rather
        // than growing through several
        let len = file.metadata().map(|m| m.len() as usize).unwrap_or(0);
        bytes.reserve_exact(len);
        file.read_to_end(&mut bytes)?;
    }
    if bytes.len() < SEGMENT_BODY_OFFSET as usize {
        return Ok(None);
    }
    let header = read_segment_header(&bytes)?;
    if header.table_id != table_id {
        return Err(ZyronError::CdcDecoderError(format!(
            "change segment {} carries table id {}, expected {}",
            path.display(),
            header.table_id,
            table_id
        )));
    }
    let body_start = SEGMENT_BODY_OFFSET as usize;
    if !header.sealed {
        return Ok(Some(bytes[body_start..].to_vec()));
    }
    let trailer = ChangeDataFeed::read_trailer(&bytes)?;
    let trailer_len = ChangeDataFeed::encode_trailer(&trailer).len();
    let body_end = bytes.len().saturating_sub(trailer_len);
    if header.columnar {
        let sliced =
            crate::segment_columns::ColumnarSegment::parse(bytes[body_start..body_end].to_vec())?;
        return Ok(Some(sliced.frames()?));
    }
    if body_end < body_start + 8 {
        return Err(ZyronError::CdcDecoderError(format!(
            "sealed change segment {} carries no block length",
            path.display()
        )));
    }
    let mut len_bytes = [0u8; 8];
    len_bytes.copy_from_slice(&bytes[body_start..body_start + 8]);
    let plain_len = u64::from_le_bytes(len_bytes) as usize;
    let block = &bytes[body_start + 8..body_end];
    Ok(Some(header.codec.decompress(block, plain_len)?))
}

/// The column-sliced form of one sealed segment, None for a segment that
/// is not sealed or not sliced, which is read as frames instead.
///
/// The file is opened and left open behind the segment, which reads its
/// header off the front and its trailer off the back and fetches each
/// block it is asked for from where the directory says it stands, so a
/// read of two columns moves two columns' bytes through memory
fn segment_columns_in(
    dir: &Path,
    table_id: u32,
    seq: u64,
) -> Result<Option<crate::segment_columns::ColumnarSegment>> {
    let _load = zyron_common::profile::scope(zyron_common::profile::Phase::ChangeScanLoad);
    let path = ChangeDataFeed::segment_path_in(dir, seq);
    let mut file = File::open(&path)?;
    let len = file.metadata()?.len() as usize;
    if len < SEGMENT_BODY_OFFSET as usize + TRAILER_SUFFIX_LEN {
        return Ok(None);
    }
    let mut front = vec![0u8; SEGMENT_BODY_OFFSET as usize];
    file.read_exact(&mut front)?;
    let header = read_segment_header(&front)?;
    if header.table_id != table_id {
        return Err(ZyronError::CdcDecoderError(format!(
            "change segment {} carries table id {}, expected {}",
            path.display(),
            header.table_id,
            table_id
        )));
    }
    if !header.sealed || !header.columnar {
        return Ok(None);
    }
    // The trailer's length is in its last bytes, so the tail is read twice
    // at most, once for the suffix and once for the whole trailer
    let mut suffix = vec![0u8; TRAILER_SUFFIX_LEN];
    crate::segment_columns::read_file_at(&file, (len - TRAILER_SUFFIX_LEN) as u64, &mut suffix)?;
    let mut len_bytes = [0u8; 4];
    len_bytes.copy_from_slice(&suffix[0..4]);
    let trailer_body = u32::from_le_bytes(len_bytes) as usize;
    let trailer_len = trailer_body + TRAILER_SUFFIX_LEN;
    if trailer_len + SEGMENT_BODY_OFFSET as usize > len {
        return Err(ZyronError::CdcDecoderError(
            "sealed change segment declares a trailer longer than the file".into(),
        ));
    }
    let mut tail = vec![0u8; trailer_len];
    crate::segment_columns::read_file_at(&file, (len - trailer_len) as u64, &mut tail)?;
    // Verified the way a whole file's trailer is, so a torn tail is
    // reported rather than read as a directory
    ChangeDataFeed::read_trailer(&tail)?;
    let body_len = len - trailer_len - SEGMENT_BODY_OFFSET as usize;
    Ok(Some(crate::segment_columns::ColumnarSegment::open(
        file,
        SEGMENT_BODY_OFFSET,
        body_len,
    )?))
}

/// A read of one range that hands its records over a batch at a time and
/// keeps its place between batches.
///
/// The segments the range reaches are decided once, and each is loaded
/// once, when the scan reaches it. Between two calls the scan holds the
/// segment it stopped inside and the offset it stopped at, so a read of
/// ten million changes costs one pass over its files rather than one per
/// batch. A segment purged or compacted away while the scan stands before
/// it is reported rather than read past
pub struct SegmentScan {
    dir: PathBuf,
    table_id: u32,
    range: ChangeRange,
    /// The last position already handed over before this scan began,
    /// which the first segment's records are read past
    resume: Option<(u64, u64)>,
    plan: ReadPlan,
    /// Which of the plan's segments the scan stands in
    at: usize,
    /// The open segment's frames as they stood when the scan planned
    open: Option<OpenTail>,
    /// The frames of the segment the scan stands in and the offset it
    /// resumes at
    current: Option<(Vec<u8>, usize)>,
}

impl SegmentScan {
    /// Hands records to the visitor from where the scan stands until the
    /// visitor answers false or the range runs out. Answers false once the
    /// range is exhausted
    pub fn next(
        &mut self,
        visit: &mut dyn FnMut(ChangeRecordRef<'_>) -> Result<bool>,
    ) -> Result<bool> {
        loop {
            if self.current.is_none() {
                let Some(seq) = self.plan.segments.get(self.at).copied() else {
                    return Ok(false);
                };
                let tail = self.open.as_ref().filter(|open| open.seq == seq);
                let frames =
                    segment_frames_in(&self.dir, self.table_id, seq, tail)?.unwrap_or_default();
                self.current = Some((frames, 0));
            }
            let Some((frames, offset)) = self.current.as_mut() else {
                return Ok(false);
            };
            let (next_offset, stopped) = walk_admitted(
                frames,
                *offset,
                self.table_id,
                &self.range,
                self.resume,
                visit,
            )?;
            *offset = next_offset;
            if stopped {
                return Ok(true);
            }
            self.current = None;
            self.at += 1;
        }
    }

    /// Takes the rest of the scan as one part per segment, each loading and
    /// walking on its own, leaving the scan with nothing to hand over
    /// through `next`. The segment the scan stands inside carries its
    /// frames and the offset it stopped at, so nothing is handed over twice
    pub fn take_parts(&mut self) -> Vec<SegmentPart> {
        let mut parts = Vec::with_capacity(self.plan.segments.len().saturating_sub(self.at));
        let mut loaded = self.current.take();
        let mut open = self.open.take();
        for &seq in &self.plan.segments[self.at.min(self.plan.segments.len())..] {
            parts.push(SegmentPart {
                dir: self.dir.clone(),
                table_id: self.table_id,
                seq,
                range: self.range.clone(),
                resume: self.resume,
                open: open.take_if(|tail| tail.seq == seq),
                loaded: loaded.take(),
            });
        }
        self.at = self.plan.segments.len();
        parts
    }

    /// What the scan planned, the segments it reaches and the ones it
    /// pruned
    pub fn plan(&self) -> &ReadPlan {
        &self.plan
    }
}

/// Walks the frames from `offset` handing the records the range admits and
/// the resume point does not exclude to the visitor, answering with the
/// offset after the last frame visited and whether the visitor stopped the
/// walk
fn walk_admitted(
    frames: &[u8],
    offset: usize,
    table_id: u32,
    range: &ChangeRange,
    resume: Option<(u64, u64)>,
    visit: &mut dyn FnMut(ChangeRecordRef<'_>) -> Result<bool>,
) -> Result<(usize, bool)> {
    let mut stopped = false;
    let next_offset = walk_frames_from(frames, offset, |record| {
        let view = ChangeRecordRef::of(record, table_id)?;
        // The three metadata columns a predicate prunes on are read out of
        // the record header, so a record the range excludes costs no field
        // decoding at all
        if !range.admits(view.commit_version, view.commit_timestamp, view.change_type) {
            return Ok(true);
        }
        if let Some((at_version, at_ordinal)) = resume {
            let past = view.commit_version > at_version
                || (view.commit_version == at_version && view.change_ordinal > at_ordinal);
            if !past {
                return Ok(true);
            }
        }
        let keep_going = visit(view)?;
        if !keep_going {
            stopped = true;
        }
        Ok(keep_going)
    })?;
    Ok((next_offset, stopped))
}

/// One segment of a scan, loaded and walked apart from the others.
///
/// What a worker decoding a window across cores takes. Loading is the file
/// read and the decompression, which is where a read of a wide table spends
/// most of its time, so each part carries everything it needs to load
/// itself on whichever thread runs it
pub struct SegmentPart {
    dir: PathBuf,
    table_id: u32,
    seq: u64,
    range: ChangeRange,
    resume: Option<(u64, u64)>,
    /// The open segment's frames as the scan took them, when this is it
    open: Option<OpenTail>,
    /// The frames and the offset a scan that stood inside this segment
    /// handed over with it
    loaded: Option<(Vec<u8>, usize)>,
}

impl SegmentPart {
    /// The segment this part reads
    pub fn seq(&self) -> u64 {
        self.seq
    }

    /// The range the part admits records from and the position it resumes
    /// after, for a reader that judges records from their columns
    pub fn admission(&self) -> (&ChangeRange, Option<(u64, u64)>) {
        (&self.range, self.resume)
    }

    /// The segment in its column-sliced form, None when it is the open
    /// segment, a segment the scan already loaded as frames, or a sealed
    /// segment stored as frames. A part answering None is read through
    /// `visit`
    pub fn columns(&mut self) -> Result<Option<crate::segment_columns::ColumnarSegment>> {
        if self.open.is_some() || self.loaded.is_some() {
            return Ok(None);
        }
        segment_columns_in(&self.dir, self.table_id, self.seq)
    }

    /// Loads the segment and hands its admitted records to the visitor
    /// until the visitor answers false or the segment ends
    pub fn visit(
        &mut self,
        visit: &mut dyn FnMut(ChangeRecordRef<'_>) -> Result<bool>,
    ) -> Result<()> {
        let (frames, offset) = match self.loaded.take() {
            Some(loaded) => loaded,
            None => {
                let frames =
                    segment_frames_in(&self.dir, self.table_id, self.seq, self.open.as_ref())?
                        .unwrap_or_default();
                (frames, 0)
            }
        };
        walk_admitted(
            &frames,
            offset,
            self.table_id,
            &self.range,
            self.resume,
            visit,
        )?;
        Ok(())
    }
}

/// What a compaction pass works over, the sealed segments as they stood
/// when the pass was planned, so the pass reads each one on its own and
/// never holds the feed whole, and the epoch that says whether they still
/// stand
pub struct CompactionPlan {
    segments: Vec<SegmentSummary>,
    epoch: u64,
    layouts: crate::segment_columns::RowLayouts,
    /// Records the planned segments hold
    pub record_count: u64,
}

// ---------------------------------------------------------------------------
// ChangeDataFeed
// ---------------------------------------------------------------------------

/// Per-table change data feed backed by a directory of change segments
pub struct ChangeDataFeed {
    pub table_id: u32,
    /// The branch whose feed on the table this is, zero for the table's
    /// own. Named in every logged append so recovery finds the directory
    branch_id: u64,
    dir: PathBuf,
    /// Shared with the sealer threads, which finish a closed segment off
    /// the write path and then bring the summary and manifest up to date
    /// under this lock
    inner: Arc<Mutex<CdfInner>>,
    enabled: AtomicBool,
    /// True while records landed since the last sync, read without the
    /// lock by a sync pass so a feed with nothing pending costs it nothing
    unsynced: AtomicBool,
    counters: Arc<FeedCounters>,
    /// The log every append's bytes are recorded in, attached once by the
    /// registry that holds the writer. An append is durable with the
    /// transaction's commit record through this log rather than through a
    /// sync of the segment file per commit, and recovery lays the logged
    /// bytes back into the file before the feed reopens. A feed with no
    /// log attached is durable only through its sync passes
    log: std::sync::OnceLock<Arc<zyron_wal::WalWriter>>,
}

/// What the feed answers without its lock, kept by every path that changes
/// them, the sealer threads included
struct FeedCounters {
    record_count: AtomicU64,
    bytes: AtomicU64,
    /// Latest commit version the feed holds, read without the lock by the
    /// paths that only need to know whether anything is pending
    latest_version: AtomicU64,
}

impl FeedCounters {
    fn publish(&self, inner: &CdfInner) {
        self.record_count.store(
            inner.records_written.saturating_sub(inner.records_purged),
            Ordering::Release,
        );
        let bytes = inner.sealed.iter().map(|s| s.bytes).sum::<u64>() + inner.open.bytes;
        self.bytes.store(bytes, Ordering::Release);
    }
}

/// A closed segment on its way to being sealed, run on a thread of its own
/// so the writer that filled it pays nothing for the compression, the
/// fsyncs or the manifest rewrite.
///
/// The segment's file is complete and flushed on disk in its raw form and
/// stays readable throughout. The sealed form replaces it by rename, and
/// the manifest is rewritten under the feed's lock afterwards, which is
/// also where a rewrite or a removal that overtook the seal is noticed and
/// the seal abandoned
struct SealJob {
    dir: PathBuf,
    table_id: u32,
    summary: SegmentSummary,
    frames: Vec<u8>,
    /// The layouts as they stood when the segment closed, which every row
    /// in it was written under
    layouts: crate::segment_columns::RowLayouts,
    epoch: u64,
    /// The counts through the end of the segment, taken as the mark once
    /// the sealed file is durable
    close_mark: SyncMark,
    inner: Arc<Mutex<CdfInner>>,
    counters: Arc<FeedCounters>,
}

impl SealJob {
    /// Seals on the calling thread, taking the feed's lock for the
    /// bookkeeping alone.
    ///
    /// The compression and the sealed file's fsync come before the lock,
    /// the rename and the summary update happen under it, and the
    /// directory sync, the manifest write and the next file's header come
    /// after it, so a writer appending meanwhile waits for none of the
    /// disk work
    fn run(self) -> Result<()> {
        let (sealed, tmp) = seal_segment_files(
            &self.dir,
            self.table_id,
            &self.summary,
            &self.frames,
            &self.layouts,
        )?;
        let dir = self.dir.clone();
        let table_id = self.table_id;
        let inner = Arc::clone(&self.inner);
        let after = {
            let mut inner = inner.lock();
            self.install(&mut inner, sealed, tmp)?
        };
        match after {
            Some(after) => after.finish(&dir, table_id),
            None => Ok(()),
        }
    }

    /// Seals with the feed's lock already held by the caller
    fn run_locked(self, inner: &mut CdfInner) -> Result<()> {
        let (sealed, tmp) = seal_segment_files(
            &self.dir,
            self.table_id,
            &self.summary,
            &self.frames,
            &self.layouts,
        )?;
        let dir = self.dir.clone();
        let table_id = self.table_id;
        match self.install(inner, sealed, tmp)? {
            Some(after) => after.finish(&dir, table_id),
            None => Ok(()),
        }
    }

    /// Puts the sealed file in place of the raw one and records it, unless
    /// a purge or a compaction rewrote the segment meanwhile or the feed was
    /// removed. What those left is what stands, and the raw form a rewrite
    /// leaves reads the same as the sealed one would. Answers with the
    /// durability work left to do off the lock, None when nothing was
    /// installed
    fn install(
        self,
        inner: &mut CdfInner,
        sealed: SegmentSummary,
        tmp: PathBuf,
    ) -> Result<Option<SealAfterInstall>> {
        let seq = self.summary.seq;
        let path = ChangeDataFeed::segment_path_in(&self.dir, seq);
        if inner.closed || inner.rewrite_epoch != self.epoch {
            let _ = fs::remove_file(&tmp);
            return Ok(None);
        }
        fs::rename(&tmp, &path)?;
        if let Some(slot) = inner.sealed.iter_mut().find(|s| s.seq == seq) {
            *slot = sealed;
            inner.chain.sealed_dirty.push(seq);
        }
        // The sealed file is durable, so the counts through this segment's
        // end may stand as the mark, unless a sync already took the mark
        // further or a closed segment before this one is still unsynced,
        // which a later seal or sync moves the mark past
        inner.closed_unsynced.retain(|closed| *closed != seq);
        if inner.synced_mark.seq <= seq && inner.closed_unsynced.iter().all(|closed| *closed > seq)
        {
            inner.synced_mark = self.close_mark;
        }
        self.counters.publish(inner);
        Ok(Some(SealAfterInstall {
            sealed_path: path,
            manifest: ChangeDataFeed::encode_manifest_pending(inner),
            next_seq: inner.open.seq + 1,
        }))
    }
}

/// The durability work a seal finishes once the feed's lock is released
struct SealAfterInstall {
    sealed_path: PathBuf,
    manifest: PendingManifest,
    /// The segment after the open one, created ahead so the next rotation
    /// adopts a file whose header is already on disk rather than syncing
    /// one on the write path
    next_seq: u64,
}

impl SealAfterInstall {
    fn finish(self, dir: &Path, table_id: u32) -> Result<()> {
        // The rename is durable before the manifest names the file sealed
        sync_parent_dir(&self.sealed_path)?;
        ChangeDataFeed::write_manifest(dir, self.manifest)?;
        ChangeDataFeed::prepare_segment(dir, table_id, self.next_seq)
    }
}

/// Writes a segment's sealed form beside its raw file, compressed with the
/// summary's codec and made durable, and answers with the summary as the
/// sealed file describes it and the path the caller renames into place.
///
/// The body is column-sliced when every row's layout is known, and the
/// frames compressed as one block otherwise. The trailer carries the
/// summary as it stood when the segment closed, so what a reader recovers
/// from the file is what the feed counted
fn seal_segment_files(
    dir: &Path,
    table_id: u32,
    summary: &SegmentSummary,
    frames: &[u8],
    layouts: &crate::segment_columns::RowLayouts,
) -> Result<(SegmentSummary, PathBuf)> {
    let seq = summary.seq;
    let codec = summary.codec;
    let path = ChangeDataFeed::segment_path_in(dir, seq);
    // Named for the seal, apart from the name a purge's rewrite of the same
    // segment takes, so the two never write through one path
    let tmp = path.with_extension("zycdf.seal");
    let (body, flags) = sealed_body(frames, codec, layouts)?;
    let mut sealed = summary.clone();
    sealed.sealed = true;
    let trailer = ChangeDataFeed::encode_trailer(&sealed);
    {
        let file = File::create(&tmp)?;
        let mut writer = BufWriter::new(file);
        write_segment_header(&mut writer, table_id, seq, flags)?;
        writer.write_all(&body)?;
        writer.write_all(&trailer)?;
        writer.flush()?;
        writer.get_ref().sync_all()?;
    }
    sealed.bytes = SEGMENT_BODY_OFFSET + body.len() as u64 + trailer.len() as u64;
    Ok((sealed, tmp))
}

/// A sealed segment's body and the header flags that describe it, the
/// column-sliced form when every row's layout is known and one compressed
/// block of frames otherwise
fn sealed_body(
    frames: &[u8],
    codec: CdfCodec,
    layouts: &crate::segment_columns::RowLayouts,
) -> Result<(Vec<u8>, u32)> {
    let codec_flags = FLAG_SEALED | ((codec as u32) << FLAG_CODEC_SHIFT);
    if let Some(sliced) = crate::segment_columns::encode(frames, layouts, codec)? {
        return Ok((sliced, codec_flags | FLAG_COLUMNAR));
    }
    let compressed = codec.compress(frames)?;
    let mut body = Vec::with_capacity(8 + compressed.len());
    body.extend_from_slice(&(frames.len() as u64).to_le_bytes());
    body.extend_from_slice(&compressed);
    Ok((body, codec_flags))
}

impl ChangeDataFeed {
    /// Opens or creates a feed with the default configuration derived from a
    /// day count
    pub fn open(data_dir: &Path, table_id: u32, retention_days: u32) -> Result<Self> {
        Self::open_with_config(
            data_dir,
            table_id,
            FeedConfig::from_retention_days(retention_days),
        )
    }

    /// Opens or creates a feed. A manifest already on disk wins over the
    /// supplied configuration for everything but the enabled flag, so a
    /// restart keeps what the operator set rather than the caller's default
    pub fn open_with_config(data_dir: &Path, table_id: u32, config: FeedConfig) -> Result<Self> {
        Self::open_in(Self::table_dir(data_dir, table_id), table_id, config)
    }

    /// Opens or creates a feed in `dir`, which is the table's own directory
    /// for the table's feed and a directory beneath it for a branch's
    pub fn open_in(dir: PathBuf, table_id: u32, config: FeedConfig) -> Result<Self> {
        fs::create_dir_all(&dir).map_err(|e| {
            ZyronError::CdcStreamError(format!(
                "failed to create the change feed directory {}: {e}",
                dir.display()
            ))
        })?;

        let mut inner = match Self::read_manifest(&dir)? {
            Some(mut loaded) => {
                loaded.config.enabled = config.enabled;
                loaded
            }
            None => CdfInner {
                layouts: crate::segment_columns::RowLayouts::new(),
                manifest_gen: std::sync::atomic::AtomicU64::new(0),
                manifest_gate: Arc::new(ManifestGate::new(0)),
                chain: ManifestChain {
                    whole_next: true,
                    ..ManifestChain::default()
                },
                config,
                sealed: Vec::new(),
                open: SegmentSummary::empty(1, CdfCodec::default()),
                writer: None,
                torn_tail: false,
                synced_bytes: SEGMENT_BODY_OFFSET,
                logged_from: None,
                logged_through: 0,
                replay_from: 0,
                synced_mark: SyncMark {
                    seq: 1,
                    ..SyncMark::default()
                },
                closed_unsynced: Vec::new(),
                versions: Vec::new(),
                records_written: 0,
                records_purged: 0,
                purge_floor: 0,
                next_seq: 2,
                ordinal_version: u64::MAX,
                next_ordinal: 0,
                rewrite_epoch: 0,
                in_flight: HashMap::new(),
                open_frames: Vec::new(),
                open_version_starts: Vec::new(),
                span_scratch: Vec::new(),
                closed: false,
                cap_purges: 0,
                spans: Vec::new(),
                span_index: HashMap::new(),
            },
        };
        inner.open.codec = inner.config.codec;

        // The manifest counts records through its mark, and the frames past
        // it are counted here from the files, which is what makes the write
        // path free of a manifest write per batch. A rotation the writer
        // made ahead of the manifest shows as a segment past the open one
        // already holding frames, and every such segment is walked in turn
        Self::recover_open_chain(&dir, table_id, &mut inner)?;
        Self::seal_raw_segments(&dir, table_id, &mut inner)?;
        // Every file is durable as recovered, so the counts stand as the
        // mark until the next manifest write
        inner.closed_unsynced.clear();
        inner.synced_mark = inner.take_mark();
        // Ordinals continue from the newest version's count, so a record
        // written after a restart never repeats the ordinal of one written
        // before it at the same version
        if let Some(last) = inner.versions.last() {
            inner.ordinal_version = last.version;
            inner.next_ordinal = last.records;
        }

        let enabled = inner.config.enabled;
        let record_count = inner.records_written - inner.records_purged;
        let bytes = inner.sealed.iter().map(|s| s.bytes).sum::<u64>() + inner.open.bytes;
        let latest = inner.versions.last().map(|v| v.version).unwrap_or(0);

        Ok(Self {
            table_id,
            branch_id: 0,
            dir,
            inner: Arc::new(Mutex::new(inner)),
            enabled: AtomicBool::new(enabled),
            // Whatever a reopen found in the files is synced once by the
            // first pass rather than assumed durable
            unsynced: AtomicBool::new(true),
            counters: Arc::new(FeedCounters {
                record_count: AtomicU64::new(record_count),
                bytes: AtomicU64::new(bytes),
                latest_version: AtomicU64::new(latest),
            }),
            log: std::sync::OnceLock::new(),
        })
    }

    /// Names the branch this feed records for, so every logged append
    /// carries it and recovery finds the branch's directory
    pub fn on_branch(mut self, branch_id: u64) -> Self {
        self.branch_id = branch_id;
        self
    }

    /// Attaches the log every append's bytes are recorded in. The first
    /// attachment is the one kept.
    ///
    /// Everything the log held for this feed at attachment is in its
    /// files, since recovery laid the logged bytes back before the feed
    /// opened and the open synced what it found, so the position replay
    /// would start from moves past every record the log holds now. Left
    /// where the manifest had it, a manifest written before the next
    /// append would send replay through appends from before this start,
    /// over a file a purge since may have laid out differently
    pub fn attach_wal(&self, wal: &Arc<zyron_wal::WalWriter>) {
        if self.log.set(Arc::clone(wal)).is_ok() {
            let mut inner = self.inner.lock();
            inner.logged_through = inner.logged_through.max(wal.next_lsn().0);
        }
    }

    /// The position in the log of the oldest logged append the segment
    /// files have not been synced through, None when everything logged is
    /// on disk. What the log has to keep until the next sync pass
    pub fn logged_from(&self) -> Option<zyron_wal::Lsn> {
        self.inner.lock().logged_from
    }

    fn table_dir(data_dir: &Path, table_id: u32) -> PathBuf {
        data_dir.join("cdf").join(format!("{table_id:08}"))
    }

    /// Where a branch's feed on a table lives, beneath the table's own
    pub fn branch_dir(data_dir: &Path, table_id: u32, branch_id: u64) -> PathBuf {
        Self::table_dir(data_dir, table_id).join(format!("branch_{branch_id}"))
    }

    /// The branches whose feeds on a table stand on disk, by branch id. A
    /// directory that is not a branch feed's is passed over
    pub fn table_branch_dirs(data_dir: &Path, table_id: u32) -> Vec<u64> {
        let Ok(entries) = fs::read_dir(Self::table_dir(data_dir, table_id)) else {
            return Vec::new();
        };
        entries
            .flatten()
            .filter_map(|entry| {
                let name = entry.file_name();
                let name = name.to_str()?;
                let id = name.strip_prefix("branch_")?.parse::<u64>().ok()?;
                entry.path().is_dir().then_some(id)
            })
            .collect()
    }

    /// The version of the table's own feed this feed's branch was taken at,
    /// zero for the table's own feed
    pub fn branch_point(&self) -> u64 {
        self.inner.lock().config.branch_point
    }

    fn manifest_path(dir: &Path) -> PathBuf {
        dir.join("manifest.zycdm")
    }

    fn segment_path(&self, seq: u64) -> PathBuf {
        self.dir.join(format!("{seq:012}.zycdf"))
    }

    fn segment_path_in(dir: &Path, seq: u64) -> PathBuf {
        dir.join(format!("{seq:012}.zycdf"))
    }

    // -----------------------------------------------------------------------
    // Manifest
    // -----------------------------------------------------------------------

    /// The manifest is a chain of records, the first holding the feed whole
    /// and each one after it what changed since the record before it, so a
    /// seal appends the segment it sealed and the versions it counted
    /// rather than writing the whole history again. The record count and
    /// the version index are written as they stood at the mark, the last
    /// durable point, and a reopen counts what the files hold past it.
    ///
    /// A whole record, everything after its kind byte and generation
    fn encode_manifest_whole(inner: &CdfInner, body: &mut Vec<u8>) {
        let mark = inner.synced_mark;
        let counted = Self::versions_at_mark(inner);
        body.reserve(256 + inner.sealed.len() * 64 + counted * 28);
        Self::write_manifest_counters(inner, body);
        body.extend_from_slice(&(inner.sealed.len() as u32).to_le_bytes());
        for summary in &inner.sealed {
            summary.write_into(body);
        }
        Self::write_version_entries(inner, body, 0, counted, mark.last_records);
        body.extend_from_slice(&(inner.spans.len() as u32).to_le_bytes());
        for span in &inner.spans {
            Self::write_span(span, body);
        }
        Self::write_layouts(inner, body);
    }

    /// A delta record, everything after its kind byte and generation. It
    /// carries the counters and the mark as they stand, the sealed
    /// segments added or replaced since the last record, the version
    /// entries past the ones it reached with the last of those written
    /// again since its count may have grown, the spans opened or extended
    /// since, and every layout
    fn encode_manifest_delta(inner: &CdfInner, body: &mut Vec<u8>) {
        let mark = inner.synced_mark;
        let counted = Self::versions_at_mark(inner);
        let from = inner.chain.versions_encoded.min(counted).saturating_sub(1);
        let mut sealed_dirty = inner.chain.sealed_dirty.clone();
        sealed_dirty.sort_unstable();
        sealed_dirty.dedup();
        let mut spans_dirty = inner.chain.spans_dirty.clone();
        spans_dirty.sort_unstable();
        spans_dirty.dedup();
        body.reserve(
            256 + sealed_dirty.len() * 64 + (counted - from) * 28 + spans_dirty.len() * 24,
        );
        Self::write_manifest_counters(inner, body);
        let changed: Vec<&SegmentSummary> = sealed_dirty
            .iter()
            .filter_map(|seq| inner.sealed.iter().find(|s| s.seq == *seq))
            .collect();
        body.extend_from_slice(&(changed.len() as u32).to_le_bytes());
        for summary in changed {
            summary.write_into(body);
        }
        Self::write_version_entries(inner, body, from, counted, mark.last_records);
        let touched: Vec<&TxnSpan> = spans_dirty
            .iter()
            .filter_map(|at| inner.spans.get(*at))
            .collect();
        body.extend_from_slice(&(touched.len() as u32).to_le_bytes());
        for span in touched {
            Self::write_span(span, body);
        }
        Self::write_layouts(inner, body);
    }

    /// Entries of the version index at or below the mark's newest version
    fn versions_at_mark(inner: &CdfInner) -> usize {
        let mark = inner.synced_mark;
        if mark.records_written == 0 {
            0
        } else {
            inner
                .versions
                .partition_point(|entry| entry.version <= mark.last_version)
        }
    }

    /// The configuration, the counters as of the mark and the mark itself,
    /// which every record carries
    fn write_manifest_counters(inner: &CdfInner, body: &mut Vec<u8>) {
        let mark = inner.synced_mark;
        inner.config.write_into(body);
        body.extend_from_slice(&mark.records_written.to_le_bytes());
        body.extend_from_slice(&inner.records_purged.to_le_bytes());
        body.extend_from_slice(&inner.purge_floor.to_le_bytes());
        body.extend_from_slice(&inner.next_seq.to_le_bytes());
        body.extend_from_slice(&inner.open.seq.to_le_bytes());
        // Where the counts reach, the segment and how many of its records
        // they cover
        body.extend_from_slice(&mark.seq.to_le_bytes());
        body.extend_from_slice(&mark.segment_records.to_le_bytes());
        // Where in the log replay starts after a crash. Every logged append
        // below it is in a synced file, so with nothing pending the record
        // says the log holds nothing this feed needs
        body.extend_from_slice(&inner.replay_from_now().to_le_bytes());
    }

    /// The version entries in `from..counted`, the newest at the mark cut
    /// to the records that had landed by then, since the ones after it are
    /// counted at reopen
    fn write_version_entries(
        inner: &CdfInner,
        body: &mut Vec<u8>,
        from: usize,
        counted: usize,
        last_records: u64,
    ) {
        body.extend_from_slice(&((counted - from) as u32).to_le_bytes());
        for (at, entry) in inner.versions[from..counted].iter().enumerate() {
            let records = if from + at + 1 == counted {
                entry.records.min(last_records)
            } else {
                entry.records
            };
            body.extend_from_slice(&entry.version.to_le_bytes());
            body.extend_from_slice(&records.to_le_bytes());
            body.extend_from_slice(&entry.prior.to_le_bytes());
            body.extend_from_slice(&entry.first_timestamp.to_le_bytes());
        }
    }

    fn write_span(span: &TxnSpan, body: &mut Vec<u8>) {
        body.extend_from_slice(&span.txn_id.to_le_bytes());
        body.extend_from_slice(&span.first.to_le_bytes());
        body.extend_from_slice(&span.last.to_le_bytes());
    }

    fn write_layouts(inner: &CdfInner, body: &mut Vec<u8>) {
        body.extend_from_slice(&(inner.layouts.len() as u32).to_le_bytes());
        for ((epoch, projected), types) in &inner.layouts {
            body.extend_from_slice(&epoch.to_le_bytes());
            body.push(u8::from(*projected));
            body.extend_from_slice(&(types.len() as u16).to_le_bytes());
            for physical in types {
                body.push(*physical as u8);
            }
        }
    }

    /// One record of the manifest file, an envelope whose header extension
    /// carries the body length, so the records are told apart on replay
    fn frame_manifest(body: &[u8]) -> Vec<u8> {
        let length = (body.len() as u32).to_le_bytes();
        envelope::encode_with(
            FormatKind::ChangeFeedManifest,
            CHANGE_FEED_MANIFEST_FORMAT_VERSION,
            0,
            &length,
            body,
        )
    }

    /// Replays the manifest's records into the feed's state, answering
    /// with None when there is no manifest.
    ///
    /// The first record has to be a whole one. A record after it that does
    /// not decode is the tail an interrupted append left and is cut off,
    /// and a record whose generation does not follow the one before it is
    /// where a record never reached the file, so the replay stops there
    /// and the file is cut to what it applied. What the records reach is
    /// the mark, and the caller counts the rest from the segment files
    fn read_manifest(dir: &Path) -> Result<Option<CdfInner>> {
        let path = Self::manifest_path(dir);
        if !path.exists() {
            return Ok(None);
        }
        let mut bytes = Vec::new();
        File::open(&path)?.read_to_end(&mut bytes)?;
        if bytes.is_empty() {
            return Ok(None);
        }
        let mut inner: Option<CdfInner> = None;
        let mut generation = 0u64;
        let mut deltas = 0u32;
        let mut offset = 0usize;
        let mut good_len = 0usize;
        while offset < bytes.len() {
            let Some((body, used)) =
                Self::next_manifest_record(&path, &bytes[offset..], inner.is_none())?
            else {
                break;
            };
            let mut cursor = ByteCursor::new(body);
            let kind = cursor.u8()?;
            let record_generation = cursor.u64()?;
            match (kind, inner.as_mut()) {
                (MANIFEST_WHOLE, _) => {
                    inner = Some(Self::decode_manifest_whole(&path, &mut cursor)?);
                    deltas = 0;
                }
                (MANIFEST_DELTA, Some(inner)) => {
                    if record_generation != generation.saturating_add(1) {
                        break;
                    }
                    Self::apply_manifest_delta(&path, inner, &mut cursor)?;
                    deltas += 1;
                }
                (MANIFEST_DELTA, None) => {
                    return Err(ZyronError::CdcDecoderError(format!(
                        "change feed manifest {} begins with a delta record rather than a \
                         whole one",
                        path.display()
                    )));
                }
                (other, _) => {
                    return Err(ZyronError::CdcDecoderError(format!(
                        "change feed manifest {} holds a record of kind {other}, which this \
                         binary does not read",
                        path.display()
                    )));
                }
            }
            generation = record_generation;
            offset += used;
            good_len = offset;
        }
        let Some(mut inner) = inner else {
            return Ok(None);
        };
        if good_len < bytes.len() {
            OpenOptions::new()
                .write(true)
                .open(&path)?
                .set_len(good_len as u64)?;
        }
        inner.synced_mark.last_version = inner.versions.last().map(|v| v.version).unwrap_or(0);
        inner.synced_mark.last_records = inner.versions.last().map(|v| v.records).unwrap_or(0);
        inner.span_index = inner
            .spans
            .iter()
            .enumerate()
            .map(|(at, span)| (span.txn_id, at))
            .collect();
        inner.manifest_gen = std::sync::atomic::AtomicU64::new(generation);
        inner.manifest_gate = Arc::new(ManifestGate::new(generation));
        inner.chain = ManifestChain {
            versions_encoded: inner.versions.len(),
            deltas_since_whole: deltas,
            ..ManifestChain::default()
        };
        Ok(Some(inner))
    }

    /// The next record of the manifest file, its body and the bytes it
    /// spans. None for a trailing record that does not decode, which is
    /// one whose append was interrupted. The first record has to decode
    fn next_manifest_record<'a>(
        path: &Path,
        bytes: &'a [u8],
        first: bool,
    ) -> Result<Option<(&'a [u8], usize)>> {
        let refuse = |detail: String| {
            ZyronError::CdcDecoderError(format!(
                "change feed manifest {} did not decode, {detail}",
                path.display()
            ))
        };
        let header = match envelope::decode_header(bytes) {
            Ok((header, extension)) => {
                if header.kind != FormatKind::ChangeFeedManifest {
                    return Err(refuse(format!("a {} record is inside it", header.kind)));
                }
                if header.version != CHANGE_FEED_MANIFEST_FORMAT_VERSION {
                    return Err(ZyronError::CdcDecoderError(format!(
                        "change feed manifest is at format version {}, this binary writes and \
                         reads {}. Upgrade through a release that still reads {} to move it \
                         forward first",
                        header.version, CHANGE_FEED_MANIFEST_FORMAT_VERSION, header.version
                    )));
                }
                if extension.len() != MANIFEST_LENGTH_EXTENSION {
                    return Err(refuse(format!(
                        "a record carries a {} byte header extension rather than the body \
                         length",
                        extension.len()
                    )));
                }
                let body_len =
                    u32::from_le_bytes([extension[0], extension[1], extension[2], extension[3]])
                        as usize;
                Some((header, body_len))
            }
            Err(e) if first => return Err(refuse(e.to_string())),
            Err(_) => None,
        };
        let Some((header, body_len)) = header else {
            return Ok(None);
        };
        let total = header.body_offset() + body_len + envelope::ENVELOPE_FOOTER_LEN;
        if total > bytes.len() {
            if first {
                return Err(refuse(format!(
                    "the first record declares {total} bytes and the file holds {}",
                    bytes.len()
                )));
            }
            return Ok(None);
        }
        match envelope::decode_as(&bytes[..total], FormatKind::ChangeFeedManifest) {
            Ok(parsed) => Ok(Some((parsed.body, total))),
            Err(e) if first => Err(refuse(e.to_string())),
            Err(_) => Ok(None),
        }
    }

    /// The counters every record carries, config first
    fn read_manifest_counters(cursor: &mut ByteCursor<'_>) -> Result<ManifestCounters> {
        let config = FeedConfig::read_from(cursor)?;
        let records_written = cursor.u64()?;
        let records_purged = cursor.u64()?;
        let purge_floor = cursor.u64()?;
        let next_seq = cursor.u64()?;
        let open_seq = cursor.u64()?;
        let mark_seq = cursor.u64()?;
        let mark_records = cursor.u64()?;
        let replay_from = cursor.u64()?;
        let mark = SyncMark {
            seq: mark_seq,
            records_written,
            last_version: 0,
            last_records: 0,
            segment_records: mark_records,
        };
        Ok(ManifestCounters {
            config,
            records_written,
            records_purged,
            purge_floor,
            next_seq,
            open_seq,
            mark,
            replay_from,
        })
    }

    fn read_version_entries(cursor: &mut ByteCursor<'_>) -> Result<Vec<VersionCount>> {
        let count = cursor.u32()? as usize;
        let mut versions = Vec::with_capacity(count);
        for _ in 0..count {
            versions.push(VersionCount {
                version: cursor.u64()?,
                records: cursor.u64()?,
                prior: cursor.u64()?,
                first_timestamp: cursor.i64()?,
            });
        }
        Ok(versions)
    }

    fn read_spans(cursor: &mut ByteCursor<'_>) -> Result<Vec<TxnSpan>> {
        let count = cursor.u32()? as usize;
        let mut spans = Vec::with_capacity(count);
        for _ in 0..count {
            spans.push(TxnSpan {
                txn_id: cursor.u64()?,
                first: cursor.u64()?,
                last: cursor.u64()?,
            });
        }
        Ok(spans)
    }

    fn read_layouts(cursor: &mut ByteCursor<'_>) -> Result<crate::segment_columns::RowLayouts> {
        let layout_count = cursor.u32()? as usize;
        let mut layouts = crate::segment_columns::RowLayouts::new();
        for _ in 0..layout_count {
            let epoch = cursor.u32()?;
            let projected = cursor.u8()? != 0;
            let columns = cursor.u16()? as usize;
            let mut types = Vec::with_capacity(columns);
            for _ in 0..columns {
                let code = cursor.u8()?;
                types.push(zyron_common::TypeId::from_u8(code).ok_or_else(|| {
                    ZyronError::CdcDecoderError(format!(
                        "change feed manifest names an unknown physical type {code}"
                    ))
                })?);
            }
            layouts.insert((epoch, projected), types);
        }
        Ok(layouts)
    }

    /// Decodes a whole record, everything after its kind byte and
    /// generation, into the feed's state as of the mark it records
    fn decode_manifest_whole(path: &Path, cursor: &mut ByteCursor<'_>) -> Result<CdfInner> {
        let ManifestCounters {
            config,
            records_written,
            records_purged,
            purge_floor,
            next_seq,
            open_seq,
            mark,
            replay_from,
        } = Self::read_manifest_counters(cursor)?;
        let sealed_count = cursor.u32()? as usize;
        let mut sealed = Vec::with_capacity(sealed_count);
        for _ in 0..sealed_count {
            sealed.push(SegmentSummary::read_from(cursor)?);
        }
        let versions = Self::read_version_entries(cursor)?;
        let spans = Self::read_spans(cursor)?;
        let layouts = Self::read_layouts(cursor)?;
        if !cursor.is_empty() {
            return Err(ZyronError::CdcDecoderError(format!(
                "change feed manifest {} holds a whole record with bytes past its end",
                path.display()
            )));
        }
        let codec = config.codec;
        Ok(CdfInner {
            layouts,
            manifest_gen: std::sync::atomic::AtomicU64::new(0),
            manifest_gate: Arc::new(ManifestGate::new(0)),
            chain: ManifestChain::default(),
            config,
            sealed,
            open: SegmentSummary::empty(open_seq, codec),
            writer: None,
            torn_tail: false,
            synced_bytes: SEGMENT_BODY_OFFSET,
            logged_from: None,
            logged_through: 0,
            replay_from,
            synced_mark: mark,
            closed_unsynced: Vec::new(),
            versions,
            records_written,
            records_purged,
            purge_floor,
            next_seq,
            ordinal_version: u64::MAX,
            next_ordinal: 0,
            rewrite_epoch: 0,
            // A transaction that was open when the process stopped ended
            // with it. Its records are still here and a read judges each
            // by the transaction's recorded outcome
            in_flight: HashMap::new(),
            open_frames: Vec::new(),
            open_version_starts: Vec::new(),
            span_scratch: Vec::new(),
            closed: false,
            cap_purges: 0,
            spans,
            span_index: HashMap::new(),
        })
    }

    /// Applies a delta record, everything after its kind byte and
    /// generation, to the state the records before it built
    fn apply_manifest_delta(
        path: &Path,
        inner: &mut CdfInner,
        cursor: &mut ByteCursor<'_>,
    ) -> Result<()> {
        let ManifestCounters {
            config,
            records_written,
            records_purged,
            purge_floor,
            next_seq,
            open_seq,
            mark,
            replay_from,
        } = Self::read_manifest_counters(cursor)?;
        let changed_count = cursor.u32()? as usize;
        let mut changed = Vec::with_capacity(changed_count);
        for _ in 0..changed_count {
            changed.push(SegmentSummary::read_from(cursor)?);
        }
        let versions = Self::read_version_entries(cursor)?;
        let spans = Self::read_spans(cursor)?;
        let layouts = Self::read_layouts(cursor)?;
        if !cursor.is_empty() {
            return Err(ZyronError::CdcDecoderError(format!(
                "change feed manifest {} holds a delta record with bytes past its end",
                path.display()
            )));
        }
        let codec = config.codec;
        inner.config = config;
        inner.records_written = records_written;
        inner.records_purged = records_purged;
        inner.purge_floor = purge_floor;
        inner.next_seq = next_seq;
        inner.open = SegmentSummary::empty(open_seq, codec);
        inner.synced_mark = mark;
        inner.replay_from = replay_from;
        for summary in changed {
            match inner.sealed.binary_search_by_key(&summary.seq, |s| s.seq) {
                Ok(at) => inner.sealed[at] = summary,
                Err(at) => inner.sealed.insert(at, summary),
            }
        }
        for entry in versions {
            match inner.versions.last_mut() {
                Some(last) if last.version == entry.version => *last = entry,
                Some(last) if entry.version > last.version => inner.versions.push(entry),
                None => inner.versions.push(entry),
                Some(last) => {
                    return Err(ZyronError::CdcDecoderError(format!(
                        "change feed manifest {} holds a delta record whose version {} is \
                         below the version {} already counted",
                        path.display(),
                        entry.version,
                        last.version
                    )));
                }
            }
        }
        for span in spans {
            match inner.spans.iter_mut().find(|s| s.txn_id == span.txn_id) {
                Some(held) => *held = span,
                None => inner.spans.push(span),
            }
        }
        inner.layouts = layouts;
        Ok(())
    }

    /// Writes the manifest's next record, a delta of what changed since
    /// the last one or the feed whole when the chain calls for it
    fn persist_manifest(dir: &Path, inner: &mut CdfInner) -> Result<()> {
        let pending = Self::encode_manifest_pending(inner);
        Self::write_manifest(dir, pending)
    }

    /// Writes the feed whole as the manifest's next record, for a change
    /// that pruned or rebuilt what the records so far describe
    fn persist_manifest_whole(dir: &Path, inner: &mut CdfInner) -> Result<()> {
        inner.chain.whole_next = true;
        Self::persist_manifest(dir, inner)
    }

    /// The manifest's next record as the feed stands, numbered, with the
    /// gate its write goes through. Encoded under the feed's lock, written
    /// with or without it, so a sealer makes its record durable without
    /// holding writers. A whole record is encoded when the chain calls for
    /// one, when the deltas since the last whole one pass the threshold a
    /// replay walks, and after a record was given up on unwritten
    fn encode_manifest_pending(inner: &mut CdfInner) -> PendingManifest {
        let generation = inner
            .manifest_gen
            .fetch_add(1, std::sync::atomic::Ordering::AcqRel)
            + 1;
        let whole = inner.chain.whole_next
            || inner.chain.deltas_since_whole >= MANIFEST_COMPACT_AFTER
            || inner.manifest_gate.take_gap();
        let mut body = Vec::new();
        body.push(if whole {
            MANIFEST_WHOLE
        } else {
            MANIFEST_DELTA
        });
        body.extend_from_slice(&generation.to_le_bytes());
        if whole {
            Self::encode_manifest_whole(inner, &mut body);
        } else {
            Self::encode_manifest_delta(inner, &mut body);
        }
        inner.chain.versions_encoded = Self::versions_at_mark(inner);
        inner.chain.sealed_dirty.clear();
        inner.chain.spans_dirty.clear();
        if whole {
            inner.chain.deltas_since_whole = 0;
            inner.chain.whole_next = false;
        } else {
            inner.chain.deltas_since_whole += 1;
        }
        PendingManifest {
            generation,
            bytes: Self::frame_manifest(&body),
            whole,
            gate: Arc::clone(&inner.manifest_gate),
            done: false,
        }
    }

    /// Writes one record in its turn. A whole record lays the file down
    /// through a temp file and an atomic rename, so a crash at any instant
    /// leaves either the old chain or the new whole, and a delta is
    /// appended and made durable, so a crash leaves at most a torn tail the
    /// next open cuts off
    fn write_manifest(dir: &Path, mut pending: PendingManifest) -> Result<()> {
        pending.gate.wait_turn(pending.generation);
        let path = Self::manifest_path(dir);
        if pending.whole {
            let tmp = path.with_extension("zycdm.tmp");
            {
                let mut file = File::create(&tmp)?;
                file.write_all(&pending.bytes)?;
                file.sync_all()?;
            }
            fs::rename(&tmp, &path)?;
            sync_parent_dir(&path)?;
        } else {
            let mut file = OpenOptions::new().append(true).open(&path)?;
            file.write_all(&pending.bytes)?;
            file.sync_data()?;
        }
        pending.done = true;
        pending.gate.finish(pending.generation, false);
        Ok(())
    }

    // -----------------------------------------------------------------------
    // Recovery
    // -----------------------------------------------------------------------

    /// Recovers the open segment and every segment a rotation opened after
    /// it before the manifest caught up.
    ///
    /// A rotation closes the open segment and starts the next one before
    /// the sealer rewrites the manifest, so a stop in between leaves the
    /// manifest naming a segment as open that is full, with the one after
    /// it holding frames. The full one is kept as a sealed entry in its raw
    /// form and the walk moves on. A segment past the open one that holds
    /// nothing is the one a sealer prepared for the next rotation.
    ///
    /// The manifest's counts reach its mark. The frames past the mark, in
    /// the segment it names and in every segment after it, are counted here
    /// from the files, closed segments first
    fn recover_open_chain(dir: &Path, table_id: u32, inner: &mut CdfInner) -> Result<()> {
        Self::recount_closed_from_mark(dir, table_id, inner)?;
        loop {
            let counted = if inner.synced_mark.seq == inner.open.seq {
                inner.synced_mark.segment_records
            } else {
                0
            };
            Self::recover_open_segment(dir, table_id, inner, counted)?;
            let next = Self::segment_path_in(dir, inner.open.seq + 1);
            let holds_frames = fs::metadata(&next)
                .map(|meta| meta.len() > SEGMENT_BODY_OFFSET)
                .unwrap_or(false);
            if !holds_frames {
                return Ok(());
            }
            let mut done = inner.open.clone();
            done.codec = inner.config.codec;
            done.sealed = false;
            inner.sealed.push(done);
            inner.open = SegmentSummary::empty(inner.open.seq + 1, inner.config.codec);
            inner.next_seq = inner.open.seq + 1;
            inner.reset_open_frames(Vec::new());
        }
    }

    /// Seals every segment the manifest lists as sealed but whose file is
    /// still in the raw form, which a stop between a rotation and its
    /// sealer leaves, then records the result
    fn seal_raw_segments(dir: &Path, table_id: u32, inner: &mut CdfInner) -> Result<()> {
        let raw: Vec<u64> = inner
            .sealed
            .iter()
            .filter(|s| !s.sealed)
            .map(|s| s.seq)
            .collect();
        if raw.is_empty() {
            return Ok(());
        }
        for seq in raw {
            let path = Self::segment_path_in(dir, seq);
            let mut bytes = Vec::new();
            File::open(&path)?.read_to_end(&mut bytes)?;
            if bytes.len() < SEGMENT_BODY_OFFSET as usize {
                continue;
            }
            let header = read_segment_header(&bytes)?;
            let Some(summary) = inner.sealed.iter().find(|s| s.seq == seq).cloned() else {
                continue;
            };
            if header.sealed {
                // The sealer's rename landed and only the manifest did not
                let mut sealed = summary;
                sealed.sealed = true;
                sealed.bytes = bytes.len() as u64;
                if let Some(slot) = inner.sealed.iter_mut().find(|s| s.seq == seq) {
                    *slot = sealed;
                }
                continue;
            }
            // A raw closed segment a stop left unsynced may end in a torn
            // frame. The summary is rebuilt from the frames that check out,
            // which is what the sealed file holds, rather than taken from
            // the manifest, which was written as the frames landed
            let body = &bytes[SEGMENT_BODY_OFFSET as usize..];
            let mut rebuilt = SegmentSummary::empty(seq, summary.codec);
            let (valid_len, _) = walk_frames(body, |record| {
                let (version, timestamp) = ChangeRecord::peek_version_timestamp(record)?;
                rebuilt.observe(
                    version,
                    timestamp,
                    ChangeRecord::peek_change_type(record)?,
                    ChangeRecord::peek_schema_version(record)?,
                );
                Ok(())
            })?;
            rebuilt.bytes = SEGMENT_BODY_OFFSET + valid_len as u64;
            let (sealed, tmp) =
                seal_segment_files(dir, table_id, &rebuilt, &body[..valid_len], &inner.layouts)?;
            fs::rename(&tmp, &path)?;
            sync_parent_dir(&path)?;
            if let Some(slot) = inner.sealed.iter_mut().find(|s| s.seq == seq) {
                *slot = sealed;
            }
        }
        Self::persist_manifest_whole(dir, inner)
    }

    /// Counts the frames of the closed segments the manifest's counts do
    /// not reach, the marked segment from its first uncounted record on and
    /// every closed segment after it whole. A segment is walked as it
    /// stands on disk, raw or sealed, and a raw one is counted only as far
    /// as its frames check out, since a torn tail is what the sealer
    /// truncates
    fn recount_closed_from_mark(dir: &Path, table_id: u32, inner: &mut CdfInner) -> Result<()> {
        let mark = inner.synced_mark;
        if mark.seq >= inner.open.seq {
            return Ok(());
        }
        let closed: Vec<u64> = inner
            .sealed
            .iter()
            .map(|s| s.seq)
            .filter(|seq| *seq >= mark.seq && *seq < inner.open.seq)
            .collect();
        for seq in closed {
            let skip = if seq == mark.seq {
                mark.segment_records
            } else {
                0
            };
            let Some(frames) = segment_frames_in(dir, table_id, seq, None)? else {
                continue;
            };
            Self::count_frames(inner, &frames, skip)?;
        }
        Ok(())
    }

    /// Counts every frame past the first `skip` into the version index and
    /// the transaction spans, answering with the length the frames check out
    /// to
    fn count_frames(inner: &mut CdfInner, frames: &[u8], skip: u64) -> Result<usize> {
        let mut seen = 0u64;
        let (valid_len, _) = walk_frames(frames, |record| {
            seen += 1;
            if seen <= skip {
                return Ok(());
            }
            let (version, timestamp) = ChangeRecord::peek_version_timestamp(record)?;
            inner.observe_version(version, timestamp);
            inner.observe_span(ChangeRecord::peek_txn_id(record)?, version, version);
            Ok(())
        })?;
        Ok(valid_len)
    }

    /// Rebuilds the open segment's summary by walking it, counts the frames
    /// past the first `counted`, which the manifest already holds, and
    /// truncates at the last frame that checks out
    fn recover_open_segment(
        dir: &Path,
        table_id: u32,
        inner: &mut CdfInner,
        counted: u64,
    ) -> Result<()> {
        let path = Self::segment_path_in(dir, inner.open.seq);
        if !path.exists() {
            let mut file = File::create(&path)?;
            write_segment_header(&mut file, table_id, inner.open.seq, 0)?;
            file.sync_all()?;
            inner.open.bytes = SEGMENT_BODY_OFFSET;
            inner.synced_bytes = SEGMENT_BODY_OFFSET;
            inner.reset_open_frames(Vec::new());
            return Ok(());
        }

        let mut bytes = Vec::new();
        File::open(&path)?.read_to_end(&mut bytes)?;
        if bytes.len() < SEGMENT_BODY_OFFSET as usize {
            let mut file = File::create(&path)?;
            write_segment_header(&mut file, table_id, inner.open.seq, 0)?;
            file.sync_all()?;
            inner.open.bytes = SEGMENT_BODY_OFFSET;
            inner.synced_bytes = SEGMENT_BODY_OFFSET;
            inner.reset_open_frames(Vec::new());
            return Ok(());
        }
        let header = read_segment_header(&bytes)?;
        if header.table_id != table_id {
            return Err(ZyronError::CdcDecoderError(format!(
                "change segment {} carries table id {}, expected {table_id}",
                path.display(),
                header.table_id
            )));
        }
        if header.sealed {
            // The manifest names a sealed file as open, which means a seal
            // committed the file and the manifest write that follows it did
            // not land. Adopting the file and opening the next one is the
            // state the seal was reaching for. The records the manifest's
            // counts stop short of are counted from the sealed frames first
            let summary = Self::read_trailer(&bytes)?;
            if let Some(frames) = segment_frames_in(dir, table_id, inner.open.seq, None)? {
                Self::count_frames(inner, &frames, counted)?;
            }
            let already = inner.sealed.iter().any(|s| s.seq == summary.seq);
            if !already {
                inner.sealed.push(summary);
                inner.sealed.sort_by_key(|s| s.seq);
            }
            inner.open = SegmentSummary::empty(inner.next_seq, inner.config.codec);
            inner.next_seq += 1;
            Self::persist_manifest_whole(dir, inner)?;
            return Self::recover_open_segment(dir, table_id, inner, 0);
        }

        let body = &bytes[SEGMENT_BODY_OFFSET as usize..];
        let mut summary = SegmentSummary::empty(inner.open.seq, inner.config.codec);
        let mut seen = 0u64;
        let (valid_len, _) = walk_frames(body, |record| {
            let (version, timestamp) = ChangeRecord::peek_version_timestamp(record)?;
            let change_type = ChangeRecord::peek_change_type(record)?;
            summary.observe(
                version,
                timestamp,
                change_type,
                ChangeRecord::peek_schema_version(record)?,
            );
            seen += 1;
            // The frames the manifest counted keep their place in the index.
            // The ones past them landed after its counts were taken and are
            // counted here
            if seen > counted {
                inner.observe_version(version, timestamp);
                inner.observe_span(ChangeRecord::peek_txn_id(record)?, version, version);
            }
            Ok(())
        })?;

        summary.bytes = SEGMENT_BODY_OFFSET + valid_len as u64;
        inner.open = summary;
        inner.synced_bytes = inner.open.bytes;
        inner.reset_open_frames(body[..valid_len].to_vec());

        // What the walk found is what the segment holds from here on, made
        // durable so the mark taken after recovery names a durable point
        let file = OpenOptions::new().write(true).open(&path)?;
        if (valid_len as u64) < body.len() as u64 {
            file.set_len(inner.open.bytes)?;
        }
        file.sync_all()?;
        Ok(())
    }

    // -----------------------------------------------------------------------
    // Configuration
    // -----------------------------------------------------------------------

    pub fn is_enabled(&self) -> bool {
        self.enabled.load(Ordering::Acquire)
    }

    pub fn enable(&self) {
        self.enabled.store(true, Ordering::Release);
        let mut inner = self.inner.lock();
        inner.config.enabled = true;
    }

    pub fn disable(&self) {
        self.enabled.store(false, Ordering::Release);
        let mut inner = self.inner.lock();
        inner.config.enabled = false;
    }

    /// The feed's configuration as it stands
    pub fn config(&self) -> FeedConfig {
        self.inner.lock().config.clone()
    }

    /// Records the layout rows written under `epoch` hold, for a writer
    /// that appends records rather than rows. A seal slices a segment into
    /// columns only when every row's layout is known
    pub fn record_layout(
        &self,
        epoch: u32,
        projected: bool,
        columns: &[zyron_catalog::PhysicalColumn],
    ) -> Result<()> {
        let mut inner = self.inner.lock();
        let key = crate::segment_columns::layout_key(epoch, projected);
        if inner.layouts.contains_key(&key) {
            return Ok(());
        }
        inner
            .layouts
            .insert(key, crate::segment_columns::layout_types(columns));
        self.sync_open_locked(&mut inner)?;
        Self::persist_manifest(&self.dir, &mut inner)
    }

    /// Replaces the configuration and makes it durable.
    ///
    /// A codec change applies to segments sealed from here on. Segments
    /// already sealed keep the codec they were written with, which each one
    /// records, so nothing has to be rewritten to change the setting
    pub fn set_config(&self, config: FeedConfig) -> Result<()> {
        let mut inner = self.inner.lock();
        self.enabled.store(config.enabled, Ordering::Release);
        inner.config = config;
        self.sync_open_locked(&mut inner)?;
        Self::persist_manifest(&self.dir, &mut inner)
    }

    /// Retention as whole days, for the callers that report in days
    pub fn retention_days(&self) -> u32 {
        self.inner.lock().config.retention_days()
    }

    // -----------------------------------------------------------------------
    // Append
    // -----------------------------------------------------------------------

    /// Appends a single change record
    pub fn append_change(&self, record: &ChangeRecord) -> Result<()> {
        self.append_batch(std::slice::from_ref(record))
    }

    /// Appends many change records under one lock acquisition and one write.
    ///
    /// The whole batch is framed onto the open segment's bytes, so the
    /// serialized cost per row is the encoding and nothing else, and the
    /// lock is held for one `write_all`
    pub fn append_batch(&self, records: &[ChangeRecord]) -> Result<()> {
        if !self.is_enabled() || records.is_empty() {
            return Ok(());
        }
        let exact: usize = records
            .iter()
            .map(|r| RECORD_FRAME_PREFIX + r.serialized_len() + RECORD_FRAME_SUFFIX)
            .sum();

        let mut inner = self.inner.lock();
        let mark = inner.mark();
        let base = inner.open_frames.len();
        let starts_base = inner.open_version_starts.len();
        inner.open_frames.reserve(exact);
        let mut floor = inner.floor();
        // Each run of one transaction's records with the versions it
        // landed at, recorded once the bytes are in the file. The buffer
        // is the feed's own, so a batch of any size allocates nothing here
        let mut spans = std::mem::take(&mut inner.span_scratch);
        spans.clear();
        for record in records {
            let header = ChangeHeader {
                commit_version: record.commit_version,
                commit_timestamp: record.commit_timestamp,
                txn_id: record.txn_id,
                schema_version: record.schema_version,
            };
            let change = RowChange {
                change_type: record.change_type,
                row_data: &record.row_data,
                primary_key_data: &record.primary_key_data,
                is_last_in_txn: record.is_last_in_txn,
                projected: record.projected,
            };
            if let Some(version) = inner.stage(&header, &change, &mut floor) {
                match spans.last_mut() {
                    Some((txn_id, first, last)) if *txn_id == record.txn_id => {
                        *first = (*first).min(version);
                        *last = (*last).max(version);
                    }
                    _ => spans.push((record.txn_id, version, version)),
                }
            }
        }
        // Logged under the first transaction in the batch. The record is
        // replayed whatever its transaction did, so which one it names
        // only says where in the log the bytes sit
        let logged_as = records[0].txn_id;
        let landed = self.land(&mut inner, mark, logged_as, base, starts_base);
        if landed.is_ok() {
            for &(txn_id, first, last) in spans.iter() {
                inner.observe_span(txn_id, first, last);
                inner.in_flight.entry(txn_id).or_insert(first);
            }
        }
        inner.span_scratch = spans;
        landed
    }

    /// Appends the changes of one write, sharing a header, with the rows
    /// borrowed from the writer.
    ///
    /// The transaction's span and its open-transaction entry are touched
    /// once for the write rather than once per row, since every change
    /// carries the same transaction
    pub fn append_rows(
        &self,
        header: &ChangeHeader,
        layout: RowLayout<'_>,
        changes: &[RowChange<'_>],
    ) -> Result<()> {
        if !self.is_enabled() || changes.is_empty() {
            return Ok(());
        }
        let exact: usize = changes.iter().map(|c| c.framed_len()).sum();

        let mut inner = self.inner.lock();
        // The first write under a layout records it, so the seal can slice
        // the rows into columns. Written to the manifest at once, because
        // a layout a restart forgot would leave every segment holding rows
        // under it framed rather than sliced
        if let Some(columns) = layout.columns {
            let key = crate::segment_columns::layout_key(header.schema_version, layout.projected);
            if !inner.layouts.contains_key(&key) {
                inner
                    .layouts
                    .insert(key, crate::segment_columns::layout_types(columns));
                self.sync_open_locked(&mut inner)?;
                Self::persist_manifest(&self.dir, &mut inner)?;
            }
        }
        let mark = inner.mark();
        let base = inner.open_frames.len();
        let starts_base = inner.open_version_starts.len();
        inner.open_frames.reserve(exact);
        let mut floor = inner.floor();
        let mut landed: Option<(u64, u64)> = None;
        for change in changes {
            if let Some(version) = inner.stage(header, change, &mut floor) {
                landed = Some(match landed {
                    Some((first, last)) => (first.min(version), last.max(version)),
                    None => (version, version),
                });
            }
        }
        self.land(&mut inner, mark, header.txn_id, base, starts_base)?;
        if let Some((first, last)) = landed {
            inner.observe_span(header.txn_id, first, last);
            inner.in_flight.entry(header.txn_id).or_insert(first);
        }
        Ok(())
    }

    /// Writes the frames staged past `base` to the open segment's file and
    /// publishes the counters, sealing the segment once it reaches its
    /// target size. The bytes are already where the segment keeps them, so
    /// a batch that never reaches the file is cut back off them and off
    /// the version starts recorded past `starts_base`
    fn land(
        &self,
        inner: &mut CdfInner,
        mark: StageMark,
        txn_id: u64,
        base: usize,
        starts_base: usize,
    ) -> Result<()> {
        let staged = inner.open_frames.len() - base;
        if staged == 0 {
            return Ok(());
        }
        let open_path = self.segment_path(inner.open.seq);
        if let Err(e) = Self::write_open(inner, &open_path, base) {
            // The counters go back to what the file holds, so the next
            // append numbers its records where this one would have and no
            // manifest counts records that never landed
            inner.unstage(mark, base, starts_base);
            return Err(e);
        }
        // The bytes are in the file, so the log records them, ahead of the
        // commit record that names their transaction. A record that could
        // not be written leaves the file cut back to what the feed counts,
        // the way a failed write does, so the log never lags the file
        if let Some(wal) = self.log.get() {
            let logged = wal.log_change_feed_frames(
                txn_id,
                self.table_id,
                self.branch_id,
                inner.open.seq,
                inner.open.bytes,
                &inner.open_frames[base..],
            );
            match logged {
                Ok(Some((first, last))) => {
                    inner.logged_from.get_or_insert(first);
                    inner.logged_through = inner.logged_through.max(last.0);
                }
                Ok(None) => {}
                Err(e) => {
                    inner.writer = None;
                    if let Err(cut) = Self::cut_open(&open_path, inner.open.bytes) {
                        inner.torn_tail = true;
                        tracing::warn!(
                            target: "zyron::cdc",
                            "a change segment at {} could not be cut back after its append \
                             could not be logged and is cut before the next one: {cut}",
                            open_path.display()
                        );
                    }
                    inner.unstage(mark, base, starts_base);
                    return Err(e);
                }
            }
        }
        inner.open.bytes += staged as u64;
        self.unsynced.store(true, Ordering::Release);

        let latest = inner.open.max_version;
        self.counters.publish(&inner);
        self.counters
            .latest_version
            .store(latest, Ordering::Release);

        if inner.open.bytes >= SEGMENT_TARGET_BYTES {
            self.rotate_locked(inner, true)?;
        }
        Ok(())
    }

    /// Appends the open segment's staged bytes, the ones past `base`, to
    /// its file. A write that fails leaves the file cut back to the bytes
    /// the feed counts, and when even the cut fails the tail is recorded
    /// as torn so the next append cuts it before writing rather than
    /// framing records after a partial one
    fn write_open(inner: &mut CdfInner, path: &Path, base: usize) -> Result<()> {
        Self::open_writer(inner, path)?;
        let wrote = {
            let CdfInner {
                writer,
                open_frames,
                ..
            } = &mut *inner;
            let writer = writer.as_mut().ok_or_else(|| {
                ZyronError::CdcStreamError("the change feed writer is not open".into())
            })?;
            writer.write_all(&open_frames[base..])
        };
        let Err(e) = wrote else {
            return Ok(());
        };
        // Closed before the cut, so the length the cut sets is the length
        // the next append writes from
        inner.writer = None;
        if let Err(cut) = Self::cut_open(path, inner.open.bytes) {
            inner.torn_tail = true;
            tracing::warn!(
                target: "zyron::cdc",
                "a change segment at {} could not be cut back after a failed append and is \
                 cut before the next one: {cut}",
                path.display()
            );
        }
        Err(e.into())
    }

    /// Cuts the open segment's file to `len` bytes
    fn cut_open(path: &Path, len: u64) -> std::io::Result<()> {
        OpenOptions::new().write(true).open(path)?.set_len(len)
    }

    fn open_writer(inner: &mut CdfInner, path: &Path) -> Result<()> {
        if inner.torn_tail {
            Self::cut_open(path, inner.open.bytes)?;
            inner.torn_tail = false;
        }
        if inner.writer.is_none() {
            inner.writer = Some(OpenOptions::new().create(true).append(true).open(path)?);
        }
        Ok(())
    }

    // -----------------------------------------------------------------------
    // Sealing
    // -----------------------------------------------------------------------

    /// Seals the open segment and opens the next one.
    ///
    /// Returns false when the open segment holds nothing, which is what makes
    /// this safe to call from a maintenance pass that does not know whether
    /// anything was written since the last one
    pub fn seal_open_segment(&self) -> Result<bool> {
        let mut inner = self.inner.lock();
        if inner.open.is_empty() {
            return Ok(false);
        }
        self.rotate_locked(&mut inner, false)?;
        Ok(true)
    }

    /// Closes the open segment and opens the next one, then seals the closed
    /// one on a thread of its own when `background` is set and here
    /// otherwise, which is what an explicit seal asks for
    fn rotate_locked(&self, inner: &mut CdfInner, background: bool) -> Result<()> {
        let job = self.close_open_locked(inner)?;
        if !background {
            return job.run_locked(inner);
        }
        let table_id = self.table_id;
        let spawned = std::thread::Builder::new()
            .name("zyron-cdf-seal".into())
            .spawn(move || {
                if let Err(e) = job.run() {
                    tracing::warn!(
                        target: "zyron::cdc",
                        "sealing a change segment of table {table_id} failed, the segment \
                         stays readable in its raw form: {e}"
                    );
                }
            });
        if let Err(e) = spawned {
            return Err(ZyronError::CdcStreamError(format!(
                "the change feed of table {table_id} could not start a sealer thread: {e}"
            )));
        }
        self.counters.publish(inner);
        Ok(())
    }

    /// The writer's part of a rotation, which is the cheap part. The closed
    /// segment's frames are already in memory and its file is complete on
    /// disk, so its summary moves into the sealed list in its raw form, the
    /// next file is adopted, and what is answered is the job that seals it
    fn close_open_locked(&self, inner: &mut CdfInner) -> Result<SealJob> {
        // A tail a failed append left is cut before the file closes, so the
        // closed file ends where its frames do
        if inner.torn_tail {
            Self::cut_open(&self.segment_path(inner.open.seq), inner.open.bytes)?;
            inner.torn_tail = false;
        }
        inner.writer = None;
        let frames = std::mem::take(&mut inner.open_frames);
        inner.open_version_starts.clear();
        let mut summary = inner.open.clone();
        summary.codec = inner.config.codec;
        summary.sealed = false;
        inner.sealed.push(summary.clone());
        inner.chain.sealed_dirty.push(summary.seq);
        // The counts through the end of the closed segment, which become
        // the mark once the seal has made the segment durable. Until then
        // the segment's raw file may end in bytes no sync reached
        let close_mark = inner.take_mark();
        if inner.synced_bytes < inner.open.bytes {
            inner.closed_unsynced.push(inner.open.seq);
        }

        let next_seq = inner.next_seq;
        Self::prepare_segment(&self.dir, self.table_id, next_seq)?;
        inner.open = SegmentSummary::empty(next_seq, inner.config.codec);
        inner.open.bytes = SEGMENT_BODY_OFFSET;
        inner.next_seq += 1;
        inner.synced_bytes = SEGMENT_BODY_OFFSET;

        Ok(SealJob {
            dir: self.dir.clone(),
            table_id: self.table_id,
            summary,
            frames,
            layouts: inner.layouts.clone(),
            epoch: inner.rewrite_epoch,
            close_mark,
            inner: Arc::clone(&self.inner),
            counters: Arc::clone(&self.counters),
        })
    }

    /// Creates a segment file holding its header alone, durably, unless it
    /// already stands. A sealer prepares the segment after the open one so
    /// a rotation adopts a file whose header is already on disk
    fn prepare_segment(dir: &Path, table_id: u32, seq: u64) -> Result<()> {
        let path = Self::segment_path_in(dir, seq);
        if path.exists() {
            return Ok(());
        }
        let mut file = File::create(&path)?;
        write_segment_header(&mut file, table_id, seq, 0)?;
        file.sync_all()?;
        Ok(())
    }

    /// The summary block a sealed segment ends with
    fn encode_trailer(summary: &SegmentSummary) -> Vec<u8> {
        let mut body = Vec::with_capacity(64);
        summary.write_into(&mut body);
        let checksum = hot_hash32(&body);
        let mut out = Vec::with_capacity(body.len() + TRAILER_SUFFIX_LEN);
        out.extend_from_slice(&body);
        out.extend_from_slice(&(body.len() as u32).to_le_bytes());
        out.extend_from_slice(&checksum.to_le_bytes());
        out.extend_from_slice(&TRAILER_MARKER);
        out
    }

    /// Reads a sealed segment's trailer from the whole file's bytes
    fn read_trailer(bytes: &[u8]) -> Result<SegmentSummary> {
        if bytes.len() < TRAILER_SUFFIX_LEN {
            return Err(ZyronError::CdcDecoderError(
                "sealed change segment is too short to hold a trailer".into(),
            ));
        }
        let suffix = &bytes[bytes.len() - TRAILER_SUFFIX_LEN..];
        if suffix[8..12] != TRAILER_MARKER {
            return Err(ZyronError::CdcDecoderError(
                "sealed change segment does not end with a trailer marker".into(),
            ));
        }
        let mut len_bytes = [0u8; 4];
        len_bytes.copy_from_slice(&suffix[0..4]);
        let body_len = u32::from_le_bytes(len_bytes) as usize;
        let mut crc_bytes = [0u8; 4];
        crc_bytes.copy_from_slice(&suffix[4..8]);
        let stored = u32::from_le_bytes(crc_bytes);
        let body_end = bytes.len() - TRAILER_SUFFIX_LEN;
        if body_len > body_end {
            return Err(ZyronError::CdcDecoderError(
                "sealed change segment declares a trailer longer than the file".into(),
            ));
        }
        let body = &bytes[body_end - body_len..body_end];
        if hot_hash32(body) != stored {
            return Err(ZyronError::CdcDecoderError(
                "sealed change segment trailer checksum does not match".into(),
            ));
        }
        let mut cursor = ByteCursor::new(body);
        SegmentSummary::read_from(&mut cursor)
    }

    // -----------------------------------------------------------------------
    // Read
    // -----------------------------------------------------------------------

    /// The segments a range will open, resolved from the summaries alone
    pub fn plan_read(&self, range: &ChangeRange) -> ReadPlan {
        let inner = self.inner.lock();
        Self::plan_locked(&inner, range)
    }

    fn plan_locked(inner: &CdfInner, range: &ChangeRange) -> ReadPlan {
        let mut segments = Vec::new();
        let mut pruned = 0usize;
        let mut candidate_records = 0u64;
        for summary in inner.sealed.iter().chain(std::iter::once(&inner.open)) {
            if summary.is_empty() {
                continue;
            }
            if summary.overlaps(range) {
                segments.push(summary.seq);
                candidate_records += summary.record_count;
            } else {
                pruned += 1;
            }
        }
        ReadPlan {
            segments,
            pruned,
            candidate_records,
        }
    }

    /// Every record in the range, in (version, ordinal) order
    pub fn read_range(&self, range: &ChangeRange) -> Result<Vec<ChangeRecord>> {
        let mut out = Vec::new();
        self.read_range_into(range, |record| {
            out.push(record);
            Ok(())
        })?;
        Ok(out)
    }

    /// Every record in the range, handed to a visitor one at a time.
    ///
    /// A scan that projects a few columns out of a wide change never needs
    /// the whole answer resident, so the visitor form is what the operator
    /// uses and `read_range` is the convenience over it
    pub fn read_range_into(
        &self,
        range: &ChangeRange,
        mut visit: impl FnMut(ChangeRecord) -> Result<()>,
    ) -> Result<ReadPlan> {
        const MAX_RACE_RETRIES: usize = 8;
        for _ in 0..MAX_RACE_RETRIES {
            let (plan, epoch, open) = self.plan_with_open_tail(range);
            if plan.segments.is_empty() {
                return Ok(plan);
            }
            let mut collected: Vec<ChangeRecord> = Vec::new();
            let mut failed = false;
            for seq in &plan.segments {
                let tail = open.as_ref().filter(|open| open.seq == *seq);
                match self.read_segment(*seq, range, tail, &mut collected) {
                    Ok(()) => {}
                    Err(e) => {
                        // A rewrite between planning and reading removes the
                        // file the plan named. The epoch check below decides
                        // whether that is what happened
                        if self.inner.lock().rewrite_epoch != epoch {
                            failed = true;
                            break;
                        }
                        return Err(e);
                    }
                }
            }
            if failed || self.inner.lock().rewrite_epoch != epoch {
                continue;
            }
            collected.sort_by(|a, b| {
                a.commit_version
                    .cmp(&b.commit_version)
                    .then(a.change_ordinal.cmp(&b.change_ordinal))
            });
            for record in collected {
                visit(record)?;
            }
            return Ok(plan);
        }
        Err(ZyronError::CdcStreamError(format!(
            "the change feed for table {} was rewritten under every read attempt",
            self.table_id
        )))
    }

    /// Plans a read and, when the plan reaches the open segment, takes the
    /// frames the range can want out of memory under the same lock, so
    /// what the read walks is exactly what stood there when it planned
    fn plan_with_open_tail(&self, range: &ChangeRange) -> (ReadPlan, u64, Option<OpenTail>) {
        let inner = self.inner.lock();
        let plan = Self::plan_locked(&inner, range);
        let open = plan.segments.contains(&inner.open.seq).then(|| OpenTail {
            seq: inner.open.seq,
            frames: inner.open_frames_from(range.start_version),
        });
        (plan, inner.rewrite_epoch, open)
    }

    /// The frames of one segment, see [`segment_frames_in`]
    fn segment_frames(&self, seq: u64, open: Option<&OpenTail>) -> Result<Option<Vec<u8>>> {
        segment_frames_in(&self.dir, self.table_id, seq, open)
    }

    /// Reads one segment's records that the range admits
    fn read_segment(
        &self,
        seq: u64,
        range: &ChangeRange,
        open: Option<&OpenTail>,
        out: &mut Vec<ChangeRecord>,
    ) -> Result<()> {
        let Some(frames) = self.segment_frames(seq, open)? else {
            return Ok(());
        };
        let table_id = self.table_id;
        walk_frames(&frames, |record| {
            let (version, timestamp) = ChangeRecord::peek_version_timestamp(record)?;
            let change_type = ChangeRecord::peek_change_type(record)?;
            // The three metadata columns a predicate can prune on are read
            // out of the record header, so a record the range excludes costs
            // no field decoding at all
            if !range.admits(version, timestamp, change_type) {
                return Ok(());
            }
            out.push(ChangeRecord::deserialize(record, table_id)?);
            Ok(())
        })?;
        Ok(())
    }

    /// Opens a read of the range that hands its records over a batch at a
    /// time, keeping its place between batches.
    ///
    /// `resume` names the last position already handed over, so a scan that
    /// continues an earlier one skips every segment that ends at or before
    /// it and every record at or before it in the segment it starts in
    pub fn open_scan(&self, range: &ChangeRange, resume: Option<(u64, u64)>) -> SegmentScan {
        let inner = self.inner.lock();
        let mut plan = Self::plan_locked(&inner, range);
        if let Some((at_version, _)) = resume {
            // A segment whose newest record is at or before the resume point
            // holds nothing to hand over. Every record of that version is
            // skipped by ordinal in the first segment kept
            let summaries: HashMap<u64, u64> = inner
                .sealed
                .iter()
                .chain(std::iter::once(&inner.open))
                .map(|summary| (summary.seq, summary.max_version))
                .collect();
            plan.segments.retain(|seq| {
                summaries
                    .get(seq)
                    .is_none_or(|max_version| *max_version >= at_version)
            });
        }
        let open = plan.segments.contains(&inner.open.seq).then(|| OpenTail {
            seq: inner.open.seq,
            frames: inner.open_frames_from(range.start_version),
        });
        SegmentScan {
            dir: self.dir.clone(),
            table_id: self.table_id,
            range: range.clone(),
            resume,
            plan,
            at: 0,
            open,
            current: None,
        }
    }

    /// Walks the range handing each record's fields to a visitor without
    /// copying its row bytes.
    ///
    /// This is what a scan reads through. The materializing forms above own
    /// a `ChangeRecord` per record, which costs an allocation each, and a
    /// scan of ten million changes decodes straight into column builders
    /// instead.
    ///
    /// Segments are walked in sequence order, and a segment's records are
    /// written in the order they were appended, so what the visitor sees is
    /// already in (commit version, position within the commit) order
    pub fn scan_range(
        &self,
        range: &ChangeRange,
        mut visit: impl FnMut(ChangeRecordRef<'_>) -> Result<()>,
    ) -> Result<ReadPlan> {
        const MAX_RACE_RETRIES: usize = 8;
        for _ in 0..MAX_RACE_RETRIES {
            let (plan, epoch, open) = self.plan_with_open_tail(range);
            if plan.segments.is_empty() {
                return Ok(plan);
            }
            let mut retry = false;
            for seq in &plan.segments {
                let tail = open.as_ref().filter(|open| open.seq == *seq);
                match self.scan_segment(*seq, range, tail, &mut visit) {
                    Ok(()) => {}
                    Err(e) => {
                        if self.inner.lock().rewrite_epoch != epoch {
                            retry = true;
                            break;
                        }
                        return Err(e);
                    }
                }
            }
            if retry || self.inner.lock().rewrite_epoch != epoch {
                continue;
            }
            return Ok(plan);
        }
        Err(ZyronError::CdcStreamError(format!(
            "the change feed for table {} was rewritten under every read attempt",
            self.table_id
        )))
    }

    /// Walks one segment's records that the range admits
    fn scan_segment(
        &self,
        seq: u64,
        range: &ChangeRange,
        open: Option<&OpenTail>,
        visit: &mut impl FnMut(ChangeRecordRef<'_>) -> Result<()>,
    ) -> Result<()> {
        let Some(frames) = self.segment_frames(seq, open)? else {
            return Ok(());
        };
        let table_id = self.table_id;
        walk_frames(&frames, |record| {
            let view = ChangeRecordRef::of(record, table_id)?;
            // The three metadata columns a predicate prunes on are read out
            // of the record header, so a record the range excludes costs no
            // field decoding at all
            if !range.admits(view.commit_version, view.commit_timestamp, view.change_type) {
                return Ok(());
            }
            visit(view)
        })?;
        Ok(())
    }

    /// Queries change records by version range, inclusive at both ends
    pub fn query_changes(&self, start_version: u64, end_version: u64) -> Result<Vec<ChangeRecord>> {
        self.read_range(&ChangeRange::versions(start_version, end_version))
    }

    /// Queries change records by timestamp range, inclusive at both ends
    pub fn query_changes_by_time(&self, start_ts: i64, end_ts: i64) -> Result<Vec<ChangeRecord>> {
        self.read_range(&ChangeRange::timestamps(start_ts, end_ts))
    }

    // -----------------------------------------------------------------------
    // Counters
    // -----------------------------------------------------------------------

    /// Records the feed holds right now
    pub fn record_count(&self) -> u64 {
        self.counters.record_count.load(Ordering::Acquire)
    }

    /// Bytes every segment occupies
    pub fn file_size_bytes(&self) -> u64 {
        self.counters.bytes.load(Ordering::Acquire)
    }

    /// The newest commit version the feed holds, None when it is empty
    pub fn latest_version(&self) -> Option<u64> {
        let latest = self.counters.latest_version.load(Ordering::Acquire);
        if latest == 0 {
            self.inner.lock().versions.last().map(|v| v.version)
        } else {
            Some(latest)
        }
    }

    /// The oldest commit version the feed still holds, None when it is empty
    pub fn oldest_version(&self) -> Option<u64> {
        self.inner.lock().versions.first().map(|v| v.version)
    }

    /// The form a segment's file takes on disk, None for a segment the feed
    /// does not hold
    pub fn segment_form(&self, seq: u64) -> Result<Option<SegmentForm>> {
        let path = self.segment_path(seq);
        if !path.exists() {
            return Ok(None);
        }
        let mut bytes = vec![0u8; SEGMENT_BODY_OFFSET as usize];
        let mut file = File::open(&path)?;
        let read = file.read(&mut bytes)?;
        if read < SEGMENT_BODY_OFFSET as usize {
            return Ok(Some(SegmentForm::Open));
        }
        let header = read_segment_header(&bytes)?;
        Ok(Some(if !header.sealed {
            SegmentForm::Open
        } else if header.columnar {
            SegmentForm::SealedColumns
        } else {
            SegmentForm::SealedFrames
        }))
    }

    /// The lowest schema epoch any record the feed still holds was written
    /// under, None when it holds none. The table keeps that epoch's layout
    /// recorded for as long as the feed holds the record
    pub fn oldest_schema_epoch(&self) -> Option<u32> {
        let inner = self.inner.lock();
        inner
            .sealed
            .iter()
            .chain(std::iter::once(&inner.open))
            .filter(|summary| !summary.is_empty())
            .map(|summary| summary.min_epoch)
            .min()
    }

    /// The highest commit version a purge has reclaimed, zero when the feed
    /// has never purged.
    ///
    /// A consumer whose position is at or below it has lost changes, which is
    /// what makes its stream stale rather than merely behind
    pub fn purge_floor(&self) -> u64 {
        self.inner.lock().purge_floor
    }

    /// The timestamp of the oldest change the feed still holds
    pub fn oldest_timestamp(&self) -> Option<i64> {
        self.inner
            .lock()
            .versions
            .first()
            .map(|v| v.first_timestamp)
    }

    /// Records after `version`, read from the per-version counters.
    ///
    /// No change file is opened, which is what lets a lag view answer for
    /// every stream on a node without touching storage
    pub fn pending_after(&self, version: u64) -> u64 {
        let inner = self.inner.lock();
        inner
            .records_written
            .saturating_sub(inner.records_at_or_below(version))
    }

    /// Records at or below `version`, counted from the feed's creation.
    ///
    /// This is the number a change stream position replicates as. Every
    /// member of a group records the same changes in the same order, so the
    /// count names the same place on all of them while the version that
    /// addresses it is each member's own
    pub fn records_at_or_below(&self, version: u64) -> u64 {
        self.inner.lock().records_at_or_below(version)
    }

    /// The version and ordinal of the record at which the feed had written
    /// exactly `count` records, which is where a read resumes after a
    /// position that consumed that many. None for a count of zero or one
    /// that names a record retention has reclaimed
    pub fn cursor_at_count(&self, count: u64) -> Option<(u64, u64)> {
        self.inner.lock().cursor_at_count(count)
    }

    /// Records written before the record at `ordinal` within `version`,
    /// counted from the feed's creation
    pub fn records_before(&self, version: u64, ordinal: u64) -> u64 {
        self.inner.lock().records_before(version, ordinal)
    }

    /// How many times a byte cap has purged this feed ahead of retention
    pub fn cap_purges(&self) -> u64 {
        self.inner.lock().cap_purges
    }

    /// The version a read that resumes after `consumed` records and takes
    /// at most `max_rows` more ends at, moved past every transaction that
    /// wrote inside the read and again beyond it. None when fewer than that
    /// many records lie past the position
    pub fn bounded_cut(&self, from_exclusive: u64, consumed: u64, max_rows: u64) -> Option<u64> {
        self.inner
            .lock()
            .bounded_cut(from_exclusive, consumed, max_rows)
    }

    /// Every transaction with records in the feed and where each one's
    /// records lie, ascending by first version
    pub fn txn_spans(&self) -> Vec<TxnSpan> {
        self.inner.lock().spans.clone()
    }

    /// What the feed holds right now.
    ///
    /// `ended` says whether a transaction is over, committed or aborted, and
    /// the transactions it says are over are forgotten as unfinished ones
    pub fn boundary(&self, ended: &dyn Fn(u64) -> bool) -> FeedBoundary {
        self.inner.lock().boundary(ended)
    }

    /// Forgets the transactions `ended` says are over, answering with how
    /// many are still open. A read forgets them as it goes, and this is
    /// for a feed nothing reads, so it does not keep every transaction
    /// that ever wrote to it
    pub fn prune_in_flight(&self, ended: &dyn Fn(u64) -> bool) -> usize {
        let mut inner = self.inner.lock();
        inner.in_flight.retain(|txn_id, _| !ended(*txn_id));
        inner.in_flight.len()
    }

    /// The lowest version any unfinished transaction wrote at, None when
    /// every record belongs to a transaction that has ended
    pub fn first_open_version(&self, ended: &dyn Fn(u64) -> bool) -> Option<u64> {
        self.inner.lock().first_open_version(ended)
    }

    /// The version at which the feed had written exactly `count` records.
    ///
    /// The inverse of `records_at_or_below`, used when a replicated position
    /// arrives as a count and this member has to name it in its own versions.
    /// A count above everything the feed holds answers with its latest
    /// version, which is the furthest a position can legitimately reach
    pub fn version_at_count(&self, count: u64) -> u64 {
        version_index::version_at_count(&self.inner.lock().versions, count)
    }

    /// Commit versions after `version`
    pub fn pending_versions_after(&self, version: u64) -> u64 {
        let inner = self.inner.lock();
        let at = inner.versions.partition_point(|e| e.version <= version);
        (inner.versions.len() - at) as u64
    }

    /// The timestamp of the oldest change after `version`, for a lag reading
    /// The newest version whose records were committed at or before
    /// `timestamp`, zero when none was. Answered from the version index,
    /// which ascends in time the way it ascends in version, rather than by
    /// walking the records up to the instant
    pub fn version_at_or_before(&self, timestamp: i64) -> u64 {
        let inner = self.inner.lock();
        let past = inner
            .versions
            .partition_point(|v| v.first_timestamp <= timestamp);
        past.checked_sub(1)
            .and_then(|at| inner.versions.get(at))
            .map(|v| v.version)
            .unwrap_or(0)
    }

    pub fn first_timestamp_after(&self, version: u64) -> Option<i64> {
        self.inner
            .lock()
            .first_after(version)
            .map(|v| v.first_timestamp)
    }

    /// Every segment summary, for the maintenance and observability paths
    pub fn segment_summaries(&self) -> Vec<SegmentSummary> {
        self.inner.lock().all_segments()
    }

    // -----------------------------------------------------------------------
    // Purge
    // -----------------------------------------------------------------------

    /// Purges records with commit_version < min_version.
    pub fn purge_before_version(&self, min_version: u64) -> Result<u64> {
        self.purge_where(|version, _| version < min_version)
    }

    /// Purges records whose commit timestamp is older than the cutoff. A
    /// hold LSN keeps records above it regardless of age, so a slow but
    /// advancing subscriber never loses changes it has not confirmed
    pub fn purge_retention(&self, cutoff_timestamp: i64, hold_lsn: Option<u64>) -> Result<u64> {
        self.purge_where(move |version, timestamp| {
            timestamp < cutoff_timestamp && hold_lsn.is_none_or(|hold| version <= hold)
        })
    }

    /// Purges oldest first until the feed is at or under `max_bytes`.
    ///
    /// A table must never stop accepting writes because its feed is full, so
    /// this reclaims rather than refusing. Returns the records it removed
    pub fn enforce_byte_cap(&self, max_bytes: u64) -> Result<u64> {
        if max_bytes == 0 || self.file_size_bytes() <= max_bytes {
            return Ok(0);
        }
        let floor = {
            let inner = self.inner.lock();
            let mut total = inner.sealed.iter().map(|s| s.bytes).sum::<u64>() + inner.open.bytes;
            let mut floor = 0u64;
            for summary in &inner.sealed {
                if total <= max_bytes {
                    break;
                }
                total -= summary.bytes;
                floor = summary.max_version + 1;
            }
            floor
        };
        if floor == 0 {
            return Ok(0);
        }
        let removed = self.purge_before_version(floor)?;
        // Counted once the cap took something, since a stream reading from
        // the oldest change judges its staleness by this count
        if removed > 0 {
            self.inner.lock().cap_purges += 1;
        }
        Ok(removed)
    }

    /// Removes every record the predicate purges, dropping whole segments
    /// where it can and rewriting the one segment a boundary falls inside
    fn purge_where(&self, purge: impl Fn(u64, i64) -> bool + Copy) -> Result<u64> {
        let mut inner = self.inner.lock();
        let segments = inner.all_segments();
        let mut removed_total = 0u64;
        let mut floor = inner.purge_floor;

        for summary in segments {
            if summary.is_empty() {
                continue;
            }
            let whole = purge(summary.max_version, summary.max_timestamp);
            let none = !purge(summary.min_version, summary.min_timestamp);
            if none {
                continue;
            }
            if whole && summary.seq != inner.open.seq {
                let path = self.segment_path(summary.seq);
                if path.exists() {
                    fs::remove_file(&path)?;
                }
                inner.sealed.retain(|s| s.seq != summary.seq);
                removed_total += summary.record_count;
                floor = floor.max(summary.max_version);
                continue;
            }
            let (removed, highest) = self.rewrite_segment(&mut inner, summary.seq, purge)?;
            removed_total += removed;
            floor = floor.max(highest);
        }

        if removed_total == 0 {
            return Ok(0);
        }

        inner.records_purged += removed_total;
        inner.purge_floor = floor;
        inner.rewrite_epoch += 1;
        // A version whose records are all gone leaves the index, and the ones
        // that remain keep their absolute `prior`, so a position below the
        // floor still resolves to the count that was written beneath it
        Self::prune_version_index(&mut inner);
        self.sync_open_locked(&mut inner)?;
        Self::persist_manifest_whole(&self.dir, &mut inner)?;
        self.counters.publish(&inner);
        Ok(removed_total)
    }

    /// Rewrites one segment keeping the records the predicate does not purge.
    ///
    /// Answers with the records it removed and the highest version among them,
    /// which is what the feed's purge floor advances to
    fn rewrite_segment(
        &self,
        inner: &mut CdfInner,
        seq: u64,
        purge: impl Fn(u64, i64) -> bool,
    ) -> Result<(u64, u64)> {
        let is_open = inner.open.seq == seq;
        if is_open {
            inner.writer = None;
        }
        // The open segment is rewritten from the frames held in memory,
        // which are the file's contents as this feed wrote them
        let open = is_open.then(|| OpenTail {
            seq,
            frames: inner.open_frames.clone(),
        });
        let mut kept: Vec<ChangeRecord> = Vec::new();
        let mut all = Vec::new();
        self.read_segment(seq, &ChangeRange::everything(), open.as_ref(), &mut all)?;
        let before = all.len() as u64;
        let mut highest_purged = 0u64;
        for record in all {
            if purge(record.commit_version, record.commit_timestamp) {
                highest_purged = highest_purged.max(record.commit_version);
            } else {
                kept.push(record);
            }
        }
        let removed = before.saturating_sub(kept.len() as u64);
        if removed == 0 {
            return Ok((0, 0));
        }

        let codec = inner
            .summary_for(seq)
            .map(|s| s.codec)
            .unwrap_or(inner.config.codec);
        let sealed = !is_open;
        let mut summary = SegmentSummary::empty(seq, codec);
        let mut frames = Vec::with_capacity(kept.len() * 64);
        for record in &kept {
            summary.observe(
                record.commit_version,
                record.commit_timestamp,
                record.change_type,
                record.schema_version,
            );
            frame_into(&mut frames, record, record.change_ordinal);
        }
        summary.sealed = sealed;

        let path = self.segment_path(seq);
        // Named for the rewrite, apart from the name a sealer writing the
        // same segment takes
        let tmp = path.with_extension("zycdf.rewrite");
        {
            let file = File::create(&tmp)?;
            let mut writer = BufWriter::new(file);
            if sealed {
                let (body, flags) = sealed_body(&frames, codec, &inner.layouts)?;
                write_segment_header(&mut writer, self.table_id, seq, flags)?;
                writer.write_all(&body)?;
                let trailer = Self::encode_trailer(&summary);
                writer.write_all(&trailer)?;
                summary.bytes = SEGMENT_BODY_OFFSET + body.len() as u64 + trailer.len() as u64;
            } else {
                write_segment_header(&mut writer, self.table_id, seq, 0)?;
                writer.write_all(&frames)?;
                summary.bytes = SEGMENT_BODY_OFFSET + frames.len() as u64;
            }
            writer.flush()?;
            writer.get_ref().sync_all()?;
        }
        fs::rename(&tmp, &path)?;
        sync_parent_dir(&path)?;

        if is_open {
            inner.open = summary;
            inner.synced_bytes = inner.open.bytes;
            inner.reset_open_frames(frames);
        } else if let Some(slot) = inner.sealed.iter_mut().find(|s| s.seq == seq) {
            *slot = summary;
        }
        Ok((removed, highest_purged))
    }

    /// Drops version index entries whose records the feed no longer holds
    fn prune_version_index(inner: &mut CdfInner) {
        let (lowest, highest) = match (
            inner
                .sealed
                .iter()
                .chain(std::iter::once(&inner.open))
                .filter(|s| !s.is_empty())
                .map(|s| s.min_version)
                .min(),
            inner
                .sealed
                .iter()
                .chain(std::iter::once(&inner.open))
                .filter(|s| !s.is_empty())
                .map(|s| s.max_version)
                .max(),
        ) {
            (Some(low), Some(high)) => (low, high),
            _ => {
                inner.versions.clear();
                inner.spans.clear();
                inner.reindex_spans();
                return;
            }
        };
        inner
            .versions
            .retain(|entry| entry.version >= lowest && entry.version <= highest);
        // A transaction whose every record is gone leaves with them
        inner.spans.retain(|span| span.last >= lowest);
        inner.reindex_spans();
    }

    // -----------------------------------------------------------------------
    // Compaction
    // -----------------------------------------------------------------------

    /// Plans a compaction pass over the sealed segments, the ones no append
    /// reaches, so records landing while the pass runs are left where they
    /// land. The open segment is compacted once it seals and a later pass
    /// reaches it
    pub fn plan_compaction(&self) -> CompactionPlan {
        let inner = self.inner.lock();
        let segments: Vec<SegmentSummary> = inner
            .sealed
            .iter()
            .filter(|s| s.sealed && !s.is_empty())
            .cloned()
            .collect();
        let record_count = segments.iter().map(|s| s.record_count).sum();
        CompactionPlan {
            segments,
            epoch: inner.rewrite_epoch,
            layouts: inner.layouts.clone(),
            record_count,
        }
    }

    /// Hands every record of the planned segments to `visit`, in version
    /// then ordinal order, one segment in memory at a time. Answers false,
    /// without visiting the rest, when a purge or another pass rewrote the
    /// feed since the plan was taken
    pub fn visit_planned(
        &self,
        plan: &CompactionPlan,
        visit: &mut dyn FnMut(&ChangeRecord) -> Result<()>,
    ) -> Result<bool> {
        let table_id = self.table_id;
        for summary in &plan.segments {
            if self.inner.lock().rewrite_epoch != plan.epoch {
                return Ok(false);
            }
            let Some(frames) = self.segment_frames(summary.seq, None)? else {
                return Ok(false);
            };
            walk_frames(&frames, |frame| {
                visit(&ChangeRecord::deserialize(frame, table_id)?)
            })?;
        }
        Ok(self.inner.lock().rewrite_epoch == plan.epoch)
    }

    /// Rewrites each planned segment with the records `keep` admits, read,
    /// decided and compressed off the lock and put in place under it, so a
    /// write appending meanwhile waits for none of that. Answers with the
    /// records removed, None when a purge or another pass rewrote the feed
    /// since the plan was taken, in which case the segments already put in
    /// place stand, each whole on its own, and the rest wait for the next
    /// pass.
    ///
    /// The version index and the spans stay as they are. A record removed
    /// keeps its place in the feed's numbering, so every position counted
    /// before the pass names the same place after it, with fewer records to
    /// hand over on the way there
    pub fn compact_planned(
        &self,
        plan: &CompactionPlan,
        keep: &mut dyn FnMut(&ChangeRecord) -> bool,
    ) -> Result<Option<u64>> {
        let mut removed_total = 0u64;
        for summary in &plan.segments {
            match self.compact_segment(plan, summary, keep)? {
                Some(removed) => removed_total += removed,
                None => return Ok(None),
            }
        }
        Ok(Some(removed_total))
    }

    /// Rewrites one planned segment, answering with the records removed,
    /// None when the feed moved on since the plan
    fn compact_segment(
        &self,
        plan: &CompactionPlan,
        summary: &SegmentSummary,
        keep: &mut dyn FnMut(&ChangeRecord) -> bool,
    ) -> Result<Option<u64>> {
        let seq = summary.seq;
        if self.inner.lock().rewrite_epoch != plan.epoch {
            return Ok(None);
        }
        let Some(frames) = self.segment_frames(seq, None)? else {
            return Ok(None);
        };
        let table_id = self.table_id;
        let mut rewritten = SegmentSummary::empty(seq, summary.codec);
        rewritten.sealed = true;
        let mut kept_frames = Vec::with_capacity(frames.len());
        let mut removed = 0u64;
        walk_frames(&frames, |frame| {
            let record = ChangeRecord::deserialize(frame, table_id)?;
            if keep(&record) {
                rewritten.observe(
                    record.commit_version,
                    record.commit_timestamp,
                    record.change_type,
                    record.schema_version,
                );
                frame_into(&mut kept_frames, &record, record.change_ordinal);
            } else {
                removed += 1;
            }
            Ok(())
        })?;
        if removed == 0 {
            return Ok(Some(0));
        }
        // The sealed form, written beside the file it replaces under a name
        // apart from the ones a sealer and a purge write through
        let path = self.segment_path(seq);
        let tmp = path.with_extension("zycdf.compact");
        let (body, flags) = sealed_body(&kept_frames, summary.codec, &plan.layouts)?;
        let trailer = Self::encode_trailer(&rewritten);
        {
            let file = File::create(&tmp)?;
            let mut writer = BufWriter::new(file);
            write_segment_header(&mut writer, table_id, seq, flags)?;
            writer.write_all(&body)?;
            writer.write_all(&trailer)?;
            writer.flush()?;
            writer.get_ref().sync_all()?;
        }
        rewritten.bytes = SEGMENT_BODY_OFFSET + body.len() as u64 + trailer.len() as u64;
        // Under the lock, the file takes the segment's place unless the feed
        // moved on. A planned segment that is gone or no longer sealed was
        // rewritten by a purge, which moved the epoch
        let mut inner = self.inner.lock();
        if inner.rewrite_epoch != plan.epoch || inner.closed {
            drop(inner);
            let _ = fs::remove_file(&tmp);
            return Ok(None);
        }
        let Some(slot) = inner.sealed.iter_mut().find(|s| s.seq == seq && s.sealed) else {
            drop(inner);
            let _ = fs::remove_file(&tmp);
            return Ok(None);
        };
        fs::rename(&tmp, &path)?;
        sync_parent_dir(&path)?;
        *slot = rewritten;
        inner.chain.sealed_dirty.push(seq);
        inner.records_purged += removed;
        Self::persist_manifest(&self.dir, &mut inner)?;
        self.counters.publish(&inner);
        Ok(Some(removed))
    }

    // -----------------------------------------------------------------------
    // Durability
    // -----------------------------------------------------------------------

    /// Forces every appended record through the device cache. Appends flush
    /// to the OS on every call, this makes them durable. Returns whether
    /// anything was pending
    pub fn sync_to_disk(&self) -> Result<bool> {
        if !self.unsynced.load(Ordering::Acquire) {
            return Ok(false);
        }
        let mut inner = self.inner.lock();
        self.sync_open_locked(&mut inner)
    }

    /// Makes every closed segment a rotation left unsynced and the open
    /// segment durable through its end, then takes the counters as the
    /// mark, all under the lock so nothing lands between the sync and the
    /// mark. Answers whether the open segment had anything to sync
    fn sync_open_locked(&self, inner: &mut CdfInner) -> Result<bool> {
        let mut closed = 0usize;
        for seq in &inner.closed_unsynced {
            let path = self.segment_path(*seq);
            if path.exists() {
                OpenOptions::new().write(true).open(&path)?.sync_data()?;
            }
            closed += 1;
        }
        inner.closed_unsynced.drain(..closed);
        let synced = if inner.open.bytes == inner.synced_bytes {
            false
        } else {
            match inner.writer.as_ref() {
                Some(writer) => {
                    writer.sync_data()?;
                }
                None => {
                    let path = self.segment_path(inner.open.seq);
                    OpenOptions::new().write(true).open(&path)?.sync_data()?;
                }
            }
            inner.synced_bytes = inner.open.bytes;
            true
        };
        inner.synced_mark = inner.take_mark();
        // Every logged append is in a synced file from here, so the log
        // need not keep any of them. Under the lock, the same lock an
        // append sets both under, so a record landing during the sync is
        // not forgotten by it
        inner.logged_from = None;
        self.unsynced.store(false, Ordering::Release);
        Ok(synced)
    }

    /// Writes the manifest, for a shutdown that wants the counters on disk
    /// without waiting for the next seal
    pub fn checkpoint(&self) -> Result<()> {
        let mut inner = self.inner.lock();
        self.sync_open_locked(&mut inner)?;
        Self::persist_manifest(&self.dir, &mut inner)
    }

    /// Removes every file the feed owns
    fn remove_files(&self) -> Result<()> {
        // Under the lock, so a sealer finishing afterwards finds the feed
        // closed and writes nothing where the directory was
        let mut inner = self.inner.lock();
        inner.closed = true;
        inner.writer = None;
        if self.dir.exists() {
            fs::remove_dir_all(&self.dir)?;
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// CdfRegistry
// ---------------------------------------------------------------------------

/// A source whose changes are derived from a store of its own rather than
/// recorded in a feed, which is what a lake table is. Its transaction log
/// is the change record.
///
/// A stream over such a source keeps a position the same way, and what the
/// stream runtime asks of a feed about that position it asks of this
pub trait DerivedChangeSource: Send + Sync {
    /// The newest version the source's changes reach, zero when none
    fn latest_version(&self) -> u64;
    /// The oldest version a read of the changes above it still answers,
    /// None when every version does. A position below it is stale
    fn oldest_readable_version(&self) -> Option<u64>;
    /// Changes above a position, from the source's own counters
    fn pending_after(&self, position: u64) -> u64;
    /// The instant of the oldest change above a position
    fn first_timestamp_after(&self, position: u64) -> Option<i64>;
    /// The last version at or before an instant, for a reset by timestamp
    fn version_at_timestamp(&self, timestamp: i64) -> u64;
    /// Records at or below a version, counted from the version the feed
    /// began at, which is the number a stream position replicates as. An
    /// error when the source could not count a version the answer needs,
    /// so a position is never moved by a count that stopped short
    fn records_at_or_below(&self, version: u64) -> Result<u64>;
    /// Records before the record at `ordinal` within `version`, counted
    /// the same way
    fn records_before(&self, version: u64, ordinal: u64) -> Result<u64>;
    /// The version and ordinal of the record at which the source had
    /// yielded exactly `count` records, None for a count of zero
    fn cursor_at_count(&self, count: u64) -> Result<Option<(u64, u64)>>;
    /// The version at which the source had yielded exactly `count`
    /// records, the inverse of `records_at_or_below`, answered from what
    /// the source has counted
    fn version_at_count(&self, count: u64) -> u64;
    /// The version a read that resumes after `consumed` records and takes
    /// at most `max_rows` more ends at. A version is one commit, so the
    /// read hands over whole transactions by ending on one. None when
    /// fewer than that many records lie past the position
    fn bounded_cut(&self, consumed: u64, max_rows: u64) -> Result<Option<u64>>;
    /// Whether an update's removed side is a record of `version`, the
    /// feed's setting as it stood when the version's records were counted,
    /// which is the setting a read of the version derives them under
    fn preimages_at(&self, version: u64) -> bool;
    /// What the source holds at one instant. `ended` says whether a
    /// transaction is over as of the read's snapshot, and the boundary
    /// names the lowest commit readers can see of one that is not, which
    /// a read stops short of
    fn boundary(&self, ended: &dyn Fn(u64) -> bool) -> Result<crate::DerivedBoundary>;
    /// The transactions with a commit of this source above `head`, whether
    /// readers can see it yet or not, apart from those `aborted` says
    /// rolled back. A reader of another source at a later instant holds
    /// these open, so a transaction is never handed over by halves
    fn writers_above(&self, head: u64, aborted: &dyn Fn(u64) -> bool) -> Result<Vec<u64>>;
    /// The transactions with a counted commit in `(from_exclusive,
    /// to_inclusive]`, each once, answered from what the source has
    /// counted so far, which a boundary read built through the head
    fn txns_in(&self, from_exclusive: u64, to_inclusive: u64) -> Vec<u64>;
    /// The lowest and highest version a transaction committed to this
    /// source at, None for one with no counted commit here
    fn span_of(&self, txn_id: u64) -> Option<(u64, u64)>;
}

/// One source of a stream read with the window the read resolved for it,
/// which `CdfRegistry::align_windows` moves so that no transaction is
/// handed over by halves
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SourceRead {
    pub table_id: u32,
    pub branch: Option<u64>,
    /// The version the read starts after
    pub from_exclusive: u64,
    /// The version the read ends at as resolved for this source alone
    pub to_inclusive: u64,
    /// The highest version the read may end at, the newest the source
    /// holds below the first commit of a transaction still open as of the
    /// read
    pub limit: u64,
}

/// Global registry of change data feeds, one per feed-enabled table
pub struct CdfRegistry {
    feeds: SccHashMap<u32, Arc<ChangeDataFeed>>,
    /// Tables whose changes come from a store of their own, by table
    derived: SccHashMap<u32, Arc<dyn DerivedChangeSource>>,
    /// A branch's head on a table whose changes are derived, by table then
    /// branch, the way `branch_feeds` holds a branch's feed on a heap
    /// table. Registered when the branch forks the table and again on the
    /// first read after a restart
    derived_branches: SccHashMap<(u32, u64), Arc<dyn DerivedChangeSource>>,
    /// A branch's feed on a table, by table then branch. Opened when the
    /// branch is created or on its first captured write, and opened again
    /// from disk on the first read after a restart
    branch_feeds: SccHashMap<(u32, u64), Arc<ChangeDataFeed>>,
    data_dir: PathBuf,
    /// The log every feed opened here records its appends in, attached
    /// once by the owner that holds the writer, before any feed is opened
    log: std::sync::OnceLock<Arc<zyron_wal::WalWriter>>,
}

impl CdfRegistry {
    pub fn new(data_dir: PathBuf) -> Self {
        Self {
            feeds: SccHashMap::new(),
            derived: SccHashMap::new(),
            derived_branches: SccHashMap::new(),
            branch_feeds: SccHashMap::new(),
            data_dir,
            log: std::sync::OnceLock::new(),
        }
    }

    /// Attaches the log every feed's appends are recorded in, to the feeds
    /// already open and to every feed opened from here. The first
    /// attachment is the one kept
    pub fn attach_wal(&self, wal: &Arc<zyron_wal::WalWriter>) {
        let _ = self.log.set(Arc::clone(wal));
        for (_, feed) in self.open_feeds() {
            feed.attach_wal(wal);
        }
    }

    /// Hands a feed opened here the log, when one is attached
    fn adopt(&self, feed: Arc<ChangeDataFeed>) -> Arc<ChangeDataFeed> {
        if let Some(wal) = self.log.get() {
            feed.attach_wal(wal);
        }
        feed
    }

    /// The oldest log position any feed still needs, the record of the
    /// oldest append not yet synced to its file across every open feed.
    /// None when every logged append is in a synced file. What the log's
    /// segment retention keeps
    pub fn retained_lsn(&self) -> Option<zyron_wal::Lsn> {
        self.open_feeds()
            .into_iter()
            .filter_map(|(_, feed)| feed.logged_from())
            .min()
    }

    /// Registers a table whose changes are derived rather than recorded,
    /// replacing what was registered for it
    pub fn register_derived(&self, table_id: u32, source: Arc<dyn DerivedChangeSource>) {
        self.register_derived_on(table_id, None, source);
    }

    /// Registers the derived source of a table, or of a branch's head on
    /// it, replacing what was registered for the same source
    pub fn register_derived_on(
        &self,
        table_id: u32,
        branch: Option<u64>,
        source: Arc<dyn DerivedChangeSource>,
    ) {
        match branch {
            Some(branch) => {
                let _ = self
                    .derived_branches
                    .upsert_sync((table_id, branch), source);
            }
            None => {
                let _ = self.derived.upsert_sync(table_id, source);
            }
        }
    }

    /// The derived source a table registered, when it did
    pub fn derived(&self, table_id: u32) -> Option<Arc<dyn DerivedChangeSource>> {
        self.derived_on(table_id, None)
    }

    /// The derived source a table, or a branch's head on it, registered
    pub fn derived_on(
        &self,
        table_id: u32,
        branch: Option<u64>,
    ) -> Option<Arc<dyn DerivedChangeSource>> {
        let mut result = None;
        match branch {
            Some(branch) => {
                self.derived_branches
                    .read_sync(&(table_id, branch), |_, v| {
                        result = Some(Arc::clone(v));
                    });
            }
            None => {
                self.derived.read_sync(&table_id, |_, v| {
                    result = Some(Arc::clone(v));
                });
            }
        }
        result
    }

    /// Forgets a table's derived source and every branch's on it
    pub fn remove_derived(&self, table_id: u32) {
        let _ = self.derived.remove_sync(&table_id);
        self.derived_branches
            .retain_sync(|(table, _), _| *table != table_id);
    }

    /// A branch's feed on a table, when the branch has one. One left on
    /// disk by an earlier life of the process is opened again here
    pub fn branch_feed(&self, table_id: u32, branch_id: u64) -> Option<Arc<ChangeDataFeed>> {
        let mut result = None;
        self.branch_feeds.read_sync(&(table_id, branch_id), |_, v| {
            result = Some(v.clone());
        });
        if result.is_some() {
            return result;
        }
        let dir = ChangeDataFeed::branch_dir(&self.data_dir, table_id, branch_id);
        if !dir.exists() {
            return None;
        }
        self.open_branch_feed(table_id, branch_id).ok()
    }

    /// Opens a branch's feed on a table, taking the branch at the table's
    /// current version when the feed does not exist yet.
    ///
    /// The feed takes the table's feed settings as they stand, and records
    /// the version the branch was taken at, so a read inside the branch
    /// knows where the table's changes end and the branch's begin
    pub fn open_branch_feed(&self, table_id: u32, branch_id: u64) -> Result<Arc<ChangeDataFeed>> {
        if let Some(existing) = self.branch_feed_open(table_id, branch_id) {
            return Ok(existing);
        }
        let parent = self.get_feed(table_id).ok_or_else(|| {
            ZyronError::CdcStreamError(format!(
                "table {table_id} has no change data feed, so a branch of it records no changes"
            ))
        })?;
        let mut config = parent.config();
        config.branch_point = parent.latest_version().unwrap_or(0);
        let feed = self.adopt(Arc::new(
            ChangeDataFeed::open_in(
                ChangeDataFeed::branch_dir(&self.data_dir, table_id, branch_id),
                table_id,
                config,
            )?
            .on_branch(branch_id),
        ));
        let _ = self
            .branch_feeds
            .insert_sync((table_id, branch_id), Arc::clone(&feed));
        Ok(feed)
    }

    fn branch_feed_open(&self, table_id: u32, branch_id: u64) -> Option<Arc<ChangeDataFeed>> {
        let mut result = None;
        self.branch_feeds.read_sync(&(table_id, branch_id), |_, v| {
            result = Some(v.clone());
        });
        result
    }

    /// Removes every feed a branch had, with its files
    pub fn remove_branch(&self, branch_id: u64) -> Result<()> {
        let mut keys = Vec::new();
        self.branch_feeds.iter_sync(|key, _| {
            if key.1 == branch_id {
                keys.push(*key);
            }
            true
        });
        for key in keys {
            if let Some((_, feed)) = self.branch_feeds.remove_sync(&key) {
                feed.disable();
                feed.remove_files()?;
            }
        }
        // A branch feed left on disk by an earlier life of the process, never
        // opened in this one, goes the same way
        for table_id in self.table_ids() {
            let dir = ChangeDataFeed::branch_dir(&self.data_dir, table_id, branch_id);
            if dir.exists() {
                fs::remove_dir_all(&dir)?;
            }
        }
        // The branch's heads on the tables whose changes are derived, with
        // the record index each one kept under the same directory
        let mut derived_tables = Vec::new();
        self.derived_branches.retain_sync(|(table, branch), _| {
            if *branch == branch_id {
                derived_tables.push(*table);
            }
            *branch != branch_id
        });
        for table_id in derived_tables {
            let dir = ChangeDataFeed::branch_dir(&self.data_dir, table_id, branch_id);
            match fs::remove_dir_all(&dir) {
                Ok(()) => {}
                Err(e) if e.kind() == std::io::ErrorKind::NotFound => {}
                Err(e) => return Err(e.into()),
            }
        }
        Ok(())
    }

    /// What several feeds hold at one instant.
    ///
    /// Every feed's lock is held while the boundaries are read, in table id
    /// order so two readers never wait on each other, and a write appends to
    /// one feed at a time under its own lock. So no record lands in any of
    /// them between the first reading and the last, and a transaction that
    /// wrote to two of them is either wholly counted or wholly open. A table
    /// with no feed answers with an empty boundary
    pub fn boundaries(
        &self,
        sources: &[(u32, Option<u64>)],
        ended: &dyn Fn(u64) -> bool,
    ) -> Vec<((u32, Option<u64>), FeedBoundary)> {
        let mut ordered: Vec<(u32, Option<u64>)> = sources.to_vec();
        ordered.sort_unstable();
        ordered.dedup();
        let feeds: Vec<((u32, Option<u64>), Option<Arc<ChangeDataFeed>>)> = ordered
            .iter()
            .map(|source| (*source, self.feed_on(source.0, source.1)))
            .collect();
        let mut guards: Vec<((u32, Option<u64>), parking_lot::MutexGuard<'_, CdfInner>)> =
            Vec::new();
        for (source, feed) in &feeds {
            if let Some(feed) = feed {
                guards.push((*source, feed.inner.lock()));
            }
        }
        let mut out = Vec::with_capacity(ordered.len());
        for source in &ordered {
            let boundary = match guards.iter_mut().find(|(key, _)| key == source) {
                Some((_, inner)) => inner.boundary(ended),
                None => FeedBoundary {
                    latest: 0,
                    records: 0,
                    first_open: None,
                },
            };
            out.push((*source, boundary));
        }
        out
    }

    /// The lowest schema epoch any feed of a table, a branch's included,
    /// still holds a record written under. None when no feed of the table
    /// holds a record. A branch feed left on disk and not opened in this
    /// process is opened to be asked, because its records decode through
    /// the table's layouts the same as the table's own
    pub fn oldest_schema_epoch(&self, table_id: u32) -> Option<u32> {
        let mut oldest = self
            .get_feed(table_id)
            .and_then(|feed| feed.oldest_schema_epoch());
        let mut branches = Vec::new();
        self.branch_feeds.iter_sync(|key, feed| {
            if key.0 == table_id {
                branches.push(Arc::clone(feed));
            }
            true
        });
        let on_disk = ChangeDataFeed::table_branch_dirs(&self.data_dir, table_id);
        for branch_id in on_disk {
            if self.branch_feed_open(table_id, branch_id).is_none()
                && let Some(feed) = self.branch_feed(table_id, branch_id)
            {
                branches.push(feed);
            }
        }
        for feed in branches {
            oldest = match (oldest, feed.oldest_schema_epoch()) {
                (Some(a), Some(b)) => Some(a.min(b)),
                (a, b) => a.or(b),
            };
        }
        oldest
    }

    /// The feed a source names, the table's own, or a branch's on it
    pub fn feed_on(&self, table_id: u32, branch: Option<u64>) -> Option<Arc<ChangeDataFeed>> {
        match branch {
            Some(branch) => self.branch_feed(table_id, branch),
            None => self.get_feed(table_id),
        }
    }

    /// One version a bounded read of several feeds ends at, None when no
    /// feed holds more than `max_rows` records past its position.
    ///
    /// Each source is `(table id, the version its read starts after, the
    /// records consumed so far)`. The cut is the lowest of the feeds' own
    /// cuts, then moved past every transaction any of the feeds saw write
    /// inside the read and again beyond it, until no feed moves it, so a
    /// transaction that touched two sources is wholly inside the read or
    /// wholly after it. Read under every feed's lock, in table id order
    pub fn bounded_cut(
        &self,
        sources: &[((u32, Option<u64>), u64, u64)],
        max_rows: u64,
    ) -> Option<u64> {
        let mut ordered: Vec<((u32, Option<u64>), u64, u64)> = sources.to_vec();
        ordered.sort_unstable_by_key(|(source, _, _)| *source);
        ordered.dedup_by_key(|(source, _, _)| *source);
        let feeds: Vec<((u32, Option<u64>), Arc<ChangeDataFeed>)> = ordered
            .iter()
            .filter_map(|(source, _, _)| {
                self.feed_on(source.0, source.1).map(|feed| (*source, feed))
            })
            .collect();
        let guards: Vec<(u64, u64, parking_lot::MutexGuard<'_, CdfInner>)> = ordered
            .iter()
            .filter_map(|(source, from, consumed)| {
                feeds
                    .iter()
                    .find(|(key, _)| key == source)
                    .map(|(_, feed)| (*from, *consumed, feed.inner.lock()))
            })
            .collect();
        let mut cut = guards
            .iter()
            .filter_map(|(from, consumed, inner)| inner.bounded_cut(*from, *consumed, max_rows))
            .min()?;
        loop {
            let moved = guards
                .iter()
                .map(|(from, _, inner)| inner.extend_cut(*from, cut))
                .max()
                .unwrap_or(cut);
            if moved == cut {
                return Some(cut);
            }
            cut = moved;
        }
    }

    /// The version each window ends at once no transaction is handed over
    /// by halves, in the order the windows were given.
    ///
    /// The heap feeds share the node's change clock, so their windows end
    /// at one version, and a derived source's window ends at a version of
    /// its own. A transaction with a commit inside any window pulls every
    /// window up to its last commit in that source, until no window moves.
    /// A transaction that cannot be held whole, because one of its commits
    /// lies above a source's limit, is left for a later read instead, and
    /// every window drops below its first commit in that source, which
    /// can leave other transactions unable to be held whole, so the two
    /// steps alternate until they agree. A window never ends below the
    /// version its read starts after. Read under every heap feed's lock,
    /// in table id order, the same order every other reader takes them in
    pub fn align_windows(&self, windows: &[SourceRead]) -> Vec<u64> {
        enum Source {
            Heap(usize),
            Derived(Arc<dyn DerivedChangeSource>),
            None,
        }
        let mut keys: Vec<(u32, Option<u64>)> = windows
            .iter()
            .filter(|w| self.derived_on(w.table_id, w.branch).is_none())
            .map(|w| (w.table_id, w.branch))
            .collect();
        keys.sort_unstable();
        keys.dedup();
        let feeds: Vec<((u32, Option<u64>), Arc<ChangeDataFeed>)> = keys
            .iter()
            .filter_map(|key| self.feed_on(key.0, key.1).map(|feed| (*key, feed)))
            .collect();
        let guards: Vec<((u32, Option<u64>), parking_lot::MutexGuard<'_, CdfInner>)> = feeds
            .iter()
            .map(|(key, feed)| (*key, feed.inner.lock()))
            .collect();
        let sources: Vec<Source> = windows
            .iter()
            .map(|w| match self.derived_on(w.table_id, w.branch) {
                Some(source) => Source::Derived(source),
                None => guards
                    .iter()
                    .position(|(key, _)| *key == (w.table_id, w.branch))
                    .map(Source::Heap)
                    .unwrap_or(Source::None),
            })
            .collect();
        let mut to: Vec<u64> = windows
            .iter()
            .map(|w| w.to_inclusive.min(w.limit))
            .collect();
        let mut limit: Vec<u64> = windows.iter().map(|w| w.limit).collect();
        // The heap windows end at one version, the lowest they resolved
        let heap_to = (0..windows.len())
            .filter(|i| matches!(sources[*i], Source::Heap(_)))
            .map(|i| to[i])
            .min();
        if let Some(shared) = heap_to {
            for (i, source) in sources.iter().enumerate() {
                if matches!(source, Source::Heap(_)) {
                    to[i] = shared;
                }
            }
        }
        // What every source needs to hold a transaction whole, the last
        // version the transaction committed to it at, None where it has no
        // commit. The heap feeds answer as one, at the highest any of them
        // needs
        let need_of = |txn: u64| -> Vec<Option<u64>> {
            let heap_need = guards
                .iter()
                .filter_map(|(_, inner)| inner.span_of(txn).map(|(_, last)| last))
                .max();
            sources
                .iter()
                .map(|source| match source {
                    Source::Heap(_) => heap_need,
                    Source::Derived(derived) => derived.span_of(txn).map(|(_, last)| last),
                    Source::None => None,
                })
                .collect()
        };
        let mut excluded: HashSet<u64> = HashSet::new();
        loop {
            // Raise every window until the transactions inside are whole in
            // every source. Each pass scans only the versions the last pass
            // did not, so a chain of transactions costs one walk of the
            // windows it reaches over
            let mut scanned: Vec<u64> = windows.iter().map(|w| w.from_exclusive).collect();
            let mut processed: HashSet<u64> = HashSet::new();
            let mut newly_excluded = false;
            loop {
                let mut inside: Vec<u64> = Vec::new();
                for i in 0..windows.len() {
                    if to[i] <= scanned[i] {
                        continue;
                    }
                    match &sources[i] {
                        Source::Heap(at) => {
                            inside.extend(guards[*at].1.txns_opening_in(scanned[i], to[i]));
                        }
                        Source::Derived(derived) => {
                            inside.extend(derived.txns_in(scanned[i], to[i]));
                        }
                        Source::None => {}
                    }
                    scanned[i] = to[i];
                }
                inside.sort_unstable();
                inside.dedup();
                let mut moved = false;
                for txn in inside {
                    if excluded.contains(&txn) || !processed.insert(txn) {
                        continue;
                    }
                    let needs = need_of(txn);
                    let over = needs
                        .iter()
                        .zip(&limit)
                        .any(|(need, limit)| need.is_some_and(|need| need > *limit));
                    if over {
                        excluded.insert(txn);
                        newly_excluded = true;
                        continue;
                    }
                    for (i, need) in needs.into_iter().enumerate() {
                        if let Some(need) = need
                            && need > to[i]
                        {
                            to[i] = need;
                            moved = true;
                        }
                    }
                }
                // The heap windows move together
                if moved
                    && let Some(shared) = (0..windows.len())
                        .filter(|i| matches!(sources[*i], Source::Heap(_)))
                        .map(|i| to[i])
                        .max()
                {
                    for (i, source) in sources.iter().enumerate() {
                        if matches!(source, Source::Heap(_)) {
                            to[i] = shared;
                        }
                    }
                }
                if !moved {
                    break;
                }
            }
            if !newly_excluded {
                break;
            }
            // Every window drops below the first commit of each excluded
            // transaction in its source, and stays there for the passes
            // that follow, the heap windows together
            for txn in &excluded {
                let heap_first = guards
                    .iter()
                    .filter_map(|(_, inner)| inner.span_of(*txn).map(|(first, _)| first))
                    .min();
                for (i, source) in sources.iter().enumerate() {
                    let first = match source {
                        Source::Heap(_) => heap_first,
                        Source::Derived(derived) => derived.span_of(*txn).map(|(first, _)| first),
                        Source::None => None,
                    };
                    if let Some(first) = first {
                        limit[i] = limit[i].min(first.saturating_sub(1));
                    }
                }
            }
            if let Some(shared) = (0..windows.len())
                .filter(|i| matches!(sources[*i], Source::Heap(_)))
                .map(|i| limit[i])
                .min()
            {
                for (i, source) in sources.iter().enumerate() {
                    if matches!(source, Source::Heap(_)) {
                        limit[i] = shared;
                    }
                }
            }
            for i in 0..windows.len() {
                to[i] = to[i].min(limit[i]);
            }
        }
        for (i, w) in windows.iter().enumerate() {
            to[i] = to[i].max(w.from_exclusive);
        }
        to
    }

    pub fn data_dir(&self) -> &Path {
        &self.data_dir
    }

    pub fn enable_for_table(
        &self,
        table_id: u32,
        retention_days: u32,
    ) -> Result<Arc<ChangeDataFeed>> {
        self.enable_with_config(table_id, FeedConfig::from_retention_days(retention_days))
    }

    /// Opens the table's feed with a configuration.
    ///
    /// Enabling a feed on a table that already holds rows records no history
    /// for them. The first change is the first record. A stream created with
    /// SHOW INITIAL ROWS is how the existing rows are read
    pub fn enable_with_config(
        &self,
        table_id: u32,
        config: FeedConfig,
    ) -> Result<Arc<ChangeDataFeed>> {
        if let Some(existing) = self.get_feed(table_id) {
            existing.set_config(config)?;
            return Ok(existing);
        }
        let feed = self.adopt(Arc::new(ChangeDataFeed::open_with_config(
            &self.data_dir,
            table_id,
            config,
        )?));
        let _ = self.feeds.insert_sync(table_id, feed.clone());
        Ok(feed)
    }

    /// Stops a table's feed recording. With `purge` the feed and its files
    /// go, and without it the feed stays registered, disabled, with its
    /// counters on disk, so the one instance a directory has is the one a
    /// later enable reuses and a seal still finishing lands in it rather
    /// than beside a second instance opened on the same files
    pub fn disable_for_table(&self, table_id: u32, purge: bool) -> Result<()> {
        if purge {
            if let Some((_, feed)) = self.feeds.remove_sync(&table_id) {
                feed.disable();
                feed.remove_files()?;
            }
            return Ok(());
        }
        if let Some(feed) = self.get_feed(table_id) {
            feed.disable();
            feed.checkpoint()?;
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

    /// Drops everything the registry holds for a table that is gone, the
    /// feeds its branches recorded into, its own feed and the feed's files,
    /// the source its changes were derived from with the index that source
    /// kept, and the directory they stood in
    pub fn remove_table(&self, table_id: u32) -> Result<()> {
        let mut branches = Vec::new();
        self.branch_feeds.iter_sync(|(table, branch), feed| {
            if *table == table_id {
                branches.push((*branch, Arc::clone(feed)));
            }
            true
        });
        for (branch, feed) in branches {
            let _ = self.branch_feeds.remove_sync(&(table_id, branch));
            feed.disable();
            feed.remove_files()?;
        }
        self.disable_for_table(table_id, true)?;
        self.remove_derived(table_id);
        match std::fs::remove_dir_all(ChangeDataFeed::table_dir(&self.data_dir, table_id)) {
            Ok(()) => Ok(()),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(()),
            Err(e) => Err(e.into()),
        }
    }

    /// Every table that has a feed open on this node
    pub fn table_ids(&self) -> Vec<u32> {
        let mut ids = Vec::new();
        self.feeds.iter_sync(|table_id, _| {
            ids.push(*table_id);
            true
        });
        ids.sort_unstable();
        ids
    }

    // -------------------------------------------------------------------
    // LSN-based truncation for publication retention
    // -------------------------------------------------------------------

    /// Removes CDF records whose commit_version is at or below target_lsn.
    /// Returns the number of records removed. Returns 0 if the table has no
    /// feed registered or no records qualify for removal.
    pub async fn truncate_before(&self, table_id: u32, lsn: u64) -> Result<u64> {
        let feed = match self.get_feed(table_id) {
            Some(f) => f,
            None => return Ok(0),
        };
        feed.purge_before_version(lsn.saturating_add(1))
    }

    /// Time-based truncation for publication retention: removes records
    /// whose commit timestamp (microseconds) is older than the cutoff. The
    /// optional hold LSN keeps every record above it regardless of age, so a
    /// slow but advancing subscriber never loses changes it has not
    /// confirmed. Returns the number of records removed, 0 when the table
    /// has no feed registered
    pub async fn truncate_retention(
        &self,
        table_id: u32,
        cutoff_timestamp: i64,
        hold_lsn: Option<u64>,
    ) -> Result<u64> {
        let feed = match self.get_feed(table_id) {
            Some(f) => f,
            None => return Ok(0),
        };
        feed.purge_retention(cutoff_timestamp, hold_lsn)
    }

    /// Every feed open on this node, a table's own and the ones its
    /// branches record into, each under its table id
    fn open_feeds(&self) -> Vec<(u32, Arc<ChangeDataFeed>)> {
        let mut feeds: Vec<(u32, Arc<ChangeDataFeed>)> = Vec::new();
        self.feeds.iter_sync(|table_id, feed| {
            feeds.push((*table_id, feed.clone()));
            true
        });
        self.branch_feeds.iter_sync(|(table_id, _), feed| {
            feeds.push((*table_id, feed.clone()));
            true
        });
        feeds
    }

    /// Forces every feed's appended records to durable storage, a branch's
    /// feeds with the table's own. Returns the number of feeds that had
    /// pending bytes plus the per-table failures, so one bad feed never
    /// hides the rest
    pub fn sync_all_feeds(&self) -> (u64, Vec<(u32, ZyronError)>) {
        let mut synced = 0u64;
        let mut failures = Vec::new();
        for (table_id, feed) in self.open_feeds() {
            match feed.sync_to_disk() {
                Ok(true) => synced += 1,
                Ok(false) => {}
                Err(e) => failures.push((table_id, e)),
            }
        }
        (synced, failures)
    }

    /// Forgets the ended transactions every feed still counts as open, run
    /// on the writer's cadence so a feed nothing reads stays bounded.
    /// Answers with how many transactions are still open across the feeds
    pub fn prune_in_flight(&self, ended: &dyn Fn(u64) -> bool) -> usize {
        self.open_feeds()
            .into_iter()
            .map(|(_, feed)| feed.prune_in_flight(ended))
            .sum()
    }

    /// Table id, record count, bytes and retention days for every open feed
    pub fn list_feeds(&self) -> Vec<(u32, u64, u64, u32)> {
        let mut result = Vec::new();
        self.feeds.iter_sync(|table_id, feed| {
            result.push((
                *table_id,
                feed.record_count(),
                feed.file_size_bytes(),
                feed.retention_days(),
            ));
            true
        });
        result.sort_by_key(|row| row.0);
        result
    }
}

// ---------------------------------------------------------------------------
// Recovery of logged appends
// ---------------------------------------------------------------------------

/// What laying the logged appends back into the segment files did
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct FrameRestore {
    /// Records laid into a segment file
    pub laid: u64,
    /// Bytes those records carried
    pub bytes: u64,
    /// Records passed over, below a feed's replay point, for a segment
    /// already sealed or purged, or for a feed whose directory is gone
    pub passed: u64,
}

/// One feed's directory as replay sees it, read once per feed
struct FeedReplay {
    dir: PathBuf,
    /// The log position the manifest says replay starts from
    replay_from: u64,
    /// The open segment as the manifest last recorded it. A record for a
    /// segment below it names one that was sealed or purged since
    open_seq: u64,
}

/// Lays every logged change feed append back into its segment file, in log
/// order, before the feeds reopen.
///
/// A record below the feed's replay point is in a synced file already and
/// is passed over, as is one for a segment the manifest has moved past or
/// whose file is sealed, since a seal or a purge laid the segment out again
/// durably. Every other record cuts the file to the offset it names and
/// writes its bytes there, so the file ends up holding exactly what the
/// log says it did, whichever of its writes the device had taken. A file
/// shorter than a record's offset lost bytes the log proves it held, which
/// is refused rather than papered over. The feed's own open then walks the
/// frames, counts them and syncs the file
pub fn restore_logged_frames(
    data_dir: &Path,
    records: &[zyron_wal::LogRecord],
) -> Result<FrameRestore> {
    let mut restore = FrameRestore::default();
    let mut feeds: HashMap<(u32, u64), Option<FeedReplay>> = HashMap::new();
    // Each segment file the log names, None for one found sealed
    let mut files: HashMap<PathBuf, Option<File>> = HashMap::new();
    for record in records {
        let frames = zyron_wal::ChangeFeedFrames::decode(&record.payload)?;
        let key = (frames.table_id, frames.branch_id);
        let feed = match feeds.entry(key) {
            std::collections::hash_map::Entry::Occupied(entry) => entry.into_mut(),
            std::collections::hash_map::Entry::Vacant(entry) => {
                let dir = if frames.branch_id == 0 {
                    ChangeDataFeed::table_dir(data_dir, frames.table_id)
                } else {
                    ChangeDataFeed::branch_dir(data_dir, frames.table_id, frames.branch_id)
                };
                let replay = if dir.is_dir() {
                    let (replay_from, open_seq) = match ChangeDataFeed::read_manifest(&dir)? {
                        Some(inner) => (inner.replay_from, inner.open.seq),
                        None => (0, 1),
                    };
                    Some(FeedReplay {
                        dir,
                        replay_from,
                        open_seq,
                    })
                } else {
                    None
                };
                entry.insert(replay)
            }
        };
        let Some(feed) = feed else {
            restore.passed += 1;
            continue;
        };
        if record.lsn.0 < feed.replay_from || frames.seq < feed.open_seq {
            restore.passed += 1;
            continue;
        }
        let path = ChangeDataFeed::segment_path_in(&feed.dir, frames.seq);
        let file = match files.entry(path.clone()) {
            std::collections::hash_map::Entry::Occupied(entry) => entry.into_mut(),
            std::collections::hash_map::Entry::Vacant(entry) => {
                let mut file = OpenOptions::new()
                    .read(true)
                    .write(true)
                    .create(true)
                    .truncate(false)
                    .open(&path)?;
                let len = file.metadata()?.len();
                if len >= SEGMENT_BODY_OFFSET {
                    let mut header = vec![0u8; SEGMENT_BODY_OFFSET as usize];
                    file.read_exact(&mut header)?;
                    if read_segment_header(&header)?.sealed {
                        // Laid out again by its seal, durably, with every
                        // frame the log could put back already in it
                        entry.insert(None);
                        restore.passed += 1;
                        continue;
                    }
                } else {
                    // A segment the log names before any of it reached
                    // disk, given the header a rotation would have written
                    file.set_len(0)?;
                    write_segment_header(&mut file, frames.table_id, frames.seq, 0)?;
                }
                entry.insert(Some(file))
            }
        };
        let Some(file) = file else {
            restore.passed += 1;
            continue;
        };
        let len = file.metadata()?.len();
        if len < frames.offset {
            return Err(ZyronError::CdcStreamError(format!(
                "change segment {} holds {len} bytes but the log records an append at byte {}, \
                 so bytes the log proves it held are gone",
                path.display(),
                frames.offset
            )));
        }
        file.set_len(frames.offset)?;
        positional_write(file, frames.offset, frames.bytes)?;
        restore.laid += 1;
        restore.bytes += frames.bytes.len() as u64;
    }
    for file in files.into_values().flatten() {
        file.sync_all()?;
    }
    Ok(restore)
}

/// Writes `bytes` at `offset` of `file`
fn positional_write(file: &mut File, offset: u64, bytes: &[u8]) -> Result<()> {
    use std::io::Seek;
    file.seek(std::io::SeekFrom::Start(offset))?;
    file.write_all(bytes)?;
    Ok(())
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
            change_ordinal: 0,
            schema_version: 1,
            row_data: vec![1, 2, 3, 4],
            primary_key_data: vec![1],
            is_last_in_txn: true,
            projected: false,
        }
    }

    #[test]
    fn test_change_type_roundtrip() {
        for v in 0..=5u8 {
            let ct = ChangeType::from_u8(v).expect("a known change type code");
            assert_eq!(ct as u8, v);
            assert_eq!(ChangeType::from_label(ct.label()), Some(ct));
        }
        assert!(ChangeType::from_u8(99).is_err());
    }

    /// A log writer over a directory of its own, with the ring the writer
    /// requires
    fn open_wal(wal_dir: &Path) -> Arc<zyron_wal::WalWriter> {
        Arc::new(
            zyron_wal::WalWriter::new(zyron_wal::WalWriterConfig {
                wal_dir: wal_dir.to_path_buf(),
                ..zyron_wal::WalWriterConfig::default()
            })
            .expect("the log opens"),
        )
    }

    /// The logged appends of a closed log, in log order
    fn logged_appends(wal_dir: &Path) -> Vec<zyron_wal::LogRecord> {
        zyron_wal::RecoveryManager::new(wal_dir)
            .expect("the log reopens")
            .recover()
            .expect("the log recovers")
            .feed_records
    }

    /// An append is durable through the log. A stop that takes the open
    /// segment's unsynced tail with it leaves the log holding every byte,
    /// and recovery lays the bytes back before the feed reopens, so the
    /// records a committed transaction wrote are all there
    #[test]
    fn test_logged_appends_are_laid_back_into_a_segment_the_device_never_took() {
        let dir = tempfile::tempdir().expect("tempdir");
        let data_dir = dir.path().join("data");
        let wal_dir = dir.path().join("wal");
        fs::create_dir_all(&data_dir).expect("data dir");
        let wal = open_wal(&wal_dir);
        let registry = CdfRegistry::new(data_dir.clone());
        registry.attach_wal(&wal);
        let feed = registry
            .enable_with_config(7, FeedConfig::default())
            .expect("the feed opens");
        assert_eq!(registry.retained_lsn(), None, "nothing logged yet");

        let records: Vec<ChangeRecord> = (1..=5u64)
            .map(|version| {
                let mut record = make_record(version, version as i64 * 1000, ChangeType::Insert);
                record.row_data = vec![version as u8; 300];
                record
            })
            .collect();
        feed.append_batch(&records).expect("append");
        let held = registry
            .retained_lsn()
            .expect("the log holds the append until a sync pass");
        wal.flush().expect("the log flushes");
        let open_seq = feed.inner.lock().open.seq;
        let segment =
            ChangeDataFeed::segment_path_in(&ChangeDataFeed::table_dir(&data_dir, 7), open_seq);
        let full_len = fs::metadata(&segment).expect("segment").len();
        assert!(full_len > SEGMENT_BODY_OFFSET);

        // The stop, with the segment file cut back to its header, which is
        // what a device that never took the buffered bytes leaves
        drop(feed);
        drop(registry);
        wal.close().expect("the log closes");
        OpenOptions::new()
            .write(true)
            .open(&segment)
            .expect("segment")
            .set_len(SEGMENT_BODY_OFFSET)
            .expect("cut");

        let logged = logged_appends(&wal_dir);
        assert!(!logged.is_empty(), "the log holds the append");
        assert!(logged.iter().all(|r| r.lsn >= held));
        let restored = restore_logged_frames(&data_dir, &logged).expect("the bytes lay back");
        assert_eq!(restored.laid as usize, logged.len());
        assert_eq!(restored.passed, 0);
        assert_eq!(
            fs::metadata(&segment).expect("segment").len(),
            full_len,
            "the file holds every byte the log did"
        );

        let feed =
            ChangeDataFeed::open_with_config(&data_dir, 7, FeedConfig::default()).expect("reopen");
        assert_eq!(feed.record_count(), 5);
        let read = feed.read_range(&ChangeRange::everything()).expect("read");
        assert_eq!(read.len(), 5);
        for (i, record) in read.iter().enumerate() {
            assert_eq!(record.commit_version, i as u64 + 1);
            assert_eq!(record.row_data, records[i].row_data);
        }
    }

    /// A sync pass moves the point the log has to keep, and a manifest
    /// written after it records that every logged append is in a synced
    /// file, so recovery passes those records over and touches nothing
    #[test]
    fn test_a_synced_append_is_passed_over_by_recovery_and_releases_the_log() {
        let dir = tempfile::tempdir().expect("tempdir");
        let data_dir = dir.path().join("data");
        let wal_dir = dir.path().join("wal");
        fs::create_dir_all(&data_dir).expect("data dir");
        let wal = open_wal(&wal_dir);
        let registry = CdfRegistry::new(data_dir.clone());
        registry.attach_wal(&wal);
        let feed = registry
            .enable_with_config(7, FeedConfig::default())
            .expect("the feed opens");
        feed.append_batch(&[make_record(1, 1000, ChangeType::Insert)])
            .expect("append");
        assert!(registry.retained_lsn().is_some());
        let (synced, failures) = registry.sync_all_feeds();
        assert_eq!((synced, failures.len()), (1, 0));
        assert_eq!(
            registry.retained_lsn(),
            None,
            "a synced feed holds nothing in the log"
        );
        feed.checkpoint()
            .expect("the manifest records the replay point");
        let open_seq = feed.inner.lock().open.seq;
        let segment =
            ChangeDataFeed::segment_path_in(&ChangeDataFeed::table_dir(&data_dir, 7), open_seq);
        let before = fs::read(&segment).expect("segment");

        // A second append after the manifest, left unsynced, is what the
        // log has to put back. The first is not
        feed.append_batch(&[make_record(2, 2000, ChangeType::Insert)])
            .expect("append");
        wal.flush().expect("the log flushes");
        let after = fs::read(&segment).expect("segment");
        drop(feed);
        drop(registry);
        wal.close().expect("the log closes");
        fs::write(&segment, &before).expect("the second append's bytes never reached the device");

        let logged = logged_appends(&wal_dir);
        assert_eq!(logged.len(), 2);
        let restored = restore_logged_frames(&data_dir, &logged).expect("recovery");
        assert_eq!((restored.laid, restored.passed), (1, 1));
        assert_eq!(fs::read(&segment).expect("segment"), after);

        let feed =
            ChangeDataFeed::open_with_config(&data_dir, 7, FeedConfig::default()).expect("reopen");
        assert_eq!(feed.record_count(), 2);
    }

    /// A record for a segment that was sealed after the append is passed
    /// over, since the seal laid the segment down again durably, and a
    /// record for a feed whose directory is gone names nothing to restore
    #[test]
    fn test_recovery_leaves_a_sealed_segment_and_a_removed_feed_alone() {
        let dir = tempfile::tempdir().expect("tempdir");
        let data_dir = dir.path().join("data");
        let wal_dir = dir.path().join("wal");
        fs::create_dir_all(&data_dir).expect("data dir");
        let wal = open_wal(&wal_dir);
        let registry = CdfRegistry::new(data_dir.clone());
        registry.attach_wal(&wal);
        let feed = registry
            .enable_with_config(7, FeedConfig::default())
            .expect("the feed opens");
        feed.append_batch(&[make_record(1, 1000, ChangeType::Insert)])
            .expect("append");
        let sealed_seq = feed.inner.lock().open.seq;
        feed.seal_open_segment().expect("seal");
        let sealed =
            ChangeDataFeed::segment_path_in(&ChangeDataFeed::table_dir(&data_dir, 7), sealed_seq);
        let sealed_bytes = fs::read(&sealed).expect("sealed segment");
        let gone = registry
            .enable_with_config(8, FeedConfig::default())
            .expect("the second feed opens");
        gone.append_batch(&[make_record(1, 1000, ChangeType::Insert)])
            .expect("append");
        wal.flush().expect("the log flushes");
        drop(gone);
        registry
            .disable_for_table(8, true)
            .expect("the second feed and its files go");
        drop(feed);
        drop(registry);
        wal.close().expect("the log closes");

        let logged = logged_appends(&wal_dir);
        assert_eq!(logged.len(), 2);
        let restored = restore_logged_frames(&data_dir, &logged).expect("recovery");
        assert_eq!((restored.laid, restored.passed), (0, 2));
        assert_eq!(fs::read(&sealed).expect("sealed segment"), sealed_bytes);
        assert!(!ChangeDataFeed::table_dir(&data_dir, 8).exists());
    }

    /// A rotation closes the open segment and starts the next one before
    /// its sealer rewrites the manifest. A stop in between leaves the
    /// manifest naming the full segment as open, in its raw form, with the
    /// next one holding frames. Opening the feed walks both, seals the raw
    /// one and counts every record once
    #[test]
    fn test_a_rotation_the_sealer_never_finished_is_recovered_at_open() {
        let dir = tempfile::tempdir().expect("tempdir");
        let feed =
            ChangeDataFeed::open_with_config(dir.path(), 7, FeedConfig::default()).expect("open");
        for version in 1..=5u64 {
            feed.append_change(&make_record(
                version,
                version as i64 * 1000,
                ChangeType::Insert,
            ))
            .expect("append");
        }
        // The writer's half of a rotation, the sealer's half dropped
        {
            let mut inner = feed.inner.lock();
            let job = feed.close_open_locked(&mut inner).expect("rotate");
            drop(job);
        }
        for version in 6..=8u64 {
            feed.append_change(&make_record(
                version,
                version as i64 * 1000,
                ChangeType::Insert,
            ))
            .expect("append past the rotation");
        }
        assert_eq!(feed.record_count(), 8);
        drop(feed);

        let reopened =
            ChangeDataFeed::open_with_config(dir.path(), 7, FeedConfig::default()).expect("reopen");
        assert_eq!(reopened.record_count(), 8, "every record counted once");
        let records = reopened
            .read_range(&ChangeRange::everything())
            .expect("read everything");
        let versions: Vec<u64> = records.iter().map(|r| r.commit_version).collect();
        assert_eq!(versions, (1..=8).collect::<Vec<u64>>());
        {
            let inner = reopened.inner.lock();
            assert_eq!(inner.sealed.len(), 1);
            assert!(inner.sealed[0].sealed, "the raw segment was sealed at open");
            assert_eq!(inner.open.seq, 2);
        }
        let bytes = {
            let mut bytes = Vec::new();
            File::open(reopened.segment_path(1))
                .expect("segment file")
                .read_to_end(&mut bytes)
                .expect("read");
            bytes
        };
        assert!(read_segment_header(&bytes).expect("header").sealed);
    }

    /// A scan hands its range over a batch at a time across sealed and
    /// open segments, and one that continues from a position starts after
    /// it, skipping the segments that end before it
    /// A layout of one 64 bit and one text column, the rows encoded the
    /// way the heap encoder lays a tuple out
    fn two_column_layout() -> Vec<zyron_catalog::PhysicalColumn> {
        vec![
            zyron_catalog::PhysicalColumn {
                column_id: zyron_catalog::ColumnId(0),
                physical_type: zyron_common::TypeId::Int64,
                fractional_digits: None,
                ordinal: 0,
            },
            zyron_catalog::PhysicalColumn {
                column_id: zyron_catalog::ColumnId(1),
                physical_type: zyron_common::TypeId::Text,
                fractional_digits: None,
                ordinal: 1,
            },
        ]
    }

    fn two_column_row(id: i64, text: Option<&str>) -> Vec<u8> {
        let mut row = vec![0u8; 1];
        row.extend_from_slice(&id.to_le_bytes());
        match text {
            Some(text) => {
                row.extend_from_slice(&(text.len() as u32).to_le_bytes());
                row.extend_from_slice(text.as_bytes());
            }
            None => {
                row[0] |= 1 << 1;
                row.extend_from_slice(&0u32.to_le_bytes());
            }
        }
        row
    }

    #[test]
    fn test_rows_written_under_a_known_layout_seal_into_columns_and_read_back_whole() {
        let dir = tempfile::tempdir().expect("tempdir");
        let feed =
            ChangeDataFeed::open_with_config(dir.path(), 9, FeedConfig::default()).expect("open");
        let layout = two_column_layout();
        let rows: Vec<Vec<u8>> = (1..=6i64)
            .map(|id| two_column_row(id, (id % 3 != 0).then_some("text")))
            .collect();
        for (i, row) in rows.iter().enumerate() {
            let header = ChangeHeader {
                commit_version: i as u64 + 1,
                commit_timestamp: (i as i64 + 1) * 1000,
                txn_id: 4,
                schema_version: 1,
            };
            feed.append_rows(
                &header,
                RowLayout {
                    columns: Some(&layout),
                    projected: false,
                },
                &[RowChange::of(ChangeType::Insert, row, true)],
            )
            .expect("append");
        }
        // A truncate, which carries no row, seals into the same file
        feed.append_change(&ChangeRecord {
            change_type: ChangeType::Truncate,
            commit_version: 7,
            commit_timestamp: 7000,
            table_id: 9,
            txn_id: 5,
            change_ordinal: 0,
            schema_version: 1,
            row_data: Vec::new(),
            primary_key_data: Vec::new(),
            is_last_in_txn: true,
            projected: false,
        })
        .expect("append");
        let seq = feed.inner.lock().open.seq;
        feed.seal_open_segment().expect("seal");
        assert_eq!(
            feed.segment_form(seq).expect("form"),
            Some(SegmentForm::SealedColumns),
            "every row's layout was known"
        );

        // The frames read back as they were written
        let records = feed.read_range(&ChangeRange::everything()).expect("read");
        assert_eq!(records.len(), 7);
        for (i, row) in rows.iter().enumerate() {
            assert_eq!(&records[i].row_data, row);
            assert_eq!(records[i].commit_version, i as u64 + 1);
        }
        assert_eq!(records[6].change_type, ChangeType::Truncate);
        assert!(records[6].row_data.is_empty());

        // The scan's part hands the file over as columns
        let mut scan = feed.open_scan(&ChangeRange::everything(), None);
        let mut parts = scan.take_parts();
        assert_eq!(parts.len(), 1);
        let sliced = parts[0]
            .columns()
            .expect("columns")
            .expect("the sealed file is sliced");
        assert_eq!(sliced.heads().len(), 7);
        assert_eq!(sliced.groups().len(), 1);
        assert_eq!(sliced.groups()[0].rows, 6);
        let text = sliced.column(0, 1).expect("the text column");
        assert_eq!(text.cell(0), b"text");
        assert!(text.is_null(2), "the third row's text is NULL");
        assert!(!sliced.heads().has_row(6), "the truncate has no row");

        // Reopening the feed keeps the layout, so the next seal slices too
        drop(feed);
        let feed =
            ChangeDataFeed::open_with_config(dir.path(), 9, FeedConfig::default()).expect("open");
        assert_eq!(feed.inner.lock().layouts.len(), 1);
        let header = ChangeHeader {
            commit_version: 8,
            commit_timestamp: 8000,
            txn_id: 6,
            schema_version: 1,
        };
        feed.append_rows(
            &header,
            RowLayout {
                columns: Some(&layout),
                projected: false,
            },
            &[RowChange::of(ChangeType::Insert, &rows[0], true)],
        )
        .expect("append");
        let seq = feed.inner.lock().open.seq;
        feed.seal_open_segment().expect("seal");
        assert_eq!(
            feed.segment_form(seq).expect("form"),
            Some(SegmentForm::SealedColumns)
        );

        // A purge that rewrites a sliced segment keeps it sliced and whole
        feed.purge_before_version(2).expect("purge");
        let records = feed.read_range(&ChangeRange::everything()).expect("read");
        assert_eq!(records.len(), 7, "the first row went, the rest stayed");
        assert_eq!(&records[0].row_data, &rows[1]);
        let first = feed.inner.lock().sealed[0].seq;
        assert_eq!(
            feed.segment_form(first).expect("form"),
            Some(SegmentForm::SealedColumns)
        );
    }

    #[test]
    fn test_rows_of_an_unknown_layout_seal_as_frames() {
        let dir = tempfile::tempdir().expect("tempdir");
        let feed =
            ChangeDataFeed::open_with_config(dir.path(), 9, FeedConfig::default()).expect("open");
        feed.append_change(&make_record(1, 1000, ChangeType::Insert))
            .expect("append");
        let seq = feed.inner.lock().open.seq;
        feed.seal_open_segment().expect("seal");
        assert_eq!(
            feed.segment_form(seq).expect("form"),
            Some(SegmentForm::SealedFrames)
        );
        let mut scan = feed.open_scan(&ChangeRange::everything(), None);
        let mut parts = scan.take_parts();
        assert!(parts[0].columns().expect("columns").is_none());
    }

    #[test]
    fn test_a_scan_keeps_its_place_across_segments_and_resumes_after_a_position() {
        let dir = tempfile::tempdir().expect("tempdir");
        let feed =
            ChangeDataFeed::open_with_config(dir.path(), 9, FeedConfig::default()).expect("open");
        for version in 1..=12u64 {
            feed.append_change(&make_record(
                version,
                version as i64 * 1000,
                ChangeType::Insert,
            ))
            .expect("append");
            if version % 4 == 0 && version < 12 {
                feed.seal_open_segment().expect("seal");
            }
        }
        // Three sealed segments of four and an open one of four, read in
        // batches of five
        let mut scan = feed.open_scan(&ChangeRange::everything(), None);
        assert_eq!(scan.plan().segments.len(), 3);
        let mut seen: Vec<u64> = Vec::new();
        let mut batches = 0;
        loop {
            let mut in_batch = 0;
            let more = scan
                .next(&mut |record| {
                    seen.push(record.commit_version);
                    in_batch += 1;
                    Ok(in_batch < 5)
                })
                .expect("scan");
            batches += 1;
            if !more {
                break;
            }
        }
        assert_eq!(seen, (1..=12).collect::<Vec<u64>>());
        assert_eq!(
            batches, 3,
            "twelve records in batches of five, the last short"
        );

        // Resuming after version 7 skips the first segment outright and the
        // records of the second at or before the position
        let mut scan = feed.open_scan(&ChangeRange::everything(), Some((7, 0)));
        assert_eq!(
            scan.plan().segments.len(),
            2,
            "the segment ending at 4 is skipped"
        );
        let mut seen: Vec<u64> = Vec::new();
        while scan
            .next(&mut |record| {
                seen.push(record.commit_version);
                Ok(true)
            })
            .expect("scan")
        {}
        assert_eq!(seen, (8..=12).collect::<Vec<u64>>());
    }

    #[test]
    fn test_change_record_serde() {
        let record = make_record(1, 1000, ChangeType::Insert);
        let mut buf = Vec::new();
        frame_into(&mut buf, &record, 7);
        // The frame is the length prefix, the body and the checksum
        let body = &buf[RECORD_FRAME_PREFIX..buf.len() - RECORD_FRAME_SUFFIX];
        let decoded = ChangeRecord::deserialize(body, 1).expect("round trips");
        assert_eq!(decoded.commit_version, 1);
        assert_eq!(decoded.commit_timestamp, 1000);
        assert_eq!(decoded.change_ordinal, 7);
        assert_eq!(decoded.change_type, ChangeType::Insert);
        assert_eq!(decoded.row_data, vec![1, 2, 3, 4]);
        assert_eq!(decoded.table_id, 1);
        assert!(decoded.is_last_in_txn);
    }

    #[test]
    fn test_segment_header_roundtrip() {
        let mut buf = Vec::new();
        write_segment_header(&mut buf, 42, 3, FLAG_SEALED).expect("writes");
        let header = read_segment_header(&buf).expect("reads");
        assert_eq!(header.table_id, 42);
        assert!(header.sealed);
    }

    #[test]
    fn test_segment_header_corruption_detected() {
        let mut buf = Vec::new();
        write_segment_header(&mut buf, 42, 1, 0).expect("writes");
        buf[21] ^= 0xFF;
        assert!(read_segment_header(&buf).is_err());
    }

    #[test]
    fn test_open_and_append() {
        let tmp = TempDir::new().expect("temp dir");
        let feed = ChangeDataFeed::open(tmp.path(), 1, 30).expect("opens");

        feed.append_change(&make_record(1, 1000, ChangeType::Insert))
            .expect("appends");
        assert_eq!(feed.record_count(), 1);
        assert!(feed.file_size_bytes() > SEGMENT_BODY_OFFSET);

        let results = feed.query_changes(0, 10).expect("queries");
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].commit_version, 1);
        assert_eq!(results[0].table_id, 1);
    }

    #[test]
    fn test_append_batch_numbers_ordinals_within_a_commit() {
        let tmp = TempDir::new().expect("temp dir");
        let feed = ChangeDataFeed::open(tmp.path(), 1, 30).expect("opens");

        let records = vec![
            make_record(7, 1000, ChangeType::UpdatePreimage),
            make_record(7, 1000, ChangeType::UpdatePostimage),
            make_record(8, 2000, ChangeType::Insert),
        ];
        feed.append_batch(&records).expect("appends");

        let read = feed.query_changes(0, 100).expect("queries");
        assert_eq!(read.len(), 3);
        assert_eq!(read[0].change_ordinal, 0);
        assert_eq!(read[1].change_ordinal, 1);
        assert_eq!(read[2].change_ordinal, 0);
    }

    #[test]
    fn test_before_image_off_drops_the_preimage() {
        let tmp = TempDir::new().expect("temp dir");
        let mut config = FeedConfig::default();
        config.before_image = false;
        let feed = ChangeDataFeed::open_with_config(tmp.path(), 1, config).expect("opens");

        feed.append_batch(&[
            make_record(1, 100, ChangeType::UpdatePreimage),
            make_record(1, 100, ChangeType::UpdatePostimage),
        ])
        .expect("appends");

        let read = feed.query_changes(0, 100).expect("queries");
        assert_eq!(read.len(), 1);
        assert_eq!(read[0].change_type, ChangeType::UpdatePostimage);
    }

    #[test]
    fn test_query_by_time() {
        let tmp = TempDir::new().expect("temp dir");
        let feed = ChangeDataFeed::open(tmp.path(), 1, 30).expect("opens");

        let records: Vec<ChangeRecord> = (1..=5)
            .map(|i| make_record(i, i as i64 * 1000, ChangeType::Insert))
            .collect();
        feed.append_batch(&records).expect("appends");

        let results = feed.query_changes_by_time(2000, 4000).expect("queries");
        assert_eq!(results.len(), 3);
    }

    #[test]
    fn test_crash_recovery_truncates_torn_write() {
        let tmp = TempDir::new().expect("temp dir");
        {
            let feed = ChangeDataFeed::open(tmp.path(), 1, 30).expect("opens");
            feed.append_change(&make_record(1, 1000, ChangeType::Insert))
                .expect("appends");
        }

        let path = tmp
            .path()
            .join("cdf")
            .join("00000001")
            .join("000000000001.zycdf");
        {
            let mut f = OpenOptions::new().append(true).open(&path).expect("opens");
            f.write_all(&[0xFF; 20]).expect("writes garbage");
        }

        let feed = ChangeDataFeed::open(tmp.path(), 1, 30).expect("reopens");
        assert_eq!(feed.record_count(), 1);
        assert_eq!(feed.query_changes(0, 10).expect("queries").len(), 1);
    }

    #[test]
    fn test_purge_before_version() {
        let tmp = TempDir::new().expect("temp dir");
        let feed = ChangeDataFeed::open(tmp.path(), 1, 30).expect("opens");

        let records: Vec<ChangeRecord> = (1..=10)
            .map(|i| make_record(i, i as i64 * 1000, ChangeType::Insert))
            .collect();
        feed.append_batch(&records).expect("appends");
        assert_eq!(feed.record_count(), 10);

        let purged = feed.purge_before_version(6).expect("purges");
        assert_eq!(purged, 5);
        assert_eq!(feed.record_count(), 5);

        let results = feed.query_changes(1, 10).expect("queries");
        assert_eq!(results.len(), 5);
        assert_eq!(results[0].commit_version, 6);
    }

    #[test]
    fn test_sealing_prunes_segments_a_range_excludes() {
        let tmp = TempDir::new().expect("temp dir");
        let feed = ChangeDataFeed::open(tmp.path(), 1, 30).expect("opens");

        for group in 0..5u64 {
            let records: Vec<ChangeRecord> = (0..4)
                .map(|i| {
                    let v = group * 10 + i;
                    make_record(v + 1, (v as i64 + 1) * 1000, ChangeType::Insert)
                })
                .collect();
            feed.append_batch(&records).expect("appends");
            feed.seal_open_segment().expect("seals");
        }

        let plan = feed.plan_read(&ChangeRange::versions(1, 4));
        assert_eq!(plan.segments.len(), 1, "one segment holds versions 1 to 4");
        assert_eq!(plan.pruned, 4);

        let rows = feed.query_changes(1, 4).expect("queries");
        assert_eq!(rows.len(), 4);
    }

    #[test]
    fn test_sealed_segments_round_trip_under_every_codec() {
        for codec in [CdfCodec::None, CdfCodec::Lz4, CdfCodec::Zstd] {
            let tmp = TempDir::new().expect("temp dir");
            let config = FeedConfig {
                codec,
                ..FeedConfig::default()
            };
            let feed = ChangeDataFeed::open_with_config(tmp.path(), 9, config).expect("opens");
            let records: Vec<ChangeRecord> = (1..=64)
                .map(|i| make_record(i, i as i64 * 10, ChangeType::Insert))
                .collect();
            feed.append_batch(&records).expect("appends");
            feed.seal_open_segment().expect("seals");
            let read = feed.query_changes(0, u64::MAX).expect("queries");
            assert_eq!(read.len(), 64, "{} lost records", codec.name());
            assert_eq!(read[63].commit_version, 64);
        }
    }

    #[test]
    fn test_the_borrowed_scan_sees_what_the_owning_read_sees() {
        let tmp = TempDir::new().expect("temp dir");
        let feed = ChangeDataFeed::open(tmp.path(), 12, 30).expect("opens");
        for group in 0..4u64 {
            let records: Vec<ChangeRecord> = (0..16)
                .map(|i| {
                    let v = group * 16 + i + 1;
                    make_record(v, v as i64, ChangeType::Insert)
                })
                .collect();
            feed.append_batch(&records).expect("appends");
            feed.seal_open_segment().expect("seals");
        }
        feed.append_batch(&[make_record(200, 200, ChangeType::Delete)])
            .expect("appends");

        let range = ChangeRange::versions(10, 200);
        let owned = feed.read_range(&range).expect("reads");
        let mut seen: Vec<(u64, u64, Vec<u8>)> = Vec::new();
        feed.scan_range(&range, |view| {
            seen.push((
                view.commit_version,
                view.change_ordinal,
                view.row_data.to_vec(),
            ));
            Ok(())
        })
        .expect("scans");

        assert_eq!(owned.len(), seen.len());
        for (record, (version, ordinal, row)) in owned.iter().zip(seen.iter()) {
            assert_eq!(record.commit_version, *version);
            assert_eq!(record.change_ordinal, *ordinal);
            assert_eq!(&record.row_data, row);
        }
        // Segments are walked in sequence order, which is already the order a
        // consumer reads in, so the scan does no sorting of its own
        assert!(seen.windows(2).all(|pair| pair[0].0 <= pair[1].0));
    }

    #[test]
    fn test_a_replicated_count_names_the_same_place_here() {
        let tmp = TempDir::new().expect("temp dir");
        let feed = ChangeDataFeed::open(tmp.path(), 13, 30).expect("opens");
        for v in [3u64, 7, 11, 19, 23] {
            feed.append_batch(&[
                make_record(v, v as i64, ChangeType::Insert),
                make_record(v, v as i64, ChangeType::Insert),
            ])
            .expect("appends");
        }
        for version in [3u64, 7, 11, 19, 23] {
            let count = feed.records_at_or_below(version);
            assert_eq!(
                feed.version_at_count(count),
                version,
                "count {count} did not name version {version}"
            );
        }
        assert_eq!(feed.records_at_or_below(23), 10);
        assert_eq!(feed.version_at_count(0), 0);
    }

    #[test]
    fn test_pending_count_comes_from_the_counters() {
        let tmp = TempDir::new().expect("temp dir");
        let feed = ChangeDataFeed::open(tmp.path(), 1, 30).expect("opens");
        let records: Vec<ChangeRecord> = (1..=100)
            .map(|i| make_record(i, i as i64, ChangeType::Insert))
            .collect();
        feed.append_batch(&records).expect("appends");

        assert_eq!(feed.pending_after(0), 100);
        assert_eq!(feed.pending_after(40), 60);
        assert_eq!(feed.pending_after(100), 0);
        assert_eq!(feed.pending_versions_after(40), 60);
        assert_eq!(feed.first_timestamp_after(40), Some(41));
    }

    #[test]
    fn test_a_purged_position_still_counts_what_is_left() {
        let tmp = TempDir::new().expect("temp dir");
        let feed = ChangeDataFeed::open(tmp.path(), 1, 30).expect("opens");
        let records: Vec<ChangeRecord> = (1..=100)
            .map(|i| make_record(i, i as i64, ChangeType::Insert))
            .collect();
        feed.append_batch(&records).expect("appends");
        feed.purge_before_version(50).expect("purges");

        // A position below the oldest retained version still reports the
        // records that remain rather than the records that ever existed
        assert_eq!(feed.pending_after(10), 51);
        assert_eq!(feed.pending_after(60), 40);
    }

    #[test]
    fn test_reopening_recovers_the_open_segment() {
        let tmp = TempDir::new().expect("temp dir");
        {
            let feed = ChangeDataFeed::open(tmp.path(), 3, 30).expect("opens");
            let records: Vec<ChangeRecord> = (1..=20)
                .map(|i| make_record(i, i as i64 * 5, ChangeType::Insert))
                .collect();
            feed.append_batch(&records).expect("appends");
        }
        let feed = ChangeDataFeed::open(tmp.path(), 3, 30).expect("reopens");
        assert_eq!(feed.record_count(), 20);
        assert_eq!(feed.latest_version(), Some(20));
        assert_eq!(feed.pending_after(10), 10);
    }

    /// A manifest written mid segment counts records through its mark and
    /// no further, so a reopen after more records landed counts each once,
    /// including the records of a version that straddles the mark
    #[test]
    fn test_records_landed_after_a_manifest_write_are_counted_once_at_reopen() {
        let tmp = TempDir::new().expect("temp dir");
        let before = {
            let feed = ChangeDataFeed::open(tmp.path(), 4, 30).expect("opens");
            for version in 1..=3u64 {
                feed.append_change(&make_record(version, version as i64, ChangeType::Insert))
                    .expect("appends");
            }
            // The manifest lands with the counts through here
            feed.checkpoint().expect("checkpoints");
            for version in 3..=6u64 {
                feed.append_change(&make_record(version, version as i64, ChangeType::Insert))
                    .expect("appends past the manifest");
            }
            let before = (
                feed.record_count(),
                feed.records_at_or_below(3),
                feed.records_at_or_below(6),
                feed.cursor_at_count(4),
                feed.cursor_at_count(7),
            );
            assert_eq!(before.0, 7);
            before
        };
        let feed = ChangeDataFeed::open(tmp.path(), 4, 30).expect("reopens");
        let after = (
            feed.record_count(),
            feed.records_at_or_below(3),
            feed.records_at_or_below(6),
            feed.cursor_at_count(4),
            feed.cursor_at_count(7),
        );
        assert_eq!(
            after, before,
            "every count reads as it did before the reopen"
        );
        let versions: Vec<u64> = feed
            .read_range(&ChangeRange::everything())
            .expect("reads everything")
            .iter()
            .map(|r| r.commit_version)
            .collect();
        assert_eq!(versions, vec![1, 2, 3, 3, 4, 5, 6]);
    }

    /// A purge writes the manifest with the open segment's records counted,
    /// and a reopen counts none of them again
    #[test]
    fn test_a_purge_then_a_reopen_keeps_every_count() {
        let tmp = TempDir::new().expect("temp dir");
        let before = {
            let feed = ChangeDataFeed::open(tmp.path(), 5, 30).expect("opens");
            for version in 1..=5u64 {
                feed.append_change(&make_record(version, version as i64, ChangeType::Insert))
                    .expect("appends");
            }
            feed.purge_before_version(3).expect("purges");
            (
                feed.record_count(),
                feed.records_at_or_below(5),
                feed.cursor_at_count(5),
                feed.pending_after(3),
            )
        };
        let feed = ChangeDataFeed::open(tmp.path(), 5, 30).expect("reopens");
        assert_eq!(
            (
                feed.record_count(),
                feed.records_at_or_below(5),
                feed.cursor_at_count(5),
                feed.pending_after(3),
            ),
            before
        );
        assert_eq!(feed.record_count(), 3);
    }

    /// A seal that installed and a manifest that did not follow it leave
    /// the manifest naming the sealed segment as open. The records the
    /// manifest's counts stop short of are counted from the sealed frames
    #[test]
    fn test_a_sealed_segment_the_manifest_still_names_open_is_counted_from_its_frames() {
        let tmp = TempDir::new().expect("temp dir");
        let before = {
            let feed = ChangeDataFeed::open(tmp.path(), 6, 30).expect("opens");
            for version in 1..=2u64 {
                feed.append_change(&make_record(version, version as i64, ChangeType::Insert))
                    .expect("appends");
            }
            feed.checkpoint().expect("checkpoints");
            for version in 3..=4u64 {
                feed.append_change(&make_record(version, version as i64, ChangeType::Insert))
                    .expect("appends past the manifest");
            }
            let before = (feed.record_count(), feed.cursor_at_count(4));
            // The seal, its file in place, its manifest write and the next
            // file's preparation dropped
            {
                let mut inner = feed.inner.lock();
                let job = feed.close_open_locked(&mut inner).expect("rotates");
                let (sealed, tmp_path) = seal_segment_files(
                    &job.dir,
                    job.table_id,
                    &job.summary,
                    &job.frames,
                    &job.layouts,
                )
                .expect("seals");
                fs::rename(&tmp_path, feed.segment_path(job.summary.seq)).expect("renames");
                drop(sealed);
                drop(job);
            }
            // The manifest on disk still names segment 1 as open with two
            // records counted
            before
        };
        let feed = ChangeDataFeed::open(tmp.path(), 6, 30).expect("reopens");
        assert_eq!((feed.record_count(), feed.cursor_at_count(4)), before);
        assert_eq!(feed.record_count(), 4);
        let inner = feed.inner.lock();
        assert_eq!(inner.sealed.len(), 1);
        assert!(inner.sealed[0].sealed);
        assert_eq!(inner.open.seq, 2);
    }

    #[test]
    fn test_byte_cap_purges_oldest_first() {
        let tmp = TempDir::new().expect("temp dir");
        let feed = ChangeDataFeed::open(tmp.path(), 4, 30).expect("opens");
        for group in 0..4u64 {
            let records: Vec<ChangeRecord> = (0..8)
                .map(|i| {
                    let v = group * 8 + i + 1;
                    make_record(v, v as i64, ChangeType::Insert)
                })
                .collect();
            feed.append_batch(&records).expect("appends");
            feed.seal_open_segment().expect("seals");
        }
        let before = feed.file_size_bytes();
        let removed = feed.enforce_byte_cap(before / 2).expect("enforces");
        assert!(removed > 0, "a cap below the feed size reclaims segments");
        assert!(feed.file_size_bytes() < before);
        // The table keeps accepting writes after the cap reclaimed
        feed.append_change(&make_record(999, 999, ChangeType::Insert))
            .expect("appends after the cap");
    }

    /// A compaction rewrites each sealed segment in place with the records
    /// it keeps, leaves the open segment alone, keeps the version index as
    /// it was so a position counted before the pass names the same place
    /// after it, and refuses a plan the feed moved on from
    #[test]
    fn test_compaction_rewrites_the_sealed_segments_in_place() {
        let tmp = TempDir::new().expect("temp dir");
        let feed = ChangeDataFeed::open(tmp.path(), 5, 30).expect("opens");
        let records: Vec<ChangeRecord> = (1..=30)
            .map(|i| make_record(i, i as i64, ChangeType::Insert))
            .collect();
        feed.append_batch(&records).expect("appends");
        feed.seal_open_segment().expect("seals");
        let open: Vec<ChangeRecord> = (31..=35)
            .map(|i| make_record(i, i as i64, ChangeType::Insert))
            .collect();
        feed.append_batch(&open)
            .expect("appends into the open segment");

        let plan = feed.plan_compaction();
        assert_eq!(plan.record_count, 30, "the open segment is not planned");
        let mut seen = 0u64;
        assert!(
            feed.visit_planned(&plan, &mut |_| {
                seen += 1;
                Ok(())
            })
            .expect("walks")
        );
        assert_eq!(seen, 30);
        let removed = feed
            .compact_planned(&plan, &mut |record| record.commit_version % 2 == 0)
            .expect("compacts");
        assert_eq!(removed, Some(15));
        assert_eq!(feed.record_count(), 20);
        assert_eq!(feed.query_changes(0, u64::MAX).expect("queries").len(), 20);
        assert_eq!(
            feed.records_at_or_below(30),
            30,
            "the numbering a position counts in is unchanged"
        );
        assert_eq!(feed.cursor_at_count(30), Some((30, 0)));
        {
            let inner = feed.inner.lock();
            assert_eq!(inner.sealed.len(), 1, "the segment is rewritten in place");
            assert!(inner.sealed[0].sealed);
            assert_eq!(inner.sealed[0].record_count, 15);
            assert_eq!(inner.open.record_count, 5);
        }

        // A plan the feed moved on from since is refused
        let stale = feed.plan_compaction();
        feed.purge_before_version(4).expect("purges");
        assert_eq!(
            feed.compact_planned(&stale, &mut |_| true)
                .expect("refuses"),
            None
        );

        // What the pass left reads back the same after a reopen
        let count = feed.record_count();
        drop(feed);
        let feed = ChangeDataFeed::open(tmp.path(), 5, 30).expect("reopens");
        assert_eq!(feed.record_count(), count);
        assert_eq!(
            feed.query_changes(0, u64::MAX).expect("queries").len() as u64,
            count
        );
    }

    #[test]
    fn test_registry_enable_and_disable() {
        let tmp = TempDir::new().expect("temp dir");
        let registry = CdfRegistry::new(tmp.path().to_path_buf());
        let feed = registry.enable_for_table(11, 3).expect("enables");
        feed.append_change(&make_record(1, 1, ChangeType::Insert))
            .expect("appends");
        assert_eq!(registry.list_feeds().len(), 1);
        registry.disable_for_table(11, true).expect("disables");
        assert!(registry.get_feed(11).is_none());
    }

    /// The records the manifest file holds, one envelope after another
    fn manifest_records(dir: &Path) -> usize {
        let bytes = fs::read(ChangeDataFeed::manifest_path(dir)).expect("reads the manifest");
        let mut offset = 0;
        let mut records = 0;
        while offset < bytes.len() {
            let (header, extension) = envelope::decode_header(&bytes[offset..]).expect("header");
            let body_len =
                u32::from_le_bytes([extension[0], extension[1], extension[2], extension[3]])
                    as usize;
            offset += header.body_offset() + body_len + envelope::ENVELOPE_FOOTER_LEN;
            records += 1;
        }
        records
    }

    /// Each seal appends one record of what it changed, a reopen replays
    /// the chain to the same index, spans and summaries, and a record torn
    /// by a stop mid append is cut off with the segment it described
    /// counted from its file
    #[test]
    fn test_a_seal_appends_a_manifest_record_and_a_reopen_replays_the_chain() {
        let tmp = TempDir::new().expect("temp dir");
        let dir = ChangeDataFeed::table_dir(tmp.path(), 8);
        let before = {
            let feed = ChangeDataFeed::open(tmp.path(), 8, 30).expect("opens");
            for seal in 0..3u64 {
                let records: Vec<ChangeRecord> = (1..=4u64)
                    .map(|i| {
                        let version = seal * 4 + i;
                        let mut record = make_record(version, version as i64, ChangeType::Insert);
                        record.txn_id = 100 + seal;
                        record
                    })
                    .collect();
                feed.append_batch(&records).expect("appends");
                feed.seal_open_segment().expect("seals");
                assert_eq!(
                    manifest_records(&dir),
                    seal as usize + 1,
                    "one record per seal, the first of them whole"
                );
            }
            let inner = feed.inner.lock();
            (
                inner.versions.clone(),
                inner.spans.clone(),
                inner.sealed.clone(),
                inner.records_written,
            )
        };
        let feed = ChangeDataFeed::open(tmp.path(), 8, 30).expect("reopens");
        {
            let inner = feed.inner.lock();
            assert_eq!(inner.versions, before.0);
            assert_eq!(inner.spans, before.1);
            assert_eq!(inner.sealed, before.2);
            assert_eq!(inner.records_written, before.3);
        }
        assert_eq!(feed.record_count(), 12);
        assert_eq!(feed.bounded_cut(0, 0, 1), Some(4));
        drop(feed);

        // The last record cut short is the tail a stop mid append leaves.
        // The reopen cuts it off and counts the segment it described from
        // the segment's own file, then lays the manifest down whole
        let path = ChangeDataFeed::manifest_path(&dir);
        let whole = fs::read(&path).expect("reads");
        fs::write(&path, &whole[..whole.len() - 7]).expect("tears the tail");
        let feed = ChangeDataFeed::open(tmp.path(), 8, 30).expect("reopens over a torn tail");
        {
            let inner = feed.inner.lock();
            assert_eq!(inner.versions, before.0);
            assert_eq!(inner.spans, before.1);
            assert_eq!(inner.sealed.len(), before.2.len());
            assert_eq!(inner.records_written, before.3);
        }
        assert_eq!(feed.record_count(), 12);
        assert_eq!(manifest_records(&dir), 1);
    }

    /// Past the threshold of delta records the next record lays the file
    /// down whole, and the chain reads the same either way
    #[test]
    fn test_the_manifest_is_laid_down_whole_past_the_delta_threshold() {
        let tmp = TempDir::new().expect("temp dir");
        let dir = ChangeDataFeed::table_dir(tmp.path(), 9);
        let feed = ChangeDataFeed::open(tmp.path(), 9, 30).expect("opens");
        for version in 1..=MANIFEST_COMPACT_AFTER as u64 + 1 {
            feed.append_change(&make_record(version, version as i64, ChangeType::Insert))
                .expect("appends");
            feed.seal_open_segment().expect("seals");
        }
        assert_eq!(
            manifest_records(&dir),
            1 + MANIFEST_COMPACT_AFTER as usize,
            "one whole record and a delta per seal after it"
        );
        let next = MANIFEST_COMPACT_AFTER as u64 + 2;
        feed.append_change(&make_record(next, next as i64, ChangeType::Insert))
            .expect("appends");
        feed.seal_open_segment().expect("seals past the threshold");
        assert_eq!(
            manifest_records(&dir),
            1,
            "the file is one whole record again"
        );
        let count = feed.record_count();
        drop(feed);
        let feed = ChangeDataFeed::open(tmp.path(), 9, 30).expect("reopens");
        assert_eq!(feed.record_count(), count);
        assert_eq!(feed.record_count(), next);
        assert_eq!(feed.inner.lock().sealed.len(), next as usize);
    }
}
