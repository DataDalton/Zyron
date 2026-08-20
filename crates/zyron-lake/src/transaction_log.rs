//! The self contained transaction log, one .zyl file per commit.
//!
//! Exclusive file creation is the whole concurrency protocol. A commit
//! encodes its entries, then tries File::create_new on the next version
//! number. The loser gets AlreadyExists, reads the winner's 128-byte
//! header in one positioned read, checks for a real conflict, and retries
//! its build closure against the winner's state. The main WAL is never
//! involved, bulk analytical commits do not queue behind the OLTP commit
//! barrier.
//!
//! Atomicity with an enclosing database transaction is by reconciliation.
//! commit() writes the version file and leaves it pending, publish() makes
//! it visible once the transaction's commit record is durable, and open()
//! discards any version whose transaction never committed together with
//! everything after it, since later versions were built on its state.
//!
//! The removed-partition summary in the header is a 64-bit bloom over the
//! removed partition ids. Two headers whose blooms do not intersect are
//! provably disjoint, so append-versus-append and most rewrite pairs
//! resolve without reading either entry section

use std::collections::BTreeMap;
use std::fs;
use std::io::{Read, Write};
use std::path::Path;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Duration;

use zyron_common::ZyronError;

use crate::codec::{Cursor, corrupt};
use crate::index::{IndexFileEntry, LakeIndexSpec};
use crate::manifest::{
    ClusterSpec, DeletePredicate, FileStats, ManifestFile, PartitionEntry, decode_delete_predicate,
    decode_index_file, decode_index_spec, decode_partition_entry, encode_delete_predicate,
    encode_index_file, encode_index_spec, encode_partition_entry,
};
use crate::paths::{LakePaths, VersionFileKind, parse_version_file_name};
use crate::predicate::{LakePredicate, PruneDecision};
use crate::prune_index::PruneIndex;
use crate::schema::LakeSchema;

pub const LOG_MAGIC: [u8; 5] = *b"ZYLOG";
pub const LOG_FORMAT_VERSION: u8 = 1;
pub const COMMIT_HEADER_LEN: usize = 128;

const MAX_COMMIT_ATTEMPTS: u32 = 16;
const BACKOFF_BASE_US: u64 = 50;
const BACKOFF_CAP_US: u64 = 5_000;
const MANIFEST_CACHE_MAX: usize = 32;
const MANIFEST_CACHE_KEEP: usize = 16;

/// Test-only stall inserted between a transactional commit's pending
/// registration and its head advance, in microseconds. Widens the gap the
/// statement ordering closes so a test can reach the interleaving, always
/// zero outside tests
#[cfg(test)]
pub(crate) static TEST_STALL_VISIBILITY_GAP_US: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(0);

// Makes the version write fail for one table, so the recovery path a real
// IO failure takes can be exercised deterministically.
//
// Scoped to a table root rather than switched on globally, because the test
// suite runs in parallel and a global failure switch would fail whichever
// unrelated commit happened to be in flight
#[cfg(test)]
pub(crate) static TEST_FAIL_VERSION_WRITE_UNDER: std::sync::Mutex<Option<std::path::PathBuf>> =
    std::sync::Mutex::new(None);

#[cfg(test)]
fn inject_version_write_failure(path: &Path) -> Option<std::io::Error> {
    let guard = TEST_FAIL_VERSION_WRITE_UNDER.lock().ok()?;
    let root = guard.as_ref()?;
    path.starts_with(root).then(|| {
        std::io::Error::new(
            std::io::ErrorKind::PermissionDenied,
            "injected version write failure",
        )
    })
}

#[cfg(not(test))]
#[inline(always)]
fn inject_version_write_failure(_path: &Path) -> Option<std::io::Error> {
    None
}

#[cfg(test)]
fn stall_visibility_gap() {
    let us = TEST_STALL_VISIBILITY_GAP_US.load(Ordering::Relaxed);
    if us != 0 {
        std::thread::sleep(Duration::from_micros(us));
    }
}

/// What a commit did, one byte in the header. Conflict rules dispatch on
/// this before any entry is read
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OperationKind {
    Append,
    Delete,
    Update,
    Merge,
    Optimize,
    Vacuum,
    SchemaChange,
    Restore,
    Convert,
    SetProperty,
}

impl OperationKind {
    pub fn name(self) -> &'static str {
        match self {
            OperationKind::Append => "APPEND",
            OperationKind::Delete => "DELETE",
            OperationKind::Update => "UPDATE",
            OperationKind::Merge => "MERGE",
            OperationKind::Optimize => "OPTIMIZE",
            OperationKind::Vacuum => "VACUUM",
            OperationKind::SchemaChange => "SCHEMA CHANGE",
            OperationKind::Restore => "RESTORE",
            OperationKind::Convert => "CONVERT",
            OperationKind::SetProperty => "SET PROPERTY",
        }
    }

    fn to_u8(self) -> u8 {
        match self {
            OperationKind::Append => 0,
            OperationKind::Delete => 1,
            OperationKind::Update => 2,
            OperationKind::Merge => 3,
            OperationKind::Optimize => 4,
            OperationKind::Vacuum => 5,
            OperationKind::SchemaChange => 6,
            OperationKind::Restore => 7,
            OperationKind::Convert => 8,
            OperationKind::SetProperty => 9,
        }
    }

    fn from_u8(v: u8) -> Option<Self> {
        Some(match v {
            0 => OperationKind::Append,
            1 => OperationKind::Delete,
            2 => OperationKind::Update,
            3 => OperationKind::Merge,
            4 => OperationKind::Optimize,
            5 => OperationKind::Vacuum,
            6 => OperationKind::SchemaChange,
            7 => OperationKind::Restore,
            8 => OperationKind::Convert,
            9 => OperationKind::SetProperty,
            _ => return None,
        })
    }

    /// Barrier operations conflict with everything concurrent
    fn is_barrier(self) -> bool {
        matches!(
            self,
            OperationKind::SchemaChange | OperationKind::Restore | OperationKind::Convert
        )
    }
}

/// Per-version audit metadata, variable length in the file body
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct CommitInfo {
    /// Committing principal
    pub identity: String,
    /// Client application and version
    pub client: String,
    /// User supplied commit annotation
    pub commit_info: String,
    pub correlation_id: String,
    pub trace_id: String,
    /// Detached signature over the entry section, empty when unsigned
    pub signature: Vec<u8>,
}

impl CommitInfo {
    fn encode_into(&self, buf: &mut Vec<u8>) -> Result<(), ZyronError> {
        for (name, s) in [
            ("identity", &self.identity),
            ("client", &self.client),
            ("commit info", &self.commit_info),
            ("correlation id", &self.correlation_id),
            ("trace id", &self.trace_id),
        ] {
            if s.len() > u16::MAX as usize {
                return Err(ZyronError::Internal(format!(
                    "commit audit {} exceeds {} bytes",
                    name,
                    u16::MAX
                )));
            }
            buf.extend_from_slice(&(s.len() as u16).to_le_bytes());
            buf.extend_from_slice(s.as_bytes());
        }
        if self.signature.len() > u16::MAX as usize {
            return Err(ZyronError::Internal(format!(
                "commit signature exceeds {} bytes",
                u16::MAX
            )));
        }
        buf.extend_from_slice(&(self.signature.len() as u16).to_le_bytes());
        buf.extend_from_slice(&self.signature);
        Ok(())
    }

    fn decode(r: &mut Cursor<'_>) -> Result<Self, ZyronError> {
        let mut fields: [String; 5] = Default::default();
        for slot in &mut fields {
            let len = r.u16()? as usize;
            *slot = r.utf8(len, "audit field")?;
        }
        let sig_len = r.u16()? as usize;
        let signature = r.take(sig_len)?.to_vec();
        let [identity, client, commit_info, correlation_id, trace_id] = fields;
        Ok(Self {
            identity,
            client,
            commit_info,
            correlation_id,
            trace_id,
            signature,
        })
    }
}

/// One state transition inside a commit
#[derive(Debug, Clone, PartialEq)]
pub enum LogEntry {
    /// New data file. `added_version` is stamped by the log at apply time
    AddFile(PartitionEntry),
    RemoveFile {
        partition_id: u64,
    },
    /// New predicate delete. `created_version` is stamped at apply time,
    /// and the id attaches to every live file the predicate may match
    AddDeletePredicate(DeletePredicate),
    RemoveDeletePredicate {
        id: u64,
    },
    /// Full replacement schema, its schema id must exceed the current one
    SchemaChange(LakeSchema),
    /// Full replacement cluster spec, its spec id must exceed the current
    SetClusterSpec(ClusterSpec),
    SetProperty {
        key: String,
        value: String,
    },
    /// New secondary index declaration, its id must be unallocated
    AddIndex(LakeIndexSpec),
    /// Drops the declaration and every file that belongs to it
    DropIndex {
        index_id: u32,
    },
    /// New index file. `added_version` is stamped by the log at apply time
    AddIndexFile(IndexFileEntry),
    RemoveIndexFile {
        index_id: u32,
        partition_id: u64,
    },
}

impl LogEntry {
    fn encode_into(&self, buf: &mut Vec<u8>) -> Result<(), ZyronError> {
        match self {
            LogEntry::AddFile(entry) => {
                buf.push(1);
                encode_partition_entry(entry, buf);
            }
            LogEntry::RemoveFile { partition_id } => {
                buf.push(2);
                buf.extend_from_slice(&partition_id.to_le_bytes());
            }
            LogEntry::AddDeletePredicate(del) => {
                buf.push(3);
                encode_delete_predicate(del, buf);
            }
            LogEntry::RemoveDeletePredicate { id } => {
                buf.push(4);
                buf.extend_from_slice(&id.to_le_bytes());
            }
            LogEntry::SchemaChange(schema) => {
                buf.push(5);
                schema.validate()?;
                schema.encode_into(buf);
            }
            LogEntry::SetClusterSpec(spec) => {
                buf.push(6);
                spec.encode_into(buf);
            }
            LogEntry::SetProperty { key, value } => {
                buf.push(7);
                if key.len() > u16::MAX as usize || value.len() > u32::MAX as usize {
                    return Err(ZyronError::Internal(format!(
                        "property \"{}\" exceeds encodable length",
                        &key[..64.min(key.len())]
                    )));
                }
                buf.extend_from_slice(&(key.len() as u16).to_le_bytes());
                buf.extend_from_slice(key.as_bytes());
                buf.extend_from_slice(&(value.len() as u32).to_le_bytes());
                buf.extend_from_slice(value.as_bytes());
            }
            LogEntry::AddIndex(spec) => {
                buf.push(8);
                spec.validate()?;
                encode_index_spec(spec, buf);
            }
            LogEntry::DropIndex { index_id } => {
                buf.push(9);
                buf.extend_from_slice(&index_id.to_le_bytes());
            }
            LogEntry::AddIndexFile(file) => {
                buf.push(10);
                encode_index_file(file, buf);
            }
            LogEntry::RemoveIndexFile {
                index_id,
                partition_id,
            } => {
                buf.push(11);
                buf.extend_from_slice(&index_id.to_le_bytes());
                buf.extend_from_slice(&partition_id.to_le_bytes());
            }
        }
        Ok(())
    }
}

/// Parses one entry from a positioned cursor. The schema sub-codec
/// consumes from its own slice, so that arm splices the cursor around it
fn decode_entry_stream<'a>(r: &mut Cursor<'a>, ctx: &'a str) -> Result<LogEntry, ZyronError> {
    let tag = r.u8()?;
    Ok(match tag {
        1 => LogEntry::AddFile(decode_partition_entry(r)?),
        2 => LogEntry::RemoveFile {
            partition_id: r.u64()?,
        },
        3 => LogEntry::AddDeletePredicate(decode_delete_predicate(r)?),
        4 => LogEntry::RemoveDeletePredicate { id: r.u64()? },
        5 => {
            let tail = r.take(r.remaining())?;
            let (schema, used) = LakeSchema::decode(tail, ctx)?;
            *r = Cursor::new(&tail[used..], ctx);
            LogEntry::SchemaChange(schema)
        }
        6 => LogEntry::SetClusterSpec(ClusterSpec::decode(r)?),
        7 => {
            let key_len = r.u16()? as usize;
            let key = r.utf8(key_len, "property key")?;
            let value_len = r.u32()? as usize;
            let value = r.utf8(value_len, "property value")?;
            LogEntry::SetProperty { key, value }
        }
        8 => LogEntry::AddIndex(decode_index_spec(r)?),
        9 => LogEntry::DropIndex { index_id: r.u32()? },
        10 => LogEntry::AddIndexFile(decode_index_file(r)?),
        11 => LogEntry::RemoveIndexFile {
            index_id: r.u32()?,
            partition_id: r.u64()?,
        },
        v => return Err(r.corrupt(format!("unknown log entry tag {}", v))),
    })
}

/// The fixed 128-byte commit header at offset 0 of every .zyl file.
/// A conflict check is one positioned read of this struct
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CommitHeader {
    pub version: u64,
    /// Version the writer built against
    pub read_version: u64,
    /// Enclosing database transaction, zero for a standalone commit
    pub db_txn_id: u64,
    pub commit_lsn: u64,
    pub timestamp_us: i64,
    pub operation: OperationKind,
    pub entry_count: u32,
    pub files_added: u32,
    pub files_removed: u32,
    pub rows_added: u64,
    pub rows_removed: u64,
    pub bytes_added: u64,
    /// 64-bit bloom over removed partition ids, zero when nothing removed.
    /// Non-intersecting blooms prove disjoint removals
    pub removed_partition_bloom: u64,
    /// Stable hash of the recorded read predicate, zero when none
    pub read_predicate_hash: u64,
    /// Offset and length of the audit block, zero offset when absent
    pub audit_off: u32,
    pub audit_len: u32,
}

impl CommitHeader {
    fn encode(&self, entry_section_crc: u32) -> [u8; COMMIT_HEADER_LEN] {
        let mut h = [0u8; COMMIT_HEADER_LEN];
        h[0..5].copy_from_slice(&LOG_MAGIC);
        h[5] = LOG_FORMAT_VERSION;
        // 6..8 flags zero, 8..12 header crc patched last
        h[12..20].copy_from_slice(&self.version.to_le_bytes());
        h[20..28].copy_from_slice(&self.read_version.to_le_bytes());
        h[28..36].copy_from_slice(&self.db_txn_id.to_le_bytes());
        h[36..44].copy_from_slice(&self.commit_lsn.to_le_bytes());
        h[44..52].copy_from_slice(&self.timestamp_us.to_le_bytes());
        h[52] = self.operation.to_u8();
        h[53..57].copy_from_slice(&self.entry_count.to_le_bytes());
        h[57..61].copy_from_slice(&self.files_added.to_le_bytes());
        h[61..65].copy_from_slice(&self.files_removed.to_le_bytes());
        h[65..73].copy_from_slice(&self.rows_added.to_le_bytes());
        h[73..81].copy_from_slice(&self.rows_removed.to_le_bytes());
        h[81..89].copy_from_slice(&self.bytes_added.to_le_bytes());
        h[89..97].copy_from_slice(&self.removed_partition_bloom.to_le_bytes());
        h[97..105].copy_from_slice(&self.read_predicate_hash.to_le_bytes());
        h[105..109].copy_from_slice(&entry_section_crc.to_le_bytes());
        h[109..113].copy_from_slice(&self.audit_off.to_le_bytes());
        h[113..117].copy_from_slice(&self.audit_len.to_le_bytes());
        let crc = crc32fast::hash(&h);
        h[8..12].copy_from_slice(&crc.to_le_bytes());
        h
    }

    /// Parses and checksum-verifies a header. Returns the header and the
    /// entry section CRC it protects
    fn decode(bytes: &[u8], ctx: &str) -> Result<(Self, u32), ZyronError> {
        if bytes.len() < COMMIT_HEADER_LEN {
            return Err(corrupt(
                ctx,
                format!(
                    "commit header needs {} bytes, got {}",
                    COMMIT_HEADER_LEN,
                    bytes.len()
                ),
            ));
        }
        if bytes[0..5] != LOG_MAGIC {
            return Err(corrupt(ctx, "bad log magic".into()));
        }
        if bytes[5] != LOG_FORMAT_VERSION {
            return Err(corrupt(
                ctx,
                format!("unsupported log format version {}", bytes[5]),
            ));
        }
        let mut check = [0u8; COMMIT_HEADER_LEN];
        check.copy_from_slice(&bytes[..COMMIT_HEADER_LEN]);
        let mut stored = [0u8; 4];
        stored.copy_from_slice(&check[8..12]);
        let stored_crc = u32::from_le_bytes(stored);
        check[8..12].fill(0);
        let actual = crc32fast::hash(&check);
        if stored_crc != actual {
            return Err(corrupt(
                ctx,
                format!(
                    "commit header checksum mismatch, stored {:#010x} computed {:#010x}",
                    stored_crc, actual
                ),
            ));
        }
        let mut r = Cursor::new(&bytes[6..COMMIT_HEADER_LEN], ctx);
        let flags = r.u16()?;
        if flags != 0 {
            return Err(corrupt(ctx, format!("unknown log flags {:#06x}", flags)));
        }
        let _crc_field = r.u32()?;
        let version = r.u64()?;
        let read_version = r.u64()?;
        let db_txn_id = r.u64()?;
        let commit_lsn = r.u64()?;
        let timestamp_us = r.i64()?;
        let op_raw = r.u8()?;
        let operation = OperationKind::from_u8(op_raw)
            .ok_or_else(|| corrupt(ctx, format!("unknown operation kind {}", op_raw)))?;
        let entry_count = r.u32()?;
        let files_added = r.u32()?;
        let files_removed = r.u32()?;
        let rows_added = r.u64()?;
        let rows_removed = r.u64()?;
        let bytes_added = r.u64()?;
        let removed_partition_bloom = r.u64()?;
        let read_predicate_hash = r.u64()?;
        let entry_section_crc = r.u32()?;
        let audit_off = r.u32()?;
        let audit_len = r.u32()?;
        Ok((
            Self {
                version,
                read_version,
                db_txn_id,
                commit_lsn,
                timestamp_us,
                operation,
                entry_count,
                files_added,
                files_removed,
                rows_added,
                rows_removed,
                bytes_added,
                removed_partition_bloom,
                read_predicate_hash,
                audit_off,
                audit_len,
            },
            entry_section_crc,
        ))
    }
}

/// Reads and verifies only the 128-byte commit header of a version file,
/// the read a conflict check pays
/// Names the file and the operation on an IO error, leaving its kind alone.
///
/// Every classifier in this module dispatches on `io::ErrorKind`: the commit
/// race arm tests for `AlreadyExists`, `is_possible_partial_write` tests for
/// `UnexpectedEof`. So the kind is carried through untouched and only the
/// message grows, which is also why this cannot use a new error variant. The
/// original's own text is kept, so the platform's raw code survives in the
/// message even though `raw_os_error` does not.
///
/// Cold and never inlined. A commit that succeeds never calls it, and the
/// formatting it does is paid only by a commit that has already failed
#[cold]
#[inline(never)]
fn io_context(operation: &str, path: &Path, e: std::io::Error) -> ZyronError {
    let kind = e.kind();
    ZyronError::Io(std::io::Error::new(
        kind,
        format!("{} {}: {}", operation, path.display(), e),
    ))
}

pub fn read_commit_header(path: &Path) -> Result<CommitHeader, ZyronError> {
    let mut file =
        fs::File::open(path).map_err(|e| io_context("open lake version file", path, e))?;
    let mut buf = [0u8; COMMIT_HEADER_LEN];
    file.read_exact(&mut buf)
        .map_err(|e| io_context("read lake version header from", path, e))?;
    let ctx = path.to_string_lossy();
    let (header, _) = CommitHeader::decode(&buf, &ctx)?;
    Ok(header)
}

/// A fully parsed version file
#[derive(Debug, Clone, PartialEq)]
pub struct VersionFileData {
    pub header: CommitHeader,
    pub entries: Vec<LogEntry>,
    pub audit: Option<CommitInfo>,
}

impl VersionFileData {
    pub fn decode<'a>(bytes: &'a [u8], ctx: &'a str) -> Result<Self, ZyronError> {
        let (header, entry_crc) = CommitHeader::decode(bytes, ctx)?;
        let body = &bytes[COMMIT_HEADER_LEN..];
        let actual = crc32fast::hash(body);
        if entry_crc != actual {
            return Err(corrupt(
                ctx,
                format!(
                    "entry section checksum mismatch, stored {:#010x} computed {:#010x}",
                    entry_crc, actual
                ),
            ));
        }
        let entries_end = if header.audit_off != 0 {
            let off = header.audit_off as usize;
            let end = off
                .checked_add(header.audit_len as usize)
                .filter(|&e| e <= bytes.len() && off >= COMMIT_HEADER_LEN)
                .ok_or_else(|| corrupt(ctx, "audit block outside the file".into()))?;
            if end != bytes.len() {
                return Err(corrupt(ctx, "audit block does not end the file".into()));
            }
            off
        } else {
            bytes.len()
        };
        let mut r = Cursor::new(&bytes[COMMIT_HEADER_LEN..entries_end], ctx);
        let mut entries = Vec::with_capacity((header.entry_count as usize).min(r.remaining()));
        for _ in 0..header.entry_count {
            entries.push(decode_entry_stream(&mut r, ctx)?);
        }
        if r.remaining() != 0 {
            return Err(corrupt(ctx, "entry section has trailing bytes".into()));
        }
        let audit = if header.audit_off != 0 {
            let mut a = Cursor::new(&bytes[entries_end..], ctx);
            let info = CommitInfo::decode(&mut a)?;
            if a.remaining() != 0 {
                return Err(corrupt(ctx, "audit block has trailing bytes".into()));
            }
            Some(info)
        } else {
            None
        };
        Ok(Self {
            header,
            entries,
            audit,
        })
    }
}

/// 64-bit bloom over removed partition ids, three bits per id
fn removed_bloom(ids: impl Iterator<Item = u64>) -> u64 {
    let mut bloom = 0u64;
    for id in ids {
        let h = splitmix64(id);
        bloom |= 1u64 << (h & 63);
        bloom |= 1u64 << ((h >> 8) & 63);
        bloom |= 1u64 << ((h >> 16) & 63);
    }
    bloom
}

fn splitmix64(mut x: u64) -> u64 {
    x = x.wrapping_add(0x9E3779B97F4A7C15);
    x = (x ^ (x >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94D049BB133111EB);
    x ^ (x >> 31)
}

/// Applies one commit's entries to a manifest state. `None` is the state
/// before the creating commit, only a SchemaChange can transition out of it
fn apply_entries(
    state: &mut Option<ManifestFile>,
    version: u64,
    timestamp_us: i64,
    entries: &[LogEntry],
) -> Result<(), ZyronError> {
    for entry in entries {
        match entry {
            LogEntry::SchemaChange(schema) => {
                schema.validate()?;
                match state {
                    None => {
                        *state = Some(ManifestFile {
                            snapshot_id: version,
                            parent_snapshot_id: 0,
                            timestamp_us,
                            schema: schema.clone(),
                            cluster_spec: ClusterSpec::none(),
                            entries: Vec::new(),
                            delete_predicates: Vec::new(),
                            properties: BTreeMap::new(),
                            indexes: Vec::new(),
                            index_files: Vec::new(),
                        });
                    }
                    Some(m) => {
                        if schema.schema_id <= m.schema.schema_id {
                            return Err(ZyronError::Internal(format!(
                                "schema id {} does not advance past {}",
                                schema.schema_id, m.schema.schema_id
                            )));
                        }
                        m.schema = schema.clone();
                    }
                }
                continue;
            }
            _ => {}
        }
        let m = state.as_mut().ok_or_else(|| {
            ZyronError::Internal("log entry before the creating schema change".into())
        })?;
        match entry {
            LogEntry::SchemaChange(_) => {}
            LogEntry::AddFile(file) => {
                let mut file = file.clone();
                file.added_version = version;
                match m
                    .entries
                    .binary_search_by_key(&file.partition_id, |e| e.partition_id)
                {
                    Ok(_) => {
                        return Err(ZyronError::Internal(format!(
                            "added partition {:#x} already exists",
                            file.partition_id
                        )));
                    }
                    Err(pos) => m.entries.insert(pos, file),
                }
            }
            LogEntry::RemoveFile { partition_id } => {
                match m
                    .entries
                    .binary_search_by_key(partition_id, |e| e.partition_id)
                {
                    Ok(pos) => {
                        m.entries.remove(pos);
                    }
                    Err(_) => {
                        return Err(ZyronError::Internal(format!(
                            "removed partition {:#x} does not exist",
                            partition_id
                        )));
                    }
                }
            }
            LogEntry::AddDeletePredicate(del) => {
                let mut del = del.clone();
                del.created_version = version;
                let pos = match m.delete_predicates.binary_search_by_key(&del.id, |p| p.id) {
                    Ok(_) => {
                        return Err(ZyronError::Internal(format!(
                            "delete predicate {} already exists",
                            del.id
                        )));
                    }
                    Err(pos) => pos,
                };
                // Disjoint field borrows: the schema types the bloom probe
                // while the entries take their predicate id
                let schema = &m.schema;
                for file in &mut m.entries {
                    if del.predicate.prune(&FileStats::new(&*file, schema))
                        != PruneDecision::CannotMatch
                    {
                        file.delete_predicate_ids.push(del.id);
                    }
                }
                m.delete_predicates.insert(pos, del);
            }
            LogEntry::RemoveDeletePredicate { id } => {
                match m.delete_predicates.binary_search_by_key(id, |p| p.id) {
                    Ok(pos) => {
                        m.delete_predicates.remove(pos);
                    }
                    Err(_) => {
                        return Err(ZyronError::Internal(format!(
                            "removed delete predicate {} does not exist",
                            id
                        )));
                    }
                }
                for file in &mut m.entries {
                    file.delete_predicate_ids.retain(|r| r != id);
                }
            }
            LogEntry::SetClusterSpec(spec) => {
                if spec.spec_id <= m.cluster_spec.spec_id {
                    return Err(ZyronError::Internal(format!(
                        "cluster spec id {} does not advance past {}",
                        spec.spec_id, m.cluster_spec.spec_id
                    )));
                }
                m.cluster_spec = spec.clone();
            }
            LogEntry::SetProperty { key, value } => {
                m.properties.insert(key.clone(), value.clone());
            }
            LogEntry::AddIndex(spec) => {
                spec.validate()?;
                for id in &spec.column_ids {
                    if m.schema.column_by_id(*id).is_none() {
                        return Err(ZyronError::Internal(format!(
                            "index \"{}\" names column {}, which is not in the schema",
                            spec.name, id
                        )));
                    }
                }
                match m
                    .indexes
                    .binary_search_by_key(&spec.index_id, |s| s.index_id)
                {
                    Ok(_) => {
                        return Err(ZyronError::Internal(format!(
                            "index id {} is already allocated",
                            spec.index_id
                        )));
                    }
                    Err(pos) => m.indexes.insert(pos, spec.clone()),
                }
            }
            LogEntry::DropIndex { index_id } => {
                match m.indexes.binary_search_by_key(index_id, |s| s.index_id) {
                    Ok(pos) => {
                        m.indexes.remove(pos);
                    }
                    Err(_) => {
                        return Err(ZyronError::Internal(format!(
                            "dropped index {} does not exist",
                            index_id
                        )));
                    }
                }
                // A dropped index takes its files with it, so no entry is
                // ever left pointing at a declaration that is gone
                m.index_files.retain(|f| f.index_id != *index_id);
            }
            LogEntry::AddIndexFile(file) => {
                if m.index_by_id(file.index_id).is_none() {
                    return Err(ZyronError::Internal(format!(
                        "index file {:#x} belongs to index {}, which is not declared",
                        file.file.partition_id, file.index_id
                    )));
                }
                let mut file = file.clone();
                file.file.added_version = version;
                let key = (file.index_id, file.file.partition_id);
                match m
                    .index_files
                    .binary_search_by_key(&key, |f| (f.index_id, f.file.partition_id))
                {
                    Ok(_) => {
                        return Err(ZyronError::Internal(format!(
                            "index {} already holds partition {:#x}",
                            key.0, key.1
                        )));
                    }
                    Err(pos) => m.index_files.insert(pos, file),
                }
            }
            LogEntry::RemoveIndexFile {
                index_id,
                partition_id,
            } => {
                let key = (*index_id, *partition_id);
                match m
                    .index_files
                    .binary_search_by_key(&key, |f| (f.index_id, f.file.partition_id))
                {
                    Ok(pos) => {
                        m.index_files.remove(pos);
                    }
                    Err(_) => {
                        return Err(ZyronError::Internal(format!(
                            "removed index {} partition {:#x} does not exist",
                            index_id, partition_id
                        )));
                    }
                }
            }
        }
    }
    if let Some(m) = state {
        m.parent_snapshot_id = m.snapshot_id;
        m.snapshot_id = version;
        m.timestamp_us = timestamp_us;
    }
    Ok(())
}

/// Answers whether an enclosing database transaction committed, the
/// reconciliation source consulted on open
pub trait CommitStatus: Send + Sync {
    fn is_committed(&self, db_txn_id: u64) -> bool;
}

/// Treats every transaction as committed, for standalone lake tables and
/// tests without a database transaction manager
pub struct AllCommitted;

impl CommitStatus for AllCommitted {
    fn is_committed(&self, _db_txn_id: u64) -> bool {
        true
    }
}

/// Everything a commit carries besides its entries
#[derive(Debug, Clone, Copy)]
pub struct CommitAttempt<'a> {
    pub operation: OperationKind,
    /// Enclosing database transaction, zero commits standalone and
    /// publishes immediately
    pub db_txn_id: u64,
    pub commit_lsn: u64,
    pub timestamp_us: i64,
    /// Predicate the writer's reads depended on, recorded for phantom
    /// detection under rule R4
    pub read_predicate: Option<&'a LakePredicate>,
    /// Version the read predicate was evaluated at. When non-zero, the
    /// commit phantom-checks the predicate against every version that
    /// landed after this base, because those writers were never met in a
    /// version-file collision and rule R4 alone would miss them. Zero
    /// means the predicate carries no base pin
    pub read_version: u64,
    pub audit: Option<&'a CommitInfo>,
}

/// Table property naming the node that owns writes to a dataset.
pub const WRITER_NODE_PROPERTY: &str = "writer_node";

// The node this process is. Set once at startup from the node identity, so
// every lake log in the process agrees on who is writing without threading
// it through every commit
static LOCAL_NODE: AtomicU64 = AtomicU64::new(0);

/// Declares which node this process is, for single-writer enforcement.
///
/// Zero means no identity was established, which turns claiming and
/// enforcement off: a tool operating on a data directory directly, or a
/// single-node deployment that never joined a mesh, has no second writer to
/// be protected from and must not be locked out of its own data.
///
/// Each log captures this when it is opened, so a server sets it once at
/// startup before opening anything. Changing it later does not retroactively
/// change who owns an already open log, which is what keeps the identity of
/// a running writer stable.
pub fn set_local_node(node_id: u64) {
    LOCAL_NODE.store(node_id, Ordering::Release);
}

/// The node this process is, zero when none was established.
#[inline]
pub fn local_node() -> u64 {
    LOCAL_NODE.load(Ordering::Acquire)
}

/// The node that owns writes to this table, None when unclaimed.
pub fn writer_node(manifest: &ManifestFile) -> Option<u64> {
    manifest
        .properties
        .get(WRITER_NODE_PROPERTY)
        .and_then(|v| v.parse().ok())
        .filter(|id| *id != 0)
}

/// What a commit from `local` must do about the recorded owner.
///
/// A dataset is written by exactly one node. That is not a limitation to
/// work around, it is what removes conflict resolution entirely: with one
/// writer the log is a total order per dataset and applying it elsewhere is
/// a replay, with no merge step and nothing to reconcile. Multi-writer
/// convergence would need last-write-wins, which loses data, or CRDT merge,
/// which cannot express an invariant like `balance >= 0`.
fn writer_decision(owner: Option<u64>, local: u64) -> WriterDecision {
    match (owner, local) {
        // No identity established, so there is nothing to claim and nobody
        // to exclude. A tool must still be able to work on its own files
        (_, 0) => WriterDecision::Proceed,
        (None, _) => WriterDecision::Claim,
        (Some(owner), local) if owner == local => WriterDecision::Proceed,
        (Some(owner), local) => WriterDecision::Foreign { owner, local },
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum WriterDecision {
    Proceed,
    Claim,
    Foreign { owner: u64, local: u64 },
}

/// Moves write ownership of a dataset to another node.
///
/// The single-writer rule would strand a dataset if the owning node never
/// came back, so ownership is transferable. It is an operator action and
/// not an automatic one on purpose: a node that looks unreachable may only
/// be partitioned, and a mesh that reassigns ownership on a timeout is a
/// mesh with two writers during the partition, which is the exact state
/// the rule exists to prevent.
///
/// Writes the property directly rather than through `commit`, because the
/// current owner is by definition not this node and `commit` would refuse.
pub fn transfer_writer(
    log: &TransactionLog,
    attempt: CommitAttempt<'_>,
    to_node: u64,
) -> Result<u64, ZyronError> {
    if to_node == 0 {
        return Err(ZyronError::ConfigError(
            "a dataset cannot be transferred to node zero, which means no node".into(),
        ));
    }
    let mut attempt = attempt;
    attempt.operation = OperationKind::SetProperty;
    // A commit whose entries are exactly one writer-property assignment is
    // a transfer, and commit recognizes that shape and lets it through.
    // Recognizing it by shape rather than by a flag or by muting the local
    // node keeps the rule thread safe: a concurrent data write from this
    // node is judged on its own entries and never sees enforcement
    // relaxed for someone else's transfer
    log.commit(attempt, |_| {
        Ok(vec![LogEntry::SetProperty {
            key: WRITER_NODE_PROPERTY.to_string(),
            value: to_node.to_string(),
        }])
    })
}

/// True when a commit is exactly a change of owner and nothing else.
///
/// Exactly, not merely including: a data write that smuggled in a writer
/// property would take ownership as a side effect, which is how a second
/// writer would get in through the one door the rule leaves open.
fn is_writer_transfer(entries: &[LogEntry]) -> bool {
    match entries {
        [LogEntry::SetProperty { key, .. }] => key == WRITER_NODE_PROPERTY,
        _ => false,
    }
}

/// One table's transaction log. All state is atomics and lock-free maps,
/// reads never take a lock
pub struct TransactionLog {
    paths: LakePaths,
    /// Branch this head reads and writes, None for main. A branch's version
    /// files live in their own directory and start after `branch_base`, so a
    /// branch costs one directory and copies no data
    branch: Option<String>,
    /// Main version the branch forked from, zero on main
    branch_base: u64,
    /// Newest version file that exists on disk
    created_head: AtomicU64,
    /// Contiguous published watermark, what readers see
    published: AtomicU64,
    /// Created but not yet published versions, version to db txn id
    pending: scc::HashMap<u64, u64>,
    /// Node this log writes as, captured when it was opened. Zero means
    /// no identity, which turns single-writer enforcement off
    local_node: AtomicU64,
    /// Reconstructed manifests by version
    manifests: scc::HashMap<u64, Arc<CachedManifest>>,
    commit_retries: AtomicU64,
}

/// One version's manifest and the projections derived from it.
///
/// A projection is a pure function of the manifest, so it is built once
/// per version, on first use, and shared by every plan reading that
/// version. Holding it beside the manifest means one cache and one
/// eviction rule rather than two that can disagree
struct CachedManifest {
    manifest: Arc<ManifestFile>,
    prune: std::sync::OnceLock<Arc<PruneIndex>>,
}

impl CachedManifest {
    fn new(manifest: ManifestFile) -> Arc<Self> {
        Arc::new(Self {
            manifest: Arc::new(manifest),
            prune: std::sync::OnceLock::new(),
        })
    }
}

// Process-global registry of open logs keyed by head directory, so every
// scan and commit against one head shares one head version, one pending set
// and one manifest cache. Follows the columnar patch manager's precedent.
//
// Main is keyed by the table root and each branch by its own version
// directory, because the two are separate heads that commit independently:
// one pending set covering both would make a branch write block a main write
static OPEN_LOGS: std::sync::OnceLock<scc::HashMap<std::path::PathBuf, Arc<TransactionLog>>> =
    std::sync::OnceLock::new();

fn open_logs() -> &'static scc::HashMap<std::path::PathBuf, Arc<TransactionLog>> {
    OPEN_LOGS.get_or_init(scc::HashMap::new)
}

impl TransactionLog {
    /// Opens a table's log through the process-global registry, one shared
    /// instance per table root. Server startup primes the registry with the
    /// real commit-status source during recovery, later callers get that
    /// instance back and `status` is only consulted for a first open
    pub fn open_shared(
        paths: LakePaths,
        status: &dyn CommitStatus,
    ) -> Result<Arc<TransactionLog>, ZyronError> {
        let key = paths.root().to_path_buf();
        if let Some(hit) = Self::lookup_registered(&key) {
            return Ok(hit);
        }
        Self::share_registered(key, Self::open(paths, status)?)
    }

    /// The registry entry a log occupies: a table's root for main, a
    /// branch's own version directory for a branch
    pub fn registry_key(&self) -> std::path::PathBuf {
        match &self.branch {
            Some(name) => self.paths.branch_dir(name),
            None => self.paths.root().to_path_buf(),
        }
    }

    /// Returns an already-registered head, without opening one.
    pub(crate) fn lookup_registered(key: &std::path::Path) -> Option<Arc<TransactionLog>> {
        open_logs().read_sync(&key.to_path_buf(), |_, v| Arc::clone(v))
    }

    /// Publishes a freshly opened head into the registry and returns the
    /// instance every caller shares. A racing insert wins, so one head is
    /// never two instances with two pending sets
    pub(crate) fn share_registered(
        key: std::path::PathBuf,
        log: TransactionLog,
    ) -> Result<Arc<TransactionLog>, ZyronError> {
        let registry = open_logs();
        let log = Arc::new(log);
        match registry.insert_sync(key.clone(), Arc::clone(&log)) {
            Ok(()) => Ok(log),
            Err(_) => registry
                .read_sync(&key, |_, v| Arc::clone(v))
                .ok_or_else(|| {
                    ZyronError::Internal("lake log registry lost a racing insert".into())
                }),
        }
    }

    /// Registers an already-open log, used by table creation and startup
    /// recovery so later scans share it
    pub fn register_shared(log: Arc<TransactionLog>) {
        let key = log.registry_key();
        let _ = open_logs().insert_sync(key, log);
    }

    /// The node this log writes as.
    #[inline]
    pub fn writer_identity(&self) -> u64 {
        self.local_node.load(Ordering::Acquire)
    }

    /// Overrides the node this log writes as.
    ///
    /// A log captures the process identity when it opens, which is what a
    /// server wants. A caller that manages several datasets on behalf of
    /// different nodes, and any test that needs two writers without
    /// disturbing the process, sets it per log instead.
    pub fn set_writer_identity(&self, node_id: u64) {
        self.local_node.store(node_id, Ordering::Release);
    }

    /// Returns a table's shared log when one is already open, without
    /// opening it. Used to report which logs a node holds
    pub fn lookup_shared(paths: &LakePaths) -> Option<Arc<TransactionLog>> {
        open_logs().read_sync(&paths.root().to_path_buf(), |_, v| Arc::clone(v))
    }

    /// Drops a table's registry entries, used by DROP TABLE. Every branch
    /// head is keyed under the table root, so dropping the table drops them
    /// with it rather than leaving heads pointing at deleted directories
    pub fn remove_shared(paths: &LakePaths) {
        let root = paths.root().to_path_buf();
        open_logs().retain_sync(|key, _| !key.starts_with(&root));
        // A dropped table has nothing left to maintain, and leaving its
        // marks behind would hold slots against the signal's bound
        crate::maintenance_signal::maintenance_signal().forget_under(&root);
    }

    /// Drops one head's registry entry, used when a branch is dropped
    pub(crate) fn remove_registered(key: &std::path::Path) {
        let _ = open_logs().remove_sync(&key.to_path_buf());
    }
}

// Lake versions written inside database transactions, awaiting the
// transaction's durable commit record. The wire layer publishes or
// abandons them when the transaction ends.
//
// The key is the owning data directory as well as the transaction id.
// Transaction ids are dense per database because they are also MVCC row
// stamps, so two databases hosted by one process both start at one and would
// otherwise claim each other's pending versions: the first to commit would
// publish both and the second would find its own already gone.
static TXN_PENDING: std::sync::OnceLock<PendingMap> = std::sync::OnceLock::new();

type PendingKey = (std::path::PathBuf, u64);
type PendingMap = scc::HashMap<PendingKey, Vec<(std::path::PathBuf, u64)>>;

fn txn_pending() -> &'static PendingMap {
    TXN_PENDING.get_or_init(scc::HashMap::new)
}

/// Records a lake version committed under a database transaction, made
/// visible by publish_txn once the transaction's commit record is durable.
/// `data_dir` is the database the transaction belongs to.
pub fn register_txn_pending(
    data_dir: &std::path::Path,
    db_txn_id: u64,
    root: std::path::PathBuf,
    version: u64,
) {
    let key = (data_dir.to_path_buf(), db_txn_id);
    let entry = (root, version);
    if txn_pending()
        .update_sync(&key, |_, list| list.push(entry.clone()))
        .is_none()
    {
        let _ = txn_pending().insert_sync(key, vec![entry]);
    }
}

/// Publishes every lake version the transaction wrote, in version order,
/// called after the transaction's commit record is durable. A transaction
/// that wrote no lake versions is one lock-free lookup
pub fn publish_txn(
    data_dir: &std::path::Path,
    db_txn_id: u64,
) -> Result<Vec<Arc<TransactionLog>>, ZyronError> {
    let Some((_, mut list)) = txn_pending().remove_sync(&(data_dir.to_path_buf(), db_txn_id))
    else {
        return Ok(Vec::new());
    };
    list.sort_by_key(|(_, version)| *version);
    let mut published: Vec<Arc<TransactionLog>> = Vec::new();
    for (root, version) in list {
        let Some(log) = open_logs().read_sync(&root, |_, v| Arc::clone(v)) else {
            return Err(ZyronError::Internal(format!(
                "lake log at {} vanished before publish of version {}",
                root.display(),
                version
            )));
        };
        log.publish(version)?;
        if !published.iter().any(|l| Arc::ptr_eq(l, &log)) {
            published.push(log);
        }
    }
    Ok(published)
}

/// Discards every lake version the transaction wrote, newest first,
/// called when the transaction aborts. Best effort, recovery is the
/// authority for anything that slips through
pub fn abandon_txn(data_dir: &std::path::Path, db_txn_id: u64) -> Vec<Arc<TransactionLog>> {
    let mut touched: Vec<Arc<TransactionLog>> = Vec::new();
    let Some((_, mut list)) = txn_pending().remove_sync(&(data_dir.to_path_buf(), db_txn_id))
    else {
        return touched;
    };
    list.sort_by_key(|(_, version)| std::cmp::Reverse(*version));
    for (root, version) in list {
        let Some(log) = open_logs().read_sync(&root, |_, v| Arc::clone(v)) else {
            continue;
        };
        if let Err(e) = log.abandon(version) {
            tracing::warn!(
                version,
                root = %root.display(),
                error = %e,
                "lake version abandon failed, recovery will discard it"
            );
        }
        if !touched.iter().any(|l| Arc::ptr_eq(l, &log)) {
            touched.push(log);
        }
    }
    touched
}

impl std::fmt::Debug for TransactionLog {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TransactionLog")
            .field("root", &self.paths.root())
            .field("created_head", &self.created_head.load(Ordering::Relaxed))
            .field("published", &self.published.load(Ordering::Relaxed))
            .finish()
    }
}

impl TransactionLog {
    /// Creates the table by writing version one with the given schema,
    /// cluster spec and properties. Fails if the log already exists
    pub fn create(
        paths: LakePaths,
        attempt: CommitAttempt<'_>,
        schema: &LakeSchema,
        cluster_spec: Option<&ClusterSpec>,
        properties: &BTreeMap<String, String>,
    ) -> Result<Self, ZyronError> {
        fs::create_dir_all(paths.log_dir())?;
        fs::create_dir_all(paths.data_dir())?;
        fs::create_dir_all(paths.tmp_dir())?;
        let mut entries = vec![LogEntry::SchemaChange(schema.clone())];
        if let Some(spec) = cluster_spec {
            entries.push(LogEntry::SetClusterSpec(spec.clone()));
        }
        for (key, value) in properties {
            entries.push(LogEntry::SetProperty {
                key: key.clone(),
                value: value.clone(),
            });
        }
        // The node that creates a dataset owns writes to it
        let local = local_node();
        if local != 0 && !properties.contains_key(WRITER_NODE_PROPERTY) {
            entries.push(LogEntry::SetProperty {
                key: WRITER_NODE_PROPERTY.to_string(),
                value: local.to_string(),
            });
        }
        Self::create_from_entries(paths, attempt, entries)
    }

    /// Materializes a table whose first version is exactly `entries`.
    ///
    /// `create` builds the entries a fresh empty table needs and comes
    /// here. A clone comes here too, with the same prefix plus one AddFile
    /// per file it starts life holding, so a cloned table and a fresh one
    /// are the same kind of thing from version one: nothing downstream has
    /// to know which it is reading
    pub(crate) fn create_from_entries(
        paths: LakePaths,
        attempt: CommitAttempt<'_>,
        entries: Vec<LogEntry>,
    ) -> Result<Self, ZyronError> {
        fs::create_dir_all(paths.log_dir())?;
        fs::create_dir_all(paths.data_dir())?;
        fs::create_dir_all(paths.tmp_dir())?;
        let log = Self {
            paths,
            branch: None,
            branch_base: 0,
            created_head: AtomicU64::new(0),
            published: AtomicU64::new(0),
            pending: scc::HashMap::new(),
            manifests: scc::HashMap::new(),
            commit_retries: AtomicU64::new(0),
            local_node: AtomicU64::new(local_node()),
        };
        let mut state = None;
        apply_entries(&mut state, 1, attempt.timestamp_us, &entries)?;
        let manifest = state
            .ok_or_else(|| ZyronError::Internal("table creation produced no manifest".into()))?;
        let bytes = encode_version_file(1, 0, &attempt, &entries, &manifest)?;
        let path = log.paths.version_file(1);
        let mut file = match fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&path)
        {
            Ok(f) => f,
            Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => {
                return Err(ZyronError::Internal(format!(
                    "lake table log already exists at {}",
                    path.display()
                )));
            }
            Err(e) => return Err(e.into()),
        };
        file.write_all(&bytes)?;
        file.sync_all()?;
        drop(file);
        let _ = log.manifests.insert_sync(1, CachedManifest::new(manifest));
        log.created_head.store(1, Ordering::Release);
        if attempt.db_txn_id == 0 {
            log.published.store(1, Ordering::Release);
            log.write_latest_hint(1);
        } else {
            let _ = log.pending.insert_sync(1, attempt.db_txn_id);
        }
        Ok(log)
    }

    /// Opens an existing log, verifying every version file after the
    /// newest checkpoint and discarding versions whose enclosing
    /// transaction never committed together with everything after them
    pub fn open(paths: LakePaths, status: &dyn CommitStatus) -> Result<Self, ZyronError> {
        let log_dir = paths.log_dir();
        let mut versions = Vec::new();
        let mut checkpoints = Vec::new();
        for dirent in fs::read_dir(&log_dir)? {
            let dirent = dirent?;
            let name = dirent.file_name();
            let Some(name) = name.to_str() else { continue };
            match parse_version_file_name(name) {
                Some((v, VersionFileKind::Version)) => versions.push(v),
                Some((v, VersionFileKind::Checkpoint)) => checkpoints.push(v),
                None => {}
            }
        }
        versions.sort_unstable();
        checkpoints.sort_unstable();
        if versions.is_empty() && checkpoints.is_empty() {
            return Err(ZyronError::Internal(format!(
                "no lake table log at {}",
                log_dir.display()
            )));
        }
        // Everything at or below the newest checkpoint is durable state,
        // reconcile only the versions after it
        let base = checkpoints.last().copied().unwrap_or(0);
        let mut head = base;
        for &v in versions.iter().filter(|&&v| v > base) {
            if v != head + 1 {
                break;
            }
            let header = read_commit_header(&paths.version_file(v))?;
            if header.version != v {
                return Err(corrupt(
                    &paths.version_file(v).to_string_lossy(),
                    format!(
                        "header says version {}, file name says {}",
                        header.version, v
                    ),
                ));
            }
            if header.db_txn_id != 0 && !status.is_committed(header.db_txn_id) {
                break;
            }
            head = v;
        }
        // Discard the uncommitted or unreachable tail, later versions were
        // built on discarded state so they go too
        for &v in versions.iter().filter(|&&v| v > head) {
            let path = paths.version_file(v);
            fs::remove_file(&path).map_err(|e| {
                ZyronError::Internal(format!(
                    "cannot discard uncommitted version file {}: {}",
                    path.display(),
                    e
                ))
            })?;
            tracing::warn!(version = v, "discarded uncommitted lake log version");
        }
        if head == 0 {
            return Err(ZyronError::Internal(format!(
                "lake table log at {} has no committed versions",
                log_dir.display()
            )));
        }
        let log = Self {
            paths,
            branch: None,
            branch_base: 0,
            created_head: AtomicU64::new(head),
            published: AtomicU64::new(head),
            pending: scc::HashMap::new(),
            manifests: scc::HashMap::new(),
            commit_retries: AtomicU64::new(0),
            local_node: AtomicU64::new(local_node()),
        };
        // Validate the surviving chain replays cleanly
        let _ = log.manifest_at(head)?;
        log.write_latest_hint(head);
        Ok(log)
    }

    /// Path layout of the table this log belongs to
    pub fn paths(&self) -> &LakePaths {
        &self.paths
    }

    /// Newest published version, what a scan reads
    pub fn latest_version(&self) -> u64 {
        self.published.load(Ordering::Acquire)
    }

    /// Newest version file on disk including pending ones
    pub fn head_version(&self) -> u64 {
        self.created_head.load(Ordering::Acquire)
    }

    /// Times a commit lost the version race and retried
    pub fn commit_retries(&self) -> u64 {
        self.commit_retries.load(Ordering::Relaxed)
    }

    /// Manifest at the newest published version
    pub fn latest_manifest(&self) -> Result<Arc<ManifestFile>, ZyronError> {
        self.manifest_at(self.latest_version())
    }

    /// Manifest at any existing version, reconstructing from the nearest
    /// checkpoint at or below it plus the version files after it
    pub fn manifest_at(&self, version: u64) -> Result<Arc<ManifestFile>, ZyronError> {
        Ok(Arc::clone(&self.cached_at(version)?.manifest))
    }

    /// Vectorized pruning projection at one version, built on first use.
    ///
    /// Every plan against a version shares one projection, so the cost of
    /// building it is paid once however many scans read that version
    pub fn prune_index_at(&self, version: u64) -> Result<Arc<PruneIndex>, ZyronError> {
        let cached = self.cached_at(version)?;
        Ok(Arc::clone(cached.prune.get_or_init(|| {
            Arc::new(PruneIndex::build(&cached.manifest))
        })))
    }

    /// Pruning projection at the newest published version
    pub fn latest_prune_index(&self) -> Result<Arc<PruneIndex>, ZyronError> {
        self.prune_index_at(self.latest_version())
    }

    fn cached_at(&self, version: u64) -> Result<Arc<CachedManifest>, ZyronError> {
        if version == 0 || version > self.head_version() {
            return Err(ZyronError::Internal(format!(
                "lake version {} does not exist, head is {}",
                version,
                self.head_version()
            )));
        }
        if let Some(hit) = self.manifests.read_sync(&version, |_, v| Arc::clone(v)) {
            return Ok(hit);
        }
        let cached = CachedManifest::new(self.reconstruct(version)?);
        let _ = self.manifests.insert_sync(version, Arc::clone(&cached));
        self.evict_cache();
        Ok(cached)
    }

    fn reconstruct(&self, version: u64) -> Result<ManifestFile, ZyronError> {
        // Nearest checkpoint at or below the target
        let mut base = 0u64;
        let ceiling = match self.branch {
            Some(_) => version.min(self.branch_base),
            None => version,
        };
        // The hint names the newest checkpoint, one small read instead of a
        // directory scan per reconstruction. It is advisory: absent, stale,
        // above the ceiling or pointing at a removed file falls back to the
        // scan below
        if let Ok(text) = fs::read_to_string(self.paths.last_checkpoint_hint()) {
            if let Ok(hinted) = text.trim().parse::<u64>() {
                if hinted > 0 && hinted <= ceiling && self.paths.checkpoint_file(hinted).exists() {
                    base = hinted;
                }
            }
        }
        if base == 0 {
            for dirent in fs::read_dir(self.paths.log_dir())? {
                let dirent = dirent?;
                let name = dirent.file_name();
                let Some(name) = name.to_str() else { continue };
                if let Some((v, VersionFileKind::Checkpoint)) = parse_version_file_name(name) {
                    // Checkpoints live on main. A branch replays from the
                    // newest one at or below its fork point, then its own
                    // versions
                    if v <= ceiling && v > base {
                        base = v;
                    }
                }
            }
        }
        let mut state = if base > 0 {
            let path = self.paths.checkpoint_file(base);
            let bytes = fs::read(&path)?;
            Some(ManifestFile::decode(&bytes, &path.to_string_lossy())?)
        } else {
            None
        };
        for v in (base + 1)..=version {
            let path = self.version_path(v);
            let bytes = fs::read(&path)?;
            let data = VersionFileData::decode(&bytes, &path.to_string_lossy())?;
            apply_entries(&mut state, v, data.header.timestamp_us, &data.entries)?;
        }
        state.ok_or_else(|| {
            ZyronError::Internal(format!("lake version {} replayed to no manifest", version))
        })
    }

    /// Opens a branch head: main's history up to the fork point plus the
    /// branch's own versions after it.
    ///
    /// A version file that fails to read or whose header disagrees with its
    /// name ends the chain, and everything after it is discarded, because a
    /// version built on discarded state is not reachable either. That is the
    /// same rule main recovery applies.
    pub fn open_branch(
        paths: LakePaths,
        branch: &str,
        branch_base: u64,
    ) -> Result<Self, ZyronError> {
        let dir = paths.branch_dir(branch);
        let mut versions = Vec::new();
        if dir.exists() {
            for dirent in fs::read_dir(&dir)? {
                let dirent = dirent?;
                let name = dirent.file_name();
                let Some(name) = name.to_str() else { continue };
                if let Some((v, VersionFileKind::Version)) = parse_version_file_name(name) {
                    if v > branch_base {
                        versions.push(v);
                    }
                }
            }
        }
        versions.sort_unstable();

        let mut head = branch_base;
        for &v in &versions {
            if v != head + 1 {
                break;
            }
            let path = paths.branch_version_file(branch, v);
            let header = match read_commit_header(&path) {
                Ok(h) => h,
                Err(_) => break,
            };
            if header.version != v {
                break;
            }
            head = v;
        }
        for &v in versions.iter().filter(|&&v| v > head) {
            let path = paths.branch_version_file(branch, v);
            fs::remove_file(&path).map_err(|e| {
                ZyronError::Internal(format!(
                    "cannot discard unreachable branch version file {}: {}",
                    path.display(),
                    e
                ))
            })?;
            tracing::warn!(
                branch,
                version = v,
                "discarded unreachable lake branch version"
            );
        }

        let log = Self {
            paths,
            branch: Some(branch.to_string()),
            branch_base,
            created_head: AtomicU64::new(head),
            published: AtomicU64::new(head),
            pending: scc::HashMap::new(),
            manifests: scc::HashMap::new(),
            commit_retries: AtomicU64::new(0),
            local_node: AtomicU64::new(local_node()),
        };
        // The chain must replay, a branch that cannot be read is not a
        // branch anyone can merge
        log.manifest_at(head)?;
        Ok(log)
    }

    /// The file one version lives in. A branch keeps its own versions
    /// after the fork point and shares everything at or below it, which is
    /// what makes a branch metadata only.
    pub(crate) fn version_path(&self, version: u64) -> std::path::PathBuf {
        match &self.branch {
            Some(name) if version > self.branch_base => {
                self.paths.branch_version_file(name, version)
            }
            _ => self.paths.version_file(version),
        }
    }

    /// The branch this head writes, None for main
    pub fn branch_name(&self) -> Option<&str> {
        self.branch.as_deref()
    }

    /// The main version a branch forked from, zero on main
    pub fn branch_base(&self) -> u64 {
        self.branch_base
    }

    fn evict_cache(&self) {
        if self.manifests.len() <= MANIFEST_CACHE_MAX {
            return;
        }
        let mut versions = Vec::with_capacity(self.manifests.len());
        self.manifests.iter_sync(|k, _| {
            versions.push(*k);
            true
        });
        versions.sort_unstable();
        let cut = versions.len().saturating_sub(MANIFEST_CACHE_KEEP);
        for v in &versions[..cut] {
            let _ = self.manifests.remove_sync(v);
        }
    }

    /// Commits one operation. The build closure receives the current base
    /// manifest and returns the entries, it reruns on every retry so ids
    /// and derived state are re-allocated against the winner's state.
    /// Returns the committed version, pending until publish() unless the
    /// attempt is standalone
    pub fn commit(
        &self,
        attempt: CommitAttempt<'_>,
        mut build: impl FnMut(&ManifestFile) -> Result<Vec<LogEntry>, ZyronError>,
    ) -> Result<u64, ZyronError> {
        // Attempts spent waiting on state that may never resolve on its own.
        // Losing a version race is deliberately not one of them: losing means
        // another writer committed, so the head advanced and the next attempt
        // builds on real progress. Bounding those would fail an append that
        // conflicts with nothing, purely because other appends were faster,
        // which is what eight concurrent appenders used to hit
        let mut waits: u32 = 0;
        let mut races: u64 = 0;
        loop {
            // A pending version from another transaction is uncommitted
            // state, building on it would tie this commit to its fate.
            // The head is read before the pending scan: registration
            // precedes the head advance on the winner side, so a base at
            // an unresolved transactional version is always caught here
            let base_version = self.head_version();
            if base_version == 0 {
                return Err(ZyronError::Internal(
                    "lake table log is empty, create it before committing".into(),
                ));
            }
            let mut foreign_pending = false;
            self.pending.iter_sync(|_, txn| {
                if *txn != attempt.db_txn_id {
                    foreign_pending = true;
                    return false;
                }
                true
            });
            if foreign_pending {
                // Nothing here guarantees the other transaction ever
                // resolves, so this is the wait that stays bounded
                if waits >= MAX_COMMIT_ATTEMPTS {
                    return Err(ZyronError::ConflictError {
                        mine: attempt.operation.name().to_string(),
                        theirs: "a pending transaction".to_string(),
                        reason: format!(
                            "waited {} attempts for another transaction's pending version to \
                             publish or abandon",
                            MAX_COMMIT_ATTEMPTS
                        ),
                    });
                }
                self.commit_retries.fetch_add(1, Ordering::Relaxed);
                backoff(waits);
                waits += 1;
                continue;
            }
            if self.head_version() != base_version {
                // The head moved between the read and the pending scan,
                // either forward past a new commit or backward past an
                // abandon, so the base is re-derived
                races += 1;
                backoff(races.min(u32::MAX as u64) as u32);
                continue;
            }
            let base = match self.manifest_at(base_version) {
                Ok(base) => base,
                Err(_) if self.head_version() != base_version => {
                    // The base was abandoned after the checks above, its
                    // retraction rolls the head back before removing the
                    // files, so a vanished base always shows here as a
                    // moved head
                    races += 1;
                    backoff(races.min(u32::MAX as u64) as u32);
                    continue;
                }
                Err(e) => return Err(e),
            };
            let mut entries = build(&base)?;
            if entries.is_empty() {
                return Err(ZyronError::Internal("commit with no entries".into()));
            }
            // Single writer per dataset, checked against the base on every
            // attempt rather than once, because a retry may be building on
            // a version another node committed in between
            match writer_decision(writer_node(&base), self.writer_identity()) {
                WriterDecision::Proceed => {}
                WriterDecision::Claim => entries.push(LogEntry::SetProperty {
                    key: WRITER_NODE_PROPERTY.to_string(),
                    value: self.writer_identity().to_string(),
                }),
                // An explicit handover, which is the only way a node other
                // than the owner may write
                WriterDecision::Foreign { .. } if is_writer_transfer(&entries) => {}
                WriterDecision::Foreign { owner, local } => {
                    return Err(ZyronError::ConflictError {
                        mine: format!("{} from node {:016x}", attempt.operation.name(), local),
                        theirs: format!("writes owned by node {:016x}", owner),
                        reason: "one node writes a dataset and the others read it, so a second writer is refused rather than merged. Transfer ownership explicitly if the owning node is gone"
                            .to_string(),
                    });
                }
            }
            // The read predicate pins the base the caller probed at. Every
            // version that landed after that base is a writer this attempt
            // never met in a version-file collision, so the phantom rule
            // runs against each of them here, exactly as rule R4 runs
            // against a collision winner. A unique probe rides this: the
            // probed key range conflicts with any concurrent commit whose
            // new file could hold the key
            if let Some(predicate) = attempt.read_predicate
                && attempt.read_version != 0
            {
                for v in attempt.read_version + 1..=base_version {
                    let data = read_racing(|| read_version_file(&self.version_path(v)))?;
                    for entry in &data.entries {
                        if let LogEntry::AddFile(file) = entry {
                            if predicate.prune(file) != PruneDecision::CannotMatch {
                                return Err(ZyronError::ConflictError {
                                    mine: attempt.operation.name().to_string(),
                                    theirs: format!("version {}", v),
                                    reason: format!(
                                        "a concurrent commit added partition {:#x} matching \
                                         the read predicate",
                                        file.partition_id
                                    ),
                                });
                            }
                        }
                    }
                }
            }
            let version = base_version + 1;
            let mut state = Some((*base).clone());
            apply_entries(&mut state, version, attempt.timestamp_us, &entries)?;
            let manifest = match state {
                Some(m) => m,
                None => {
                    return Err(ZyronError::Internal(
                        "commit replayed to no manifest".into(),
                    ));
                }
            };
            let bytes = encode_version_file(version, base_version, &attempt, &entries, &base)?;
            let path = self.version_path(version);
            match fs::OpenOptions::new()
                .write(true)
                .create_new(true)
                .open(&path)
            {
                Ok(mut file) => {
                    // A version file that exists but holds no decodable
                    // header stops every later commit, which reads it as the
                    // winner it lost to, and stops `open`, which reads every
                    // header after the newest checkpoint. Recovering from
                    // that needs REPAIR rather than a restart.
                    //
                    // This attempt created the file and holds its only
                    // handle, and the head has not moved, so no reader can
                    // have adopted the version. Unlinking it is therefore
                    // safe and keeps a failed write costing one statement.
                    // A committer that already raced to this path and saw
                    // `AlreadyExists` finds the file gone and takes the
                    // retry it takes for an abandoned version
                    let mut written = match inject_version_write_failure(&path) {
                        Some(injected) => Err(injected),
                        None => file.write_all(&bytes),
                    };
                    if written.is_ok() {
                        written = file.sync_all();
                    }
                    if let Err(e) = written {
                        drop(file);
                        if let Err(unlink) = fs::remove_file(&path) {
                            tracing::error!(
                                target: "zyron::lake",
                                version,
                                path = %path.display(),
                                error = %unlink,
                                "a partly written lake version file could not be removed, the \
                                 table will refuse commits until REPAIR discards it"
                            );
                        }
                        return Err(io_context("write lake version file", &path, e));
                    }
                    drop(file);
                    let _ = self
                        .manifests
                        .insert_sync(version, CachedManifest::new(manifest));
                    self.evict_cache();
                    if attempt.db_txn_id == 0 {
                        self.created_head.fetch_max(version, Ordering::AcqRel);
                        self.advance_published();
                        self.write_latest_hint(self.latest_version());
                        self.mark_for_maintenance();
                    } else {
                        // Pending registration precedes the head advance.
                        // Once the shared head includes this version,
                        // advance_published and concurrent committers act on
                        // it, and only the pending entry tells them its
                        // transaction has not resolved yet
                        let _ = self.pending.insert_sync(version, attempt.db_txn_id);
                        #[cfg(test)]
                        stall_visibility_gap();
                        self.created_head.fetch_max(version, Ordering::AcqRel);
                    }
                    return Ok(version);
                }
                Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => {
                    self.commit_retries.fetch_add(1, Ordering::Relaxed);
                    let winner = match read_racing(|| read_commit_header(&path)) {
                        Ok(w) => w,
                        Err(_) if !path.exists() => {
                            // The winner abandoned before resolving, its
                            // version number is free again and nothing
                            // conflicts with this attempt
                            races += 1;
                            backoff(races.min(u32::MAX as u64) as u32);
                            continue;
                        }
                        Err(e) => return Err(e),
                    };
                    if winner.db_txn_id == 0 {
                        // A standalone winner is committed by construction
                        // and can never be abandoned, so its version is
                        // adopted directly. This is also how a version
                        // written through another handle of the same
                        // directory reaches this instance's head
                        self.created_head
                            .fetch_max(winner.version, Ordering::AcqRel);
                        self.advance_published();
                    } else {
                        // A transactional winner registers its version
                        // pending before advancing the head itself, both in
                        // this instance's memory. Adopting the version from
                        // the file alone would make it discoverable before
                        // that registration, and writing the head here
                        // could re-raise a version an abandon just rolled
                        // back, so this waits for the winner's own advance
                        // or for the file to vanish when it abandons
                        let mut spins: u32 = 0;
                        loop {
                            if self.head_version() >= winner.version
                                || self.latest_version() >= winner.version
                            {
                                break;
                            }
                            if !path.exists() {
                                break;
                            }
                            if spins >= MAX_COMMIT_ATTEMPTS {
                                return Err(ZyronError::ConflictError {
                                    mine: attempt.operation.name().to_string(),
                                    theirs: winner.operation.name().to_string(),
                                    reason: format!(
                                        "waited {} attempts for the winning transaction's \
                                         version {} to resolve",
                                        MAX_COMMIT_ATTEMPTS, winner.version
                                    ),
                                });
                            }
                            backoff(spins);
                            spins += 1;
                        }
                        if !path.exists()
                            && self.head_version() < winner.version
                            && self.latest_version() < winner.version
                        {
                            // The winner was abandoned before it was
                            // adopted, so nothing conflicts with this
                            // attempt
                            races += 1;
                            backoff(races.min(u32::MAX as u64) as u32);
                            continue;
                        }
                    }
                    if let Some(reason) = self.check_conflict(&attempt, &entries, &winner, &path)? {
                        return Err(ZyronError::ConflictError {
                            mine: attempt.operation.name().to_string(),
                            theirs: winner.operation.name().to_string(),
                            reason,
                        });
                    }
                    // Losing is routine under concurrency and is not an
                    // error, so it is traced rather than surfaced
                    tracing::trace!(
                        target: "zyron::lake",
                        lost_version = version,
                        winner = winner.operation.name(),
                        "commit lost a version race, rebuilding on the winner"
                    );
                    races += 1;
                    backoff(races.min(u32::MAX as u64) as u32);
                }
                Err(e) => return Err(io_context("create lake version file", &path, e)),
            }
        }
    }

    /// Evaluates rules R1 to R8, header first, entries only when a rule
    /// needs them. Returns the conflict reason or None
    fn check_conflict(
        &self,
        attempt: &CommitAttempt<'_>,
        my_entries: &[LogEntry],
        winner: &CommitHeader,
        winner_path: &Path,
    ) -> Result<Option<String>, ZyronError> {
        // R7 a vacuum recomputes from any base
        if attempt.operation == OperationKind::Vacuum {
            return Ok(None);
        }
        // R1 and R2 barrier operations conflict unconditionally
        if attempt.operation.is_barrier() || winner.operation.is_barrier() {
            return Ok(Some("schema change, restore and convert serialize".into()));
        }
        // R6 two appends never touch the same file
        if attempt.operation == OperationKind::Append && winner.operation == OperationKind::Append {
            return Ok(None);
        }
        // R3 removed sets, bloom first, exact entries only on a hit
        let my_removed: Vec<u64> = my_entries
            .iter()
            .filter_map(|e| match e {
                LogEntry::RemoveFile { partition_id } => Some(*partition_id),
                _ => None,
            })
            .collect();
        let my_bloom = removed_bloom(my_removed.iter().copied());
        if my_bloom & winner.removed_partition_bloom != 0 {
            let winner_data = read_racing(|| read_version_file(winner_path))?;
            for entry in &winner_data.entries {
                if let LogEntry::RemoveFile { partition_id } = entry {
                    if my_removed.contains(partition_id) {
                        return Ok(Some(format!("both removed partition {:#x}", partition_id)));
                    }
                }
            }
        }
        // R5 an optimize over a disjoint file set inherits on retry
        if attempt.operation == OperationKind::Optimize
            && matches!(
                winner.operation,
                OperationKind::Delete | OperationKind::Update | OperationKind::Merge
            )
        {
            return Ok(None);
        }
        // R4 phantom write, the winner added a file my read predicate
        // may match
        if let Some(read_predicate) = attempt.read_predicate {
            if winner.files_added > 0 {
                let winner_data = read_racing(|| read_version_file(winner_path))?;
                for entry in &winner_data.entries {
                    if let LogEntry::AddFile(file) = entry {
                        if read_predicate.prune(file) != PruneDecision::CannotMatch {
                            return Ok(Some(format!(
                                "concurrent {} added partition {:#x} matching the read predicate",
                                winner.operation.name(),
                                file.partition_id
                            )));
                        }
                    }
                }
            }
        }
        // R8 property writes conflict on intersecting keys
        if attempt.operation == OperationKind::SetProperty
            && winner.operation == OperationKind::SetProperty
        {
            let winner_data = read_racing(|| read_version_file(winner_path))?;
            for entry in my_entries {
                if let LogEntry::SetProperty { key, .. } = entry {
                    for theirs in &winner_data.entries {
                        if let LogEntry::SetProperty { key: their_key, .. } = theirs {
                            if key == their_key {
                                return Ok(Some(format!("both set property \"{}\"", key)));
                            }
                        }
                    }
                }
            }
        }
        Ok(None)
    }

    /// Makes a pending version visible. Visibility advances only through
    /// the contiguous published prefix, a later version never becomes
    /// readable while an earlier one is still pending
    pub fn publish(&self, version: u64) -> Result<(), ZyronError> {
        if self.pending.remove_sync(&version).is_none() {
            return Err(ZyronError::Internal(format!(
                "lake version {} is not pending",
                version
            )));
        }
        self.advance_published();
        self.write_latest_hint(self.latest_version());
        self.mark_for_maintenance();
        Ok(())
    }

    /// Tells the node's maintenance signal that this head moved.
    ///
    /// Every decision background maintenance makes reads the manifest, and
    /// the manifest is a function of the published version, so this is the
    /// only moment at which any of those decisions can change. Recording it
    /// is what lets a worker leave an untouched table alone instead of
    /// rebuilding its manifest on a timer to find out nothing happened.
    ///
    /// Called once per published version rather than once per commit: a
    /// version created inside a database transaction is not visible until
    /// that transaction publishes it, and maintenance reads what is visible
    fn mark_for_maintenance(&self) {
        let Some(table_id) = self.paths.table_id() else {
            // A log outside a table directory has no catalog entry, so no
            // worker could resolve it back to a table to maintain
            return;
        };
        crate::maintenance_signal::maintenance_signal().mark(
            self.registry_key(),
            table_id,
            self.branch.is_none(),
        );
    }

    fn advance_published(&self) {
        loop {
            let cur = self.published.load(Ordering::Acquire);
            let next = cur + 1;
            if next > self.head_version() || self.pending.read_sync(&next, |_, _| ()).is_some() {
                return;
            }
            if self
                .published
                .compare_exchange(cur, next, Ordering::AcqRel, Ordering::Acquire)
                .is_err()
            {
                // Another publisher advanced, re-evaluate from its result
                continue;
            }
        }
    }

    /// Discards a pending version after its transaction aborted. Only the
    /// newest created version can be abandoned, later versions would have
    /// waited on it rather than build over it
    pub fn abandon(&self, version: u64) -> Result<(), ZyronError> {
        if self.pending.read_sync(&version, |_, _| ()).is_none() {
            return Err(ZyronError::Internal(format!(
                "lake version {} is not pending",
                version
            )));
        }
        // Retraction reverses registration: the head rolls back before the
        // registration and the files go away, so a committer that saw the
        // version through the head can still resolve it, and one that finds
        // the files gone always observes the moved head. The exchange also
        // claims the abandon, a concurrent second abandon fails the guard
        if self
            .created_head
            .compare_exchange(version, version - 1, Ordering::AcqRel, Ordering::Acquire)
            .is_err()
        {
            return Err(ZyronError::Internal(format!(
                "abandon of version {} but the head is {}",
                version,
                self.head_version()
            )));
        }
        if self.pending.remove_sync(&version).is_none() {
            return Err(ZyronError::Internal(format!(
                "lake version {} is not pending",
                version
            )));
        }
        let _ = self.manifests.remove_sync(&version);
        fs::remove_file(self.version_path(version))?;
        Ok(())
    }

    /// Writes a manifest checkpoint for a published version, the random
    /// access point replay starts from
    pub fn checkpoint(&self, version: u64) -> Result<(), ZyronError> {
        // Checkpoints and version GC belong to the table's main log. A
        // branch's versions are reclaimed when the branch is dropped, so
        // running either here would rewrite main's files from a branch head
        if let Some(name) = &self.branch {
            return Err(ZyronError::BranchConflict(format!(
                "branch \"{}\" does not checkpoint, the table's main log does",
                name
            )));
        }
        if version > self.latest_version() {
            return Err(ZyronError::Internal(format!(
                "checkpoint of unpublished version {}",
                version
            )));
        }
        let target = self.paths.checkpoint_file(version);
        if target.exists() {
            return Ok(());
        }
        let manifest = self.manifest_at(version)?;
        let bytes = manifest.encode();
        let tmp = self
            .paths
            .tmp_dir()
            .join(format!("checkpoint_{}.zym", version));
        fs::create_dir_all(self.paths.tmp_dir())?;
        {
            let mut file = fs::File::create(&tmp)?;
            file.write_all(&bytes)?;
            file.sync_all()?;
        }
        if let Err(e) = fs::rename(&tmp, &target) {
            // A concurrent checkpointer winning the rename is success
            let _ = fs::remove_file(&tmp);
            if !target.exists() {
                return Err(e.into());
            }
        }
        let hint = self.paths.last_checkpoint_hint();
        if let Err(e) = fs::write(&hint, version.to_string()) {
            tracing::warn!(error = %e, "checkpoint hint write failed, hint is advisory");
        }
        Ok(())
    }

    /// Deletes version files no longer needed to reconstruct any version
    /// at or above the retention floor. Returns how many files were
    /// removed. Nothing is deleted unless a checkpoint at or below the
    /// floor exists to replay from
    pub fn gc_versions(&self, retain_min_version: u64) -> Result<usize, ZyronError> {
        if let Some(name) = &self.branch {
            return Err(ZyronError::BranchConflict(format!(
                "branch \"{}\" does not collect versions, dropping it reclaims them",
                name
            )));
        }
        let mut checkpoints = Vec::new();
        let mut versions = Vec::new();
        for dirent in fs::read_dir(self.paths.log_dir())? {
            let dirent = dirent?;
            let name = dirent.file_name();
            let Some(name) = name.to_str() else { continue };
            match parse_version_file_name(name) {
                Some((v, VersionFileKind::Checkpoint)) => checkpoints.push(v),
                Some((v, VersionFileKind::Version)) => versions.push(v),
                None => {}
            }
        }
        let Some(base) = checkpoints
            .iter()
            .copied()
            .filter(|&c| c <= retain_min_version)
            .max()
        else {
            return Ok(0);
        };
        let mut removed = 0usize;
        for v in versions.into_iter().filter(|&v| v <= base) {
            fs::remove_file(self.paths.version_file(v))?;
            let _ = self.manifests.remove_sync(&v);
            removed += 1;
        }
        for c in checkpoints.into_iter().filter(|&c| c < base) {
            fs::remove_file(self.paths.checkpoint_file(c))?;
            removed += 1;
        }
        Ok(removed)
    }

    fn write_latest_hint(&self, version: u64) {
        // A branch numbers its versions above the fork point, so its head is
        // a version main does not have. Each head writes its own hint
        let hint = match &self.branch {
            Some(name) => self.paths.branch_latest_hint(name),
            None => self.paths.latest_hint(),
        };
        if let Err(e) = fs::write(&hint, version.to_string()) {
            tracing::warn!(error = %e, "latest hint write failed, hint is advisory");
        }
    }
}

/// Sleeps for a random span up to the exponential ceiling for this attempt.
///
/// The randomness is the load-bearing part. A deterministic backoff wakes
/// every loser of a race at the same instant, so they collide again and the
/// same writer keeps losing: eight appenders under it produced sixteen
/// consecutive losses for one of them. Spreading the wake times across the
/// interval breaks that synchronization, which is what makes repeated loss
/// exponentially unlikely rather than routine.
///
/// The generator is per thread and seeded once from a global counter, so it
/// costs no syscall and no shared state on a path taken only after a commit
/// has already lost a race
fn backoff(attempt: u32) {
    let ceiling = (BACKOFF_BASE_US << attempt.min(16)).min(BACKOFF_CAP_US);
    let us = JITTER.with(|cell| {
        let mut rng = cell.borrow_mut();
        rng.nextU64() % (ceiling + 1)
    });
    std::thread::sleep(Duration::from_micros(us));
}

thread_local! {
    static JITTER: std::cell::RefCell<zyron_common::Xoshiro256pp> = {
        static SEED: AtomicU64 = AtomicU64::new(0x243F_6A88_85A3_08D3);
        let mut seed = SEED.fetch_add(0x9E37_79B9_7F4A_7C15, Ordering::Relaxed);
        std::cell::RefCell::new(zyron_common::Xoshiro256pp::fromSeed(
            zyron_common::splitMix64(&mut seed),
        ))
    };
}

fn read_version_file(path: &Path) -> Result<VersionFileData, ZyronError> {
    let bytes = fs::read(path).map_err(|e| io_context("read lake version file", path, e))?;
    VersionFileData::decode(&bytes, &path.to_string_lossy())
}

// How long a loser waits for the winner's in-flight write to become fully
// readable, far above any write_all latency
const RACING_READ_ATTEMPTS: u32 = 200;
const RACING_READ_PAUSE: Duration = Duration::from_millis(1);

/// A conflict check can observe the winner's file between create_new and
/// write_all. A short partial read here is the race, not corruption, so
/// retry briefly before treating the error as real
fn read_racing<T>(mut read: impl FnMut() -> Result<T, ZyronError>) -> Result<T, ZyronError> {
    let mut last = None;
    for _ in 0..RACING_READ_ATTEMPTS {
        match read() {
            Ok(v) => return Ok(v),
            Err(e) if is_possible_partial_write(&e) => {
                last = Some(e);
                std::thread::sleep(RACING_READ_PAUSE);
            }
            Err(e) => return Err(e),
        }
    }
    match last {
        Some(e) => Err(e),
        None => Err(ZyronError::Internal("racing read produced no error".into())),
    }
}

fn is_possible_partial_write(e: &ZyronError) -> bool {
    match e {
        ZyronError::Io(io) => io.kind() == std::io::ErrorKind::UnexpectedEof,
        ZyronError::ManifestCorrupted { .. } => true,
        _ => false,
    }
}

/// Serializes one version file. Statistics in the header are derived from
/// the entries against the base manifest the commit built on
fn encode_version_file(
    version: u64,
    read_version: u64,
    attempt: &CommitAttempt<'_>,
    entries: &[LogEntry],
    base: &ManifestFile,
) -> Result<Vec<u8>, ZyronError> {
    if entries.len() > u32::MAX as usize {
        return Err(ZyronError::Internal(
            "too many entries in one commit".into(),
        ));
    }
    let mut files_added = 0u32;
    let mut files_removed = 0u32;
    let mut rows_added = 0u64;
    let mut rows_removed = 0u64;
    let mut bytes_added = 0u64;
    let mut removed_ids = Vec::new();
    for entry in entries {
        match entry {
            LogEntry::AddFile(file) => {
                files_added += 1;
                rows_added = rows_added.saturating_add(file.row_count);
                bytes_added = bytes_added.saturating_add(file.size_bytes);
            }
            LogEntry::RemoveFile { partition_id } => {
                files_removed += 1;
                removed_ids.push(*partition_id);
                let removed = base.entry_for(*partition_id).ok_or_else(|| {
                    ZyronError::Internal(format!(
                        "commit removes partition {:#x} absent from its base",
                        partition_id
                    ))
                })?;
                rows_removed = rows_removed.saturating_add(removed.row_count);
            }
            _ => {}
        }
    }
    let mut body = Vec::with_capacity(256 * entries.len());
    for entry in entries {
        entry.encode_into(&mut body)?;
    }
    let audit = attempt.audit;
    let mut audit_off = 0u32;
    let mut audit_len = 0u32;
    if let Some(info) = audit {
        let start = body.len();
        info.encode_into(&mut body)?;
        audit_off = (COMMIT_HEADER_LEN + start) as u32;
        audit_len = (body.len() - start) as u32;
    }
    let header = CommitHeader {
        version,
        read_version,
        db_txn_id: attempt.db_txn_id,
        commit_lsn: attempt.commit_lsn,
        timestamp_us: attempt.timestamp_us,
        operation: attempt.operation,
        entry_count: entries.len() as u32,
        files_added,
        files_removed,
        rows_added,
        rows_removed,
        bytes_added,
        removed_partition_bloom: removed_bloom(removed_ids.into_iter()),
        read_predicate_hash: attempt
            .read_predicate
            .map(|p| p.stable_hash().max(1))
            .unwrap_or(0),
        audit_off,
        audit_len,
    };
    let entry_crc = crc32fast::hash(&body);
    let mut out = Vec::with_capacity(COMMIT_HEADER_LEN + body.len());
    out.extend_from_slice(&header.encode(entry_crc));
    out.extend_from_slice(&body);
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::manifest::ColumnStatsEntry;
    use crate::predicate::{ColumnBounds, CompareOp, LakeValue};
    use crate::schema::LakeColumn;
    use std::thread;
    use zyron_common::TypeId;

    fn schema() -> LakeSchema {
        LakeSchema::new(
            1,
            vec![LakeColumn {
                id: 0,
                name: "id".into(),
                type_id: TypeId::Int64,
                nullable: false,
                fractional_digits: None,
                tz_offset_secs: None,
                max_length: None,
                default_expr: None,
            }],
        )
        .expect("valid schema")
    }

    fn data_file(partition_id: u64, min: i64, max: i64, rows: u64) -> PartitionEntry {
        PartitionEntry {
            partition_id,
            size_bytes: rows * 8,
            row_count: rows,
            added_version: 0,
            cluster_spec_id: 0,
            column_stats: std::sync::Arc::new(vec![ColumnStatsEntry {
                ndv: Some(rows),
                column_id: 0,
                bounds: ColumnBounds {
                    min: Some(LakeValue::Int(min)),
                    max: Some(LakeValue::Int(max)),
                    null_count: 0,
                    row_count: rows,
                },
                bloom: None,
                size_bytes: None,
            }]),
            delete_predicate_ids: vec![],
        }
    }

    fn attempt(op: OperationKind) -> CommitAttempt<'static> {
        CommitAttempt {
            operation: op,
            db_txn_id: 0,
            commit_lsn: 100,
            timestamp_us: 1_754_700_000_000_000,
            read_predicate: None,
            read_version: 0,
            audit: None,
        }
    }

    fn new_log(dir: &Path) -> TransactionLog {
        TransactionLog::create(
            LakePaths::new(dir, 7),
            attempt(OperationKind::SchemaChange),
            &schema(),
            None,
            &BTreeMap::from([("target_file_size".to_string(), "268435456".to_string())]),
        )
        .expect("create")
    }

    #[test]
    fn test_create_append_and_read_back() {
        let dir = tempfile::tempdir().expect("tempdir");
        let log = new_log(dir.path());
        assert_eq!(log.latest_version(), 1);

        let v = log
            .commit(attempt(OperationKind::Append), |_| {
                Ok(vec![LogEntry::AddFile(data_file(0x10, 1, 100, 50))])
            })
            .expect("append");
        assert_eq!(v, 2);
        assert_eq!(log.latest_version(), 2);

        let m = log.latest_manifest().expect("manifest");
        assert_eq!(m.snapshot_id, 2);
        assert_eq!(m.entries.len(), 1);
        assert_eq!(m.entries[0].added_version, 2);
        assert_eq!(
            m.properties.get("target_file_size").map(|s| s.as_str()),
            Some("268435456")
        );

        let header =
            read_commit_header(&LakePaths::new(dir.path(), 7).version_file(2)).expect("header");
        assert_eq!(header.version, 2);
        assert_eq!(header.operation, OperationKind::Append);
        assert_eq!(header.files_added, 1);
        assert_eq!(header.rows_added, 50);
        assert_eq!(header.removed_partition_bloom, 0);
    }

    #[test]
    fn test_concurrent_commits_serialize_via_create_new() {
        let dir = tempfile::tempdir().expect("tempdir");
        let log = Arc::new(new_log(dir.path()));
        let threads = 8;
        let per_thread = 5;
        let mut handles = Vec::new();
        for t in 0..threads {
            let log = Arc::clone(&log);
            handles.push(thread::spawn(move || {
                for i in 0..per_thread {
                    let id = (t as u64) * 1000 + i as u64;
                    log.commit(attempt(OperationKind::Append), |_| {
                        Ok(vec![LogEntry::AddFile(data_file(id, 0, 10, 1))])
                    })
                    .expect("append");
                }
            }));
        }
        for h in handles {
            h.join().expect("thread");
        }
        let expected_head = 1 + (threads * per_thread) as u64;
        assert_eq!(log.latest_version(), expected_head);
        let m = log.latest_manifest().expect("manifest");
        assert_eq!(m.entries.len(), threads * per_thread);
        // Every version number exists exactly once, no gaps
        let paths = LakePaths::new(dir.path(), 7);
        for v in 1..=expected_head {
            assert!(paths.version_file(v).exists(), "version {} missing", v);
        }
        assert!(
            log.commit_retries() > 0,
            "contention must have caused retries"
        );
    }

    /// A transactional commit must be registered pending before its version
    /// becomes discoverable through the shared head. In the reverse order a
    /// concurrent standalone commit sees the head advance with no pending
    /// entry, builds on the uncommitted manifest and publishes past it, so
    /// every scan dirty-reads the uncommitted version and the transaction
    /// can neither publish nor abandon it afterwards. The stall widens the
    /// registration gap so the interleaving is reachable, and the sound
    /// ordering must hold the invariants for every iteration regardless
    #[test]
    fn test_transactional_version_is_pending_before_it_is_discoverable() {
        let dir = tempfile::tempdir().expect("tempdir");
        let log = Arc::new(new_log(dir.path()));
        TEST_STALL_VISIBILITY_GAP_US.store(2_000, Ordering::Relaxed);
        let mut expected_rows: u64 = 0;
        for i in 0..200u64 {
            let txn_attempt = CommitAttempt {
                db_txn_id: 7,
                ..attempt(OperationKind::Append)
            };
            let a = Arc::clone(&log);
            let ta = thread::spawn(move || {
                a.commit(txn_attempt, |_| {
                    Ok(vec![LogEntry::AddFile(data_file(0x1000 + i, 0, 10, 1))])
                })
            });
            let b = Arc::clone(&log);
            let tb = thread::spawn(move || {
                b.commit(attempt(OperationKind::Append), |_| {
                    Ok(vec![LogEntry::AddFile(data_file(0x2000 + i, 0, 10, 1))])
                })
            });
            let va = ta
                .join()
                .expect("transactional thread")
                .expect("transactional append");
            // The standalone committer must not observe the version before
            // its registration, so publication never crosses it while it is
            // unresolved. Resolving promptly also clears the standalone
            // committer's bounded wait on the pending entry
            if i % 2 == 0 {
                log.abandon(va).unwrap_or_else(|e| {
                    panic!(
                        "iteration {i}: the uncommitted version {va} escaped its \
                         transaction and cannot be abandoned: {e}"
                    )
                });
            } else {
                log.publish(va).expect("publish the transactional version");
                expected_rows += 1;
            }
            tb.join()
                .expect("standalone thread")
                .expect("standalone append");
            expected_rows += 1;
            let live: u64 = log
                .latest_manifest()
                .expect("manifest")
                .entries
                .iter()
                .map(|e| e.row_count)
                .sum();
            assert_eq!(
                live, expected_rows,
                "iteration {i}: published rows diverged, an uncommitted or \
                 abandoned version leaked into the published prefix"
            );
        }
        TEST_STALL_VISIBILITY_GAP_US.store(0, Ordering::Relaxed);
    }

    /// A commit that declares what it read pins the version it read at.
    /// Every version committed after that base is a writer the attempt
    /// never collided with in the version-file namespace, so the phantom
    /// rule has to run against each of them at commit time. A unique-key
    /// probe passes its key range this way: without the gap check, two
    /// writers who both probed an empty range and then commit in sequence
    /// both succeed, and the duplicate lands silently
    #[test]
    fn test_read_predicate_pins_the_probe_base_across_the_gap() {
        let dir = tempfile::tempdir().expect("tempdir");
        let log = new_log(dir.path());
        let probed_version = log.head_version();

        // Another writer lands keys [10, 20] after the probe
        log.commit(attempt(OperationKind::Append), |_| {
            Ok(vec![LogEntry::AddFile(data_file(0xA1, 10, 20, 5))])
        })
        .expect("concurrent append");

        // This attempt probed [15, 25] against the pre-append base, the
        // concurrent file intersects the range so the commit must conflict
        let overlapping = LakePredicate::And(vec![
            LakePredicate::Compare {
                column_id: 0,
                op: CompareOp::GtEq,
                value: LakeValue::Int(15),
            },
            LakePredicate::Compare {
                column_id: 0,
                op: CompareOp::LtEq,
                value: LakeValue::Int(25),
            },
        ]);
        let mut pinned = attempt(OperationKind::Append);
        pinned.read_predicate = Some(&overlapping);
        pinned.read_version = probed_version;
        let err = log
            .commit(pinned, |_| {
                Ok(vec![LogEntry::AddFile(data_file(0xA2, 15, 15, 1))])
            })
            .expect_err("the gap write intersects the probed range");
        assert!(
            matches!(err, ZyronError::ConflictError { .. }),
            "expected a conflict, got {err:?}"
        );

        // A probe whose range the gap write cannot touch commits cleanly
        let disjoint = LakePredicate::And(vec![
            LakePredicate::Compare {
                column_id: 0,
                op: CompareOp::GtEq,
                value: LakeValue::Int(100),
            },
            LakePredicate::Compare {
                column_id: 0,
                op: CompareOp::LtEq,
                value: LakeValue::Int(200),
            },
        ]);
        let mut clean = attempt(OperationKind::Append);
        clean.read_predicate = Some(&disjoint);
        clean.read_version = probed_version;
        log.commit(clean, |_| {
            Ok(vec![LogEntry::AddFile(data_file(0xA3, 100, 100, 1))])
        })
        .expect("a disjoint range does not conflict");
    }

    /// The invariant every observer relies on: a transactional version that
    /// is discoverable through the shared head is already registered
    /// pending. The stall sits between the two steps, so if the head ever
    /// advances first there is a wide window in which this fails
    #[test]
    fn test_head_advance_implies_pending_registration() {
        let dir = tempfile::tempdir().expect("tempdir");
        let log = Arc::new(new_log(dir.path()));
        TEST_STALL_VISIBILITY_GAP_US.store(5_000, Ordering::Relaxed);
        let base = log.head_version();
        let a = Arc::clone(&log);
        let ta = thread::spawn(move || {
            a.commit(
                CommitAttempt {
                    db_txn_id: 9,
                    ..attempt(OperationKind::Append)
                },
                |_| Ok(vec![LogEntry::AddFile(data_file(0x9000, 0, 10, 1))]),
            )
        });
        while log.head_version() <= base {
            std::hint::spin_loop();
        }
        let v = log.head_version();
        let registered = log.pending.read_sync(&v, |_, _| ()).is_some();
        assert!(
            registered,
            "version {v} is discoverable through the head but not registered pending"
        );
        let va = ta.join().expect("transactional thread").expect("append");
        assert_eq!(va, v);
        log.abandon(va).expect("abandon");
        TEST_STALL_VISIBILITY_GAP_US.store(0, Ordering::Relaxed);
    }

    /// Losing a version race is not a conflict, so it must not spend the
    /// retry budget that exists for states which may never resolve.
    ///
    /// Eight appenders at twenty commits each used to exhaust sixteen
    /// attempts and fail an operation that conflicts with nothing, because
    /// a deterministic backoff woke every loser together and the same
    /// writer kept losing. The load here is four times the older test's on
    /// purpose: that is where it broke
    #[test]
    fn test_sustained_concurrent_appends_never_exhaust_the_retry_budget() {
        let dir = tempfile::tempdir().expect("tempdir");
        let log = Arc::new(new_log(dir.path()));
        let threads = 8;
        let per_thread = 20;
        let mut handles = Vec::new();
        for t in 0..threads {
            let log = Arc::clone(&log);
            handles.push(thread::spawn(move || {
                for i in 0..per_thread {
                    let id = (t as u64) * 1000 + i as u64;
                    log.commit(attempt(OperationKind::Append), |_| {
                        Ok(vec![LogEntry::AddFile(data_file(id, 0, 10, 1))])
                    })
                    .unwrap_or_else(|e| {
                        panic!(
                            "append {} from writer {} failed, but two appends never conflict: {e}",
                            i, t
                        )
                    });
                }
            }));
        }
        for h in handles {
            h.join().expect("thread");
        }

        let commits = (threads * per_thread) as u64;
        assert_eq!(log.latest_version(), commits + 1);
        let m = log.latest_manifest().expect("manifest");
        assert_eq!(m.entries.len() as usize, threads * per_thread);
        // Every version exists exactly once, so no commit was lost and none
        // reused a number
        let paths = LakePaths::new(dir.path(), 7);
        for v in 1..=commits + 1 {
            assert!(paths.version_file(v).exists(), "version {} missing", v);
        }
        assert!(
            log.commit_retries() > 0,
            "nothing was contended, so this measured nothing"
        );
    }

    #[test]
    fn test_barrier_operations_conflict() {
        let dir = tempfile::tempdir().expect("tempdir");
        let log = new_log(dir.path());
        // A second handle over the same directory simulates another writer
        let stale =
            TransactionLog::open(LakePaths::new(dir.path(), 7), &AllCommitted).expect("open");
        let mut evolved = schema();
        evolved.schema_id = 2;
        log.commit(attempt(OperationKind::SchemaChange), |_| {
            Ok(vec![LogEntry::SchemaChange(evolved.clone())])
        })
        .expect("schema change");

        let err = stale
            .commit(attempt(OperationKind::Append), |_| {
                Ok(vec![LogEntry::AddFile(data_file(0x99, 0, 1, 1))])
            })
            .expect_err("must conflict");
        match err {
            ZyronError::ConflictError { mine, theirs, .. } => {
                assert_eq!(mine, "APPEND");
                assert_eq!(theirs, "SCHEMA CHANGE");
            }
            other => panic!("unexpected error {:?}", other),
        }
    }

    #[test]
    fn test_disjoint_removes_retry_and_both_apply() {
        let dir = tempfile::tempdir().expect("tempdir");
        let log = new_log(dir.path());
        log.commit(attempt(OperationKind::Append), |_| {
            Ok(vec![
                LogEntry::AddFile(data_file(0xA, 1, 100, 10)),
                LogEntry::AddFile(data_file(0xB, 101, 200, 10)),
            ])
        })
        .expect("append");

        let stale =
            TransactionLog::open(LakePaths::new(dir.path(), 7), &AllCommitted).expect("open");
        log.commit(attempt(OperationKind::Delete), |_| {
            Ok(vec![LogEntry::RemoveFile { partition_id: 0xA }])
        })
        .expect("remove A");

        stale
            .commit(attempt(OperationKind::Delete), |_| {
                Ok(vec![LogEntry::RemoveFile { partition_id: 0xB }])
            })
            .expect("disjoint remove retries and lands");
        assert!(stale.commit_retries() > 0);
        let m = stale.latest_manifest().expect("manifest");
        assert!(m.entries.is_empty(), "both files removed");
    }

    #[test]
    fn test_same_remove_conflicts() {
        let dir = tempfile::tempdir().expect("tempdir");
        let log = new_log(dir.path());
        log.commit(attempt(OperationKind::Append), |_| {
            Ok(vec![LogEntry::AddFile(data_file(0xA, 1, 100, 10))])
        })
        .expect("append");
        let stale =
            TransactionLog::open(LakePaths::new(dir.path(), 7), &AllCommitted).expect("open");
        log.commit(attempt(OperationKind::Delete), |_| {
            Ok(vec![LogEntry::RemoveFile { partition_id: 0xA }])
        })
        .expect("remove");

        let err = stale
            .commit(attempt(OperationKind::Delete), |_| {
                Ok(vec![LogEntry::RemoveFile { partition_id: 0xA }])
            })
            .expect_err("same remove must conflict");
        assert!(err.to_string().contains("both removed"));
    }

    #[test]
    fn test_phantom_write_detected_by_read_predicate() {
        let dir = tempfile::tempdir().expect("tempdir");
        let log = new_log(dir.path());
        log.commit(attempt(OperationKind::Append), |_| {
            Ok(vec![LogEntry::AddFile(data_file(0xA, 1, 100, 10))])
        })
        .expect("append");

        // The deleter read "id <= 100" and the concurrent append adds a
        // file inside that range
        let read = LakePredicate::Compare {
            column_id: 0,
            op: CompareOp::LtEq,
            value: LakeValue::Int(100),
        };
        let stale =
            TransactionLog::open(LakePaths::new(dir.path(), 7), &AllCommitted).expect("open");
        log.commit(attempt(OperationKind::Append), |_| {
            Ok(vec![LogEntry::AddFile(data_file(0xC, 1, 50, 5))])
        })
        .expect("phantom append");

        let mut with_read = attempt(OperationKind::Delete);
        with_read.read_predicate = Some(&read);
        let err = stale
            .commit(with_read, |_| {
                Ok(vec![LogEntry::RemoveFile { partition_id: 0xA }])
            })
            .expect_err("phantom must conflict");
        assert!(err.to_string().contains("read predicate"));

        // The same shape with a non-matching added file goes through
        let log2 =
            TransactionLog::open(LakePaths::new(dir.path(), 7), &AllCommitted).expect("open");
        let stale2 =
            TransactionLog::open(LakePaths::new(dir.path(), 7), &AllCommitted).expect("open");
        log2.commit(attempt(OperationKind::Append), |_| {
            Ok(vec![LogEntry::AddFile(data_file(0xD, 200, 300, 5))])
        })
        .expect("append outside the range");
        let mut with_read2 = attempt(OperationKind::Delete);
        with_read2.read_predicate = Some(&read);
        stale2
            .commit(with_read2, |_| {
                Ok(vec![LogEntry::RemoveFile { partition_id: 0xC }])
            })
            .expect("non-matching append does not conflict");
    }

    #[test]
    fn test_pending_publish_and_abandon() {
        let dir = tempfile::tempdir().expect("tempdir");
        let log = new_log(dir.path());

        let mut txn_commit = attempt(OperationKind::Append);
        txn_commit.db_txn_id = 42;
        let v = log
            .commit(txn_commit, |_| {
                Ok(vec![LogEntry::AddFile(data_file(0x1, 0, 9, 3))])
            })
            .expect("pending commit");
        assert_eq!(v, 2);
        assert_eq!(log.latest_version(), 1, "pending version is invisible");
        assert_eq!(log.head_version(), 2);

        log.publish(v).expect("publish");
        assert_eq!(log.latest_version(), 2);

        let mut txn2 = attempt(OperationKind::Append);
        txn2.db_txn_id = 43;
        let v3 = log
            .commit(txn2, |_| {
                Ok(vec![LogEntry::AddFile(data_file(0x2, 0, 9, 3))])
            })
            .expect("second pending");
        log.abandon(v3).expect("abandon");
        assert_eq!(log.head_version(), 2);
        assert_eq!(log.latest_version(), 2);
        assert!(!LakePaths::new(dir.path(), 7).version_file(v3).exists());

        // The abandoned version number is reusable
        let v3b = log
            .commit(attempt(OperationKind::Append), |_| {
                Ok(vec![LogEntry::AddFile(data_file(0x3, 0, 9, 3))])
            })
            .expect("recommit");
        assert_eq!(v3b, 3);
    }

    /// Background maintenance is woken by commits rather than by a clock,
    /// so a published version that failed to reach the signal is a table
    /// that silently stops being maintained. A version still pending inside
    /// a database transaction is not visible to a reader yet, so it is
    /// announced when it publishes and not before
    #[test]
    fn test_a_published_version_tells_maintenance_its_head_moved() {
        let signal = crate::maintenance_signal::maintenance_signal();
        let dir = tempfile::tempdir().expect("tempdir");
        let log = new_log(dir.path());
        let key = log.registry_key();

        let before = signal.generation();
        log.commit(attempt(OperationKind::Append), |_| {
            Ok(vec![LogEntry::AddFile(data_file(0x11, 0, 9, 3))])
        })
        .expect("append");
        assert!(
            signal.generation() > before,
            "a standalone commit publishes immediately, so it announces immediately"
        );
        assert!(signal.is_marked(&key), "and it names the head that moved");

        // A transactional version is created but not readable, so nothing
        // maintenance would decide has changed yet
        signal.forget_under(log.paths().root());
        let mut txn = attempt(OperationKind::Append);
        txn.db_txn_id = 77;
        let pending = log
            .commit(txn, |_| {
                Ok(vec![LogEntry::AddFile(data_file(0x12, 0, 9, 3))])
            })
            .expect("pending commit");
        assert!(
            !signal.is_marked(&key),
            "an unpublished version is not something maintenance can act on"
        );

        log.publish(pending).expect("publish");
        assert!(
            signal.is_marked(&key),
            "publishing it is what makes it maintenance's business"
        );

        // Dropping the table takes its marks with it, so a head nothing can
        // resolve does not hold a slot against the signal's bound
        TransactionLog::remove_shared(log.paths());
        assert!(!signal.is_marked(&key));
    }

    /// A version file that exists and holds no decodable header is worse
    /// than a failed commit: every later commit reads it as the winner it
    /// lost to, and `open` reads every header after the newest checkpoint,
    /// so the table refuses writes and refuses to open until REPAIR
    /// discards it. A write that fails has to take its own file with it.
    ///
    /// The error also has to say which file and which operation. A failure
    /// this rare gives one observation, and an `Io` carrying neither spends
    /// it
    #[test]
    fn test_a_failed_version_write_unlinks_its_own_file() {
        let dir = tempfile::tempdir().expect("tempdir");
        let log = new_log(dir.path());
        let base = log.latest_version();
        let claimed = log.paths().version_file(base + 1);

        *TEST_FAIL_VERSION_WRITE_UNDER
            .lock()
            .unwrap_or_else(|e| e.into_inner()) = Some(dir.path().to_path_buf());
        let err = log
            .commit(attempt(OperationKind::Append), |_| {
                Ok(vec![LogEntry::AddFile(data_file(0x21, 0, 9, 3))])
            })
            .expect_err("the injected write failure has to reach the caller");
        *TEST_FAIL_VERSION_WRITE_UNDER
            .lock()
            .unwrap_or_else(|e| e.into_inner()) = None;

        let text = err.to_string();
        assert!(
            text.contains("write lake version file"),
            "the error has to name the operation, got: {text}"
        );
        assert!(
            text.contains(&claimed.display().to_string()),
            "the error has to name the file, got: {text}"
        );
        assert!(
            matches!(&err, ZyronError::Io(io) if io.kind() == std::io::ErrorKind::PermissionDenied),
            "adding context must not change the kind every classifier dispatches on"
        );

        assert!(
            !claimed.exists(),
            "a version file the write never filled has to be unlinked"
        );
        assert_eq!(log.latest_version(), base, "the head did not move");

        // The version number is free again and the log still commits
        let next = log
            .commit(attempt(OperationKind::Append), |_| {
                Ok(vec![LogEntry::AddFile(data_file(0x22, 0, 9, 3))])
            })
            .expect("the log still accepts commits after a failed write");
        assert_eq!(next, base + 1);

        // And the table still opens, which a torn header would have stopped
        let reopened = TransactionLog::open(LakePaths::new(dir.path(), 7), &AllCommitted)
            .expect("a table that survived a failed write still opens");
        assert_eq!(reopened.latest_version(), base + 1);
        assert_eq!(
            reopened.latest_manifest().expect("manifest").entries.len(),
            1
        );
    }

    struct RejectTxn(u64);

    impl CommitStatus for RejectTxn {
        fn is_committed(&self, db_txn_id: u64) -> bool {
            db_txn_id != self.0
        }
    }

    #[test]
    fn test_open_discards_uncommitted_tail_and_cascade() {
        let dir = tempfile::tempdir().expect("tempdir");
        let paths = LakePaths::new(dir.path(), 7);
        {
            let log = new_log(dir.path());
            let mut txn = attempt(OperationKind::Append);
            txn.db_txn_id = 7;
            let v2 = log
                .commit(txn, |_| {
                    Ok(vec![LogEntry::AddFile(data_file(0x1, 0, 9, 3))])
                })
                .expect("txn commit");
            log.publish(v2).expect("publish");
            // A standalone commit lands on top of the transactional one
            log.commit(attempt(OperationKind::Append), |_| {
                Ok(vec![LogEntry::AddFile(data_file(0x2, 0, 9, 3))])
            })
            .expect("standalone");
            assert_eq!(log.latest_version(), 3);
        }
        // Crash happened before transaction 7's commit record became
        // durable, its version and everything built after it must go
        let log = TransactionLog::open(paths.clone(), &RejectTxn(7)).expect("open");
        assert_eq!(log.latest_version(), 1);
        assert!(!paths.version_file(2).exists());
        assert!(!paths.version_file(3).exists());
        let m = log.latest_manifest().expect("manifest");
        assert!(m.entries.is_empty());
    }

    #[test]
    fn test_checkpoint_gc_and_time_travel_floor() {
        let dir = tempfile::tempdir().expect("tempdir");
        let log = new_log(dir.path());
        for i in 0..4u64 {
            log.commit(attempt(OperationKind::Append), |_| {
                Ok(vec![LogEntry::AddFile(data_file(0x10 + i, 0, 9, 2))])
            })
            .expect("append");
        }
        assert_eq!(log.latest_version(), 5);
        log.checkpoint(3).expect("checkpoint");
        let removed = log.gc_versions(3).expect("gc");
        assert_eq!(removed, 3, "versions 1 to 3 deleted");

        // Reconstruction from the checkpoint plus the tail still works
        let m4 = log.manifest_at(4).expect("v4 from checkpoint");
        assert_eq!(m4.entries.len(), 3);
        let m5 = log.manifest_at(5).expect("v5 from checkpoint");
        assert_eq!(m5.entries.len(), 4);

        // Versions below the floor are gone for good
        let fresh =
            TransactionLog::open(LakePaths::new(dir.path(), 7), &AllCommitted).expect("reopen");
        assert_eq!(fresh.latest_version(), 5);
        assert!(fresh.manifest_at(2).is_err());
    }

    #[test]
    fn test_reopen_replays_to_identical_state() {
        let dir = tempfile::tempdir().expect("tempdir");
        let before = {
            let log = new_log(dir.path());
            log.commit(attempt(OperationKind::Append), |_| {
                Ok(vec![LogEntry::AddFile(data_file(0xA, 1, 100, 10))])
            })
            .expect("append");
            log.commit(attempt(OperationKind::Delete), |base| {
                assert!(base.entry_for(0xA).is_some());
                Ok(vec![LogEntry::AddDeletePredicate(DeletePredicate {
                    id: 1,
                    sql: "id > 50".into(),
                    predicate: LakePredicate::Compare {
                        column_id: 0,
                        op: CompareOp::Gt,
                        value: LakeValue::Int(50),
                    },
                    created_version: 0,
                    pending_rows: 0,
                })])
            })
            .expect("predicate delete");
            (*log.latest_manifest().expect("manifest")).clone()
        };
        let log =
            TransactionLog::open(LakePaths::new(dir.path(), 7), &AllCommitted).expect("reopen");
        let after = (*log.latest_manifest().expect("manifest")).clone();
        assert_eq!(before, after);
        // The predicate attached to the file it may match
        assert_eq!(after.entries[0].delete_predicate_ids, vec![1]);
    }

    #[test]
    fn test_audit_block_roundtrip() {
        let dir = tempfile::tempdir().expect("tempdir");
        let log = new_log(dir.path());
        let audit = CommitInfo {
            identity: "svc_ingest".into(),
            client: "zyron-streaming 0.1".into(),
            commit_info: "nightly load".into(),
            correlation_id: "corr-123".into(),
            trace_id: "trace-9".into(),
            signature: vec![1, 2, 3],
        };
        let mut with_audit = attempt(OperationKind::Append);
        with_audit.audit = Some(&audit);
        let v = log
            .commit(with_audit, |_| {
                Ok(vec![LogEntry::AddFile(data_file(0x5, 0, 9, 1))])
            })
            .expect("commit");
        let data =
            read_version_file(&LakePaths::new(dir.path(), 7).version_file(v)).expect("read back");
        assert_eq!(data.audit, Some(audit));
        assert_eq!(data.entries.len(), 1);
    }

    #[test]
    fn test_corrupted_version_file_is_rejected() {
        let dir = tempfile::tempdir().expect("tempdir");
        let paths = LakePaths::new(dir.path(), 7);
        {
            let log = new_log(dir.path());
            log.commit(attempt(OperationKind::Append), |_| {
                Ok(vec![LogEntry::AddFile(data_file(0x1, 0, 9, 3))])
            })
            .expect("append");
        }
        let path = paths.version_file(2);
        let mut bytes = fs::read(&path).expect("read");
        let last = bytes.len() - 1;
        bytes[last] ^= 0xFF;
        fs::write(&path, &bytes).expect("write corruption");
        let err = TransactionLog::open(paths, &AllCommitted).expect_err("must reject");
        assert!(err.to_string().contains("checksum"));
    }

    #[test]
    fn test_schema_change_entry_roundtrip_in_stream() {
        let dir = tempfile::tempdir().expect("tempdir");
        let log = new_log(dir.path());
        let mut evolved = schema();
        evolved.schema_id = 2;
        evolved.columns.push(LakeColumn {
            id: 1,
            name: "added".into(),
            type_id: TypeId::Text,
            nullable: true,
            fractional_digits: None,
            tz_offset_secs: None,
            max_length: None,
            default_expr: None,
        });
        evolved.next_column_id = 2;
        let v = log
            .commit(attempt(OperationKind::SchemaChange), |_| {
                Ok(vec![
                    LogEntry::SchemaChange(evolved.clone()),
                    LogEntry::SetProperty {
                        key: "note".into(),
                        value: "widened".into(),
                    },
                ])
            })
            .expect("schema change");
        let data =
            read_version_file(&LakePaths::new(dir.path(), 7).version_file(v)).expect("read back");
        assert_eq!(data.entries.len(), 2);
        assert_eq!(data.entries[0], LogEntry::SchemaChange(evolved.clone()));
        let m = log.latest_manifest().expect("manifest");
        assert_eq!(m.schema, evolved);
        assert_eq!(
            m.properties.get("note").map(|s| s.as_str()),
            Some("widened")
        );
    }
}
