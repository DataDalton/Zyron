//! The replicated log: what a command looks like, and how it reaches disk.
//!
//! ## One writer, one file, one fsync per batch
//!
//! Consensus is a sequence, so the log is an append-only file and a single
//! thread owns the handle. Appends are staged into a reusable buffer by
//! whoever holds the consensus lock and handed to that thread as one message
//! per batch; the thread drains everything queued, writes it with one
//! `write_all`, and calls `sync_data` once. That is group commit: a hundred
//! proposals arriving inside one fsync interval cost one fsync between them,
//! which is the only way a consensus group reaches six figures of writes a
//! second on hardware where a single fsync is tens of microseconds.
//!
//! Durability is published as an index on an atomic. Nothing replies to a
//! leader, and no proposal is reported committed, before that index has
//! passed the entry in question. A reply sent ahead of the fsync would let a
//! node acknowledge an entry it could lose on a power cut, and a majority of
//! such acknowledgements is exactly the case Raft's safety argument rules out.
//!
//! ## Why truncation is cheap and compaction is not
//!
//! A follower whose log diverges from the leader's discards the divergent
//! tail. That is `set_len` on the file plus a pop from the offset table, so it
//! costs the same whether one entry or ten thousand are dropped.
//!
//! Discarding the head, after a snapshot has made it redundant, cannot be a
//! `set_len`. The retained tail is copied into a new file which is renamed
//! over the old one. It is O(retained), and the retained tail right after a
//! snapshot is small by construction, so this is bounded by the snapshot
//! cadence rather than by the age of the group.

use std::collections::VecDeque;
use std::fs::{File, OpenOptions};
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

use crossbeam::channel::{Receiver, Sender, TryRecvError, unbounded};
use zyron_common::checksum::hash32;
use zyron_common::error::{Result, ZyronError};
use zyron_common::format::envelope::{self, ENVELOPE_HEADER_LEN};
use zyron_common::format::{FormatKind, FormatVersion, RecordVersion};

use crate::NodeId;
use crate::codec::{Cursor, put_bytes, put_str, put_u8, put_u64};
use crate::membership::ClusterConfig;

/// Name of the log file under the raft directory
pub const LOG_FILE: &str = "raft.log";
/// Where a compaction assembles the retained tail before the rename
const LOG_TMP_FILE: &str = "raft.log.tmp";

/// Version the log file is written at, declared in the format registry
const FILE_VERSION: FormatVersion = crate::format::RAFT_LOG_FORMAT_VERSION;
/// Envelope header 20, base_index 8, prev_term 8
const FILE_HEADER_LEN: usize = 36;

/// Version tag every record carries, so one log can hold records written
/// across a rolling upgrade
const RECORD_VERSION: RecordVersion = crate::format::RAFT_ENTRY_RECORD_VERSION;
/// magic 4, record_version 1, pad 3, payload_len 4, term 8, index 8,
/// checksum 4
const RECORD_HEADER_LEN: usize = 32;

/// Largest single command payload.
///
/// Sixteen megabytes is far above any row batch or WAL record group and keeps
/// a corrupt length prefix from being taken at face value on read back
pub const MAX_COMMAND_BYTES: usize = 16 * 1024 * 1024;

/// Floor on the residency cap.
///
/// A command larger than the cap is evicted the moment it is durable, which is
/// correct but pointless work, and a budget below this is not a budget anyone
/// meant to set
pub const MIN_RESIDENT_BYTES: usize = 64 * 1024;

/// What a committed entry does to the state machine.
///
/// The data commands come first because they are the common case and the tag
/// is the first byte a decoder reads. The membership commands are entries
/// rather than side channel messages so that every node applies a
/// configuration change at exactly the same point in the sequence, which is
/// what stops two nodes disagreeing about who may vote
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RaftCommand {
    /// The entry a new leader appends before anything else.
    ///
    /// It carries nothing, and it exists so the leader can commit an entry
    /// from its own term. Until it does, the leader cannot know which of the
    /// entries it inherited are committed, and it must not serve a
    /// linearizable read
    Noop,
    /// Sets a key
    Put { key: Vec<u8>, value: Vec<u8> },
    /// Removes a key
    Delete { key: Vec<u8> },
    /// One transaction's effects, encoded by the state machine.
    ///
    /// Opaque here on purpose. Consensus orders these and never looks inside,
    /// so the storage engine can change what a transaction ships without the
    /// log format moving. A transaction too large for one entry sends a run of
    /// them and the last one carries the flag that completes it
    Data { payload: Vec<u8> },
    /// Brings a node into the group as a learner.
    ///
    /// Safe as a single entry because a learner is in no quorum, so no node
    /// changes what it counts as a majority when it applies this
    AddNode { node_id: NodeId, address: String },
    /// Drops a node that is not a voter.
    ///
    /// Removing a voter goes through [`RaftCommand::JointConfig`] instead
    RemoveNode { node_id: NodeId },
    /// Records that a snapshot covers the log up to this point.
    ///
    /// Appended when a snapshot is taken so that the fact is part of the
    /// replicated sequence rather than only of local metadata
    Snapshot {
        last_included_index: u64,
        last_included_term: u64,
    },
    /// Enters a joint configuration: both voter sets are in force
    JointConfig {
        old: ClusterConfig,
        new: ClusterConfig,
    },
    /// Leaves a joint configuration for the incoming voter set alone
    FinalConfig { config: ClusterConfig },
}

const TAG_NOOP: u8 = 0;
const TAG_PUT: u8 = 1;
const TAG_DELETE: u8 = 2;
const TAG_DATA: u8 = 3;
const TAG_ADD_NODE: u8 = 4;
const TAG_REMOVE_NODE: u8 = 5;
const TAG_SNAPSHOT: u8 = 6;
const TAG_JOINT_CONFIG: u8 = 7;
const TAG_FINAL_CONFIG: u8 = 8;

impl RaftCommand {
    /// Whether applying this entry changes who votes.
    ///
    /// The leader allows one such entry to be uncommitted at a time, because
    /// two overlapping voter changes can produce configurations that no
    /// single quorum spans
    pub fn is_config_change(&self) -> bool {
        matches!(
            self,
            RaftCommand::AddNode { .. }
                | RaftCommand::RemoveNode { .. }
                | RaftCommand::JointConfig { .. }
                | RaftCommand::FinalConfig { .. }
        )
    }

    pub fn encode(&self, buf: &mut Vec<u8>) {
        match self {
            RaftCommand::Noop => put_u8(buf, TAG_NOOP),
            RaftCommand::Put { key, value } => {
                put_u8(buf, TAG_PUT);
                put_bytes(buf, key);
                put_bytes(buf, value);
            }
            RaftCommand::Delete { key } => {
                put_u8(buf, TAG_DELETE);
                put_bytes(buf, key);
            }
            RaftCommand::Data { payload } => {
                put_u8(buf, TAG_DATA);
                put_bytes(buf, payload);
            }
            RaftCommand::AddNode { node_id, address } => {
                put_u8(buf, TAG_ADD_NODE);
                put_u64(buf, *node_id);
                put_str(buf, address);
            }
            RaftCommand::RemoveNode { node_id } => {
                put_u8(buf, TAG_REMOVE_NODE);
                put_u64(buf, *node_id);
            }
            RaftCommand::Snapshot {
                last_included_index,
                last_included_term,
            } => {
                put_u8(buf, TAG_SNAPSHOT);
                put_u64(buf, *last_included_index);
                put_u64(buf, *last_included_term);
            }
            RaftCommand::JointConfig { old, new } => {
                put_u8(buf, TAG_JOINT_CONFIG);
                old.encode(buf);
                new.encode(buf);
            }
            RaftCommand::FinalConfig { config } => {
                put_u8(buf, TAG_FINAL_CONFIG);
                config.encode(buf);
            }
        }
    }

    pub fn decode(c: &mut Cursor<'_>) -> Result<Self> {
        let tag = c.u8()?;
        Ok(match tag {
            TAG_NOOP => RaftCommand::Noop,
            TAG_PUT => RaftCommand::Put {
                key: c.vec()?,
                value: c.vec()?,
            },
            TAG_DELETE => RaftCommand::Delete { key: c.vec()? },
            TAG_DATA => RaftCommand::Data { payload: c.vec()? },
            TAG_ADD_NODE => RaftCommand::AddNode {
                node_id: c.u64()?,
                address: c.string()?,
            },
            TAG_REMOVE_NODE => RaftCommand::RemoveNode { node_id: c.u64()? },
            TAG_SNAPSHOT => RaftCommand::Snapshot {
                last_included_index: c.u64()?,
                last_included_term: c.u64()?,
            },
            TAG_JOINT_CONFIG => RaftCommand::JointConfig {
                old: ClusterConfig::decode(c)?,
                new: ClusterConfig::decode(c)?,
            },
            TAG_FINAL_CONFIG => RaftCommand::FinalConfig {
                config: ClusterConfig::decode(c)?,
            },
            other => {
                return Err(ZyronError::EncodingFailed(format!(
                    "raft command tag {other} is not known to this build"
                )));
            }
        })
    }

    /// Roughly how many bytes this command encodes to.
    ///
    /// Used to size a replication batch before building it, so the batch
    /// builder does not encode past its byte budget and then throw the excess
    /// away
    pub fn encoded_len(&self) -> usize {
        match self {
            RaftCommand::Noop => 1,
            RaftCommand::Put { key, value } => 1 + 4 + key.len() + 4 + value.len(),
            RaftCommand::Delete { key } => 1 + 4 + key.len(),
            RaftCommand::Data { payload } => 1 + 4 + payload.len(),
            RaftCommand::AddNode { address, .. } => 1 + 8 + 4 + address.len(),
            RaftCommand::RemoveNode { .. } => 1 + 8,
            RaftCommand::Snapshot { .. } => 1 + 16,
            RaftCommand::JointConfig { old, new } => 1 + config_len(old) + config_len(new),
            RaftCommand::FinalConfig { config } => 1 + config_len(config),
        }
    }
}

fn config_len(c: &ClusterConfig) -> usize {
    4 + c
        .nodes
        .iter()
        .map(|n| 8 + 4 + n.address.len() + 2)
        .sum::<usize>()
}

/// One entry in the replicated log.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RaftLogEntry {
    /// The term of the leader that created this entry
    pub term: u64,
    /// Position in the log, one based and gapless
    pub index: u64,
    pub command: RaftCommand,
}

impl RaftLogEntry {
    pub fn new(term: u64, index: u64, command: RaftCommand) -> Self {
        Self {
            term,
            index,
            command,
        }
    }

    /// Appends the on-disk and on-wire record for this entry.
    ///
    /// The same bytes go to the file and to a follower, so a follower's copy
    /// of an entry is byte identical to the leader's and checksums the same
    pub fn encode_record(&self, buf: &mut Vec<u8>) {
        let header_at = buf.len();
        buf.resize(header_at + RECORD_HEADER_LEN, 0);
        self.command.encode(buf);
        let payload_len = buf.len() - header_at - RECORD_HEADER_LEN;
        let checksum =
            record_checksum(self.term, self.index, &buf[header_at + RECORD_HEADER_LEN..]);
        let header = &mut buf[header_at..header_at + RECORD_HEADER_LEN];
        header[0..4].copy_from_slice(&FormatKind::RaftLog.magic());
        header[4] = RECORD_VERSION.get();
        // 5..8 reserved, already zeroed
        header[8..12].copy_from_slice(&(payload_len as u32).to_le_bytes());
        header[12..20].copy_from_slice(&self.term.to_le_bytes());
        header[20..28].copy_from_slice(&self.index.to_le_bytes());
        header[28..32].copy_from_slice(&checksum.to_le_bytes());
    }

    /// Reads one record, returning it with the number of bytes it occupied
    pub fn decode_record(data: &[u8]) -> Result<(Self, usize)> {
        if data.len() < RECORD_HEADER_LEN {
            return Err(ZyronError::EncodingFailed(
                "raft log record header is truncated".into(),
            ));
        }
        let magic = [data[0], data[1], data[2], data[3]];
        if magic != FormatKind::RaftLog.magic() {
            return Err(ZyronError::EncodingFailed(format!(
                "raft log record magic is {}",
                zyron_common::format::envelope::printable_magic(&magic)
            )));
        }
        let record_version = RecordVersion::read(&data[4..])
            .map_err(|e| ZyronError::EncodingFailed(format!("raft log record version tag, {e}")))?;
        if record_version > RECORD_VERSION {
            return Err(ZyronError::EncodingFailed(format!(
                "raft log record is at version {}, this binary reads up to {}. Upgrade \
                 through a release that still reads {} to replay this log",
                record_version.get(),
                RECORD_VERSION.get(),
                record_version.get()
            )));
        }
        let payload_len = u32::from_le_bytes([data[8], data[9], data[10], data[11]]) as usize;
        if payload_len > MAX_COMMAND_BYTES {
            return Err(ZyronError::EncodingFailed(format!(
                "raft log record claims a {payload_len} byte payload"
            )));
        }
        let total = RECORD_HEADER_LEN + payload_len;
        if data.len() < total {
            return Err(ZyronError::EncodingFailed(
                "raft log record payload is truncated".into(),
            ));
        }
        let term = u64::from_le_bytes([
            data[12], data[13], data[14], data[15], data[16], data[17], data[18], data[19],
        ]);
        let index = u64::from_le_bytes([
            data[20], data[21], data[22], data[23], data[24], data[25], data[26], data[27],
        ]);
        let stored = u32::from_le_bytes([data[28], data[29], data[30], data[31]]);
        let payload = &data[RECORD_HEADER_LEN..total];
        let computed = record_checksum(term, index, payload);
        if stored != computed {
            return Err(ZyronError::RaftLogCorrupted {
                index,
                reason: format!("checksum {stored:#010x} does not match computed {computed:#010x}"),
            });
        }
        let mut c = Cursor::new(payload);
        let command = RaftCommand::decode(&mut c)?;
        Ok((
            Self {
                term,
                index,
                command,
            },
            total,
        ))
    }

    pub fn encoded_len(&self) -> usize {
        RECORD_HEADER_LEN + self.command.encoded_len()
    }
}

fn record_checksum(term: u64, index: u64, payload: &[u8]) -> u32 {
    // The term and index are folded in with the payload so a record cannot be
    // read as a different position in the log with its checksum still passing
    let mut head = [0u8; 16];
    head[0..8].copy_from_slice(&term.to_le_bytes());
    head[8..16].copy_from_slice(&index.to_le_bytes());
    let a = hash32(&head);
    let b = hash32(payload);
    a ^ b.rotate_left(13)
}

// ---------------------------------------------------------------------------
// Durability: the single owner of the file
// ---------------------------------------------------------------------------

enum LogOp {
    /// A run of consecutive records, already encoded
    Append {
        start_index: u64,
        blob: Vec<u8>,
        lengths: Vec<u32>,
    },
    /// Drop everything after this index
    Truncate { index: u64 },
    /// Drop everything up to and including this index
    Compact { index: u64, prev_term: u64 },
    /// Discard the whole log and restart it after a snapshot
    Reset {
        base_index: u64,
        prev_term: u64,
        /// Raw offset that the first record after the reset will carry, which
        /// is what the new shift is measured against
        raw_base: u64,
    },
    /// The log is going away, so the writer stops.
    ///
    /// Sent explicitly rather than inferred from the channel closing, because
    /// clones of the handle outlive the log itself and would keep the channel
    /// open forever
    Shutdown,
}

/// The handle the consensus side keeps: a queue and two published numbers.
///
/// Cloneable and cheap, because the replication path reads `persisted` on
/// every reply it is about to send
pub struct LogWriterHandle {
    tx: Sender<LogOp>,
    persisted: Arc<AtomicU64>,
    failed: Arc<AtomicBool>,
    error: Arc<parking_lot::Mutex<Option<String>>>,
    /// Published on every advance. A watch rather than a notify because a
    /// waiter that checks the index after the writer has already moved past
    /// it must not park forever on a wakeup it missed
    persisted_tx: Arc<tokio::sync::watch::Sender<u64>>,
}

impl Clone for LogWriterHandle {
    fn clone(&self) -> Self {
        Self {
            tx: self.tx.clone(),
            persisted: Arc::clone(&self.persisted),
            failed: Arc::clone(&self.failed),
            error: Arc::clone(&self.error),
            persisted_tx: Arc::clone(&self.persisted_tx),
        }
    }
}

impl LogWriterHandle {
    /// The highest index known to be on stable storage
    #[inline]
    pub fn persisted_index(&self) -> u64 {
        self.persisted.load(Ordering::Acquire)
    }

    /// A receiver that changes every time `persisted_index` advances
    pub fn subscribe(&self) -> tokio::sync::watch::Receiver<u64> {
        self.persisted_tx.subscribe()
    }

    /// The write that failed, if one did.
    ///
    /// A failed writer is terminal for this node: it can no longer promise
    /// durability, so callers turn this into an error rather than waiting for
    /// an index that will never be published
    pub fn failure(&self) -> Option<String> {
        if !self.failed.load(Ordering::Acquire) {
            return None;
        }
        self.error.lock().clone()
    }

    fn send(&self, op: LogOp) -> Result<()> {
        if let Some(err) = self.failure() {
            return Err(ZyronError::WalWriteFailed(format!(
                "raft log writer has failed: {err}"
            )));
        }
        self.tx
            .send(op)
            .map_err(|_| ZyronError::WalWriteFailed("raft log writer thread is not running".into()))
    }
}

struct LogStore {
    file: File,
    path: PathBuf,
    tmp_path: PathBuf,
    /// Offsets of each record, `offsets[i]` belongs to index `base_index + i`
    offsets: VecDeque<u64>,
    base_index: u64,
    prev_term: u64,
    end_offset: u64,
    /// Published so a reader outside this thread can turn a recorded offset
    /// into a position in the file as it stands now
    geometry: Arc<parking_lot::RwLock<FileGeometry>>,
}

impl LogStore {
    fn apply(&mut self, op: LogOp) -> Result<()> {
        match op {
            LogOp::Append {
                start_index,
                blob,
                lengths,
            } => self.append(start_index, &blob, &lengths),
            LogOp::Truncate { index } => self.truncate_after(index),
            LogOp::Compact { index, prev_term } => self.compact_to(index, prev_term),
            LogOp::Reset {
                base_index,
                prev_term,
                raw_base,
            } => self.reset(base_index, prev_term, raw_base),
            LogOp::Shutdown => Ok(()),
        }
    }

    fn append(&mut self, start_index: u64, blob: &[u8], lengths: &[u32]) -> Result<()> {
        let expected = self.base_index + self.offsets.len() as u64;
        if start_index != expected {
            return Err(ZyronError::RaftLogCorrupted {
                index: start_index,
                reason: format!("append starts at {start_index} but the file ends at {expected}"),
            });
        }
        self.file
            .write_all(blob)
            .map_err(|e| ZyronError::WalWriteFailed(format!("raft log append failed: {e}")))?;
        let mut at = self.end_offset;
        for len in lengths {
            self.offsets.push_back(at);
            at += u64::from(*len);
        }
        self.end_offset = at;
        Ok(())
    }

    fn truncate_after(&mut self, index: u64) -> Result<()> {
        let last = self.base_index + self.offsets.len() as u64 - 1;
        if self.offsets.is_empty() || index >= last {
            return Ok(());
        }
        let keep = if index < self.base_index {
            0
        } else {
            (index - self.base_index + 1) as usize
        };
        let new_end = if keep == 0 {
            FILE_HEADER_LEN as u64
        } else {
            self.offsets[keep]
        };
        self.file
            .set_len(new_end)
            .map_err(|e| ZyronError::WalWriteFailed(format!("raft log truncate failed: {e}")))?;
        self.file
            .seek(SeekFrom::End(0))
            .map_err(|e| ZyronError::WalWriteFailed(format!("raft log seek failed: {e}")))?;
        self.offsets.truncate(keep);
        self.end_offset = new_end;
        Ok(())
    }

    /// Copies the tail past `index` into a fresh file and renames it over the
    /// old one, so the head that the snapshot covers stops costing disk
    fn compact_to(&mut self, index: u64, prev_term: u64) -> Result<()> {
        if index < self.base_index {
            return Ok(());
        }
        let last = self.base_index + self.offsets.len() as u64;
        let drop_count = ((index + 1).min(last) - self.base_index) as usize;
        let from_offset = if drop_count >= self.offsets.len() {
            self.end_offset
        } else {
            self.offsets[drop_count]
        };
        let new_base = index + 1;

        self.file.sync_data().map_err(|e| {
            ZyronError::WalWriteFailed(format!("raft log sync before compact: {e}"))
        })?;
        let mut src = File::open(&self.path)
            .map_err(|e| ZyronError::IoError(format!("reopen raft log for compaction: {e}")))?;
        src.seek(SeekFrom::Start(from_offset))
            .map_err(|e| ZyronError::IoError(format!("seek raft log for compaction: {e}")))?;

        let mut tmp = OpenOptions::new()
            .create(true)
            .write(true)
            .truncate(true)
            .open(&self.tmp_path)
            .map_err(|e| ZyronError::IoError(format!("create compacted raft log: {e}")))?;
        write_file_header(&mut tmp, new_base, prev_term)?;
        let mut buf = vec![0u8; 1 << 20];
        loop {
            let n = src
                .read(&mut buf)
                .map_err(|e| ZyronError::IoError(format!("read raft log for compaction: {e}")))?;
            if n == 0 {
                break;
            }
            tmp.write_all(&buf[..n])
                .map_err(|e| ZyronError::IoError(format!("write compacted raft log: {e}")))?;
        }
        tmp.sync_all()
            .map_err(|e| ZyronError::IoError(format!("sync compacted raft log: {e}")))?;
        drop(tmp);
        drop(src);
        drop(std::mem::replace(&mut self.file, dummy_file()?));
        std::fs::rename(&self.tmp_path, &self.path)
            .map_err(|e| ZyronError::IoError(format!("rename compacted raft log: {e}")))?;
        // Opened for writing rather than appending, because truncating the
        // tail is a `set_len` and Windows grants append-only handles no right
        // to move the end of a file. The single owner seeks to the end once
        // and every write advances from there
        self.file = OpenOptions::new()
            .read(true)
            .write(true)
            .open(&self.path)
            .map_err(|e| ZyronError::IoError(format!("reopen compacted raft log: {e}")))?;
        self.file
            .seek(SeekFrom::End(0))
            .map_err(|e| ZyronError::IoError(format!("seek compacted raft log: {e}")))?;

        let shift = from_offset - FILE_HEADER_LEN as u64;
        for _ in 0..drop_count.min(self.offsets.len()) {
            self.offsets.pop_front();
        }
        for off in self.offsets.iter_mut() {
            *off -= shift;
        }
        self.base_index = new_base;
        self.prev_term = prev_term;
        self.end_offset -= shift;
        // Published after the rename, so a reader that still holds the old
        // handle is also still using the old shift and stays consistent with
        // the file it is actually reading
        {
            let mut geometry = self.geometry.write();
            geometry.shift += shift;
            geometry.generation += 1;
        }
        Ok(())
    }

    fn reset(&mut self, base_index: u64, prev_term: u64, raw_base: u64) -> Result<()> {
        self.file
            .set_len(0)
            .map_err(|e| ZyronError::WalWriteFailed(format!("raft log reset failed: {e}")))?;
        self.file
            .seek(SeekFrom::Start(0))
            .map_err(|e| ZyronError::WalWriteFailed(format!("raft log seek failed: {e}")))?;
        write_file_header(&mut self.file, base_index, prev_term)?;
        self.offsets.clear();
        self.base_index = base_index;
        self.prev_term = prev_term;
        self.end_offset = FILE_HEADER_LEN as u64;
        {
            let mut geometry = self.geometry.write();
            geometry.shift = raw_base - FILE_HEADER_LEN as u64;
            geometry.generation += 1;
        }
        Ok(())
    }

    fn last_index(&self) -> u64 {
        self.base_index + self.offsets.len() as u64 - 1
    }
}

fn dummy_file() -> Result<File> {
    // A placeholder handle so the real one can be dropped before a rename on
    // platforms that refuse to rename over an open file
    OpenOptions::new()
        .read(true)
        .write(true)
        .create(true)
        .truncate(true)
        .open(std::env::temp_dir().join("zyron-raft-compaction-placeholder"))
        .map_err(|e| ZyronError::IoError(format!("open placeholder handle: {e}")))
}

fn write_file_header(file: &mut File, base_index: u64, prev_term: u64) -> Result<()> {
    let mut extension = [0u8; FILE_HEADER_LEN - ENVELOPE_HEADER_LEN];
    extension[0..8].copy_from_slice(&base_index.to_le_bytes());
    extension[8..16].copy_from_slice(&prev_term.to_le_bytes());
    let mut header = [0u8; FILE_HEADER_LEN];
    header[0..ENVELOPE_HEADER_LEN].copy_from_slice(&envelope::encode_header(
        FormatKind::RaftLog,
        FILE_VERSION,
        0,
        &extension,
    ));
    header[ENVELOPE_HEADER_LEN..].copy_from_slice(&extension);
    file.write_all(&header)
        .map_err(|e| ZyronError::WalWriteFailed(format!("write raft log header: {e}")))
}

/// What the writer thread publishes back.
///
/// Deliberately not a [`LogWriterHandle`]: that carries a `Sender`, and a
/// thread holding a sender for its own receiver never sees the channel close
struct WriterPublish {
    persisted: Arc<AtomicU64>,
    failed: Arc<AtomicBool>,
    error: Arc<parking_lot::Mutex<Option<String>>>,
    persisted_tx: Arc<tokio::sync::watch::Sender<u64>>,
}

fn writer_loop(mut store: LogStore, rx: Receiver<LogOp>, handle: WriterPublish) {
    let mut batch: Vec<LogOp> = Vec::with_capacity(64);
    'outer: loop {
        match rx.recv() {
            Ok(op) => batch.push(op),
            Err(_) => break,
        }
        // Everything already queued joins this fsync
        loop {
            match rx.try_recv() {
                Ok(op) => batch.push(op),
                Err(TryRecvError::Empty) => break,
                Err(TryRecvError::Disconnected) => break,
            }
        }
        let mut failure: Option<String> = None;
        let mut stopping = false;
        for op in batch.drain(..) {
            if matches!(op, LogOp::Shutdown) {
                stopping = true;
                break;
            }
            if let Err(e) = store.apply(op) {
                failure = Some(e.to_string());
                break;
            }
        }
        if failure.is_none() {
            if let Err(e) = store.file.sync_data() {
                failure = Some(format!("raft log fsync failed: {e}"));
            }
        }
        match failure {
            Some(err) => {
                *handle.error.lock() = Some(err);
                handle.failed.store(true, Ordering::Release);
                // A change with no advance, so a waiter wakes and sees the
                // failure rather than waiting for an index that will not come
                handle.persisted_tx.send_modify(|_| {});
                break;
            }
            None => {
                let at = store.last_index();
                handle.persisted.store(at, Ordering::Release);
                handle.persisted_tx.send_replace(at);
            }
        }
        if stopping {
            break 'outer;
        }
    }
}

// ---------------------------------------------------------------------------
// The consensus-side view of the log
// ---------------------------------------------------------------------------

/// The replicated log as consensus sees it: entries in memory, durability on
/// an atomic, and the file behind a single writer thread.
///
/// Entries live in memory from the compaction point forward. That is bounded
/// by the snapshot threshold rather than by the age of the group, which is the
/// reason snapshots exist at all
/// One entry's place in the log.
///
/// The term and the file position stay resident for every entry, because the
/// consistency check and the pager need them and together they are 24 bytes.
/// The decoded command is what a residency cap drops, and it is the part that
/// is measured in megabytes once transactions rather than keys are replicated
struct LogSlot {
    term: u64,
    /// Position this record was first written at, before any compaction
    /// shifted the file under it. [`FileGeometry::shift`] converts it
    raw_offset: u64,
    /// Encoded record length, header included
    len: u32,
    /// Whether eviction must leave this one alone.
    ///
    /// Configuration changes are read back by index while deciding whether a
    /// voter change may be proposed and while rebuilding membership across a
    /// compaction. They are a few hundred bytes and there are as many of them
    /// as there have been cluster changes, so pinning them costs nothing and
    /// takes both of those reads off the paging path
    pinned: bool,
    /// The decoded command, dropped once the record is durable and the memory
    /// is wanted elsewhere
    entry: Option<Arc<RaftLogEntry>>,
}

/// Where the records currently sit, as the writer thread last left them.
///
/// A compaction rewrites the file, so every retained record moves down by the
/// same number of bytes. Rather than walking the slot table to fix offsets,
/// the shift accumulates here and a read subtracts it. Compaction stays O(1)
/// on the consensus side however much is retained
#[derive(Clone, Copy, Debug)]
struct FileGeometry {
    /// Total bytes removed from the front of the file
    shift: u64,
    /// Bumped whenever the file is replaced, so a cached handle is noticed as
    /// stale rather than read from after a rename
    generation: u64,
}

/// Reads records back from the log file for entries no longer held in memory.
///
/// Has its own handle so a page-in never contends with the writer thread and
/// never runs under the consensus lock
pub struct LogPager {
    path: PathBuf,
    geometry: Arc<parking_lot::RwLock<FileGeometry>>,
    file: parking_lot::Mutex<Option<(u64, File)>>,
}

impl LogPager {
    /// Reads one record and checks it is the one that was asked for.
    ///
    /// The identity check is the whole safety argument for reading outside the
    /// consensus lock: a compaction that lands between reading the geometry
    /// and reading the file produces a record at the wrong index, which is
    /// caught here and retried against the new geometry rather than returned
    pub fn read(
        &self,
        index: u64,
        term: u64,
        raw_offset: u64,
        len: u32,
    ) -> Result<Arc<RaftLogEntry>> {
        let mut last: Option<ZyronError> = None;
        for _ in 0..2 {
            match self.read_once(index, term, raw_offset, len) {
                Ok(entry) => return Ok(entry),
                Err(e) => {
                    // Drop the handle so a retry opens whatever the path names
                    // now rather than whatever it named before the rename
                    *self.file.lock() = None;
                    last = Some(e);
                }
            }
        }
        Err(last.unwrap_or_else(|| ZyronError::RaftLogCorrupted {
            index,
            reason: "record could not be read back from the log file".into(),
        }))
    }

    fn read_once(
        &self,
        index: u64,
        term: u64,
        raw_offset: u64,
        len: u32,
    ) -> Result<Arc<RaftLogEntry>> {
        let geometry = *self.geometry.read();
        let offset =
            raw_offset
                .checked_sub(geometry.shift)
                .ok_or_else(|| ZyronError::RaftLogCorrupted {
                    index,
                    reason: "record sits before the start of the file, so a snapshot covers it"
                        .into(),
                })?;

        let mut guard = self.file.lock();
        let stale = match guard.as_ref() {
            Some((generation, _)) => *generation != geometry.generation,
            None => true,
        };
        if stale {
            let file = File::open(&self.path)
                .map_err(|e| ZyronError::IoError(format!("open raft log for read back: {e}")))?;
            *guard = Some((geometry.generation, file));
        }
        let Some((_, file)) = guard.as_mut() else {
            return Err(ZyronError::IoError("raft log read handle vanished".into()));
        };

        let mut buf = vec![0u8; len as usize];
        file.seek(SeekFrom::Start(offset))
            .map_err(|e| ZyronError::IoError(format!("seek raft log for read back: {e}")))?;
        file.read_exact(&mut buf)
            .map_err(|e| ZyronError::IoError(format!("read raft log record: {e}")))?;
        drop(guard);

        let (entry, used) = RaftLogEntry::decode_record(&buf)?;
        if used != buf.len() || entry.index != index || entry.term != term {
            return Err(ZyronError::RaftLogCorrupted {
                index,
                reason: format!(
                    "read back index {} term {} where index {index} term {term} was expected",
                    entry.index, entry.term
                ),
            });
        }
        Ok(Arc::new(entry))
    }
}

/// One entry a replication batch still has to read from disk.
#[derive(Debug, Clone, Copy)]
pub struct PagedRecord {
    pub index: u64,
    pub term: u64,
    raw_offset: u64,
    len: u32,
}

impl PagedRecord {
    #[inline]
    pub fn raw_offset(&self) -> u64 {
        self.raw_offset
    }

    #[inline]
    pub fn len(&self) -> u32 {
        self.len
    }
}

/// One position in a planned replication batch.
///
/// Built under the consensus lock, resolved outside it, so a batch that spans
/// the residency boundary costs the lock nothing more than one that does not
pub enum SliceItem {
    Resident(Arc<RaftLogEntry>),
    Paged(PagedRecord),
}

impl SliceItem {
    #[inline]
    pub fn index(&self) -> u64 {
        match self {
            SliceItem::Resident(e) => e.index,
            SliceItem::Paged(p) => p.index,
        }
    }
}

/// The replicated log as consensus sees it: a slot per entry, durability on an
/// atomic, and the file behind a single writer thread.
///
/// Slots live from the compaction point forward and are bounded by the
/// snapshot threshold. Decoded commands live from wherever the residency cap
/// puts the boundary forward, which is what keeps a log of megabyte
/// transactions from being a log of megabytes of memory
pub struct RaftLog {
    slots: VecDeque<LogSlot>,
    /// Indexes of the configuration entries still held, ascending.
    ///
    /// Membership is derived by replaying these, and the derivation runs on
    /// every truncation and every configuration proposal. Walking this list
    /// costs what the configuration history costs, where walking the slots
    /// would cost what the whole retained log costs
    config_indexes: Vec<u64>,
    /// Index of `slots[0]`, one past whatever a snapshot covers
    start_index: u64,
    /// Term of the entry at `start_index - 1`, from the snapshot when the head
    /// has been compacted away
    prev_term: u64,
    /// Records encoded but not yet handed to the writer
    stage: Vec<u8>,
    stage_lengths: Vec<u32>,
    stage_start: u64,
    /// Raw offset the next record will occupy
    write_cursor: u64,
    /// Bytes held by resident commands
    resident_bytes: usize,
    /// What `resident_bytes` is kept under
    resident_limit: usize,
    /// Oldest index that might still hold a resident command
    evict_next: u64,
    pager: Arc<LogPager>,
    handle: LogWriterHandle,
    writer: Option<std::thread::JoinHandle<()>>,
}

impl RaftLog {
    /// Opens the log under `dir`, replaying whatever survived.
    ///
    /// A record that fails its checksum, breaks its framing, or breaks index
    /// continuity ends the replay and the file is cut back to the last good
    /// record. A partial tail is the normal shape of a log after a crash, and
    /// silently keeping entries past a hole would let a node claim a match
    /// with a leader that it does not have.
    ///
    /// `resident_limit` bounds the bytes of decoded commands held in memory. A
    /// replay that reads more than that drops the oldest back down before
    /// returning, so a restart costs the memory of steady state rather than
    /// the size of the whole log
    pub fn open(dir: &Path, resident_limit: usize) -> Result<Self> {
        std::fs::create_dir_all(dir)
            .map_err(|e| ZyronError::IoError(format!("create raft directory: {e}")))?;
        let path = dir.join(LOG_FILE);
        let tmp_path = dir.join(LOG_TMP_FILE);
        let _ = std::fs::remove_file(&tmp_path);

        let existed = path.exists();
        let mut file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(false)
            .open(&path)
            .map_err(|e| ZyronError::IoError(format!("open raft log: {e}")))?;

        let mut base_index = 1u64;
        let mut prev_term = 0u64;
        let mut slots: VecDeque<LogSlot> = VecDeque::new();
        let mut config_indexes: Vec<u64> = Vec::new();
        let mut offsets: VecDeque<u64> = VecDeque::new();
        let mut end_offset = FILE_HEADER_LEN as u64;
        let mut resident_bytes = 0usize;

        let len = file
            .metadata()
            .map_err(|e| ZyronError::IoError(format!("stat raft log: {e}")))?
            .len();
        if !existed || len < FILE_HEADER_LEN as u64 {
            file.set_len(0)
                .map_err(|e| ZyronError::IoError(format!("reset short raft log: {e}")))?;
            file.seek(SeekFrom::Start(0))
                .map_err(|e| ZyronError::IoError(format!("seek raft log: {e}")))?;
            write_file_header(&mut file, base_index, prev_term)?;
            file.sync_all()
                .map_err(|e| ZyronError::IoError(format!("sync new raft log: {e}")))?;
        } else {
            let mut header = [0u8; FILE_HEADER_LEN];
            file.seek(SeekFrom::Start(0))
                .map_err(|e| ZyronError::IoError(format!("seek raft log: {e}")))?;
            file.read_exact(&mut header)
                .map_err(|e| ZyronError::IoError(format!("read raft log header: {e}")))?;
            let (parsed, extension) =
                envelope::decode_header(&header).map_err(|e| ZyronError::RaftLogCorrupted {
                    index: 0,
                    reason: e.to_string(),
                })?;
            if parsed.kind != FormatKind::RaftLog {
                return Err(ZyronError::RaftLogCorrupted {
                    index: 0,
                    reason: format!("file header names a {} file, not a raft log", parsed.kind),
                });
            }
            if parsed.version != FILE_VERSION {
                return Err(ZyronError::RaftLogCorrupted {
                    index: 0,
                    reason: format!(
                        "log is at format version {}, this binary writes and reads \
                         {FILE_VERSION}. Upgrade through a release that still reads {} to \
                         move the log forward first",
                        parsed.version, parsed.version
                    ),
                });
            }
            base_index = u64::from_le_bytes([
                extension[0],
                extension[1],
                extension[2],
                extension[3],
                extension[4],
                extension[5],
                extension[6],
                extension[7],
            ]);
            prev_term = u64::from_le_bytes([
                extension[8],
                extension[9],
                extension[10],
                extension[11],
                extension[12],
                extension[13],
                extension[14],
                extension[15],
            ]);

            let mut body = Vec::with_capacity((len as usize).saturating_sub(FILE_HEADER_LEN));
            file.read_to_end(&mut body)
                .map_err(|e| ZyronError::IoError(format!("read raft log body: {e}")))?;

            let mut at = 0usize;
            let mut expect = base_index;
            while at < body.len() {
                match RaftLogEntry::decode_record(&body[at..]) {
                    Ok((entry, used)) if entry.index == expect => {
                        let offset = FILE_HEADER_LEN as u64 + at as u64;
                        offsets.push_back(offset);
                        resident_bytes += used;
                        let pinned = entry.command.is_config_change();
                        if pinned {
                            config_indexes.push(entry.index);
                        }
                        slots.push_back(LogSlot {
                            term: entry.term,
                            raw_offset: offset,
                            len: used as u32,
                            pinned,
                            entry: Some(Arc::new(entry)),
                        });
                        at += used;
                        expect += 1;
                    }
                    Ok((entry, _)) => {
                        tracing::warn!(
                            found = entry.index,
                            expected = expect,
                            "raft log breaks index continuity, cutting the tail"
                        );
                        break;
                    }
                    Err(e) => {
                        tracing::warn!(offset = at, error = %e, "raft log tail is unreadable, cutting it");
                        break;
                    }
                }
            }
            end_offset = FILE_HEADER_LEN as u64 + at as u64;
            if end_offset != len {
                file.set_len(end_offset)
                    .map_err(|e| ZyronError::IoError(format!("cut raft log tail: {e}")))?;
                file.sync_all()
                    .map_err(|e| ZyronError::IoError(format!("sync cut raft log: {e}")))?;
            }
        }

        drop(file);
        let mut file = OpenOptions::new()
            .read(true)
            .write(true)
            .open(&path)
            .map_err(|e| ZyronError::IoError(format!("reopen raft log for append: {e}")))?;
        file.seek(SeekFrom::Start(end_offset))
            .map_err(|e| ZyronError::IoError(format!("seek raft log to its end: {e}")))?;

        let last_index = base_index + slots.len() as u64 - 1;
        let geometry = Arc::new(parking_lot::RwLock::new(FileGeometry {
            shift: 0,
            generation: 0,
        }));
        let store = LogStore {
            file,
            path: path.clone(),
            tmp_path,
            offsets,
            base_index,
            prev_term,
            end_offset,
            geometry: Arc::clone(&geometry),
        };
        let (tx, rx) = unbounded();
        let (persisted_tx, _persisted_rx) = tokio::sync::watch::channel(last_index);
        let handle = LogWriterHandle {
            tx,
            persisted: Arc::new(AtomicU64::new(last_index)),
            failed: Arc::new(AtomicBool::new(false)),
            error: Arc::new(parking_lot::Mutex::new(None)),
            persisted_tx: Arc::new(persisted_tx),
        };
        let thread_handle = WriterPublish {
            persisted: Arc::clone(&handle.persisted),
            failed: Arc::clone(&handle.failed),
            error: Arc::clone(&handle.error),
            persisted_tx: Arc::clone(&handle.persisted_tx),
        };
        let writer = std::thread::Builder::new()
            .name("zyron-raft-log".into())
            .spawn(move || writer_loop(store, rx, thread_handle))
            .map_err(|e| ZyronError::IoError(format!("spawn raft log writer: {e}")))?;

        let mut log = Self {
            slots,
            config_indexes,
            start_index: base_index,
            prev_term,
            stage: Vec::with_capacity(64 * 1024),
            stage_lengths: Vec::with_capacity(256),
            stage_start: 0,
            write_cursor: end_offset,
            resident_bytes,
            resident_limit: resident_limit.max(MIN_RESIDENT_BYTES),
            evict_next: base_index,
            pager: Arc::new(LogPager {
                path,
                geometry,
                file: parking_lot::Mutex::new(None),
            }),
            handle,
            writer: Some(writer),
        };
        log.evict();
        Ok(log)
    }

    /// Index of the entry just before the first one still held
    #[inline]
    pub fn first_index(&self) -> u64 {
        self.start_index
    }

    #[inline]
    pub fn prev_index(&self) -> u64 {
        self.start_index - 1
    }

    #[inline]
    pub fn prev_term(&self) -> u64 {
        self.prev_term
    }

    #[inline]
    pub fn last_index(&self) -> u64 {
        self.start_index + self.slots.len() as u64 - 1
    }

    #[inline]
    pub fn last_term(&self) -> u64 {
        match self.slots.back() {
            Some(s) => s.term,
            None => self.prev_term,
        }
    }

    /// How many entries are held, which is one half of the snapshot trigger
    #[inline]
    pub fn len(&self) -> usize {
        self.slots.len()
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.slots.is_empty()
    }

    /// Bytes the entries past the compaction point occupy on disk, which is
    /// the other half of the snapshot trigger. Counting entries alone is right
    /// for a log of keys and wrong for a log of transactions, where ten
    /// thousand entries can be ten gigabytes
    #[inline]
    pub fn stored_bytes(&self) -> u64 {
        self.write_cursor - self.head_offset()
    }

    /// Raw offset of the first entry still held
    #[inline]
    fn head_offset(&self) -> u64 {
        match self.slots.front() {
            Some(slot) => slot.raw_offset,
            None => self.write_cursor,
        }
    }

    /// Bytes of decoded commands currently held in memory
    #[inline]
    pub fn resident_bytes(&self) -> usize {
        self.resident_bytes
    }

    /// A handle that reads evicted records back, usable without the consensus
    /// lock
    #[inline]
    pub fn pager(&self) -> Arc<LogPager> {
        Arc::clone(&self.pager)
    }

    /// The term of the entry at `index`, or None when it has been compacted
    /// away or does not exist yet
    pub fn term_at(&self, index: u64) -> Option<u64> {
        if index == self.prev_index() {
            return Some(self.prev_term);
        }
        if index < self.start_index || index > self.last_index() {
            return None;
        }
        Some(self.slots[(index - self.start_index) as usize].term)
    }

    /// The entry at `index` when it is still in memory.
    ///
    /// Configuration changes are pinned resident, so a caller looking for one
    /// always finds it. Anything else can have been evicted and comes back
    /// through [`Self::get`] instead
    pub fn entry(&self, index: u64) -> Option<&Arc<RaftLogEntry>> {
        if index < self.start_index || index > self.last_index() {
            return None;
        }
        self.slots[(index - self.start_index) as usize]
            .entry
            .as_ref()
    }

    /// Whether the entry at `index` changes the voter set.
    ///
    /// Answered from the slot rather than the command, so it holds for an
    /// evicted entry as well as a resident one
    pub fn is_config_change_at(&self, index: u64) -> bool {
        if index < self.start_index || index > self.last_index() {
            return false;
        }
        self.slots[(index - self.start_index) as usize].pinned
    }

    /// Indexes of every configuration entry still held, ascending.
    ///
    /// The entries at these indexes are pinned resident, so [`Self::entry`]
    /// always finds them
    #[inline]
    pub fn config_change_indexes(&self) -> &[u64] {
        &self.config_indexes
    }

    /// Reads one entry, paging it back from the file when it has been evicted,
    /// or says why it is not there
    pub fn get(&self, index: u64) -> Result<RaftLogEntry> {
        if index < self.start_index {
            return Err(ZyronError::RaftLogCorrupted {
                index,
                reason: "entry was discarded by a snapshot".into(),
            });
        }
        if index > self.last_index() {
            return Err(ZyronError::RaftLogCorrupted {
                index,
                reason: format!("log ends at {}", self.last_index()),
            });
        }
        let slot = &self.slots[(index - self.start_index) as usize];
        if let Some(entry) = slot.entry.as_ref() {
            return Ok((**entry).clone());
        }
        let entry = self
            .pager
            .read(index, slot.term, slot.raw_offset, slot.len)?;
        Ok((*entry).clone())
    }

    /// Appends one entry, staging its record for the writer
    pub fn append(&mut self, entry: RaftLogEntry) -> Result<u64> {
        let expected = self.last_index() + 1;
        if entry.index != expected {
            return Err(ZyronError::RaftLogCorrupted {
                index: entry.index,
                reason: format!(
                    "append is out of sequence, the log ends at {}",
                    self.last_index()
                ),
            });
        }
        if entry.command.encoded_len() > MAX_COMMAND_BYTES {
            return Err(ZyronError::EncodingFailed(format!(
                "raft command is {} bytes, the limit is {MAX_COMMAND_BYTES}",
                entry.command.encoded_len()
            )));
        }
        if self.stage_lengths.is_empty() {
            self.stage_start = entry.index;
        }
        let before = self.stage.len();
        entry.encode_record(&mut self.stage);
        let len = (self.stage.len() - before) as u32;
        self.stage_lengths.push(len);
        let index = entry.index;
        let pinned = entry.command.is_config_change();
        if pinned {
            self.config_indexes.push(index);
        }
        self.slots.push_back(LogSlot {
            term: entry.term,
            raw_offset: self.write_cursor,
            len,
            pinned,
            entry: Some(Arc::new(entry)),
        });
        self.write_cursor += u64::from(len);
        self.resident_bytes += len as usize;
        Ok(index)
    }

    /// Hands everything staged to the writer thread.
    ///
    /// Called at the end of a proposal batch and at the end of handling one
    /// AppendEntries, so a batch of either costs one fsync.
    ///
    /// The records are copied out at exactly their size and the staging
    /// buffer is kept. Handing the buffer over instead and allocating a
    /// replacement shipped its whole capacity with every flush, so a single
    /// hundred and fifty byte entry travelled as a sixty four kilobyte
    /// allocation, and a burst of concurrent proposals put hundreds of
    /// megabytes through the allocator to move a few hundred kilobytes of log
    pub fn flush_pending(&mut self) -> Result<()> {
        if self.stage_lengths.is_empty() {
            self.evict();
            return Ok(());
        }
        let blob = self.stage.as_slice().to_vec();
        self.stage.clear();
        let lengths = std::mem::take(&mut self.stage_lengths);
        let sent = self.handle.send(LogOp::Append {
            start_index: self.stage_start,
            blob,
            lengths,
        });
        self.evict();
        sent
    }

    /// Drops resident commands from the oldest forward until the cap is met.
    ///
    /// Only records the writer has already put on disk are candidates, because
    /// a page-in reads the file and a staged record is not in it yet. Oldest
    /// first rather than newest: the tail is what an in-step follower is
    /// asking for, and the head is what nobody has wanted since it committed
    fn evict(&mut self) {
        if self.resident_bytes <= self.resident_limit {
            return;
        }
        let persisted = self.handle.persisted_index();
        let last = self.last_index();
        if self.evict_next < self.start_index {
            self.evict_next = self.start_index;
        }
        while self.resident_bytes > self.resident_limit
            && self.evict_next <= persisted
            && self.evict_next <= last
        {
            let at = (self.evict_next - self.start_index) as usize;
            let slot = &mut self.slots[at];
            if !slot.pinned && slot.entry.take().is_some() {
                self.resident_bytes -= slot.len as usize;
            }
            self.evict_next += 1;
        }
    }

    /// Drops every entry after `index`.
    ///
    /// Staged records are flushed first so the writer sees the append and the
    /// truncation in the order they happened
    pub fn truncate_after(&mut self, index: u64) -> Result<()> {
        if index >= self.last_index() {
            return Ok(());
        }
        self.flush_pending()?;
        let keep = if index < self.prev_index() {
            0
        } else {
            (index - self.prev_index()) as usize
        };
        if let Some(slot) = self.slots.get(keep) {
            self.write_cursor = slot.raw_offset;
        }
        for slot in self.slots.iter().skip(keep) {
            if slot.entry.is_some() {
                self.resident_bytes -= slot.len as usize;
            }
        }
        self.slots.truncate(keep);
        self.config_indexes.retain(|&i| i <= index);
        let after = self.last_index() + 1;
        if self.evict_next > after {
            self.evict_next = after;
        }
        self.handle.send(LogOp::Truncate { index })
    }

    /// Discards everything a snapshot has made redundant
    pub fn compact_to(&mut self, index: u64, term: u64) -> Result<()> {
        if index < self.start_index {
            return Ok(());
        }
        self.flush_pending()?;
        let drop_count = ((index + 1).min(self.last_index() + 1) - self.start_index) as usize;
        for _ in 0..drop_count {
            if let Some(slot) = self.slots.pop_front() {
                if slot.entry.is_some() {
                    self.resident_bytes -= slot.len as usize;
                }
            }
        }
        self.start_index = index + 1;
        self.prev_term = term;
        self.config_indexes.retain(|&i| i > index);
        if self.evict_next < self.start_index {
            self.evict_next = self.start_index;
        }
        self.handle.send(LogOp::Compact {
            index,
            prev_term: term,
        })
    }

    /// Throws the whole log away and restarts it past a received snapshot
    pub fn reset_to(&mut self, index: u64, term: u64) -> Result<()> {
        self.stage.clear();
        self.stage_lengths.clear();
        self.slots.clear();
        self.config_indexes.clear();
        self.resident_bytes = 0;
        self.start_index = index + 1;
        self.prev_term = term;
        self.evict_next = self.start_index;
        self.handle.persisted.store(index, Ordering::Release);
        // The raw cursor keeps climbing across a reset. The writer turns it
        // into the new shift, so an offset recorded before the reset and one
        // recorded after it still convert to a file position the same way
        let raw_base = self.write_cursor;
        self.handle.send(LogOp::Reset {
            base_index: index + 1,
            prev_term: term,
            raw_base,
        })
    }

    /// A run of entries starting at `from`, bounded by count and by bytes.
    ///
    /// Resident entries are handed out behind an `Arc` so a batch to three
    /// followers shares one copy of each command rather than cloning it three
    /// times. Evicted ones come back as a descriptor the caller reads outside
    /// the consensus lock, so a follower far enough behind to need the disk
    /// does not stall the heartbeat that proves the leader still leads
    pub fn plan_slice(&self, from: u64, max_entries: usize, max_bytes: usize) -> Vec<SliceItem> {
        let mut out = Vec::new();
        if from > self.last_index() || from < self.start_index {
            return out;
        }
        let start = (from - self.start_index) as usize;
        let mut bytes = 0usize;
        for (i, slot) in self.slots.iter().enumerate().skip(start) {
            let len = slot.len as usize;
            // At least one entry always goes, so a command larger than the
            // byte budget still replicates rather than wedging the follower
            if !out.is_empty() && (out.len() >= max_entries || bytes + len > max_bytes) {
                break;
            }
            bytes += len;
            let index = self.start_index + i as u64;
            out.push(match slot.entry.as_ref() {
                Some(entry) => SliceItem::Resident(Arc::clone(entry)),
                None => SliceItem::Paged(PagedRecord {
                    index,
                    term: slot.term,
                    raw_offset: slot.raw_offset,
                    len: slot.len,
                }),
            });
            if out.len() >= max_entries {
                break;
            }
        }
        out
    }

    #[inline]
    pub fn persisted_index(&self) -> u64 {
        self.handle.persisted_index()
    }

    pub fn writer_handle(&self) -> LogWriterHandle {
        self.handle.clone()
    }

    /// Whether the file holds everything that has been appended
    pub fn is_durable(&self) -> bool {
        self.handle.persisted_index() >= self.last_index()
    }
}

impl Drop for RaftLog {
    fn drop(&mut self) {
        // Told to stop rather than left to notice the channel closing, because
        // other holders of the handle keep the channel open. Joining means the
        // file is closed by the time the directory is reopened or removed
        let _ = self.handle.tx.send(LogOp::Shutdown);
        if let Some(writer) = self.writer.take() {
            let _ = writer.join();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Large enough that a test which is not about eviction never trips it
    const TEST_RESIDENT_LIMIT: usize = 64 * 1024 * 1024;

    fn data_entry(term: u64, index: u64, size: usize) -> RaftLogEntry {
        RaftLogEntry::new(
            term,
            index,
            RaftCommand::Data {
                payload: vec![(index % 251) as u8; size],
            },
        )
    }

    /// The cap is on decoded commands, and it is met by dropping the oldest.
    /// The point of the whole arrangement is that the entries stay readable
    #[test]
    fn eviction_holds_the_cap_and_the_entries_still_read_back() {
        let dir = tempfile::tempdir().expect("tempdir");
        // One entry is 64KiB of payload, so sixteen of them cannot fit in a
        // cap of four
        let cap = 4 * 64 * 1024;
        let mut log = RaftLog::open(dir.path(), cap).expect("open");
        for index in 1..=16u64 {
            log.append(data_entry(1, index, 64 * 1024)).expect("append");
        }
        log.flush_pending().expect("flush");
        wait_durable(&log);
        // Flushing again runs the eviction pass against the now durable tail
        log.flush_pending().expect("flush");

        assert!(
            log.resident_bytes() <= cap,
            "resident {} is over the cap {cap}",
            log.resident_bytes()
        );
        assert!(
            log.entry(1).is_none(),
            "the oldest entry should have been evicted first"
        );
        assert!(
            log.entry(16).is_some(),
            "the newest entry is what a follower in step is asking for"
        );

        // Every entry still reads, evicted or not, and comes back identical
        for index in 1..=16u64 {
            let entry = log.get(index).expect("read back");
            assert_eq!(entry, data_entry(1, index, 64 * 1024));
        }
    }

    /// A batch that spans the residency boundary is planned under the lock and
    /// resolved outside it, so it must name every entry in order
    #[test]
    fn a_plan_names_evicted_and_resident_entries_in_order() {
        let dir = tempfile::tempdir().expect("tempdir");
        let cap = 2 * 64 * 1024;
        let mut log = RaftLog::open(dir.path(), cap).expect("open");
        for index in 1..=8u64 {
            log.append(data_entry(3, index, 64 * 1024)).expect("append");
        }
        log.flush_pending().expect("flush");
        wait_durable(&log);
        log.flush_pending().expect("flush");

        let plan = log.plan_slice(1, 8, 1 << 30);
        assert_eq!(plan.len(), 8);
        for (i, item) in plan.iter().enumerate() {
            assert_eq!(item.index(), i as u64 + 1);
        }
        assert!(
            plan.iter().any(|item| matches!(item, SliceItem::Paged(_))),
            "the head of the log should have been evicted"
        );

        let pager = log.pager();
        for item in &plan {
            let entry = match item {
                SliceItem::Resident(entry) => Arc::clone(entry),
                SliceItem::Paged(record) => pager
                    .read(record.index, record.term, record.raw_offset, record.len)
                    .expect("page in"),
            };
            assert_eq!(*entry, data_entry(3, entry.index, 64 * 1024));
        }
    }

    /// Compaction rewrites the file, so every offset recorded before it moves.
    /// A read after one must land on the same entries it did before
    #[test]
    fn a_page_in_survives_a_compaction() {
        let dir = tempfile::tempdir().expect("tempdir");
        let cap = 64 * 1024;
        let mut log = RaftLog::open(dir.path(), cap).expect("open");
        for index in 1..=12u64 {
            log.append(data_entry(2, index, 64 * 1024)).expect("append");
        }
        log.flush_pending().expect("flush");
        wait_durable(&log);
        log.flush_pending().expect("flush");

        log.compact_to(4, 2).expect("compact");
        wait_durable(&log);
        assert_eq!(log.first_index(), 5);

        for index in 5..=12u64 {
            let entry = log.get(index).expect("read back after compaction");
            assert_eq!(entry, data_entry(2, index, 64 * 1024));
        }
        assert!(
            log.get(4).is_err(),
            "an entry the snapshot covers is gone, not silently wrong"
        );
    }

    /// A reset restarts the file, and offsets recorded before it must not be
    /// read as though they still name a position in the new one
    #[test]
    fn a_page_in_survives_a_reset() {
        let dir = tempfile::tempdir().expect("tempdir");
        let cap = 64 * 1024;
        let mut log = RaftLog::open(dir.path(), cap).expect("open");
        for index in 1..=6u64 {
            log.append(data_entry(1, index, 64 * 1024)).expect("append");
        }
        log.flush_pending().expect("flush");
        wait_durable(&log);

        log.reset_to(40, 5).expect("reset");
        for index in 41..=52u64 {
            log.append(data_entry(6, index, 64 * 1024)).expect("append");
        }
        log.flush_pending().expect("flush");
        wait_durable(&log);
        log.flush_pending().expect("flush");

        assert_eq!(log.first_index(), 41);
        for index in 41..=52u64 {
            let entry = log.get(index).expect("read back after reset");
            assert_eq!(entry, data_entry(6, index, 64 * 1024));
        }
    }

    /// A restart must cost the memory of steady state, not the size of the
    /// log it just replayed
    #[test]
    fn a_replay_drops_back_to_the_cap() {
        let dir = tempfile::tempdir().expect("tempdir");
        let cap = 3 * 64 * 1024;
        {
            let mut log = RaftLog::open(dir.path(), cap).expect("open");
            for index in 1..=20u64 {
                log.append(data_entry(1, index, 64 * 1024)).expect("append");
            }
            log.flush_pending().expect("flush");
            wait_durable(&log);
        }
        let log = RaftLog::open(dir.path(), cap).expect("reopen");
        assert_eq!(log.last_index(), 20);
        assert!(
            log.resident_bytes() <= cap,
            "a replay left {} bytes resident against a cap of {cap}",
            log.resident_bytes()
        );
        for index in 1..=20u64 {
            assert_eq!(
                log.get(index).expect("read back"),
                data_entry(1, index, 64 * 1024)
            );
        }
    }

    /// Configuration changes are read by index while membership is rebuilt, so
    /// eviction has to leave them alone however far behind they fall
    #[test]
    fn a_configuration_change_is_never_evicted() {
        let dir = tempfile::tempdir().expect("tempdir");
        let cap = 64 * 1024;
        let mut log = RaftLog::open(dir.path(), cap).expect("open");
        log.append(RaftLogEntry::new(
            1,
            1,
            RaftCommand::AddNode {
                node_id: 7,
                address: "h:1".into(),
            },
        ))
        .expect("append");
        for index in 2..=16u64 {
            log.append(data_entry(1, index, 64 * 1024)).expect("append");
        }
        log.flush_pending().expect("flush");
        wait_durable(&log);
        log.flush_pending().expect("flush");

        assert!(log.is_config_change_at(1));
        assert!(
            log.entry(1).is_some(),
            "a voter change must stay in memory whatever the cap says"
        );
        assert!(!log.is_config_change_at(2));
    }

    /// The byte figure the snapshot trigger reads has to follow the file, not
    /// the memory the cap has already given back
    #[test]
    fn stored_bytes_tracks_the_file_rather_than_memory() {
        let dir = tempfile::tempdir().expect("tempdir");
        let cap = 64 * 1024;
        let mut log = RaftLog::open(dir.path(), cap).expect("open");
        for index in 1..=10u64 {
            log.append(data_entry(1, index, 64 * 1024)).expect("append");
        }
        log.flush_pending().expect("flush");
        wait_durable(&log);
        log.flush_pending().expect("flush");

        let stored = log.stored_bytes();
        assert!(
            stored >= 10 * 64 * 1024,
            "ten sixty four kilobyte entries occupy at least that, got {stored}"
        );
        assert!(log.resident_bytes() < stored as usize);

        log.compact_to(6, 1).expect("compact");
        let after = log.stored_bytes();
        assert!(
            after < stored,
            "compaction should have taken bytes off the log, {stored} to {after}"
        );
    }

    fn put(term: u64, index: u64, k: &str) -> RaftLogEntry {
        RaftLogEntry::new(
            term,
            index,
            RaftCommand::Put {
                key: k.as_bytes().to_vec(),
                value: vec![7u8; 32],
            },
        )
    }

    fn wait_durable(log: &RaftLog) {
        for _ in 0..2000 {
            if log.is_durable() {
                return;
            }
            std::thread::sleep(std::time::Duration::from_millis(1));
        }
        panic!("log never became durable");
    }

    /// The list of configuration entry positions is what membership is
    /// derived from, so every operation that moves the log has to keep it
    /// exact: append adds, truncation cuts the tail, compaction cuts the
    /// head, and a reopen rebuilds it from the file
    #[test]
    fn config_change_indexes_follow_the_log() {
        let dir = tempfile::tempdir().expect("tempdir");
        let config = |index: u64| {
            RaftLogEntry::new(
                1,
                index,
                RaftCommand::AddNode {
                    node_id: index,
                    address: format!("h:{index}"),
                },
            )
        };
        {
            let mut log = RaftLog::open(dir.path(), TEST_RESIDENT_LIMIT).expect("open");
            for index in 1..=8u64 {
                if index % 3 == 0 {
                    log.append(config(index)).expect("append");
                } else {
                    log.append(data_entry(1, index, 32)).expect("append");
                }
            }
            log.flush_pending().expect("flush");
            assert_eq!(log.config_change_indexes(), &[3, 6]);

            log.truncate_after(5).expect("truncate");
            assert_eq!(log.config_change_indexes(), &[3]);

            log.append(config(6)).expect("append");
            log.flush_pending().expect("flush");
            wait_durable(&log);
            log.compact_to(4, 1).expect("compact");
            assert_eq!(log.config_change_indexes(), &[6]);
        }
        // A reopen finds the surviving configuration entry from the file
        let log = RaftLog::open(dir.path(), TEST_RESIDENT_LIMIT).expect("reopen");
        assert_eq!(log.config_change_indexes(), &[6]);
    }

    #[test]
    fn every_command_round_trips() {
        let commands = vec![
            RaftCommand::Noop,
            RaftCommand::Put {
                key: b"k".to_vec(),
                value: b"v".to_vec(),
            },
            RaftCommand::Delete { key: b"k".to_vec() },
            RaftCommand::Data {
                payload: vec![1, 2, 3],
            },
            RaftCommand::AddNode {
                node_id: 9,
                address: "h:1".into(),
            },
            RaftCommand::RemoveNode { node_id: 9 },
            RaftCommand::Snapshot {
                last_included_index: 4,
                last_included_term: 2,
            },
            RaftCommand::JointConfig {
                old: ClusterConfig::of_voters([(1, "a:1".to_string())]),
                new: ClusterConfig::of_voters([(2, "b:1".to_string())]),
            },
            RaftCommand::FinalConfig {
                config: ClusterConfig::of_voters([(2, "b:1".to_string())]),
            },
        ];
        for command in commands {
            let entry = RaftLogEntry::new(3, 11, command.clone());
            let mut buf = Vec::new();
            entry.encode_record(&mut buf);
            let (back, used) = RaftLogEntry::decode_record(&buf).expect("decode");
            assert_eq!(used, buf.len());
            assert_eq!(back, entry);
        }
    }

    #[test]
    fn a_flipped_byte_fails_the_checksum() {
        let entry = put(1, 1, "key");
        let mut buf = Vec::new();
        entry.encode_record(&mut buf);
        let last = buf.len() - 1;
        buf[last] ^= 0xFF;
        assert!(RaftLogEntry::decode_record(&buf).is_err());
    }

    #[test]
    fn append_and_read_back_after_reopen() {
        let dir = tempfile::tempdir().expect("tempdir");
        {
            let mut log = RaftLog::open(dir.path(), TEST_RESIDENT_LIMIT).expect("open");
            for i in 1..=200u64 {
                log.append(put(1, i, &format!("k{i}"))).expect("append");
            }
            log.flush_pending().expect("flush");
            wait_durable(&log);
            assert_eq!(log.last_index(), 200);
        }
        let log = RaftLog::open(dir.path(), TEST_RESIDENT_LIMIT).expect("reopen");
        assert_eq!(log.last_index(), 200);
        assert_eq!(log.first_index(), 1);
        assert_eq!(log.get(200).expect("get").index, 200);
    }

    #[test]
    fn truncation_removes_the_tail_on_disk() {
        let dir = tempfile::tempdir().expect("tempdir");
        {
            let mut log = RaftLog::open(dir.path(), TEST_RESIDENT_LIMIT).expect("open");
            for i in 1..=50u64 {
                log.append(put(1, i, "k")).expect("append");
            }
            log.flush_pending().expect("flush");
            wait_durable(&log);
            log.truncate_after(20).expect("truncate");
            assert_eq!(log.last_index(), 20);
            for i in 21..=30u64 {
                log.append(put(2, i, "k")).expect("append");
            }
            log.flush_pending().expect("flush");
            wait_durable(&log);
        }
        let log = RaftLog::open(dir.path(), TEST_RESIDENT_LIMIT).expect("reopen");
        assert_eq!(log.last_index(), 30);
        assert_eq!(log.term_at(25), Some(2));
        assert_eq!(log.term_at(20), Some(1));
    }

    #[test]
    fn compaction_drops_the_head_and_survives_reopen() {
        let dir = tempfile::tempdir().expect("tempdir");
        {
            let mut log = RaftLog::open(dir.path(), TEST_RESIDENT_LIMIT).expect("open");
            for i in 1..=100u64 {
                log.append(put(3, i, "k")).expect("append");
            }
            log.flush_pending().expect("flush");
            wait_durable(&log);
            log.compact_to(60, 3).expect("compact");
            assert_eq!(log.first_index(), 61);
            assert_eq!(log.last_index(), 100);
            assert_eq!(log.term_at(60), Some(3));
            assert!(log.term_at(59).is_none());
            for i in 101..=110u64 {
                log.append(put(3, i, "k")).expect("append");
            }
            log.flush_pending().expect("flush");
            wait_durable(&log);
        }
        let log = RaftLog::open(dir.path(), TEST_RESIDENT_LIMIT).expect("reopen");
        assert_eq!(log.first_index(), 61);
        assert_eq!(log.last_index(), 110);
        assert_eq!(log.get(110).expect("get").index, 110);
    }

    #[test]
    fn a_torn_tail_is_cut_rather_than_kept() {
        let dir = tempfile::tempdir().expect("tempdir");
        {
            let mut log = RaftLog::open(dir.path(), TEST_RESIDENT_LIMIT).expect("open");
            for i in 1..=20u64 {
                log.append(put(1, i, "k")).expect("append");
            }
            log.flush_pending().expect("flush");
            wait_durable(&log);
        }
        // Half a record survives a power cut
        let path = dir.path().join(LOG_FILE);
        let len = std::fs::metadata(&path).expect("stat").len();
        let file = OpenOptions::new().write(true).open(&path).expect("open");
        file.set_len(len - 10).expect("tear");
        drop(file);

        let log = RaftLog::open(dir.path(), TEST_RESIDENT_LIMIT).expect("reopen");
        assert_eq!(log.last_index(), 19);
    }

    #[test]
    fn slice_respects_both_budgets() {
        let dir = tempfile::tempdir().expect("tempdir");
        let mut log = RaftLog::open(dir.path(), TEST_RESIDENT_LIMIT).expect("open");
        for i in 1..=100u64 {
            log.append(put(1, i, "k")).expect("append");
        }
        log.flush_pending().expect("flush");
        assert_eq!(log.plan_slice(1, 10, 1 << 20).len(), 10);
        let one = log.entry(1).expect("entry").encoded_len();
        assert_eq!(log.plan_slice(1, 1000, one * 3).len(), 3);
        // A budget smaller than one entry still moves one, so replication
        // cannot stall on an oversized command
        assert_eq!(log.plan_slice(1, 1000, 1).len(), 1);
    }

    #[test]
    fn reset_restarts_the_log_past_a_snapshot() {
        let dir = tempfile::tempdir().expect("tempdir");
        {
            let mut log = RaftLog::open(dir.path(), TEST_RESIDENT_LIMIT).expect("open");
            for i in 1..=30u64 {
                log.append(put(1, i, "k")).expect("append");
            }
            log.flush_pending().expect("flush");
            wait_durable(&log);
            log.reset_to(500, 9).expect("reset");
            assert_eq!(log.last_index(), 500);
            assert_eq!(log.last_term(), 9);
            log.append(put(9, 501, "k")).expect("append");
            log.flush_pending().expect("flush");
            wait_durable(&log);
        }
        let log = RaftLog::open(dir.path(), TEST_RESIDENT_LIMIT).expect("reopen");
        assert_eq!(log.first_index(), 501);
        assert_eq!(log.last_index(), 501);
        assert_eq!(log.prev_term(), 9);
    }

    #[test]
    fn an_out_of_sequence_append_is_refused() {
        let dir = tempfile::tempdir().expect("tempdir");
        let mut log = RaftLog::open(dir.path(), TEST_RESIDENT_LIMIT).expect("open");
        log.append(put(1, 1, "k")).expect("append");
        assert!(log.append(put(1, 3, "k")).is_err());
    }
}
