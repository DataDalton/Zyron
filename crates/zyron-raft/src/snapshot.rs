//! Replacing a log prefix with the state it produced.
//!
//! ## What a snapshot is here
//!
//! A checkpoint of the state machine plus the two numbers that say where in
//! the log it sits. Those numbers are the whole reason the file is useful to
//! consensus: without them the receiver cannot answer a later consistency
//! check, because it would hold state with no idea which entry produced it.
//!
//! The membership travels with it as well. A node that catches up from a
//! snapshot has never seen the configuration entries that the snapshot
//! covers, so it would otherwise come up believing it is in a group of one.
//!
//! ## Streaming, not loading
//!
//! A snapshot is as large as the database. It moves in chunks, one message per
//! chunk, and the sender reads the next chunk only after the previous one has
//! been acknowledged. The reply is the backpressure: a receiver on a slow disk
//! slows the sender rather than filling the sender's memory. Neither side ever
//! holds more than one chunk.
//!
//! Chunks go out on their own transport lane, so a gigabyte in flight does not
//! sit in front of the heartbeats that keep the group alive.

use std::fs::{File, OpenOptions};
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};

use zyron_common::error::{Result, ZyronError};
use zyron_common::format::envelope;
use zyron_common::format::{FormatKind, FormatVersion};

use crate::NodeId;
use crate::codec::{Cursor, put_bool, put_bytes, put_u64};
use crate::membership::ClusterConfig;

/// Where the pointer to the current snapshot lives
pub const SNAPSHOT_META_FILE: &str = "raft.snapshot.meta";
const SNAPSHOT_META_TMP: &str = "raft.snapshot.meta.tmp";
/// Version the snapshot pointer is written at, declared in the format registry
const META_VERSION: FormatVersion = crate::format::SNAPSHOT_TRANSFER_FORMAT_VERSION;

/// Where a snapshot sits in the log, and what it is worth.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SnapshotMeta {
    /// The last log index the checkpoint reflects
    pub last_included_index: u64,
    /// That entry's term, needed to answer the next consistency check
    pub last_included_term: u64,
    /// The membership in force at that index, so a node catching up from this
    /// snapshot knows who else is in the group
    pub config: ClusterConfig,
    /// Size of the data file, so a receiver can report progress
    pub size_bytes: u64,
}

impl SnapshotMeta {
    pub fn encode(&self, buf: &mut Vec<u8>) {
        put_u64(buf, self.last_included_index);
        put_u64(buf, self.last_included_term);
        put_u64(buf, self.size_bytes);
        self.config.encode(buf);
    }

    pub fn decode(c: &mut Cursor<'_>) -> Result<Self> {
        Ok(Self {
            last_included_index: c.u64()?,
            last_included_term: c.u64()?,
            size_bytes: c.u64()?,
            config: ClusterConfig::decode(c)?,
        })
    }
}

/// A snapshot on this node's disk.
#[derive(Debug, Clone)]
pub struct RaftSnapshot {
    pub meta: SnapshotMeta,
    /// The checkpoint file the state machine wrote
    pub data: PathBuf,
}

/// The snapshots this node holds, which is at most one.
///
/// Keeping only the newest is deliberate. An older snapshot can answer no
/// question the newer one cannot, and holding several would double the disk a
/// node needs for the same durability
pub struct SnapshotStore {
    dir: PathBuf,
    current: Option<RaftSnapshot>,
    /// The data file open for sending, with the path it was opened at. A
    /// transfer reads a thousand chunks from a gigabyte, and opening the
    /// file for each of them is a thousand opens for nothing
    sender: Option<(PathBuf, File)>,
}

impl SnapshotStore {
    /// Opens the store, adopting a snapshot whose data file is actually there.
    ///
    /// A pointer with no file behind it is what a crash between the two writes
    /// looks like. It is discarded rather than trusted, and the node falls
    /// back to its log
    pub fn open(dir: &Path) -> Result<Self> {
        std::fs::create_dir_all(dir)
            .map_err(|e| ZyronError::IoError(format!("create snapshot directory: {e}")))?;
        let _ = std::fs::remove_file(dir.join(SNAPSHOT_META_TMP));
        let meta_path = dir.join(SNAPSHOT_META_FILE);
        let mut store = Self {
            dir: dir.to_path_buf(),
            current: None,
            sender: None,
        };
        if meta_path.exists() {
            let bytes = std::fs::read(&meta_path)
                .map_err(|e| ZyronError::IoError(format!("read snapshot pointer: {e}")))?;
            match decode_meta_file(&bytes) {
                Ok(meta) => {
                    let data = store.data_path(meta.last_included_index);
                    if data.exists() {
                        store.current = Some(RaftSnapshot { meta, data });
                    } else {
                        tracing::warn!(
                            index = meta.last_included_index,
                            "snapshot pointer has no data file, falling back to the log"
                        );
                        let _ = std::fs::remove_file(&meta_path);
                    }
                }
                Err(e) => {
                    tracing::warn!(error = %e, "snapshot pointer is unreadable, falling back to the log");
                    let _ = std::fs::remove_file(&meta_path);
                }
            }
        }
        store.sweep_stale();
        Ok(store)
    }

    pub fn current(&self) -> Option<&RaftSnapshot> {
        self.current.as_ref()
    }

    /// The index a snapshot covers, or zero when there is none
    pub fn last_included_index(&self) -> u64 {
        self.current
            .as_ref()
            .map(|s| s.meta.last_included_index)
            .unwrap_or(0)
    }

    pub fn last_included_term(&self) -> u64 {
        self.current
            .as_ref()
            .map(|s| s.meta.last_included_term)
            .unwrap_or(0)
    }

    pub fn dir(&self) -> &Path {
        &self.dir
    }

    pub fn data_path(&self, index: u64) -> PathBuf {
        self.dir.join(format!("snapshot-{index:020}.zsnap"))
    }

    /// Where a snapshot being received is assembled, so a transfer that dies
    /// halfway never looks like a complete snapshot
    pub fn incoming_path(&self, index: u64) -> PathBuf {
        self.dir.join(format!("snapshot-{index:020}.incoming"))
    }

    /// Adopts a finished snapshot: the data file is renamed into place, the
    /// pointer is written, and anything older is removed.
    ///
    /// The pointer is written after the data, so a crash between the two
    /// leaves the previous snapshot in force rather than a pointer to a file
    /// that is still being written
    pub fn publish(&mut self, meta: SnapshotMeta, staged: &Path) -> Result<RaftSnapshot> {
        let data = self.data_path(meta.last_included_index);
        if staged != data {
            std::fs::rename(staged, &data)
                .map_err(|e| ZyronError::IoError(format!("move snapshot into place: {e}")))?;
        }
        let size_bytes = std::fs::metadata(&data)
            .map_err(|e| ZyronError::IoError(format!("stat snapshot: {e}")))?
            .len();
        let meta = SnapshotMeta { size_bytes, ..meta };
        self.write_meta(&meta)?;
        let snapshot = RaftSnapshot { meta, data };
        self.current = Some(snapshot.clone());
        // The file held open for sending is the one the sweep removes, and
        // an open file cannot be removed on every platform
        self.sender = None;
        self.sweep_stale();
        Ok(snapshot)
    }

    fn write_meta(&self, meta: &SnapshotMeta) -> Result<()> {
        let mut body = Vec::with_capacity(128);
        meta.encode(&mut body);
        let out = envelope::encode(FormatKind::SnapshotTransfer, META_VERSION, &body);

        let tmp = self.dir.join(SNAPSHOT_META_TMP);
        let mut file = OpenOptions::new()
            .create(true)
            .write(true)
            .truncate(true)
            .open(&tmp)
            .map_err(|e| ZyronError::IoError(format!("open snapshot pointer: {e}")))?;
        file.write_all(&out)
            .map_err(|e| ZyronError::IoError(format!("write snapshot pointer: {e}")))?;
        file.sync_all()
            .map_err(|e| ZyronError::IoError(format!("sync snapshot pointer: {e}")))?;
        drop(file);
        std::fs::rename(&tmp, self.dir.join(SNAPSHOT_META_FILE))
            .map_err(|e| ZyronError::IoError(format!("rename snapshot pointer: {e}")))
    }

    /// Removes snapshot data and part files that the current pointer does not
    /// name
    fn sweep_stale(&self) {
        let keep = self.current.as_ref().map(|s| s.data.clone());
        let Ok(entries) = std::fs::read_dir(&self.dir) else {
            return;
        };
        for entry in entries.flatten() {
            let path = entry.path();
            let Some(name) = path.file_name().and_then(|n| n.to_str()) else {
                continue;
            };
            if !name.starts_with("snapshot-") {
                continue;
            }
            if keep.as_deref() == Some(path.as_path()) {
                continue;
            }
            let _ = std::fs::remove_file(&path);
        }
    }

    /// Reads one chunk of the current snapshot.
    ///
    /// Opening per chunk rather than holding a handle keeps the sender
    /// stateless, so a transfer that is abandoned leaks nothing and a transfer
    /// that is restarted from a different offset needs no bookkeeping
    pub fn read_chunk(&mut self, offset: u64, max_bytes: usize) -> Result<(Vec<u8>, bool)> {
        let Some(snapshot) = self.current.as_ref() else {
            return Err(ZyronError::Internal(
                "asked for a snapshot chunk with no snapshot on this node".into(),
            ));
        };
        let reopen = !matches!(&self.sender, Some((path, _)) if *path == snapshot.data);
        if reopen {
            let file = File::open(&snapshot.data)
                .map_err(|e| ZyronError::IoError(format!("open snapshot for send: {e}")))?;
            self.sender = Some((snapshot.data.clone(), file));
        }
        let Some((_, file)) = self.sender.as_mut() else {
            return Err(ZyronError::Internal(
                "the snapshot file did not stay open".into(),
            ));
        };
        file.seek(SeekFrom::Start(offset))
            .map_err(|e| ZyronError::IoError(format!("seek snapshot: {e}")))?;
        let mut buf = vec![0u8; max_bytes];
        let mut filled = 0usize;
        while filled < max_bytes {
            let n = file
                .read(&mut buf[filled..])
                .map_err(|e| ZyronError::IoError(format!("read snapshot: {e}")))?;
            if n == 0 {
                break;
            }
            filled += n;
        }
        buf.truncate(filled);
        let done = offset + filled as u64 >= snapshot.meta.size_bytes;
        Ok((buf, done))
    }

    /// Writes one received chunk into the staging file.
    ///
    /// An offset that does not continue the file is refused rather than
    /// creating a hole, because a hole would produce a checkpoint that fails
    /// its own checksum much later and for no visible reason.
    ///
    /// `sync` is set only on the last chunk. A staging file is not a snapshot:
    /// it becomes one when it is renamed into place and the pointer beside it
    /// is written, and both of those happen after the final chunk. Forcing the
    /// platter on every megabyte protects nothing that an unfinished transfer
    /// could lose, and it halved the rate a gigabyte moved at
    pub fn write_chunk(&self, index: u64, offset: u64, data: &[u8], sync: bool) -> Result<u64> {
        let path = self.incoming_path(index);
        let mut file = if offset == 0 {
            OpenOptions::new()
                .create(true)
                .write(true)
                .truncate(true)
                .open(&path)
        } else {
            OpenOptions::new().write(true).open(&path)
        }
        .map_err(|e| ZyronError::IoError(format!("open incoming snapshot: {e}")))?;
        let have = file
            .metadata()
            .map_err(|e| ZyronError::IoError(format!("stat incoming snapshot: {e}")))?
            .len();
        if offset > have {
            return Err(ZyronError::Internal(format!(
                "snapshot chunk at offset {offset} would leave a hole, the file holds {have} bytes"
            )));
        }
        file.seek(SeekFrom::Start(offset))
            .map_err(|e| ZyronError::IoError(format!("seek incoming snapshot: {e}")))?;
        file.write_all(data)
            .map_err(|e| ZyronError::IoError(format!("write incoming snapshot: {e}")))?;
        if sync {
            file.sync_all()
                .map_err(|e| ZyronError::IoError(format!("sync incoming snapshot: {e}")))?;
        }
        Ok(offset + data.len() as u64)
    }

    /// Discards a transfer that will not finish
    pub fn abandon_incoming(&self, index: u64) {
        let _ = std::fs::remove_file(self.incoming_path(index));
    }
}

fn decode_meta_file(bytes: &[u8]) -> Result<SnapshotMeta> {
    let parsed = envelope::decode_as(bytes, FormatKind::SnapshotTransfer)
        .map_err(|e| ZyronError::RecoveryFailed(format!("snapshot pointer, {e}")))?;
    if parsed.header.version != META_VERSION {
        return Err(ZyronError::RecoveryFailed(format!(
            "snapshot pointer is at format version {}, this binary writes and reads \
             {META_VERSION}. Upgrade through a release that still reads {} to move it \
             forward first",
            parsed.header.version, parsed.header.version
        )));
    }
    SnapshotMeta::decode(&mut Cursor::new(parsed.body))
}

/// One chunk of a snapshot on its way to a follower.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct InstallSnapshotRequest {
    pub term: u64,
    pub leader_id: NodeId,
    pub last_included_index: u64,
    pub last_included_term: u64,
    /// Membership at the snapshot point, carried on every chunk so a receiver
    /// that joined the transfer late still gets it
    pub config: ClusterConfig,
    /// Byte offset this chunk starts at
    pub offset: u64,
    pub data: Vec<u8>,
    /// True on the last chunk, which is what tells the receiver to adopt it
    pub done: bool,
}

impl InstallSnapshotRequest {
    pub fn encode(&self, buf: &mut Vec<u8>) {
        put_u64(buf, self.term);
        put_u64(buf, self.leader_id);
        put_u64(buf, self.last_included_index);
        put_u64(buf, self.last_included_term);
        put_u64(buf, self.offset);
        put_bool(buf, self.done);
        self.config.encode(buf);
        put_bytes(buf, &self.data);
    }

    pub fn decode(c: &mut Cursor<'_>) -> Result<Self> {
        let term = c.u64()?;
        let leader_id = c.u64()?;
        let last_included_index = c.u64()?;
        let last_included_term = c.u64()?;
        let offset = c.u64()?;
        let done = c.bool()?;
        let config = ClusterConfig::decode(c)?;
        let data = c.vec()?;
        Ok(Self {
            term,
            leader_id,
            last_included_index,
            last_included_term,
            config,
            offset,
            data,
            done,
        })
    }
}

/// The receiver's answer to one chunk.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct InstallSnapshotReply {
    pub term: u64,
    pub success: bool,
    /// How much the receiver now holds, which is where the next chunk starts.
    /// Sent rather than assumed so a retried chunk does not duplicate bytes
    pub bytes_received: u64,
    pub follower_id: NodeId,
}

impl InstallSnapshotReply {
    pub fn encode(&self, buf: &mut Vec<u8>) {
        put_u64(buf, self.term);
        put_bool(buf, self.success);
        put_u64(buf, self.bytes_received);
        put_u64(buf, self.follower_id);
    }

    pub fn decode(c: &mut Cursor<'_>) -> Result<Self> {
        Ok(Self {
            term: c.u64()?,
            success: c.bool()?,
            bytes_received: c.u64()?,
            follower_id: c.u64()?,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn config() -> ClusterConfig {
        ClusterConfig::of_voters([(1, "a:1".to_string()), (2, "b:1".to_string())])
    }

    fn stage(dir: &Path, name: &str, bytes: &[u8]) -> PathBuf {
        let p = dir.join(name);
        std::fs::write(&p, bytes).expect("write");
        p
    }

    #[test]
    fn publish_then_reopen_finds_the_snapshot() {
        let dir = tempfile::tempdir().expect("tempdir");
        let staged = stage(dir.path(), "staged.tmp", &vec![3u8; 4096]);
        {
            let mut store = SnapshotStore::open(dir.path()).expect("open");
            assert!(store.current().is_none());
            let meta = SnapshotMeta {
                last_included_index: 700,
                last_included_term: 4,
                config: config(),
                size_bytes: 0,
            };
            let snap = store.publish(meta, &staged).expect("publish");
            assert_eq!(snap.meta.size_bytes, 4096);
        }
        let store = SnapshotStore::open(dir.path()).expect("reopen");
        let snap = store.current().expect("current");
        assert_eq!(snap.meta.last_included_index, 700);
        assert_eq!(snap.meta.last_included_term, 4);
        assert_eq!(snap.meta.config, config());
    }

    #[test]
    fn publishing_a_newer_snapshot_removes_the_older_file() {
        let dir = tempfile::tempdir().expect("tempdir");
        let mut store = SnapshotStore::open(dir.path()).expect("open");
        let first = stage(dir.path(), "one.tmp", &vec![1u8; 128]);
        store
            .publish(
                SnapshotMeta {
                    last_included_index: 10,
                    last_included_term: 1,
                    config: config(),
                    size_bytes: 0,
                },
                &first,
            )
            .expect("publish");
        let old = store.data_path(10);
        assert!(old.exists());

        let second = stage(dir.path(), "two.tmp", &vec![2u8; 128]);
        store
            .publish(
                SnapshotMeta {
                    last_included_index: 20,
                    last_included_term: 2,
                    config: config(),
                    size_bytes: 0,
                },
                &second,
            )
            .expect("publish");
        assert!(!old.exists());
        assert!(store.data_path(20).exists());
    }

    #[test]
    fn a_pointer_with_no_data_is_discarded() {
        let dir = tempfile::tempdir().expect("tempdir");
        {
            let mut store = SnapshotStore::open(dir.path()).expect("open");
            let staged = stage(dir.path(), "one.tmp", &vec![1u8; 64]);
            store
                .publish(
                    SnapshotMeta {
                        last_included_index: 5,
                        last_included_term: 1,
                        config: config(),
                        size_bytes: 0,
                    },
                    &staged,
                )
                .expect("publish");
        }
        std::fs::remove_file(dir.path().join("snapshot-00000000000000000005.zsnap"))
            .expect("remove");
        let store = SnapshotStore::open(dir.path()).expect("reopen");
        assert!(store.current().is_none());
    }

    #[test]
    fn chunks_stream_out_and_back_in_byte_for_byte() {
        let dir = tempfile::tempdir().expect("tempdir");
        let payload: Vec<u8> = (0..50_000u32).map(|i| (i % 251) as u8).collect();
        let staged = stage(dir.path(), "one.tmp", &payload);
        let mut store = SnapshotStore::open(dir.path()).expect("open");
        store
            .publish(
                SnapshotMeta {
                    last_included_index: 9,
                    last_included_term: 3,
                    config: config(),
                    size_bytes: 0,
                },
                &staged,
            )
            .expect("publish");

        let receiver_dir = tempfile::tempdir().expect("tempdir");
        let receiver = SnapshotStore::open(receiver_dir.path()).expect("open");
        let mut offset = 0u64;
        loop {
            let (chunk, done) = store.read_chunk(offset, 4096).expect("read");
            offset = receiver
                .write_chunk(9, offset, &chunk, done)
                .expect("write");
            if done {
                break;
            }
        }
        let got = std::fs::read(receiver.incoming_path(9)).expect("read back");
        assert_eq!(got, payload);
    }

    #[test]
    fn a_chunk_that_would_leave_a_hole_is_refused() {
        let dir = tempfile::tempdir().expect("tempdir");
        let store = SnapshotStore::open(dir.path()).expect("open");
        store.write_chunk(1, 0, &[1, 2, 3], false).expect("first");
        assert!(store.write_chunk(1, 100, &[4, 5], false).is_err());
    }

    #[test]
    fn install_messages_round_trip() {
        let req = InstallSnapshotRequest {
            term: 5,
            leader_id: 2,
            last_included_index: 900,
            last_included_term: 4,
            config: config(),
            offset: 4096,
            data: vec![7u8; 300],
            done: false,
        };
        let mut buf = Vec::new();
        req.encode(&mut buf);
        assert_eq!(
            InstallSnapshotRequest::decode(&mut Cursor::new(&buf)).expect("decode"),
            req
        );

        let reply = InstallSnapshotReply {
            term: 5,
            success: true,
            bytes_received: 4396,
            follower_id: 3,
        };
        let mut buf = Vec::new();
        reply.encode(&mut buf);
        assert_eq!(
            InstallSnapshotReply::decode(&mut Cursor::new(&buf)).expect("decode"),
            reply
        );
    }
}
