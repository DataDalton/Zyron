//! The thing consensus is protecting, and the seam it is reached through.
//!
//! Raft agrees on a sequence of commands. What those commands mean is not its
//! business, so the engine sits behind [`StateMachine`] and the consensus code
//! never names a table, a WAL record, or a key.
//!
//! ## Checkpoints are captured, then written
//!
//! Snapshotting has to be consistent with an exact log index, and the only
//! place that index is unambiguous is inside the apply loop. Writing a
//! multi-gigabyte checkpoint inside the apply loop would stall every apply
//! behind it for the duration.
//!
//! So the trait splits the two: [`StateMachine::begin_checkpoint`] captures
//! the state as of the applies made so far and must be fast, and the returned
//! [`CheckpointSource`] writes the bytes afterwards on a blocking thread while
//! applies carry on. [`MemoryStateMachine`] makes the capture cheap by holding
//! values behind `Arc`, so the capture clones pointers rather than payloads
//! and its cost is set by the key count rather than by the byte count.

use std::collections::HashMap;
use std::future::Future;
use std::io::{BufReader, BufWriter, Read, Write};
use std::path::Path;
use std::pin::Pin;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use zyron_common::checksum::{hash32, hash64};
use zyron_common::error::{Result, ZyronError};

use crate::log::{RaftCommand, RaftLogEntry};

/// What one apply pass returns: the highest index that reached the machine
pub type ApplyFuture<'a> = Pin<Box<dyn Future<Output = Result<u64>> + Send + 'a>>;

/// A captured state, waiting to be written.
///
/// Handed to a blocking thread by the apply loop, so it must be `Send` and
/// must not hold a lock on the machine it came from
pub trait CheckpointSource: Send {
    /// The log index this capture reflects
    fn last_applied(&self) -> u64;
    /// Writes the capture to `path` and returns the byte count
    fn write_to(&mut self, path: &Path) -> Result<u64>;
}

/// What committed entries are applied to.
pub trait StateMachine: Send + Sync {
    /// Applies one committed entry.
    ///
    /// Called in index order, exactly once per entry per node. An error here
    /// stops the apply loop rather than skipping the entry, because skipping
    /// would let two nodes diverge while both believing they applied the log
    fn apply(&self, index: u64, command: &RaftCommand) -> Result<()>;

    /// Applies a run of committed entries.
    ///
    /// This is what the apply loop calls. The default walks the run through
    /// [`Self::apply`], which is right for a machine that decides in memory.
    /// A machine that has to await something, a storage engine writing rows
    /// through its own transaction manager for instance, overrides this
    /// instead of blocking a runtime worker inside the synchronous form.
    ///
    /// Returns the highest index applied, which is not always the last entry
    /// in the run: an apply that fails part way reports how far it got so the
    /// loop does not claim entries that never landed
    fn apply_batch<'a>(&'a self, entries: &'a [Arc<RaftLogEntry>]) -> ApplyFuture<'a> {
        Box::pin(async move {
            let mut last = 0u64;
            for entry in entries {
                self.apply(entry.index, &entry.command)?;
                last = entry.index;
            }
            Ok(last)
        })
    }

    /// Captures the state as of the applies made so far
    fn begin_checkpoint(&self) -> Result<Box<dyn CheckpointSource>>;

    /// Replaces the state with a checkpoint written by another node
    fn restore(&self, path: &Path, last_included_index: u64) -> Result<()>;

    /// The highest index this machine has applied, for a machine whose state
    /// outlives the process. A machine that starts empty reports zero and the
    /// node replays from its snapshot
    fn applied_index(&self) -> u64 {
        0
    }

    /// The highest index the log may be compacted to without stranding this
    /// machine.
    ///
    /// A machine that stages multi-entry transactions has to see every entry
    /// of an unfinished one again after a restart, starting from its first.
    /// Compacting past that first entry would leave a restart needing entries
    /// nobody holds. A machine with no such state answers `u64::MAX`, which
    /// constrains nothing
    fn retain_floor(&self) -> u64 {
        u64::MAX
    }

    /// A value that differs when two nodes hold different state, used by
    /// operators and tests to compare replicas without shipping the data.
    ///
    /// Order independent, so two nodes that applied the same entries agree
    /// regardless of how their maps are laid out
    fn digest(&self) -> u64 {
        0
    }
}

const SM_MAGIC: [u8; 8] = *b"ZYRAFTSM";
const SM_VERSION: u32 = 1;
/// magic 8, version 4, pad 4, applied 8, count 8
const SM_HEADER_LEN: usize = 32;

/// A key value state machine held in memory.
///
/// The commands Raft replicates are `Put` and `Delete`, so this is the
/// complete implementation of them rather than a stand-in. Values sit behind
/// `Arc` so a checkpoint capture is a pointer copy per key
pub struct MemoryStateMachine {
    map: parking_lot::RwLock<HashMap<Arc<[u8]>, Arc<[u8]>>>,
    applied: AtomicU64,
    /// Order independent digest, maintained incrementally so comparing two
    /// nodes does not mean walking a million keys
    digest: AtomicU64,
    commands: AtomicU64,
}

impl Default for MemoryStateMachine {
    fn default() -> Self {
        Self::new()
    }
}

impl MemoryStateMachine {
    pub fn new() -> Self {
        Self {
            map: parking_lot::RwLock::new(HashMap::new()),
            applied: AtomicU64::new(0),
            digest: AtomicU64::new(0),
            commands: AtomicU64::new(0),
        }
    }

    pub fn get(&self, key: &[u8]) -> Option<Arc<[u8]>> {
        self.map.read().get(key).cloned()
    }

    pub fn len(&self) -> usize {
        self.map.read().len()
    }

    pub fn is_empty(&self) -> bool {
        self.map.read().is_empty()
    }

    /// How many commands have been applied, which is not the applied index
    /// because configuration entries and no-ops advance the index without
    /// touching the map
    pub fn commands_applied(&self) -> u64 {
        self.commands.load(Ordering::Relaxed)
    }

    /// The contribution one pair makes to the digest.
    ///
    /// Folded in with wrapping addition so removing a pair is subtracting the
    /// same value, and so the total does not depend on the order pairs were
    /// applied in
    fn pair_hash(key: &[u8], value: &[u8]) -> u64 {
        hash64(key).rotate_left(17) ^ hash64(value)
    }

    fn digest_add(&self, key: &[u8], value: &[u8]) {
        self.digest
            .fetch_add(Self::pair_hash(key, value), Ordering::Relaxed);
    }

    fn digest_remove(&self, key: &[u8], value: &[u8]) {
        self.digest
            .fetch_sub(Self::pair_hash(key, value), Ordering::Relaxed);
    }
}

impl StateMachine for MemoryStateMachine {
    fn apply(&self, index: u64, command: &RaftCommand) -> Result<()> {
        match command {
            RaftCommand::Put { key, value } => {
                let k: Arc<[u8]> = Arc::from(key.as_slice());
                let v: Arc<[u8]> = Arc::from(value.as_slice());
                let mut map = self.map.write();
                if let Some(old) = map.insert(Arc::clone(&k), Arc::clone(&v)) {
                    self.digest_remove(key, &old);
                }
                drop(map);
                self.digest_add(key, value);
                self.commands.fetch_add(1, Ordering::Relaxed);
            }
            RaftCommand::Delete { key } => {
                let mut map = self.map.write();
                let removed = map.remove(key.as_slice());
                drop(map);
                if let Some(old) = removed {
                    self.digest_remove(key, &old);
                }
                self.commands.fetch_add(1, Ordering::Relaxed);
            }
            // Everything else changes consensus state rather than user state.
            // The node applies those itself, and the machine only has to move
            // its applied index past them
            RaftCommand::Noop
            | RaftCommand::Data { .. }
            | RaftCommand::AddNode { .. }
            | RaftCommand::RemoveNode { .. }
            | RaftCommand::Snapshot { .. }
            | RaftCommand::JointConfig { .. }
            | RaftCommand::FinalConfig { .. } => {}
        }
        self.applied.store(index, Ordering::Release);
        Ok(())
    }

    fn begin_checkpoint(&self) -> Result<Box<dyn CheckpointSource>> {
        let map = self.map.read();
        let applied = self.applied.load(Ordering::Acquire);
        let entries: Vec<(Arc<[u8]>, Arc<[u8]>)> = map
            .iter()
            .map(|(k, v)| (Arc::clone(k), Arc::clone(v)))
            .collect();
        drop(map);
        Ok(Box::new(MemoryCheckpoint { applied, entries }))
    }

    fn restore(&self, path: &Path, last_included_index: u64) -> Result<()> {
        let file = std::fs::File::open(path)
            .map_err(|e| ZyronError::IoError(format!("open state machine checkpoint: {e}")))?;
        let mut reader = BufReader::with_capacity(1 << 20, file);

        let mut header = [0u8; SM_HEADER_LEN];
        reader
            .read_exact(&mut header)
            .map_err(|e| ZyronError::IoError(format!("read checkpoint header: {e}")))?;
        if header[0..8] != SM_MAGIC {
            return Err(ZyronError::RecoveryFailed(
                "state machine checkpoint magic does not match".into(),
            ));
        }
        let version = u32::from_le_bytes([header[8], header[9], header[10], header[11]]);
        if version != SM_VERSION {
            return Err(ZyronError::RecoveryFailed(format!(
                "state machine checkpoint version {version} is not {SM_VERSION}"
            )));
        }
        let applied = u64::from_le_bytes([
            header[16], header[17], header[18], header[19], header[20], header[21], header[22],
            header[23],
        ]);
        let count = u64::from_le_bytes([
            header[24], header[25], header[26], header[27], header[28], header[29], header[30],
            header[31],
        ]);

        let mut map: HashMap<Arc<[u8]>, Arc<[u8]>> = HashMap::with_capacity(count as usize);
        let mut digest = 0u64;
        let mut running = 0u32;
        let mut len_buf = [0u8; 4];
        for _ in 0..count {
            reader
                .read_exact(&mut len_buf)
                .map_err(|e| ZyronError::IoError(format!("read checkpoint key length: {e}")))?;
            let klen = u32::from_le_bytes(len_buf) as usize;
            let mut key = vec![0u8; klen];
            reader
                .read_exact(&mut key)
                .map_err(|e| ZyronError::IoError(format!("read checkpoint key: {e}")))?;
            reader
                .read_exact(&mut len_buf)
                .map_err(|e| ZyronError::IoError(format!("read checkpoint value length: {e}")))?;
            let vlen = u32::from_le_bytes(len_buf) as usize;
            let mut value = vec![0u8; vlen];
            reader
                .read_exact(&mut value)
                .map_err(|e| ZyronError::IoError(format!("read checkpoint value: {e}")))?;
            running = fold_checksum(running, &key, &value);
            digest = digest.wrapping_add(MemoryStateMachine::pair_hash(&key, &value));
            map.insert(Arc::from(key.as_slice()), Arc::from(value.as_slice()));
        }
        let mut trailer = [0u8; 4];
        reader
            .read_exact(&mut trailer)
            .map_err(|e| ZyronError::IoError(format!("read checkpoint trailer: {e}")))?;
        let stored = u32::from_le_bytes(trailer);
        if stored != running {
            return Err(ZyronError::RecoveryFailed(format!(
                "state machine checkpoint checksum {stored:#010x} does not match computed {running:#010x}"
            )));
        }

        *self.map.write() = map;
        self.digest.store(digest, Ordering::Release);
        // The index the leader stamped wins over the one in the file, because
        // it is the index the consensus layer will resume replication from
        self.applied
            .store(last_included_index.max(applied), Ordering::Release);
        Ok(())
    }

    fn applied_index(&self) -> u64 {
        self.applied.load(Ordering::Acquire)
    }

    fn digest(&self) -> u64 {
        self.digest.load(Ordering::Relaxed)
    }
}

/// Folds one pair into the running checkpoint checksum
fn fold_checksum(acc: u32, key: &[u8], value: &[u8]) -> u32 {
    acc.rotate_left(7) ^ hash32(key) ^ hash32(value).rotate_left(13)
}

struct MemoryCheckpoint {
    applied: u64,
    entries: Vec<(Arc<[u8]>, Arc<[u8]>)>,
}

impl CheckpointSource for MemoryCheckpoint {
    fn last_applied(&self) -> u64 {
        self.applied
    }

    fn write_to(&mut self, path: &Path) -> Result<u64> {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)
                .map_err(|e| ZyronError::IoError(format!("create checkpoint directory: {e}")))?;
        }
        let file = std::fs::File::create(path)
            .map_err(|e| ZyronError::IoError(format!("create checkpoint: {e}")))?;
        let mut writer = BufWriter::with_capacity(1 << 20, file);

        let mut header = [0u8; SM_HEADER_LEN];
        header[0..8].copy_from_slice(&SM_MAGIC);
        header[8..12].copy_from_slice(&SM_VERSION.to_le_bytes());
        header[16..24].copy_from_slice(&self.applied.to_le_bytes());
        header[24..32].copy_from_slice(&(self.entries.len() as u64).to_le_bytes());
        writer
            .write_all(&header)
            .map_err(|e| ZyronError::IoError(format!("write checkpoint header: {e}")))?;

        let mut written = SM_HEADER_LEN as u64;
        let mut running = 0u32;
        for (key, value) in &self.entries {
            writer
                .write_all(&(key.len() as u32).to_le_bytes())
                .map_err(|e| ZyronError::IoError(format!("write checkpoint key length: {e}")))?;
            writer
                .write_all(key)
                .map_err(|e| ZyronError::IoError(format!("write checkpoint key: {e}")))?;
            writer
                .write_all(&(value.len() as u32).to_le_bytes())
                .map_err(|e| ZyronError::IoError(format!("write checkpoint value length: {e}")))?;
            writer
                .write_all(value)
                .map_err(|e| ZyronError::IoError(format!("write checkpoint value: {e}")))?;
            running = fold_checksum(running, key, value);
            written += 8 + key.len() as u64 + value.len() as u64;
        }
        writer
            .write_all(&running.to_le_bytes())
            .map_err(|e| ZyronError::IoError(format!("write checkpoint trailer: {e}")))?;
        written += 4;
        let file = writer
            .into_inner()
            .map_err(|e| ZyronError::IoError(format!("flush checkpoint: {e}")))?;
        file.sync_all()
            .map_err(|e| ZyronError::IoError(format!("sync checkpoint: {e}")))?;
        Ok(written)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn put(k: &str, v: &str) -> RaftCommand {
        RaftCommand::Put {
            key: k.as_bytes().to_vec(),
            value: v.as_bytes().to_vec(),
        }
    }

    #[test]
    fn put_and_delete_move_the_map_and_the_digest() {
        let sm = MemoryStateMachine::new();
        sm.apply(1, &put("a", "1")).expect("apply");
        sm.apply(2, &put("b", "2")).expect("apply");
        assert_eq!(sm.len(), 2);
        assert_eq!(sm.get(b"a").as_deref(), Some(&b"1"[..]));
        let with_both = sm.digest();

        sm.apply(3, &RaftCommand::Delete { key: b"b".to_vec() })
            .expect("apply");
        assert_eq!(sm.len(), 1);
        assert_ne!(sm.digest(), with_both);

        // Putting it back reaches the same digest, which is the property that
        // makes the digest usable for comparing two replicas
        sm.apply(4, &put("b", "2")).expect("apply");
        assert_eq!(sm.digest(), with_both);
        assert_eq!(sm.applied_index(), 4);
    }

    #[test]
    fn overwriting_a_key_replaces_its_contribution() {
        let a = MemoryStateMachine::new();
        a.apply(1, &put("k", "one")).expect("apply");
        a.apply(2, &put("k", "two")).expect("apply");
        let b = MemoryStateMachine::new();
        b.apply(1, &put("k", "two")).expect("apply");
        assert_eq!(a.digest(), b.digest());
    }

    #[test]
    fn apply_order_does_not_change_the_digest() {
        let a = MemoryStateMachine::new();
        let b = MemoryStateMachine::new();
        for i in 0..100u32 {
            a.apply(i as u64 + 1, &put(&format!("k{i}"), &format!("v{i}")))
                .expect("apply");
        }
        for i in (0..100u32).rev() {
            b.apply(200 - i as u64, &put(&format!("k{i}"), &format!("v{i}")))
                .expect("apply");
        }
        assert_eq!(a.digest(), b.digest());
    }

    #[test]
    fn checkpoint_round_trips_into_a_fresh_machine() {
        let dir = tempfile::tempdir().expect("tempdir");
        let sm = MemoryStateMachine::new();
        for i in 0..500u32 {
            sm.apply(i as u64 + 1, &put(&format!("key{i}"), &format!("value{i}")))
                .expect("apply");
        }
        let mut source = sm.begin_checkpoint().expect("capture");
        assert_eq!(source.last_applied(), 500);
        let path = dir.path().join("snap.zsnap");
        let bytes = source.write_to(&path).expect("write");
        assert!(bytes > 0);

        let other = MemoryStateMachine::new();
        other.restore(&path, 500).expect("restore");
        assert_eq!(other.len(), sm.len());
        assert_eq!(other.digest(), sm.digest());
        assert_eq!(other.applied_index(), 500);
        assert_eq!(other.get(b"key42").as_deref(), Some(&b"value42"[..]));
    }

    #[test]
    fn a_damaged_checkpoint_is_refused() {
        let dir = tempfile::tempdir().expect("tempdir");
        let sm = MemoryStateMachine::new();
        sm.apply(1, &put("a", "1")).expect("apply");
        let path = dir.path().join("snap.zsnap");
        sm.begin_checkpoint()
            .expect("capture")
            .write_to(&path)
            .expect("write");
        let mut bytes = std::fs::read(&path).expect("read");
        let last = bytes.len() - 6;
        bytes[last] ^= 0xFF;
        std::fs::write(&path, &bytes).expect("write");
        assert!(MemoryStateMachine::new().restore(&path, 1).is_err());
    }

    #[test]
    fn a_capture_does_not_see_later_applies() {
        let dir = tempfile::tempdir().expect("tempdir");
        let sm = MemoryStateMachine::new();
        sm.apply(1, &put("a", "1")).expect("apply");
        let mut source = sm.begin_checkpoint().expect("capture");
        sm.apply(2, &put("b", "2")).expect("apply");
        let path = dir.path().join("snap.zsnap");
        source.write_to(&path).expect("write");

        let restored = MemoryStateMachine::new();
        restored.restore(&path, 1).expect("restore");
        assert_eq!(restored.len(), 1);
        assert!(restored.get(b"b").is_none());
    }
}
