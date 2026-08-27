//! Operator state management for streaming operators.
//!
//! Provides a custom lock-free FlatStateMap built on open-addressing with
//! Robin Hood probing for single-threaded operator use, plus RCU-based
//! concurrent reads for queryable state. No external hash map dependency.
//! The bottleneck is memory bandwidth, not software.
//!
//! Layout: contiguous flat arrays for metadata, hashes, keys, and values.
//! Snapshot is a bulk memory copy of the arrays. Restore replaces them.
//! Directly serializable for distributed state transfer.

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};

use zyron_common::fnv1a_64;
use zyron_common::{Result, ZyronError};

// ---------------------------------------------------------------------------
// StateSnapshot
// ---------------------------------------------------------------------------

/// Serialized snapshot of operator state for checkpointing.
/// Contains (namespace, key, value) triples.
#[derive(Debug, Clone)]
pub struct StateSnapshot {
    pub data: Vec<(Vec<u8>, Vec<u8>, Vec<u8>)>,
    pub snapshot_id: u64,
}

impl StateSnapshot {
    pub fn empty(snapshot_id: u64) -> Self {
        Self {
            data: Vec::new(),
            snapshot_id,
        }
    }

    /// Total size in bytes of all stored data.
    pub fn size_bytes(&self) -> usize {
        self.data
            .iter()
            .map(|(ns, k, v)| ns.len() + k.len() + v.len())
            .sum()
    }

    /// Number of entries.
    pub fn entry_count(&self) -> usize {
        self.data.len()
    }
}

// ---------------------------------------------------------------------------
// StateBackend trait
// ---------------------------------------------------------------------------

/// Trait for operator state storage backends.
pub trait StateBackend: Send + Sync {
    fn get(&self, namespace: &[u8], key: &[u8]) -> Result<Option<Vec<u8>>>;
    fn put(&self, namespace: &[u8], key: &[u8], value: &[u8]) -> Result<()>;
    fn delete(&self, namespace: &[u8], key: &[u8]) -> Result<()>;

    /// Scan all keys with the given prefix in a namespace.
    fn prefix_scan(&self, namespace: &[u8], prefix: &[u8]) -> Result<Vec<(Vec<u8>, Vec<u8>)>>;

    /// Create a snapshot of all state for checkpointing.
    fn snapshot(&self) -> Result<StateSnapshot>;

    /// Restore state from a checkpoint snapshot.
    fn restore(&self, snapshot: &StateSnapshot) -> Result<()>;

    /// Clear all entries in a namespace.
    fn clear_namespace(&self, namespace: &[u8]) -> Result<()>;

    /// Total number of entries across all namespaces.
    fn entry_count(&self) -> usize;

    /// Approximate size in bytes.
    fn size_bytes(&self) -> usize;
}

// ---------------------------------------------------------------------------
// Slot metadata for FlatStateMap
// ---------------------------------------------------------------------------

/// Per-slot metadata word.
/// Bits 0-30: probe distance from the ideal slot.
/// Bit 31: occupied flag (1 = occupied, 0 = empty).
/// A single byte capped the distance at 127; a probe chain past that
/// aliased distances through the mask and broke every lookup invariant,
/// so the word is wide enough for any distance a table can produce
const SLOT_EMPTY: u32 = 0;
const SLOT_OCCUPIED_BIT: u32 = 0x8000_0000;

#[inline(always)]
fn slot_is_empty(meta: u32) -> bool {
    meta == SLOT_EMPTY
}

#[inline(always)]
fn slot_is_occupied(meta: u32) -> bool {
    meta & SLOT_OCCUPIED_BIT != 0
}

#[inline(always)]
fn slot_distance(meta: u32) -> u32 {
    meta & !SLOT_OCCUPIED_BIT
}

#[inline(always)]
fn make_meta(distance: u32) -> u32 {
    debug_assert!(distance < SLOT_OCCUPIED_BIT);
    SLOT_OCCUPIED_BIT | distance
}

// ---------------------------------------------------------------------------
// FlatStateMap: open-addressing Robin Hood hash table
// ---------------------------------------------------------------------------

/// Maximum load factor before resize (87.5%).
const MAX_LOAD_NUMER: usize = 7;
const MAX_LOAD_DENOM: usize = 8;

/// Minimum capacity.
const MIN_CAPACITY: usize = 16;

/// Open-addressing hash table with Robin Hood probing.
///
/// Layout uses four parallel arrays for cache-optimal access:
/// - `meta`: 4 bytes per slot (occupied flag + probe distance)
/// - `hashes`: pre-computed u64 hash per slot (avoids re-hashing on probe)
/// - `keys`: (namespace, key) pair per slot
/// - `values`: value bytes per slot
///
/// Robin Hood probing bounds worst-case probe length to O(log n) by
/// displacing entries with shorter probe distances. This keeps the
/// table fast even at high load factors.
///
/// Single-threaded. No locks, no atomics on the hot path.
pub struct FlatStateMap {
    meta: Vec<u32>,
    hashes: Vec<u64>,
    keys: Vec<(Vec<u8>, Vec<u8>)>,
    values: Vec<Vec<u8>>,
    capacity: usize,
    mask: usize,
    len: usize,
    total_bytes: usize,
}

impl FlatStateMap {
    pub fn new() -> Self {
        Self::with_capacity(MIN_CAPACITY)
    }

    pub fn with_capacity(min_cap: usize) -> Self {
        let capacity = min_cap.next_power_of_two().max(MIN_CAPACITY);
        Self {
            meta: vec![SLOT_EMPTY; capacity],
            hashes: vec![0u64; capacity],
            keys: (0..capacity).map(|_| (Vec::new(), Vec::new())).collect(),
            values: (0..capacity).map(|_| Vec::new()).collect(),
            capacity,
            mask: capacity - 1,
            len: 0,
            total_bytes: 0,
        }
    }

    /// Compute the hash for a (namespace, key) pair.
    #[inline(always)]
    pub fn compute_hash(namespace: &[u8], key: &[u8]) -> u64 {
        let ns_hash = fnv1a_64(namespace);
        let key_hash = fnv1a_64(key);
        // Canonical combiner, the table is in-memory only so the combined
        // value carries no persistence contract
        zyron_common::hash_combine(ns_hash, key_hash)
    }

    pub fn get(&self, namespace: &[u8], key: &[u8]) -> Option<&Vec<u8>> {
        let hash = Self::compute_hash(namespace, key);
        let mut idx = (hash as usize) & self.mask;
        let mut dist: u32 = 0;

        loop {
            let m = self.meta[idx];
            if slot_is_empty(m) {
                return None;
            }
            // Robin Hood: if current slot's distance < our probe distance,
            // the key cannot be further ahead.
            if slot_distance(m) < dist {
                return None;
            }
            if self.hashes[idx] == hash && self.keys[idx].0 == namespace && self.keys[idx].1 == key
            {
                return Some(&self.values[idx]);
            }
            idx = (idx + 1) & self.mask;
            dist += 1;
        }
    }

    pub fn put(&mut self, namespace: &[u8], key: &[u8], value: &[u8]) {
        let hash = Self::compute_hash(namespace, key);
        let mut idx = (hash as usize) & self.mask;
        let mut dist: u32 = 0;

        // Single probe loop: find existing key or the Robin Hood insertion point.
        loop {
            let m = self.meta[idx];
            if slot_is_empty(m) {
                // Empty slot. Key does not exist. Insert here.
                self.len += 1;
                self.total_bytes += namespace.len() + key.len() + value.len();
                if self.len * MAX_LOAD_DENOM > self.capacity * MAX_LOAD_NUMER {
                    self.grow();
                    self.insert_inner(hash, (namespace.to_vec(), key.to_vec()), value.to_vec());
                    return;
                }
                self.meta[idx] = make_meta(dist);
                self.hashes[idx] = hash;
                self.keys[idx] = (namespace.to_vec(), key.to_vec());
                self.values[idx] = value.to_vec();
                return;
            }

            let existing_dist = slot_distance(m);

            // Check for matching key before checking Robin Hood condition,
            // since equal-distance slots could contain our key.
            if self.hashes[idx] == hash && self.keys[idx].0 == namespace && self.keys[idx].1 == key
            {
                // Update in-place.
                let old_size = self.values[idx].len();
                self.values[idx] = value.to_vec();
                self.total_bytes = self.total_bytes - old_size + value.len();
                return;
            }

            if existing_dist < dist {
                // Robin Hood condition: key does not exist. Insert here and
                // displace the current occupant forward.
                self.len += 1;
                self.total_bytes += namespace.len() + key.len() + value.len();
                if self.len * MAX_LOAD_DENOM > self.capacity * MAX_LOAD_NUMER {
                    self.grow();
                    self.insert_inner(hash, (namespace.to_vec(), key.to_vec()), value.to_vec());
                    return;
                }
                // Swap in our entry and continue displacing the evicted one.
                let mut cur_hash = hash;
                let mut cur_key = (namespace.to_vec(), key.to_vec());
                let mut cur_value = value.to_vec();
                let mut cur_dist = dist;
                let mut cur_idx = idx;
                let mut cur_existing_dist;

                // Displace the entry at cur_idx, then continue Robin Hood.
                loop {
                    let cm = self.meta[cur_idx];
                    if slot_is_empty(cm) {
                        self.meta[cur_idx] = make_meta(cur_dist);
                        self.hashes[cur_idx] = cur_hash;
                        self.keys[cur_idx] = cur_key;
                        self.values[cur_idx] = cur_value;
                        return;
                    }
                    cur_existing_dist = slot_distance(cm);
                    if cur_existing_dist < cur_dist {
                        self.meta[cur_idx] = make_meta(cur_dist);
                        std::mem::swap(&mut self.hashes[cur_idx], &mut cur_hash);
                        std::mem::swap(&mut self.keys[cur_idx], &mut cur_key);
                        std::mem::swap(&mut self.values[cur_idx], &mut cur_value);
                        cur_dist = cur_existing_dist;
                    }
                    cur_idx = (cur_idx + 1) & self.mask;
                    cur_dist += 1;
                }
            }

            idx = (idx + 1) & self.mask;
            dist += 1;
        }
    }

    /// Robin Hood insertion: insert entry, displacing entries with shorter
    /// probe distances along the way.
    fn insert_inner(&mut self, mut hash: u64, mut key: (Vec<u8>, Vec<u8>), mut value: Vec<u8>) {
        let mut idx = (hash as usize) & self.mask;
        let mut dist: u32 = 0;

        loop {
            let m = self.meta[idx];
            if slot_is_empty(m) {
                // Empty slot, place here.
                self.meta[idx] = make_meta(dist);
                self.hashes[idx] = hash;
                self.keys[idx] = key;
                self.values[idx] = value;
                return;
            }

            // Robin Hood: steal from the rich (shorter distance).
            let existing_dist = slot_distance(m);
            if existing_dist < dist {
                // Swap current entry with the one in this slot.
                self.meta[idx] = make_meta(dist);
                std::mem::swap(&mut self.hashes[idx], &mut hash);
                std::mem::swap(&mut self.keys[idx], &mut key);
                std::mem::swap(&mut self.values[idx], &mut value);
                dist = existing_dist;
            }

            idx = (idx + 1) & self.mask;
            dist += 1;
        }
    }

    pub fn delete(&mut self, namespace: &[u8], key: &[u8]) -> bool {
        let hash = Self::compute_hash(namespace, key);
        let mut idx = (hash as usize) & self.mask;
        let mut dist: u32 = 0;

        loop {
            let m = self.meta[idx];
            if slot_is_empty(m) {
                return false;
            }
            if slot_distance(m) < dist {
                return false;
            }
            if self.hashes[idx] == hash && self.keys[idx].0 == namespace && self.keys[idx].1 == key
            {
                let removed_size =
                    self.keys[idx].0.len() + self.keys[idx].1.len() + self.values[idx].len();
                self.total_bytes -= removed_size;
                self.len -= 1;

                // Backward-shift deletion: shift subsequent entries back to
                // fill the gap, maintaining Robin Hood invariant.
                self.meta[idx] = SLOT_EMPTY;
                let mut prev = idx;
                let mut cur = (idx + 1) & self.mask;
                loop {
                    let cm = self.meta[cur];
                    if slot_is_empty(cm) || slot_distance(cm) == 0 {
                        break;
                    }
                    // Shift entry back: decrease its probe distance by 1.
                    self.meta[prev] = make_meta(slot_distance(cm) - 1);
                    self.hashes[prev] = self.hashes[cur];
                    self.keys.swap(prev, cur);
                    self.values.swap(prev, cur);
                    self.meta[cur] = SLOT_EMPTY;
                    prev = cur;
                    cur = (cur + 1) & self.mask;
                }
                return true;
            }
            idx = (idx + 1) & self.mask;
            dist += 1;
        }
    }

    /// Grow the table to double capacity and re-insert all entries.
    fn grow(&mut self) {
        let new_capacity = self.capacity * 2;
        let old_meta = std::mem::replace(&mut self.meta, vec![SLOT_EMPTY; new_capacity]);
        let old_hashes = std::mem::replace(&mut self.hashes, vec![0u64; new_capacity]);
        let mut old_keys = std::mem::replace(
            &mut self.keys,
            (0..new_capacity)
                .map(|_| (Vec::new(), Vec::new()))
                .collect(),
        );
        let mut old_values = std::mem::replace(
            &mut self.values,
            (0..new_capacity).map(|_| Vec::new()).collect(),
        );

        let old_len = self.len;
        let old_total_bytes = self.total_bytes;
        self.capacity = new_capacity;
        self.mask = new_capacity - 1;
        self.len = 0;
        self.total_bytes = 0;

        for i in 0..old_meta.len() {
            if slot_is_occupied(old_meta[i]) {
                let key = std::mem::take(&mut old_keys[i]);
                let value = std::mem::take(&mut old_values[i]);
                self.insert_inner(old_hashes[i], key, value);
            }
        }
        self.len = old_len;
        self.total_bytes = old_total_bytes;
    }

    /// Iterate all entries. Callback receives (namespace, key, value).
    pub fn iter(&self, mut f: impl FnMut(&[u8], &[u8], &[u8])) {
        for i in 0..self.capacity {
            if slot_is_occupied(self.meta[i]) {
                f(&self.keys[i].0, &self.keys[i].1, &self.values[i]);
            }
        }
    }

    /// Bulk snapshot: iterate all entries into a Vec. Contiguous scan of
    /// parallel arrays, limited by memory bandwidth not software overhead.
    pub fn to_snapshot(&self, snapshot_id: u64) -> StateSnapshot {
        let mut data = Vec::with_capacity(self.len);
        for i in 0..self.capacity {
            if slot_is_occupied(self.meta[i]) {
                data.push((
                    self.keys[i].0.clone(),
                    self.keys[i].1.clone(),
                    self.values[i].clone(),
                ));
            }
        }
        StateSnapshot { data, snapshot_id }
    }

    /// Bulk restore: clear and re-insert all entries from snapshot.
    pub fn from_snapshot(&mut self, snapshot: &StateSnapshot) {
        // Reset to appropriately sized table.
        let needed = (snapshot.data.len() * MAX_LOAD_DENOM / MAX_LOAD_NUMER + 1)
            .next_power_of_two()
            .max(MIN_CAPACITY);
        *self = Self::with_capacity(needed);

        for (ns, key, val) in &snapshot.data {
            let hash = Self::compute_hash(ns, key);
            self.total_bytes += ns.len() + key.len() + val.len();
            self.len += 1;
            self.insert_inner(hash, (ns.clone(), key.clone()), val.clone());
        }
    }

    /// Remove all entries matching a namespace.
    /// Rebuilds the table keeping only non-matching entries (O(n) instead of O(n^2)).
    pub fn clear_namespace(&mut self, namespace: &[u8]) {
        let mut new_map = FlatStateMap::with_capacity(self.capacity);
        for i in 0..self.capacity {
            if slot_is_occupied(self.meta[i]) && self.keys[i].0 != namespace {
                let key = std::mem::take(&mut self.keys[i]);
                let value = std::mem::take(&mut self.values[i]);
                let hash = self.hashes[i];
                new_map.total_bytes += key.0.len() + key.1.len() + value.len();
                new_map.len += 1;
                new_map.insert_inner(hash, key, value);
            }
        }
        *self = new_map;
    }

    pub fn clear(&mut self) {
        self.meta.fill(SLOT_EMPTY);
        self.len = 0;
        self.total_bytes = 0;
    }

    /// Number of entries in the map.
    pub fn len(&self) -> usize {
        self.len
    }

    /// Returns true if the map contains no entries.
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Approximate total size in bytes of all stored keys and values.
    pub fn total_bytes(&self) -> usize {
        self.total_bytes
    }
}

// ---------------------------------------------------------------------------
// HeapStateBackend: FlatStateMap + RCU for queryable state
// ---------------------------------------------------------------------------

/// In-memory state backend using a custom FlatStateMap.
///
/// The core map uses open-addressing with Robin Hood probing. Single-threaded
/// on the write path (no locks, no atomics). For concurrent reads (queryable
/// state, metrics), a parking_lot::RwLock protects the map. The RwLock is
/// only contended during external reads, never on the operator's own path
/// since the operator holds &mut self.
pub struct HeapStateBackend {
    store: parking_lot::RwLock<FlatStateMap>,
    next_snapshot_id: AtomicU64,
}

impl HeapStateBackend {
    pub fn new() -> Self {
        Self {
            store: parking_lot::RwLock::new(FlatStateMap::new()),
            next_snapshot_id: AtomicU64::new(1),
        }
    }

    /// Creates a backend pre-sized for the expected number of entries.
    /// Avoids resize overhead during initial population.
    pub fn with_capacity(expected_entries: usize) -> Self {
        Self {
            store: parking_lot::RwLock::new(FlatStateMap::with_capacity(expected_entries)),
            next_snapshot_id: AtomicU64::new(1),
        }
    }
}

impl Default for HeapStateBackend {
    fn default() -> Self {
        Self::new()
    }
}

impl StateBackend for HeapStateBackend {
    fn get(&self, namespace: &[u8], key: &[u8]) -> Result<Option<Vec<u8>>> {
        let guard = self.store.read();
        Ok(guard.get(namespace, key).cloned())
    }

    fn put(&self, namespace: &[u8], key: &[u8], value: &[u8]) -> Result<()> {
        let mut guard = self.store.write();
        guard.put(namespace, key, value);
        Ok(())
    }

    fn delete(&self, namespace: &[u8], key: &[u8]) -> Result<()> {
        let mut guard = self.store.write();
        guard.delete(namespace, key);
        Ok(())
    }

    fn prefix_scan(&self, namespace: &[u8], prefix: &[u8]) -> Result<Vec<(Vec<u8>, Vec<u8>)>> {
        let guard = self.store.read();
        let mut results = Vec::new();
        guard.iter(|ns, key, val| {
            if ns == namespace && key.starts_with(prefix) {
                results.push((key.to_vec(), val.to_vec()));
            }
        });
        Ok(results)
    }

    fn snapshot(&self) -> Result<StateSnapshot> {
        let snapshot_id = self.next_snapshot_id.fetch_add(1, Ordering::Relaxed);
        let guard = self.store.read();
        Ok(guard.to_snapshot(snapshot_id))
    }

    fn restore(&self, snapshot: &StateSnapshot) -> Result<()> {
        let mut guard = self.store.write();
        guard.from_snapshot(snapshot);
        Ok(())
    }

    fn clear_namespace(&self, namespace: &[u8]) -> Result<()> {
        let mut guard = self.store.write();
        guard.clear_namespace(namespace);
        Ok(())
    }

    fn entry_count(&self) -> usize {
        self.store.read().len()
    }

    fn size_bytes(&self) -> usize {
        self.store.read().total_bytes()
    }
}

// ---------------------------------------------------------------------------
// DiskStateBackend: FlatStateMap write buffer + file-backed storage
// ---------------------------------------------------------------------------

/// Disk-backed state backend for large state that exceeds memory.
///
/// A FlatStateMap write buffer absorbs writes and flushes to timestamped
/// segment files once it crosses the byte cap. Reads merge the buffer over
/// the segments newest first, deletes write tombstones so a flushed value
/// stays dead, and a flush that leaves too many segments compacts them all
/// into one, dropping tombstones. Everything runs under the single state
/// mutex, matching the one-operator-thread ownership model.
///
/// Segment record layout, all little-endian:
///   [ns_len u32][ns][key_len u32][key][tag u8][val_len u32][val]
/// where tag 1 is a live value and tag 0 a tombstone.
pub struct DiskStateBackend {
    data_dir: PathBuf,
    write_buffer: parking_lot::Mutex<FlatStateMap>,
    write_buffer_bytes: AtomicUsize,
    max_write_buffer_bytes: usize,
    next_snapshot_id: AtomicU64,
}

/// Segments a flush leaves behind before the next flush compacts them all.
const DISK_STATE_COMPACT_AT: usize = 4;

const TAG_VALUE: u8 = 1;
const TAG_TOMBSTONE: u8 = 0;

impl DiskStateBackend {
    pub fn new(data_dir: &Path, max_write_buffer_bytes: usize) -> Result<Self> {
        std::fs::create_dir_all(data_dir)
            .map_err(|e| ZyronError::StreamingError(format!("failed to create state dir: {e}")))?;
        Ok(Self {
            data_dir: data_dir.to_path_buf(),
            write_buffer: parking_lot::Mutex::new(FlatStateMap::new()),
            write_buffer_bytes: AtomicUsize::new(0),
            max_write_buffer_bytes,
            next_snapshot_id: AtomicU64::new(1),
        })
    }

    /// Segment paths sorted oldest first by their timestamp names.
    fn segment_paths(&self) -> Result<Vec<PathBuf>> {
        let mut paths = Vec::new();
        let dir = std::fs::read_dir(&self.data_dir)
            .map_err(|e| ZyronError::StreamingError(format!("state dir read failed: {e}")))?;
        for entry in dir {
            let entry =
                entry.map_err(|e| ZyronError::StreamingError(format!("state dir entry: {e}")))?;
            let name = entry.file_name();
            let Some(name) = name.to_str() else { continue };
            if name.starts_with("state_") && name.ends_with(".dat") {
                paths.push(entry.path());
            }
        }
        paths.sort();
        Ok(paths)
    }

    /// Walks one segment's records in file order.
    fn walk_segment(path: &Path, mut f: impl FnMut(&[u8], &[u8], u8, &[u8])) -> Result<()> {
        let data = std::fs::read(path)
            .map_err(|e| ZyronError::StreamingError(format!("state segment read failed: {e}")))?;
        let mut off = 0usize;
        let read_len = |data: &[u8], off: &mut usize| -> Result<usize> {
            if *off + 4 > data.len() {
                return Err(ZyronError::StreamingError(format!(
                    "state segment {} truncated",
                    path.display()
                )));
            }
            let n = u32::from_le_bytes(data[*off..*off + 4].try_into().unwrap_or([0; 4])) as usize;
            *off += 4;
            Ok(n)
        };
        while off < data.len() {
            let ns_len = read_len(&data, &mut off)?;
            let ns_end = off + ns_len;
            let key_len_off = ns_end;
            if key_len_off > data.len() {
                return Err(ZyronError::StreamingError(format!(
                    "state segment {} truncated",
                    path.display()
                )));
            }
            let ns_range = off..ns_end;
            off = key_len_off;
            let key_len = read_len(&data, &mut off)?;
            let key_end = off + key_len;
            if key_end + 1 > data.len() {
                return Err(ZyronError::StreamingError(format!(
                    "state segment {} truncated",
                    path.display()
                )));
            }
            let key_range = off..key_end;
            let tag = data[key_end];
            off = key_end + 1;
            let val_len = read_len(&data, &mut off)?;
            let val_end = off + val_len;
            if val_end > data.len() {
                return Err(ZyronError::StreamingError(format!(
                    "state segment {} truncated",
                    path.display()
                )));
            }
            f(
                &data[ns_range.clone()],
                &data[key_range.clone()],
                tag,
                &data[off..val_end],
            );
            off = val_end;
        }
        Ok(())
    }

    /// The full merged view: every segment oldest first, buffer last, so a
    /// newer write or tombstone always wins. Tombstoned keys are dropped.
    fn merged_view(
        &self,
        buffer: &FlatStateMap,
    ) -> Result<std::collections::BTreeMap<(Vec<u8>, Vec<u8>), Vec<u8>>> {
        let mut merged: std::collections::BTreeMap<(Vec<u8>, Vec<u8>), Vec<u8>> =
            std::collections::BTreeMap::new();
        for path in self.segment_paths()? {
            Self::walk_segment(&path, |ns, key, tag, val| {
                let k = (ns.to_vec(), key.to_vec());
                if tag == TAG_VALUE {
                    merged.insert(k, val.to_vec());
                } else {
                    merged.remove(&k);
                }
            })?;
        }
        buffer.iter(|ns, key, val| {
            let k = (ns.to_vec(), key.to_vec());
            match val.split_first() {
                Some((&TAG_VALUE, rest)) => {
                    merged.insert(k, rest.to_vec());
                }
                _ => {
                    merged.remove(&k);
                }
            }
        });
        Ok(merged)
    }

    fn flush_if_needed(&self) -> Result<()> {
        if self.write_buffer_bytes.load(Ordering::Relaxed) >= self.max_write_buffer_bytes {
            self.flush()?;
        }
        Ok(())
    }

    fn segment_file_name(&self) -> PathBuf {
        let timestamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos();
        self.data_dir.join(format!("state_{timestamp:032}.dat"))
    }

    fn flush(&self) -> Result<()> {
        let mut guard = self.write_buffer.lock();
        if guard.len() == 0 {
            return Ok(());
        }

        // Collect sorted entries, tag byte already inside the stored value.
        let mut entries: Vec<(Vec<u8>, Vec<u8>, Vec<u8>)> = Vec::with_capacity(guard.len());
        guard.iter(|ns, key, val| {
            entries.push((ns.to_vec(), key.to_vec(), val.to_vec()));
        });
        entries.sort();

        let mut data = Vec::new();
        for (ns, key, tagged) in &entries {
            let (tag, val) = tagged
                .split_first()
                .map(|(t, rest)| (*t, rest))
                .unwrap_or((TAG_TOMBSTONE, &[][..]));
            data.extend_from_slice(&(ns.len() as u32).to_le_bytes());
            data.extend_from_slice(ns);
            data.extend_from_slice(&(key.len() as u32).to_le_bytes());
            data.extend_from_slice(key);
            data.push(tag);
            data.extend_from_slice(&(val.len() as u32).to_le_bytes());
            data.extend_from_slice(val);
        }

        std::fs::write(self.segment_file_name(), &data)
            .map_err(|e| ZyronError::StreamingError(format!("state flush failed: {e}")))?;

        // The buffer clears only after the segment is durable on disk, a
        // failed write keeps the state readable in memory
        guard.clear();
        self.write_buffer_bytes.store(0, Ordering::Relaxed);

        // Compaction folds every segment into one, dropping tombstones, so
        // segment count and dead data stay bounded
        let paths = self.segment_paths()?;
        if paths.len() >= DISK_STATE_COMPACT_AT {
            let merged = self.merged_view(&guard)?;
            let mut data = Vec::new();
            for ((ns, key), val) in &merged {
                data.extend_from_slice(&(ns.len() as u32).to_le_bytes());
                data.extend_from_slice(ns);
                data.extend_from_slice(&(key.len() as u32).to_le_bytes());
                data.extend_from_slice(key);
                data.push(TAG_VALUE);
                data.extend_from_slice(&(val.len() as u32).to_le_bytes());
                data.extend_from_slice(val);
            }
            std::fs::write(self.segment_file_name(), &data).map_err(|e| {
                ZyronError::StreamingError(format!("state compaction write failed: {e}"))
            })?;
            for path in paths {
                std::fs::remove_file(&path).map_err(|e| {
                    ZyronError::StreamingError(format!("state segment remove failed: {e}"))
                })?;
            }
        }
        Ok(())
    }

    /// Removes every segment file. Used when a restore or namespace clear
    /// replaces the merged state wholesale.
    fn remove_all_segments(&self) -> Result<()> {
        for path in self.segment_paths()? {
            std::fs::remove_file(&path).map_err(|e| {
                ZyronError::StreamingError(format!("state segment remove failed: {e}"))
            })?;
        }
        Ok(())
    }
}

impl StateBackend for DiskStateBackend {
    fn get(&self, namespace: &[u8], key: &[u8]) -> Result<Option<Vec<u8>>> {
        let guard = self.write_buffer.lock();
        if let Some(tagged) = guard.get(namespace, key) {
            return Ok(match tagged.split_first() {
                Some((&TAG_VALUE, rest)) => Some(rest.to_vec()),
                _ => None,
            });
        }
        // Miss in the buffer: newest segment holding the key decides
        let mut found: Option<Option<Vec<u8>>> = None;
        for path in self.segment_paths()? {
            Self::walk_segment(&path, |ns, k, tag, val| {
                if ns == namespace && k == key {
                    found = Some(if tag == TAG_VALUE {
                        Some(val.to_vec())
                    } else {
                        None
                    });
                }
            })?;
        }
        Ok(found.flatten())
    }

    fn put(&self, namespace: &[u8], key: &[u8], value: &[u8]) -> Result<()> {
        let entry_size = namespace.len() + key.len() + value.len() + 1;
        {
            let mut guard = self.write_buffer.lock();
            let mut tagged = Vec::with_capacity(value.len() + 1);
            tagged.push(TAG_VALUE);
            tagged.extend_from_slice(value);
            guard.put(namespace, key, &tagged);
        }
        self.write_buffer_bytes
            .fetch_add(entry_size, Ordering::Relaxed);
        self.flush_if_needed()?;
        Ok(())
    }

    fn delete(&self, namespace: &[u8], key: &[u8]) -> Result<()> {
        // A tombstone rather than a buffer removal, so a value already
        // flushed to a segment stays dead
        let mut guard = self.write_buffer.lock();
        guard.put(namespace, key, &[TAG_TOMBSTONE]);
        Ok(())
    }

    fn prefix_scan(&self, namespace: &[u8], prefix: &[u8]) -> Result<Vec<(Vec<u8>, Vec<u8>)>> {
        let guard = self.write_buffer.lock();
        let merged = self.merged_view(&guard)?;
        Ok(merged
            .into_iter()
            .filter(|((ns, key), _)| ns == namespace && key.starts_with(prefix))
            .map(|((_, key), val)| (key, val))
            .collect())
    }

    fn snapshot(&self) -> Result<StateSnapshot> {
        let snapshot_id = self.next_snapshot_id.fetch_add(1, Ordering::Relaxed);
        let guard = self.write_buffer.lock();
        let merged = self.merged_view(&guard)?;
        let mut full = FlatStateMap::with_capacity(merged.len());
        for ((ns, key), val) in &merged {
            full.put(ns, key, val);
        }
        Ok(full.to_snapshot(snapshot_id))
    }

    fn restore(&self, snapshot: &StateSnapshot) -> Result<()> {
        let mut guard = self.write_buffer.lock();
        self.remove_all_segments()?;
        guard.from_snapshot(snapshot);
        // Snapshot values are raw, re-tag them as live buffer values
        let mut entries: Vec<(Vec<u8>, Vec<u8>, Vec<u8>)> = Vec::with_capacity(guard.len());
        guard.iter(|ns, key, val| {
            entries.push((ns.to_vec(), key.to_vec(), val.to_vec()));
        });
        guard.clear();
        for (ns, key, val) in &entries {
            let mut tagged = Vec::with_capacity(val.len() + 1);
            tagged.push(TAG_VALUE);
            tagged.extend_from_slice(val);
            guard.put(ns, key, &tagged);
        }
        self.write_buffer_bytes
            .store(guard.total_bytes(), Ordering::Relaxed);
        Ok(())
    }

    fn clear_namespace(&self, namespace: &[u8]) -> Result<()> {
        let mut guard = self.write_buffer.lock();
        // Flushed entries of this namespace must die too, so every merged
        // key in the namespace gets a tombstone before the buffer copy of
        // the namespace is dropped
        let merged = self.merged_view(&guard)?;
        guard.clear_namespace(namespace);
        for (ns, key) in merged.keys() {
            if ns == namespace {
                guard.put(ns, key, &[TAG_TOMBSTONE]);
            }
        }
        self.write_buffer_bytes
            .store(guard.total_bytes(), Ordering::Relaxed);
        Ok(())
    }

    fn entry_count(&self) -> usize {
        let guard = self.write_buffer.lock();
        self.merged_view(&guard).map(|m| m.len()).unwrap_or(0)
    }

    fn size_bytes(&self) -> usize {
        self.write_buffer_bytes.load(Ordering::Relaxed)
    }
}

// ---------------------------------------------------------------------------
// QueryableState
// ---------------------------------------------------------------------------

/// Wraps a StateBackend for SQL-accessible point lookups.
/// SQL: SELECT * FROM STREAMING_STATE('name') WHERE key = 'x'
pub struct QueryableState {
    backend: Arc<dyn StateBackend>,
    namespace: Vec<u8>,
    name: String,
}

impl QueryableState {
    pub fn new(backend: Arc<dyn StateBackend>, namespace: Vec<u8>, name: String) -> Self {
        Self {
            backend,
            namespace,
            name,
        }
    }

    pub fn get(&self, key: &[u8]) -> Result<Option<Vec<u8>>> {
        self.backend.get(&self.namespace, key)
    }

    pub fn scan_prefix(&self, prefix: &[u8]) -> Result<Vec<(Vec<u8>, Vec<u8>)>> {
        self.backend.prefix_scan(&self.namespace, prefix)
    }

    pub fn name(&self) -> &str {
        &self.name
    }

    pub fn entry_count(&self) -> usize {
        self.backend.entry_count()
    }

    pub fn size_bytes(&self) -> usize {
        self.backend.size_bytes()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // -- FlatStateMap direct tests --

    #[test]
    fn test_flat_map_basic_crud() {
        let mut map = FlatStateMap::new();
        map.put(b"ns", b"key1", b"val1");
        map.put(b"ns", b"key2", b"val2");
        assert_eq!(map.len, 2);

        assert_eq!(map.get(b"ns", b"key1"), Some(&b"val1".to_vec()));
        assert_eq!(map.get(b"ns", b"key2"), Some(&b"val2".to_vec()));
        assert_eq!(map.get(b"ns", b"missing"), None);

        // Update.
        map.put(b"ns", b"key1", b"updated");
        assert_eq!(map.get(b"ns", b"key1"), Some(&b"updated".to_vec()));
        assert_eq!(map.len, 2);

        // Delete.
        assert!(map.delete(b"ns", b"key1"));
        assert_eq!(map.get(b"ns", b"key1"), None);
        assert_eq!(map.len, 1);

        assert!(!map.delete(b"ns", b"nonexistent"));
    }

    #[test]
    fn test_flat_map_grow() {
        let mut map = FlatStateMap::with_capacity(16);
        // Insert enough entries to trigger at least one resize.
        for i in 0u32..100 {
            let key = i.to_le_bytes();
            let val = (i * 10).to_le_bytes();
            map.put(b"ns", &key, &val);
        }
        assert_eq!(map.len, 100);
        assert!(map.capacity >= 128);

        // Verify all entries survived the resize.
        for i in 0u32..100 {
            let key = i.to_le_bytes();
            let expected = (i * 10).to_le_bytes();
            assert_eq!(map.get(b"ns", &key), Some(&expected.to_vec()));
        }
    }

    #[test]
    fn test_flat_map_robin_hood_correctness() {
        // Insert many entries with the same hash prefix to stress Robin Hood.
        let mut map = FlatStateMap::with_capacity(64);
        for i in 0u32..50 {
            let key = format!("key_{i}");
            let val = format!("val_{i}");
            map.put(b"ns", key.as_bytes(), val.as_bytes());
        }
        // Delete from the middle and verify remaining entries.
        for i in (0u32..50).step_by(3) {
            let key = format!("key_{i}");
            assert!(map.delete(b"ns", key.as_bytes()));
        }
        for i in 0u32..50 {
            let key = format!("key_{i}");
            if i % 3 == 0 {
                assert_eq!(map.get(b"ns", key.as_bytes()), None);
            } else {
                let expected = format!("val_{i}");
                assert_eq!(map.get(b"ns", key.as_bytes()), Some(&expected.into_bytes()));
            }
        }
    }

    #[test]
    fn test_flat_map_snapshot_restore() {
        let mut map = FlatStateMap::new();
        for i in 0u32..1000 {
            map.put(b"ns", &i.to_le_bytes(), &(i * 2).to_le_bytes());
        }
        let snapshot = map.to_snapshot(1);
        assert_eq!(snapshot.entry_count(), 1000);

        let mut map2 = FlatStateMap::new();
        map2.from_snapshot(&snapshot);
        assert_eq!(map2.len, 1000);
        for i in 0u32..1000 {
            assert_eq!(
                map2.get(b"ns", &i.to_le_bytes()),
                Some(&(i * 2).to_le_bytes().to_vec())
            );
        }
    }

    #[test]
    fn test_flat_map_clear_namespace() {
        let mut map = FlatStateMap::new();
        map.put(b"ns1", b"k1", b"v1");
        map.put(b"ns1", b"k2", b"v2");
        map.put(b"ns2", b"k3", b"v3");

        map.clear_namespace(b"ns1");
        assert_eq!(map.len, 1);
        assert_eq!(map.get(b"ns1", b"k1"), None);
        assert_eq!(map.get(b"ns2", b"k3"), Some(&b"v3".to_vec()));
    }

    // -- HeapStateBackend tests (via trait) --

    #[test]
    fn test_heap_backend_crud() {
        let backend = HeapStateBackend::new();
        backend.put(b"test_ns", b"key1", b"value1").unwrap();
        assert_eq!(
            backend.get(b"test_ns", b"key1").unwrap(),
            Some(b"value1".to_vec())
        );
        assert_eq!(backend.entry_count(), 1);

        backend.put(b"test_ns", b"key1", b"value2").unwrap();
        assert_eq!(
            backend.get(b"test_ns", b"key1").unwrap(),
            Some(b"value2".to_vec())
        );
        assert_eq!(backend.entry_count(), 1);

        backend.delete(b"test_ns", b"key1").unwrap();
        assert_eq!(backend.get(b"test_ns", b"key1").unwrap(), None);
        assert_eq!(backend.entry_count(), 0);
    }

    #[test]
    fn test_heap_backend_prefix_scan() {
        let backend = HeapStateBackend::new();
        backend.put(b"ns", b"user:1", b"alice").unwrap();
        backend.put(b"ns", b"user:2", b"bob").unwrap();
        backend.put(b"ns", b"order:1", b"item").unwrap();

        let results = backend.prefix_scan(b"ns", b"user:").unwrap();
        assert_eq!(results.len(), 2);
    }

    #[test]
    fn test_heap_backend_snapshot_restore() {
        let backend = HeapStateBackend::new();
        backend.put(b"ns", b"k1", b"v1").unwrap();
        backend.put(b"ns", b"k2", b"v2").unwrap();

        let snapshot = backend.snapshot().unwrap();
        assert_eq!(snapshot.entry_count(), 2);

        backend.clear_namespace(b"ns").unwrap();
        assert_eq!(backend.entry_count(), 0);

        backend.restore(&snapshot).unwrap();
        assert_eq!(backend.entry_count(), 2);
        assert_eq!(backend.get(b"ns", b"k1").unwrap(), Some(b"v1".to_vec()));
    }

    #[test]
    fn test_heap_backend_clear_namespace() {
        let backend = HeapStateBackend::new();
        backend.put(b"ns1", b"k1", b"v1").unwrap();
        backend.put(b"ns2", b"k2", b"v2").unwrap();

        backend.clear_namespace(b"ns1").unwrap();
        assert_eq!(backend.get(b"ns1", b"k1").unwrap(), None);
        assert_eq!(backend.get(b"ns2", b"k2").unwrap(), Some(b"v2".to_vec()));
    }

    #[test]
    fn test_disk_backend_crud() {
        let dir = tempfile::tempdir().unwrap();
        let backend = DiskStateBackend::new(dir.path(), 1024 * 1024).unwrap();

        backend.put(b"ns", b"key", b"val").unwrap();
        assert_eq!(backend.get(b"ns", b"key").unwrap(), Some(b"val".to_vec()));

        backend.delete(b"ns", b"key").unwrap();
        assert_eq!(backend.get(b"ns", b"key").unwrap(), None);
    }

    #[test]
    fn test_disk_backend_snapshot_restore() {
        let dir = tempfile::tempdir().unwrap();
        let backend = DiskStateBackend::new(dir.path(), 1024 * 1024).unwrap();

        backend.put(b"ns", b"k1", b"v1").unwrap();
        backend.put(b"ns", b"k2", b"v2").unwrap();

        let snapshot = backend.snapshot().unwrap();
        backend.clear_namespace(b"ns").unwrap();
        assert_eq!(backend.entry_count(), 0);

        backend.restore(&snapshot).unwrap();
        assert_eq!(backend.entry_count(), 2);
    }

    // A value flushed to a segment must stay readable, a delete of it must
    // stick through the tombstone, and compaction must not lose either
    #[test]
    fn test_disk_backend_reads_survive_flush_and_compaction() {
        let dir = tempfile::tempdir().unwrap();
        // Tiny buffer cap so every few puts trigger a flush
        let backend = DiskStateBackend::new(dir.path(), 32).unwrap();

        for i in 0..20u32 {
            let key = format!("k{i:02}");
            let val = format!("v{i:02}");
            backend.put(b"ns", key.as_bytes(), val.as_bytes()).unwrap();
        }
        // Every key readable although the buffer flushed many times over
        for i in 0..20u32 {
            let key = format!("k{i:02}");
            let val = format!("v{i:02}");
            assert_eq!(
                backend.get(b"ns", key.as_bytes()).unwrap(),
                Some(val.into_bytes()),
                "flushed key {key} must stay readable"
            );
        }

        // Overwrite one flushed key and delete another, both must stick
        backend.put(b"ns", b"k03", b"updated").unwrap();
        backend.delete(b"ns", b"k05").unwrap();
        // Force more flushes so the tombstone and update land in segments
        for i in 20..40u32 {
            let key = format!("k{i:02}");
            backend.put(b"ns", key.as_bytes(), b"pad").unwrap();
        }
        assert_eq!(
            backend.get(b"ns", b"k03").unwrap(),
            Some(b"updated".to_vec())
        );
        assert_eq!(backend.get(b"ns", b"k05").unwrap(), None);

        let scanned = backend.prefix_scan(b"ns", b"k0").unwrap();
        let keys: Vec<String> = scanned
            .iter()
            .map(|(k, _)| String::from_utf8_lossy(k).into_owned())
            .collect();
        assert!(keys.contains(&"k03".to_string()));
        assert!(!keys.contains(&"k05".to_string()));

        assert_eq!(backend.entry_count(), 39, "40 keys minus one deleted");
    }

    #[test]
    fn test_queryable_state() {
        let backend = Arc::new(HeapStateBackend::new());
        backend.put(b"balances", b"user1", b"100").unwrap();
        backend.put(b"balances", b"user2", b"200").unwrap();

        let qs = QueryableState::new(backend, b"balances".to_vec(), "user-balances".into());
        assert_eq!(qs.get(b"user1").unwrap(), Some(b"100".to_vec()));
        assert_eq!(qs.name(), "user-balances");
    }

    #[test]
    fn test_state_snapshot_size() {
        let snap = StateSnapshot {
            data: vec![
                (b"ns".to_vec(), b"k1".to_vec(), b"v1".to_vec()),
                (b"ns".to_vec(), b"k2".to_vec(), b"v2".to_vec()),
            ],
            snapshot_id: 1,
        };
        assert_eq!(snap.entry_count(), 2);
        assert_eq!(snap.size_bytes(), 12);
    }

    #[test]
    fn test_flat_map_high_volume() {
        let mut map = FlatStateMap::new();
        let count = 10_000u32;
        for i in 0..count {
            map.put(b"ns", &i.to_le_bytes(), &[0xAB; 100]);
        }
        assert_eq!(map.len, count as usize);

        let snapshot = map.to_snapshot(1);
        assert_eq!(snapshot.entry_count(), count as usize);

        map.clear();
        assert_eq!(map.len, 0);

        map.from_snapshot(&snapshot);
        assert_eq!(map.len, count as usize);

        // Verify random sample.
        for i in (0..count).step_by(100) {
            assert_eq!(map.get(b"ns", &i.to_le_bytes()), Some(&vec![0xAB; 100]));
        }
    }
}
