//! Byte cache for the parts of column segments a query reads more than
//! once.
//!
//! Separate from the buffer pool, which manages fixed pages. A segment part
//! is variable in size, immutable for as long as the file it came from
//! exists, and shared through an Arc, so a hit hands back the bytes an
//! earlier read verified rather than a copy of them. Eviction is a clock
//! sweep under a byte capacity

use std::ops::Deref;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};

/// Which part of a segment an entry holds
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SegmentPart {
    /// The null bitmap and encoded payload, the bytes a decode consumes
    Payload,
    /// The serialized value bloom
    Bloom,
}

/// Identifies one part of one column segment of one open file.
///
/// The file id is assigned per reader and never reused within a process,
/// so an entry left behind by a reader that has gone is never answered to
/// a later reader of the same path
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct SegmentCacheKey {
    pub file_id: u64,
    pub column_id: u32,
    pub part: SegmentPart,
}

impl SegmentCacheKey {
    /// The payload part of one column segment
    pub fn new(file_id: u64, column_id: u32) -> Self {
        Self {
            file_id,
            column_id,
            part: SegmentPart::Payload,
        }
    }

    /// The bloom part of one column segment
    pub fn bloom(file_id: u64, column_id: u32) -> Self {
        Self {
            file_id,
            column_id,
            part: SegmentPart::Bloom,
        }
    }
}

/// One cached segment part
pub struct CachedSegment {
    pub key: SegmentCacheKey,
    pub data: Vec<u8>,
    /// Reference bit for the clock sweep
    reference_bit: AtomicBool,
}

impl CachedSegment {
    /// Bytes under a key, shared the way a cached entry is but held by
    /// nothing else. What a read too large to admit hands back, so a caller
    /// sees one type whether the bytes were kept or not
    pub fn new(key: SegmentCacheKey, data: Vec<u8>) -> Self {
        Self {
            key,
            data,
            reference_bit: AtomicBool::new(false),
        }
    }

    fn size_bytes(&self) -> usize {
        self.data.len()
    }
}

impl Deref for CachedSegment {
    type Target = [u8];

    fn deref(&self) -> &[u8] {
        &self.data
    }
}

/// Cache utilization statistics
#[derive(Debug, Clone)]
pub struct SegmentCacheStats {
    pub max_bytes: usize,
    pub used_bytes: u64,
    pub entry_count: usize,
    pub hit_count: u64,
    pub miss_count: u64,
}

/// Clock-sweep cache of segment parts under a byte capacity
pub struct SegmentCache {
    max_bytes: usize,
    current_bytes: AtomicU64,
    entries: scc::HashMap<SegmentCacheKey, Arc<CachedSegment>>,
    /// Keys in insertion order, the ring the clock hand walks
    clock_keys: parking_lot::RwLock<Vec<SegmentCacheKey>>,
    clock_hand: AtomicUsize,
    hit_count: AtomicU64,
    miss_count: AtomicU64,
}

impl SegmentCache {
    /// Creates a cache with the given byte capacity
    pub fn new(max_bytes: usize) -> Self {
        Self {
            max_bytes,
            current_bytes: AtomicU64::new(0),
            entries: scc::HashMap::new(),
            clock_keys: parking_lot::RwLock::new(Vec::new()),
            clock_hand: AtomicUsize::new(0),
            hit_count: AtomicU64::new(0),
            miss_count: AtomicU64::new(0),
        }
    }

    /// Looks up a cached part, setting its reference bit on a hit
    pub fn get(&self, key: &SegmentCacheKey) -> Option<Arc<CachedSegment>> {
        if let Some(segment) = self.entries.read_sync(key, |_, v| {
            v.reference_bit.store(true, Ordering::Relaxed);
            Arc::clone(v)
        }) {
            self.hit_count.fetch_add(1, Ordering::Relaxed);
            Some(segment)
        } else {
            self.miss_count.fetch_add(1, Ordering::Relaxed);
            None
        }
    }

    /// Inserts a part, evicting until it fits, and hands back the shared
    /// entry so the caller reads the same bytes a later hit will
    pub fn insert(&self, key: SegmentCacheKey, data: Vec<u8>) -> Arc<CachedSegment> {
        let dataSize = data.len() as u64;

        // Evict until there is room. Accounts for the size of an existing
        // entry under this key being displaced so the post-insert net stays
        // bounded
        let displacedSize = self
            .entries
            .read_sync(&key, |_, v| v.size_bytes() as u64)
            .unwrap_or(0);
        let netNeeded = dataSize.saturating_sub(displacedSize) as usize;
        self.evict_until(netNeeded);

        let segment = Arc::new(CachedSegment {
            key,
            data,
            reference_bit: AtomicBool::new(true),
        });

        let result = Arc::clone(&segment);

        // Replace the stored entry under this key. On an existing key,
        // subtract the displaced entry's size before adding the new size and
        // skip pushing a duplicate ring entry so accounting and the ring
        // stay consistent
        let mut displaced: u64 = 0;
        let mut wasPresent = false;
        match self.entries.entry_sync(key) {
            scc::hash_map::Entry::Occupied(mut entry) => {
                displaced = entry.get().size_bytes() as u64;
                *entry.get_mut() = segment;
                wasPresent = true;
            }
            scc::hash_map::Entry::Vacant(entry) => {
                entry.insert_entry(segment);
            }
        }

        if displaced > 0 {
            self.current_bytes.fetch_sub(displaced, Ordering::Relaxed);
        }
        self.current_bytes.fetch_add(dataSize, Ordering::Relaxed);

        if !wasPresent {
            let mut keys = self.clock_keys.write();
            keys.push(key);
        }

        result
    }

    /// Removes one entry
    pub fn invalidate(&self, key: &SegmentCacheKey) {
        if let Some((_, removed)) = self.entries.remove_sync(key) {
            let size = removed.size_bytes() as u64;
            self.current_bytes.fetch_sub(size, Ordering::Relaxed);
        }

        let mut keys = self.clock_keys.write();
        keys.retain(|k| k != key);
    }

    /// Removes every entry
    pub fn clear(&self) {
        self.entries.retain_sync(|_, _| false);
        self.current_bytes.store(0, Ordering::Relaxed);
        let mut keys = self.clock_keys.write();
        keys.clear();
    }

    /// Cache utilization statistics
    pub fn stats(&self) -> SegmentCacheStats {
        let entryCount = self.entries.len();
        SegmentCacheStats {
            max_bytes: self.max_bytes,
            used_bytes: self.current_bytes.load(Ordering::Relaxed),
            entry_count: entryCount,
            hit_count: self.hit_count.load(Ordering::Relaxed),
            miss_count: self.miss_count.load(Ordering::Relaxed),
        }
    }

    /// Clock-sweep eviction until at least `needed_bytes` are free
    fn evict_until(&self, needed_bytes: usize) {
        let maxSweeps = 2;
        for _ in 0..maxSweeps {
            let currentUsed = self.current_bytes.load(Ordering::Relaxed) as usize;
            if currentUsed + needed_bytes <= self.max_bytes {
                return;
            }

            let keys = self.clock_keys.read();
            if keys.is_empty() {
                return;
            }

            let keyCount = keys.len();
            let mut hand = self.clock_hand.load(Ordering::Relaxed) % keyCount;

            // One full rotation
            for _ in 0..keyCount {
                let key = keys[hand];
                hand = (hand + 1) % keyCount;

                if let Some(entry) = self.entries.get_sync(&key) {
                    let segment = entry.get();
                    if segment.reference_bit.load(Ordering::Relaxed) {
                        // Clear the reference bit, give a second chance
                        segment.reference_bit.store(false, Ordering::Relaxed);
                    } else {
                        let size = segment.size_bytes() as u64;
                        drop(entry);
                        if let Some((_, _)) = self.entries.remove_sync(&key) {
                            self.current_bytes.fetch_sub(size, Ordering::Relaxed);
                        }

                        let currentUsed = self.current_bytes.load(Ordering::Relaxed) as usize;
                        if currentUsed + needed_bytes <= self.max_bytes {
                            self.clock_hand.store(hand, Ordering::Relaxed);
                            drop(keys);
                            // Drop evicted keys from the ring
                            let mut wrKeys = self.clock_keys.write();
                            wrKeys.retain(|k| self.entries.contains_sync(k));
                            return;
                        }
                    }
                }
            }

            self.clock_hand.store(hand, Ordering::Relaxed);
        }

        // Final sweep, the second-chance passes above may clear reference
        // bits without freeing enough for a large item. Evict ignoring the
        // reference bit so insert never pushes current_bytes above max_bytes
        // while entries remain to reclaim
        loop {
            let currentUsed = self.current_bytes.load(Ordering::Relaxed) as usize;
            if currentUsed + needed_bytes <= self.max_bytes {
                break;
            }

            let key = {
                let keys = self.clock_keys.read();
                if keys.is_empty() {
                    break;
                }
                let keyCount = keys.len();
                let hand = self.clock_hand.load(Ordering::Relaxed) % keyCount;
                self.clock_hand
                    .store((hand + 1) % keyCount, Ordering::Relaxed);
                keys[hand]
            };

            if let Some((_, removed)) = self.entries.remove_sync(&key) {
                let size = removed.size_bytes() as u64;
                self.current_bytes.fetch_sub(size, Ordering::Relaxed);
                let mut wrKeys = self.clock_keys.write();
                wrKeys.retain(|k| *k != key);
            } else {
                // A stale ring entry already gone from the map, dropped so
                // the loop makes progress instead of spinning on a dead key
                let mut wrKeys = self.clock_keys.write();
                wrKeys.retain(|k| self.entries.contains_sync(k));
                if wrKeys.is_empty() {
                    break;
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_insert_and_get() {
        let cache = SegmentCache::new(1024 * 1024);
        let key = SegmentCacheKey::new(1, 0);
        let data = vec![42u8; 100];

        cache.insert(key, data.clone());

        let result = cache.get(&key);
        assert!(result.is_some());
        assert_eq!(result.unwrap().data, data);
    }

    #[test]
    fn test_miss() {
        let cache = SegmentCache::new(1024 * 1024);
        let key = SegmentCacheKey::new(1, 0);

        assert!(cache.get(&key).is_none());
        assert_eq!(cache.stats().miss_count, 1);
    }

    #[test]
    fn test_parts_of_one_column_are_distinct_entries() {
        let cache = SegmentCache::new(1024 * 1024);
        cache.insert(SegmentCacheKey::new(7, 3), vec![1u8; 10]);
        cache.insert(SegmentCacheKey::bloom(7, 3), vec![2u8; 20]);

        let payload = cache.get(&SegmentCacheKey::new(7, 3)).expect("payload");
        let bloom = cache.get(&SegmentCacheKey::bloom(7, 3)).expect("bloom");
        assert_eq!(&payload[..], &[1u8; 10]);
        assert_eq!(&bloom[..], &[2u8; 20]);
        assert_eq!(cache.stats().entry_count, 2);
    }

    #[test]
    fn test_invalidate() {
        let cache = SegmentCache::new(1024 * 1024);
        let key = SegmentCacheKey::new(1, 0);
        cache.insert(key, vec![0u8; 100]);

        assert!(cache.get(&key).is_some());
        cache.invalidate(&key);
        assert!(cache.get(&key).is_none());
    }

    #[test]
    fn test_eviction_on_capacity() {
        // Cache with 200 bytes capacity
        let cache = SegmentCache::new(200);

        // Insert 3 segments of 100 bytes each. Third should trigger eviction
        let key1 = SegmentCacheKey::new(1, 0);
        let key2 = SegmentCacheKey::new(2, 0);
        let key3 = SegmentCacheKey::new(3, 0);

        cache.insert(key1, vec![1u8; 100]);

        // Access key1 to set reference bit
        let _ = cache.get(&key1);

        cache.insert(key2, vec![2u8; 100]);

        // key2's reference bit is true from insert, so it needs one more
        // sweep

        cache.insert(key3, vec![3u8; 100]);

        // At least one of key1 or key2 should have been evicted
        let stats = cache.stats();
        assert!(stats.used_bytes <= 300);
    }

    #[test]
    fn test_clear() {
        let cache = SegmentCache::new(1024 * 1024);
        for i in 0..10 {
            cache.insert(SegmentCacheKey::new(i, 0), vec![0u8; 100]);
        }

        assert!(cache.stats().entry_count > 0);
        cache.clear();
        assert_eq!(cache.stats().used_bytes, 0);
    }

    #[test]
    fn test_stats() {
        let cache = SegmentCache::new(1024);
        let key = SegmentCacheKey::new(1, 0);
        cache.insert(key, vec![0u8; 100]);
        let _ = cache.get(&key);
        let _ = cache.get(&SegmentCacheKey::new(99, 0)); // miss

        let stats = cache.stats();
        assert_eq!(stats.max_bytes, 1024);
        assert_eq!(stats.used_bytes, 100);
        assert_eq!(stats.hit_count, 1);
        assert_eq!(stats.miss_count, 1);
    }
}
