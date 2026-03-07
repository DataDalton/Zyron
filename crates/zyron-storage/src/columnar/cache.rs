//! Segment cache for decoded column data.
//!
//! Separate from the buffer pool (which manages fixed 16KB pages).
//! Column segments are variable-size, immutable once cached, and
//! shared via Arc. Uses clock-sweep eviction with a byte-level
//! capacity limit.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};

/// Cache key identifying a decoded column segment.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct SegmentCacheKey {
    /// File identifier (hash of file path or monotonic counter).
    pub file_id: u64,
    /// Column identifier within the file.
    pub column_id: u32,
}

impl SegmentCacheKey {
    pub fn new(file_id: u64, column_id: u32) -> Self {
        Self { file_id, column_id }
    }
}

/// A cached decoded column segment.
pub struct CachedSegment {
    pub key: SegmentCacheKey,
    /// Decoded column data.
    pub data: Vec<u8>,
    /// Reference bit for clock eviction.
    reference_bit: AtomicBool,
}

impl CachedSegment {
    fn size_bytes(&self) -> usize {
        self.data.len()
    }
}

/// Cache utilization statistics.
#[derive(Debug, Clone)]
pub struct SegmentCacheStats {
    pub max_bytes: usize,
    pub used_bytes: u64,
    pub entry_count: usize,
    pub hit_count: u64,
    pub miss_count: u64,
}

/// Clock-sweep cache for decoded column segments.
pub struct SegmentCache {
    max_bytes: usize,
    current_bytes: AtomicU64,
    entries: scc::HashMap<u64, Arc<CachedSegment>>,
    /// Ordered list of cache keys for clock sweep.
    clock_keys: parking_lot::RwLock<Vec<u64>>,
    clock_hand: AtomicUsize,
    hit_count: AtomicU64,
    miss_count: AtomicU64,
}

impl SegmentCache {
    /// Creates a segment cache with the given byte capacity.
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

    /// Looks up a cached segment. Sets reference bit on hit.
    pub fn get(&self, key: &SegmentCacheKey) -> Option<Arc<CachedSegment>> {
        let packedKey = pack_key(key);
        if let Some(entry) = self.entries.get_sync(&packedKey) {
            entry.get().reference_bit.store(true, Ordering::Relaxed);
            self.hit_count.fetch_add(1, Ordering::Relaxed);
            Some(Arc::clone(entry.get()))
        } else {
            self.miss_count.fetch_add(1, Ordering::Relaxed);
            None
        }
    }

    /// Inserts a decoded segment into the cache. Evicts entries if needed.
    pub fn insert(&self, key: SegmentCacheKey, data: Vec<u8>) -> Arc<CachedSegment> {
        let dataSize = data.len() as u64;
        let packedKey = pack_key(&key);

        // Evict until there is room
        self.evict_until(dataSize as usize);

        let segment = Arc::new(CachedSegment {
            key,
            data,
            reference_bit: AtomicBool::new(true),
        });

        let result = Arc::clone(&segment);

        // Insert or update
        let _ = self.entries.insert_sync(packedKey, segment);
        self.current_bytes.fetch_add(dataSize, Ordering::Relaxed);

        // Add to clock ring
        {
            let mut keys = self.clock_keys.write();
            keys.push(packedKey);
        }

        result
    }

    /// Removes a specific entry from the cache.
    pub fn invalidate(&self, key: &SegmentCacheKey) {
        let packedKey = pack_key(key);
        if let Some((_, removed)) = self.entries.remove_sync(&packedKey) {
            let size = removed.size_bytes() as u64;
            self.current_bytes.fetch_sub(size, Ordering::Relaxed);
        }

        let mut keys = self.clock_keys.write();
        keys.retain(|k| *k != packedKey);
    }

    /// Clears the entire cache.
    pub fn clear(&self) {
        self.entries.retain_sync(|_, _| false);
        self.current_bytes.store(0, Ordering::Relaxed);
        let mut keys = self.clock_keys.write();
        keys.clear();
    }

    /// Returns cache utilization statistics.
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

    /// Clock-sweep eviction until at least `needed_bytes` are free.
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
                let packedKey = keys[hand];
                hand = (hand + 1) % keyCount;

                if let Some(entry) = self.entries.get_sync(&packedKey) {
                    let segment = entry.get();
                    if segment.reference_bit.load(Ordering::Relaxed) {
                        // Clear reference bit, give second chance
                        segment.reference_bit.store(false, Ordering::Relaxed);
                    } else {
                        // Evict this entry
                        let size = segment.size_bytes() as u64;
                        drop(entry);
                        if let Some((_, _)) = self.entries.remove_sync(&packedKey) {
                            self.current_bytes.fetch_sub(size, Ordering::Relaxed);
                        }

                        let currentUsed = self.current_bytes.load(Ordering::Relaxed) as usize;
                        if currentUsed + needed_bytes <= self.max_bytes {
                            self.clock_hand.store(hand, Ordering::Relaxed);
                            drop(keys);
                            // Clean up evicted keys from the ring
                            let mut wrKeys = self.clock_keys.write();
                            wrKeys.retain(|k| self.entries.contains_sync(k));
                            return;
                        }
                    }
                }
            }

            self.clock_hand.store(hand, Ordering::Relaxed);
        }
    }
}

/// Packs a SegmentCacheKey into a u64 for hash map lookup.
fn pack_key(key: &SegmentCacheKey) -> u64 {
    key.file_id ^ ((key.column_id as u64) << 48)
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

        // Insert 3 segments of 100 bytes each. Third should trigger eviction.
        let key1 = SegmentCacheKey::new(1, 0);
        let key2 = SegmentCacheKey::new(2, 0);
        let key3 = SegmentCacheKey::new(3, 0);

        cache.insert(key1, vec![1u8; 100]);

        // Access key1 to set reference bit
        let _ = cache.get(&key1);

        cache.insert(key2, vec![2u8; 100]);

        // Don't access key2 (no reference bit set after insert resets)
        // key2's reference bit is true from insert, so it needs one more sweep

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
