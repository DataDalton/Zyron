//! Lock-free page table for buffer pool page ID to frame ID mapping.

use crate::frame::FrameId;
use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};
use zyron_common::page::PageId;

/// Direct array size for file_id=0 pages. Covers first 16384 pages (~256MB at 16KB pages).
const DIRECT_PATH_SIZE: usize = 16384;

/// Sentinel value for empty slots in direct path.
const EMPTY_FRAME: u32 = u32::MAX;

/// Sentinel value for empty key slots in hash table.
const EMPTY_KEY: u64 = u64::MAX;

/// Sentinel value for deleted key slots (tombstone).
const TOMBSTONE_KEY: u64 = u64::MAX - 1;

/// Lock-free page table mapping PageId to FrameId.
///
/// Uses two-tier lookup:
/// - Direct array for file_id=0, page_num < 16384 (~2-3ns lookup)
/// - Open-addressing hash table with linear probing for overflow (~8-12ns lookup)
pub struct PageTable {
    /// Direct array for file_id=0 pages. Stores frame_id directly.
    direct_path: Box<[AtomicU32; DIRECT_PATH_SIZE]>,
    /// Hash table keys (full 64-bit PageId).
    hash_keys: Box<[AtomicU64]>,
    /// Hash table values (frame_id).
    hash_values: Box<[AtomicU32]>,
    /// Bitmask for hash table indexing (hash_size - 1).
    hash_mask: usize,
}

impl PageTable {
    /// Creates a new page table with capacity for the given number of frames.
    pub fn new(capacity: usize) -> Self {
        // Hash table size = next power of 2, 2x capacity for ~50% load factor
        let hash_size = (capacity * 2).next_power_of_two().max(1024);

        // Initialize direct path array with empty sentinel
        let direct_path: Box<[AtomicU32; DIRECT_PATH_SIZE]> = {
            let mut v = Vec::with_capacity(DIRECT_PATH_SIZE);
            for _ in 0..DIRECT_PATH_SIZE {
                v.push(AtomicU32::new(EMPTY_FRAME));
            }
            v.into_boxed_slice().try_into().unwrap()
        };

        // Initialize separate key and value arrays for hash table
        let hash_keys: Box<[AtomicU64]> = (0..hash_size)
            .map(|_| AtomicU64::new(EMPTY_KEY))
            .collect();

        let hash_values: Box<[AtomicU32]> = (0..hash_size)
            .map(|_| AtomicU32::new(EMPTY_FRAME))
            .collect();

        Self {
            direct_path,
            hash_keys,
            hash_values,
            hash_mask: hash_size - 1,
        }
    }

    /// Looks up a page ID and returns its frame ID if present.
    #[inline(always)]
    pub fn get(&self, page_id: PageId) -> Option<FrameId> {
        if page_id.file_id == 0 && page_id.page_num < DIRECT_PATH_SIZE as u32 {
            let val = self.direct_path[page_id.page_num as usize].load(Ordering::Acquire);
            if val != EMPTY_FRAME {
                return Some(FrameId(val));
            }
            return None;
        }
        self.get_from_hash(page_id)
    }

    #[inline]
    fn get_from_hash(&self, page_id: PageId) -> Option<FrameId> {
        let key = page_id.as_u64();
        let mut idx = self.hash_index(key);

        for _ in 0..self.hash_keys.len() {
            let stored_key = self.hash_keys[idx].load(Ordering::Acquire);
            if stored_key == EMPTY_KEY {
                return None;
            }
            if stored_key == key {
                let frame_id = self.hash_values[idx].load(Ordering::Acquire);
                return Some(FrameId(frame_id));
            }
            // Skip tombstones and continue probing
            idx = (idx + 1) & self.hash_mask;
        }
        None
    }

    /// Inserts a page ID to frame ID mapping. Returns true on success.
    pub fn insert(&self, page_id: PageId, frame_id: FrameId) -> bool {
        if page_id.file_id == 0 && page_id.page_num < DIRECT_PATH_SIZE as u32 {
            self.direct_path[page_id.page_num as usize].store(frame_id.0, Ordering::Release);
            return true;
        }
        self.insert_to_hash(page_id, frame_id)
    }

    fn insert_to_hash(&self, page_id: PageId, frame_id: FrameId) -> bool {
        let key = page_id.as_u64();
        let mut idx = self.hash_index(key);

        for _ in 0..self.hash_keys.len() {
            let stored_key = self.hash_keys[idx].load(Ordering::Acquire);
            if stored_key == EMPTY_KEY || stored_key == TOMBSTONE_KEY {
                // Empty or tombstone slot - insert here
                self.hash_values[idx].store(frame_id.0, Ordering::Release);
                self.hash_keys[idx].store(key, Ordering::Release);
                return true;
            }
            if stored_key == key {
                // Update existing entry
                self.hash_values[idx].store(frame_id.0, Ordering::Release);
                return true;
            }
            idx = (idx + 1) & self.hash_mask;
        }
        false // Table full
    }

    /// Removes a page ID mapping. Returns the frame ID if it was present.
    pub fn remove(&self, page_id: PageId) -> Option<FrameId> {
        if page_id.file_id == 0 && page_id.page_num < DIRECT_PATH_SIZE as u32 {
            let old = self.direct_path[page_id.page_num as usize].swap(EMPTY_FRAME, Ordering::AcqRel);
            if old != EMPTY_FRAME {
                return Some(FrameId(old));
            }
            return None;
        }
        self.remove_from_hash(page_id)
    }

    fn remove_from_hash(&self, page_id: PageId) -> Option<FrameId> {
        let key = page_id.as_u64();
        let mut idx = self.hash_index(key);

        for _ in 0..self.hash_keys.len() {
            let stored_key = self.hash_keys[idx].load(Ordering::Acquire);
            if stored_key == EMPTY_KEY {
                return None;
            }
            if stored_key == key {
                let frame_id = self.hash_values[idx].load(Ordering::Acquire);
                self.hash_keys[idx].store(TOMBSTONE_KEY, Ordering::Release);
                return Some(FrameId(frame_id));
            }
            idx = (idx + 1) & self.hash_mask;
        }
        None
    }

    /// Computes hash table index for a key.
    #[inline(always)]
    fn hash_index(&self, key: u64) -> usize {
        // FxHash-style multiply for distribution
        let hash = key.wrapping_mul(0x517cc1b727220a95);
        (hash as usize) & self.hash_mask
    }

    /// Returns true if the page ID is in the table.
    pub fn contains(&self, page_id: PageId) -> bool {
        self.get(page_id).is_some()
    }

    /// Returns the number of entries in the table.
    pub fn len(&self) -> usize {
        let mut count = 0;
        for slot in self.direct_path.iter() {
            let val = slot.load(Ordering::Relaxed);
            if val != EMPTY_FRAME {
                count += 1;
            }
        }
        for slot in self.hash_keys.iter() {
            let val = slot.load(Ordering::Relaxed);
            if val != EMPTY_KEY && val != TOMBSTONE_KEY {
                count += 1;
            }
        }
        count
    }

    /// Returns true if the table is empty.
    #[allow(dead_code)]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Iterates over all entries, calling the provided function for each.
    /// Returns early if the function returns false.
    pub fn for_each<F>(&self, mut f: F)
    where
        F: FnMut(PageId, FrameId) -> bool,
    {
        // Iterate direct path
        for (page_num, slot) in self.direct_path.iter().enumerate() {
            let val = slot.load(Ordering::Relaxed);
            if val != EMPTY_FRAME {
                let page_id = PageId::new(0, page_num as u32);
                if !f(page_id, FrameId(val)) {
                    return;
                }
            }
        }

        // Iterate hash table
        for (idx, key_slot) in self.hash_keys.iter().enumerate() {
            let key = key_slot.load(Ordering::Relaxed);
            if key != EMPTY_KEY && key != TOMBSTONE_KEY {
                let frame_id = self.hash_values[idx].load(Ordering::Relaxed);
                let page_id = PageId::from_u64(key);
                if !f(page_id, FrameId(frame_id)) {
                    return;
                }
            }
        }
    }
}


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_direct_path_insert_get() {
        let table = PageTable::new(100);
        let page_id = PageId::new(0, 42);
        let frame_id = FrameId(7);

        assert!(table.insert(page_id, frame_id));
        assert_eq!(table.get(page_id), Some(frame_id));
        assert!(table.contains(page_id));
    }

    #[test]
    fn test_direct_path_remove() {
        let table = PageTable::new(100);
        let page_id = PageId::new(0, 42);
        let frame_id = FrameId(7);

        table.insert(page_id, frame_id);
        assert_eq!(table.remove(page_id), Some(frame_id));
        assert_eq!(table.get(page_id), None);
        assert!(!table.contains(page_id));
    }

    #[test]
    fn test_hash_path_insert_get() {
        let table = PageTable::new(100);
        // Use file_id > 0 to go through hash path
        let page_id = PageId::new(1, 42);
        let frame_id = FrameId(7);

        assert!(table.insert(page_id, frame_id));
        assert_eq!(table.get(page_id), Some(frame_id));
    }

    #[test]
    fn test_hash_path_remove() {
        let table = PageTable::new(100);
        let page_id = PageId::new(1, 42);
        let frame_id = FrameId(7);

        table.insert(page_id, frame_id);
        assert_eq!(table.remove(page_id), Some(frame_id));
        assert_eq!(table.get(page_id), None);
    }

    #[test]
    fn test_high_page_num_uses_hash() {
        let table = PageTable::new(100);
        // page_num >= DIRECT_PATH_SIZE goes through hash
        let page_id = PageId::new(0, DIRECT_PATH_SIZE as u32 + 100);
        let frame_id = FrameId(5);

        assert!(table.insert(page_id, frame_id));
        assert_eq!(table.get(page_id), Some(frame_id));
    }

    #[test]
    fn test_len() {
        let table = PageTable::new(100);

        assert_eq!(table.len(), 0);
        assert!(table.is_empty());

        table.insert(PageId::new(0, 1), FrameId(1));
        table.insert(PageId::new(0, 2), FrameId(2));
        table.insert(PageId::new(1, 1), FrameId(3));

        assert_eq!(table.len(), 3);
        assert!(!table.is_empty());
    }

    #[test]
    fn test_update_existing() {
        let table = PageTable::new(100);
        let page_id = PageId::new(0, 42);

        table.insert(page_id, FrameId(1));
        assert_eq!(table.get(page_id), Some(FrameId(1)));

        table.insert(page_id, FrameId(2));
        assert_eq!(table.get(page_id), Some(FrameId(2)));
        assert_eq!(table.len(), 1);
    }
}
