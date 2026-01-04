//! Heap page implementation using slotted page format.
//!
//! Page layout:
//! ```text
//! +------------------+
//! | Page Header (32) |
//! +------------------+
//! | Slot Array       |  <- Grows downward
//! | (4 bytes/slot)   |
//! +------------------+
//! |                  |
//! | Free Space       |
//! |                  |
//! +------------------+
//! | Tuple Data       |  <- Grows upward
//! +------------------+
//! ```

use crate::tuple::Tuple;
use zyron_common::page::{PageHeader, PageId, PageType, PAGE_SIZE};
use zyron_common::{Result, ZyronError};

/// Slot identifier within a page.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct SlotId(pub u16);

impl SlotId {
    /// Invalid slot ID.
    pub const INVALID: SlotId = SlotId(u16::MAX);

    /// Returns true if this is a valid slot ID.
    pub fn is_valid(&self) -> bool {
        self.0 != u16::MAX
    }
}

impl std::fmt::Display for SlotId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "slot:{}", self.0)
    }
}

/// A slot in the slot array pointing to tuple data.
///
/// Layout (4 bytes):
/// - offset: 2 bytes (offset from page start to tuple data)
/// - length: 2 bytes (length of tuple data including header)
#[derive(Debug, Clone, Copy, Default)]
pub struct TupleSlot {
    /// Offset from page start to tuple data.
    pub offset: u16,
    /// Length of tuple data (0 = deleted/empty slot).
    pub length: u16,
}

impl TupleSlot {
    /// Size of a slot entry in bytes.
    pub const SIZE: usize = 4;

    /// Creates a new slot.
    pub fn new(offset: u16, length: u16) -> Self {
        Self { offset, length }
    }

    /// Returns true if this slot is empty/deleted.
    pub fn is_empty(&self) -> bool {
        self.length == 0
    }

    /// Serializes the slot to bytes.
    pub fn to_bytes(&self) -> [u8; Self::SIZE] {
        let mut buf = [0u8; Self::SIZE];
        buf[0..2].copy_from_slice(&self.offset.to_le_bytes());
        buf[2..4].copy_from_slice(&self.length.to_le_bytes());
        buf
    }

    /// Deserializes a slot from bytes.
    pub fn from_bytes(buf: &[u8]) -> Self {
        Self {
            offset: u16::from_le_bytes([buf[0], buf[1]]),
            length: u16::from_le_bytes([buf[2], buf[3]]),
        }
    }
}

/// Heap page header extension.
///
/// Stored after the standard PageHeader.
/// Layout (8 bytes):
/// - slot_count: 2 bytes
/// - free_space_start: 2 bytes (end of slot array)
/// - free_space_end: 2 bytes (start of tuple data)
/// - reserved: 2 bytes
#[derive(Debug, Clone, Copy)]
pub struct HeapPageHeader {
    /// Number of slots in the slot array.
    pub slot_count: u16,
    /// Offset where free space starts (after slot array).
    pub free_space_start: u16,
    /// Offset where free space ends (before tuple data).
    pub free_space_end: u16,
    /// Reserved for future use.
    pub reserved: u16,
}

impl HeapPageHeader {
    /// Size of the heap page header in bytes.
    pub const SIZE: usize = 8;

    /// Offset of heap header in page (after PageHeader).
    pub const OFFSET: usize = PageHeader::SIZE;

    /// Creates a new heap page header.
    pub fn new() -> Self {
        Self {
            slot_count: 0,
            free_space_start: (PageHeader::SIZE + Self::SIZE) as u16,
            free_space_end: PAGE_SIZE as u16,
            reserved: 0,
        }
    }

    /// Returns the amount of free space available.
    pub fn free_space(&self) -> usize {
        if self.free_space_end > self.free_space_start {
            (self.free_space_end - self.free_space_start) as usize
        } else {
            0
        }
    }

    /// Serializes to bytes.
    pub fn to_bytes(&self) -> [u8; Self::SIZE] {
        let mut buf = [0u8; Self::SIZE];
        buf[0..2].copy_from_slice(&self.slot_count.to_le_bytes());
        buf[2..4].copy_from_slice(&self.free_space_start.to_le_bytes());
        buf[4..6].copy_from_slice(&self.free_space_end.to_le_bytes());
        buf[6..8].copy_from_slice(&self.reserved.to_le_bytes());
        buf
    }

    /// Deserializes from bytes.
    pub fn from_bytes(buf: &[u8]) -> Self {
        Self {
            slot_count: u16::from_le_bytes([buf[0], buf[1]]),
            free_space_start: u16::from_le_bytes([buf[2], buf[3]]),
            free_space_end: u16::from_le_bytes([buf[4], buf[5]]),
            reserved: u16::from_le_bytes([buf[6], buf[7]]),
        }
    }
}

impl Default for HeapPageHeader {
    fn default() -> Self {
        Self::new()
    }
}

/// A heap page for storing variable-length tuples.
pub struct HeapPage {
    /// Page data buffer.
    data: Box<[u8; PAGE_SIZE]>,
}

impl HeapPage {
    /// Minimum data offset after headers.
    const DATA_START: usize = PageHeader::SIZE + HeapPageHeader::SIZE;

    /// Creates a new empty heap page.
    pub fn new(page_id: PageId) -> Self {
        let mut data = Box::new([0u8; PAGE_SIZE]);

        // Initialize page header
        let page_header = PageHeader::new(page_id, PageType::Heap);
        data[..PageHeader::SIZE].copy_from_slice(&page_header.to_bytes());

        // Initialize heap header
        let heap_header = HeapPageHeader::new();
        let offset = HeapPageHeader::OFFSET;
        data[offset..offset + HeapPageHeader::SIZE].copy_from_slice(&heap_header.to_bytes());

        Self { data }
    }

    /// Creates a heap page from raw page data.
    pub fn from_bytes(data: [u8; PAGE_SIZE]) -> Self {
        Self {
            data: Box::new(data),
        }
    }

    /// Returns the raw page data.
    pub fn as_bytes(&self) -> &[u8; PAGE_SIZE] {
        &self.data
    }

    /// Returns the heap header.
    fn heap_header(&self) -> HeapPageHeader {
        let offset = HeapPageHeader::OFFSET;
        HeapPageHeader::from_bytes(&self.data[offset..offset + HeapPageHeader::SIZE])
    }

    /// Writes the heap header back to the page.
    fn set_heap_header(&mut self, header: HeapPageHeader) {
        let offset = HeapPageHeader::OFFSET;
        self.data[offset..offset + HeapPageHeader::SIZE].copy_from_slice(&header.to_bytes());
    }

    /// Returns the number of slots in the page.
    pub fn slot_count(&self) -> u16 {
        self.heap_header().slot_count
    }

    /// Returns the amount of free space available.
    pub fn free_space(&self) -> usize {
        self.heap_header().free_space()
    }

    /// Returns the offset of a slot in the slot array.
    fn slot_offset(slot_id: SlotId) -> usize {
        Self::DATA_START + (slot_id.0 as usize) * TupleSlot::SIZE
    }

    /// Reads a slot from the slot array.
    pub fn get_slot(&self, slot_id: SlotId) -> Option<TupleSlot> {
        let header = self.heap_header();
        if slot_id.0 >= header.slot_count {
            return None;
        }

        let offset = Self::slot_offset(slot_id);
        Some(TupleSlot::from_bytes(&self.data[offset..offset + TupleSlot::SIZE]))
    }

    /// Writes a slot to the slot array.
    fn set_slot(&mut self, slot_id: SlotId, slot: TupleSlot) {
        let offset = Self::slot_offset(slot_id);
        self.data[offset..offset + TupleSlot::SIZE].copy_from_slice(&slot.to_bytes());
    }

    /// Inserts a tuple into the page.
    ///
    /// Returns the slot ID where the tuple was inserted.
    pub fn insert_tuple(&mut self, tuple: &Tuple) -> Result<SlotId> {
        let tuple_size = tuple.size_on_disk();

        let mut header = self.heap_header();

        // Check for a deleted slot we can reuse
        let mut reuse_slot: Option<SlotId> = None;
        for i in 0..header.slot_count {
            if let Some(slot) = self.get_slot(SlotId(i)) {
                if slot.is_empty() {
                    reuse_slot = Some(SlotId(i));
                    break;
                }
            }
        }

        // Calculate required space
        let need_new_slot = reuse_slot.is_none();
        let space_needed = if need_new_slot {
            tuple_size + TupleSlot::SIZE
        } else {
            tuple_size
        };

        if header.free_space() < space_needed {
            return Err(ZyronError::PageFull);
        }

        // Allocate tuple space (grows upward from end)
        header.free_space_end -= tuple_size as u16;
        let tuple_offset = header.free_space_end;

        // Write tuple data
        let tuple_bytes = tuple.serialize();
        self.data[tuple_offset as usize..tuple_offset as usize + tuple_size]
            .copy_from_slice(&tuple_bytes);

        // Get or create slot
        let slot_id = if let Some(sid) = reuse_slot {
            sid
        } else {
            let sid = SlotId(header.slot_count);
            header.slot_count += 1;
            header.free_space_start += TupleSlot::SIZE as u16;
            sid
        };

        // Write slot
        let slot = TupleSlot::new(tuple_offset, tuple_size as u16);
        self.set_slot(slot_id, slot);

        // Update header
        self.set_heap_header(header);

        Ok(slot_id)
    }

    /// Reads a tuple from the page.
    pub fn get_tuple(&self, slot_id: SlotId) -> Option<Tuple> {
        let slot = self.get_slot(slot_id)?;

        if slot.is_empty() {
            return None;
        }

        let start = slot.offset as usize;
        let end = start + slot.length as usize;

        Tuple::deserialize(&self.data[start..end])
    }

    /// Deletes a tuple from the page.
    ///
    /// This marks the slot as empty but doesn't reclaim space immediately.
    pub fn delete_tuple(&mut self, slot_id: SlotId) -> bool {
        if let Some(slot) = self.get_slot(slot_id) {
            if slot.is_empty() {
                return false;
            }

            // Mark slot as empty
            let empty_slot = TupleSlot::new(slot.offset, 0);
            self.set_slot(slot_id, empty_slot);
            return true;
        }
        false
    }

    /// Updates a tuple in place if it fits, otherwise returns error.
    pub fn update_tuple(&mut self, slot_id: SlotId, tuple: &Tuple) -> Result<()> {
        let old_slot = self.get_slot(slot_id).ok_or_else(|| {
            ZyronError::TupleNotFound(format!("slot {} not found", slot_id))
        })?;

        if old_slot.is_empty() {
            return Err(ZyronError::TupleNotFound(format!("slot {} is empty", slot_id)));
        }

        let new_size = tuple.size_on_disk();
        let old_size = old_slot.length as usize;

        // Only allow in-place update if new tuple fits in old space
        if new_size > old_size {
            return Err(ZyronError::PageFull);
        }

        // Write new tuple data
        let tuple_bytes = tuple.serialize();
        let start = old_slot.offset as usize;
        self.data[start..start + new_size].copy_from_slice(&tuple_bytes);

        // Update slot length (offset stays the same)
        let new_slot = TupleSlot::new(old_slot.offset, new_size as u16);
        self.set_slot(slot_id, new_slot);

        Ok(())
    }

    /// Iterates over all valid tuples in the page.
    pub fn iter(&self) -> HeapPageIterator<'_> {
        HeapPageIterator {
            page: self,
            current_slot: 0,
            slot_count: self.slot_count(),
        }
    }

    /// Returns true if the page can fit a tuple of the given size.
    pub fn can_fit(&self, tuple_size: usize) -> bool {
        let header = self.heap_header();

        // Check if we can reuse a deleted slot
        for i in 0..header.slot_count {
            if let Some(slot) = self.get_slot(SlotId(i)) {
                if slot.is_empty() {
                    // Just need space for tuple data
                    return header.free_space() >= tuple_size;
                }
            }
        }

        // Need space for both slot and tuple
        header.free_space() >= tuple_size + TupleSlot::SIZE
    }
}

/// Iterator over tuples in a heap page.
pub struct HeapPageIterator<'a> {
    page: &'a HeapPage,
    current_slot: u16,
    slot_count: u16,
}

impl<'a> Iterator for HeapPageIterator<'a> {
    type Item = (SlotId, Tuple);

    fn next(&mut self) -> Option<Self::Item> {
        while self.current_slot < self.slot_count {
            let slot_id = SlotId(self.current_slot);
            self.current_slot += 1;

            if let Some(tuple) = self.page.get_tuple(slot_id) {
                return Some((slot_id, tuple));
            }
        }
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tuple::TupleHeader;
    use bytes::Bytes;

    fn create_test_page() -> HeapPage {
        HeapPage::new(PageId::new(0, 0))
    }

    #[test]
    fn test_slot_id() {
        let slot = SlotId(5);
        assert!(slot.is_valid());
        assert_eq!(slot.to_string(), "slot:5");

        assert!(!SlotId::INVALID.is_valid());
    }

    #[test]
    fn test_tuple_slot_roundtrip() {
        let slot = TupleSlot::new(100, 50);
        let bytes = slot.to_bytes();
        let recovered = TupleSlot::from_bytes(&bytes);

        assert_eq!(recovered.offset, 100);
        assert_eq!(recovered.length, 50);
    }

    #[test]
    fn test_tuple_slot_empty() {
        let empty = TupleSlot::new(100, 0);
        assert!(empty.is_empty());

        let valid = TupleSlot::new(100, 50);
        assert!(!valid.is_empty());
    }

    #[test]
    fn test_heap_page_new() {
        let page = create_test_page();

        assert_eq!(page.slot_count(), 0);
        assert!(page.free_space() > 0);
    }

    #[test]
    fn test_heap_page_insert_tuple() {
        let mut page = create_test_page();
        let data = Bytes::from_static(b"hello world");
        let tuple = Tuple::new(data, 1);

        let slot_id = page.insert_tuple(&tuple).unwrap();
        assert_eq!(slot_id.0, 0);
        assert_eq!(page.slot_count(), 1);
    }

    #[test]
    fn test_heap_page_get_tuple() {
        let mut page = create_test_page();
        let data = Bytes::from_static(b"test data");
        let tuple = Tuple::new(data.clone(), 42);

        let slot_id = page.insert_tuple(&tuple).unwrap();
        let retrieved = page.get_tuple(slot_id).unwrap();

        assert_eq!(retrieved.data(), &data);
        assert_eq!(retrieved.header().xmin, 42);
    }

    #[test]
    fn test_heap_page_multiple_tuples() {
        let mut page = create_test_page();

        for i in 0..10 {
            let data = Bytes::from(format!("tuple {}", i));
            let tuple = Tuple::new(data, i);
            page.insert_tuple(&tuple).unwrap();
        }

        assert_eq!(page.slot_count(), 10);

        for i in 0..10 {
            let tuple = page.get_tuple(SlotId(i)).unwrap();
            assert_eq!(tuple.header().xmin, i as u32);
        }
    }

    #[test]
    fn test_heap_page_delete_tuple() {
        let mut page = create_test_page();
        let data = Bytes::from_static(b"to be deleted");
        let tuple = Tuple::new(data, 1);

        let slot_id = page.insert_tuple(&tuple).unwrap();
        assert!(page.get_tuple(slot_id).is_some());

        assert!(page.delete_tuple(slot_id));
        assert!(page.get_tuple(slot_id).is_none());
    }

    #[test]
    fn test_heap_page_reuse_slot() {
        let mut page = create_test_page();

        // Insert and delete
        let data1 = Bytes::from_static(b"first");
        let tuple1 = Tuple::new(data1, 1);
        let slot1 = page.insert_tuple(&tuple1).unwrap();
        page.delete_tuple(slot1);

        // Insert again - should reuse slot
        let data2 = Bytes::from_static(b"second");
        let tuple2 = Tuple::new(data2.clone(), 2);
        let slot2 = page.insert_tuple(&tuple2).unwrap();

        assert_eq!(slot1, slot2); // Same slot reused
        assert_eq!(page.slot_count(), 1); // No new slots added

        let retrieved = page.get_tuple(slot2).unwrap();
        assert_eq!(retrieved.data(), &data2);
    }

    #[test]
    fn test_heap_page_update_tuple() {
        let mut page = create_test_page();

        // Insert a tuple
        let data1 = Bytes::from(vec![0u8; 100]);
        let tuple1 = Tuple::new(data1, 1);
        let slot_id = page.insert_tuple(&tuple1).unwrap();

        // Update with smaller tuple (should succeed)
        let data2 = Bytes::from(vec![1u8; 50]);
        let tuple2 = Tuple::new(data2.clone(), 2);
        page.update_tuple(slot_id, &tuple2).unwrap();

        let retrieved = page.get_tuple(slot_id).unwrap();
        assert_eq!(retrieved.header().xmin, 2);
    }

    #[test]
    fn test_heap_page_update_too_large() {
        let mut page = create_test_page();

        // Insert a small tuple
        let data1 = Bytes::from(vec![0u8; 10]);
        let tuple1 = Tuple::new(data1, 1);
        let slot_id = page.insert_tuple(&tuple1).unwrap();

        // Try to update with larger tuple (should fail)
        let data2 = Bytes::from(vec![1u8; 100]);
        let tuple2 = Tuple::new(data2, 2);
        let result = page.update_tuple(slot_id, &tuple2);

        assert!(matches!(result, Err(ZyronError::PageFull)));
    }

    #[test]
    fn test_heap_page_iterator() {
        let mut page = create_test_page();

        for i in 0..5 {
            let data = Bytes::from(format!("tuple {}", i));
            let tuple = Tuple::new(data, i);
            page.insert_tuple(&tuple).unwrap();
        }

        // Delete one tuple
        page.delete_tuple(SlotId(2));

        // Iterator should skip deleted tuple
        let tuples: Vec<_> = page.iter().collect();
        assert_eq!(tuples.len(), 4);

        // Check that slot 2 was skipped
        let slot_ids: Vec<_> = tuples.iter().map(|(id, _)| id.0).collect();
        assert!(!slot_ids.contains(&2));
    }

    #[test]
    fn test_heap_page_can_fit() {
        let mut page = create_test_page();

        // Should be able to fit a small tuple
        assert!(page.can_fit(100));

        // Fill the page with large tuples
        while page.can_fit(1000) {
            let data = Bytes::from(vec![0u8; 1000 - TupleHeader::SIZE]);
            let tuple = Tuple::new(data, 1);
            page.insert_tuple(&tuple).unwrap();
        }

        // Should not be able to fit another large tuple
        assert!(!page.can_fit(1000));
    }

    #[test]
    fn test_heap_page_page_full() {
        let mut page = create_test_page();

        // Try to insert a tuple larger than page
        let huge_data = Bytes::from(vec![0u8; PAGE_SIZE]);
        let huge_tuple = Tuple::new(huge_data, 1);
        let result = page.insert_tuple(&huge_tuple);

        assert!(matches!(result, Err(ZyronError::PageFull)));
    }

    #[test]
    fn test_heap_page_from_bytes() {
        let mut page = create_test_page();
        let data = Bytes::from_static(b"persistent data");
        let tuple = Tuple::new(data.clone(), 999);
        let slot_id = page.insert_tuple(&tuple).unwrap();

        // Get raw bytes
        let raw_bytes = *page.as_bytes();

        // Reconstruct from bytes
        let recovered_page = HeapPage::from_bytes(raw_bytes);
        let recovered_tuple = recovered_page.get_tuple(slot_id).unwrap();

        assert_eq!(recovered_tuple.data(), &data);
        assert_eq!(recovered_tuple.header().xmin, 999);
    }

    #[test]
    fn test_heap_page_get_nonexistent_slot() {
        let page = create_test_page();

        assert!(page.get_slot(SlotId(0)).is_none());
        assert!(page.get_tuple(SlotId(0)).is_none());
    }

    #[test]
    fn test_heap_page_delete_nonexistent() {
        let mut page = create_test_page();
        assert!(!page.delete_tuple(SlotId(0)));
    }

    #[test]
    fn test_heap_page_delete_already_deleted() {
        let mut page = create_test_page();
        let data = Bytes::from_static(b"data");
        let tuple = Tuple::new(data, 1);
        let slot_id = page.insert_tuple(&tuple).unwrap();

        assert!(page.delete_tuple(slot_id));
        assert!(!page.delete_tuple(slot_id)); // Already deleted
    }
}
