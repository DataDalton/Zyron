//! B+Tree page implementations (leaf and internal nodes).

use bytes::Bytes;
use crate::tuple::TupleId;
use super::constants::MIN_FILL_FACTOR;
use super::types::{compare_keys_fast, DeleteResult, InternalEntry, InternalPageHeader, LeafEntry, LeafPageHeader};
use zyron_common::page::{PageHeader, PageId, PageType, PAGE_SIZE};
use zyron_common::{Result, ZyronError};

/// B+ tree leaf page (slotted page format).
pub struct BTreeLeafPage {
    /// Page data buffer.
    data: Box<[u8; PAGE_SIZE]>,
}

impl BTreeLeafPage {
    /// Slot array start offset after headers.
    const SLOT_ARRAY_START: usize = PageHeader::SIZE + LeafPageHeader::SIZE;

    /// Size of each slot (offset:2 + len:2).
    const SLOT_SIZE: usize = 4;

    /// Creates a new empty leaf page.
    pub fn new(page_id: PageId) -> Self {
        let mut data = Box::new([0u8; PAGE_SIZE]);

        // Initialize page header
        let page_header = PageHeader::new(page_id, PageType::BTreeLeaf);
        data[..PageHeader::SIZE].copy_from_slice(&page_header.to_bytes());

        // Initialize leaf header with data_end at PAGE_SIZE
        let leaf_header = LeafPageHeader::new();
        let offset = LeafPageHeader::OFFSET;
        data[offset..offset + LeafPageHeader::SIZE].copy_from_slice(&leaf_header.to_bytes());

        Self { data }
    }

    /// Creates a leaf page from raw bytes.
    pub fn from_bytes(data: [u8; PAGE_SIZE]) -> Self {
        Self {
            data: Box::new(data),
        }
    }

    /// Returns the raw page data.
    pub fn as_bytes(&self) -> &[u8; PAGE_SIZE] {
        &self.data
    }

    /// Returns the leaf header.
    fn leaf_header(&self) -> LeafPageHeader {
        let offset = LeafPageHeader::OFFSET;
        LeafPageHeader::from_bytes(&self.data[offset..offset + LeafPageHeader::SIZE])
    }

    /// Writes the leaf header.
    fn set_leaf_header(&mut self, header: LeafPageHeader) {
        let offset = LeafPageHeader::OFFSET;
        self.data[offset..offset + LeafPageHeader::SIZE].copy_from_slice(&header.to_bytes());
    }

    /// Returns the number of entries in this leaf.
    pub fn num_entries(&self) -> u16 {
        self.leaf_header().num_slots
    }

    /// Returns the amount of free space available.
    /// Free space = data_end - (SLOT_ARRAY_START + num_slots * SLOT_SIZE)
    pub fn free_space(&self) -> usize {
        let header = self.leaf_header();
        let slot_array_end = Self::SLOT_ARRAY_START + (header.num_slots as usize * Self::SLOT_SIZE);
        (header.data_end as usize).saturating_sub(slot_array_end)
    }

    /// Returns the next leaf page ID.
    pub fn next_leaf(&self) -> Option<PageId> {
        let next = self.leaf_header().next_leaf;
        if next == u64::MAX {
            None
        } else {
            Some(PageId::from_u64(next))
        }
    }

    /// Sets the next leaf page ID.
    pub fn set_next_leaf(&mut self, page_id: Option<PageId>) {
        let mut header = self.leaf_header();
        header.next_leaf = page_id.map(|p| p.as_u64()).unwrap_or(u64::MAX);
        self.set_leaf_header(header);
    }

    /// Reads all entries from the leaf (via slot array).
    pub fn entries(&self) -> Vec<LeafEntry> {
        let header = self.leaf_header();
        let num_slots = header.num_slots as usize;
        let mut entries = Vec::with_capacity(num_slots);

        for slot_idx in 0..num_slots {
            let slot_offset = Self::SLOT_ARRAY_START + slot_idx * Self::SLOT_SIZE;
            let entry_offset =
                u16::from_le_bytes([self.data[slot_offset], self.data[slot_offset + 1]]) as usize;

            if let Some((entry, _)) = LeafEntry::from_bytes(&self.data[entry_offset..]) {
                entries.push(entry);
            }
        }

        entries
    }

    /// Binary search for a key. Returns Ok(index) if found, Err(index) for insertion point.
    pub fn search(&self, key: &[u8]) -> std::result::Result<usize, usize> {
        let entries = self.entries();
        entries.binary_search_by(|e| compare_keys_fast(e.key.as_ref(), key))
    }

    /// Inserts a key-value pair into the leaf. Returns error if page is full.
    /// Uses single-pass in-place insertion for efficiency.
    #[inline]
    pub fn insert(&mut self, key: Bytes, tuple_id: TupleId) -> Result<()> {
        Self::insert_in_slice(&mut *self.data, &key, tuple_id)
    }

    /// Writes entries to the page using slotted format.
    fn write_entries(&mut self, entries: &[LeafEntry]) -> Result<()> {
        let num_entries = entries.len();

        // Calculate total space needed
        let slot_space = num_entries * Self::SLOT_SIZE;
        let entry_space: usize = entries.iter().map(|e| e.size_on_disk()).sum();
        let slot_array_end = Self::SLOT_ARRAY_START + slot_space;

        if slot_array_end + entry_space > PAGE_SIZE {
            return Err(ZyronError::NodeFull);
        }

        // Write entries backward from end and slots forward from start
        let mut data_end = PAGE_SIZE;

        for (slot_idx, entry) in entries.iter().enumerate() {
            let bytes = entry.to_bytes();
            data_end -= bytes.len();
            self.data[data_end..data_end + bytes.len()].copy_from_slice(&bytes);

            // Write slot
            let slot_offset = Self::SLOT_ARRAY_START + slot_idx * Self::SLOT_SIZE;
            self.data[slot_offset..slot_offset + 2]
                .copy_from_slice(&(data_end as u16).to_le_bytes());
            self.data[slot_offset + 2..slot_offset + 4]
                .copy_from_slice(&(bytes.len() as u16).to_le_bytes());
        }

        // Update header
        let mut header = self.leaf_header();
        header.num_slots = num_entries as u16;
        header.data_end = data_end as u16;
        self.set_leaf_header(header);
        Ok(())
    }

    /// Gets the value for a key.
    pub fn get(&self, key: &[u8]) -> Option<TupleId> {
        Self::get_in_slice(&*self.data, key)
    }

    /// Gets the value for a key using slotted page format.
    /// Binary search directly on slot array for O(log n) lookup - no offset building needed.
    #[inline(always)]
    pub fn get_in_slice(data: &[u8], key: &[u8]) -> Option<TupleId> {
        // Parse header - read num_slots as single u16
        let header_offset = LeafPageHeader::OFFSET;
        let num_slots = u16::from_le_bytes([data[header_offset], data[header_offset + 1]]) as usize;

        if num_slots == 0 {
            return None;
        }

        // Binary search with packed slot reads and u64 prefix comparison
        let mut low = 0usize;
        let mut high = num_slots;

        while low < high {
            let mid = low + (high - low) / 2;
            let slot_off = Self::SLOT_ARRAY_START + mid * Self::SLOT_SIZE;

            // Read slot as packed u32 (offset:u16 + len:u16)
            let packed = u32::from_le_bytes([
                data[slot_off],
                data[slot_off + 1],
                data[slot_off + 2],
                data[slot_off + 3],
            ]);
            let entry_off = (packed & 0xFFFF) as usize;

            // Read key_len from entry
            let key_len = u16::from_le_bytes([data[entry_off], data[entry_off + 1]]) as usize;
            let entry_key = &data[entry_off + 2..entry_off + 2 + key_len];

            match compare_keys_fast(key, entry_key) {
                std::cmp::Ordering::Equal => {
                    let tuple_offset = entry_off + 2 + key_len;
                    // Read page_id as u64 directly
                    let page_id = PageId::from_u64(u64::from_le_bytes([
                        data[tuple_offset],
                        data[tuple_offset + 1],
                        data[tuple_offset + 2],
                        data[tuple_offset + 3],
                        data[tuple_offset + 4],
                        data[tuple_offset + 5],
                        data[tuple_offset + 6],
                        data[tuple_offset + 7],
                    ]));
                    let slot_id =
                        u16::from_le_bytes([data[tuple_offset + 8], data[tuple_offset + 9]]);
                    return Some(TupleId::new(page_id, slot_id));
                }
                std::cmp::Ordering::Less => high = mid,
                std::cmp::Ordering::Greater => low = mid + 1,
            }
        }
        None
    }

    /// Inserts a key-value pair using slotted page format.
    /// Binary search for O(log n) lookup, only shift 4-byte slots instead of full entries.
    /// Returns Ok(()) on success, Err(NodeFull) if page is full, Err(DuplicateKey) if key exists.
    #[inline(always)]
    pub fn insert_in_slice(data: &mut [u8], key: &[u8], tuple_id: TupleId) -> Result<()> {
        // Parse header
        let header_offset = LeafPageHeader::OFFSET;
        let num_slots = u16::from_le_bytes([data[header_offset], data[header_offset + 1]]) as usize;
        let raw_data_end =
            u16::from_le_bytes([data[header_offset + 2], data[header_offset + 3]]) as usize;

        // Handle uninitialized pages (data_end == 0 means page was never written)
        let data_end = if raw_data_end == 0 || raw_data_end > PAGE_SIZE {
            PAGE_SIZE
        } else {
            raw_data_end
        };

        // Entry size: key_len(2) + key + page_id(8) + slot_id(2)
        let entry_size = 2 + key.len() + 10;

        // Calculate free space: between slot array end and data start
        let slot_array_end = Self::SLOT_ARRAY_START + num_slots * Self::SLOT_SIZE;
        let free_space = data_end.saturating_sub(slot_array_end);

        // Need space for both entry data and new slot
        if free_space < entry_size + Self::SLOT_SIZE {
            return Err(ZyronError::NodeFull);
        }

        // Binary search through slot array to find insertion point
        let mut low = 0usize;
        let mut high = num_slots;

        while low < high {
            let mid = low + (high - low) / 2;
            let slot_off = Self::SLOT_ARRAY_START + mid * Self::SLOT_SIZE;

            // Packed slot read: offset:u16 + len:u16 as single u32
            let packed = u32::from_le_bytes([
                data[slot_off],
                data[slot_off + 1],
                data[slot_off + 2],
                data[slot_off + 3],
            ]);
            let entry_off = (packed & 0xFFFF) as usize;

            let key_len = u16::from_le_bytes([data[entry_off], data[entry_off + 1]]) as usize;
            let entry_key = &data[entry_off + 2..entry_off + 2 + key_len];

            match compare_keys_fast(key, entry_key) {
                std::cmp::Ordering::Equal => return Err(ZyronError::DuplicateKey),
                std::cmp::Ordering::Less => high = mid,
                std::cmp::Ordering::Greater => low = mid + 1,
            }
        }

        let insert_slot_idx = low;

        // Write entry data at the end (grows backward)
        let new_data_end = data_end - entry_size;
        let mut write_offset = new_data_end;
        data[write_offset..write_offset + 2].copy_from_slice(&(key.len() as u16).to_le_bytes());
        write_offset += 2;
        data[write_offset..write_offset + key.len()].copy_from_slice(key);
        write_offset += key.len();
        data[write_offset..write_offset + 8]
            .copy_from_slice(&tuple_id.page_id.as_u64().to_le_bytes());
        write_offset += 8;
        data[write_offset..write_offset + 2].copy_from_slice(&tuple_id.slot_id.to_le_bytes());

        // Shift slots forward to make room for new slot (only 4 bytes per slot)
        let insert_slot_offset = Self::SLOT_ARRAY_START + insert_slot_idx * Self::SLOT_SIZE;
        let slots_to_shift = num_slots - insert_slot_idx;
        if slots_to_shift > 0 {
            let shift_start = insert_slot_offset;
            let shift_end = shift_start + slots_to_shift * Self::SLOT_SIZE;
            data.copy_within(shift_start..shift_end, shift_start + Self::SLOT_SIZE);
        }

        // Write new slot (offset:2 + len:2)
        data[insert_slot_offset..insert_slot_offset + 2]
            .copy_from_slice(&(new_data_end as u16).to_le_bytes());
        data[insert_slot_offset + 2..insert_slot_offset + 4]
            .copy_from_slice(&(entry_size as u16).to_le_bytes());

        // Update header
        let new_num_slots = (num_slots + 1) as u16;
        data[header_offset..header_offset + 2].copy_from_slice(&new_num_slots.to_le_bytes());
        data[header_offset + 2..header_offset + 4]
            .copy_from_slice(&(new_data_end as u16).to_le_bytes());

        Ok(())
    }

    /// Deletes a key from the leaf. Returns DeleteResult indicating outcome.
    pub fn delete(&mut self, key: &[u8]) -> DeleteResult {
        match self.search(key) {
            Ok(idx) => {
                let mut entries = self.entries();
                entries.remove(idx);
                if self.write_entries(&entries).is_ok() {
                    if self.is_underfull() {
                        DeleteResult::Underfull
                    } else {
                        DeleteResult::Ok
                    }
                } else {
                    DeleteResult::NotFound
                }
            }
            Err(_) => DeleteResult::NotFound,
        }
    }

    /// Returns true if this leaf is underfull (below MIN_FILL_FACTOR capacity).
    ///
    /// An underfull node should trigger rebalancing (borrowing from siblings
    /// or merging with a sibling) to maintain B+ tree balance invariants.
    pub fn is_underfull(&self) -> bool {
        let header = self.leaf_header();
        let entry_data_space = PAGE_SIZE - header.data_end as usize;
        let slot_space = header.num_slots as usize * Self::SLOT_SIZE;
        let used_space = entry_data_space + slot_space;
        let total_data_space = PAGE_SIZE - Self::SLOT_ARRAY_START;
        let fill_ratio = used_space as f64 / total_data_space as f64;
        fill_ratio < MIN_FILL_FACTOR && self.num_entries() > 0
    }

    /// Returns the minimum number of bytes that should be used to avoid underflow.
    pub fn min_used_space(&self) -> usize {
        let total_data_space = PAGE_SIZE - Self::SLOT_ARRAY_START;
        (total_data_space as f64 * MIN_FILL_FACTOR) as usize
    }

    /// Borrows entries from a right sibling to fix underflow.
    ///
    /// Returns the new separator key that should replace the old separator
    /// in the parent node, or None if borrowing is not possible.
    pub fn borrow_from_right(&mut self, right_sibling: &mut BTreeLeafPage) -> Option<Bytes> {
        if right_sibling.num_entries() <= 1 {
            return None; // Can't borrow if sibling would become empty
        }

        let mut right_entries = right_sibling.entries();
        let borrowed = right_entries.remove(0);

        let mut my_entries = self.entries();
        my_entries.push(borrowed);

        // Write updated entries
        if self.write_entries(&my_entries).is_err() {
            return None;
        }
        if right_sibling.write_entries(&right_entries).is_err() {
            return None;
        }

        // New separator is the first key of the right sibling after borrowing
        right_entries.first().map(|e| e.key.clone())
    }

    /// Borrows entries from a left sibling to fix underflow.
    ///
    /// Returns the new separator key that should replace the old separator
    /// in the parent node, or None if borrowing is not possible.
    pub fn borrow_from_left(&mut self, left_sibling: &mut BTreeLeafPage) -> Option<Bytes> {
        if left_sibling.num_entries() <= 1 {
            return None; // Can't borrow if sibling would become empty
        }

        let mut left_entries = left_sibling.entries();
        let borrowed = left_entries.pop()?;

        let mut my_entries = self.entries();
        my_entries.insert(0, borrowed);

        // Write updated entries
        if self.write_entries(&my_entries).is_err() {
            return None;
        }
        if left_sibling.write_entries(&left_entries).is_err() {
            return None;
        }

        // New separator is the first key of this node after borrowing
        my_entries.first().map(|e| e.key.clone())
    }

    /// Merges this leaf with its right sibling.
    ///
    /// All entries from right_sibling are moved into this leaf.
    /// The right sibling becomes empty and should be deallocated.
    /// Returns true if merge succeeded.
    pub fn merge_with_right(&mut self, right_sibling: &mut BTreeLeafPage) -> bool {
        let mut my_entries = self.entries();
        let right_entries = right_sibling.entries();

        my_entries.extend(right_entries);

        // Update next_leaf pointer to skip the merged sibling
        let new_next = right_sibling.next_leaf();
        self.set_next_leaf(new_next);

        self.write_entries(&my_entries).is_ok()
    }

    /// Returns true if this leaf can fit another entry of the given size.
    pub fn can_fit(&self, entry_size: usize) -> bool {
        self.free_space() >= entry_size
    }

    /// Splits this leaf into two. Returns (split_key, new_right_page).
    pub fn split(&mut self, new_page_id: PageId) -> (Bytes, BTreeLeafPage) {
        let entries = self.entries();
        let mid = entries.len() / 2;

        let left_entries: Vec<_> = entries[..mid].to_vec();
        let right_entries: Vec<_> = entries[mid..].to_vec();

        // The split key is the first key of the right page
        let split_key = right_entries[0].key.clone();

        // Rewrite left page
        let _ = self.write_entries(&left_entries);

        // Create right page
        let mut right_page = BTreeLeafPage::new(new_page_id);
        let _ = right_page.write_entries(&right_entries);

        // Link pages
        let old_next = self.next_leaf();
        self.set_next_leaf(Some(new_page_id));
        right_page.set_next_leaf(old_next);

        (split_key, right_page)
    }
}

/// B+ tree internal page.
pub struct BTreeInternalPage {
    /// Page data buffer.
    data: Box<[u8; PAGE_SIZE]>,
}

impl BTreeInternalPage {
    /// Data start offset after headers.
    const DATA_START: usize = PageHeader::SIZE + InternalPageHeader::SIZE;

    /// Size of the leftmost child pointer.
    const LEFTMOST_PTR_SIZE: usize = 8;

    /// Creates a new empty internal page.
    pub fn new(page_id: PageId, level: u16) -> Self {
        let mut data = Box::new([0u8; PAGE_SIZE]);

        // Initialize page header
        let page_header = PageHeader::new(page_id, PageType::BTreeInternal);
        data[..PageHeader::SIZE].copy_from_slice(&page_header.to_bytes());

        // Initialize internal header
        let internal_header = InternalPageHeader::new(level);
        let offset = InternalPageHeader::OFFSET;
        data[offset..offset + InternalPageHeader::SIZE]
            .copy_from_slice(&internal_header.to_bytes());

        Self { data }
    }

    /// Creates an internal page from raw bytes.
    pub fn from_bytes(data: [u8; PAGE_SIZE]) -> Self {
        Self {
            data: Box::new(data),
        }
    }

    /// Returns the raw page data.
    pub fn as_bytes(&self) -> &[u8; PAGE_SIZE] {
        &self.data
    }

    /// Returns the internal header.
    fn internal_header(&self) -> InternalPageHeader {
        let offset = InternalPageHeader::OFFSET;
        InternalPageHeader::from_bytes(&self.data[offset..offset + InternalPageHeader::SIZE])
    }

    /// Writes the internal header.
    fn set_internal_header(&mut self, header: InternalPageHeader) {
        let offset = InternalPageHeader::OFFSET;
        self.data[offset..offset + InternalPageHeader::SIZE].copy_from_slice(&header.to_bytes());
    }

    /// Returns the number of keys in this internal node.
    pub fn num_keys(&self) -> u16 {
        self.internal_header().num_keys
    }

    /// Returns the level of this internal node.
    pub fn level(&self) -> u16 {
        self.internal_header().level
    }

    /// Returns the amount of free space available.
    pub fn free_space(&self) -> usize {
        PAGE_SIZE - self.internal_header().free_space_offset as usize
    }

    /// Gets the leftmost child pointer.
    pub fn leftmost_child(&self) -> PageId {
        let offset = Self::DATA_START;
        let bytes = &self.data[offset..offset + 8];
        PageId::from_u64(u64::from_le_bytes([
            bytes[0], bytes[1], bytes[2], bytes[3], bytes[4], bytes[5], bytes[6], bytes[7],
        ]))
    }

    /// Sets the leftmost child pointer.
    pub fn set_leftmost_child(&mut self, page_id: PageId) {
        let offset = Self::DATA_START;
        self.data[offset..offset + 8].copy_from_slice(&page_id.as_u64().to_le_bytes());

        // Update header if this is the first entry
        let mut header = self.internal_header();
        if header.free_space_offset == Self::DATA_START as u16 {
            header.free_space_offset = (Self::DATA_START + Self::LEFTMOST_PTR_SIZE) as u16;
            self.set_internal_header(header);
        }
    }

    /// Reads all entries from the internal node.
    pub fn entries(&self) -> Vec<InternalEntry> {
        let header = self.internal_header();
        let mut entries = Vec::with_capacity(header.num_keys as usize);
        let mut offset = Self::DATA_START + Self::LEFTMOST_PTR_SIZE;

        for _ in 0..header.num_keys {
            if let Some((entry, consumed)) = InternalEntry::from_bytes(&self.data[offset..]) {
                entries.push(entry);
                offset += consumed;
            } else {
                break;
            }
        }

        entries
    }

    /// Finds the child page for a given key.
    pub fn find_child(&self, key: &[u8]) -> PageId {
        Self::find_child_in_slice(&*self.data, key)
    }

    /// Finds the child page for a given key directly from raw page data.
    /// Uses linear search for small pages (common case), binary search for large ones.
    #[inline(always)]
    pub fn find_child_in_slice(data: &[u8], key: &[u8]) -> PageId {
        // Parse header to get num_keys
        let header_offset = InternalPageHeader::OFFSET;
        let num_keys = u16::from_le_bytes([data[header_offset], data[header_offset + 1]]) as usize;

        // Leftmost child pointer
        let leftmost_offset = Self::DATA_START;
        let leftmost = PageId::from_u64(u64::from_le_bytes([
            data[leftmost_offset],
            data[leftmost_offset + 1],
            data[leftmost_offset + 2],
            data[leftmost_offset + 3],
            data[leftmost_offset + 4],
            data[leftmost_offset + 5],
            data[leftmost_offset + 6],
            data[leftmost_offset + 7],
        ]));

        if num_keys == 0 {
            return leftmost;
        }

        // For internal nodes with <= 8 entries, linear search is faster
        // (avoids offset array allocation)
        if num_keys <= 8 {
            let mut offset = Self::DATA_START + Self::LEFTMOST_PTR_SIZE;
            let mut last_child = leftmost;

            for _ in 0..num_keys {
                let key_len = u16::from_le_bytes([data[offset], data[offset + 1]]) as usize;
                let entry_key = &data[offset + 2..offset + 2 + key_len];

                if compare_keys_fast(key, entry_key).is_lt() {
                    return last_child;
                }

                // Update last_child to this entry's child pointer
                let child_offset = offset + 2 + key_len;
                last_child = PageId::from_u64(u64::from_le_bytes([
                    data[child_offset],
                    data[child_offset + 1],
                    data[child_offset + 2],
                    data[child_offset + 3],
                    data[child_offset + 4],
                    data[child_offset + 5],
                    data[child_offset + 6],
                    data[child_offset + 7],
                ]));

                offset += 2 + key_len + 8;
            }

            return last_child;
        }

        // For larger pages, use binary search with offset indexing
        let mut offsets = [0usize; 1024];
        let limit = num_keys.min(1024);
        let mut offset = Self::DATA_START + Self::LEFTMOST_PTR_SIZE;

        for i in 0..limit {
            offsets[i] = offset;
            let key_len = u16::from_le_bytes([data[offset], data[offset + 1]]) as usize;
            offset += 2 + key_len + 8;
        }

        let mut low = 0usize;
        let mut high = limit;

        while low < high {
            let mid = low + (high - low) / 2;
            let entry_offset = offsets[mid];
            let key_len = u16::from_le_bytes([data[entry_offset], data[entry_offset + 1]]) as usize;
            let entry_key = &data[entry_offset + 2..entry_offset + 2 + key_len];

            if compare_keys_fast(key, entry_key).is_lt() {
                high = mid;
            } else {
                low = mid + 1;
            }
        }

        if low == 0 {
            leftmost
        } else {
            let entry_offset = offsets[low - 1];
            let key_len = u16::from_le_bytes([data[entry_offset], data[entry_offset + 1]]) as usize;
            let child_offset = entry_offset + 2 + key_len;
            PageId::from_u64(u64::from_le_bytes([
                data[child_offset],
                data[child_offset + 1],
                data[child_offset + 2],
                data[child_offset + 3],
                data[child_offset + 4],
                data[child_offset + 5],
                data[child_offset + 6],
                data[child_offset + 7],
            ]))
        }
    }

    /// Inserts a key and right child pointer.
    /// Uses in-place insertion for efficiency.
    #[inline]
    pub fn insert(&mut self, key: Bytes, right_child: PageId) -> Result<()> {
        Self::insert_in_slice(&mut *self.data, key.as_ref(), right_child)
    }

    /// Inserts a key and child pointer directly into raw page data.
    /// Returns Ok(()) on success, Err(NodeFull) if page is full.
    #[inline(always)]
    pub fn insert_in_slice(data: &mut [u8], key: &[u8], right_child: PageId) -> Result<()> {
        // Parse header
        let header_offset = InternalPageHeader::OFFSET;
        let num_keys = u16::from_le_bytes([data[header_offset], data[header_offset + 1]]) as usize;
        let raw_free_offset =
            u16::from_le_bytes([data[header_offset + 2], data[header_offset + 3]]) as usize;

        // Handle uninitialized pages
        let free_space_offset = if raw_free_offset < Self::DATA_START + Self::LEFTMOST_PTR_SIZE {
            Self::DATA_START + Self::LEFTMOST_PTR_SIZE
        } else {
            raw_free_offset
        };

        // Entry size: key_len(2) + key + page_id(8)
        let entry_size = 2 + key.len() + 8;
        let free_space = PAGE_SIZE - free_space_offset;

        if free_space < entry_size {
            return Err(ZyronError::NodeFull);
        }

        // Find insertion point using linear scan (internal nodes have fewer entries)
        let mut insert_offset = Self::DATA_START + Self::LEFTMOST_PTR_SIZE;
        let mut offset = insert_offset;

        for _ in 0..num_keys {
            let key_len = u16::from_le_bytes([data[offset], data[offset + 1]]) as usize;
            let entry_key = &data[offset + 2..offset + 2 + key_len];
            let entry_total = 2 + key_len + 8;

            if compare_keys_fast(key, entry_key).is_lt() {
                insert_offset = offset;
                break;
            }
            offset += entry_total;
            insert_offset = offset;
        }

        // Shift existing entries to make room
        let bytes_to_shift = free_space_offset - insert_offset;
        if bytes_to_shift > 0 {
            data.copy_within(insert_offset..free_space_offset, insert_offset + entry_size);
        }

        // Write the new entry
        let mut write_offset = insert_offset;
        data[write_offset..write_offset + 2].copy_from_slice(&(key.len() as u16).to_le_bytes());
        write_offset += 2;
        data[write_offset..write_offset + key.len()].copy_from_slice(key);
        write_offset += key.len();
        data[write_offset..write_offset + 8].copy_from_slice(&right_child.as_u64().to_le_bytes());

        // Update header
        let new_num_keys = (num_keys + 1) as u16;
        let new_free_offset = (free_space_offset + entry_size) as u16;
        data[header_offset..header_offset + 2].copy_from_slice(&new_num_keys.to_le_bytes());
        data[header_offset + 2..header_offset + 4].copy_from_slice(&new_free_offset.to_le_bytes());

        Ok(())
    }

    /// Writes entries to the page.
    fn write_entries(&mut self, entries: &[InternalEntry]) -> Result<()> {
        let mut header = self.internal_header();
        let mut offset = Self::DATA_START + Self::LEFTMOST_PTR_SIZE;

        for entry in entries {
            let bytes = entry.to_bytes();
            if offset + bytes.len() > PAGE_SIZE {
                return Err(ZyronError::NodeFull);
            }
            self.data[offset..offset + bytes.len()].copy_from_slice(&bytes);
            offset += bytes.len();
        }

        header.num_keys = entries.len() as u16;
        header.free_space_offset = offset as u16;
        self.set_internal_header(header);
        Ok(())
    }

    /// Splits this internal node. Returns (promoted_key, new_right_page).
    pub fn split(&mut self, new_page_id: PageId) -> (Bytes, BTreeInternalPage) {
        let entries = self.entries();
        let mid = entries.len() / 2;

        let left_entries: Vec<_> = entries[..mid].to_vec();
        let promoted_key = entries[mid].key.clone();
        let right_first_child = entries[mid].child_page_id;
        let right_entries: Vec<_> = entries[mid + 1..].to_vec();

        // Rewrite left page
        let _ = self.write_entries(&left_entries);

        // Create right page
        let mut right_page = BTreeInternalPage::new(new_page_id, self.level());
        right_page.set_leftmost_child(right_first_child);
        let _ = right_page.write_entries(&right_entries);

        (promoted_key, right_page)
    }

    /// Returns true if this node can fit another entry of the given size.
    pub fn can_fit(&self, entry_size: usize) -> bool {
        self.free_space() >= entry_size
    }

    /// Returns true if this internal node is underfull (below MIN_FILL_FACTOR capacity).
    ///
    /// An underfull node should trigger rebalancing (borrowing from siblings
    /// or merging with a sibling) to maintain B+ tree balance invariants.
    pub fn is_underfull(&self) -> bool {
        let used_space = self.internal_header().free_space_offset as usize
            - Self::DATA_START
            - Self::LEFTMOST_PTR_SIZE;
        let total_data_space = PAGE_SIZE - Self::DATA_START - Self::LEFTMOST_PTR_SIZE;
        let fill_ratio = used_space as f64 / total_data_space as f64;
        fill_ratio < MIN_FILL_FACTOR && self.num_keys() > 0
    }

    /// Returns the minimum number of bytes that should be used to avoid underflow.
    pub fn min_used_space(&self) -> usize {
        let total_data_space = PAGE_SIZE - Self::DATA_START - Self::LEFTMOST_PTR_SIZE;
        (total_data_space as f64 * MIN_FILL_FACTOR) as usize
    }

    /// Deletes a key from the internal node. Returns DeleteResult indicating outcome.
    pub fn delete(&mut self, key: &[u8]) -> DeleteResult {
        let entries = self.entries();
        let pos = entries.iter().position(|e| e.key.as_ref() == key);

        match pos {
            Some(idx) => {
                let mut entries = entries;
                entries.remove(idx);
                if self.write_entries(&entries).is_ok() {
                    if self.is_underfull() {
                        DeleteResult::Underfull
                    } else {
                        DeleteResult::Ok
                    }
                } else {
                    DeleteResult::NotFound
                }
            }
            None => DeleteResult::NotFound,
        }
    }

    /// Borrows an entry from a right sibling to fix underflow.
    ///
    /// The separator_key is the key in the parent that separates this node from the sibling.
    /// Returns the new separator key that should replace the old one in the parent,
    /// or None if borrowing is not possible.
    pub fn borrow_from_right(
        &mut self,
        right_sibling: &mut BTreeInternalPage,
        separator_key: Bytes,
    ) -> Option<Bytes> {
        if right_sibling.num_keys() <= 1 {
            return None; // Can't borrow if sibling would become too empty
        }

        let mut right_entries = right_sibling.entries();
        let borrowed = right_entries.remove(0);

        // The separator comes down to become a key in this node
        let new_entry = InternalEntry {
            key: separator_key,
            child_page_id: right_sibling.leftmost_child(),
        };

        let mut my_entries = self.entries();
        my_entries.push(new_entry);

        // Update right sibling's leftmost child to the borrowed entry's child
        right_sibling.set_leftmost_child(borrowed.child_page_id);

        // Write updated entries
        if self.write_entries(&my_entries).is_err() {
            return None;
        }
        if right_sibling.write_entries(&right_entries).is_err() {
            return None;
        }

        // The borrowed key becomes the new separator in the parent
        Some(borrowed.key)
    }

    /// Borrows an entry from a left sibling to fix underflow.
    ///
    /// The separator_key is the key in the parent that separates this node from the sibling.
    /// Returns the new separator key that should replace the old one in the parent,
    /// or None if borrowing is not possible.
    pub fn borrow_from_left(
        &mut self,
        left_sibling: &mut BTreeInternalPage,
        separator_key: Bytes,
    ) -> Option<Bytes> {
        if left_sibling.num_keys() <= 1 {
            return None; // Can't borrow if sibling would become too empty
        }

        let mut left_entries = left_sibling.entries();
        let borrowed = left_entries.pop()?;

        // The separator comes down to become a key in this node
        let new_entry = InternalEntry {
            key: separator_key,
            child_page_id: self.leftmost_child(),
        };

        let mut my_entries = self.entries();
        my_entries.insert(0, new_entry);

        // Update this node's leftmost child to the borrowed entry's child
        self.set_leftmost_child(borrowed.child_page_id);

        // Write updated entries
        if self.write_entries(&my_entries).is_err() {
            return None;
        }
        if left_sibling.write_entries(&left_entries).is_err() {
            return None;
        }

        // The borrowed key becomes the new separator in the parent
        Some(borrowed.key)
    }

    /// Merges this internal node with its right sibling.
    ///
    /// The separator_key is the key from the parent that separates the two nodes.
    /// All entries from right_sibling are moved into this node.
    /// Returns true if merge succeeded.
    pub fn merge_with_right(
        &mut self,
        right_sibling: &BTreeInternalPage,
        separator_key: Bytes,
    ) -> bool {
        let mut my_entries = self.entries();

        // The separator key comes down with the right sibling's leftmost child
        let separator_entry = InternalEntry {
            key: separator_key,
            child_page_id: right_sibling.leftmost_child(),
        };
        my_entries.push(separator_entry);

        // Add all entries from the right sibling
        my_entries.extend(right_sibling.entries());

        self.write_entries(&my_entries).is_ok()
    }
}
