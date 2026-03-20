//! B+Tree page implementations (leaf and internal nodes).

use super::constants::MIN_FILL_FACTOR;
use super::types::{
    DeleteResult, InternalEntry, InternalEntryView, InternalPageHeader, LeafEntry, LeafEntryView,
    LeafPageHeader, compare_keys,
};
use crate::tuple::TupleId;
use bytes::Bytes;
use zyron_common::page::{PAGE_SIZE, PageHeader, PageId, PageType};
use zyron_common::{Result, ZyronError};

/// B+ tree leaf page (slotted page format).
pub struct BTreeLeafPage {
    /// Page data buffer.
    data: Box<[u8; PAGE_SIZE]>,
}

impl BTreeLeafPage {
    /// Slot array start offset after headers.
    pub(crate) const SLOT_ARRAY_START: usize = PageHeader::SIZE + LeafPageHeader::SIZE;

    /// Size of each slot (offset:2 + len:2).
    pub(crate) const SLOT_SIZE: usize = 4;

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

    /// Zero-copy read of all entries. Borrows keys from page buffer.
    pub fn entry_views(&self) -> Vec<LeafEntryView<'_>> {
        let header = self.leaf_header();
        let num_slots = header.num_slots as usize;
        let mut views = Vec::with_capacity(num_slots);

        for slot_idx in 0..num_slots {
            let slot_offset = Self::SLOT_ARRAY_START + slot_idx * Self::SLOT_SIZE;
            let entry_offset =
                u16::from_le_bytes([self.data[slot_offset], self.data[slot_offset + 1]]) as usize;

            if let Some((view, _)) = LeafEntryView::from_bytes(&self.data[entry_offset..]) {
                views.push(view);
            }
        }

        views
    }

    /// Binary search for a key. Returns Ok(index) if found, Err(index) for insertion point.
    pub fn search(&self, key: &[u8]) -> std::result::Result<usize, usize> {
        let views = self.entry_views();
        views.binary_search_by(|e| compare_keys(e.key, key))
    }

    /// Inserts a key-value pair into the leaf. Returns error if page is full.
    /// Uses single-pass in-place insertion for efficiency.
    #[inline]
    pub fn insert(&mut self, key: Bytes, tuple_id: TupleId) -> Result<()> {
        Self::insert_in_slice(&mut *self.data, &key, tuple_id)
    }

    /// Writes entries to the page using slotted format.
    /// Uses write_to_slice to avoid BytesMut allocation per entry.
    fn write_entries(&mut self, entries: &[LeafEntry]) -> Result<()> {
        let num_entries = entries.len();

        let slot_space = num_entries * Self::SLOT_SIZE;
        let entry_space: usize = entries.iter().map(|e| e.size_on_disk()).sum();
        let slot_array_end = Self::SLOT_ARRAY_START + slot_space;

        if slot_array_end + entry_space > PAGE_SIZE {
            return Err(ZyronError::NodeFull);
        }

        let mut data_end = PAGE_SIZE;

        for (slot_idx, entry) in entries.iter().enumerate() {
            let entry_size = entry.size_on_disk();
            data_end -= entry_size;
            entry.write_to_slice(&mut *self.data, data_end);

            let slot_offset = Self::SLOT_ARRAY_START + slot_idx * Self::SLOT_SIZE;
            self.data[slot_offset..slot_offset + 2]
                .copy_from_slice(&(data_end as u16).to_le_bytes());
            self.data[slot_offset + 2..slot_offset + 4]
                .copy_from_slice(&(entry_size as u16).to_le_bytes());
        }

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
    /// Returns TupleId with file_id=0. Caller sets file_id from index context.
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

            match compare_keys(key, entry_key) {
                std::cmp::Ordering::Equal => {
                    let tuple_offset = entry_off + 2 + key_len;
                    // Read page_num as u32, slot_id as u16
                    let page_num = u32::from_le_bytes([
                        data[tuple_offset],
                        data[tuple_offset + 1],
                        data[tuple_offset + 2],
                        data[tuple_offset + 3],
                    ]);
                    let slot_id =
                        u16::from_le_bytes([data[tuple_offset + 4], data[tuple_offset + 5]]);
                    let page_id = PageId::new(0, page_num as u64);
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
    /// Stores page_num (u32) + slot_id (u16) = 6 bytes per entry (file_id is implicit).
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

        // Entry size: key_len(2) + key + page_num(4) + slot_id(2)
        let entry_size = 2 + key.len() + 6;

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

            match compare_keys(key, entry_key) {
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
        data[write_offset..write_offset + 4]
            .copy_from_slice(&(tuple_id.page_id.page_num as u32).to_le_bytes());
        write_offset += 4;
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
    /// Uses entry_views to avoid Bytes allocation during search, then
    /// materializes remaining entries for the rewrite.
    pub fn delete(&mut self, key: &[u8]) -> DeleteResult {
        match self.search(key) {
            Ok(idx) => {
                let views = self.entry_views();
                let owned: Vec<LeafEntry> = views
                    .iter()
                    .enumerate()
                    .filter(|(i, _)| *i != idx)
                    .map(|(_, v)| v.to_owned())
                    .collect();
                drop(views);
                self.write_entries(&owned)
                    .expect("write_entries failed after delete, page data corrupted");
                if self.is_underfull() {
                    DeleteResult::Underfull
                } else {
                    DeleteResult::Ok
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
    /// Uses entry views for reads, write_to_slice for writes.
    pub fn borrow_from_right(&mut self, right_sibling: &mut BTreeLeafPage) -> Option<Bytes> {
        if right_sibling.num_entries() <= 1 {
            return None;
        }

        let right_views = right_sibling.entry_views();
        let borrowed = right_views[0].to_owned();
        let new_sep = Bytes::copy_from_slice(right_views[1].key);
        let new_right: Vec<LeafEntry> = right_views[1..].iter().map(|v| v.to_owned()).collect();
        drop(right_views);

        right_sibling.write_entries(&new_right).ok()?;

        let mut my_entries = self.entries();
        my_entries.push(borrowed);
        self.write_entries(&my_entries).ok()?;

        Some(new_sep)
    }

    /// Borrows entries from a left sibling to fix underflow.
    /// Uses entry views for reads, write_to_slice for writes.
    pub fn borrow_from_left(&mut self, left_sibling: &mut BTreeLeafPage) -> Option<Bytes> {
        if left_sibling.num_entries() <= 1 {
            return None;
        }

        let left_views = left_sibling.entry_views();
        let last_idx = left_views.len() - 1;
        let borrowed = left_views[last_idx].to_owned();
        let new_sep = Bytes::copy_from_slice(left_views[last_idx].key);
        let new_left: Vec<LeafEntry> = left_views[..last_idx]
            .iter()
            .map(|v| v.to_owned())
            .collect();
        drop(left_views);

        left_sibling.write_entries(&new_left).ok()?;

        let mut my_entries = self.entries();
        my_entries.insert(0, borrowed);
        self.write_entries(&my_entries).ok()?;

        Some(new_sep)
    }

    /// Merges this leaf with its right sibling.
    /// Right sibling is read via views, self uses owned entries since it grows.
    pub fn merge_with_right(&mut self, right_sibling: &mut BTreeLeafPage) -> bool {
        let mut my_entries = self.entries();
        let right_views = right_sibling.entry_views();
        my_entries.extend(right_views.iter().map(|v| v.to_owned()));

        let new_next = right_sibling.next_leaf();
        self.set_next_leaf(new_next);

        self.write_entries(&my_entries).is_ok()
    }

    /// Returns true if this leaf can fit another entry of the given size.
    pub fn can_fit(&self, entry_size: usize) -> bool {
        self.free_space() >= entry_size
    }

    /// Splits this leaf into two. Returns (split_key, new_right_page).
    /// Uses entry_views to avoid Bytes allocation during read, materializes
    /// both halves for the rewrite (required since write regions overlap reads).
    pub fn split(&mut self, new_page_id: PageId) -> (Bytes, BTreeLeafPage) {
        let views = self.entry_views();
        let mid = views.len() / 2;
        let split_key = Bytes::copy_from_slice(views[mid].key);

        let left_owned: Vec<LeafEntry> = views[..mid].iter().map(|v| v.to_owned()).collect();
        let right_owned: Vec<LeafEntry> = views[mid..].iter().map(|v| v.to_owned()).collect();
        drop(views);

        let _ = self.write_entries(&left_owned);

        let mut right_page = BTreeLeafPage::new(new_page_id);
        let _ = right_page.write_entries(&right_owned);

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

    /// Reads all entries from the internal node (allocates per key).
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

    /// Zero-copy read of all entries. Borrows keys from page buffer.
    pub fn entry_views(&self) -> Vec<InternalEntryView<'_>> {
        let header = self.internal_header();
        let mut views = Vec::with_capacity(header.num_keys as usize);
        let mut offset = Self::DATA_START + Self::LEFTMOST_PTR_SIZE;

        for _ in 0..header.num_keys {
            if let Some((view, consumed)) = InternalEntryView::from_bytes(&self.data[offset..]) {
                views.push(view);
                offset += consumed;
            } else {
                break;
            }
        }

        views
    }

    /// Reads entry views via raw pointer, bypassing borrow checker.
    /// Internal page entries are sequential, so writes to self.data at the same
    /// offsets would overlap with reads. Callers must write to a DIFFERENT buffer
    /// (right_page, sibling) or ensure the write region is beyond the read region.
    ///
    /// SAFETY: Returned views reference self.data via raw pointer with 'static lifetime.
    /// The caller must ensure self.data is not deallocated while views are in use.
    unsafe fn entry_views_raw(&self) -> Vec<InternalEntryView<'static>> {
        let header = self.internal_header();
        let mut views = Vec::with_capacity(header.num_keys as usize);
        let ptr = self.data.as_ptr();
        let mut offset = Self::DATA_START + Self::LEFTMOST_PTR_SIZE;

        for _ in 0..header.num_keys {
            let remaining = PAGE_SIZE - offset;
            let buf = unsafe { std::slice::from_raw_parts(ptr.add(offset), remaining) };
            if let Some((view, consumed)) = InternalEntryView::from_bytes(buf) {
                views.push(view);
                offset += consumed;
            } else {
                break;
            }
        }

        views
    }

    /// Finds the child page for a given key.
    pub fn find_child(&self, key: &[u8]) -> PageId {
        Self::find_child_in_slice(&*self.data, key)
    }

    /// Finds the child page for a given key directly from raw page data.
    /// Uses linear search for small pages (common case), binary search for large ones.
    /// Child pointers stored as page_num (u32). Returns PageId with file_id=0.
    #[inline(always)]
    pub fn find_child_in_slice(data: &[u8], key: &[u8]) -> PageId {
        // Parse header to get num_keys
        let header_offset = InternalPageHeader::OFFSET;
        let num_keys = u16::from_le_bytes([data[header_offset], data[header_offset + 1]]) as usize;

        // Leftmost child pointer (stored as u64 for page linking)
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
        if num_keys <= 8 {
            let mut offset = Self::DATA_START + Self::LEFTMOST_PTR_SIZE;
            let mut last_child = leftmost;

            for _ in 0..num_keys {
                let key_len = u16::from_le_bytes([data[offset], data[offset + 1]]) as usize;
                let entry_key = &data[offset + 2..offset + 2 + key_len];

                if compare_keys(key, entry_key).is_lt() {
                    return last_child;
                }

                // Read child page_num as u32
                let child_offset = offset + 2 + key_len;
                let page_num = u32::from_le_bytes([
                    data[child_offset],
                    data[child_offset + 1],
                    data[child_offset + 2],
                    data[child_offset + 3],
                ]);
                last_child = PageId::new(0, page_num as u64);

                offset += 2 + key_len + 4;
            }

            return last_child;
        }

        // For larger pages, use binary search with offset indexing.
        // Max entries per internal page: PAGE_SIZE / min_entry_size.
        // min_entry_size = key_len(2) + key(1) + page_num(4) = 7 bytes.
        // 16384 / 7 = 2340. Use 2048 (power of 2, fits comfortably).
        let mut offsets = [0usize; 2048];
        let limit = num_keys.min(2048);
        let mut offset = Self::DATA_START + Self::LEFTMOST_PTR_SIZE;

        for o in offsets.iter_mut().take(limit) {
            *o = offset;
            let key_len = u16::from_le_bytes([data[offset], data[offset + 1]]) as usize;
            offset += 2 + key_len + 4;
        }

        let mut low = 0usize;
        let mut high = limit;

        while low < high {
            let mid = low + (high - low) / 2;
            let entry_offset = offsets[mid];
            let key_len = u16::from_le_bytes([data[entry_offset], data[entry_offset + 1]]) as usize;
            let entry_key = &data[entry_offset + 2..entry_offset + 2 + key_len];

            if compare_keys(key, entry_key).is_lt() {
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
            let page_num = u32::from_le_bytes([
                data[child_offset],
                data[child_offset + 1],
                data[child_offset + 2],
                data[child_offset + 3],
            ]);
            PageId::new(0, page_num as u64)
        }
    }

    /// Inserts a key and right child pointer.
    /// Uses in-place insertion for efficiency.
    #[inline]
    pub fn insert(&mut self, key: Bytes, right_child: PageId) -> Result<()> {
        Self::insert_in_slice(&mut *self.data, key.as_ref(), right_child)
    }

    /// Inserts a key and child pointer directly into raw page data.
    /// Stores child page_num as u32 (4 bytes). file_id is implicit from index context.
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

        // Entry size: key_len(2) + key + page_num(4)
        let entry_size = 2 + key.len() + 4;
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
            let entry_total = 2 + key_len + 4;

            if compare_keys(key, entry_key).is_lt() {
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
        data[write_offset..write_offset + 4]
            .copy_from_slice(&(right_child.page_num as u32).to_le_bytes());

        // Update header
        let new_num_keys = (num_keys + 1) as u16;
        let new_free_offset = (free_space_offset + entry_size) as u16;
        data[header_offset..header_offset + 2].copy_from_slice(&new_num_keys.to_le_bytes());
        data[header_offset + 2..header_offset + 4].copy_from_slice(&new_free_offset.to_le_bytes());

        Ok(())
    }

    /// Writes entries to the page using write_to_slice (no BytesMut alloc).
    fn write_entries(&mut self, entries: &[InternalEntry]) -> Result<()> {
        let mut header = self.internal_header();
        let mut offset = Self::DATA_START + Self::LEFTMOST_PTR_SIZE;

        for entry in entries {
            let entry_size = entry.size_on_disk();
            if offset + entry_size > PAGE_SIZE {
                return Err(ZyronError::NodeFull);
            }
            entry.write_to_slice(&mut *self.data, offset);
            offset += entry_size;
        }

        header.num_keys = entries.len() as u16;
        header.free_space_offset = offset as u16;
        self.set_internal_header(header);
        Ok(())
    }

    /// Splits this internal node. Returns (promoted_key, new_right_page).
    /// Uses raw pointer reads to avoid per-entry Bytes allocation.
    pub fn split(&mut self, new_page_id: PageId) -> (Bytes, BTreeInternalPage) {
        // SAFETY: Views read via raw pointer. Left half is written to self (sequential
        // entries start at same offset, but left half is always <= original size so
        // the write is bounded within the original data). Right half writes to a
        // different buffer (right_page).
        let views = unsafe { self.entry_views_raw() };
        let mid = views.len() / 2;

        let promoted_key = Bytes::copy_from_slice(views[mid].key);
        let right_first_child = views[mid].child_page_id;
        let level = self.level();

        // Left half: entries are already in place at their current offsets
        // (they're the first `mid` entries in sequential order). Just update the header.
        let mut left_end = Self::DATA_START + Self::LEFTMOST_PTR_SIZE;
        for view in &views[..mid] {
            left_end += view.size_on_disk();
        }
        let mut header = self.internal_header();
        header.num_keys = mid as u16;
        header.free_space_offset = left_end as u16;
        self.set_internal_header(header);

        // Write right half to new page (different buffer, no overlap)
        let mut right_page = BTreeInternalPage::new(new_page_id, level);
        right_page.set_leftmost_child(right_first_child);
        let mut offset = Self::DATA_START + Self::LEFTMOST_PTR_SIZE;
        for view in &views[mid + 1..] {
            view.write_to_slice(&mut *right_page.data, offset);
            offset += view.size_on_disk();
        }
        let mut rh = right_page.internal_header();
        rh.num_keys = (views.len() - mid - 1) as u16;
        rh.free_space_offset = offset as u16;
        right_page.set_internal_header(rh);

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
        // SAFETY: Views read via raw pointer. Deleting one entry and rewriting
        // the rest sequentially is safe because the written data is always
        // <= the original data size (one entry removed).
        let views = unsafe { self.entry_views_raw() };
        let pos = views.iter().position(|v| v.key == key);

        match pos {
            Some(idx) => {
                // Entries before idx are already in place. Compute offset of entry[idx].
                let mut offset = Self::DATA_START + Self::LEFTMOST_PTR_SIZE;
                for view in &views[..idx] {
                    offset += view.size_on_disk();
                }
                // Skip entry[idx], write entries after it starting at offset
                for view in &views[idx + 1..] {
                    view.write_to_slice(&mut *self.data, offset);
                    offset += view.size_on_disk();
                }
                let mut header = self.internal_header();
                header.num_keys = (views.len() - 1) as u16;
                header.free_space_offset = offset as u16;
                self.set_internal_header(header);

                if self.is_underfull() {
                    DeleteResult::Underfull
                } else {
                    DeleteResult::Ok
                }
            }
            None => DeleteResult::NotFound,
        }
    }

    /// Borrows an entry from a right sibling to fix underflow.
    pub fn borrow_from_right(
        &mut self,
        right_sibling: &mut BTreeInternalPage,
        separator_key: Bytes,
    ) -> Option<Bytes> {
        if right_sibling.num_keys() <= 1 {
            return None;
        }

        // SAFETY: Raw reads from both pages. Self grows by one entry (appended past
        // existing data). Right sibling shrinks (fewer entries, no overlap issue).
        let right_views = unsafe { right_sibling.entry_views_raw() };
        let my_views = unsafe { self.entry_views_raw() };

        let new_sep = Bytes::copy_from_slice(right_views[0].key);
        let borrowed_child = right_views[0].child_page_id;
        let right_leftmost = right_sibling.leftmost_child();

        // Self: existing entries are already in place. Append separator past them.
        let sep_view = InternalEntryView {
            key: &separator_key,
            child_page_id: right_leftmost,
        };
        let my_count = my_views.len() + 1;
        let mut append_offset = Self::DATA_START + Self::LEFTMOST_PTR_SIZE;
        for view in my_views.iter() {
            append_offset += view.size_on_disk();
        }
        sep_view.write_to_slice(&mut *self.data, append_offset);
        append_offset += sep_view.size_on_disk();
        let mut header = self.internal_header();
        header.num_keys = my_count as u16;
        header.free_space_offset = append_offset as u16;
        self.set_internal_header(header);

        // Right sibling: remove first entry by shifting remaining entries left.
        // Uses copy_within to handle overlapping regions safely.
        right_sibling.set_leftmost_child(borrowed_child);
        let first_entry_size = right_views[0].size_on_disk();
        let entries_start = Self::DATA_START + Self::LEFTMOST_PTR_SIZE;
        let old_free = right_sibling.internal_header().free_space_offset as usize;
        let src_start = entries_start + first_entry_size;
        let shift_len = old_free - src_start;
        if shift_len > 0 {
            right_sibling
                .data
                .copy_within(src_start..old_free, entries_start);
        }
        let mut rh = right_sibling.internal_header();
        rh.num_keys = (right_views.len() - 1) as u16;
        rh.free_space_offset = (old_free - first_entry_size) as u16;
        right_sibling.set_internal_header(rh);

        Some(new_sep)
    }

    /// Borrows an entry from a left sibling to fix underflow.
    pub fn borrow_from_left(
        &mut self,
        left_sibling: &mut BTreeInternalPage,
        separator_key: Bytes,
    ) -> Option<Bytes> {
        if left_sibling.num_keys() <= 1 {
            return None;
        }

        // SAFETY: Raw reads. Self grows by one entry (prepended). Left shrinks.
        // For self: prepending shifts all data, but raw views point to old locations
        // which are read before being overwritten. Since we write sequentially from
        // the start, entry[0] is written first (from sep_view, not from self),
        // then entry[1] is written from my_views[0] which was read before any overlap.
        let left_views = unsafe { left_sibling.entry_views_raw() };
        let my_views = unsafe { self.entry_views_raw() };

        let last_idx = left_views.len() - 1;
        let new_sep = Bytes::copy_from_slice(left_views[last_idx].key);
        let borrowed_child = left_views[last_idx].child_page_id;
        let my_leftmost = self.leftmost_child();

        // Write self: separator + existing entries
        self.set_leftmost_child(borrowed_child);
        let sep_view = InternalEntryView {
            key: &separator_key,
            child_page_id: my_leftmost,
        };
        let my_count = 1 + my_views.len();
        // For prepend on sequential layout, we need owned entries to avoid overlap
        let owned_my: Vec<InternalEntry> = my_views.iter().map(|v| v.to_owned()).collect();
        let mut offset = Self::DATA_START + Self::LEFTMOST_PTR_SIZE;
        sep_view.write_to_slice(&mut *self.data, offset);
        offset += sep_view.size_on_disk();
        for entry in &owned_my {
            entry.write_to_slice(&mut *self.data, offset);
            offset += entry.size_on_disk();
        }
        let mut header = self.internal_header();
        header.num_keys = my_count as u16;
        header.free_space_offset = offset as u16;
        self.set_internal_header(header);

        // Left sibling: removing last entry is just a header update (truncation).
        // Entries before last_idx are already in place.
        let mut left_end = Self::DATA_START + Self::LEFTMOST_PTR_SIZE;
        for view in &left_views[..last_idx] {
            left_end += view.size_on_disk();
        }
        let mut lh = left_sibling.internal_header();
        lh.num_keys = last_idx as u16;
        lh.free_space_offset = left_end as u16;
        left_sibling.set_internal_header(lh);

        Some(new_sep)
    }

    /// Merges this internal node with its right sibling.
    pub fn merge_with_right(
        &mut self,
        right_sibling: &BTreeInternalPage,
        separator_key: Bytes,
    ) -> bool {
        // SAFETY: Raw reads from both pages. Self grows (existing entries + sep + right),
        // but existing entries are written first at the same offsets (identity copy),
        // then separator and right entries are appended past the original data.
        let my_views = unsafe { self.entry_views_raw() };
        let right_views = unsafe { right_sibling.entry_views_raw() };

        let right_leftmost = right_sibling.leftmost_child();
        let sep_view = InternalEntryView {
            key: &separator_key,
            child_page_id: right_leftmost,
        };

        let total = my_views.len() + 1 + right_views.len();
        // Skip identity copy of my_views (already in place), start appending at their end
        let mut offset = Self::DATA_START + Self::LEFTMOST_PTR_SIZE;
        for view in my_views.iter() {
            offset += view.size_on_disk();
        }
        // Append separator + right entries past existing data
        for view in std::iter::once(&sep_view).chain(right_views.iter()) {
            let sz = view.size_on_disk();
            if offset + sz > PAGE_SIZE {
                return false;
            }
            view.write_to_slice(&mut *self.data, offset);
            offset += sz;
        }
        let mut header = self.internal_header();
        header.num_keys = total as u16;
        header.free_space_offset = offset as u16;
        self.set_internal_header(header);

        true
    }
}
