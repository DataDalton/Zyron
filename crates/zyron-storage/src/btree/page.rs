//! B+Tree page implementations (leaf and internal nodes).

use super::constants::MIN_FILL_FACTOR;
use super::types::{
    DeleteResult, InternalEntry, InternalEntryView, InternalPageHeader, LeafEntry, LeafEntryView,
    LeafPageHeader, compare_keys,
};
use bytes::Bytes;
use zyron_common::RowLocator;
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

    /// Slot index holding `key`, or None when the leaf does not hold it.
    /// Searches the slot array directly, so it allocates nothing.
    #[inline]
    fn find_slot(data: &[u8], key: &[u8]) -> Option<usize> {
        let header_offset = LeafPageHeader::OFFSET;
        let num_slots = u16::from_le_bytes([data[header_offset], data[header_offset + 1]]) as usize;

        let mut low = 0usize;
        let mut high = num_slots;
        while low < high {
            let mid = low + (high - low) / 2;
            let slot_off = Self::SLOT_ARRAY_START + mid * Self::SLOT_SIZE;
            let entry_off = u16::from_le_bytes([data[slot_off], data[slot_off + 1]]) as usize;
            let key_len = u16::from_le_bytes([data[entry_off], data[entry_off + 1]]) as usize;
            let entry_key = &data[entry_off + 2..entry_off + 2 + key_len];

            let cmp = compare_keys(key, entry_key);
            if cmp == std::cmp::Ordering::Equal {
                return Some(mid);
            }
            let is_less = cmp == std::cmp::Ordering::Less;
            high = if is_less { mid } else { high };
            low = if is_less { low } else { mid + 1 };
        }
        None
    }

    /// Inserts a key-value pair into the leaf. Returns error if page is full.
    /// Uses single-pass in-place insertion for efficiency.
    #[inline]
    pub fn insert(&mut self, key: Bytes, locator: RowLocator) -> Result<()> {
        Self::insert_in_slice(&mut *self.data, &key, locator)
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
    pub fn get(&self, key: &[u8]) -> Option<RowLocator> {
        Self::get_in_slice(&*self.data, key)
    }

    /// Gets the value for a key using slotted page format
    /// Binary search directly on slot array for O(log n) lookup, no offset building needed
    /// Heap locators decode with file_id=0. Caller sets file_id from index context
    ///
    /// Search loop is structured for branchless lowering, the Less/Greater
    /// arms become cmov pairs for low and high, only the rare Equal arm
    /// takes a real branch which is well-predicted on the typical
    /// either-found-once-or-not-at-all access pattern
    #[inline(always)]
    pub fn get_in_slice(data: &[u8], key: &[u8]) -> Option<RowLocator> {
        // Parse header - read num_slots as single u16
        let header_offset = LeafPageHeader::OFFSET;
        let num_slots = u16::from_le_bytes([data[header_offset], data[header_offset + 1]]) as usize;

        if num_slots == 0 {
            return None;
        }

        // Branchless binary search with packed slot reads
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

            let cmp = compare_keys(key, entry_key);
            if cmp == std::cmp::Ordering::Equal {
                let payload_offset = entry_off + 2 + key_len;
                return RowLocator::read_payload(&data[payload_offset..]);
            }
            let is_less = cmp == std::cmp::Ordering::Less;
            high = if is_less { mid } else { high };
            low = if is_less { low } else { mid + 1 };
        }
        None
    }

    /// Inserts a key-value pair using slotted page format
    /// Binary search for O(log n) lookup, only shift 4-byte slots instead of full entries
    /// Stores the fixed RowLocator payload per entry (heap file_id is implicit)
    /// Returns Ok(()) on success, Err(NodeFull) if page is full, Err(DuplicateKey) if key exists
    ///
    /// Search loop is structured for branchless lowering, both low and high
    /// are updated via plain assignments selected by the comparison result so
    /// the compiler can emit cmov instead of conditional branches, only the
    /// duplicate-key path takes a real branch and that one is correctly
    /// predicted on the rare-Equal common case
    #[inline(always)]
    pub fn insert_in_slice(data: &mut [u8], key: &[u8], locator: RowLocator) -> Result<()> {
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

        // Entry size: key_len(2) + key + locator payload
        let entry_size = 2 + key.len() + locator.payload_len();

        // Calculate free space: between slot array end and data start
        let slot_array_end = Self::SLOT_ARRAY_START + num_slots * Self::SLOT_SIZE;
        let free_space = data_end.saturating_sub(slot_array_end);

        // Need space for both entry data and new slot
        if free_space < entry_size + Self::SLOT_SIZE {
            return Err(ZyronError::NodeFull);
        }

        // Branchless binary search through slot array to find insertion point
        // The match-on-Ordering form had an unpredictable 3-way branch per
        // iteration which mispredicted ~50% of the time on random keys, this
        // form lowers to cmp + cmov for low/high updates and a single
        // predictable branch for the rare duplicate-key path
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

            let cmp = compare_keys(key, entry_key);
            if cmp == std::cmp::Ordering::Equal {
                return Err(ZyronError::DuplicateKey);
            }
            let is_less = cmp == std::cmp::Ordering::Less;
            // Both arms unconditional, compiler lowers to cmov pair
            high = if is_less { mid } else { high };
            low = if is_less { low } else { mid + 1 };
        }

        let insert_slot_idx = low;

        // Write entry data at the end (grows backward)
        let new_data_end = data_end - entry_size;
        let mut write_offset = new_data_end;
        data[write_offset..write_offset + 2].copy_from_slice(&(key.len() as u16).to_le_bytes());
        write_offset += 2;
        data[write_offset..write_offset + key.len()].copy_from_slice(key);
        write_offset += key.len();
        locator.write_payload(&mut data[write_offset..]);

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
        let Some(idx) = Self::find_slot(&*self.data, key) else {
            return DeleteResult::NotFound;
        };

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

    /// Splits this leaf into two halves at the midpoint, returns
    /// (split_key, new_right_page). Uses entry_views to avoid Bytes
    /// allocation during read, materializes both halves for the rewrite
    /// (required since write regions overlap reads)
    pub fn split(&mut self, new_page_id: PageId) -> (Bytes, BTreeLeafPage) {
        self.split_for_key(None, new_page_id)
    }

    /// Splits this leaf, biasing the layout for an upcoming insert
    ///
    /// When `new_key` is given and is strictly greater than every key
    /// already on this page, the page is performing a rightmost-insert
    /// (auto-increment, time-ordered, UUID v7 with timestamp prefix, etc).
    /// In that case the existing entries all stay on the left page and
    /// the right page starts empty, so the next insert lands alone on
    /// the right and existing pages stay densely packed near 100%
    /// instead of the 50% utilization a midpoint split produces. The
    /// detection costs one byte comparison per insert and is no-op for
    /// random/non-monotonic key distributions which fall through to the
    /// midpoint split
    pub fn split_for_key(
        &mut self,
        new_key: Option<&[u8]>,
        new_page_id: PageId,
    ) -> (Bytes, BTreeLeafPage) {
        if let Some(key) = new_key {
            let views = self.entry_views();
            if let Some(last) = views.last()
                && compare_keys(key, last.key).is_gt()
            {
                // Right-bias split, leaf keeps every existing entry, the
                // empty right page is sized to receive the incoming key
                let split_key = Bytes::copy_from_slice(key);
                drop(views);
                let mut right_page = BTreeLeafPage::new(new_page_id);
                let old_next = self.next_leaf();
                self.set_next_leaf(Some(new_page_id));
                right_page.set_next_leaf(old_next);
                return (split_key, right_page);
            }
        }

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

    /// First byte of the slot array, immediately after the leftmost pointer.
    const SLOT_START: usize = Self::DATA_START + Self::LEFTMOST_PTR_SIZE;

    /// Bytes per slot: key head (8) + child page_num (4) + key_len (2) +
    /// key_off (2). A power of two, so slot addressing is a shift
    const SLOT_SIZE: usize = 16;

    /// Keys this long or shorter are held entirely in the slot's key head,
    /// so they cost no bytes outside the slot array
    const INLINE_KEY_LEN: usize = 8;

    /// Bytes available to the slot array and the long key region together.
    const USABLE: usize = PAGE_SIZE - Self::SLOT_START;

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

    /// The first eight key bytes in big-endian order, zero padded. Comparing
    /// two of these as u64 reproduces lexicographic order over the first
    /// eight bytes, and the zero padding keeps a short key below any longer
    /// key that extends it
    #[inline(always)]
    fn key_head(key: &[u8]) -> u64 {
        let mut head = [0u8; 8];
        let n = if key.len() < Self::INLINE_KEY_LEN {
            key.len()
        } else {
            Self::INLINE_KEY_LEN
        };
        head[..n].copy_from_slice(&key[..n]);
        u64::from_be_bytes(head)
    }

    /// Byte offset of slot `idx`.
    #[inline(always)]
    const fn slot_offset(idx: usize) -> usize {
        Self::SLOT_START + idx * Self::SLOT_SIZE
    }

    #[inline(always)]
    fn slot_head(data: &[u8], slot_off: usize) -> u64 {
        u64::from_be_bytes([
            data[slot_off],
            data[slot_off + 1],
            data[slot_off + 2],
            data[slot_off + 3],
            data[slot_off + 4],
            data[slot_off + 5],
            data[slot_off + 6],
            data[slot_off + 7],
        ])
    }

    #[inline(always)]
    fn slot_child(data: &[u8], slot_off: usize) -> u32 {
        u32::from_le_bytes([
            data[slot_off + 8],
            data[slot_off + 9],
            data[slot_off + 10],
            data[slot_off + 11],
        ])
    }

    #[inline(always)]
    fn slot_key_len(data: &[u8], slot_off: usize) -> usize {
        u16::from_le_bytes([data[slot_off + 12], data[slot_off + 13]]) as usize
    }

    #[inline(always)]
    fn slot_key_off(data: &[u8], slot_off: usize) -> usize {
        u16::from_le_bytes([data[slot_off + 14], data[slot_off + 15]]) as usize
    }

    /// The key bytes of one slot. A key of at most eight bytes is read
    /// straight out of the big-endian key head, so it needs no second load
    /// and no space outside the slot
    #[inline(always)]
    fn slot_key(data: &[u8], slot_off: usize) -> &[u8] {
        let len = Self::slot_key_len(data, slot_off);
        if len <= Self::INLINE_KEY_LEN {
            &data[slot_off..slot_off + len]
        } else {
            let off = Self::slot_key_off(data, slot_off);
            &data[off..off + len]
        }
    }

    /// Orders `key` against the key in slot `slot_off`. `head` is the caller's
    /// precomputed key head. Differing heads settle the comparison from the
    /// slot alone, two short keys settle it from the lengths, and only a key
    /// longer than the head reaches a full byte compare
    #[inline(always)]
    fn cmp_slot(data: &[u8], slot_off: usize, key: &[u8], head: u64) -> std::cmp::Ordering {
        let slot_head = Self::slot_head(data, slot_off);
        if slot_head != head {
            return head.cmp(&slot_head);
        }
        let len = Self::slot_key_len(data, slot_off);
        if key.len() <= Self::INLINE_KEY_LEN && len <= Self::INLINE_KEY_LEN {
            return key.len().cmp(&len);
        }
        key.cmp(Self::slot_key(data, slot_off))
    }

    /// Index of the first slot whose key is strictly greater than `key`.
    ///
    /// The loop is structured for branchless lowering, both bounds are
    /// updated by plain assignments the compiler turns into a cmov pair, so
    /// a descent costs one predictable-free comparison per level of the
    /// slot array rather than a mispredicting branch
    #[inline(always)]
    fn upper_bound(data: &[u8], num_keys: usize, key: &[u8], head: u64) -> usize {
        let mut low = 0usize;
        let mut high = num_keys;
        while low < high {
            let mid = low + (high - low) / 2;
            let is_less = Self::cmp_slot(data, Self::slot_offset(mid), key, head).is_lt();
            high = if is_less { mid } else { high };
            low = if is_less { low } else { mid + 1 };
        }
        low
    }

    /// Number of keys in this internal node.
    pub fn num_keys(&self) -> u16 {
        self.internal_header().num_keys
    }

    /// Level of this node in the tree.
    pub fn level(&self) -> u16 {
        self.internal_header().level
    }

    /// Bytes still available for slots and long keys.
    pub fn free_space(&self) -> usize {
        let header = self.internal_header();
        let slot_end = Self::slot_offset(header.num_keys as usize);
        (header.key_region_start as usize).saturating_sub(slot_end)
    }

    /// Returns the leftmost child pointer.
    pub fn leftmost_child(&self) -> PageId {
        Self::leftmost_child_in_slice(&*self.data)
    }

    /// Sets the leftmost child pointer.
    pub fn set_leftmost_child(&mut self, page_id: PageId) {
        let offset = Self::DATA_START;
        self.data[offset..offset + Self::LEFTMOST_PTR_SIZE]
            .copy_from_slice(&page_id.as_u64().to_le_bytes());
    }

    /// Reads all entries from the internal node (allocates per key).
    pub fn entries(&self) -> Vec<InternalEntry> {
        let num_keys = self.num_keys() as usize;
        let mut entries = Vec::with_capacity(num_keys);
        for idx in 0..num_keys {
            let slot_off = Self::slot_offset(idx);
            entries.push(InternalEntry {
                key: Bytes::copy_from_slice(Self::slot_key(&*self.data, slot_off)),
                child_page_id: PageId::new(0, Self::slot_child(&*self.data, slot_off) as u64),
            });
        }
        entries
    }

    /// Zero-copy read of all entries. Borrows keys from page buffer.
    pub fn entry_views(&self) -> Vec<InternalEntryView<'_>> {
        let num_keys = self.num_keys() as usize;
        let mut views = Vec::with_capacity(num_keys);
        for idx in 0..num_keys {
            let slot_off = Self::slot_offset(idx);
            views.push(InternalEntryView {
                key: Self::slot_key(&*self.data, slot_off),
                child_page_id: PageId::new(0, Self::slot_child(&*self.data, slot_off) as u64),
            });
        }
        views
    }

    /// Finds the child page for a given key.
    pub fn find_child(&self, key: &[u8]) -> PageId {
        Self::find_child_in_slice(&*self.data, key)
    }

    /// Reads the leftmost child pointer straight from the page bytes, so a
    /// descent does not have to materialize a whole BTreeInternalPage.
    #[inline(always)]
    pub fn leftmost_child_in_slice(data: &[u8]) -> PageId {
        let offset = Self::DATA_START;
        PageId::from_u64(u64::from_le_bytes([
            data[offset],
            data[offset + 1],
            data[offset + 2],
            data[offset + 3],
            data[offset + 4],
            data[offset + 5],
            data[offset + 6],
            data[offset + 7],
        ]))
    }

    /// Finds the child page for a given key directly from raw page data.
    /// Child pointers are stored as page_num (u32), so the returned PageId
    /// carries file_id 0 and callers stamp the owning file from context.
    ///
    /// Binary search runs over the fixed-stride slot array, so one probe is
    /// one cache line that already holds the key head, the child pointer and
    /// the key length. A page of short keys is routed without ever reading
    /// the key bytes
    #[inline(always)]
    pub fn find_child_in_slice(data: &[u8], key: &[u8]) -> PageId {
        let header_offset = InternalPageHeader::OFFSET;
        let num_keys = u16::from_le_bytes([data[header_offset], data[header_offset + 1]]) as usize;

        if num_keys == 0 {
            return Self::leftmost_child_in_slice(data);
        }

        let head = Self::key_head(key);
        let pos = Self::upper_bound(data, num_keys, key, head);
        if pos == 0 {
            Self::leftmost_child_in_slice(data)
        } else {
            PageId::new(0, Self::slot_child(data, Self::slot_offset(pos - 1)) as u64)
        }
    }

    /// Like `find_child_in_slice` but also returns this node's exclusive
    /// upper-bound separator for the chosen child: the first key strictly
    /// greater than `key`, meaning the chosen subtree holds only keys less
    /// than it. `None` when `key` routes to the rightmost child (no bound at
    /// this level). Used by batched insert to route a sorted run of keys to
    /// one leaf without re-descending per key.
    pub fn find_child_with_upper(data: &[u8], key: &[u8]) -> (u32, Option<Vec<u8>>) {
        let header_offset = InternalPageHeader::OFFSET;
        let num_keys = u16::from_le_bytes([data[header_offset], data[header_offset + 1]]) as usize;
        let leftmost = Self::leftmost_child_in_slice(data).page_num as u32;

        if num_keys == 0 {
            return (leftmost, None);
        }

        let head = Self::key_head(key);
        let pos = Self::upper_bound(data, num_keys, key, head);
        let child = if pos == 0 {
            leftmost
        } else {
            Self::slot_child(data, Self::slot_offset(pos - 1))
        };
        let bound = if pos < num_keys {
            Some(Self::slot_key(data, Self::slot_offset(pos)).to_vec())
        } else {
            None
        };
        (child, bound)
    }

    /// Inserts a key and right child pointer.
    #[inline]
    pub fn insert(&mut self, key: Bytes, right_child: PageId) -> Result<()> {
        Self::insert_in_slice(&mut *self.data, key.as_ref(), right_child)
    }

    /// Inserts a key and child pointer directly into raw page data.
    /// Stores child page_num as u32 (4 bytes). file_id is implicit from index context.
    /// Returns Ok(()) on success, Err(NodeFull) if page is full.
    #[inline(always)]
    pub fn insert_in_slice(data: &mut [u8], key: &[u8], right_child: PageId) -> Result<()> {
        if key.len() > u16::MAX as usize {
            return Err(ZyronError::NodeFull);
        }

        let header_offset = InternalPageHeader::OFFSET;
        let num_keys = u16::from_le_bytes([data[header_offset], data[header_offset + 1]]) as usize;
        let raw_region =
            u16::from_le_bytes([data[header_offset + 2], data[header_offset + 3]]) as usize;

        // A page that was never written through this codec carries a zero or
        // an out-of-range value here, the long key region then starts at the
        // page end
        let key_region_start = if raw_region <= Self::SLOT_START || raw_region > PAGE_SIZE {
            PAGE_SIZE
        } else {
            raw_region
        };

        let spill = key.len() > Self::INLINE_KEY_LEN;
        let needed = Self::SLOT_SIZE + if spill { key.len() } else { 0 };
        let slot_end = Self::slot_offset(num_keys);
        if key_region_start.saturating_sub(slot_end) < needed {
            return Err(ZyronError::NodeFull);
        }

        let head = Self::key_head(key);
        let pos = Self::upper_bound(data, num_keys, key, head);

        // Open a slot-sized gap at the insertion point
        let insert_off = Self::slot_offset(pos);
        if slot_end > insert_off {
            data.copy_within(insert_off..slot_end, insert_off + Self::SLOT_SIZE);
        }

        let mut new_region = key_region_start;
        let key_off = if spill {
            new_region -= key.len();
            data[new_region..new_region + key.len()].copy_from_slice(key);
            new_region
        } else {
            0
        };

        data[insert_off..insert_off + 8].copy_from_slice(&head.to_be_bytes());
        data[insert_off + 8..insert_off + 12]
            .copy_from_slice(&(right_child.page_num as u32).to_le_bytes());
        data[insert_off + 12..insert_off + 14].copy_from_slice(&(key.len() as u16).to_le_bytes());
        data[insert_off + 14..insert_off + 16].copy_from_slice(&(key_off as u16).to_le_bytes());

        data[header_offset..header_offset + 2]
            .copy_from_slice(&((num_keys + 1) as u16).to_le_bytes());
        data[header_offset + 2..header_offset + 4]
            .copy_from_slice(&(new_region as u16).to_le_bytes());

        Ok(())
    }

    /// Rewrites the long key region so it holds only the keys the slot array
    /// still points at, reclaiming the space left by replaced or removed
    /// separators. A page whose keys all fit in their slots has no region and
    /// skips the walk entirely
    fn compact_key_region(&mut self) {
        let num_keys = self.num_keys() as usize;
        let mut moved: Vec<(usize, Vec<u8>)> = Vec::new();
        for idx in 0..num_keys {
            let slot_off = Self::slot_offset(idx);
            let len = Self::slot_key_len(&*self.data, slot_off);
            if len > Self::INLINE_KEY_LEN {
                let off = Self::slot_key_off(&*self.data, slot_off);
                moved.push((slot_off, self.data[off..off + len].to_vec()));
            }
        }

        let mut region = PAGE_SIZE;
        for (slot_off, key) in &moved {
            region -= key.len();
            self.data[region..region + key.len()].copy_from_slice(key);
            self.data[slot_off + 14..slot_off + 16].copy_from_slice(&(region as u16).to_le_bytes());
        }

        let mut header = self.internal_header();
        header.key_region_start = region as u16;
        self.set_internal_header(header);
    }

    /// True when this page holds long keys, so its region is worth compacting.
    #[inline]
    fn has_long_keys(&self) -> bool {
        (self.internal_header().key_region_start as usize) < PAGE_SIZE
    }

    /// The separator key at entry index `idx`, or None when `idx` is past the
    /// end. Reads the one slot rather than materializing every entry.
    pub fn separator_key_at(&self, idx: usize) -> Option<Bytes> {
        if idx >= self.num_keys() as usize {
            return None;
        }
        Some(Bytes::copy_from_slice(Self::slot_key(
            &*self.data,
            Self::slot_offset(idx),
        )))
    }

    /// Total bytes the long key region owes to keys the slot array still
    /// points at.
    fn live_long_bytes(data: &[u8], num_keys: usize) -> usize {
        let mut total = 0;
        for idx in 0..num_keys {
            let len = Self::slot_key_len(data, Self::slot_offset(idx));
            if len > Self::INLINE_KEY_LEN {
                total += len;
            }
        }
        total
    }

    /// Replaces the separator key at entry index `idx`, keeping its child
    /// pointer. Used after a borrow shifts the boundary between two children.
    /// Returns false when the replacement key does not fit, in which case the
    /// page is left exactly as it was.
    pub fn set_separator_key(&mut self, idx: usize, new_key: Bytes) -> bool {
        let num_keys = self.num_keys() as usize;
        if idx >= num_keys || new_key.len() > u16::MAX as usize {
            return false;
        }

        let slot_off = Self::slot_offset(idx);
        let old_len = Self::slot_key_len(&*self.data, slot_off);
        let old_long = if old_len > Self::INLINE_KEY_LEN {
            old_len
        } else {
            0
        };
        let new_long = if new_key.len() > Self::INLINE_KEY_LEN {
            new_key.len()
        } else {
            0
        };

        let mut key_off = 0usize;
        if new_long > 0 || old_long > 0 {
            // The key being replaced gives its own bytes back, so measure the
            // region without it
            let slot_end = Self::slot_offset(num_keys);
            let in_use = Self::live_long_bytes(&*self.data, num_keys) - old_long;
            if new_long > (PAGE_SIZE - slot_end) - in_use {
                return false;
            }
            // Zeroing the length drops the old key from the live set, so the
            // compaction below reclaims its bytes
            self.data[slot_off + 12..slot_off + 14].copy_from_slice(&0u16.to_le_bytes());
            self.compact_key_region();
            if new_long > 0 {
                let mut header = self.internal_header();
                let region = header.key_region_start as usize - new_key.len();
                self.data[region..region + new_key.len()].copy_from_slice(&new_key);
                header.key_region_start = region as u16;
                self.set_internal_header(header);
                key_off = region;
            }
        }

        self.data[slot_off..slot_off + 8].copy_from_slice(&Self::key_head(&new_key).to_be_bytes());
        self.data[slot_off + 12..slot_off + 14]
            .copy_from_slice(&(new_key.len() as u16).to_le_bytes());
        self.data[slot_off + 14..slot_off + 16].copy_from_slice(&(key_off as u16).to_le_bytes());
        true
    }

    /// Removes entry `idx` (a separator and its right child pointer) from this
    /// node. The leftmost child pointer is untouched, so the child at slot
    /// `idx+1` is dropped from routing (callers merge that child away first).
    pub fn remove_entry(&mut self, idx: usize) {
        let num_keys = self.num_keys() as usize;
        if idx >= num_keys {
            return;
        }

        let start = Self::slot_offset(idx);
        let end = Self::slot_offset(num_keys);
        self.data.copy_within(start + Self::SLOT_SIZE..end, start);

        let mut header = self.internal_header();
        header.num_keys = (num_keys - 1) as u16;
        self.set_internal_header(header);

        if self.has_long_keys() {
            self.compact_key_region();
        }
    }

    /// Splits this internal node. Returns (promoted_key, new_right_page).
    ///
    /// The left half keeps its slots exactly where they are, so the split
    /// copies only the right half's slot range and, when the page carries
    /// long keys, their bytes.
    pub fn split(&mut self, new_page_id: PageId) -> (Bytes, BTreeInternalPage) {
        let num_keys = self.num_keys() as usize;
        let mid = num_keys / 2;
        let level = self.level();

        let mid_off = Self::slot_offset(mid);
        let promoted_key = Bytes::copy_from_slice(Self::slot_key(&*self.data, mid_off));
        let right_first_child = PageId::new(0, Self::slot_child(&*self.data, mid_off) as u64);

        let mut right_page = BTreeInternalPage::new(new_page_id, level);
        right_page.set_leftmost_child(right_first_child);

        let right_count = num_keys - mid - 1;
        let mut region = PAGE_SIZE;
        for i in 0..right_count {
            let src_off = Self::slot_offset(mid + 1 + i);
            let dst_off = Self::slot_offset(i);
            right_page.data[dst_off..dst_off + Self::SLOT_SIZE]
                .copy_from_slice(&self.data[src_off..src_off + Self::SLOT_SIZE]);

            let len = Self::slot_key_len(&*self.data, src_off);
            if len > Self::INLINE_KEY_LEN {
                let off = Self::slot_key_off(&*self.data, src_off);
                region -= len;
                right_page.data[region..region + len].copy_from_slice(&self.data[off..off + len]);
                right_page.data[dst_off + 14..dst_off + 16]
                    .copy_from_slice(&(region as u16).to_le_bytes());
            }
        }

        let mut right_header = right_page.internal_header();
        right_header.num_keys = right_count as u16;
        right_header.key_region_start = region as u16;
        right_page.set_internal_header(right_header);

        let mut header = self.internal_header();
        header.num_keys = mid as u16;
        self.set_internal_header(header);
        if self.has_long_keys() {
            self.compact_key_region();
        }

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
        let header = self.internal_header();
        if header.num_keys == 0 {
            return false;
        }
        let used = header.num_keys as usize * Self::SLOT_SIZE
            + (PAGE_SIZE - header.key_region_start as usize);
        (used as f64 / Self::USABLE as f64) < MIN_FILL_FACTOR
    }

    /// Deletes a key from the internal node. Returns DeleteResult indicating outcome.
    pub fn delete(&mut self, key: &[u8]) -> DeleteResult {
        let num_keys = self.num_keys() as usize;
        if num_keys == 0 {
            return DeleteResult::NotFound;
        }

        let head = Self::key_head(key);
        let pos = Self::upper_bound(&*self.data, num_keys, key, head);
        if pos == 0 {
            return DeleteResult::NotFound;
        }
        let idx = pos - 1;
        if Self::cmp_slot(&*self.data, Self::slot_offset(idx), key, head)
            != std::cmp::Ordering::Equal
        {
            return DeleteResult::NotFound;
        }

        self.remove_entry(idx);
        if self.is_underfull() {
            DeleteResult::Underfull
        } else {
            DeleteResult::Ok
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

        let first_off = Self::slot_offset(0);
        let new_sep = Bytes::copy_from_slice(Self::slot_key(&*right_sibling.data, first_off));
        let borrowed_child =
            PageId::new(0, Self::slot_child(&*right_sibling.data, first_off) as u64);
        let right_leftmost = right_sibling.leftmost_child();

        // The separator sorts above every key already here, so the insert
        // lands at the end of the slot array
        if Self::insert_in_slice(&mut *self.data, &separator_key, right_leftmost).is_err() {
            return None;
        }

        right_sibling.set_leftmost_child(borrowed_child);
        right_sibling.remove_entry(0);

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

        let last_idx = left_sibling.num_keys() as usize - 1;
        let last_off = Self::slot_offset(last_idx);
        let new_sep = Bytes::copy_from_slice(Self::slot_key(&*left_sibling.data, last_off));
        let borrowed_child = PageId::new(0, Self::slot_child(&*left_sibling.data, last_off) as u64);
        let my_leftmost = self.leftmost_child();

        // The separator sorts below every key already here, so the insert
        // lands at the front of the slot array
        if Self::insert_in_slice(&mut *self.data, &separator_key, my_leftmost).is_err() {
            return None;
        }
        self.set_leftmost_child(borrowed_child);

        left_sibling.remove_entry(last_idx);

        Some(new_sep)
    }

    /// Merges this internal node with its right sibling.
    pub fn merge_with_right(
        &mut self,
        right_sibling: &BTreeInternalPage,
        separator_key: Bytes,
    ) -> bool {
        let right_count = right_sibling.num_keys() as usize;

        let mut needed = Self::SLOT_SIZE * (right_count + 1);
        if separator_key.len() > Self::INLINE_KEY_LEN {
            needed += separator_key.len();
        }
        for idx in 0..right_count {
            let len = Self::slot_key_len(&*right_sibling.data, Self::slot_offset(idx));
            if len > Self::INLINE_KEY_LEN {
                needed += len;
            }
        }

        if self.free_space() < needed && self.has_long_keys() {
            self.compact_key_region();
        }
        if self.free_space() < needed {
            return false;
        }

        let right_leftmost = right_sibling.leftmost_child();
        if Self::insert_in_slice(&mut *self.data, &separator_key, right_leftmost).is_err() {
            return false;
        }
        for idx in 0..right_count {
            let slot_off = Self::slot_offset(idx);
            let key = Self::slot_key(&*right_sibling.data, slot_off);
            let child = PageId::new(0, Self::slot_child(&*right_sibling.data, slot_off) as u64);
            if Self::insert_in_slice(&mut *self.data, key, child).is_err() {
                return false;
            }
        }

        true
    }
}
