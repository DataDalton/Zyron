//! Heap page implementation using slotted page format.
//!
//! Page layout:
//! ```text
//! +------------------+
//! | Page Header (40) |
//! +------------------+
//! | Slot Array       |  <- Grows downward
//! | (16 bytes/slot)  |     offset plus the tuple header
//! +------------------+
//! |                  |
//! | Free Space       |
//! |                  |
//! +------------------+
//! | Tuple Data       |  <- Grows upward
//! +------------------+
//! ```
//!
//! Each slot carries its tuple header rather than storing it beside the data.
//! A scan then walks one dense array instead of following an offset into the
//! data region for every row, and stamping xmax writes into that array too.
//! The bytes move rather than grow: the slot gains the twelve header bytes
//! that the data region loses.

use super::constants::{DATA_START, HEAP_HEADER_OFFSET, HEAP_HEADER_SIZE, TUPLE_SLOT_SIZE};
use crate::TupleId;
use crate::tuple::{Tuple, TupleFlags, TupleHeader, TupleView};
use zyron_common::page::{PAGE_SIZE, PageHeader, PageId, PageType};
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

/// Rows ahead of the copy whose bytes are warmed. Measured against a bare
/// copy loop over the same rows, eight is where the gain flattens
const INSERT_PREFETCH_ROWS: usize = 8;

/// Starts the loads for one row body without waiting on them.
#[inline(always)]
fn prefetch_row(row: &Tuple) {
    #[cfg(target_arch = "x86_64")]
    {
        let data = row.data();
        let ptr = data.as_ptr();
        let mut at = 0usize;
        while at < data.len() {
            // SAFETY: at stays below the row length, and a prefetch has no
            // architectural effect beyond warming the cache
            unsafe {
                std::arch::x86_64::_mm_prefetch(
                    ptr.add(at) as *const i8,
                    std::arch::x86_64::_MM_HINT_T0,
                );
            }
            at += 64;
        }
    }
    #[cfg(not(target_arch = "x86_64"))]
    let _ = row;
}

/// A slot in the slot array. It points at the tuple data and carries the
/// tuple header, so reading a row header costs no second load.
///
/// Layout (16 bytes):
/// - offset: 2 bytes (offset from page start to tuple data, 0 = empty slot)
/// - data_len: 2 bytes
/// - flags: 2 bytes
/// - reserved: 2 bytes
/// - xmin: 4 bytes
/// - xmax: 4 bytes
///
/// offset and data_len share the leading u32, which lets a writer publish a
/// finished slot with one release store once the fields above it are in place
#[derive(Debug, Clone, Copy, Default)]
pub struct TupleSlot {
    /// Offset from page start to tuple data, 0 on an empty slot.
    pub offset: u16,
    /// The tuple header, held here rather than beside the data.
    pub header: TupleHeader,
}

impl TupleSlot {
    /// Size of a slot entry in bytes.
    pub const SIZE: usize = TUPLE_SLOT_SIZE;

    /// Byte offset of the xmax field within a slot.
    pub(crate) const XMAX_OFFSET: usize = 12;

    /// Creates a slot for a tuple whose data begins at `offset`.
    pub fn new(offset: u16, header: TupleHeader) -> Self {
        Self { offset, header }
    }

    /// The slot a deleted tuple leaves behind. Offset 0 lies inside the page
    /// header, so it can never name real tuple data
    pub fn empty() -> Self {
        Self {
            offset: 0,
            header: TupleHeader::default(),
        }
    }

    /// Returns true if this slot is empty/deleted.
    pub fn is_empty(&self) -> bool {
        self.offset == 0
    }

    /// Bytes this tuple occupies in the data region.
    pub fn data_len(&self) -> usize {
        self.header.data_len as usize
    }

    /// Serializes the slot to bytes.
    pub fn to_bytes(&self) -> [u8; Self::SIZE] {
        let mut buf = [0u8; Self::SIZE];
        buf[0..2].copy_from_slice(&self.offset.to_le_bytes());
        buf[2..4].copy_from_slice(&self.header.data_len.to_le_bytes());
        buf[4..6].copy_from_slice(&self.header.flags.0.to_le_bytes());
        buf[8..12].copy_from_slice(&self.header.xmin.to_le_bytes());
        buf[12..16].copy_from_slice(&self.header.xmax.to_le_bytes());
        buf
    }

    /// Deserializes a slot from bytes.
    pub fn from_bytes(buf: &[u8]) -> Self {
        Self {
            offset: u16::from_le_bytes([buf[0], buf[1]]),
            header: TupleHeader {
                flags: TupleFlags(u16::from_le_bytes([buf[4], buf[5]])),
                data_len: u16::from_le_bytes([buf[2], buf[3]]),
                xmin: u32::from_le_bytes([buf[8], buf[9], buf[10], buf[11]]),
                xmax: u32::from_le_bytes([buf[12], buf[13], buf[14], buf[15]]),
            },
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
    pub const SIZE: usize = HEAP_HEADER_SIZE;

    /// Offset of heap header in page (after PageHeader).
    pub const OFFSET: usize = HEAP_HEADER_OFFSET;

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

    /// Packs the four u16 fields into one u64 in little-endian layout that
    /// matches the on-disk serialization. Used by the lock-free insert path
    /// for atomic CAS on the in-memory header bytes.
    #[inline]
    pub fn to_u64(self) -> u64 {
        (self.slot_count as u64)
            | ((self.free_space_start as u64) << 16)
            | ((self.free_space_end as u64) << 32)
            | ((self.reserved as u64) << 48)
    }

    /// Inverse of to_u64.
    #[inline]
    pub fn from_u64(v: u64) -> Self {
        Self {
            slot_count: v as u16,
            free_space_start: (v >> 16) as u16,
            free_space_end: (v >> 32) as u16,
            reserved: (v >> 48) as u16,
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
    /// Offset where slot array begins (after PageHeader + HeapPageHeader).
    pub const DATA_START: usize = DATA_START;

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

    /// Initializes a reused page buffer without zeroing the full 8KB.
    /// Only writes the 40-byte header area. Safe for reused buffers because
    /// slot_count=0 prevents reading stale slot data and free_space_end=PAGE_SIZE
    /// prevents reading stale tuple data. Each inserted tuple overwrites the
    /// next slot and tuple position sequentially.
    #[inline]
    pub fn init_fresh_slice_reuse(data: &mut [u8; PAGE_SIZE], page_id: PageId) {
        let page_header = PageHeader::new(page_id, PageType::Heap);
        data[..PageHeader::SIZE].copy_from_slice(&page_header.to_bytes());
        let heap_header = HeapPageHeader::new();
        let offset = HeapPageHeader::OFFSET;
        data[offset..offset + HeapPageHeader::SIZE].copy_from_slice(&heap_header.to_bytes());
    }

    /// Creates a heap page from raw page data.
    pub fn from_bytes(data: [u8; PAGE_SIZE]) -> Self {
        Self {
            data: Box::new(data),
        }
    }

    // =========================================================================
    // Static In-Slice Methods (for zero-copy operations)
    // =========================================================================

    /// Reads the heap header from a slice.
    #[inline]
    pub fn heap_header_from_slice(data: &[u8]) -> HeapPageHeader {
        let offset = HeapPageHeader::OFFSET;
        HeapPageHeader::from_bytes(&data[offset..offset + HeapPageHeader::SIZE])
    }

    /// Writes the heap header to a slice.
    #[inline]
    pub fn set_heap_header_in_slice(data: &mut [u8], header: HeapPageHeader) {
        let offset = HeapPageHeader::OFFSET;
        data[offset..offset + HeapPageHeader::SIZE].copy_from_slice(&header.to_bytes());
    }

    /// Returns free space from a slice.
    #[inline]
    pub fn free_space_in_slice(data: &[u8]) -> usize {
        Self::heap_header_from_slice(data).free_space()
    }

    /// Returns total usable space (free + reclaimable from deleted tuples) from a slice.
    #[inline]
    pub fn total_usable_space_in_slice(data: &[u8]) -> usize {
        let header = Self::heap_header_from_slice(data);
        let free = header.free_space();
        let mut active_tuple_space = 0usize;
        for i in 0..header.slot_count {
            if let Some(slot) = Self::get_slot_from_slice(data, SlotId(i), header.slot_count)
                && !slot.is_empty()
            {
                active_tuple_space += slot.data_len();
            }
        }
        let tuple_area_size = (PAGE_SIZE as u16 - header.free_space_end) as usize;
        free + tuple_area_size.saturating_sub(active_tuple_space)
    }

    /// Stamps `xmax` on a tuple in place within a page slice. The field lives
    /// in the tuple slot. Returns false if the slot is empty. Used by the
    /// delete/update path while holding the frame write lock so the change is
    /// not lost to a concurrent append.
    pub fn set_tuple_xmax_in_slice(data: &mut [u8], slot_id: SlotId, xmax: u32) -> bool {
        let header = Self::heap_header_from_slice(data);
        let Some(slot) = Self::get_slot_from_slice(data, slot_id, header.slot_count) else {
            return false;
        };
        if slot.is_empty() {
            return false;
        }
        let off = Self::slot_offset(slot_id) + TupleSlot::XMAX_OFFSET;
        data[off..off + 4].copy_from_slice(&xmax.to_le_bytes());
        true
    }

    /// Clears `xmax` back to 0 on a tuple in place within a page slice, undoing a
    /// self-delete stamp. Returns false if the slot is empty. Used by ROLLBACK TO
    /// SAVEPOINT to restore a row the transaction deleted after the savepoint.
    pub fn clear_tuple_xmax_in_slice(data: &mut [u8], slot_id: SlotId) -> bool {
        let header = Self::heap_header_from_slice(data);
        let Some(slot) = Self::get_slot_from_slice(data, slot_id, header.slot_count) else {
            return false;
        };
        if slot.is_empty() {
            return false;
        }
        let off = Self::slot_offset(slot_id) + TupleSlot::XMAX_OFFSET;
        data[off..off + 4].copy_from_slice(&0u32.to_le_bytes());
        true
    }

    /// Prunes tuples for which `is_dead(xmin, xmax)` returns true and reclaims
    /// their space by zeroing the slot and compacting the page. The caller
    /// supplies a predicate that is true only for versions dead to every live
    /// snapshot (a committed delete below the frozen horizon, or an aborted
    /// insert). Slot ids of surviving tuples are preserved by compaction, so
    /// outstanding tuple and index references stay valid. Returns true if any
    /// tuple was pruned. This is the on-access pruning that keeps MVCC-updated
    /// heaps compact without waiting for vacuum.
    pub fn prune_dead_in_slice(data: &mut [u8], is_dead: &impl Fn(u32, u32) -> bool) -> bool {
        let header = Self::heap_header_from_slice(data);
        let mut pruned = false;
        for i in 0..header.slot_count {
            let slot_id = SlotId(i);
            let Some(slot) = Self::get_slot_from_slice(data, slot_id, header.slot_count) else {
                continue;
            };
            if slot.is_empty() {
                continue;
            }
            if is_dead(slot.header.xmin, slot.header.xmax) {
                // Mark the slot empty; compaction below reclaims the bytes.
                Self::set_slot_in_slice(data, slot_id, TupleSlot::empty());
                pruned = true;
            }
        }
        if pruned {
            Self::compact_in_slice(data);
        }
        pruned
    }

    /// Vacuums a page slice in place. Two actions:
    /// 1. Clears the `xmax` stamp on a live row whose deleter aborted, so a row
    ///    that is still visible never carries a reference to an aborted
    ///    transaction below the frozen horizon (which would let the horizon, or
    ///    commit-status truncation, make it look deleted).
    /// 2. Prunes tuples dead to every snapshot (aborted insert, or committed
    ///    delete below the horizon) and compacts to reclaim their bytes.
    ///
    /// `is_dead(xmin, xmax)` marks reclaimable versions; `is_aborted(xid)` marks
    /// a transaction that aborted. Returns (tuples reclaimed, page modified).
    pub fn vacuum_in_slice(
        data: &mut [u8],
        is_dead: &impl Fn(u32, u32) -> bool,
        is_aborted: &impl Fn(u32) -> bool,
    ) -> (u64, bool) {
        Self::vacuum_in_slice_inner(data, is_dead, is_aborted, None)
    }

    /// Same as `vacuum_in_slice` but, before pruning, records each reclaimed
    /// tuple as (slot_id, row data) into `dead_out`. The vacuum worker uses this
    /// to delete the reclaimed rows' B+tree index entries: the composite index
    /// key includes the row's value and tuple id, so the entry is removed using
    /// the row image that is about to disappear from the heap.
    pub fn vacuum_in_slice_collect(
        data: &mut [u8],
        is_dead: &impl Fn(u32, u32) -> bool,
        is_aborted: &impl Fn(u32) -> bool,
        dead_out: &mut Vec<(u16, Vec<u8>)>,
    ) -> (u64, bool) {
        Self::vacuum_in_slice_inner(data, is_dead, is_aborted, Some(dead_out))
    }

    fn vacuum_in_slice_inner(
        data: &mut [u8],
        is_dead: &impl Fn(u32, u32) -> bool,
        is_aborted: &impl Fn(u32) -> bool,
        mut dead_out: Option<&mut Vec<(u16, Vec<u8>)>>,
    ) -> (u64, bool) {
        let header = Self::heap_header_from_slice(data);
        let mut reclaimed = 0u64;
        let mut modified = false;
        for i in 0..header.slot_count {
            let slot_id = SlotId(i);
            let Some(slot) = Self::get_slot_from_slice(data, slot_id, header.slot_count) else {
                continue;
            };
            if slot.is_empty() {
                continue;
            }
            if is_dead(slot.header.xmin, slot.header.xmax) {
                reclaimed += 1;
                // Capture the row image before prune_dead_in_slice zeroes it, so
                // its index entries can be removed against the heap-resident key.
                if let Some(out) = dead_out.as_deref_mut() {
                    let ds = slot.offset as usize;
                    let de = ds + slot.data_len();
                    if de <= data.len() {
                        out.push((i, data[ds..de].to_vec()));
                    }
                }
            } else if slot.header.xmax != 0 && is_aborted(slot.header.xmax) {
                // Live row whose deleter aborted: clear the stale stamp.
                let xoff = Self::slot_offset(slot_id) + TupleSlot::XMAX_OFFSET;
                data[xoff..xoff + 4].copy_from_slice(&0u32.to_le_bytes());
                modified = true;
            }
        }
        if Self::prune_dead_in_slice(data, is_dead) {
            modified = true;
        }
        (reclaimed, modified)
    }

    /// Compacts a page slice by moving all active tuples together.
    /// Eliminates holes from deleted tuples and maximizes contiguous free space.
    fn compact_in_slice(data: &mut [u8]) {
        let header = Self::heap_header_from_slice(data);

        // Collect active tuples: (slot_id, offset, length)
        let mut active: Vec<(SlotId, usize, usize)> = Vec::new();
        for i in 0..header.slot_count {
            let slot_id = SlotId(i);
            if let Some(slot) = Self::get_slot_from_slice(data, slot_id, header.slot_count)
                && !slot.is_empty()
            {
                active.push((slot_id, slot.offset as usize, slot.data_len()));
            }
        }

        // Pack tuples from the page end downward, processing highest original
        // offset first. The first-processed tuple (closest to the end) keeps
        // or moves toward the end; every later tuple has a lower original
        // offset and a lower destination, so an already-written destination
        // (higher address) can never clobber a not-yet-moved source (lower
        // address). Ascending order is unsafe in-place: writing the end-most
        // destination first overwrites the next tuple's source bytes.
        active.sort_unstable_by_key(|&(_, offset, _)| std::cmp::Reverse(offset));

        // Rewrite tuple data from the end of the page.
        let mut new_free_space_end = PAGE_SIZE as u16;
        for &(slot_id, old_offset, length) in &active {
            new_free_space_end -= length as u16;
            let new_offset = new_free_space_end as usize;
            data.copy_within(old_offset..old_offset + length, new_offset);
            let mut moved = Self::get_slot_from_slice(data, slot_id, header.slot_count)
                .unwrap_or_else(TupleSlot::empty);
            moved.offset = new_free_space_end;
            Self::set_slot_in_slice(data, slot_id, moved);
        }

        // Update header
        let mut new_header = header;
        new_header.free_space_end = new_free_space_end;
        Self::set_heap_header_in_slice(data, new_header);
    }

    /// Reads slot `slot` from a page image, or None when the index is past
    /// the slot array, the slot is empty, or its data range leaves the page.
    /// The one decoder every reader of a raw heap page should go through
    #[inline]
    pub fn live_slot_in_slice(data: &[u8], slot: u16) -> Option<TupleSlot> {
        let header = Self::heap_header_from_slice(data);
        let slot = Self::get_slot_from_slice(data, SlotId(slot), header.slot_count)?;
        if slot.is_empty() {
            return None;
        }
        if slot.offset as usize + slot.data_len() > data.len() {
            return None;
        }
        Some(slot)
    }

    /// Reads a slot from a slice.
    #[inline]
    fn get_slot_from_slice(data: &[u8], slot_id: SlotId, slot_count: u16) -> Option<TupleSlot> {
        if slot_id.0 >= slot_count {
            return None;
        }
        let offset = Self::DATA_START + (slot_id.0 as usize) * TupleSlot::SIZE;
        Some(TupleSlot::from_bytes(
            &data[offset..offset + TupleSlot::SIZE],
        ))
    }

    /// Writes a slot to a slice.
    #[inline]
    fn set_slot_in_slice(data: &mut [u8], slot_id: SlotId, slot: TupleSlot) {
        let offset = Self::DATA_START + (slot_id.0 as usize) * TupleSlot::SIZE;
        data[offset..offset + TupleSlot::SIZE].copy_from_slice(&slot.to_bytes());
    }

    /// Counts active tuples in a page slice without allocation.
    #[inline]
    pub fn tuple_count_in_slice(data: &[u8]) -> usize {
        let header = Self::heap_header_from_slice(data);
        let mut count = 0;
        for i in 0..header.slot_count {
            if let Some(slot) = Self::get_slot_from_slice(data, SlotId(i), header.slot_count)
                && !slot.is_empty()
            {
                count += 1;
            }
        }
        count
    }

    /// Reads a tuple from a slice at the given slot (allocates).
    #[inline]
    pub fn get_tuple_from_slice(data: &[u8], slot_id: SlotId) -> Option<Tuple> {
        let header = Self::heap_header_from_slice(data);
        let slot = Self::get_slot_from_slice(data, slot_id, header.slot_count)?;
        if slot.is_empty() {
            return None;
        }
        let start = slot.offset as usize;
        let end = start + slot.data_len();
        if end > data.len() {
            return None;
        }
        Some(Tuple::with_header(slot.header, data[start..end].to_vec()))
    }

    /// Zero-copy tuple read from a slice. Borrows data from the page buffer.
    #[inline]
    pub fn get_tuple_view_from_slice<'a>(data: &'a [u8], slot_id: SlotId) -> Option<TupleView<'a>> {
        let header = Self::heap_header_from_slice(data);
        let slot = Self::get_slot_from_slice(data, slot_id, header.slot_count)?;
        if slot.is_empty() {
            return None;
        }
        let start = slot.offset as usize;
        let end = start + slot.data_len();
        if end > data.len() {
            return None;
        }
        Some(TupleView::new(slot.header, &data[start..end]))
    }

    /// Lock-free burst insert, one CAS reserves N slots and N tuple-byte
    /// ranges, then writes tuple data and commits each slot via atomic
    /// Release store
    ///
    /// Returns the count actually inserted, less than `tuples.len()` when
    /// the page fills, 0 when nothing fits so the caller rolls over
    ///
    /// Concurrent writers on the same page interleave bursts, each CAS
    /// claims a disjoint slot+byte range. Readers observing slot_count >= N
    /// see either a non-zero slot length (committed) or zero (in flight)
    ///
    /// # Safety
    /// `page_ptr` points to a PAGE_SIZE buffer with an 8-byte aligned
    /// heap header at HEAP_HEADER_OFFSET, kept live and pinned through
    /// the call
    pub unsafe fn insert_tuples_burst(
        page_ptr: *mut u8,
        page_id: PageId,
        tuples: &[Tuple],
        results: &mut Vec<TupleId>,
    ) -> usize {
        use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};

        if tuples.is_empty() {
            return 0;
        }

        let header_atomic = unsafe { &*(page_ptr.add(HEAP_HEADER_OFFSET) as *const AtomicU64) };

        loop {
            let packed = header_atomic.load(Ordering::Acquire);
            let hdr = HeapPageHeader::from_u64(packed);
            let mut free = hdr.free_space();

            // Greedy fit, in input order
            let mut n_fit: usize = 0;
            let mut tuple_bytes_total: usize = 0;
            for t in tuples {
                // The data region holds only row bytes now, the header rides
                // in the slot
                let cost = t.data().len() + TupleSlot::SIZE;
                if free < cost {
                    break;
                }
                free -= cost;
                tuple_bytes_total += t.data().len();
                n_fit += 1;
            }
            if n_fit == 0 {
                return 0;
            }

            let slot_bytes_total = n_fit * TupleSlot::SIZE;
            let new_hdr = HeapPageHeader {
                slot_count: hdr.slot_count + n_fit as u16,
                free_space_start: hdr.free_space_start + slot_bytes_total as u16,
                free_space_end: hdr.free_space_end - tuple_bytes_total as u16,
                reserved: hdr.reserved,
            };
            let new_packed = new_hdr.to_u64();

            match header_atomic.compare_exchange_weak(
                packed,
                new_packed,
                Ordering::AcqRel,
                Ordering::Acquire,
            ) {
                Ok(_) => {
                    // Claim succeeded, write each tuple in the reserved range
                    let base_slot = hdr.slot_count;
                    let mut tuple_offset = hdr.free_space_end;
                    for (i, t) in tuples[..n_fit].iter().enumerate() {
                        // Warm a row further down the batch. Row bodies are
                        // separate allocations, but the Tuple structs holding
                        // their pointers are contiguous, so the address is
                        // known well before the copy needs it. Reaching past
                        // n_fit is deliberate, the rows after this page still
                        // get inserted
                        if let Some(ahead) = tuples.get(i + INSERT_PREFETCH_ROWS) {
                            prefetch_row(ahead);
                        }
                        let data = t.data();
                        let ts = data.len();
                        tuple_offset -= ts as u16;
                        unsafe {
                            std::ptr::copy_nonoverlapping(
                                data.as_ptr(),
                                page_ptr.add(tuple_offset as usize),
                                ts,
                            );

                            let slot_id = base_slot + i as u16;
                            let slot_addr =
                                page_ptr.add(Self::DATA_START + slot_id as usize * TupleSlot::SIZE);
                            // Everything above the publish word first, so a
                            // reader that sees the slot sees a whole header
                            let th = t.header();
                            // flags with the reserved half zeroed, then the two
                            // transaction ids as one word
                            (slot_addr.add(4) as *mut u32)
                                .write_unaligned((th.flags.0 as u32).to_le());
                            (slot_addr.add(8) as *mut u64).write_unaligned(
                                ((th.xmin as u64) | ((th.xmax as u64) << 32)).to_le(),
                            );

                            let slot_atomic = &*(slot_addr as *const AtomicU32);
                            let slot_packed = (tuple_offset as u32) | ((ts as u32) << 16);
                            slot_atomic.store(slot_packed, Ordering::Release);
                            results.push(TupleId::new(page_id, slot_id));
                        }
                    }
                    return n_fit;
                }
                Err(_) => {
                    std::hint::spin_loop();
                    continue;
                }
            }
        }
    }

    /// Returns the raw page data.
    pub fn as_bytes(&self) -> &[u8; PAGE_SIZE] {
        &self.data
    }

    /// Returns mutable raw page data.
    pub fn as_bytes_mut(&mut self) -> &mut [u8; PAGE_SIZE] {
        &mut self.data
    }

    /// Returns the heap header.
    fn heap_header(&self) -> HeapPageHeader {
        let offset = HeapPageHeader::OFFSET;
        HeapPageHeader::from_bytes(&self.data[offset..offset + HeapPageHeader::SIZE])
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
        Some(TupleSlot::from_bytes(
            &self.data[offset..offset + TupleSlot::SIZE],
        ))
    }

    /// Writes a slot to the slot array.
    fn set_slot(&mut self, slot_id: SlotId, slot: TupleSlot) {
        let offset = Self::slot_offset(slot_id);
        self.data[offset..offset + TupleSlot::SIZE].copy_from_slice(&slot.to_bytes());
    }

    /// Inserts a tuple into the page, reusing a slot a delete left behind
    /// when one exists. Returns Err(PageFull) when the page cannot hold it.
    pub fn insert_tuple(&mut self, tuple: &Tuple) -> Result<SlotId> {
        let data_len = tuple.data().len();
        let mut header = self.heap_header();

        let mut reuse: Option<SlotId> = None;
        for i in 0..header.slot_count {
            if let Some(slot) = self.get_slot(SlotId(i))
                && slot.is_empty()
            {
                reuse = Some(SlotId(i));
                break;
            }
        }

        let needed = data_len + if reuse.is_some() { 0 } else { TupleSlot::SIZE };
        if header.free_space() < needed {
            if Self::total_usable_space_in_slice(&*self.data) < needed {
                return Err(ZyronError::PageFull);
            }
            Self::compact_in_slice(&mut *self.data);
            header = self.heap_header();
        }

        header.free_space_end -= data_len as u16;
        let offset = header.free_space_end;
        let start = offset as usize;
        self.data[start..start + data_len].copy_from_slice(tuple.data());

        let slot_id = match reuse {
            Some(sid) => sid,
            None => {
                let sid = SlotId(header.slot_count);
                header.slot_count += 1;
                header.free_space_start += TupleSlot::SIZE as u16;
                sid
            }
        };
        Self::set_heap_header_in_slice(&mut *self.data, header);
        self.set_slot(slot_id, TupleSlot::new(offset, *tuple.header()));
        Ok(slot_id)
    }

    /// Reads a tuple from the page (allocates).
    pub fn get_tuple(&self, slot_id: SlotId) -> Option<Tuple> {
        let slot = self.get_slot(slot_id)?;

        if slot.is_empty() {
            return None;
        }

        let start = slot.offset as usize;
        let end = start + slot.data_len();
        if end > self.data.len() {
            return None;
        }
        Some(Tuple::with_header(
            slot.header,
            self.data[start..end].to_vec(),
        ))
    }

    /// Zero-copy tuple read. Borrows data from this page's buffer.
    #[inline]
    pub fn get_tuple_view(&self, slot_id: SlotId) -> Option<TupleView<'_>> {
        Self::get_tuple_view_from_slice(&*self.data, slot_id)
    }

    /// Stamps a tuple's `xmax` with the deleting transaction id, in place,
    /// without freeing the slot. This is the MVCC delete: the row stays
    /// physically present and is hidden by snapshot visibility once the deleting
    /// transaction commits, so an aborted delete leaves the row visible and
    /// vacuum reclaims the space later. Returns false if the slot is empty.
    pub fn set_tuple_xmax(&mut self, slot_id: SlotId, xmax: u32) -> bool {
        let Some(slot) = self.get_slot(slot_id) else {
            return false;
        };
        if slot.is_empty() {
            return false;
        }
        // xmax lives in the tuple slot
        let off = Self::slot_offset(slot_id) + TupleSlot::XMAX_OFFSET;
        self.data[off..off + 4].copy_from_slice(&xmax.to_le_bytes());
        true
    }

    /// Deletes a tuple from the page.
    ///
    /// This marks the slot as empty but doesn't reclaim space immediately.
    pub fn delete_tuple(&mut self, slot_id: SlotId) -> bool {
        if let Some(slot) = self.get_slot(slot_id)
            && !slot.is_empty()
        {
            // Mark slot as empty
            self.set_slot(slot_id, TupleSlot::empty());
            return true;
        }
        false
    }

    /// Updates a tuple in place if it fits, otherwise returns error.
    pub fn update_tuple(&mut self, slot_id: SlotId, tuple: &Tuple) -> Result<()> {
        let old_slot = self
            .get_slot(slot_id)
            .ok_or_else(|| ZyronError::TupleNotFound(format!("slot {} not found", slot_id)))?;

        if old_slot.is_empty() {
            return Err(ZyronError::TupleNotFound(format!(
                "slot {} is empty",
                slot_id
            )));
        }

        let new_size = tuple.data().len();
        let old_size = old_slot.data_len();

        // Only allow in-place update if new tuple fits in old space
        if new_size > old_size {
            return Err(ZyronError::PageFull);
        }

        // The data region holds only the row bytes, the header rides in the slot
        let start = old_slot.offset as usize;
        self.data[start..start + new_size].copy_from_slice(tuple.data());
        self.set_slot(slot_id, TupleSlot::new(old_slot.offset, *tuple.header()));

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
            if let Some(slot) = self.get_slot(SlotId(i))
                && slot.is_empty()
            {
                // Just need space for tuple data
                return header.free_space() >= tuple_size;
            }
        }

        // Need space for both slot and tuple
        header.free_space() >= tuple_size + TupleSlot::SIZE
    }

    /// Calculates space that can be reclaimed by compaction.
    /// This is the total space used by deleted tuple data.
    pub fn reclaimable_space(&self) -> usize {
        let header = self.heap_header();
        let mut active_tuple_space = 0usize;

        for i in 0..header.slot_count {
            if let Some(slot) = self.get_slot(SlotId(i))
                && !slot.is_empty()
            {
                active_tuple_space += slot.data_len();
            }
        }

        // Total tuple area = page end - free_space_end
        let tuple_area_size = (PAGE_SIZE as u16 - header.free_space_end) as usize;
        // Reclaimable = tuple area - active tuples
        tuple_area_size.saturating_sub(active_tuple_space)
    }

    /// Compacts the page by moving all active tuples together.
    /// Eliminates holes from deleted tuples and maximizes contiguous free space.
    pub fn compact(&mut self) {
        // In-place: offset-sorted `copy_within` packing, zero per-tuple
        // allocation. Shares the exact algorithm used by the inline append
        // path (`compact_in_slice`) so there is one compaction implementation.
        Self::compact_in_slice(&mut self.data[..]);
    }

    /// Returns total usable space including reclaimable space from deleted tuples.
    pub fn total_usable_space(&self) -> usize {
        self.free_space() + self.reclaimable_space()
    }
}

/// Iterator over tuples in a heap page (allocates per tuple).
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
        let slot = TupleSlot::new(100, TupleHeader::new(50, 7));
        let bytes = slot.to_bytes();
        let recovered = TupleSlot::from_bytes(&bytes);

        assert_eq!(recovered.offset, 100);
        assert_eq!(recovered.header.data_len, 50);
        assert_eq!(recovered.header.xmin, 7);
    }

    #[test]
    fn test_tuple_slot_empty() {
        let empty = TupleSlot::empty();
        assert!(empty.is_empty());

        let valid = TupleSlot::new(100, TupleHeader::new(50, 1));
        assert!(!valid.is_empty());
    }

    #[test]
    fn test_heap_page_new() {
        let page = create_test_page();

        assert_eq!(page.slot_count(), 0);
        assert!(page.free_space() > 0);
    }

    #[test]
    fn test_prune_dead_in_slice_preserves_live_slots() {
        let mut page = create_test_page();
        let s0 = page
            .insert_tuple(&Tuple::new(b"alpha".to_vec(), 10))
            .unwrap();
        let s1 = page
            .insert_tuple(&Tuple::new(b"bravo".to_vec(), 11))
            .unwrap();
        let s2 = page
            .insert_tuple(&Tuple::new(b"charlie".to_vec(), 12))
            .unwrap();

        // Mark the middle tuple deleted by a committed transaction below horizon.
        assert!(page.set_tuple_xmax(s1, 50));
        let free_before = page.free_space();

        let is_dead = |_xmin: u32, xmax: u32| xmax != 0 && xmax < 100;
        assert!(HeapPage::prune_dead_in_slice(page.as_bytes_mut(), &is_dead));

        // Survivors keep their slot ids and data; the pruned slot reads empty.
        assert_eq!(page.get_tuple(s0).unwrap().data(), b"alpha");
        assert!(page.get_tuple(s1).is_none());
        assert_eq!(page.get_tuple(s2).unwrap().data(), b"charlie");
        // Pruning reclaimed the middle tuple's bytes.
        assert!(page.free_space() > free_before);
    }

    #[test]
    fn test_prune_dead_in_slice_keeps_live_rows() {
        let mut page = create_test_page();
        let s0 = page
            .insert_tuple(&Tuple::new(b"keep".to_vec(), 10))
            .unwrap();
        // A live row (xmax == 0) and a delete above the horizon must survive.
        let s1 = page
            .insert_tuple(&Tuple::new(b"recent".to_vec(), 11))
            .unwrap();
        assert!(page.set_tuple_xmax(s1, 200));

        let is_dead = |_xmin: u32, xmax: u32| xmax != 0 && xmax < 100;
        assert!(!HeapPage::prune_dead_in_slice(
            page.as_bytes_mut(),
            &is_dead
        ));
        assert_eq!(page.get_tuple(s0).unwrap().data(), b"keep");
        assert_eq!(page.get_tuple(s1).unwrap().data(), b"recent");
    }

    #[test]
    fn test_vacuum_in_slice_clears_aborted_delete_and_reclaims_dead() {
        let mut page = create_test_page();
        // live: committed inserter, not deleted.
        let s_live = page
            .insert_tuple(&Tuple::new(b"live".to_vec(), 10))
            .unwrap();
        // aborted insert: inserter aborted -> dead.
        let s_ai = page
            .insert_tuple(&Tuple::new(b"abrt-ins".to_vec(), 20))
            .unwrap();
        // committed delete below horizon -> dead.
        let s_cd = page
            .insert_tuple(&Tuple::new(b"cmt-del".to_vec(), 11))
            .unwrap();
        assert!(page.set_tuple_xmax(s_cd, 12));
        // aborted delete: deleter aborted, row stays live, stamp must clear.
        let s_ad = page
            .insert_tuple(&Tuple::new(b"abrt-del".to_vec(), 13))
            .unwrap();
        assert!(page.set_tuple_xmax(s_ad, 21));

        let is_dead = |xmin: u32, xmax: u32| xmin == 20 || xmax == 12;
        let is_aborted = |xid: u32| xid == 20 || xid == 21;
        let (reclaimed, modified) =
            HeapPage::vacuum_in_slice(page.as_bytes_mut(), &is_dead, &is_aborted);

        assert!(modified);
        assert_eq!(reclaimed, 2); // aborted insert + committed delete
        // Dead rows are gone.
        assert!(page.get_tuple(s_ai).is_none());
        assert!(page.get_tuple(s_cd).is_none());
        // Live row intact.
        assert_eq!(page.get_tuple(s_live).unwrap().data(), b"live");
        // Aborted-delete row stays live and its xmax stamp is cleared.
        let ad = page.get_tuple(s_ad).unwrap();
        assert_eq!(ad.data(), b"abrt-del");
        assert_eq!(ad.header().xmax, 0);
    }

    #[test]
    fn test_heap_page_insert_tuple() {
        let mut page = create_test_page();
        let data = b"hello world".to_vec();
        let tuple = Tuple::new(data, 1);

        let slot_id = page.insert_tuple(&tuple).unwrap();
        assert_eq!(slot_id.0, 0);
        assert_eq!(page.slot_count(), 1);
    }

    #[test]
    fn test_heap_page_get_tuple() {
        let mut page = create_test_page();
        let data = b"test data".to_vec();
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
            let data = format!("tuple {}", i).into_bytes();
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
        let data = b"to be deleted".to_vec();
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
        let data1 = b"first".to_vec();
        let tuple1 = Tuple::new(data1, 1);
        let slot1 = page.insert_tuple(&tuple1).unwrap();
        page.delete_tuple(slot1);

        // Insert again - should reuse slot
        let data2 = b"second".to_vec();
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
        let data1 = vec![0u8; 100];
        let tuple1 = Tuple::new(data1, 1);
        let slot_id = page.insert_tuple(&tuple1).unwrap();

        // Update with smaller tuple (should succeed)
        let data2 = vec![1u8; 50];
        let tuple2 = Tuple::new(data2.clone(), 2);
        page.update_tuple(slot_id, &tuple2).unwrap();

        let retrieved = page.get_tuple(slot_id).unwrap();
        assert_eq!(retrieved.header().xmin, 2);
    }

    #[test]
    fn test_heap_page_update_too_large() {
        let mut page = create_test_page();

        // Insert a small tuple
        let data1 = vec![0u8; 10];
        let tuple1 = Tuple::new(data1, 1);
        let slot_id = page.insert_tuple(&tuple1).unwrap();

        // Try to update with larger tuple (should fail)
        let data2 = vec![1u8; 100];
        let tuple2 = Tuple::new(data2, 2);
        let result = page.update_tuple(slot_id, &tuple2);

        assert!(matches!(result, Err(ZyronError::PageFull)));
    }

    #[test]
    fn test_heap_page_iterator() {
        let mut page = create_test_page();

        for i in 0..5 {
            let data = format!("tuple {}", i).into_bytes();
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
            let data = vec![0u8; 1000 - TupleHeader::SIZE];
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
        let huge_data = vec![0u8; PAGE_SIZE];
        let huge_tuple = Tuple::new(huge_data, 1);
        let result = page.insert_tuple(&huge_tuple);

        assert!(matches!(result, Err(ZyronError::PageFull)));
    }

    #[test]
    fn test_heap_page_from_bytes() {
        let mut page = create_test_page();
        let data = b"persistent data".to_vec();
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
        let data = b"data".to_vec();
        let tuple = Tuple::new(data, 1);
        let slot_id = page.insert_tuple(&tuple).unwrap();

        assert!(page.delete_tuple(slot_id));
        assert!(!page.delete_tuple(slot_id)); // Already deleted
    }

    #[test]
    fn test_heap_page_compact() {
        let mut page = create_test_page();

        // Insert 3 tuples
        let data1 = vec![1u8; 1000];
        let data2 = vec![2u8; 1000];
        let data3 = vec![3u8; 1000];
        let slot1 = page.insert_tuple(&Tuple::new(data1.clone(), 1)).unwrap();
        let slot2 = page.insert_tuple(&Tuple::new(data2.clone(), 2)).unwrap();
        let slot3 = page.insert_tuple(&Tuple::new(data3.clone(), 3)).unwrap();

        let free_space_before = page.free_space();

        // Delete middle tuple
        page.delete_tuple(slot2);

        // Free space should not change (just slot marked empty)
        assert_eq!(page.free_space(), free_space_before);

        // But reclaimable space should be > 0
        assert!(page.reclaimable_space() > 0);

        // Compact the page
        page.compact();

        // After compaction, free space should increase by reclaimable amount
        assert!(page.free_space() > free_space_before);
        assert_eq!(page.reclaimable_space(), 0);

        // Remaining tuples should still be accessible
        let t1 = page.get_tuple(slot1).unwrap();
        let t3 = page.get_tuple(slot3).unwrap();
        assert_eq!(t1.data(), &data1);
        assert_eq!(t3.data(), &data3);

        // Deleted tuple should still be None
        assert!(page.get_tuple(slot2).is_none());
    }

    #[test]
    fn test_heap_page_compact_insert_after_delete() {
        let mut page = create_test_page();

        // Fill page with large tuples (2KB each)
        let tuple_size = 2000;
        let mut slots = Vec::new();
        while page.can_fit(tuple_size + TupleSlot::SIZE) {
            let data = vec![slots.len() as u8; tuple_size - TupleHeader::SIZE];
            let slot = page
                .insert_tuple(&Tuple::new(data, slots.len() as u32))
                .unwrap();
            slots.push(slot);
        }

        // Page should be nearly full
        let initial_slot_count = slots.len();
        assert!(initial_slot_count >= 3);

        // Delete all tuples
        for slot in &slots {
            page.delete_tuple(*slot);
        }

        // Free space is still small (data not reclaimed)
        let _free_space_after_delete = page.free_space();

        // But total usable space should be high
        let total_usable = page.total_usable_space();
        assert!(total_usable > tuple_size);

        // Insert should succeed (compaction happens automatically)
        let new_data = vec![99u8; tuple_size - TupleHeader::SIZE];
        let result = page.insert_tuple(&Tuple::new(new_data.clone(), 99));
        assert!(result.is_ok());

        // Verify the new tuple is accessible
        let new_slot = result.unwrap();
        let retrieved = page.get_tuple(new_slot).unwrap();
        assert_eq!(retrieved.data(), &new_data);
    }
}
