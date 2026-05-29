//! Lock-free in-memory page storage for B+Tree nodes.
//!
//! Pages are organized into fixed-size chunks installed lazily. Each
//! page number maps directly to a (chunk_idx, slot_idx) pair so reads,
//! in-slice writes, and allocations proceed without any global lock.
//!
//! Concurrency primitives per page:
//!  - version: even = stable readable, odd = exclusive writer in progress.
//!    Readers snapshot version before+after data copy, mismatch or odd =
//!    retry. CAS writers CAS even to odd, copy data, store version+2.
//!  - lock_for_write returns a WriteGuard that keeps the page version
//!    odd for the duration of a multi-step structural operation (node
//!    split). While the guard is held, all CAS writers and readers on
//!    the locked page retry, so the structural mutation is observed
//!    atomically by everyone else.

use parking_lot::Mutex;
use std::cell::UnsafeCell;
use std::sync::atomic::{AtomicPtr, AtomicU32, AtomicU64, Ordering};
use zyron_common::page::PAGE_SIZE;

// 1024 slots per chunk, 8KB per slot, 8MB per chunk
const SLOTS_PER_CHUNK: usize = 1024;
const SLOT_INDEX_MASK: usize = SLOTS_PER_CHUNK - 1;
const CHUNK_INDEX_SHIFT: usize = 10;

// 16K chunks, supports 16M pages = 128GB of index data
// 16K * 8 bytes = 128KB outer pointer array, allocated once at startup
const MAX_CHUNKS: usize = 16384;

struct PageSlot {
    version: AtomicU64,
    data: UnsafeCell<[u8; PAGE_SIZE]>,
}

// All concurrent access goes through the version protocol or the split
// latch, UnsafeCell access is gated by these atomic operations
unsafe impl Send for PageSlot {}
unsafe impl Sync for PageSlot {}

#[repr(C)]
struct Chunk {
    slots: [PageSlot; SLOTS_PER_CHUNK],
}

impl Chunk {
    fn new_zeroed() -> *mut Chunk {
        // The all-zero bit pattern is a valid PageSlot, AtomicU64::new(0)
        // is 0, AtomicU32::new(0) is 0, [0u8; PAGE_SIZE] is the zero page
        // We alloc_zeroed directly to avoid 1024 individual PageSlot inits
        let layout = std::alloc::Layout::new::<Chunk>();
        let raw = unsafe { std::alloc::alloc_zeroed(layout) as *mut Chunk };
        if raw.is_null() {
            std::alloc::handle_alloc_error(layout);
        }
        raw
    }

    unsafe fn dealloc(raw: *mut Chunk) {
        let layout = std::alloc::Layout::new::<Chunk>();
        unsafe { std::alloc::dealloc(raw as *mut u8, layout) }
    }
}

pub struct InMemoryPageStore {
    chunks: Box<[AtomicPtr<Chunk>]>,
    next_page: AtomicU32,
    install_lock: Mutex<()>,
}

impl InMemoryPageStore {
    pub fn new() -> Self {
        let mut v = Vec::with_capacity(MAX_CHUNKS);
        for _ in 0..MAX_CHUNKS {
            v.push(AtomicPtr::new(std::ptr::null_mut()));
        }
        Self {
            chunks: v.into_boxed_slice(),
            next_page: AtomicU32::new(0),
            install_lock: Mutex::new(()),
        }
    }

    #[inline]
    fn chunk_and_slot(page_num: u32) -> (usize, usize) {
        let p = page_num as usize;
        (p >> CHUNK_INDEX_SHIFT, p & SLOT_INDEX_MASK)
    }

    // Loads or lazily installs the chunk, returns its pointer
    // Fast path is a single Acquire load, slow path serializes on
    // install_lock to avoid duplicate allocations on first touch
    #[inline]
    fn ensure_chunk(&self, chunk_idx: usize) -> *mut Chunk {
        let slot = &self.chunks[chunk_idx];
        let p = slot.load(Ordering::Acquire);
        if !p.is_null() {
            return p;
        }
        let _g = self.install_lock.lock();
        let p2 = slot.load(Ordering::Acquire);
        if !p2.is_null() {
            return p2;
        }
        let raw = Chunk::new_zeroed();
        slot.store(raw, Ordering::Release);
        raw
    }

    #[inline]
    fn slot(&self, page_num: u32) -> &PageSlot {
        let (ci, si) = Self::chunk_and_slot(page_num);
        debug_assert!(ci < MAX_CHUNKS, "page_num {} exceeds capacity", page_num);
        let p = self.ensure_chunk(ci);
        // SAFETY: ensure_chunk returns a valid Chunk pointer that lives
        // for the lifetime of self, chunks are only freed in Drop
        unsafe { &(*p).slots[si] }
    }

    #[inline]
    fn slot_if_installed(&self, page_num: u32) -> Option<&PageSlot> {
        let (ci, si) = Self::chunk_and_slot(page_num);
        if ci >= MAX_CHUNKS {
            return None;
        }
        let p = self.chunks[ci].load(Ordering::Acquire);
        if p.is_null() {
            return None;
        }
        unsafe { Some(&(*p).slots[si]) }
    }

    /// Allocates a new page and returns its page number. Lock-free.
    #[inline]
    pub fn allocate(&self) -> u32 {
        let p = self.next_page.fetch_add(1, Ordering::Relaxed);
        // Touch the chunk so future accesses do not race the install
        let _ = self.slot(p);
        p
    }

    /// Bulk-allocates `count` consecutive pages, returns the first page
    /// number. Used by checkpoint loading for arena-style initialization.
    pub fn bulk_allocate(&self, count: usize) -> u32 {
        if count == 0 {
            return self.next_page.load(Ordering::Relaxed);
        }
        let start = self.next_page.fetch_add(count as u32, Ordering::Relaxed);
        let end = start.saturating_add(count as u32);
        let (start_ci, _) = Self::chunk_and_slot(start);
        let last_ci = Self::chunk_and_slot(end.saturating_sub(1)).0;
        for ci in start_ci..=last_ci {
            self.ensure_chunk(ci);
        }
        start
    }

    /// Optimistic lock-free read. Returns None if the page has never been
    /// touched (chunk not installed). Returns Err(()) if the read tore due
    /// to a concurrent writer, caller should retry.
    #[inline]
    pub fn try_read(&self, page_num: u32) -> Option<Result<[u8; PAGE_SIZE], ()>> {
        let slot = self.slot_if_installed(page_num)?;
        let v1 = slot.version.load(Ordering::Acquire);
        if v1 & 1 != 0 {
            return Some(Err(()));
        }
        let data = unsafe { *slot.data.get() };
        let v2 = slot.version.load(Ordering::Acquire);
        if v1 != v2 {
            return Some(Err(()));
        }
        Some(Ok(data))
    }

    /// Same as try_read but also returns the validated version for use
    /// with try_versioned_write.
    #[inline]
    pub fn try_read_versioned(&self, page_num: u32) -> Option<Result<([u8; PAGE_SIZE], u64), ()>> {
        let slot = self.slot_if_installed(page_num)?;
        let v1 = slot.version.load(Ordering::Acquire);
        if v1 & 1 != 0 {
            return Some(Err(()));
        }
        let data = unsafe { *slot.data.get() };
        let v2 = slot.version.load(Ordering::Acquire);
        if v1 != v2 {
            return Some(Err(()));
        }
        Some(Ok((data, v1)))
    }

    /// Reads a page until it is stable, blocking via short spins on torn reads.
    /// Returns None only if the chunk has not been installed.
    #[inline]
    pub fn read_stable(&self, page_num: u32) -> Option<[u8; PAGE_SIZE]> {
        let slot = self.slot_if_installed(page_num)?;
        loop {
            let v1 = slot.version.load(Ordering::Acquire);
            if v1 & 1 != 0 {
                std::hint::spin_loop();
                continue;
            }
            let data = unsafe { *slot.data.get() };
            let v2 = slot.version.load(Ordering::Acquire);
            if v1 == v2 {
                return Some(data);
            }
            std::hint::spin_loop();
        }
    }

    /// CAS-write. Succeeds only if the page version equals expected_version,
    /// meaning no concurrent writer has modified the page since the read.
    #[inline]
    pub fn try_versioned_write(
        &self,
        page_num: u32,
        data: &[u8; PAGE_SIZE],
        expected_version: u64,
    ) -> bool {
        let Some(slot) = self.slot_if_installed(page_num) else {
            return false;
        };
        let odd = expected_version | 1;
        if slot
            .version
            .compare_exchange(expected_version, odd, Ordering::AcqRel, Ordering::Relaxed)
            .is_err()
        {
            return false;
        }
        unsafe {
            (*slot.data.get()).copy_from_slice(data);
        }
        slot.version.store(expected_version + 2, Ordering::Release);
        true
    }

    /// Force-write that ignores the current version. Caller must hold the
    /// split_latch on this page so no concurrent split races. CAS writers
    /// and readers using the version protocol still proceed safely because
    /// this method bumps the version even-odd-even like any other writer.
    pub fn force_write(&self, page_num: u32, data: &[u8; PAGE_SIZE]) {
        let slot = self.slot(page_num);
        loop {
            let v = slot.version.load(Ordering::Acquire);
            if v & 1 != 0 {
                std::hint::spin_loop();
                continue;
            }
            if slot
                .version
                .compare_exchange(v, v | 1, Ordering::AcqRel, Ordering::Relaxed)
                .is_ok()
            {
                unsafe {
                    (*slot.data.get()).copy_from_slice(data);
                }
                slot.version.store(v + 2, Ordering::Release);
                return;
            }
        }
    }

    /// Acquires exclusive write access to a page by CAS-ing the version
    /// from even to odd, spinning until acquired. The returned guard holds
    /// version odd until dropped, so all readers and CAS writers retry
    /// while the multi-step structural operation runs.
    #[inline]
    pub fn lock_for_write(&self, page_num: u32) -> WriteGuard<'_> {
        let slot = self.slot(page_num);
        loop {
            let v = slot.version.load(Ordering::Acquire);
            if v & 1 != 0 {
                std::hint::spin_loop();
                continue;
            }
            if slot
                .version
                .compare_exchange(v, v | 1, Ordering::AcqRel, Ordering::Relaxed)
                .is_ok()
            {
                return WriteGuard {
                    slot,
                    base_version: v,
                };
            }
            std::hint::spin_loop();
        }
    }

    // ----- exclusive-access methods, single-threaded init / recovery -----

    /// Returns a reference to the raw page bytes. Caller must guarantee no
    /// concurrent writers, typically during single-threaded initialization
    /// or recovery. Returns None if the chunk has not been installed.
    pub fn get(&self, page_num: u32) -> Option<&[u8; PAGE_SIZE]> {
        let slot = self.slot_if_installed(page_num)?;
        Some(unsafe { &*slot.data.get() })
    }

    /// Returns a mutable reference to the page bytes. Caller must hold
    /// &mut self, used only by checkpoint loading paths.
    pub fn get_mut(&mut self, page_num: u32) -> Option<&mut [u8; PAGE_SIZE]> {
        let slot = self.slot_if_installed(page_num)?;
        // SAFETY: &mut self guarantees no other thread holds a reference
        // to any part of this store
        Some(unsafe { &mut *slot.data.get() })
    }

    /// Single-threaded write helper used by initialization and checkpoint
    /// loading. Bumps version to an even value so any future concurrent
    /// reader sees the new data.
    pub fn write(&mut self, page_num: u32, data: &[u8; PAGE_SIZE]) {
        let slot = self.slot(page_num);
        unsafe {
            (*slot.data.get()).copy_from_slice(data);
        }
        let v = slot.version.load(Ordering::Relaxed);
        let next = (v & !1) + 2;
        slot.version.store(next, Ordering::Release);
    }
}

impl Default for InMemoryPageStore {
    fn default() -> Self {
        Self::new()
    }
}

impl Drop for InMemoryPageStore {
    fn drop(&mut self) {
        for slot in self.chunks.iter() {
            let p = slot.load(Ordering::Relaxed);
            if !p.is_null() {
                unsafe { Chunk::dealloc(p) };
            }
        }
    }
}

pub struct WriteGuard<'a> {
    slot: &'a PageSlot,
    base_version: u64,
}

impl<'a> WriteGuard<'a> {
    /// Reads the current page bytes while the guard is held. Safe because
    /// the guard's odd-version state excludes other writers.
    #[inline]
    pub fn read(&self) -> [u8; PAGE_SIZE] {
        unsafe { *self.slot.data.get() }
    }

    /// Writes new bytes into the page while the guard is held. The data
    /// becomes visible to readers only when the guard drops and the
    /// version transitions back to even.
    #[inline]
    pub fn write(&self, data: &[u8; PAGE_SIZE]) {
        unsafe {
            (*self.slot.data.get()).copy_from_slice(data);
        }
    }
}

impl<'a> Drop for WriteGuard<'a> {
    fn drop(&mut self) {
        self.slot
            .version
            .store(self.base_version + 2, Ordering::Release);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn allocate_assigns_sequential_page_numbers() {
        let s = InMemoryPageStore::new();
        assert_eq!(s.allocate(), 0);
        assert_eq!(s.allocate(), 1);
        assert_eq!(s.allocate(), 2);
    }

    #[test]
    fn try_read_returns_none_for_unallocated() {
        let s = InMemoryPageStore::new();
        assert!(s.try_read(0).is_none());
    }

    #[test]
    fn try_read_returns_data_after_write() {
        let mut s = InMemoryPageStore::new();
        let p = s.allocate();
        let mut data = [0u8; PAGE_SIZE];
        data[0] = 0xAB;
        data[42] = 0xCD;
        s.write(p, &data);
        let read = s.try_read(p).unwrap().unwrap();
        assert_eq!(read[0], 0xAB);
        assert_eq!(read[42], 0xCD);
    }

    #[test]
    fn try_versioned_write_succeeds_on_matching_version() {
        let mut s = InMemoryPageStore::new();
        let p = s.allocate();
        s.write(p, &[0u8; PAGE_SIZE]);
        let (data, v) = s.try_read_versioned(p).unwrap().unwrap();
        let mut next = data;
        next[0] = 1;
        assert!(s.try_versioned_write(p, &next, v));
    }

    #[test]
    fn try_versioned_write_fails_on_stale_version() {
        let mut s = InMemoryPageStore::new();
        let p = s.allocate();
        s.write(p, &[0u8; PAGE_SIZE]);
        let (_, v) = s.try_read_versioned(p).unwrap().unwrap();
        let mut next = [0u8; PAGE_SIZE];
        next[0] = 1;
        s.force_write(p, &next);
        let mut other = [0u8; PAGE_SIZE];
        other[0] = 2;
        assert!(!s.try_versioned_write(p, &other, v));
    }

    #[test]
    fn bulk_allocate_returns_first_and_assigns_sequential() {
        let s = InMemoryPageStore::new();
        let first = s.bulk_allocate(5);
        assert_eq!(first, 0);
        assert_eq!(s.allocate(), 5);
    }

    #[test]
    fn lock_for_write_is_exclusive() {
        use std::sync::Arc;
        use std::sync::atomic::AtomicU32;
        let s = Arc::new(InMemoryPageStore::new());
        let p = s.allocate();
        let counter = Arc::new(AtomicU32::new(0));
        let mut handles = Vec::new();
        for _ in 0..4 {
            let s = Arc::clone(&s);
            let counter = Arc::clone(&counter);
            handles.push(std::thread::spawn(move || {
                for _ in 0..1000 {
                    let _g = s.lock_for_write(p);
                    let v = counter.load(Ordering::Relaxed);
                    counter.store(v + 1, Ordering::Relaxed);
                }
            }));
        }
        for h in handles {
            h.join().unwrap();
        }
        assert_eq!(counter.load(Ordering::Relaxed), 4000);
    }
}
