//! In-memory page storage for B+Tree nodes.

use std::cell::UnsafeCell;
use std::sync::atomic::{AtomicU64, Ordering};
use zyron_common::page::PAGE_SIZE;

/// Page slot combining data with a version stamp for optimistic reads.
///
/// Version protocol: even = stable (readable), odd = write in progress.
/// Readers snapshot the version before and after copying. If the versions
/// differ or either is odd, the read is retried.
struct PageSlot {
    version: AtomicU64,
    /// Raw pointer to a PAGE_SIZE byte array.
    /// For individually-allocated pages (owned=true), this points to a heap allocation.
    /// For arena-allocated pages (owned=false), this points into a PageArena.
    data: UnsafeCell<*mut [u8; PAGE_SIZE]>,
    /// True if this slot individually owns its allocation (must free on drop).
    owned: bool,
}

// PageSlot is Send + Sync because all concurrent access goes through
// the version stamp protocol (writers hold &mut Self or bump version
// to odd before writing). UnsafeCell is only accessed through
// try_versioned_write / try_read which enforce the correct ordering.
unsafe impl Send for PageSlot {}
unsafe impl Sync for PageSlot {}

impl PageSlot {
    fn new() -> Self {
        let ptr = Box::into_raw(Box::new([0u8; PAGE_SIZE]));
        Self {
            version: AtomicU64::new(0),
            data: UnsafeCell::new(ptr),
            owned: true,
        }
    }
}

impl Drop for PageSlot {
    fn drop(&mut self) {
        if self.owned {
            unsafe { drop(Box::from_raw(*self.data.get_mut())); }
        }
        // Arena-allocated pages: memory is freed by InMemoryPageStore::drop.
    }
}

/// Bulk memory arena for checkpoint-loaded pages.
/// Holds a single contiguous allocation containing multiple PAGE_SIZE pages.
struct PageArena {
    ptr: *mut u8,
    layout: std::alloc::Layout,
}

unsafe impl Send for PageArena {}
unsafe impl Sync for PageArena {}

impl Drop for PageArena {
    fn drop(&mut self) {
        unsafe { std::alloc::dealloc(self.ptr, self.layout); }
    }
}

/// In-memory page storage for B+Tree nodes.
///
/// All pages are stored in RAM in a Vec. Page numbers map directly to Vec indices.
/// This eliminates disk I/O overhead and BufferPool lock contention from the hot path.
///
/// Each page has a version stamp for optimistic lock-free reads. Writers bump
/// the version to odd before writing and to even after. Readers that observe
/// a version change or an odd version retry their read. The existing get()/get_mut()
/// methods remain for callers holding an external RwLock during structural modifications
/// (node splits, merges).
pub struct InMemoryPageStore {
    /// Pages stored by page number (index = page_num).
    pages: Vec<PageSlot>,
    /// Bulk memory arenas for checkpoint-loaded pages.
    arenas: Vec<PageArena>,
}

impl InMemoryPageStore {
    /// Creates a new empty page store.
    pub fn new() -> Self {
        Self { pages: Vec::new(), arenas: Vec::new() }
    }
    /// Allocates a new page and returns its page number.
    #[inline]
    pub fn allocate(&mut self) -> u32 {
        let page_num = self.pages.len() as u32;
        self.pages.push(PageSlot::new());
        page_num
    }

    /// Gets a page by page number (read-only).
    /// Caller must hold an external read lock. No version stamp check.
    #[inline]
    pub fn get(&self, page_num: u32) -> Option<&[u8; PAGE_SIZE]> {
        self.pages
            .get(page_num as usize)
            .map(|slot| unsafe { &**slot.data.get() })
    }

    /// Gets a mutable page by page number.
    /// Caller must hold an external write lock. No version stamp check.
    #[inline]
    pub fn get_mut(&mut self, page_num: u32) -> Option<&mut [u8; PAGE_SIZE]> {
        self.pages
            .get_mut(page_num as usize)
            .map(|slot| unsafe { &mut **slot.data.get_mut() })
    }

    /// Writes page data at a specific page number (caller holds &mut self).
    #[inline]
    pub fn write(&mut self, page_num: u32, data: &[u8; PAGE_SIZE]) {
        if let Some(slot) = self.pages.get_mut(page_num as usize) {
            unsafe { (**slot.data.get_mut()).copy_from_slice(data); }
            let v = slot.version.load(Ordering::Relaxed);
            slot.version.store(v + 2, Ordering::Release);
        }
    }

    /// Optimistic lock-free read. Returns the page data if the version was
    /// stable throughout the copy. Returns None if the page does not exist.
    /// Returns Err(()) if the read was torn (version changed during copy).
    ///
    /// Callers should retry on Err(()) in a loop.
    #[inline]
    pub fn try_read(&self, page_num: u32) -> Option<Result<[u8; PAGE_SIZE], ()>> {
        let slot = self.pages.get(page_num as usize)?;

        let v1 = slot.version.load(Ordering::Acquire);
        if v1 & 1 != 0 {
            return Some(Err(()));
        }

        let data = unsafe { **slot.data.get() };

        let v2 = slot.version.load(Ordering::Acquire);
        if v1 != v2 {
            return Some(Err(()));
        }

        Some(Ok(data))
    }

    /// Optimistic lock-free read that also returns the validated version.
    /// Combines version check + data copy into a single operation,
    /// eliminating the TOCTOU gap between separate version check and read.
    #[inline]
    pub fn try_read_versioned(&self, page_num: u32) -> Option<Result<([u8; PAGE_SIZE], u64), ()>> {
        let slot = self.pages.get(page_num as usize)?;

        let v1 = slot.version.load(Ordering::Acquire);
        if v1 & 1 != 0 {
            return Some(Err(()));
        }

        let data = unsafe { **slot.data.get() };

        let v2 = slot.version.load(Ordering::Acquire);
        if v1 != v2 {
            return Some(Err(()));
        }

        Some(Ok((data, v1)))
    }

    /// CAS-based write: only succeeds if the page version matches `expected_version`.
    #[inline]
    pub fn try_versioned_write(
        &self,
        page_num: u32,
        data: &[u8; PAGE_SIZE],
        expected_version: u64,
    ) -> bool {
        let Some(slot) = self.pages.get(page_num as usize) else {
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
            (**slot.data.get()).copy_from_slice(data);
        }

        slot.version
            .store(expected_version + 2, Ordering::Release);
        true
    }

    /// Pre-allocates `count` pages from a single contiguous arena allocation.
    /// Returns the starting page number.
    /// Uses one large allocation instead of per-page allocator calls.
    /// Memory is zeroed to pre-fault OS virtual memory pages.
    pub fn bulk_allocate(&mut self, count: usize) -> u32 {
        let start = self.pages.len() as u32;
        if count == 0 { return start; }

        let total_bytes = count * PAGE_SIZE;
        let layout = std::alloc::Layout::from_size_align(total_bytes, 16).unwrap();
        let base = unsafe { std::alloc::alloc_zeroed(layout) };
        if base.is_null() {
            std::alloc::handle_alloc_error(layout);
        }

        self.arenas.push(PageArena { ptr: base, layout });

        // Create PageSlots pointing directly into the arena.
        // No Box wrapper needed. The arena owns the memory and frees it on drop.
        self.pages.reserve(count);
        for i in 0..count {
            let page_ptr = unsafe { base.add(i * PAGE_SIZE) as *mut [u8; PAGE_SIZE] };
            self.pages.push(PageSlot {
                version: AtomicU64::new(0),
                data: UnsafeCell::new(page_ptr),
                owned: false,
            });
        }
        start
    }

}
