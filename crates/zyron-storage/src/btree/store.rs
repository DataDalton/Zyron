//! Lock-free in-memory page storage for B+Tree nodes.
//!
//! Pages are organized into fixed-size chunks installed lazily. Each
//! page number maps directly to a (chunk_idx, slot_idx) pair so reads,
//! writes, and allocations proceed without any global lock.
//!
//! Concurrency model (RCU / copy-on-write pointer swap):
//!  - Each slot holds an atomic pointer to an immutable page buffer. A
//!    published buffer is never mutated in place, so readers load the
//!    pointer once and read the buffer they got with no version check and
//!    no retry. Reads are wait-free.
//!  - Writers build a new buffer, apply their change, and swap the slot
//!    pointer to it. The old buffer is retired through epoch-based
//!    reclamation so a reader that still holds it is never freed under.
//!  - The per-slot version counter (even = stable, odd = a writer is
//!    publishing) serializes writers against each other only. Readers
//!    ignore it. A CAS writer publishes by claiming the version even->odd,
//!    swapping the pointer, then storing version+2. A second writer that
//!    sees a changed version retries; readers are never affected.
//!  - lock_for_write claims the version for a multi-step structural
//!    operation (node split) and publishes the result via pointer swap
//!    when the guard drops. Other writers retry while it is held, readers
//!    keep reading the pre-split buffer.

use crossbeam::epoch::{self, Atomic, Owned};
use crossbeam::queue::ArrayQueue;
use parking_lot::Mutex;
use std::cell::Cell;
use std::sync::OnceLock;
use std::sync::atomic::{AtomicPtr, AtomicU32, AtomicU64, Ordering};
use zyron_common::page::PAGE_SIZE;

// Recycled page-buffer pool. A publish allocates a fresh 8KB buffer and the
// old buffer is retired; rather than round-tripping the allocator on every
// write, retired buffers are pushed back here (after the epoch grace period,
// so no reader still holds them) and reused. All buffers are PAGE_SIZE, so the
// pool is shared process-wide across every index. Capacity bounds the recycled
// memory; a push that overflows frees the buffer instead.
const BUFFER_POOL_CAP: usize = 8192; // 8192 * 8KB = 64MB ceiling

// Raw page-buffer pointer made Send so it can cross into the epoch-deferred
// recycle closure. The buffer is unreachable (already swapped out) and the
// epoch grace period guarantees no reader holds it when the closure runs.
struct SendPtr(*mut [u8; PAGE_SIZE]);
unsafe impl Send for SendPtr {}

fn buffer_pool() -> &'static ArrayQueue<SendPtr> {
    static POOL: OnceLock<ArrayQueue<SendPtr>> = OnceLock::new();
    POOL.get_or_init(|| ArrayQueue::new(BUFFER_POOL_CAP))
}

/// Takes a page buffer from the pool, or allocates one if the pool is empty.
/// Contents are undefined, the caller fully overwrites before publishing.
#[inline]
fn take_buffer() -> *mut [u8; PAGE_SIZE] {
    match buffer_pool().pop() {
        Some(SendPtr(raw)) => raw,
        None => Box::into_raw(Box::new([0u8; PAGE_SIZE])),
    }
}

/// Returns a retired page buffer to the pool, or frees it if the pool is full.
#[inline]
fn return_buffer(raw: *mut [u8; PAGE_SIZE]) {
    if buffer_pool().push(SendPtr(raw)).is_err() {
        // SAFETY: raw came from Box::into_raw and is no longer referenced.
        unsafe { drop(Box::from_raw(raw)) };
    }
}

// 1024 slots per chunk, 8KB per slot, 8MB per chunk
const SLOTS_PER_CHUNK: usize = 1024;
const SLOT_INDEX_MASK: usize = SLOTS_PER_CHUNK - 1;
const CHUNK_INDEX_SHIFT: usize = 10;

// 16K chunks, supports 16M pages = 128GB of index data
// 16K * 8 bytes = 128KB outer pointer array, allocated once at startup
const MAX_CHUNKS: usize = 16384;

struct PageSlot {
    // even = stable, odd = a writer is publishing. Serializes writers only.
    version: AtomicU64,
    // Pointer to the current immutable page buffer, null until first write.
    page: Atomic<[u8; PAGE_SIZE]>,
}

#[repr(C)]
struct Chunk {
    slots: [PageSlot; SLOTS_PER_CHUNK],
}

impl Chunk {
    fn new_zeroed() -> *mut Chunk {
        // The all-zero bit pattern is a valid Chunk: AtomicU64::new(0) is 0
        // and Atomic::null() is the null pointer (zero bits). We alloc_zeroed
        // directly to avoid 1024 individual PageSlot inits.
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
        // Touch the chunk so future accesses do not race the install. The
        // slot's page pointer stays null until the first write.
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

    /// Wait-free read. Borrows the current immutable page buffer for as long
    /// as `guard` stays pinned. Returns None if the page has never been
    /// written. One atomic load and a null check, with no loop, no retry and
    /// no copy, so it completes in a bounded number of steps whatever other
    /// threads are doing.
    ///
    /// A writer never mutates a published buffer, it installs a new one and
    /// retires the old through epoch reclamation, and that reclamation cannot
    /// run while `guard` is pinned. A traversal therefore pins once and
    /// borrows every page it visits rather than copying PAGE_SIZE bytes per
    /// level.
    #[inline]
    pub fn page_ref<'g>(
        &'g self,
        page_num: u32,
        guard: &'g epoch::Guard,
    ) -> Option<&'g [u8; PAGE_SIZE]> {
        let slot = self.slot_if_installed(page_num)?;
        let shared = slot.page.load(Ordering::Acquire, guard);
        if shared.is_null() {
            return None;
        }
        // SAFETY: the buffer is immutable once published, and it is retired
        // only after a grace period that cannot elapse while `guard` is
        // pinned, so the borrow stays valid for 'g
        Some(unsafe { shared.deref() })
    }

    /// Read for the writer CAS protocol: returns the buffer plus the version
    /// token to pass to try_versioned_write. Validates the version around the
    /// copy so the (data, version) pair is consistent for a read-modify-write.
    /// Returns Err(()) if a writer is mid-publish so the calling writer
    /// retries. Readers should use page_ref, which never retries.
    #[inline]
    pub fn try_read_versioned(&self, page_num: u32) -> Option<Result<([u8; PAGE_SIZE], u64), ()>> {
        let slot = self.slot_if_installed(page_num)?;
        let v1 = slot.version.load(Ordering::Acquire);
        if v1 & 1 != 0 {
            return Some(Err(()));
        }
        let guard = epoch::pin();
        let shared = slot.page.load(Ordering::Acquire, &guard);
        if shared.is_null() {
            return None;
        }
        let data = unsafe { *shared.deref() };
        let v2 = slot.version.load(Ordering::Acquire);
        if v1 != v2 {
            return Some(Err(()));
        }
        Some(Ok((data, v1)))
    }

    /// CAS-publish. Succeeds only if the page version still equals
    /// expected_version, meaning no concurrent writer has published since the
    /// read. Builds a new immutable buffer and swaps it in, retiring the old.
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
        // Claim the version even->odd. Other writers retry while odd.
        if slot
            .version
            .compare_exchange(
                expected_version,
                expected_version | 1,
                Ordering::AcqRel,
                Ordering::Relaxed,
            )
            .is_err()
        {
            return false;
        }
        self.publish(slot, data);
        slot.version.store(expected_version + 2, Ordering::Release);
        true
    }

    /// Unconditional publish. Caller is the only writer of this page for the
    /// duration (freshly allocated split sibling, recovery), but readers may
    /// be reading the old buffer, so the swap still goes through epoch
    /// reclamation. Spins only against other writers via the version claim.
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
                self.publish(slot, data);
                slot.version.store(v + 2, Ordering::Release);
                return;
            }
            std::hint::spin_loop();
        }
    }

    /// Swaps in a new immutable buffer holding `data` and retires the old one
    /// through epoch reclamation. The caller must hold the version claim
    /// (odd) so two writers cannot swap concurrently.
    #[inline]
    fn publish(&self, slot: &PageSlot, data: &[u8; PAGE_SIZE]) {
        let raw = take_buffer();
        // SAFETY: take_buffer returns an owned buffer we fully initialize.
        unsafe { *raw = *data };
        self.publish_raw(slot, raw);
    }

    /// Swaps in `raw` (an owned, fully-initialized page buffer) as the new
    /// immutable page and retires the old one to the pool after the epoch
    /// grace period. The caller must hold the version claim.
    #[inline]
    fn publish_raw(&self, slot: &PageSlot, raw: *mut [u8; PAGE_SIZE]) {
        let new = unsafe { Owned::from_raw(raw) };
        let guard = epoch::pin();
        let old = slot.page.swap(new, Ordering::AcqRel, &guard);
        if !old.is_null() {
            let old_raw = old.as_raw() as *mut [u8; PAGE_SIZE];
            // SAFETY: the old buffer is immutable and no longer reachable from
            // the slot. The deferred recycle runs only after every reader that
            // could hold it has unpinned, so reuse cannot race a reader. The
            // closure touches only the 'static pool and the unreachable
            // buffer, so deferring it unchecked is sound.
            unsafe { guard.defer_unchecked(move || return_buffer(old_raw)) };
        }
    }

    /// Loads the current page content into a fresh working buffer from the
    /// pool, or a zeroed buffer if the page has never been written. The caller
    /// owns the returned raw buffer and must publish or return it.
    #[inline]
    fn load_working(slot: &PageSlot) -> *mut [u8; PAGE_SIZE] {
        let raw = take_buffer();
        let guard = epoch::pin();
        let shared = slot.page.load(Ordering::Acquire, &guard);
        // SAFETY: raw is owned by this caller; fully initialize it.
        unsafe {
            if shared.is_null() {
                *raw = [0u8; PAGE_SIZE];
            } else {
                *raw = *shared.deref();
            }
        }
        raw
    }

    /// Attempts to acquire the write claim only if the current (even) version
    /// equals `expected`. Returns the guard on success, None without touching
    /// the page on mismatch or while another writer holds it. Used by
    /// prepare-then-publish split: the split halves are built off-claim
    /// against a versioned snapshot and committed only when the leaf has not
    /// changed since, so the publish is a single pointer swap on guard drop.
    #[inline]
    pub fn try_lock_for_write_at(&self, page_num: u32, expected: u64) -> Option<WriteGuard<'_>> {
        let slot = self.slot(page_num);
        if slot
            .version
            .compare_exchange(expected, expected | 1, Ordering::AcqRel, Ordering::Relaxed)
            .is_ok()
        {
            Some(WriteGuard {
                store: self,
                slot,
                base_version: expected,
                working: Self::load_working(slot),
                dirty: Cell::new(false),
            })
        } else {
            None
        }
    }

    /// Claims the write version for a multi-step structural operation,
    /// spinning until acquired. Other writers retry while the claim is held;
    /// readers keep reading the current buffer. The guard publishes the
    /// mutated buffer via pointer swap when it drops.
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
                    store: self,
                    slot,
                    base_version: v,
                    working: Self::load_working(slot),
                    dirty: Cell::new(false),
                };
            }
            std::hint::spin_loop();
        }
    }

    // ----- exclusive-access methods, single-threaded init / recovery -----

    /// Returns a reference to the raw page bytes. Caller must guarantee no
    /// concurrent writers, typically during single-threaded initialization
    /// or recovery. Returns None if the page has never been written.
    pub fn get(&self, page_num: u32) -> Option<&[u8; PAGE_SIZE]> {
        let slot = self.slot_if_installed(page_num)?;
        let guard = epoch::pin();
        let shared = slot.page.load(Ordering::Acquire, &guard);
        if shared.is_null() {
            return None;
        }
        // SAFETY: exclusive-access contract, the buffer outlives this borrow
        // because no concurrent writer swaps it.
        Some(unsafe { &*shared.as_raw() })
    }

    /// Returns a mutable reference to the page bytes, installing a zeroed
    /// buffer if the page has never been written. Caller must hold &mut self,
    /// used only by checkpoint loading paths.
    pub fn get_mut(&mut self, page_num: u32) -> Option<&mut [u8; PAGE_SIZE]> {
        let slot = self.slot(page_num);
        let guard = epoch::pin();
        let shared = slot.page.load(Ordering::Acquire, &guard);
        let raw = if shared.is_null() {
            let owned = Owned::new([0u8; PAGE_SIZE]);
            let installed = owned.into_shared(&guard);
            slot.page.store(installed, Ordering::Release);
            installed.as_raw() as *mut [u8; PAGE_SIZE]
        } else {
            shared.as_raw() as *mut [u8; PAGE_SIZE]
        };
        // SAFETY: &mut self guarantees no other thread accesses this store.
        Some(unsafe { &mut *raw })
    }

    /// Single-threaded write helper used by initialization and checkpoint
    /// loading. Publishes a new buffer and bumps the version even.
    pub fn write(&mut self, page_num: u32, data: &[u8; PAGE_SIZE]) {
        let slot = self.slot(page_num);
        // &mut self: no concurrent readers, free the old buffer immediately.
        let guard = epoch::pin();
        let new = Owned::new(*data);
        let old = slot.page.swap(new, Ordering::AcqRel, &guard);
        if !old.is_null() {
            unsafe { drop(old.into_owned()) };
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
        // No concurrent access during Drop: free each page buffer, then the
        // chunk arrays.
        let guard = epoch::pin();
        for chunk_slot in self.chunks.iter() {
            let p = chunk_slot.load(Ordering::Relaxed);
            if p.is_null() {
                continue;
            }
            for si in 0..SLOTS_PER_CHUNK {
                let page = unsafe { &(*p).slots[si].page };
                let shared = page.load(Ordering::Relaxed, &guard);
                if !shared.is_null() {
                    unsafe { drop(shared.into_owned()) };
                }
            }
            unsafe { Chunk::dealloc(p) };
        }
    }
}

/// Exclusive write claim over a single page for a multi-step structural
/// operation. Mutations accumulate in a private working buffer and are
/// published as one immutable buffer when the guard drops, so readers never
/// observe an intermediate state and never retry.
pub struct WriteGuard<'a> {
    store: &'a InMemoryPageStore,
    slot: &'a PageSlot,
    base_version: u64,
    // Pooled working buffer, owned by this guard. Mutations accumulate here
    // and publish as one immutable page when the guard drops.
    working: *mut [u8; PAGE_SIZE],
    dirty: Cell<bool>,
}

impl<'a> WriteGuard<'a> {
    /// Reads the current working bytes.
    #[inline]
    pub fn read(&self) -> [u8; PAGE_SIZE] {
        // SAFETY: the guard exclusively owns the working buffer for its
        // lifetime, so there is no aliasing access.
        unsafe { *self.working }
    }

    /// Writes new bytes into the working buffer. The data becomes visible to
    /// readers only when the guard drops and publishes the buffer.
    #[inline]
    pub fn write(&self, data: &[u8; PAGE_SIZE]) {
        // SAFETY: exclusive ownership for the guard's lifetime.
        unsafe {
            (*self.working).copy_from_slice(data);
        }
        self.dirty.set(true);
    }
}

impl<'a> Drop for WriteGuard<'a> {
    fn drop(&mut self) {
        if self.dirty.get() {
            // Publish the working buffer directly as the new immutable page,
            // retiring the old one to the pool.
            self.store.publish_raw(self.slot, self.working);
        } else {
            // Unchanged: return the working buffer to the pool unused.
            return_buffer(self.working);
        }
        // Release the write claim back to even. Readers never observed odd.
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
    fn page_ref_returns_none_for_unallocated() {
        let s = InMemoryPageStore::new();
        let guard = epoch::pin();
        assert!(s.page_ref(0, &guard).is_none());
    }

    #[test]
    fn page_ref_returns_data_after_write() {
        let mut s = InMemoryPageStore::new();
        let p = s.allocate();
        let mut data = [0u8; PAGE_SIZE];
        data[0] = 0xAB;
        data[42] = 0xCD;
        s.write(p, &data);
        let guard = epoch::pin();
        let read = s.page_ref(p, &guard).unwrap();
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
    fn lock_for_write_publishes_on_drop() {
        let s = InMemoryPageStore::new();
        let p = s.allocate();
        {
            let g = s.lock_for_write(p);
            let mut data = g.read();
            data[7] = 0x5A;
            g.write(&data);
        }
        let guard = epoch::pin();
        let read = s.page_ref(p, &guard).unwrap();
        assert_eq!(read[7], 0x5A);
    }

    #[test]
    fn lock_for_write_is_exclusive() {
        use std::sync::Arc;
        let s = Arc::new(InMemoryPageStore::new());
        let p = s.allocate();
        s.force_write(p, &[0u8; PAGE_SIZE]);
        let mut handles = Vec::new();
        for _ in 0..4 {
            let s = Arc::clone(&s);
            handles.push(std::thread::spawn(move || {
                for _ in 0..1000 {
                    let g = s.lock_for_write(p);
                    let mut data = g.read();
                    let v = u32::from_le_bytes([data[0], data[1], data[2], data[3]]);
                    data[0..4].copy_from_slice(&(v + 1).to_le_bytes());
                    g.write(&data);
                }
            }));
        }
        for h in handles {
            h.join().unwrap();
        }
        let guard = epoch::pin();
        let final_data = s.page_ref(p, &guard).unwrap();
        let count =
            u32::from_le_bytes([final_data[0], final_data[1], final_data[2], final_data[3]]);
        assert_eq!(count, 4000);
    }
}
