//! Buffer pool manager.

use crate::frame::{BufferFrame, FrameId};
use crate::page_table::{InsertOutcome, PageTable};
use crate::replacer::{ClockReplacer, Replacer};
use std::sync::OnceLock;
use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};
use sysinfo::System;
use zyron_common::page::{PAGE_SIZE, PageId};
use zyron_common::{Result, ZyronError};
use zyron_pressure::hot_set::PrefetchReport;

/// Fraction of the pool held back from a prefetch, as a divisor of the frame
/// count.
///
/// A survivor absorbing a departing node's traffic needs frames for the
/// queries it is already serving. Sixteenths leaves that headroom without
/// making the handover pointless on a pool that is mostly free.
const PREFETCH_RESERVE_DIVISOR: usize = 16;

/// Write callback invoked to flush a dirty victim during eviction.
/// Writes the page durably to disk so a dirty frame is never reused while its
/// contents are unwritten. Wired from the pool construction site.
/// The buffer is the pool's private copy of the frame, passed mutably so the
/// writer can stamp integrity fields in place instead of copying the page.
/// The third argument is the page's dirty LSN, so the writer can hold the
/// write until the WAL is durable up to it: a data page must never reach
/// disk ahead of the log records that produced it.
pub type EvictWriteFn =
    std::sync::Arc<dyn Fn(PageId, &mut [u8; PAGE_SIZE], u64) -> Result<()> + Send + Sync>;

/// What a flush_all callback did with the page it was offered. A callback
/// that filters by file id answers Skipped for a page it does not own, and
/// the pool keeps that page's dirty state instead of marking an unwritten
/// page clean
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FlushOutcome {
    /// The page was written durably, the frame may stay clean
    Written,
    /// The callback does not own the page and did not write it
    Skipped,
}

// ---------------------------------------------------------------------------
// Lock-free Treiber stack for buffer pool frame allocation
// ---------------------------------------------------------------------------

/// Sentinel value indicating end of stack (no next frame).
const TREIBER_NULL: u32 = u32::MAX;

/// Lock-free Treiber stack for buffer pool frame allocation.
///
/// Uses an array-based approach where each slot corresponds to a FrameId.
/// The value at each slot is the next FrameId in the stack chain.
/// ABA prevention uses a generation counter packed into the head pointer.
struct TreiberFreeList {
    /// Per-frame next pointers. next[frame_id] = next frame in stack.
    next: Box<[AtomicU32]>,
    /// Stack head: upper 32 bits = generation, lower 32 bits = frame_id.
    /// TREIBER_NULL in lower bits means empty stack.
    head: AtomicU64,
}

impl TreiberFreeList {
    /// Creates a new Treiber stack with all frames pushed (frame 0 at bottom).
    fn new(num_frames: usize) -> Self {
        let next: Box<[AtomicU32]> = (0..num_frames)
            .map(|i| {
                if i == 0 {
                    AtomicU32::new(TREIBER_NULL) // Bottom of stack
                } else {
                    AtomicU32::new((i - 1) as u32) // Points to previous frame
                }
            })
            .collect();

        // Top of stack is the last frame
        let top = if num_frames > 0 {
            (num_frames - 1) as u32
        } else {
            TREIBER_NULL
        };
        let head = AtomicU64::new(Self::pack(0, top));

        Self { next, head }
    }

    #[inline(always)]
    fn pack(generation: u32, frame_id: u32) -> u64 {
        ((generation as u64) << 32) | (frame_id as u64)
    }

    #[inline(always)]
    fn unpack(packed: u64) -> (u32, u32) {
        ((packed >> 32) as u32, packed as u32)
    }

    /// Pops a frame from the stack. Returns None if empty.
    #[inline]
    fn pop(&self) -> Option<FrameId> {
        loop {
            let current = self.head.load(Ordering::Acquire);
            let (generation, top) = Self::unpack(current);

            if top == TREIBER_NULL {
                return None;
            }

            let next_top = self.next[top as usize].load(Ordering::Acquire);
            let new_head = Self::pack(generation.wrapping_add(1), next_top);

            match self.head.compare_exchange_weak(
                current,
                new_head,
                Ordering::AcqRel,
                Ordering::Relaxed,
            ) {
                Ok(_) => return Some(FrameId(top)),
                Err(_) => std::hint::spin_loop(),
            }
        }
    }

    /// Pushes a frame onto the stack.
    #[inline]
    fn push(&self, frame_id: FrameId) {
        loop {
            let current = self.head.load(Ordering::Acquire);
            let (generation, top) = Self::unpack(current);

            self.next[frame_id.0 as usize].store(top, Ordering::Release);
            let new_head = Self::pack(generation.wrapping_add(1), frame_id.0);

            match self.head.compare_exchange_weak(
                current,
                new_head,
                Ordering::AcqRel,
                Ordering::Relaxed,
            ) {
                Ok(_) => return,
                Err(_) => std::hint::spin_loop(),
            }
        }
    }

    /// Pops up to `count` frames in a single CAS operation.
    /// Walks the chain to find the Nth node, then atomically swings the head
    /// past all N nodes. Falls back to per-element pop on CAS failure.
    #[inline]
    fn pop_many(&self, count: usize) -> Vec<FrameId> {
        if count == 0 {
            return Vec::new();
        }
        loop {
            let current = self.head.load(Ordering::Acquire);
            let (generation, top) = Self::unpack(current);
            if top == TREIBER_NULL {
                return Vec::new();
            }

            // Walk the chain to collect up to `count` frame IDs.
            let mut collected = Vec::with_capacity(count);
            let mut cursor = top;
            for _ in 0..count {
                if cursor == TREIBER_NULL {
                    break;
                }
                collected.push(FrameId(cursor));
                cursor = self.next[cursor as usize].load(Ordering::Acquire);
            }

            // CAS head from current top to the node after the last collected.
            let new_head = Self::pack(generation.wrapping_add(1), cursor);
            match self.head.compare_exchange_weak(
                current,
                new_head,
                Ordering::AcqRel,
                Ordering::Relaxed,
            ) {
                Ok(_) => return collected,
                Err(_) => std::hint::spin_loop(),
            }
        }
    }

    /// Pushes multiple frames onto the stack.
    #[inline]
    fn push_many(&self, frame_ids: &[FrameId]) {
        for &fid in frame_ids {
            self.push(fid);
        }
    }

    /// Returns approximate count by traversing the stack.
    /// Not linearizable but sufficient for stats reporting.
    fn approximate_count(&self) -> usize {
        let mut count = 0;
        let (_, mut current) = Self::unpack(self.head.load(Ordering::Relaxed));
        while current != TREIBER_NULL && count < self.next.len() {
            count += 1;
            current = self.next[current as usize].load(Ordering::Relaxed);
        }
        count
    }
}

/// Configuration for the buffer pool.
#[derive(Debug, Clone)]
pub struct BufferPoolConfig {
    /// Number of frames in the pool.
    pub num_frames: usize,
}

impl Default for BufferPoolConfig {
    fn default() -> Self {
        Self { num_frames: 1024 }
    }
}

/// Buffer pool manager.
///
/// Manages a fixed-size pool of page frames with:
/// - Page ID to frame ID mapping (lock-free page table)
/// - Free frame list for new pages
/// - Clock replacement for eviction
/// - Pin counting for concurrent access
pub struct BufferPool {
    /// Configuration.
    config: BufferPoolConfig,
    /// Array of buffer frames.
    frames: Vec<BufferFrame>,
    /// Page ID to frame ID mapping (lock-free reads).
    page_table: PageTable,
    /// Lock-free stack of free frame IDs.
    free_list: TreiberFreeList,
    /// Page replacement policy.
    replacer: ClockReplacer,
    /// Write hook that flushes a dirty victim during eviction. A dirty page
    /// leaves the pool only through this hook, which writes it before its
    /// mapping comes down. A pool without one cannot evict a dirty page and
    /// sweeps past it instead
    evict_writer: OnceLock<EvictWriteFn>,
    /// Serializes flusher-side page writes (background trickle, checkpoint
    /// flush, full flush) so two flushers can never write one page's images
    /// to disk in opposite order. Eviction does not take it: a claimed
    /// victim is unreachable to every flusher through the failed pin.
    flush_serial: parking_lot::Mutex<()>,
}

impl BufferPool {
    /// Creates a new buffer pool.
    pub fn new(config: BufferPoolConfig) -> Self {
        let num_frames = config.num_frames;

        // Initialize frames
        let frames: Vec<_> = (0..num_frames)
            .map(|i| BufferFrame::new(FrameId(i as u32)))
            .collect();

        Self {
            config,
            frames,
            page_table: PageTable::new(num_frames),
            free_list: TreiberFreeList::new(num_frames),
            replacer: ClockReplacer::new(num_frames),
            evict_writer: OnceLock::new(),
            flush_serial: parking_lot::Mutex::new(()),
        }
    }

    /// Installs the write hook used to flush a dirty victim during eviction.
    ///
    /// Set once from the pool construction site, and required by any pool
    /// whose working set outgrows its frames. Eviction writes the page
    /// through this callback while it still holds the frame, so the page is
    /// on disk before its page table mapping comes down and a reader
    /// faulting it back in cannot find a stale image. Without a hook there is
    /// nowhere to put the bytes, so a dirty frame is never taken as a victim
    /// and a pool of entirely dirty frames reports itself full.
    ///
    /// Returns an error if a hook was already installed.
    pub fn set_evict_writer(&self, writer: EvictWriteFn) -> Result<()> {
        self.evict_writer
            .set(writer)
            .map_err(|_| ZyronError::Internal("evict writer already set".to_string()))
    }

    /// Creates a buffer pool sized to 25% of available system RAM.
    ///
    /// Queries the system for available memory and allocates 25% of it
    /// for the buffer pool. Minimum 1,000 frames to ensure useful caching
    /// even on low-memory systems. No upper limit - systems with terabytes
    /// of RAM can use it all.
    ///
    /// For a system with 16GB RAM, this allocates ~4GB (~250k frames).
    pub fn auto_sized() -> Self {
        let mut sys = System::new_all();
        sys.refresh_memory();

        let available_bytes = sys.available_memory() as usize;
        let target_bytes = available_bytes / 4; // 25% of available RAM
        let num_frames = (target_bytes / PAGE_SIZE).max(1_000);

        Self::new(BufferPoolConfig { num_frames })
    }

    /// Returns the number of frames in the pool.
    pub fn num_frames(&self) -> usize {
        self.config.num_frames
    }

    /// Returns the approximate number of free frames.
    pub fn free_count(&self) -> usize {
        self.free_list.approximate_count()
    }

    /// Returns the number of pages currently in the pool.
    pub fn page_count(&self) -> usize {
        self.page_table.len()
    }

    /// Checks if a page is in the buffer pool.
    pub fn contains(&self, page_id: PageId) -> bool {
        self.page_table.contains(page_id)
    }

    /// Fetches a page from the buffer pool.
    ///
    /// If the page is not in the pool, returns None.
    /// The page is pinned before being returned.
    ///
    /// The pin is claim-aware: an eviction claims its victim before
    /// removing the mapping, so a lookup can hand back a frame that is
    /// already being replaced. A pin refused by the claim retries the
    /// lookup until the mapping disappears, and a pin that lands after the
    /// frame changed tenants is detected by re-reading the frame's page id
    /// under the pin and released.
    #[inline(always)]
    pub fn fetch_page(&self, page_id: PageId) -> Option<&BufferFrame> {
        let mut round: u32 = 0;
        loop {
            let frame_id = self.page_table.get(page_id)?;
            let frame = &self.frames[frame_id.0 as usize];
            if !frame.try_pin() {
                // claimed mid eviction, the mapping is on its way out
                retry_pause(&mut round);
                #[cfg(debug_assertions)]
                stall_diagnostic(round, "fetch_page pin refused", page_id, frame);
                continue;
            }
            if frame.page_id() == Some(page_id) {
                // Record access for clock algorithm (sets reference bit)
                self.replacer.record_access(frame_id);
                return Some(frame);
            }
            frame.unpin();
            retry_pause(&mut round);
            #[cfg(debug_assertions)]
            stall_diagnostic(round, "fetch_page tenant mismatch", page_id, frame);
        }
    }

    /// Allocates a frame for a new page.
    ///
    /// Tries to get a free frame first, then evicts if necessary. A dirty
    /// victim is written to disk through the evict-writer hook before its
    /// frame is reused, and a pool with no hook does not take a dirty frame
    /// at all.
    /// Returns a frame owned exclusively by this caller, already pinned once.
    ///
    /// Both paths hand back a claimed frame so the caller never has to
    /// re-acquire it: a free-list frame is pinned here, and an evicted one is
    /// claimed inside the sweep. Handing back an unpinned frame left a window
    /// in which a second sweep could take it.
    fn allocate_frame(&self) -> Result<FrameId> {
        // Try free list first (lock-free pop). The pop owns the frame id,
        // but an eviction sweep may hold a transient claim on the frame
        // while discovering it has no tenant, so ownership is taken with
        // the same claim CAS the sweep uses. A blind pin here could land
        // between the sweep's claim and its tenant check, and the setup's
        // page id would then make the sweep keep the frame as a victim:
        // two owners, one buffer
        if let Some(frame_id) = self.free_list.pop() {
            let frame = &self.frames[frame_id.0 as usize];
            let mut round: u32 = 0;
            while !frame.try_claim() {
                retry_pause(&mut round);
            }
            return Ok(frame_id);
        }

        // Claim the victim in the same step that finds it unpinned, so a
        // concurrent sweep cannot choose the same frame. The compare-exchange
        // is Acquire on success, so a concurrent pin (also Acquire) orders
        // before this decision and a page a thread is pinning is never taken.
        // A frame with no page belongs to the free list, claiming it here
        // would give it two owners, so it is skipped.
        //
        // A dirty page leaves the pool only by being written, and only the
        // hook can write it. Taking one without a hook would drop its mapping
        // while its bytes were still nowhere but memory, and the next reader
        // to fault the page in would install the stale disk image over them
        let write_through = self.evict_writer.get();
        let victim_id = self.replacer.evict(|fid| {
            let frame = &self.frames[fid.0 as usize];
            if !frame.try_claim() {
                return false;
            }
            if frame.page_id().is_none() {
                frame.unclaim();
                return false;
            }
            if write_through.is_none() && frame.is_dirty() {
                frame.unclaim();
                return false;
            }
            true
        });

        if let Some(victim_id) = victim_id {
            let frame = &self.frames[victim_id.0 as usize];

            // Write a dirty victim before reusing its frame, and before its
            // mapping comes down, so the page is on disk for whoever faults
            // it back in. The sweep above never offers a dirty frame without
            // a hook, so a missing one here is a broken invariant rather than
            // a page to hand away
            if frame.is_dirty()
                && let Some(page_id) = frame.page_id()
            {
                let mut data = Box::new([0u8; PAGE_SIZE]);
                let dirty_lsn = frame.dirty_lsn();
                let data_guard = frame.read_data();
                data.copy_from_slice(&**data_guard);
                drop(data_guard);

                let Some(write) = write_through else {
                    frame.unclaim();
                    self.replacer.record_access(victim_id);
                    return Err(ZyronError::Internal(
                        "a dirty frame was taken as a victim with no write hook".to_string(),
                    ));
                };
                // On failure the frame is released back to the pool, dirty
                // and in the page table, and the error surfaces so the dirty
                // page is not silently lost. Leaving it claimed would hang
                // every later reader of that page on a claim that nothing
                // will ever drop
                if let Err(e) = write(page_id, &mut data, dirty_lsn) {
                    frame.unclaim();
                    self.replacer.record_access(victim_id);
                    return Err(e);
                }
                frame.set_dirty(false);
            }

            // Remove old page from page table
            if let Some(old_page_id) = frame.page_id() {
                self.page_table.remove(old_page_id);
            }

            return Ok(victim_id);
        }

        Err(ZyronError::BufferPoolFull)
    }

    /// Inserts a new page into the buffer pool.
    ///
    /// If the page already exists, returns the existing frame.
    /// The page is pinned before being returned.
    ///
    /// A dirty page displaced to make room is written by the pool itself, so
    /// the caller has nothing to flush.
    #[inline]
    pub fn new_page(&self, page_id: PageId) -> Result<&BufferFrame> {
        self.new_page_inner(page_id, None).map(|(frame, _)| frame)
    }

    /// Like `new_page` but reports whether this call installed the frame.
    /// A caller that initializes a page in place must do so only on a fresh
    /// install, because an existing frame already holds live content.
    #[inline]
    pub fn new_page_reporting_fresh(&self, page_id: PageId) -> Result<(&BufferFrame, bool)> {
        self.new_page_inner(page_id, None)
    }

    /// Like `new_page` but reports whether the returned frame was freshly
    /// installed for this id. A frame that already held the page carries
    /// content as new or newer than any disk image, so `init` is copied in
    /// only on a fresh install, and before the mapping publishes: once the
    /// page table names this frame a concurrent fetch may pin it, and it
    /// must never observe the zeroed frame a later copy would fill
    fn new_page_inner(&self, page_id: PageId, init: Option<&[u8]>) -> Result<(&BufferFrame, bool)> {
        let mut round: u32 = 0;
        loop {
            // Check if page already exists
            if let Some(frame_id) = self.page_table.get(page_id) {
                let frame = &self.frames[frame_id.0 as usize];
                if !frame.try_pin() {
                    // claimed mid eviction, the mapping is on its way out
                    retry_pause(&mut round);
                    #[cfg(debug_assertions)]
                    stall_diagnostic(round, "new_page pin refused", page_id, frame);
                    continue;
                }
                if frame.page_id() == Some(page_id) {
                    self.replacer.record_access(frame_id);
                    return Ok((frame, false));
                }
                frame.unpin();
                retry_pause(&mut round);
                #[cfg(debug_assertions)]
                stall_diagnostic(round, "new_page tenant mismatch", page_id, frame);
                continue;
            }

            // Allocate a frame
            let frame_id = self.allocate_frame()?;

            // Set up the frame. allocate_frame hands it back claimed or
            // pinned, and that ownership is what keeps a concurrent
            // eviction sweep off it, so the reset must not clear it
            let frame = &self.frames[frame_id.0 as usize];
            frame.reset_keeping_pin();
            if let Some(data) = init {
                frame.copy_from(data);
            }
            frame.set_page_id(Some(page_id));
            // An eviction claim converts to a normal pin only after the
            // frame carries its new identity, so a stale reader that pins
            // in the gap sees the changed page id and retreats. A frame
            // from the free list already holds a plain pin
            frame.claim_to_pin_if_claimed();

            // Publish the mapping atomically. A concurrent new_page for the
            // same id can race here, insert_if_absent resolves both to a
            // single winner.
            match self.page_table.insert_if_absent(page_id, frame_id) {
                InsertOutcome::Inserted => return Ok((frame, true)),
                InsertOutcome::Existing(winner_id) => {
                    // Another caller already installed a frame for this id.
                    // The identity clears before the unpin, so an eviction
                    // sweep that takes the freed frame cannot remove the
                    // winner's mapping through a stale page id
                    frame.set_page_id(None);
                    frame.unpin();
                    self.free_list.push(frame_id);

                    let winner = &self.frames[winner_id.0 as usize];
                    if winner.try_pin() {
                        if winner.page_id() == Some(page_id) {
                            self.replacer.record_access(winner_id);
                            return Ok((winner, false));
                        }
                        winner.unpin();
                    }
                    // the winner is already being replaced, retry from the
                    // table lookup
                    retry_pause(&mut round);
                    continue;
                }
                InsertOutcome::TableFull => {
                    // Page table is full. Release the frame instead of
                    // leaking it and surface the failure.
                    frame.set_page_id(None);
                    frame.unpin();
                    self.free_list.push(frame_id);
                    return Err(ZyronError::BufferPoolFull);
                }
            }
        }
    }

    /// Loads page data into the buffer pool.
    ///
    /// This is used when reading a page from disk. A dirty page displaced to
    /// make room is written by the pool itself.
    #[inline]
    pub fn load_page(&self, page_id: PageId, data: &[u8]) -> Result<&BufferFrame> {
        // Only a freshly installed frame takes the caller's bytes, and it
        // takes them before its mapping publishes. A frame that already
        // held the page, whether found directly or through a lost install
        // race, holds content as new or newer than the disk image, and
        // copying the stale bytes over it would erase committed writes and
        // later flush them durably
        let (frame, _fresh) = self.new_page_inner(page_id, Some(data))?;
        Ok(frame)
    }

    /// Unpins a page in the buffer pool.
    ///
    /// If the page becomes unpinned (pin count = 0), it becomes evictable.
    /// Evictability is determined by pin_count during eviction, not tracked separately.
    #[inline]
    pub fn unpin_page(&self, page_id: PageId, is_dirty: bool) -> bool {
        if let Some(frame_id) = self.page_table.get(page_id) {
            let frame = &self.frames[frame_id.0 as usize];

            if is_dirty {
                frame.set_dirty(true);
            }

            frame.unpin();
            // No need to update replacer - evict() checks pin_count directly
            return true;
        }
        false
    }

    /// Flushes a page to the provided callback.
    ///
    /// The callback receives the page data if the page is dirty.
    /// Returns true if the page was flushed.
    ///
    /// The dirty state is captured and cleared before the copy, so a write
    /// landing while the flush is in flight re-marks the frame and stays
    /// discoverable instead of being wiped clean by a post-flush clear. A
    /// failed flush restores the state so the page is retried.
    pub fn flush_page<F>(&self, page_id: PageId, mut flush_fn: F) -> Result<bool>
    where
        F: FnMut(PageId, &mut [u8]) -> Result<()>,
    {
        let Some(frame_id) = self.page_table.get(page_id) else {
            return Ok(false);
        };
        let frame = &self.frames[frame_id.0 as usize];
        if !frame.try_pin() {
            // claimed by an eviction, whose own write-through covers it
            return Ok(false);
        }
        if frame.page_id() != Some(page_id) {
            frame.unpin();
            return Ok(false);
        }
        let flush_order = self.flush_serial.lock();
        if !frame.is_dirty() {
            drop(flush_order);
            frame.unpin();
            return Ok(false);
        }
        let expected_lsn = frame.dirty_lsn();
        frame.set_dirty(false);
        let _ = frame.clear_dirty_lsn(expected_lsn);
        let mut data: Box<[u8; PAGE_SIZE]> = {
            let guard = frame.read_data();
            Box::new(**guard)
        };
        let outcome = flush_fn(page_id, &mut data[..]);
        drop(flush_order);
        match outcome {
            Ok(()) => {
                frame.unpin();
                Ok(true)
            }
            Err(e) => {
                // restore so the page stays discoverable and is retried
                frame.set_dirty(true);
                frame.set_dirty_lsn(expected_lsn);
                frame.unpin();
                Err(e)
            }
        }
    }

    /// Flushes all dirty pages.
    ///
    /// Attempts every dirty page even if some flushes fail. Pages that fail stay
    /// dirty so a later flush retries them. Returns the number of pages flushed
    /// on full success. On partial failure returns an error reporting how many
    /// pages failed and how many succeeded, plus the first underlying error, so
    /// the caller knows the flush was incomplete.
    pub fn flush_all<F>(&self, mut flush_fn: F) -> Result<usize>
    where
        F: FnMut(PageId, &mut [u8]) -> Result<FlushOutcome>,
    {
        let mut flushed = 0;
        let mut failed = 0;
        let mut first_error: Option<ZyronError> = None;

        // Collect dirty pages first to avoid holding guards during flush
        let mut dirty_pages = Vec::new();
        self.page_table.for_each(|page_id, frame_id| {
            dirty_pages.push((page_id, frame_id));
            true // continue iteration
        });

        let flush_order = self.flush_serial.lock();
        for (page_id, frame_id) in dirty_pages {
            let frame = &self.frames[frame_id.0 as usize];
            if !frame.try_pin() {
                // claimed by an eviction, whose own write-through covers it
                continue;
            }
            if frame.page_id() != Some(page_id) || !frame.is_dirty() {
                frame.unpin();
                continue;
            }
            // Capture and clear the dirty state before the copy: a write
            // landing during the flush re-marks the frame instead of being
            // wiped clean by a clear that runs after the write
            let expected_lsn = frame.dirty_lsn();
            frame.set_dirty(false);
            let _ = frame.clear_dirty_lsn(expected_lsn);
            let mut data: Box<[u8; PAGE_SIZE]> = {
                let guard = frame.read_data();
                Box::new(**guard)
            };
            match flush_fn(page_id, &mut data[..]) {
                Ok(FlushOutcome::Written) => {
                    frame.unpin();
                    flushed += 1;
                }
                Ok(FlushOutcome::Skipped) => {
                    // The callback does not own this page and wrote nothing.
                    // The dirty state comes back so the owner's own flush
                    // still sees the page, a clean-looking frame here would
                    // let eviction discard the only current copy
                    frame.set_dirty(true);
                    frame.set_dirty_lsn(expected_lsn);
                    frame.unpin();
                }
                Err(e) => {
                    // Restore so the page is retried, record the error and
                    // keep flushing the rest.
                    frame.set_dirty(true);
                    frame.set_dirty_lsn(expected_lsn);
                    frame.unpin();
                    failed += 1;
                    if first_error.is_none() {
                        first_error = Some(e);
                    }
                }
            }
        }
        drop(flush_order);

        match first_error {
            Some(e) => Err(ZyronError::Internal(format!(
                "flush_all incomplete: {} flushed, {} failed, first error: {}",
                flushed, failed, e
            ))),
            None => Ok(flushed),
        }
    }

    /// Deletes a page from the buffer pool.
    ///
    /// Returns true if the page was deleted.
    /// Returns false if the page is pinned or not in the pool.
    ///
    /// The claim comes first: it excludes readers, eviction sweeps and a
    /// second delete for the whole removal, so the mapping never has to be
    /// re-inserted (the re-insert could clobber a concurrent fault-in) and
    /// the reset can never wipe a pin a reader took in between.
    pub fn delete_page(&self, page_id: PageId) -> bool {
        let Some(frame_id) = self.page_table.get(page_id) else {
            return false;
        };
        let frame = &self.frames[frame_id.0 as usize];
        if !frame.try_claim() {
            // pinned by a reader or owned by an eviction, refuse
            return false;
        }
        // The frame may have changed tenants between the lookup and the
        // claim, in which case it belongs to another page now
        if frame.page_id() != Some(page_id) || self.page_table.get(page_id) != Some(frame_id) {
            frame.unclaim();
            return false;
        }
        self.page_table.remove(page_id);
        self.replacer.remove(frame_id);
        frame.reset();
        self.free_list.push(frame_id);
        true
    }

    /// Returns a read guard for page data.
    pub fn read_page(&self, page_id: PageId) -> Option<PageReadGuard<'_>> {
        let frame = self.fetch_page(page_id)?;
        Some(PageReadGuard {
            pool: self,
            page_id,
            frame,
        })
    }

    /// Returns raw pointer to frame data for a pinned page without acquiring RwLock.
    ///
    /// # Safety
    /// Caller must ensure:
    /// - Page is pinned before calling and stays pinned during use
    /// - No concurrent writers exist
    /// - Pointer is not dereferenced after unpin
    #[inline(always)]
    pub unsafe fn frame_data_ptr(&self, page_id: PageId) -> Option<*const [u8; PAGE_SIZE]> {
        let frame_id = self.page_table.get(page_id)?;
        let frame = &self.frames[frame_id.0 as usize];
        Some(unsafe { frame.data_ptr() })
    }

    /// Returns mutable raw pointer to frame data for a pinned page.
    ///
    /// # Safety
    /// - Page must be pinned before calling and stay pinned during use
    /// - Caller must ensure exclusive write access (no concurrent readers/writers)
    /// - Pointer must not be dereferenced after unpin
    #[inline(always)]
    pub unsafe fn frame_data_ptr_mut(&self, page_id: PageId) -> Option<*mut [u8; PAGE_SIZE]> {
        let frame_id = self.page_table.get(page_id)?;
        let frame = &self.frames[frame_id.0 as usize];
        Some(unsafe { frame.data_ptr_mut() })
    }

    /// Pins multiple pages at once for batch read operations.
    ///
    /// Returns one flag per requested page, true where a resident frame
    /// was pinned. A page with no frame is the caller's to load, and only
    /// flagged pages may be unpinned afterwards, an unflagged unpin would
    /// steal a pin another thread holds on a later fault-in.
    /// Use with `batch_unpin` over exactly the flagged pages.
    #[inline]
    pub fn batch_pin(&self, page_ids: &[PageId]) -> Vec<bool> {
        let mut pinned = Vec::with_capacity(page_ids.len());
        for &pid in page_ids {
            let mut got = false;
            if let Some(frame_id) = self.page_table.get(pid) {
                let frame = &self.frames[frame_id.0 as usize];
                // A frame claimed by an eviction or re-tenanted since the
                // lookup counts as absent, the caller loads the page instead
                if frame.try_pin() {
                    if frame.page_id() == Some(pid) {
                        self.replacer.record_access(frame_id);
                        got = true;
                    } else {
                        frame.unpin();
                    }
                }
            }
            pinned.push(got);
        }
        pinned
    }

    /// Unpins multiple pages at once after batch read operations.
    #[inline]
    pub fn batch_unpin(&self, page_ids: &[PageId]) {
        for &pid in page_ids {
            if let Some(frame_id) = self.page_table.get(pid) {
                self.frames[frame_id.0 as usize].unpin();
            }
        }
    }

    /// Unpins multiple pages and marks them dirty for batch write operations.
    #[inline]
    pub fn batch_unpin_dirty(&self, page_ids: &[PageId]) {
        for &pid in page_ids {
            self.unpin_page(pid, true);
        }
    }

    /// Allocates frames for multiple new pages in batch.
    ///
    /// Single pass through allocation - reduces lock contention.
    pub fn batch_new_pages(&self, page_ids: &[PageId]) -> Result<Vec<&BufferFrame>> {
        let mut frames = Vec::with_capacity(page_ids.len());
        for &page_id in page_ids {
            frames.push(self.new_page(page_id)?);
        }
        Ok(frames)
    }

    /// Pre-allocates frame IDs from the free list in bulk.
    ///
    /// Single mutex acquisition drains up to `count` frames. Returns the
    /// reserved frame IDs. Caller uses `load_reserved_frame` to set up
    /// each frame individually, and `release_reserved_frames` to return
    /// any unused frames.
    pub fn reserve_frames(&self, count: usize) -> Vec<FrameId> {
        if count == 0 {
            return Vec::new();
        }
        self.free_list.pop_many(count)
    }

    /// Loads a page into a pre-reserved frame.
    ///
    /// Skips the free-list lock since the frame was already reserved.
    /// The frame is pinned, marked dirty, and data is copied in. The
    /// mapping publishes through insert_if_absent, a blind insert here
    /// could silently replace a mapping a concurrent fault-in installed
    /// and leave two frames claiming one page. Returns false when another
    /// thread won, in which case the reserved frame returns to the free
    /// list and the caller reads through the winner's mapping.
    /// Caller must call unpin_page when done with an installed page.
    pub fn load_reserved_frame(
        &self,
        frame_id: FrameId,
        page_id: PageId,
        data: &[u8; PAGE_SIZE],
    ) -> bool {
        let frame = &self.frames[frame_id.0 as usize];
        frame.reset();
        frame.set_page_id(Some(page_id));
        frame.pin();
        frame.copy_from(data);
        frame.set_dirty(true);
        match self.page_table.insert_if_absent(page_id, frame_id) {
            InsertOutcome::Inserted => {
                self.replacer.record_access(frame_id);
                true
            }
            InsertOutcome::Existing(_) | InsertOutcome::TableFull => {
                frame.set_page_id(None);
                frame.set_dirty(false);
                frame.unpin();
                self.free_list.push(frame_id);
                false
            }
        }
    }

    /// Returns unused reserved frames back to the free list.
    pub fn release_reserved_frames(&self, frame_ids: &[FrameId]) {
        if frame_ids.is_empty() {
            return;
        }
        self.free_list.push_many(frame_ids);
    }

    /// Returns a write guard for page data.
    pub fn write_page(&self, page_id: PageId) -> Option<PageWriteGuard<'_>> {
        let frame = self.fetch_page(page_id)?;
        Some(PageWriteGuard {
            pool: self,
            page_id,
            frame,
        })
    }

    /// Marks a page dirty and stamps it with the given LSN for checkpoint ordering.
    /// The LSN is only written if this is the first dirty since last flush (CAS from 0).
    #[inline]
    pub fn mark_dirty_with_lsn(&self, page_id: PageId, lsn: u64) {
        if let Some(frame_id) = self.page_table.get(page_id) {
            let frame = &self.frames[frame_id.0 as usize];
            frame.set_dirty(true);
            frame.set_dirty_lsn(lsn);
        }
    }

    /// Returns true if any frame is dirty at or below the boundary.
    /// Early-exit scan, O(1) best case when the first frame matches.
    ///
    /// A dirty frame whose LSN was never stamped (dirtied through an
    /// unpin with no WAL record, the FSM write path, or mid-stamp) has an
    /// unknown age, so it counts as dirty below every boundary. Missing it
    /// let checkpoints delete WAL segments whose redo records were the
    /// only durable copy of such a page's committed writes.
    pub fn has_dirty_pages_below(&self, below_lsn: u64) -> bool {
        let mut found = false;
        self.page_table.for_each(|_page_id, frame_id| {
            let frame = &self.frames[frame_id.0 as usize];
            let dlsn = frame.dirty_lsn();
            if (dlsn > 0 && dlsn <= below_lsn) || (dlsn == 0 && frame.is_dirty()) {
                found = true;
                return false; // stop iteration
            }
            true
        });
        found
    }

    /// Collects dirty frames with dirty_lsn <= below_lsn, sorted oldest-first.
    /// Skips pinned pages to avoid blocking the background writer.
    /// Returns up to `limit` entries as (page_id, frame_id, dirty_lsn).
    ///
    /// A dirty frame with no LSN stamp has an unknown age, so it is
    /// collected under every boundary and sorts first, otherwise it would
    /// stay invisible to the background writer forever.
    pub fn collect_dirty_pages(&self, below_lsn: u64, limit: usize) -> Vec<(PageId, FrameId, u64)> {
        let mut dirty = Vec::new();
        self.page_table.for_each(|page_id, frame_id| {
            let frame = &self.frames[frame_id.0 as usize];
            let dlsn = frame.dirty_lsn();
            let eligible = (dlsn > 0 && dlsn <= below_lsn) || (dlsn == 0 && frame.is_dirty());
            if eligible && !frame.is_pinned() {
                dirty.push((page_id, frame_id, dlsn));
            }
            true
        });
        dirty.sort_unstable_by_key(|&(_, _, lsn)| lsn);
        dirty.truncate(limit);
        dirty
    }

    /// Flushes a single dirty page collected by `collect_dirty_pages`.
    /// Returns true if the page was flushed, false if evicted, re-dirtied
    /// since collection, or already clean.
    ///
    /// The dirty state is captured and cleared before the copy: from that
    /// point a landing write's CAS-from-zero LSN stamp succeeds and the
    /// dirty flag re-arms, so the newer version stays discoverable. The
    /// old shape held the LSN through the flush, which made the landing
    /// write's stamp fail and the post-flush clear mark it clean. The pin
    /// is held across the I/O so the frame cannot be evicted and reused
    /// while its image is being written.
    /// `keep_pinned` leaves the frame pinned on a successful write, so the
    /// caller can batch fsync across pages and re-dirty this exact frame if
    /// that fsync fails. Without the pin the frame could be evicted as clean
    /// between the write and the fsync, and an fsync failure would then have
    /// no frame to re-dirty: the page's only current image would be an
    /// unsynced file write. The caller MUST unpin after its fsync resolves
    pub fn flush_dirty_frame<F>(
        &self,
        page_id: PageId,
        frame_id: FrameId,
        expected_lsn: u64,
        keep_pinned: bool,
        flush_fn: F,
    ) -> Result<bool>
    where
        F: FnOnce(PageId, &mut [u8; PAGE_SIZE]) -> Result<()>,
    {
        // Pin first, then verify the tenancy under the pin. Checking the
        // page id before pinning left a window in which the frame was
        // evicted and reused, and this writer then wrote the next tenant's
        // bytes into the expected page's disk slot
        let frame = &self.frames[frame_id.0 as usize];
        if !frame.try_pin() {
            return Ok(false);
        }
        if frame.page_id() != Some(page_id) {
            frame.unpin();
            return Ok(false);
        }

        let flush_order = self.flush_serial.lock();
        if !frame.is_dirty() {
            drop(flush_order);
            frame.unpin();
            return Ok(false);
        }
        frame.set_dirty(false);
        if !frame.clear_dirty_lsn(expected_lsn) {
            // Re-stamped since collection, a newer write owns the state
            // now, restore the flag and leave it for the next cycle
            frame.set_dirty(true);
            drop(flush_order);
            frame.unpin();
            return Ok(false);
        }

        // Copy data out while pinned
        let mut buf = Box::new([0u8; PAGE_SIZE]);
        let data = frame.read_data();
        buf.copy_from_slice(&**data);
        drop(data);

        // Write to disk
        let outcome = flush_fn(page_id, &mut buf);
        drop(flush_order);
        match outcome {
            Ok(()) => {
                if !keep_pinned {
                    frame.unpin();
                }
                Ok(true)
            }
            Err(e) => {
                // Restore so the page stays discoverable and is retried
                frame.set_dirty(true);
                frame.set_dirty_lsn(expected_lsn);
                frame.unpin();
                Err(e)
            }
        }
    }

    /// Releases the pin `flush_dirty_frame(keep_pinned = true)` held, after
    /// the caller's batched fsync resolved. `redirty` restores the page's
    /// dirty state first, for the fsync-failure path, and lands on this
    /// exact frame because the pin kept it resident
    pub fn finish_pinned_flush(&self, frame_id: FrameId, dirty_lsn: u64, redirty: bool) {
        let frame = &self.frames[frame_id.0 as usize];
        if redirty {
            frame.set_dirty(true);
            frame.set_dirty_lsn(dirty_lsn);
        }
        frame.unpin();
    }

    /// Returns statistics about the buffer pool.
    /// The resident pages worth handing to another node, hottest first.
    ///
    /// Hotness is the clock's own reference bit, which is the only recency
    /// signal a buffer pool can produce without paying for it on every page
    /// access. Pages touched since the hand last passed come first, the rest
    /// of the resident set follows, and residency alone is already strong
    /// evidence: these are the pages that survived eviction.
    ///
    /// Pinned pages are included. A page pinned at the instant the manifest is
    /// taken is being used right now, which is the strongest possible reason
    /// for a survivor to want it.
    pub fn hot_pages(&self, limit: usize) -> Vec<PageId> {
        if limit == 0 {
            return Vec::new();
        }
        let mut referenced = Vec::with_capacity(limit.min(self.config.num_frames));
        let mut resident = Vec::new();
        self.page_table.for_each(|page_id, frame_id| {
            if self.replacer.is_referenced(frame_id) {
                referenced.push(page_id);
            } else if resident.len() < limit {
                resident.push(page_id);
            }
            // Stop once the referenced set alone fills the manifest, because
            // nothing found later can outrank what is already held
            referenced.len() < limit
        });
        if referenced.len() >= limit {
            referenced.truncate(limit);
            return referenced;
        }
        let room = limit - referenced.len();
        resident.truncate(room);
        referenced.extend(resident);
        referenced
    }

    /// Reads a departing node's working set in, so the traffic that follows it
    /// lands on a warm pool instead of a cold one.
    ///
    /// Never evicts. A prefetch that displaced resident pages would trade a
    /// survivor's own warm set for a guess about the traffic it is about to
    /// inherit, and the survivor's set is the one that is definitely being
    /// used. Pages beyond the free-frame reserve are declined and counted.
    ///
    /// `read` supplies page bytes. It returns None for a page that no longer
    /// exists, which is expected: the manifest was taken before the drain and
    /// the departing node may have dropped pages since.
    pub fn prefetch<F>(&self, pages: &[PageId], mut read: F) -> PrefetchReport
    where
        F: FnMut(PageId) -> Option<Vec<u8>>,
    {
        let reserve = (self.config.num_frames / PREFETCH_RESERVE_DIVISOR).max(1);
        let mut report = PrefetchReport {
            requested: pages.len() as u64,
            ..PrefetchReport::default()
        };
        for page_id in pages {
            if self.page_table.contains(*page_id) {
                report.already_resident += 1;
                continue;
            }
            if self.free_count() <= reserve {
                report.declined += 1;
                continue;
            }
            let Some(bytes) = read(*page_id) else {
                report.failed += 1;
                continue;
            };
            match self.load_page(*page_id, &bytes) {
                Ok(frame) => {
                    // Loading pins the frame. A prefetched page has no reader
                    // yet, so the pin is released immediately and the page is
                    // evictable from the moment it lands
                    frame.unpin();
                    report.loaded += 1;
                }
                Err(_) => report.failed += 1,
            }
        }
        report
    }

    pub fn stats(&self) -> BufferPoolStats {
        let mut pinned_count = 0;
        let mut dirty_count = 0;

        self.page_table.for_each(|_, frame_id| {
            let frame = &self.frames[frame_id.0 as usize];
            if frame.is_pinned() {
                pinned_count += 1;
            }
            if frame.is_dirty() {
                dirty_count += 1;
            }
            true // continue iteration
        });

        BufferPoolStats {
            total_frames: self.config.num_frames,
            free_frames: self.free_count(),
            used_frames: self.page_table.len(),
            pinned_frames: pinned_count,
            dirty_frames: dirty_count,
        }
    }
}

/// One round of a claim-race retry: a brief spin for the first rounds while
/// the racing owner finishes its handful of instructions, then a timeslice
/// yield so a descheduled owner can run instead of being starved by the
/// spinner under CPU oversubscription
#[inline]
fn retry_pause(round: &mut u32) {
    *round = round.saturating_add(1);
    if *round < 16 {
        std::hint::spin_loop();
    } else {
        std::thread::yield_now();
    }
}

/// Debug-build watchdog for the claim-retry loops: a loop this hot retrying
/// for millions of rounds means the protocol leaked a claim or wedged a
/// mapping, and a loud panic with the frame's state beats an silent hang
#[cfg(debug_assertions)]
fn stall_diagnostic(round: u32, site: &str, page_id: PageId, frame: &BufferFrame) {
    if round == 5_000_000 {
        panic!(
            "{site} stalled on page {page_id}: frame {} pin_count {:#x} tenant {:?} dirty {}",
            frame.frame_id(),
            frame.pin_count(),
            frame.page_id(),
            frame.is_dirty()
        );
    }
}

/// Statistics about the buffer pool.
#[derive(Debug, Clone)]
pub struct BufferPoolStats {
    /// Total number of frames.
    pub total_frames: usize,
    /// Number of free frames.
    pub free_frames: usize,
    /// Number of frames with pages.
    pub used_frames: usize,
    /// Number of pinned frames.
    pub pinned_frames: usize,
    /// Number of dirty frames.
    pub dirty_frames: usize,
}

/// RAII guard for reading a page.
pub struct PageReadGuard<'a> {
    pool: &'a BufferPool,
    page_id: PageId,
    frame: &'a BufferFrame,
}

impl<'a> PageReadGuard<'a> {
    /// Returns the page ID.
    pub fn page_id(&self) -> PageId {
        self.page_id
    }

    /// Returns the page data.
    pub fn data(&self) -> parking_lot::RwLockReadGuard<'_, Box<[u8; PAGE_SIZE]>> {
        self.frame.read_data()
    }
}

impl Drop for PageReadGuard<'_> {
    fn drop(&mut self) {
        self.pool.unpin_page(self.page_id, false);
    }
}

/// RAII guard for writing a page.
pub struct PageWriteGuard<'a> {
    pool: &'a BufferPool,
    page_id: PageId,
    frame: &'a BufferFrame,
}

impl<'a> PageWriteGuard<'a> {
    /// Returns the page ID.
    pub fn page_id(&self) -> PageId {
        self.page_id
    }

    /// Returns mutable access to page data.
    pub fn data_mut(&self) -> parking_lot::RwLockWriteGuard<'_, Box<[u8; PAGE_SIZE]>> {
        self.frame.write_data()
    }

    /// Marks the page as dirty.
    pub fn set_dirty(&self) {
        self.frame.set_dirty(true);
    }
}

impl Drop for PageWriteGuard<'_> {
    fn drop(&mut self) {
        self.pool.unpin_page(self.page_id, self.frame.is_dirty());
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_pool(num_frames: usize) -> BufferPool {
        BufferPool::new(BufferPoolConfig { num_frames })
    }

    /// A page dirtied without an LSN stamp (an unpin-dirty with no WAL
    /// record, the FSM write path) must still block WAL truncation below
    /// any boundary and must be visible to the background flusher,
    /// otherwise its only durable copy is the stale disk image and the
    /// redo that would rebuild it gets deleted
    #[test]
    fn test_dirty_without_lsn_blocks_checkpoint_and_reaches_the_flusher() {
        let pool = create_test_pool(4);
        let pid = PageId::new(0, 3);
        pool.new_page(pid).expect("new page");
        pool.unpin_page(pid, true);

        assert!(
            pool.has_dirty_pages_below(1),
            "a dirty page with no LSN must block truncation below any boundary"
        );
        let collected = pool.collect_dirty_pages(1, 16);
        assert!(
            collected.iter().any(|&(p, _, _)| p == pid),
            "the background flusher must see a dirty page with no LSN"
        );
    }

    /// A write landing while the frame is mid-flush must leave the frame
    /// dirty afterwards. The old shape held the dirty LSN through the
    /// flush, so the landing write's CAS-from-zero stamp failed and the
    /// post-flush clear marked the newer version clean and undiscoverable
    #[test]
    fn test_flush_dirty_frame_keeps_a_write_that_lands_during_the_flush() {
        let pool = create_test_pool(4);
        let pid = PageId::new(0, 5);
        pool.new_page(pid).expect("new page");
        pool.unpin_page(pid, true);
        pool.mark_dirty_with_lsn(pid, 40);

        let frame_id = pool.page_table.get(pid).expect("mapped");
        let flushed = pool
            .flush_dirty_frame(pid, frame_id, 40, false, |_, _| {
                // a foreground write lands while the flush is in flight
                pool.mark_dirty_with_lsn(pid, 90);
                Ok(())
            })
            .expect("flush");
        assert!(flushed);

        let frame = &pool.frames[frame_id.0 as usize];
        assert!(
            frame.is_dirty(),
            "the write that landed during the flush was marked clean"
        );
        assert_eq!(
            frame.dirty_lsn(),
            90,
            "the landing write's LSN stamp was swallowed by the in-flight flush"
        );
    }

    /// Same property for flush_all, whose old shape cleared the dirty flag
    /// after dropping the data guard, erasing a write that landed between
    /// the copy and the clear
    #[test]
    fn test_flush_all_keeps_a_write_that_lands_during_the_flush() {
        let pool = create_test_pool(4);
        let pid = PageId::new(0, 6);
        pool.new_page(pid).expect("new page");
        pool.unpin_page(pid, true);
        pool.mark_dirty_with_lsn(pid, 40);

        pool.flush_all(|p, _| {
            if p == pid {
                pool.mark_dirty_with_lsn(pid, 90);
            }
            Ok(FlushOutcome::Written)
        })
        .expect("flush all");

        let frame_id = pool.page_table.get(pid).expect("mapped");
        let frame = &pool.frames[frame_id.0 as usize];
        assert!(
            frame.is_dirty(),
            "the write that landed during flush_all was marked clean"
        );
        assert_eq!(frame.dirty_lsn(), 90, "the landing write's LSN was lost");
    }

    /// A page's first eight bytes carry its page number as the identity
    /// marker the aliasing hammers below validate against
    fn marker_page(page_num: u64) -> [u8; PAGE_SIZE] {
        let mut data = [0u8; PAGE_SIZE];
        data[..8].copy_from_slice(&page_num.to_le_bytes());
        data
    }

    /// A fetched frame must serve the requested page's bytes for as long
    /// as the fetch pin is held. The pool is far smaller than the page set
    /// so evictions run constantly, and a fetch that pins over an eviction
    /// claim without revalidating hands back the next tenant's bytes
    #[test]
    fn test_fetch_never_serves_another_pages_bytes() {
        let pool = std::sync::Arc::new(create_test_pool(4));
        let pages: u64 = 16;
        let violations = std::sync::atomic::AtomicU64::new(0);
        std::thread::scope(|s| {
            for t in 0..3 {
                let pool = std::sync::Arc::clone(&pool);
                s.spawn(move || {
                    for i in 0..40_000u64 {
                        let page_num = (i.wrapping_mul(2654435761).wrapping_add(t)) % pages;
                        let pid = PageId::new(0, page_num);
                        if pool.load_page(pid, &marker_page(page_num)).is_ok() {
                            pool.unpin_page(pid, false);
                        }
                    }
                });
            }
            for t in 0..3 {
                let pool = std::sync::Arc::clone(&pool);
                let violations = &violations;
                s.spawn(move || {
                    for i in 0..40_000u64 {
                        let page_num = (i.wrapping_mul(2246822519).wrapping_add(t)) % pages;
                        let pid = PageId::new(0, page_num);
                        if let Some(frame) = pool.fetch_page(pid) {
                            let guard = frame.read_data();
                            let seen = u64::from_le_bytes(guard[..8].try_into().expect("8 bytes"));
                            drop(guard);
                            if seen != page_num {
                                violations.fetch_add(1, Ordering::Relaxed);
                            }
                            frame.unpin();
                        }
                    }
                });
            }
        });
        assert_eq!(
            violations.load(Ordering::Relaxed),
            0,
            "a pinned fetch served another page's bytes"
        );
    }

    /// delete_page must not wipe a pin a concurrent reader holds, and a
    /// reader that fetched the page must see its bytes until it unpins.
    /// The old shape removed the mapping, reset the frame (pin count
    /// included) and re-inserted when pinned, all as separate steps
    #[test]
    fn test_delete_page_never_wipes_a_readers_pin() {
        let pool = std::sync::Arc::new(create_test_pool(4));
        let pid = PageId::new(0, 7);
        let violations = std::sync::atomic::AtomicU64::new(0);
        std::thread::scope(|s| {
            {
                let pool = std::sync::Arc::clone(&pool);
                s.spawn(move || {
                    for _ in 0..60_000u64 {
                        pool.delete_page(pid);
                        if pool.load_page(pid, &marker_page(7)).is_ok() {
                            pool.unpin_page(pid, false);
                        }
                    }
                });
            }
            {
                let pool = std::sync::Arc::clone(&pool);
                let violations = &violations;
                s.spawn(move || {
                    for _ in 0..60_000u64 {
                        if let Some(frame) = pool.fetch_page(pid) {
                            let guard = frame.read_data();
                            let seen = u64::from_le_bytes(guard[..8].try_into().expect("8 bytes"));
                            drop(guard);
                            if seen != 7 {
                                violations.fetch_add(1, Ordering::Relaxed);
                            }
                            frame.unpin();
                        }
                    }
                });
            }
        });
        assert_eq!(
            violations.load(Ordering::Relaxed),
            0,
            "delete_page let a pinned reader observe foreign or wiped bytes"
        );
        // No thread holds a pin here, a wiped pin would show as residue
        if let Some(frame_id) = pool.page_table.get(pid) {
            assert_eq!(
                pool.frames[frame_id.0 as usize].pin_count(),
                0,
                "pin accounting corrupted by delete_page"
            );
        }
    }

    #[test]
    fn test_buffer_pool_new() {
        let pool = create_test_pool(10);

        assert_eq!(pool.num_frames(), 10);
        assert_eq!(pool.free_count(), 10);
        assert_eq!(pool.page_count(), 0);
    }

    #[test]
    fn test_buffer_pool_new_page() {
        let pool = create_test_pool(10);
        let page_id = PageId::new(0, 1);

        let frame = pool.new_page(page_id).unwrap();

        assert_eq!(frame.page_id(), Some(page_id));
        assert!(frame.is_pinned());
        assert_eq!(pool.free_count(), 9);
        assert_eq!(pool.page_count(), 1);
        assert!(pool.contains(page_id));
    }

    #[test]
    fn test_buffer_pool_fetch_existing() {
        let pool = create_test_pool(10);
        let page_id = PageId::new(0, 1);

        pool.new_page(page_id).unwrap();
        pool.unpin_page(page_id, false);

        let frame = pool.fetch_page(page_id).unwrap();
        assert_eq!(frame.page_id(), Some(page_id));
        assert!(frame.is_pinned());
    }

    #[test]
    fn test_buffer_pool_fetch_nonexistent() {
        let pool = create_test_pool(10);
        let page_id = PageId::new(0, 1);

        assert!(pool.fetch_page(page_id).is_none());
    }

    #[test]
    fn test_buffer_pool_unpin() {
        let pool = create_test_pool(10);
        let page_id = PageId::new(0, 1);

        let frame = pool.new_page(page_id).unwrap();
        assert!(frame.is_pinned());

        pool.unpin_page(page_id, false);
        assert!(!frame.is_pinned());
    }

    #[test]
    fn test_buffer_pool_dirty_tracking() {
        let pool = create_test_pool(10);
        let page_id = PageId::new(0, 1);

        pool.new_page(page_id).unwrap();
        pool.unpin_page(page_id, true);

        let frame = pool.fetch_page(page_id).unwrap();
        assert!(frame.is_dirty());
    }

    #[test]
    fn test_buffer_pool_eviction() {
        let pool = create_test_pool(3);

        // Fill the pool
        for i in 0..3 {
            let page_id = PageId::new(0, i);
            pool.new_page(page_id).unwrap();
            pool.unpin_page(page_id, false);
        }

        assert_eq!(pool.free_count(), 0);
        assert_eq!(pool.page_count(), 3);

        // Add one more page, should evict
        let new_page_id = PageId::new(0, 99);
        pool.new_page(new_page_id).unwrap();

        assert_eq!(pool.page_count(), 3);
        assert!(pool.contains(new_page_id));
    }

    #[test]
    fn test_buffer_pool_eviction_dirty() {
        use std::sync::Arc;
        use std::sync::Mutex;

        let pool = create_test_pool(1);
        let written: Arc<Mutex<Vec<(PageId, u8)>>> = Arc::new(Mutex::new(Vec::new()));
        let sink = Arc::clone(&written);
        pool.set_evict_writer(Arc::new(move |pid, data: &mut [u8; PAGE_SIZE], _lsn| {
            sink.lock().unwrap().push((pid, data[0]));
            Ok(())
        }))
        .unwrap();

        // Add dirty page with some data
        let page_id1 = PageId::new(0, 1);
        let frame = pool.new_page(page_id1).unwrap();
        frame.write_data()[0] = 0xAB;
        pool.unpin_page(page_id1, true);

        // Add another page, which evicts the dirty one through the hook
        let page_id2 = PageId::new(0, 2);
        pool.new_page(page_id2).unwrap();

        assert_eq!(written.lock().unwrap().as_slice(), &[(page_id1, 0xAB)]);
    }

    #[test]
    fn test_buffer_pool_full_all_pinned() {
        let pool = create_test_pool(2);

        // Fill pool with pinned pages
        pool.new_page(PageId::new(0, 1)).unwrap();
        pool.new_page(PageId::new(0, 2)).unwrap();

        // Try to add another page (should fail)
        let result = pool.new_page(PageId::new(0, 3));
        assert!(matches!(result, Err(ZyronError::BufferPoolFull)));
    }

    #[test]
    fn test_buffer_pool_delete_page() {
        let pool = create_test_pool(10);
        let page_id = PageId::new(0, 1);

        pool.new_page(page_id).unwrap();
        pool.unpin_page(page_id, false);

        assert!(pool.contains(page_id));
        assert!(pool.delete_page(page_id));
        assert!(!pool.contains(page_id));
        assert_eq!(pool.free_count(), 10);
    }

    #[test]
    fn test_buffer_pool_delete_pinned_page() {
        let pool = create_test_pool(10);
        let page_id = PageId::new(0, 1);

        pool.new_page(page_id).unwrap();
        // Don't unpin

        assert!(!pool.delete_page(page_id));
        assert!(pool.contains(page_id));
    }

    #[test]
    fn test_buffer_pool_load_page() {
        let pool = create_test_pool(10);
        let page_id = PageId::new(0, 1);
        let data = [0xABu8; PAGE_SIZE];

        let frame = pool.load_page(page_id, &data).unwrap();

        let frame_data = frame.read_data();
        assert_eq!(frame_data[0], 0xAB);
        assert_eq!(frame_data[100], 0xAB);
    }

    #[test]
    fn test_buffer_pool_flush_page() {
        let pool = create_test_pool(10);
        let page_id = PageId::new(0, 1);

        pool.new_page(page_id).unwrap();
        pool.unpin_page(page_id, true);

        let mut flushed_pages = vec![];
        let result = pool.flush_page(page_id, |pid, _data| {
            flushed_pages.push(pid);
            Ok(())
        });

        assert!(result.unwrap());
        assert_eq!(flushed_pages, vec![page_id]);

        // Page should no longer be dirty
        let frame = pool.fetch_page(page_id).unwrap();
        assert!(!frame.is_dirty());
    }

    #[test]
    fn test_buffer_pool_flush_all() {
        let pool = create_test_pool(10);

        // Add multiple dirty pages
        for i in 0..5 {
            let page_id = PageId::new(0, i);
            pool.new_page(page_id).unwrap();
            pool.unpin_page(page_id, true);
        }

        let mut flushed_count = 0;
        let result = pool.flush_all(|_pid, _data| {
            flushed_count += 1;
            Ok(FlushOutcome::Written)
        });

        assert_eq!(result.unwrap(), 5);
        assert_eq!(flushed_count, 5);
    }

    // A file-filtered flusher sharing the pool must not mark pages it
    // declined as clean: the skipped page keeps its dirty bit and dirty
    // LSN so the owning subsystem's flush and the checkpoint still see it
    #[test]
    fn test_flush_all_skipped_pages_stay_dirty() {
        let pool = create_test_pool(10);

        let owned = PageId::new(0, 1);
        let foreign = PageId::new(7, 1);
        for page_id in [owned, foreign] {
            pool.new_page(page_id).unwrap();
            pool.unpin_page(page_id, true);
        }
        pool.mark_dirty_with_lsn(foreign, 42);

        let flushed = pool
            .flush_all(|pid, _data| {
                if pid.file_id == 0 {
                    Ok(FlushOutcome::Written)
                } else {
                    Ok(FlushOutcome::Skipped)
                }
            })
            .expect("flush all");
        assert_eq!(flushed, 1, "only the owned page counts as flushed");

        let owned_frame = pool.fetch_page(owned).unwrap();
        assert!(!owned_frame.is_dirty(), "written page is clean");
        pool.unpin_page(owned, false);

        let foreign_frame = pool.fetch_page(foreign).unwrap();
        assert!(foreign_frame.is_dirty(), "skipped page keeps its dirty bit");
        assert_eq!(
            foreign_frame.dirty_lsn(),
            42,
            "skipped page keeps its dirty LSN"
        );
        pool.unpin_page(foreign, false);
    }

    #[test]
    fn test_buffer_pool_read_guard() {
        let pool = create_test_pool(10);
        let page_id = PageId::new(0, 1);

        pool.new_page(page_id).unwrap();
        pool.unpin_page(page_id, false);

        {
            let guard = pool.read_page(page_id).unwrap();
            assert_eq!(guard.page_id(), page_id);
            // Guard holds one pin
        }

        // After guard dropped, page should be unpinned (pin_count = 0)
        // Fetch adds a new pin, so pin_count becomes 1
        let frame = pool.fetch_page(page_id).unwrap();
        assert_eq!(frame.pin_count(), 1);
    }

    #[test]
    fn test_buffer_pool_write_guard() {
        let pool = create_test_pool(10);
        let page_id = PageId::new(0, 1);

        pool.new_page(page_id).unwrap();
        pool.unpin_page(page_id, false);

        {
            let guard = pool.write_page(page_id).unwrap();
            guard.set_dirty();
            {
                let mut data = guard.data_mut();
                data[0] = 0xFF;
            }
        }

        // After guard dropped, page should be dirty
        let frame = pool.fetch_page(page_id).unwrap();
        assert!(frame.is_dirty());
        assert_eq!(frame.read_data()[0], 0xFF);
    }

    #[test]
    fn test_buffer_pool_stats() {
        let pool = create_test_pool(10);

        // Add some pages
        for i in 0..5 {
            let page_id = PageId::new(0, i);
            pool.new_page(page_id).unwrap();
            if i % 2 == 0 {
                pool.unpin_page(page_id, true); // Dirty
            }
            // Odd pages remain pinned
        }

        let stats = pool.stats();
        assert_eq!(stats.total_frames, 10);
        assert_eq!(stats.free_frames, 5);
        assert_eq!(stats.used_frames, 5);
        assert_eq!(stats.pinned_frames, 2); // Pages 1, 3
        assert_eq!(stats.dirty_frames, 3); // Pages 0, 2, 4
    }

    #[test]
    fn test_buffer_pool_duplicate_new_page() {
        let pool = create_test_pool(10);
        let page_id = PageId::new(0, 1);

        pool.new_page(page_id).unwrap();
        pool.unpin_page(page_id, false);

        // Adding same page again should return existing frame
        let frame = pool.new_page(page_id).unwrap();

        assert_eq!(frame.page_id(), Some(page_id));
        assert_eq!(pool.page_count(), 1);
    }

    /// Concurrent allocations under an empty free list land on distinct
    /// frames.
    ///
    /// The clock replacer hands back a victim without marking it taken, so
    /// two sweeps running at once can choose the same frame. Both callers
    /// then reset it, install their own page id, and publish a mapping to
    /// it: two page ids alias one frame, and a read of one returns the
    /// other's bytes.
    ///
    /// The free list is drained first, because that is the only state in
    /// which allocation reaches the replacer at all.
    #[test]
    fn test_concurrent_eviction_never_hands_one_frame_to_two_pages() {
        use std::sync::Arc;

        const FRAMES: usize = 8;
        const THREADS: usize = 16;

        for round in 0..200u64 {
            let pool = Arc::new(create_test_pool(FRAMES));

            // Fill every frame, then unpin so all of them are evictable and
            // the free list is empty
            for f in 0..FRAMES as u64 {
                let seed = PageId::new(3, 10_000 + round * 1000 + f);
                pool.new_page(seed).expect("seed page");
                // Dirty, so evicting one copies its 16 KB out before the
                // caller can pin it. That copy is the window in which a
                // second sweep can choose the same victim
                pool.unpin_page(seed, true);
            }
            assert_eq!(pool.free_count(), 0, "round {round}: free list drained");

            let barrier = Arc::new(std::sync::Barrier::new(THREADS));
            let mut handles = Vec::new();
            for t in 0..THREADS as u64 {
                let pool = Arc::clone(&pool);
                let barrier = Arc::clone(&barrier);
                handles.push(std::thread::spawn(move || {
                    let page_id = PageId::new(4, round * 100 + t);
                    barrier.wait();
                    let frame = pool.new_page(page_id).map(|f| f.frame_id());
                    (page_id, frame.ok())
                }));
            }
            let got: Vec<(PageId, Option<FrameId>)> = handles
                .into_iter()
                .map(|h| h.join().expect("thread"))
                .collect();

            // Two live page ids may never share a frame
            let mut by_frame: std::collections::HashMap<u32, Vec<PageId>> =
                std::collections::HashMap::new();
            for (page_id, frame) in &got {
                if let Some(f) = frame {
                    by_frame.entry(f.0).or_default().push(*page_id);
                }
            }
            for (frame, pages) in &by_frame {
                assert_eq!(
                    pages.len(),
                    1,
                    "round {round}: frame {frame} handed to {pages:?}"
                );
            }
            // And the table agrees with what each caller was given
            for (page_id, frame) in &got {
                if let Some(f) = frame {
                    assert_eq!(
                        pool.page_table.get(*page_id),
                        Some(*f),
                        "round {round}: {page_id:?} maps elsewhere"
                    );
                }
            }
        }
    }

    #[test]
    fn test_concurrent_new_page_resolves_to_one_frame() {
        use std::sync::Arc;
        use std::sync::atomic::{AtomicU32, Ordering};

        // Direct path id and hash path id both exercised
        for page_id in [PageId::new(0, 7), PageId::new(3, 99)] {
            let pool = Arc::new(create_test_pool(64));
            let barrier = Arc::new(std::sync::Barrier::new(8));
            let mut handles = Vec::new();

            for _ in 0..8 {
                let pool = Arc::clone(&pool);
                let barrier = Arc::clone(&barrier);
                handles.push(std::thread::spawn(move || {
                    barrier.wait();
                    let frame = pool.new_page(page_id).unwrap();
                    let fid = frame.frame_id();
                    pool.unpin_page(page_id, false);
                    fid
                }));
            }

            let mut frame_ids: Vec<_> = handles.into_iter().map(|h| h.join().unwrap()).collect();
            frame_ids.sort_unstable_by_key(|f| f.0);
            frame_ids.dedup_by_key(|f| f.0);

            // All concurrent inserts for one page id resolve to a single frame
            assert_eq!(frame_ids.len(), 1, "page id mapped to multiple frames");
            assert_eq!(pool.page_count(), 1);
            assert_eq!(pool.page_table.get(page_id), Some(frame_ids[0]));

            // No frames leaked: pool started with 64, one is in use
            let in_use = AtomicU32::new(0);
            pool.page_table.for_each(|_, _| {
                in_use.fetch_add(1, Ordering::Relaxed);
                true
            });
            assert_eq!(in_use.load(Ordering::Relaxed), 1);
        }
    }

    #[test]
    fn test_concurrent_unpin_never_underflows() {
        use std::sync::Arc;

        let pool = Arc::new(create_test_pool(4));
        let page_id = PageId::new(0, 1);

        // Pin the page eight times across threads
        pool.new_page(page_id).unwrap(); // pin 1
        for _ in 0..7 {
            pool.fetch_page(page_id).unwrap();
        }
        let frame_id = pool.page_table.get(page_id).unwrap();
        assert_eq!(pool.frames[frame_id.0 as usize].pin_count(), 8);

        // Unpin sixteen times concurrently, twice the pin count. The saturating
        // unpin must floor at zero and never wrap to a huge value.
        let mut handles = Vec::new();
        for _ in 0..16 {
            let pool = Arc::clone(&pool);
            handles.push(std::thread::spawn(move || {
                pool.unpin_page(page_id, false);
            }));
        }
        for h in handles {
            h.join().unwrap();
        }

        assert_eq!(pool.frames[frame_id.0 as usize].pin_count(), 0);
        assert!(!pool.frames[frame_id.0 as usize].is_pinned());
    }

    #[test]
    fn test_evict_writer_flushes_dirty_victim() {
        use std::sync::Arc;
        use std::sync::Mutex;

        let pool = create_test_pool(1);
        let written: Arc<Mutex<Vec<(PageId, u8)>>> = Arc::new(Mutex::new(Vec::new()));
        let sink = Arc::clone(&written);
        let writer: EvictWriteFn = Arc::new(move |pid, data, _lsn| {
            sink.lock().unwrap().push((pid, data[0]));
            Ok(())
        });
        pool.set_evict_writer(writer).unwrap();

        // Dirty the only frame
        let page_id1 = PageId::new(0, 1);
        let frame = pool.new_page(page_id1).unwrap();
        frame.write_data()[0] = 0xCD;
        pool.unpin_page(page_id1, true);

        // Force eviction, which writes the victim through the hook
        let page_id2 = PageId::new(0, 2);
        pool.new_page(page_id2).unwrap();

        let log = written.lock().unwrap();
        assert_eq!(log.as_slice(), &[(page_id1, 0xCD)]);
    }

    /// A pool with no write hook refuses the eviction rather than taking a
    /// dirty page it cannot write.
    ///
    /// Eviction used to copy a dirty victim's bytes out for the caller to
    /// write and take its page table mapping down straight away. Between
    /// that and the caller's write the page was absent from the pool and
    /// stale on disk, so a reader faulting it back in installed the stale
    /// image and every write still only in those bytes was gone. There is
    /// nowhere to put a dirty page without a hook, so the frame is not taken
    /// and the pool reports itself full instead.
    #[test]
    fn a_pool_with_no_write_hook_refuses_to_evict_a_dirty_page() {
        let pool = create_test_pool(2);

        let resident = [PageId::new(0, 1), PageId::new(0, 2)];
        for (i, page_id) in resident.iter().enumerate() {
            let frame = pool.new_page(*page_id).unwrap();
            frame.write_data()[0] = 0xA0 + i as u8;
            pool.unpin_page(*page_id, true);
        }

        // Both frames hold a dirty page and nothing can write one out
        let err = pool
            .new_page(PageId::new(0, 3))
            .expect_err("a dirty page was evicted with nowhere to write it");
        assert!(
            matches!(err, ZyronError::BufferPoolFull),
            "unexpected error: {err:?}"
        );

        // Both pages are still where they were, with what they held
        for (i, page_id) in resident.iter().enumerate() {
            let frame = pool
                .fetch_page(*page_id)
                .expect("a refused eviction left a page out of the pool");
            assert_eq!(frame.read_data()[0], 0xA0 + i as u8);
            assert!(frame.is_dirty(), "the page was quietly marked written");
            pool.unpin_page(*page_id, false);
        }
    }

    /// A clean page is still evictable without a hook, because there is
    /// nothing to write.
    #[test]
    fn a_pool_with_no_write_hook_still_evicts_a_clean_page() {
        let pool = create_test_pool(1);

        let page_id1 = PageId::new(0, 1);
        pool.new_page(page_id1).unwrap();
        pool.unpin_page(page_id1, false);

        let page_id2 = PageId::new(0, 2);
        pool.new_page(page_id2).expect("a clean frame is reusable");
        pool.unpin_page(page_id2, false);
        assert!(pool.fetch_page(page_id1).is_none());
    }

    #[test]
    fn test_flush_all_continues_past_errors() {
        let pool = create_test_pool(10);
        for i in 0..5 {
            let page_id = PageId::new(0, i);
            pool.new_page(page_id).unwrap();
            pool.unpin_page(page_id, true);
        }

        // Fail the flush for one page, the rest must still be attempted
        let mut attempted = 0;
        let result = pool.flush_all(|pid, _data| {
            attempted += 1;
            if pid.page_num == 2 {
                Err(ZyronError::IoError("disk full".to_string()))
            } else {
                Ok(FlushOutcome::Written)
            }
        });

        // Every dirty page was attempted, and the incomplete flush is reported
        assert_eq!(attempted, 5);
        assert!(result.is_err());

        // The failed page stays dirty for retry, the others are clean
        assert!(pool.fetch_page(PageId::new(0, 2)).unwrap().is_dirty());
        assert!(!pool.fetch_page(PageId::new(0, 0)).unwrap().is_dirty());
    }
}
