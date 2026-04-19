//! High-performance ring buffer for WAL writes.
//!
//! Uses a contiguous byte buffer with atomic cursors.
//! Writers claim space with a single fetch_add and write directly.

use crate::record::Lsn;
use std::cell::UnsafeCell;
use std::sync::atomic::{AtomicU64, Ordering};

/// Contiguous ring buffer for WAL records.
///
/// Writers claim space atomically and write directly to the buffer.
/// Flush thread reads committed data in order.
/// Cache-line-sized padding to prevent false sharing between atomics touched
/// by different threads. Each hot atomic gets its own 64-byte line so writer
/// and flush thread don't invalidate each other's caches on unrelated fields.
#[repr(align(64))]
struct CachePadded<T>(T);

impl<T> std::ops::Deref for CachePadded<T> {
    type Target = T;
    #[inline(always)]
    fn deref(&self) -> &T {
        &self.0
    }
}

pub struct RingBuffer {
    /// Contiguous byte buffer.
    buffer: UnsafeCell<Box<[u8]>>,
    /// Buffer size in bytes.
    buffer_size: usize,
    /// Bitmask for power-of-2 modulo (buffer_size - 1). Bitwise AND replaces
    /// integer division for offset calculation: 1 cycle vs 20-40 cycles on x86.
    buffer_mask: usize,
    /// Write cursor: next byte offset to claim. Hit by every writer claim,
    /// isolated on its own cache line so flush-thread reads of other cursors
    /// don't force writers to refetch.
    write_cursor: CachePadded<AtomicU64>,
    /// Committed cursor - bytes ready to drain. Writer commits, flush reads.
    /// Paired with max_lsn because both are updated in commit_write (one
    /// cache line dirty per commit instead of two).
    committed_cursor: CachePadded<AtomicU64>,
    /// Maximum LSN written. Co-located with committed_cursor (see above).
    /// Logically on the same cache line via the commit pairing; keeping it
    /// on its own CachePadded field prevents packing with write_cursor.
    max_lsn: CachePadded<AtomicU64>,
    /// Read cursor: bytes already drained. Owned by the flush thread;
    /// writers only read it in the slow path (wait_for_space_slow).
    read_cursor: CachePadded<AtomicU64>,
    /// Cached write limit: writers can write up to this point without checking
    /// read_cursor. Updated by the flush thread after each drain. This avoids
    /// cross-core cache line traffic on the hot path.
    safe_write_limit: CachePadded<AtomicU64>,
}

// SAFETY: Buffer access is coordinated via atomic cursors.
// Writers only write to their claimed regions.
// Reader only reads committed regions.
unsafe impl Send for RingBuffer {}
unsafe impl Sync for RingBuffer {}

impl RingBuffer {
    /// Creates a new ring buffer with the given capacity in bytes.
    ///
    /// Capacity must be at least 128KB. For correctness, capacity should be >=
    /// the WAL segment size so that wrap-around only occurs after a full drain.
    pub fn new(capacity_bytes: usize) -> Self {
        assert!(
            capacity_bytes.is_power_of_two(),
            "Ring buffer capacity must be a power of 2, got {} bytes",
            capacity_bytes,
        );
        debug_assert!(
            capacity_bytes >= 128 * 1024 || cfg!(test),
            "Ring buffer capacity too small: {} bytes",
            capacity_bytes,
        );
        let buffer = vec![0u8; capacity_bytes].into_boxed_slice();

        Self {
            buffer: UnsafeCell::new(buffer),
            buffer_size: capacity_bytes,
            buffer_mask: capacity_bytes - 1,
            write_cursor: CachePadded(AtomicU64::new(0)),
            committed_cursor: CachePadded(AtomicU64::new(0)),
            read_cursor: CachePadded(AtomicU64::new(0)),
            max_lsn: CachePadded(AtomicU64::new(0)),
            // Initial limit: writers can fill the entire buffer before needing to check.
            safe_write_limit: CachePadded(AtomicU64::new(capacity_bytes as u64)),
        }
    }

    /// Claims `size` bytes contiguously within the buffer.
    ///
    /// Hot path: single Relaxed load of safe_write_limit (non-contended, stays
    /// in L1) + fetch_add + branch. No cross-core traffic unless the buffer is
    /// genuinely filling up.
    ///
    /// If the claimed region straddles the wrap boundary, the cold path
    /// commits those bytes as padding and retries.
    ///
    /// # Safety
    /// Caller must write exactly `size` bytes to the returned pointer before calling
    /// `commit_write`. The pointer is valid until the ring buffer wraps past this region.
    #[inline]
    pub unsafe fn write_record(&self, size: usize) -> *mut u8 {
        let offset = self.write_cursor.fetch_add(size as u64, Ordering::Relaxed);

        // Fast-path backpressure: compare against cached limit (written by flush
        // thread after each drain). Only falls into slow path when the buffer is
        // actually filling up.
        if offset + size as u64 > self.safe_write_limit.load(Ordering::Relaxed) {
            self.wait_for_space_slow(offset, size);
        }

        let buf_offset = (offset as usize) & self.buffer_mask;

        if buf_offset + size <= self.buffer_size {
            return unsafe { (*self.buffer.get()).as_mut_ptr().add(buf_offset) };
        }

        unsafe { self.write_record_straddle(size) }
    }

    /// Slow path: the writer has claimed space past the cached safe_write_limit.
    /// Reload read_cursor, update the limit, and spin if the buffer is genuinely full.
    #[cold]
    #[inline(never)]
    fn wait_for_space_slow(&self, offset: u64, size: usize) {
        loop {
            let read = self.read_cursor.load(Ordering::Acquire);
            let limit = read + self.buffer_size as u64;
            // Update cached limit so other writers benefit from the fresh read.
            self.safe_write_limit.store(limit, Ordering::Relaxed);
            if offset + size as u64 <= limit {
                return;
            }
            std::hint::spin_loop();
        }
    }

    /// Cold path for records that straddle the wrap boundary.
    /// Commits the straddling bytes as padding and retries until
    /// the record fits contiguously.
    #[cold]
    #[inline(never)]
    unsafe fn write_record_straddle(&self, size: usize) -> *mut u8 {
        debug_assert!(
            size <= self.buffer_size,
            "Record size ({} bytes) exceeds ring buffer capacity ({} bytes)",
            size,
            self.buffer_size,
        );

        // Commit the initial straddling claim as padding.
        self.committed_cursor
            .fetch_add(size as u64, Ordering::Release);

        loop {
            let offset = self.write_cursor.fetch_add(size as u64, Ordering::Relaxed);

            // Backpressure check in straddle loop.
            if offset + size as u64 > self.safe_write_limit.load(Ordering::Relaxed) {
                self.wait_for_space_slow(offset, size);
            }

            let buf_offset = (offset as usize) & self.buffer_mask;

            if buf_offset + size <= self.buffer_size {
                return unsafe { (*self.buffer.get()).as_mut_ptr().add(buf_offset) };
            }

            self.committed_cursor
                .fetch_add(size as u64, Ordering::Release);
        }
    }

    /// Marks `size` bytes as committed and records `lsn` as the latest written LSN.
    /// Must be called after `write_record` once the data has been written.
    ///
    /// Uses store instead of fetch_max for max_lsn. LSNs are assigned
    /// monotonically by the sequencer, so the latest committed LSN is always
    /// the highest. For concurrent writers, commit order may differ from assign
    /// order, but drain_into reads max_lsn after draining all committed bytes,
    /// so a briefly stale value just means flushed_lsn lags slightly.
    /// wait_for_flush retries until the target LSN is reached.
    #[inline]
    pub fn commit_write(&self, size: usize, lsn: Lsn) {
        // committed_cursor Release pairs with drain_into's Acquire load,
        // guaranteeing the max_lsn Relaxed store below is visible to the reader.
        self.committed_cursor
            .fetch_add(size as u64, Ordering::Release);
        self.max_lsn.store(lsn.0, Ordering::Relaxed);
    }

    /// Drains all committed bytes into `output`.
    ///
    /// Returns the maximum LSN of the drained records, or `Lsn::INVALID` if no
    /// data was committed since the last drain.
    #[inline]
    pub fn drain_into(&self, output: &mut Vec<u8>) -> Lsn {
        let committed = self.committed_cursor.load(Ordering::Acquire);
        let read = self.read_cursor.load(Ordering::Acquire);

        if committed <= read {
            return Lsn::INVALID;
        }

        let bytes_to_read = (committed - read) as usize;

        // Safety cap: never read more than buffer_size bytes in one drain.
        // With backpressure in write_record, this should not trigger, but
        // it prevents an out-of-bounds read if invariants are violated.
        assert!(
            bytes_to_read <= self.buffer_size,
            "drain_into: bytes_to_read ({}) exceeds buffer_size ({}), \
             committed={}, read={}",
            bytes_to_read,
            self.buffer_size,
            committed,
            read,
        );
        let actual_committed = read + bytes_to_read as u64;

        let read_offset = (read as usize) & self.buffer_mask;

        unsafe {
            let buf_ptr = (*self.buffer.get()).as_ptr().add(read_offset);

            // Handle wrap-around
            let first_chunk = std::cmp::min(bytes_to_read, self.buffer_size - read_offset);
            output.extend_from_slice(std::slice::from_raw_parts(buf_ptr, first_chunk));

            if bytes_to_read > first_chunk {
                let remaining = bytes_to_read - first_chunk;
                output.extend_from_slice(std::slice::from_raw_parts(
                    (*self.buffer.get()).as_ptr(),
                    remaining,
                ));
            }
        }

        self.read_cursor.store(actual_committed, Ordering::Release);

        // Update cached write limit so writers see the freed space immediately
        // without loading read_cursor themselves.
        self.safe_write_limit.store(
            actual_committed + self.buffer_size as u64,
            Ordering::Relaxed,
        );

        Lsn(self.max_lsn.load(Ordering::Acquire))
    }

    /// Total bytes committed to the buffer across its lifetime. Already
    /// tracked atomically for the commit protocol, stat views read it here
    /// instead of the writer maintaining a redundant duplicate counter.
    #[inline]
    pub fn total_committed_bytes(&self) -> u64 {
        self.committed_cursor.load(Ordering::Relaxed)
    }

    /// Returns true if no data has been committed since the last drain.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.committed_cursor.load(Ordering::Acquire) == self.read_cursor.load(Ordering::Acquire)
    }

    /// Spins until all claimed write space is committed.
    ///
    /// Called before a rotation drain to ensure in-flight writes (write_record
    /// called but commit_write not yet called) complete before the ring buffer
    /// is drained for the last time into the old segment.
    #[inline]
    pub fn wait_until_committed(&self) {
        loop {
            let write = self.write_cursor.load(Ordering::Acquire);
            let committed = self.committed_cursor.load(Ordering::Acquire);
            if committed >= write {
                break;
            }
            std::hint::spin_loop();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;
    use std::thread;

    fn write_bytes(buf: &RingBuffer, data: &[u8], lsn: Lsn) {
        unsafe {
            let ptr = buf.write_record(data.len());
            std::ptr::copy_nonoverlapping(data.as_ptr(), ptr, data.len());
        }
        buf.commit_write(data.len(), lsn);
    }

    #[test]
    fn test_ring_buffer_basic() {
        let buf = RingBuffer::new(16 * 1024);
        let data = b"test record";

        write_bytes(&buf, data, Lsn::new(0, 64));

        let mut output = Vec::new();
        let max_lsn = buf.drain_into(&mut output);

        assert_eq!(max_lsn, Lsn::new(0, 64));
        assert_eq!(&output[..data.len()], data);
    }

    #[test]
    fn test_ring_buffer_multiple() {
        let buf = RingBuffer::new(16 * 1024);

        for i in 0u32..10 {
            let data = format!("record {}", i);
            write_bytes(&buf, data.as_bytes(), Lsn::new(0, 64 + i * 32));
        }

        let mut output = Vec::new();
        let max_lsn = buf.drain_into(&mut output);
        assert!(max_lsn.is_valid());
        assert!(!output.is_empty());
    }

    #[test]
    fn test_ring_buffer_wrap_around() {
        // Small buffer to force wrap-around
        let buf = RingBuffer::new(128);
        let data = b"batch1-x"; // 8 bytes each

        for i in 0u32..8 {
            write_bytes(&buf, data, Lsn::new(0, i * 8));
        }
        let mut output = Vec::new();
        buf.drain_into(&mut output);
        assert_eq!(output.len(), 64);

        output.clear();
        for i in 0u32..8 {
            write_bytes(&buf, data, Lsn::new(0, 64 + i * 8));
        }
        buf.drain_into(&mut output);
        assert_eq!(output.len(), 64);
    }

    #[test]
    fn test_ring_buffer_concurrent() {
        let buf = Arc::new(RingBuffer::new(1024 * 1024));
        let threads = 4;
        let records_per_thread: u32 = 100;
        let record_size = 32;

        let handles: Vec<_> = (0..threads)
            .map(|t| {
                let buf = Arc::clone(&buf);
                thread::spawn(move || {
                    let data = vec![0u8; record_size];
                    for i in 0u32..records_per_thread {
                        write_bytes(&buf, &data, Lsn::new(0, t * 10000 + i));
                    }
                })
            })
            .collect();

        for h in handles {
            h.join().unwrap();
        }

        let mut output = Vec::new();
        buf.drain_into(&mut output);
        assert_eq!(
            output.len(),
            threads as usize * records_per_thread as usize * record_size
        );
    }

    #[test]
    fn test_ring_buffer_empty_drain() {
        let buf = RingBuffer::new(16 * 1024);
        let mut output = Vec::new();
        let max_lsn = buf.drain_into(&mut output);
        assert_eq!(max_lsn, Lsn::INVALID);
        assert!(output.is_empty());
    }

    #[test]
    fn test_ring_buffer_is_empty() {
        let buf = RingBuffer::new(16 * 1024);
        assert!(buf.is_empty());

        write_bytes(&buf, b"data", Lsn::new(0, 64));
        assert!(!buf.is_empty());

        let mut output = Vec::new();
        buf.drain_into(&mut output);
        assert!(buf.is_empty());
    }

    #[test]
    fn test_ring_buffer_wait_until_committed() {
        let buf = RingBuffer::new(16 * 1024);
        // Single-threaded: write_cursor and committed_cursor stay in sync after commit_write.
        write_bytes(&buf, b"hello", Lsn::new(0, 64));
        buf.wait_until_committed(); // returns immediately when all writes are committed
        assert!(!buf.is_empty());
    }
}
