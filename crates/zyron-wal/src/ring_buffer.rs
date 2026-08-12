//! High-performance ring buffer for WAL writes.
//!
//! Uses a contiguous byte buffer with atomic cursors.
//! Writers claim space with a single fetch_add and write directly.

use crate::constants::{CHECKSUM_SIZE, HEADER_SIZE, OFF_LSN, OFF_PAYLOAD_LEN, OFF_RECORD_TYPE};
use crate::record::{LogRecordType, Lsn};
use std::cell::UnsafeCell;
use std::sync::OnceLock;
use std::sync::atomic::{AtomicU8, AtomicU64, Ordering};
use zyron_common::profile::{self, Phase};

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

/// Spins a writer takes before it wakes the drain. Long enough that a
/// drain already running catches up without paying an unpark, short
/// enough that a parked one is woken in microseconds.
const SPIN_BEFORE_WAKING_DRAIN: u32 = 256;

pub struct RingBuffer {
    /// The draining thread, registered as it starts. A writer that fills
    /// the ring wakes it rather than spinning against it while it sleeps.
    drain_waker: OnceLock<std::thread::Thread>,
    /// Contiguous byte buffer.
    buffer: UnsafeCell<Box<[u8]>>,
    /// Buffer size in bytes.
    buffer_size: usize,
    /// Bitmask for power-of-2 modulo (buffer_size - 1). Bitwise AND replaces
    /// integer division for offset calculation: 1 cycle vs 20-40 cycles on x86.
    buffer_mask: usize,
    /// log2(buffer_size). The wrap generation of an absolute offset is
    /// offset >> buffer_shift, used to stamp and validate per-record publish
    /// markers so a stale byte from a previous wrap can never be read as
    /// written data.
    buffer_shift: u32,
    /// Per-byte-offset publish markers, one entry per ring byte. A producer
    /// stamps published[offset & mask] with the wrap-generation marker of the
    /// record claimed at that offset, as a Release store after the record's
    /// bytes are written. The flush thread (sole consumer) reads them with
    /// Acquire to compute the contiguous written watermark. A never-written or
    /// previous-generation slot carries a different marker, so the consumer
    /// never advances over an unwritten slot. This replaces in-order producer
    /// publish: producers no longer wait on each other, so commit throughput
    /// does not convoy under high concurrency.
    published: Box<[AtomicU8]>,
    /// Write cursor: next byte offset to claim. Hit by every writer claim,
    /// isolated on its own cache line so flush-thread reads of other cursors
    /// don't force writers to refetch.
    write_cursor: CachePadded<AtomicU64>,
    /// Committed cursor - the contiguous written watermark. Advanced solely by
    /// the flush thread in advance_committed by walking publish markers; the
    /// drain reads [read_cursor, committed_cursor).
    committed_cursor: CachePadded<AtomicU64>,
    /// Maximum LSN written, computed by the flush thread as it advances the
    /// watermark over published records.
    max_lsn: CachePadded<AtomicU64>,
    /// Total non-padding record count drained past the watermark, advanced by
    /// the flush thread so observers (zyron_stat_wal, tests) see counts without
    /// the writer maintaining a contended counter on the hot path.
    committed_records: CachePadded<AtomicU64>,
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
        // One publish marker per ring byte. Zero means "no record published at
        // this offset for the current generation"; the first generation stamps
        // marker 1, so the initial zero never reads as published.
        let published: Box<[AtomicU8]> = (0..capacity_bytes)
            .map(|_| AtomicU8::new(0))
            .collect::<Vec<_>>()
            .into_boxed_slice();

        Self {
            drain_waker: OnceLock::new(),
            buffer: UnsafeCell::new(buffer),
            buffer_size: capacity_bytes,
            buffer_mask: capacity_bytes - 1,
            buffer_shift: capacity_bytes.trailing_zeros(),
            published,
            write_cursor: CachePadded(AtomicU64::new(0)),
            committed_cursor: CachePadded(AtomicU64::new(0)),
            read_cursor: CachePadded(AtomicU64::new(0)),
            max_lsn: CachePadded(AtomicU64::new(0)),
            committed_records: CachePadded(AtomicU64::new(0)),
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
    /// `commit_write` with the returned claim offset. The pointer is valid until the
    /// ring buffer wraps past this region.
    #[inline]
    pub unsafe fn write_record(&self, size: usize) -> (*mut u8, u64) {
        let offset = self.write_cursor.fetch_add(size as u64, Ordering::Relaxed);

        // Fast-path backpressure: compare against cached limit (written by flush
        // thread after each drain). Only falls into slow path when the buffer is
        // actually filling up.
        if offset + size as u64 > self.safe_write_limit.load(Ordering::Relaxed) {
            self.wait_for_space_slow(offset, size);
        }

        let buf_offset = (offset as usize) & self.buffer_mask;

        if buf_offset + size <= self.buffer_size {
            return (
                unsafe { (*self.buffer.get()).as_mut_ptr().add(buf_offset) },
                offset,
            );
        }

        unsafe { self.write_record_straddle(offset, size) }
    }

    /// Wrap-generation marker for a record claimed at absolute `offset`, in the
    /// range 1..=255. The generation is offset / buffer_size; the marker is
    /// (gen mod 255) + 1, so it is never 0. The consumer resets a slot to 0
    /// after walking past it (see advance_committed), so a slot is non-zero only
    /// during the live publish window of its current generation: 0 means
    /// unpublished, a non-zero value means published this generation. The
    /// generation in the marker is a second layer of defense; the reset alone
    /// makes the scan correct regardless of how the byte's record boundaries
    /// shift across wraps.
    #[inline]
    fn gen_marker(&self, offset: u64) -> u8 {
        (((offset >> self.buffer_shift) % 255) as u8) + 1
    }

    /// Publishes the record claimed at `offset`, wait-free. Stamps the publish
    /// marker for this offset with a Release store after the caller has written
    /// the record's bytes, so the flush thread that reads the marker with
    /// Acquire also observes the record bytes. Producers do not wait on each
    /// other; the flush thread computes the contiguous written watermark in
    /// advance_committed.
    #[inline]
    pub fn publish(&self, offset: u64) {
        let slot = (offset as usize) & self.buffer_mask;
        self.published[slot].store(self.gen_marker(offset), Ordering::Release);
    }

    /// Stamps a valid padding record (LogRecordType::Invalid) of `size` bytes
    /// at logical `offset`, handling the wrap boundary byte by byte. Only the
    /// record_type and payload_len header fields are written; the rest of the
    /// header and the payload are left as-is and the checksum is backfilled by
    /// the flush thread before the segment write. That is self-consistent
    /// because the checksum is computed over whatever bytes are drained, and
    /// recovery skips a padding record's content entirely. Without this the
    /// straddle gap would drain as stale bytes that break recovery's parser.
    #[inline]
    unsafe fn write_padding_record(&self, offset: u64, size: usize) {
        debug_assert!(
            size >= HEADER_SIZE + CHECKSUM_SIZE,
            "padding region {} smaller than an empty record",
            size
        );
        let payload_len = (size - HEADER_SIZE - CHECKSUM_SIZE) as u16;
        let buf = unsafe { (*self.buffer.get()).as_mut_ptr() };
        let mask = self.buffer_mask;
        let off = offset as usize;
        unsafe {
            *buf.add((off + OFF_RECORD_TYPE) & mask) = LogRecordType::Invalid as u8;
            let pl = payload_len.to_le_bytes();
            *buf.add((off + OFF_PAYLOAD_LEN) & mask) = pl[0];
            *buf.add((off + OFF_PAYLOAD_LEN + 1) & mask) = pl[1];
        }
    }

    /// Registers the draining thread so a writer that fills the ring can
    /// wake it. Called once by the flush thread as it starts.
    pub fn register_drain_thread(&self) {
        self.drain_waker.set(std::thread::current()).ok();
    }

    /// Slow path: the writer has claimed space past the cached safe_write_limit.
    /// Reload read_cursor, update the limit, and wait if the buffer is genuinely full.
    ///
    /// The drain parks between wakeups and is woken on commit, so a
    /// statement large enough to fill the ring before it commits will find
    /// it asleep. Spinning against a sleeping thread is a livelock, not a
    /// wait: it holds the core the drain needs to free the space being
    /// waited for. So the spin is bounded, and past it this wakes the
    /// drain and yields, which turns a hang into backpressure.
    #[cold]
    #[inline(never)]
    fn wait_for_space_slow(&self, offset: u64, size: usize) {
        let fits = |limit: u64| offset + size as u64 <= limit;
        // Bounded spin first. The drain usually catches up inside this,
        // and an unpark plus a scheduler dispatch costs far more
        for _ in 0..SPIN_BEFORE_WAKING_DRAIN {
            let limit = self.refresh_write_limit();
            if fits(limit) {
                return;
            }
            std::hint::spin_loop();
        }
        loop {
            if let Some(drain) = self.drain_waker.get() {
                drain.unpark();
            }
            std::thread::yield_now();
            if fits(self.refresh_write_limit()) {
                return;
            }
        }
    }

    /// Rereads the consumer's cursor and republishes the limit every
    /// writer caches, so one writer's reload serves the rest
    #[inline]
    fn refresh_write_limit(&self) -> u64 {
        let limit = self.read_cursor.load(Ordering::Acquire) + self.buffer_size as u64;
        self.safe_write_limit.store(limit, Ordering::Relaxed);
        limit
    }

    /// Cold path for records that straddle the wrap boundary.
    /// Commits the straddling bytes as padding and retries until
    /// the record fits contiguously.
    #[cold]
    #[inline(never)]
    unsafe fn write_record_straddle(&self, first_offset: u64, size: usize) -> (*mut u8, u64) {
        debug_assert!(
            size <= self.buffer_size,
            "Record size ({} bytes) exceeds ring buffer capacity ({} bytes)",
            size,
            self.buffer_size,
        );

        // Stamp the initial straddling claim with a padding record and publish
        // it. The header is written before the publish marker store, so the
        // flush thread that observes the marker never drains stale bytes for it.
        unsafe { self.write_padding_record(first_offset, size) };
        self.publish(first_offset);

        loop {
            let offset = self.write_cursor.fetch_add(size as u64, Ordering::Relaxed);

            // Backpressure check in straddle loop.
            if offset + size as u64 > self.safe_write_limit.load(Ordering::Relaxed) {
                self.wait_for_space_slow(offset, size);
            }

            let buf_offset = (offset as usize) & self.buffer_mask;

            if buf_offset + size <= self.buffer_size {
                // Real record fits here; its publish happens via publish(offset)
                // after the caller writes the record bytes.
                return (
                    unsafe { (*self.buffer.get()).as_mut_ptr().add(buf_offset) },
                    offset,
                );
            }

            // This claim also straddles: stamp it as padding and publish it.
            unsafe { self.write_padding_record(offset, size) };
            self.publish(offset);
        }
    }

    /// Advances the contiguous written watermark (committed_cursor) over every
    /// record that producers have published since the last call. Called only by
    /// the flush thread (the sole consumer), so committed_cursor, max_lsn and
    /// committed_records have a single writer and need no RMW.
    ///
    /// Walks records from the current watermark: at each record start it reads
    /// the publish marker with Acquire and stops at the first slot whose marker
    /// does not match the expected generation (an unpublished or
    /// previous-generation slot). A matching marker guarantees the record's
    /// header and payload bytes are visible, so it reads the record length to
    /// step to the next record. Reads are wrap-aware so a straddling padding
    /// record's header (written across the ring boundary) is parsed correctly.
    pub fn advance_committed(&self) {
        let mask = self.buffer_mask;
        let write = self.write_cursor.load(Ordering::Acquire);
        let mut w = self.committed_cursor.load(Ordering::Relaxed);
        if w >= write {
            return;
        }
        let buf = unsafe { (*self.buffer.get()).as_ptr() };
        let mut max_lsn = self.max_lsn.load(Ordering::Relaxed);
        let mut max_lsn_dirty = false;
        let mut new_records: u64 = 0;
        let mut advanced = false;

        while w < write {
            let slot = (w as usize) & mask;
            if self.published[slot].load(Ordering::Acquire) != self.gen_marker(w) {
                // Record at this offset is not yet published; the watermark
                // stops here until its producer stamps the marker.
                break;
            }
            // Reset the marker now that the watermark covers this record. The
            // slot cannot be reused by a later generation until the drain
            // advances read_cursor past it (backpressure gates reuse on
            // read_cursor, not on this marker), so resetting here is race-free
            // and leaves the slot at 0 = unpublished for its next generation.
            self.published[slot].store(0, Ordering::Relaxed);
            // SAFETY: the Acquire marker load above synchronizes with the
            // producer's Release publish, so this record's header bytes are
            // visible. Byte reads are masked to handle a header that wraps the
            // ring boundary (straddling padding record).
            let base = w as usize;
            let record_type = unsafe { *buf.add((base + OFF_RECORD_TYPE) & mask) };
            let pl_lo = unsafe { *buf.add((base + OFF_PAYLOAD_LEN) & mask) };
            let pl_hi = unsafe { *buf.add((base + OFF_PAYLOAD_LEN + 1) & mask) };
            let payload_len = u16::from_le_bytes([pl_lo, pl_hi]) as usize;
            let record_size = HEADER_SIZE + payload_len + CHECKSUM_SIZE;

            // Padding records (LogRecordType::Invalid == 0) carry no LSN and are
            // not counted; only real records advance max_lsn and the count.
            if record_type != 0 {
                let mut lsn_bytes = [0u8; 8];
                for (i, b) in lsn_bytes.iter_mut().enumerate() {
                    *b = unsafe { *buf.add((base + OFF_LSN + i) & mask) };
                }
                let lsn = u64::from_le_bytes(lsn_bytes);
                if lsn > max_lsn {
                    max_lsn = lsn;
                    max_lsn_dirty = true;
                }
                new_records += 1;
            }

            w += record_size as u64;
            advanced = true;
        }

        if advanced {
            if max_lsn_dirty {
                self.max_lsn.store(max_lsn, Ordering::Relaxed);
            }
            if new_records > 0 {
                self.committed_records
                    .fetch_add(new_records, Ordering::Relaxed);
            }
            // Release so a drain that Acquire-loads committed_cursor observes
            // all the record bytes the watermark now covers.
            self.committed_cursor.store(w, Ordering::Release);
        }
    }

    /// Drains all committed bytes into `output`.
    ///
    /// Returns the maximum LSN of the drained records, or `Lsn::INVALID` if no
    /// data was committed since the last drain.
    #[inline]
    pub fn drain_into(&self, output: &mut Vec<u8>) -> Lsn {
        // Advance the contiguous watermark over newly published records before
        // copying, so a caller that drains directly (tests, rotation residual)
        // sees all published data without a separate advance call.
        {
            let _s = profile::scope(Phase::FlushAdvance);
            self.advance_committed();
        }
        let _drain_span = profile::scope(Phase::FlushDrain);
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

    /// Total records committed to the buffer across its lifetime, updated
    /// synchronously in commit_write so observers see the count immediately
    #[inline]
    pub fn total_committed_records(&self) -> u64 {
        self.committed_records.load(Ordering::Relaxed)
    }

    /// Returns true when the ring is fully quiescent: every claimed byte has
    /// been drained. Used by the flush loop's idle check and by flush()/shutdown
    /// to know nothing remains to write.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.write_cursor.load(Ordering::Acquire) == self.read_cursor.load(Ordering::Acquire)
    }

    /// Returns true when producers have claimed space not yet drained. A hint
    /// for the flush thread: there is eventual work even if those records are
    /// not published yet (the producer is mid-write between claim and publish).
    #[inline]
    pub fn has_pending(&self) -> bool {
        self.write_cursor.load(Ordering::Acquire) > self.read_cursor.load(Ordering::Acquire)
    }

    /// Returns true when the watermark has advanced past the read cursor, i.e.
    /// there are published records ready to copy to disk. Reads committed_cursor
    /// as last advanced by advance_committed; call that first to refresh it.
    #[inline]
    pub fn has_drainable(&self) -> bool {
        self.committed_cursor.load(Ordering::Acquire) > self.read_cursor.load(Ordering::Acquire)
    }

    /// Advances the watermark, then spins until every claimed byte is published
    /// and covered by it. Called by the flush thread before a rotation drain so
    /// in-flight writes (claimed via write_record but not yet published)
    /// complete before the ring is drained into the old segment.
    #[inline]
    pub fn wait_until_committed(&self) {
        loop {
            self.advance_committed();
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
    use crate::record::{LogRecord, record_size_for_payload, serialize_raw_deferred};
    use std::sync::Arc;
    use std::thread;

    /// Writes one valid WAL record with `payload` as its body. The watermark
    /// parses record headers, so tests must write real records, not raw bytes.
    /// Returns the on-disk record size.
    fn write_record(buf: &RingBuffer, payload: &[u8], lsn: Lsn) -> usize {
        let size = record_size_for_payload(payload.len());
        let offset;
        unsafe {
            let (ptr, off) = buf.write_record(size);
            offset = off;
            serialize_raw_deferred(
                ptr,
                lsn,
                Lsn::INVALID,
                1,
                LogRecordType::Insert as u8,
                0,
                payload,
            );
        }
        buf.publish(offset);
        size
    }

    #[test]
    fn test_ring_buffer_basic() {
        let buf = RingBuffer::new(16 * 1024);
        let payload = b"test record";

        write_record(&buf, payload, Lsn::new(0, 64));

        let mut output = Vec::new();
        let max_lsn = buf.drain_into(&mut output);

        assert_eq!(max_lsn, Lsn::new(0, 64));
        // Checksums are deferred (zero placeholder); parse without verification.
        let recs = LogRecord::parse_all_trusted(bytes::Bytes::from(output));
        assert_eq!(recs.len(), 1);
        assert_eq!(&recs[0].payload[..], payload);
    }

    #[test]
    fn test_ring_buffer_multiple() {
        let buf = RingBuffer::new(16 * 1024);

        for i in 0u32..10 {
            let data = format!("record {}", i);
            write_record(&buf, data.as_bytes(), Lsn::new(0, 64 + i * 32));
        }

        let mut output = Vec::new();
        let max_lsn = buf.drain_into(&mut output);
        assert!(max_lsn.is_valid());
        let recs = LogRecord::parse_all_trusted(bytes::Bytes::from(output));
        assert_eq!(recs.len(), 10);
    }

    #[test]
    fn test_ring_buffer_wrap_around() {
        // Small ring to force wrap-around and straddle padding. Drain after each
        // record so the single-threaded writer never blocks on backpressure
        // (no flush thread here to advance the read cursor). 8-byte payload =>
        // 36-byte record; over 30 writes the 256-byte ring wraps several times,
        // exercising the straddle padding and the watermark walk over it.
        let buf = RingBuffer::new(256);
        let payload = b"batch1-x";

        for n in 0u32..30 {
            write_record(&buf, payload, Lsn::new(0, n + 1));
            let mut output = Vec::new();
            buf.drain_into(&mut output);
            // The drain may include a leading Invalid padding record when the
            // real record straddled the wrap boundary; the real record must be
            // present with its payload intact.
            let recs = LogRecord::parse_all_trusted(bytes::Bytes::from(output));
            let real: Vec<_> = recs
                .iter()
                .filter(|r| r.record_type == LogRecordType::Insert)
                .collect();
            assert_eq!(real.len(), 1, "exactly one real record per drain");
            assert_eq!(&real[0].payload[..], payload);
        }
    }

    #[test]
    fn test_ring_buffer_concurrent() {
        let buf = Arc::new(RingBuffer::new(1024 * 1024));
        let threads = 4;
        let records_per_thread: u32 = 100;
        let payload = vec![0u8; 32];
        let record_size = record_size_for_payload(payload.len());

        let handles: Vec<_> = (0..threads)
            .map(|t| {
                let buf = Arc::clone(&buf);
                let payload = payload.clone();
                thread::spawn(move || {
                    for i in 0u32..records_per_thread {
                        write_record(&buf, &payload, Lsn::new(0, t * 10000 + i + 1));
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
        let recs = LogRecord::parse_all_trusted(bytes::Bytes::from(output));
        assert_eq!(recs.len(), threads as usize * records_per_thread as usize);
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

        write_record(&buf, b"data", Lsn::new(0, 64));
        assert!(!buf.is_empty());

        let mut output = Vec::new();
        buf.drain_into(&mut output);
        assert!(buf.is_empty());
    }

    #[test]
    fn test_ring_buffer_wait_until_committed() {
        let buf = RingBuffer::new(16 * 1024);
        write_record(&buf, b"hello", Lsn::new(0, 64));
        buf.wait_until_committed();
        // Published but not drained: claimed bytes still outstanding.
        assert!(buf.has_pending());
    }
}
