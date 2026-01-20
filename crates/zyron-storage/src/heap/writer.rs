//! Buffered heap writer with zero-copy parallel inserts.
//!
//! Stages tuples in a write buffer and flushes them in batches using:
//! - Bin-packing to assign tuples to pages
//! - Zero-copy writes directly to buffer pool frames
//! - Parallel page writes using scoped threads

use super::file::HeapFile;
use super::page::{HeapPage, TupleSlot};
use crate::tuple::{Tuple, TupleId};
use std::collections::HashMap;
use std::sync::Arc;
use std::thread;
use std::time::Instant;
use zyron_common::page::{PAGE_SIZE, PageId};
use zyron_common::{Result, ZyronError};

/// Default buffer capacity (1MB).
const DEFAULT_BUFFER_CAPACITY: usize = 1024 * 1024;

/// Staged tuple waiting to be written.
struct StagedTuple {
    /// Pre-serialized tuple bytes (header + data).
    serialized: Vec<u8>,
}

/// Statistics for the write buffer.
#[derive(Debug, Default, Clone)]
pub struct WriteBufferStats {
    /// Number of flush operations performed.
    pub flush_count: u64,
    /// Total tuples written.
    pub tuples_written: u64,
    /// Total time spent in flush operations (nanoseconds).
    pub flush_time_ns: u64,
    /// Time spent bin-packing tuples (nanoseconds).
    pub bin_pack_time_ns: u64,
    /// Time spent writing pages (nanoseconds).
    pub page_write_time_ns: u64,
}

/// Buffered heap writer for high-throughput inserts.
///
/// Stages tuples in memory and writes them in batches using zero-copy
/// operations directly on buffer pool frames. Parallel writes to different
/// pages maximize throughput.
pub struct BufferedHeapWriter {
    /// Heap file to write to.
    heap: Arc<HeapFile>,
    /// Staged tuples waiting to be flushed.
    buffer: Vec<StagedTuple>,
    /// Current size of staged data in bytes.
    staged_bytes: usize,
    /// Maximum buffer capacity in bytes.
    capacity_bytes: usize,
    /// Write statistics.
    stats: WriteBufferStats,
}

impl BufferedHeapWriter {
    /// Creates a new buffered writer with default capacity (1MB).
    pub fn new(heap: Arc<HeapFile>) -> Self {
        Self::with_capacity(heap, DEFAULT_BUFFER_CAPACITY)
    }

    /// Creates a new buffered writer with specified capacity in bytes.
    pub fn with_capacity(heap: Arc<HeapFile>, capacity_bytes: usize) -> Self {
        Self {
            heap,
            buffer: Vec::with_capacity(capacity_bytes / 64), // Estimate ~64 bytes avg tuple
            staged_bytes: 0,
            capacity_bytes,
            stats: WriteBufferStats::default(),
        }
    }

    /// Returns the current write statistics.
    pub fn stats(&self) -> &WriteBufferStats {
        &self.stats
    }

    /// Returns the number of staged tuples.
    pub fn staged_count(&self) -> usize {
        self.buffer.len()
    }

    /// Returns the total bytes staged.
    pub fn staged_bytes(&self) -> usize {
        self.staged_bytes
    }

    /// Stages a tuple for batched insert.
    ///
    /// Auto-flushes when buffer reaches capacity.
    pub async fn stage(&mut self, tuple: &Tuple) -> Result<()> {
        // Pre-serialize the tuple once (avoids repeated serialization in write path)
        let serialized = tuple.serialize();
        let size = serialized.len();

        if self.staged_bytes + size > self.capacity_bytes {
            self.flush().await?;
        }

        self.buffer.push(StagedTuple { serialized });
        self.staged_bytes += size;

        Ok(())
    }

    /// Stages multiple tuples for batched insert.
    ///
    /// Auto-flushes when buffer reaches capacity.
    pub async fn stage_batch(&mut self, tuples: &[Tuple]) -> Result<()> {
        for tuple in tuples {
            self.stage(tuple).await?;
        }
        Ok(())
    }

    /// Stages all tuples synchronously without any async overhead.
    ///
    /// Pre-serializes all tuples into the buffer. Call flush() after to write.
    /// This is the fastest path for bulk inserts where you control buffer size.
    #[inline]
    pub fn stage_all_sync(&mut self, tuples: &[Tuple]) {
        self.buffer.reserve(tuples.len());
        for tuple in tuples {
            let serialized = tuple.serialize();
            self.staged_bytes += serialized.len();
            self.buffer.push(StagedTuple { serialized });
        }
    }

    /// Flushes all staged tuples using zero-copy parallel writes.
    ///
    /// Returns the TupleIds of all inserted tuples.
    pub async fn flush(&mut self) -> Result<Vec<TupleId>> {
        if self.buffer.is_empty() {
            return Ok(Vec::new());
        }

        let flush_start = Instant::now();

        // Step 1: Bin-pack tuples into page assignments
        // New pages are allocated and loaded into pool (already pinned)
        let bin_start = Instant::now();
        let (page_batches, new_page_ids) = self.bin_pack_tuples().await?;
        self.stats.bin_pack_time_ns += bin_start.elapsed().as_nanos() as u64;

        // All pages are new pages (already pinned by new_page())
        // No need for batch_pin since all pages were just created

        // Step 2: Zero-copy writes (parallel for multiple pages)
        let write_start = Instant::now();
        let results = self.write_pages(&page_batches)?;
        self.stats.page_write_time_ns += write_start.elapsed().as_nanos() as u64;

        // Step 3: Batch unpin and mark dirty (all pages are new and pinned)
        self.heap.pool().batch_unpin_dirty(&new_page_ids);

        // Step 4: Batch FSM update
        self.heap.flush_fsm_updates().await?;

        // Clear buffer
        self.buffer.clear();
        self.staged_bytes = 0;
        self.stats.flush_count += 1;
        self.stats.tuples_written += results.len() as u64;
        self.stats.flush_time_ns += flush_start.elapsed().as_nanos() as u64;

        Ok(results)
    }

    /// Assigns tuples to pages using sequential fill (no bin-packing overhead).
    ///
    /// Pre-calculates page count and allocates all pages in ONE async call,
    /// eliminating sequential await overhead. Returns (batches, new_page_ids).
    async fn bin_pack_tuples(&self) -> Result<(HashMap<PageId, Vec<usize>>, Vec<PageId>)> {
        // Step 1: Calculate total space needed and estimate pages required
        let total_tuple_bytes: usize = self
            .buffer
            .iter()
            .map(|t| t.serialized.len() + TupleSlot::SIZE)
            .sum();

        let usable_per_page = PAGE_SIZE - HeapPage::DATA_START;
        // Calculate pages more accurately: account for TupleSlot overhead per tuple
        // Each tuple needs: serialized.len() + TupleSlot::SIZE bytes
        // We already included TupleSlot::SIZE in total_tuple_bytes above
        let base_pages = (total_tuple_bytes + usable_per_page - 1) / usable_per_page;
        // Add 50% buffer to ensure we have enough pages (handles rounding/fragmentation)
        let pages_needed = base_pages + base_pages / 2 + 1;

        if pages_needed == 0 {
            return Ok((HashMap::new(), Vec::new()));
        }

        // Step 2: SINGLE async call to allocate all pages (eliminates N sequential awaits)
        let heap_file_id = self.heap.heap_file_id();
        let allocated = self
            .heap
            .disk()
            .allocate_pages_batch(heap_file_id, pages_needed as u32)
            .await?;

        // Step 3: Batch initialize frames (sync, single pass)
        let pool = self.heap.pool();
        let (frames, evicted) = pool.batch_new_pages(&allocated)?;

        // Flush any evicted dirty pages
        for ev in evicted {
            self.heap.disk().write_page(ev.page_id, &*ev.data).await?;
        }

        // Initialize all page headers
        for (i, &page_id) in allocated.iter().enumerate() {
            let new_page = HeapPage::new(page_id);
            frames[i].copy_from(new_page.as_bytes());
        }

        // Update heap page count
        for _ in 0..pages_needed {
            self.heap.increment_heap_page_count();
        }

        // Step 4: Sequential fill (no sorting, no bin-packing - just like insert_batch)
        let mut batches: HashMap<PageId, Vec<usize>> = HashMap::new();
        let mut current_page_idx = 0;
        let mut current_remaining = usable_per_page;

        for idx in 0..self.buffer.len() {
            let space_needed = self.buffer[idx].serialized.len() + TupleSlot::SIZE;

            // Move to next page if current doesn't have space
            if current_remaining < space_needed {
                current_page_idx += 1;
                current_remaining = usable_per_page;
            }

            current_remaining -= space_needed;
            batches
                .entry(allocated[current_page_idx])
                .or_default()
                .push(idx);
        }

        Ok((batches, allocated))
    }

    /// Writes tuples to pages using zero-copy operations.
    ///
    /// Uses parallel writes when multiple pages are involved.
    fn write_pages(&self, batches: &HashMap<PageId, Vec<usize>>) -> Result<Vec<TupleId>> {
        let batch_vec: Vec<_> = batches.iter().collect();
        let num_batches = batch_vec.len();

        if num_batches == 0 {
            return Ok(Vec::new());
        }

        if num_batches == 1 {
            // Single page - no parallelism needed
            let (page_id, indices) = batch_vec[0];
            return self.write_single_page(*page_id, indices);
        }

        // Multiple pages - use parallel writes with scoped threads
        let results: Vec<Result<Vec<TupleId>>> = thread::scope(|s| {
            let handles: Vec<_> = batch_vec
                .into_iter()
                .map(|(page_id, indices)| {
                    let pool = self.heap.pool();
                    let heap = &self.heap;
                    let buffer = &self.buffer;
                    let page_id = *page_id;

                    s.spawn(move || {
                        // Get mutable pointer to frame (page already initialized in bin_pack_tuples)
                        let data_ptr = unsafe {
                            pool.frame_data_ptr_mut(page_id)
                                .ok_or(ZyronError::Internal("Frame not found".into()))?
                        };
                        let data = unsafe { &mut *data_ptr };

                        let mut page_results = Vec::with_capacity(indices.len());
                        for &idx in indices {
                            let staged = &buffer[idx];

                            // Direct insert of pre-serialized bytes (no allocation)
                            let (slot_id, _free) =
                                HeapPage::insert_tuple_bytes_in_slice(data, &staged.serialized)?;
                            page_results.push(TupleId::new(page_id, slot_id.0));
                        }

                        // Defer FSM update
                        let free_space = HeapPage::free_space_in_slice(data);
                        heap.defer_fsm_update(page_id.page_num, free_space);

                        Ok(page_results)
                    })
                })
                .collect();

            handles.into_iter().map(|h| h.join().unwrap()).collect()
        });

        // Flatten results
        let mut all_results = Vec::new();
        for r in results {
            all_results.extend(r?);
        }
        Ok(all_results)
    }

    /// Writes tuples to a single page (page already initialized in bin_pack_tuples).
    fn write_single_page(&self, page_id: PageId, indices: &[usize]) -> Result<Vec<TupleId>> {
        let pool = self.heap.pool();

        let data_ptr = unsafe {
            pool.frame_data_ptr_mut(page_id)
                .ok_or(ZyronError::Internal("Frame not found".into()))?
        };
        let data = unsafe { &mut *data_ptr };

        let mut results = Vec::with_capacity(indices.len());
        for &idx in indices {
            let staged = &self.buffer[idx];

            // Direct insert of pre-serialized bytes (no allocation)
            let (slot_id, _free) = HeapPage::insert_tuple_bytes_in_slice(data, &staged.serialized)?;
            results.push(TupleId::new(page_id, slot_id.0));
        }

        // Defer FSM update
        let free_space = HeapPage::free_space_in_slice(data);
        self.heap.defer_fsm_update(page_id.page_num, free_space);

        Ok(results)
    }
}

impl Drop for BufferedHeapWriter {
    fn drop(&mut self) {
        if !self.buffer.is_empty() {
            // Tuples will be lost if not flushed - log warning in debug mode
            #[cfg(debug_assertions)]
            eprintln!(
                "WARNING: BufferedHeapWriter dropped with {} unflushed tuples",
                self.buffer.len()
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::disk::{DiskManager, DiskManagerConfig};
    use tempfile::tempdir;
    use zyron_buffer::{BufferPool, BufferPoolConfig};

    async fn create_test_heap() -> (Arc<HeapFile>, tempfile::TempDir) {
        let dir = tempdir().unwrap();
        let config = DiskManagerConfig {
            data_dir: dir.path().to_path_buf(),
            fsync_enabled: false,
        };
        let disk = Arc::new(DiskManager::new(config).await.unwrap());
        let pool = Arc::new(BufferPool::new(BufferPoolConfig { num_frames: 100 }));
        let heap = Arc::new(HeapFile::with_defaults(disk, pool).unwrap());
        (heap, dir)
    }

    #[tokio::test]
    async fn test_buffered_writer_new() {
        let (heap, _dir) = create_test_heap().await;
        let writer = BufferedHeapWriter::new(heap);
        assert_eq!(writer.staged_count(), 0);
        assert_eq!(writer.staged_bytes(), 0);
    }

    #[tokio::test]
    async fn test_buffered_writer_stage() {
        let (heap, _dir) = create_test_heap().await;
        let mut writer = BufferedHeapWriter::new(heap);

        let tuple = Tuple::new(b"hello world".to_vec(), 1);
        writer.stage(&tuple).await.unwrap();

        assert_eq!(writer.staged_count(), 1);
        assert!(writer.staged_bytes() > 0);
    }

    #[tokio::test]
    async fn test_buffered_writer_flush() {
        let (heap, _dir) = create_test_heap().await;
        let mut writer = BufferedHeapWriter::new(Arc::clone(&heap));

        for i in 0..10 {
            let tuple = Tuple::new(format!("tuple {}", i).into_bytes(), i);
            writer.stage(&tuple).await.unwrap();
        }

        let ids = writer.flush().await.unwrap();
        assert_eq!(ids.len(), 10);
        assert_eq!(writer.staged_count(), 0);

        // Verify tuples are readable
        for (i, id) in ids.iter().enumerate() {
            let tuple = heap.get(*id).await.unwrap().unwrap();
            assert_eq!(tuple.header().xmin, i as u32);
        }
    }

    #[tokio::test]
    async fn test_buffered_writer_auto_flush() {
        let (heap, _dir) = create_test_heap().await;
        // Small buffer to trigger auto-flush
        let mut writer = BufferedHeapWriter::with_capacity(Arc::clone(&heap), 1000);

        // Stage enough tuples to trigger auto-flush
        for i in 0..100 {
            let tuple = Tuple::new(vec![i as u8; 50], i as u32);
            writer.stage(&tuple).await.unwrap();
        }

        // Should have auto-flushed multiple times
        assert!(writer.stats().flush_count > 0);

        // Final flush for any remaining
        writer.flush().await.unwrap();
    }

    #[tokio::test]
    async fn test_buffered_writer_large_batch() {
        let (heap, _dir) = create_test_heap().await;
        let mut writer = BufferedHeapWriter::new(Arc::clone(&heap));

        // Insert enough to span multiple pages
        let tuple_size = 500;
        for i in 0..100 {
            let tuple = Tuple::new(vec![i as u8; tuple_size], i as u32);
            writer.stage(&tuple).await.unwrap();
        }

        let ids = writer.flush().await.unwrap();
        assert_eq!(ids.len(), 100);

        // Verify all tuples readable via scan
        let guard = heap.scan().unwrap();
        assert_eq!(guard.count(), 100);
    }

    #[tokio::test]
    async fn test_buffered_writer_stats() {
        let (heap, _dir) = create_test_heap().await;
        let mut writer = BufferedHeapWriter::new(heap);

        for i in 0..50 {
            let tuple = Tuple::new(format!("tuple {}", i).into_bytes(), i);
            writer.stage(&tuple).await.unwrap();
        }

        writer.flush().await.unwrap();

        let stats = writer.stats();
        assert_eq!(stats.flush_count, 1);
        assert_eq!(stats.tuples_written, 50);
        assert!(stats.flush_time_ns > 0);
    }
}
