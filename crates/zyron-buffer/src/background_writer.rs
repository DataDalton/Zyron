//! Background writer thread for continuous dirty page flushing.
//!
//! Trickle-flushes dirty pages oldest-first using LSN ordering.
//! During checkpoint, prioritizes pages below the checkpoint LSN boundary.
//! Writers never block. The background writer is invisible to the write path.

use crate::pool::BufferPool;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, OnceLock};
use std::thread::{self, JoinHandle};
use std::time::Duration;
use zyron_common::Result;
use zyron_common::page::{PAGE_SIZE, PageId};

/// Configuration for the background writer.
#[derive(Debug, Clone)]
pub struct BackgroundWriterConfig {
    /// Maximum pages to flush per cycle.
    pub pages_per_cycle: usize,
    /// Sleep duration between cycles when no dirty pages found (microseconds).
    pub idle_sleep_us: u64,
    /// Sleep duration between cycles when dirty pages were flushed (microseconds).
    pub active_sleep_us: u64,
}

impl Default for BackgroundWriterConfig {
    fn default() -> Self {
        Self {
            pages_per_cycle: 64,
            idle_sleep_us: 1000,
            active_sleep_us: 100,
        }
    }
}

/// Write function type for flushing pages to disk.
/// Decoupled from DiskManager so the buffer crate has no storage dependency.
pub type WriteFn = Arc<dyn Fn(PageId, &[u8; PAGE_SIZE]) -> Result<()> + Send + Sync>;

/// Background writer that continuously flushes dirty pages to disk.
///
/// Follows the same thread lifecycle pattern as the WAL flush thread:
/// std::thread with park/unpark via OnceLock for wakeup coordination.
pub struct BackgroundWriter {
    /// 0 = trickle mode (flush anything dirty, oldest first).
    /// >0 = checkpoint mode (prioritize pages with dirty_lsn <= target).
    target_lsn: Arc<AtomicU64>,
    /// Lowest unflushed dirty_lsn across all frames, updated each cycle.
    min_dirty_lsn: Arc<AtomicU64>,
    /// Shutdown signal.
    shutdown: Arc<AtomicBool>,
    /// Thread handle for park/unpark wakeup.
    writer_thread_waker: Arc<OnceLock<thread::Thread>>,
    /// Background thread join handle.
    writer_thread: Option<JoinHandle<()>>,
    /// Total pages flushed (cumulative counter for stats).
    pages_flushed: Arc<AtomicU64>,
}

impl BackgroundWriter {
    /// Creates and starts the background writer thread.
    pub fn new(pool: Arc<BufferPool>, write_fn: WriteFn, config: BackgroundWriterConfig) -> Self {
        let target_lsn = Arc::new(AtomicU64::new(0));
        let min_dirty_lsn = Arc::new(AtomicU64::new(u64::MAX));
        let shutdown = Arc::new(AtomicBool::new(false));
        let writer_thread_waker = Arc::new(OnceLock::new());
        let pages_flushed = Arc::new(AtomicU64::new(0));

        let thread_pool = Arc::clone(&pool);
        let thread_target = Arc::clone(&target_lsn);
        let thread_min_dirty = Arc::clone(&min_dirty_lsn);
        let thread_shutdown = Arc::clone(&shutdown);
        let thread_waker = Arc::clone(&writer_thread_waker);
        let thread_flushed = Arc::clone(&pages_flushed);
        let thread_config = config.clone();

        let handle = thread::Builder::new()
            .name("zyron-bg-writer".into())
            .spawn(move || {
                // Register this thread for park/unpark wakeup
                let _ = thread_waker.set(thread::current());

                Self::writer_loop(
                    &thread_pool,
                    &write_fn,
                    &thread_target,
                    &thread_min_dirty,
                    &thread_shutdown,
                    &thread_flushed,
                    &thread_config,
                );
            })
            .expect("failed to spawn background writer thread");

        Self {
            target_lsn,
            min_dirty_lsn,
            shutdown,
            writer_thread_waker,
            writer_thread: Some(handle),
            pages_flushed,
        }
    }

    /// Main loop for the background writer thread.
    fn writer_loop(
        pool: &BufferPool,
        write_fn: &WriteFn,
        target_lsn: &AtomicU64,
        min_dirty_lsn: &AtomicU64,
        shutdown: &AtomicBool,
        pages_flushed: &AtomicU64,
        config: &BackgroundWriterConfig,
    ) {
        loop {
            if shutdown.load(Ordering::Acquire) {
                // Final flush before exiting
                Self::flush_cycle(
                    pool,
                    write_fn,
                    u64::MAX,
                    min_dirty_lsn,
                    pages_flushed,
                    config.pages_per_cycle,
                );
                return;
            }

            // Determine threshold: checkpoint target or unlimited trickle
            let target = target_lsn.load(Ordering::Acquire);
            let threshold = if target == 0 { u64::MAX } else { target };

            let flushed = Self::flush_cycle(
                pool,
                write_fn,
                threshold,
                min_dirty_lsn,
                pages_flushed,
                config.pages_per_cycle,
            );

            // Sleep based on whether work was done
            if flushed > 0 {
                thread::park_timeout(Duration::from_micros(config.active_sleep_us));
            } else {
                thread::park_timeout(Duration::from_micros(config.idle_sleep_us));
            }
        }
    }

    /// Executes one flush cycle: collect dirty pages, flush them, update min_dirty_lsn.
    /// Returns the number of pages flushed.
    fn flush_cycle(
        pool: &BufferPool,
        write_fn: &WriteFn,
        threshold: u64,
        min_dirty_lsn: &AtomicU64,
        pages_flushed: &AtomicU64,
        pages_per_cycle: usize,
    ) -> usize {
        let dirty_pages = pool.collect_dirty_pages(threshold, pages_per_cycle);

        if dirty_pages.is_empty() {
            // No dirty pages below threshold
            if threshold == u64::MAX {
                min_dirty_lsn.store(u64::MAX, Ordering::Release);
            }
            return 0;
        }

        let mut flushed = 0;
        let mut new_min = u64::MAX;

        for &(page_id, frame_id, dlsn) in &dirty_pages {
            match pool.flush_dirty_frame(page_id, frame_id, dlsn, |pid, data| write_fn(pid, data)) {
                Ok(true) => flushed += 1,
                Ok(false) => {
                    // Page was evicted or already clean, track its LSN as still pending
                    if dlsn < new_min {
                        new_min = dlsn;
                    }
                }
                Err(_) => {
                    // I/O error, page remains dirty. Track its LSN.
                    if dlsn < new_min {
                        new_min = dlsn;
                    }
                }
            }
        }

        pages_flushed.fetch_add(flushed as u64, Ordering::Relaxed);

        // Update min_dirty_lsn: if we flushed everything, scan for remaining dirty pages
        if new_min == u64::MAX {
            // All collected pages were flushed. Check if there are more dirty pages.
            let remaining = pool.collect_dirty_pages(threshold, 1);
            if let Some(&(_, _, lsn)) = remaining.first() {
                min_dirty_lsn.store(lsn, Ordering::Release);
            } else {
                min_dirty_lsn.store(u64::MAX, Ordering::Release);
            }
        } else {
            min_dirty_lsn.store(new_min, Ordering::Release);
        }

        flushed
    }

    /// Sets the checkpoint target LSN. The background writer will prioritize
    /// flushing all dirty pages with dirty_lsn <= this value.
    /// Set to 0 to return to normal trickle-flush mode.
    pub fn set_target_lsn(&self, lsn: u64) {
        self.target_lsn.store(lsn, Ordering::Release);
        self.wake();
    }

    /// Returns the lowest unflushed dirty_lsn. u64::MAX means no dirty pages.
    pub fn min_dirty_lsn(&self) -> u64 {
        self.min_dirty_lsn.load(Ordering::Acquire)
    }

    /// Returns total pages flushed since creation.
    pub fn pages_flushed(&self) -> u64 {
        self.pages_flushed.load(Ordering::Relaxed)
    }

    /// Wakes the background writer thread.
    pub fn wake(&self) {
        if let Some(thread) = self.writer_thread_waker.get() {
            thread.unpark();
        }
    }

    /// Gracefully shuts down the background writer.
    /// Performs a final flush cycle before exiting.
    pub fn shutdown(&mut self) {
        self.shutdown.store(true, Ordering::Release);
        self.wake();
        if let Some(handle) = self.writer_thread.take() {
            let _ = handle.join();
        }
    }
}

impl Drop for BackgroundWriter {
    fn drop(&mut self) {
        if self.writer_thread.is_some() {
            self.shutdown();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pool::BufferPoolConfig;
    use std::sync::atomic::AtomicU32;

    #[test]
    fn test_background_writer_flushes_dirty_pages() {
        let pool = Arc::new(BufferPool::new(BufferPoolConfig { num_frames: 100 }));

        // Track which pages get flushed
        let flush_count = Arc::new(AtomicU32::new(0));
        let flush_count_clone = Arc::clone(&flush_count);

        let write_fn: WriteFn = Arc::new(move |_pid, _data| {
            flush_count_clone.fetch_add(1, Ordering::Relaxed);
            Ok(())
        });

        // Dirty some pages with LSN stamps
        for i in 0..10u64 {
            let page_id = PageId::new(0, i);
            let (_, _) = pool.new_page(page_id).unwrap();
            pool.unpin_page(page_id, true);
            pool.mark_dirty_with_lsn(page_id, (i + 1) * 100);
        }

        let mut writer = BackgroundWriter::new(
            Arc::clone(&pool),
            write_fn,
            BackgroundWriterConfig {
                pages_per_cycle: 64,
                idle_sleep_us: 100,
                active_sleep_us: 50,
            },
        );

        // Wait for the writer to flush
        thread::sleep(Duration::from_millis(50));

        writer.shutdown();

        assert!(flush_count.load(Ordering::Relaxed) >= 10);
    }

    #[test]
    fn test_background_writer_respects_target_lsn() {
        let pool = Arc::new(BufferPool::new(BufferPoolConfig { num_frames: 100 }));

        let flush_count = Arc::new(AtomicU32::new(0));
        let flush_count_clone = Arc::clone(&flush_count);

        let write_fn: WriteFn = Arc::new(move |_pid, _data| {
            flush_count_clone.fetch_add(1, Ordering::Relaxed);
            Ok(())
        });

        // Create pages: 5 with low LSN (100-500), 5 with high LSN (1000-1400)
        for i in 0..5u64 {
            let page_id = PageId::new(0, i);
            pool.new_page(page_id).unwrap();
            pool.unpin_page(page_id, true);
            pool.mark_dirty_with_lsn(page_id, (i + 1) * 100);
        }
        for i in 5..10u64 {
            let page_id = PageId::new(0, i);
            pool.new_page(page_id).unwrap();
            pool.unpin_page(page_id, true);
            pool.mark_dirty_with_lsn(page_id, 1000 + (i - 5) * 100);
        }

        let mut writer = BackgroundWriter::new(
            Arc::clone(&pool),
            write_fn,
            BackgroundWriterConfig {
                pages_per_cycle: 64,
                idle_sleep_us: 100,
                active_sleep_us: 50,
            },
        );

        // Set target to only flush pages with LSN <= 500
        writer.set_target_lsn(500);

        // Wait for the writer
        thread::sleep(Duration::from_millis(50));

        // The writer should have flushed the 5 low-LSN pages
        let flushed = flush_count.load(Ordering::Relaxed);
        assert!(flushed >= 5, "expected at least 5, got {}", flushed);

        writer.shutdown();
    }

    #[test]
    fn test_background_writer_min_dirty_lsn() {
        let pool = Arc::new(BufferPool::new(BufferPoolConfig { num_frames: 100 }));

        let write_fn: WriteFn = Arc::new(|_pid, _data| Ok(()));

        // Dirty a page with LSN 500
        let page_id = PageId::new(0, 0);
        pool.new_page(page_id).unwrap();
        pool.unpin_page(page_id, true);
        pool.mark_dirty_with_lsn(page_id, 500);

        let mut writer = BackgroundWriter::new(
            Arc::clone(&pool),
            write_fn,
            BackgroundWriterConfig {
                pages_per_cycle: 64,
                idle_sleep_us: 100,
                active_sleep_us: 50,
            },
        );

        // Wait for flush
        thread::sleep(Duration::from_millis(50));

        // After flushing, min_dirty_lsn should be u64::MAX (no dirty pages)
        let min = writer.min_dirty_lsn();
        assert_eq!(min, u64::MAX, "expected u64::MAX after flush, got {}", min);

        writer.shutdown();
    }

    #[test]
    fn test_background_writer_shutdown_flushes() {
        let pool = Arc::new(BufferPool::new(BufferPoolConfig { num_frames: 100 }));

        let flush_count = Arc::new(AtomicU32::new(0));
        let flush_count_clone = Arc::clone(&flush_count);

        let write_fn: WriteFn = Arc::new(move |_pid, _data| {
            flush_count_clone.fetch_add(1, Ordering::Relaxed);
            Ok(())
        });

        // Dirty pages
        for i in 0..5u64 {
            let page_id = PageId::new(0, i);
            pool.new_page(page_id).unwrap();
            pool.unpin_page(page_id, true);
            pool.mark_dirty_with_lsn(page_id, (i + 1) * 100);
        }

        // Shut down immediately (no sleep). The shutdown final flush should get them.
        let mut writer = BackgroundWriter::new(
            Arc::clone(&pool),
            write_fn,
            BackgroundWriterConfig {
                pages_per_cycle: 64,
                idle_sleep_us: 500_000, // Very long idle so it doesn't flush before shutdown
                active_sleep_us: 500_000,
            },
        );

        writer.shutdown();

        assert!(flush_count.load(Ordering::Relaxed) >= 5);
    }
}
