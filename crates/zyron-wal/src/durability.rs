//! Off-critical-path durability wakeups.
//!
//! A synchronous committer blocks until the WAL has durably flushed its commit
//! LSN. Waking those committers is expensive at high concurrency (one thread
//! resume each), and doing it on the flush thread stalls the next device write
//! behind the wakes. This module moves all waking onto a dedicated notifier
//! thread so the flush thread issues device writes back to back, and wakes only
//! the committers a flush actually satisfied (target LSN ordered), not every
//! parked committer.
//!
//! The flush thread, after storing the new flushed LSN, pokes the notifier with
//! a single unpark. The notifier drains the satisfied waiters while the flush
//! thread is already inside the next write-through.
//!
//! Lost-wakeup safety: the flush thread stores flushed_lsn (Release) before
//! poking the notifier; the notifier reads flushed_lsn under the queue lock
//! before draining; a waiter reads flushed_lsn under that same lock before it
//! registers. The lock serializes register vs drain, so a waiter either observes
//! a satisfying flushed_lsn and returns without parking, or is registered before
//! the drain that will wake it. A waiter parks only while flushed_lsn is below
//! its target, which means the flush thread still has its record to write and
//! will flush and poke again.

use std::cmp::Reverse;
use std::collections::BinaryHeap;
use std::collections::HashMap;
use std::sync::Arc;
use std::sync::OnceLock;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::thread::{JoinHandle, Thread};

use parking_lot::Mutex;
use zyron_common::profile::{self, Phase};

use crate::record::Lsn;
use crate::writer::FlushWaker;

/// Bounded pre-spin before a committer parks. Group commit often satisfies a
/// committer within a flush cycle; a short spin lets the quickest commits return
/// without ever registering or parking. Kept small so an oversubscribed core is
/// not burned on a commit that will wait a full flush cycle anyway.
const PRE_SPIN_ITERS: u32 = 400;

struct Waiters {
    /// Min-heap of (target_lsn, seq) so the lowest unmet target is examined
    /// first and the drain stops at the first target the flush has not reached.
    heap: BinaryHeap<Reverse<(u64, u64)>>,
    /// Parked committer threads keyed by seq. A heap entry whose seq is absent
    /// here was already woken or cancelled and is skipped when popped.
    threads: HashMap<u64, Thread>,
    /// Monotonic registration id.
    next_seq: u64,
}

/// Wakes durability waiters off the flush thread's critical path, in target-LSN
/// order. Owns a notifier thread; the flush thread only pokes it.
pub struct DurabilityNotifier {
    inner: Mutex<Waiters>,
    flushed_lsn: Arc<AtomicU64>,
    flush_io_error: Arc<AtomicBool>,
    shutdown: Arc<AtomicBool>,
    /// The notifier thread, for poke() to unpark. Set when the thread starts.
    notifier_waker: OnceLock<Thread>,
    /// Async durability waker (DurabilityQueue::wake_satisfied), invoked by the
    /// notifier so async waiters are also woken off the flush critical path.
    durable_waker: Arc<OnceLock<FlushWaker>>,
}

impl DurabilityNotifier {
    pub fn new(
        flushed_lsn: Arc<AtomicU64>,
        flush_io_error: Arc<AtomicBool>,
        shutdown: Arc<AtomicBool>,
        durable_waker: Arc<OnceLock<FlushWaker>>,
    ) -> Arc<Self> {
        Arc::new(Self {
            inner: Mutex::new(Waiters {
                heap: BinaryHeap::new(),
                threads: HashMap::new(),
                next_seq: 0,
            }),
            flushed_lsn,
            flush_io_error,
            shutdown,
            notifier_waker: OnceLock::new(),
            durable_waker,
        })
    }

    /// Spawns the notifier thread. It parks until poked, then wakes every waiter
    /// the current flushed_lsn satisfies plus the async waiters, and re-parks.
    pub fn spawn(self: &Arc<Self>) -> JoinHandle<()> {
        let me = Arc::clone(self);
        std::thread::spawn(move || {
            me.notifier_waker.set(std::thread::current()).ok();
            loop {
                me.drain();
                if me.shutdown.load(Ordering::Acquire) {
                    break;
                }
                std::thread::park();
            }
            // Final drain so a flush that completed during shutdown still wakes
            // its waiters before the thread exits.
            me.drain();
            me.wake_async();
        })
    }

    /// Unparks the notifier thread. Called by the flush thread after it stores a
    /// new flushed_lsn (and on shutdown). A single cheap unpark; the per-waiter
    /// wakes happen on the notifier thread, overlapped with the next device
    /// write.
    #[inline]
    pub fn poke(&self) {
        if let Some(t) = self.notifier_waker.get() {
            t.unpark();
        }
    }

    /// Wakes every registered sync waiter whose target the current flushed_lsn
    /// satisfies, then the async waiters. Runs on the notifier thread.
    fn drain(&self) {
        let _s = profile::scope(Phase::FlushWake);
        let flushed = self.flushed_lsn.load(Ordering::Acquire);
        let errored = self.flush_io_error.load(Ordering::Acquire);
        let mut to_wake: Vec<Thread> = Vec::new();
        {
            let mut inner = self.inner.lock();
            while let Some(Reverse((target, seq))) = inner.heap.peek().copied() {
                // On an I/O error wake everyone so they observe the error and
                // return; otherwise stop at the first unmet target.
                if target > flushed && !errored {
                    break;
                }
                inner.heap.pop();
                if let Some(t) = inner.threads.remove(&seq) {
                    to_wake.push(t);
                }
            }
        }
        // Unpark outside the lock so a woken committer re-registering does not
        // contend with this drain.
        for t in to_wake {
            t.unpark();
        }
        self.wake_async();
    }

    /// Invokes the async durability waker if one is registered.
    #[inline]
    fn wake_async(&self) {
        if let Some(waker) = self.durable_waker.get() {
            waker();
        }
    }

    /// Blocks the calling thread until flushed_lsn reaches `target`. Pre-spins
    /// briefly, then registers and parks; the notifier unparks it once a flush
    /// satisfies its target. Returns an error if the flush thread reports an I/O
    /// failure (which never advances flushed_lsn, so the waiter would otherwise
    /// block forever).
    pub fn wait(&self, target: Lsn) -> zyron_common::Result<()> {
        let target = target.0;
        let _s = profile::scope(Phase::DurabilityWait);

        // Pre-spin: catch the commits a near-complete flush satisfies without
        // registering or parking.
        let mut spins = 0u32;
        loop {
            if self.flushed_lsn.load(Ordering::Acquire) >= target {
                return Ok(());
            }
            if self.flush_io_error.load(Ordering::Acquire) {
                return Err(io_error());
            }
            if spins >= PRE_SPIN_ITERS {
                break;
            }
            spins += 1;
            std::hint::spin_loop();
        }

        // Register under the lock. The notifier reads flushed_lsn under this same
        // lock when it drains, so the recheck here closes the lost-wakeup window.
        let seq = {
            let mut inner = self.inner.lock();
            if self.flushed_lsn.load(Ordering::Acquire) >= target {
                return Ok(());
            }
            if self.flush_io_error.load(Ordering::Acquire) {
                return Err(io_error());
            }
            let seq = inner.next_seq;
            inner.next_seq += 1;
            inner.heap.push(Reverse((target, seq)));
            inner.threads.insert(seq, std::thread::current());
            seq
        };

        let result = loop {
            if self.flushed_lsn.load(Ordering::Acquire) >= target {
                break Ok(());
            }
            if self.flush_io_error.load(Ordering::Acquire) {
                break Err(io_error());
            }
            std::thread::park();
        };

        // Remove our registration if the notifier has not already. The heap
        // entry, if still present, is skipped on pop because its seq is gone.
        self.inner.lock().threads.remove(&seq);
        result
    }
}

#[inline]
fn io_error() -> zyron_common::ZyronError {
    zyron_common::ZyronError::WalWriteFailed("flush thread encountered an I/O error".into())
}
