//! Targeted commit-durability wakeups.
//!
//! A committing transaction awaits until the WAL has flushed (and fsync'd) its
//! commit LSN. A single broadcast channel would wake every awaiting committer on
//! every flush, so at high concurrency most wakeups are spurious: a committer
//! re-checks its target LSN, finds it unmet, and sleeps again. This queue wakes
//! only the committers whose target LSN the latest flush satisfied.
//!
//! Lost-wakeup safety rests on one ordering rule: the flush thread stores the
//! new flushed LSN before taking the queue lock to drain satisfied waiters, and
//! a waiter reads the flushed LSN while holding that same lock before it
//! registers. The lock serializes the two, so a waiter either observes the new
//! LSN and resolves immediately or is registered in time for the drain to wake
//! it. No waiter can register just after a flush that would have satisfied it
//! and then sleep forever.

use std::collections::BinaryHeap;
use std::collections::HashMap;
use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;
use std::task::{Context, Poll, Waker};

use parking_lot::Mutex;
use zyron_wal::WalWriter;
use zyron_wal::record::Lsn;

/// Heap entry ordering waiters by ascending target LSN. BinaryHeap is a
/// max-heap, so the target is negated by wrapping in Reverse to pop the
/// smallest target first.
type HeapEntry = std::cmp::Reverse<(u64, u64)>;

struct QueueInner {
    /// Min-heap of (target_lsn, seq) so the lowest target is drained first.
    heap: BinaryHeap<HeapEntry>,
    /// Live waiter wakers keyed by seq. A heap entry whose seq is absent here
    /// was already woken or dropped and is skipped when popped.
    wakers: HashMap<u64, Waker>,
    /// Monotonic id assigned to each waiter registration.
    next_seq: u64,
}

/// Wakes committed-LSN durability waiters in target-LSN order.
pub struct DurabilityQueue {
    wal: Arc<WalWriter>,
    inner: Mutex<QueueInner>,
}

impl DurabilityQueue {
    pub fn new(wal: Arc<WalWriter>) -> Arc<Self> {
        Arc::new(Self {
            wal,
            inner: Mutex::new(QueueInner {
                heap: BinaryHeap::new(),
                wakers: HashMap::new(),
                next_seq: 0,
            }),
        })
    }

    /// Invoked by the WAL flush thread after a flush has advanced (and fsync'd)
    /// the flushed LSN. Wakes every registered waiter whose target the current
    /// flushed LSN now satisfies. Reads the flushed LSN itself so it always sees
    /// the value the just-finished flush stored.
    pub fn wake_satisfied(&self) {
        let flushed = self.wal.flushed_lsn().0;
        let mut to_wake: Vec<Waker> = Vec::new();
        {
            let mut inner = self.inner.lock();
            while let Some(std::cmp::Reverse((target, seq))) = inner.heap.peek().copied() {
                if target > flushed {
                    // Lowest target still unmet, the rest are higher.
                    break;
                }
                inner.heap.pop();
                if let Some(waker) = inner.wakers.remove(&seq) {
                    to_wake.push(waker);
                }
            }
        }
        // Wake outside the lock so a woken task re-polling on this thread (in a
        // current-thread runtime) does not deadlock on the queue lock.
        for waker in to_wake {
            waker.wake();
        }
    }

    /// Returns a future that resolves once the WAL has durably flushed `target`.
    pub fn wait(self: &Arc<Self>, target: Lsn) -> DurableWait {
        DurableWait {
            queue: Arc::clone(self),
            target: target.0,
            seq: None,
        }
    }
}

/// Future awaiting durability of a single commit LSN.
pub struct DurableWait {
    queue: Arc<DurabilityQueue>,
    target: u64,
    /// Registration id once this future has parked a waker in the queue.
    seq: Option<u64>,
}

impl Future for DurableWait {
    type Output = ();

    fn poll(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<()> {
        // Cheap check before locking.
        if self.queue.wal.flushed_lsn().0 >= self.target {
            self.clear_registration();
            return Poll::Ready(());
        }

        let mut inner = self.queue.inner.lock();
        // Re-check under the lock. The flush thread stores flushed_lsn before
        // taking this lock to drain, so a value still below target here means a
        // later flush will observe our registration below and wake us.
        if self.queue.wal.flushed_lsn().0 >= self.target {
            drop(inner);
            self.clear_registration();
            return Poll::Ready(());
        }

        match self.seq {
            Some(seq) => {
                // Already registered: refresh the waker in case the task moved.
                inner.wakers.insert(seq, cx.waker().clone());
            }
            None => {
                let seq = inner.next_seq;
                inner.next_seq += 1;
                inner.heap.push(std::cmp::Reverse((self.target, seq)));
                inner.wakers.insert(seq, cx.waker().clone());
                drop(inner);
                self.seq = Some(seq);
                return Poll::Pending;
            }
        }
        Poll::Pending
    }
}

impl DurableWait {
    /// Removes this waiter's waker registration. The stale heap entry is left in
    /// place and skipped when popped, avoiding an O(n) heap removal.
    fn clear_registration(&mut self) {
        if let Some(seq) = self.seq.take() {
            self.queue.inner.lock().wakers.remove(&seq);
        }
    }
}

impl Drop for DurableWait {
    fn drop(&mut self) {
        // A future dropped before resolving (cancelled commit, timeout) must not
        // leave its waker registered.
        self.clear_registration();
    }
}
