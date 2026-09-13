//! Exclusive locks over change stream positions.
//!
//! A transactional consume reads a stream and records where it read to. Two
//! consumers of one stream must not both take the same changes, so the first
//! read takes an exclusive lock on the stream's position and holds it until
//! the transaction ends. A second transaction reading the same stream parks
//! until the first commits or aborts, and then sees what is left.
//!
//! Keyed by stream id rather than by row locator, because a position is one
//! catalog row rather than a row of a user table, and a consumer holds exactly
//! one lock per stream it reads.
//!
//! Released through `unlock_all` by abort and by the drop of an active
//! transaction, so a rolled back or killed transaction leaves the position
//! where it was. A commit does not release it. The advance the consumer
//! recorded is installed in the catalog after the commit record lands, and
//! the installation is what releases the lock, so a waiter that wakes reads
//! the position the holder left rather than the one it started from

use std::sync::Arc;
use std::time::Duration;

use tokio::sync::Notify;
use zyron_common::{Result, ZyronError};

use super::deadlock::WaitForGraph;

/// Upper bound on a blocking position wait before the request fails, which
/// backstops a holder that never ends
const POSITION_WAIT_TIMEOUT: Duration = Duration::from_secs(10);

/// Exclusive locks over change stream positions.
///
/// Uses scc::HashMap for lock-free concurrent access. Each entry maps a stream
/// id to the transaction holding it. A per-transaction inverse list makes
/// release O(k) in the streams that transaction read rather than O(n) in the
/// streams the node has. Waiters park on a per-stream Notify that release
/// fires, so a blocked consume wakes as soon as the holder commits or aborts
pub struct StreamPositionLocks {
    holders: scc::HashMap<u64, u64>,
    txn_streams: scc::HashMap<u64, Vec<u64>>,
    waiters: scc::HashMap<u64, Arc<Notify>>,
    wait_graph: Arc<WaitForGraph>,
}

impl StreamPositionLocks {
    /// Creates an empty table sharing the wait-for graph the row lock table
    /// owns, so a cycle across the two kinds of lock is still detected
    pub fn new(wait_graph: Arc<WaitForGraph>) -> Self {
        Self {
            holders: scc::HashMap::new(),
            txn_streams: scc::HashMap::new(),
            waiters: scc::HashMap::new(),
            wait_graph,
        }
    }

    /// Takes the lock without waiting.
    ///
    /// Ok when it was granted or this transaction already holds it. Err with
    /// the holding transaction id otherwise. This is the peek-adjacent path a
    /// caller uses when it would rather report contention than wait
    pub fn try_lock(&self, txn_id: u64, stream_id: u64) -> std::result::Result<(), u64> {
        match self.holders.entry_sync(stream_id) {
            scc::hash_map::Entry::Occupied(entry) => {
                let holder = *entry.get();
                if holder == txn_id {
                    Ok(())
                } else {
                    Err(holder)
                }
            }
            scc::hash_map::Entry::Vacant(entry) => {
                entry.insert_entry(txn_id);
                self.txn_streams
                    .entry_sync(txn_id)
                    .or_default()
                    .get_mut()
                    .push(stream_id);
                Ok(())
            }
        }
    }

    /// Takes the lock, parking until the holder releases it or the wait times
    /// out.
    ///
    /// Registers on the stream's Notify before each re-check, so a release
    /// landing between the check and the await still wakes this waiter. Each
    /// park records a waiter-to-holder edge in the wait-for graph, and an edge
    /// that closes a cycle fails this request rather than completing it
    pub async fn lock_wait(&self, txn_id: u64, stream_id: u64) -> Result<()> {
        if self.try_lock(txn_id, stream_id).is_ok() {
            return Ok(());
        }
        let deadline = tokio::time::Instant::now() + POSITION_WAIT_TIMEOUT;
        loop {
            let notify = self.waiter_handle(stream_id);
            let notified = notify.notified();
            tokio::pin!(notified);
            notified.as_mut().enable();
            let holder = match self.try_lock(txn_id, stream_id) {
                Ok(()) => {
                    self.wait_graph.remove_edge(txn_id);
                    return Ok(());
                }
                Err(holder) => holder,
            };
            // Re-point the edge at the current holder. add_edge does not
            // replace an existing edge, so the stale one is removed first
            self.wait_graph.remove_edge(txn_id);
            if self.wait_graph.add_edge(txn_id, holder).is_some() {
                return Err(ZyronError::transaction_conflict(
                    txn_id,
                    format!(
                        "deadlock detected, txn {txn_id} waiting on change stream {stream_id} \
                         held by txn {holder} closes a wait cycle"
                    ),
                ));
            }
            if tokio::time::timeout_at(deadline, notified).await.is_err() {
                self.wait_graph.remove_edge(txn_id);
                return Err(ZyronError::transaction_conflict(
                    txn_id,
                    format!("lock wait timeout on change stream {stream_id} held by txn {holder}"),
                ));
            }
        }
    }

    /// The transaction holding a stream's position, None when it is free
    pub fn holder(&self, stream_id: u64) -> Option<u64> {
        self.holders.read_sync(&stream_id, |_, holder| *holder)
    }

    /// Whether this transaction already holds the stream's position
    pub fn holds(&self, txn_id: u64, stream_id: u64) -> bool {
        self.holder(stream_id) == Some(txn_id)
    }

    /// Releases every position this transaction holds and wakes their waiters
    pub fn unlock_all(&self, txn_id: u64) {
        if self.txn_streams.is_empty() {
            return;
        }
        if let Some((_, streams)) = self.txn_streams.remove_sync(&txn_id) {
            for stream_id in streams {
                self.release(stream_id, txn_id);
            }
        }
    }

    /// How many positions this transaction holds
    pub fn held_count(&self, txn_id: u64) -> usize {
        self.txn_streams
            .read_sync(&txn_id, |_, streams| streams.len())
            .unwrap_or(0)
    }

    /// Positions held across every transaction, for the diagnostics view
    pub fn lock_count(&self) -> usize {
        self.holders.len()
    }

    fn waiter_handle(&self, stream_id: u64) -> Arc<Notify> {
        match self.waiters.entry_sync(stream_id) {
            scc::hash_map::Entry::Occupied(entry) => Arc::clone(entry.get()),
            scc::hash_map::Entry::Vacant(entry) => {
                let notify = Arc::new(Notify::new());
                entry.insert_entry(Arc::clone(&notify));
                notify
            }
        }
    }

    fn release(&self, stream_id: u64, txn_id: u64) {
        let mut released = false;
        if let scc::hash_map::Entry::Occupied(entry) = self.holders.entry_sync(stream_id) {
            if *entry.get() == txn_id {
                let _ = entry.remove();
                released = true;
            }
        }
        if released && !self.waiters.is_empty() {
            if let Some((_, notify)) = self.waiters.remove_sync(&stream_id) {
                notify.notify_waiters();
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn table() -> StreamPositionLocks {
        StreamPositionLocks::new(Arc::new(WaitForGraph::new()))
    }

    #[test]
    fn test_a_free_position_is_granted() {
        let locks = table();
        assert!(locks.try_lock(1, 7).is_ok());
        assert_eq!(locks.holder(7), Some(1));
        assert!(locks.holds(1, 7));
    }

    #[test]
    fn test_a_held_position_reports_its_holder() {
        let locks = table();
        assert!(locks.try_lock(1, 7).is_ok());
        assert_eq!(locks.try_lock(2, 7), Err(1));
        // Re-taking a position this transaction already holds is granted, so a
        // second read inside one transaction does not deadlock against itself
        assert!(locks.try_lock(1, 7).is_ok());
    }

    #[test]
    fn test_unlock_all_frees_every_position() {
        let locks = table();
        assert!(locks.try_lock(1, 7).is_ok());
        assert!(locks.try_lock(1, 8).is_ok());
        assert_eq!(locks.held_count(1), 2);
        locks.unlock_all(1);
        assert_eq!(locks.holder(7), None);
        assert_eq!(locks.holder(8), None);
        assert!(locks.try_lock(2, 7).is_ok());
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn test_a_waiter_proceeds_once_the_holder_releases() {
        let locks = Arc::new(table());
        assert!(locks.try_lock(1, 7).is_ok());

        let waiting = Arc::clone(&locks);
        let handle = tokio::spawn(async move { waiting.lock_wait(2, 7).await });

        // The holder releases, which is what the waiter is parked on
        tokio::task::yield_now().await;
        locks.unlock_all(1);

        handle
            .await
            .expect("the waiting task ran to completion")
            .expect("the waiter took the position");
        assert_eq!(locks.holder(7), Some(2));
    }
}
