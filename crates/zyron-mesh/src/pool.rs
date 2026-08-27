//! Nodes the mesh is keeping ready, and what they are still paid for.
//!
//! A warm pool is the difference between a scale-out that takes a minute and
//! one that takes a second. It is also the difference between a bill that
//! reflects the work done and one that reflects the work anticipated, so the
//! size is a cap the operator sets and never a target this code reaches for:
//! a mesh that projects no need keeps no warm pool whatever the cap says.
//!
//! ## Why the paid interval lives here
//!
//! Capacity billed by time is paid for in blocks. Giving a node back halfway
//! through a block does not refund the rest of it, and asking for a
//! replacement inside that block pays for it twice. So a node that drains
//! while it is still paid for goes back into the pool instead of away, and
//! what remains on its interval is what the driver's reclaim rule reads.

use std::collections::HashMap;
use std::sync::Mutex;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};

use crate::rpc::NodeRef;

/// A node kept ready, and when what was paid for it runs out.
#[derive(Debug, Clone)]
struct Warm {
    node: NodeRef,
    /// When the interval this node is paid through ends. None for capacity
    /// that is not billed by time, which is every on-premises deployment
    paid_until: Option<Instant>,
}

/// The nodes this mesh is holding ready.
pub struct WarmPool {
    /// Most recently offered first, so a node that was just drained is the
    /// next one taken. It is the warmest of them: its caches are the least
    /// stale and its interval has the most left on it
    ready: Mutex<Vec<Warm>>,
    /// Largest pool the operator will pay for. Zero keeps none
    cap: u32,
    /// How long the mesh expects to stay quiet, published by the projection
    /// and read by the reclaim rule
    idle_window_ms: AtomicU64,
    /// Intervals for nodes that have left the pool but are still paid for
    paid: Mutex<HashMap<u64, Instant>>,
}

impl WarmPool {
    pub fn new(cap: u32) -> Self {
        Self {
            ready: Mutex::new(Vec::new()),
            cap,
            idle_window_ms: AtomicU64::new(0),
            paid: Mutex::new(HashMap::new()),
        }
    }

    /// Largest pool the operator will pay for.
    pub fn cap(&self) -> u32 {
        self.cap
    }

    pub fn len(&self) -> usize {
        self.ready.lock().map(|r| r.len()).unwrap_or(0)
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Puts a node into the pool, or drops it when the pool is full.
    ///
    /// Returns whether it was kept. A node that does not fit is not an error:
    /// the cap is what the operator agreed to pay for, and holding one more
    /// than that because it happened to be free is how a bill grows without a
    /// decision.
    pub fn offer(&self, node: NodeRef) -> bool {
        let mut ready = match self.ready.lock() {
            Ok(ready) => ready,
            Err(poisoned) => poisoned.into_inner(),
        };
        if ready.len() >= self.cap as usize {
            return false;
        }
        if ready.iter().any(|w| w.node.node_id == node.node_id) {
            return true;
        }
        let paid_until = self
            .paid
            .lock()
            .ok()
            .and_then(|p| p.get(&node.node_id).copied());
        ready.push(Warm { node, paid_until });
        true
    }

    /// Claims the warmest node, or None when the pool is empty.
    pub fn take(&self) -> Option<NodeRef> {
        let mut ready = match self.ready.lock() {
            Ok(ready) => ready,
            Err(poisoned) => poisoned.into_inner(),
        };
        ready.pop().map(|w| w.node)
    }

    /// Records that a node is paid for until a given moment.
    pub fn record_paid_until(&self, node: &NodeRef, until: Instant) {
        if let Ok(mut paid) = self.paid.lock() {
            paid.insert(node.node_id, until);
        }
        if let Ok(mut ready) = self.ready.lock() {
            for warm in ready.iter_mut() {
                if warm.node.node_id == node.node_id {
                    warm.paid_until = Some(until);
                }
            }
        }
    }

    /// What is left of the interval a node is already paid for.
    ///
    /// Zero for capacity that is not billed by time, and zero once the
    /// interval has passed, which is what makes the reclaim rule allow it.
    pub fn remaining_paid_interval(&self, node: &NodeRef) -> Duration {
        let until = match self.paid.lock() {
            Ok(paid) => paid.get(&node.node_id).copied(),
            Err(poisoned) => poisoned.into_inner().get(&node.node_id).copied(),
        };
        match until {
            Some(until) => until.saturating_duration_since(Instant::now()),
            None => Duration::ZERO,
        }
    }

    /// Publishes how long the mesh expects to stay quiet.
    ///
    /// Comes from the arrival trend the pressure projection fits, and is the
    /// other half of the reclaim rule: an interval with time left on it is
    /// only worth keeping if the quiet is going to outlast it.
    pub fn set_predicted_idle_window(&self, window: Duration) {
        self.idle_window_ms.store(
            window.as_millis().min(u64::MAX as u128) as u64,
            Ordering::Relaxed,
        );
    }

    pub fn predicted_idle_window(&self) -> Duration {
        Duration::from_millis(self.idle_window_ms.load(Ordering::Relaxed))
    }

    /// Drops everything remembered about a node that has been given back.
    pub fn forget(&self, node: &NodeRef) {
        if let Ok(mut ready) = self.ready.lock() {
            ready.retain(|w| w.node.node_id != node.node_id);
        }
        if let Ok(mut paid) = self.paid.lock() {
            paid.remove(&node.node_id);
        }
    }

    /// Every node currently held ready.
    pub fn nodes(&self) -> Vec<NodeRef> {
        match self.ready.lock() {
            Ok(ready) => ready.iter().map(|w| w.node.clone()).collect(),
            Err(poisoned) => poisoned
                .into_inner()
                .iter()
                .map(|w| w.node.clone())
                .collect(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn node(id: u64) -> NodeRef {
        NodeRef::new(id, format!("node-{id}"))
    }

    /// A pool the operator sized at zero keeps nothing, however much is
    /// offered to it.
    #[test]
    fn a_cap_of_zero_keeps_nothing() {
        let pool = WarmPool::new(0);
        assert!(!pool.offer(node(1)));
        assert!(pool.is_empty());
        assert!(pool.take().is_none());
    }

    /// The pool holds up to its cap and refuses past it.
    #[test]
    fn the_cap_is_a_ceiling_not_a_target() {
        let pool = WarmPool::new(2);
        assert!(pool.offer(node(1)));
        assert!(pool.offer(node(2)));
        assert!(!pool.offer(node(3)), "the pool took one past its cap");
        assert_eq!(pool.len(), 2);
    }

    /// The most recently offered node is the next one taken, because it is
    /// the warmest.
    #[test]
    fn the_warmest_node_is_taken_first() {
        let pool = WarmPool::new(3);
        pool.offer(node(1));
        pool.offer(node(2));
        assert_eq!(pool.take().map(|n| n.node_id), Some(2));
        assert_eq!(pool.take().map(|n| n.node_id), Some(1));
        assert!(pool.take().is_none());
    }

    /// Offering the same node twice keeps one of it.
    #[test]
    fn a_node_offered_twice_is_held_once() {
        let pool = WarmPool::new(4);
        pool.offer(node(7));
        pool.offer(node(7));
        assert_eq!(pool.len(), 1);
    }

    /// A node with nothing paid on it has nothing remaining, which is what
    /// lets the reclaim rule give it back immediately.
    #[test]
    fn capacity_that_is_not_billed_by_time_has_no_interval_left() {
        let pool = WarmPool::new(1);
        assert_eq!(pool.remaining_paid_interval(&node(1)), Duration::ZERO);
    }

    /// An interval with time left on it reports that time, and an interval
    /// that has passed reports none.
    #[test]
    fn a_paid_interval_counts_down_and_then_stops() {
        let pool = WarmPool::new(1);
        pool.record_paid_until(&node(1), Instant::now() + Duration::from_secs(60));
        let remaining = pool.remaining_paid_interval(&node(1));
        assert!(
            remaining > Duration::from_secs(50) && remaining <= Duration::from_secs(60),
            "{remaining:?}"
        );

        pool.record_paid_until(&node(2), Instant::now() - Duration::from_secs(1));
        assert_eq!(pool.remaining_paid_interval(&node(2)), Duration::ZERO);
    }

    /// A node given back is forgotten, interval and all, so a later node that
    /// reuses its id does not inherit its billing.
    #[test]
    fn a_reclaimed_node_leaves_nothing_behind() {
        let pool = WarmPool::new(2);
        pool.record_paid_until(&node(5), Instant::now() + Duration::from_secs(60));
        pool.offer(node(5));
        pool.forget(&node(5));
        assert!(pool.is_empty());
        assert_eq!(pool.remaining_paid_interval(&node(5)), Duration::ZERO);
    }
}
