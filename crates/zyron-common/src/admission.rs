//! Whether this node takes new work, and what it has in flight.
//!
//! One place the accept loop, the query path, the mesh drain handler, the
//! readiness probe, and the upgrade driver all read. A drain is one flag
//! flip here, and "is it finished" is one answer here, so no two of those
//! readers can disagree about the node's state

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

/// What the node has in hand at one instant
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct InFlightCounts {
    pub queries: u64,
    pub sessions: u64,
    pub transactions: u64,
}

/// The node's admission state and in-flight counters
#[derive(Debug, Default)]
pub struct Admission {
    /// Set once startup completes, cleared again while draining
    accepting: AtomicBool,
    /// Set by a drain, cleared when the drain is abandoned
    draining: AtomicBool,
    queries: AtomicU64,
    sessions: AtomicU64,
    transactions: AtomicU64,
}

/// Which counter a guard holds
#[derive(Debug, Clone, Copy)]
enum Held {
    Query,
    Session,
    Transaction,
}

/// Holds one unit of in-flight work and gives it back on drop, so a
/// connection that panics or is cancelled still lets the drain finish
pub struct InFlightGuard {
    admission: Arc<Admission>,
    held: Held,
}

impl Drop for InFlightGuard {
    fn drop(&mut self) {
        let counter = match self.held {
            Held::Query => &self.admission.queries,
            Held::Session => &self.admission.sessions,
            Held::Transaction => &self.admission.transactions,
        };
        decrement(counter);
    }
}

fn decrement(counter: &AtomicU64) {
    // Saturating, so a counter can never wrap below zero from an unmatched
    // release and report a drain that never finishes
    let mut current = counter.load(Ordering::Relaxed);
    loop {
        let next = current.saturating_sub(1);
        match counter.compare_exchange_weak(current, next, Ordering::AcqRel, Ordering::Relaxed) {
            Ok(_) => return,
            Err(observed) => current = observed,
        }
    }
}

impl Admission {
    pub fn new() -> Self {
        Self::default()
    }

    /// Opens the door. Startup calls this once the node can serve
    pub fn mark_accepting(&self) {
        self.accepting.store(true, Ordering::Release);
    }

    /// Whether a new connection may be accepted right now
    pub fn is_accepting(&self) -> bool {
        self.accepting.load(Ordering::Acquire) && !self.draining.load(Ordering::Acquire)
    }

    /// Stops taking new work. Returns false when a drain was already under
    /// way, so a repeated request is recognized rather than counted twice
    pub fn begin_drain(&self) -> bool {
        !self.draining.swap(true, Ordering::AcqRel)
    }

    /// Takes new work again, which a drain that was abandoned needs
    pub fn end_drain(&self) {
        self.draining.store(false, Ordering::Release);
    }

    pub fn is_draining(&self) -> bool {
        self.draining.load(Ordering::Acquire)
    }

    /// A session began. The guard closes it
    pub fn open_session(self: &Arc<Self>) -> InFlightGuard {
        self.sessions.fetch_add(1, Ordering::AcqRel);
        InFlightGuard {
            admission: Arc::clone(self),
            held: Held::Session,
        }
    }

    /// A statement began running. The guard finishes it
    pub fn begin_query(self: &Arc<Self>) -> InFlightGuard {
        self.queries.fetch_add(1, Ordering::AcqRel);
        InFlightGuard {
            admission: Arc::clone(self),
            held: Held::Query,
        }
    }

    /// A transaction opened. The guard closes it
    pub fn open_transaction(self: &Arc<Self>) -> InFlightGuard {
        self.transactions.fetch_add(1, Ordering::AcqRel);
        InFlightGuard {
            admission: Arc::clone(self),
            held: Held::Transaction,
        }
    }

    pub fn in_flight(&self) -> InFlightCounts {
        InFlightCounts {
            queries: self.queries.load(Ordering::Acquire),
            sessions: self.sessions.load(Ordering::Acquire),
            transactions: self.transactions.load(Ordering::Acquire),
        }
    }

    /// Whether a drain has nothing left to wait for. Sessions that hold no
    /// query and no transaction are idle and may be ended by the restart
    pub fn is_quiescent(&self) -> bool {
        let counts = self.in_flight();
        counts.queries == 0 && counts.transactions == 0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_node_accepts_only_after_startup_and_not_while_draining() {
        let admission = Admission::new();
        assert!(!admission.is_accepting());
        admission.mark_accepting();
        assert!(admission.is_accepting());
        assert!(admission.begin_drain());
        assert!(!admission.begin_drain());
        assert!(admission.is_draining());
        assert!(!admission.is_accepting());
        admission.end_drain();
        assert!(admission.is_accepting());
    }

    #[test]
    fn guards_count_and_release_on_drop() {
        let admission = Arc::new(Admission::new());
        let session = admission.open_session();
        let query = admission.begin_query();
        let txn = admission.open_transaction();
        assert_eq!(
            admission.in_flight(),
            InFlightCounts {
                queries: 1,
                sessions: 1,
                transactions: 1
            }
        );
        assert!(!admission.is_quiescent());
        drop(query);
        assert!(!admission.is_quiescent());
        drop(txn);
        assert!(
            admission.is_quiescent(),
            "an idle session does not hold a drain"
        );
        drop(session);
        assert_eq!(admission.in_flight(), InFlightCounts::default());
    }

    #[test]
    fn a_counter_never_wraps_below_zero() {
        let counter = AtomicU64::new(0);
        decrement(&counter);
        assert_eq!(counter.load(Ordering::Relaxed), 0);
    }
}
