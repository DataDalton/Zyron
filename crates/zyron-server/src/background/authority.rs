//! Which node may start work that changes data.
//!
//! A background worker that rewrites a lake table, expires rows past their
//! retention or runs a schedule is producing changes, not reacting to them.
//! On a node in a consensus group those changes go through the group like
//! any other write. The leader runs them, captures what they change into a
//! replication changeset and commits through the log, so every member
//! applies the same change at the same position. A follower produces none,
//! because a change it made on its own would be held by that member alone,
//! and nothing anywhere would report the divergence.
//!
//! Physical work that leaves the visible data alone is not gated. Vacuum in
//! particular must not be: the oldest snapshot a node may reclaim behind is
//! its own, and a follower serving reads has readers the leader knows nothing
//! about. The same goes for checkpoints, statistics, log archiving, columnar
//! compaction, lake log collapse and vacuum, and the in-memory feature store,
//! each of which is a node's own layout of data every member holds.
//!
//! A node outside a group answers yes to everything, which is what keeps
//! consensus an addition to a single node rather than a mode it runs in.

use std::sync::{Arc, OnceLock};

/// Whether this node may start work that changes data.
#[derive(Clone, Default)]
pub struct WriteAuthority {
    /// True when this node's configuration puts it in a group
    expects_group: bool,
    /// Whether this node leads its group now, installed once the group has
    /// started. Shared by every clone, so a worker started before the group
    /// joined reads the same answer as one started after
    leads: Arc<OnceLock<Arc<dyn Fn() -> bool + Send + Sync>>>,
}

impl WriteAuthority {
    /// A node in no group, which decides everything for itself
    pub fn alone() -> Self {
        Self {
            expects_group: false,
            leads: Arc::new(OnceLock::new()),
        }
    }

    /// A node whose configuration puts it in a group. Until the group has
    /// started and `attach_group` has run, the node produces no changes,
    /// because it does not yet know whether it leads
    pub fn pending() -> Self {
        Self {
            expects_group: true,
            leads: Arc::new(OnceLock::new()),
        }
    }

    /// Installs the leadership reading once the group has started. The
    /// first installation is the one kept
    pub fn attach_group(&self, leads: Arc<dyn Fn() -> bool + Send + Sync>) {
        let _ = self.leads.set(leads);
    }

    /// True while this node may start work that produces changes, always
    /// outside a group, and while leading one. The workers asking commit
    /// what they change through the group, so a change made here reaches
    /// every member, and a follower makes none
    #[inline]
    pub fn may_write(&self) -> bool {
        !self.expects_group || self.leads.get().is_some_and(|leads| leads())
    }

    /// True when this node is in a group at all, for a worker whose work
    /// reaches the group another way or that reports why it is idle
    #[inline]
    pub fn in_group(&self) -> bool {
        self.expects_group
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicBool, Ordering};

    #[test]
    fn a_node_outside_a_group_decides_for_itself() {
        let authority = WriteAuthority::alone();
        assert!(authority.may_write());
        assert!(!authority.in_group());
    }

    #[test]
    fn the_default_is_a_node_outside_a_group() {
        assert!(WriteAuthority::default().may_write());
    }

    /// A member that does not know whether it leads writes nothing, and one
    /// that knows follows the reading as it moves
    #[test]
    fn a_member_of_a_group_writes_while_it_leads() {
        let authority = WriteAuthority::pending();
        assert!(!authority.may_write());
        assert!(authority.in_group());

        let leading = Arc::new(AtomicBool::new(false));
        let reading = Arc::clone(&leading);
        let worker_copy = authority.clone();
        authority.attach_group(Arc::new(move || reading.load(Ordering::Relaxed)));
        assert!(!worker_copy.may_write());
        leading.store(true, Ordering::Relaxed);
        assert!(worker_copy.may_write());
        leading.store(false, Ordering::Relaxed);
        assert!(!authority.may_write());
    }
}
