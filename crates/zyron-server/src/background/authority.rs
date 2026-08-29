//! Which node may start work that changes data.
//!
//! A background worker that rewrites a lake table, refreshes a materialized
//! view, expires rows past their retention or runs a schedule is producing
//! changes, not reacting to them. None of those workers capture what they
//! change into a replication changeset yet, so on a node in a consensus group
//! their writes would exist on that node alone: the leader would expire rows
//! its followers keep forever, and nothing anywhere would report it. Until
//! their writes replicate, a member of a group answers no to all of them,
//! which trades staleness everyone can see for divergence nobody can.
//!
//! Physical work that leaves the visible data alone is not gated. Vacuum in
//! particular must not be: the oldest snapshot a node may reclaim behind is
//! its own, and a follower serving reads has readers the leader knows nothing
//! about. The same goes for checkpoints, statistics and log archiving.
//!
//! A node outside a group answers yes to everything, which is what keeps
//! consensus an addition to a single node rather than a mode it runs in.

/// Whether this node may start work that changes data.
#[derive(Clone, Default)]
pub struct WriteAuthority {
    /// True when this node's configuration puts it in a group
    expects_group: bool,
}

impl WriteAuthority {
    /// A node in no group, which decides everything for itself
    pub fn alone() -> Self {
        Self {
            expects_group: false,
        }
    }

    /// A node whose configuration puts it in a group
    pub fn pending() -> Self {
        Self {
            expects_group: true,
        }
    }

    /// True while this node may start work that produces changes.
    ///
    /// The workers asking do not capture what they change, so in a group the
    /// answer is no on every node, the leader included: a change only one
    /// member holds is a divergence, not a feature. When their writes go
    /// through capture this becomes a leadership question again
    #[inline]
    pub fn may_write(&self) -> bool {
        !self.expects_group
    }

    /// True when this node is in a group at all, for a worker that reports
    /// why it is idle
    #[inline]
    pub fn in_group(&self) -> bool {
        self.expects_group
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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

    /// Uncaptured writes on any member of a group would be held by that
    /// member alone, so membership itself is the refusal
    #[test]
    fn a_member_of_a_group_writes_nothing_uncaptured() {
        let authority = WriteAuthority::pending();
        assert!(!authority.may_write());
        assert!(authority.in_group());
    }
}
