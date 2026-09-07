// -----------------------------------------------------------------------------
// Streaming-job ownership reconciler.
//
// A streaming job's definition lives in the catalog and reaches every member of
// a group, because a member that does not hold it cannot take the job over. The
// runner is a different thing: it reads a source and writes a sink
// continuously, so exactly one member runs it or the sink is written once per
// member.
//
// Ownership follows the group's leader, and leadership moves without asking
// anyone. So rather than starting a runner when a job is created and hoping it
// stays where it was put, this reconciles: the node that leads runs every
// Active job, a node that does not lead runs none, and each pass corrects
// whatever the last leadership change left behind.
// -----------------------------------------------------------------------------

use std::sync::Arc;
use std::sync::atomic::AtomicBool;
use std::time::Duration;

/// How often ownership is reconciled. A leadership change is not announced to
/// this loop, so the interval is how long a job can be unrun after a failover
pub const DEFAULT_INTERVAL_SECS: u64 = 5;

/// What one pass decided, for the caller to act on and for the operator views
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Reconciled {
    /// Jobs started because this node leads and they were not running
    pub started: usize,
    /// Jobs stopped because this node no longer leads
    pub stopped: usize,
}

impl Reconciled {
    pub fn changed(&self) -> bool {
        self.started > 0 || self.stopped > 0
    }
}

/// Runs `on_tick` until shutdown. The tick owns the reconcile so this loop
/// stays free of the catalog and the job manager
pub async fn streaming_job_owner_loop<F>(
    shutdown: Arc<AtomicBool>,
    wake: Arc<tokio::sync::Notify>,
    interval_secs: u64,
    mut on_tick: F,
) where
    F: FnMut() + Send + 'static,
{
    let mut ticker = tokio::time::interval(Duration::from_secs(interval_secs.max(1)));
    while super::tick_until_shutdown(&mut ticker, &shutdown, &wake).await {
        on_tick();
    }
}

/// Decides what one pass should do.
///
/// `leads` is whether this node owns the jobs now, which on a node in no group
/// is always true because there is nobody else to own them. `active` names the
/// jobs the catalog says should be running, and `running` the ones this node
/// has a runner for
pub fn plan(leads: bool, active: &[u32], running: &[u32]) -> (Vec<u32>, Vec<u32>) {
    if !leads {
        // Every runner here belongs to whichever node leads now
        return (Vec::new(), running.to_vec());
    }
    let start: Vec<u32> = active
        .iter()
        .copied()
        .filter(|id| !running.contains(id))
        .collect();
    // A runner for a job the catalog no longer calls Active, because it was
    // paused or dropped on another member and that reached this one
    let stop: Vec<u32> = running
        .iter()
        .copied()
        .filter(|id| !active.contains(id))
        .collect();
    (start, stop)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_node_that_leads_runs_every_active_job() {
        let (start, stop) = plan(true, &[1, 2, 3], &[2]);
        assert_eq!(start, vec![1, 3]);
        assert!(stop.is_empty());
    }

    /// The whole point of the pass. A node that lost the group keeps its
    /// runners until something stops them, and two members writing one sink is
    /// the failure this exists to prevent
    #[test]
    fn a_node_that_stopped_leading_gives_up_every_runner() {
        let (start, stop) = plan(false, &[1, 2], &[1, 2]);
        assert!(start.is_empty());
        assert_eq!(stop, vec![1, 2]);
    }

    #[test]
    fn a_job_no_longer_active_is_stopped_where_it_runs() {
        let (start, stop) = plan(true, &[1], &[1, 7]);
        assert!(start.is_empty());
        assert_eq!(stop, vec![7]);
    }

    #[test]
    fn a_settled_leader_changes_nothing() {
        let (start, stop) = plan(true, &[1, 2], &[1, 2]);
        assert!(start.is_empty());
        assert!(stop.is_empty());
        assert!(!Reconciled::default().changed());
    }
}
