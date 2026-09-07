// -----------------------------------------------------------------------------
// Upgrade board reconciler.
//
// The board holds one row per member, and until this existed a node only ever
// wrote its own. The coordinator of a rolling upgrade writes a row for every
// node it drives, so its board is complete while it drives, but a follower's
// board showed one row: itself. `SHOW UPGRADE STATE` on a follower reported a
// single node in a group of five, which reads as though the others do not
// exist rather than as though this node cannot see them.
//
// Each member's row is that member's own journal state, so it is asked for
// rather than inferred. A member that does not answer keeps whatever row the
// board already held, because a node being restarted is exactly when it stops
// answering and exactly when the coordinator's row for it is the true one.
// -----------------------------------------------------------------------------

use zyron_common::format::NodeUpgradeState;

/// How often peer rows are refreshed. A phase change on another member is not
/// announced to this loop, so the interval is how stale a peer's row can be
pub const DEFAULT_INTERVAL_SECS: u64 = 10;

/// What one pass decided to write, so a caller can act on it and a test can
/// read it without a board
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct Reconciled {
    /// Rows to publish, replacing whatever the board holds for those members
    pub write: Vec<NodeUpgradeState>,
    /// Members that did not report, whose existing row is kept
    pub kept: Vec<String>,
}

impl Reconciled {
    pub fn changed(&self) -> bool {
        !self.write.is_empty()
    }
}

/// Decides what one pass writes to the board.
///
/// `reported` is what each peer said about itself, `None` for one that could
/// not be reached or runs a release that does not report a row. `held` names
/// the members the board already has a row for.
///
/// Three rules, and the middle one is the one that matters:
///
/// - A member that reported gets its row written, because a member's own
///   journal beats anything another node inferred about it.
/// - A member that did not report and already has a row keeps it. A node is
///   unreachable precisely while it is restarting, which is when the
///   coordinator's row for it is the accurate one, so clearing it there would
///   throw away the only true statement about that node.
/// - A member that did not report and has no row at all is recorded as not
///   reachable, so it appears on the board rather than being missing. A node
///   absent from the board reads as a node that does not exist
pub fn plan(
    reported: &[(String, Option<NodeUpgradeState>)],
    held: &[String],
    now_secs: u64,
) -> Reconciled {
    let mut out = Reconciled::default();
    for (name, state) in reported {
        match state {
            Some(state) => out.write.push(state.clone()),
            None if held.iter().any(|h| h == name) => out.kept.push(name.clone()),
            None => out.write.push(NodeUpgradeState {
                node_id: name.clone(),
                from_version: String::new(),
                to_version: String::new(),
                phase: zyron_common::format::UpgradePhase::Idle,
                started_at_secs: 0,
                updated_at_secs: now_secs,
                is_leader: false,
                message: "this node has not answered a status probe".to_string(),
            }),
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use zyron_common::format::UpgradePhase;

    fn state(node: &str, phase: UpgradePhase) -> NodeUpgradeState {
        NodeUpgradeState {
            node_id: node.to_string(),
            from_version: "0.12.0".to_string(),
            to_version: "0.12.1".to_string(),
            phase,
            started_at_secs: 10,
            updated_at_secs: 20,
            is_leader: false,
            message: String::new(),
        }
    }

    #[test]
    fn a_member_that_reported_has_its_own_row_written() {
        let reported = vec![
            (
                "node-2".to_string(),
                Some(state("node-2", UpgradePhase::Rolling)),
            ),
            (
                "node-3".to_string(),
                Some(state("node-3", UpgradePhase::Staging)),
            ),
        ];
        let decided = plan(&reported, &[], 100);
        assert!(decided.changed());
        assert_eq!(decided.write.len(), 2);
        assert_eq!(decided.write[0].phase, UpgradePhase::Rolling);
        assert_eq!(decided.write[1].phase, UpgradePhase::Staging);
        assert!(decided.kept.is_empty());
    }

    /// A node is unreachable exactly while it restarts, which is when the
    /// coordinator's row for it is the accurate one. Clearing it there would
    /// throw away the only true statement about that node
    #[test]
    fn a_member_that_went_quiet_keeps_the_row_the_board_already_had() {
        let reported = vec![("node-2".to_string(), None)];
        let decided = plan(&reported, &["node-2".to_string()], 100);
        assert!(!decided.changed(), "a quiet member overwrote its own row");
        assert_eq!(decided.kept, vec!["node-2".to_string()]);
    }

    /// A member missing from the board reads as a member that does not exist,
    /// so one that has never reported is recorded rather than left out
    #[test]
    fn a_member_that_never_reported_is_recorded_as_unreachable() {
        let reported = vec![("node-4".to_string(), None)];
        let decided = plan(&reported, &["node-2".to_string()], 555);
        assert_eq!(decided.write.len(), 1);
        let row = &decided.write[0];
        assert_eq!(row.node_id, "node-4");
        assert_eq!(row.phase, UpgradePhase::Idle);
        assert_eq!(row.updated_at_secs, 555);
        assert!(row.message.contains("not answered"), "{}", row.message);
        assert!(decided.kept.is_empty());
    }

    #[test]
    fn a_pass_with_nothing_to_report_writes_nothing() {
        let decided = plan(&[], &["node-1".to_string()], 100);
        assert!(!decided.changed());
        assert!(decided.kept.is_empty());
    }

    /// A reported row replaces a held one, so a member coming back from a
    /// restart corrects whatever was recorded while it was away
    #[test]
    fn a_member_that_answers_again_replaces_the_row_that_was_held_for_it() {
        let reported = vec![(
            "node-2".to_string(),
            Some(state("node-2", UpgradePhase::Completed)),
        )];
        let decided = plan(&reported, &["node-2".to_string()], 100);
        assert_eq!(decided.write.len(), 1);
        assert_eq!(decided.write[0].phase, UpgradePhase::Completed);
        assert!(decided.kept.is_empty());
    }
}
