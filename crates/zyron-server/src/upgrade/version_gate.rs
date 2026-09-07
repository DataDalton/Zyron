//! The cluster version gate.
//!
//! The members of one group run two adjacent releases for the length of a
//! rolling upgrade, and a rolled-back member runs the older one for as long
//! as the pause after it lasts. Anything one member puts in front of the
//! others, a kind of log entry, a mesh call, a field in a message, is used
//! only once every member runs a binary that reads it. This is where that is
//! decided. The gate holds the version each member last answered with,
//! takes the lowest as the group's floor, and says whether a thing
//! introduced at a given release may be used.
//!
//! A member that did not answer has no known version, and the floor is then
//! unknown rather than taken over the members that did answer. A member mid
//! restart comes back on either the new binary or the old one, and the gate
//! cannot tell which until it answers. A member that is gone for good is
//! removed from the group, which takes it out of the gate.
//!
//! A reading is trusted for a short time so a burst of gated uses shares
//! one round of probes, and the driver drops it whenever it restarts or
//! rolls back a member, since those are the moments a version changes

use std::time::{Duration, Instant};

use zyron_common::format::BinaryVersion;

/// How long a reading is trusted before the members are asked again. A
/// version changes when a node restarts, which the driver drops the reading
/// for when it drove the restart, and a restart by hand is seen inside this
pub const READING_TTL: Duration = Duration::from_secs(5);

/// The release that added the status probe the floor is read through.
///
/// A member older than this serves no such path and answers that it does not
/// know it, so its version cannot be asked for at all. That is a different
/// fact from a member being unreachable, and the two lead an operator to
/// different places, so the gate keeps them apart
pub const VERSION_PROBE_ADDED_IN: BinaryVersion = BinaryVersion::new(0, 12, 0);

/// What one member answered when asked what it runs
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MemberVersion {
    pub name: String,
    /// The version the member runs, or why it is not known
    pub version: Result<BinaryVersion, UnknownVersion>,
}

/// Why a member's version could not be read
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum UnknownVersion {
    /// The member answered that it does not serve the status path, which only
    /// a release older than `VERSION_PROBE_ADDED_IN` does. Its exact version
    /// is unreadable, but it is certainly below that release
    PredatesProbe,
    /// The member could not be reached, refused, or answered something that
    /// is not a version
    NotAnswered(String),
}

impl std::fmt::Display for UnknownVersion {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            UnknownVersion::PredatesProbe => write!(
                f,
                "it runs a release older than {VERSION_PROBE_ADDED_IN}, which is when a node \
                 could first be asked its version"
            ),
            UnknownVersion::NotAnswered(reason) => write!(f, "{reason}"),
        }
    }
}

/// The lowest version any member runs, or the reason it cannot be known
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Floor {
    /// Every member answered, and this is the lowest with the member that
    /// runs it
    Known {
        version: BinaryVersion,
        member: String,
    },
    /// A member runs a release older than the one that added the status
    /// probe, so it cannot be asked what it runs. Its exact version is
    /// unreadable and it is certainly below `VERSION_PROBE_ADDED_IN`
    Predates { member: String },
    /// A member did not answer, so nothing can be said about the group
    Unknown { member: String, reason: String },
}

impl Floor {
    /// The floor over a set of answers. A member without a version makes the
    /// floor unknown whatever the others answered, and otherwise the lowest
    /// version is the floor, the earlier member on a tie.
    ///
    /// A member that could not be reached outranks one that merely predates
    /// the probe, because an outage is the more urgent fact and an old member
    /// would otherwise mask it for the length of an upgrade
    pub fn over(members: &[MemberVersion]) -> Floor {
        if let Some(member) = members
            .iter()
            .find(|m| matches!(&m.version, Err(UnknownVersion::NotAnswered(_))))
        {
            let reason = match &member.version {
                Err(unknown) => unknown.to_string(),
                Ok(_) => String::new(),
            };
            return Floor::Unknown {
                member: member.name.clone(),
                reason,
            };
        }
        if let Some(member) = members
            .iter()
            .find(|m| matches!(&m.version, Err(UnknownVersion::PredatesProbe)))
        {
            return Floor::Predates {
                member: member.name.clone(),
            };
        }
        let mut lowest: Option<(BinaryVersion, &str)> = None;
        for member in members {
            if let Ok(version) = &member.version {
                match lowest {
                    Some((floor, _)) if floor <= *version => {}
                    _ => lowest = Some((*version, member.name.as_str())),
                }
            }
        }
        match lowest {
            Some((version, member)) => Floor::Known {
                version,
                member: member.to_string(),
            },
            None => Floor::Unknown {
                member: "the group".to_string(),
                reason: "no member was asked".to_string(),
            },
        }
    }

    /// Whether something introduced at a release may be used with this
    /// floor. The refusal names the member that holds the group back
    pub fn allows(&self, introduced: BinaryVersion) -> Result<(), String> {
        match self {
            Floor::Known { version, .. } if *version >= introduced => Ok(()),
            Floor::Known { version, member } => Err(format!(
                "{member} runs {version}, and this needs every member of the group at \
                 {introduced} or later"
            )),
            // Certainly below the probe's own release, so anything introduced
            // there or later is refused on a fact rather than on not knowing.
            // Naming the release sends an operator to finish the upgrade
            // instead of hunting a network fault that is not there
            Floor::Predates { member } if introduced >= VERSION_PROBE_ADDED_IN => Err(format!(
                "{member} runs a release older than {VERSION_PROBE_ADDED_IN}, and this needs \
                 every member of the group at {introduced} or later"
            )),
            Floor::Predates { member } => Err(format!(
                "the version {member} runs cannot be read, it runs a release older than \
                 {VERSION_PROBE_ADDED_IN}, and this needs every member of the group at \
                 {introduced} or later"
            )),
            Floor::Unknown { member, reason } => Err(format!(
                "the version {member} runs is not known, {reason}, and this needs every member \
                 of the group at {introduced} or later"
            )),
        }
    }
}

/// Whether a peer's refusal to answer a status probe means it is too old to
/// have the path, rather than that something else went wrong.
///
/// A peer answers 404 both when it has no route for the path and when the
/// call was addressed to a different node, and both arrive as the same error.
/// The two are told apart by what the peer says it does not know: the path
/// itself, or a node. Only the first is an old release
pub fn predates_probe(what: &str) -> bool {
    what == zyron_mesh::PATH_NODE_STATUS
}

/// One reading of the floor and when it was taken
#[derive(Debug, Clone)]
struct Reading {
    floor: Floor,
    taken_at: Instant,
}

/// The gate this node holds, read and refreshed by the driver
#[derive(Debug, Default)]
pub struct VersionGate {
    reading: parking_lot::Mutex<Option<Reading>>,
}

impl VersionGate {
    pub fn new() -> Self {
        Self::default()
    }

    /// The floor from the last reading, while that reading is fresh
    pub fn fresh(&self, now: Instant) -> Option<Floor> {
        let reading = self.reading.lock();
        reading
            .as_ref()
            .filter(|r| now.saturating_duration_since(r.taken_at) < READING_TTL)
            .map(|r| r.floor.clone())
    }

    /// Records a reading taken now
    pub fn record(&self, floor: Floor, now: Instant) {
        *self.reading.lock() = Some(Reading {
            floor,
            taken_at: now,
        });
    }

    /// Drops the reading, for a moment a member's version is known to change
    pub fn invalidate(&self) {
        *self.reading.lock() = None;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn runs(name: &str, version: &str) -> MemberVersion {
        MemberVersion {
            name: name.to_string(),
            version: BinaryVersion::parse(version)
                .ok_or_else(|| UnknownVersion::NotAnswered(format!("`{version}` does not parse"))),
        }
    }

    fn silent(name: &str) -> MemberVersion {
        MemberVersion {
            name: name.to_string(),
            version: Err(UnknownVersion::NotAnswered(
                "it did not answer a status probe".to_string(),
            )),
        }
    }

    /// A member from before the status path existed, which answers that it
    /// does not know it
    fn predates(name: &str) -> MemberVersion {
        MemberVersion {
            name: name.to_string(),
            version: Err(UnknownVersion::PredatesProbe),
        }
    }

    #[test]
    fn test_the_floor_is_the_lowest_version_and_names_who_runs_it() {
        let floor = Floor::over(&[
            runs("node-1", "0.12.0"),
            runs("node-2", "0.11.0"),
            runs("node-3", "0.12.1"),
        ]);
        assert_eq!(
            floor,
            Floor::Known {
                version: BinaryVersion::new(0, 11, 0),
                member: "node-2".to_string(),
            }
        );
        floor
            .allows(BinaryVersion::new(0, 11, 0))
            .expect("the floor itself is allowed");
        let refusal = floor
            .allows(BinaryVersion::new(0, 12, 0))
            .expect_err("held");
        assert!(refusal.contains("node-2 runs 0.11.0"), "{refusal}");
        assert!(refusal.contains("0.12.0 or later"), "{refusal}");
    }

    #[test]
    fn test_a_tie_names_the_earlier_member() {
        let floor = Floor::over(&[runs("node-1", "0.12.0"), runs("node-2", "0.12.0")]);
        assert_eq!(
            floor,
            Floor::Known {
                version: BinaryVersion::new(0, 12, 0),
                member: "node-1".to_string(),
            }
        );
    }

    #[test]
    fn test_one_silent_member_makes_the_floor_unknown_whatever_the_others_run() {
        let floor = Floor::over(&[
            runs("node-1", "0.12.0"),
            silent("node-2"),
            runs("node-3", "0.11.0"),
        ]);
        assert_eq!(
            floor,
            Floor::Unknown {
                member: "node-2".to_string(),
                reason: "it did not answer a status probe".to_string(),
            }
        );
        let refusal = floor
            .allows(BinaryVersion::new(0, 12, 0))
            .expect_err("held");
        assert!(refusal.contains("node-2"), "{refusal}");
        assert!(refusal.contains("did not answer"), "{refusal}");
    }

    /// A member from before the status path existed is reported as old, not
    /// as broken.
    ///
    /// This is the arrangement of the first upgrade the gate ever sees: the
    /// release that added the status path is the release being upgraded to,
    /// so every member still on the old one answers that it does not know
    /// the path. Calling that a failed probe would send an operator hunting a
    /// network fault through the whole rollout
    #[test]
    fn test_a_member_from_before_the_probe_is_reported_as_old_not_as_silent() {
        let floor = Floor::over(&[runs("node-1", "0.12.0"), predates("node-2")]);
        assert_eq!(
            floor,
            Floor::Predates {
                member: "node-2".to_string(),
            }
        );
        let refusal = floor
            .allows(BinaryVersion::new(0, 12, 0))
            .expect_err("a member below the probe's release holds this back");
        assert!(refusal.contains("node-2"), "{refusal}");
        assert!(
            refusal.contains("older than 0.12.0"),
            "the refusal has to name the release, got {refusal}"
        );
        assert!(
            !refusal.contains("not known"),
            "an old member is not an unknown one, got {refusal}"
        );
    }

    /// An outage is the more urgent fact, and an old member would otherwise
    /// mask it for the length of an upgrade
    #[test]
    fn test_a_silent_member_outranks_one_that_only_predates_the_probe() {
        let floor = Floor::over(&[predates("node-2"), silent("node-3")]);
        assert!(
            matches!(&floor, Floor::Unknown { member, .. } if member == "node-3"),
            "{floor:?}"
        );
    }

    /// A peer answers 404 both for a path it has no route for and for a call
    /// addressed to another node. Only the first is an old release, and
    /// mistaking a misrouted call for one would report a routing fault as a
    /// version that needs upgrading
    #[test]
    fn test_only_a_missing_path_reads_as_an_old_release() {
        assert!(predates_probe(zyron_mesh::PATH_NODE_STATUS));
        assert!(!predates_probe("node 7 is not this node"));
        assert!(!predates_probe("ticket"));
        assert!(!predates_probe(""));
        assert!(
            !predates_probe("/internal/mesh/v1/prefetch"),
            "another mesh path is not the status probe"
        );
    }

    /// Below the probe's own release the gate genuinely cannot say, because
    /// a member that predates it could be any older release
    #[test]
    fn test_a_member_that_predates_the_probe_still_refuses_an_older_requirement() {
        let floor = Floor::over(&[predates("node-2")]);
        let refusal = floor
            .allows(BinaryVersion::new(0, 9, 0))
            .expect_err("its exact version is still unreadable");
        assert!(refusal.contains("cannot be read"), "{refusal}");
    }

    #[test]
    fn test_a_version_that_does_not_parse_is_unknown() {
        let floor = Floor::over(&[runs("node-1", "0.12.0"), runs("node-2", "nightly")]);
        assert!(
            matches!(&floor, Floor::Unknown { member, reason }
                if member == "node-2" && reason.contains("does not parse")),
            "{floor:?}"
        );
        assert!(matches!(Floor::over(&[]), Floor::Unknown { .. }));
    }

    #[test]
    fn test_a_reading_is_trusted_inside_the_ttl_and_dropped_on_demand() {
        let gate = VersionGate::new();
        let now = Instant::now();
        assert_eq!(gate.fresh(now), None);
        let floor = Floor::Known {
            version: BinaryVersion::new(0, 12, 0),
            member: "node-1".to_string(),
        };
        gate.record(floor.clone(), now);
        assert_eq!(gate.fresh(now), Some(floor.clone()));
        assert_eq!(
            gate.fresh(now + READING_TTL - Duration::from_millis(1)),
            Some(floor)
        );
        assert_eq!(gate.fresh(now + READING_TTL), None);
        gate.record(
            Floor::Unknown {
                member: "node-2".to_string(),
                reason: "restarting".to_string(),
            },
            now,
        );
        gate.invalidate();
        assert_eq!(gate.fresh(now), None);
    }
}
