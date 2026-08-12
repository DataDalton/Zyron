//! Clustering policy: how a table's layout is chosen and when it is
//! revisited.
//!
//! These are the persisted encodings, so they live here rather than in
//! the parser: the catalog stores the codes, the metrics render them,
//! and the parser produces them. One definition, one numbering.

/// Physical ordering strategy for one clustering key.
///
/// The numbering is persisted in the lake manifest and the catalog, so it
/// is fixed. What each curve is for lives with the implementation in the
/// lake crate, this is the name and the code
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ClusterStrategy {
    /// Z-order bit interleave across the keys
    BitInterleave,
    /// Hilbert space filling curve
    SpaceFilling,
    /// Contiguous value ranges per file
    #[default]
    RangePartition,
    /// Spread the key across files, for shard keys
    AntiCluster,
}

impl ClusterStrategy {
    pub fn to_u8(self) -> u8 {
        match self {
            ClusterStrategy::BitInterleave => 0,
            ClusterStrategy::SpaceFilling => 1,
            ClusterStrategy::RangePartition => 2,
            ClusterStrategy::AntiCluster => 3,
        }
    }

    pub fn from_u8(v: u8) -> Option<Self> {
        Some(match v {
            0 => ClusterStrategy::BitInterleave,
            1 => ClusterStrategy::SpaceFilling,
            2 => ClusterStrategy::RangePartition,
            3 => ClusterStrategy::AntiCluster,
            _ => return None,
        })
    }

    pub fn as_str(self) -> &'static str {
        match self {
            ClusterStrategy::BitInterleave => "BitInterleave",
            ClusterStrategy::SpaceFilling => "SpaceFilling",
            ClusterStrategy::RangePartition => "RangePartition",
            ClusterStrategy::AntiCluster => "AntiCluster",
        }
    }

    /// Resolves the name written in `CLUSTER BY (col USING <strategy>)`.
    /// The two curves users know by another name accept it
    pub fn from_name(name: &str) -> Option<Self> {
        Some(match name.to_ascii_lowercase().as_str() {
            "bitinterleave" | "zorder" => ClusterStrategy::BitInterleave,
            "spacefilling" | "hilbert" => ClusterStrategy::SpaceFilling,
            "rangepartition" | "range" => ClusterStrategy::RangePartition,
            "anticluster" => ClusterStrategy::AntiCluster,
            _ => return None,
        })
    }
}

/// One clustering key with its strategy.
///
/// Both storage tiers order rows by these, the lake writer for a .zyr file
/// and the heap fold tier for the segments it produces, so the type lives
/// here and neither crate owns it
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ClusterKey {
    pub column_id: u32,
    pub strategy: ClusterStrategy,
    /// Strategy parameter, range boundary count or interleave bits, zero
    /// when the strategy takes none
    pub param: u32,
}

/// How the clustering choice interacts with measurement.
///
/// The three values are three states of knowledge. Force is the operator
/// knowing something measurement cannot see. Auto is nobody knowing, so
/// measurement decides. Hybrid is the operator knowing part of it and
/// measurement filling in around keys it may not touch.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ClusterMode {
    /// Operator-pinned layout, `CLUSTER BY (a, b)` or `... FORCE`
    #[default]
    Force,
    /// Measurement decides everything, `CLUSTER BY AUTO`
    Auto,
    /// Listed keys are anchors, measurement fills in, `CLUSTER BY (a) AUTO`
    Hybrid,
}

impl ClusterMode {
    pub fn to_u8(self) -> u8 {
        match self {
            ClusterMode::Force => 0,
            ClusterMode::Auto => 1,
            ClusterMode::Hybrid => 2,
        }
    }

    pub fn from_u8(v: u8) -> Option<Self> {
        Some(match v {
            0 => ClusterMode::Force,
            1 => ClusterMode::Auto,
            2 => ClusterMode::Hybrid,
            _ => return None,
        })
    }

    pub fn as_str(self) -> &'static str {
        match self {
            ClusterMode::Force => "FORCE",
            ClusterMode::Auto => "AUTO",
            ClusterMode::Hybrid => "HYBRID",
        }
    }

    /// True when measurement is allowed to change the layout. Under Force
    /// measurement keeps running and keeps reporting, it just does not
    /// get to act
    pub fn measurement_decides(self) -> bool {
        matches!(self, ClusterMode::Auto | ClusterMode::Hybrid)
    }
}

/// When clustering maintenance runs for a table.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ClusteringSchedule {
    /// Only when `OPTIMIZE` asks for it
    #[default]
    OnDemand,
    /// Background passes fold newly appended files into the clustered set
    Incremental,
    /// Background passes run whenever there is drift to remove
    Continuous,
}

impl ClusteringSchedule {
    pub fn to_u8(self) -> u8 {
        match self {
            ClusteringSchedule::OnDemand => 0,
            ClusteringSchedule::Incremental => 1,
            ClusteringSchedule::Continuous => 2,
        }
    }

    pub fn from_u8(v: u8) -> Option<Self> {
        Some(match v {
            0 => ClusteringSchedule::OnDemand,
            1 => ClusteringSchedule::Incremental,
            2 => ClusteringSchedule::Continuous,
            _ => return None,
        })
    }

    pub fn as_str(self) -> &'static str {
        match self {
            ClusteringSchedule::OnDemand => "ONDEMAND",
            ClusteringSchedule::Incremental => "INCREMENTAL",
            ClusteringSchedule::Continuous => "CONTINUOUS",
        }
    }

    /// True when a background worker may start a pass without being asked
    pub fn runs_in_background(self) -> bool {
        matches!(
            self,
            ClusteringSchedule::Incremental | ClusteringSchedule::Continuous
        )
    }
}

/// Why a clustering proposal was accepted or refused, as a metric label.
/// The gate's decision type carries the numbers, this carries the name
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ClusterDecision {
    Accepted,
    RejectedWorse,
    RejectedBelowThreshold,
    RejectedAnchorConflict,
    RejectedReplayDiverged,
}

impl ClusterDecision {
    pub fn as_str(self) -> &'static str {
        match self {
            ClusterDecision::Accepted => "accepted",
            ClusterDecision::RejectedWorse => "rejected_worse",
            ClusterDecision::RejectedBelowThreshold => "rejected_below_threshold",
            ClusterDecision::RejectedAnchorConflict => "rejected_anchor_conflict",
            ClusterDecision::RejectedReplayDiverged => "rejected_replay_diverged",
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_codes_round_trip_and_stay_put() {
        // The numbering is persisted in the catalog, so it is fixed
        for (mode, code) in [
            (ClusterMode::Force, 0u8),
            (ClusterMode::Auto, 1),
            (ClusterMode::Hybrid, 2),
        ] {
            assert_eq!(mode.to_u8(), code);
            assert_eq!(ClusterMode::from_u8(code), Some(mode));
        }
        assert_eq!(ClusterMode::from_u8(3), None);
        for (schedule, code) in [
            (ClusteringSchedule::OnDemand, 0u8),
            (ClusteringSchedule::Incremental, 1),
            (ClusteringSchedule::Continuous, 2),
        ] {
            assert_eq!(schedule.to_u8(), code);
            assert_eq!(ClusteringSchedule::from_u8(code), Some(schedule));
        }
        assert_eq!(ClusteringSchedule::from_u8(3), None);
    }

    #[test]
    fn test_force_keeps_measuring_but_does_not_act() {
        assert!(!ClusterMode::Force.measurement_decides());
        assert!(ClusterMode::Auto.measurement_decides());
        assert!(ClusterMode::Hybrid.measurement_decides());
        assert!(!ClusteringSchedule::OnDemand.runs_in_background());
        assert!(ClusteringSchedule::Incremental.runs_in_background());
        assert!(ClusteringSchedule::Continuous.runs_in_background());
    }
}
