//! Storage tiers a columnar segment can live on.
//!
//! A tier is a directory the operator points at storage of a given speed and
//! price. The engine keeps one representation of the tier and one cost table
//! for it, because a second copy of either is how the planner and the
//! relocation path come to disagree about what a tier costs.

use crate::{Result, ZyronError};

/// Where a segment's bytes live, coldest tiers last. The discriminant is
/// what the catalog stores, so the order is part of the on-disk form.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum StorageTier {
    Hot = 0,
    Warm = 1,
    Cold = 2,
    Archive = 3,
}

impl StorageTier {
    /// Reads the catalog's stored byte. An unknown value reads as Hot, which
    /// names where the fold writes and is where a segment with no recorded
    /// tier actually is.
    pub fn from_u8(v: u8) -> StorageTier {
        match v {
            1 => StorageTier::Warm,
            2 => StorageTier::Cold,
            3 => StorageTier::Archive,
            _ => StorageTier::Hot,
        }
    }

    pub fn parse(s: &str) -> Result<StorageTier> {
        match s.to_ascii_lowercase().as_str() {
            "hot" => Ok(StorageTier::Hot),
            "warm" => Ok(StorageTier::Warm),
            "cold" => Ok(StorageTier::Cold),
            "archive" => Ok(StorageTier::Archive),
            other => Err(ZyronError::Internal(format!("unknown tier: {other}"))),
        }
    }

    /// The tier's name as SQL spells it, and as its segment directory is
    /// named on disk.
    pub fn name(&self) -> &'static str {
        match self {
            StorageTier::Hot => "hot",
            StorageTier::Warm => "warm",
            StorageTier::Cold => "cold",
            StorageTier::Archive => "archive",
        }
    }

    /// Scan-cost multiplier the planner applies for data on this tier.
    pub fn cost_multiplier(&self) -> f64 {
        match self {
            StorageTier::Hot => 1.0,
            StorageTier::Warm => 1.5,
            StorageTier::Cold => 4.0,
            StorageTier::Archive => 20.0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_colder_tiers_cost_more_to_read() {
        assert!(StorageTier::Hot.cost_multiplier() < StorageTier::Warm.cost_multiplier());
        assert!(StorageTier::Warm.cost_multiplier() < StorageTier::Cold.cost_multiplier());
        assert!(StorageTier::Cold.cost_multiplier() < StorageTier::Archive.cost_multiplier());
    }

    #[test]
    fn test_names_round_trip_through_parse() {
        for t in [
            StorageTier::Hot,
            StorageTier::Warm,
            StorageTier::Cold,
            StorageTier::Archive,
        ] {
            assert_eq!(StorageTier::parse(t.name()).expect("parses"), t);
            assert_eq!(StorageTier::from_u8(t as u8), t);
        }
        assert!(StorageTier::parse("frozen").is_err());
    }
}
