//! Tier model re-export for lifecycle policy handlers.
//!
//! A tier is a directory the relocation moves segment files into, and a read
//! of a cold segment is the same positioned read as a hot one, so there is no
//! per-tier transport or fetch cache here. The tier model itself lives in
//! zyron-common so the planner can cost a scan by the tier its segments sit
//! on without depending on this crate. One definition, one cost table.

pub use zyron_common::StorageTier;
