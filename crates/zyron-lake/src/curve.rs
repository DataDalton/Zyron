//! Ordering curves, defined in zyron-common.
//!
//! Both storage tiers order rows by the same curves, the lake writer for a
//! .zyr file and the heap fold tier for the segments it produces, so the
//! definition lives where both can reach it and this crate names it.

pub use zyron_common::curve::{normalize_component, ordering_key, ordering_key_into};
