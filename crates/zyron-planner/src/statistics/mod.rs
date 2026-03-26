//! Statistics infrastructure for cardinality estimation.
//!
//! Provides histogram-based selectivity estimation, HyperLogLog distinct count
//! estimation, reservoir sampling for bounded-memory statistics collection,
//! and mutation tracking for auto-analyze triggers.

pub mod collector;
pub mod histogram;
mod hyperloglog;

pub use hyperloglog::{HyperLogLog, hash_bytes};
