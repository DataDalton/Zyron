//! ZyronDB common types, errors, and utilities.
//!
//! This crate provides shared definitions used across all ZyronDB components.

pub mod checksum;
pub mod config;
pub mod error;
pub mod interval;
pub mod page;
pub mod types;
pub mod zerocopy;

pub use checksum::{
    ALGORITHM_VERSION, Hasher, ZyBuildHasher, ZyBuildHasherSeeded, hash32, hash32_seeded, hash64,
    hash64_seeded, hash128, hash128_seeded,
};
pub use config::{ServerConfig, StorageConfig};
pub use error::{Result, ZyronError};
pub use interval::{
    Interval, days_from_ymd, days_in_month, is_leap, parse_interval_string, ymd_from_days,
};
pub use page::{PAGE_SIZE, PageHeader, PageId};
pub use types::TypeId;
