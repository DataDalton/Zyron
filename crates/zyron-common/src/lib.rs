//! Zyron common types, errors, and utilities.
//!
//! This crate provides shared definitions used across all Zyron components.

pub mod checksum;
pub mod config;
pub mod error;
pub mod interval;
pub mod obs_metrics;
pub mod page;
pub mod prng;
pub mod profile;
pub mod types;
pub mod zerocopy;

pub use checksum::{
    ALGORITHM_VERSION, FX_K, Hasher, IdentityBuildHasher, IdentityHasher, PreHashMap,
    ZyBuildHasher, ZyBuildHasherSeeded, fx_finalize, fx_mix, hash32, hash32_seeded, hash64,
    hash64_seeded, hash128, hash128_seeded,
};
pub use config::{ServerConfig, StorageConfig};
pub use error::{Result, ZyronError};
pub use interval::{
    Interval, days_from_ymd, days_in_month, is_leap, parse_date_days, parse_interval_string,
    parse_timestamp_micros, ymd_from_days,
};
pub use obs_metrics::{LabeledMetrics, TlsDirection};
pub use page::{BranchCatalog, BranchFiles, PAGE_SIZE, PageHeader, PageId};
pub use prng::{ReservoirL, Xoshiro256pp, splitMix64};
pub use types::TypeId;
