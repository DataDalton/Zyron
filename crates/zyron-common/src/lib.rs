//! Zyron common types, errors, and utilities.
//!
//! This crate provides shared definitions used across all Zyron components.

pub mod array_value;
pub mod checksum;
pub mod cluster;
pub mod config;
pub mod curve;
pub mod decimal;
pub mod doc_registry;
pub mod error;
pub mod foreign;
pub mod interval;
pub mod io_stats;
pub mod node;
pub mod obs_metrics;
pub mod page;
pub mod prng;
pub mod profile;
pub mod row_locator;
pub mod storage_tier;
pub mod types;
pub mod zerocopy;

pub use array_value::ArrayView;
pub use checksum::{
    ALGORITHM_VERSION, FX_K, Hasher, IdentityBuildHasher, IdentityHasher, PreHashMap,
    ZyBuildHasher, ZyBuildHasherSeeded, fx_mix, hash32, hash32_seeded, hash64, hash64_seeded,
    hash128, hash128_seeded, mix_finalize_2round, mix_finalize_3round,
};
pub use cluster::{ClusterDecision, ClusterKey, ClusterMode, ClusterStrategy, ClusteringSchedule};
pub use config::{DeploymentMode, ServerConfig, StorageConfig};
pub use curve::{CellFamily, cell_family, normalize_component, ordering_key};
pub use decimal::{
    MAX_DECIMAL_SCALE, check_precision, decimal_from_f64, decimal_to_f64, format_decimal,
    parse_decimal, rescale,
};
pub use doc_registry::DocRegistry;
pub use error::{Result, ZyronError};
pub use foreign::ForeignRequest;
pub use interval::{
    Interval, days_from_ymd, days_in_month, is_leap, parse_date_days, parse_interval_string,
    parse_timestamp_micros, ymd_from_days,
};
pub use io_stats::{IndexIOStats, IndexIOStatsRegistry, TableIOStats, TableIOStatsRegistry};
pub use node::{
    IDENTITY_FILE, NodeIdentity, PEERS_FILE, PeerEntry, PeerRegistry, peer_timestamp_us,
};
pub use obs_metrics::{LabeledMetrics, TlsDirection};
pub use page::{BranchCatalog, BranchFiles, PAGE_SIZE, PageHeader, PageId};
pub use prng::{ReservoirL, Xoshiro256pp, splitMix64};
pub use row_locator::RowLocator;
pub use storage_tier::StorageTier;
pub use types::TypeId;
