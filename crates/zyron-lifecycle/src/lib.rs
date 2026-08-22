//! Data lifecycle management for Zyron.
//!
//! Retention/TTL, tiered storage, archival, soft delete, legal hold,
//! GDPR erasure, data classification, compliance auditing, recycle bin,
//! crypto-shredding, and a tamper-evident audit chain.
//!
//! Hot-path reads (legal hold lookup) are lock-free via RcuMap and atomics.
//! Background workers own their own threads and may use local locks. No
//! unwrap on any path; all fallible work returns Result.

pub mod archive;
pub mod audit_chain;
pub mod classification;
pub mod compliance;
pub mod cryptoshred;
pub mod dryrun;
pub mod dsar;
pub mod erasure;
pub mod legal_hold;
pub mod presets;
pub mod recyclebin;
pub mod residency;
pub mod scheduling;
pub mod soft_delete;
pub mod tiered_storage;
pub mod ttl;
pub mod worm;
