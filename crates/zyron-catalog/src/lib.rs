//! Catalog module for ZyronDB.
//!
//! Manages database metadata: databases, schemas, tables, columns, indexes,
//! constraints, and statistics. Persists catalog entries in heap file system
//! tables, caches them in memory with LRU eviction, and logs DDL operations
//! to the WAL for crash safety.

pub mod cache;
pub mod catalog;
pub mod encoding;
pub mod ids;
pub mod resolver;
pub mod schema;
pub mod stats;
pub mod storage;

pub use cache::CatalogCache;
pub use catalog::Catalog;
pub use ids::{
    CATALOG_SCHEMA_ID, ColumnId, DEFAULT_SCHEMA_ID, DatabaseId, IndexId, Oid, OidAllocator,
    SYSTEM_DATABASE_ID, SchemaId, SequenceId, TableId, USER_OID_START,
};
pub use resolver::NameResolver;
pub use schema::{
    ColumnEntry, ConstraintEntry, ConstraintType, DatabaseEntry, IndexColumnEntry, IndexEntry,
    IndexType, SchemaEntry, TableEntry,
};
pub use stats::{ColumnStats, Histogram, TableStats, analyze_table};
pub use storage::{CatalogStorage, HeapCatalogStorage};
