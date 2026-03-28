//! Data versioning, time travel, SCD, temporal tables, branching, and diff
//! for ZyronDB.

pub mod branch;
pub mod diff;
pub mod page_version_map;
pub mod scd;
pub mod snapshot;
pub mod temporal;
pub mod version;

pub use branch::{BranchEntry, BranchId, BranchManager, ConflictEntry, MergeResult};
pub use diff::{ColumnDiff, DiffFormat, DiffStats, DiffType, RowDiff, TableDiff};
pub use page_version_map::{PageVersionMap, PageVersionRange};
pub use scd::{ScdActions, ScdConfig, ScdHandler, ScdType, SurrogateKeyGenerator};
pub use snapshot::{SnapshotReader, TableSnapshot};
pub use temporal::{
    ApplicationTimeTable, BiTemporalTable, MAX_TIMESTAMP, SystemVersionedTable, TemporalConfig,
    TemporalQuery, TemporalType,
};
pub use version::{OperationType, VersionEntry, VersionId, VersionLog};
