//! ZyronLake table format.
//!
//! A lake table stores immutable .zyr data files addressed by a self
//! contained transaction log. Visibility is by log version: every commit
//! writes one numbered version file under the table's log directory with
//! File::create_new as the optimistic concurrency primitive, and periodic
//! manifest checkpoints collapse the log so readers open one file instead
//! of replaying history.
//!
//! This crate deliberately depends on zyron-common only, so the catalog,
//! planner and executor can depend on it without a cycle.

mod cells;
pub mod branch;
pub mod changefeed;
mod codec;
pub mod constraints;
pub mod convert;
pub mod curve;
pub mod encoded_filter;
pub mod feedback;
pub mod follow;
mod hll;
pub mod crosstable;
pub mod history;
pub mod index;

pub mod maintenance;
pub mod manifest;
pub mod operations;
pub mod paths;
pub mod planner;
pub mod predicate;
pub mod prune_index;
pub mod reader;
pub mod repair;
pub mod schema;
pub mod time_travel;
pub mod transaction_log;
pub mod workload;
pub mod writer;

pub use manifest::{
    ClusterKey, ClusterMode, ClusterSpec, ClusterStrategy, ClusteringSchedule, ColumnStatsEntry,
    DeletePredicate, FileStats, ManifestFile, PartitionEntry, CLUSTERING_ANCHORS_PROPERTY,
    CLUSTERING_MODE_PROPERTY, CLUSTERING_SCHEDULE_PROPERTY,
};
pub use operations::{
    append_rows, delete_all, delete_where, optimize, update_where, vacuum_data_files,
    AppendOutcome, DeleteOutcome, OptimizeOutcome, UpdateOutcome,
};
pub use branch::{
    branch_info, create_branch, drop_branch, list_branches, merge_branch, open_branch,
    open_branch_shared, BranchInfo, MergeOutcome,
};
pub use changefeed::{
    change_row_counts, changed_ordinals, changes_between, ChangeDescriptor, ChangeKind,
};
pub use constraints::{
    check_foreign_key, check_unique, check_unique_replacing, ForeignKeyOutcome, UniqueCheckStats,
    UniqueOutcome, UniqueSpec,
};
pub use curve::{normalize_component, ordering_key};
pub use encoded_filter::StoredFilter;
pub use follow::{
    apply_versions, decode_hex, decode_log_rows, leader_head, load_cursor, read_versions_after,
    sync, FollowedVersion, Freshness,
};
pub use maintenance::{
    resume_cluster_passes, run_cluster_pass, ClusterPassOptions, ClusterPassOutcome, PassState,
    ResumeOptions, DEFAULT_MAX_INPUTS, DEFAULT_ROWS_PER_FILE,
};
pub use feedback::{evaluate, skip_rate, Decision, GateConfig, PredicateClass};
pub use planner::{
    bootstrap, choose_strategy, evidence_from_manifest, measured_selectivity, measured_skip_rate,
    predicate_classes, propose, ColumnEvidence,
};
pub use workload::{
    column_term, current_epoch, epoch_of, observe_scan, observe_scan_result, observer, term_column,
    ObserverStats, WorkloadObserver, TERM_BYTES_CONSIDERED, TERM_BYTES_SKIPPED, TERM_EQUALITY,
    TERM_RANGE, TERM_ROWS_MATCHED, TERM_ROWS_SCANNED, TERMS_PER_COLUMN,
};
pub use convert::{load_lake_from_rows, read_all_rows, reclaim_orphan_root};
pub use crosstable::{
    clear_recovered_intents, intent_state, recover_intents, CrossTableTxn, IntentAware,
    IntentRecovery, IntentState, INTENT_TXN_FLAG,
};
pub use history::{
    diff_versions, schema_at_version, table_history, version_details, version_files,
    version_lineage, VersionDetails, VersionDiff, VersionRecord,
};
pub use paths::LakePaths;
pub use repair::{
    cleanup_orphans, repair, validate, OrphanReport, Problem, RepairOptions, RepairReport,
    ValidationReport,
};
pub use predicate::{ColumnBounds, CompareOp, LakePredicate, LakeValue, PruneDecision, StatsSource};
pub use prune_index::{with_sweep, PruneIndex, PruneScratch};
pub use schema::{LakeColumn, LakeSchema};
pub use reader::{evaluate_row, DecodedColumn, LakeFileReader, ZoneVerdict};
pub use index::{
    covers_table, entries_for_file, group_by_partition, index_schema, probe_equal, probe_range,
    value_to_index_cell, write_index_files, IndexBatch, IndexFileEntry, IndexProbeStats,
    LakeIndexSpec, RangeBound, RowAddress, ENTRIES_PER_INDEX_FILE,
};
pub use time_travel::{manifest_as_of, resolve_version, TimeTravelSpec};
pub use transaction_log::{
    abandon_txn, local_node, publish_txn, register_txn_pending, set_local_node, transfer_writer,
    writer_node, AllCommitted, CommitAttempt, CommitHeader, CommitInfo, CommitStatus, LogEntry,
    OperationKind, TransactionLog, VersionFileData, WRITER_NODE_PROPERTY,
};
pub use writer::{write_data_file, ColumnData, WriteRequest};
