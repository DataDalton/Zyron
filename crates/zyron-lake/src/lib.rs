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

pub mod branch;
mod cells;
pub mod changefeed;
pub mod clone;
mod codec;
pub mod compaction_history;
pub mod constraints;
pub mod convert;
pub mod crosstable;
pub mod curve;
pub mod encoded_filter;
pub mod feedback;
pub mod follow;
pub mod history;
mod hll;
pub mod index;

pub mod maintenance;
pub mod maintenance_signal;
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

pub use branch::{
    BranchInfo, MergeOutcome, branch_info, create_branch, drop_branch, list_branches, merge_branch,
    open_branch, open_branch_shared,
};
pub use changefeed::{
    ChangeDescriptor, ChangeKind, change_row_counts, changed_ordinals, changes_between,
};
pub use clone::{
    CLONE_SOURCE_TABLE_PROPERTY, CLONE_SOURCE_VERSION_PROPERTY, CloneOutcome, clone_source,
    clone_table, release_pin,
};
pub use constraints::{
    ForeignKeyOutcome, UniqueCheckStats, UniqueOutcome, UniqueSpec, check_foreign_key,
    check_unique, check_unique_replacing, unique_probe_range,
};
pub use convert::{load_lake_from_rows, read_all_rows, reclaim_orphan_root};
pub use crosstable::{
    CrossTableTxn, INTENT_TXN_FLAG, IntentAware, IntentRecovery, IntentState,
    clear_recovered_intents, intent_state, recover_intents,
};
pub use curve::{normalize_component, ordering_key};
pub use encoded_filter::StoredFilter;
pub use feedback::{Decision, GateConfig, PredicateClass, evaluate, skip_rate};
pub use follow::{
    FollowedVersion, Freshness, apply_versions, decode_hex, decode_log_rows, leader_head,
    load_cursor, read_versions_after, sync,
};
pub use history::{
    VersionDetails, VersionDiff, VersionRecord, diff_versions, schema_at_version, table_history,
    version_details, version_files, version_lineage,
};
pub use index::{
    ENTRIES_PER_INDEX_FILE, IndexBatch, IndexFileEntry, IndexProbeStats, LakeIndexSpec, RangeBound,
    RowAddress, covers_table, entries_for_file, group_by_partition, index_schema,
    point_probe_read_bytes, probe_equal, probe_range, range_probe_read_bytes, scan_read_bytes,
    value_to_index_cell, write_index_files,
};
pub use maintenance::{
    ClusterPassOptions, ClusterPassOutcome, DEFAULT_MAX_INPUTS, DEFAULT_MAX_PROPOSED_KEYS,
    DEFAULT_ROWS_PER_FILE, PassState, ResumeOptions, TablePassOptions, TablePassReport,
    drifted_file_count, resume_cluster_passes, run_cluster_pass, run_table_cluster_pass,
};
pub use maintenance_signal::{
    DirtyHead, DirtyHeads, MAX_TRACKED_HEADS, MaintenanceSignal, maintenance_signal,
};
pub use manifest::{
    AUTO_COMPACT_DEAD_ROW_RATIO_PROPERTY, AUTO_COMPACT_SMALL_FILE_RATIO_PROPERTY,
    CLUSTER_REPAIR_INTERVAL_SECS_PROPERTY, CLUSTER_REPAIR_MAX_INPUTS_PROPERTY,
    CLUSTER_REPAIR_URGENCY_THRESHOLD_PROPERTY, CLUSTERING_ANCHORS_PROPERTY,
    CLUSTERING_MODE_PROPERTY, CLUSTERING_SCHEDULE_PROPERTY, ClusterKey, ClusterMode, ClusterSpec,
    ClusterStrategy, ClusteringSchedule, ColumnStatsEntry, CompactionNeed, CompactionTrigger,
    DEFAULT_AUTO_COMPACT_DEAD_ROW_RATIO, DEFAULT_AUTO_COMPACT_SMALL_FILE_RATIO,
    DEFAULT_CLUSTER_REPAIR_INTERVAL_SECS, DEFAULT_CLUSTER_REPAIR_URGENCY_THRESHOLD,
    DeletePredicate, FileStats, MIN_SMALL_FILES_TO_MERGE, ManifestFile, PartitionEntry,
    TARGET_ROWS_PER_FILE_PROPERTY,
};
pub use operations::{
    AppendOutcome, DeleteOutcome, OptimizeOutcome, RestoreOutcome, UpdateOutcome, append_rows,
    delete_all, delete_where, optimize, restore_to_version, update_where, vacuum_data_files,
};
pub use paths::LakePaths;
pub use planner::{
    ColumnEvidence, bootstrap, choose_strategy, evidence_from_manifest, measured_selectivity,
    measured_skip_rate, predicate_classes, propose,
};
pub use predicate::{
    ClusterFit, ClusterFitEstimate, ColumnBounds, CompareOp, LakePredicate, LakeValue,
    PruneDecision, StatsSource, cluster_fit_estimate,
};
pub use prune_index::{PruneIndex, PruneScratch, with_sweep};
pub use reader::{DecodedColumn, LakeFileReader, ZoneVerdict, evaluate_row};
pub use repair::{
    OrphanReport, Problem, RepairOptions, RepairReport, ValidationReport, cleanup_orphans, repair,
    validate,
};
pub use schema::{DerivedColumn, LakeColumn, LakeSchema};
pub use time_travel::{TimeTravelSpec, manifest_as_of, resolve_version};
pub use transaction_log::{
    AllCommitted, CommitAttempt, CommitHeader, CommitInfo, CommitStatus, LogEntry, OperationKind,
    TransactionLog, VersionFileData, WRITER_NODE_PROPERTY, abandon_txn, local_node, publish_txn,
    register_txn_pending, set_local_node, transfer_writer, writer_node,
};
pub use workload::{
    ObserverStats, TERM_BYTES_CONSIDERED, TERM_BYTES_SKIPPED, TERM_EQUALITY, TERM_JOIN_KEY,
    TERM_RANGE, TERM_ROWS_MATCHED, TERM_ROWS_SCANNED, TERMS_PER_COLUMN, WorkloadObserver,
    column_term, current_epoch, epoch_of, observe_join, observe_scan, observe_scan_result,
    observer, term_column,
};
pub use writer::{ColumnData, WriteRequest, write_data_file};
