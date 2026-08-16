//! Table operations composed from the writer, reader and transaction log.
//!
//! Every operation runs inside commit's build closure so a retry after a
//! lost version race re-derives everything, file choices, predicate ids
//! and statistics, against the winner's state.
//!
//! Data file identity is allocated from entropy, not a counter. Two
//! concurrent writers with a sequential counter would stage the same
//! partition path and the loser's rename would clobber the winner's data.
//! With random 64-bit ids, checked against the base manifest, a data file
//! is written once under a name nobody else can pick and the commit is
//! pure metadata, so losing the version race never touches another
//! writer's bytes.
//!
//! A predicate delete drops the files it provably covers whole, records
//! one delete predicate for the files it may match, and leaves the rest
//! untouched. Optimize is the physical application: it rewrites attached
//! files through the survivor mask and retires predicates that no longer
//! attach anywhere

use std::collections::BTreeSet;
use std::collections::hash_map::RandomState;
use std::fs;
use std::hash::{BuildHasher, Hasher};
use std::path::PathBuf;
use std::time::{SystemTime, UNIX_EPOCH};

use zyron_common::ZyronError;

use crate::index::{self, LakeIndexSpec};
use crate::manifest::{ClusterStrategy, DeletePredicate, ManifestFile, PartitionEntry};
use crate::paths::{parse_data_file_name, parse_index_file_name};
use crate::predicate::{LakePredicate, PruneDecision};
use crate::reader::LakeFileReader;
use crate::transaction_log::{CommitAttempt, LogEntry, OperationKind, TransactionLog};
use crate::writer::{ColumnData, WriteRequest, write_data_file_ordered};

/// Outcome of an append
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AppendOutcome {
    pub version: u64,
    pub partition_id: u64,
    pub rows: u64,
    /// `order[ordinal]` is the input row that landed at that ordinal.
    ///
    /// The writer sorts a batch by the table's cluster keys, so an input
    /// row's position is not its address. A caller that has to address the
    /// rows it just wrote, to maintain a search index over them, needs the
    /// permutation rather than the row count
    pub order: Vec<usize>,
}

/// Outcome of a predicate delete. `version` is None when nothing could
/// match and no commit was written
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DeleteOutcome {
    pub version: Option<u64>,
    /// Files the predicate fully covered, removed whole
    pub files_removed: usize,
    /// Rows in those removed files, exact
    pub rows_removed: u64,
    /// Every live row the predicate matched, across whole-file removals
    /// and partially covered files. A file the statistics fully cover
    /// contributes its row count with no IO, only a partially covered
    /// file is read, and only its predicate columns
    pub rows_matched: u64,
    /// True when a delete predicate was recorded for partially covered
    /// files, applied physically by a later optimize
    pub predicate_recorded: bool,
}

/// Outcome of an optimize pass. `version` is None when no file had a
/// predicate to apply
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct OptimizeOutcome {
    pub version: Option<u64>,
    pub files_removed: usize,
    pub files_written: usize,
    pub rows_written: u64,
    pub predicates_retired: usize,
}

/// Appends one row batch as a new data file
pub fn append_rows(
    log: &TransactionLog,
    attempt: CommitAttempt<'_>,
    table_id: u64,
    columns: &[ColumnData],
) -> Result<AppendOutcome, ZyronError> {
    let mut attempt = attempt;
    attempt.operation = OperationKind::Append;
    let mut staged: Option<(u64, PartitionEntry, Vec<usize>)> = None;
    // Every file this attempt put on disk, data and index alike. A retry
    // built against a different base has to unlink all of them, not just
    // the data file, or a losing attempt leaks its index deltas
    let mut staged_paths: Vec<PathBuf> = Vec::new();
    let result = log.commit(attempt, |base| {
        for path in staged_paths.drain(..) {
            let _ = fs::remove_file(path);
        }
        staged = None;
        let partition_id = allocate_partition_id(base);
        let sort_keys: Vec<u32> = base.cluster_spec.keys.iter().map(|k| k.column_id).collect();
        // The declared curve per key, so a file is laid out the way the
        // spec asked rather than always ascending
        let sort_strategies: Vec<ClusterStrategy> =
            base.cluster_spec.keys.iter().map(|k| k.strategy).collect();
        let written = write_data_file_ordered(
            log.paths(),
            &base.schema,
            &WriteRequest {
                partition_id,
                columns,
                sort_keys: &sort_keys,
                sort_strategies: &sort_strategies,
                cluster_spec_id: base.cluster_spec.spec_id,
                table_id,
                bloom_columns: &base.bloom_columns(),
                index_id: None,
            },
        )?;
        staged_paths.push(log.paths().data_file(partition_id));
        staged = Some((partition_id, written.entry.clone(), written.order.clone()));

        // Index deltas ride in the same commit as the rows they address,
        // so no version ever exists where the rows are visible and the
        // index does not name them
        let mut used: Vec<u64> = vec![partition_id];
        let mut entries = vec![LogEntry::AddFile(written.entry)];
        entries.extend(index::delta_entries_for_written_file(
            log.paths(),
            base,
            partition_id,
            table_id,
            columns,
            &written.order,
            &mut || {
                let id = allocate_unused_partition_id(base, &used);
                used.push(id);
                id
            },
        )?);
        for entry in &entries {
            if let LogEntry::AddIndexFile(file) = entry {
                staged_paths.push(
                    log.paths()
                        .index_file(file.index_id, file.file.partition_id),
                );
            }
        }
        Ok(entries)
    });
    match result {
        Ok(version) => {
            let (partition_id, entry, order) = staged.ok_or_else(|| {
                ZyronError::Internal("append committed without a staged file".into())
            })?;
            Ok(AppendOutcome {
                version,
                partition_id,
                rows: entry.row_count,
                order,
            })
        }
        Err(e) => {
            for path in staged_paths {
                let _ = fs::remove_file(path);
            }
            Err(e)
        }
    }
}

/// What a create index commit produced.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CreateIndexOutcome {
    pub version: u64,
    pub index_id: u32,
    /// Rows the backfill indexed
    pub rows: u64,
}

/// Declares an index and backfills it over every live data file in one
/// commit.
///
/// The declaration and its entries land together, so no version exists
/// where the index is declared but empty, which is the state a reader
/// would have to treat as "index says this table has no rows"
pub fn create_index(
    log: &TransactionLog,
    attempt: CommitAttempt<'_>,
    table_id: u64,
    name: &str,
    column_ids: &[u32],
    unique: bool,
) -> Result<CreateIndexOutcome, ZyronError> {
    let mut attempt = attempt;
    attempt.operation = OperationKind::SchemaChange;
    let mut staged: Vec<PathBuf> = Vec::new();
    let mut built_rows = 0u64;
    let mut built_index_id = 0u32;
    let result = log.commit(attempt, |base| {
        for path in staged.drain(..) {
            let _ = fs::remove_file(path);
        }
        if base.index_by_name(name).is_some() {
            return Err(ZyronError::Internal(format!(
                "index \"{}\" already exists on this table",
                name
            )));
        }
        // Ids ascend from the highest ever allocated so a dropped index
        // never has its id reused, which keeps a stale file from being
        // mistaken for a live one
        let index_id = base.indexes.last().map(|s| s.index_id + 1).unwrap_or(1);
        let spec = LakeIndexSpec {
            index_id,
            name: name.to_string(),
            column_ids: column_ids.to_vec(),
            unique,
        };
        spec.validate()?;
        for id in column_ids {
            if base.schema.column_by_id(*id).is_none() {
                return Err(ZyronError::Internal(format!(
                    "index \"{}\" names column {}, which is not in the schema",
                    name, id
                )));
            }
        }
        let mut used: Vec<u64> = Vec::new();
        let built = index::build_entries(log.paths(), base, &spec, table_id, &mut || {
            let id = allocate_unused_partition_id(base, &used);
            used.push(id);
            id
        })?;
        let mut rows = 0u64;
        for entry in &built {
            if let LogEntry::AddIndexFile(file) = entry {
                rows += file.file.row_count;
                staged.push(log.paths().index_file(index_id, file.file.partition_id));
            }
        }
        built_rows = rows;
        built_index_id = index_id;
        let mut entries = vec![LogEntry::AddIndex(spec)];
        entries.extend(built);
        Ok(entries)
    });
    match result {
        Ok(version) => Ok(CreateIndexOutcome {
            version,
            index_id: built_index_id,
            rows: built_rows,
        }),
        Err(e) => {
            for path in staged {
                let _ = fs::remove_file(path);
            }
            Err(e)
        }
    }
}

/// Drops an index by name. Its files stop being referenced immediately and
/// vacuum reclaims them once no retained version names them
pub fn drop_index(
    log: &TransactionLog,
    attempt: CommitAttempt<'_>,
    name: &str,
) -> Result<u64, ZyronError> {
    let mut attempt = attempt;
    attempt.operation = OperationKind::SchemaChange;
    log.commit(attempt, |base| {
        let spec = base.index_by_name(name).ok_or_else(|| {
            ZyronError::Internal(format!("index \"{}\" does not exist on this table", name))
        })?;
        Ok(vec![LogEntry::DropIndex {
            index_id: spec.index_id,
        }])
    })
}

/// Rebuilds every index over the live file set in one commit, replacing
/// whatever files they currently have.
///
/// This is also index compaction. Every write commit appends its own index
/// file, so a table taking many small writes accumulates runs whose key
/// ranges overlap, and overlapping ranges are exactly what stops the
/// manifest pruning a probe down to one file. Rebuilding re-sorts the whole
/// index into disjoint ranges again.
///
/// REINDEX runs it, and `compact_indexes_if_fragmented` runs it on its own
/// when the runs have grown past the point where pruning still works
pub fn rebuild_indexes(
    log: &TransactionLog,
    attempt: CommitAttempt<'_>,
    table_id: u64,
) -> Result<u64, ZyronError> {
    let mut attempt = attempt;
    attempt.operation = OperationKind::SchemaChange;
    let mut staged: Vec<PathBuf> = Vec::new();
    let result = log.commit(attempt, |base| {
        for path in staged.drain(..) {
            let _ = fs::remove_file(path);
        }
        let mut entries = Vec::new();
        let mut used: Vec<u64> = Vec::new();
        for file in &base.index_files {
            entries.push(LogEntry::RemoveIndexFile {
                index_id: file.index_id,
                partition_id: file.file.partition_id,
            });
        }
        for spec in &base.indexes {
            let built = index::build_entries(log.paths(), base, spec, table_id, &mut || {
                let id = allocate_unused_partition_id(base, &used);
                used.push(id);
                id
            })?;
            for entry in &built {
                if let LogEntry::AddIndexFile(file) = entry {
                    staged.push(
                        log.paths()
                            .index_file(spec.index_id, file.file.partition_id),
                    );
                }
            }
            entries.extend(built);
        }
        Ok(entries)
    });
    if result.is_err() {
        for path in staged {
            let _ = fs::remove_file(path);
        }
    }
    result
}

/// How many index files one index may hold per whole file of entries
/// before its ranges have overlapped enough to stop pruning.
///
/// Each write commit appends a file covering whatever keys that write
/// touched, so ranges overlap and a probe has to open every run whose range
/// admits the key. At this ratio a probe is already opening several files
/// where a compacted index would open one
const INDEX_FRAGMENTATION_LIMIT: usize = 4;

/// Rebuilds the indexes of a table whose runs have fragmented past the
/// point where the manifest can prune a probe to one file.
///
/// Returns the version it committed, or None when every index is still
/// compact enough to leave alone. Cheap to call: the decision reads the
/// manifest and opens nothing
pub fn compact_indexes_if_fragmented(
    log: &TransactionLog,
    attempt: CommitAttempt<'_>,
    table_id: u64,
) -> Result<Option<u64>, ZyronError> {
    let manifest = log.latest_manifest()?;
    if manifest.indexes.is_empty() {
        return Ok(None);
    }
    let fragmented = manifest.indexes.iter().any(|spec| {
        let files: Vec<&crate::index::IndexFileEntry> = manifest
            .index_files
            .iter()
            .filter(|f| f.index_id == spec.index_id)
            .collect();
        let entries: u64 = files.iter().map(|f| f.file.row_count).sum();
        // The count a compacted index of this size would have, so the test
        // is about fragmentation rather than about how big the table is
        let ideal = (entries as usize)
            .div_ceil(crate::index::ENTRIES_PER_INDEX_FILE)
            .max(1);
        files.len() > ideal * INDEX_FRAGMENTATION_LIMIT
    });
    if !fragmented {
        return Ok(None);
    }
    rebuild_indexes(log, attempt, table_id).map(Some)
}

/// Deletes by predicate. Files the predicate fully covers are removed
/// outright with no data IO, files it may match get the predicate
/// attached, files it cannot match are untouched
pub fn delete_where(
    log: &TransactionLog,
    attempt: CommitAttempt<'_>,
    predicate: &LakePredicate,
    sql: &str,
) -> Result<DeleteOutcome, ZyronError> {
    // Zero-effect deletes commit nothing. The check runs on the newest
    // published state, which is this statement's snapshot
    let snapshot = log.latest_manifest()?;
    let touches_any = snapshot
        .entries
        .iter()
        .any(|e| snapshot.prune_file(predicate, e) != PruneDecision::CannotMatch);
    if !touches_any {
        return Ok(DeleteOutcome {
            version: None,
            files_removed: 0,
            rows_removed: 0,
            rows_matched: 0,
            predicate_recorded: false,
        });
    }
    let mut attempt = attempt;
    attempt.operation = OperationKind::Delete;
    attempt.read_predicate = Some(predicate);
    let mut files_removed = 0usize;
    let mut rows_removed = 0u64;
    let mut rows_matched = 0u64;
    let mut predicate_recorded = false;
    let mut zero_effect = false;
    let result = log.commit(attempt, |base| {
        let mut entries = Vec::new();
        let mut partial = false;
        files_removed = 0;
        rows_removed = 0;
        rows_matched = 0;
        for file in &base.entries {
            match base.prune_file(predicate, file) {
                PruneDecision::FullyCovers => {
                    entries.push(LogEntry::RemoveFile {
                        partition_id: file.partition_id,
                    });
                    files_removed += 1;
                    rows_removed += file.row_count;
                    rows_matched += file.row_count;
                }
                PruneDecision::MayMatch => {
                    partial = true;
                    rows_matched += count_matching_rows(log, base, file, predicate)?;
                }
                PruneDecision::CannotMatch => {}
            }
        }
        if partial {
            let next_id = base.delete_predicates.last().map(|p| p.id + 1).unwrap_or(1);
            entries.push(LogEntry::AddDeletePredicate(DeletePredicate {
                id: next_id,
                sql: sql.to_string(),
                predicate: predicate.clone(),
                created_version: 0,
            }));
        }
        predicate_recorded = partial;
        // A delete matching no live row changes nothing. Committing an
        // entry for it would pollute the manifest with a predicate that
        // can never remove a row and would be re-evaluated by every
        // later read
        if entries.is_empty() || rows_matched == 0 {
            zero_effect = true;
            return Err(ZyronError::Internal("delete matched nothing".into()));
        }
        Ok(entries)
    });
    match result {
        Ok(version) => Ok(DeleteOutcome {
            version: Some(version),
            files_removed,
            rows_removed,
            rows_matched,
            predicate_recorded,
        }),
        Err(_) if zero_effect => Ok(DeleteOutcome {
            version: None,
            files_removed: 0,
            rows_removed: 0,
            rows_matched: 0,
            predicate_recorded: false,
        }),
        Err(e) => Err(e),
    }
}

/// Counts live rows of one partially covered file the predicate matches.
/// Rows already removed by an earlier predicate delete are excluded, so a
/// row is never counted twice across successive deletes. Only the columns
/// the predicates reference are decoded
fn count_matching_rows(
    log: &TransactionLog,
    base: &ManifestFile,
    file: &PartitionEntry,
    predicate: &LakePredicate,
) -> Result<u64, ZyronError> {
    let reader = LakeFileReader::open(log.paths(), file.partition_id)?;
    let row_count = reader.row_count();
    if row_count == 0 {
        return Ok(0);
    }
    let keep = reader.delete_survivors(&base.schema, base, file)?;
    let columns = reader.read_predicate_columns(&base.schema, &[predicate])?;
    let compiled = crate::reader::CompiledPredicate::new(predicate, &columns);
    let mut matched = 0u64;
    for row in 0..row_count {
        if keep[row / 8] & (1 << (row % 8)) == 0 {
            continue;
        }
        if compiled.evaluate(&columns, row) == Some(true) {
            matched += 1;
        }
    }
    Ok(matched)
}

/// Outcome of an update
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct UpdateOutcome {
    pub version: u64,
    pub rows_updated: u64,
    /// The data file the new images landed in, and the row order it was
    /// written in, so a caller can address the rows it just wrote to
    /// maintain a search index over them
    pub partition_id: u64,
    pub order: Vec<usize>,
    pub files_removed: usize,
    pub predicate_recorded: bool,
}

/// Replaces matching rows with new images in one commit. The old rows go
/// away exactly as a predicate delete does, whole files dropped where the
/// statistics cover them, a recorded predicate where they do not, and the
/// new images land in one new data file.
///
/// The entry order carries the correctness: the delete predicate is
/// recorded before the file is added, and a file added after a predicate
/// does not carry it. Reversing that would let the predicate delete the
/// very rows the update just wrote, which is what happens whenever an
/// update leaves rows still satisfying its own WHERE clause, as in
/// `SET n = n + 1 WHERE n < 10`.
///
/// `rows` is the caller's already-materialized new image, one cell per
/// row per column. `matched` is how many live rows it replaces, known
/// exactly by the caller because it read them
pub fn update_where(
    log: &TransactionLog,
    attempt: CommitAttempt<'_>,
    table_id: u64,
    predicate: Option<&LakePredicate>,
    sql: &str,
    rows: &[ColumnData],
    matched: u64,
) -> Result<UpdateOutcome, ZyronError> {
    let mut attempt = attempt;
    attempt.operation = OperationKind::Update;
    attempt.read_predicate = predicate;
    let mut staged: Vec<PathBuf> = Vec::new();
    let mut files_removed = 0usize;
    let mut predicate_recorded = false;
    // The file the new images landed in, so the caller can address them
    let mut written_rows: Option<(u64, Vec<usize>)> = None;
    let result = log.commit(attempt, |base| {
        for path in staged.drain(..) {
            let _ = fs::remove_file(path);
        }
        let mut entries = Vec::new();
        files_removed = 0;
        predicate_recorded = false;
        match predicate {
            Some(p) => {
                let mut partial = false;
                for file in &base.entries {
                    match base.prune_file(p, file) {
                        PruneDecision::FullyCovers => {
                            entries.push(LogEntry::RemoveFile {
                                partition_id: file.partition_id,
                            });
                            files_removed += 1;
                        }
                        PruneDecision::MayMatch => partial = true,
                        PruneDecision::CannotMatch => {}
                    }
                }
                if partial {
                    let next_id = base.delete_predicates.last().map(|d| d.id + 1).unwrap_or(1);
                    entries.push(LogEntry::AddDeletePredicate(DeletePredicate {
                        id: next_id,
                        sql: sql.to_string(),
                        predicate: p.clone(),
                        created_version: 0,
                    }));
                    predicate_recorded = true;
                }
            }
            None => {
                // Every row is replaced, so every file goes and no
                // predicate can ever apply again
                for file in &base.entries {
                    entries.push(LogEntry::RemoveFile {
                        partition_id: file.partition_id,
                    });
                    files_removed += 1;
                }
                for del in &base.delete_predicates {
                    entries.push(LogEntry::RemoveDeletePredicate { id: del.id });
                }
            }
        }
        // The new file is added last so it inherits no delete predicate
        let partition_id = allocate_partition_id(base);
        let sort_keys: Vec<u32> = base.cluster_spec.keys.iter().map(|k| k.column_id).collect();
        // The declared curve per key, so a file is laid out the way the
        // spec asked rather than always ascending
        let sort_strategies: Vec<ClusterStrategy> =
            base.cluster_spec.keys.iter().map(|k| k.strategy).collect();
        let written = write_data_file_ordered(
            log.paths(),
            &base.schema,
            &WriteRequest {
                partition_id,
                columns: rows,
                sort_keys: &sort_keys,
                sort_strategies: &sort_strategies,
                cluster_spec_id: base.cluster_spec.spec_id,
                table_id,
                bloom_columns: &base.bloom_columns(),
                index_id: None,
            },
        )?;
        staged.push(log.paths().data_file(partition_id));
        written_rows = Some((partition_id, written.order.clone()));
        entries.push(LogEntry::AddFile(written.entry));

        // The replaced rows leave their index entries behind, addressing
        // files this commit removed. A probe drops those, so what the
        // index still needs is entries for the new images
        let mut used: Vec<u64> = vec![partition_id];
        let index_entries = index::delta_entries_for_written_file(
            log.paths(),
            base,
            partition_id,
            table_id,
            rows,
            &written.order,
            &mut || {
                let id = allocate_unused_partition_id(base, &used);
                used.push(id);
                id
            },
        )?;
        for entry in &index_entries {
            if let LogEntry::AddIndexFile(file) = entry {
                staged.push(
                    log.paths()
                        .index_file(file.index_id, file.file.partition_id),
                );
            }
        }
        entries.extend(index_entries);
        Ok(entries)
    });
    match result {
        Ok(version) => {
            let (partition_id, order) = written_rows.ok_or_else(|| {
                ZyronError::Internal("update committed without a written file".into())
            })?;
            Ok(UpdateOutcome {
                version,
                rows_updated: matched,
                partition_id,
                order,
                files_removed,
                predicate_recorded,
            })
        }
        Err(e) => {
            for path in staged {
                let _ = fs::remove_file(path);
            }
            Err(e)
        }
    }
}

/// Deletes every row by removing every live file. No predicate is
/// recorded because nothing survives to filter, and no data is read
pub fn delete_all(
    log: &TransactionLog,
    attempt: CommitAttempt<'_>,
) -> Result<DeleteOutcome, ZyronError> {
    let snapshot = log.latest_manifest()?;
    if snapshot.entries.is_empty() {
        return Ok(DeleteOutcome {
            version: None,
            files_removed: 0,
            rows_removed: 0,
            rows_matched: 0,
            predicate_recorded: false,
        });
    }
    let mut attempt = attempt;
    attempt.operation = OperationKind::Delete;
    let mut files_removed = 0usize;
    let mut rows_removed = 0u64;
    let mut rows_matched = 0u64;
    let mut zero_effect = false;
    let result = log.commit(attempt, |base| {
        files_removed = base.entries.len();
        rows_removed = base.entries.iter().map(|e| e.row_count).sum();
        if base.entries.is_empty() {
            zero_effect = true;
            return Err(ZyronError::Internal("table is already empty".into()));
        }
        // Rows an earlier predicate delete already removed are not
        // deleted again. A file carrying no predicate contributes its
        // whole row count with no IO
        rows_matched = 0;
        for file in &base.entries {
            if file.delete_predicate_ids.is_empty() {
                rows_matched += file.row_count;
                continue;
            }
            let reader = LakeFileReader::open(log.paths(), file.partition_id)?;
            let keep = reader.delete_survivors(&base.schema, base, file)?;
            rows_matched += keep.iter().map(|b| b.count_ones() as u64).sum::<u64>();
        }
        let mut entries: Vec<LogEntry> = base
            .entries
            .iter()
            .map(|e| LogEntry::RemoveFile {
                partition_id: e.partition_id,
            })
            .collect();
        // Every predicate loses the last file it could attach to, so none
        // of them can ever remove a row again
        for del in &base.delete_predicates {
            entries.push(LogEntry::RemoveDeletePredicate { id: del.id });
        }
        Ok(entries)
    });
    match result {
        Ok(version) => Ok(DeleteOutcome {
            version: Some(version),
            files_removed,
            rows_removed,
            rows_matched,
            predicate_recorded: false,
        }),
        Err(_) if zero_effect => Ok(DeleteOutcome {
            version: None,
            files_removed: 0,
            rows_removed: 0,
            rows_matched: 0,
            predicate_recorded: false,
        }),
        Err(e) => Err(e),
    }
}

/// Applies attached delete predicates physically. Every file carrying a
/// predicate is rewritten through its survivor mask, inputs are removed,
/// survivors coalesce into one new file, and predicates that no longer
/// attach to any live file are retired
pub fn optimize(
    log: &TransactionLog,
    attempt: CommitAttempt<'_>,
    table_id: u64,
) -> Result<OptimizeOutcome, ZyronError> {
    let snapshot = log.latest_manifest()?;
    if snapshot
        .entries
        .iter()
        .all(|e| e.delete_predicate_ids.is_empty())
    {
        return Ok(OptimizeOutcome {
            version: None,
            files_removed: 0,
            files_written: 0,
            rows_written: 0,
            predicates_retired: 0,
        });
    }
    let mut attempt = attempt;
    attempt.operation = OperationKind::Optimize;
    let mut staged: Vec<PathBuf> = Vec::new();
    let mut files_removed = 0usize;
    let mut files_written = 0usize;
    let mut rows_written = 0u64;
    let mut predicates_retired = 0usize;
    let mut zero_effect = false;
    let result = log.commit(attempt, |base| {
        for path in staged.drain(..) {
            let _ = fs::remove_file(path);
        }
        let inputs: Vec<&PartitionEntry> = base
            .entries
            .iter()
            .filter(|e| !e.delete_predicate_ids.is_empty())
            .collect();
        files_removed = inputs.len();
        files_written = 0;
        rows_written = 0;
        if inputs.is_empty() {
            zero_effect = true;
            return Err(ZyronError::Internal("nothing to optimize".into()));
        }
        // Survivor rows from every input, in schema column order
        let mut batch: Vec<ColumnData> = base
            .schema
            .columns
            .iter()
            .map(|c| ColumnData {
                column_id: c.id,
                cells: Vec::new(),
            })
            .collect();
        for input in &inputs {
            let reader = LakeFileReader::open(log.paths(), input.partition_id)?;
            let keep = reader.delete_survivors(&base.schema, base, input)?;
            let decoded: Vec<_> = base
                .schema
                .columns
                .iter()
                .map(|c| reader.read_column(c))
                .collect::<Result<_, _>>()?;
            for row in 0..reader.row_count() {
                if keep[row / 8] & (1 << (row % 8)) == 0 {
                    continue;
                }
                for (slot, col) in batch.iter_mut().zip(decoded.iter()) {
                    slot.cells.push(col.cell(row).map(|c| c.to_vec()));
                }
            }
        }
        let mut entries: Vec<LogEntry> = inputs
            .iter()
            .map(|f| LogEntry::RemoveFile {
                partition_id: f.partition_id,
            })
            .collect();
        let survivor_rows = batch.first().map(|c| c.cells.len()).unwrap_or(0);
        if survivor_rows > 0 {
            let partition_id = allocate_partition_id(base);
            let sort_keys: Vec<u32> = base.cluster_spec.keys.iter().map(|k| k.column_id).collect();
            let sort_strategies: Vec<ClusterStrategy> =
                base.cluster_spec.keys.iter().map(|k| k.strategy).collect();
            let written = write_data_file_ordered(
                log.paths(),
                &base.schema,
                &WriteRequest {
                    partition_id,
                    columns: &batch,
                    sort_keys: &sort_keys,
                    sort_strategies: &sort_strategies,
                    cluster_spec_id: base.cluster_spec.spec_id,
                    table_id,
                    bloom_columns: &base.bloom_columns(),
                    index_id: None,
                },
            )?;
            staged.push(log.paths().data_file(partition_id));
            files_written = 1;
            rows_written = written.entry.row_count;
            entries.push(LogEntry::AddFile(written.entry));

            // The rewrite moved rows, so every entry addressing an input
            // is now stale. Index files covering an input are dropped and
            // the partitions they also covered are re-indexed, which is
            // what keeps coverage complete across the rewrite
            let removed: Vec<u64> = inputs.iter().map(|f| f.partition_id).collect();
            let (drops, orphaned) = index::stale_index_files(base, &removed);
            entries.extend(drops);
            let mut used: Vec<u64> = vec![partition_id];
            for spec in &base.indexes {
                let mut index_batch = index::IndexBatch::new(spec);
                index::entries_for_written_file(
                    &base.schema,
                    spec,
                    partition_id,
                    &batch,
                    &written.order,
                    &mut index_batch,
                )?;
                for orphan in &orphaned {
                    let Some(entry) = base.entry_for(*orphan) else {
                        continue;
                    };
                    index::entries_for_file(log.paths(), base, spec, entry, &mut index_batch)?;
                }
                for file in index::write_index_files(
                    log.paths(),
                    &base.schema,
                    spec,
                    table_id,
                    index_batch,
                    &mut || {
                        let id = allocate_unused_partition_id(base, &used);
                        used.push(id);
                        id
                    },
                )? {
                    staged.push(
                        log.paths()
                            .index_file(spec.index_id, file.file.partition_id),
                    );
                    entries.push(LogEntry::AddIndexFile(file));
                }
            }
        }
        // A predicate attached only to inputs no longer attaches anywhere,
        // the rewrite applied it for good
        let input_ids: BTreeSet<u64> = inputs.iter().map(|f| f.partition_id).collect();
        let mut retire: BTreeSet<u64> = inputs
            .iter()
            .flat_map(|f| f.delete_predicate_ids.iter().copied())
            .collect();
        for file in &base.entries {
            if !input_ids.contains(&file.partition_id) {
                for id in &file.delete_predicate_ids {
                    retire.remove(id);
                }
            }
        }
        predicates_retired = retire.len();
        for id in retire {
            entries.push(LogEntry::RemoveDeletePredicate { id });
        }
        Ok(entries)
    });
    match result {
        Ok(version) => Ok(OptimizeOutcome {
            version: Some(version),
            files_removed,
            files_written,
            rows_written,
            predicates_retired,
        }),
        Err(_) if zero_effect => Ok(OptimizeOutcome {
            version: None,
            files_removed: 0,
            files_written: 0,
            rows_written: 0,
            predicates_retired: 0,
        }),
        Err(e) => {
            for path in staged {
                let _ = fs::remove_file(path);
            }
            Err(e)
        }
    }
}

/// Deletes data files no reachable version references anymore, orphans
/// from failed commits and files removed before the retention floor.
/// Returns how many files were deleted
pub fn vacuum_data_files(
    log: &TransactionLog,
    retain_min_version: u64,
) -> Result<usize, ZyronError> {
    let head = log.head_version();
    let floor = retain_min_version.clamp(1, head);
    // Every partition referenced by any version the log can still replay
    let mut referenced: BTreeSet<u64> = BTreeSet::new();
    // Index files are reclaimed on the same rule as data files, keyed by
    // the index they belong to. A dropped index leaves its files behind
    // and this is what collects them
    let mut referenced_index: BTreeSet<(u32, u64)> = BTreeSet::new();
    for version in floor..=head {
        match log.manifest_at(version) {
            Ok(manifest) => {
                for entry in &manifest.entries {
                    referenced.insert(entry.partition_id);
                }
                for file in &manifest.index_files {
                    referenced_index.insert((file.index_id, file.file.partition_id));
                }
            }
            // Versions below the surviving checkpoint chain are gone,
            // their files are exactly what vacuum reclaims
            Err(_) => continue,
        }
    }
    let mut removed = 0usize;
    for dirent in fs::read_dir(log.paths().data_dir())? {
        let dirent = dirent?;
        let name = dirent.file_name();
        let Some(name) = name.to_str() else { continue };
        // Staging leftovers from crashed writers are always garbage
        if name.ends_with(".zyr.tmp") {
            fs::remove_file(dirent.path())?;
            removed += 1;
            continue;
        }
        if let Some(key) = parse_index_file_name(name) {
            if !referenced_index.contains(&key) {
                fs::remove_file(dirent.path())?;
                removed += 1;
            }
            continue;
        }
        let Some(partition_id) = parse_data_file_name(name) else {
            continue;
        };
        if !referenced.contains(&partition_id) {
            fs::remove_file(dirent.path())?;
            removed += 1;
        }
    }
    Ok(removed)
}

/// Random 64-bit file identity, collision checked against the base
/// manifest. Entropy rather than a counter so concurrent writers can
/// never stage the same path, losing a version race never touches
/// another writer's bytes
/// A partition id unused by the base manifest and by ids this commit has
/// already handed out. One commit writes a data file and one index file
/// per index, and every one of them needs a distinct name
pub(crate) fn allocate_unused_partition_id(base: &ManifestFile, used: &[u64]) -> u64 {
    loop {
        let candidate = allocate_partition_id(base);
        if !used.contains(&candidate)
            && !base
                .index_files
                .iter()
                .any(|f| f.file.partition_id == candidate)
        {
            return candidate;
        }
    }
}

pub(crate) fn allocate_partition_id(base: &ManifestFile) -> u64 {
    loop {
        let mut h = RandomState::new().build_hasher();
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_nanos() as u64)
            .unwrap_or(0);
        h.write_u64(nanos);
        let candidate = h.finish();
        if candidate != 0 && base.entry_for(candidate).is_none() {
            return candidate;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::paths::LakePaths;
    use crate::predicate::{CompareOp, LakeValue};
    use crate::schema::LakeColumn;
    use crate::schema::LakeSchema;
    use std::collections::BTreeMap;
    use std::sync::Arc;
    use zyron_common::TypeId;

    fn schema() -> LakeSchema {
        LakeSchema::new(
            1,
            vec![
                LakeColumn {
                    id: 0,
                    name: "id".into(),
                    type_id: TypeId::Int64,
                    nullable: false,
                    fractional_digits: None,
                    tz_offset_secs: None,
                    max_length: None,
                    default_expr: None,
                },
                LakeColumn {
                    id: 1,
                    name: "name".into(),
                    type_id: TypeId::Varchar,
                    nullable: true,
                    fractional_digits: None,
                    tz_offset_secs: None,
                    max_length: None,
                    default_expr: None,
                },
            ],
        )
        .expect("schema")
    }

    fn attempt() -> CommitAttempt<'static> {
        CommitAttempt {
            operation: OperationKind::Append,
            db_txn_id: 0,
            commit_lsn: 1,
            timestamp_us: 1_754_700_000_000_000,
            read_predicate: None,
            read_version: 0,
            audit: None,
        }
    }

    fn new_log(dir: &std::path::Path) -> TransactionLog {
        let mut create = attempt();
        create.operation = OperationKind::SchemaChange;
        TransactionLog::create(
            LakePaths::new(dir, 7),
            create,
            &schema(),
            None,
            &BTreeMap::new(),
        )
        .expect("create")
    }

    fn batch(ids: &[i64], names: &[Option<&str>]) -> Vec<ColumnData> {
        vec![
            ColumnData {
                column_id: 0,
                cells: ids.iter().map(|v| Some(v.to_le_bytes().to_vec())).collect(),
            },
            ColumnData {
                column_id: 1,
                cells: names
                    .iter()
                    .map(|v| v.map(|s| s.as_bytes().to_vec()))
                    .collect(),
            },
        ]
    }

    fn id_below(limit: i64) -> LakePredicate {
        LakePredicate::Compare {
            column_id: 0,
            op: CompareOp::Lt,
            value: LakeValue::Int(limit),
        }
    }

    #[test]
    fn test_append_and_concurrent_appends_never_collide() {
        let dir = tempfile::tempdir().expect("tempdir");
        let log = Arc::new(new_log(dir.path()));
        let out = append_rows(&log, attempt(), 7, &batch(&[1, 2], &[Some("a"), Some("b")]))
            .expect("append");
        assert_eq!(out.rows, 2);

        let mut handles = Vec::new();
        for t in 0..4 {
            let log = Arc::clone(&log);
            handles.push(std::thread::spawn(move || {
                for i in 0..3i64 {
                    append_rows(&log, attempt(), 7, &batch(&[t * 100 + i], &[Some("x")]))
                        .expect("concurrent append");
                }
            }));
        }
        for h in handles {
            h.join().expect("thread");
        }
        let m = log.latest_manifest().expect("manifest");
        assert_eq!(m.entries.len(), 13);
        // Every entry's file exists and ids are unique
        for entry in &m.entries {
            assert!(log.paths().data_file(entry.partition_id).exists());
        }
        let total_rows: u64 = m.entries.iter().map(|e| e.row_count).sum();
        assert_eq!(total_rows, 14);
    }

    #[test]
    fn test_delete_fully_covering_drops_files_without_io() {
        let dir = tempfile::tempdir().expect("tempdir");
        let log = new_log(dir.path());
        append_rows(&log, attempt(), 7, &batch(&[1, 2], &[Some("a"), Some("b")]))
            .expect("low file");
        append_rows(
            &log,
            attempt(),
            7,
            &batch(&[100, 200], &[Some("c"), Some("d")]),
        )
        .expect("high file");

        let out = delete_where(&log, attempt(), &id_below(50), "id < 50").expect("delete");
        assert_eq!(out.files_removed, 1);
        assert_eq!(out.rows_removed, 2);
        assert_eq!(out.rows_matched, 2);
        assert!(!out.predicate_recorded);
        let m = log.latest_manifest().expect("manifest");
        assert_eq!(m.entries.len(), 1);
        assert!(m.delete_predicates.is_empty());

        // A predicate matching nothing commits nothing
        let noop = delete_where(&log, attempt(), &id_below(-100), "id < -100").expect("noop");
        assert_eq!(noop.version, None);
    }

    #[test]
    fn test_partial_delete_records_predicate_and_optimize_applies_it() {
        let dir = tempfile::tempdir().expect("tempdir");
        let log = new_log(dir.path());
        append_rows(
            &log,
            attempt(),
            7,
            &batch(&[10, 60, 90], &[Some("a"), Some("b"), Some("c")]),
        )
        .expect("append");

        let out = delete_where(&log, attempt(), &id_below(50), "id < 50").expect("delete");
        assert!(out.predicate_recorded);
        assert_eq!(out.files_removed, 0);
        // Only the one row below the cut matched, counted from the file
        assert_eq!(out.rows_matched, 1);
        // A second delete over the same range matches nothing new, the
        // first delete already removed that row, so it commits nothing
        // and records no second predicate
        let again = delete_where(&log, attempt(), &id_below(50), "id < 50").expect("delete");
        assert_eq!(again.rows_matched, 0);
        assert_eq!(again.version, None);
        let m = log.latest_manifest().expect("manifest");
        assert_eq!(m.delete_predicates.len(), 1);
        assert_eq!(m.entries[0].delete_predicate_ids.len(), 1);

        // The reader already filters the deleted row before any rewrite
        let reader = LakeFileReader::open(log.paths(), m.entries[0].partition_id).expect("open");
        let keep = reader
            .delete_survivors(&m.schema, &m, &m.entries[0])
            .expect("survivors");
        assert_eq!(keep[0].count_ones(), 2);

        let opt = optimize(&log, attempt(), 7).expect("optimize");
        assert_eq!(opt.files_removed, 1);
        assert_eq!(opt.files_written, 1);
        assert_eq!(opt.rows_written, 2);
        assert_eq!(opt.predicates_retired, 1);
        let m2 = log.latest_manifest().expect("manifest");
        assert!(m2.delete_predicates.is_empty());
        assert_eq!(m2.entries.len(), 1);
        assert_eq!(m2.entries[0].row_count, 2);
        assert!(m2.entries[0].delete_predicate_ids.is_empty());

        // A second optimize has nothing to do
        let idle = optimize(&log, attempt(), 7).expect("idle");
        assert_eq!(idle.version, None);
    }

    #[test]
    fn test_update_does_not_delete_the_rows_it_just_wrote() {
        let dir = tempfile::tempdir().expect("tempdir");
        let log = new_log(dir.path());
        append_rows(
            &log,
            attempt(),
            7,
            &batch(&[1, 2, 60], &[Some("a"), Some("b"), Some("c")]),
        )
        .expect("append");

        // The classic trap: the new images still satisfy the predicate the
        // update matched on. Recording the delete before adding the file
        // is what keeps them alive
        let predicate = id_below(50);
        let new_rows = batch(&[2, 3], &[Some("a"), Some("b")]);
        let out = update_where(
            &log,
            attempt(),
            7,
            Some(&predicate),
            "id < 50",
            &new_rows,
            2,
        )
        .expect("update");
        assert_eq!(out.rows_updated, 2);

        let m = log.latest_manifest().expect("manifest");
        let mut live = 0usize;
        for entry in &m.entries {
            let reader = LakeFileReader::open(log.paths(), entry.partition_id).expect("open");
            let keep = reader
                .delete_survivors(&m.schema, &m, entry)
                .expect("survivors");
            live += keep.iter().map(|b| b.count_ones() as usize).sum::<usize>();
        }
        // Two updated rows plus the untouched row above the cut
        assert_eq!(live, 3, "the updated rows must survive their own predicate");
    }

    #[test]
    fn test_update_without_predicate_replaces_everything() {
        let dir = tempfile::tempdir().expect("tempdir");
        let log = new_log(dir.path());
        append_rows(&log, attempt(), 7, &batch(&[1, 2], &[Some("a"), Some("b")])).expect("append");
        append_rows(&log, attempt(), 7, &batch(&[3], &[Some("c")])).expect("append");

        let new_rows = batch(&[9, 9, 9], &[Some("z"), Some("z"), Some("z")]);
        let out = update_where(&log, attempt(), 7, None, "TRUE", &new_rows, 3).expect("update");
        assert_eq!(out.files_removed, 2);
        assert!(!out.predicate_recorded);
        let m = log.latest_manifest().expect("manifest");
        assert_eq!(m.entries.len(), 1);
        assert_eq!(m.entries[0].row_count, 3);
        assert!(m.delete_predicates.is_empty());
    }

    #[test]
    fn test_vacuum_reclaims_orphans_and_respects_time_travel() {
        let dir = tempfile::tempdir().expect("tempdir");
        let log = new_log(dir.path());
        append_rows(&log, attempt(), 7, &batch(&[1], &[Some("a")])).expect("append");
        let m1 = log.latest_manifest().expect("manifest");
        let old_pid = m1.entries[0].partition_id;

        // Fully-covering delete removes the file from the current version
        delete_where(&log, attempt(), &id_below(100), "id < 100").expect("delete");

        // An orphan from a failed writer and a staging leftover
        std::fs::write(log.paths().data_file(0xDEAD), b"orphan").expect("orphan");
        std::fs::write(
            log.paths().data_dir().join("p-0000000000000bad.zyr.tmp"),
            b"tmp",
        )
        .expect("tmp");

        // Retaining from version 1 keeps the removed file for time travel
        let removed = vacuum_data_files(&log, 1).expect("vacuum");
        assert_eq!(removed, 2);
        assert!(log.paths().data_file(old_pid).exists());

        // Retaining only the head reclaims it
        let removed = vacuum_data_files(&log, log.latest_version()).expect("vacuum head");
        assert_eq!(removed, 1);
        assert!(!log.paths().data_file(old_pid).exists());
    }
}
