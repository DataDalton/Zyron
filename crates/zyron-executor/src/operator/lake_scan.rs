//! Lake table scan operator.
//!
//! Reads a lake table's data files as named by its transaction log
//! manifest at the resolved version. Visibility is by log version, there
//! are no MVCC system columns and no patch overlay, a file is either in
//! the manifest or it is not, and rows removed by predicate deletes are
//! filtered through the manifest's delete predicates at read time.
//!
//! A column the file predates reads as all NULL, so a schema-evolved
//! table scans every file without rewrites. The query predicate is
//! applied after decode exactly like the heap and columnar scans, then
//! column-level security, so a lake read of the same rows returns
//! identical results.

use std::collections::VecDeque;
use std::sync::Arc;

use zyron_common::{Result, ZyronError};
use zyron_lake::{
    AllCommitted, LakeFileReader, LakePaths, ManifestFile, PruneDecision, TimeTravelSpec,
    TransactionLog, resolve_version,
};
use zyron_planner::binder::BoundExpr;
use zyron_planner::logical::{AsOfTarget, LogicalColumn};

use crate::batch::{
    BATCH_SIZE, DataBatch, create_builders, decode_fixed_scalar, decode_varlen_scalar,
    finalize_builders,
};
use crate::column::ScalarValue;
use crate::compute::column_to_mask;
use crate::context::ExecutionContext;
use crate::expr::evaluate;
use crate::operator::{ExecutionBatch, Operator, OperatorResult, apply_column_security};

/// Which head of a lake table a statement addresses.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum LakeHead<'a> {
    Main,
    /// Named by the query, so the table has to have it
    Named(&'a str),
    /// The session's branch. A table this branch never forked reads and
    /// writes through to main, matching the heap, where a branch overlays
    /// the pages it copied and reads the rest from the main line
    Session(&'a str),
}

/// The head a lake statement addresses.
///
/// A query qualifier names its own branch and wins. A version or timestamp
/// qualifier reads main's history, because that is the axis it names.
/// Otherwise the session's branch applies, so `USE BRANCH work` isolates a
/// lake table's reads and writes the way it isolates a heap table's.
#[inline]
pub(crate) fn effective_head<'a>(
    ctx: &'a ExecutionContext,
    as_of: Option<&'a AsOfTarget>,
) -> LakeHead<'a> {
    match as_of {
        Some(AsOfTarget::Branch(name)) => LakeHead::Named(name.as_str()),
        Some(_) => LakeHead::Main,
        None => match ctx.active_branch_name.as_deref() {
            Some(name) => LakeHead::Session(name),
            None => LakeHead::Main,
        },
    }
}

fn branch_error(table_name: &str, branch: &str, e: ZyronError) -> ZyronError {
    ZyronError::ExecutionError(format!(
        "branch \"{}\" on lake table \"{}\": {}",
        branch, table_name, e
    ))
}

/// Opens the head a lake statement reads.
pub(crate) fn open_lake_head(
    paths: &LakePaths,
    table_name: &str,
    head: LakeHead<'_>,
) -> Result<Arc<TransactionLog>> {
    match head {
        LakeHead::Main => TransactionLog::open_shared(paths.clone(), &AllCommitted),
        LakeHead::Named(name) => zyron_lake::open_branch_shared(paths, name)
            .map_err(|e| branch_error(table_name, name, e)),
        // A branch that never forked this table has nothing of its own to
        // show, so the main line is what it sees
        LakeHead::Session(name) => match zyron_lake::open_branch_shared(paths, name) {
            Ok(log) => Ok(log),
            Err(ZyronError::BranchNotFound(_)) => {
                TransactionLog::open_shared(paths.clone(), &AllCommitted)
            }
            Err(e) => Err(branch_error(table_name, name, e)),
        },
    }
}

/// Opens the head a lake statement writes, forking the table onto the
/// session's branch when the branch has not touched it yet.
///
/// Creating that head writes one marker file and copies no data, and it
/// forks at the table's current version, which is what the heap gives for
/// the same case: pages the branch never copied read through to main, so a
/// table the branch has not written carries main's rows into it.
pub(crate) fn open_lake_write_head(
    paths: &LakePaths,
    table_name: &str,
    head: LakeHead<'_>,
) -> Result<Arc<TransactionLog>> {
    let name = match head {
        LakeHead::Main => return TransactionLog::open_shared(paths.clone(), &AllCommitted),
        LakeHead::Named(name) | LakeHead::Session(name) => name,
    };
    match zyron_lake::open_branch_shared(paths, name) {
        Ok(log) => Ok(log),
        Err(ZyronError::BranchNotFound(_)) if matches!(head, LakeHead::Session(_)) => {
            let main = TransactionLog::open_shared(paths.clone(), &AllCommitted)?;
            let created_us = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_micros() as i64)
                .unwrap_or(0);
            match zyron_lake::create_branch(&main, name, None, created_us) {
                // A concurrent statement on the same session branch forking
                // the same table is the same outcome
                Ok(_) | Err(ZyronError::BranchAlreadyExists(_)) => {}
                Err(e) => return Err(branch_error(table_name, name, e)),
            }
            zyron_lake::open_branch_shared(paths, name)
                .map_err(|e| branch_error(table_name, name, e))
        }
        Err(e) => Err(branch_error(table_name, name, e)),
    }
}

/// The ordinal range a keep mask still admits, as the first and one past
/// the last surviving row.
///
/// Decoding is per column and the range decides how much of one is
/// materialized, so this is what turns a mask holding one row into a decode
/// of one row rather than of the file. An empty mask reports an empty range
fn surviving_span(keep: &[u8], row_count: usize) -> (usize, usize) {
    let first = keep.iter().position(|b| *b != 0);
    let Some(first_byte) = first else {
        return (0, 0);
    };
    let last_byte = keep.iter().rposition(|b| *b != 0).unwrap_or(first_byte);
    let start = first_byte * 8 + keep[first_byte].trailing_zeros() as usize;
    let end = (last_byte * 8 + (8 - keep[last_byte].leading_zeros() as usize)).min(row_count);
    (start.min(row_count), end.max(start.min(row_count)))
}

/// Equality terms a lowered predicate asserts at its top level, as column
/// id to value.
///
/// Only conjuncts count. A term under an OR does not have to hold, so
/// using it to address rows would drop the rows the other arm matches
fn equality_terms(
    predicate: &zyron_lake::LakePredicate,
    out: &mut Vec<(u32, zyron_lake::LakeValue)>,
) {
    match predicate {
        zyron_lake::LakePredicate::Compare {
            column_id,
            op: zyron_lake::CompareOp::Eq,
            value,
        } => out.push((*column_id, value.clone())),
        zyron_lake::LakePredicate::And(children) => {
            for child in children {
                equality_terms(child, out);
            }
        }
        _ => {}
    }
}

/// Whether reading an index costs less than answering the predicate from
/// the files pruning already left.
///
/// Both sides are bytes the manifest already knows, so the decision reads
/// no file. What each side counts is what its plan reads and the other's
/// does not:
///
/// - the scan opens every surviving data file and reads the leading key
///   column's whole segment out of each
/// - the probe opens every index file its key bounds admit and reads it
///
/// Projected columns are on neither side because both plans read them.
/// Two things the probe saves go uncounted: the trailing columns of a
/// composite key, which the scan also has to read and the probe does not,
/// and the surviving files the probe does not address, which are not known
/// until it runs. So the comparison understates the index, and it errs
/// toward the scan, which is the safe direction: declining an index that
/// would have helped costs a speedup, while taking one that does not help
/// costs the query.
///
/// The comparison is what makes an index optional rather than mandatory.
/// On a column the files are already ordered by, pruning reaches one file
/// and the index is pure overhead. On a column they are not, pruning
/// reaches every file and the index is the only thing that can help
fn index_is_worth_probing(
    manifest: &ManifestFile,
    spec: &zyron_lake::LakeIndexSpec,
    surviving: &[u64],
    probe_bytes: u64,
) -> bool {
    // A manifest that does not carry per-column sizes cannot support this
    // comparison, and guessing one would decide an access path on a number
    // nobody measured
    let Some(scan_bytes) = zyron_lake::scan_read_bytes(manifest, surviving, spec.column_ids[0])
    else {
        return false;
    };
    probe_bytes < scan_bytes
}

/// Range terms a lowered predicate asserts at its top level, as column id
/// to a low and high bound. Only conjuncts count, for the same reason
/// equality terms do
fn range_terms(
    predicate: &zyron_lake::LakePredicate,
    out: &mut Vec<(
        u32,
        Option<zyron_lake::RangeBound>,
        Option<zyron_lake::RangeBound>,
    )>,
) {
    use zyron_lake::CompareOp;
    match predicate {
        zyron_lake::LakePredicate::Compare {
            column_id,
            op,
            value,
        } => {
            let bound = zyron_lake::RangeBound {
                value: value.clone(),
                inclusive: matches!(op, CompareOp::GtEq | CompareOp::LtEq),
            };
            let (low, high) = match op {
                CompareOp::Gt | CompareOp::GtEq => (Some(bound), None),
                CompareOp::Lt | CompareOp::LtEq => (None, Some(bound)),
                _ => return,
            };
            // Two bounds on one column narrow each other rather than
            // producing two entries the caller would have to reconcile
            if let Some(slot) = out.iter_mut().find(|(id, _, _)| id == column_id) {
                if low.is_some() {
                    slot.1 = low;
                }
                if high.is_some() {
                    slot.2 = high;
                }
            } else {
                out.push((*column_id, low, high));
            }
        }
        zyron_lake::LakePredicate::And(children) => {
            for child in children {
                range_terms(child, out);
            }
        }
        _ => {}
    }
}

/// What a secondary index resolved for one scan
struct IndexResolution {
    /// Rows the probe addressed, keyed by data file and ascending within it
    rows: std::collections::BTreeMap<u64, Vec<u64>>,
    /// The index that produced them
    name: String,
    /// Index files the probe opened, so a plan can show the probe's own
    /// cost rather than only its effect on the data files
    files_read: usize,
}

/// Resolves a range predicate to row addresses through an index that leads
/// with the bounded column.
fn resolve_range_through_index(
    paths: &LakePaths,
    manifest: &ManifestFile,
    lowered: &zyron_lake::LakePredicate,
    surviving: &[u64],
) -> Result<Option<IndexResolution>> {
    let mut terms = Vec::new();
    range_terms(lowered, &mut terms);
    if terms.is_empty() {
        return Ok(None);
    }
    for spec in &manifest.indexes {
        let leading = spec.column_ids[0];
        let Some((_, low, high)) = terms.iter().find(|(id, _, _)| *id == leading) else {
            continue;
        };
        if low.is_none() && high.is_none() {
            continue;
        }
        if !zyron_lake::covers_table(manifest, spec.index_id) {
            continue;
        }
        let probe_bytes =
            zyron_lake::range_probe_read_bytes(manifest, spec, low.as_ref(), high.as_ref())?;
        if !index_is_worth_probing(manifest, spec, surviving, probe_bytes) {
            continue;
        }
        let (addresses, stats) =
            zyron_lake::probe_range(paths, manifest, spec, low.as_ref(), high.as_ref())?;
        return Ok(Some(IndexResolution {
            rows: zyron_lake::group_by_partition(&addresses),
            name: spec.name.clone(),
            files_read: stats.files_opened,
        }));
    }
    Ok(None)
}

/// Resolves a predicate to row addresses through a secondary index, when
/// one leads with a column the predicate pins to a value.
///
/// Returns None when no index applies, when the best candidate does not
/// cover every live data file, or when the key's value has no stored form.
/// Every one of those falls back to the scan, so an index only ever
/// removes work
fn resolve_through_index(
    paths: &LakePaths,
    manifest: &ManifestFile,
    lowered: &zyron_lake::LakePredicate,
    // Files pruning left, which is what the index has to beat
    surviving: &[u64],
) -> Result<Option<IndexResolution>> {
    if manifest.indexes.is_empty() {
        return Ok(None);
    }
    let mut terms = Vec::new();
    equality_terms(lowered, &mut terms);
    if terms.is_empty() {
        return Ok(None);
    }

    // The index whose key the predicate pins furthest, so a composite
    // index is preferred over a single column one covering the same term
    let mut best: Option<(&zyron_lake::LakeIndexSpec, Vec<zyron_lake::LakeValue>)> = None;
    for spec in &manifest.indexes {
        let mut key = Vec::with_capacity(spec.column_ids.len());
        for column_id in &spec.column_ids {
            let Some((_, value)) = terms.iter().find(|(id, _)| id == column_id) else {
                break;
            };
            key.push(value.clone());
        }
        // A partially pinned key cannot address rows: the index is sorted
        // on the whole key, so the leading run it selects is not the set
        // the predicate names
        if key.len() != spec.column_ids.len() {
            continue;
        }
        if !zyron_lake::covers_table(manifest, spec.index_id) {
            continue;
        }
        if best
            .as_ref()
            .map(|(current, _)| spec.column_ids.len() > current.column_ids.len())
            .unwrap_or(true)
        {
            best = Some((spec, key));
        }
    }
    if let Some((spec, key)) = &best {
        let probe_bytes = zyron_lake::point_probe_read_bytes(manifest, spec, key.first())?;
        if !index_is_worth_probing(manifest, spec, surviving, probe_bytes) {
            // Pruning already reduced the file set far enough that the
            // index would cost more to read than it saves. Declining is
            // the whole point of measuring: an index is a way to read
            // less, so using one that reads more is the wrong call however
            // available it is
            return Ok(None);
        }
    }
    let Some((spec, key)) = best else {
        // No index has its whole key pinned. A range on an index's leading
        // column is still answerable, and it is the case pruning helps
        // least with: bounds on a column the files are not ordered by
        // reject nothing, so without this every file is read
        return resolve_range_through_index(paths, manifest, lowered, surviving);
    };

    // The key's stored bytes, under the index schema's own column types
    let schema = zyron_lake::index_schema(&manifest.schema, spec)?;
    let mut cells: Vec<Option<Vec<u8>>> = Vec::with_capacity(key.len());
    for (position, value) in key.iter().enumerate() {
        let column = &schema.columns[position];
        let width = column.physical_type_id().fixed_size().unwrap_or(0);
        match zyron_lake::value_to_index_cell(column.physical_type_id(), width, value) {
            Some(cell) => cells.push(Some(cell)),
            // A constant with no stored form cannot be looked up, and the
            // scan answers it exactly
            None => return Ok(None),
        }
    }
    let borrowed: Vec<Option<&[u8]>> = cells.iter().map(|c| c.as_deref()).collect();
    let (addresses, stats) = zyron_lake::probe_equal(paths, manifest, spec, &borrowed)?;
    Ok(Some(IndexResolution {
        rows: zyron_lake::group_by_partition(&addresses),
        name: spec.name.clone(),
        files_read: stats.files_opened,
    }))
}

/// Reads the data files of one lake table at one log version.
pub struct LakeScanOperator {
    ctx: Arc<ExecutionContext>,
    table_id: u32,
    output_columns: Vec<LogicalColumn>,
    predicate: Option<BoundExpr>,
    paths: LakePaths,
    manifest: Arc<ManifestFile>,
    /// Partition ids in manifest order, consumed front to back.
    files: Vec<u64>,
    /// Files the manifest listed that statistics excluded, reported so a
    /// caller can see how much IO the predicate saved.
    files_pruned: usize,
    file_idx: usize,
    /// When set, emit RowLocator::Lake per surviving row for DML addressing.
    emit_locators: bool,
    pending: VecDeque<ExecutionBatch>,
    finished: bool,
    /// The predicate's exact lowering, kept so the finished scan can report
    /// what it selected against the same terms the plan observed
    lowered: Option<zyron_lake::LakePredicate>,
    /// Rows decoded and rows returned, reported once when the scan is
    /// exhausted. Selectivity is the one thing statistics cannot say in
    /// advance, and it is what places the clustering planner's replay probe
    rows_scanned: u64,
    rows_matched: u64,
    /// Bytes the manifest listed and bytes its statistics excluded, so
    /// EXPLAIN ANALYZE can report the IO the predicate saved rather than
    /// leaving it to be inferred from the row count
    bytes_considered: u64,
    bytes_skipped: u64,
    /// The predicate lowered onto stored bytes, applied per file before
    /// any projected column is decoded. None when nothing lowered
    stored_filter: Option<zyron_lake::StoredFilter>,
    /// Files whose zone maps or encoded bytes left no surviving row, and
    /// what they would have cost to decode. Counted here rather than at
    /// build time because they are only known once the file is opened
    files_skipped_on_read: usize,
    bytes_skipped_on_read: u64,
    /// Rows a secondary index resolved the predicate to, keyed by data
    /// file and ascending within it. Present only when an index led with a
    /// column the predicate compares for equality and its files covered
    /// every live data file. The scan then reads those rows instead of
    /// every row of the surviving files, and the exact row filter still
    /// runs so the index narrows work and never decides the answer
    index_rows: Option<std::collections::BTreeMap<u64, Vec<u64>>>,
    /// The index that produced them, reported by EXPLAIN
    index_name: Option<String>,
    /// Index files the probe opened, zero when no index answered
    index_files_read: usize,
    /// Where the counters are published, so EXPLAIN ANALYZE reports every
    /// file skipped rather than only the ones the manifest rejected
    metrics: Option<Arc<crate::operator::OperatorMetrics>>,
    /// This table's IO counters, updated per file with the rows it yielded and
    /// the bytes of projected column segments read to yield them. A file the
    /// manifest pruned, or one its zone maps emptied, contributes zero of both,
    /// which is how skipping shows up as bytes the query never read
    io_stats: Option<Arc<zyron_common::TableIOStats>>,
}

impl LakeScanOperator {
    pub fn new(
        ctx: Arc<ExecutionContext>,
        table_id: zyron_catalog::TableId,
        columns: Vec<LogicalColumn>,
        predicate: Option<BoundExpr>,
        // The planner's exact lowering of the predicate, when it has one
        lowered: Option<zyron_lake::LakePredicate>,
        as_of: Option<AsOfTarget>,
    ) -> Result<Self> {
        let table_entry = ctx.get_table_entry(table_id)?;
        if !table_entry.lake.is_lake() {
            return Err(ZyronError::ExecutionError(format!(
                "lake scan of non-lake table \"{}\"",
                table_entry.name
            )));
        }
        // The registry is primed with the reconciled log at server startup,
        // a first open here only happens for a freshly created table whose
        // every version is already durable
        let paths = LakePaths::new(ctx.disk_manager.data_dir(), table_entry.id.0);
        // A branch is an alternate log head over the same data files, so
        // reading one is opening its log instead of main's. Everything
        // downstream is unchanged: the manifest names files, the files are
        // immutable, and both heads address the same directory
        let head = effective_head(&ctx, as_of.as_ref());
        let log = open_lake_head(&paths, &table_entry.name, head)?;
        let version = match as_of {
            // The named head, which is what a branch qualifier or a
            // branched session asks for
            None | Some(AsOfTarget::Branch(_)) => log.latest_version(),
            Some(AsOfTarget::Version(v)) => resolve_version(&log, TimeTravelSpec::Version(v))?,
            Some(AsOfTarget::Timestamp(us)) => {
                resolve_version(&log, TimeTravelSpec::Timestamp(us))?
            }
        };
        let manifest = log.manifest_at(version)?;
        // Every projected column must exist in the lake schema. Files that
        // predate a column decode it as NULL, the schema is the authority
        for col in &columns {
            if manifest
                .schema
                .column_by_id(col.column_id.0 as u32)
                .is_none()
            {
                return Err(ZyronError::ExecutionError(format!(
                    "column \"{}\" is not in lake table \"{}\"",
                    col.name, table_entry.name
                )));
            }
        }
        // File pruning. The query predicate lowers to the lake IR when it
        // has an exact equivalent, and files whose statistics prove they
        // cannot match are dropped with no IO at all. The row filter still
        // runs on what survives, so this only ever removes work.
        //
        // The version's projection answers the whole file set in one
        // branch-free sweep. Where the sweep has no proof and reports
        // itself short of exact, the manifest's typed statistics decide
        // that file, so the exact cost is paid only for what survived
        let mut bytes_considered = 0u64;
        let mut bytes_skipped = 0u64;
        let files: Vec<u64> = match &lowered {
            Some(lowered) => {
                let prune = log.prune_index_at(version)?;
                if prune.file_count() != manifest.entries.len() {
                    return Err(ZyronError::ExecutionError(format!(
                        "lake scan: pruning projection covers {} files, manifest version {} lists {}",
                        prune.file_count(),
                        version,
                        manifest.entries.len()
                    )));
                }
                zyron_lake::with_sweep(&prune, lowered, |mask, complete| {
                    let mut kept = Vec::with_capacity(manifest.entries.len());
                    for (f, entry) in manifest.entries.iter().enumerate() {
                        bytes_considered += entry.size_bytes;
                        let cannot = mask[f] == 1
                            || (!complete
                                && manifest.prune_file(lowered, entry)
                                    == PruneDecision::CannotMatch);
                        if cannot {
                            bytes_skipped += entry.size_bytes;
                        } else {
                            kept.push(entry.partition_id);
                        }
                    }
                    kept
                })
            }
            None => {
                bytes_considered = manifest.entries.iter().map(|e| e.size_bytes).sum();
                manifest.entries.iter().map(|e| e.partition_id).collect()
            }
        };
        // A secondary index answers an equality the statistics could only
        // narrow. It runs after pruning so it inherits the file set the
        // manifest already reduced, and it is consulted only when its
        // files cover every live data file, which is what makes an index
        // that is behind decline rather than answer short
        let resolved = match &lowered {
            Some(lowered) => resolve_through_index(&paths, &manifest, lowered, &files)?,
            None => None,
        };
        let files: Vec<u64> = match &resolved {
            Some(index) => files
                .into_iter()
                .filter(|partition_id| index.rows.contains_key(partition_id))
                .collect(),
            None => files,
        };
        let index_files_read = resolved.as_ref().map(|i| i.files_read).unwrap_or(0);
        let (index_rows, index_name) = match resolved {
            Some(index) => (Some(index.rows), Some(index.name)),
            None => (None, None),
        };
        let files_pruned = manifest.entries.len() - files.len();

        // One observation per planned scan, never per row. Pruning is
        // decided from the manifest before a byte is read, so the skip
        // measurement is already complete and observing again when the
        // scan finishes would count the same decision twice
        if let Some(lowered) = &lowered {
            zyron_lake::observe_scan(
                table_entry.id.0,
                lowered,
                bytes_considered,
                bytes_skipped,
                zyron_lake::current_epoch(),
            );
        }
        // The predicate over stored bytes, lowered once because it depends
        // on the schema and the predicate and not on any file
        let stored_filter = lowered
            .as_ref()
            .and_then(|p| zyron_lake::StoredFilter::lower(p, &manifest.schema));
        let io_stats = ctx.table_io_stats_for(table_entry.id.0);
        if let Some(stats) = &io_stats {
            stats.record_seq_scan();
        }
        Ok(Self {
            ctx,
            table_id: table_entry.id.0,
            output_columns: columns,
            predicate,
            paths,
            manifest,
            files,
            files_pruned,
            file_idx: 0,
            emit_locators: false,
            pending: VecDeque::new(),
            finished: false,
            lowered,
            rows_scanned: 0,
            rows_matched: 0,
            bytes_considered,
            bytes_skipped,
            stored_filter,
            files_skipped_on_read: 0,
            bytes_skipped_on_read: 0,
            index_rows,
            index_name,
            index_files_read,
            metrics: None,
            io_stats,
        })
    }

    /// Publishes the pruning counters where EXPLAIN ANALYZE reads them.
    ///
    /// Manifest pruning is final before the first batch, but a file the
    /// manifest kept can still be dropped once its zone maps are read, so
    /// the counters are republished as that happens
    pub fn with_metrics(mut self, metrics: Option<Arc<crate::operator::OperatorMetrics>>) -> Self {
        self.metrics = metrics;
        self.publish_pruning();
        self
    }

    fn publish_pruning(&self) {
        let Some(metrics) = &self.metrics else {
            return;
        };
        metrics.set_aux(
            crate::operator::AUX_FILES_CONSIDERED,
            (self.files.len() + self.files_pruned) as u64,
        );
        metrics.set_aux(
            crate::operator::AUX_FILES_PRUNED,
            (self.files_pruned + self.files_skipped_on_read) as u64,
        );
        metrics.set_aux(crate::operator::AUX_BYTES_CONSIDERED, self.bytes_considered);
        metrics.set_aux(
            crate::operator::AUX_BYTES_PRUNED,
            self.bytes_skipped + self.bytes_skipped_on_read,
        );
        // Which access path answered the predicate. Without this a plan
        // shows only that files were pruned, and a scan that consulted an
        // index reads the same as one whose statistics happened to be
        // enough, which is the difference somebody diagnosing a slow point
        // lookup is looking for
        metrics.set_aux(
            crate::operator::AUX_INDEX_FILES_READ,
            self.index_files_read as u64,
        );
        metrics.set_aux(
            crate::operator::AUX_INDEX_ROWS_ADDRESSED,
            self.index_rows_addressed() as u64,
        );
    }

    /// Data files this scan will open, after statistics pruning.
    pub fn files_scanned(&self) -> usize {
        self.files.len()
    }

    /// Data files the manifest listed that the predicate's statistics
    /// excluded, so no byte of them is read.
    pub fn files_pruned(&self) -> usize {
        self.files_pruned
    }

    /// Data files opened whose zone maps or encoded bytes left no
    /// surviving row, so no projected column of them was decoded.
    pub fn files_skipped_on_read(&self) -> usize {
        self.files_skipped_on_read
    }

    /// Bytes the manifest listed across every file it named.
    pub fn bytes_considered(&self) -> u64 {
        self.bytes_considered
    }

    /// Bytes in the files the statistics excluded, the IO saved.
    pub fn bytes_pruned(&self) -> u64 {
        self.bytes_skipped + self.bytes_skipped_on_read
    }

    /// The secondary index this scan resolved its predicate through, when
    /// one applied. None means every surviving file was read in full
    pub fn index_used(&self) -> Option<&str> {
        self.index_name.as_deref()
    }

    /// Rows a secondary index addressed, across every file. Zero when no
    /// index applied
    pub fn index_rows_addressed(&self) -> usize {
        self.index_rows
            .as_ref()
            .map(|m| m.values().map(|v| v.len()).sum())
            .unwrap_or(0)
    }

    /// Emits one RowLocator::Lake per surviving row so DML can address the
    /// source rows.
    pub fn with_locators(mut self) -> Self {
        self.emit_locators = true;
        self
    }

    fn load_file(&mut self, partition_id: u64) -> Result<()> {
        let entry = self.manifest.entry_for(partition_id).ok_or_else(|| {
            ZyronError::ExecutionError(format!(
                "lake scan: partition {:#x} vanished from its manifest",
                partition_id
            ))
        })?;
        let reader = LakeFileReader::open(&self.paths, partition_id)?;
        let row_count = reader.row_count();
        if row_count == 0 {
            return Ok(());
        }
        let mut keep = reader.delete_survivors(&self.manifest.schema, &self.manifest, entry)?;

        // An index resolved this file to specific rows, so everything it
        // did not name is cleared before any projected column is decoded.
        // The exact row filter still runs on what is left, which is what
        // keeps the index a way to read less rather than a second source
        // of truth about which rows match
        if let Some(rows) = self.index_rows.as_ref().and_then(|m| m.get(&partition_id)) {
            let mut addressed = vec![0u8; keep.len()];
            for ordinal in rows {
                let row = *ordinal as usize;
                if row < row_count {
                    addressed[row / 8] |= 1 << (row % 8);
                }
            }
            for (k, a) in keep.iter_mut().zip(addressed.iter()) {
                *k &= *a;
            }
        }

        // What the manifest could not reject, the file's own zone maps and
        // encoded bytes still can. A file whose bounds admit the predicate
        // can still hold no zone that does, and a term answered from a
        // dictionary or a run length segment never materializes the values
        // it rejects. Both run before a projected column is decoded
        // Skipped when an index already named this file's matching rows.
        // The filter answers the same question by reading the predicate's
        // whole column segment, which is the cost the index exists to
        // avoid, and the exact row filter still runs on what survives so
        // the terms the index did not consume are applied either way
        if self.index_rows.is_none()
            && let Some(filter) = &self.stored_filter
            && let Some(mask) = reader.rows_matching(filter)?
        {
            for (k, m) in keep.iter_mut().zip(mask.iter()) {
                *k &= *m;
            }
            if keep.iter().all(|b| *b == 0) {
                self.files_skipped_on_read += 1;
                self.bytes_skipped_on_read += entry.size_bytes;
                self.publish_pruning();
                // No projected column was decoded, but the terms answered on
                // encoded bytes read their own segments, and that is what
                // rejecting the file cost
                self.record_file_io(0, reader.bytes_read());
                return Ok(());
            }
        }

        // COUNT(*) projects nothing. A batch built from zero column builders
        // reports zero rows, so counting one would answer zero for a file
        // full of rows, which is a wrong answer rather than a slow one. The
        // heap scan carries the same fast path
        if self.output_columns.is_empty() && self.predicate.is_none() {
            let mut kept = 0usize;
            let mut locators: Vec<zyron_common::RowLocator> = Vec::new();
            for r in 0..row_count {
                if keep[r / 8] & (1 << (r % 8)) == 0 {
                    continue;
                }
                kept += 1;
                if self.emit_locators {
                    locators.push(zyron_common::RowLocator::Lake {
                        file_id: partition_id,
                        ordinal: r as u64,
                    });
                }
            }
            if kept > 0 {
                self.queue_batch(DataBatch::with_row_count(kept), locators)?;
            }
            self.record_file_io(kept as u64, reader.bytes_read());
            return Ok(());
        }

        // The ordinals still standing, which bound how much of each
        // projected column has to be decoded. An index that resolved this
        // file to a handful of rows leaves a span of a handful, so a point
        // read stops paying for the whole column
        let (span_start, span_end) = surviving_span(&keep, row_count);
        if span_start == span_end {
            self.record_file_io(0, reader.bytes_read());
            return Ok(());
        }

        // One decoded column per projected column, schema-evolved columns
        // absent from the file come back as all NULL
        let mut decoded = Vec::with_capacity(self.output_columns.len());
        for col in &self.output_columns {
            let lake_col = self
                .manifest
                .schema
                .column_by_id(col.column_id.0 as u32)
                .ok_or_else(|| {
                    ZyronError::ExecutionError(format!(
                        "lake scan: column \"{}\" missing from the manifest schema",
                        col.name
                    ))
                })?;
            decoded.push((
                col.type_id,
                lake_col.physical_type_id().fixed_size().unwrap_or(0),
                reader.read_column_range(lake_col, span_start, span_end)?,
            ));
        }

        let mut builders = create_builders(&self.output_columns, row_count.min(BATCH_SIZE));
        let mut locators: Vec<zyron_common::RowLocator> = Vec::new();
        let mut in_batch = 0usize;
        // Rows this file yielded, counted before the exact row filter runs so
        // the number means rows the scan read rather than rows it returned
        let mut rows_yielded: u64 = 0;
        for r in 0..row_count {
            if keep[r / 8] & (1 << (r % 8)) == 0 {
                continue;
            }
            for (ci, (type_id, value_size, col)) in decoded.iter().enumerate() {
                // push_owned moves a decoded text or binary allocation into
                // the column instead of copying it a second time
                let sv = match col.cell(r) {
                    None => ScalarValue::Null,
                    Some(cell) if *value_size == 0 => decode_varlen_scalar(*type_id, cell),
                    Some(cell) => decode_fixed_scalar(*type_id, cell),
                };
                builders[ci].push_owned(sv);
            }
            if self.emit_locators {
                locators.push(zyron_common::RowLocator::Lake {
                    file_id: partition_id,
                    ordinal: r as u64,
                });
            }
            in_batch += 1;
            rows_yielded += 1;
            if in_batch == BATCH_SIZE {
                let batch = finalize_builders(std::mem::replace(
                    &mut builders,
                    create_builders(&self.output_columns, BATCH_SIZE),
                ));
                let locs = std::mem::take(&mut locators);
                self.queue_batch(batch, locs)?;
                in_batch = 0;
            }
        }
        if in_batch > 0 {
            let batch = finalize_builders(builders);
            self.queue_batch(batch, locators)?;
        }
        self.record_file_io(rows_yielded, reader.bytes_read());
        Ok(())
    }

    /// Folds one file's totals into the table counters, once per file rather
    /// than once per row or once per column.
    #[inline]
    fn record_file_io(&self, rows: u64, bytes: u64) {
        if let Some(stats) = &self.io_stats {
            stats.record_seq_batch(rows, bytes);
        }
    }

    fn queue_batch(
        &mut self,
        batch: DataBatch,
        locators: Vec<zyron_common::RowLocator>,
    ) -> Result<()> {
        self.rows_scanned += batch.num_rows as u64;
        let (filtered, kept_locs) = if let Some(ref predicate) = self.predicate {
            let mask_col = evaluate(predicate, &batch, &self.output_columns, &self.ctx.params)?;
            let mask = column_to_mask(&mask_col);
            let kept = if self.emit_locators {
                mask.iter()
                    .zip(locators.iter())
                    .filter_map(|(&k, l)| if k { Some(l.clone()) } else { None })
                    .collect()
            } else {
                Vec::new()
            };
            (batch.filter(&mask), kept)
        } else {
            (batch, locators)
        };
        self.rows_matched += filtered.num_rows as u64;
        if filtered.num_rows == 0 {
            return Ok(());
        }
        let secured =
            apply_column_security(&self.ctx, self.table_id, &self.output_columns, filtered);
        if self.emit_locators {
            self.pending
                .push_back(ExecutionBatch::with_locators(secured, kept_locs));
        } else {
            self.pending.push_back(ExecutionBatch::new(secured));
        }
        Ok(())
    }
}

impl Operator for LakeScanOperator {
    fn next(&mut self) -> OperatorResult<'_> {
        Box::pin(async move {
            loop {
                if let Some(b) = self.pending.pop_front() {
                    return Ok(Some(b));
                }
                if self.finished || self.file_idx >= self.files.len() {
                    // One report per finished scan, never per row
                    if !self.finished {
                        if let Some(lowered) = &self.lowered {
                            zyron_lake::observe_scan_result(
                                self.table_id,
                                lowered,
                                self.rows_scanned,
                                self.rows_matched,
                                zyron_lake::current_epoch(),
                            );
                        }
                        if let Some(index) = self.index_used() {
                            tracing::debug!(
                                target: "zyron::lake",
                                table = self.table_id,
                                index,
                                rows_addressed = self.index_rows_addressed(),
                                files_read = self.files.len(),
                                files_pruned = self.files_pruned,
                                "lake scan resolved its predicate through an index"
                            );
                        }
                    }
                    self.finished = true;
                    return Ok(None);
                }
                let partition_id = self.files[self.file_idx];
                self.file_idx += 1;
                self.load_file(partition_id)?;
            }
        })
    }
}

/// Replaces matching rows of a lake table. The child scan produces the
/// matching rows over every column, the assignments produce their new
/// images, and one commit removes the old rows and adds the new ones.
pub struct LakeUpdateOperator {
    child: Box<dyn Operator>,
    ctx: Arc<ExecutionContext>,
    table_id: zyron_catalog::TableId,
    assignments: Vec<zyron_planner::binder::BoundAssignment>,
    check_constraints: Vec<BoundExpr>,
    predicate: Option<zyron_lake::LakePredicate>,
    sql: String,
    input_schema: Vec<LogicalColumn>,
    finished: bool,
}

impl LakeUpdateOperator {
    pub fn new(
        child: Box<dyn Operator>,
        ctx: Arc<ExecutionContext>,
        table_id: zyron_catalog::TableId,
        assignments: Vec<zyron_planner::binder::BoundAssignment>,
        check_constraints: Vec<BoundExpr>,
        predicate: Option<zyron_lake::LakePredicate>,
        sql: String,
        input_schema: Vec<LogicalColumn>,
    ) -> Self {
        Self {
            child,
            ctx,
            table_id,
            assignments,
            check_constraints,
            predicate,
            sql,
            input_schema,
            finished: false,
        }
    }
}

impl Operator for LakeUpdateOperator {
    fn next(&mut self) -> OperatorResult<'_> {
        Box::pin(async move {
            if self.finished {
                return Ok(None);
            }
            self.finished = true;
            self.ctx.ensure_writable("UPDATE")?;
            let table_entry = self.ctx.get_table_entry(self.table_id)?;

            // Accumulate the new images. The child projects every column
            // in table order, so a batch is already a full row image and
            // an assignment replaces one of its columns in place
            let mut columns: Vec<zyron_lake::ColumnData> = table_entry
                .columns
                .iter()
                .map(|c| {
                    zyron_lake::ColumnData::with_capacity(
                        c.id.0 as u32,
                        c.physical_type_id().fixed_size().unwrap_or(0),
                        0,
                    )
                })
                .collect();
            let mut matched = 0u64;
            // The new images, kept only when a search index has to be
            // maintained over them. An update writes a new data file, so
            // the rows it replaces lose their addresses and the new ones
            // have to be registered under theirs or the row stops being
            // findable by every search index on the table
            let indexes = self.ctx.index_snapshot_for_table(self.table_id.0);
            let needs_search_maintenance = !indexes.fts.is_empty()
                || !indexes.vector.is_empty()
                || !indexes.spatial.is_empty();
            // AFTER UPDATE fires on the committed images too, so they are
            // kept when the table has triggers as well
            let has_triggers = !self
                .ctx
                .catalog
                .triggers_for_table(self.table_id)
                .is_empty();
            let keep_images = needs_search_maintenance || has_triggers;
            let mut images: Vec<crate::batch::DataBatch> = Vec::new();
            // Old and new images kept for the post-commit referential pass
            // when another table references this one
            let has_referencing = !self
                .ctx
                .catalog
                .referencing_constraints(table_entry.id)
                .is_empty();
            let mut fk_pairs: Vec<(crate::batch::DataBatch, crate::batch::DataBatch)> = Vec::new();
            // Old and new images kept for the CDC notification after the
            // commit, so a feed on the table sees the replacement the same
            // way it sees a heap update
            let cdc_capture = self.ctx.cdc_hook.is_some();
            let mut cdc_pairs: Vec<(crate::batch::DataBatch, crate::batch::DataBatch)> = Vec::new();
            while let Some(batch) = self.child.next().await? {
                self.ctx.check_cancelled()?;
                if batch.batch.num_rows == 0 {
                    continue;
                }
                let mut image = batch.batch.clone();
                for assignment in &self.assignments {
                    let new_col = crate::expr::evaluate(
                        &assignment.value,
                        &batch.batch,
                        &self.input_schema,
                        &self.ctx.params,
                    )?;
                    let ce = table_entry
                        .columns
                        .iter()
                        .find(|c| c.id == assignment.column_id)
                        .ok_or_else(|| {
                            ZyronError::Internal(format!(
                                "assignment column {:?} not in table",
                                assignment.column_id
                            ))
                        })?;
                    let new_col = if new_col.type_id != ce.type_id {
                        crate::compute::cast_column(&new_col, ce.type_id)?
                    } else {
                        new_col
                    };
                    let idx = self
                        .input_schema
                        .iter()
                        .position(|lc| lc.column_id == assignment.column_id)
                        .ok_or_else(|| {
                            ZyronError::Internal(format!(
                                "assignment column {:?} not in the update projection",
                                assignment.column_id
                            ))
                        })?;
                    image.columns[idx] = new_col;
                }
                // BEFORE UPDATE sees the new image, the same one CHECK is
                // about to read, so a trigger that rejects a row stops it
                // reaching storage
                crate::trigger::fire_row_triggers(
                    &self.ctx,
                    self.table_id,
                    zyron_catalog::TriggerEntry::TIMING_BEFORE,
                    zyron_catalog::TriggerEntry::EVENT_UPDATE,
                    &image,
                    &table_entry.columns,
                )
                .await?;

                // Arrays take the element width their column declares before
                // any check reads the row
                crate::operator::modify::normalize_array_elements(
                    &mut image,
                    &table_entry.columns,
                )?;

                // Decimals take their column's scale before the row is
                // encoded, otherwise the encoder receives a float or a
                // wrong-scale integer and writes zero
                crate::operator::modify::normalize_decimal_columns(
                    &mut image,
                    &table_entry.columns,
                )?;

                // The image is what CHECK sees, so a violating update
                // aborts before anything is written
                crate::operator::modify::enforce_check_constraints(
                    &self.check_constraints,
                    &image,
                    &table_entry.columns,
                    &self.ctx.params,
                )?;

                // Child-side foreign keys hold on the new image before any
                // write, so an update orphaning this row aborts cleanly
                crate::operator::fk::check_child_fks(&self.ctx, &table_entry, &image)
                    .await?
                    .deny_diversion(&table_entry.name)?;

                // ON UPDATE referential actions that land before the write,
                // so moving a referenced key runs the declared action
                if has_referencing {
                    crate::operator::fk::enforce_parent_update(
                        &self.ctx,
                        &table_entry,
                        &batch.batch,
                        &image,
                        crate::operator::fk::FkPhase::BeforeWrite,
                    )
                    .await?;
                    fk_pairs.push((batch.batch.clone(), image.clone()));
                }
                for (ci, col_entry) in table_entry.columns.iter().enumerate() {
                    let value_size = col_entry.physical_type_id().fixed_size().unwrap_or(0);
                    let column = &image.columns[ci];
                    for r in 0..image.num_rows {
                        let cell = match column.get_scalar(r) {
                            crate::column::ScalarValue::Null => None,
                            ref v => Some(crate::batch::encode_scalar_value(
                                col_entry.type_id,
                                v,
                                value_size,
                            )),
                        };
                        columns[ci].push(cell.as_deref());
                    }
                }
                matched += image.num_rows as u64;
                if cdc_capture {
                    cdc_pairs.push((batch.batch.clone(), image.clone()));
                }
                if keep_images {
                    images.push(image);
                }
            }

            if matched == 0 {
                return Ok(Some(ExecutionBatch::new(
                    crate::operator::modify::count_batch(0),
                )));
            }

            let paths = LakePaths::new(self.ctx.disk_manager.data_dir(), table_entry.id.0);
            // The branch this session writes, so the replaced rows come off
            // the branch head and the new file lands on it. Uniqueness and
            // the predicate below read the same head, or an update would
            // compare against rows the branch does not have
            let head = effective_head(&self.ctx, None);
            let log = open_lake_write_head(&paths, &table_entry.name, head)?;
            let root = log.registry_key();
            let timestamp_us = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_micros() as i64)
                .unwrap_or(0);
            // Uniqueness over the new images, with the rows this statement
            // replaces excluded from the stored side. A row keeping its key
            // must not collide with the copy being rewritten, and a row
            // taking a key a surviving row holds still must
            let probe = crate::operator::modify::enforce_lake_unique(
                &log,
                &table_entry,
                &columns,
                self.predicate.as_ref(),
            )?;

            // Off the worker thread. The commit writes files and
            // fsyncs them, and waits between attempts when it loses a
            // race, so holding a runtime worker for it stalls every
            // other connection that thread was going to serve. The
            // handoff is microseconds against a commit measured in
            // milliseconds.
            //
            // The attempt is built inside the task because it borrows
            // the probe, which moves in with it
            let blocking_log = Arc::clone(&log);
            let blocking_predicate = self.predicate.clone();
            let blocking_sql = self.sql.clone();
            let blocking_table_id = table_entry.id.0 as u64;
            let blocking_txn_id = self.ctx.lake_txn_id();
            // The statement's own deadline, so a commit losing races to
            // other writers stops waiting when the statement is over time
            // instead of waiting forever. Unset when the session has no
            // statement timeout, which waits as before
            let blocking_deadline = self.ctx.deadline();
            let outcome = tokio::task::spawn_blocking(move || {
                let attempt = zyron_lake::CommitAttempt {
                    operation: zyron_lake::OperationKind::Update,
                    db_txn_id: blocking_txn_id,
                    commit_lsn: 0,
                    timestamp_us,
                    // The probe's key ranges pinned at the probed head, so a
                    // concurrent commit that lands a probed key after the
                    // probe conflicts instead of committing a duplicate
                    read_predicate: probe.as_ref().map(|(p, _)| p),
                    read_version: probe.as_ref().map(|(_, v)| *v).unwrap_or(0),
                    audit: None,
                    deadline: blocking_deadline,
                };
                zyron_lake::update_where(
                    &blocking_log,
                    attempt,
                    blocking_table_id,
                    blocking_predicate.as_ref(),
                    &blocking_sql,
                    &columns,
                    matched,
                )
            })
            .await
            .map_err(|e| {
                zyron_common::ZyronError::Internal(format!(
                    "lake update task failed to run to completion: {e}"
                ))
            })??;
            zyron_lake::register_txn_pending(
                self.ctx.disk_manager.data_dir(),
                self.ctx.lake_txn_id(),
                root,
                outcome.version,
            );
            self.ctx.mark_wrote_wal();
            // The rows only have addresses once the commit assigned them,
            // so the search indexes take the new images here rather than
            // per batch above
            if needs_search_maintenance {
                crate::operator::modify::maintain_lake_search_indexes(
                    &self.ctx,
                    &table_entry,
                    &images,
                    outcome.partition_id,
                    &outcome.order,
                )?;
            }
            // One commit replaced the old images with the new, so the feed
            // gets one notification pairing them under the committed version
            if let Some(ref hook) = self.ctx.cdc_hook {
                let mut old_encoded: Vec<Vec<u8>> = Vec::new();
                let mut new_encoded: Vec<Vec<u8>> = Vec::new();
                for (old, new) in &cdc_pairs {
                    for r in 0..old.num_rows {
                        old_encoded.push(crate::batch::encode_row(old, r, &table_entry.columns));
                    }
                    for r in 0..new.num_rows {
                        new_encoded.push(crate::batch::encode_row(new, r, &table_entry.columns));
                    }
                }
                let old_refs: Vec<&[u8]> = old_encoded.iter().map(|v| v.as_slice()).collect();
                let new_refs: Vec<&[u8]> = new_encoded.iter().map(|v| v.as_slice()).collect();
                if let Err(e) = hook.on_update(
                    self.table_id.0,
                    &old_refs,
                    &new_refs,
                    outcome.version,
                    timestamp_us,
                    self.ctx.txn_id,
                    true,
                ) {
                    eprintln!("CDC update hook failed: {e}");
                }
            }
            // ON UPDATE actions that need the moved key committed before
            // they can re-check the children against it
            for (old_batch, new_image) in &fk_pairs {
                crate::operator::fk::enforce_parent_update(
                    &self.ctx,
                    &table_entry,
                    old_batch,
                    new_image,
                    crate::operator::fk::FkPhase::AfterWrite,
                )
                .await?;
            }
            // AFTER UPDATE fires once the new images are committed, because
            // a trigger body that reads the table has to see them
            for image in &images {
                crate::trigger::fire_row_triggers(
                    &self.ctx,
                    self.table_id,
                    zyron_catalog::TriggerEntry::TIMING_AFTER,
                    zyron_catalog::TriggerEntry::EVENT_UPDATE,
                    image,
                    &table_entry.columns,
                )
                .await?;
            }
            Ok(Some(ExecutionBatch::new(
                crate::operator::modify::count_batch(outcome.rows_updated as i64),
            )))
        })
    }
}

/// Records a predicate delete in a lake table's log. Files the predicate
/// fully covers are dropped whole with no data IO, files it may match
/// carry the predicate until a later optimize rewrites them, and readers
/// filter through it meanwhile so the delete is visible immediately.
pub struct LakeDeleteOperator {
    ctx: Arc<ExecutionContext>,
    table_id: zyron_catalog::TableId,
    predicate: Option<zyron_lake::LakePredicate>,
    /// The same row-selecting predicate in bound form, so referential
    /// enforcement can gather exactly the rows the delete removes
    bound_predicate: Option<BoundExpr>,
    sql: String,
    finished: bool,
}

impl LakeDeleteOperator {
    pub fn new(
        ctx: Arc<ExecutionContext>,
        table_id: zyron_catalog::TableId,
        predicate: Option<zyron_lake::LakePredicate>,
        bound_predicate: Option<BoundExpr>,
        sql: String,
    ) -> Self {
        Self {
            ctx,
            table_id,
            predicate,
            bound_predicate,
            sql,
            finished: false,
        }
    }
}

impl Operator for LakeDeleteOperator {
    fn next(&mut self) -> OperatorResult<'_> {
        Box::pin(async move {
            if self.finished {
                return Ok(None);
            }
            self.finished = true;
            self.ctx.ensure_writable("DELETE")?;
            let table_entry = self.ctx.get_table_entry(self.table_id)?;

            // Referential actions, DELETE triggers and the CDC notification
            // run over the rows this delete removes, so they are gathered
            // first when any of the three applies. The gather scan reads the
            // same effective head the commit below writes, and the bound
            // predicate reproduces exactly the rows the lowered predicate
            // removes
            let needs_old_rows = !self
                .ctx
                .catalog
                .referencing_constraints(table_entry.id)
                .is_empty()
                || !self
                    .ctx
                    .catalog
                    .triggers_for_table(self.table_id)
                    .is_empty()
                || self.ctx.cdc_hook.is_some();
            let mut old_batches: Vec<crate::batch::DataBatch> = Vec::new();
            if needs_old_rows {
                let scan_columns: Vec<LogicalColumn> = table_entry
                    .columns
                    .iter()
                    .map(|c| LogicalColumn {
                        table_idx: Some(0),
                        column_id: c.id,
                        name: c.name.clone(),
                        type_id: c.type_id,
                        nullable: c.nullable,
                        fractional_digits: c.fractional_digits,
                    })
                    .collect();
                let mut scan = LakeScanOperator::new(
                    Arc::clone(&self.ctx),
                    self.table_id,
                    scan_columns,
                    self.bound_predicate.clone(),
                    self.predicate.clone(),
                    None,
                )?;
                while let Some(b) = scan.next().await? {
                    if b.batch.num_rows == 0 {
                        continue;
                    }
                    crate::operator::fk::enforce_parent_delete(
                        &self.ctx,
                        &table_entry,
                        &b.batch,
                        crate::operator::fk::FkPhase::BeforeWrite,
                    )
                    .await?;
                    crate::trigger::fire_row_triggers(
                        &self.ctx,
                        self.table_id,
                        zyron_catalog::TriggerEntry::TIMING_BEFORE,
                        zyron_catalog::TriggerEntry::EVENT_DELETE,
                        &b.batch,
                        &table_entry.columns,
                    )
                    .await?;
                    old_batches.push(b.batch);
                }
            }

            let paths = LakePaths::new(self.ctx.disk_manager.data_dir(), table_entry.id.0);
            // The branch head, so the delete records against the files the
            // branch has rather than main's
            let head = effective_head(&self.ctx, None);
            let log = open_lake_write_head(&paths, &table_entry.name, head)?;
            let root = log.registry_key();
            let timestamp_us = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_micros() as i64)
                .unwrap_or(0);
            // Off the worker thread, as the update path above. A delete
            // commits a predicate rather than a file, but it still writes
            // and fsyncs a version and still waits when it loses a race
            let blocking_log = Arc::clone(&log);
            let blocking_predicate = self.predicate.clone();
            let blocking_sql = self.sql.clone();
            let blocking_txn_id = self.ctx.lake_txn_id();
            // The statement's own deadline, so a commit losing races to
            // other writers stops waiting when the statement is over time
            // instead of waiting forever. Unset when the session has no
            // statement timeout, which waits as before
            let blocking_deadline = self.ctx.deadline();
            let outcome = tokio::task::spawn_blocking(move || {
                let attempt = zyron_lake::CommitAttempt {
                    operation: zyron_lake::OperationKind::Delete,
                    db_txn_id: blocking_txn_id,
                    commit_lsn: 0,
                    timestamp_us,
                    read_predicate: None,
                    read_version: 0,
                    audit: None,
                    deadline: blocking_deadline,
                };
                // No predicate deletes every row, which the always-true
                // predicate over a null-free existence check cannot
                // express, so it is its own path: every file is covered
                match &blocking_predicate {
                    Some(p) => zyron_lake::delete_where(&blocking_log, attempt, p, &blocking_sql),
                    None => zyron_lake::delete_all(&blocking_log, attempt),
                }
            })
            .await
            .map_err(|e| {
                zyron_common::ZyronError::Internal(format!(
                    "lake delete task failed to run to completion: {e}"
                ))
            })??;
            if let Some(version) = outcome.version {
                zyron_lake::register_txn_pending(
                    self.ctx.disk_manager.data_dir(),
                    self.ctx.lake_txn_id(),
                    root,
                    version,
                );
                self.ctx.mark_wrote_wal();
                // The rows the commit removed were gathered above, so the
                // feed sees the same images the triggers do
                if let Some(ref hook) = self.ctx.cdc_hook {
                    let mut encoded: Vec<Vec<u8>> = Vec::new();
                    for old in &old_batches {
                        for r in 0..old.num_rows {
                            encoded.push(crate::batch::encode_row(old, r, &table_entry.columns));
                        }
                    }
                    let refs: Vec<&[u8]> = encoded.iter().map(|v| v.as_slice()).collect();
                    if let Err(e) = hook.on_delete(
                        self.table_id.0,
                        &refs,
                        version,
                        timestamp_us,
                        self.ctx.txn_id,
                        true,
                    ) {
                        eprintln!("CDC delete hook failed: {e}");
                    }
                }
            }
            // ON DELETE SET DEFAULT re-checks against the parent with these
            // rows gone, and AFTER DELETE fires once the removal committed
            for old in &old_batches {
                crate::operator::fk::enforce_parent_delete(
                    &self.ctx,
                    &table_entry,
                    old,
                    crate::operator::fk::FkPhase::AfterWrite,
                )
                .await?;
                crate::trigger::fire_row_triggers(
                    &self.ctx,
                    self.table_id,
                    zyron_catalog::TriggerEntry::TIMING_AFTER,
                    zyron_catalog::TriggerEntry::EVENT_DELETE,
                    old,
                    &table_entry.columns,
                )
                .await?;
            }
            Ok(Some(ExecutionBatch::new(
                crate::operator::modify::count_batch(outcome.rows_matched as i64),
            )))
        })
    }
}
