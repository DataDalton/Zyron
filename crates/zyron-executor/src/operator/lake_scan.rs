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

use zyron_common::profile::{self, Phase};
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
use crate::operator::{
    ExecutionBatch, MetaAcc, Operator, OperatorResult, apply_column_security, expose_column_value,
    fold_rows_into_meta_accs,
};
use zyron_planner::physical::{MetaAggKind, MetaAggSpec};

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

/// Materializes one batch from the decoded columns, taking the rows
/// `ordinals` names in that order.
///
/// Column at a time. A fixed-width column whose buffer carries its type goes
/// through the bulk append, which resolves the decode and the destination once
/// for the whole run. Everything else, text and binary included, falls back to
/// the scalar per value, so an unlisted type stays correct and only stays
/// slow.
fn build_batch(
    output_columns: &[zyron_planner::logical::LogicalColumn],
    decoded: &[(zyron_common::TypeId, usize, zyron_lake::DecodedColumn)],
    ordinals: &[u32],
) -> DataBatch {
    let mut builders = create_builders(output_columns, ordinals.len());
    // Rows in order and with no gaps, which is what a scan that filtered
    // nothing produces. The gather is then the identity, and the cells can
    // be taken as one run instead of addressed one ordinal at a time
    let run_start = match (ordinals.first(), ordinals.last()) {
        (Some(&first), Some(&last)) if last as usize - first as usize + 1 == ordinals.len() => {
            Some(first)
        }
        _ => None,
    };
    for (ci, (type_id, value_size, col)) in decoded.iter().enumerate() {
        // Two concrete iterators rather than one boxed one. The whole point is
        // to take work out of a per-value loop, and a dynamic call per value
        // would put more back than the hoisting removes
        let took = if *value_size == 0 {
            false
        } else if let Some(flat) = col.flat_cells() {
            match run_start.and_then(|start| flat.run(start, ordinals.len())) {
                Some(bytes) => builders[ci].extend_fixed_run(*type_id, bytes, ordinals.len()),
                None => builders[ci].extend_fixed(*type_id, flat.cells(ordinals)),
            }
        } else {
            builders[ci].extend_fixed(*type_id, ordinals.iter().map(|&o| col.cell(o as usize)))
        };
        if took {
            continue;
        }
        for &o in ordinals {
            // push_owned moves a decoded text or binary allocation into
            // the column instead of copying it a second time
            let sv = match col.cell(o as usize) {
                None => ScalarValue::Null,
                Some(cell) if *value_size == 0 => decode_varlen_scalar(*type_id, cell),
                Some(cell) => decode_fixed_scalar(*type_id, cell),
            };
            builders[ci].push_owned(sv);
        }
    }
    finalize_builders(builders)
}

/// The part of a row filter the stored filter does not answer itself.
///
/// A conjunct whose lowering is exact is applied to the keep mask before
/// any projected column is decoded, so evaluating it again over decoded
/// values can only agree with what is already decided. What is left is what
/// the row filter still has to run, and one inexact conjunct beside it no
/// longer makes the whole predicate its own work.
///
/// Exactness is asked of the schema the manifest holds, which is the schema
/// the mask was built from, so this cannot claim a conjunct the mask did
/// not actually answer. The plan narrows the projection on the same
/// question against the catalog's schema, and where the two disagree the
/// scan decodes the predicate's own columns rather than going without them
fn scan_residual(
    predicate: &BoundExpr,
    table_entry: &zyron_catalog::TableEntry,
    schema: &zyron_lake::LakeSchema,
) -> Option<BoundExpr> {
    let mut kept: Vec<BoundExpr> = Vec::new();
    for conjunct in split_conjuncts(predicate) {
        let answered = zyron_planner::lake_predicate::lower_predicate(
            &conjunct,
            &table_entry.columns,
            &table_entry.cluster.derived,
        )
        .and_then(|lowered| zyron_lake::StoredFilter::lower(&lowered, schema))
        .is_some_and(|filter| filter.is_exact());
        if !answered {
            kept.push(conjunct);
        }
    }
    match kept.len() {
        0 => None,
        1 => kept.pop(),
        _ => {
            let mut combined = kept.remove(0);
            for conjunct in kept {
                combined = BoundExpr::BinaryOp {
                    left: Box::new(combined),
                    op: zyron_parser::ast::BinaryOperator::And,
                    right: Box::new(conjunct),
                    type_id: zyron_common::TypeId::Boolean,
                };
            }
            Some(combined)
        }
    }
}

/// The top level conjuncts of a predicate
fn split_conjuncts(expr: &BoundExpr) -> Vec<BoundExpr> {
    match expr {
        BoundExpr::Nested(inner) => split_conjuncts(inner),
        BoundExpr::BinaryOp {
            left,
            op: zyron_parser::ast::BinaryOperator::And,
            right,
            ..
        } => {
            let mut out = split_conjuncts(left);
            out.extend(split_conjuncts(right));
            out
        }
        other => vec![other.clone()],
    }
}

/// Decodes rows `start..end` of each of `columns` out of one data file.
///
/// Cells decode by the physical type, which is what sizes them. A
/// TIMESTAMP(p>6) column stores 16 byte i128 picoseconds, and decoding it
/// as its logical type would read half the cell and hand the i128 builder
/// a variant it zeroes. Every other type's physical form is its logical one.
///
/// The cells come back in the shape the plan's column declares, which is
/// the table's current type. A file written while the column was narrower,
/// or a version read from before the column widened, hands over cells the
/// reader widens on the way out, so the batch is typed the way the plan
/// expects whatever version the scan reads
fn decode_range_columns(
    reader: &LakeFileReader,
    schema: &zyron_lake::LakeSchema,
    columns: &[LogicalColumn],
    start: usize,
    end: usize,
) -> Result<Vec<(zyron_common::TypeId, usize, zyron_lake::DecodedColumn)>> {
    let mut decoded = Vec::with_capacity(columns.len());
    for col in columns {
        let lake_col = schema.column_by_id(col.column_id.0 as u32).ok_or_else(|| {
            ZyronError::ExecutionError(format!(
                "lake scan: column \"{}\" missing from the manifest schema",
                col.name
            ))
        })?;
        let wanted = declared_as(lake_col, col.type_id, col.fractional_digits);
        let physical = wanted.physical_type_id();
        decoded.push((
            physical,
            physical.fixed_size().unwrap_or(0),
            reader.read_column_range(&wanted, start, end)?,
        ));
    }
    Ok(decoded)
}

/// The lake column as the plan declares it, so the reader hands the cells
/// over in the plan's shape rather than the shape one manifest version
/// declares. Borrowed when the two agree, which is the common case
pub fn declared_as<'a>(
    lake_col: &'a zyron_lake::LakeColumn,
    type_id: zyron_common::TypeId,
    fractional_digits: Option<u8>,
) -> std::borrow::Cow<'a, zyron_lake::LakeColumn> {
    if lake_col.type_id == type_id && lake_col.fractional_digits == fractional_digits {
        return std::borrow::Cow::Borrowed(lake_col);
    }
    let mut wanted = lake_col.clone();
    wanted.type_id = type_id;
    wanted.fractional_digits = fractional_digits;
    std::borrow::Cow::Owned(wanted)
}

/// The columns a row filter reads, when the projection does not carry all
/// of them.
///
/// Returns empty whenever every column the predicate references is
/// projected, which is what the filter is then evaluated against. A plan
/// that withheld a column, because the scan was expected to answer the
/// predicate on stored bytes, leaves the full set here so the filter has
/// something to read if that expectation does not hold for a file
fn filter_columns_outside(
    predicate: &Option<BoundExpr>,
    projected: &[LogicalColumn],
    schema: &zyron_lake::LakeSchema,
) -> Vec<LogicalColumn> {
    let Some(predicate) = predicate else {
        return Vec::new();
    };
    let mut refs = zyron_planner::collect_column_refs(predicate);
    refs.sort_unstable_by_key(|r| (r.table_idx, r.column_id.0));
    refs.dedup_by_key(|r| (r.table_idx, r.column_id.0));
    let covered = |r: &zyron_planner::binder::ColumnRef| {
        projected
            .iter()
            .any(|c| c.table_idx == Some(r.table_idx) && c.column_id == r.column_id)
    };
    if refs.iter().all(covered) {
        return Vec::new();
    }
    // Typed from the reference the binder resolved rather than from the
    // schema, so a column this manifest does not name is still described
    // and the decode is what reports it missing
    refs.iter()
        .map(|r| LogicalColumn {
            table_idx: Some(r.table_idx),
            column_id: r.column_id,
            name: schema
                .column_by_id(r.column_id.0 as u32)
                .map(|c| c.name.clone())
                .unwrap_or_else(|| format!("column {}", r.column_id.0)),
            type_id: r.type_id,
            nullable: r.nullable,
            fractional_digits: r.fractional_digits,
        })
        .collect()
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
    /// A cursor over `files` shared with the other scans of one fan-out,
    /// each taking the next file nobody has claimed. None for a scan that
    /// reads its list alone
    shared_cursor: Option<Arc<std::sync::atomic::AtomicUsize>>,
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
    /// The predicate lowered against each older shape a file of this scan
    /// holds, by the schema id the file was written under, so the files of
    /// one shape share one lowering rather than each lowering it again
    old_shape_filters: std::collections::HashMap<u64, Option<zyron_lake::StoredFilter>>,
    /// Whether the lowered predicate is equivalent to the bound one rather
    /// than merely implied by it.
    ///
    /// A scan lowering keeps the conjuncts with a lake form and drops the
    /// rest, which is what buys pruning for a predicate one LIKE would
    /// otherwise make unprunable. The stored filter's exactness is about
    /// what it was lowered from, so it says nothing about the predicate
    /// unless nothing was dropped
    lowering_is_complete: bool,
    /// Columns the row filter reads, kept only when the projection does not
    /// carry all of them.
    ///
    /// The plan withholds a column that exists only to feed a predicate the
    /// scan was expected to answer on encoded bytes. A file whose lowering
    /// turns out short of exact still has to evaluate that predicate, and it
    /// does so against a batch built from these rather than from the
    /// projection, which no longer holds what it reads. Empty in every case
    /// where the projection already covers the predicate, which is the case
    /// the plan intends
    filter_columns: Vec<LogicalColumn>,
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
        // A scan lowering keeps the conjuncts that have a lake form and
        // drops the rest, so it is implied by the predicate rather than
        // equivalent to it. The stored filter is exact for what it was
        // lowered from, which is the predicate itself only when nothing was
        // dropped, and skipping the row filter on anything less would
        // return the rows a dropped conjunct excludes
        let lowering_is_complete = predicate.as_ref().is_none_or(|p| {
            zyron_planner::lake_predicate::lower_predicate(
                p,
                &table_entry.columns,
                &table_entry.cluster.derived,
            )
            .is_some()
        });
        // The conjuncts the stored filter answers exactly are already in the
        // keep mask, so what is left of the predicate is all the row filter
        // has to run. Only when the filter actually runs: an index path
        // resolves rows without it and answers no term at all
        let predicate = match (&stored_filter, index_rows.is_none()) {
            (Some(_), true) => predicate
                .as_ref()
                .and_then(|p| scan_residual(p, &table_entry, &manifest.schema)),
            _ => predicate,
        };
        let filter_columns = filter_columns_outside(&predicate, &columns, &manifest.schema);
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
            shared_cursor: None,
            emit_locators: false,
            pending: VecDeque::new(),
            finished: false,
            lowered,
            rows_scanned: 0,
            rows_matched: 0,
            bytes_considered,
            bytes_skipped,
            stored_filter,
            old_shape_filters: std::collections::HashMap::new(),
            lowering_is_complete,
            filter_columns,
            files_skipped_on_read: 0,
            bytes_skipped_on_read: 0,
            index_rows,
            index_name,
            index_files_read,
            metrics: None,
            io_stats,
        })
    }

    /// The data files this scan will read, after pruning.
    ///
    /// A parallel aggregate splits the work by file, and splitting the
    /// pruned set rather than the manifest's whole list is what keeps the
    /// workers even when a predicate has already removed most of it
    pub fn files(&self) -> &[u64] {
        &self.files
    }

    /// The manifest version this scan resolved to, for a caller deciding
    /// how to run it from the statistics the version carries
    pub fn manifest(&self) -> &ManifestFile {
        &self.manifest
    }

    /// Narrows a built scan to the files named, dropping the rest.
    ///
    /// The metadata aggregate reads its answer off the manifest for every
    /// file whose statistics settle it, and needs a scan of only what is
    /// left over. The byte counters follow the narrowing, so what the scan
    /// reports considering is what it opened
    pub fn restrict_to_files(&mut self, keep: &std::collections::HashSet<u64>) {
        self.files.retain(|id| keep.contains(id));
        let considered: u64 = self
            .manifest
            .entries
            .iter()
            .filter(|e| keep.contains(&e.partition_id))
            .map(|e| e.size_bytes)
            .sum();
        self.files_pruned = self.manifest.entries.len() - self.files.len();
        self.bytes_skipped += self.bytes_considered.saturating_sub(considered);
        self.bytes_considered = considered;
    }

    /// Hands this scan the file list of a fan-out and the cursor its
    /// workers claim from, so each file is read by exactly one of them and
    /// a slow worker takes fewer files than a fast one.
    ///
    /// The list replaces the scan's own so every worker indexes the same
    /// order, and the counters follow it as a narrowing does
    pub fn share_files(&mut self, files: &[u64], cursor: Arc<std::sync::atomic::AtomicUsize>) {
        let keep: std::collections::HashSet<u64> = files.iter().copied().collect();
        self.restrict_to_files(&keep);
        self.files = files.to_vec();
        self.shared_cursor = Some(cursor);
    }

    /// The position of the next file to read, claimed from the shared
    /// cursor when there is one
    fn next_file_position(&mut self) -> usize {
        match &self.shared_cursor {
            Some(cursor) => cursor.fetch_add(1, std::sync::atomic::Ordering::Relaxed),
            None => {
                let at = self.file_idx;
                self.file_idx += 1;
                at
            }
        }
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
        let _total = profile::scope(Phase::LakeLoadFile);
        let entry = self.manifest.entry_for(partition_id).ok_or_else(|| {
            ZyronError::ExecutionError(format!(
                "lake scan: partition {:#x} vanished from its manifest",
                partition_id
            ))
        })?;
        let reader = {
            let _s = profile::scope(Phase::LakeOpenFile);
            LakeFileReader::open_shared_in(&self.manifest, &self.paths, partition_id)?
        };
        // The reader is shared with whatever scanned this file before, so its
        // counter is a running total. What this scan read is the difference
        // across its own reads
        let bytes_before = reader.bytes_read();
        let bytes_read = || reader.bytes_read().saturating_sub(bytes_before);
        let row_count = reader.row_count();
        if row_count == 0 {
            return Ok(());
        }
        let mut keep = {
            let _s = profile::scope(Phase::LakeDeleteSurvivors);
            reader.delete_survivors(&self.manifest.schema, &self.manifest, entry)?
        };

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
        // Whether what the stored filter left standing is the matching rows
        // themselves rather than a superset of them. An exact lowering
        // answers the predicate on encoded bytes, so evaluating it a second
        // time over decoded values can only agree with what is already
        // decided. An index path does not run the filter at all, so it
        // decides nothing here
        // A file written while a column had a narrower type holds that
        // column's cells at the narrower width, and the stored filter
        // compares constants against stored bytes. Such a file is answered
        // by a filter lowered against its own schema, so the constants are
        // encoded the way its cells are and the answer stays exact, and the
        // files of one shape share that lowering. A file holding every
        // column as declared, which is every file of a table whose types
        // never changed, uses the filter lowered once
        let file_filter: Option<std::borrow::Cow<'_, zyron_lake::StoredFilter>> =
            if self.manifest.types_changed_since(entry.schema_id) {
                let manifest = &self.manifest;
                let lowered = self.lowered.as_ref();
                self.old_shape_filters
                    .entry(entry.schema_id)
                    .or_insert_with(|| {
                        let schema = manifest.file_schema(entry);
                        lowered.and_then(|p| zyron_lake::StoredFilter::lower(p, &schema))
                    })
                    .as_ref()
                    .map(std::borrow::Cow::Borrowed)
            } else {
                self.stored_filter.as_ref().map(std::borrow::Cow::Borrowed)
            };
        let answered = self.index_rows.is_none()
            && self.lowering_is_complete
            && file_filter
                .as_deref()
                .is_some_and(|filter| filter.is_exact());
        let stored_mask = {
            let _s = profile::scope(Phase::LakeStoredFilter);
            match (self.index_rows.is_none(), file_filter.as_deref()) {
                (true, Some(filter)) => reader.rows_matching(filter)?,
                _ => None,
            }
        };
        if let Some(mask) = stored_mask {
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
                self.record_file_io(0, bytes_read());
                return Ok(());
            }
        }

        // COUNT(*) projects nothing. A batch built from zero column builders
        // reports zero rows, so counting one would answer zero for a file
        // full of rows, which is a wrong answer rather than a slow one. The
        // heap scan carries the same fast path.
        // A predicate the stored filter answered exactly is already applied
        // to the keep mask, so the count is over what stands
        if self.output_columns.is_empty() && (self.predicate.is_none() || answered) {
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
                self.queue_batch(DataBatch::with_row_count(kept), locators, true, None)?;
            }
            self.record_file_io(kept as u64, bytes_read());
            return Ok(());
        }

        // The ordinals still standing, which bound how much of each
        // projected column has to be decoded. An index that resolved this
        // file to a handful of rows leaves a span of a handful, so a point
        // read stops paying for the whole column
        let (span_start, span_end) = surviving_span(&keep, row_count);
        if span_start == span_end {
            self.record_file_io(0, bytes_read());
            return Ok(());
        }

        // One decoded column per projected column, schema-evolved columns
        // absent from the file come back as all NULL
        let decode = profile::scope(Phase::LakeDecodeColumns);
        let decoded = decode_range_columns(
            &reader,
            &self.manifest.schema,
            &self.output_columns,
            span_start,
            span_end,
        )?;
        // The row filter's own columns, decoded only when the projection
        // does not carry them and the stored filter did not settle the
        // predicate for this file
        let residual = if answered || self.filter_columns.is_empty() {
            Vec::new()
        } else {
            decode_range_columns(
                &reader,
                &self.manifest.schema,
                &self.filter_columns,
                span_start,
                span_end,
            )?
        };

        // A batch's ordinals first, then one pass per column over them, rather
        // than one pass per row over the columns. Which scalar to build from a
        // cell and which buffer it belongs in are the same answers for every
        // value in a column, and settling them per value is what a decoded
        // column scan spends its time on. Gathering the ordinals costs four
        // bytes per row of a batch and takes that decision out of the loop
        drop(decode);
        let mut ordinals: Vec<u32> = Vec::with_capacity(BATCH_SIZE.min(row_count));
        // Rows this file yielded, counted before the exact row filter runs so
        // the number means rows the scan read rather than rows it returned
        let mut rows_yielded: u64 = 0;
        let mut r = span_start;
        loop {
            while r < span_end && ordinals.len() < BATCH_SIZE {
                if keep[r / 8] & (1 << (r % 8)) != 0 {
                    ordinals.push(r as u32);
                }
                r += 1;
            }
            if ordinals.is_empty() {
                break;
            }
            let (batch, filter_batch) = {
                let _s = profile::scope(Phase::LakeBuildBatch);
                // A projection of nothing is a row count, which is what a
                // COUNT(*) whose filter columns were withheld asks for
                let batch = if self.output_columns.is_empty() {
                    DataBatch::with_row_count(ordinals.len())
                } else {
                    build_batch(&self.output_columns, &decoded, &ordinals)
                };
                let filter_batch = (!residual.is_empty())
                    .then(|| build_batch(&self.filter_columns, &residual, &ordinals));
                (batch, filter_batch)
            };
            let locators: Vec<zyron_common::RowLocator> = if self.emit_locators {
                ordinals
                    .iter()
                    .map(|&o| zyron_common::RowLocator::Lake {
                        file_id: partition_id,
                        ordinal: o as u64,
                    })
                    .collect()
            } else {
                Vec::new()
            };
            rows_yielded += ordinals.len() as u64;
            self.queue_batch(batch, locators, answered, filter_batch)?;
            ordinals.clear();
            if r >= span_end {
                break;
            }
        }
        self.record_file_io(rows_yielded, bytes_read());
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

    /// Adds one batch to the output queue, applying the row filter unless
    /// the scan has already answered it.
    ///
    /// `answered` says the keep mask this batch was gathered from is the
    /// matching rows and not a superset, which is what an exactly lowered
    /// stored filter produces. The predicate then has no row left to
    /// remove, so evaluating it would rebuild an all-keep mask and
    /// `filter` would copy every column of the batch to reproduce it.
    ///
    /// `filter_batch` carries the predicate's own columns for the case
    /// where the projection no longer holds them, gathered over the same
    /// ordinals so its mask addresses the same rows
    fn queue_batch(
        &mut self,
        batch: DataBatch,
        locators: Vec<zyron_common::RowLocator>,
        answered: bool,
        filter_batch: Option<DataBatch>,
    ) -> Result<()> {
        let _s = profile::scope(Phase::LakeQueueBatch);
        self.rows_scanned += batch.num_rows as u64;
        let (filtered, kept_locs) = match &self.predicate {
            Some(predicate) if !answered => {
                let mask = {
                    let (eval_batch, eval_schema) = match &filter_batch {
                        Some(fb) => (fb, self.filter_columns.as_slice()),
                        None => (&batch, self.output_columns.as_slice()),
                    };
                    let mask_col = evaluate(predicate, eval_batch, eval_schema, &self.ctx.params)?;
                    column_to_mask(&mask_col)
                };
                // A predicate that removed nothing still describes every row
                // of the batch, and rebuilding it column by column to say so
                // copies as much as a filter that removed most of them
                if mask.iter().all(|k| *k) {
                    (batch, locators)
                } else {
                    let kept = if self.emit_locators {
                        mask.iter()
                            .zip(locators.iter())
                            .filter_map(|(&k, l)| if k { Some(*l) } else { None })
                            .collect()
                    } else {
                        Vec::new()
                    };
                    // A batch of no columns is a row count, and selecting
                    // rows out of one is counting the rows that stand
                    let filtered = if batch.num_columns() == 0 {
                        DataBatch::with_row_count(mask.iter().filter(|k| **k).count())
                    } else {
                        batch.filter(&mask)
                    };
                    (filtered, kept)
                }
            }
            _ => (batch, locators),
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
            let _scan = profile::scope(Phase::ExecLakeScanNext);
            loop {
                if let Some(b) = self.pending.pop_front() {
                    return Ok(Some(b));
                }
                let position = if self.finished {
                    self.files.len()
                } else {
                    self.next_file_position()
                };
                if position >= self.files.len() {
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
                let partition_id = self.files[position];
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

            // The child projects the live columns in table order, and every
            // read below is by position in the table's whole column list, so
            // a table with a dropped column widens each batch to that list
            // before an assignment replaces one of its columns in place
            let shape = crate::operator::modify::TableShape::of(&table_entry);
            let fill = match &shape {
                Some(shape) => shape.widen_schema(&mut self.input_schema, &table_entry)?,
                None => false,
            };
            // Accumulate the new images, the live columns taken from their
            // positions in the image, since a dropped column holds its
            // position there with nothing to store
            let live: Vec<(usize, &zyron_catalog::ColumnEntry)> = table_entry
                .columns
                .iter()
                .enumerate()
                .filter(|(_, c)| !c.dropped)
                .collect();
            let mut columns: Vec<zyron_lake::ColumnData> = live
                .iter()
                .map(|(_, c)| {
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
            // commit, so a hook recording this table's rows sees the
            // replacement the same way it sees a heap update. Kept only
            // when the hook records them, the images are for it alone
            let cdc_capture = self.ctx.captures_changes()
                && self.ctx.cdc_hook.as_ref().is_some_and(|hook| {
                    hook.records_rows_of(self.table_id.0, self.ctx.active_branch_id)
                });
            let mut cdc_pairs: Vec<(crate::batch::DataBatch, crate::batch::DataBatch)> = Vec::new();
            while let Some(mut batch) = self.child.next().await? {
                self.ctx.check_cancelled()?;
                if batch.batch.num_rows == 0 {
                    continue;
                }
                if fill && let Some(shape) = &shape {
                    shape.fill_dropped(&mut batch.batch, &table_entry.name)?;
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
                    &self.ctx,
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
                for (li, (ci, col_entry)) in live.iter().enumerate() {
                    let value_size = col_entry.physical_type_id().fixed_size().unwrap_or(0);
                    let column = &image.columns[*ci];
                    for r in 0..image.num_rows {
                        let cell = match column.get_scalar(r) {
                            crate::column::ScalarValue::Null => None,
                            ref v => Some(crate::batch::encode_scalar_value(
                                col_entry.type_id,
                                v,
                                value_size,
                            )),
                        };
                        columns[li].push(cell.as_deref());
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
            // The commit registered its version with the pending registry
            // the transaction's end publishes from
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
            if cdc_capture && let Some(capture) = self.ctx.change_capture(outcome.version) {
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
                // A lake commit dates its own version, so the instant the
                // records carry is the commit's rather than the capture's
                let timestamp = match self.ctx.change_capture_mode {
                    crate::context::ChangeCaptureMode::Applied => capture.timestamp,
                    _ => timestamp_us,
                };
                capture
                    .hook
                    .on_update(
                        self.table_id.0,
                        &old_refs,
                        &new_refs,
                        capture.version,
                        timestamp,
                        self.ctx.txn_id,
                        true,
                        capture.branch,
                    )
                    .map_err(|e| {
                        ZyronError::ExecutionError(format!("CDC update hook failed: {e}"))
                    })?;
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
            // The images are gathered for the hook only when it records
            // this table's rows, a hook deriving the table's changes from
            // its log reads nothing here
            let cdc_capture = self.ctx.captures_changes()
                && self.ctx.cdc_hook.as_ref().is_some_and(|hook| {
                    hook.records_rows_of(self.table_id.0, self.ctx.active_branch_id)
                });
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
                || cdc_capture;
            let mut old_batches: Vec<crate::batch::DataBatch> = Vec::new();
            if needs_old_rows {
                // The lake holds the live columns alone, and the rows they
                // make are read by position in the table's whole column
                // list, so each gathered batch is widened to that list
                let scan_columns: Vec<LogicalColumn> = table_entry
                    .live_columns()
                    .map(|c| LogicalColumn {
                        table_idx: Some(0),
                        column_id: c.id,
                        name: c.name.clone(),
                        type_id: c.type_id,
                        nullable: c.nullable,
                        fractional_digits: c.fractional_digits,
                    })
                    .collect();
                let shape = crate::operator::modify::TableShape::of(&table_entry);
                let mut scan = LakeScanOperator::new(
                    Arc::clone(&self.ctx),
                    self.table_id,
                    scan_columns,
                    self.bound_predicate.clone(),
                    self.predicate.clone(),
                    None,
                )?;
                while let Some(mut b) = scan.next().await? {
                    if b.batch.num_rows == 0 {
                        continue;
                    }
                    if let Some(shape) = &shape {
                        shape.widen(&mut b.batch, &table_entry.name)?;
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
                // The commit registered its version with the pending
                // registry the transaction's end publishes from
                self.ctx.mark_wrote_wal();
                // The rows the commit removed were gathered above, so the
                // feed sees the same images the triggers do
                if cdc_capture && let Some(capture) = self.ctx.change_capture(version) {
                    let mut encoded: Vec<Vec<u8>> = Vec::new();
                    for old in &old_batches {
                        for r in 0..old.num_rows {
                            encoded.push(crate::batch::encode_row(old, r, &table_entry.columns));
                        }
                    }
                    let refs: Vec<&[u8]> = encoded.iter().map(|v| v.as_slice()).collect();
                    let timestamp = match self.ctx.change_capture_mode {
                        crate::context::ChangeCaptureMode::Applied => capture.timestamp,
                        _ => timestamp_us,
                    };
                    capture
                        .hook
                        .on_delete(
                            self.table_id.0,
                            &refs,
                            capture.version,
                            timestamp,
                            self.ctx.txn_id,
                            true,
                            capture.branch,
                        )
                        .map_err(|e| {
                            ZyronError::ExecutionError(format!("CDC delete hook failed: {e}"))
                        })?;
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

// ---------------------------------------------------------------------------
// Lake metadata aggregate
// ---------------------------------------------------------------------------

/// Answers an ungrouped SUM, MIN, MAX or COUNT from the lake manifest,
/// opening no data file.
///
/// The manifest is already resident and already records, per file and per
/// column, the bounds and null count that settle MIN, MAX and both counts,
/// and the exact total that settles SUM. A whole table aggregate is
/// therefore a fold over statistics, and it costs the file count rather
/// than the row count.
///
/// A file the statistics do not describe is scanned instead: one carrying
/// delete predicates, whose live rows are not the rows the statistics were
/// taken over, and one missing the stat an aggregate needs, which is what
/// a column added after the file was written looks like. That is decided
/// per file, so a delete against one file leaves the rest on the fast path
pub struct LakeMetadataAggregateOperator {
    ctx: Arc<ExecutionContext>,
    table_id: zyron_catalog::TableId,
    specs: Vec<MetaAggSpec>,
    schema: Vec<LogicalColumn>,
    as_of: Option<AsOfTarget>,
    done: bool,
}

/// What one file contributes to one aggregate, out of its statistics alone
enum StatAnswer {
    /// The file holds no non-null value in this column, so it moves
    /// nothing. Distinct from carrying no statistics at all
    Empty,
    Count(i64),
    Value(ScalarValue),
    Sum(i128),
}

impl LakeMetadataAggregateOperator {
    pub fn new(
        ctx: Arc<ExecutionContext>,
        table_id: zyron_catalog::TableId,
        specs: Vec<MetaAggSpec>,
        schema: Vec<LogicalColumn>,
        as_of: Option<AsOfTarget>,
    ) -> Self {
        Self {
            ctx,
            table_id,
            specs,
            schema,
            as_of,
            done: false,
        }
    }

    /// What one file's statistics say about one aggregate, or None when
    /// they say nothing and the file has to be read.
    ///
    /// A column with no stats entry is one the file predates, which reads
    /// as all NULL, but nothing here separates that from a writer that
    /// recorded none, so it is answered by a scan
    fn answer_from_stats(
        spec: &MetaAggSpec,
        entry: &zyron_lake::PartitionEntry,
        manifest: &zyron_lake::ManifestFile,
    ) -> Option<StatAnswer> {
        if spec.kind == MetaAggKind::CountStar {
            return Some(StatAnswer::Count(entry.row_count as i64));
        }
        let column_id = spec.column_id?.0 as u32;
        let stats = entry.stats_for(column_id)?;
        let live = stats
            .bounds
            .row_count
            .saturating_sub(stats.bounds.null_count);
        if spec.kind == MetaAggKind::CountCol {
            return Some(StatAnswer::Count(live as i64));
        }
        // A file written while the column had a narrower shape recorded
        // its bounds and sum in that shape, and an answer in it would be
        // read as the declared one. Such a file is read rather than
        // answered from what it recorded
        if manifest
            .written_type_at(column_id, entry.schema_id)
            .is_some()
        {
            return None;
        }
        match spec.kind {
            MetaAggKind::CountStar | MetaAggKind::CountCol => None,
            MetaAggKind::Min | MetaAggKind::Max => {
                if live == 0 {
                    return Some(StatAnswer::Empty);
                }
                let bound = if spec.kind == MetaAggKind::Max {
                    stats.bounds.max.as_ref()
                } else {
                    stats.bounds.min.as_ref()
                }?;
                let physical = manifest.schema.column_by_id(column_id)?.physical_type_id();
                stat_to_scalar(physical, bound).map(StatAnswer::Value)
            }
            MetaAggKind::Sum => {
                if live == 0 {
                    return Some(StatAnswer::Empty);
                }
                stats.sum.map(StatAnswer::Sum)
            }
        }
    }

    /// Folds one file's statistics answers into the accumulators. Every
    /// aggregate has an answer by the time this runs, so a file is never
    /// counted half from statistics and half from rows
    fn fold_answers(
        answers: Vec<StatAnswer>,
        accs: &mut [MetaAcc],
        specs: &[MetaAggSpec],
    ) -> Result<()> {
        for (si, answer) in answers.into_iter().enumerate() {
            match (answer, &mut accs[si]) {
                (StatAnswer::Empty, _) => {}
                (StatAnswer::Count(n), MetaAcc::Count(c)) => *c += n,
                (StatAnswer::Value(v), MetaAcc::MinMax(m)) => {
                    MetaAcc::fold_minmax(m, v, specs[si].kind == MetaAggKind::Max);
                }
                (StatAnswer::Sum(v), MetaAcc::Sum { total, any }) => {
                    *total = total.checked_add(v).ok_or_else(|| {
                        ZyronError::ExecutionError(
                            "SUM overflowed its 128-bit accumulator".to_string(),
                        )
                    })?;
                    *any = true;
                }
                _ => {}
            }
        }
        Ok(())
    }

    /// The columns a fallback scan has to project, and where each
    /// aggregate reads its target in the batch that comes back.
    ///
    /// COUNT(*) needs no column of its own, so a request holding only that
    /// one drives the scan off the first column in the table
    fn fallback_projection(
        &self,
        table_entry: &zyron_catalog::TableEntry,
    ) -> Result<(Vec<LogicalColumn>, Vec<Option<usize>>)> {
        let as_projected = |ce: &zyron_catalog::ColumnEntry| LogicalColumn {
            table_idx: Some(0),
            column_id: ce.id,
            name: ce.name.clone(),
            type_id: ce.type_id,
            nullable: ce.nullable,
            fractional_digits: ce.fractional_digits,
        };
        let mut proj: Vec<LogicalColumn> = Vec::new();
        let mut col_to_proj: std::collections::HashMap<u16, usize> =
            std::collections::HashMap::new();
        for s in &self.specs {
            if let Some(cid) = s.column_id
                && !col_to_proj.contains_key(&cid.0)
            {
                let ce = table_entry
                    .columns
                    .iter()
                    .find(|c| c.id == cid)
                    .ok_or_else(|| {
                        ZyronError::ExecutionError(
                            "lake metadata aggregate: column not found".into(),
                        )
                    })?;
                col_to_proj.insert(cid.0, proj.len());
                proj.push(as_projected(ce));
            }
        }
        // A count over no column reads one the lake still holds, which a
        // dropped column is not
        if proj.is_empty()
            && let Some(ce) = table_entry.live_columns().next()
        {
            proj.push(as_projected(ce));
        }
        let proj_idx = self
            .specs
            .iter()
            .map(|s| s.column_id.and_then(|c| col_to_proj.get(&c.0).copied()))
            .collect();
        Ok((proj, proj_idx))
    }
}

/// One manifest statistic as the scalar a scan of the same value would
/// have produced, so an aggregate answered from statistics and one
/// answered from rows agree on the variant as well as the number
fn stat_to_scalar(
    physical: zyron_common::TypeId,
    value: &zyron_lake::LakeValue,
) -> Option<ScalarValue> {
    use zyron_lake::LakeValue;
    let mut cell = [0u8; 16];
    match value {
        LakeValue::Null => return Some(ScalarValue::Null),
        LakeValue::Str(s) => return Some(decode_varlen_scalar(physical, s.as_bytes())),
        LakeValue::Bytes(b) => return Some(decode_varlen_scalar(physical, b)),
        LakeValue::Bool(b) => cell[0] = u8::from(*b),
        // A signed value's low bytes are the narrower width's own two's
        // complement bytes, so one 16-byte little endian buffer serves
        // every width the column can be
        LakeValue::Int(v) => cell = (*v as i128).to_le_bytes(),
        LakeValue::Int128(v) => cell = v.to_le_bytes(),
        LakeValue::UInt(v) => cell = (*v as u128).to_le_bytes(),
        LakeValue::UInt128(v) => cell = v.to_le_bytes(),
        LakeValue::Float(f) => {
            if physical.fixed_size()? == 4 {
                cell[..4].copy_from_slice(&(*f as f32).to_le_bytes());
            } else {
                cell[..8].copy_from_slice(&f.to_le_bytes());
            }
        }
    }
    let width = physical.fixed_size()?;
    if width > cell.len() {
        return None;
    }
    Some(decode_fixed_scalar(physical, &cell[..width]))
}

impl Operator for LakeMetadataAggregateOperator {
    fn next(&mut self) -> OperatorResult<'_> {
        Box::pin(async move {
            if self.done {
                return Ok(None);
            }
            self.done = true;

            let table_entry = self.ctx.get_table_entry(self.table_id)?;
            let paths = LakePaths::new(self.ctx.disk_manager.data_dir(), table_entry.id.0);
            let head = effective_head(&self.ctx, self.as_of.as_ref());
            let log = open_lake_head(&paths, &table_entry.name, head)?;
            let version = match &self.as_of {
                None | Some(AsOfTarget::Branch(_)) => log.latest_version(),
                Some(AsOfTarget::Version(v)) => resolve_version(&log, TimeTravelSpec::Version(*v))?,
                Some(AsOfTarget::Timestamp(us)) => {
                    resolve_version(&log, TimeTravelSpec::Timestamp(*us))?
                }
            };
            let manifest = log.manifest_at(version)?;

            let mut accs = MetaAcc::for_specs(&self.specs);
            let mut scan_files: std::collections::HashSet<u64> = std::collections::HashSet::new();
            for entry in manifest.entries.iter() {
                // A file under a delete predicate still holds the rows its
                // statistics counted, so nothing it recorded describes what
                // is live in it
                let answers: Option<Vec<StatAnswer>> = if entry.delete_predicate_ids.is_empty() {
                    self.specs
                        .iter()
                        .map(|s| Self::answer_from_stats(s, entry, &manifest))
                        .collect()
                } else {
                    None
                };
                match answers {
                    Some(answers) => Self::fold_answers(answers, &mut accs, &self.specs)?,
                    None => {
                        scan_files.insert(entry.partition_id);
                    }
                }
            }

            if !scan_files.is_empty() {
                let (proj, proj_idx) = self.fallback_projection(&table_entry)?;
                let mut scan = LakeScanOperator::new(
                    self.ctx.clone(),
                    self.table_id,
                    proj,
                    None,
                    None,
                    self.as_of.clone(),
                )?;
                scan.restrict_to_files(&scan_files);
                fold_rows_into_meta_accs(Box::new(scan), &self.specs, &proj_idx, &mut accs).await?;
            }

            // MIN, MAX and SUM all hand back something derived from actual
            // cell values, so they answer to the same column level policy a
            // row scan enforces. COUNT exposes no value
            let table_id = self.table_id.0;
            let mut builders = create_builders(&self.schema, 1);
            for (si, acc) in accs.into_iter().enumerate() {
                let exposes_value = !matches!(acc, MetaAcc::Count(_));
                let mut sv = acc.finish(self.specs[si].return_type)?;
                if exposes_value {
                    sv = expose_column_value(&self.ctx, table_id, self.specs[si].column_id, sv);
                }
                builders[si].push(&sv);
            }
            Ok(Some(ExecutionBatch::new(finalize_builders(builders))))
        })
    }
}
