//! Shared row fetch for search scan operators.
//!
//! FTS, vector and spatial results are registry ordinals. This resolves
//! them to row locators once, then serves per-row fetches: heap hits read
//! their page directly, columnar hits are pre-fetched in one batched pass
//! through the columnar scan machinery (segment decode, patch overlay and
//! MVCC visibility included), grouped so only segments holding hits open.
//!
//! Lake hits are pre-fetched the same way against the table's manifest,
//! grouped by data file so each one opens once and filtered through the
//! file's delete predicates so a search never returns a row a scan would
//! not. Which manifest is the time-travel qualifier's answer: a lake index
//! only ever gains postings and lake data files are immutable, so a hit
//! resolved against the manifest at a past version reads that version's
//! rows exactly.

use std::collections::{BTreeMap, HashMap, HashSet};
use std::sync::Arc;

use zyron_common::{Result, RowLocator, ZyronError};
use zyron_planner::logical::AsOfTarget;
use zyron_planner::logical::LogicalColumn;

use crate::batch::{decode_fixed_scalar, decode_varlen_scalar};
use crate::column::ScalarValue;
use crate::context::ExecutionContext;
use crate::operator::Operator;
use crate::operator::column_scan::ColumnScanOperator;

/// The commit LSN a heap or columnar hit is dated against under time travel.
///
/// Only a version qualifier maps to one. A timestamp needs the system
/// versioning predicate and a branch needs the copy-on-write overlay, and the
/// planner keeps both of those on the storage scan rather than routing a
/// search there, so neither reaches this.
pub(crate) fn heap_as_of_version(as_of: Option<&AsOfTarget>) -> Option<u64> {
    match as_of {
        Some(AsOfTarget::Version(v)) => Some(*v),
        _ => None,
    }
}

/// Visibility for one heap row: commit-LSN version visibility under time
/// travel, live-snapshot MVCC otherwise. Matches the columnar scan's oracle
/// so a search and a scan agree on which rows a version holds.
pub(crate) fn heap_row_visible(
    ctx: &Arc<ExecutionContext>,
    heap_version: Option<u64>,
    xmin: u64,
    xmax: u64,
) -> bool {
    match heap_version {
        Some(v) => ctx.snapshot.status_map().is_visible_at_version(xmin, xmax, v),
        None => ctx.snapshot.is_visible(xmin, xmax),
    }
}

/// Resolved locators plus pre-fetched columnar rows for one search result set
pub struct DocRowFetcher {
    /// positionally aligned with the search results
    pub locators: Vec<Option<RowLocator>>,
    /// projected values for every columnar-resident hit, in output column order
    columnar_rows: HashMap<(u64, u64), Vec<ScalarValue>>,
    /// projected values for every lake-resident hit, keyed by data file and
    /// ordinal, in output column order
    lake_rows: HashMap<(u64, u64), Vec<ScalarValue>>,
}

impl DocRowFetcher {
    /// Resolves every result DocId and pre-fetches the columnar hits.
    /// Requires ctx.doc_registry when any result exists.
    pub async fn prepare(
        ctx: &Arc<ExecutionContext>,
        table_id: zyron_catalog::TableId,
        output_columns: &[LogicalColumn],
        docs: &[u64],
        as_of: Option<&AsOfTarget>,
    ) -> Result<Self> {
        let mut locators = Vec::with_capacity(docs.len());
        if docs.is_empty() {
            return Ok(Self {
                locators,
                columnar_rows: HashMap::new(),
                lake_rows: HashMap::new(),
            });
        }
        let Some(reg) = &ctx.doc_registry else {
            return Err(ZyronError::Internal(
                "search result mapping requires the document registry".into(),
            ));
        };
        reg.map_docs(table_id.0, docs, &mut locators);

        let mut wanted: HashSet<(u64, u64)> = HashSet::new();
        let mut files: HashSet<u64> = HashSet::new();
        let mut lake_wanted: BTreeMap<u64, Vec<u64>> = BTreeMap::new();
        for loc in locators.iter().flatten() {
            match loc {
                RowLocator::Columnar { file_id, sys_rowid } => {
                    wanted.insert((*file_id, *sys_rowid));
                    files.insert(*file_id);
                }
                RowLocator::Lake { file_id, ordinal } => {
                    lake_wanted.entry(*file_id).or_default().push(*ordinal);
                }
                RowLocator::Heap { .. } => {}
            }
        }
        let columnar_rows =
            fetch_columnar_rows(ctx, table_id, output_columns, wanted, files, as_of).await?;
        let lake_rows = fetch_lake_rows(ctx, table_id, output_columns, lake_wanted, as_of)?;
        Ok(Self {
            locators,
            columnar_rows,
            lake_rows,
        })
    }

    /// Pre-fetches the columnar rows for a plain locator slice, for callers
    /// that track their own result ordering (the locators field stays empty)
    pub async fn prepare_columnar_only(
        ctx: &Arc<ExecutionContext>,
        table_id: zyron_catalog::TableId,
        output_columns: &[LogicalColumn],
        locs: &[RowLocator],
        as_of: Option<&AsOfTarget>,
    ) -> Result<Self> {
        let mut wanted: HashSet<(u64, u64)> = HashSet::new();
        let mut files: HashSet<u64> = HashSet::new();
        let mut lake_wanted: BTreeMap<u64, Vec<u64>> = BTreeMap::new();
        for loc in locs {
            match loc {
                RowLocator::Columnar { file_id, sys_rowid } => {
                    wanted.insert((*file_id, *sys_rowid));
                    files.insert(*file_id);
                }
                RowLocator::Lake { file_id, ordinal } => {
                    lake_wanted.entry(*file_id).or_default().push(*ordinal);
                }
                RowLocator::Heap { .. } => {}
            }
        }
        let columnar_rows =
            fetch_columnar_rows(ctx, table_id, output_columns, wanted, files, as_of).await?;
        let lake_rows = fetch_lake_rows(ctx, table_id, output_columns, lake_wanted, as_of)?;
        Ok(Self {
            locators: Vec::new(),
            columnar_rows,
            lake_rows,
        })
    }

    /// The pre-fetched projected values for a columnar hit. None when the
    /// row is not visible to this snapshot or was reclaimed.
    pub fn columnar_row(&self, file_id: u64, sys_rowid: u64) -> Option<&Vec<ScalarValue>> {
        self.columnar_rows.get(&(file_id, sys_rowid))
    }

    /// The pre-fetched projected values for a lake hit. None when the data
    /// file is no longer live at the newest version, or when a delete
    /// predicate removed the row, both of which a scan would also skip
    pub fn lake_row(&self, file_id: u64, ordinal: u64) -> Option<&Vec<ScalarValue>> {
        self.lake_rows.get(&(file_id, ordinal))
    }
}

/// Reads the addressed rows of a lake table, one open per data file.
///
/// A row whose file has left the manifest, or that a delete predicate
/// removed, is absent from the result rather than returned, which is what
/// keeps a search from resurrecting a row a scan cannot see. The time-travel
/// qualifier picks which manifest that is, so the same rule answers a past
/// version: a row appended since is in no file that version names, and a row
/// deleted since is in a file that version's predicates do not remove
fn fetch_lake_rows(
    ctx: &Arc<ExecutionContext>,
    table_id: zyron_catalog::TableId,
    output_columns: &[LogicalColumn],
    wanted: BTreeMap<u64, Vec<u64>>,
    as_of: Option<&AsOfTarget>,
) -> Result<HashMap<(u64, u64), Vec<ScalarValue>>> {
    let mut out = HashMap::new();
    if wanted.is_empty() {
        return Ok(out);
    }
    let paths = zyron_lake::LakePaths::new(ctx.disk_manager.data_dir(), table_id.0);
    // A branch is an alternate head over the same immutable files, so
    // reading one is opening its log instead of main's. A document whose
    // file that head does not name resolves to nothing below, which is what
    // keeps a branch's rows out of a main search and main's out of a
    // branch's without either index knowing branches exist
    let head = crate::operator::lake_scan::effective_head(ctx, as_of);
    let table_name = ctx
        .get_table_entry(table_id)
        .map(|t| t.name.clone())
        .unwrap_or_default();
    let log = crate::operator::lake_scan::open_lake_head(&paths, &table_name, head)?;
    let version = match as_of {
        None | Some(AsOfTarget::Branch(_)) => log.latest_version(),
        Some(AsOfTarget::Version(v)) => {
            zyron_lake::resolve_version(&log, zyron_lake::TimeTravelSpec::Version(*v))?
        }
        Some(AsOfTarget::Timestamp(us)) => {
            zyron_lake::resolve_version(&log, zyron_lake::TimeTravelSpec::Timestamp(*us))?
        }
    };
    let manifest = log.manifest_at(version)?;

    for (partition_id, mut ordinals) in wanted {
        let Some(entry) = manifest.entry_for(partition_id) else {
            // The file was rewritten or dropped, so its rows are addressed
            // by a different file now and this locator is dead
            continue;
        };
        ordinals.sort_unstable();
        ordinals.dedup();
        let reader = zyron_lake::LakeFileReader::open(&paths, partition_id)?;
        let row_count = reader.row_count();
        let keep = reader.delete_survivors(&manifest.schema, &manifest, entry)?;

        let mut decoded = Vec::with_capacity(output_columns.len());
        for col in output_columns {
            let lake_col = manifest
                .schema
                .column_by_id(col.column_id.0 as u32)
                .ok_or_else(|| {
                    ZyronError::ExecutionError(format!(
                        "column \"{}\" is not in the lake schema",
                        col.name
                    ))
                })?;
            decoded.push((
                col.type_id,
                lake_col.physical_type_id().fixed_size().unwrap_or(0),
                reader.read_column(lake_col)?,
            ));
        }

        for ordinal in ordinals {
            let row = ordinal as usize;
            if row >= row_count || keep[row / 8] & (1 << (row % 8)) == 0 {
                continue;
            }
            let values: Vec<ScalarValue> = decoded
                .iter()
                .map(|(type_id, value_size, column)| match column.cell(row) {
                    None => ScalarValue::Null,
                    Some(cell) if *value_size == 0 => decode_varlen_scalar(*type_id, cell),
                    Some(cell) => decode_fixed_scalar(*type_id, cell),
                })
                .collect();
            out.insert((partition_id, ordinal), values);
        }
    }
    Ok(out)
}

/// Batched fetch of the wanted columnar rows through the columnar scan
/// machinery, segment decode, patch overlay and MVCC visibility included
async fn fetch_columnar_rows(
    ctx: &Arc<ExecutionContext>,
    table_id: zyron_catalog::TableId,
    output_columns: &[LogicalColumn],
    wanted: HashSet<(u64, u64)>,
    files: HashSet<u64>,
    as_of: Option<&AsOfTarget>,
) -> Result<HashMap<(u64, u64), Vec<ScalarValue>>> {
    let mut columnar_rows = HashMap::with_capacity(wanted.len());
    if wanted.is_empty() {
        return Ok(columnar_rows);
    }
    // A version dates folded rows by commit LSN, matching the hybrid scan
    let as_of_version = match as_of {
        Some(AsOfTarget::Version(v)) => Some(*v),
        _ => None,
    };
    let mut op =
        ColumnScanOperator::new_for_dml(ctx.clone(), table_id, output_columns.to_vec(), None)?
            .with_file_filter(files)
            .with_as_of(as_of_version);
    while let Some(eb) = op.next().await? {
        let Some(locs) = eb.locators.as_ref() else {
            continue;
        };
        for (row, loc) in locs.iter().enumerate() {
            let RowLocator::Columnar { file_id, sys_rowid } = loc else {
                continue;
            };
            let key = (*file_id, *sys_rowid);
            if !wanted.contains(&key) {
                continue;
            }
            let vals: Vec<ScalarValue> =
                eb.batch.columns.iter().map(|c| c.get_scalar(row)).collect();
            columnar_rows.insert(key, vals);
        }
    }
    Ok(columnar_rows)
}
