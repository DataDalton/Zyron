//! Secondary indexes on lake tables.
//!
//! An index is a lake artifact, not a side structure. Its entries live in
//! sorted immutable `.zyr` files named in the same manifest as the data
//! files, committed through the same transaction log, pruned by the same
//! statistics and read by the same reader. That is what makes an index
//! correct under the format's own features: a version-as-of read sees the
//! index the manifest of that version names, a clone shares index files
//! without copying them, and two writers resolve index commits under the
//! optimistic concurrency rules that already govern data commits.
//!
//! Two properties of the format are what rule out a mutable tree keyed by
//! row position. Such a tree knows only the newest state, so it cannot
//! answer a past version and every time-travel query has to bypass it. And
//! every clustering or compaction pass rewrites the files its entries point
//! into, which costs one tree mutation per moved row per index where a file
//! set costs one sequential write.
//!
//! ## Layout
//!
//! One index file holds, per row of the data files it covers, the index's
//! key columns followed by the row's address: the data file's partition id
//! and the row's ordinal within it. Rows are sorted by the key, so the
//! file's own min and max bound a contiguous key range and a probe reads
//! one file rather than all of them.
//!
//! ## Staleness cannot produce a wrong answer
//!
//! An index file records which data partitions it covers. An index answers
//! a query only when its live files cover every live data partition, so a
//! rewrite that has not been indexed yet makes the index decline and the
//! query falls back to a scan. Coverage is compared as sets, never
//! assumed, which means an index can be behind but can never be wrong.

use std::collections::{BTreeMap, HashSet};

use zyron_common::ZyronError;

use crate::cells::compare_cells;
use crate::manifest::{ManifestFile, PartitionEntry};
use crate::paths::LakePaths;
use crate::predicate::{LakeValue, PruneDecision};
use crate::reader::{LakeFileReader, ZoneVerdict};
use crate::schema::{LakeColumn, LakeSchema};
use crate::writer::{ColumnData, WriteRequest, write_data_file};

/// A declared index on a lake table.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LakeIndexSpec {
    /// Stable identity, allocated from the manifest and never reused
    pub index_id: u32,
    pub name: String,
    /// Key columns in declaration order, at least one
    pub column_ids: Vec<u32>,
    /// Whether the index also enforces uniqueness over its key
    pub unique: bool,
}

impl LakeIndexSpec {
    pub fn validate(&self) -> Result<(), ZyronError> {
        if self.column_ids.is_empty() {
            return Err(ZyronError::Internal(format!(
                "index \"{}\" names no columns",
                self.name
            )));
        }
        let mut seen = HashSet::with_capacity(self.column_ids.len());
        for id in &self.column_ids {
            if !seen.insert(*id) {
                return Err(ZyronError::Internal(format!(
                    "index \"{}\" names column {} twice",
                    self.name, id
                )));
            }
        }
        Ok(())
    }
}

/// One index file plus the data partitions its entries address.
#[derive(Debug, Clone, PartialEq)]
pub struct IndexFileEntry {
    pub index_id: u32,
    /// Data partitions this file holds entries for, ascending. Coverage is
    /// recorded rather than derived because deriving it would mean reading
    /// every index file to answer whether the index may be used at all
    pub covers: Vec<u64>,
    /// The file itself, carrying the key columns' bounds so a probe prunes
    /// index files exactly the way a scan prunes data files
    pub file: PartitionEntry,
}

/// Column id the address pair takes inside an index file. Key columns take
/// the leading ids, so the address always follows them
const ADDRESS_PARTITION_OFFSET: u32 = 0;
const ADDRESS_ORDINAL_OFFSET: u32 = 1;

/// Entries one index file holds.
///
/// This is the quantity a probe actually pays. Decoding is whole-segment,
/// so the cost of answering one key is the size of the file that key lands
/// in, not the size of the index. Splitting a sorted run into files of this
/// size gives every file a disjoint key range, which lets the manifest's
/// own min and max pruning pick exactly one before anything is opened.
///
/// A single file holding the whole index would make a probe decode every
/// entry in the table, which is more work than scanning the data
pub const ENTRIES_PER_INDEX_FILE: usize = 8192;

/// The schema one index's files are written under: the key columns in
/// declaration order, then the two address columns.
///
/// Key columns keep their table type so the bounds, blooms and zone maps
/// an index file carries compare the same way the data files' do. Their
/// ids are renumbered to their position, because an index file's schema is
/// its own and the probe addresses columns by position
pub fn index_schema(table: &LakeSchema, spec: &LakeIndexSpec) -> Result<LakeSchema, ZyronError> {
    spec.validate()?;
    let mut columns = Vec::with_capacity(spec.column_ids.len() + 2);
    for (position, id) in spec.column_ids.iter().enumerate() {
        let source = table.column_by_id(*id).ok_or_else(|| {
            ZyronError::Internal(format!(
                "index \"{}\" names column {}, which is not in the schema",
                spec.name, id
            ))
        })?;
        columns.push(LakeColumn {
            id: position as u32,
            name: source.name.clone(),
            type_id: source.type_id,
            // A key column is nullable inside the index whatever the table
            // says, because an index entry is written for every row
            nullable: true,
            fractional_digits: source.fractional_digits,
            tz_offset_secs: source.tz_offset_secs,
            max_length: source.max_length,
            default_expr: None,
        });
    }
    let base = spec.column_ids.len() as u32;
    for (offset, name) in [
        (ADDRESS_PARTITION_OFFSET, "__partition"),
        (ADDRESS_ORDINAL_OFFSET, "__ordinal"),
    ] {
        columns.push(LakeColumn {
            id: base + offset,
            name: name.into(),
            type_id: zyron_common::TypeId::Int64,
            nullable: false,
            fractional_digits: None,
            tz_offset_secs: None,
            max_length: None,
            default_expr: None,
        });
    }
    LakeSchema::new(1, columns)
}

/// Where one indexed row lives.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct RowAddress {
    pub partition_id: u64,
    pub ordinal: u64,
}

/// Index entries accumulated for one commit, before they are written.
///
/// Rows arrive grouped by data file and are sorted by key once at the end,
/// because a sort over the whole batch is one pass where a sorted insert
/// per row would be one shift per row
pub struct IndexBatch {
    key_columns: usize,
    /// Key cells per row, one inner vec per key column
    keys: Vec<Vec<Option<Vec<u8>>>>,
    addresses: Vec<RowAddress>,
    covers: HashSet<u64>,
}

impl IndexBatch {
    pub fn new(spec: &LakeIndexSpec) -> Self {
        Self {
            key_columns: spec.column_ids.len(),
            keys: vec![Vec::new(); spec.column_ids.len()],
            addresses: Vec::new(),
            covers: HashSet::new(),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.addresses.is_empty()
    }

    pub fn len(&self) -> usize {
        self.addresses.len()
    }

    /// Records that this batch accounts for a data partition, including
    /// one whose rows were all deleted. Coverage is about which partitions
    /// were examined, not which produced entries, so an emptied file still
    /// has to be declared or the index would read as incomplete forever
    pub fn cover(&mut self, partition_id: u64) {
        self.covers.insert(partition_id);
    }

    /// Adds one row's entry. `cells` are the key columns in declaration
    /// order, NULL included, so the index holds an entry for every row
    pub fn push(&mut self, cells: &[Option<&[u8]>], address: RowAddress) {
        debug_assert_eq!(cells.len(), self.key_columns);
        for (column, cell) in self.keys.iter_mut().zip(cells) {
            column.push(cell.map(|c| c.to_vec()));
        }
        self.covers.insert(address.partition_id);
        self.addresses.push(address);
    }
}

/// Writes an accumulated batch as one or more index files and returns their
/// entries.
///
/// The batch is sorted by key once, then split into files of
/// `ENTRIES_PER_INDEX_FILE`. Because the split follows the sort, each file
/// holds a contiguous key range and the ranges are disjoint, so the
/// manifest's own bounds pick exactly one file for a point probe and a
/// contiguous span for a range one. That is what keeps the cost of a probe
/// proportional to one file rather than to the whole index.
///
/// Each write goes through the same path as a data file, so the index
/// inherits exact statistics, zone maps, blooms and encoding selection
pub fn write_index_files(
    paths: &LakePaths,
    table: &LakeSchema,
    spec: &LakeIndexSpec,
    table_id: u64,
    batch: IndexBatch,
    next_partition: &mut dyn FnMut() -> u64,
) -> Result<Vec<IndexFileEntry>, ZyronError> {
    let schema = index_schema(table, spec)?;
    let key_columns = batch.key_columns;
    let rows = batch.addresses.len();

    // Partitions this batch accounted for that produced no entry, because
    // every row of them was deleted. They still have to be declared or the
    // index reads as incomplete forever, so they ride on the first file
    let mut entryless: Vec<u64> = batch
        .covers
        .iter()
        .copied()
        .filter(|p| !batch.addresses.iter().any(|a| a.partition_id == *p))
        .collect();
    entryless.sort_unstable();

    if rows == 0 {
        // An empty index still has to name what it covers, so the file set
        // is one empty file rather than nothing at all
        let file = write_one_index_file(
            paths,
            &schema,
            spec,
            next_partition(),
            table_id,
            key_columns,
            &batch.keys,
            &batch.addresses,
            &[],
        )?;
        return Ok(vec![IndexFileEntry {
            index_id: spec.index_id,
            covers: entryless,
            file,
        }]);
    }

    // One sort for the whole batch, on the same normalized key the file
    // writer orders by, so the split boundaries agree with the order inside
    // each file
    let physical: Vec<zyron_common::TypeId> = schema.columns[..key_columns]
        .iter()
        .map(|c| c.physical_type_id())
        .collect();
    let mut order: Vec<usize> = (0..rows).collect();
    order.sort_by(|&a, &b| {
        sort_key_of(&batch.keys, &physical, a).cmp(&sort_key_of(&batch.keys, &physical, b))
    });

    // Split on sort-key run boundaries rather than at a fixed count. The
    // sort key is truncated for strings and wide integers, so entries
    // sharing one can be in any order among themselves. Cutting through
    // such a run would put a value in one file and a neighbour that
    // compares below it in the next, and the two files' bounds would then
    // overlap in the column's real order, which is exactly what stops the
    // manifest pruning a probe to one file.
    //
    // A run longer than the target makes its file bigger, which is correct
    // and unavoidable: those entries have to be found together
    let mut boundaries: Vec<usize> = Vec::new();
    let mut start = 0usize;
    while start < order.len() {
        let mut end = (start + ENTRIES_PER_INDEX_FILE).min(order.len());
        if end < order.len() {
            let key = sort_key_of(&batch.keys, &physical, order[end - 1]);
            while end < order.len() && sort_key_of(&batch.keys, &physical, order[end]) == key {
                end += 1;
            }
        }
        boundaries.push(end);
        start = end;
    }

    let mut out = Vec::with_capacity(boundaries.len());
    let mut chunk_start = 0usize;
    for (chunk_index, chunk_end) in boundaries.into_iter().enumerate() {
        let chunk = &order[chunk_start..chunk_end];
        chunk_start = chunk_end;
        let mut keys: Vec<Vec<Option<Vec<u8>>>> = Vec::with_capacity(key_columns);
        for column in &batch.keys {
            keys.push(chunk.iter().map(|r| column[*r].clone()).collect());
        }
        let addresses: Vec<RowAddress> = chunk.iter().map(|r| batch.addresses[*r]).collect();
        let mut covers: Vec<u64> = addresses.iter().map(|a| a.partition_id).collect();
        if chunk_index == 0 {
            covers.extend(entryless.iter().copied());
        }
        covers.sort_unstable();
        covers.dedup();
        let file = write_one_index_file(
            paths,
            &schema,
            spec,
            next_partition(),
            table_id,
            key_columns,
            &keys,
            &addresses,
            &covers,
        )?;
        out.push(IndexFileEntry {
            index_id: spec.index_id,
            covers,
            file,
        });
    }
    Ok(out)
}

/// The normalized sort key of one entry, the same one the file writer
/// orders by, so a split made here agrees with the order inside each file
fn sort_key_of(
    keys: &[Vec<Option<Vec<u8>>>],
    physical: &[zyron_common::TypeId],
    row: usize,
) -> Vec<(bool, u64)> {
    keys.iter()
        .zip(physical)
        .map(|(column, physical)| match &column[row] {
            // Nulls sort last, matching the writer
            None => (true, 0),
            Some(cell) => (
                false,
                zyron_common::curve::normalize_component(*physical, cell),
            ),
        })
        .collect()
}

/// Writes one index file from already-chunked cells.
#[allow(clippy::too_many_arguments)]
fn write_one_index_file(
    paths: &LakePaths,
    schema: &LakeSchema,
    spec: &LakeIndexSpec,
    partition_id: u64,
    table_id: u64,
    key_columns: usize,
    keys: &[Vec<Option<Vec<u8>>>],
    addresses: &[RowAddress],
    _covers: &[u64],
) -> Result<PartitionEntry, ZyronError> {
    let mut columns: Vec<ColumnData> = Vec::with_capacity(schema.columns.len());
    for (position, cells) in keys.iter().enumerate() {
        columns.push(ColumnData {
            column_id: position as u32,
            cells: cells.clone(),
        });
    }
    let base = key_columns as u32;
    columns.push(ColumnData {
        column_id: base + ADDRESS_PARTITION_OFFSET,
        cells: addresses
            .iter()
            .map(|a| Some((a.partition_id as i64).to_le_bytes().to_vec()))
            .collect(),
    });
    columns.push(ColumnData {
        column_id: base + ADDRESS_ORDINAL_OFFSET,
        cells: addresses
            .iter()
            .map(|a| Some((a.ordinal as i64).to_le_bytes().to_vec()))
            .collect(),
    });

    // Ascending on the key, never a curve. A curve spreads a key across
    // the file so a probe could not bisect, and the index exists precisely
    // to answer one key
    let sort_keys: Vec<u32> = (0..key_columns as u32).collect();
    let sort_strategies = vec![crate::manifest::ClusterStrategy::RangePartition; sort_keys.len()];
    write_data_file(
        paths,
        schema,
        &WriteRequest {
            partition_id,
            columns: &columns,
            sort_keys: &sort_keys,
            sort_strategies: &sort_strategies,
            cluster_spec_id: 0,
            table_id,
            // The leading key is what a probe looks up, and a bloom on it
            // rejects a file the bounds admit for a value it does not hold
            bloom_columns: &[0],
            index_id: Some(spec.index_id),
        },
    )
}

/// Builds the index entries for one data file, reading only the key
/// columns it needs.
///
/// Rows removed by a delete predicate are skipped, so an index never
/// points at a row a reader would filter out
pub fn entries_for_file(
    paths: &LakePaths,
    manifest: &ManifestFile,
    spec: &LakeIndexSpec,
    entry: &PartitionEntry,
    batch: &mut IndexBatch,
) -> Result<(), ZyronError> {
    batch.cover(entry.partition_id);
    let reader = LakeFileReader::open(paths, entry.partition_id)?;
    let rows = reader.row_count();
    if rows == 0 {
        return Ok(());
    }
    let keep = reader.delete_survivors(&manifest.schema, manifest, entry)?;
    let mut columns = Vec::with_capacity(spec.column_ids.len());
    for id in &spec.column_ids {
        let column = manifest.schema.column_by_id(*id).ok_or_else(|| {
            ZyronError::Internal(format!(
                "index \"{}\" names column {}, which is not in the schema",
                spec.name, id
            ))
        })?;
        columns.push(reader.read_column(column)?);
    }
    let mut cells: Vec<Option<&[u8]>> = Vec::with_capacity(columns.len());
    for row in 0..rows {
        if keep[row / 8] & (1 << (row % 8)) == 0 {
            continue;
        }
        cells.clear();
        cells.extend(columns.iter().map(|c| c.cell(row)));
        batch.push(
            &cells,
            RowAddress {
                partition_id: entry.partition_id,
                ordinal: row as u64,
            },
        );
    }
    Ok(())
}

/// Whether an index's live files account for every live data partition.
///
/// This is the whole staleness contract. A probe runs only when it holds,
/// so an index that is behind declines and the caller scans, and an index
/// can never answer with fewer rows than the table has
pub fn covers_table(manifest: &ManifestFile, index_id: u32) -> bool {
    let mut covered: HashSet<u64> = HashSet::new();
    for file in &manifest.index_files {
        if file.index_id == index_id {
            covered.extend(file.covers.iter().copied());
        }
    }
    manifest
        .entries
        .iter()
        .all(|entry| covered.contains(&entry.partition_id))
}

/// What a probe had to read, so a caller can prove the statistics did
/// their job rather than assume it.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct IndexProbeStats {
    /// Live index files for this index
    pub files_considered: usize,
    /// Files whose key bounds and bloom admitted the probe
    pub files_admitted: usize,
    /// Files opened
    pub files_opened: usize,
    /// Files opened whose zone maps held no candidate, so no key column of
    /// them was decoded
    pub files_zone_rejected: usize,
    /// Entries the bisection compared
    pub entries_examined: u64,
    /// Entries returned, before the caller reads any row
    pub entries_matched: u64,
    /// Entries dropped because the data file they addressed is no longer
    /// live. A rewrite leaves them behind and the fresh entries for the
    /// rewritten rows arrive in the same commit, so dropping them here is
    /// what keeps a probe from addressing a file that is gone
    pub entries_stale: u64,
}

/// Looks one key up in an index, returning the rows that carry it.
///
/// Index files are pruned by their key bounds first, then by the leading
/// key's bloom, then by zone maps, and only what survives is decoded and
/// bisected. Addresses into data files the manifest no longer lists are
/// dropped, so a probe never names a file a reader cannot open
pub fn probe_equal(
    paths: &LakePaths,
    manifest: &ManifestFile,
    spec: &LakeIndexSpec,
    key: &[Option<&[u8]>],
) -> Result<(Vec<RowAddress>, IndexProbeStats), ZyronError> {
    let mut stats = IndexProbeStats::default();
    let mut out = Vec::new();
    if key.len() != spec.column_ids.len() {
        return Err(ZyronError::Internal(format!(
            "index \"{}\" takes {} key columns, probed with {}",
            spec.name,
            spec.column_ids.len(),
            key.len()
        )));
    }
    // A NULL component matches nothing under an equality probe, the same
    // rule the unique check applies
    let Some(leading) = key[0] else {
        return Ok((out, stats));
    };
    let schema = index_schema(&manifest.schema, spec)?;
    let leading_column = &schema.columns[0];
    let physical = leading_column.physical_type_id();
    let live: HashSet<u64> = manifest.entries.iter().map(|e| e.partition_id).collect();
    let bound = crate::cells::cell_to_value(physical, leading);

    for file in &manifest.index_files {
        if file.index_id != spec.index_id {
            continue;
        }
        stats.files_considered += 1;
        // The file's own key bounds, then its bloom. Both answer from the
        // manifest with no IO at all
        if let Some(value) = &bound
            && !admits_value(&schema, &file.file, 0, value)
        {
            continue;
        }
        stats.files_admitted += 1;

        let reader =
            LakeFileReader::open_path(&paths.index_file(spec.index_id, file.file.partition_id))?;
        stats.files_opened += 1;
        let rows = reader.row_count();
        if rows == 0 {
            continue;
        }
        let span = match reader.zone_span_for_cells(leading_column, &[leading])? {
            ZoneVerdict::NoMatch => {
                stats.files_zone_rejected += 1;
                continue;
            }
            ZoneVerdict::Span(span) => span,
            ZoneVerdict::Undecided => 0..rows,
        };

        // The leading key is decoded over the span the zone maps left, and
        // the bisection runs inside it. Everything else is decoded only
        // over the run the bisection selected, which is what keeps the cost
        // of one key proportional to the answer rather than to the file
        let leading_decoded = reader.read_column_range(leading_column, span.start, span.end)?;
        let range = leading_decoded.sort_key_range_in(leading, span);
        if range.is_empty() {
            continue;
        }
        let mut key_columns = Vec::with_capacity(spec.column_ids.len());
        key_columns.push(leading_decoded);
        for position in 1..spec.column_ids.len() {
            key_columns.push(reader.read_column_range(
                &schema.columns[position],
                range.start,
                range.end,
            )?);
        }
        let base = spec.column_ids.len();
        let partitions = reader.read_column_range(&schema.columns[base], range.start, range.end)?;
        let ordinals =
            reader.read_column_range(&schema.columns[base + 1], range.start, range.end)?;

        for row in range {
            stats.entries_examined += 1;
            if !matches_key(&key_columns, row, key, physical) {
                continue;
            }
            let (Some(partition), Some(ordinal)) = (partitions.cell(row), ordinals.cell(row))
            else {
                continue;
            };
            let address = RowAddress {
                partition_id: read_i64(partition) as u64,
                ordinal: read_i64(ordinal) as u64,
            };
            if !live.contains(&address.partition_id) {
                stats.entries_stale += 1;
                continue;
            }
            stats.entries_matched += 1;
            out.push(address);
        }
    }
    out.sort_unstable();
    Ok((out, stats))
}

/// One end of a range probe, or its absence for an open side.
#[derive(Debug, Clone, PartialEq)]
pub struct RangeBound {
    pub value: LakeValue,
    /// True for `>=` and `<=`, false for `>` and `<`
    pub inclusive: bool,
}

/// Looks a key range up in an index, returning the rows it addresses.
///
/// Index files hold disjoint key ranges, so the manifest's own bounds pick
/// the contiguous span of files the range touches and the rest are never
/// opened. Inside a file the leading key ascends, so the matching run is
/// found by two bisections rather than by reading the file.
///
/// Only the leading key column is bounded. A trailing component of a
/// composite key does not order the file, so bounding it here would drop
/// rows the exact filter is responsible for
pub fn probe_range(
    paths: &LakePaths,
    manifest: &ManifestFile,
    spec: &LakeIndexSpec,
    low: Option<&RangeBound>,
    high: Option<&RangeBound>,
) -> Result<(Vec<RowAddress>, IndexProbeStats), ZyronError> {
    let mut stats = IndexProbeStats::default();
    let mut out = Vec::new();
    if low.is_none() && high.is_none() {
        return Ok((out, stats));
    }
    let schema = index_schema(&manifest.schema, spec)?;
    let leading_column = &schema.columns[0];
    let physical = leading_column.physical_type_id();
    let width = physical.fixed_size().unwrap_or(0);
    let live: HashSet<u64> = manifest.entries.iter().map(|e| e.partition_id).collect();

    // The bounds as stored cells, so comparisons run in the column's own
    // order rather than on the value enum
    let low_cell = match low {
        Some(bound) => match crate::cells::value_to_cell(physical, width, &bound.value) {
            Some(cell) => Some((cell.as_slice().to_vec(), bound.inclusive)),
            // A bound with no stored form cannot narrow anything, and the
            // exact filter answers it
            None => return Ok((out, stats)),
        },
        None => None,
    };
    let high_cell = match high {
        Some(bound) => match crate::cells::value_to_cell(physical, width, &bound.value) {
            Some(cell) => Some((cell.as_slice().to_vec(), bound.inclusive)),
            None => return Ok((out, stats)),
        },
        None => None,
    };

    for file in &manifest.index_files {
        if file.index_id != spec.index_id {
            continue;
        }
        stats.files_considered += 1;
        if !admits_range(&schema, &file.file, low, high) {
            continue;
        }
        stats.files_admitted += 1;

        let reader =
            LakeFileReader::open_path(&paths.index_file(spec.index_id, file.file.partition_id))?;
        stats.files_opened += 1;
        let rows = reader.row_count();
        if rows == 0 {
            continue;
        }
        let key_column = reader.read_column(leading_column)?;
        let base = spec.column_ids.len();
        let partitions = reader.read_column(&schema.columns[base])?;
        let ordinals = reader.read_column(&schema.columns[base + 1])?;

        // Two bisections bound the run. The sort key is truncated for
        // strings and wide integers, so the ends are widened by one run and
        // the exact comparison below decides them
        let start = match &low_cell {
            Some((cell, _)) => key_column.sort_key_range_in(cell, 0..rows).start,
            None => 0,
        };
        let end = match &high_cell {
            Some((cell, _)) => key_column.sort_key_range_in(cell, start..rows).end,
            None => rows,
        };

        for row in start..end {
            stats.entries_examined += 1;
            let Some(cell) = key_column.cell(row) else {
                continue;
            };
            if !in_range(physical, cell, &low_cell, &high_cell) {
                continue;
            }
            let (Some(partition), Some(ordinal)) = (partitions.cell(row), ordinals.cell(row))
            else {
                continue;
            };
            let address = RowAddress {
                partition_id: read_i64(partition) as u64,
                ordinal: read_i64(ordinal) as u64,
            };
            if !live.contains(&address.partition_id) {
                stats.entries_stale += 1;
                continue;
            }
            stats.entries_matched += 1;
            out.push(address);
        }
    }
    out.sort_unstable();
    Ok((out, stats))
}

/// Whether one stored cell falls inside the probed bounds.
fn in_range(
    physical: zyron_common::TypeId,
    cell: &[u8],
    low: &Option<(Vec<u8>, bool)>,
    high: &Option<(Vec<u8>, bool)>,
) -> bool {
    if let Some((bound, inclusive)) = low {
        let ord = compare_cells(physical, cell, bound);
        if ord.is_lt() || (ord.is_eq() && !inclusive) {
            return false;
        }
    }
    if let Some((bound, inclusive)) = high {
        let ord = compare_cells(physical, cell, bound);
        if ord.is_gt() || (ord.is_eq() && !inclusive) {
            return false;
        }
    }
    true
}

/// Whether one index file's key bounds overlap the probed range.
fn admits_range(
    index_schema: &LakeSchema,
    entry: &PartitionEntry,
    low: Option<&RangeBound>,
    high: Option<&RangeBound>,
) -> bool {
    let stats = crate::manifest::FileStats::new(entry, index_schema);
    if let Some(bound) = low {
        let op = if bound.inclusive {
            crate::predicate::CompareOp::GtEq
        } else {
            crate::predicate::CompareOp::Gt
        };
        let predicate = crate::predicate::LakePredicate::Compare {
            column_id: 0,
            op,
            value: bound.value.clone(),
        };
        if predicate.prune(&stats) == PruneDecision::CannotMatch {
            return false;
        }
    }
    if let Some(bound) = high {
        let op = if bound.inclusive {
            crate::predicate::CompareOp::LtEq
        } else {
            crate::predicate::CompareOp::Lt
        };
        let predicate = crate::predicate::LakePredicate::Compare {
            column_id: 0,
            op,
            value: bound.value.clone(),
        };
        if predicate.prune(&stats) == PruneDecision::CannotMatch {
            return false;
        }
    }
    true
}

/// Whether one index file's statistics admit a key value on a column.
///
/// The statistics are read against the index's own schema, not the
/// table's, because an index file's columns are the index's key columns
/// renumbered. Bounds answer first and the leading key's bloom answers
/// after them, both from the manifest with no IO
fn admits_value(
    index_schema: &LakeSchema,
    entry: &PartitionEntry,
    column_id: u32,
    value: &LakeValue,
) -> bool {
    let predicate = crate::predicate::LakePredicate::Compare {
        column_id,
        op: crate::predicate::CompareOp::Eq,
        value: value.clone(),
    };
    predicate.prune(&crate::manifest::FileStats::new(entry, index_schema))
        != PruneDecision::CannotMatch
}

/// Whether one index row carries exactly the probed key.
fn matches_key(
    columns: &[crate::reader::DecodedColumn],
    row: usize,
    key: &[Option<&[u8]>],
    leading_physical: zyron_common::TypeId,
) -> bool {
    for (position, want) in key.iter().enumerate() {
        let have = columns[position].cell(row);
        match (have, want) {
            (None, None) => {}
            (Some(a), Some(b)) => {
                let physical = if position == 0 {
                    leading_physical
                } else {
                    // Trailing components compare under their own column's
                    // type, which the decoded column already carries
                    columns[position].physical_type_id()
                };
                if !compare_cells(physical, a, b).is_eq() {
                    return false;
                }
            }
            _ => return false,
        }
    }
    true
}

/// The stored bytes an index key value takes, under the index column's own
/// physical type. None for a constant with no representable stored form,
/// which cannot be looked up and is left to the exact filter
pub fn value_to_index_cell(
    physical: zyron_common::TypeId,
    width: usize,
    value: &LakeValue,
) -> Option<Vec<u8>> {
    crate::cells::value_to_cell(physical, width, value).map(|c| c.as_slice().to_vec())
}

fn read_i64(cell: &[u8]) -> i64 {
    let mut a = [0u8; 8];
    let n = cell.len().min(8);
    a[..n].copy_from_slice(&cell[..n]);
    i64::from_le_bytes(a)
}

/// Builds the entries a freshly written data file contributes to one
/// index, addressing the rows the write just placed.
///
/// `order[ordinal]` is the input row at that ordinal, which the writer
/// returns, so nothing is read back. Every row of a new file is live: a
/// delete predicate recorded later attaches to the file and the probe
/// filters through it then
pub fn entries_for_written_file(
    schema: &LakeSchema,
    spec: &LakeIndexSpec,
    partition_id: u64,
    columns: &[ColumnData],
    order: &[usize],
    batch: &mut IndexBatch,
) -> Result<(), ZyronError> {
    batch.cover(partition_id);
    let mut key_columns = Vec::with_capacity(spec.column_ids.len());
    for id in &spec.column_ids {
        if schema.column_by_id(*id).is_none() {
            return Err(ZyronError::Internal(format!(
                "index \"{}\" names column {}, which is not in the schema",
                spec.name, id
            )));
        }
        let data = columns.iter().find(|c| c.column_id == *id).ok_or_else(|| {
            ZyronError::Internal(format!(
                "index \"{}\" needs column {}, which the batch does not carry",
                spec.name, id
            ))
        })?;
        key_columns.push(data);
    }
    let mut cells: Vec<Option<&[u8]>> = Vec::with_capacity(key_columns.len());
    for (ordinal, row) in order.iter().enumerate() {
        cells.clear();
        cells.extend(key_columns.iter().map(|c| c.cells[*row].as_deref()));
        batch.push(
            &cells,
            RowAddress {
                partition_id,
                ordinal: ordinal as u64,
            },
        );
    }
    Ok(())
}

/// The log entries that bring every declared index up to date with one
/// freshly written data file.
///
/// Called inside a commit closure, so the entries land in the same version
/// as the data file itself and the index is never observable as behind
pub fn delta_entries_for_written_file(
    paths: &LakePaths,
    base: &ManifestFile,
    partition_id: u64,
    table_id: u64,
    columns: &[ColumnData],
    order: &[usize],
    next_partition: &mut dyn FnMut() -> u64,
) -> Result<Vec<crate::transaction_log::LogEntry>, ZyronError> {
    let mut out = Vec::new();
    for spec in &base.indexes {
        let mut batch = IndexBatch::new(spec);
        entries_for_written_file(&base.schema, spec, partition_id, columns, order, &mut batch)?;
        for file in write_index_files(paths, &base.schema, spec, table_id, batch, next_partition)? {
            out.push(crate::transaction_log::LogEntry::AddIndexFile(file));
        }
    }
    Ok(out)
}

/// The log entries that build one index over every live data file.
///
/// The whole table is read once and emitted as one sorted, range-split file
/// set, which is the layout a probe prunes best: disjoint ranges across
/// files, ascending within one
pub fn build_entries(
    paths: &LakePaths,
    base: &ManifestFile,
    spec: &LakeIndexSpec,
    table_id: u64,
    next_partition: &mut dyn FnMut() -> u64,
) -> Result<Vec<crate::transaction_log::LogEntry>, ZyronError> {
    let mut batch = IndexBatch::new(spec);
    for entry in &base.entries {
        entries_for_file(paths, base, spec, entry, &mut batch)?;
    }
    Ok(
        write_index_files(paths, &base.schema, spec, table_id, batch, next_partition)?
            .into_iter()
            .map(crate::transaction_log::LogEntry::AddIndexFile)
            .collect(),
    )
}

/// Log entries dropping every index file whose coverage names a partition
/// that is going away, so a rewrite never leaves an entry addressing a
/// file the manifest no longer lists.
///
/// A file covering both a removed and a surviving partition is dropped
/// too, and the surviving partition is re-indexed by the caller. Keeping
/// it would leave the index reading as complete while half its entries
/// pointed at a file that is gone
pub fn stale_index_files(
    base: &ManifestFile,
    removed: &[u64],
) -> (Vec<crate::transaction_log::LogEntry>, Vec<u64>) {
    let removed: HashSet<u64> = removed.iter().copied().collect();
    let mut entries = Vec::new();
    let mut orphaned: HashSet<u64> = HashSet::new();
    for file in &base.index_files {
        if !file.covers.iter().any(|p| removed.contains(p)) {
            continue;
        }
        entries.push(crate::transaction_log::LogEntry::RemoveIndexFile {
            index_id: file.index_id,
            partition_id: file.file.partition_id,
        });
        // Partitions this file also covered that are not going away lose
        // their entries with it, so they need indexing again
        orphaned.extend(file.covers.iter().filter(|p| !removed.contains(p)));
    }
    let mut orphaned: Vec<u64> = orphaned.into_iter().collect();
    orphaned.sort_unstable();
    (entries, orphaned)
}

/// Groups probe results by data file so a fetch opens each file once.
pub fn group_by_partition(addresses: &[RowAddress]) -> BTreeMap<u64, Vec<u64>> {
    let mut out: BTreeMap<u64, Vec<u64>> = BTreeMap::new();
    for address in addresses {
        out.entry(address.partition_id)
            .or_default()
            .push(address.ordinal);
    }
    for ordinals in out.values_mut() {
        ordinals.sort_unstable();
        ordinals.dedup();
    }
    out
}
