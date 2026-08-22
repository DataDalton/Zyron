//! Unique and primary key enforcement.
//!
//! A declared constraint that does nothing is a lie, so these are enforced
//! by default. The cost is kept off the write path by the statistics the
//! writer already computed: a candidate key is compared against every live
//! file's bounds first, then against its value bloom, and only a file that
//! survives both is opened. A monotonic key opens one file, a clustered key
//! a handful, and a key outside every file's range opens none at all.
//!
//! Rows already removed by a delete predicate do not conflict, so the check
//! runs against the file's live rows rather than everything it was written
//! with.

use zyron_common::ZyronError;

use crate::cells::{cell_to_value, compare_cells};
use crate::manifest::{ManifestFile, PartitionEntry};
use crate::paths::LakePaths;
use crate::predicate::{CompareOp, LakePredicate, PruneDecision, StatsSource};
use crate::reader::{DecodedColumn, LakeFileReader, ZoneVerdict};
use crate::schema::LakeColumn;
use crate::writer::ColumnData;

/// One unique or primary key constraint to enforce.
#[derive(Debug, Clone)]
pub struct UniqueSpec {
    pub name: String,
    /// Key columns in declaration order, at least one
    pub column_ids: Vec<u32>,
}

/// What the check had to read, so a caller can prove the statistics did
/// their job rather than assume it.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct UniqueCheckStats {
    /// Live files in the manifest
    pub files_considered: usize,
    /// Files whose bounds admitted a candidate key
    pub files_admitted: usize,
    /// Files actually opened and read
    pub files_opened: usize,
    /// Files opened whose zone maps held no candidate key, so no key
    /// column of them was decoded
    pub files_zone_rejected: usize,
    /// Files whose key columns were decoded
    pub files_decoded: usize,
    /// Files answered by bisecting the sort key rather than by walking
    /// every live row
    pub files_bisected: usize,
    pub rows_scanned: u64,
}

/// Where a duplicate was found.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum UniqueOutcome {
    Ok,
    /// Two rows of the incoming batch carry the same key
    DuplicateInBatch {
        first_row: usize,
        second_row: usize,
    },
    /// An incoming row's key already exists in a data file
    DuplicateWithStored {
        row: usize,
        partition_id: u64,
    },
}

/// A contiguous key buffer: one allocation for the whole batch rather than
/// one per row, indexed by offset.
struct KeySet {
    blob: Vec<u8>,
    /// (start, end) per row, empty range for a row skipped as NULL
    spans: Vec<(u32, u32)>,
    /// The first row carrying each distinct key, found by hashing the key
    /// bytes and comparing them where they already sit.
    ///
    /// Keyed by row rather than by the bytes themselves: the blob above
    /// exists so a batch costs one allocation instead of one per row, and
    /// an index owning a copy of every key would have spent them anyway
    index: KeyIndex,
}

impl KeySet {
    fn key(&self, row: usize) -> Option<&[u8]> {
        key_in(&self.blob, &self.spans, row)
    }

    /// The first row that carried this key, or None when no row did
    fn first_row(&self, key: &[u8]) -> Option<usize> {
        self.index.row_of(&self.blob, &self.spans, key)
    }

    /// One row per distinct key, paired with the key it carries
    fn distinct(&self) -> impl Iterator<Item = (usize, &[u8])> + '_ {
        self.index
            .rows()
            .filter_map(|row| self.key(row).map(|key| (row, key)))
    }

    /// Whether any row carried a key at all
    fn has_keys(&self) -> bool {
        self.index.len > 0
    }
}

/// An empty slot. Row numbers come from a batch, which cannot reach this
const NO_ROW: u32 = u32::MAX;

/// The first row carrying each distinct key, addressed by a hash of the key
/// bytes.
///
/// Open addressed with linear probing at a load factor of one half, where
/// probing averages under two slots. Slots hold a row number and the keys
/// stay in the blob, so indexing a batch copies none of them: a table keyed
/// by the bytes themselves would own a copy of every key, which is the
/// allocation the blob exists to avoid
struct KeyIndex {
    slots: Vec<u32>,
    /// One less than the slot count, which is the index mask
    mask: usize,
    len: usize,
}

impl KeyIndex {
    fn with_capacity(rows: usize) -> Self {
        let slots = rows
            .saturating_add(1)
            .saturating_mul(2)
            .next_power_of_two()
            .max(16);
        Self {
            slots: vec![NO_ROW; slots],
            mask: slots - 1,
            len: 0,
        }
    }

    /// The slot this key holds, or the empty slot it would take.
    ///
    /// The table stops at half full, so a probe always reaches an empty slot
    /// and the walk terminates
    #[inline]
    fn probe(
        &self,
        blob: &[u8],
        spans: &[(u32, u32)],
        key: &[u8],
        hash: u64,
    ) -> Result<usize, usize> {
        let mut slot = (hash as usize) & self.mask;
        loop {
            let held = self.slots[slot];
            if held == NO_ROW {
                return Err(slot);
            }
            if key_in(blob, spans, held as usize) == Some(key) {
                return Ok(slot);
            }
            slot = (slot + 1) & self.mask;
        }
    }

    /// The row that first carried this key
    fn row_of(&self, blob: &[u8], spans: &[(u32, u32)], key: &[u8]) -> Option<usize> {
        match self.probe(blob, spans, key, zyron_common::hash64(key)) {
            Ok(slot) => Some(self.slots[slot] as usize),
            Err(_) => None,
        }
    }

    /// Records `row` as the first to carry its key, or leaves the row
    /// already holding it in place
    fn insert(&mut self, blob: &[u8], spans: &[(u32, u32)], row: usize, key: &[u8]) {
        debug_assert!(row < NO_ROW as usize, "a row number reached the empty slot");
        if let Err(slot) = self.probe(blob, spans, key, zyron_common::hash64(key)) {
            self.slots[slot] = row as u32;
            self.len += 1;
        }
    }

    /// One row per distinct key, in slot order
    fn rows(&self) -> impl Iterator<Item = usize> + '_ {
        self.slots
            .iter()
            .filter(|&&row| row != NO_ROW)
            .map(|&row| row as usize)
    }
}

/// Rows whose key no parent row has accounted for yet.
///
/// A bit per row rather than a hash set: the members are row numbers the
/// caller already bounded, so membership is an index rather than a hash,
/// and clearing one is a single word write
struct MissingRows {
    bits: Vec<u64>,
    remaining: usize,
}

impl MissingRows {
    fn new(rows: usize) -> Self {
        Self {
            bits: vec![0u64; rows.div_ceil(64)],
            remaining: 0,
        }
    }

    fn insert(&mut self, row: usize) {
        if !self.contains(row) {
            self.bits[row / 64] |= 1u64 << (row % 64);
            self.remaining += 1;
        }
    }

    #[inline]
    fn contains(&self, row: usize) -> bool {
        self.bits
            .get(row / 64)
            .is_some_and(|word| word & (1u64 << (row % 64)) != 0)
    }

    fn remove(&mut self, row: usize) {
        if self.contains(row) {
            self.bits[row / 64] &= !(1u64 << (row % 64));
            self.remaining -= 1;
        }
    }

    fn is_empty(&self) -> bool {
        self.remaining == 0
    }

    /// Every row still missing, ascending
    fn iter(&self) -> impl Iterator<Item = usize> + '_ {
        self.bits.iter().enumerate().flat_map(|(word, bits)| {
            let mut rest = *bits;
            std::iter::from_fn(move || {
                if rest == 0 {
                    return None;
                }
                let bit = rest.trailing_zeros() as usize;
                rest &= rest - 1;
                Some(word * 64 + bit)
            })
        })
    }
}

/// One row's key inside a blob, or None for a row skipped as NULL
#[inline]
fn key_in<'a>(blob: &'a [u8], spans: &[(u32, u32)], row: usize) -> Option<&'a [u8]> {
    let (start, end) = spans[row];
    if start == end {
        None
    } else {
        Some(&blob[start as usize..end as usize])
    }
}

/// Appends one row's composite key. Components are length prefixed so
/// ("a", "bc") and ("ab", "c") stay distinct.
fn push_key(blob: &mut Vec<u8>, cells: &[Option<&[u8]>]) {
    for cell in cells {
        let bytes = cell.unwrap_or(&[]);
        blob.extend_from_slice(&(bytes.len() as u32).to_le_bytes());
        blob.extend_from_slice(bytes);
    }
}

/// Builds the batch's composite keys plus the leading column's range.
///
/// Returns None when the batch is empty. The key blob is one allocation for
/// the whole batch, indexed by per-row spans.
fn build_keys<'a>(
    manifest: &ManifestFile,
    spec: &UniqueSpec,
    batch: &'a [ColumnData],
) -> Result<
    Option<(
        KeySet,
        zyron_common::TypeId,
        Option<&'a [u8]>,
        Option<&'a [u8]>,
    )>,
    ZyronError,
> {
    if spec.column_ids.is_empty() {
        return Err(ZyronError::Internal(format!(
            "constraint \"{}\" names no columns",
            spec.name
        )));
    }
    let mut key_columns = Vec::with_capacity(spec.column_ids.len());
    for id in &spec.column_ids {
        let data = batch.iter().find(|c| c.column_id == *id).ok_or_else(|| {
            ZyronError::Internal(format!(
                "constraint \"{}\" needs column {}, which the batch does not carry",
                spec.name, id
            ))
        })?;
        key_columns.push(data);
    }
    let row_count = key_columns[0].len();
    if row_count == 0 {
        return Ok(None);
    }
    let leading_physical = manifest
        .schema
        .column_by_id(spec.column_ids[0])
        .ok_or_else(|| {
            ZyronError::Internal(format!(
                "constraint \"{}\" names column {}, which is not in the schema",
                spec.name, spec.column_ids[0]
            ))
        })?
        .physical_type_id();

    let mut keys = KeySet {
        blob: Vec::with_capacity(row_count * 16),
        spans: Vec::with_capacity(row_count),
        index: KeyIndex::with_capacity(row_count),
    };
    let leading = key_columns[0];
    let mut min_cell: Option<&[u8]> = None;
    let mut max_cell: Option<&[u8]> = None;

    for row in 0..row_count {
        let mut cells = Vec::with_capacity(key_columns.len());
        let mut has_null = false;
        for column in &key_columns {
            match column.cell(row) {
                Some(cell) => cells.push(Some(cell)),
                None => {
                    has_null = true;
                    break;
                }
            }
        }
        if has_null {
            keys.spans.push((0, 0));
            continue;
        }
        let start = keys.blob.len() as u32;
        push_key(&mut keys.blob, &cells);
        keys.spans.push((start, keys.blob.len() as u32));

        if let Some(cell) = leading.cell(row) {
            if min_cell
                .map(|m| compare_cells(leading_physical, cell, m).is_lt())
                .unwrap_or(true)
            {
                min_cell = Some(cell);
            }
            if max_cell
                .map(|m| compare_cells(leading_physical, cell, m).is_gt())
                .unwrap_or(true)
            {
                max_cell = Some(cell);
            }
        }
    }
    // Filled after the blob stops growing, so every entry addresses bytes
    // that will not move again
    let KeySet { blob, spans, index } = &mut keys;
    for row in 0..row_count {
        let Some(key) = key_in(blob, spans, row) else {
            continue;
        };
        index.insert(blob, spans, row, key);
    }
    Ok(Some((keys, leading_physical, min_cell, max_cell)))
}

/// The two bounds predicates that rule out a file whose leading-column range
/// misses the batch entirely.
fn leading_range(
    spec: &UniqueSpec,
    physical: zyron_common::TypeId,
    min_cell: Option<&[u8]>,
    max_cell: Option<&[u8]>,
) -> Option<(LakePredicate, LakePredicate)> {
    let (lo, hi) = (min_cell?, max_cell?);
    let lo = cell_to_value(physical, lo)?;
    let hi = cell_to_value(physical, hi)?;
    Some((
        LakePredicate::Compare {
            column_id: spec.column_ids[0],
            op: CompareOp::GtEq,
            value: lo,
        },
        LakePredicate::Compare {
            column_id: spec.column_ids[0],
            op: CompareOp::LtEq,
            value: hi,
        },
    ))
}

/// The read footprint of one spec's unique probe over this batch: the
/// leading key column bounded to the batch's [min, max], or None when the
/// batch carries no non-NULL leading cell. A commit passes this as its read
/// predicate so a concurrent append whose new file admits a key in the
/// probed range conflicts instead of landing a duplicate the probe could
/// not see
pub fn unique_probe_range(
    manifest: &ManifestFile,
    spec: &UniqueSpec,
    batch: &[ColumnData],
) -> Result<Option<LakePredicate>, ZyronError> {
    if spec.column_ids.is_empty() {
        return Ok(None);
    }
    let Some(leading) = batch.iter().find(|c| c.column_id == spec.column_ids[0]) else {
        return Ok(None);
    };
    let leading_physical = manifest
        .schema
        .column_by_id(spec.column_ids[0])
        .ok_or_else(|| {
            ZyronError::Internal(format!(
                "constraint \"{}\" names column {}, which is not in the schema",
                spec.name, spec.column_ids[0]
            ))
        })?
        .physical_type_id();
    let mut min_cell: Option<&[u8]> = None;
    let mut max_cell: Option<&[u8]> = None;
    for cell in leading.iter().flatten() {
        if min_cell
            .map(|m| compare_cells(leading_physical, cell, m).is_lt())
            .unwrap_or(true)
        {
            min_cell = Some(cell);
        }
        if max_cell
            .map(|m| compare_cells(leading_physical, cell, m).is_gt())
            .unwrap_or(true)
        {
            max_cell = Some(cell);
        }
    }
    Ok(leading_range(spec, leading_physical, min_cell, max_cell)
        .map(|(lower, upper)| LakePredicate::And(vec![lower, upper])))
}

/// Enforces one unique constraint over an incoming batch.
///
/// Returns the first violation found and what the check had to read. A row
/// whose key has a NULL component is skipped: SQL admits any number of them.
pub fn check_unique(
    paths: &LakePaths,
    manifest: &ManifestFile,
    spec: &UniqueSpec,
    batch: &[ColumnData],
) -> Result<(UniqueOutcome, UniqueCheckStats), ZyronError> {
    check_unique_replacing(paths, manifest, spec, batch, None)
}

/// Enforces one unique constraint over rows that replace existing ones.
///
/// `superseded` is the predicate naming the rows this statement removes.
/// They are excluded from the stored side, so an UPDATE that leaves a key
/// unchanged does not collide with the copy it is rewriting, while a second
/// row taking a key that a surviving row already holds still does.
pub fn check_unique_replacing(
    paths: &LakePaths,
    manifest: &ManifestFile,
    spec: &UniqueSpec,
    batch: &[ColumnData],
    superseded: Option<&LakePredicate>,
) -> Result<(UniqueOutcome, UniqueCheckStats), ZyronError> {
    let mut stats = UniqueCheckStats::default();
    let Some((keys, leading_physical, min_cell, max_cell)) = build_keys(manifest, spec, batch)?
    else {
        return Ok((UniqueOutcome::Ok, stats));
    };

    // A key the batch carries twice is a violation before any file is read
    let row_count = keys.spans.len();
    for row in 0..row_count {
        let Some(key) = keys.key(row) else { continue };
        match keys.first_row(key) {
            Some(first_row) if first_row != row => {
                return Ok((
                    UniqueOutcome::DuplicateInBatch {
                        first_row,
                        second_row: row,
                    },
                    stats,
                ));
            }
            _ => {}
        }
    }
    if !keys.has_keys() {
        // Every row's key is NULL somewhere, nothing to enforce
        return Ok((UniqueOutcome::Ok, stats));
    }

    let leading = manifest
        .schema
        .column_by_id(spec.column_ids[0])
        .ok_or_else(|| {
            ZyronError::Internal(format!(
                "constraint \"{}\" names column {}, which is not in the schema",
                spec.name, spec.column_ids[0]
            ))
        })?;
    let range = leading_range(spec, leading_physical, min_cell, max_cell);
    for entry in &manifest.entries {
        stats.files_considered += 1;
        if let Some((lower, upper)) = &range {
            let file_stats = manifest.file_stats(entry);
            if lower.prune(&file_stats) == PruneDecision::CannotMatch
                || upper.prune(&file_stats) == PruneDecision::CannotMatch
            {
                continue;
            }
        }
        stats.files_admitted += 1;

        // The value bloom answers for the leading column exactly, so a file
        // whose bloom admits no candidate key is skipped without any read
        if spec.column_ids.len() == 1
            && !bloom_admits_keys(
                manifest,
                entry,
                spec,
                keys.distinct().map(|(_, key)| key),
                leading_physical,
            )?
        {
            continue;
        }

        if let Some(outcome) = scan_file(
            paths, manifest, entry, spec, leading, &keys, superseded, &mut stats,
        )? {
            return Ok((outcome, stats));
        }
    }

    Ok((UniqueOutcome::Ok, stats))
}

/// The leading component of a composite key, which is the column every
/// statistic in this file is keyed by.
fn leading_cell(key: &[u8]) -> &[u8] {
    let len = u32::from_le_bytes([key[0], key[1], key[2], key[3]]) as usize;
    &key[4..4 + len]
}

/// One file opened for a key probe, narrowed as far as its statistics and
/// its ordering allow before any key column is decoded.
struct FileProbe {
    columns: Vec<DecodedColumn>,
    keep: Vec<u8>,
    /// Rows the zone maps left, everything outside provably holds no key
    span: std::ops::Range<usize>,
    /// True when rows ascend by the leading key, so a candidate is found
    /// by bisection instead of by walking the span
    sorted: bool,
}

impl FileProbe {
    fn live(&self, row: usize) -> bool {
        self.keep[row / 8] & (1 << (row % 8)) != 0
    }

    /// This row's composite key, or None when any component is NULL. NULL
    /// components never take part in a key comparison
    fn key_at<'s>(&self, row: usize, scratch: &'s mut Vec<u8>) -> Option<&'s [u8]> {
        scratch.clear();
        let mut cells = Vec::with_capacity(self.columns.len());
        for column in &self.columns {
            cells.push(Some(column.cell(row)?));
        }
        push_key(scratch, &cells);
        Some(scratch.as_slice())
    }
}

/// Opens one file and narrows it to the rows that can carry a candidate
/// key, decoding the key columns only when some zone admits one.
///
/// `superseded` names rows this statement is replacing. They are cleared
/// from the live set before any key is compared, because a row on its way
/// out cannot constrain the row replacing it. Without it every UPDATE that
/// leaves a key unchanged would collide with the copy it is rewriting.
///
/// Returns None when the file provably holds no candidate, which costs its
/// zone region and no column payload at all
#[allow(clippy::too_many_arguments)]
fn open_probe(
    paths: &LakePaths,
    manifest: &ManifestFile,
    entry: &PartitionEntry,
    spec: &UniqueSpec,
    leading: &LakeColumn,
    candidates: &[&[u8]],
    superseded: Option<&LakePredicate>,
    stats: &mut UniqueCheckStats,
) -> Result<Option<FileProbe>, ZyronError> {
    let reader = LakeFileReader::open(paths, entry.partition_id)?;
    stats.files_opened += 1;
    let rows = reader.row_count();
    if rows == 0 {
        return Ok(None);
    }

    // Zone maps before payload. A file whose bounds admit a key can still
    // have no zone that does, and rejecting it here reads nothing else
    let span = match reader.zone_span_for_cells(leading, candidates)? {
        ZoneVerdict::NoMatch => {
            stats.files_zone_rejected += 1;
            return Ok(None);
        }
        ZoneVerdict::Span(span) => span,
        ZoneVerdict::Undecided => 0..rows,
    };

    let mut keep = reader.delete_survivors(&manifest.schema, manifest, entry)?;

    // Rows this statement replaces stop constraining before any key is
    // built. Three-valued logic decides it the way a delete predicate is
    // decided: only a row the predicate provably matches is superseded, an
    // unknown outcome leaves the row in force
    if let Some(predicate) = superseded {
        let columns = reader.read_predicate_columns(&manifest.schema, &[predicate])?;
        let compiled = crate::reader::CompiledPredicate::new(predicate, &columns);
        for row in 0..rows {
            if keep[row / 8] & (1 << (row % 8)) == 0 {
                continue;
            }
            if compiled.evaluate(&columns, row) == Some(true) {
                keep[row / 8] &= !(1 << (row % 8));
            }
        }
    }

    let mut columns = Vec::with_capacity(spec.column_ids.len());
    for id in &spec.column_ids {
        let column = manifest.schema.column_by_id(*id).ok_or_else(|| {
            ZyronError::Internal(format!(
                "constraint \"{}\" names column {}, which is not in the schema",
                spec.name, id
            ))
        })?;
        columns.push(reader.read_column(column)?);
    }
    stats.files_decoded += 1;
    let sorted = reader.is_sorted_by(leading.id);
    if sorted {
        stats.files_bisected += 1;
    }
    Ok(Some(FileProbe {
        columns,
        keep,
        span,
        sorted,
    }))
}

/// Rows of the probe that can carry `key`, ascending.
///
/// A sorted file bisects to the run sharing the key's sort key, which is a
/// few rows even in a file of millions. An unsorted one has to offer the
/// whole surviving span, and the caller compares each row once
fn rows_for_key(probe: &FileProbe, key: &[u8]) -> std::ops::Range<usize> {
    if probe.sorted {
        probe.columns[0].sort_key_range_in(leading_cell(key), probe.span.clone())
    } else {
        probe.span.clone()
    }
}

/// Reads one file's live rows and looks for a key the batch also carries.
#[allow(clippy::too_many_arguments)]
fn scan_file(
    paths: &LakePaths,
    manifest: &ManifestFile,
    entry: &PartitionEntry,
    spec: &UniqueSpec,
    leading: &LakeColumn,
    keys: &KeySet,
    superseded: Option<&LakePredicate>,
    stats: &mut UniqueCheckStats,
) -> Result<Option<UniqueOutcome>, ZyronError> {
    let candidates: Vec<&[u8]> = keys.distinct().map(|(_, key)| leading_cell(key)).collect();
    let Some(probe) = open_probe(
        paths,
        manifest,
        entry,
        spec,
        leading,
        &candidates,
        superseded,
        stats,
    )?
    else {
        return Ok(None);
    };

    let mut scratch = Vec::with_capacity(32);
    if probe.sorted {
        // One bisection per candidate key. Every row it lands on already
        // shares the candidate's sort key, so the exact comparison runs a
        // handful of times rather than once per stored row
        for (batch_row, key) in keys.distinct() {
            for row in rows_for_key(&probe, key) {
                if !probe.live(row) {
                    continue;
                }
                stats.rows_scanned += 1;
                if probe.key_at(row, &mut scratch) == Some(key) {
                    return Ok(Some(UniqueOutcome::DuplicateWithStored {
                        row: batch_row,
                        partition_id: entry.partition_id,
                    }));
                }
            }
        }
        return Ok(None);
    }

    for row in probe.span.clone() {
        if !probe.live(row) {
            continue;
        }
        stats.rows_scanned += 1;
        let Some(key) = probe.key_at(row, &mut scratch) else {
            continue;
        };
        if let Some(batch_row) = keys.first_row(key) {
            return Ok(Some(UniqueOutcome::DuplicateWithStored {
                row: batch_row,
                partition_id: entry.partition_id,
            }));
        }
    }
    Ok(None)
}

/// Whether every referenced key was found in the parent.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ForeignKeyOutcome {
    Ok,
    /// Every row whose referenced key is not in the parent, ascending.
    ///
    /// All of them, not just the first: a constraint that quarantines has to
    /// divert each rejected row, and two rows sharing one absent key are two
    /// rejected rows.
    Missing {
        rows: Vec<usize>,
    },
}

/// Checks that every non-NULL key in `values` exists in a lake parent.
///
/// `values` carries the child's referencing cells under the PARENT's column
/// ids, so the probe reads the parent's own key columns. The same statistics
/// drive it as the unique check: a file whose bounds or bloom rule out every
/// still-missing key is never opened, and the scan stops as soon as every
/// key has been found.
///
/// A row with any NULL key component is not checked, which is SQL's MATCH
/// SIMPLE rule: a partially NULL foreign key references nothing.
pub fn check_foreign_key(
    paths: &LakePaths,
    manifest: &ManifestFile,
    parent_column_ids: &[u32],
    values: &[ColumnData],
) -> Result<(ForeignKeyOutcome, UniqueCheckStats), ZyronError> {
    let spec = UniqueSpec {
        name: "foreign key".into(),
        column_ids: parent_column_ids.to_vec(),
    };
    let mut stats = UniqueCheckStats::default();
    let built = build_keys(manifest, &spec, values)?;
    let Some((keys, leading_physical, min_cell, max_cell)) = built else {
        return Ok((ForeignKeyOutcome::Ok, stats));
    };
    if !keys.has_keys() {
        return Ok((ForeignKeyOutcome::Ok, stats));
    }

    // Keys still to find, dropped as parent rows account for them. Held as
    // the row each key came in on rather than as a copy of its bytes: the
    // bytes are already in the batch's blob and every one of them would
    // otherwise be allocated again here, and again for each parent file
    let mut missing = MissingRows::new(keys.spans.len());
    for (row, _) in keys.distinct() {
        missing.insert(row);
    }
    let range = leading_range(&spec, leading_physical, min_cell, max_cell);
    let leading = manifest
        .schema
        .column_by_id(spec.column_ids[0])
        .ok_or_else(|| {
            ZyronError::Internal(format!(
                "foreign key references column {}, which is not in the parent schema",
                spec.column_ids[0]
            ))
        })?;

    for entry in &manifest.entries {
        stats.files_considered += 1;
        if missing.is_empty() {
            break;
        }
        if let Some((lower, upper)) = &range {
            let file_stats = manifest.file_stats(entry);
            if lower.prune(&file_stats) == PruneDecision::CannotMatch
                || upper.prune(&file_stats) == PruneDecision::CannotMatch
            {
                continue;
            }
        }
        stats.files_admitted += 1;
        if spec.column_ids.len() == 1
            && !bloom_admits_keys(
                manifest,
                entry,
                &spec,
                missing.iter().filter_map(|row| keys.key(row)),
                leading_physical,
            )?
        {
            continue;
        }
        mark_present(
            paths,
            manifest,
            entry,
            &spec,
            leading,
            &keys,
            &mut missing,
            &mut stats,
        )?;
    }

    if missing.is_empty() {
        return Ok((ForeignKeyOutcome::Ok, stats));
    }
    // Row level, not key level: every row carrying a key the parent lacks is
    // a rejected row, including duplicates of the same key
    let mut rows = Vec::new();
    for row in 0..keys.spans.len() {
        let Some(key) = keys.key(row) else { continue };
        if keys
            .first_row(key)
            .is_some_and(|first| missing.contains(first))
        {
            rows.push(row);
        }
    }
    Ok((ForeignKeyOutcome::Missing { rows }, stats))
}

/// Removes from `missing` every key this file's live rows account for.
fn mark_present(
    paths: &LakePaths,
    manifest: &ManifestFile,
    entry: &PartitionEntry,
    spec: &UniqueSpec,
    leading: &LakeColumn,
    keys: &KeySet,
    missing: &mut MissingRows,
    stats: &mut UniqueCheckStats,
) -> Result<(), ZyronError> {
    let candidates: Vec<&[u8]> = missing
        .iter()
        .filter_map(|row| keys.key(row))
        .map(leading_cell)
        .collect();
    let Some(probe) = open_probe(
        paths,
        manifest,
        entry,
        spec,
        leading,
        &candidates,
        None,
        stats,
    )?
    else {
        return Ok(());
    };

    let mut scratch = Vec::with_capacity(32);
    if probe.sorted {
        // One bisection per key still missing, so a parent file answers in
        // time proportional to the keys asked about rather than to its size
        // The rows still wanted, taken before the loop because it removes
        // from the set as it goes. Four bytes apiece, where copying the keys
        // themselves cost an allocation each on every parent file
        let asked: Vec<usize> = missing.iter().collect();
        for batch_row in asked {
            let Some(key) = keys.key(batch_row) else {
                continue;
            };
            for row in rows_for_key(&probe, key) {
                if !probe.live(row) {
                    continue;
                }
                stats.rows_scanned += 1;
                if probe.key_at(row, &mut scratch) == Some(key) {
                    missing.remove(batch_row);
                    break;
                }
            }
        }
        return Ok(());
    }

    for row in probe.span.clone() {
        if missing.is_empty() {
            break;
        }
        if !probe.live(row) {
            continue;
        }
        stats.rows_scanned += 1;
        let Some(key) = probe.key_at(row, &mut scratch) else {
            continue;
        };
        if let Some(batch_row) = keys.first_row(key) {
            missing.remove(batch_row);
        }
    }
    Ok(())
}

/// True when the file's bloom admits at least one of the given keys.
fn bloom_admits_keys<'a>(
    manifest: &ManifestFile,
    entry: &PartitionEntry,
    spec: &UniqueSpec,
    keys: impl Iterator<Item = &'a [u8]>,
    physical: zyron_common::TypeId,
) -> Result<bool, ZyronError> {
    let Some(stats) = entry.stats_for(spec.column_ids[0]) else {
        return Ok(true);
    };
    if stats.bloom.is_none() {
        return Ok(true);
    }
    let file_stats = manifest.file_stats(entry);
    for key in keys {
        let cell = &key[4..];
        let Some(value) = cell_to_value(physical, cell) else {
            return Ok(true);
        };
        if file_stats.may_contain(spec.column_ids[0], &value) {
            return Ok(true);
        }
    }
    Ok(false)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::operations::{append_rows, delete_where};
    use crate::predicate::LakeValue;
    use crate::schema::{LakeColumn, LakeSchema};
    use crate::transaction_log::{CommitAttempt, OperationKind, TransactionLog};
    use std::collections::BTreeMap;
    use zyron_common::TypeId;

    /// Builds a blob and spans from whole keys, the shape `build_keys` leaves
    fn blob_of(keys: &[Option<&[u8]>]) -> (Vec<u8>, Vec<(u32, u32)>) {
        let mut blob = Vec::new();
        let mut spans = Vec::new();
        for key in keys {
            match key {
                None => spans.push((0, 0)),
                Some(bytes) => {
                    let start = blob.len() as u32;
                    blob.extend_from_slice(bytes);
                    spans.push((start, blob.len() as u32));
                }
            }
        }
        (blob, spans)
    }

    #[test]
    fn the_key_index_answers_with_the_first_row_that_carried_a_key() {
        let keys: Vec<Option<&[u8]>> = vec![
            Some(b"alpha"),
            Some(b"beta"),
            None,
            Some(b"alpha"),
            Some(b"gamma"),
        ];
        let (blob, spans) = blob_of(&keys);
        let mut index = KeyIndex::with_capacity(keys.len());
        for row in 0..keys.len() {
            if let Some(key) = key_in(&blob, &spans, row) {
                index.insert(&blob, &spans, row, key);
            }
        }

        assert_eq!(index.len, 3, "alpha repeats and one row is null");
        assert_eq!(index.row_of(&blob, &spans, b"alpha"), Some(0));
        assert_eq!(index.row_of(&blob, &spans, b"beta"), Some(1));
        assert_eq!(index.row_of(&blob, &spans, b"gamma"), Some(4));
        assert_eq!(index.row_of(&blob, &spans, b"delta"), None);
        assert_eq!(
            index.row_of(&blob, &spans, b"alph"),
            None,
            "a prefix is a different key"
        );

        let mut rows: Vec<usize> = index.rows().collect();
        rows.sort_unstable();
        assert_eq!(rows, vec![0, 1, 4]);
    }

    #[test]
    fn the_key_index_keeps_colliding_keys_apart() {
        // Far more keys than slots would hold without probing, so the walk
        // past an occupied slot is exercised at every load below one half
        let owned: Vec<Vec<u8>> = (0..500u32).map(|v| v.to_le_bytes().to_vec()).collect();
        let keys: Vec<Option<&[u8]>> = owned.iter().map(|v| Some(v.as_slice())).collect();
        let (blob, spans) = blob_of(&keys);
        let mut index = KeyIndex::with_capacity(keys.len());
        for row in 0..keys.len() {
            let key = key_in(&blob, &spans, row).expect("key");
            index.insert(&blob, &spans, row, key);
        }
        assert_eq!(index.len, keys.len(), "every key is distinct");
        for row in 0..keys.len() {
            let key = key_in(&blob, &spans, row).expect("key");
            assert_eq!(
                index.row_of(&blob, &spans, key),
                Some(row),
                "row {row} did not find its own key"
            );
        }
    }

    #[test]
    fn missing_rows_tracks_membership_and_empties_exactly_once() {
        let mut missing = MissingRows::new(200);
        assert!(missing.is_empty());
        for row in [0usize, 63, 64, 65, 199] {
            missing.insert(row);
        }
        // A repeat must not count twice, or the set would never empty
        missing.insert(64);
        assert_eq!(missing.iter().collect::<Vec<_>>(), vec![0, 63, 64, 65, 199]);
        assert!(missing.contains(63) && !missing.contains(62));

        // Removing one that was never there must not go below zero
        missing.remove(100);
        assert_eq!(missing.iter().count(), 5);
        for row in [0usize, 63, 64, 65] {
            missing.remove(row);
        }
        assert!(!missing.is_empty(), "one row is still unaccounted for");
        assert_eq!(missing.iter().collect::<Vec<_>>(), vec![199]);
        missing.remove(199);
        assert!(missing.is_empty());
        assert_eq!(missing.iter().count(), 0);
    }

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
                    name: "tag".into(),
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

    fn attempt(timestamp_us: i64) -> CommitAttempt<'static> {
        CommitAttempt {
            operation: OperationKind::Append,
            db_txn_id: 0,
            commit_lsn: 1,
            timestamp_us,
            read_predicate: None,
            read_version: 0,
            audit: None,
            deadline: None,
        }
    }

    fn batch(ids: &[i64], tags: &[Option<&str>]) -> Vec<ColumnData> {
        vec![
            ColumnData::from_cells(
                0,
                ids.iter().map(|v| Some(v.to_le_bytes().to_vec())).collect(),
            ),
            ColumnData::from_cells(
                1,
                tags.iter()
                    .map(|t| t.map(|s| s.as_bytes().to_vec()))
                    .collect(),
            ),
        ]
    }

    fn ids_only(ids: &[i64]) -> Vec<ColumnData> {
        let tags: Vec<Option<&str>> = ids.iter().map(|_| Some("x")).collect();
        batch(ids, &tags)
    }

    fn new_log(dir: &std::path::Path) -> TransactionLog {
        TransactionLog::create(
            LakePaths::new(dir, 31),
            CommitAttempt {
                operation: OperationKind::SchemaChange,
                ..attempt(100)
            },
            &schema(),
            None,
            &BTreeMap::new(),
        )
        .expect("create")
    }

    /// A table clustered ascending on its key, so appends write ordered
    /// files the way a declared primary key makes them
    fn clustered_log(dir: &std::path::Path) -> TransactionLog {
        TransactionLog::create(
            LakePaths::new(dir, 31),
            CommitAttempt {
                operation: OperationKind::SchemaChange,
                ..attempt(100)
            },
            &schema(),
            Some(&crate::ClusterSpec {
                spec_id: 1,
                keys: vec![crate::manifest::ClusterKey {
                    column_id: 0,
                    strategy: crate::manifest::ClusterStrategy::RangePartition,
                    param: 0,
                }],
            }),
            &BTreeMap::new(),
        )
        .expect("create")
    }

    fn pk() -> UniqueSpec {
        UniqueSpec {
            name: "pk_id".into(),
            column_ids: vec![0],
        }
    }

    #[test]
    fn test_foreign_key_finds_referenced_keys_and_names_the_first_missing_row() {
        let dir = tempfile::TempDir::new().expect("temp dir");
        let log = new_log(dir.path());
        append_rows(&log, attempt(200), 31, &ids_only(&[1, 2, 3])).expect("append");
        let manifest = log.latest_manifest().expect("manifest");

        // Every referenced key exists
        let (outcome, _) =
            check_foreign_key(log.paths(), &manifest, &[0], &ids_only(&[1, 3, 1])).expect("check");
        assert_eq!(outcome, ForeignKeyOutcome::Ok);

        // The first row whose key is absent is named
        let (outcome, _) =
            check_foreign_key(log.paths(), &manifest, &[0], &ids_only(&[1, 9, 2])).expect("check");
        assert_eq!(outcome, ForeignKeyOutcome::Missing { rows: vec![1] });
    }

    #[test]
    fn test_foreign_key_stops_reading_once_every_key_is_found() {
        let dir = tempfile::TempDir::new().expect("temp dir");
        let log = new_log(dir.path());
        for (n, ids) in [vec![1i64, 2, 3], vec![10, 11, 12], vec![20, 21, 22]]
            .into_iter()
            .enumerate()
        {
            append_rows(&log, attempt(200 + n as i64), 31, &ids_only(&ids)).expect("append");
        }
        let manifest = log.latest_manifest().expect("manifest");

        // A key in one file's range opens that file only
        let (outcome, stats) =
            check_foreign_key(log.paths(), &manifest, &[0], &ids_only(&[11])).expect("check");
        assert_eq!(outcome, ForeignKeyOutcome::Ok);
        assert_eq!(stats.files_admitted, 1);
        assert_eq!(stats.files_opened, 1, "one file, not three");

        // A key outside every range opens nothing and still reports missing
        let (outcome, stats) =
            check_foreign_key(log.paths(), &manifest, &[0], &ids_only(&[99])).expect("check");
        assert_eq!(outcome, ForeignKeyOutcome::Missing { rows: vec![0] });
        assert_eq!(stats.files_opened, 0, "bounds alone answered");
    }

    #[test]
    fn test_a_null_referencing_key_is_not_checked() {
        let dir = tempfile::TempDir::new().expect("temp dir");
        let log = new_log(dir.path());
        append_rows(&log, attempt(200), 31, &ids_only(&[1])).expect("append");
        let manifest = log.latest_manifest().expect("manifest");

        // MATCH SIMPLE: a NULL component references nothing, so no check
        let values = vec![ColumnData::from_cells(
            0,
            vec![None, Some(1i64.to_le_bytes().to_vec())],
        )];
        let (outcome, _) = check_foreign_key(log.paths(), &manifest, &[0], &values).expect("check");
        assert_eq!(outcome, ForeignKeyOutcome::Ok);
    }

    #[test]
    fn test_a_deleted_parent_row_no_longer_satisfies_a_reference() {
        let dir = tempfile::TempDir::new().expect("temp dir");
        let log = new_log(dir.path());
        append_rows(&log, attempt(200), 31, &ids_only(&[1, 2, 40])).expect("append");
        delete_where(
            &log,
            CommitAttempt {
                operation: OperationKind::Delete,
                ..attempt(300)
            },
            &LakePredicate::Compare {
                column_id: 0,
                op: CompareOp::Lt,
                value: LakeValue::Int(3),
            },
            "id < 3",
        )
        .expect("delete");
        let manifest = log.latest_manifest().expect("manifest");

        let (outcome, _) =
            check_foreign_key(log.paths(), &manifest, &[0], &ids_only(&[40])).expect("check");
        assert_eq!(outcome, ForeignKeyOutcome::Ok);
        let (outcome, _) =
            check_foreign_key(log.paths(), &manifest, &[0], &ids_only(&[1])).expect("check");
        assert_eq!(outcome, ForeignKeyOutcome::Missing { rows: vec![0] });
    }

    #[test]
    fn test_unique_is_enforced_against_stored_rows_and_within_the_batch() {
        let dir = tempfile::TempDir::new().expect("temp dir");
        let log = new_log(dir.path());
        append_rows(&log, attempt(200), 31, &ids_only(&[1, 2, 3])).expect("append");
        let manifest = log.latest_manifest().expect("manifest");

        // A fresh key passes
        let (outcome, _) =
            check_unique(log.paths(), &manifest, &pk(), &ids_only(&[4, 5])).expect("check");
        assert_eq!(outcome, UniqueOutcome::Ok);

        // A key already stored is refused, naming the file it collided with
        let (outcome, _) =
            check_unique(log.paths(), &manifest, &pk(), &ids_only(&[9, 2])).expect("check");
        match outcome {
            UniqueOutcome::DuplicateWithStored { row, partition_id } => {
                assert_eq!(row, 1);
                assert_eq!(partition_id, manifest.entries[0].partition_id);
            }
            other => panic!("expected a stored duplicate, got {other:?}"),
        }

        // Two rows of one batch collide with each other
        let (outcome, _) =
            check_unique(log.paths(), &manifest, &pk(), &ids_only(&[7, 8, 7])).expect("check");
        assert_eq!(
            outcome,
            UniqueOutcome::DuplicateInBatch {
                first_row: 0,
                second_row: 2
            }
        );
    }

    #[test]
    fn test_unique_enforcement_probes_only_files_min_max_admits() {
        let dir = tempfile::TempDir::new().expect("temp dir");
        let log = new_log(dir.path());
        // Four files with disjoint, ascending key ranges, the shape a
        // monotonic key produces
        for (n, ids) in [
            vec![1i64, 2, 3],
            vec![10, 11, 12],
            vec![20, 21, 22],
            vec![30, 31, 32],
        ]
        .into_iter()
        .enumerate()
        {
            append_rows(&log, attempt(200 + n as i64), 31, &ids_only(&ids)).expect("append");
        }
        let manifest = log.latest_manifest().expect("manifest");
        assert_eq!(manifest.entries.len(), 4);

        // A key above every stored range opens nothing at all
        let (outcome, stats) =
            check_unique(log.paths(), &manifest, &pk(), &ids_only(&[99])).expect("check");
        assert_eq!(outcome, UniqueOutcome::Ok);
        assert_eq!(stats.files_considered, 4);
        assert_eq!(stats.files_opened, 0, "bounds alone answered");
        assert_eq!(stats.rows_scanned, 0);

        // A key inside one file's range opens that file only
        let (outcome, stats) =
            check_unique(log.paths(), &manifest, &pk(), &ids_only(&[21])).expect("check");
        assert!(matches!(outcome, UniqueOutcome::DuplicateWithStored { .. }));
        // The scan stops at the collision, so files_considered depends on
        // where the manifest's ordering put that file. What matters is that
        // bounds admitted exactly one and only that one was read
        assert_eq!(stats.files_admitted, 1);
        assert_eq!(stats.files_opened, 1, "one file, not four");
        assert!(
            stats.rows_scanned <= 3,
            "scanned {} rows, one file holds three",
            stats.rows_scanned
        );

        // A batch spanning two ranges opens two
        let (_, stats) =
            check_unique(log.paths(), &manifest, &pk(), &ids_only(&[13, 19])).expect("check");
        assert!(
            stats.files_opened <= 2,
            "opened {} files for a two-range batch",
            stats.files_opened
        );
    }

    #[test]
    fn test_a_sorted_file_bisects_instead_of_walking_every_row() {
        let dir = tempfile::TempDir::new().expect("temp dir");
        // Clustered ascending on the key, which is what a table with a
        // primary key bootstraps to and what makes the file ordered
        let log = clustered_log(dir.path());
        let ids: Vec<i64> = (0..4096).collect();
        append_rows(&log, attempt(200), 31, &ids_only(&ids)).expect("append");
        let manifest = log.latest_manifest().expect("manifest");
        assert_eq!(manifest.entries.len(), 1);

        // A key the file holds is found without inspecting the rows before
        // it. Bisection lands on the run sharing its sort key, and the file
        // holds one row per key
        let (outcome, stats) =
            check_unique(log.paths(), &manifest, &pk(), &ids_only(&[3000])).expect("check");
        assert!(matches!(outcome, UniqueOutcome::DuplicateWithStored { .. }));
        assert_eq!(stats.files_bisected, 1, "the file claims ascending order");
        assert_eq!(
            stats.rows_scanned, 1,
            "bisection compared one row, a walk would have compared thousands"
        );
    }

    #[test]
    fn test_a_curve_ordered_file_is_not_bisected() {
        let dir = tempfile::TempDir::new().expect("temp dir");
        let paths = LakePaths::new(dir.path(), 31);
        // Z-order places rows by every key at once, so the leading column
        // is not monotonic and the file must not claim ascending order
        let ids: Vec<i64> = (0..64).rev().collect();
        let tags: Vec<Option<&str>> = ids.iter().map(|_| Some("x")).collect();
        let columns = batch(&ids, &tags);
        crate::writer::write_data_file(
            &paths,
            &schema(),
            &crate::writer::WriteRequest {
                partition_id: 0xC1,
                columns: &columns,
                sort_keys: &[0, 1],
                sort_strategies: &[
                    crate::manifest::ClusterStrategy::BitInterleave,
                    crate::manifest::ClusterStrategy::BitInterleave,
                ],
                cluster_spec_id: 1,
                table_id: 31,
                bloom_columns: &[],
                index_id: None,
            },
        )
        .expect("write");

        let reader = LakeFileReader::open(&paths, 0xC1).expect("open");
        assert!(
            !reader.is_sorted_by(0),
            "a curve ordered file claims no ascending column"
        );
    }

    #[test]
    fn test_a_deleted_row_does_not_block_reinserting_its_key() {
        let dir = tempfile::TempDir::new().expect("temp dir");
        let log = new_log(dir.path());
        append_rows(&log, attempt(200), 31, &ids_only(&[1, 2, 3, 40])).expect("append");
        delete_where(
            &log,
            CommitAttempt {
                operation: OperationKind::Delete,
                ..attempt(300)
            },
            &LakePredicate::Compare {
                column_id: 0,
                op: CompareOp::Lt,
                value: LakeValue::Int(3),
            },
            "id < 3",
        )
        .expect("delete");

        let manifest = log.latest_manifest().expect("manifest");
        // 1 and 2 were deleted, so their keys are free again
        let (outcome, _) =
            check_unique(log.paths(), &manifest, &pk(), &ids_only(&[1, 2])).expect("check");
        assert_eq!(outcome, UniqueOutcome::Ok);
        // 3 survived the delete and still collides
        let (outcome, _) =
            check_unique(log.paths(), &manifest, &pk(), &ids_only(&[3])).expect("check");
        assert!(matches!(outcome, UniqueOutcome::DuplicateWithStored { .. }));
    }

    #[test]
    fn test_a_null_key_component_never_conflicts() {
        let dir = tempfile::TempDir::new().expect("temp dir");
        let log = new_log(dir.path());
        append_rows(&log, attempt(200), 31, &batch(&[1, 2], &[None, Some("b")])).expect("append");
        let manifest = log.latest_manifest().expect("manifest");

        let spec = UniqueSpec {
            name: "uq_tag".into(),
            column_ids: vec![1],
        };
        // Any number of NULLs coexist under a unique constraint
        let (outcome, _) = check_unique(
            log.paths(),
            &manifest,
            &spec,
            &batch(&[7, 8], &[None, None]),
        )
        .expect("check");
        assert_eq!(outcome, UniqueOutcome::Ok);

        // A real value still collides
        let (outcome, _) =
            check_unique(log.paths(), &manifest, &spec, &batch(&[9], &[Some("b")])).expect("check");
        assert!(matches!(outcome, UniqueOutcome::DuplicateWithStored { .. }));
    }

    #[test]
    fn test_a_composite_key_distinguishes_component_boundaries() {
        let dir = tempfile::TempDir::new().expect("temp dir");
        let log = new_log(dir.path());
        append_rows(&log, attempt(200), 31, &batch(&[1], &[Some("ab")])).expect("append");
        let manifest = log.latest_manifest().expect("manifest");

        let spec = UniqueSpec {
            name: "uq_id_tag".into(),
            column_ids: vec![0, 1],
        };
        // Same leading value, different trailing value
        let (outcome, _) =
            check_unique(log.paths(), &manifest, &spec, &batch(&[1], &[Some("a")])).expect("check");
        assert_eq!(outcome, UniqueOutcome::Ok);
        // The exact pair collides
        let (outcome, _) = check_unique(log.paths(), &manifest, &spec, &batch(&[1], &[Some("ab")]))
            .expect("check");
        assert!(matches!(outcome, UniqueOutcome::DuplicateWithStored { .. }));
    }
}
