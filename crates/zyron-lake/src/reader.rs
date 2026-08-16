//! Data file reader.
//!
//! Opens one immutable .zyr data file, decodes projected columns, and
//! filters rows through the manifest's delete predicates. A column absent
//! from the file reads as all NULL, which is what makes adding a column a
//! metadata-only change, old files never get rewritten.
//!
//! Predicate evaluation over rows is three valued. A delete predicate
//! removes only rows it provably matches, an unknown outcome keeps the
//! row, mirroring SQL DELETE semantics

use std::sync::atomic::{AtomicU64, Ordering};

use zyron_common::curve::normalize_component;
use zyron_common::{TypeId, ZyronError};
use zyron_storage::columnar::{
    STAT_VALUE_SIZE, SlotOrder, SortOrder, ZONE_MAP_BATCH_SIZE, ZoneMapEntry, ZyrFileReader,
    compare_value_to_slot, slot_order,
};
use zyron_storage::encoding::Predicate;

use crate::cells::{compare_cell_to_value, compare_cells};
use crate::encoded_filter::{ColumnEvidence, StoredFilter, rows_matching};
use crate::manifest::{ManifestFile, PartitionEntry};
use crate::paths::LakePaths;
use crate::predicate::{CompareOp, LakePredicate, LakeValue};
use crate::schema::{LakeColumn, LakeSchema};

/// What one open data file can answer without decoding a column.
///
/// A column the file predates has no segment, so every one of its rows
/// reads as NULL and no comparison over it matches. Answering zero rows
/// there is what lets a schema-evolved table skip a whole file instead of
/// decoding it to find nothing
struct FileEvidence<'a> {
    reader: &'a LakeFileReader,
}

impl ColumnEvidence for FileEvidence<'_> {
    fn row_count(&self) -> usize {
        self.reader.row_count
    }

    fn zone_maps(&self, column_id: u32) -> Result<Vec<ZoneMapEntry>, ZyronError> {
        if !self.reader.reader.has_segment(column_id) {
            return Ok(Vec::new());
        }
        self.reader
            .reader
            .read_segment_metadata(column_id, self.reader.row_count)
            .map(|(_, zones)| zones)
    }

    fn eval(
        &self,
        column_id: u32,
        value_size: usize,
        predicate: &Predicate<'_>,
    ) -> Result<Vec<u8>, ZyronError> {
        if !self.reader.reader.has_segment(column_id) {
            return Ok(vec![0u8; self.reader.row_count.div_ceil(8)]);
        }
        // Evaluating on encoded bytes still reads the segment payload, so it
        // counts the same as decoding the column would have
        self.reader.bytes_read.fetch_add(
            self.reader.reader.segment_bytes(column_id),
            Ordering::Relaxed,
        );
        self.reader.reader.eval_column_predicate(
            column_id,
            self.reader.row_count,
            value_size,
            predicate,
        )
    }
}

/// One decoded column with NULL tracking, cells addressed by row ordinal
#[derive(Debug)]
pub struct DecodedColumn {
    pub column_id: u32,
    physical: TypeId,
    /// Fixed cell width, zero routes to the variable-length layout
    value_size: usize,
    /// Flat fixed-width cells, or the canonical varlen buffer of
    /// row count, offsets array and value blob
    data: Vec<u8>,
    /// Indexed by the row's ordinal in the whole file, not by its position
    /// in a decoded range, because the segment stores one bitmap for every
    /// row and a range does not carve it up
    null_bitmap: Vec<u8>,
    /// Rows this column decoded, which is the length of the range and not
    /// the file's row count
    row_count: usize,
    /// Ordinal the decoded range starts at. Zero for a whole column, so
    /// every caller addresses rows by their ordinal in the file either way
    base: usize,
    /// True for a column the file predates, every cell is NULL
    all_null: bool,
}

impl DecodedColumn {
    /// The type the cells compare and decode under
    pub fn physical_type_id(&self) -> TypeId {
        self.physical
    }

    /// Rows this column decoded, as the ordinal range they occupy in the
    /// file. A whole column starts at zero
    pub fn decoded_range(&self) -> std::ops::Range<usize> {
        self.base..self.base + self.row_count
    }

    /// Cell bytes for one row addressed by its ordinal in the file, None
    /// for NULL and for a row outside the decoded range
    pub fn cell(&self, row: usize) -> Option<&[u8]> {
        if self.all_null || self.is_null(row) {
            return None;
        }
        let local = row.checked_sub(self.base)?;
        if local >= self.row_count {
            return None;
        }
        if self.value_size > 0 {
            let start = local * self.value_size;
            Some(&self.data[start..start + self.value_size])
        } else {
            let off_base = 4 + local * 4;
            let mut a = [0u8; 4];
            a.copy_from_slice(&self.data[off_base..off_base + 4]);
            let start = u32::from_le_bytes(a) as usize;
            a.copy_from_slice(&self.data[off_base + 4..off_base + 8]);
            let end = u32::from_le_bytes(a) as usize;
            let blob_base = 4 + (self.row_count + 1) * 4;
            Some(&self.data[blob_base + start..blob_base + end])
        }
    }

    fn is_null(&self, row: usize) -> bool {
        if self.null_bitmap.is_empty() {
            return false;
        }
        match self.null_bitmap.get(row / 8) {
            Some(byte) => byte & (1 << (row % 8)) != 0,
            None => true,
        }
    }

    /// Rows that can hold `cell` in a file sorted ascending on this
    /// column, found by bisection over `span` rather than by inspecting
    /// every row.
    ///
    /// The bisection runs on the same normalized key the writer sorted by,
    /// which is the only order the rows are actually in. That key is a
    /// leading eight bytes for a string and the high half of a 128-bit
    /// integer, so rows sharing it are in no particular order among
    /// themselves and the returned range is a superset of the equal rows.
    /// The caller confirms each one with an exact comparison, which is a
    /// short walk rather than a scan.
    ///
    /// The caller owns the sortedness claim. It is checked against the
    /// file header at the one place a span is produced
    pub fn sort_key_range_in(
        &self,
        cell: &[u8],
        span: std::ops::Range<usize>,
    ) -> std::ops::Range<usize> {
        let target = normalize_component(self.physical, cell);
        // Nulls sort last, so their key is above every value's
        let key = |row: usize| -> (bool, u64) {
            match self.cell(row) {
                None => (true, 0),
                Some(other) => (false, normalize_component(self.physical, other)),
            }
        };
        let want = (false, target);
        let lo = partition_point(span.clone(), |row| key(row) < want);
        let hi = partition_point(lo..span.end, |row| key(row) <= want);
        lo..hi
    }

    /// Whether `cell` equals this row's value exactly, under the column's
    /// own comparison rather than the truncated sort key.
    pub fn cell_equals(&self, row: usize, cell: &[u8]) -> bool {
        match self.cell(row) {
            None => false,
            Some(other) => compare_cells(self.physical, other, cell).is_eq(),
        }
    }
}

/// First index in `range` where `pred` stops holding, for a predicate that
/// is true over a prefix of the range and false after it.
fn partition_point(range: std::ops::Range<usize>, pred: impl Fn(usize) -> bool) -> usize {
    let (mut lo, mut hi) = (range.start, range.end);
    while lo < hi {
        let mid = lo + (hi - lo) / 2;
        if pred(mid) {
            lo = mid + 1;
        } else {
            hi = mid;
        }
    }
    lo
}

/// What a sorted file's zone maps decided about a set of candidate keys,
/// before any column payload was read.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ZoneVerdict {
    /// The column carries no order the zone maps can use, so the caller
    /// must read it to answer
    Undecided,
    /// No zone can hold any candidate. The file holds none of them and
    /// nothing but its zone region was read
    NoMatch,
    /// Candidates can only live in this row range
    Span(std::ops::Range<usize>),
}

/// Reader over one data file of one lake table
pub struct LakeFileReader {
    reader: ZyrFileReader,
    row_count: usize,
    /// Column payload bytes this reader has pulled off disk, summed over every
    /// segment it decoded or evaluated a predicate against.
    ///
    /// Zone map reads are deliberately excluded. They are the metadata that
    /// decides skipping, not the data, and the heap columnar scan excludes its
    /// own header and zone reads for the same reason, so the two formats report
    /// one comparable quantity: bytes of column data the query had to touch
    bytes_read: AtomicU64,
}

impl LakeFileReader {
    /// Opens the data file for one partition id
    pub fn open(paths: &LakePaths, partition_id: u64) -> Result<Self, ZyronError> {
        Self::open_path(&paths.data_file(partition_id))
    }

    /// Opens a data file by path. A clustering pass reads back the
    /// candidates it staged, which live outside `data/` until the gate
    /// accepts them
    pub fn open_path(path: &std::path::Path) -> Result<Self, ZyronError> {
        let reader = ZyrFileReader::open(path)?;
        let row_count = reader.row_count() as usize;
        Ok(Self {
            reader,
            row_count,
            bytes_read: AtomicU64::new(0),
        })
    }

    pub fn row_count(&self) -> usize {
        self.row_count
    }

    /// Column payload bytes read through this reader so far. A scan reads it
    /// once when it is done with the file, so the per column adds never appear
    /// on a row path
    pub fn bytes_read(&self) -> u64 {
        self.bytes_read.load(Ordering::Relaxed)
    }

    /// Whether the file holds this column at all. False for a column the
    /// file predates, whose rows all read as NULL
    pub fn has_column(&self, column_id: u32) -> bool {
        self.reader.has_segment(column_id)
    }

    /// Rows a lowered filter admits, as a keep bitmask of ceil(rows/8)
    /// bytes, with no column decoded.
    ///
    /// Zone maps reject whole spans of rows from bounds alone, and what
    /// they cannot reject is answered on the encoded bytes. The mask is a
    /// superset of the matching rows, never a subset, so the exact row
    /// filter still decides what is returned. None means the filter
    /// decided nothing and every row stands
    pub fn rows_matching(&self, filter: &StoredFilter) -> Result<Option<Vec<u8>>, ZyronError> {
        rows_matching(filter, &FileEvidence { reader: self })
    }

    /// Whether this file's rows ascend by `column_id`.
    ///
    /// Only a range partition on the leading key produces that order. The
    /// multi-dimensional curves place rows by every key at once, so their
    /// leading column is not monotonic and the writer records no order for
    /// them
    pub fn is_sorted_by(&self, column_id: u32) -> bool {
        self.reader.sort_order() == SortOrder::Asc
            && self.reader.primary_key_column_id() == column_id
    }

    /// The rows that can hold any of `cells`, decided from one column's
    /// zone maps with no payload read.
    ///
    /// A zone covers ZONE_MAP_BATCH_SIZE rows and records their bounds, so
    /// a file whose own bounds admit a key can still have no zone that
    /// does. Rejecting it here costs the zone region, which is sized by
    /// the row count rather than by the data.
    ///
    /// Sortedness is not required. Zones are examined in order rather than
    /// bisected, so an unordered file answers correctly and simply returns
    /// a wider span. `cells` may be in any order
    pub fn zone_span_for_cells(
        &self,
        column: &LakeColumn,
        cells: &[&[u8]],
    ) -> Result<ZoneVerdict, ZyronError> {
        if cells.is_empty() {
            return Ok(ZoneVerdict::NoMatch);
        }
        let physical = column.physical_type_id();
        let value_size = physical.fixed_size().unwrap_or(0);
        // A slot holds a 32-byte prefix for a variable-length value and the
        // whole value otherwise. Anything wider carries no usable bound
        let usable = value_size == 0 || (1..=STAT_VALUE_SIZE).contains(&value_size);
        if !usable || !self.reader.has_segment(column.id) {
            return Ok(ZoneVerdict::Undecided);
        }
        let order = slot_order(physical);
        if order == SlotOrder::Lexicographic && value_size != 0 {
            // A fixed-width byte family compares by its bytes in both
            // directions, which is neither shape the slots record
            return Ok(ZoneVerdict::Undecided);
        }
        let (_, zones) = self
            .reader
            .read_segment_metadata(column.id, self.row_count)?;
        if zones.is_empty() {
            return Ok(ZoneVerdict::Undecided);
        }
        let admits = |zone: &ZoneMapEntry, cell: &[u8]| -> bool {
            compare_value_to_slot(cell, &zone.min_value, value_size, order).is_ge()
                && compare_value_to_slot(cell, &zone.max_value, value_size, order).is_le()
        };
        let mut first: Option<usize> = None;
        let mut last: usize = 0;
        for (z, zone) in zones.iter().enumerate() {
            if cells.iter().any(|cell| admits(zone, cell)) {
                first.get_or_insert(z);
                last = z;
            }
        }
        let Some(first) = first else {
            return Ok(ZoneVerdict::NoMatch);
        };
        let batch = ZONE_MAP_BATCH_SIZE as usize;
        let start = first * batch;
        let end = ((last + 1) * batch).min(self.row_count);
        Ok(ZoneVerdict::Span(start..end))
    }

    /// Decodes one column. A column the file predates, added to the
    /// schema after the file was written, decodes as all NULL
    pub fn read_column(&self, col: &LakeColumn) -> Result<DecodedColumn, ZyronError> {
        self.read_column_range(col, 0, self.row_count)
    }

    /// Decodes rows `start..end` of one column, addressing them by their
    /// ordinal in the file exactly as the full decode does.
    ///
    /// A point read wants a handful of rows, and decoding the segment to
    /// reach them makes the cost of one row the cost of the column. The
    /// returned column answers `cell` for ordinals inside the range and
    /// None outside it, so a caller that holds absolute ordinals does not
    /// have to rebase them
    pub fn read_column_range(
        &self,
        col: &LakeColumn,
        start: usize,
        end: usize,
    ) -> Result<DecodedColumn, ZyronError> {
        let physical = col.physical_type_id();
        let value_size = physical.fixed_size().unwrap_or(0);
        let start = start.min(self.row_count);
        let end = end.clamp(start, self.row_count);
        if !self.reader.has_segment(col.id) {
            return Ok(DecodedColumn {
                column_id: col.id,
                physical,
                value_size,
                data: Vec::new(),
                null_bitmap: Vec::new(),
                row_count: end - start,
                base: start,
                all_null: true,
            });
        }
        self.bytes_read
            .fetch_add(self.reader.segment_bytes(col.id), Ordering::Relaxed);
        let (data, null_bitmap) =
            self.reader
                .decode_column_range(col.id, self.row_count, value_size, start, end)?;
        Ok(DecodedColumn {
            column_id: col.id,
            physical,
            value_size,
            data,
            null_bitmap,
            row_count: end - start,
            base: start,
            all_null: false,
        })
    }

    /// Decodes every column a set of predicates references. Ids missing
    /// from the schema are rejected, the planner never lowers them
    pub fn read_predicate_columns(
        &self,
        schema: &LakeSchema,
        predicates: &[&LakePredicate],
    ) -> Result<Vec<DecodedColumn>, ZyronError> {
        let mut ids: Vec<u32> = predicates
            .iter()
            .flat_map(|p| p.referenced_columns())
            .collect();
        ids.sort_unstable();
        ids.dedup();
        let mut columns = Vec::with_capacity(ids.len());
        for id in ids {
            let col = schema.column_by_id(id).ok_or_else(|| {
                ZyronError::Internal(format!("predicate references unknown column {}", id))
            })?;
            columns.push(self.read_column(col)?);
        }
        Ok(columns)
    }

    /// Rows that survive the file's delete predicates, as a keep bitmask
    /// of ceil(rows/8) bytes. A row is removed only when some predicate
    /// provably matches it
    pub fn delete_survivors(
        &self,
        schema: &LakeSchema,
        manifest: &ManifestFile,
        entry: &PartitionEntry,
    ) -> Result<Vec<u8>, ZyronError> {
        let mut keep = vec![0xFFu8; self.row_count.div_ceil(8)];
        trim_mask_tail(&mut keep, self.row_count);
        if entry.delete_predicate_ids.is_empty() {
            return Ok(keep);
        }
        let mut predicates = Vec::with_capacity(entry.delete_predicate_ids.len());
        for id in &entry.delete_predicate_ids {
            let del = manifest.predicate_by_id(*id).ok_or_else(|| {
                ZyronError::Internal(format!(
                    "partition {:#x} references missing delete predicate {}",
                    entry.partition_id, id
                ))
            })?;
            predicates.push(&del.predicate);
        }
        let columns = self.read_predicate_columns(schema, &predicates)?;
        // Compiled once per file, the row loop resolves no column ids
        let compiled: Vec<CompiledPredicate> = predicates
            .iter()
            .map(|p| CompiledPredicate::new(p, &columns))
            .collect();
        for row in 0..self.row_count {
            for pred in &compiled {
                if pred.evaluate(&columns, row) == Some(true) {
                    keep[row / 8] &= !(1 << (row % 8));
                    break;
                }
            }
        }
        Ok(keep)
    }
}

/// Zeroes mask bits past the last real row
fn trim_mask_tail(mask: &mut [u8], row_count: usize) {
    let tail = row_count % 8;
    if tail != 0 {
        if let Some(last) = mask.last_mut() {
            *last &= (1u8 << tail) - 1;
        }
    }
}

/// A predicate resolved once against a decoded column slice. Column ids
/// became direct indices and NULL-literal comparisons collapsed to a
/// constant, so the per-row walk searches and re-inspects nothing. A
/// column absent from the slice keeps index None and reads as NULL,
/// matching `evaluate_row`
pub struct CompiledPredicate<'a> {
    node: CompiledNode<'a>,
}

enum CompiledNode<'a> {
    /// A comparison against a NULL literal, unknown for every row
    Unknown,
    Compare {
        col: Option<usize>,
        op: CompareOp,
        value: &'a LakeValue,
    },
    IsNull {
        col: Option<usize>,
    },
    IsNotNull {
        col: Option<usize>,
    },
    In {
        col: Option<usize>,
        values: &'a [LakeValue],
    },
    And(Vec<CompiledNode<'a>>),
    Or(Vec<CompiledNode<'a>>),
    Not(Box<CompiledNode<'a>>),
}

impl<'a> CompiledPredicate<'a> {
    /// Resolves every column reference in the predicate to its index in
    /// `columns`. Callers that evaluate many rows compile once and reuse
    pub fn new(predicate: &'a LakePredicate, columns: &[DecodedColumn]) -> Self {
        Self {
            node: compile_node(predicate, columns),
        }
    }

    /// Three-valued evaluation of one row. None is unknown, the SQL
    /// outcome of any comparison touching NULL. `columns` must be the
    /// slice this predicate was compiled against
    pub fn evaluate(&self, columns: &[DecodedColumn], row: usize) -> Option<bool> {
        evaluate_node(&self.node, columns, row)
    }
}

fn compile_node<'a>(predicate: &'a LakePredicate, columns: &[DecodedColumn]) -> CompiledNode<'a> {
    let find = |id: u32| columns.iter().position(|c| c.column_id == id);
    match predicate {
        LakePredicate::Compare {
            column_id,
            op,
            value,
        } => {
            if matches!(value, LakeValue::Null) {
                CompiledNode::Unknown
            } else {
                CompiledNode::Compare {
                    col: find(*column_id),
                    op: *op,
                    value,
                }
            }
        }
        LakePredicate::IsNull { column_id } => CompiledNode::IsNull {
            col: find(*column_id),
        },
        LakePredicate::IsNotNull { column_id } => CompiledNode::IsNotNull {
            col: find(*column_id),
        },
        LakePredicate::In { column_id, values } => CompiledNode::In {
            col: find(*column_id),
            values,
        },
        LakePredicate::And(children) => {
            CompiledNode::And(children.iter().map(|c| compile_node(c, columns)).collect())
        }
        LakePredicate::Or(children) => {
            CompiledNode::Or(children.iter().map(|c| compile_node(c, columns)).collect())
        }
        LakePredicate::Not(inner) => CompiledNode::Not(Box::new(compile_node(inner, columns))),
    }
}

fn evaluate_node(node: &CompiledNode<'_>, columns: &[DecodedColumn], row: usize) -> Option<bool> {
    match node {
        CompiledNode::Unknown => None,
        CompiledNode::Compare { col, op, value } => {
            let c = &columns[(*col)?];
            let cell = c.cell(row)?;
            let ord = compare_cell_to_value(c.physical, cell, value)?;
            Some(match op {
                CompareOp::Eq => ord.is_eq(),
                CompareOp::NotEq => ord.is_ne(),
                CompareOp::Lt => ord.is_lt(),
                CompareOp::LtEq => ord.is_le(),
                CompareOp::Gt => ord.is_gt(),
                CompareOp::GtEq => ord.is_ge(),
            })
        }
        CompiledNode::IsNull { col } => match col {
            Some(i) => Some(columns[*i].cell(row).is_none()),
            None => Some(true),
        },
        CompiledNode::IsNotNull { col } => match col {
            Some(i) => Some(columns[*i].cell(row).is_some()),
            None => Some(false),
        },
        CompiledNode::In { col, values } => {
            let c = &columns[(*col)?];
            let cell = c.cell(row)?;
            let mut unknown = false;
            for v in values.iter() {
                if matches!(v, LakeValue::Null) {
                    unknown = true;
                    continue;
                }
                match compare_cell_to_value(c.physical, cell, v) {
                    Some(ord) if ord.is_eq() => return Some(true),
                    Some(_) => {}
                    None => unknown = true,
                }
            }
            if unknown { None } else { Some(false) }
        }
        CompiledNode::And(children) => {
            let mut unknown = false;
            for child in children {
                match evaluate_node(child, columns, row) {
                    Some(false) => return Some(false),
                    None => unknown = true,
                    Some(true) => {}
                }
            }
            if unknown { None } else { Some(true) }
        }
        CompiledNode::Or(children) => {
            let mut unknown = false;
            for child in children {
                match evaluate_node(child, columns, row) {
                    Some(true) => return Some(true),
                    None => unknown = true,
                    Some(false) => {}
                }
            }
            if unknown { None } else { Some(false) }
        }
        CompiledNode::Not(inner) => evaluate_node(inner, columns, row).map(|b| !b),
    }
}

/// Three-valued predicate evaluation over one row. None is unknown, the
/// SQL outcome of any comparison touching NULL. Columns the slice does
/// not carry read as NULL. Compiles per call, a loop over many rows
/// should compile a `CompiledPredicate` once instead
pub fn evaluate_row(
    predicate: &LakePredicate,
    columns: &[DecodedColumn],
    row: usize,
) -> Option<bool> {
    CompiledPredicate::new(predicate, columns).evaluate(columns, row)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::LakeValue;
    use crate::manifest::DeletePredicate;
    use crate::writer::{ColumnData, WriteRequest, write_data_file};
    use std::collections::BTreeMap;

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

    fn int_cells(values: &[Option<i64>]) -> Vec<Option<Vec<u8>>> {
        values
            .iter()
            .map(|v| v.map(|x| x.to_le_bytes().to_vec()))
            .collect()
    }

    fn str_cells(values: &[Option<&str>]) -> Vec<Option<Vec<u8>>> {
        values
            .iter()
            .map(|v| v.map(|s| s.as_bytes().to_vec()))
            .collect()
    }

    fn write_sample(paths: &LakePaths, partition_id: u64) -> PartitionEntry {
        let columns = vec![
            ColumnData {
                column_id: 0,
                cells: int_cells(&[Some(30), Some(10), Some(20)]),
            },
            ColumnData {
                column_id: 1,
                cells: str_cells(&[Some("carol"), Some("alice"), None]),
            },
        ];
        write_data_file(
            paths,
            &schema(),
            &WriteRequest {
                partition_id,
                columns: &columns,
                sort_keys: &[0],
                sort_strategies: &[],
                cluster_spec_id: 0,
                table_id: 7,
                bloom_columns: &[],
                index_id: None,
            },
        )
        .expect("write")
    }

    #[test]
    fn test_write_read_roundtrip_with_sort_and_stats() {
        let dir = tempfile::tempdir().expect("tempdir");
        let paths = LakePaths::new(dir.path(), 7);
        let entry = write_sample(&paths, 0x1);

        assert_eq!(entry.row_count, 3);
        let id_stats = entry.stats_for(0).expect("id stats");
        assert_eq!(id_stats.bounds.min, Some(LakeValue::Int(10)));
        assert_eq!(id_stats.bounds.max, Some(LakeValue::Int(30)));
        assert_eq!(id_stats.bounds.null_count, 0);
        let name_stats = entry.stats_for(1).expect("name stats");
        assert_eq!(name_stats.bounds.min, Some(LakeValue::Str("alice".into())));
        assert_eq!(name_stats.bounds.max, Some(LakeValue::Str("carol".into())));
        assert_eq!(name_stats.bounds.null_count, 1);

        let reader = LakeFileReader::open(&paths, 0x1).expect("open");
        assert_eq!(reader.row_count(), 3);
        let s = schema();
        let id_col = reader.read_column(&s.columns[0]).expect("id");
        let name_col = reader.read_column(&s.columns[1]).expect("name");
        // Sorted ascending by id, the null name rides with id 20
        let ids: Vec<i64> = (0..3)
            .map(|r| {
                let mut a = [0u8; 8];
                a.copy_from_slice(id_col.cell(r).expect("cell"));
                i64::from_le_bytes(a)
            })
            .collect();
        assert_eq!(ids, vec![10, 20, 30]);
        assert_eq!(name_col.cell(0), Some(&b"alice"[..]));
        assert_eq!(name_col.cell(1), None);
        assert_eq!(name_col.cell(2), Some(&b"carol"[..]));
    }

    /// One file holding two disjoint runs, so its own bounds span the gap
    /// between them and no zone does
    fn write_gapped(paths: &LakePaths, partition_id: u64) {
        let mut ids: Vec<Option<i64>> = (0..2048).map(Some).collect();
        ids.extend((1_000_000..1_002_048).map(Some));
        let names: Vec<Option<&str>> = ids.iter().map(|_| Some("x")).collect();
        let columns = vec![
            ColumnData {
                column_id: 0,
                cells: int_cells(&ids),
            },
            ColumnData {
                column_id: 1,
                cells: str_cells(&names),
            },
        ];
        write_data_file(
            paths,
            &schema(),
            &WriteRequest {
                partition_id,
                columns: &columns,
                sort_keys: &[0],
                sort_strategies: &[],
                cluster_spec_id: 0,
                table_id: 7,
                bloom_columns: &[],
                index_id: None,
            },
        )
        .expect("write");
    }

    #[test]
    fn test_zone_maps_reject_a_value_the_file_bounds_admit() {
        let dir = tempfile::tempdir().expect("tempdir");
        let paths = LakePaths::new(dir.path(), 7);
        write_gapped(&paths, 0x2);
        let reader = LakeFileReader::open(&paths, 0x2).expect("open");
        let s = schema();

        // A value inside the file's overall range but inside no zone's
        let gap = 500_000i64.to_le_bytes();
        assert_eq!(
            reader
                .zone_span_for_cells(&s.columns[0], &[&gap[..]])
                .expect("zones"),
            ZoneVerdict::NoMatch,
            "no zone holds it, so the file holds no such row"
        );
        assert_eq!(reader.bytes_read(), 0, "no column payload was read");

        // A value a zone does hold narrows to that zone rather than to the
        // whole file
        let held = 1_000_500i64.to_le_bytes();
        match reader
            .zone_span_for_cells(&s.columns[0], &[&held[..]])
            .expect("zones")
        {
            ZoneVerdict::Span(span) => {
                assert!(span.contains(&2548), "the row carrying it is inside");
                assert!(
                    span.len() < reader.row_count(),
                    "spanned {} of {} rows",
                    span.len(),
                    reader.row_count()
                );
            }
            other => panic!("expected a narrowed span, got {other:?}"),
        }
    }

    #[test]
    fn test_sort_key_range_finds_a_value_without_walking_the_file() {
        let dir = tempfile::tempdir().expect("tempdir");
        let paths = LakePaths::new(dir.path(), 7);
        write_gapped(&paths, 0x3);
        let reader = LakeFileReader::open(&paths, 0x3).expect("open");
        let s = schema();
        assert!(reader.is_sorted_by(0), "ascending on the sort key");
        let column = reader.read_column(&s.columns[0]).expect("id");

        let target = 1_500i64.to_le_bytes();
        let range = column.sort_key_range_in(&target, 0..reader.row_count());
        assert_eq!(range, 1500..1501);
        assert!(column.cell_equals(1500, &target));

        // A value the file does not hold selects nothing, in the gap and
        // past the end alike
        for absent in [500_000i64, 9_000_000] {
            let cell = absent.to_le_bytes();
            assert!(
                column
                    .sort_key_range_in(&cell, 0..reader.row_count())
                    .is_empty(),
                "{absent} is not stored"
            );
        }
    }

    #[test]
    fn test_schema_evolved_column_reads_all_null() {
        let dir = tempfile::tempdir().expect("tempdir");
        let paths = LakePaths::new(dir.path(), 7);
        write_sample(&paths, 0x1);

        let reader = LakeFileReader::open(&paths, 0x1).expect("open");
        let added = LakeColumn {
            id: 2,
            name: "added".into(),
            type_id: TypeId::Float64,
            nullable: true,
            fractional_digits: None,
            tz_offset_secs: None,
            max_length: None,
            default_expr: None,
        };
        let col = reader.read_column(&added).expect("absent column");
        for r in 0..3 {
            assert_eq!(col.cell(r), None);
        }
    }

    #[test]
    fn test_delete_survivors_keep_unknown_rows() {
        let dir = tempfile::tempdir().expect("tempdir");
        let paths = LakePaths::new(dir.path(), 7);
        let mut entry = write_sample(&paths, 0x1);
        entry.delete_predicate_ids = vec![5];

        // Deletes rows whose name is alice. The null-name row is unknown
        // and must survive
        let manifest = ManifestFile {
            snapshot_id: 2,
            parent_snapshot_id: 1,
            timestamp_us: 0,
            schema: schema(),
            cluster_spec: crate::ClusterSpec::none(),
            entries: vec![entry.clone()],
            delete_predicates: vec![DeletePredicate {
                id: 5,
                sql: "name = 'alice'".into(),
                predicate: LakePredicate::Compare {
                    column_id: 1,
                    op: CompareOp::Eq,
                    value: LakeValue::Str("alice".into()),
                },
                created_version: 2,
            }],
            properties: BTreeMap::new(),
            indexes: Vec::new(),
            index_files: Vec::new(),
        };
        let reader = LakeFileReader::open(&paths, 0x1).expect("open");
        let keep = reader
            .delete_survivors(&schema(), &manifest, &entry)
            .expect("survivors");
        // Row order after sort: alice(10), null(20), carol(30)
        assert_eq!(keep[0] & 0b001, 0, "alice row deleted");
        assert_ne!(keep[0] & 0b010, 0, "unknown row kept");
        assert_ne!(keep[0] & 0b100, 0, "carol row kept");
    }

    #[test]
    fn test_not_null_enforced_at_write() {
        let dir = tempfile::tempdir().expect("tempdir");
        let paths = LakePaths::new(dir.path(), 7);
        let columns = vec![
            ColumnData {
                column_id: 0,
                cells: int_cells(&[Some(1), None]),
            },
            ColumnData {
                column_id: 1,
                cells: str_cells(&[Some("a"), Some("b")]),
            },
        ];
        let err = write_data_file(
            &paths,
            &schema(),
            &WriteRequest {
                partition_id: 0x1,
                columns: &columns,
                sort_keys: &[],
                sort_strategies: &[],
                cluster_spec_id: 0,
                table_id: 7,
                bloom_columns: &[],
                index_id: None,
            },
        )
        .expect_err("must reject");
        assert!(err.to_string().contains("non-nullable"));
    }

    #[test]
    fn test_row_predicate_three_valued_logic() {
        let dir = tempfile::tempdir().expect("tempdir");
        let paths = LakePaths::new(dir.path(), 7);
        write_sample(&paths, 0x1);
        let reader = LakeFileReader::open(&paths, 0x1).expect("open");
        let s = schema();
        let cols = vec![
            reader.read_column(&s.columns[0]).expect("id"),
            reader.read_column(&s.columns[1]).expect("name"),
        ];
        // Row 1 has a null name
        let name_eq = LakePredicate::Compare {
            column_id: 1,
            op: CompareOp::Eq,
            value: LakeValue::Str("alice".into()),
        };
        assert_eq!(evaluate_row(&name_eq, &cols, 0), Some(true));
        assert_eq!(evaluate_row(&name_eq, &cols, 1), None);
        assert_eq!(evaluate_row(&name_eq, &cols, 2), Some(false));

        // NOT propagates unknown
        let not_eq = LakePredicate::Not(Box::new(name_eq.clone()));
        assert_eq!(evaluate_row(&not_eq, &cols, 1), None);

        // OR with a true arm resolves despite the unknown
        let or = LakePredicate::Or(vec![
            name_eq.clone(),
            LakePredicate::Compare {
                column_id: 0,
                op: CompareOp::Eq,
                value: LakeValue::Int(20),
            },
        ]);
        assert_eq!(evaluate_row(&or, &cols, 1), Some(true));

        // IN with null cell is unknown
        let in_pred = LakePredicate::In {
            column_id: 1,
            values: vec![LakeValue::Str("zoe".into())],
        };
        assert_eq!(evaluate_row(&in_pred, &cols, 1), None);
        assert_eq!(evaluate_row(&in_pred, &cols, 0), Some(false));
    }
}
