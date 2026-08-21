//! Data file writer.
//!
//! Turns one sorted row batch into one immutable .zyr file at the lake's
//! partition-addressed path and returns the manifest entry describing it.
//! Lake files carry only user columns, visibility is by log version, so
//! there are no MVCC system columns and no patch store.
//!
//! Statistics are exact. Min, max and null count come off the segment
//! build, which orders and counts the column to lay its zone maps and
//! bloom down, so the manifest entry is filled from a pass the write was
//! making anyway rather than a second one. A declared bloom column's filter
//! is copied into the entry as well, so pruning probes it with zero IO,
//! while an undeclared column's filter stays in the segment where it costs
//! the manifest nothing

use std::path::Path;

use zyron_common::{TypeId, ZyronError};
use zyron_storage::columnar::{
    BloomPolicy, ColumnSegment, SegmentOptions, SortOrder, ZyrFileHeader, ZyrFileWriter,
};

use crate::cells::{CellFamily, cell_family, cell_to_value, compare_cells};
use crate::curve::{normalize_component, ordering_key_into};
use crate::manifest::ClusterStrategy;
use crate::manifest::{ColumnStatsEntry, PartitionEntry};
use crate::paths::{LakePaths, data_file_name, index_file_name};
use crate::predicate::ColumnBounds;
use crate::schema::LakeSchema;

/// One column of a row batch, cells in row order, packed into a single
/// allocation.
///
/// A column whose cells are all one width stores them at that stride with
/// null slots zero-filled, which makes the buffer exactly what a segment
/// build encodes, so the build reads it instead of packing its own copy.
/// Any other column stores its bytes end to end and addresses them through
/// `ends`.
///
/// The shape this replaced held one heap allocation per cell. Building a
/// ten thousand row Int64 column that way measured 153.8us against 2.5us
/// packed, and walking it to pack, bound and count nulls measured 14.9us
/// against 1.9us, because a cell was a pointer into its own allocation
/// rather than the next eight bytes
#[derive(Debug, Clone, Default)]
pub struct ColumnData {
    pub column_id: u32,
    /// Cell bytes back to back
    values: Vec<u8>,
    /// End offset of each cell, empty while the width is fixed
    ends: Vec<usize>,
    /// Bytes per cell, zero once the cells stopped agreeing on a width
    width: usize,
    /// One bit per cell, set when that cell is null. Empty while none is
    nulls: Vec<u8>,
    /// Nulls pushed before any width was known, all of them still waiting
    /// for the first non-null cell to say how wide a slot they occupy
    leading_nulls: usize,
    rows: usize,
}

impl ColumnData {
    /// An empty column ready to take `rows` cells of `width` bytes each.
    ///
    /// A width of zero, or a cell that does not match the width, puts the
    /// column on the offset-addressed layout, so a caller that does not
    /// know the width in advance passes zero and loses only the segment
    /// build's zero-copy read
    pub fn with_capacity(column_id: u32, width: usize, rows: usize) -> Self {
        Self {
            column_id,
            values: Vec::with_capacity(width.saturating_mul(rows)),
            ends: if width == 0 {
                Vec::with_capacity(rows)
            } else {
                Vec::new()
            },
            width,
            nulls: Vec::new(),
            leading_nulls: 0,
            rows: 0,
        }
    }

    /// Builds from owned per-cell buffers.
    ///
    /// The width is taken from the cells: a column whose non-null cells are
    /// all one length is stored at that stride, anything else through
    /// offsets. Only the segment build's zero-copy read depends on which,
    /// and it asks for the width it expects rather than trusting this
    pub fn from_cells(column_id: u32, cells: Vec<Option<Vec<u8>>>) -> Self {
        let mut width = None;
        let mut uniform = true;
        for cell in cells.iter().flatten() {
            match width {
                None => width = Some(cell.len()),
                Some(w) if w == cell.len() => {}
                Some(_) => {
                    uniform = false;
                    break;
                }
            }
        }
        let width = if uniform { width.unwrap_or(0) } else { 0 };
        let mut out = Self::with_capacity(column_id, width, cells.len());
        for cell in &cells {
            out.push(cell.as_deref());
        }
        out
    }

    /// Appends one cell, null when None
    pub fn push(&mut self, cell: Option<&[u8]>) {
        match cell {
            None => {
                self.mark_null();
                if self.width > 0 {
                    self.values.resize(self.values.len() + self.width, 0);
                } else if self.undecided() {
                    // No width has been named yet, so how many bytes this
                    // slot occupies is not known. Counted here and settled
                    // by the first non-null cell
                    self.leading_nulls += 1;
                } else {
                    self.ends.push(self.values.len());
                }
            }
            Some(value) => {
                // A column built without a width takes one from its first
                // non-null cell and back-fills the nulls ahead of it. A
                // producer that does not know the width up front would
                // otherwise store every column through offsets and lose the
                // segment build's zero-copy read without saying so, and a
                // column whose cells then disagree spills anyway
                if self.undecided() {
                    if value.is_empty() {
                        // A zero-byte cell names no width, so the column
                        // stays on offsets and every null ahead of it
                        // becomes a cell of no bytes
                        self.flush_leading_nulls();
                    } else {
                        self.width = value.len();
                        self.ends = Vec::new();
                        self.values.resize(self.leading_nulls * self.width, 0);
                        self.leading_nulls = 0;
                    }
                }
                if self.width > 0 && value.len() != self.width {
                    self.spill_to_offsets();
                }
                self.values.extend_from_slice(value);
                if self.width == 0 {
                    self.ends.push(self.values.len());
                }
            }
        }
        self.rows += 1;
    }

    /// Whether every cell so far is a null pushed before any width was
    /// known, which is the one state the next non-null cell can still
    /// choose a layout from
    #[inline]
    fn undecided(&self) -> bool {
        self.width == 0 && self.leading_nulls == self.rows
    }

    /// Puts the waiting nulls on the offset layout, where each one is a
    /// cell of no bytes ending where the buffer still begins
    fn flush_leading_nulls(&mut self) {
        self.ends.resize(self.ends.len() + self.leading_nulls, 0);
        self.leading_nulls = 0;
    }

    /// Moves a fixed-width column onto the offset layout, which is what a
    /// cell disagreeing with the width forces. The cells already stored are
    /// all one width, so their ends are that width apart
    fn spill_to_offsets(&mut self) {
        debug_assert!(self.width > 0);
        self.ends = (1..=self.rows).map(|row| row * self.width).collect();
        self.width = 0;
    }

    /// Records the cell about to be pushed as null
    fn mark_null(&mut self) {
        let byte = self.rows / 8;
        if byte >= self.nulls.len() {
            self.nulls.resize(byte + 1, 0);
        }
        self.nulls[byte] |= 1 << (self.rows % 8);
    }

    /// Whether the cell at `row` is null
    #[inline]
    pub fn is_null(&self, row: usize) -> bool {
        self.nulls
            .get(row / 8)
            .is_some_and(|byte| byte & (1 << (row % 8)) != 0)
    }

    /// Cells in the column
    #[inline]
    pub fn len(&self) -> usize {
        self.rows
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.rows == 0
    }

    /// Bytes per cell, zero when the cells are addressed through offsets
    #[inline]
    pub fn width(&self) -> usize {
        self.width
    }

    /// The bytes of one cell, whether or not it is null
    #[inline]
    fn span(&self, row: usize) -> &[u8] {
        if self.width > 0 {
            &self.values[row * self.width..(row + 1) * self.width]
        } else {
            let start = if row == 0 { 0 } else { self.ends[row - 1] };
            &self.values[start..self.ends[row]]
        }
    }

    /// One cell, None when it is null
    #[inline]
    pub fn cell(&self, row: usize) -> Option<&[u8]> {
        debug_assert!(
            row < self.rows,
            "row {row} is past the column's {} cells",
            self.rows
        );
        if row >= self.rows || self.is_null(row) {
            return None;
        }
        Some(self.span(row))
    }

    /// Every cell in row order
    pub fn iter(&self) -> ColumnCells<'_> {
        ColumnCells {
            column: self,
            row: 0,
        }
    }

    /// The packed buffer a fixed-width segment build encodes, with null
    /// slots zero-filled, or None when this column is not stored at that
    /// width.
    ///
    /// The caller passes the width its schema says the column has rather
    /// than reading one off the data, so a variable-width column whose
    /// cells happen to agree on a length is not mistaken for a fixed one
    pub fn packed_at(&self, width: usize) -> Option<&[u8]> {
        (width > 0 && self.width == width).then_some(self.values.as_slice())
    }

    /// The null bitmap, one bit per cell set when the cell is null. Empty
    /// when the column has no nulls at all
    pub fn null_bitmap(&self) -> &[u8] {
        &self.nulls
    }

    /// The first null cell, which is what a non-nullable column is checked
    /// against
    pub fn first_null(&self) -> Option<usize> {
        if self.nulls.is_empty() {
            return None;
        }
        (0..self.rows).find(|&row| self.is_null(row))
    }
}

/// Cells of one column in row order
pub struct ColumnCells<'a> {
    column: &'a ColumnData,
    row: usize,
}

impl<'a> Iterator for ColumnCells<'a> {
    type Item = Option<&'a [u8]>;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        if self.row >= self.column.rows {
            return None;
        }
        let cell = if self.column.is_null(self.row) {
            None
        } else {
            Some(self.column.span(self.row))
        };
        self.row += 1;
        Some(cell)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let left = self.column.rows - self.row;
        (left, Some(left))
    }
}

impl ExactSizeIterator for ColumnCells<'_> {}

impl<'a> IntoIterator for &'a ColumnData {
    type Item = Option<&'a [u8]>;
    type IntoIter = ColumnCells<'a>;

    fn into_iter(self) -> ColumnCells<'a> {
        self.iter()
    }
}

/// One file's worth of rows plus the layout it is written under
#[derive(Debug)]
pub struct WriteRequest<'a> {
    /// Data file identity, must be unallocated in the target manifest
    pub partition_id: u64,
    /// Every schema column exactly once, any order
    pub columns: &'a [ColumnData],
    /// Column ids to sort by, in declaration order, nulls last. Empty
    /// writes rows in arrival order
    pub sort_keys: &'a [u32],
    /// Ordering curve each key column asks for, parallel to `sort_keys`.
    /// A shorter list leaves the remaining keys on RangePartition, which is
    /// plain ascending order
    pub sort_strategies: &'a [ClusterStrategy],
    /// Cluster spec the layout was chosen by, recorded per file so a past
    /// version reads its own layout metadata
    pub cluster_spec_id: u32,
    pub table_id: u64,
    /// Columns a table property declared a bloom filter for, forced whatever
    /// their cardinality says. Every other column keeps the cardinality
    /// heuristic
    pub bloom_columns: &'a [u32],
    /// The index this file belongs to, which decides its name. None writes
    /// a table data file, Some writes one file of that index. Both kinds
    /// sit in `data/` and both are `.zyr`, so the name is what separates
    /// them and a sweep never has to open a file to classify it
    pub index_id: Option<u32>,
}

impl WriteRequest<'_> {
    /// The name this request's file takes, wherever it is written
    fn file_name(&self) -> String {
        match self.index_id {
            None => data_file_name(self.partition_id),
            Some(index_id) => index_file_name(index_id, self.partition_id),
        }
    }
}

/// One written file and the row order it was written in.
///
/// `order[ordinal]` is the input row that landed at that ordinal. Index
/// maintenance needs it to address rows the write just placed, and having
/// it here is what keeps an index delta from re-reading a file the writer
/// has this instant finished producing
#[derive(Debug)]
pub struct WrittenFile {
    pub entry: PartitionEntry,
    pub order: Vec<usize>,
}

/// Writes one data file into the table's `data/` directory and returns
/// its manifest entry. The file appears atomically, the storage writer
/// stages to a temp name and renames on finalize. Nothing references the
/// file until a commit adds the entry
pub fn write_data_file(
    paths: &LakePaths,
    schema: &LakeSchema,
    req: &WriteRequest<'_>,
) -> Result<PartitionEntry, ZyronError> {
    write_data_file_at(&paths.data_dir(), schema, req).map(|w| w.entry)
}

/// Writes one data file and keeps the row order it produced, for a caller
/// that has to address the rows it just wrote
pub fn write_data_file_ordered(
    paths: &LakePaths,
    schema: &LakeSchema,
    req: &WriteRequest<'_>,
) -> Result<WrittenFile, ZyronError> {
    write_data_file_at(&paths.data_dir(), schema, req)
}

/// Writes one data file into an arbitrary directory under its partition
/// name. A clustering pass writes candidates into `_staging/pass_<id>/`
/// and renames them into `data/` only once the gate accepts them, which
/// is what keeps a rejected pass from ever touching the active set
pub fn write_data_file_at(
    dir: &Path,
    schema: &LakeSchema,
    req: &WriteRequest<'_>,
) -> Result<WrittenFile, ZyronError> {
    let row_count = validate_batch(schema, req.columns)?;
    let order = stored_order(schema, req, row_count)?;
    // An unclustered write stores rows where they arrived, which is what
    // lets a fixed-width column go to the segment build as the buffer the
    // batch already holds rather than a copy of it
    let in_arrival_order = order.iter().enumerate().all(|(slot, &row)| slot == row);
    // The curve the leading key declared, which decides whether the header
    // may claim the rows ascend by it
    let leading_strategy = req
        .sort_strategies
        .first()
        .copied()
        .unwrap_or(ClusterStrategy::RangePartition);

    std::fs::create_dir_all(dir)?;
    let path = dir.join(req.file_name());
    if path.exists() {
        return Err(ZyronError::Internal(format!(
            "data file for partition {:#x} already exists at {}",
            req.partition_id,
            path.display()
        )));
    }
    let header = ZyrFileHeader {
        format_version: zyron_storage::columnar::ZYR_FORMAT_VERSION,
        column_count: schema.columns.len() as u32,
        row_count: row_count as u64,
        table_id: req.table_id,
        xmin_range_lo: 0,
        xmin_range_hi: 0,
        xmax_range_lo: 0,
        xmax_range_hi: 0,
        primary_key_column_id: req.sort_keys.first().copied().unwrap_or(0),
        // Only a range partition orders rows ascending by the leading key.
        // The other curves consume every key at once, so rows land in curve
        // order and the leading column is not monotonic across the file. A
        // header claiming Asc there would let an ordered lookup binary
        // search a column that is not sorted and miss rows that are present
        sort_order: match (req.sort_keys.is_empty(), leading_strategy) {
            (true, _) => SortOrder::None,
            (false, ClusterStrategy::RangePartition) => SortOrder::Asc,
            (false, _) => SortOrder::None,
        },
        // Filled in by finalize once the index has a position
        segment_index_offset: 0,
        segment_index_size: 0,
    };
    let mut writer = ZyrFileWriter::create(&path, header)?;

    // Columns are written and described in schema order so the manifest
    // stats arrive sorted by column id after the id sort below
    let mut column_stats = Vec::with_capacity(schema.columns.len());
    for col in &schema.columns {
        let data = req
            .columns
            .iter()
            .find(|c| c.column_id == col.id)
            .ok_or_else(|| ZyronError::Internal(format!("column \"{}\" has no data", col.name)))?;
        let physical = col.physical_type_id();
        let value_size = physical.fixed_size().unwrap_or(0);

        // The whole column is materialized here already, so the encoding
        // is picked by trial encoding every row rather than a prefix, and
        // the distinct count is the whole column's rather than a capped one
        let declared_bloom = req.bloom_columns.contains(&col.id);
        let options = SegmentOptions {
            bloom: if declared_bloom {
                BloomPolicy::Force
            } else {
                BloomPolicy::Auto
            },
            exact_encoding: true,
            distinct_sketch: true,
        };

        // The column in stored order, as the buffer the segment build
        // encodes. A batch already holds a fixed-width column at that
        // stride, so an unsorted write hands its buffer over untouched and
        // a sorted one gathers into a new buffer at the same stride.
        // Building a pointer per cell instead makes every later pass chase
        // one, and the segment build then packs a buffer of its own out of
        // them
        // Held so the stored-order cells outlive the borrow the statistics
        // take of them
        let mut gathered: Vec<u8> = Vec::new();
        let mut gathered_nulls: Vec<u8> = Vec::new();
        let mut views: Vec<Option<&[u8]>> = Vec::new();
        let (segment, cells) = match data.packed_at(value_size) {
            Some(values) if in_arrival_order => (
                ColumnSegment::build_packed(
                    col.id,
                    physical,
                    value_size,
                    values,
                    data.null_bitmap(),
                    row_count,
                    options,
                )?,
                StoredCells::Packed {
                    values,
                    nulls: data.null_bitmap(),
                    width: value_size,
                    rows: row_count,
                },
            ),
            Some(values) => {
                // SAFETY: the loop below writes every slot, a null one
                // zeroed and the rest copied, before anything reads the
                // buffer. Zeroing up front would memset the whole column
                // only to overwrite almost all of it
                let span = row_count * value_size;
                #[allow(clippy::uninit_vec)]
                {
                    gathered.reserve_exact(span);
                    unsafe { gathered.set_len(span) };
                }
                for (slot, &row) in order.iter().enumerate() {
                    let out = slot * value_size..(slot + 1) * value_size;
                    if data.is_null(row) {
                        if gathered_nulls.is_empty() {
                            gathered_nulls = vec![0u8; row_count.div_ceil(8)];
                        }
                        gathered_nulls[slot / 8] |= 1 << (slot % 8);
                        // A null slot carries a deterministic zero
                        gathered[out].fill(0);
                        continue;
                    }
                    gathered[out]
                        .copy_from_slice(&values[row * value_size..(row + 1) * value_size]);
                }
                (
                    ColumnSegment::build_packed(
                        col.id,
                        physical,
                        value_size,
                        &gathered,
                        &gathered_nulls,
                        row_count,
                        options,
                    )?,
                    StoredCells::Packed {
                        values: &gathered,
                        nulls: &gathered_nulls,
                        width: value_size,
                        rows: row_count,
                    },
                )
            }
            // A variable-width column, or one whose cells did not agree on
            // a width, is addressed a cell at a time
            None => {
                views.reserve_exact(row_count);
                for &row in &order {
                    views.push(data.cell(row));
                }
                (
                    ColumnSegment::build_with_options(
                        col.id, physical, value_size, &views, options,
                    )?,
                    StoredCells::Views(&views),
                )
            }
        };
        let (zone_bytes, bloom_bytes) = segment_frame_bytes(&segment);
        let segment_bytes = writer.write_segment(
            col.id,
            &segment.header.to_bytes(),
            bloom_bytes.as_deref(),
            &zone_bytes,
            &segment.null_bitmap,
            &segment.encoded_data,
        )?;

        // Only a declared column's filter is carried into the manifest. A
        // bloom is ten bits per value, so carrying every column's would make
        // the manifest grow with the rows in the table rather than with the
        // files in it, and the manifest is held in memory per version. Every
        // filter is still written into the segment, where a probe that has
        // opened the file reads it from the offset the header records. What
        // a declared column buys is the probe that answers before the file
        // is opened at all
        let bloom_for_manifest = if declared_bloom { bloom_bytes } else { None };
        column_stats.push(column_stats_entry(
            col,
            cells,
            &segment,
            row_count,
            bloom_for_manifest,
            segment_bytes,
        ));
    }
    let size_bytes = writer.finalize(true)?;

    column_stats.sort_by_key(|s| s.column_id);
    Ok(WrittenFile {
        entry: PartitionEntry {
            partition_id: req.partition_id,
            size_bytes,
            row_count: row_count as u64,
            added_version: 0,
            cluster_spec_id: req.cluster_spec_id,
            column_stats: std::sync::Arc::new(column_stats),
            delete_predicate_ids: Vec::new(),
        },
        order,
    })
}

/// The order this batch's rows are stored in: a permutation over row
/// indices ordered by the key columns under the curve each one declared,
/// nulls last, stable for determinism. A batch with no sort key stores rows
/// in arrival order, and a declared strategy that did not change the order
/// would be a note in the manifest and nothing else.
///
/// The curve is the first key's, because a multi-dimensional curve consumes
/// every dimension at once and keys declared after it only refine rows the
/// curve placed together. It decides both the permutation and whether the
/// header may claim ascending order.
///
/// This is the whole of what a cluster key costs a write, which is why it
/// is reachable on its own: measuring it as the difference between a keyed
/// and an unkeyed file makes it the gap between two numbers that carry the
/// encode and the fsync, and a pass costing tens of microseconds does not
/// survive that subtraction
pub fn stored_order(
    schema: &LakeSchema,
    req: &WriteRequest<'_>,
    row_count: usize,
) -> Result<Vec<usize>, ZyronError> {
    if req.sort_keys.is_empty() {
        return Ok((0..row_count).collect());
    }
    let leading_strategy = req
        .sort_strategies
        .first()
        .copied()
        .unwrap_or(ClusterStrategy::RangePartition);

    let mut keys = Vec::with_capacity(req.sort_keys.len());
    for key_id in req.sort_keys {
        let col = schema.column_by_id(*key_id).ok_or_else(|| {
            ZyronError::Internal(format!("sort key column {} is not in the schema", key_id))
        })?;
        let data = req
            .columns
            .iter()
            .find(|c| c.column_id == *key_id)
            .ok_or_else(|| {
                ZyronError::Internal(format!("sort key column {} has no data", key_id))
            })?;
        keys.push((col.physical_type_id(), data));
    }
    // One ordering key per row, built once rather than per comparison: a
    // sort does O(n log n) comparisons and this makes each of them a byte
    // compare over `8 * dimensions` bytes.
    //
    // The keys are packed end to end in one buffer at a fixed stride, the
    // same shape the clustering pass carries them in. A key per row in its
    // own `Vec` costs an allocation per row and leaves every one of those
    // comparisons dereferencing a pointer to reach the bytes, which on a
    // million row file is a million allocations under a sort that touches
    // them twenty million times
    let key_len = keys.len() * 8;
    let key_span = row_count * key_len;
    // SAFETY: every curve writes the whole of the slot it is handed, so the
    // loop below covers all `key_span` bytes before anything reads them.
    // Zeroing first would write the buffer twice, and on a pass over ten
    // million rows that is the buffer itself again in wasted stores
    #[allow(clippy::uninit_vec)]
    let mut key_bytes: Vec<u8> = {
        let mut buf = Vec::with_capacity(key_span);
        unsafe { buf.set_len(key_span) };
        buf
    };
    // A byte per row for something most columns do not have. Taken only
    // once a null actually arrives, and read through `row_is_null`
    let mut null_key: Vec<bool> = Vec::new();
    let mut axes = Vec::with_capacity(keys.len());
    for row in 0..row_count {
        axes.clear();
        let mut any_null = false;
        for (physical, data) in &keys {
            match data.cell(row) {
                Some(cell) => axes.push(normalize_component(*physical, cell)),
                None => {
                    any_null = true;
                    axes.push(0);
                }
            }
        }
        // Nulls last, whatever the curve: they have no position on it
        if any_null {
            if null_key.is_empty() {
                null_key = vec![false; row_count];
            }
            null_key[row] = true;
        }
        ordering_key_into(
            leading_strategy,
            &axes,
            &mut key_bytes[row * key_len..(row + 1) * key_len],
        );
    }

    // Sorting each key beside its own row keeps every comparison on one
    // contiguous record. Permuting a list of indices instead reads the key
    // through the index, so each of the O(n log n) comparisons reaches into
    // a second buffer at a row the previous comparison did not touch.
    //
    // Carrying the row inside the record also settles ties by row, which is
    // the order a stable sort of the keys alone produces, so the unstable
    // sort here returns the same permutation without its scratch buffer.
    //
    // A key wider than one word, or a column with nulls to push past the
    // end, falls through to the general compare
    // Empty means no row was null, so no row reads as null
    let row_is_null = |row: usize| null_key.get(row).copied().unwrap_or(false);
    let has_nulls = !null_key.is_empty();
    if !has_nulls && (key_len == 8 || key_len == 16) {
        let width_err = || ZyronError::Internal(format!("ordering key is not {} bytes", key_len));
        if key_len == 8 {
            let mut keyed: Vec<(u64, u32)> = Vec::with_capacity(row_count);
            for row in 0..row_count {
                let at = row * 8;
                let word =
                    u64::from_be_bytes(key_bytes[at..at + 8].try_into().map_err(|_| width_err())?);
                keyed.push((word, row as u32));
            }
            keyed.sort_unstable();
            return Ok(keyed.into_iter().map(|(_, row)| row as usize).collect());
        }
        let mut keyed: Vec<(u128, u32)> = Vec::with_capacity(row_count);
        for row in 0..row_count {
            let at = row * 16;
            let word =
                u128::from_be_bytes(key_bytes[at..at + 16].try_into().map_err(|_| width_err())?);
            keyed.push((word, row as u32));
        }
        keyed.sort_unstable();
        return Ok(keyed.into_iter().map(|(_, row)| row as usize).collect());
    }

    let key_at = |row: usize| &key_bytes[row * key_len..(row + 1) * key_len];
    let mut order: Vec<usize> = (0..row_count).collect();
    order.sort_by(|&a, &b| {
        row_is_null(a)
            .cmp(&row_is_null(b))
            .then_with(|| key_at(a).cmp(key_at(b)))
    });
    Ok(order)
}

/// The column's smallest and largest stored cell, at full width.
///
/// A segment build already ordered the column to lay its zone maps down, so
/// the rows it landed on are the rows the manifest bounds come from, and no
/// second comparison pass is needed. A fixed-width byte family is the
/// exception: every lake reader orders those cells by their bytes from the
/// first, and a stat slot orders them as a little endian integer, so a
/// column of that shape is compared here. It is the same shape the stored
/// filter refuses to push down for
/// The two byte regions the file writer is handed alongside a built
/// segment: its zone maps flattened, and its bloom filter serialized.
///
/// Reachable on its own because it is real per column work that is neither
/// encoding nor IO, and a phase measured as the gap between a whole write
/// and the parts of it that were timed is a remainder carrying every other
/// measurement's error
pub fn segment_frame_bytes(segment: &ColumnSegment) -> (Vec<u8>, Option<Vec<u8>>) {
    let zone_bytes: Vec<u8> = segment
        .zone_maps
        .iter()
        .flat_map(|z| z.to_bytes())
        .collect();
    let bloom_bytes = segment.bloom_filter.as_ref().map(|b| b.to_bytes());
    (zone_bytes, bloom_bytes)
}

/// The manifest entry describing one written column.
///
/// Bounds come off the segment build's own extremes rather than a second
/// walk of the column, except where the segment's slot order is not the
/// order the lake compares those cells in. `bloom_for_manifest` is the
/// filter to carry, which is None for a column that did not declare one
pub fn column_stats_entry(
    col: &crate::schema::LakeColumn,
    cells: StoredCells<'_>,
    segment: &ColumnSegment,
    row_count: usize,
    bloom_for_manifest: Option<Vec<u8>>,
    segment_bytes: u64,
) -> ColumnStatsEntry {
    let physical = col.physical_type_id();
    let (min_cell, max_cell) = column_extrema(physical, cells, segment);
    ColumnStatsEntry {
        column_id: col.id,
        bounds: ColumnBounds {
            min: min_cell.and_then(|c| cell_to_value(physical, c)),
            max: max_cell.and_then(|c| cell_to_value(physical, c)),
            null_count: segment.header.null_count,
            row_count: row_count as u64,
        },
        bloom: bloom_for_manifest,
        ndv: segment.ndv,
        size_bytes: Some(segment_bytes),
    }
}

/// One column's cells in stored order, in whichever shape the write path
/// is holding them.
///
/// A fixed-width column reaches the segment build as the packed buffer the
/// encoder reads, and everything downstream that wants a cell addresses it
/// by slot rather than through a pointer array that would have to be built
/// only to be read twice
#[derive(Clone, Copy)]
pub enum StoredCells<'a> {
    /// Cells at a fixed stride with a null bitmap, in stored order
    Packed {
        values: &'a [u8],
        nulls: &'a [u8],
        width: usize,
        rows: usize,
    },
    /// A pointer per cell, which is what a variable-width column leaves
    Views(&'a [Option<&'a [u8]>]),
}

impl<'a> StoredCells<'a> {
    /// The cell at one stored-order slot, None when it is null or past the
    /// end
    #[inline]
    pub fn get(&self, slot: usize) -> Option<&'a [u8]> {
        match *self {
            StoredCells::Packed {
                values,
                nulls,
                width,
                rows,
            } => {
                if slot >= rows
                    || nulls
                        .get(slot / 8)
                        .is_some_and(|byte| byte & (1 << (slot % 8)) != 0)
                {
                    return None;
                }
                values.get(slot * width..(slot + 1) * width)
            }
            StoredCells::Views(views) => views.get(slot).copied().flatten(),
        }
    }

    /// Cells in stored order
    pub fn len(&self) -> usize {
        match *self {
            StoredCells::Packed { rows, .. } => rows,
            StoredCells::Views(views) => views.len(),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Every non-null cell in stored order
    pub fn present(&self) -> impl Iterator<Item = &'a [u8]> + '_ {
        (0..self.len()).filter_map(|slot| self.get(slot))
    }
}

fn column_extrema<'a>(
    physical: TypeId,
    cells: StoredCells<'a>,
    segment: &ColumnSegment,
) -> (Option<&'a [u8]>, Option<&'a [u8]>) {
    if physical.fixed_size().is_some() && cell_family(physical) == CellFamily::Bytes {
        // The one family whose segment slot order disagrees with the lake's
        // cell order, so the extremes the segment recorded are not the ones
        // a manifest bound may claim
        let mut min_cell: Option<&'a [u8]> = None;
        let mut max_cell: Option<&'a [u8]> = None;
        for cell in cells.present() {
            if min_cell.is_none_or(|m| compare_cells(physical, cell, m).is_lt()) {
                min_cell = Some(cell);
            }
            if max_cell.is_none_or(|m| compare_cells(physical, cell, m).is_gt()) {
                max_cell = Some(cell);
            }
        }
        return (min_cell, max_cell);
    }
    (
        segment.min_row.and_then(|row| cells.get(row)),
        segment.max_row.and_then(|row| cells.get(row)),
    )
}

/// Checks the batch covers the schema exactly and enforces NOT NULL at
/// the format boundary. Returns the row count
fn validate_batch(schema: &LakeSchema, columns: &[ColumnData]) -> Result<usize, ZyronError> {
    if columns.len() != schema.columns.len() {
        return Err(ZyronError::Internal(format!(
            "batch has {} columns, schema has {}",
            columns.len(),
            schema.columns.len()
        )));
    }
    let mut row_count = None;
    for data in columns {
        let col = schema.column_by_id(data.column_id).ok_or_else(|| {
            ZyronError::Internal(format!(
                "batch column {} is not in the schema",
                data.column_id
            ))
        })?;
        match row_count {
            None => row_count = Some(data.len()),
            Some(n) if n != data.len() => {
                return Err(ZyronError::Internal(format!(
                    "column \"{}\" has {} rows, expected {}",
                    col.name,
                    data.len(),
                    n
                )));
            }
            Some(_) => {}
        }
        if !col.nullable {
            if let Some(row) = data.first_null() {
                return Err(ZyronError::Internal(format!(
                    "null value in non-nullable column \"{}\" at row {}",
                    col.name, row
                )));
            }
        }
    }
    match row_count {
        Some(0) | None => Err(ZyronError::Internal("empty row batch".into())),
        Some(n) => Ok(n),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Whatever layout a column ends up on, the cells that come back out of
    /// it are the cells that went in. Every shape a batch takes: fixed
    /// width, fixed width with nulls, all null, variable width, empty, and
    /// a fixed-width column that a mismatched cell forces onto offsets
    #[test]
    fn a_column_returns_the_cells_it_was_given() {
        let shapes: Vec<(&str, usize, Vec<Option<Vec<u8>>>)> = vec![
            ("empty", 8, Vec::new()),
            (
                "fixed width",
                8,
                (0..64i64).map(|v| Some(v.to_le_bytes().to_vec())).collect(),
            ),
            (
                "fixed width with nulls",
                8,
                (0..64i64)
                    .map(|v| (v % 3 != 0).then(|| v.to_le_bytes().to_vec()))
                    .collect(),
            ),
            ("all null", 8, vec![None; 64]),
            ("leading null", 8, {
                let mut cells: Vec<Option<Vec<u8>>> =
                    (0..8i64).map(|v| Some(v.to_le_bytes().to_vec())).collect();
                cells[0] = None;
                cells
            }),
            (
                "variable width",
                0,
                vec![
                    Some(b"a".to_vec()),
                    None,
                    Some(Vec::new()),
                    Some(b"a longer value".to_vec()),
                ],
            ),
            (
                "a cell that breaks the width",
                8,
                vec![
                    Some(1i64.to_le_bytes().to_vec()),
                    Some(2i64.to_le_bytes().to_vec()),
                    Some(b"three".to_vec()),
                    None,
                    Some(4i64.to_le_bytes().to_vec()),
                ],
            ),
        ];

        for (name, width, cells) in shapes {
            let expected: Vec<Option<&[u8]>> = cells.iter().map(|c| c.as_deref()).collect();

            // Built cell by cell through the width it was declared at
            let mut built = ColumnData::with_capacity(7, width, cells.len());
            for cell in &cells {
                built.push(cell.as_deref());
            }
            // And built from the owned cells, which infers the width
            let inferred = ColumnData::from_cells(7, cells.clone());

            for column in [&built, &inferred] {
                assert_eq!(column.len(), cells.len(), "{name}: row count");
                assert_eq!(column.is_empty(), cells.is_empty(), "{name}: emptiness");
                let read: Vec<Option<&[u8]>> = column.iter().collect();
                assert_eq!(read, expected, "{name}: cells through the iterator");
                for (row, want) in expected.iter().enumerate() {
                    assert_eq!(&column.cell(row), want, "{name}: cell {row}");
                    assert_eq!(column.is_null(row), want.is_none(), "{name}: null {row}");
                }
                assert_eq!(
                    column.first_null(),
                    expected.iter().position(|c| c.is_none()),
                    "{name}: first null"
                );
            }
        }
    }

    /// The packed buffer is the one a fixed-width segment build encodes, so
    /// it has to be every cell at the declared stride with null slots
    /// zeroed. A column that is not stored at the width the caller asks for
    /// declines rather than handing back a buffer that means something else
    #[test]
    fn the_packed_buffer_is_the_segment_builds_own_layout() {
        let cells: Vec<Option<Vec<u8>>> = (0..32i64)
            .map(|v| (v % 5 != 0).then(|| v.to_le_bytes().to_vec()))
            .collect();
        let column = ColumnData::from_cells(0, cells.clone());
        let packed = column.packed_at(8).expect("a fixed width column packs");
        assert_eq!(packed.len(), cells.len() * 8);
        for (row, cell) in cells.iter().enumerate() {
            let slot = &packed[row * 8..(row + 1) * 8];
            match cell {
                Some(value) => assert_eq!(slot, value.as_slice(), "row {row} holds its value"),
                None => assert_eq!(slot, &[0u8; 8], "a null slot is zero filled"),
            }
        }

        // A width the column is not stored at, and a variable-width column,
        // both decline
        assert_eq!(column.packed_at(4), None, "the wrong width declines");
        assert_eq!(column.packed_at(0), None, "no width declines");
        let varlen = ColumnData::from_cells(0, vec![Some(b"a".to_vec()), Some(b"bb".to_vec())]);
        assert_eq!(varlen.packed_at(2), None, "a ragged column declines");
    }

    /// A column that takes a mismatched cell moves onto the offset layout
    /// and keeps everything it already held, which is what stops the width
    /// check in the segment build from reading a corrupted value instead of
    /// reporting the mismatch
    #[test]
    fn the_one_word_ordering_matches_a_compare_of_the_whole_key() {
        // A single word key takes a packed sort and a wider one or a column
        // with nulls takes the general compare. Both have to place the rows
        // the same way, ties included, or a file's stored order would depend
        // on which branch ran
        use zyron_common::types::TypeId;

        let rows = 400usize;
        for (name, values) in [
            ("ascending", (0..rows).map(|r| r as i64).collect::<Vec<_>>()),
            ("descending", (0..rows).map(|r| -(r as i64)).collect()),
            ("ties", (0..rows).map(|r| (r % 7) as i64).collect()),
            ("one value", vec![42i64; rows]),
            (
                "across zero",
                (0..rows).map(|r| r as i64 - rows as i64 / 2).collect(),
            ),
        ] {
            let schema = LakeSchema::new(
                1,
                vec![LakeColumn {
                    id: 0,
                    name: "k".into(),
                    type_id: TypeId::Int64,
                    nullable: false,
                    fractional_digits: None,
                    tz_offset_secs: None,
                    max_length: None,
                    default_expr: None,
                }],
            )
            .expect("schema");
            let mut column = ColumnData::with_capacity(0, 8, rows);
            for value in &values {
                column.push(Some(&value.to_le_bytes()));
            }
            let columns = vec![column];
            let order = stored_order(
                &schema,
                &WriteRequest {
                    partition_id: 0,
                    columns: &columns,
                    sort_keys: &[0],
                    sort_strategies: &[ClusterStrategy::RangePartition],
                    cluster_spec_id: 1,
                    table_id: 1,
                    bloom_columns: &[],
                    index_id: None,
                },
                rows,
            )
            .expect("order");

            assert_eq!(order.len(), rows, "{name} lost rows");
            let mut seen: Vec<usize> = order.clone();
            seen.sort_unstable();
            assert_eq!(
                seen,
                (0..rows).collect::<Vec<_>>(),
                "{name} is not a permutation"
            );
            for pair in order.windows(2) {
                let (a, b) = (pair[0], pair[1]);
                assert!(
                    (values[a], a) <= (values[b], b),
                    "{name} placed row {a} with key {} before row {b} with key {}",
                    values[a],
                    values[b]
                );
            }
        }
    }

    #[test]
    fn nulls_ahead_of_the_first_value_do_not_cost_the_column_its_stride() {
        // A producer that cannot know its width up front takes one from the
        // first cell that has one. A leading null carries no width, so the
        // column has to keep waiting rather than settle for offsets
        for leading in 0..5usize {
            let mut column = ColumnData::with_capacity(3, 0, 0);
            for _ in 0..leading {
                column.push(None);
            }
            for value in 0..4i64 {
                column.push(Some(&value.to_le_bytes()));
            }
            assert_eq!(
                column.width(),
                8,
                "{leading} leading nulls left the column off its stride"
            );
            let packed = column
                .packed_at(8)
                .expect("a column of eight byte cells packs");
            assert_eq!(packed.len(), (leading + 4) * 8);
            assert_eq!(
                column.len(),
                leading + 4,
                "the leading nulls are still cells"
            );
            for row in 0..leading {
                assert!(column.is_null(row), "row {row} should read as null");
                assert_eq!(column.cell(row), None);
                assert_eq!(
                    &packed[row * 8..(row + 1) * 8],
                    &[0u8; 8],
                    "a null slot is zero filled so the encoder reads a placeholder"
                );
            }
            for value in 0..4i64 {
                let row = leading + value as usize;
                assert_eq!(column.cell(row), Some(&value.to_le_bytes()[..]));
            }
        }
    }

    #[test]
    fn a_column_of_only_nulls_reads_back_as_nulls() {
        // Nothing ever names a width, so the waiting nulls are all there is
        let mut column = ColumnData::with_capacity(3, 0, 0);
        for _ in 0..6 {
            column.push(None);
        }
        assert_eq!(column.len(), 6);
        assert_eq!(column.width(), 0, "no cell named a width");
        assert_eq!(column.packed_at(8), None, "and none can be claimed");
        for row in 0..6 {
            assert!(column.is_null(row));
            assert_eq!(column.cell(row), None);
        }
        assert_eq!(column.first_null(), Some(0));
    }

    #[test]
    fn a_zero_byte_cell_after_leading_nulls_stays_on_offsets() {
        // An empty cell names no width either, so the column cannot adopt
        // one from it, and the nulls ahead of it are cells of no bytes
        let mut column = ColumnData::with_capacity(3, 0, 0);
        column.push(None);
        column.push(None);
        column.push(Some(b""));
        column.push(Some(b"tail"));
        assert_eq!(column.width(), 0, "an empty cell names no stride");
        assert_eq!(column.len(), 4);
        assert_eq!(column.cell(0), None);
        assert_eq!(column.cell(1), None);
        assert_eq!(column.cell(2), Some(&b""[..]), "an empty cell is not null");
        assert_eq!(column.cell(3), Some(&b"tail"[..]));
    }

    #[test]
    fn a_width_taken_from_a_later_cell_still_spills_when_cells_disagree() {
        // Adopting a width from the first value has to leave the column able
        // to give it up, with the back-filled nulls addressed correctly
        let mut column = ColumnData::with_capacity(3, 0, 0);
        column.push(None);
        column.push(Some(b"ab"));
        column.push(Some(b"cde"));
        assert_eq!(column.width(), 0, "disagreeing cells spill to offsets");
        assert_eq!(column.len(), 3);
        assert_eq!(column.cell(0), None);
        assert_eq!(column.cell(1), Some(&b"ab"[..]));
        assert_eq!(column.cell(2), Some(&b"cde"[..]));
    }

    #[test]
    fn a_mismatched_cell_moves_the_column_off_its_stride() {
        let mut column = ColumnData::with_capacity(0, 8, 4);
        column.push(Some(&1i64.to_le_bytes()));
        column.push(None);
        assert_eq!(column.width(), 8, "still on its stride");
        column.push(Some(b"a longer cell"));
        assert_eq!(column.width(), 0, "the mismatch moved it off");
        column.push(Some(&4i64.to_le_bytes()));

        assert_eq!(column.cell(0), Some(1i64.to_le_bytes().as_slice()));
        assert_eq!(column.cell(1), None);
        assert_eq!(column.cell(2), Some(b"a longer cell".as_slice()));
        assert_eq!(column.cell(3), Some(4i64.to_le_bytes().as_slice()));
        assert_eq!(column.len(), 4);
        assert_eq!(column.packed_at(8), None);
    }
    use crate::cells::cell_to_value;
    use crate::manifest::FileStats;
    use crate::predicate::{CompareOp, LakePredicate, LakeValue, PruneDecision, StatsSource};
    use crate::schema::LakeColumn;
    use zyron_common::TypeId;

    fn column(id: u32, name: &str, type_id: TypeId) -> LakeColumn {
        LakeColumn {
            id,
            name: name.into(),
            type_id,
            nullable: true,
            fractional_digits: None,
            tz_offset_secs: None,
            max_length: None,
            default_expr: None,
        }
    }

    fn cells<const N: usize>(values: [&[u8]; N]) -> Vec<Option<Vec<u8>>> {
        values.iter().map(|v| Some(v.to_vec())).collect()
    }

    #[test]
    fn test_a_declared_curve_changes_the_file_layout() {
        use crate::curve::{normalize_component, ordering_key};
        use crate::manifest::ClusterStrategy;
        use crate::reader::LakeFileReader;

        let dir = tempfile::TempDir::new().expect("temp dir");
        let paths = LakePaths::new(dir.path(), 61);
        let schema = LakeSchema::new(
            1,
            vec![column(0, "x", TypeId::Int64), column(1, "y", TypeId::Int64)],
        )
        .expect("schema");

        // A grid, so the two curves have something to disagree about
        let mut xs = Vec::new();
        let mut ys = Vec::new();
        for x in 0..8i64 {
            for y in 0..8i64 {
                xs.push(Some(x.to_le_bytes().to_vec()));
                ys.push(Some(y.to_le_bytes().to_vec()));
            }
        }
        let columns = vec![ColumnData::from_cells(0, xs), ColumnData::from_cells(1, ys)];

        let write = |partition_id: u64, strategy: ClusterStrategy| {
            write_data_file(
                &paths,
                &schema,
                &WriteRequest {
                    partition_id,
                    columns: &columns,
                    sort_keys: &[0, 1],
                    sort_strategies: &[strategy, strategy],
                    cluster_spec_id: 1,
                    table_id: 61,
                    bloom_columns: &[],
                    index_id: None,
                },
            )
            .expect("write")
        };
        write(0x71, ClusterStrategy::RangePartition);
        write(0x72, ClusterStrategy::BitInterleave);

        // Read each file's rows back in stored order
        let read = |partition_id: u64| -> Vec<(i64, i64)> {
            let reader = LakeFileReader::open(&paths, partition_id).expect("open");
            let x = reader
                .read_column(schema.column_by_id(0).expect("x"))
                .expect("x");
            let y = reader
                .read_column(schema.column_by_id(1).expect("y"))
                .expect("y");
            (0..reader.row_count())
                .map(|row| {
                    let mut a = [0u8; 8];
                    a.copy_from_slice(x.cell(row).expect("cell"));
                    let mut b = [0u8; 8];
                    b.copy_from_slice(y.cell(row).expect("cell"));
                    (i64::from_le_bytes(a), i64::from_le_bytes(b))
                })
                .collect()
        };
        let ranged = read(0x71);
        let interleaved = read(0x72);
        assert_eq!(ranged.len(), 64);
        assert_eq!(interleaved.len(), 64);
        assert_ne!(
            ranged, interleaved,
            "a declared curve that did not change the layout would be a note and nothing else"
        );

        // Range order is plain ascending by the leading column
        assert_eq!(ranged[0], (0, 0));
        assert_eq!(ranged[7], (0, 7));
        assert_eq!(ranged[8], (1, 0));

        // The interleaved file matches what the curve says, row for row
        let mut expected: Vec<(i64, i64)> = ranged.clone();
        expected.sort_by_key(|(x, y)| {
            ordering_key(
                ClusterStrategy::BitInterleave,
                &[
                    normalize_component(TypeId::Int64, &x.to_le_bytes()),
                    normalize_component(TypeId::Int64, &y.to_le_bytes()),
                ],
            )
        });
        assert_eq!(interleaved, expected);
    }

    /// The manifest entry's bounds, null count and distinct estimate all
    /// come off the segment build now, so they have to be what a flat scan
    /// of the same cells says. A bound that drifted would prune away files
    /// that hold matching rows.
    ///
    /// Every ordered family is covered, including the fixed-width byte
    /// family whose stat slots order by the little endian reading and whose
    /// cells order by their bytes, which is the one shape the writer still
    /// compares itself
    #[test]
    fn test_manifest_bounds_match_a_flat_scan_of_the_cells() {
        let tmp = tempfile::TempDir::new().expect("temp dir");
        let paths = LakePaths::new(tmp.path(), 91);
        let schema = LakeSchema::new(
            1,
            vec![
                column(0, "signed", TypeId::Int64),
                column(1, "unsigned", TypeId::UInt64),
                column(2, "real", TypeId::Float64),
                column(3, "text", TypeId::Varchar),
                column(4, "id", TypeId::Uuid),
                column(5, "flag", TypeId::Boolean),
                column(6, "ts", TypeId::Timestamp),
                column(7, "sparse", TypeId::Int32),
                // An unordered family records no bounds at all
                column(8, "doc", TypeId::Json),
            ],
        )
        .expect("schema");

        // Enough rows to span several zones, scattered so the extremes do
        // not land at either end of the file
        let rows = 5_000usize;
        let scatter = |i: usize| (i * 2_731) % rows;
        let cells_for = |id: u32| -> Vec<Option<Vec<u8>>> {
            (0..rows)
                .map(|i| {
                    let n = scatter(i);
                    match id {
                        0 => Some(((n as i64) - 2_500).to_le_bytes().to_vec()),
                        1 => Some((u64::MAX - n as u64).to_le_bytes().to_vec()),
                        2 => Some(((n as f64) - 2_500.5).to_le_bytes().to_vec()),
                        3 => {
                            let mut value = vec![b'p'; 32];
                            value.extend_from_slice(format!("{:08}", n).as_bytes());
                            Some(value)
                        }
                        4 => {
                            let mut id = [0u8; 16];
                            id[..8].copy_from_slice(&(n as u64).to_be_bytes());
                            id[8..].copy_from_slice(&(n as u64).to_le_bytes());
                            Some(id.to_vec())
                        }
                        5 => Some(vec![(n % 2) as u8]),
                        6 => Some(
                            (((n as i64) - 2_500) * 86_400_000_000)
                                .to_le_bytes()
                                .to_vec(),
                        ),
                        7 => {
                            if (1_024..2_048).contains(&i) || i % 7 == 0 {
                                None
                            } else {
                                Some(((n as i32) - 2_500).to_le_bytes().to_vec())
                            }
                        }
                        _ => Some(format!("{{\"n\":{}}}", n).into_bytes()),
                    }
                })
                .collect()
        };
        let columns: Vec<ColumnData> = schema
            .columns
            .iter()
            .map(|col| ColumnData::from_cells(col.id, cells_for(col.id)))
            .collect();

        // Written under a sort key, so the stored order is a permutation of
        // the input and the rows the segment reports are file rows
        let entry = write_data_file(
            &paths,
            &schema,
            &WriteRequest {
                partition_id: 0x91,
                columns: &columns,
                sort_keys: &[0],
                sort_strategies: &[ClusterStrategy::RangePartition],
                cluster_spec_id: 0,
                table_id: 91,
                bloom_columns: &[],
                index_id: None,
            },
        )
        .expect("write");

        for data in &columns {
            let col = schema.column_by_id(data.column_id).expect("column");
            let physical = col.physical_type_id();
            let name = &col.name;
            let stats = entry.stats_for(data.column_id).expect("stats");

            let mut min_cell: Option<&[u8]> = None;
            let mut max_cell: Option<&[u8]> = None;
            let mut nulls = 0u64;
            let mut distinct = std::collections::HashSet::new();
            for cell in data.iter() {
                match cell {
                    Some(cell) => {
                        distinct.insert(cell);
                        if min_cell.is_none_or(|m| compare_cells(physical, cell, m).is_lt()) {
                            min_cell = Some(cell);
                        }
                        if max_cell.is_none_or(|m| compare_cells(physical, cell, m).is_gt()) {
                            max_cell = Some(cell);
                        }
                    }
                    None => nulls += 1,
                }
            }

            assert_eq!(
                stats.bounds.min,
                min_cell.and_then(|c| cell_to_value(physical, c)),
                "column \"{name}\" recorded a minimum the cells do not agree with"
            );
            assert_eq!(
                stats.bounds.max,
                max_cell.and_then(|c| cell_to_value(physical, c)),
                "column \"{name}\" recorded a maximum the cells do not agree with"
            );
            assert_eq!(
                stats.bounds.null_count, nulls,
                "column \"{name}\" recorded the wrong null count"
            );
            assert_eq!(stats.bounds.row_count, rows as u64);

            let ndv = stats
                .ndv
                .unwrap_or_else(|| panic!("column \"{name}\" carries no distinct estimate"));
            let truth = distinct.len() as f64;
            let error = (ndv as f64 - truth).abs() / truth;
            assert!(
                error < 0.05,
                "column \"{name}\" estimated {ndv} distinct against {truth}"
            );
        }

        // An unordered family has no comparison, so it records no bounds
        let doc = entry.stats_for(8).expect("stats");
        assert!(doc.bounds.min.is_none() && doc.bounds.max.is_none());
    }

    /// A manifest entry has to stay metadata.
    ///
    /// A bloom is ten bits per value, so copying one into the entry for
    /// every column that clears the cardinality heuristic makes the manifest
    /// grow with the rows in the table rather than the files in it, and the
    /// manifest is held in memory per version. An undeclared column keeps
    /// its filter in the segment, where a probe that has opened the file
    /// still finds it, and the entry carries nothing for it.
    #[test]
    fn test_only_a_declared_bloom_column_puts_its_filter_in_the_manifest() {
        use zyron_storage::columnar::ZyrFileReader;

        let tmp = tempfile::TempDir::new().expect("temp dir");
        let paths = LakePaths::new(tmp.path(), 77);
        let schema = LakeSchema::new(
            1,
            vec![
                column(0, "declared", TypeId::Int64),
                column(1, "undeclared", TypeId::Int64),
            ],
        )
        .expect("schema");

        // Both columns are high cardinality, so the heuristic would build a
        // filter for each of them
        let rows = 4096usize;
        let columns = vec![
            ColumnData::from_cells(
                0,
                (0..rows)
                    .map(|i| Some((i as i64).to_le_bytes().to_vec()))
                    .collect(),
            ),
            ColumnData::from_cells(
                1,
                (0..rows)
                    .map(|i| Some(((i as i64) * 31).to_le_bytes().to_vec()))
                    .collect(),
            ),
        ];
        let entry = write_data_file(
            &paths,
            &schema,
            &WriteRequest {
                partition_id: 0x77,
                columns: &columns,
                sort_keys: &[],
                sort_strategies: &[],
                cluster_spec_id: 0,
                table_id: 77,
                bloom_columns: &[0],
                index_id: None,
            },
        )
        .expect("write");

        let declared = entry.stats_for(0).expect("declared stats");
        let undeclared = entry.stats_for(1).expect("undeclared stats");
        assert!(
            declared.bloom.is_some(),
            "a declared column buys the probe that answers before the file is opened"
        );
        assert!(
            undeclared.bloom.is_none(),
            "an undeclared column must not spend manifest bytes on a filter"
        );

        // The undeclared column still has its filter in the file, so a probe
        // that has opened it loses nothing
        let reader = ZyrFileReader::open(&paths.data_file(0x77)).expect("open");
        for column_id in [0u32, 1] {
            let bloom = reader
                .read_bloom(column_id)
                .expect("read bloom")
                .unwrap_or_else(|| panic!("column {column_id} carries no filter in its segment"));
            let data = columns
                .iter()
                .find(|c| c.column_id == column_id)
                .expect("column");
            for cell in data.iter().flatten() {
                assert!(
                    bloom.might_contain(cell),
                    "column {column_id} filter denies a value the segment stored"
                );
            }
        }

        // And the manifest still prunes on the declared column, which is
        // what the bytes were spent for
        let stats = FileStats::new(&entry, &schema);
        assert!(stats.may_contain(0, &LakeValue::Int(40)));
        assert!(
            !stats.may_contain(0, &LakeValue::Int(rows as i64 + 5)),
            "a value the declared column never stored is provably absent"
        );
        // The undeclared column prunes nothing from the manifest, which is a
        // missed skip and never a dropped row
        assert!(stats.may_contain(1, &LakeValue::Int(7)));
    }

    /// A bloom answering "absent" for a value the writer stored would drop
    /// rows that exist, so the bytes the prune path probes with must be the
    /// bytes ColumnSegment::build inserted, for every type family.
    #[test]
    fn test_lake_bloom_canonical_bytes_match_segment_build() {
        let tmp = tempfile::TempDir::new().expect("temp dir");
        let paths = LakePaths::new(tmp.path(), 1);
        let schema = LakeSchema::new(
            1,
            vec![
                column(0, "i64", TypeId::Int64),
                column(1, "i32", TypeId::Int32),
                column(2, "u64", TypeId::UInt64),
                column(3, "f64", TypeId::Float64),
                column(4, "f32", TypeId::Float32),
                column(5, "text", TypeId::Varchar),
                column(6, "id", TypeId::Uuid),
                column(7, "flag", TypeId::Boolean),
                column(8, "ts", TypeId::Timestamp),
            ],
        )
        .expect("schema");

        let uuids: Vec<Option<Vec<u8>>> = (0u8..4).map(|n| Some([n; 16].to_vec())).collect();
        let columns = vec![
            ColumnData::from_cells(
                0,
                cells([
                    &(-9_000_000_000i64).to_le_bytes(),
                    &0i64.to_le_bytes(),
                    &42i64.to_le_bytes(),
                    &i64::MAX.to_le_bytes(),
                ]),
            ),
            ColumnData::from_cells(
                1,
                cells([
                    &(-7i32).to_le_bytes(),
                    &1i32.to_le_bytes(),
                    &900i32.to_le_bytes(),
                    &i32::MIN.to_le_bytes(),
                ]),
            ),
            ColumnData::from_cells(
                2,
                cells([
                    &0u64.to_le_bytes(),
                    &7u64.to_le_bytes(),
                    &u64::MAX.to_le_bytes(),
                    &1234u64.to_le_bytes(),
                ]),
            ),
            ColumnData::from_cells(
                3,
                cells([
                    &(-1.5f64).to_le_bytes(),
                    &0.0f64.to_le_bytes(),
                    &3.25f64.to_le_bytes(),
                    &1e300f64.to_le_bytes(),
                ]),
            ),
            ColumnData::from_cells(
                4,
                cells([
                    &(-1.5f32).to_le_bytes(),
                    &0.5f32.to_le_bytes(),
                    &7.75f32.to_le_bytes(),
                    &100.0f32.to_le_bytes(),
                ]),
            ),
            ColumnData::from_cells(5, cells([b"alice", b"bob\0\0", b"", b"carol"])),
            ColumnData::from_cells(6, uuids),
            ColumnData::from_cells(7, cells([&[0u8], &[1u8], &[1u8], &[0u8]])),
            ColumnData::from_cells(
                8,
                cells([
                    &(-86_400_000_000i64).to_le_bytes(),
                    &0i64.to_le_bytes(),
                    &1_700_000_000_000_000i64.to_le_bytes(),
                    &99i64.to_le_bytes(),
                ]),
            ),
        ];

        // Force a filter on every column so the probe is exercised even where
        // the cardinality heuristic would skip one
        let bloom_columns: Vec<u32> = schema.columns.iter().map(|c| c.id).collect();
        let entry = write_data_file(
            &paths,
            &schema,
            &WriteRequest {
                partition_id: 0x51,
                columns: &columns,
                sort_keys: &[],
                sort_strategies: &[],
                cluster_spec_id: 0,
                table_id: 1,
                bloom_columns: &bloom_columns,
                index_id: None,
            },
        )
        .expect("write");

        let stats = FileStats::new(&entry, &schema);
        for data in &columns {
            let col = schema.column_by_id(data.column_id).expect("column");
            let file_stats = entry.stats_for(data.column_id).expect("stats");
            assert!(
                file_stats.bloom.is_some(),
                "column {} must carry the forced bloom",
                col.name
            );
            for cell in data.iter().flatten() {
                let value = cell_to_value(col.physical_type_id(), cell)
                    .unwrap_or_else(|| panic!("column {} cell has no value", col.name));
                assert!(
                    stats.may_contain(data.column_id, &value),
                    "column {} bloom denies a value it stored: {:?}",
                    col.name,
                    value
                );
            }
        }
    }

    /// The bloom is actually consulted: a constant inside the column min/max
    /// that the file never stored prunes the file with no IO.
    #[test]
    fn test_bloom_prunes_a_value_inside_the_bounds() {
        let tmp = tempfile::TempDir::new().expect("temp dir");
        let paths = LakePaths::new(tmp.path(), 2);
        let schema = LakeSchema::new(1, vec![column(0, "k", TypeId::Int64)]).expect("schema");
        let columns = vec![ColumnData::from_cells(
            0,
            (0..512i64)
                .map(|i| Some((i * 10).to_le_bytes().to_vec()))
                .collect(),
        )];
        let entry = write_data_file(
            &paths,
            &schema,
            &WriteRequest {
                partition_id: 0x52,
                columns: &columns,
                sort_keys: &[0],
                sort_strategies: &[],
                cluster_spec_id: 0,
                table_id: 2,
                bloom_columns: &[0],
                index_id: None,
            },
        )
        .expect("write");

        let stats = FileStats::new(&entry, &schema);
        let present = LakePredicate::Compare {
            column_id: 0,
            op: CompareOp::Eq,
            value: LakeValue::Int(1230),
        };
        assert_eq!(present.prune(&stats), PruneDecision::MayMatch);

        // Every multiple of ten is stored, so a gap value sits inside the
        // bounds and is provably absent
        let absent = LakePredicate::Compare {
            column_id: 0,
            op: CompareOp::Eq,
            value: LakeValue::Int(1231),
        };
        assert_eq!(absent.prune(&stats), PruneDecision::CannotMatch);

        // Bounds alone cannot reach that answer
        assert_eq!(absent.prune(&entry), PruneDecision::MayMatch);
    }

    /// A constant with no provable stored form must prune nothing rather
    /// than probe with bytes that are not the column cells.
    #[test]
    fn test_unrepresentable_constant_prunes_nothing() {
        let tmp = tempfile::TempDir::new().expect("temp dir");
        let paths = LakePaths::new(tmp.path(), 3);
        let schema = LakeSchema::new(1, vec![column(0, "k", TypeId::Int32)]).expect("schema");
        let columns = vec![ColumnData::from_cells(
            0,
            (0..300i32)
                .map(|v| Some(v.to_le_bytes().to_vec()))
                .collect(),
        )];
        let entry = write_data_file(
            &paths,
            &schema,
            &WriteRequest {
                partition_id: 0x53,
                columns: &columns,
                sort_keys: &[],
                sort_strategies: &[],
                cluster_spec_id: 0,
                table_id: 3,
                bloom_columns: &[0],
                index_id: None,
            },
        )
        .expect("write");

        let stats = FileStats::new(&entry, &schema);
        // A string constant against an integer column has no cell form
        assert!(stats.may_contain(0, &LakeValue::Str("100".into())));
    }
}
