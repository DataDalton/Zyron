//! Columnar scan operator.
//!
//! Reads a table's registered .zyr segments through ZyrFileReader, decodes
//! each projected column plus the sys_rowid, sys_xmin and sys_supersede MVCC
//! columns, applies the columnar patch overlay, and emits only rows visible
//! to the execution snapshot. Visibility uses the same
//! Snapshot::is_visible(xmin, xmax) oracle as the heap path. A value patch
//! overrides a column for snapshots that see the patching transaction; a
//! supersede hides the row for snapshots that see the deleting transaction.
//! An optional predicate is applied after decode, then column-level security,
//! mirroring the heap scan so a columnar read and a heap read of the same
//! rows return identical results.

use std::sync::Arc;

use zyron_catalog::TableEntry;
use zyron_common::Result;
use zyron_parser::ast::{BinaryOperator, LiteralValue};
use zyron_planner::binder::{BoundExpr, ColumnRef};
use zyron_planner::logical::LogicalColumn;
use zyron_storage::columnar::{
    ColumnarPatchManager, PatchStore, RowOverlay, SYS_COL_ROWID, SYS_COL_SUPERSEDE, SYS_COL_XMIN,
    ZyrFileReader, segment_regions,
};
use zyron_storage::encoding::{create_encoding, varlen_slice_rows};

use crate::batch::{
    BATCH_SIZE, ColumnBuilder, DataBatch, ResolvedPath, create_builders, decode_fixed_scalar,
    decode_varlen_scalar, finalize_builders,
};
use crate::column::ScalarValue;
use crate::compute::column_to_mask;
use crate::context::ExecutionContext;
use crate::expr::evaluate;
use crate::operator::{
    ExecutionBatch, MetaAcc, Operator, OperatorResult, apply_column_security, expose_column_value,
    fold_rows_into_meta_accs,
};

/// Per-projected-column decode plan.
struct ColPlan {
    column_id: u32,
    type_id: zyron_common::types::TypeId,
    /// Fixed byte width, or 0 for the variable-length canonical layout.
    value_size: usize,
    /// What a segment written before this column existed reads as. A segment
    /// holds one stored column per column the table had when it was folded,
    /// so a column added afterwards has no segment in that file and its rows
    /// take the value recorded when it was added
    absent: ScalarValue,
}

/// One promoted variant path a segment stores as a column of its own, for a
/// path this statement reads.
///
/// The stored values are what the extraction returns, so a read served from
/// the column and a read that walks the document give the same answer. What
/// the column cannot answer for is a row whose variant was patched after the
/// fold, which is resolved from the patched document instead
#[derive(Clone)]
struct ShredRead {
    /// Column of the segment file holding the extracted values
    seg_column_id: u32,
    /// Position of the variant column in `col_plans`
    variant_plan_idx: usize,
    /// The variant column, as an expression names it
    variant_column_id: u16,
    /// Dotted path, as `variant_extract` names it
    path: String,
}

/// Reads registered .zyr segments for a table with snapshot visibility and
/// patch-overlay resolution.
pub struct ColumnScanOperator {
    ctx: Arc<ExecutionContext>,
    table_entry: Arc<TableEntry>,
    output_columns: Vec<LogicalColumn>,
    predicate: Option<BoundExpr>,
    col_plans: Vec<ColPlan>,
    /// (file_id, path) for each registered segment, consumed front to back.
    segments: Vec<(u64, String)>,
    seg_idx: usize,
    patch_store: Option<Arc<PatchStore>>,
    /// When set, emit (file_id, sys_rowid) per surviving row for the DML
    /// patch path instead of plain batches.
    emit_locators: bool,
    /// When set, rows are dated by commit LSN for time-travel: a row is visible
    /// at this version when its sys_xmin committed at or before it and its
    /// supersede (if any) committed after it. None uses live-snapshot MVCC.
    as_of_version: Option<u64>,
    pending: std::collections::VecDeque<ExecutionBatch>,
    finished: bool,
    /// This table's IO counters, resolved once at construction. Updated per
    /// segment with the rows it yielded and the encoded bytes read to yield
    /// them, so a segment rejected by its header or zone maps contributes rows
    /// and bytes of zero.
    io_stats: Option<Arc<zyron_common::TableIOStats>>,
    /// Segments the predicate rejected from their header, zone maps or value
    /// bloom, so no column of them was decoded.
    ///
    /// Reported rather than inferred from the bytes counter, because a
    /// rejection reads a header and a bloom and those are metadata the byte
    /// counter deliberately leaves out, so a skipped segment and a segment
    /// with nothing to read look the same there
    segments_skipped: usize,
    /// Which table instance the projected columns belong to, so a resolved
    /// path is offered to the column references that name this scan's table
    /// and to no others
    table_idx: Option<usize>,
    /// Per segment file, the promoted paths it stores that this statement
    /// reads. Captured at construction, so the set cannot change under a
    /// scan that is already running
    shreds: std::collections::HashMap<u64, Vec<ShredRead>>,
}

impl ColumnScanOperator {
    pub fn new(
        ctx: Arc<ExecutionContext>,
        table_id: zyron_catalog::TableId,
        columns: Vec<LogicalColumn>,
        predicate: Option<BoundExpr>,
    ) -> Result<Self> {
        Self::new_inner(ctx, table_id, columns, predicate, false, None)
    }

    /// Variant that also emits per-row (file_id, sys_rowid) locators so
    /// UPDATE and DELETE can route columnar-resident rows to the patch log.
    pub fn new_for_dml(
        ctx: Arc<ExecutionContext>,
        table_id: zyron_catalog::TableId,
        columns: Vec<LogicalColumn>,
        predicate: Option<BoundExpr>,
    ) -> Result<Self> {
        Self::new_inner(ctx, table_id, columns, predicate, true, None)
    }

    /// Variant restricted to a set of segment file ids. The metadata
    /// aggregate path uses this to resolve only the dirty segments instead
    /// of falling back to a full-table scan when any segment has overlay.
    pub fn new_for_files(
        ctx: Arc<ExecutionContext>,
        table_id: zyron_catalog::TableId,
        columns: Vec<LogicalColumn>,
        predicate: Option<BoundExpr>,
        only_files: std::collections::HashSet<u64>,
    ) -> Result<Self> {
        Self::new_inner(ctx, table_id, columns, predicate, false, Some(only_files))
    }

    /// Restricts the scan to a set of segment file ids after construction.
    /// The search result fetch composes this with new_for_dml so only
    /// segments holding hits are opened while locators still emit.
    pub fn with_file_filter(mut self, only_files: std::collections::HashSet<u64>) -> Self {
        self.segments.retain(|(fid, _)| only_files.contains(fid));
        self
    }

    fn new_inner(
        ctx: Arc<ExecutionContext>,
        table_id: zyron_catalog::TableId,
        columns: Vec<LogicalColumn>,
        predicate: Option<BoundExpr>,
        emit_locators: bool,
        only_files: Option<std::collections::HashSet<u64>>,
    ) -> Result<Self> {
        let table_entry = ctx.get_table_entry(table_id)?;

        let mut col_plans = Vec::with_capacity(columns.len());
        for c in &columns {
            let ce = table_entry
                .columns
                .iter()
                .find(|x| x.id == c.column_id)
                .ok_or_else(|| {
                    zyron_common::ZyronError::ExecutionError(format!(
                        "columnar scan: column {} not found in table {}",
                        c.column_id.0, table_entry.name
                    ))
                })?;
            let phys = ce.physical_type_id();
            col_plans.push(ColPlan {
                column_id: ce.id.0 as u32,
                type_id: ce.type_id,
                value_size: phys.fixed_size().unwrap_or(0),
                absent: crate::epoch_decode::absent_value_of(ce),
            });
        }

        let segments: Vec<(u64, String)> = table_entry
            .columnar
            .segments
            .iter()
            .filter(|s| {
                only_files
                    .as_ref()
                    .map(|f| f.contains(&s.file_id))
                    .unwrap_or(true)
            })
            .map(|s| (s.file_id, s.path.clone()))
            .collect();

        // The patch store is process-global, keyed by the columnar directory
        // (parent of any segment path). Skipped when there are no segments.
        let patch_store = match segments.first() {
            Some((_, p)) => Some(ColumnarPatchManager::store_for_segment(
                table_entry.id.0 as u64,
                std::path::Path::new(p),
            )?),
            None => None,
        };

        let io_stats = ctx.table_io_stats_for(table_id.0);
        if let Some(stats) = &io_stats {
            stats.record_seq_scan();
        }

        // Promoted paths this statement reads, matched against what each
        // segment actually stores. A path nothing asked for is left on disk,
        // and a segment written before the path was promoted keeps answering
        // from its documents
        let table_idx = columns.first().and_then(|c| c.table_idx);
        let wanted = ctx.variant_paths();
        let mut shreds: std::collections::HashMap<u64, Vec<ShredRead>> =
            std::collections::HashMap::new();
        if let Some(table_idx) = table_idx
            && wanted.iter().any(|w| w.table_idx == table_idx)
        {
            for seg in &table_entry.columnar.segments {
                if seg.shredded.is_empty() {
                    continue;
                }
                let mut reads = Vec::new();
                for sc in &seg.shredded {
                    let asked = wanted.iter().any(|w| {
                        w.table_idx == table_idx
                            && w.column_id == sc.variant_column_id
                            && w.path == sc.path
                    });
                    if !asked {
                        continue;
                    }
                    // The variant column has to be projected, because a row
                    // patched after the fold is answered from the patched
                    // document, which means reading that document here
                    let Some(variant_plan_idx) = col_plans
                        .iter()
                        .position(|p| p.column_id == sc.variant_column_id as u32)
                    else {
                        continue;
                    };
                    reads.push(ShredRead {
                        seg_column_id: sc.column_id,
                        variant_plan_idx,
                        variant_column_id: sc.variant_column_id,
                        path: sc.path.clone(),
                    });
                }
                if !reads.is_empty() {
                    shreds.insert(seg.file_id, reads);
                }
            }
        }

        Ok(Self {
            ctx,
            table_entry,
            output_columns: columns,
            predicate,
            col_plans,
            segments,
            seg_idx: 0,
            patch_store,
            emit_locators,
            as_of_version: None,
            pending: std::collections::VecDeque::new(),
            finished: false,
            io_stats,
            segments_skipped: 0,
            table_idx,
            shreds,
        })
    }

    /// Registered segments the predicate rejected before decoding any column
    /// of them.
    pub fn segments_skipped(&self) -> usize {
        self.segments_skipped
    }

    /// Sets the time-travel version. Rows are then dated by commit LSN instead
    /// of resolved against the live snapshot, so a query as of a past version
    /// sees the folded rows that were live at that version.
    pub fn with_as_of(mut self, as_of_version: Option<u64>) -> Self {
        self.as_of_version = as_of_version;
        self
    }

    /// Visibility oracle for one row: commit-LSN version visibility under
    /// time-travel, live-snapshot MVCC otherwise.
    #[inline]
    fn visible(&self, xmin: u64, xmax: u64) -> bool {
        match self.as_of_version {
            Some(v) => self
                .ctx
                .snapshot
                .status_map()
                .is_visible_at_version(xmin, xmax, v),
            None => self.ctx.snapshot.is_visible(xmin, xmax),
        }
    }

    /// Parses the header regions out of an already-read raw segment buffer
    /// and decodes the encoded column. Lets the scan read every needed
    /// column with one file open (see `read_segments_each`).
    fn decode_raw(
        column_id: u32,
        raw: &[u8],
        row_count: usize,
        value_size: usize,
    ) -> Result<(Vec<u8>, Vec<u8>)> {
        let regions = segment_regions(raw, column_id, row_count)?;
        let null_bitmap = raw[regions.null_bitmap.clone()].to_vec();
        let enc = regions.verified_payload(raw, column_id)?;
        let decoded =
            create_encoding(regions.header.encoding_type).decode(enc, row_count, value_size)?;
        Ok((decoded, null_bitmap))
    }

    /// Resolves a column's value for one row through the patch overlay. The
    /// newest value patch whose patch_xid is visible to the snapshot wins,
    /// otherwise the base encoded value is used.
    fn resolve_value(
        &self,
        overlay: Option<&RowOverlay>,
        column_id: u32,
        type_id: zyron_common::types::TypeId,
        value_size: usize,
        base_is_null: bool,
        base_bytes: Option<&[u8]>,
    ) -> ScalarValue {
        if let Some(ov) = overlay
            && let Some(chain) = ov.patches.get(&column_id)
        {
            let mut best: Option<&zyron_storage::columnar::ValuePatch> = None;
            for p in chain {
                // A patch is the value for S when its creating transaction is
                // visible to S (treated as an xmin with no delete).
                if self.visible(p.patch_xid, 0) {
                    match best {
                        Some(b) if b.patch_xid >= p.patch_xid => {}
                        _ => best = Some(p),
                    }
                }
            }
            if let Some(p) = best {
                return if value_size == 0 {
                    decode_varlen_scalar(type_id, &p.value)
                } else {
                    decode_fixed_scalar(type_id, &p.value)
                };
            }
        }
        if base_is_null {
            return ScalarValue::Null;
        }
        match base_bytes {
            Some(b) if value_size == 0 => decode_varlen_scalar(type_id, b),
            Some(b) => decode_fixed_scalar(type_id, b),
            None => ScalarValue::Null,
        }
    }

    fn load_segment(&mut self, file_id: u64, path: &str) -> Result<()> {
        let reader = ZyrFileReader::open(std::path::Path::new(path))?;
        let row_count = reader.header().row_count as usize;
        if row_count == 0 {
            return Ok(());
        }

        if self.segment_rejected(&reader, row_count, file_id)? {
            self.segments_skipped += 1;
            return Ok(());
        }

        let read_u64 = |buf: &[u8], i: usize| -> u64 {
            let s = i * 8;
            u64::from_le_bytes(buf[s..s + 8].try_into().unwrap())
        };

        // One file open per segment for every needed column (sys columns
        // then projected columns), instead of reopening per column.
        let mut col_ids: Vec<u32> = vec![SYS_COL_ROWID, SYS_COL_XMIN, SYS_COL_SUPERSEDE];
        for p in &self.col_plans {
            col_ids.push(p.column_id);
        }
        self.load_segment_rows(reader, row_count, file_id, col_ids, read_u64)
    }

    /// Whether the predicate proves this segment holds no matching row.
    ///
    /// Answered from metadata alone: the segment header's bounds, its zone
    /// maps, and its value bloom. Every one of them is sized by the row
    /// count or the cardinality rather than by the data, so rejecting a
    /// segment costs a fraction of decoding one.
    ///
    /// A patched (dirty) segment is never rejected: a value patch could move
    /// a row into range, and the patch is not in the metadata
    fn segment_rejected(
        &self,
        reader: &ZyrFileReader,
        row_count: usize,
        file_id: u64,
    ) -> Result<bool> {
        // Segment-level predicate pruning: if a fixed integer or
        // integer-backed temporal projected column has a range/equality
        // constraint disjoint from this segment's header [min, max], the
        // whole segment is skipped with a single small header read and zero
        // row decode. Signed and temporal columns prune too: the header
        // min/max are stored in two's complement order, so a negative bound
        // (e.g. a pre-1970 picosecond timestamp) compares correctly here.
        // A patched (dirty) segment is never pruned: a value patch could
        // move a row into range.
        if let Some(pred) = &self.predicate {
            let branch = self.ctx.active_branch_id.unwrap_or(0);
            let dirty = self
                .patch_store
                .as_ref()
                .map(|s| s.file_has_overlay_on(branch, file_id))
                .unwrap_or(false);
            if !dirty {
                for p in &self.col_plans {
                    // Segment min/max are stored with two's complement order
                    // (compare_stat_slots_typed at build), so signed integer
                    // and integer-backed temporal columns prune correctly now,
                    // not just unsigned ints. An unsigned column wider than 8
                    // bytes (UInt128) is skipped: its high-bit-set values do
                    // not map onto the i128 predicate-bound domain.
                    let signed = zyron_storage::columnar::stat_slot_is_signed(p.type_id);
                    let unsigned = matches!(
                        p.type_id,
                        zyron_common::types::TypeId::UInt8
                            | zyron_common::types::TypeId::UInt16
                            | zyron_common::types::TypeId::UInt32
                            | zyron_common::types::TypeId::UInt64
                    );
                    let prunable = (signed && p.value_size >= 1 && p.value_size <= 16)
                        || (unsigned && p.value_size >= 1 && p.value_size <= 8);
                    if !prunable {
                        continue;
                    }
                    let (lo, hi) = predicate_int_bounds(pred, p.column_id);
                    if lo.is_none() && hi.is_none() {
                        continue;
                    }
                    let (h, zones) = reader.read_segment_metadata(p.column_id, row_count)?;
                    if h.null_count >= row_count as u64 {
                        continue;
                    }
                    let width = p.value_size;
                    let le = |slot: &[u8; 32]| -> i128 {
                        let mut v: u128 = 0;
                        for k in 0..width {
                            v |= (slot[k] as u128) << (8 * k);
                        }
                        if signed && width < 16 {
                            // Sign-extend a w-byte two's complement value into
                            // i128 so a negative segment bound stays negative.
                            let sign_bit = 1u128 << (8 * width - 1);
                            if v & sign_bit != 0 {
                                v |= !((1u128 << (8 * width)) - 1);
                            }
                        }
                        v as i128
                    };
                    let smin = le(&h.min_value);
                    let smax = le(&h.max_value);
                    let lo = lo.unwrap_or(i128::MIN);
                    let hi = hi.unwrap_or(i128::MAX);
                    if smax < lo || smin > hi {
                        // Predicate range cannot intersect this segment.
                        return Ok(true);
                    }
                    // A segment's bounds are the union of its zones, so it
                    // can admit a range that no zone holds. This is what an
                    // ordering the workload asked for buys over an ascending
                    // primary key sort: the same rows in a layout whose
                    // zones are narrow enough to reject the whole segment.
                    // The zone region is sized by the row count rather than
                    // by the data, so the check costs a few kilobytes
                    // against decoding every column.
                    if !zones.is_empty()
                        && !zones
                            .iter()
                            .any(|z| le(&z.max_value) >= lo && le(&z.min_value) <= hi)
                    {
                        return Ok(true);
                    }
                    // Bounds and zones only say the value falls inside a
                    // range they cover, and for a high cardinality column
                    // every zone covers it, so an equality no row satisfies
                    // still reaches the decode. The value bloom answers
                    // whether the segment holds that exact cell, and it is
                    // built for exactly the columns bounds cannot narrow:
                    // cardinality at or above the threshold and an encoding
                    // with no membership answer of its own. A filter wider
                    // than the payload it would save reading is not a
                    // saving, whatever it answers
                    if lo == hi
                        && bloom_worth_reading(&h)
                        && let Some(cell) = int_cell_bytes(lo, width, signed)
                        && let Some(bloom) = reader.read_bloom(p.column_id)?
                        && !bloom.might_contain(&cell)
                    {
                        return Ok(true);
                    }
                }
                // The same question for a column whose cells are bytes,
                // where no bound is derived at all today, so an equality on
                // a text column decodes every segment to find nothing
                for p in &self.col_plans {
                    if p.value_size != 0
                        || !matches!(
                            p.type_id,
                            zyron_common::types::TypeId::Varchar
                                | zyron_common::types::TypeId::Text
                        )
                    {
                        continue;
                    }
                    let groups = predicate_equal_bytes(pred, p.column_id);
                    if groups.is_empty() {
                        continue;
                    }
                    let header = reader.read_segment_header(p.column_id)?;
                    if !bloom_worth_reading(&header) {
                        continue;
                    }
                    let Some(bloom) = reader.read_bloom(p.column_id)? else {
                        continue;
                    };
                    // Each group is one term the rows have to satisfy, so a
                    // group with no member the segment can hold is a term no
                    // row here satisfies
                    if groups
                        .iter()
                        .any(|group| !group.iter().any(|v| bloom.might_contain(v)))
                    {
                        return Ok(true);
                    }
                }
            }
        }
        Ok(false)
    }

    fn load_segment_rows(
        &mut self,
        reader: ZyrFileReader,
        row_count: usize,
        file_id: u64,
        mut col_ids: Vec<u32>,
        read_u64: impl Fn(&[u8], usize) -> u64,
    ) -> Result<()> {
        // Promoted paths this segment stores that the statement reads. Held
        // by value so the segment loop owns them while `self` is borrowed
        // again to queue the batches
        let shreds: Vec<ShredRead> = self.shreds.get(&file_id).cloned().unwrap_or_default();
        let shred_base = col_ids.len();
        for sr in &shreds {
            col_ids.push(sr.seg_column_id);
        }
        // Read+decode+drop one column at a time so peak raw memory is a
        // single segment instead of every requested segment held at once.
        // Decoded buffers stay resident because row iteration is row-major
        // across all projected columns. col_ids order is the sys columns,
        // the projected columns, then the shredded ones, matching the index
        // passed to the callback
        let mut decoded: Vec<Option<(Vec<u8>, Vec<u8>)>> =
            (0..col_ids.len()).map(|_| None).collect();
        // Encoded bytes pulled out of this segment, summed across every
        // column it read. A segment rejected above never reaches here, which
        // is what makes skipping show up as bytes not read
        let mut segment_bytes: u64 = 0;
        reader.read_segments_each(&col_ids, |idx, bytes| {
            let raw = match bytes {
                Some(raw) => raw,
                // A promoted path the registry names but this file does not
                // hold is read out of the documents instead, the same as a
                // segment written before the path was promoted
                None if idx >= shred_base => return Ok(()),
                // A projected column this file does not hold is a column
                // added after the fold wrote it, so its rows read the value
                // recorded when the column was added
                None if idx >= 3 => return Ok(()),
                None => {
                    return Err(zyron_common::ZyronError::ExecutionError(
                        "columnar scan: missing segment for column".into(),
                    ));
                }
            };
            segment_bytes += raw.len() as u64;
            let value_size = if idx < 3 {
                8
            } else if idx < shred_base {
                self.col_plans[idx - 3].value_size
            } else {
                0
            };
            decoded[idx] = Some(Self::decode_raw(col_ids[idx], raw, row_count, value_size)?);
            Ok(())
        })?;
        let take = |slot: &mut Option<(Vec<u8>, Vec<u8>)>| -> Result<(Vec<u8>, Vec<u8>)> {
            slot.take().ok_or_else(|| {
                zyron_common::ZyronError::ExecutionError(
                    "columnar scan: missing segment for column".into(),
                )
            })
        };
        let (rowid_bytes, _) = take(&mut decoded[0])?;
        let (xmin_bytes, _) = take(&mut decoded[1])?;
        let (supersede_bytes, _) = take(&mut decoded[2])?;

        // Which projected columns this file predates. Their rows take the
        // column's recorded absent value rather than bytes the file never
        // wrote
        let mut absent_cols: Vec<bool> = vec![false; self.col_plans.len()];
        let mut decoded_cols: Vec<(Vec<u8>, Vec<u8>, bool)> =
            Vec::with_capacity(self.col_plans.len());
        for (k, p) in self.col_plans.iter().enumerate() {
            match decoded[3 + k].take() {
                Some((bytes, nullbm)) => decoded_cols.push((bytes, nullbm, p.value_size == 0)),
                None => {
                    absent_cols[k] = true;
                    decoded_cols.push((Vec::new(), Vec::new(), false));
                }
            }
        }
        let mut varlen_rows: Vec<Option<Vec<&[u8]>>> = Vec::with_capacity(decoded_cols.len());
        for (k, (bytes, _, is_varlen)) in decoded_cols.iter().enumerate() {
            if *is_varlen && !absent_cols[k] {
                varlen_rows.push(Some(varlen_slice_rows(bytes, row_count)?));
            } else {
                varlen_rows.push(None);
            }
        }

        // The extracted values, one entry per shredded path this segment
        // serves. A path whose column was absent drops out here, so the
        // expression falls back to the document walk for it
        let mut shred_cols: Vec<(&ShredRead, &[u8], Vec<&[u8]>)> = Vec::with_capacity(shreds.len());
        for (k, sr) in shreds.iter().enumerate() {
            let Some((bytes, nullbm)) = decoded[shred_base + k].as_ref() else {
                continue;
            };
            shred_cols.push((sr, nullbm, varlen_slice_rows(bytes, row_count)?));
        }

        // Snapshot this file's overlay once under a single lock, instead of
        // a per-row lock acquisition and clone.
        let overlay_map = {
            let branch = self.ctx.active_branch_id.unwrap_or(0);
            match &self.patch_store {
                Some(s) if s.file_has_overlay_on(branch, file_id) => {
                    Some(s.file_overlay_on(branch, file_id))
                }
                _ => None,
            }
        };
        // Row-level predicate pre-skip: for clean rows (no overlay), an
        // unsigned fixed-int projected column whose decoded value lies
        // outside the conservative [lo, hi] the predicate requires cannot
        // satisfy the predicate, so the row is skipped before any column is
        // materialized. The authoritative predicate still runs in
        // queue_batch, so this only drops rows that path would also drop
        // (predicate_int_bounds guarantees outside-range implies false). A
        // row with overlay or a null in the bound column is never pre-skipped
        // (overlay can change the value, null has its own predicate
        // semantics), so it falls through to the full path.
        let row_bounds: Vec<(usize, usize, i128, i128, bool)> = match &self.predicate {
            Some(pred) => self
                .col_plans
                .iter()
                .enumerate()
                .filter_map(|(ci, p)| {
                    // A column this file predates has no bytes to bound
                    if absent_cols[ci] {
                        return None;
                    }
                    let signed = zyron_storage::columnar::stat_slot_is_signed(p.type_id);
                    let unsigned = matches!(
                        p.type_id,
                        zyron_common::types::TypeId::UInt8
                            | zyron_common::types::TypeId::UInt16
                            | zyron_common::types::TypeId::UInt32
                            | zyron_common::types::TypeId::UInt64
                    );
                    let prunable = (signed && p.value_size >= 1 && p.value_size <= 16)
                        || (unsigned && p.value_size >= 1 && p.value_size <= 8);
                    if !prunable {
                        return None;
                    }
                    let (lo, hi) = predicate_int_bounds(pred, p.column_id);
                    if lo.is_none() && hi.is_none() {
                        return None;
                    }
                    Some((
                        ci,
                        p.value_size,
                        lo.unwrap_or(i128::MIN),
                        hi.unwrap_or(i128::MAX),
                        signed,
                    ))
                })
                .collect(),
            None => Vec::new(),
        };

        let mut builders = create_builders(&self.output_columns, row_count.min(BATCH_SIZE));
        let mut shred_builders: Vec<ColumnBuilder> = shred_cols
            .iter()
            .map(|_| {
                ColumnBuilder::new(zyron_common::types::TypeId::Text, row_count.min(BATCH_SIZE))
            })
            .collect();
        let mut locators: Vec<(u64, u64)> = Vec::new();
        let mut in_batch = 0usize;
        // Visible rows this segment yielded, counted before the predicate runs
        // so the number means rows the scan read rather than rows it returned.
        let mut rows_yielded: u64 = 0;

        for r in 0..row_count {
            let sys_rowid = read_u64(&rowid_bytes, r);
            let xmin = read_u64(&xmin_bytes, r);
            let base_supersede = read_u64(&supersede_bytes, r);

            let overlay: Option<&RowOverlay> = overlay_map
                .as_ref()
                .and_then(|m| m.get(&sys_rowid))
                .map(|a| a.as_ref());

            // Visibility: base supersede plus every overlay supersede. A
            // delete is visible to S when is_visible(xmin, sup) is false.
            if !self.visible(xmin, base_supersede) {
                continue;
            }
            if let Some(ov) = overlay {
                let mut hidden = false;
                for &sup in &ov.supersedes {
                    if !self.visible(xmin, sup) {
                        hidden = true;
                        break;
                    }
                }
                if hidden {
                    continue;
                }
            }

            if overlay.is_none() && !row_bounds.is_empty() {
                let mut skip = false;
                for &(ci, vs, lo, hi, signed) in &row_bounds {
                    let (bytes, nullbm, _) = &decoded_cols[ci];
                    let is_null = !nullbm.is_empty() && (nullbm[r / 8] >> (r % 8)) & 1 == 1;
                    if is_null {
                        continue;
                    }
                    let slot = &bytes[r * vs..(r + 1) * vs];
                    let mut v: u128 = 0;
                    for (k, b) in slot.iter().enumerate() {
                        v |= (*b as u128) << (8 * k);
                    }
                    if signed && vs < 16 {
                        // Sign-extend a vs-byte two's complement value so a
                        // negative row value stays below a negative bound.
                        let sign_bit = 1u128 << (8 * vs - 1);
                        if v & sign_bit != 0 {
                            v |= !((1u128 << (8 * vs)) - 1);
                        }
                    }
                    let v = v as i128;
                    if v < lo || v > hi {
                        skip = true;
                        break;
                    }
                }
                if skip {
                    continue;
                }
            }

            for (ci, p) in self.col_plans.iter().enumerate() {
                if absent_cols[ci] {
                    // A patch written after the fold still applies: the
                    // column exists now, so an update to it lands in the
                    // overlay even though the segment predates it
                    let sv = self.resolve_value(
                        overlay,
                        p.column_id,
                        p.type_id,
                        p.value_size,
                        true,
                        None,
                    );
                    builders[ci].push_owned(if sv.is_null() { p.absent.clone() } else { sv });
                    continue;
                }
                let (bytes, nullbm, is_varlen) = &decoded_cols[ci];
                let is_null = !nullbm.is_empty() && (nullbm[r / 8] >> (r % 8)) & 1 == 1;
                let base_bytes: Option<&[u8]> = if is_null {
                    None
                } else if *is_varlen {
                    Some(varlen_rows[ci].as_ref().expect("varlen rows")[r])
                } else {
                    let vs = p.value_size;
                    Some(&bytes[r * vs..(r + 1) * vs])
                };
                // push_owned moves a decoded text or binary allocation into
                // the column instead of copying it a second time
                let sv = self.resolve_value(
                    overlay,
                    p.column_id,
                    p.type_id,
                    p.value_size,
                    is_null,
                    base_bytes,
                );
                builders[ci].push_owned(sv);
            }
            for (k, (sr, nullbm, rows)) in shred_cols.iter().enumerate() {
                // A row whose variant was patched after the fold is answered
                // from the patched document, because the stored column
                // describes what the row held when the segment was written
                let patched = overlay
                    .map(|ov| ov.patches.contains_key(&(sr.variant_column_id as u32)))
                    .unwrap_or(false);
                let value = if patched {
                    self.extracted_from_document(overlay, sr, r, &decoded_cols, &varlen_rows)
                } else if !nullbm.is_empty() && (nullbm[r / 8] >> (r % 8)) & 1 == 1 {
                    None
                } else {
                    match std::str::from_utf8(rows[r]) {
                        Ok(text) => Some(text.to_string()),
                        // The stored value was written from text, so bytes
                        // that are not text mean this column is damaged.
                        // Reading the document is slower and right
                        Err(_) => self.extracted_from_document(
                            overlay,
                            sr,
                            r,
                            &decoded_cols,
                            &varlen_rows,
                        ),
                    }
                };
                match value {
                    Some(text) => shred_builders[k].push_owned(ScalarValue::Utf8(text)),
                    None => shred_builders[k].push_owned(ScalarValue::Null),
                }
            }
            if self.emit_locators {
                locators.push((file_id, sys_rowid));
            }
            in_batch += 1;
            rows_yielded += 1;

            if in_batch == BATCH_SIZE {
                let batch = finalize_builders(std::mem::replace(
                    &mut builders,
                    create_builders(&self.output_columns, BATCH_SIZE),
                ));
                let resolved =
                    take_resolved(&mut shred_builders, &shred_cols, self.table_idx, BATCH_SIZE);
                let locs = std::mem::take(&mut locators);
                self.queue_batch(batch.with_resolved(resolved), locs)?;
                in_batch = 0;
            }
        }

        if in_batch > 0 {
            let batch = finalize_builders(builders);
            let resolved = take_resolved(&mut shred_builders, &shred_cols, self.table_idx, 0);
            self.queue_batch(batch.with_resolved(resolved), locators)?;
        }
        if let Some(stats) = &self.io_stats {
            stats.record_seq_batch(rows_yielded, segment_bytes);
        }
        Ok(())
    }

    /// The path read out of one row's document, resolved through the patch
    /// overlay so a row rewritten after the fold answers with what it holds
    /// now. This is the answer the stored column stands in for
    fn extracted_from_document(
        &self,
        overlay: Option<&RowOverlay>,
        shred: &ShredRead,
        row: usize,
        decoded_cols: &[(Vec<u8>, Vec<u8>, bool)],
        varlen_rows: &[Option<Vec<&[u8]>>],
    ) -> Option<String> {
        let plan = &self.col_plans[shred.variant_plan_idx];
        let (bytes, nullbm, is_varlen) = &decoded_cols[shred.variant_plan_idx];
        let is_null = !nullbm.is_empty() && (nullbm[row / 8] >> (row % 8)) & 1 == 1;
        let base_bytes: Option<&[u8]> = if is_null {
            None
        } else if *is_varlen {
            Some(varlen_rows[shred.variant_plan_idx].as_ref()?[row])
        } else {
            let vs = plan.value_size;
            Some(&bytes[row * vs..(row + 1) * vs])
        };
        match self.resolve_value(
            overlay,
            plan.column_id,
            plan.type_id,
            plan.value_size,
            is_null,
            base_bytes,
        ) {
            ScalarValue::Utf8(text) => {
                crate::variant_shred::extract_scalar_text(&text, &shred.path)
            }
            _ => None,
        }
    }

    fn queue_batch(&mut self, batch: DataBatch, locators: Vec<(u64, u64)>) -> Result<()> {
        let (filtered, kept_locs) = if let Some(ref predicate) = self.predicate {
            let mask_col = evaluate(predicate, &batch, &self.output_columns, &self.ctx.params)?;
            let mask = column_to_mask(&mask_col);
            let kept = if self.emit_locators {
                mask.iter()
                    .zip(locators.iter())
                    .filter_map(|(&k, l)| if k { Some(*l) } else { None })
                    .collect()
            } else {
                Vec::new()
            };
            (batch.filter(&mask), kept)
        } else {
            (batch, locators)
        };
        if filtered.num_rows == 0 {
            return Ok(());
        }
        let secured = apply_column_security(
            &self.ctx,
            self.table_entry.id.0,
            &self.output_columns,
            filtered,
        );
        if self.emit_locators {
            self.pending
                .push_back(ExecutionBatch::with_columnar_locators(secured, kept_locs));
        } else {
            self.pending.push_back(ExecutionBatch::new(secured));
        }
        Ok(())
    }
}

/// Finishes the current batch's extracted values and starts fresh builders
/// for the next one
fn take_resolved(
    builders: &mut Vec<ColumnBuilder>,
    shreds: &[(&ShredRead, &[u8], Vec<&[u8]>)],
    table_idx: Option<usize>,
    capacity: usize,
) -> Vec<ResolvedPath> {
    let Some(table_idx) = table_idx else {
        return Vec::new();
    };
    if builders.is_empty() {
        return Vec::new();
    }
    let fresh: Vec<ColumnBuilder> = builders
        .iter()
        .map(|_| ColumnBuilder::new(zyron_common::types::TypeId::Text, capacity))
        .collect();
    std::mem::replace(builders, fresh)
        .into_iter()
        .zip(shreds)
        .map(|(b, (sr, _, _))| ResolvedPath {
            table_idx,
            column_id: sr.variant_column_id,
            path: sr.path.clone(),
            values: b.finish(),
        })
        .collect()
}

impl Operator for ColumnScanOperator {
    fn next(&mut self) -> OperatorResult<'_> {
        Box::pin(async move {
            loop {
                if let Some(b) = self.pending.pop_front() {
                    return Ok(Some(b));
                }
                if self.finished || self.seg_idx >= self.segments.len() {
                    self.finished = true;
                    return Ok(None);
                }
                let (file_id, path) = self.segments[self.seg_idx].clone();
                self.seg_idx += 1;
                self.load_segment(file_id, &path)?;
            }
        })
    }
}

/// Hybrid scan: the union of the columnar segments and the heap residual.
/// A folded row is physically deleted from the heap at fold time, so the
/// heap scan returns only not-yet-folded rows and the columnar scan returns
/// only folded rows. The two sets are disjoint per snapshot, so draining
/// columnar then heap is an exact, double-count-free union.
pub struct HybridScanOperator {
    columnar: ColumnScanOperator,
    heap: crate::operator::scan::SeqScanOperator,
    columnar_done: bool,
}

impl HybridScanOperator {
    pub fn new(columnar: ColumnScanOperator, heap: crate::operator::scan::SeqScanOperator) -> Self {
        Self {
            columnar,
            heap,
            columnar_done: false,
        }
    }
}

impl Operator for HybridScanOperator {
    fn next(&mut self) -> OperatorResult<'_> {
        Box::pin(async move {
            let _scan =
                zyron_common::profile::scope(zyron_common::profile::Phase::ExecHybridScanNext);
            if !self.columnar_done {
                match self.columnar.next().await? {
                    Some(b) => return Ok(Some(b)),
                    None => self.columnar_done = true,
                }
            }
            self.heap.next().await
        })
    }
}

// ---------------------------------------------------------------------------
// Columnar metadata aggregate
// ---------------------------------------------------------------------------

use zyron_planner::physical::{MetaAggKind, MetaAggSpec};

/// Answers ungrouped MIN/MAX/COUNT from columnar segment headers plus the
/// heap residual, without decoding the folded rows. When a table's patch
/// overlay is non-empty the columnar side falls back to a full columnar scan
/// (overlay-resolved) so the result stays MVCC-correct; the heap residual is
/// always aggregated by a real scan. Clean cold data takes the header-only
/// fast path, which is the orders-of-magnitude win.
pub struct ColumnarMetadataAggregateOperator {
    ctx: Arc<ExecutionContext>,
    table_id: zyron_catalog::TableId,
    specs: Vec<MetaAggSpec>,
    schema: Vec<LogicalColumn>,
    done: bool,
}

impl ColumnarMetadataAggregateOperator {
    pub fn new(
        ctx: Arc<ExecutionContext>,
        table_id: zyron_catalog::TableId,
        specs: Vec<MetaAggSpec>,
        schema: Vec<LogicalColumn>,
    ) -> Self {
        Self {
            ctx,
            table_id,
            specs,
            schema,
            done: false,
        }
    }
}

impl Operator for ColumnarMetadataAggregateOperator {
    fn next(&mut self) -> OperatorResult<'_> {
        Box::pin(async move {
            if self.done {
                return Ok(None);
            }
            self.done = true;

            let te = self.ctx.get_table_entry(self.table_id)?;

            // Projection: the distinct target columns, or the first column as
            // a driver when only COUNT(*) is requested.
            let mut proj: Vec<LogicalColumn> = Vec::new();
            let mut col_to_proj: std::collections::HashMap<u16, usize> =
                std::collections::HashMap::new();
            for s in &self.specs {
                if let Some(cid) = s.column_id
                    && !col_to_proj.contains_key(&cid.0)
                {
                    let ce = te.columns.iter().find(|c| c.id == cid).ok_or_else(|| {
                        zyron_common::ZyronError::ExecutionError(
                            "metadata aggregate: column not found".into(),
                        )
                    })?;
                    col_to_proj.insert(cid.0, proj.len());
                    proj.push(LogicalColumn {
                        table_idx: Some(0),
                        column_id: ce.id,
                        name: ce.name.clone(),
                        type_id: ce.type_id,
                        nullable: ce.nullable,
                        fractional_digits: ce.fractional_digits,
                    });
                }
            }
            if proj.is_empty() {
                if let Some(ce) = te.columns.first() {
                    proj.push(LogicalColumn {
                        table_idx: Some(0),
                        column_id: ce.id,
                        name: ce.name.clone(),
                        type_id: ce.type_id,
                        nullable: ce.nullable,
                        fractional_digits: ce.fractional_digits,
                    });
                }
            }
            let proj_idx: Vec<Option<usize>> = self
                .specs
                .iter()
                .map(|s| s.column_id.and_then(|c| col_to_proj.get(&c.0).copied()))
                .collect();

            let mut accs = MetaAcc::for_specs(&self.specs);

            // Columnar contribution.
            let segments = &te.columnar.segments;
            let store = match segments.first() {
                Some(s0) => Some(
                    zyron_storage::columnar::ColumnarPatchManager::store_for_segment(
                        self.table_id.0 as u64,
                        std::path::Path::new(&s0.path),
                    )?,
                ),
                None => None,
            };
            // Per-segment: a clean segment is answered from its header with
            // no decode; only segments that actually carry overlay entries
            // are resolved by a scan. One UPDATE no longer disables the
            // metadata fast path for the whole table.
            let branch = self.ctx.active_branch_id.unwrap_or(0);
            let mut dirty: std::collections::HashSet<u64> = std::collections::HashSet::new();
            if let Some(s) = &store {
                for seg in segments {
                    if s.file_has_overlay_on(branch, seg.file_id) {
                        dirty.insert(seg.file_id);
                    }
                }
            }

            if !dirty.is_empty() {
                let cs = ColumnScanOperator::new_for_files(
                    self.ctx.clone(),
                    self.table_id,
                    proj.clone(),
                    None,
                    dirty.clone(),
                )?;
                fold_rows_into_meta_accs(Box::new(cs), &self.specs, &proj_idx, &mut accs).await?;
            }
            {
                // Clean segments: answer from segment headers, no row decode.
                for seg in segments.iter().filter(|s| !dirty.contains(&s.file_id)) {
                    let reader = ZyrFileReader::open(std::path::Path::new(&seg.path))?;
                    let rc = reader.row_count() as i64;
                    for (si, spec) in self.specs.iter().enumerate() {
                        match (&spec.kind, &mut accs[si]) {
                            (MetaAggKind::CountStar, MetaAcc::Count(c)) => *c += rc,
                            (MetaAggKind::CountCol, MetaAcc::Count(c)) => {
                                if let Some(cid) = spec.column_id {
                                    let h = reader.read_segment_header(cid.0 as u32)?;
                                    *c += rc - h.null_count as i64;
                                }
                            }
                            (MetaAggKind::Min, MetaAcc::MinMax(m))
                            | (MetaAggKind::Max, MetaAcc::MinMax(m)) => {
                                if let Some(cid) = spec.column_id {
                                    let ce = te.columns.iter().find(|c| c.id == cid).ok_or_else(
                                        || {
                                            zyron_common::ZyronError::ExecutionError(
                                                "meta agg: column missing".into(),
                                            )
                                        },
                                    )?;
                                    let phys = ce.physical_type_id();
                                    let sz = phys.fixed_size().ok_or_else(|| {
                                        zyron_common::ZyronError::ExecutionError(
                                            "meta agg: non-fixed column".into(),
                                        )
                                    })?;
                                    let h = reader.read_segment_header(cid.0 as u32)?;
                                    if h.null_count < reader.row_count() {
                                        let slot = if spec.kind == MetaAggKind::Max {
                                            &h.max_value
                                        } else {
                                            &h.min_value
                                        };
                                        let sv = decode_fixed_scalar(phys, &slot[..sz]);
                                        MetaAcc::fold_minmax(m, sv, spec.kind == MetaAggKind::Max);
                                    }
                                }
                            }
                            _ => {}
                        }
                    }
                }
            }

            // Heap residual contribution (rows not yet folded). When the heap
            // file has zero pages no unfolded row can exist, so the scan is
            // skipped entirely. This is exact, not heuristic: a heap with no
            // allocated pages holds no tuples, so the aggregate cannot miss a
            // row. A fully folded heap still keeps its pages (fold zeroes slot
            // lengths, it does not free pages), so that case still scans.
            let heap_pages = self
                .ctx
                .get_heap_file(self.table_id)
                .await?
                .num_pages_cached();
            if heap_pages > 0 {
                let heap = crate::operator::scan::SeqScanOperator::new(
                    self.ctx.clone(),
                    self.table_id,
                    proj.clone(),
                    None,
                    false,
                    None,
                )
                .await?;
                fold_rows_into_meta_accs(Box::new(heap), &self.specs, &proj_idx, &mut accs).await?;
            }

            // Materialize the single result row. MIN and MAX expose an
            // actual column value, so they honor the same column level
            // classification and masking the row scan path enforces. COUNT
            // exposes no value and is left intact.
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

/// Conservatively derives the inclusive integer range `[lo, hi]` that column
/// `col` must lie in for `e` to possibly be true. `None` on a side means
/// unbounded there; `(None, None)` means "no usable constraint" (the caller
/// then does not skip and scans normally, so this is always correctness
/// safe). Only AND of simple `col CMP int-literal` comparisons is analyzed.
/// Whether reading a segment's value bloom can save more than it costs.
///
/// Rejecting a segment here skips decoding every projected column of it and
/// the three sys columns beside them, and the row loop over all of that, so
/// what the probe avoids is the decoded size rather than the encoded one. A
/// filter as wide as the values it would materialize is where that stops
/// being a saving. Both sizes are in the header the probe has already read
fn bloom_worth_reading(header: &zyron_storage::columnar::SegmentHeader) -> bool {
    header.bloom_filter_size > 0 && u64::from(header.bloom_filter_size) < header.raw_size
}

/// The stored cell an integer-backed constant equals, or None when the
/// constant does not fit the column's width.
///
/// A segment records its cells as `width` little endian bytes and its bloom
/// holds those bytes, so a probe has to present the same. A constant the
/// width cannot hold has no cell to present, and the bounds check has
/// already decided that segment anyway
fn int_cell_bytes(value: i128, width: usize, signed: bool) -> Option<Vec<u8>> {
    if width == 0 || width > 16 {
        return None;
    }
    if width < 16 {
        let bits = 8 * width as u32;
        let fits = if signed {
            let limit = 1i128 << (bits - 1);
            (-limit..limit).contains(&value)
        } else {
            (0..(1i128 << bits)).contains(&value)
        };
        if !fits {
            return None;
        }
    } else if !signed && value < 0 {
        return None;
    }
    Some(value.to_le_bytes()[..width].to_vec())
}

/// Byte-string values a column has to hold for the predicate to select any
/// row, as one group per term.
///
/// A row satisfies the predicate only if it satisfies every group, so a
/// group whose members a segment provably holds none of is a term no row of
/// that segment satisfies. A disjunction contributes nothing: its arms are
/// alternatives and rejecting one says nothing about the other
fn predicate_equal_bytes(e: &BoundExpr, col: u32) -> Vec<Vec<Vec<u8>>> {
    let is_col = |x: &BoundExpr| matches!(x, BoundExpr::ColumnRef(ColumnRef { column_id, .. }) if column_id.0 as u32 == col);
    let as_str = |x: &BoundExpr| match x {
        BoundExpr::Literal {
            value: LiteralValue::String(s),
            ..
        } => Some(s.as_bytes().to_vec()),
        _ => None,
    };
    match e {
        BoundExpr::Nested(inner) => predicate_equal_bytes(inner, col),
        BoundExpr::BinaryOp {
            left,
            op: BinaryOperator::And,
            right,
            ..
        } => {
            let mut groups = predicate_equal_bytes(left, col);
            groups.extend(predicate_equal_bytes(right, col));
            groups
        }
        BoundExpr::BinaryOp {
            left,
            op: BinaryOperator::Eq,
            right,
            ..
        } => {
            let cell = if is_col(left) {
                as_str(right)
            } else if is_col(right) {
                as_str(left)
            } else {
                None
            };
            cell.map(|c| vec![vec![c]]).unwrap_or_default()
        }
        BoundExpr::InList {
            expr,
            list,
            negated: false,
            ..
        } if is_col(expr) => {
            let mut members = Vec::with_capacity(list.len());
            for item in list {
                // One member with no byte form makes the whole membership
                // unprovable, since a row it admits must not be dropped
                match as_str(item) {
                    Some(c) => members.push(c),
                    None => return Vec::new(),
                }
            }
            if members.is_empty() {
                Vec::new()
            } else {
                vec![members]
            }
        }
        _ => Vec::new(),
    }
}

fn predicate_int_bounds(e: &BoundExpr, col: u32) -> (Option<i128>, Option<i128>) {
    fn col_lit<'a>(l: &'a BoundExpr, r: &'a BoundExpr, col: u32) -> Option<(bool, i128)> {
        // Returns (col_on_left, literal) when exactly one side is the target
        // column ref and the other is an integer literal.
        let as_col = |x: &BoundExpr| matches!(x, BoundExpr::ColumnRef(ColumnRef { column_id, .. }) if column_id.0 as u32 == col);
        let as_int = |x: &BoundExpr| match x {
            BoundExpr::Literal {
                value: LiteralValue::Integer(v),
                ..
            } => Some(*v as i128),
            _ => None,
        };
        if as_col(l) {
            as_int(r).map(|v| (true, v))
        } else if as_col(r) {
            as_int(l).map(|v| (false, v))
        } else {
            None
        }
    }
    match e {
        BoundExpr::Nested(inner) => predicate_int_bounds(inner, col),
        BoundExpr::BinaryOp {
            left, op, right, ..
        } => match op {
            BinaryOperator::And => {
                let (l1, h1) = predicate_int_bounds(left, col);
                let (l2, h2) = predicate_int_bounds(right, col);
                let lo = match (l1, l2) {
                    (Some(a), Some(b)) => Some(a.max(b)),
                    (a, b) => a.or(b),
                };
                let hi = match (h1, h2) {
                    (Some(a), Some(b)) => Some(a.min(b)),
                    (a, b) => a.or(b),
                };
                (lo, hi)
            }
            BinaryOperator::Eq => match col_lit(left, right, col) {
                Some((_, v)) => (Some(v), Some(v)),
                None => (None, None),
            },
            BinaryOperator::Lt => match col_lit(left, right, col) {
                Some((true, v)) => (None, Some(v - 1)),
                Some((false, v)) => (Some(v + 1), None),
                None => (None, None),
            },
            BinaryOperator::LtEq => match col_lit(left, right, col) {
                Some((true, v)) => (None, Some(v)),
                Some((false, v)) => (Some(v), None),
                None => (None, None),
            },
            BinaryOperator::Gt => match col_lit(left, right, col) {
                Some((true, v)) => (Some(v + 1), None),
                Some((false, v)) => (None, Some(v - 1)),
                None => (None, None),
            },
            BinaryOperator::GtEq => match col_lit(left, right, col) {
                Some((true, v)) => (Some(v), None),
                Some((false, v)) => (None, Some(v)),
                None => (None, None),
            },
            _ => (None, None),
        },
        _ => (None, None),
    }
}
