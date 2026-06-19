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
    ColumnarPatchManager, PatchStore, RowOverlay, SEGMENT_HEADER_SIZE, SYS_COL_ROWID,
    SYS_COL_SUPERSEDE, SYS_COL_XMIN, SegmentHeader, ZONE_MAP_BATCH_SIZE, ZONE_MAP_ENTRY_SIZE,
    ZyrFileReader,
};
use zyron_storage::encoding::{create_encoding, varlen_slice_rows};

use crate::batch::{
    BATCH_SIZE, DataBatch, create_builders, decode_fixed_scalar, decode_varlen_scalar,
    finalize_builders,
};
use crate::column::ScalarValue;
use crate::compute::column_to_mask;
use crate::context::ExecutionContext;
use crate::expr::evaluate;
use crate::operator::{ExecutionBatch, Operator, OperatorResult, apply_column_security};

/// Per-projected-column decode plan.
struct ColPlan {
    column_id: u32,
    type_id: zyron_common::types::TypeId,
    /// Fixed byte width, or 0 for the variable-length canonical layout.
    value_size: usize,
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
            Some((_, p)) => {
                let dir = std::path::Path::new(p)
                    .parent()
                    .map(|d| d.to_path_buf())
                    .unwrap_or_else(|| std::path::PathBuf::from("."));
                Some(ColumnarPatchManager::global(&dir).store(table_entry.id.0 as u64)?)
            }
            None => None,
        };

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
        })
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

    fn decode_column(
        reader: &ZyrFileReader,
        column_id: u32,
        row_count: usize,
        value_size: usize,
    ) -> Result<(Vec<u8>, Vec<u8>)> {
        let raw = reader.read_segment_raw(column_id)?;
        Self::decode_raw(&raw, row_count, value_size)
    }

    /// Parses the header regions out of an already-read raw segment buffer
    /// and decodes the encoded column. Lets the scan read every needed
    /// column with one file open (see `read_segments_each`).
    fn decode_raw(raw: &[u8], row_count: usize, value_size: usize) -> Result<(Vec<u8>, Vec<u8>)> {
        if raw.len() < SEGMENT_HEADER_SIZE {
            return Err(zyron_common::ZyronError::ExecutionError(
                "columnar scan: segment shorter than header".into(),
            ));
        }
        let mut hdr_buf = [0u8; SEGMENT_HEADER_SIZE];
        hdr_buf.copy_from_slice(&raw[..SEGMENT_HEADER_SIZE]);
        let hdr = SegmentHeader::from_bytes(&hdr_buf)?;

        let bloom_len = hdr.bloom_filter_size as usize;
        let zones = row_count.div_ceil(ZONE_MAP_BATCH_SIZE as usize);
        let zonemap_len = zones * ZONE_MAP_ENTRY_SIZE;
        let nullbm_len = if hdr.null_count > 0 {
            row_count.div_ceil(8)
        } else {
            0
        };
        let enc_start = SEGMENT_HEADER_SIZE + bloom_len + zonemap_len + nullbm_len;
        let enc_end = enc_start + hdr.encoded_size as usize;
        if enc_end > raw.len() {
            return Err(zyron_common::ZyronError::ExecutionError(
                "columnar scan: encoded region exceeds segment".into(),
            ));
        }
        let null_bitmap = raw[SEGMENT_HEADER_SIZE + bloom_len + zonemap_len..enc_start].to_vec();
        // Granular integrity: CRC the encoded payload over exactly the bytes
        // about to be decoded (already in memory, zero extra IO). The header
        // self-verified its own header_crc in from_bytes.
        let enc = &raw[enc_start..enc_end];
        let crc = zyron_common::hash32(enc);
        if crc != hdr.data_checksum {
            return Err(zyron_common::ZyronError::InvalidZyrFile(format!(
                "columnar segment payload checksum mismatch: stored 0x{:08x}, computed 0x{:08x}",
                hdr.data_checksum, crc
            )));
        }
        let encoder = create_encoding(hdr.encoding_type);
        let decoded = encoder.decode(enc, row_count, value_size)?;
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
            let dirty = self
                .patch_store
                .as_ref()
                .map(|s| s.file_has_overlay(file_id))
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
                    let hb = reader.read_segment_header_bytes(p.column_id)?;
                    let h = SegmentHeader::from_bytes(&hb)?;
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
                        return Ok(());
                    }
                }
            }
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
        // Read+decode+drop one column at a time so peak raw memory is a
        // single segment instead of every requested segment held at once.
        // Decoded buffers stay resident because row iteration is row-major
        // across all projected columns. col_ids order is sys columns then
        // projected columns, matching the index passed to the callback.
        let mut decoded: Vec<Option<(Vec<u8>, Vec<u8>)>> =
            (0..col_ids.len()).map(|_| None).collect();
        reader.read_segments_each(&col_ids, |idx, bytes| {
            let raw = bytes.ok_or_else(|| {
                zyron_common::ZyronError::ExecutionError(
                    "columnar scan: missing segment for column".into(),
                )
            })?;
            let value_size = if idx < 3 {
                8
            } else {
                self.col_plans[idx - 3].value_size
            };
            decoded[idx] = Some(Self::decode_raw(raw, row_count, value_size)?);
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

        let mut decoded_cols: Vec<(Vec<u8>, Vec<u8>, bool)> =
            Vec::with_capacity(self.col_plans.len());
        for (k, p) in self.col_plans.iter().enumerate() {
            let (bytes, nullbm) = take(&mut decoded[3 + k])?;
            decoded_cols.push((bytes, nullbm, p.value_size == 0));
        }
        let mut varlen_rows: Vec<Option<Vec<&[u8]>>> = Vec::with_capacity(decoded_cols.len());
        for (bytes, _, is_varlen) in &decoded_cols {
            if *is_varlen {
                varlen_rows.push(Some(varlen_slice_rows(bytes, row_count)?));
            } else {
                varlen_rows.push(None);
            }
        }

        // Snapshot this file's overlay once under a single lock, instead of
        // a per-row lock acquisition and clone.
        let overlay_map = match &self.patch_store {
            Some(s) if s.file_has_overlay(file_id) => Some(s.file_overlay(file_id)),
            _ => None,
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
        let mut locators: Vec<(u64, u64)> = Vec::new();
        let mut in_batch = 0usize;

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
                let sv = self.resolve_value(
                    overlay,
                    p.column_id,
                    p.type_id,
                    p.value_size,
                    is_null,
                    base_bytes,
                );
                builders[ci].push(&sv);
            }
            if self.emit_locators {
                locators.push((file_id, sys_rowid));
            }
            in_batch += 1;

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
        Ok(())
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

enum Acc {
    Count(i64),
    MinMax(Option<ScalarValue>),
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

    fn fold_minmax(cur: &mut Option<ScalarValue>, v: ScalarValue, want_max: bool) {
        if matches!(v, ScalarValue::Null) {
            return;
        }
        match cur {
            None => *cur = Some(v),
            Some(c) => {
                if let Some(ord) = v.partial_cmp(c) {
                    let take = if want_max {
                        ord == std::cmp::Ordering::Greater
                    } else {
                        ord == std::cmp::Ordering::Less
                    };
                    if take {
                        *cur = Some(v);
                    }
                }
            }
        }
    }

    async fn aggregate_scan(
        &self,
        mut op: Box<dyn Operator>,
        proj_idx: &[Option<usize>],
        accs: &mut [Acc],
    ) -> Result<()> {
        while let Some(eb) = op.next().await? {
            let b = &eb.batch;
            for (si, spec) in self.specs.iter().enumerate() {
                match (&spec.kind, &mut accs[si]) {
                    (MetaAggKind::CountStar, Acc::Count(c)) => {
                        *c += b.num_rows as i64;
                    }
                    (MetaAggKind::CountCol, Acc::Count(c)) => {
                        if let Some(ci) = proj_idx[si] {
                            let col = &b.columns[ci];
                            for r in 0..b.num_rows {
                                if !col.is_null(r) {
                                    *c += 1;
                                }
                            }
                        }
                    }
                    (MetaAggKind::Min, Acc::MinMax(m)) | (MetaAggKind::Max, Acc::MinMax(m)) => {
                        if let Some(ci) = proj_idx[si] {
                            let want_max = spec.kind == MetaAggKind::Max;
                            let col = &b.columns[ci];
                            for r in 0..b.num_rows {
                                if !col.is_null(r) {
                                    Self::fold_minmax(m, col.get_scalar(r), want_max);
                                }
                            }
                        }
                    }
                    _ => {}
                }
            }
        }
        Ok(())
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
                        ts_precision: ce.ts_precision,
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
                        ts_precision: ce.ts_precision,
                    });
                }
            }
            let proj_idx: Vec<Option<usize>> = self
                .specs
                .iter()
                .map(|s| s.column_id.and_then(|c| col_to_proj.get(&c.0).copied()))
                .collect();

            let mut accs: Vec<Acc> = self
                .specs
                .iter()
                .map(|s| match s.kind {
                    MetaAggKind::CountStar | MetaAggKind::CountCol => Acc::Count(0),
                    MetaAggKind::Min | MetaAggKind::Max => Acc::MinMax(None),
                })
                .collect();

            // Columnar contribution.
            let segments = &te.columnar.segments;
            let store = match segments.first() {
                Some(s0) => {
                    let dir = std::path::Path::new(&s0.path)
                        .parent()
                        .map(|d| d.to_path_buf())
                        .unwrap_or_else(|| std::path::PathBuf::from("."));
                    Some(
                        zyron_storage::columnar::ColumnarPatchManager::global(&dir)
                            .store(self.table_id.0 as u64)?,
                    )
                }
                None => None,
            };
            // Per-segment: a clean segment is answered from its header with
            // no decode; only segments that actually carry overlay entries
            // are resolved by a scan. One UPDATE no longer disables the
            // metadata fast path for the whole table.
            let mut dirty: std::collections::HashSet<u64> = std::collections::HashSet::new();
            if let Some(s) = &store {
                for seg in segments {
                    if s.file_has_overlay(seg.file_id) {
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
                self.aggregate_scan(Box::new(cs), &proj_idx, &mut accs)
                    .await?;
            }
            {
                // Clean segments: answer from segment headers, no row decode.
                for seg in segments.iter().filter(|s| !dirty.contains(&s.file_id)) {
                    let reader = ZyrFileReader::open(std::path::Path::new(&seg.path))?;
                    let rc = reader.row_count() as i64;
                    for (si, spec) in self.specs.iter().enumerate() {
                        match (&spec.kind, &mut accs[si]) {
                            (MetaAggKind::CountStar, Acc::Count(c)) => *c += rc,
                            (MetaAggKind::CountCol, Acc::Count(c)) => {
                                if let Some(cid) = spec.column_id {
                                    let hb = reader.read_segment_header_bytes(cid.0 as u32)?;
                                    let h = SegmentHeader::from_bytes(&hb)?;
                                    *c += rc - h.null_count as i64;
                                }
                            }
                            (MetaAggKind::Min, Acc::MinMax(m))
                            | (MetaAggKind::Max, Acc::MinMax(m)) => {
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
                                    let hb = reader.read_segment_header_bytes(cid.0 as u32)?;
                                    let h = SegmentHeader::from_bytes(&hb)?;
                                    if h.null_count < reader.row_count() {
                                        let slot = if spec.kind == MetaAggKind::Max {
                                            &h.max_value
                                        } else {
                                            &h.min_value
                                        };
                                        let sv = decode_fixed_scalar(phys, &slot[..sz]);
                                        Self::fold_minmax(m, sv, spec.kind == MetaAggKind::Max);
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
                self.aggregate_scan(Box::new(heap), &proj_idx, &mut accs)
                    .await?;
            }

            // Materialize the single result row. MIN/MAX expose an actual
            // column value, so they must honor the same column-level
            // classification/masking the row-scan path enforces: if the
            // session is not cleared for the column, or a masking policy
            // applies to it, deny by returning NULL. COUNT does not expose a
            // value and is left intact.
            let table_id = self.table_id.0;
            let sec = self
                .ctx
                .security_context
                .as_ref()
                .zip(self.ctx.security_manager.as_ref());
            let mut builders = create_builders(&self.schema, 1);
            for (si, acc) in accs.into_iter().enumerate() {
                let sv = match acc {
                    Acc::Count(c) => ScalarValue::Int64(c),
                    Acc::MinMax(m) => {
                        let mut v = m.unwrap_or(ScalarValue::Null);
                        if let (Some(cid), Some((sc, sm))) = (self.specs[si].column_id, sec) {
                            let cleared = sm.classification_store.check_clearance(
                                sc.clearance,
                                table_id,
                                cid.0,
                            );
                            let mut probe = String::new();
                            let has_mask = sm.masking_policy_store.apply_masking(
                                table_id,
                                cid.0,
                                "",
                                &sc.effective_roles,
                                &mut probe,
                            );
                            if !cleared || has_mask {
                                v = ScalarValue::Null;
                            }
                        }
                        v
                    }
                };
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
