//! Reading a heap tuple through the layout it was written under.
//!
//! A tuple's bytes carry no column count and no type list. The slot carries a
//! schema epoch, and the table records what each epoch's positional layout
//! was, so the two together say how to walk the row. Reading a row through the
//! current column list instead is what makes an added column misalign every
//! byte after it, which is why the epoch is resolved before the first byte is
//! read rather than assumed.
//!
//! Three things can differ between the layout a row was written under and the
//! table as it stands now. A column the row predates is filled from the value
//! recorded when it was added. A column since dropped still occupies its
//! position, so the cursor walks it and pushes nothing, or NULL when the
//! projection still names the column, which a write path reading a row image
//! over the table's whole column list does. A column whose type widened is
//! decoded at the width the bytes carry and widened on the way into the
//! builder.
//!
//! The work of deciding all three is done once per (table, epoch, projection)
//! and kept as a plan, so the per-row path is a table lookup and a walk.
//!
//! A change data feed configured with a column subset records each row
//! narrowed to the subset in force at its epoch, and flags the record. Such a
//! record decodes through the epoch's layout narrowed the same way, which is
//! a second set of plans over the same epochs

use std::collections::{HashMap, HashSet};

use zyron_catalog::{ColumnEntry, ColumnId, PhysicalColumn, TableEntry};
use zyron_common::{Result, RowLocator, TypeId, ZyronError};

use crate::batch::{ColumnBuilder, decode_fixed_scalar, decode_varlen_scalar};
use crate::column::ScalarValue;

/// How a value read at the epoch's width becomes a value of the column's
/// current type.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Widen {
    /// The bytes already carry the current type
    None,
    /// A signed integer read at a narrower width, sign extended into the
    /// current one
    Signed(TypeId),
    /// An unsigned integer read at a narrower width
    Unsigned(TypeId),
    /// A microsecond timestamp read into a column that now keeps picoseconds
    MicrosToPicos,
    /// A scaled decimal whose declared scale grew, multiplied by the
    /// difference so the value it denotes is unchanged
    DecimalRescale { from: u8, to: u8 },
}

/// What the decoder does with one physical column of an epoch's layout.
#[derive(Debug, Clone)]
pub(crate) enum Step {
    /// Read the value and push it into a builder
    Take {
        builder: u16,
        /// Width and length prefix the bytes carry
        physical: TypeId,
        /// Logical type the value is interpreted as, which drives text and
        /// binary decoding
        logical: TypeId,
        widen: Widen,
        /// Ciphertext stays binary, decoding it as its logical text type
        /// would corrupt it through a lossy utf8 conversion
        encrypted: bool,
    },
    /// Walk the value to keep the cursor aligned and push nothing, which is
    /// what a column dropped since this epoch was written leaves behind
    Discard { physical: TypeId },
}

impl Step {
    /// The type the bytes of this column carry
    pub(crate) fn physical(&self) -> TypeId {
        match self {
            Step::Take { physical, .. } => *physical,
            Step::Discard { physical } => *physical,
        }
    }
}

/// Everything needed to read one epoch's rows into one projection.
#[derive(Debug, Clone)]
pub struct EpochPlan {
    /// Bytes of null bitmap the layout puts in front of the values
    bitmap_len: usize,
    /// One entry per physical column, in encoded order
    steps: Vec<Step>,
    /// Builders whose column the layout does not carry, with the value a row
    /// written before that column reads as
    absent: Vec<(u16, ScalarValue)>,
    /// Shortest a row of this layout can be: the bitmap, every fixed width,
    /// and the four length bytes each variable-width value carries. A row
    /// below this cannot span the layout whatever its contents
    min_len: usize,
    /// True when no step is variable-width, which makes `min_len` the exact
    /// length of every row and the bounds check one comparison
    all_fixed: bool,
    /// Step index of the first variable-width value, and the byte offset it
    /// sits at. Everything before it is fixed, so a bounds walk starts there
    /// rather than at the bitmap
    varlen_start: (usize, usize),
}

impl EpochPlan {
    /// One step per physical column of the layout, in encoded order, which
    /// is what a reader holding the columns apart rather than in rows
    /// follows
    pub(crate) fn steps(&self) -> &[Step] {
        &self.steps
    }

    /// The builders whose column the layout does not carry, with the value
    /// every row of the layout reads for them
    pub(crate) fn absent(&self) -> &[(u16, ScalarValue)] {
        &self.absent
    }

    /// Whether `data` spans a whole row of this layout without running past
    /// its end. Mirrors the cursor advancement in `decode_into`, so a row
    /// that passes here decodes without an out-of-bounds index.
    #[inline]
    pub fn spans(&self, data: &[u8]) -> bool {
        // Every fixed width and every length prefix is known when the plan is
        // built, so a row shorter than their sum is refused without looking at
        // a step, and a layout with no variable-width value is settled outright
        if data.len() < self.min_len {
            return false;
        }
        if self.all_fixed {
            return true;
        }
        // The fixed prefix has already been covered by the length check, so the
        // walk starts at the first value whose width the bytes decide
        let (start_step, start_offset) = self.varlen_start;
        let mut offset = start_offset;
        for step in &self.steps[start_step..] {
            let physical = match step {
                Step::Take { physical, .. } => *physical,
                Step::Discard { physical } => *physical,
            };
            match physical.fixed_size() {
                Some(size) => {
                    offset += size;
                    if offset > data.len() {
                        return false;
                    }
                }
                None => {
                    if offset + 4 > data.len() {
                        return false;
                    }
                    let len = u32::from_le_bytes([
                        data[offset],
                        data[offset + 1],
                        data[offset + 2],
                        data[offset + 3],
                    ]) as usize;
                    offset += 4 + len;
                    if offset > data.len() {
                        return false;
                    }
                }
            }
        }
        true
    }

    /// Walks one row's bytes and fills the builders this plan was built for.
    #[inline]
    pub fn decode_into(&self, data: &[u8], builders: &mut [ColumnBuilder]) {
        let null_bitmap = &data[..self.bitmap_len];
        let mut offset = self.bitmap_len;

        for (i, step) in self.steps.iter().enumerate() {
            let is_null = (null_bitmap[i / 8] >> (i % 8)) & 1 == 1;
            match step {
                Step::Take {
                    builder,
                    physical,
                    logical,
                    widen,
                    encrypted,
                } => {
                    let b = *builder as usize;
                    if let Some(size) = physical.fixed_size() {
                        if is_null {
                            builders[b].push_null();
                        } else {
                            let bytes = &data[offset..offset + size];
                            match widen {
                                Widen::None => {
                                    if !builders[b].push_fixed(*physical, bytes) {
                                        let scalar = decode_fixed_scalar(*physical, bytes);
                                        builders[b].push_owned(scalar);
                                    }
                                }
                                _ => {
                                    let scalar = decode_fixed_scalar(*physical, bytes);
                                    builders[b].push_owned(apply_widen(*widen, scalar));
                                }
                            }
                        }
                        offset += size;
                    } else {
                        let len = u32::from_le_bytes([
                            data[offset],
                            data[offset + 1],
                            data[offset + 2],
                            data[offset + 3],
                        ]) as usize;
                        offset += 4;
                        if is_null {
                            builders[b].push_null();
                        } else {
                            let bytes = &data[offset..offset + len];
                            let scalar = if *encrypted {
                                ScalarValue::Binary(bytes.to_vec())
                            } else {
                                decode_varlen_scalar(*logical, bytes)
                            };
                            builders[b].push_owned(scalar);
                        }
                        offset += len;
                    }
                }
                Step::Discard { physical } => {
                    if let Some(size) = physical.fixed_size() {
                        offset += size;
                    } else {
                        let len = u32::from_le_bytes([
                            data[offset],
                            data[offset + 1],
                            data[offset + 2],
                            data[offset + 3],
                        ]) as usize;
                        offset += 4 + len;
                    }
                }
            }
        }

        for (builder, value) in &self.absent {
            builders[*builder as usize].push(value);
        }
    }
}

/// Widens one decoded value into the type its column now declares.
#[inline]
pub(crate) fn apply_widen(widen: Widen, scalar: ScalarValue) -> ScalarValue {
    let as_i128 = |s: &ScalarValue| -> Option<i128> {
        match s {
            ScalarValue::Int8(v) => Some(*v as i128),
            ScalarValue::Int16(v) => Some(*v as i128),
            ScalarValue::Int32(v) => Some(*v as i128),
            ScalarValue::Int64(v) => Some(*v as i128),
            ScalarValue::Int128(v) => Some(*v),
            _ => None,
        }
    };
    let as_u128 = |s: &ScalarValue| -> Option<u128> {
        match s {
            ScalarValue::UInt8(v) => Some(*v as u128),
            ScalarValue::UInt16(v) => Some(*v as u128),
            ScalarValue::UInt32(v) => Some(*v as u128),
            ScalarValue::UInt64(v) => Some(*v as u128),
            _ => None,
        }
    };
    match widen {
        Widen::None => scalar,
        Widen::Signed(target) => match (as_i128(&scalar), target) {
            (Some(v), TypeId::Int16) => ScalarValue::Int16(v as i16),
            (Some(v), TypeId::Int32) => ScalarValue::Int32(v as i32),
            (Some(v), TypeId::Int64) => ScalarValue::Int64(v as i64),
            (Some(v), TypeId::Int128) => ScalarValue::Int128(v),
            _ => scalar,
        },
        Widen::Unsigned(target) => match (as_u128(&scalar), target) {
            (Some(v), TypeId::UInt16) => ScalarValue::UInt16(v as u16),
            (Some(v), TypeId::UInt32) => ScalarValue::UInt32(v as u32),
            (Some(v), TypeId::UInt64) => ScalarValue::UInt64(v as u64),
            (Some(v), TypeId::UInt128) => ScalarValue::Int128(v as i128),
            _ => scalar,
        },
        Widen::MicrosToPicos => match scalar {
            ScalarValue::Int64(us) => ScalarValue::Int128(us as i128 * 1_000_000),
            other => other,
        },
        Widen::DecimalRescale { from, to } => match scalar {
            ScalarValue::Int128(v) => {
                let factor = 10i128.pow((to - from) as u32);
                ScalarValue::Int128(v.saturating_mul(factor))
            }
            other => other,
        },
    }
}

/// Rank of a signed integer type, used to tell a widening from a change of
/// family. None for every type that is not a signed integer.
fn signed_rank(t: TypeId) -> Option<u8> {
    match t {
        TypeId::Int8 => Some(1),
        TypeId::Int16 => Some(2),
        TypeId::Int32 => Some(3),
        TypeId::Int64 => Some(4),
        TypeId::Int128 => Some(5),
        _ => None,
    }
}

fn unsigned_rank(t: TypeId) -> Option<u8> {
    match t {
        TypeId::UInt8 => Some(1),
        TypeId::UInt16 => Some(2),
        TypeId::UInt32 => Some(3),
        TypeId::UInt64 => Some(4),
        TypeId::UInt128 => Some(5),
        _ => None,
    }
}

/// How a value stored at `from` reaches the column's current shape.
fn widening_for(from: &PhysicalColumn, to: &ColumnEntry) -> Widen {
    let target = to.physical_type_id();
    if from.physical_type == target {
        // A decimal keeps its scale in the catalog rather than in the bytes,
        // so the same physical width can still need a rescale
        if target == TypeId::Decimal {
            let old = from.fractional_digits.unwrap_or(0);
            let new = to.fractional_digits.unwrap_or(0);
            if new > old {
                return Widen::DecimalRescale { from: old, to: new };
            }
        }
        return Widen::None;
    }
    if from.physical_type == TypeId::Int64
        && target == TypeId::Int128
        && matches!(to.type_id, TypeId::Timestamp | TypeId::TimestampTz)
    {
        return Widen::MicrosToPicos;
    }
    if let (Some(f), Some(t)) = (signed_rank(from.physical_type), signed_rank(target))
        && t > f
    {
        return Widen::Signed(target);
    }
    if let (Some(f), Some(t)) = (unsigned_rank(from.physical_type), unsigned_rank(target))
        && t > f
    {
        return Widen::Unsigned(target);
    }
    Widen::None
}

/// The value a column absent from an epoch's layout reads as in a row of that
/// epoch.
///
/// The bytes were recorded once, when the column was added, encoded in the
/// column's physical type. A column with no recorded value reads NULL, which
/// is what adding a column without a DEFAULT means.
pub fn absent_value_of(column: &ColumnEntry) -> ScalarValue {
    absent_scalar(column)
}

fn absent_scalar(column: &ColumnEntry) -> ScalarValue {
    let Some(bytes) = &column.absent_value else {
        return ScalarValue::Null;
    };
    let physical = column.physical_type_id();
    match physical.fixed_size() {
        Some(size) if bytes.len() >= size => decode_fixed_scalar(physical, &bytes[..size]),
        Some(_) => ScalarValue::Null,
        None => {
            if column.is_encrypted() {
                ScalarValue::Binary(bytes.clone())
            } else {
                decode_varlen_scalar(column.type_id, bytes)
            }
        }
    }
}

/// Every plan one table's rows need for one projection.
///
/// Built once per scan. A table carries a handful of epochs at most, so the
/// plans are held in a dense list the epoch indexes directly and the per-row
/// cost is one bounds-checked lookup.
#[derive(Debug, Clone)]
pub struct EpochDecoder {
    /// Indexed by epoch. A None slot is an epoch this table never wrote
    plans: Vec<Option<EpochPlan>>,
    table_id: u32,
    table_name: String,
    /// The epoch writes stamp right now, which is the one a cursor starts on
    current_epoch: u16,
}

/// A decoder held against one epoch's plan.
///
/// A scan walks a page in slot order, and a table's rows are overwhelmingly
/// written under one layout, so resolving the plan for every row repeats a
/// lookup whose answer almost never changes. The cursor keeps the last answer
/// and re-resolves only when a row carries a different epoch, which turns the
/// per-row cost into one integer compare.
pub struct EpochCursor<'a> {
    decoder: &'a EpochDecoder,
    epoch: u16,
    plan: Option<&'a EpochPlan>,
}

impl<'a> EpochCursor<'a> {
    /// Points the cursor at `epoch`, doing nothing when it is already there.
    #[inline(always)]
    fn seek(&mut self, epoch: u16) {
        if epoch != self.epoch {
            self.epoch = epoch;
            self.plan = self.decoder.plan(epoch);
        }
    }

    /// Decodes one row, reporting an epoch the table never wrote.
    #[inline]
    pub fn decode(
        &mut self,
        epoch: u16,
        data: &[u8],
        at: Option<RowLocator>,
        builders: &mut [ColumnBuilder],
    ) -> Result<()> {
        self.seek(epoch);
        match self.plan {
            Some(plan) => {
                plan.decode_into(data, builders);
                Ok(())
            }
            None => Err(self.decoder.unknown_epoch(epoch, at)),
        }
    }

    /// Decodes one row when the caller drops a malformed row rather than
    /// reporting it.
    #[inline]
    pub fn try_decode(&mut self, epoch: u16, data: &[u8], builders: &mut [ColumnBuilder]) -> bool {
        self.seek(epoch);
        match self.plan {
            Some(plan) if plan.spans(data) => {
                plan.decode_into(data, builders);
                true
            }
            _ => false,
        }
    }
}

impl EpochDecoder {
    /// Builds the plans for `table`, filling the builders that `output_ids`
    /// describes in that order.
    ///
    /// `output_ids` names the current columns the caller wants, so a
    /// projection that skips a column produces a plan that walks its bytes
    /// and pushes nothing.
    pub fn new(table: &TableEntry, output_ids: &[ColumnId]) -> Self {
        Self::build(table, output_ids, false)
    }

    /// A decoder that also fills a dropped column the projection names, from
    /// the rows that still carry it. A change read that renders each record
    /// in the shape it was written under asks for this
    pub fn new_with_dropped(table: &TableEntry, output_ids: &[ColumnId]) -> Self {
        Self::build(table, output_ids, true)
    }

    fn build(table: &TableEntry, output_ids: &[ColumnId], include_dropped: bool) -> Self {
        let mut builder_of: HashMap<ColumnId, u16> = HashMap::with_capacity(output_ids.len());
        for (b, id) in output_ids.iter().enumerate() {
            builder_of.insert(*id, b as u16);
        }
        let by_id: HashMap<ColumnId, &ColumnEntry> =
            table.columns.iter().map(|c| (c.id, c)).collect();

        let highest = table
            .schema_epochs
            .iter()
            .map(|e| e.epoch)
            .max()
            .unwrap_or(table.schema_epoch);
        let mut plans: Vec<Option<EpochPlan>> = vec![None; highest as usize + 1];

        if !table.pre_stamp_columns.is_empty() {
            plans[0] = Some(build_plan(
                &table.pre_stamp_columns,
                &carried_by(&table.pre_stamp_columns),
                &by_id,
                &builder_of,
                table,
                include_dropped,
            ));
        }
        for recorded in &table.schema_epochs {
            let slot = recorded.epoch as usize;
            if slot < plans.len() {
                plans[slot] = Some(build_plan(
                    &recorded.columns,
                    &carried_by(&recorded.columns),
                    &by_id,
                    &builder_of,
                    table,
                    include_dropped,
                ));
            }
        }

        Self {
            plans,
            table_id: table.id.0,
            table_name: table.name.clone(),
            current_epoch: table.schema_epoch,
        }
    }

    /// A cursor primed on the epoch writes stamp now.
    ///
    /// Taken once outside a row loop, so the rows that carry that epoch, which
    /// is nearly all of them, never resolve a plan again.
    #[inline]
    pub fn cursor(&self) -> EpochCursor<'_> {
        EpochCursor {
            decoder: self,
            epoch: self.current_epoch,
            plan: self.plan(self.current_epoch),
        }
    }

    /// The plan for one epoch, or None when the table never wrote that
    /// layout.
    #[inline]
    pub fn plan(&self, epoch: u16) -> Option<&EpochPlan> {
        self.plans.get(epoch as usize).and_then(|p| p.as_ref())
    }

    /// Reports a tuple whose epoch names no layout this table ever wrote.
    ///
    /// A row like this cannot be read at all, because every byte after the
    /// bitmap would land at the wrong column. Naming the page and slot makes
    /// the report actionable rather than a statement that something is wrong
    /// somewhere.
    pub fn unknown_epoch(&self, epoch: u16, at: Option<RowLocator>) -> ZyronError {
        let site = match at {
            Some(RowLocator::Heap { page, slot }) => {
                format!("page {} slot {}", page.page_num, slot)
            }
            Some(other) => format!("row {other:?}"),
            None => "an unlocated row".to_string(),
        };
        ZyronError::CatalogCorrupted(format!(
            "table \"{}\" (id {}) has a tuple at {} stamped with schema epoch {}, which names no \
             layout the table ever wrote. The row cannot be decoded, because every column after \
             the null bitmap would be read at the wrong offset",
            self.table_name, self.table_id, site, epoch
        ))
    }

    /// Decodes one row, resolving its layout from the epoch its slot carries.
    #[inline]
    pub fn decode(
        &self,
        epoch: u16,
        data: &[u8],
        at: Option<RowLocator>,
        builders: &mut [ColumnBuilder],
    ) -> Result<()> {
        let Some(plan) = self.plan(epoch) else {
            return Err(self.unknown_epoch(epoch, at));
        };
        plan.decode_into(data, builders);
        Ok(())
    }

    /// Decodes one row when the caller has already decided that a malformed
    /// row is dropped rather than reported.
    ///
    /// Returns false without touching the builders when the epoch names no
    /// layout or the bytes do not span the row, so the caller's row count and
    /// its builders stay in step.
    #[inline]
    pub fn try_decode(&self, epoch: u16, data: &[u8], builders: &mut [ColumnBuilder]) -> bool {
        let Some(plan) = self.plan(epoch) else {
            return false;
        };
        if !plan.spans(data) {
            return false;
        }
        plan.decode_into(data, builders);
        true
    }
}

/// What a projected change record of one epoch decodes through
#[derive(Debug, Clone)]
enum ProjectedPlan {
    /// The plan over the columns the feed recorded at the epoch
    Plan(EpochPlan),
    /// The projection asks for something no record of this epoch holds, and
    /// the first such record fails with this rather than reading NULL for a
    /// value the table had
    Refused(String),
}

/// Every plan one table's projected change records need for one projection.
///
/// A change data feed configured with a column subset records each row
/// narrowed to the subset in force at its epoch and flags the record. The
/// epoch names both the table's layout and the subset, so the plan for an
/// epoch walks that layout narrowed to that subset. A column the subset left
/// out is not absent the way a column added later is. The table had a value
/// the feed never recorded, so a projection asking for it is refused on the
/// first record of that epoch rather than answered NULL
#[derive(Debug, Clone)]
pub struct ProjectedEpochDecoder {
    /// Indexed by epoch. A None slot is an epoch this table never wrote
    plans: Vec<Option<ProjectedPlan>>,
    table_id: u32,
    table_name: String,
}

impl ProjectedEpochDecoder {
    /// Builds the plans for `table`'s projected records, filling the
    /// builders that `output_ids` describes in that order. `include_dropped`
    /// also fills a dropped column the projection names from the records
    /// that carry it
    pub fn new(table: &TableEntry, output_ids: &[ColumnId], include_dropped: bool) -> Self {
        let mut builder_of: HashMap<ColumnId, u16> = HashMap::with_capacity(output_ids.len());
        for (b, id) in output_ids.iter().enumerate() {
            builder_of.insert(*id, b as u16);
        }
        let by_id: HashMap<ColumnId, &ColumnEntry> =
            table.columns.iter().map(|c| (c.id, c)).collect();
        let highest = table
            .schema_epochs
            .iter()
            .map(|e| e.epoch)
            .max()
            .unwrap_or(table.schema_epoch);
        let mut plans: Vec<Option<ProjectedPlan>> = vec![None; highest as usize + 1];

        let epochs = std::iter::once((0u16, table.pre_stamp_columns.as_slice()))
            .filter(|(_, layout)| !layout.is_empty())
            .chain(
                table
                    .schema_epochs
                    .iter()
                    .map(|recorded| (recorded.epoch, recorded.columns.as_slice())),
            );
        for (epoch, layout) in epochs {
            let slot = epoch as usize;
            if slot >= plans.len() {
                continue;
            }
            plans[slot] = Some(Self::plan_for(
                table,
                epoch,
                layout,
                &by_id,
                &builder_of,
                include_dropped,
            ));
        }
        Self {
            plans,
            table_id: table.id.0,
            table_name: table.name.clone(),
        }
    }

    /// The plan one epoch's projected records decode through, or the refusal
    /// every such record meets
    fn plan_for(
        table: &TableEntry,
        epoch: u16,
        layout: &[PhysicalColumn],
        by_id: &HashMap<ColumnId, &ColumnEntry>,
        builder_of: &HashMap<ColumnId, u16>,
        include_dropped: bool,
    ) -> ProjectedPlan {
        let recorded = table.cdf.columns_at(epoch);
        if recorded.is_empty() {
            return ProjectedPlan::Refused(format!(
                "table \"{}\" (id {}) has a change record stamped with schema epoch {epoch} that \
                 carries a column subset, and no subset was in force at that epoch, so the \
                 record cannot be decoded",
                table.name, table.id.0
            ));
        }
        // A column the layout carries and the subset left out was never
        // recorded, and a projection asking for it is refused by name
        let carried = carried_by(layout);
        let mut wanted: Vec<(&ColumnId, &u16)> = builder_of.iter().collect();
        wanted.sort_by_key(|(_, builder)| **builder);
        for (id, _) in wanted {
            if carried.contains(id) && !recorded.contains(id) {
                let name = by_id
                    .get(id)
                    .map(|c| c.name.as_str())
                    .unwrap_or("<unknown>");
                return ProjectedPlan::Refused(format!(
                    "column '{name}' is outside the change data feed's cdf_columns list that \
                     was in force on table '{}' at schema epoch {epoch}, so the feed holds no \
                     value for it in the changes written under that list. Read a later \
                     version range or leave the column out of the query",
                    table.name
                ));
            }
        }
        let projected: Vec<PhysicalColumn> = layout
            .iter()
            .filter(|column| recorded.contains(&column.column_id))
            .enumerate()
            .map(|(ordinal, column)| PhysicalColumn {
                column_id: column.column_id,
                physical_type: column.physical_type,
                fractional_digits: column.fractional_digits,
                ordinal: ordinal as u16,
            })
            .collect();
        ProjectedPlan::Plan(build_plan(
            &projected,
            &carried,
            by_id,
            builder_of,
            table,
            include_dropped,
        ))
    }

    /// The plan one epoch's projected records decode through, reporting an
    /// epoch the table never wrote or a column the feed never recorded at
    /// that epoch
    pub(crate) fn plan(&self, epoch: u16) -> Result<&EpochPlan> {
        match self.plans.get(epoch as usize).and_then(|p| p.as_ref()) {
            Some(ProjectedPlan::Plan(plan)) => Ok(plan),
            Some(ProjectedPlan::Refused(message)) => {
                Err(ZyronError::CdcDecoderError(message.clone()))
            }
            None => Err(ZyronError::CatalogCorrupted(format!(
                "table \"{}\" (id {}) has a change record stamped with schema epoch {epoch},                  which names no layout the table ever wrote. The record cannot be decoded,                  because every column after the null bitmap would be read at the wrong offset",
                self.table_name, self.table_id
            ))),
        }
    }

    /// Decodes one projected record, reporting an epoch the table never
    /// wrote or a column the feed never recorded at that epoch
    #[inline]
    pub fn decode(&self, epoch: u16, data: &[u8], builders: &mut [ColumnBuilder]) -> Result<()> {
        match self.plans.get(epoch as usize).and_then(|p| p.as_ref()) {
            Some(ProjectedPlan::Plan(plan)) => {
                if !plan.spans(data) {
                    return Err(ZyronError::CdcDecoderError(format!(
                        "table \"{}\" (id {}) has a change record of {} bytes stamped with \
                         schema epoch {epoch} that is shorter than the column subset recorded \
                         at that epoch",
                        self.table_name,
                        self.table_id,
                        data.len()
                    )));
                }
                plan.decode_into(data, builders);
                Ok(())
            }
            Some(ProjectedPlan::Refused(message)) => {
                Err(ZyronError::CdcDecoderError(message.clone()))
            }
            None => Err(ZyronError::CatalogCorrupted(format!(
                "table \"{}\" (id {}) has a change record stamped with schema epoch {epoch}, \
                 which names no layout the table ever wrote. The record cannot be decoded, \
                 because every column after the null bitmap would be read at the wrong offset",
                self.table_name, self.table_id
            ))),
        }
    }
}

/// The column ids a layout carries
fn carried_by(layout: &[PhysicalColumn]) -> HashSet<ColumnId> {
    layout.iter().map(|p| p.column_id).collect()
}

/// Builds one epoch's plan from its recorded layout.
///
/// `carried` names the columns the epoch's full layout holds, which is the
/// layout itself for a whole row and the full layout for a projected one, so
/// a column the projection left out is not mistaken for one added later
fn build_plan(
    layout: &[PhysicalColumn],
    carried: &HashSet<ColumnId>,
    by_id: &HashMap<ColumnId, &ColumnEntry>,
    builder_of: &HashMap<ColumnId, u16>,
    table: &TableEntry,
    include_dropped: bool,
) -> EpochPlan {
    let mut steps = Vec::with_capacity(layout.len());
    for physical in layout {
        match (
            by_id.get(&physical.column_id),
            builder_of.get(&physical.column_id),
        ) {
            // The column still exists and this projection wants it, or it
            // was dropped and the projection asked for the dropped ones too
            (Some(column), Some(builder)) if !column.dropped || include_dropped => {
                steps.push(Step::Take {
                    builder: *builder,
                    physical: physical.physical_type,
                    logical: column.type_id,
                    widen: widening_for(physical, column),
                    encrypted: column.is_encrypted(),
                })
            }
            // Either the column was dropped, or this projection skips it.
            // Both walk the bytes and push nothing
            _ => steps.push(Step::Discard {
                physical: physical.physical_type,
            }),
        }
    }

    // A column the layout does not carry is a column added after this epoch,
    // so every row of the epoch reads the value recorded when it was added.
    // A dropped column the projection asked for and this layout never held
    // reads the same way when the projection wants the dropped ones, and a
    // dropped column the projection names without wanting them reads NULL
    // from every row, whether the layout carried its bytes, which the plan
    // walks past, or not
    let mut absent = Vec::new();
    for column in &table.columns {
        let Some(builder) = builder_of.get(&column.id) else {
            continue;
        };
        if column.dropped && !include_dropped {
            absent.push((*builder, ScalarValue::Null));
            continue;
        }
        if carried.contains(&column.id) {
            continue;
        }
        absent.push((*builder, absent_scalar(column)));
    }
    // Pushed in builder order so a projection that lists several added
    // columns fills them in the order its own schema names
    absent.sort_by_key(|(b, _)| *b);

    let bitmap_len = layout.len().div_ceil(8);
    let mut min_len = bitmap_len;
    let mut all_fixed = true;
    let mut varlen_start = (steps.len(), bitmap_len);
    for (i, step) in steps.iter().enumerate() {
        let physical = match step {
            Step::Take { physical, .. } => *physical,
            Step::Discard { physical } => *physical,
        };
        match physical.fixed_size() {
            Some(size) => min_len += size,
            None => {
                if all_fixed {
                    varlen_start = (i, min_len);
                    all_fixed = false;
                }
                min_len += 4;
            }
        }
    }

    EpochPlan {
        bitmap_len,
        steps,
        absent,
        min_len,
        all_fixed,
        varlen_start,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use zyron_catalog::{EpochColumns, TableId};

    fn column(id: u16, name: &str, type_id: TypeId, ordinal: u16) -> ColumnEntry {
        ColumnEntry {
            id: ColumnId(id),
            table_id: TableId(1),
            name: name.to_string(),
            type_id,
            ordinal,
            nullable: true,
            default_expr: None,
            max_length: None,
            fractional_digits: None,
            tz_offset_secs: None,
            element_type: None,
            attrs: Default::default(),
            absent_value: None,
            dropped: false,
        }
    }

    fn table(columns: Vec<ColumnEntry>) -> TableEntry {
        let mut entry = TableEntry {
            id: TableId(1),
            schema_id: zyron_catalog::SchemaId(1),
            name: "t".to_string(),
            heap_file_id: 1,
            fsm_file_id: 2,
            columns,
            constraints: Vec::new(),
            created_at: 0,
            versioning_enabled: false,
            scd_type: None,
            system_versioned: false,
            history_table_id: None,
            cdf_enabled: false,
            cdf_retention_days: 0,
            lifecycle: Default::default(),
            columnar: Default::default(),
            dropped_at: None,
            expectations: Vec::new(),
            time_travel_retention_secs: 0,
            lake: Default::default(),
            cluster: Default::default(),
            foreign: Default::default(),
            schema_epoch: 0,
            schema_epochs: Vec::new(),
            pre_stamp_columns: Vec::new(),
            cdf: Default::default(),
        };
        entry.seal_initial_epoch();
        entry
    }

    /// Encodes one row under a layout, the way the heap encoder does
    fn encode(values: &[Option<ScalarValue>], types: &[TypeId]) -> Vec<u8> {
        let mut bitmap = vec![0u8; values.len().div_ceil(8)];
        let mut body = Vec::new();
        for (i, value) in values.iter().enumerate() {
            let physical = types[i];
            match value {
                None => {
                    bitmap[i / 8] |= 1 << (i % 8);
                    match physical.fixed_size() {
                        Some(size) => body.extend(std::iter::repeat_n(0u8, size)),
                        None => body.extend_from_slice(&0u32.to_le_bytes()),
                    }
                }
                Some(ScalarValue::Int32(v)) => body.extend_from_slice(&v.to_le_bytes()),
                Some(ScalarValue::Int64(v)) => body.extend_from_slice(&v.to_le_bytes()),
                Some(ScalarValue::Utf8(s)) => {
                    body.extend_from_slice(&(s.len() as u32).to_le_bytes());
                    body.extend_from_slice(s.as_bytes());
                }
                Some(other) => panic!("test encoder does not carry {other:?}"),
            }
        }
        bitmap.extend_from_slice(&body);
        bitmap
    }

    fn builders_for(n: usize, types: &[TypeId]) -> Vec<ColumnBuilder> {
        (0..n).map(|i| ColumnBuilder::new(types[i], 4)).collect()
    }

    #[test]
    fn test_current_epoch_decodes_every_column() {
        let t = table(vec![
            column(0, "a", TypeId::Int32, 0),
            column(1, "b", TypeId::Text, 1),
        ]);
        let ids = vec![ColumnId(0), ColumnId(1)];
        let decoder = EpochDecoder::new(&t, &ids);
        let row = encode(
            &[
                Some(ScalarValue::Int32(7)),
                Some(ScalarValue::Utf8("hi".into())),
            ],
            &[TypeId::Int32, TypeId::Text],
        );
        let mut builders = builders_for(2, &[TypeId::Int32, TypeId::Text]);
        decoder
            .decode(1, &row, None, &mut builders)
            .expect("decodes");
        let a = builders.remove(0).finish();
        let b = builders.remove(0).finish();
        assert_eq!(a.get_scalar(0), ScalarValue::Int32(7));
        assert_eq!(b.get_scalar(0), ScalarValue::Utf8("hi".into()));
    }

    #[test]
    fn test_a_row_older_than_a_column_reads_its_absent_value() {
        let mut t = table(vec![column(0, "a", TypeId::Int32, 0)]);
        let mut added = column(1, "b", TypeId::Int32, 1);
        added.absent_value = Some(7i32.to_le_bytes().to_vec());
        t.columns.push(added);
        t.push_schema_epoch(t.current_physical_columns());

        let ids = vec![ColumnId(0), ColumnId(1)];
        let decoder = EpochDecoder::new(&t, &ids);
        // A row written under epoch 1, which had only column a
        let row = encode(&[Some(ScalarValue::Int32(3))], &[TypeId::Int32]);
        let mut builders = builders_for(2, &[TypeId::Int32, TypeId::Int32]);
        decoder
            .decode(1, &row, None, &mut builders)
            .expect("decodes");
        let a = builders.remove(0).finish();
        let b = builders.remove(0).finish();
        assert_eq!(a.get_scalar(0), ScalarValue::Int32(3));
        assert_eq!(b.get_scalar(0), ScalarValue::Int32(7));
    }

    #[test]
    fn test_a_dropped_column_is_walked_and_not_pushed() {
        let mut t = table(vec![
            column(0, "a", TypeId::Int32, 0),
            column(1, "gone", TypeId::Text, 1),
            column(2, "c", TypeId::Int32, 2),
        ]);
        t.columns[1].dropped = true;

        let ids = vec![ColumnId(0), ColumnId(2)];
        let decoder = EpochDecoder::new(&t, &ids);
        let row = encode(
            &[
                Some(ScalarValue::Int32(1)),
                Some(ScalarValue::Utf8("skipped".into())),
                Some(ScalarValue::Int32(9)),
            ],
            &[TypeId::Int32, TypeId::Text, TypeId::Int32],
        );
        let mut builders = builders_for(2, &[TypeId::Int32, TypeId::Int32]);
        decoder
            .decode(1, &row, None, &mut builders)
            .expect("decodes");
        let a = builders.remove(0).finish();
        let c = builders.remove(0).finish();
        assert_eq!(a.get_scalar(0), ScalarValue::Int32(1));
        assert_eq!(c.get_scalar(0), ScalarValue::Int32(9));
    }

    #[test]
    fn test_a_dropped_column_the_projection_names_reads_null_from_every_row() {
        let mut t = table(vec![
            column(0, "a", TypeId::Int32, 0),
            column(1, "gone", TypeId::Text, 1),
            column(2, "c", TypeId::Int32, 2),
        ]);
        t.columns[1].dropped = true;

        // The projection is the table's whole column list, the shape a write
        // path reads a row image in, so the dropped column has a builder
        let ids = vec![ColumnId(0), ColumnId(1), ColumnId(2)];
        let decoder = EpochDecoder::new(&t, &ids);
        let row = encode(
            &[
                Some(ScalarValue::Int32(1)),
                Some(ScalarValue::Utf8("written before the drop".into())),
                Some(ScalarValue::Int32(9)),
            ],
            &[TypeId::Int32, TypeId::Text, TypeId::Int32],
        );
        let mut builders = builders_for(3, &[TypeId::Int32, TypeId::Text, TypeId::Int32]);
        decoder
            .decode(1, &row, None, &mut builders)
            .expect("decodes");
        let a = builders.remove(0).finish();
        let gone = builders.remove(0).finish();
        let c = builders.remove(0).finish();
        assert_eq!(a.get_scalar(0), ScalarValue::Int32(1));
        assert_eq!(
            gone.len(),
            1,
            "the dropped column has a row like every other"
        );
        assert_eq!(gone.get_scalar(0), ScalarValue::Null);
        assert_eq!(c.get_scalar(0), ScalarValue::Int32(9));

        // The same row read as the record wrote it keeps the value
        let with_dropped = EpochDecoder::new_with_dropped(&t, &ids);
        let mut builders = builders_for(3, &[TypeId::Int32, TypeId::Text, TypeId::Int32]);
        with_dropped
            .decode(1, &row, None, &mut builders)
            .expect("decodes");
        let gone = builders.remove(1).finish();
        assert_eq!(
            gone.get_scalar(0),
            ScalarValue::Utf8("written before the drop".into())
        );
    }

    #[test]
    fn test_a_narrower_integer_widens_on_the_way_in() {
        let mut t = table(vec![column(0, "a", TypeId::Int32, 0)]);
        // The column is BIGINT now, and epoch 1 wrote it as INT
        t.columns[0].type_id = TypeId::Int64;
        t.push_schema_epoch(t.current_physical_columns());

        let ids = vec![ColumnId(0)];
        let decoder = EpochDecoder::new(&t, &ids);
        let row = encode(&[Some(ScalarValue::Int32(-5))], &[TypeId::Int32]);
        let mut builders = builders_for(1, &[TypeId::Int64]);
        decoder
            .decode(1, &row, None, &mut builders)
            .expect("decodes");
        let a = builders.remove(0).finish();
        assert_eq!(a.get_scalar(0), ScalarValue::Int64(-5));
    }

    #[test]
    fn test_an_unknown_epoch_is_reported_with_the_row_it_names() {
        let t = table(vec![column(0, "a", TypeId::Int32, 0)]);
        let decoder = EpochDecoder::new(&t, &[ColumnId(0)]);
        let at = RowLocator::Heap {
            page: zyron_common::page::PageId::new(1, 42),
            slot: 3,
        };
        let mut builders = builders_for(1, &[TypeId::Int32]);
        let err = decoder
            .decode(9, &[0u8; 8], Some(at), &mut builders)
            .expect_err("unknown epoch");
        let text = err.to_string();
        assert!(text.contains("\"t\""), "{text}");
        assert!(text.contains("page 42"), "{text}");
        assert!(text.contains("slot 3"), "{text}");
        assert!(text.contains("epoch 9"), "{text}");
    }

    /// A table whose feed records `recorded` from epoch 1 on, with a key
    fn narrowed_table(recorded: &[u16]) -> TableEntry {
        let mut t = table(vec![
            column(0, "id", TypeId::Int32, 0),
            column(1, "amount", TypeId::Int32, 1),
            column(2, "note", TypeId::Text, 2),
        ]);
        t.cdf.column_sets.push(zyron_catalog::CdfColumnSet {
            from_epoch: 1,
            columns: recorded.iter().map(|id| ColumnId(*id)).collect(),
        });
        t
    }

    #[test]
    fn test_a_projected_record_decodes_through_the_narrowed_layout() {
        let t = narrowed_table(&[0, 2]);
        let decoder = ProjectedEpochDecoder::new(&t, &[ColumnId(0), ColumnId(2)], false);
        // The record holds the key and the note, in the layout's order,
        // with a bitmap sized to the two of them
        let row = encode(
            &[
                Some(ScalarValue::Int32(7)),
                Some(ScalarValue::Utf8("hi".into())),
            ],
            &[TypeId::Int32, TypeId::Text],
        );
        let mut builders = builders_for(2, &[TypeId::Int32, TypeId::Text]);
        decoder.decode(1, &row, &mut builders).expect("decodes");
        let id = builders.remove(0).finish();
        let note = builders.remove(0).finish();
        assert_eq!(id.get_scalar(0), ScalarValue::Int32(7));
        assert_eq!(note.get_scalar(0), ScalarValue::Utf8("hi".into()));
    }

    #[test]
    fn test_a_column_the_subset_left_out_is_refused_by_name() {
        let t = narrowed_table(&[0, 2]);
        let decoder = ProjectedEpochDecoder::new(&t, &[ColumnId(0), ColumnId(1)], false);
        let row = encode(
            &[
                Some(ScalarValue::Int32(7)),
                Some(ScalarValue::Utf8("hi".into())),
            ],
            &[TypeId::Int32, TypeId::Text],
        );
        let mut builders = builders_for(2, &[TypeId::Int32, TypeId::Int32]);
        let text = decoder
            .decode(1, &row, &mut builders)
            .expect_err("refused")
            .to_string();
        assert!(text.contains("'amount'"), "{text}");
        assert!(text.contains("epoch 1"), "{text}");
    }

    #[test]
    fn test_a_column_added_after_a_projected_record_reads_its_absent_value() {
        let mut t = narrowed_table(&[0, 2]);
        let mut added = column(3, "late", TypeId::Int32, 3);
        added.absent_value = Some(9i32.to_le_bytes().to_vec());
        t.columns.push(added);
        t.push_schema_epoch(t.current_physical_columns());
        let decoder = ProjectedEpochDecoder::new(&t, &[ColumnId(0), ColumnId(3)], false);
        let row = encode(
            &[
                Some(ScalarValue::Int32(7)),
                Some(ScalarValue::Utf8("hi".into())),
            ],
            &[TypeId::Int32, TypeId::Text],
        );
        let mut builders = builders_for(2, &[TypeId::Int32, TypeId::Int32]);
        decoder.decode(1, &row, &mut builders).expect("decodes");
        let id = builders.remove(0).finish();
        let late = builders.remove(0).finish();
        assert_eq!(id.get_scalar(0), ScalarValue::Int32(7));
        assert_eq!(late.get_scalar(0), ScalarValue::Int32(9));
    }

    #[test]
    fn test_a_projected_record_at_an_epoch_with_no_subset_is_refused() {
        let t = table(vec![column(0, "id", TypeId::Int32, 0)]);
        let decoder = ProjectedEpochDecoder::new(&t, &[ColumnId(0)], false);
        let row = encode(&[Some(ScalarValue::Int32(7))], &[TypeId::Int32]);
        let mut builders = builders_for(1, &[TypeId::Int32]);
        let text = decoder
            .decode(1, &row, &mut builders)
            .expect_err("refused")
            .to_string();
        assert!(text.contains("no subset was in force"), "{text}");
    }

    #[test]
    fn test_the_null_bitmap_is_sized_from_the_epoch_not_the_current_schema() {
        // Eight columns at epoch 1, a ninth added at epoch 2
        let mut cols: Vec<ColumnEntry> = (0..8)
            .map(|i| column(i, &format!("c{i}"), TypeId::Int32, i))
            .collect();
        let mut t = table(std::mem::take(&mut cols));
        let mut ninth = column(8, "c8", TypeId::Int32, 8);
        ninth.absent_value = Some(42i32.to_le_bytes().to_vec());
        t.columns.push(ninth);
        t.push_schema_epoch(t.current_physical_columns());

        let ids: Vec<ColumnId> = (0..9).map(ColumnId).collect();
        let decoder = EpochDecoder::new(&t, &ids);

        let old_plan = decoder.plan(1).expect("epoch 1");
        let new_plan = decoder.plan(2).expect("epoch 2");
        assert_eq!(old_plan.bitmap_len, 1, "eight columns take one bitmap byte");
        assert_eq!(new_plan.bitmap_len, 2, "nine columns take two");

        let values: Vec<Option<ScalarValue>> =
            (0..8).map(|i| Some(ScalarValue::Int32(i as i32))).collect();
        let types = vec![TypeId::Int32; 8];
        let row = encode(&values, &types);
        let mut builders = builders_for(9, &vec![TypeId::Int32; 9]);
        decoder
            .decode(1, &row, None, &mut builders)
            .expect("decodes");
        let finished: Vec<_> = builders.into_iter().map(|b| b.finish()).collect();
        for i in 0..8 {
            assert_eq!(finished[i].get_scalar(0), ScalarValue::Int32(i as i32));
        }
        assert_eq!(finished[8].get_scalar(0), ScalarValue::Int32(42));
    }
}
