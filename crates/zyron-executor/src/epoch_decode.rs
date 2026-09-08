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
//! position, so the cursor walks it and pushes nothing. A column whose type
//! widened is decoded at the width the bytes carry and widened on the way into
//! the builder.
//!
//! The work of deciding all three is done once per (table, epoch, projection)
//! and kept as a plan, so the per-row path is a table lookup and a walk.

use std::collections::HashMap;

use zyron_catalog::{ColumnEntry, ColumnId, PhysicalColumn, TableEntry};
use zyron_common::{Result, RowLocator, TypeId, ZyronError};

use crate::batch::{ColumnBuilder, decode_fixed_scalar, decode_varlen_scalar};
use crate::column::ScalarValue;

/// How a value read at the epoch's width becomes a value of the column's
/// current type.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Widen {
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
enum Step {
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
fn apply_widen(widen: Widen, scalar: ScalarValue) -> ScalarValue {
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
                &by_id,
                &builder_of,
                table,
            ));
        }
        for recorded in &table.schema_epochs {
            let slot = recorded.epoch as usize;
            if slot < plans.len() {
                plans[slot] = Some(build_plan(&recorded.columns, &by_id, &builder_of, table));
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

/// Builds one epoch's plan from its recorded layout.
fn build_plan(
    layout: &[PhysicalColumn],
    by_id: &HashMap<ColumnId, &ColumnEntry>,
    builder_of: &HashMap<ColumnId, u16>,
    table: &TableEntry,
) -> EpochPlan {
    let mut steps = Vec::with_capacity(layout.len());
    for physical in layout {
        match (
            by_id.get(&physical.column_id),
            builder_of.get(&physical.column_id),
        ) {
            // The column still exists and this projection wants it
            (Some(column), Some(builder)) if !column.dropped => steps.push(Step::Take {
                builder: *builder,
                physical: physical.physical_type,
                logical: column.type_id,
                widen: widening_for(physical, column),
                encrypted: column.is_encrypted(),
            }),
            // Either the column was dropped, or this projection skips it.
            // Both walk the bytes and push nothing
            _ => steps.push(Step::Discard {
                physical: physical.physical_type,
            }),
        }
    }

    // A column the layout does not carry is a column added after this epoch,
    // so every row of the epoch reads the value recorded when it was added
    let carried: std::collections::HashSet<ColumnId> = layout.iter().map(|p| p.column_id).collect();
    let mut absent = Vec::new();
    for column in &table.columns {
        if column.dropped || carried.contains(&column.id) {
            continue;
        }
        if let Some(builder) = builder_of.get(&column.id) {
            absent.push((*builder, absent_scalar(column)));
        }
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
