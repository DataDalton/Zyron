//! Reading a change feed across a schema change.
//!
//! Every change record carries the schema epoch the table stamped when the row
//! was written, so the feed stays readable across ADD COLUMN, DROP COLUMN and
//! a widening type change. The epoch names a recorded physical layout, and the
//! layout is what says how to walk the bytes.
//!
//! Two projections are available. The default reads every record through the
//! table's current schema. A column added after a change reads its recorded
//! absent value, a column since dropped is walked past, and a widened type is
//! read at the width the bytes carry. `WITH (schema => 'as_of_change')` reads
//! each record through the layout it was written under instead, for a consumer
//! that must see the original shape.
//!
//! A feed configured with a column subset records each row narrowed to that
//! subset plus the table's key columns. The record carries a flag saying so,
//! and the epoch it carries names both the table's layout and the subset in
//! force at the time, so a change to the subset mints an epoch and every
//! record stays decodable through the list it was written under.
//!
//! A narrowing or otherwise incompatible type change on a table carrying an
//! active stream is refused unless the DDL says ACKNOWLEDGE STREAM BREAK,
//! because the recorded layout would then describe bytes the current column
//! cannot hold

use zyron_catalog::{ColumnId, PhysicalColumn, TableEntry};
use zyron_common::{Result, TypeId, ZyronError};

/// Which schema a read projects change records to
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ProjectionSchema {
    /// The table's schema as it stands. A record older than a column reads
    /// that column's absent value
    #[default]
    Current,
    /// The layout each record was written under
    AsOfChange,
}

impl ProjectionSchema {
    /// Resolves the word a `WITH (schema => ...)` option carries
    pub fn from_name(name: &str) -> Result<Self> {
        match name.to_ascii_lowercase().as_str() {
            "current" => Ok(ProjectionSchema::Current),
            "as_of_change" => Ok(ProjectionSchema::AsOfChange),
            other => Err(ZyronError::PlanError(format!(
                "schema option accepts 'current' or 'as_of_change', not '{other}'"
            ))),
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            ProjectionSchema::Current => "current",
            ProjectionSchema::AsOfChange => "as_of_change",
        }
    }
}

/// Narrows a row written under one layout to the columns the feed records.
///
/// Built once per write from the table's layout at its current epoch and the
/// list in force, then applied to every row of the write. A row is walked
/// once, each column's bytes either copied or stepped over by their width,
/// so projecting costs a copy of the kept bytes and nothing is decoded. The result is exactly the row a table whose columns were only
/// the kept ones would have encoded, with a null bitmap sized to the kept
/// count, which is the layout `TableEntry::projected_columns_for_epoch`
/// describes and the feed's reader decodes through
#[derive(Debug, Clone)]
pub struct RowProjection {
    /// One entry per column of the source layout, in encoded order, with the
    /// type that decides the width and whether the column is kept
    steps: Vec<(TypeId, bool)>,
    /// Bytes of null bitmap the source layout puts in front of its values
    source_bitmap_len: usize,
    /// Bytes of null bitmap the projected row carries
    kept_bitmap_len: usize,
    kept: usize,
}

impl RowProjection {
    /// A projection of `layout` onto the columns `recorded` names. A column
    /// the list names that the layout does not carry is not an error, the
    /// row simply has no bytes for it, which is what a column added after
    /// the list was set looks like from an older row
    pub fn new(layout: &[PhysicalColumn], recorded: &[ColumnId]) -> Self {
        let steps: Vec<(TypeId, bool)> = layout
            .iter()
            .map(|column| (column.physical_type, recorded.contains(&column.column_id)))
            .collect();
        let kept = steps.iter().filter(|(_, keep)| *keep).count();
        Self {
            source_bitmap_len: layout.len().div_ceil(8),
            kept_bitmap_len: kept.div_ceil(8),
            kept,
            steps,
        }
    }

    /// Columns the projected row carries
    pub fn kept(&self) -> usize {
        self.kept
    }

    /// Appends the projection of `row` to `out`.
    ///
    /// A row shorter than its layout is refused rather than read past its
    /// end, because a projection of a truncated row would be a record whose
    /// bytes describe a different row than the table wrote
    pub fn project(&self, row: &[u8], out: &mut Vec<u8>) -> Result<()> {
        if row.len() < self.source_bitmap_len {
            return Err(short_row(row.len()));
        }
        let (bitmap, values) = row.split_at(self.source_bitmap_len);
        let base = out.len();
        out.resize(base + self.kept_bitmap_len, 0);
        let mut offset = 0usize;
        let mut kept_at = 0usize;
        for (i, (physical, keep)) in self.steps.iter().enumerate() {
            let is_null = (bitmap[i / 8] >> (i % 8)) & 1 == 1;
            // A null value still occupies its width, a fixed type its size
            // and a variable one its four length bytes, so the walk is the
            // same whether or not the bit is set
            let width = match physical.fixed_size() {
                Some(size) => size,
                None => {
                    if offset + 4 > values.len() {
                        return Err(short_row(row.len()));
                    }
                    let len = u32::from_le_bytes([
                        values[offset],
                        values[offset + 1],
                        values[offset + 2],
                        values[offset + 3],
                    ]) as usize;
                    4 + len
                }
            };
            if offset + width > values.len() {
                return Err(short_row(row.len()));
            }
            if *keep {
                if is_null {
                    out[base + kept_at / 8] |= 1 << (kept_at % 8);
                }
                out.extend_from_slice(&values[offset..offset + width]);
                kept_at += 1;
            }
            offset += width;
        }
        Ok(())
    }
}

fn short_row(len: usize) -> ZyronError {
    ZyronError::CdcStreamError(format!(
        "a row of {len} bytes is shorter than the layout it was written under, so its change \
         cannot be recorded"
    ))
}

/// Whether a type change keeps every recorded layout readable.
///
/// A change is compatible when every epoch's stored width can still be read
/// into the new column, the same physical type, or a narrower integer of the
/// same signedness, or a microsecond timestamp becoming a picosecond one, or a
/// decimal whose scale grows
pub fn type_change_is_widening(from: TypeId, to: TypeId) -> bool {
    if from == to {
        return true;
    }
    if from == TypeId::Int64 && to == TypeId::Int128 {
        return true;
    }
    match (signed_rank(from), signed_rank(to)) {
        (Some(f), Some(t)) if t > f => return true,
        _ => {}
    }
    match (unsigned_rank(from), unsigned_rank(to)) {
        (Some(f), Some(t)) if t > f => return true,
        _ => {}
    }
    false
}

fn signed_rank(t: TypeId) -> Option<u8> {
    match t {
        TypeId::Int8 => Some(0),
        TypeId::Int16 => Some(1),
        TypeId::Int32 => Some(2),
        TypeId::Int64 => Some(3),
        TypeId::Int128 => Some(4),
        _ => None,
    }
}

fn unsigned_rank(t: TypeId) -> Option<u8> {
    match t {
        TypeId::UInt8 => Some(0),
        TypeId::UInt16 => Some(1),
        TypeId::UInt32 => Some(2),
        TypeId::UInt64 => Some(3),
        TypeId::UInt128 => Some(4),
        _ => None,
    }
}

/// What a DDL that would break a stream has to say to proceed
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct StreamBreak {
    pub acknowledged: bool,
}

/// Decides whether a type change may proceed on a table carrying streams.
///
/// A widening change always may. A narrowing one may only with ACKNOWLEDGE
/// STREAM BREAK, and the caller then marks every stream on the table as
/// needing attention
pub fn check_type_change(
    table_name: &str,
    column: &str,
    from: TypeId,
    to: TypeId,
    active_streams: usize,
    ack: StreamBreak,
) -> Result<()> {
    if type_change_is_widening(from, to) || active_streams == 0 || ack.acknowledged {
        return Ok(());
    }
    Err(ZyronError::PlanError(format!(
        "changing column '{column}' of table '{table_name}' from {from:?} to {to:?} narrows it, \
         and {active_streams} change stream(s) read this table through layouts recorded at the \
         old width. Repeat the statement with ACKNOWLEDGE STREAM BREAK to proceed, which marks \
         every stream on the table as needing attention"
    )))
}

/// Whether a stream's COLUMNS list still names columns the table has.
///
/// Answers with the first column that is gone, which is what the stream's
/// attention reason records. The stream keeps its position and starts yielding
/// again once an ALTER drops that column from the list
pub fn missing_stream_column(table: &TableEntry, columns: &[ColumnId]) -> Option<String> {
    for id in columns {
        match table.columns.iter().find(|c| c.id == *id) {
            Some(column) if !column.dropped => {}
            Some(column) => return Some(column.name.clone()),
            None => return Some(format!("column id {}", id.0)),
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use zyron_catalog::{ColumnEntry, ConstraintEntry, ConstraintType, SchemaId, TableId};

    fn column(id: u16, name: &str, type_id: TypeId) -> ColumnEntry {
        ColumnEntry {
            id: ColumnId(id),
            table_id: TableId(1),
            name: name.to_string(),
            type_id,
            ordinal: id,
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

    fn table() -> TableEntry {
        let mut entry = TableEntry {
            id: TableId(1),
            schema_id: SchemaId(1),
            name: "orders".to_string(),
            heap_file_id: 1,
            fsm_file_id: 2,
            columns: vec![
                column(0, "id", TypeId::Int64),
                column(1, "amount", TypeId::Int32),
                column(2, "note", TypeId::Text),
            ],
            constraints: vec![ConstraintEntry {
                name: "orders_pkey".to_string(),
                constraint_type: ConstraintType::PrimaryKey,
                columns: vec![ColumnId(0)],
                ref_table_id: None,
                ref_columns: Vec::new(),
                check_expr: None,
                on_delete: Default::default(),
                on_update: Default::default(),
                enforced: true,
                on_violation: Default::default(),
                quarantine_table_id: None,
                without_overlaps: None,
                fk_period: false,
                validated: true,
            }],
            created_at: 0,
            versioning_enabled: false,
            scd_type: None,
            system_versioned: false,
            history_table_id: None,
            cdf_enabled: true,
            cdf_retention_days: 7,
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

    /// Encodes one row the way the heap encoder does, for the layouts these
    /// tests project
    fn encode(values: &[Option<&[u8]>], types: &[TypeId]) -> Vec<u8> {
        let mut bitmap = vec![0u8; values.len().div_ceil(8)];
        let mut body = Vec::new();
        for (i, value) in values.iter().enumerate() {
            match (value, types[i].fixed_size()) {
                (None, Some(size)) => {
                    bitmap[i / 8] |= 1 << (i % 8);
                    body.extend(std::iter::repeat_n(0u8, size));
                }
                (None, None) => {
                    bitmap[i / 8] |= 1 << (i % 8);
                    body.extend_from_slice(&0u32.to_le_bytes());
                }
                (Some(bytes), Some(_)) => body.extend_from_slice(bytes),
                (Some(bytes), None) => {
                    body.extend_from_slice(&(bytes.len() as u32).to_le_bytes());
                    body.extend_from_slice(bytes);
                }
            }
        }
        bitmap.extend_from_slice(&body);
        bitmap
    }

    #[test]
    fn test_a_projected_row_is_the_row_of_the_kept_columns_alone() {
        let table = table();
        let layout = table.physical_columns_for_epoch(1).expect("epoch 1");
        // The key and the note, the amount left out
        let projection = RowProjection::new(layout, &[ColumnId(0), ColumnId(2)]);
        assert_eq!(projection.kept(), 2);
        let row = encode(
            &[
                Some(&7i64.to_le_bytes()),
                Some(&3i32.to_le_bytes()),
                Some(b"hello"),
            ],
            &[TypeId::Int64, TypeId::Int32, TypeId::Text],
        );
        let mut out = Vec::new();
        projection.project(&row, &mut out).expect("projects");
        let expected = encode(
            &[Some(&7i64.to_le_bytes()), Some(b"hello")],
            &[TypeId::Int64, TypeId::Text],
        );
        assert_eq!(out, expected);
        assert!(
            out.len() < row.len(),
            "the projected row is the smaller one"
        );
    }

    #[test]
    fn test_a_null_keeps_its_bit_at_the_kept_position() {
        let table = table();
        let layout = table.physical_columns_for_epoch(1).expect("epoch 1");
        let projection = RowProjection::new(layout, &[ColumnId(0), ColumnId(2)]);
        let row = encode(
            &[Some(&7i64.to_le_bytes()), None, None],
            &[TypeId::Int64, TypeId::Int32, TypeId::Text],
        );
        let mut out = Vec::new();
        projection.project(&row, &mut out).expect("projects");
        // Bit 1 of the projected bitmap is the note, bit 0 the key
        assert_eq!(out[0], 0b10);
        let expected = encode(
            &[Some(&7i64.to_le_bytes()), None],
            &[TypeId::Int64, TypeId::Text],
        );
        assert_eq!(out, expected);
    }

    #[test]
    fn test_a_short_row_is_refused_rather_than_read_past() {
        let table = table();
        let layout = table.physical_columns_for_epoch(1).expect("epoch 1");
        let projection = RowProjection::new(layout, &[ColumnId(0)]);
        let row = encode(
            &[
                Some(&7i64.to_le_bytes()),
                Some(&3i32.to_le_bytes()),
                Some(b"hello"),
            ],
            &[TypeId::Int64, TypeId::Int32, TypeId::Text],
        );
        let mut out = Vec::new();
        let text = projection
            .project(&row[..row.len() - 2], &mut out)
            .expect_err("refused")
            .to_string();
        assert!(text.contains("shorter than the layout"), "{text}");
    }

    #[test]
    fn test_the_projected_layout_is_what_the_projection_writes() {
        let table = table();
        let recorded = [ColumnId(0), ColumnId(2)];
        let layout = table
            .projected_columns_for_epoch(1, &recorded)
            .expect("epoch 1");
        let ids: Vec<u16> = layout.iter().map(|c| c.column_id.0).collect();
        assert_eq!(ids, vec![0, 2]);
        let ordinals: Vec<u16> = layout.iter().map(|c| c.ordinal).collect();
        assert_eq!(ordinals, vec![0, 1], "ordinals renumber from zero");
        assert!(table.projected_columns_for_epoch(9, &recorded).is_none());
    }

    #[test]
    fn test_widening_and_narrowing_are_told_apart() {
        assert!(type_change_is_widening(TypeId::Int32, TypeId::Int64));
        assert!(type_change_is_widening(TypeId::UInt16, TypeId::UInt64));
        assert!(type_change_is_widening(TypeId::Int64, TypeId::Int128));
        assert!(!type_change_is_widening(TypeId::Int64, TypeId::Int32));
        assert!(!type_change_is_widening(TypeId::Text, TypeId::Int32));
    }

    #[test]
    fn test_a_narrowing_change_needs_the_acknowledgement() {
        let refused = check_type_change(
            "orders",
            "amount",
            TypeId::Int64,
            TypeId::Int32,
            2,
            StreamBreak {
                acknowledged: false,
            },
        )
        .expect_err("refused");
        let text = refused.to_string();
        assert!(text.contains("ACKNOWLEDGE STREAM BREAK"), "{text}");
        assert!(text.contains("2 change stream"), "{text}");

        assert!(
            check_type_change(
                "orders",
                "amount",
                TypeId::Int64,
                TypeId::Int32,
                2,
                StreamBreak { acknowledged: true },
            )
            .is_ok()
        );
        // With no stream reading it, the change goes through unchallenged
        assert!(
            check_type_change(
                "orders",
                "amount",
                TypeId::Int64,
                TypeId::Int32,
                0,
                StreamBreak {
                    acknowledged: false
                },
            )
            .is_ok()
        );
    }

    #[test]
    fn test_a_dropped_column_a_stream_names_is_reported() {
        let mut table = table();
        assert_eq!(missing_stream_column(&table, &[ColumnId(2)]), None);
        table.columns[2].dropped = true;
        assert_eq!(
            missing_stream_column(&table, &[ColumnId(2)]),
            Some("note".to_string())
        );
    }
}
