//! Reading a heap tuple through the layout it was written under.
//!
//! A tuple's bytes say nothing about how many columns they hold. The slot
//! carries an epoch, the table records what each epoch's layout was, and the
//! two together are the only thing that can walk the row. These check the four
//! things that can differ between the layout a row was written under and the
//! table as it stands: a column the row predates, a column since dropped, a
//! column whose type widened, and a null bitmap that changed length.
//!
//! Run: cargo test -p zyron-executor --test tuple_epoch_decode_test -- --nocapture

use zyron_catalog::{ColumnEntry, ColumnId, SchemaId, TableEntry, TableId};
use zyron_common::TypeId;
use zyron_common::page::PageId;
use zyron_executor::batch::{ColumnBuilder, create_builders};
use zyron_executor::column::ScalarValue;
use zyron_executor::epoch_decode::EpochDecoder;
use zyron_planner::logical::LogicalColumn;

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
        schema_id: SchemaId(1),
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

/// Encodes one row the way the heap encoder does: a null bitmap sized from the
/// layout, then each value at its physical width, variable-length values
/// behind a four-byte length.
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
            Some(ScalarValue::Int16(v)) => body.extend_from_slice(&v.to_le_bytes()),
            Some(ScalarValue::Int32(v)) => body.extend_from_slice(&v.to_le_bytes()),
            Some(ScalarValue::Int64(v)) => body.extend_from_slice(&v.to_le_bytes()),
            Some(ScalarValue::Int128(v)) => body.extend_from_slice(&v.to_le_bytes()),
            Some(ScalarValue::Utf8(s)) => {
                body.extend_from_slice(&(s.len() as u32).to_le_bytes());
                body.extend_from_slice(s.as_bytes());
            }
            Some(other) => panic!("the test encoder does not carry {other:?}"),
        }
    }
    bitmap.extend_from_slice(&body);
    bitmap
}

fn logical_of(table: &TableEntry) -> Vec<LogicalColumn> {
    table
        .live_columns()
        .map(|c| LogicalColumn {
            table_idx: Some(0),
            column_id: c.id,
            name: c.name.clone(),
            type_id: c.type_id,
            nullable: c.nullable,
            fractional_digits: c.fractional_digits,
        })
        .collect()
}

fn decode_row(table: &TableEntry, epoch: u16, row: &[u8]) -> Vec<ScalarValue> {
    let logical = logical_of(table);
    let ids: Vec<ColumnId> = logical.iter().map(|c| c.column_id).collect();
    let decoder = EpochDecoder::new(table, &ids);
    let mut builders: Vec<ColumnBuilder> = create_builders(&logical, 1);
    decoder
        .decode(epoch, row, None, &mut builders)
        .expect("the row decodes");
    builders
        .into_iter()
        .map(|b| b.finish().get_scalar(0))
        .collect()
}

#[test]
fn test_a_row_of_the_current_epoch_decodes_every_column() {
    let t = table(vec![
        column(0, "a", TypeId::Int32, 0),
        column(1, "b", TypeId::Text, 1),
        column(2, "c", TypeId::Int64, 2),
    ]);
    let row = encode(
        &[
            Some(ScalarValue::Int32(7)),
            Some(ScalarValue::Utf8("hello".into())),
            Some(ScalarValue::Int64(-1)),
        ],
        &[TypeId::Int32, TypeId::Text, TypeId::Int64],
    );
    let values = decode_row(&t, 1, &row);
    assert_eq!(values[0], ScalarValue::Int32(7));
    assert_eq!(values[1], ScalarValue::Utf8("hello".into()));
    assert_eq!(values[2], ScalarValue::Int64(-1));
}

#[test]
fn test_a_row_older_than_a_column_reads_the_value_recorded_when_it_was_added() {
    let mut t = table(vec![column(0, "a", TypeId::Int32, 0)]);
    let mut added = column(1, "added", TypeId::Int32, 1);
    added.absent_value = Some(7i32.to_le_bytes().to_vec());
    t.columns.push(added);
    t.push_schema_epoch(t.current_physical_columns());

    // Epoch 1 held only column a
    let old = encode(&[Some(ScalarValue::Int32(3))], &[TypeId::Int32]);
    let values = decode_row(&t, 1, &old);
    assert_eq!(values[0], ScalarValue::Int32(3));
    assert_eq!(values[1], ScalarValue::Int32(7), "the absent value is read");

    // Epoch 2 holds both, and the stored value wins over the absent one
    let new = encode(
        &[Some(ScalarValue::Int32(3)), Some(ScalarValue::Int32(99))],
        &[TypeId::Int32, TypeId::Int32],
    );
    let values = decode_row(&t, 2, &new);
    assert_eq!(values[1], ScalarValue::Int32(99));
}

#[test]
fn test_a_column_added_without_a_default_reads_null_on_older_rows() {
    let mut t = table(vec![column(0, "a", TypeId::Text, 0)]);
    t.columns.push(column(1, "added", TypeId::Text, 1));
    t.push_schema_epoch(t.current_physical_columns());

    let old = encode(&[Some(ScalarValue::Utf8("x".into()))], &[TypeId::Text]);
    let values = decode_row(&t, 1, &old);
    assert_eq!(values[0], ScalarValue::Utf8("x".into()));
    assert_eq!(values[1], ScalarValue::Null);
}

#[test]
fn test_a_dropped_column_is_walked_past_and_never_pushed() {
    let mut t = table(vec![
        column(0, "a", TypeId::Int32, 0),
        column(1, "gone", TypeId::Text, 1),
        column(2, "c", TypeId::Int32, 2),
    ]);
    t.columns[1].dropped = true;

    let row = encode(
        &[
            Some(ScalarValue::Int32(1)),
            Some(ScalarValue::Utf8(
                "this must not shift the next column".into(),
            )),
            Some(ScalarValue::Int32(9)),
        ],
        &[TypeId::Int32, TypeId::Text, TypeId::Int32],
    );
    let values = decode_row(&t, 1, &row);
    assert_eq!(values.len(), 2, "the dropped column is not a live column");
    assert_eq!(values[0], ScalarValue::Int32(1));
    assert_eq!(
        values[1],
        ScalarValue::Int32(9),
        "the column after the placeholder is misaligned"
    );
}

#[test]
fn test_a_column_added_after_a_drop_lands_after_the_placeholder() {
    let mut t = table(vec![
        column(0, "a", TypeId::Int32, 0),
        column(1, "gone", TypeId::Int32, 1),
    ]);
    t.columns[1].dropped = true;
    let mut added = column(2, "later", TypeId::Int32, 2);
    added.absent_value = Some(5i32.to_le_bytes().to_vec());
    t.columns.push(added);
    t.push_schema_epoch(t.current_physical_columns());

    // A row of epoch 1, written when the dropped column still held a value
    let old = encode(
        &[Some(ScalarValue::Int32(1)), Some(ScalarValue::Int32(2))],
        &[TypeId::Int32, TypeId::Int32],
    );
    let values = decode_row(&t, 1, &old);
    assert_eq!(values, vec![ScalarValue::Int32(1), ScalarValue::Int32(5)]);

    // A row of epoch 2, which still carries the placeholder because dropping
    // a column never moves the bytes after it
    let new = encode(
        &[
            Some(ScalarValue::Int32(1)),
            Some(ScalarValue::Int32(2)),
            Some(ScalarValue::Int32(8)),
        ],
        &[TypeId::Int32, TypeId::Int32, TypeId::Int32],
    );
    let values = decode_row(&t, 2, &new);
    assert_eq!(values, vec![ScalarValue::Int32(1), ScalarValue::Int32(8)]);
}

#[test]
fn test_a_narrower_integer_widens_on_the_way_into_the_batch() {
    let mut t = table(vec![column(0, "n", TypeId::Int32, 0)]);
    t.columns[0].type_id = TypeId::Int64;
    t.push_schema_epoch(t.current_physical_columns());

    for value in [0i32, 1, -1, i32::MIN, i32::MAX] {
        let row = encode(&[Some(ScalarValue::Int32(value))], &[TypeId::Int32]);
        let values = decode_row(&t, 1, &row);
        assert_eq!(
            values[0],
            ScalarValue::Int64(value as i64),
            "a {value} written as INT did not widen"
        );
    }
}

#[test]
fn test_a_short_text_reads_back_whole_under_a_longer_declaration() {
    let mut t = table(vec![column(0, "s", TypeId::Varchar, 0)]);
    t.columns[0].max_length = Some(10);
    let mut t = {
        t.push_schema_epoch(t.current_physical_columns());
        t
    };
    // The declaration widens, which changes no byte
    t.columns[0].max_length = Some(40);
    t.push_schema_epoch(t.current_physical_columns());

    let row = encode(
        &[Some(ScalarValue::Utf8("abcdefghij".into()))],
        &[TypeId::Varchar],
    );
    for epoch in [1u16, 2] {
        let values = decode_row(&t, epoch, &row);
        assert_eq!(values[0], ScalarValue::Utf8("abcdefghij".into()));
    }
}

#[test]
fn test_the_null_bitmap_is_sized_from_the_epoch_not_the_current_schema() {
    // Eight columns, then a ninth. Eight fit one bitmap byte and nine need
    // two, so a row of the first layout read under the second would take its
    // first value out of the bitmap's second byte
    let mut t = table(
        (0..8)
            .map(|i| column(i, &format!("c{i}"), TypeId::Int32, i))
            .collect(),
    );
    let mut ninth = column(8, "c8", TypeId::Int32, 8);
    ninth.absent_value = Some(42i32.to_le_bytes().to_vec());
    t.columns.push(ninth);
    t.push_schema_epoch(t.current_physical_columns());

    let old_values: Vec<Option<ScalarValue>> =
        (0..8).map(|i| Some(ScalarValue::Int32(i as i32))).collect();
    let old = encode(&old_values, &vec![TypeId::Int32; 8]);
    assert_eq!(old.len(), 1 + 8 * 4, "eight columns take one bitmap byte");

    let values = decode_row(&t, 1, &old);
    for i in 0..8 {
        assert_eq!(
            values[i],
            ScalarValue::Int32(i as i32),
            "column {i} misread"
        );
    }
    assert_eq!(values[8], ScalarValue::Int32(42));

    let mut new_values = old_values.clone();
    new_values.push(Some(ScalarValue::Int32(100)));
    let new = encode(&new_values, &vec![TypeId::Int32; 9]);
    assert_eq!(new.len(), 2 + 9 * 4, "nine columns take two bitmap bytes");
    let values = decode_row(&t, 2, &new);
    assert_eq!(values[8], ScalarValue::Int32(100));
}

#[test]
fn test_nulls_in_an_older_layout_land_on_the_right_columns() {
    let mut t = table(vec![
        column(0, "a", TypeId::Int32, 0),
        column(1, "b", TypeId::Int32, 1),
        column(2, "c", TypeId::Int32, 2),
    ]);
    t.columns.push(column(3, "d", TypeId::Int32, 3));
    t.push_schema_epoch(t.current_physical_columns());

    let row = encode(
        &[
            Some(ScalarValue::Int32(1)),
            None,
            Some(ScalarValue::Int32(3)),
        ],
        &[TypeId::Int32; 3],
    );
    let values = decode_row(&t, 1, &row);
    assert_eq!(values[0], ScalarValue::Int32(1));
    assert_eq!(values[1], ScalarValue::Null);
    assert_eq!(values[2], ScalarValue::Int32(3));
    assert_eq!(values[3], ScalarValue::Null);
}

#[test]
fn test_a_row_stamped_zero_reads_through_the_recorded_pre_stamp_layout() {
    let mut t = table(vec![column(0, "a", TypeId::Int32, 0)]);
    // What the upgrade recorded: the layout the table's unstamped rows carry
    t.pre_stamp_columns = t.current_physical_columns();
    // A column added after the upgrade
    let mut added = column(1, "added", TypeId::Int32, 1);
    added.absent_value = Some(11i32.to_le_bytes().to_vec());
    t.columns.push(added);
    t.push_schema_epoch(t.current_physical_columns());

    let unstamped = encode(&[Some(ScalarValue::Int32(4))], &[TypeId::Int32]);
    let values = decode_row(&t, 0, &unstamped);
    assert_eq!(values[0], ScalarValue::Int32(4));
    assert_eq!(values[1], ScalarValue::Int32(11));
}

#[test]
fn test_an_epoch_the_table_never_wrote_is_reported_with_the_row_it_names() {
    let t = table(vec![column(0, "a", TypeId::Int32, 0)]);
    let ids = vec![ColumnId(0)];
    let decoder = EpochDecoder::new(&t, &ids);
    let mut builders = create_builders(&logical_of(&t), 1);
    let at = zyron_common::RowLocator::Heap {
        page: PageId::new(1, 314),
        slot: 27,
    };
    let err = decoder
        .decode(9, &[0u8; 8], Some(at), &mut builders)
        .expect_err("an epoch with no recorded layout cannot be read");
    let text = err.to_string();
    for expected in ["\"t\"", "page 314", "slot 27", "epoch 9"] {
        assert!(
            text.contains(expected),
            "the report omits {expected}: {text}"
        );
    }
}

#[test]
fn test_a_retired_epoch_is_reported_rather_than_read_through_another_layout() {
    let mut t = table(vec![column(0, "a", TypeId::Int32, 0)]);
    t.columns.push(column(1, "b", TypeId::Int32, 1));
    t.push_schema_epoch(t.current_physical_columns());
    // Vacuum reported that no live tuple carries epoch 1 any more
    t.schema_epochs.retain(|e| e.epoch >= 2);

    let ids: Vec<ColumnId> = vec![ColumnId(0), ColumnId(1)];
    let decoder = EpochDecoder::new(&t, &ids);
    let mut builders = create_builders(&logical_of(&t), 1);
    let err = decoder
        .decode(1, &[0u8; 12], None, &mut builders)
        .expect_err("a retired epoch names no layout");
    assert!(err.to_string().contains("epoch 1"), "{err}");
}

#[test]
fn test_a_projection_that_skips_a_column_still_walks_its_bytes() {
    let t = table(vec![
        column(0, "a", TypeId::Int32, 0),
        column(1, "skipped", TypeId::Text, 1),
        column(2, "c", TypeId::Int32, 2),
    ]);
    // Only the first and last are asked for
    let ids = vec![ColumnId(0), ColumnId(2)];
    let decoder = EpochDecoder::new(&t, &ids);
    let logical: Vec<LogicalColumn> = t
        .live_columns()
        .filter(|c| ids.contains(&c.id))
        .map(|c| LogicalColumn {
            table_idx: Some(0),
            column_id: c.id,
            name: c.name.clone(),
            type_id: c.type_id,
            nullable: c.nullable,
            fractional_digits: c.fractional_digits,
        })
        .collect();
    let mut builders = create_builders(&logical, 1);
    let row = encode(
        &[
            Some(ScalarValue::Int32(5)),
            Some(ScalarValue::Utf8("wide enough to shift things".into())),
            Some(ScalarValue::Int32(6)),
        ],
        &[TypeId::Int32, TypeId::Text, TypeId::Int32],
    );
    decoder
        .decode(1, &row, None, &mut builders)
        .expect("decodes");
    let values: Vec<ScalarValue> = builders
        .into_iter()
        .map(|b| b.finish().get_scalar(0))
        .collect();
    assert_eq!(values, vec![ScalarValue::Int32(5), ScalarValue::Int32(6)]);
}

#[test]
fn test_a_row_that_does_not_span_its_layout_is_declined_rather_than_misread() {
    let t = table(vec![
        column(0, "a", TypeId::Int32, 0),
        column(1, "b", TypeId::Int32, 1),
    ]);
    let ids = vec![ColumnId(0), ColumnId(1)];
    let decoder = EpochDecoder::new(&t, &ids);
    let mut builders = create_builders(&logical_of(&t), 1);
    // One bitmap byte and one four-byte value, where the layout needs two
    let truncated = vec![0u8; 5];
    assert!(
        !decoder.try_decode(1, &truncated, &mut builders),
        "a row shorter than its layout was read anyway"
    );
}
