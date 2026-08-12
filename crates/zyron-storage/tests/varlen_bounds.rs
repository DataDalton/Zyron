//! A variable-length column's recorded bounds have to bracket its values.
//!
//! Stat slots hold at most 32 bytes, so a longer value is held as a
//! prefix. A prefix sorts below the value it came from, which makes it a
//! sound lower bound and an unsound upper one, so the maximum is rounded
//! up to the smallest 32-byte string above every value carrying it.
//!
//! The comparison also has to run from the first byte. Slots for
//! fixed-width columns are compared from the last, because those hold
//! little endian numbers, and reading a string that way orders it by its
//! tail.

use zyron_common::TypeId;
use zyron_storage::columnar::{
    ColumnSegment, SlotOrder, STAT_VALUE_SIZE, compare_stat_slots_typed, compare_value_to_slot,
    slot_order, varlen_upper_slot,
};

fn segment_of(values: &[&[u8]]) -> ColumnSegment {
    let refs: Vec<Option<&[u8]>> = values.iter().map(|v| Some(*v)).collect();
    ColumnSegment::build(0, TypeId::Varchar, 0, &refs).expect("segment builds")
}

/// Whether the recorded bounds admit `value`, which is the question every
/// pruning decision asks
fn bounds_admit(segment: &ColumnSegment, value: &[u8]) -> bool {
    let order = slot_order(TypeId::Varchar);
    compare_value_to_slot(value, &segment.header.min_value, 0, order)
        != std::cmp::Ordering::Less
        && compare_value_to_slot(value, &segment.header.max_value, 0, order)
            != std::cmp::Ordering::Greater
}

#[test]
fn test_a_variable_length_column_records_bounds_that_bracket_its_values() {
    let values: Vec<&[u8]> = vec![b"pear", b"apple", b"zebra", b"mango"];
    let segment = segment_of(&values);

    // Every value the column holds is inside its own bounds
    for v in &values {
        assert!(
            bounds_admit(&segment, v),
            "the bounds exclude {:?}, which the column holds",
            std::str::from_utf8(v)
        );
    }
    // And values outside the range are outside the bounds
    assert!(!bounds_admit(&segment, b"aardvark"));
    assert!(!bounds_admit(&segment, b"zzz"));
    // Something between the extremes is admitted, which is all bounds can say
    assert!(bounds_admit(&segment, b"kiwi"));
}

#[test]
fn test_rows_of_different_lengths_do_not_order_by_their_tails() {
    // "b" is above "az" lexicographically and below it when the bytes are
    // read from the last, which is how a fixed-width slot is compared
    let values: Vec<&[u8]> = vec![b"az", b"b", b"a"];
    let segment = segment_of(&values);
    for v in &values {
        assert!(
            bounds_admit(&segment, v),
            "the bounds exclude {:?}",
            std::str::from_utf8(v)
        );
    }
    assert!(!bounds_admit(&segment, b"c"), "c is above every value");
}

#[test]
fn test_a_value_longer_than_a_slot_still_falls_inside_its_bounds() {
    let long_a = vec![b'a'; STAT_VALUE_SIZE + 40];
    let mut long_z = vec![b'z'; STAT_VALUE_SIZE];
    long_z.extend_from_slice(b"tail-beyond-the-prefix");
    let values: Vec<&[u8]> = vec![&long_a, &long_z, b"middle"];
    let segment = segment_of(&values);

    for v in &values {
        assert!(
            bounds_admit(&segment, v),
            "the bounds exclude a {} byte value",
            v.len()
        );
    }

    // The rounded-up maximum is strictly above the value it came from,
    // which is what makes it a bound rather than a truncation
    let upper = varlen_upper_slot(&long_z);
    assert_eq!(
        compare_value_to_slot(&long_z, &upper, 0, SlotOrder::Lexicographic),
        std::cmp::Ordering::Less,
        "a truncated maximum has to be rounded above its value"
    );
    // A value that fits needs no rounding
    assert_eq!(
        varlen_upper_slot(b"short"),
        {
            let mut slot = [0u8; STAT_VALUE_SIZE];
            slot[..5].copy_from_slice(b"short");
            slot
        },
        "a value inside a slot is its own bound"
    );
    // Nothing sits above an all-ones prefix, and that slot bounds anything
    let all_ones = vec![0xFFu8; STAT_VALUE_SIZE + 1];
    assert_eq!(varlen_upper_slot(&all_ones), [0xFF; STAT_VALUE_SIZE]);
}

#[test]
fn test_slots_compare_from_the_first_byte() {
    let order = slot_order(TypeId::Varchar);
    assert_eq!(order, SlotOrder::Lexicographic);

    let slot = |s: &[u8]| {
        let mut out = [0u8; STAT_VALUE_SIZE];
        out[..s.len()].copy_from_slice(s);
        out
    };
    assert_eq!(
        compare_stat_slots_typed(&slot(b"apple"), &slot(b"banana"), 0, order),
        std::cmp::Ordering::Less
    );
    // A shorter value is below one that extends it
    assert_eq!(
        compare_stat_slots_typed(&slot(b"app"), &slot(b"apple"), 0, order),
        std::cmp::Ordering::Less
    );
    assert_eq!(
        compare_stat_slots_typed(&slot(b"apple"), &slot(b"apple"), 0, order),
        std::cmp::Ordering::Equal
    );
}
