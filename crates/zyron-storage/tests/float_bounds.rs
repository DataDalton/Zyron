//! A float column's recorded bounds have to be its value bounds.
//!
//! Stat slots hold raw little endian value bytes and are compared as
//! unsigned integers with a sign fix for two's complement. Neither of
//! those orders IEEE floats: a negative float's bit pattern is larger the
//! further from zero it is, so under an unsigned reading the extremes of
//! a mixed-sign column are the smallest non-negative and the most
//! negative, and the largest value is never recorded at all.
//!
//! That is not only a lost pruning opportunity. `MIN(c)` and `MAX(c)`
//! over a folded column are answered straight out of these bytes, so a
//! wrong order is a wrong answer.

use zyron_storage::columnar::{ColumnSegment, STAT_VALUE_SIZE};
use zyron_common::TypeId;

fn cells(values: &[f64]) -> Vec<[u8; 8]> {
    values.iter().map(|v| v.to_le_bytes()).collect()
}

fn slot_f64(slot: &[u8; STAT_VALUE_SIZE]) -> f64 {
    let mut buf = [0u8; 8];
    buf.copy_from_slice(&slot[..8]);
    f64::from_le_bytes(buf)
}

fn bounds_of(values: &[f64]) -> (f64, f64) {
    let raw = cells(values);
    let refs: Vec<Option<&[u8]>> = raw.iter().map(|c| Some(&c[..])).collect();
    let segment =
        ColumnSegment::build(0, TypeId::Float64, 8, &refs).expect("segment builds");
    (
        slot_f64(&segment.header.min_value),
        slot_f64(&segment.header.max_value),
    )
}

#[test]
fn test_a_float_column_records_its_value_bounds() {
    // All non-negative: an unsigned reading already agrees with the values
    let (min, max) = bounds_of(&[1.5, 0.25, 7.0, 3.0]);
    assert_eq!((min, max), (0.25, 7.0));

    // All negative: the unsigned reading runs backwards
    let (min, max) = bounds_of(&[-1.5, -0.25, -7.0, -3.0]);
    assert_eq!((min, max), (-7.0, -0.25));

    // Mixed, which is where an unsigned reading loses the largest value
    let (min, max) = bounds_of(&[-2.0, 3.0, -1.0, 0.5]);
    assert_eq!(
        (min, max),
        (-2.0, 3.0),
        "the bounds must be the smallest and largest values"
    );

    // Zero has two spellings and they compare equal
    let (min, max) = bounds_of(&[-0.0, 0.0, 1.0]);
    assert_eq!(max, 1.0);
    assert_eq!(min, 0.0, "negative zero is not below zero");

    // A single value is both bounds
    let (min, max) = bounds_of(&[-4.25]);
    assert_eq!((min, max), (-4.25, -4.25));
}

#[test]
fn test_a_float_zone_records_its_value_bounds() {
    // Two zones worth of rows, the second holding the extremes, so a zone
    // map that ordered them wrongly would disagree with the segment
    let mut values: Vec<f64> = (0..1_024).map(|i| (i % 7) as f64).collect();
    values.extend((0..1_024).map(|i| if i % 2 == 0 { -(i as f64) } else { i as f64 }));

    let raw = cells(&values);
    let refs: Vec<Option<&[u8]>> = raw.iter().map(|c| Some(&c[..])).collect();
    let segment =
        ColumnSegment::build(0, TypeId::Float64, 8, &refs).expect("segment builds");
    assert_eq!(segment.zone_maps.len(), 2);

    let expected_min = values.iter().cloned().fold(f64::INFINITY, f64::min);
    let expected_max = values.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    assert_eq!(slot_f64(&segment.header.min_value), expected_min);
    assert_eq!(slot_f64(&segment.header.max_value), expected_max);

    // Every zone's own bounds bracket the rows it covers
    for (z, zone) in segment.zone_maps.iter().enumerate() {
        let start = z * 1_024;
        let end = ((z + 1) * 1_024).min(values.len());
        let zone_min = values[start..end].iter().cloned().fold(f64::INFINITY, f64::min);
        let zone_max = values[start..end]
            .iter()
            .cloned()
            .fold(f64::NEG_INFINITY, f64::max);
        assert_eq!(slot_f64(&zone.min_value), zone_min, "zone {} min", z);
        assert_eq!(slot_f64(&zone.max_value), zone_max, "zone {} max", z);
    }
}
