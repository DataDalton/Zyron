//! Segment bloom filters must be addressable on disk, and a caller holding
//! a whole column must be able to pick its encoding from the whole column.
//!
//! The bloom bytes were written into every .zyr segment while the header
//! recorded offset 0, so nothing could find them. These tests pin the offset
//! the writer actually uses and the reader that follows it.
//!
//! Run: cargo test -p zyron-storage --test columnar_bloom_offset

use zyron_common::types::TypeId;
use zyron_storage::columnar::{
    BloomPolicy, ColumnSegment, SEGMENT_HEADER_SIZE, SegmentOptions, SortOrder, ZyrFileHeader,
    ZyrFileReader, ZyrFileWriter,
};
use zyron_storage::encoding::{create_encoding, select_encoding, select_encoding_exact};

fn write_one_column(path: &std::path::Path, values: &[Option<&[u8]>], policy: BloomPolicy) {
    let segment = ColumnSegment::build_with_options(
        0,
        TypeId::Int64,
        8,
        values,
        SegmentOptions {
            bloom: policy,
            exact_encoding: false,
        },
    )
    .expect("build");
    assert_eq!(
        segment.header.bloom_filter_offset,
        if segment.bloom_filter.is_some() {
            SEGMENT_HEADER_SIZE as u64
        } else {
            0
        },
        "a segment bloom starts right after the header"
    );

    let header = ZyrFileHeader {
        format_version: zyron_storage::columnar::ZYR_FORMAT_VERSION,
        column_count: 1,
        row_count: values.len() as u64,
        table_id: 1,
        xmin_range_lo: 0,
        xmin_range_hi: 0,
        xmax_range_lo: 0,
        xmax_range_hi: 0,
        primary_key_column_id: 0,
        sort_order: SortOrder::None,
    };
    let mut writer = ZyrFileWriter::create(path, header).expect("create");
    let zone_bytes: Vec<u8> = segment
        .zone_maps
        .iter()
        .flat_map(|z| z.to_bytes())
        .collect();
    let bloom_bytes = segment.bloom_filter.as_ref().map(|b| b.to_bytes());
    writer
        .write_segment(
            0,
            &segment.header.to_bytes(),
            bloom_bytes.as_deref(),
            &zone_bytes,
            &segment.null_bitmap,
            &segment.encoded_data,
        )
        .expect("write segment");
    writer.finalize(false).expect("finalize");
}

#[test]
fn test_read_bloom_finds_the_filter_the_writer_placed() {
    let tmp = tempfile::TempDir::new().expect("temp dir");
    let path = tmp.path().join("bloom.zyr");

    let owned: Vec<[u8; 8]> = (0..2048i64).map(|v| (v * 3).to_le_bytes()).collect();
    let values: Vec<Option<&[u8]>> = owned.iter().map(|v| Some(v.as_slice())).collect();
    write_one_column(&path, &values, BloomPolicy::Auto);

    let reader = ZyrFileReader::open(&path).expect("open");
    let bloom = reader
        .read_bloom(0)
        .expect("read bloom")
        .expect("a 2048-distinct-value column carries a bloom");
    for v in &owned {
        assert!(
            bloom.might_contain(v.as_slice()),
            "the filter must admit every value the segment inserted"
        );
    }
    // A value the column never held, well inside its range
    assert!(!bloom.might_contain(&1i64.to_le_bytes()));

    // The data region still decodes, so moving the recorded offset did not
    // shift what the reader treats as payload
    let (decoded, nulls) = reader.decode_column(0, owned.len(), 8).expect("decode");
    assert!(nulls.is_empty());
    assert_eq!(decoded.len(), owned.len() * 8);
    assert_eq!(&decoded[..8], owned[0].as_slice());
    assert_eq!(
        &decoded[decoded.len() - 8..],
        owned[owned.len() - 1].as_slice()
    );
}

#[test]
fn test_suppressed_and_forced_bloom_policies_are_honored() {
    let tmp = tempfile::TempDir::new().expect("temp dir");

    // Two distinct values, far below the cardinality threshold
    let owned: Vec<[u8; 8]> = (0..64i64).map(|v| (v % 2).to_le_bytes()).collect();
    let values: Vec<Option<&[u8]>> = owned.iter().map(|v| Some(v.as_slice())).collect();

    let auto = tmp.path().join("auto.zyr");
    write_one_column(&auto, &values, BloomPolicy::Auto);
    assert!(
        ZyrFileReader::open(&auto)
            .expect("open")
            .read_bloom(0)
            .expect("read")
            .is_none(),
        "the cardinality heuristic skips a two-value column"
    );

    let forced = tmp.path().join("forced.zyr");
    write_one_column(&forced, &values, BloomPolicy::Force);
    let bloom = ZyrFileReader::open(&forced)
        .expect("open")
        .read_bloom(0)
        .expect("read")
        .expect("Force builds one whatever the cardinality says");
    assert!(bloom.might_contain(&0i64.to_le_bytes()));
    assert!(bloom.might_contain(&1i64.to_le_bytes()));

    let suppressed = tmp.path().join("suppressed.zyr");
    let high: Vec<[u8; 8]> = (0..2048i64).map(|v| (v * 7).to_le_bytes()).collect();
    let high_values: Vec<Option<&[u8]>> = high.iter().map(|v| Some(v.as_slice())).collect();
    write_one_column(&suppressed, &high_values, BloomPolicy::Suppress);
    assert!(
        ZyrFileReader::open(&suppressed)
            .expect("open")
            .read_bloom(0)
            .expect("read")
            .is_none(),
        "Suppress refuses a filter the heuristic would have built"
    );
}

/// Encoded byte count for a fixed-width column under one encoding choice.
fn encoded_len(encoding: zyron_storage::encoding::EncodingType, raw: &[u8], rows: usize) -> usize {
    create_encoding(encoding)
        .encode(raw, rows, 8)
        .expect("encode")
        .len()
}

#[test]
fn test_select_encoding_exact_never_picks_worse_than_sampling() {
    // A column whose leading rows are full-range values no integer encoding
    // helps with, followed by a long constant-step run FastLanes collapses.
    // Sampling only ever sees the incompressible prefix and gives up.
    const ROWS: usize = 8192;
    let mut state = 0x2545_F491_4F6C_DD1Du64;
    let owned: Vec<[u8; 8]> = (0..ROWS)
        .map(|i| {
            if i < 1024 {
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                (state as i64).to_le_bytes()
            } else {
                (i as i64).to_le_bytes()
            }
        })
        .collect();
    let values: Vec<Option<&[u8]>> = owned.iter().map(|v| Some(v.as_slice())).collect();
    let raw: Vec<u8> = owned.iter().flat_map(|v| v.iter().copied()).collect();

    let sampled = select_encoding(TypeId::Int64, &values);
    let exact = select_encoding_exact(TypeId::Int64, &values);
    assert_ne!(
        sampled, exact,
        "the prefix is meant to mislead the sampling variant"
    );
    assert!(
        encoded_len(exact, &raw, ROWS) <= encoded_len(sampled, &raw, ROWS),
        "exact selection produced {:?} at {} bytes against sampled {:?} at {} bytes",
        exact,
        encoded_len(exact, &raw, ROWS),
        sampled,
        encoded_len(sampled, &raw, ROWS)
    );

    // On a column with no misleading prefix the two agree, so exact costs
    // nothing but the extra trial encode
    let plain: Vec<[u8; 8]> = (0..ROWS).map(|i| (i as i64 * 5).to_le_bytes()).collect();
    let plain_values: Vec<Option<&[u8]>> = plain.iter().map(|v| Some(v.as_slice())).collect();
    assert_eq!(
        select_encoding(TypeId::Int64, &plain_values),
        select_encoding_exact(TypeId::Int64, &plain_values)
    );
}
