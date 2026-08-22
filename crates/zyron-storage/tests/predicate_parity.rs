//! Every encoding's pushdown answer has to be the answer decoding and
//! comparing gives.
//!
//! `eval_predicate` exists to skip the decode, not to answer differently.
//! Two segments of one column choose their encodings independently, so a
//! predicate that resolved one way against a bit-packed segment and another
//! against a run length one would return different rows for the same query
//! depending on how the writer happened to pack each file.
//!
//! The row counts run either side of a byte boundary because the mask is
//! built eight rows at a time and the last partial byte is where a bit past
//! the row count would appear.

use zyron_storage::encoding::{EncodingType, Predicate, create_encoding, eval_predicate_on_raw};

/// Row counts either side of the mask's byte boundary, plus sizes that span
/// many bytes
const ROW_COUNTS: &[usize] = &[1, 7, 8, 9, 63, 64, 65, 1_000, 1_024, 4_097];

/// Encodings a fixed-width column can be written with
const FIXED_ENCODINGS: &[EncodingType] = &[
    EncodingType::Unencoded,
    EncodingType::BitPack,
    EncodingType::Rle,
    EncodingType::Dictionary,
    EncodingType::FastLanes,
];

/// Encodings a variable-length column can be written with
const VARLEN_ENCODINGS: &[EncodingType] = &[
    EncodingType::Unencoded,
    EncodingType::Dictionary,
    EncodingType::Fsst,
];

fn pack(values: &[i64], width: usize) -> Vec<u8> {
    let mut raw = Vec::with_capacity(values.len() * width);
    for value in values {
        raw.extend_from_slice(&value.to_le_bytes()[..width]);
    }
    raw
}

fn cell(value: i64, width: usize) -> Vec<u8> {
    value.to_le_bytes()[..width].to_vec()
}

/// Asserts the encoding answers the predicate exactly as decoding and
/// comparing does, and that no bit lands past the last row
fn assert_parity(
    encoding_type: EncodingType,
    label: &str,
    raw: &[u8],
    row_count: usize,
    value_size: usize,
    predicate: &Predicate,
    predicate_label: &str,
) {
    let encoding = create_encoding(encoding_type);
    let encoded = encoding
        .encode(raw, row_count, value_size)
        .unwrap_or_else(|e| panic!("{label}/{encoding_type:?}: encode failed: {e}"));
    let decoded = encoding
        .decode(&encoded, row_count, value_size)
        .unwrap_or_else(|e| panic!("{label}/{encoding_type:?}: decode failed: {e}"));
    assert_eq!(
        decoded, raw,
        "{label}/{encoding_type:?}: the column did not survive a round trip"
    );

    let expected = eval_predicate_on_raw(&decoded, row_count, value_size, predicate)
        .unwrap_or_else(|e| panic!("{label}/{encoding_type:?}: reference eval failed: {e}"));
    let actual = encoding
        .eval_predicate(&encoded, row_count, value_size, predicate)
        .unwrap_or_else(|e| panic!("{label}/{encoding_type:?}: pushdown eval failed: {e}"));

    assert_eq!(
        actual.len(),
        row_count.div_ceil(8),
        "{label}/{encoding_type:?}/{predicate_label}: the mask is the wrong length"
    );
    if let Some(last) = actual.last() {
        let trailing = row_count % 8;
        if trailing != 0 {
            assert_eq!(
                last & !((1u8 << trailing) - 1),
                0,
                "{label}/{encoding_type:?}/{predicate_label}: a bit is set past the last row"
            );
        }
    }
    assert_eq!(
        actual, expected,
        "{label}/{encoding_type:?}/{predicate_label}: pushdown disagrees with decode and compare"
    );
}

/// Fixed-width columns, over every shape that reaches a different branch of
/// the encoded-domain evaluators
#[test]
fn fixed_width_pushdown_answers_what_decoding_answers() {
    for &rows in ROW_COUNTS {
        let shapes: Vec<(&str, Vec<i64>)> = vec![
            ("ascending", (0..rows).map(|i| i as i64).collect()),
            (
                "constant step",
                (0..rows).map(|i| 100 + i as i64 * 7).collect(),
            ),
            (
                "scattered",
                (0..rows).map(|i| ((i * 2_731) % rows) as i64).collect(),
            ),
            (
                "low cardinality",
                (0..rows).map(|i| (i % 5) as i64).collect(),
            ),
            ("one run", (0..rows).map(|_| 42i64).collect()),
            ("runs of eight", (0..rows).map(|i| (i / 8) as i64).collect()),
        ];

        for width in [4usize, 8] {
            for (shape, values) in &shapes {
                let raw = pack(values, width);
                let present = values[values.len() / 3];
                let absent = (rows as i64) * 1_000 + 7;
                let low = values.iter().copied().min().unwrap_or(0);
                let high = values.iter().copied().max().unwrap_or(0);
                let mid = low + (high - low) / 2;

                let lo_cell = cell(low.max(0) + 1, width);
                let hi_cell = cell(mid, width);
                let present_cell = cell(present, width);
                let absent_cell = cell(absent, width);
                let other_cell = cell(values[values.len() / 2], width);
                let in_present: [&[u8]; 2] = [&present_cell, &other_cell];
                let in_mixed: [&[u8]; 2] = [&present_cell, &absent_cell];
                let in_absent: [&[u8]; 1] = [&absent_cell];

                let predicates: Vec<(&str, Predicate)> = vec![
                    ("eq present", Predicate::Equality(&present_cell)),
                    ("eq absent", Predicate::Equality(&absent_cell)),
                    (
                        "range both bounds",
                        Predicate::Range {
                            low: Some(&lo_cell),
                            high: Some(&hi_cell),
                        },
                    ),
                    (
                        "range low only",
                        Predicate::Range {
                            low: Some(&hi_cell),
                            high: None,
                        },
                    ),
                    (
                        "range high only",
                        Predicate::Range {
                            low: None,
                            high: Some(&hi_cell),
                        },
                    ),
                    (
                        "range unbounded",
                        Predicate::Range {
                            low: None,
                            high: None,
                        },
                    ),
                    (
                        "range empty",
                        Predicate::Range {
                            low: Some(&absent_cell),
                            high: Some(&lo_cell),
                        },
                    ),
                    ("in two present", Predicate::In(&in_present)),
                    ("in one absent", Predicate::In(&in_mixed)),
                    ("in all absent", Predicate::In(&in_absent)),
                ];

                let label = format!("{shape} w{width} n{rows}");
                for encoding_type in FIXED_ENCODINGS {
                    // FastLanes packs 4 and 8 byte cells, and a wider or
                    // narrower column goes to another encoding
                    for (predicate_label, predicate) in &predicates {
                        assert_parity(
                            *encoding_type,
                            &label,
                            &raw,
                            rows,
                            width,
                            predicate,
                            predicate_label,
                        );
                    }
                }
                // A column of one repeated value is what Constant stores
                if *shape == "one run" {
                    for (predicate_label, predicate) in &predicates {
                        assert_parity(
                            EncodingType::Constant,
                            &label,
                            &raw,
                            rows,
                            width,
                            predicate,
                            predicate_label,
                        );
                    }
                }
            }
        }
    }
}

/// Variable-length columns, where the order is the bytes from the first and
/// a dictionary resolves through its sorted entries
#[test]
fn variable_length_pushdown_answers_what_decoding_answers() {
    for &rows in ROW_COUNTS {
        let shapes: Vec<(&str, Vec<Vec<u8>>)> = vec![
            (
                "distinct",
                (0..rows)
                    .map(|i| format!("row-{:08}", i).into_bytes())
                    .collect(),
            ),
            (
                "enum like",
                (0..rows)
                    .map(|i| {
                        [
                            b"active".to_vec(),
                            b"inactive".to_vec(),
                            b"suspended".to_vec(),
                        ][i % 3]
                            .clone()
                    })
                    .collect(),
            ),
            (
                "mixed lengths with an empty",
                (0..rows)
                    .map(|i| match i % 4 {
                        0 => Vec::new(),
                        1 => b"a".to_vec(),
                        2 => format!("{}", (i * 2_731) % rows).into_bytes(),
                        _ => vec![b'z'; 40 + (i % 11)],
                    })
                    .collect(),
            ),
            ("one run", (0..rows).map(|_| b"same".to_vec()).collect()),
        ];

        for (shape, values) in &shapes {
            let views: Vec<Option<&[u8]>> = values.iter().map(|v| Some(v.as_slice())).collect();
            let raw = zyron_storage::encoding::varlen_pack(&views);
            let present = values[values.len() / 3].clone();
            let other = values[values.len() / 2].clone();
            let absent = b"\xff\xff-no-such-value".to_vec();
            let lo = b"a".to_vec();
            let hi = b"row-99999999".to_vec();
            let in_present: [&[u8]; 2] = [&present, &other];
            let in_mixed: [&[u8]; 2] = [&present, &absent];
            let in_absent: [&[u8]; 1] = [&absent];

            let predicates: Vec<(&str, Predicate)> = vec![
                ("eq present", Predicate::Equality(&present)),
                ("eq absent", Predicate::Equality(&absent)),
                (
                    "range both bounds",
                    Predicate::Range {
                        low: Some(&lo),
                        high: Some(&hi),
                    },
                ),
                (
                    "range low only",
                    Predicate::Range {
                        low: Some(&lo),
                        high: None,
                    },
                ),
                (
                    "range high only",
                    Predicate::Range {
                        low: None,
                        high: Some(&hi),
                    },
                ),
                (
                    "range empty",
                    Predicate::Range {
                        low: Some(&absent),
                        high: Some(&lo),
                    },
                ),
                ("in two present", Predicate::In(&in_present)),
                ("in one absent", Predicate::In(&in_mixed)),
                ("in all absent", Predicate::In(&in_absent)),
            ];

            let label = format!("{shape} varlen n{rows}");
            for encoding_type in VARLEN_ENCODINGS {
                for (predicate_label, predicate) in &predicates {
                    assert_parity(
                        *encoding_type,
                        &label,
                        &raw,
                        rows,
                        0,
                        predicate,
                        predicate_label,
                    );
                }
            }
            if *shape == "one run" {
                for (predicate_label, predicate) in &predicates {
                    assert_parity(
                        EncodingType::Constant,
                        &label,
                        &raw,
                        rows,
                        0,
                        predicate,
                        predicate_label,
                    );
                }
            }
        }
    }
}

/// Float columns, which ALP resolves in float order. A Range is never
/// pushed to one, because the stored bytes put negatives above positives
/// and reverse their order, so equality and membership are what a float
/// column answers from its encoded form
#[test]
fn float_pushdown_answers_what_decoding_answers() {
    for &rows in ROW_COUNTS {
        for width in [4usize, 8] {
            let values: Vec<f64> = (0..rows)
                .map(|i| ((i * 2_731) % rows) as f64 * 1.25 - 512.0)
                .collect();
            let mut raw = Vec::with_capacity(rows * width);
            for value in &values {
                if width == 4 {
                    raw.extend_from_slice(&(*value as f32).to_le_bytes());
                } else {
                    raw.extend_from_slice(&value.to_le_bytes());
                }
            }
            let float_cell = |value: f64| -> Vec<u8> {
                if width == 4 {
                    (value as f32).to_le_bytes().to_vec()
                } else {
                    value.to_le_bytes().to_vec()
                }
            };
            let present = float_cell(values[values.len() / 3]);
            let other = float_cell(values[values.len() / 2]);
            let absent = float_cell(9_999_999.5);
            let in_present: [&[u8]; 2] = [&present, &other];
            let in_absent: [&[u8]; 1] = [&absent];

            let predicates: Vec<(&str, Predicate)> = vec![
                ("eq present", Predicate::Equality(&present)),
                ("eq absent", Predicate::Equality(&absent)),
                ("in two present", Predicate::In(&in_present)),
                ("in all absent", Predicate::In(&in_absent)),
            ];

            let label = format!("float w{width} n{rows}");
            for encoding_type in [EncodingType::Unencoded, EncodingType::Alp] {
                for (predicate_label, predicate) in &predicates {
                    assert_parity(
                        encoding_type,
                        &label,
                        &raw,
                        rows,
                        width,
                        predicate,
                        predicate_label,
                    );
                }
            }
        }
    }
}
