#![allow(non_snake_case, unused_assignments)]

//! Encoding Engine Benchmark Suite
//!
//! Integration tests for ZyronDB columnar encoding:
//! - FastLanes, FSST, ALP, Dictionary, RLE round-trip correctness
//! - Encoding selection heuristics
//! - Compressed predicate evaluation
//!
//! Run: cargo test -p zyron-storage --test encoding_bench --release -- --nocapture

use zyron_bench_harness::*;

use rand::Rng;
use std::time::Instant;

use zyron_common::types::TypeId;
use zyron_storage::columnar::{
    ColumnDescriptor, ColumnSegment, STAT_VALUE_SIZE, SegmentHeader, ZYR_FORMAT_VERSION,
};
use zyron_storage::encoding::{
    EncodingType, Predicate, create_encoding, eval_predicate_on_raw, select_encoding,
};

// Performance targets
const FASTLANES_DECODE_TARGET_INT_SEC: f64 = 120_000_000_000.0;
const FSST_DECOMPRESS_TARGET_GB_SEC: f64 = 6.0;
const ALP_DECODE_TARGET_FLOAT_SEC: f64 = 3_000_000_000.0;
const DICTIONARY_LOOKUP_TARGET_NS: f64 = 3.0;
const RLE_DECODE_TARGET_VAL_SEC: f64 = 60_000_000_000.0;
const COMPRESSED_EVAL_SPEEDUP_TARGET: f64 = 3.0;
const ENCODING_SELECT_TARGET_US: f64 = 500.0;

static BENCHMARK_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

// =============================================================================
// Test 1: Round-Trip Correctness (all 8 encodings)
// =============================================================================

#[test]
fn test_encoding_round_trip_correctness() {
    zyron_bench_harness::init("encoding");
    let _benchGuard = BENCHMARK_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    const ROW_COUNT: usize = 100_000;

    tprintln!("\n=== Phase 1.7: Encoding Round-Trip Correctness ===");
    tprintln!("Rows per encoding: {}", ROW_COUNT);

    let utilBefore = take_util_snapshot();

    // -- FastLanes: sequential u32 --
    {
        tprintln!("\n  FastLanes (sequential i32):");
        let mut rawData = Vec::with_capacity(ROW_COUNT * 4);
        for i in 0..ROW_COUNT {
            rawData.extend_from_slice(&(1_000_000u32 + i as u32).to_le_bytes());
        }

        let encoder = create_encoding(EncodingType::FastLanes);
        let encoded = encoder
            .encode(&rawData, ROW_COUNT, 4)
            .expect("encode failed");
        tprintln!(
            "    Compression ratio: {:.2}x",
            rawData.len() as f64 / encoded.len() as f64
        );
        assert!(
            encoded.len() < rawData.len(),
            "FastLanes should compress sequential data"
        );

        let decoded = encoder
            .decode(&encoded, ROW_COUNT, 4)
            .expect("decode failed");
        assert_eq!(decoded, rawData, "FastLanes round-trip mismatch");

        // Performance: time decode
        let mut decodeResults = Vec::with_capacity(VALIDATION_RUNS);
        for _ in 0..VALIDATION_RUNS {
            let start = Instant::now();
            std::hint::black_box(encoder.decode(&encoded, ROW_COUNT, 4).unwrap());
            let elapsed = start.elapsed().as_secs_f64();
            decodeResults.push(ROW_COUNT as f64 / elapsed);
        }
        validate_metric(
            "Encoding Round-Trip",
            "FastLanes decode (int/sec)",
            decodeResults,
            FASTLANES_DECODE_TARGET_INT_SEC,
            true,
        );
    }

    // -- FSST: 32-byte fixed strings --
    {
        tprintln!("\n  FSST (32-byte strings):");
        let valueSize = 32;
        let mut rawData = Vec::with_capacity(ROW_COUNT * valueSize);
        for i in 0..ROW_COUNT {
            let base = format!("row_{:05}_padding_abcdefgh", i % 1000);
            let mut val = [0u8; 32];
            let bytes = base.as_bytes();
            let copyLen = bytes.len().min(32);
            val[..copyLen].copy_from_slice(&bytes[..copyLen]);
            rawData.extend_from_slice(&val);
        }

        let encoder = create_encoding(EncodingType::Fsst);
        let encoded = encoder
            .encode(&rawData, ROW_COUNT, valueSize)
            .expect("encode failed");
        tprintln!(
            "    Compression ratio: {:.2}x",
            rawData.len() as f64 / encoded.len() as f64
        );

        let decoded = encoder
            .decode(&encoded, ROW_COUNT, valueSize)
            .expect("decode failed");
        assert_eq!(decoded, rawData, "FSST round-trip mismatch");

        let mut decodeResults = Vec::with_capacity(VALIDATION_RUNS);
        for _ in 0..VALIDATION_RUNS {
            let start = Instant::now();
            std::hint::black_box(encoder.decode(&encoded, ROW_COUNT, valueSize).unwrap());
            let elapsed = start.elapsed().as_secs_f64();
            decodeResults.push((ROW_COUNT * valueSize) as f64 / elapsed / 1e9);
        }
        validate_metric(
            "Encoding Round-Trip",
            "FSST decompress (GB/sec)",
            decodeResults,
            FSST_DECOMPRESS_TARGET_GB_SEC,
            true,
        );
    }

    // -- ALP: f64 with 2 decimal places --
    {
        tprintln!("\n  ALP (f64 decimals):");
        let mut rawData = Vec::with_capacity(ROW_COUNT * 8);
        for i in 0..ROW_COUNT {
            rawData.extend_from_slice(&(i as f64 * 0.01 + 100.0).to_le_bytes());
        }

        let encoder = create_encoding(EncodingType::Alp);
        let encoded = encoder
            .encode(&rawData, ROW_COUNT, 8)
            .expect("encode failed");
        tprintln!(
            "    Compression ratio: {:.2}x",
            rawData.len() as f64 / encoded.len() as f64
        );
        assert!(
            encoded.len() < rawData.len(),
            "ALP should compress decimal floats"
        );

        let decoded = encoder
            .decode(&encoded, ROW_COUNT, 8)
            .expect("decode failed");
        // ALP uses epsilon-tolerance encoding, so round-trip may not be bit-exact.
        for i in 0..ROW_COUNT {
            let orig = f64::from_le_bytes(rawData[i * 8..(i + 1) * 8].try_into().unwrap());
            let dec = f64::from_le_bytes(decoded[i * 8..(i + 1) * 8].try_into().unwrap());
            assert!(
                (orig - dec).abs() < 1e-10,
                "ALP mismatch at row {}: {} vs {}",
                i,
                orig,
                dec
            );
        }

        let mut decodeResults = Vec::with_capacity(VALIDATION_RUNS);
        for _ in 0..VALIDATION_RUNS {
            let start = Instant::now();
            std::hint::black_box(encoder.decode(&encoded, ROW_COUNT, 8).unwrap());
            let elapsed = start.elapsed().as_secs_f64();
            decodeResults.push(ROW_COUNT as f64 / elapsed);
        }
        validate_metric(
            "Encoding Round-Trip",
            "ALP decode (float/sec)",
            decodeResults,
            ALP_DECODE_TARGET_FLOAT_SEC,
            true,
        );
    }

    // -- Dictionary: 10 distinct i32 values --
    {
        tprintln!("\n  Dictionary (10 distinct i32):");
        let dictVals: Vec<[u8; 4]> = (0..10u32).map(|v| (v * 1000).to_le_bytes()).collect();
        let mut rawData = Vec::with_capacity(ROW_COUNT * 4);
        for i in 0..ROW_COUNT {
            rawData.extend_from_slice(&dictVals[i % 10]);
        }

        let encoder = create_encoding(EncodingType::Dictionary);
        let encoded = encoder
            .encode(&rawData, ROW_COUNT, 4)
            .expect("encode failed");
        tprintln!(
            "    Compression ratio: {:.2}x",
            rawData.len() as f64 / encoded.len() as f64
        );

        let decoded = encoder
            .decode(&encoded, ROW_COUNT, 4)
            .expect("decode failed");
        assert_eq!(decoded, rawData, "Dictionary round-trip mismatch");

        let mut decodeResults = Vec::with_capacity(VALIDATION_RUNS);
        for _ in 0..VALIDATION_RUNS {
            let start = Instant::now();
            std::hint::black_box(encoder.decode(&encoded, ROW_COUNT, 4).unwrap());
            let elapsed = start.elapsed().as_secs_f64();
            decodeResults.push(elapsed * 1e9 / ROW_COUNT as f64);
        }
        validate_metric(
            "Encoding Round-Trip",
            "Dictionary lookup (ns/val)",
            decodeResults,
            DICTIONARY_LOOKUP_TARGET_NS,
            false,
        );
    }

    // -- RLE: runs of 1000 --
    {
        tprintln!("\n  RLE (runs of 1000):");
        let mut rawData = Vec::with_capacity(ROW_COUNT * 4);
        for i in 0..ROW_COUNT {
            rawData.extend_from_slice(&((i / 1000) as u32).to_le_bytes());
        }

        let encoder = create_encoding(EncodingType::Rle);
        let encoded = encoder
            .encode(&rawData, ROW_COUNT, 4)
            .expect("encode failed");
        tprintln!(
            "    Compression ratio: {:.2}x",
            rawData.len() as f64 / encoded.len() as f64
        );
        assert!(
            encoded.len() < rawData.len(),
            "RLE should compress repetitive data"
        );

        let decoded = encoder
            .decode(&encoded, ROW_COUNT, 4)
            .expect("decode failed");
        assert_eq!(decoded, rawData, "RLE round-trip mismatch");

        let mut decodeResults = Vec::with_capacity(VALIDATION_RUNS);
        for _ in 0..VALIDATION_RUNS {
            let start = Instant::now();
            std::hint::black_box(encoder.decode(&encoded, ROW_COUNT, 4).unwrap());
            let elapsed = start.elapsed().as_secs_f64();
            decodeResults.push(ROW_COUNT as f64 / elapsed);
        }
        validate_metric(
            "Encoding Round-Trip",
            "RLE decode (val/sec)",
            decodeResults,
            RLE_DECODE_TARGET_VAL_SEC,
            true,
        );
    }

    // -- BitPack: boolean alternating --
    {
        tprintln!("\n  BitPack (boolean):");
        let mut rawData = Vec::with_capacity(ROW_COUNT);
        for i in 0..ROW_COUNT {
            rawData.push((i % 2) as u8);
        }

        let encoder = create_encoding(EncodingType::BitPack);
        let encoded = encoder
            .encode(&rawData, ROW_COUNT, 1)
            .expect("encode failed");
        tprintln!(
            "    Compression ratio: {:.2}x",
            rawData.len() as f64 / encoded.len() as f64
        );

        let decoded = encoder
            .decode(&encoded, ROW_COUNT, 1)
            .expect("decode failed");
        assert_eq!(decoded, rawData, "BitPack round-trip mismatch");
    }

    // -- Constant: all 42 --
    {
        tprintln!("\n  Constant (all 42):");
        let val = 42u32.to_le_bytes();
        let mut rawData = Vec::with_capacity(ROW_COUNT * 4);
        for _ in 0..ROW_COUNT {
            rawData.extend_from_slice(&val);
        }

        let encoder = create_encoding(EncodingType::Constant);
        let encoded = encoder
            .encode(&rawData, ROW_COUNT, 4)
            .expect("encode failed");
        tprintln!(
            "    Encoded size: {} bytes (raw: {})",
            encoded.len(),
            rawData.len()
        );
        assert!(encoded.len() < 20, "Constant encoding should be tiny");

        let decoded = encoder
            .decode(&encoded, ROW_COUNT, 4)
            .expect("decode failed");
        assert_eq!(decoded, rawData, "Constant round-trip mismatch");
    }

    // -- Unencoded: random u32 --
    {
        tprintln!("\n  Unencoded (random u32):");
        let mut rng = rand::rng();
        let mut rawData = Vec::with_capacity(ROW_COUNT * 4);
        for _ in 0..ROW_COUNT {
            rawData.extend_from_slice(&rng.random::<u32>().to_le_bytes());
        }

        let encoder = create_encoding(EncodingType::Unencoded);
        let encoded = encoder
            .encode(&rawData, ROW_COUNT, 4)
            .expect("encode failed");
        assert_eq!(
            encoded.len(),
            rawData.len(),
            "Unencoded should be same size"
        );

        let decoded = encoder
            .decode(&encoded, ROW_COUNT, 4)
            .expect("decode failed");
        assert_eq!(decoded, rawData, "Unencoded round-trip mismatch");
    }

    // -- Edge cases --
    tprintln!("\n  Edge cases:");

    // Single row
    for encType in [
        EncodingType::FastLanes,
        EncodingType::Rle,
        EncodingType::BitPack,
        EncodingType::Constant,
        EncodingType::Dictionary,
        EncodingType::Unencoded,
    ] {
        let encoder = create_encoding(encType);
        let raw = 42u32.to_le_bytes().to_vec();
        let encoded = encoder
            .encode(&raw, 1, 4)
            .expect("single row encode failed");
        let decoded = encoder
            .decode(&encoded, 1, 4)
            .expect("single row decode failed");
        assert_eq!(
            decoded, raw,
            "single row round-trip failed for {:?}",
            encType
        );
    }
    tprintln!("    Single row: all encodings pass");

    // All zeros
    for encType in [
        EncodingType::FastLanes,
        EncodingType::Rle,
        EncodingType::BitPack,
        EncodingType::Constant,
        EncodingType::Dictionary,
        EncodingType::Unencoded,
    ] {
        let encoder = create_encoding(encType);
        let raw = vec![0u8; 100 * 4];
        let encoded = encoder
            .encode(&raw, 100, 4)
            .expect("all-zeros encode failed");
        let decoded = encoder
            .decode(&encoded, 100, 4)
            .expect("all-zeros decode failed");
        assert_eq!(
            decoded, raw,
            "all-zeros round-trip failed for {:?}",
            encType
        );
    }
    tprintln!("    All zeros: all encodings pass");

    // Max u32 values
    {
        let encoder = create_encoding(EncodingType::FastLanes);
        let mut raw = Vec::with_capacity(100 * 4);
        for _ in 0..100 {
            raw.extend_from_slice(&u32::MAX.to_le_bytes());
        }
        let encoded = encoder.encode(&raw, 100, 4).expect("max-val encode failed");
        let decoded = encoder
            .decode(&encoded, 100, 4)
            .expect("max-val decode failed");
        assert_eq!(decoded, raw, "max-val round-trip failed for FastLanes");
    }
    tprintln!("    Max u32: FastLanes pass");

    let utilAfter = take_util_snapshot();
    record_test_util("Encoding Round-Trip", utilBefore, utilAfter);
    tprintln!("\n  Round-trip correctness: ALL PASS");
}

// =============================================================================
// Test 2: Encoding Selection
// =============================================================================

#[test]
fn test_encoding_selection() {
    zyron_bench_harness::init("encoding");
    let _benchGuard = BENCHMARK_LOCK.lock().unwrap_or_else(|e| e.into_inner());

    tprintln!("\n=== Phase 1.7: Encoding Selection ===");

    let utilBefore = take_util_snapshot();

    // Constant: all identical
    {
        let val = [42u8, 0, 0, 0];
        let sample: Vec<Option<&[u8]>> = (0..1000).map(|_| Some(val.as_slice())).collect();
        let result = select_encoding(TypeId::Int32, &sample);
        assert_eq!(
            result,
            EncodingType::Constant,
            "all-identical should select Constant"
        );
        tprintln!("  All-identical -> Constant: PASS");
    }

    // Boolean -> BitPack
    {
        let t = [1u8];
        let f = [0u8];
        let sample: Vec<Option<&[u8]>> = (0..1000)
            .map(|i| {
                if i % 2 == 0 {
                    Some(t.as_slice())
                } else {
                    Some(f.as_slice())
                }
            })
            .collect();
        let result = select_encoding(TypeId::Boolean, &sample);
        assert_eq!(
            result,
            EncodingType::BitPack,
            "boolean should select BitPack"
        );
        tprintln!("  Boolean -> BitPack: PASS");
    }

    // Low-cardinality strings -> Dictionary
    {
        let vals: Vec<[u8; 4]> = (0..10u32).map(|v| v.to_le_bytes()).collect();
        let sample: Vec<Option<&[u8]>> = (0..100_000)
            .map(|i| Some(vals[i % 10].as_slice()))
            .collect();
        let result = select_encoding(TypeId::Int32, &sample);
        assert_eq!(
            result,
            EncodingType::Dictionary,
            "low-cardinality should select Dictionary"
        );
        tprintln!("  Low-cardinality (10 distinct, 100K rows) -> Dictionary: PASS");
    }

    // Sequential integers -> FastLanes
    {
        let vals: Vec<[u8; 4]> = (0..1000u32).map(|v| v.to_le_bytes()).collect();
        let sample: Vec<Option<&[u8]>> = vals.iter().map(|v| Some(v.as_slice())).collect();
        let result = select_encoding(TypeId::Int32, &sample);
        assert_eq!(
            result,
            EncodingType::FastLanes,
            "sequential integers should select FastLanes"
        );
        tprintln!("  Sequential integers -> FastLanes: PASS");
    }

    // Random floats with 2 decimal places -> ALP
    {
        let vals: Vec<[u8; 8]> = (0..1000)
            .map(|i| (i as f64 * 0.01 + 100.0).to_le_bytes())
            .collect();
        let sample: Vec<Option<&[u8]>> = vals.iter().map(|v| Some(v.as_slice())).collect();
        let result = select_encoding(TypeId::Float64, &sample);
        assert_eq!(
            result,
            EncodingType::Alp,
            "decimal floats should select ALP"
        );
        tprintln!("  Decimal floats -> ALP: PASS");
    }

    // Boolean column -> BitPack
    {
        let vals: Vec<[u8; 1]> = (0..1000).map(|i| [(i % 2) as u8]).collect();
        let sample: Vec<Option<&[u8]>> = vals.iter().map(|v| Some(v.as_slice())).collect();
        let result = select_encoding(TypeId::Boolean, &sample);
        assert_eq!(
            result,
            EncodingType::BitPack,
            "boolean column should select BitPack"
        );
        tprintln!("  Boolean column -> BitPack: PASS");
    }

    // Single-value column -> Constant
    {
        let val = [99u8, 0, 0, 0];
        let sample: Vec<Option<&[u8]>> = (0..1000).map(|_| Some(val.as_slice())).collect();
        let result = select_encoding(TypeId::Int32, &sample);
        assert_eq!(
            result,
            EncodingType::Constant,
            "single-value should select Constant"
        );
        tprintln!("  Single-value column -> Constant: PASS");
    }

    // High-entropy random bytes -> Unencoded
    {
        let mut rng = rand::rng();
        let vals: Vec<[u8; 4]> = (0..1000)
            .map(|_| rng.random::<u32>().to_le_bytes())
            .collect();
        let sample: Vec<Option<&[u8]>> = vals.iter().map(|v| Some(v.as_slice())).collect();
        let result = select_encoding(TypeId::Int32, &sample);
        // High cardinality (1000 distinct out of 1000) with random data.
        // FastLanes trial-encode may or may not compress random data.
        // Accept FastLanes or Unencoded since both are valid for random integers.
        assert!(
            result == EncodingType::Unencoded || result == EncodingType::FastLanes,
            "high-entropy should select Unencoded or FastLanes, got {:?}",
            result
        );
        tprintln!("  High-entropy random -> {:?}: PASS", result);
    }

    // Empty sample -> Unencoded
    {
        let result = select_encoding(TypeId::Int32, &[]);
        assert_eq!(
            result,
            EncodingType::Unencoded,
            "empty sample should select Unencoded"
        );
        tprintln!("  Empty sample -> Unencoded: PASS");
    }

    // Performance: time encoding selection
    let mut selectResults = Vec::with_capacity(VALIDATION_RUNS);
    let selectVals: Vec<[u8; 4]> = (0..1024u32).map(|v| v.to_le_bytes()).collect();
    let selectSample: Vec<Option<&[u8]>> = selectVals.iter().map(|v| Some(v.as_slice())).collect();

    for _ in 0..VALIDATION_RUNS {
        let start = Instant::now();
        for _ in 0..1000 {
            let _ = select_encoding(TypeId::Int32, &selectSample);
        }
        let elapsed = start.elapsed().as_secs_f64();
        selectResults.push(elapsed * 1e6 / 1000.0);
    }

    validate_metric(
        "Encoding Selection",
        "Encoding select (us/col)",
        selectResults,
        ENCODING_SELECT_TARGET_US,
        false,
    );

    let utilAfter = take_util_snapshot();
    record_test_util("Encoding Selection", utilBefore, utilAfter);
    tprintln!("\n  Encoding selection: ALL PASS");
}

// =============================================================================
// Test 3: Query-on-Compressed
// =============================================================================

#[test]
fn test_query_on_compressed() {
    zyron_bench_harness::init("encoding");
    let _benchGuard = BENCHMARK_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    const ROW_COUNT: usize = 100_000;

    tprintln!("\n=== Phase 1.7: Query-on-Compressed ===");
    tprintln!("Rows: {}", ROW_COUNT);

    let utilBefore = take_util_snapshot();

    // Helper: check that compressed eval matches decoded eval
    fn verify_predicate_match(
        encoder: &dyn zyron_storage::encoding::Encoding,
        rawData: &[u8],
        encoded: &[u8],
        rowCount: usize,
        valueSize: usize,
        predicate: &Predicate,
        label: &str,
    ) {
        let compressedBitmask = encoder
            .eval_predicate(encoded, rowCount, valueSize, predicate)
            .expect("compressed eval failed");
        let decodedBitmask = eval_predicate_on_raw(rawData, rowCount, valueSize, predicate)
            .expect("raw eval failed");
        assert_eq!(
            compressedBitmask, decodedBitmask,
            "bitmask mismatch for {}",
            label
        );
    }

    // -- Dictionary: equality --
    {
        let dictVals: Vec<[u8; 4]> = (0..10u32).map(|v| (v * 100).to_le_bytes()).collect();
        let mut rawData = Vec::with_capacity(ROW_COUNT * 4);
        for i in 0..ROW_COUNT {
            rawData.extend_from_slice(&dictVals[i % 10]);
        }

        let encoder = create_encoding(EncodingType::Dictionary);
        let encoded = encoder.encode(&rawData, ROW_COUNT, 4).unwrap();

        let target = 300u32.to_le_bytes();
        verify_predicate_match(
            encoder.as_ref(),
            &rawData,
            &encoded,
            ROW_COUNT,
            4,
            &Predicate::Equality(&target),
            "Dictionary equality",
        );
        tprintln!("  Dictionary equality: PASS");
    }

    // -- RLE: range --
    {
        let mut rawData = Vec::with_capacity(ROW_COUNT * 4);
        for i in 0..ROW_COUNT {
            rawData.extend_from_slice(&((i / 1000) as u32).to_le_bytes());
        }

        let encoder = create_encoding(EncodingType::Rle);
        let encoded = encoder.encode(&rawData, ROW_COUNT, 4).unwrap();

        let lo = 10u32.to_le_bytes();
        let hi = 20u32.to_le_bytes();
        verify_predicate_match(
            encoder.as_ref(),
            &rawData,
            &encoded,
            ROW_COUNT,
            4,
            &Predicate::Range {
                low: Some(&lo),
                high: Some(&hi),
            },
            "RLE range",
        );
        tprintln!("  RLE range: PASS");
    }

    // -- BitPack: equality (bitmask) --
    {
        let mut rawData = Vec::with_capacity(ROW_COUNT);
        for i in 0..ROW_COUNT {
            rawData.push((i % 2) as u8);
        }

        let encoder = create_encoding(EncodingType::BitPack);
        let encoded = encoder.encode(&rawData, ROW_COUNT, 1).unwrap();

        let target = [1u8];
        verify_predicate_match(
            encoder.as_ref(),
            &rawData,
            &encoded,
            ROW_COUNT,
            1,
            &Predicate::Equality(&target),
            "BitPack equality",
        );
        tprintln!("  BitPack equality: PASS");
    }

    // -- Constant: equality match and miss --
    {
        let val = 42u32.to_le_bytes();
        let mut rawData = Vec::with_capacity(ROW_COUNT * 4);
        for _ in 0..ROW_COUNT {
            rawData.extend_from_slice(&val);
        }

        let encoder = create_encoding(EncodingType::Constant);
        let encoded = encoder.encode(&rawData, ROW_COUNT, 4).unwrap();

        let matchTarget = 42u32.to_le_bytes();
        verify_predicate_match(
            encoder.as_ref(),
            &rawData,
            &encoded,
            ROW_COUNT,
            4,
            &Predicate::Equality(&matchTarget),
            "Constant equality match",
        );

        let missTarget = 99u32.to_le_bytes();
        let missBitmask = encoder
            .eval_predicate(&encoded, ROW_COUNT, 4, &Predicate::Equality(&missTarget))
            .unwrap();
        assert!(
            missBitmask.iter().all(|&b| b == 0),
            "Constant miss should return all zeros"
        );
        tprintln!("  Constant equality: PASS");
    }

    // -- FastLanes: range predicate speedup measurement --
    {
        let mut rawData = Vec::with_capacity(ROW_COUNT * 4);
        for i in 0..ROW_COUNT {
            rawData.extend_from_slice(&(i as u32).to_le_bytes());
        }

        let encoder = create_encoding(EncodingType::FastLanes);
        let encoded = encoder.encode(&rawData, ROW_COUNT, 4).unwrap();

        let lo = 40_000u32.to_le_bytes();
        let hi = 49_999u32.to_le_bytes();
        let rangePred = Predicate::Range {
            low: Some(&lo),
            high: Some(&hi),
        };

        // Verify correctness
        verify_predicate_match(
            encoder.as_ref(),
            &rawData,
            &encoded,
            ROW_COUNT,
            4,
            &rangePred,
            "FastLanes range",
        );
        tprintln!("  FastLanes range: PASS");

        // Measure speedup: compressed eval vs decode+eval
        let mut speedupResults = Vec::with_capacity(VALIDATION_RUNS);
        for _ in 0..VALIDATION_RUNS {
            let lo2 = 40_000u32.to_le_bytes();
            let hi2 = 49_999u32.to_le_bytes();
            let pred = Predicate::Range {
                low: Some(&lo2),
                high: Some(&hi2),
            };

            let startRaw = Instant::now();
            for _ in 0..100 {
                let _ = eval_predicate_on_raw(&rawData, ROW_COUNT, 4, &pred).unwrap();
            }
            let rawTime = startRaw.elapsed().as_secs_f64();

            let startComp = Instant::now();
            for _ in 0..100 {
                let _ = encoder
                    .eval_predicate(&encoded, ROW_COUNT, 4, &pred)
                    .unwrap();
            }
            let compTime = startComp.elapsed().as_secs_f64();

            let speedup = rawTime / compTime;
            speedupResults.push(speedup);
        }

        validate_metric(
            "Query-on-Compressed",
            "Compressed eval speedup (x)",
            speedupResults,
            COMPRESSED_EVAL_SPEEDUP_TARGET,
            true,
        );
    }

    // -- ALP: range --
    {
        let mut rawData = Vec::with_capacity(ROW_COUNT * 8);
        for i in 0..ROW_COUNT {
            rawData.extend_from_slice(&(i as f64 * 0.01 + 100.0).to_le_bytes());
        }

        let encoder = create_encoding(EncodingType::Alp);
        let encoded = encoder.encode(&rawData, ROW_COUNT, 8).unwrap();

        let lo = 500.0f64.to_le_bytes();
        let hi = 600.0f64.to_le_bytes();
        verify_predicate_match(
            encoder.as_ref(),
            &rawData,
            &encoded,
            ROW_COUNT,
            8,
            &Predicate::Range {
                low: Some(&lo),
                high: Some(&hi),
            },
            "ALP range",
        );
        tprintln!("  ALP range: PASS");
    }

    // -- FSST: equality --
    {
        let valueSize = 32;
        let mut rawData = Vec::with_capacity(ROW_COUNT * valueSize);
        for i in 0..ROW_COUNT {
            let base = format!("key_{:05}_padding_abcdefgh_", i % 1000);
            let mut val = [0u8; 32];
            let bytes = base.as_bytes();
            let copyLen = bytes.len().min(32);
            val[..copyLen].copy_from_slice(&bytes[..copyLen]);
            rawData.extend_from_slice(&val);
        }

        let encoder = create_encoding(EncodingType::Fsst);
        let encoded = encoder.encode(&rawData, ROW_COUNT, valueSize).unwrap();

        let mut target = [0u8; 32];
        let targetStr = format!("key_{:05}_padding_abcdefgh_", 500);
        let tBytes = targetStr.as_bytes();
        let tCopyLen = tBytes.len().min(32);
        target[..tCopyLen].copy_from_slice(&tBytes[..tCopyLen]);

        verify_predicate_match(
            encoder.as_ref(),
            &rawData,
            &encoded,
            ROW_COUNT,
            valueSize,
            &Predicate::Equality(&target),
            "FSST equality",
        );
        tprintln!("  FSST equality: PASS");
    }

    let utilAfter = take_util_snapshot();
    record_test_util("Query-on-Compressed", utilBefore, utilAfter);
    tprintln!("\n  Query-on-compressed: ALL PASS");
}
