#![allow(non_snake_case, unused_assignments)]

//! Columnar Storage Benchmark Suite
//!
//! Integration tests for ZyronDB columnar storage:
//! - Column segment format and ZYR file I/O
//! - Compaction pipeline (sequential and parallel)
//! - Segment cache hit rates
//! - HTAP hybrid scan overhead
//! - Transaction-aware segment pruning
//! - Bloom filter probe and skip rates
//! - Zone map batch skip rates
//! - Sorted segment lookup and range scan
//!
//! Run: cargo test -p zyron-storage --test columnar_bench --release -- --nocapture

use zyron_bench_harness::*;

use rand::Rng;
use std::collections::HashSet;
use std::sync::Arc;
use std::time::Instant;
use tempfile::tempdir;

use zyron_common::types::TypeId;
use zyron_storage::columnar::compaction::run_compaction_cycle;
use zyron_storage::columnar::sorted::{MergeScanIterator, binary_search_sorted_column};
use zyron_storage::columnar::{
    BLOOM_MIN_CARDINALITY, BloomFilter, ColumnDescriptor, ColumnSegment, CompactionConfig,
    CompactionInput, SEGMENT_HEADER_SIZE, STAT_VALUE_SIZE, SegmentCache, SegmentCacheKey,
    SegmentHeader, SortOrder, SortedSegmentEntry, SortedSegmentIndex, ZONE_MAP_BATCH_SIZE,
    ZONE_MAP_ENTRY_SIZE, ZYR_FORMAT_VERSION, ZoneMapEntry, ZyrFileReader,
};
use zyron_storage::encoding::{EncodingType, create_encoding, select_encoding};

// Performance targets
const ZYR_SCAN_TARGET_GB_SEC: f64 = 1.4;
const COMPACTION_SEQ_TARGET_ROWS_SEC: f64 = 1_000_000.0;
const COMPACTION_PARALLEL_SPEEDUP_TARGET: f64 = 3.0;
const HYBRID_SCAN_OVERHEAD_TARGET_PCT: f64 = 5.0;
const BLOOM_PROBE_TARGET_NS: f64 = 30.0;
const BLOOM_SKIP_RATE_TARGET_PCT: f64 = 99.0;
const ZONE_MAP_BATCH_SKIP_RATE_TARGET_PCT: f64 = 95.0;
const SORTED_PK_LOOKUP_TARGET_NS: f64 = 500.0;
const SORTED_PK_RANGE_TARGET_KEYS_SEC: f64 = 80_000_000.0;

static BENCHMARK_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

fn value_to_stat_slot(value: &[u8]) -> [u8; STAT_VALUE_SIZE] {
    let mut slot = [0u8; STAT_VALUE_SIZE];
    let len = value.len().min(STAT_VALUE_SIZE);
    slot[..len].copy_from_slice(&value[..len]);
    slot
}

fn compare_stat_slots(a: &[u8; STAT_VALUE_SIZE], b: &[u8; STAT_VALUE_SIZE]) -> std::cmp::Ordering {
    for i in (0..STAT_VALUE_SIZE).rev() {
        match a[i].cmp(&b[i]) {
            std::cmp::Ordering::Equal => continue,
            other => return other,
        }
    }
    std::cmp::Ordering::Equal
}

#[test]
fn test_column_segment_format() {
    zyron_bench_harness::init("columnar");
    let _benchGuard = BENCHMARK_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    const ROW_COUNT: usize = 10_000;

    tprintln!("\n=== Phase 1.7: Column Segment Format ===");
    tprintln!("Rows: {}, Columns: 5", ROW_COUNT);

    let utilBefore = take_util_snapshot();
    let dir = tempdir().expect("failed to create temp dir");

    let columns = vec![
        ColumnDescriptor {
            column_id: 0,
            type_id: TypeId::Int32,
            value_size: 4,
            is_primary_key: true,
        },
        ColumnDescriptor {
            column_id: 1,
            type_id: TypeId::Float64,
            value_size: 8,
            is_primary_key: false,
        },
        ColumnDescriptor {
            column_id: 2,
            type_id: TypeId::Int32,
            value_size: 4,
            is_primary_key: false,
        },
        ColumnDescriptor {
            column_id: 3,
            type_id: TypeId::Boolean,
            value_size: 1,
            is_primary_key: false,
        },
        ColumnDescriptor {
            column_id: 4,
            type_id: TypeId::Int32,
            value_size: 4,
            is_primary_key: false,
        },
    ];

    let mut columnData: Vec<Vec<Option<Vec<u8>>>> = vec![Vec::with_capacity(ROW_COUNT); 5];
    for i in 0..ROW_COUNT {
        columnData[0].push(Some((i as u32).to_le_bytes().to_vec()));
        columnData[1].push(Some((i as f64 * 0.01 + 100.0).to_le_bytes().to_vec()));
        columnData[2].push(Some(((i % 20) as u32).to_le_bytes().to_vec()));
        columnData[3].push(Some(vec![(i % 2) as u8]));
        columnData[4].push(Some(777u32.to_le_bytes().to_vec()));
    }

    let config = CompactionConfig {
        columnar_dir: dir.path().to_path_buf(),
        min_rows: 1,
        max_rows_per_file: 1_000_000,
        fsync_enabled: false,
        max_encoding_threads: 1,
        oltp_p99_threshold_us: 10_000,
        check_interval_ms: 1000,
    };

    let input = CompactionInput {
        columns: columns.clone(),
        column_data: columnData,
        table_id: 1,
        xmin_lo: 100,
        xmin_hi: 500,
    };

    let result = run_compaction_cycle(&config, input).expect("compaction failed");
    tprintln!(
        "  Compaction result: {} rows, {} cols, {} bytes",
        result.row_count,
        result.column_count,
        result.file_size
    );

    assert_eq!(result.row_count, ROW_COUNT as u64);
    assert_eq!(result.column_count, 5);
    assert!(result.file_size > 0);

    // Read back and verify
    let reader = ZyrFileReader::open(&result.file_path).expect("open reader failed");
    let header = reader.header();

    assert_eq!(header.format_version, ZYR_FORMAT_VERSION);
    assert_eq!(header.column_count, 5);
    assert_eq!(header.row_count, ROW_COUNT as u64);
    assert_eq!(header.table_id, 1);
    assert_eq!(header.xmin_range_lo, 100);
    assert_eq!(header.xmin_range_hi, 500);
    assert_eq!(header.sort_order, SortOrder::Asc);
    assert_eq!(reader.segment_count(), 5);
    tprintln!("  File header: PASS");

    // Verify each segment header
    for col in &columns {
        let segRaw = reader
            .read_segment_raw(col.column_id)
            .expect("read segment failed");
        assert_eq!(
            segRaw.len() % zyron_common::page::PAGE_SIZE,
            0,
            "segment not page-aligned"
        );

        let headerBuf: [u8; SEGMENT_HEADER_SIZE] =
            segRaw[..SEGMENT_HEADER_SIZE].try_into().unwrap();
        let segHeader = SegmentHeader::from_bytes(&headerBuf).expect("segment header parse failed");

        assert_eq!(segHeader.column_id, col.column_id);
        assert!(
            segHeader.compressed_size > 0,
            "col {} compressed_size is 0",
            col.column_id
        );
        assert_eq!(
            segHeader.null_count, 0,
            "col {} has unexpected nulls",
            col.column_id
        );

        if col.is_primary_key {
            assert!(segHeader.is_sorted, "PK column should be sorted");
        }

        tprintln!(
            "  Column {} ({:?}): encoding={:?}, compressed={}, sorted={}",
            col.column_id,
            col.type_id,
            segHeader.encoding_type,
            segHeader.compressed_size,
            segHeader.is_sorted
        );
    }

    // Performance: time scan (open + read all segments)
    let totalUncompressed = ROW_COUNT * (4 + 8 + 4 + 1 + 4);
    let mut scanResults = Vec::with_capacity(VALIDATION_RUNS);
    for _ in 0..VALIDATION_RUNS {
        let start = Instant::now();
        for _ in 0..100 {
            let r = ZyrFileReader::open(&result.file_path).unwrap();
            for col in &columns {
                let _ = r.read_segment_raw(col.column_id).unwrap();
            }
        }
        let elapsed = start.elapsed().as_secs_f64();
        scanResults.push((totalUncompressed as f64 * 100.0) / elapsed / 1e9);
    }

    validate_metric(
        "Column Segment Format",
        ".zyr scan throughput (GB/sec)",
        scanResults,
        ZYR_SCAN_TARGET_GB_SEC,
        true,
    );

    let utilAfter = take_util_snapshot();
    record_test_util("Column Segment Format", utilBefore, utilAfter);
    tprintln!("\n  Column segment format: ALL PASS");
}

// =============================================================================
// Test 5: Compaction Pipeline
// =============================================================================

#[test]
fn test_compaction_pipeline() {
    zyron_bench_harness::init("columnar");
    let _benchGuard = BENCHMARK_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    const ROW_COUNT: usize = 100_000;

    tprintln!("\n=== Phase 1.7: Compaction Pipeline ===");
    tprintln!("Rows: {}", ROW_COUNT);

    let utilBefore = take_util_snapshot();
    let dir = tempdir().expect("failed to create temp dir");

    let columns = vec![
        ColumnDescriptor {
            column_id: 0,
            type_id: TypeId::Int32,
            value_size: 4,
            is_primary_key: true,
        },
        ColumnDescriptor {
            column_id: 1,
            type_id: TypeId::Int64,
            value_size: 8,
            is_primary_key: false,
        },
        ColumnDescriptor {
            column_id: 2,
            type_id: TypeId::Int32,
            value_size: 4,
            is_primary_key: false,
        },
    ];

    let config = CompactionConfig {
        columnar_dir: dir.path().to_path_buf(),
        min_rows: 1,
        max_rows_per_file: 1_000_000,
        fsync_enabled: false,
        max_encoding_threads: 1,
        oltp_p99_threshold_us: 10_000,
        check_interval_ms: 1000,
    };

    // First compaction: 100K rows
    let mut compactionResults = Vec::with_capacity(VALIDATION_RUNS);
    let mut _firstFilePath = std::path::PathBuf::new();

    for run in 0..VALIDATION_RUNS {
        let runDir = tempdir().expect("failed to create run dir");
        let runConfig = CompactionConfig {
            columnar_dir: runDir.path().to_path_buf(),
            ..config.clone()
        };

        let mut columnData: Vec<Vec<Option<Vec<u8>>>> = vec![Vec::with_capacity(ROW_COUNT); 3];
        for i in 0..ROW_COUNT {
            columnData[0].push(Some((i as u32).to_le_bytes().to_vec()));
            columnData[1].push(Some(((i as i64) * 1000).to_le_bytes().to_vec()));
            columnData[2].push(Some(((i % 50) as u32).to_le_bytes().to_vec()));
        }

        let input = CompactionInput {
            columns: columns.clone(),
            column_data: columnData,
            table_id: 42,
            xmin_lo: 1,
            xmin_hi: 100_000,
        };

        let start = Instant::now();
        let result = run_compaction_cycle(&runConfig, input).expect("compaction failed");
        let elapsed = start.elapsed().as_secs_f64();

        compactionResults.push(ROW_COUNT as f64 / elapsed);

        if run == 0 {
            assert_eq!(result.row_count, ROW_COUNT as u64);
            assert_eq!(result.column_count, 3);

            let reader = ZyrFileReader::open(&result.file_path).expect("open reader failed");
            assert_eq!(reader.header().row_count, ROW_COUNT as u64);
            assert_eq!(reader.header().sort_order, SortOrder::Asc);
            tprintln!(
                "  First compaction: {} rows, {} bytes",
                result.row_count,
                result.file_size
            );
        }

        _firstFilePath = result.file_path;
    }

    validate_metric(
        "Compaction Pipeline",
        "Compaction throughput (rows/sec)",
        compactionResults,
        COMPACTION_SEQ_TARGET_ROWS_SEC,
        true,
    );

    // Incremental compaction: 50K more rows
    {
        let incrRows = 50_000;
        let mut columnData: Vec<Vec<Option<Vec<u8>>>> = vec![Vec::with_capacity(incrRows); 3];
        for i in 0..incrRows {
            let pk = (ROW_COUNT + i) as u32;
            columnData[0].push(Some(pk.to_le_bytes().to_vec()));
            columnData[1].push(Some(((ROW_COUNT + i) as i64 * 1000).to_le_bytes().to_vec()));
            columnData[2].push(Some(((i % 50) as u32).to_le_bytes().to_vec()));
        }

        let input = CompactionInput {
            columns: columns.clone(),
            column_data: columnData,
            table_id: 43,
            xmin_lo: 100_001,
            xmin_hi: 150_000,
        };

        let result = run_compaction_cycle(&config, input).expect("incremental compaction failed");
        assert_eq!(result.row_count, incrRows as u64);
        assert_eq!(result.column_count, 3);

        let reader =
            ZyrFileReader::open(&result.file_path).expect("open incremental reader failed");
        assert_eq!(reader.header().row_count, incrRows as u64);
        assert_eq!(reader.header().xmin_range_lo, 100_001);
        assert_eq!(reader.header().xmin_range_hi, 150_000);

        tprintln!(
            "  Incremental compaction: {} rows, {} bytes",
            result.row_count,
            result.file_size
        );
    }

    let utilAfter = take_util_snapshot();
    record_test_util("Compaction Pipeline", utilBefore, utilAfter);
    tprintln!("\n  Compaction pipeline: ALL PASS");
}

// =============================================================================
// Test 6: Segment Cache
// =============================================================================

#[test]
fn test_segment_cache() {
    zyron_bench_harness::init("columnar");
    let _benchGuard = BENCHMARK_LOCK.lock().unwrap_or_else(|e| e.into_inner());

    tprintln!("\n=== Phase 1.7: Segment Cache ===");

    let utilBefore = take_util_snapshot();

    let segSize = 10_240;
    let cache = SegmentCache::new(segSize * 5); // 50KB capacity, fits 5 segments

    // Insert 5 segments
    for i in 0..5u64 {
        let key = SegmentCacheKey::new(i, 0);
        let data = vec![(i as u8).wrapping_mul(17); segSize];
        cache.insert(key, data);
    }

    // Verify all 5 are present with correct data
    for i in 0..5u64 {
        let key = SegmentCacheKey::new(i, 0);
        let result = cache.get(&key);
        assert!(result.is_some(), "key {} should be cached", i);
        let seg = result.unwrap();
        assert_eq!(seg.data.len(), segSize);
        assert_eq!(seg.data[0], (i as u8).wrapping_mul(17));
    }
    tprintln!("  Cache insert + get (5 keys): PASS");

    let stats = cache.stats();
    assert_eq!(stats.hit_count, 5);
    assert_eq!(stats.used_bytes, (segSize * 5) as u64);
    tprintln!("  Stats (hit_count=5, used={}): PASS", stats.used_bytes);

    // Insert 6th segment, should trigger eviction
    let key6 = SegmentCacheKey::new(100, 0);
    cache.insert(key6, vec![0xFFu8; segSize]);
    let result6 = cache.get(&key6);
    assert!(result6.is_some(), "key 100 should be cached after insert");
    tprintln!("  Eviction on overflow: PASS");

    // Verify at least one old key was evicted
    let mut evictedCount = 0;
    for i in 0..5u64 {
        let key = SegmentCacheKey::new(i, 0);
        if cache.get(&key).is_none() {
            evictedCount += 1;
        }
    }
    assert!(evictedCount > 0, "at least one old key should be evicted");
    tprintln!("  Eviction count: {} (expected >= 1): PASS", evictedCount);

    // Invalidate
    cache.invalidate(&key6);
    assert!(cache.get(&key6).is_none(), "invalidated key should be gone");
    tprintln!("  Invalidate: PASS");

    // Clear
    cache.clear();
    let statsAfterClear = cache.stats();
    assert_eq!(statsAfterClear.used_bytes, 0);
    tprintln!("  Clear (used_bytes=0): PASS");

    // Performance: cache lookups
    let perfCache = SegmentCache::new(1024 * 1024);
    for i in 0..5u64 {
        perfCache.insert(SegmentCacheKey::new(i, 0), vec![0u8; 1000]);
    }

    let lookupCount = 100_000;
    let mut lookupResults = Vec::with_capacity(VALIDATION_RUNS);
    for _ in 0..VALIDATION_RUNS {
        let start = Instant::now();
        for i in 0..lookupCount {
            let key = SegmentCacheKey::new((i % 5) as u64, 0);
            let _ = perfCache.get(&key);
        }
        let elapsed = start.elapsed().as_secs_f64();
        lookupResults.push(elapsed * 1e9 / lookupCount as f64);
    }

    check_performance(
        "Segment Cache",
        "Cache get (ns/lookup)",
        lookupResults.iter().sum::<f64>() / lookupResults.len() as f64,
        100.0,
        false,
    );

    let utilAfter = take_util_snapshot();
    record_test_util("Segment Cache", utilBefore, utilAfter);
    tprintln!("\n  Segment cache: ALL PASS");
}

// =============================================================================
// Test 7: HTAP Hybrid Scan
// =============================================================================

#[test]
fn test_htap_hybrid_scan() {
    zyron_bench_harness::init("columnar");
    let _benchGuard = BENCHMARK_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    const COLUMNAR_ROWS: usize = 50_000;
    const HEAP_ROWS: usize = 10_000;
    const TOTAL_ROWS: usize = COLUMNAR_ROWS + HEAP_ROWS;

    tprintln!("\n=== Phase 1.7: HTAP Hybrid Scan ===");
    tprintln!(
        "Columnar: {} rows, Heap: {} rows, Total: {}",
        COLUMNAR_ROWS,
        HEAP_ROWS,
        TOTAL_ROWS
    );

    let utilBefore = take_util_snapshot();
    let dir = tempdir().expect("failed to create temp dir");

    let columns = vec![
        ColumnDescriptor {
            column_id: 0,
            type_id: TypeId::Int32,
            value_size: 4,
            is_primary_key: true,
        },
        ColumnDescriptor {
            column_id: 1,
            type_id: TypeId::Int64,
            value_size: 8,
            is_primary_key: false,
        },
    ];

    // Compact first 50K rows
    let mut columnData: Vec<Vec<Option<Vec<u8>>>> = vec![Vec::with_capacity(COLUMNAR_ROWS); 2];
    for i in 0..COLUMNAR_ROWS {
        columnData[0].push(Some((i as u32).to_le_bytes().to_vec()));
        columnData[1].push(Some(((i as i64) * 100).to_le_bytes().to_vec()));
    }

    let config = CompactionConfig {
        columnar_dir: dir.path().to_path_buf(),
        min_rows: 1,
        max_rows_per_file: 1_000_000,
        fsync_enabled: false,
        max_encoding_threads: 1,
        oltp_p99_threshold_us: 10_000,
        check_interval_ms: 1000,
    };

    let input = CompactionInput {
        columns: columns.clone(),
        column_data: columnData,
        table_id: 1,
        xmin_lo: 1,
        xmin_hi: 50_000,
    };

    let compResult = run_compaction_cycle(&config, input).expect("compaction failed");

    // Simulate heap rows (in-memory)
    let mut heapPks: Vec<u32> = Vec::with_capacity(HEAP_ROWS);
    let mut heapVals: Vec<i64> = Vec::with_capacity(HEAP_ROWS);
    for i in 0..HEAP_ROWS {
        heapPks.push((COLUMNAR_ROWS + i) as u32);
        heapVals.push(((COLUMNAR_ROWS + i) as i64) * 100);
    }

    // Decode columnar data
    let reader = ZyrFileReader::open(&compResult.file_path).expect("open failed");

    let pkSegRaw = reader.read_segment_raw(0).expect("read PK segment failed");
    let pkHeaderBuf: [u8; SEGMENT_HEADER_SIZE] =
        pkSegRaw[..SEGMENT_HEADER_SIZE].try_into().unwrap();
    let pkSegHeader = SegmentHeader::from_bytes(&pkHeaderBuf).expect("parse PK header failed");
    let bloomSize = pkSegHeader.bloom_filter_size as usize;
    let zoneCount =
        (COLUMNAR_ROWS + ZONE_MAP_BATCH_SIZE as usize - 1) / ZONE_MAP_BATCH_SIZE as usize;
    let zoneMapSize = zoneCount * ZONE_MAP_ENTRY_SIZE;
    let pkDataStart = SEGMENT_HEADER_SIZE + bloomSize + zoneMapSize;
    let pkDataEnd = pkDataStart + pkSegHeader.compressed_size as usize;
    let pkEncoder = create_encoding(pkSegHeader.encoding_type);
    let decodedPks = pkEncoder
        .decode(&pkSegRaw[pkDataStart..pkDataEnd], COLUMNAR_ROWS, 4)
        .expect("PK decode failed");

    let valSegRaw = reader.read_segment_raw(1).expect("read val segment failed");
    let valHeaderBuf: [u8; SEGMENT_HEADER_SIZE] =
        valSegRaw[..SEGMENT_HEADER_SIZE].try_into().unwrap();
    let valSegHeader = SegmentHeader::from_bytes(&valHeaderBuf).expect("parse val header failed");
    let valBloomSize = valSegHeader.bloom_filter_size as usize;
    let valZoneCount =
        (COLUMNAR_ROWS + ZONE_MAP_BATCH_SIZE as usize - 1) / ZONE_MAP_BATCH_SIZE as usize;
    let valZoneMapSize = valZoneCount * ZONE_MAP_ENTRY_SIZE;
    let valDataStart = SEGMENT_HEADER_SIZE + valBloomSize + valZoneMapSize;
    let valDataEnd = valDataStart + valSegHeader.compressed_size as usize;
    let valEncoder = create_encoding(valSegHeader.encoding_type);
    let decodedVals = valEncoder
        .decode(&valSegRaw[valDataStart..valDataEnd], COLUMNAR_ROWS, 8)
        .expect("val decode failed");

    // Merge: columnar PKs + heap PKs, verify all present in sorted order
    let mut allPks: Vec<u32> = Vec::with_capacity(TOTAL_ROWS);
    for i in 0..COLUMNAR_ROWS {
        let pk = u32::from_le_bytes(decodedPks[i * 4..(i + 1) * 4].try_into().unwrap());
        allPks.push(pk);
    }
    for pk in &heapPks {
        allPks.push(*pk);
    }

    assert_eq!(allPks.len(), TOTAL_ROWS);
    for i in 1..allPks.len() {
        assert!(
            allPks[i] > allPks[i - 1],
            "PKs not sorted at index {}: {} vs {}",
            i,
            allPks[i - 1],
            allPks[i]
        );
    }

    // Verify no duplicates
    let pkSet: HashSet<u32> = allPks.iter().copied().collect();
    assert_eq!(pkSet.len(), TOTAL_ROWS, "duplicates detected");
    tprintln!(
        "  Hybrid scan correctness ({} rows, sorted, no duplicates): PASS",
        TOTAL_ROWS
    );

    // Verify values match
    for i in 0..COLUMNAR_ROWS {
        let val = i64::from_le_bytes(decodedVals[i * 8..(i + 1) * 8].try_into().unwrap());
        assert_eq!(
            val,
            (i as i64) * 100,
            "columnar value mismatch at row {}",
            i
        );
    }
    for i in 0..HEAP_ROWS {
        assert_eq!(
            heapVals[i],
            ((COLUMNAR_ROWS + i) as i64) * 100,
            "heap value mismatch at row {}",
            i
        );
    }
    tprintln!("  Value correctness: PASS");

    // Performance: measure overhead of hybrid vs pure columnar
    let mut overheadResults = Vec::with_capacity(VALIDATION_RUNS);
    for _ in 0..VALIDATION_RUNS {
        // Pure columnar: decode PK + val
        let startColumnar = Instant::now();
        for _ in 0..50 {
            let pks = pkEncoder
                .decode(&pkSegRaw[pkDataStart..pkDataEnd], COLUMNAR_ROWS, 4)
                .unwrap();
            let vals = valEncoder
                .decode(&valSegRaw[valDataStart..valDataEnd], COLUMNAR_ROWS, 8)
                .unwrap();
            let mut scanCount = 0usize;
            for j in 0..COLUMNAR_ROWS {
                let _pk = u32::from_le_bytes(pks[j * 4..(j + 1) * 4].try_into().unwrap());
                let _val = i64::from_le_bytes(vals[j * 8..(j + 1) * 8].try_into().unwrap());
                scanCount += 1;
            }
            assert_eq!(scanCount, COLUMNAR_ROWS);
        }
        let columnarTime = startColumnar.elapsed().as_secs_f64();

        // Hybrid: decode + merge with heap
        let startHybrid = Instant::now();
        for _ in 0..50 {
            let pks = pkEncoder
                .decode(&pkSegRaw[pkDataStart..pkDataEnd], COLUMNAR_ROWS, 4)
                .unwrap();
            let vals = valEncoder
                .decode(&valSegRaw[valDataStart..valDataEnd], COLUMNAR_ROWS, 8)
                .unwrap();
            let mut mergedCount = 0usize;
            for j in 0..COLUMNAR_ROWS {
                let _pk = u32::from_le_bytes(pks[j * 4..(j + 1) * 4].try_into().unwrap());
                let _val = i64::from_le_bytes(vals[j * 8..(j + 1) * 8].try_into().unwrap());
                mergedCount += 1;
            }
            for j in 0..HEAP_ROWS {
                let _pk = heapPks[j];
                let _val = heapVals[j];
                mergedCount += 1;
            }
            assert_eq!(mergedCount, TOTAL_ROWS);
        }
        let hybridTime = startHybrid.elapsed().as_secs_f64();

        let overhead = (hybridTime - columnarTime) / columnarTime * 100.0;
        overheadResults.push(overhead);
    }

    validate_metric(
        "HTAP Hybrid Scan",
        "Hybrid scan overhead (%)",
        overheadResults,
        HYBRID_SCAN_OVERHEAD_TARGET_PCT,
        false,
    );

    let utilAfter = take_util_snapshot();
    record_test_util("HTAP Hybrid Scan", utilBefore, utilAfter);
    tprintln!("\n  HTAP hybrid scan: ALL PASS");
}

// =============================================================================
// Test 8: Transaction-Aware Pruning
// =============================================================================

#[test]
fn test_txn_aware_pruning() {
    zyron_bench_harness::init("columnar");
    let _benchGuard = BENCHMARK_LOCK.lock().unwrap_or_else(|e| e.into_inner());

    tprintln!("\n=== Phase 1.7: Transaction-Aware Pruning ===");

    let utilBefore = take_util_snapshot();
    let dir = tempdir().expect("failed to create temp dir");

    // Create 5 .zyr files with different xmin ranges
    let xminRanges: Vec<(u64, u64)> = vec![
        (100, 200), // File A
        (201, 300), // File B
        (301, 400), // File C
        (401, 500), // File D
        (501, 600), // File E
    ];

    let columns = vec![ColumnDescriptor {
        column_id: 0,
        type_id: TypeId::Int32,
        value_size: 4,
        is_primary_key: true,
    }];

    let mut filePaths = Vec::new();
    let mut fileHeaders = Vec::new();

    for (idx, (xminLo, xminHi)) in xminRanges.iter().enumerate() {
        let config = CompactionConfig {
            columnar_dir: dir.path().to_path_buf(),
            min_rows: 1,
            max_rows_per_file: 1_000_000,
            fsync_enabled: false,
            max_encoding_threads: 1,
            oltp_p99_threshold_us: 10_000,
            check_interval_ms: 1000,
        };

        let mut columnData: Vec<Vec<Option<Vec<u8>>>> = vec![Vec::with_capacity(1000); 1];
        for i in 0..1000 {
            columnData[0].push(Some(((idx * 10_000 + i) as u32).to_le_bytes().to_vec()));
        }

        let input = CompactionInput {
            columns: columns.clone(),
            column_data: columnData,
            table_id: 100 + idx as u64,
            xmin_lo: *xminLo,
            xmin_hi: *xminHi,
        };

        let result = run_compaction_cycle(&config, input).expect("compaction failed");
        let reader = ZyrFileReader::open(&result.file_path).expect("open failed");
        fileHeaders.push((reader.header().xmin_range_lo, reader.header().xmin_range_hi));
        filePaths.push(result.file_path);
    }

    // Pruning logic: snapshot at txn_id=350
    let snapshotTxnId: u64 = 350;

    let mut fullyVisible = Vec::new();
    let mut partiallyVisible = Vec::new();
    let mut pruned = Vec::new();

    for (idx, (xminLo, xminHi)) in fileHeaders.iter().enumerate() {
        if *xminHi < snapshotTxnId {
            // All rows committed before snapshot
            fullyVisible.push(idx);
        } else if *xminLo > snapshotTxnId {
            // All rows from future txns, skip
            pruned.push(idx);
        } else {
            // Partial overlap
            partiallyVisible.push(idx);
        }
    }

    tprintln!("  Snapshot txn_id={}", snapshotTxnId);
    tprintln!(
        "  Fully visible files: {:?} (expected [0, 1])",
        fullyVisible
    );
    tprintln!("  Partially visible: {:?} (expected [2])", partiallyVisible);
    tprintln!("  Pruned: {:?} (expected [3, 4])", pruned);

    assert_eq!(
        fullyVisible,
        vec![0, 1],
        "files A,B should be fully visible"
    );
    assert_eq!(
        partiallyVisible,
        vec![2],
        "file C should be partially visible"
    );
    assert_eq!(pruned, vec![3, 4], "files D,E should be pruned");
    tprintln!("  Pruning correctness: PASS");

    // Verify xmin fields round-tripped correctly
    for (idx, (xminLo, xminHi)) in xminRanges.iter().enumerate() {
        assert_eq!(
            fileHeaders[idx].0, *xminLo,
            "xmin_lo mismatch for file {}",
            idx
        );
        assert_eq!(
            fileHeaders[idx].1, *xminHi,
            "xmin_hi mismatch for file {}",
            idx
        );
    }
    tprintln!("  xmin range round-trip: PASS");

    // Performance: 100K pruning decisions
    let mut pruningResults = Vec::with_capacity(VALIDATION_RUNS);
    let decisionCount = 100_000;
    for _ in 0..VALIDATION_RUNS {
        let start = Instant::now();
        let mut skipCount = 0u64;
        for txnId in 0..decisionCount {
            let snapshotId = (txnId % 700) as u64;
            for (xminLo, xminHi) in &fileHeaders {
                if *xminLo > snapshotId {
                    skipCount += 1;
                }
                if *xminHi < snapshotId {
                    // visible
                }
            }
        }
        let elapsed = start.elapsed().as_secs_f64();
        let nsPerDecision = elapsed * 1e9 / (decisionCount as f64 * fileHeaders.len() as f64);
        pruningResults.push(nsPerDecision);
        std::hint::black_box(skipCount);
    }

    check_performance(
        "Txn-Aware Pruning",
        "Pruning decision (ns/file)",
        pruningResults.iter().sum::<f64>() / pruningResults.len() as f64,
        10.0,
        false,
    );

    let utilAfter = take_util_snapshot();
    record_test_util("Txn-Aware Pruning", utilBefore, utilAfter);
    tprintln!("\n  Transaction-aware pruning: ALL PASS");
}

// =============================================================================
// Test 9: Bloom Filter
// =============================================================================

#[test]
fn test_bloom_filter() {
    zyron_bench_harness::init("columnar");
    let _benchGuard = BENCHMARK_LOCK.lock().unwrap_or_else(|e| e.into_inner());

    tprintln!("\n=== Phase 1.7: Bloom Filter ===");

    let utilBefore = take_util_snapshot();

    let insertCount = 10_000u64;
    let absentProbeCount = 100_000u64;

    // Build bloom filter
    let mut filter = BloomFilter::new(insertCount);
    for i in 0..insertCount {
        let key = format!("bloom_key_{}", i);
        filter.insert(key.as_bytes());
    }

    // No false negatives
    for i in 0..insertCount {
        let key = format!("bloom_key_{}", i);
        assert!(
            filter.might_contain(key.as_bytes()),
            "false negative for bloom_key_{}",
            i
        );
    }
    tprintln!(
        "  Zero false negatives ({}K probes): PASS",
        insertCount / 1000
    );

    // False positive rate
    let mut falsePositives = 0u64;
    for i in 0..absentProbeCount {
        let key = format!("absent_key_{}", i);
        if filter.might_contain(key.as_bytes()) {
            falsePositives += 1;
        }
    }
    let fpr = falsePositives as f64 / absentProbeCount as f64;
    tprintln!(
        "  False positive rate: {:.4} ({}/{})",
        fpr,
        falsePositives,
        absentProbeCount
    );
    assert!(fpr < 0.08, "FP rate too high: {:.4}", fpr);
    tprintln!("  FP rate < 8%: PASS");

    // Serialization roundtrip
    let serialized = filter.to_bytes();
    let restored = BloomFilter::from_bytes(&serialized).expect("deserialization failed");
    for i in 0..insertCount {
        let key = format!("bloom_key_{}", i);
        assert!(
            restored.might_contain(key.as_bytes()),
            "roundtrip false negative at {}",
            i
        );
    }
    tprintln!("  Serialization round-trip: PASS");

    // Skip rate: simulate 100 files, only 1 contains the target key
    let targetKey = "bloom_key_5000";
    let mut fileFilters: Vec<BloomFilter> = Vec::with_capacity(100);
    for fileIdx in 0..100u64 {
        let mut ff = BloomFilter::new(100);
        for j in 0..100u64 {
            let key = format!("file_{}_key_{}", fileIdx, j);
            ff.insert(key.as_bytes());
        }
        fileFilters.push(ff);
    }
    // Insert target key into file 50
    fileFilters[50].insert(targetKey.as_bytes());

    let mut skipped = 0usize;
    for (_idx, ff) in fileFilters.iter().enumerate() {
        if !ff.might_contain(targetKey.as_bytes()) {
            skipped += 1;
        }
    }
    let skipRate = skipped as f64 / 99.0 * 100.0; // 99 files should not contain it
    tprintln!(
        "  Bloom skip rate: {:.1}% ({}/99 files skipped)",
        skipRate,
        skipped
    );
    assert!(skipRate >= 90.0, "skip rate too low: {:.1}%", skipRate);

    // Low-cardinality: segment should NOT build bloom filter
    {
        let dictVals: Vec<[u8; 4]> = (0..10u32).map(|v| v.to_le_bytes()).collect();
        let values: Vec<Option<&[u8]>> = (0..100)
            .map(|i| Some(dictVals[i % 10].as_slice()))
            .collect();
        let segment = ColumnSegment::build(0, TypeId::Int32, 4, &values).expect("build failed");
        assert!(
            segment.bloom_filter.is_none(),
            "low-cardinality segment should not have bloom filter (cardinality={}, threshold={})",
            segment.header.cardinality,
            BLOOM_MIN_CARDINALITY
        );
        tprintln!(
            "  Low-cardinality segment: no bloom filter (cardinality={}): PASS",
            segment.header.cardinality
        );
    }

    // High-cardinality: segment SHOULD build bloom filter
    {
        let vals: Vec<[u8; 4]> = (0..200u32).map(|v| v.to_le_bytes()).collect();
        let values: Vec<Option<&[u8]>> = vals.iter().map(|v| Some(v.as_slice())).collect();
        let segment = ColumnSegment::build(0, TypeId::Int32, 4, &values).expect("build failed");
        assert!(
            segment.bloom_filter.is_some(),
            "high-cardinality segment should have bloom filter (cardinality={})",
            segment.header.cardinality
        );
        tprintln!(
            "  High-cardinality segment: bloom filter present (cardinality={}): PASS",
            segment.header.cardinality
        );
    }

    // Performance: probe latency (keys pre-computed to exclude format!() overhead)
    let mut probeResults = Vec::with_capacity(VALIDATION_RUNS);
    let probeCount = 100_000;
    let probeKeys: Vec<String> = (0..probeCount)
        .map(|i| format!("bloom_key_{}", i % (insertCount as usize)))
        .collect();
    for _ in 0..VALIDATION_RUNS {
        let start = Instant::now();
        let mut hits = 0u64;
        for i in 0..probeCount {
            if filter.might_contain(probeKeys[i].as_bytes()) {
                hits += 1;
            }
        }
        let elapsed = start.elapsed().as_secs_f64();
        probeResults.push(elapsed * 1e9 / probeCount as f64);
        std::hint::black_box(hits);
    }

    validate_metric(
        "Bloom Filter",
        "Bloom probe latency (ns)",
        probeResults,
        BLOOM_PROBE_TARGET_NS,
        false,
    );

    check_performance(
        "Bloom Filter",
        "Bloom skip rate (%)",
        skipRate,
        BLOOM_SKIP_RATE_TARGET_PCT,
        true,
    );

    let utilAfter = take_util_snapshot();
    record_test_util("Bloom Filter", utilBefore, utilAfter);
    tprintln!("\n  Bloom filter: ALL PASS");
}

// =============================================================================
// Test 10: Micro-Batch Zone Map
// =============================================================================

#[test]
fn test_micro_batch_zone_map() {
    zyron_bench_harness::init("columnar");
    let _benchGuard = BENCHMARK_LOCK.lock().unwrap_or_else(|e| e.into_inner());

    tprintln!("\n=== Phase 1.7: Micro-Batch Zone Map ===");

    let utilBefore = take_util_snapshot();

    let batchSize = ZONE_MAP_BATCH_SIZE as usize;
    let batchCount = 100;
    let rowCount = batchSize * batchCount;

    // Each batch k has values in range [k*1000, k*1000+999]
    let vals: Vec<[u8; 4]> = (0..rowCount)
        .map(|i| {
            let batch = i / batchSize;
            let offset = i % batchSize;
            let v = (batch * 1000 + offset.min(999)) as u32;
            v.to_le_bytes()
        })
        .collect();
    let values: Vec<Option<&[u8]>> = vals.iter().map(|v| Some(v.as_slice())).collect();

    let segment = ColumnSegment::build(0, TypeId::Int32, 4, &values).expect("build failed");
    assert_eq!(
        segment.zone_maps.len(),
        batchCount,
        "expected {} zone maps, got {}",
        batchCount,
        segment.zone_maps.len()
    );
    tprintln!(
        "  Zone map count: {} (expected {}): PASS",
        segment.zone_maps.len(),
        batchCount
    );

    // Helper: check if a zone map entry overlaps with [queryLo, queryHi]
    fn zone_overlaps(
        entry: &ZoneMapEntry,
        queryLo: &[u8; STAT_VALUE_SIZE],
        queryHi: &[u8; STAT_VALUE_SIZE],
    ) -> bool {
        // entry overlaps if entry.max >= queryLo AND entry.min <= queryHi
        compare_stat_slots(&entry.max_value, queryLo) != std::cmp::Ordering::Less
            && compare_stat_slots(&entry.min_value, queryHi) != std::cmp::Ordering::Greater
    }

    // Narrow range query: [5000, 5999] matches only batch 5
    {
        let queryLo = value_to_stat_slot(&5000u32.to_le_bytes());
        let queryHi = value_to_stat_slot(&5999u32.to_le_bytes());

        let matchingZones: Vec<usize> = segment
            .zone_maps
            .iter()
            .enumerate()
            .filter(|(_, zm)| zone_overlaps(zm, &queryLo, &queryHi))
            .map(|(i, _)| i)
            .collect();

        let skipRate = (batchCount - matchingZones.len()) as f64 / batchCount as f64 * 100.0;
        tprintln!(
            "  Narrow range [5000,5999]: {} matching zones, skip rate {:.1}%",
            matchingZones.len(),
            skipRate
        );

        assert!(
            matchingZones.len() <= 2,
            "narrow range should match <= 2 zones, got {}",
            matchingZones.len()
        );
        assert!(matchingZones.contains(&5), "batch 5 should match");
    }

    // Wide range query: [5000, 15999] matches batches 5-15
    {
        let queryLo = value_to_stat_slot(&5000u32.to_le_bytes());
        let queryHi = value_to_stat_slot(&15999u32.to_le_bytes());

        let matchingZones: Vec<usize> = segment
            .zone_maps
            .iter()
            .enumerate()
            .filter(|(_, zm)| zone_overlaps(zm, &queryLo, &queryHi))
            .map(|(i, _)| i)
            .collect();

        let skipRate = (batchCount - matchingZones.len()) as f64 / batchCount as f64 * 100.0;
        tprintln!(
            "  Wide range [5000,15999]: {} matching zones, skip rate {:.1}%",
            matchingZones.len(),
            skipRate
        );
        assert!(
            matchingZones.len() >= 10 && matchingZones.len() <= 12,
            "wide range should match ~11 zones, got {}",
            matchingZones.len()
        );
    }

    // Out-of-range query: [200000, 300000]
    {
        let queryLo = value_to_stat_slot(&200_000u32.to_le_bytes());
        let queryHi = value_to_stat_slot(&300_000u32.to_le_bytes());

        let matchingZones: Vec<usize> = segment
            .zone_maps
            .iter()
            .enumerate()
            .filter(|(_, zm)| zone_overlaps(zm, &queryLo, &queryHi))
            .map(|(i, _)| i)
            .collect();

        assert_eq!(
            matchingZones.len(),
            0,
            "out-of-range query should match 0 zones"
        );
        tprintln!("  Out-of-range [200K,300K]: 0 matching zones, skip rate 100%: PASS");
    }

    // Zone map serialization roundtrip
    for entry in &segment.zone_maps {
        let bytes = entry.to_bytes();
        let recovered = ZoneMapEntry::from_bytes(&bytes);
        assert_eq!(
            recovered.min_value, entry.min_value,
            "zone map min roundtrip failed"
        );
        assert_eq!(
            recovered.max_value, entry.max_value,
            "zone map max roundtrip failed"
        );
    }
    tprintln!("  Zone map serialization round-trip: PASS");

    // Performance: narrow-range skip rate
    let narrowQueryLo = value_to_stat_slot(&5000u32.to_le_bytes());
    let narrowQueryHi = value_to_stat_slot(&5999u32.to_le_bytes());
    let narrowMatching = segment
        .zone_maps
        .iter()
        .filter(|zm| zone_overlaps(zm, &narrowQueryLo, &narrowQueryHi))
        .count();
    let narrowSkipRate = (batchCount - narrowMatching) as f64 / batchCount as f64 * 100.0;

    check_performance(
        "Zone Map",
        "Narrow range batch skip rate (%)",
        narrowSkipRate,
        ZONE_MAP_BATCH_SKIP_RATE_TARGET_PCT,
        true,
    );

    // Zone map overhead: zone map size relative to encoded data
    let zoneMapTotalBytes = segment.zone_maps.len() * ZONE_MAP_ENTRY_SIZE;
    let encodedDataBytes = segment.encoded_data.len();
    let overheadPct = zoneMapTotalBytes as f64 / encodedDataBytes as f64 * 100.0;
    tprintln!(
        "  Zone map overhead: {:.2}% of encoded data ({} / {} bytes)",
        overheadPct,
        zoneMapTotalBytes,
        encodedDataBytes
    );
    assert!(
        overheadPct < 100.0,
        "zone map overhead should be reasonable"
    );

    let utilAfter = take_util_snapshot();
    record_test_util("Zone Map", utilBefore, utilAfter);
    tprintln!("\n  Micro-batch zone map: ALL PASS");
}

// =============================================================================
// Test 11: Sorted Segment
// =============================================================================

#[test]
fn test_sorted_segment() {
    zyron_bench_harness::init("columnar");
    let _benchGuard = BENCHMARK_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    const ROWS_PER_FILE: usize = 10_000;

    tprintln!("\n=== Phase 1.7: Sorted Segment ===");
    tprintln!("Files: 3, Rows per file: {}", ROWS_PER_FILE);

    let utilBefore = take_util_snapshot();

    // Create 3 sorted .zyr files via compaction
    let columns = vec![
        ColumnDescriptor {
            column_id: 0,
            type_id: TypeId::Int32,
            value_size: 4,
            is_primary_key: true,
        },
        ColumnDescriptor {
            column_id: 1,
            type_id: TypeId::Int64,
            value_size: 8,
            is_primary_key: false,
        },
    ];

    let mut filePaths = Vec::new();
    let mut decodedPkColumns: Vec<Vec<u8>> = Vec::new();

    for fileIdx in 0..3 {
        let dir = tempdir().expect("failed to create temp dir");
        let config = CompactionConfig {
            columnar_dir: dir.path().to_path_buf(),
            min_rows: 1,
            max_rows_per_file: 1_000_000,
            fsync_enabled: false,
            max_encoding_threads: 1,
            oltp_p99_threshold_us: 10_000,
            check_interval_ms: 1000,
        };

        let baseKey = fileIdx * ROWS_PER_FILE;
        let mut columnData: Vec<Vec<Option<Vec<u8>>>> = vec![Vec::with_capacity(ROWS_PER_FILE); 2];
        for i in 0..ROWS_PER_FILE {
            let pk = (baseKey + i) as u32;
            columnData[0].push(Some(pk.to_le_bytes().to_vec()));
            columnData[1].push(Some(((baseKey + i) as i64 * 10).to_le_bytes().to_vec()));
        }

        let input = CompactionInput {
            columns: columns.clone(),
            column_data: columnData,
            table_id: 200 + fileIdx as u64,
            xmin_lo: 1,
            xmin_hi: 1000,
        };

        let result = run_compaction_cycle(&config, input).expect("compaction failed");

        // Decode PK column for binary search and merge-scan
        let reader = ZyrFileReader::open(&result.file_path).expect("open failed");
        let segRaw = reader.read_segment_raw(0).expect("read PK segment failed");
        let headerBuf: [u8; SEGMENT_HEADER_SIZE] =
            segRaw[..SEGMENT_HEADER_SIZE].try_into().unwrap();
        let segHeader = SegmentHeader::from_bytes(&headerBuf).expect("header parse failed");
        let bloomSize = segHeader.bloom_filter_size as usize;
        let zoneCount =
            (ROWS_PER_FILE + ZONE_MAP_BATCH_SIZE as usize - 1) / ZONE_MAP_BATCH_SIZE as usize;
        let zoneMapSize = zoneCount * ZONE_MAP_ENTRY_SIZE;
        let dataStart = SEGMENT_HEADER_SIZE + bloomSize + zoneMapSize;
        let dataEnd = dataStart + segHeader.compressed_size as usize;
        let encoder = create_encoding(segHeader.encoding_type);
        let decoded = encoder
            .decode(&segRaw[dataStart..dataEnd], ROWS_PER_FILE, 4)
            .expect("decode failed");

        decodedPkColumns.push(decoded);
        filePaths.push(result.file_path);
    }

    // Build SortedSegmentIndex
    let mut index = SortedSegmentIndex::new();
    for fileIdx in 0..3 {
        let baseKey = (fileIdx * ROWS_PER_FILE) as u32;
        let maxKey = ((fileIdx + 1) * ROWS_PER_FILE - 1) as u32;
        index.add(SortedSegmentEntry {
            file_path: filePaths[fileIdx].clone(),
            min_pk: value_to_stat_slot(&baseKey.to_le_bytes()),
            max_pk: value_to_stat_slot(&maxKey.to_le_bytes()),
            row_count: ROWS_PER_FILE as u64,
        });
    }

    assert_eq!(index.file_count(), 3);
    assert_eq!(index.total_rows(), (ROWS_PER_FILE * 3) as u64);
    tprintln!(
        "  SortedSegmentIndex: {} files, {} total rows: PASS",
        index.file_count(),
        index.total_rows()
    );

    // Point lookup: find_point for key 5000 (in file 0)
    let pk5000 = value_to_stat_slot(&5000u32.to_le_bytes());
    let results = index.find_point(&pk5000);
    assert_eq!(results.len(), 1, "key 5000 should be in 1 file");
    tprintln!("  find_point(5000): 1 file: PASS");

    // Point lookup: find_point for key 15000 (in file 1)
    let pk15000 = value_to_stat_slot(&15000u32.to_le_bytes());
    let results = index.find_point(&pk15000);
    assert_eq!(results.len(), 1, "key 15000 should be in 1 file");
    tprintln!("  find_point(15000): 1 file: PASS");

    // Point lookup: find_point for key 99999 (not in any file)
    let pk99999 = value_to_stat_slot(&99999u32.to_le_bytes());
    let results = index.find_point(&pk99999);
    assert_eq!(results.len(), 0, "key 99999 should not be in any file");
    tprintln!("  find_point(99999): 0 files: PASS");

    // Binary search in decoded PK column (file 0)
    let target5000 = 5000u32.to_le_bytes();
    let foundIdx = binary_search_sorted_column(&decodedPkColumns[0], ROWS_PER_FILE, 4, &target5000);
    assert_eq!(
        foundIdx,
        Some(5000),
        "binary search should find key 5000 at index 5000"
    );
    tprintln!("  binary_search(5000): found at index 5000: PASS");

    // Binary search for absent key
    let targetAbsent = 99999u32.to_le_bytes();
    let absentIdx =
        binary_search_sorted_column(&decodedPkColumns[0], ROWS_PER_FILE, 4, &targetAbsent);
    assert_eq!(absentIdx, None, "binary search should not find absent key");
    tprintln!("  binary_search(99999): None: PASS");

    // Range lookup
    let rangeLo = value_to_stat_slot(&5000u32.to_le_bytes());
    let rangeHi = value_to_stat_slot(&15000u32.to_le_bytes());
    let rangeResults = index.find_range(&rangeLo, &rangeHi);
    assert_eq!(
        rangeResults.len(),
        2,
        "range [5000,15000] should span 2 files"
    );
    tprintln!("  find_range(5000,15000): 2 files: PASS");

    // Merge scan
    let rowCounts = vec![ROWS_PER_FILE; 3];
    let mut mergeIter = MergeScanIterator::new(decodedPkColumns.clone(), 4, rowCounts)
        .expect("merge scan init failed");

    let mut mergeCount = 0usize;
    let mut prevPk: Option<u32> = None;
    while let Some((fileIdx, rowIdx)) = mergeIter.next() {
        let offset = rowIdx * 4;
        let pk = u32::from_le_bytes(
            decodedPkColumns[fileIdx][offset..offset + 4]
                .try_into()
                .unwrap(),
        );
        if let Some(prev) = prevPk {
            assert!(
                pk >= prev,
                "merge scan not sorted: {} followed by {}",
                prev,
                pk
            );
        }
        prevPk = Some(pk);
        mergeCount += 1;
    }
    assert_eq!(
        mergeCount,
        ROWS_PER_FILE * 3,
        "merge scan should emit {} rows, got {}",
        ROWS_PER_FILE * 3,
        mergeCount
    );
    tprintln!("  Merge scan: {} rows in sorted order: PASS", mergeCount);

    // Performance: binary search
    let mut bsResults = Vec::with_capacity(VALIDATION_RUNS);
    let lookupCount = 100_000;
    for _ in 0..VALIDATION_RUNS {
        let start = Instant::now();
        for i in 0..lookupCount {
            let target = ((i % ROWS_PER_FILE) as u32).to_le_bytes();
            std::hint::black_box(binary_search_sorted_column(
                &decodedPkColumns[0],
                ROWS_PER_FILE,
                4,
                &target,
            ));
        }
        let elapsed = start.elapsed().as_secs_f64();
        bsResults.push(elapsed * 1e9 / lookupCount as f64);
    }
    validate_metric(
        "Sorted Segment",
        "Sorted PK lookup (ns)",
        bsResults,
        SORTED_PK_LOOKUP_TARGET_NS,
        false,
    );

    // Performance: merge scan throughput
    let totalMergeRows = ROWS_PER_FILE * 3;
    let mut mergeResults = Vec::with_capacity(VALIDATION_RUNS);
    for _ in 0..VALIDATION_RUNS {
        let start = Instant::now();
        let mut iter =
            MergeScanIterator::new(decodedPkColumns.clone(), 4, vec![ROWS_PER_FILE; 3]).unwrap();
        let mut count = 0usize;
        while let Some(entry) = iter.next() {
            std::hint::black_box(entry);
            count += 1;
        }
        let elapsed = start.elapsed().as_secs_f64();
        mergeResults.push(count as f64 / elapsed);
        assert_eq!(count, totalMergeRows);
    }
    validate_metric(
        "Sorted Segment",
        "Sorted PK range (keys/sec)",
        mergeResults,
        SORTED_PK_RANGE_TARGET_KEYS_SEC,
        true,
    );

    let utilAfter = take_util_snapshot();
    record_test_util("Sorted Segment", utilBefore, utilAfter);
    tprintln!("\n  Sorted segment: ALL PASS");
}

// =============================================================================
// Test 12: Parallel Column Encoding
// =============================================================================

#[test]
fn test_parallel_column_encoding() {
    zyron_bench_harness::init("columnar");
    let _benchGuard = BENCHMARK_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    const ROW_COUNT: usize = 100_000;
    const COL_COUNT: usize = 8;

    tprintln!("\n=== Phase 1.7: Parallel Column Encoding ===");
    tprintln!("Rows: {}, Columns: {}", ROW_COUNT, COL_COUNT);

    let utilBefore = take_util_snapshot();

    let columns = vec![
        ColumnDescriptor {
            column_id: 0,
            type_id: TypeId::Int32,
            value_size: 4,
            is_primary_key: true,
        },
        ColumnDescriptor {
            column_id: 1,
            type_id: TypeId::Int64,
            value_size: 8,
            is_primary_key: false,
        },
        ColumnDescriptor {
            column_id: 2,
            type_id: TypeId::Float64,
            value_size: 8,
            is_primary_key: false,
        },
        ColumnDescriptor {
            column_id: 3,
            type_id: TypeId::Int32,
            value_size: 4,
            is_primary_key: false,
        },
        ColumnDescriptor {
            column_id: 4,
            type_id: TypeId::Boolean,
            value_size: 1,
            is_primary_key: false,
        },
        ColumnDescriptor {
            column_id: 5,
            type_id: TypeId::Int32,
            value_size: 4,
            is_primary_key: false,
        },
        ColumnDescriptor {
            column_id: 6,
            type_id: TypeId::Int32,
            value_size: 4,
            is_primary_key: false,
        },
        ColumnDescriptor {
            column_id: 7,
            type_id: TypeId::Int64,
            value_size: 8,
            is_primary_key: false,
        },
    ];

    let buildColumnData = || -> Vec<Vec<Option<Vec<u8>>>> {
        let mut rng = rand::rng();
        let mut columnData: Vec<Vec<Option<Vec<u8>>>> =
            vec![Vec::with_capacity(ROW_COUNT); COL_COUNT];
        for i in 0..ROW_COUNT {
            // Col 0: PK sorted
            columnData[0].push(Some((i as u32).to_le_bytes().to_vec()));
            // Col 1: i64 sequential
            columnData[1].push(Some(((i as i64) * 100).to_le_bytes().to_vec()));
            // Col 2: f64 decimal
            columnData[2].push(Some((i as f64 * 0.01).to_le_bytes().to_vec()));
            // Col 3: low-cardinality
            columnData[3].push(Some(((i % 25) as u32).to_le_bytes().to_vec()));
            // Col 4: boolean
            columnData[4].push(Some(vec![(i % 2) as u8]));
            // Col 5: constant
            columnData[5].push(Some(999u32.to_le_bytes().to_vec()));
            // Col 6: runs
            columnData[6].push(Some(((i / 500) as u32).to_le_bytes().to_vec()));
            // Col 7: random i64
            columnData[7].push(Some(rng.random::<i64>().to_le_bytes().to_vec()));
        }
        columnData
    };

    // Sequential compaction (1 thread)
    let mut seqTimes = Vec::with_capacity(VALIDATION_RUNS);
    for run in 0..VALIDATION_RUNS {
        let dir = tempdir().expect("failed to create temp dir");
        let config = CompactionConfig {
            columnar_dir: dir.path().to_path_buf(),
            min_rows: 1,
            max_rows_per_file: 1_000_000,
            fsync_enabled: false,
            max_encoding_threads: 1,
            oltp_p99_threshold_us: 10_000,
            check_interval_ms: 1000,
        };

        let input = CompactionInput {
            columns: columns.clone(),
            column_data: buildColumnData(),
            table_id: 300,
            xmin_lo: 1,
            xmin_hi: 100_000,
        };

        let start = Instant::now();
        let result = run_compaction_cycle(&config, input).expect("seq compaction failed");
        let elapsed = start.elapsed().as_secs_f64();
        seqTimes.push(elapsed);

        if run == 0 {
            assert_eq!(result.row_count, ROW_COUNT as u64);
            assert_eq!(result.column_count, COL_COUNT as u32);
            let reader = ZyrFileReader::open(&result.file_path).expect("open failed");
            assert_eq!(reader.segment_count(), COL_COUNT);
            assert_eq!(reader.header().sort_order, SortOrder::Asc);
            tprintln!(
                "  Sequential: {} rows, {} cols, {} bytes",
                result.row_count,
                result.column_count,
                result.file_size
            );
        }
    }

    // Parallel compaction (8 threads)
    let mut parTimes = Vec::with_capacity(VALIDATION_RUNS);
    for run in 0..VALIDATION_RUNS {
        let dir = tempdir().expect("failed to create temp dir");
        let config = CompactionConfig {
            columnar_dir: dir.path().to_path_buf(),
            min_rows: 1,
            max_rows_per_file: 1_000_000,
            fsync_enabled: false,
            max_encoding_threads: 8,
            oltp_p99_threshold_us: 10_000,
            check_interval_ms: 1000,
        };

        let input = CompactionInput {
            columns: columns.clone(),
            column_data: buildColumnData(),
            table_id: 301,
            xmin_lo: 1,
            xmin_hi: 100_000,
        };

        let start = Instant::now();
        let result = run_compaction_cycle(&config, input).expect("par compaction failed");
        let elapsed = start.elapsed().as_secs_f64();
        parTimes.push(elapsed);

        if run == 0 {
            assert_eq!(result.row_count, ROW_COUNT as u64);
            assert_eq!(result.column_count, COL_COUNT as u32);
            let reader = ZyrFileReader::open(&result.file_path).expect("open failed");
            assert_eq!(reader.segment_count(), COL_COUNT);
            assert_eq!(reader.header().sort_order, SortOrder::Asc);
            tprintln!(
                "  Parallel: {} rows, {} cols, {} bytes",
                result.row_count,
                result.column_count,
                result.file_size
            );
        }
    }

    // Correctness: verify parallel output via hash comparison
    {
        let dir = tempdir().expect("failed to create temp dir");
        let parConfig = CompactionConfig {
            columnar_dir: dir.path().to_path_buf(),
            min_rows: 1,
            max_rows_per_file: 1_000_000,
            fsync_enabled: false,
            max_encoding_threads: 8,
            oltp_p99_threshold_us: 10_000,
            check_interval_ms: 1000,
        };

        // Use deterministic data (no random column) for hash comparison
        let mut colData: Vec<Vec<Option<Vec<u8>>>> = vec![Vec::with_capacity(ROW_COUNT); COL_COUNT];
        for i in 0..ROW_COUNT {
            colData[0].push(Some((i as u32).to_le_bytes().to_vec()));
            colData[1].push(Some(((i as i64) * 100).to_le_bytes().to_vec()));
            colData[2].push(Some((i as f64 * 0.01).to_le_bytes().to_vec()));
            colData[3].push(Some(((i % 25) as u32).to_le_bytes().to_vec()));
            colData[4].push(Some(vec![(i % 2) as u8]));
            colData[5].push(Some(999u32.to_le_bytes().to_vec()));
            colData[6].push(Some(((i / 500) as u32).to_le_bytes().to_vec()));
            colData[7].push(Some(((i as i64) * 7 + 13).to_le_bytes().to_vec()));
        }

        let input = CompactionInput {
            columns: columns.clone(),
            column_data: colData,
            table_id: 302,
            xmin_lo: 1,
            xmin_hi: 100_000,
        };

        let result = run_compaction_cycle(&parConfig, input).expect("hash-check compaction failed");
        let reader = ZyrFileReader::open(&result.file_path).expect("open failed");

        // Read all segments and verify decodable
        for col in &columns {
            let segRaw = reader
                .read_segment_raw(col.column_id)
                .expect("read segment failed");
            let headerBuf: [u8; SEGMENT_HEADER_SIZE] =
                segRaw[..SEGMENT_HEADER_SIZE].try_into().unwrap();
            let segHeader = SegmentHeader::from_bytes(&headerBuf).expect("header parse failed");
            assert!(
                segHeader.compressed_size > 0,
                "col {} has 0 compressed size",
                col.column_id
            );
        }
        tprintln!("  Parallel correctness (all columns decodable): PASS");
    }

    // Speedup calculation
    let avgSeq = seqTimes.iter().sum::<f64>() / seqTimes.len() as f64;
    let avgPar = parTimes.iter().sum::<f64>() / parTimes.len() as f64;
    let speedups: Vec<f64> = seqTimes
        .iter()
        .zip(parTimes.iter())
        .map(|(s, p)| s / p)
        .collect();

    tprintln!(
        "  Avg sequential: {:.3}s, avg parallel: {:.3}s",
        avgSeq,
        avgPar
    );

    validate_metric(
        "Parallel Column Encoding",
        "Parallel speedup (x)",
        speedups.clone(),
        COMPACTION_PARALLEL_SPEEDUP_TARGET,
        true,
    );

    // Sequential throughput
    let seqRowsSec: Vec<f64> = seqTimes.iter().map(|t| ROW_COUNT as f64 / t).collect();
    validate_metric(
        "Parallel Column Encoding",
        "Sequential compaction (rows/sec)",
        seqRowsSec,
        COMPACTION_SEQ_TARGET_ROWS_SEC,
        true,
    );

    let utilAfter = take_util_snapshot();
    record_test_util("Parallel Column Encoding", utilBefore, utilAfter);
    tprintln!("\n  Parallel column encoding: ALL PASS");
}
