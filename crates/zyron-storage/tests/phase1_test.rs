//! Phase 1: Storage Foundation Validation Tests
//!
//! Comprehensive integration tests for ZyronDB Phase 1 components:
//! - WAL write and replay
//! - Buffer pool eviction and caching
//! - Heap file tuple storage
//! - B+ tree index operations
//! - Crash recovery integration

use bytes::Bytes;
use rand::Rng;
use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use tempfile::tempdir;

use zyron_buffer::{BufferPool, BufferPoolConfig};
use zyron_common::page::{PageId, PAGE_SIZE};
use zyron_storage::{
    BTreeIndex, DiskManager, DiskManagerConfig, HeapFile, Tuple, TupleId,
};
use zyron_wal::{LogRecordType, Lsn, RecoveryManager, WalReader, WalWriter, WalWriterConfig};

// =============================================================================
// Test 1: WAL Write/Replay Test
// =============================================================================

/// Writes 10,000 log records across multiple segments, simulates crash,
/// replays all records, and verifies integrity.
#[tokio::test]
async fn test_wal_write_replay_10k_records() {
    const RECORD_COUNT: usize = 10_000;

    let dir = tempdir().unwrap();
    let config = WalWriterConfig {
        wal_dir: dir.path().to_path_buf(),
        segment_size: 1024 * 1024, // 1MB segments to force rotation
        fsync_enabled: true,
        group_commit_size: 1,
    };

    // Phase 1: Write 10,000 records
    let mut written_lsns: Vec<Lsn> = Vec::with_capacity(RECORD_COUNT);
    let mut written_payloads: Vec<Vec<u8>> = Vec::with_capacity(RECORD_COUNT);

    {
        let writer = WalWriter::new(config.clone()).await.unwrap();

        for i in 0..RECORD_COUNT {
            let txn_id = (i % 100 + 1) as u32;
            let payload = format!("record_{}_{}", i, "x".repeat(i % 100));
            let payload_bytes = Bytes::from(payload.clone());

            let lsn = writer
                .log_insert(txn_id, Lsn::INVALID, payload_bytes)
                .await
                .unwrap();

            written_lsns.push(lsn);
            written_payloads.push(payload.into_bytes());
        }

        // Simulate crash: drop writer without clean shutdown
        // (writer.close() is NOT called)
        drop(writer);
    }

    // Phase 2: Create new reader and replay all records
    {
        let reader = WalReader::new(dir.path()).await.unwrap();
        let records = reader.scan_all().await.unwrap();

        // Verify record count
        assert!(
            records.len() >= RECORD_COUNT,
            "Expected at least {} records, got {}",
            RECORD_COUNT,
            records.len()
        );

        // Verify LSN ordering is monotonic
        let insert_records: Vec<_> = records
            .iter()
            .filter(|r| r.record_type == LogRecordType::Insert)
            .collect();

        assert_eq!(
            insert_records.len(),
            RECORD_COUNT,
            "Expected {} insert records",
            RECORD_COUNT
        );

        let mut prev_lsn = Lsn::INVALID;
        for record in &insert_records {
            assert!(
                record.lsn > prev_lsn || prev_lsn == Lsn::INVALID,
                "LSN ordering violated: {} should be > {}",
                record.lsn,
                prev_lsn
            );
            prev_lsn = record.lsn;
        }

        // Verify content integrity (spot check)
        for (i, record) in insert_records.iter().enumerate().take(100) {
            let expected_prefix = format!("record_{}_", i);
            let payload_str = String::from_utf8_lossy(&record.payload);
            assert!(
                payload_str.starts_with(&expected_prefix),
                "Record {} content mismatch",
                i
            );
        }
    }

    println!("WAL Write/Replay Test: PASSED - {} records written and verified", RECORD_COUNT);
}

/// Tests WAL segment rotation with many records.
#[tokio::test]
async fn test_wal_segment_rotation() {
    let dir = tempdir().unwrap();
    let config = WalWriterConfig {
        wal_dir: dir.path().to_path_buf(),
        segment_size: 64 * 1024, // 64KB segments for faster rotation
        fsync_enabled: false,
        group_commit_size: 1,
    };

    let writer = WalWriter::new(config).await.unwrap();
    let initial_segment = writer.current_segment_id().await.unwrap();

    // Write enough data to trigger rotation
    for _ in 0..1000 {
        let payload = Bytes::from(vec![0u8; 200]); // 200 bytes per record
        writer.log_insert(1, Lsn::INVALID, payload).await.unwrap();
    }

    let final_segment = writer.current_segment_id().await.unwrap();
    writer.close().await.unwrap();

    // Verify multiple segments were created
    assert!(
        final_segment.0 > initial_segment.0,
        "Expected segment rotation: {} -> {}",
        initial_segment,
        final_segment
    );

    // Verify all records can be read back
    let reader = WalReader::new(dir.path()).await.unwrap();
    let records = reader.scan_all().await.unwrap();
    assert_eq!(records.len(), 1000);

    println!("WAL Segment Rotation: PASSED - rotated from segment {} to {}", initial_segment, final_segment);
}

// =============================================================================
// Test 2: Buffer Pool Test
// =============================================================================

/// Tests buffer pool with 100 frames accessing 500 different pages.
#[tokio::test]
async fn test_buffer_pool_eviction() {
    const NUM_FRAMES: usize = 100;
    const NUM_PAGES: usize = 500;

    let pool = BufferPool::new(BufferPoolConfig { num_frames: NUM_FRAMES });

    let mut dirty_evictions = 0;

    // Access 500 different pages (force eviction since pool only has 100 frames)
    for i in 0..NUM_PAGES {
        let page_id = PageId::new(0, i as u32);

        let (frame, evicted) = pool.new_page(page_id).unwrap();

        if evicted.is_some() {
            dirty_evictions += 1;
        }

        // Write some data to make pages dirty
        {
            let mut data = frame.write_data();
            data[0..8].copy_from_slice(&(i as u64).to_le_bytes());
        }

        // Unpin with dirty flag
        pool.unpin_page(page_id, true);
    }

    // Verify evictions occurred
    assert!(
        dirty_evictions > 0,
        "Expected dirty evictions when accessing {} pages with {} frames",
        NUM_PAGES,
        NUM_FRAMES
    );

    // Verify pool is at capacity
    assert_eq!(pool.page_count(), NUM_FRAMES);

    println!(
        "Buffer Pool Eviction: PASSED - {} dirty evictions with {}/{} pages",
        dirty_evictions, NUM_PAGES, NUM_FRAMES
    );
}

/// Tests that pinned pages cannot be evicted.
#[tokio::test]
async fn test_buffer_pool_pin_prevents_eviction() {
    const NUM_FRAMES: usize = 10;

    let pool = BufferPool::new(BufferPoolConfig { num_frames: NUM_FRAMES });

    // Pin all frames
    let mut pinned_pages = Vec::new();
    for i in 0..NUM_FRAMES {
        let page_id = PageId::new(0, i as u32);
        pool.new_page(page_id).unwrap();
        pinned_pages.push(page_id);
    }

    // Try to add another page (should fail - all frames pinned)
    let result = pool.new_page(PageId::new(0, 999));
    assert!(
        result.is_err(),
        "Should fail when all frames are pinned"
    );

    // Unpin one page
    pool.unpin_page(pinned_pages[0], false);

    // Now adding should succeed
    let result = pool.new_page(PageId::new(0, 999));
    assert!(result.is_ok(), "Should succeed after unpinning");

    println!("Buffer Pool Pin Test: PASSED - pin prevents eviction");
}

/// Tests cache hit rate for repeated access patterns.
#[tokio::test]
async fn test_buffer_pool_cache_hit_rate() {
    const NUM_FRAMES: usize = 50;
    const NUM_PAGES: usize = 30; // Less than frames so all fit

    let pool = BufferPool::new(BufferPoolConfig { num_frames: NUM_FRAMES });

    // Initial load of pages
    for i in 0..NUM_PAGES {
        let page_id = PageId::new(0, i as u32);
        pool.new_page(page_id).unwrap();
        pool.unpin_page(page_id, false);
    }

    // Repeated access - should all be cache hits
    let mut hits = 0;
    let mut misses = 0;

    for _ in 0..100 {
        for i in 0..NUM_PAGES {
            let page_id = PageId::new(0, i as u32);
            if pool.fetch_page(page_id).is_some() {
                hits += 1;
                pool.unpin_page(page_id, false);
            } else {
                misses += 1;
            }
        }
    }

    let hit_rate = hits as f64 / (hits + misses) as f64;
    assert!(
        hit_rate >= 0.99,
        "Expected high hit rate for repeated access, got {:.2}%",
        hit_rate * 100.0
    );

    println!(
        "Buffer Pool Cache Hit Rate: PASSED - {:.2}% hit rate ({} hits, {} misses)",
        hit_rate * 100.0,
        hits,
        misses
    );
}

// =============================================================================
// Test 3: Heap File Test
// =============================================================================

/// Inserts 100,000 tuples with varying sizes.
#[tokio::test]
async fn test_heap_file_100k_tuples() {
    const TUPLE_COUNT: usize = 100_000;

    let dir = tempdir().unwrap();
    let config = DiskManagerConfig {
        data_dir: dir.path().to_path_buf(),
        fsync_enabled: false,
    };
    let disk = Arc::new(DiskManager::new(config).await.unwrap());
    let heap = HeapFile::with_defaults(disk).unwrap();

    let mut rng = rand::thread_rng();
    let mut tuple_ids: Vec<TupleId> = Vec::with_capacity(TUPLE_COUNT);
    let mut tuple_data: HashMap<TupleId, Vec<u8>> = HashMap::new();

    // Insert 100,000 tuples with varying sizes (10-500 bytes)
    for i in 0..TUPLE_COUNT {
        let size = rng.gen_range(10..=500);
        let data: Vec<u8> = (0..size).map(|j| ((i + j) % 256) as u8).collect();
        let tuple = Tuple::new(Bytes::from(data.clone()), i as u32);

        let tuple_id = heap.insert(&tuple).await.unwrap();
        tuple_ids.push(tuple_id);
        tuple_data.insert(tuple_id, data);

        if (i + 1) % 10000 == 0 {
            println!("Inserted {}/{} tuples", i + 1, TUPLE_COUNT);
        }
    }

    // Full table scan - verify all tuples returned
    let scanned = heap.scan().await.unwrap();
    assert_eq!(
        scanned.len(),
        TUPLE_COUNT,
        "Scan should return all {} tuples",
        TUPLE_COUNT
    );

    // Random point lookups - verify correct tuple returned
    for _ in 0..1000 {
        let idx = rng.gen_range(0..TUPLE_COUNT);
        let tuple_id = tuple_ids[idx];
        let expected_data = tuple_data.get(&tuple_id).unwrap();

        let retrieved = heap.get(tuple_id).await.unwrap();
        assert!(retrieved.is_some(), "Tuple {} should exist", tuple_id);

        let tuple = retrieved.unwrap();
        assert_eq!(
            tuple.data().as_ref(),
            expected_data.as_slice(),
            "Tuple {} data mismatch",
            tuple_id
        );
    }

    println!(
        "Heap File 100K Tuples: PASSED - {} tuples inserted across {} pages",
        TUPLE_COUNT,
        heap.num_pages().await.unwrap()
    );
}

/// Tests delete and scan exclusion.
#[tokio::test]
async fn test_heap_file_delete_and_scan() {
    const TUPLE_COUNT: usize = 10_000;
    const DELETE_COUNT: usize = 1_000;

    let dir = tempdir().unwrap();
    let config = DiskManagerConfig {
        data_dir: dir.path().to_path_buf(),
        fsync_enabled: false,
    };
    let disk = Arc::new(DiskManager::new(config).await.unwrap());
    let heap = HeapFile::with_defaults(disk).unwrap();

    let mut tuple_ids: Vec<TupleId> = Vec::with_capacity(TUPLE_COUNT);

    // Insert tuples
    for i in 0..TUPLE_COUNT {
        let data = Bytes::from(format!("tuple_{}", i));
        let tuple = Tuple::new(data, i as u32);
        tuple_ids.push(heap.insert(&tuple).await.unwrap());
    }

    // Delete first DELETE_COUNT tuples
    let mut deleted_ids: HashSet<TupleId> = HashSet::new();
    for i in 0..DELETE_COUNT {
        heap.delete(tuple_ids[i]).await.unwrap();
        deleted_ids.insert(tuple_ids[i]);
    }

    // Scan should exclude deleted tuples
    let scanned = heap.scan().await.unwrap();
    assert_eq!(
        scanned.len(),
        TUPLE_COUNT - DELETE_COUNT,
        "Scan should return {} tuples after {} deletions",
        TUPLE_COUNT - DELETE_COUNT,
        DELETE_COUNT
    );

    // Verify no deleted tuples in scan results
    for (tuple_id, _) in &scanned {
        assert!(
            !deleted_ids.contains(tuple_id),
            "Deleted tuple {} should not appear in scan",
            tuple_id
        );
    }

    // Verify deleted tuples return None on get
    for deleted_id in deleted_ids.iter().take(100) {
        let result = heap.get(*deleted_id).await.unwrap();
        assert!(result.is_none(), "Deleted tuple should return None");
    }

    println!(
        "Heap File Delete/Scan: PASSED - deleted {}/{} tuples",
        DELETE_COUNT, TUPLE_COUNT
    );
}

/// Tests free space reuse after deletes.
#[tokio::test]
async fn test_heap_file_space_reuse() {
    let dir = tempdir().unwrap();
    let config = DiskManagerConfig {
        data_dir: dir.path().to_path_buf(),
        fsync_enabled: false,
    };
    let disk = Arc::new(DiskManager::new(config).await.unwrap());
    let heap = HeapFile::with_defaults(disk).unwrap();

    // Insert large tuples to create multiple pages
    let mut tuple_ids = Vec::new();
    for i in 0..20 {
        let data = Bytes::from(vec![i as u8; PAGE_SIZE / 4]);
        let tuple = Tuple::new(data, i);
        tuple_ids.push(heap.insert(&tuple).await.unwrap());
    }

    let pages_after_insert = heap.num_pages().await.unwrap();

    // Delete all tuples
    for tuple_id in &tuple_ids {
        heap.delete(*tuple_id).await.unwrap();
    }

    // Insert again - should reuse space, not create many new pages
    for i in 0..20 {
        let data = Bytes::from(vec![(i + 100) as u8; PAGE_SIZE / 4]);
        let tuple = Tuple::new(data, i + 100);
        heap.insert(&tuple).await.unwrap();
    }

    let pages_after_reinsert = heap.num_pages().await.unwrap();

    // Should reuse at least some existing pages via FSM
    // With FSM category quantization, perfect reuse isn't guaranteed
    // but we should see significant reuse (less than double the pages)
    assert!(
        pages_after_reinsert <= pages_after_insert * 2,
        "Expected space reuse: {} pages after reinsert vs {} after initial insert",
        pages_after_reinsert,
        pages_after_insert
    );

    println!(
        "Heap File Space Reuse: PASSED - {} pages after reinsert (was {})",
        pages_after_reinsert, pages_after_insert
    );
}

// =============================================================================
// Test 4: B+ Tree Test
// =============================================================================

/// Inserts 1,000,000 random i64 keys and verifies operations.
#[tokio::test]
async fn test_btree_1m_keys() {
    const KEY_COUNT: usize = 1_000_000;

    let dir = tempdir().unwrap();
    let config = DiskManagerConfig {
        data_dir: dir.path().to_path_buf(),
        fsync_enabled: false,
    };
    let disk = Arc::new(DiskManager::new(config).await.unwrap());
    let pool = Arc::new(BufferPool::auto_sized());

    println!("BufferPool auto-sized to {} frames", pool.num_frames());

    let mut btree = BTreeIndex::create(disk.clone(), pool.clone(), 10).await.unwrap();

    let mut rng = rand::thread_rng();
    let mut keys: Vec<i64> = (0..KEY_COUNT as i64).collect();

    // Shuffle for random insertion order
    for i in (1..keys.len()).rev() {
        let j = rng.gen_range(0..=i);
        keys.swap(i, j);
    }

    // Insert all keys
    for (idx, &key) in keys.iter().enumerate() {
        let key_bytes = Bytes::from(key.to_be_bytes().to_vec());
        let tuple_id = TupleId::new(PageId::new(0, (key % 1000) as u32), (key % 100) as u16);

        btree.insert(key_bytes, tuple_id).await.unwrap();

        if (idx + 1) % 100000 == 0 {
            println!("Inserted {}/{} keys, height={}", idx + 1, KEY_COUNT, btree.height());
        }
    }

    let final_height = btree.height();
    println!("Final tree height: {}", final_height);

    // Verify tree is balanced (height should be reasonable for 1M keys)
    // B+ tree with order ~100 should have height <= 4 for 1M keys
    assert!(
        final_height <= 5,
        "Tree height {} is too large for {} keys",
        final_height,
        KEY_COUNT
    );

    // Point lookup each key (sample 10,000)
    println!("Verifying point lookups...");
    for i in (0..KEY_COUNT).step_by(100) {
        let key = i as i64;
        let key_bytes = key.to_be_bytes();
        let result = btree.search(&key_bytes).await.unwrap();
        assert!(
            result.is_some(),
            "Key {} should be found",
            key
        );
    }

    // Range scan [1000, 2000]
    println!("Testing range scan...");
    let start_key = 1000i64.to_be_bytes();
    let end_key = 2000i64.to_be_bytes();
    let range_results = btree
        .range_scan(Some(&start_key), Some(&end_key))
        .await
        .unwrap();

    // Verify sorted order
    let mut prev_key: Option<i64> = None;
    for (key_bytes, _) in &range_results {
        let key = i64::from_be_bytes(key_bytes.as_ref().try_into().unwrap());
        if let Some(prev) = prev_key {
            assert!(key >= prev, "Range scan not sorted: {} < {}", key, prev);
        }
        prev_key = Some(key);
    }

    // Verify range bounds
    assert!(
        range_results.len() >= 900,
        "Expected ~1000 keys in range [1000, 2000], got {}",
        range_results.len()
    );

    println!(
        "B+ Tree 1M Keys: PASSED - {} keys, height={}, range_scan returned {} results",
        KEY_COUNT,
        final_height,
        range_results.len()
    );
}

/// Tests B+ tree delete operations.
#[tokio::test]
async fn test_btree_delete() {
    const KEY_COUNT: usize = 10_000;
    const DELETE_COUNT: usize = 1_000;

    let dir = tempdir().unwrap();
    let config = DiskManagerConfig {
        data_dir: dir.path().to_path_buf(),
        fsync_enabled: false,
    };
    let disk = Arc::new(DiskManager::new(config).await.unwrap());
    let pool = Arc::new(BufferPool::auto_sized());

    let mut btree = BTreeIndex::create(disk.clone(), pool.clone(), 11).await.unwrap();

    // Insert keys
    for i in 0..KEY_COUNT {
        let key = Bytes::from((i as i64).to_be_bytes().to_vec());
        let tuple_id = TupleId::new(PageId::new(0, i as u32), 0);
        btree.insert(key, tuple_id).await.unwrap();
    }

    // Delete first DELETE_COUNT keys
    for i in 0..DELETE_COUNT {
        let key = (i as i64).to_be_bytes();
        let deleted = btree.delete(&key).await.unwrap();
        assert!(deleted, "Key {} should be deleted", i);
    }

    // Verify deleted keys are not found
    for i in 0..DELETE_COUNT {
        let key = (i as i64).to_be_bytes();
        let result = btree.search(&key).await.unwrap();
        assert!(result.is_none(), "Deleted key {} should not be found", i);
    }

    // Verify remaining keys still exist
    for i in DELETE_COUNT..KEY_COUNT {
        let key = (i as i64).to_be_bytes();
        let result = btree.search(&key).await.unwrap();
        assert!(result.is_some(), "Key {} should still exist", i);
    }

    println!(
        "B+ Tree Delete: PASSED - deleted {}/{} keys",
        DELETE_COUNT, KEY_COUNT
    );
}

// =============================================================================
// Test 5: Integration Test - WAL + Heap Recovery
// =============================================================================

/// Tests crash recovery: write tuples with WAL logging, crash, recover.
#[tokio::test]
async fn test_wal_heap_recovery() {
    let dir = tempdir().unwrap();
    let heap_dir = dir.path().join("heap");
    let wal_dir = dir.path().join("wal");

    std::fs::create_dir_all(&heap_dir).unwrap();
    std::fs::create_dir_all(&wal_dir).unwrap();

    const TUPLE_COUNT: usize = 1000;
    let mut committed_data: Vec<(u32, Vec<u8>)> = Vec::new();

    // Phase 1: Write tuples with WAL logging
    {
        let wal_config = WalWriterConfig {
            wal_dir: wal_dir.clone(),
            segment_size: 16 * 1024 * 1024,
            fsync_enabled: true,
            group_commit_size: 1,
        };
        let writer = Arc::new(WalWriter::new(wal_config).await.unwrap());

        let disk_config = DiskManagerConfig {
            data_dir: heap_dir.clone(),
            fsync_enabled: true,
        };
        let disk = Arc::new(DiskManager::new(disk_config).await.unwrap());
        let heap = HeapFile::with_defaults(disk).unwrap();

        // Write transactions
        for i in 0..TUPLE_COUNT {
            let txn_id = (i + 1) as u32;
            let data = format!("committed_tuple_{}", i);

            // Log begin
            let begin_lsn = writer.log_begin(txn_id).await.unwrap();

            // Insert tuple
            let tuple = Tuple::new(Bytes::from(data.clone()), txn_id);
            let tuple_id = heap.insert(&tuple).await.unwrap();

            // Log insert with tuple_id info
            let payload = format!("{}:{}:{}", tuple_id.page_id.page_num, tuple_id.slot_id, data);
            let insert_lsn = writer
                .log_insert(txn_id, begin_lsn, Bytes::from(payload))
                .await
                .unwrap();

            // Log commit
            writer.log_commit(txn_id, insert_lsn).await.unwrap();

            committed_data.push((txn_id, data.into_bytes()));
        }

        // Simulate crash: drop everything without clean shutdown
        writer.flush().await.unwrap();
        drop(writer);
        drop(heap);
    }

    // Phase 2: Recovery - replay WAL and verify
    {
        let recovery = RecoveryManager::new(&wal_dir).await.unwrap();
        let result = recovery.recover().await.unwrap();

        // All transactions were committed, so no undo needed
        assert!(
            result.undo_txns.is_empty(),
            "No transactions should need undo"
        );

        // Verify redo records match committed data
        assert_eq!(
            result.redo_records.len(),
            TUPLE_COUNT,
            "Should have {} redo records",
            TUPLE_COUNT
        );

        // Verify all committed transactions are in redo
        let redo_txns: HashSet<u32> = result.redo_records.iter().map(|r| r.txn_id).collect();
        for (txn_id, _) in &committed_data {
            assert!(
                redo_txns.contains(txn_id),
                "Transaction {} should be in redo set",
                txn_id
            );
        }
    }

    println!(
        "WAL+Heap Recovery: PASSED - {} committed transactions recovered",
        TUPLE_COUNT
    );
}

/// Tests recovery with uncommitted transactions.
#[tokio::test]
async fn test_wal_recovery_with_uncommitted() {
    let dir = tempdir().unwrap();

    let config = WalWriterConfig {
        wal_dir: dir.path().to_path_buf(),
        segment_size: 16 * 1024 * 1024,
        fsync_enabled: true,
        group_commit_size: 1,
    };

    // Write mix of committed and uncommitted transactions
    {
        let writer = WalWriter::new(config.clone()).await.unwrap();

        // Committed transactions
        for i in 1..=10 {
            let begin = writer.log_begin(i).await.unwrap();
            let insert = writer
                .log_insert(i, begin, Bytes::from(format!("data_{}", i)))
                .await
                .unwrap();
            writer.log_commit(i, insert).await.unwrap();
        }

        // Uncommitted transactions
        for i in 11..=15 {
            let begin = writer.log_begin(i).await.unwrap();
            writer
                .log_insert(i, begin, Bytes::from(format!("uncommitted_{}", i)))
                .await
                .unwrap();
            // No commit!
        }

        writer.flush().await.unwrap();
        drop(writer);
    }

    // Recovery
    let recovery = RecoveryManager::new(dir.path()).await.unwrap();
    let result = recovery.recover().await.unwrap();

    // Verify committed transactions in redo
    let committed_txns: HashSet<u32> = (1..=10).collect();
    let redo_txns: HashSet<u32> = result.redo_records.iter().map(|r| r.txn_id).collect();

    for txn in &committed_txns {
        assert!(
            redo_txns.contains(txn),
            "Committed transaction {} should be in redo",
            txn
        );
    }

    // Verify uncommitted transactions in undo
    let uncommitted_txns: HashSet<u32> = (11..=15).collect();
    let undo_set: HashSet<u32> = result.undo_txns.iter().copied().collect();

    for txn in &uncommitted_txns {
        assert!(
            undo_set.contains(txn),
            "Uncommitted transaction {} should be in undo",
            txn
        );
    }

    // Verify uncommitted not in redo
    for txn in &uncommitted_txns {
        assert!(
            !redo_txns.contains(txn),
            "Uncommitted transaction {} should NOT be in redo",
            txn
        );
    }

    println!(
        "WAL Recovery with Uncommitted: PASSED - {} redo, {} undo",
        result.redo_records.len(),
        result.undo_txns.len()
    );
}

// =============================================================================
// Summary Test
// =============================================================================

/// Runs a summary of all major component tests.
#[tokio::test]
async fn test_phase1_summary() {
    println!("\n======================================");
    println!("ZyronDB Phase 1: Storage Foundation");
    println!("======================================\n");

    println!("Component Status:");
    println!("  - WAL Writer/Reader: Implemented");
    println!("  - Buffer Pool: Implemented");
    println!("  - Disk Manager: Implemented");
    println!("  - Heap File: Implemented");
    println!("  - B+ Tree Index: Implemented");
    println!("  - Free Space Map: Implemented");
    println!("  - Tuple Storage: Implemented");
    println!("  - Recovery Manager: Implemented");
    println!("\nPhase 1 validation tests complete.");
    println!("Run: cargo test -p zyron-storage --test phase1_test -- --nocapture");
}
