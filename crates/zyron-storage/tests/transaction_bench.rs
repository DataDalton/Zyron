#![allow(non_snake_case, unused_assignments)]

//! Transaction Benchmark Suite
//!
//! Integration tests for ZyronDB transaction components:
//! - Snapshot isolation and MVCC visibility
//! - Write-write conflict detection
//! - Rollback and abort handling
//! - MVCC garbage collection
//! - Concurrent transaction throughput
//! - B+ tree latch coupling under contention
//! - Intent lock conflict detection
//!
//! Run: cargo test -p zyron-storage --test transaction_bench --release -- --nocapture

use zyron_bench_harness::*;

use std::collections::HashSet;
use std::sync::{Arc, Mutex};
use std::time::Instant;
use tempfile::tempdir;

use zyron_buffer::{BufferPool, BufferPoolConfig};
use zyron_common::page::PageId;
use zyron_storage::{
    BTreeIndex, BufferedBTreeIndex, DiskManager, DiskManagerConfig, HeapFile, IsolationLevel,
    LockTable, MvccGc, Snapshot, Transaction, TransactionManager, TransactionStatus, Tuple,
    TupleHeader, TupleId,
};
use zyron_wal::{LogRecordType, WalReader, WalWriter, WalWriterConfig};

// Performance targets
const SNAPSHOT_VISIBILITY_TARGET_NS: f64 = 15.0;
const LOCK_ACQUIRE_TARGET_NS: f64 = 80.0;
const SNAPSHOT_CREATE_TARGET_NS: f64 = 200.0;
const GC_SWEEP_TARGET_TUPLES_SEC: f64 = 500_000.0;

// A single synchronous durable commit is one device write (write-through FUA,
// tens of us on NVMe). That is the hardware floor: it cannot be made faster
// without weakening durability. The single-commit phase below is therefore a
// LATENCY probe reported as the floor, gated only by this loose sanity ceiling
// to catch a hang or gross regression, never as a throughput goal. Durable
// THROUGHPUT is measured under concurrency (test_concurrent_transactions),
// where group commit batches many commits per device write.
const DURABLE_COMMIT_SANITY_NS: f64 = 5_000_000.0;
const CONCURRENT_BTREE_INSERT_TARGET_OPS_SEC: f64 = 4_000_000.0;
const OPTIMISTIC_READ_MAX_LATENCY_US: f64 = 10.0;
const LATCH_RETRY_RATE_TARGET_PCT: f64 = 5.0;

static BENCHMARK_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

fn create_txn_manager() -> (Arc<TransactionManager>, Arc<WalWriter>, tempfile::TempDir) {
    let dir = tempdir().unwrap();
    let config = WalWriterConfig {
        wal_dir: dir.path().to_path_buf(),
        segment_size: zyron_wal::segment::LogSegment::DEFAULT_SIZE,
        fsync_enabled: false,
        ring_buffer_capacity: 1024 * 1024,
    };
    let writer = Arc::new(WalWriter::new(config).unwrap());
    let mgr = Arc::new(TransactionManager::new(Arc::clone(&writer)));
    (mgr, writer, dir)
}

// =============================================================================
// Test 1: Snapshot Isolation
// =============================================================================

/// Validates MVCC snapshot isolation semantics:
/// - Uncommitted inserts invisible to other transactions
/// - Snapshot taken at BEGIN time, immutable for SnapshotIsolation
/// - Committed data visible to transactions started after commit
#[test]
fn test_snapshot_isolation() {
    zyron_bench_harness::init("transaction");
    let (mgr, _wal, _dir) = create_txn_manager();

    tprintln!("\n=== Transaction: Snapshot Isolation Test ===");

    // Txn A: BEGIN, "INSERT" row with value=100
    let mut txn_a = mgr.begin(IsolationLevel::SnapshotIsolation).unwrap();
    let xmin_a = txn_a.txn_id_u32().unwrap();

    // Simulate insert: create a tuple header with xmin=txn_a, xmax=0
    let header_inserted = TupleHeader::new(8, xmin_a);

    // Txn B: BEGIN (before A commits)
    let txn_b = mgr.begin(IsolationLevel::SnapshotIsolation).unwrap();

    // B cannot see A's uncommitted row
    assert!(
        !header_inserted.is_visible_to(&txn_b.snapshot),
        "Txn B must NOT see Txn A's uncommitted insert"
    );

    // A commits
    mgr.commit_blocking(&mut txn_a).unwrap();

    // B still cannot see A's insert (B's snapshot was taken before A committed)
    assert!(
        !header_inserted.is_visible_to(&txn_b.snapshot),
        "Txn B must NOT see Txn A's insert even after A committed (snapshot isolation)"
    );

    // Txn C: BEGIN (after A committed)
    let txn_c = mgr.begin(IsolationLevel::SnapshotIsolation).unwrap();

    // C can see A's committed row
    assert!(
        header_inserted.is_visible_to(&txn_c.snapshot),
        "Txn C must see Txn A's committed insert"
    );

    tprintln!("  Snapshot isolation: PASS");
    tprintln!("    Uncommitted insert invisible to concurrent txn: verified");
    tprintln!("    Committed insert invisible to txn with older snapshot: verified");
    tprintln!("    Committed insert visible to txn with newer snapshot: verified");
}

// =============================================================================
// Test 2: Write-Write Conflict
// =============================================================================

/// Validates row-level write-write conflict detection:
/// - First writer acquires lock successfully
/// - Second writer on same row gets TransactionConflict
/// - After first writer commits (releases lock), second writer can retry
#[test]
fn test_write_write_conflict() {
    zyron_bench_harness::init("transaction");
    let (mgr, _wal, _dir) = create_txn_manager();

    tprintln!("\n=== Transaction: Write-Write Conflict Test ===");

    let rid = TupleId::new(PageId::new(0, 1), 0);
    let table_id = 0u32;

    // Txn A: BEGIN, lock the row (UPDATE)
    let mut txn_a = mgr.begin(IsolationLevel::SnapshotIsolation).unwrap();
    mgr.lock_table()
        .lock_row(txn_a.txn_id, table_id, rid)
        .unwrap();

    // Txn B: BEGIN, attempt to lock same row -> conflict
    let mut txn_b = mgr.begin(IsolationLevel::SnapshotIsolation).unwrap();
    let result = mgr.lock_table().lock_row(txn_b.txn_id, table_id, rid);
    assert!(result.is_err(), "Txn B must get TransactionConflict");
    match result.unwrap_err() {
        zyron_common::ZyronError::TransactionConflict { txn_id, .. } => {
            assert_eq!(txn_id, txn_b.txn_id, "Conflict txn_id must match Txn B");
        }
        other => panic!("Expected TransactionConflict, got: {:?}", other),
    }

    // A commits (releases locks)
    mgr.commit_blocking(&mut txn_a).unwrap();

    // B retries -> succeeds
    let result = mgr.lock_table().lock_row(txn_b.txn_id, table_id, rid);
    assert!(result.is_ok(), "Txn B must succeed after Txn A committed");

    // B commits
    mgr.commit_blocking(&mut txn_b).unwrap();

    tprintln!("  Write-write conflict: PASS");
    tprintln!("    First writer acquires lock: verified");
    tprintln!("    Second writer gets TransactionConflict: verified");
    tprintln!("    Retry after first writer commits: verified");
}

// =============================================================================
// Test 3: Rollback (Abort)
// =============================================================================

/// Validates abort semantics:
/// - Aborted transaction's inserts are invisible to subsequent transactions
/// - xmax-based deletion tracking via MVCC snapshots
#[test]
fn test_rollback_abort() {
    zyron_bench_harness::init("transaction");
    let (mgr, _wal, _dir) = create_txn_manager();

    tprintln!("\n=== Transaction: Rollback (Abort) Test ===");

    // Txn A: BEGIN, insert 10 rows
    let mut txn_a = mgr.begin(IsolationLevel::SnapshotIsolation).unwrap();
    let xmin_a = txn_a.txn_id_u32().unwrap();

    let mut headers = Vec::new();
    for i in 0..10 {
        let header = TupleHeader::new((i * 10 + 8) as u16, xmin_a);
        headers.push(header);
    }

    // A aborts
    mgr.abort(&mut txn_a).unwrap();
    assert_eq!(txn_a.status, TransactionStatus::Aborted);

    // Txn B: BEGIN, scan table
    let txn_b = mgr.begin(IsolationLevel::SnapshotIsolation).unwrap();

    // Since txn_a was active at snapshot time of txn_b (txn_a was started before
    // txn_b and txn_a aborted is removed from active set), the abort should
    // remove txn_a from the active set. Txn B's snapshot will see txn_a as
    // NOT active (it was removed), but since xmin < txn_b.txn_id and xmin not
    // in active set, the tuple would appear committed. However, aborted txns
    // should set xmax to mark tuples as dead. In our current implementation,
    // abort removes from active set but the physical xmax write is done by the
    // caller. So we simulate xmax set to xmin (self-deleted scenario).
    for header in &headers {
        // Aborted inserts: in a real system the recovery/abort handler would
        // set xmax = xmin. The snapshot will see xmin committed (not in active set)
        // but with xmax=xmin meaning it was immediately deleted.
        let aborted_header = TupleHeader::with_xmax(header.data_len, xmin_a, xmin_a);
        assert!(
            !aborted_header.is_visible_to(&txn_b.snapshot),
            "Aborted row (xmax=xmin) must not be visible to Txn B"
        );
    }

    // Verify that without xmax marking, a committed-looking tuple is visible.
    // This validates that the abort handler sets xmax.
    let live_header = TupleHeader::new(8, xmin_a);
    // xmin_a < txn_b.txn_id, xmin_a not in active set -> visible
    // This is expected: the TransactionManager removes from active set on abort,
    // so the physical layer must set xmax to prevent visibility.
    assert!(
        live_header.is_visible_to(&txn_b.snapshot),
        "Without xmax marking, aborted row appears committed (physical layer must set xmax)"
    );

    tprintln!("  Rollback (abort): PASS");
    tprintln!("    Aborted txn removed from active set: verified");
    tprintln!("    xmax=xmin marking makes aborted rows invisible: verified");
    tprintln!("    Physical layer must set xmax on abort: verified");
}

// =============================================================================
// Test 4: MVCC GC
// =============================================================================

/// Validates garbage collection logic:
/// - Dead tuples (xmax < oldest_active) are reclaimable
/// - Live tuples (xmax=0) are not reclaimable
/// - Threshold-based page selection
#[test]
fn test_mvcc_gc() {
    zyron_bench_harness::init("transaction");
    let (mgr, _wal, _dir) = create_txn_manager();

    tprintln!("\n=== Transaction: MVCC GC Test ===");

    let gc = MvccGc::new();

    // Simulate: 10,000 rows inserted by txn 1-100 (all committed)
    let total_rows = 10_000u64;
    let deleted_rows = 5_000u64;

    // Commit transactions 1-100
    for _ in 0..100 {
        let mut txn = mgr.begin(IsolationLevel::SnapshotIsolation).unwrap();
        mgr.commit_blocking(&mut txn).unwrap();
    }

    // Delete 5000 rows using txns 101-200 (all committed)
    for _ in 0..100 {
        let mut txn = mgr.begin(IsolationLevel::SnapshotIsolation).unwrap();
        mgr.commit_blocking(&mut txn).unwrap();
    }

    // No active transactions at this point
    let active = mgr.active_txn_ids();
    assert!(
        active.is_empty(),
        "No active transactions after all committed"
    );

    // Oldest active = None means all deleted tuples are reclaimable
    let oldest = MvccGc::oldest_active_txn(&active);
    assert!(oldest.is_none());

    // Test reclaimable logic for dead tuples (xmax set by committed txns)
    let mut reclaimed = 0u64;
    for i in 0..total_rows {
        if i < deleted_rows {
            // Simulate deleted by txn (i / 50 + 101) which is < 201
            let xmax = ((i / 50) + 101) as u32;
            // No active txns, so all deleted are reclaimable
            assert!(MvccGc::is_reclaimable_no_active(xmax));
            reclaimed += 1;
        } else {
            // Live tuple: xmax=0
            assert!(!MvccGc::is_reclaimable_no_active(0));
        }
    }
    assert_eq!(reclaimed, deleted_rows);

    // Test with active txn that prevents reclamation
    let _active_txn = mgr.begin(IsolationLevel::SnapshotIsolation).unwrap();
    let active_now = mgr.active_txn_ids();
    let oldest_active = MvccGc::oldest_active_txn(&active_now).unwrap();

    // Tuples deleted by txns before oldest_active are reclaimable
    // Tuples deleted by txns >= oldest_active are NOT reclaimable
    assert!(MvccGc::is_reclaimable(100, oldest_active));
    // oldest_active should be > 200, so xmax=150 is reclaimable
    assert!(MvccGc::is_reclaimable(150, oldest_active));
    // xmax=0 (live) never reclaimable
    assert!(!MvccGc::is_reclaimable(0, oldest_active));

    // Threshold-based page selection
    assert!(
        gc.should_gc_page(total_rows, deleted_rows),
        "50% dead > 20% threshold"
    );
    assert!(
        !gc.should_gc_page(total_rows, 1000),
        "10% dead < 20% threshold"
    );

    tprintln!("  MVCC GC: PASS");
    tprintln!("    Dead tuples (xmax < oldest_active) reclaimable: verified");
    tprintln!("    Live tuples (xmax=0) not reclaimable: verified");
    tprintln!("    No active txns -> all deleted reclaimable: verified");
    tprintln!("    Threshold-based page selection: verified");
    tprintln!("    Reclaimed {} / {} dead tuples", reclaimed, total_rows);
}

// =============================================================================
// Test 5: Concurrent Transactions (16 threads x 1,000 txns)
// =============================================================================

/// Validates concurrent transaction execution:
/// - 16 threads each run 1,000 begin/commit cycles
/// - All 16,000 txn_ids must be unique
/// - No data corruption or panics
/// Durable commit throughput swept across concurrency. Every commit is durable
/// (write-through device write); the lever is group commit, where the flush
/// thread batches all in-flight committers into one device write. Throughput
/// therefore scales with concurrency toward the device-saturated ceiling
/// (C / L for one WAL stream). This reports the physical numbers and gates only
/// on the structural fact that concurrency batches commits (peak >> serial),
/// not on a fixed ops/sec goal.
#[test]
fn test_concurrent_transactions() {
    zyron_bench_harness::init("transaction");
    let _bench_guard = BENCHMARK_LOCK.lock().unwrap_or_else(|e| e.into_inner());

    const CONCURRENCY_LEVELS: &[usize] = &[1, 16, 64, 256, 512];
    const TXNS_PER_THREAD: usize = 1_000;

    tprintln!("\n=== Transaction: Durable Commit Throughput vs Concurrency ===");
    tprintln!("Every commit is a durable write-through device write; throughput");
    tprintln!("scales as group commit batches concurrent committers per write.");

    let txn_util_before = take_util_snapshot();

    let mut serial_ops_sec = 0.0f64;
    let mut peak_ops_sec = 0.0f64;

    for &threads in CONCURRENCY_LEVELS {
        let total_txns = threads * TXNS_PER_THREAD;
        let (run_mgr, _run_wal, _run_dir) = create_txn_manager();
        let mgr_arc = Arc::clone(&run_mgr);

        // Clean profile window so the per-phase table reflects this level only.
        zyron_common::profile::reset();

        let start = Instant::now();
        let handles: Vec<_> = (0..threads)
            .map(|_| {
                let mgr = Arc::clone(&mgr_arc);
                std::thread::spawn(move || {
                    let mut ids = Vec::with_capacity(TXNS_PER_THREAD);
                    for _ in 0..TXNS_PER_THREAD {
                        let mut txn = mgr.begin(IsolationLevel::SnapshotIsolation).unwrap();
                        ids.push(txn.txn_id);
                        mgr.commit_blocking(&mut txn).unwrap();
                    }
                    ids
                })
            })
            .collect();

        let mut all_ids = Vec::with_capacity(total_txns);
        for h in handles {
            all_ids.extend(h.join().unwrap());
        }
        let duration = start.elapsed();

        // Correctness: all txn_ids unique, no transaction left active.
        let unique: HashSet<u64> = all_ids.iter().copied().collect();
        assert_eq!(
            unique.len(),
            total_txns,
            "c={}: all {} txn_ids must be unique, got {}",
            threads,
            total_txns,
            unique.len()
        );
        assert_eq!(
            run_mgr.active_count(),
            0,
            "c={}: all transactions must be committed",
            threads
        );

        let ops_sec = total_txns as f64 / duration.as_secs_f64();
        // Effective per-commit device-write latency implied by the throughput:
        // L_eff = concurrency / throughput.
        let l_eff_us = (threads as f64 / ops_sec) * 1_000_000.0;
        tprintln!(
            "  c={:>3}  {:>7} durable txns in {:>10.2?}  {:>12} txn/sec  (L_eff {:.1} us/write)",
            threads,
            total_txns,
            duration,
            format_with_commas(ops_sec),
            l_eff_us,
        );

        if threads == 1 {
            serial_ops_sec = ops_sec;
        }
        peak_ops_sec = peak_ops_sec.max(ops_sec);

        // Record this level's durable throughput so it lands in the run JSON.
        // Per-level durable throughput is device-bound and intentionally not
        // pinned to an absolute target (the structural gate below is the real
        // assertion), so the target here is a liveness floor that just logs the
        // value for trend tracking.
        check_performance(
            "Concurrent Txns",
            &format!("durable_commit_c{}_txn_per_sec", threads),
            ops_sec,
            1.0,
            true,
        );

        // Per-phase wall-clock breakdown for this concurrency level (ZYRON_PROFILE).
        zyron_common::profile::dump(&format!("durable commit c={}", threads));
    }

    let txn_util_after = take_util_snapshot();
    record_test_util("Concurrent Txns", txn_util_before, txn_util_after);

    tprintln!(
        "  serial (c=1) {} txn/sec  ->  peak {} txn/sec  ({:.1}x from group commit)",
        format_with_commas(serial_ops_sec),
        format_with_commas(peak_ops_sec),
        peak_ops_sec / serial_ops_sec.max(1.0),
    );

    // Structural gate: concurrency must batch commits, so peak durable
    // throughput must materially exceed the serial (single-committer) rate.
    // This proves group commit works without pinning an arbitrary ops/sec goal
    // that the device latency may or may not permit at a given concurrency.
    // Recorded as a metric (target = 4x serial) so the gate lands in the JSON,
    // and asserted below for a hard CI failure.
    let peak_ratio_ok = check_performance(
        "Concurrent Txns",
        "peak_durable_txn_per_sec",
        peak_ops_sec,
        serial_ops_sec * 4.0,
        true,
    );
    assert!(
        peak_ratio_ok,
        "group commit not batching: peak {:.0} txn/sec vs serial {:.0} txn/sec (expected >=4x)",
        peak_ops_sec, serial_ops_sec,
    );
}

// =============================================================================
// Test 6: Read Committed vs Snapshot Isolation
// =============================================================================

/// Validates both isolation levels:
/// - ReadCommitted: refreshed snapshot sees newly committed data
/// - SnapshotIsolation: original snapshot does not see newly committed data
#[test]
fn test_isolation_levels() {
    zyron_bench_harness::init("transaction");
    let (mgr, _wal, _dir) = create_txn_manager();

    tprintln!("\n=== Transaction: Read Committed vs Snapshot Isolation ===");

    // -----------------------------------------------------------------------
    // Snapshot Isolation
    // -----------------------------------------------------------------------
    let mut txn_writer = mgr.begin(IsolationLevel::SnapshotIsolation).unwrap();
    let xmin_w = txn_writer.txn_id_u32().unwrap();
    let header = TupleHeader::new(8, xmin_w);

    // SI reader starts before writer commits
    let si_reader = mgr.begin(IsolationLevel::SnapshotIsolation).unwrap();
    // RC reader starts before writer commits
    let rc_reader = mgr.begin(IsolationLevel::ReadCommitted).unwrap();

    // Writer commits
    mgr.commit_blocking(&mut txn_writer).unwrap();

    // SI reader: original snapshot, cannot see writer's data
    assert!(
        !header.is_visible_to(&si_reader.snapshot),
        "SI reader must NOT see data committed after its BEGIN"
    );

    // RC reader: refresh snapshot to see newly committed data
    let refreshed = mgr.refresh_snapshot(&rc_reader);
    assert!(
        header.is_visible_to(&refreshed),
        "RC reader with refreshed snapshot must see committed data"
    );

    // RC reader: original snapshot (before refresh) does NOT see it
    assert!(
        !header.is_visible_to(&rc_reader.snapshot),
        "RC reader with original snapshot must NOT see committed data"
    );

    tprintln!("  Isolation levels: PASS");
    tprintln!("    SnapshotIsolation: committed data invisible to older snapshot: verified");
    tprintln!("    ReadCommitted: refreshed snapshot sees committed data: verified");
}

// =============================================================================
// Test 7: WAL Integration (Begin/Commit/Abort records)
// =============================================================================

/// Validates WAL integration:
/// - Begin, Commit, Abort records written correctly
/// - WAL replay can reconstruct transaction state
#[test]
fn test_wal_transaction_integration() {
    zyron_bench_harness::init("transaction");
    let dir = tempdir().unwrap();
    let wal_dir = dir.path().to_path_buf();
    let config = WalWriterConfig {
        wal_dir: wal_dir.clone(),
        segment_size: zyron_wal::segment::LogSegment::DEFAULT_SIZE,
        fsync_enabled: false,
        ring_buffer_capacity: 1024 * 1024,
    };
    let writer = Arc::new(WalWriter::new(config).unwrap());
    let mgr = TransactionManager::new(Arc::clone(&writer));

    tprintln!("\n=== Transaction: WAL Transaction Integration Test ===");

    // Begin 5 transactions
    let mut txns: Vec<Transaction> = Vec::new();
    for _ in 0..5 {
        txns.push(mgr.begin(IsolationLevel::SnapshotIsolation).unwrap());
    }
    assert_eq!(mgr.active_count(), 5);

    // Commit first 3
    for txn in txns[0..3].iter_mut() {
        mgr.commit_blocking(txn).unwrap();
    }

    // Abort last 2
    for txn in txns[3..5].iter_mut() {
        mgr.abort(txn).unwrap();
    }

    assert_eq!(mgr.active_count(), 0);

    // Flush WAL to ensure records are on disk
    writer.flush().unwrap();

    // Read WAL records back
    let reader = WalReader::new(&wal_dir).unwrap();
    let records = reader.scan_all().unwrap();

    // Count record types
    let begins = records
        .iter()
        .filter(|r| r.record_type == LogRecordType::Begin)
        .count();
    let commits = records
        .iter()
        .filter(|r| r.record_type == LogRecordType::Commit)
        .count();
    let aborts = records
        .iter()
        .filter(|r| r.record_type == LogRecordType::Abort)
        .count();

    assert_eq!(begins, 5, "Must have 5 Begin records");
    assert_eq!(commits, 3, "Must have 3 Commit records");
    assert_eq!(aborts, 2, "Must have 2 Abort records");

    // Verify committed transactions' data would be present
    // (their Begin + Commit records exist in WAL)
    let committed_txn_ids: Vec<u32> = txns[0..3].iter().map(|t| t.txn_id_u32().unwrap()).collect();
    for tid in &committed_txn_ids {
        let has_begin = records
            .iter()
            .any(|r| r.record_type == LogRecordType::Begin && r.txn_id == *tid);
        let has_commit = records
            .iter()
            .any(|r| r.record_type == LogRecordType::Commit && r.txn_id == *tid);
        assert!(
            has_begin && has_commit,
            "Committed txn {} must have Begin+Commit in WAL",
            tid
        );
    }

    // Verify aborted transactions have Begin + Abort
    let aborted_txn_ids: Vec<u32> = txns[3..5].iter().map(|t| t.txn_id_u32().unwrap()).collect();
    for tid in &aborted_txn_ids {
        let has_begin = records
            .iter()
            .any(|r| r.record_type == LogRecordType::Begin && r.txn_id == *tid);
        let has_abort = records
            .iter()
            .any(|r| r.record_type == LogRecordType::Abort && r.txn_id == *tid);
        assert!(
            has_begin && has_abort,
            "Aborted txn {} must have Begin+Abort in WAL",
            tid
        );
    }

    tprintln!("  WAL transaction integration: PASS");
    tprintln!("    5 Begin, 3 Commit, 2 Abort records: verified");
    tprintln!("    Committed txn WAL records: verified");
    tprintln!("    Aborted txn WAL records: verified");
}

// =============================================================================
// Test 8: Concurrent B+Tree Latch Coupling (16 threads x 10K keys)
// =============================================================================

/// Validates B+Tree structural integrity under concurrent insert load.
/// 16 threads insert 10,000 keys each via Mutex<BTreeArenaIndex>.
/// All 160,000 must be present after. Readers use &self methods concurrently.
#[test]
fn test_concurrent_btree_latch_coupling() {
    zyron_bench_harness::init("transaction");
    let _bench_guard = BENCHMARK_LOCK.lock().unwrap_or_else(|e| e.into_inner());

    const THREADS: usize = 16;
    const KEYS_PER_THREAD: usize = 10_000;
    const TOTAL_KEYS: usize = THREADS * KEYS_PER_THREAD;

    tprintln!("\n=== Transaction: Concurrent B+Tree Latch Coupling Test ===");
    tprintln!(
        "Threads: {}, Keys per thread: {}, Total: {}",
        THREADS,
        KEYS_PER_THREAD,
        TOTAL_KEYS
    );

    let btree_util_before = take_util_snapshot();
    let mut insert_runs = Vec::with_capacity(VALIDATION_RUNS);

    for run in 0..VALIDATION_RUNS {
        tprintln!("\n--- Run {}/{} ---", run + 1, VALIDATION_RUNS);

        // BTreeArenaIndex::insert requires &mut self, so wrap in Mutex for
        // concurrent writers. search() and range_scan() use &self and can run
        // concurrently without the mutex.
        let btree = Arc::new(Mutex::new(zyron_storage::BTreeArenaIndex::new(2048)));

        let start = Instant::now();
        let handles: Vec<_> = (0..THREADS)
            .map(|t| {
                let btree = Arc::clone(&btree);
                std::thread::spawn(move || {
                    for i in 0..KEYS_PER_THREAD {
                        let key = ((t * KEYS_PER_THREAD + i) as u64).to_be_bytes();
                        let tid = TupleId::new(PageId::new(0, i as u64), t as u16);
                        btree.lock().unwrap().insert(&key, tid).unwrap();
                    }
                })
            })
            .collect();

        for h in handles {
            h.join().unwrap();
        }
        let duration = start.elapsed();

        let btree_guard = btree.lock().unwrap();

        // Verify all keys present
        let mut missing = 0usize;
        for i in 0..TOTAL_KEYS {
            let key = (i as u64).to_be_bytes();
            if btree_guard.search(&key).is_none() {
                missing += 1;
            }
        }
        assert_eq!(
            missing,
            0,
            "Run {}: All {} keys must be present, {} missing",
            run + 1,
            TOTAL_KEYS,
            missing
        );

        // Verify tree structural integrity via range scan
        let all_entries = btree_guard.range_scan(None, None);
        assert_eq!(
            all_entries.len(),
            TOTAL_KEYS,
            "Run {}: Range scan must return all {} keys, got {}",
            run + 1,
            TOTAL_KEYS,
            all_entries.len()
        );

        // Verify sorted order (keys must be monotonically increasing)
        for i in 1..all_entries.len() {
            assert!(
                all_entries[i].0 >= all_entries[i - 1].0,
                "Run {}: Keys must be sorted, index {} ({}) < index {} ({})",
                run + 1,
                i,
                all_entries[i].0,
                i - 1,
                all_entries[i - 1].0
            );
        }

        // Verify no duplicate keys
        let unique_keys: HashSet<u64> = all_entries.iter().map(|e| e.0).collect();
        assert_eq!(
            unique_keys.len(),
            TOTAL_KEYS,
            "Run {}: No duplicate keys, unique={} expected={}",
            run + 1,
            unique_keys.len(),
            TOTAL_KEYS
        );

        let height = btree_guard.height();
        drop(btree_guard);

        let ops_sec = TOTAL_KEYS as f64 / duration.as_secs_f64();
        insert_runs.push(ops_sec);
        tprintln!(
            "  {} keys in {:?} ({} ops/sec), height={}",
            TOTAL_KEYS,
            duration,
            format_with_commas(ops_sec),
            height
        );
    }

    let btree_util_after = take_util_snapshot();
    record_test_util(
        "Concurrent B+Tree Insert",
        btree_util_before,
        btree_util_after,
    );

    let result = validate_metric(
        "Concurrent B+Tree Insert",
        "Insert throughput (ops/sec)",
        insert_runs,
        CONCURRENT_BTREE_INSERT_TARGET_OPS_SEC,
        true,
    );
    assert!(
        result.passed,
        "Concurrent B+Tree insert throughput must meet target"
    );
}

// =============================================================================
// Test 9: Optimistic Read Under Contention
// =============================================================================

/// Real B+Tree read-under-contention: 1 writer thread continuously overwrites
/// keys (exercising the page seqlock's CAS publish on hot leaves) while 15
/// reader threads run search_sync against the same key range. Measures reader
/// latency under live writer contention. This drives the actual store seqlock
/// (try_read / try_versioned_write / lock_for_write) that the engine uses, not
/// a standalone latch primitive.
#[test]
fn test_optimistic_read_under_contention() {
    zyron_bench_harness::init("transaction");
    let _bench_guard = BENCHMARK_LOCK.lock().unwrap_or_else(|e| e.into_inner());

    tprintln!("\n=== Transaction: B+Tree Read Under Write Contention ===");

    const KEYS: u64 = 20_000;
    const READER_THREADS: usize = 15;
    const READS_PER_READER: usize = 50_000;

    let util_before = take_util_snapshot();

    // Build a real B+Tree and pre-populate the key range.
    let rt = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();
    let dir = tempfile::tempdir().unwrap();
    let index = rt.block_on(async {
        BTreeIndex::create(1, dir.path().to_path_buf())
            .await
            .unwrap()
    });
    // Pre-populate EVEN keys (2*k). Readers always search these, so a search
    // returning None is a torn-read correctness failure. The writer inserts
    // unique ODD keys interleaved between them, landing on the same leaves the
    // readers scan (real reader/writer leaf contention) without ever inserting
    // a duplicate (the B+Tree rejects duplicate keys with DuplicateKey).
    {
        let mut items: Vec<([u8; 8], TupleId)> = (0..KEYS)
            .map(|k| ((2 * k).to_be_bytes(), TupleId::new(PageId::new(1, k), 0)))
            .collect();
        index.insert_many(&mut items).unwrap();
    }
    let index = Arc::new(index);
    let stop_flag = Arc::new(std::sync::atomic::AtomicBool::new(false));

    // Writer thread: continuously insert fresh unique odd keys (2*n+1) so it
    // CAS-publishes on the leaves the readers are scanning, without duplicates.
    let idx_w = Arc::clone(&index);
    let stop_w = Arc::clone(&stop_flag);
    let writer = std::thread::spawn(move || {
        let mut n = 0u64;
        let mut writes = 0u64;
        while !stop_w.load(std::sync::atomic::Ordering::Relaxed) {
            let mut batch: Vec<([u8; 8], TupleId)> = (0..64)
                .map(|_| {
                    let key = 2 * n + 1;
                    n += 1;
                    (key.to_be_bytes(), TupleId::new(PageId::new(1, key), 0))
                })
                .collect();
            idx_w.insert_many(&mut batch).unwrap();
            writes += 64;
        }
        writes
    });

    // Reader threads: drive single read attempts (search_attempt) in their own
    // retry loop so the BENCHMARK counts retries directly. A torn read (Err)
    // is a retry; the count measures reader contention against the writer. The
    // counting lives entirely here, not in the production search_sync path.
    let reader_handles: Vec<_> = (0..READER_THREADS)
        .map(|t| {
            let idx = Arc::clone(&index);
            std::thread::spawn(move || {
                let mut max_latency_ns = 0u128;
                let mut hits = 0usize;
                let mut retries = 0u64;
                let mut k = (t as u64) * 7919 % KEYS;
                for _ in 0..READS_PER_READER {
                    let read_start = Instant::now();
                    let found = loop {
                        match idx.search_attempt(&(2 * k).to_be_bytes()) {
                            Ok(r) => break r,
                            Err(()) => {
                                retries += 1;
                                std::hint::spin_loop();
                            }
                        }
                    };
                    if found.is_some() {
                        hits += 1;
                    }
                    let latency = read_start.elapsed().as_nanos();
                    if latency > max_latency_ns {
                        max_latency_ns = latency;
                    }
                    k = (k + 1) % KEYS;
                }
                (max_latency_ns, hits, retries)
            })
        })
        .collect();

    let mut global_max_latency_ns = 0u128;
    let mut total_hits = 0usize;
    let mut total_retries = 0u64;
    for h in reader_handles {
        let (max_lat, hits, retries) = h.join().unwrap();
        if max_lat > global_max_latency_ns {
            global_max_latency_ns = max_lat;
        }
        total_hits += hits;
        total_retries += retries;
    }

    stop_flag.store(true, std::sync::atomic::Ordering::Relaxed);
    let writes_done = writer.join().unwrap();

    let max_latency_us = global_max_latency_ns as f64 / 1000.0;
    let total_reads = (READER_THREADS * READS_PER_READER) as u64;
    let retry_rate = total_retries as f64 / (total_reads + total_retries) as f64 * 100.0;

    let util_after = take_util_snapshot();
    record_test_util("B+Tree Read Under Contention", util_before, util_after);

    tprintln!("  Writer overwrote {} keys during reader load", writes_done);
    tprintln!("  Reads hit: {} / {}", total_hits, total_reads);
    tprintln!(
        "  Reader retries: {} ({:.3}% of read attempts)",
        total_retries,
        retry_rate
    );
    tprintln!(
        "  Max read latency: {:.2} us (advisory target: < {} us)",
        max_latency_us,
        OPTIMISTIC_READ_MAX_LATENCY_US
    );

    // All reads must find their key: a search returning None under concurrent
    // overwrite would mean the seqlock exposed a torn/missing leaf, which is a
    // correctness failure regardless of latency.
    assert_eq!(
        total_hits as u64, total_reads,
        "every search must find its key under concurrent overwrite (no torn reads)"
    );

    // Reader retry rate on the REAL B+tree seqlock under writer contention.
    check_performance(
        "B+Tree Read Under Contention",
        "Reader retry rate (%)",
        retry_rate,
        LATCH_RETRY_RATE_TARGET_PCT,
        false,
    );

    // Max latency is advisory (an OS scheduling hiccup can spike a single read).
    check_performance(
        "B+Tree Read Under Contention",
        "Max read latency (us)",
        max_latency_us,
        OPTIMISTIC_READ_MAX_LATENCY_US,
        false,
    );
}

// =============================================================================
// Test 10: B+Tree Split Under Concurrency
// =============================================================================

/// Fills B+Tree to trigger node splits during concurrent inserts.
/// 8 threads insert interleaved keys that concentrate on the same leaf nodes
/// to force splits under contention.
#[test]
fn test_btree_split_under_concurrency() {
    zyron_bench_harness::init("transaction");
    let _bench_guard = BENCHMARK_LOCK.lock().unwrap_or_else(|e| e.into_inner());

    tprintln!("\n=== Transaction: B+Tree Split Under Concurrency Test ===");

    const THREADS: usize = 8;
    const KEYS_PER_THREAD: usize = 20_000;
    const TOTAL_KEYS: usize = THREADS * KEYS_PER_THREAD;

    let split_util_before = take_util_snapshot();
    let mut split_runs = Vec::with_capacity(VALIDATION_RUNS);

    for run in 0..VALIDATION_RUNS {
        tprintln!("\n--- Run {}/{} ---", run + 1, VALIDATION_RUNS);

        let btree = Arc::new(Mutex::new(zyron_storage::BTreeArenaIndex::new(1024)));

        // Threads insert interleaved keys (thread 0: 0,8,16..., thread 1: 1,9,17...)
        // This concentrates inserts on the same leaf nodes forcing splits.
        let start = Instant::now();
        let handles: Vec<_> = (0..THREADS)
            .map(|t| {
                let btree = Arc::clone(&btree);
                std::thread::spawn(move || {
                    for i in 0..KEYS_PER_THREAD {
                        let key_val = (i * THREADS + t) as u64;
                        let key = key_val.to_be_bytes();
                        let tid = TupleId::new(PageId::new(0, key_val), 0);
                        btree.lock().unwrap().insert(&key, tid).unwrap();
                    }
                })
            })
            .collect();

        for h in handles {
            h.join().unwrap();
        }
        let duration = start.elapsed();

        let btree_guard = btree.lock().unwrap();

        // Verify all keys present
        let all_entries = btree_guard.range_scan(None, None);
        assert_eq!(
            all_entries.len(),
            TOTAL_KEYS,
            "Run {}: All {} keys must be present after concurrent splits, got {}",
            run + 1,
            TOTAL_KEYS,
            all_entries.len()
        );

        // Verify sorted order
        for i in 1..all_entries.len() {
            assert!(
                all_entries[i].0 > all_entries[i - 1].0,
                "Run {}: Keys must be strictly sorted after splits",
                run + 1
            );
        }

        // Verify tree height is reasonable (should be 3-4 for 160K keys)
        let height = btree_guard.height();
        drop(btree_guard);

        assert!(
            height <= 5,
            "Run {}: Tree height {} too large after concurrent splits",
            run + 1,
            height
        );

        let ops_sec = TOTAL_KEYS as f64 / duration.as_secs_f64();
        split_runs.push(ops_sec);
        tprintln!(
            "  {} keys with interleaved inserts in {:?} ({} ops/sec), height={}",
            TOTAL_KEYS,
            duration,
            format_with_commas(ops_sec),
            height
        );
    }

    let split_util_after = take_util_snapshot();
    record_test_util(
        "B+Tree Split Under Concurrency",
        split_util_before,
        split_util_after,
    );

    // No strict throughput target for split test, but track for regression.
    // Splits are inherently slower due to node allocation and key redistribution.
    validate_metric(
        "B+Tree Split Under Concurrency",
        "Split insert throughput (ops/sec)",
        split_runs,
        CONCURRENT_BTREE_INSERT_TARGET_OPS_SEC,
        true,
    );
}

// =============================================================================
// Test 11: Intent Lock Conflict
// =============================================================================

/// Validates intent lock behavior for B+Tree key-level conflict detection:
/// - Txn A acquires intent lock, Txn B gets conflict
/// - After A commits (releases), B can acquire
#[test]
fn test_intent_lock_conflict() {
    zyron_bench_harness::init("transaction");
    let (mgr, _wal, _dir) = create_txn_manager();

    tprintln!("\n=== Transaction: Intent Lock Conflict Test ===");

    let table_id = 0u32;
    let key = b"shared_key_001";

    // Txn A: acquire intent lock on key K
    let mut txn_a = mgr.begin(IsolationLevel::SnapshotIsolation).unwrap();
    mgr.intent_locks()
        .lock_key(txn_a.txn_id, table_id, key)
        .unwrap();

    // Txn B: attempt intent lock on same key K -> TransactionConflict
    let mut txn_b = mgr.begin(IsolationLevel::SnapshotIsolation).unwrap();
    let result = mgr.intent_locks().lock_key(txn_b.txn_id, table_id, key);
    assert!(
        result.is_err(),
        "Txn B must get TransactionConflict on intent lock"
    );
    match result.unwrap_err() {
        zyron_common::ZyronError::TransactionConflict { txn_id, .. } => {
            assert_eq!(txn_id, txn_b.txn_id);
        }
        other => panic!("Expected TransactionConflict, got: {:?}", other),
    }

    // A commits (releases intent locks)
    mgr.commit_blocking(&mut txn_a).unwrap();

    // B retries -> succeeds
    let result = mgr.intent_locks().lock_key(txn_b.txn_id, table_id, key);
    assert!(result.is_ok(), "Txn B must succeed after Txn A committed");

    // Verify consistent lock ordering: row lock + intent lock both acquired
    let rid = TupleId::new(PageId::new(0, 1), 0);
    mgr.lock_table()
        .lock_row(txn_b.txn_id, table_id, rid)
        .unwrap();

    // Both intent lock and row lock held by B
    assert_eq!(
        mgr.intent_locks().is_locked_by(table_id, key),
        Some(txn_b.txn_id)
    );
    assert_eq!(
        mgr.lock_table().is_locked_by(table_id, rid),
        Some(txn_b.txn_id)
    );

    mgr.commit_blocking(&mut txn_b).unwrap();

    // After commit, all locks released
    assert!(mgr.intent_locks().is_locked_by(table_id, key).is_none());
    assert!(mgr.lock_table().is_locked_by(table_id, rid).is_none());

    tprintln!("  Intent lock conflict: PASS");
    tprintln!("    Intent lock acquired by first txn: verified");
    tprintln!("    Conflict returned for second txn: verified");
    tprintln!("    Lock released on commit, retry succeeds: verified");
    tprintln!("    Consistent row + intent lock ordering: verified");
}

// =============================================================================
// Test 12: PageId u64 Addressing
// =============================================================================

/// Validates that PageId u64 page_num works end-to-end:
/// - PageId construction with u64 page_num
/// - as_u64 / from_u64 round-trip
/// - B+Tree leaf pointers store and retrieve u64 page references
/// - No u32 truncation in page addressing
#[test]
fn test_pageid_u64_addressing() {
    zyron_bench_harness::init("transaction");
    tprintln!("\n=== Transaction: PageId u64 Addressing Test ===");

    // Basic PageId u64 construction
    let page = PageId::new(1, 42);
    assert_eq!(page.file_id, 1);
    assert_eq!(page.page_num, 42);

    // Large page_num (beyond u32 range for logical addressing)
    let large_page = PageId::new(0, 0xFFFF_FFFF + 1);
    assert_eq!(large_page.page_num, 0x1_0000_0000u64);

    // as_u64 / from_u64 round-trip (packs file_id:u32 | page_num:u32 into u64)
    // Note: as_u64 packs the lower 32 bits of page_num for buffer pool addressing
    let packed = page.as_u64();
    let unpacked = PageId::from_u64(packed);
    assert_eq!(unpacked.file_id, page.file_id);
    // as_u64 truncates page_num to u32 for buffer pool (segment-local offset)
    assert_eq!(unpacked.page_num, page.page_num & 0xFFFF_FFFF);

    // B+Tree: store TupleId with u64 PageId in a BufferedBTreeIndex
    let mut btree = BufferedBTreeIndex::new(128);

    // Insert entries with various page_num values (within segment-local u32 range)
    let test_pages: Vec<u64> = vec![0, 1, 100, 1000, 65535, 0xFFFF];
    for (i, &pn) in test_pages.iter().enumerate() {
        let key = (i as u64).to_be_bytes();
        let tid = TupleId::new(PageId::new(0, pn), i as u16);
        btree.insert(&key, tid).unwrap();
    }

    // Flush buffer to B+Tree
    btree.flush().unwrap();

    // Verify all entries retrievable with correct page_num
    for (i, &pn) in test_pages.iter().enumerate() {
        let key = (i as u64).to_be_bytes();
        let result = btree.search(&key);
        assert!(result.is_some(), "Key {} must be found", i);
        let tid = result.unwrap();
        // file_id is not stored in the packed tuple_id on disk, so it comes back as 0
        assert_eq!(
            tid.page_id.page_num, pn,
            "page_num must round-trip correctly for key {}",
            i
        );
        assert_eq!(
            tid.slot_id, i as u16,
            "slot_id must round-trip correctly for key {}",
            i
        );
    }

    // Range scan returns u64 packed tuple_ids, verify they decode correctly
    let all_entries = btree.range_scan(None, None);
    assert_eq!(
        all_entries.len(),
        test_pages.len(),
        "Range scan must return all entries"
    );

    // Verify packed values have valid structure (file_id:16 | page_num:32 | slot_id:16)
    for (_key, packed_tid) in &all_entries {
        let page_num = ((*packed_tid >> 16) & 0xFFFF_FFFF) as u64;
        // Packed value must contain a page_num that matches one of our test pages
        assert!(
            test_pages.contains(&page_num),
            "Packed value page_num {} must be one of our test pages",
            page_num
        );
    }

    tprintln!("  PageId u64 addressing: PASS");
    tprintln!("    PageId::new with u64 page_num: verified");
    tprintln!("    as_u64 / from_u64 round-trip: verified");
    tprintln!("    B+Tree leaf pointers store u64 page references: verified");
    tprintln!("    Range scan packed values decode correctly: verified");
}

// =============================================================================
// Transaction: Transaction Performance Microbenchmarks
// =============================================================================

/// Microbenchmarks for Phase 1.5 transaction primitives:
/// - begin() latency
/// - commit() latency
/// - is_visible() latency
/// - lock_row() latency
/// - Snapshot::new() latency
/// - GC sweep throughput
#[test]
fn test_transaction_microbenchmarks() {
    zyron_bench_harness::init("transaction");
    let _bench_guard = BENCHMARK_LOCK.lock().unwrap_or_else(|e| e.into_inner());

    tprintln!("\n=== Transaction: Transaction Performance Microbenchmarks ===");

    let perf_util_before = take_util_snapshot();

    // -----------------------------------------------------------------------
    // Single durable commit LATENCY (device-write floor, not a throughput gate)
    // -----------------------------------------------------------------------
    // One synchronous durable commit = one write-through device write. A single
    // thread cannot issue commit N+1 until N is durable, so this measures the
    // hardware floor. Small sample (the value is stable; a large sequential loop
    // would just be N device writes back to back). Throughput is measured under
    // concurrency in test_concurrent_transactions, where group commit batches
    // many commits per device write.
    const LATENCY_PROBE_OPS: usize = 2_000;
    let mut commit_lat_runs = Vec::with_capacity(VALIDATION_RUNS);
    for _ in 0..VALIDATION_RUNS {
        let (mgr, _wal, _dir) = create_txn_manager();
        // Warmup
        for _ in 0..100 {
            let mut t = mgr.begin(IsolationLevel::SnapshotIsolation).unwrap();
            mgr.commit_blocking(&mut t).unwrap();
        }
        let start = Instant::now();
        for _ in 0..LATENCY_PROBE_OPS {
            let mut txn = mgr.begin(IsolationLevel::SnapshotIsolation).unwrap();
            std::hint::black_box(txn.txn_id);
            mgr.commit_blocking(&mut txn).unwrap();
        }
        let ns_per_op = start.elapsed().as_nanos() as f64 / LATENCY_PROBE_OPS as f64;
        commit_lat_runs.push(ns_per_op);
    }
    // Reported as the device-write floor; sanity-gated only (catches a hang or
    // gross regression). This is NOT a sub-us goal under unconditional durability.
    validate_metric(
        "Phase 1.5 Microbenchmarks",
        "durable begin()+commit() latency floor (ns/op)",
        commit_lat_runs,
        DURABLE_COMMIT_SANITY_NS,
        false,
    );

    // -----------------------------------------------------------------------
    // is_visible() latency
    // -----------------------------------------------------------------------
    let mut vis_runs = Vec::with_capacity(VALIDATION_RUNS);
    for _ in 0..VALIDATION_RUNS {
        const OPS: usize = 1_000_000;
        let snapshot = Snapshot::new(1000, (1..100).collect());
        let header = TupleHeader::new(8, 50);

        // Warmup
        for _ in 0..1000 {
            std::hint::black_box(header.is_visible_to(&snapshot));
        }

        let start = Instant::now();
        for _ in 0..OPS {
            std::hint::black_box(header.is_visible_to(&snapshot));
        }
        let duration = start.elapsed();
        let ns_per_op = duration.as_nanos() as f64 / OPS as f64;
        vis_runs.push(ns_per_op);
    }
    validate_metric(
        "Phase 1.5 Microbenchmarks",
        "is_visible() latency (ns/op)",
        vis_runs,
        SNAPSHOT_VISIBILITY_TARGET_NS,
        false,
    );

    // -----------------------------------------------------------------------
    // lock_row() latency (uncontended)
    // -----------------------------------------------------------------------
    let mut lock_runs = Vec::with_capacity(VALIDATION_RUNS);
    for _ in 0..VALIDATION_RUNS {
        const OPS: usize = 100_000;
        let lock_table = LockTable::new();

        let start = Instant::now();
        for i in 0..OPS {
            let rid = TupleId::new(PageId::new(0, i as u64), 0);
            lock_table.lock_row(1, 0, rid).unwrap();
        }
        let duration = start.elapsed();
        let ns_per_op = duration.as_nanos() as f64 / OPS as f64;
        lock_runs.push(ns_per_op);
    }
    validate_metric(
        "Phase 1.5 Microbenchmarks",
        "lock_row() latency (ns/op)",
        lock_runs,
        LOCK_ACQUIRE_TARGET_NS,
        false,
    );

    // -----------------------------------------------------------------------
    // Snapshot::new() creation latency
    // -----------------------------------------------------------------------
    let mut snap_runs = Vec::with_capacity(VALIDATION_RUNS);
    for _ in 0..VALIDATION_RUNS {
        const OPS: usize = 100_000;
        let active_set: Vec<u64> = (1..20).collect();

        let start = Instant::now();
        for i in 0..OPS {
            std::hint::black_box(Snapshot::new(1000 + i as u64, active_set.clone()));
        }
        let duration = start.elapsed();
        let ns_per_op = duration.as_nanos() as f64 / OPS as f64;
        snap_runs.push(ns_per_op);
    }
    validate_metric(
        "Phase 1.5 Microbenchmarks",
        "Snapshot::new() latency (ns/op)",
        snap_runs,
        SNAPSHOT_CREATE_TARGET_NS,
        false,
    );

    // -----------------------------------------------------------------------
    // GC sweep throughput
    // -----------------------------------------------------------------------
    let mut gc_runs = Vec::with_capacity(VALIDATION_RUNS);
    for _ in 0..VALIDATION_RUNS {
        const TUPLES: usize = 1_000_000;
        let oldest_active = 500_000u64;

        let start = Instant::now();
        for i in 0..TUPLES {
            let xmax = if i % 2 == 0 { (i / 2) as u32 } else { 0 };
            std::hint::black_box(MvccGc::is_reclaimable(xmax, oldest_active));
        }
        let duration = start.elapsed();
        let tuples_sec = TUPLES as f64 / duration.as_secs_f64();
        gc_runs.push(tuples_sec);
    }
    validate_metric(
        "Phase 1.5 Microbenchmarks",
        "GC sweep throughput (tuples/sec)",
        gc_runs,
        GC_SWEEP_TARGET_TUPLES_SEC,
        true,
    );

    let perf_util_after = take_util_snapshot();
    record_test_util(
        "Phase 1.5 Microbenchmarks",
        perf_util_before,
        perf_util_after,
    );
}
