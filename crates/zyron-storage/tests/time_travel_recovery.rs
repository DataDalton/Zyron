//! Crash-recovery for time-travel commit-LSN dating.
//!
//! Time-travel dates each transaction by the LSN of its commit record. After a
//! crash that state is rebuilt from two sources: the commit-status map persisted
//! at the last checkpoint, and the WAL redo tail for transactions that committed
//! after it. This test drives the real path (real commits, real persist/load,
//! real RecoveryManager::recover, real record_committed_at) and asserts that
//! commit_lsn and version visibility survive a crash for both a checkpointed
//! transaction and a post-checkpoint one.
//!
//! Run: cargo test -p zyron-storage --test time_travel_recovery -- --nocapture

use std::sync::Arc;

use zyron_storage::txn::{IsolationLevel, TransactionManager};
use zyron_storage::{RetentionClock, TxnStatusMap};
use zyron_wal::{RecoveryManager, WalWriter, WalWriterConfig};

#[test]
fn commit_lsn_dating_survives_crash_for_checkpointed_and_redo_tail_txns() {
    let tmp = tempfile::TempDir::new().expect("tmp");
    let wal_dir = tmp.path().join("wal");
    let data_dir = tmp.path().join("data");
    std::fs::create_dir_all(&wal_dir).unwrap();
    std::fs::create_dir_all(&data_dir).unwrap();

    // --- Live operation before the crash ---
    let wal = Arc::new(
        WalWriter::new(WalWriterConfig {
            wal_dir: wal_dir.clone(),
            segment_size: 16 * 1024 * 1024,
            fsync_enabled: false,
            ring_buffer_capacity: 1024 * 1024,
        })
        .expect("wal"),
    );
    let txns = Arc::new(TransactionManager::new(Arc::clone(&wal)));
    let status = txns.status_map();
    // A retained version exists, so transactions are dated by commit LSN.
    status.enable_lsn_tracking();

    // Transaction A commits, then a checkpoint persists the commit-status map.
    let mut a = txns.begin(IsolationLevel::ReadCommitted).expect("begin a");
    let a_id = a.txn_id;
    txns.commit_blocking(&mut a).expect("commit a");
    let a_lsn = status.commit_lsn(a_id).expect("a committed");
    assert!(a_lsn > 0, "a has a real commit LSN");

    // Checkpoint: persist the commit-status map and the retention clock. The
    // clock samples the durable LSN at a wall-clock time.
    let clock = txns.retention_clock();
    let ckpt_lsn = wal.flushed_lsn().0;
    clock.record(1_000, ckpt_lsn);
    status.persist(&data_dir).expect("persist clog");
    clock.persist(&data_dir).expect("persist clock");

    // Transaction B commits AFTER the checkpoint, so it is only in the WAL redo
    // tail, not in the persisted commit-status map.
    let mut b = txns.begin(IsolationLevel::ReadCommitted).expect("begin b");
    let b_id = b.txn_id;
    txns.commit_blocking(&mut b).expect("commit b");
    let b_lsn = status.commit_lsn(b_id).expect("b committed");
    assert!(b_lsn > a_lsn, "b committed after a");

    // --- Crash: drop the manager and WAL writer without a clean shutdown ---
    drop(txns);
    drop(wal);

    // --- Recovery, mirroring the server startup sequence ---
    let recovered = Arc::new(TxnStatusMap::new());
    // Load the checkpointed commit-status map (restores tracking + the dawn).
    recovered.load(&data_dir).expect("load clog");
    assert!(recovered.lsn_tracking_enabled(), "tracking restored");
    // Replay the WAL redo tail and re-date every committed transaction.
    let recovery = RecoveryManager::new(&wal_dir).expect("recovery mgr");
    let result = recovery.recover().expect("recover");
    for &(txn_id, commit_lsn) in &result.committed_txns {
        recovered.record_committed_at(txn_id as u64, commit_lsn);
    }
    let recovered_clock = RetentionClock::new();
    recovered_clock.load(&data_dir).expect("load clock");

    // --- Assertions ---
    // A (checkpointed) keeps its commit LSN from the persisted map.
    assert_eq!(
        recovered.commit_lsn(a_id),
        Some(a_lsn),
        "checkpointed transaction's commit LSN survives recovery"
    );
    // B (post-checkpoint) is rebuilt from the WAL redo tail.
    assert_eq!(
        recovered.commit_lsn(b_id),
        Some(b_lsn),
        "post-checkpoint transaction's commit LSN is rebuilt from the WAL"
    );
    // The retention clock survives, so a window still maps to a floor LSN.
    assert_eq!(
        recovered_clock.lsn_at(1_000),
        ckpt_lsn,
        "the checkpoint clock sample survives recovery"
    );
    assert_eq!(
        recovered_clock.lsn_at(999),
        0,
        "no sample before the checkpoint"
    );

    // Version visibility uses the recovered dating: a version between A and B
    // sees A's row but not B's.
    let mid = (a_lsn + b_lsn) / 2;
    assert!(
        recovered.is_visible_at_version(a_id as u64, 0, mid),
        "row inserted by A is visible at a version after A committed"
    );
    assert!(
        !recovered.is_visible_at_version(b_id as u64, 0, mid),
        "row inserted by B is not visible at a version before B committed"
    );
}
