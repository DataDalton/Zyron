//! Commit status must be observable before the committer's locks release.
//!
//! A waiter that wins a released row lock immediately consults the status
//! map: the FOR UPDATE recheck decides whether the row changed under it and
//! the unique probe decides whether a competing key committed. If the lock
//! releases first and the status publishes after, the waiter reads the
//! committer as still in flight, accepts a stale image under an exclusive
//! lock, and a unique check admits a duplicate committed key.
//!
//! The window is a handful of instructions inside commit, so a spinning
//! waiter on another core is what makes it observable at all. Every
//! iteration that wins the lock must find the releaser already committed.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use zyron_common::RowLocator;
use zyron_common::page::PageId;

use super::{IsolationLevel, LockMode, TransactionManager};

/// A transaction below the id fence is always visible in the active set.
///
/// The vacuum horizon is computed as (oldest active, capped by the id
/// fence). begin used to allocate the id first and claim the proc-array
/// slot after, so a sampler landing between the two saw the fence advanced
/// past a transaction it could not see, and pruning could reclaim versions
/// whose deleter later aborts. The invariant checked here: an id observed
/// active now and below a fence sampled earlier must already have been
/// active at that earlier sample.
#[test]
fn test_beginning_txn_is_never_above_the_fence_and_invisible() {
    let dir = tempfile::TempDir::new().expect("temp dir");
    let config = zyron_wal::WalWriterConfig {
        wal_dir: dir.path().to_path_buf(),
        ..Default::default()
    };
    let writer = Arc::new(zyron_wal::WalWriter::new(config).expect("wal"));
    let mgr = Arc::new(TransactionManager::new(Arc::clone(&writer)));

    let stop = Arc::new(AtomicBool::new(false));
    let churn_mgr = Arc::clone(&mgr);
    let churn_stop = Arc::clone(&stop);
    let churn = std::thread::spawn(move || {
        while !churn_stop.load(Ordering::Relaxed) {
            let mut t = churn_mgr
                .begin(IsolationLevel::ReadCommitted)
                .expect("begin");
            let _ = churn_mgr.commit_read_only(&mut t);
        }
    });

    let mut prev_fence = 0u64;
    let mut prev_active: Vec<u64> = Vec::new();
    for _ in 0..200_000 {
        let fence = mgr.next_txn_id();
        let active = mgr.proc_array_shared().active_txn_ids();
        for &id in &active {
            if id < prev_fence && !prev_active.contains(&id) {
                stop.store(true, Ordering::Relaxed);
                let _ = churn.join();
                panic!(
                    "txn {id} is active and below the earlier fence {prev_fence}, but was invisible then, a horizon computed at that moment sat above it"
                );
            }
        }
        prev_fence = fence;
        prev_active = active;
    }
    stop.store(true, Ordering::Relaxed);
    churn.join().expect("churn thread");
}

#[test]
fn test_commit_status_is_visible_before_locks_release() {
    let dir = tempfile::TempDir::new().expect("temp dir");
    let config = zyron_wal::WalWriterConfig {
        wal_dir: dir.path().to_path_buf(),
        ..Default::default()
    };
    let writer = Arc::new(zyron_wal::WalWriter::new(config).expect("wal"));
    let mgr = Arc::new(TransactionManager::new(Arc::clone(&writer)));

    let locator = RowLocator::Heap {
        page: PageId::new(1, 0),
        slot: 0,
    };
    const ITERATIONS: usize = 20_000;
    for i in 0..ITERATIONS {
        let mut t1 = mgr.begin(IsolationLevel::ReadCommitted).expect("begin t1");
        let t1_id = t1.txn_id;
        mgr.lock_table()
            .lock_row(t1_id, 1, locator, LockMode::Exclusive)
            .expect("t1 locks the row");

        let spinner_mgr = Arc::clone(&mgr);
        let go = Arc::new(AtomicBool::new(false));
        let go2 = Arc::clone(&go);
        let waiter = std::thread::spawn(move || {
            let t2 = spinner_mgr
                .begin(IsolationLevel::ReadCommitted)
                .expect("begin t2");
            let t2_id = t2.txn_id;
            go2.store(true, Ordering::Release);
            // Spin until the released lock is won, then read the status in
            // the very next instruction, which is what the FOR UPDATE
            // recheck and the unique probe both do
            loop {
                if spinner_mgr
                    .lock_table()
                    .lock_row_or_holder(t2_id, 1, locator, LockMode::Exclusive)
                    .is_ok()
                {
                    let committed = spinner_mgr.status_map().is_committed(t1_id);
                    spinner_mgr.lock_table().unlock_all(t2_id);
                    let mut t2 = t2;
                    let _ = spinner_mgr.commit_read_only(&mut t2);
                    return committed;
                }
                std::hint::spin_loop();
            }
        });

        while !go.load(Ordering::Acquire) {
            std::hint::spin_loop();
        }
        mgr.commit_blocking(&mut t1).expect("commit t1");

        let committed = waiter.join().expect("waiter thread");
        assert!(
            committed,
            "iteration {i}: the waiter won txn {t1_id}'s lock before its commit status published"
        );
    }
}
