//! Integration test: an aborted transaction's inserted rows are invisible to
//! later snapshots, while a committed transaction's rows are visible. This
//! exercises the commit-status map through the real heap + transaction path.

use std::sync::Arc;

use zyron_buffer::{BufferPool, BufferPoolConfig};
use zyron_storage::txn::IsolationLevel;
use zyron_storage::{DiskManager, DiskManagerConfig, HeapFile, TransactionManager, Tuple};
use zyron_wal::{WalWriter, WalWriterConfig};

async fn setup() -> (
    Arc<DiskManager>,
    Arc<BufferPool>,
    Arc<TransactionManager>,
    tempfile::TempDir,
) {
    let dir = tempfile::tempdir().unwrap();
    let disk = Arc::new(
        DiskManager::new(zyron_bench_harness::disk_config(dir.path().join("data")))
            .await
            .unwrap(),
    );
    let pool = Arc::new(BufferPool::new(zyron_bench_harness::buffer_pool_config()));
    let wal =
        Arc::new(WalWriter::new(zyron_bench_harness::wal_config(dir.path().join("wal"))).unwrap());
    std::fs::create_dir_all(dir.path().join("wal")).unwrap();
    let txnm = Arc::new(TransactionManager::new(wal));
    (disk, pool, txnm, dir)
}

#[tokio::test]
async fn aborted_insert_invisible_committed_insert_visible() {
    let (disk, pool, txnm, _dir) = setup().await;
    let heap = HeapFile::with_defaults(disk, pool).unwrap();

    // Transaction A inserts a row, then aborts. No physical undo runs.
    let mut a = txnm.begin(IsolationLevel::ReadCommitted).unwrap();
    let a_tid = heap
        .insert_batch(&[Tuple::new(b"aborted-row".to_vec(), a.txn_id as u32)])
        .await
        .unwrap()
        .remove(0);
    txnm.abort(&mut a).unwrap();

    // Transaction C inserts a row, then commits.
    let mut c = txnm.begin(IsolationLevel::ReadCommitted).unwrap();
    let c_tid = heap
        .insert_batch(&[Tuple::new(b"committed-row".to_vec(), c.txn_id as u32)])
        .await
        .unwrap()
        .remove(0);
    txnm.commit_blocking(&mut c).unwrap();

    // A later reader must see C's row and must NOT see A's aborted row, even
    // though both rows are physically present in the heap.
    let reader = txnm.begin(IsolationLevel::ReadCommitted).unwrap();

    let a_tuple = heap
        .get(a_tid)
        .await
        .unwrap()
        .expect("row physically present");
    assert!(
        !reader
            .snapshot
            .is_visible(a_tuple.header().xmin as u64, a_tuple.header().xmax as u64),
        "aborted transaction's insert must be invisible"
    );

    let c_tuple = heap
        .get(c_tid)
        .await
        .unwrap()
        .expect("row physically present");
    assert!(
        reader
            .snapshot
            .is_visible(c_tuple.header().xmin as u64, c_tuple.header().xmax as u64),
        "committed transaction's insert must be visible"
    );
}

#[tokio::test]
async fn status_survives_reload_via_persisted_clog() {
    // The commit-status map persists and reloads, so an aborted transaction's
    // status is known after a restart even without replaying its WAL.
    let dir = tempfile::tempdir().unwrap();
    let data_dir = dir.path().to_path_buf();
    {
        let m = zyron_storage::TxnStatusMap::new();
        m.record_committed(7);
        m.record_aborted(9);
        m.persist(&data_dir).unwrap();
    }
    let reloaded = zyron_storage::TxnStatusMap::new();
    reloaded.load(&data_dir).unwrap();
    assert!(reloaded.is_committed(7));
    assert!(reloaded.is_aborted(9));
    assert!(!reloaded.is_committed(9));
}
