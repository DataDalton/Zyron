//! Heap pages put back from the log after a crash.
//!
//! A heap with a log attached records every page change it makes. What
//! these prove is that rows written, stamped, rewritten, freed and vacuumed
//! before the process died, with none of their pages flushed, come back
//! exactly from the log, that a page already on disk is not replayed over,
//! and that replaying twice leaves the same pages once.
//!
//! Run: `cargo test -p zyron-storage --test heap_redo_test`

use std::path::Path;
use std::sync::Arc;

use zyron_buffer::{BufferPool, BufferPoolConfig};
use zyron_storage::heap_redo::{RedoStats, apply_page_records, log_vacuum};
use zyron_storage::{
    DiskManager, DiskManagerConfig, HeapFile, HeapFileConfig, HeapPage, PageVacuum, Tuple, TupleId,
};
use zyron_wal::{RecoveryManager, WalWriter, WalWriterConfig};

const HEAP_FILE: u32 = 41;
const FSM_FILE: u32 = 42;

/// Wide enough that a handful of rows fill a page, so a batch spans pages
const ROW_BYTES: usize = 3_000;

struct Engine {
    disk: Arc<DiskManager>,
    pool: Arc<BufferPool>,
    wal: Arc<WalWriter>,
}

async fn open(root: &Path) -> Engine {
    let data_dir = root.join("data");
    let wal_dir = root.join("wal");
    std::fs::create_dir_all(&data_dir).expect("data dir");
    let disk = Arc::new(
        DiskManager::new(DiskManagerConfig {
            data_dir,
            fsync_enabled: false,
            ..Default::default()
        })
        .await
        .expect("disk"),
    );
    let pool = Arc::new(BufferPool::new(BufferPoolConfig { num_frames: 256 }));
    let wal = Arc::new(
        WalWriter::new(WalWriterConfig {
            wal_dir,
            segment_size: 1024 * 1024,
            fsync_enabled: false,
            ring_buffer_capacity: 1024 * 1024,
        })
        .expect("wal"),
    );
    Engine { disk, pool, wal }
}

async fn heap(engine: &Engine, logged: bool) -> HeapFile {
    let heap = HeapFile::new(
        Arc::clone(&engine.disk),
        Arc::clone(&engine.pool),
        HeapFileConfig {
            heap_file_id: HEAP_FILE,
            fsm_file_id: FSM_FILE,
        },
    )
    .expect("heap");
    if logged {
        heap.attach_wal(&engine.wal);
    }
    heap.init_cache().await.expect("init cache");
    heap
}

fn row(tag: u8, xmin: u64) -> Tuple {
    Tuple::with_epoch(vec![tag; ROW_BYTES], xmin, 1)
}

/// Every live row as (tag, xmin, xmax), ordered by position
fn live_rows(heap: &HeapFile) -> Vec<(u8, u64, u64)> {
    let scan = heap.scan().expect("scan");
    let mut rows = Vec::new();
    scan.for_each(|_, view| {
        rows.push((view.data[0], view.header.xmin, view.header.xmax));
    });
    rows
}

/// Recovers the log at `root` into a fresh engine and reports what the
/// replay did
async fn recover(root: &Path) -> (Engine, RedoStats) {
    let result = RecoveryManager::new(&root.join("wal"))
        .expect("recovery manager")
        .recover()
        .expect("recover");
    let engine = open(root).await;
    let stats = apply_page_records(
        &engine.disk,
        &engine.pool,
        &result.page_records,
        zyron_storage::heap_redo::BadPageRecord::Stop,
    )
    .await
    .expect("replay");
    (engine, stats)
}

#[tokio::test]
async fn rows_written_before_a_crash_come_back_from_the_log() {
    let dir = tempfile::TempDir::new().expect("temp dir");
    let expected = {
        let engine = open(dir.path()).await;
        let heap = heap(&engine, true).await;

        // Three bursts across several pages, then every kind of change
        let first: Vec<Tuple> = (1..=6).map(|tag| row(tag, 10)).collect();
        let ids = heap.insert_batch(&first).await.expect("append");
        let second: Vec<Tuple> = (7..=9).map(|tag| row(tag, 11)).collect();
        let more = heap.insert_batch(&second).await.expect("append");
        assert!(
            ids.iter()
                .chain(more.iter())
                .any(|id| id.page_id != ids[0].page_id),
            "the rows span more than one page"
        );
        // Stamped as deleted by a later transaction
        heap.mark_deleted_batch(&ids[..2], 12, 0, None, false)
            .await
            .expect("stamp");
        // One stamp set then cleared, as a rollback to a savepoint does
        heap.set_xmax(ids[2], 13).await.expect("set");
        heap.clear_xmax(ids[2]).await.expect("clear");
        // One row rewritten in place inside its footprint
        heap.update(ids[3], &row(43, 10)).await.expect("update");
        // One slot freed outright
        assert!(heap.delete(more[0]).await.expect("free"));
        // A vacuum pass over the first page, logged the way the vacuum
        // worker logs it. The rows the aborted transaction wrote are
        // reclaimed, so the rows stamped by it read as live again
        let page_id = ids[0].page_id;
        let frame = engine.pool.fetch_page(page_id).expect("resident");
        let mut changes = PageVacuum::default();
        {
            let mut guard = frame.write_data();
            let is_dead = |xmin: u64, _xmax: u64| xmin == 11;
            let is_aborted = |xid: u64| xid == 12;
            HeapPage::vacuum_in_slice(&mut guard[..], &is_dead, &is_aborted, &mut changes);
            log_vacuum(&engine.wal, &engine.pool, page_id, &changes).expect("log the vacuum");
        }
        engine.pool.unpin_page(page_id, true);

        engine.wal.flush().expect("the log is durable");
        let expected = live_rows(&heap);
        // The process dies here with every change in memory. The heap and
        // the pool go without a flush, and the writer's drop drains what
        // the log still holds in memory
        drop(heap);
        drop(engine);
        expected
    };
    assert!(!expected.is_empty());
    assert!(
        expected.iter().any(|&(tag, _, _)| tag == 43),
        "{expected:?}"
    );
    assert!(
        expected.iter().all(|&(tag, _, _)| tag != 7),
        "the freed row is gone: {expected:?}"
    );
    assert!(
        expected.iter().any(|&(tag, _, xmax)| tag == 3 && xmax == 0),
        "the cleared stamp reads as live: {expected:?}"
    );

    let (engine, stats) = recover(dir.path()).await;
    assert!(stats.applied > 0, "{stats:?}");
    assert_eq!(stats.already_held, 0, "{stats:?}");
    assert_eq!(stats.dropped, 0, "{stats:?}");
    let heap = heap(&engine, false).await;
    assert_eq!(live_rows(&heap), expected);
}

#[tokio::test]
async fn a_page_already_on_disk_is_not_replayed_over() {
    let dir = tempfile::TempDir::new().expect("temp dir");
    let expected = {
        let engine = open(dir.path()).await;
        let heap = heap(&engine, true).await;
        let first: Vec<Tuple> = (1..=4).map(|tag| row(tag, 20)).collect();
        heap.insert_batch(&first).await.expect("append");
        // Every page reaches disk stamped with the newest change it holds
        engine.wal.flush().expect("durable");
        heap.flush().await.expect("flush");
        // Then more rows land on the same and later pages and never reach
        // disk
        let second: Vec<Tuple> = (5..=7).map(|tag| row(tag, 21)).collect();
        heap.insert_batch(&second).await.expect("append");
        engine.wal.flush().expect("durable");
        let expected = live_rows(&heap);
        // The process dies here. The heap and the pool go without a flush,
        // and the writer's drop drains what the log still holds in memory
        drop(heap);
        drop(engine);
        expected
    };

    let (engine, stats) = recover(dir.path()).await;
    assert!(
        stats.already_held > 0,
        "the flushed pages hold their records: {stats:?}"
    );
    assert!(stats.applied > 0, "{stats:?}");
    let heap = heap(&engine, false).await;
    assert_eq!(live_rows(&heap), expected);
}

#[tokio::test]
async fn replaying_twice_leaves_the_same_pages_once() {
    let dir = tempfile::TempDir::new().expect("temp dir");
    let expected = {
        let engine = open(dir.path()).await;
        let heap = heap(&engine, true).await;
        let rows: Vec<Tuple> = (1..=5).map(|tag| row(tag, 30)).collect();
        let ids: Vec<TupleId> = heap.insert_batch(&rows).await.expect("append");
        heap.mark_deleted_batch(&ids[..1], 31, 0, None, false)
            .await
            .expect("stamp");
        engine.wal.flush().expect("durable");
        let expected = live_rows(&heap);
        // The process dies here. The heap and the pool go without a flush,
        // and the writer's drop drains what the log still holds in memory
        drop(heap);
        drop(engine);
        expected
    };

    let result = RecoveryManager::new(&dir.path().join("wal"))
        .expect("recovery manager")
        .recover()
        .expect("recover");
    let engine = open(dir.path()).await;
    let first = apply_page_records(
        &engine.disk,
        &engine.pool,
        &result.page_records,
        zyron_storage::heap_redo::BadPageRecord::Stop,
    )
    .await
    .expect("replay");
    let second = apply_page_records(
        &engine.disk,
        &engine.pool,
        &result.page_records,
        zyron_storage::heap_redo::BadPageRecord::Stop,
    )
    .await
    .expect("replay again");
    assert!(first.applied > 0);
    assert_eq!(second.applied, 0, "{second:?}");
    assert_eq!(second.already_held, first.applied, "{second:?}");
    let heap = heap(&engine, false).await;
    assert_eq!(live_rows(&heap), expected);
}
