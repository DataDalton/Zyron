//! Folded rows must behave exactly like heap rows under DML.
//!
//! The columnar DELETE and UPDATE paths run the same value based subsystem
//! sequence as the heap paths, foreign key referential actions, triggers,
//! unique checks and CDC capture, with only the physical row removal
//! differing. These tests drive real plans over a really folded table and
//! assert the subsystems fire.
//!
//! Run: cargo test -p zyron-server --test columnar_dml_parity_test

use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use zyron_buffer::{BufferPool, BufferPoolConfig};
use zyron_catalog::storage::{CatalogStorage, HeapCatalogStorage};
use zyron_catalog::{Catalog, CatalogCache, DatabaseId};
use zyron_executor::context::{CdcHook, ExecutionContext};
use zyron_parser::ast::{
    ColumnDef, DataType, ReferentialAction, TableConstraint, TableConstraintKind,
};
use zyron_server::background::compaction::{CompactionWorker, CompactionWorkerConfig};
use zyron_storage::columnar::ColumnarPatchManager;
use zyron_storage::txn::{IsolationLevel, TransactionManager};
use zyron_storage::{DiskManager, DiskManagerConfig, HeapFile, HeapFileConfig, Tuple};
use zyron_wal::{WalWriter, WalWriterConfig};

fn col(name: &str, ty: DataType) -> ColumnDef {
    ColumnDef {
        name: name.to_string(),
        data_type: ty,
        nullable: Some(true),
        default: None,
        constraints: vec![],
    }
}

/// Two BigInt columns, null bitmap byte then the two values
fn encode_row2(a: i64, b: i64) -> Vec<u8> {
    let mut d = Vec::with_capacity(17);
    d.push(0u8);
    d.extend_from_slice(&a.to_le_bytes());
    d.extend_from_slice(&b.to_le_bytes());
    d
}

/// BigInt id then a vector payload, null bitmap byte, 8-byte id,
/// 4-byte length prefix and raw f32 LE bytes
fn encode_row_id_vec(id: i64, v: &[f32]) -> Vec<u8> {
    let mut d = Vec::with_capacity(13 + v.len() * 4);
    d.push(0u8);
    d.extend_from_slice(&id.to_le_bytes());
    d.extend_from_slice(&((v.len() * 4) as u32).to_le_bytes());
    for f in v {
        d.extend_from_slice(&f.to_le_bytes());
    }
    d
}

struct Env {
    catalog: Arc<Catalog>,
    txn_manager: Arc<TransactionManager>,
    disk: Arc<DiskManager>,
    pool: Arc<BufferPool>,
    wal: Arc<WalWriter>,
    db: DatabaseId,
    parent_id: zyron_catalog::TableId,
    columnar_dir: std::path::PathBuf,
    _tmp: tempfile::TempDir,
}

/// Parent table with N folded rows, child table with one heap row
/// referencing parent k = 3
async fn setup() -> Env {
    let tmp = tempfile::tempdir().expect("tmp");
    let data_dir = tmp.path().join("data");
    let wal_dir = tmp.path().join("wal");
    std::fs::create_dir_all(&data_dir).unwrap();
    std::fs::create_dir_all(&wal_dir).unwrap();
    let columnar_dir = data_dir.join("columnar");

    let disk = Arc::new(
        DiskManager::new(zyron_bench_harness::disk_config(data_dir.clone()))
            .await
            .unwrap(),
    );
    let pool = Arc::new(BufferPool::new(zyron_bench_harness::buffer_pool_config()));
    let wal = Arc::new(WalWriter::new(zyron_bench_harness::wal_config(wal_dir)).unwrap());
    let storage = HeapCatalogStorage::new(Arc::clone(&disk), Arc::clone(&pool)).unwrap();
    storage.init_cache().await.unwrap();
    let storage: Arc<dyn CatalogStorage> = Arc::new(storage);
    let cache = Arc::new(CatalogCache::new(1024, 256));
    let catalog = Catalog::new(Arc::clone(&storage), cache, Arc::clone(&wal))
        .await
        .unwrap();
    let db = catalog.create_database("db", "admin").await.unwrap();
    let schema = catalog.create_schema(db, "app", "admin").await.unwrap();

    let parent_id = catalog
        .create_table(
            schema,
            "parent",
            &[col("k", DataType::BigInt), col("v", DataType::BigInt)],
            &[],
        )
        .await
        .unwrap();
    let _child_id = catalog
        .create_table(
            schema,
            "child",
            &[col("ck", DataType::BigInt), col("pk", DataType::BigInt)],
            &[TableConstraint {
                name: None,
                kind: TableConstraintKind::ForeignKey {
                    columns: vec!["pk".into()],
                    ref_table: "parent".into(),
                    ref_columns: vec!["k".into()],
                    on_delete: ReferentialAction::Restrict,
                    on_update: ReferentialAction::Restrict,
                },
                enforced: true,
                on_violation: zyron_parser::ast::ViolationAction::Fail,
            }],
        )
        .await
        .unwrap();

    // txn counter above the rows' xmin so they fold
    let txn_manager = Arc::new(TransactionManager::with_start_txn_id(Arc::clone(&wal), 100));

    let pe = catalog.get_table_by_id(parent_id).unwrap();
    let parent_heap = HeapFile::new(
        Arc::clone(&disk),
        Arc::clone(&pool),
        HeapFileConfig {
            heap_file_id: pe.heap_file_id,
            fsm_file_id: pe.fsm_file_id,
        },
    )
    .unwrap();
    let mut tuples = Vec::new();
    for i in 0..12i64 {
        tuples.push(Tuple::new(encode_row2(i, i * 100), 1));
    }
    parent_heap.insert_batch(&tuples).await.unwrap();
    parent_heap.flush().await.unwrap();

    let ce = catalog.get_table(schema, "child").expect("child entry");
    let child_heap = HeapFile::new(
        Arc::clone(&disk),
        Arc::clone(&pool),
        HeapFileConfig {
            heap_file_id: ce.heap_file_id,
            fsm_file_id: ce.fsm_file_id,
        },
    )
    .unwrap();
    child_heap
        .insert_batch(&[Tuple::new(encode_row2(1, 3), 1)])
        .await
        .unwrap();
    child_heap.flush().await.unwrap();

    let cfg = CompactionWorkerConfig {
        min_rows: 4,
        columnar_dir: columnar_dir.clone(),
        ..CompactionWorkerConfig::default()
    };
    let (rows, _segs) = {
        let c = &catalog;
        let t = &txn_manager;
        let d = &disk;
        let p = &pool;
        let w = &wal;
        let cf = &cfg;
        tokio::task::block_in_place(|| {
            let rt = tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()
                .unwrap();
            CompactionWorker::run_cycle(&rt, c, t, d, p, w, cf, None, None, None, None)
        })
    };
    assert!(rows >= 12, "parent rows folded, got {rows}");
    let pe = catalog.get_table_by_id(parent_id).unwrap();
    assert!(
        !pe.columnar.segments.is_empty(),
        "parent has a columnar segment"
    );

    Env {
        catalog: Arc::new(catalog) as Arc<Catalog>,
        txn_manager,
        disk,
        pool,
        wal,
        db,
        parent_id,
        columnar_dir,
        _tmp: tmp,
    }
}

async fn run_sql(
    env: &Env,
    ctx_tweak: impl FnOnce(&mut ExecutionContext),
    sql: &str,
) -> Result<Vec<zyron_executor::batch::DataBatch>, String> {
    let stmt = zyron_parser::parse(sql)
        .map_err(|e| e.to_string())?
        .into_iter()
        .next()
        .expect("one statement");
    let plan = zyron_planner::plan(&env.catalog, env.db, vec!["app".into()], stmt, None)
        .await
        .map_err(|e| e.to_string())?;
    let mut txn = env
        .txn_manager
        .begin(IsolationLevel::ReadCommitted)
        .expect("begin");
    let mut ctx = ExecutionContext::new(
        Arc::clone(&env.catalog),
        Arc::clone(&env.wal),
        Arc::clone(&env.pool),
        Arc::clone(&env.disk),
        txn.txn_id as u32,
        txn.snapshot.clone(),
    );
    ctx.heap_files = Some(Arc::new(scc::HashMap::new()));
    ctx.btree_indexes = Some(Arc::new(scc::HashMap::new()));
    ctx.intent_locks = Some(Arc::clone(env.txn_manager.intent_locks()));
    ctx.row_locks = Some(Arc::clone(env.txn_manager.lock_table()));
    ctx_tweak(&mut ctx);
    let ctx = Arc::new(ctx);
    match zyron_executor::execute(plan, &ctx).await {
        Ok(batches) => {
            env.txn_manager.commit(&mut txn).await.expect("commit");
            Ok(batches)
        }
        Err(e) => {
            let _ = env.txn_manager.abort(&mut txn);
            Err(e.to_string())
        }
    }
}

/// Executes one statement inside the caller's open transaction, recording
/// undo through the transaction's log, without committing
async fn run_sql_in_txn(
    env: &Env,
    txn: &zyron_storage::txn::Transaction,
    sql: &str,
) -> Result<(), String> {
    let stmt = zyron_parser::parse(sql)
        .map_err(|e| e.to_string())?
        .into_iter()
        .next()
        .expect("one statement");
    let plan = zyron_planner::plan(&env.catalog, env.db, vec!["app".into()], stmt, None)
        .await
        .map_err(|e| e.to_string())?;
    let mut ctx = ExecutionContext::new(
        Arc::clone(&env.catalog),
        Arc::clone(&env.wal),
        Arc::clone(&env.pool),
        Arc::clone(&env.disk),
        txn.txn_id as u32,
        txn.snapshot.clone(),
    );
    ctx.heap_files = Some(Arc::new(scc::HashMap::new()));
    ctx.btree_indexes = Some(Arc::new(scc::HashMap::new()));
    ctx.intent_locks = Some(Arc::clone(env.txn_manager.intent_locks()));
    ctx.row_locks = Some(Arc::clone(env.txn_manager.lock_table()));
    ctx.undo_log = Some(txn.undo_log());
    let ctx = Arc::new(ctx);
    zyron_executor::execute(plan, &ctx)
        .await
        .map(|_| ())
        .map_err(|e| e.to_string())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn savepoint_rollback_revokes_folded_row_delete_and_update() {
    use zyron_storage::UndoEntry;
    use zyron_storage::columnar::PatchStore;

    let env = setup().await;
    let pe = env.catalog.get_table_by_id(env.parent_id).unwrap();
    let seg = pe.columnar.segments[0].clone();
    let store = ColumnarPatchManager::global(&env.columnar_dir)
        .store(env.parent_id.0 as u64)
        .unwrap();
    let v_col = pe.columns.iter().find(|c| c.name == "v").unwrap().id.0 as u32;

    let mut txn = env
        .txn_manager
        .begin(IsolationLevel::ReadCommitted)
        .expect("begin");
    txn.savepoint("sp".into(), 0, 0);

    run_sql_in_txn(&env, &txn, "DELETE FROM parent WHERE k = 7")
        .await
        .expect("folded delete in txn");
    run_sql_in_txn(&env, &txn, "UPDATE parent SET v = 4242 WHERE k = 8")
        .await
        .expect("folded update in txn");

    // both writes are live in the overlay
    assert!(
        store
            .row_overlay(seg.file_id, seg.sys_rowid_lo + 7)
            .is_some_and(|o| !o.supersedes.is_empty()),
        "supersede present before rollback"
    );
    assert!(
        store
            .row_overlay(seg.file_id, seg.sys_rowid_lo + 8)
            .is_some_and(|o| o.patches.contains_key(&v_col)),
        "value patch present before rollback"
    );

    // roll back and apply the recorded undo exactly as the wire handler does,
    // WAL log the revoke first, then revoke in the store
    let rollback = txn.rollback_to_savepoint("sp").expect("savepoint exists");
    assert_eq!(
        rollback.undo.len(),
        2,
        "one supersede and one patch recorded"
    );
    let txn_id = txn.txn_id;
    for entry in &rollback.undo {
        match entry {
            UndoEntry::ColumnarSupersede {
                table_id,
                branch,
                file_id,
                sys_rowid,
            } => {
                let mut pl = Vec::with_capacity(40);
                pl.extend_from_slice(&(*table_id as u64).to_le_bytes());
                pl.extend_from_slice(&branch.to_le_bytes());
                pl.extend_from_slice(&file_id.to_le_bytes());
                pl.extend_from_slice(&sys_rowid.to_le_bytes());
                pl.extend_from_slice(&txn_id.to_le_bytes());
                let lsn = env.wal.log_columnar_supersede_revoke(&pl).unwrap();
                store
                    .revoke_supersede(*branch, *file_id, *sys_rowid, txn_id, lsn.0)
                    .unwrap();
            }
            UndoEntry::ColumnarPatch {
                table_id,
                branch,
                file_id,
                sys_rowid,
                column_id,
            } => {
                let mut pl = Vec::with_capacity(44);
                pl.extend_from_slice(&(*table_id as u64).to_le_bytes());
                pl.extend_from_slice(&branch.to_le_bytes());
                pl.extend_from_slice(&file_id.to_le_bytes());
                pl.extend_from_slice(&sys_rowid.to_le_bytes());
                pl.extend_from_slice(&column_id.to_le_bytes());
                pl.extend_from_slice(&txn_id.to_le_bytes());
                let lsn = env.wal.log_columnar_patch_revoke(&pl).unwrap();
                store
                    .revoke_value_patch(*branch, *file_id, *sys_rowid, *column_id, txn_id, lsn.0)
                    .unwrap();
            }
            other => panic!("unexpected undo entry {other:?}"),
        }
    }
    env.txn_manager.commit(&mut txn).await.expect("commit");

    // the live overlay is clean again
    assert!(
        store
            .row_overlay(seg.file_id, seg.sys_rowid_lo + 7)
            .is_none(),
        "supersede revoked from the live overlay"
    );
    assert!(
        store
            .row_overlay(seg.file_id, seg.sys_rowid_lo + 8)
            .is_none(),
        "value patch revoked from the live overlay"
    );

    // the on disk log replays to the same clean state, this is the crash
    // after rollback shape
    let patch_path = env
        .columnar_dir
        .join(format!("{}.zyrpatch", env.parent_id.0));
    let reopened = PatchStore::open(&patch_path).expect("reopen patch log");
    assert!(
        reopened
            .row_overlay(seg.file_id, seg.sys_rowid_lo + 7)
            .is_none(),
        "replayed log holds no supersede for the rolled back delete"
    );
    assert!(
        reopened
            .row_overlay(seg.file_id, seg.sys_rowid_lo + 8)
            .is_none(),
        "replayed log holds no patch for the rolled back update"
    );

    // and the rows read back with their original values
    let rows = run_sql(&env, |_| {}, "SELECT v FROM parent WHERE k = 7")
        .await
        .expect("select restored row");
    let n: usize = rows.iter().map(|b| b.num_rows).sum();
    assert_eq!(n, 1, "rolled back delete leaves the row visible");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn folded_parent_delete_respects_fk_restrict() {
    let env = setup().await;

    // k = 3 is referenced by the heap resident child row, the delete must
    // abort inside the columnar path before any supersede is written
    let err = run_sql(&env, |_| {}, "DELETE FROM parent WHERE k = 3")
        .await
        .expect_err("FK restrict must abort the folded row delete");
    assert!(
        err.to_lowercase().contains("foreign key") || err.to_lowercase().contains("referenc"),
        "error must name the FK violation, got: {err}"
    );

    // no supersede leaked for the protected row
    let pe = env.catalog.get_table_by_id(env.parent_id).unwrap();
    let seg = pe.columnar.segments[0].clone();
    let store = ColumnarPatchManager::global(&env.columnar_dir)
        .store(env.parent_id.0 as u64)
        .unwrap();
    assert!(
        store
            .row_overlay(seg.file_id, seg.sys_rowid_lo + 3)
            .is_none(),
        "aborted delete must leave no patch overlay"
    );

    // an unreferenced folded row deletes normally through the same path
    run_sql(&env, |_| {}, "DELETE FROM parent WHERE k = 11")
        .await
        .expect("unreferenced folded row deletes");
    assert!(
        store
            .row_overlay(seg.file_id, seg.sys_rowid_lo + 11)
            .is_some(),
        "delete of the unreferenced row wrote its supersede"
    );
}

struct CountingHook {
    deletes: AtomicUsize,
    delete_rows: AtomicUsize,
    updates: AtomicUsize,
}

impl CdcHook for CountingHook {
    fn on_insert(
        &self,
        _table_id: u32,
        _tuples: &[&[u8]],
        _version: u64,
        _timestamp: i64,
        _txn_id: u32,
        _is_last_in_txn: bool,
    ) -> zyron_common::Result<()> {
        Ok(())
    }
    fn on_delete(
        &self,
        _table_id: u32,
        old_data: &[&[u8]],
        _version: u64,
        _timestamp: i64,
        _txn_id: u32,
        _is_last_in_txn: bool,
    ) -> zyron_common::Result<()> {
        self.deletes.fetch_add(1, Ordering::SeqCst);
        self.delete_rows.fetch_add(old_data.len(), Ordering::SeqCst);
        Ok(())
    }
    fn on_update(
        &self,
        _table_id: u32,
        old_data: &[&[u8]],
        new_data: &[&[u8]],
        _version: u64,
        _timestamp: i64,
        _txn_id: u32,
        _is_last_in_txn: bool,
    ) -> zyron_common::Result<()> {
        assert_eq!(old_data.len(), new_data.len());
        self.updates.fetch_add(old_data.len(), Ordering::SeqCst);
        Ok(())
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn folded_row_dml_reaches_cdc_hook() {
    let env = setup().await;
    let hook = Arc::new(CountingHook {
        deletes: AtomicUsize::new(0),
        delete_rows: AtomicUsize::new(0),
        updates: AtomicUsize::new(0),
    });

    let h = Arc::clone(&hook);
    run_sql(
        &env,
        move |ctx| ctx.cdc_hook = Some(h),
        "DELETE FROM parent WHERE k = 5",
    )
    .await
    .expect("folded row delete succeeds");
    assert_eq!(hook.deletes.load(Ordering::SeqCst), 1, "CDC delete fired");
    assert_eq!(
        hook.delete_rows.load(Ordering::SeqCst),
        1,
        "one old row captured"
    );

    let h = Arc::clone(&hook);
    run_sql(
        &env,
        move |ctx| ctx.cdc_hook = Some(h),
        "UPDATE parent SET v = 9999 WHERE k = 6",
    )
    .await
    .expect("folded row update succeeds");
    assert_eq!(hook.updates.load(Ordering::SeqCst), 1, "CDC update fired");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn folded_row_write_write_conflict_via_row_locks() {
    use zyron_common::RowLocator;

    let env = setup().await;
    let pe = env.catalog.get_table_by_id(env.parent_id).unwrap();
    let seg = pe.columnar.segments[0].clone();

    // txn1 updates a folded row and holds its exclusive row lock
    let mut txn1 = env
        .txn_manager
        .begin(IsolationLevel::ReadCommitted)
        .expect("begin txn1");
    run_sql_in_txn(&env, &txn1, "UPDATE parent SET v = 1 WHERE k = 7")
        .await
        .expect("folded update in txn1");
    let locator = RowLocator::Columnar {
        file_id: seg.file_id,
        sys_rowid: seg.sys_rowid_lo + 7,
    };
    assert_eq!(
        env.txn_manager
            .lock_table()
            .is_locked_by(env.parent_id.0, locator),
        Some(txn1.txn_id),
        "columnar DML holds the row lock keyed on the columnar locator"
    );

    // a second writer of the same folded row conflicts deterministically
    let mut txn2 = env
        .txn_manager
        .begin(IsolationLevel::ReadCommitted)
        .expect("begin txn2");
    let err = run_sql_in_txn(&env, &txn2, "UPDATE parent SET v = 2 WHERE k = 7")
        .await
        .expect_err("second writer of the locked folded row conflicts");
    assert!(err.contains("locked by txn"), "got: {err}");
    let err = run_sql_in_txn(&env, &txn2, "DELETE FROM parent WHERE k = 7")
        .await
        .expect_err("delete of the locked folded row conflicts");
    assert!(err.contains("locked by txn"), "got: {err}");

    // an unrelated folded row is not blocked
    run_sql_in_txn(&env, &txn2, "UPDATE parent SET v = 3 WHERE k = 8")
        .await
        .expect("unrelated folded row writes fine");

    // abort releases txn1's lock, txn2 retries clean
    env.txn_manager.abort(&mut txn1).expect("abort txn1");
    run_sql_in_txn(&env, &txn2, "UPDATE parent SET v = 2 WHERE k = 7")
        .await
        .expect("retry after release succeeds");
    env.txn_manager.abort(&mut txn2).expect("abort txn2");
}

/// Runs one compaction cycle with the doc registry and btree registry
/// attached so folded rows re-point their documents and index entries,
/// returns the folded row count
fn fold_all_with_btree(
    env: &Env,
    reg: &Arc<zyron_common::DocRegistry>,
    btree: Option<&Arc<scc::HashMap<u32, Arc<zyron_storage::BTreeIndex>>>>,
) -> u64 {
    let cfg = CompactionWorkerConfig {
        min_rows: 4,
        columnar_dir: env.columnar_dir.clone(),
        ..CompactionWorkerConfig::default()
    };
    let c = &env.catalog;
    let t = &env.txn_manager;
    let d = &env.disk;
    let p = &env.pool;
    let w = &env.wal;
    let cf = &cfg;
    tokio::task::block_in_place(|| {
        let rt = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap();
        CompactionWorker::run_cycle(&rt, c, t, d, p, w, cf, None, Some(reg), btree, None).0
    })
}

/// Runs one compaction cycle with only the doc registry attached
fn fold_all(env: &Env, reg: &Arc<zyron_common::DocRegistry>) -> u64 {
    fold_all_with_btree(env, reg, None)
}

/// Creates the notes table with a catalog fulltext index and a live FTS
/// index, then inserts five rows through SQL so the insert maintenance
/// path allocates documents and indexes every row
async fn create_notes_with_fts(
    env: &Env,
    reg: &Arc<zyron_common::DocRegistry>,
    fts_mgr: &Arc<zyron_search::FtsManager>,
) -> (zyron_catalog::TableId, zyron_catalog::IndexId) {
    let schema = env
        .catalog
        .get_table_by_id(env.parent_id)
        .unwrap()
        .schema_id;
    let notes_id = env
        .catalog
        .create_table(
            schema,
            "notes",
            &[col("id", DataType::BigInt), col("body", DataType::Text)],
            &[],
        )
        .await
        .unwrap();
    let body_col = env
        .catalog
        .get_table_by_id(notes_id)
        .unwrap()
        .columns
        .iter()
        .find(|c| c.name == "body")
        .unwrap()
        .id
        .0;
    let idx_id = env
        .catalog
        .create_index(
            notes_id,
            schema,
            "notes_body_fts",
            &["body".to_string()],
            false,
            zyron_catalog::IndexType::Fulltext,
        )
        .await
        .unwrap();
    fts_mgr
        .create_index(idx_id.0, notes_id.0, vec![body_col])
        .unwrap();

    let m = Arc::clone(fts_mgr);
    let r = Arc::clone(reg);
    run_sql(
        env,
        move |ctx| {
            ctx.set_fts_manager(m);
            ctx.doc_registry = Some(r);
        },
        "INSERT INTO notes (id, body) VALUES \
         (1, 'quantum spindle calibration'), (2, 'ordinary text'), \
         (3, 'more ordinary text'), (4, 'even more filler'), (5, 'padding row')",
    )
    .await
    .expect("insert with fts maintenance");
    (notes_id, idx_id)
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn fts_finds_a_row_after_it_folds_and_its_heap_slot_is_gone() {
    use zyron_common::{DocRegistry, RowLocator};

    let env = setup().await;
    let reg = Arc::new(DocRegistry::new());
    let fts_mgr = Arc::new(zyron_search::FtsManager::new());
    let (notes_id, idx_id) = create_notes_with_fts(&env, &reg, &fts_mgr).await;

    assert_eq!(
        fts_mgr.get_index(idx_id.0).expect("live index").doc_count(),
        5,
        "insert maintenance indexed every row"
    );

    let search_sql = "SELECT id FROM notes WHERE MATCH(body) AGAINST('quantum')";

    // heap resident: the search finds the row
    let (m, r) = (Arc::clone(&fts_mgr), Arc::clone(&reg));
    let rows = run_sql(
        &env,
        move |ctx| {
            ctx.set_fts_manager(m);
            ctx.doc_registry = Some(r);
        },
        search_sql,
    )
    .await
    .expect("search before fold");
    let total: usize = rows.iter().map(|b| b.num_rows).sum();
    assert_eq!(total, 1, "one hit before the fold");

    // the row's document points at the heap
    let doc = {
        let te = env.catalog.get_table_by_id(notes_id).unwrap();
        assert!(te.columnar.segments.is_empty());
        let mut found = None;
        for ord in 0..reg.ordinal_count(notes_id.0) {
            if let Some(RowLocator::Heap { .. }) = reg.locator(notes_id.0, ord) {
                found = Some(ord);
                break;
            }
        }
        found.expect("a heap document exists before the fold")
    };

    // fold, which also zeroes the folded heap slots
    let rows_folded = fold_all(&env, &reg);
    assert!(rows_folded >= 5, "notes rows folded, got {rows_folded}");
    assert!(
        !env.catalog
            .get_table_by_id(notes_id)
            .unwrap()
            .columnar
            .segments
            .is_empty(),
        "notes has a columnar segment"
    );

    // the document kept its ordinal, re-pointed at the columnar row
    match reg.locator(notes_id.0, doc) {
        Some(RowLocator::Columnar { .. }) => {}
        other => panic!("document should point at columnar after the fold, got {other:?}"),
    }

    // the search still finds the row, served from the columnar segment
    let (m, r) = (Arc::clone(&fts_mgr), Arc::clone(&reg));
    let rows = run_sql(
        &env,
        move |ctx| {
            ctx.set_fts_manager(m);
            ctx.doc_registry = Some(r);
        },
        search_sql,
    )
    .await
    .expect("search after fold");
    let total: usize = rows.iter().map(|b| b.num_rows).sum();
    assert_eq!(total, 1, "the folded row is still searchable");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn vector_search_finds_a_row_after_it_folds_and_its_heap_slot_is_gone() {
    use zyron_common::{DocRegistry, RowLocator};
    use zyron_executor::column::ScalarValue;

    let env = setup().await;
    let reg = Arc::new(DocRegistry::new());
    let vec_mgr = Arc::new(zyron_search::vector::VectorIndexManager::new());

    // a vector table with a real catalog vector index and a live HNSW index
    let schema = env
        .catalog
        .get_table_by_id(env.parent_id)
        .unwrap()
        .schema_id;
    let emb_id = env
        .catalog
        .create_table(
            schema,
            "embeddings",
            &[
                col("id", DataType::BigInt),
                col("emb", DataType::Vector(Some(4))),
            ],
            &[],
        )
        .await
        .unwrap();
    let emb_col = env
        .catalog
        .get_table_by_id(emb_id)
        .unwrap()
        .columns
        .iter()
        .find(|c| c.name == "emb")
        .unwrap()
        .id
        .0;
    let idx_id = env
        .catalog
        .create_index(
            emb_id,
            schema,
            "embeddings_emb_hnsw",
            &["emb".to_string()],
            false,
            zyron_catalog::IndexType::Vector,
        )
        .await
        .unwrap();
    let vec_idx = vec_mgr
        .create_index(
            idx_id.0,
            emb_id.0,
            emb_col,
            4,
            zyron_search::vector::HnswConfig::default(),
        )
        .unwrap();

    // rows land in the heap directly, then the index and the registry are
    // populated the way the server rebuild path does, one ordinal per
    // heap locator
    let te = env.catalog.get_table_by_id(emb_id).unwrap();
    let heap = HeapFile::new(
        Arc::clone(&env.disk),
        Arc::clone(&env.pool),
        HeapFileConfig {
            heap_file_id: te.heap_file_id,
            fsm_file_id: te.fsm_file_id,
        },
    )
    .unwrap();
    let vectors: [[f32; 4]; 5] = [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
        [0.5, 0.5, 0.5, 0.5],
    ];
    let tuples: Vec<Tuple> = vectors
        .iter()
        .enumerate()
        .map(|(i, v)| Tuple::new(encode_row_id_vec(i as i64 + 1, v), 1))
        .collect();
    let tids = heap.insert_batch(&tuples).await.unwrap();
    heap.flush().await.unwrap();
    for (i, tid) in tids.iter().enumerate() {
        let doc = reg.allocate(emb_id.0, tid.locator());
        zyron_search::vector::VectorSearch::insert(vec_idx.as_ref(), doc, &vectors[i]).unwrap();
    }

    let search_sql = "SELECT id FROM embeddings WHERE emb <-> ARRAY[0.9, 0.1, 0.05, 0.05] LIMIT 1";

    // heap resident: the nearest neighbor search returns the closest row
    let (m, r) = (Arc::clone(&vec_mgr), Arc::clone(&reg));
    let rows = run_sql(
        &env,
        move |ctx| {
            ctx.set_vector_manager(m);
            ctx.doc_registry = Some(r);
        },
        search_sql,
    )
    .await
    .expect("vector search before fold");
    let total: usize = rows.iter().map(|b| b.num_rows).sum();
    assert_eq!(total, 1, "one nearest neighbor before the fold");
    let hit = rows.iter().find(|b| b.num_rows > 0).unwrap();
    assert!(
        matches!(hit.columns[0].data.get_scalar(0), ScalarValue::Int64(1)),
        "the closest vector's row id is returned"
    );

    // a document points at the heap before the fold
    let doc = (0..reg.ordinal_count(emb_id.0))
        .find(|&o| matches!(reg.locator(emb_id.0, o), Some(RowLocator::Heap { .. })))
        .expect("a heap document exists before the fold");

    let rows_folded = fold_all(&env, &reg);
    assert!(
        rows_folded >= 5,
        "embeddings rows folded, got {rows_folded}"
    );
    assert!(
        !env.catalog
            .get_table_by_id(emb_id)
            .unwrap()
            .columnar
            .segments
            .is_empty(),
        "embeddings has a columnar segment"
    );

    // the document kept its ordinal, re-pointed at the columnar row
    match reg.locator(emb_id.0, doc) {
        Some(RowLocator::Columnar { .. }) => {}
        other => panic!("document should point at columnar after the fold, got {other:?}"),
    }

    // the search still finds the row, served from the columnar segment
    let (m, r) = (Arc::clone(&vec_mgr), Arc::clone(&reg));
    let rows = run_sql(
        &env,
        move |ctx| {
            ctx.set_vector_manager(m);
            ctx.doc_registry = Some(r);
        },
        search_sql,
    )
    .await
    .expect("vector search after fold");
    let total: usize = rows.iter().map(|b| b.num_rows).sum();
    assert_eq!(total, 1, "the folded row is still the nearest neighbor");
    let hit = rows.iter().find(|b| b.num_rows > 0).unwrap();
    assert!(
        matches!(hit.columns[0].data.get_scalar(0), ScalarValue::Int64(1)),
        "the folded row id is returned"
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn spatial_search_finds_a_row_after_it_folds_and_its_heap_slot_is_gone() {
    use zyron_common::{DocRegistry, RowLocator};
    use zyron_executor::column::ScalarValue;

    let env = setup().await;
    let reg = Arc::new(DocRegistry::new());
    let spatial_mgr = Arc::new(zyron_types::spatial_index::SpatialIndexManager::new());

    // a geometry table with a real catalog spatial index and a live R-tree
    let schema = env
        .catalog
        .get_table_by_id(env.parent_id)
        .unwrap()
        .schema_id;
    let places_id = env
        .catalog
        .create_table(
            schema,
            "places",
            &[col("id", DataType::BigInt), col("loc", DataType::Geometry)],
            &[],
        )
        .await
        .unwrap();
    let idx_id = env
        .catalog
        .create_index(
            places_id,
            schema,
            "places_loc_rtree",
            &["loc".to_string()],
            false,
            zyron_catalog::IndexType::Spatial,
        )
        .await
        .unwrap();
    spatial_mgr.create_index(idx_id.0, 2, 4326);

    // SQL insert drives the spatial maintenance, one document per row
    let (s, r) = (Arc::clone(&spatial_mgr), Arc::clone(&reg));
    run_sql(
        &env,
        move |ctx| {
            ctx.set_spatial_manager(s);
            ctx.doc_registry = Some(r);
        },
        "INSERT INTO places (id, loc) VALUES \
         (1, st_geom_from_text('POINT(10 10)')), (2, st_geom_from_text('POINT(20 20)')), \
         (3, st_geom_from_text('POINT(30 30)')), (4, st_geom_from_text('POINT(40 40)')), \
         (5, st_geom_from_text('POINT(50 50)'))",
    )
    .await
    .expect("insert with spatial maintenance");
    assert_eq!(
        reg.ordinal_count(places_id.0),
        5,
        "insert maintenance allocated one document per row"
    );

    let search_sql = "SELECT id FROM places WHERE st_dwithin(loc, st_make_point(10.0, 10.0), 5.0)";

    // heap resident: the radius search returns only the matching point
    let (s, r) = (Arc::clone(&spatial_mgr), Arc::clone(&reg));
    let rows = run_sql(
        &env,
        move |ctx| {
            ctx.set_spatial_manager(s);
            ctx.doc_registry = Some(r);
        },
        search_sql,
    )
    .await
    .expect("spatial search before fold");
    let total: usize = rows.iter().map(|b| b.num_rows).sum();
    assert_eq!(total, 1, "one point within the radius before the fold");
    let hit = rows.iter().find(|b| b.num_rows > 0).unwrap();
    assert!(
        matches!(hit.columns[0].data.get_scalar(0), ScalarValue::Int64(1)),
        "the matching point's row id is returned"
    );

    // a document points at the heap before the fold
    let doc = (0..reg.ordinal_count(places_id.0))
        .find(|&o| matches!(reg.locator(places_id.0, o), Some(RowLocator::Heap { .. })))
        .expect("a heap document exists before the fold");

    let rows_folded = fold_all(&env, &reg);
    assert!(rows_folded >= 5, "places rows folded, got {rows_folded}");
    assert!(
        !env.catalog
            .get_table_by_id(places_id)
            .unwrap()
            .columnar
            .segments
            .is_empty(),
        "places has a columnar segment"
    );

    // the document kept its ordinal, re-pointed at the columnar row
    match reg.locator(places_id.0, doc) {
        Some(RowLocator::Columnar { .. }) => {}
        other => panic!("document should point at columnar after the fold, got {other:?}"),
    }

    // the search still finds the row, served from the columnar segment
    let (s, r) = (Arc::clone(&spatial_mgr), Arc::clone(&reg));
    let rows = run_sql(
        &env,
        move |ctx| {
            ctx.set_spatial_manager(s);
            ctx.doc_registry = Some(r);
        },
        search_sql,
    )
    .await
    .expect("spatial search after fold");
    let total: usize = rows.iter().map(|b| b.num_rows).sum();
    assert_eq!(total, 1, "the folded row is still within the radius");
    let hit = rows.iter().find(|b| b.num_rows > 0).unwrap();
    assert!(
        matches!(hit.columns[0].data.get_scalar(0), ScalarValue::Int64(1)),
        "the folded row id is returned"
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn columnar_update_and_delete_maintain_the_fts_index_on_folded_rows() {
    use zyron_common::DocRegistry;

    let env = setup().await;
    let reg = Arc::new(DocRegistry::new());
    let fts_mgr = Arc::new(zyron_search::FtsManager::new());
    let (notes_id, _idx_id) = create_notes_with_fts(&env, &reg, &fts_mgr).await;

    let rows_folded = fold_all(&env, &reg);
    assert!(rows_folded >= 5, "notes rows folded, got {rows_folded}");
    assert!(
        !env.catalog
            .get_table_by_id(notes_id)
            .unwrap()
            .columnar
            .segments
            .is_empty(),
        "notes has a columnar segment"
    );

    // update the folded row through the columnar patch path, the old
    // document must retire and the new body must become searchable
    let (m, r) = (Arc::clone(&fts_mgr), Arc::clone(&reg));
    run_sql(
        &env,
        move |ctx| {
            ctx.set_fts_manager(m);
            ctx.doc_registry = Some(r);
        },
        "UPDATE notes SET body = 'nebula archive' WHERE id = 1",
    )
    .await
    .expect("columnar update with fts maintenance");

    for (term, expected, why) in [
        (
            "quantum",
            0usize,
            "old body must stop matching after the columnar update",
        ),
        (
            "nebula",
            1usize,
            "new body must match after the columnar update",
        ),
    ] {
        let (m, r) = (Arc::clone(&fts_mgr), Arc::clone(&reg));
        let rows = run_sql(
            &env,
            move |ctx| {
                ctx.set_fts_manager(m);
                ctx.doc_registry = Some(r);
            },
            &format!("SELECT id FROM notes WHERE MATCH(body) AGAINST('{term}')"),
        )
        .await
        .expect("search after columnar update");
        let total: usize = rows.iter().map(|b| b.num_rows).sum();
        assert_eq!(total, expected, "{why}");
    }

    // delete the folded row, its document must leave every index
    let (m, r) = (Arc::clone(&fts_mgr), Arc::clone(&reg));
    run_sql(
        &env,
        move |ctx| {
            ctx.set_fts_manager(m);
            ctx.doc_registry = Some(r);
        },
        "DELETE FROM notes WHERE id = 1",
    )
    .await
    .expect("columnar delete with fts maintenance");

    let (m, r) = (Arc::clone(&fts_mgr), Arc::clone(&reg));
    let rows = run_sql(
        &env,
        move |ctx| {
            ctx.set_fts_manager(m);
            ctx.doc_registry = Some(r);
        },
        "SELECT id FROM notes WHERE MATCH(body) AGAINST('nebula')",
    )
    .await
    .expect("search after columnar delete");
    let total: usize = rows.iter().map(|b| b.num_rows).sum();
    assert_eq!(total, 0, "the deleted folded row must leave the index");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn unique_conflicts_are_detected_between_folded_rows() {
    use zyron_common::DocRegistry;

    let env = setup().await;
    let reg = Arc::new(DocRegistry::new());
    let btree_reg: Arc<scc::HashMap<u32, Arc<zyron_storage::BTreeIndex>>> =
        Arc::new(scc::HashMap::new());

    // an accounts table with a unique btree index on email and a live tree
    let schema = env
        .catalog
        .get_table_by_id(env.parent_id)
        .unwrap()
        .schema_id;
    let accounts_id = env
        .catalog
        .create_table(
            schema,
            "accounts",
            &[col("id", DataType::BigInt), col("email", DataType::Text)],
            &[],
        )
        .await
        .unwrap();
    let idx_id = env
        .catalog
        .create_index(
            accounts_id,
            schema,
            "accounts_email_key",
            &["email".to_string()],
            true,
            zyron_catalog::IndexType::BTree,
        )
        .await
        .unwrap();
    let idx_dir = env.columnar_dir.parent().unwrap().join("idx");
    std::fs::create_dir_all(&idx_dir).unwrap();
    let tree = Arc::new(
        zyron_storage::BTreeIndex::create(idx_id.0, idx_dir)
            .await
            .unwrap(),
    );
    let _ = btree_reg.insert_sync(idx_id.0, tree);

    let b = Arc::clone(&btree_reg);
    run_sql(
        &env,
        move |ctx| ctx.btree_indexes = Some(b),
        "INSERT INTO accounts (id, email) VALUES \
         (1, 'a@x.io'), (2, 'b@x.io'), (3, 'c@x.io'), (4, 'd@x.io'), (5, 'e@x.io')",
    )
    .await
    .expect("insert with btree maintenance");

    // heap resident: a duplicate already conflicts
    let b = Arc::clone(&btree_reg);
    let err = run_sql(
        &env,
        move |ctx| ctx.btree_indexes = Some(b),
        "INSERT INTO accounts (id, email) VALUES (6, 'a@x.io')",
    )
    .await
    .expect_err("duplicate against a heap row conflicts");
    assert!(err.to_lowercase().contains("unique"), "got: {err}");

    let rows_folded = fold_all_with_btree(&env, &reg, Some(&btree_reg));
    assert!(rows_folded >= 5, "accounts rows folded, got {rows_folded}");
    assert!(
        !env.catalog
            .get_table_by_id(accounts_id)
            .unwrap()
            .columnar
            .segments
            .is_empty(),
        "accounts has a columnar segment"
    );

    // a duplicate against a folded row conflicts through the re-keyed entry
    let b = Arc::clone(&btree_reg);
    let err = run_sql(
        &env,
        move |ctx| ctx.btree_indexes = Some(b),
        "INSERT INTO accounts (id, email) VALUES (6, 'a@x.io')",
    )
    .await
    .expect_err("duplicate against a folded row conflicts");
    assert!(err.to_lowercase().contains("unique"), "got: {err}");

    // a fresh value inserts fine
    let b = Arc::clone(&btree_reg);
    run_sql(
        &env,
        move |ctx| ctx.btree_indexes = Some(b),
        "INSERT INTO accounts (id, email) VALUES (7, 'f@x.io')",
    )
    .await
    .expect("fresh value inserts");

    // updating a folded row onto another folded row's value conflicts
    let b = Arc::clone(&btree_reg);
    let err = run_sql(
        &env,
        move |ctx| ctx.btree_indexes = Some(b),
        "UPDATE accounts SET email = 'b@x.io' WHERE id = 1",
    )
    .await
    .expect_err("update onto another folded row's unique value conflicts");
    assert!(err.to_lowercase().contains("unique"), "got: {err}");

    // a same-value update of the row itself is not a self-conflict
    let b = Arc::clone(&btree_reg);
    run_sql(
        &env,
        move |ctx| ctx.btree_indexes = Some(b),
        "UPDATE accounts SET email = 'a@x.io' WHERE id = 1",
    )
    .await
    .expect("same-value update excludes the row's own entries");

    // moving a folded row to a fresh value frees the old one and claims
    // the new one
    let b = Arc::clone(&btree_reg);
    run_sql(
        &env,
        move |ctx| ctx.btree_indexes = Some(b),
        "UPDATE accounts SET email = 'z@x.io' WHERE id = 1",
    )
    .await
    .expect("update to a fresh value succeeds");

    let b = Arc::clone(&btree_reg);
    run_sql(
        &env,
        move |ctx| ctx.btree_indexes = Some(b),
        "INSERT INTO accounts (id, email) VALUES (8, 'a@x.io')",
    )
    .await
    .expect("the old value is free, its stale entry is refuted by the current value");

    let b = Arc::clone(&btree_reg);
    let err = run_sql(
        &env,
        move |ctx| ctx.btree_indexes = Some(b),
        "INSERT INTO accounts (id, email) VALUES (9, 'z@x.io')",
    )
    .await
    .expect_err("the new value is claimed by the updated folded row");
    assert!(err.to_lowercase().contains("unique"), "got: {err}");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn secondary_index_point_lookup_returns_the_row_after_it_folds() {
    use zyron_common::DocRegistry;
    use zyron_executor::column::ScalarValue;

    let env = setup().await;
    let reg = Arc::new(DocRegistry::new());
    let btree_reg: Arc<scc::HashMap<u32, Arc<zyron_storage::BTreeIndex>>> =
        Arc::new(scc::HashMap::new());

    // an items table with a non-unique btree index on k and a live tree
    let schema = env
        .catalog
        .get_table_by_id(env.parent_id)
        .unwrap()
        .schema_id;
    let items_id = env
        .catalog
        .create_table(
            schema,
            "items",
            &[col("id", DataType::BigInt), col("k", DataType::BigInt)],
            &[],
        )
        .await
        .unwrap();
    let idx_id = env
        .catalog
        .create_index(
            items_id,
            schema,
            "items_k_idx",
            &["k".to_string()],
            false,
            zyron_catalog::IndexType::BTree,
        )
        .await
        .unwrap();
    let idx_dir = env.columnar_dir.parent().unwrap().join("idx");
    std::fs::create_dir_all(&idx_dir).unwrap();
    let tree = Arc::new(
        zyron_storage::BTreeIndex::create(idx_id.0, idx_dir)
            .await
            .unwrap(),
    );
    let _ = btree_reg.insert_sync(idx_id.0, tree);

    let b = Arc::clone(&btree_reg);
    run_sql(
        &env,
        move |ctx| ctx.btree_indexes = Some(b),
        "INSERT INTO items (id, k) VALUES (1, 10), (2, 20), (3, 30), (4, 40), (5, 50)",
    )
    .await
    .expect("insert with btree maintenance");

    let assert_lookup = |rows: Vec<zyron_executor::batch::DataBatch>, expect_id: i64, why: &str| {
        let total: usize = rows.iter().map(|b| b.num_rows).sum();
        assert_eq!(total, 1, "{why}: expected one row");
        let hit = rows.iter().find(|b| b.num_rows > 0).unwrap();
        assert!(
            matches!(hit.columns[0].data.get_scalar(0), ScalarValue::Int64(v) if v == expect_id),
            "{why}: wrong id"
        );
    };

    // heap resident: the index point lookup finds the row
    let b = Arc::clone(&btree_reg);
    let rows = run_sql(
        &env,
        move |ctx| ctx.btree_indexes = Some(b),
        "SELECT id FROM items WHERE k = 30",
    )
    .await
    .expect("lookup before fold");
    assert_lookup(rows, 3, "before fold");

    let rows_folded = fold_all_with_btree(&env, &reg, Some(&btree_reg));
    assert!(rows_folded >= 5, "items rows folded, got {rows_folded}");
    assert!(
        !env.catalog
            .get_table_by_id(items_id)
            .unwrap()
            .columnar
            .segments
            .is_empty(),
        "items has a columnar segment"
    );

    // folded: the point lookup routes through the re-keyed index and serves
    // the row from the columnar segment
    let b = Arc::clone(&btree_reg);
    let rows = run_sql(
        &env,
        move |ctx| ctx.btree_indexes = Some(b),
        "SELECT id FROM items WHERE k = 30",
    )
    .await
    .expect("lookup after fold");
    assert_lookup(rows, 3, "after fold via index");

    // with no live tree registered the executor unions both stores instead
    // of dropping folded rows through the heap-only fallback
    let rows = run_sql(&env, |_| {}, "SELECT id FROM items WHERE k = 30")
        .await
        .expect("lookup after fold without a live tree");
    assert_lookup(rows, 3, "after fold via hybrid fallback");

    // DML through the index route patches the folded row and re-keys it
    let b = Arc::clone(&btree_reg);
    run_sql(
        &env,
        move |ctx| ctx.btree_indexes = Some(b),
        "UPDATE items SET k = 35 WHERE k = 30",
    )
    .await
    .expect("update through the index route");

    let b = Arc::clone(&btree_reg);
    let rows = run_sql(
        &env,
        move |ctx| ctx.btree_indexes = Some(b),
        "SELECT id FROM items WHERE k = 35",
    )
    .await
    .expect("lookup of the new value");
    assert_lookup(rows, 3, "new value after update");

    let b = Arc::clone(&btree_reg);
    let rows = run_sql(
        &env,
        move |ctx| ctx.btree_indexes = Some(b),
        "SELECT id FROM items WHERE k = 30",
    )
    .await
    .expect("lookup of the old value");
    let total: usize = rows.iter().map(|b| b.num_rows).sum();
    assert_eq!(
        total, 0,
        "the stale entry is refuted by the post-filter on the current value"
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn reindex_rebuild_covers_folded_rows() {
    use zyron_common::DocRegistry;
    use zyron_executor::column::ScalarValue;
    use zyron_executor::operator::Operator;

    let env = setup().await;
    let reg = Arc::new(DocRegistry::new());
    let btree_reg: Arc<scc::HashMap<u32, Arc<zyron_storage::BTreeIndex>>> =
        Arc::new(scc::HashMap::new());

    let schema = env
        .catalog
        .get_table_by_id(env.parent_id)
        .unwrap()
        .schema_id;
    let logs_id = env
        .catalog
        .create_table(
            schema,
            "logs",
            &[col("id", DataType::BigInt), col("tag", DataType::BigInt)],
            &[],
        )
        .await
        .unwrap();
    let idx_id = env
        .catalog
        .create_index(
            logs_id,
            schema,
            "logs_tag_idx",
            &["tag".to_string()],
            false,
            zyron_catalog::IndexType::BTree,
        )
        .await
        .unwrap();
    let idx_dir = env.columnar_dir.parent().unwrap().join("idx");
    std::fs::create_dir_all(&idx_dir).unwrap();
    let tree = Arc::new(
        zyron_storage::BTreeIndex::create(idx_id.0, idx_dir)
            .await
            .unwrap(),
    );
    let _ = btree_reg.insert_sync(idx_id.0, tree);

    let b = Arc::clone(&btree_reg);
    run_sql(
        &env,
        move |ctx| ctx.btree_indexes = Some(b),
        "INSERT INTO logs (id, tag) VALUES (1, 10), (2, 20), (3, 30), (4, 40), (5, 50)",
    )
    .await
    .expect("insert with btree maintenance");

    let rows_folded = fold_all_with_btree(&env, &reg, Some(&btree_reg));
    assert!(rows_folded >= 5, "logs rows folded, got {rows_folded}");

    // Simulate a restart losing the in-memory tree, then rebuild the way
    // handle_reindex does: heap residual pages (none remain folded) plus
    // one columnar collection pass feeding rebuild_btree_index_from_batch
    let idx_dir2 = env.columnar_dir.parent().unwrap().join("idx2");
    std::fs::create_dir_all(&idx_dir2).unwrap();
    let fresh = Arc::new(
        zyron_storage::BTreeIndex::create(idx_id.0, idx_dir2)
            .await
            .unwrap(),
    );

    let te = env.catalog.get_table_by_id(logs_id).unwrap();
    let tag_col = te.columns.iter().find(|c| c.name == "tag").unwrap().id;
    let mut txn = env
        .txn_manager
        .begin(IsolationLevel::ReadCommitted)
        .unwrap();
    let scan_ctx = Arc::new(ExecutionContext::new(
        Arc::clone(&env.catalog),
        Arc::clone(&env.wal),
        Arc::clone(&env.pool),
        Arc::clone(&env.disk),
        txn.txn_id as u32,
        txn.snapshot.clone(),
    ));
    let logical: Vec<zyron_planner::logical::LogicalColumn> = te
        .columns
        .iter()
        .map(|c| zyron_planner::logical::LogicalColumn {
            table_idx: Some(0),
            column_id: c.id,
            name: c.name.clone(),
            type_id: c.type_id,
            nullable: c.nullable,
            fractional_digits: c.fractional_digits,
        })
        .collect();
    let mut op = zyron_executor::operator::column_scan::ColumnScanOperator::new_for_dml(
        Arc::clone(&scan_ctx),
        logs_id,
        logical,
        None,
    )
    .unwrap();
    let mut rebuilt = 0usize;
    while let Some(eb) = op.next().await.unwrap() {
        let locs = eb.locators.clone().expect("dml scan emits locators");
        rebuilt += zyron_executor::operator::modify::rebuild_btree_index_from_batch(
            te.as_ref(),
            &eb.batch,
            &locs,
            &[tag_col],
            &fresh,
        );
    }
    env.txn_manager.abort(&mut txn).unwrap();
    assert_eq!(rebuilt, 5, "every folded row re-entered the index");

    btree_reg.remove_sync(&idx_id.0);
    let _ = btree_reg.insert_sync(idx_id.0, fresh);

    // the rebuilt index serves the folded row through the point lookup
    let b = Arc::clone(&btree_reg);
    let rows = run_sql(
        &env,
        move |ctx| ctx.btree_indexes = Some(b),
        "SELECT id FROM logs WHERE tag = 30",
    )
    .await
    .expect("lookup through the rebuilt index");
    let total: usize = rows.iter().map(|b| b.num_rows).sum();
    assert_eq!(total, 1, "one hit through the rebuilt index");
    let hit = rows.iter().find(|b| b.num_rows > 0).unwrap();
    assert!(
        matches!(hit.columns[0].data.get_scalar(0), ScalarValue::Int64(3)),
        "the folded row id is returned"
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn parent_side_fk_actions_reach_folded_child_rows() {
    use zyron_common::DocRegistry;

    let env = setup().await;
    let reg = Arc::new(DocRegistry::new());
    let schema = env
        .catalog
        .get_table_by_id(env.parent_id)
        .unwrap()
        .schema_id;

    let fk = |ref_table: &str, action: ReferentialAction| TableConstraint {
        name: None,
        kind: TableConstraintKind::ForeignKey {
            columns: vec!["pk".into()],
            ref_table: ref_table.into(),
            ref_columns: vec!["k".into()],
            on_delete: action,
            on_update: ReferentialAction::Restrict,
        },
        enforced: true,
        on_violation: zyron_parser::ast::ViolationAction::Fail,
    };
    let cols = || vec![col("k", DataType::BigInt), col("v", DataType::BigInt)];
    let child_cols = || vec![col("ck", DataType::BigInt), col("pk", DataType::BigInt)];

    env.catalog
        .create_table(schema, "p2", &cols(), &[])
        .await
        .unwrap();
    env.catalog
        .create_table(
            schema,
            "c2",
            &child_cols(),
            &[fk("p2", ReferentialAction::Restrict)],
        )
        .await
        .unwrap();
    env.catalog
        .create_table(schema, "p3", &cols(), &[])
        .await
        .unwrap();
    let c3_id = env
        .catalog
        .create_table(
            schema,
            "c3",
            &child_cols(),
            &[fk("p3", ReferentialAction::Cascade)],
        )
        .await
        .unwrap();

    for sql in [
        "INSERT INTO p2 (k, v) VALUES (1, 1), (2, 2), (3, 3), (4, 4), (5, 5)",
        "INSERT INTO c2 (ck, pk) VALUES (1, 3), (2, 3), (3, 4), (4, 5), (5, 1)",
        "INSERT INTO p3 (k, v) VALUES (1, 1), (2, 2), (3, 3), (4, 4), (5, 5)",
        "INSERT INTO c3 (ck, pk) VALUES (1, 3), (2, 3), (3, 4), (4, 5), (5, 1)",
    ] {
        run_sql(&env, |_| {}, sql).await.expect("seed insert");
    }

    let rows_folded = fold_all(&env, &reg);
    assert!(
        rows_folded >= 20,
        "all four tables folded, got {rows_folded}"
    );
    for tid in [c3_id] {
        assert!(
            !env.catalog
                .get_table_by_id(tid)
                .unwrap()
                .columnar
                .segments
                .is_empty(),
            "child rows are columnar resident"
        );
    }

    // RESTRICT: folded child rows referencing k = 3 block the parent delete
    let err = run_sql(&env, |_| {}, "DELETE FROM p2 WHERE k = 3")
        .await
        .expect_err("folded children must block the parent delete");
    assert!(
        err.to_lowercase().contains("foreign key"),
        "error must name the FK violation, got: {err}"
    );

    // an unreferenced parent row deletes fine
    run_sql(&env, |_| {}, "DELETE FROM p2 WHERE k = 2")
        .await
        .expect("unreferenced parent row deletes");

    // CASCADE: deleting the parent removes its folded children
    run_sql(&env, |_| {}, "DELETE FROM p3 WHERE k = 3")
        .await
        .expect("cascade delete of folded children");
    let rows = run_sql(&env, |_| {}, "SELECT ck FROM c3 WHERE pk = 3")
        .await
        .expect("select cascaded children");
    let total: usize = rows.iter().map(|b| b.num_rows).sum();
    assert_eq!(total, 0, "cascade removed the folded children");
    let rows = run_sql(&env, |_| {}, "SELECT ck FROM c3")
        .await
        .expect("select surviving children");
    let total: usize = rows.iter().map(|b| b.num_rows).sum();
    assert_eq!(total, 3, "unreferenced folded children survive");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn spatial_rebuild_from_table_covers_folded_rows() {
    use zyron_common::DocRegistry;
    use zyron_executor::column::ScalarValue;

    let env = setup().await;
    let reg = Arc::new(DocRegistry::new());
    let spatial_mgr = Arc::new(zyron_types::spatial_index::SpatialIndexManager::new());

    let schema = env
        .catalog
        .get_table_by_id(env.parent_id)
        .unwrap()
        .schema_id;
    let places_id = env
        .catalog
        .create_table(
            schema,
            "places",
            &[col("id", DataType::BigInt), col("loc", DataType::Geometry)],
            &[],
        )
        .await
        .unwrap();
    let idx_id = env
        .catalog
        .create_index(
            places_id,
            schema,
            "places_loc_rtree",
            &["loc".to_string()],
            false,
            zyron_catalog::IndexType::Spatial,
        )
        .await
        .unwrap();
    spatial_mgr.create_index(idx_id.0, 2, 4326);

    let (s, r) = (Arc::clone(&spatial_mgr), Arc::clone(&reg));
    run_sql(
        &env,
        move |ctx| {
            ctx.set_spatial_manager(s);
            ctx.doc_registry = Some(r);
        },
        "INSERT INTO places (id, loc) VALUES \
         (1, st_geom_from_text('POINT(10 10)')), (2, st_geom_from_text('POINT(20 20)')), \
         (3, st_geom_from_text('POINT(30 30)')), (4, st_geom_from_text('POINT(40 40)')), \
         (5, st_geom_from_text('POINT(50 50)'))",
    )
    .await
    .expect("insert with spatial maintenance");

    let folded = fold_all(&env, &reg);
    assert!(folded >= 5, "places rows folded, got {folded}");
    let te = env.catalog.get_table_by_id(places_id).unwrap();
    assert!(
        !te.columnar.segments.is_empty(),
        "places is columnar resident"
    );

    // simulate a lost snapshot: a fresh manager rebuilt from the table must
    // cover the folded rows through the columnar pass, the heap slots are
    // already vacuumed by the fold
    let fresh = zyron_types::spatial_index::SpatialIndexManager::new();
    fresh.create_index(idx_id.0, 2, 4326);
    let idx_entry = env
        .catalog
        .get_indexes_for_table(places_id)
        .into_iter()
        .find(|i| i.id == idx_id)
        .expect("index entry");
    zyron_server::rebuild_spatial_index_from_table(
        &fresh,
        &reg,
        idx_id.0,
        2,
        &env.pool,
        &env.disk,
        &env.catalog,
        &env.wal,
        &env.txn_manager,
        idx_entry.as_ref(),
        te.as_ref(),
    )
    .await
    .expect("rebuild from table");
    let tree = fresh.get(idx_id.0).expect("tree registered");
    assert_eq!(tree.len(), 5, "rebuild indexed every folded row");

    // the rebuilt tree serves the folded row through SQL
    let fresh = Arc::new(fresh);
    let (s, r) = (Arc::clone(&fresh), Arc::clone(&reg));
    let rows = run_sql(
        &env,
        move |ctx| {
            ctx.set_spatial_manager(s);
            ctx.doc_registry = Some(r);
        },
        "SELECT id FROM places WHERE st_dwithin(loc, st_make_point(10.0, 10.0), 5.0)",
    )
    .await
    .expect("spatial search after rebuild");
    let total: usize = rows.iter().map(|b| b.num_rows).sum();
    assert_eq!(total, 1, "the folded point is served by the rebuilt tree");
    let hit = rows.iter().find(|b| b.num_rows > 0).unwrap();
    assert!(
        matches!(hit.columns[0].data.get_scalar(0), ScalarValue::Int64(1)),
        "the folded row id is returned"
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn no_index_fk_existence_check_sees_folded_parent_rows() {
    let env = setup().await;

    // parent.k carries no index, so the child-side FK check falls back to
    // the visible-key set. All parent rows are folded by setup, a child
    // referencing one must still insert
    run_sql(&env, |_| {}, "INSERT INTO child (ck, pk) VALUES (20, 9)")
        .await
        .expect("child referencing a folded parent inserts");

    // a missing parent still violates the constraint
    let err = run_sql(&env, |_| {}, "INSERT INTO child (ck, pk) VALUES (21, 999)")
        .await
        .expect_err("missing parent must violate the FK");
    assert!(
        err.to_lowercase().contains("foreign key"),
        "error must name the FK violation, got: {err}"
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn streaming_upsert_replaces_and_deletes_folded_rows() {
    use zyron_auth::privilege::{GrantEntry, ObjectType, PrivilegeState, PrivilegeType};
    use zyron_auth::{ClassificationLevel, QueryLimits, RoleId, SecurityContext, UserId};
    use zyron_common::DocRegistry;
    use zyron_executor::column::ScalarValue;
    use zyron_streaming::{CdfChange, ZyronUpsertSink};

    let env = setup().await;
    let reg = Arc::new(DocRegistry::new());
    let schema = env
        .catalog
        .get_table_by_id(env.parent_id)
        .unwrap()
        .schema_id;

    let target_id = env
        .catalog
        .create_table(
            schema,
            "stream_target",
            &[col("k", DataType::BigInt), col("v", DataType::BigInt)],
            &[TableConstraint {
                name: None,
                kind: TableConstraintKind::PrimaryKey(vec!["k".into()]),
                enforced: true,
                on_violation: zyron_parser::ast::ViolationAction::Fail,
            }],
        )
        .await
        .unwrap();

    run_sql(
        &env,
        |_| {},
        "INSERT INTO stream_target (k, v) VALUES (1, 10), (2, 20), (3, 30), (4, 40), (5, 50), (6, 60)",
    )
    .await
    .expect("seed insert");

    let folded = fold_all(&env, &reg);
    assert!(folded >= 6, "target rows folded, got {folded}");
    let te = env.catalog.get_table_by_id(target_id).unwrap();
    assert!(
        !te.columnar.segments.is_empty(),
        "target is columnar resident"
    );
    let seg = te.columnar.segments[0].clone();

    // sink dependencies: heap handle, security manager with INSERT + DELETE
    // granted, session context carrying the granted role
    let heap = Arc::new(
        HeapFile::new(
            Arc::clone(&env.disk),
            Arc::clone(&env.pool),
            HeapFileConfig {
                heap_file_id: te.heap_file_id,
                fsm_file_id: te.fsm_file_id,
            },
        )
        .unwrap(),
    );
    let auth_storage: Arc<dyn zyron_auth::storage::AuthStorage> = Arc::new(
        zyron_auth::HeapAuthStorage::new(Arc::clone(&env.disk), Arc::clone(&env.pool)).unwrap(),
    );
    let sm = Arc::new(
        zyron_auth::SecurityManager::new(auth_storage)
            .await
            .unwrap(),
    );
    let role = RoleId(5);
    for privilege in [PrivilegeType::Insert, PrivilegeType::Delete] {
        sm.privilege_store
            .grant(GrantEntry {
                grantee: role,
                privilege,
                object_type: ObjectType::Table,
                object_id: target_id.0,
                columns: None,
                state: PrivilegeState::Grant,
                with_grant_option: false,
                granted_by: RoleId(0),
                valid_from: None,
                valid_until: None,
                time_window: None,
                object_pattern: None,
                no_inherit: false,
                mask_function: None,
            })
            .unwrap();
    }
    let attrs = zyron_auth::SessionAttributes {
        role_id: role,
        department: None,
        region: None,
        clearance: ClassificationLevel::Public,
        ip_address: "127.0.0.1".to_string(),
        connection_time: 0,
        custom: std::collections::HashMap::new(),
    };
    let sec_ctx = SecurityContext::new(
        UserId(1),
        role,
        vec![role],
        vec![role],
        ClassificationLevel::Public,
        attrs,
        None,
        QueryLimits::default(),
    );

    let sink = ZyronUpsertSink::new(
        target_id.0,
        vec![0u16],
        vec![zyron_common::TypeId::Int64, zyron_common::TypeId::Int64],
        Arc::clone(&env.catalog),
        heap,
        Arc::clone(&env.txn_manager),
        Arc::clone(&env.wal),
        Arc::new(parking_lot::Mutex::new(sec_ctx)),
        sm,
        Arc::new(zyron_common::TableIOStatsRegistry::new()),
    )
    .unwrap();

    // the construction scan found every folded row through its Columnar locator
    assert_eq!(sink.live_row_count(), 6, "folded rows are in the PK map");

    // an update postimage for a folded PK supersedes the folded row and lands
    // the new image on the heap, exactly one live row with the new value
    tokio::task::block_in_place(|| {
        sink.write_batch(vec![CdfChange {
            commit_version: 1,
            commit_timestamp: 0,
            change_type: zyron_cdc::ChangeType::UpdatePostimage,
            row_data: encode_row2(5, 555),
            primary_key_data: Vec::new(),
        }])
    })
    .expect("upsert update of a folded row");
    assert_eq!(sink.live_row_count(), 6, "update replaced, not duplicated");

    // rows folded in insert order, k = 5 sits at offset 4
    let store = ColumnarPatchManager::global(&env.columnar_dir)
        .store(target_id.0 as u64)
        .unwrap();
    assert!(
        store
            .row_overlay(seg.file_id, seg.sys_rowid_lo + 4)
            .is_some_and(|o| !o.supersedes.is_empty()),
        "folded row superseded by the upsert"
    );

    let rows = run_sql(&env, |_| {}, "SELECT v FROM stream_target WHERE k = 5")
        .await
        .expect("select upserted row");
    let total: usize = rows.iter().map(|b| b.num_rows).sum();
    assert_eq!(total, 1, "exactly one live row for the upserted PK");
    let hit = rows.iter().find(|b| b.num_rows > 0).unwrap();
    assert!(
        matches!(hit.columns[0].data.get_scalar(0), ScalarValue::Int64(555)),
        "the new value is visible"
    );

    // a delete change for a folded PK removes it
    tokio::task::block_in_place(|| {
        sink.write_batch(vec![CdfChange {
            commit_version: 2,
            commit_timestamp: 0,
            change_type: zyron_cdc::ChangeType::Delete,
            row_data: encode_row2(6, 60),
            primary_key_data: Vec::new(),
        }])
    })
    .expect("upsert delete of a folded row");
    assert_eq!(sink.live_row_count(), 5, "delete removed the folded row");
    let rows = run_sql(&env, |_| {}, "SELECT v FROM stream_target WHERE k = 6")
        .await
        .expect("select deleted row");
    let total: usize = rows.iter().map(|b| b.num_rows).sum();
    assert_eq!(total, 0, "the folded row is gone");

    // untouched folded rows stay visible
    let rows = run_sql(&env, |_| {}, "SELECT v FROM stream_target")
        .await
        .expect("select remaining rows");
    let total: usize = rows.iter().map(|b| b.num_rows).sum();
    assert_eq!(total, 5, "five live rows remain");
}
