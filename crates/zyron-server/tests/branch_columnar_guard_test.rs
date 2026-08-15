//! Branch DML over columnar resident rows is isolated from the main line.
//!
//! A branch delete or update of a folded row writes to the branch's own
//! patch overlay. The main line keeps the row untouched, the branch sees
//! its own state, and pre fork patches stay visible on the branch through
//! the explicit copy record written before the branch's first write to a
//! row, mirroring heap page COW at row granularity.
//!
//! Run: cargo test -p zyron-server --test branch_columnar_guard_test

use std::sync::Arc;

use zyron_buffer::{BufferPool, BufferPoolConfig};
use zyron_catalog::storage::HeapCatalogStorage;
use zyron_catalog::{Catalog, CatalogCache, SYSTEM_DATABASE_ID};
use zyron_common::RowLocator;
use zyron_executor::batch::DataBatch;
use zyron_executor::column::{Column, ColumnData};
use zyron_executor::context::ExecutionContext;
use zyron_executor::operator::modify::{DeleteOperator, UpdateOperator};
use zyron_executor::operator::{ExecutionBatch, Operator, OperatorResult};
use zyron_parser::ast::{ColumnDef, DataType};
use zyron_storage::columnar::{ColumnarPatchManager, ZyrFileHeader, ZyrFileWriter};
use zyron_storage::txn::{IsolationLevel, TransactionManager};
use zyron_storage::{DiskManager, DiskManagerConfig};
use zyron_wal::{WalWriter, WalWriterConfig};

/// Yields one preloaded batch then exhausts
struct OneShot(Option<ExecutionBatch>);

impl Operator for OneShot {
    fn next(&mut self) -> OperatorResult<'_> {
        Box::pin(async move { Ok(self.0.take()) })
    }
}

fn col(name: &str) -> ColumnDef {
    ColumnDef {
        name: name.to_string(),
        data_type: DataType::BigInt,
        nullable: Some(true),
        default: None,
        constraints: vec![],
    }
}

struct Env {
    ctx_main: Arc<ExecutionContext>,
    ctx_branch: Arc<ExecutionContext>,
    table: zyron_catalog::TableId,
    columnar_dir: std::path::PathBuf,
    _tmp: tempfile::TempDir,
}

/// Catalog with one table that carries a registered columnar segment so the
/// patch store resolves, plus one main line context and one on branch 7
async fn setup() -> Env {
    let tmp = tempfile::tempdir().expect("tmp");
    let data_dir = tmp.path().join("data");
    let wal_dir = tmp.path().join("wal");
    std::fs::create_dir_all(&data_dir).unwrap();
    std::fs::create_dir_all(&wal_dir).unwrap();
    let columnar_dir = data_dir.join("columnar");
    std::fs::create_dir_all(&columnar_dir).unwrap();

    let disk = Arc::new(
        DiskManager::new(zyron_bench_harness::disk_config(data_dir.clone()))
            .await
            .unwrap(),
    );
    let pool = Arc::new(BufferPool::new(zyron_bench_harness::buffer_pool_config()));
    let wal = Arc::new(WalWriter::new(zyron_bench_harness::wal_config(wal_dir)).unwrap());
    let storage = Arc::new(HeapCatalogStorage::new(Arc::clone(&disk), Arc::clone(&pool)).unwrap());
    let cache = Arc::new(CatalogCache::new(64, 16));
    let catalog = Arc::new(
        Catalog::new(storage, cache, Arc::clone(&wal))
            .await
            .unwrap(),
    );
    let schema = catalog
        .create_schema(SYSTEM_DATABASE_ID, "public", "t")
        .await
        .unwrap();
    let table = catalog
        .create_table(schema, "metrics", &[col("k"), col("v")], &[])
        .await
        .unwrap();

    // register a minimal real .zyr so columnar_patch_store resolves the dir
    let seg_path = columnar_dir.join("table_test_seg.zyr");
    let writer = ZyrFileWriter::create(
        &seg_path,
        ZyrFileHeader {
            format_version: 1,
            column_count: 0,
            row_count: 0,
            table_id: table.0 as u64,
            xmin_range_lo: 0,
            xmin_range_hi: 0,
            xmax_range_lo: 0,
            xmax_range_hi: 0,
            primary_key_column_id: 0,
            sort_order: zyron_storage::columnar::SortOrder::None,
        },
    )
    .unwrap();
    writer.finalize(false).unwrap();
    let mut te = (*catalog.get_table_by_id(table).unwrap()).clone();
    te.columnar
        .segments
        .push(zyron_catalog::schema::ColumnarSegmentEntry {
            file_id: 1,
            path: seg_path.to_string_lossy().into_owned(),
            row_count: 0,
            sys_rowid_lo: 0,
            sys_rowid_hi: 0,
            sys_xmin_lo: 0,
            sys_xmin_hi: 0,
            // No clustering policy is declared here, which is what zero means
            cluster_spec_id: 0,
            // Hot, the tier the fold writes into
            storage_tier: 0,
        });
    catalog.update_table(te).await.unwrap();

    let txn_manager = TransactionManager::new(Arc::clone(&wal));
    let make_ctx = |branch: Option<u64>| {
        let txn = txn_manager.begin(IsolationLevel::ReadCommitted).unwrap();
        let mut ctx = ExecutionContext::new(
            Arc::clone(&catalog),
            Arc::clone(&wal),
            Arc::clone(&pool),
            Arc::clone(&disk),
            txn.txn_id as u32,
            txn.snapshot.clone(),
        );
        ctx.active_branch_id = branch;
        Arc::new(ctx)
    };
    let ctx_main = make_ctx(None);
    let ctx_branch = make_ctx(Some(7));
    Env {
        ctx_main,
        ctx_branch,
        table,
        columnar_dir,
        _tmp: tmp,
    }
}

/// One row batch tagged as columnar resident, rowid 0 in file 1
fn columnar_batch() -> ExecutionBatch {
    let batch = DataBatch::new(vec![
        Column::new(ColumnData::Int64(vec![1]), zyron_common::TypeId::Int64),
        Column::new(ColumnData::Int64(vec![100]), zyron_common::TypeId::Int64),
    ]);
    ExecutionBatch::with_locators(
        batch,
        vec![RowLocator::Columnar {
            file_id: 1,
            sys_rowid: 0,
        }],
    )
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn branch_delete_of_columnar_row_stays_off_the_main_line() {
    let env = setup().await;
    let mut op = DeleteOperator::new(
        Box::new(OneShot(Some(columnar_batch()))),
        Arc::clone(&env.ctx_branch),
        env.table,
    );
    op.next().await.expect("branch delete succeeds");

    let store = ColumnarPatchManager::global(&env.columnar_dir)
        .store(env.table.0 as u64)
        .unwrap();
    assert!(
        store.row_overlay(1, 0).is_none(),
        "main line overlay untouched by the branch delete"
    );
    let branch_view = store
        .row_overlay_on(7, 1, 0)
        .expect("branch sees its delete");
    assert!(
        !branch_view.supersedes.is_empty(),
        "branch overlay carries the supersede"
    );
    assert!(
        store.row_overlay_on(3, 1, 0).is_none(),
        "an unrelated branch sees the main line, which has nothing"
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn branch_write_copies_pre_fork_main_patches_and_shields_later_ones() {
    let env = setup().await;
    let store = ColumnarPatchManager::global(&env.columnar_dir)
        .store(env.table.0 as u64)
        .unwrap();

    // pre fork main line patch on column 0, the k column, distinct from
    // the v column the branch assigns below
    store
        .append_value_patch(0, 1, 0, 0, 40, 1, &111i64.to_le_bytes())
        .unwrap();

    // branch update assigns v = 4242, touching the row so the copy record
    // snapshots the pre fork main state into the branch first
    let te = env.ctx_branch.catalog.get_table_by_id(env.table).unwrap();
    let v_id = te.columns.iter().find(|c| c.name == "v").unwrap().id;
    let assign = zyron_planner::binder::BoundAssignment {
        column_id: v_id,
        value: zyron_planner::binder::BoundExpr::Literal {
            value: zyron_parser::ast::LiteralValue::Integer(4242),
            type_id: zyron_common::TypeId::Int64,
        },
    };
    let mut op = UpdateOperator::new(
        Box::new(OneShot(Some(columnar_batch()))),
        Arc::clone(&env.ctx_branch),
        env.table,
        vec![assign],
        vec![],
        vec![],
    );
    op.next().await.expect("branch update succeeds");

    // later main line patch must not leak into the branch view
    store
        .append_value_patch(0, 1, 0, 0, 90, 2, &999i64.to_le_bytes())
        .unwrap();

    let branch_view = store.row_overlay_on(7, 1, 0).expect("branch shadowed row");
    let chain = branch_view.patches.get(&0).expect("copied pre fork chain");
    assert_eq!(
        chain.len(),
        1,
        "branch copied exactly the pre fork patch on col 1"
    );
    assert_eq!(
        chain[0].patch_xid, 40,
        "pre fork patch visible on the branch"
    );
    let v_chain = branch_view
        .patches
        .get(&(v_id.0 as u32))
        .expect("branch's own assignment chain");
    assert_eq!(v_chain.len(), 1, "branch wrote its own v patch");

    let main_view = store.row_overlay(1, 0).expect("main keeps its own chain");
    assert_eq!(
        main_view.patches.get(&0).map(|c| c.len()),
        Some(2),
        "main line accumulated both of its k patches"
    );
    assert!(
        main_view.patches.get(&(v_id.0 as u32)).is_none(),
        "the branch's v assignment never reached the main line"
    );

    // main delete after the branch touch stays off the branch too
    let mut del = DeleteOperator::new(
        Box::new(OneShot(Some(columnar_batch()))),
        Arc::clone(&env.ctx_main),
        env.table,
    );
    del.next().await.expect("main delete succeeds");
    let branch_view = store.row_overlay_on(7, 1, 0).expect("branch row survives");
    assert!(
        branch_view.supersedes.is_empty(),
        "main line delete does not supersede the branch's shadowed row"
    );
}
