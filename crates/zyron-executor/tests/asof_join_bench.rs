//! ASOF Join Benchmark Suite
//!
//! The merge rate and the memory it holds, which are the two things the
//! operator's shape is chosen for: one pass over each input, and state that
//! is one row rather than a table of rows.
//!
//! Performance Targets:
//! | Test                                    | Metric     | Target                    |
//! |-----------------------------------------|------------|---------------------------|
//! | ASOF JOIN 10M x 1M, both pre-sorted     | throughput | 20M left rows/s           |
//! | ASOF JOIN 10M x 1M, both unsorted       | latency    | <= 2x the two sorts alone |
//! | ASOF JOIN peak memory, pre-sorted       | bytes      | 2 batches per side        |
//!
//! Validation Requirements:
//! - Each benchmark runs 5 iterations
//! - Results averaged across all 5 runs
//! - Pass/fail determined by average performance
//! - Individual runs logged for variance analysis
//!
//! Run: cargo test --release -p zyron-executor --test asof_join_bench -- --nocapture

use std::alloc::{GlobalAlloc, Layout};
use std::sync::atomic::{AtomicI64, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use zyron_bench_harness::*;
use zyron_common::TypeId;
use zyron_executor::batch::{BATCH_SIZE, ColumnBuilder, DataBatch};
use zyron_executor::column::ScalarValue;
use zyron_executor::context::ExecutionContext;
use zyron_executor::operator::asof_join::AsofJoinOperator;
use zyron_executor::operator::sort::SortOperator;
use zyron_executor::operator::{ExecutionBatch, Operator, OperatorResult};
use zyron_parser::ast::JoinType;
use zyron_planner::binder::{BoundExpr, BoundOrderBy, ColumnRef};
use zyron_planner::logical::AsofDirection;
use zyron_planner::logical::LogicalColumn;
use zyron_planner::physical::AsofJoinSpec;

// =============================================================================
// Counting allocator
//
// The memory target is about what the operator holds while it runs, which no
// system-wide reading can separate from everything else on the machine. Every
// allocation this process makes is counted instead, so the figure is the
// engine's own and nothing else's.
// =============================================================================

struct CountingAllocator;

/// Bytes allocated inside the window minus bytes freed inside it, which is
/// net growth rather than total live memory.
///
/// Signed, because a window frees blocks allocated before it opened and the
/// running figure legitimately goes below where it started. An unsigned
/// counter wraps on the first such free and reports the whole address space
static NET_BYTES: AtomicI64 = AtomicI64::new(0);
/// The largest net growth seen, which is what the operator held at its worst
static PEAK_BYTES: AtomicI64 = AtomicI64::new(0);
/// Peak tracking costs an atomic per allocation, so it is off except inside
/// the window a measurement covers
static TRACKING: AtomicUsize = AtomicUsize::new(0);

/// Records a change in bytes held and keeps the high-water mark.
#[inline]
fn account(delta: i64) {
    if TRACKING.load(Ordering::Relaxed) != 1 {
        return;
    }
    let net = NET_BYTES.fetch_add(delta, Ordering::Relaxed) + delta;
    if delta > 0 {
        PEAK_BYTES.fetch_max(net, Ordering::Relaxed);
    }
}

unsafe impl GlobalAlloc for CountingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let ptr = unsafe { mimalloc::MiMalloc.alloc(layout) };
        if !ptr.is_null() {
            account(layout.size() as i64);
        }
        ptr
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        account(-(layout.size() as i64));
        unsafe { mimalloc::MiMalloc.dealloc(ptr, layout) }
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        let out = unsafe { mimalloc::MiMalloc.realloc(ptr, layout, new_size) };
        if !out.is_null() {
            account(new_size as i64 - layout.size() as i64);
        }
        out
    }
}

#[global_allocator]
static GLOBAL: CountingAllocator = CountingAllocator;

/// Starts counting from zero.
fn begin_tracking() {
    NET_BYTES.store(0, Ordering::SeqCst);
    PEAK_BYTES.store(0, Ordering::SeqCst);
    TRACKING.store(1, Ordering::SeqCst);
}

/// Stops counting and returns the largest net growth seen.
fn end_tracking() -> i64 {
    TRACKING.store(0, Ordering::SeqCst);
    PEAK_BYTES.load(Ordering::SeqCst).max(0)
}

/// The suites run one at a time
static BENCHMARK_LOCK: Mutex<()> = Mutex::new(());

const VALIDATION_RUNS: usize = 5;

/// One merge pass over two ordered inputs
const ASOF_SORTED_TARGET_ROWS_SEC: f64 = 20_000_000.0;
/// The whole join over unsorted inputs, against the two sorts alone.
/// Anything above this means the merge is doing more than one pass
const ASOF_UNSORTED_RATIO_LIMIT: f64 = 2.0;
/// One input batch per side plus the held row and the output batch being
/// filled. Measured in batches of the left side's own width, so the figure
/// says the same thing whatever a batch happens to hold
const ASOF_PEAK_BATCHES_PER_SIDE: f64 = 2.0;

// =============================================================================
// Fixtures
// =============================================================================

struct MemoryOperator {
    batches: Vec<DataBatch>,
    at: usize,
}

impl MemoryOperator {
    fn new(batches: Vec<DataBatch>) -> Self {
        Self { batches, at: 0 }
    }
}

impl Operator for MemoryOperator {
    fn next(&mut self) -> OperatorResult<'_> {
        Box::pin(async move {
            if self.at >= self.batches.len() {
                return Ok(None);
            }
            // The batch is moved out, so the feeding side holds nothing of
            // it and whatever stays alive is the merge's own state
            let batch = std::mem::replace(&mut self.batches[self.at], DataBatch::empty());
            self.at += 1;
            Ok(Some(ExecutionBatch::new(batch)))
        })
    }
}

/// A (key, ts, payload) side. `keys` distinct symbols, timestamps ascending
/// within each, which is the order the merge reads.
///
/// `shuffle` steps the timestamps by a stride coprime with the row count, so
/// the rows arrive in neither key nor timestamp order and the sort has real
/// work to do.
fn side(rows: usize, keys: i64, table_idx: usize, shuffle: bool) -> Vec<DataBatch> {
    let mut ordered: Vec<(i64, i64)> = Vec::with_capacity(rows);
    for r in 0..rows {
        ordered.push((r as i64 % keys, (r as i64) / keys));
    }
    ordered.sort_unstable();
    if shuffle {
        let stride = 7919usize;
        let mut shuffled = Vec::with_capacity(rows);
        for i in 0..rows {
            shuffled.push(ordered[(i * stride) % rows]);
        }
        ordered = shuffled;
    }

    let mut batches = Vec::with_capacity(rows.div_ceil(BATCH_SIZE));
    let mut produced = 0usize;
    while produced < rows {
        let n = BATCH_SIZE.min(rows - produced);
        let mut key = ColumnBuilder::new(TypeId::Int64, n);
        let mut ts = ColumnBuilder::new(TypeId::Int64, n);
        let mut payload = ColumnBuilder::new(TypeId::Int64, n);
        for r in 0..n {
            let (k, t) = ordered[produced + r];
            key.push(&ScalarValue::Int64(k));
            ts.push(&ScalarValue::Int64(t));
            payload.push(&ScalarValue::Int64((produced + r) as i64));
        }
        batches.push(DataBatch::new(vec![
            key.finish(),
            ts.finish(),
            payload.finish(),
        ]));
        produced += n;
    }
    let _ = table_idx;
    batches
}

fn schema_of(table_idx: usize) -> Vec<LogicalColumn> {
    ["key", "ts", "payload"]
        .iter()
        .enumerate()
        .map(|(i, name)| LogicalColumn {
            table_idx: Some(table_idx),
            column_id: zyron_catalog::ColumnId(i as u16),
            name: (*name).to_string(),
            type_id: TypeId::Int64,
            nullable: false,
            fractional_digits: None,
        })
        .collect()
}

fn column(table_idx: usize, id: u16) -> BoundExpr {
    BoundExpr::ColumnRef(ColumnRef {
        table_idx,
        column_id: zyron_catalog::ColumnId(id),
        type_id: TypeId::Int64,
        nullable: false,
        fractional_digits: None,
    })
}

/// The join under test: each left row takes the latest right row at or
/// before its own timestamp, within its own key.
fn spec(join_type: JoinType) -> Box<AsofJoinSpec> {
    Box::new(AsofJoinSpec {
        equality_keys: vec![(column(0, 0), column(1, 0))],
        match_left: column(0, 1),
        match_right: column(1, 1),
        direction: AsofDirection::Backward,
        tolerance: None,
        join_type,
        left_schema: schema_of(0),
        right_schema: schema_of(1),
        left_sorted: true,
        right_sorted: true,
    })
}

/// The order both sides walk in.
fn sort_keys(table_idx: usize) -> Vec<BoundOrderBy> {
    vec![
        BoundOrderBy {
            expr: column(table_idx, 0),
            asc: true,
            nulls_first: false,
        },
        BoundOrderBy {
            expr: column(table_idx, 1),
            asc: true,
            nulls_first: false,
        },
    ]
}

/// A context with a real catalog behind it, which is what the operator's
/// cancellation and batch size come from.
async fn context() -> (Arc<ExecutionContext>, tempfile::TempDir) {
    let tmp = tempfile::TempDir::new().expect("temp dir");
    let (data_dir, wal_dir) = zyron_bench_harness::create_dirs(tmp.path()).expect("dirs");
    let wal = Arc::new(
        zyron_wal::WalWriter::new(zyron_bench_harness::wal_config(&wal_dir)).expect("wal"),
    );
    let disk = Arc::new(
        zyron_storage::DiskManager::new(zyron_bench_harness::disk_config(&data_dir))
            .await
            .expect("disk"),
    );
    let pool = Arc::new(zyron_buffer::BufferPool::new(
        zyron_bench_harness::buffer_pool_config(),
    ));
    let storage = Arc::new(
        zyron_catalog::HeapCatalogStorage::new(Arc::clone(&disk), Arc::clone(&pool))
            .expect("storage"),
    );
    let cache = Arc::new(zyron_catalog::CatalogCache::new(256, 64));
    let catalog = Arc::new(
        zyron_catalog::Catalog::new(storage, cache, Arc::clone(&wal))
            .await
            .expect("catalog"),
    );
    let txn_manager = Arc::new(zyron_storage::txn::TransactionManager::new(Arc::clone(
        &wal,
    )));
    let txn = txn_manager
        .begin(zyron_storage::txn::IsolationLevel::ReadCommitted)
        .expect("begin");
    let ctx = ExecutionContext::new(catalog, wal, pool, disk, txn.txn_id, txn.snapshot.clone());
    (Arc::new(ctx), tmp)
}

async fn drain(op: &mut dyn Operator) -> usize {
    let mut rows = 0usize;
    while let Some(batch) = op.next().await.expect("operator") {
        rows += batch.num_rows();
    }
    rows
}

// =============================================================================
// Test 1: Merge rate over pre-sorted inputs
// =============================================================================

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn test_asof_join_10m_by_1m_pre_sorted() {
    zyron_bench_harness::init("asof_join");
    let _guard = BENCHMARK_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    const LEFT_ROWS: usize = 10_000_000;
    const RIGHT_ROWS: usize = 1_000_000;
    const KEYS: i64 = 1000;

    tprintln!("\n=== ASOF Join Merge Rate, Pre-sorted ===");
    tprintln!(
        "Left rows: {}, right rows: {}, keys: {}",
        LEFT_ROWS,
        RIGHT_ROWS,
        KEYS
    );

    let (ctx, _tmp) = context().await;
    let left = side(LEFT_ROWS, KEYS, 0, false);
    let right = side(RIGHT_ROWS, KEYS, 1, false);
    let mut runs = Vec::with_capacity(VALIDATION_RUNS);

    let util_before = take_util_snapshot();
    for run in 0..VALIDATION_RUNS {
        let mut op = AsofJoinOperator::new(
            Box::new(MemoryOperator::new(left.clone())),
            Box::new(MemoryOperator::new(right.clone())),
            spec(JoinType::Left),
            Arc::clone(&ctx),
        );
        let start = Instant::now();
        let produced = drain(&mut op).await;
        let duration = start.elapsed();
        assert_eq!(
            produced, LEFT_ROWS,
            "run {run}: the LEFT form keeps every left row"
        );
        let per_sec = LEFT_ROWS as f64 / duration.as_secs_f64();
        tprintln!(
            "  Run {}/{}: {} left rows/sec ({:?})",
            run + 1,
            VALIDATION_RUNS,
            format_with_commas(per_sec),
            duration
        );
        runs.push(per_sec);
    }
    record_test_util("ASOF merge rate", util_before, take_util_snapshot());

    let result = validate_metric_with_unit(
        "ASOF merge rate",
        "ASOF pre-sorted",
        " left rows/s",
        runs,
        ASOF_SORTED_TARGET_ROWS_SEC,
        true,
    );
    assert!(
        result.passed,
        "ASOF avg {:.0} left rows/s < target {:.0}",
        result.average, ASOF_SORTED_TARGET_ROWS_SEC
    );
}

// =============================================================================
// Test 2: The whole join over unsorted inputs, against the sorts alone
// =============================================================================

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn test_asof_join_unsorted_against_the_sorts_alone() {
    zyron_bench_harness::init("asof_join");
    let _guard = BENCHMARK_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    // A tenth of the sorted run's rows, because this one sorts both sides
    // five times over and the ratio is what the target is about
    const LEFT_ROWS: usize = 1_000_000;
    const RIGHT_ROWS: usize = 100_000;
    const KEYS: i64 = 1000;

    tprintln!("\n=== ASOF Join Over Unsorted Inputs ===");
    tprintln!("Left rows: {}, right rows: {}", LEFT_ROWS, RIGHT_ROWS);

    let (ctx, _tmp) = context().await;
    let left = side(LEFT_ROWS, KEYS, 0, true);
    let right = side(RIGHT_ROWS, KEYS, 1, true);

    let mut sorts_only = Vec::with_capacity(VALIDATION_RUNS);
    let mut whole_join = Vec::with_capacity(VALIDATION_RUNS);
    let util_before = take_util_snapshot();
    for run in 0..VALIDATION_RUNS {
        // The two sorts alone, which is what the join has to pay before it
        // can merge anything
        let start = Instant::now();
        let mut sort_left = SortOperator::new(
            Box::new(MemoryOperator::new(left.clone())),
            sort_keys(0),
            schema_of(0),
            None,
        );
        let sorted_left = drain(&mut sort_left).await;
        let mut sort_right = SortOperator::new(
            Box::new(MemoryOperator::new(right.clone())),
            sort_keys(1),
            schema_of(1),
            None,
        );
        let sorted_right = drain(&mut sort_right).await;
        let sorts = start.elapsed();
        assert_eq!(sorted_left, LEFT_ROWS);
        assert_eq!(sorted_right, RIGHT_ROWS);

        // The same two sorts with the merge on top
        let start = Instant::now();
        let sort_left = SortOperator::new(
            Box::new(MemoryOperator::new(left.clone())),
            sort_keys(0),
            schema_of(0),
            None,
        );
        let sort_right = SortOperator::new(
            Box::new(MemoryOperator::new(right.clone())),
            sort_keys(1),
            schema_of(1),
            None,
        );
        let mut op = AsofJoinOperator::new(
            Box::new(sort_left),
            Box::new(sort_right),
            spec(JoinType::Left),
            Arc::clone(&ctx),
        );
        let produced = drain(&mut op).await;
        let joined = start.elapsed();
        assert_eq!(produced, LEFT_ROWS);

        tprintln!(
            "  Run {}/{}: sorts {:?}, whole join {:?}",
            run + 1,
            VALIDATION_RUNS,
            sorts,
            joined
        );
        sorts_only.push(sorts.as_secs_f64() * 1000.0);
        whole_join.push(joined.as_secs_f64() * 1000.0);
    }
    record_test_util("ASOF unsorted", util_before, take_util_snapshot());

    let sorts = record_metric("ASOF unsorted", "two sorts alone", "ms", sorts_only);
    let joined = record_metric("ASOF unsorted", "sorts plus merge", "ms", whole_join);
    let ratio = joined / sorts.max(f64::MIN_POSITIVE);
    tprintln!(
        "  The whole join costs {:.3}x the sorts alone (limit {:.2}x)",
        ratio,
        ASOF_UNSORTED_RATIO_LIMIT
    );
    record_metric("ASOF unsorted", "join over sorts", "x", vec![ratio]);
    assert!(
        ratio <= ASOF_UNSORTED_RATIO_LIMIT,
        "the join costs {ratio:.3}x the sorts alone, over the {ASOF_UNSORTED_RATIO_LIMIT:.2}x limit"
    );
}

// =============================================================================
// Test 3: What the merge holds while it runs
// =============================================================================

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn test_asof_join_holds_two_batches_per_side() {
    zyron_bench_harness::init("asof_join");
    let _guard = BENCHMARK_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    const RIGHT_ROWS: usize = 100_000;
    const KEYS: i64 = 1000;

    tprintln!("\n=== ASOF Join Peak Memory ===");

    let (ctx, _tmp) = context().await;
    // Three columns of eight bytes per row, which is what one input batch of
    // this fixture holds
    let batch_bytes = (BATCH_SIZE * 3 * 8) as f64;
    let budget = batch_bytes * ASOF_PEAK_BATCHES_PER_SIDE * 2.0;
    tprintln!(
        "  One batch holds {} bytes, so the budget is {} bytes",
        format_with_commas(batch_bytes),
        format_with_commas(budget)
    );

    // Two left sizes an order of magnitude apart. State that grows with the
    // input would show up as a peak that grows with it
    let mut peaks = Vec::new();
    for left_rows in [1_000_000usize, 10_000_000usize] {
        let left = side(left_rows, KEYS, 0, false);
        let right = side(RIGHT_ROWS, KEYS, 1, false);
        let mut op = AsofJoinOperator::new(
            Box::new(MemoryOperator::new(left)),
            Box::new(MemoryOperator::new(right)),
            spec(JoinType::Left),
            Arc::clone(&ctx),
        );
        // Counting starts once the fixtures are built, so what it measures
        // is the merge and the batches it hands upward
        begin_tracking();
        let produced = drain(&mut op).await;
        let peak = end_tracking() as f64;
        assert_eq!(produced, left_rows);
        tprintln!(
            "  {} left rows: peak {} bytes, {:.2} batches per side",
            format_with_commas(left_rows as f64),
            format_with_commas(peak),
            peak / (batch_bytes * 2.0)
        );
        peaks.push(peak);
    }

    let per_side: Vec<f64> = peaks.iter().map(|p| p / (batch_bytes * 2.0)).collect();
    let result = validate_metric_with_unit(
        "ASOF peak memory",
        "ASOF batches per side",
        " batches",
        per_side.clone(),
        ASOF_PEAK_BATCHES_PER_SIDE,
        false,
    );
    assert!(
        result.passed,
        "the merge held {:.2} batches per side, over the {:.1} budget",
        result.average, ASOF_PEAK_BATCHES_PER_SIDE
    );
    // Ten times the rows must not cost ten times the memory, which is what
    // "one pass holding one row" means and what a materializing merge would
    // fail
    let growth = peaks[1] / peaks[0].max(f64::MIN_POSITIVE);
    tprintln!("  Ten times the left rows cost {:.3}x the peak", growth);
    record_metric(
        "ASOF peak memory",
        "peak growth over 10x rows",
        "x",
        vec![growth],
    );
    assert!(
        growth < 2.0,
        "peak memory grew {growth:.2}x when the input grew tenfold, so the merge is holding rows"
    );
    let _ = Duration::from_secs(0);
}
