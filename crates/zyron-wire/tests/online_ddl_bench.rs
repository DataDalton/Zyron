//! Online Heap DDL Benchmark Suite
//!
//! Measures the paths a schema change takes while the table stays open, and
//! the two guards that protect the workload the change runs beside, the decode
//! cost every read now pays and the foreground latency a build is allowed to
//! cost. Every measurement drives the same code a server runs, through the DDL
//! dispatcher, the executor and the storage engine.
//!
//! Performance Targets (Minimum Threshold):
//! | Operation                                       | Metric     | Minimum   |
//! |-------------------------------------------------|------------|-----------|
//! | ADD COLUMN on a 10M-row heap table              | latency    | 50ms      |
//! | DROP COLUMN on a 10M-row heap table             | latency    | 50ms      |
//! | Compatible SET TYPE on a 10M-row heap table     | latency    | 50ms      |
//! | Epoch-aware decode, current epoch               | overhead   | none      |
//! | Epoch-aware decode, old epoch, 2 absent columns | overhead   | 20ns      |
//! | Index build, 10M rows, 8-byte key               | throughput | 2M rows/s |
//! | bulk_build_sorted vs sequential insert, 10M     | speedup    | 5x        |
//! | Index build peak memory                         | RSS growth | budget+256MB |
//! | Publish to writers maintaining                  | latency    | 1ms       |
//! | Wait for old transactions, none active          | latency    | 1ms       |
//! | Shadow rewrite, 1M rows, TEXT to INT            | throughput | 1M rows/s |
//! | Dual-write hook overhead on a writer            | latency    | 2us       |
//! | Foreground OLTP p99 during a build              | degradation| 10%       |
//! | ddl_progress read                               | latency    | 1ms       |
//! | Vacuum epoch retirement check                   | latency    | 10us      |
//!
//! Validation Requirements:
//! - Each measurement runs 5 iterations
//! - Results averaged across all 5 runs
//! - Pass/fail determined by the average
//!
//! Run: cargo test -p zyron-wire --release --test online_ddl_bench -- --nocapture --test-threads=1

// No `#[global_allocator]` here. zyron-wire installs mimalloc for the whole
// binary, and a second declaration in a test that links it is a conflict

mod common;

use std::hint::black_box;
use std::sync::Arc;
use std::sync::Mutex;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::time::Instant;

use bytes::Bytes;
use common::{create_test_server, exec_ddl, exec_dml, new_session, query_values};
use zyron_bench_harness::*;
use zyron_catalog::{ColumnId, IndexState, PhysicalColumn, TableEntry, TableId};
use zyron_common::page::PageId;
use zyron_common::{RowLocator, TypeId};
use zyron_executor::batch::{ColumnBuilder, create_builders};
use zyron_executor::epoch_decode::EpochDecoder;
use zyron_planner::logical::LogicalColumn;
use zyron_storage::BTreeIndex;
use zyron_wire::connection::ServerState;

// A suite whose measurements each want the whole machine runs them one at a
// time, whatever the harness was invoked with
static BENCHMARK_LOCK: Mutex<()> = Mutex::new(());

// =============================================================================
// Targets
// =============================================================================

const ALTER_LATENCY_TARGET_MS: f64 = 50.0;
const DECODE_OLD_EPOCH_TARGET_NS: f64 = 20.0;
const INDEX_BUILD_TARGET_ROWS_PER_SEC: f64 = 2_000_000.0;
const BULK_BUILD_SPEEDUP_TARGET: f64 = 5.0;
const PUBLISH_TARGET_MS: f64 = 1.0;
const WAIT_TARGET_MS: f64 = 1.0;
const SHADOW_REWRITE_TARGET_ROWS_PER_SEC: f64 = 1_000_000.0;
const DUAL_WRITE_TARGET_US: f64 = 2.0;
const OLTP_DEGRADATION_TARGET_PCT: f64 = 10.0;
const DDL_PROGRESS_READ_TARGET_MS: f64 = 1.0;
const EPOCH_RETIREMENT_TARGET_US: f64 = 10.0;

/// Bytes of headroom the index build is allowed above its spill budget.
const INDEX_BUILD_RSS_HEADROOM_BYTES: f64 = 256.0 * 1024.0 * 1024.0;

/// The budget the build's run buffers work within, which the target is stated
/// against. Matches `DEFAULT_RUN_BUDGET_BYTES` in the build.
const SPILL_BUDGET_BYTES: f64 = 64.0 * 1024.0 * 1024.0;

// =============================================================================
// Scales
// =============================================================================

/// Rows behind the three catalog-only column changes, which is the number
/// their target is stated against.
const ALTER_ROWS: i64 = 10_000_000;

/// Rows the index build covers.
///
/// Every metric this suite records names the row count it was taken at, so a
/// throughput read out of the run file is read against the size it was
/// measured on.
const INDEX_BUILD_ROWS: i64 = 10_000_000;

/// Keys loaded into a tree both ways for the speedup comparison.
const BULK_KEYS: u64 = 10_000_000;

/// Rows a shadow rewrite casts and copies. Each run loads its own source,
/// because the rewrite swaps the table it read.
const SHADOW_ROWS: i64 = 1_000_000;

/// Rows decoded per timed pass in the decode guards.
const DECODE_ROWS: usize = 2_000_000;

/// Foreground statements timed for a p99, on each side of the comparison.
const OLTP_OPS: usize = 20_000;

/// Rows in one seed statement. One parse and one plan cover the whole chunk,
/// so a larger chunk loads faster until the statement itself gets unwieldy.
const SEED_CHUNK: i64 = 5_000;

// =============================================================================
// Shared helpers
// =============================================================================

/// Runs a DDL statement and fails the run with its message when it is refused.
async fn ddl(server: &Arc<ServerState>, sql: &str) {
    let mut session = new_session();
    exec_ddl(server, &mut session, sql)
        .await
        .unwrap_or_else(|e| panic!("`{sql}` was refused: {e}"));
}

/// Loads `count` rows of `(k, v)` through the production INSERT path.
async fn seed_pairs(server: &Arc<ServerState>, table: &str, count: i64) {
    let mut from = 0i64;
    while from < count {
        let n = SEED_CHUNK.min(count - from);
        let mut sql = String::with_capacity(n as usize * 24 + 32);
        sql.push_str("INSERT INTO ");
        sql.push_str(table);
        sql.push_str(" VALUES ");
        for i in 0..n {
            if i > 0 {
                sql.push(',');
            }
            let id = from + i;
            sql.push('(');
            sql.push_str(itoa::Buffer::new().format(id));
            sql.push(',');
            sql.push_str(itoa::Buffer::new().format(id * 7 % 1_000_003));
            sql.push(')');
        }
        exec_dml(server, &sql).await;
        from += n;
    }
}

/// Loads `count` rows whose second column holds a number written as text,
/// which is what an incompatible SET TYPE has to cast.
async fn seed_text_numbers(server: &Arc<ServerState>, table: &str, count: i64) {
    let mut from = 0i64;
    while from < count {
        let n = SEED_CHUNK.min(count - from);
        let mut sql = String::with_capacity(n as usize * 28 + 32);
        sql.push_str("INSERT INTO ");
        sql.push_str(table);
        sql.push_str(" VALUES ");
        for i in 0..n {
            if i > 0 {
                sql.push(',');
            }
            let id = from + i;
            sql.push('(');
            sql.push_str(itoa::Buffer::new().format(id));
            sql.push_str(",'");
            sql.push_str(itoa::Buffer::new().format(id % 100_000));
            sql.push_str("')");
        }
        exec_dml(server, &sql).await;
        from += n;
    }
}

/// Resident bytes this process holds right now, zero when the platform will
/// not answer.
fn rss_bytes(system: &mut sysinfo::System) -> u64 {
    let Some(pid) = sysinfo::get_current_pid().ok() else {
        return 0;
    };
    system.refresh_processes(sysinfo::ProcessesToUpdate::Some(&[pid]), true);
    system.process(pid).map(|p| p.memory()).unwrap_or(0)
}

/// The value at the 99th percentile of a latency list, in the list's own unit.
fn p99(mut samples: Vec<f64>) -> f64 {
    if samples.is_empty() {
        return 0.0;
    }
    samples.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let idx = ((samples.len() as f64) * 0.99).ceil() as usize;
    samples[idx.min(samples.len() - 1)]
}

/// The catalog's entry for one index of a table, by name.
fn index_named(
    server: &Arc<ServerState>,
    table_id: TableId,
    name: &str,
) -> Arc<zyron_catalog::IndexEntry> {
    server
        .catalog
        .get_indexes_for_table(table_id)
        .into_iter()
        .find(|i| i.name == name)
        .unwrap_or_else(|| panic!("index `{name}` is in the catalog"))
}

fn locator(n: u64) -> RowLocator {
    RowLocator::Heap {
        page: PageId::new(0, n / 64),
        slot: (n % 64) as u16,
    }
}

/// Builds the key an index writes, the value big-endian so the byte order the
/// tree compares on is the numeric order, followed by the suffix naming the
/// row it points at. The tree reads the row out of that suffix, so measuring
/// against a key without one measures a shape no index holds.
fn key_for(n: u64) -> Bytes {
    let mut key = n.to_be_bytes().to_vec();
    locator(n).append_key_suffix(&mut key);
    Bytes::from(key)
}

// =============================================================================
// 1, 2, 3. The three column changes that write no row
// =============================================================================

/// ADD COLUMN, DROP COLUMN and a compatible SET TYPE each write one catalog
/// row and touch no heap page, so their cost is the same whatever the table
/// holds. The table under them is large enough that a rewrite would be
/// unmissable if one happened.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn bench_catalog_only_column_changes() {
    zyron_bench_harness::init("online_ddl");
    let _lock = BENCHMARK_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    calibrate();
    let before = take_util_snapshot();

    tprintln!("\n=== Catalog-only column changes ===");

    let (server, schema, _tmp) = create_test_server().await;
    ddl(&server, "CREATE TABLE big (k BIGINT, v INT)").await;
    let load = Instant::now();
    seed_pairs(&server, "big", ALTER_ROWS).await;
    tprintln!(
        "  seeded {} rows in {:.1}s",
        format_with_commas(ALTER_ROWS as f64),
        load.elapsed().as_secs_f64()
    );

    let table_id = server
        .catalog
        .get_table(schema, "big")
        .expect("the seeded table is in the catalog")
        .id;
    let heap_before = server
        .catalog
        .get_table_by_id(table_id)
        .expect("table")
        .heap_file_id;

    // ADD COLUMN, one new column per run so no run repeats another's work
    let mut runs = Vec::with_capacity(VALIDATION_RUNS);
    for i in 0..VALIDATION_RUNS {
        let start = Instant::now();
        ddl(
            &server,
            &format!("ALTER TABLE big ADD COLUMN a{i} INT DEFAULT 7"),
        )
        .await;
        runs.push(start.elapsed().as_secs_f64() * 1_000.0);
    }
    let v = validate_metric_with_unit(
        "Catalog-only column changes",
        &format!("ADD COLUMN, {} rows", format_with_commas(ALTER_ROWS as f64)),
        "ms",
        runs,
        ALTER_LATENCY_TARGET_MS,
        false,
    );
    assert!(v.passed, "ADD COLUMN exceeded its target");

    // DROP COLUMN takes the same columns back off
    let mut runs = Vec::with_capacity(VALIDATION_RUNS);
    for i in 0..VALIDATION_RUNS {
        let start = Instant::now();
        ddl(&server, &format!("ALTER TABLE big DROP COLUMN a{i}")).await;
        runs.push(start.elapsed().as_secs_f64() * 1_000.0);
    }
    let v = validate_metric_with_unit(
        "Catalog-only column changes",
        &format!(
            "DROP COLUMN, {} rows",
            format_with_commas(ALTER_ROWS as f64)
        ),
        "ms",
        runs,
        ALTER_LATENCY_TARGET_MS,
        false,
    );
    assert!(v.passed, "DROP COLUMN exceeded its target");

    // A compatible SET TYPE widens in place. Each run widens a column added
    // for it, so every run measures the same shape of change
    for i in 0..VALIDATION_RUNS {
        ddl(&server, &format!("ALTER TABLE big ADD COLUMN w{i} INT")).await;
    }
    let mut runs = Vec::with_capacity(VALIDATION_RUNS);
    for i in 0..VALIDATION_RUNS {
        let start = Instant::now();
        ddl(
            &server,
            &format!("ALTER TABLE big ALTER COLUMN w{i} TYPE BIGINT"),
        )
        .await;
        runs.push(start.elapsed().as_secs_f64() * 1_000.0);
    }
    let v = validate_metric_with_unit(
        "Catalog-only column changes",
        &format!(
            "compatible SET TYPE, {} rows",
            format_with_commas(ALTER_ROWS as f64)
        ),
        "ms",
        runs,
        ALTER_LATENCY_TARGET_MS,
        false,
    );
    assert!(v.passed, "compatible SET TYPE exceeded its target");

    // The heap file is the evidence that none of the fifteen changes rewrote
    // a row. A rewrite installs a new file
    let heap_after = server
        .catalog
        .get_table_by_id(table_id)
        .expect("table")
        .heap_file_id;
    assert_eq!(
        heap_before, heap_after,
        "a catalog-only column change replaced the heap file"
    );

    let after = take_util_snapshot();
    record_test_util("Catalog-only column changes", before, after);
}

// =============================================================================
// 4, 5. What every read now pays
// =============================================================================

/// Builds a table whose current epoch holds `columns` INT columns, plus an
/// older epoch two columns narrower so a row stamped with it reads two absent
/// values.
fn table_with_two_epochs(columns: u16) -> TableEntry {
    let entry_columns: Vec<zyron_catalog::ColumnEntry> = (0..columns)
        .map(|i| zyron_catalog::ColumnEntry {
            id: ColumnId(i + 1),
            table_id: TableId(1),
            name: format!("c{i}"),
            type_id: TypeId::Int32,
            ordinal: i,
            nullable: true,
            default_expr: None,
            max_length: None,
            fractional_digits: None,
            tz_offset_secs: None,
            element_type: None,
            attrs: Default::default(),
            absent_value: Some(0i32.to_le_bytes().to_vec()),
            dropped: false,
        })
        .collect();

    let mut table = TableEntry {
        id: TableId(1),
        schema_id: zyron_catalog::SchemaId(1),
        name: "decode_target".to_string(),
        heap_file_id: 1,
        fsm_file_id: 2,
        columns: entry_columns,
        constraints: Vec::new(),
        created_at: 0,
        versioning_enabled: false,
        scd_type: None,
        system_versioned: false,
        history_table_id: None,
        cdf_enabled: false,
        cdf_retention_days: 0,
        lifecycle: Default::default(),
        columnar: Default::default(),
        dropped_at: None,
        expectations: Vec::new(),
        time_travel_retention_secs: 0,
        lake: Default::default(),
        cluster: Default::default(),
        foreign: Default::default(),
        schema_epoch: 0,
        schema_epochs: Vec::new(),
        pre_stamp_columns: Vec::new(),
        cdf: Default::default(),
    };

    // Epoch 1 is the narrow layout, two columns short of what the table
    // declares now. Epoch 2 is every column
    let narrow: Vec<PhysicalColumn> = table
        .columns
        .iter()
        .take(columns as usize - 2)
        .map(|c| c.physical_column())
        .collect();
    table.schema_epoch = 1;
    table.schema_epochs = vec![zyron_catalog::EpochColumns {
        epoch: 1,
        columns: narrow,
    }];
    table.push_schema_epoch(table.current_physical_columns());
    table
}

/// Encodes one row at a layout of `count` INT columns, none of them null.
fn encode_ints(count: usize) -> Vec<u8> {
    let mut out = vec![0u8; count.div_ceil(8)];
    for i in 0..count {
        out.extend_from_slice(&(i as i32).to_le_bytes());
    }
    out
}

fn logical_of(table: &TableEntry) -> Vec<LogicalColumn> {
    table
        .live_columns()
        .map(|c| LogicalColumn {
            table_idx: Some(0),
            column_id: c.id,
            name: c.name.clone(),
            type_id: c.type_id,
            nullable: c.nullable,
            fractional_digits: c.fractional_digits,
        })
        .collect()
}

/// Decodes `DECODE_ROWS` rows through one plan, rebuilding the builders every
/// batch so the allocation cost lands on both sides of the comparison.
fn timed_decode(
    plan: &zyron_executor::epoch_decode::EpochPlan,
    row: &[u8],
    logical: &[LogicalColumn],
) -> f64 {
    const BATCH: usize = 4_096;
    let start = Instant::now();
    let mut done = 0usize;
    while done < DECODE_ROWS {
        let n = BATCH.min(DECODE_ROWS - done);
        let mut builders: Vec<ColumnBuilder> = create_builders(logical, n);
        for _ in 0..n {
            plan.decode_into(black_box(row), &mut builders);
        }
        black_box(&builders);
        done += n;
    }
    start.elapsed().as_nanos() as f64 / DECODE_ROWS as f64
}

/// The current epoch has to cost what the positional walk cost before an epoch
/// was consulted, and an older epoch's absent columns have to cost only their
/// own pushes.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn bench_epoch_decode() {
    zyron_bench_harness::init("online_ddl");
    let _lock = BENCHMARK_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    calibrate();
    let before = take_util_snapshot();

    tprintln!("\n=== Epoch-aware decode ===");

    let columns = 10u16;
    let table = table_with_two_epochs(columns);
    let output: Vec<ColumnId> = table.live_columns().map(|c| c.id).collect();
    let decoder = EpochDecoder::new(&table, &output);
    let logical = logical_of(&table);

    let current_row = encode_ints(columns as usize);
    let old_row = encode_ints(columns as usize - 2);

    let current_plan = decoder
        .plan(table.schema_epoch)
        .expect("current epoch plan");
    let old_plan = decoder.plan(1).expect("the older epoch plan");

    // The plan the current epoch resolves to is the positional walk on its
    // own, so timing it is timing the decoder without the epoch step
    let mut plan_runs = Vec::with_capacity(VALIDATION_RUNS);
    for _ in 0..VALIDATION_RUNS {
        plan_runs.push(timed_decode(current_plan, &current_row, &logical));
    }
    let plan_only = record_metric(
        "Epoch-aware decode",
        "current epoch, plan resolved once",
        "ns",
        plan_runs,
    );

    // The same rows through the decoder, which resolves the epoch per row.
    // The gap between the two is the whole cost the phase added to a read at
    // the current epoch
    // The same rows through the cursor a scan holds across a batch, which is
    // the path every read takes
    const BATCH: usize = 4_096;
    let mut full_runs = Vec::with_capacity(VALIDATION_RUNS);
    for _ in 0..VALIDATION_RUNS {
        let start = Instant::now();
        let mut done = 0usize;
        while done < DECODE_ROWS {
            let n = BATCH.min(DECODE_ROWS - done);
            let mut builders: Vec<ColumnBuilder> = create_builders(&logical, n);
            let mut cursor = decoder.cursor();
            for _ in 0..n {
                let ok =
                    cursor.try_decode(table.schema_epoch, black_box(&current_row), &mut builders);
                debug_assert!(ok, "the current epoch decodes");
            }
            black_box(&builders);
            done += n;
        }
        full_runs.push(start.elapsed().as_nanos() as f64 / DECODE_ROWS as f64);
    }
    let with_lookup = record_metric(
        "Epoch-aware decode",
        "current epoch, epoch resolved per row",
        "ns",
        full_runs,
    );

    let added = (with_lookup - plan_only).max(0.0);
    tprintln!(
        "  epoch lookup adds {:.4}ns per row at the current epoch",
        added
    );
    // "None measurable" is one sampling interval of the clock this loop reads,
    // which is what a difference has to clear to be a difference at all
    let resolution = 1.0;
    let current_epoch_passed = check_performance_with_unit(
        "Epoch-aware decode",
        "current epoch overhead against the resolved plan",
        "ns",
        added,
        resolution,
        false,
    );

    // An older epoch pushes an absent value for each of the two columns the
    // row predates, and that is all it is allowed to cost
    let mut old_runs = Vec::with_capacity(VALIDATION_RUNS);
    for _ in 0..VALIDATION_RUNS {
        old_runs.push(timed_decode(old_plan, &old_row, &logical));
    }
    let old_epoch = record_metric(
        "Epoch-aware decode",
        "older epoch, 2 absent columns",
        "ns",
        old_runs,
    );
    let absent_overhead = (old_epoch - plan_only).max(0.0);
    let passed = check_performance_with_unit(
        "Epoch-aware decode",
        "older epoch overhead, 2 absent columns",
        "ns",
        absent_overhead,
        DECODE_OLD_EPOCH_TARGET_NS,
        false,
    );
    let after = take_util_snapshot();
    record_test_util("Epoch-aware decode", before, after);

    assert!(
        current_epoch_passed,
        "resolving the epoch per row costs more than the clock can resolve"
    );
    assert!(passed, "an older epoch's absent columns cost too much");
}

// =============================================================================
// 6, 8. Building an index over a table that is already full
// =============================================================================

/// The build reads one snapshot, sorts outside memory and writes the tree in
/// one pass. Its throughput is the number that decides how long a table is
/// without its index, and its resident set is what decides whether the build
/// can run beside the workload at all.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn bench_index_build() {
    zyron_bench_harness::init("online_ddl");
    let _lock = BENCHMARK_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    calibrate_storage();
    let before = take_util_snapshot();

    tprintln!("\n=== Index build ===");

    let (server, schema, _tmp) = create_test_server().await;
    ddl(&server, "CREATE TABLE bulk (k BIGINT, v BIGINT)").await;
    let load = Instant::now();
    seed_pairs(&server, "bulk", INDEX_BUILD_ROWS).await;
    tprintln!(
        "  seeded {} rows in {:.1}s",
        format_with_commas(INDEX_BUILD_ROWS as f64),
        load.elapsed().as_secs_f64()
    );
    let table_id = server.catalog.get_table(schema, "bulk").expect("table").id;

    let mut system = sysinfo::System::new();
    let mut runs = Vec::with_capacity(VALIDATION_RUNS);
    let mut rss_runs = Vec::with_capacity(VALIDATION_RUNS);
    for i in 0..VALIDATION_RUNS {
        let name = format!("ix_build_{i}");
        let baseline = rss_bytes(&mut system);
        let peak = Arc::new(AtomicU64::new(baseline));
        let sampling = Arc::new(AtomicBool::new(true));

        // A sampler on its own thread, because the build never returns to this
        // one until it is finished
        let peak_handle = Arc::clone(&peak);
        let sampling_handle = Arc::clone(&sampling);
        let sampler = std::thread::spawn(move || {
            let mut sys = sysinfo::System::new();
            while sampling_handle.load(Ordering::Relaxed) {
                let now = rss_bytes(&mut sys);
                peak_handle.fetch_max(now, Ordering::Relaxed);
                std::thread::sleep(std::time::Duration::from_millis(50));
            }
        });

        let start = Instant::now();
        ddl(&server, &format!("CREATE INDEX {name} ON bulk (k)")).await;
        let elapsed = start.elapsed().as_secs_f64();

        sampling.store(false, Ordering::Relaxed);
        sampler.join().unwrap_or_else(|_| panic!("the RSS sampler"));

        runs.push(INDEX_BUILD_ROWS as f64 / elapsed.max(1e-9));
        rss_runs.push(peak.load(Ordering::Relaxed).saturating_sub(baseline) as f64);

        assert_eq!(
            index_named(&server, table_id, &name).state,
            IndexState::Ready,
            "the build left the index short of Ready"
        );
        ddl(&server, &format!("DROP INDEX {name}")).await;
    }

    let v = validate_metric_with_unit(
        "Index build",
        &format!(
            "throughput, {} rows, 8-byte key",
            format_with_commas(INDEX_BUILD_ROWS as f64)
        ),
        " rows/s",
        runs,
        INDEX_BUILD_TARGET_ROWS_PER_SEC,
        true,
    );
    assert!(v.passed, "the index build missed its throughput target");

    let v = validate_metric_with_unit(
        "Index build",
        "peak RSS growth",
        " bytes",
        rss_runs,
        SPILL_BUDGET_BYTES + INDEX_BUILD_RSS_HEADROOM_BYTES,
        false,
    );
    assert!(v.passed, "the index build held more memory than its budget");

    let after = take_util_snapshot();
    record_test_util("Index build", before, after);
}

// =============================================================================
// 7. Writing a tree level by level against descending for every key
// =============================================================================

/// The whole reason a build sorts first. Both trees end up holding the same
/// entries, so the only difference the ratio reports is how they were written.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn bench_bulk_build_speedup() {
    zyron_bench_harness::init("online_ddl");
    let _lock = BENCHMARK_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    calibrate_storage();
    let before = take_util_snapshot();

    tprintln!("\n=== bulk_build_sorted against sequential insert ===");

    let tmp = tempfile::TempDir::new().expect("tmp");
    let items: Vec<(Bytes, RowLocator)> =
        (0..BULK_KEYS).map(|n| (key_for(n), locator(n))).collect();

    let mut bulk_runs = Vec::with_capacity(VALIDATION_RUNS);
    let mut sequential_runs = Vec::with_capacity(VALIDATION_RUNS);
    for i in 0..VALIDATION_RUNS {
        let bulk = BTreeIndex::create(100 + i as u32, tmp.path().to_path_buf())
            .await
            .expect("bulk tree");
        let start = Instant::now();
        let built = bulk
            .bulk_build_sorted(items.iter().cloned())
            .expect("bulk build");
        bulk_runs.push(start.elapsed().as_secs_f64());
        assert_eq!(built, BULK_KEYS, "the bulk load lost entries");

        let sequential = BTreeIndex::create(200 + i as u32, tmp.path().to_path_buf())
            .await
            .expect("sequential tree");
        let start = Instant::now();
        for (key, loc) in &items {
            sequential.insert_sync(key, *loc).expect("insert");
        }
        sequential_runs.push(start.elapsed().as_secs_f64());
    }

    let bulk_avg = record_metric(
        "Bulk build",
        &format!(
            "bulk_build_sorted, {} keys",
            format_with_commas(BULK_KEYS as f64)
        ),
        "s",
        bulk_runs,
    );
    let sequential_avg = record_metric(
        "Bulk build",
        &format!(
            "sequential insert, {} keys",
            format_with_commas(BULK_KEYS as f64)
        ),
        "s",
        sequential_runs,
    );

    let speedup = sequential_avg / bulk_avg.max(1e-9);
    let passed = check_performance_with_unit(
        "Bulk build",
        "speedup against sequential insert",
        "x",
        speedup,
        BULK_BUILD_SPEEDUP_TARGET,
        true,
    );
    assert!(passed, "the bulk loader is not far enough ahead");

    let after = take_util_snapshot();
    record_test_util("Bulk build", before, after);
}

// =============================================================================
// 9, 10. The two steps that make a build correct
// =============================================================================

/// Publishing is what puts the index in front of every writer, and waiting is
/// what makes the writers that started earlier finish. Both sit in front of
/// the scan, so both are latency a statement pays before any row is read.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn bench_publish_and_wait() {
    zyron_bench_harness::init("online_ddl");
    let _lock = BENCHMARK_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    calibrate();
    let before = take_util_snapshot();

    tprintln!("\n=== Publish and wait ===");

    let (server, schema, _tmp) = create_test_server().await;
    ddl(&server, "CREATE TABLE pub_t (k BIGINT, v INT)").await;
    seed_pairs(&server, "pub_t", 1_000).await;
    let table_id = server.catalog.get_table(schema, "pub_t").expect("table").id;
    let checkpoint_dir = server.data_dir.join("indexes");
    std::fs::create_dir_all(&checkpoint_dir).expect("index dir");

    // Publication is the catalog row, the empty tree and the registry insert.
    // From the end of it every writer maintains the index
    let mut runs = Vec::with_capacity(VALIDATION_RUNS);
    for i in 0..VALIDATION_RUNS {
        let name = format!("ix_pub_{i}");
        let start = Instant::now();
        let index_id = server
            .catalog
            .create_btree_index(
                table_id,
                schema,
                &name,
                &[("k".to_string(), true)],
                false,
                IndexState::Building,
            )
            .await
            .expect("publish");
        let entry = index_named(&server, table_id, &name);
        let btree = Arc::new(
            BTreeIndex::create(entry.index_file_id, checkpoint_dir.clone())
                .await
                .expect("empty tree"),
        );
        let _ = server
            .btree_indexes
            .insert_async(index_id.0, Arc::clone(&btree))
            .await;
        runs.push(start.elapsed().as_secs_f64() * 1_000.0);

        server
            .catalog
            .drop_index(table_id, &name)
            .await
            .expect("drop the published index");
        let _ = server.btree_indexes.remove_async(&index_id.0).await;
    }
    let v = validate_metric_with_unit(
        "Publish and wait",
        "publish to writers maintaining",
        "ms",
        runs,
        PUBLISH_TARGET_MS,
        false,
    );
    assert!(v.passed, "publication took too long");

    // The wait with nothing left running is one pass over the proc array. The
    // ids handed in ended long ago, so the scan finds none of them
    let stale: Vec<u64> = (1..=64u64).collect();
    let mut runs = Vec::with_capacity(VALIDATION_RUNS);
    for _ in 0..VALIDATION_RUNS {
        const PASSES: usize = 1_000;
        let start = Instant::now();
        for _ in 0..PASSES {
            zyron_wire::index_build::wait_for_transactions_active_at(
                &server,
                black_box(&stale),
                &[],
                None,
            )
            .await
            .expect("the wait returns when nothing it named is running");
        }
        runs.push(start.elapsed().as_secs_f64() * 1_000.0 / PASSES as f64);
    }
    let v = validate_metric_with_unit(
        "Publish and wait",
        "wait for old transactions, none active",
        "ms",
        runs,
        WAIT_TARGET_MS,
        false,
    );
    assert!(v.passed, "the wait costs too much when nothing is running");

    let after = take_util_snapshot();
    record_test_util("Publish and wait", before, after);
}

// =============================================================================
// 11, 12. Casting a column the rows cannot be read as
// =============================================================================

/// A rewrite copies every row through a cast into a shadow table while the
/// source keeps taking writes, then swaps. Its throughput decides how long the
/// dual write is in the way, and the dual write is what a writer pays for it.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn bench_shadow_rewrite() {
    zyron_bench_harness::init("online_ddl");
    let _lock = BENCHMARK_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    calibrate_storage();
    let before = take_util_snapshot();

    tprintln!("\n=== Shadow rewrite ===");

    let mut runs = Vec::with_capacity(VALIDATION_RUNS);
    let mut plain_write = Vec::with_capacity(VALIDATION_RUNS);
    let mut mirrored_write = Vec::with_capacity(VALIDATION_RUNS);

    for run in 0..VALIDATION_RUNS {
        // The rewrite swaps the table out, so each run gets its own server and
        // its own freshly loaded source
        let (server, schema, _tmp) = create_test_server().await;
        ddl(&server, "CREATE TABLE priced (k BIGINT, price TEXT)").await;
        seed_text_numbers(&server, "priced", SHADOW_ROWS).await;

        // What a single-row insert costs with nothing mirroring it
        let start = Instant::now();
        for i in 0..1_000i64 {
            exec_dml(
                &server,
                &format!("INSERT INTO priced VALUES ({}, '1')", SHADOW_ROWS + i),
            )
            .await;
        }
        plain_write.push(start.elapsed().as_secs_f64() * 1_000_000.0 / 1_000.0);

        // A writer runs across the whole rewrite, so the copy competes with
        // the dual write exactly as it does in production
        let writer_server = Arc::clone(&server);
        let writer_us = Arc::new(AtomicU64::new(0));
        let writer_rows = Arc::new(AtomicU64::new(0));
        let stop = Arc::new(AtomicBool::new(false));
        let us_handle = Arc::clone(&writer_us);
        let rows_handle = Arc::clone(&writer_rows);
        let stop_handle = Arc::clone(&stop);
        let writer = tokio::spawn(async move {
            let mut next = SHADOW_ROWS + 1_000_000;
            while !stop_handle.load(Ordering::Relaxed) {
                let start = Instant::now();
                exec_dml(
                    &writer_server,
                    &format!("INSERT INTO priced VALUES ({next}, '2')"),
                )
                .await;
                us_handle.fetch_add(start.elapsed().as_micros() as u64, Ordering::Relaxed);
                rows_handle.fetch_add(1, Ordering::Relaxed);
                next += 1;
            }
        });

        let source = server
            .catalog
            .get_table(schema, "priced")
            .expect("the source table");
        let start = Instant::now();
        let outcome = zyron_wire::shadow_rewrite::run(
            &server,
            schema,
            &source,
            "price",
            TypeId::Int32,
            None,
            None,
            "bench",
            &[],
        )
        .await
        .expect("the rewrite runs");
        let elapsed = start.elapsed().as_secs_f64();
        stop.store(true, Ordering::Relaxed);
        writer.await.expect("writer");

        if let zyron_wire::shadow_rewrite::RewriteOutcome::CastFailed {
            locator,
            value,
            reason,
        } = &outcome
        {
            panic!("the rewrite refused {value:?} at {locator:?}: {reason}");
        }
        runs.push(SHADOW_ROWS as f64 / elapsed.max(1e-9));

        let rows = writer_rows.load(Ordering::Relaxed).max(1);
        mirrored_write.push(writer_us.load(Ordering::Relaxed) as f64 / rows as f64);
        tprintln!(
            "  run {}: rewrote {} rows in {:.2}s while a writer added {}",
            run,
            format_with_commas(SHADOW_ROWS as f64),
            elapsed,
            format_with_commas(rows as f64)
        );
    }

    let v = validate_metric_with_unit(
        "Shadow rewrite",
        &format!(
            "throughput, {} rows, TEXT to INT",
            format_with_commas(SHADOW_ROWS as f64)
        ),
        " rows/s",
        runs,
        SHADOW_REWRITE_TARGET_ROWS_PER_SEC,
        true,
    );
    assert!(v.passed, "the shadow rewrite missed its throughput target");

    let plain = record_metric("Shadow rewrite", "insert with no shadow", "us", plain_write);
    let mirrored = record_metric(
        "Shadow rewrite",
        "insert while the shadow is filling",
        "us",
        mirrored_write,
    );
    let added = (mirrored - plain).max(0.0);
    let passed = check_performance_with_unit(
        "Shadow rewrite",
        "dual-write hook overhead per row",
        "us",
        added,
        DUAL_WRITE_TARGET_US,
        false,
    );
    assert!(passed, "the dual write costs a writer too much");

    let after = take_util_snapshot();
    record_test_util("Shadow rewrite", before, after);
}

// =============================================================================
// 13. What the workload beside a build feels
// =============================================================================

/// The number that decides whether a build can run at all during the day. The
/// same foreground statements are timed twice, once with the table to
/// themselves and once with a build reading it.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn bench_oltp_during_build() {
    zyron_bench_harness::init("online_ddl");
    let _lock = BENCHMARK_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    calibrate_storage();
    let before = take_util_snapshot();

    tprintln!("\n=== Foreground OLTP during a build ===");

    let (server, _schema_id, _tmp) = create_test_server().await;
    ddl(&server, "CREATE TABLE oltp (k BIGINT, v BIGINT)").await;
    seed_pairs(&server, "oltp", INDEX_BUILD_ROWS).await;
    // The foreground reads are point lookups, so they go through an index the
    // table already has. The build the comparison runs against covers the
    // other column, which is what makes it a second index rather than the one
    // being read
    ddl(&server, "CREATE INDEX ix_oltp_k ON oltp (k)").await;

    /// One pass of foreground point reads, returning every latency it saw.
    async fn foreground(server: &Arc<ServerState>, ops: usize) -> Vec<f64> {
        let mut samples = Vec::with_capacity(ops);
        for i in 0..ops {
            let k = (i as i64 * 7919) % INDEX_BUILD_ROWS;
            let start = Instant::now();
            let rows = query_values(server, &format!("SELECT v FROM oltp WHERE k = {k}")).await;
            samples.push(start.elapsed().as_secs_f64() * 1_000.0);
            debug_assert!(!rows.is_empty(), "the point read found no row for {k}");
        }
        samples
    }

    let mut baseline_runs = Vec::with_capacity(VALIDATION_RUNS);
    let mut during_runs = Vec::with_capacity(VALIDATION_RUNS);
    for i in 0..VALIDATION_RUNS {
        // Undisturbed
        baseline_runs.push(p99(foreground(&server, OLTP_OPS).await));

        // Beside a build. The foreground pass has to finish inside the build
        // for the comparison to mean anything, which the row count makes sure
        // of
        let name = format!("ix_oltp_{i}");
        let build_server = Arc::clone(&server);
        let build_name = name.clone();
        let build = tokio::spawn(async move {
            ddl(
                &build_server,
                &format!("CREATE INDEX {build_name} ON oltp (v)"),
            )
            .await;
        });
        during_runs.push(p99(foreground(&server, OLTP_OPS).await));
        build.await.expect("build");
        ddl(&server, &format!("DROP INDEX {name}")).await;
    }

    let baseline = record_metric(
        "OLTP during a build",
        "point read p99, no build",
        "ms",
        baseline_runs,
    );
    let during = record_metric(
        "OLTP during a build",
        "point read p99, build running",
        "ms",
        during_runs,
    );
    let degradation = ((during - baseline) / baseline.max(1e-9)) * 100.0;
    let passed = check_performance_with_unit(
        "OLTP during a build",
        "p99 degradation",
        "%",
        degradation.max(0.0),
        OLTP_DEGRADATION_TARGET_PCT,
        false,
    );
    assert!(passed, "a build costs the foreground too much");

    let after = take_util_snapshot();
    record_test_util("OLTP during a build", before, after);
}

// =============================================================================
// 14, 15. Watching a change, and clearing up after one
// =============================================================================

/// The progress view is read by anyone watching a change, and the retirement
/// check runs once per table per vacuum cycle whether or not anything retires.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn bench_progress_and_retirement() {
    zyron_bench_harness::init("online_ddl");
    let _lock = BENCHMARK_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    calibrate();
    let before = take_util_snapshot();

    tprintln!("\n=== Progress view and epoch retirement ===");

    let (server, schema, _tmp) = create_test_server().await;
    ddl(&server, "CREATE TABLE watched (k BIGINT, v INT)").await;
    seed_pairs(&server, "watched", 10_000).await;

    // Rows for the view to carry while it is read, which is the shape a read
    // finds during a real change
    let _handles: Vec<_> = (0..8)
        .map(|i| {
            server.ddl_progress.begin(
                "watched",
                &format!("ix_{i}"),
                zyron_wire::ddl_progress::DdlOperation::CreateIndex,
                "bench",
            )
        })
        .collect();

    let mut runs = Vec::with_capacity(VALIDATION_RUNS);
    for _ in 0..VALIDATION_RUNS {
        const READS: usize = 2_000;
        let start = Instant::now();
        for _ in 0..READS {
            let (_fields, rows) =
                zyron_wire::system_core_views::build("storage", "ddl_progress", &server)
                    .await
                    .expect("the progress view builds");
            black_box(rows.len());
        }
        runs.push(start.elapsed().as_secs_f64() * 1_000.0 / READS as f64);
    }
    let v = validate_metric_with_unit(
        "Progress and retirement",
        "ddl_progress read",
        "ms",
        runs,
        DDL_PROGRESS_READ_TARGET_MS,
        false,
    );
    assert!(v.passed, "reading the progress view costs too much");

    // The retirement check a vacuum cycle runs per table. Several epochs are
    // live and the floor sits below them, so the check finds nothing to retire
    // and returns without writing, which is the common cycle
    ddl(&server, "ALTER TABLE watched ADD COLUMN a INT DEFAULT 1").await;
    ddl(&server, "ALTER TABLE watched ADD COLUMN b INT DEFAULT 2").await;
    let table_id = server
        .catalog
        .get_table(schema, "watched")
        .expect("table")
        .id;

    let mut runs = Vec::with_capacity(VALIDATION_RUNS);
    for _ in 0..VALIDATION_RUNS {
        const CHECKS: usize = 20_000;
        let start = Instant::now();
        for _ in 0..CHECKS {
            let retired = server
                .catalog
                .retire_schema_epochs(table_id, black_box(1), true, false, None)
                .await
                .expect("the check runs");
            debug_assert!(!retired, "nothing should retire with the floor at 1");
        }
        runs.push(start.elapsed().as_secs_f64() * 1_000_000.0 / CHECKS as f64);
    }
    let v = validate_metric_with_unit(
        "Progress and retirement",
        "vacuum epoch retirement check, per table",
        "us",
        runs,
        EPOCH_RETIREMENT_TARGET_US,
        false,
    );
    assert!(
        v.passed,
        "the retirement check costs a vacuum cycle too much"
    );

    let after = take_util_snapshot();
    record_test_util("Progress and retirement", before, after);
}
