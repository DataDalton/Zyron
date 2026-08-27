//! A real query that exceeds its memory budget spills instead of failing.
//!
//! The operator-level tests prove the external sort is correct. This one
//! proves it is reachable: a query planned and executed the way the server
//! plans and executes one, against a memory budget it cannot fit in, that
//! used to come back as "query exceeds its memory budget" and now comes back
//! as rows.
//!
//! The revert proof is in the same file. The identical query with no spill
//! directory still fails, which is what makes the passing case evidence about
//! spilling rather than evidence that the budget was generous.
//!
//! Run: cargo test -p zyron-wire --test spill_end_to_end_test

mod common;

use std::sync::Arc;

use common::create_test_server;
use zyron_executor::spill::{SpillDirectory, SpillStats};

/// Rows the sorted table holds. Enough that the sort's working set is far
/// past the budget below.
const ROWS: i64 = 20_000;

/// What one query may hold at once. Small enough that a twenty thousand row
/// sort cannot fit, large enough that the plan itself is unaffected.
const BUDGET_BYTES: u64 = 64 * 1024;

fn spill_directory(name: &str) -> Arc<SpillDirectory> {
    let root = std::env::temp_dir().join(format!("zyron_spill_e2e_{}_{name}", std::process::id()));
    let _ = std::fs::remove_dir_all(&root);
    std::fs::create_dir_all(&root).expect("scratch");
    Arc::new(SpillDirectory::open(&root, 512 * 1024 * 1024).expect("open"))
}

/// Builds a table whose sort will not fit in the budget.
async fn seeded_server(
    name: &str,
    spill: Option<Arc<SpillDirectory>>,
) -> (Arc<zyron_wire::connection::ServerState>, tempfile::TempDir) {
    let (state, _schema, tmp) = create_test_server().await;
    common::exec_ddl(
        &state,
        &mut common::new_session(),
        "CREATE TABLE wide (id BIGINT, payload TEXT)",
    )
    .await
    .expect("ddl");

    // Rows wide enough that the sort's buffered input dwarfs the budget
    let mut values = Vec::new();
    for i in 0..ROWS {
        let key = (i * 2_654_435_761i64).rem_euclid(1_000_003);
        values.push(format!("({key}, 'payload-{i}-{}')", "x".repeat(48)));
        if values.len() == 500 {
            common::exec_dml(
                &state,
                &format!("INSERT INTO public.wide VALUES {}", values.join(", ")),
            )
            .await;
            values.clear();
        }
    }
    if !values.is_empty() {
        common::exec_dml(
            &state,
            &format!("INSERT INTO public.wide VALUES {}", values.join(", ")),
        )
        .await;
    }

    // The harness builds a ServerState literal, so the two knobs this test is
    // about are set here rather than through a config file
    let state = Arc::new(zyron_wire::connection::ServerState {
        max_query_memory: Some(BUDGET_BYTES),
        spill_directory: spill,
        ..Arc::try_unwrap(state).unwrap_or_else(|_| panic!("state is uniquely held"))
    });
    let _ = name;
    (state, tmp)
}

/// The query that used to fail.
///
/// A limit, so the result is small and the sort is not. The memory budget
/// covers the whole query, and a result set genuinely is held in memory: a
/// query returning twenty thousand rows exceeds a sixty-four kilobyte budget
/// on the result alone, spill or no spill. What spilling fixes is the
/// operator's own buffering, so that is what this measures.
const SORTING_QUERY: &str = "SELECT id FROM public.wide ORDER BY id LIMIT 50";

/// Rows the query asks for.
const WANTED: usize = 50;

/// The keys the seeded table holds, which is what the expected answer is
/// computed from rather than from a second run of the engine.
fn seeded_keys() -> Vec<i64> {
    (0..ROWS)
        .map(|i| (i * 2_654_435_761i64).rem_euclid(1_000_003))
        .collect()
}

/// With somewhere to spill, the query returns every row in order.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_sort_past_its_budget_spills_and_returns_rows() {
    let dir = spill_directory("spills");
    let (state, _tmp) = seeded_server("spills", Some(Arc::clone(&dir))).await;

    let before = SpillStats::global()
        .sorts_spilled
        .load(std::sync::atomic::Ordering::Relaxed);

    let values = common::query_values(&state, SORTING_QUERY).await;
    assert_eq!(
        values.len(),
        WANTED,
        "the spilling sort returned {} of {WANTED} rows",
        values.len()
    );

    // Against the answer computed here, not against a second run of the
    // engine: a merge that is consistently wrong would agree with itself
    let mut expected = seeded_keys();
    expected.sort_unstable();
    expected.truncate(WANTED);

    let got: Vec<i64> = values
        .iter()
        .enumerate()
        .map(|(row, columns)| match &columns[0] {
            zyron_executor::column::ScalarValue::Int64(v) => *v,
            other => panic!("row {row} came back as {other:?}"),
        })
        .collect();
    assert_eq!(got, expected, "the spilling sort returned the wrong rows");

    assert!(
        SpillStats::global()
            .sorts_spilled
            .load(std::sync::atomic::Ordering::Relaxed)
            > before,
        "the query fit after all, so this proves nothing about spilling"
    );
    assert_eq!(
        dir.reserved_bytes(),
        0,
        "the finished query left spill files behind"
    );
}

/// The revert proof. The same query with nowhere to spill still fails on its
/// budget, which is what the passing case above is measured against.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn the_same_query_without_a_spill_directory_still_fails() {
    let (state, _tmp) = seeded_server("no_spill", None).await;

    let error = common::query_error(&state, SORTING_QUERY).await;
    assert!(
        error.contains("memory budget"),
        "expected the budget failure this test is the control for, got: {error}"
    );
}

/// Spilling is visible where an operator would look for it.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn the_view_reports_what_spilling_cost() {
    let dir = spill_directory("view");
    let refusals_before = SpillStats::global()
        .quota_refusals
        .load(std::sync::atomic::Ordering::Relaxed);
    let (state, _tmp) = seeded_server("view", Some(std::sync::Arc::clone(&dir))).await;
    assert_eq!(
        common::query_values(&state, SORTING_QUERY).await.len(),
        WANTED
    );

    let (fields, rows) =
        zyron_wire::pressure_views::query_pressure_view("zyron_sys.pressure.spill_stats", None)
            .expect("the view answers");
    assert_eq!(rows.len(), 1, "spill stats is one row about the node");

    let column = |name: &str| -> u64 {
        let idx = fields
            .iter()
            .position(|f| f.name == name)
            .unwrap_or_else(|| panic!("no {name} column"));
        String::from_utf8(rows[0][idx].clone().expect("a value"))
            .expect("utf8")
            .parse()
            .expect("a number")
    };

    assert!(
        column("files_created") > 0,
        "no spill file was ever created"
    );
    assert!(column("bytes_written") > 0);
    assert!(column("bytes_read") > 0, "nothing was read back");
    assert!(column("sorts_spilled") > 0);
    assert!(column("runs_written") > 0);
    assert!(column("peak_live_bytes") > 0);
    assert!(
        column("files_live") <= column("files_created"),
        "the view reported more live files than it ever created"
    );
    // Every counter above is monotonic, so another query spilling at the same
    // moment can only make them larger. Liveness is not: files_live counts the
    // node, and this query is not the only thing on it. Whether this query
    // cleaned up is a question about its own directory
    assert_eq!(
        dir.reserved_bytes(),
        0,
        "a spill file outlived the query that wrote it"
    );
    assert_eq!(
        column("quota_refusals"),
        refusals_before,
        "the quota refused a query"
    );
}
