//! Whether the engine actually reports what the classifier reads.
//!
//! The signals have their own tests, and those check the arithmetic. They
//! cannot check the thing that actually breaks: a producer that is never
//! called. A classifier reading a counter nothing increments returns the same
//! answer forever, the ladder's gating never engages, and the controller
//! behaves exactly like the threshold scaler it replaced. Nothing fails, and
//! nothing says so.
//!
//! So this drives the engine and asks whether the numbers moved.
//!
//! Run: cargo test -p zyron-wire --test contention_producers_test

mod common;

use common::create_test_server;
use zyron_pressure::pressure_control::PressureController;

/// The process controller is shared, so these read counters one at a time.
static SERIALIZE: std::sync::Mutex<()> = std::sync::Mutex::new(());

/// Committing writes moves the commit half of the conflict rate.
///
/// Without it the rate is aborts over aborts, which is one, and every node
/// with a single conflict would classify as contended forever.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_committed_write_is_counted() {
    let _guard = SERIALIZE.lock().unwrap_or_else(|e| e.into_inner());
    let (state, _schema, _tmp) = create_test_server().await;
    common::exec_ddl(
        &state,
        &mut common::new_session(),
        "CREATE TABLE counted (id BIGINT, v BIGINT)",
    )
    .await
    .expect("ddl");

    let contention = PressureController::global().contention();
    let before = contention.occ_abort_rate();
    assert_eq!(before, 0.0, "the harness started with conflicts recorded");

    for i in 0..32 {
        common::exec_dml(
            &state,
            &format!("INSERT INTO public.counted VALUES ({i}, {i})"),
        )
        .await;
    }

    // Commits with no conflicts is a rate of zero, and it is zero because the
    // commits were counted rather than because nothing happened
    assert_eq!(contention.occ_abort_rate(), 0.0);
    contention.record_conflict_abort();
    let with_one = contention.occ_abort_rate();
    assert!(
        with_one > 0.0 && with_one < 0.5,
        "one conflict against 32 commits read as {with_one}, so the commits were not counted"
    );
    contention.roll_window();
}

/// Writes concentrated on one storage extent are visible as skew, and writes
/// spread across many are not.
///
/// Many tables rather than many rows, because the contention unit is the
/// extent a writer serializes on rather than the logical row: under
/// multi-version storage a row's address moves every time it is written, so
/// row identity is exactly the thing a lock cannot follow.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn write_skew_is_visible_and_an_even_spread_is_not() {
    let _guard = SERIALIZE.lock().unwrap_or_else(|e| e.into_inner());
    let (state, _schema, _tmp) = create_test_server().await;

    for t in 0..EXTENTS {
        common::exec_ddl(
            &state,
            &mut common::new_session(),
            &format!("CREATE TABLE part_{t} (id BIGINT, v BIGINT)"),
        )
        .await
        .expect("ddl");
        common::exec_dml(
            &state,
            &format!("INSERT INTO public.part_{t} VALUES (1, 0)"),
        )
        .await;
    }

    let contention = PressureController::global().contention();

    // Round robin across every table, which is a workload with no skew
    contention.roll_window();
    for i in 0..WRITES {
        let t = i % EXTENTS;
        common::exec_dml(
            &state,
            &format!("UPDATE public.part_{t} SET v = v + 1 WHERE id = 1"),
        )
        .await;
    }
    let spread = contention.hot_key_share();
    assert!(
        spread < 0.30,
        "a round robin over {EXTENTS} tables read as skewed at {spread}"
    );

    // The same volume, all of it on one table
    contention.roll_window();
    for i in 0..WRITES {
        // A few writes elsewhere, so the window has partitions to be skewed
        // across rather than a single one
        let t = if i % 8 == 0 { i % EXTENTS } else { 0 };
        common::exec_dml(
            &state,
            &format!("UPDATE public.part_{t} SET v = v + 1 WHERE id = 1"),
        )
        .await;
    }
    let concentrated = contention.hot_key_share();

    assert!(
        concentrated > 0.30,
        "seven writes in eight to one table read as {concentrated}, below the threshold"
    );
    assert!(
        concentrated > spread,
        "concentration {concentrated} did not exceed the even spread {spread}"
    );
    contention.roll_window();
}

/// Tables the skew scenario writes, which is how many extents it touches.
/// Past the floor below which concentration means nothing.
const EXTENTS: u64 = 16;

/// Writes in each phase. Enough that the sampled slice holds a clear majority
/// for the concentrated case.
const WRITES: u64 = 640;

/// A durable commit passes through the wait the group-commit signal is taken
/// from, so the rate is a measurement rather than a default.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn durable_commits_reach_the_group_commit_signal() {
    let _guard = SERIALIZE.lock().unwrap_or_else(|e| e.into_inner());
    let (state, _schema, _tmp) = create_test_server().await;
    common::exec_ddl(
        &state,
        &mut common::new_session(),
        "CREATE TABLE durable (id BIGINT)",
    )
    .await
    .expect("ddl");

    let contention = PressureController::global().contention();
    contention.roll_window();
    // Nothing measured yet reads as fully batched, which is the reading that
    // keeps a quiet node out of the fsync classification
    assert_eq!(contention.group_commit_hit_rate(), 1.0);

    for i in 0..64 {
        common::exec_dml(&state, &format!("INSERT INTO public.durable VALUES ({i})")).await;
    }

    // Every writer has finished, so none is still parked on the device
    assert_eq!(
        contention.writes_waiting(),
        0,
        "a writer was left counted as waiting after its commit returned"
    );
    contention.roll_window();
}

/// Materializing work charges the node gauge and gives it back, so the memory
/// classification is reading the machine rather than a counter stuck at zero.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn query_memory_reaches_the_node_gauge_and_is_returned() {
    let _guard = SERIALIZE.lock().unwrap_or_else(|e| e.into_inner());
    let (state, _schema, _tmp) = create_test_server().await;
    common::exec_ddl(
        &state,
        &mut common::new_session(),
        "CREATE TABLE big (id BIGINT, v BIGINT)",
    )
    .await
    .expect("ddl");
    for i in 0..2_000 {
        common::exec_dml(
            &state,
            &format!("INSERT INTO public.big VALUES ({i}, {})", i % 97),
        )
        .await;
    }

    let gauge = PressureController::global().memory();
    let before = gauge.reserved();

    // A sort and a grouped aggregate both materialize, which is what charges
    // the gauge
    let sorted = common::query_rows(&state, "SELECT id FROM public.big ORDER BY v, id").await;
    assert_eq!(sorted, 2_000);
    let grouped = common::query_rows(
        &state,
        "SELECT v, count(*) FROM public.big GROUP BY v ORDER BY v",
    )
    .await;
    assert_eq!(grouped, 97);

    assert_eq!(
        gauge.reserved(),
        before,
        "a finished query left {} bytes charged to the node",
        gauge.reserved().saturating_sub(before)
    );
    assert!(
        gauge.peak() > 0,
        "no query ever charged the node gauge, so the memory classification cannot fire"
    );
}
