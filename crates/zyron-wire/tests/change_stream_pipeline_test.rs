//! Pipeline stages over change streams.
//!
//! A CONSUME CHANGES stage reads a stream and runs its load in one
//! transaction that moves the stream's position, so a stage that fails
//! leaves the position where it was and the retry reads the same changes.
//! MAX ROWS bounds one run so a backlog drains over several. An APPLY CHANGES
//! stage maintains its target the way the statement does. A stream created
//! with SHOW INITIAL ROWS seeds the target on the first run and continues
//! incrementally on the next with no change to the definition.
//!
//! Run: cargo test -p zyron-wire --test change_stream_pipeline_test -- --nocapture

use std::sync::Arc;

use zyron_wire::connection::ServerState;
use zyron_wire::session::Session;

mod common;
use common::*;

async fn seeded(server: &Arc<ServerState>, session: &mut Option<Session>, storage: Storage) {
    exec_ddl(
        server,
        session,
        &storage.create("CREATE TABLE orders (id BIGINT PRIMARY KEY, region TEXT, total BIGINT)"),
    )
    .await
    .expect("create orders");
    exec_ddl(
        server,
        session,
        "ALTER TABLE orders SET (change_data_feed = true)",
    )
    .await
    .expect("turn the feed on");
}

fn position_of(server: &Arc<ServerState>, stream: &str) -> (u64, u64) {
    let table_id = table_id_of(server, "orders");
    let entry = server
        .catalog
        .list_change_streams()
        .into_iter()
        .find(|e| e.name == stream)
        .expect("the stream exists");
    (entry.position_of(table_id), entry.consumed_of(table_id))
}

fn rows_processed(server: &Arc<ServerState>, pipeline: &str) -> u64 {
    server
        .catalog
        .get_pipeline_by_name(pipeline)
        .expect("the pipeline exists")
        .rows_processed
}

async fn count(server: &Arc<ServerState>, sql: &str) -> i64 {
    let rows = query_values(server, sql).await;
    match rows.first().and_then(|r| r.first()) {
        Some(zyron_executor::column::ScalarValue::Int64(n)) => *n,
        other => panic!("expected a count from {sql}, got {other:?}"),
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_consume_stage_loads_its_target_and_moves_the_position_in_its_commit() {
    for storage in Storage::BOTH {
        a_consume_stage_loads_its_target_and_moves_the_position_in_its_commit_on(storage).await;
    }
}

async fn a_consume_stage_loads_its_target_and_moves_the_position_in_its_commit_on(
    storage: Storage,
) {
    println!("{storage}");
    let (server, _schema, _tmp) = create_test_server_with_cdc().await;
    let mut session = new_session();
    seeded(&server, &mut session, storage).await;
    exec_ddl(
        &server,
        &mut session,
        "CREATE CHANGE STREAM order_changes ON TABLE orders",
    )
    .await
    .expect("create the stream");
    for id in 1..=5 {
        exec_dml(
            &server,
            &format!("INSERT INTO orders VALUES ({id}, 'eu', {})", id * 10),
        )
        .await;
    }
    exec_ddl(
        &server,
        &mut session,
        "CREATE PIPELINE bronze AS (\
            STAGE land (CONSUME CHANGES FROM order_changes INTO bronze_orders)\
        )",
    )
    .await
    .expect("create the pipeline");

    // The first run creates the target from the change set, metadata
    // columns included, and loads every pending change
    exec_ddl(&server, &mut session, "RUN PIPELINE bronze")
        .await
        .expect("the first run");
    assert_eq!(rows_processed(&server, "bronze"), 5);
    assert_eq!(
        count(&server, "SELECT COUNT(*) FROM bronze_orders").await,
        5
    );
    assert_eq!(
        count(
            &server,
            "SELECT COUNT(*) FROM bronze_orders WHERE _change_type = 'insert'"
        )
        .await,
        5
    );
    let (_, consumed) = position_of(&server, "order_changes");
    assert_eq!(consumed, 5, "the position moved with the load");

    // Nothing pending, nothing loaded, the position stays
    exec_ddl(&server, &mut session, "RUN PIPELINE bronze")
        .await
        .expect("an empty run");
    assert_eq!(rows_processed(&server, "bronze"), 0);
    assert_eq!(
        count(&server, "SELECT COUNT(*) FROM bronze_orders").await,
        5
    );

    // A later change is what the next run loads
    exec_dml(&server, "UPDATE orders SET total = 11 WHERE id = 1").await;
    exec_dml(&server, "DELETE FROM orders WHERE id = 2").await;
    exec_ddl(&server, &mut session, "RUN PIPELINE bronze")
        .await
        .expect("the incremental run");
    assert_eq!(
        rows_processed(&server, "bronze"),
        3,
        "a preimage, a postimage and a delete"
    );
    assert_eq!(
        count(&server, "SELECT COUNT(*) FROM bronze_orders").await,
        8
    );
    let (_, consumed) = position_of(&server, "order_changes");
    assert_eq!(consumed, 8);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_stage_that_fails_leaves_the_position_and_the_retry_reads_the_same_changes() {
    for storage in Storage::BOTH {
        a_stage_that_fails_leaves_the_position_and_the_retry_reads_the_same_changes_on(storage)
            .await;
    }
}

async fn a_stage_that_fails_leaves_the_position_and_the_retry_reads_the_same_changes_on(
    storage: Storage,
) {
    println!("{storage}");
    let (server, _schema, _tmp) = create_test_server_with_cdc().await;
    let mut session = new_session();
    seeded(&server, &mut session, storage).await;
    exec_ddl(
        &server,
        &mut session,
        "CREATE CHANGE STREAM order_changes ON TABLE orders",
    )
    .await
    .expect("create the stream");
    exec_ddl(
        &server,
        &mut session,
        "CREATE TABLE silver_orders (id BIGINT PRIMARY KEY, total BIGINT)",
    )
    .await
    .expect("create the target");
    for id in 1..=3 {
        exec_dml(
            &server,
            &format!("INSERT INTO orders VALUES ({id}, 'eu', {id})"),
        )
        .await;
    }
    // A row already in the target makes the first load collide on the key
    exec_dml(&server, "INSERT INTO silver_orders VALUES (2, 0)").await;
    exec_ddl(
        &server,
        &mut session,
        "CREATE PIPELINE silver AS (\
            STAGE load (CONSUME CHANGES FROM order_changes AS (\
                INSERT INTO silver_orders SELECT id, total FROM changes\
            ))\
        )",
    )
    .await
    .expect("create the pipeline");

    let failed = exec_ddl(&server, &mut session, "RUN PIPELINE silver")
        .await
        .expect_err("the load collides on the key");
    assert!(
        failed.contains("silver_orders") || failed.to_lowercase().contains("unique"),
        "{failed}"
    );
    let (_, consumed) = position_of(&server, "order_changes");
    assert_eq!(consumed, 0, "a failed stage moves nothing");
    assert_eq!(
        count(&server, "SELECT COUNT(*) FROM silver_orders").await,
        1
    );
    let status = server
        .catalog
        .get_pipeline_by_name("silver")
        .expect("the pipeline")
        .status_msg
        .clone()
        .unwrap_or_default();
    assert!(
        !status.is_empty(),
        "the failure is recorded on the pipeline"
    );

    // Once the collision is gone, the retry reads the same three changes
    exec_dml(&server, "DELETE FROM silver_orders WHERE id = 2").await;
    exec_ddl(&server, &mut session, "RUN PIPELINE silver")
        .await
        .expect("the retry");
    assert_eq!(rows_processed(&server, "silver"), 3);
    assert_eq!(
        count(&server, "SELECT COUNT(*) FROM silver_orders").await,
        3
    );
    let (_, consumed) = position_of(&server, "order_changes");
    assert_eq!(consumed, 3);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn max_rows_drains_a_backlog_over_several_bounded_runs() {
    for storage in Storage::BOTH {
        max_rows_drains_a_backlog_over_several_bounded_runs_on(storage).await;
    }
}

async fn max_rows_drains_a_backlog_over_several_bounded_runs_on(storage: Storage) {
    println!("{storage}");
    let (server, _schema, _tmp) = create_test_server_with_cdc().await;
    let mut session = new_session();
    seeded(&server, &mut session, storage).await;
    exec_ddl(
        &server,
        &mut session,
        "CREATE CHANGE STREAM order_changes ON TABLE orders",
    )
    .await
    .expect("create the stream");
    // Fifty transactions of twenty rows each, a backlog of a thousand
    for batch in 0..50u64 {
        let values: Vec<String> = (0..20u64)
            .map(|i| {
                let id = batch * 20 + i + 1;
                format!("({id}, 'eu', {id})")
            })
            .collect();
        exec_dml(
            &server,
            &format!("INSERT INTO orders VALUES {}", values.join(", ")),
        )
        .await;
    }
    exec_ddl(
        &server,
        &mut session,
        "CREATE PIPELINE drain AS (\
            STAGE land (CONSUME CHANGES FROM order_changes MAX ROWS 300 INTO bronze_orders)\
        )",
    )
    .await
    .expect("create the pipeline");

    // Each run takes at most the bound, rounded up to whole transactions,
    // and the runs together drain the backlog
    let mut runs = Vec::new();
    loop {
        exec_ddl(&server, &mut session, "RUN PIPELINE drain")
            .await
            .expect("a bounded run");
        let took = rows_processed(&server, "drain");
        if took == 0 {
            break;
        }
        assert!(took <= 300, "a run stays inside the bound: {took}");
        runs.push(took);
        assert!(
            runs.len() <= 10,
            "the backlog drains in a handful of runs: {runs:?}"
        );
    }
    assert_eq!(runs.iter().sum::<u64>(), 1000, "{runs:?}");
    assert!(runs.len() >= 4, "the bound split the backlog: {runs:?}");
    assert_eq!(
        count(&server, "SELECT COUNT(*) FROM bronze_orders").await,
        1000
    );
    let (_, consumed) = position_of(&server, "order_changes");
    assert_eq!(consumed, 1000);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn an_apply_stage_maintains_its_target() {
    for storage in Storage::BOTH {
        an_apply_stage_maintains_its_target_on(storage).await;
    }
}

async fn an_apply_stage_maintains_its_target_on(storage: Storage) {
    println!("{storage}");
    let (server, _schema, _tmp) = create_test_server_with_cdc().await;
    let mut session = new_session();
    seeded(&server, &mut session, storage).await;
    exec_ddl(
        &server,
        &mut session,
        "CREATE CHANGE STREAM order_changes ON TABLE orders",
    )
    .await
    .expect("create the stream");
    exec_ddl(
        &server,
        &mut session,
        "CREATE TABLE dim_orders (id BIGINT PRIMARY KEY, region TEXT, total BIGINT)",
    )
    .await
    .expect("create the target");
    exec_ddl(
        &server,
        &mut session,
        "CREATE PIPELINE gold AS (\
            STAGE apply (APPLY CHANGES INTO dim_orders FROM order_changes KEYS (id) \
                SEQUENCE BY _commit_version)\
        )",
    )
    .await
    .expect("create the pipeline");

    for id in 1..=4 {
        exec_dml(
            &server,
            &format!("INSERT INTO orders VALUES ({id}, 'eu', {id})"),
        )
        .await;
    }
    exec_dml(&server, "UPDATE orders SET total = 40 WHERE id = 4").await;
    exec_ddl(&server, &mut session, "RUN PIPELINE gold")
        .await
        .expect("the first apply");
    assert_eq!(count(&server, "SELECT COUNT(*) FROM dim_orders").await, 4);
    assert_eq!(
        count(&server, "SELECT total FROM dim_orders WHERE id = 4").await,
        40,
        "the last change to a key wins"
    );

    exec_dml(&server, "DELETE FROM orders WHERE id = 1").await;
    exec_dml(&server, "UPDATE orders SET total = 22 WHERE id = 2").await;
    exec_ddl(&server, &mut session, "RUN PIPELINE gold")
        .await
        .expect("the second apply");
    assert_eq!(count(&server, "SELECT COUNT(*) FROM dim_orders").await, 3);
    assert_eq!(
        count(&server, "SELECT total FROM dim_orders WHERE id = 2").await,
        22
    );
    let (_, consumed) = position_of(&server, "order_changes");
    assert_eq!(
        consumed, 9,
        "four inserts, two updates of two records each, one delete"
    );

    // The stage's run is listed beside a statement's
    let runs = zyron_wire::change_stream_dispatch::apply_runs().list();
    assert!(
        runs.iter()
            .any(|r| r.target == "dim_orders" && r.rows_upserted > 0),
        "the stage's runs are recorded"
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_backfill_seeds_the_target_on_the_first_run_and_continues_on_the_next() {
    for storage in Storage::BOTH {
        a_backfill_seeds_the_target_on_the_first_run_and_continues_on_the_next_on(storage).await;
    }
}

async fn a_backfill_seeds_the_target_on_the_first_run_and_continues_on_the_next_on(
    storage: Storage,
) {
    println!("{storage}");
    let (server, _schema, _tmp) = create_test_server_with_cdc().await;
    let mut session = new_session();
    seeded(&server, &mut session, storage).await;
    for id in 1..=6 {
        exec_dml(
            &server,
            &format!("INSERT INTO orders VALUES ({id}, 'eu', {id})"),
        )
        .await;
    }
    // The stream is created after the rows exist, showing them first
    exec_ddl(
        &server,
        &mut session,
        "CREATE CHANGE STREAM order_changes ON TABLE orders SHOW INITIAL ROWS",
    )
    .await
    .expect("create the stream");
    exec_ddl(
        &server,
        &mut session,
        "CREATE TABLE dim_orders (id BIGINT PRIMARY KEY, region TEXT, total BIGINT)",
    )
    .await
    .expect("create the target");
    exec_ddl(
        &server,
        &mut session,
        "CREATE PIPELINE gold AS (\
            STAGE apply (APPLY CHANGES INTO dim_orders FROM order_changes KEYS (id))\
        )",
    )
    .await
    .expect("create the pipeline");

    // Run one seeds the target with every existing row
    exec_ddl(&server, &mut session, "RUN PIPELINE gold")
        .await
        .expect("the seeding run");
    assert_eq!(count(&server, "SELECT COUNT(*) FROM dim_orders").await, 6);

    // Run two continues from the feed with no change to the definition
    exec_dml(&server, "INSERT INTO orders VALUES (7, 'us', 7)").await;
    exec_dml(&server, "UPDATE orders SET total = 60 WHERE id = 6").await;
    exec_ddl(&server, &mut session, "RUN PIPELINE gold")
        .await
        .expect("the incremental run");
    assert_eq!(count(&server, "SELECT COUNT(*) FROM dim_orders").await, 7);
    assert_eq!(
        count(&server, "SELECT total FROM dim_orders WHERE id = 6").await,
        60
    );

    // And a third run finds nothing, the seed never repeats
    exec_ddl(&server, &mut session, "RUN PIPELINE gold")
        .await
        .expect("an empty run");
    assert_eq!(count(&server, "SELECT COUNT(*) FROM dim_orders").await, 7);
    assert_eq!(rows_processed(&server, "gold"), 0);
}

/// A pipeline on change data runs when its stream holds MIN ROWS pending
/// changes or MAX WAIT has passed with at least one, deciding off the
/// pending counter and never moving the position itself
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_pipeline_on_change_data_runs_at_min_rows_or_after_max_wait() {
    for storage in Storage::BOTH {
        a_pipeline_on_change_data_runs_at_min_rows_or_after_max_wait_on(storage).await;
    }
}

async fn a_pipeline_on_change_data_runs_at_min_rows_or_after_max_wait_on(storage: Storage) {
    println!("{storage}");
    use zyron_wire::change_data_trigger::{TriggerWatch, due_pipelines, run_due_pipelines};

    let (server, _schema, _tmp) = create_test_server_with_cdc().await;
    let mut session = new_session();
    seeded(&server, &mut session, storage).await;
    exec_ddl(
        &server,
        &mut session,
        "CREATE CHANGE STREAM order_changes ON TABLE orders",
    )
    .await
    .expect("create the stream");
    exec_ddl(
        &server,
        &mut session,
        "CREATE PIPELINE bronze ON CHANGE DATA FROM order_changes MIN ROWS 3 MAX WAIT 2 SECONDS \
         AS (STAGE land (CONSUME CHANGES FROM order_changes INTO bronze_orders))",
    )
    .await
    .expect("create the pipeline");
    let refused = exec_ddl(
        &server,
        &mut session,
        "CREATE PIPELINE nowhere ON CHANGE DATA FROM missing AS (STAGE land (SOURCE orders, TARGET t))",
    )
    .await
    .expect_err("a trigger over a stream that does not exist is refused");
    assert!(refused.contains("not a change stream"), "{refused}");

    let mut watch = TriggerWatch::new();
    let start = 1_000_000_000i64;
    assert!(
        due_pipelines(&server, &mut watch, start).is_empty(),
        "nothing pends"
    );

    // Two pending changes are below the bound and the wait has not passed
    exec_dml(&server, "INSERT INTO orders VALUES (1, 'eu', 1)").await;
    exec_dml(&server, "INSERT INTO orders VALUES (2, 'eu', 2)").await;
    assert!(due_pipelines(&server, &mut watch, start + 100_000).is_empty());
    let (_, consumed) = position_of(&server, "order_changes");
    assert_eq!(consumed, 0, "deciding moves nothing");

    // The third reaches the bound and the run takes all three
    exec_dml(&server, "INSERT INTO orders VALUES (3, 'eu', 3)").await;
    let outcomes = run_due_pipelines(&server, &mut watch, start + 200_000).await;
    assert_eq!(outcomes.len(), 1, "{outcomes:?}");
    assert_eq!(outcomes[0].0, "bronze");
    assert!(outcomes[0].1.is_ok(), "{outcomes:?}");
    assert_eq!(
        count(&server, "SELECT COUNT(*) FROM bronze_orders").await,
        3
    );
    let (_, consumed) = position_of(&server, "order_changes");
    assert_eq!(consumed, 3);
    assert!(
        due_pipelines(&server, &mut watch, start + 300_000).is_empty(),
        "drained"
    );

    // One pending change waits, and runs once the wait has passed
    exec_dml(&server, "INSERT INTO orders VALUES (4, 'eu', 4)").await;
    assert!(due_pipelines(&server, &mut watch, start + 400_000).is_empty());
    assert!(due_pipelines(&server, &mut watch, start + 1_400_000).is_empty());
    let outcomes = run_due_pipelines(&server, &mut watch, start + 2_500_000).await;
    assert_eq!(outcomes.len(), 1, "the wait passed: {outcomes:?}");
    assert_eq!(
        count(&server, "SELECT COUNT(*) FROM bronze_orders").await,
        4
    );

    // A drained stream resets the wait, so the next change waits in full
    exec_dml(&server, "INSERT INTO orders VALUES (5, 'eu', 5)").await;
    assert!(due_pipelines(&server, &mut watch, start + 4_000_000).is_empty());
    assert_eq!(
        due_pipelines(&server, &mut watch, start + 6_100_000).len(),
        1,
        "two seconds after the change arrived"
    );
}
