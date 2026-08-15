//! The TPC-H and TPC-C workloads run in process against the engine.
//!
//! The generators and query text have unit tests of their own in
//! `zyron-tpc`, but those only prove the SQL is well formed as text. What
//! matters is whether the engine executes it, so these load a small database
//! through the real DDL and DML path and run every query and every
//! transaction against it.
//!
//! A query the engine refuses fails this suite by name. That is the point: a
//! benchmark that skipped the queries it could not run would report a number
//! for a workload nobody asked for.
//!
//! Run: cargo test -p zyron-wire --test tpc_workload_test

mod common;

use common::{create_test_server, exec_ddl, exec_dml_result, new_session, query_values};

/// Loads a schema and streams a generated dataset through the DML path.
async fn load(
    server: &std::sync::Arc<zyron_wire::connection::ServerState>,
    session: &mut Option<zyron_wire::session::Session>,
    schema: Vec<&'static str>,
    generate: impl FnOnce(&mut dyn FnMut(&str) -> Result<(), String>) -> Result<(), String>,
) {
    for ddl in schema {
        exec_ddl(server, session, ddl)
            .await
            .unwrap_or_else(|e| panic!("schema statement failed: {ddl}\n{e}"));
    }
    // The generator hands back SQL faster than a test wants to await inside
    // its callback, so the statements are collected and then executed. The
    // scale factors here are small enough that this is bounded
    let mut statements: Vec<String> = Vec::new();
    let mut sink = |sql: &str| {
        statements.push(sql.to_string());
        Ok(())
    };
    generate(&mut sink).expect("generate");
    for sql in &statements {
        exec_dml_result(server, sql).await.unwrap_or_else(|e| {
            panic!("load statement failed: {}\n{e}", &sql[..sql.len().min(160)])
        });
    }
}

/// Every TPC-H query executes against a loaded database.
///
/// The scale factor is small so the suite stays quick. Correctness of the
/// answers is the benchmark driver's business; what is pinned here is that
/// the engine runs all 22 rather than refusing some.
#[tokio::test]
async fn test_every_tpch_query_executes_against_a_loaded_database() {
    let (server, _schema, _tmp) = create_test_server().await;
    let mut session = new_session();

    load(&server, &mut session, zyron_tpc::tpch::schema(), |sink| {
        zyron_tpc::tpch::generate(0.0002, 200, sink)
    })
    .await;

    // Every query is attempted so one refusal does not hide the others. A
    // query returning no rows still executed, which is a fact about the
    // data rather than a failure
    let mut refused: Vec<String> = Vec::new();
    for (name, sql) in zyron_tpc::tpch::queries() {
        if let Err(e) = common::query_result(&server, sql).await {
            refused.push(format!("{name}: {e}"));
        }
    }
    assert!(
        refused.is_empty(),
        "the engine refused {} of 22 TPC-H queries:\n{}",
        refused.len(),
        refused.join("\n")
    );
}

/// The TPC-H load itself round-trips: the row counts the generator promises
/// are the row counts the engine stores.
#[tokio::test]
async fn test_the_tpch_load_stores_the_rows_the_generator_promises() {
    let (server, _schema, _tmp) = create_test_server().await;
    let mut session = new_session();
    let scale = 0.0002;

    load(&server, &mut session, zyron_tpc::tpch::schema(), |sink| {
        zyron_tpc::tpch::generate(scale, 200, sink)
    })
    .await;

    for (table, promised) in zyron_tpc::tpch::row_counts(scale) {
        let rows = query_values(&server, &format!("SELECT COUNT(*) FROM {table}")).await;
        let stored = match rows.first().and_then(|r| r.first()) {
            Some(zyron_executor::column::ScalarValue::Int64(v)) => *v,
            other => panic!("COUNT(*) on {table} returned {other:?}"),
        };
        if table == "lineitem" {
            // One to seven lines per order, so the promise is an average
            assert!(stored > 0, "lineitem loaded nothing");
        } else {
            assert_eq!(
                stored, promised,
                "{table} stored {stored}, promised {promised}"
            );
        }
    }
}

/// Every TPC-C transaction executes against a loaded database.
///
/// One warehouse with the specification's full item count would be 100k
/// items and 100k stock rows, which is more than this suite should build, so
/// the workload runs against a warehouse loaded at reduced item breadth. The
/// statements are the same ones the driver issues.
#[tokio::test]
async fn test_every_tpcc_transaction_executes_against_a_loaded_database() {
    let (server, _schema, _tmp) = create_test_server().await;
    let mut session = new_session();

    for ddl in zyron_tpc::tpcc::schema() {
        exec_ddl(&server, &mut session, ddl)
            .await
            .unwrap_or_else(|e| panic!("schema statement failed: {ddl}\n{e}"));
    }

    // A hand-built warehouse rather than the full generator, so the suite
    // exercises the transactions without loading a hundred thousand items
    let seed = [
        "INSERT INTO warehouse VALUES (1, 'w1', 's1', 's2', 'city', 'CA', '900001111', 0.0500, 300000.00)",
        "INSERT INTO district VALUES (1, 1, 'd1', 's1', 's2', 'city', 'CA', '900001111', 0.0500, 30000.00, 3001)",
        "INSERT INTO item VALUES (1, 100, 'item one', 10.00, 'data ORIGINAL here')",
        "INSERT INTO item VALUES (2, 200, 'item two', 20.00, 'plain data here')",
        "INSERT INTO stock VALUES (1, 1, 50, 'd1', 'd2', 'd3', 'd4', 'd5', 'd6', 'd7', 'd8', 'd9', 'd10', 0, 0, 0, 'sdata')",
        "INSERT INTO stock VALUES (2, 1, 8, 'd1', 'd2', 'd3', 'd4', 'd5', 'd6', 'd7', 'd8', 'd9', 'd10', 0, 0, 0, 'sdata')",
        "INSERT INTO customer VALUES (1, 1, 1, 'first', 'OE', 'BARBARBAR', 's1', 's2', 'city', 'CA', '900001111', '1234567890123456', TIMESTAMP '2026-01-01 00:00:00', 'GC', 50000.00, 0.0500, -10.00, 10.00, 1, 0, 'cdata')",
        "INSERT INTO orders VALUES (3000, 1, 1, 1, TIMESTAMP '2026-01-01 00:00:00', NULL, 5, 1)",
        "INSERT INTO new_order VALUES (3000, 1, 1)",
        "INSERT INTO order_line VALUES (3000, 1, 1, 1, 1, 1, NULL, 5, 50.00, 'dist-info')",
        "INSERT INTO history VALUES (1, 1, 1, 1, 1, TIMESTAMP '2026-01-01 00:00:00', 10.00, 'seed')",
    ];
    for sql in seed {
        exec_dml_result(&server, sql)
            .await
            .unwrap_or_else(|e| panic!("seed failed: {sql}\n{e}"));
    }

    let mut rng = zyron_tpc::Rng::new(17);
    let mut refused: Vec<String> = Vec::new();
    for kind in [
        zyron_tpc::tpcc::Transaction::NewOrder,
        zyron_tpc::tpcc::Transaction::Payment,
        zyron_tpc::tpcc::Transaction::OrderStatus,
        zyron_tpc::tpcc::Transaction::Delivery,
        zyron_tpc::tpcc::Transaction::StockLevel,
    ] {
        // A single warehouse and district, so every generated key resolves
        for sql in zyron_tpc::tpcc::statements(kind, 1, &mut rng) {
            let sql = sql
                .replace("c_d_id = 2", "c_d_id = 1")
                .replace("d_id = 2", "d_id = 1");
            if let Err(e) = exec_dml_result(&server, &sql).await {
                refused.push(format!("{}: {e}\n  {sql}", kind.name()));
            }
        }
    }
    assert!(
        refused.is_empty(),
        "the engine refused {} TPC-C statements:\n{}",
        refused.len(),
        refused.join("\n")
    );
}
