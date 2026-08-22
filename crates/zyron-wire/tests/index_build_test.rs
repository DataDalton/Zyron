//! Building a B+tree index over rows that already exist.
//!
//! CREATE INDEX on a populated table created an empty tree and never filled
//! it, so every query the planner then routed through that index returned
//! nothing for the rows that predated it. REINDEX did fill its tree, but read
//! the heap through a freshly constructed handle whose page count came off the
//! file on disk, and skipped any page the buffer pool did not already hold, so
//! it dropped rows in both directions on exactly the large tables that need it.
//!
//! Both statements now collect live rows through one shared path. These tests
//! pin what that path has to cover.

mod common;

use common::{
    create_test_server, create_test_server_with_pool_frames, exec_ddl, exec_dml, exec_dml_result,
    new_session, query_rows, query_values,
};
use zyron_executor::column::ScalarValue;

/// The lake log version a table currently sits at, so a test can name a
/// version it wrote rather than guess one
fn current_lake_version(
    server: &std::sync::Arc<zyron_wire::connection::ServerState>,
    table: &str,
) -> u64 {
    let entry = server
        .catalog
        .list_all_tables()
        .into_iter()
        .find(|t| t.name == table)
        .expect("table");
    let paths = zyron_lake::LakePaths::new(server.disk_manager.data_dir(), entry.id.0);
    zyron_lake::TransactionLog::lookup_shared(&paths)
        .expect("lake log")
        .latest_version()
}

/// The first column of every row as an integer, so a test can name the
/// values it expects rather than the scalar shape they arrive in
fn first_ints(rows: &[Vec<ScalarValue>]) -> Vec<i64> {
    rows.iter()
        .map(|row| match row.first() {
            Some(ScalarValue::Int8(v)) => *v as i64,
            Some(ScalarValue::Int16(v)) => *v as i64,
            Some(ScalarValue::Int32(v)) => *v as i64,
            Some(ScalarValue::Int64(v)) => *v,
            other => panic!("expected an integer column, got {other:?}"),
        })
        .collect()
}

#[tokio::test]
async fn test_create_index_on_a_populated_table_covers_the_rows_that_predate_it() {
    let (server, _schema, _tmp) = create_test_server().await;
    let mut session = new_session();

    exec_ddl(&server, &mut session, "CREATE TABLE t (id INT, v INT)")
        .await
        .expect("create");
    for i in 0..30 {
        exec_dml(&server, &format!("INSERT INTO t VALUES ({i}, {})", i * 10)).await;
    }

    // The answer before the index exists is the answer the index must not change
    let before = query_values(&server, "SELECT v FROM t WHERE id = 7").await;
    assert_eq!(before.len(), 1, "the row is there before any index");

    exec_ddl(&server, &mut session, "CREATE INDEX t_id_idx ON t (id)")
        .await
        .expect("create index");

    let after = query_values(&server, "SELECT v FROM t WHERE id = 7").await;
    assert_eq!(
        after, before,
        "creating an index changed the answer, so the index is missing rows"
    );

    // Every key, not just one, and rows inserted after the index still land
    for i in 0..30 {
        assert_eq!(
            query_rows(&server, &format!("SELECT v FROM t WHERE id = {i}")).await,
            1,
            "id {i} is missing from the index"
        );
    }
    exec_dml(&server, "INSERT INTO t VALUES (100, 1000)").await;
    assert_eq!(
        query_rows(&server, "SELECT v FROM t WHERE id = 100").await,
        1
    );
    assert_eq!(query_rows(&server, "SELECT v FROM t").await, 31);
}

#[tokio::test]
async fn test_create_index_covers_rows_on_pages_the_buffer_pool_does_not_hold() {
    // Four frames, so a table of more than four pages cannot be resident. A
    // page read straight off disk must be indexed like any other; skipping it
    // is how an index silently loses whole pages of rows
    let (server, _schema, _tmp) = create_test_server_with_pool_frames(4).await;
    let mut session = new_session();

    exec_ddl(
        &server,
        &mut session,
        "CREATE TABLE wide (id INT, payload VARCHAR)",
    )
    .await
    .expect("create");

    // Rows wide enough that a few hundred span well past four pages
    let padding = "y".repeat(400);
    for chunk in 0..12 {
        let values: String = (0..50)
            .map(|i| format!("({}, '{padding}')", chunk * 50 + i))
            .collect::<Vec<_>>()
            .join(", ");
        exec_dml(&server, &format!("INSERT INTO wide VALUES {values}")).await;
    }
    assert_eq!(query_rows(&server, "SELECT id FROM wide").await, 600);

    exec_ddl(
        &server,
        &mut session,
        "CREATE INDEX wide_id_idx ON wide (id)",
    )
    .await
    .expect("create index");

    // Probe across the whole key range, so a page dropped anywhere shows up
    for id in [0, 1, 137, 299, 300, 451, 598, 599] {
        assert_eq!(
            query_rows(
                &server,
                &format!("SELECT payload FROM wide WHERE id = {id}")
            )
            .await,
            1,
            "id {id} is missing from the index, so its page was skipped"
        );
    }
    assert_eq!(query_rows(&server, "SELECT id FROM wide").await, 600);
}

#[tokio::test]
async fn test_an_index_built_after_deletes_does_not_resurrect_them() {
    // REINDEX shares this collection path, and it is the one place where being
    // wrong in the other direction matters: an index that carries a deleted
    // row makes an index scan return a row a sequential scan does not
    let (server, _schema, _tmp) = create_test_server_with_pool_frames(4).await;
    let mut session = new_session();

    exec_ddl(
        &server,
        &mut session,
        "CREATE TABLE r (id INT, payload VARCHAR)",
    )
    .await
    .expect("create");

    let padding = "z".repeat(400);
    for chunk in 0..8 {
        let values: String = (0..50)
            .map(|i| format!("({}, '{padding}')", chunk * 50 + i))
            .collect::<Vec<_>>()
            .join(", ");
        exec_dml(&server, &format!("INSERT INTO r VALUES {values}")).await;
    }
    exec_dml(&server, "DELETE FROM r WHERE id >= 380").await;
    assert_eq!(query_rows(&server, "SELECT id FROM r").await, 380);

    exec_ddl(&server, &mut session, "CREATE INDEX r_id_idx ON r (id)")
        .await
        .expect("create index");

    for id in [0, 199, 379] {
        assert_eq!(
            query_rows(&server, &format!("SELECT payload FROM r WHERE id = {id}")).await,
            1,
            "live row {id} is missing from the index"
        );
    }
    for id in [380, 399] {
        assert_eq!(
            query_rows(&server, &format!("SELECT payload FROM r WHERE id = {id}")).await,
            0,
            "deleted row {id} came back through the index"
        );
    }
    assert_eq!(query_rows(&server, "SELECT id FROM r").await, 380);
}

#[tokio::test]
async fn test_create_index_on_a_lake_table_covers_every_row_and_keeps_covering_them() {
    let (server, _schema, _tmp) = create_test_server().await;
    let mut session = new_session();

    exec_ddl(
        &server,
        &mut session,
        "CREATE TABLE lt (id INT, v INT) USING ZYRONLAKE",
    )
    .await
    .expect("create lake");
    let values: String = (0..40)
        .map(|i| format!("({i}, {})", i * 3))
        .collect::<Vec<_>>()
        .join(", ");
    exec_dml(&server, &format!("INSERT INTO lt VALUES {values}")).await;

    // The answers before the index exists are the answers it must not change
    let before = query_values(&server, "SELECT v FROM lt WHERE id = 11").await;
    assert_eq!(before.len(), 1);

    exec_ddl(&server, &mut session, "CREATE INDEX lt_id_idx ON lt (id)")
        .await
        .expect("create index");

    // Every key the table already held
    for i in 0..40 {
        assert_eq!(
            first_ints(&query_values(&server, &format!("SELECT v FROM lt WHERE id = {i}")).await),
            vec![i * 3],
            "id {i} is missing from the lake index"
        );
    }
    assert_eq!(query_rows(&server, "SELECT v FROM lt").await, 40);

    // Rows written after the build are indexed by the commit that writes
    // them, so a probe finds them without a rebuild
    exec_dml(&server, "INSERT INTO lt VALUES (100, 300), (101, 303)").await;
    assert_eq!(
        first_ints(&query_values(&server, "SELECT v FROM lt WHERE id = 100").await),
        vec![300]
    );
    assert_eq!(query_rows(&server, "SELECT v FROM lt").await, 42);

    // A deleted row stops being returned even though its index entry is
    // still on disk, because the fetch filters through the delete
    // predicate exactly as a scan does
    exec_dml(&server, "DELETE FROM lt WHERE id = 11").await;
    assert_eq!(
        query_rows(&server, "SELECT v FROM lt WHERE id = 11").await,
        0
    );
    assert_eq!(query_rows(&server, "SELECT v FROM lt").await, 41);

    // An update rewrites the row into a new file, and the index follows it
    exec_dml(&server, "UPDATE lt SET v = 999 WHERE id = 12").await;
    assert_eq!(
        first_ints(&query_values(&server, "SELECT v FROM lt WHERE id = 12").await),
        vec![999]
    );
    assert_eq!(query_rows(&server, "SELECT v FROM lt").await, 41);
}

#[tokio::test]
async fn test_a_range_predicate_on_an_indexed_lake_column_returns_the_same_rows() {
    let (server, _schema, _tmp) = create_test_server().await;
    let mut session = new_session();

    exec_ddl(
        &server,
        &mut session,
        "CREATE TABLE lrange (id INT, v INT) USING ZYRONLAKE",
    )
    .await
    .expect("create lake");
    // v runs opposite to id, so the file bounds on v cannot narrow a range
    // over it and the index is the only thing that can
    for chunk in 0..4 {
        let values: String = (chunk * 25..chunk * 25 + 25)
            .map(|i| format!("({i}, {})", 1000 - i))
            .collect::<Vec<_>>()
            .join(", ");
        exec_dml(&server, &format!("INSERT INTO lrange VALUES {values}")).await;
    }

    // The answers before the index exists are the answers it must not change
    let before_closed =
        query_rows(&server, "SELECT id FROM lrange WHERE v >= 950 AND v <= 970").await;
    let before_open = query_rows(&server, "SELECT id FROM lrange WHERE v > 990").await;
    assert_eq!(before_closed, 21);
    assert_eq!(before_open, 10);

    exec_ddl(
        &server,
        &mut session,
        "CREATE INDEX lrange_v_ix ON lrange (v)",
    )
    .await
    .expect("create index");

    assert_eq!(
        query_rows(&server, "SELECT id FROM lrange WHERE v >= 950 AND v <= 970").await,
        before_closed,
        "a closed range changed answer once the index was consulted"
    );
    assert_eq!(
        query_rows(&server, "SELECT id FROM lrange WHERE v > 990").await,
        before_open,
        "an open range changed answer once the index was consulted"
    );
    // Exclusive ends drop their own endpoints and nothing else
    assert_eq!(
        query_rows(&server, "SELECT id FROM lrange WHERE v > 950 AND v < 970").await,
        19
    );
    // A range past every stored value selects nothing
    assert_eq!(
        query_rows(&server, "SELECT id FROM lrange WHERE v > 5000").await,
        0
    );
    assert_eq!(query_rows(&server, "SELECT id FROM lrange").await, 100);
}

#[tokio::test]
async fn test_a_unique_index_on_a_lake_table_refuses_a_duplicate() {
    let (server, _schema, _tmp) = create_test_server().await;
    let mut session = new_session();

    exec_ddl(
        &server,
        &mut session,
        "CREATE TABLE luq (id INT, code INT) USING ZYRONLAKE",
    )
    .await
    .expect("create lake");
    exec_dml(&server, "INSERT INTO luq VALUES (1, 100), (2, 200)").await;
    exec_ddl(
        &server,
        &mut session,
        "CREATE UNIQUE INDEX luq_code_ix ON luq (code)",
    )
    .await
    .expect("create unique index");

    // A declared unique index that admits a duplicate is a lie, and the
    // heap refuses this, so the lake has to as well
    let err = exec_dml_result(&server, "INSERT INTO luq VALUES (3, 100)").await;
    assert!(
        err.is_err(),
        "a duplicate landed under a unique index on a lake table"
    );
    assert_eq!(query_rows(&server, "SELECT id FROM luq").await, 2);

    // A fresh value still lands
    exec_dml(&server, "INSERT INTO luq VALUES (4, 400)").await;
    assert_eq!(query_rows(&server, "SELECT id FROM luq").await, 3);
}

#[tokio::test]
async fn test_a_lake_update_enforces_uniqueness_without_colliding_with_itself() {
    let (server, _schema, _tmp) = create_test_server().await;
    let mut session = new_session();

    exec_ddl(
        &server,
        &mut session,
        "CREATE TABLE luu (id INT, code INT) USING ZYRONLAKE",
    )
    .await
    .expect("create lake");
    exec_dml(
        &server,
        "INSERT INTO luu VALUES (1, 100), (2, 200), (3, 300)",
    )
    .await;
    exec_ddl(
        &server,
        &mut session,
        "CREATE UNIQUE INDEX luu_code_ix ON luu (code)",
    )
    .await
    .expect("create unique index");

    // Rewriting a row without touching its key must not collide with the
    // copy it is replacing
    exec_dml(&server, "UPDATE luu SET id = 11 WHERE code = 100").await;
    assert_eq!(
        query_rows(&server, "SELECT id FROM luu WHERE code = 100").await,
        1
    );
    assert_eq!(query_rows(&server, "SELECT id FROM luu").await, 3);

    // Moving a row onto a key a surviving row holds must still be refused
    let err = exec_dml_result(&server, "UPDATE luu SET code = 200 WHERE code = 300").await;
    assert!(
        err.is_err(),
        "an update produced a duplicate under a unique index"
    );
    assert_eq!(
        query_rows(&server, "SELECT id FROM luu WHERE code = 200").await,
        1
    );
    assert_eq!(query_rows(&server, "SELECT id FROM luu").await, 3);

    // Moving a row onto a free key still works
    exec_dml(&server, "UPDATE luu SET code = 400 WHERE code = 300").await;
    assert_eq!(
        query_rows(&server, "SELECT id FROM luu WHERE code = 400").await,
        1
    );
    assert_eq!(query_rows(&server, "SELECT id FROM luu").await, 3);
}

#[tokio::test]
async fn test_time_travel_on_an_indexed_lake_table_reads_the_past_version() {
    let (server, _schema, _tmp) = create_test_server().await;
    let mut session = new_session();

    exec_ddl(
        &server,
        &mut session,
        "CREATE TABLE ltt (id INT, v INT) USING ZYRONLAKE",
    )
    .await
    .expect("create lake");
    exec_dml(&server, "INSERT INTO ltt VALUES (1, 10), (2, 20)").await;
    exec_ddl(&server, &mut session, "CREATE INDEX ltt_id_ix ON ltt (id)")
        .await
        .expect("create index");
    let two_rows = current_lake_version(&server, "ltt");

    exec_dml(&server, "INSERT INTO ltt VALUES (3, 30), (4, 40)").await;
    assert_eq!(query_rows(&server, "SELECT id FROM ltt").await, 4);

    // Time travel works on an indexed lake table, and reads the file set
    // that version named rather than the newest one
    assert_eq!(
        query_rows(
            &server,
            &format!("SELECT id FROM ltt VERSION AS OF {two_rows}")
        )
        .await,
        2,
        "a past version returned the newest row set"
    );
    // A predicate at a past version resolves against that version too
    assert_eq!(
        query_rows(
            &server,
            &format!("SELECT id FROM ltt VERSION AS OF {two_rows} WHERE id = 3")
        )
        .await,
        0,
        "row 3 did not exist at that version"
    );
    assert_eq!(
        query_rows(
            &server,
            &format!("SELECT id FROM ltt VERSION AS OF {two_rows} WHERE id = 1")
        )
        .await,
        1
    );
}

#[tokio::test]
async fn test_dropping_a_lake_index_leaves_every_row_readable() {
    let (server, _schema, _tmp) = create_test_server().await;
    let mut session = new_session();

    exec_ddl(
        &server,
        &mut session,
        "CREATE TABLE ld (id INT, v INT) USING ZYRONLAKE",
    )
    .await
    .expect("create lake");
    let values: String = (0..25)
        .map(|i| format!("({i}, {})", i * 7))
        .collect::<Vec<_>>()
        .join(", ");
    exec_dml(&server, &format!("INSERT INTO ld VALUES {values}")).await;
    exec_ddl(&server, &mut session, "CREATE INDEX ld_id_idx ON ld (id)")
        .await
        .expect("create index");
    assert_eq!(
        query_rows(&server, "SELECT v FROM ld WHERE id = 9").await,
        1
    );

    exec_ddl(&server, &mut session, "DROP INDEX ld_id_idx")
        .await
        .expect("drop index");

    // Without the index every query falls back to the scan and answers the
    // same, which is what makes the index a way to read less and not a
    // second source of truth
    for i in 0..25 {
        assert_eq!(
            first_ints(&query_values(&server, &format!("SELECT v FROM ld WHERE id = {i}")).await),
            vec![i * 7],
            "id {i} became unreadable after the index was dropped"
        );
    }
    assert_eq!(query_rows(&server, "SELECT v FROM ld").await, 25);
}

#[tokio::test]
async fn test_reindex_rebuilds_a_lake_index_and_it_still_answers() {
    let (server, _schema, _tmp) = create_test_server().await;
    let mut session = new_session();

    exec_ddl(
        &server,
        &mut session,
        "CREATE TABLE lr (id INT, v INT) USING ZYRONLAKE",
    )
    .await
    .expect("create lake");
    for chunk in 0..4 {
        let values: String = (chunk * 10..chunk * 10 + 10)
            .map(|i| format!("({i}, {})", i * 2))
            .collect::<Vec<_>>()
            .join(", ");
        exec_dml(&server, &format!("INSERT INTO lr VALUES {values}")).await;
    }
    exec_ddl(&server, &mut session, "CREATE INDEX lr_id_idx ON lr (id)")
        .await
        .expect("create index");

    // REINDEX is a utility statement handled on the connection, so the
    // rebuild it routes to is called directly here
    let table = server.catalog.get_table(_schema, "lr").expect("table");
    zyron_wire::index_build::rebuild_lake_indexes(&server, &table)
        .await
        .expect("rebuild");

    for i in 0..40 {
        assert_eq!(
            first_ints(&query_values(&server, &format!("SELECT v FROM lr WHERE id = {i}")).await),
            vec![i * 2],
            "id {i} is missing after REINDEX"
        );
    }
    assert_eq!(query_rows(&server, "SELECT v FROM lr").await, 40);
}
