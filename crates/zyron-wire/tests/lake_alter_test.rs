//! Column-shape ALTER on a lake table commits a schema change to the
//! table's transaction log instead of running the heap rewrite engine.
//!
//! The heap rewrite re-encodes every row into side heap files and swaps
//! the catalog. A lake table's rows live in its log, so that path either
//! desynced the catalog from the lake schema (bricking every SELECT) or
//! re-appended every existing row while the old files stayed live,
//! doubling the table.
//!
//! Run: cargo test -p zyron-wire --test lake_alter_test

mod common;

use common::{create_test_server, exec_ddl, exec_dml, new_session, query_values};
use zyron_executor::column::ScalarValue;

#[tokio::test]
async fn test_lake_add_column_serves_null_for_old_rows() {
    let (server, _schema, _tmp) = create_test_server().await;
    let mut session = new_session();
    exec_ddl(
        &server,
        &mut session,
        "CREATE TABLE lk (id BIGINT NOT NULL, name TEXT) USING ZYRONLAKE",
    )
    .await
    .expect("create");
    exec_dml(&server, "INSERT INTO lk VALUES (1, 'a'), (2, 'b')").await;

    exec_ddl(&server, &mut session, "ALTER TABLE lk ADD COLUMN v BIGINT")
        .await
        .expect("add column");

    // Old rows read back whole, with NULL in the added column
    let rows = query_values(&server, "SELECT id, name, v FROM lk ORDER BY id").await;
    assert_eq!(
        rows.len(),
        2,
        "both existing rows survive the schema change"
    );
    assert_eq!(rows[0][2], ScalarValue::Null);
    assert_eq!(rows[1][2], ScalarValue::Null);

    // New rows carry the column
    exec_dml(&server, "INSERT INTO lk VALUES (3, 'c', 30)").await;
    let rows = query_values(&server, "SELECT v FROM lk WHERE id = 3").await;
    assert_eq!(rows.len(), 1);
    assert_eq!(rows[0][0], ScalarValue::Int64(30));
}

#[tokio::test]
async fn test_lake_drop_column_keeps_every_row_once() {
    let (server, _schema, _tmp) = create_test_server().await;
    let mut session = new_session();
    exec_ddl(
        &server,
        &mut session,
        "CREATE TABLE lk (id BIGINT NOT NULL, name TEXT) USING ZYRONLAKE",
    )
    .await
    .expect("create");
    exec_dml(&server, "INSERT INTO lk VALUES (1, 'a'), (2, 'b')").await;

    exec_ddl(&server, &mut session, "ALTER TABLE lk DROP COLUMN name")
        .await
        .expect("drop column");

    let rows = query_values(&server, "SELECT id FROM lk ORDER BY id").await;
    assert_eq!(
        rows.iter()
            .map(|r| match r[0] {
                ScalarValue::Int64(v) => v,
                ref other => panic!("expected Int64, got {other:?}"),
            })
            .collect::<Vec<i64>>(),
        vec![1, 2],
        "each row exists exactly once after the drop"
    );

    // The narrowed shape accepts inserts
    exec_dml(&server, "INSERT INTO lk VALUES (3)").await;
    assert_eq!(query_values(&server, "SELECT id FROM lk").await.len(), 3);
}

/// The catalog keeps a dropped column's place, so a row image is laid out
/// over every column the table ever had while the lake holds the live ones
/// alone. Every write path has to agree on that, and a column dropped from
/// the middle of the list is where a positional slip would show
#[tokio::test]
async fn test_lake_writes_after_a_drop_from_the_middle_of_the_list() {
    let (server, _schema, _tmp) = create_test_server().await;
    let mut session = new_session();
    exec_ddl(
        &server,
        &mut session,
        "CREATE TABLE lk (id BIGINT NOT NULL, gone TEXT NOT NULL, keep BIGINT) USING ZYRONLAKE",
    )
    .await
    .expect("create");
    exec_dml(
        &server,
        "INSERT INTO lk VALUES (1, 'x', 10), (2, 'y', 20), (3, 'z', 30)",
    )
    .await;
    exec_ddl(&server, &mut session, "ALTER TABLE lk DROP COLUMN gone")
        .await
        .expect("drop column");
    let table = server
        .catalog
        .get_table(_schema, "lk")
        .expect("the table exists");
    assert_eq!(table.columns.len(), 3, "the placeholder stays in the entry");
    assert_eq!(table.live_columns().count(), 2);

    exec_dml(&server, "INSERT INTO lk VALUES (4, 40)").await;
    exec_dml(&server, "UPDATE lk SET keep = keep + 1 WHERE id = 1").await;
    exec_dml(&server, "DELETE FROM lk WHERE id = 2").await;
    exec_dml(&server, "UPDATE lk SET keep = keep * 2 WHERE id = 4").await;

    let rows = query_values(&server, "SELECT id, keep FROM lk ORDER BY id").await;
    assert_eq!(
        rows,
        vec![
            vec![ScalarValue::Int64(1), ScalarValue::Int64(11)],
            vec![ScalarValue::Int64(3), ScalarValue::Int64(30)],
            vec![ScalarValue::Int64(4), ScalarValue::Int64(80)],
        ]
    );
    assert_eq!(
        query_values(&server, "SELECT COUNT(*) FROM lk").await[0][0],
        ScalarValue::Int64(3)
    );

    // A column added after the drop takes the next id and reads back
    exec_ddl(
        &server,
        &mut session,
        "ALTER TABLE lk ADD COLUMN later TEXT",
    )
    .await
    .expect("add column");
    exec_dml(&server, "INSERT INTO lk VALUES (5, 50, 'late')").await;
    let rows = query_values(&server, "SELECT id, keep, later FROM lk ORDER BY id").await;
    assert_eq!(rows.len(), 4);
    assert_eq!(rows[0][2], ScalarValue::Null);
    assert_eq!(rows[3][2], ScalarValue::Utf8("late".to_string()));
}

/// Narrowing a column's type would reinterpret every stored cell, which
/// needs a data rewrite the lake path does not run, so it is refused
/// rather than doubling or corrupting the table.
#[tokio::test]
async fn test_lake_narrowing_a_column_is_refused() {
    let (server, _schema, _tmp) = create_test_server().await;
    let mut session = new_session();
    exec_ddl(
        &server,
        &mut session,
        "CREATE TABLE lk (id BIGINT NOT NULL, v BIGINT) USING ZYRONLAKE",
    )
    .await
    .expect("create");
    exec_dml(&server, "INSERT INTO lk VALUES (1, 10)").await;

    let refused = exec_ddl(
        &server,
        &mut session,
        "ALTER TABLE lk ALTER COLUMN v TYPE INT",
    )
    .await
    .expect_err("a lake column is not narrowed");
    assert!(refused.contains("full rewrite"), "{refused}");

    // The refusal changed nothing
    let rows = query_values(&server, "SELECT id, v FROM lk").await;
    assert_eq!(rows.len(), 1);
    assert_eq!(rows[0][1], ScalarValue::Int64(10));
}

/// The ids one query returned, in the order it returned them
fn ids(rows: &[Vec<ScalarValue>]) -> Vec<i64> {
    rows.iter()
        .map(|r| match r[0] {
            ScalarValue::Int64(v) => v,
            ref other => panic!("expected Int64, got {other:?}"),
        })
        .collect()
}

/// A data file keeps its cells at the width its columns had when it was
/// written. Widening a column is a schema change, and every file written
/// before it hands its cells over widened, so a scan, a predicate answered
/// on stored bytes, a bloom, a metadata aggregate, an update, a delete, a
/// compaction and a read of an earlier version all see the values the
/// files hold at the width the column declares now
#[tokio::test]
async fn test_lake_widening_a_column_reads_every_file_at_the_new_width() {
    let (server, _schema, _tmp) = create_test_server().await;
    let mut session = new_session();
    exec_ddl(
        &server,
        &mut session,
        "CREATE TABLE lk (id BIGINT NOT NULL, total INTEGER, name TEXT) USING ZYRONLAKE",
    )
    .await
    .expect("create");
    // Two files written at four bytes per total, both signs and both
    // extremes among them
    exec_dml(
        &server,
        "INSERT INTO lk VALUES (1, -5, 'a'), (2, 7, 'b'), (3, 2147483647, 'c')",
    )
    .await;
    exec_dml(&server, "INSERT INTO lk VALUES (4, -2147483648, 'd')").await;

    exec_ddl(
        &server,
        &mut session,
        "ALTER TABLE lk ALTER COLUMN total TYPE BIGINT",
    )
    .await
    .expect("a widening is a schema change");
    // A value the old width could not hold lands in a file written at the
    // new one
    exec_dml(&server, "INSERT INTO lk VALUES (5, 5000000000, 'e')").await;

    let rows = query_values(&server, "SELECT total FROM lk ORDER BY id").await;
    assert_eq!(
        rows,
        vec![
            vec![ScalarValue::Int64(-5)],
            vec![ScalarValue::Int64(7)],
            vec![ScalarValue::Int64(2_147_483_647)],
            vec![ScalarValue::Int64(-2_147_483_648)],
            vec![ScalarValue::Int64(5_000_000_000)],
        ],
        "every file reads at the declared width, sign and extremes kept"
    );

    // Predicates over the widened column, answered on stored bytes for the
    // old files and on the new file alike
    assert_eq!(
        ids(&query_values(&server, "SELECT id FROM lk WHERE total = -5").await),
        vec![1]
    );
    assert_eq!(
        ids(&query_values(&server, "SELECT id FROM lk WHERE total > 100 ORDER BY id").await),
        vec![3, 5]
    );
    assert_eq!(
        ids(&query_values(&server, "SELECT id FROM lk WHERE total >= 5000000000").await),
        vec![5],
        "a constant the old width cannot hold matches only the new file"
    );
    assert_eq!(
        ids(&query_values(&server, "SELECT id FROM lk WHERE total < 0 ORDER BY id").await),
        vec![1, 4]
    );
    assert_eq!(
        ids(&query_values(
            &server,
            "SELECT id FROM lk WHERE total IN (7, -2147483648) ORDER BY id"
        )
        .await),
        vec![2, 4]
    );

    // Aggregates the manifest answers read the statistics the old files
    // recorded, which are values rather than bytes
    let rows = query_values(&server, "SELECT SUM(total), MIN(total), MAX(total) FROM lk").await;
    assert_eq!(rows.len(), 1);
    assert_eq!(rows[0][0], ScalarValue::Int64(5_000_000_001));
    assert_eq!(rows[0][1], ScalarValue::Int64(-2_147_483_648));
    assert_eq!(rows[0][2], ScalarValue::Int64(5_000_000_000));

    // The version before the widening reads the same rows at the width
    // the column declares now, since a plan is typed by the table as it
    // stands. The table's creation is version one and each insert one more
    let rows = query_values(&server, "SELECT total FROM lk VERSION AS OF 3 ORDER BY id").await;
    assert_eq!(
        rows,
        vec![
            vec![ScalarValue::Int64(-5)],
            vec![ScalarValue::Int64(7)],
            vec![ScalarValue::Int64(2_147_483_647)],
            vec![ScalarValue::Int64(-2_147_483_648)],
        ]
    );

    // An update reads the old file's cells widened and writes the new
    // image at the new width, a delete matches on the widened cells
    exec_dml(&server, "UPDATE lk SET total = total + 1 WHERE id = 1").await;
    assert_eq!(
        query_values(&server, "SELECT total FROM lk WHERE id = 1").await,
        vec![vec![ScalarValue::Int64(-4)]]
    );
    exec_dml(&server, "DELETE FROM lk WHERE total < 0").await;
    assert_eq!(
        ids(&query_values(&server, "SELECT id FROM lk ORDER BY id").await),
        vec![2, 3, 5]
    );

    // A compaction rewrites the survivors at the new width
    let optimized = common::wire_query(&server, &["OPTIMIZE TABLE lk"]).await;
    assert!(optimized[0].errors.is_empty(), "{:?}", optimized[0].errors);
    let rows = query_values(&server, "SELECT id, total FROM lk ORDER BY id").await;
    assert_eq!(
        rows,
        vec![
            vec![ScalarValue::Int64(2), ScalarValue::Int64(7)],
            vec![ScalarValue::Int64(3), ScalarValue::Int64(2_147_483_647)],
            vec![ScalarValue::Int64(5), ScalarValue::Int64(5_000_000_000)],
        ]
    );
}

/// The widenings beside the integer ones, a text bound growing, a
/// timestamp gaining digits past the microsecond, and a decimal gaining
/// scale. A column widened after its rows were written reads exactly as a
/// column declared that wide from the start
#[tokio::test]
async fn test_lake_widening_text_bounds_timestamps_and_decimals() {
    let (server, _schema, _tmp) = create_test_server().await;
    let mut session = new_session();
    exec_ddl(
        &server,
        &mut session,
        "CREATE TABLE w (id BIGINT NOT NULL, tag VARCHAR(4), at TIMESTAMP, price DECIMAL(10,2)) \
         USING ZYRONLAKE",
    )
    .await
    .expect("create");
    exec_ddl(
        &server,
        &mut session,
        "CREATE TABLE ctl (id BIGINT NOT NULL, tag VARCHAR(8), at TIMESTAMP(9), \
         price DECIMAL(12,4)) USING ZYRONLAKE",
    )
    .await
    .expect("create the control");
    let first = "(1, 'abcd', '2024-01-02 03:04:05.678901', 12.34)";
    let second = "(2, 'abcdefgh', '2024-01-02 03:04:05.678902', 56.7891)";
    exec_dml(&server, &format!("INSERT INTO w VALUES {first}")).await;
    exec_dml(&server, &format!("INSERT INTO ctl VALUES {first}")).await;

    for change in [
        "ALTER TABLE w ALTER COLUMN tag TYPE VARCHAR(8)",
        "ALTER TABLE w ALTER COLUMN at TYPE TIMESTAMP(9)",
        "ALTER TABLE w ALTER COLUMN price TYPE DECIMAL(12,4)",
    ] {
        exec_ddl(&server, &mut session, change)
            .await
            .expect("a widening is a schema change");
    }
    exec_dml(&server, &format!("INSERT INTO w VALUES {second}")).await;
    exec_dml(&server, &format!("INSERT INTO ctl VALUES {second}")).await;

    let widened = query_values(
        &server,
        "SELECT id, tag, CAST(at AS TEXT), CAST(price AS TEXT) FROM w ORDER BY id",
    )
    .await;
    let declared = query_values(
        &server,
        "SELECT id, tag, CAST(at AS TEXT), CAST(price AS TEXT) FROM ctl ORDER BY id",
    )
    .await;
    assert_eq!(widened.len(), 2);
    assert_eq!(
        widened, declared,
        "a widened column reads as one declared wide from the start"
    );
    assert_eq!(
        query_values(&server, "SELECT id FROM w WHERE tag = 'abcd'").await,
        vec![vec![ScalarValue::Int64(1)]]
    );
    assert_eq!(
        query_values(&server, "SELECT id FROM w WHERE price > 20").await,
        vec![vec![ScalarValue::Int64(2)]]
    );
}

/// A type change the stored cells do not read through as the new type is
/// refused as a rewrite, and a widening of a column a lake index is keyed
/// on is refused until the index is dropped, since its files hold the
/// keys at the old width
#[tokio::test]
async fn test_lake_type_change_that_is_not_a_widening_is_refused() {
    let (server, _schema, _tmp) = create_test_server().await;
    let mut session = new_session();
    exec_ddl(
        &server,
        &mut session,
        "CREATE TABLE r (id BIGINT NOT NULL, n BIGINT, m INTEGER, s TEXT) USING ZYRONLAKE",
    )
    .await
    .expect("create");
    exec_dml(&server, "INSERT INTO r VALUES (1, 10, 20, 'x')").await;

    for (change, reason) in [
        ("ALTER TABLE r ALTER COLUMN n TYPE INTEGER", "narrowing"),
        (
            "ALTER TABLE r ALTER COLUMN n TYPE TEXT",
            "a change of family",
        ),
        (
            "ALTER TABLE r ALTER COLUMN s TYPE VARCHAR(2)",
            "bounding an unbounded text",
        ),
    ] {
        let refused = exec_ddl(&server, &mut session, change)
            .await
            .expect_err(reason);
        assert!(refused.contains("full rewrite"), "{reason}: {refused}");
    }
    assert_eq!(
        query_values(&server, "SELECT n, m FROM r").await,
        vec![vec![ScalarValue::Int64(10), ScalarValue::Int32(20)]],
        "a refused change leaves the table as it was"
    );

    exec_ddl(&server, &mut session, "CREATE INDEX r_m_ix ON r (m)")
        .await
        .expect("a lake index on the column");
    let refused = exec_ddl(
        &server,
        &mut session,
        "ALTER TABLE r ALTER COLUMN m TYPE BIGINT",
    )
    .await
    .expect_err("an indexed column is not widened under its index");
    assert!(refused.contains("drop the index first"), "{refused}");

    exec_ddl(&server, &mut session, "DROP INDEX r_m_ix")
        .await
        .expect("drop the index");
    exec_ddl(
        &server,
        &mut session,
        "ALTER TABLE r ALTER COLUMN m TYPE BIGINT",
    )
    .await
    .expect("widened once the index is gone");
    exec_dml(&server, "INSERT INTO r VALUES (2, 11, 3000000000, 'y')").await;
    assert_eq!(
        query_values(&server, "SELECT m FROM r ORDER BY id").await,
        vec![
            vec![ScalarValue::Int64(20)],
            vec![ScalarValue::Int64(3_000_000_000)]
        ]
    );
}

/// An added column whose constraint or default the old rows cannot satisfy
/// is refused instead of serving values the declaration contradicts.
#[tokio::test]
async fn test_lake_add_column_not_null_or_default_is_refused() {
    let (server, _schema, _tmp) = create_test_server().await;
    let mut session = new_session();
    exec_ddl(
        &server,
        &mut session,
        "CREATE TABLE lk (id BIGINT NOT NULL) USING ZYRONLAKE",
    )
    .await
    .expect("create");
    exec_dml(&server, "INSERT INTO lk VALUES (1)").await;

    let result = exec_ddl(
        &server,
        &mut session,
        "ALTER TABLE lk ADD COLUMN v BIGINT NOT NULL",
    )
    .await;
    assert!(
        result.is_err(),
        "NOT NULL on existing rows with no backfill must be refused"
    );

    let result = exec_ddl(
        &server,
        &mut session,
        "ALTER TABLE lk ADD COLUMN w BIGINT DEFAULT 7",
    )
    .await;
    assert!(
        result.is_err(),
        "a DEFAULT the old rows will not serve must be refused"
    );
}
