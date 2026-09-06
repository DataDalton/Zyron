//! Aggregates a lake table answers out of its manifest.
//!
//! Run: cargo test -p zyron-wire --test lake_metadata_aggregate_test
//!
//! A manifest records, per file and per column, the bounds and null count
//! that settle MIN, MAX and both counts, and the exact total that settles
//! SUM. An ungrouped aggregate with no predicate is therefore a fold over
//! statistics that opens no data file, and these tests hold it to two
//! things: the plan says so, the byte counters agree, and the answer is
//! the one a scan of the same rows gives.
//!
//! The interesting cases are the ones where the statistics do not
//! describe what is live. A file under a delete predicate still counts the
//! deleted rows, and that file has to be read instead of trusted, while
//! its neighbours stay on the fast path.

use std::sync::Arc;
use std::sync::atomic::Ordering;

use zyron_wire::connection::ServerState;

mod common;
use common::{create_test_server, exec_ddl, exec_dml, new_session, query_values, render_plan};

/// Bytes of table data read while answering one query, which is zero for
/// an aggregate settled from the manifest
async fn bytes_read_answering(server: &Arc<ServerState>, table: &str, sql: &str) -> u64 {
    let id = server
        .catalog
        .list_all_tables()
        .into_iter()
        .find(|t| t.name == table)
        .expect("table")
        .id
        .0;
    let counter = || {
        server
            .table_io_stats
            .get_or_create(id)
            .bytes_read
            .load(Ordering::Relaxed)
    };
    let before = counter();
    let _ = query_values(server, sql).await;
    counter() - before
}

/// One scalar result rendered, so two access paths can be compared as
/// values without the test naming a variant
async fn scalar(server: &Arc<ServerState>, sql: &str) -> String {
    let rows = query_values(server, sql).await;
    assert_eq!(rows.len(), 1, "{sql} returned {} rows", rows.len());
    format!("{:?}", rows[0].first().expect("one column"))
}

/// Four data files of five rows each, one file per insert, with a null
/// amount in the middle file and an all-null column beside them
async fn ledger_of_four_files() -> (Arc<ServerState>, tempfile::TempDir) {
    let (server, _schema_id, tmp) = create_test_server().await;
    let mut session = new_session();
    exec_ddl(
        &server,
        &mut session,
        "CREATE TABLE ledger (id BIGINT NOT NULL, region BIGINT, amount BIGINT, \
         label TEXT, unused BIGINT) USING ZYRONLAKE",
    )
    .await
    .expect("create the lake table");
    for file in 0..4i64 {
        let rows: Vec<String> = (0..5i64)
            .map(|r| {
                let id = file * 5 + r;
                // One null amount, in a file that also holds real values,
                // so a file is never all null and never wholly present
                let amount = if id == 7 {
                    "NULL".to_string()
                } else {
                    (id * 3 - 10).to_string()
                };
                format!("({id}, {}, {amount}, 'row{id}', NULL)", id % 3)
            })
            .collect();
        exec_dml(
            &server,
            &format!("INSERT INTO ledger VALUES {}", rows.join(", ")),
        )
        .await;
    }
    (server, tmp)
}

#[tokio::test]
async fn an_unfiltered_sum_is_answered_without_opening_a_file() {
    let (server, _tmp) = ledger_of_four_files().await;

    let plan = render_plan(&server, "SELECT SUM(amount) FROM ledger").await;
    assert!(
        plan.contains("LakeMetadataAggregate") && plan.contains("manifest-statistics"),
        "an unfiltered SUM should be planned off the manifest, plan was:\n{plan}"
    );

    // The mechanism, not the answer: the byte counter counts payload
    // reads, so a fold over statistics leaves it where it was
    let bytes = bytes_read_answering(&server, "ledger", "SELECT SUM(amount) FROM ledger").await;
    assert_eq!(
        bytes, 0,
        "a manifest aggregate read {bytes} bytes of table data"
    );
}

#[tokio::test]
async fn the_manifest_answer_is_the_answer_a_scan_gives() {
    let (server, _tmp) = ledger_of_four_files().await;

    // Every id is non-negative, so this predicate keeps every row while
    // taking the aggregate off the manifest and onto a scan
    for (metadata, scanned) in [
        (
            "SELECT SUM(amount) FROM ledger",
            "SELECT SUM(amount) FROM ledger WHERE id >= 0",
        ),
        (
            "SELECT COUNT(*) FROM ledger",
            "SELECT COUNT(*) FROM ledger WHERE id >= 0",
        ),
        (
            "SELECT COUNT(amount) FROM ledger",
            "SELECT COUNT(amount) FROM ledger WHERE id >= 0",
        ),
        (
            "SELECT MIN(amount) FROM ledger",
            "SELECT MIN(amount) FROM ledger WHERE id >= 0",
        ),
        (
            "SELECT MAX(amount) FROM ledger",
            "SELECT MAX(amount) FROM ledger WHERE id >= 0",
        ),
        (
            "SELECT MIN(label) FROM ledger",
            "SELECT MIN(label) FROM ledger WHERE id >= 0",
        ),
        (
            "SELECT MAX(label) FROM ledger",
            "SELECT MAX(label) FROM ledger WHERE id >= 0",
        ),
    ] {
        let from_manifest = scalar(&server, metadata).await;
        let from_scan = scalar(&server, scanned).await;
        assert_eq!(
            from_manifest, from_scan,
            "{metadata} answered {from_manifest} where a scan answered {from_scan}"
        );
    }
}

#[tokio::test]
async fn a_null_cell_is_left_out_and_an_all_null_column_sums_to_null() {
    let (server, _tmp) = ledger_of_four_files().await;

    // Row 7 is the null amount. Nineteen of twenty rows hold a value
    assert_eq!(
        scalar(&server, "SELECT COUNT(*) FROM ledger").await,
        "Int64(20)"
    );
    assert_eq!(
        scalar(&server, "SELECT COUNT(amount) FROM ledger").await,
        "Int64(19)"
    );
    let expected: i64 = (0..20).filter(|id| *id != 7).map(|id| id * 3 - 10).sum();
    assert_eq!(
        scalar(&server, "SELECT SUM(amount) FROM ledger").await,
        format!("Int64({expected})")
    );

    // A column with no value anywhere sums to NULL, not to zero, which is
    // what makes the accumulator's "any value at all" flag load bearing
    assert_eq!(
        scalar(&server, "SELECT SUM(unused) FROM ledger").await,
        "Null"
    );
    assert_eq!(
        scalar(&server, "SELECT MIN(unused) FROM ledger").await,
        "Null"
    );
    assert_eq!(
        scalar(&server, "SELECT COUNT(unused) FROM ledger").await,
        "Int64(0)"
    );
}

#[tokio::test]
async fn a_file_under_a_delete_predicate_is_read_rather_than_trusted() {
    let (server, _tmp) = ledger_of_four_files().await;
    let before = scalar(&server, "SELECT SUM(amount) FROM ledger").await;

    // Two rows out of the first file. Its statistics still count them, so
    // that file has to be scanned while the other three stay on the
    // manifest
    exec_dml(&server, "DELETE FROM ledger WHERE id IN (1, 2)").await;

    let after = scalar(&server, "SELECT SUM(amount) FROM ledger").await;
    assert_ne!(
        before, after,
        "the delete left the sum unchanged, so the deleted rows were still counted"
    );
    let expected: i64 = (0..20)
        .filter(|id| *id != 7 && *id != 1 && *id != 2)
        .map(|id| id * 3 - 10)
        .sum();
    assert_eq!(after, format!("Int64({expected})"));
    assert_eq!(
        after,
        scalar(&server, "SELECT SUM(amount) FROM ledger WHERE id >= 0").await,
        "the mixed path disagreed with a scan of the same rows"
    );
    // A count is the aggregate most tempting to take straight off the
    // manifest's row count, and that number still holds the deleted rows
    assert_eq!(
        scalar(&server, "SELECT COUNT(*) FROM ledger").await,
        "Int64(18)"
    );
}

#[tokio::test]
async fn a_predicate_leaves_the_aggregate_on_the_scan() {
    let (server, _tmp) = ledger_of_four_files().await;
    let plan = render_plan(&server, "SELECT SUM(amount) FROM ledger WHERE region = 1").await;
    assert!(
        !plan.contains("LakeMetadataAggregate"),
        "a filtered aggregate must not be answered from whole file statistics, plan was:\n{plan}"
    );
    assert!(
        plan.contains("LakeScan"),
        "a filtered aggregate should still scan, plan was:\n{plan}"
    );
}

#[tokio::test]
async fn a_grouped_or_distinct_aggregate_stays_on_the_scan() {
    let (server, _tmp) = ledger_of_four_files().await;
    for sql in [
        "SELECT region, SUM(amount) FROM ledger GROUP BY region",
        "SELECT SUM(DISTINCT amount) FROM ledger",
        "SELECT AVG(amount) FROM ledger",
    ] {
        let plan = render_plan(&server, sql).await;
        assert!(
            !plan.contains("LakeMetadataAggregate"),
            "{sql} is not answerable from whole file statistics, plan was:\n{plan}"
        );
    }
}

#[tokio::test]
async fn a_sum_over_a_float_column_is_not_pushed_to_the_manifest() {
    let (server, _schema_id, _tmp) = create_test_server().await;
    let mut session = new_session();
    exec_ddl(
        &server,
        &mut session,
        "CREATE TABLE prices (id BIGINT NOT NULL, cost DOUBLE PRECISION) USING ZYRONLAKE",
    )
    .await
    .expect("create the lake table");
    exec_dml(
        &server,
        "INSERT INTO prices VALUES (1, 0.1), (2, 0.2), (3, 0.3)",
    )
    .await;

    // Folding per file and then across files associates differently from
    // folding row by row, and the two can round apart, so a float sum is
    // left to the scan that already defines the answer
    let plan = render_plan(&server, "SELECT SUM(cost) FROM prices").await;
    assert!(
        !plan.contains("LakeMetadataAggregate"),
        "a float SUM must not be answered from per file totals, plan was:\n{plan}"
    );

    // MIN and MAX over the same column are still exact, so they push down
    let plan = render_plan(&server, "SELECT MIN(cost), MAX(cost) FROM prices").await;
    assert!(
        plan.contains("LakeMetadataAggregate"),
        "a float MIN/MAX is an exact extremum and should push down, plan was:\n{plan}"
    );
}

#[tokio::test]
async fn time_travel_sums_the_version_it_names() {
    let (server, _tmp) = ledger_of_four_files().await;
    let whole = scalar(&server, "SELECT SUM(amount) FROM ledger").await;

    // Version 2 is the manifest the first insert published, so it lists
    // one file and its five rows
    assert_eq!(
        scalar(&server, "SELECT COUNT(*) FROM ledger VERSION AS OF 2").await,
        "Int64(5)"
    );
    let early = scalar(&server, "SELECT SUM(amount) FROM ledger VERSION AS OF 2").await;
    let expected: i64 = (0..5).map(|id| id * 3 - 10).sum();
    assert_eq!(early, format!("Int64({expected})"));
    assert_ne!(
        whole, early,
        "time travel read the current statistics rather than the named version's"
    );

    // The same version through a scan, so the statistics a version hands
    // back are the statistics of the files that version lists
    assert_eq!(
        early,
        scalar(
            &server,
            "SELECT SUM(amount) FROM ledger VERSION AS OF 2 WHERE id >= 0"
        )
        .await
    );
}
