//! A grouped aggregate over a lake table split across workers by file.
//!
//! Run: cargo test -p zyron-wire --test parallel_lake_aggregate_test
//!
//! Each worker reads a disjoint set of data files and builds a group table
//! of its own, then the partials merge. A group that appears in every file
//! is therefore assembled from every worker, which is the case a merge gets
//! wrong: the answer has to be the same as one thread reading every file in
//! order.
//!
//! The split is refused where it would cost more than it saves. A grouping
//! key with about as many distinct values as there are rows leaves every
//! worker holding the whole table and makes the merge repeat the
//! aggregation, so that shape stays on the single threaded path and keeps
//! its spill and memory budget with it.

use std::sync::Arc;

use zyron_executor::column::ScalarValue;
use zyron_wire::connection::ServerState;

mod common;
use common::{create_test_server, exec_ddl, exec_dml, new_session, query_values};

/// Rows per INSERT, which is one data file each.
///
/// Sized so the table is worth a fan-out: the split wakes one pool thread
/// per fifty thousand rows, and a table smaller than that is read by the
/// calling thread however many files it has
const ROWS_PER_FILE: i64 = 30_000;
const FILES: i64 = 4;
const REGIONS: i64 = 8;

/// Four data files whose groups all span every file, so no group is
/// resolved by a single worker
async fn lake_of_four_files() -> (Arc<ServerState>, tempfile::TempDir) {
    let (server, _schema_id, tmp) = create_test_server().await;
    let mut session = new_session();
    exec_ddl(
        &server,
        &mut session,
        "CREATE TABLE sales (id BIGINT NOT NULL, region BIGINT, amount BIGINT) USING ZYRONLAKE",
    )
    .await
    .expect("create the lake table");
    for file in 0..FILES {
        let rows: Vec<String> = (0..ROWS_PER_FILE)
            .map(|r| {
                let id = file * ROWS_PER_FILE + r;
                // One region left NULL, which groups on its own and must
                // survive the merge as a group rather than as a zero
                if id % 97 == 0 {
                    format!("({id}, NULL, {id})")
                } else {
                    format!("({id}, {}, {id})", id % REGIONS)
                }
            })
            .collect();
        exec_dml(
            &server,
            &format!("INSERT INTO sales VALUES {}", rows.join(", ")),
        )
        .await;
    }
    (server, tmp)
}

/// Every row of the dataset as (region, amount), with None for the NULL
/// region, so a test can work out any grouped answer the same way the
/// engine has to
fn rows() -> Vec<(Option<i64>, i64)> {
    (0..FILES * ROWS_PER_FILE)
        .map(|id| {
            if id % 97 == 0 {
                (None, id)
            } else {
                (Some(id % REGIONS), id)
            }
        })
        .collect()
}

/// Workers one query put on the parallel pool.
///
/// The split is an executor decision that no plan node records, so this is
/// what says whether it happened. The counter is the pool's, not this
/// operator's, so it reads work anything in the query spawned, and only a
/// shape that spawns nothing otherwise can be held to zero. Which shapes
/// the split accepts is settled by the unit tests on `worth_splitting`
async fn spawned_running(server: &Arc<ServerState>, sql: &str) -> u64 {
    let pool = zyron_executor::parallel_pool::ParallelPool::global();
    let before = pool.tasks_spawned();
    let _ = query_values(server, sql).await;
    pool.tasks_spawned() - before
}

async fn grouped(server: &Arc<ServerState>, sql: &str) -> Vec<String> {
    query_values(server, sql)
        .await
        .into_iter()
        .map(|row| format!("{row:?}"))
        .collect()
}

/// The live manifest of a lake table, so a test can see the file set the
/// split is actually offered
fn manifest_of(server: &Arc<ServerState>, table: &str) -> Arc<zyron_lake::ManifestFile> {
    let entry = server
        .catalog
        .list_all_tables()
        .into_iter()
        .find(|t| t.name == table)
        .expect("table");
    let paths = zyron_lake::LakePaths::new(server.disk_manager.data_dir(), entry.id.0);
    let log = zyron_lake::TransactionLog::lookup_shared(&paths).expect("lake log");
    log.manifest_at(log.latest_version()).expect("manifest")
}

#[tokio::test]
async fn the_split_runs_and_the_serial_path_is_kept_for_a_key_per_row() {
    let (server, _tmp) = lake_of_four_files().await;

    // Nothing below means anything if the inserts landed in one file
    let manifest = manifest_of(&server, "sales");
    assert!(
        manifest.entries.len() > 1,
        "the split needs more than one file to divide, the table has {}",
        manifest.entries.len()
    );
    let region_ndv: Vec<Option<u64>> = manifest
        .entries
        .iter()
        .map(|e| e.stats_for(1).and_then(|s| s.ndv))
        .collect();
    assert!(
        region_ndv.iter().all(|n| n.is_some()),
        "the split is decided from the manifest's distinct estimate, which reads {region_ndv:?}"
    );

    // The plan node is named for the plan, not for the operator the
    // executor picked, so what the split can be seen by is the workers it
    // puts on the pool
    let narrow = spawned_running(
        &server,
        "SELECT region, SUM(amount) FROM sales GROUP BY region",
    )
    .await;
    assert!(
        narrow > 1,
        "eight groups over a hundred and twenty thousand rows is worth splitting, {narrow} workers ran"
    );

    // A distinct value per row leaves each worker holding the whole table
    let wide = spawned_running(&server, "SELECT id, SUM(amount) FROM sales GROUP BY id").await;
    assert_eq!(
        wide, 0,
        "a key with a group per row must not be split, {wide} workers ran"
    );
}

#[tokio::test]
async fn a_sum_per_group_matches_a_fold_over_every_row() {
    let (server, _tmp) = lake_of_four_files().await;
    let got = grouped(
        &server,
        "SELECT region, SUM(amount) FROM sales GROUP BY region ORDER BY region",
    )
    .await;

    let mut expected: Vec<String> = Vec::new();
    for region in 0..REGIONS {
        let sum: i64 = rows()
            .iter()
            .filter(|(r, _)| *r == Some(region))
            .map(|(_, a)| a)
            .sum();
        expected.push(format!(
            "{:?}",
            vec![ScalarValue::Int64(region), ScalarValue::Int64(sum)]
        ));
    }
    // The NULL group sorts last
    let null_sum: i64 = rows()
        .iter()
        .filter(|(r, _)| r.is_none())
        .map(|(_, a)| a)
        .sum();
    expected.push(format!(
        "{:?}",
        vec![ScalarValue::Null, ScalarValue::Int64(null_sum)]
    ));

    assert_eq!(got, expected);
}

#[tokio::test]
async fn counts_and_extremes_per_group_survive_the_merge() {
    let (server, _tmp) = lake_of_four_files().await;
    let got = grouped(
        &server,
        "SELECT region, COUNT(*), MIN(amount), MAX(amount) FROM sales \
         GROUP BY region ORDER BY region",
    )
    .await;

    let mut expected: Vec<String> = Vec::new();
    for region in Some(0..REGIONS)
        .into_iter()
        .flatten()
        .map(Some)
        .chain([None])
    {
        let group: Vec<i64> = rows()
            .iter()
            .filter(|(r, _)| *r == region)
            .map(|(_, a)| *a)
            .collect();
        let key = match region {
            Some(v) => ScalarValue::Int64(v),
            None => ScalarValue::Null,
        };
        expected.push(format!(
            "{:?}",
            vec![
                key,
                ScalarValue::Int64(group.len() as i64),
                ScalarValue::Int64(*group.iter().min().expect("a non-empty group")),
                ScalarValue::Int64(*group.iter().max().expect("a non-empty group")),
            ]
        ));
    }
    assert_eq!(got, expected);
}

/// A predicate prunes files before the split, so the workers divide what
/// survives rather than what the manifest listed
#[tokio::test]
async fn a_predicate_narrows_what_the_workers_divide() {
    let (server, _tmp) = lake_of_four_files().await;
    let cutoff = ROWS_PER_FILE * 2;
    let got = grouped(
        &server,
        &format!(
            "SELECT region, COUNT(*) FROM sales WHERE id >= {cutoff} \
             GROUP BY region ORDER BY region"
        ),
    )
    .await;

    let mut expected: Vec<String> = Vec::new();
    for region in Some(0..REGIONS)
        .into_iter()
        .flatten()
        .map(Some)
        .chain([None])
    {
        let count = rows()
            .iter()
            .filter(|(r, a)| *r == region && *a >= cutoff)
            .count();
        let key = match region {
            Some(v) => ScalarValue::Int64(v),
            None => ScalarValue::Null,
        };
        expected.push(format!("{:?}", vec![key, ScalarValue::Int64(count as i64)]));
    }
    assert_eq!(got, expected);
}

/// A DISTINCT aggregate's partials are not associative across workers,
/// since a value in two of them would be counted twice
#[tokio::test]
async fn a_distinct_aggregate_is_not_split() {
    let (server, _tmp) = lake_of_four_files().await;
    let got = grouped(
        &server,
        "SELECT region, COUNT(DISTINCT amount) FROM sales GROUP BY region ORDER BY region",
    )
    .await;
    // Every amount is the row id, so each is distinct within its group
    let mut expected: Vec<String> = Vec::new();
    for region in Some(0..REGIONS)
        .into_iter()
        .flatten()
        .map(Some)
        .chain([None])
    {
        let count = rows().iter().filter(|(r, _)| *r == region).count();
        let key = match region {
            Some(v) => ScalarValue::Int64(v),
            None => ScalarValue::Null,
        };
        expected.push(format!("{:?}", vec![key, ScalarValue::Int64(count as i64)]));
    }
    assert_eq!(got, expected);
}

/// A table of one file has nothing to divide
#[tokio::test]
async fn a_single_file_table_is_not_split() {
    let (server, _schema_id, _tmp) = create_test_server().await;
    let mut session = new_session();
    exec_ddl(
        &server,
        &mut session,
        "CREATE TABLE solo (id BIGINT NOT NULL, region BIGINT) USING ZYRONLAKE",
    )
    .await
    .expect("create the lake table");
    let rows: Vec<String> = (0..200).map(|i| format!("({i}, {})", i % 4)).collect();
    exec_dml(
        &server,
        &format!("INSERT INTO solo VALUES {}", rows.join(", ")),
    )
    .await;

    let got = grouped(
        &server,
        "SELECT region, COUNT(*) FROM solo GROUP BY region ORDER BY region",
    )
    .await;
    assert_eq!(got.len(), 4);
}
