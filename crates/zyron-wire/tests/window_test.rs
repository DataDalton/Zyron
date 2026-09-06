//! Window functions over partitions, frames and peers.
//!
//! Run: cargo test -p zyron-wire --test window_test
//!
//! The rows arrive in neither partition nor order-key order, two rows tie
//! on the order key, one row's value is NULL and one partition key is
//! NULL, so every shape below has something to get wrong. Every answer is
//! worked out here by hand from the table, and the rows are read back in
//! id order so the window column can be checked row by row.

mod common;

use std::sync::Arc;

use common::{create_test_server, exec_ddl, exec_dml, new_session, query_values};
use zyron_executor::column::ScalarValue;
use zyron_wire::connection::ServerState;

/// id, grp, ts, v. Group 1 holds a tie on ts between ids 2 and 3 and a
/// NULL value at id 4, group 2 is plain, and ids 7 and 8 have a NULL group
const ROWS: [(i64, Option<i64>, i64, Option<i64>); 10] = [
    (1, Some(1), 1, Some(10)),
    (2, Some(1), 2, Some(20)),
    (3, Some(1), 2, Some(30)),
    (4, Some(1), 3, None),
    (5, Some(2), 1, Some(5)),
    (6, Some(2), 2, Some(15)),
    (7, None, 1, Some(7)),
    (8, None, 2, Some(8)),
    (9, Some(2), 3, Some(25)),
    (10, Some(1), 4, Some(40)),
];

/// The order the rows are inserted in, so the scan hands them over in
/// neither partition nor order-key order
const ARRIVAL: [usize; 10] = [4, 0, 8, 2, 6, 1, 9, 3, 5, 7];

async fn events() -> (Arc<ServerState>, tempfile::TempDir) {
    let (server, _schema_id, tmp) = create_test_server().await;
    let mut session = new_session();
    exec_ddl(
        &server,
        &mut session,
        "CREATE TABLE w (id BIGINT NOT NULL, grp BIGINT, ts BIGINT, v BIGINT)",
    )
    .await
    .expect("create the table");
    let sql = |o: Option<i64>| o.map_or("NULL".to_string(), |x| x.to_string());
    let values: Vec<String> = ARRIVAL
        .iter()
        .map(|&i| {
            let (id, grp, ts, v) = ROWS[i];
            format!("({id}, {}, {ts}, {})", sql(grp), sql(v))
        })
        .collect();
    exec_dml(
        &server,
        &format!("INSERT INTO w VALUES {}", values.join(", ")),
    )
    .await;
    (server, tmp)
}

/// Column `col` of every row as an integer or NULL, rows in id order
async fn ints(server: &Arc<ServerState>, sql: &str, col: usize) -> Vec<Option<i64>> {
    query_values(server, sql)
        .await
        .into_iter()
        .map(|row| match &row[col] {
            ScalarValue::Null => None,
            ScalarValue::Int64(v) => Some(*v),
            ScalarValue::Int32(v) => Some(*v as i64),
            other => panic!("expected an integer, got {other:?}"),
        })
        .collect()
}

async fn floats(server: &Arc<ServerState>, sql: &str, col: usize) -> Vec<Option<f64>> {
    query_values(server, sql)
        .await
        .into_iter()
        .map(|row| match &row[col] {
            ScalarValue::Null => None,
            ScalarValue::Float64(v) => Some(*v),
            other => panic!("expected a float, got {other:?}"),
        })
        .collect()
}

#[tokio::test]
async fn a_running_sum_shares_its_value_across_peers() {
    let (server, _tmp) = events().await;
    let got = ints(
        &server,
        "SELECT id, SUM(v) OVER (PARTITION BY grp ORDER BY ts) FROM w ORDER BY id",
        1,
    )
    .await;
    // ids 2 and 3 tie on ts, so both see the sum through the tie, and the
    // NULL value at id 4 adds nothing
    assert_eq!(
        got,
        vec![
            Some(10),
            Some(60),
            Some(60),
            Some(60),
            Some(5),
            Some(20),
            Some(7),
            Some(15),
            Some(45),
            Some(100),
        ]
    );
}

#[tokio::test]
async fn a_rows_frame_folds_one_row_at_a_time() {
    let (server, _tmp) = events().await;
    let got = ints(
        &server,
        "SELECT id, SUM(v) OVER (PARTITION BY grp ORDER BY ts, id \
         ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) FROM w ORDER BY id",
        1,
    )
    .await;
    assert_eq!(
        got,
        vec![
            Some(10),
            Some(30),
            Some(60),
            Some(60),
            Some(5),
            Some(20),
            Some(7),
            Some(15),
            Some(45),
            Some(100),
        ]
    );
}

#[tokio::test]
async fn counts_over_a_whole_partition_reach_every_row() {
    let (server, _tmp) = events().await;
    let sql = "SELECT id, COUNT(*) OVER (PARTITION BY grp), COUNT(v) OVER (PARTITION BY grp) \
               FROM w ORDER BY id";
    let all = ints(&server, sql, 1).await;
    let present = ints(&server, sql, 2).await;
    let by_id = |g1: i64, g2: i64, gn: i64| {
        vec![
            Some(g1),
            Some(g1),
            Some(g1),
            Some(g1),
            Some(g2),
            Some(g2),
            Some(gn),
            Some(gn),
            Some(g2),
            Some(g1),
        ]
    };
    assert_eq!(all, by_id(5, 3, 2));
    // COUNT(v) leaves out the NULL value at id 4
    assert_eq!(present, by_id(4, 3, 2));
}

#[tokio::test]
async fn running_avg_min_and_max_skip_the_null_value() {
    let (server, _tmp) = events().await;
    let sql = "SELECT id, AVG(v) OVER (PARTITION BY grp ORDER BY ts, id), \
               MIN(v) OVER (PARTITION BY grp ORDER BY ts, id), \
               MAX(v) OVER (PARTITION BY grp ORDER BY ts, id) FROM w ORDER BY id";
    let avg = floats(&server, sql, 1).await;
    let min = ints(&server, sql, 2).await;
    let max = ints(&server, sql, 3).await;
    assert_eq!(
        avg,
        vec![
            Some(10.0),
            Some(15.0),
            Some(20.0),
            Some(20.0),
            Some(5.0),
            Some(10.0),
            Some(7.0),
            Some(7.5),
            Some(15.0),
            Some(25.0),
        ]
    );
    assert_eq!(
        min,
        vec![
            Some(10),
            Some(10),
            Some(10),
            Some(10),
            Some(5),
            Some(5),
            Some(7),
            Some(7),
            Some(5),
            Some(10),
        ]
    );
    assert_eq!(
        max,
        vec![
            Some(10),
            Some(20),
            Some(30),
            Some(30),
            Some(5),
            Some(15),
            Some(7),
            Some(8),
            Some(25),
            Some(40),
        ]
    );
}

#[tokio::test]
async fn ranks_tie_and_row_numbers_do_not() {
    let (server, _tmp) = events().await;
    let sql = "SELECT id, RANK() OVER (PARTITION BY grp ORDER BY ts), \
               DENSE_RANK() OVER (PARTITION BY grp ORDER BY ts), \
               ROW_NUMBER() OVER (PARTITION BY grp ORDER BY ts) FROM w ORDER BY id";
    let rank = ints(&server, sql, 1).await;
    let dense = ints(&server, sql, 2).await;
    let number = ints(&server, sql, 3).await;
    assert_eq!(
        rank,
        vec![
            Some(1),
            Some(2),
            Some(2),
            Some(4),
            Some(1),
            Some(2),
            Some(1),
            Some(2),
            Some(3),
            Some(5),
        ]
    );
    assert_eq!(
        dense,
        vec![
            Some(1),
            Some(2),
            Some(2),
            Some(3),
            Some(1),
            Some(2),
            Some(1),
            Some(2),
            Some(3),
            Some(4),
        ]
    );
    // The tied pair takes row numbers 2 and 3 between them, in whichever
    // order the tie fell
    let mut tied = vec![number[1], number[2]];
    tied.sort();
    assert_eq!(tied, vec![Some(2), Some(3)]);
    assert_eq!(number[0], Some(1));
    assert_eq!(number[3], Some(4));
    assert_eq!(number[9], Some(5));
    assert_eq!(
        &number[4..9],
        &[Some(1), Some(2), Some(1), Some(2), Some(3)]
    );
}

#[tokio::test]
async fn lag_and_lead_look_along_the_partition() {
    let (server, _tmp) = events().await;
    let sql = "SELECT id, LAG(v) OVER (PARTITION BY grp ORDER BY ts, id), \
               LEAD(v) OVER (PARTITION BY grp ORDER BY ts, id) FROM w ORDER BY id";
    let lag = ints(&server, sql, 1).await;
    let lead = ints(&server, sql, 2).await;
    assert_eq!(
        lag,
        vec![
            None,
            Some(10),
            Some(20),
            Some(30),
            None,
            Some(5),
            None,
            Some(7),
            Some(15),
            None,
        ]
    );
    assert_eq!(
        lead,
        vec![
            Some(20),
            Some(30),
            None,
            Some(40),
            Some(15),
            Some(25),
            Some(8),
            None,
            None,
            None,
        ]
    );
}

#[tokio::test]
async fn first_value_and_a_framed_last_value() {
    let (server, _tmp) = events().await;
    let sql = "SELECT id, FIRST_VALUE(v) OVER (PARTITION BY grp ORDER BY ts, id), \
               LAST_VALUE(v) OVER (PARTITION BY grp ORDER BY ts, id \
               ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) FROM w ORDER BY id";
    let first = ints(&server, sql, 1).await;
    let last = ints(&server, sql, 2).await;
    assert_eq!(
        first,
        vec![
            Some(10),
            Some(10),
            Some(10),
            Some(10),
            Some(5),
            Some(5),
            Some(7),
            Some(7),
            Some(5),
            Some(10),
        ]
    );
    // The current row is the last of its frame, so each row sees its own
    // value, the NULL at id 4 included
    assert_eq!(
        last,
        vec![
            Some(10),
            Some(20),
            Some(30),
            None,
            Some(5),
            Some(15),
            Some(7),
            Some(8),
            Some(25),
            Some(40),
        ]
    );
}

#[tokio::test]
async fn two_windows_in_one_query_keep_their_own_partitions() {
    let (server, _tmp) = events().await;
    let sql = "SELECT id, SUM(v) OVER (PARTITION BY grp ORDER BY ts, id), \
               SUM(v) OVER (ORDER BY id) FROM w ORDER BY id";
    let by_group = ints(&server, sql, 1).await;
    let over_all = ints(&server, sql, 2).await;
    assert_eq!(
        by_group,
        vec![
            Some(10),
            Some(30),
            Some(60),
            Some(60),
            Some(5),
            Some(20),
            Some(7),
            Some(15),
            Some(45),
            Some(100),
        ]
    );
    assert_eq!(
        over_all,
        vec![
            Some(10),
            Some(30),
            Some(60),
            Some(60),
            Some(65),
            Some(80),
            Some(87),
            Some(95),
            Some(120),
            Some(160),
        ]
    );
}

#[tokio::test]
async fn a_partition_without_an_order_sums_whole() {
    let (server, _tmp) = events().await;
    let got = ints(
        &server,
        "SELECT id, SUM(v) OVER (PARTITION BY grp) FROM w ORDER BY id",
        1,
    )
    .await;
    assert_eq!(
        got,
        vec![
            Some(100),
            Some(100),
            Some(100),
            Some(100),
            Some(45),
            Some(45),
            Some(15),
            Some(15),
            Some(45),
            Some(100),
        ]
    );
}

#[tokio::test]
async fn no_rows_means_no_window_rows() {
    let (server, _tmp) = events().await;
    let got = query_values(&server, "SELECT id, SUM(v) OVER () FROM w WHERE id > 100").await;
    assert!(got.is_empty());
}
