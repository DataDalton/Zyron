//! ORDER BY under a LIMIT keeps only the rows the limit can use.
//!
//! Run: cargo test -p zyron-wire --test top_n_test
//!
//! The planner hands the cap to the sort, and the sort then reads its input
//! holding about a batch beyond the cap rather than every row. The answer
//! is the rows a full sort would put first, so every test here takes its
//! expectation from the unlimited query and asserts the limited one is a
//! prefix of it. Scores are a permutation of the ids, so no two rows tie on
//! score and the prefix is unambiguous.

mod common;

use std::sync::Arc;

use common::{create_test_server, exec_ddl, exec_dml, new_session, query_values};
use zyron_executor::column::ScalarValue;
use zyron_planner::physical::PhysicalPlan;
use zyron_wire::connection::ServerState;

/// Rows in the table, spread over enough inserts for the sort to see
/// several batches and cut its buffer more than once
const ROWS: i64 = 5_000;
const ROWS_PER_INSERT: i64 = 500;
/// Coprime with ROWS, so score is a permutation of id
const STRIDE: i64 = 7_919;

async fn events_table() -> (Arc<ServerState>, tempfile::TempDir) {
    let (server, _schema_id, tmp) = create_test_server().await;
    let mut session = new_session();
    exec_ddl(
        &server,
        &mut session,
        "CREATE TABLE events (id BIGINT NOT NULL, score BIGINT, bucket BIGINT)",
    )
    .await
    .expect("create the table");
    for chunk in 0..(ROWS / ROWS_PER_INSERT) {
        let rows: Vec<String> = (0..ROWS_PER_INSERT)
            .map(|r| {
                let id = chunk * ROWS_PER_INSERT + r;
                // Every hundredth score is NULL, so an ordering has null
                // rows to place at either end
                if id % 100 == 0 {
                    format!("({id}, NULL, {})", id % 3)
                } else {
                    format!("({id}, {}, {})", (id * STRIDE) % ROWS, id % 3)
                }
            })
            .collect();
        exec_dml(
            &server,
            &format!("INSERT INTO events VALUES {}", rows.join(", ")),
        )
        .await;
    }
    (server, tmp)
}

async fn rows_of(server: &Arc<ServerState>, sql: &str) -> Vec<String> {
    query_values(server, sql)
        .await
        .into_iter()
        .map(|row| format!("{row:?}"))
        .collect()
}

async fn plan_of(server: &Arc<ServerState>, sql: &str) -> PhysicalPlan {
    let stmt = zyron_parser::parse(sql)
        .expect("parse")
        .into_iter()
        .next()
        .expect("one statement");
    zyron_planner::plan(
        &server.catalog,
        zyron_catalog::DatabaseId(1),
        vec!["zyron_test".to_string()],
        stmt,
        None,
    )
    .await
    .expect("plan")
}

/// The limit carried by the sort in a plan, reached through the
/// single-child nodes above it. None means the plan has no sort
fn sort_limit(plan: &PhysicalPlan) -> Option<Option<u64>> {
    let mut node = plan;
    loop {
        match node {
            PhysicalPlan::Sort { limit, .. } => return Some(*limit),
            PhysicalPlan::Project { child, .. }
            | PhysicalPlan::Filter { child, .. }
            | PhysicalPlan::Limit { child, .. } => node = child.as_ref(),
            _ => return None,
        }
    }
}

/// The limit and offset of the Limit node at the top of a plan
fn top_limit(plan: &PhysicalPlan) -> Option<(Option<u64>, Option<u64>)> {
    match plan {
        PhysicalPlan::Limit { limit, offset, .. } => Some((*limit, *offset)),
        _ => None,
    }
}

#[tokio::test]
async fn the_plan_hands_the_cap_to_the_sort_and_keeps_the_limit_above_it() {
    let (server, _tmp) = events_table().await;

    let plan = plan_of(
        &server,
        "SELECT id, score FROM events ORDER BY score LIMIT 5 OFFSET 3",
    )
    .await;
    assert_eq!(
        sort_limit(&plan),
        Some(Some(8)),
        "the sort keeps offset plus limit rows"
    );
    assert_eq!(
        top_limit(&plan),
        Some((Some(5), Some(3))),
        "the offset is still applied above the sort"
    );

    let unlimited = plan_of(&server, "SELECT id, score FROM events ORDER BY score").await;
    assert_eq!(
        sort_limit(&unlimited),
        Some(None),
        "a sort with no limit above it keeps every row"
    );

    // An offset alone caps nothing, every row past it is wanted
    let offset_only = plan_of(
        &server,
        "SELECT id, score FROM events ORDER BY score OFFSET 3",
    )
    .await;
    assert_eq!(sort_limit(&offset_only), Some(None));
}

#[tokio::test]
async fn a_limit_is_a_prefix_of_the_full_ordering_in_either_direction() {
    let (server, _tmp) = events_table().await;

    let full = rows_of(&server, "SELECT id, score FROM events ORDER BY score").await;
    assert_eq!(full.len(), ROWS as usize);
    let limited = rows_of(
        &server,
        "SELECT id, score FROM events ORDER BY score LIMIT 5",
    )
    .await;
    assert_eq!(limited, full[..5]);

    let full_desc = rows_of(&server, "SELECT id, score FROM events ORDER BY score DESC").await;
    let limited_desc = rows_of(
        &server,
        "SELECT id, score FROM events ORDER BY score DESC LIMIT 7",
    )
    .await;
    assert_eq!(limited_desc, full_desc[..7]);
}

#[tokio::test]
async fn an_offset_shifts_the_prefix() {
    let (server, _tmp) = events_table().await;
    let full = rows_of(&server, "SELECT id, score FROM events ORDER BY score").await;
    let window = rows_of(
        &server,
        "SELECT id, score FROM events ORDER BY score LIMIT 5 OFFSET 3",
    )
    .await;
    assert_eq!(window, full[3..8]);
}

/// The sort holds later rows against the row at its limit, and a null key
/// sits at whichever end the ordering puts it, so both placements are
/// checked against the full ordering
#[tokio::test]
async fn null_keys_land_where_the_ordering_puts_them() {
    let (server, _tmp) = events_table().await;

    let full = rows_of(
        &server,
        "SELECT id, score FROM events ORDER BY score DESC NULLS FIRST, id",
    )
    .await;
    let limited = rows_of(
        &server,
        "SELECT id, score FROM events ORDER BY score DESC NULLS FIRST, id LIMIT 3",
    )
    .await;
    assert_eq!(limited, full[..3]);
    assert!(
        limited.iter().all(|row| row.contains("Null")),
        "nulls first puts the null rows inside the limit: {limited:?}"
    );

    // Fifty scores are null and sort last, so a limit ten past the values
    // ends with ten null rows
    let full_last = rows_of(
        &server,
        "SELECT id, score FROM events ORDER BY score NULLS LAST, id",
    )
    .await;
    let past_values = (ROWS - ROWS / 100 + 10) as usize;
    let limited_last = rows_of(
        &server,
        &format!("SELECT id, score FROM events ORDER BY score NULLS LAST, id LIMIT {past_values}"),
    )
    .await;
    assert_eq!(limited_last, full_last[..past_values]);
    assert_eq!(
        limited_last
            .iter()
            .filter(|row| row.contains("Null"))
            .count(),
        10
    );
}

#[tokio::test]
async fn an_expression_key_with_a_second_key_keeps_the_same_prefix() {
    let (server, _tmp) = events_table().await;
    let full = rows_of(
        &server,
        "SELECT id, score FROM events ORDER BY score % 7, id",
    )
    .await;
    let limited = rows_of(
        &server,
        "SELECT id, score FROM events ORDER BY score % 7, id LIMIT 10",
    )
    .await;
    assert_eq!(limited, full[..10]);
}

#[tokio::test]
async fn a_limit_beyond_the_rows_returns_them_all() {
    let (server, _tmp) = events_table().await;
    let full = rows_of(&server, "SELECT id, score FROM events ORDER BY score").await;
    let limited = rows_of(
        &server,
        "SELECT id, score FROM events ORDER BY score LIMIT 10000",
    )
    .await;
    assert_eq!(limited, full);
}

/// A key most rows share: once the sort holds its limit's worth of the
/// smallest value, every later row ties with the bound and is dropped,
/// which must still leave a full limit of rows
#[tokio::test]
async fn ties_at_the_bound_still_fill_the_limit() {
    let (server, _tmp) = events_table().await;
    let limited = query_values(&server, "SELECT bucket FROM events ORDER BY bucket LIMIT 3").await;
    assert_eq!(limited.len(), 3);
    assert!(
        limited.iter().all(|row| row[0] == ScalarValue::Int64(0)),
        "{limited:?}"
    );
}

/// The sort above a grouped aggregate takes the cap too
#[tokio::test]
async fn a_grouped_query_under_a_limit() {
    let (server, _tmp) = events_table().await;
    let plan = plan_of(
        &server,
        "SELECT bucket, COUNT(*) FROM events GROUP BY bucket ORDER BY bucket DESC LIMIT 2",
    )
    .await;
    assert_eq!(sort_limit(&plan), Some(Some(2)));

    let full = rows_of(
        &server,
        "SELECT bucket, COUNT(*) FROM events GROUP BY bucket ORDER BY bucket DESC",
    )
    .await;
    let limited = rows_of(
        &server,
        "SELECT bucket, COUNT(*) FROM events GROUP BY bucket ORDER BY bucket DESC LIMIT 2",
    )
    .await;
    assert_eq!(limited, full[..2]);
}
