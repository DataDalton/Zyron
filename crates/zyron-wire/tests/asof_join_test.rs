//! ASOF JOIN, through the engine.
//!
//! Run: cargo test -p zyron-wire --test asof_join_test -- --nocapture

mod common;

use common::{exec_ddl, exec_dml, new_session, query_error, query_values, render_plan};
use zyron_executor::column::ScalarValue;
use zyron_parser::{parse, statement_to_sql};

fn number(value: &ScalarValue) -> Option<i64> {
    match value {
        ScalarValue::Int8(v) => Some(*v as i64),
        ScalarValue::Int16(v) => Some(*v as i64),
        ScalarValue::Int32(v) => Some(*v as i64),
        ScalarValue::Int64(v) => Some(*v),
        _ => None,
    }
}

fn column(rows: &[Vec<ScalarValue>], at: usize) -> Vec<i64> {
    rows.iter()
        .map(|r| number(&r[at]).unwrap_or(i64::MIN))
        .collect()
}

/// Trades and quotes on two symbols, with the quote timestamps deliberately
/// off the trade timestamps so a nearest match is a real choice.
async fn market_server() -> (
    std::sync::Arc<zyron_wire::connection::ServerState>,
    tempfile::TempDir,
) {
    let (server, _schema, tmp) = common::create_test_server().await;
    let mut session = new_session();
    exec_ddl(
        &server,
        &mut session,
        "CREATE TABLE trades (symbol TEXT, ts BIGINT, price INT)",
    )
    .await
    .expect("create trades");
    exec_ddl(
        &server,
        &mut session,
        "CREATE TABLE quotes (symbol TEXT, ts BIGINT, bid INT)",
    )
    .await
    .expect("create quotes");
    exec_dml(
        &server,
        "INSERT INTO trades VALUES \
         ('AAA', 100, 1), ('AAA', 200, 2), ('AAA', 300, 3), \
         ('BBB', 150, 4), ('BBB', 250, 5)",
    )
    .await;
    exec_dml(
        &server,
        "INSERT INTO quotes VALUES \
         ('AAA', 90, 10), ('AAA', 190, 20), \
         ('BBB', 140, 30), ('BBB', 260, 40)",
    )
    .await;
    (server, tmp)
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn asof_join_returns_the_latest_quote_at_or_before_each_trade() {
    let (server, _tmp) = market_server().await;
    let rows = query_values(
        &server,
        "SELECT t.ts, q.bid FROM trades AS t \
         ASOF JOIN quotes AS q MATCH_CONDITION (t.ts >= q.ts) ON t.symbol = q.symbol \
         ORDER BY t.ts",
    )
    .await;
    // AAA 100 -> quote 90 (bid 10), AAA 200 -> 190 (20), AAA 300 -> 190 (20)
    // BBB 150 -> 140 (30), BBB 250 -> 140 (30). BBB's 260 quote is later
    // than both BBB trades, so it never matches
    assert_eq!(column(&rows, 0), vec![100, 150, 200, 250, 300]);
    assert_eq!(column(&rows, 1), vec![10, 30, 20, 30, 20]);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn the_result_matches_a_correlated_subquery_over_the_same_data() {
    let (server, _tmp) = market_server().await;
    let by_join = query_values(
        &server,
        "SELECT t.ts, q.bid FROM trades AS t \
         ASOF JOIN quotes AS q MATCH_CONDITION (t.ts >= q.ts) ON t.symbol = q.symbol \
         ORDER BY t.ts",
    )
    .await;
    let by_subquery = query_values(
        &server,
        "SELECT t.ts, (SELECT q.bid FROM quotes AS q \
                       WHERE q.symbol = t.symbol AND q.ts <= t.ts \
                       ORDER BY q.ts DESC LIMIT 1) \
         FROM trades AS t ORDER BY t.ts",
    )
    .await;
    let expected: Vec<(i64, i64)> = by_subquery
        .iter()
        .filter(|r| number(&r[1]).is_some())
        .map(|r| (number(&r[0]).unwrap_or(0), number(&r[1]).unwrap_or(0)))
        .collect();
    let actual: Vec<(i64, i64)> = by_join
        .iter()
        .map(|r| (number(&r[0]).unwrap_or(0), number(&r[1]).unwrap_or(0)))
        .collect();
    assert_eq!(
        actual, expected,
        "the merge and a correlated lookup have to give the same answer"
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_trade_with_no_earlier_quote_is_dropped_by_the_inner_form_and_kept_by_the_left_form() {
    let (server, _schema, _tmp) = common::create_test_server().await;
    let mut session = new_session();
    exec_ddl(
        &server,
        &mut session,
        "CREATE TABLE trades (symbol TEXT, ts BIGINT)",
    )
    .await
    .expect("create trades");
    exec_ddl(
        &server,
        &mut session,
        "CREATE TABLE quotes (symbol TEXT, ts BIGINT, bid INT)",
    )
    .await
    .expect("create quotes");
    // The first trade is before every quote on its symbol
    exec_dml(
        &server,
        "INSERT INTO trades VALUES ('AAA', 50), ('AAA', 200)",
    )
    .await;
    exec_dml(&server, "INSERT INTO quotes VALUES ('AAA', 100, 7)").await;

    let inner = query_values(
        &server,
        "SELECT t.ts FROM trades AS t ASOF JOIN quotes AS q \
         MATCH_CONDITION (t.ts >= q.ts) ON t.symbol = q.symbol ORDER BY t.ts",
    )
    .await;
    assert_eq!(
        column(&inner, 0),
        vec![200],
        "the unmatched trade is dropped"
    );

    let left = query_values(
        &server,
        "SELECT t.ts, q.bid FROM trades AS t ASOF LEFT JOIN quotes AS q \
         MATCH_CONDITION (t.ts >= q.ts) ON t.symbol = q.symbol ORDER BY t.ts",
    )
    .await;
    assert_eq!(column(&left, 0), vec![50, 200], "the LEFT form keeps it");
    assert_eq!(
        column(&left, 1),
        vec![i64::MIN, 7],
        "an unmatched left row carries NULL on the right"
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_tolerance_leaves_a_trade_unmatched_whose_nearest_quote_is_older() {
    let (server, _schema, _tmp) = common::create_test_server().await;
    let mut session = new_session();
    exec_ddl(
        &server,
        &mut session,
        "CREATE TABLE trades (symbol TEXT, ts BIGINT)",
    )
    .await
    .expect("create trades");
    exec_ddl(
        &server,
        &mut session,
        "CREATE TABLE quotes (symbol TEXT, ts BIGINT, bid INT)",
    )
    .await
    .expect("create quotes");
    // Timestamps are microseconds: the second trade is ten minutes after the
    // quote, well outside a five minute reach
    let quote_at = 1_000_000_000i64;
    exec_dml(
        &server,
        &format!(
            "INSERT INTO trades VALUES ('AAA', {}), ('AAA', {})",
            quote_at + 60_000_000,
            quote_at + 600_000_000
        ),
    )
    .await;
    exec_dml(
        &server,
        &format!("INSERT INTO quotes VALUES ('AAA', {quote_at}, 7)"),
    )
    .await;

    let rows = query_values(
        &server,
        "SELECT t.ts, q.bid FROM trades AS t ASOF LEFT JOIN quotes AS q \
         MATCH_CONDITION (t.ts >= q.ts AND t.ts - q.ts <= INTERVAL '5 minutes') \
         ON t.symbol = q.symbol ORDER BY t.ts",
    )
    .await;
    assert_eq!(rows.len(), 2);
    assert_eq!(
        number(&rows[0][1]),
        Some(7),
        "one minute back is inside the five minute reach"
    );
    assert!(
        matches!(rows[1][1], ScalarValue::Null),
        "ten minutes back is outside it, got {:?}",
        rows[1][1]
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn the_four_directions_pick_the_value_each_one_names() {
    let (server, _schema, _tmp) = common::create_test_server().await;
    let mut session = new_session();
    exec_ddl(&server, &mut session, "CREATE TABLE l (ts BIGINT)")
        .await
        .expect("create l");
    exec_ddl(&server, &mut session, "CREATE TABLE r (ts BIGINT, v INT)")
        .await
        .expect("create r");
    // The left value sits exactly on one right value, with one on each side
    exec_dml(&server, "INSERT INTO l VALUES (100)").await;
    exec_dml(&server, "INSERT INTO r VALUES (90, 1), (100, 2), (110, 3)").await;

    let read = |op: &'static str| {
        let server = server.clone();
        async move {
            let rows = query_values(
                &server,
                &format!("SELECT r.v FROM l ASOF LEFT JOIN r MATCH_CONDITION (l.ts {op} r.ts)"),
            )
            .await;
            number(&rows[0][0])
        }
    };

    assert_eq!(
        read(">=").await,
        Some(2),
        ">= takes the greatest right value not above the left one, equality included"
    );
    assert_eq!(
        read(">").await,
        Some(1),
        "> excludes equality, so it falls back to 90"
    );
    assert_eq!(
        read("<=").await,
        Some(2),
        "<= takes the least right value not below the left one, equality included"
    );
    assert_eq!(
        read("<").await,
        Some(3),
        "< excludes equality, so it reaches 110"
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn an_asof_join_with_no_on_clause_treats_the_whole_input_as_one_group() {
    let (server, _schema, _tmp) = common::create_test_server().await;
    let mut session = new_session();
    exec_ddl(&server, &mut session, "CREATE TABLE l (ts BIGINT)")
        .await
        .expect("create l");
    exec_ddl(&server, &mut session, "CREATE TABLE r (ts BIGINT, v INT)")
        .await
        .expect("create r");
    exec_dml(&server, "INSERT INTO l VALUES (100), (200)").await;
    exec_dml(&server, "INSERT INTO r VALUES (50, 1), (150, 2)").await;

    let rows = query_values(
        &server,
        "SELECT l.ts, r.v FROM l ASOF JOIN r MATCH_CONDITION (l.ts >= r.ts) ORDER BY l.ts",
    )
    .await;
    assert_eq!(column(&rows, 0), vec![100, 200]);
    assert_eq!(column(&rows, 1), vec![1, 2]);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn explain_names_the_join_the_direction_and_each_side_sort() {
    let (server, _tmp) = market_server().await;
    let plan = render_plan(
        &server,
        "SELECT t.ts, q.bid FROM trades AS t ASOF JOIN quotes AS q \
         MATCH_CONDITION (t.ts >= q.ts) ON t.symbol = q.symbol",
    )
    .await;
    assert!(plan.contains("AsofJoin"), "EXPLAIN names the join\n{plan}");
    assert!(
        plan.contains("at or before"),
        "EXPLAIN names the match direction\n{plan}"
    );
    assert!(
        plan.contains("left_sort") && plan.contains("right_sort"),
        "EXPLAIN says what happened to each side's sort\n{plan}"
    );
    assert!(
        plan.contains("sorted"),
        "an unclustered heap table shows the sort\n{plan}"
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn there_is_no_right_or_full_asof_join() {
    let (server, _tmp) = market_server().await;
    // RIGHT and FULL do not parse as ASOF forms at all, so the parser refuses
    // them before the binder is reached
    for sql in [
        "SELECT * FROM trades ASOF RIGHT JOIN quotes MATCH_CONDITION (trades.ts >= quotes.ts)",
        "SELECT * FROM trades ASOF FULL JOIN quotes MATCH_CONDITION (trades.ts >= quotes.ts)",
    ] {
        assert!(
            parse(sql).is_err(),
            "`{sql}` should not parse; ASOF has an inner form and a LEFT form only"
        );
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn an_inequality_in_the_on_clause_is_refused() {
    let (server, _tmp) = market_server().await;
    let error = query_error(
        &server,
        "SELECT * FROM trades AS t ASOF JOIN quotes AS q \
         MATCH_CONDITION (t.ts >= q.ts) ON t.price > q.bid",
    )
    .await;
    assert!(
        error.contains("equalities only"),
        "the refusal says where an inequality belongs, got {error}"
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_match_condition_over_two_columns_of_one_side_is_refused() {
    let (server, _tmp) = market_server().await;
    let error = query_error(
        &server,
        "SELECT * FROM trades AS t ASOF JOIN quotes AS q MATCH_CONDITION (t.ts >= t.price)",
    )
    .await;
    assert!(
        error.contains("left relation") && error.contains("right relation"),
        "the refusal says the condition spans both sides, got {error}"
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn the_unparser_writes_an_asof_join_back_as_written() {
    for sql in [
        "SELECT * FROM trades ASOF JOIN quotes MATCH_CONDITION (trades.ts >= quotes.ts) ON trades.symbol = quotes.symbol",
        "SELECT * FROM trades ASOF LEFT JOIN quotes MATCH_CONDITION (trades.ts < quotes.ts)",
    ] {
        let first = parse(sql).expect("parses");
        let rendered = statement_to_sql(&first[0]).expect("renders");
        assert!(
            rendered.contains("ASOF") && rendered.contains("MATCH_CONDITION"),
            "stored SQL keeps the construct as written, got {rendered}"
        );
        let second = parse(&rendered).expect("the rendering parses");
        assert_eq!(second, first);
    }
}
