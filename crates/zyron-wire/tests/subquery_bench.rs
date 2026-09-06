//! What a subquery costs against the join that answers the same question.
//!
//! Run: cargo test -p zyron-wire --test subquery_bench --release -- --nocapture
//!
//! Every measurement here is a pair. One side is the shape a person writes
//! when they mean "customers who ordered something", which is a correlated
//! EXISTS or an IN over a subquery. The other is the join a planner would
//! rewrite it into, hand written so it runs today. Both are asserted to
//! return the same answer before either is timed, so the ratio between them
//! is the cost of the shape and nothing else.
//!
//! The ratio is the measurement. An absolute number here says as much about
//! the machine as about the engine, but "the correlated form costs N times
//! the join form on identical rows and an identical answer" holds on any
//! machine, and it is what decorrelation is worth.

use std::sync::Arc;

use zyron_bench_harness::*;
use zyron_executor::column::ScalarValue;
use zyron_wire::connection::ServerState;

mod common;
use common::{create_test_server, exec_ddl, exec_dml, new_session, query_values};

/// Outer rows. A correlated subquery runs its plan once per row of this, so
/// the suite's runtime is set here and the second size below is what proves
/// that is what is happening
fn customers() -> usize {
    if measuring() { 2_000 } else { 100 }
}

/// Outer rows for the scaling pair. Doubling only this side doubles a
/// correlated form's work and leaves a join's build side untouched
fn customers_doubled() -> usize {
    customers() * 2
}

/// Inner rows. Ten per customer, so a correlated pass over the inner table
/// is real work rather than a lookup that fits in cache
fn orders_per_customer() -> usize {
    10
}

/// Customers with no order at all, which is what the anti join shapes find
const CUSTOMERS_WITHOUT_ORDERS: usize = 4;

fn reps() -> usize {
    if measuring() { 3 } else { 1 }
}

/// Serializes the suite, since each test builds its own tables and a
/// correlated shape saturates one core for the length of the query
static BENCH_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

struct Section {
    _guard: std::sync::MutexGuard<'static, ()>,
}

fn section(title: &str) -> Section {
    let guard = BENCH_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    init("subquery");
    tprintln!("");
    tprintln!("=== {} ===", title);
    Section { _guard: guard }
}

/// A customers table and an orders table pointing at it, loaded in bulk
/// statements so the load itself is not the measurement.
///
/// The last few customers get no orders, so EXISTS and NOT EXISTS both have
/// something to find and neither degenerates into "all rows" or "no rows"
async fn two_tables(suffix: &str, customer_count: usize) -> (Arc<ServerState>, tempfile::TempDir) {
    let (server, _schema_id, tmp) = create_test_server().await;
    let mut session = new_session();
    let cust = format!("cust_{suffix}");
    let ord = format!("ord_{suffix}");
    exec_ddl(
        &server,
        &mut session,
        &format!("CREATE TABLE {cust} (id BIGINT NOT NULL, region BIGINT, name TEXT)"),
    )
    .await
    .expect("create the customers table");
    exec_ddl(
        &server,
        &mut session,
        &format!("CREATE TABLE {ord} (id BIGINT NOT NULL, cust_id BIGINT, amount BIGINT)"),
    )
    .await
    .expect("create the orders table");

    for chunk in (0..customer_count).collect::<Vec<_>>().chunks(2_000) {
        let values: Vec<String> = chunk
            .iter()
            .map(|c| format!("({c}, {}, 'cust-{c}')", c % 64))
            .collect();
        exec_dml(
            &server,
            &format!("INSERT INTO {cust} VALUES {}", values.join(", ")),
        )
        .await;
    }

    // Orders skip the last few customers, so those rows are the ones an
    // anti join has to find and a semi join has to leave out
    let with_orders = customer_count.saturating_sub(CUSTOMERS_WITHOUT_ORDERS);
    let mut order_id = 0usize;
    let mut pending: Vec<String> = Vec::new();
    for c in 0..with_orders {
        for k in 0..orders_per_customer() {
            pending.push(format!("({order_id}, {c}, {})", (c * 7 + k) as i64 % 1_000));
            order_id += 1;
        }
        if pending.len() >= 5_000 {
            exec_dml(
                &server,
                &format!("INSERT INTO {ord} VALUES {}", pending.join(", ")),
            )
            .await;
            pending.clear();
        }
    }
    if !pending.is_empty() {
        exec_dml(
            &server,
            &format!("INSERT INTO {ord} VALUES {}", pending.join(", ")),
        )
        .await;
    }
    (server, tmp)
}

fn render(rows: &[Vec<ScalarValue>]) -> String {
    format!("{rows:?}")
}

async fn time_query(server: &Arc<ServerState>, sql: &str) -> f64 {
    let start = std::time::Instant::now();
    let _ = query_values(server, sql).await;
    start.elapsed().as_secs_f64() * 1_000_000.0
}

/// Times a subquery shape against the join that answers the same question,
/// records both, and reports what the subquery form costs.
///
/// Answers are compared first. A ratio between two queries that do not
/// agree measures nothing, and the join rewrite is only a control if it is
/// actually equivalent
async fn compare_shapes(
    server: &Arc<ServerState>,
    metric: &str,
    subquery_sql: &str,
    join_sql: &str,
) -> f64 {
    let subquery_answer = query_values(server, subquery_sql).await;
    let join_answer = query_values(server, join_sql).await;
    assert_eq!(
        render(&subquery_answer),
        render(&join_answer),
        "{metric}: the subquery form and the join form disagree, so the pair measures nothing\n  \
         subquery: {subquery_sql}\n  join: {join_sql}"
    );

    let mut subquery_runs = Vec::with_capacity(reps());
    let mut join_runs = Vec::with_capacity(reps());
    for rep in 0..reps() {
        // Alternate, so neither shape always runs into a cache the other
        // left warm
        if rep % 2 == 0 {
            subquery_runs.push(time_query(server, subquery_sql).await);
            join_runs.push(time_query(server, join_sql).await);
        } else {
            join_runs.push(time_query(server, join_sql).await);
            subquery_runs.push(time_query(server, subquery_sql).await);
        }
    }

    let subquery_us = record_metric(
        "subquery",
        &format!("{metric}, subquery form"),
        "us",
        subquery_runs,
    );
    let join_us = record_metric("subquery", &format!("{metric}, join form"), "us", join_runs);
    let ratio = subquery_us / join_us.max(f64::MIN_POSITIVE);
    tprintln!(
        "  {}: the subquery form costs {:.1}x the join form ({} vs {})",
        metric,
        ratio,
        format_measurement(subquery_us),
        format_measurement(join_us)
    );
    ratio
}

/// `WHERE EXISTS (correlated)`, which is a semi join written the way people
/// write it
#[tokio::test]
async fn test_correlated_exists_against_a_semi_join() {
    let _section = section("Correlated EXISTS");
    let (server, _tmp) = two_tables("ce", customers()).await;
    tprintln!("  Outer rows: {}", customers());
    tprintln!(
        "  Inner rows: {}",
        (customers() - CUSTOMERS_WITHOUT_ORDERS) * orders_per_customer()
    );

    compare_shapes(
        &server,
        "Correlated EXISTS",
        "SELECT COUNT(*) FROM cust_ce c WHERE EXISTS \
         (SELECT 1 FROM ord_ce o WHERE o.cust_id = c.id)",
        "SELECT COUNT(DISTINCT c.id) FROM cust_ce c JOIN ord_ce o ON o.cust_id = c.id",
    )
    .await;
}

/// `WHERE NOT EXISTS (correlated)`, which is an anti join
#[tokio::test]
async fn test_correlated_not_exists_against_an_anti_join() {
    let _section = section("Correlated NOT EXISTS");
    let (server, _tmp) = two_tables("cne", customers()).await;

    compare_shapes(
        &server,
        "Correlated NOT EXISTS",
        "SELECT COUNT(*) FROM cust_cne c WHERE NOT EXISTS \
         (SELECT 1 FROM ord_cne o WHERE o.cust_id = c.id)",
        "SELECT COUNT(*) FROM cust_cne c LEFT JOIN ord_cne o ON o.cust_id = c.id \
         WHERE o.id IS NULL",
    )
    .await;
}

/// A correlated aggregate in the select list, which is a group-by joined
/// back to the outer table
#[tokio::test]
async fn test_correlated_scalar_in_projection() {
    let _section = section("Correlated Scalar In Projection");
    let (server, _tmp) = two_tables("cs", customers()).await;

    compare_shapes(
        &server,
        "Correlated scalar",
        "SELECT SUM(x) FROM (SELECT (SELECT MAX(o.amount) FROM ord_cs o WHERE o.cust_id = c.id) \
         AS x FROM cust_cs c) t",
        "SELECT SUM(m.mx) FROM cust_cs c LEFT JOIN \
         (SELECT cust_id, MAX(amount) AS mx FROM ord_cs GROUP BY cust_id) m ON m.cust_id = c.id",
    )
    .await;
}

/// `IN (SELECT ...)` with no correlation. The subquery is materialized once
/// and folded into a list, so this is not once per outer row. What it costs
/// instead is one comparison pass per value in that list
#[tokio::test]
async fn test_uncorrelated_in_subquery_against_a_semi_join() {
    let _section = section("Uncorrelated IN Subquery");
    let (server, _tmp) = two_tables("ui", customers()).await;
    tprintln!(
        "  Distinct values the subquery materializes: {}",
        customers() - CUSTOMERS_WITHOUT_ORDERS
    );

    compare_shapes(
        &server,
        "Uncorrelated IN",
        "SELECT COUNT(*) FROM cust_ui c WHERE c.id IN (SELECT o.cust_id FROM ord_ui o)",
        "SELECT COUNT(DISTINCT c.id) FROM cust_ui c JOIN ord_ui o ON o.cust_id = c.id",
    )
    .await;
}

/// The same correlated EXISTS over twice the outer rows against a byte
/// identical inner table.
///
/// Two customer tables in one database share one orders table, so the only
/// thing that changes between the two timings is how many outer rows there
/// are. Varying the inner side as well would confound the two growth rates
/// and the pair would say nothing about either.
///
/// A join builds its hash from the inner side and probes once per outer
/// row, so doubling the outer side roughly doubles the probe and leaves the
/// build alone. A form that runs its inner plan once per outer row doubles
/// the whole query. The gap between those growth rates is what says the
/// cost is per outer row rather than merely large
#[tokio::test]
async fn test_correlated_exists_growth_in_outer_rows() {
    let _section = section("Correlated EXISTS Growth");
    let small = customers();
    let large = customers_doubled();

    // The large customer table is built first and the small one is a prefix
    // of it, so both address the same orders and the same order rows match
    let (server, _tmp) = two_tables("g", large).await;
    let mut session = new_session();
    exec_ddl(
        &server,
        &mut session,
        "CREATE TABLE cust_g_small (id BIGINT NOT NULL, region BIGINT, name TEXT)",
    )
    .await
    .expect("create the small customers table");
    exec_dml(
        &server,
        &format!("INSERT INTO cust_g_small SELECT * FROM cust_g WHERE id < {small}"),
    )
    .await;

    let counted = query_values(&server, "SELECT COUNT(*) FROM cust_g_small").await;
    let expected = format!("{:?}", vec![vec![ScalarValue::Int64(small as i64)]]);
    assert_eq!(
        format!("{counted:?}"),
        expected,
        "the small customer table has to hold exactly {small} rows for the pair to mean anything"
    );

    let subquery = |t: &str| {
        format!(
            "SELECT COUNT(*) FROM {t} c WHERE EXISTS (SELECT 1 FROM ord_g o WHERE o.cust_id = c.id)"
        )
    };
    let join = |t: &str| {
        format!("SELECT COUNT(DISTINCT c.id) FROM {t} c JOIN ord_g o ON o.cust_id = c.id")
    };

    let mut small_runs = Vec::with_capacity(reps());
    let mut large_runs = Vec::with_capacity(reps());
    let mut join_small_runs = Vec::with_capacity(reps());
    let mut join_large_runs = Vec::with_capacity(reps());
    for _ in 0..reps() {
        small_runs.push(time_query(&server, &subquery("cust_g_small")).await);
        large_runs.push(time_query(&server, &subquery("cust_g")).await);
        join_small_runs.push(time_query(&server, &join("cust_g_small")).await);
        join_large_runs.push(time_query(&server, &join("cust_g")).await);
    }

    let small_us = record_metric(
        "subquery",
        "EXISTS growth, subquery at 1x outer rows",
        "us",
        small_runs,
    );
    let large_us = record_metric(
        "subquery",
        "EXISTS growth, subquery at 2x outer rows",
        "us",
        large_runs,
    );
    let join_small_us = record_metric(
        "subquery",
        "EXISTS growth, join at 1x outer rows",
        "us",
        join_small_runs,
    );
    let join_large_us = record_metric(
        "subquery",
        "EXISTS growth, join at 2x outer rows",
        "us",
        join_large_runs,
    );

    tprintln!(
        "  Outer rows {} to {} against one fixed inner table: subquery form grows {:.2}x, join form grows {:.2}x",
        small,
        large,
        large_us / small_us.max(f64::MIN_POSITIVE),
        join_large_us / join_small_us.max(f64::MIN_POSITIVE)
    );
}
