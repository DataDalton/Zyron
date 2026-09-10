//! PIVOT and UNPIVOT, through the engine.
//!
//! Run: cargo test -p zyron-wire --test pivot_test -- --nocapture

mod common;

use common::{exec_ddl, exec_dml, new_session, query_error, query_values, render_plan};
use zyron_executor::column::ScalarValue;
use zyron_parser::{parse, statement_to_sql};

fn number(value: &ScalarValue) -> Option<f64> {
    match value {
        ScalarValue::Int8(v) => Some(*v as f64),
        ScalarValue::Int16(v) => Some(*v as f64),
        ScalarValue::Int32(v) => Some(*v as f64),
        ScalarValue::Int64(v) => Some(*v as f64),
        ScalarValue::Int128(v) => Some(*v as f64),
        ScalarValue::Float32(v) => Some(*v as f64),
        ScalarValue::Float64(v) => Some(*v),
        _ => None,
    }
}

fn text(value: &ScalarValue) -> String {
    match value {
        ScalarValue::Utf8(s) => s.clone(),
        ScalarValue::Null => "<null>".to_string(),
        other => format!("{other:?}"),
    }
}

/// A four-region, four-quarter sales table with one row per pair.
async fn sales_server() -> (
    std::sync::Arc<zyron_wire::connection::ServerState>,
    tempfile::TempDir,
) {
    let (server, _schema, tmp) = common::create_test_server().await;
    let mut session = new_session();
    exec_ddl(
        &server,
        &mut session,
        "CREATE TABLE sales (region TEXT, quarter TEXT, amount INT)",
    )
    .await
    .expect("create");
    let regions = ["east", "west", "north", "south"];
    let quarters = ["Q1", "Q2", "Q3", "Q4"];
    let mut values = Vec::new();
    for (r, region) in regions.iter().enumerate() {
        for (q, quarter) in quarters.iter().enumerate() {
            values.push(format!(
                "('{region}', '{quarter}', {})",
                (r + 1) * (q + 1) * 10
            ));
        }
    }
    exec_dml(
        &server,
        &format!("INSERT INTO sales VALUES {}", values.join(", ")),
    )
    .await;
    (server, tmp)
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn pivot_over_four_regions_and_four_quarters_produces_the_four_by_four_result() {
    let (server, _tmp) = sales_server().await;
    let rows = query_values(
        &server,
        "SELECT * FROM sales PIVOT (SUM(amount) FOR quarter IN ('Q1', 'Q2', 'Q3', 'Q4')) AS p ORDER BY region",
    )
    .await;
    assert_eq!(rows.len(), 4, "one row per region");
    assert_eq!(rows[0].len(), 5, "the region plus one column per quarter");

    // Rows come back ordered by region: east, north, south, west
    let by_region: Vec<(String, Vec<f64>)> = rows
        .iter()
        .map(|r| {
            (
                text(&r[0]),
                r[1..]
                    .iter()
                    .map(|v| number(v).unwrap_or(f64::NAN))
                    .collect(),
            )
        })
        .collect();
    let east = by_region
        .iter()
        .find(|(name, _)| name == "east")
        .expect("east is a row");
    assert_eq!(east.1, vec![10.0, 20.0, 30.0, 40.0]);
    let south = by_region
        .iter()
        .find(|(name, _)| name == "south")
        .expect("south is a row");
    assert_eq!(south.1, vec![40.0, 80.0, 120.0, 160.0]);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn two_aggregates_produce_eight_columns_named_by_alias() {
    let (server, _tmp) = sales_server().await;
    let rows = query_values(
        &server,
        "SELECT * FROM sales PIVOT (SUM(amount) AS total, COUNT(amount) AS n FOR quarter IN ('Q1', 'Q2', 'Q3', 'Q4')) AS p ORDER BY region",
    )
    .await;
    assert_eq!(rows.len(), 4);
    assert_eq!(
        rows[0].len(),
        9,
        "the region plus two aggregates over four quarters"
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_pivot_alias_names_the_output_column() {
    let (server, _tmp) = sales_server().await;
    // The rewrite is a grouped aggregate, so the pivot's own value aliases
    // decide what the output columns are called
    let rows = query_values(
        &server,
        "SELECT region, q1 FROM sales PIVOT (SUM(amount) FOR quarter IN ('Q1' AS q1, 'Q2' AS q2)) AS p ORDER BY region",
    )
    .await;
    assert_eq!(rows.len(), 4);
    assert_eq!(rows[0].len(), 2, "the alias is addressable by name");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn explain_shows_the_conditional_aggregate_rewrite() {
    let (server, _tmp) = sales_server().await;
    let plan = render_plan(
        &server,
        "SELECT * FROM sales PIVOT (SUM(amount) FOR quarter IN ('Q1', 'Q2')) AS p",
    )
    .await;
    assert!(
        plan.contains("Aggregate"),
        "the rewrite is a grouped aggregate and EXPLAIN should show it\n{plan}"
    );
    assert!(
        !plan.contains("Pivot"),
        "nothing hides: the plan is the rewrite, not a pivot operator\n{plan}"
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_subquery_value_list_is_refused_with_the_two_statement_message() {
    let (server, _tmp) = sales_server().await;
    let error = query_error(
        &server,
        "SELECT * FROM sales PIVOT (SUM(amount) FOR quarter IN (SELECT DISTINCT quarter FROM sales))",
    )
    .await;
    assert!(
        error.contains("static"),
        "the refusal says the list has to be static, got {error}"
    );
    assert!(
        error.contains("SELECT DISTINCT"),
        "the refusal names the first of the two statements, got {error}"
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn the_unparser_writes_a_pivot_back_as_written() {
    let sql = "SELECT * FROM sales PIVOT (SUM(amount) AS total FOR quarter IN ('Q1' AS q1, 'Q2' AS q2)) AS p";
    let first = parse(sql).expect("parses");
    let rendered = statement_to_sql(&first[0]).expect("renders");
    assert!(
        rendered.contains("PIVOT") && rendered.contains("FOR"),
        "stored SQL keeps the construct as written, got {rendered}"
    );
    let second = parse(&rendered).expect("the rendering parses");
    assert_eq!(second, first, "the statement does not survive a round trip");
}

// ---------------------------------------------------------------------------
// UNPIVOT
// ---------------------------------------------------------------------------

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn unpivot_of_four_month_columns_produces_four_rows_per_input_row() {
    let (server, _schema, _tmp) = common::create_test_server().await;
    let mut session = new_session();
    exec_ddl(
        &server,
        &mut session,
        "CREATE TABLE budget (dept TEXT, jan INT, feb INT, mar INT, apr INT)",
    )
    .await
    .expect("create");
    exec_dml(
        &server,
        "INSERT INTO budget VALUES ('ops', 1, 2, 3, 4), ('eng', 10, 20, 30, 40)",
    )
    .await;

    let rows = query_values(
        &server,
        "SELECT dept, month, amount FROM budget UNPIVOT (amount FOR month IN (jan, feb, mar, apr)) AS u ORDER BY dept, amount",
    )
    .await;
    assert_eq!(rows.len(), 8, "four months over two departments");
    assert_eq!(
        rows.iter().map(|r| text(&r[1])).collect::<Vec<_>>(),
        vec!["jan", "feb", "mar", "apr", "jan", "feb", "mar", "apr"],
        "the name column takes the source column's name"
    );
    assert_eq!(
        rows.iter()
            .map(|r| number(&r[2]).unwrap_or(f64::NAN))
            .collect::<Vec<_>>(),
        vec![10.0, 20.0, 30.0, 40.0, 1.0, 2.0, 3.0, 4.0]
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn exclude_nulls_drops_a_null_month_and_include_nulls_keeps_it() {
    let (server, _schema, _tmp) = common::create_test_server().await;
    let mut session = new_session();
    exec_ddl(
        &server,
        &mut session,
        "CREATE TABLE budget (dept TEXT, jan INT, feb INT)",
    )
    .await
    .expect("create");
    exec_dml(&server, "INSERT INTO budget VALUES ('ops', 1, NULL)").await;

    let excluded = query_values(
        &server,
        "SELECT month FROM budget UNPIVOT EXCLUDE NULLS (amount FOR month IN (jan, feb)) AS u",
    )
    .await;
    assert_eq!(excluded.len(), 1, "the null month is dropped");
    assert_eq!(text(&excluded[0][0]), "jan");

    let included = query_values(
        &server,
        "SELECT month FROM budget UNPIVOT INCLUDE NULLS (amount FOR month IN (jan, feb)) AS u ORDER BY month",
    )
    .await;
    assert_eq!(included.len(), 2, "the null month is kept");

    // The default with no modifier drops nulls, like EXCLUDE NULLS
    let default = query_values(
        &server,
        "SELECT month FROM budget UNPIVOT (amount FOR month IN (jan, feb)) AS u",
    )
    .await;
    assert_eq!(default.len(), 1);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_label_renames_the_group_in_the_name_column() {
    let (server, _schema, _tmp) = common::create_test_server().await;
    let mut session = new_session();
    exec_ddl(
        &server,
        &mut session,
        "CREATE TABLE budget (dept TEXT, jan INT, feb INT)",
    )
    .await
    .expect("create");
    exec_dml(&server, "INSERT INTO budget VALUES ('ops', 1, 2)").await;

    let rows = query_values(
        &server,
        "SELECT month FROM budget UNPIVOT (amount FOR month IN (jan AS 'January', feb AS 'February')) AS u ORDER BY amount",
    )
    .await;
    assert_eq!(
        rows.iter().map(|r| text(&r[0])).collect::<Vec<_>>(),
        vec!["January", "February"]
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_two_value_column_unpivot_with_tuple_groups_works() {
    let (server, _schema, _tmp) = common::create_test_server().await;
    let mut session = new_session();
    exec_ddl(
        &server,
        &mut session,
        "CREATE TABLE budget (dept TEXT, jan_amt INT, jan_tax INT, feb_amt INT, feb_tax INT)",
    )
    .await
    .expect("create");
    exec_dml(&server, "INSERT INTO budget VALUES ('ops', 1, 2, 3, 4)").await;

    let rows = query_values(
        &server,
        "SELECT month, amount, tax FROM budget UNPIVOT ((amount, tax) FOR month IN ((jan_amt, jan_tax) AS 'Jan', (feb_amt, feb_tax) AS 'Feb')) AS u ORDER BY amount",
    )
    .await;
    assert_eq!(rows.len(), 2, "one row per group");
    assert_eq!(text(&rows[0][0]), "Jan");
    assert_eq!(number(&rows[0][1]), Some(1.0));
    assert_eq!(number(&rows[0][2]), Some(2.0));
    assert_eq!(text(&rows[1][0]), "Feb");
    assert_eq!(number(&rows[1][1]), Some(3.0));
    assert_eq!(number(&rows[1][2]), Some(4.0));
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn an_unpivot_drops_the_columns_it_consumed_and_keeps_the_rest() {
    let (server, _schema, _tmp) = common::create_test_server().await;
    let mut session = new_session();
    exec_ddl(
        &server,
        &mut session,
        "CREATE TABLE budget (dept TEXT, year INT, jan INT, feb INT)",
    )
    .await
    .expect("create");
    exec_dml(&server, "INSERT INTO budget VALUES ('ops', 2026, 1, 2)").await;

    let rows = query_values(
        &server,
        "SELECT * FROM budget UNPIVOT (amount FOR month IN (jan, feb)) AS u ORDER BY amount",
    )
    .await;
    assert_eq!(rows.len(), 2);
    assert_eq!(
        rows[0].len(),
        4,
        "dept and year travel, jan and feb become month and amount"
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn the_unparser_writes_an_unpivot_back_as_written() {
    let sql = "SELECT * FROM budget UNPIVOT EXCLUDE NULLS (amount FOR month IN (jan AS 'Jan', feb AS 'Feb')) AS u";
    let first = parse(sql).expect("parses");
    let rendered = statement_to_sql(&first[0]).expect("renders");
    assert!(
        rendered.contains("UNPIVOT"),
        "stored SQL keeps the construct as written, got {rendered}"
    );
    let second = parse(&rendered).expect("the rendering parses");
    assert_eq!(second, first);
}

/// A PIVOT over a derived table binds that table's shape once.
///
/// Learning the input's output columns binds it, and the rewritten query
/// binds it again to run it. Without a memo of the shape, a PIVOT whose input
/// is itself a PIVOT binds the innermost relation once per level of nesting.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_pivot_over_a_derived_table_produces_the_same_result_as_over_the_table() {
    let (server, _tmp) = sales_server().await;
    let direct = query_values(
        &server,
        "SELECT * FROM sales PIVOT (SUM(amount) FOR quarter IN ('Q1', 'Q2')) AS p ORDER BY region",
    )
    .await;
    let derived = query_values(
        &server,
        "SELECT * FROM (SELECT region, quarter, amount FROM sales) AS s \
         PIVOT (SUM(amount) FOR quarter IN ('Q1', 'Q2')) AS p ORDER BY region",
    )
    .await;
    assert_eq!(
        derived.len(),
        direct.len(),
        "the derived input yields the same rows"
    );
    for (a, b) in direct.iter().zip(&derived) {
        let left: Vec<String> = a.iter().map(text).collect();
        let right: Vec<String> = b.iter().map(text).collect();
        assert_eq!(left, right, "the derived input yields the same values");
    }
}
