//! UNNEST, FLATTEN and the array functions, through the engine.
//!
//! Run: cargo test -p zyron-wire --test unnest_test -- --nocapture

mod common;

use common::{exec_ddl, exec_dml, new_session, query_error, query_values, render_plan};
use zyron_executor::column::ScalarValue;

fn ints(rows: &[Vec<ScalarValue>], column: usize) -> Vec<i64> {
    rows.iter()
        .map(|r| match &r[column] {
            ScalarValue::Int64(v) => *v,
            ScalarValue::Int32(v) => *v as i64,
            ScalarValue::Null => i64::MIN,
            other => panic!("expected an integer, got {other:?}"),
        })
        .collect()
}

fn texts(rows: &[Vec<ScalarValue>], column: usize) -> Vec<String> {
    rows.iter()
        .map(|r| match &r[column] {
            ScalarValue::Utf8(s) => s.clone(),
            ScalarValue::Null => "<null>".to_string(),
            other => format!("{other:?}"),
        })
        .collect()
}

// ---------------------------------------------------------------------------
// UNNEST
// ---------------------------------------------------------------------------

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn unnest_yields_one_row_per_element_in_order() {
    let (server, _schema, _tmp) = common::create_test_server().await;
    let rows = query_values(&server, "SELECT e FROM UNNEST(ARRAY[10, 20, 30]) AS t (e)").await;
    assert_eq!(ints(&rows, 0), vec![10, 20, 30]);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn with_ordinality_numbers_the_rows_from_one() {
    let (server, _schema, _tmp) = common::create_test_server().await;
    let rows = query_values(
        &server,
        "SELECT e, n FROM UNNEST(ARRAY[7, 8, 9]) WITH ORDINALITY AS t (e, n)",
    )
    .await;
    assert_eq!(ints(&rows, 0), vec![7, 8, 9]);
    assert_eq!(ints(&rows, 1), vec![1, 2, 3], "the position starts at 1");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn two_arrays_zip_to_the_longer_with_null_padding() {
    let (server, _schema, _tmp) = common::create_test_server().await;
    let rows = query_values(
        &server,
        "SELECT a, b FROM UNNEST(ARRAY[1, 2, 3], ARRAY[10, 20, 30, 40, 50]) AS t (a, b)",
    )
    .await;
    assert_eq!(rows.len(), 5, "the longer array decides the row count");
    assert_eq!(ints(&rows, 0), vec![1, 2, 3, i64::MIN, i64::MIN]);
    assert_eq!(ints(&rows, 1), vec![10, 20, 30, 40, 50]);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn an_empty_array_yields_no_rows() {
    let (server, _schema, _tmp) = common::create_test_server().await;
    let rows = query_values(&server, "SELECT e FROM UNNEST(ARRAY[]) AS t (e)").await;
    assert!(rows.is_empty());
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn unnest_under_lateral_joins_back_on_the_parent_key() {
    let (server, _schema, _tmp) = common::create_test_server().await;
    let mut session = new_session();
    exec_ddl(
        &server,
        &mut session,
        "CREATE TABLE orders (id INT, items INT[])",
    )
    .await
    .expect("create");
    exec_dml(
        &server,
        "INSERT INTO orders VALUES (1, ARRAY[5, 6]), (2, ARRAY[7]), (3, ARRAY[8, 9, 10])",
    )
    .await;

    let rows = query_values(
        &server,
        "SELECT o.id, e FROM orders AS o, LATERAL UNNEST(o.items) AS u (e) ORDER BY o.id, e",
    )
    .await;
    assert_eq!(rows.len(), 6, "two plus one plus three elements");
    assert_eq!(ints(&rows, 0), vec![1, 1, 2, 3, 3, 3]);
    assert_eq!(ints(&rows, 1), vec![5, 6, 7, 8, 9, 10]);

    let summed = query_values(
        &server,
        "SELECT SUM(e) FROM orders AS o, LATERAL UNNEST(o.items) AS u (e)",
    )
    .await;
    assert_eq!(ints(&summed, 0), vec![45]);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_bare_reference_to_a_preceding_relation_is_refused_naming_lateral() {
    let (server, _schema, _tmp) = common::create_test_server().await;
    let mut session = new_session();
    exec_ddl(
        &server,
        &mut session,
        "CREATE TABLE orders (id INT, items INT[])",
    )
    .await
    .expect("create");

    let error = query_error(
        &server,
        "SELECT o.id, e FROM orders AS o, UNNEST(o.items) AS u (e)",
    )
    .await;
    assert!(
        error.contains("LATERAL"),
        "the refusal has to name LATERAL so a plan is never silently correlated, got {error}"
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn an_expansion_carries_only_the_columns_the_query_reads() {
    let (server, _schema, _tmp) = common::create_test_server().await;
    let mut session = new_session();
    exec_ddl(
        &server,
        &mut session,
        "CREATE TABLE orders (id INT, note TEXT, items INT[])",
    )
    .await
    .expect("create");

    let stmt =
        zyron_parser::parse("SELECT o.id, e FROM orders AS o, LATERAL UNNEST(o.items) AS u (e)")
            .expect("parses")
            .remove(0);
    let plan = zyron_planner::plan(
        &server.catalog,
        zyron_catalog::DatabaseId(1),
        vec!["zyron_test".into()],
        stmt,
        None,
    )
    .await
    .expect("plans");

    // The array being unnested and a column nothing reads would otherwise be
    // gathered once per produced row and thrown away, so ten elements would
    // cost ten copies of a value with no reader
    let mut found = None;
    fn walk(plan: &zyron_planner::physical::PhysicalPlan, found: &mut Option<usize>) {
        if let zyron_planner::physical::PhysicalPlan::ExpandRows { spec, .. } = plan {
            *found = Some(spec.carry.len());
        }
        plan.for_each_child(&mut |child| walk(child, found));
    }
    walk(&plan, &mut found);
    assert_eq!(
        found,
        Some(1),
        "the expansion should carry only o.id, not the array it reads or the column nothing reads"
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn explain_names_the_expansion_and_its_source_column() {
    let (server, _schema, _tmp) = common::create_test_server().await;
    let mut session = new_session();
    exec_ddl(
        &server,
        &mut session,
        "CREATE TABLE orders (id INT, items INT[])",
    )
    .await
    .expect("create");
    let plan = render_plan(
        &server,
        "SELECT e FROM orders AS o, LATERAL UNNEST(o.items) AS u (e)",
    )
    .await;
    assert!(
        plan.contains("Unnest"),
        "EXPLAIN should name Unnest\n{plan}"
    );
    assert!(
        plan.contains("source"),
        "EXPLAIN should name the source column\n{plan}"
    );
}

// ---------------------------------------------------------------------------
// FLATTEN
// ---------------------------------------------------------------------------

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn flatten_yields_the_six_documented_columns() {
    let (server, _schema, _tmp) = common::create_test_server().await;
    let mut session = new_session();
    exec_ddl(
        &server,
        &mut session,
        "CREATE TABLE docs (id INT, d VARIANT)",
    )
    .await
    .expect("create");
    exec_dml(
        &server,
        r#"INSERT INTO docs VALUES (1, '{"a": 1, "b": [10, 20]}')"#,
    )
    .await;

    let rows = query_values(
        &server,
        "SELECT f.seq, f.key, f.path, f.index FROM docs AS d, LATERAL FLATTEN(d.d) AS f ORDER BY f.seq",
    )
    .await;
    assert_eq!(rows.len(), 2, "two members at the top level");
    assert_eq!(ints(&rows, 0), vec![1, 2], "seq numbers from 1");
    assert_eq!(texts(&rows, 1), vec!["a".to_string(), "b".to_string()]);
    assert_eq!(texts(&rows, 2), vec!["a".to_string(), "b".to_string()]);
    assert_eq!(
        ints(&rows, 3),
        vec![i64::MIN, i64::MIN],
        "an object member has no index"
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn flatten_over_an_array_carries_the_index_and_no_key() {
    let (server, _schema, _tmp) = common::create_test_server().await;
    let mut session = new_session();
    exec_ddl(
        &server,
        &mut session,
        "CREATE TABLE docs (id INT, d VARIANT)",
    )
    .await
    .expect("create");
    exec_dml(&server, r#"INSERT INTO docs VALUES (1, '[100, 200, 300]')"#).await;

    let rows = query_values(
        &server,
        "SELECT f.index, f.value FROM docs AS d, LATERAL FLATTEN(d.d) AS f ORDER BY f.seq",
    )
    .await;
    assert_eq!(
        ints(&rows, 0),
        vec![0, 1, 2],
        "array positions are zero based"
    );
    assert_eq!(
        texts(&rows, 1),
        vec!["100".to_string(), "200".to_string(), "300".to_string()]
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn outer_yields_one_null_value_row_for_an_empty_array() {
    let (server, _schema, _tmp) = common::create_test_server().await;
    let mut session = new_session();
    exec_ddl(
        &server,
        &mut session,
        "CREATE TABLE docs (id INT, d VARIANT)",
    )
    .await
    .expect("create");
    exec_dml(&server, r#"INSERT INTO docs VALUES (1, '[]')"#).await;

    let without = query_values(
        &server,
        "SELECT f.value FROM docs AS d, LATERAL FLATTEN(d.d) AS f",
    )
    .await;
    assert!(without.is_empty(), "an empty array reaches no member");

    let with_outer = query_values(
        &server,
        "SELECT f.value FROM docs AS d, LATERAL FLATTEN(d.d, outer => TRUE) AS f",
    )
    .await;
    assert_eq!(with_outer.len(), 1, "outer keeps the input row");
    assert!(
        matches!(with_outer[0][0], ScalarValue::Null),
        "the value column is null, got {:?}",
        with_outer[0][0]
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn recursive_reaches_a_leaf_three_levels_down_with_the_right_path() {
    let (server, _schema, _tmp) = common::create_test_server().await;
    let mut session = new_session();
    exec_ddl(
        &server,
        &mut session,
        "CREATE TABLE docs (id INT, d VARIANT)",
    )
    .await
    .expect("create");
    exec_dml(
        &server,
        r#"INSERT INTO docs VALUES (1, '{"a": {"b": {"c": 7}}}')"#,
    )
    .await;

    let shallow = query_values(
        &server,
        "SELECT f.path FROM docs AS d, LATERAL FLATTEN(d.d) AS f ORDER BY f.seq",
    )
    .await;
    assert_eq!(texts(&shallow, 0), vec!["a".to_string()]);

    let deep = query_values(
        &server,
        "SELECT f.path, f.value FROM docs AS d, LATERAL FLATTEN(d.d, recursive => TRUE) AS f ORDER BY f.seq",
    )
    .await;
    assert_eq!(
        texts(&deep, 0),
        vec!["a".to_string(), "a.b".to_string(), "a.b.c".to_string()]
    );
    assert_eq!(
        texts(&deep, 1).last().map(|s| s.as_str()),
        Some("7"),
        "the leaf's value is the number it holds"
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_path_starts_the_walk_where_it_names() {
    let (server, _schema, _tmp) = common::create_test_server().await;
    let mut session = new_session();
    exec_ddl(
        &server,
        &mut session,
        "CREATE TABLE docs (id INT, d VARIANT)",
    )
    .await
    .expect("create");
    exec_dml(
        &server,
        r#"INSERT INTO docs VALUES (1, '{"a": {"b": [{"c": 1}, {"c": 2}]}}')"#,
    )
    .await;

    let rows = query_values(
        &server,
        "SELECT f.index, f.value FROM docs AS d, LATERAL FLATTEN(d.d, path => 'a.b[*]') AS f ORDER BY f.seq",
    )
    .await;
    assert_eq!(rows.len(), 2, "the walk starts at the named array");
    assert_eq!(ints(&rows, 0), vec![0, 1]);
}

// ---------------------------------------------------------------------------
// Array functions
// ---------------------------------------------------------------------------

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn array_functions_answer_on_empty_single_and_null_inputs() {
    let (server, _schema, _tmp) = common::create_test_server().await;
    let mut session = new_session();
    exec_ddl(
        &server,
        &mut session,
        "CREATE TABLE arrs (id INT, v INT[], t TEXT[])",
    )
    .await
    .expect("create");
    exec_dml(
        &server,
        "INSERT INTO arrs VALUES (1, ARRAY[], ARRAY[]), (2, ARRAY[5], ARRAY['x']), (3, NULL, NULL)",
    )
    .await;

    let lengths = query_values(&server, "SELECT array_length(v) FROM arrs ORDER BY id").await;
    assert_eq!(
        ints(&lengths, 0),
        vec![0, 1, i64::MIN],
        "empty is zero, one element is one, NULL is NULL"
    );

    let positions =
        query_values(&server, "SELECT array_position(v, 5) FROM arrs ORDER BY id").await;
    assert_eq!(ints(&positions, 0), vec![i64::MIN, 1, i64::MIN]);

    let contains = query_values(&server, "SELECT array_contains(v, 5) FROM arrs ORDER BY id").await;
    let flags: Vec<Option<bool>> = contains
        .iter()
        .map(|r| match r[0] {
            ScalarValue::Boolean(b) => Some(b),
            _ => None,
        })
        .collect();
    assert_eq!(flags, vec![Some(false), Some(true), None]);

    let joined = query_values(
        &server,
        "SELECT array_to_string(t, ',') FROM arrs ORDER BY id",
    )
    .await;
    assert_eq!(
        texts(&joined, 0),
        vec!["".to_string(), "x".to_string(), "<null>".to_string()]
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn array_distinct_sort_slice_and_concat_return_arrays_that_unnest() {
    let (server, _schema, _tmp) = common::create_test_server().await;
    let rows = query_values(
        &server,
        "SELECT e FROM UNNEST(array_distinct(ARRAY[3, 1, 3, 2, 1])) AS t (e)",
    )
    .await;
    assert_eq!(
        ints(&rows, 0),
        vec![3, 1, 2],
        "duplicates go, first appearance order stays"
    );

    let sorted = query_values(
        &server,
        "SELECT e FROM UNNEST(array_sort(ARRAY[3, 1, 2])) AS t (e)",
    )
    .await;
    assert_eq!(ints(&sorted, 0), vec![1, 2, 3]);

    let sliced = query_values(
        &server,
        "SELECT e FROM UNNEST(array_slice(ARRAY[10, 20, 30, 40], 2, 2)) AS t (e)",
    )
    .await;
    assert_eq!(ints(&sliced, 0), vec![20, 30], "start is one based");

    let joined = query_values(
        &server,
        "SELECT e FROM UNNEST(array_concat(ARRAY[1, 2], ARRAY[3])) AS t (e)",
    )
    .await;
    assert_eq!(ints(&joined, 0), vec![1, 2, 3]);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn string_to_array_and_back_round_trip() {
    let (server, _schema, _tmp) = common::create_test_server().await;
    let rows = query_values(
        &server,
        "SELECT e FROM UNNEST(string_to_array('a,b,c', ',')) AS t (e)",
    )
    .await;
    assert_eq!(
        texts(&rows, 0),
        vec!["a".to_string(), "b".to_string(), "c".to_string()]
    );

    let back = query_values(
        &server,
        "SELECT array_to_string(string_to_array('a,b,c', ','), '-')",
    )
    .await;
    assert_eq!(texts(&back, 0), vec!["a-b-c".to_string()]);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_lambda_filters_and_transforms_the_same_way_unnest_plus_aggregation_does() {
    let (server, _schema, _tmp) = common::create_test_server().await;
    let mut session = new_session();
    exec_ddl(&server, &mut session, "CREATE TABLE arrs (id INT, v INT[])")
        .await
        .expect("create");
    exec_dml(
        &server,
        "INSERT INTO arrs VALUES (1, ARRAY[1, 2, 3, 4]), (2, ARRAY[10, 20])",
    )
    .await;

    let filtered = query_values(
        &server,
        "SELECT array_length(array_filter(v, x -> x > 2)) FROM arrs ORDER BY id",
    )
    .await;
    assert_eq!(ints(&filtered, 0), vec![2, 2]);

    // The lambda's answer has to equal what UNNEST plus aggregation gives
    let by_lambda = query_values(
        &server,
        "SELECT e FROM arrs AS a, LATERAL UNNEST(array_transform(a.v, x -> x * 2)) AS u (e) ORDER BY e",
    )
    .await;
    let by_unnest = query_values(
        &server,
        "SELECT e * 2 FROM arrs AS a, LATERAL UNNEST(a.v) AS u (e) ORDER BY e * 2",
    )
    .await;
    assert_eq!(
        ints(&by_lambda, 0),
        ints(&by_unnest, 0),
        "a lambda and an unnest have to agree element for element"
    );

    let summed = query_values(
        &server,
        "SELECT SUM(e) FROM arrs AS a, LATERAL UNNEST(array_filter(a.v, x -> x > 2)) AS u (e)",
    )
    .await;
    assert_eq!(ints(&summed, 0), vec![3 + 4 + 10 + 20]);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_lambda_that_does_not_yield_a_boolean_is_refused_by_array_filter() {
    let (server, _schema, _tmp) = common::create_test_server().await;
    let error = query_error(&server, "SELECT array_filter(ARRAY[1, 2], x -> x + 1)").await;
    assert!(
        error.contains("boolean"),
        "the refusal names what a filter's lambda has to yield, got {error}"
    );
}
