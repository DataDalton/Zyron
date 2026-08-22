//! ARRAY values end to end: constructed, stored, read back and subscripted.
//!
//! An array is one variable-length payload, so no storage path knows it is an
//! array. What the tests here pin is that the encoding survives a round trip
//! through that path unchanged, that a null element stays distinct from an
//! empty one, and that a subscript addresses the element it names rather than
//! its neighbour.
//!
//! Run: cargo test -p zyron-wire --test array_value_test

mod common;

use common::{create_test_server, exec_ddl, exec_dml, new_session, query_values};
use zyron_executor::column::ScalarValue;

/// Every row's single column as text, in row order
fn texts(rows: &[Vec<ScalarValue>]) -> Vec<String> {
    rows.iter()
        .map(|row| match row.first() {
            Some(ScalarValue::Utf8(s)) => s.clone(),
            Some(ScalarValue::Null) | None => "NULL".to_string(),
            other => panic!("expected text, got {other:?}"),
        })
        .collect()
}

#[tokio::test]
async fn test_array_column_round_trips_through_storage() {
    let (server, _schema, _tmp) = create_test_server().await;
    let mut session = new_session();
    exec_ddl(
        &server,
        &mut session,
        "CREATE TABLE arr (id INT, nums INT[])",
    )
    .await
    .expect("create table");

    exec_dml(&server, "INSERT INTO arr VALUES (1, ARRAY[10, 20, 30])").await;
    exec_dml(&server, "INSERT INTO arr VALUES (2, ARRAY[])").await;
    exec_dml(&server, "INSERT INTO arr VALUES (3, ARRAY[7, NULL, 9])").await;
    exec_dml(&server, "INSERT INTO arr VALUES (4, NULL)").await;

    assert_eq!(
        texts(&query_values(&server, "SELECT CAST(nums AS TEXT) FROM arr ORDER BY id",).await),
        vec![
            "{10,20,30}".to_string(),
            "{}".to_string(),
            // A null element keeps its position rather than collapsing the list
            "{7,NULL,9}".to_string(),
            "NULL".to_string(),
        ]
    );

    // The stored value carries the element type the column declares, not the
    // one an integer literal happens to bind to, so an INT[] holds four-byte
    // elements rather than eight-byte ones
    let stored = query_values(&server, "SELECT nums FROM arr WHERE id = 1").await;
    let ScalarValue::Binary(bytes) = stored[0][0].clone() else {
        panic!("an array column reads back as its encoding");
    };
    let view = zyron_common::ArrayView::parse(&bytes).expect("the stored bytes are an array");
    assert_eq!(view.element_type(), zyron_common::TypeId::Int32);
    assert_eq!(view.len(), 3);
    assert_eq!(
        view.get(1),
        Some(Some(&20i32.to_le_bytes()[..])),
        "elements are stored at the declared width"
    );
}

#[tokio::test]
async fn test_array_subscript_addresses_the_element_it_names() {
    let (server, _schema, _tmp) = create_test_server().await;
    let mut session = new_session();
    exec_ddl(
        &server,
        &mut session,
        "CREATE TABLE sub (id INT, nums INT[])",
    )
    .await
    .expect("create table");
    exec_dml(&server, "INSERT INTO sub VALUES (1, ARRAY[10, 20, 30])").await;

    let read = |sql: &'static str| {
        let server = server.clone();
        async move {
            query_values(&server, sql)
                .await
                .first()
                .and_then(|row| row.first().cloned())
        }
    };

    // Subscripts are one-based, so the first element is at 1
    assert_eq!(
        read("SELECT nums[1] FROM sub").await,
        Some(ScalarValue::Int32(10))
    );
    assert_eq!(
        read("SELECT nums[2] FROM sub").await,
        Some(ScalarValue::Int32(20))
    );
    assert_eq!(
        read("SELECT nums[3] FROM sub").await,
        Some(ScalarValue::Int32(30))
    );
    // Outside the array reads as absent rather than as a neighbour
    assert_eq!(
        read("SELECT nums[4] FROM sub").await,
        Some(ScalarValue::Null)
    );
    assert_eq!(
        read("SELECT nums[0] FROM sub").await,
        Some(ScalarValue::Null)
    );
}

#[tokio::test]
async fn test_text_array_keeps_elements_distinct() {
    let (server, _schema, _tmp) = create_test_server().await;
    let mut session = new_session();
    exec_ddl(
        &server,
        &mut session,
        "CREATE TABLE tags (id INT, labels TEXT[])",
    )
    .await
    .expect("create table");
    exec_dml(
        &server,
        "INSERT INTO tags VALUES (1, ARRAY['alpha', '', 'has,comma'])",
    )
    .await;

    // An empty element is not a null element, and a comma inside an element
    // must not read as a separator
    assert_eq!(
        texts(&query_values(&server, "SELECT CAST(labels AS TEXT) FROM tags").await),
        vec!["{alpha,\"\",\"has,comma\"}".to_string()]
    );
    assert_eq!(
        query_values(&server, "SELECT labels[3] FROM tags")
            .await
            .first()
            .and_then(|row| row.first().cloned()),
        Some(ScalarValue::Utf8("has,comma".to_string()))
    );
}

/// An array built from column values, not literals, so the per-row assembly
/// path is what answers rather than a constant folded once.
#[tokio::test]
async fn test_array_built_from_column_values_varies_per_row() {
    let (server, _schema, _tmp) = create_test_server().await;
    let mut session = new_session();
    exec_ddl(
        &server,
        &mut session,
        "CREATE TABLE pairs (id INT, a INT, b INT)",
    )
    .await
    .expect("create table");
    exec_dml(
        &server,
        "INSERT INTO pairs VALUES (1, 10, 20), (2, 30, 40), (3, 50, NULL)",
    )
    .await;

    assert_eq!(
        texts(
            &query_values(
                &server,
                "SELECT CAST(ARRAY[a, b] AS TEXT) FROM pairs ORDER BY id",
            )
            .await
        ),
        vec![
            "{10,20}".to_string(),
            "{30,40}".to_string(),
            "{50,NULL}".to_string(),
        ]
    );
}
