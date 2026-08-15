//! Search predicates evaluated by the storage scan rather than an index.
//!
//! The planner routes a match, distance or spatial predicate to its index
//! operator when one covers the table and the read is of the current state.
//! Every other case is answered row by row: a table with no such index, and a
//! time-travel read on a store whose index describes only the newest state.
//!
//! The property under test is that the answer does not depend on which route
//! ran. An index changes how fast a query is, never which rows it returns.
//!
//! Run: cargo test -p zyron-wire --test search_without_index_test

mod common;

use common::{create_test_server, exec_ddl, exec_dml, new_session, query_values};
use zyron_executor::column::ScalarValue;

/// The first column of every row as an integer
fn first_ints(rows: &[Vec<ScalarValue>]) -> Vec<i64> {
    let mut out: Vec<i64> = rows
        .iter()
        .map(|row| match row.first() {
            Some(ScalarValue::Int8(v)) => *v as i64,
            Some(ScalarValue::Int16(v)) => *v as i64,
            Some(ScalarValue::Int32(v)) => *v as i64,
            Some(ScalarValue::Int64(v)) => *v,
            other => panic!("expected an integer column, got {other:?}"),
        })
        .collect();
    out.sort_unstable();
    out
}

#[tokio::test]
async fn test_match_against_answers_without_a_fulltext_index() {
    let (server, _schema, _tmp) = create_test_server().await;
    let mut session = new_session();
    exec_ddl(
        &server,
        &mut session,
        "CREATE TABLE nodocs (id INT, body VARCHAR)",
    )
    .await
    .expect("create table");
    exec_dml(
        &server,
        "INSERT INTO nodocs VALUES \
         (1, 'the quick brown fox'), \
         (2, 'a quick brown cat'), \
         (3, 'slow green turtles'), \
         (4, 'the fox and the hound')",
    )
    .await;

    assert_eq!(
        first_ints(
            &query_values(
                &server,
                "SELECT id FROM nodocs WHERE MATCH(body) AGAINST('fox')",
            )
            .await
        ),
        vec![1, 4],
        "a single term selects the rows carrying it"
    );

    assert_eq!(
        first_ints(
            &query_values(
                &server,
                "SELECT id FROM nodocs WHERE MATCH(body) AGAINST('+quick +brown')",
            )
            .await
        ),
        vec![1, 2],
        "both required terms have to be present"
    );

    assert_eq!(
        first_ints(
            &query_values(
                &server,
                "SELECT id FROM nodocs WHERE MATCH(body) AGAINST('+quick -cat')",
            )
            .await
        ),
        vec![1],
        "a negated term removes the row carrying it"
    );

    assert_eq!(
        first_ints(
            &query_values(
                &server,
                "SELECT id FROM nodocs WHERE MATCH(body) AGAINST('\"brown fox\"')",
            )
            .await
        ),
        vec![1],
        "a phrase needs the words adjacent and in order"
    );

    assert!(
        first_ints(
            &query_values(
                &server,
                "SELECT id FROM nodocs WHERE MATCH(body) AGAINST('aardvark')",
            )
            .await
        )
        .is_empty(),
        "a term no row carries matches nothing"
    );
}

/// The same queries against the same rows, once with a full-text index and
/// once without. Both routes must return the same rows.
#[tokio::test]
async fn test_indexed_and_unindexed_match_return_the_same_rows() {
    let (server, _schema, _tmp) = create_test_server().await;
    let mut session = new_session();
    for table in ["withidx", "noidx"] {
        exec_ddl(
            &server,
            &mut session,
            &format!("CREATE TABLE {table} (id INT, body VARCHAR)"),
        )
        .await
        .expect("create table");
    }
    exec_ddl(
        &server,
        &mut session,
        "CREATE FULLTEXT INDEX withidx_body_fts ON withidx (body)",
    )
    .await
    .expect("create fulltext index");

    let rows = [
        "the quick brown fox jumps over the lazy dog",
        "a quick brown cat naps in the sun",
        "slow green turtles crossing the road",
        "the fox and the hound",
        "quick thinking wins races",
    ];
    for table in ["withidx", "noidx"] {
        for (i, body) in rows.iter().enumerate() {
            exec_dml(
                &server,
                &format!("INSERT INTO {table} VALUES ({}, '{body}')", i + 1),
            )
            .await;
        }
    }

    for query in [
        "fox",
        "quick",
        "+quick +brown",
        "+quick -cat",
        "\"brown fox\"",
        "missing",
    ] {
        let indexed = first_ints(
            &query_values(
                &server,
                &format!("SELECT id FROM withidx WHERE MATCH(body) AGAINST('{query}')"),
            )
            .await,
        );
        let scanned = first_ints(
            &query_values(
                &server,
                &format!("SELECT id FROM noidx WHERE MATCH(body) AGAINST('{query}')"),
            )
            .await,
        );
        assert_eq!(indexed, scanned, "query {query:?} depended on the index");
    }
}

#[tokio::test]
async fn test_vector_distance_answers_without_a_vector_index() {
    let (server, _schema, _tmp) = create_test_server().await;
    let mut session = new_session();
    exec_ddl(
        &server,
        &mut session,
        "CREATE TABLE novec (id INT, embedding VECTOR(3))",
    )
    .await
    .expect("create table");
    exec_dml(
        &server,
        "INSERT INTO novec VALUES \
         (1, ARRAY[1.0, 0.0, 0.0]), \
         (2, ARRAY[0.0, 1.0, 0.0]), \
         (3, ARRAY[0.9, 0.1, 0.0])",
    )
    .await;

    // Rows 1 and 3 point nearly the same way, row 2 is orthogonal
    let near = first_ints(
        &query_values(
            &server,
            "SELECT id FROM novec WHERE embedding <=> ARRAY[1.0, 0.0, 0.0] < 0.05",
        )
        .await,
    );
    assert_eq!(
        near,
        vec![1, 3],
        "cosine distance selects the rows pointing the same way"
    );

    let exact = first_ints(
        &query_values(
            &server,
            "SELECT id FROM novec WHERE embedding <-> ARRAY[0.0, 1.0, 0.0] < 0.001",
        )
        .await,
    );
    assert_eq!(exact, vec![2], "L2 distance to a row's own vector is zero");
}

/// A vector column's dimension is what its index is built for and what every
/// distance is computed over, so a row of the wrong length is refused rather
/// than scored against unrelated components.
#[tokio::test]
async fn test_a_vector_of_the_wrong_dimension_is_refused() {
    let (server, _schema, _tmp) = create_test_server().await;
    let mut session = new_session();
    exec_ddl(
        &server,
        &mut session,
        "CREATE TABLE dimcheck (id INT, embedding VECTOR(3))",
    )
    .await
    .expect("create table");

    let err = common::exec_dml_result(&server, "INSERT INTO dimcheck VALUES (1, ARRAY[1.0, 2.0])")
        .await
        .expect_err("a two-component vector does not fit a three-component column")
        .to_string();
    assert!(
        err.contains("3 components") && err.contains("has 2"),
        "the error should name both lengths: {err}"
    );

    exec_dml(
        &server,
        "INSERT INTO dimcheck VALUES (1, ARRAY[1.0, 2.0, 3.0])",
    )
    .await;
    assert_eq!(
        first_ints(&query_values(&server, "SELECT id FROM dimcheck").await),
        vec![1]
    );
}

/// A match predicate under a time-travel read is answered by the storage
/// scan, because a heap delete retires the document and the index no longer
/// describes the rows a past version held.
#[tokio::test]
async fn test_match_under_time_travel_leaves_the_index_for_the_scan() {
    let (server, _schema, _tmp) = create_test_server().await;
    let mut session = new_session();
    exec_ddl(
        &server,
        &mut session,
        "CREATE TABLE tdocs (id INT, body VARCHAR)",
    )
    .await
    .expect("create table");
    exec_ddl(
        &server,
        &mut session,
        "CREATE FULLTEXT INDEX tdocs_body_fts ON tdocs (body)",
    )
    .await
    .expect("create fulltext index");
    exec_dml(
        &server,
        "INSERT INTO tdocs VALUES (1, 'shared marker alpha'), (2, 'shared marker beta')",
    )
    .await;

    // The plan for a current read takes the index; the plan for a version
    // read takes the storage scan and evaluates the predicate row by row
    let current = plan_for(
        &server,
        "SELECT id FROM tdocs WHERE MATCH(body) AGAINST('marker')",
    )
    .await
    .expect("plan a current read");
    assert!(
        format!("{current:?}").contains("FulltextScan"),
        "a current read should take the index"
    );

    // A version past every commit holds every row, so the storage scan has to
    // return exactly what the index returns for the current state. Asserting
    // only the plan shape would pass even if the scan could not evaluate the
    // predicate at all
    let travelled_sql =
        "SELECT id FROM tdocs VERSION AS OF 4294967295 WHERE MATCH(body) AGAINST('marker')";
    let travelled = plan_for(&server, travelled_sql)
        .await
        .expect("plan a version read");
    assert!(
        !format!("{travelled:?}").contains("FulltextScan"),
        "a time-travel read must not take an index describing only the newest state"
    );

    assert_eq!(
        first_ints(&query_values(&server, travelled_sql).await),
        first_ints(
            &query_values(
                &server,
                "SELECT id FROM tdocs WHERE MATCH(body) AGAINST('marker')",
            )
            .await
        ),
        "the version read must answer the match predicate, not only plan differently"
    );
}

async fn plan_for(
    server: &std::sync::Arc<zyron_wire::connection::ServerState>,
    sql: &str,
) -> Result<zyron_planner::physical::PhysicalPlan, String> {
    let stmt = zyron_parser::parse(sql)
        .expect("parse")
        .into_iter()
        .next()
        .expect("one statement");
    zyron_planner::plan(
        &server.catalog,
        zyron_catalog::DatabaseId(1),
        vec!["public".into()],
        stmt,
        None,
    )
    .await
    .map_err(|e| e.to_string())
}
