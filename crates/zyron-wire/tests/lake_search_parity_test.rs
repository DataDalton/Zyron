//! Search indexes over lake tables.
//!
//! A lake row is addressed by data file and ordinal rather than by page and
//! slot, and the document registry holds either shape, so where a row is
//! stored does not decide whether a search can find it.
//!
//! These tests pin the two halves that make that true: a lake write
//! registers its rows under their addresses, and a search scan resolves
//! those addresses back to rows through the table's data files, filtered
//! through the same delete predicates a scan applies.

mod common;

use common::{create_test_server, exec_ddl, exec_dml, new_session, query_rows, query_values};
use zyron_executor::column::ScalarValue;

/// The first column of every row as an integer
fn first_ints(rows: &[Vec<ScalarValue>]) -> Vec<i64> {
    rows.iter()
        .map(|row| match row.first() {
            Some(ScalarValue::Int8(v)) => *v as i64,
            Some(ScalarValue::Int16(v)) => *v as i64,
            Some(ScalarValue::Int32(v)) => *v as i64,
            Some(ScalarValue::Int64(v)) => *v,
            other => panic!("expected an integer column, got {other:?}"),
        })
        .collect()
}

#[tokio::test]
async fn test_full_text_search_finds_rows_of_a_lake_table() {
    let (server, _schema, _tmp) = create_test_server().await;
    let mut session = new_session();

    exec_ddl(
        &server,
        &mut session,
        "CREATE TABLE ldocs (id INT, body VARCHAR) USING ZYRONLAKE",
    )
    .await
    .expect("create lake table");
    exec_ddl(
        &server,
        &mut session,
        "CREATE FULLTEXT INDEX ldocs_body_fts ON ldocs (body)",
    )
    .await
    .expect("create fulltext index");

    for i in 0..20 {
        exec_dml(
            &server,
            &format!("INSERT INTO ldocs VALUES ({i}, 'the quick brown fox number {i}')"),
        )
        .await;
    }
    // One row nothing else shares a term with, so a hit on it cannot come
    // from a scan that ignored the query
    exec_dml(
        &server,
        "INSERT INTO ldocs VALUES (99, 'an entirely separate pangolin')",
    )
    .await;

    assert_eq!(query_rows(&server, "SELECT id FROM ldocs").await, 21);

    let hits = query_rows(
        &server,
        "SELECT id FROM ldocs WHERE MATCH(body) AGAINST('quick')",
    )
    .await;
    assert_eq!(
        hits, 20,
        "a lake row indexed for full text search must come back from it"
    );

    let unique = first_ints(
        &query_values(
            &server,
            "SELECT id FROM ldocs WHERE MATCH(body) AGAINST('pangolin')",
        )
        .await,
    );
    assert_eq!(unique, vec![99], "the term selects exactly its own row");

    let absent = query_rows(
        &server,
        "SELECT id FROM ldocs WHERE MATCH(body) AGAINST('aardvark')",
    )
    .await;
    assert_eq!(absent, 0, "a term no row carries matches nothing");
}

#[tokio::test]
async fn test_a_lake_row_deleted_after_indexing_is_not_returned_by_search() {
    let (server, _schema, _tmp) = create_test_server().await;
    let mut session = new_session();

    exec_ddl(
        &server,
        &mut session,
        "CREATE TABLE lsd (id INT, body VARCHAR) USING ZYRONLAKE",
    )
    .await
    .expect("create lake table");
    exec_ddl(
        &server,
        &mut session,
        "CREATE FULLTEXT INDEX lsd_body_fts ON lsd (body)",
    )
    .await
    .expect("create fulltext index");

    exec_dml(
        &server,
        "INSERT INTO lsd VALUES (1, 'shared marker alpha'), (2, 'shared marker beta')",
    )
    .await;
    assert_eq!(
        query_rows(
            &server,
            "SELECT id FROM lsd WHERE MATCH(body) AGAINST('marker')"
        )
        .await,
        2
    );

    // The delete records a predicate rather than rewriting the file, so the
    // index entry survives it. The fetch has to filter the row out exactly
    // as a scan does, or search resurrects a deleted row
    exec_dml(&server, "DELETE FROM lsd WHERE id = 1").await;
    assert_eq!(query_rows(&server, "SELECT id FROM lsd").await, 1);
    assert_eq!(
        first_ints(
            &query_values(
                &server,
                "SELECT id FROM lsd WHERE MATCH(body) AGAINST('marker')",
            )
            .await
        ),
        vec![2],
        "the deleted row came back through the search index"
    );
}

/// A lake search index only ever gains postings, and lake data files are
/// immutable and named per version, so a hit resolved against the manifest at
/// a past version reads that version's rows. A row deleted since is in a file
/// that version still names, and a row appended since is in no file it names.
#[tokio::test]
async fn test_lake_search_at_a_past_version_reads_that_version() {
    let (server, _schema, _tmp) = create_test_server().await;
    let mut session = new_session();

    exec_ddl(
        &server,
        &mut session,
        "CREATE TABLE lsv (id INT, body VARCHAR) USING ZYRONLAKE",
    )
    .await
    .expect("create lake table");
    exec_ddl(
        &server,
        &mut session,
        "CREATE FULLTEXT INDEX lsv_body_fts ON lsv (body)",
    )
    .await
    .expect("create fulltext index");

    exec_dml(
        &server,
        "INSERT INTO lsv VALUES (1, 'shared marker alpha'), (2, 'shared marker beta')",
    )
    .await;
    // Version 2 holds both rows: version 1 is the table's creation
    let at_two = 2u64;

    exec_dml(&server, "DELETE FROM lsv WHERE id = 1").await;
    exec_dml(&server, "INSERT INTO lsv VALUES (3, 'shared marker gamma')").await;

    let now = first_ints(
        &query_values(
            &server,
            "SELECT id FROM lsv WHERE MATCH(body) AGAINST('marker')",
        )
        .await,
    );
    let mut now_sorted = now.clone();
    now_sorted.sort_unstable();
    assert_eq!(
        now_sorted,
        vec![2, 3],
        "the current search sees the delete and the append"
    );

    let past = first_ints(
        &query_values(
            &server,
            &format!(
                "SELECT id FROM lsv VERSION AS OF {at_two} WHERE MATCH(body) AGAINST('marker')"
            ),
        )
        .await,
    );
    let mut past_sorted = past.clone();
    past_sorted.sort_unstable();
    assert_eq!(
        past_sorted,
        vec![1, 2],
        "the past version holds the row deleted since and not the row appended since"
    );

    // The same rows a plain scan reports at that version, so search and scan
    // do not disagree about what a version contains
    let scanned = first_ints(
        &query_values(
            &server,
            &format!("SELECT id FROM lsv VERSION AS OF {at_two}"),
        )
        .await,
    );
    let mut scanned_sorted = scanned;
    scanned_sorted.sort_unstable();
    assert_eq!(scanned_sorted, past_sorted);
}

/// Searching a branch reads that branch's head, leaving main alone.
#[tokio::test]
async fn test_lake_search_in_a_branch_reads_the_branch_head() {
    let (server, schema, _tmp) = create_test_server().await;
    let mut session = new_session();

    exec_ddl(
        &server,
        &mut session,
        "CREATE TABLE lsb (id INT, body VARCHAR) USING ZYRONLAKE",
    )
    .await
    .expect("create lake table");
    exec_ddl(
        &server,
        &mut session,
        "CREATE FULLTEXT INDEX lsb_body_fts ON lsb (body)",
    )
    .await
    .expect("create fulltext index");
    exec_dml(&server, "INSERT INTO lsb VALUES (1, 'shared marker alpha')").await;

    exec_ddl(&server, &mut session, "CREATE BRANCH work ON lsb")
        .await
        .expect("create branch");
    // Main gains a row the branch forked before, so the two heads differ
    exec_dml(&server, "INSERT INTO lsb VALUES (2, 'shared marker beta')").await;

    let entry = server.catalog.get_table(schema, "lsb").expect("entry");
    let paths = zyron_lake::LakePaths::new(server.disk_manager.data_dir(), entry.id.0);
    assert!(
        zyron_lake::open_branch(&paths, "work").is_ok(),
        "the branch log must open"
    );

    let on_main = first_ints(
        &query_values(
            &server,
            "SELECT id FROM lsb WHERE MATCH(body) AGAINST('marker')",
        )
        .await,
    );
    let mut main_sorted = on_main;
    main_sorted.sort_unstable();
    assert_eq!(main_sorted, vec![1, 2]);

    let on_branch = first_ints(
        &query_values(
            &server,
            "SELECT id FROM lsb IN BRANCH 'work' WHERE MATCH(body) AGAINST('marker')",
        )
        .await,
    );
    assert_eq!(
        on_branch,
        vec![1],
        "the branch forked before the second row, so its head does not hold it"
    );
}

#[tokio::test]
async fn test_search_and_scan_agree_on_a_lake_table_after_an_update() {
    let (server, _schema, _tmp) = create_test_server().await;
    let mut session = new_session();

    exec_ddl(
        &server,
        &mut session,
        "CREATE TABLE lsu (id INT, body VARCHAR) USING ZYRONLAKE",
    )
    .await
    .expect("create lake table");
    exec_ddl(
        &server,
        &mut session,
        "CREATE FULLTEXT INDEX lsu_body_fts ON lsu (body)",
    )
    .await
    .expect("create fulltext index");

    exec_dml(
        &server,
        "INSERT INTO lsu VALUES (1, 'keepable widget'), (2, 'keepable gadget')",
    )
    .await;
    assert_eq!(
        query_rows(
            &server,
            "SELECT id FROM lsu WHERE MATCH(body) AGAINST('keepable')"
        )
        .await,
        2
    );

    // An update removes the old file and writes a new one, so every locator
    // the index held for it goes stale. What must not happen is a row that
    // no longer exists coming back
    exec_dml(
        &server,
        "UPDATE lsu SET body = 'replaced widget' WHERE id = 1",
    )
    .await;
    assert_eq!(query_rows(&server, "SELECT id FROM lsu").await, 2);

    let hits = first_ints(
        &query_values(
            &server,
            "SELECT id FROM lsu WHERE MATCH(body) AGAINST('keepable')",
        )
        .await,
    );
    assert!(
        !hits.contains(&1),
        "row 1 no longer carries the term, so search must not return it"
    );

    // The other half of the same property: the new image has to be
    // findable by the terms it now carries, or an update silently removes
    // a row from every search index on the table
    assert_eq!(
        first_ints(
            &query_values(
                &server,
                "SELECT id FROM lsu WHERE MATCH(body) AGAINST('replaced')",
            )
            .await
        ),
        vec![1],
        "the updated row is not findable by its new text"
    );
}
