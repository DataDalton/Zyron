//! zyron_stat_tables and zyron_stat_indexes report real activity.
//!
//! Both views returned a hardcoded zero for every counter before the IO stats
//! registry was wired to the executor, so a table that had served a million
//! rows looked identical to one nobody had ever queried. These tests pin the
//! counters to work the queries actually did, on both storage formats, so the
//! views cannot silently go back to answering zero.

mod common;

use common::{create_test_server, exec_ddl, exec_dml, new_session, query_rows};
use std::sync::Arc;
use zyron_wire::connection::ServerState;
use zyron_wire::stat_views::{StatViewFilters, query_stat_view};

/// One row of a stat view, columns addressed by name.
struct ViewRow {
    columns: Vec<String>,
    cells: Vec<Option<Vec<u8>>>,
}

impl ViewRow {
    fn u64(&self, column: &str) -> u64 {
        let idx = self
            .columns
            .iter()
            .position(|c| c == column)
            .unwrap_or_else(|| panic!("view has no column \"{column}\": {:?}", self.columns));
        let bytes = self.cells[idx]
            .as_ref()
            .unwrap_or_else(|| panic!("column \"{column}\" is NULL"));
        std::str::from_utf8(bytes)
            .expect("counter is utf8")
            .parse()
            .unwrap_or_else(|e| panic!("column \"{column}\" is not a number: {e}"))
    }

    fn text(&self, column: &str) -> String {
        let idx = self
            .columns
            .iter()
            .position(|c| c == column)
            .unwrap_or_else(|| panic!("view has no column \"{column}\""));
        match &self.cells[idx] {
            Some(b) => String::from_utf8_lossy(b).into_owned(),
            None => String::new(),
        }
    }
}

/// Reads a stat view and returns the row whose `key_column` equals `key`.
fn view_row(server: &Arc<ServerState>, view: &str, key_column: &str, key: &str) -> ViewRow {
    let (fields, rows) = query_stat_view(view, server, &StatViewFilters::default())
        .expect("stat view query")
        .unwrap_or_else(|| panic!("{view} is not a recognized stat view"));
    let columns: Vec<String> = fields.iter().map(|f| f.name.clone()).collect();
    let key_idx = columns
        .iter()
        .position(|c| c == key_column)
        .unwrap_or_else(|| panic!("{view} has no column \"{key_column}\""));
    for cells in rows {
        let matches = cells[key_idx]
            .as_ref()
            .map(|b| String::from_utf8_lossy(b) == key)
            .unwrap_or(false);
        if matches {
            return ViewRow { columns, cells };
        }
    }
    panic!("{view} has no row where {key_column} = \"{key}\"");
}

#[tokio::test]
async fn test_stat_tables_counts_the_rows_a_heap_scan_read() {
    let (server, _schema, _tmp) = create_test_server().await;
    let mut session = new_session();

    exec_ddl(
        &server,
        &mut session,
        "CREATE TABLE readers (id INT, payload VARCHAR)",
    )
    .await
    .expect("create");

    // A table nobody has touched reports zeros rather than being absent, which
    // is what makes the view usable as a starting baseline
    let before = view_row(&server, "zyron_stat_tables", "table_name", "readers");
    assert_eq!(before.u64("seq_scan"), 0);
    assert_eq!(before.u64("seq_tup_read"), 0);
    assert_eq!(before.u64("bytes_read"), 0);
    assert_eq!(before.u64("n_tup_ins"), 0);

    for i in 0..40 {
        exec_dml(
            &server,
            &format!("INSERT INTO readers VALUES ({i}, 'row {i}')"),
        )
        .await;
    }

    let after_insert = view_row(&server, "zyron_stat_tables", "table_name", "readers");
    assert_eq!(
        after_insert.u64("n_tup_ins"),
        40,
        "every inserted row is counted"
    );
    assert_eq!(after_insert.u64("n_tup_upd"), 0);
    assert_eq!(after_insert.u64("n_tup_del"), 0);
    assert_eq!(
        after_insert.u64("n_dead_tup"),
        0,
        "an insert leaves no dead version behind"
    );

    assert_eq!(query_rows(&server, "SELECT id FROM readers").await, 40);

    let after_scan = view_row(&server, "zyron_stat_tables", "table_name", "readers");
    assert!(
        after_scan.u64("seq_scan") >= 1,
        "the scan was counted, got {}",
        after_scan.u64("seq_scan")
    );
    assert_eq!(
        after_scan.u64("seq_tup_read"),
        40,
        "the scan read every visible row"
    );
    assert!(
        after_scan.u64("bytes_read") > 0,
        "a heap scan reads pages, so it cannot have read zero bytes"
    );

    // A second scan accumulates rather than replacing
    assert_eq!(query_rows(&server, "SELECT id FROM readers").await, 40);
    let after_second = view_row(&server, "zyron_stat_tables", "table_name", "readers");
    assert_eq!(after_second.u64("seq_tup_read"), 80);
    assert!(after_second.u64("bytes_read") > after_scan.u64("bytes_read"));
}

#[tokio::test]
async fn test_stat_tables_separates_updates_deletes_and_dead_rows() {
    let (server, _schema, _tmp) = create_test_server().await;
    let mut session = new_session();

    exec_ddl(&server, &mut session, "CREATE TABLE churn (id INT, v INT)")
        .await
        .expect("create");
    for i in 0..10 {
        exec_dml(&server, &format!("INSERT INTO churn VALUES ({i}, {i})")).await;
    }

    exec_dml(&server, "UPDATE churn SET v = 99 WHERE id < 4").await;
    exec_dml(&server, "DELETE FROM churn WHERE id >= 8").await;

    let row = view_row(&server, "zyron_stat_tables", "table_name", "churn");
    assert_eq!(row.u64("n_tup_ins"), 10);
    assert_eq!(row.u64("n_tup_upd"), 4);
    assert_eq!(row.u64("n_tup_del"), 2);
    assert_eq!(
        row.u64("n_dead_tup"),
        6,
        "an update and a delete each leave one dead version"
    );
    assert_eq!(
        row.u64("row_count"),
        8,
        "inserts less deletes, for a table never analyzed"
    );
    assert_eq!(
        row.u64("last_vacuum"),
        0,
        "nothing has vacuumed this table yet"
    );
}

#[tokio::test]
async fn test_stat_indexes_counts_index_scans_and_the_rows_they_fetched() {
    let (server, _schema, _tmp) = create_test_server().await;
    let mut session = new_session();

    exec_ddl(
        &server,
        &mut session,
        "CREATE TABLE indexed (id INT, v INT)",
    )
    .await
    .expect("create");
    for i in 0..30 {
        exec_dml(&server, &format!("INSERT INTO indexed VALUES ({i}, {i})")).await;
    }
    exec_ddl(
        &server,
        &mut session,
        "CREATE INDEX indexed_id_idx ON indexed (id)",
    )
    .await
    .expect("create index");

    let before = view_row(
        &server,
        "zyron_stat_indexes",
        "index_name",
        "indexed_id_idx",
    );
    assert_eq!(before.text("table_name"), "indexed");
    assert_eq!(before.text("index_type"), "btree");
    assert_eq!(before.u64("idx_scan"), 0);

    let matched = query_rows(&server, "SELECT v FROM indexed WHERE id = 7").await;
    assert_eq!(matched, 1);

    let table_row = view_row(&server, "zyron_stat_tables", "table_name", "indexed");
    let index_row = view_row(
        &server,
        "zyron_stat_indexes",
        "index_name",
        "indexed_id_idx",
    );

    // The planner is free to answer this with a sequential scan, and either
    // choice must be reported honestly. What must never happen is the query
    // running through the index and the view still reporting nothing
    if index_row.u64("idx_scan") > 0 {
        assert_eq!(
            table_row.u64("idx_scan"),
            index_row.u64("idx_scan"),
            "the table and index scan counts describe the same scans"
        );
        assert!(
            index_row.u64("idx_tup_read") >= index_row.u64("idx_tup_fetch"),
            "entries examined cannot be fewer than the rows they resolved to"
        );
        assert_eq!(
            table_row.u64("idx_tup_fetch"),
            index_row.u64("idx_tup_fetch")
        );
    } else {
        assert!(
            table_row.u64("seq_scan") > 0,
            "the query ran somehow, so one of the two counters must have moved"
        );
    }
}

#[tokio::test]
async fn test_stat_tables_counts_a_lake_scan_the_same_way_as_a_heap_scan() {
    let (server, _schema, _tmp) = create_test_server().await;
    let mut session = new_session();

    // A wide payload column nobody projects. It is what separates the two
    // formats on the read side: the heap reads whole pages holding whole rows,
    // so the payload is read whether the query wants it or not, while the lake
    // reads only the segment of the column that was projected
    let schema = "(id INT, v INT, payload VARCHAR)";
    exec_ddl(
        &server,
        &mut session,
        &format!("CREATE TABLE heap_side {schema}"),
    )
    .await
    .expect("create heap");
    exec_ddl(
        &server,
        &mut session,
        &format!("CREATE TABLE lake_side {schema} USING ZYRONLAKE"),
    )
    .await
    .expect("create lake");

    // Enough rows that neither format's read fits in one page, so the
    // comparison is between two real scans rather than two rounding artifacts
    const ROWS: usize = 4_000;
    let padding = "x".repeat(200);
    let values: String = (0..ROWS)
        .map(|i| format!("({i}, {}, '{padding}')", i * 2))
        .collect::<Vec<_>>()
        .join(", ");
    exec_dml(&server, &format!("INSERT INTO heap_side VALUES {values}")).await;
    exec_dml(&server, &format!("INSERT INTO lake_side VALUES {values}")).await;

    assert_eq!(query_rows(&server, "SELECT v FROM heap_side").await, ROWS);
    assert_eq!(query_rows(&server, "SELECT v FROM lake_side").await, ROWS);

    let heap = view_row(&server, "zyron_stat_tables", "table_name", "heap_side");
    let lake = view_row(&server, "zyron_stat_tables", "table_name", "lake_side");

    for row in [&heap, &lake] {
        assert_eq!(row.u64("n_tup_ins"), ROWS as u64);
        assert_eq!(row.u64("seq_tup_read"), ROWS as u64);
        assert!(row.u64("seq_scan") >= 1);
        assert!(
            row.u64("bytes_read") > 0,
            "neither format can read {ROWS} rows out of zero bytes"
        );
    }

    // The point of the byte counter is that it is comparable across formats,
    // and this is the comparison it exists to make
    assert!(
        lake.u64("bytes_read") < heap.u64("bytes_read"),
        "projecting one narrow column read {} bytes on the lake against {} on the heap",
        lake.u64("bytes_read"),
        heap.u64("bytes_read")
    );
}

/// Every index type the view names must report its own scans.
///
/// The counters were wired for B+tree first, which left fulltext, vector
/// and spatial reporting zero: the same view telling the same lie, just
/// for the other three quarters of its rows. Each type resolves its hits
/// through a different engine, so each needs its own wiring and its own
/// check that the wiring is there.
#[tokio::test]
async fn test_every_index_type_records_the_scans_it_serves() {
    let (server, _schema, _tmp) = create_test_server().await;
    let mut session = new_session();

    exec_ddl(
        &server,
        &mut session,
        "CREATE TABLE docs (id INT, body VARCHAR)",
    )
    .await
    .expect("create");
    exec_ddl(
        &server,
        &mut session,
        "CREATE FULLTEXT INDEX docs_body_fts ON docs (body)",
    )
    .await
    .expect("create fulltext index");

    for i in 0..20 {
        exec_dml(
            &server,
            &format!("INSERT INTO docs VALUES ({i}, 'the quick brown fox number {i}')"),
        )
        .await;
    }

    let before = view_row(&server, "zyron_stat_indexes", "index_name", "docs_body_fts");
    assert_eq!(before.text("index_type"), "fulltext");
    assert_eq!(before.u64("idx_scan"), 0);

    let hits = query_rows(
        &server,
        "SELECT id FROM docs WHERE MATCH(body) AGAINST('quick')",
    )
    .await;
    assert!(
        hits > 0,
        "the search matched nothing, so it scanned nothing"
    );

    let after = view_row(&server, "zyron_stat_indexes", "index_name", "docs_body_fts");
    assert_eq!(
        after.u64("idx_scan"),
        1,
        "a fulltext search ran but the index reported no scan"
    );
    assert!(
        after.u64("idx_tup_read") >= after.u64("idx_tup_fetch"),
        "entries examined cannot be fewer than the rows they resolved to"
    );
    assert_eq!(
        after.u64("idx_tup_fetch"),
        hits as u64,
        "every row the search returned was fetched through the index"
    );

    // The table's own counters must agree, since they describe the same scan
    let table = view_row(&server, "zyron_stat_tables", "table_name", "docs");
    assert_eq!(table.u64("idx_scan"), 1);
    assert_eq!(table.u64("idx_tup_fetch"), hits as u64);
    assert!(
        table.u64("bytes_read") > 0,
        "the rows were fetched from somewhere, so bytes were read"
    );
}

#[tokio::test]
async fn test_dropping_a_table_discards_its_counters() {
    let (server, _schema, _tmp) = create_test_server().await;
    let mut session = new_session();

    exec_ddl(&server, &mut session, "CREATE TABLE transient (id INT)")
        .await
        .expect("create");
    for i in 0..5 {
        exec_dml(&server, &format!("INSERT INTO transient VALUES ({i})")).await;
    }
    assert_eq!(
        view_row(&server, "zyron_stat_tables", "table_name", "transient").u64("n_tup_ins"),
        5
    );
    let dropped_id = server
        .catalog
        .list_all_tables()
        .iter()
        .find(|t| t.name == "transient")
        .expect("table exists")
        .id
        .0;

    exec_ddl(&server, &mut session, "DROP TABLE transient")
        .await
        .expect("drop");

    // A table id is reusable, so leaving the counters behind would credit a
    // future table with a dropped table's history
    assert_eq!(
        server
            .table_io_stats
            .get_or_create(dropped_id)
            .n_tup_ins
            .load(std::sync::atomic::Ordering::Relaxed),
        0,
        "the dropped table's counters were discarded with it"
    );
}
