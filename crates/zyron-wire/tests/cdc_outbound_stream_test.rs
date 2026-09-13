//! Outbound CDC streams hold their delivery position in a change stream.
//!
//! An outbound stream created with ON TABLE gets a change stream named after
//! it, positioned at the table's current version, and one created with FROM
//! CHANGE STREAM consumes the stream it names. Either way the position is
//! the one `zyron_sys.cdc.change_streams` shows, and no replication slot is
//! created beside it.
//!
//! Run: cargo test -p zyron-wire --test cdc_outbound_stream_test -- --nocapture

use std::sync::Arc;

use zyron_wire::connection::ServerState;
use zyron_wire::session::Session;

mod common;
use common::*;

async fn seeded(server: &Arc<ServerState>, session: &mut Option<Session>, storage: Storage) {
    exec_ddl(
        server,
        session,
        &storage.create("CREATE TABLE orders (id BIGINT PRIMARY KEY, total BIGINT)"),
    )
    .await
    .expect("create orders");
    exec_ddl(
        server,
        session,
        "ALTER TABLE orders SET (change_data_feed = true)",
    )
    .await
    .expect("turn the feed on");
    for id in 1..=3 {
        exec_dml(server, &format!("INSERT INTO orders VALUES ({id}, {id})")).await;
    }
}

/// One system view, rows keyed by a column, cells read as text
async fn view(server: &Arc<ServerState>, name: &str) -> (Vec<String>, Vec<Vec<String>>) {
    let (fields, rows) = zyron_wire::system_views::query_system_view(
        name,
        server,
        &zyron_wire::system_views::SystemViewFilters::default(),
    )
    .await
    .expect("the view answers")
    .unwrap_or_else(|| panic!("{name} is not a system view"));
    let columns: Vec<String> = fields.iter().map(|f| f.name.clone()).collect();
    let rows = rows
        .into_iter()
        .map(|cells| {
            cells
                .into_iter()
                .map(|cell| {
                    cell.map(|b| String::from_utf8_lossy(&b).into_owned())
                        .unwrap_or_default()
                })
                .collect()
        })
        .collect();
    (columns, rows)
}

fn cell<'a>(columns: &[String], row: &'a [String], name: &str) -> &'a str {
    let at = columns
        .iter()
        .position(|c| c == name)
        .unwrap_or_else(|| panic!("no column {name} in {columns:?}"));
    &row[at]
}

/// The row of a view whose `key` column reads `value`
async fn row_of(
    server: &Arc<ServerState>,
    view_name: &str,
    key: &str,
    value: &str,
) -> Option<(Vec<String>, Vec<String>)> {
    let (columns, rows) = view(server, view_name).await;
    let row = rows.into_iter().find(|r| cell(&columns, r, key) == value)?;
    Some((columns, row))
}

const WEBHOOK: &str = "TO webhook WITH (url = 'http://127.0.0.1:9/changes', batch_size = '10')";

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn an_outbound_stream_holds_its_position_in_a_change_stream() {
    for storage in Storage::BOTH {
        an_outbound_stream_holds_its_position_in_a_change_stream_on(storage).await;
    }
}

async fn an_outbound_stream_holds_its_position_in_a_change_stream_on(storage: Storage) {
    println!("{storage}");
    let (server, _schema, _tmp) = create_test_server_with_cdc().await;
    let mut session = new_session();
    seeded(&server, &mut session, storage).await;

    exec_ddl(
        &server,
        &mut session,
        &format!("CREATE CDC STREAM order_events ON TABLE orders {WEBHOOK}"),
    )
    .await
    .expect("create the outbound stream");

    // The position lives in a change stream named after the outbound
    // stream, at the table's current version, so what delivers is what
    // changes from here on
    let (columns, row) = row_of(
        &server,
        "zyron_sys.cdc.change_streams",
        "stream",
        "__cdc_order_events",
    )
    .await
    .expect("the change stream is listed");
    assert_eq!(cell(&columns, &row, "consumed"), "orders=3");
    assert_eq!(cell(&columns, &row, "pending_rows"), "0");

    let (columns, row) = row_of(
        &server,
        "zyron_sys.stat.cdc_streams",
        "name",
        "order_events",
    )
    .await
    .expect("the outbound stream is listed");
    assert_eq!(cell(&columns, &row, "change_stream"), "__cdc_order_events");

    // No replication slot was created beside the change stream, because
    // the change stream is the one position the outbound stream has
    let (_, slots) = view(&server, "zyron_sys.stat.replication_slots").await;
    assert!(
        slots.is_empty(),
        "no slot beside the change stream: {slots:?}"
    );

    // A change after creation is what the stream has pending
    exec_dml(&server, "INSERT INTO orders VALUES (4, 4)").await;
    let (columns, row) = row_of(
        &server,
        "zyron_sys.cdc.change_streams",
        "stream",
        "__cdc_order_events",
    )
    .await
    .expect("the change stream is listed");
    assert_eq!(cell(&columns, &row, "pending_rows"), "1");

    // Dropping the outbound stream drops the change stream it created
    exec_ddl(&server, &mut session, "DROP CDC STREAM order_events")
        .await
        .expect("drop the outbound stream");
    assert!(
        row_of(
            &server,
            "zyron_sys.cdc.change_streams",
            "stream",
            "__cdc_order_events",
        )
        .await
        .is_none(),
        "the change stream created for the outbound stream goes with it"
    );
    assert!(
        row_of(
            &server,
            "zyron_sys.stat.cdc_streams",
            "name",
            "order_events"
        )
        .await
        .is_none()
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn an_outbound_stream_consumes_the_change_stream_it_names() {
    for storage in Storage::BOTH {
        an_outbound_stream_consumes_the_change_stream_it_names_on(storage).await;
    }
}

async fn an_outbound_stream_consumes_the_change_stream_it_names_on(storage: Storage) {
    println!("{storage}");
    let (server, _schema, _tmp) = create_test_server_with_cdc().await;
    let mut session = new_session();
    seeded(&server, &mut session, storage).await;
    exec_ddl(
        &server,
        &mut session,
        "CREATE CHANGE STREAM order_changes ON TABLE orders AT VERSION 0",
    )
    .await
    .expect("create the change stream");

    exec_ddl(
        &server,
        &mut session,
        &format!("CREATE CDC STREAM order_feed FROM CHANGE STREAM order_changes {WEBHOOK}"),
    )
    .await
    .expect("create the outbound stream over the named change stream");

    let (columns, row) = row_of(&server, "zyron_sys.stat.cdc_streams", "name", "order_feed")
        .await
        .expect("the outbound stream is listed");
    assert_eq!(cell(&columns, &row, "change_stream"), "order_changes");
    // No stream of the outbound stream's own was created
    assert!(
        row_of(
            &server,
            "zyron_sys.cdc.change_streams",
            "stream",
            "__cdc_order_feed"
        )
        .await
        .is_none()
    );
    // The named stream keeps the position it had, which is where delivery
    // starts, so the rows the stream was created behind are what delivers
    let (columns, row) = row_of(
        &server,
        "zyron_sys.cdc.change_streams",
        "stream",
        "order_changes",
    )
    .await
    .expect("the named stream is listed");
    assert_eq!(cell(&columns, &row, "pending_rows"), "3");

    // Dropping the outbound stream leaves the stream the operator named
    exec_ddl(&server, &mut session, "DROP CDC STREAM order_feed")
        .await
        .expect("drop the outbound stream");
    assert!(
        row_of(
            &server,
            "zyron_sys.cdc.change_streams",
            "stream",
            "order_changes"
        )
        .await
        .is_some(),
        "a change stream the operator named is theirs to keep"
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn an_outbound_stream_is_refused_where_it_could_not_deliver() {
    let (server, _schema, _tmp) = create_test_server_with_cdc().await;
    let mut session = new_session();
    seeded(&server, &mut session, Storage::Heap).await;
    exec_ddl(
        &server,
        &mut session,
        "CREATE TABLE plain (id BIGINT PRIMARY KEY, v BIGINT)",
    )
    .await
    .expect("create a table with no feed");
    exec_ddl(
        &server,
        &mut session,
        "CREATE TABLE other (id BIGINT PRIMARY KEY, v BIGINT)",
    )
    .await
    .expect("create another table");
    exec_ddl(
        &server,
        &mut session,
        "ALTER TABLE other SET (change_data_feed = true)",
    )
    .await
    .expect("turn the feed on");

    // A table with no feed has nothing to deliver
    let refused = exec_ddl(
        &server,
        &mut session,
        &format!("CREATE CDC STREAM no_feed ON TABLE plain {WEBHOOK}"),
    )
    .await
    .expect_err("a table without a feed is refused");
    assert!(refused.contains("change data feed"), "{refused}");

    // A multi-table change stream is refused, an outbound stream delivers
    // one table's changes
    exec_ddl(
        &server,
        &mut session,
        "CREATE CHANGE STREAM both ON TABLES (orders, other)",
    )
    .await
    .expect("create a multi-table stream");
    let refused = exec_ddl(
        &server,
        &mut session,
        &format!("CREATE CDC STREAM two_tables FROM CHANGE STREAM both {WEBHOOK}"),
    )
    .await
    .expect_err("a multi-table stream is refused");
    assert!(refused.contains("reads 2 tables"), "{refused}");

    // A change stream that does not exist is refused by name
    let refused = exec_ddl(
        &server,
        &mut session,
        &format!("CREATE CDC STREAM missing FROM CHANGE STREAM nowhere {WEBHOOK}"),
    )
    .await
    .expect_err("a missing stream is refused");
    assert!(refused.contains("does not exist"), "{refused}");

    // A change stream already holding the name an outbound stream would
    // create is refused rather than shared
    exec_ddl(
        &server,
        &mut session,
        "CREATE CHANGE STREAM __cdc_taken ON TABLE orders",
    )
    .await
    .expect("create a stream under the implicit name");
    let refused = exec_ddl(
        &server,
        &mut session,
        &format!("CREATE CDC STREAM taken ON TABLE orders {WEBHOOK}"),
    )
    .await
    .expect_err("a taken implicit name is refused");
    assert!(refused.contains("already exists"), "{refused}");

    // Nothing above left an outbound stream or a slot behind
    let (_, streams) = view(&server, "zyron_sys.stat.cdc_streams").await;
    assert!(streams.is_empty(), "{streams:?}");
    let (_, slots) = view(&server, "zyron_sys.stat.replication_slots").await;
    assert!(slots.is_empty(), "{slots:?}");
}

/// The outbound stream's position is the change stream it consumes and
/// nothing else. The definition, the pump and the sinks declare no
/// position field of their own, so a second place progress could be
/// recorded does not exist to drift from the first
#[test]
fn no_second_checkpoint_exists_in_the_outbound_path() {
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("the crates directory");
    let sources = [
        "zyron-cdc/src/cdc_stream.rs",
        "zyron-cdc/src/sink_io.rs",
        "zyron-server/src/background/cdc_stream_pump.rs",
    ];
    let position_names = [
        "last_lsn",
        "checkpoint",
        "checkpoint_lsn",
        "confirmed_lsn",
        "delivered_lsn",
        "position",
        "last_delivered",
        "last_position",
    ];
    // A struct field declaration, `name:` at the start of a line with an
    // optional visibility in front of it
    let declares_position = |line: &str| {
        let line = line.trim_start();
        let line = line
            .strip_prefix("pub(crate) ")
            .or_else(|| line.strip_prefix("pub "))
            .unwrap_or(line);
        position_names.iter().any(|name| {
            line.strip_prefix(name)
                .is_some_and(|rest| rest.trim_start().starts_with(':'))
        })
    };
    for source in sources {
        let text = std::fs::read_to_string(root.join(source))
            .unwrap_or_else(|e| panic!("{source} reads: {e}"));
        for (number, line) in text.lines().enumerate() {
            assert!(
                !declares_position(line),
                "{source}:{} declares a position of its own: {line}",
                number + 1
            );
        }
    }
    let definition = std::fs::read_to_string(root.join("zyron-cdc/src/cdc_stream.rs"))
        .expect("the outbound stream definition");
    assert!(
        definition.contains("pub change_stream: String"),
        "the outbound stream names the change stream it consumes"
    );
}
