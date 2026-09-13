//! The outbound stream pump on a group.
//!
//! An outbound stream's position is the change stream it consumes, and the
//! member that leads the group is the one that delivers. What these prove is
//! that a pass on a follower delivers nothing, a pass on the leader delivers
//! every committed change once and moves the position in a commit the group
//! agrees, so every member reads the same position afterwards, and a stream
//! state written before positions were change streams is carried over into
//! one, at the place its delivery slot had reached, with the slot retired.
//!
//! Run: `cargo test -p zyron-server --test cdc_pump_test`

mod common;

use std::time::Duration;

use common::{Group, WebhookListener as Listener, positions_on_every_member};
use zyron_server::background::cdc_stream_pump::run_pump_once;

/// The consumed count each member records for a change stream over one
/// table, in node order
fn consumed_on_every_member(group: &Group, stream: &str, table: &str) -> Vec<u64> {
    positions_on_every_member(group, stream, table)
        .into_iter()
        .map(|(_, consumed)| consumed)
        .collect()
}

/// What every member's feed holds for a table, rendered row by row in
/// commit order. A member whose feed differs from the leader's would hand a
/// consumer different changes after a failover, so the feeds themselves
/// are compared, not only the positions over them
async fn feed_on_every_member(group: &Group, table: &str) -> Vec<Vec<String>> {
    let mut feeds = Vec::with_capacity(group.nodes.len());
    for node in &group.nodes {
        feeds.push(
            node.rows(&format!(
                "SELECT _change_type, id, total FROM table_changes({table}, 0, LATEST) \
                 ORDER BY _commit_version, _change_ordinal"
            ))
            .await,
        );
    }
    feeds
}

/// Whether a member's slot manager still holds the delivery slot of an
/// outbound stream, in node order
fn slot_held_on_every_member(group: &Group, stream: &str) -> Vec<bool> {
    group
        .nodes
        .iter()
        .map(|node| {
            node._server
                .slot_manager
                .as_ref()
                .map(|slots| slots.get_slot(&format!("{stream}_slot")).is_ok())
                .unwrap_or(false)
        })
        .collect()
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn the_leader_delivers_once_and_every_member_reads_the_position() {
    let group = Group::start(3).await;
    let leader = group.leader(Duration::from_secs(10)).await;
    let node = &group.nodes[leader];
    let listener = Listener::start();

    node.ddl("CREATE TABLE orders (id BIGINT PRIMARY KEY, total BIGINT)")
        .await
        .expect("create the table");
    node.ddl("ALTER TABLE orders SET (change_data_feed = true)")
        .await
        .expect("turn the feed on");
    node.ddl(&format!(
        "CREATE CDC STREAM order_events ON TABLE orders TO webhook WITH (url = '{}', \
         batch_size = '10')",
        listener.url
    ))
    .await
    .expect("create the outbound stream");
    group.settle(leader, Duration::from_secs(10)).await;

    // Every member holds the change stream the outbound stream consumes,
    // positioned at the table's current version, with nothing consumed yet
    assert_eq!(
        consumed_on_every_member(&group, "__cdc_order_events", "orders"),
        vec![0, 0, 0]
    );
    assert_eq!(
        slot_held_on_every_member(&group, "order_events"),
        vec![false, false, false],
        "no slot is created beside the change stream"
    );

    node.write(&[
        "INSERT INTO orders VALUES (1, 10)",
        "INSERT INTO orders VALUES (2, 20)",
        "INSERT INTO orders VALUES (3, 30)",
    ])
    .await
    .expect("insert three rows");
    group.settle(leader, Duration::from_secs(10)).await;

    // A follower's pass delivers nothing, whatever it holds
    let follower = (leader + 1) % group.nodes.len();
    assert_eq!(run_pump_once(&group.nodes[follower]._server).await, 0);
    assert!(listener.bodies().is_empty());

    // The leader's pass delivers every committed change once
    assert_eq!(run_pump_once(&node._server).await, 3);
    let delivered = listener.bodies().join("\n");
    for total in [10, 20, 30] {
        assert!(
            delivered.contains(&format!("\"total\":\"{total}\"")),
            "the row with total {total} reached the sink: {delivered}"
        );
    }

    // The position moved in a commit the group agreed, so every member
    // reads the same count
    group.settle(leader, Duration::from_secs(10)).await;
    assert_eq!(
        consumed_on_every_member(&group, "__cdc_order_events", "orders"),
        vec![3, 3, 3]
    );

    // A second pass finds nothing to deliver, and a later change delivers
    // exactly once more. An update is two records, the row before and the
    // row after, which is what the feed records for it
    assert_eq!(run_pump_once(&node._server).await, 0);
    node.write(&["UPDATE orders SET total = 11 WHERE id = 1"])
        .await
        .expect("update a row");
    group.settle(leader, Duration::from_secs(10)).await;
    assert_eq!(run_pump_once(&node._server).await, 2);
    group.settle(leader, Duration::from_secs(10)).await;
    assert_eq!(
        consumed_on_every_member(&group, "__cdc_order_events", "orders"),
        vec![5, 5, 5]
    );
    assert_eq!(
        listener.bodies().len(),
        2,
        "one post per pass that delivered"
    );
    let delivered = listener.bodies().join("\n");
    assert!(delivered.contains("\"total\":\"11\""), "{delivered}");

    // Every member's feed is the same sequence of the same records, three
    // inserts, then the row before and the row after the update, each
    // decoded from the images the entry carried rather than from what the
    // member's own operators happened to see
    let feeds = feed_on_every_member(&group, "orders").await;
    assert_eq!(
        feeds[leader].len(),
        5,
        "the leader's feed holds every record: {:?}",
        feeds[leader]
    );
    for (member, feed) in group.nodes.iter().zip(&feeds) {
        assert_eq!(
            feed, &feeds[leader],
            "{} holds a different feed from the leader",
            member.name
        );
    }

    group.shutdown().await;
}

/// A stream state written before positions were change streams names the
/// change stream without holding it, and records its delivery in a slot.
/// The pump creates the stream through the group at the count the slot's
/// version names, so the changes the slot had not yet delivered are what
/// deliver, once, and every member retires its slot
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_delivery_slot_is_carried_over_into_a_change_stream() {
    let group = Group::start(3).await;
    let leader = group.leader(Duration::from_secs(10)).await;
    let node = &group.nodes[leader];
    let listener = Listener::start();

    node.ddl("CREATE TABLE ledger (id BIGINT PRIMARY KEY, amount BIGINT)")
        .await
        .expect("create the table");
    node.ddl("ALTER TABLE ledger SET (change_data_feed = true)")
        .await
        .expect("turn the feed on");
    node.write(&[
        "INSERT INTO ledger VALUES (1, 100)",
        "INSERT INTO ledger VALUES (2, 200)",
    ])
    .await
    .expect("two rows before the slot's position");
    node.write(&[
        "INSERT INTO ledger VALUES (3, 300)",
        "INSERT INTO ledger VALUES (4, 400)",
    ])
    .await
    .expect("two rows past the slot's position");
    group.settle(leader, Duration::from_secs(10)).await;

    // The state a version 1 writer left, migrated. The outbound stream
    // names its change stream, and each member still holds the slot the
    // old writer delivered from, positioned after the first two rows on
    // the leader
    let table_id = node
        .catalog
        .get_table(node.schema, "ledger")
        .expect("the table")
        .id
        .0;
    let feed = node
        ._server
        .cdc_registry
        .as_ref()
        .expect("the feeds")
        .get_feed(table_id)
        .expect("the ledger feed");
    let second = feed
        .cursor_at_count(2)
        .map(|(version, _)| version)
        .expect("the version of the second record");
    for member in &group.nodes {
        let manager = member
            ._server
            .cdc_stream_manager
            .as_ref()
            .expect("the outbound stream manager");
        manager
            .create_stream(zyron_cdc::cdc_stream::CdcOutputStream {
                name: "ledger_events".to_string(),
                table_id,
                change_stream: zyron_cdc::cdc_stream::implicit_change_stream_name("ledger_events"),
                sink: zyron_cdc::cdc_stream::CdcSinkConfig::Webhook {
                    url: listener.url.clone(),
                    headers: Vec::new(),
                    batch_size: 10,
                },
                decoder_plugin: zyron_cdc::decoder::DecoderPlugin::Debezium,
                filter: None,
                include_columns: None,
                batch_size: 10,
                batch_interval_ms: 100,
                active: true,
                retry_policy: zyron_cdc::cdc_stream::StreamRetryPolicy::default(),
            })
            .expect("register the migrated stream state");
        let slots = member
            ._server
            .slot_manager
            .as_ref()
            .expect("the slot manager");
        slots
            .create_slot(
                "ledger_events_slot",
                zyron_cdc::decoder::DecoderPlugin::Debezium,
                Some(vec![table_id]),
            )
            .expect("the old writer's slot");
        slots
            .advance_slot("ledger_events_slot", zyron_wal::record::Lsn(second))
            .expect("the slot's position");
    }
    assert_eq!(
        slot_held_on_every_member(&group, "ledger_events"),
        vec![true, true, true]
    );

    // The leader's first pass creates the stream through the group at the
    // slot's count and delivers what the slot had not yet delivered
    assert_eq!(run_pump_once(&node._server).await, 2);
    let delivered = listener.bodies().join("\n");
    assert!(!delivered.contains("\"amount\":\"100\""), "{delivered}");
    assert!(!delivered.contains("\"amount\":\"200\""), "{delivered}");
    assert!(delivered.contains("\"amount\":\"300\""), "{delivered}");
    assert!(delivered.contains("\"amount\":\"400\""), "{delivered}");
    group.settle(leader, Duration::from_secs(10)).await;
    assert_eq!(
        consumed_on_every_member(&group, "__cdc_ledger_events", "ledger"),
        vec![4, 4, 4]
    );

    // The leader retired its slot as it carried the record over, and each
    // follower retires its own on its next pass
    for member in &group.nodes {
        run_pump_once(&member._server).await;
    }
    assert_eq!(
        slot_held_on_every_member(&group, "ledger_events"),
        vec![false, false, false]
    );

    group.shutdown().await;
}
