//! Change streams over a lake table on a group.
//!
//! A lake table records no feed of its own. Its transaction log is the
//! record, and every member derives the same change records from the same
//! versions, counted into an index of its own. What these prove is that
//! every member counts the same records, that a consume on the leader moves
//! the position in the commit that carried its rows so every member resolves
//! the same count to the same version of its own log, that a bounded
//! consume hands over whole versions, that a reset by position lands on the
//! same place everywhere, and that the outbound pump delivers a lake table's
//! changes once from the leader with every member reading the position it
//! left.
//!
//! Run: `cargo test -p zyron-server --test lake_change_stream_group_test`

mod common;

use std::sync::Arc;
use std::time::Duration;

use common::{Group, Node, WebhookListener, WireClient, positions_on_every_member};
use zyron_server::background::cdc_stream_pump::run_pump_once;

fn table_id_on(node: &Node, table: &str) -> u32 {
    node.catalog
        .get_table(node.schema, table)
        .expect("the table")
        .id
        .0
}

/// The source a member registered for a lake table's feed, which is the
/// member's own transaction log behind an index of what each version yields
fn lake_source(node: &Node, table_id: u32) -> Arc<dyn zyron_cdc::DerivedChangeSource> {
    node._server
        .cdc_registry
        .as_ref()
        .expect("the feeds")
        .derived(table_id)
        .unwrap_or_else(|| panic!("{} registered no source for the lake table", node.name))
}

/// The newest version each member's log holds and the records its index
/// counts through it, in node order
fn counts_on_every_member(group: &Group, table: &str) -> Vec<(u64, u64)> {
    group
        .nodes
        .iter()
        .map(|node| {
            let source = lake_source(node, table_id_on(node, table));
            let latest = source.latest_version();
            let records = source
                .records_at_or_below(latest)
                .unwrap_or_else(|e| panic!("{} could not count its lake versions: {e}", node.name));
            (latest, records)
        })
        .collect()
}

/// Checks that every member holds the same position for a stream, that the
/// count is `consumed`, and that each member's own index counts exactly
/// that many records through the version its position names, which is
/// what a replicated count resolved to a member's own log has to satisfy.
/// A count above zero also names the version the member's index reaches it
/// at, since a consume ends on a whole version
fn assert_position_localized(group: &Group, stream: &str, table: &str, consumed: u64) {
    let positions = positions_on_every_member(group, stream, table);
    for (node, position) in group.nodes.iter().zip(&positions) {
        assert_eq!(
            *position, positions[0],
            "{} holds position {position:?} for {stream} where {} holds {:?}",
            node.name, group.nodes[0].name, positions[0]
        );
        assert_eq!(
            position.1, consumed,
            "{} consumed {} records of {stream}",
            node.name, position.1
        );
        let source = lake_source(node, table_id_on(node, table));
        let counted = source
            .records_at_or_below(position.0)
            .unwrap_or_else(|e| panic!("{} could not count its lake versions: {e}", node.name));
        assert_eq!(
            counted, position.1,
            "{} counts {counted} records through version {}, its position says {}",
            node.name, position.0, position.1
        );
        if consumed > 0 {
            assert_eq!(
                source.version_at_count(consumed),
                position.0,
                "{} resolved count {consumed} to version {} where its own index names version {}",
                node.name,
                position.0,
                source.version_at_count(consumed)
            );
        }
    }
}

async fn run(client: &mut WireClient, sql: &str) {
    let (_, errors) = client.query(sql).await;
    assert!(errors.is_empty(), "`{sql}` was refused: {errors:?}");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn every_member_counts_the_same_lake_changes_and_resolves_the_same_position() {
    let group = Group::start(3).await;
    let leader = group.leader(Duration::from_secs(10)).await;
    let node = &group.nodes[leader];
    let addr = node.serve_wire().await;
    let mut client = WireClient::connect(addr).await;
    run(&mut client, "SET search_path = zyron_test").await;

    run(
        &mut client,
        "CREATE TABLE lake_orders (id BIGINT NOT NULL, total BIGINT NOT NULL) USING ZYRONLAKE",
    )
    .await;
    run(
        &mut client,
        "ALTER TABLE lake_orders SET (change_data_feed = true)",
    )
    .await;
    run(
        &mut client,
        "CREATE CHANGE STREAM lake_orders_s ON TABLE lake_orders",
    )
    .await;
    run(&mut client, "CREATE TABLE landed (id BIGINT, total BIGINT)").await;
    run(
        &mut client,
        "CREATE PIPELINE drain AS (STAGE land (CONSUME CHANGES FROM lake_orders_s MAX ROWS 4 \
         INTO landed))",
    )
    .await;
    group.settle(leader, Duration::from_secs(10)).await;

    // Three commits of three rows each, one lake version per commit
    for batch in 0..3u64 {
        let base = batch * 3;
        run(
            &mut client,
            &format!(
                "INSERT INTO lake_orders VALUES ({}, {}), ({}, {}), ({}, {})",
                base + 1,
                (base + 1) * 10,
                base + 2,
                (base + 2) * 10,
                base + 3,
                (base + 3) * 10
            ),
        )
        .await;
    }
    group.settle(leader, Duration::from_secs(10)).await;

    // Every member holds the same versions and counts the same records
    // through them
    let counts = counts_on_every_member(&group, "lake_orders");
    assert_eq!(
        counts[leader].1, 9,
        "the leader counts every row: {counts:?}"
    );
    for (member, count) in group.nodes.iter().zip(&counts) {
        assert_eq!(
            *count, counts[leader],
            "{} counts {count:?} where the leader counts {:?}",
            member.name, counts[leader]
        );
    }
    assert_position_localized(&group, "lake_orders_s", "lake_orders", 0);

    // A bounded consume takes its bound rounded up to whole versions, and
    // each run's position reaches every member as the count the run
    // consumed to, resolved to that member's own version
    let mut consumed = 0u64;
    let mut runs = 0;
    loop {
        run(&mut client, "RUN PIPELINE drain").await;
        group.settle(leader, Duration::from_secs(10)).await;
        let landed = node.count("landed").await as u64;
        if landed == consumed {
            break;
        }
        runs += 1;
        let took = landed - consumed;
        assert_eq!(
            took % 3,
            0,
            "a run took {took} rows, which is not whole versions"
        );
        assert!(
            took <= 6,
            "a run took {took} rows, past its bound of four rounded up to a whole version"
        );
        consumed = landed;
        assert_position_localized(&group, "lake_orders_s", "lake_orders", consumed);
        assert!(runs <= 4, "the backlog drains in a handful of runs");
    }
    assert_eq!(consumed, 9, "every row landed");
    assert_eq!(
        runs, 2,
        "three rows a version and a bound of four is two versions a run"
    );

    // A reset by position names the same place on every member, the count
    // first and the version resolved from it
    run(
        &mut client,
        "ALTER CHANGE STREAM lake_orders_s RESET TO POSITION 3",
    )
    .await;
    group.settle(leader, Duration::from_secs(10)).await;
    assert_position_localized(&group, "lake_orders_s", "lake_orders", 3);

    // A plain consume from the reset position hands over what lies past it
    run(
        &mut client,
        "INSERT INTO landed SELECT id, total FROM lake_orders_s",
    )
    .await;
    group.settle(leader, Duration::from_secs(10)).await;
    assert_eq!(node.count("landed").await, 15);
    assert_position_localized(&group, "lake_orders_s", "lake_orders", 9);
    for member in &group.nodes {
        assert_eq!(
            member.count("landed").await,
            15,
            "{} holds a different target from the leader",
            member.name
        );
    }

    client.terminate().await;
    group.shutdown().await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn the_leader_delivers_a_lake_tables_changes_once() {
    let group = Group::start(3).await;
    let leader = group.leader(Duration::from_secs(10)).await;
    let node = &group.nodes[leader];
    let listener = WebhookListener::start();
    let addr = node.serve_wire().await;
    let mut client = WireClient::connect(addr).await;
    run(&mut client, "SET search_path = zyron_test").await;

    run(
        &mut client,
        "CREATE TABLE lake_ledger (id BIGINT NOT NULL, amount BIGINT NOT NULL) USING ZYRONLAKE",
    )
    .await;
    run(
        &mut client,
        "ALTER TABLE lake_ledger SET (change_data_feed = true)",
    )
    .await;
    run(
        &mut client,
        "INSERT INTO lake_ledger VALUES (1, 100), (2, 200)",
    )
    .await;
    group.settle(leader, Duration::from_secs(10)).await;

    // The outbound stream starts at the table's current version, which is
    // the count of every record committed so far, the same number on every
    // member
    run(
        &mut client,
        &format!(
            "CREATE CDC STREAM lake_events ON TABLE lake_ledger TO webhook WITH (url = '{}', \
             batch_size = '10')",
            listener.url
        ),
    )
    .await;
    group.settle(leader, Duration::from_secs(10)).await;
    assert_position_localized(&group, "__cdc_lake_events", "lake_ledger", 2);

    run(
        &mut client,
        "INSERT INTO lake_ledger VALUES (3, 300), (4, 400)",
    )
    .await;
    run(&mut client, "INSERT INTO lake_ledger VALUES (5, 500)").await;
    group.settle(leader, Duration::from_secs(10)).await;

    // A follower's pass delivers nothing, the leader's delivers every
    // change committed since the stream was created, once
    let follower = (leader + 1) % group.nodes.len();
    assert_eq!(run_pump_once(&group.nodes[follower]._server).await, 0);
    assert!(listener.bodies().is_empty());
    assert_eq!(run_pump_once(&node._server).await, 3);
    let delivered = listener.bodies().join("\n");
    for amount in [300, 400, 500] {
        assert!(
            delivered.contains(&format!("\"amount\":\"{amount}\"")),
            "the row with amount {amount} reached the sink: {delivered}"
        );
    }
    assert!(!delivered.contains("\"amount\":\"100\""), "{delivered}");
    group.settle(leader, Duration::from_secs(10)).await;
    assert_position_localized(&group, "__cdc_lake_events", "lake_ledger", 5);
    assert_eq!(run_pump_once(&node._server).await, 0);

    // Every member counts the same records through the same versions
    let counts = counts_on_every_member(&group, "lake_ledger");
    assert_eq!(counts[leader].1, 5, "{counts:?}");
    for (member, count) in group.nodes.iter().zip(&counts) {
        assert_eq!(*count, counts[leader], "{} counts {count:?}", member.name);
    }

    client.terminate().await;
    group.shutdown().await;
}

/// The head a branch keeps on a lake table on one member, the version it
/// forked the table at, the newest version it holds and the rows its
/// files hold at that version
fn branch_head_on(node: &Node, table: &str, branch: &str) -> (u64, u64, u64) {
    let paths = zyron_lake::LakePaths::new(node.disk.data_dir(), table_id_on(node, table));
    let log = zyron_lake::open_branch_shared(&paths, branch)
        .unwrap_or_else(|e| panic!("{} holds no head for branch {branch}: {e}", node.name));
    let latest = log.latest_version();
    let rows = log
        .manifest_at(latest)
        .unwrap_or_else(|e| panic!("{} cannot read the branch head: {e}", node.name))
        .entries
        .iter()
        .map(|entry| entry.row_count)
        .sum();
    (log.branch_base(), latest, rows)
}

/// A write on a branch of a lake table reaches every member as a commit on
/// the head the branch keeps on the table there, over the same fork
/// version, with the files the commit added, so every member holds the
/// branch's rows while the table itself stays as it was, and a merge lands
/// the rows on every member's table
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_branch_write_on_a_lake_table_reaches_every_member() {
    let group = Group::start(3).await;
    let leader = group.leader(Duration::from_secs(10)).await;
    let node = &group.nodes[leader];
    let addr = node.serve_wire().await;
    let mut client = WireClient::connect(addr).await;
    run(&mut client, "SET search_path = zyron_test").await;
    run(
        &mut client,
        "CREATE TABLE lake_notes (id BIGINT NOT NULL, body TEXT) USING ZYRONLAKE",
    )
    .await;
    run(&mut client, "INSERT INTO lake_notes VALUES (1, 'main')").await;
    run(&mut client, "CREATE BRANCH dev").await;
    run(&mut client, "USE BRANCH dev").await;
    run(&mut client, "INSERT INTO lake_notes VALUES (2, 'branch')").await;
    run(&mut client, "INSERT INTO lake_notes VALUES (3, 'branch')").await;
    group.settle(leader, Duration::from_secs(10)).await;

    // Every member holds the branch's head at the same versions over the
    // same base, with the branch's rows in it, and the table as it was
    let heads: Vec<(u64, u64, u64)> = group
        .nodes
        .iter()
        .map(|member| branch_head_on(member, "lake_notes", "dev"))
        .collect();
    assert_eq!(
        heads[leader].1,
        heads[leader].0 + 2,
        "two branch commits above the fork: {heads:?}"
    );
    assert_eq!(heads[leader].2, 3, "the branch holds every row: {heads:?}");
    for (member, head) in group.nodes.iter().zip(&heads) {
        assert_eq!(
            *head, heads[leader],
            "{} holds {head:?} where the leader holds {:?}",
            member.name, heads[leader]
        );
        assert_eq!(
            member.count("lake_notes").await,
            1,
            "{} shows the branch's rows on the table",
            member.name
        );
    }

    // A merge lands the branch's rows on every member's table
    run(&mut client, "MERGE BRANCH dev INTO main").await;
    group.settle(leader, Duration::from_secs(10)).await;
    for member in &group.nodes {
        assert_eq!(
            member.count("lake_notes").await,
            3,
            "{} did not land the merge",
            member.name
        );
    }

    client.terminate().await;
    group.shutdown().await;
}
