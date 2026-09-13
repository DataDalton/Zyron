//! Background writers on a group.
//!
//! A schedule's body and a retention cycle's deletes are writes the leader
//! makes on its own initiative. What these prove is that the leader commits
//! them through the group, so every member holds the rows the body wrote
//! and the rows retention expired, and that a schedule's run itself travels
//! with the body's commit, so every member records the same last run and
//! next run and a member elected later continues the schedule from the next
//! period rather than running the body again at once. A body that fails
//! still moves every member past the period it failed in.
//!
//! Run: `cargo test -p zyron-server --test background_group_test`

mod common;

use std::sync::Arc;
use std::time::Duration;

use common::{Group, Node, WireClient};
use zyron_server::background::retention::{RetentionStats, run_retention_cycle};
use zyron_wire::ddl_dispatch::run_due_schedules;

/// An instant far past any schedule's creation, so a schedule is due
const FAR_FUTURE: i64 = 10_000_000_000_000_000;

async fn run(client: &mut WireClient, sql: &str) {
    let (_, errors) = client.query(sql).await;
    assert!(errors.is_empty(), "`{sql}` was refused: {errors:?}");
}

/// The instants each member records for a schedule, in node order
fn schedule_on_every_member(group: &Group, name: &str) -> Vec<(Option<i64>, Option<i64>)> {
    group
        .nodes
        .iter()
        .map(|node| {
            let entry = node
                .catalog
                .get_schedule_by_name(name)
                .unwrap_or_else(|| panic!("{} holds no schedule {name}", node.name));
            (entry.last_run, entry.next_run)
        })
        .collect()
}

async fn count_on_every_member(group: &Group, table: &str) -> Vec<i64> {
    let mut counts = Vec::with_capacity(group.nodes.len());
    for node in &group.nodes {
        counts.push(node.count(table).await);
    }
    counts
}

fn assert_same_everywhere<T: PartialEq + std::fmt::Debug>(group: &Group, values: &[T], what: &str) {
    for (node, value) in group.nodes.iter().zip(values) {
        assert_eq!(
            *value, values[0],
            "{} holds {what} {value:?} where {} holds {:?}",
            node.name, group.nodes[0].name, values[0]
        );
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_schedules_run_reaches_every_member_with_the_rows_it_wrote() {
    let group = Group::start(3).await;
    let leader = group.leader(Duration::from_secs(10)).await;
    let node: &Arc<Node> = &group.nodes[leader];
    let addr = node.serve_wire().await;
    let mut client = WireClient::connect(addr).await;
    run(&mut client, "SET search_path = zyron_test").await;

    run(&mut client, "CREATE TABLE landed (id BIGINT)").await;
    run(
        &mut client,
        "CREATE SCHEDULE land EVERY 1 SECONDS DO INSERT INTO landed (id) VALUES (1)",
    )
    .await;
    run(
        &mut client,
        "CREATE SCHEDULE broken EVERY 1 SECONDS DO INSERT INTO nowhere (id) VALUES (1)",
    )
    .await;
    group.settle(leader, Duration::from_secs(10)).await;
    let created = schedule_on_every_member(&group, "land");
    assert_same_everywhere(&group, &created, "the schedule's instants");
    assert_eq!(created[0].0, None, "nothing has run yet");

    // The leader's sweep runs the body once and records the run beside the
    // rows, so every member holds one row and the same instants. The body
    // that fails still moves every member past this period
    let report = run_due_schedules(&node._server, FAR_FUTURE).await;
    assert_eq!(report.executed, 1);
    assert_eq!(report.failed, 1);
    assert_eq!(report.held, 0);
    group.settle(leader, Duration::from_secs(10)).await;

    assert_eq!(count_on_every_member(&group, "landed").await, vec![1, 1, 1]);
    let ran = schedule_on_every_member(&group, "land");
    assert_same_everywhere(&group, &ran, "the schedule's instants");
    assert_eq!(ran[0].0, Some(FAR_FUTURE), "the run is recorded");
    let next = ran[0].1.expect("the next run is set");
    assert!(
        next > FAR_FUTURE && next <= FAR_FUTURE + 1_000_000,
        "the next run lies within one period of the sweep: {next}"
    );
    let failed = schedule_on_every_member(&group, "broken");
    assert_same_everywhere(&group, &failed, "the failing schedule's instants");
    assert_eq!(failed[0].0, None, "a failed body records no last run");
    assert!(
        failed[0].1.is_some_and(|at| at > FAR_FUTURE),
        "a failed body still moves the schedule to its next period"
    );

    // A sweep at the same instant finds nothing due, on the leader and on a
    // member that took over, which is what stops a new leader running the
    // body again
    let again = run_due_schedules(&node._server, FAR_FUTURE).await;
    assert_eq!((again.executed, again.failed), (0, 0));
    group.settle(leader, Duration::from_secs(10)).await;
    assert_eq!(count_on_every_member(&group, "landed").await, vec![1, 1, 1]);

    client.terminate().await;
    group.shutdown().await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_retention_cycle_expires_rows_on_every_member() {
    let group = Group::start(3).await;
    let leader = group.leader(Duration::from_secs(10)).await;
    let node: &Arc<Node> = &group.nodes[leader];
    let addr = node.serve_wire().await;
    let mut client = WireClient::connect(addr).await;
    run(&mut client, "SET search_path = zyron_test").await;

    run(
        &mut client,
        "CREATE TABLE events (id BIGINT PRIMARY KEY, created_at TIMESTAMP)",
    )
    .await;
    run(
        &mut client,
        "INSERT INTO events (id, created_at) VALUES (1, '2000-01-01 00:00:00'), \
         (2, '2000-01-02 00:00:00'), (3, '2999-01-01 00:00:00')",
    )
    .await;
    run(
        &mut client,
        "ALTER TABLE events SET TTL 30 DAYS ON created_at",
    )
    .await;
    group.settle(leader, Duration::from_secs(10)).await;
    assert_eq!(count_on_every_member(&group, "events").await, vec![3, 3, 3]);

    // A follower's cycle expires nothing, whatever it holds
    let holds = Arc::new(zyron_lifecycle::legal_hold::LegalHoldRegistry::new());
    let stats = RetentionStats::new();
    let follower = (leader + 1) % group.nodes.len();
    run_retention_cycle(&group.nodes[follower]._server, &holds, &stats, false).await;
    assert_eq!(count_on_every_member(&group, "events").await, vec![3, 3, 3]);

    // The leader's cycle expires the two rows past the window through the
    // group, so every member keeps the same one
    run_retention_cycle(&node._server, &holds, &stats, true).await;
    assert_eq!(
        stats
            .rows_deleted
            .load(std::sync::atomic::Ordering::Relaxed),
        2
    );
    group.settle(leader, Duration::from_secs(10)).await;
    assert_eq!(count_on_every_member(&group, "events").await, vec![1, 1, 1]);
    for member in &group.nodes {
        assert_eq!(
            member.ints("SELECT id FROM events").await,
            vec![3],
            "{} keeps the row inside the window",
            member.name
        );
    }

    client.terminate().await;
    group.shutdown().await;
}
