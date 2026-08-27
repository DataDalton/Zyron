//! A hash join too large for memory produces the right answer instead of an
//! error.
//!
//! Every test here compares the spilling join against the same join run in
//! memory with no spill directory at all. That is the only comparison worth
//! making: a partitioned join that is nearly right is worse than one that
//! fails, and a join checked only against itself would agree with its own
//! mistakes. The in-memory run is given room to succeed, so the two differ in
//! exactly one thing.
//!
//! Output order is not part of the contract, because partitions are emitted
//! in partition order rather than input order. Every comparison is therefore
//! over sorted rows, which still catches a dropped row, a duplicated row, a
//! wrong pairing, and a null-padded row that should have matched.
//!
//! Run: cargo test -p zyron-executor --test join_spill_test

use std::collections::VecDeque;
use std::sync::Arc;
use std::sync::atomic::Ordering;

use zyron_catalog::ColumnId;
use zyron_common::TypeId;
use zyron_executor::batch::DataBatch;
use zyron_executor::column::{Column, ColumnData, NullBitmap};
use zyron_executor::operator::join::HashJoinOperator;
use zyron_executor::operator::{ExecutionBatch, Operator, OperatorResult};
use zyron_executor::spill::{SpillDirectory, SpillStats};
use zyron_parser::ast::JoinType;
use zyron_planner::binder::{BoundExpr, ColumnRef};
use zyron_planner::logical::LogicalColumn;

/// Bytes a join may hold at once. Small enough that every input below is
/// partitioned, large enough to hold a partition of one.
const TINY_BUDGET: u64 = 24 * 1024;

/// Rows per input batch, so the inputs arrive the way an operator feeds them.
const FEED_BATCH: usize = 256;

struct Feed {
    batches: VecDeque<DataBatch>,
}

impl Operator for Feed {
    fn next(&mut self) -> OperatorResult<'_> {
        Box::pin(async move { Ok(self.batches.pop_front().map(ExecutionBatch::new)) })
    }
}

fn directory(name: &str) -> Arc<SpillDirectory> {
    let root = std::env::temp_dir().join(format!("zyron_join_spill_{}_{name}", std::process::id()));
    let _ = std::fs::remove_dir_all(&root);
    std::fs::create_dir_all(&root).expect("scratch");
    Arc::new(SpillDirectory::open(&root, 512 * 1024 * 1024).expect("open"))
}

fn column(table_idx: usize, id: u16, name: &str, nullable: bool) -> LogicalColumn {
    LogicalColumn {
        name: name.into(),
        type_id: TypeId::Int64,
        nullable,
        fractional_digits: None,
        table_idx: Some(table_idx),
        column_id: ColumnId(id),
    }
}

fn key(table_idx: usize, id: u16) -> BoundExpr {
    BoundExpr::ColumnRef(ColumnRef {
        table_idx,
        column_id: ColumnId(id),
        type_id: TypeId::Int64,
        nullable: true,
        fractional_digits: None,
    })
}

/// Two Int64 columns: the join key and a payload that proves which row came
/// back rather than only how many did.
fn side(table_idx: usize, prefix: &str) -> Vec<LogicalColumn> {
    vec![
        column(table_idx, 0, &format!("{prefix}_key"), true),
        column(table_idx, 1, &format!("{prefix}_payload"), true),
    ]
}

/// Builds an input from key and payload pairs, chunked into batches.
fn feed(rows: &[(Option<i64>, i64)]) -> Box<dyn Operator> {
    let mut batches = VecDeque::new();
    for chunk in rows.chunks(FEED_BATCH) {
        let mut keys = Vec::with_capacity(chunk.len());
        let mut nulls = NullBitmap::none(chunk.len());
        let mut payloads = Vec::with_capacity(chunk.len());
        for (row, (k, payload)) in chunk.iter().enumerate() {
            match k {
                Some(k) => keys.push(*k),
                None => {
                    keys.push(0);
                    nulls.set_null(row);
                }
            }
            payloads.push(*payload);
        }
        batches.push_back(DataBatch::new(vec![
            Column::with_nulls_ts(ColumnData::Int64(keys), nulls, TypeId::Int64, None),
            Column::new(ColumnData::Int64(payloads), TypeId::Int64),
        ]));
    }
    Box::new(Feed { batches })
}

/// Deterministic keys with a controlled number of distinct values, so a test
/// decides for itself how much a partition holds.
fn keys(count: usize, distinct: i64) -> Vec<(Option<i64>, i64)> {
    let mut out = Vec::with_capacity(count);
    let mut state = 0x9E37_79B9_7F4A_7C15u64;
    for payload in 0..count {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        out.push((Some((state % distinct as u64) as i64), payload as i64));
    }
    out
}

/// Runs the join and returns every output row as a flat tuple.
async fn run(mut op: Box<dyn Operator>) -> Vec<Vec<Option<i64>>> {
    let mut rows = Vec::new();
    while let Some(eb) = op.next().await.expect("the join answers") {
        for row in 0..eb.batch.num_rows {
            rows.push(
                eb.batch
                    .columns
                    .iter()
                    .map(|c| {
                        if c.is_null(row) {
                            None
                        } else {
                            match &c.data {
                                ColumnData::Int64(v) => Some(v[row]),
                                other => panic!("unexpected output column {other:?}"),
                            }
                        }
                    })
                    .collect(),
            );
        }
    }
    rows.sort();
    rows
}

fn join(
    left: Box<dyn Operator>,
    right: Box<dyn Operator>,
    join_type: JoinType,
    condition: Option<BoundExpr>,
) -> HashJoinOperator {
    HashJoinOperator::new(
        left,
        right,
        join_type,
        vec![key(0, 0)],
        vec![key(1, 0)],
        condition,
        side(0, "l"),
        side(1, "r"),
    )
}

/// What one comparison produced: the same join run with room and run with a
/// budget it cannot fit in, and how many spill files the second one wrote.
struct BothWays {
    memory: Vec<Vec<Option<i64>>>,
    spilled: Vec<Vec<Option<i64>>>,
    files: u64,
}

/// Runs the join twice, once each way.
///
/// The file count comes off the directory rather than the process counters,
/// because the tests in this binary run at the same time and a process
/// counter would report what all of them did.
async fn both_ways(
    name: &str,
    left: &[(Option<i64>, i64)],
    right: &[(Option<i64>, i64)],
    join_type: JoinType,
    condition: Option<BoundExpr>,
) -> BothWays {
    both_ways_within(name, left, right, join_type, condition, TINY_BUDGET).await
}

/// The same comparison against a chosen budget, for the tests that need one
/// small enough that neither side of a partition can build.
async fn both_ways_within(
    name: &str,
    left: &[(Option<i64>, i64)],
    right: &[(Option<i64>, i64)],
    join_type: JoinType,
    condition: Option<BoundExpr>,
    budget: u64,
) -> BothWays {
    let memory = run(Box::new(join(
        feed(left),
        feed(right),
        join_type,
        condition.clone(),
    )))
    .await;

    let dir = directory(name);
    let mut spilling = join(feed(left), feed(right), join_type, condition);
    spilling.set_spill(Some(Arc::clone(&dir)), budget);
    let spilled = run(Box::new(spilling)).await;

    assert_eq!(
        dir.reserved_bytes(),
        0,
        "{name} left spill files behind when it finished"
    );
    BothWays {
        memory,
        spilled,
        files: dir.files_created(),
    }
}

/// Asserts the two answers agree, and that the second one came out of a join
/// that really did spill.
fn agree(name: &str, run: &BothWays) {
    assert!(!run.memory.is_empty(), "{name} produced no rows to compare");
    assert_eq!(
        run.spilled.len(),
        run.memory.len(),
        "{name}: spilling returned {} rows, memory returned {}",
        run.spilled.len(),
        run.memory.len()
    );
    assert_eq!(
        run.spilled, run.memory,
        "{name}: spilling returned different rows"
    );
    assert!(
        run.files > 0,
        "{name} fit after all, so it proves nothing about spilling"
    );
}

/// Both sides too large, so both are partitioned and the join runs partition
/// by partition.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn an_inner_join_past_its_budget_partitions_and_agrees() {
    let before = SpillStats::global().joins_spilled.load(Ordering::Relaxed);
    let left = keys(4_000, 700);
    let right = keys(4_000, 700);
    let outcome = both_ways("inner", &left, &right, JoinType::Inner, None).await;
    agree("inner", &outcome);
    assert!(
        outcome.files >= 2,
        "a partitioned join wrote {} files, which is not a partitioning",
        outcome.files
    );
    assert!(
        SpillStats::global().joins_spilled.load(Ordering::Relaxed) > before,
        "no join anywhere reported partitioning"
    );
}

/// Left outer, so unmatched build rows have to survive the partitioning.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_left_outer_join_keeps_its_unmatched_rows() {
    // Disjoint key ranges for part of each side, so there are genuinely
    // unmatched rows on both sides rather than a lucky full match
    let mut left = keys(3_000, 400);
    left.extend((0..600).map(|i| (Some(10_000 + i), 90_000 + i)));
    let right = keys(3_000, 400);
    let outcome = both_ways("left", &left, &right, JoinType::Left, None).await;
    agree("left", &outcome);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_right_outer_join_keeps_its_unmatched_rows() {
    let left = keys(3_000, 400);
    let mut right = keys(3_000, 400);
    right.extend((0..600).map(|i| (Some(10_000 + i), 90_000 + i)));
    let outcome = both_ways("right", &left, &right, JoinType::Right, None).await;
    agree("right", &outcome);
}

/// Full outer, which owes unmatched rows on both sides at once.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_full_outer_join_keeps_both_sides() {
    let mut left = keys(2_500, 300);
    left.extend((0..500).map(|i| (Some(20_000 + i), 80_000 + i)));
    let mut right = keys(2_500, 300);
    right.extend((0..500).map(|i| (Some(30_000 + i), 70_000 + i)));
    let outcome = both_ways("full", &left, &right, JoinType::Full, None).await;
    agree("full", &outcome);
}

/// NULL keys never match, on either side, whichever side they are on.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn null_keys_still_match_nothing_across_partitions() {
    let mut left = keys(2_000, 300);
    let mut right = keys(2_000, 300);
    for i in 0..400 {
        left.push((None, 50_000 + i));
        right.push((None, 60_000 + i));
    }
    let outcome = both_ways("nulls", &left, &right, JoinType::Full, None).await;
    agree("nulls", &outcome);
    // A matched row carries a payload from both sides. One of those with a
    // NULL on either key is a NULL that joined, which nothing may do
    let null_pairs = outcome
        .spilled
        .iter()
        .filter(|row| row[1].is_some() && row[3].is_some())
        .filter(|row| row[0].is_none() || row[2].is_none())
        .count();
    assert_eq!(null_pairs, 0, "a NULL key matched something");
}

/// A residual condition has to be evaluated inside each partition, against
/// rows read back off disk.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_residual_condition_survives_partitioning() {
    let condition = BoundExpr::BinaryOp {
        left: Box::new(key(0, 1)),
        op: zyron_parser::ast::BinaryOperator::Lt,
        right: Box::new(key(1, 1)),
        type_id: TypeId::Boolean,
    };
    let left = keys(2_500, 300);
    let right = keys(2_500, 300);
    let outcome = both_ways("residual", &left, &right, JoinType::Left, Some(condition)).await;
    agree("residual", &outcome);
}

/// One key value for every row, so no hash separates the partition and the
/// join has to pass the build side over the probe side in blocks.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn one_key_for_every_row_falls_back_to_blocks() {
    // Small counts: this is a cross product, so the output is the product of
    // the two sides and grows fast
    let left: Vec<(Option<i64>, i64)> = (0..2_000).map(|i| (Some(7), i)).collect();
    let right: Vec<(Option<i64>, i64)> = (0..40).map(|i| (Some(7), 1_000 + i)).collect();
    let outcome = both_ways("degenerate", &left, &right, JoinType::Inner, None).await;
    assert_eq!(
        outcome.memory.len(),
        2_000 * 40,
        "the control run did not produce the cross product"
    );
    agree("degenerate", &outcome);
}

/// The blocked pass again, with outer rows on both sides. A probe row that
/// matched block three must not be emitted as unmatched by block one.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_blocked_pass_emits_each_unmatched_row_once() {
    let mut left: Vec<(Option<i64>, i64)> = (0..2_000).map(|i| (Some(7), i)).collect();
    left.extend((0..50).map(|i| (Some(8), 500_000 + i)));
    let mut right: Vec<(Option<i64>, i64)> = (0..30).map(|i| (Some(7), 1_000 + i)).collect();
    right.extend((0..50).map(|i| (Some(9), 600_000 + i)));
    let outcome = both_ways("blocked_full", &left, &right, JoinType::Full, None).await;
    agree("blocked_full", &outcome);
}

/// Neither side of the one partition fits, so neither can build and the
/// build side is passed over the probe side in blocks.
///
/// The other two degenerate tests never reach this: their second side is
/// small enough to build, and the join reverses roles instead. Here both
/// sides are past the budget, which is the only way into the blocked pass.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn neither_side_fitting_one_key_passes_the_build_side_in_blocks() {
    // A budget a few hundred rows cannot fit in, so both sides spill and the
    // cross product stays a size a test can check exactly
    const CRAMPED: u64 = 2 * 1024;
    let before = SpillStats::global().blocked_passes.load(Ordering::Relaxed);

    let left: Vec<(Option<i64>, i64)> = (0..300).map(|i| (Some(7), i)).collect();
    let right: Vec<(Option<i64>, i64)> = (0..300).map(|i| (Some(7), 1_000 + i)).collect();
    let outcome = both_ways_within("blocked", &left, &right, JoinType::Inner, None, CRAMPED).await;
    assert_eq!(
        outcome.memory.len(),
        300 * 300,
        "the control run did not produce the cross product"
    );
    agree("blocked", &outcome);
    assert!(
        SpillStats::global().blocked_passes.load(Ordering::Relaxed) > before,
        "the partition split after all, so the blocked pass never ran"
    );
}

/// The blocked pass owing outer rows on both sides at once, which is where a
/// probe row that matched a later block must not be emitted by an earlier
/// one.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_blocked_full_outer_join_emits_each_row_once() {
    const CRAMPED: u64 = 2 * 1024;
    let before = SpillStats::global().blocked_passes.load(Ordering::Relaxed);

    // One key on both sides in bulk, plus keys only one side has, so there
    // are unmatched rows to place on both
    let mut left: Vec<(Option<i64>, i64)> = (0..250).map(|i| (Some(7), i)).collect();
    left.extend((0..60).map(|i| (Some(100 + i), 500_000 + i)));
    let mut right: Vec<(Option<i64>, i64)> = (0..250).map(|i| (Some(7), 1_000 + i)).collect();
    right.extend((0..60).map(|i| (Some(200 + i), 600_000 + i)));

    let outcome = both_ways_within(
        "blocked_outer",
        &left,
        &right,
        JoinType::Full,
        None,
        CRAMPED,
    )
    .await;
    assert_eq!(
        outcome.memory.len(),
        250 * 250 + 60 + 60,
        "the control run is not the shape this test is about"
    );
    agree("blocked_outer", &outcome);
    assert!(
        SpillStats::global().blocked_passes.load(Ordering::Relaxed) > before,
        "the partition split after all, so the blocked pass never ran"
    );
}

/// A build side that fits against a probe side that does not. Nothing is
/// written: the small side builds and the large one streams past it.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_small_build_side_streams_the_probe_side_without_writing() {
    let left = keys(60, 60);
    let right = keys(6_000, 500);
    let outcome = both_ways("stream_probe", &left, &right, JoinType::Left, None).await;
    assert_eq!(
        outcome.spilled, outcome.memory,
        "streaming the probe side returned different rows"
    );
    assert_eq!(
        outcome.files, 0,
        "a join whose build side fit still wrote {} spill files",
        outcome.files
    );
}

/// The mirror image: the first side read does not fit and is partitioned,
/// then the second one turns out to fit and builds instead.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_large_left_side_gives_way_to_a_small_right_one() {
    let left = keys(6_000, 500);
    let right = keys(60, 60);
    let outcome = both_ways("reversed", &left, &right, JoinType::Right, None).await;
    agree("reversed", &outcome);
}

/// A join with room to spill but no need to spill takes the path it always
/// took.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_join_that_fits_writes_nothing() {
    let dir = directory("fits");
    let left = keys(200, 50);
    let right = keys(200, 50);
    let mut op = join(feed(&left), feed(&right), JoinType::Inner, None);
    op.set_spill(Some(Arc::clone(&dir)), 64 * 1024 * 1024);
    let spilled = run(Box::new(op)).await;
    let memory = run(Box::new(join(
        feed(&left),
        feed(&right),
        JoinType::Inner,
        None,
    )))
    .await;
    assert_eq!(spilled, memory);
    assert_eq!(
        dir.files_created(),
        0,
        "a join that fit inside its budget still wrote a spill file"
    );
}

/// Nowhere to spill is still an honest failure rather than an unbounded one.
/// This is the control the tests above are measured against: without a spill
/// directory the same shape fails on the budget.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn without_a_spill_directory_the_budget_still_fails_the_query() {
    let left = keys(4_000, 700);
    let right = keys(4_000, 700);
    let mut op = join(feed(&left), feed(&right), JoinType::Inner, None);
    op.set_memory_budget(Some(zyron_executor::context::QueryMemoryBudget::new(
        TINY_BUDGET,
    )));
    let mut op: Box<dyn Operator> = Box::new(op);
    let mut error = None;
    loop {
        match op.next().await {
            Ok(Some(_)) => continue,
            Ok(None) => break,
            Err(e) => {
                error = Some(e.to_string());
                break;
            }
        }
    }
    let error = error.expect("a join with no room and nowhere to go should fail");
    assert!(
        error.contains("memory budget"),
        "expected the budget failure this is the control for, got: {error}"
    );
}
