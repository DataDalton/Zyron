//! A GROUP BY with more groups than fit produces the right answer instead of
//! an error.
//!
//! The design being checked is what makes this correct for every aggregate
//! rather than only the decomposable ones. Accumulator state never goes to
//! disk: once the group table reaches its budget it stops taking new groups,
//! rows for groups already in it keep folding into them, and rows for new
//! groups are written to a partition chosen by the hash of the grouping key.
//! So a group lives in exactly one place and nothing is ever merged. Both
//! halves of that need proving: that the resident groups are complete, and
//! that the routed ones are.
//!
//! Every test compares against the same aggregate with room, so the two runs
//! differ in one thing. Output order is not part of the contract, since
//! partitions are emitted after the resident groups, so rows are sorted before
//! they are compared.
//!
//! Run: cargo test -p zyron-executor --test aggregate_spill_test

use std::collections::VecDeque;
use std::sync::Arc;
use std::sync::atomic::Ordering;

use zyron_catalog::ColumnId;
use zyron_common::TypeId;
use zyron_executor::batch::DataBatch;
use zyron_executor::column::{Column, ColumnData, NullBitmap, ScalarValue};
use zyron_executor::operator::aggregate::HashAggregateOperator;
use zyron_executor::operator::{ExecutionBatch, Operator, OperatorResult};
use zyron_executor::spill::{SpillDirectory, SpillStats};
use zyron_planner::binder::{BoundExpr, ColumnRef};
use zyron_planner::logical::{AggregateExpr, LogicalColumn};

/// Bytes of group state the aggregate may hold. Small enough that the inputs
/// below overflow it many times over.
const TINY_BUDGET: u64 = 8 * 1024;

/// Rows per input batch.
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
    let root = std::env::temp_dir().join(format!("zyron_agg_spill_{}_{name}", std::process::id()));
    let _ = std::fs::remove_dir_all(&root);
    std::fs::create_dir_all(&root).expect("scratch");
    Arc::new(SpillDirectory::open(&root, 512 * 1024 * 1024).expect("open"))
}

fn column(id: u16, name: &str, type_id: TypeId) -> LogicalColumn {
    LogicalColumn {
        name: name.into(),
        type_id,
        nullable: true,
        fractional_digits: None,
        table_idx: Some(0),
        column_id: ColumnId(id),
    }
}

fn column_ref(id: u16, type_id: TypeId) -> BoundExpr {
    BoundExpr::ColumnRef(ColumnRef {
        table_idx: 0,
        column_id: ColumnId(id),
        type_id,
        nullable: true,
        fractional_digits: None,
    })
}

/// Input: a grouping key and a value, both Int64, the key nullable.
fn input_schema() -> Vec<LogicalColumn> {
    vec![column(0, "k", TypeId::Int64), column(1, "v", TypeId::Int64)]
}

/// Builds an input from key and value pairs.
fn feed(rows: &[(Option<i64>, i64)]) -> Box<dyn Operator> {
    let mut batches = VecDeque::new();
    for chunk in rows.chunks(FEED_BATCH) {
        let mut keys = Vec::with_capacity(chunk.len());
        let mut nulls = NullBitmap::none(chunk.len());
        let mut values = Vec::with_capacity(chunk.len());
        for (row, (k, v)) in chunk.iter().enumerate() {
            match k {
                Some(k) => keys.push(*k),
                None => {
                    keys.push(0);
                    nulls.set_null(row);
                }
            }
            values.push(*v);
        }
        batches.push_back(DataBatch::new(vec![
            Column::with_nulls_ts(ColumnData::Int64(keys), nulls, TypeId::Int64, None),
            Column::new(ColumnData::Int64(values), TypeId::Int64),
        ]));
    }
    Box::new(Feed { batches })
}

/// Deterministic keys spread over a chosen number of distinct values, with
/// each key appearing many times so the resident groups genuinely keep taking
/// rows after the table freezes.
fn rows_over(count: usize, distinct: i64) -> Vec<(Option<i64>, i64)> {
    let mut out = Vec::with_capacity(count);
    let mut state = 0x2545_F491_4F6C_DD1Du64;
    for i in 0..count {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        out.push((Some((state % distinct as u64) as i64), i as i64));
    }
    out
}

fn aggregate(function: &str, arg: Option<BoundExpr>, out_type: TypeId) -> AggregateExpr {
    AggregateExpr {
        function_name: function.to_string(),
        args: arg.into_iter().collect(),
        distinct: false,
        return_type: out_type,
        uda: None,
    }
}

/// One grouped aggregate, built either with room or with a budget it cannot
/// fit in.
fn build(
    rows: &[(Option<i64>, i64)],
    aggregates: Vec<AggregateExpr>,
    spill: Option<(Arc<SpillDirectory>, u64)>,
) -> Box<dyn Operator> {
    let mut output_schema = vec![column(0, "k", TypeId::Int64)];
    for (i, agg) in aggregates.iter().enumerate() {
        output_schema.push(column(10 + i as u16, "agg", agg.return_type));
    }
    let mut op = HashAggregateOperator::new(
        feed(rows),
        vec![column_ref(0, TypeId::Int64)],
        aggregates,
        input_schema(),
        output_schema,
    );
    if let Some((dir, threshold)) = spill {
        op.set_spill(Some(dir), threshold);
    }
    Box::new(op)
}

/// Drains an aggregate and returns its rows, sorted.
async fn run(mut op: Box<dyn Operator>) -> Vec<Vec<Option<i128>>> {
    let mut rows = Vec::new();
    while let Some(eb) = op.next().await.expect("the aggregate answers") {
        for row in 0..eb.batch.num_rows {
            rows.push(
                eb.batch
                    .columns
                    .iter()
                    .map(|c| {
                        if c.is_null(row) {
                            None
                        } else {
                            match c.get_scalar(row) {
                                ScalarValue::Int64(v) => Some(v as i128),
                                ScalarValue::Int128(v) => Some(v),
                                ScalarValue::Int32(v) => Some(v as i128),
                                ScalarValue::Float64(v) => Some(v as i128),
                                other => panic!("unexpected output value {other:?}"),
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

/// What one comparison produced.
struct BothWays {
    memory: Vec<Vec<Option<i128>>>,
    spilled: Vec<Vec<Option<i128>>>,
    files: u64,
}

async fn both_ways(
    name: &str,
    rows: &[(Option<i64>, i64)],
    make: impl Fn() -> Vec<AggregateExpr>,
) -> BothWays {
    both_ways_within(name, rows, make, TINY_BUDGET).await
}

async fn both_ways_within(
    name: &str,
    rows: &[(Option<i64>, i64)],
    make: impl Fn() -> Vec<AggregateExpr>,
    budget: u64,
) -> BothWays {
    let memory = run(build(rows, make(), None)).await;
    let dir = directory(name);
    let spilled = run(build(rows, make(), Some((Arc::clone(&dir), budget)))).await;
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

/// Distinct keys the input actually holds.
///
/// Counted rather than assumed: a scrambled key over a range does not touch
/// every value in it, so an expectation taken from the range would be
/// asserting about the generator instead of the aggregate.
fn distinct_keys(rows: &[(Option<i64>, i64)]) -> usize {
    let mut keys: Vec<Option<i64>> = rows.iter().map(|(k, _)| *k).collect();
    keys.sort();
    keys.dedup();
    keys.len()
}

fn agree(name: &str, run: &BothWays) {
    assert!(!run.memory.is_empty(), "{name} produced no rows to compare");
    assert_eq!(
        run.spilled.len(),
        run.memory.len(),
        "{name}: spilling returned {} groups, memory returned {}",
        run.spilled.len(),
        run.memory.len()
    );
    assert_eq!(
        run.spilled, run.memory,
        "{name}: spilling returned different groups"
    );
    assert!(
        run.files > 0,
        "{name} fit after all, so it proves nothing about spilling"
    );
}

/// More groups than the table holds, with SUM and COUNT over each.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn more_groups_than_fit_still_aggregate_correctly() {
    let before = SpillStats::global()
        .aggregates_spilled
        .load(Ordering::Relaxed);
    let rows = rows_over(20_000, 4_000);
    let outcome = both_ways("sum_count", &rows, || {
        vec![
            aggregate("sum", Some(column_ref(1, TypeId::Int64)), TypeId::Int64),
            aggregate("count", Some(column_ref(1, TypeId::Int64)), TypeId::Int64),
        ]
    })
    .await;
    agree("sum_count", &outcome);
    assert_eq!(
        outcome.spilled.len(),
        distinct_keys(&rows),
        "not every group came back once"
    );
    assert!(
        SpillStats::global()
            .aggregates_spilled
            .load(Ordering::Relaxed)
            > before,
        "no aggregate anywhere reported spilling"
    );
}

/// The property the whole design rests on: a group that was resident when the
/// table froze must still have collected every one of its later rows, and a
/// group that was routed must have collected all of its rows in its partition.
///
/// Checked by counting. Every input row belongs to exactly one group, so the
/// counts summed over the groups must equal the input row count, and each
/// group's own count must match what the memory run produced. A group split
/// between the resident table and a partition would show up as two rows for
/// one key, and a group whose rows were lost would show up as a low count.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn no_group_is_split_and_no_row_is_lost() {
    const ROWS: usize = 30_000;
    let rows = rows_over(ROWS, 6_000);
    let outcome = both_ways("counts", &rows, || {
        vec![aggregate(
            "count",
            Some(column_ref(1, TypeId::Int64)),
            TypeId::Int64,
        )]
    })
    .await;
    agree("counts", &outcome);

    let mut keys: Vec<Option<i128>> = outcome.spilled.iter().map(|row| row[0]).collect();
    let distinct = keys.len();
    keys.sort();
    keys.dedup();
    assert_eq!(distinct, keys.len(), "a key came back on more than one row");

    let total: i128 = outcome
        .spilled
        .iter()
        .map(|row| row[1].expect("a count is never null"))
        .sum();
    assert_eq!(total as usize, ROWS, "rows went missing across partitions");
}

/// A key that is NULL is one group, and it must survive routing like any
/// other. NULL keys hash together, so they land in one partition.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_null_key_is_one_group_wherever_it_lands() {
    let mut rows = rows_over(10_000, 3_000);
    for i in 0..500 {
        rows.push((None, 900_000 + i));
    }
    let outcome = both_ways("nulls", &rows, || {
        vec![aggregate(
            "count",
            Some(column_ref(1, TypeId::Int64)),
            TypeId::Int64,
        )]
    })
    .await;
    agree("nulls", &outcome);
    let null_groups: Vec<&Vec<Option<i128>>> = outcome
        .spilled
        .iter()
        .filter(|row| row[0].is_none())
        .collect();
    assert_eq!(
        null_groups.len(),
        1,
        "the NULL key came back as {} groups",
        null_groups.len()
    );
    assert_eq!(
        null_groups[0][1],
        Some(500),
        "the NULL group lost rows on its way to a partition"
    );
}

/// MIN and MAX, which keep a value rather than a running total, so a group
/// whose rows were split across two places would report the extreme of one
/// half.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn min_and_max_see_every_row_of_their_group() {
    let rows = rows_over(20_000, 5_000);
    let outcome = both_ways("min_max", &rows, || {
        vec![
            aggregate("min", Some(column_ref(1, TypeId::Int64)), TypeId::Int64),
            aggregate("max", Some(column_ref(1, TypeId::Int64)), TypeId::Int64),
        ]
    })
    .await;
    agree("min_max", &outcome);
}

/// An aggregate whose state grows per distinct value and whose merge is a no
/// op. Nothing may merge it, and nothing does: its group is either resident
/// or in one partition, and either way one accumulator sees every row.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_distinct_aggregate_is_correct_because_nothing_merges_it() {
    // Values repeat within a group, so COUNT(DISTINCT v) differs from COUNT(v)
    // and a lost or double-counted row shows up
    let mut rows = Vec::new();
    for i in 0..20_000usize {
        rows.push((Some((i % 2_000) as i64), (i % 37) as i64));
    }
    let outcome = both_ways("distinct", &rows, || {
        vec![AggregateExpr {
            function_name: "count".to_string(),
            args: vec![column_ref(1, TypeId::Int64)],
            distinct: true,
            return_type: TypeId::Int64,
            uda: None,
        }]
    })
    .await;
    agree("distinct", &outcome);
}

/// AVG, whose finalize divides a sum by a count, so a group that saw half its
/// rows returns a plausible wrong number rather than an obvious one.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn an_average_over_a_routed_group_is_the_average_of_all_of_it() {
    let rows = rows_over(20_000, 4_000);
    let outcome = both_ways("avg", &rows, || {
        vec![aggregate(
            "avg",
            Some(column_ref(1, TypeId::Int64)),
            TypeId::Float64,
        )]
    })
    .await;
    agree("avg", &outcome);
}

/// A budget so small that one round of partitioning is not enough, so a
/// partition is read back and split again.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_partition_that_still_does_not_fit_splits_again() {
    const CRAMPED: u64 = 1024;
    let rows = rows_over(40_000, 20_000);
    let outcome = both_ways_within(
        "reflow",
        &rows,
        || {
            vec![aggregate(
                "count",
                Some(column_ref(1, TypeId::Int64)),
                TypeId::Int64,
            )]
        },
        CRAMPED,
    )
    .await;
    agree("reflow", &outcome);
    assert_eq!(outcome.spilled.len(), distinct_keys(&rows));
    assert!(
        outcome.spilled.len() > 15_000,
        "the input did not hold enough groups to force a second split"
    );
    // One level of sixteen partitions could not have held this many groups at
    // this budget, so the partitions were split again
    assert!(
        outcome.files > 16,
        "only {} files, so nothing was re-partitioned",
        outcome.files
    );
}

/// Few groups over many rows: the table never freezes, nothing is written,
/// and the path taken is the one that was always taken.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_grouping_that_collapses_its_input_writes_nothing() {
    let rows = rows_over(50_000, 12);
    let dir = directory("collapsing");
    let spilled = run(build(
        &rows,
        vec![aggregate(
            "count",
            Some(column_ref(1, TypeId::Int64)),
            TypeId::Int64,
        )],
        Some((Arc::clone(&dir), TINY_BUDGET)),
    ))
    .await;
    let memory = run(build(
        &rows,
        vec![aggregate(
            "count",
            Some(column_ref(1, TypeId::Int64)),
            TypeId::Int64,
        )],
        None,
    ))
    .await;
    assert_eq!(spilled, memory);
    assert_eq!(spilled.len(), 12);
    assert_eq!(
        dir.files_created(),
        0,
        "twelve groups were spilled, which means the budget was not the reason"
    );
}

/// The control. Without a spill directory the same aggregate still fails on
/// its budget, which is what makes the passing tests above evidence about
/// spilling rather than about a generous budget.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn without_a_spill_directory_the_budget_still_fails_the_query() {
    let rows = rows_over(20_000, 4_000);
    let mut op = HashAggregateOperator::new(
        feed(&rows),
        vec![column_ref(0, TypeId::Int64)],
        vec![aggregate(
            "count",
            Some(column_ref(1, TypeId::Int64)),
            TypeId::Int64,
        )],
        input_schema(),
        vec![
            column(0, "k", TypeId::Int64),
            column(10, "agg", TypeId::Int64),
        ],
    );
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
    let error = error.expect("an aggregate with no room and nowhere to go should fail");
    assert!(
        error.contains("memory budget"),
        "expected the budget failure this is the control for, got: {error}"
    );
}
