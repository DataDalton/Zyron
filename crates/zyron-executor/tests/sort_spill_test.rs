//! A sort too large for memory produces the right answer instead of an error.
//!
//! The failure this replaces was not subtle: a sort whose input outgrew the
//! query memory budget returned "query exceeds its memory budget" and the
//! query was over. Every other database solved that decades ago by putting
//! the excess on disk, and the property that matters is that the answer is
//! identical and only the speed changes.
//!
//! So every test here asserts the full ordering against a sort done in memory
//! by the standard library, not a spot check. An external sort that is nearly
//! right is worse than one that fails.
//!
//! Run: cargo test -p zyron-executor --test sort_spill_test

use std::collections::VecDeque;
use std::sync::Arc;
use std::sync::atomic::Ordering;

use zyron_common::page::PageId;
use zyron_common::{RowLocator, TypeId};
use zyron_executor::batch::DataBatch;
use zyron_executor::column::{Column, ColumnData, NullBitmap};
use zyron_executor::operator::sort::SortOperator;
use zyron_executor::operator::{ExecutionBatch, Operator, OperatorResult};
use zyron_executor::spill::{SpillDirectory, SpillStats};
use zyron_planner::binder::{BoundExpr, BoundOrderBy, ColumnRef};
use zyron_planner::logical::LogicalColumn;

/// Bytes the sort may hold before it writes a run. Far below every input
/// below, so the spill path is what runs.
const TINY_BUDGET: u64 = 16 * 1024;

struct Feed {
    batches: VecDeque<DataBatch>,
}

impl Operator for Feed {
    fn next(&mut self) -> OperatorResult<'_> {
        Box::pin(async move { Ok(self.batches.pop_front().map(ExecutionBatch::new)) })
    }
}

fn schema() -> Vec<LogicalColumn> {
    vec![LogicalColumn {
        name: "v".into(),
        type_id: TypeId::Int64,
        nullable: true,
        fractional_digits: None,
        table_idx: Some(0),
        column_id: zyron_catalog::ColumnId(0),
    }]
}

fn ascending() -> Vec<BoundOrderBy> {
    vec![BoundOrderBy {
        expr: BoundExpr::ColumnRef(ColumnRef {
            table_idx: 0,
            column_id: zyron_catalog::ColumnId(0),
            type_id: TypeId::Int64,
            nullable: true,
            fractional_digits: None,
        }),
        asc: true,
        nulls_first: false,
    }]
}

fn directory(name: &str) -> Arc<SpillDirectory> {
    let root = std::env::temp_dir().join(format!("zyron_sort_spill_{}_{name}", std::process::id()));
    let _ = std::fs::remove_dir_all(&root);
    std::fs::create_dir_all(&root).expect("scratch");
    Arc::new(SpillDirectory::open(&root, 512 * 1024 * 1024).expect("open"))
}

/// Deterministic and genuinely unsorted, so the same run happens every time
/// and the sort has real work to do.
fn scrambled(count: i64) -> Vec<i64> {
    let mut out = Vec::with_capacity(count as usize);
    let mut state = 0x2545_F491_4F6C_DD1Du64;
    for _ in 0..count {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        out.push((state % (count as u64 * 4)) as i64);
    }
    out
}

fn feed_of(values: &[i64], rows_per_batch: usize) -> Feed {
    Feed {
        batches: values
            .chunks(rows_per_batch)
            .map(|chunk| {
                DataBatch::new(vec![Column::new(
                    ColumnData::Int64(chunk.to_vec()),
                    TypeId::Int64,
                )])
            })
            .collect(),
    }
}

async fn drain(op: &mut SortOperator) -> Vec<i64> {
    let mut out = Vec::new();
    while let Some(eb) = op.next().await.expect("the sort returned an error") {
        match &eb.batch.columns[0].data {
            ColumnData::Int64(v) => out.extend_from_slice(v),
            other => panic!("wrong output type: {other:?}"),
        }
    }
    out
}

/// The thing this exists for.
#[tokio::test]
async fn a_sort_too_large_for_memory_spills_and_still_sorts() {
    let values = scrambled(40_000);
    let mut expected = values.clone();
    expected.sort_unstable();

    let dir = directory("large");
    let before = SpillStats::global().sorts_spilled.load(Ordering::Relaxed);

    let mut op = SortOperator::new(
        Box::new(feed_of(&values, 1_000)),
        ascending(),
        schema(),
        None,
    );
    op.set_spill(Some(Arc::clone(&dir)), TINY_BUDGET);

    let out = drain(&mut op).await;
    assert_eq!(out.len(), expected.len(), "the sort lost or invented rows");
    assert_eq!(out, expected, "the merged output is not in order");
    assert!(
        SpillStats::global().sorts_spilled.load(Ordering::Relaxed) > before,
        "nothing spilled, so this measured the in-memory path"
    );
    assert_eq!(
        dir.reserved_bytes(),
        0,
        "spill files outlived the sort that wrote them"
    );
}

/// The same input under a budget it fits in takes the untouched in-memory
/// path and gives the same answer.
#[tokio::test]
async fn the_in_memory_path_gives_the_same_answer() {
    let values = scrambled(8_000);
    let mut expected = values.clone();
    expected.sort_unstable();

    let dir = directory("fits");
    let mut op = SortOperator::new(
        Box::new(feed_of(&values, 1_000)),
        ascending(),
        schema(),
        None,
    );
    op.set_spill(Some(Arc::clone(&dir)), 512 * 1024 * 1024);

    assert_eq!(drain(&mut op).await, expected);
    assert_eq!(dir.reserved_bytes(), 0, "a sort that fit still spilled");
}

/// A node with no spill directory behaves exactly as it did before.
#[tokio::test]
async fn without_a_spill_directory_the_old_path_runs() {
    let values = scrambled(2_000);
    let mut expected = values.clone();
    expected.sort_unstable();

    let mut op = SortOperator::new(Box::new(feed_of(&values, 500)), ascending(), schema(), None);
    assert_eq!(drain(&mut op).await, expected);
}

/// Descending survives the merge, because the comparator the merge runs is
/// the one the runs were written with.
#[tokio::test]
async fn a_descending_sort_merges_in_the_right_direction() {
    let values = scrambled(20_000);
    let mut expected = values.clone();
    expected.sort_unstable_by(|a, b| b.cmp(a));

    let mut order = ascending();
    order[0].asc = false;
    let mut op = SortOperator::new(Box::new(feed_of(&values, 800)), order, schema(), None);
    op.set_spill(Some(directory("descending")), TINY_BUDGET);

    assert_eq!(drain(&mut op).await, expected);
}

/// A limit is honoured across the merge, and each run is truncated to it so
/// the spill stays proportional to the answer rather than the input.
#[tokio::test]
async fn a_limit_is_honoured_across_the_merge() {
    let values = scrambled(30_000);
    let mut expected = values.clone();
    expected.sort_unstable();
    expected.truncate(100);

    let mut op = SortOperator::new(
        Box::new(feed_of(&values, 700)),
        ascending(),
        schema(),
        Some(100),
    );
    op.set_spill(Some(directory("limit")), TINY_BUDGET);

    let out = drain(&mut op).await;
    assert_eq!(out.len(), 100, "the limit was not applied");
    assert_eq!(out, expected);
}

/// Nulls keep their place through a merge. A separate path from the values,
/// because the comparator tests nullness before it reads anything.
#[tokio::test]
async fn nulls_sort_last_through_a_merge() {
    let mut batches = VecDeque::new();
    let mut expected_values = Vec::new();
    let mut null_count = 0usize;
    for chunk in 0..40i64 {
        let mut data = Vec::with_capacity(500);
        let mut nulls = NullBitmap::none(500);
        for row in 0..500i64 {
            let v = (chunk * 7 + row * 13) % 4096;
            data.push(v);
            if row % 11 == 0 {
                nulls.set_null(row as usize);
                null_count += 1;
            } else {
                expected_values.push(v);
            }
        }
        batches.push_back(DataBatch::new(vec![Column {
            data: ColumnData::Int64(data),
            nulls,
            type_id: TypeId::Int64,
            fractional_digits: None,
        }]));
    }
    expected_values.sort_unstable();

    let mut op = SortOperator::new(Box::new(Feed { batches }), ascending(), schema(), None);
    op.set_spill(Some(directory("nulls")), TINY_BUDGET);

    let mut non_null = Vec::new();
    let mut trailing_nulls = 0usize;
    let mut seen_null = false;
    while let Some(eb) = op.next().await.expect("sort") {
        let column = &eb.batch.columns[0];
        let ColumnData::Int64(v) = &column.data else {
            panic!("wrong type back");
        };
        for row in 0..eb.batch.num_rows {
            if column.is_null(row) {
                seen_null = true;
                trailing_nulls += 1;
            } else {
                assert!(!seen_null, "a value came out after a null under nulls last");
                non_null.push(v[row]);
            }
        }
    }
    assert_eq!(trailing_nulls, null_count, "nulls were lost or invented");
    assert_eq!(non_null, expected_values);
}

/// A spilling sort under row locking still names the rows it ordered, which
/// is the query shape that most needs to be able to spill.
#[tokio::test]
async fn locators_survive_a_spilling_sort() {
    struct LocatorFeed {
        batches: VecDeque<DataBatch>,
        next_page: u64,
    }
    impl Operator for LocatorFeed {
        fn next(&mut self) -> OperatorResult<'_> {
            Box::pin(async move {
                let Some(batch) = self.batches.pop_front() else {
                    return Ok(None);
                };
                let base = self.next_page;
                self.next_page += batch.num_rows as u64;
                let locators = (0..batch.num_rows as u64)
                    .map(|i| RowLocator::Heap {
                        page: PageId::new(0, base + i),
                        slot: 0,
                    })
                    .collect();
                Ok(Some(ExecutionBatch {
                    batch,
                    locators: Some(locators),
                }))
            })
        }
    }

    let mut batches = VecDeque::new();
    let mut pairs: Vec<(i64, u64)> = Vec::new();
    for chunk in 0..30u64 {
        let mut data = Vec::with_capacity(400);
        for row in 0..400u64 {
            let key = (((chunk * 400 + row) * 2_654_435_761) % 100_000) as i64;
            data.push(key);
            pairs.push((key, chunk * 400 + row));
        }
        batches.push_back(DataBatch::new(vec![Column::new(
            ColumnData::Int64(data),
            TypeId::Int64,
        )]));
    }
    pairs.sort_by_key(|(k, _)| *k);

    let mut op = SortOperator::new(
        Box::new(LocatorFeed {
            batches,
            next_page: 0,
        }),
        ascending(),
        schema(),
        None,
    )
    .with_locator_tracking();
    op.set_spill(Some(directory("locators")), TINY_BUDGET);

    let mut got: Vec<(i64, u64)> = Vec::new();
    while let Some(eb) = op.next().await.expect("sort") {
        let ColumnData::Int64(v) = &eb.batch.columns[0].data else {
            panic!("wrong type back");
        };
        let locators = eb.locators.expect("a locator tracking sort dropped them");
        assert_eq!(locators.len(), eb.batch.num_rows);
        for (row, loc) in locators.iter().enumerate() {
            let RowLocator::Heap { page, .. } = loc else {
                panic!("wrong locator kind back");
            };
            got.push((v[row], page.page_num));
        }
    }
    assert_eq!(got.len(), pairs.len());
    for (i, ((key, page), (expected_key, expected_page))) in
        got.iter().zip(pairs.iter()).enumerate()
    {
        assert_eq!(key, expected_key, "row {i} key");
        assert_eq!(page, expected_page, "row {i} kept the wrong locator");
    }
}

/// Text keys merge correctly, which exercises the variable-width side of the
/// spill codec and the comparator that reads it back.
#[tokio::test]
async fn text_keys_merge_correctly() {
    let mut values: Vec<String> = Vec::new();
    let mut state = 0x9E37_79B9_7F4A_7C15u64;
    for _ in 0..20_000 {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        values.push(format!("key-{:012}", state % 1_000_000_000));
    }
    let mut expected = values.clone();
    expected.sort();

    let batches: VecDeque<DataBatch> = values
        .chunks(500)
        .map(|chunk| {
            DataBatch::new(vec![Column::new(
                ColumnData::Utf8(chunk.to_vec()),
                TypeId::Text,
            )])
        })
        .collect();

    let mut text_schema = schema();
    text_schema[0].type_id = TypeId::Text;
    let mut order = ascending();
    if let BoundExpr::ColumnRef(cr) = &mut order[0].expr {
        cr.type_id = TypeId::Text;
    }

    let mut op = SortOperator::new(Box::new(Feed { batches }), order, text_schema, None);
    op.set_spill(Some(directory("text")), TINY_BUDGET);

    let mut out: Vec<String> = Vec::new();
    while let Some(eb) = op.next().await.expect("sort") {
        match &eb.batch.columns[0].data {
            ColumnData::Utf8(v) => out.extend_from_slice(v),
            other => panic!("wrong type back: {other:?}"),
        }
    }
    assert_eq!(out.len(), expected.len());
    assert_eq!(out, expected);
}
