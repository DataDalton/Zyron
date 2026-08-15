//! Deciding which columnar segments a tier relocation may move.
//!
//! `ALTER TABLE ... MOVE WHERE <predicate> TO TIER <t>` names rows, but the
//! unit that relocates is a whole `.zyr` file. A segment can move only when
//! every row in it that the statement's snapshot can see satisfies the
//! predicate, because moving one holding a row the predicate excludes would
//! put that row on a tier nobody asked for.
//!
//! The coverage pass reads each segment once through the ordinary columnar
//! scan, so MVCC visibility and the patch overlay apply exactly as they do
//! to a query. A segment whose rows have all been superseded reports zero
//! live rows and is left alone: relocating it would move bytes no read can
//! reach, and the merge pass is what removes it.

use std::collections::HashSet;
use std::sync::Arc;

use zyron_common::Result;
use zyron_planner::binder::BoundExpr;
use zyron_planner::logical::LogicalColumn;

use crate::column::ScalarValue;
use crate::compute::column_to_mask;
use crate::context::ExecutionContext;
use crate::expr::evaluate;
use crate::operator::Operator;
use crate::operator::column_scan::ColumnScanOperator;

/// How much of one segment a predicate covers.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SegmentCoverage {
    pub file_id: u64,
    /// Rows in the file visible to the scanning snapshot.
    pub live_rows: u64,
    /// Visible rows the predicate holds for.
    pub matching_rows: u64,
}

impl SegmentCoverage {
    /// True when the predicate holds for every visible row, which is what
    /// makes the whole file movable. A segment with no visible rows is not
    /// covered: there is nothing to move it on behalf of.
    #[inline]
    pub fn fully_covered(&self) -> bool {
        self.live_rows > 0 && self.matching_rows == self.live_rows
    }
}

/// Reads every registered segment of `table_id` and reports, per segment, how
/// many visible rows it holds and how many of those satisfy `predicate`.
///
/// The predicate is bound against the table at table_idx 0, matching
/// `bind_table_predicate`, so `columns` must be the table's full column list
/// in ordinal order for the column references to resolve.
pub async fn segment_predicate_coverage(
    ctx: &Arc<ExecutionContext>,
    table_id: zyron_catalog::TableId,
    columns: &[LogicalColumn],
    predicate: &BoundExpr,
) -> Result<Vec<SegmentCoverage>> {
    let table_entry = ctx.get_table_entry(table_id)?;
    let file_ids: Vec<u64> = table_entry
        .columnar
        .segments
        .iter()
        .map(|s| s.file_id)
        .collect();

    let no_params: Vec<ScalarValue> = Vec::new();
    let mut out = Vec::with_capacity(file_ids.len());
    for file_id in file_ids {
        let mut only = HashSet::with_capacity(1);
        only.insert(file_id);
        // The scan takes no predicate of its own: it would prune segments by
        // their zone maps, and a pruned segment yields no rows, which would
        // read back as full coverage of an empty file
        let mut scan = ColumnScanOperator::new_for_files(
            Arc::clone(ctx),
            table_id,
            columns.to_vec(),
            None,
            only,
        )?;
        let mut live_rows: u64 = 0;
        let mut matching_rows: u64 = 0;
        while let Some(exec_batch) = scan.next().await? {
            let batch = exec_batch.batch;
            live_rows += batch.num_rows as u64;
            let mask_col = evaluate(predicate, &batch, columns, &no_params)?;
            matching_rows += column_to_mask(&mask_col).iter().filter(|k| **k).count() as u64;
        }
        out.push(SegmentCoverage {
            file_id,
            live_rows,
            matching_rows,
        });
    }
    Ok(out)
}
