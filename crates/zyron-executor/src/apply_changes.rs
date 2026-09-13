//! APPLY CHANGES as one pass over the target.
//!
//! The changes are ranked once per key, by the planner's own window
//! machinery, and held in memory keyed by the KEYS columns. Type 1 holds the
//! winning change per key. Type 2 holds every change of a key in sequence
//! order, because each one is a version of the row. The target is then read
//! once, and each of its rows is either left alone, deleted, updated in
//! place or, under type 2, walked forward through its key's changes, closed
//! and reopened as the tracked columns move. Whatever the change set holds
//! for a key the target does not have is inserted. Every write goes through
//! the same operators an INSERT, UPDATE or DELETE statement uses, so
//! constraints, indexes, generated columns, triggers, the change feed and
//! the replication changeset all see the apply as ordinary row writes.
//!
//! What this is not is a statement rewritten into DML with a subquery per row.
//! That form evaluates the ranked change set once per target row, which is
//! a cost that grows with the product of the two rather than their sum

use std::collections::HashMap;
use std::sync::Arc;

use zyron_catalog::{ColumnId, TableId};
use zyron_common::{Result, TypeId, ZyronError};
use zyron_planner::binder::{
    BoundAssignment, BoundExpectation, BoundExpr, BoundGeneratedColumn, ColumnRef,
};
use zyron_planner::logical::LogicalColumn;
use zyron_planner::physical::PhysicalPlan;

use crate::batch::{ColumnBuilder, DataBatch};
use crate::column::{Column, ScalarValue};
use crate::context::ExecutionContext;
use crate::operator::modify::{DeleteOperator, InsertOperator, UpdateOperator};
use crate::operator::{ExecutionBatch, Operator, OperatorResult};

/// What an INSERT into the target carries besides its rows, taken from the
/// plan of one
pub struct InsertShape {
    pub target_columns: Vec<ColumnId>,
    pub column_defaults: Vec<(ColumnId, BoundExpr)>,
    pub check_constraints: Vec<BoundExpr>,
    pub expectations: Vec<BoundExpectation>,
    pub generated_columns: Vec<BoundGeneratedColumn>,
}

/// What an UPDATE of the target carries besides its assignments
pub struct UpdateShape {
    pub check_constraints: Vec<BoundExpr>,
    pub generated_columns: Vec<BoundGeneratedColumn>,
}

/// The reserved columns a type 2 target carries
#[derive(Debug, Clone, Copy)]
pub struct HistoryColumns {
    pub start_at: ColumnId,
    pub end_at: ColumnId,
    pub is_current: ColumnId,
}

/// One APPLY CHANGES, resolved to plans and column positions
pub struct ApplyJob {
    pub table_id: TableId,
    /// The winning change per key, with the applied columns, the sequence
    /// value and the delete and truncate flags
    pub winners: PhysicalPlan,
    /// Every row of the target the apply may touch, with the reader's row
    /// security already in it
    pub target: PhysicalPlan,
    pub insert: InsertShape,
    pub update: UpdateShape,
    /// The target columns the apply writes, keys included, each with the
    /// position its value takes in a winners row
    pub applied: Vec<(ColumnId, usize)>,
    pub keys: Vec<ColumnId>,
    pub ignore_null_updates: bool,
    /// Position of the sequence value in a winners row
    pub sequence_at: usize,
    /// Position of the delete flag in a winners row
    pub is_delete_at: usize,
    /// Position of the truncate flag in a winners row, None when the
    /// statement recognizes no truncate
    pub is_truncate_at: Option<usize>,
    /// Position of the change's rank within its key in a winners row, one
    /// being the newest change of the key
    pub rank_at: usize,
    /// Type 2 when set, the reserved columns and the columns whose change
    /// opens a new version
    pub history: Option<(HistoryColumns, Vec<ColumnId>)>,
}

/// What an apply did
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct ApplyCounts {
    pub upserted: u64,
    pub deleted: u64,
    pub versioned: u64,
    /// Whether a change the set carried cleared the target before the
    /// rest landed
    pub truncated: bool,
}

/// Rows already in memory, handed to a write operator as its child
struct StagedSource {
    batches: std::vec::IntoIter<ExecutionBatch>,
}

impl StagedSource {
    fn new(batches: Vec<ExecutionBatch>) -> Box<dyn Operator> {
        Box::new(Self {
            batches: batches.into_iter(),
        })
    }
}

impl Operator for StagedSource {
    fn next(&mut self) -> OperatorResult<'_> {
        Box::pin(async move { Ok(self.batches.next()) })
    }
}

/// The changes the apply lands, held by key.
///
/// Every change sits in `ordered` beside the others of its key, oldest
/// first, and a key names the span of `ordered` that holds its chain. Type 1
/// chains are one change long
struct Winners {
    batches: Vec<DataBatch>,
    /// Batch and row of every change, grouped by key and ordered within a
    /// key by ascending sequence
    ordered: Vec<(usize, usize)>,
    /// Key values to the chain's first position in `ordered` and its length
    by_key: HashMap<Vec<ScalarValue>, (usize, usize)>,
    /// Whether the target held a row for the key, at the chain's first
    /// position
    seen: Vec<bool>,
}

impl Winners {
    fn value(&self, at: (usize, usize), position: usize) -> ScalarValue {
        self.batches[at.0].column(position).get_scalar(at.1)
    }

    fn is_set(&self, at: (usize, usize), position: usize) -> bool {
        match self.batches[at.0].column(position).get_scalar(at.1) {
            ScalarValue::Boolean(b) => b,
            _ => false,
        }
    }

    /// The newest change of a chain
    fn newest(&self, chain: (usize, usize)) -> (usize, usize) {
        self.ordered[chain.0 + chain.1 - 1]
    }

    /// A chain's changes, oldest first
    fn chain(&self, chain: (usize, usize)) -> &[(usize, usize)] {
        &self.ordered[chain.0..chain.0 + chain.1]
    }
}

/// Runs one apply and answers with what it did
pub async fn run_apply(ctx: &Arc<ExecutionContext>, job: ApplyJob) -> Result<ApplyCounts> {
    ctx.ensure_writable("APPLY CHANGES")?;
    let table = ctx.get_table_entry(job.table_id)?;
    let winners = collect_winners(ctx, &job).await?;
    let mut counts = ApplyCounts::default();

    // A source truncate clears the target before anything lands, so the
    // rows the same change set carries are what the target keeps
    if let Some(at) = job.is_truncate_at {
        let truncating = (0..winners.batches.len())
            .any(|b| (0..winners.batches[b].num_rows).any(|r| winners.is_set((b, r), at)));
        if truncating {
            let all = scan_target(ctx, &job).await?;
            counts.deleted += delete_rows(ctx, job.table_id, all).await?;
            counts.truncated = true;
        }
    }

    let target_schema = job.target.output_schema();
    let key_positions = positions_of(&target_schema, &job.keys, &table.name)?;
    let history = match &job.history {
        Some((columns, tracked)) => Some(HistoryPositions {
            columns: *columns,
            current_at: positions_of(&target_schema, &[columns.is_current], &table.name)?[0],
            start_at: positions_of(&target_schema, &[columns.start_at], &table.name)?[0],
            end_at: positions_of(&target_schema, &[columns.end_at], &table.name)?[0],
            tracked: tracked.clone(),
        }),
        None => None,
    };

    // The columns an update writes, every applied column but the keys, plus
    // the reserved ones under type 2. The new image of each row is computed
    // here, and the operator writes it the way it writes any assignment
    let mut assigned: Vec<ColumnId> = job
        .applied
        .iter()
        .map(|(id, _)| *id)
        .filter(|id| !job.keys.contains(id))
        .collect();
    if let Some((columns, _)) = &job.history {
        assigned.push(columns.end_at);
        assigned.push(columns.is_current);
    }
    let assigned_positions = positions_of(&target_schema, &assigned, &table.name)?;

    let mut winners = winners;
    let mut deletes: Vec<ExecutionBatch> = Vec::new();
    let mut updates: Vec<ExecutionBatch> = Vec::new();
    let mut versioned = 0u64;
    let mut upserts = 0u64;
    // The positions of the applied columns in a target row, in applied order
    let applied_positions: Vec<usize> = positions_of(
        &target_schema,
        &job.applied.iter().map(|(id, _)| *id).collect::<Vec<_>>(),
        &table.name,
    )?;
    let mut sink = InsertSink::new(&job, &table)?;

    let target_rows = scan_target_keyed(ctx, &job, &key_positions, &winners).await?;
    // Under type 2, the newest sequence value the target holds for each key
    // the change set names, over the key's current and closed rows alike. A
    // change no newer than it has already been absorbed, which is what makes
    // applying a range twice land the same history once, a deleted key
    // included
    let absorbed: HashMap<Vec<ScalarValue>, ScalarValue> = match &history {
        None => HashMap::new(),
        Some(h) => {
            let mut absorbed = HashMap::new();
            let mut key = Vec::with_capacity(key_positions.len());
            for staged in &target_rows {
                let batch = &staged.batch;
                for row in 0..batch.num_rows {
                    key.clear();
                    for at in &key_positions {
                        key.push(batch.column(*at).get_scalar(row));
                    }
                    if !winners.by_key.contains_key(&key) {
                        continue;
                    }
                    let start = batch.column(h.start_at).get_scalar(row);
                    let end = batch.column(h.end_at).get_scalar(row);
                    let newest = later_of(start, end);
                    match absorbed.get_mut(&key) {
                        Some(held) => {
                            let held: &mut ScalarValue = held;
                            *held = later_of(held.clone(), newest);
                        }
                        None => {
                            absorbed.insert(key.clone(), newest);
                        }
                    }
                }
            }
            absorbed
        }
    };
    for staged in target_rows {
        let Some(locators) = staged.locators.as_ref() else {
            return Err(ZyronError::Internal(
                "APPLY CHANGES read the target without row locators".to_string(),
            ));
        };
        let batch = &staged.batch;
        let rows = batch.num_rows;
        let mut delete_mask = vec![false; rows];
        let mut update_mask = vec![false; rows];
        let mut new_values: Vec<ColumnBuilder> = assigned_positions
            .iter()
            .map(|at| ColumnBuilder::new(batch.column(*at).type_id, rows))
            .collect();
        let mut key = Vec::with_capacity(key_positions.len());
        for row in 0..rows {
            if let Some(h) = &history {
                if !matches!(
                    batch.column(h.current_at).get_scalar(row),
                    ScalarValue::Boolean(true)
                ) {
                    continue;
                }
            }
            key.clear();
            for at in &key_positions {
                key.push(batch.column(*at).get_scalar(row));
            }
            let Some(chain) = winners.by_key.get(&key).copied() else {
                continue;
            };
            let at = winners.newest(chain);
            if job.is_truncate_at.is_some_and(|t| winners.is_set(at, t)) {
                continue;
            }
            winners.seen[chain.0] = true;

            match &history {
                None => {
                    if winners.is_set(at, job.is_delete_at) {
                        delete_mask[row] = true;
                        continue;
                    }
                    update_mask[row] = true;
                    upserts += 1;
                    for (slot, column) in assigned.iter().enumerate() {
                        let old = batch.column(assigned_positions[slot]).get_scalar(row);
                        new_values[slot].push_owned(next_value(&job, &winners, at, *column, old));
                    }
                }
                Some(h) => {
                    let held: Vec<ScalarValue> = applied_positions
                        .iter()
                        .map(|at| batch.column(*at).get_scalar(row))
                        .collect();
                    let fate = walk_history(
                        &job,
                        &winners,
                        h,
                        chain,
                        Some(held),
                        absorbed.get(&key),
                        &mut sink,
                        &mut versioned,
                        &mut upserts,
                    )?;
                    let (values, end, current) = match fate {
                        RowFate::Untouched => continue,
                        RowFate::Closed { end, values } => (values, Some(end), false),
                        RowFate::Updated(values) => (values, None, true),
                    };
                    update_mask[row] = true;
                    for (slot, column) in assigned.iter().enumerate() {
                        let old = batch.column(assigned_positions[slot]).get_scalar(row);
                        let value = if *column == h.columns.end_at {
                            end.clone().unwrap_or(old)
                        } else if *column == h.columns.is_current {
                            ScalarValue::Boolean(current)
                        } else {
                            match job.applied.iter().position(|(id, _)| id == column) {
                                Some(applied_slot) => values[applied_slot].clone(),
                                None => old,
                            }
                        };
                        new_values[slot].push_owned(value);
                    }
                }
            }
        }

        if delete_mask.iter().any(|d| *d) {
            let kept: Vec<zyron_common::RowLocator> = locators
                .iter()
                .zip(delete_mask.iter())
                .filter(|(_, keep)| **keep)
                .map(|(l, _)| *l)
                .collect();
            deletes.push(ExecutionBatch::with_locators(
                batch.filter(&delete_mask),
                kept,
            ));
        }
        if update_mask.iter().any(|u| *u) {
            let kept: Vec<zyron_common::RowLocator> = locators
                .iter()
                .zip(update_mask.iter())
                .filter(|(_, keep)| **keep)
                .map(|(l, _)| *l)
                .collect();
            let mut columns = batch.filter(&update_mask).columns;
            for builder in new_values {
                columns.push(builder.finish());
            }
            updates.push(ExecutionBatch::with_locators(DataBatch::new(columns), kept));
        }
    }

    counts.deleted += delete_rows(ctx, job.table_id, deletes).await?;
    if !updates.is_empty() {
        let mut schema = target_schema.clone();
        let mut assignments = Vec::with_capacity(assigned.len());
        for (slot, column) in assigned.iter().enumerate() {
            let source = &target_schema[assigned_positions[slot]];
            // The new value sits beside the old image under its own table
            // index, so an assignment names it and nothing else
            schema.push(LogicalColumn {
                table_idx: Some(1),
                ..source.clone()
            });
            assignments.push(BoundAssignment {
                column_id: *column,
                value: BoundExpr::ColumnRef(ColumnRef {
                    table_idx: 1,
                    column_id: *column,
                    type_id: source.type_id,
                    nullable: true,
                    fractional_digits: source.fractional_digits,
                }),
            });
        }
        let mut op = UpdateOperator::new(
            StagedSource::new(updates),
            Arc::clone(ctx),
            job.table_id,
            assignments,
            schema,
            job.update.check_constraints.clone(),
            job.update.generated_columns.clone(),
        );
        drain(&mut op).await?;
    }
    counts.upserted += upserts;
    counts.versioned += versioned;

    // What the target had no row for opens one. Under type 1 that is the
    // key's winning change. Under type 2 it is the key's chain walked from
    // nothing, which opens a version per change and closes each one the
    // next change replaces
    match &history {
        None => sink.push_unseen_winners(&job, &winners)?,
        Some(h) => {
            let chains: Vec<(&Vec<ScalarValue>, (usize, usize))> = winners
                .by_key
                .iter()
                .map(|(key, chain)| (key, *chain))
                .filter(|(_, chain)| !winners.seen[chain.0])
                .collect();
            for (key, chain) in chains {
                walk_history(
                    &job,
                    &winners,
                    h,
                    chain,
                    None,
                    absorbed.get(key),
                    &mut sink,
                    &mut versioned,
                    &mut upserts,
                )?;
            }
        }
    }
    let inserts = sink.finish();
    if !inserts.is_empty() {
        let inserted: u64 = inserts.iter().map(|b| b.batch.num_rows as u64).sum();
        let mut op = InsertOperator::new(
            StagedSource::new(inserts),
            Arc::clone(ctx),
            job.table_id,
            job.insert.target_columns.clone(),
            job.insert.column_defaults.clone(),
            job.insert.check_constraints.clone(),
            job.insert.expectations.clone(),
            job.insert.generated_columns.clone(),
        );
        drain(&mut op).await?;
        counts.upserted += inserted;
    }
    Ok(counts)
}

/// What a type 2 walk did to the current row the target held for a key
enum RowFate {
    /// No change of the key touched it
    Untouched,
    /// A change closed it at `end`, holding `values` for the applied
    /// columns as they stood when it closed
    Closed {
        end: ScalarValue,
        values: Vec<ScalarValue>,
    },
    /// Only untracked columns moved, so it stays current with `values`
    Updated(Vec<ScalarValue>),
}

/// The version a type 2 walk has open
enum Open {
    /// The row the target holds, with the applied columns' values as they
    /// stand and whether an untracked change has moved them
    Target {
        values: Vec<ScalarValue>,
        moved: bool,
    },
    /// A version this apply opened, at `start`
    Version {
        start: ScalarValue,
        values: Vec<ScalarValue>,
    },
}

/// Walks one key's changes in sequence order under type 2, from the row the
/// target holds when it holds one.
///
/// A delete closes the open version at the change's sequence value. A
/// change to a tracked column closes the open version there and opens a new
/// one at the same value. A change to untracked columns alone moves the open
/// version's values in place. A version this walk opened and a later change
/// closed is inserted closed, the one still open at the end is inserted
/// current, and what happened to the target's own row is answered.
///
/// A change no newer than `absorbed`, the newest sequence value the target
/// holds for the key, is one the target has already taken, so it is passed
/// over. That is what makes applying a range twice land the same history
/// once
#[allow(clippy::too_many_arguments)]
fn walk_history(
    job: &ApplyJob,
    winners: &Winners,
    h: &HistoryPositions,
    chain: (usize, usize),
    held: Option<Vec<ScalarValue>>,
    absorbed: Option<&ScalarValue>,
    sink: &mut InsertSink,
    versioned: &mut u64,
    upserts: &mut u64,
) -> Result<RowFate> {
    let mut open = held.map(|values| Open::Target {
        values,
        moved: false,
    });
    let mut fate = RowFate::Untouched;
    for at in winners.chain(chain) {
        let at = *at;
        if job.is_truncate_at.is_some_and(|t| winners.is_set(at, t)) {
            continue;
        }
        let is_delete = winners.is_set(at, job.is_delete_at);
        let sequence = winners.value(at, job.sequence_at);
        if absorbed.is_some_and(|absorbed| {
            matches!(
                sequence.partial_cmp(absorbed),
                Some(std::cmp::Ordering::Less | std::cmp::Ordering::Equal)
            )
        }) {
            continue;
        }
        open = match open.take() {
            None => {
                // Nothing is open, so a delete finds nothing and any other
                // change opens the first version of this walk
                if is_delete {
                    None
                } else {
                    Some(Open::Version {
                        start: sequence,
                        values: applied_values(job, winners, at, None),
                    })
                }
            }
            Some(Open::Target { values, moved }) => {
                if is_delete {
                    *versioned += 1;
                    fate = RowFate::Closed {
                        end: sequence,
                        values,
                    };
                    None
                } else {
                    let next = applied_values(job, winners, at, Some(&values));
                    if tracked_moved(job, h, &values, &next) {
                        *versioned += 1;
                        fate = RowFate::Closed {
                            end: sequence.clone(),
                            values,
                        };
                        Some(Open::Version {
                            start: sequence,
                            values: next,
                        })
                    } else {
                        if !moved {
                            *upserts += 1;
                        }
                        fate = RowFate::Updated(next.clone());
                        Some(Open::Target {
                            values: next,
                            moved: true,
                        })
                    }
                }
            }
            Some(Open::Version { start, values }) => {
                if is_delete {
                    *versioned += 1;
                    sink.push(job, &values, start, Some(sequence), false);
                    None
                } else {
                    let next = applied_values(job, winners, at, Some(&values));
                    if tracked_moved(job, h, &values, &next) {
                        *versioned += 1;
                        sink.push(job, &values, start, Some(sequence.clone()), false);
                        Some(Open::Version {
                            start: sequence,
                            values: next,
                        })
                    } else {
                        Some(Open::Version {
                            start,
                            values: next,
                        })
                    }
                }
            }
        };
    }
    if let Some(Open::Version { start, values }) = open {
        sink.push(job, &values, start, None, true);
    }
    Ok(fate)
}

/// The applied columns' values a change carries, in applied order, with
/// IGNORE NULL UPDATES leaving what the open version holds when the change
/// carries NULL
fn applied_values(
    job: &ApplyJob,
    winners: &Winners,
    at: (usize, usize),
    previous: Option<&[ScalarValue]>,
) -> Vec<ScalarValue> {
    job.applied
        .iter()
        .enumerate()
        .map(|(slot, (_, position))| {
            let value = winners.value(at, *position);
            match previous {
                Some(previous) if job.ignore_null_updates && matches!(value, ScalarValue::Null) => {
                    previous[slot].clone()
                }
                _ => value,
            }
        })
        .collect()
}

/// The later of two sequence values, with NULL, which an open row's end
/// holds, reading as the earliest of all
fn later_of(a: ScalarValue, b: ScalarValue) -> ScalarValue {
    match a.partial_cmp(&b) {
        Some(std::cmp::Ordering::Less) => b,
        _ => a,
    }
}

/// Whether a change moves any column that opens a new version
fn tracked_moved(
    job: &ApplyJob,
    h: &HistoryPositions,
    open: &[ScalarValue],
    next: &[ScalarValue],
) -> bool {
    h.tracked.iter().any(|column| {
        job.applied
            .iter()
            .position(|(id, _)| id == column)
            .is_some_and(|slot| !scalars_equal(&open[slot], &next[slot]))
    })
}

/// Type 2 positions resolved against the target's scan schema
struct HistoryPositions {
    columns: HistoryColumns,
    current_at: usize,
    start_at: usize,
    end_at: usize,
    tracked: Vec<ColumnId>,
}

/// Runs a write operator to completion
async fn drain(op: &mut dyn Operator) -> Result<()> {
    while op.next().await?.is_some() {}
    Ok(())
}

/// Deletes the staged rows, answering with how many
async fn delete_rows(
    ctx: &Arc<ExecutionContext>,
    table_id: TableId,
    batches: Vec<ExecutionBatch>,
) -> Result<u64> {
    if batches.is_empty() {
        return Ok(0);
    }
    let rows: u64 = batches.iter().map(|b| b.batch.num_rows as u64).sum();
    let mut op = DeleteOperator::new(StagedSource::new(batches), Arc::clone(ctx), table_id);
    drain(&mut op).await?;
    Ok(rows)
}

/// Reads every target row the apply may touch, with its locator
async fn scan_target(ctx: &Arc<ExecutionContext>, job: &ApplyJob) -> Result<Vec<ExecutionBatch>> {
    let built = crate::executor::build_scan_with_tuple_ids(job.target.clone(), ctx).await?;
    let mut op = built.op;
    let mut out = Vec::new();
    while let Some(batch) = op.next().await? {
        ctx.check_cancelled()?;
        if batch.batch.num_rows > 0 {
            out.push(batch);
        }
    }
    Ok(out)
}

/// The target's rows whose key the change set names, kept as the scan
/// streams past and the rest dropped, so what the apply holds is the
/// change set's worth of the target rather than the whole of it. A closed
/// type 2 row of a named key is kept the same way, since what the key has
/// absorbed is read from it
async fn scan_target_keyed(
    ctx: &Arc<ExecutionContext>,
    job: &ApplyJob,
    key_positions: &[usize],
    winners: &Winners,
) -> Result<Vec<ExecutionBatch>> {
    let built = crate::executor::build_scan_with_tuple_ids(job.target.clone(), ctx).await?;
    let mut op = built.op;
    let mut out = Vec::new();
    let mut key = Vec::with_capacity(key_positions.len());
    while let Some(staged) = op.next().await? {
        ctx.check_cancelled()?;
        let rows = staged.batch.num_rows;
        if rows == 0 {
            continue;
        }
        let mut mask = vec![false; rows];
        let mut kept = 0usize;
        for row in 0..rows {
            key.clear();
            for at in key_positions {
                key.push(staged.batch.column(*at).get_scalar(row));
            }
            if winners.by_key.contains_key(&key) {
                mask[row] = true;
                kept += 1;
            }
        }
        if kept == 0 {
            continue;
        }
        if kept == rows {
            out.push(staged);
            continue;
        }
        let locators = staged.locators.as_ref().map(|locators| {
            locators
                .iter()
                .zip(&mask)
                .filter(|(_, keep)| **keep)
                .map(|(locator, _)| locator.clone())
                .collect()
        });
        out.push(ExecutionBatch {
            batch: staged.batch.filter(&mask),
            locators,
        });
    }
    Ok(out)
}

/// Executes the winners plan and indexes the rows by key.
///
/// Each applied column is converted to the target column's type here, so a
/// source that carries a wider integer or a narrower string lands as the
/// target declares it and a key compares equal to the target's own
async fn collect_winners(ctx: &Arc<ExecutionContext>, job: &ApplyJob) -> Result<Winners> {
    let mut batches = crate::execute(job.winners.clone(), ctx).await?;
    let table = ctx.get_table_entry(job.table_id)?;
    // The sequence value is compared with the start a type 2 row holds,
    // which was written from it into the start column, so it is read in
    // that column's type
    let sequence_type = job.history.as_ref().and_then(|(columns, _)| {
        table
            .columns
            .iter()
            .find(|c| c.id == columns.start_at)
            .map(|c| (c.type_id, c.fractional_digits))
    });
    for batch in batches.iter_mut() {
        if let Some((type_id, digits)) = sequence_type {
            let held = &batch.columns[job.sequence_at];
            if type_id == TypeId::Decimal {
                batch.columns[job.sequence_at] =
                    crate::compute::cast_column_to_decimal(held, digits.unwrap_or(0))?;
            } else if held.type_id != type_id {
                batch.columns[job.sequence_at] = crate::compute::cast_column(held, type_id)?;
            }
        }
        for (column, position) in &job.applied {
            let Some(own) = table.columns.iter().find(|c| c.id == *column) else {
                continue;
            };
            let held = &batch.columns[*position];
            let converted = if own.type_id == TypeId::Decimal {
                crate::compute::cast_column_to_decimal(held, own.fractional_digits.unwrap_or(0))?
            } else if held.type_id != own.type_id {
                crate::compute::cast_column(held, own.type_id)?
            } else {
                continue;
            };
            batch.columns[*position] = converted;
        }
    }
    let key_positions: Vec<usize> = job
        .keys
        .iter()
        .map(|key| {
            job.applied
                .iter()
                .find(|(id, _)| id == key)
                .map(|(_, at)| *at)
                .ok_or_else(|| {
                    ZyronError::Internal(format!(
                        "APPLY CHANGES key column {key:?} is not among the applied columns"
                    ))
                })
        })
        .collect::<Result<Vec<_>>>()?;
    // Each key's chain is laid out once its length is known, and a change
    // lands in it by rank, the newest change, ranked one, at the chain's
    // end and the oldest at its start
    let total: usize = batches.iter().map(|b| b.num_rows).sum();
    let mut by_key: HashMap<Vec<ScalarValue>, (usize, usize)> = HashMap::with_capacity(total);
    for batch in &batches {
        for r in 0..batch.num_rows {
            let key: Vec<ScalarValue> = key_positions
                .iter()
                .map(|at| batch.column(*at).get_scalar(r))
                .collect();
            by_key.entry(key).or_insert((0, 0)).1 += 1;
        }
    }
    let mut offset = 0usize;
    for chain in by_key.values_mut() {
        chain.0 = offset;
        offset += chain.1;
    }
    let unplaced = (usize::MAX, usize::MAX);
    let mut ordered = vec![unplaced; total];
    for (b, batch) in batches.iter().enumerate() {
        for r in 0..batch.num_rows {
            let key: Vec<ScalarValue> = key_positions
                .iter()
                .map(|at| batch.column(*at).get_scalar(r))
                .collect();
            let Some(&(start, len)) = by_key.get(&key) else {
                continue;
            };
            let rank = match batch.column(job.rank_at).get_scalar(r) {
                ScalarValue::Int64(rank) => rank,
                ScalarValue::Int32(rank) => rank as i64,
                other => {
                    return Err(ZyronError::Internal(format!(
                        "APPLY CHANGES ranked a change as {other:?} rather than a number"
                    )));
                }
            };
            if rank < 1 || rank as usize > len {
                return Err(ZyronError::Internal(format!(
                    "APPLY CHANGES ranked a change {rank} within a key of {len} changes"
                )));
            }
            let slot = start + len - rank as usize;
            if ordered[slot] != unplaced {
                return Err(ZyronError::ExecutionError(
                    "APPLY CHANGES found two changes of one key at one rank, which the \
                     numbering should have told apart"
                        .to_string(),
                ));
            }
            ordered[slot] = (b, r);
        }
    }
    Ok(Winners {
        batches,
        ordered,
        by_key,
        seen: vec![false; total],
    })
}

/// The value a winner supplies for a target column, None when the apply
/// does not write that column
fn winner_value(
    job: &ApplyJob,
    winners: &Winners,
    at: (usize, usize),
    column: ColumnId,
) -> Option<ScalarValue> {
    job.applied
        .iter()
        .find(|(id, _)| *id == column)
        .map(|(_, position)| winners.value(at, *position))
}

/// The value a target column takes from a winning change, with IGNORE
/// NULL UPDATES leaving what the target holds when the change carries NULL
fn next_value(
    job: &ApplyJob,
    winners: &Winners,
    at: (usize, usize),
    column: ColumnId,
    old: ScalarValue,
) -> ScalarValue {
    match winner_value(job, winners, at, column) {
        Some(ScalarValue::Null) if job.ignore_null_updates => old,
        Some(value) => value,
        None => old,
    }
}

/// Whether two values are the same for the purpose of opening a version.
/// NULL matches NULL here, the way IS DISTINCT FROM reads it
fn scalars_equal(a: &ScalarValue, b: &ScalarValue) -> bool {
    a == b
}

fn target_position(schema: &[LogicalColumn], column: ColumnId) -> Option<usize> {
    schema.iter().position(|c| c.column_id == column)
}

fn positions_of(schema: &[LogicalColumn], columns: &[ColumnId], table: &str) -> Result<Vec<usize>> {
    columns
        .iter()
        .map(|column| {
            target_position(schema, *column).ok_or_else(|| {
                ZyronError::Internal(format!(
                    "APPLY CHANGES column {column:?} is not in the scan of table '{table}'"
                ))
            })
        })
        .collect()
}

/// The rows the apply inserts, shaped to the insert's column list and
/// handed over a batch at a time
struct InsertSink {
    types: Vec<TypeId>,
    /// For each insert column, the applied slot its value comes from, None
    /// for a reserved history column
    from_applied: Vec<Option<usize>>,
    /// The reserved columns under type 2
    history: Option<HistoryColumns>,
    builders: Vec<ColumnBuilder>,
    rows: usize,
    out: Vec<ExecutionBatch>,
}

impl InsertSink {
    fn new(job: &ApplyJob, table: &zyron_catalog::TableEntry) -> Result<Self> {
        let types: Vec<TypeId> = job
            .insert
            .target_columns
            .iter()
            .map(|column| {
                table
                    .columns
                    .iter()
                    .find(|c| c.id == *column)
                    .map(|c| c.type_id)
                    .ok_or_else(|| {
                        ZyronError::Internal(format!(
                            "APPLY CHANGES inserts column {column:?}, which table '{}' does not \
                             have",
                            table.name
                        ))
                    })
            })
            .collect::<Result<Vec<_>>>()?;
        let from_applied = job
            .insert
            .target_columns
            .iter()
            .map(|column| job.applied.iter().position(|(id, _)| id == column))
            .collect();
        let builders = types
            .iter()
            .map(|t| ColumnBuilder::new(*t, crate::batch::BATCH_SIZE))
            .collect();
        Ok(Self {
            types,
            from_applied,
            history: job.history.as_ref().map(|(columns, _)| *columns),
            builders,
            rows: 0,
            out: Vec::new(),
        })
    }

    /// One row to insert, the applied columns' values and, under type 2,
    /// the version's bounds and whether it is the current one
    fn push(
        &mut self,
        job: &ApplyJob,
        values: &[ScalarValue],
        start: ScalarValue,
        end: Option<ScalarValue>,
        current: bool,
    ) {
        for (slot, column) in job.insert.target_columns.iter().enumerate() {
            let value = match (self.history, self.from_applied[slot]) {
                (Some(h), _) if *column == h.start_at => start.clone(),
                (Some(h), _) if *column == h.end_at => end.clone().unwrap_or(ScalarValue::Null),
                (Some(h), _) if *column == h.is_current => ScalarValue::Boolean(current),
                (_, Some(applied)) => values[applied].clone(),
                (_, None) => ScalarValue::Null,
            };
            self.builders[slot].push_owned(value);
        }
        self.rows += 1;
        if self.rows == crate::batch::BATCH_SIZE {
            self.flush();
        }
    }

    /// Every winning change whose key the target had no row for, under
    /// type 1. A delete of a row that is not there inserts nothing, and
    /// neither does a truncate, which names no row
    fn push_unseen_winners(&mut self, job: &ApplyJob, winners: &Winners) -> Result<()> {
        let chains: Vec<(usize, usize)> = winners
            .by_key
            .values()
            .copied()
            .filter(|chain| !winners.seen[chain.0])
            .collect();
        for chain in chains {
            let at = winners.newest(chain);
            if winners.is_set(at, job.is_delete_at) {
                continue;
            }
            if job.is_truncate_at.is_some_and(|t| winners.is_set(at, t)) {
                continue;
            }
            let values = applied_values(job, winners, at, None);
            self.push(job, &values, ScalarValue::Null, None, true);
        }
        Ok(())
    }

    fn flush(&mut self) {
        if self.rows == 0 {
            return;
        }
        let columns: Vec<Column> = std::mem::replace(
            &mut self.builders,
            self.types
                .iter()
                .map(|t| ColumnBuilder::new(*t, crate::batch::BATCH_SIZE))
                .collect(),
        )
        .into_iter()
        .map(|builder| builder.finish())
        .collect();
        self.out.push(ExecutionBatch::new(DataBatch::new(columns)));
        self.rows = 0;
    }

    fn finish(mut self) -> Vec<ExecutionBatch> {
        self.flush();
        self.out
    }
}
