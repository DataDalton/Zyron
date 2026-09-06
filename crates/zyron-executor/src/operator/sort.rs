//! Sort operator for ordering results.
//!
//! Materializes the child output, computes sort indices, reorders
//! all data in a single take() pass, then emits the sorted batch as
//! one output by move, no copy. Uses radix sort for integer key types.
//!
//! ## Under a limit
//!
//! The buffer is bounded. Once it holds a batch or a limit's worth beyond
//! the limit, a selection pass cuts it back to the limit, and the row at
//! the limit becomes a bound that a later row has to beat to be buffered
//! at all. The input is then read in one pass holding about a batch beyond
//! the limit, and only the rows the limit keeps are ever sorted.
//!
//! ## When the input does not fit
//!
//! The whole-input path above is the fast one and is what runs whenever the
//! input fits, untouched. A sort that would outgrow its memory budget instead
//! writes sorted runs to spill files and merges them on the way out, so the
//! answer is the same and only the speed changes.
//!
//! Two properties of that path are worth stating because they are what make it
//! worth having:
//!
//! - **The merge streams.** Runs are read a batch at a time and the output is
//!   produced incrementally, so the merged result never exists in memory all
//!   at once. Materializing it would spend the memory the spill was for.
//! - **Nothing is spilled unless it has to be.** The threshold is the budget
//!   the query was given, so a sort that fits pays nothing for this: no run is
//!   written, no file is created, and the code below the threshold is the code
//!   that ran before.

use std::borrow::Cow;

use zyron_common::{Result, RowLocator, ZyronError};
use zyron_planner::binder::{BoundExpr, BoundOrderBy};
use zyron_planner::logical::LogicalColumn;

use crate::batch::DataBatch;
use crate::column::{Column, ColumnData};
use crate::compute;
use crate::expr::{evaluate, evaluate_borrowed, resolve_column_index};
use crate::operator::{ExecutionBatch, Operator, OperatorResult};

/// Rows a bounded sort buffers before cutting back to its limit.
///
/// A cut is a selection pass over the buffer, so it waits until the buffer
/// holds a batch or another limit's worth beyond the limit, whichever is
/// more. Every cut then discards at least as many rows as it keeps, and the
/// whole input costs at most two passes of selection
fn top_n_trigger(keep: usize) -> usize {
    keep.saturating_add(keep.max(crate::batch::BATCH_SIZE))
}

/// What a cut leaves: the rows the limit keeps, their locators when they
/// are tracked, and the one-row key of the row at the limit
struct CutBuffer {
    batch: DataBatch,
    locators: Option<Vec<RowLocator>>,
    bound: Vec<Column>,
}

/// Sorts all input rows by the given order-by expressions.
/// Materializes the entire input before producing output.
/// If limit is set, only the top-N rows are retained.
pub struct SortOperator {
    child: Box<dyn Operator>,
    order_by: Vec<BoundOrderBy>,
    input_schema: Vec<LogicalColumn>,
    limit: Option<u64>,
    /// Fully sorted batch, emitted whole by move on the first next() call
    sorted_batch: Option<DataBatch>,
    finished: bool,
    /// Carry row locators through the sort permutation. Set only when the
    /// sort feeds a row-locking operator, the normal path pays nothing
    track_locators: bool,
    /// Locators permuted into sorted order, emitted alongside sorted_batch
    sorted_locators: Option<Vec<RowLocator>>,
    /// Query memory budget the buffered input reserves against. None runs
    /// unbudgeted.
    memory_budget: Option<std::sync::Arc<crate::context::QueryMemoryBudget>>,
    /// Where runs go when the input outgrows the budget. None means the sort
    /// has nowhere to spill and fails instead, which is what a node with no
    /// data directory does
    spill: Option<std::sync::Arc<crate::spill::SpillDirectory>>,
    /// Bytes the budget allows before a run is written. Zero means never
    spill_threshold_bytes: u64,
    /// Sorted runs already on disk, merged on the way out
    runs: Vec<crate::spill::SpillReader>,
    /// The streaming merge, built on the first next() once runs exist
    merge: Option<MergeState>,
    /// Rows already emitted, so a limit is honoured across a merge
    emitted: u64,
    /// Context whose cancel flag the merge phase polls between output
    /// batches. The input side is covered by the consumer-side poll wrapper,
    /// but the merge runs after consumption and reads spill files without
    /// touching a tokio resource, so it polls and yields on its own
    poll_ctx: Option<std::sync::Arc<crate::context::ExecutionContext>>,
    /// Merge batches emitted since the last explicit yield
    merge_batches: u32,
}

impl SortOperator {
    pub fn new(
        child: Box<dyn Operator>,
        order_by: Vec<BoundOrderBy>,
        input_schema: Vec<LogicalColumn>,
        limit: Option<u64>,
    ) -> Self {
        Self {
            child,
            order_by,
            input_schema,
            limit,
            sorted_batch: None,
            finished: false,
            track_locators: false,
            sorted_locators: None,
            memory_budget: None,
            spill: None,
            spill_threshold_bytes: 0,
            runs: Vec::new(),
            merge: None,
            emitted: 0,
            poll_ctx: None,
            merge_batches: 0,
        }
    }

    /// Installs the context whose cancel flag the spill merge polls.
    pub fn set_poll_context(&mut self, ctx: std::sync::Arc<crate::context::ExecutionContext>) {
        self.poll_ctx = Some(ctx);
    }

    /// Attaches the query memory budget. Set by the operator builder from
    /// the execution context.
    pub fn set_memory_budget(
        &mut self,
        budget: Option<std::sync::Arc<crate::context::QueryMemoryBudget>>,
    ) {
        self.memory_budget = budget;
    }

    /// Gives the sort somewhere to put the input that does not fit.
    ///
    /// The threshold is what the query may hold, not what the machine has: a
    /// sort spills when it would exceed its own budget, which is the same
    /// point at which it used to fail.
    pub fn set_spill(
        &mut self,
        directory: Option<std::sync::Arc<crate::spill::SpillDirectory>>,
        threshold_bytes: u64,
    ) {
        self.spill = directory;
        self.spill_threshold_bytes = threshold_bytes;
    }

    /// Carries row locators through the sort so a row-locking operator
    /// above still addresses storage rows. Disables the indices-free
    /// in-place fast path, which cannot express its permutation
    pub fn with_locator_tracking(mut self) -> Self {
        self.track_locators = true;
        self
    }

    async fn materialize(&mut self) -> Result<()> {
        // A limit of nothing keeps nothing, and the input need not be read
        // to say so
        if self.limit == Some(0) {
            self.finished = true;
            return Ok(());
        }

        let mut all_columns: Vec<Vec<Column>> = Vec::new();
        let mut all_locators: Vec<RowLocator> = Vec::new();
        let mut total_rows = 0usize;

        // Bytes held in memory right now. With somewhere to spill it decides
        // when a run is written, distinct from the query budget: the budget
        // then stops being a cap on the whole sort and becomes a cap on how
        // much of it is resident at once. Under a limit it is what a cut
        // hands back to the budget
        let mut resident_bytes = 0u64;
        let spilling = self.spill.is_some() && self.spill_threshold_bytes > 0;

        // A limit bounds the buffer. Past the trigger it is cut back to the
        // limit, and the row at the limit becomes the bound a later row has
        // to beat to be buffered at all
        let keep = self.limit.map(|l| l as usize);
        let trigger = keep.map(top_n_trigger);
        let (ascending, nulls_first) = self.directions();
        let mut bound: Option<Vec<Column>> = None;

        loop {
            let Some(mut eb) = self.child.next().await? else {
                break;
            };
            if eb.batch.num_rows == 0 {
                continue;
            }

            if let Some(bound) = bound.as_ref() {
                let keys = self.batch_keys(&eb.batch)?;
                let key_refs: Vec<&Column> = keys.iter().map(|c| c.as_ref()).collect();
                let bound_refs: Vec<&Column> = bound.iter().collect();
                let ahead = compute::rows_before_bound(
                    &key_refs,
                    &bound_refs,
                    &ascending,
                    &nulls_first,
                    eb.batch.num_rows,
                );
                if ahead.is_empty() {
                    continue;
                }
                if ahead.len() < eb.batch.num_rows {
                    let locators = eb
                        .locators
                        .map(|locs| ahead.iter().map(|&i| locs[i as usize]).collect());
                    eb = ExecutionBatch {
                        batch: eb.batch.take(&ahead),
                        locators,
                    };
                }
            }

            let batch_bytes = eb.batch.approx_bytes();
            if !spilling {
                if let Some(budget) = &self.memory_budget {
                    // No spill directory, so the budget is the hard limit
                    // it always was and exceeding it is still a failure
                    budget.reserve(batch_bytes)?;
                }
            }
            resident_bytes += batch_bytes;
            total_rows += eb.batch.num_rows;
            if all_columns.is_empty() {
                all_columns.resize_with(eb.batch.num_columns(), Vec::new);
            }
            if self.track_locators {
                let locs = eb.locators.ok_or_else(|| {
                    ZyronError::ExecutionError(
                        "sort under row locking received a batch without row locators".to_string(),
                    )
                })?;
                all_locators.extend(locs);
            }
            for (i, col) in eb.batch.columns.into_iter().enumerate() {
                all_columns[i].push(col);
            }

            if let (Some(keep), Some(trigger)) = (keep, trigger) {
                if total_rows >= trigger {
                    let cut = self.cut_to_limit(
                        std::mem::take(&mut all_columns),
                        std::mem::take(&mut all_locators),
                        total_rows,
                        keep,
                        &ascending,
                        &nulls_first,
                    )?;
                    let kept_bytes = cut.batch.approx_bytes();
                    if !spilling {
                        if let Some(budget) = &self.memory_budget {
                            budget.release(resident_bytes.saturating_sub(kept_bytes));
                        }
                    }
                    resident_bytes = kept_bytes;
                    total_rows = cut.batch.num_rows;
                    all_columns = cut.batch.columns.into_iter().map(|c| vec![c]).collect();
                    all_locators = cut.locators.unwrap_or_default();
                    bound = Some(cut.bound);
                }
            }

            if spilling && resident_bytes >= self.spill_threshold_bytes {
                self.flush_run(
                    std::mem::take(&mut all_columns),
                    std::mem::take(&mut all_locators),
                    total_rows,
                )?;
                total_rows = 0;
                resident_bytes = 0;
            }
        }

        if total_rows == 0 && self.runs.is_empty() {
            self.finished = true;
            return Ok(());
        }

        // Runs on disk, so the tail joins them and the answer comes out of a
        // merge rather than a single sort
        if !self.runs.is_empty() {
            if total_rows > 0 {
                self.flush_run(all_columns, all_locators, total_rows)?;
            }
            self.merge = Some(self.build_merge()?);
            return Ok(());
        }

        // Single-key integer ColumnRef: radix sort directly from batches.
        // Avoids concat (reads batch columns in-place) and avoids take
        // (extracts sorted values via reverse XOR transform).
        if self.order_by.len() == 1 {
            if let BoundExpr::ColumnRef(cr) = &self.order_by[0].expr {
                let key_idx = resolve_column_index(cr.table_idx, cr.column_id, &self.input_schema)?;
                let num_cols = all_columns.len();
                let key_batches = &all_columns[key_idx];
                let has_nulls = key_batches.iter().any(|c| c.nulls.has_nulls());

                if !has_nulls && num_cols == 1 && !self.track_locators {
                    // Single column: produce sorted values directly from the
                    // batches, no indices needed. Skipped under locator
                    // tracking, a values-only sort has no permutation to
                    // apply to the locators
                    let type_id = key_batches[0].type_id;
                    let mut sorted =
                        match compute::radix_sort_batches_values(key_batches, self.order_by[0].asc)
                        {
                            Some(data) => Column::new(data, type_id),
                            None => {
                                let mut merged = concat_columns(key_batches);
                                compute::sort_column_inplace(
                                    &mut merged.data,
                                    self.order_by[0].asc,
                                );
                                merged
                            }
                        };
                    if let Some(limit) = self.limit {
                        let limit = limit as usize;
                        if total_rows > limit {
                            sorted.data.truncate(limit);
                            sorted.nulls = crate::column::NullBitmap::none(limit);
                        }
                    }
                    self.sorted_batch = Some(DataBatch::new(vec![sorted]));
                    return Ok(());
                }

                if !has_nulls {
                    // Multi-column: radix sort with value extraction for key,
                    // take() for non-key columns. The key pairs are built
                    // straight from the batches, so the key is never
                    // concatenated separately.
                    if let Some((mut indices, mut sorted_key)) =
                        compute::radix_sort_column_batches(key_batches, self.order_by[0].asc)
                    {
                        if let Some(limit) = self.limit {
                            indices.truncate(limit as usize);
                        }
                        let final_len = indices.len();
                        let key_type = all_columns[key_idx][0].type_id;
                        if final_len < total_rows {
                            sorted_key.truncate(final_len);
                        }
                        let mut sorted_key_opt = Some(sorted_key);
                        let idx_slice = &indices[..final_len];
                        let mut result_columns = Vec::with_capacity(num_cols);
                        for (col_idx, col_batches) in all_columns.iter().enumerate() {
                            if col_idx == key_idx {
                                result_columns
                                    .push(Column::new(sorted_key_opt.take().unwrap(), key_type));
                            } else {
                                let merged = concat_columns(col_batches);
                                result_columns.push(merged.take(idx_slice));
                            }
                        }
                        if self.track_locators {
                            self.sorted_locators = Some(
                                idx_slice
                                    .iter()
                                    .map(|&i| all_locators[i as usize])
                                    .collect(),
                            );
                        }
                        self.sorted_batch = Some(DataBatch::new(result_columns));
                        return Ok(());
                    }
                }
            }
        }

        // Fallback: concat all columns, sort_indices, take.
        let mut merged_columns: Vec<Column> = Vec::with_capacity(all_columns.len());
        for col_batches in &all_columns {
            merged_columns.push(concat_columns(col_batches));
        }
        let merged = DataBatch::new(merged_columns);

        // Evaluate sort key columns. For ColumnRef expressions, borrow
        // directly from the merged batch to avoid cloning the data.
        let mut ascending = Vec::with_capacity(self.order_by.len());
        let mut nulls_first = Vec::with_capacity(self.order_by.len());

        let mut key_sources: Vec<Option<usize>> = Vec::with_capacity(self.order_by.len());
        let mut owned_sort_columns: Vec<Column> = Vec::new();
        for ob in &self.order_by {
            ascending.push(ob.asc);
            nulls_first.push(ob.nulls_first);
            if let BoundExpr::ColumnRef(cr) = &ob.expr {
                let idx = resolve_column_index(cr.table_idx, cr.column_id, &self.input_schema)?;
                key_sources.push(Some(idx));
            } else {
                let col = evaluate(&ob.expr, &merged, &self.input_schema, &[])?;
                key_sources.push(None);
                owned_sort_columns.push(col);
            }
        }

        let mut owned_idx = 0;
        let sort_refs: Vec<&Column> = key_sources
            .iter()
            .map(|src| match src {
                Some(idx) => &merged.columns[*idx],
                None => {
                    let col = &owned_sort_columns[owned_idx];
                    owned_idx += 1;
                    col
                }
            })
            .collect();

        let mut indices = compute::sort_indices(&sort_refs, &ascending, &nulls_first, total_rows);

        // Apply top-N limit.
        if let Some(limit) = self.limit {
            let limit = limit as usize;
            if indices.len() > limit {
                indices.truncate(limit);
            }
        }

        // Single full take() pass: reorder all data once.
        if self.track_locators {
            self.sorted_locators =
                Some(indices.iter().map(|&i| all_locators[i as usize]).collect());
        }
        self.sorted_batch = Some(merged.take(&indices));
        Ok(())
    }

    /// Sorts what is in memory and writes it out as one run.
    ///
    /// A run is sorted before it is written, which is what makes the merge a
    /// merge rather than a second sort. Under a limit each run is truncated to
    /// it: the global top N is a subset of the union of the per-run top N, so
    /// keeping more than N per run writes bytes the merge will discard.
    fn flush_run(
        &mut self,
        all_columns: Vec<Vec<Column>>,
        all_locators: Vec<RowLocator>,
        total_rows: usize,
    ) -> Result<()> {
        if total_rows == 0 {
            return Ok(());
        }
        let directory = self
            .spill
            .as_ref()
            .ok_or_else(|| ZyronError::ExecutionError("sort has nowhere to spill".into()))?;

        let (batch, locators) = self.sort_in_memory(all_columns, all_locators, total_rows)?;
        let mut writer = directory.create()?;
        writer.write_batch_with_locators(&batch, locators.as_deref())?;
        self.runs.push(writer.finish()?);
        crate::spill::SpillStats::global()
            .runs_merged
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        if self.runs.len() == 1 {
            crate::spill::SpillStats::global()
                .sorts_spilled
                .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        }
        Ok(())
    }

    /// Concatenates, sorts, and reorders one run's worth of input.
    ///
    /// The general path rather than the radix fast paths: a run is an
    /// intermediate nobody sees, and the fast paths exist to produce a final
    /// answer without an index vector, which a run cannot use because the
    /// merge needs the rows in their sorted order rather than the values.
    fn sort_in_memory(
        &self,
        all_columns: Vec<Vec<Column>>,
        all_locators: Vec<RowLocator>,
        total_rows: usize,
    ) -> Result<(DataBatch, Option<Vec<RowLocator>>)> {
        let mut merged_columns: Vec<Column> = Vec::with_capacity(all_columns.len());
        for col_batches in &all_columns {
            merged_columns.push(concat_columns(col_batches));
        }
        let merged = DataBatch::new(merged_columns);

        let (ascending, nulls_first, key_columns) = self.key_columns(&merged)?;
        let refs: Vec<&Column> = key_columns.iter().collect();
        let mut indices = compute::sort_indices(&refs, &ascending, &nulls_first, total_rows);
        if let Some(limit) = self.limit {
            indices.truncate(limit as usize);
        }
        let locators = if self.track_locators {
            Some(
                indices
                    .iter()
                    .map(|&i| all_locators[i as usize])
                    .collect::<Vec<_>>(),
            )
        } else {
            None
        };
        Ok((merged.take(&indices), locators))
    }

    /// The direction and null placement of each key, in key order
    fn directions(&self) -> (Vec<bool>, Vec<bool>) {
        (
            self.order_by.iter().map(|ob| ob.asc).collect(),
            self.order_by.iter().map(|ob| ob.nulls_first).collect(),
        )
    }

    /// The sort key of every row of a batch, borrowed wherever the key is
    /// a column of the batch itself
    fn batch_keys<'a>(&self, batch: &'a DataBatch) -> Result<Vec<Cow<'a, Column>>> {
        self.order_by
            .iter()
            .map(|ob| evaluate_borrowed(&ob.expr, batch, &self.input_schema, &[]))
            .collect()
    }

    /// Cuts the buffer back to the rows the limit keeps.
    ///
    /// One selection pass over the buffer finds them, and the row at the
    /// limit comes back as the bound every later row is held against. The
    /// kept rows are in no particular order, the final sort orders them
    fn cut_to_limit(
        &self,
        all_columns: Vec<Vec<Column>>,
        all_locators: Vec<RowLocator>,
        total_rows: usize,
        keep: usize,
        ascending: &[bool],
        nulls_first: &[bool],
    ) -> Result<CutBuffer> {
        let merged = DataBatch::new(all_columns.iter().map(|b| concat_columns(b)).collect());
        let keys = self.batch_keys(&merged)?;
        let key_refs: Vec<&Column> = keys.iter().map(|c| c.as_ref()).collect();
        let indices =
            compute::select_first_indices(&key_refs, ascending, nulls_first, total_rows, keep);
        let kept = &indices[..keep];
        let at_limit = [indices[keep - 1]];
        let bound = key_refs.iter().map(|c| c.take(&at_limit)).collect();
        let locators = self
            .track_locators
            .then(|| kept.iter().map(|&i| all_locators[i as usize]).collect());
        Ok(CutBuffer {
            batch: merged.take(kept),
            locators,
            bound,
        })
    }

    /// Evaluates the sort key against a batch, owning every key column.
    ///
    /// Owned rather than borrowed because the merge holds the keys alongside
    /// the batch they came from and a borrow would tie the two together for
    /// the run's lifetime.
    fn key_columns(&self, batch: &DataBatch) -> Result<(Vec<bool>, Vec<bool>, Vec<Column>)> {
        let mut ascending = Vec::with_capacity(self.order_by.len());
        let mut nulls_first = Vec::with_capacity(self.order_by.len());
        let mut columns = Vec::with_capacity(self.order_by.len());
        for ob in &self.order_by {
            ascending.push(ob.asc);
            nulls_first.push(ob.nulls_first);
            let column = match &ob.expr {
                BoundExpr::ColumnRef(cr) => {
                    let idx = resolve_column_index(cr.table_idx, cr.column_id, &self.input_schema)?;
                    batch.columns[idx].clone()
                }
                other => evaluate(other, batch, &self.input_schema, &[])?,
            };
            columns.push(column);
        }
        Ok((ascending, nulls_first, columns))
    }

    /// Opens every run and primes the merge with its first batch.
    fn build_merge(&mut self) -> Result<MergeState> {
        let mut runs = Vec::with_capacity(self.runs.len());
        for reader in std::mem::take(&mut self.runs) {
            let mut run = MergeRun {
                reader,
                batch: None,
                keys: Vec::new(),
                locators: None,
                cursor: 0,
            };
            self.advance_run(&mut run)?;
            if run.batch.is_some() {
                runs.push(run);
            }
        }
        let mut ascending = Vec::with_capacity(self.order_by.len());
        let mut nulls_first = Vec::with_capacity(self.order_by.len());
        for ob in &self.order_by {
            ascending.push(ob.asc);
            nulls_first.push(ob.nulls_first);
        }
        Ok(MergeState {
            runs,
            ascending,
            nulls_first,
        })
    }

    /// Loads a run's next batch and evaluates its key columns.
    fn advance_run(&self, run: &mut MergeRun) -> Result<()> {
        loop {
            match run.reader.read_batch_with_locators()? {
                Some((batch, locators)) => {
                    if batch.num_rows == 0 {
                        continue;
                    }
                    let (_, _, keys) = self.key_columns(&batch)?;
                    run.keys = keys;
                    run.locators = locators;
                    run.batch = Some(batch);
                    run.cursor = 0;
                    return Ok(());
                }
                None => {
                    run.batch = None;
                    run.keys.clear();
                    run.locators = None;
                    run.cursor = 0;
                    return Ok(());
                }
            }
        }
    }

    /// Produces the next merged batch, or None when every run is drained.
    ///
    /// Emits in output-batch sized pieces rather than one result, because the
    /// whole point of having spilled is that the whole result does not fit.
    ///
    /// Each row is copied into the output the moment it is picked. Recording
    /// the row and gathering afterwards would be one pass cheaper and wrong:
    /// picking the last row of a run loads that run's next batch, and the
    /// recorded position then points into a batch that has been replaced.
    fn next_merged(&mut self) -> Result<Option<ExecutionBatch>> {
        const MERGE_BATCH_ROWS: usize = 8_192;

        let Some(mut state) = self.merge.take() else {
            return Ok(None);
        };
        if let Some(limit) = self.limit {
            if self.emitted >= limit {
                return Ok(None);
            }
        }

        // The schema comes from whichever run still has rows. Every run was
        // written by this operator, so they all carry the same one
        let Some(template) = state.runs.iter().find_map(|r| r.batch.as_ref()) else {
            return Ok(None);
        };
        let width = template.num_columns();
        let mut builders: Vec<ColumnData> = Vec::with_capacity(width);
        let mut types: Vec<(zyron_common::TypeId, Option<u8>)> = Vec::with_capacity(width);
        for column in &template.columns {
            builders.push(ColumnData::with_capacity(column.type_id, MERGE_BATCH_ROWS));
            types.push((column.type_id, column.fractional_digits));
        }
        let mut null_flags: Vec<Vec<bool>> = vec![Vec::with_capacity(MERGE_BATCH_ROWS); width];
        let mut out_locators: Vec<RowLocator> = Vec::new();
        let mut rows = 0usize;

        while rows < MERGE_BATCH_ROWS {
            if let Some(limit) = self.limit {
                if self.emitted + rows as u64 >= limit {
                    break;
                }
            }
            let Some(winner) = state.smallest() else {
                break;
            };

            {
                let run = &state.runs[winner];
                let Some(batch) = run.batch.as_ref() else {
                    break;
                };
                for (col, builder) in builders.iter_mut().enumerate() {
                    let source = &batch.columns[col];
                    builder.push_from(&source.data, run.cursor);
                    null_flags[col].push(source.is_null(run.cursor));
                }
                if let Some(locs) = run.locators.as_ref() {
                    out_locators.push(locs[run.cursor]);
                }
            }
            rows += 1;

            let run = &mut state.runs[winner];
            run.cursor += 1;
            let drained = run
                .batch
                .as_ref()
                .map(|b| run.cursor >= b.num_rows)
                .unwrap_or(true);
            if drained {
                self.advance_run(run)?;
            }
        }

        if rows == 0 {
            self.merge = None;
            return Ok(None);
        }

        let mut columns = Vec::with_capacity(width);
        for (col, data) in builders.into_iter().enumerate() {
            let mut nulls = crate::column::NullBitmap::none(rows);
            for (row, is_null) in null_flags[col].iter().enumerate() {
                if *is_null {
                    nulls.set_null(row);
                }
            }
            let (type_id, fractional_digits) = types[col];
            columns.push(Column {
                data,
                nulls,
                type_id,
                fractional_digits,
            });
        }

        self.emitted += rows as u64;
        self.merge = Some(state);
        Ok(Some(ExecutionBatch {
            batch: DataBatch::new(columns),
            locators: if self.track_locators {
                Some(out_locators)
            } else {
                None
            },
        }))
    }
}

impl Operator for SortOperator {
    fn next(&mut self) -> OperatorResult<'_> {
        Box::pin(async move {
            if self.finished {
                return Ok(None);
            }

            if self.sorted_batch.is_none() && self.merge.is_none() {
                // The sort does all its work in one pass, so one timer around
                // that pass is the whole cost, and the row count is what came
                // out of it
                let mut timer = crate::calibrate::BatchTimer::start(
                    zyron_pressure::capability::OperatorKind::Sort,
                );
                self.materialize().await?;
                timer.rows(self.sorted_batch.as_ref().map(|b| b.num_rows).unwrap_or(0) as u64);
            }

            // A sort that spilled produces its answer a batch at a time. The
            // merged result is larger than memory by construction, so handing
            // it out whole would spend exactly the memory the spill saved
            if self.merge.is_some() {
                // The merge is post-consumption compute over spill files and
                // never touches a tokio leaf resource, so it polls the cancel
                // flag itself and periodically hands the worker back
                if let Some(ctx) = &self.poll_ctx {
                    ctx.check_cancelled()?;
                }
                self.merge_batches = self.merge_batches.wrapping_add(1);
                if self.merge_batches % 16 == 0 {
                    tokio::task::yield_now().await;
                }
                let mut timer = crate::calibrate::BatchTimer::start(
                    zyron_pressure::capability::OperatorKind::Sort,
                );
                return match self.next_merged()? {
                    Some(eb) => {
                        timer.rows(eb.batch.num_rows as u64);
                        Ok(Some(eb))
                    }
                    None => {
                        self.finished = true;
                        Ok(None)
                    }
                };
            }

            // The sort already materialized everything, so the whole result
            // moves out as one batch instead of being re-copied in chunks.
            self.finished = true;
            match self.sorted_batch.take() {
                Some(batch) => {
                    let locators = self.sorted_locators.take();
                    Ok(Some(ExecutionBatch { batch, locators }))
                }
                None => Ok(None),
            }
        })
    }
}

/// One sorted run being merged.
struct MergeRun {
    reader: crate::spill::SpillReader,
    batch: Option<DataBatch>,
    /// Key columns of the current batch, evaluated once when it was loaded
    keys: Vec<Column>,
    locators: Option<Vec<RowLocator>>,
    cursor: usize,
}

/// The k-way merge over sorted runs.
struct MergeState {
    runs: Vec<MergeRun>,
    ascending: Vec<bool>,
    nulls_first: Vec<bool>,
}

impl MergeState {
    /// The run whose head row sorts first.
    ///
    /// A linear scan over the run heads rather than a heap. Runs are as many
    /// as the input divided by the memory budget, which is tens at the sizes
    /// that spill at all, and at that width a scan over contiguous heads beats
    /// a heap's pointer chasing and its sift on every pop.
    fn smallest(&self) -> Option<usize> {
        let mut best: Option<usize> = None;
        for (i, run) in self.runs.iter().enumerate() {
            if run.batch.is_none() {
                continue;
            }
            let Some(current) = best else {
                best = Some(i);
                continue;
            };
            let a: Vec<&Column> = self.runs[i].keys.iter().collect();
            let b: Vec<&Column> = self.runs[current].keys.iter().collect();
            let ord = compute::compare_rows_across(
                &a,
                self.runs[i].cursor,
                &b,
                self.runs[current].cursor,
                &self.ascending,
                &self.nulls_first,
            );
            if ord == std::cmp::Ordering::Less {
                best = Some(i);
            }
        }
        best
    }
}

/// Concatenates multiple columns of the same type into one.
/// Uses typed bulk extend_from to avoid per-row ScalarValue allocation.
fn concat_columns(columns: &[Column]) -> Column {
    if columns.is_empty() {
        return Column::null_column(zyron_common::TypeId::Null, 0);
    }
    if columns.len() == 1 {
        return columns[0].clone();
    }

    let type_id = columns[0].type_id;
    let total_len: usize = columns.iter().map(|c| c.len()).sum();

    let mut data = crate::column::ColumnData::with_capacity(type_id, total_len);
    let mut nulls = crate::column::NullBitmap::empty();

    for col in columns {
        data.extend_from(&col.data);
        nulls.extend_from(&col.nulls);
    }

    Column::with_nulls(data, nulls, type_id)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::column::ColumnData;
    use crate::context::QueryMemoryBudget;
    use crate::operator::ExecutionBatch;
    use zyron_common::TypeId;

    struct FeedOp {
        batches: Vec<DataBatch>,
    }

    impl Operator for FeedOp {
        fn next(&mut self) -> crate::operator::OperatorResult<'_> {
            Box::pin(async move { Ok(self.batches.pop().map(ExecutionBatch::new)) })
        }
    }

    fn int_batch(values: Vec<i64>) -> DataBatch {
        DataBatch::new(vec![Column::new(ColumnData::Int64(values), TypeId::Int64)])
    }

    fn int_schema() -> Vec<LogicalColumn> {
        vec![LogicalColumn {
            name: "v".into(),
            type_id: TypeId::Int64,
            nullable: false,
            fractional_digits: None,
            table_idx: Some(0),
            column_id: zyron_catalog::ColumnId(0),
        }]
    }

    fn order() -> Vec<BoundOrderBy> {
        vec![BoundOrderBy {
            expr: BoundExpr::ColumnRef(zyron_planner::binder::ColumnRef {
                table_idx: 0,
                column_id: zyron_catalog::ColumnId(0),
                type_id: TypeId::Int64,
                nullable: false,
                fractional_digits: None,
            }),
            asc: true,
            nulls_first: false,
        }]
    }

    // A sort over more bytes than the budget allows fails the query with
    // the budget error instead of materializing everything
    #[tokio::test]
    async fn sort_respects_query_memory_budget() {
        let feed = FeedOp {
            batches: vec![
                int_batch((0..1000).collect()),
                int_batch((0..1000).collect()),
            ],
        };
        let mut op = SortOperator::new(Box::new(feed), order(), int_schema(), None);
        op.set_memory_budget(Some(QueryMemoryBudget::new(64)));
        let err = match op.next().await {
            Err(e) => e,
            Ok(_) => panic!("64 byte budget must refuse"),
        };
        assert!(
            err.to_string().contains("memory budget"),
            "unexpected error: {err}"
        );
    }

    // The same input sorts fine with a budget that fits, and unbudgeted
    #[tokio::test]
    async fn sort_within_budget_succeeds() {
        let feed = FeedOp {
            batches: vec![int_batch(vec![3, 1, 2])],
        };
        let mut op = SortOperator::new(Box::new(feed), order(), int_schema(), None);
        op.set_memory_budget(Some(QueryMemoryBudget::new(1024 * 1024)));
        let eb = op.next().await.unwrap().expect("sorted batch");
        match &eb.batch.columns[0].data {
            ColumnData::Int64(v) => assert_eq!(v, &vec![1, 2, 3]),
            other => panic!("unexpected column {other:?}"),
        }
    }

    /// Twenty batches of a thousand, a permutation of the row numbers by a
    /// stride, so every batch carries some of the first rows and a bounded
    /// sort has to cut its buffer more than once
    fn strided_batches() -> (Vec<DataBatch>, Vec<i64>) {
        let total = 20_000i64;
        let all: Vec<i64> = (0..total).map(|i| (i * 7_919) % total).collect();
        let batches = all.chunks(1_000).map(|c| int_batch(c.to_vec())).collect();
        (batches, all)
    }

    fn desc_order() -> Vec<BoundOrderBy> {
        let mut order = order();
        order[0].asc = false;
        order
    }

    async fn drain(op: &mut SortOperator) -> Vec<Option<i64>> {
        let mut out = Vec::new();
        while let Some(eb) = op.next().await.unwrap() {
            let column = &eb.batch.columns[0];
            let ColumnData::Int64(values) = &column.data else {
                panic!("unexpected column {:?}", column.data);
            };
            for (row, value) in values.iter().enumerate() {
                out.push((!column.is_null(row)).then_some(*value));
            }
        }
        out
    }

    fn present(values: &[i64]) -> Vec<Option<i64>> {
        values.iter().map(|&v| Some(v)).collect()
    }

    #[tokio::test]
    async fn a_limit_keeps_the_first_rows_across_many_cuts() {
        let (batches, mut all) = strided_batches();
        let mut op =
            SortOperator::new(Box::new(FeedOp { batches }), order(), int_schema(), Some(7));
        let got = drain(&mut op).await;
        all.sort_unstable();
        assert_eq!(got, present(&all[..7]));
    }

    #[tokio::test]
    async fn a_descending_limit_keeps_the_last_rows_first() {
        let (batches, mut all) = strided_batches();
        let mut op = SortOperator::new(
            Box::new(FeedOp { batches }),
            desc_order(),
            int_schema(),
            Some(5),
        );
        let got = drain(&mut op).await;
        all.sort_unstable_by(|a, b| b.cmp(a));
        assert_eq!(got, present(&all[..5]));
    }

    #[tokio::test]
    async fn a_limit_past_the_rows_sorts_them_all() {
        let (batches, mut all) = strided_batches();
        let mut op = SortOperator::new(
            Box::new(FeedOp { batches }),
            order(),
            int_schema(),
            Some(50_000),
        );
        let got = drain(&mut op).await;
        all.sort_unstable();
        assert_eq!(got, present(&all));
    }

    // Twenty thousand rows are more bytes than this budget holds, and a
    // sort under a limit never holds more than about a batch beyond the
    // limit, so the same input sorts within it
    #[tokio::test]
    async fn a_limit_holds_the_buffer_within_a_budget_the_input_would_not_fit() {
        let (batches, mut all) = strided_batches();
        let budget_bytes = 32 * 1024;

        let mut unbounded = SortOperator::new(
            Box::new(FeedOp {
                batches: batches.clone(),
            }),
            order(),
            int_schema(),
            None,
        );
        unbounded.set_memory_budget(Some(QueryMemoryBudget::new(budget_bytes)));
        let err = match unbounded.next().await {
            Err(e) => e,
            Ok(_) => panic!("the whole input must not fit the budget"),
        };
        assert!(
            err.to_string().contains("memory budget"),
            "unexpected error: {err}"
        );

        let mut bounded =
            SortOperator::new(Box::new(FeedOp { batches }), order(), int_schema(), Some(7));
        let budget = QueryMemoryBudget::new(budget_bytes);
        bounded.set_memory_budget(Some(budget.clone()));
        let got = drain(&mut bounded).await;
        all.sort_unstable();
        assert_eq!(got, present(&all[..7]));
        // What stays reserved is the buffer the sort kept, not the input
        assert!(
            budget.used() < budget_bytes,
            "{} bytes still reserved",
            budget.used()
        );
    }

    fn int_batch_with_nulls(values: Vec<i64>, null_at: &[usize]) -> DataBatch {
        let mut nulls = crate::column::NullBitmap::none(values.len());
        for &row in null_at {
            nulls.set_null(row);
        }
        DataBatch::new(vec![Column::with_nulls(
            ColumnData::Int64(values),
            nulls,
            TypeId::Int64,
        )])
    }

    // Nulls sort last here, so a limit inside the values sees none of them
    // and a limit past the values ends in them, whichever batch they came in
    #[tokio::test]
    async fn null_keys_stay_out_of_a_limit_until_the_values_run_short() {
        let batches: Vec<DataBatch> = (0..5)
            .map(|b| {
                let values: Vec<i64> = (0..1_000)
                    .map(|i| ((b * 1_000 + i) * 7_919) % 5_000)
                    .collect();
                int_batch_with_nulls(values, &[0, 500])
            })
            .collect();
        let hidden: std::collections::HashSet<i64> = (0..5)
            .flat_map(|b| {
                [
                    (b * 1_000 * 7_919) % 5_000,
                    ((b * 1_000 + 500) * 7_919) % 5_000,
                ]
            })
            .collect();
        let mut values: Vec<i64> = (0..5_000).filter(|v| !hidden.contains(v)).collect();
        values.sort_unstable();

        let mut inside = SortOperator::new(
            Box::new(FeedOp {
                batches: batches.clone(),
            }),
            order(),
            int_schema(),
            Some(7),
        );
        assert_eq!(drain(&mut inside).await, present(&values[..7]));

        let mut past = SortOperator::new(
            Box::new(FeedOp { batches }),
            order(),
            int_schema(),
            Some(4_995),
        );
        let mut expected = present(&values);
        expected.extend(std::iter::repeat_n(None, 5));
        assert_eq!(drain(&mut past).await, expected);
    }

    // A key every row shares: once the sort holds its limit's worth, every
    // later row ties with the bound and is dropped, which must still leave
    // a full limit of rows
    #[tokio::test]
    async fn ties_at_the_bound_still_fill_the_limit() {
        let batches = (0..5).map(|_| int_batch(vec![5; 1_000])).collect();
        let mut op =
            SortOperator::new(Box::new(FeedOp { batches }), order(), int_schema(), Some(3));
        assert_eq!(drain(&mut op).await, present(&[5, 5, 5]));
    }

    /// Feeds batches whose row locators are the row values, so the kept
    /// rows are known by their locators alone
    struct LocatorFeed {
        batches: Vec<DataBatch>,
    }

    impl Operator for LocatorFeed {
        fn next(&mut self) -> crate::operator::OperatorResult<'_> {
            Box::pin(async move {
                Ok(self.batches.pop().map(|batch| {
                    let ColumnData::Int64(values) = &batch.columns[0].data else {
                        panic!("integer batches only");
                    };
                    let locators = values
                        .iter()
                        .map(|&v| RowLocator::Lake {
                            file_id: 0,
                            ordinal: v as u64,
                        })
                        .collect();
                    ExecutionBatch::with_locators(batch, locators)
                }))
            })
        }
    }

    #[tokio::test]
    async fn a_limit_carries_the_kept_rows_locators() {
        let (batches, mut all) = strided_batches();
        let mut op = SortOperator::new(
            Box::new(LocatorFeed { batches }),
            order(),
            int_schema(),
            Some(7),
        )
        .with_locator_tracking();
        let eb = op.next().await.unwrap().expect("the kept rows");
        all.sort_unstable();
        let expected: Vec<RowLocator> = all[..7]
            .iter()
            .map(|&v| RowLocator::Lake {
                file_id: 0,
                ordinal: v as u64,
            })
            .collect();
        assert_eq!(eb.locators.expect("locators"), expected);
        match &eb.batch.columns[0].data {
            ColumnData::Int64(v) => assert_eq!(v, &all[..7]),
            other => panic!("unexpected column {other:?}"),
        }
    }

    /// Counts how often the sort asks for input
    struct CountingFeed {
        pulls: std::sync::Arc<std::sync::atomic::AtomicUsize>,
    }

    impl Operator for CountingFeed {
        fn next(&mut self) -> crate::operator::OperatorResult<'_> {
            Box::pin(async move {
                self.pulls
                    .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                Ok(None)
            })
        }
    }

    #[tokio::test]
    async fn a_limit_of_zero_reads_nothing() {
        let pulls = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let mut op = SortOperator::new(
            Box::new(CountingFeed {
                pulls: pulls.clone(),
            }),
            order(),
            int_schema(),
            Some(0),
        );
        assert!(op.next().await.unwrap().is_none());
        assert_eq!(pulls.load(std::sync::atomic::Ordering::Relaxed), 0);
    }
}
