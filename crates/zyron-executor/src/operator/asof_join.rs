//! ASOF join: each left row takes the nearest right row in one direction
//! along an ordered column, within the equality group the ON clause names.
//!
//! Both inputs arrive ordered by (equality keys, match column), so the join
//! is one pass over each. Inside an equality group the merge advances the
//! right side while its match column is still on the near side of the left
//! row's, holding the last one it passed. That held row is the answer for
//! every left row until the right side passes it, so the state is one row
//! rather than a table of rows.
//!
//! Nothing is materialized: the operator holds one batch per side, the held
//! right row, and the output batch it is filling. A left row's match is
//! always at or after the previous left row's, so neither cursor ever
//! rewinds and each input is read exactly once.
//!
//! The merge reads keys and match values off the batches as ordered values
//! rather than as scalars. An integer of any width orders as one wide
//! integer, so the common case, an integer or timestamp key and a timestamp
//! match column, compares two machine words per row and builds nothing. The
//! right side's answer is copied straight from the batch it sits in into
//! the output builders, a cell at a time with no scalar in between, and a
//! candidate the merge passes over without ever answering is never copied

use std::cmp::Ordering;
use std::sync::Arc;

use zyron_common::{Result, TypeId, ZyronError};
use zyron_parser::ast::JoinType;
use zyron_planner::binder::BoundExpr;
use zyron_planner::logical::AsofTolerance;
use zyron_planner::physical::AsofJoinSpec;

use crate::batch::{ColumnBuilder, DataBatch, create_builders};
use crate::column::{Column, ColumnData, ScalarValue};
use crate::context::ExecutionContext;
use crate::expr::evaluate;
use crate::operator::{ExecutionBatch, Operator, OperatorResult};

/// Merges two ordered inputs into ASOF matches.
pub struct AsofJoinOperator {
    left: Cursor,
    right: Cursor,
    spec: Box<AsofJoinSpec>,
    ctx: Arc<ExecutionContext>,
    batch_size: usize,
    /// The tolerance's value, read once because it is a constant of the
    /// statement rather than of a row
    tolerance: Option<Ordered>,
    tolerance_read: bool,
    /// The right row the merge is holding as the current candidate, and the
    /// equality group it belongs to. Cleared when the left side moves into a
    /// group the held row is not in
    held: Option<HeldRow>,
    /// The left rows one output batch gathers and the right rows matched off
    /// the batch in hand. Held across batches so neither is allocated per
    /// batch
    left_rows: Vec<u32>,
    right_pending: Vec<u32>,
    finished: bool,
}

/// A key or match value as the merge orders it.
///
/// An integer of any width, a timestamp and a date all order as one wide
/// integer read off the column, so the row costs no scalar. Anything else
/// orders through its scalar
#[derive(Debug, Clone, PartialEq)]
enum Ordered {
    Null,
    Int(i128),
    Other(ScalarValue),
}

impl Ordered {
    /// The value at a row of a column, as the merge orders it
    #[inline]
    fn at(column: &Column, row: usize) -> Self {
        if column.is_null(row) {
            return Ordered::Null;
        }
        match column.data.i128_at(row) {
            Some(v) => Ordered::Int(v),
            None => Ordered::Other(column.data.get_scalar(row)),
        }
    }

    fn of(value: ScalarValue) -> Self {
        match value {
            ScalarValue::Null => Ordered::Null,
            other => match other.to_i128() {
                Some(v) => Ordered::Int(v),
                None => Ordered::Other(other),
            },
        }
    }
}

/// The right row that is the nearest candidate so far.
///
/// Its columns are not read out of the batch when it is taken. Walking a
/// group takes every row at or before the left row in turn, and only the last
/// of them answers, so materializing each one would copy a full row for every
/// right row the merge passes rather than for every row it emits. The columns
/// are read at the point they are needed, and once more only if the cursor is
/// about to leave the batch they live in.
struct HeldRow {
    key: Vec<Ordered>,
    match_value: Ordered,
    /// The row this candidate sits at in the right cursor's current batch,
    /// meaningful only while `values` is empty
    row: usize,
    /// The candidate's columns, filled when the batch holding it is about to
    /// be replaced. Empty while the row is still readable off the batch
    values: Vec<ScalarValue>,
    /// True once `values` holds the row, which is what tells the emitter to
    /// read there rather than off the batch
    materialized: bool,
}

/// One side's position: the batch it is reading, the row inside it, and the
/// key and match columns evaluated once for that batch.
struct Cursor {
    input: Box<dyn Operator>,
    schema: Vec<zyron_planner::logical::LogicalColumn>,
    keys: Vec<BoundExpr>,
    match_column: BoundExpr,
    batch: Option<LoadedBatch>,
    row: usize,
    exhausted: bool,
}

/// One batch with its key and match columns already evaluated.
///
/// The columns are held as columns of their own rather than as positions
/// in the batch. A key read through a position costs a branch and a second
/// indirection on every read, and the merge reads a key several times per
/// row while the copy that avoids that is paid once per batch
struct LoadedBatch {
    batch: DataBatch,
    key_columns: Vec<Column>,
    match_column: Column,
}

impl Cursor {
    fn new(
        input: Box<dyn Operator>,
        schema: Vec<zyron_planner::logical::LogicalColumn>,
        keys: Vec<BoundExpr>,
        match_column: BoundExpr,
    ) -> Self {
        Self {
            input,
            schema,
            keys,
            match_column,
            batch: None,
            row: 0,
            exhausted: false,
        }
    }

    /// True when the cursor is on a row of the batch it holds, which needs
    /// no input and is the answer the merge's inner loop asks for
    #[inline]
    fn has_row(&self) -> bool {
        match &self.batch {
            Some(loaded) => self.row < loaded.batch.num_rows,
            None => false,
        }
    }

    /// Makes sure a row is available, pulling batches until one has rows or
    /// the input is done. Returns false when the side is exhausted.
    async fn ensure_row(&mut self) -> Result<bool> {
        loop {
            if self.has_row() {
                return Ok(true);
            }
            if self.exhausted {
                return Ok(false);
            }
            match self.input.next().await? {
                None => {
                    self.exhausted = true;
                    self.batch = None;
                    return Ok(false);
                }
                Some(eb) => {
                    let batch = eb.batch;
                    if batch.num_rows == 0 {
                        continue;
                    }
                    // Evaluated once per batch, so the merge reads values
                    // rather than re-evaluating an expression per row
                    let key_columns: Vec<Column> = self
                        .keys
                        .iter()
                        .map(|k| evaluate(k, &batch, &self.schema, &[]))
                        .collect::<Result<Vec<_>>>()?;
                    let match_column = evaluate(&self.match_column, &batch, &self.schema, &[])?;
                    self.batch = Some(LoadedBatch {
                        batch,
                        key_columns,
                        match_column,
                    });
                    self.row = 0;
                }
            }
        }
    }

    /// The key of the row the cursor is on, written into a buffer the caller
    /// reuses.
    ///
    /// A fresh Vec per row is one allocation per row of the input, which for
    /// ten million left rows is ten million of them and the largest single
    /// cost in the merge.
    #[inline]
    fn key_into(&self, out: &mut Vec<Ordered>) {
        out.clear();
        if let Some(loaded) = &self.batch {
            out.extend(loaded.key_columns.iter().map(|c| Ordered::at(c, self.row)));
        }
    }

    /// The key of the row the cursor is on, ordered against a key already in
    /// hand.
    ///
    /// Compares column by column off the batch, so a key that differs in its
    /// first column costs one read rather than a materialized tuple.
    #[inline]
    fn compare_key_to(&self, other: &[Ordered]) -> Ordering {
        let Some(loaded) = &self.batch else {
            return Ordering::Greater;
        };
        for (column, want) in loaded.key_columns.iter().zip(other) {
            match compare_key_parts(&Ordered::at(column, self.row), want) {
                Ordering::Equal => continue,
                other => return other,
            }
        }
        Ordering::Equal
    }

    /// The key of the row this cursor is on, ordered against the key of the
    /// row another cursor is on.
    ///
    /// Both sides are read off their batches to the first column that
    /// differs, so ordering two rows costs no tuple on either side
    #[inline]
    fn compare_key_to_row(&self, other: &Cursor) -> Ordering {
        let (Some(mine), Some(theirs)) = (&self.batch, &other.batch) else {
            return Ordering::Greater;
        };
        for (column, want) in mine.key_columns.iter().zip(&theirs.key_columns) {
            match compare_key_parts(
                &Ordered::at(column, self.row),
                &Ordered::at(want, other.row),
            ) {
                Ordering::Equal => continue,
                other => return other,
            }
        }
        Ordering::Equal
    }

    /// The match value of the row the cursor is on.
    #[inline]
    fn match_value(&self) -> Ordered {
        match &self.batch {
            Some(loaded) => Ordered::at(&loaded.match_column, self.row),
            None => Ordered::Null,
        }
    }

    #[inline]
    fn advance(&mut self) {
        self.row += 1;
    }

    /// True when the cursor has read past the batch it holds, so the next
    /// `ensure_row` replaces it.
    fn at_batch_end(&self) -> bool {
        !self.has_row()
    }

    /// Every column of one row of the batch the cursor holds, written into a
    /// buffer the caller reuses.
    fn row_into(&self, at: usize, out: &mut Vec<ScalarValue>) {
        out.clear();
        if let Some(loaded) = &self.batch
            && at < loaded.batch.num_rows
        {
            out.extend(loaded.batch.columns.iter().map(|c| c.get_scalar(at)));
        }
    }
}

/// Why the merge stopped reading left rows
enum Halted {
    /// The left batch is read out, or the output batch is full
    Done,
    /// The right side's batch is spent and the next one has to be pulled
    /// before the left row under the cursor can be answered
    NeedsRight,
}

/// What the merge found for one left row
enum Candidate {
    /// The held row answers
    Matched,
    /// No right row answers this left row
    Unmatched,
    /// The right side's batch is spent and the next has to be pulled before
    /// the answer is known
    NeedsInput,
}

impl AsofJoinOperator {
    pub fn new(
        left: Box<dyn Operator>,
        right: Box<dyn Operator>,
        spec: Box<AsofJoinSpec>,
        ctx: Arc<ExecutionContext>,
    ) -> Self {
        let batch_size = ctx.batch_size.max(1);
        let left_keys: Vec<BoundExpr> = spec.equality_keys.iter().map(|(l, _)| l.clone()).collect();
        let right_keys: Vec<BoundExpr> =
            spec.equality_keys.iter().map(|(_, r)| r.clone()).collect();
        let left_schema = spec.left_schema.clone();
        let right_schema = spec.right_schema.clone();
        let match_left = spec.match_left.clone();
        let match_right = spec.match_right.clone();
        Self {
            left: Cursor::new(left, left_schema, left_keys, match_left),
            right: Cursor::new(right, right_schema, right_keys, match_right),
            spec,
            ctx,
            batch_size,
            tolerance: None,
            tolerance_read: false,
            held: None,
            left_rows: Vec::with_capacity(batch_size),
            right_pending: Vec::with_capacity(batch_size),
            finished: false,
        }
    }

    /// The tolerance's value, evaluated once against a one-row probe because
    /// it is a constant of the statement rather than of a row.
    fn read_tolerance(&mut self) -> Result<()> {
        if self.tolerance_read {
            return Ok(());
        }
        self.tolerance_read = true;
        let Some(tolerance) = &self.spec.tolerance else {
            return Ok(());
        };
        let probe = DataBatch::new(vec![Column::new(ColumnData::Int64(vec![0]), TypeId::Int64)]);
        let column = evaluate(&tolerance.bound, &probe, &[], &[])?;
        if column.len() == 0 {
            return Err(ZyronError::ExecutionError(
                "a MATCH_CONDITION tolerance has to be a constant of the statement".to_string(),
            ));
        }
        self.tolerance = Some(Ordered::of(column.get_scalar(0)));
        Ok(())
    }

    /// Advances the right side over the batch it holds so its held candidate
    /// is the nearest row for the left row the merge is on, and says whether
    /// that candidate matches, or that the batch is spent before the answer
    /// is known.
    ///
    /// Runs with no input of its own, so the merge's inner loop over a batch
    /// of left rows is a plain loop. The matched row's values stay in
    /// `self.held` rather than being handed back, so a left row that matches
    /// costs no copy of them: the caller pushes straight out of the held row
    #[inline]
    fn candidate_in_batch(&mut self, left_match: &Ordered) -> Candidate {
        // A held row from an earlier equality group is not a candidate for
        // this one. The left key is read off its batch for the comparison
        // rather than gathered into a tuple the row is done with
        if let Some(held) = &self.held
            && self.left.compare_key_to(&held.key) != Ordering::Equal
        {
            self.held = None;
        }

        let backward = self.spec.direction.is_backward();
        let allows_equal = self.spec.direction.allows_equal();

        loop {
            if !self.right.has_row() {
                if self.right.exhausted {
                    break;
                }
                return Candidate::NeedsInput;
            }
            // A held row's key is this left row's key, checked above, so
            // while one is held the right side is ordered against the key
            // already in hand and only the right side is read. With none
            // held both keys are read off their batches, which still
            // gathers no tuple on either side
            let against = match &self.held {
                Some(held) => self.right.compare_key_to(&held.key),
                None => self.right.compare_key_to_row(&self.left),
            };
            match against {
                // The right side is still in an earlier group
                Ordering::Less => {
                    self.right.advance();
                    continue;
                }
                // Past this group, so nothing more can match it
                Ordering::Greater => break,
                Ordering::Equal => {}
            }
            let right_match = self.right.match_value();
            let ordering = compare_values(&right_match, left_match);
            let take = match ordering {
                Ordering::Less => backward,
                Ordering::Equal => allows_equal,
                Ordering::Greater => false,
            };
            if backward {
                if take {
                    // The nearest so far; keep walking in case a nearer one
                    // follows
                    self.hold_current(right_match);
                    self.right.advance();
                    continue;
                }
                // Past the left row, so the held one is the answer and this
                // row stays for the next left row
                break;
            }
            // Reaching forward, the first row not before the left one is the
            // answer, and it stays under the cursor for the next left row
            if take || ordering == Ordering::Greater {
                self.hold_current(right_match);
                break;
            }
            self.right.advance();
        }

        let Some(held) = &self.held else {
            return Candidate::Unmatched;
        };
        // A held row is in this left row's group by construction: one held
        // before the walk was dropped above if its key differed, and one the
        // walk took was taken off a right row whose key compared equal. The
        // key is not read again to say so
        debug_assert_eq!(
            self.left.compare_key_to(&held.key),
            Ordering::Equal,
            "the ASOF merge held a row outside the left row's group"
        );
        // The held row was the answer for an earlier left row and the left
        // side has moved on since, so it is checked against this left row
        // rather than assumed to still be on the right side of it
        let acceptable = match compare_values(&held.match_value, left_match) {
            Ordering::Equal => allows_equal,
            Ordering::Less => backward,
            Ordering::Greater => !backward,
        };
        if !acceptable {
            return Candidate::Unmatched;
        }
        if within_tolerance(
            &held.match_value,
            left_match,
            self.tolerance.as_ref(),
            self.spec.tolerance.as_ref(),
        ) {
            Candidate::Matched
        } else {
            Candidate::Unmatched
        }
    }

    /// Takes the right row the cursor is on as the candidate, reusing the
    /// buffers the previous candidate held rather than allocating two vectors
    /// per right row advanced.
    #[inline]
    fn hold_current(&mut self, match_value: Ordered) {
        let row = self.right.row;
        match &mut self.held {
            Some(held) => {
                self.right.key_into(&mut held.key);
                held.match_value = match_value;
                held.row = row;
                held.values.clear();
                held.materialized = false;
            }
            None => {
                let mut key = Vec::new();
                self.right.key_into(&mut key);
                self.held = Some(HeldRow {
                    key,
                    match_value,
                    row,
                    values: Vec::new(),
                    materialized: false,
                });
            }
        }
    }

    /// Pulls the right side's next batch, keeping any held row readable.
    ///
    /// A held row that is still only a position is read out of its batch
    /// before the cursor replaces that batch, which is the one time the
    /// merge copies a right row it has not emitted.
    async fn right_pull(&mut self) -> Result<bool> {
        if let Some(held) = &mut self.held
            && !held.materialized
            && self.right.at_batch_end()
        {
            self.right.row_into(held.row, &mut held.values);
            held.materialized = true;
        }
        self.right.ensure_row().await
    }

    /// Builders for the right half of an output batch, shaped like the right
    /// batch's own columns so a matched row copies cell by cell with no
    /// scalar in between, or like the schema when no right batch is in hand
    fn right_builders(&self, want: usize) -> Vec<ColumnBuilder> {
        match &self.right.batch {
            Some(loaded) => loaded
                .batch
                .columns
                .iter()
                .map(|column| ColumnBuilder::shaped_like(column, want))
                .collect(),
            None => create_builders(&self.spec.right_schema, want),
        }
    }

    /// Takes the held row into the output batch, on the run of right rows
    /// still readable off the batch in hand, and as its own cells out of
    /// the copy the merge took of it otherwise
    #[inline]
    fn emit_held(&self, pending: &mut Vec<u32>, builders: &mut [ColumnBuilder]) -> Result<()> {
        let Some(held) = &self.held else {
            return Err(ZyronError::ExecutionError(
                "the ASOF merge reported a match with no held row".to_string(),
            ));
        };
        if !held.materialized {
            if self.right.batch.is_none() {
                return Err(ZyronError::ExecutionError(
                    "the ASOF merge holds a right row with no batch to read it from".to_string(),
                ));
            }
            pending.push(held.row as u32);
            return Ok(());
        }
        // Its cells come from the copy rather than the batch, so the run
        // reaching up to it goes out first and order is kept
        self.flush_pending(pending, builders)?;
        for (builder, value) in builders.iter_mut().zip(&held.values) {
            builder.push(value);
        }
        Ok(())
    }

    /// Appends the run of right rows gathered so far to the output
    /// builders, one gather per column rather than one copy per row, and
    /// empties it.
    ///
    /// Called before anything that is not a row of the batch in hand goes
    /// out, and before that batch is replaced, so the indices always name
    /// the batch they were read against
    fn flush_pending(&self, pending: &mut Vec<u32>, builders: &mut [ColumnBuilder]) -> Result<()> {
        if pending.is_empty() {
            return Ok(());
        }
        let Some(loaded) = &self.right.batch else {
            return Err(ZyronError::ExecutionError(
                "the ASOF merge gathered right rows with no batch to read them from".to_string(),
            ));
        };
        for (builder, column) in builders.iter_mut().zip(&loaded.batch.columns) {
            if !builder.gather_rows_from(column, pending) {
                for &row in pending.iter() {
                    builder.push_row_from(column, row as usize);
                }
            }
        }
        pending.clear();
        Ok(())
    }

    /// Puts the left half of the output batch into `columns`.
    ///
    /// The rows a merge keeps are read in order, so they are a run of the
    /// batch whenever nothing between the first and the last was dropped.
    /// A run that is the whole batch and leaves it spent is moved out of it
    /// whole, a shorter run is one copy per column, and only a run broken by
    /// dropped rows is gathered index by index
    fn gather_left(
        &mut self,
        left_rows: &[u32],
        rows_in_batch: usize,
        left_width: usize,
        columns: &mut Vec<Column>,
    ) -> Result<()> {
        let first = left_rows[0] as usize;
        let len = left_rows.len();
        let run = left_rows[len - 1] as usize == first + len - 1;
        if run
            && first == 0
            && len == rows_in_batch
            && self.left.row >= rows_in_batch
            && let Some(loaded) = self.left.batch.take()
        {
            self.left.row = 0;
            columns.extend(loaded.batch.columns.into_iter().take(left_width));
            return Ok(());
        }
        let Some(loaded) = self.left.batch.as_ref() else {
            return Err(ZyronError::ExecutionError(
                "the ASOF merge lost the left batch it was gathering from".to_string(),
            ));
        };
        for column in loaded.batch.columns.iter().take(left_width) {
            columns.push(if run {
                column.slice(first, len)
            } else {
                column.take(left_rows)
            });
        }
        Ok(())
    }

    /// Matches left rows against the right side until the left batch is read
    /// out, the output batch is full, or the right side needs its next batch.
    ///
    /// Every row of the merge runs here rather than in the async body, so
    /// the cursors, the counters and the row being matched are ordinary
    /// locals. A left row the right side cannot answer yet is left under the
    /// cursor, and the caller re-enters on it once the pull is done
    fn merge_rows(
        &mut self,
        left_rows: &mut Vec<u32>,
        pending: &mut Vec<u32>,
        builders: &mut [ColumnBuilder],
        rows_in_batch: usize,
        keep_unmatched: bool,
    ) -> Result<Halted> {
        while self.left.row < rows_in_batch && left_rows.len() < self.batch_size {
            // Cancellation is polled per batch of rows rather than per row:
            // the check is a relaxed load, and ten million of them is time
            // the merge is not spending on rows
            if left_rows.len() % 4096 == 0 {
                self.ctx.check_cancelled()?;
            }
            let left_match = self.left.match_value();
            match self.candidate_in_batch(&left_match) {
                Candidate::NeedsInput => return Ok(Halted::NeedsRight),
                Candidate::Matched => {
                    left_rows.push(self.left.row as u32);
                    self.emit_held(pending, builders)?;
                }
                Candidate::Unmatched if keep_unmatched => {
                    left_rows.push(self.left.row as u32);
                    self.flush_pending(pending, builders)?;
                    for builder in builders.iter_mut() {
                        builder.push_null();
                    }
                }
                Candidate::Unmatched => {}
            }
            self.left.advance();
        }
        Ok(Halted::Done)
    }

    /// Fills one output batch, or returns None when the left side is done.
    ///
    /// One output batch is gathered from one left input batch, so the left
    /// columns come out with a single take over the batch the cursor is on
    /// rather than a copy per row. The right side is pulled only when its
    /// batch is spent, so between pulls the merge is a plain loop over rows
    async fn next_batch(&mut self) -> Result<Option<DataBatch>> {
        self.read_tolerance()?;
        let keep_unmatched = matches!(self.spec.join_type, JoinType::Left);
        let left_width = self.spec.left_schema.len();
        let right_width = self.spec.right_schema.len();

        loop {
            if !self.left.ensure_row().await? {
                return Ok(None);
            }
            // The right side has a batch in hand before the builders are
            // shaped, so a first output batch copies off the right batch
            // rather than through scalars
            if !self.right.has_row() && !self.right.exhausted {
                self.right_pull().await?;
            }
            let rows_in_batch = self
                .left
                .batch
                .as_ref()
                .map(|l| l.batch.num_rows)
                .unwrap_or(0);
            let want = self.batch_size.min(rows_in_batch);
            let mut right_builders = self.right_builders(want);

            // Both buffers belong to the operator and are refilled here, so
            // neither the gathered rows nor a row's key is allocated again
            let mut left_rows = std::mem::take(&mut self.left_rows);
            let mut pending = std::mem::take(&mut self.right_pending);
            left_rows.clear();
            pending.clear();
            // The rows run through a plain call, which holds its counters and
            // the row it is on in registers. Run inside the async body they
            // live in its frame instead, because the pull below is an await
            // the whole loop is wrapped around
            loop {
                let halted = self.merge_rows(
                    &mut left_rows,
                    &mut pending,
                    &mut right_builders,
                    rows_in_batch,
                    keep_unmatched,
                )?;
                if matches!(halted, Halted::Done) {
                    break;
                }
                // The rows gathered so far name the batch the pull is about
                // to replace
                self.flush_pending(&mut pending, &mut right_builders)?;
                self.right_pull().await?;
            }
            self.flush_pending(&mut pending, &mut right_builders)?;

            self.right_pending = pending;

            if left_rows.is_empty() {
                // Every left row read so far was unmatched and dropped, so
                // there is nothing to hand up yet; read on
                self.left_rows = left_rows;
                continue;
            }
            let mut columns: Vec<Column> = Vec::with_capacity(left_width + right_width);
            self.gather_left(&left_rows, rows_in_batch, left_width, &mut columns)?;
            self.left_rows = left_rows;
            for builder in right_builders {
                columns.push(builder.finish());
            }
            return Ok(Some(DataBatch::new(columns)));
        }
    }
}

impl Operator for AsofJoinOperator {
    fn next(&mut self) -> OperatorResult<'_> {
        Box::pin(async move {
            if self.finished {
                return Ok(None);
            }
            match self.next_batch().await? {
                Some(batch) => Ok(Some(ExecutionBatch::new(batch))),
                None => {
                    self.finished = true;
                    Ok(None)
                }
            }
        })
    }
}

/// Orders one column of an equality key against another. A null key never
/// equals another, which is what keeps a null from joining, so it orders
/// after every value, and so does a pair the values cannot order
#[inline]
fn compare_key_parts(left: &Ordered, right: &Ordered) -> Ordering {
    match (left, right) {
        (Ordered::Int(a), Ordered::Int(b)) => a.cmp(b),
        (Ordered::Null, _) | (_, Ordered::Null) => Ordering::Greater,
        (Ordered::Other(a), Ordered::Other(b)) => a.partial_cmp(b).unwrap_or(Ordering::Greater),
        _ => Ordering::Greater,
    }
}

/// Orders two equality-key tuples by their first difference.
///
/// The merge itself orders a key against a row rather than against another
/// tuple, so this is the statement of what that ordering means and what the
/// tests hold it to
#[cfg(test)]
fn compare_keys(left: &[Ordered], right: &[Ordered]) -> Ordering {
    for (l, r) in left.iter().zip(right) {
        match compare_key_parts(l, r) {
            Ordering::Equal => continue,
            other => return other,
        }
    }
    Ordering::Equal
}

/// Orders two match-column values. A null orders after every value, so it
/// never becomes anyone's nearest row.
#[inline]
fn compare_values(left: &Ordered, right: &Ordered) -> Ordering {
    match (left, right) {
        (Ordered::Int(a), Ordered::Int(b)) => a.cmp(b),
        (Ordered::Null, _) => Ordering::Greater,
        (_, Ordered::Null) => Ordering::Less,
        (Ordered::Other(a), Ordered::Other(b)) => a.partial_cmp(b).unwrap_or(Ordering::Greater),
        _ => Ordering::Greater,
    }
}

/// True when a candidate lies within the reach the match condition allows.
/// Without a tolerance every candidate is in reach.
fn within_tolerance(
    candidate: &Ordered,
    target: &Ordered,
    tolerance: Option<&Ordered>,
    declared: Option<&AsofTolerance>,
) -> bool {
    let (Some(tolerance), Some(declared)) = (tolerance, declared) else {
        return true;
    };
    let (Some(a), Some(b), Some(bound)) = (
        numeric_of(candidate),
        numeric_of(target),
        numeric_of(tolerance),
    ) else {
        return false;
    };
    let distance = (b - a).abs();
    if declared.inclusive {
        distance <= bound
    } else {
        distance < bound
    }
}

/// A value as the number its distance is measured in. A timestamp is
/// microseconds and an interval is converted to microseconds, so both sides
/// of a distance are in the same units.
fn numeric_of(value: &Ordered) -> Option<f64> {
    match value {
        Ordered::Null => None,
        Ordered::Int(v) => Some(*v as f64),
        Ordered::Other(ScalarValue::Float32(v)) => Some(*v as f64),
        Ordered::Other(ScalarValue::Float64(v)) => Some(*v),
        Ordered::Other(ScalarValue::Interval(i)) => Some(interval_micros(i)),
        Ordered::Other(_) => None,
    }
}

/// An interval's span in microseconds. A month is the average Gregorian
/// month, which is the only figure available without a calendar date, and a
/// tolerance written in months is a coarse bound by nature.
fn interval_micros(interval: &zyron_common::Interval) -> f64 {
    const MICROS_PER_DAY: f64 = 86_400_000_000.0;
    const DAYS_PER_MONTH: f64 = 30.436_875;
    interval.months as f64 * DAYS_PER_MONTH * MICROS_PER_DAY
        + interval.days as f64 * MICROS_PER_DAY
        + interval.nanoseconds as f64 / 1000.0
}

#[cfg(test)]
mod tests {
    use super::*;
    use zyron_planner::logical::AsofDirection;

    #[test]
    fn a_null_key_never_equals_another() {
        let a = vec![Ordered::Null];
        let b = vec![Ordered::Null];
        assert_ne!(compare_keys(&a, &b), Ordering::Equal);
    }

    #[test]
    fn keys_order_by_their_first_difference() {
        let a = vec![Ordered::Int(1), Ordered::Int(5)];
        let b = vec![Ordered::Int(1), Ordered::Int(9)];
        assert_eq!(compare_keys(&a, &b), Ordering::Less);
        assert_eq!(compare_keys(&a, &a), Ordering::Equal);
    }

    #[test]
    fn integers_of_different_widths_order_as_one_wide_integer() {
        let narrow = Column::new(ColumnData::Int32(vec![7]), TypeId::Int32);
        let wide = Column::new(ColumnData::Int64(vec![7]), TypeId::Int64);
        assert_eq!(
            compare_key_parts(&Ordered::at(&narrow, 0), &Ordered::at(&wide, 0)),
            Ordering::Equal
        );
        let text = Column::new(ColumnData::Utf8(vec!["b".to_string()]), TypeId::Text);
        assert_eq!(
            compare_key_parts(
                &Ordered::at(&text, 0),
                &Ordered::Other(ScalarValue::Utf8("a".to_string()))
            ),
            Ordering::Greater
        );
    }

    #[test]
    fn a_null_cell_reads_as_null_whatever_the_column_holds() {
        let mut column = Column::new(ColumnData::Int64(vec![5]), TypeId::Int64);
        column.nulls = crate::column::NullBitmap::all_null(1);
        assert_eq!(Ordered::at(&column, 0), Ordered::Null);
    }

    #[test]
    fn a_null_match_value_orders_after_every_value() {
        assert_eq!(
            compare_values(&Ordered::Null, &Ordered::Int(5)),
            Ordering::Greater
        );
        assert_eq!(
            compare_values(&Ordered::Int(5), &Ordered::Null),
            Ordering::Less
        );
    }

    #[test]
    fn a_tolerance_bounds_how_far_a_match_reaches() {
        let declared = AsofTolerance {
            bound: BoundExpr::Literal {
                value: zyron_parser::ast::LiteralValue::Integer(10),
                type_id: TypeId::Int64,
            },
            inclusive: true,
        };
        let bound = Ordered::of(ScalarValue::Int64(10));
        assert!(within_tolerance(
            &Ordered::Int(95),
            &Ordered::Int(100),
            Some(&bound),
            Some(&declared)
        ));
        assert!(!within_tolerance(
            &Ordered::Int(80),
            &Ordered::Int(100),
            Some(&bound),
            Some(&declared)
        ));
        // Exactly at the bound matches only when it was written with <=
        assert!(within_tolerance(
            &Ordered::Int(90),
            &Ordered::Int(100),
            Some(&bound),
            Some(&declared)
        ));
        let strict = AsofTolerance {
            inclusive: false,
            ..declared
        };
        assert!(!within_tolerance(
            &Ordered::Int(90),
            &Ordered::Int(100),
            Some(&bound),
            Some(&strict)
        ));
    }

    #[test]
    fn no_tolerance_leaves_the_reach_unbounded() {
        assert!(within_tolerance(
            &Ordered::Int(0),
            &Ordered::Int(1_000_000),
            None,
            None
        ));
    }

    #[test]
    fn an_interval_is_measured_in_the_microseconds_a_timestamp_holds() {
        let five_minutes = zyron_common::Interval {
            months: 0,
            days: 0,
            nanoseconds: 5 * 60 * 1_000_000_000,
        };
        assert_eq!(interval_micros(&five_minutes), 5.0 * 60.0 * 1_000_000.0);
        assert_eq!(
            numeric_of(&Ordered::of(ScalarValue::Interval(five_minutes))),
            Some(5.0 * 60.0 * 1_000_000.0)
        );
    }

    #[test]
    fn direction_says_which_way_the_match_reaches() {
        assert!(AsofDirection::Backward.is_backward());
        assert!(AsofDirection::BackwardStrict.is_backward());
        assert!(!AsofDirection::Forward.is_backward());
        assert!(AsofDirection::Backward.allows_equal());
        assert!(!AsofDirection::BackwardStrict.allows_equal());
        assert!(AsofDirection::Forward.allows_equal());
        assert!(!AsofDirection::ForwardStrict.allows_equal());
    }
}
