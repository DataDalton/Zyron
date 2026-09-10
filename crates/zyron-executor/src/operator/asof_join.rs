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

use std::cmp::Ordering;
use std::sync::Arc;

use zyron_common::{Result, TypeId, ZyronError};
use zyron_parser::ast::JoinType;
use zyron_planner::binder::BoundExpr;
use zyron_planner::logical::AsofTolerance;
use zyron_planner::physical::AsofJoinSpec;

use crate::batch::{ColumnBuilder, DataBatch};
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
    tolerance: Option<ScalarValue>,
    tolerance_read: bool,
    /// The right row the merge is holding as the current candidate, and the
    /// equality group it belongs to. Cleared when the left side moves into a
    /// group the held row is not in
    held: Option<HeldRow>,
    finished: bool,
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
    key: Vec<ScalarValue>,
    match_value: ScalarValue,
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

    /// Makes sure a row is available, pulling batches until one has rows or
    /// the input is done. Returns false when the side is exhausted.
    async fn ensure_row(&mut self) -> Result<bool> {
        loop {
            if let Some(loaded) = &self.batch
                && self.row < loaded.batch.num_rows
            {
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
    fn key_into(&self, out: &mut Vec<ScalarValue>) {
        out.clear();
        if let Some(loaded) = &self.batch {
            out.extend(loaded.key_columns.iter().map(|c| c.get_scalar(self.row)));
        }
    }

    /// The key of the row the cursor is on, ordered against a key already in
    /// hand.
    ///
    /// Compares column by column off the batch, so a key that differs in its
    /// first column costs one read rather than a materialized tuple.
    fn compare_key_to(&self, other: &[ScalarValue]) -> Ordering {
        let Some(loaded) = &self.batch else {
            return Ordering::Greater;
        };
        for (column, want) in loaded.key_columns.iter().zip(other) {
            let have = column.get_scalar(self.row);
            if matches!(have, ScalarValue::Null) || matches!(want, ScalarValue::Null) {
                return Ordering::Greater;
            }
            match have.partial_cmp(want) {
                Some(Ordering::Equal) => continue,
                Some(other) => return other,
                None => return Ordering::Greater,
            }
        }
        Ordering::Equal
    }

    /// The match value of the row the cursor is on.
    fn match_value(&self) -> ScalarValue {
        match &self.batch {
            Some(loaded) => loaded.match_column.get_scalar(self.row),
            None => ScalarValue::Null,
        }
    }

    fn advance(&mut self) {
        self.row += 1;
    }

    /// True when the cursor has read past the batch it holds, so the next
    /// `ensure_row` replaces it.
    fn at_batch_end(&self) -> bool {
        match &self.batch {
            Some(loaded) => self.row >= loaded.batch.num_rows,
            None => true,
        }
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

    /// One column of one row of the batch the cursor holds.
    fn value_at(&self, at: usize, column: usize) -> ScalarValue {
        match &self.batch {
            Some(loaded) => match loaded.batch.columns.get(column) {
                Some(c) if at < loaded.batch.num_rows => c.get_scalar(at),
                _ => ScalarValue::Null,
            },
            None => ScalarValue::Null,
        }
    }
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
        self.tolerance = Some(column.get_scalar(0));
        Ok(())
    }

    /// Advances the right side so its held candidate is the nearest row for
    /// the left row the merge is on, and says whether that candidate matches.
    ///
    /// The matched row's values stay in `self.held` rather than being handed
    /// back, so a left row that matches costs no copy of them: the caller
    /// pushes straight out of the held row.
    async fn candidate_for(
        &mut self,
        left_key: &[ScalarValue],
        left_match: &ScalarValue,
    ) -> Result<bool> {
        // A held row from an earlier equality group is not a candidate for
        // this one
        if let Some(held) = &self.held
            && compare_keys(&held.key, left_key) != Ordering::Equal
        {
            self.held = None;
        }

        let backward = self.spec.direction.is_backward();
        let allows_equal = self.spec.direction.allows_equal();

        while self.right_ensure_row().await? {
            // The right key is compared off the batch rather than gathered
            // into a tuple, so a row in another group costs one read
            match self.right.compare_key_to(left_key) {
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
            return Ok(false);
        };
        if compare_keys(&held.key, left_key) != Ordering::Equal {
            return Ok(false);
        }
        // The held row was the answer for an earlier left row and the left
        // side has moved on since, so it is checked against this left row
        // rather than assumed to still be on the right side of it
        let acceptable = match compare_values(&held.match_value, left_match) {
            Ordering::Equal => allows_equal,
            Ordering::Less => backward,
            Ordering::Greater => !backward,
        };
        if !acceptable {
            return Ok(false);
        }
        Ok(within_tolerance(
            &held.match_value,
            left_match,
            self.tolerance.as_ref(),
            self.spec.tolerance.as_ref(),
        ))
    }

    /// Takes the right row the cursor is on as the candidate, reusing the
    /// buffers the previous candidate held rather than allocating two vectors
    /// per right row advanced.
    fn hold_current(&mut self, match_value: ScalarValue) {
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

    /// Advances the right cursor, keeping any held row readable.
    ///
    /// A held row that is still only a position is read out of its batch
    /// before the cursor replaces that batch, which is the one time the
    /// merge copies a right row it has not emitted.
    async fn right_ensure_row(&mut self) -> Result<bool> {
        if let Some(held) = &mut self.held
            && !held.materialized
            && self.right.at_batch_end()
        {
            self.right.row_into(held.row, &mut held.values);
            held.materialized = true;
        }
        self.right.ensure_row().await
    }

    /// Fills one output batch, or returns None when the left side is done.
    ///
    /// One output batch is gathered from one left input batch, so the left
    /// columns come out with a single take over the batch the cursor is on
    /// rather than a copy per row.
    async fn next_batch(&mut self) -> Result<Option<DataBatch>> {
        self.read_tolerance()?;
        let keep_unmatched = matches!(self.spec.join_type, JoinType::Left);
        let left_width = self.spec.left_schema.len();
        let right_width = self.spec.right_schema.len();

        loop {
            if !self.left.ensure_row().await? {
                return Ok(None);
            }
            let rows_in_batch = self
                .left
                .batch
                .as_ref()
                .map(|l| l.batch.num_rows)
                .unwrap_or(0);
            let want = self.batch_size.min(rows_in_batch);
            let mut left_rows: Vec<u32> = Vec::with_capacity(want);
            let mut right_builders: Vec<ColumnBuilder> = self
                .spec
                .right_schema
                .iter()
                .map(|c| ColumnBuilder::new(c.type_id, want))
                .collect();

            // One buffer for the left key, refilled per row rather than
            // allocated per row
            let mut left_key: Vec<ScalarValue> = Vec::new();
            while self.left.row < rows_in_batch && left_rows.len() < self.batch_size {
                // Cancellation is polled per batch of rows rather than per
                // row: the check is a relaxed load, and ten million of them
                // is time the merge is not spending on rows
                if left_rows.len() % 4096 == 0 {
                    self.ctx.check_cancelled()?;
                }
                self.left.key_into(&mut left_key);
                let left_match = self.left.match_value();
                if self.candidate_for(&left_key, &left_match).await? {
                    let Some(held) = self.held.as_mut() else {
                        return Err(ZyronError::ExecutionError(
                            "the ASOF merge reported a match with no held row".to_string(),
                        ));
                    };
                    left_rows.push(self.left.row as u32);
                    // Read out of the batch the first time this candidate
                    // answers, and out of the held row afterwards. One held
                    // row is the answer for a run of left rows, so the copy
                    // is paid once for the run rather than per output row,
                    // and a candidate the merge passed over without ever
                    // answering is never copied at all
                    if !held.materialized {
                        self.right.row_into(held.row, &mut held.values);
                        held.materialized = true;
                    }
                    for (i, builder) in right_builders.iter_mut().enumerate() {
                        builder.push(&held.values[i]);
                    }
                } else if keep_unmatched {
                    left_rows.push(self.left.row as u32);
                    for builder in right_builders.iter_mut() {
                        builder.push_null();
                    }
                }
                self.left.advance();
            }

            if left_rows.is_empty() {
                // Every left row read so far was unmatched and dropped, so
                // there is nothing to hand up yet; read on
                continue;
            }
            let Some(loaded) = self.left.batch.as_ref() else {
                return Err(ZyronError::ExecutionError(
                    "the ASOF merge lost the left batch it was gathering from".to_string(),
                ));
            };
            let mut columns: Vec<Column> = Vec::with_capacity(left_width + right_width);
            for column in loaded.batch.columns.iter().take(left_width) {
                columns.push(column.take(&left_rows));
            }
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

/// Orders two equality-key tuples. A null key never equals another, which is
/// what keeps a null from joining, so it orders after every value.
fn compare_keys(left: &[ScalarValue], right: &[ScalarValue]) -> Ordering {
    for (l, r) in left.iter().zip(right) {
        if matches!(l, ScalarValue::Null) || matches!(r, ScalarValue::Null) {
            return Ordering::Greater;
        }
        match l.partial_cmp(r) {
            Some(Ordering::Equal) => continue,
            Some(other) => return other,
            None => return Ordering::Greater,
        }
    }
    Ordering::Equal
}

/// Orders two match-column values. A null orders after every value, so it
/// never becomes anyone's nearest row.
fn compare_values(left: &ScalarValue, right: &ScalarValue) -> Ordering {
    if matches!(left, ScalarValue::Null) {
        return Ordering::Greater;
    }
    if matches!(right, ScalarValue::Null) {
        return Ordering::Less;
    }
    left.partial_cmp(right).unwrap_or(Ordering::Greater)
}

/// True when a candidate lies within the reach the match condition allows.
/// Without a tolerance every candidate is in reach.
fn within_tolerance(
    candidate: &ScalarValue,
    target: &ScalarValue,
    tolerance: Option<&ScalarValue>,
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
fn numeric_of(value: &ScalarValue) -> Option<f64> {
    match value {
        ScalarValue::Int8(v) => Some(*v as f64),
        ScalarValue::Int16(v) => Some(*v as f64),
        ScalarValue::Int32(v) => Some(*v as f64),
        ScalarValue::Int64(v) => Some(*v as f64),
        ScalarValue::Int128(v) => Some(*v as f64),
        ScalarValue::UInt8(v) => Some(*v as f64),
        ScalarValue::UInt16(v) => Some(*v as f64),
        ScalarValue::UInt32(v) => Some(*v as f64),
        ScalarValue::UInt64(v) => Some(*v as f64),
        ScalarValue::Float32(v) => Some(*v as f64),
        ScalarValue::Float64(v) => Some(*v),
        ScalarValue::Interval(i) => Some(interval_micros(i)),
        _ => None,
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
        let a = vec![ScalarValue::Null];
        let b = vec![ScalarValue::Null];
        assert_ne!(compare_keys(&a, &b), Ordering::Equal);
    }

    #[test]
    fn keys_order_by_their_first_difference() {
        let a = vec![ScalarValue::Int64(1), ScalarValue::Int64(5)];
        let b = vec![ScalarValue::Int64(1), ScalarValue::Int64(9)];
        assert_eq!(compare_keys(&a, &b), Ordering::Less);
        assert_eq!(compare_keys(&a, &a), Ordering::Equal);
    }

    #[test]
    fn a_null_match_value_orders_after_every_value() {
        assert_eq!(
            compare_values(&ScalarValue::Null, &ScalarValue::Int64(5)),
            Ordering::Greater
        );
        assert_eq!(
            compare_values(&ScalarValue::Int64(5), &ScalarValue::Null),
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
        let bound = ScalarValue::Int64(10);
        assert!(within_tolerance(
            &ScalarValue::Int64(95),
            &ScalarValue::Int64(100),
            Some(&bound),
            Some(&declared)
        ));
        assert!(!within_tolerance(
            &ScalarValue::Int64(80),
            &ScalarValue::Int64(100),
            Some(&bound),
            Some(&declared)
        ));
        // Exactly at the bound matches only when it was written with <=
        assert!(within_tolerance(
            &ScalarValue::Int64(90),
            &ScalarValue::Int64(100),
            Some(&bound),
            Some(&declared)
        ));
        let strict = AsofTolerance {
            inclusive: false,
            ..declared
        };
        assert!(!within_tolerance(
            &ScalarValue::Int64(90),
            &ScalarValue::Int64(100),
            Some(&bound),
            Some(&strict)
        ));
    }

    #[test]
    fn no_tolerance_leaves_the_reach_unbounded() {
        assert!(within_tolerance(
            &ScalarValue::Int64(0),
            &ScalarValue::Int64(1_000_000),
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
