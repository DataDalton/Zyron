//! Expression evaluator for bound expressions against DataBatch columns.
//!
//! Evaluates BoundExpr trees from the planner, producing Column results
//! using the custom compute kernels.

use std::borrow::Cow;

use zyron_catalog::ColumnId;
use zyron_common::{Result, TypeId, ZyronError};
use zyron_parser::ast::{BinaryOperator, LiteralValue, UnaryOperator};
use zyron_planner::binder::{BoundExpr, BoundWhen, ColumnRef};
use zyron_planner::logical::LogicalColumn;

use crate::batch::DataBatch;
use crate::column::{Column, ColumnData, NullBitmap, ScalarValue};
use crate::compute::{
    self, ArithOp, CmpOp, bool_and, bool_not, bool_or, cast_column, column_to_mask, compare,
    concat_strings, is_not_null, is_null, negate,
};

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Evaluates a ColumnRef without cloning, returning a borrow into the batch.
/// Falls back to `evaluate` for non-ColumnRef expressions, which still produces
/// an owned Column. Operators that only read the result (hash aggregate, sort
/// keys, filter predicates with chained refs) should prefer this entry so a
/// ColumnRef does not pay the full Vec<T> clone cost on every batch.
pub fn evaluate_borrowed<'a>(
    expr: &BoundExpr,
    batch: &'a DataBatch,
    schema: &[LogicalColumn],
    params: &[ScalarValue],
) -> Result<Cow<'a, Column>> {
    if let BoundExpr::ColumnRef(col_ref) = expr {
        let idx = resolve_column_index(col_ref.table_idx, col_ref.column_id, schema)?;
        return Ok(Cow::Borrowed(batch.column(idx)));
    }
    Ok(Cow::Owned(evaluate(expr, batch, schema, params)?))
}

/// Evaluates a bound expression against a DataBatch, returning the result as a Column.
/// The `params` slice provides values for query parameters ($1, $2, ...).
pub fn evaluate(
    expr: &BoundExpr,
    batch: &DataBatch,
    schema: &[LogicalColumn],
    params: &[ScalarValue],
) -> Result<Column> {
    match expr {
        BoundExpr::ColumnRef(col_ref) => evaluate_column_ref(col_ref, batch, schema),
        BoundExpr::Literal { value, type_id } => evaluate_literal(value, *type_id, batch.num_rows),
        BoundExpr::BinaryOp {
            left, op, right, ..
        } => evaluate_binary_op(left, *op, right, batch, schema, params),
        BoundExpr::UnaryOp {
            op, expr: inner, ..
        } => evaluate_unary_op(*op, inner, batch, schema, params),
        BoundExpr::IsNull {
            expr: inner,
            negated,
        } => evaluate_is_null(inner, *negated, batch, schema, params),
        BoundExpr::InList {
            expr: inner,
            list,
            negated,
        } => evaluate_in_list(inner, list, *negated, batch, schema, params),
        BoundExpr::Between {
            expr: inner,
            low,
            high,
            negated,
        } => evaluate_between(inner, low, high, *negated, batch, schema, params),
        BoundExpr::Like {
            expr: inner,
            pattern,
            negated,
        } => {
            let col = evaluate(inner, batch, schema, params)?;
            let pat = evaluate(pattern, batch, schema, params)?;
            compute::like(&col, &pat, *negated)
        }
        BoundExpr::ILike {
            expr: inner,
            pattern,
            negated,
        } => {
            let col = evaluate(inner, batch, schema, params)?;
            let pat = evaluate(pattern, batch, schema, params)?;
            compute::ilike(&col, &pat, *negated)
        }
        BoundExpr::Function { name, args, .. } => {
            evaluate_function(name, args, batch, schema, params)
        }
        BoundExpr::AggregateFunction { .. } => Err(ZyronError::ExecutionError(
            "aggregate functions must be evaluated by the aggregate operator".to_string(),
        )),
        BoundExpr::Cast {
            expr: inner,
            target_type,
            fractional_digits,
        } => {
            let col = evaluate(inner, batch, schema, params)?;
            // A cast naming a decimal carries the scale to land on, which
            // the generic cast has no way to know
            if *target_type == TypeId::Decimal {
                return crate::compute::cast_column_to_decimal(
                    &col,
                    fractional_digits.unwrap_or(0),
                );
            }
            cast_column(&col, *target_type)
        }
        BoundExpr::Case {
            operand,
            conditions,
            else_result,
            type_id,
        } => evaluate_case(
            operand.as_deref(),
            conditions,
            else_result.as_deref(),
            *type_id,
            batch,
            schema,
            params,
        ),
        BoundExpr::Nested(inner) => evaluate(inner, batch, schema, params),
        BoundExpr::Subquery { .. } | BoundExpr::Exists { .. } | BoundExpr::InSubquery { .. } => {
            Err(ZyronError::ExecutionError(
                "subqueries not supported in executor".to_string(),
            ))
        }
        BoundExpr::WindowFunction { .. } => Err(ZyronError::ExecutionError(
            "window functions not supported yet".to_string(),
        )),
        BoundExpr::Parameter { index, .. } => evaluate_parameter(*index, params, batch.num_rows),
        BoundExpr::TemporalRef { inner, .. } => {
            // Pure-evaluator path returns the inner column at the current
            // snapshot. Time-travel-aware operators (ROW_DIFF) inspect the
            // temporal qualifier through a parallel dispatch path before
            // reaching this generic evaluator
            evaluate(inner, batch, schema, params)
        }
    }
}

/// Finds the column position in the schema by matching table_idx and column_id.
pub fn resolve_column_index(
    table_idx: usize,
    column_id: ColumnId,
    schema: &[LogicalColumn],
) -> Result<usize> {
    for (i, col) in schema.iter().enumerate() {
        if col.table_idx == Some(table_idx) && col.column_id == column_id {
            return Ok(i);
        }
    }
    Err(ZyronError::ExecutionError(format!(
        "column not found in schema: table_idx={table_idx}, column_id={column_id}"
    )))
}

// ---------------------------------------------------------------------------
// Parameter lookup
// ---------------------------------------------------------------------------

/// Looks up a query parameter ($1, $2, ...) by index and expands it to a Column.
/// Parameter indices are 1-based in SQL but stored as 1-based in BoundExpr.
fn evaluate_parameter(index: usize, params: &[ScalarValue], num_rows: usize) -> Result<Column> {
    // Parameters use 1-based indexing ($1 = index 1).
    if index == 0 || index > params.len() {
        return Err(ZyronError::ExecutionError(format!(
            "parameter ${index} is out of range, {} parameter(s) provided",
            params.len()
        )));
    }
    let scalar = &params[index - 1];
    if matches!(scalar, ScalarValue::Null) {
        return Ok(Column::null_column(TypeId::Null, num_rows));
    }
    let type_id = scalar.type_id();
    // Vectorized broadcast (single vec![v; n] fill) instead of num_rows
    // push_scalar calls, each a 30+-variant match. Null is handled above.
    let data = ColumnData::from_scalar(scalar, num_rows);
    Ok(Column::new(data, type_id))
}

// ---------------------------------------------------------------------------
// Column reference
// ---------------------------------------------------------------------------

fn evaluate_column_ref(
    col_ref: &ColumnRef,
    batch: &DataBatch,
    schema: &[LogicalColumn],
) -> Result<Column> {
    let idx = resolve_column_index(col_ref.table_idx, col_ref.column_id, schema)?;
    Ok(batch.column(idx).clone())
}

// ---------------------------------------------------------------------------
// Literals
// ---------------------------------------------------------------------------

/// Converts a parsed literal directly to a ScalarValue coerced to the
/// target column type. Returns None when the combination is not one of
/// the common cases, the caller falls back to the full evaluator path
/// for those (e.g. Decimal, FixedBinary, Interval, casts from non-integer
/// to integer, etc.).
///
/// Hot-path entry for ValuesOperator. Skipping the 1-row Column alloc
/// + cast_column dispatch per cell drops the per-batch ValuesOp cost by
/// ~10x for the literal-only INSERT VALUES workload.
pub fn literal_to_scalar(value: &LiteralValue, target: TypeId) -> Option<ScalarValue> {
    match (value, target) {
        (LiteralValue::Null, _) => Some(ScalarValue::Null),
        (LiteralValue::Integer(v), TypeId::Int8) => Some(ScalarValue::Int8(*v as i8)),
        (LiteralValue::Integer(v), TypeId::Int16) => Some(ScalarValue::Int16(*v as i16)),
        (LiteralValue::Integer(v), TypeId::Int32) => Some(ScalarValue::Int32(*v as i32)),
        (LiteralValue::Integer(v), TypeId::Int64) => Some(ScalarValue::Int64(*v)),
        (LiteralValue::Integer(v), TypeId::Int128) => Some(ScalarValue::Int128(*v as i128)),
        (LiteralValue::Integer(v), TypeId::UInt8) => Some(ScalarValue::UInt8(*v as u8)),
        (LiteralValue::Integer(v), TypeId::UInt16) => Some(ScalarValue::UInt16(*v as u16)),
        (LiteralValue::Integer(v), TypeId::UInt32) => Some(ScalarValue::UInt32(*v as u32)),
        (LiteralValue::Integer(v), TypeId::UInt64) => Some(ScalarValue::UInt64(*v as u64)),
        (LiteralValue::Integer(v), TypeId::Float32) => Some(ScalarValue::Float32(*v as f32)),
        (LiteralValue::Integer(v), TypeId::Float64) => Some(ScalarValue::Float64(*v as f64)),
        (LiteralValue::Float(v), TypeId::Float32) => Some(ScalarValue::Float32(*v as f32)),
        (LiteralValue::Float(v), TypeId::Float64) => Some(ScalarValue::Float64(*v)),
        (LiteralValue::Boolean(b), TypeId::Boolean) => Some(ScalarValue::Boolean(*b)),
        (LiteralValue::String(s), TypeId::Text | TypeId::Char | TypeId::Varchar) => {
            Some(ScalarValue::Utf8(s.clone()))
        }
        _ => None,
    }
}

/// Coerces an already-extracted scalar (e.g. a bound `$N` parameter) to the
/// target column type without allocating a Column. Returns None for type
/// combinations not covered here, the caller then falls back to the full
/// `cast_column` path. Mirrors the common widenings the literal fast path
/// handles.
pub fn coerce_scalar_to(scalar: &ScalarValue, target: TypeId) -> Option<ScalarValue> {
    match scalar {
        ScalarValue::Null => Some(ScalarValue::Null),
        ScalarValue::Int64(v) => match target {
            TypeId::Int64 => Some(ScalarValue::Int64(*v)),
            TypeId::Int8 => Some(ScalarValue::Int8(*v as i8)),
            TypeId::Int16 => Some(ScalarValue::Int16(*v as i16)),
            TypeId::Int32 => Some(ScalarValue::Int32(*v as i32)),
            TypeId::Int128 => Some(ScalarValue::Int128(*v as i128)),
            TypeId::UInt8 => Some(ScalarValue::UInt8(*v as u8)),
            TypeId::UInt16 => Some(ScalarValue::UInt16(*v as u16)),
            TypeId::UInt32 => Some(ScalarValue::UInt32(*v as u32)),
            TypeId::UInt64 => Some(ScalarValue::UInt64(*v as u64)),
            TypeId::Float32 => Some(ScalarValue::Float32(*v as f32)),
            TypeId::Float64 => Some(ScalarValue::Float64(*v as f64)),
            _ => None,
        },
        ScalarValue::Float64(v) => match target {
            TypeId::Float64 => Some(ScalarValue::Float64(*v)),
            TypeId::Float32 => Some(ScalarValue::Float32(*v as f32)),
            _ => None,
        },
        ScalarValue::Float32(v) => match target {
            TypeId::Float32 => Some(ScalarValue::Float32(*v)),
            TypeId::Float64 => Some(ScalarValue::Float64(*v as f64)),
            _ => None,
        },
        ScalarValue::Boolean(b) => match target {
            TypeId::Boolean => Some(ScalarValue::Boolean(*b)),
            _ => None,
        },
        ScalarValue::Utf8(s) => match target {
            TypeId::Text | TypeId::Char | TypeId::Varchar => Some(ScalarValue::Utf8(s.clone())),
            _ => None,
        },
        // Already-correct integer/binary variants pass through when they
        // match the target exactly.
        other => {
            if other.type_id() == target {
                Some(other.clone())
            } else {
                None
            }
        }
    }
}

fn evaluate_literal(value: &LiteralValue, type_id: TypeId, num_rows: usize) -> Result<Column> {
    match value {
        LiteralValue::Integer(v) => Ok(Column::new(
            ColumnData::Int64(vec![*v; num_rows]),
            TypeId::Int64,
        )),
        LiteralValue::Int128(v) => Ok(Column::new(
            ColumnData::Int128(vec![*v; num_rows]),
            TypeId::Int128,
        )),
        LiteralValue::Float(v) => Ok(Column::new(
            ColumnData::Float64(vec![*v; num_rows]),
            TypeId::Float64,
        )),
        LiteralValue::String(s) => Ok(Column::new(
            ColumnData::Utf8(vec![s.clone(); num_rows]),
            TypeId::Text,
        )),
        LiteralValue::Boolean(b) => Ok(Column::new(
            ColumnData::Boolean(vec![*b; num_rows]),
            TypeId::Boolean,
        )),
        LiteralValue::Null => Ok(Column::null_column(type_id, num_rows)),
        LiteralValue::Interval(i) => Ok(Column::new(
            ColumnData::Interval(vec![*i; num_rows]),
            TypeId::Interval,
        )),
    }
}

// ---------------------------------------------------------------------------
// Binary operators
// ---------------------------------------------------------------------------

/// Returns the common type two numeric operands should be cast to before a
/// comparison or arithmetic op, or None when either side is not numeric.
fn common_numeric_type(a: TypeId, b: TypeId) -> Option<TypeId> {
    fn rank(t: TypeId) -> Option<u8> {
        match t {
            TypeId::Int8 | TypeId::UInt8 => Some(1),
            TypeId::Int16 | TypeId::UInt16 => Some(2),
            TypeId::Int32 | TypeId::UInt32 => Some(3),
            TypeId::Int64 | TypeId::UInt64 => Some(4),
            TypeId::Int128 | TypeId::UInt128 => Some(5),
            TypeId::Float32 => Some(6),
            TypeId::Float64 | TypeId::Decimal | TypeId::Money => Some(7),
            _ => None,
        }
    }
    let ra = rank(a)?;
    let rb = rank(b)?;
    let hi = ra.max(rb);
    Some(match hi {
        6 => TypeId::Float32,
        7 => TypeId::Float64,
        5 => TypeId::Int128,
        _ => TypeId::Int64,
    })
}

/// Hybrid Logical Clock state: high 48 bits = physical milliseconds since the
/// Unix epoch, low 16 bits = logical counter. Process-wide and lock-free.
static HLC_STATE: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// Standard HLC send rule, generated lock-free via a single CAS loop. The
/// returned value is monotonic and causally ordered under plain integer
/// comparison (it is stored in the i128 HLC column, high bits zero).
fn next_hlc() -> u64 {
    use std::sync::atomic::Ordering;
    let phys_ms = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0)
        & 0xFFFF_FFFF_FFFF; // 48 bits
    loop {
        let prev = HLC_STATE.load(Ordering::Acquire);
        let prev_ms = prev >> 16;
        let prev_logical = prev & 0xFFFF;
        let (new_ms, new_logical) = if phys_ms > prev_ms {
            (phys_ms, 0)
        } else if prev_logical == 0xFFFF {
            // Logical counter exhausted within this millisecond: advance the
            // physical part to preserve strict monotonicity.
            (prev_ms + 1, 0)
        } else {
            (prev_ms, prev_logical + 1)
        };
        let next = (new_ms << 16) | new_logical;
        if HLC_STATE
            .compare_exchange_weak(prev, next, Ordering::AcqRel, Ordering::Acquire)
            .is_ok()
        {
            return next;
        }
    }
}

/// Casts two numeric columns to their common type so a comparison sees matching
/// types, mirroring the coercion the binary-comparison path applies. Non-numeric
/// or already-equal pairs are left untouched.
fn coerce_numeric_pair(left: &mut Column, right: &mut Column) -> Result<()> {
    let lt = left.type_id;
    let rt = right.type_id;
    if lt != rt {
        if let Some(common) = common_numeric_type(lt, rt) {
            if lt != common {
                *left = compute::cast_column(left, common)?;
            }
            if rt != common {
                *right = compute::cast_column(right, common)?;
            }
        }
    }
    Ok(())
}

#[inline]
fn is_ts_col(col: &Column) -> bool {
    matches!(col.type_id, TypeId::Timestamp | TypeId::TimestampTz)
}

#[inline]
fn ts_col_is_ps(col: &Column) -> bool {
    is_ts_col(col) && col.fractional_digits.unwrap_or(6) > 6
}

/// When exactly one operand is a picosecond (p>6) timestamp and the other a
/// microsecond (p<=6) timestamp, scale the microsecond side up to picoseconds
/// so both express the same instant in the same unit. Equal units or
/// non-timestamp operands are left untouched.
fn normalize_ts_pair(left: &mut Column, right: &mut Column) -> Result<()> {
    if !(is_ts_col(left) && is_ts_col(right)) {
        return Ok(());
    }
    let lps = ts_col_is_ps(left);
    let rps = ts_col_is_ps(right);
    if lps == rps {
        return Ok(());
    }
    if lps {
        *right = crate::compute::scale_us_to_ps(right, left.fractional_digits)?;
    } else {
        *left = crate::compute::scale_us_to_ps(left, right.fractional_digits)?;
    }
    Ok(())
}

fn evaluate_binary_op(
    left: &BoundExpr,
    op: BinaryOperator,
    right: &BoundExpr,
    batch: &DataBatch,
    schema: &[LogicalColumn],
    params: &[ScalarValue],
) -> Result<Column> {
    let mut left_col = evaluate(left, batch, schema, params)?;
    let mut right_col = evaluate(right, batch, schema, params)?;

    // Cross-precision timestamp normalization (B5): when the two operands are
    // timestamps stored in different units (one i64 microseconds for p<=6, the
    // other i128 picoseconds for p>6), scale the microsecond side up to
    // picoseconds (exact x1_000_000) so the same instant compares equal.
    // Never the reverse - downcasting ps->us would lose information.
    if matches!(
        op,
        BinaryOperator::Plus
            | BinaryOperator::Minus
            | BinaryOperator::Eq
            | BinaryOperator::Neq
            | BinaryOperator::Lt
            | BinaryOperator::Gt
            | BinaryOperator::LtEq
            | BinaryOperator::GtEq
    ) {
        normalize_ts_pair(&mut left_col, &mut right_col)?;
    }

    // Numeric coercion: comparisons and arithmetic between different numeric
    // widths (e.g. an INT32 column vs an INT64 integer literal) must align to
    // a common type. Without this the ScalarValue fallback compares distinct
    // enum variants and never matches, so every column-vs-literal predicate
    // would be all-false.
    if matches!(
        op,
        BinaryOperator::Plus
            | BinaryOperator::Minus
            | BinaryOperator::Multiply
            | BinaryOperator::Divide
            | BinaryOperator::Modulo
            | BinaryOperator::Eq
            | BinaryOperator::Neq
            | BinaryOperator::Lt
            | BinaryOperator::Gt
            | BinaryOperator::LtEq
            | BinaryOperator::GtEq
    ) {
        // A decimal aligns by scale rather than by width. Letting the common
        // numeric type decide would pick a float and read the scaled integer
        // as a plain number, so `v > 10.00` would compare 1050 against 10
        if left_col.type_id == TypeId::Decimal || right_col.type_id == TypeId::Decimal {
            if let Some((l, r)) = compute::align_decimal_operands(&left_col, &right_col)? {
                left_col = l;
                right_col = r;
            }
        } else {
            let lt = left_col.type_id;
            let rt = right_col.type_id;
            if lt != rt {
                if let Some(common) = common_numeric_type(lt, rt) {
                    if lt != common {
                        left_col = compute::cast_column(&left_col, common)?;
                    }
                    if rt != common {
                        right_col = compute::cast_column(&right_col, common)?;
                    }
                }
            }
        }
    }

    // Intercept interval arithmetic before falling into the generic numeric path.
    if matches!(
        op,
        BinaryOperator::Plus | BinaryOperator::Minus | BinaryOperator::Multiply
    ) {
        if let Some(result) = try_interval_arithmetic(op, &left_col, &right_col)? {
            return Ok(result);
        }
    }

    match op {
        BinaryOperator::Plus => compute::arithmetic(&left_col, &right_col, ArithOp::Add),
        BinaryOperator::Minus => compute::arithmetic(&left_col, &right_col, ArithOp::Sub),
        BinaryOperator::Multiply => compute::arithmetic(&left_col, &right_col, ArithOp::Mul),
        BinaryOperator::Divide => compute::arithmetic(&left_col, &right_col, ArithOp::Div),
        BinaryOperator::Modulo => compute::arithmetic(&left_col, &right_col, ArithOp::Mod),
        BinaryOperator::Eq => compare(&left_col, &right_col, CmpOp::Eq),
        BinaryOperator::Neq => compare(&left_col, &right_col, CmpOp::Neq),
        BinaryOperator::Lt => compare(&left_col, &right_col, CmpOp::Lt),
        BinaryOperator::Gt => compare(&left_col, &right_col, CmpOp::Gt),
        BinaryOperator::LtEq => compare(&left_col, &right_col, CmpOp::LtEq),
        BinaryOperator::GtEq => compare(&left_col, &right_col, CmpOp::GtEq),
        BinaryOperator::And => bool_and(&left_col, &right_col),
        BinaryOperator::Or => bool_or(&left_col, &right_col),
        BinaryOperator::Concat => concat_strings(&left_col, &right_col),
    }
}

/// Handles arithmetic involving intervals: timestamp +/- interval, interval +/- interval,
/// interval * numeric. Returns Ok(None) when neither operand is an interval (fall through).
fn try_interval_arithmetic(
    op: BinaryOperator,
    left: &Column,
    right: &Column,
) -> Result<Option<Column>> {
    use zyron_common::{Interval, TypeId as TI};

    let is_timestamp =
        |t: TypeId| matches!(t, TI::Timestamp | TI::TimestampTz | TI::Time | TI::Date);

    // interval +/- interval -> interval
    if left.type_id == TI::Interval && right.type_id == TI::Interval {
        let la = match &left.data {
            ColumnData::Interval(v) => v,
            _ => return Ok(None),
        };
        let ra = match &right.data {
            ColumnData::Interval(v) => v,
            _ => return Ok(None),
        };
        let n = la.len().min(ra.len());
        let mut out: Vec<Interval> = Vec::with_capacity(n);
        for i in 0..n {
            let v = match op {
                BinaryOperator::Plus => la[i].add(ra[i]),
                BinaryOperator::Minus => la[i].subtract(ra[i]),
                _ => return Ok(None),
            };
            out.push(v);
        }
        return Ok(Some(Column::new(ColumnData::Interval(out), TI::Interval)));
    }

    // timestamp +/- interval -> timestamp (micros-based i64 columns)
    if is_timestamp(left.type_id) && right.type_id == TI::Interval {
        return Ok(Some(timestamp_interval_op(left, right, op, false)?));
    }
    if is_timestamp(right.type_id)
        && left.type_id == TI::Interval
        && matches!(op, BinaryOperator::Plus)
    {
        // interval + timestamp -> timestamp (commutative)
        return Ok(Some(timestamp_interval_op(right, left, op, true)?));
    }

    // interval * numeric -> interval
    if left.type_id == TI::Interval && right.type_id.is_numeric() {
        return Ok(Some(interval_scalar_mul(left, right)?));
    }
    if right.type_id == TI::Interval && left.type_id.is_numeric() {
        return Ok(Some(interval_scalar_mul(right, left)?));
    }

    Ok(None)
}

fn timestamp_interval_op(
    ts: &Column,
    iv: &Column,
    op: BinaryOperator,
    iv_on_left: bool,
) -> Result<Column> {
    let ts_values: &[i64] = match &ts.data {
        ColumnData::Int64(v) => v,
        ColumnData::Int32(v) => {
            // Date column: rare, but promote to timestamp-micros by scaling days -> us
            let promoted: Vec<i64> = v.iter().map(|&d| (d as i64) * 86_400_000_000).collect();
            let mut result: Vec<i64> = Vec::with_capacity(promoted.len());
            let iv_values = match &iv.data {
                ColumnData::Interval(v) => v,
                _ => {
                    return Err(zyron_common::ZyronError::ExecutionError(
                        "Interval column expected".into(),
                    ));
                }
            };
            let n = promoted.len().min(iv_values.len());
            for i in 0..n {
                let base = promoted[i];
                let adjusted = match (op, iv_on_left) {
                    (BinaryOperator::Plus, _) => iv_values[i].add_to_timestamp_micros(base),
                    (BinaryOperator::Minus, false) => {
                        iv_values[i].subtract_from_timestamp_micros(base)
                    }
                    _ => base,
                };
                result.push(adjusted);
            }
            return Ok(Column::new(
                ColumnData::Int64(result),
                zyron_common::TypeId::Timestamp,
            ));
        }
        _ => {
            return Err(zyron_common::ZyronError::ExecutionError(
                "Timestamp column must be Int64 or Int32".into(),
            ));
        }
    };

    let iv_values = match &iv.data {
        ColumnData::Interval(v) => v,
        _ => {
            return Err(zyron_common::ZyronError::ExecutionError(
                "Interval column expected".into(),
            ));
        }
    };

    let n = ts_values.len().min(iv_values.len());
    let mut result: Vec<i64> = Vec::with_capacity(n);
    for i in 0..n {
        let base = ts_values[i];
        let adjusted = match (op, iv_on_left) {
            (BinaryOperator::Plus, _) => iv_values[i].add_to_timestamp_micros(base),
            (BinaryOperator::Minus, false) => iv_values[i].subtract_from_timestamp_micros(base),
            _ => base,
        };
        result.push(adjusted);
    }

    Ok(Column::new(ColumnData::Int64(result), ts.type_id))
}

fn interval_scalar_mul(iv: &Column, scalar: &Column) -> Result<Column> {
    let iv_values = match &iv.data {
        ColumnData::Interval(v) => v,
        _ => {
            return Err(zyron_common::ZyronError::ExecutionError(
                "Interval column expected".into(),
            ));
        }
    };
    let scalar_as_i64: Vec<i64> = match &scalar.data {
        ColumnData::Int8(v) => v.iter().map(|&x| x as i64).collect(),
        ColumnData::Int16(v) => v.iter().map(|&x| x as i64).collect(),
        ColumnData::Int32(v) => v.iter().map(|&x| x as i64).collect(),
        ColumnData::Int64(v) => v.clone(),
        ColumnData::UInt8(v) => v.iter().map(|&x| x as i64).collect(),
        ColumnData::UInt16(v) => v.iter().map(|&x| x as i64).collect(),
        ColumnData::UInt32(v) => v.iter().map(|&x| x as i64).collect(),
        ColumnData::UInt64(v) => v.iter().map(|&x| x as i64).collect(),
        ColumnData::Float32(v) => v.iter().map(|&x| x as i64).collect(),
        ColumnData::Float64(v) => v.iter().map(|&x| x as i64).collect(),
        _ => {
            return Err(zyron_common::ZyronError::ExecutionError(
                "Scalar must be numeric for interval multiplication".into(),
            ));
        }
    };

    let n = iv_values.len().min(scalar_as_i64.len());
    let mut out: Vec<zyron_common::Interval> = Vec::with_capacity(n);
    for i in 0..n {
        out.push(iv_values[i].multiply_by(scalar_as_i64[i]));
    }
    Ok(Column::new(
        ColumnData::Interval(out),
        zyron_common::TypeId::Interval,
    ))
}

// ---------------------------------------------------------------------------
// Unary operators
// ---------------------------------------------------------------------------

fn evaluate_unary_op(
    op: UnaryOperator,
    expr: &BoundExpr,
    batch: &DataBatch,
    schema: &[LogicalColumn],
    params: &[ScalarValue],
) -> Result<Column> {
    let col = evaluate(expr, batch, schema, params)?;
    match op {
        UnaryOperator::Not => bool_not(&col),
        UnaryOperator::Minus => negate(&col),
    }
}

// ---------------------------------------------------------------------------
// IS NULL / IS NOT NULL
// ---------------------------------------------------------------------------

fn evaluate_is_null(
    expr: &BoundExpr,
    negated: bool,
    batch: &DataBatch,
    schema: &[LogicalColumn],
    params: &[ScalarValue],
) -> Result<Column> {
    let col = evaluate(expr, batch, schema, params)?;
    if negated {
        Ok(is_not_null(&col))
    } else {
        Ok(is_null(&col))
    }
}

// ---------------------------------------------------------------------------
// IN list
// ---------------------------------------------------------------------------

fn evaluate_in_list(
    expr: &BoundExpr,
    list: &[BoundExpr],
    negated: bool,
    batch: &DataBatch,
    schema: &[LogicalColumn],
    params: &[ScalarValue],
) -> Result<Column> {
    let expr_col = evaluate(expr, batch, schema, params)?;
    let num_rows = batch.num_rows;

    if list.is_empty() {
        let val = negated;
        return Ok(Column::new(
            ColumnData::Boolean(vec![val; num_rows]),
            TypeId::Boolean,
        ));
    }

    let first = evaluate(&list[0], batch, schema, params)?;
    let mut e0 = expr_col.clone();
    let mut f0 = first;
    normalize_ts_pair(&mut e0, &mut f0)?;
    coerce_numeric_pair(&mut e0, &mut f0)?;
    let mut combined = compare(&e0, &f0, CmpOp::Eq)?;

    for item in &list[1..] {
        let item_col = evaluate(item, batch, schema, params)?;
        let mut e = expr_col.clone();
        let mut it = item_col;
        normalize_ts_pair(&mut e, &mut it)?;
        coerce_numeric_pair(&mut e, &mut it)?;
        let cmp_result = compare(&e, &it, CmpOp::Eq)?;
        combined = bool_or(&combined, &cmp_result)?;
    }

    if negated {
        bool_not(&combined)
    } else {
        Ok(combined)
    }
}

// ---------------------------------------------------------------------------
// BETWEEN
// ---------------------------------------------------------------------------

fn evaluate_between(
    expr: &BoundExpr,
    low: &BoundExpr,
    high: &BoundExpr,
    negated: bool,
    batch: &DataBatch,
    schema: &[LogicalColumn],
    params: &[ScalarValue],
) -> Result<Column> {
    let expr_col = evaluate(expr, batch, schema, params)?;
    let low_col = evaluate(low, batch, schema, params)?;
    let high_col = evaluate(high, batch, schema, params)?;

    // Cross-precision timestamp normalization (B5) per comparison.
    let mut e_lo = expr_col.clone();
    let mut lo = low_col.clone();
    normalize_ts_pair(&mut e_lo, &mut lo)?;
    let gte_low = compare(&e_lo, &lo, CmpOp::GtEq)?;
    let mut e_hi = expr_col.clone();
    let mut hi = high_col.clone();
    normalize_ts_pair(&mut e_hi, &mut hi)?;
    let lte_high = compare(&e_hi, &hi, CmpOp::LtEq)?;
    let result = bool_and(&gte_low, &lte_high)?;

    if negated {
        bool_not(&result)
    } else {
        Ok(result)
    }
}

// ---------------------------------------------------------------------------
// CASE expression
// ---------------------------------------------------------------------------

/// Evaluates a CASE, merging the branches into the single result type the
/// binder computed for the whole expression.
///
/// Every branch is cast to that type before it is merged. Branches routinely
/// differ in physical type even when they agree logically, as `THEN price
/// ELSE 0` does, and merging those columns as they came would push a value
/// of one variant into a buffer of another.
fn evaluate_case(
    operand: Option<&BoundExpr>,
    conditions: &[BoundWhen],
    else_result: Option<&BoundExpr>,
    result_type: TypeId,
    batch: &DataBatch,
    schema: &[LogicalColumn],
    params: &[ScalarValue],
) -> Result<Column> {
    let num_rows = batch.num_rows;

    // Start with else branch or null, in the type the whole CASE produces
    let mut result = if let Some(else_expr) = else_result {
        let col = evaluate(else_expr, batch, schema, params)?;
        coerce_case_branch(col, result_type)?
    } else {
        Column::null_column(result_type, num_rows)
    };

    let operand_col = match operand {
        Some(op) => Some(evaluate(op, batch, schema, params)?),
        None => None,
    };

    // Process conditions in reverse so first match wins.
    for when in conditions.iter().rev() {
        let condition_bool = if let Some(ref op_col) = operand_col {
            let cond_col = evaluate(&when.condition, batch, schema, params)?;
            compare(op_col, &cond_col, CmpOp::Eq)?
        } else {
            evaluate(&when.condition, batch, schema, params)?
        };

        let then_col = evaluate(&when.result, batch, schema, params)?;
        let mut then_col = coerce_case_branch(then_col, result.type_id)?;
        // A decimal branch meets the running result on a common scale, the
        // wider of the two, because the type alone does not say where the
        // point sits and merging two scales would move it
        if let Some((aligned_result, aligned_then)) =
            compute::align_decimal_operands(&result, &then_col)?
        {
            result = aligned_result;
            then_col = aligned_then;
        }
        let mask = column_to_mask(&condition_bool);

        // Use typed push_from to build result without ScalarValue. Both
        // start empty because the loop below appends every row: seeding the
        // bitmap at `num_rows` left it twice the data's length, and every
        // read of it landed in the seeded half, so a CASE that produced a
        // null reported a value instead.
        let mut new_data = ColumnData::with_capacity(result.type_id, num_rows);
        let mut new_nulls = NullBitmap::empty();

        for i in 0..num_rows {
            if mask[i] {
                new_nulls.push_from(&then_col.nulls, i);
                new_data.push_from(&then_col.data, i);
            } else {
                new_nulls.push_from(&result.nulls, i);
                new_data.push_from(&result.data, i);
            }
        }

        result = Column::with_nulls(new_data, new_nulls, result.type_id);
    }

    Ok(result)
}

/// Puts one CASE branch into the result type, leaving it alone when it
/// already carries the same physical layout.
///
/// A branch whose column already matches skips the cast, so the common case
/// where every branch agrees costs one comparison.
fn coerce_case_branch(col: Column, target: TypeId) -> Result<Column> {
    if col.type_id == target {
        return Ok(col);
    }
    crate::compute::cast_column(&col, target)
}

// ---------------------------------------------------------------------------
// Scalar functions
// ---------------------------------------------------------------------------

fn evaluate_function(
    name: &str,
    args: &[BoundExpr],
    batch: &DataBatch,
    schema: &[LogicalColumn],
    params: &[ScalarValue],
) -> Result<Column> {
    match name {
        // Current transaction wall-clock time. Timestamps are stored as i64
        // microseconds since the Unix epoch (ColumnData::Int64). Broadcast one
        // value per row so it composes like a literal in assignments/filters.
        "now" | "current_timestamp" | "transaction_timestamp" | "statement_timestamp" => {
            let micros = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_micros() as i64)
                .unwrap_or(0);
            let n = batch.num_rows.max(1);
            Ok(Column::new(
                ColumnData::Int64(vec![micros; n]),
                TypeId::TimestampTz,
            ))
        }
        // Hybrid Logical Clock: a monotonic, causally-ordered timestamp.
        // Packed 48-bit physical milliseconds + 16-bit logical counter into a
        // single value, stored on the i128 HLC physical path so plain integer
        // comparison yields causal order. Generation is lock-free (one
        // AtomicU64 CAS) per the standard HLC send rule.
        "hlc_now" => {
            let h = next_hlc() as i128;
            let n = batch.num_rows.max(1);
            Ok(Column::new_ts(
                ColumnData::Int128(vec![h; n]),
                TypeId::Hlc,
                None,
            ))
        }
        // time_bucket(width, ts): floor each timestamp down to a multiple of
        // `width`, the foundational downsampling primitive for
        // `GROUP BY time_bucket(...)`. width is an integer count in the
        // column's storage unit (microseconds for p<=6, picoseconds for p>6)
        // or an INTERVAL. Floor is toward negative infinity so pre-epoch
        // timestamps bucket correctly. Result keeps the timestamp's type and
        // precision.
        // time_bucket_gapfill computes the same bucket as time_bucket (so
        // grouping is identical); the planner inserts a GapFill node above the
        // aggregate to densify absent buckets.
        "time_bucket" | "time_bucket_gapfill" => {
            if args.len() != 2 {
                return Err(ZyronError::ExecutionError(
                    "time_bucket(width, ts) takes exactly 2 arguments".to_string(),
                ));
            }
            let width_col = evaluate(&args[0], batch, schema, params)?;
            let ts = evaluate(&args[1], batch, schema, params)?;
            let is_ps = ts_col_is_ps(&ts);
            // Resolve the bucket width into the timestamp's storage unit.
            let width: i128 = match width_col.data.get_scalar(0) {
                ScalarValue::Int8(v) => v as i128,
                ScalarValue::Int16(v) => v as i128,
                ScalarValue::Int32(v) => v as i128,
                ScalarValue::Int64(v) => v as i128,
                ScalarValue::Int128(v) => v,
                ScalarValue::Interval(iv) => {
                    // Interval is i64 nanoseconds. Convert to the column unit.
                    let ns = iv.nanoseconds as i128;
                    if is_ps {
                        ns * 1000
                    } else if ns % 1000 == 0 {
                        ns / 1000
                    } else {
                        return Err(ZyronError::ExecutionError(
                            "time_bucket interval width is not a whole microsecond \
                             for a microsecond-precision column"
                                .to_string(),
                        ));
                    }
                }
                other => {
                    return Err(ZyronError::ExecutionError(format!(
                        "time_bucket width must be an integer or interval, got {other:?}"
                    )));
                }
            };
            if width <= 0 {
                return Err(ZyronError::ExecutionError(
                    "time_bucket width must be positive".to_string(),
                ));
            }
            let bucket = |v: i128| -> i128 { v.div_euclid(width) * width };
            let data = match &ts.data {
                ColumnData::Int64(v) => {
                    ColumnData::Int64(v.iter().map(|&x| bucket(x as i128) as i64).collect())
                }
                ColumnData::Int128(v) => ColumnData::Int128(v.iter().map(|&x| bucket(x)).collect()),
                _ => {
                    return Err(ZyronError::ExecutionError(
                        "time_bucket second argument must be a timestamp column".to_string(),
                    ));
                }
            };
            Ok(Column::with_nulls_ts(
                data,
                ts.nulls.clone(),
                ts.type_id,
                ts.fractional_digits,
            ))
        }
        // locf(col): last-observation-carried-forward. Replaces each NULL with
        // the most recent preceding non-NULL value in the batch (rows are
        // expected to arrive in the desired order, e.g. after ORDER BY). A
        // leading run of NULLs stays NULL.
        "locf" => {
            if args.len() != 1 {
                return Err(ZyronError::ExecutionError(
                    "locf(col) takes exactly 1 argument".to_string(),
                ));
            }
            let col = evaluate(&args[0], batch, schema, params)?;
            let n = col.len();
            let mut data = ColumnData::with_capacity(col.type_id, n);
            let mut nulls = NullBitmap::none(n);
            let mut last: Option<ScalarValue> = None;
            for i in 0..n {
                if col.is_null(i) {
                    match &last {
                        Some(v) => data.push_scalar(v),
                        None => {
                            data.push_scalar(&ScalarValue::Null);
                            nulls.set_null(i);
                        }
                    }
                } else {
                    let v = col.data.get_scalar(i);
                    data.push_scalar(&v);
                    last = Some(v);
                }
            }
            Ok(Column::with_nulls_ts(
                data,
                nulls,
                col.type_id,
                col.fractional_digits,
            ))
        }
        // interpolate(col): linear interpolation. Each NULL between two
        // non-NULL values is filled on the straight line between the nearest
        // preceding and following non-NULL (by row index). Leading or trailing
        // NULL runs (no bracketing pair) stay NULL. Numeric columns only.
        "interpolate" => {
            if args.len() != 1 {
                return Err(ZyronError::ExecutionError(
                    "interpolate(col) takes exactly 1 argument".to_string(),
                ));
            }
            let col = evaluate(&args[0], batch, schema, params)?;
            let n = col.len();
            // Pull values as f64 with null mask.
            let as_f64 = |i: usize| -> Option<f64> {
                if col.is_null(i) {
                    return None;
                }
                match col.data.get_scalar(i) {
                    ScalarValue::Int8(v) => Some(v as f64),
                    ScalarValue::Int16(v) => Some(v as f64),
                    ScalarValue::Int32(v) => Some(v as f64),
                    ScalarValue::Int64(v) => Some(v as f64),
                    ScalarValue::Int128(v) => Some(v as f64),
                    ScalarValue::Float32(v) => Some(v as f64),
                    ScalarValue::Float64(v) => Some(v),
                    _ => None,
                }
            };
            let known: Vec<(usize, f64)> =
                (0..n).filter_map(|i| as_f64(i).map(|v| (i, v))).collect();
            let mut data = ColumnData::with_capacity(col.type_id, n);
            let mut nulls = NullBitmap::none(n);
            let int_like = matches!(
                col.type_id,
                TypeId::Int8
                    | TypeId::Int16
                    | TypeId::Int32
                    | TypeId::Int64
                    | TypeId::Int128
                    | TypeId::Timestamp
                    | TypeId::TimestampTz
            );
            for i in 0..n {
                if !col.is_null(i) {
                    data.push_scalar(&col.data.get_scalar(i));
                    continue;
                }
                // Find bracketing known points.
                let before = known.iter().rev().find(|(k, _)| *k < i).copied();
                let after = known.iter().find(|(k, _)| *k > i).copied();
                match (before, after) {
                    (Some((i0, v0)), Some((i1, v1))) => {
                        let t = (i - i0) as f64 / (i1 - i0) as f64;
                        let v = v0 + (v1 - v0) * t;
                        let sv = if int_like {
                            ScalarValue::Int64(v.round() as i64)
                        } else {
                            ScalarValue::Float64(v)
                        };
                        // Coerce to the column's variant via push_scalar's
                        // typed path by casting through a single-row column.
                        let tmp = crate::compute::cast_column(
                            &Column::new(
                                if int_like {
                                    ColumnData::Int64(vec![match sv {
                                        ScalarValue::Int64(x) => x,
                                        _ => 0,
                                    }])
                                } else {
                                    ColumnData::Float64(vec![v])
                                },
                                if int_like {
                                    TypeId::Int64
                                } else {
                                    TypeId::Float64
                                },
                            ),
                            col.type_id,
                        )?;
                        data.push_scalar(&tmp.data.get_scalar(0));
                    }
                    _ => {
                        data.push_scalar(&ScalarValue::Null);
                        nulls.set_null(i);
                    }
                }
            }
            Ok(Column::with_nulls_ts(
                data,
                nulls,
                col.type_id,
                col.fractional_digits,
            ))
        }
        "abs" => {
            let col = evaluate(&args[0], batch, schema, params)?;
            eval_abs(&col)
        }
        "upper" => {
            let col = evaluate(&args[0], batch, schema, params)?;
            eval_string_transform(&col, |s| s.to_uppercase())
        }
        "lower" => {
            let col = evaluate(&args[0], batch, schema, params)?;
            eval_string_transform(&col, |s| s.to_lowercase())
        }
        "length" => {
            let col = evaluate(&args[0], batch, schema, params)?;
            eval_length(&col)
        }
        "coalesce" => eval_coalesce(args, batch, schema, params),
        "nullif" => {
            let a = evaluate(&args[0], batch, schema, params)?;
            let b = evaluate(&args[1], batch, schema, params)?;
            eval_nullif(&a, &b)
        }
        "char_length" | "character_length" => {
            let col = evaluate(&args[0], batch, schema, params)?;
            eval_char_length(&col)
        }
        "octet_length" => {
            let col = evaluate(&args[0], batch, schema, params)?;
            eval_octet_length(&col)
        }
        "ceil" | "ceiling" => {
            let col = evaluate(&args[0], batch, schema, params)?;
            eval_float_unary(&col, f64::ceil, f32::ceil)
        }
        "floor" => {
            let col = evaluate(&args[0], batch, schema, params)?;
            eval_float_unary(&col, f64::floor, f32::floor)
        }
        // round and trunc accept an optional per row digit count, positive
        // digits keep fractional places, negative digits zero places left of
        // the decimal point, integer inputs pass through except negative
        // digits which round the integer itself
        "round" => eval_round_trunc(args, batch, schema, params, true),
        "trunc" | "truncate" => eval_round_trunc(args, batch, schema, params, false),
        "trim" => eval_trim(args, batch, schema, params, TrimSide::Both),
        "ltrim" => eval_trim(args, batch, schema, params, TrimSide::Leading),
        "rtrim" => eval_trim(args, batch, schema, params, TrimSide::Trailing),
        "substring" | "substr" => eval_substring(args, batch, schema, params),
        "replace" => {
            if args.len() != 3 {
                return Err(ZyronError::ExecutionError(
                    "replace(string, from, to) takes exactly 3 arguments".to_string(),
                ));
            }
            let s = evaluate(&args[0], batch, schema, params)?;
            let from = evaluate(&args[1], batch, schema, params)?;
            let to = evaluate(&args[2], batch, schema, params)?;
            eval_replace(&s, &from, &to)
        }
        "concat" => eval_concat(args, batch, schema, params),
        "greatest" => eval_greatest_least(args, batch, schema, params, true),
        "least" => eval_greatest_least(args, batch, schema, params, false),
        // current_date is days since the Unix epoch, current_time is
        // microseconds since UTC midnight, both broadcast per row so they
        // compose like literals
        "current_date" => {
            let secs = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_secs() as i64)
                .unwrap_or(0);
            let days = secs.div_euclid(86_400) as i32;
            let n = batch.num_rows.max(1);
            Ok(Column::new(ColumnData::Int32(vec![days; n]), TypeId::Date))
        }
        "current_time" => {
            let micros = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_micros() as i64)
                .unwrap_or(0);
            let of_day = micros.rem_euclid(86_400_000_000);
            let n = batch.num_rows.max(1);
            Ok(Column::new(
                ColumnData::Int64(vec![of_day; n]),
                TypeId::Time,
            ))
        }
        "extract" | "date_part" => eval_extract(args, batch, schema, params),
        "array" => eval_array(args, batch, schema, params),
        "array_subscript" => eval_array_subscript(args, batch, schema, params),
        // Search predicates evaluated row by row. The planner routes these to
        // an index operator when one covers the table and the read is of the
        // current state; otherwise the storage scan evaluates them here, so
        // an unindexed table and a time-travel read answer the same question
        "match_against" => eval_match_against(args, batch, schema, params),
        "vector_distance_cosine" | "vector_distance_l2" | "vector_distance_dot" => {
            eval_vector_distance(name, args, batch, schema, params)
        }
        n if name_matches_ml(n) => eval_ml_scalar(n, args, batch, schema, params),
        _ => crate::types_bridge::evaluate_types_function(name, args, batch, schema, params),
    }
}

/// `ARRAY[a, b, c]`.
///
/// Every element is evaluated as a column, so an array can be built from
/// per-row values and not only from literals, then one encoded array is
/// assembled per row. Elements share a type: the first one that is not null
/// sets it and the rest are cast to it, which is what makes the encoding
/// addressable by index.
fn eval_array(
    args: &[BoundExpr],
    batch: &DataBatch,
    schema: &[LogicalColumn],
    params: &[ScalarValue],
) -> Result<Column> {
    let rows = batch.num_rows.max(1);
    if args.is_empty() {
        let empty = zyron_common::array_value::encode(TypeId::Null, &[]);
        return Ok(Column::new(
            ColumnData::Binary(vec![empty; rows]),
            TypeId::Array,
        ));
    }

    let mut columns: Vec<Column> = args
        .iter()
        .map(|a| evaluate(a, batch, schema, params))
        .collect::<Result<Vec<_>>>()?;

    // The element type is the first one carrying values. An all-null array
    // has no type to take, and stays untyped
    let element_type = columns
        .iter()
        .map(|c| c.type_id)
        .find(|t| *t != TypeId::Null)
        .unwrap_or(TypeId::Null);
    for column in &mut columns {
        if column.type_id != element_type && column.type_id != TypeId::Null {
            *column = crate::compute::cast_column(column, element_type)?;
        }
    }

    let width = element_type.fixed_size().unwrap_or(0);
    let mut out = Vec::with_capacity(rows);
    let mut payloads: Vec<Option<Vec<u8>>> = Vec::with_capacity(columns.len());
    for row in 0..rows {
        payloads.clear();
        for column in &columns {
            let scalar = if row < column.len() {
                column.get_scalar(row)
            } else {
                ScalarValue::Null
            };
            payloads.push(match scalar {
                ScalarValue::Null => None,
                other => Some(crate::batch::encode_scalar_value(
                    element_type,
                    &other,
                    width,
                )),
            });
        }
        let borrowed: Vec<Option<&[u8]>> = payloads.iter().map(|p| p.as_deref()).collect();
        out.push(zyron_common::array_value::encode(element_type, &borrowed));
    }
    Ok(Column::new(ColumnData::Binary(out), TypeId::Array))
}

/// `array[index]`, one-based the way SQL subscripts are.
///
/// An index outside the array reads as NULL rather than failing, matching how
/// a missing element reads everywhere else.
fn eval_array_subscript(
    args: &[BoundExpr],
    batch: &DataBatch,
    schema: &[LogicalColumn],
    params: &[ScalarValue],
) -> Result<Column> {
    if args.len() != 2 {
        return Err(ZyronError::ExecutionError(
            "array subscript takes an array and an index".to_string(),
        ));
    }
    let arrays = evaluate(&args[0], batch, schema, params)?;
    let indexes = evaluate(&args[1], batch, schema, params)?;
    let rows = batch.num_rows.max(1);

    // The element type comes from the encoded value rather than the plan, so
    // a column whose element type was not known at bind time still decodes
    let element_type = (0..rows.min(arrays.len()))
        .find_map(|row| match arrays.get_scalar(row) {
            ScalarValue::Binary(bytes) => {
                zyron_common::ArrayView::parse(&bytes).map(|v| v.element_type())
            }
            _ => None,
        })
        .unwrap_or(TypeId::Null);

    let mut data = ColumnData::with_capacity(element_type, rows);
    let mut nulls = crate::column::NullBitmap::empty();
    for row in 0..rows {
        let scalar = subscript_one(&arrays, &indexes, row, element_type);
        nulls.push(scalar.is_null());
        data.push_scalar(&scalar);
    }
    Ok(Column::with_nulls(data, nulls, element_type))
}

/// Reads one row's subscripted element, or NULL when the array, the index or
/// the position does not resolve.
fn subscript_one(
    arrays: &Column,
    indexes: &Column,
    row: usize,
    element_type: TypeId,
) -> ScalarValue {
    if row >= arrays.len() || arrays.nulls.is_null(row) {
        return ScalarValue::Null;
    }
    let ScalarValue::Binary(bytes) = arrays.get_scalar(row) else {
        return ScalarValue::Null;
    };
    let Some(view) = zyron_common::ArrayView::parse(&bytes) else {
        return ScalarValue::Null;
    };
    let index_row = if row < indexes.len() { row } else { 0 };
    let one_based = match indexes.get_scalar(index_row) {
        ScalarValue::Int8(v) => v as i64,
        ScalarValue::Int16(v) => v as i64,
        ScalarValue::Int32(v) => v as i64,
        ScalarValue::Int64(v) => v,
        ScalarValue::UInt8(v) => v as i64,
        ScalarValue::UInt16(v) => v as i64,
        ScalarValue::UInt32(v) => v as i64,
        ScalarValue::UInt64(v) => v as i64,
        _ => return ScalarValue::Null,
    };
    if one_based < 1 {
        return ScalarValue::Null;
    }
    match view.get((one_based - 1) as usize) {
        Some(Some(payload)) => {
            if element_type.fixed_size().unwrap_or(0) > 0 {
                crate::batch::decode_fixed_scalar(element_type, payload)
            } else {
                crate::batch::decode_varlen_scalar(element_type, payload)
            }
        }
        _ => ScalarValue::Null,
    }
}

/// Row-wise `MATCH (cols) AGAINST ('query')`.
///
/// Returns 1.0 for a matching row and 0.0 otherwise. The indexed path returns
/// a BM25 score, which needs corpus statistics a single row does not carry, so
/// this reports membership rather than inventing a rank. Row membership is
/// identical either way, which is what a predicate depends on.
fn eval_match_against(
    args: &[BoundExpr],
    batch: &DataBatch,
    schema: &[LogicalColumn],
    params: &[ScalarValue],
) -> Result<Column> {
    let Some((query_arg, column_args)) = args.split_last() else {
        return Err(ZyronError::ExecutionError(
            "match_against takes at least one column and a query".to_string(),
        ));
    };
    let query_column = evaluate(query_arg, batch, schema, params)?;
    let query_text = match query_column.get_scalar(0) {
        ScalarValue::Utf8(s) => s,
        other => {
            return Err(ZyronError::ExecutionError(format!(
                "match_against expects a string query, got {other:?}"
            )));
        }
    };
    let query = zyron_search::FtsQueryParser::parse(&query_text)?;
    let analyzer = zyron_search::SimpleAnalyzer;

    let text_columns: Vec<Column> = column_args
        .iter()
        .map(|a| evaluate(a, batch, schema, params))
        .collect::<Result<Vec<_>>>()?;

    let mut scores = Vec::with_capacity(batch.num_rows);
    let mut document = String::with_capacity(256);
    for row in 0..batch.num_rows {
        // One document per row, the matched columns joined the way the index
        // concatenates them at write time
        document.clear();
        for col in &text_columns {
            if col.nulls.is_null(row) {
                continue;
            }
            if let ColumnData::Utf8(values) = &col.data
                && let Some(s) = values.get(row)
            {
                if !document.is_empty() {
                    document.push(' ');
                }
                document.push_str(s);
            }
        }
        let tokens = zyron_search::Analyzer::analyze(&analyzer, &document);
        let terms: Vec<&str> = tokens.iter().map(|t| t.term.as_str()).collect();
        scores.push(if query.matches_terms(&terms, &analyzer) {
            1.0
        } else {
            0.0
        });
    }
    Ok(Column::new(ColumnData::Float64(scores), TypeId::Float64))
}

/// Row-wise vector distance between a vector column and a query vector.
/// Same metrics the vector index scores with, so a scan and an index search
/// order rows identically.
fn eval_vector_distance(
    name: &str,
    args: &[BoundExpr],
    batch: &DataBatch,
    schema: &[LogicalColumn],
    params: &[ScalarValue],
) -> Result<Column> {
    if args.len() != 2 {
        return Err(ZyronError::ExecutionError(format!(
            "{name} takes exactly two vectors"
        )));
    }
    let left = vector_side(&args[0], batch, schema, params)?;
    let right = vector_side(&args[1], batch, schema, params)?;
    let mut out = Vec::with_capacity(batch.num_rows);
    let mut lhs: Vec<f32> = Vec::new();
    let mut rhs: Vec<f32> = Vec::new();
    for row in 0..batch.num_rows {
        let (Some(a), Some(b)) = (left.at(row, &mut lhs), right.at(row, &mut rhs)) else {
            out.push(f64::NAN);
            continue;
        };
        if a.len() != b.len() {
            return Err(ZyronError::ExecutionError(format!(
                "{name} needs equal dimensions, got {} and {}",
                a.len(),
                b.len()
            )));
        }
        let d = match name {
            "vector_distance_cosine" => {
                let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
                let na: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
                let nb: f32 = b.iter().map(|y| y * y).sum::<f32>().sqrt();
                if na == 0.0 || nb == 0.0 {
                    1.0
                } else {
                    1.0 - (dot / (na * nb)) as f64
                }
            }
            "vector_distance_l2" => a
                .iter()
                .zip(b)
                .map(|(x, y)| ((x - y) as f64).powi(2))
                .sum::<f64>()
                .sqrt(),
            // Dot product is a similarity, negated so a smaller value is a
            // nearer row exactly as the other two metrics read
            _ => -(a.iter().zip(b).map(|(x, y)| (x * y) as f64).sum::<f64>()),
        };
        out.push(d);
    }
    Ok(Column::new(ColumnData::Float64(out), TypeId::Float64))
}

/// One side of a distance call: either the same vector for every row, which
/// is how a query vector is written, or one vector per row read out of a
/// vector column's raw f32 bytes.
enum VectorSide {
    Constant(Vec<f32>),
    PerRow(Column),
}

impl VectorSide {
    /// The vector for one row. Per-row values are decoded into `scratch`, so
    /// the loop over rows allocates once rather than per row.
    fn at<'a>(&'a self, row: usize, scratch: &'a mut Vec<f32>) -> Option<&'a [f32]> {
        match self {
            VectorSide::Constant(v) => Some(v.as_slice()),
            VectorSide::PerRow(col) => {
                if col.nulls.is_null(row) {
                    return None;
                }
                let ColumnData::Binary(blobs) = &col.data else {
                    return None;
                };
                let bytes = blobs.get(row)?;
                if bytes.is_empty() || bytes.len() % 4 != 0 {
                    return None;
                }
                scratch.clear();
                scratch.extend(
                    bytes
                        .chunks_exact(4)
                        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])),
                );
                Some(scratch.as_slice())
            }
        }
    }
}

/// Reads one argument of a distance call. An array constructor of numeric
/// literals is the query vector and folds to a constant; anything else is
/// evaluated as a column.
fn vector_side(
    arg: &BoundExpr,
    batch: &DataBatch,
    schema: &[LogicalColumn],
    params: &[ScalarValue],
) -> Result<VectorSide> {
    let inner = match arg {
        BoundExpr::Nested(e) => e.as_ref(),
        other => other,
    };
    if let BoundExpr::Function { name, args, .. } = inner
        && name == "array"
        && !args.is_empty()
    {
        let mut values = Vec::with_capacity(args.len());
        for a in args {
            let column = evaluate(a, batch, schema, params)?;
            let v = match column.get_scalar(0) {
                ScalarValue::Float32(f) => f as f64,
                ScalarValue::Float64(f) => f,
                ScalarValue::Int32(i) => i as f64,
                ScalarValue::Int64(i) => i as f64,
                other => {
                    return Err(ZyronError::ExecutionError(format!(
                        "a query vector takes numeric elements, got {other:?}"
                    )));
                }
            };
            values.push(v as f32);
        }
        return Ok(VectorSide::Constant(values));
    }
    Ok(VectorSide::PerRow(evaluate(arg, batch, schema, params)?))
}

fn name_matches_ml(name: &str) -> bool {
    matches!(
        name.to_ascii_uppercase().as_str(),
        "PREDICT"
            | "ATE"
            | "ATT"
            | "PROPENSITY_SCORE"
            | "DIFF_IN_DIFF"
            | "TREND"
            | "PSI"
            | "KS_TEST"
    )
}

fn eval_ml_scalar(
    name: &str,
    args: &[BoundExpr],
    batch: &DataBatch,
    schema: &[LogicalColumn],
    params: &[ScalarValue],
) -> Result<Column> {
    let upper = name.to_ascii_uppercase();
    match upper.as_str() {
        "PREDICT" => eval_predict(args, batch, schema, params),
        "ATE" => eval_ate(args, batch, schema, params),
        "ATT" => eval_att(args, batch, schema, params),
        "PROPENSITY_SCORE" => eval_propensity(args, batch, schema, params),
        "DIFF_IN_DIFF" => eval_did(args, batch, schema, params),
        "TREND" => eval_trend(args, batch, schema, params),
        "PSI" => eval_psi(args, batch, schema, params),
        "KS_TEST" => eval_ks(args, batch, schema, params),
        _ => Err(ZyronError::ExecutionError(format!(
            "ML scalar '{}' not implemented",
            name
        ))),
    }
}

/// PREDICT('model_name', col1, col2, ...) returns prediction per row
fn eval_predict(
    args: &[BoundExpr],
    batch: &DataBatch,
    schema: &[LogicalColumn],
    params: &[ScalarValue],
) -> Result<Column> {
    if args.is_empty() {
        return Err(ZyronError::ExecutionError(
            "PREDICT requires model name and feature args".to_string(),
        ));
    }
    let model_name = literal_string(&args[0]).ok_or_else(|| {
        ZyronError::ExecutionError("PREDICT first argument must be model name literal".to_string())
    })?;
    let handle = zyron_analytics::InferenceHandle::resolve(&model_name).ok_or_else(|| {
        ZyronError::ExecutionError(format!("model '{}' not found in cache", model_name))
    })?;
    let feature_cols: Vec<Column> = args[1..]
        .iter()
        .map(|a| evaluate(a, batch, schema, params))
        .collect::<Result<Vec<_>>>()?;
    let p = handle.model.featureColumns.len();
    if feature_cols.len() != p {
        return Err(ZyronError::ExecutionError(format!(
            "PREDICT got {} feature columns, model expects {}",
            feature_cols.len(),
            p
        )));
    }
    let n = batch.num_rows;
    let mut data = ColumnData::with_capacity(TypeId::Float64, n);
    let mut nulls = NullBitmap::empty();
    let mut row = vec![0.0f64; p];
    for i in 0..n {
        let mut any_null = false;
        for (j, col) in feature_cols.iter().enumerate() {
            if col.is_null(i) {
                any_null = true;
                break;
            }
            row[j] = scalar_to_f64(&col.data.get_scalar(i));
        }
        if any_null {
            nulls.push(true);
            data.push_default();
        } else {
            let v = handle.predictOne(&row);
            nulls.push(false);
            data.push_scalar(&ScalarValue::Float64(v));
        }
    }
    Ok(Column::with_nulls(data, nulls, TypeId::Float64))
}

fn eval_ate(
    args: &[BoundExpr],
    batch: &DataBatch,
    schema: &[LogicalColumn],
    params: &[ScalarValue],
) -> Result<Column> {
    let (outcome, treatment, covariates, p) = collect_causal_inputs(args, batch, schema, params)?;
    let est = zyron_analytics::ate(&outcome, &treatment, &covariates, p)?;
    broadcast_f64(est, batch.num_rows)
}

fn eval_att(
    args: &[BoundExpr],
    batch: &DataBatch,
    schema: &[LogicalColumn],
    params: &[ScalarValue],
) -> Result<Column> {
    let (outcome, treatment, covariates, p) = collect_causal_inputs(args, batch, schema, params)?;
    let est = zyron_analytics::att(&outcome, &treatment, &covariates, p)?;
    broadcast_f64(est, batch.num_rows)
}

fn eval_propensity(
    args: &[BoundExpr],
    batch: &DataBatch,
    schema: &[LogicalColumn],
    params: &[ScalarValue],
) -> Result<Column> {
    if args.len() < 2 {
        return Err(ZyronError::ExecutionError(
            "PROPENSITY_SCORE requires (treatment, covariate1, ...)".to_string(),
        ));
    }
    let treatment_col = evaluate(&args[0], batch, schema, params)?;
    let p = args.len() - 1;
    let n = batch.num_rows;
    let mut treatment = Vec::with_capacity(n);
    for i in 0..n {
        treatment.push(scalar_to_f64(&treatment_col.data.get_scalar(i)));
    }
    let mut covariates: Vec<f64> = Vec::with_capacity(n * p);
    let cov_cols: Vec<Column> = args[1..]
        .iter()
        .map(|a| evaluate(a, batch, schema, params))
        .collect::<Result<Vec<_>>>()?;
    for i in 0..n {
        for c in &cov_cols {
            covariates.push(scalar_to_f64(&c.data.get_scalar(i)));
        }
    }
    let scores = zyron_analytics::propensityScore(&treatment, &covariates, p)?;
    let mut data = ColumnData::with_capacity(TypeId::Float64, scores.len());
    let mut nulls = NullBitmap::empty();
    for s in scores {
        nulls.push(false);
        data.push_scalar(&ScalarValue::Float64(s));
    }
    Ok(Column::with_nulls(data, nulls, TypeId::Float64))
}

fn eval_did(
    args: &[BoundExpr],
    batch: &DataBatch,
    schema: &[LogicalColumn],
    params: &[ScalarValue],
) -> Result<Column> {
    if args.len() < 3 {
        return Err(ZyronError::ExecutionError(
            "DIFF_IN_DIFF requires (outcome, treatment, post[, time])".to_string(),
        ));
    }
    let outcome = collect_column_f64(&args[0], batch, schema, params)?;
    let treatment = collect_column_f64(&args[1], batch, schema, params)?;
    let post = if args.len() >= 4 {
        collect_column_f64(&args[3], batch, schema, params)?
    } else {
        collect_column_f64(&args[2], batch, schema, params)?
    };
    let est = zyron_analytics::diffInDiff(&outcome, &treatment, &post)?;
    broadcast_f64(est, batch.num_rows)
}

fn eval_trend(
    args: &[BoundExpr],
    batch: &DataBatch,
    schema: &[LogicalColumn],
    params: &[ScalarValue],
) -> Result<Column> {
    if args.is_empty() {
        return Err(ZyronError::ExecutionError(
            "TREND requires a value column".to_string(),
        ));
    }
    let values = collect_column_f64(&args[0], batch, schema, params)?;
    let (slope, _intercept) = zyron_analytics::trend(&values);
    broadcast_f64(slope, batch.num_rows)
}

fn eval_psi(
    args: &[BoundExpr],
    batch: &DataBatch,
    schema: &[LogicalColumn],
    params: &[ScalarValue],
) -> Result<Column> {
    if args.len() != 2 {
        return Err(ZyronError::ExecutionError(
            "PSI requires (actual_histogram_col, expected_histogram_col)".to_string(),
        ));
    }
    let a = collect_column_u64(&args[0], batch, schema, params)?;
    let b = collect_column_u64(&args[1], batch, schema, params)?;
    let psi = zyron_analytics::ml::transforms::psi(&a, &b);
    broadcast_f64(psi, batch.num_rows)
}

fn eval_ks(
    args: &[BoundExpr],
    batch: &DataBatch,
    schema: &[LogicalColumn],
    params: &[ScalarValue],
) -> Result<Column> {
    if args.len() != 2 {
        return Err(ZyronError::ExecutionError(
            "KS_TEST requires (sample_a, sample_b)".to_string(),
        ));
    }
    let a = collect_column_f64(&args[0], batch, schema, params)?;
    let b = collect_column_f64(&args[1], batch, schema, params)?;
    let d = zyron_analytics::ml::transforms::ksStatistic(&a, &b);
    broadcast_f64(d, batch.num_rows)
}

fn collect_causal_inputs(
    args: &[BoundExpr],
    batch: &DataBatch,
    schema: &[LogicalColumn],
    params: &[ScalarValue],
) -> Result<(Vec<f64>, Vec<f64>, Vec<f64>, usize)> {
    if args.len() < 3 {
        return Err(ZyronError::ExecutionError(
            "causal estimator requires (outcome, treatment, covariate1, ...)".to_string(),
        ));
    }
    let outcome = collect_column_f64(&args[0], batch, schema, params)?;
    let treatment = collect_column_f64(&args[1], batch, schema, params)?;
    let p = args.len() - 2;
    let n = outcome.len();
    let mut covariates = Vec::with_capacity(n * p);
    let cov_cols: Vec<Column> = args[2..]
        .iter()
        .map(|a| evaluate(a, batch, schema, params))
        .collect::<Result<Vec<_>>>()?;
    for i in 0..n {
        for c in &cov_cols {
            covariates.push(scalar_to_f64(&c.data.get_scalar(i)));
        }
    }
    Ok((outcome, treatment, covariates, p))
}

fn collect_column_f64(
    expr: &BoundExpr,
    batch: &DataBatch,
    schema: &[LogicalColumn],
    params: &[ScalarValue],
) -> Result<Vec<f64>> {
    let col = evaluate(expr, batch, schema, params)?;
    let n = batch.num_rows;
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        if col.is_null(i) {
            out.push(f64::NAN);
        } else {
            out.push(scalar_to_f64(&col.data.get_scalar(i)));
        }
    }
    Ok(out)
}

fn collect_column_u64(
    expr: &BoundExpr,
    batch: &DataBatch,
    schema: &[LogicalColumn],
    params: &[ScalarValue],
) -> Result<Vec<u64>> {
    let col = evaluate(expr, batch, schema, params)?;
    let n = batch.num_rows;
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        if col.is_null(i) {
            out.push(0);
        } else {
            out.push(scalar_to_f64(&col.data.get_scalar(i)).max(0.0) as u64);
        }
    }
    Ok(out)
}

fn broadcast_f64(value: f64, n: usize) -> Result<Column> {
    let mut data = ColumnData::with_capacity(TypeId::Float64, n);
    let mut nulls = NullBitmap::empty();
    for _ in 0..n {
        nulls.push(false);
        data.push_scalar(&ScalarValue::Float64(value));
    }
    Ok(Column::with_nulls(data, nulls, TypeId::Float64))
}

fn scalar_to_f64(v: &ScalarValue) -> f64 {
    match v {
        ScalarValue::Null => f64::NAN,
        ScalarValue::Boolean(b) => {
            if *b {
                1.0
            } else {
                0.0
            }
        }
        ScalarValue::Int8(x) => *x as f64,
        ScalarValue::Int16(x) => *x as f64,
        ScalarValue::Int32(x) => *x as f64,
        ScalarValue::Int64(x) => *x as f64,
        ScalarValue::Int128(x) => *x as f64,
        ScalarValue::UInt8(x) => *x as f64,
        ScalarValue::UInt16(x) => *x as f64,
        ScalarValue::UInt32(x) => *x as f64,
        ScalarValue::UInt64(x) => *x as f64,
        ScalarValue::Float32(f) => *f as f64,
        ScalarValue::Float64(f) => *f,
        _ => 0.0,
    }
}

fn literal_string(expr: &BoundExpr) -> Option<String> {
    use zyron_parser::ast::LiteralValue;
    match expr {
        BoundExpr::Literal {
            value: LiteralValue::String(s),
            ..
        } => Some(s.clone()),
        _ => None,
    }
}

fn eval_abs(col: &Column) -> Result<Column> {
    // Typed fast paths for common numeric types.
    match &col.data {
        ColumnData::Int64(v) => {
            let result: Vec<i64> = v.iter().map(|x| x.wrapping_abs()).collect();
            return Ok(Column::with_nulls(
                ColumnData::Int64(result),
                col.nulls.clone(),
                col.type_id,
            ));
        }
        ColumnData::Float64(v) => {
            let result: Vec<f64> = v.iter().map(|x| x.abs()).collect();
            return Ok(Column::with_nulls(
                ColumnData::Float64(result),
                col.nulls.clone(),
                col.type_id,
            ));
        }
        ColumnData::Int32(v) => {
            let result: Vec<i32> = v.iter().map(|x| x.wrapping_abs()).collect();
            return Ok(Column::with_nulls(
                ColumnData::Int32(result),
                col.nulls.clone(),
                col.type_id,
            ));
        }
        ColumnData::Float32(v) => {
            let result: Vec<f32> = v.iter().map(|x| x.abs()).collect();
            return Ok(Column::with_nulls(
                ColumnData::Float32(result),
                col.nulls.clone(),
                col.type_id,
            ));
        }
        _ => {}
    }

    // Fallback for rare types.
    let len = col.len();
    let mut data = ColumnData::with_capacity(col.type_id, len);
    let mut nulls = NullBitmap::empty();

    for i in 0..len {
        if col.is_null(i) {
            nulls.push(true);
            data.push_default();
            continue;
        }
        nulls.push(false);
        let scalar = col.data.get_scalar(i);
        let abs_val = match scalar {
            ScalarValue::Int64(v) => ScalarValue::Int64(v.wrapping_abs()),
            ScalarValue::Float64(v) => ScalarValue::Float64(v.abs()),
            ScalarValue::Int32(v) => ScalarValue::Int32(v.wrapping_abs()),
            ScalarValue::Float32(v) => ScalarValue::Float32(v.abs()),
            other => other,
        };
        data.push_scalar(&abs_val);
    }

    Ok(Column::with_nulls(data, nulls, col.type_id))
}

fn eval_string_transform(col: &Column, transform: fn(&str) -> String) -> Result<Column> {
    let strings = match &col.data {
        ColumnData::Utf8(v) => v,
        _ => {
            return Err(ZyronError::ExecutionError(
                "string function requires string column".to_string(),
            ));
        }
    };
    let result: Vec<String> = strings
        .iter()
        .enumerate()
        .map(|(i, s)| {
            if col.is_null(i) {
                String::new()
            } else {
                transform(s)
            }
        })
        .collect();
    Ok(Column::with_nulls(
        ColumnData::Utf8(result),
        col.nulls.clone(),
        TypeId::Text,
    ))
}

fn eval_length(col: &Column) -> Result<Column> {
    let strings = match &col.data {
        ColumnData::Utf8(v) => v,
        _ => {
            return Err(ZyronError::ExecutionError(
                "length() requires string column".to_string(),
            ));
        }
    };
    let result: Vec<i64> = strings
        .iter()
        .enumerate()
        .map(|(i, s)| {
            if col.is_null(i) {
                0
            } else {
                s.chars().count() as i64
            }
        })
        .collect();
    Ok(Column::with_nulls(
        ColumnData::Int64(result),
        col.nulls.clone(),
        TypeId::Int64,
    ))
}

fn eval_coalesce(
    args: &[BoundExpr],
    batch: &DataBatch,
    schema: &[LogicalColumn],
    params: &[ScalarValue],
) -> Result<Column> {
    let num_rows = batch.num_rows;
    let last_idx = args.len() - 1;
    let mut result = evaluate(&args[last_idx], batch, schema, params)?;

    for arg in args[..last_idx].iter().rev() {
        let arg_col = evaluate(arg, batch, schema, params)?;
        let mut new_data = ColumnData::with_capacity(result.type_id, num_rows);
        let mut new_nulls = NullBitmap::none(num_rows);

        for i in 0..num_rows {
            if !arg_col.is_null(i) {
                new_nulls.push_from(&arg_col.nulls, i);
                new_data.push_from(&arg_col.data, i);
            } else {
                new_nulls.push_from(&result.nulls, i);
                new_data.push_from(&result.data, i);
            }
        }

        result = Column::with_nulls(new_data, new_nulls, result.type_id);
    }

    Ok(result)
}

fn eval_nullif(a: &Column, b: &Column) -> Result<Column> {
    let len = a.len();
    let mut data = ColumnData::with_capacity(a.type_id, len);
    let mut nulls = NullBitmap::none(len);

    for i in 0..len {
        if a.is_null(i) {
            nulls.push(true);
            data.push_default();
        } else if !b.is_null(i) && values_equal_at(&a.data, i, &b.data, i) {
            nulls.push(true);
            data.push_default();
        } else {
            nulls.push(false);
            data.push_from(&a.data, i);
        }
    }

    Ok(Column::with_nulls(data, nulls, a.type_id))
}

/// Typed equality check for two values at given indices across ColumnData instances.
#[inline]
fn values_equal_at(a: &ColumnData, a_idx: usize, b: &ColumnData, b_idx: usize) -> bool {
    match (a, b) {
        (ColumnData::Boolean(va), ColumnData::Boolean(vb)) => va[a_idx] == vb[b_idx],
        (ColumnData::Int8(va), ColumnData::Int8(vb)) => va[a_idx] == vb[b_idx],
        (ColumnData::Int16(va), ColumnData::Int16(vb)) => va[a_idx] == vb[b_idx],
        (ColumnData::Int32(va), ColumnData::Int32(vb)) => va[a_idx] == vb[b_idx],
        (ColumnData::Int64(va), ColumnData::Int64(vb)) => va[a_idx] == vb[b_idx],
        (ColumnData::Int128(va), ColumnData::Int128(vb)) => va[a_idx] == vb[b_idx],
        (ColumnData::UInt8(va), ColumnData::UInt8(vb)) => va[a_idx] == vb[b_idx],
        (ColumnData::UInt16(va), ColumnData::UInt16(vb)) => va[a_idx] == vb[b_idx],
        (ColumnData::UInt32(va), ColumnData::UInt32(vb)) => va[a_idx] == vb[b_idx],
        (ColumnData::UInt64(va), ColumnData::UInt64(vb)) => va[a_idx] == vb[b_idx],
        (ColumnData::Float32(va), ColumnData::Float32(vb)) => {
            va[a_idx].to_bits() == vb[b_idx].to_bits()
        }
        (ColumnData::Float64(va), ColumnData::Float64(vb)) => {
            va[a_idx].to_bits() == vb[b_idx].to_bits()
        }
        (ColumnData::Utf8(va), ColumnData::Utf8(vb)) => va[a_idx] == vb[b_idx],
        (ColumnData::Binary(va), ColumnData::Binary(vb)) => va[a_idx] == vb[b_idx],
        (ColumnData::FixedBinary16(va), ColumnData::FixedBinary16(vb)) => va[a_idx] == vb[b_idx],
        _ => a.get_scalar(a_idx) == b.get_scalar(b_idx),
    }
}

/// Typed less-than for two values at given indices, same variant on both
/// sides, NaN orders greater than every number so folds are total
#[inline]
fn values_less_at(a: &ColumnData, a_idx: usize, b: &ColumnData, b_idx: usize) -> bool {
    match (a, b) {
        (ColumnData::Boolean(va), ColumnData::Boolean(vb)) => va[a_idx] < vb[b_idx],
        (ColumnData::Int8(va), ColumnData::Int8(vb)) => va[a_idx] < vb[b_idx],
        (ColumnData::Int16(va), ColumnData::Int16(vb)) => va[a_idx] < vb[b_idx],
        (ColumnData::Int32(va), ColumnData::Int32(vb)) => va[a_idx] < vb[b_idx],
        (ColumnData::Int64(va), ColumnData::Int64(vb)) => va[a_idx] < vb[b_idx],
        (ColumnData::Int128(va), ColumnData::Int128(vb)) => va[a_idx] < vb[b_idx],
        (ColumnData::UInt8(va), ColumnData::UInt8(vb)) => va[a_idx] < vb[b_idx],
        (ColumnData::UInt16(va), ColumnData::UInt16(vb)) => va[a_idx] < vb[b_idx],
        (ColumnData::UInt32(va), ColumnData::UInt32(vb)) => va[a_idx] < vb[b_idx],
        (ColumnData::UInt64(va), ColumnData::UInt64(vb)) => va[a_idx] < vb[b_idx],
        (ColumnData::Float32(va), ColumnData::Float32(vb)) => {
            match va[a_idx].partial_cmp(&vb[b_idx]) {
                Some(o) => o == std::cmp::Ordering::Less,
                None => !va[a_idx].is_nan() && vb[b_idx].is_nan(),
            }
        }
        (ColumnData::Float64(va), ColumnData::Float64(vb)) => {
            match va[a_idx].partial_cmp(&vb[b_idx]) {
                Some(o) => o == std::cmp::Ordering::Less,
                None => !va[a_idx].is_nan() && vb[b_idx].is_nan(),
            }
        }
        (ColumnData::Utf8(va), ColumnData::Utf8(vb)) => va[a_idx] < vb[b_idx],
        (ColumnData::Binary(va), ColumnData::Binary(vb)) => va[a_idx] < vb[b_idx],
        (ColumnData::FixedBinary16(va), ColumnData::FixedBinary16(vb)) => va[a_idx] < vb[b_idx],
        _ => false,
    }
}

fn utf8_or_err(col: &Column, fname: &str) -> Result<()> {
    match &col.data {
        ColumnData::Utf8(_) => Ok(()),
        _ => Err(ZyronError::ExecutionError(format!(
            "{fname}() requires a string argument"
        ))),
    }
}

fn eval_char_length(col: &Column) -> Result<Column> {
    utf8_or_err(col, "char_length")?;
    let strings = match &col.data {
        ColumnData::Utf8(v) => v,
        _ => unreachable!(),
    };
    let result: Vec<i64> = strings
        .iter()
        .enumerate()
        .map(|(i, s)| {
            if col.is_null(i) {
                0
            } else {
                s.chars().count() as i64
            }
        })
        .collect();
    Ok(Column::with_nulls(
        ColumnData::Int64(result),
        col.nulls.clone(),
        TypeId::Int64,
    ))
}

fn eval_octet_length(col: &Column) -> Result<Column> {
    let result: Vec<i64> = match &col.data {
        ColumnData::Utf8(v) => v
            .iter()
            .enumerate()
            .map(|(i, s)| if col.is_null(i) { 0 } else { s.len() as i64 })
            .collect(),
        ColumnData::Binary(v) => v
            .iter()
            .enumerate()
            .map(|(i, b)| if col.is_null(i) { 0 } else { b.len() as i64 })
            .collect(),
        _ => {
            return Err(ZyronError::ExecutionError(
                "octet_length() requires a string or binary argument".to_string(),
            ));
        }
    };
    Ok(Column::with_nulls(
        ColumnData::Int64(result),
        col.nulls.clone(),
        TypeId::Int64,
    ))
}

/// ceil and floor, integers pass through, floats apply the op per lane
fn eval_float_unary(col: &Column, op64: fn(f64) -> f64, op32: fn(f32) -> f32) -> Result<Column> {
    match &col.data {
        ColumnData::Float64(v) => Ok(Column::with_nulls(
            ColumnData::Float64(v.iter().map(|&x| op64(x)).collect()),
            col.nulls.clone(),
            col.type_id,
        )),
        ColumnData::Float32(v) => Ok(Column::with_nulls(
            ColumnData::Float32(v.iter().map(|&x| op32(x)).collect()),
            col.nulls.clone(),
            col.type_id,
        )),
        ColumnData::Int8(_)
        | ColumnData::Int16(_)
        | ColumnData::Int32(_)
        | ColumnData::Int64(_)
        | ColumnData::Int128(_)
        | ColumnData::UInt8(_)
        | ColumnData::UInt16(_)
        | ColumnData::UInt32(_)
        | ColumnData::UInt64(_) => Ok(col.clone()),
        _ => Err(ZyronError::ExecutionError(
            "ceil/floor requires a numeric argument".to_string(),
        )),
    }
}

/// Per row digit count for round and trunc, None marks a NULL digit row
fn digits_at(col: &Column, i: usize) -> Result<Option<i32>> {
    if col.is_null(i) {
        return Ok(None);
    }
    let d = match col.data.get_scalar(i) {
        ScalarValue::Int8(v) => v as i64,
        ScalarValue::Int16(v) => v as i64,
        ScalarValue::Int32(v) => v as i64,
        ScalarValue::Int64(v) => v,
        ScalarValue::Int128(v) => v.clamp(i32::MIN as i128, i32::MAX as i128) as i64,
        other => {
            return Err(ZyronError::ExecutionError(format!(
                "round/trunc digit count must be an integer, got {other:?}"
            )));
        }
    };
    Ok(Some(d.clamp(-400, 400) as i32))
}

/// Rounds or truncates one i128 to a multiple of 10^(-d) for d < 0,
/// half away from zero when rounding
fn int_round_trunc(v: i128, d: i32, round: bool) -> i128 {
    if d >= 0 {
        return v;
    }
    if d <= -39 {
        return 0;
    }
    let p = 10i128.pow((-d) as u32);
    if round {
        let half = p / 2;
        let adj = if v >= 0 {
            v.saturating_add(half)
        } else {
            v.saturating_sub(half)
        };
        (adj / p) * p
    } else {
        (v / p) * p
    }
}

fn float_round_trunc(v: f64, d: i32, round: bool) -> f64 {
    if d == 0 {
        return if round { v.round() } else { v.trunc() };
    }
    let scale = 10f64.powi(d.clamp(-308, 308));
    let scaled = v * scale;
    if !scaled.is_finite() {
        return v;
    }
    let r = if round {
        scaled.round()
    } else {
        scaled.trunc()
    };
    r / scale
}

fn eval_round_trunc(
    args: &[BoundExpr],
    batch: &DataBatch,
    schema: &[LogicalColumn],
    params: &[ScalarValue],
    round: bool,
) -> Result<Column> {
    if args.is_empty() || args.len() > 2 {
        return Err(ZyronError::ExecutionError(
            "round/trunc takes 1 or 2 arguments".to_string(),
        ));
    }
    let col = evaluate(&args[0], batch, schema, params)?;
    let dcol = if args.len() == 2 {
        Some(evaluate(&args[1], batch, schema, params)?)
    } else {
        None
    };
    let n = col.len();
    let digit = |i: usize| -> Result<Option<i32>> {
        match &dcol {
            Some(d) => digits_at(d, i),
            None => Ok(Some(0)),
        }
    };

    macro_rules! int_lanes {
        ($v:expr, $variant:ident, $ty:ty) => {{
            let mut out: Vec<$ty> = Vec::with_capacity(n);
            let mut nulls = NullBitmap::none(n);
            for (i, &x) in $v.iter().enumerate() {
                match digit(i)? {
                    Some(d) if !col.is_null(i) => {
                        out.push(int_round_trunc(x as i128, d, round) as $ty)
                    }
                    _ => {
                        out.push(0 as $ty);
                        nulls.set_null(i);
                    }
                }
            }
            for i in 0..n {
                if col.is_null(i) {
                    nulls.set_null(i);
                }
            }
            Ok(Column::with_nulls(
                ColumnData::$variant(out),
                nulls,
                col.type_id,
            ))
        }};
    }

    match &col.data {
        ColumnData::Float64(v) => {
            let mut out: Vec<f64> = Vec::with_capacity(n);
            let mut nulls = NullBitmap::none(n);
            for (i, &x) in v.iter().enumerate() {
                match digit(i)? {
                    Some(d) if !col.is_null(i) => out.push(float_round_trunc(x, d, round)),
                    _ => {
                        out.push(0.0);
                        nulls.set_null(i);
                    }
                }
            }
            for i in 0..n {
                if col.is_null(i) {
                    nulls.set_null(i);
                }
            }
            Ok(Column::with_nulls(
                ColumnData::Float64(out),
                nulls,
                col.type_id,
            ))
        }
        ColumnData::Float32(v) => {
            let mut out: Vec<f32> = Vec::with_capacity(n);
            let mut nulls = NullBitmap::none(n);
            for (i, &x) in v.iter().enumerate() {
                match digit(i)? {
                    Some(d) if !col.is_null(i) => {
                        out.push(float_round_trunc(x as f64, d, round) as f32)
                    }
                    _ => {
                        out.push(0.0);
                        nulls.set_null(i);
                    }
                }
            }
            for i in 0..n {
                if col.is_null(i) {
                    nulls.set_null(i);
                }
            }
            Ok(Column::with_nulls(
                ColumnData::Float32(out),
                nulls,
                col.type_id,
            ))
        }
        ColumnData::Int8(v) => int_lanes!(v, Int8, i8),
        ColumnData::Int16(v) => int_lanes!(v, Int16, i16),
        ColumnData::Int32(v) => int_lanes!(v, Int32, i32),
        ColumnData::Int64(v) => int_lanes!(v, Int64, i64),
        ColumnData::Int128(v) => int_lanes!(v, Int128, i128),
        _ => Err(ZyronError::ExecutionError(
            "round/trunc requires a numeric argument".to_string(),
        )),
    }
}

enum TrimSide {
    Both,
    Leading,
    Trailing,
}

fn eval_trim(
    args: &[BoundExpr],
    batch: &DataBatch,
    schema: &[LogicalColumn],
    params: &[ScalarValue],
    side: TrimSide,
) -> Result<Column> {
    if args.is_empty() || args.len() > 2 {
        return Err(ZyronError::ExecutionError(
            "trim takes 1 or 2 arguments".to_string(),
        ));
    }
    let col = evaluate(&args[0], batch, schema, params)?;
    utf8_or_err(&col, "trim")?;
    let charset = if args.len() == 2 {
        let c = evaluate(&args[1], batch, schema, params)?;
        utf8_or_err(&c, "trim")?;
        Some(c)
    } else {
        None
    };
    let strings = match &col.data {
        ColumnData::Utf8(v) => v,
        _ => unreachable!(),
    };
    let n = col.len();
    let mut out: Vec<String> = Vec::with_capacity(n);
    let mut nulls = NullBitmap::none(n);
    for (i, s) in strings.iter().enumerate() {
        if col.is_null(i) {
            out.push(String::new());
            nulls.set_null(i);
            continue;
        }
        // SQL trim strips spaces by default, an explicit charset strips any
        // of its characters
        let trimmed = match &charset {
            None => match side {
                TrimSide::Both => s.trim_matches(' '),
                TrimSide::Leading => s.trim_start_matches(' '),
                TrimSide::Trailing => s.trim_end_matches(' '),
            },
            Some(c) => {
                if c.is_null(i) {
                    out.push(String::new());
                    nulls.set_null(i);
                    continue;
                }
                let set: Vec<char> = match &c.data {
                    ColumnData::Utf8(v) => v[i].chars().collect(),
                    _ => unreachable!(),
                };
                let pred = |ch: char| set.contains(&ch);
                match side {
                    TrimSide::Both => s.trim_matches(pred),
                    TrimSide::Leading => s.trim_start_matches(pred),
                    TrimSide::Trailing => s.trim_end_matches(pred),
                }
            }
        };
        out.push(trimmed.to_string());
    }
    Ok(Column::with_nulls(
        ColumnData::Utf8(out),
        nulls,
        TypeId::Text,
    ))
}

/// One based character addressed substring with SQL overlap semantics, a
/// start below 1 shortens the window instead of erroring
fn substring_chars(s: &str, start: i64, len: Option<i64>) -> Result<String> {
    if let Some(l) = len {
        if l < 0 {
            return Err(ZyronError::ExecutionError(
                "negative substring length not allowed".to_string(),
            ));
        }
    }
    let end_excl = match len {
        Some(l) => start.saturating_add(l),
        None => i64::MAX,
    };
    let begin = start.max(1);
    if end_excl <= begin {
        return Ok(String::new());
    }
    let skip = (begin - 1) as usize;
    let take = (end_excl - begin) as usize;
    Ok(s.chars().skip(skip).take(take).collect())
}

fn eval_substring(
    args: &[BoundExpr],
    batch: &DataBatch,
    schema: &[LogicalColumn],
    params: &[ScalarValue],
) -> Result<Column> {
    if args.len() < 2 || args.len() > 3 {
        return Err(ZyronError::ExecutionError(
            "substring(string, start [, length]) takes 2 or 3 arguments".to_string(),
        ));
    }
    let col = evaluate(&args[0], batch, schema, params)?;
    utf8_or_err(&col, "substring")?;
    let start_col = evaluate(&args[1], batch, schema, params)?;
    let len_col = if args.len() == 3 {
        Some(evaluate(&args[2], batch, schema, params)?)
    } else {
        None
    };
    let int_at = |c: &Column, i: usize, what: &str| -> Result<Option<i64>> {
        if c.is_null(i) {
            return Ok(None);
        }
        match c.data.get_scalar(i) {
            ScalarValue::Int8(v) => Ok(Some(v as i64)),
            ScalarValue::Int16(v) => Ok(Some(v as i64)),
            ScalarValue::Int32(v) => Ok(Some(v as i64)),
            ScalarValue::Int64(v) => Ok(Some(v)),
            ScalarValue::Int128(v) => Ok(Some(v.clamp(i64::MIN as i128, i64::MAX as i128) as i64)),
            other => Err(ZyronError::ExecutionError(format!(
                "substring {what} must be an integer, got {other:?}"
            ))),
        }
    };
    let strings = match &col.data {
        ColumnData::Utf8(v) => v,
        _ => unreachable!(),
    };
    let n = col.len();
    let mut out: Vec<String> = Vec::with_capacity(n);
    let mut nulls = NullBitmap::none(n);
    for (i, s) in strings.iter().enumerate() {
        if col.is_null(i) {
            out.push(String::new());
            nulls.set_null(i);
            continue;
        }
        let start = match int_at(&start_col, i, "start")? {
            Some(v) => v,
            None => {
                out.push(String::new());
                nulls.set_null(i);
                continue;
            }
        };
        let len = match &len_col {
            Some(lc) => match int_at(lc, i, "length")? {
                Some(v) => Some(v),
                None => {
                    out.push(String::new());
                    nulls.set_null(i);
                    continue;
                }
            },
            None => None,
        };
        out.push(substring_chars(s, start, len)?);
    }
    Ok(Column::with_nulls(
        ColumnData::Utf8(out),
        nulls,
        TypeId::Text,
    ))
}

fn eval_replace(s: &Column, from: &Column, to: &Column) -> Result<Column> {
    utf8_or_err(s, "replace")?;
    utf8_or_err(from, "replace")?;
    utf8_or_err(to, "replace")?;
    let (sv, fv, tv) = match (&s.data, &from.data, &to.data) {
        (ColumnData::Utf8(a), ColumnData::Utf8(b), ColumnData::Utf8(c)) => (a, b, c),
        _ => unreachable!(),
    };
    let n = s.len();
    let mut out: Vec<String> = Vec::with_capacity(n);
    let mut nulls = NullBitmap::none(n);
    for i in 0..n {
        if s.is_null(i) || from.is_null(i) || to.is_null(i) {
            out.push(String::new());
            nulls.set_null(i);
            continue;
        }
        // an empty search string returns the input unchanged
        if fv[i].is_empty() {
            out.push(sv[i].clone());
        } else {
            out.push(sv[i].replace(&fv[i], &tv[i]));
        }
    }
    Ok(Column::with_nulls(
        ColumnData::Utf8(out),
        nulls,
        TypeId::Text,
    ))
}

/// Variadic concat, NULL arguments contribute nothing and the result is
/// never NULL, non string arguments cast to text
fn eval_concat(
    args: &[BoundExpr],
    batch: &DataBatch,
    schema: &[LogicalColumn],
    params: &[ScalarValue],
) -> Result<Column> {
    let n = batch.num_rows.max(1);
    let mut cols: Vec<Column> = Vec::with_capacity(args.len());
    for a in args {
        let c = evaluate(a, batch, schema, params)?;
        let c = match &c.data {
            ColumnData::Utf8(_) => c,
            _ => crate::compute::cast_column(&c, TypeId::Varchar)?,
        };
        cols.push(c);
    }
    let mut out: Vec<String> = Vec::with_capacity(n);
    for i in 0..n {
        let mut s = String::new();
        for c in &cols {
            if i < c.len() && !c.is_null(i) {
                if let ColumnData::Utf8(v) = &c.data {
                    s.push_str(&v[i]);
                }
            }
        }
        out.push(s);
    }
    Ok(Column::new(ColumnData::Utf8(out), TypeId::Text))
}

/// Row wise maximum or minimum across the arguments, NULLs are ignored and
/// the result is NULL only when every argument is NULL
/// Splits a day count since the Unix epoch into its calendar year, month and
/// day, in the proleptic Gregorian calendar.
fn civil_from_epoch_days(days: i64) -> (i64, i64, i64) {
    let z = days + 719_468;
    let era = if z >= 0 { z } else { z - 146_096 } / 146_097;
    let doe = z - era * 146_097;
    let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146_096) / 365;
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    (if m <= 2 { y + 1 } else { y }, m, d)
}

/// `EXTRACT(field FROM source)`, also reachable as `date_part`.
///
/// A DATE is stored as days since the epoch and a TIMESTAMP as microseconds,
/// so the field is read from the calendar split of whichever unit the source
/// carries. The result is an integer, which is what every comparison and
/// grouping the field feeds expects.
fn eval_extract(
    args: &[BoundExpr],
    batch: &DataBatch,
    schema: &[LogicalColumn],
    params: &[ScalarValue],
) -> Result<Column> {
    if args.len() != 2 {
        return Err(ZyronError::ExecutionError(format!(
            "extract takes a field and a source, got {} argument(s)",
            args.len()
        )));
    }
    let field_col = evaluate(&args[0], batch, schema, params)?;
    let field = match field_col.data {
        ColumnData::Utf8(ref v) if !v.is_empty() => v[0].to_ascii_lowercase(),
        _ => {
            return Err(ZyronError::ExecutionError(
                "extract needs a constant field name, like EXTRACT(YEAR FROM ts)".to_string(),
            ));
        }
    };
    let source = evaluate(&args[1], batch, schema, params)?;

    // Every source reduces to microseconds since the epoch, so one split
    // serves dates, timestamps and times alike
    let micros_of = |i: usize| -> Option<i64> {
        if source.is_null(i) {
            return None;
        }
        match (&source.data, source.type_id) {
            (ColumnData::Int32(v), TypeId::Date) => Some(v[i] as i64 * 86_400_000_000),
            (ColumnData::Int64(v), _) => Some(v[i]),
            (ColumnData::Int32(v), _) => Some(v[i] as i64),
            (ColumnData::Int128(v), _) => Some(v[i] as i64),
            _ => None,
        }
    };

    let n = source.len();
    let mut out: Vec<i64> = Vec::with_capacity(n);
    let mut nulls = NullBitmap::empty();
    for i in 0..n {
        let Some(micros) = micros_of(i) else {
            out.push(0);
            nulls.push(true);
            continue;
        };
        let days = micros.div_euclid(86_400_000_000);
        let time_of_day = micros.rem_euclid(86_400_000_000);
        let (year, month, day) = civil_from_epoch_days(days);
        let value = match field.as_str() {
            "year" => year,
            "month" => month,
            "day" => day,
            "hour" => time_of_day / 3_600_000_000,
            "minute" => time_of_day / 60_000_000 % 60,
            "second" => time_of_day / 1_000_000 % 60,
            "millisecond" => time_of_day / 1_000 % 60_000,
            "microsecond" => time_of_day % 60_000_000,
            "quarter" => (month - 1) / 3 + 1,
            // The epoch fell on a Thursday, so day zero is weekday four
            "dow" | "dayofweek" => (days + 4).rem_euclid(7),
            "doy" | "dayofyear" => days - crate::expr::epoch_days_from_civil(year, 1, 1) + 1,
            "epoch" => micros / 1_000_000,
            "decade" => year / 10,
            "century" => (year - 1) / 100 + 1,
            "millennium" => (year - 1) / 1000 + 1,
            other => {
                return Err(ZyronError::ExecutionError(format!(
                    "extract does not know the field '{other}'"
                )));
            }
        };
        out.push(value);
        nulls.push(false);
    }
    Ok(Column::with_nulls(
        ColumnData::Int64(out),
        nulls,
        TypeId::Int64,
    ))
}

/// The inverse of `civil_from_epoch_days`, for the day-of-year field.
pub(crate) fn epoch_days_from_civil(year: i64, month: i64, day: i64) -> i64 {
    let y = if month <= 2 { year - 1 } else { year };
    let era = if y >= 0 { y } else { y - 399 } / 400;
    let yoe = y - era * 400;
    let doy = (153 * (if month > 2 { month - 3 } else { month + 9 }) + 2) / 5 + day - 1;
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    era * 146_097 + doe - 719_468
}

fn eval_greatest_least(
    args: &[BoundExpr],
    batch: &DataBatch,
    schema: &[LogicalColumn],
    params: &[ScalarValue],
    greatest: bool,
) -> Result<Column> {
    if args.is_empty() {
        return Err(ZyronError::ExecutionError(
            "greatest/least requires at least 1 argument".to_string(),
        ));
    }
    let first = evaluate(&args[0], batch, schema, params)?;
    let target = first.type_id;
    let fractional_digits = first.fractional_digits;
    let mut cols: Vec<Column> = Vec::with_capacity(args.len());
    for (idx, a) in args.iter().enumerate() {
        let c = if idx == 0 {
            first.clone()
        } else {
            evaluate(a, batch, schema, params)?
        };
        let c = if c.type_id == target {
            c
        } else {
            crate::compute::cast_column(&c, target)?
        };
        cols.push(c);
    }
    let n = cols.iter().map(|c| c.len()).max().unwrap_or(0);
    let mut data = ColumnData::with_capacity(target, n);
    let mut nulls = NullBitmap::none(n);
    for i in 0..n {
        // best holds the winning argument index for this row
        let mut best: Option<usize> = None;
        for (k, c) in cols.iter().enumerate() {
            if i >= c.len() || c.is_null(i) {
                continue;
            }
            match best {
                None => best = Some(k),
                Some(b) => {
                    let winner_loses = if greatest {
                        values_less_at(&cols[b].data, i, &c.data, i)
                    } else {
                        values_less_at(&c.data, i, &cols[b].data, i)
                    };
                    if winner_loses {
                        best = Some(k);
                    }
                }
            }
        }
        match best {
            Some(k) => data.push_from(&cols[k].data, i),
            None => {
                data.push_default();
                nulls.set_null(i);
            }
        }
    }
    Ok(Column::with_nulls_ts(
        data,
        nulls,
        target,
        fractional_digits,
    ))
}

#[cfg(test)]
mod scalar_fn_tests {
    use super::*;

    fn lit_int(v: i64) -> BoundExpr {
        BoundExpr::Literal {
            value: LiteralValue::Integer(v),
            type_id: TypeId::Int64,
        }
    }
    fn lit_float(v: f64) -> BoundExpr {
        BoundExpr::Literal {
            value: LiteralValue::Float(v),
            type_id: TypeId::Float64,
        }
    }
    fn lit_str(s: &str) -> BoundExpr {
        BoundExpr::Literal {
            value: LiteralValue::String(s.to_string()),
            type_id: TypeId::Text,
        }
    }
    fn lit_null(t: TypeId) -> BoundExpr {
        BoundExpr::Literal {
            value: LiteralValue::Null,
            type_id: t,
        }
    }
    fn one_row_batch() -> DataBatch {
        DataBatch::new(vec![Column::new(ColumnData::Int64(vec![0]), TypeId::Int64)])
    }
    fn call(name: &str, args: &[BoundExpr]) -> Result<Column> {
        evaluate_function(name, args, &one_row_batch(), &[], &[])
    }
    fn f64_at0(c: &Column) -> f64 {
        match c.data.get_scalar(0) {
            ScalarValue::Float64(v) => v,
            other => panic!("expected Float64, got {other:?}"),
        }
    }
    fn i64_at0(c: &Column) -> i64 {
        match c.data.get_scalar(0) {
            ScalarValue::Int64(v) => v,
            other => panic!("expected Int64, got {other:?}"),
        }
    }
    fn str_at0(c: &Column) -> String {
        match c.data.get_scalar(0) {
            ScalarValue::Utf8(v) => v,
            other => panic!("expected Utf8, got {other:?}"),
        }
    }

    /// Row-wise vector distance over a vector column and a query vector.
    /// A vector column holds raw little-endian f32 bytes, and the query side
    /// arrives as an array constructor, so both shapes have to read.
    #[test]
    fn vector_distance_reads_a_vector_column_against_a_query_vector() {
        fn vector_bytes(v: &[f32]) -> Vec<u8> {
            v.iter().flat_map(|f| f.to_le_bytes()).collect()
        }
        let batch = DataBatch::new(vec![Column::new(
            ColumnData::Binary(vec![
                vector_bytes(&[1.0, 0.0, 0.0]),
                vector_bytes(&[0.0, 1.0, 0.0]),
                vector_bytes(&[0.9, 0.1, 0.0]),
            ]),
            TypeId::Vector,
        )]);
        let schema = vec![LogicalColumn {
            table_idx: Some(0),
            column_id: zyron_catalog::ColumnId(1),
            name: "embedding".to_string(),
            type_id: TypeId::Vector,
            nullable: false,
            fractional_digits: None,
        }];
        let column_ref = BoundExpr::ColumnRef(zyron_planner::binder::ColumnRef {
            table_idx: 0,
            column_id: zyron_catalog::ColumnId(1),
            type_id: TypeId::Vector,
            nullable: false,
            fractional_digits: None,
        });
        let query = BoundExpr::Function {
            name: "array".to_string(),
            args: vec![lit_float(1.0), lit_float(0.0), lit_float(0.0)],
            return_type: TypeId::Array,
            distinct: false,
        };

        let cosine = evaluate_function(
            "vector_distance_cosine",
            &[column_ref.clone(), query.clone()],
            &batch,
            &schema,
            &[],
        )
        .unwrap();
        let ColumnData::Float64(values) = &cosine.data else {
            panic!("expected a float column");
        };
        assert!(
            values[0].abs() < 1e-6,
            "a row equal to the query is at zero"
        );
        assert!(
            (values[1] - 1.0).abs() < 1e-6,
            "an orthogonal row is at one"
        );
        assert!(
            values[2] > 0.0 && values[2] < 0.01,
            "a nearly parallel row is close, got {}",
            values[2]
        );

        let l2 = evaluate_function(
            "vector_distance_l2",
            &[column_ref, query],
            &batch,
            &schema,
            &[],
        )
        .unwrap();
        let ColumnData::Float64(values) = &l2.data else {
            panic!("expected a float column");
        };
        assert!(values[0].abs() < 1e-6);
        assert!((values[1] - std::f64::consts::SQRT_2).abs() < 1e-6);
    }

    #[test]
    fn round_floats_half_away_from_zero() {
        assert_eq!(f64_at0(&call("round", &[lit_float(2.5)]).unwrap()), 3.0);
        assert_eq!(f64_at0(&call("round", &[lit_float(-2.5)]).unwrap()), -3.0);
        assert_eq!(f64_at0(&call("round", &[lit_float(2.4)]).unwrap()), 2.0);
        let two_digits = call("round", &[lit_float(3.14159), lit_int(2)]).unwrap();
        assert!((f64_at0(&two_digits) - 3.14).abs() < 1e-12);
        let neg_digits = call("round", &[lit_float(1234.5), lit_int(-2)]).unwrap();
        assert_eq!(f64_at0(&neg_digits), 1200.0);
    }

    #[test]
    fn round_integers_pass_through_and_negative_digits_round_the_integer() {
        assert_eq!(i64_at0(&call("round", &[lit_int(42)]).unwrap()), 42);
        assert_eq!(
            i64_at0(&call("round", &[lit_int(1250), lit_int(-2)]).unwrap()),
            1300
        );
        assert_eq!(
            i64_at0(&call("round", &[lit_int(-1250), lit_int(-2)]).unwrap()),
            -1300
        );
        assert_eq!(
            i64_at0(&call("trunc", &[lit_int(1299), lit_int(-2)]).unwrap()),
            1200
        );
    }

    #[test]
    fn trunc_moves_toward_zero() {
        assert_eq!(f64_at0(&call("trunc", &[lit_float(2.9)]).unwrap()), 2.0);
        assert_eq!(f64_at0(&call("trunc", &[lit_float(-2.9)]).unwrap()), -2.0);
        assert_eq!(f64_at0(&call("truncate", &[lit_float(5.5)]).unwrap()), 5.0);
    }

    #[test]
    fn round_null_input_and_null_digits_are_null() {
        let c = call("round", &[lit_null(TypeId::Float64)]).unwrap();
        assert!(c.is_null(0));
        let c = call("round", &[lit_float(1.5), lit_null(TypeId::Int64)]).unwrap();
        assert!(c.is_null(0));
    }

    #[test]
    fn ceil_and_floor() {
        assert_eq!(f64_at0(&call("ceil", &[lit_float(2.1)]).unwrap()), 3.0);
        assert_eq!(f64_at0(&call("ceiling", &[lit_float(-2.1)]).unwrap()), -2.0);
        assert_eq!(f64_at0(&call("floor", &[lit_float(2.9)]).unwrap()), 2.0);
        assert_eq!(f64_at0(&call("floor", &[lit_float(-2.1)]).unwrap()), -3.0);
        assert_eq!(i64_at0(&call("ceil", &[lit_int(7)]).unwrap()), 7);
    }

    #[test]
    fn trim_strips_spaces_only_by_default() {
        assert_eq!(str_at0(&call("trim", &[lit_str("  x  ")]).unwrap()), "x");
        assert_eq!(str_at0(&call("ltrim", &[lit_str("  x  ")]).unwrap()), "x  ");
        assert_eq!(str_at0(&call("rtrim", &[lit_str("  x  ")]).unwrap()), "  x");
        assert_eq!(
            str_at0(&call("trim", &[lit_str("\tx\t")]).unwrap()),
            "\tx\t"
        );
    }

    #[test]
    fn trim_with_explicit_charset() {
        assert_eq!(
            str_at0(&call("trim", &[lit_str("xxaxx"), lit_str("x")]).unwrap()),
            "a"
        );
        assert_eq!(
            str_at0(&call("ltrim", &[lit_str("xyay"), lit_str("xy")]).unwrap()),
            "ay"
        );
    }

    #[test]
    fn substring_is_one_based_with_overlap_semantics() {
        assert_eq!(
            str_at0(&call("substring", &[lit_str("hello"), lit_int(2)]).unwrap()),
            "ello"
        );
        assert_eq!(
            str_at0(&call("substring", &[lit_str("hello"), lit_int(2), lit_int(2)]).unwrap()),
            "el"
        );
        assert_eq!(
            str_at0(&call("substring", &[lit_str("hello"), lit_int(0), lit_int(3)]).unwrap()),
            "he"
        );
        assert_eq!(
            str_at0(&call("substring", &[lit_str("hello"), lit_int(-2), lit_int(4)]).unwrap()),
            "h"
        );
        assert_eq!(
            str_at0(&call("substr", &[lit_str("h\u{e9}llo"), lit_int(2), lit_int(2)]).unwrap()),
            "\u{e9}l"
        );
        assert!(call("substring", &[lit_str("x"), lit_int(1), lit_int(-1)]).is_err());
    }

    #[test]
    fn replace_all_occurrences_and_empty_search_is_identity() {
        assert_eq!(
            str_at0(&call("replace", &[lit_str("aaa"), lit_str("a"), lit_str("b")]).unwrap()),
            "bbb"
        );
        assert_eq!(
            str_at0(&call("replace", &[lit_str("abc"), lit_str(""), lit_str("x")]).unwrap()),
            "abc"
        );
        let c = call(
            "replace",
            &[lit_str("abc"), lit_null(TypeId::Text), lit_str("x")],
        )
        .unwrap();
        assert!(c.is_null(0));
    }

    #[test]
    fn concat_skips_nulls_and_casts_non_strings() {
        assert_eq!(
            str_at0(
                &call(
                    "concat",
                    &[lit_str("a"), lit_null(TypeId::Text), lit_str("b")]
                )
                .unwrap()
            ),
            "ab"
        );
        assert_eq!(
            str_at0(&call("concat", &[lit_int(1), lit_str("x")]).unwrap()),
            "1x"
        );
    }

    #[test]
    fn greatest_and_least_ignore_nulls() {
        assert_eq!(
            i64_at0(&call("greatest", &[lit_int(1), lit_int(3), lit_int(2)]).unwrap()),
            3
        );
        assert_eq!(
            i64_at0(&call("least", &[lit_null(TypeId::Int64), lit_int(5), lit_int(3)]).unwrap()),
            3
        );
        let c = call(
            "greatest",
            &[lit_null(TypeId::Int64), lit_null(TypeId::Int64)],
        )
        .unwrap();
        assert!(c.is_null(0));
        assert_eq!(
            str_at0(&call("greatest", &[lit_str("apple"), lit_str("pear")]).unwrap()),
            "pear"
        );
    }

    #[test]
    fn length_family_counts_chars_and_octets() {
        assert_eq!(
            i64_at0(&call("length", &[lit_str("h\u{e9}llo")]).unwrap()),
            5
        );
        assert_eq!(
            i64_at0(&call("char_length", &[lit_str("h\u{e9}llo")]).unwrap()),
            5
        );
        assert_eq!(
            i64_at0(&call("octet_length", &[lit_str("h\u{e9}llo")]).unwrap()),
            6
        );
    }

    #[test]
    fn current_date_and_time_produce_sane_values() {
        let d = call("current_date", &[]).unwrap();
        assert_eq!(d.type_id, TypeId::Date);
        match d.data.get_scalar(0) {
            ScalarValue::Int32(days) => assert!(days > 20_000),
            other => panic!("expected Int32 days, got {other:?}"),
        }
        let t = call("current_time", &[]).unwrap();
        assert_eq!(t.type_id, TypeId::Time);
        match t.data.get_scalar(0) {
            ScalarValue::Int64(us) => assert!((0..86_400_000_000).contains(&us)),
            other => panic!("expected Int64 micros, got {other:?}"),
        }
    }
}
