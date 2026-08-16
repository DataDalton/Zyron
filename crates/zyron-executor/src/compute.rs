//! Vectorized compute kernels for column operations.
//!
//! Provides filter, comparison, arithmetic, boolean logic, sorting,
//! and hashing operations on typed column vectors. All hot-path kernels
//! use typed dispatch to operate directly on Vec<T> arrays, avoiding
//! ScalarValue intermediaries.

use std::cmp::Ordering;

use zyron_common::{Result, TypeId, ZyronError};

// Re-export the canonical IdentityHasher / PreHashMap from zyron-common so
// existing call sites (aggregate, distinct, setop) stay unchanged. The one
// definition lives in zyron-common alongside the FxHash mixing primitives.
pub use zyron_common::{IdentityHasher, PreHashMap};

use crate::column::{Column, ColumnData, NullBitmap, ScalarValue};

// ---------------------------------------------------------------------------
// Comparison kernels
// ---------------------------------------------------------------------------

/// Comparison operation types.
#[derive(Debug, Clone, Copy)]
pub enum CmpOp {
    Eq,
    Neq,
    Lt,
    Gt,
    LtEq,
    GtEq,
}

/// Generates a typed comparison fast path for Ord types (integers, etc.).
macro_rules! typed_cmp_ord {
    ($left:expr, $right:expr, $left_nulls:expr, $right_nulls:expr, $len:expr, $op:expr, $variant:ident) => {
        if let (ColumnData::$variant(l), ColumnData::$variant(r)) = ($left, $right) {
            let mut result = Vec::with_capacity($len);
            let mut nulls = NullBitmap::none($len);
            for i in 0..$len {
                if $left_nulls.is_null(i) || $right_nulls.is_null(i) {
                    nulls.set_null(i);
                    result.push(false);
                } else {
                    result.push(match $op {
                        CmpOp::Eq => l[i] == r[i],
                        CmpOp::Neq => l[i] != r[i],
                        CmpOp::Lt => l[i] < r[i],
                        CmpOp::Gt => l[i] > r[i],
                        CmpOp::LtEq => l[i] <= r[i],
                        CmpOp::GtEq => l[i] >= r[i],
                    });
                }
            }
            return Ok(Column::with_nulls(
                ColumnData::Boolean(result),
                nulls,
                TypeId::Boolean,
            ));
        }
    };
}

/// Generates a typed comparison fast path for PartialOrd types (floats).
macro_rules! typed_cmp_partial {
    ($left:expr, $right:expr, $left_nulls:expr, $right_nulls:expr, $len:expr, $op:expr, $variant:ident) => {
        if let (ColumnData::$variant(l), ColumnData::$variant(r)) = ($left, $right) {
            let mut result = Vec::with_capacity($len);
            let mut nulls = NullBitmap::none($len);
            for i in 0..$len {
                if $left_nulls.is_null(i) || $right_nulls.is_null(i) {
                    nulls.set_null(i);
                    result.push(false);
                } else {
                    result.push(match $op {
                        CmpOp::Eq => l[i] == r[i],
                        CmpOp::Neq => l[i] != r[i],
                        CmpOp::Lt => l[i] < r[i],
                        CmpOp::Gt => l[i] > r[i],
                        CmpOp::LtEq => l[i] <= r[i],
                        CmpOp::GtEq => l[i] >= r[i],
                    });
                }
            }
            return Ok(Column::with_nulls(
                ColumnData::Boolean(result),
                nulls,
                TypeId::Boolean,
            ));
        }
    };
}

/// Compares two columns element-wise, producing a boolean result column.
/// Uses typed fast paths for common types, falling back to ScalarValue for mixed types.
pub fn compare(left: &Column, right: &Column, op: CmpOp) -> Result<Column> {
    let len = left.len();
    if len != right.len() {
        return Err(ZyronError::ExecutionError(
            "compare: column length mismatch".to_string(),
        ));
    }

    // A decimal's stored integer is the value times ten to its scale, so it
    // only means the same thing as the other side once both sit on one
    // scale. Comparing the raw integer against a plain number would read
    // 10.50 as 1050 and make every row larger than any literal. The wider
    // scale is used so neither side loses digits.
    if let Some((l, r)) = align_decimal_operands(left, right)? {
        return compare(&l, &r, op);
    }

    // Typed fast paths: direct array comparison without ScalarValue.
    typed_cmp_ord!(
        &left.data,
        &right.data,
        &left.nulls,
        &right.nulls,
        len,
        op,
        Int64
    );
    typed_cmp_ord!(
        &left.data,
        &right.data,
        &left.nulls,
        &right.nulls,
        len,
        op,
        Int32
    );
    typed_cmp_ord!(
        &left.data,
        &right.data,
        &left.nulls,
        &right.nulls,
        len,
        op,
        Int16
    );
    typed_cmp_ord!(
        &left.data,
        &right.data,
        &left.nulls,
        &right.nulls,
        len,
        op,
        Int8
    );
    typed_cmp_ord!(
        &left.data,
        &right.data,
        &left.nulls,
        &right.nulls,
        len,
        op,
        Int128
    );
    typed_cmp_ord!(
        &left.data,
        &right.data,
        &left.nulls,
        &right.nulls,
        len,
        op,
        UInt8
    );
    typed_cmp_ord!(
        &left.data,
        &right.data,
        &left.nulls,
        &right.nulls,
        len,
        op,
        UInt16
    );
    typed_cmp_ord!(
        &left.data,
        &right.data,
        &left.nulls,
        &right.nulls,
        len,
        op,
        UInt32
    );
    typed_cmp_ord!(
        &left.data,
        &right.data,
        &left.nulls,
        &right.nulls,
        len,
        op,
        UInt64
    );
    typed_cmp_ord!(
        &left.data,
        &right.data,
        &left.nulls,
        &right.nulls,
        len,
        op,
        Boolean
    );
    typed_cmp_ord!(
        &left.data,
        &right.data,
        &left.nulls,
        &right.nulls,
        len,
        op,
        Utf8
    );
    typed_cmp_partial!(
        &left.data,
        &right.data,
        &left.nulls,
        &right.nulls,
        len,
        op,
        Float64
    );
    typed_cmp_partial!(
        &left.data,
        &right.data,
        &left.nulls,
        &right.nulls,
        len,
        op,
        Float32
    );

    // Fallback: ScalarValue comparison for mixed or rare types.
    let mut result = Vec::with_capacity(len);
    let mut nulls = NullBitmap::none(len);
    for i in 0..len {
        if left.is_null(i) || right.is_null(i) {
            nulls.set_null(i);
            result.push(false);
        } else {
            let l = left.data.get_scalar(i);
            let r = right.data.get_scalar(i);
            result.push(match op {
                CmpOp::Eq => l == r,
                CmpOp::Neq => l != r,
                CmpOp::Lt => l.partial_cmp(&r).is_some_and(|o| o == Ordering::Less),
                CmpOp::Gt => l.partial_cmp(&r).is_some_and(|o| o == Ordering::Greater),
                CmpOp::LtEq => l.partial_cmp(&r).is_some_and(|o| o != Ordering::Greater),
                CmpOp::GtEq => l.partial_cmp(&r).is_some_and(|o| o != Ordering::Less),
            });
        }
    }
    Ok(Column::with_nulls(
        ColumnData::Boolean(result),
        nulls,
        TypeId::Boolean,
    ))
}

// ---------------------------------------------------------------------------
// Arithmetic kernels
// ---------------------------------------------------------------------------

/// Arithmetic operation types.
#[derive(Debug, Clone, Copy)]
pub enum ArithOp {
    Add,
    Sub,
    Mul,
    Div,
    Mod,
}

/// Applies an arithmetic operation element-wise on two columns.
/// Uses typed fast paths for Int64 and Float64.
/// Arithmetic on two decimals already sharing a scale.
///
/// Addition and subtraction keep the scale. A product of two values at scale
/// `s` lands at scale `2s`, and a quotient at scale zero, so each is brought
/// back to `s`, which keeps the result in the same units as its operands
/// rather than silently changing where the point sits.
fn decimal_arithmetic(left: &Column, right: &Column, scale: u8, op: ArithOp) -> Result<Column> {
    let (ColumnData::Int128(l), ColumnData::Int128(r)) = (&left.data, &right.data) else {
        return Err(ZyronError::ExecutionError(
            "decimal arithmetic needs two scaled operands".to_string(),
        ));
    };
    let len = left.len();
    let factor = zyron_common::decimal::scale_factor(scale)?;
    let mut out: Vec<i128> = Vec::with_capacity(len);
    let mut nulls = NullBitmap::none(len);
    for i in 0..len {
        if left.is_null(i) || right.is_null(i) {
            nulls.set_null(i);
            out.push(0);
            continue;
        }
        let (a, b) = (l[i], r[i]);
        let value = match op {
            ArithOp::Add => a.checked_add(b),
            ArithOp::Sub => a.checked_sub(b),
            // The product carries both scales, so it is divided back down
            ArithOp::Mul => a
                .checked_mul(b)
                .map(|p| zyron_common::rescale(p, scale.saturating_mul(2).min(38), scale).ok())
                .and_then(|v| v)
                .or_else(|| a.checked_mul(b).and_then(|p| p.checked_div(factor))),
            // A quotient of two same-scale values has no scale, so the
            // dividend is raised first to land the result back on this one.
            // The last digit rounds half away from zero, the same rule
            // multiplication applies, a truncating quotient would make
            // 2.00 / 3.00 read 0.66 while 2.00 * (1.00 / 3.00) rounds
            ArithOp::Div => {
                if b == 0 {
                    nulls.set_null(i);
                    out.push(0);
                    continue;
                }
                a.checked_mul(factor).and_then(|n| {
                    let half = (b / 2).abs();
                    let adjust = if (n < 0) == (b < 0) { half } else { -half };
                    n.checked_add(adjust).and_then(|n| n.checked_div(b))
                })
            }
            ArithOp::Mod => {
                if b == 0 {
                    nulls.set_null(i);
                    out.push(0);
                    continue;
                }
                a.checked_rem(b)
            }
        };
        match value {
            Some(v) => out.push(v),
            None => {
                return Err(ZyronError::ExecutionError(
                    "decimal arithmetic overflowed".to_string(),
                ));
            }
        }
    }
    Ok(Column::with_nulls_ts(
        ColumnData::Int128(out),
        nulls,
        TypeId::Decimal,
        Some(scale),
    ))
}

pub fn arithmetic(left: &Column, right: &Column, op: ArithOp) -> Result<Column> {
    let len = left.len();
    if len != right.len() {
        return Err(ZyronError::ExecutionError(
            "arithmetic: column length mismatch".to_string(),
        ));
    }

    // A decimal keeps its own units through arithmetic. Falling into the
    // generic numeric path would read its scaled integer as a plain number
    if left.type_id == TypeId::Decimal || right.type_id == TypeId::Decimal {
        let (l, r) = match align_decimal_operands(left, right)? {
            Some(pair) => pair,
            None => (left.clone(), right.clone()),
        };
        let scale = l.fractional_digits.unwrap_or(0);
        return decimal_arithmetic(&l, &r, scale, op);
    }

    // Int64 fast path.
    if let (ColumnData::Int64(l), ColumnData::Int64(r)) = (&left.data, &right.data) {
        let mut result = Vec::with_capacity(len);
        let mut nulls = NullBitmap::none(len);
        for i in 0..len {
            if left.is_null(i) || right.is_null(i) {
                nulls.set_null(i);
                result.push(0);
            } else {
                result.push(match op {
                    ArithOp::Add => l[i].wrapping_add(r[i]),
                    ArithOp::Sub => l[i].wrapping_sub(r[i]),
                    ArithOp::Mul => l[i].wrapping_mul(r[i]),
                    ArithOp::Div => {
                        if r[i] == 0 {
                            return Err(ZyronError::ExecutionError("division by zero".to_string()));
                        }
                        l[i] / r[i]
                    }
                    ArithOp::Mod => {
                        if r[i] == 0 {
                            return Err(ZyronError::ExecutionError("modulo by zero".to_string()));
                        }
                        l[i] % r[i]
                    }
                });
            }
        }
        return Ok(Column::with_nulls(
            ColumnData::Int64(result),
            nulls,
            TypeId::Int64,
        ));
    }

    // Int32 fast path.
    if let (ColumnData::Int32(l), ColumnData::Int32(r)) = (&left.data, &right.data) {
        let mut result = Vec::with_capacity(len);
        let mut nulls = NullBitmap::none(len);
        for i in 0..len {
            if left.is_null(i) || right.is_null(i) {
                nulls.set_null(i);
                result.push(0);
            } else {
                result.push(match op {
                    ArithOp::Add => l[i].wrapping_add(r[i]),
                    ArithOp::Sub => l[i].wrapping_sub(r[i]),
                    ArithOp::Mul => l[i].wrapping_mul(r[i]),
                    ArithOp::Div => {
                        if r[i] == 0 {
                            return Err(ZyronError::ExecutionError("division by zero".to_string()));
                        }
                        l[i] / r[i]
                    }
                    ArithOp::Mod => {
                        if r[i] == 0 {
                            return Err(ZyronError::ExecutionError("modulo by zero".to_string()));
                        }
                        l[i] % r[i]
                    }
                });
            }
        }
        return Ok(Column::with_nulls(
            ColumnData::Int32(result),
            nulls,
            TypeId::Int32,
        ));
    }

    // Float64 fast path.
    if let (ColumnData::Float64(l), ColumnData::Float64(r)) = (&left.data, &right.data) {
        let mut result = Vec::with_capacity(len);
        let mut nulls = NullBitmap::none(len);
        for i in 0..len {
            if left.is_null(i) || right.is_null(i) {
                nulls.set_null(i);
                result.push(0.0);
            } else {
                result.push(match op {
                    ArithOp::Add => l[i] + r[i],
                    ArithOp::Sub => l[i] - r[i],
                    ArithOp::Mul => l[i] * r[i],
                    ArithOp::Div => l[i] / r[i],
                    ArithOp::Mod => l[i] % r[i],
                });
            }
        }
        return Ok(Column::with_nulls(
            ColumnData::Float64(result),
            nulls,
            TypeId::Float64,
        ));
    }

    // Fallback: ScalarValue path for mixed or promoted types.
    let out_type = promote_numeric(left.type_id, right.type_id);
    let mut data = ColumnData::with_capacity(out_type, len);
    let mut nulls = NullBitmap::none(len);
    for i in 0..len {
        if left.is_null(i) || right.is_null(i) {
            nulls.set_null(i);
            data.push_default();
        } else {
            let l = left.data.get_scalar(i);
            let r = right.data.get_scalar(i);
            let result = apply_arith(&l, &r, op, out_type)?;
            data.push_scalar(&result);
        }
    }
    Ok(Column::with_nulls(data, nulls, out_type))
}

/// Applies arithmetic on two scalar values, producing a result whose variant
/// matches `out_type` so it can be pushed into a column of that type. Integer
/// operands compute exactly in i128 and narrow to the output integer type;
/// floating-point operands compute in f64. A float output type or a float
/// operand forces the floating-point path.
fn apply_arith(
    left: &ScalarValue,
    right: &ScalarValue,
    op: ArithOp,
    out_type: TypeId,
) -> Result<ScalarValue> {
    let out_float = matches!(out_type, TypeId::Float32 | TypeId::Float64);
    let operand_float = matches!(left, ScalarValue::Float32(_) | ScalarValue::Float64(_))
        || matches!(right, ScalarValue::Float32(_) | ScalarValue::Float64(_));

    if !out_float && !operand_float {
        if let (Some(l), Some(r)) = (left.to_i128(), right.to_i128()) {
            let v = match op {
                ArithOp::Add => l.wrapping_add(r),
                ArithOp::Sub => l.wrapping_sub(r),
                ArithOp::Mul => l.wrapping_mul(r),
                ArithOp::Div => {
                    if r == 0 {
                        return Err(ZyronError::ExecutionError("division by zero".to_string()));
                    }
                    l / r
                }
                ArithOp::Mod => {
                    if r == 0 {
                        return Err(ZyronError::ExecutionError("modulo by zero".to_string()));
                    }
                    l % r
                }
            };
            return Ok(int_scalar_for_type(out_type, v));
        }
    }

    let l = left.to_f64().unwrap_or(0.0);
    let r = right.to_f64().unwrap_or(0.0);
    let result = match op {
        ArithOp::Add => l + r,
        ArithOp::Sub => l - r,
        ArithOp::Mul => l * r,
        ArithOp::Div => l / r,
        ArithOp::Mod => l % r,
    };
    if out_type == TypeId::Float32 {
        Ok(ScalarValue::Float32(result as f32))
    } else {
        Ok(ScalarValue::Float64(result))
    }
}

/// Builds an integer ScalarValue of the given type from an i128 result.
fn int_scalar_for_type(out_type: TypeId, v: i128) -> ScalarValue {
    match out_type {
        TypeId::Int8 => ScalarValue::Int8(v as i8),
        TypeId::Int16 => ScalarValue::Int16(v as i16),
        TypeId::Int32 => ScalarValue::Int32(v as i32),
        TypeId::Int64 => ScalarValue::Int64(v as i64),
        TypeId::Int128 => ScalarValue::Int128(v),
        TypeId::UInt8 => ScalarValue::UInt8(v as u8),
        TypeId::UInt16 => ScalarValue::UInt16(v as u16),
        TypeId::UInt32 => ScalarValue::UInt32(v as u32),
        TypeId::UInt64 => ScalarValue::UInt64(v as u64),
        _ => ScalarValue::Int64(v as i64),
    }
}

/// Promotes two numeric types to a common output type.
fn promote_numeric(a: TypeId, b: TypeId) -> TypeId {
    if a == b {
        return a;
    }
    if a == TypeId::Float64 || b == TypeId::Float64 {
        return TypeId::Float64;
    }
    if a == TypeId::Float32 || b == TypeId::Float32 {
        return TypeId::Float64;
    }
    if a == TypeId::Int128 || b == TypeId::Int128 || a == TypeId::Decimal || b == TypeId::Decimal {
        return TypeId::Int128;
    }
    TypeId::Int64
}

// ---------------------------------------------------------------------------
// String concatenation
// ---------------------------------------------------------------------------

/// Concatenates two string columns element-wise.
pub fn concat_strings(left: &Column, right: &Column) -> Result<Column> {
    let len = left.len();
    if len != right.len() {
        return Err(ZyronError::ExecutionError(
            "concat: column length mismatch".to_string(),
        ));
    }

    let mut result = Vec::with_capacity(len);
    let mut nulls = NullBitmap::none(len);

    if let (ColumnData::Utf8(lv), ColumnData::Utf8(rv)) = (&left.data, &right.data) {
        for i in 0..len {
            if left.is_null(i) || right.is_null(i) {
                nulls.set_null(i);
                result.push(String::new());
            } else {
                let mut s = String::with_capacity(lv[i].len() + rv[i].len());
                s.push_str(&lv[i]);
                s.push_str(&rv[i]);
                result.push(s);
            }
        }
    }

    Ok(Column::with_nulls(
        ColumnData::Utf8(result),
        nulls,
        TypeId::Text,
    ))
}

// ---------------------------------------------------------------------------
// Boolean logic kernels
// ---------------------------------------------------------------------------

/// Element-wise AND of two boolean columns.
pub fn bool_and(left: &Column, right: &Column) -> Result<Column> {
    let len = left.len();
    let l = left.as_bools();
    let r = right.as_bools();
    let mut result = Vec::with_capacity(len);
    let mut nulls = NullBitmap::none(len);

    for i in 0..len {
        let l_null = left.is_null(i);
        let r_null = right.is_null(i);
        if l_null && r_null {
            nulls.set_null(i);
            result.push(false);
        } else if l_null {
            if !r[i] {
                result.push(false);
            } else {
                nulls.set_null(i);
                result.push(false);
            }
        } else if r_null {
            if !l[i] {
                result.push(false);
            } else {
                nulls.set_null(i);
                result.push(false);
            }
        } else {
            result.push(l[i] && r[i]);
        }
    }

    Ok(Column::with_nulls(
        ColumnData::Boolean(result),
        nulls,
        TypeId::Boolean,
    ))
}

/// Element-wise OR of two boolean columns.
pub fn bool_or(left: &Column, right: &Column) -> Result<Column> {
    let len = left.len();
    let l = left.as_bools();
    let r = right.as_bools();
    let mut result = Vec::with_capacity(len);
    let mut nulls = NullBitmap::none(len);

    for i in 0..len {
        let l_null = left.is_null(i);
        let r_null = right.is_null(i);
        if l_null && r_null {
            nulls.set_null(i);
            result.push(false);
        } else if l_null {
            if r[i] {
                result.push(true);
            } else {
                nulls.set_null(i);
                result.push(false);
            }
        } else if r_null {
            if l[i] {
                result.push(true);
            } else {
                nulls.set_null(i);
                result.push(false);
            }
        } else {
            result.push(l[i] || r[i]);
        }
    }

    Ok(Column::with_nulls(
        ColumnData::Boolean(result),
        nulls,
        TypeId::Boolean,
    ))
}

/// Element-wise NOT of a boolean column.
pub fn bool_not(col: &Column) -> Result<Column> {
    let vals = col.as_bools();
    let result: Vec<bool> = vals.iter().map(|v| !v).collect();
    Ok(Column::with_nulls(
        ColumnData::Boolean(result),
        col.nulls.clone(),
        TypeId::Boolean,
    ))
}

/// Element-wise negation of a numeric column. Uses typed fast paths.
pub fn negate(col: &Column) -> Result<Column> {
    match &col.data {
        ColumnData::Int64(v) => {
            let result: Vec<i64> = v.iter().map(|x| x.wrapping_neg()).collect();
            Ok(Column::with_nulls(
                ColumnData::Int64(result),
                col.nulls.clone(),
                col.type_id,
            ))
        }
        ColumnData::Float64(v) => {
            let result: Vec<f64> = v.iter().map(|x| -x).collect();
            Ok(Column::with_nulls(
                ColumnData::Float64(result),
                col.nulls.clone(),
                col.type_id,
            ))
        }
        ColumnData::Int32(v) => {
            let result: Vec<i32> = v.iter().map(|x| x.wrapping_neg()).collect();
            Ok(Column::with_nulls(
                ColumnData::Int32(result),
                col.nulls.clone(),
                col.type_id,
            ))
        }
        ColumnData::Float32(v) => {
            let result: Vec<f32> = v.iter().map(|x| -x).collect();
            Ok(Column::with_nulls(
                ColumnData::Float32(result),
                col.nulls.clone(),
                col.type_id,
            ))
        }
        _ => {
            // Fallback for rare types.
            let len = col.len();
            let mut data = ColumnData::with_capacity(col.type_id, len);
            for i in 0..len {
                if col.is_null(i) {
                    data.push_default();
                    continue;
                }
                let scalar = col.data.get_scalar(i);
                let negated = match scalar {
                    ScalarValue::Int8(v) => ScalarValue::Int8(-v),
                    ScalarValue::Int16(v) => ScalarValue::Int16(-v),
                    ScalarValue::Int128(v) => ScalarValue::Int128(-v),
                    other => other,
                };
                data.push_scalar(&negated);
            }
            Ok(Column::with_nulls(data, col.nulls.clone(), col.type_id))
        }
    }
}

// ---------------------------------------------------------------------------
// IS NULL / IS NOT NULL
// ---------------------------------------------------------------------------

/// Produces a boolean column: true where the input is null.
pub fn is_null(col: &Column) -> Column {
    let len = col.len();
    let result: Vec<bool> = (0..len).map(|i| col.is_null(i)).collect();
    Column::new(ColumnData::Boolean(result), TypeId::Boolean)
}

/// Produces a boolean column: true where the input is not null.
pub fn is_not_null(col: &Column) -> Column {
    let len = col.len();
    let result: Vec<bool> = (0..len).map(|i| !col.is_null(i)).collect();
    Column::new(ColumnData::Boolean(result), TypeId::Boolean)
}

// ---------------------------------------------------------------------------
// LIKE / ILIKE pattern matching
// ---------------------------------------------------------------------------

/// SQL LIKE pattern matching.
pub fn like(col: &Column, pattern: &Column, negated: bool) -> Result<Column> {
    like_impl(col, pattern, negated, false)
}

/// Case-insensitive SQL LIKE pattern matching.
pub fn ilike(col: &Column, pattern: &Column, negated: bool) -> Result<Column> {
    like_impl(col, pattern, negated, true)
}

fn like_impl(
    col: &Column,
    pattern: &Column,
    negated: bool,
    case_insensitive: bool,
) -> Result<Column> {
    let len = col.len();
    let mut result = Vec::with_capacity(len);
    let mut nulls = NullBitmap::none(len);

    for i in 0..len {
        if col.is_null(i) || pattern.is_null(i) {
            nulls.set_null(i);
            result.push(false);
            continue;
        }
        let val = match &col.data {
            ColumnData::Utf8(v) => &v[i],
            _ => {
                result.push(false);
                continue;
            }
        };
        let pat = match &pattern.data {
            ColumnData::Utf8(v) => &v[i],
            _ => {
                result.push(false);
                continue;
            }
        };
        let matched = if case_insensitive {
            sql_like_match(&val.to_lowercase(), &pat.to_lowercase())
        } else {
            sql_like_match(val, pat)
        };
        result.push(if negated { !matched } else { matched });
    }

    Ok(Column::with_nulls(
        ColumnData::Boolean(result),
        nulls,
        TypeId::Boolean,
    ))
}

fn sql_like_match(text: &str, pattern: &str) -> bool {
    let t: Vec<char> = text.chars().collect();
    let p: Vec<char> = pattern.chars().collect();
    sql_like_dp(&t, &p)
}

fn sql_like_dp(text: &[char], pattern: &[char]) -> bool {
    let (m, n) = (text.len(), pattern.len());
    let mut prev = vec![false; n + 1];
    let mut curr = vec![false; n + 1];
    prev[0] = true;
    for j in 1..=n {
        if pattern[j - 1] == '%' {
            prev[j] = prev[j - 1];
        }
    }
    for i in 1..=m {
        curr[0] = false;
        for j in 1..=n {
            let pc = pattern[j - 1];
            if pc == '%' {
                curr[j] = curr[j - 1] || prev[j];
            } else if pc == '_' || pc == text[i - 1] {
                curr[j] = prev[j - 1];
            } else {
                curr[j] = false;
            }
        }
        std::mem::swap(&mut prev, &mut curr);
    }
    prev[n]
}

// ---------------------------------------------------------------------------
// Type casting
// ---------------------------------------------------------------------------

/// Casts a column to a target type.
/// Scales a microsecond (i64) timestamp column up to picoseconds (i128),
/// exact x1_000_000. An already-i128 (ps) column passes through. Nulls and the
/// logical type are preserved; the result carries `ps_precision`. Used to
/// normalize a p<=6 operand to a p>6 operand before compare/arith so the two
/// instants are expressed in the same unit (never the reverse, which would
/// lose information).
pub fn scale_us_to_ps(col: &Column, ps_precision: Option<u8>) -> Result<Column> {
    let len = col.len();
    let data = match &col.data {
        ColumnData::Int64(v) => {
            ColumnData::Int128(v.iter().map(|&x| x as i128 * 1_000_000).collect())
        }
        ColumnData::Int128(v) => ColumnData::Int128(v.clone()),
        _ => {
            return Err(ZyronError::ExecutionError(
                "scale_us_to_ps expects an integer timestamp column".to_string(),
            ));
        }
    };
    debug_assert_eq!(data.len(), len);
    Ok(Column::with_nulls_ts(
        data,
        col.nulls.clone(),
        col.type_id,
        ps_precision,
    ))
}

/// Puts two operands on one decimal scale when either is a decimal.
///
/// Returns None when neither side is one, which is the common case and costs
/// two type comparisons. Returns None too when both already share a scale,
/// so a column compared against another of its own type converts nothing.
///
/// The wider of the two scales wins, so the side with fewer digits is padded
/// rather than the side with more being rounded away.
pub fn align_decimal_operands(left: &Column, right: &Column) -> Result<Option<(Column, Column)>> {
    let left_decimal = left.type_id == TypeId::Decimal;
    let right_decimal = right.type_id == TypeId::Decimal;
    if !left_decimal && !right_decimal {
        return Ok(None);
    }
    let left_scale = left.fractional_digits.unwrap_or(0);
    let right_scale = right.fractional_digits.unwrap_or(0);
    if left_decimal && right_decimal && left_scale == right_scale {
        return Ok(None);
    }
    let target = if left_decimal && right_decimal {
        left_scale.max(right_scale)
    } else if left_decimal {
        left_scale
    } else {
        right_scale
    };
    Ok(Some((
        cast_column_to_decimal(left, target)?,
        cast_column_to_decimal(right, target)?,
    )))
}

/// Converts a column to DECIMAL at a known scale.
///
/// The stored form is an i128 holding the value multiplied by ten to the
/// scale, so this is the only conversion to a decimal that can be complete.
/// A column already at that scale is returned unchanged, and one at another
/// scale is moved onto this one.
pub fn cast_column_to_decimal(col: &Column, scale: u8) -> Result<Column> {
    let len = col.len();
    let mut out: Vec<i128> = Vec::with_capacity(len);
    let source_scale = if col.type_id == TypeId::Decimal {
        Some(col.fractional_digits.unwrap_or(0))
    } else {
        None
    };
    for i in 0..len {
        if col.is_null(i) {
            out.push(0);
            continue;
        }
        let value = match (&col.data, source_scale) {
            // Already a decimal, so only the scale changes
            (ColumnData::Int128(v), Some(from)) => zyron_common::rescale(v[i], from, scale)?,
            _ => match col.data.get_scalar(i) {
                ScalarValue::Utf8(ref s) => zyron_common::parse_decimal(s, scale)?,
                ScalarValue::Float32(f) => {
                    crate::operator::modify::decimal_from_float(f as f64, scale)?
                }
                ScalarValue::Float64(f) => crate::operator::modify::decimal_from_float(f, scale)?,
                other => {
                    let whole = match other {
                        ScalarValue::Int8(v) => v as i128,
                        ScalarValue::Int16(v) => v as i128,
                        ScalarValue::Int32(v) => v as i128,
                        ScalarValue::Int64(v) => v as i128,
                        ScalarValue::Int128(v) => v,
                        ScalarValue::UInt8(v) => v as i128,
                        ScalarValue::UInt16(v) => v as i128,
                        ScalarValue::UInt32(v) => v as i128,
                        ScalarValue::UInt64(v) => v as i128,
                        ScalarValue::Boolean(b) => i128::from(b),
                        ref bad => {
                            return Err(ZyronError::ExecutionError(format!(
                                "cannot cast {bad} to DECIMAL"
                            )));
                        }
                    };
                    whole
                        .checked_mul(zyron_common::decimal::scale_factor(scale)?)
                        .ok_or_else(|| {
                            ZyronError::ExecutionError(format!(
                                "value {whole} overflows a decimal at scale {scale}"
                            ))
                        })?
                }
            },
        };
        out.push(value);
    }
    Ok(Column::with_nulls_ts(
        ColumnData::Int128(out),
        col.nulls.clone(),
        TypeId::Decimal,
        Some(scale),
    ))
}

pub fn cast_column(col: &Column, target: TypeId) -> Result<Column> {
    // An array and a vector are both byte-backed, so a per-scalar cast cannot
    // tell them apart. These conversions read the source column's type as
    // well as the target and are resolved before that point
    if let Some(converted) = cast_array_column(col, target)? {
        return Ok(converted);
    }
    // A decimal's value is an integer scaled by digits the target type alone
    // does not carry, so there is nothing here to cast onto. The column is
    // handed back untouched for the caller that knows the scale to convert,
    // which is the write path against a declared column and the explicit
    // CAST that names its own precision. Casting blind would push a value of
    // one variant into an i128 buffer and store a zero.
    if target == TypeId::Decimal && col.type_id != TypeId::Decimal {
        return Ok(col.clone());
    }
    // A decimal source converts by value. Its i128 holds the value times
    // ten to the column's scale, so handing the raw integer to the scalar
    // cast would move the point by that factor
    if col.type_id == TypeId::Decimal && target != TypeId::Decimal {
        return cast_column_from_decimal(col, target);
    }
    let len = col.len();
    let mut data = ColumnData::with_capacity(target, len);
    let mut nulls = NullBitmap::none(len);

    for i in 0..len {
        if col.is_null(i) {
            nulls.set_null(i);
            data.push_default();
            continue;
        }
        let scalar = col.data.get_scalar(i);
        let casted = cast_scalar(&scalar, target)?;
        data.push_scalar(&casted);
    }

    Ok(Column::with_nulls(data, nulls, target))
}

/// Casts a decimal column to another type by value. Integer targets round
/// half away from zero, matching the rule an assignment losing digits
/// applies. Float targets divide the scale factor out. Text targets render
/// the fixed-point form
fn cast_column_from_decimal(col: &Column, target: TypeId) -> Result<Column> {
    let scale = col.fractional_digits.unwrap_or(0);
    let len = col.len();
    let mut data = ColumnData::with_capacity(target, len);
    let mut nulls = NullBitmap::none(len);
    for i in 0..len {
        if col.is_null(i) {
            nulls.set_null(i);
            data.push_default();
            continue;
        }
        let raw = match col.data.get_scalar(i) {
            ScalarValue::Int128(v) => v,
            other => {
                // A decimal-typed column always stores i128, anything else
                // is a buffer mismatch upstream
                return Err(ZyronError::ExecutionError(format!(
                    "decimal column holds a non-i128 value {other}"
                )));
            }
        };
        let casted = match target {
            TypeId::Float32 | TypeId::Float64 => {
                let value = raw as f64 / 10f64.powi(scale as i32);
                cast_scalar(&ScalarValue::Float64(value), target)?
            }
            TypeId::Char | TypeId::Varchar | TypeId::Text => {
                ScalarValue::Utf8(zyron_common::format_decimal(raw, scale))
            }
            _ => {
                let whole = zyron_common::rescale(raw, scale, 0)?;
                cast_scalar(&ScalarValue::Int128(whole), target)?
            }
        };
        data.push_scalar(&casted);
    }
    Ok(Column::with_nulls(data, nulls, target))
}

/// Conversions between the array encoding and the shapes it converts to.
/// Returns None when neither side is an array, leaving the generic
/// per-scalar cast to answer.
fn cast_array_column(col: &Column, target: TypeId) -> Result<Option<Column>> {
    let from_array = col.type_id == TypeId::Array;
    let from_vector = col.type_id == TypeId::Vector;
    if !from_array && !(from_vector && target == TypeId::Array) {
        return Ok(None);
    }
    if from_array && target == TypeId::Array {
        return Ok(Some(col.clone()));
    }

    let len = col.len();
    let mut nulls = NullBitmap::none(len);
    let text_target = matches!(target, TypeId::Text | TypeId::Varchar | TypeId::Char);
    let mut data = ColumnData::with_capacity(target, len);

    for row in 0..len {
        if col.is_null(row) {
            nulls.set_null(row);
            data.push_default();
            continue;
        }
        let ScalarValue::Binary(bytes) = col.data.get_scalar(row) else {
            nulls.set_null(row);
            data.push_default();
            continue;
        };

        // A vector holds packed f32 with no header, so converting it to an
        // array is a re-encode rather than a reinterpretation
        if from_vector {
            let payloads: Vec<Option<&[u8]>> = bytes.chunks_exact(4).map(|c| Some(c)).collect();
            let encoded = zyron_common::array_value::encode(TypeId::Float32, &payloads);
            data.push_scalar(&ScalarValue::Binary(encoded));
            continue;
        }

        let Some(view) = zyron_common::ArrayView::parse(&bytes) else {
            return Err(ZyronError::ExecutionError(
                "value is not a well-formed array".to_string(),
            ));
        };
        if text_target {
            data.push_scalar(&ScalarValue::Utf8(view.render_text()));
            continue;
        }
        if target == TypeId::Vector {
            data.push_scalar(&ScalarValue::Binary(array_to_vector_bytes(&view)?));
            continue;
        }
        if matches!(target, TypeId::Binary | TypeId::Bytea | TypeId::Varbinary) {
            data.push_scalar(&ScalarValue::Binary(bytes));
            continue;
        }
        return Err(ZyronError::ExecutionError(format!(
            "cannot cast an array to {}",
            target
        )));
    }
    Ok(Some(Column::with_nulls(data, nulls, target)))
}

/// Packs an array's elements as the little-endian f32 sequence a vector
/// column stores. Every element has to be a number and present, because a
/// vector has no null slot and a distance over a missing one is undefined.
fn array_to_vector_bytes(view: &zyron_common::ArrayView<'_>) -> Result<Vec<u8>> {
    let element_type = view.element_type();
    let width = element_type.fixed_size().unwrap_or(0);
    let mut out = Vec::with_capacity(view.len() * 4);
    for (i, element) in view.iter().enumerate() {
        let Some(payload) = element else {
            return Err(ZyronError::ExecutionError(format!(
                "a vector takes no null element, position {} is null",
                i + 1
            )));
        };
        let scalar = if width > 0 {
            crate::batch::decode_fixed_scalar(element_type, payload)
        } else {
            crate::batch::decode_varlen_scalar(element_type, payload)
        };
        let Some(value) = scalar.to_f64() else {
            return Err(ZyronError::ExecutionError(format!(
                "a vector takes numeric elements, position {} is {}",
                i + 1,
                element_type
            )));
        };
        out.extend_from_slice(&(value as f32).to_le_bytes());
    }
    Ok(out)
}

/// Casts a single scalar value to the target type.
pub fn cast_scalar(value: &ScalarValue, target: TypeId) -> Result<ScalarValue> {
    if value.is_null() {
        return Ok(ScalarValue::Null);
    }

    match target {
        TypeId::Int64 => match value {
            ScalarValue::Int8(v) => Ok(ScalarValue::Int64(*v as i64)),
            ScalarValue::Int16(v) => Ok(ScalarValue::Int64(*v as i64)),
            ScalarValue::Int32(v) => Ok(ScalarValue::Int64(*v as i64)),
            ScalarValue::Int64(v) => Ok(ScalarValue::Int64(*v)),
            // Narrowing from the 128-bit width, refused when it would not fit
            // rather than wrapping into a different number
            ScalarValue::Int128(v) => i64::try_from(*v).map(ScalarValue::Int64).map_err(|_| {
                ZyronError::ExecutionError(format!("value {v} is out of range for Int64"))
            }),
            ScalarValue::UInt8(v) => Ok(ScalarValue::Int64(*v as i64)),
            ScalarValue::UInt16(v) => Ok(ScalarValue::Int64(*v as i64)),
            ScalarValue::UInt32(v) => Ok(ScalarValue::Int64(*v as i64)),
            ScalarValue::UInt64(v) => Ok(ScalarValue::Int64(*v as i64)),
            ScalarValue::Float32(v) => Ok(ScalarValue::Int64(*v as i64)),
            ScalarValue::Float64(v) => Ok(ScalarValue::Int64(*v as i64)),
            ScalarValue::Boolean(v) => Ok(ScalarValue::Int64(if *v { 1 } else { 0 })),
            ScalarValue::Utf8(s) => s
                .parse::<i64>()
                .map(ScalarValue::Int64)
                .map_err(|_| ZyronError::ExecutionError(format!("cannot cast '{s}' to Int64"))),
            _ => Err(ZyronError::ExecutionError(format!(
                "cannot cast {value} to Int64"
            ))),
        },
        TypeId::Float64 => match value {
            ScalarValue::Float64(v) => Ok(ScalarValue::Float64(*v)),
            ScalarValue::Float32(v) => Ok(ScalarValue::Float64(*v as f64)),
            ScalarValue::Utf8(s) => s
                .parse::<f64>()
                .map(ScalarValue::Float64)
                .map_err(|_| ZyronError::ExecutionError(format!("cannot cast '{s}' to Float64"))),
            other => match other.to_f64() {
                Some(f) => Ok(ScalarValue::Float64(f)),
                None => Err(ZyronError::ExecutionError(format!(
                    "cannot cast {value} to Float64"
                ))),
            },
        },
        TypeId::Text | TypeId::Varchar | TypeId::Char => Ok(ScalarValue::Utf8(format!("{value}"))),
        TypeId::Boolean => match value {
            ScalarValue::Boolean(v) => Ok(ScalarValue::Boolean(*v)),
            ScalarValue::Int64(v) => Ok(ScalarValue::Boolean(*v != 0)),
            ScalarValue::Utf8(s) => match s.to_lowercase().as_str() {
                "true" | "t" | "1" | "yes" => Ok(ScalarValue::Boolean(true)),
                "false" | "f" | "0" | "no" => Ok(ScalarValue::Boolean(false)),
                _ => Err(ZyronError::ExecutionError(format!(
                    "cannot cast '{s}' to Boolean"
                ))),
            },
            _ => Err(ZyronError::ExecutionError(format!(
                "cannot cast {value} to Boolean"
            ))),
        },
        // Timestamps are stored as i64 microseconds since the Unix epoch.
        // Strings are parsed via the validating ISO-8601/SQL parser; integers
        // are taken as already-microsecond values. (A p>6 column's i64->i128
        // picosecond scaling is layered by the INSERT path.)
        TypeId::Timestamp | TypeId::TimestampTz => match value {
            ScalarValue::Utf8(s) => {
                Ok(ScalarValue::Int64(zyron_common::parse_timestamp_micros(s)?))
            }
            ScalarValue::Int8(v) => Ok(ScalarValue::Int64(*v as i64)),
            ScalarValue::Int16(v) => Ok(ScalarValue::Int64(*v as i64)),
            ScalarValue::Int32(v) => Ok(ScalarValue::Int64(*v as i64)),
            ScalarValue::Int64(v) => Ok(ScalarValue::Int64(*v)),
            // Never silently wrap a 128-bit value into the i64 microsecond
            // slot: a real picosecond instant is ~1e21 and would overflow i64
            // and corrupt the timestamp. An out-of-range value is a hard
            // error (matching the Date arm), an in-range one casts exactly.
            ScalarValue::Int128(v) => i64::try_from(*v).map(ScalarValue::Int64).map_err(|_| {
                ZyronError::ExecutionError(format!(
                    "value {v} is out of range for an i64-microsecond TIMESTAMP \
                     (cast a picosecond value to TIMESTAMP(p>6), not TIMESTAMP)"
                ))
            }),
            _ => Err(ZyronError::ExecutionError(format!(
                "cannot cast {value} to TIMESTAMP"
            ))),
        },
        TypeId::Date => match value {
            ScalarValue::Utf8(s) => Ok(ScalarValue::Int32(zyron_common::parse_date_days(s)?)),
            _ => Ok(ScalarValue::Int32(
                i32::try_from(checked_int(value, "DATE")?).map_err(|_| {
                    ZyronError::ExecutionError(format!("value {value} out of range for DATE"))
                })?,
            )),
        },
        TypeId::Int32 => Ok(ScalarValue::Int32(
            i32::try_from(checked_int(value, "INTEGER")?).map_err(|_| {
                ZyronError::ExecutionError(format!("value {value} out of range for INTEGER"))
            })?,
        )),
        TypeId::Int16 => Ok(ScalarValue::Int16(
            i16::try_from(checked_int(value, "SMALLINT")?).map_err(|_| {
                ZyronError::ExecutionError(format!("value {value} out of range for SMALLINT"))
            })?,
        )),
        TypeId::Int8 => Ok(ScalarValue::Int8(
            i8::try_from(checked_int(value, "TINYINT")?).map_err(|_| {
                ZyronError::ExecutionError(format!("value {value} out of range for TINYINT"))
            })?,
        )),
        TypeId::UInt64 => match value {
            // A full-width u64 above i64::MAX cannot pass through checked_int,
            // so parse the unsigned text form directly before the i128 path.
            ScalarValue::Utf8(s) => s
                .parse::<u64>()
                .map(ScalarValue::UInt64)
                .map_err(|_| ZyronError::ExecutionError(format!("cannot cast '{s}' to UInt64"))),
            ScalarValue::UInt64(v) => Ok(ScalarValue::UInt64(*v)),
            _ => {
                let raw = checked_int(value, "unsigned integer")?;
                if raw < 0 || raw > u64::MAX as i128 {
                    return Err(ZyronError::ExecutionError(format!(
                        "value {value} out of range for UInt64"
                    )));
                }
                Ok(ScalarValue::UInt64(raw as u64))
            }
        },
        TypeId::UInt32 | TypeId::UInt16 | TypeId::UInt8 => {
            let raw = checked_int(value, "unsigned integer")?;
            let max: i128 = match target {
                TypeId::UInt8 => u8::MAX as i128,
                TypeId::UInt16 => u16::MAX as i128,
                _ => u32::MAX as i128,
            };
            if raw < 0 || raw > max {
                return Err(ZyronError::ExecutionError(format!(
                    "value {value} out of range for {target:?}"
                )));
            }
            Ok(match target {
                TypeId::UInt8 => ScalarValue::UInt8(raw as u8),
                TypeId::UInt16 => ScalarValue::UInt16(raw as u16),
                _ => ScalarValue::UInt32(raw as u32),
            })
        }
        TypeId::Int128 | TypeId::UInt128 => match value {
            // Parse the exact decimal text form so a 128-bit value above the
            // i64 range round-trips. Non-text inputs keep widening behavior.
            ScalarValue::Utf8(s) => s.parse::<i128>().map(ScalarValue::Int128).map_err(|_| {
                ZyronError::ExecutionError(format!("cannot cast '{s}' to {target:?}"))
            }),
            ScalarValue::Int128(v) => Ok(ScalarValue::Int128(*v)),
            _ => checked_int(value, "128-bit integer").map(ScalarValue::Int128),
        },
        TypeId::Binary | TypeId::Bytea => match value {
            ScalarValue::Binary(b) => Ok(ScalarValue::Binary(b.clone())),
            ScalarValue::FixedBinary16(b) => Ok(ScalarValue::Binary(b.to_vec())),
            ScalarValue::Utf8(s) => decode_hex(s)
                .map(ScalarValue::Binary)
                .ok_or_else(|| ZyronError::ExecutionError(format!("cannot cast '{s}' to Binary"))),
            _ => Err(ZyronError::ExecutionError(format!(
                "cannot cast {value} to Binary"
            ))),
        },
        TypeId::Uuid => match value {
            ScalarValue::FixedBinary16(b) => Ok(ScalarValue::FixedBinary16(*b)),
            ScalarValue::Binary(b) if b.len() == 16 => {
                let mut out = [0u8; 16];
                out.copy_from_slice(b);
                Ok(ScalarValue::FixedBinary16(out))
            }
            ScalarValue::Utf8(s) => {
                let bytes = decode_hex(s).filter(|b| b.len() == 16).ok_or_else(|| {
                    ZyronError::ExecutionError(format!("cannot cast '{s}' to FixedBinary16"))
                })?;
                let mut out = [0u8; 16];
                out.copy_from_slice(&bytes);
                Ok(ScalarValue::FixedBinary16(out))
            }
            _ => Err(ZyronError::ExecutionError(format!(
                "cannot cast {value} to FixedBinary16"
            ))),
        },
        _ => Ok(value.clone()),
    }
}

/// Decodes a hex string into bytes, ignoring an optional 0x or \x prefix.
/// Returns None when the input has an odd length or a non-hex digit.
fn decode_hex(s: &str) -> Option<Vec<u8>> {
    let trimmed = s
        .strip_prefix("0x")
        .or_else(|| s.strip_prefix("0X"))
        .or_else(|| s.strip_prefix("\\x"))
        .unwrap_or(s);
    if trimmed.len() % 2 != 0 {
        return None;
    }
    let bytes = trimmed.as_bytes();
    let mut out = Vec::with_capacity(trimmed.len() / 2);
    let mut i = 0;
    while i < bytes.len() {
        let hi = (bytes[i] as char).to_digit(16)?;
        let lo = (bytes[i + 1] as char).to_digit(16)?;
        out.push(((hi << 4) | lo) as u8);
        i += 2;
    }
    Some(out)
}

/// Coerces an integer-like scalar to an i128 for range-checked narrowing.
/// Rejects out-of-range narrowing (e.g. a BIGINT value that does not fit an
/// INTEGER column) with an error instead of silently wrapping. Floats must be
/// finite and integral within the target range.
fn checked_int(value: &ScalarValue, target_name: &str) -> Result<i128> {
    let out_of_range =
        || ZyronError::ExecutionError(format!("value {value} out of range for {target_name}"));
    match value {
        ScalarValue::Int8(v) => Ok(*v as i128),
        ScalarValue::Int16(v) => Ok(*v as i128),
        ScalarValue::Int32(v) => Ok(*v as i128),
        ScalarValue::Int64(v) => Ok(*v as i128),
        ScalarValue::Int128(v) => Ok(*v),
        ScalarValue::UInt8(v) => Ok(*v as i128),
        ScalarValue::UInt16(v) => Ok(*v as i128),
        ScalarValue::UInt32(v) => Ok(*v as i128),
        ScalarValue::UInt64(v) => Ok(*v as i128),
        ScalarValue::Boolean(b) => Ok(if *b { 1 } else { 0 }),
        ScalarValue::Float32(f) => {
            let f = *f as f64;
            if !f.is_finite() || f.fract() != 0.0 {
                return Err(out_of_range());
            }
            Ok(f as i128)
        }
        ScalarValue::Float64(f) => {
            if !f.is_finite() || f.fract() != 0.0 {
                return Err(out_of_range());
            }
            Ok(*f as i128)
        }
        ScalarValue::Utf8(s) => s
            .parse::<i128>()
            .map_err(|_| ZyronError::ExecutionError(format!("cannot cast '{s}' to {target_name}"))),
        _ => Err(ZyronError::ExecutionError(format!(
            "cannot cast {value} to {target_name}"
        ))),
    }
}

// ---------------------------------------------------------------------------
// Typed row comparison (for sort_indices)
// ---------------------------------------------------------------------------

/// Compares two values within the same ColumnData directly, without ScalarValue.
#[inline]
fn compare_column_values(data: &ColumnData, a: usize, b: usize) -> Ordering {
    match data {
        ColumnData::Boolean(v) => v[a].cmp(&v[b]),
        ColumnData::Int8(v) => v[a].cmp(&v[b]),
        ColumnData::Int16(v) => v[a].cmp(&v[b]),
        ColumnData::Int32(v) => v[a].cmp(&v[b]),
        ColumnData::Int64(v) => v[a].cmp(&v[b]),
        ColumnData::Int128(v) => v[a].cmp(&v[b]),
        ColumnData::UInt8(v) => v[a].cmp(&v[b]),
        ColumnData::UInt16(v) => v[a].cmp(&v[b]),
        ColumnData::UInt32(v) => v[a].cmp(&v[b]),
        ColumnData::UInt64(v) => v[a].cmp(&v[b]),
        ColumnData::Float32(v) => v[a].partial_cmp(&v[b]).unwrap_or(Ordering::Equal),
        ColumnData::Float64(v) => v[a].partial_cmp(&v[b]).unwrap_or(Ordering::Equal),
        ColumnData::Utf8(v) => v[a].cmp(&v[b]),
        ColumnData::Binary(v) => v[a].cmp(&v[b]),
        ColumnData::FixedBinary16(v) => v[a].cmp(&v[b]),
        ColumnData::Interval(v) => v[a].cmp(&v[b]),
    }
}

/// Compares two rows across multiple sort columns using typed dispatch.
/// No ScalarValue allocation.
#[inline]
pub fn compare_rows_typed(
    columns: &[&Column],
    ascending: &[bool],
    nulls_first: &[bool],
    a: usize,
    b: usize,
) -> Ordering {
    for (i, col) in columns.iter().enumerate() {
        let a_null = col.is_null(a);
        let b_null = col.is_null(b);
        let nf = nulls_first[i];

        match (a_null, b_null) {
            (true, true) => continue,
            (true, false) => {
                return if nf {
                    Ordering::Less
                } else {
                    Ordering::Greater
                };
            }
            (false, true) => {
                return if nf {
                    Ordering::Greater
                } else {
                    Ordering::Less
                };
            }
            (false, false) => {}
        }

        let ord = compare_column_values(&col.data, a, b);
        let ord = if ascending[i] { ord } else { ord.reverse() };
        if ord != Ordering::Equal {
            return ord;
        }
    }
    Ordering::Equal
}

/// Compares two rows when no sort columns contain nulls.
/// Eliminates null bitmap lookups from every comparison call.
#[inline]
fn compare_rows_no_nulls(columns: &[&Column], ascending: &[bool], a: usize, b: usize) -> Ordering {
    for (i, col) in columns.iter().enumerate() {
        let ord = compare_column_values(&col.data, a, b);
        let ord = if ascending[i] { ord } else { ord.reverse() };
        if ord != Ordering::Equal {
            return ord;
        }
    }
    Ordering::Equal
}

// ---------------------------------------------------------------------------
// Sort indices
// ---------------------------------------------------------------------------

/// Sorts indices by a single Ord-typed column without any enum dispatch
/// or function call overhead in the inner comparison loop.
macro_rules! sort_single_ord {
    ($indices:expr, $data:expr, $asc:expr) => {
        if $asc {
            $indices.sort_unstable_by(|&a, &b| $data[a as usize].cmp(&$data[b as usize]));
        } else {
            $indices.sort_unstable_by(|&a, &b| $data[b as usize].cmp(&$data[a as usize]));
        }
    };
}

// ---------------------------------------------------------------------------
// Radix sort (LSD, 8-bit passes, histograms and key range gathered while
// the key buffer is built so no pass re-reads the data just to count)
// ---------------------------------------------------------------------------

/// Per-byte histograms plus key range, filled by the same loop that builds
/// the key buffer. Knowing every histogram up front removes the counting
/// read from each radix pass, and a byte whose histogram is a single bucket
/// is constant across all keys, so its pass is the identity and is skipped.
/// Boxed because the eight histograms are 8KB, too large for a stack local.
struct RadixPrep {
    counts: Box<[[u32; 256]; 8]>,
    min_key: u64,
    max_key: u64,
}

impl RadixPrep {
    fn new() -> Self {
        Self {
            counts: Box::new([[0u32; 256]; 8]),
            min_key: u64::MAX,
            max_key: 0,
        }
    }

    /// Folds one key into the range and all eight byte histograms.
    #[inline]
    fn record(&mut self, key: u64) {
        self.min_key = self.min_key.min(key);
        self.max_key = self.max_key.max(key);
        let c = &mut *self.counts;
        c[0][(key & 0xFF) as usize] += 1;
        c[1][((key >> 8) & 0xFF) as usize] += 1;
        c[2][((key >> 16) & 0xFF) as usize] += 1;
        c[3][((key >> 24) & 0xFF) as usize] += 1;
        c[4][((key >> 32) & 0xFF) as usize] += 1;
        c[5][((key >> 40) & 0xFF) as usize] += 1;
        c[6][((key >> 48) & 0xFF) as usize] += 1;
        c[7][((key >> 56) & 0xFF) as usize] += 1;
    }

    /// Number of low bytes that differ between the smallest and largest
    /// key. Bytes at and above the returned index are identical across
    /// the whole input, so no pass ever needs to sort by them.
    fn needed_bytes(&self) -> usize {
        let diff = self.min_key ^ self.max_key;
        if diff == 0 {
            0
        } else {
            (64 - diff.leading_zeros() as usize + 7) / 8
        }
    }

    /// True when every one of the n keys carries the same value in this
    /// byte position, making its counting-sort pass the identity.
    fn byte_is_constant(&self, byte_idx: usize, n: usize) -> bool {
        self.counts[byte_idx].iter().any(|&c| c as usize == n)
    }

    /// Exclusive prefix sum of the histogram for one byte position.
    fn offsets(&self, byte_idx: usize) -> [u32; 256] {
        let mut offsets = [0u32; 256];
        let mut running = 0u32;
        for (slot, &count) in offsets.iter_mut().zip(self.counts[byte_idx].iter()) {
            *slot = running;
            running += count;
        }
        offsets
    }
}

/// 8-bit LSD radix sort over (key, original_index) pairs that emits its
/// result as separate index and value vectors. The pass over the highest
/// varying byte runs last and scatters straight into the two outputs,
/// applying `untransform` to turn each key back into a column value, so
/// the sort needs no copy-back pass and no separate extraction sweep.
/// Constant bytes below the top varying one are skipped outright.
fn radix_scatter_pairs<T, F>(
    mut pairs: Vec<(u64, u32)>,
    prep: &RadixPrep,
    untransform: F,
) -> (Vec<u32>, Vec<T>)
where
    T: Copy,
    F: Fn(u64) -> T,
{
    let n = pairs.len();
    let needed = prep.needed_bytes();
    let mut indices: Vec<u32> = Vec::with_capacity(n);
    let mut values: Vec<T> = Vec::with_capacity(n);

    if needed == 0 {
        // Every key is identical, so input order is already sorted order.
        for &(key, idx) in pairs.iter() {
            indices.push(idx);
            values.push(untransform(key));
        }
        return (indices, values);
    }

    // Ping-pong passes over the varying bytes below the top one. The first
    // pass scatters into reserved-but-unwritten scratch capacity, later
    // passes alternate between the two fully written buffers.
    let intermediate: Vec<usize> = (0..needed - 1)
        .filter(|&b| !prep.byte_is_constant(b, n))
        .collect();
    let mut scratch: Vec<(u64, u32)> =
        Vec::with_capacity(if intermediate.is_empty() { 0 } else { n });
    let mut src_is_pairs = true;
    for (pass, &byte_idx) in intermediate.iter().enumerate() {
        let shift = (byte_idx * 8) as u32;
        let mut offsets = prep.offsets(byte_idx);
        if pass == 0 {
            let out = scratch.spare_capacity_mut();
            for &p in pairs.iter() {
                let byte = ((p.0 >> shift) & 0xFF) as usize;
                let dest = offsets[byte] as usize;
                offsets[byte] += 1;
                out[dest].write(p);
            }
            // SAFETY: the offsets are exclusive prefix sums of a histogram
            // over exactly these n keys, so the destinations enumerate
            // 0..n with no gap or repeat and every slot was written above.
            // The element type is Copy, so no drop obligations exist.
            unsafe { scratch.set_len(n) };
            src_is_pairs = false;
        } else {
            let (input, output) = if src_is_pairs {
                (pairs.as_slice(), scratch.as_mut_slice())
            } else {
                (scratch.as_slice(), pairs.as_mut_slice())
            };
            for &p in input.iter() {
                let byte = ((p.0 >> shift) & 0xFF) as usize;
                let dest = offsets[byte] as usize;
                offsets[byte] += 1;
                output[dest] = p;
            }
            src_is_pairs = !src_is_pairs;
        }
    }

    // Final pass over the top varying byte writes both outputs directly.
    let byte_idx = needed - 1;
    let shift = (byte_idx * 8) as u32;
    let mut offsets = prep.offsets(byte_idx);
    let input: &[(u64, u32)] = if src_is_pairs { &pairs } else { &scratch };
    {
        let idx_out = indices.spare_capacity_mut();
        let val_out = values.spare_capacity_mut();
        for &(key, idx) in input.iter() {
            let byte = ((key >> shift) & 0xFF) as usize;
            let dest = offsets[byte] as usize;
            offsets[byte] += 1;
            idx_out[dest].write(idx);
            val_out[dest].write(untransform(key));
        }
    }
    // SAFETY: same argument as above, the scatter destinations cover 0..n
    // exactly once, so both vectors are fully written up to n. Both element
    // types are Copy.
    unsafe {
        indices.set_len(n);
        values.set_len(n);
    }
    (indices, values)
}

/// Index-only variant of `radix_scatter_pairs`. The unit value output
/// occupies no memory and its writes compile away, leaving a scatter
/// that emits just the sorted index vector.
fn radix_sort_pair_indices(pairs: Vec<(u64, u32)>, prep: &RadixPrep) -> Vec<u32> {
    radix_scatter_pairs(pairs, prep, |_| ()).0
}

/// Values-only counterpart of `radix_scatter_pairs` for single-column
/// sorts that need no permutation indices. Half the bandwidth of the
/// pair sort, 8 bytes per element per pass instead of 16.
fn radix_scatter_values<T, F>(mut keys: Vec<u64>, prep: &RadixPrep, untransform: F) -> Vec<T>
where
    T: Copy,
    F: Fn(u64) -> T,
{
    let n = keys.len();
    let needed = prep.needed_bytes();
    let mut values: Vec<T> = Vec::with_capacity(n);

    if needed == 0 {
        for &key in keys.iter() {
            values.push(untransform(key));
        }
        return values;
    }

    let intermediate: Vec<usize> = (0..needed - 1)
        .filter(|&b| !prep.byte_is_constant(b, n))
        .collect();
    let mut scratch: Vec<u64> = Vec::with_capacity(if intermediate.is_empty() { 0 } else { n });
    let mut src_is_keys = true;
    for (pass, &byte_idx) in intermediate.iter().enumerate() {
        let shift = (byte_idx * 8) as u32;
        let mut offsets = prep.offsets(byte_idx);
        if pass == 0 {
            let out = scratch.spare_capacity_mut();
            for &key in keys.iter() {
                let byte = ((key >> shift) & 0xFF) as usize;
                let dest = offsets[byte] as usize;
                offsets[byte] += 1;
                out[dest].write(key);
            }
            // SAFETY: exclusive prefix sums over these n keys enumerate
            // destinations 0..n exactly once, so every slot was written.
            // u64 is Copy.
            unsafe { scratch.set_len(n) };
            src_is_keys = false;
        } else {
            let (input, output) = if src_is_keys {
                (keys.as_slice(), scratch.as_mut_slice())
            } else {
                (scratch.as_slice(), keys.as_mut_slice())
            };
            for &key in input.iter() {
                let byte = ((key >> shift) & 0xFF) as usize;
                let dest = offsets[byte] as usize;
                offsets[byte] += 1;
                output[dest] = key;
            }
            src_is_keys = !src_is_keys;
        }
    }

    let byte_idx = needed - 1;
    let shift = (byte_idx * 8) as u32;
    let mut offsets = prep.offsets(byte_idx);
    let input: &[u64] = if src_is_keys { &keys } else { &scratch };
    {
        let out = values.spare_capacity_mut();
        for &key in input.iter() {
            let byte = ((key >> shift) & 0xFF) as usize;
            let dest = offsets[byte] as usize;
            offsets[byte] += 1;
            out[dest].write(untransform(key));
        }
    }
    // SAFETY: same destination-coverage argument, all n slots written.
    // T is Copy.
    unsafe { values.set_len(n) };
    values
}

/// Radix sort for signed integers, returning sorted indices. Truncates
/// each value to its unsigned twin so the key stays within the narrow
/// width, then XORs the narrow sign bit to convert signed order to
/// unsigned order. Sign-extending instead would set every high bit on
/// negatives and sort them after positives. The key range and all byte
/// histograms are gathered in the same loop that builds the pairs.
macro_rules! radix_sort_signed {
    ($data:expr, $asc:expr, $uty:ty, $sign_bit:expr) => {{
        let n = $data.len();
        let mut pairs: Vec<(u64, u32)> = Vec::with_capacity(n);
        let mut prep = RadixPrep::new();
        if $asc {
            for (i, &v) in $data.iter().enumerate() {
                let key = (v as $uty as u64) ^ $sign_bit;
                prep.record(key);
                pairs.push((key, i as u32));
            }
        } else {
            for (i, &v) in $data.iter().enumerate() {
                let key = !((v as $uty as u64) ^ $sign_bit);
                prep.record(key);
                pairs.push((key, i as u32));
            }
        }
        radix_sort_pair_indices(pairs, &prep)
    }};
}

/// Radix sort for unsigned integers, returning sorted indices.
macro_rules! radix_sort_unsigned {
    ($data:expr, $asc:expr) => {{
        let n = $data.len();
        let mut pairs: Vec<(u64, u32)> = Vec::with_capacity(n);
        let mut prep = RadixPrep::new();
        if $asc {
            for (i, &v) in $data.iter().enumerate() {
                let key = v as u64;
                prep.record(key);
                pairs.push((key, i as u32));
            }
        } else {
            for (i, &v) in $data.iter().enumerate() {
                let key = !(v as u64);
                prep.record(key);
                pairs.push((key, i as u32));
            }
        }
        radix_sort_pair_indices(pairs, &prep)
    }};
}

/// Radix sort across multiple column batches, returning both sorted indices
/// and the sorted key column data. Builds pairs directly from unconcatenated
/// batches (no concat memcpy) while gathering the key range and all byte
/// histograms, then lets the final radix pass scatter untransformed values
/// straight into the outputs. Returns None for a non-matching column type.
macro_rules! radix_extract_signed {
    ($batches:expr, $asc:expr, $variant:ident, $ty:ty, $uty:ty, $sign_bit:expr) => {{
        let total: usize = $batches.iter().map(|c| c.len()).sum();
        let mut pairs: Vec<(u64, u32)> = Vec::with_capacity(total);
        let mut prep = RadixPrep::new();
        let mut off = 0u32;
        for col in $batches {
            if let ColumnData::$variant(v) = &col.data {
                if $asc {
                    for (i, &val) in v.iter().enumerate() {
                        let key = (val as $uty as u64) ^ $sign_bit;
                        prep.record(key);
                        pairs.push((key, off + i as u32));
                    }
                } else {
                    for (i, &val) in v.iter().enumerate() {
                        let key = !((val as $uty as u64) ^ $sign_bit);
                        prep.record(key);
                        pairs.push((key, off + i as u32));
                    }
                }
                off += v.len() as u32;
            } else {
                return None;
            }
        }
        let (indices, sorted) = if $asc {
            radix_scatter_pairs(pairs, &prep, |k: u64| (k ^ $sign_bit) as $ty)
        } else {
            radix_scatter_pairs(pairs, &prep, |k: u64| ((!k) ^ $sign_bit) as $ty)
        };
        Some((indices, ColumnData::$variant(sorted)))
    }};
}

macro_rules! radix_extract_unsigned {
    ($batches:expr, $asc:expr, $variant:ident, $ty:ty) => {{
        let total: usize = $batches.iter().map(|c| c.len()).sum();
        let mut pairs: Vec<(u64, u32)> = Vec::with_capacity(total);
        let mut prep = RadixPrep::new();
        let mut off = 0u32;
        for col in $batches {
            if let ColumnData::$variant(v) = &col.data {
                if $asc {
                    for (i, &val) in v.iter().enumerate() {
                        let key = val as u64;
                        prep.record(key);
                        pairs.push((key, off + i as u32));
                    }
                } else {
                    for (i, &val) in v.iter().enumerate() {
                        let key = !(val as u64);
                        prep.record(key);
                        pairs.push((key, off + i as u32));
                    }
                }
                off += v.len() as u32;
            } else {
                return None;
            }
        }
        let (indices, sorted) = if $asc {
            radix_scatter_pairs(pairs, &prep, |k: u64| k as $ty)
        } else {
            radix_scatter_pairs(pairs, &prep, |k: u64| (!k) as $ty)
        };
        Some((indices, ColumnData::$variant(sorted)))
    }};
}

/// Values-only variants for single-column sorts. The key-build loop fuses
/// what were separate concat, sign transform, range scan and per-pass
/// counting reads into one pass over the batches.
macro_rules! radix_values_signed {
    ($batches:expr, $asc:expr, $variant:ident, $ty:ty, $uty:ty, $sign_bit:expr) => {{
        let total: usize = $batches.iter().map(|c| c.len()).sum();
        let mut keys: Vec<u64> = Vec::with_capacity(total);
        let mut prep = RadixPrep::new();
        for col in $batches {
            if let ColumnData::$variant(v) = &col.data {
                if $asc {
                    for &val in v.iter() {
                        let key = (val as $uty as u64) ^ $sign_bit;
                        prep.record(key);
                        keys.push(key);
                    }
                } else {
                    for &val in v.iter() {
                        let key = !((val as $uty as u64) ^ $sign_bit);
                        prep.record(key);
                        keys.push(key);
                    }
                }
            } else {
                return None;
            }
        }
        let sorted = if $asc {
            radix_scatter_values(keys, &prep, |k: u64| (k ^ $sign_bit) as $ty)
        } else {
            radix_scatter_values(keys, &prep, |k: u64| ((!k) ^ $sign_bit) as $ty)
        };
        Some(ColumnData::$variant(sorted))
    }};
}

macro_rules! radix_values_unsigned {
    ($batches:expr, $asc:expr, $variant:ident, $ty:ty) => {{
        let total: usize = $batches.iter().map(|c| c.len()).sum();
        let mut keys: Vec<u64> = Vec::with_capacity(total);
        let mut prep = RadixPrep::new();
        for col in $batches {
            if let ColumnData::$variant(v) = &col.data {
                if $asc {
                    for &val in v.iter() {
                        let key = val as u64;
                        prep.record(key);
                        keys.push(key);
                    }
                } else {
                    for &val in v.iter() {
                        let key = !(val as u64);
                        prep.record(key);
                        keys.push(key);
                    }
                }
            } else {
                return None;
            }
        }
        let sorted = if $asc {
            radix_scatter_values(keys, &prep, |k: u64| k as $ty)
        } else {
            radix_scatter_values(keys, &prep, |k: u64| (!k) as $ty)
        };
        Some(ColumnData::$variant(sorted))
    }};
}

/// Radix sort on integer column data split across multiple batches.
/// Returns (sorted_indices, sorted_column_data). Indices reference positions
/// in the conceptual concatenated column. Sorted column data is extracted
/// directly from radix sort pairs via reverse transform, avoiding
/// the random-access gather that take() would require.
pub fn radix_sort_column_batches(
    batches: &[Column],
    ascending: bool,
) -> Option<(Vec<u32>, ColumnData)> {
    if batches.is_empty() || batches.iter().any(|c| c.nulls.has_nulls()) {
        return None;
    }
    match &batches[0].data {
        ColumnData::Int64(_) => {
            radix_extract_signed!(batches, ascending, Int64, i64, u64, 0x8000_0000_0000_0000u64)
        }
        ColumnData::Int32(_) => {
            radix_extract_signed!(batches, ascending, Int32, i32, u32, 0x8000_0000u64)
        }
        ColumnData::Int16(_) => {
            radix_extract_signed!(batches, ascending, Int16, i16, u16, 0x8000u64)
        }
        ColumnData::Int8(_) => radix_extract_signed!(batches, ascending, Int8, i8, u8, 0x80u64),
        ColumnData::UInt64(_) => radix_extract_unsigned!(batches, ascending, UInt64, u64),
        ColumnData::UInt32(_) => radix_extract_unsigned!(batches, ascending, UInt32, u32),
        ColumnData::UInt16(_) => radix_extract_unsigned!(batches, ascending, UInt16, u16),
        ColumnData::UInt8(_) => radix_extract_unsigned!(batches, ascending, UInt8, u8),
        _ => None,
    }
}

/// Minimum element count for radix sort. Below this threshold,
/// pdqsort (sort_unstable) is faster due to lower constant overhead.
const RADIX_SORT_THRESHOLD: usize = 256;

/// Sorts a single integer key column split across batches, producing the
/// sorted values directly with no permutation indices. One fused loop
/// builds the transformed key buffer while gathering the range and byte
/// histograms, and the final radix pass writes untransformed values
/// straight into the output. Returns None for non-integer types, columns
/// with nulls, or inputs below the radix threshold, where the caller's
/// concat plus comparison sort is the better path.
pub fn radix_sort_batches_values(batches: &[Column], ascending: bool) -> Option<ColumnData> {
    let total: usize = batches.iter().map(|c| c.len()).sum();
    if batches.is_empty()
        || total < RADIX_SORT_THRESHOLD
        || batches.iter().any(|c| c.nulls.has_nulls())
    {
        return None;
    }
    match &batches[0].data {
        ColumnData::Int64(_) => {
            radix_values_signed!(batches, ascending, Int64, i64, u64, 0x8000_0000_0000_0000u64)
        }
        ColumnData::Int32(_) => {
            radix_values_signed!(batches, ascending, Int32, i32, u32, 0x8000_0000u64)
        }
        ColumnData::Int16(_) => {
            radix_values_signed!(batches, ascending, Int16, i16, u16, 0x8000u64)
        }
        ColumnData::Int8(_) => radix_values_signed!(batches, ascending, Int8, i8, u8, 0x80u64),
        ColumnData::UInt64(_) => radix_values_unsigned!(batches, ascending, UInt64, u64),
        ColumnData::UInt32(_) => radix_values_unsigned!(batches, ascending, UInt32, u32),
        ColumnData::UInt16(_) => radix_values_unsigned!(batches, ascending, UInt16, u16),
        ColumnData::UInt8(_) => radix_values_unsigned!(batches, ascending, UInt8, u8),
        _ => None,
    }
}

/// Sorts column data in-place with comparison sorts (pdqsort). This is the
/// fallback behind `radix_sort_batches_values`, serving the non-integer
/// types, integer inputs below the radix threshold, and any type the radix
/// path declines.
pub fn sort_column_inplace(data: &mut ColumnData, ascending: bool) {
    match data {
        ColumnData::Int64(v) => {
            if ascending {
                v.sort_unstable();
            } else {
                v.sort_unstable_by(|a, b| b.cmp(a));
            }
        }
        ColumnData::Int32(v) => {
            if ascending {
                v.sort_unstable();
            } else {
                v.sort_unstable_by(|a, b| b.cmp(a));
            }
        }
        ColumnData::Int16(v) => {
            if ascending {
                v.sort_unstable();
            } else {
                v.sort_unstable_by(|a, b| b.cmp(a));
            }
        }
        ColumnData::Int8(v) => {
            if ascending {
                v.sort_unstable();
            } else {
                v.sort_unstable_by(|a, b| b.cmp(a));
            }
        }
        ColumnData::UInt64(v) => {
            if ascending {
                v.sort_unstable();
            } else {
                v.sort_unstable_by(|a, b| b.cmp(a));
            }
        }
        ColumnData::UInt32(v) => {
            if ascending {
                v.sort_unstable();
            } else {
                v.sort_unstable_by(|a, b| b.cmp(a));
            }
        }
        ColumnData::UInt16(v) => {
            if ascending {
                v.sort_unstable();
            } else {
                v.sort_unstable_by(|a, b| b.cmp(a));
            }
        }
        ColumnData::UInt8(v) => {
            if ascending {
                v.sort_unstable();
            } else {
                v.sort_unstable_by(|a, b| b.cmp(a));
            }
        }
        ColumnData::Float64(v) => {
            if ascending {
                v.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            } else {
                v.sort_unstable_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
            }
        }
        ColumnData::Float32(v) => {
            if ascending {
                v.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            } else {
                v.sort_unstable_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
            }
        }
        ColumnData::Int128(v) => {
            if ascending {
                v.sort_unstable();
            } else {
                v.sort_unstable_by(|a, b| b.cmp(a));
            }
        }
        ColumnData::Boolean(v) => {
            if ascending {
                v.sort_unstable();
            } else {
                v.sort_unstable_by(|a, b| b.cmp(a));
            }
        }
        ColumnData::Utf8(v) => {
            if ascending {
                v.sort_unstable();
            } else {
                v.sort_unstable_by(|a, b| b.cmp(a));
            }
        }
        ColumnData::Binary(v) => {
            if ascending {
                v.sort_unstable();
            } else {
                v.sort_unstable_by(|a, b| b.cmp(a));
            }
        }
        ColumnData::FixedBinary16(v) => {
            if ascending {
                v.sort_unstable();
            } else {
                v.sort_unstable_by(|a, b| b.cmp(a));
            }
        }
        ColumnData::Interval(v) => {
            if ascending {
                v.sort_unstable();
            } else {
                v.sort_unstable_by(|a, b| b.cmp(a));
            }
        }
    }
}

/// Computes sort indices using typed comparison (no ScalarValue allocation).
/// When no sort columns contain nulls, uses a streamlined comparison path
/// that skips all null bitmap lookups. For single-key integer sorts, uses
/// LSD radix sort which is O(w*n) sequential instead of O(n log n) random.
pub fn sort_indices(
    columns: &[&Column],
    ascending: &[bool],
    nulls_first: &[bool],
    num_rows: usize,
) -> Vec<u32> {
    let any_nulls = columns.iter().any(|c| c.nulls.has_nulls());

    // Single-key no-null path: use radix sort for integers,
    // comparison sort for other types.
    if !any_nulls && columns.len() == 1 {
        let asc = ascending[0];
        match &columns[0].data {
            ColumnData::Int64(v) => {
                return radix_sort_signed!(v, asc, u64, 0x8000_0000_0000_0000u64)
            }
            ColumnData::Int32(v) => return radix_sort_signed!(v, asc, u32, 0x8000_0000u64),
            ColumnData::Int16(v) => return radix_sort_signed!(v, asc, u16, 0x8000u64),
            ColumnData::Int8(v) => return radix_sort_signed!(v, asc, u8, 0x80u64),
            ColumnData::UInt64(v) => return radix_sort_unsigned!(v, asc),
            ColumnData::UInt32(v) => return radix_sort_unsigned!(v, asc),
            ColumnData::UInt16(v) => return radix_sort_unsigned!(v, asc),
            ColumnData::UInt8(v) => return radix_sort_unsigned!(v, asc),
            // i128 doesn't fit in u64, fall through to comparison sort.
            _ => {}
        }

        // Comparison sort fallback for non-integer types.
        let mut indices: Vec<u32> = (0..num_rows as u32).collect();
        match &columns[0].data {
            ColumnData::Int128(v) => sort_single_ord!(indices, v, asc),
            ColumnData::Boolean(v) => sort_single_ord!(indices, v, asc),
            ColumnData::Utf8(v) => sort_single_ord!(indices, v, asc),
            ColumnData::Binary(v) => sort_single_ord!(indices, v, asc),
            ColumnData::FixedBinary16(v) => sort_single_ord!(indices, v, asc),
            ColumnData::Float64(v) => {
                if asc {
                    indices.sort_unstable_by(|&a, &b| {
                        v[a as usize]
                            .partial_cmp(&v[b as usize])
                            .unwrap_or(Ordering::Equal)
                    });
                } else {
                    indices.sort_unstable_by(|&a, &b| {
                        v[b as usize]
                            .partial_cmp(&v[a as usize])
                            .unwrap_or(Ordering::Equal)
                    });
                }
            }
            ColumnData::Float32(v) => {
                if asc {
                    indices.sort_unstable_by(|&a, &b| {
                        v[a as usize]
                            .partial_cmp(&v[b as usize])
                            .unwrap_or(Ordering::Equal)
                    });
                } else {
                    indices.sort_unstable_by(|&a, &b| {
                        v[b as usize]
                            .partial_cmp(&v[a as usize])
                            .unwrap_or(Ordering::Equal)
                    });
                }
            }
            // Integer types already handled above by radix sort.
            _ => {
                indices.sort_unstable_by(|&a, &b| {
                    compare_column_values(&columns[0].data, a as usize, b as usize)
                });
                if !asc {
                    indices.reverse();
                }
            }
        }
        return indices;
    }

    let mut indices: Vec<u32> = (0..num_rows as u32).collect();
    if any_nulls {
        indices.sort_unstable_by(|&a, &b| {
            compare_rows_typed(columns, ascending, nulls_first, a as usize, b as usize)
        });
    } else {
        indices.sort_unstable_by(|&a, &b| {
            compare_rows_no_nulls(columns, ascending, a as usize, b as usize)
        });
    }
    indices
}

// ---------------------------------------------------------------------------
// Flat hash table for join build
// ---------------------------------------------------------------------------

/// Flat bucket array for chained hash table, replacing HashMap for join builds.
/// Uses direct `hash & mask` indexing into a power-of-2 bucket array.
/// Each bucket stores the head of a chain in an external entries array.
/// Insert is O(1) (swap bucket head), making this ideal for build-heavy
/// workloads where build rows outnumber probe rows.
pub struct FlatHashTable {
    buckets: Vec<u32>,
    mask: u32,
}

impl FlatHashTable {
    /// Creates a flat hash table sized for the expected number of entries.
    /// Uses 2x capacity for low chain lengths (~0.5 average).
    pub fn with_capacity(expected: usize) -> Self {
        let capacity = (expected * 2).next_power_of_two().max(16);
        Self {
            buckets: vec![u32::MAX; capacity],
            mask: (capacity - 1) as u32,
        }
    }

    /// Inserts a new entry. Returns the previous head of the chain for this
    /// bucket (u32::MAX if empty). The caller stores (value, prev_head)
    /// in their entries array.
    #[inline(always)]
    pub fn insert(&mut self, hash: u64, entry_idx: u32) -> u32 {
        let bucket = (hash as u32) & self.mask;
        let prev = self.buckets[bucket as usize];
        self.buckets[bucket as usize] = entry_idx;
        prev
    }

    /// Returns the head of the chain for the given hash, or u32::MAX if empty.
    #[inline(always)]
    pub fn get(&self, hash: u64) -> u32 {
        self.buckets[(hash as u32 & self.mask) as usize]
    }

    /// Prefetches the cache line for the bucket corresponding to a hash.
    #[inline(always)]
    pub fn prefetch(&self, hash: u64) {
        let bucket = (hash as u32 & self.mask) as usize;
        let ptr = unsafe { self.buckets.as_ptr().add(bucket) };
        #[cfg(target_arch = "x86_64")]
        unsafe {
            std::arch::x86_64::_mm_prefetch(ptr as *const i8, std::arch::x86_64::_MM_HINT_T0);
        }
        #[cfg(target_arch = "x86")]
        unsafe {
            std::arch::x86::_mm_prefetch(ptr as *const i8, std::arch::x86::_MM_HINT_T0);
        }
        #[cfg(not(any(target_arch = "x86_64", target_arch = "x86")))]
        {
            let _ = ptr;
        }
    }
}

// ---------------------------------------------------------------------------
// Fast hash primitives
// ---------------------------------------------------------------------------

/// Golden ratio constant for hash combination.
const HASH_GOLDEN: u64 = 0x9e3779b97f4a7c15;

/// Mixes a value into a hash seed (boost::hash_combine approach).
#[inline(always)]
pub fn hash_combine(seed: u64, value: u64) -> u64 {
    seed ^ (value
        .wrapping_add(HASH_GOLDEN)
        .wrapping_add(seed << 6)
        .wrapping_add(seed >> 2))
}

/// Murmurhash3 64-bit finalizer for output distribution.
#[inline(always)]
pub fn hash_finalize(mut x: u64) -> u64 {
    x ^= x >> 33;
    x = x.wrapping_mul(0xff51afd7ed558ccd);
    x ^= x >> 33;
    x = x.wrapping_mul(0xc4ceb9fe1a85ec53);
    x ^= x >> 33;
    x
}

/// Fibonacci hash for a single integer value. Multiply by the golden ratio
/// constant, then mix high bits into low bits for bucket distribution.
/// Bijection on u64 (distinct inputs produce distinct outputs), so hash
/// equality implies key equality with zero false positives.
/// Used by the fused join path for single integer key columns.
#[inline(always)]
pub fn hash_int(v: u64) -> u64 {
    let h = v.wrapping_mul(HASH_GOLDEN);
    h ^ (h >> 32)
}

/// FNV-1a hash for variable-length byte data.
#[inline]
fn hash_bytes_fnv(bytes: &[u8]) -> u64 {
    let mut h: u64 = 0xcbf29ce484222325;
    for &b in bytes {
        h ^= b as u64;
        h = h.wrapping_mul(0x100000001b3);
    }
    h
}

/// Hashes an integer-like column batch into the hashes array.
/// Hoists the null check outside the per-row loop.
macro_rules! hash_int_column_batch {
    ($hashes:expr, $data:expr, $has_nulls:expr, $nulls:expr, $num_rows:expr) => {
        if $has_nulls {
            for i in 0..$num_rows {
                if $nulls.is_null(i) {
                    $hashes[i] = hash_combine($hashes[i], HASH_GOLDEN);
                } else {
                    $hashes[i] = hash_combine($hashes[i], $data[i] as u64);
                }
            }
        } else {
            for i in 0..$num_rows {
                $hashes[i] = hash_combine($hashes[i], $data[i] as u64);
            }
        }
    };
}

// ---------------------------------------------------------------------------
// Typed row hashing
// ---------------------------------------------------------------------------

/// Computes a hash for a single row across multiple columns.
/// Uses typed dispatch with fast hash combination (no SipHash overhead).
pub fn hash_row(columns: &[&Column], row: usize) -> u64 {
    let mut h = 0u64;
    for col in columns {
        if col.is_null(row) {
            h = hash_combine(h, HASH_GOLDEN);
        } else {
            h = match &col.data {
                ColumnData::Boolean(v) => hash_combine(h, v[row] as u64),
                ColumnData::Int8(v) => hash_combine(h, v[row] as u64),
                ColumnData::Int16(v) => hash_combine(h, v[row] as u64),
                ColumnData::Int32(v) => hash_combine(h, v[row] as u64),
                ColumnData::Int64(v) => hash_combine(h, v[row] as u64),
                ColumnData::Int128(v) => {
                    hash_combine(hash_combine(h, v[row] as u64), (v[row] >> 64) as u64)
                }
                ColumnData::UInt8(v) => hash_combine(h, v[row] as u64),
                ColumnData::UInt16(v) => hash_combine(h, v[row] as u64),
                ColumnData::UInt32(v) => hash_combine(h, v[row] as u64),
                ColumnData::UInt64(v) => hash_combine(h, v[row]),
                ColumnData::Float32(v) => hash_combine(h, v[row].to_bits() as u64),
                ColumnData::Float64(v) => hash_combine(h, v[row].to_bits()),
                ColumnData::Utf8(v) => hash_combine(h, hash_bytes_fnv(v[row].as_bytes())),
                ColumnData::Binary(v) => hash_combine(h, hash_bytes_fnv(&v[row])),
                ColumnData::FixedBinary16(v) => {
                    let lo = u64::from_le_bytes(v[row][0..8].try_into().unwrap());
                    let hi = u64::from_le_bytes(v[row][8..16].try_into().unwrap());
                    hash_combine(hash_combine(h, lo), hi)
                }
                ColumnData::Interval(v) => {
                    let i = v[row];
                    let packed =
                        (i.months as u64) ^ ((i.days as u64) << 32) ^ (i.nanoseconds as u64);
                    hash_combine(h, packed)
                }
            };
        }
    }
    hash_finalize(h)
}

/// Batch-computes hashes for all rows across the given columns.
/// Dispatches type once per column (not per row) and uses fast hash
/// combination instead of SipHash for ~5x throughput improvement.
pub fn hash_column_batch(columns: &[&Column], num_rows: usize) -> Vec<u64> {
    let mut hashes = vec![0u64; num_rows];
    hash_column_batch_into(columns, num_rows, &mut hashes);
    hashes
}

/// Batch-computes hashes into a pre-allocated buffer.
/// Buffer must be at least num_rows long and pre-zeroed for initial hashing,
/// or contain existing hash values to combine with additional columns.
pub fn hash_column_batch_into(columns: &[&Column], num_rows: usize, hashes: &mut [u64]) {
    for col in columns {
        let has_nulls = col.nulls.has_nulls();
        match &col.data {
            ColumnData::Boolean(v) => {
                hash_int_column_batch!(hashes, v, has_nulls, col.nulls, num_rows)
            }
            ColumnData::Int8(v) => {
                hash_int_column_batch!(hashes, v, has_nulls, col.nulls, num_rows)
            }
            ColumnData::Int16(v) => {
                hash_int_column_batch!(hashes, v, has_nulls, col.nulls, num_rows)
            }
            ColumnData::Int32(v) => {
                hash_int_column_batch!(hashes, v, has_nulls, col.nulls, num_rows)
            }
            ColumnData::Int64(v) => {
                hash_int_column_batch!(hashes, v, has_nulls, col.nulls, num_rows)
            }
            ColumnData::Int128(v) => {
                if has_nulls {
                    for i in 0..num_rows {
                        if col.nulls.is_null(i) {
                            hashes[i] = hash_combine(hashes[i], HASH_GOLDEN);
                        } else {
                            hashes[i] = hash_combine(
                                hash_combine(hashes[i], v[i] as u64),
                                (v[i] >> 64) as u64,
                            );
                        }
                    }
                } else {
                    for i in 0..num_rows {
                        hashes[i] =
                            hash_combine(hash_combine(hashes[i], v[i] as u64), (v[i] >> 64) as u64);
                    }
                }
            }
            ColumnData::UInt8(v) => {
                hash_int_column_batch!(hashes, v, has_nulls, col.nulls, num_rows)
            }
            ColumnData::UInt16(v) => {
                hash_int_column_batch!(hashes, v, has_nulls, col.nulls, num_rows)
            }
            ColumnData::UInt32(v) => {
                hash_int_column_batch!(hashes, v, has_nulls, col.nulls, num_rows)
            }
            ColumnData::UInt64(v) => {
                hash_int_column_batch!(hashes, v, has_nulls, col.nulls, num_rows)
            }
            ColumnData::Float32(v) => {
                if has_nulls {
                    for i in 0..num_rows {
                        if col.nulls.is_null(i) {
                            hashes[i] = hash_combine(hashes[i], HASH_GOLDEN);
                        } else {
                            hashes[i] = hash_combine(hashes[i], v[i].to_bits() as u64);
                        }
                    }
                } else {
                    for i in 0..num_rows {
                        hashes[i] = hash_combine(hashes[i], v[i].to_bits() as u64);
                    }
                }
            }
            ColumnData::Float64(v) => {
                if has_nulls {
                    for i in 0..num_rows {
                        if col.nulls.is_null(i) {
                            hashes[i] = hash_combine(hashes[i], HASH_GOLDEN);
                        } else {
                            hashes[i] = hash_combine(hashes[i], v[i].to_bits());
                        }
                    }
                } else {
                    for i in 0..num_rows {
                        hashes[i] = hash_combine(hashes[i], v[i].to_bits());
                    }
                }
            }
            ColumnData::Utf8(v) => {
                if has_nulls {
                    for i in 0..num_rows {
                        if col.nulls.is_null(i) {
                            hashes[i] = hash_combine(hashes[i], HASH_GOLDEN);
                        } else {
                            hashes[i] = hash_combine(hashes[i], hash_bytes_fnv(v[i].as_bytes()));
                        }
                    }
                } else {
                    for i in 0..num_rows {
                        hashes[i] = hash_combine(hashes[i], hash_bytes_fnv(v[i].as_bytes()));
                    }
                }
            }
            ColumnData::Binary(v) => {
                if has_nulls {
                    for i in 0..num_rows {
                        if col.nulls.is_null(i) {
                            hashes[i] = hash_combine(hashes[i], HASH_GOLDEN);
                        } else {
                            hashes[i] = hash_combine(hashes[i], hash_bytes_fnv(&v[i]));
                        }
                    }
                } else {
                    for i in 0..num_rows {
                        hashes[i] = hash_combine(hashes[i], hash_bytes_fnv(&v[i]));
                    }
                }
            }
            ColumnData::FixedBinary16(v) => {
                if has_nulls {
                    for i in 0..num_rows {
                        if col.nulls.is_null(i) {
                            hashes[i] = hash_combine(hashes[i], HASH_GOLDEN);
                        } else {
                            let lo = u64::from_le_bytes(v[i][0..8].try_into().unwrap());
                            let hi = u64::from_le_bytes(v[i][8..16].try_into().unwrap());
                            hashes[i] = hash_combine(hash_combine(hashes[i], lo), hi);
                        }
                    }
                } else {
                    for i in 0..num_rows {
                        let lo = u64::from_le_bytes(v[i][0..8].try_into().unwrap());
                        let hi = u64::from_le_bytes(v[i][8..16].try_into().unwrap());
                        hashes[i] = hash_combine(hash_combine(hashes[i], lo), hi);
                    }
                }
            }
            ColumnData::Interval(v) => {
                let mix = |iv: &zyron_common::Interval| -> u64 {
                    (iv.months as u64) ^ ((iv.days as u64) << 32) ^ (iv.nanoseconds as u64)
                };
                if has_nulls {
                    for i in 0..num_rows {
                        if col.nulls.is_null(i) {
                            hashes[i] = hash_combine(hashes[i], HASH_GOLDEN);
                        } else {
                            hashes[i] = hash_combine(hashes[i], mix(&v[i]));
                        }
                    }
                } else {
                    for i in 0..num_rows {
                        hashes[i] = hash_combine(hashes[i], mix(&v[i]));
                    }
                }
            }
        }
    }

    // Final avalanche pass for good bit distribution in hash tables.
    for h in hashes[..num_rows].iter_mut() {
        *h = hash_finalize(*h);
    }
}

// ---------------------------------------------------------------------------
// Typed row equality
// ---------------------------------------------------------------------------

/// Checks if two values within the same ColumnData are equal at indices a and b.
#[inline]
fn column_values_equal(data: &ColumnData, a: usize, b: usize) -> bool {
    match data {
        ColumnData::Boolean(v) => v[a] == v[b],
        ColumnData::Int8(v) => v[a] == v[b],
        ColumnData::Int16(v) => v[a] == v[b],
        ColumnData::Int32(v) => v[a] == v[b],
        ColumnData::Int64(v) => v[a] == v[b],
        ColumnData::Int128(v) => v[a] == v[b],
        ColumnData::UInt8(v) => v[a] == v[b],
        ColumnData::UInt16(v) => v[a] == v[b],
        ColumnData::UInt32(v) => v[a] == v[b],
        ColumnData::UInt64(v) => v[a] == v[b],
        ColumnData::Float32(v) => v[a].to_bits() == v[b].to_bits(),
        ColumnData::Float64(v) => v[a].to_bits() == v[b].to_bits(),
        ColumnData::Utf8(v) => v[a] == v[b],
        ColumnData::Binary(v) => v[a] == v[b],
        ColumnData::FixedBinary16(v) => v[a] == v[b],
        ColumnData::Interval(v) => v[a] == v[b],
    }
}

/// Compares one value in column a against one value in column b.
/// The two columns hold the same logical type but live in different batches.
/// NULL is never equal to anything including another NULL.
#[inline]
pub fn cross_column_value_equal(a: &Column, a_idx: usize, b: &Column, b_idx: usize) -> bool {
    if a.is_null(a_idx) || b.is_null(b_idx) {
        return false;
    }
    cross_column_data_equal(&a.data, a_idx, &b.data, b_idx)
}

#[inline]
fn cross_column_data_equal(a: &ColumnData, a_idx: usize, b: &ColumnData, b_idx: usize) -> bool {
    match (a, b) {
        (ColumnData::Boolean(x), ColumnData::Boolean(y)) => x[a_idx] == y[b_idx],
        (ColumnData::Int8(x), ColumnData::Int8(y)) => x[a_idx] == y[b_idx],
        (ColumnData::Int16(x), ColumnData::Int16(y)) => x[a_idx] == y[b_idx],
        (ColumnData::Int32(x), ColumnData::Int32(y)) => x[a_idx] == y[b_idx],
        (ColumnData::Int64(x), ColumnData::Int64(y)) => x[a_idx] == y[b_idx],
        (ColumnData::Int128(x), ColumnData::Int128(y)) => x[a_idx] == y[b_idx],
        (ColumnData::UInt8(x), ColumnData::UInt8(y)) => x[a_idx] == y[b_idx],
        (ColumnData::UInt16(x), ColumnData::UInt16(y)) => x[a_idx] == y[b_idx],
        (ColumnData::UInt32(x), ColumnData::UInt32(y)) => x[a_idx] == y[b_idx],
        (ColumnData::UInt64(x), ColumnData::UInt64(y)) => x[a_idx] == y[b_idx],
        (ColumnData::Float32(x), ColumnData::Float32(y)) => {
            x[a_idx].to_bits() == y[b_idx].to_bits()
        }
        (ColumnData::Float64(x), ColumnData::Float64(y)) => {
            x[a_idx].to_bits() == y[b_idx].to_bits()
        }
        (ColumnData::Utf8(x), ColumnData::Utf8(y)) => x[a_idx] == y[b_idx],
        (ColumnData::Binary(x), ColumnData::Binary(y)) => x[a_idx] == y[b_idx],
        (ColumnData::FixedBinary16(x), ColumnData::FixedBinary16(y)) => x[a_idx] == y[b_idx],
        (ColumnData::Interval(x), ColumnData::Interval(y)) => x[a_idx] == y[b_idx],
        // Mixed integer widths compare by value so a probe key narrower or wider
        // than the build key still matches when the numeric value is identical.
        _ => match (numeric_as_i128(a, a_idx), numeric_as_i128(b, b_idx)) {
            (Some(x), Some(y)) => x == y,
            _ => false,
        },
    }
}

#[inline]
fn numeric_as_i128(data: &ColumnData, idx: usize) -> Option<i128> {
    match data {
        ColumnData::Int8(v) => Some(v[idx] as i128),
        ColumnData::Int16(v) => Some(v[idx] as i128),
        ColumnData::Int32(v) => Some(v[idx] as i128),
        ColumnData::Int64(v) => Some(v[idx] as i128),
        ColumnData::Int128(v) => Some(v[idx]),
        ColumnData::UInt8(v) => Some(v[idx] as i128),
        ColumnData::UInt16(v) => Some(v[idx] as i128),
        ColumnData::UInt32(v) => Some(v[idx] as i128),
        ColumnData::UInt64(v) => Some(v[idx] as i128),
        _ => None,
    }
}

/// Checks equality of two rows within the same set of columns.
/// Used by distinct and setop for hash collision resolution.
#[inline]
pub fn rows_equal_typed(columns: &[&Column], row_a: usize, row_b: usize) -> bool {
    for col in columns {
        let a_null = col.is_null(row_a);
        let b_null = col.is_null(row_b);
        if a_null != b_null {
            return false;
        }
        if a_null {
            continue;
        } // both null
        if !column_values_equal(&col.data, row_a, row_b) {
            return false;
        }
    }
    true
}

// ---------------------------------------------------------------------------
// Filter helper
// ---------------------------------------------------------------------------

/// Converts a boolean Column to a Vec<bool> mask for use with filter operations.
pub fn column_to_mask(col: &Column) -> Vec<bool> {
    let bools = col.as_bools();
    if !col.nulls.has_nulls() {
        return bools.to_vec();
    }
    let len = col.len();
    let mut mask = Vec::with_capacity(len);
    for i in 0..len {
        mask.push(!col.is_null(i) && bools[i]);
    }
    mask
}

#[cfg(test)]
mod ps_tests {
    use super::*;
    use crate::column::Column;
    use zyron_common::TypeId;

    #[test]
    fn test_scale_us_to_ps_exact_and_order_preserving() {
        // Microsecond timestamps -> picoseconds, exact x1_000_000.
        let us: Vec<i64> = vec![-5, 0, 1, 1_700_000_000_000_000, 999];
        let col = Column::new(ColumnData::Int64(us.clone()), TypeId::TimestampTz);
        let ps = scale_us_to_ps(&col, Some(9)).unwrap();
        match &ps.data {
            ColumnData::Int128(v) => {
                for (i, &u) in us.iter().enumerate() {
                    assert_eq!(v[i], u as i128 * 1_000_000);
                }
            }
            _ => panic!("expected Int128"),
        }
        assert_eq!(ps.type_id, TypeId::TimestampTz);
        assert_eq!(ps.fractional_digits, Some(9));
        // Already-ps passes through unchanged.
        let again = scale_us_to_ps(&ps, Some(9)).unwrap();
        match (&ps.data, &again.data) {
            (ColumnData::Int128(a), ColumnData::Int128(b)) => assert_eq!(a, b),
            _ => panic!("expected Int128"),
        }
    }

    #[test]
    fn test_arithmetic_int32_columns_preserve_type_and_values() {
        // Int32 is neither the Int64 nor Float64 fast path; column-column math
        // must still produce correct Int32 results, not zeros.
        let l = Column::new(ColumnData::Int32(vec![3, 1, 2]), TypeId::Int32);
        let r = Column::new(ColumnData::Int32(vec![1, 2, 3]), TypeId::Int32);

        let add = arithmetic(&l, &r, ArithOp::Add).unwrap();
        assert_eq!(add.type_id, TypeId::Int32);
        assert!(matches!(add.data, ColumnData::Int32(ref v) if v == &[4, 3, 5]));

        let sub = arithmetic(&r, &l, ArithOp::Sub).unwrap();
        assert!(matches!(sub.data, ColumnData::Int32(ref v) if v == &[-2, 1, 1]));

        let mul = arithmetic(&l, &r, ArithOp::Mul).unwrap();
        assert!(matches!(mul.data, ColumnData::Int32(ref v) if v == &[3, 2, 6]));
    }

    #[test]
    fn test_arithmetic_int16_columns_compute_exactly() {
        // Smaller integer width also routes through the scalar fallback.
        let l = Column::new(ColumnData::Int16(vec![100, -5]), TypeId::Int16);
        let r = Column::new(ColumnData::Int16(vec![25, 5]), TypeId::Int16);
        let add = arithmetic(&l, &r, ArithOp::Add).unwrap();
        assert_eq!(add.type_id, TypeId::Int16);
        assert!(matches!(add.data, ColumnData::Int16(ref v) if v == &[125, 0]));
    }

    #[test]
    fn test_cross_precision_same_instant_compares_equal() {
        // Same wall-clock instant: 1_700_000_000 s. p6 stores us, p9 stores ps.
        let us_instant: i64 = 1_700_000_000_000_000; // microseconds
        let left = Column::new(ColumnData::Int64(vec![us_instant]), TypeId::TimestampTz);
        let right = Column::new_ts(
            ColumnData::Int128(vec![us_instant as i128 * 1_000_000]),
            TypeId::TimestampTz,
            Some(9),
        );
        // Normalize the us side up to ps, then they must be equal.
        let scaled = scale_us_to_ps(&left, Some(9)).unwrap();
        let eq = compare(&scaled, &right, CmpOp::Eq).unwrap();
        match &eq.data {
            ColumnData::Boolean(b) => assert!(b[0], "same instant must compare equal"),
            _ => panic!("expected Boolean"),
        }
    }
}

#[cfg(test)]
mod radix_sort_tests {
    use super::*;
    use crate::column::{Column, NullBitmap};
    use zyron_common::TypeId;

    /// Deterministic 64-bit generator so every run sorts the same data
    fn lcg(state: &mut u64) -> u64 {
        *state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        *state
    }

    macro_rules! check_values_sort {
        ($variant:ident, $ty:ty, $type_id:expr, $make:expr) => {{
            let mut state = 0x9E3779B97F4A7C15u64;
            // Three uneven batches totalling well past the radix threshold
            let lens = [300usize, 257, 144];
            let mut batches: Vec<Column> = Vec::new();
            let mut oracle: Vec<$ty> = Vec::new();
            for len in lens {
                let vals: Vec<$ty> = (0..len).map(|_| $make(lcg(&mut state))).collect();
                oracle.extend_from_slice(&vals);
                batches.push(Column::new(ColumnData::$variant(vals), $type_id));
            }
            for asc in [true, false] {
                let mut expect = oracle.clone();
                expect.sort_unstable();
                if !asc {
                    expect.reverse();
                }
                let sorted = radix_sort_batches_values(&batches, asc)
                    .expect("radix path must accept this input");
                match sorted {
                    ColumnData::$variant(v) => assert_eq!(v, expect),
                    other => panic!("wrong output variant: {:?}", other.len()),
                }
            }
        }};
    }

    #[test]
    fn values_sort_matches_a_comparison_oracle_for_every_integer_type() {
        check_values_sort!(Int64, i64, TypeId::Int64, |r: u64| r as i64);
        check_values_sort!(Int32, i32, TypeId::Int32, |r: u64| r as i32);
        check_values_sort!(Int16, i16, TypeId::Int16, |r: u64| r as i16);
        check_values_sort!(Int8, i8, TypeId::Int8, |r: u64| r as i8);
        check_values_sort!(UInt64, u64, TypeId::UInt64, |r: u64| r);
        check_values_sort!(UInt32, u32, TypeId::UInt32, |r: u64| r as u32);
        check_values_sort!(UInt16, u16, TypeId::UInt16, |r: u64| r as u16);
        check_values_sort!(UInt8, u8, TypeId::UInt8, |r: u64| r as u8);
    }

    #[test]
    fn values_sort_handles_negatives_duplicates_and_narrow_ranges() {
        // Mixed signs with heavy duplication, verifies the sign-bit
        // transform and that skipped constant high bytes stay correct
        let vals: Vec<i64> = (0..600).map(|i| ((i % 7) as i64) - 3).collect();
        let batches = vec![Column::new(ColumnData::Int64(vals.clone()), TypeId::Int64)];
        let mut expect = vals.clone();
        expect.sort_unstable();
        match radix_sort_batches_values(&batches, true).expect("radix accepts i64") {
            ColumnData::Int64(v) => assert_eq!(v, expect),
            _ => panic!("wrong variant"),
        }

        // Values differing only in bytes 0 and 2, byte 1 constant across
        // all keys, exercises the intermediate constant-byte pass skip
        let vals: Vec<i64> = (0..600).map(|i| ((i * 37) % 251) * 0x1_0000 + (i % 13)).collect();
        let batches = vec![Column::new(ColumnData::Int64(vals.clone()), TypeId::Int64)];
        let mut expect = vals.clone();
        expect.sort_unstable();
        match radix_sort_batches_values(&batches, true).expect("radix accepts i64") {
            ColumnData::Int64(v) => assert_eq!(v, expect),
            _ => panic!("wrong variant"),
        }

        // All keys equal, the zero-pass branch must reproduce the input
        let vals: Vec<i64> = vec![42; 600];
        let batches = vec![Column::new(ColumnData::Int64(vals.clone()), TypeId::Int64)];
        match radix_sort_batches_values(&batches, false).expect("radix accepts i64") {
            ColumnData::Int64(v) => assert_eq!(v, vals),
            _ => panic!("wrong variant"),
        }
    }

    #[test]
    fn values_sort_declines_small_inputs_nulls_and_non_integers() {
        let small = vec![Column::new(ColumnData::Int64(vec![3, 1, 2]), TypeId::Int64)];
        assert!(radix_sort_batches_values(&small, true).is_none());

        let mut with_null = Column::new(ColumnData::Int64(vec![0; 600]), TypeId::Int64);
        with_null.nulls = NullBitmap::none(600);
        with_null.nulls.set_null(5);
        assert!(radix_sort_batches_values(&[with_null], true).is_none());

        let floats = vec![Column::new(ColumnData::Float64(vec![0.0; 600]), TypeId::Float64)];
        assert!(radix_sort_batches_values(&floats, true).is_none());
    }

    #[test]
    fn pair_sort_returns_matching_indices_values_and_stable_order() {
        let mut state = 0xDEADBEEFCAFEF00Du64;
        // Duplicate-heavy keys so stability is observable
        let lens = [400usize, 311];
        let mut batches: Vec<Column> = Vec::new();
        let mut concat: Vec<i64> = Vec::new();
        for len in lens {
            let vals: Vec<i64> = (0..len).map(|_| (lcg(&mut state) % 50) as i64 - 25).collect();
            concat.extend_from_slice(&vals);
            batches.push(Column::new(ColumnData::Int64(vals), TypeId::Int64));
        }
        for asc in [true, false] {
            let (indices, sorted) =
                radix_sort_column_batches(&batches, asc).expect("radix accepts i64");
            let sorted = match sorted {
                ColumnData::Int64(v) => v,
                _ => panic!("wrong variant"),
            };
            let mut expect = concat.clone();
            expect.sort_unstable();
            if !asc {
                expect.reverse();
            }
            assert_eq!(sorted, expect);
            // Indices permute the concatenated input onto the sorted values
            assert_eq!(indices.len(), concat.len());
            for (pos, &idx) in indices.iter().enumerate() {
                assert_eq!(concat[idx as usize], sorted[pos]);
            }
            // LSD radix is stable, equal keys keep their original order
            for w in indices.windows(2) {
                if concat[w[0] as usize] == concat[w[1] as usize] {
                    assert!(w[0] < w[1], "equal keys reordered: {} then {}", w[0], w[1]);
                }
            }
        }
    }

    #[test]
    fn sort_indices_radix_path_orders_negatives_correctly() {
        let vals: Vec<i64> = (0..500).map(|i| 250 - i as i64).collect();
        let col = Column::new(ColumnData::Int64(vals.clone()), TypeId::Int64);
        let indices = sort_indices(&[&col], &[true], &[false], vals.len());
        let ordered: Vec<i64> = indices.iter().map(|&i| vals[i as usize]).collect();
        let mut expect = vals.clone();
        expect.sort_unstable();
        assert_eq!(ordered, expect);
    }
}
