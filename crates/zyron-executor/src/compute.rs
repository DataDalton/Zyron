//! Vectorized compute kernels for column operations.
//!
//! Provides filter, comparison, arithmetic, boolean logic, sorting,
//! and hashing operations on typed column vectors. All hot-path kernels
//! use typed dispatch to operate directly on Vec<T> arrays, avoiding
//! ScalarValue intermediaries.

use std::cmp::Ordering;
use std::collections::HashMap;
use std::hash::{BuildHasherDefault, Hasher};

use zyron_common::{Result, TypeId, ZyronError};

use crate::column::{Column, ColumnData, NullBitmap, ScalarValue};

// ---------------------------------------------------------------------------
// Identity hasher for pre-computed u64 hashes
// ---------------------------------------------------------------------------

/// Hasher that uses a pre-computed u64 hash value directly, avoiding
/// double-hashing when the HashMap key is already a well-distributed hash.
#[derive(Default)]
pub struct IdentityHasher(u64);

impl Hasher for IdentityHasher {
    #[inline]
    fn finish(&self) -> u64 {
        self.0
    }

    fn write(&mut self, _bytes: &[u8]) {
        // u64 keys call write_u64 directly.
    }

    #[inline]
    fn write_u64(&mut self, i: u64) {
        self.0 = i;
    }
}

/// HashMap type using pre-computed hash keys without re-hashing.
pub type PreHashMap<K, V> = HashMap<K, V, BuildHasherDefault<IdentityHasher>>;

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
pub fn arithmetic(left: &Column, right: &Column, op: ArithOp) -> Result<Column> {
    let len = left.len();
    if len != right.len() {
        return Err(ZyronError::ExecutionError(
            "arithmetic: column length mismatch".to_string(),
        ));
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
            let result = apply_arith(&l, &r, op)?;
            data.push_scalar(&result);
        }
    }
    Ok(Column::with_nulls(data, nulls, out_type))
}

/// Applies arithmetic on two scalar values.
fn apply_arith(left: &ScalarValue, right: &ScalarValue, op: ArithOp) -> Result<ScalarValue> {
    match (left, right) {
        (ScalarValue::Int64(l), ScalarValue::Int64(r)) => {
            let result = match op {
                ArithOp::Add => l.wrapping_add(*r),
                ArithOp::Sub => l.wrapping_sub(*r),
                ArithOp::Mul => l.wrapping_mul(*r),
                ArithOp::Div => {
                    if *r == 0 {
                        return Err(ZyronError::ExecutionError("division by zero".to_string()));
                    }
                    l / r
                }
                ArithOp::Mod => {
                    if *r == 0 {
                        return Err(ZyronError::ExecutionError("modulo by zero".to_string()));
                    }
                    l % r
                }
            };
            Ok(ScalarValue::Int64(result))
        }
        (ScalarValue::Float64(l), ScalarValue::Float64(r)) => {
            let result = match op {
                ArithOp::Add => l + r,
                ArithOp::Sub => l - r,
                ArithOp::Mul => l * r,
                ArithOp::Div => l / r,
                ArithOp::Mod => l % r,
            };
            Ok(ScalarValue::Float64(result))
        }
        _ => {
            let l = left.to_f64().unwrap_or(0.0);
            let r = right.to_f64().unwrap_or(0.0);
            let result = match op {
                ArithOp::Add => l + r,
                ArithOp::Sub => l - r,
                ArithOp::Mul => l * r,
                ArithOp::Div => l / r,
                ArithOp::Mod => l % r,
            };
            Ok(ScalarValue::Float64(result))
        }
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
pub fn cast_column(col: &Column, target: TypeId) -> Result<Column> {
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
        TypeId::Int32 | TypeId::Date => match value {
            ScalarValue::Int8(v) => Ok(ScalarValue::Int32(*v as i32)),
            ScalarValue::Int16(v) => Ok(ScalarValue::Int32(*v as i32)),
            ScalarValue::Int32(v) => Ok(ScalarValue::Int32(*v)),
            ScalarValue::Int64(v) => Ok(ScalarValue::Int32(*v as i32)),
            ScalarValue::Float64(v) => Ok(ScalarValue::Int32(*v as i32)),
            ScalarValue::Utf8(s) => s
                .parse::<i32>()
                .map(ScalarValue::Int32)
                .map_err(|_| ZyronError::ExecutionError(format!("cannot cast '{s}' to Int32"))),
            _ => Err(ZyronError::ExecutionError(format!(
                "cannot cast {value} to Int32"
            ))),
        },
        _ => Ok(value.clone()),
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
// Radix sort (LSD, 8-bit, 4 passes for 32-bit keys, 8 passes for 64-bit)
// ---------------------------------------------------------------------------

/// 8-bit LSD radix sort on (sort_key, original_index) pairs.
/// Converts O(n log n) random-access comparison sort into O(w * n)
/// sequential passes where w = key_bytes. For 500K i64 values this is
/// 8 sequential passes over contiguous memory vs ~9.5M random accesses.
fn radix_sort_pairs(pairs: &mut [(u64, u32)], scratch: &mut [(u64, u32)]) {
    let n = pairs.len();
    // Determine how many bytes actually vary across keys by finding
    // the differing bits between min and max key values. For signed
    // integers after sign-bit XOR, the high bytes are identical
    // across all positive (or all negative) values, so only the
    // low bytes that carry real variation are radix-sorted.
    let (mut min_key, mut max_key) = (u64::MAX, 0u64);
    for p in pairs.iter() {
        min_key = min_key.min(p.0);
        max_key = max_key.max(p.0);
    }
    let diff = min_key ^ max_key;
    let needed_bytes = if diff == 0 {
        0
    } else {
        (64 - diff.leading_zeros() as usize + 7) / 8
    };

    let mut src = true; // true = pairs is source, false = scratch is source
    for byte_idx in 0..needed_bytes {
        let shift = byte_idx * 8;
        let mut counts = [0u32; 256];

        let (input, output) = if src {
            (pairs as &[(u64, u32)], scratch as &mut [(u64, u32)])
        } else {
            (scratch as &[(u64, u32)], pairs as &mut [(u64, u32)])
        };

        // Count occurrences of each byte value.
        for i in 0..n {
            let byte = ((input[i].0 >> shift) & 0xFF) as usize;
            counts[byte] += 1;
        }

        // Prefix sum to get starting positions.
        let mut offsets = [0u32; 256];
        let mut running = 0u32;
        for i in 0..256 {
            offsets[i] = running;
            running += counts[i];
        }

        // Scatter into output.
        for i in 0..n {
            let byte = ((input[i].0 >> shift) & 0xFF) as usize;
            let dest = offsets[byte] as usize;
            offsets[byte] += 1;
            output[dest] = input[i];
        }

        src = !src;
    }

    // If result ended up in scratch, copy back.
    if !src {
        pairs.copy_from_slice(scratch);
    }
}

/// Radix sort for signed integers. XORs the sign bit to convert signed
/// order to unsigned order, radix sorts, then extracts the indices.
macro_rules! radix_sort_signed {
    ($data:expr, $asc:expr, $sign_bit:expr) => {{
        let n = $data.len();
        let mut pairs: Vec<(u64, u32)> = Vec::with_capacity(n);
        if $asc {
            for (i, &v) in $data.iter().enumerate() {
                pairs.push(((v as u64) ^ $sign_bit, i as u32));
            }
        } else {
            for (i, &v) in $data.iter().enumerate() {
                pairs.push((!((v as u64) ^ $sign_bit), i as u32));
            }
        }
        let mut scratch = vec![(0u64, 0u32); n];
        radix_sort_pairs(&mut pairs, &mut scratch);
        pairs.into_iter().map(|p| p.1).collect()
    }};
}

/// Radix sort for unsigned integers.
macro_rules! radix_sort_unsigned {
    ($data:expr, $asc:expr) => {{
        let n = $data.len();
        let mut pairs: Vec<(u64, u32)> = Vec::with_capacity(n);
        if $asc {
            for (i, &v) in $data.iter().enumerate() {
                pairs.push((v as u64, i as u32));
            }
        } else {
            for (i, &v) in $data.iter().enumerate() {
                pairs.push((!(v as u64), i as u32));
            }
        }
        let mut scratch = vec![(0u64, 0u32); n];
        radix_sort_pairs(&mut pairs, &mut scratch);
        pairs.into_iter().map(|p| p.1).collect()
    }};
}

/// Radix sort across multiple column batches, returning both sorted indices
/// and the sorted key column data. Builds pairs directly from unconcatenated
/// batches (avoids concat memcpy), then extracts sorted values via reverse
/// XOR transform (avoids random-access take/gather). Returns None for
/// non-integer types or columns with nulls.
macro_rules! radix_extract_signed {
    ($batches:expr, $asc:expr, $variant:ident, $ty:ty, $sign_bit:expr) => {{
        let total: usize = $batches.iter().map(|c| c.len()).sum();
        let mut pairs: Vec<(u64, u32)> = Vec::with_capacity(total);
        let mut off = 0u32;
        for col in $batches {
            if let ColumnData::$variant(v) = &col.data {
                if $asc {
                    for (i, &val) in v.iter().enumerate() {
                        pairs.push(((val as u64) ^ $sign_bit, off + i as u32));
                    }
                } else {
                    for (i, &val) in v.iter().enumerate() {
                        pairs.push((!((val as u64) ^ $sign_bit), off + i as u32));
                    }
                }
                off += v.len() as u32;
            } else {
                return None;
            }
        }
        let mut scratch = vec![(0u64, 0u32); total];
        radix_sort_pairs(&mut pairs, &mut scratch);
        let mut indices = Vec::with_capacity(total);
        let mut sorted: Vec<$ty> = Vec::with_capacity(total);
        if $asc {
            for &(key, idx) in &pairs {
                indices.push(idx);
                sorted.push((key ^ $sign_bit) as $ty);
            }
        } else {
            for &(key, idx) in &pairs {
                indices.push(idx);
                sorted.push(((!key) ^ $sign_bit) as $ty);
            }
        }
        Some((indices, ColumnData::$variant(sorted)))
    }};
}

macro_rules! radix_extract_unsigned {
    ($batches:expr, $asc:expr, $variant:ident, $ty:ty) => {{
        let total: usize = $batches.iter().map(|c| c.len()).sum();
        let mut pairs: Vec<(u64, u32)> = Vec::with_capacity(total);
        let mut off = 0u32;
        for col in $batches {
            if let ColumnData::$variant(v) = &col.data {
                if $asc {
                    for (i, &val) in v.iter().enumerate() {
                        pairs.push((val as u64, off + i as u32));
                    }
                } else {
                    for (i, &val) in v.iter().enumerate() {
                        pairs.push((!(val as u64), off + i as u32));
                    }
                }
                off += v.len() as u32;
            } else {
                return None;
            }
        }
        let mut scratch = vec![(0u64, 0u32); total];
        radix_sort_pairs(&mut pairs, &mut scratch);
        let mut indices = Vec::with_capacity(total);
        let mut sorted: Vec<$ty> = Vec::with_capacity(total);
        if $asc {
            for &(key, idx) in &pairs {
                indices.push(idx);
                sorted.push(key as $ty);
            }
        } else {
            for &(key, idx) in &pairs {
                indices.push(idx);
                sorted.push((!key) as $ty);
            }
        }
        Some((indices, ColumnData::$variant(sorted)))
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
            radix_extract_signed!(batches, ascending, Int64, i64, 0x8000_0000_0000_0000u64)
        }
        ColumnData::Int32(_) => {
            radix_extract_signed!(batches, ascending, Int32, i32, 0x8000_0000u64)
        }
        ColumnData::Int16(_) => radix_extract_signed!(batches, ascending, Int16, i16, 0x8000u64),
        ColumnData::Int8(_) => radix_extract_signed!(batches, ascending, Int8, i8, 0x80u64),
        ColumnData::UInt64(_) => radix_extract_unsigned!(batches, ascending, UInt64, u64),
        ColumnData::UInt32(_) => radix_extract_unsigned!(batches, ascending, UInt32, u32),
        ColumnData::UInt16(_) => radix_extract_unsigned!(batches, ascending, UInt16, u16),
        ColumnData::UInt8(_) => radix_extract_unsigned!(batches, ascending, UInt8, u8),
        _ => None,
    }
}

/// 8-bit LSD radix sort on plain u64 values (no index pairs).
/// Half the bandwidth of radix_sort_pairs: 8 bytes per element vs 16.
/// Uses the same byte-skip optimization to only sort varying bytes.
fn radix_sort_u64_values(data: &mut [u64], scratch: &mut [u64]) {
    let n = data.len();
    let (mut min_key, mut max_key) = (u64::MAX, 0u64);
    for &k in data.iter() {
        min_key = min_key.min(k);
        max_key = max_key.max(k);
    }
    let diff = min_key ^ max_key;
    let needed_bytes = if diff == 0 {
        0
    } else {
        (64 - diff.leading_zeros() as usize + 7) / 8
    };

    let mut src_is_data = true;
    for byte_idx in 0..needed_bytes {
        let shift = byte_idx * 8;
        let mut counts = [0u32; 256];

        let (input, output): (&[u64], &mut [u64]) = if src_is_data {
            (data as &[u64], scratch)
        } else {
            (scratch as &[u64], data)
        };

        for i in 0..n {
            counts[((input[i] >> shift) & 0xFF) as usize] += 1;
        }

        let mut offsets = [0u32; 256];
        let mut running = 0u32;
        for i in 0..256 {
            offsets[i] = running;
            running += counts[i];
        }

        for i in 0..n {
            let byte = ((input[i] >> shift) & 0xFF) as usize;
            let dest = offsets[byte] as usize;
            offsets[byte] += 1;
            output[dest] = input[i];
        }

        src_is_data = !src_is_data;
    }

    if !src_is_data {
        data.copy_from_slice(scratch);
    }
}

/// Radix sort for i64 slices in-place. Reinterprets as &mut [u64] which
/// is safe because i64 and u64 have identical size, alignment, and all
/// bit patterns are valid for both types.
fn radix_sort_i64_inplace(data: &mut [i64], ascending: bool) {
    let keys: &mut [u64] =
        unsafe { std::slice::from_raw_parts_mut(data.as_mut_ptr() as *mut u64, data.len()) };
    const SIGN_BIT: u64 = 0x8000_0000_0000_0000;
    if ascending {
        for k in keys.iter_mut() {
            *k ^= SIGN_BIT;
        }
    } else {
        for k in keys.iter_mut() {
            *k = !(*k ^ SIGN_BIT);
        }
    }
    let mut scratch = vec![0u64; keys.len()];
    radix_sort_u64_values(keys, &mut scratch);
    if ascending {
        for k in keys.iter_mut() {
            *k ^= SIGN_BIT;
        }
    } else {
        for k in keys.iter_mut() {
            *k = (!*k) ^ SIGN_BIT;
        }
    }
}

/// Radix sort for u64 slices in-place.
fn radix_sort_u64_inplace(data: &mut [u64], ascending: bool) {
    if !ascending {
        for k in data.iter_mut() {
            *k = !*k;
        }
    }
    let mut scratch = vec![0u64; data.len()];
    radix_sort_u64_values(data, &mut scratch);
    if !ascending {
        for k in data.iter_mut() {
            *k = !*k;
        }
    }
}

/// Radix sort for signed integer types smaller than 64 bits.
/// Widens to Vec<u64>, sorts, then writes back. This avoids the unsound
/// transmute that would create a u64 slice from a smaller-typed buffer.
macro_rules! radix_sort_signed_widened {
    ($data:expr, $asc:expr, $sign_bit:expr) => {{
        let n = $data.len();
        let mut keys: Vec<u64> = if $asc {
            $data.iter().map(|&v| (v as u64) ^ $sign_bit).collect()
        } else {
            $data.iter().map(|&v| !((v as u64) ^ $sign_bit)).collect()
        };
        let mut scratch = vec![0u64; n];
        radix_sort_u64_values(&mut keys, &mut scratch);
        if $asc {
            for (i, &k) in keys.iter().enumerate() {
                $data[i] = (k ^ $sign_bit) as _;
            }
        } else {
            for (i, &k) in keys.iter().enumerate() {
                $data[i] = ((!k) ^ $sign_bit) as _;
            }
        }
    }};
}

/// Radix sort for unsigned integer types smaller than 64 bits.
macro_rules! radix_sort_unsigned_widened {
    ($data:expr, $asc:expr) => {{
        let n = $data.len();
        let mut keys: Vec<u64> = if $asc {
            $data.iter().map(|&v| v as u64).collect()
        } else {
            $data.iter().map(|&v| !(v as u64)).collect()
        };
        let mut scratch = vec![0u64; n];
        radix_sort_u64_values(&mut keys, &mut scratch);
        if $asc {
            for (i, &k) in keys.iter().enumerate() {
                $data[i] = k as _;
            }
        } else {
            for (i, &k) in keys.iter().enumerate() {
                $data[i] = (!k) as _;
            }
        }
    }};
}

/// Minimum element count for radix sort. Below this threshold,
/// pdqsort (sort_unstable) is faster due to lower constant overhead.
const RADIX_SORT_THRESHOLD: usize = 256;

/// Sorts column data in-place. For integer types >= 256 elements, uses
/// values-only radix sort (O(w*n) sequential, half the bandwidth of
/// pair-based radix sort). For smaller arrays and non-integer types,
/// falls back to sort_unstable (pdqsort).
pub fn sort_column_inplace(data: &mut ColumnData, ascending: bool) {
    match data {
        ColumnData::Int64(v) if v.len() >= RADIX_SORT_THRESHOLD => {
            radix_sort_i64_inplace(v, ascending);
        }
        ColumnData::Int64(v) => {
            if ascending {
                v.sort_unstable();
            } else {
                v.sort_unstable_by(|a, b| b.cmp(a));
            }
        }
        ColumnData::Int32(v) if v.len() >= RADIX_SORT_THRESHOLD => {
            radix_sort_signed_widened!(v, ascending, 0x8000_0000u64);
        }
        ColumnData::Int32(v) => {
            if ascending {
                v.sort_unstable();
            } else {
                v.sort_unstable_by(|a, b| b.cmp(a));
            }
        }
        ColumnData::Int16(v) if v.len() >= RADIX_SORT_THRESHOLD => {
            radix_sort_signed_widened!(v, ascending, 0x8000u64);
        }
        ColumnData::Int16(v) => {
            if ascending {
                v.sort_unstable();
            } else {
                v.sort_unstable_by(|a, b| b.cmp(a));
            }
        }
        ColumnData::Int8(v) if v.len() >= RADIX_SORT_THRESHOLD => {
            radix_sort_signed_widened!(v, ascending, 0x80u64);
        }
        ColumnData::Int8(v) => {
            if ascending {
                v.sort_unstable();
            } else {
                v.sort_unstable_by(|a, b| b.cmp(a));
            }
        }
        ColumnData::UInt64(v) if v.len() >= RADIX_SORT_THRESHOLD => {
            radix_sort_u64_inplace(v, ascending);
        }
        ColumnData::UInt64(v) => {
            if ascending {
                v.sort_unstable();
            } else {
                v.sort_unstable_by(|a, b| b.cmp(a));
            }
        }
        ColumnData::UInt32(v) if v.len() >= RADIX_SORT_THRESHOLD => {
            radix_sort_unsigned_widened!(v, ascending);
        }
        ColumnData::UInt32(v) => {
            if ascending {
                v.sort_unstable();
            } else {
                v.sort_unstable_by(|a, b| b.cmp(a));
            }
        }
        ColumnData::UInt16(v) if v.len() >= RADIX_SORT_THRESHOLD => {
            radix_sort_unsigned_widened!(v, ascending);
        }
        ColumnData::UInt16(v) => {
            if ascending {
                v.sort_unstable();
            } else {
                v.sort_unstable_by(|a, b| b.cmp(a));
            }
        }
        ColumnData::UInt8(v) if v.len() >= RADIX_SORT_THRESHOLD => {
            radix_sort_unsigned_widened!(v, ascending);
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
    }
}

/// Radix sort on a single already-concatenated column. Returns both sorted
/// indices and the sorted column data. Same as radix_sort_column_batches
/// but takes contiguous data (better cache behavior during pair construction).
pub fn radix_sort_contiguous(col: &Column, ascending: bool) -> Option<(Vec<u32>, ColumnData)> {
    if col.nulls.has_nulls() {
        return None;
    }
    // Wrap in a single-element slice to reuse the batch macros.
    let batches = std::slice::from_ref(col);
    match &col.data {
        ColumnData::Int64(_) => {
            radix_extract_signed!(batches, ascending, Int64, i64, 0x8000_0000_0000_0000u64)
        }
        ColumnData::Int32(_) => {
            radix_extract_signed!(batches, ascending, Int32, i32, 0x8000_0000u64)
        }
        ColumnData::Int16(_) => radix_extract_signed!(batches, ascending, Int16, i16, 0x8000u64),
        ColumnData::Int8(_) => radix_extract_signed!(batches, ascending, Int8, i8, 0x80u64),
        ColumnData::UInt64(_) => radix_extract_unsigned!(batches, ascending, UInt64, u64),
        ColumnData::UInt32(_) => radix_extract_unsigned!(batches, ascending, UInt32, u32),
        ColumnData::UInt16(_) => radix_extract_unsigned!(batches, ascending, UInt16, u16),
        ColumnData::UInt8(_) => radix_extract_unsigned!(batches, ascending, UInt8, u8),
        _ => None,
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
            ColumnData::Int64(v) => return radix_sort_signed!(v, asc, 0x8000_0000_0000_0000u64),
            ColumnData::Int32(v) => return radix_sort_signed!(v, asc, 0x8000_0000u64),
            ColumnData::Int16(v) => return radix_sort_signed!(v, asc, 0x8000u64),
            ColumnData::Int8(v) => return radix_sort_signed!(v, asc, 0x80u64),
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
