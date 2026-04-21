// -----------------------------------------------------------------------------
// Row codec and expression evaluator
// -----------------------------------------------------------------------------
//
// Self-contained NSM row decoder/encoder and an ExprSpec evaluator used by
// the streaming job runner. The layout matches zyron_executor::batch and
// zyron_storage::heap: a null bitmap followed by column values in ordinal
// order. Fixed-size types are inline at TypeId::fixed_size() bytes, variable
// length types carry a 4-byte little-endian length prefix.
//
// The evaluator produces StreamValue, a minimal scalar type covering the
// cases streaming jobs need: Null, Bool, I64, F64, I128, Utf8, and Binary.
// Integer widening to i64 is done at decode time so BinaryOp can operate
// on a uniform type.

use zyron_common::{Result, TypeId, ZyronError};

use crate::job_runner::{BinaryOpKind, ExprSpec};

// -----------------------------------------------------------------------------
// StreamValue
// -----------------------------------------------------------------------------

/// Minimal scalar type used by the runner's expression evaluator. Wider types
/// (i32, i16, u32, etc) fold to I64. Decimal and Int128 fold to I128.
#[derive(Debug, Clone)]
pub enum StreamValue {
    Null,
    Bool(bool),
    I64(i64),
    I128(i128),
    F64(f64),
    Utf8(String),
    Binary(Vec<u8>),
}

impl StreamValue {
    /// Returns the value as a boolean, treating Null as false. Used by the
    /// predicate pipeline. Non-boolean types yield an error.
    pub fn as_bool(&self) -> Result<bool> {
        match self {
            StreamValue::Bool(b) => Ok(*b),
            StreamValue::Null => Ok(false),
            _ => Err(ZyronError::StreamingError(
                "predicate did not evaluate to a boolean".to_string(),
            )),
        }
    }

    /// Returns the value as i64 if representable, else an error.
    pub fn as_i64(&self) -> Result<i64> {
        match self {
            StreamValue::I64(v) => Ok(*v),
            StreamValue::I128(v) => i64::try_from(*v).map_err(|_| {
                ZyronError::StreamingError("i128 value does not fit in i64".to_string())
            }),
            _ => Err(ZyronError::StreamingError(format!(
                "expected integer, got {self:?}"
            ))),
        }
    }

    /// Returns the value as f64 if numeric, else an error.
    pub fn as_f64(&self) -> Result<f64> {
        match self {
            StreamValue::F64(v) => Ok(*v),
            StreamValue::I64(v) => Ok(*v as f64),
            StreamValue::I128(v) => Ok(*v as f64),
            _ => Err(ZyronError::StreamingError(format!(
                "expected number, got {self:?}"
            ))),
        }
    }
}

// -----------------------------------------------------------------------------
// Decode
// -----------------------------------------------------------------------------

/// Decodes an NSM tuple payload into a vector of StreamValue. The tuple layout
/// must match the heap encoding produced by zyron_executor::batch::encode_row.
pub fn decode_row(data: &[u8], types: &[TypeId]) -> Result<Vec<StreamValue>> {
    let num_cols = types.len();
    let bitmap_len = num_cols.div_ceil(8);
    if data.len() < bitmap_len {
        return Err(ZyronError::StreamingError(
            "row payload shorter than null bitmap".to_string(),
        ));
    }
    let bitmap = &data[..bitmap_len];
    let mut off = bitmap_len;
    let mut out = Vec::with_capacity(num_cols);

    for (i, ty) in types.iter().enumerate() {
        let is_null = (bitmap[i / 8] >> (i % 8)) & 1 == 1;
        if let Some(size) = ty.fixed_size() {
            if off + size > data.len() {
                return Err(ZyronError::StreamingError(format!(
                    "row payload truncated at column {i}"
                )));
            }
            if is_null {
                out.push(StreamValue::Null);
            } else {
                out.push(decode_fixed(*ty, &data[off..off + size])?);
            }
            off += size;
        } else {
            if off + 4 > data.len() {
                return Err(ZyronError::StreamingError(format!(
                    "missing varlen length prefix at column {i}"
                )));
            }
            let len = u32::from_le_bytes([data[off], data[off + 1], data[off + 2], data[off + 3]])
                as usize;
            off += 4;
            if off + len > data.len() {
                return Err(ZyronError::StreamingError(format!(
                    "varlen payload truncated at column {i}"
                )));
            }
            if is_null {
                out.push(StreamValue::Null);
            } else {
                out.push(decode_varlen(*ty, &data[off..off + len])?);
            }
            off += len;
        }
    }
    Ok(out)
}

fn decode_fixed(ty: TypeId, bytes: &[u8]) -> Result<StreamValue> {
    Ok(match ty {
        TypeId::Null => StreamValue::Null,
        TypeId::Boolean => StreamValue::Bool(bytes[0] != 0),
        TypeId::Int8 => StreamValue::I64(i8::from_le_bytes([bytes[0]]) as i64),
        TypeId::Int16 => {
            StreamValue::I64(i16::from_le_bytes(bytes[..2].try_into().unwrap()) as i64)
        }
        TypeId::Int32 | TypeId::Date => {
            StreamValue::I64(i32::from_le_bytes(bytes[..4].try_into().unwrap()) as i64)
        }
        TypeId::Int64 | TypeId::Time | TypeId::Timestamp | TypeId::TimestampTz => {
            StreamValue::I64(i64::from_le_bytes(bytes[..8].try_into().unwrap()))
        }
        TypeId::Int128 | TypeId::Decimal | TypeId::UInt128 => {
            StreamValue::I128(i128::from_le_bytes(bytes[..16].try_into().unwrap()))
        }
        TypeId::UInt8 => StreamValue::I64(bytes[0] as i64),
        TypeId::UInt16 => {
            StreamValue::I64(u16::from_le_bytes(bytes[..2].try_into().unwrap()) as i64)
        }
        TypeId::UInt32 => {
            StreamValue::I64(u32::from_le_bytes(bytes[..4].try_into().unwrap()) as i64)
        }
        TypeId::UInt64 => {
            StreamValue::I64(u64::from_le_bytes(bytes[..8].try_into().unwrap()) as i64)
        }
        TypeId::Float32 => {
            StreamValue::F64(f32::from_le_bytes(bytes[..4].try_into().unwrap()) as f64)
        }
        TypeId::Float64 => StreamValue::F64(f64::from_le_bytes(bytes[..8].try_into().unwrap())),
        TypeId::Uuid => StreamValue::Binary(bytes[..16].to_vec()),
        TypeId::Interval => StreamValue::Binary(bytes[..16].to_vec()),
        _ => {
            return Err(ZyronError::StreamingError(format!(
                "streaming runner cannot decode fixed type {ty:?}"
            )));
        }
    })
}

fn decode_varlen(ty: TypeId, bytes: &[u8]) -> Result<StreamValue> {
    Ok(match ty {
        TypeId::Char | TypeId::Varchar | TypeId::Text | TypeId::Json | TypeId::Jsonb => {
            StreamValue::Utf8(String::from_utf8_lossy(bytes).into_owned())
        }
        TypeId::Binary
        | TypeId::Varbinary
        | TypeId::Bytea
        | TypeId::Array
        | TypeId::Composite
        | TypeId::Vector => StreamValue::Binary(bytes.to_vec()),
        _ => StreamValue::Binary(bytes.to_vec()),
    })
}

// -----------------------------------------------------------------------------
// Encode
// -----------------------------------------------------------------------------

/// Encodes a vector of StreamValue back into the NSM tuple layout, widening
/// or narrowing to match the target column types. Returns an error if a
/// value does not fit its declared type.
pub fn encode_row(values: &[StreamValue], types: &[TypeId]) -> Result<Vec<u8>> {
    if values.len() != types.len() {
        return Err(ZyronError::StreamingError(format!(
            "projection arity mismatch: got {} values, expected {}",
            values.len(),
            types.len()
        )));
    }
    let num_cols = types.len();
    let bitmap_len = num_cols.div_ceil(8);
    let mut buf = vec![0u8; bitmap_len];

    for (i, ty) in types.iter().enumerate() {
        let v = &values[i];
        let is_null = matches!(v, StreamValue::Null);
        if is_null {
            buf[i / 8] |= 1 << (i % 8);
        }
        if let Some(size) = ty.fixed_size() {
            if is_null {
                buf.extend(std::iter::repeat_n(0u8, size));
            } else {
                encode_fixed(&mut buf, *ty, v)?;
            }
        } else {
            let body = if is_null {
                Vec::new()
            } else {
                encode_varlen(*ty, v)?
            };
            buf.extend_from_slice(&(body.len() as u32).to_le_bytes());
            buf.extend_from_slice(&body);
        }
    }
    Ok(buf)
}

fn encode_fixed(buf: &mut Vec<u8>, ty: TypeId, v: &StreamValue) -> Result<()> {
    match (ty, v) {
        (TypeId::Boolean, StreamValue::Bool(b)) => {
            buf.push(if *b { 1 } else { 0 });
        }
        (TypeId::Int8, _) => buf.extend_from_slice(&(v.as_i64()? as i8).to_le_bytes()),
        (TypeId::Int16, _) => buf.extend_from_slice(&(v.as_i64()? as i16).to_le_bytes()),
        (TypeId::Int32, _) | (TypeId::Date, _) => {
            buf.extend_from_slice(&(v.as_i64()? as i32).to_le_bytes())
        }
        (TypeId::Int64, _)
        | (TypeId::Time, _)
        | (TypeId::Timestamp, _)
        | (TypeId::TimestampTz, _) => buf.extend_from_slice(&v.as_i64()?.to_le_bytes()),
        (TypeId::Int128, _) | (TypeId::Decimal, _) | (TypeId::UInt128, _) => {
            let iv = match v {
                StreamValue::I128(x) => *x,
                StreamValue::I64(x) => *x as i128,
                _ => {
                    return Err(ZyronError::StreamingError(format!(
                        "cannot encode {v:?} as {ty:?}"
                    )));
                }
            };
            buf.extend_from_slice(&iv.to_le_bytes());
        }
        (TypeId::UInt8, _) => buf.push(v.as_i64()? as u8),
        (TypeId::UInt16, _) => buf.extend_from_slice(&(v.as_i64()? as u16).to_le_bytes()),
        (TypeId::UInt32, _) => buf.extend_from_slice(&(v.as_i64()? as u32).to_le_bytes()),
        (TypeId::UInt64, _) => buf.extend_from_slice(&(v.as_i64()? as u64).to_le_bytes()),
        (TypeId::Float32, _) => buf.extend_from_slice(&(v.as_f64()? as f32).to_le_bytes()),
        (TypeId::Float64, _) => buf.extend_from_slice(&v.as_f64()?.to_le_bytes()),
        (TypeId::Uuid, StreamValue::Binary(b)) | (TypeId::Interval, StreamValue::Binary(b))
            if b.len() == 16 =>
        {
            buf.extend_from_slice(b);
        }
        _ => {
            return Err(ZyronError::StreamingError(format!(
                "cannot encode {v:?} as {ty:?}"
            )));
        }
    }
    Ok(())
}

fn encode_varlen(ty: TypeId, v: &StreamValue) -> Result<Vec<u8>> {
    match (ty, v) {
        (TypeId::Char, StreamValue::Utf8(s))
        | (TypeId::Varchar, StreamValue::Utf8(s))
        | (TypeId::Text, StreamValue::Utf8(s))
        | (TypeId::Json, StreamValue::Utf8(s))
        | (TypeId::Jsonb, StreamValue::Utf8(s)) => Ok(s.as_bytes().to_vec()),
        (TypeId::Binary, StreamValue::Binary(b))
        | (TypeId::Varbinary, StreamValue::Binary(b))
        | (TypeId::Bytea, StreamValue::Binary(b))
        | (TypeId::Array, StreamValue::Binary(b))
        | (TypeId::Composite, StreamValue::Binary(b))
        | (TypeId::Vector, StreamValue::Binary(b)) => Ok(b.clone()),
        _ => Err(ZyronError::StreamingError(format!(
            "cannot encode {v:?} as varlen {ty:?}"
        ))),
    }
}

// -----------------------------------------------------------------------------
// Expression evaluator
// -----------------------------------------------------------------------------

/// Evaluates an ExprSpec against a decoded source row. Returns the computed
/// StreamValue. Handles Null propagation: any operand that is Null produces
/// Null except in boolean AND/OR which follow SQL three-valued logic.
pub fn eval_expr(expr: &ExprSpec, row: &[StreamValue]) -> Result<StreamValue> {
    match expr {
        ExprSpec::LiteralBool(b) => Ok(StreamValue::Bool(*b)),
        ExprSpec::LiteralI64(v) => Ok(StreamValue::I64(*v)),
        ExprSpec::LiteralF64(v) => Ok(StreamValue::F64(*v)),
        ExprSpec::LiteralString(s) => Ok(StreamValue::Utf8(s.clone())),
        ExprSpec::ColumnRef { ordinal } => row.get(*ordinal as usize).cloned().ok_or_else(|| {
            ZyronError::StreamingError(format!(
                "column ordinal {} out of range (row has {} cols)",
                ordinal,
                row.len()
            ))
        }),
        ExprSpec::Not(inner) => {
            let v = eval_expr(inner, row)?;
            match v {
                StreamValue::Null => Ok(StreamValue::Null),
                StreamValue::Bool(b) => Ok(StreamValue::Bool(!b)),
                _ => Err(ZyronError::StreamingError(
                    "NOT applied to non-boolean".to_string(),
                )),
            }
        }
        ExprSpec::BinaryOp { op, left, right } => {
            let l = eval_expr(left, row)?;
            let r = eval_expr(right, row)?;
            eval_binary(*op, &l, &r)
        }
    }
}

fn eval_binary(op: BinaryOpKind, l: &StreamValue, r: &StreamValue) -> Result<StreamValue> {
    // AND and OR implement SQL three-valued logic: they absorb Null only when
    // the other operand does not short-circuit the answer.
    if matches!(op, BinaryOpKind::And) {
        return match (l, r) {
            (StreamValue::Bool(false), _) | (_, StreamValue::Bool(false)) => {
                Ok(StreamValue::Bool(false))
            }
            (StreamValue::Null, _) | (_, StreamValue::Null) => Ok(StreamValue::Null),
            (StreamValue::Bool(a), StreamValue::Bool(b)) => Ok(StreamValue::Bool(*a && *b)),
            _ => Err(ZyronError::StreamingError(
                "AND requires boolean operands".to_string(),
            )),
        };
    }
    if matches!(op, BinaryOpKind::Or) {
        return match (l, r) {
            (StreamValue::Bool(true), _) | (_, StreamValue::Bool(true)) => {
                Ok(StreamValue::Bool(true))
            }
            (StreamValue::Null, _) | (_, StreamValue::Null) => Ok(StreamValue::Null),
            (StreamValue::Bool(a), StreamValue::Bool(b)) => Ok(StreamValue::Bool(*a || *b)),
            _ => Err(ZyronError::StreamingError(
                "OR requires boolean operands".to_string(),
            )),
        };
    }

    // All other ops propagate Null.
    if matches!(l, StreamValue::Null) || matches!(r, StreamValue::Null) {
        return Ok(StreamValue::Null);
    }

    match op {
        BinaryOpKind::Eq => Ok(StreamValue::Bool(compare_values(l, r)? == 0)),
        BinaryOpKind::NotEq => Ok(StreamValue::Bool(compare_values(l, r)? != 0)),
        BinaryOpKind::Lt => Ok(StreamValue::Bool(compare_values(l, r)? < 0)),
        BinaryOpKind::LtEq => Ok(StreamValue::Bool(compare_values(l, r)? <= 0)),
        BinaryOpKind::Gt => Ok(StreamValue::Bool(compare_values(l, r)? > 0)),
        BinaryOpKind::GtEq => Ok(StreamValue::Bool(compare_values(l, r)? >= 0)),
        BinaryOpKind::Add => arith(l, r, |a, b| a + b, |a, b| a.wrapping_add(b)),
        BinaryOpKind::Sub => arith(l, r, |a, b| a - b, |a, b| a.wrapping_sub(b)),
        BinaryOpKind::Mul => arith(l, r, |a, b| a * b, |a, b| a.wrapping_mul(b)),
        BinaryOpKind::Div => {
            if let (Ok(a), Ok(b)) = (l.as_i64(), r.as_i64()) {
                if b == 0 {
                    Err(ZyronError::StreamingError("division by zero".to_string()))
                } else {
                    Ok(StreamValue::I64(a / b))
                }
            } else {
                let a = l.as_f64()?;
                let b = r.as_f64()?;
                if b == 0.0 {
                    Err(ZyronError::StreamingError("division by zero".to_string()))
                } else {
                    Ok(StreamValue::F64(a / b))
                }
            }
        }
        BinaryOpKind::And | BinaryOpKind::Or => unreachable!("handled above"),
    }
}

/// Compares two values. Mixed numeric types promote to f64.
fn compare_values(l: &StreamValue, r: &StreamValue) -> Result<i32> {
    match (l, r) {
        (StreamValue::Bool(a), StreamValue::Bool(b)) => Ok(a.cmp(b) as i32),
        (StreamValue::Utf8(a), StreamValue::Utf8(b)) => Ok(a.cmp(b) as i32),
        (StreamValue::Binary(a), StreamValue::Binary(b)) => Ok(a.cmp(b) as i32),
        _ => {
            // Try integer compare first to preserve exact semantics.
            if let (Ok(a), Ok(b)) = (l.as_i64(), r.as_i64()) {
                return Ok(a.cmp(&b) as i32);
            }
            let a = l.as_f64()?;
            let b = r.as_f64()?;
            Ok(a.partial_cmp(&b).map(|o| o as i32).unwrap_or(0))
        }
    }
}

/// Applies arithmetic to two numeric values. Uses i64 math when both sides
/// are integral, else promotes to f64.
fn arith(
    l: &StreamValue,
    r: &StreamValue,
    fop: impl Fn(f64, f64) -> f64,
    iop: impl Fn(i64, i64) -> i64,
) -> Result<StreamValue> {
    if matches!(l, StreamValue::F64(_)) || matches!(r, StreamValue::F64(_)) {
        return Ok(StreamValue::F64(fop(l.as_f64()?, r.as_f64()?)));
    }
    Ok(StreamValue::I64(iop(l.as_i64()?, r.as_i64()?)))
}

// -----------------------------------------------------------------------------
// Tests
// -----------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn roundtrip_int_row() {
        let types = vec![TypeId::Int64, TypeId::Int32];
        let values = vec![StreamValue::I64(42), StreamValue::I64(7)];
        let bytes = encode_row(&values, &types).unwrap();
        let decoded = decode_row(&bytes, &types).unwrap();
        assert_eq!(decoded.len(), 2);
        assert_eq!(decoded[0].as_i64().unwrap(), 42);
        assert_eq!(decoded[1].as_i64().unwrap(), 7);
    }

    #[test]
    fn roundtrip_mixed_row_with_varlen_and_null() {
        let types = vec![TypeId::Int64, TypeId::Varchar, TypeId::Boolean];
        let values = vec![
            StreamValue::Null,
            StreamValue::Utf8("hello".to_string()),
            StreamValue::Bool(true),
        ];
        let bytes = encode_row(&values, &types).unwrap();
        let decoded = decode_row(&bytes, &types).unwrap();
        assert!(matches!(decoded[0], StreamValue::Null));
        match &decoded[1] {
            StreamValue::Utf8(s) => assert_eq!(s, "hello"),
            other => panic!("expected Utf8, got {other:?}"),
        }
        assert!(matches!(decoded[2], StreamValue::Bool(true)));
    }

    #[test]
    fn eval_predicate_greater_than_literal() {
        let row = vec![StreamValue::I64(100), StreamValue::F64(1000.0)];
        let expr = ExprSpec::BinaryOp {
            op: BinaryOpKind::Gt,
            left: Box::new(ExprSpec::ColumnRef { ordinal: 1 }),
            right: Box::new(ExprSpec::LiteralI64(500)),
        };
        assert!(eval_expr(&expr, &row).unwrap().as_bool().unwrap());
    }

    #[test]
    fn eval_and_short_circuit_null_to_false() {
        let row = vec![StreamValue::Null];
        let expr = ExprSpec::BinaryOp {
            op: BinaryOpKind::And,
            left: Box::new(ExprSpec::LiteralBool(false)),
            right: Box::new(ExprSpec::ColumnRef { ordinal: 0 }),
        };
        // false AND anything => false (SQL three-valued logic).
        assert!(!eval_expr(&expr, &row).unwrap().as_bool().unwrap());
    }

    #[test]
    fn eval_arithmetic_on_column() {
        let row = vec![StreamValue::I64(10), StreamValue::I64(3)];
        let expr = ExprSpec::BinaryOp {
            op: BinaryOpKind::Add,
            left: Box::new(ExprSpec::ColumnRef { ordinal: 0 }),
            right: Box::new(ExprSpec::ColumnRef { ordinal: 1 }),
        };
        assert_eq!(eval_expr(&expr, &row).unwrap().as_i64().unwrap(), 13);
    }

    #[test]
    fn eval_not_on_bool() {
        let row: Vec<StreamValue> = vec![];
        let expr = ExprSpec::Not(Box::new(ExprSpec::LiteralBool(false)));
        assert!(eval_expr(&expr, &row).unwrap().as_bool().unwrap());
    }
}
