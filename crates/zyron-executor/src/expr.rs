//! Expression evaluator for bound expressions against DataBatch columns.
//!
//! Evaluates BoundExpr trees from the planner, producing Column results
//! using the custom compute kernels.

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

/// Evaluates a bound expression against a DataBatch, returning the result as a Column.
pub fn evaluate(expr: &BoundExpr, batch: &DataBatch, schema: &[LogicalColumn]) -> Result<Column> {
    match expr {
        BoundExpr::ColumnRef(col_ref) => evaluate_column_ref(col_ref, batch, schema),
        BoundExpr::Literal { value, type_id } => evaluate_literal(value, *type_id, batch.num_rows),
        BoundExpr::BinaryOp {
            left, op, right, ..
        } => evaluate_binary_op(left, *op, right, batch, schema),
        BoundExpr::UnaryOp {
            op, expr: inner, ..
        } => evaluate_unary_op(*op, inner, batch, schema),
        BoundExpr::IsNull {
            expr: inner,
            negated,
        } => evaluate_is_null(inner, *negated, batch, schema),
        BoundExpr::InList {
            expr: inner,
            list,
            negated,
        } => evaluate_in_list(inner, list, *negated, batch, schema),
        BoundExpr::Between {
            expr: inner,
            low,
            high,
            negated,
        } => evaluate_between(inner, low, high, *negated, batch, schema),
        BoundExpr::Like {
            expr: inner,
            pattern,
            negated,
        } => {
            let col = evaluate(inner, batch, schema)?;
            let pat = evaluate(pattern, batch, schema)?;
            compute::like(&col, &pat, *negated)
        }
        BoundExpr::ILike {
            expr: inner,
            pattern,
            negated,
        } => {
            let col = evaluate(inner, batch, schema)?;
            let pat = evaluate(pattern, batch, schema)?;
            compute::ilike(&col, &pat, *negated)
        }
        BoundExpr::Function { name, args, .. } => evaluate_function(name, args, batch, schema),
        BoundExpr::AggregateFunction { .. } => Err(ZyronError::ExecutionError(
            "aggregate functions must be evaluated by the aggregate operator".to_string(),
        )),
        BoundExpr::Cast {
            expr: inner,
            target_type,
        } => {
            let col = evaluate(inner, batch, schema)?;
            cast_column(&col, *target_type)
        }
        BoundExpr::Case {
            operand,
            conditions,
            else_result,
            ..
        } => evaluate_case(
            operand.as_deref(),
            conditions,
            else_result.as_deref(),
            batch,
            schema,
        ),
        BoundExpr::Nested(inner) => evaluate(inner, batch, schema),
        BoundExpr::Subquery { .. } | BoundExpr::Exists { .. } | BoundExpr::InSubquery { .. } => {
            Err(ZyronError::ExecutionError(
                "subqueries not supported in executor".to_string(),
            ))
        }
        BoundExpr::WindowFunction { .. } => Err(ZyronError::ExecutionError(
            "window functions not supported yet".to_string(),
        )),
        BoundExpr::Parameter { .. } => Err(ZyronError::ExecutionError(
            "parameters not supported yet".to_string(),
        )),
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

fn evaluate_literal(value: &LiteralValue, type_id: TypeId, num_rows: usize) -> Result<Column> {
    match value {
        LiteralValue::Integer(v) => Ok(Column::new(
            ColumnData::Int64(vec![*v; num_rows]),
            TypeId::Int64,
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
    }
}

// ---------------------------------------------------------------------------
// Binary operators
// ---------------------------------------------------------------------------

fn evaluate_binary_op(
    left: &BoundExpr,
    op: BinaryOperator,
    right: &BoundExpr,
    batch: &DataBatch,
    schema: &[LogicalColumn],
) -> Result<Column> {
    let left_col = evaluate(left, batch, schema)?;
    let right_col = evaluate(right, batch, schema)?;

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

// ---------------------------------------------------------------------------
// Unary operators
// ---------------------------------------------------------------------------

fn evaluate_unary_op(
    op: UnaryOperator,
    expr: &BoundExpr,
    batch: &DataBatch,
    schema: &[LogicalColumn],
) -> Result<Column> {
    let col = evaluate(expr, batch, schema)?;
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
) -> Result<Column> {
    let col = evaluate(expr, batch, schema)?;
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
) -> Result<Column> {
    let expr_col = evaluate(expr, batch, schema)?;
    let num_rows = batch.num_rows;

    if list.is_empty() {
        let val = negated;
        return Ok(Column::new(
            ColumnData::Boolean(vec![val; num_rows]),
            TypeId::Boolean,
        ));
    }

    let first = evaluate(&list[0], batch, schema)?;
    let mut combined = compare(&expr_col, &first, CmpOp::Eq)?;

    for item in &list[1..] {
        let item_col = evaluate(item, batch, schema)?;
        let cmp_result = compare(&expr_col, &item_col, CmpOp::Eq)?;
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
) -> Result<Column> {
    let expr_col = evaluate(expr, batch, schema)?;
    let low_col = evaluate(low, batch, schema)?;
    let high_col = evaluate(high, batch, schema)?;

    let gte_low = compare(&expr_col, &low_col, CmpOp::GtEq)?;
    let lte_high = compare(&expr_col, &high_col, CmpOp::LtEq)?;
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

fn evaluate_case(
    operand: Option<&BoundExpr>,
    conditions: &[BoundWhen],
    else_result: Option<&BoundExpr>,
    batch: &DataBatch,
    schema: &[LogicalColumn],
) -> Result<Column> {
    let num_rows = batch.num_rows;

    // Start with else branch or null.
    let mut result = if let Some(else_expr) = else_result {
        evaluate(else_expr, batch, schema)?
    } else {
        Column::null_column(TypeId::Text, num_rows)
    };

    let operand_col = match operand {
        Some(op) => Some(evaluate(op, batch, schema)?),
        None => None,
    };

    // Process conditions in reverse so first match wins.
    for when in conditions.iter().rev() {
        let condition_bool = if let Some(ref op_col) = operand_col {
            let cond_col = evaluate(&when.condition, batch, schema)?;
            compare(op_col, &cond_col, CmpOp::Eq)?
        } else {
            evaluate(&when.condition, batch, schema)?
        };

        let then_col = evaluate(&when.result, batch, schema)?;
        let mask = column_to_mask(&condition_bool);

        // Use typed push_from to build result without ScalarValue.
        let mut new_data = ColumnData::with_capacity(result.type_id, num_rows);
        let mut new_nulls = NullBitmap::none(num_rows);

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

// ---------------------------------------------------------------------------
// Scalar functions
// ---------------------------------------------------------------------------

fn evaluate_function(
    name: &str,
    args: &[BoundExpr],
    batch: &DataBatch,
    schema: &[LogicalColumn],
) -> Result<Column> {
    match name {
        "abs" => {
            let col = evaluate(&args[0], batch, schema)?;
            eval_abs(&col)
        }
        "upper" => {
            let col = evaluate(&args[0], batch, schema)?;
            eval_string_transform(&col, |s| s.to_uppercase())
        }
        "lower" => {
            let col = evaluate(&args[0], batch, schema)?;
            eval_string_transform(&col, |s| s.to_lowercase())
        }
        "length" => {
            let col = evaluate(&args[0], batch, schema)?;
            eval_length(&col)
        }
        "coalesce" => eval_coalesce(args, batch, schema),
        "nullif" => {
            let a = evaluate(&args[0], batch, schema)?;
            let b = evaluate(&args[1], batch, schema)?;
            eval_nullif(&a, &b)
        }
        _ => Err(ZyronError::ExecutionError(format!(
            "unknown function: {name}"
        ))),
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
        .map(|(i, s)| if col.is_null(i) { 0 } else { s.len() as i64 })
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
) -> Result<Column> {
    let num_rows = batch.num_rows;
    let last_idx = args.len() - 1;
    let mut result = evaluate(&args[last_idx], batch, schema)?;

    for arg in args[..last_idx].iter().rev() {
        let arg_col = evaluate(arg, batch, schema)?;
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
