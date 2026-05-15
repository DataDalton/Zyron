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
        } => {
            let col = evaluate(inner, batch, schema, params)?;
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
    let mut data = ColumnData::with_capacity(type_id, num_rows);
    for _ in 0..num_rows {
        data.push_scalar(scalar);
    }
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
        LiteralValue::Interval(i) => Ok(Column::new(
            ColumnData::Interval(vec![*i; num_rows]),
            TypeId::Interval,
        )),
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
    params: &[ScalarValue],
) -> Result<Column> {
    let left_col = evaluate(left, batch, schema, params)?;
    let right_col = evaluate(right, batch, schema, params)?;

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
    let mut combined = compare(&expr_col, &first, CmpOp::Eq)?;

    for item in &list[1..] {
        let item_col = evaluate(item, batch, schema, params)?;
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
    params: &[ScalarValue],
) -> Result<Column> {
    let expr_col = evaluate(expr, batch, schema, params)?;
    let low_col = evaluate(low, batch, schema, params)?;
    let high_col = evaluate(high, batch, schema, params)?;

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
    params: &[ScalarValue],
) -> Result<Column> {
    let num_rows = batch.num_rows;

    // Start with else branch or null.
    let mut result = if let Some(else_expr) = else_result {
        evaluate(else_expr, batch, schema, params)?
    } else {
        Column::null_column(TypeId::Text, num_rows)
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
    params: &[ScalarValue],
) -> Result<Column> {
    match name {
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
        n if name_matches_ml(n) => eval_ml_scalar(n, args, batch, schema, params),
        _ => crate::types_bridge::evaluate_types_function(name, args, batch, schema, params),
    }
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
