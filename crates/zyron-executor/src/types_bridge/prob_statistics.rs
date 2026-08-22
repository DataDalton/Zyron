//! Dispatch arms for zyron_types::probabilistic sketches and
//! zyron_types::statistics functions
//!
//! Sketch values (HLL, Bloom, T-Digest, CMS) travel as their module's tagged
//! byte serialization inside ColumnData::Binary cells. Array typed results
//! are JSON number or bool arrays encoded as text bytes inside Binary cells.
//! Array typed statistics inputs accept a JSON array of numbers in a Utf8 or
//! Binary cell
//!
//! Error policy
//! - wrong arg count or un-coercible column type returns ExecutionError
//! - create fns reject out of range parameters with ExecutionError since a
//!   bad parameter is a call site bug not row data
//! - invalid, mismatched, or empty sketch bytes on non create fns yield a
//!   NULL row, matching the lenient hll_count and bloom_contains arms in mod.rs
//! - malformed JSON array input to a statistics fn yields a NULL row, as does
//!   a data domain error such as mismatched x and y lengths
//! - a NULL in any input row makes that output row NULL

use crate::column::{Column, ColumnData, NullBitmap, ScalarValue};
use zyron_common::{Result, TypeId, ZyronError};
use zyron_types::probabilistic as prob;
use zyron_types::statistics as stats;

pub(super) fn dispatch(name: &str, args: &[Column], num_rows: usize) -> Option<Result<Column>> {
    // every function here takes arguments, no zero arg generators
    let _ = num_rows;
    Some(match name {
        // ---------- probabilistic sketches ----------
        // hll_count, bloom_contains, cms_estimate have arms in mod.rs
        "hll_create" => hll_create_col(args),
        "hll_add" => sketch_value_to_sketch(args, "hll_add", TypeId::HyperLogLog, |s, v| {
            let mut sk = s.to_vec();
            prob::hll_add(&mut sk, v)?;
            Ok(sk)
        }),
        "hll_merge" => {
            sketch_pair_to_sketch(args, "hll_merge", TypeId::HyperLogLog, prob::hll_merge)
        }
        "hll_error" => sketch_to_float(args, "hll_error", prob::hll_error),
        "bloom_create" => bloom_create_col(args),
        "bloom_add" => sketch_value_to_sketch(args, "bloom_add", TypeId::BloomFilter, |s, v| {
            let mut f = s.to_vec();
            prob::bloom_add(&mut f, v)?;
            Ok(f)
        }),
        "bloom_merge" => {
            sketch_pair_to_sketch(args, "bloom_merge", TypeId::BloomFilter, prob::bloom_merge)
        }
        "bloom_false_positive_rate" => sketch_to_float(
            args,
            "bloom_false_positive_rate",
            prob::bloom_false_positive_rate,
        ),
        "cms_create" => cms_create_col(args),
        "cms_add" => cms_add_col(args),
        "cms_merge" => {
            sketch_pair_to_sketch(args, "cms_merge", TypeId::CountMinSketch, prob::cms_merge)
        }
        "tdigest_create" => tdigest_create_col(args),
        "tdigest_add" => tdigest_add_col(args),
        "tdigest_merge" => {
            sketch_pair_to_sketch(args, "tdigest_merge", TypeId::TDigest, prob::tdigest_merge)
        }
        "tdigest_quantile" => {
            sketch_float_to_float(args, "tdigest_quantile", prob::tdigest_quantile)
        }
        "tdigest_cdf" => sketch_float_to_float(args, "tdigest_cdf", prob::tdigest_cdf),

        // ---------- statistics ----------
        "correlation" => {
            two_array_to_float(args, "correlation", |x, y| stats::correlation(x, y).ok())
        }
        "covariance" => two_array_to_float(args, "covariance", |x, y| stats::covariance(x, y).ok()),
        "zscore" => zscore_col(args),
        "percentile" => percentile_col(args),
        "stddev_pop" => one_array_to_float(args, "stddev_pop", |v| Some(stats::stddev_pop(v))),
        "stddev_sample" => {
            one_array_to_float(args, "stddev_sample", |v| Some(stats::stddev_sample(v)))
        }
        "variance_pop" => {
            one_array_to_float(args, "variance_pop", |v| Some(stats::variance_pop(v)))
        }
        "variance_sample" => {
            one_array_to_float(args, "variance_sample", |v| Some(stats::variance_sample(v)))
        }
        "skewness" => one_array_to_float(args, "skewness", |v| Some(stats::skewness(v))),
        "kurtosis" => one_array_to_float(args, "kurtosis", |v| Some(stats::kurtosis(v))),
        "linear_regression" => two_array_to_json(args, "linear_regression", |x, y| {
            // output is [slope, intercept, r_squared]
            stats::linear_regression(x, y)
                .ok()
                .map(|(slope, intercept, r2)| floats_to_json(&[slope, intercept, r2]))
        }),
        "exponential_smoothing" => array_scalar_to_json(args, "exponential_smoothing", |v, a| {
            Some(floats_to_json(&stats::exponential_smoothing(v, a)))
        }),
        "forecast_linear" => forecast_linear_col(args),
        "moving_average" => moving_average_col(args),
        "weighted_moving_average" => two_array_to_json(args, "weighted_moving_average", |v, w| {
            stats::weighted_moving_average(v, w)
                .ok()
                .map(|r| floats_to_json(&r))
        }),
        "outlier_detect_zscore" => array_scalar_to_json(args, "outlier_detect_zscore", |v, t| {
            Some(bools_to_json(&stats::outlier_detect_zscore(v, t)))
        }),
        "outlier_detect_iqr" => array_scalar_to_json(args, "outlier_detect_iqr", |v, f| {
            Some(bools_to_json(&stats::outlier_detect_iqr(v, f)))
        }),
        _ => return None,
    })
}

// ---------------------------------------------------------------------------
// per cell readers
// ---------------------------------------------------------------------------

/// Reads a numeric cell as f64, NULL or non numeric yields None
fn scalar_f64(col: &Column, row: usize) -> Option<f64> {
    match col.get_scalar(row) {
        ScalarValue::Null => None,
        v => v.to_f64(),
    }
}

/// Reads an integer cell tolerantly, floats truncate toward zero
fn scalar_int(col: &Column, row: usize) -> Option<i64> {
    match col.get_scalar(row) {
        ScalarValue::Null => None,
        v => v
            .to_i128()
            .and_then(|x| i64::try_from(x).ok())
            .or_else(|| v.to_f64().map(|f| f as i64)),
    }
}

/// Byte representation of a cell for sketch hashing, strings hash their UTF-8
/// bytes, integers hash their 64 bit LE pattern, floats their f64 LE bytes
fn scalar_hash_bytes(col: &Column, row: usize) -> Option<Vec<u8>> {
    match col.get_scalar(row) {
        ScalarValue::Null => None,
        ScalarValue::Utf8(s) => Some(s.into_bytes()),
        ScalarValue::Binary(b) => Some(b),
        ScalarValue::FixedBinary16(b) => Some(b.to_vec()),
        ScalarValue::Boolean(b) => Some(vec![u8::from(b)]),
        ScalarValue::Float64(v) => Some(v.to_le_bytes().to_vec()),
        ScalarValue::Float32(v) => Some((v as f64).to_le_bytes().to_vec()),
        v => v.to_i128().map(|x| (x as i64).to_le_bytes().to_vec()),
    }
}

/// Borrows each cell of a sketch column as raw bytes
fn column_cell_bytes<'a>(col: &'a Column, fn_name: &str) -> Result<Vec<&'a [u8]>> {
    match &col.data {
        ColumnData::Binary(v) => Ok(v.iter().map(|b| b.as_slice()).collect()),
        ColumnData::Utf8(v) => Ok(v.iter().map(|s| s.as_bytes()).collect()),
        _ => Err(ZyronError::ExecutionError(format!(
            "{fn_name} expects a sketch bytes argument"
        ))),
    }
}

/// Extracts each cell of a JSON array column as text
fn json_cell_texts(col: &Column, fn_name: &str) -> Result<Vec<String>> {
    match &col.data {
        ColumnData::Utf8(v) => Ok(v.clone()),
        ColumnData::Binary(v) => Ok(v
            .iter()
            .map(|b| String::from_utf8_lossy(b).into_owned())
            .collect()),
        _ => Err(ZyronError::ExecutionError(format!(
            "{fn_name} expects a JSON array of numbers as a text or binary argument"
        ))),
    }
}

/// Rejects columns that cannot coerce to a numeric scalar
fn check_numeric_arg(col: &Column, fn_name: &str) -> Result<()> {
    match &col.data {
        ColumnData::Utf8(_)
        | ColumnData::Binary(_)
        | ColumnData::FixedBinary16(_)
        | ColumnData::Interval(_) => Err(ZyronError::ExecutionError(format!(
            "{fn_name} expects a numeric argument"
        ))),
        _ => Ok(()),
    }
}

/// Rejects columns whose cells have no hashable byte form
fn check_hashable_arg(col: &Column, fn_name: &str) -> Result<()> {
    match &col.data {
        ColumnData::Interval(_) => Err(ZyronError::ExecutionError(format!(
            "{fn_name} value argument has no byte representation"
        ))),
        _ => Ok(()),
    }
}

/// Parses a JSON array where every element is a number, anything else None
fn parse_number_array(text: &str) -> Option<Vec<f64>> {
    let v: serde_json::Value = serde_json::from_str(text).ok()?;
    let arr = v.as_array()?;
    let mut out = Vec::with_capacity(arr.len());
    for e in arr {
        out.push(e.as_f64()?);
    }
    Some(out)
}

fn min_len(args: &[Column]) -> usize {
    args.iter().map(|c| c.data.len()).min().unwrap_or(0)
}

// ---------------------------------------------------------------------------
// output builders, None rows become SQL NULL
// ---------------------------------------------------------------------------

fn binary_rows_to_column(rows: Vec<Option<Vec<u8>>>, type_id: TypeId) -> Column {
    let n = rows.len();
    let mut data = Vec::with_capacity(n);
    let mut nulls = NullBitmap::none(n);
    for (i, r) in rows.into_iter().enumerate() {
        match r {
            Some(b) => data.push(b),
            None => {
                data.push(Vec::new());
                nulls.set_null(i);
            }
        }
    }
    Column::with_nulls(ColumnData::Binary(data), nulls, type_id)
}

fn float_rows_to_column(rows: Vec<Option<f64>>) -> Column {
    let n = rows.len();
    let mut data = Vec::with_capacity(n);
    let mut nulls = NullBitmap::none(n);
    for (i, r) in rows.into_iter().enumerate() {
        match r {
            Some(v) => data.push(v),
            None => {
                data.push(0.0);
                nulls.set_null(i);
            }
        }
    }
    Column::with_nulls(ColumnData::Float64(data), nulls, TypeId::Float64)
}

fn json_rows_to_column(rows: Vec<Option<String>>) -> Column {
    let n = rows.len();
    let mut data = Vec::with_capacity(n);
    let mut nulls = NullBitmap::none(n);
    for (i, r) in rows.into_iter().enumerate() {
        match r {
            Some(s) => data.push(s.into_bytes()),
            None => {
                data.push(Vec::new());
                nulls.set_null(i);
            }
        }
    }
    Column::with_nulls(ColumnData::Binary(data), nulls, TypeId::Array)
}

/// Serializes f64 values as a JSON array, non finite values encode as null
fn floats_to_json(vals: &[f64]) -> String {
    let arr: Vec<serde_json::Value> = vals
        .iter()
        .map(|&v| {
            serde_json::Number::from_f64(v)
                .map(serde_json::Value::Number)
                .unwrap_or(serde_json::Value::Null)
        })
        .collect();
    serde_json::Value::Array(arr).to_string()
}

fn bools_to_json(vals: &[bool]) -> String {
    serde_json::Value::Array(vals.iter().map(|&b| serde_json::Value::Bool(b)).collect()).to_string()
}

// ---------------------------------------------------------------------------
// sketch combinators
// ---------------------------------------------------------------------------

fn sketch_to_float<F: Fn(&[u8]) -> Result<f64>>(
    args: &[Column],
    fn_name: &str,
    f: F,
) -> Result<Column> {
    if args.len() != 1 {
        return Err(ZyronError::ExecutionError(format!(
            "{fn_name}(sketch) takes exactly 1 argument"
        )));
    }
    let cells = column_cell_bytes(&args[0], fn_name)?;
    let mut rows = Vec::with_capacity(cells.len());
    for i in 0..cells.len() {
        if args[0].nulls.is_null(i) {
            rows.push(None);
            continue;
        }
        rows.push(f(cells[i]).ok());
    }
    Ok(float_rows_to_column(rows))
}

fn sketch_float_to_float<F: Fn(&[u8], f64) -> Result<f64>>(
    args: &[Column],
    fn_name: &str,
    f: F,
) -> Result<Column> {
    if args.len() != 2 {
        return Err(ZyronError::ExecutionError(format!(
            "{fn_name}(sketch, value) takes exactly 2 arguments"
        )));
    }
    check_numeric_arg(&args[1], fn_name)?;
    let cells = column_cell_bytes(&args[0], fn_name)?;
    let n = cells.len().min(args[1].data.len());
    let mut rows = Vec::with_capacity(n);
    for i in 0..n {
        if args[0].nulls.is_null(i) {
            rows.push(None);
            continue;
        }
        rows.push(match scalar_f64(&args[1], i) {
            Some(x) => f(cells[i], x).ok(),
            None => None,
        });
    }
    Ok(float_rows_to_column(rows))
}

fn sketch_value_to_sketch<F: Fn(&[u8], &[u8]) -> Result<Vec<u8>>>(
    args: &[Column],
    fn_name: &str,
    out_type: TypeId,
    f: F,
) -> Result<Column> {
    if args.len() != 2 {
        return Err(ZyronError::ExecutionError(format!(
            "{fn_name}(sketch, value) takes exactly 2 arguments"
        )));
    }
    check_hashable_arg(&args[1], fn_name)?;
    let cells = column_cell_bytes(&args[0], fn_name)?;
    let n = cells.len().min(args[1].data.len());
    let mut rows = Vec::with_capacity(n);
    for i in 0..n {
        if args[0].nulls.is_null(i) {
            rows.push(None);
            continue;
        }
        rows.push(match scalar_hash_bytes(&args[1], i) {
            Some(v) => f(cells[i], &v).ok(),
            None => None,
        });
    }
    Ok(binary_rows_to_column(rows, out_type))
}

fn sketch_pair_to_sketch<F: Fn(&[u8], &[u8]) -> Result<Vec<u8>>>(
    args: &[Column],
    fn_name: &str,
    out_type: TypeId,
    f: F,
) -> Result<Column> {
    if args.len() != 2 {
        return Err(ZyronError::ExecutionError(format!(
            "{fn_name}(sketch, sketch) takes exactly 2 arguments"
        )));
    }
    let a = column_cell_bytes(&args[0], fn_name)?;
    let b = column_cell_bytes(&args[1], fn_name)?;
    let n = a.len().min(b.len());
    let mut rows = Vec::with_capacity(n);
    for i in 0..n {
        if args[0].nulls.is_null(i) || args[1].nulls.is_null(i) {
            rows.push(None);
            continue;
        }
        // mismatched parameters or invalid bytes yield NULL
        rows.push(f(a[i], b[i]).ok());
    }
    Ok(binary_rows_to_column(rows, out_type))
}

// ---------------------------------------------------------------------------
// sketch creators, parameter range errors abort the call
// ---------------------------------------------------------------------------

fn hll_create_col(args: &[Column]) -> Result<Column> {
    if args.len() != 1 {
        return Err(ZyronError::ExecutionError(
            "hll_create(precision) takes exactly 1 argument".to_string(),
        ));
    }
    check_numeric_arg(&args[0], "hll_create")?;
    let n = args[0].data.len();
    let mut rows = Vec::with_capacity(n);
    for i in 0..n {
        match scalar_int(&args[0], i) {
            None => rows.push(None),
            Some(p) => {
                if !(4..=16).contains(&p) {
                    return Err(ZyronError::ExecutionError(format!(
                        "hll_create precision must be 4..=16, got {p}"
                    )));
                }
                rows.push(Some(prob::hll_create(p as u8)?));
            }
        }
    }
    Ok(binary_rows_to_column(rows, TypeId::HyperLogLog))
}

fn bloom_create_col(args: &[Column]) -> Result<Column> {
    if args.len() != 2 {
        return Err(ZyronError::ExecutionError(
            "bloom_create(expected_items, false_positive_rate) takes exactly 2 arguments"
                .to_string(),
        ));
    }
    check_numeric_arg(&args[0], "bloom_create")?;
    check_numeric_arg(&args[1], "bloom_create")?;
    let n = min_len(args);
    let mut rows = Vec::with_capacity(n);
    for i in 0..n {
        let items = scalar_int(&args[0], i);
        let fpr = scalar_f64(&args[1], i);
        match (items, fpr) {
            (Some(it), Some(rate)) => {
                if it < 1 {
                    return Err(ZyronError::ExecutionError(format!(
                        "bloom_create expected_items must be positive, got {it}"
                    )));
                }
                rows.push(Some(prob::bloom_create(it as u64, rate)?));
            }
            _ => rows.push(None),
        }
    }
    Ok(binary_rows_to_column(rows, TypeId::BloomFilter))
}

fn cms_create_col(args: &[Column]) -> Result<Column> {
    if args.len() != 2 {
        return Err(ZyronError::ExecutionError(
            "cms_create(width, depth) takes exactly 2 arguments".to_string(),
        ));
    }
    check_numeric_arg(&args[0], "cms_create")?;
    check_numeric_arg(&args[1], "cms_create")?;
    let n = min_len(args);
    let mut rows = Vec::with_capacity(n);
    for i in 0..n {
        let width = scalar_int(&args[0], i);
        let depth = scalar_int(&args[1], i);
        match (width, depth) {
            (Some(w), Some(d)) => {
                if !(1..=u32::MAX as i64).contains(&w) {
                    return Err(ZyronError::ExecutionError(format!(
                        "cms_create width must be in 1..=4294967295, got {w}"
                    )));
                }
                if !(1..=16).contains(&d) {
                    return Err(ZyronError::ExecutionError(format!(
                        "cms_create depth must be in 1..=16, got {d}"
                    )));
                }
                rows.push(Some(prob::cms_create(w as u32, d as u32)?));
            }
            _ => rows.push(None),
        }
    }
    Ok(binary_rows_to_column(rows, TypeId::CountMinSketch))
}

fn tdigest_create_col(args: &[Column]) -> Result<Column> {
    if args.len() != 1 {
        return Err(ZyronError::ExecutionError(
            "tdigest_create(compression) takes exactly 1 argument".to_string(),
        ));
    }
    check_numeric_arg(&args[0], "tdigest_create")?;
    let n = args[0].data.len();
    let mut rows = Vec::with_capacity(n);
    for i in 0..n {
        match scalar_f64(&args[0], i) {
            // module rejects compression outside [1, 10000]
            Some(c) => rows.push(Some(prob::tdigest_create(c)?)),
            None => rows.push(None),
        }
    }
    Ok(binary_rows_to_column(rows, TypeId::TDigest))
}

// ---------------------------------------------------------------------------
// sketch fns with bespoke shapes
// ---------------------------------------------------------------------------

fn tdigest_add_col(args: &[Column]) -> Result<Column> {
    if args.len() != 2 {
        return Err(ZyronError::ExecutionError(
            "tdigest_add(tdigest, value) takes exactly 2 arguments".to_string(),
        ));
    }
    check_numeric_arg(&args[1], "tdigest_add")?;
    let cells = column_cell_bytes(&args[0], "tdigest_add")?;
    let n = cells.len().min(args[1].data.len());
    let mut rows = Vec::with_capacity(n);
    for i in 0..n {
        if args[0].nulls.is_null(i) {
            rows.push(None);
            continue;
        }
        rows.push(match scalar_f64(&args[1], i) {
            Some(x) => {
                let mut digest = cells[i].to_vec();
                match prob::tdigest_add(&mut digest, x) {
                    Ok(()) => Some(digest),
                    Err(_) => None,
                }
            }
            None => None,
        });
    }
    Ok(binary_rows_to_column(rows, TypeId::TDigest))
}

/// cms_add(cms, value [, count]) with count defaulting to 1
fn cms_add_col(args: &[Column]) -> Result<Column> {
    if !(2..=3).contains(&args.len()) {
        return Err(ZyronError::ExecutionError(
            "cms_add(cms, value [, count]) takes 2 or 3 arguments".to_string(),
        ));
    }
    check_hashable_arg(&args[1], "cms_add")?;
    if args.len() == 3 {
        check_numeric_arg(&args[2], "cms_add")?;
    }
    let cells = column_cell_bytes(&args[0], "cms_add")?;
    let n = min_len(args);
    let mut rows = Vec::with_capacity(n);
    for i in 0..n {
        if args[0].nulls.is_null(i) {
            rows.push(None);
            continue;
        }
        let value = match scalar_hash_bytes(&args[1], i) {
            Some(v) => v,
            None => {
                rows.push(None);
                continue;
            }
        };
        let count = if args.len() == 3 {
            match scalar_int(&args[2], i) {
                Some(c) if c >= 0 => c as u64,
                Some(c) => {
                    return Err(ZyronError::ExecutionError(format!(
                        "cms_add count must be non negative, got {c}"
                    )));
                }
                None => {
                    rows.push(None);
                    continue;
                }
            }
        } else {
            1
        };
        let mut sketch = cells[i].to_vec();
        rows.push(match prob::cms_add(&mut sketch, &value, count) {
            Ok(()) => Some(sketch),
            Err(_) => None,
        });
    }
    Ok(binary_rows_to_column(rows, TypeId::CountMinSketch))
}

// ---------------------------------------------------------------------------
// statistics combinators
// ---------------------------------------------------------------------------

fn one_array_to_float<F: Fn(&[f64]) -> Option<f64>>(
    args: &[Column],
    fn_name: &str,
    f: F,
) -> Result<Column> {
    if args.len() != 1 {
        return Err(ZyronError::ExecutionError(format!(
            "{fn_name}(values) takes exactly 1 argument"
        )));
    }
    let texts = json_cell_texts(&args[0], fn_name)?;
    let mut rows = Vec::with_capacity(texts.len());
    for (i, t) in texts.iter().enumerate() {
        if args[0].nulls.is_null(i) {
            rows.push(None);
            continue;
        }
        rows.push(parse_number_array(t).and_then(|v| f(&v)));
    }
    Ok(float_rows_to_column(rows))
}

fn two_array_to_float<F: Fn(&[f64], &[f64]) -> Option<f64>>(
    args: &[Column],
    fn_name: &str,
    f: F,
) -> Result<Column> {
    if args.len() != 2 {
        return Err(ZyronError::ExecutionError(format!(
            "{fn_name}(x, y) takes exactly 2 arguments"
        )));
    }
    let xs = json_cell_texts(&args[0], fn_name)?;
    let ys = json_cell_texts(&args[1], fn_name)?;
    let n = xs.len().min(ys.len());
    let mut rows = Vec::with_capacity(n);
    for i in 0..n {
        if args[0].nulls.is_null(i) || args[1].nulls.is_null(i) {
            rows.push(None);
            continue;
        }
        rows.push(
            match (parse_number_array(&xs[i]), parse_number_array(&ys[i])) {
                (Some(x), Some(y)) => f(&x, &y),
                _ => None,
            },
        );
    }
    Ok(float_rows_to_column(rows))
}

fn array_scalar_to_json<F: Fn(&[f64], f64) -> Option<String>>(
    args: &[Column],
    fn_name: &str,
    f: F,
) -> Result<Column> {
    if args.len() != 2 {
        return Err(ZyronError::ExecutionError(format!(
            "{fn_name}(values, param) takes exactly 2 arguments"
        )));
    }
    check_numeric_arg(&args[1], fn_name)?;
    let texts = json_cell_texts(&args[0], fn_name)?;
    let n = texts.len().min(args[1].data.len());
    let mut rows = Vec::with_capacity(n);
    for i in 0..n {
        if args[0].nulls.is_null(i) {
            rows.push(None);
            continue;
        }
        rows.push(
            match (parse_number_array(&texts[i]), scalar_f64(&args[1], i)) {
                (Some(v), Some(p)) => f(&v, p),
                _ => None,
            },
        );
    }
    Ok(json_rows_to_column(rows))
}

fn two_array_to_json<F: Fn(&[f64], &[f64]) -> Option<String>>(
    args: &[Column],
    fn_name: &str,
    f: F,
) -> Result<Column> {
    if args.len() != 2 {
        return Err(ZyronError::ExecutionError(format!(
            "{fn_name}(x, y) takes exactly 2 arguments"
        )));
    }
    let xs = json_cell_texts(&args[0], fn_name)?;
    let ys = json_cell_texts(&args[1], fn_name)?;
    let n = xs.len().min(ys.len());
    let mut rows = Vec::with_capacity(n);
    for i in 0..n {
        if args[0].nulls.is_null(i) || args[1].nulls.is_null(i) {
            rows.push(None);
            continue;
        }
        rows.push(
            match (parse_number_array(&xs[i]), parse_number_array(&ys[i])) {
                (Some(x), Some(y)) => f(&x, &y),
                _ => None,
            },
        );
    }
    Ok(json_rows_to_column(rows))
}

// ---------------------------------------------------------------------------
// statistics fns with bespoke shapes
// ---------------------------------------------------------------------------

fn zscore_col(args: &[Column]) -> Result<Column> {
    if args.len() != 3 {
        return Err(ZyronError::ExecutionError(
            "zscore(value, mean, stddev) takes exactly 3 arguments".to_string(),
        ));
    }
    for a in args {
        check_numeric_arg(a, "zscore")?;
    }
    let n = min_len(args);
    let mut rows = Vec::with_capacity(n);
    for i in 0..n {
        rows.push(
            match (
                scalar_f64(&args[0], i),
                scalar_f64(&args[1], i),
                scalar_f64(&args[2], i),
            ) {
                (Some(v), Some(m), Some(s)) => Some(stats::zscore(v, m, s)),
                _ => None,
            },
        );
    }
    Ok(float_rows_to_column(rows))
}

fn percentile_col(args: &[Column]) -> Result<Column> {
    if args.len() != 2 {
        return Err(ZyronError::ExecutionError(
            "percentile(values, p) takes exactly 2 arguments".to_string(),
        ));
    }
    check_numeric_arg(&args[1], "percentile")?;
    let texts = json_cell_texts(&args[0], "percentile")?;
    let n = texts.len().min(args[1].data.len());
    let mut rows = Vec::with_capacity(n);
    for i in 0..n {
        if args[0].nulls.is_null(i) {
            rows.push(None);
            continue;
        }
        rows.push(
            match (parse_number_array(&texts[i]), scalar_f64(&args[1], i)) {
                (Some(mut v), Some(p)) => Some(stats::percentile(&mut v, p)),
                _ => None,
            },
        );
    }
    Ok(float_rows_to_column(rows))
}

fn moving_average_col(args: &[Column]) -> Result<Column> {
    if args.len() != 2 {
        return Err(ZyronError::ExecutionError(
            "moving_average(values, window) takes exactly 2 arguments".to_string(),
        ));
    }
    check_numeric_arg(&args[1], "moving_average")?;
    let texts = json_cell_texts(&args[0], "moving_average")?;
    let n = texts.len().min(args[1].data.len());
    let mut rows = Vec::with_capacity(n);
    for i in 0..n {
        if args[0].nulls.is_null(i) {
            rows.push(None);
            continue;
        }
        // negative window yields NULL, window 0 echoes the input unchanged
        rows.push(
            match (parse_number_array(&texts[i]), scalar_int(&args[1], i)) {
                (Some(v), Some(w)) if w >= 0 => {
                    Some(floats_to_json(&stats::moving_average(&v, w as usize)))
                }
                _ => None,
            },
        );
    }
    Ok(json_rows_to_column(rows))
}

fn forecast_linear_col(args: &[Column]) -> Result<Column> {
    if args.len() != 3 {
        return Err(ZyronError::ExecutionError(
            "forecast_linear(x, y, future_x) takes exactly 3 arguments".to_string(),
        ));
    }
    let xs = json_cell_texts(&args[0], "forecast_linear")?;
    let ys = json_cell_texts(&args[1], "forecast_linear")?;
    let fs = json_cell_texts(&args[2], "forecast_linear")?;
    let n = xs.len().min(ys.len()).min(fs.len());
    let mut rows = Vec::with_capacity(n);
    for i in 0..n {
        if args.iter().any(|a| a.nulls.is_null(i)) {
            rows.push(None);
            continue;
        }
        rows.push(
            match (
                parse_number_array(&xs[i]),
                parse_number_array(&ys[i]),
                parse_number_array(&fs[i]),
            ) {
                (Some(x), Some(y), Some(future)) => stats::forecast_linear(&x, &y, &future)
                    .ok()
                    .map(|r| floats_to_json(&r)),
                _ => None,
            },
        );
    }
    Ok(json_rows_to_column(rows))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn utf8_col(vals: &[&str]) -> Column {
        Column::new(
            ColumnData::Utf8(vals.iter().map(|s| s.to_string()).collect()),
            TypeId::Text,
        )
    }

    fn i32_col(vals: &[i32]) -> Column {
        Column::new(ColumnData::Int32(vals.to_vec()), TypeId::Int32)
    }

    fn f64_col(vals: &[f64]) -> Column {
        Column::new(ColumnData::Float64(vals.to_vec()), TypeId::Float64)
    }

    fn bin_col(cells: Vec<Vec<u8>>, type_id: TypeId) -> Column {
        Column::new(ColumnData::Binary(cells), type_id)
    }

    fn bin_cell(col: &Column, row: usize) -> Vec<u8> {
        match &col.data {
            ColumnData::Binary(v) => v[row].clone(),
            other => panic!("expected binary column, got {:?}", other),
        }
    }

    fn f64_cell(col: &Column, row: usize) -> f64 {
        match &col.data {
            ColumnData::Float64(v) => v[row],
            other => panic!("expected float column, got {:?}", other),
        }
    }

    #[test]
    fn hll_create_and_add_produce_countable_sketch() {
        let created = dispatch("hll_create", &[i32_col(&[12])], 1)
            .unwrap()
            .unwrap();
        assert_eq!(created.type_id, TypeId::HyperLogLog);
        let sketch = bin_cell(&created, 0);
        assert_eq!(prob::hll_count(&sketch).unwrap(), 0);

        let added = dispatch(
            "hll_add",
            &[
                bin_col(vec![sketch], TypeId::HyperLogLog),
                utf8_col(&["alpha"]),
            ],
            1,
        )
        .unwrap()
        .unwrap();
        let count = prob::hll_count(&bin_cell(&added, 0)).unwrap();
        assert!(count >= 1 && count <= 2);
    }

    #[test]
    fn bloom_merge_unions_both_filters() {
        let empty = prob::bloom_create(100, 0.01).unwrap();
        let a = dispatch(
            "bloom_add",
            &[
                bin_col(vec![empty.clone()], TypeId::BloomFilter),
                utf8_col(&["foo"]),
            ],
            1,
        )
        .unwrap()
        .unwrap();
        let b = dispatch(
            "bloom_add",
            &[
                bin_col(vec![empty], TypeId::BloomFilter),
                utf8_col(&["bar"]),
            ],
            1,
        )
        .unwrap()
        .unwrap();
        let merged = dispatch("bloom_merge", &[a, b], 1).unwrap().unwrap();
        let filter = bin_cell(&merged, 0);
        assert!(prob::bloom_contains(&filter, b"foo").unwrap());
        assert!(prob::bloom_contains(&filter, b"bar").unwrap());
    }

    #[test]
    fn hll_add_propagates_null_rows() {
        let valid = prob::hll_create(10).unwrap();
        let mut nulls = NullBitmap::none(2);
        nulls.set_null(1);
        let sketches = Column::with_nulls(
            ColumnData::Binary(vec![valid, Vec::new()]),
            nulls,
            TypeId::HyperLogLog,
        );
        let out = dispatch("hll_add", &[sketches, utf8_col(&["x", "y"])], 2)
            .unwrap()
            .unwrap();
        assert!(!out.nulls.is_null(0));
        assert!(out.nulls.is_null(1));
    }

    #[test]
    fn tdigest_quantile_finds_median() {
        let mut digest = prob::tdigest_create(100.0).unwrap();
        for i in 1..=100 {
            prob::tdigest_add(&mut digest, i as f64).unwrap();
        }
        let out = dispatch(
            "tdigest_quantile",
            &[bin_col(vec![digest], TypeId::TDigest), f64_col(&[0.5])],
            1,
        )
        .unwrap()
        .unwrap();
        assert!((f64_cell(&out, 0) - 50.5).abs() < 5.0);
    }

    #[test]
    fn correlation_on_json_arrays_is_one_for_linear_data() {
        let out = dispatch(
            "correlation",
            &[utf8_col(&["[1,2,3,4]"]), utf8_col(&["[2,4,6,8]"])],
            1,
        )
        .unwrap()
        .unwrap();
        assert!((f64_cell(&out, 0) - 1.0).abs() < 1e-10);
    }

    #[test]
    fn moving_average_emits_json_array() {
        let out = dispatch(
            "moving_average",
            &[utf8_col(&["[1,2,3,4,5]"]), i32_col(&[3])],
            1,
        )
        .unwrap()
        .unwrap();
        assert_eq!(out.type_id, TypeId::Array);
        let parsed: serde_json::Value = serde_json::from_slice(&bin_cell(&out, 0)).unwrap();
        let vals: Vec<f64> = parsed
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_f64().unwrap())
            .collect();
        assert_eq!(vals, vec![1.0, 1.5, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn malformed_json_array_yields_null_row() {
        let out = dispatch("stddev_pop", &[utf8_col(&["not json", "[1,2,3]"])], 2)
            .unwrap()
            .unwrap();
        assert!(out.nulls.is_null(0));
        assert!(!out.nulls.is_null(1));
    }

    #[test]
    fn linear_regression_returns_slope_intercept_r2() {
        let out = dispatch(
            "linear_regression",
            &[utf8_col(&["[0,1,2]"]), utf8_col(&["[1,3,5]"])],
            1,
        )
        .unwrap()
        .unwrap();
        let parsed: serde_json::Value = serde_json::from_slice(&bin_cell(&out, 0)).unwrap();
        let vals: Vec<f64> = parsed
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_f64().unwrap())
            .collect();
        assert!((vals[0] - 2.0).abs() < 1e-10);
        assert!((vals[1] - 1.0).abs() < 1e-10);
        assert!((vals[2] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn names_owned_by_mod_rs_return_none() {
        assert!(dispatch("cms_estimate", &[], 1).is_none());
        assert!(dispatch("hll_count", &[], 1).is_none());
        assert!(dispatch("bloom_contains", &[], 1).is_none());
        assert!(dispatch("no_such_fn", &[], 1).is_none());
    }
}
