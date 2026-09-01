//! Batch scoped data quality metrics for table expectations.
//!
//! Each function computes one verdict over the whole incoming batch and
//! broadcasts it, so an expectation like NULL_RATE(email, 0.05) passes or
//! fails the statement's rows together. An empty batch passes every
//! metric, there is nothing to judge.

use std::time::{SystemTime, UNIX_EPOCH};

use crate::column::{Column, ColumnData, ScalarValue};
use zyron_common::{Result, TypeId, ZyronError};

pub(super) fn dispatch(name: &str, args: &[Column], num_rows: usize) -> Option<Result<Column>> {
    Some(match name {
        "null_rate" => null_rate_impl(args, num_rows),
        "distinct_rate" => distinct_rate_impl(args, num_rows),
        "freshness" => freshness_impl(args, num_rows),
        "collated_compare" => collated_compare_impl(args, num_rows),
        "collation_sort_key" => collation_sort_key_impl(args, num_rows),
        "variant_extract" => variant_extract_impl(args, num_rows),
        "struct_field" => struct_field_impl(args, num_rows),
        "map_value" => map_value_impl(args, num_rows),
        "nested_json" => nested_json_impl(args, num_rows),
        _ => return None,
    })
}

/// The type id an embedded integer argument carries.
fn embedded_type_id(args: &[Column], idx: usize, sig: &str) -> Result<TypeId> {
    let raw = match args.get(idx).map(|c| &c.data) {
        Some(ColumnData::Int64(v)) => v.first().copied(),
        _ => None,
    };
    raw.and_then(|v| u8::try_from(v).ok())
        .and_then(TypeId::from_u8)
        .ok_or_else(|| ZyronError::ExecutionError(format!("{sig}: bad type argument")))
}

/// The bytes of a nested value for one row, when the column holds them.
fn nested_bytes(column: &Column, row: usize) -> Option<&[u8]> {
    if column.nulls.is_null(row) {
        return None;
    }
    match &column.data {
        ColumnData::Binary(v) => v.get(row).map(|b| b.as_slice()),
        _ => None,
    }
}

/// Builds the result column for one path step, given each row's bytes.
fn step_column(
    rows: usize,
    result_type: TypeId,
    mut value_of: impl FnMut(usize) -> ScalarValue,
) -> Column {
    let mut builder = crate::batch::ColumnBuilder::new(result_type, rows);
    for row in 0..rows {
        builder.push_owned(value_of(row));
    }
    builder.finish()
}

/// struct_field(value, ordinal, result_type) reads one declared field of a
/// stored STRUCT by the position the declaration fixes, which is two loads
/// rather than a search through the value for a name
fn struct_field_impl(args: &[Column], num_rows: usize) -> Result<Column> {
    const SIG: &str = "struct_field(value, ordinal, result_type)";
    if args.len() != 3 {
        return Err(ZyronError::ExecutionError(format!(
            "{SIG} takes exactly three arguments"
        )));
    }
    let ordinal = match &args[1].data {
        ColumnData::Int64(v) => v.first().copied().unwrap_or(0).max(0) as usize,
        _ => {
            return Err(ZyronError::ExecutionError(format!(
                "{SIG}: the ordinal must arrive as an embedded integer"
            )));
        }
    };
    let result_type = embedded_type_id(args, 2, SIG)?;
    let rows = num_rows.max(1);
    Ok(step_column(rows, result_type, |row| {
        match nested_bytes(&args[0], row) {
            Some(bytes) => crate::nested_codec::field_scalar(bytes, ordinal, result_type),
            None => ScalarValue::Null,
        }
    }))
}

/// map_value(value, key, key_type, result_type) finds one entry of a stored
/// MAP by binary search over its keys
fn map_value_impl(args: &[Column], num_rows: usize) -> Result<Column> {
    const SIG: &str = "map_value(value, key, key_type, result_type)";
    if args.len() != 4 {
        return Err(ZyronError::ExecutionError(format!(
            "{SIG} takes exactly four arguments"
        )));
    }
    let key_text = match &args[1].data {
        ColumnData::Utf8(v) => v.first().cloned().unwrap_or_default(),
        _ => {
            return Err(ZyronError::ExecutionError(format!(
                "{SIG}: the key must arrive as an embedded string"
            )));
        }
    };
    let key_type = embedded_type_id(args, 2, SIG)?;
    let result_type = embedded_type_id(args, 3, SIG)?;
    let rows = num_rows.max(1);
    // A key the declared key type cannot hold matches no entry, which is the
    // same answer an absent key gives
    let Some(key) = crate::nested_codec::encode_map_key(&key_text, key_type) else {
        return Ok(Column::null_column(result_type, rows));
    };
    Ok(step_column(rows, result_type, |row| {
        match nested_bytes(&args[0], row) {
            Some(bytes) => crate::nested_codec::key_scalar(bytes, &key, result_type),
            None => ScalarValue::Null,
        }
    }))
}

/// nested_json(value, shape) renders a stored STRUCT or MAP back to the JSON
/// text a client reads. The shape rides with the call because the executor
/// has no catalog to look one up in
fn nested_json_impl(args: &[Column], num_rows: usize) -> Result<Column> {
    const SIG: &str = "nested_json(value, shape)";
    if args.len() != 2 {
        return Err(ZyronError::ExecutionError(format!(
            "{SIG} takes exactly two arguments"
        )));
    }
    let shape_bytes = match &args[1].data {
        ColumnData::Binary(v) => v.first().cloned().unwrap_or_default(),
        _ => {
            return Err(ZyronError::ExecutionError(format!(
                "{SIG}: the shape must arrive as embedded bytes"
            )));
        }
    };
    let shape = zyron_catalog::schema::NestedShape::decode(&shape_bytes)
        .map_err(|e| ZyronError::ExecutionError(format!("{SIG}: {e}")))?;
    let rows = num_rows.max(1);
    let mut out = Vec::with_capacity(rows);
    let mut nulls = crate::column::NullBitmap::none(rows);
    for row in 0..rows {
        match nested_bytes(&args[0], row) {
            Some(bytes) => out.push(crate::nested_codec::render_json_text(bytes, &shape)),
            None => {
                nulls.set_null(row);
                out.push(String::new());
            }
        }
    }
    Ok(Column::with_nulls(
        ColumnData::Utf8(out),
        nulls,
        TypeId::Text,
    ))
}

/// variant_extract(value, 'dotted.path') resolves dotted access into a
/// VARIANT, STRUCT, or MAP value stored as JSON text. A missing path or a
/// non scalar target is NULL, matching JSON access semantics
fn variant_extract_impl(args: &[Column], num_rows: usize) -> Result<Column> {
    const SIG: &str = "variant_extract(value, path)";
    if args.len() != 2 {
        return Err(ZyronError::ExecutionError(format!(
            "{SIG} takes exactly two arguments"
        )));
    }
    let values = match &args[0].data {
        ColumnData::Utf8(v) => v,
        _ => {
            return Err(ZyronError::ExecutionError(format!(
                "{SIG}: the value must be a variant or struct stored as JSON text"
            )));
        }
    };
    let path = match &args[1].data {
        ColumnData::Utf8(v) => v.first().cloned().unwrap_or_default(),
        _ => {
            return Err(ZyronError::ExecutionError(format!(
                "{SIG}: the path must arrive as an embedded string"
            )));
        }
    };
    let rows = num_rows.max(1);
    let mut out = Vec::with_capacity(rows);
    let mut nulls = crate::column::NullBitmap::none(rows);
    for row in 0..rows {
        if args[0].nulls.is_null(row) {
            nulls.set_null(row);
            out.push(String::new());
            continue;
        }
        let json = values
            .get(row)
            .or_else(|| values.first())
            .map(String::as_str)
            .unwrap_or("");
        match crate::variant_shred::extract_scalar_text(json, &path) {
            Some(text) => out.push(text),
            None => {
                nulls.set_null(row);
                out.push(String::new());
            }
        }
    }
    Ok(Column::with_nulls(
        ColumnData::Utf8(out),
        nulls,
        TypeId::Text,
    ))
}

fn collation_config(sig: &str, args: &[Column]) -> Result<(String, String, bool)> {
    let text_at = |idx: usize, what: &str| -> Result<String> {
        match args.get(idx).map(|c| &c.data) {
            Some(ColumnData::Utf8(v)) => v
                .first()
                .cloned()
                .ok_or_else(|| ZyronError::ExecutionError(format!("{sig}: empty {what}"))),
            _ => Err(ZyronError::ExecutionError(format!(
                "{sig}: {what} must arrive as an embedded string"
            ))),
        }
    };
    let locale = text_at(args.len() - 3, "locale")?;
    let provider = text_at(args.len() - 2, "provider")?;
    let case_sensitive = match args.last().map(|c| &c.data) {
        Some(ColumnData::Boolean(v)) => v.first().copied().unwrap_or(true),
        _ => {
            return Err(ZyronError::ExecutionError(format!(
                "{sig}: case sensitivity must arrive as an embedded boolean"
            )));
        }
    };
    Ok((locale, provider, case_sensitive))
}

/// collated_compare(a, b, locale, provider, case_sensitive) compares two
/// strings under the collation, returning -1, 0, or 1 per row
fn collated_compare_impl(args: &[Column], num_rows: usize) -> Result<Column> {
    const SIG: &str = "collated_compare(a, b, locale, provider, case_sensitive)";
    if args.len() != 5 {
        return Err(ZyronError::ExecutionError(format!(
            "{SIG} takes exactly five arguments"
        )));
    }
    let (locale, provider, case_sensitive) = collation_config(SIG, args)?;
    let collator = zyron_types::collation::cached_collator(&locale, &provider, case_sensitive)?;
    let left = match &args[0].data {
        ColumnData::Utf8(v) => v,
        _ => {
            return Err(ZyronError::ExecutionError(format!(
                "{SIG}: both compared values must be strings"
            )));
        }
    };
    let right = match &args[1].data {
        ColumnData::Utf8(v) => v,
        _ => {
            return Err(ZyronError::ExecutionError(format!(
                "{SIG}: both compared values must be strings"
            )));
        }
    };
    let rows = num_rows.max(1);
    let mut out = Vec::with_capacity(rows);
    let mut nulls = crate::column::NullBitmap::none(rows);
    for row in 0..rows {
        if args[0].nulls.is_null(row) || args[1].nulls.is_null(row) {
            nulls.set_null(row);
            out.push(0);
            continue;
        }
        // A single value literal broadcasts against a full column
        let a = left
            .get(row)
            .or_else(|| left.first())
            .map(String::as_str)
            .unwrap_or("");
        let b = right
            .get(row)
            .or_else(|| right.first())
            .map(String::as_str)
            .unwrap_or("");
        out.push(match collator.compare(a, b) {
            std::cmp::Ordering::Less => -1,
            std::cmp::Ordering::Equal => 0,
            std::cmp::Ordering::Greater => 1,
        });
    }
    Ok(Column::with_nulls(
        ColumnData::Int32(out),
        nulls,
        TypeId::Int32,
    ))
}

/// collation_sort_key(text, locale, provider, case_sensitive) renders each
/// string as a byte key whose plain order matches the collated order, so
/// ORDER BY sorts collated text through the ordinary comparator
fn collation_sort_key_impl(args: &[Column], num_rows: usize) -> Result<Column> {
    const SIG: &str = "collation_sort_key(text, locale, provider, case_sensitive)";
    if args.len() != 4 {
        return Err(ZyronError::ExecutionError(format!(
            "{SIG} takes exactly four arguments"
        )));
    }
    let (_locale, _provider, case_sensitive) = collation_config(SIG, args)?;
    let values = match &args[0].data {
        ColumnData::Utf8(v) => v,
        _ => {
            return Err(ZyronError::ExecutionError(format!(
                "{SIG}: the value must be a string"
            )));
        }
    };
    let rows = num_rows.max(1);
    let mut out = Vec::with_capacity(rows);
    let mut nulls = crate::column::NullBitmap::none(rows);
    for row in 0..rows {
        if args[0].nulls.is_null(row) {
            nulls.set_null(row);
            out.push(Vec::new());
            continue;
        }
        let value = values
            .get(row)
            .or_else(|| values.first())
            .map(String::as_str)
            .unwrap_or("");
        out.push(zyron_types::collation::sort_key(value, case_sensitive));
    }
    Ok(Column::with_nulls(
        ColumnData::Binary(out),
        nulls,
        TypeId::Bytea,
    ))
}

fn broadcast_bool(value: bool, num_rows: usize) -> Column {
    let rows = num_rows.max(1);
    Column::new(ColumnData::Boolean(vec![value; rows]), TypeId::Boolean)
}

fn threshold_arg(sig: &str, args: &[Column], idx: usize) -> Result<f64> {
    let col = args.get(idx).ok_or_else(|| {
        ZyronError::ExecutionError(format!("{sig} requires a threshold argument"))
    })?;
    match &col.data {
        ColumnData::Float64(v) => v
            .first()
            .copied()
            .ok_or_else(|| ZyronError::ExecutionError(format!("{sig}: empty threshold"))),
        ColumnData::Float32(v) => Ok(v.first().copied().unwrap_or(0.0) as f64),
        ColumnData::Int64(v) => Ok(v.first().copied().unwrap_or(0) as f64),
        ColumnData::Int32(v) => Ok(v.first().copied().unwrap_or(0) as f64),
        _ => Err(ZyronError::ExecutionError(format!(
            "{sig}: threshold must be a number"
        ))),
    }
}

/// NULL_RATE(col, threshold): passes when the fraction of NULL values in
/// the batch stays at or under the threshold
fn null_rate_impl(args: &[Column], num_rows: usize) -> Result<Column> {
    const SIG: &str = "null_rate(column, threshold)";
    if args.len() != 2 {
        return Err(ZyronError::ExecutionError(format!(
            "{SIG} takes exactly two arguments"
        )));
    }
    let threshold = threshold_arg(SIG, args, 1)?;
    if num_rows == 0 {
        return Ok(broadcast_bool(true, num_rows));
    }
    let nulls = (0..num_rows).filter(|&r| args[0].nulls.is_null(r)).count();
    let rate = nulls as f64 / num_rows as f64;
    Ok(broadcast_bool(rate <= threshold, num_rows))
}

/// DISTINCT_RATE(col, threshold): passes when the fraction of distinct
/// non NULL values reaches the threshold
fn distinct_rate_impl(args: &[Column], num_rows: usize) -> Result<Column> {
    const SIG: &str = "distinct_rate(column, threshold)";
    if args.len() != 2 {
        return Err(ZyronError::ExecutionError(format!(
            "{SIG} takes exactly two arguments"
        )));
    }
    let threshold = threshold_arg(SIG, args, 1)?;
    if num_rows == 0 {
        return Ok(broadcast_bool(true, num_rows));
    }
    let mut seen = std::collections::HashSet::with_capacity(num_rows);
    let mut non_null = 0usize;
    for row in 0..num_rows {
        if args[0].nulls.is_null(row) {
            continue;
        }
        non_null += 1;
        seen.insert(format!("{:?}", args[0].get_scalar(row)));
    }
    if non_null == 0 {
        return Ok(broadcast_bool(true, num_rows));
    }
    let rate = seen.len() as f64 / non_null as f64;
    Ok(broadcast_bool(rate >= threshold, num_rows))
}

fn parse_age_micros(text: &str) -> Result<i64> {
    let t = text.trim();
    let (digits, scale_micros) = if let Some(rest) = t.strip_suffix("ms") {
        (rest, 1_000i64)
    } else if let Some(rest) = t.strip_suffix('s') {
        (rest, 1_000_000)
    } else if let Some(rest) = t.strip_suffix('m') {
        (rest, 60_000_000)
    } else if let Some(rest) = t.strip_suffix('h') {
        (rest, 3_600_000_000)
    } else if let Some(rest) = t.strip_suffix('d') {
        (rest, 86_400_000_000)
    } else {
        // A bare number counts seconds
        (t, 1_000_000)
    };
    digits
        .trim()
        .parse::<i64>()
        .map(|n| n.saturating_mul(scale_micros))
        .map_err(|_| {
            ZyronError::ExecutionError(format!(
                "max_age wants a duration like '30s', '5m', '1h', or '2d', got '{text}'"
            ))
        })
}

/// FRESHNESS(timestamp_col, max_age): passes when the newest timestamp in
/// the batch is no older than max_age
fn freshness_impl(args: &[Column], num_rows: usize) -> Result<Column> {
    const SIG: &str = "freshness(timestamp_column, max_age)";
    if args.len() != 2 {
        return Err(ZyronError::ExecutionError(format!(
            "{SIG} takes exactly two arguments"
        )));
    }
    let max_age_micros = match &args[1].data {
        ColumnData::Utf8(v) => parse_age_micros(v.first().map(String::as_str).unwrap_or(""))?,
        ColumnData::Int64(v) => v.first().copied().unwrap_or(0).saturating_mul(1_000_000),
        ColumnData::Int32(v) => (v.first().copied().unwrap_or(0) as i64).saturating_mul(1_000_000),
        _ => {
            return Err(ZyronError::ExecutionError(format!(
                "{SIG}: max_age must be a duration string or seconds"
            )));
        }
    };
    if num_rows == 0 {
        return Ok(broadcast_bool(true, num_rows));
    }
    let mut newest: Option<i64> = None;
    for row in 0..num_rows {
        if args[0].nulls.is_null(row) {
            continue;
        }
        let value = match args[0].get_scalar(row) {
            ScalarValue::Int64(v) => v,
            ScalarValue::Int32(v) => v as i64,
            _ => {
                return Err(ZyronError::ExecutionError(format!(
                    "{SIG}: the column must hold timestamps"
                )));
            }
        };
        newest = Some(newest.map_or(value, |cur| cur.max(value)));
    }
    let Some(newest) = newest else {
        // All values NULL, nothing fresh to measure, so the batch fails
        return Ok(broadcast_bool(false, num_rows));
    };
    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_micros() as i64)
        .unwrap_or(0);
    Ok(broadcast_bool(now - newest <= max_age_micros, num_rows))
}
