//! Bridge between the executor's Column-based evaluation and zyron-types
//! primitive-based functions.
//!
//! This module dispatches function calls by name to the appropriate zyron-types
//! module, extracting inputs from columns and wrapping results back into columns.

use crate::batch::DataBatch;
use crate::column::{Column, ColumnData, NullBitmap, ScalarValue};
use crate::expr::evaluate;
use zyron_common::{Result, TypeId, ZyronError};
use zyron_planner::binder::BoundExpr;
use zyron_planner::logical::LogicalColumn;

/// Evaluates a function by delegating to the appropriate zyron-types module.
/// Returns an error if the function name is unknown.
pub fn evaluate_types_function(
    name: &str,
    args: &[BoundExpr],
    batch: &DataBatch,
    schema: &[LogicalColumn],
    params: &[ScalarValue],
) -> Result<Column> {
    // Evaluate all arguments into Columns first
    let evaluated_args: Vec<Column> = args
        .iter()
        .map(|a| evaluate(a, batch, schema, params))
        .collect::<Result<Vec<_>>>()?;

    let lower = name.to_lowercase();
    let num_rows = batch.num_rows;

    match lower.as_str() {
        // ---------- fuzzy (string -> numeric/string) ----------
        "levenshtein" => two_string_to_int(&evaluated_args, |a, b| {
            let mut buf = zyron_types::fuzzy::FuzzyBuffer::new();
            zyron_types::fuzzy::levenshtein(a, b, &mut buf) as i32
        }),
        "levenshtein_similarity" => two_string_to_float(&evaluated_args, |a, b| {
            let mut buf = zyron_types::fuzzy::FuzzyBuffer::new();
            zyron_types::fuzzy::levenshtein_similarity(a, b, &mut buf)
        }),
        "damerau_levenshtein" => two_string_to_int(&evaluated_args, |a, b| {
            zyron_types::fuzzy::damerau_levenshtein(a, b) as i32
        }),
        "jaro_similarity" => {
            two_string_to_float(&evaluated_args, zyron_types::fuzzy::jaro_similarity)
        }
        "jaro_winkler" => two_string_to_float(&evaluated_args, zyron_types::fuzzy::jaro_winkler),
        "soundex" => one_string_to_string(&evaluated_args, zyron_types::fuzzy::soundex),
        "metaphone" => one_string_to_string(&evaluated_args, zyron_types::fuzzy::metaphone),
        "nysiis" => one_string_to_string(&evaluated_args, zyron_types::fuzzy::nysiis),

        // ---------- string_ops ----------
        "initcap" => one_string_to_string(&evaluated_args, zyron_types::string_ops::initcap),
        "camel_case" => one_string_to_string(&evaluated_args, zyron_types::string_ops::camel_case),
        "snake_case" => one_string_to_string(&evaluated_args, zyron_types::string_ops::snake_case),
        "kebab_case" => one_string_to_string(&evaluated_args, zyron_types::string_ops::kebab_case),
        "pascal_case" => {
            one_string_to_string(&evaluated_args, zyron_types::string_ops::pascal_case)
        }
        "title_case" => one_string_to_string(&evaluated_args, zyron_types::string_ops::title_case),
        "slug" => one_string_to_string(&evaluated_args, zyron_types::string_ops::slug),
        "strip_html" => one_string_to_string(&evaluated_args, zyron_types::string_ops::strip_html),

        // ---------- formatting ----------
        "format_bytes" => one_int_to_string(&evaluated_args, |bytes| {
            zyron_types::formatting::format_bytes(bytes.unsigned_abs())
        }),
        "format_duration" => {
            one_float_to_string(&evaluated_args, zyron_types::formatting::format_duration)
        }
        "format_ordinal" => {
            one_int_to_string(&evaluated_args, zyron_types::formatting::format_ordinal)
        }

        // ---------- color ----------
        "color_from_rgb" | "color_rgb" => three_int_to_uint32(&evaluated_args, |r, g, b| {
            zyron_types::color::color_from_rgb(r as u8, g as u8, b as u8)
        }),
        "color_from_hex" | "color_hex" => one_string_to_uint32(&evaluated_args, |s| {
            zyron_types::color::color_from_hex(s).unwrap_or(0)
        }),
        "color_to_hex" => one_uint32_to_string(&evaluated_args, zyron_types::color::color_to_hex),

        // ---------- data_quality validators ----------
        "validate_email" => {
            one_string_to_bool(&evaluated_args, zyron_types::data_quality::validate_email)
        }
        "validate_url" => {
            one_string_to_bool(&evaluated_args, zyron_types::data_quality::validate_url)
        }
        "validate_json" => {
            one_string_to_bool(&evaluated_args, zyron_types::data_quality::validate_json)
        }
        "validate_uuid" => {
            one_string_to_bool(&evaluated_args, zyron_types::data_quality::validate_uuid)
        }
        "validate_credit_card" => one_string_to_bool(
            &evaluated_args,
            zyron_types::data_quality::validate_credit_card,
        ),
        "validate_isbn" => {
            one_string_to_bool(&evaluated_args, zyron_types::identifier::validate_isbn)
        }
        "validate_iban" => {
            one_string_to_bool(&evaluated_args, zyron_types::identifier::validate_iban)
        }
        "validate_ean" => {
            one_string_to_bool(&evaluated_args, zyron_types::identifier::validate_ean)
        }
        "validate_vin" => {
            one_string_to_bool(&evaluated_args, zyron_types::identifier::validate_vin)
        }
        "validate_issn" => {
            one_string_to_bool(&evaluated_args, zyron_types::identifier::validate_issn)
        }
        "validate_swift" => {
            one_string_to_bool(&evaluated_args, zyron_types::identifier::validate_swift)
        }
        "validate_ssn" => {
            one_string_to_bool(&evaluated_args, zyron_types::identifier::validate_ssn)
        }

        // ---------- encoding ----------
        "hex_encode" => one_bytes_to_string(&evaluated_args, zyron_types::encoding::hex_encode),
        "base58_encode" => {
            one_bytes_to_string(&evaluated_args, zyron_types::encoding::base58_encode)
        }
        "base32_encode" => {
            one_bytes_to_string(&evaluated_args, zyron_types::encoding::base32_encode)
        }
        "base64url_encode" => {
            one_bytes_to_string(&evaluated_args, zyron_types::encoding::base64url_encode)
        }
        "crc32" => one_bytes_to_int32(&evaluated_args, |b| zyron_types::encoding::crc32(b) as i32),
        "crc32c" => {
            one_bytes_to_int32(&evaluated_args, |b| zyron_types::encoding::crc32c(b) as i32)
        }
        "xxhash64" => one_bytes_to_int64(&evaluated_args, |b| {
            zyron_types::encoding::xxhash64(b) as i64
        }),

        // ---------- semver ----------
        "semver_compare" => two_string_to_int(&evaluated_args, |a, b| {
            let pa = zyron_types::semver::semver_parse(a).unwrap_or(0);
            let pb = zyron_types::semver::semver_parse(b).unwrap_or(0);
            zyron_types::semver::semver_compare(pa, pb)
        }),

        // ---------- id_gen (scalar returns for each row) ----------
        "gen_uuid_v4" | "uuid_v4" => {
            let mut data = Vec::with_capacity(num_rows);
            for _ in 0..num_rows {
                data.push(zyron_types::id_gen::uuid_v4());
            }
            Ok(Column::new(ColumnData::FixedBinary16(data), TypeId::Uuid))
        }
        "gen_uuid_v7" | "uuid_v7" => {
            let mut data = Vec::with_capacity(num_rows);
            for _ in 0..num_rows {
                data.push(zyron_types::id_gen::uuid_v7());
            }
            Ok(Column::new(ColumnData::FixedBinary16(data), TypeId::Uuid))
        }
        "gen_ulid" | "ulid" => {
            let data: Vec<String> = (0..num_rows).map(|_| zyron_types::id_gen::ulid()).collect();
            Ok(Column::new(ColumnData::Utf8(data), TypeId::Varchar))
        }
        "gen_nanoid" => {
            let len = evaluated_args
                .first()
                .and_then(|c| column_first_int(c))
                .unwrap_or(21) as usize;
            let data: Vec<String> = (0..num_rows)
                .map(|_| zyron_types::id_gen::nanoid(len))
                .collect();
            Ok(Column::new(ColumnData::Utf8(data), TypeId::Varchar))
        }
        "gen_ksuid" | "ksuid" => {
            let data: Vec<String> = (0..num_rows)
                .map(|_| zyron_types::id_gen::ksuid())
                .collect();
            Ok(Column::new(ColumnData::Utf8(data), TypeId::Varchar))
        }
        "gen_cuid2" | "cuid2" => {
            let data: Vec<String> = (0..num_rows)
                .map(|_| zyron_types::id_gen::cuid2())
                .collect();
            Ok(Column::new(ColumnData::Utf8(data), TypeId::Varchar))
        }
        "gen_tsid" | "tsid" => {
            let data: Vec<i64> = (0..num_rows).map(|_| zyron_types::id_gen::tsid()).collect();
            Ok(Column::new(ColumnData::Int64(data), TypeId::Int64))
        }

        // ---------- probabilistic (row-wise where applicable) ----------
        "bloom_contains" => two_bytes_to_bool(&evaluated_args, |filter, value| {
            zyron_types::probabilistic::bloom_contains(filter, value).unwrap_or(false)
        }),
        "hll_count" => one_bytes_to_int64(&evaluated_args, |sketch| {
            zyron_types::probabilistic::hll_count(sketch).unwrap_or(0) as i64
        }),
        "cms_estimate" => two_bytes_to_int64(&evaluated_args, |sketch, value| {
            zyron_types::probabilistic::cms_estimate(sketch, value).unwrap_or(0) as i64
        }),

        _ => Err(ZyronError::ExecutionError(format!(
            "unknown function: {}",
            name
        ))),
    }
}

// ---------------------------------------------------------------------------
// Helper functions for common dispatch patterns
// ---------------------------------------------------------------------------

fn arg_count_check(args: &[Column], expected: usize) -> Result<()> {
    if args.len() != expected {
        return Err(ZyronError::ExecutionError(format!(
            "expected {} arguments, got {}",
            expected,
            args.len()
        )));
    }
    Ok(())
}

fn column_strings(col: &Column) -> Result<Vec<&str>> {
    match &col.data {
        ColumnData::Utf8(v) => Ok(v.iter().map(|s| s.as_str()).collect()),
        _ => Err(ZyronError::ExecutionError("expected string column".into())),
    }
}

fn column_bytes(col: &Column) -> Result<Vec<&[u8]>> {
    match &col.data {
        ColumnData::Binary(v) => Ok(v.iter().map(|b| b.as_slice()).collect()),
        ColumnData::Utf8(v) => Ok(v.iter().map(|s| s.as_bytes()).collect()),
        _ => Err(ZyronError::ExecutionError("expected binary column".into())),
    }
}

fn column_ints(col: &Column) -> Result<Vec<i64>> {
    match &col.data {
        ColumnData::Int64(v) => Ok(v.clone()),
        ColumnData::Int32(v) => Ok(v.iter().map(|&x| x as i64).collect()),
        ColumnData::Int16(v) => Ok(v.iter().map(|&x| x as i64).collect()),
        ColumnData::Int8(v) => Ok(v.iter().map(|&x| x as i64).collect()),
        ColumnData::UInt32(v) => Ok(v.iter().map(|&x| x as i64).collect()),
        ColumnData::UInt64(v) => Ok(v.iter().map(|&x| x as i64).collect()),
        _ => Err(ZyronError::ExecutionError("expected integer column".into())),
    }
}

fn column_floats(col: &Column) -> Result<Vec<f64>> {
    match &col.data {
        ColumnData::Float64(v) => Ok(v.clone()),
        ColumnData::Float32(v) => Ok(v.iter().map(|&x| x as f64).collect()),
        ColumnData::Int64(v) => Ok(v.iter().map(|&x| x as f64).collect()),
        ColumnData::Int32(v) => Ok(v.iter().map(|&x| x as f64).collect()),
        _ => Err(ZyronError::ExecutionError("expected numeric column".into())),
    }
}

fn column_uint32s(col: &Column) -> Result<Vec<u32>> {
    match &col.data {
        ColumnData::UInt32(v) => Ok(v.clone()),
        ColumnData::Int32(v) => Ok(v.iter().map(|&x| x as u32).collect()),
        _ => Err(ZyronError::ExecutionError("expected u32 column".into())),
    }
}

fn column_first_int(col: &Column) -> Option<i64> {
    column_ints(col).ok().and_then(|v| v.first().copied())
}

fn one_string_to_string<F: Fn(&str) -> String>(args: &[Column], f: F) -> Result<Column> {
    arg_count_check(args, 1)?;
    let strings = column_strings(&args[0])?;
    let n = strings.len();
    let data: Vec<String> = strings.iter().map(|s| f(s)).collect();
    Ok(Column::with_nulls(
        ColumnData::Utf8(data),
        args[0].nulls.clone(),
        TypeId::Varchar,
    ))
    .map(|c| {
        let _ = n;
        c
    })
}

fn one_string_to_bool<F: Fn(&str) -> bool>(args: &[Column], f: F) -> Result<Column> {
    arg_count_check(args, 1)?;
    let strings = column_strings(&args[0])?;
    let data: Vec<bool> = strings.iter().map(|s| f(s)).collect();
    Ok(Column::with_nulls(
        ColumnData::Boolean(data),
        args[0].nulls.clone(),
        TypeId::Boolean,
    ))
}

fn one_string_to_uint32<F: Fn(&str) -> u32>(args: &[Column], f: F) -> Result<Column> {
    arg_count_check(args, 1)?;
    let strings = column_strings(&args[0])?;
    let data: Vec<u32> = strings.iter().map(|s| f(s)).collect();
    Ok(Column::with_nulls(
        ColumnData::UInt32(data),
        args[0].nulls.clone(),
        TypeId::Color,
    ))
}

fn one_uint32_to_string<F: Fn(u32) -> String>(args: &[Column], f: F) -> Result<Column> {
    arg_count_check(args, 1)?;
    let values = column_uint32s(&args[0])?;
    let data: Vec<String> = values.iter().map(|&v| f(v)).collect();
    Ok(Column::with_nulls(
        ColumnData::Utf8(data),
        args[0].nulls.clone(),
        TypeId::Varchar,
    ))
}

fn one_int_to_string<F: Fn(i64) -> String>(args: &[Column], f: F) -> Result<Column> {
    arg_count_check(args, 1)?;
    let values = column_ints(&args[0])?;
    let data: Vec<String> = values.iter().map(|&v| f(v)).collect();
    Ok(Column::with_nulls(
        ColumnData::Utf8(data),
        args[0].nulls.clone(),
        TypeId::Varchar,
    ))
}

fn one_float_to_string<F: Fn(f64) -> String>(args: &[Column], f: F) -> Result<Column> {
    arg_count_check(args, 1)?;
    let values = column_floats(&args[0])?;
    let data: Vec<String> = values.iter().map(|&v| f(v)).collect();
    Ok(Column::with_nulls(
        ColumnData::Utf8(data),
        args[0].nulls.clone(),
        TypeId::Varchar,
    ))
}

fn two_string_to_int<F: Fn(&str, &str) -> i32>(args: &[Column], f: F) -> Result<Column> {
    arg_count_check(args, 2)?;
    let a = column_strings(&args[0])?;
    let b = column_strings(&args[1])?;
    let n = a.len().min(b.len());
    let data: Vec<i32> = (0..n).map(|i| f(a[i], b[i])).collect();
    Ok(Column::new(ColumnData::Int32(data), TypeId::Int32))
}

fn two_string_to_float<F: Fn(&str, &str) -> f64>(args: &[Column], f: F) -> Result<Column> {
    arg_count_check(args, 2)?;
    let a = column_strings(&args[0])?;
    let b = column_strings(&args[1])?;
    let n = a.len().min(b.len());
    let data: Vec<f64> = (0..n).map(|i| f(a[i], b[i])).collect();
    Ok(Column::new(ColumnData::Float64(data), TypeId::Float64))
}

fn three_int_to_uint32<F: Fn(i64, i64, i64) -> u32>(args: &[Column], f: F) -> Result<Column> {
    arg_count_check(args, 3)?;
    let a = column_ints(&args[0])?;
    let b = column_ints(&args[1])?;
    let c = column_ints(&args[2])?;
    let n = a.len().min(b.len()).min(c.len());
    let data: Vec<u32> = (0..n).map(|i| f(a[i], b[i], c[i])).collect();
    Ok(Column::new(ColumnData::UInt32(data), TypeId::Color))
}

fn one_bytes_to_string<F: Fn(&[u8]) -> String>(args: &[Column], f: F) -> Result<Column> {
    arg_count_check(args, 1)?;
    let values = column_bytes(&args[0])?;
    let data: Vec<String> = values.iter().map(|b| f(b)).collect();
    Ok(Column::with_nulls(
        ColumnData::Utf8(data),
        args[0].nulls.clone(),
        TypeId::Varchar,
    ))
}

fn one_bytes_to_int32<F: Fn(&[u8]) -> i32>(args: &[Column], f: F) -> Result<Column> {
    arg_count_check(args, 1)?;
    let values = column_bytes(&args[0])?;
    let data: Vec<i32> = values.iter().map(|b| f(b)).collect();
    Ok(Column::with_nulls(
        ColumnData::Int32(data),
        args[0].nulls.clone(),
        TypeId::Int32,
    ))
}

fn one_bytes_to_int64<F: Fn(&[u8]) -> i64>(args: &[Column], f: F) -> Result<Column> {
    arg_count_check(args, 1)?;
    let values = column_bytes(&args[0])?;
    let data: Vec<i64> = values.iter().map(|b| f(b)).collect();
    Ok(Column::with_nulls(
        ColumnData::Int64(data),
        args[0].nulls.clone(),
        TypeId::Int64,
    ))
}

fn two_bytes_to_bool<F: Fn(&[u8], &[u8]) -> bool>(args: &[Column], f: F) -> Result<Column> {
    arg_count_check(args, 2)?;
    let a = column_bytes(&args[0])?;
    let b = column_bytes(&args[1])?;
    let n = a.len().min(b.len());
    let data: Vec<bool> = (0..n).map(|i| f(a[i], b[i])).collect();
    Ok(Column::new(ColumnData::Boolean(data), TypeId::Boolean))
}

fn two_bytes_to_int64<F: Fn(&[u8], &[u8]) -> i64>(args: &[Column], f: F) -> Result<Column> {
    arg_count_check(args, 2)?;
    let a = column_bytes(&args[0])?;
    let b = column_bytes(&args[1])?;
    let n = a.len().min(b.len());
    let data: Vec<i64> = (0..n).map(|i| f(a[i], b[i])).collect();
    Ok(Column::new(ColumnData::Int64(data), TypeId::Int64))
}

// Silence unused warnings (NullBitmap imported but may not be needed directly)
fn _use_nullbitmap() -> NullBitmap {
    NullBitmap::empty()
}
