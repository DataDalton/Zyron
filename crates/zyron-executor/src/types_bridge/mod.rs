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
    let lower = name.to_lowercase();

    // ROW_DIFF with temporal-qualified args needs storage-side snapshot lookups
    // that the current evaluator does not have. Surface a clear error pointing
    // at the wiring gap rather than silently returning an empty diff
    if lower == "row_diff"
        && args
            .iter()
            .any(|a| matches!(a, BoundExpr::TemporalRef { .. }))
    {
        return Err(ZyronError::ExecutionError(
            "ROW_DIFF with AS OF / VERSION AS OF / IN BRANCH requires \
             system-versioned table snapshot lookups, which are not yet \
             plumbed through the executor. Use plain ROW_DIFF or fetch \
             the rows separately and diff via a CTE."
                .to_string(),
        ));
    }

    // Evaluate all arguments into Columns first
    let evaluated_args: Vec<Column> = args
        .iter()
        .map(|a| evaluate(a, batch, schema, params))
        .collect::<Result<Vec<_>>>()?;

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

        // ---------- diff and patch (JSON-shaped row args) ----------
        "row_diff" => two_string_to_string(&evaluated_args, |old_json, new_json| {
            row_diff_json_strings(old_json, new_json)
        }),
        "json_diff" => two_string_to_string(&evaluated_args, |a, b| json_diff_strings(a, b)),
        "json_patch" => two_string_to_string(&evaluated_args, |target, patch| {
            json_patch_strings(target, patch)
        }),
        "json_merge_patch" => two_string_to_string(&evaluated_args, |target, patch| {
            json_merge_patch_strings(target, patch)
        }),

        // ---------- JSON access operators (->, ->>, #>, #>>, @>, <@, ?, ?|, ?&) ----------
        "json_get" => two_string_to_nullable_string(&evaluated_args, |json, key| {
            json_get_impl(json, key, false)
        }),
        "json_get_text" => two_string_to_nullable_string(&evaluated_args, |json, key| {
            json_get_impl(json, key, true)
        }),
        "json_get_path" => two_string_to_nullable_string(&evaluated_args, |json, path| {
            json_get_path_impl(json, path, false)
        }),
        "json_get_path_text" => two_string_to_nullable_string(&evaluated_args, |json, path| {
            json_get_path_impl(json, path, true)
        }),
        "jsonb_contains" => two_string_to_bool(&evaluated_args, json_contains_impl),
        "jsonb_contained_by" => {
            two_string_to_bool(&evaluated_args, |a, b| json_contains_impl(b, a))
        }
        "jsonb_exists" => two_string_to_bool(&evaluated_args, json_exists_impl),
        "jsonb_exists_any" => two_string_to_bool(&evaluated_args, |json, keys| {
            json_exists_any_all(json, keys, false)
        }),
        "jsonb_exists_all" => two_string_to_bool(&evaluated_args, |json, keys| {
            json_exists_any_all(json, keys, true)
        }),
        "text_diff" => two_string_to_string(&evaluated_args, |a, b| text_diff_strings(a, b)),
        "text_patch" => two_string_to_string(&evaluated_args, |orig, patch| {
            text_patch_strings(orig, patch)
        }),

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

        // ---------- checksums ----------
        "xxhash32" => one_bytes_to_int32(&evaluated_args, |b| {
            zyron_types::checksum::xxhash32(b) as i32
        }),
        "adler32" => one_bytes_to_int32(&evaluated_args, |b| {
            zyron_types::checksum::adler32(b) as i32
        }),
        "fnvhash" | "fnv1a_64" => one_bytes_to_int64(&evaluated_args, |b| {
            zyron_types::checksum::fnv1a_64(b) as i64
        }),
        "siphash" => one_bytes_to_int64(&evaluated_args, |b| {
            zyron_types::checksum::siphash(b) as i64
        }),
        "city_hash" | "cityhash64" => one_bytes_to_int64(&evaluated_args, |b| {
            zyron_types::checksum::city_hash64(b) as i64
        }),

        // ---------- natural sort ----------
        "version_compare" => two_string_to_int(&evaluated_args, |a, b| {
            zyron_types::natural_sort::version_compare(a, b)
        }),
        "natural_compare" => two_string_to_int(&evaluated_args, |a, b| {
            match zyron_types::natural_sort::natural_compare(a, b) {
                std::cmp::Ordering::Less => -1,
                std::cmp::Ordering::Equal => 0,
                std::cmp::Ordering::Greater => 1,
            }
        }),
        "ip_compare" => {
            two_string_to_int(
                &evaluated_args,
                |a, b| match zyron_types::natural_sort::ip_compare(a, b) {
                    std::cmp::Ordering::Less => -1,
                    std::cmp::Ordering::Equal => 0,
                    std::cmp::Ordering::Greater => 1,
                },
            )
        }
        "path_compare" => {
            two_string_to_int(
                &evaluated_args,
                |a, b| match zyron_types::natural_sort::path_compare(a, b) {
                    std::cmp::Ordering::Less => -1,
                    std::cmp::Ordering::Equal => 0,
                    std::cmp::Ordering::Greater => 1,
                },
            )
        }
        "natural_sort_key" => one_string_to_bytes(&evaluated_args, |s| {
            zyron_types::natural_sort::natural_sort_key(s).into_bytes()
        }),

        // ---------- file detection ----------
        "detect_mime_type" => one_bytes_to_string(&evaluated_args, |b| {
            zyron_types::file_detect::detect_mime_type(b).to_string()
        }),
        "detect_encoding" => one_bytes_to_string(&evaluated_args, |b| {
            zyron_types::file_detect::detect_encoding(b).to_string()
        }),
        "is_binary" => one_bytes_to_bool(&evaluated_args, zyron_types::file_detect::is_binary),
        "file_extension" => one_string_to_string(&evaluated_args, |s| {
            zyron_types::file_detect::file_extension(s).to_string()
        }),

        // ---------- document processing ----------
        "markdown_to_html" => one_string_to_string(&evaluated_args, |s| {
            zyron_types::document::markdown_to_html(s)
        }),
        "html_to_text" => {
            one_string_to_string(&evaluated_args, |s| zyron_types::document::html_to_text(s))
        }
        "html_to_markdown" => one_string_to_string(&evaluated_args, |s| {
            zyron_types::document::html_to_markdown(s)
        }),
        "sanitize_html" => one_string_to_string(&evaluated_args, |s| {
            // Default allow-list of safe tags. Callers wanting a custom set
            // pass arrays through a future overload.
            zyron_types::document::sanitize_html(
                s,
                &[
                    "p",
                    "h1",
                    "h2",
                    "h3",
                    "h4",
                    "h5",
                    "h6",
                    "strong",
                    "em",
                    "code",
                    "pre",
                    "ul",
                    "ol",
                    "li",
                    "a",
                    "br",
                    "hr",
                    "blockquote",
                ],
            )
        }),

        // ---------- barcode/QR ----------
        "qr_encode" => one_string_to_bytes(&evaluated_args, |s| {
            zyron_types::barcode::qr_encode(s, zyron_types::barcode::QrErrorCorrection::M)
                .unwrap_or_default()
        }),
        "qr_decode" => one_bytes_to_string(&evaluated_args, |b| {
            zyron_types::barcode::qr_decode(b).unwrap_or_default()
        }),
        "data_matrix_encode" => one_string_to_bytes(&evaluated_args, |s| {
            zyron_types::barcode::data_matrix_encode(s).unwrap_or_default()
        }),
        "data_matrix_decode" => one_bytes_to_string(&evaluated_args, |b| {
            zyron_types::barcode::data_matrix_decode(b).unwrap_or_default()
        }),
        "barcode_decode" => one_bytes_to_string(&evaluated_args, |b| {
            zyron_types::barcode::barcode_decode(b)
                .map(|(s, _)| s)
                .unwrap_or_default()
        }),

        other => dispatch_extended(other, &evaluated_args, num_rows).unwrap_or_else(|| {
            Err(ZyronError::ExecutionError(format!(
                "unknown function: {}",
                name
            )))
        }),
    }
}

/// Chains the per family dispatch modules, first Some wins. Array typed
/// results are encoded as JSON text bytes inside Binary cells, opaque
/// domain values (money, quantity, range, sketches) use their module's
/// byte serialization
fn dispatch_extended(name: &str, args: &[Column], num_rows: usize) -> Option<Result<Column>> {
    bitfield_crypto::dispatch(name, args, num_rows)
        .or_else(|| color_fingerprint::dispatch(name, args, num_rows))
        .or_else(|| idgen_identifier::dispatch(name, args, num_rows))
        .or_else(|| network_url::dispatch(name, args, num_rows))
        .or_else(|| money_quantity::dispatch(name, args, num_rows))
        .or_else(|| cron_range_time::dispatch(name, args, num_rows))
        .or_else(|| regex_strings::dispatch(name, args, num_rows))
        .or_else(|| state_rate_tree::dispatch(name, args, num_rows))
        .or_else(|| prob_statistics::dispatch(name, args, num_rows))
        .or_else(|| matrix_geo::dispatch(name, args, num_rows))
        .or_else(|| financial_ts::dispatch(name, args, num_rows))
        .or_else(|| misc_domains::dispatch(name, args, num_rows))
}

mod bitfield_crypto;
mod color_fingerprint;
mod cron_range_time;
mod financial_ts;
mod idgen_identifier;
mod matrix_geo;
mod misc_domains;
mod money_quantity;
mod network_url;
mod prob_statistics;
mod regex_strings;
mod state_rate_tree;

fn one_string_to_bytes<F: Fn(&str) -> Vec<u8>>(args: &[Column], f: F) -> Result<Column> {
    arg_count_check(args, 1)?;
    let strings = column_strings(&args[0])?;
    let data: Vec<Vec<u8>> = strings.iter().map(|s| f(s)).collect();
    Ok(Column::with_nulls(
        ColumnData::Binary(data),
        args[0].nulls.clone(),
        TypeId::Bytea,
    ))
}

fn one_bytes_to_bool<F: Fn(&[u8]) -> bool>(args: &[Column], f: F) -> Result<Column> {
    arg_count_check(args, 1)?;
    let values = column_bytes(&args[0])?;
    let data: Vec<bool> = values.iter().map(|b| f(b)).collect();
    Ok(Column::with_nulls(
        ColumnData::Boolean(data),
        args[0].nulls.clone(),
        TypeId::Boolean,
    ))
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

fn two_string_to_string<F: Fn(&str, &str) -> String>(args: &[Column], f: F) -> Result<Column> {
    arg_count_check(args, 2)?;
    let a = column_strings(&args[0])?;
    let b = column_strings(&args[1])?;
    if a.len() != b.len() {
        return Err(ZyronError::ExecutionError(
            "two_string_to_string: column length mismatch".to_string(),
        ));
    }
    let data: Vec<String> = a.iter().zip(b.iter()).map(|(x, y)| f(x, y)).collect();
    Ok(Column::with_nulls(
        ColumnData::Utf8(data),
        args[0].nulls.clone(),
        TypeId::Varchar,
    ))
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

// ---------------------------------------------------------------------------
// Diff/patch JSON adapters
// ---------------------------------------------------------------------------

/// Diff two JSON object strings as if they were row tuples. Each top-level
/// key is treated as a column name, values are stringified for comparison
/// Returns a JSON array of {column, old_value, new_value}
fn row_diff_json_strings(old_json: &str, new_json: &str) -> String {
    let old_v: serde_json::Value = match serde_json::from_str(old_json) {
        Ok(v) => v,
        Err(_) => return format!(r#"{{"error":"old_json not valid JSON"}}"#),
    };
    let new_v: serde_json::Value = match serde_json::from_str(new_json) {
        Ok(v) => v,
        Err(_) => return format!(r#"{{"error":"new_json not valid JSON"}}"#),
    };
    let old_map = old_v.as_object();
    let new_map = new_v.as_object();
    if old_map.is_none() || new_map.is_none() {
        return r#"{"error":"row_diff requires JSON object inputs"}"#.to_string();
    }
    let old_map = old_map.unwrap();
    let new_map = new_map.unwrap();
    let mut keys: std::collections::BTreeSet<&str> = std::collections::BTreeSet::new();
    keys.extend(old_map.keys().map(|s| s.as_str()));
    keys.extend(new_map.keys().map(|s| s.as_str()));
    let mut changes = Vec::new();
    for k in keys {
        let o = old_map.get(k);
        let n = new_map.get(k);
        let same = match (o, n) {
            (Some(a), Some(b)) => a == b,
            (None, None) => true,
            _ => false,
        };
        if !same {
            let old_repr = o.map(|v| value_as_string(v));
            let new_repr = n.map(|v| value_as_string(v));
            let mut entry = serde_json::Map::new();
            entry.insert(
                "column".to_string(),
                serde_json::Value::String(k.to_string()),
            );
            entry.insert(
                "old_value".to_string(),
                old_repr
                    .map(serde_json::Value::String)
                    .unwrap_or(serde_json::Value::Null),
            );
            entry.insert(
                "new_value".to_string(),
                new_repr
                    .map(serde_json::Value::String)
                    .unwrap_or(serde_json::Value::Null),
            );
            changes.push(serde_json::Value::Object(entry));
        }
    }
    serde_json::Value::Array(changes).to_string()
}

fn value_as_string(v: &serde_json::Value) -> String {
    match v {
        serde_json::Value::String(s) => s.clone(),
        other => other.to_string(),
    }
}

fn json_diff_strings(a: &str, b: &str) -> String {
    zyron_types::diff::json_diff(a, b)
        .unwrap_or_else(|e| format!(r#"{{"error":"{}"}}"#, e.to_string().replace('"', "\\\"")))
}

fn json_patch_strings(target: &str, patch: &str) -> String {
    zyron_types::diff::json_patch(target, patch)
        .unwrap_or_else(|e| format!(r#"{{"error":"{}"}}"#, e.to_string().replace('"', "\\\"")))
}

fn json_merge_patch_strings(target: &str, patch: &str) -> String {
    zyron_types::diff::json_merge_patch(target, patch)
        .unwrap_or_else(|e| format!(r#"{{"error":"{}"}}"#, e.to_string().replace('"', "\\\"")))
}

fn text_diff_strings(a: &str, b: &str) -> String {
    zyron_types::diff::text_diff(a, b)
}

fn text_patch_strings(orig: &str, patch_json: &str) -> String {
    zyron_types::diff::text_patch(orig, patch_json)
        .unwrap_or_else(|e| format!(r#"{{"error":"{}"}}"#, e.to_string().replace('"', "\\\"")))
}

// ---------------------------------------------------------------------------
// JSON access operators
// ---------------------------------------------------------------------------

/// Applies a two-string predicate row-wise, producing a boolean column. A NULL
/// in either input yields NULL.
fn two_string_to_bool<F: Fn(&str, &str) -> bool>(args: &[Column], f: F) -> Result<Column> {
    arg_count_check(args, 2)?;
    let a = column_strings(&args[0])?;
    let b = column_strings(&args[1])?;
    if a.len() != b.len() {
        return Err(ZyronError::ExecutionError(
            "two_string_to_bool: column length mismatch".to_string(),
        ));
    }
    let mut data = Vec::with_capacity(a.len());
    let mut nulls = NullBitmap::none(a.len());
    for (i, (x, y)) in a.iter().zip(b.iter()).enumerate() {
        if args[0].nulls.is_null(i) || args[1].nulls.is_null(i) {
            data.push(false);
            nulls.set_null(i);
        } else {
            data.push(f(x, y));
        }
    }
    Ok(Column::with_nulls(
        ColumnData::Boolean(data),
        nulls,
        TypeId::Boolean,
    ))
}

/// Applies a two-string function row-wise where the result may be absent. A
/// NULL input or a `None` result yields SQL NULL, matching Postgres `->`/`->>`
/// returning NULL for a missing key.
fn two_string_to_nullable_string<F: Fn(&str, &str) -> Option<String>>(
    args: &[Column],
    f: F,
) -> Result<Column> {
    arg_count_check(args, 2)?;
    let a = column_strings(&args[0])?;
    let b = column_strings(&args[1])?;
    if a.len() != b.len() {
        return Err(ZyronError::ExecutionError(
            "two_string_to_nullable_string: column length mismatch".to_string(),
        ));
    }
    let mut data = Vec::with_capacity(a.len());
    let mut nulls = NullBitmap::none(a.len());
    for (i, (x, y)) in a.iter().zip(b.iter()).enumerate() {
        if args[0].nulls.is_null(i) || args[1].nulls.is_null(i) {
            data.push(String::new());
            nulls.set_null(i);
            continue;
        }
        match f(x, y) {
            Some(s) => data.push(s),
            None => {
                data.push(String::new());
                nulls.set_null(i);
            }
        }
    }
    Ok(Column::with_nulls(
        ColumnData::Utf8(data),
        nulls,
        TypeId::Varchar,
    ))
}

/// Serializes a resolved JSON value for output. `as_text` unwraps a JSON string
/// to its raw text and maps JSON null to SQL NULL (Postgres `->>` semantics);
/// otherwise the value is serialized as JSON text (Postgres `->`).
fn json_output(v: &serde_json::Value, as_text: bool) -> Option<String> {
    if as_text {
        match v {
            serde_json::Value::Null => None,
            serde_json::Value::String(s) => Some(s.clone()),
            other => Some(other.to_string()),
        }
    } else {
        Some(v.to_string())
    }
}

/// Resolves an array element index, supporting Postgres negative indexing.
fn json_array_index(arr: &[serde_json::Value], key: &str) -> Option<usize> {
    let idx: i64 = key.parse().ok()?;
    let resolved = if idx < 0 { arr.len() as i64 + idx } else { idx };
    if resolved < 0 || resolved as usize >= arr.len() {
        None
    } else {
        Some(resolved as usize)
    }
}

/// `->` / `->>`: object field by key or array element by integer index.
fn json_get_impl(json: &str, key: &str, as_text: bool) -> Option<String> {
    let v: serde_json::Value = serde_json::from_str(json).ok()?;
    let got = match &v {
        serde_json::Value::Object(map) => map.get(key)?,
        serde_json::Value::Array(arr) => arr.get(json_array_index(arr, key)?)?,
        _ => return None,
    };
    json_output(got, as_text)
}

/// Parses a Postgres text-array path literal like `{a,b,0}` into its elements.
fn parse_text_array(s: &str) -> Vec<String> {
    let trimmed = s.trim();
    let inner = trimmed
        .strip_prefix('{')
        .and_then(|x| x.strip_suffix('}'))
        .unwrap_or(trimmed);
    if inner.trim().is_empty() {
        return Vec::new();
    }
    inner
        .split(',')
        .map(|p| p.trim().trim_matches('"').to_string())
        .collect()
}

/// `#>` / `#>>`: extract by a text-array path, walking objects and arrays.
fn json_get_path_impl(json: &str, path: &str, as_text: bool) -> Option<String> {
    let v: serde_json::Value = serde_json::from_str(json).ok()?;
    let mut cur = &v;
    for key in parse_text_array(path) {
        cur = match cur {
            serde_json::Value::Object(map) => map.get(&key)?,
            serde_json::Value::Array(arr) => arr.get(json_array_index(arr, &key)?)?,
            _ => return None,
        };
    }
    json_output(cur, as_text)
}

/// `@>`: deep containment of `b` within `a`.
fn json_contains_impl(a: &str, b: &str) -> bool {
    match (
        serde_json::from_str::<serde_json::Value>(a),
        serde_json::from_str::<serde_json::Value>(b),
    ) {
        (Ok(av), Ok(bv)) => json_value_contains(&av, &bv),
        _ => false,
    }
}

fn json_value_contains(a: &serde_json::Value, b: &serde_json::Value) -> bool {
    use serde_json::Value;
    match (a, b) {
        (Value::Object(am), Value::Object(bm)) => bm
            .iter()
            .all(|(k, bv)| am.get(k).is_some_and(|av| json_value_contains(av, bv))),
        (Value::Array(aa), Value::Array(ba)) => ba
            .iter()
            .all(|be| aa.iter().any(|ae| json_value_contains(ae, be))),
        _ => a == b,
    }
}

/// `?`: does `key` exist as a top-level object key or array string element.
fn json_exists_impl(json: &str, key: &str) -> bool {
    match serde_json::from_str::<serde_json::Value>(json) {
        Ok(serde_json::Value::Object(m)) => m.contains_key(key),
        Ok(serde_json::Value::Array(a)) => a.iter().any(|v| v.as_str() == Some(key)),
        Ok(serde_json::Value::String(s)) => s == key,
        _ => false,
    }
}

/// `?|` (any) / `?&` (all): does the json contain any/all of the given keys.
fn json_exists_any_all(json: &str, keys: &str, require_all: bool) -> bool {
    let parsed = serde_json::from_str::<serde_json::Value>(json).ok();
    let key_list = parse_text_array(keys);
    if key_list.is_empty() {
        return require_all;
    }
    let exists = |k: &str| match &parsed {
        Some(serde_json::Value::Object(m)) => m.contains_key(k),
        Some(serde_json::Value::Array(a)) => a.iter().any(|v| v.as_str() == Some(k)),
        Some(serde_json::Value::String(s)) => s == k,
        _ => false,
    };
    if require_all {
        key_list.iter().all(|k| exists(k))
    } else {
        key_list.iter().any(|k| exists(k))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn row_diff_json_emits_changed_columns_only() {
        let out = row_diff_json_strings(
            r#"{"id":1,"name":"alice","city":"nyc"}"#,
            r#"{"id":1,"name":"alice","city":"sf"}"#,
        );
        let parsed: serde_json::Value = serde_json::from_str(&out).unwrap();
        let arr = parsed.as_array().expect("expected array");
        assert_eq!(arr.len(), 1);
        let entry = arr[0].as_object().unwrap();
        assert_eq!(entry["column"], serde_json::Value::String("city".into()));
        assert_eq!(entry["old_value"], serde_json::Value::String("nyc".into()));
        assert_eq!(entry["new_value"], serde_json::Value::String("sf".into()));
    }

    #[test]
    fn row_diff_json_handles_added_and_removed_columns() {
        let out = row_diff_json_strings(r#"{"a":1,"b":2}"#, r#"{"a":1,"c":3}"#);
        let parsed: serde_json::Value = serde_json::from_str(&out).unwrap();
        let arr = parsed.as_array().expect("expected array");
        assert_eq!(arr.len(), 2);
        let mut cols: Vec<String> = arr
            .iter()
            .map(|e| e["column"].as_str().unwrap().to_string())
            .collect();
        cols.sort();
        assert_eq!(cols, vec!["b".to_string(), "c".to_string()]);
    }

    #[test]
    fn row_diff_invalid_json_returns_error_object() {
        let out = row_diff_json_strings("not json", "{}");
        assert!(out.contains("error"));
    }
}
