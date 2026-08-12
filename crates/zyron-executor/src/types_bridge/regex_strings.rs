//! Dispatch arms for regex_type, string_ops extraction and truncation,
//! json_schema validation, word level diff, and data_quality date checks
//!
//! Array and Composite results are encoded as JSON text bytes inside Binary
//! cells per the types bridge physical value contract

use crate::column::{Column, ColumnData, NullBitmap};
use zyron_common::{Result, TypeId, ZyronError};

pub(super) fn dispatch(name: &str, args: &[Column], num_rows: usize) -> Option<Result<Column>> {
    // no zero-arg generators in this family, every fn derives rows from args
    let _ = num_rows;
    Some(match name {
        "regex_match" => regex_match_col(args),
        "regex_match_compiled" => regex_match_compiled_col(args),
        "regex_find" => regex_find_col(args),
        "regex_find_compiled" => regex_find_compiled_col(args),
        "regex_find_all" => regex_find_all_col(args),
        "regex_capture" => regex_capture_col(args),
        "regex_split" => regex_split_col(args),
        "regex_count" => regex_count_col(args),
        "regex_replace" => regex_replace_col(args),
        "regex_replace_all" => regex_replace_all_col(args),
        "regex_compile" => regex_compile_col(args),
        "extract_emails" => extract_list_col(
            args,
            "extract_emails",
            zyron_types::string_ops::extract_emails,
        ),
        "extract_urls" => {
            extract_list_col(args, "extract_urls", zyron_types::string_ops::extract_urls)
        }
        "extract_phone_numbers" => extract_list_col(
            args,
            "extract_phone_numbers",
            zyron_types::string_ops::extract_phone_numbers,
        ),
        "truncate_chars" => truncate_chars_col(args),
        "truncate_words" => truncate_words_col(args),
        "json_schema_validate" | "validate_json_schema" => json_schema_validate_col(args, name),
        "json_schema_errors" => json_schema_errors_col(args),
        "text_diff_words" => text_diff_words_col(args),
        "is_valid_date" => is_valid_date_col(args),
        _ => return None,
    })
}

// ---------------------------------------------------------------------------
// column readers and row-wise drivers
// ---------------------------------------------------------------------------

fn expect_args(args: &[Column], expected: usize, sig: &str) -> Result<()> {
    if args.len() != expected {
        let noun = if expected == 1 {
            "argument"
        } else {
            "arguments"
        };
        return Err(ZyronError::ExecutionError(format!(
            "{} takes exactly {} {}",
            sig, expected, noun
        )));
    }
    Ok(())
}

fn row_count(args: &[Column], fn_name: &str) -> Result<usize> {
    let n = args.first().map(|c| c.data.len()).unwrap_or(0);
    for c in args {
        if c.data.len() != n {
            return Err(ZyronError::ExecutionError(format!(
                "{} argument column length mismatch",
                fn_name
            )));
        }
    }
    Ok(n)
}

fn utf8_rows<'a>(col: &'a Column, fn_name: &str) -> Result<&'a [String]> {
    match &col.data {
        ColumnData::Utf8(v) => Ok(v),
        _ => Err(ZyronError::ExecutionError(format!(
            "{} expects a string argument",
            fn_name
        ))),
    }
}

fn binary_rows<'a>(col: &'a Column, fn_name: &str) -> Result<Vec<&'a [u8]>> {
    match &col.data {
        ColumnData::Binary(v) => Ok(v.iter().map(|b| b.as_slice()).collect()),
        ColumnData::Utf8(v) => Ok(v.iter().map(|s| s.as_bytes()).collect()),
        _ => Err(ZyronError::ExecutionError(format!(
            "{} expects a binary argument",
            fn_name
        ))),
    }
}

fn int_rows(col: &Column, fn_name: &str) -> Result<Vec<i64>> {
    super::column_ints(col)
        .map_err(|_| ZyronError::ExecutionError(format!("{} expects an integer argument", fn_name)))
}

fn any_null(args: &[Column], row: usize) -> bool {
    args.iter().any(|c| c.nulls.is_null(row))
}

/// Runs f per row over one string input, a NULL input row or a None result
/// yields NULL, matching the lenient parser-style arms in the parent module
fn one_string_nullable<T, F>(
    args: &[Column],
    sig: &str,
    fn_name: &str,
    default: T,
    f: F,
) -> Result<(Vec<T>, NullBitmap)>
where
    T: Clone,
    F: Fn(&str) -> Option<T>,
{
    expect_args(args, 1, sig)?;
    let n = row_count(args, fn_name)?;
    let a = utf8_rows(&args[0], fn_name)?;
    let mut out = Vec::with_capacity(n);
    let mut nulls = NullBitmap::none(n);
    for i in 0..n {
        if args[0].nulls.is_null(i) {
            out.push(default.clone());
            nulls.set_null(i);
            continue;
        }
        match f(&a[i]) {
            Some(v) => out.push(v),
            None => {
                out.push(default.clone());
                nulls.set_null(i);
            }
        }
    }
    Ok((out, nulls))
}

/// Runs f per row over two string inputs with the same NULL leniency, an
/// invalid regex pattern therefore NULLs the row instead of failing the batch
fn two_string_nullable<T, F>(
    args: &[Column],
    sig: &str,
    fn_name: &str,
    default: T,
    f: F,
) -> Result<(Vec<T>, NullBitmap)>
where
    T: Clone,
    F: Fn(&str, &str) -> Option<T>,
{
    expect_args(args, 2, sig)?;
    let n = row_count(args, fn_name)?;
    let a = utf8_rows(&args[0], fn_name)?;
    let b = utf8_rows(&args[1], fn_name)?;
    let mut out = Vec::with_capacity(n);
    let mut nulls = NullBitmap::none(n);
    for i in 0..n {
        if any_null(args, i) {
            out.push(default.clone());
            nulls.set_null(i);
            continue;
        }
        match f(&a[i], &b[i]) {
            Some(v) => out.push(v),
            None => {
                out.push(default.clone());
                nulls.set_null(i);
            }
        }
    }
    Ok((out, nulls))
}

/// Runs f per row over three string inputs, same NULL semantics
fn three_string_nullable<T, F>(
    args: &[Column],
    sig: &str,
    fn_name: &str,
    default: T,
    f: F,
) -> Result<(Vec<T>, NullBitmap)>
where
    T: Clone,
    F: Fn(&str, &str, &str) -> Option<T>,
{
    expect_args(args, 3, sig)?;
    let n = row_count(args, fn_name)?;
    let a = utf8_rows(&args[0], fn_name)?;
    let b = utf8_rows(&args[1], fn_name)?;
    let c = utf8_rows(&args[2], fn_name)?;
    let mut out = Vec::with_capacity(n);
    let mut nulls = NullBitmap::none(n);
    for i in 0..n {
        if any_null(args, i) {
            out.push(default.clone());
            nulls.set_null(i);
            continue;
        }
        match f(&a[i], &b[i], &c[i]) {
            Some(v) => out.push(v),
            None => {
                out.push(default.clone());
                nulls.set_null(i);
            }
        }
    }
    Ok((out, nulls))
}

/// Runs f per row over a text column and a compiled-regex Binary column,
/// undecodable or invalid pattern bytes yield NULL for that row
fn text_compiled_nullable<T, F>(
    args: &[Column],
    sig: &str,
    fn_name: &str,
    default: T,
    f: F,
) -> Result<(Vec<T>, NullBitmap)>
where
    T: Clone,
    F: Fn(&str, &zyron_types::regex_type::CompiledRegex) -> Option<T>,
{
    expect_args(args, 2, sig)?;
    let n = row_count(args, fn_name)?;
    let texts = utf8_rows(&args[0], fn_name)?;
    let compiled = binary_rows(&args[1], fn_name)?;
    let mut out = Vec::with_capacity(n);
    let mut nulls = NullBitmap::none(n);
    for i in 0..n {
        if any_null(args, i) {
            out.push(default.clone());
            nulls.set_null(i);
            continue;
        }
        let result = compiled_from_bytes(compiled[i]).and_then(|c| f(&texts[i], &c));
        match result {
            Some(v) => out.push(v),
            None => {
                out.push(default.clone());
                nulls.set_null(i);
            }
        }
    }
    Ok((out, nulls))
}

/// CompiledRegex has no byte serialization in zyron-types, so a compiled
/// regex cell stores the validated pattern source and consumers recompile
fn compiled_from_bytes(bytes: &[u8]) -> Option<zyron_types::regex_type::CompiledRegex> {
    let pattern = std::str::from_utf8(bytes).ok()?;
    zyron_types::regex_type::regex_compile(pattern).ok()
}

// ---------------------------------------------------------------------------
// JSON cell encoders
// ---------------------------------------------------------------------------

fn json_string_array(items: &[String]) -> Vec<u8> {
    let arr: Vec<serde_json::Value> = items
        .iter()
        .map(|s| serde_json::Value::String(s.clone()))
        .collect();
    serde_json::Value::Array(arr).to_string().into_bytes()
}

fn span_json(start: usize, end: usize) -> Vec<u8> {
    serde_json::json!({"start": start, "end": end})
        .to_string()
        .into_bytes()
}

fn spans_json(spans: &[(usize, usize)]) -> Vec<u8> {
    let arr: Vec<serde_json::Value> = spans
        .iter()
        .map(|&(s, e)| serde_json::json!([s, e]))
        .collect();
    serde_json::Value::Array(arr).to_string().into_bytes()
}

fn capture_json(groups: &[Option<String>]) -> Vec<u8> {
    let arr: Vec<serde_json::Value> = groups
        .iter()
        .map(|g| match g {
            Some(s) => serde_json::Value::String(s.clone()),
            None => serde_json::Value::Null,
        })
        .collect();
    serde_json::Value::Array(arr).to_string().into_bytes()
}

/// DiffOp derives no Serialize, hand encoded as {"op","text"} objects
fn diff_ops_json(ops: &[zyron_types::diff::DiffOp]) -> Vec<u8> {
    use zyron_types::diff::DiffOp;
    let arr: Vec<serde_json::Value> = ops
        .iter()
        .map(|op| {
            let (kind, text) = match op {
                DiffOp::Equal(t) => ("equal", t),
                DiffOp::Insert(t) => ("insert", t),
                DiffOp::Delete(t) => ("delete", t),
            };
            serde_json::json!({"op": kind, "text": text})
        })
        .collect();
    serde_json::Value::Array(arr).to_string().into_bytes()
}

// ---------------------------------------------------------------------------
// regex_type
// ---------------------------------------------------------------------------

fn regex_match_col(args: &[Column]) -> Result<Column> {
    let (data, nulls) = two_string_nullable(
        args,
        "regex_match(text, pattern)",
        "regex_match",
        false,
        |t, p| zyron_types::regex_type::regex_match(t, p).ok(),
    )?;
    Ok(Column::with_nulls(
        ColumnData::Boolean(data),
        nulls,
        TypeId::Boolean,
    ))
}

fn regex_match_compiled_col(args: &[Column]) -> Result<Column> {
    let (data, nulls) = text_compiled_nullable(
        args,
        "regex_match_compiled(text, regex)",
        "regex_match_compiled",
        false,
        |t, c| zyron_types::regex_type::regex_match_compiled(t, c).ok(),
    )?;
    Ok(Column::with_nulls(
        ColumnData::Boolean(data),
        nulls,
        TypeId::Boolean,
    ))
}

// composite span cell is JSON object bytes {"start","end"}, no match is NULL
fn regex_find_col(args: &[Column]) -> Result<Column> {
    let (data, nulls) = two_string_nullable(
        args,
        "regex_find(text, pattern)",
        "regex_find",
        Vec::new(),
        |t, p| {
            zyron_types::regex_type::regex_find(t, p)
                .ok()
                .flatten()
                .map(|(s, e)| span_json(s, e))
        },
    )?;
    Ok(Column::with_nulls(
        ColumnData::Binary(data),
        nulls,
        TypeId::Composite,
    ))
}

fn regex_find_compiled_col(args: &[Column]) -> Result<Column> {
    let (data, nulls) = text_compiled_nullable(
        args,
        "regex_find_compiled(text, regex)",
        "regex_find_compiled",
        Vec::new(),
        |t, c| {
            zyron_types::regex_type::regex_find_compiled(t, c)
                .ok()
                .flatten()
                .map(|(s, e)| span_json(s, e))
        },
    )?;
    Ok(Column::with_nulls(
        ColumnData::Binary(data),
        nulls,
        TypeId::Composite,
    ))
}

// array cell is JSON [[start,end],..], empty array when nothing matches
fn regex_find_all_col(args: &[Column]) -> Result<Column> {
    let (data, nulls) = two_string_nullable(
        args,
        "regex_find_all(text, pattern)",
        "regex_find_all",
        Vec::new(),
        |t, p| {
            zyron_types::regex_type::regex_find_all(t, p)
                .ok()
                .map(|spans| spans_json(&spans))
        },
    )?;
    Ok(Column::with_nulls(
        ColumnData::Binary(data),
        nulls,
        TypeId::Array,
    ))
}

// array cell holds full match at index 0 then groups, unmatched groups are
// JSON null, no match at all yields an empty array per the zyron-types fn
fn regex_capture_col(args: &[Column]) -> Result<Column> {
    let (data, nulls) = two_string_nullable(
        args,
        "regex_capture(text, pattern)",
        "regex_capture",
        Vec::new(),
        |t, p| {
            zyron_types::regex_type::regex_capture(t, p)
                .ok()
                .map(|groups| capture_json(&groups))
        },
    )?;
    Ok(Column::with_nulls(
        ColumnData::Binary(data),
        nulls,
        TypeId::Array,
    ))
}

fn regex_split_col(args: &[Column]) -> Result<Column> {
    let (data, nulls) = two_string_nullable(
        args,
        "regex_split(text, pattern)",
        "regex_split",
        Vec::new(),
        |t, p| {
            zyron_types::regex_type::regex_split(t, p)
                .ok()
                .map(|parts| json_string_array(&parts))
        },
    )?;
    Ok(Column::with_nulls(
        ColumnData::Binary(data),
        nulls,
        TypeId::Array,
    ))
}

fn regex_count_col(args: &[Column]) -> Result<Column> {
    let (data, nulls) = two_string_nullable(
        args,
        "regex_count(text, pattern)",
        "regex_count",
        0i32,
        |t, p| {
            zyron_types::regex_type::regex_count(t, p)
                .ok()
                .map(|c| c as i32)
        },
    )?;
    Ok(Column::with_nulls(
        ColumnData::Int32(data),
        nulls,
        TypeId::Int32,
    ))
}

fn regex_replace_col(args: &[Column]) -> Result<Column> {
    let (data, nulls) = three_string_nullable(
        args,
        "regex_replace(text, pattern, replacement)",
        "regex_replace",
        String::new(),
        |t, p, r| zyron_types::regex_type::regex_replace(t, p, r).ok(),
    )?;
    Ok(Column::with_nulls(
        ColumnData::Utf8(data),
        nulls,
        TypeId::Varchar,
    ))
}

fn regex_replace_all_col(args: &[Column]) -> Result<Column> {
    let (data, nulls) = three_string_nullable(
        args,
        "regex_replace_all(text, pattern, replacement)",
        "regex_replace_all",
        String::new(),
        |t, p, r| zyron_types::regex_type::regex_replace_all(t, p, r).ok(),
    )?;
    Ok(Column::with_nulls(
        ColumnData::Utf8(data),
        nulls,
        TypeId::Varchar,
    ))
}

// validates the pattern by compiling it, the Bytea cell stores the pattern
// source since CompiledRegex has no byte form, invalid pattern yields NULL
fn regex_compile_col(args: &[Column]) -> Result<Column> {
    let (data, nulls) = one_string_nullable(
        args,
        "regex_compile(pattern)",
        "regex_compile",
        Vec::new(),
        |p| {
            zyron_types::regex_type::regex_compile(p)
                .ok()
                .map(|_| p.as_bytes().to_vec())
        },
    )?;
    Ok(Column::with_nulls(
        ColumnData::Binary(data),
        nulls,
        TypeId::Bytea,
    ))
}

// ---------------------------------------------------------------------------
// string_ops
// ---------------------------------------------------------------------------

fn extract_list_col(args: &[Column], fn_name: &str, f: fn(&str) -> Vec<String>) -> Result<Column> {
    let sig = format!("{}(text)", fn_name);
    let (data, nulls) = one_string_nullable(args, &sig, fn_name, Vec::new(), |s| {
        Some(json_string_array(&f(s)))
    })?;
    Ok(Column::with_nulls(
        ColumnData::Binary(data),
        nulls,
        TypeId::Array,
    ))
}

fn truncate_chars_col(args: &[Column]) -> Result<Column> {
    expect_args(args, 3, "truncate_chars(text, count, suffix)")?;
    let n = row_count(args, "truncate_chars")?;
    let texts = utf8_rows(&args[0], "truncate_chars")?;
    let counts = int_rows(&args[1], "truncate_chars")?;
    let suffixes = utf8_rows(&args[2], "truncate_chars")?;
    let mut data = Vec::with_capacity(n);
    let mut nulls = NullBitmap::none(n);
    for i in 0..n {
        if any_null(args, i) {
            data.push(String::new());
            nulls.set_null(i);
            continue;
        }
        let count = counts[i].max(0) as usize;
        data.push(zyron_types::string_ops::truncate_chars(
            &texts[i],
            count,
            &suffixes[i],
        ));
    }
    Ok(Column::with_nulls(
        ColumnData::Utf8(data),
        nulls,
        TypeId::Varchar,
    ))
}

fn truncate_words_col(args: &[Column]) -> Result<Column> {
    expect_args(args, 2, "truncate_words(text, count)")?;
    let n = row_count(args, "truncate_words")?;
    let texts = utf8_rows(&args[0], "truncate_words")?;
    let counts = int_rows(&args[1], "truncate_words")?;
    let mut data = Vec::with_capacity(n);
    let mut nulls = NullBitmap::none(n);
    for i in 0..n {
        if any_null(args, i) {
            data.push(String::new());
            nulls.set_null(i);
            continue;
        }
        let count = counts[i].max(0) as usize;
        data.push(zyron_types::string_ops::truncate_words(&texts[i], count));
    }
    Ok(Column::with_nulls(
        ColumnData::Utf8(data),
        nulls,
        TypeId::Varchar,
    ))
}

// ---------------------------------------------------------------------------
// json_schema
// ---------------------------------------------------------------------------

// unparseable json or schema yields NULL for that row rather than failing
// the batch, consistent with the parser-style leniency above
fn json_schema_validate_col(args: &[Column], name: &str) -> Result<Column> {
    let sig = format!("{}(json, schema)", name);
    let (data, nulls) = two_string_nullable(args, &sig, name, false, |j, s| {
        zyron_types::json_schema::json_schema_validate(j, s).ok()
    })?;
    Ok(Column::with_nulls(
        ColumnData::Boolean(data),
        nulls,
        TypeId::Boolean,
    ))
}

fn json_schema_errors_col(args: &[Column]) -> Result<Column> {
    let (data, nulls) = two_string_nullable(
        args,
        "json_schema_errors(json, schema)",
        "json_schema_errors",
        Vec::new(),
        |j, s| {
            zyron_types::json_schema::json_schema_errors(j, s)
                .ok()
                .map(|errs| json_string_array(&errs))
        },
    )?;
    Ok(Column::with_nulls(
        ColumnData::Binary(data),
        nulls,
        TypeId::Array,
    ))
}

// ---------------------------------------------------------------------------
// diff
// ---------------------------------------------------------------------------

fn text_diff_words_col(args: &[Column]) -> Result<Column> {
    let (data, nulls) = two_string_nullable(
        args,
        "text_diff_words(old, new)",
        "text_diff_words",
        Vec::new(),
        |a, b| Some(diff_ops_json(&zyron_types::diff::text_diff_words(a, b))),
    )?;
    Ok(Column::with_nulls(
        ColumnData::Binary(data),
        nulls,
        TypeId::Array,
    ))
}

// ---------------------------------------------------------------------------
// data_quality
// ---------------------------------------------------------------------------

fn is_valid_date_col(args: &[Column]) -> Result<Column> {
    let (data, nulls) = two_string_nullable(
        args,
        "is_valid_date(text, format)",
        "is_valid_date",
        false,
        |t, f| Some(zyron_types::data_quality::is_valid_date(t, f)),
    )?;
    Ok(Column::with_nulls(
        ColumnData::Boolean(data),
        nulls,
        TypeId::Boolean,
    ))
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

    #[test]
    fn regex_match_values_invalid_pattern_and_null_input() {
        let texts = utf8_col(&["hello", "world", "abc"]);
        let pats = utf8_col(&["h.llo", "^x", "("]);
        let col = dispatch("regex_match", &[texts, pats], 3).unwrap().unwrap();
        match &col.data {
            ColumnData::Boolean(v) => {
                assert!(v[0]);
                assert!(!v[1]);
            }
            other => panic!("expected boolean data, got {:?}", other),
        }
        assert!(!col.nulls.is_null(0));
        assert!(col.nulls.is_null(2));

        let mut nulls = NullBitmap::none(2);
        nulls.set_null(1);
        let texts = Column::with_nulls(
            ColumnData::Utf8(vec!["abc".to_string(), String::new()]),
            nulls,
            TypeId::Text,
        );
        let pats = utf8_col(&["a", "a"]);
        let col = dispatch("regex_match", &[texts, pats], 2).unwrap().unwrap();
        assert!(!col.nulls.is_null(0));
        assert!(col.nulls.is_null(1));
    }

    #[test]
    fn regex_find_emits_composite_span_and_null_on_no_match() {
        let texts = utf8_col(&["hello world", "zzz"]);
        let pats = utf8_col(&["world", "abc"]);
        let col = dispatch("regex_find", &[texts, pats], 2).unwrap().unwrap();
        assert_eq!(col.type_id, TypeId::Composite);
        match &col.data {
            ColumnData::Binary(v) => {
                let parsed: serde_json::Value = serde_json::from_slice(&v[0]).unwrap();
                assert_eq!(parsed["start"], 6);
                assert_eq!(parsed["end"], 11);
            }
            other => panic!("expected binary data, got {:?}", other),
        }
        assert!(col.nulls.is_null(1));
    }

    #[test]
    fn regex_capture_groups_as_json_array() {
        let texts = utf8_col(&["hello world"]);
        let pats = utf8_col(&["(\\w+) (\\w+)"]);
        let col = dispatch("regex_capture", &[texts, pats], 1)
            .unwrap()
            .unwrap();
        assert_eq!(col.type_id, TypeId::Array);
        match &col.data {
            ColumnData::Binary(v) => {
                let parsed: serde_json::Value = serde_json::from_slice(&v[0]).unwrap();
                assert_eq!(parsed[0], "hello world");
                assert_eq!(parsed[1], "hello");
                assert_eq!(parsed[2], "world");
            }
            other => panic!("expected binary data, got {:?}", other),
        }
    }

    #[test]
    fn regex_replace_all_rowwise() {
        let texts = utf8_col(&["a1b2c3"]);
        let pats = utf8_col(&["\\d"]);
        let reps = utf8_col(&["X"]);
        let col = dispatch("regex_replace_all", &[texts, pats, reps], 1)
            .unwrap()
            .unwrap();
        match &col.data {
            ColumnData::Utf8(v) => assert_eq!(v[0], "aXbXcX"),
            other => panic!("expected utf8 data, got {:?}", other),
        }
    }

    #[test]
    fn truncate_chars_appends_suffix() {
        let texts = utf8_col(&["Hello, World!", "Hi"]);
        let counts = Column::new(ColumnData::Int64(vec![5, 10]), TypeId::Int64);
        let suffixes = utf8_col(&["...", "..."]);
        let col = dispatch("truncate_chars", &[texts, counts, suffixes], 2)
            .unwrap()
            .unwrap();
        match &col.data {
            ColumnData::Utf8(v) => {
                assert_eq!(v[0], "Hello...");
                assert_eq!(v[1], "Hi");
            }
            other => panic!("expected utf8 data, got {:?}", other),
        }
    }

    #[test]
    fn extract_emails_json_array() {
        let texts = utf8_col(&["reach info@example.com today"]);
        let col = dispatch("extract_emails", &[texts], 1).unwrap().unwrap();
        assert_eq!(col.type_id, TypeId::Array);
        match &col.data {
            ColumnData::Binary(v) => {
                let parsed: serde_json::Value = serde_json::from_slice(&v[0]).unwrap();
                assert_eq!(parsed[0], "info@example.com");
            }
            other => panic!("expected binary data, got {:?}", other),
        }
    }

    #[test]
    fn json_schema_validate_and_alias() {
        let jsons = utf8_col(&["42", "\"x\""]);
        let schemas = utf8_col(&["{\"type\":\"number\"}", "{\"type\":\"number\"}"]);
        let col = dispatch("json_schema_validate", &[jsons.clone(), schemas.clone()], 2)
            .unwrap()
            .unwrap();
        match &col.data {
            ColumnData::Boolean(v) => {
                assert!(v[0]);
                assert!(!v[1]);
            }
            other => panic!("expected boolean data, got {:?}", other),
        }
        let alias = dispatch("validate_json_schema", &[jsons, schemas], 2)
            .unwrap()
            .unwrap();
        match &alias.data {
            ColumnData::Boolean(v) => assert!(v[0]),
            other => panic!("expected boolean data, got {:?}", other),
        }
    }

    #[test]
    fn text_diff_words_emits_op_objects() {
        let old = utf8_col(&["the quick fox"]);
        let new = utf8_col(&["the slow fox"]);
        let col = dispatch("text_diff_words", &[old, new], 1)
            .unwrap()
            .unwrap();
        assert_eq!(col.type_id, TypeId::Array);
        match &col.data {
            ColumnData::Binary(v) => {
                let parsed: serde_json::Value = serde_json::from_slice(&v[0]).unwrap();
                let arr = parsed.as_array().unwrap();
                assert!(!arr.is_empty());
                assert!(arr.iter().all(|op| op.get("op").is_some()));
                assert!(arr.iter().any(|op| op["op"] == "delete"));
                assert!(arr.iter().any(|op| op["op"] == "insert"));
            }
            other => panic!("expected binary data, got {:?}", other),
        }
    }
}
