//! Dispatch for barcode encoding, diff/patch relations, markdown extractors,
//! natural-sort ranking, xxhash128, and bare window-function names
//!
//! Array typed results are JSON text bytes inside Binary cells per the
//! types_bridge physical mapping

use crate::column::{Column, ColumnData, NullBitmap};
use zyron_common::{Result, TypeId, ZyronError};

pub(super) fn dispatch(name: &str, args: &[Column], _num_rows: usize) -> Option<Result<Column>> {
    Some(match name {
        // bare window-function calls outside an OVER context, the error is
        // the defined execution semantics for these names, not a stub
        "cume_dist" | "delta" | "dense_rank" | "derivative" | "ema" | "first_value" | "lag"
        | "last_value" | "lead" | "moving_avg" | "nth_value" | "ntile" | "percent_rank"
        | "rank" | "rate" | "row_number" => window_requires_over(name),

        "barcode_encode" => barcode_encode_impl(args),
        "change_log" => change_log_impl(args),
        "json_diff_table" => json_diff_table_impl(args),
        "json_equals" => json_equals_impl(args),
        "row_diff_ordinal" => row_diff_ordinal_impl(args),
        "markdown_extract_headers" => markdown_pairs_impl(args, "markdown_extract_headers", |s| {
            zyron_types::document::markdown_extract_headers(s)
                .into_iter()
                .map(|(level, text)| {
                    (
                        serde_json::Value::from(level as i64),
                        serde_json::Value::from(text),
                    )
                })
                .collect()
        }),
        "markdown_extract_links" => markdown_pairs_impl(args, "markdown_extract_links", |s| {
            zyron_types::document::markdown_extract_links(s)
                .into_iter()
                .map(|(text, url)| (serde_json::Value::from(text), serde_json::Value::from(url)))
                .collect()
        }),
        "markdown_extract_code_blocks" => {
            markdown_pairs_impl(args, "markdown_extract_code_blocks", |s| {
                zyron_types::document::markdown_extract_code_blocks(s)
                    .into_iter()
                    .map(|(lang, code)| {
                        (serde_json::Value::from(lang), serde_json::Value::from(code))
                    })
                    .collect()
            })
        }
        "custom_order_rank" => custom_order_rank_impl(args),
        "ip_sort_key" => ip_sort_key_impl(args),
        "xxhash128" => xxhash128_impl(args),
        _ => return None,
    })
}

fn window_requires_over(name: &str) -> Result<Column> {
    Err(ZyronError::ExecutionError(format!(
        "{} is a window function and requires an OVER clause",
        name
    )))
}

// barcode_encode(text [, format text]) -> bytea PNG, format defaults to
// code128, accepted names are code128 code39 ean13 ean8 upca with dashes
// and underscores ignored. Per-row encode failure (unsupported characters,
// wrong digit count) yields NULL, matching the parser-style convention
fn barcode_encode_impl(args: &[Column]) -> Result<Column> {
    if args.is_empty() || args.len() > 2 {
        return Err(ZyronError::ExecutionError(
            "barcode_encode(text [, format text]) takes 1 or 2 arguments".to_string(),
        ));
    }
    let data = super::column_strings(&args[0])?;
    let formats = if args.len() == 2 {
        Some(super::column_strings(&args[1])?)
    } else {
        None
    };
    if let Some(f) = &formats {
        if f.len() != data.len() {
            return Err(ZyronError::ExecutionError(
                "barcode_encode column length mismatch".to_string(),
            ));
        }
    }
    let mut cells = Vec::with_capacity(data.len());
    let mut nulls = NullBitmap::none(data.len());
    for i in 0..data.len() {
        let inputNull = args[0].nulls.is_null(i) || (args.len() == 2 && args[1].nulls.is_null(i));
        if inputNull {
            cells.push(Vec::new());
            nulls.set_null(i);
            continue;
        }
        let format = match &formats {
            Some(f) => parse_barcode_format(f[i])?,
            None => zyron_types::barcode::BarcodeFormat::Code128,
        };
        match zyron_types::barcode::barcode_encode(data[i], format) {
            Ok(bytes) => cells.push(bytes),
            Err(_) => {
                cells.push(Vec::new());
                nulls.set_null(i);
            }
        }
    }
    Ok(Column::with_nulls(
        ColumnData::Binary(cells),
        nulls,
        TypeId::Bytea,
    ))
}

// unrecognized format names are a structural error, not a per-row NULL
fn parse_barcode_format(s: &str) -> Result<zyron_types::barcode::BarcodeFormat> {
    use zyron_types::barcode::BarcodeFormat;
    let norm: String = s
        .chars()
        .filter(|c| *c != '-' && *c != '_')
        .collect::<String>()
        .to_ascii_lowercase();
    match norm.as_str() {
        "code128" => Ok(BarcodeFormat::Code128),
        "code39" => Ok(BarcodeFormat::Code39),
        "ean13" => Ok(BarcodeFormat::Ean13),
        "ean8" => Ok(BarcodeFormat::Ean8),
        "upca" | "upc" => Ok(BarcodeFormat::UpcA),
        _ => Err(ZyronError::ExecutionError(format!(
            "barcode_encode unknown format '{}', expected code128, code39, ean13, ean8, or upca",
            s
        ))),
    }
}

// change_log(table, from_version, to_version, entries_json) -> array
// zyron_types change_log is a pure formatter over caller-supplied history
// rows, so the rows arrive as a JSON array of [version, operation, column,
// old, new] tuples in the 4th argument. Unparseable entries JSON yields
// NULL for that row
fn change_log_impl(args: &[Column]) -> Result<Column> {
    if args.len() != 4 {
        return Err(ZyronError::ExecutionError(
            "change_log(table, from_version, to_version, entries_json) takes exactly 4 arguments"
                .to_string(),
        ));
    }
    let tables = super::column_strings(&args[0])?;
    let fromVersions = super::column_ints(&args[1])?;
    let toVersions = super::column_ints(&args[2])?;
    let entries = super::column_strings(&args[3])?;
    let n = tables.len();
    if fromVersions.len() != n || toVersions.len() != n || entries.len() != n {
        return Err(ZyronError::ExecutionError(
            "change_log column length mismatch".to_string(),
        ));
    }
    let mut cells = Vec::with_capacity(n);
    let mut nulls = NullBitmap::none(n);
    for i in 0..n {
        if (0..4).any(|a| args[a].nulls.is_null(i)) {
            cells.push(Vec::new());
            nulls.set_null(i);
            continue;
        }
        match change_log_row(tables[i], fromVersions[i], toVersions[i], entries[i]) {
            Some(json) => cells.push(json.into_bytes()),
            None => {
                cells.push(Vec::new());
                nulls.set_null(i);
            }
        }
    }
    Ok(Column::with_nulls(
        ColumnData::Binary(cells),
        nulls,
        TypeId::Array,
    ))
}

// entry tuples must be [number, string, string, string|null, string|null]
// ChangeLogEntry does not derive Serialize so the output JSON is built by hand
fn change_log_row(table: &str, from: i64, to: i64, entriesJson: &str) -> Option<String> {
    struct RawEntry {
        version: i64,
        op: String,
        column: String,
        old: Option<String>,
        new: Option<String>,
    }
    let parsed: serde_json::Value = serde_json::from_str(entriesJson).ok()?;
    let arr = parsed.as_array()?;
    let mut raws = Vec::with_capacity(arr.len());
    for e in arr {
        let t = e.as_array()?;
        if t.len() != 5 {
            return None;
        }
        let old = match &t[3] {
            serde_json::Value::Null => None,
            v => Some(v.as_str()?.to_string()),
        };
        let new = match &t[4] {
            serde_json::Value::Null => None,
            v => Some(v.as_str()?.to_string()),
        };
        raws.push(RawEntry {
            version: t[0].as_i64()?,
            op: t[1].as_str()?.to_string(),
            column: t[2].as_str()?.to_string(),
            old,
            new,
        });
    }
    let tuples: Vec<(i64, &str, &str, Option<&str>, Option<&str>)> = raws
        .iter()
        .map(|r| {
            (
                r.version,
                r.op.as_str(),
                r.column.as_str(),
                r.old.as_deref(),
                r.new.as_deref(),
            )
        })
        .collect();
    let log = zyron_types::diff_patch::change_log(table, from, to, &tuples);
    let rows: Vec<serde_json::Value> = log
        .iter()
        .map(|e| {
            serde_json::json!({
                "table": e.table,
                "version": e.version,
                "operation": e.operation,
                "column": e.column,
                "old_value": e.old_value,
                "new_value": e.new_value,
            })
        })
        .collect();
    Some(serde_json::Value::Array(rows).to_string())
}

// json_diff_table(json, json) -> array of {path, op, old_value, new_value}
// objects. Invalid JSON input yields NULL for that row
fn json_diff_table_impl(args: &[Column]) -> Result<Column> {
    if args.len() != 2 {
        return Err(ZyronError::ExecutionError(
            "json_diff_table(json, json) takes exactly 2 arguments".to_string(),
        ));
    }
    let a = super::column_strings(&args[0])?;
    let b = super::column_strings(&args[1])?;
    if a.len() != b.len() {
        return Err(ZyronError::ExecutionError(
            "json_diff_table column length mismatch".to_string(),
        ));
    }
    let mut cells = Vec::with_capacity(a.len());
    let mut nulls = NullBitmap::none(a.len());
    for i in 0..a.len() {
        if args[0].nulls.is_null(i) || args[1].nulls.is_null(i) {
            cells.push(Vec::new());
            nulls.set_null(i);
            continue;
        }
        match zyron_types::diff_patch::json_diff_table(a[i], b[i]) {
            Ok(rows) => {
                let vals: Vec<serde_json::Value> = rows
                    .iter()
                    .map(|r| {
                        serde_json::json!({
                            "path": r.path,
                            "op": r.op,
                            "old_value": r.old_value,
                            "new_value": r.new_value,
                        })
                    })
                    .collect();
                cells.push(serde_json::Value::Array(vals).to_string().into_bytes());
            }
            Err(_) => {
                cells.push(Vec::new());
                nulls.set_null(i);
            }
        }
    }
    Ok(Column::with_nulls(
        ColumnData::Binary(cells),
        nulls,
        TypeId::Array,
    ))
}

// row_diff_ordinal(columns_json, old_json, new_json) -> JSON array of
// {column, old_value, new_value} for changed positions, the three inputs
// are parallel JSON arrays with null marking an absent value. Invalid or
// mismatched input yields NULL for that row
fn row_diff_ordinal_impl(args: &[Column]) -> Result<Column> {
    if args.len() != 3 {
        return Err(ZyronError::ExecutionError(
            "row_diff_ordinal(columns, old_values, new_values) takes exactly 3 arguments"
                .to_string(),
        ));
    }
    let cols = super::column_strings(&args[0])?;
    let olds = super::column_strings(&args[1])?;
    let news = super::column_strings(&args[2])?;
    let n = cols.len();
    let parseStrings = |s: &str| -> Option<Vec<Option<String>>> {
        let v: serde_json::Value = serde_json::from_str(s).ok()?;
        let arr = v.as_array()?;
        Some(
            arr.iter()
                .map(|x| match x {
                    serde_json::Value::Null => None,
                    serde_json::Value::String(s) => Some(s.clone()),
                    other => Some(other.to_string()),
                })
                .collect(),
        )
    };
    let mut cells = Vec::with_capacity(n);
    let mut nulls = NullBitmap::none(n);
    for i in 0..n {
        if args[0].nulls.is_null(i) || args[1].nulls.is_null(i) || args[2].nulls.is_null(i) {
            cells.push(Vec::new());
            nulls.set_null(i);
            continue;
        }
        let parsed = (|| {
            let names: Vec<String> = serde_json::from_str::<serde_json::Value>(cols[i])
                .ok()?
                .as_array()?
                .iter()
                .map(|x| x.as_str().map(|s| s.to_string()).unwrap_or_default())
                .collect();
            let old = parseStrings(olds[i])?;
            let new = parseStrings(news[i])?;
            let nameRefs: Vec<&str> = names.iter().map(|s| s.as_str()).collect();
            let oldRefs: Vec<Option<&str>> = old.iter().map(|o| o.as_deref()).collect();
            let newRefs: Vec<Option<&str>> = new.iter().map(|o| o.as_deref()).collect();
            let changes =
                zyron_types::diff_patch::row_diff_ordinal(&nameRefs, &oldRefs, &newRefs).ok()?;
            let vals: Vec<serde_json::Value> = changes
                .iter()
                .map(|c| {
                    serde_json::json!({
                        "column": c.column,
                        "old_value": c.old_value,
                        "new_value": c.new_value,
                    })
                })
                .collect();
            Some(serde_json::Value::Array(vals).to_string().into_bytes())
        })();
        match parsed {
            Some(bytes) => cells.push(bytes),
            None => {
                cells.push(Vec::new());
                nulls.set_null(i);
            }
        }
    }
    Ok(Column::with_nulls(
        ColumnData::Binary(cells),
        nulls,
        TypeId::Array,
    ))
}

// json_equals(json, json) -> structural equality after parsing, so key
// order and whitespace differences compare equal. Invalid JSON yields NULL
fn json_equals_impl(args: &[Column]) -> Result<Column> {
    if args.len() != 2 {
        return Err(ZyronError::ExecutionError(
            "json_equals(json, json) takes exactly 2 arguments".to_string(),
        ));
    }
    let a = super::column_strings(&args[0])?;
    let b = super::column_strings(&args[1])?;
    if a.len() != b.len() {
        return Err(ZyronError::ExecutionError(
            "json_equals column length mismatch".to_string(),
        ));
    }
    let mut cells = Vec::with_capacity(a.len());
    let mut nulls = NullBitmap::none(a.len());
    for i in 0..a.len() {
        if args[0].nulls.is_null(i) || args[1].nulls.is_null(i) {
            cells.push(false);
            nulls.set_null(i);
            continue;
        }
        let pa: std::result::Result<serde_json::Value, _> = serde_json::from_str(a[i]);
        let pb: std::result::Result<serde_json::Value, _> = serde_json::from_str(b[i]);
        match (pa, pb) {
            (Ok(va), Ok(vb)) => cells.push(va == vb),
            _ => {
                cells.push(false);
                nulls.set_null(i);
            }
        }
    }
    Ok(Column::with_nulls(
        ColumnData::Boolean(cells),
        nulls,
        TypeId::Boolean,
    ))
}

// shared shape for the markdown extractors, each row becomes a JSON array
// of two-element arrays
fn markdown_pairs_impl<F>(args: &[Column], fnLabel: &str, f: F) -> Result<Column>
where
    F: Fn(&str) -> Vec<(serde_json::Value, serde_json::Value)>,
{
    if args.len() != 1 {
        return Err(ZyronError::ExecutionError(format!(
            "{}(text) takes exactly 1 argument",
            fnLabel
        )));
    }
    let strings = super::column_strings(&args[0])?;
    let cells: Vec<Vec<u8>> = strings
        .iter()
        .map(|s| {
            let pairs: Vec<serde_json::Value> = f(s)
                .into_iter()
                .map(|(x, y)| serde_json::Value::Array(vec![x, y]))
                .collect();
            serde_json::Value::Array(pairs).to_string().into_bytes()
        })
        .collect();
    Ok(Column::with_nulls(
        ColumnData::Binary(cells),
        args[0].nulls.clone(),
        TypeId::Array,
    ))
}

// custom_order_rank(value, order [, unknown]) -> int32
// order accepts a JSON array of strings or a {a,b,c} text-array literal
// unknown is 'first' or 'last' (default last), unlisted values rank at
// i32 MIN or MAX so they sort at the requested end
fn custom_order_rank_impl(args: &[Column]) -> Result<Column> {
    if args.len() < 2 || args.len() > 3 {
        return Err(ZyronError::ExecutionError(
            "custom_order_rank(value, order [, unknown]) takes 2 or 3 arguments".to_string(),
        ));
    }
    let values = super::column_strings(&args[0])?;
    let orders = super::column_strings(&args[1])?;
    let unknowns = if args.len() == 3 {
        Some(super::column_strings(&args[2])?)
    } else {
        None
    };
    let n = values.len();
    if orders.len() != n || unknowns.as_ref().is_some_and(|u| u.len() != n) {
        return Err(ZyronError::ExecutionError(
            "custom_order_rank column length mismatch".to_string(),
        ));
    }
    let mut data = Vec::with_capacity(n);
    let mut nulls = NullBitmap::none(n);
    for i in 0..n {
        let inputNull = args[0].nulls.is_null(i)
            || args[1].nulls.is_null(i)
            || (args.len() == 3 && args[2].nulls.is_null(i));
        if inputNull {
            data.push(0);
            nulls.set_null(i);
            continue;
        }
        let order = parse_order_list(orders[i]);
        let orderRefs: Vec<&str> = order.iter().map(|s| s.as_str()).collect();
        let unknown = match unknowns.as_ref().map(|u| u[i]) {
            None => zyron_types::natural_sort::UnknownPosition::Last,
            Some(s) => match s.to_ascii_lowercase().as_str() {
                "first" => zyron_types::natural_sort::UnknownPosition::First,
                "last" => zyron_types::natural_sort::UnknownPosition::Last,
                other => {
                    return Err(ZyronError::ExecutionError(format!(
                        "custom_order_rank unknown position '{}', expected first or last",
                        other
                    )));
                }
            },
        };
        data.push(zyron_types::natural_sort::custom_order_rank(
            values[i], &orderRefs, unknown,
        ));
    }
    Ok(Column::with_nulls(
        ColumnData::Int32(data),
        nulls,
        TypeId::Int32,
    ))
}

// order list from JSON array text, falling back to the {a,b,c} literal
// parser shared with the JSON key-array operators
fn parse_order_list(s: &str) -> Vec<String> {
    let trimmed = s.trim();
    if trimmed.starts_with('[') {
        if let Ok(serde_json::Value::Array(arr)) =
            serde_json::from_str::<serde_json::Value>(trimmed)
        {
            return arr
                .into_iter()
                .map(|v| match v {
                    serde_json::Value::String(x) => x,
                    other => other.to_string(),
                })
                .collect();
        }
    }
    super::parse_text_array(s)
}

// ip_sort_key(text) -> 16-byte bytea, IPv4 maps into the v4-in-v6 form so
// mixed families sort together. Unparseable addresses yield NULL
fn ip_sort_key_impl(args: &[Column]) -> Result<Column> {
    if args.len() != 1 {
        return Err(ZyronError::ExecutionError(
            "ip_sort_key(text) takes exactly 1 argument".to_string(),
        ));
    }
    let addrs = super::column_strings(&args[0])?;
    let mut cells = Vec::with_capacity(addrs.len());
    let mut nulls = NullBitmap::none(addrs.len());
    for (i, a) in addrs.iter().enumerate() {
        if args[0].nulls.is_null(i) {
            cells.push(Vec::new());
            nulls.set_null(i);
            continue;
        }
        match zyron_types::natural_sort::ip_sort_key(a) {
            Some(key) => cells.push(key.to_vec()),
            None => {
                cells.push(Vec::new());
                nulls.set_null(i);
            }
        }
    }
    Ok(Column::with_nulls(
        ColumnData::Binary(cells),
        nulls,
        TypeId::Bytea,
    ))
}

// xxhash128(bytea) -> int128, the u128 digest is bit-reinterpreted as i128
fn xxhash128_impl(args: &[Column]) -> Result<Column> {
    if args.len() != 1 {
        return Err(ZyronError::ExecutionError(
            "xxhash128(bytea) takes exactly 1 argument".to_string(),
        ));
    }
    let values = super::column_bytes(&args[0])?;
    let data: Vec<i128> = values
        .iter()
        .map(|b| zyron_types::checksum::xxhash128(b) as i128)
        .collect();
    Ok(Column::with_nulls(
        ColumnData::Int128(data),
        args[0].nulls.clone(),
        TypeId::Int128,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn utf8_col(values: Vec<&str>) -> Column {
        Column::new(
            ColumnData::Utf8(values.into_iter().map(str::to_string).collect()),
            TypeId::Text,
        )
    }

    #[test]
    fn bare_window_function_call_errors() {
        let col = Column::new(ColumnData::Int64(vec![1]), TypeId::Int64);
        let result = dispatch("row_number", &[col], 1).expect("dispatch arm");
        let msg = match result {
            Err(ZyronError::ExecutionError(m)) => m,
            _ => panic!("expected ExecutionError"),
        };
        assert_eq!(
            msg,
            "row_number is a window function and requires an OVER clause"
        );
        let col = utf8_col(vec!["x"]);
        assert!(dispatch("lag", &[col], 1).expect("dispatch arm").is_err());
    }

    #[test]
    fn xxhash128_matches_direct_call() {
        let col = utf8_col(vec!["hello"]);
        let out = dispatch("xxhash128", &[col], 1).unwrap().unwrap();
        let expected = zyron_types::checksum::xxhash128(b"hello") as i128;
        match &out.data {
            ColumnData::Int128(v) => assert_eq!(v[0], expected),
            _ => panic!("expected Int128 column"),
        }
    }

    #[test]
    fn ip_sort_key_maps_v4_and_nulls_invalid() {
        let col = utf8_col(vec!["1.2.3.4", "not an ip"]);
        let out = dispatch("ip_sort_key", &[col], 2).unwrap().unwrap();
        match &out.data {
            ColumnData::Binary(v) => {
                assert_eq!(v[0].len(), 16);
                assert_eq!(&v[0][10..12], &[0xff, 0xff]);
                assert_eq!(&v[0][12..16], &[1, 2, 3, 4]);
            }
            _ => panic!("expected Binary column"),
        }
        assert!(!out.nulls.is_null(0));
        assert!(out.nulls.is_null(1));
    }

    #[test]
    fn markdown_extract_headers_emits_level_text_pairs() {
        let col = utf8_col(vec!["# Title\n\n## Sub"]);
        let out = dispatch("markdown_extract_headers", &[col], 1)
            .unwrap()
            .unwrap();
        let cell = match &out.data {
            ColumnData::Binary(v) => &v[0],
            _ => panic!("expected Binary column"),
        };
        let parsed: serde_json::Value =
            serde_json::from_str(std::str::from_utf8(cell).unwrap()).unwrap();
        assert_eq!(parsed, serde_json::json!([[1, "Title"], [2, "Sub"]]));
    }

    #[test]
    fn custom_order_rank_ranks_by_list_position() {
        let values = utf8_col(vec!["silver", "unlisted"]);
        let orders = utf8_col(vec![
            r#"["gold","silver","bronze"]"#,
            r#"["gold","silver","bronze"]"#,
        ]);
        let out = dispatch("custom_order_rank", &[values, orders], 2)
            .unwrap()
            .unwrap();
        match &out.data {
            ColumnData::Int32(v) => {
                assert_eq!(v[0], 1);
                assert_eq!(v[1], i32::MAX);
            }
            _ => panic!("expected Int32 column"),
        }
    }

    #[test]
    fn json_diff_table_null_input_propagates() {
        let mut nulls = NullBitmap::none(2);
        nulls.set_null(1);
        let a = Column::with_nulls(
            ColumnData::Utf8(vec![r#"{"a":1}"#.to_string(), String::new()]),
            nulls,
            TypeId::Json,
        );
        let b = utf8_col(vec![r#"{"a":2}"#, "{}"]);
        let out = dispatch("json_diff_table", &[a, b], 2).unwrap().unwrap();
        assert!(!out.nulls.is_null(0));
        assert!(out.nulls.is_null(1));
        let cell = match &out.data {
            ColumnData::Binary(v) => &v[0],
            _ => panic!("expected Binary column"),
        };
        let parsed: serde_json::Value =
            serde_json::from_str(std::str::from_utf8(cell).unwrap()).unwrap();
        let arr = parsed.as_array().unwrap();
        assert_eq!(arr.len(), 1);
        assert_eq!(arr[0]["op"], "replace");
        assert_eq!(arr[0]["path"], "/a");
    }

    #[test]
    fn change_log_filters_version_window() {
        let table = utf8_col(vec!["users"]);
        let from = Column::new(ColumnData::Int64(vec![1]), TypeId::Int64);
        let to = Column::new(ColumnData::Int64(vec![3]), TypeId::Int64);
        let entries = utf8_col(vec![
            r#"[[1,"insert","name",null,"a"],[2,"update","name","a","b"],[3,"update","name","b","c"]]"#,
        ]);
        let out = dispatch("change_log", &[table, from, to, entries], 1)
            .unwrap()
            .unwrap();
        let cell = match &out.data {
            ColumnData::Binary(v) => &v[0],
            _ => panic!("expected Binary column"),
        };
        let parsed: serde_json::Value =
            serde_json::from_str(std::str::from_utf8(cell).unwrap()).unwrap();
        let arr = parsed.as_array().unwrap();
        assert_eq!(arr.len(), 2);
        assert_eq!(arr[0]["version"], 2);
        assert_eq!(arr[0]["table"], "users");
        assert_eq!(arr[1]["new_value"], "c");
    }

    #[test]
    fn barcode_encode_defaults_code128_and_nulls_bad_input() {
        let data = utf8_col(vec!["HELLO123"]);
        let out = dispatch("barcode_encode", &[data], 1).unwrap().unwrap();
        match &out.data {
            ColumnData::Binary(v) => {
                assert_eq!(
                    &v[0][..8],
                    &[0x89, b'P', b'N', b'G', 0x0D, 0x0A, 0x1A, 0x0A]
                );
            }
            _ => panic!("expected Binary column"),
        }
        let data = utf8_col(vec!["notdigits"]);
        let fmt = utf8_col(vec!["ean13"]);
        let out = dispatch("barcode_encode", &[data, fmt], 1)
            .unwrap()
            .unwrap();
        assert!(out.nulls.is_null(0));
    }
}
