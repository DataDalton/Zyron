// Table formatting for query results in standard, expanded, and CSV modes.
// ANSI colors are injected for the headers, separators, NULL markers, and
// numeric columns; CSV output stays colorless so the file content is
// parseable verbatim.

use crate::color;

/// Per-column type hint used to decide coloring. Derived from the PG-style
/// type OID in the RowDescription message. Values outside the known OID
/// ranges map to `Other`, which displays uncolored.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ColumnKind {
    Numeric,
    Other,
}

/// Returns the column kind for a PostgreSQL-style type OID. Our server
/// advertises the same OIDs the wire protocol reserves for numeric types
/// (bool is grouped with numeric for column alignment since it renders as
/// `t`/`f` rather than text). Anything else is treated as text/other.
pub fn classify_type_oid(type_oid: i32) -> ColumnKind {
    match type_oid {
        // int2, int4, int8, float4, float8, numeric, oid, money
        21 | 23 | 20 | 700 | 701 | 1700 | 26 | 790 => ColumnKind::Numeric,
        // bool
        16 => ColumnKind::Numeric,
        _ => ColumnKind::Other,
    }
}

/// Renders columns and rows as an aligned text table with separator lines.
/// `kinds` must be the same length as `columns` (one entry per column); it
/// drives per-column value coloring. Callers that don't have type metadata
/// can pass a vector of all `ColumnKind::Other`.
pub fn render_table(columns: &[String], rows: &[Vec<String>], kinds: &[ColumnKind]) -> String {
    if columns.is_empty() {
        return String::new();
    }

    let col_count = columns.len();

    // Compute the max width for each column, starting from header widths.
    // Width is measured on the unstyled text; ANSI escapes are added later
    // and don't contribute to display columns.
    let mut widths: Vec<usize> = columns.iter().map(|c| c.len()).collect();
    for row in rows {
        for (i, val) in row.iter().enumerate() {
            if i < col_count && val.len() > widths[i] {
                widths[i] = val.len();
            }
        }
    }

    let mut out = String::new();

    let separator = build_separator(&widths);

    // Header row (bold column names). The separator lines stay uncolored
    // so they blend into the terminal frame.
    out.push_str(&separator);
    out.push('\n');
    out.push_str(&format_header(columns, &widths));
    out.push('\n');
    out.push_str(&separator);
    out.push('\n');

    for row in rows {
        out.push_str(&format_row(row, &widths, kinds));
        out.push('\n');
    }

    out.push_str(&separator);
    out.push('\n');
    out.push_str(&color::dim(&format!("({} rows)", rows.len())));

    out
}

/// Renders results in expanded (vertical) format, one field per line per record.
pub fn render_expanded(columns: &[String], rows: &[Vec<String>], kinds: &[ColumnKind]) -> String {
    if columns.is_empty() || rows.is_empty() {
        return color::dim("(0 rows)").to_string();
    }

    let max_col_width = columns.iter().map(|c| c.len()).max().unwrap_or(0);
    let mut out = String::new();

    for (row_idx, row) in rows.iter().enumerate() {
        // Record header in dim.
        let record_label = format!("-[ RECORD {} ]", row_idx + 1);
        let dash_fill = if record_label.len() < max_col_width + 20 {
            "-".repeat(max_col_width + 20 - record_label.len())
        } else {
            String::new()
        };
        out.push_str(&color::dim(&format!("{}{}", record_label, dash_fill)));
        out.push('\n');

        for (col_idx, col_name) in columns.iter().enumerate() {
            let value = row.get(col_idx).map(|s| s.as_str()).unwrap_or("");
            let padded_name = format!("{:width$}", col_name, width = max_col_width);
            let kind = kinds.get(col_idx).copied().unwrap_or(ColumnKind::Other);
            out.push_str(&color::bold(&padded_name));
            out.push_str(&color::dim(" | "));
            out.push_str(&color_value(value, kind));
            out.push('\n');
        }
    }

    out
}

/// Renders results as a JSON array of row objects. Numeric-kind columns
/// are emitted as bare JSON numbers when the raw text parses as a valid
/// number, otherwise they fall back to strings so the output is always
/// syntactically valid. NULL maps to JSON `null`.
///
/// Machine-readable format: no ANSI, no padding, one object per row,
/// compact separators. Suitable for piping into `jq` or consuming from
/// scripts.
pub fn render_json(columns: &[String], rows: &[Vec<String>], kinds: &[ColumnKind]) -> String {
    let mut out = String::from("[");
    let mut first_row = true;
    for row in rows {
        if !first_row {
            out.push(',');
        }
        first_row = false;
        out.push('{');
        let mut first_col = true;
        for (i, col) in columns.iter().enumerate() {
            if !first_col {
                out.push(',');
            }
            first_col = false;
            out.push_str(&json_escape(col));
            out.push(':');
            let value = row.get(i).map(|s| s.as_str()).unwrap_or("");
            let kind = kinds.get(i).copied().unwrap_or(ColumnKind::Other);
            append_json_value(&mut out, value, kind);
        }
        out.push('}');
    }
    out.push(']');
    out
}

/// Encodes a single cell value as a JSON fragment and pushes it onto
/// `out`. Bare `NULL` text maps to JSON null. Numeric-kind cells are
/// emitted as numbers when the text round-trips through `f64::parse`
/// (covers int, float, negative, exponent forms); anything else becomes
/// a JSON string.
fn append_json_value(out: &mut String, value: &str, kind: ColumnKind) {
    if value == "NULL" {
        out.push_str("null");
        return;
    }
    if kind == ColumnKind::Numeric {
        // Only emit a bare JSON number when the raw wire text is itself
        // a valid JSON number. Integer + float path covered by f64
        // parsing; booleans render as "t"/"f" on our wire so fall
        // through to string.
        if value.parse::<f64>().is_ok() {
            out.push_str(value);
            return;
        }
    }
    out.push_str(&json_escape(value));
}

/// RFC 8259 string encoding: wrap in double quotes, escape control
/// characters and backslash/quote with the standard short forms.
fn json_escape(s: &str) -> String {
    let mut out = String::with_capacity(s.len() + 2);
    out.push('"');
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            '\x08' => out.push_str("\\b"),
            '\x0c' => out.push_str("\\f"),
            c if (c as u32) < 0x20 => {
                out.push_str(&format!("\\u{:04x}", c as u32));
            }
            c => out.push(c),
        }
    }
    out.push('"');
    out
}

/// Renders results as RFC 4180 CSV with proper quoting and escaping. No
/// color codes are emitted so the file content is machine-readable.
pub fn render_csv(columns: &[String], rows: &[Vec<String>]) -> String {
    let mut out = String::new();

    let header_fields: Vec<String> = columns.iter().map(|c| csv_escape(c)).collect();
    out.push_str(&header_fields.join(","));
    out.push('\n');

    for row in rows {
        let fields: Vec<String> = row.iter().map(|v| csv_escape(v)).collect();
        out.push_str(&fields.join(","));
        out.push('\n');
    }

    out
}

// Builds a +-...-+-...-+ separator line from column widths.
fn build_separator(widths: &[usize]) -> String {
    let parts: Vec<String> = widths.iter().map(|w| "-".repeat(w + 2)).collect();
    color::dim(&format!("+{}+", parts.join("+")))
}

/// Formats the header row with bold column names and dim pipe separators.
fn format_header(values: &[String], widths: &[usize]) -> String {
    let cells: Vec<String> = widths
        .iter()
        .enumerate()
        .map(|(i, w)| {
            let val = values.get(i).map(|s| s.as_str()).unwrap_or("");
            format!(" {} ", color::bold(&format!("{:width$}", val, width = w)))
        })
        .collect();
    let pipe = color::dim("|");
    format!("{p}{inner}{p}", p = pipe, inner = cells.join(&pipe))
}

/// Formats a data row. NULL is dim italic gray; numeric columns are cyan.
fn format_row(values: &[String], widths: &[usize], kinds: &[ColumnKind]) -> String {
    let cells: Vec<String> = widths
        .iter()
        .enumerate()
        .map(|(i, w)| {
            let raw = values.get(i).map(|s| s.as_str()).unwrap_or("");
            let padded = format!("{:width$}", raw, width = w);
            let kind = kinds.get(i).copied().unwrap_or(ColumnKind::Other);
            format!(" {} ", color_value(&padded, kind))
        })
        .collect();
    let pipe = color::dim("|");
    format!("{p}{inner}{p}", p = pipe, inner = cells.join(&pipe))
}

/// Applies the value-specific color for a cell. NULL always wins (dim
/// italic gray, distinguishable from a legitimate string "NULL"). Numeric
/// columns render in cyan so they stand out from text and row indices.
fn color_value(value: &str, kind: ColumnKind) -> String {
    if value.trim() == "NULL" {
        return color::dim(&color::italic(value));
    }
    match kind {
        ColumnKind::Numeric => color::cyan(value),
        ColumnKind::Other => value.to_string(),
    }
}

// Escapes a field for CSV output. Fields containing commas, quotes, or newlines
// are wrapped in double quotes with internal quotes doubled.
fn csv_escape(field: &str) -> String {
    if field.contains(',') || field.contains('"') || field.contains('\n') || field.contains('\r') {
        let escaped = field.replace('"', "\"\"");
        format!("\"{}\"", escaped)
    } else {
        field.to_string()
    }
}
