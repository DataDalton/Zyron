// Table formatting for query results in standard, expanded, and CSV modes.

/// Renders columns and rows as an aligned text table with separator lines.
pub fn render_table(columns: &[String], rows: &[Vec<String>]) -> String {
    if columns.is_empty() {
        return String::new();
    }

    let col_count = columns.len();

    // Compute the max width for each column, starting from header widths
    let mut widths: Vec<usize> = columns.iter().map(|c| c.len()).collect();
    for row in rows {
        for (i, val) in row.iter().enumerate() {
            if i < col_count && val.len() > widths[i] {
                widths[i] = val.len();
            }
        }
    }

    let mut out = String::new();

    // Build separator line: +---------+---------+
    let separator = build_separator(&widths);

    // Header row
    out.push_str(&separator);
    out.push('\n');
    out.push_str(&format_row(columns, &widths));
    out.push('\n');
    out.push_str(&separator);
    out.push('\n');

    // Data rows
    for row in rows {
        out.push_str(&format_row(row, &widths));
        out.push('\n');
    }

    out.push_str(&separator);
    out.push('\n');
    out.push_str(&format!("({} rows)", rows.len()));

    out
}

/// Renders results in expanded (vertical) format, one field per line per record.
pub fn render_expanded(columns: &[String], rows: &[Vec<String>]) -> String {
    if columns.is_empty() || rows.is_empty() {
        return "(0 rows)".to_string();
    }

    let max_col_width = columns.iter().map(|c| c.len()).max().unwrap_or(0);
    let mut out = String::new();

    for (row_idx, row) in rows.iter().enumerate() {
        // Record header
        let record_label = format!("-[ RECORD {} ]", row_idx + 1);
        let dash_fill = if record_label.len() < max_col_width + 20 {
            "-".repeat(max_col_width + 20 - record_label.len())
        } else {
            String::new()
        };
        out.push_str(&record_label);
        out.push_str(&dash_fill);
        out.push('\n');

        for (col_idx, col_name) in columns.iter().enumerate() {
            let value = row.get(col_idx).map(|s| s.as_str()).unwrap_or("");
            // Right-pad column name to align the pipe separator
            out.push_str(&format!(
                "{:width$} | {}",
                col_name,
                value,
                width = max_col_width
            ));
            out.push('\n');
        }
    }

    out
}

/// Renders results as RFC 4180 CSV with proper quoting and escaping.
pub fn render_csv(columns: &[String], rows: &[Vec<String>]) -> String {
    let mut out = String::new();

    // Header line
    let header_fields: Vec<String> = columns.iter().map(|c| csv_escape(c)).collect();
    out.push_str(&header_fields.join(","));
    out.push('\n');

    // Data lines
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
    format!("+{}+", parts.join("+"))
}

// Formats a single row with padded cells and pipe separators.
fn format_row(values: &[String], widths: &[usize]) -> String {
    let cells: Vec<String> = widths
        .iter()
        .enumerate()
        .map(|(i, w)| {
            let val = values.get(i).map(|s| s.as_str()).unwrap_or("");
            format!(" {:width$} ", val, width = w)
        })
        .collect();
    format!("|{}|", cells.join("|"))
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
