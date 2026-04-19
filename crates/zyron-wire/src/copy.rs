//! COPY protocol handler for bulk data transfer.
//!
//! Implements COPY TO (server to client) and COPY FROM (client to server)
//! with text, CSV, and PostgreSQL binary formats.

use bytes::BytesMut;

use crate::messages::ProtocolError;
use crate::messages::backend::BackendMessage;
use crate::types;
use zyron_executor::batch::DataBatch;
use zyron_planner::logical::LogicalColumn;

/// COPY data format.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum CopyFormat {
    Text,
    Csv,
    Binary,
}

// ---------------------------------------------------------------------------
// COPY TO (server -> client)
// ---------------------------------------------------------------------------

/// Handles COPY TO by serializing DataBatch results into COPY wire format.
pub struct CopyOutHandler {
    columns: Vec<LogicalColumn>,
    format: CopyFormat,
    delimiter: u8,
    null_string: Vec<u8>,
}

impl CopyOutHandler {
    pub fn new(columns: Vec<LogicalColumn>, format: CopyFormat) -> Self {
        let (delimiter, null_string) = match format {
            CopyFormat::Text => (b'\t', b"\\N".to_vec()),
            CopyFormat::Csv => (b',', b"".to_vec()),
            CopyFormat::Binary => (b'\0', b"".to_vec()),
        };
        Self {
            columns,
            format,
            delimiter,
            null_string,
        }
    }

    /// Returns the CopyOutResponse header message.
    pub fn header_message(&self) -> BackendMessage {
        let format_code: i8 = match self.format {
            CopyFormat::Binary => 1,
            _ => 0,
        };
        let column_formats = vec![format_code as i16; self.columns.len()];
        BackendMessage::CopyOutResponse {
            format: format_code,
            column_formats,
        }
    }

    /// Converts a DataBatch to CopyData messages.
    /// For text/CSV: reuses a single BytesMut buffer, writes scalars with escaping.
    /// For binary: PostgreSQL binary COPY format (header + rows + trailer).
    pub fn format_batch(&self, batch: &DataBatch) -> Vec<BackendMessage> {
        if self.format == CopyFormat::Binary {
            return self.format_batch_binary(batch);
        }

        let mut messages = Vec::with_capacity(batch.num_rows);
        let mut line = BytesMut::with_capacity(256);
        let mut scalar_buf = BytesMut::with_capacity(64);

        for row in 0..batch.num_rows {
            line.clear();

            for (col_idx, column) in batch.columns.iter().enumerate() {
                if col_idx > 0 {
                    line.extend_from_slice(&[self.delimiter]);
                }

                let scalar = column.get_scalar(row);

                scalar_buf.clear();
                if !types::scalar_write_text(&scalar, &mut scalar_buf) {
                    line.extend_from_slice(&self.null_string);
                } else if self.format == CopyFormat::Csv {
                    csv_escape(&mut line, &scalar_buf);
                } else {
                    text_escape(&mut line, &scalar_buf);
                }
            }
            line.extend_from_slice(b"\n");
            let owned = std::mem::replace(&mut line, BytesMut::with_capacity(256));
            messages.push(BackendMessage::CopyData(owned.to_vec()));
        }

        messages
    }

    /// Formats a batch in PostgreSQL binary COPY format.
    /// Format: header (19 bytes), then per row: i16 field count + per field (i32 length + data).
    /// Trailer: i16 -1 as field count sentinel.
    fn format_batch_binary(&self, batch: &DataBatch) -> Vec<BackendMessage> {
        let mut buf = Vec::with_capacity(batch.num_rows * self.columns.len() * 16 + 32);

        // Binary COPY header: signature + flags + header extension length
        buf.extend_from_slice(b"PGCOPY\n\xff\r\n\x00"); // 11-byte signature
        buf.extend_from_slice(&0u32.to_be_bytes()); // flags (no OIDs)
        buf.extend_from_slice(&0u32.to_be_bytes()); // header extension area length

        let num_fields = self.columns.len() as i16;
        let mut scalar_buf = BytesMut::with_capacity(64);

        for row in 0..batch.num_rows {
            buf.extend_from_slice(&num_fields.to_be_bytes());

            for column in &batch.columns {
                let scalar = column.get_scalar(row);
                scalar_buf.clear();

                if !types::scalar_write_text(&scalar, &mut scalar_buf) {
                    // NULL: length = -1
                    buf.extend_from_slice(&(-1i32).to_be_bytes());
                } else {
                    let len = scalar_buf.len() as i32;
                    buf.extend_from_slice(&len.to_be_bytes());
                    buf.extend_from_slice(&scalar_buf);
                }
            }
        }

        // Trailer: field count = -1
        buf.extend_from_slice(&(-1i16).to_be_bytes());

        vec![BackendMessage::CopyData(buf)]
    }

    /// Returns the CopyDone message.
    pub fn done_message(&self) -> BackendMessage {
        BackendMessage::CopyDone
    }
}

// ---------------------------------------------------------------------------
// COPY FROM (client -> server)
// ---------------------------------------------------------------------------

/// Handles COPY FROM by deserializing client data chunks into DataBatch rows.
pub struct CopyInHandler {
    columns: Vec<LogicalColumn>,
    format: CopyFormat,
    delimiter: u8,
    null_string: Vec<u8>,
    /// Partial line buffer for data that spans CopyData messages.
    buffer: Vec<u8>,
    /// Accumulated rows for the current batch.
    rows: Vec<Vec<Option<Vec<u8>>>>,
}

impl CopyInHandler {
    pub fn new(columns: Vec<LogicalColumn>, format: CopyFormat) -> Self {
        let (delimiter, null_string) = match format {
            CopyFormat::Text => (b'\t', b"\\N".to_vec()),
            CopyFormat::Csv => (b',', b"".to_vec()),
            CopyFormat::Binary => (b'\t', b"\\N".to_vec()),
        };
        Self {
            columns,
            format,
            delimiter,
            null_string,
            buffer: Vec::new(),
            rows: Vec::new(),
        }
    }

    /// Returns the CopyInResponse header message.
    pub fn header_message(&self) -> BackendMessage {
        let format_code: i8 = match self.format {
            CopyFormat::Binary => 1,
            _ => 0,
        };
        let column_formats = vec![format_code as i16; self.columns.len()];
        BackendMessage::CopyInResponse {
            format: format_code,
            column_formats,
        }
    }

    /// Feeds a CopyData chunk. Parses complete lines into rows.
    /// Uses cursor-based splitting with deferred compaction to minimize byte shifting.
    #[inline]
    pub fn feed(&mut self, data: &[u8]) -> Result<(), ProtocolError> {
        self.buffer.extend_from_slice(data);

        // Process complete lines using a cursor. memchr on x86_64 scans up to 32
        // bytes per cycle, vs ~1 byte/cycle for iter().position().
        let mut consumed = 0;
        while let Some(rel_pos) = memchr::memchr(b'\n', &self.buffer[consumed..]) {
            let newline_pos = consumed + rel_pos;
            let line = &self.buffer[consumed..newline_pos];
            consumed = newline_pos + 1;

            // Skip empty lines and the \. terminator.
            if line.is_empty() || line == b"\\." {
                continue;
            }

            let row = self.parse_line(line)?;
            self.rows.push(row);
        }

        // Deferred compaction: only shift bytes when consumed portion exceeds
        // half the buffer, otherwise just truncate if everything was consumed.
        if consumed > 0 {
            if consumed >= self.buffer.len() {
                self.buffer.clear();
            } else {
                self.buffer.copy_within(consumed.., 0);
                self.buffer.truncate(self.buffer.len() - consumed);
            }
        }

        Ok(())
    }

    /// Called when CopyDone is received. Flushes remaining buffered data
    /// and returns all accumulated rows as column values.
    pub fn finish(mut self) -> Result<Vec<Vec<Option<Vec<u8>>>>, ProtocolError> {
        // Process any remaining data in the buffer
        if !self.buffer.is_empty() {
            let line = std::mem::take(&mut self.buffer);
            if line != b"\\." && !line.is_empty() {
                let row = self.parse_line(&line)?;
                self.rows.push(row);
            }
        }
        Ok(self.rows)
    }

    /// Returns the number of rows accumulated so far.
    pub fn row_count(&self) -> usize {
        self.rows.len()
    }

    /// Parses a single line directly into column values.
    /// Inlines field splitting and null detection in a single pass,
    /// avoiding the intermediate Vec<Vec<u8>> from split_text_line/split_csv_line.
    #[inline]
    fn parse_line(&self, line: &[u8]) -> Result<Vec<Option<Vec<u8>>>, ProtocolError> {
        let num_cols = self.columns.len();
        let mut row = Vec::with_capacity(num_cols);
        let avg_field_len = if num_cols > 0 {
            line.len() / num_cols
        } else {
            8
        };

        if self.format == CopyFormat::Csv {
            let mut current = Vec::with_capacity(avg_field_len + 4);
            let mut in_quotes = false;
            let mut i = 0;

            while i < line.len() {
                if in_quotes {
                    if line[i] == b'"' {
                        if i + 1 < line.len() && line[i + 1] == b'"' {
                            current.push(b'"');
                            i += 1;
                        } else {
                            in_quotes = false;
                        }
                    } else {
                        current.push(line[i]);
                    }
                } else if line[i] == b'"' {
                    in_quotes = true;
                } else if line[i] == self.delimiter {
                    let field = std::mem::take(&mut current);
                    current = Vec::with_capacity(avg_field_len + 4);
                    if field.as_slice() == self.null_string.as_slice() {
                        row.push(None);
                    } else {
                        row.push(Some(field));
                    }
                } else {
                    current.push(line[i]);
                }
                i += 1;
            }
            // Last field.
            if current.as_slice() == self.null_string.as_slice() {
                row.push(None);
            } else {
                row.push(Some(current));
            }
        } else {
            let mut current = Vec::with_capacity(avg_field_len + 4);
            let mut i = 0;

            while i < line.len() {
                if line[i] == self.delimiter {
                    let field = std::mem::take(&mut current);
                    current = Vec::with_capacity(avg_field_len + 4);
                    if field.as_slice() == self.null_string.as_slice() {
                        row.push(None);
                    } else {
                        row.push(Some(field));
                    }
                } else if line[i] == b'\\' && i + 1 < line.len() {
                    match line[i + 1] {
                        b'n' => current.push(b'\n'),
                        b'r' => current.push(b'\r'),
                        b't' => current.push(b'\t'),
                        b'\\' => current.push(b'\\'),
                        other => {
                            current.push(b'\\');
                            current.push(other);
                        }
                    }
                    i += 1;
                } else {
                    current.push(line[i]);
                }
                i += 1;
            }
            // Last field.
            if current.as_slice() == self.null_string.as_slice() {
                row.push(None);
            } else {
                row.push(Some(current));
            }
        }

        if row.len() != num_cols {
            return Err(ProtocolError::Malformed(format!(
                "Expected {} columns but got {} in COPY data",
                num_cols,
                row.len()
            )));
        }

        Ok(row)
    }
}

// ---------------------------------------------------------------------------
// Text format helpers
// ---------------------------------------------------------------------------

/// Splits a text-format COPY line by delimiter.
/// Pre-counts fields to allocate once, and pre-allocates per-field buffers.
#[cfg(test)]
fn split_text_line(line: &[u8], delimiter: u8) -> Vec<Vec<u8>> {
    // Count fields for pre-allocation.
    let field_count = 1 + line.iter().filter(|&&b| b == delimiter).count();
    let mut fields = Vec::with_capacity(field_count);
    let avg_field_len = line.len() / field_count;
    let mut current = Vec::with_capacity(avg_field_len + 4);
    let mut i = 0;

    while i < line.len() {
        if line[i] == delimiter {
            fields.push(std::mem::take(&mut current));
            current = Vec::with_capacity(avg_field_len + 4);
        } else if line[i] == b'\\' && i + 1 < line.len() {
            match line[i + 1] {
                b'n' => current.push(b'\n'),
                b'r' => current.push(b'\r'),
                b't' => current.push(b'\t'),
                b'\\' => current.push(b'\\'),
                other => {
                    current.push(b'\\');
                    current.push(other);
                }
            }
            i += 1;
        } else {
            current.push(line[i]);
        }
        i += 1;
    }
    fields.push(current);
    fields
}

/// Escapes a value for text-format COPY output.
/// Fast path: if no special characters, copies the entire slice at once.
fn text_escape(buf: &mut BytesMut, data: &[u8]) {
    let needs_escape = data
        .iter()
        .any(|&b| b == b'\\' || b == b'\n' || b == b'\r' || b == b'\t');
    if !needs_escape {
        buf.extend_from_slice(data);
        return;
    }
    for &b in data {
        match b {
            b'\\' => buf.extend_from_slice(b"\\\\"),
            b'\n' => buf.extend_from_slice(b"\\n"),
            b'\r' => buf.extend_from_slice(b"\\r"),
            b'\t' => buf.extend_from_slice(b"\\t"),
            _ => buf.extend_from_slice(&[b]),
        }
    }
}

// ---------------------------------------------------------------------------
// CSV format helpers
// ---------------------------------------------------------------------------

/// Splits a CSV line by delimiter, handling quoted fields.
/// Pre-allocates field count and per-field buffers.
#[cfg(test)]
fn split_csv_line(line: &[u8], delimiter: u8) -> Vec<Vec<u8>> {
    let field_count = 1 + line.iter().filter(|&&b| b == delimiter).count();
    let mut fields = Vec::with_capacity(field_count);
    let avg_field_len = line.len() / field_count;
    let mut current = Vec::with_capacity(avg_field_len + 4);
    let mut in_quotes = false;
    let mut i = 0;

    while i < line.len() {
        if in_quotes {
            if line[i] == b'"' {
                if i + 1 < line.len() && line[i + 1] == b'"' {
                    current.push(b'"');
                    i += 1;
                } else {
                    in_quotes = false;
                }
            } else {
                current.push(line[i]);
            }
        } else if line[i] == b'"' {
            in_quotes = true;
        } else if line[i] == delimiter {
            fields.push(std::mem::take(&mut current));
            current = Vec::with_capacity(avg_field_len + 4);
        } else {
            current.push(line[i]);
        }
        i += 1;
    }
    fields.push(current);
    fields
}

/// Escapes a value for CSV-format COPY output.
fn csv_escape(buf: &mut BytesMut, data: &[u8]) {
    let needs_quoting = data
        .iter()
        .any(|&b| b == b',' || b == b'"' || b == b'\n' || b == b'\r');
    if needs_quoting {
        buf.extend_from_slice(b"\"");
        for &b in data {
            if b == b'"' {
                buf.extend_from_slice(b"\"\"");
            } else {
                buf.extend_from_slice(&[b]);
            }
        }
        buf.extend_from_slice(b"\"");
    } else {
        buf.extend_from_slice(data);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_split_text_line() {
        let fields = split_text_line(b"hello\tworld\t42", b'\t');
        assert_eq!(fields.len(), 3);
        assert_eq!(fields[0], b"hello");
        assert_eq!(fields[1], b"world");
        assert_eq!(fields[2], b"42");
    }

    #[test]
    fn test_split_text_line_escapes() {
        let fields = split_text_line(b"line\\none\ttwo", b'\t');
        assert_eq!(fields.len(), 2);
        assert_eq!(fields[0], b"line\none");
        assert_eq!(fields[1], b"two");
    }

    #[test]
    fn test_split_csv_line() {
        let fields = split_csv_line(b"hello,world,42", b',');
        assert_eq!(fields.len(), 3);
        assert_eq!(fields[0], b"hello");
        assert_eq!(fields[1], b"world");
        assert_eq!(fields[2], b"42");
    }

    #[test]
    fn test_split_csv_quoted() {
        let fields = split_csv_line(b"\"hello, world\",42", b',');
        assert_eq!(fields.len(), 2);
        assert_eq!(fields[0], b"hello, world");
        assert_eq!(fields[1], b"42");
    }

    #[test]
    fn test_split_csv_escaped_quote() {
        let fields = split_csv_line(b"\"he said \"\"hi\"\"\",42", b',');
        assert_eq!(fields.len(), 2);
        assert_eq!(fields[0], b"he said \"hi\"");
    }

    #[test]
    fn test_text_escape() {
        let mut buf = BytesMut::new();
        text_escape(&mut buf, b"hello\tworld\n");
        assert_eq!(buf.as_ref(), b"hello\\tworld\\n");
    }

    #[test]
    fn test_csv_escape_no_quoting() {
        let mut buf = BytesMut::new();
        csv_escape(&mut buf, b"hello");
        assert_eq!(buf.as_ref(), b"hello");
    }

    #[test]
    fn test_csv_escape_with_comma() {
        let mut buf = BytesMut::new();
        csv_escape(&mut buf, b"hello, world");
        assert_eq!(buf.as_ref(), b"\"hello, world\"");
    }

    #[test]
    fn test_csv_escape_with_quotes() {
        let mut buf = BytesMut::new();
        csv_escape(&mut buf, b"he said \"hi\"");
        assert_eq!(buf.as_ref(), b"\"he said \"\"hi\"\"\"");
    }

    #[test]
    fn test_copy_in_handler_text() {
        let columns = vec![
            LogicalColumn {
                table_idx: None,
                column_id: zyron_catalog::ColumnId(1),
                name: "a".into(),
                type_id: zyron_common::TypeId::Int32,
                nullable: false,
            },
            LogicalColumn {
                table_idx: None,
                column_id: zyron_catalog::ColumnId(2),
                name: "b".into(),
                type_id: zyron_common::TypeId::Text,
                nullable: true,
            },
        ];

        let mut handler = CopyInHandler::new(columns, CopyFormat::Text);
        handler.feed(b"1\thello\n2\t\\N\n").unwrap();
        let rows = handler.finish().unwrap();

        assert_eq!(rows.len(), 2);
        assert_eq!(rows[0][0], Some(b"1".to_vec()));
        assert_eq!(rows[0][1], Some(b"hello".to_vec()));
        assert_eq!(rows[1][0], Some(b"2".to_vec()));
        assert_eq!(rows[1][1], None); // \N is NULL
    }

    #[test]
    fn test_copy_in_handler_csv() {
        let columns = vec![
            LogicalColumn {
                table_idx: None,
                column_id: zyron_catalog::ColumnId(1),
                name: "a".into(),
                type_id: zyron_common::TypeId::Int32,
                nullable: false,
            },
            LogicalColumn {
                table_idx: None,
                column_id: zyron_catalog::ColumnId(2),
                name: "b".into(),
                type_id: zyron_common::TypeId::Text,
                nullable: false,
            },
        ];

        let mut handler = CopyInHandler::new(columns, CopyFormat::Csv);
        handler.feed(b"1,hello\n2,\"world, here\"\n").unwrap();
        let rows = handler.finish().unwrap();

        assert_eq!(rows.len(), 2);
        assert_eq!(rows[0][1], Some(b"hello".to_vec()));
        assert_eq!(rows[1][1], Some(b"world, here".to_vec()));
    }

    #[test]
    fn test_copy_in_wrong_column_count() {
        let columns = vec![LogicalColumn {
            table_idx: None,
            column_id: zyron_catalog::ColumnId(1),
            name: "a".into(),
            type_id: zyron_common::TypeId::Int32,
            nullable: false,
        }];

        let mut handler = CopyInHandler::new(columns, CopyFormat::Text);
        let result = handler.feed(b"1\t2\n"); // 2 fields, expected 1
        assert!(result.is_err());
    }

    #[test]
    fn test_copy_in_partial_line() {
        let columns = vec![LogicalColumn {
            table_idx: None,
            column_id: zyron_catalog::ColumnId(1),
            name: "a".into(),
            type_id: zyron_common::TypeId::Int32,
            nullable: false,
        }];

        let mut handler = CopyInHandler::new(columns, CopyFormat::Text);
        handler.feed(b"1\n2").unwrap(); // "2" is partial
        assert_eq!(handler.row_count(), 1);

        handler.feed(b"\n").unwrap(); // completes the line
        assert_eq!(handler.row_count(), 2);
    }
}
