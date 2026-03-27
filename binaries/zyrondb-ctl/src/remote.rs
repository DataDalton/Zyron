#![allow(non_snake_case)]
//! Minimal synchronous PostgreSQL wire protocol client.
//!
//! Connects to a ZyronDB server (or any PostgreSQL-compatible server) using
//! the v3 simple query protocol over a plain TCP socket.

use std::io::{BufReader, BufWriter, Read, Write};
use std::net::TcpStream;

/// Column names and row data returned from a query.
pub struct QueryResult {
    pub columns: Vec<String>,
    pub rows: Vec<Vec<String>>,
    pub tag: String,
}

/// Synchronous PostgreSQL wire protocol client for admin commands.
pub struct RemoteClient {
    reader: BufReader<TcpStream>,
    writer: BufWriter<TcpStream>,
}

impl RemoteClient {
    /// Opens a TCP connection and performs the PostgreSQL startup handshake.
    pub fn connect(host: &str, port: u16, user: &str, db: &str) -> Result<Self, String> {
        let addr = format!("{}:{}", host, port);
        let stream = TcpStream::connect(&addr)
            .map_err(|e| format!("failed to connect to {}: {}", addr, e))?;

        let readHalf = stream
            .try_clone()
            .map_err(|e| format!("failed to clone stream: {}", e))?;
        let mut client = Self {
            reader: BufReader::new(readHalf),
            writer: BufWriter::new(stream),
        };

        client.sendStartup(user, db)?;
        client.handleStartupResponse()?;
        Ok(client)
    }

    /// Sends a SQL statement via the SimpleQuery protocol and reads the full
    /// response (RowDescription, DataRow, CommandComplete, ReadyForQuery).
    pub fn execute(&mut self, sql: &str) -> Result<QueryResult, String> {
        self.sendQuery(sql)?;
        self.readQueryResponse()
    }

    /// Sends a Terminate message and drops the connection.
    pub fn close(&mut self) -> Result<(), String> {
        // Terminate: 'X' + length 4
        let msg: [u8; 5] = [b'X', 0, 0, 0, 4];
        self.writer
            .write_all(&msg)
            .map_err(|e| format!("failed to send terminate: {}", e))?;
        self.writer
            .flush()
            .map_err(|e| format!("failed to flush terminate: {}", e))?;
        Ok(())
    }

    // -----------------------------------------------------------------------
    // Internal protocol helpers
    // -----------------------------------------------------------------------

    fn sendStartup(&mut self, user: &str, db: &str) -> Result<(), String> {
        let mut payload = Vec::new();
        // Protocol version 3.0
        payload.extend_from_slice(&196608_i32.to_be_bytes());

        payload.extend_from_slice(b"user\0");
        payload.extend_from_slice(user.as_bytes());
        payload.push(0);

        payload.extend_from_slice(b"database\0");
        payload.extend_from_slice(db.as_bytes());
        payload.push(0);

        payload.push(0); // terminal null

        let len = (payload.len() + 4) as i32;
        self.writer
            .write_all(&len.to_be_bytes())
            .map_err(|e| format!("write startup length: {}", e))?;
        self.writer
            .write_all(&payload)
            .map_err(|e| format!("write startup payload: {}", e))?;
        self.writer
            .flush()
            .map_err(|e| format!("flush startup: {}", e))?;
        Ok(())
    }

    fn handleStartupResponse(&mut self) -> Result<(), String> {
        loop {
            let (msgType, body) = self.readMessage()?;
            match msgType {
                b'R' => {
                    // AuthenticationOk = 0, anything else is unsupported
                    if body.len() >= 4 {
                        let authType = i32::from_be_bytes([body[0], body[1], body[2], body[3]]);
                        if authType != 0 {
                            return Err(format!("unsupported auth type: {}", authType));
                        }
                    }
                }
                b'S' | b'K' => {
                    // ParameterStatus or BackendKeyData, skip
                }
                b'Z' => {
                    return Ok(());
                }
                b'E' => {
                    let msg = parseErrorFields(&body);
                    return Err(msg);
                }
                _ => {}
            }
        }
    }

    fn sendQuery(&mut self, sql: &str) -> Result<(), String> {
        let bodyLen = sql.len() + 1; // SQL + null terminator
        let len = (bodyLen + 4) as i32;

        self.writer
            .write_all(&[b'Q'])
            .map_err(|e| format!("write query tag: {}", e))?;
        self.writer
            .write_all(&len.to_be_bytes())
            .map_err(|e| format!("write query length: {}", e))?;
        self.writer
            .write_all(sql.as_bytes())
            .map_err(|e| format!("write query body: {}", e))?;
        self.writer
            .write_all(&[0])
            .map_err(|e| format!("write query null: {}", e))?;
        self.writer
            .flush()
            .map_err(|e| format!("flush query: {}", e))?;
        Ok(())
    }

    fn readQueryResponse(&mut self) -> Result<QueryResult, String> {
        let mut columns: Vec<String> = Vec::new();
        let mut rows: Vec<Vec<String>> = Vec::new();
        let mut tag = String::new();

        loop {
            let (msgType, body) = self.readMessage()?;
            match msgType {
                b'T' => {
                    columns = parseRowDescription(&body);
                }
                b'D' => {
                    rows.push(parseDataRow(&body));
                }
                b'C' => {
                    tag = String::from_utf8_lossy(&body)
                        .trim_end_matches('\0')
                        .to_string();
                }
                b'E' => {
                    let msg = parseErrorFields(&body);
                    return Err(msg);
                }
                b'I' => {
                    // EmptyQueryResponse
                }
                b'Z' => {
                    return Ok(QueryResult { columns, rows, tag });
                }
                _ => {}
            }
        }
    }

    /// Reads one wire protocol message: 1-byte type, 4-byte length, then body.
    fn readMessage(&mut self) -> Result<(u8, Vec<u8>), String> {
        let mut typeBuf = [0u8; 1];
        self.reader
            .read_exact(&mut typeBuf)
            .map_err(|e| format!("read message type: {}", e))?;

        let mut lenBuf = [0u8; 4];
        self.reader
            .read_exact(&mut lenBuf)
            .map_err(|e| format!("read message length: {}", e))?;

        let len = i32::from_be_bytes(lenBuf) as usize;
        let bodyLen = len.saturating_sub(4);

        let mut body = vec![0u8; bodyLen];
        if bodyLen > 0 {
            self.reader
                .read_exact(&mut body)
                .map_err(|e| format!("read message body: {}", e))?;
        }

        Ok((typeBuf[0], body))
    }
}

// ---------------------------------------------------------------------------
// Message parsing helpers
// ---------------------------------------------------------------------------

/// Parses a RowDescription message body into column names.
fn parseRowDescription(body: &[u8]) -> Vec<String> {
    if body.len() < 2 {
        return Vec::new();
    }
    let fieldCount = i16::from_be_bytes([body[0], body[1]]) as usize;
    let mut names = Vec::with_capacity(fieldCount);
    let mut pos = 2;

    for _ in 0..fieldCount {
        if pos >= body.len() {
            break;
        }
        let nameEnd = body[pos..]
            .iter()
            .position(|&b| b == 0)
            .unwrap_or(body.len() - pos);
        let name = String::from_utf8_lossy(&body[pos..pos + nameEnd]).to_string();
        names.push(name);
        pos += nameEnd + 1; // skip null
        pos += 18; // table_oid(4) + col_attr(2) + type_oid(4) + type_size(2) + type_mod(4) + format(2)
    }

    names
}

/// Parses a DataRow message body into string values.
fn parseDataRow(body: &[u8]) -> Vec<String> {
    if body.len() < 2 {
        return Vec::new();
    }
    let colCount = i16::from_be_bytes([body[0], body[1]]) as usize;
    let mut values = Vec::with_capacity(colCount);
    let mut pos = 2;

    for _ in 0..colCount {
        if pos + 4 > body.len() {
            break;
        }
        let valLen = i32::from_be_bytes([body[pos], body[pos + 1], body[pos + 2], body[pos + 3]]);
        pos += 4;

        if valLen == -1 {
            values.push("NULL".to_string());
        } else if valLen < -1 {
            values.push("<invalid>".to_string());
            break;
        } else {
            let end = pos + valLen as usize;
            if end <= body.len() {
                let val = String::from_utf8_lossy(&body[pos..end]).to_string();
                values.push(val);
                pos = end;
            } else {
                values.push("<truncated>".to_string());
                break;
            }
        }
    }

    values
}

/// Extracts the human-readable error message ('M' field) from an ErrorResponse body.
fn parseErrorFields(body: &[u8]) -> String {
    let mut message = String::new();
    let mut pos = 0;

    while pos < body.len() {
        let fieldType = body[pos];
        pos += 1;
        if fieldType == 0 {
            break;
        }
        let strEnd = body[pos..]
            .iter()
            .position(|&b| b == 0)
            .unwrap_or(body.len() - pos);
        let value = String::from_utf8_lossy(&body[pos..pos + strEnd]);
        pos += strEnd + 1;

        if fieldType == b'M' {
            message = value.to_string();
        }
    }

    if message.is_empty() {
        "unknown error".into()
    } else {
        message
    }
}
