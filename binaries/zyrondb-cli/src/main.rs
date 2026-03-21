//! ZyronDB CLI client.
//!
//! A minimal psql-like client that connects via TCP, sends SQL queries using
//! the PostgreSQL simple query protocol, and displays results as formatted text.

use bytes::{BufMut, BytesMut};
use std::io::{self, BufRead, Write};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpStream;

const DEFAULT_HOST: &str = "127.0.0.1";
const DEFAULT_PORT: u16 = 5432;
const DEFAULT_USER: &str = "zyron";
const DEFAULT_DATABASE: &str = "zyron";

struct CliOptions {
    host: String,
    port: u16,
    user: String,
    database: String,
}

fn parse_args() -> Option<CliOptions> {
    let args: Vec<String> = std::env::args().collect();
    let mut opts = CliOptions {
        host: DEFAULT_HOST.into(),
        port: DEFAULT_PORT,
        user: DEFAULT_USER.into(),
        database: DEFAULT_DATABASE.into(),
    };
    let mut i = 1;

    while i < args.len() {
        match args[i].as_str() {
            "--help" | "-h" => {
                println!(
                    "Usage: zyrondb-cli [OPTIONS]

Options:
  --host <addr>    Server host (default: 127.0.0.1)
  --port <number>  Server port (default: 5432)
  --user <name>    Username (default: zyron)
  --database <db>  Database name (default: zyron)
  --help, -h       Print this help"
                );
                return None;
            }
            "--host" => {
                i += 1;
                opts.host = args[i].clone();
            }
            "--port" => {
                i += 1;
                opts.port = args[i].parse().expect("invalid port");
            }
            "--user" => {
                i += 1;
                opts.user = args[i].clone();
            }
            "--database" => {
                i += 1;
                opts.database = args[i].clone();
            }
            other => {
                eprintln!("Unknown argument: {}", other);
                std::process::exit(1);
            }
        }
        i += 1;
    }

    Some(opts)
}

#[tokio::main]
async fn main() {
    let opts = match parse_args() {
        Some(o) => o,
        None => return,
    };

    let addr = format!("{}:{}", opts.host, opts.port);
    let mut stream = match TcpStream::connect(&addr).await {
        Ok(s) => s,
        Err(e) => {
            eprintln!("Failed to connect to {}: {}", addr, e);
            std::process::exit(1);
        }
    };

    // Send startup message
    if let Err(e) = send_startup(&mut stream, &opts.user, &opts.database).await {
        eprintln!("Startup failed: {}", e);
        std::process::exit(1);
    }

    // Read authentication and parameter status messages until ReadyForQuery
    if let Err(e) = handle_startup_response(&mut stream).await {
        eprintln!("Authentication failed: {}", e);
        std::process::exit(1);
    }

    println!(
        "Connected to {}:{} as {} (database: {})",
        opts.host, opts.port, opts.user, opts.database
    );
    println!("Type SQL queries, \\q to quit.\n");

    // REPL loop
    let stdin = io::stdin();
    let mut input = String::new();

    loop {
        print!("zyron> ");
        io::stdout().flush().unwrap();
        input.clear();

        if stdin.lock().read_line(&mut input).unwrap() == 0 {
            break; // EOF
        }

        let trimmed = input.trim();
        if trimmed.is_empty() {
            continue;
        }

        match trimmed {
            "\\q" | "quit" | "exit" => break,
            _ => {}
        }

        // Send simple query
        if let Err(e) = send_query(&mut stream, trimmed).await {
            eprintln!("Query send error: {}", e);
            continue;
        }

        // Read and display results
        if let Err(e) = read_query_response(&mut stream).await {
            eprintln!("Query error: {}", e);
        }
    }

    // Send Terminate message
    let mut buf = BytesMut::with_capacity(5);
    buf.put_u8(b'X');
    buf.put_i32(4);
    let _ = stream.write_all(&buf).await;

    println!("Disconnected.");
}

/// Sends a PostgreSQL StartupMessage.
async fn send_startup(stream: &mut TcpStream, user: &str, database: &str) -> io::Result<()> {
    let mut payload = BytesMut::new();
    // Protocol version 3.0
    payload.put_i32(196608); // 3 << 16

    payload.extend_from_slice(b"user\0");
    payload.extend_from_slice(user.as_bytes());
    payload.put_u8(0);

    payload.extend_from_slice(b"database\0");
    payload.extend_from_slice(database.as_bytes());
    payload.put_u8(0);

    payload.put_u8(0); // terminal null

    let len = (payload.len() + 4) as i32;
    let mut msg = BytesMut::with_capacity(len as usize);
    msg.put_i32(len);
    msg.extend_from_slice(&payload);

    stream.write_all(&msg).await?;
    stream.flush().await?;
    Ok(())
}

/// Reads startup response until ReadyForQuery.
async fn handle_startup_response(stream: &mut TcpStream) -> io::Result<()> {
    loop {
        let msg_type = stream.read_u8().await?;
        let len = stream.read_i32().await? as usize;
        let body_len = len.saturating_sub(4);

        let mut body = vec![0u8; body_len];
        if body_len > 0 {
            stream.read_exact(&mut body).await?;
        }

        match msg_type {
            b'R' => {
                // AuthenticationOk = 0
                if body.len() >= 4 {
                    let auth_type = i32::from_be_bytes([body[0], body[1], body[2], body[3]]);
                    if auth_type != 0 {
                        return Err(io::Error::new(
                            io::ErrorKind::PermissionDenied,
                            format!("Unsupported auth type: {}", auth_type),
                        ));
                    }
                }
            }
            b'S' => {
                // ParameterStatus - ignore
            }
            b'K' => {
                // BackendKeyData - ignore
            }
            b'Z' => {
                // ReadyForQuery
                return Ok(());
            }
            b'E' => {
                let msg = parse_error_fields(&body);
                return Err(io::Error::new(io::ErrorKind::Other, msg));
            }
            _ => {
                // Unknown, skip
            }
        }
    }
}

/// Sends a SimpleQuery message.
async fn send_query(stream: &mut TcpStream, sql: &str) -> io::Result<()> {
    let body_len = sql.len() + 1; // SQL + null terminator
    let len = (body_len + 4) as i32;

    let mut msg = BytesMut::with_capacity(1 + len as usize);
    msg.put_u8(b'Q');
    msg.put_i32(len);
    msg.extend_from_slice(sql.as_bytes());
    msg.put_u8(0);

    stream.write_all(&msg).await?;
    stream.flush().await?;
    Ok(())
}

/// Reads query response messages (RowDescription, DataRow, CommandComplete,
/// ErrorResponse) until ReadyForQuery.
async fn read_query_response(stream: &mut TcpStream) -> io::Result<()> {
    let mut column_names: Vec<String> = Vec::new();
    let mut row_count = 0u64;

    loop {
        let msg_type = stream.read_u8().await?;
        let len = stream.read_i32().await? as usize;
        let body_len = len.saturating_sub(4);

        let mut body = vec![0u8; body_len];
        if body_len > 0 {
            stream.read_exact(&mut body).await?;
        }

        match msg_type {
            b'T' => {
                // RowDescription
                column_names = parse_row_description(&body);
                // Print header
                println!("{}", column_names.join(" | "));
                println!("{}", "-".repeat(column_names.len() * 15));
            }
            b'D' => {
                // DataRow
                let values = parse_data_row(&body);
                println!("{}", values.join(" | "));
                row_count += 1;
            }
            b'C' => {
                // CommandComplete
                let tag = String::from_utf8_lossy(&body)
                    .trim_end_matches('\0')
                    .to_string();
                if row_count > 0 {
                    println!("({} rows)", row_count);
                }
                println!("{}", tag);
            }
            b'E' => {
                // ErrorResponse
                let msg = parse_error_fields(&body);
                eprintln!("ERROR: {}", msg);
            }
            b'Z' => {
                // ReadyForQuery
                return Ok(());
            }
            b'I' => {
                // EmptyQueryResponse
                println!("(empty query)");
            }
            _ => {
                // Skip unknown message types
            }
        }
    }
}

/// Parses a RowDescription message body into column names.
fn parse_row_description(body: &[u8]) -> Vec<String> {
    if body.len() < 2 {
        return Vec::new();
    }
    let field_count = i16::from_be_bytes([body[0], body[1]]) as usize;
    let mut names = Vec::with_capacity(field_count);
    let mut pos = 2;

    for _ in 0..field_count {
        // Read null-terminated field name
        let name_end = body[pos..].iter().position(|&b| b == 0).unwrap_or(0);
        let name = String::from_utf8_lossy(&body[pos..pos + name_end]).to_string();
        names.push(name);
        pos += name_end + 1; // skip null
        pos += 18; // table_oid(4) + col_attr(2) + type_oid(4) + type_size(2) + type_mod(4) + format(2)
    }

    names
}

/// Parses a DataRow message body into string values.
fn parse_data_row(body: &[u8]) -> Vec<String> {
    if body.len() < 2 {
        return Vec::new();
    }
    let col_count = i16::from_be_bytes([body[0], body[1]]) as usize;
    let mut values = Vec::with_capacity(col_count);
    let mut pos = 2;

    for _ in 0..col_count {
        if pos + 4 > body.len() {
            break;
        }
        let val_len = i32::from_be_bytes([body[pos], body[pos + 1], body[pos + 2], body[pos + 3]]);
        pos += 4;

        if val_len == -1 {
            values.push("NULL".to_string());
        } else if val_len < -1 {
            values.push("<invalid>".to_string());
            break;
        } else {
            let end = pos + val_len as usize;
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

/// Extracts error message from an ErrorResponse body.
fn parse_error_fields(body: &[u8]) -> String {
    let mut message = String::new();
    let mut pos = 0;

    while pos < body.len() {
        let field_type = body[pos];
        pos += 1;
        if field_type == 0 {
            break;
        }
        let str_end = body[pos..]
            .iter()
            .position(|&b| b == 0)
            .unwrap_or(body.len() - pos);
        let value = String::from_utf8_lossy(&body[pos..pos + str_end]);
        pos += str_end + 1;

        if field_type == b'M' {
            message = value.to_string();
        }
    }

    if message.is_empty() {
        "Unknown error".into()
    } else {
        message
    }
}
