//! ZyronDB CLI client.
//!
//! A psql-like client that connects via TCP, sends SQL queries using
//! the PostgreSQL simple query protocol, and displays formatted results.
//! Supports meta-commands, multi-line input, query history, and multiple
//! output formats (table, expanded, CSV).

mod completion;
mod display;
mod history;
mod meta;

use bytes::{BufMut, BytesMut};
use std::fs::File;
use std::io::{self, BufRead, Read, Write};
use std::net::TcpStream;
use std::time::Instant;

use meta::{MetaCommand, meta_to_sql, parse_meta_command};

const DEFAULT_HOST: &str = "127.0.0.1";
const DEFAULT_PORT: u16 = 5432;
const DEFAULT_USER: &str = "zyron";
const DEFAULT_DATABASE: &str = "zyron";
const MAX_HISTORY_ENTRIES: usize = 1000;

struct CliOptions {
    host: String,
    port: u16,
    user: String,
    database: String,
}

/// Tracks client-side display and input state for the REPL session.
struct CliState {
    timing: bool,
    expanded: bool,
    csv_mode: bool,
    output_file: Option<File>,
    history: Vec<String>,
    multiline_buffer: String,
}

impl CliState {
    fn new(history: Vec<String>) -> Self {
        Self {
            timing: false,
            expanded: false,
            csv_mode: false,
            output_file: None,
            history,
            multiline_buffer: String::new(),
        }
    }
}

/// Query result data returned from read_query_response.
struct QueryResult {
    columns: Vec<String>,
    rows: Vec<Vec<String>>,
    command_tag: Option<String>,
    error: Option<String>,
}

/// Parses a connection string in the format zyron://user:pass@host:port/database.
fn parse_connection_string(url: &str) -> Result<CliOptions, String> {
    let rest = url
        .strip_prefix("zyron://")
        .ok_or_else(|| "Connection string must start with zyron://".to_string())?;

    let mut user = DEFAULT_USER.to_string();
    let mut host = DEFAULT_HOST.to_string();
    let mut port = DEFAULT_PORT;
    let mut database = DEFAULT_DATABASE.to_string();

    // Split into authority and path: user:pass@host:port / database
    let (authority, path) = match rest.find('/') {
        Some(idx) => (&rest[..idx], &rest[idx + 1..]),
        None => (rest, ""),
    };

    if !path.is_empty() {
        database = path.to_string();
    }

    // Split authority into userinfo and hostinfo
    let (userinfo, hostinfo) = match authority.find('@') {
        Some(idx) => (&authority[..idx], &authority[idx + 1..]),
        None => ("", authority),
    };

    // Parse user from userinfo (ignore password after colon)
    if !userinfo.is_empty() {
        let user_part = match userinfo.find(':') {
            Some(idx) => &userinfo[..idx],
            None => userinfo,
        };
        if !user_part.is_empty() {
            user = user_part.to_string();
        }
    }

    // Parse host and port from hostinfo
    if !hostinfo.is_empty() {
        match hostinfo.rfind(':') {
            Some(idx) => {
                let h = &hostinfo[..idx];
                let p = &hostinfo[idx + 1..];
                if !h.is_empty() {
                    host = h.to_string();
                }
                port = p
                    .parse()
                    .map_err(|_| format!("Invalid port in connection string: {}", p))?;
            }
            None => {
                host = hostinfo.to_string();
            }
        }
    }

    Ok(CliOptions {
        host,
        port,
        user,
        database,
    })
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
            "--help" => {
                print_help();
                return None;
            }
            "--host" | "-h" => {
                i += 1;
                if i >= args.len() {
                    eprintln!("Missing value for --host");
                    std::process::exit(1);
                }
                opts.host = args[i].clone();
            }
            "--port" | "-p" => {
                i += 1;
                if i >= args.len() {
                    eprintln!("Missing value for --port");
                    std::process::exit(1);
                }
                match args[i].parse() {
                    Ok(p) => opts.port = p,
                    Err(_) => {
                        eprintln!("Invalid port: {}", args[i]);
                        std::process::exit(1);
                    }
                }
            }
            "--user" | "-U" => {
                i += 1;
                if i >= args.len() {
                    eprintln!("Missing value for --user");
                    std::process::exit(1);
                }
                opts.user = args[i].clone();
            }
            "--database" | "-d" => {
                i += 1;
                if i >= args.len() {
                    eprintln!("Missing value for --database");
                    std::process::exit(1);
                }
                opts.database = args[i].clone();
            }
            "--version" => {
                println!("zyrondb-cli {}", env!("CARGO_PKG_VERSION"));
                return None;
            }
            other => {
                // Check if it looks like a connection string
                if other.starts_with("zyron://") {
                    match parse_connection_string(other) {
                        Ok(parsed) => {
                            opts = parsed;
                        }
                        Err(e) => {
                            eprintln!("Invalid connection string: {}", e);
                            std::process::exit(1);
                        }
                    }
                } else {
                    eprintln!("Unknown argument: {}", other);
                    std::process::exit(1);
                }
            }
        }
        i += 1;
    }

    Some(opts)
}

fn print_help() {
    println!(
        "Usage: zyrondb-cli [OPTIONS] [CONNECTION_STRING]

Options:
  -h, --host <addr>      Server host (default: 127.0.0.1)
  -p, --port <number>    Server port (default: 5432)
  -U, --user <name>      Username (default: zyron)
  -d, --database <db>    Database name (default: zyron)
  --version              Print version
  --help                 Print this help

Connection string format:
  zyron://user:pass@host:port/database

Meta-commands:
  \\dt               List tables
  \\d <table>        Describe table columns
  \\di               List indexes
  \\du               List users
  \\dp               List privileges
  \\timing           Toggle query timing display
  \\x                Toggle expanded output
  \\csv              Toggle CSV output
  \\o [file]         Send output to file (no arg to reset to stdout)
  \\i <file>         Execute commands from file
  \\q                Quit
  \\?                Show this help"
    );
}

fn main() {
    let opts = match parse_args() {
        Some(o) => o,
        None => return,
    };

    let addr = format!("{}:{}", opts.host, opts.port);
    let mut stream = match TcpStream::connect(&addr) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("Failed to connect to {}: {}", addr, e);
            std::process::exit(1);
        }
    };

    // Send startup message
    if let Err(e) = send_startup(&mut stream, &opts.user, &opts.database) {
        eprintln!("Startup failed: {}", e);
        std::process::exit(1);
    }

    // Read authentication and parameter status messages until ReadyForQuery
    if let Err(e) = handle_startup_response(&mut stream) {
        eprintln!("Authentication failed: {}", e);
        std::process::exit(1);
    }

    println!(
        "Connected to {}:{} as {} (database: {})",
        opts.host, opts.port, opts.user, opts.database
    );
    println!("Type SQL queries, \\q to quit, \\? for help.\n");

    // Load history
    let hist_path = history::history_path();
    let hist = history::load_history(&hist_path);
    let mut state = CliState::new(hist);

    // REPL loop
    let stdin = io::stdin();
    let mut input = String::new();

    loop {
        // Show prompt based on multi-line state
        if state.multiline_buffer.is_empty() {
            print!("zyron=> ");
        } else {
            print!("zyron-> ");
        }
        if io::stdout().flush().is_err() {
            break;
        }

        input.clear();
        match stdin.lock().read_line(&mut input) {
            Ok(0) => break, // EOF
            Ok(_) => {}
            Err(_) => break,
        }

        let trimmed = input.trim();
        if trimmed.is_empty() {
            continue;
        }

        // Check for meta-command (only when not in multi-line mode)
        if state.multiline_buffer.is_empty() {
            if let Some(cmd) = parse_meta_command(trimmed) {
                handle_meta_command(&cmd, &mut state, &mut stream);
                match cmd {
                    MetaCommand::Quit => break,
                    _ => continue,
                }
            }
        }

        // Append to multi-line buffer
        if !state.multiline_buffer.is_empty() {
            state.multiline_buffer.push(' ');
        }
        state.multiline_buffer.push_str(trimmed);

        // Check if the statement is complete
        if is_statement_complete(&state.multiline_buffer) {
            let query = state.multiline_buffer.clone();
            state.multiline_buffer.clear();

            history::add_to_history(&mut state.history, &query);
            execute_and_display(&query, &mut state, &mut stream);
        }
    }

    // Save history on exit
    let _ = history::save_history(&hist_path, &state.history, MAX_HISTORY_ENTRIES);

    // Send Terminate message
    let mut buf = BytesMut::with_capacity(5);
    buf.put_u8(b'X');
    buf.put_i32(4);
    let _ = stream.write_all(&buf);

    println!("Disconnected.");
}

/// Handles a parsed meta-command, dispatching to local actions or server queries.
fn handle_meta_command(cmd: &MetaCommand, state: &mut CliState, stream: &mut TcpStream) {
    match cmd {
        MetaCommand::ToggleTiming => {
            state.timing = !state.timing;
            println!("Timing is {}.", if state.timing { "on" } else { "off" });
        }
        MetaCommand::ToggleExpanded => {
            state.expanded = !state.expanded;
            println!(
                "Expanded display is {}.",
                if state.expanded { "on" } else { "off" }
            );
        }
        MetaCommand::ToggleCsv => {
            state.csv_mode = !state.csv_mode;
            println!(
                "CSV output is {}.",
                if state.csv_mode { "on" } else { "off" }
            );
        }
        MetaCommand::OutputFile(path) => match path {
            Some(p) => match File::create(p) {
                Ok(f) => {
                    state.output_file = Some(f);
                    println!("Output redirected to {}", p);
                }
                Err(e) => {
                    eprintln!("Could not open file {}: {}", p, e);
                }
            },
            None => {
                state.output_file = None;
                println!("Output reset to stdout.");
            }
        },
        MetaCommand::InputFile(path) => {
            execute_file(path, state, stream);
        }
        MetaCommand::Help => {
            print_meta_help();
        }
        MetaCommand::Unknown(s) => {
            eprintln!("Unknown command: {}", s);
        }
        MetaCommand::Quit => {
            // Handled by caller
        }
        // Server-side meta-commands
        _ => {
            if let Some(sql) = meta_to_sql(cmd) {
                execute_and_display(&sql, state, stream);
            }
        }
    }
}

/// Prints the list of available meta-commands.
fn print_meta_help() {
    println!(
        "Meta-commands:
  \\dt               List tables
  \\d <table>        Describe table columns
  \\di               List indexes
  \\du               List users
  \\dp               List privileges
  \\timing           Toggle query timing display
  \\x                Toggle expanded output
  \\csv              Toggle CSV output
  \\o [file]         Send output to file (no arg to reset)
  \\i <file>         Execute commands from file
  \\q                Quit
  \\?                Show this help"
    );
}

/// Executes SQL commands from a file, one statement at a time.
fn execute_file(path: &str, state: &mut CliState, stream: &mut TcpStream) {
    let contents = match std::fs::read_to_string(path) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("Could not read file {}: {}", path, e);
            return;
        }
    };

    let mut buffer = String::new();
    for line in contents.lines() {
        let trimmed = line.trim();
        if trimmed.is_empty() || trimmed.starts_with("--") {
            continue;
        }

        if !buffer.is_empty() {
            buffer.push(' ');
        }
        buffer.push_str(trimmed);

        if is_statement_complete(&buffer) {
            execute_and_display(&buffer, state, stream);
            buffer.clear();
        }
    }

    // Execute any remaining buffer content
    if !buffer.trim().is_empty() {
        execute_and_display(&buffer, state, stream);
    }
}

/// Sends a query to the server, reads the response, and displays formatted output.
fn execute_and_display(sql: &str, state: &mut CliState, stream: &mut TcpStream) {
    let start = Instant::now();

    if let Err(e) = send_query(stream, sql) {
        eprintln!("Query send error: {}", e);
        return;
    }

    let result = match read_query_response(stream) {
        Ok(r) => r,
        Err(e) => {
            eprintln!("Query error: {}", e);
            return;
        }
    };

    let elapsed = start.elapsed();

    // Handle error responses
    if let Some(ref err_msg) = result.error {
        eprintln!("ERROR: {}", err_msg);
    }

    // Format the output
    let output = if !result.columns.is_empty() {
        if state.csv_mode {
            display::render_csv(&result.columns, &result.rows)
        } else if state.expanded {
            display::render_expanded(&result.columns, &result.rows)
        } else {
            display::render_table(&result.columns, &result.rows)
        }
    } else if let Some(ref tag) = result.command_tag {
        tag.clone()
    } else {
        String::new()
    };

    if !output.is_empty() {
        write_output(state, &output);
    }

    if state.timing {
        let timing_msg = format!("Time: {:.3} ms", elapsed.as_secs_f64() * 1000.0);
        write_output(state, &timing_msg);
    }
}

/// Writes text to the output file if set, otherwise to stdout.
fn write_output(state: &mut CliState, text: &str) {
    match state.output_file {
        Some(ref mut f) => {
            let _ = writeln!(f, "{}", text);
        }
        None => {
            println!("{}", text);
        }
    }
}

/// Returns true if the buffer contains a complete SQL statement
/// (ends with a semicolon, ignoring trailing whitespace).
fn is_statement_complete(buffer: &str) -> bool {
    buffer.trim_end().ends_with(';')
}

// ---------------------------------------------------------------------------
// Wire protocol helpers (synchronous, using std::net::TcpStream)
// ---------------------------------------------------------------------------

/// Reads exactly one byte from the stream.
fn read_u8(stream: &mut TcpStream) -> io::Result<u8> {
    let mut buf = [0u8; 1];
    stream.read_exact(&mut buf)?;
    Ok(buf[0])
}

/// Reads a big-endian i32 from the stream.
fn read_i32(stream: &mut TcpStream) -> io::Result<i32> {
    let mut buf = [0u8; 4];
    stream.read_exact(&mut buf)?;
    Ok(i32::from_be_bytes(buf))
}

/// Sends a PostgreSQL StartupMessage.
fn send_startup(stream: &mut TcpStream, user: &str, database: &str) -> io::Result<()> {
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

    stream.write_all(&msg)?;
    stream.flush()?;
    Ok(())
}

/// Reads startup response until ReadyForQuery.
fn handle_startup_response(stream: &mut TcpStream) -> io::Result<()> {
    loop {
        let msg_type = read_u8(stream)?;
        let len = read_i32(stream)? as usize;
        let body_len = len.saturating_sub(4);

        let mut body = vec![0u8; body_len];
        if body_len > 0 {
            stream.read_exact(&mut body)?;
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
                // ParameterStatus, ignore
            }
            b'K' => {
                // BackendKeyData, ignore
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
fn send_query(stream: &mut TcpStream, sql: &str) -> io::Result<()> {
    let body_len = sql.len() + 1; // SQL + null terminator
    let len = (body_len + 4) as i32;

    let mut msg = BytesMut::with_capacity(1 + len as usize);
    msg.put_u8(b'Q');
    msg.put_i32(len);
    msg.extend_from_slice(sql.as_bytes());
    msg.put_u8(0);

    stream.write_all(&msg)?;
    stream.flush()?;
    Ok(())
}

/// Reads query response messages (RowDescription, DataRow, CommandComplete,
/// ErrorResponse) until ReadyForQuery. Returns structured result data.
fn read_query_response(stream: &mut TcpStream) -> io::Result<QueryResult> {
    let mut result = QueryResult {
        columns: Vec::new(),
        rows: Vec::new(),
        command_tag: None,
        error: None,
    };

    loop {
        let msg_type = read_u8(stream)?;
        let len = read_i32(stream)? as usize;
        let body_len = len.saturating_sub(4);

        let mut body = vec![0u8; body_len];
        if body_len > 0 {
            stream.read_exact(&mut body)?;
        }

        match msg_type {
            b'T' => {
                // RowDescription
                result.columns = parse_row_description(&body);
            }
            b'D' => {
                // DataRow
                let values = parse_data_row(&body);
                result.rows.push(values);
            }
            b'C' => {
                // CommandComplete
                let tag = String::from_utf8_lossy(&body)
                    .trim_end_matches('\0')
                    .to_string();
                result.command_tag = Some(tag);
            }
            b'E' => {
                // ErrorResponse
                let msg = parse_error_fields(&body);
                result.error = Some(msg);
            }
            b'Z' => {
                // ReadyForQuery
                return Ok(result);
            }
            b'I' => {
                // EmptyQueryResponse
                result.command_tag = Some("(empty query)".to_string());
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
        if pos >= body.len() {
            break;
        }
        // Read null-terminated field name
        let name_end = body[pos..]
            .iter()
            .position(|&b| b == 0)
            .unwrap_or(body.len() - pos);
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

/// Extracts the error message from an ErrorResponse body.
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
