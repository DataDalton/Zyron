//! ZyronDB CLI client.
//!
//! A psql-like client that connects via TCP, sends SQL queries using
//! the PostgreSQL simple query protocol, and displays formatted results.
//! Supports meta-commands, multi-line input, query history, and multiple
//! output formats (table, expanded, CSV).

mod color;
mod command;
mod command_help;
mod command_sql;
mod completion;
mod display;
mod history;

use bytes::{BufMut, BytesMut};
use std::fs::File;
use std::io::{self, Read, Write};
use std::net::TcpStream;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};

use command::{Command, ParseError};
use display::ColumnKind;

/// Transaction state reported by the server's `ReadyForQuery` message.
/// Tracked on `CliState` so the prompt can reflect open/failed transactions
/// (`*=>` / `!=>` suffix) without an extra round-trip.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TxnStatus {
    Idle,
    InTransaction,
    Failed,
}

impl TxnStatus {
    fn from_byte(b: u8) -> Self {
        match b {
            b'T' => TxnStatus::InTransaction,
            b'E' => TxnStatus::Failed,
            _ => TxnStatus::Idle,
        }
    }
}

const DEFAULT_HOST: &str = "127.0.0.1";
const DEFAULT_PORT: u16 = 5432;
const DEFAULT_USER: &str = "zyron";
const DEFAULT_DATABASE: &str = "zyron";
const MAX_HISTORY_ENTRIES: usize = 1000;

#[derive(Debug, Clone)]
struct CliOptions {
    host: String,
    port: u16,
    user: String,
    database: String,
    /// Output format override. Defaults to Table for tty stdout, Json for
    /// pipe/redirect. Always applied to the top-level command output; the
    /// SQL sub-REPL respects the `csv` / `expanded` toggles separately.
    format: OutputFormat,
    /// If set, the CLI dispatches this command and exits instead of
    /// running the interactive REPL. Populated from positional args.
    one_shot_command: Option<String>,
}

/// Output format selection for the top-level command layer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OutputFormat {
    /// Human-readable aligned table (default on tty).
    Table,
    /// CSV with RFC 4180 quoting.
    Csv,
    /// JSON array of objects, one object per row (default when stdout
    /// isn't a tty so pipes into `jq` work out of the box).
    Json,
}

/// Tracks client-side display and input state for the REPL session.
/// Multi-line buffering is local to `sql_sub_repl`; command mode is
/// single-line so there's no shared buffer at this level.
struct CliState {
    timing: bool,
    expanded: bool,
    csv_mode: bool,
    output_file: Option<File>,
    history: Vec<String>,
    /// Current transaction state, refreshed after every server round-trip.
    /// Defaults to Idle before the first query.
    txn_status: TxnStatus,
    /// Connection target, retained so `status` / `conninfo` can reprint
    /// the banner without re-parsing argv.
    conn: CliOptions,
    /// Advertised server version string, captured from the
    /// `ParameterStatus("server_version", ...)` message during startup.
    /// None if the server did not advertise one.
    server_version: Option<String>,
}

impl CliState {
    fn new(history: Vec<String>, conn: CliOptions, server_version: Option<String>) -> Self {
        Self {
            timing: false,
            expanded: false,
            csv_mode: false,
            output_file: None,
            history,
            txn_status: TxnStatus::Idle,
            conn,
            server_version,
        }
    }
}

/// Query result data returned from read_query_response.
struct QueryResult {
    columns: Vec<String>,
    /// Per-column semantic kind (numeric vs other) for display coloring.
    /// Derived from the RowDescription message's type OID field.
    column_kinds: Vec<ColumnKind>,
    rows: Vec<Vec<String>>,
    command_tag: Option<String>,
    error: Option<ErrorDetail>,
    /// Server-reported transaction state from the final `ReadyForQuery`.
    /// Applied to `CliState.txn_status` after each query.
    txn_status: TxnStatus,
}

/// Parsed fields from a server `ErrorResponse`. Enough structure to render
/// a psql-style error block with severity coloring and a position caret.
#[derive(Debug, Clone)]
struct ErrorDetail {
    severity: String,
    message: String,
    /// 1-based character position within the submitted SQL, if the server
    /// provided one. Used by `render_error` to point a caret at the
    /// offending token.
    position: Option<usize>,
    detail: Option<String>,
    hint: Option<String>,
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
        format: default_format_for_tty(),
        one_shot_command: None,
    })
}

/// Picks the default output format based on whether stdout is a terminal.
/// A tty gets the aligned Table view; a pipe/redirect gets JSON so the
/// output is machine-parseable without a flag.
fn default_format_for_tty() -> OutputFormat {
    use std::io::IsTerminal;
    if std::io::stdout().is_terminal() {
        OutputFormat::Table
    } else {
        OutputFormat::Json
    }
}

fn parse_args() -> Option<CliOptions> {
    let args: Vec<String> = std::env::args().collect();
    let mut opts = CliOptions {
        host: DEFAULT_HOST.into(),
        port: DEFAULT_PORT,
        user: DEFAULT_USER.into(),
        database: DEFAULT_DATABASE.into(),
        format: default_format_for_tty(),
        one_shot_command: None,
    };
    let mut i = 1;
    let mut trailing_tokens: Vec<String> = Vec::new();

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
            "--format" => {
                i += 1;
                if i >= args.len() {
                    eprintln!("Missing value for --format (expected table|csv|json)");
                    std::process::exit(1);
                }
                opts.format = match args[i].to_ascii_lowercase().as_str() {
                    "table" => OutputFormat::Table,
                    "csv" => OutputFormat::Csv,
                    "json" => OutputFormat::Json,
                    other => {
                        eprintln!(
                            "Unknown --format value '{}' (expected table, csv, json)",
                            other
                        );
                        std::process::exit(1);
                    }
                };
            }
            "--" => {
                // Everything after `--` is positional, regardless of
                // leading dashes.
                i += 1;
                while i < args.len() {
                    trailing_tokens.push(args[i].clone());
                    i += 1;
                }
            }
            other => {
                if other.starts_with("zyron://") {
                    match parse_connection_string(other) {
                        Ok(parsed) => {
                            // Preserve CLI-level overrides we've already
                            // accumulated (format, positional tokens).
                            let format = opts.format;
                            opts = parsed;
                            opts.format = format;
                        }
                        Err(e) => {
                            eprintln!("Invalid connection string: {}", e);
                            std::process::exit(1);
                        }
                    }
                } else if other.starts_with('-') {
                    eprintln!("Unknown argument: {}", other);
                    std::process::exit(1);
                } else {
                    // Positional token: collect as part of the one-shot
                    // command so users can run `zyrondb-cli status` or
                    // `zyrondb-cli tables describe users` directly.
                    trailing_tokens.push(args[i].clone());
                }
            }
        }
        i += 1;
    }

    if !trailing_tokens.is_empty() {
        opts.one_shot_command = Some(trailing_tokens.join(" "));
    }
    Some(opts)
}

fn print_help() {
    println!(
        "Usage: zyrondb-cli [OPTIONS] [CONNECTION_STRING] [COMMAND...]

With no trailing COMMAND, zyrondb-cli starts the interactive REPL.
Otherwise it dispatches the given command one-shot and exits:
  zyrondb-cli status
  zyrondb-cli tables describe users
  zyrondb-cli sql \"SELECT * FROM orders LIMIT 5\"

Options:
  -h, --host <addr>      Server host (default: 127.0.0.1)
  -p, --port <number>    Server port (default: 5432)
  -U, --user <name>      Username (default: zyron)
  -d, --database <db>    Database name (default: zyron)
  --format <fmt>         Output format for one-shot commands and non-tty
                         stdout: table | csv | json. Defaults to table
                         on a tty, json when piped.
  --version              Print version
  --help                 Print this help

Connection string format:
  zyron://user:pass@host:port/database

Top-level commands (type `help` at the prompt for the full list):
  help [command]           Show help for all or one command
  status                   Show connection and server state
  tables                   List tables; `tables describe <name>` for columns
  indexes                  List or manage indexes
  users                    List or manage users and privileges
  stats [view]             Show server counters; views: wal, tables, buffer...
  sql                      Enter the SQL sub-REPL, or `sql \"<stmt>\"` for one-shot
  watch [secs] <command>   Refresh any command on an interval until Ctrl-C
  source <file>            Run commands from a file
  exit                     Quit"
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

    // Read authentication and parameter status messages until ReadyForQuery.
    // The server's `server_version` ParameterStatus is retained for the
    // welcome banner and for `\conninfo`.
    let server_version = match handle_startup_response(&mut stream) {
        Ok(v) => v,
        Err(e) => {
            eprintln!("Authentication failed: {}", e);
            std::process::exit(1);
        }
    };

    // Load history. Rustyline reads the file directly so we don't need the
    // custom loader for its internal buffer, but we still use the path helper
    // so the location matches what the existing installation wrote.
    let hist_path = history::history_path();
    let hist = history::load_history(&hist_path);
    let mut state = CliState::new(hist, opts.clone(), server_version.clone());

    // One-shot execution: when the shell invocation carried a positional
    // command (e.g. `zyrondb-cli tables list`), dispatch it and exit
    // without entering the REPL. The welcome banner is suppressed so
    // pipes stay clean.
    if let Some(cmd_str) = opts.one_shot_command.clone() {
        let exit_code = run_one_shot(&cmd_str, &mut state, &mut stream);
        send_terminate(&mut stream);
        std::process::exit(exit_code);
    }

    print_welcome_banner(&state);

    // Build the command-mode line editor. SQL sub-mode uses a separately
    // constructed editor with a SQL-keyword helper; keeping the two
    // independent lets each carry its own completion vocabulary and
    // history view without bleed-through.
    let rl_config = rustyline::Config::builder().auto_add_history(false).build();
    let mut editor = match rustyline::Editor::<
        completion::CommandHelper,
        rustyline::history::DefaultHistory,
    >::with_config(rl_config)
    {
        Ok(e) => e,
        Err(err) => {
            eprintln!("Failed to initialize line editor: {}", err);
            std::process::exit(1);
        }
    };
    editor.set_helper(Some(completion::CommandHelper));
    if hist_path.exists() {
        let _ = editor.load_history(&hist_path);
    }

    loop {
        let prompt = build_command_prompt(&state);
        let line = match editor.readline(&prompt) {
            Ok(s) => s,
            Err(rustyline::error::ReadlineError::Interrupted) => {
                // Ctrl-C: cancel this input and re-prompt.
                continue;
            }
            Err(rustyline::error::ReadlineError::Eof) => break,
            Err(err) => {
                eprintln!("Input error: {}", err);
                break;
            }
        };

        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }

        let _ = editor.add_history_entry(trimmed);
        history::add_to_history(&mut state.history, trimmed);

        match Command::parse(trimmed) {
            Ok(Command::Exit) => break,
            Ok(cmd) => {
                handle_command(cmd, &mut state, &mut stream);
            }
            Err(ParseError::Empty) => {}
            Err(err) => {
                print_parse_error(&err);
            }
        }
    }

    // Save history on exit. Rustyline persists in-session line-edits;
    // our history module keeps the raw submitted strings for forward
    // compatibility with any future tool that reads the file directly.
    let _ = editor.save_history(&hist_path);
    let _ = history::save_history(&hist_path, &state.history, MAX_HISTORY_ENTRIES);

    // Send Terminate message
    let mut buf = BytesMut::with_capacity(5);
    buf.put_u8(b'X');
    buf.put_i32(4);
    let _ = stream.write_all(&buf);

    println!("Disconnected.");
}

/// Dispatches a parsed `Command`. Local actions (help, toggles, clear,
/// sub-mode entry) are handled directly; commands that need the server
/// are routed through `command_sql::command_to_sql` and sent over the
/// wire. `Exit` is handled by the REPL loop; this function is never
/// called with it.
fn handle_command(cmd: Command, state: &mut CliState, stream: &mut TcpStream) {
    match cmd {
        // -- Local actions --
        Command::Help { path } => {
            if path.is_empty() {
                command_help::print_index();
            } else if !command_help::print_command(&path[0]) {
                eprintln!("{}: {}", color::bold_red("no help for"), path.join(" "),);
            }
        }
        Command::Status => print_welcome_banner(state),
        Command::Version => print_version_line(state),
        Command::Clear => {
            if color::enabled() {
                print!("\x1b[2J\x1b[H");
            } else {
                print!("\x0c");
            }
            let _ = io::stdout().flush();
        }
        Command::Sql(None) => sql_sub_repl(state, stream),
        Command::Sql(Some(sql)) => execute_and_display(&sql, state, stream),
        Command::Watch {
            interval_secs,
            inner,
        } => {
            run_watch_command(interval_secs, &inner, state, stream);
        }
        Command::Source { path } => execute_file(&path, state, stream),
        Command::Timing => {
            state.timing = !state.timing;
            println!(
                "Timing is {}.",
                color::bold_cyan(if state.timing { "on" } else { "off" })
            );
        }
        Command::Expanded => {
            state.expanded = !state.expanded;
            println!(
                "Expanded display is {}.",
                color::bold_cyan(if state.expanded { "on" } else { "off" })
            );
        }
        Command::Csv => {
            state.csv_mode = !state.csv_mode;
            println!(
                "CSV output is {}.",
                color::bold_cyan(if state.csv_mode { "on" } else { "off" })
            );
        }
        Command::OutputFile(path) => match path {
            Some(p) => match File::create(&p) {
                Ok(f) => {
                    state.output_file = Some(f);
                    println!("Output redirected to {}", color::bold_cyan(&p));
                }
                Err(e) => {
                    eprintln!("{}: {}: {}", color::bold_red("open failed"), p, e);
                }
            },
            None => {
                state.output_file = None;
                println!("Output reset to stdout.");
            }
        },

        // -- Server-bound commands translate to SQL --
        other => {
            if let Some(sql) = command_sql::command_to_sql(&other) {
                execute_and_display(&sql, state, stream);
            }
        }
    }
}

/// SQL sub-REPL. Constructed with its own rustyline editor and
/// `SqlHelper` so tab completion and keyword highlighting target the
/// parser vocabulary instead of the command vocabulary. Exits on
/// `exit`, `quit`, `:q`, or Ctrl-D. Ctrl-C discards the current
/// multi-line buffer but stays in the sub-REPL.
fn sql_sub_repl(state: &mut CliState, stream: &mut TcpStream) {
    let config = rustyline::Config::builder().auto_add_history(false).build();
    let mut editor = match rustyline::Editor::<
        completion::SqlHelper,
        rustyline::history::DefaultHistory,
    >::with_config(config)
    {
        Ok(e) => e,
        Err(err) => {
            eprintln!("{}: {}", color::bold_red("sql sub-editor failed"), err);
            return;
        }
    };
    editor.set_helper(Some(completion::SqlHelper));

    println!(
        "{}  {}  {}",
        color::bold_cyan("sql mode"),
        color::dim("terminate statements with ;"),
        color::dim("type 'exit' or Ctrl-D to return"),
    );

    let mut buffer = String::new();
    loop {
        let prompt = build_sql_prompt(state, buffer.is_empty());
        let line = match editor.readline(&prompt) {
            Ok(l) => l,
            Err(rustyline::error::ReadlineError::Interrupted) => {
                if !buffer.is_empty() {
                    buffer.clear();
                    println!("{}", color::dim("(input buffer cleared)"));
                }
                continue;
            }
            Err(rustyline::error::ReadlineError::Eof) => break,
            Err(err) => {
                eprintln!("{}: {}", color::bold_red("input error"), err);
                break;
            }
        };

        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }

        // Sub-mode exit keywords only trigger on a bare line (no prior
        // buffer content) so SQL identifiers happening to spell "exit"
        // don't trap the user.
        if buffer.is_empty() {
            let low = trimmed.trim_end_matches(';').trim().to_ascii_lowercase();
            if matches!(low.as_str(), "exit" | "quit" | ":q" | ":wq" | ":x" | "back") {
                break;
            }
        }

        let _ = editor.add_history_entry(trimmed);

        if !buffer.is_empty() {
            buffer.push(' ');
        }
        buffer.push_str(trimmed);

        if is_statement_complete(&buffer) {
            let query = std::mem::take(&mut buffer);
            history::add_to_history(&mut state.history, &query);
            execute_and_display(&query, state, stream);
        }
    }
}

/// Dispatches a single command from the OS shell (`zyrondb-cli tables
/// list`) and returns the process exit code. Suppresses the welcome
/// banner, the prompt, the line editor; the one-shot path is strictly
/// for scripting. Returns 0 on success, 2 on parse error, 1 on dispatch
/// error (e.g. attempt to enter interactive sub-modes).
fn run_one_shot(input: &str, state: &mut CliState, stream: &mut TcpStream) -> i32 {
    match Command::parse(input) {
        Ok(Command::Exit) => 0,
        Ok(Command::Sql(None)) | Ok(Command::Watch { .. }) | Ok(Command::Help { path: _ }) => {
            // Interactive-only: the banner, the sub-REPL, and the
            // watch refresh loop all assume a tty; also `help`
            // printing to a pipe is rarely what the caller wants.
            // Route help through anyway (useful for `| grep`) but
            // reject the others.
            if matches!(
                input.trim().split_whitespace().next(),
                Some("help") | Some("?") | Some("h")
            ) {
                handle_command(Command::parse(input).unwrap(), state, stream);
                0
            } else {
                eprintln!(
                    "{}: {} is interactive-only and not supported in one-shot mode",
                    color::bold_red("error"),
                    input.split_whitespace().next().unwrap_or(input),
                );
                1
            }
        }
        Ok(cmd) => {
            handle_command(cmd, state, stream);
            0
        }
        Err(err) => {
            print_parse_error(&err);
            2
        }
    }
}

/// Sends the PG `Terminate` ('X') message on the wire so the server can
/// cleanly close the session. Used by both the REPL exit path and the
/// one-shot exit path.
fn send_terminate(stream: &mut TcpStream) {
    let mut buf = BytesMut::with_capacity(5);
    buf.put_u8(b'X');
    buf.put_i32(4);
    let _ = stream.write_all(&buf);
}

/// Encodes a string as a JSON-escaped fragment including the surrounding
/// quotes. Used by command-tag rendering in JSON output mode.
fn json_string(s: &str) -> String {
    let mut out = String::with_capacity(s.len() + 2);
    out.push('"');
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if (c as u32) < 0x20 => out.push_str(&format!("\\u{:04x}", c as u32)),
            c => out.push(c),
        }
    }
    out.push('"');
    out
}

/// Prints an elided connection + version summary line, matching the
/// `version` command's output. Separate from the banner so `version`
/// doesn't reprint the logo every time.
fn print_version_line(state: &CliState) {
    println!(
        "{}  {}  ·  server {}",
        color::bold("zyrondb-cli"),
        color::dim(env!("CARGO_PKG_VERSION")),
        state.server_version.as_deref().unwrap_or("unknown"),
    );
}

/// Prints a parse error with color and a typo suggestion when the error
/// is an unknown top-level command.
fn print_parse_error(err: &ParseError) {
    match err {
        ParseError::UnknownCommand(name) => {
            eprint!("{}: {}", color::bold_red("unknown command"), name,);
            if let Some(suggestion) = command::closest_command(name) {
                eprint!(
                    "  {} {}",
                    color::dim("did you mean"),
                    color::bold_cyan(suggestion),
                );
                eprint!("{}", color::dim("?"));
            }
            eprintln!();
            eprintln!("{}", color::dim("Type `help` to see all commands."));
        }
        ParseError::MissingArg { command, expected } => {
            eprintln!(
                "{}: {} requires {}",
                color::bold_red("error"),
                command,
                expected,
            );
        }
        ParseError::BadSyntax { command, detail } => {
            eprintln!("{}: {}: {}", color::bold_red("error"), command, detail,);
        }
        ParseError::Empty => {}
    }
}

/// Runs a command on a timer until Ctrl-C. The command is re-parsed each
/// tick so dynamic interval args inside the inner command still work.
fn run_watch_command(
    interval_secs: u64,
    inner: &[String],
    state: &mut CliState,
    stream: &mut TcpStream,
) {
    let stop = watch_interrupt_flag();
    stop.store(false, Ordering::Release);

    let inner_str = inner.join(" ");
    let interval = Duration::from_secs(interval_secs);
    while !stop.load(Ordering::Acquire) {
        if color::enabled() {
            print!("\x1b[2J\x1b[H");
            let _ = io::stdout().flush();
        } else {
            println!();
        }
        println!(
            "{} {}  interval {}s  {}",
            color::bold_cyan("watch"),
            color::dim(&current_local_time_string()),
            interval_secs,
            color::dim("(Ctrl-C to stop)"),
        );
        println!("{}  {}", color::dim("→"), color::bold(&inner_str));
        println!();

        match Command::parse(&inner_str) {
            Ok(Command::Exit) | Ok(Command::Sql(None)) | Ok(Command::Watch { .. }) => {
                eprintln!(
                    "{}",
                    color::bold_red("watch cannot re-run interactive commands")
                );
                break;
            }
            Ok(cmd) => handle_command(cmd, state, stream),
            Err(e) => {
                print_parse_error(&e);
                break;
            }
        }

        let sleep_slice = Duration::from_millis(100);
        let mut remaining = interval;
        while remaining > Duration::ZERO && !stop.load(Ordering::Acquire) {
            let step = remaining.min(sleep_slice);
            std::thread::sleep(step);
            remaining = remaining.saturating_sub(step);
        }
    }
    println!();
}

/// Repeatedly runs `sql` every `interval_secs` seconds until the user
/// presses Ctrl-C. A one-shot ctrlc handler flips a shared atomic flag;
/// the handler is installed only once for the process via `OnceLock` so
/// repeated `\watch` invocations don't pile up handlers. Between each
/// iteration the screen is cleared so the output behaves like a
/// self-refreshing dashboard.
/// Shared interrupt flag raised by the ctrlc handler. Installed lazily so
/// CLI sessions that never use `watch` pay no cost for signal handling.
fn watch_interrupt_flag() -> &'static AtomicBool {
    use std::sync::OnceLock;
    static FLAG: OnceLock<&'static AtomicBool> = OnceLock::new();
    FLAG.get_or_init(|| {
        let flag: &'static AtomicBool = Box::leak(Box::new(AtomicBool::new(false)));
        let f = flag;
        // Best-effort handler install. If another signal handler is already
        // in place we still let the REPL run without Ctrl-C support for
        // `\watch` rather than aborting.
        let _ = ctrlc::set_handler(move || {
            f.store(true, Ordering::Release);
        });
        flag
    })
}

/// Returns a compact `HH:MM:SS` string using only `std::time`. Good enough
/// for a dashboard timestamp; avoids pulling in `chrono` or `time` for
/// this single use.
fn current_local_time_string() -> String {
    use std::time::{SystemTime, UNIX_EPOCH};
    let now = SystemTime::now();
    let secs = now
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    let secs_today = secs % 86_400;
    let h = secs_today / 3600;
    let m = (secs_today % 3600) / 60;
    let s = secs_today % 60;
    format!("{:02}:{:02}:{:02} UTC", h, m, s)
}

/// Executes commands from a file, one per non-blank, non-comment line.
/// Each line is parsed as a `Command` and dispatched as if the user had
/// typed it at the prompt. Comment conventions: `#` for shell-style,
/// `--` for SQL-style. Blank lines are skipped.
fn execute_file(path: &str, state: &mut CliState, stream: &mut TcpStream) {
    let contents = match std::fs::read_to_string(path) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("{}: {}: {}", color::bold_red("read failed"), path, e);
            return;
        }
    };

    for (line_idx, raw_line) in contents.lines().enumerate() {
        let line = raw_line.trim();
        if line.is_empty() || line.starts_with('#') || line.starts_with("--") {
            continue;
        }
        match Command::parse(line) {
            Ok(Command::Exit) => return,
            Ok(cmd) => handle_command(cmd, state, stream),
            Err(ParseError::Empty) => {}
            Err(err) => {
                eprintln!(
                    "{} {}: {}",
                    color::bold_red(&format!("{}:{}:", path, line_idx + 1)),
                    color::dim(line),
                    err,
                );
            }
        }
    }
}

/// Sends a query to the server, reads the response, and displays formatted output.
/// Spawns a lightweight elapsed-timer thread so long-running queries show a
/// live "Running... 2.4s" line instead of appearing to hang. The timer is
/// cleared and its line erased just before the result prints.
fn execute_and_display(sql: &str, state: &mut CliState, stream: &mut TcpStream) {
    let start = Instant::now();

    if let Err(e) = send_query(stream, sql) {
        eprintln!("{}: {}", color::bold_red("send error"), e);
        return;
    }

    let timer_stop = Arc::new(AtomicBool::new(false));
    let timer_handle = spawn_elapsed_timer(Arc::clone(&timer_stop), start);

    let result = match read_query_response(stream) {
        Ok(r) => r,
        Err(e) => {
            timer_stop.store(true, Ordering::Release);
            if let Some(h) = timer_handle {
                let _ = h.join();
            }
            eprintln!("{}: {}", color::bold_red("query error"), e);
            return;
        }
    };

    timer_stop.store(true, Ordering::Release);
    if let Some(h) = timer_handle {
        let _ = h.join();
    }

    let elapsed = start.elapsed();

    // Keep the CLI's view of the transaction in sync with the server so the
    // next prompt reflects BEGIN/COMMIT/ROLLBACK without a second round-trip.
    state.txn_status = result.txn_status;

    // Colorized error block before any data output.
    if let Some(ref err) = result.error {
        eprintln!("{}", render_error(err, sql));
    }

    // Format the output. Precedence: explicit sub-mode toggles (`csv`,
    // `expanded`) win over the global `--format` flag so interactive
    // overrides keep behaving the way users expect. When neither toggle
    // is set, fall back to the format picked at startup (Table for tty,
    // Json for pipe).
    let output = if !result.columns.is_empty() {
        if state.csv_mode {
            display::render_csv(&result.columns, &result.rows)
        } else if state.expanded {
            display::render_expanded(&result.columns, &result.rows, &result.column_kinds)
        } else {
            match state.conn.format {
                OutputFormat::Csv => display::render_csv(&result.columns, &result.rows),
                OutputFormat::Json => {
                    display::render_json(&result.columns, &result.rows, &result.column_kinds)
                }
                OutputFormat::Table => {
                    display::render_table(&result.columns, &result.rows, &result.column_kinds)
                }
            }
        }
    } else if let Some(ref tag) = result.command_tag {
        match state.conn.format {
            OutputFormat::Json => format!("{{\"command\":{}}}", json_string(tag)),
            _ => color::bold_green(tag),
        }
    } else {
        String::new()
    };

    if !output.is_empty() {
        write_output(state, &output);
    }

    if state.timing {
        let timing_msg = color::dim(&format!("Time: {:.3} ms", elapsed.as_secs_f64() * 1000.0));
        write_output(state, &timing_msg);
    }
}

/// Spawns a background thread that reprints an elapsed-time line while
/// the main thread blocks on the wire read. Stops when the atomic flag is
/// set, clears its own line so the result output lands at column 0.
///
/// Returns `None` when colors are disabled (non-tty stdout): no animation
/// would render correctly, so we skip the thread entirely.
fn spawn_elapsed_timer(
    stop: Arc<AtomicBool>,
    start: Instant,
) -> Option<std::thread::JoinHandle<()>> {
    if !color::enabled() {
        return None;
    }
    let handle = std::thread::spawn(move || {
        // Give fast queries a chance to finish before the spinner shows up.
        // This avoids flashing the line for sub-200ms queries.
        let warmup = Duration::from_millis(250);
        std::thread::sleep(warmup);
        if stop.load(Ordering::Acquire) {
            return;
        }
        let spinner = ['|', '/', '-', '\\'];
        let mut tick: usize = 0;
        let mut rendered = false;
        while !stop.load(Ordering::Acquire) {
            let elapsed = start.elapsed().as_secs_f64();
            let line = color::dim(&format!(
                "{} running... {:.1}s",
                spinner[tick % spinner.len()],
                elapsed,
            ));
            print!("\r{}\x1b[K", line);
            let _ = io::stdout().flush();
            rendered = true;
            tick += 1;
            std::thread::sleep(Duration::from_millis(100));
        }
        if rendered {
            // Clear the spinner line so result output is not appended to it.
            print!("\r\x1b[K");
            let _ = io::stdout().flush();
        }
    });
    Some(handle)
}

/// Prompt for the top-level command mode: `zyron>` (no transaction
/// marker — command mode is auto-commit per command, not under user
/// transaction control).
fn build_command_prompt(_state: &CliState) -> String {
    format!("{} ", color::bold_blue("zyron>"))
}

/// Prompt inside the SQL sub-REPL. Shows the database name, transaction
/// marker (`*=>` / `!=>`), and a `->` continuation prompt when a
/// multi-line statement is in progress. `first_line` is `true` when the
/// input buffer is empty (so we draw the full `db=>` form) and `false`
/// for continuation lines.
fn build_sql_prompt(state: &CliState, first_line: bool) -> String {
    if !first_line {
        return format!(
            "{} ",
            color::bold_blue(&format!("{}->", state.conn.database))
        );
    }
    let (marker, db_colored) = match state.txn_status {
        TxnStatus::Idle => ("=>".to_string(), color::bold_blue(&state.conn.database)),
        TxnStatus::InTransaction => (
            format!("{}{}", color::bold_yellow("*"), "=>"),
            color::bold_blue(&state.conn.database),
        ),
        TxnStatus::Failed => (
            format!("{}{}", color::bold_red("!"), "=>"),
            color::bold_red(&state.conn.database),
        ),
    };
    format!("{}{} ", db_colored, marker)
}

/// Renders a structured `ErrorDetail` in psql style:
/// `ERROR:  <message>` on the first line, optional `DETAIL` / `HINT` lines
/// below, and a two-line query-with-caret block when the server reported
/// a character position.
fn render_error(err: &ErrorDetail, sql: &str) -> String {
    let mut out = String::new();
    out.push_str(&color::bold_red(&format!("{}:", err.severity)));
    out.push(' ');
    out.push_str(&err.message);

    if let Some(pos) = err.position {
        if let Some(caret) = render_caret(sql, pos) {
            out.push('\n');
            out.push_str(&caret);
        }
    }
    if let Some(ref detail) = err.detail {
        out.push('\n');
        out.push_str(&color::bold(&format!("{}: ", "DETAIL")));
        out.push_str(detail);
    }
    if let Some(ref hint) = err.hint {
        out.push('\n');
        out.push_str(&color::bold_cyan(&format!("{}: ", "HINT")));
        out.push_str(hint);
    }
    out
}

/// Given the original SQL and a 1-based character position, returns the
/// offending line followed by a `^`-caret line pointing at the position.
/// Multiline statements are handled by finding the line containing the
/// position and indenting the caret by the in-line column count. Returns
/// `None` when the position falls outside the string.
fn render_caret(sql: &str, position_1based: usize) -> Option<String> {
    if position_1based == 0 || position_1based > sql.len() + 1 {
        return None;
    }
    let idx = position_1based - 1;
    // Locate the start of the line containing `idx`.
    let line_start = sql[..idx].rfind('\n').map(|p| p + 1).unwrap_or(0);
    let line_end = sql[line_start..]
        .find('\n')
        .map(|p| line_start + p)
        .unwrap_or(sql.len());
    let line = &sql[line_start..line_end];
    let col = idx - line_start;
    let caret_padding = " ".repeat(col);
    let mut out = String::new();
    out.push_str(&color::dim("LINE: "));
    out.push_str(line);
    out.push('\n');
    out.push_str(&color::dim("      "));
    out.push_str(&caret_padding);
    out.push_str(&color::bold_red("^"));
    Some(out)
}

/// Prints the one-time welcome banner with the ASCII logo and connection
/// summary. Called after startup handshake succeeds but before the editor
/// starts reading user input.
fn print_welcome_banner(state: &CliState) {
    // Logo in bold blue, block by block so the ANSI codes don't clip across
    // lines in terminals that reset color at every newline.
    for logo_line in color::ZYRON_LOGO.lines() {
        println!("{}", color::bold_blue(logo_line));
    }
    println!();

    let client_line = format!(
        "{}  {}  {}  server {}",
        color::bold("zyrondb-cli"),
        color::dim(env!("CARGO_PKG_VERSION")),
        color::dim("·"),
        state.server_version.as_deref().unwrap_or("unknown"),
    );
    println!("{}", client_line);

    println!(
        "Connected as {} @ {}:{}/{}",
        color::bold_cyan(&state.conn.user),
        state.conn.host,
        state.conn.port,
        color::bold_cyan(&state.conn.database),
    );
    println!(
        "Type {} for help, {} to quit.\n",
        color::bold_cyan("\\?"),
        color::bold_cyan("\\q"),
    );
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

/// Reads startup response until ReadyForQuery. Returns the server version
/// captured from `ParameterStatus("server_version", ...)` when present.
fn handle_startup_response(stream: &mut TcpStream) -> io::Result<Option<String>> {
    let mut server_version: Option<String> = None;
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
                // ParameterStatus: cstring name + cstring value. Capture
                // server_version so the banner can display it.
                if let Some((name, value)) = parse_cstring_pair(&body) {
                    if name == "server_version" {
                        server_version = Some(value);
                    }
                }
            }
            b'K' => {
                // BackendKeyData, ignore
            }
            b'Z' => {
                // ReadyForQuery
                return Ok(server_version);
            }
            b'E' => {
                let detail = parse_error_fields(&body);
                return Err(io::Error::new(io::ErrorKind::Other, detail.message));
            }
            _ => {
                // Unknown, skip
            }
        }
    }
}

/// Parses two consecutive null-terminated strings from `body`. Returns
/// `None` if either is missing.
fn parse_cstring_pair(body: &[u8]) -> Option<(String, String)> {
    let name_end = body.iter().position(|&b| b == 0)?;
    let name = String::from_utf8_lossy(&body[..name_end]).to_string();
    let rest = &body[name_end + 1..];
    let value_end = rest.iter().position(|&b| b == 0)?;
    let value = String::from_utf8_lossy(&rest[..value_end]).to_string();
    Some((name, value))
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
/// ErrorResponse) until ReadyForQuery. Returns structured result data
/// including per-column type info and the final transaction state.
fn read_query_response(stream: &mut TcpStream) -> io::Result<QueryResult> {
    let mut result = QueryResult {
        columns: Vec::new(),
        column_kinds: Vec::new(),
        rows: Vec::new(),
        command_tag: None,
        error: None,
        txn_status: TxnStatus::Idle,
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
                let (names, kinds) = parse_row_description(&body);
                result.columns = names;
                result.column_kinds = kinds;
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
                result.error = Some(parse_error_fields(&body));
            }
            b'Z' => {
                // ReadyForQuery: last byte is the transaction status char.
                if let Some(&b) = body.first() {
                    result.txn_status = TxnStatus::from_byte(b);
                }
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

/// Parses a RowDescription message body into column names paired with the
/// per-column kind used for display coloring. Each field descriptor is
/// `cstring name + int32 table_oid + int16 col_attr + int32 type_oid + ...`;
/// we read the type OID to decide numeric-vs-other.
fn parse_row_description(body: &[u8]) -> (Vec<String>, Vec<ColumnKind>) {
    if body.len() < 2 {
        return (Vec::new(), Vec::new());
    }
    let field_count = i16::from_be_bytes([body[0], body[1]]) as usize;
    let mut names = Vec::with_capacity(field_count);
    let mut kinds = Vec::with_capacity(field_count);
    let mut pos = 2;

    for _ in 0..field_count {
        if pos >= body.len() {
            break;
        }
        // Read null-terminated field name.
        let name_end = body[pos..]
            .iter()
            .position(|&b| b == 0)
            .unwrap_or(body.len() - pos);
        let name = String::from_utf8_lossy(&body[pos..pos + name_end]).to_string();
        names.push(name);
        pos += name_end + 1; // skip null terminator

        // Field descriptor remainder: table_oid(4) + col_attr(2) +
        // type_oid(4) + type_size(2) + type_mod(4) + format(2) = 18 bytes.
        // Extract type_oid at offset +6 from `pos` so display can color
        // numeric columns.
        let type_oid = if pos + 10 <= body.len() {
            i32::from_be_bytes([body[pos + 6], body[pos + 7], body[pos + 8], body[pos + 9]])
        } else {
            0
        };
        kinds.push(display::classify_type_oid(type_oid));
        pos += 18;
    }

    (names, kinds)
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

/// Parses an `ErrorResponse` body into structured severity, message,
/// position, detail, and hint fields. Fields follow the PG wire format:
/// `(byte field_type, cstring value)+` terminated by a `0` byte.
fn parse_error_fields(body: &[u8]) -> ErrorDetail {
    let mut severity = String::new();
    let mut message = String::new();
    let mut position: Option<usize> = None;
    let mut detail: Option<String> = None;
    let mut hint: Option<String> = None;
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
        let value = String::from_utf8_lossy(&body[pos..pos + str_end]).to_string();
        pos += str_end + 1;

        match field_type {
            b'S' | b'V' => {
                // 'S' is severity (localized); 'V' is severity (unlocalized).
                // Prefer V when present but fall back to S.
                if severity.is_empty() || field_type == b'V' {
                    severity = value;
                }
            }
            b'M' => message = value,
            b'P' => {
                if let Ok(n) = value.parse::<usize>() {
                    position = Some(n);
                }
            }
            b'D' => detail = Some(value),
            b'H' => hint = Some(value),
            _ => {}
        }
    }

    if severity.is_empty() {
        severity = "ERROR".to_string();
    }
    if message.is_empty() {
        message = "Unknown error".to_string();
    }

    ErrorDetail {
        severity,
        message,
        position,
        detail,
        hint,
    }
}
