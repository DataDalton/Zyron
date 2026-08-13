//! Command-first CLI grammar and dispatcher.
//!
//! The top-level REPL interprets plain nouns ("tables", "users", "stats")
//! rather than SQL. SQL is available as an explicit sub-mode or one-shot:
//! `sql` opens a persistent SQL REPL, `sql "SELECT ..."` runs one query.
//!
//! Every command that needs server interaction compiles to SQL against
//! either regular statements (GRANT/REVOKE/DROP/TRUNCATE/VACUUM) or stat
//! views (`SELECT ... FROM zyron_stat_*`). This keeps server-side
//! compatibility with any PG-protocol client and lets the server-side
//! privilege machinery enforce access identically for CLI and GUI clients.
//!
//! Parsing is whitespace-tokenized with double-quote grouping. Good enough
//! for admin commands; SQL is never parsed here, only passed through to
//! `sql "..."` one-shot mode.

use std::fmt;

// ---------------------------------------------------------------------------
// Command enum
// ---------------------------------------------------------------------------

/// Single top-level command the CLI recognizes. Parsed from a user input
/// line by `Command::parse`; dispatched by `main::handle_command`.
#[derive(Debug, Clone)]
pub enum Command {
    /// Prints either the top-level command index or, if `path` is given,
    /// focused help for the named command or noun.
    Help { path: Vec<String> },
    /// Short banner of connection + server status + active toggles.
    Status,
    /// Client and server version strings on one line.
    Version,
    /// Tables subcommands: list, describe, drop, truncate.
    Tables(TablesAction),
    /// Indexes subcommands: list, describe, drop, reindex.
    Indexes(IndexesAction),
    /// Users subcommands: list, create, drop, grant, revoke.
    Users(UsersAction),
    /// Lists all schemas in the current database.
    Schemas,
    /// Lists all databases on the server.
    Databases,
    /// Stats subcommands: overall summary or a named view.
    Stats(Option<StatsView>),
    /// Compacts and rewrites a table to reclaim dead-tuple space.
    Vacuum { table: String },
    /// Refreshes optimizer statistics for a table.
    Analyze { table: String },
    /// Forces a WAL checkpoint immediately rather than waiting for the
    /// background trigger.
    Checkpoint,
    /// SQL escape. `None` enters persistent sub-mode; `Some(sql)` runs
    /// the given query once and returns to command mode.
    Sql(Option<String>),
    /// Repeatedly runs the given command every `interval_secs` seconds
    /// until Ctrl-C. Argument list is the full command to re-dispatch.
    Watch {
        interval_secs: u64,
        inner: Vec<String>,
    },
    /// Reads a file and dispatches each non-blank, non-comment line as a
    /// command. Useful for scripting admin runbooks.
    Source { path: String },
    /// Clears the terminal screen.
    Clear,
    /// Leaves the REPL.
    Exit,
    /// Toggles client-side query-timing display in the SQL sub-mode.
    Timing,
    /// Toggles the expanded `-[ RECORD N ]-` result format.
    Expanded,
    /// Toggles CSV output for the next SQL query.
    Csv,
    /// Redirects output to a file or resets to stdout.
    OutputFile(Option<String>),

    // -----------------------------------------------------------------------
    // Ops surface
    // -----------------------------------------------------------------------
    /// WAL introspection and maintenance.
    Wal(WalAction),
    /// Replication-slot management.
    Slots(SlotsAction),
    /// CDC streams, feeds, ingests, publications.
    Cdc(CdcAction),
    /// Server configuration: show / set / reload.
    Config(ConfigAction),
    /// Active session introspection.
    Sessions(SessionsAction),
    /// Table archive / restore (ARCHIVE TABLE / RESTORE TABLE SQL).
    Archive {
        table: String,
        where_clause: String,
        to_path: String,
    },
    Restore {
        table: String,
        from_path: String,
        into_target: Option<String>,
    },
    /// Version branch introspection.
    Branches,
    /// Trigger introspection.
    Triggers,
    /// Streaming job introspection.
    Jobs,
    /// Buffer pool / bgwriter counters.
    Buffer,
}

#[derive(Debug, Clone)]
pub enum WalAction {
    /// `SELECT * FROM zyron_stat_wal`.
    Status,
}

#[derive(Debug, Clone)]
pub enum SlotsAction {
    List,
    Create { name: String },
    Drop { name: String },
}

#[derive(Debug, Clone)]
pub enum CdcAction {
    /// Lists outbound CDC streams.
    Streams,
    /// Lists CDC feeds (per-table change-feed registry).
    Feeds,
    /// Lists inbound CDC ingestion jobs.
    Ingests,
    /// `DROP CDC STREAM <name>`.
    DropStream { name: String },
}

#[derive(Debug, Clone)]
pub enum ConfigAction {
    /// `SHOW ALL` or `SHOW <key>` when `key` is set.
    Show { key: Option<String> },
    /// `ALTER SYSTEM SET <key> = <value>`.
    Set { key: String, value: String },
}

#[derive(Debug, Clone)]
pub enum SessionsAction {
    /// `SELECT * FROM zyron_stat_activity`.
    List,
}

#[derive(Debug, Clone)]
pub enum TablesAction {
    List,
    Describe(String),
    Drop(String),
    Truncate(String),
}

#[derive(Debug, Clone)]
pub enum IndexesAction {
    List,
    Describe(String),
    Drop(String),
    Reindex(String),
}

#[derive(Debug, Clone)]
pub enum UsersAction {
    List,
    Create(String),
    Drop(String),
    /// GRANT <privilege> ON <object> TO <user>.
    Grant {
        privilege: String,
        object: String,
        user: String,
    },
    /// REVOKE <privilege> ON <object> FROM <user>.
    Revoke {
        privilege: String,
        object: String,
        user: String,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StatsView {
    Summary,
    Wal,
    Tables,
    Indexes,
    Buffer,
    Connections,
}

impl fmt::Display for StatsView {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            StatsView::Summary => "summary",
            StatsView::Wal => "wal",
            StatsView::Tables => "tables",
            StatsView::Indexes => "indexes",
            StatsView::Buffer => "buffer",
            StatsView::Connections => "connections",
        })
    }
}

// ---------------------------------------------------------------------------
// Parse errors
// ---------------------------------------------------------------------------

/// Represents a user-visible parse failure. `UnknownCommand` carries the
/// offending token so the dispatcher can compute fuzzy suggestions via
/// `ALL_COMMAND_NAMES`.
#[derive(Debug, Clone)]
pub enum ParseError {
    Empty,
    UnknownCommand(String),
    MissingArg {
        command: &'static str,
        expected: &'static str,
    },
    BadSyntax {
        command: &'static str,
        detail: String,
    },
}

impl fmt::Display for ParseError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ParseError::Empty => f.write_str("empty input"),
            ParseError::UnknownCommand(name) => write!(f, "unknown command: {}", name),
            ParseError::MissingArg { command, expected } => {
                write!(f, "{} requires {}", command, expected)
            }
            ParseError::BadSyntax { command, detail } => {
                write!(f, "{}: {}", command, detail)
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Canonical command names for help + fuzzy suggestions
// ---------------------------------------------------------------------------

/// Flat list of every top-level command noun/verb the parser accepts, used
/// by the help-generator and the typo-suggestion fallback. Aliases are not
/// listed here to keep the suggestion list clean; misspellings of aliases
/// still resolve to the canonical name via the fuzzy matcher.
pub const TOP_LEVEL_COMMANDS: &[&str] = &[
    "help",
    "status",
    "version",
    "tables",
    "indexes",
    "users",
    "schemas",
    "databases",
    "stats",
    "vacuum",
    "analyze",
    "checkpoint",
    "sql",
    "watch",
    "source",
    "clear",
    "exit",
    "timing",
    "expanded",
    "csv",
    "output",
    // Ops commands.
    "wal",
    "slots",
    "cdc",
    "config",
    "sessions",
    "archive",
    "restore",
    "branches",
    "triggers",
    "jobs",
    "buffer",
];

// ---------------------------------------------------------------------------
// Top-level tokenizer
// ---------------------------------------------------------------------------

/// Splits `input` into whitespace-delimited tokens, honoring double-quoted
/// spans as a single token. Backslash inside a quoted span escapes the
/// next character. Returns the token list in the order they appeared.
///
/// Unclosed quotes produce a single trailing token containing everything
/// after the open quote, so the caller's arg-count check surfaces the
/// error rather than silently dropping tail content.
pub fn tokenize(input: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut current = String::new();
    let mut in_quotes = false;
    let mut chars = input.chars().peekable();

    while let Some(c) = chars.next() {
        if in_quotes {
            match c {
                '\\' => {
                    if let Some(next) = chars.next() {
                        current.push(next);
                    }
                }
                '"' => in_quotes = false,
                _ => current.push(c),
            }
        } else if c == '"' {
            in_quotes = true;
        } else if c.is_whitespace() {
            if !current.is_empty() {
                out.push(std::mem::take(&mut current));
            }
        } else {
            current.push(c);
        }
    }

    if !current.is_empty() {
        out.push(current);
    }
    out
}

// ---------------------------------------------------------------------------
// Command parser
// ---------------------------------------------------------------------------

impl Command {
    /// Parses a single input line into a `Command`. Recognizes short
    /// aliases (`q` / `quit` for `exit`, `?` for `help`) and lowercases
    /// the leading verb before matching. SQL bodies passed to `sql
    /// "..."` are preserved verbatim.
    pub fn parse(input: &str) -> Result<Command, ParseError> {
        let tokens = tokenize(input);
        if tokens.is_empty() {
            return Err(ParseError::Empty);
        }
        let head = tokens[0].to_ascii_lowercase();
        let rest = &tokens[1..];

        match head.as_str() {
            "help" | "?" | "h" => Ok(Command::Help {
                path: rest.iter().map(|s| s.to_ascii_lowercase()).collect(),
            }),
            "status" | "info" => Ok(Command::Status),
            "version" | "ver" => Ok(Command::Version),
            "tables" | "table" | "tbls" => parse_tables(rest),
            "indexes" | "index" | "idx" => parse_indexes(rest),
            "users" | "user" => parse_users(rest),
            "schemas" | "schema" => Ok(Command::Schemas),
            "databases" | "dbs" | "db" => Ok(Command::Databases),
            "stats" | "stat" => parse_stats(rest),
            "vacuum" => match rest.first() {
                Some(table) => Ok(Command::Vacuum {
                    table: table.clone(),
                }),
                None => Err(ParseError::MissingArg {
                    command: "vacuum",
                    expected: "a table name",
                }),
            },
            "analyze" => match rest.first() {
                Some(table) => Ok(Command::Analyze {
                    table: table.clone(),
                }),
                None => Err(ParseError::MissingArg {
                    command: "analyze",
                    expected: "a table name",
                }),
            },
            "checkpoint" => Ok(Command::Checkpoint),
            "sql" => {
                if rest.is_empty() {
                    Ok(Command::Sql(None))
                } else {
                    Ok(Command::Sql(Some(rest.join(" "))))
                }
            }
            "watch" => parse_watch(rest),
            "source" => match rest.first() {
                Some(path) => Ok(Command::Source { path: path.clone() }),
                None => Err(ParseError::MissingArg {
                    command: "source",
                    expected: "a file path",
                }),
            },
            "clear" | "cls" => Ok(Command::Clear),
            "exit" | "quit" | "q" | "bye" | "logout" | ":q" | ":wq" | ":x" => Ok(Command::Exit),
            "timing" => Ok(Command::Timing),
            "expanded" | "x" => Ok(Command::Expanded),
            "csv" => Ok(Command::Csv),
            "output" | "o" => Ok(Command::OutputFile(rest.first().cloned())),

            "wal" => parse_wal(rest),
            "slots" | "slot" => parse_slots(rest),
            "cdc" => parse_cdc(rest),
            "config" | "conf" => parse_config(rest),
            "sessions" | "session" | "activity" => parse_sessions(rest),
            "archive" => parse_archive(rest),
            "restore" => parse_restore(rest),
            "branches" | "branch" => Ok(Command::Branches),
            "triggers" | "trigger" => Ok(Command::Triggers),
            "jobs" | "job" | "streaming" => Ok(Command::Jobs),
            "buffer" | "buffers" | "bgwriter" => Ok(Command::Buffer),

            _ => Err(ParseError::UnknownCommand(tokens[0].clone())),
        }
    }
}

fn parse_wal(rest: &[String]) -> Result<Command, ParseError> {
    match rest.first().map(|s| s.to_ascii_lowercase()).as_deref() {
        None | Some("status") | Some("stat") => Ok(Command::Wal(WalAction::Status)),
        Some(other) => Err(ParseError::BadSyntax {
            command: "wal",
            detail: format!("unknown subcommand '{}' (try status)", other),
        }),
    }
}

fn parse_slots(rest: &[String]) -> Result<Command, ParseError> {
    let sub = rest
        .first()
        .map(|s| s.to_ascii_lowercase())
        .unwrap_or_else(|| "list".to_string());
    match sub.as_str() {
        "list" | "ls" => Ok(Command::Slots(SlotsAction::List)),
        "create" | "add" => match rest.get(1).cloned() {
            Some(name) => Ok(Command::Slots(SlotsAction::Create { name })),
            None => Err(ParseError::MissingArg {
                command: "slots create",
                expected: "a slot name",
            }),
        },
        "drop" | "delete" | "remove" => match rest.get(1).cloned() {
            Some(name) => Ok(Command::Slots(SlotsAction::Drop { name })),
            None => Err(ParseError::MissingArg {
                command: "slots drop",
                expected: "a slot name",
            }),
        },
        other => Err(ParseError::BadSyntax {
            command: "slots",
            detail: format!("unknown subcommand '{}' (try list, create, drop)", other),
        }),
    }
}

fn parse_cdc(rest: &[String]) -> Result<Command, ParseError> {
    let sub = rest
        .first()
        .map(|s| s.to_ascii_lowercase())
        .unwrap_or_else(|| "streams".to_string());
    match sub.as_str() {
        "streams" | "stream" => Ok(Command::Cdc(CdcAction::Streams)),
        "feeds" | "feed" => Ok(Command::Cdc(CdcAction::Feeds)),
        "ingests" | "ingest" => Ok(Command::Cdc(CdcAction::Ingests)),
        "drop" => {
            // cdc drop stream <name>
            let kind = rest.get(1).map(|s| s.to_ascii_lowercase());
            let name = rest.get(2).cloned();
            match (kind.as_deref(), name) {
                (Some("stream"), Some(n)) => Ok(Command::Cdc(CdcAction::DropStream { name: n })),
                _ => Err(ParseError::BadSyntax {
                    command: "cdc drop",
                    detail: "expected 'cdc drop stream <name>'".to_string(),
                }),
            }
        }
        other => Err(ParseError::BadSyntax {
            command: "cdc",
            detail: format!(
                "unknown subcommand '{}' (try streams, feeds, ingests, drop stream <name>)",
                other
            ),
        }),
    }
}

fn parse_config(rest: &[String]) -> Result<Command, ParseError> {
    let sub = rest
        .first()
        .map(|s| s.to_ascii_lowercase())
        .unwrap_or_else(|| "show".to_string());
    match sub.as_str() {
        "show" | "get" => Ok(Command::Config(ConfigAction::Show {
            key: rest.get(1).cloned(),
        })),
        "set" => {
            let key = rest.get(1).cloned();
            // Allow values with spaces by re-joining everything after the key.
            let value = if rest.len() > 2 {
                Some(rest[2..].join(" "))
            } else {
                None
            };
            match (key, value) {
                (Some(k), Some(v)) => Ok(Command::Config(ConfigAction::Set { key: k, value: v })),
                _ => Err(ParseError::BadSyntax {
                    command: "config set",
                    detail: "expected 'config set <key> <value>'".to_string(),
                }),
            }
        }
        other => Err(ParseError::BadSyntax {
            command: "config",
            detail: format!("unknown subcommand '{}' (try show, set)", other),
        }),
    }
}

fn parse_sessions(rest: &[String]) -> Result<Command, ParseError> {
    match rest.first().map(|s| s.to_ascii_lowercase()).as_deref() {
        None | Some("list") | Some("ls") => Ok(Command::Sessions(SessionsAction::List)),
        Some(other) => Err(ParseError::BadSyntax {
            command: "sessions",
            detail: format!("unknown subcommand '{}' (try list)", other),
        }),
    }
}

/// Expects `archive <table> where <predicate...> to <path>`. The WHERE
/// predicate may contain whitespace; we consume tokens up to the `to`
/// keyword.
fn parse_archive(rest: &[String]) -> Result<Command, ParseError> {
    let table = rest.first().cloned().ok_or(ParseError::MissingArg {
        command: "archive",
        expected: "a table name",
    })?;
    if rest.get(1).map(|s| s.to_ascii_lowercase()).as_deref() != Some("where") {
        return Err(ParseError::BadSyntax {
            command: "archive",
            detail: "expected 'archive <table> where <predicate> to <path>'".to_string(),
        });
    }
    let to_pos = rest
        .iter()
        .position(|s| s.eq_ignore_ascii_case("to"))
        .ok_or(ParseError::BadSyntax {
            command: "archive",
            detail: "missing `to <path>` clause".to_string(),
        })?;
    let where_tokens = &rest[2..to_pos];
    let to_tokens = &rest[to_pos + 1..];
    if where_tokens.is_empty() {
        return Err(ParseError::MissingArg {
            command: "archive",
            expected: "a predicate after `where`",
        });
    }
    if to_tokens.is_empty() {
        return Err(ParseError::MissingArg {
            command: "archive",
            expected: "a path after `to`",
        });
    }
    Ok(Command::Archive {
        table,
        where_clause: where_tokens.join(" "),
        to_path: to_tokens.join(" "),
    })
}

/// Expects `restore <table> from <path> [into <target>]`.
fn parse_restore(rest: &[String]) -> Result<Command, ParseError> {
    let table = rest.first().cloned().ok_or(ParseError::MissingArg {
        command: "restore",
        expected: "a table name",
    })?;
    if rest.get(1).map(|s| s.to_ascii_lowercase()).as_deref() != Some("from") {
        return Err(ParseError::BadSyntax {
            command: "restore",
            detail: "expected 'restore <table> from <path> [into <target>]'".to_string(),
        });
    }
    let from_path = rest.get(2).cloned().ok_or(ParseError::MissingArg {
        command: "restore",
        expected: "a path after `from`",
    })?;
    let into_target = if rest.len() > 3 {
        if rest[3].eq_ignore_ascii_case("into") {
            rest.get(4).cloned()
        } else {
            return Err(ParseError::BadSyntax {
                command: "restore",
                detail: "expected 'into <target>' after the source path".to_string(),
            });
        }
    } else {
        None
    };
    Ok(Command::Restore {
        table,
        from_path,
        into_target,
    })
}

fn parse_tables(rest: &[String]) -> Result<Command, ParseError> {
    let sub = rest
        .first()
        .map(|s| s.to_ascii_lowercase())
        .unwrap_or_else(|| "list".to_string());
    let arg = rest.get(1).cloned();
    match sub.as_str() {
        "list" | "ls" => Ok(Command::Tables(TablesAction::List)),
        "describe" | "desc" | "show" => match arg {
            Some(name) => Ok(Command::Tables(TablesAction::Describe(name))),
            None => Err(ParseError::MissingArg {
                command: "tables describe",
                expected: "a table name",
            }),
        },
        "drop" => match arg {
            Some(name) => Ok(Command::Tables(TablesAction::Drop(name))),
            None => Err(ParseError::MissingArg {
                command: "tables drop",
                expected: "a table name",
            }),
        },
        "truncate" => match arg {
            Some(name) => Ok(Command::Tables(TablesAction::Truncate(name))),
            None => Err(ParseError::MissingArg {
                command: "tables truncate",
                expected: "a table name",
            }),
        },
        other => Err(ParseError::BadSyntax {
            command: "tables",
            detail: format!(
                "unknown subcommand '{}' (try list, describe, drop, truncate)",
                other
            ),
        }),
    }
}

fn parse_indexes(rest: &[String]) -> Result<Command, ParseError> {
    let sub = rest
        .first()
        .map(|s| s.to_ascii_lowercase())
        .unwrap_or_else(|| "list".to_string());
    let arg = rest.get(1).cloned();
    match sub.as_str() {
        "list" | "ls" => Ok(Command::Indexes(IndexesAction::List)),
        "describe" | "desc" | "show" => match arg {
            Some(name) => Ok(Command::Indexes(IndexesAction::Describe(name))),
            None => Err(ParseError::MissingArg {
                command: "indexes describe",
                expected: "an index name",
            }),
        },
        "drop" => match arg {
            Some(name) => Ok(Command::Indexes(IndexesAction::Drop(name))),
            None => Err(ParseError::MissingArg {
                command: "indexes drop",
                expected: "an index name",
            }),
        },
        "reindex" => match arg {
            Some(name) => Ok(Command::Indexes(IndexesAction::Reindex(name))),
            None => Err(ParseError::MissingArg {
                command: "indexes reindex",
                expected: "an index name",
            }),
        },
        other => Err(ParseError::BadSyntax {
            command: "indexes",
            detail: format!(
                "unknown subcommand '{}' (try list, describe, drop, reindex)",
                other
            ),
        }),
    }
}

fn parse_users(rest: &[String]) -> Result<Command, ParseError> {
    let sub = rest
        .first()
        .map(|s| s.to_ascii_lowercase())
        .unwrap_or_else(|| "list".to_string());
    match sub.as_str() {
        "list" | "ls" => Ok(Command::Users(UsersAction::List)),
        "create" | "add" => match rest.get(1).cloned() {
            Some(name) => Ok(Command::Users(UsersAction::Create(name))),
            None => Err(ParseError::MissingArg {
                command: "users create",
                expected: "a user name",
            }),
        },
        "drop" | "delete" | "remove" => match rest.get(1).cloned() {
            Some(name) => Ok(Command::Users(UsersAction::Drop(name))),
            None => Err(ParseError::MissingArg {
                command: "users drop",
                expected: "a user name",
            }),
        },
        "grant" => parse_grant(&rest[1..]),
        "revoke" => parse_revoke(&rest[1..]),
        other => Err(ParseError::BadSyntax {
            command: "users",
            detail: format!(
                "unknown subcommand '{}' (try list, create, drop, grant, revoke)",
                other
            ),
        }),
    }
}

/// Expects `<priv> on <object> to <user>` (tokens after the grant verb).
fn parse_grant(tokens: &[String]) -> Result<Command, ParseError> {
    let (priv_, object, user) = parse_grant_like(tokens, "to", "users grant")?;
    Ok(Command::Users(UsersAction::Grant {
        privilege: priv_,
        object,
        user,
    }))
}

/// Expects `<priv> on <object> from <user>` (tokens after the revoke verb).
fn parse_revoke(tokens: &[String]) -> Result<Command, ParseError> {
    let (priv_, object, user) = parse_grant_like(tokens, "from", "users revoke")?;
    Ok(Command::Users(UsersAction::Revoke {
        privilege: priv_,
        object,
        user,
    }))
}

/// Shared grammar for `<priv> on <object> <direction_keyword> <user>`.
fn parse_grant_like(
    tokens: &[String],
    direction_keyword: &'static str,
    command: &'static str,
) -> Result<(String, String, String), ParseError> {
    // Expected shape: priv ON object direction_keyword user
    if tokens.len() != 5 {
        return Err(ParseError::BadSyntax {
            command,
            detail: format!(
                "expected '<privilege> on <object> {} <user>'",
                direction_keyword
            ),
        });
    }
    if tokens[1].to_ascii_lowercase() != "on" || tokens[3].to_ascii_lowercase() != direction_keyword
    {
        return Err(ParseError::BadSyntax {
            command,
            detail: format!(
                "expected '<privilege> on <object> {} <user>'",
                direction_keyword
            ),
        });
    }
    Ok((tokens[0].clone(), tokens[2].clone(), tokens[4].clone()))
}

fn parse_stats(rest: &[String]) -> Result<Command, ParseError> {
    match rest.first().map(|s| s.to_ascii_lowercase()).as_deref() {
        None => Ok(Command::Stats(None)),
        Some("summary") => Ok(Command::Stats(Some(StatsView::Summary))),
        Some("wal") => Ok(Command::Stats(Some(StatsView::Wal))),
        Some("tables") => Ok(Command::Stats(Some(StatsView::Tables))),
        Some("indexes") => Ok(Command::Stats(Some(StatsView::Indexes))),
        Some("buffer") | Some("buffers") => Ok(Command::Stats(Some(StatsView::Buffer))),
        Some("connections") | Some("conns") => Ok(Command::Stats(Some(StatsView::Connections))),
        Some(other) => Err(ParseError::BadSyntax {
            command: "stats",
            detail: format!(
                "unknown view '{}' (try summary, wal, tables, indexes, buffer, connections)",
                other
            ),
        }),
    }
}

fn parse_watch(rest: &[String]) -> Result<Command, ParseError> {
    if rest.is_empty() {
        return Err(ParseError::MissingArg {
            command: "watch",
            expected: "a command to re-run (optionally preceded by an interval in seconds)",
        });
    }
    // First token is an integer interval; default to 2s when the first
    // token is a command name.
    let (interval_secs, inner_tokens) = match rest[0].parse::<u64>() {
        Ok(n) if n > 0 => (n, &rest[1..]),
        Ok(_) => {
            return Err(ParseError::BadSyntax {
                command: "watch",
                detail: "interval must be a positive integer (seconds)".to_string(),
            });
        }
        Err(_) => (2, rest),
    };
    if inner_tokens.is_empty() {
        return Err(ParseError::MissingArg {
            command: "watch",
            expected: "a command to re-run after the interval",
        });
    }
    Ok(Command::Watch {
        interval_secs,
        inner: inner_tokens.to_vec(),
    })
}

// ---------------------------------------------------------------------------
// Fuzzy match for typo suggestions
// ---------------------------------------------------------------------------

/// Returns the closest top-level command name to `input` whose Levenshtein
/// distance is `<= 2`, or `None` when nothing is that close. Operates on
/// byte distance (ASCII command names) so the cost is trivial.
pub fn closest_command(input: &str) -> Option<&'static str> {
    let input_lower = input.to_ascii_lowercase();
    let mut best: Option<(&'static str, usize)> = None;
    for name in TOP_LEVEL_COMMANDS {
        let d = levenshtein(&input_lower, name);
        if d > 2 {
            continue;
        }
        if best.map_or(true, |(_, bd)| d < bd) {
            best = Some((name, d));
        }
    }
    best.map(|(n, _)| n)
}

/// Classic DP Levenshtein distance on bytes. Two rolling rows; O(n*m)
/// time, O(min(n,m)) space. Input is expected to be short ASCII command
/// names, so the allocation is tiny and one-shot.
fn levenshtein(a: &str, b: &str) -> usize {
    let a_bytes = a.as_bytes();
    let b_bytes = b.as_bytes();
    if a_bytes.is_empty() {
        return b_bytes.len();
    }
    if b_bytes.is_empty() {
        return a_bytes.len();
    }
    let (short, long) = if a_bytes.len() < b_bytes.len() {
        (a_bytes, b_bytes)
    } else {
        (b_bytes, a_bytes)
    };
    let mut prev: Vec<usize> = (0..=short.len()).collect();
    let mut curr: Vec<usize> = vec![0; short.len() + 1];
    for (i, &lb) in long.iter().enumerate() {
        curr[0] = i + 1;
        for (j, &sb) in short.iter().enumerate() {
            let cost = if lb == sb { 0 } else { 1 };
            curr[j + 1] = (curr[j] + 1).min(prev[j + 1] + 1).min(prev[j] + cost);
        }
        std::mem::swap(&mut prev, &mut curr);
    }
    prev[short.len()]
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tokenize_plain() {
        assert_eq!(
            tokenize("tables describe users"),
            vec!["tables", "describe", "users",]
        );
    }

    #[test]
    fn tokenize_quoted_span() {
        assert_eq!(
            tokenize("sql \"SELECT a FROM b\""),
            vec!["sql", "SELECT a FROM b",]
        );
    }

    #[test]
    fn tokenize_escape_inside_quotes() {
        assert_eq!(tokenize("sql \"a \\\"b\\\" c\""), vec!["sql", "a \"b\" c"]);
    }

    #[test]
    fn parse_exit_aliases() {
        for input in ["exit", "quit", "q", "bye", ":q"] {
            assert!(matches!(Command::parse(input), Ok(Command::Exit)));
        }
    }

    #[test]
    fn parse_tables_list_default() {
        assert!(matches!(
            Command::parse("tables"),
            Ok(Command::Tables(TablesAction::List))
        ));
    }

    #[test]
    fn parse_tables_describe_requires_arg() {
        assert!(matches!(
            Command::parse("tables describe"),
            Err(ParseError::MissingArg { .. })
        ));
    }

    #[test]
    fn parse_watch_interval_optional() {
        let cmd = Command::parse("watch tables").unwrap();
        match cmd {
            Command::Watch {
                interval_secs,
                inner,
            } => {
                assert_eq!(interval_secs, 2);
                assert_eq!(inner, vec!["tables".to_string()]);
            }
            _ => panic!("expected Watch"),
        }

        let cmd = Command::parse("watch 5 stats wal").unwrap();
        match cmd {
            Command::Watch {
                interval_secs,
                inner,
            } => {
                assert_eq!(interval_secs, 5);
                assert_eq!(inner, vec!["stats".to_string(), "wal".to_string()]);
            }
            _ => panic!("expected Watch"),
        }
    }

    #[test]
    fn parse_grant_full_form() {
        let cmd = Command::parse("users grant SELECT on orders to alice").unwrap();
        match cmd {
            Command::Users(UsersAction::Grant {
                privilege,
                object,
                user,
            }) => {
                assert_eq!(privilege, "SELECT");
                assert_eq!(object, "orders");
                assert_eq!(user, "alice");
            }
            _ => panic!("expected Grant"),
        }
    }

    #[test]
    fn fuzzy_suggest_nearby() {
        assert_eq!(closest_command("tbls"), Some("tables"));
        assert_eq!(closest_command("statz"), Some("stats"));
        // Too far, no suggestion.
        assert_eq!(closest_command("xyzzy"), None);
    }
}
