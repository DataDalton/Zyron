//! Human-readable help text for the command vocabulary.
//!
//! Rather than generating help from attribute macros or parser definitions
//! (which would force either heavy deps or brittle reflection), help is
//! hand-written as a sequence of `CommandHelp` records adjacent to the
//! `Command` enum. The trade-off is manual drift risk between code and
//! docs, which a single new test in `command.rs` could catch by asserting
//! every `TOP_LEVEL_COMMANDS` entry has a matching help record.

use crate::color;

/// Description of a single top-level command for the help screen. Kept
/// static and heap-free so `help` dispatch is a simple array walk.
struct CommandHelp {
    name: &'static str,
    /// Short one-line summary shown in the index.
    summary: &'static str,
    /// Full help text printed by `help <name>`. May include examples,
    /// syntax lines, and alias mentions. Rendered with leading-indent
    /// preserved; blank lines intentional.
    detail: &'static str,
}

/// All top-level command help entries. Order matches the order they're
/// printed in the index.
const ENTRIES: &[CommandHelp] = &[
    CommandHelp {
        name: "help",
        summary: "Show command help. `help <name>` drills into a specific command.",
        detail: "\
Usage:
  help                  Show the command index.
  help <command>        Show detailed help for a single command.

Aliases: ?, h",
    },
    CommandHelp {
        name: "status",
        summary: "Connection + server status summary.",
        detail: "\
Usage:
  status

Prints the current user, host:port, database, server version, and
active display toggles (timing, expanded, csv, output file).

Aliases: info",
    },
    CommandHelp {
        name: "version",
        summary: "Print client and server version strings.",
        detail: "\
Usage:
  version

Aliases: ver",
    },
    CommandHelp {
        name: "tables",
        summary: "List, describe, drop, or truncate tables.",
        detail: "\
Usage:
  tables                         Short for `tables list`.
  tables list                    List every table in every schema.
  tables describe <name>         Show columns, types, and constraints.
  tables drop <name>             DROP TABLE <name>.
  tables truncate <name>         TRUNCATE TABLE <name>.

Aliases for the noun: table, tbls.
Aliases for `describe`: desc, show.
Aliases for `list`: ls.",
    },
    CommandHelp {
        name: "indexes",
        summary: "List, describe, drop, or reindex indexes.",
        detail: "\
Usage:
  indexes                        Short for `indexes list`.
  indexes list
  indexes describe <name>
  indexes drop <name>
  indexes reindex <name>

Aliases for the noun: index, idx.",
    },
    CommandHelp {
        name: "users",
        summary: "List, create, drop, grant, or revoke user privileges.",
        detail: "\
Usage:
  users                                            Short for `users list`.
  users list
  users create <name>
  users drop <name>
  users grant <privilege> on <object> to <user>
  users revoke <privilege> on <object> from <user>

Examples:
  users grant SELECT on orders to alice
  users revoke INSERT on orders from bob

Aliases for `create`: add.
Aliases for `drop`: delete, remove.",
    },
    CommandHelp {
        name: "schemas",
        summary: "List all schemas in the current database.",
        detail: "\
Usage:
  schemas

Aliases: schema",
    },
    CommandHelp {
        name: "databases",
        summary: "List all databases on the server.",
        detail: "\
Usage:
  databases

Aliases: dbs, db",
    },
    CommandHelp {
        name: "stats",
        summary: "Show server counters and statistics.",
        detail: "\
Usage:
  stats                         Summary across all subsystems.
  stats wal                     WAL write/replay counters and current LSN.
  stats tables                  Per-table row and page counts.
  stats indexes                 Per-index scan / fetch counters.
  stats buffer                  Buffer-pool hit/miss/eviction counters.
  stats connections             Active sessions and their state.

Tip: combine with `watch`:
  watch 2 stats wal             Refreshes every 2 seconds.",
    },
    CommandHelp {
        name: "vacuum",
        summary: "Reclaim dead-tuple space in a table.",
        detail: "\
Usage:
  vacuum <table>",
    },
    CommandHelp {
        name: "analyze",
        summary: "Refresh optimizer statistics for a table.",
        detail: "\
Usage:
  analyze <table>",
    },
    CommandHelp {
        name: "checkpoint",
        summary: "Force a WAL checkpoint immediately.",
        detail: "\
Usage:
  checkpoint

Normally the background trigger runs checkpoints automatically; this
command is for diagnostic or pre-backup use.",
    },
    CommandHelp {
        name: "sql",
        summary: "Run SQL. `sql` enters a sub-REPL; `sql \"...\"` runs one query.",
        detail: "\
Usage:
  sql                           Enter SQL sub-REPL. Inside it, type SQL
                                statements terminated by `;`. Use
                                `exit` or Ctrl-D to return to command mode.
  sql \"<statement>\"             Run one SQL statement and return.

Inside the sub-REPL, the prompt is `zyron(sql)=>`. Transactions are
reflected with `*=>` (in txn) and `!=>` (failed txn) markers.",
    },
    CommandHelp {
        name: "watch",
        summary: "Repeat another command on a timer until Ctrl-C.",
        detail: "\
Usage:
  watch <command...>              Refresh every 2 seconds (default).
  watch <secs> <command...>       Custom interval.

Examples:
  watch stats wal
  watch 5 tables
  watch 1 sql \"SELECT count(*) FROM orders\"

Press Ctrl-C to stop.",
    },
    CommandHelp {
        name: "source",
        summary: "Run commands from a file, one per line.",
        detail: "\
Usage:
  source <path>

Each non-empty, non-comment line is dispatched as if typed at the
prompt. Lines starting with `#` or `--` are treated as comments.",
    },
    CommandHelp {
        name: "clear",
        summary: "Clear the terminal screen.",
        detail: "\
Usage:
  clear

Ctrl-L also clears the screen via rustyline.

Aliases: cls",
    },
    CommandHelp {
        name: "exit",
        summary: "Leave the CLI.",
        detail: "\
Usage:
  exit

Aliases: quit, q, bye, logout, :q, :wq, :x. Ctrl-D also exits.",
    },
    CommandHelp {
        name: "timing",
        summary: "Toggle display of query elapsed time in SQL sub-mode.",
        detail: "\
Usage:
  timing",
    },
    CommandHelp {
        name: "expanded",
        summary: "Toggle expanded row format in SQL sub-mode results.",
        detail: "\
Usage:
  expanded

Aliases: x",
    },
    CommandHelp {
        name: "csv",
        summary: "Toggle CSV output format for SQL sub-mode results.",
        detail: "\
Usage:
  csv",
    },
    CommandHelp {
        name: "output",
        summary: "Redirect output to a file (no arg resets to stdout).",
        detail: "\
Usage:
  output <path>                Write subsequent results to this file.
  output                        Reset back to stdout.

Aliases: o",
    },
    CommandHelp {
        name: "wal",
        summary: "WAL introspection (LSN, segments, write/replay counters).",
        detail: "\
Usage:
  wal                           Short for `wal status`.
  wal status                    SELECT * FROM zyron_stat_wal",
    },
    CommandHelp {
        name: "slots",
        summary: "Manage replication slots.",
        detail: "\
Usage:
  slots                          Short for `slots list`.
  slots list
  slots create <name>            CREATE REPLICATION SLOT
  slots drop <name>              DROP REPLICATION SLOT

Aliases for the noun: slot.",
    },
    CommandHelp {
        name: "cdc",
        summary: "Change-data-capture streams, feeds, and ingestion jobs.",
        detail: "\
Usage:
  cdc                            Short for `cdc streams`.
  cdc streams                    Outbound CDC stream registry.
  cdc feeds                      Per-table change-feed status.
  cdc ingests                    Inbound CDC ingestion jobs.
  cdc drop stream <name>         DROP CDC STREAM",
    },
    CommandHelp {
        name: "config",
        summary: "Show or update runtime configuration.",
        detail: "\
Usage:
  config                         Short for `config show`.
  config show                    SHOW ALL
  config show <key>              SHOW <key>
  config set <key> <value>       ALTER SYSTEM SET <key> = <value>

Aliases for the noun: conf.",
    },
    CommandHelp {
        name: "sessions",
        summary: "Active sessions and their current activity.",
        detail: "\
Usage:
  sessions                       Short for `sessions list`.
  sessions list                  SELECT * FROM zyron_stat_activity

Aliases for the noun: session, activity.",
    },
    CommandHelp {
        name: "archive",
        summary: "ARCHIVE TABLE: move rows matching a predicate out to a file.",
        detail: "\
Usage:
  archive <table> where <predicate> to <path>

Example:
  archive orders where created_at < '2024-01-01' to 'archives/orders_2023.zyr'",
    },
    CommandHelp {
        name: "restore",
        summary: "RESTORE TABLE: load previously-archived rows back.",
        detail: "\
Usage:
  restore <table> from <path>
  restore <table> from <path> into <target_table>",
    },
    CommandHelp {
        name: "branches",
        summary: "List version branches on the server.",
        detail: "\
Usage:
  branches                       SELECT * FROM zyron_stat_branches

Aliases: branch.",
    },
    CommandHelp {
        name: "triggers",
        summary: "List triggers registered on tables.",
        detail: "\
Usage:
  triggers                       SELECT * FROM zyron_stat_triggers

Aliases: trigger.",
    },
    CommandHelp {
        name: "jobs",
        summary: "List streaming jobs.",
        detail: "\
Usage:
  jobs                           SELECT * FROM zyron_stat_streaming_jobs

Aliases: job, streaming.",
    },
    CommandHelp {
        name: "buffer",
        summary: "Buffer-pool and background-writer counters.",
        detail: "\
Usage:
  buffer                         SELECT * FROM zyron_stat_bgwriter

Aliases: buffers, bgwriter.",
    },
];

/// Prints the top-level command index: every command + its one-line
/// summary, aligned. Called when the user types bare `help` or `?`.
pub fn print_index() {
    let width = ENTRIES.iter().map(|e| e.name.len()).max().unwrap_or(0);
    println!("{}", color::bold("Commands:"));
    println!();
    for e in ENTRIES {
        println!(
            "  {}   {}",
            color::bold_cyan(&format!("{:width$}", e.name, width = width)),
            e.summary,
        );
    }
    println!();
    println!(
        "{}",
        color::dim("Type `help <command>` for detailed usage, or `sql` to enter the SQL sub-REPL."),
    );
}

/// Prints detailed help for a specific command, or an error message when
/// the name is unknown. Recognizes the canonical command names only;
/// aliases fall through to the index.
pub fn print_command(name: &str) -> bool {
    let lower = name.to_ascii_lowercase();
    if let Some(entry) = ENTRIES.iter().find(|e| e.name == lower) {
        println!(
            "{}  {}",
            color::bold_cyan(entry.name),
            color::dim(entry.summary)
        );
        println!();
        println!("{}", entry.detail);
        true
    } else {
        false
    }
}
