//! Command-to-SQL translator.
//!
//! Every `Command` that needs server interaction compiles to a single SQL
//! statement (or small sequence) so the wire-protocol path stays uniform
//! with direct SQL submission. The server's privilege machinery,
//! classification / masking, and transaction bookkeeping all run
//! identically whether the query came from the CLI's command layer or
//! from a pgAdmin SQL editor.
//!
//! Commands that do not require a server round-trip (help, clear, exit,
//! toggle flags, `sql` sub-mode) are handled by `main::handle_command`
//! directly and do not appear here.

use crate::command::{
    CdcAction, Command, ConfigAction, IndexesAction, SessionsAction, SlotsAction, StatsView,
    TablesAction, UsersAction, WalAction,
};

/// Returns the SQL statement(s) that implement `cmd` on the server. For
/// purely local commands (help, clear, toggles, etc.) returns `None`.
///
/// Every statement here names an entity of the `zyron_sys` catalog by its
/// canonical three-part name. The system entities answer a projection, a
/// conjunction of equalities, LIMIT, and OFFSET, and refuse everything else,
/// so none of these carries an ORDER BY: a view that cannot sort would
/// reject the statement rather than return unsorted rows.
pub fn command_to_sql(cmd: &Command) -> Option<String> {
    match cmd {
        Command::Tables(TablesAction::List) => Some(
            "SELECT table_name, schema_name, storage_format, column_count \
             FROM zyron_sys.core.tables"
                .into(),
        ),
        Command::Tables(TablesAction::Describe(name)) => {
            Some(format!("SHOW COLUMNS FROM {}", quote_ident(name)))
        }
        Command::Tables(TablesAction::Drop(name)) => {
            Some(format!("DROP TABLE {}", quote_ident(name)))
        }
        Command::Tables(TablesAction::Truncate(name)) => {
            Some(format!("TRUNCATE TABLE {}", quote_ident(name)))
        }

        Command::Indexes(IndexesAction::List) => Some(
            "SELECT index_name, table_name, schema_name, index_type, is_unique \
             FROM zyron_sys.storage.indexes"
                .into(),
        ),
        Command::Indexes(IndexesAction::Describe(name)) => Some(format!(
            "SELECT index_name, table_name, index_type, key_columns, is_unique \
             FROM zyron_sys.storage.indexes WHERE index_name = {}",
            quote_string(name)
        )),
        Command::Indexes(IndexesAction::Drop(name)) => {
            Some(format!("DROP INDEX {}", quote_ident(name)))
        }
        Command::Indexes(IndexesAction::Reindex(name)) => {
            Some(format!("REINDEX {}", quote_ident(name)))
        }

        Command::Users(UsersAction::List) => Some(
            "SELECT username, can_login, is_superuser, locked, valid_until \
             FROM zyron_sys.security.users"
                .into(),
        ),
        Command::Users(UsersAction::Create(name)) => {
            Some(format!("CREATE USER {}", quote_ident(name)))
        }
        Command::Users(UsersAction::Drop(name)) => Some(format!("DROP USER {}", quote_ident(name))),
        Command::Users(UsersAction::Grant {
            privilege,
            object,
            user,
        }) => Some(format!(
            "GRANT {} ON {} TO {}",
            sanitize_keyword(privilege),
            quote_ident(object),
            quote_ident(user),
        )),
        Command::Users(UsersAction::Revoke {
            privilege,
            object,
            user,
        }) => Some(format!(
            "REVOKE {} ON {} FROM {}",
            sanitize_keyword(privilege),
            quote_ident(object),
            quote_ident(user),
        )),

        Command::Schemas => Some(
            "SELECT catalog_name, schema_name, owner, is_system \
             FROM zyron_sys.core.schemas"
                .into(),
        ),
        Command::Databases => Some(
            "SELECT catalog_name, owner, created_at, is_system \
             FROM zyron_sys.core.databases"
                .into(),
        ),

        Command::Stats(view) => Some(stats_sql(*view)),

        Command::Vacuum { table } => Some(format!("VACUUM {}", quote_ident(table))),
        Command::Analyze { table } => Some(format!("ANALYZE {}", quote_ident(table))),
        Command::Checkpoint => Some("CHECKPOINT".into()),

        // -- Ops surface --
        Command::Wal(WalAction::Status) => Some("SELECT * FROM zyron_sys.stat.wal".into()),

        Command::Slots(SlotsAction::List) => Some(
            "SELECT name, plugin, active, restart_lsn, confirmed_lsn, lag_bytes \
             FROM zyron_sys.stat.replication_slots"
                .into(),
        ),
        Command::Slots(SlotsAction::Create { name }) => {
            Some(format!("CREATE REPLICATION SLOT {}", quote_ident(name)))
        }
        Command::Slots(SlotsAction::Drop { name }) => {
            Some(format!("DROP REPLICATION SLOT {}", quote_ident(name)))
        }

        Command::Cdc(CdcAction::Streams) => Some("SELECT * FROM zyron_sys.stat.cdc_streams".into()),
        Command::Cdc(CdcAction::Feeds) => Some("SELECT * FROM zyron_sys.stat.cdc_feeds".into()),
        Command::Cdc(CdcAction::Ingests) => Some("SELECT * FROM zyron_sys.stat.cdc_ingests".into()),
        Command::Cdc(CdcAction::DropStream { name }) => {
            Some(format!("DROP CDC STREAM {}", quote_ident(name)))
        }

        Command::Config(ConfigAction::Show { key }) => Some(match key {
            Some(k) => format!("SHOW {}", sanitize_keyword(k)),
            None => "SHOW ALL".into(),
        }),
        Command::Config(ConfigAction::Set { key, value }) => Some(format!(
            "ALTER SYSTEM SET {} = {}",
            sanitize_keyword(key),
            quote_string(value)
        )),

        Command::Sessions(SessionsAction::List) => {
            Some("SELECT * FROM zyron_sys.stat.activity".into())
        }

        Command::Archive {
            table,
            where_clause,
            to_path,
        } => Some(format!(
            "ARCHIVE TABLE {} WHERE {} TO {}",
            quote_ident(table),
            where_clause,
            quote_string(to_path),
        )),
        Command::Restore {
            table,
            from_path,
            into_target,
        } => Some(match into_target {
            Some(t) => format!(
                "RESTORE TABLE {} FROM {} INTO {}",
                quote_ident(table),
                quote_string(from_path),
                quote_ident(t),
            ),
            None => format!(
                "RESTORE TABLE {} FROM {}",
                quote_ident(table),
                quote_string(from_path),
            ),
        }),

        Command::Branches => Some("SELECT * FROM zyron_sys.stat.branches".into()),
        Command::Triggers => Some("SELECT * FROM zyron_sys.stat.trigger_executions".into()),
        Command::Jobs => Some("SELECT * FROM zyron_sys.streaming.jobs".into()),
        Command::Buffer => Some("SELECT * FROM zyron_sys.stat.bgwriter".into()),

        // Non-server or handled directly by the dispatcher.
        Command::Help { .. }
        | Command::Status
        | Command::Version
        | Command::Sql(_)
        | Command::Watch { .. }
        | Command::Source { .. }
        | Command::Clear
        | Command::Exit
        | Command::Timing
        | Command::Expanded
        | Command::Csv
        | Command::OutputFile(_) => None,
    }
}

/// Returns the SQL backing a specific stats view. The summary view groups
/// several counters into a single result set via UNION ALL so the CLI
/// renders one table rather than six.
fn stats_sql(view: Option<StatsView>) -> String {
    match view {
        Some(StatsView::Wal) => "SELECT * FROM zyron_sys.stat.wal".into(),
        Some(StatsView::Tables) => "SELECT * FROM zyron_sys.stat.tables".into(),
        Some(StatsView::Indexes) => "SELECT * FROM zyron_sys.stat.indexes".into(),
        Some(StatsView::Buffer) => "SELECT * FROM zyron_sys.stat.bgwriter".into(),
        Some(StatsView::Connections) => "SELECT * FROM zyron_sys.stat.activity".into(),
        None | Some(StatsView::Summary) => "SELECT * FROM zyron_sys.stat.summary".into(),
    }
}

/// Wraps `name` as a double-quoted SQL identifier, doubling any embedded
/// quotes. Callers hand us user-supplied identifiers whose lexical form
/// may collide with SQL keywords (`user`, `schema`, etc.); quoting always
/// is safer than sometimes.
fn quote_ident(name: &str) -> String {
    let escaped = name.replace('"', "\"\"");
    format!("\"{}\"", escaped)
}

/// Single-quotes `value` for inclusion in a SQL string literal, doubling
/// any embedded apostrophes.
fn quote_string(value: &str) -> String {
    let escaped = value.replace('\'', "''");
    format!("'{}'", escaped)
}

/// Privilege names (`SELECT`, `INSERT`, `ALL`, etc.) must not be quoted
/// like identifiers because the SQL grammar rejects quoted privilege
/// keywords. Strips any character that isn't alphanumeric or underscore
/// so we can't inject SQL through the privilege slot.
fn sanitize_keyword(word: &str) -> String {
    word.chars()
        .filter(|c| c.is_ascii_alphanumeric() || *c == '_')
        .collect::<String>()
        .to_ascii_uppercase()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tables_list_sql() {
        let cmd = Command::Tables(TablesAction::List);
        let sql = command_to_sql(&cmd).unwrap();
        assert!(sql.starts_with("SELECT table_name"));
    }

    #[test]
    fn tables_describe_quotes_identifier() {
        let cmd = Command::Tables(TablesAction::Describe("users".into()));
        assert_eq!(command_to_sql(&cmd).unwrap(), "SHOW COLUMNS FROM \"users\"");
    }

    #[test]
    fn grant_sanitizes_privilege() {
        let cmd = Command::Users(UsersAction::Grant {
            privilege: "SELECT; DROP TABLE users".into(),
            object: "orders".into(),
            user: "alice".into(),
        });
        let sql = command_to_sql(&cmd).unwrap();
        assert!(sql.starts_with("GRANT SELECTDROPTABLEUSERS"));
        assert!(sql.contains("ON \"orders\""));
        assert!(sql.contains("TO \"alice\""));
    }

    #[test]
    fn quote_ident_escapes_quotes() {
        assert_eq!(quote_ident("a\"b"), "\"a\"\"b\"");
    }

    #[test]
    fn help_has_no_sql() {
        assert!(command_to_sql(&Command::Help { path: vec![] }).is_none());
    }
}
