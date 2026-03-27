// Meta-command parsing and dispatch for backslash commands.

/// Recognized CLI meta-commands triggered by backslash prefix.
pub enum MetaCommand {
    ListTables,
    DescribeTable(String),
    ListIndexes,
    ListUsers,
    ListPrivileges,
    ToggleTiming,
    ToggleExpanded,
    ToggleCsv,
    OutputFile(Option<String>),
    InputFile(String),
    Quit,
    Help,
    Unknown(String),
}

/// Parses input starting with '\' into a MetaCommand.
/// Returns None if the input does not start with '\'.
pub fn parse_meta_command(input: &str) -> Option<MetaCommand> {
    let trimmed = input.trim();
    if !trimmed.starts_with('\\') {
        return None;
    }

    let mut parts = trimmed.splitn(2, char::is_whitespace);
    let cmd = parts.next().unwrap_or("");
    let arg = parts.next().map(|s| s.trim().to_string());

    let meta = match cmd {
        "\\dt" => MetaCommand::ListTables,
        "\\d" => {
            if let Some(table) = arg {
                if table.is_empty() {
                    MetaCommand::ListTables
                } else {
                    MetaCommand::DescribeTable(table)
                }
            } else {
                MetaCommand::ListTables
            }
        }
        "\\di" => MetaCommand::ListIndexes,
        "\\du" => MetaCommand::ListUsers,
        "\\dp" => MetaCommand::ListPrivileges,
        "\\timing" => MetaCommand::ToggleTiming,
        "\\x" => MetaCommand::ToggleExpanded,
        "\\csv" => MetaCommand::ToggleCsv,
        "\\o" => MetaCommand::OutputFile(arg.filter(|s| !s.is_empty())),
        "\\i" => {
            if let Some(file) = arg.filter(|s| !s.is_empty()) {
                MetaCommand::InputFile(file)
            } else {
                MetaCommand::Unknown("\\i requires a filename".to_string())
            }
        }
        "\\q" => MetaCommand::Quit,
        "\\?" => MetaCommand::Help,
        other => MetaCommand::Unknown(other.to_string()),
    };

    Some(meta)
}

/// Converts a meta-command into a SQL query string for server-side execution.
/// Returns None for client-side-only commands.
pub fn meta_to_sql(cmd: &MetaCommand) -> Option<String> {
    match cmd {
        MetaCommand::ListTables => Some(
            "SELECT table_name, row_count, page_count, last_analyze FROM zyron_stat_tables"
                .to_string(),
        ),
        MetaCommand::DescribeTable(table) => Some(format!("SHOW COLUMNS FROM {}", table)),
        MetaCommand::ListIndexes => Some(
            "SELECT index_name, table_name, index_type, idx_scan FROM zyron_stat_indexes"
                .to_string(),
        ),
        MetaCommand::ListUsers => Some("SHOW users".to_string()),
        MetaCommand::ListPrivileges => Some("SHOW privileges".to_string()),
        MetaCommand::ToggleTiming
        | MetaCommand::ToggleExpanded
        | MetaCommand::ToggleCsv
        | MetaCommand::OutputFile(_)
        | MetaCommand::InputFile(_)
        | MetaCommand::Quit
        | MetaCommand::Help
        | MetaCommand::Unknown(_) => None,
    }
}
