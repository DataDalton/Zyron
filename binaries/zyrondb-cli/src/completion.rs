// Tab completion for SQL keywords and meta-commands.

const SQL_KEYWORDS: &[&str] = &[
    "SELECT",
    "INSERT",
    "UPDATE",
    "DELETE",
    "CREATE",
    "DROP",
    "ALTER",
    "FROM",
    "WHERE",
    "JOIN",
    "LEFT",
    "RIGHT",
    "INNER",
    "OUTER",
    "ON",
    "AND",
    "OR",
    "NOT",
    "IN",
    "BETWEEN",
    "LIKE",
    "ORDER",
    "BY",
    "GROUP",
    "HAVING",
    "LIMIT",
    "OFFSET",
    "AS",
    "SET",
    "VALUES",
    "INTO",
    "TABLE",
    "INDEX",
    "VIEW",
    "SCHEMA",
    "BEGIN",
    "COMMIT",
    "ROLLBACK",
    "EXPLAIN",
    "ANALYZE",
    "VACUUM",
    "CHECKPOINT",
    "SHOW",
    "GRANT",
    "REVOKE",
];

const META_COMMANDS: &[&str] = &[
    "\\dt", "\\d", "\\di", "\\du", "\\dp", "\\timing", "\\x", "\\csv", "\\o", "\\i", "\\q", "\\?",
];

/// Returns completion candidates matching the last word in the input.
/// Matches SQL keywords (case insensitive) and meta-commands by prefix.
pub fn complete(input: &str) -> Vec<String> {
    let last_word = match input.split_whitespace().last() {
        Some(w) => w,
        None => return Vec::new(),
    };

    if last_word.is_empty() {
        return Vec::new();
    }

    let mut matches = Vec::new();

    if last_word.starts_with('\\') {
        // Match against meta-commands
        for cmd in META_COMMANDS {
            if cmd.starts_with(last_word) {
                matches.push(cmd.to_string());
            }
        }
    } else {
        // Match against SQL keywords (case insensitive prefix match)
        let upper = last_word.to_uppercase();
        for kw in SQL_KEYWORDS {
            if kw.starts_with(&upper) {
                matches.push(kw.to_string());
            }
        }
    }

    matches
}
