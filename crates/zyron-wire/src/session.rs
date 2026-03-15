//! Per-connection session state.
//!
//! Tracks GUC (Grand Unified Configuration) variables, transaction state,
//! and connection identity (user, database) for each client connection.

use std::collections::HashMap;

use zyron_catalog::DatabaseId;

use crate::messages::TransactionState;

/// Per-connection session state holding variables and transaction status.
pub struct Session {
    /// Session variables (search_path, timezone, client_encoding, etc.).
    variables: HashMap<String, String>,
    /// Current transaction state for ReadyForQuery responses.
    txn_state: TransactionState,
    /// Current database name.
    pub database: String,
    /// Current user name.
    pub user: String,
    /// Database ID resolved from catalog at connection startup.
    pub database_id: DatabaseId,
    /// Parsed search path for the planner.
    pub search_path: Vec<String>,
}

impl Session {
    /// Creates a new session with default PostgreSQL-compatible parameters.
    pub fn new(user: String, database: String, database_id: DatabaseId) -> Self {
        let variables = HashMap::from([
            ("server_version".into(), String::from("16.0")),
            ("server_encoding".into(), String::from("UTF8")),
            ("client_encoding".into(), String::from("UTF8")),
            ("DateStyle".into(), String::from("ISO, MDY")),
            ("TimeZone".into(), String::from("UTC")),
            ("integer_datetimes".into(), String::from("on")),
            ("standard_conforming_strings".into(), String::from("on")),
            ("search_path".into(), String::from("\"$user\", public")),
            ("is_superuser".into(), String::from("on")),
            ("session_authorization".into(), user.clone()),
        ]);

        Self {
            variables,
            txn_state: TransactionState::Idle,
            database,
            user,
            database_id,
            search_path: vec!["public".into()],
        }
    }

    /// Returns the current transaction state.
    pub fn transaction_state(&self) -> TransactionState {
        self.txn_state
    }

    /// Updates the transaction state.
    pub fn set_transaction_state(&mut self, state: TransactionState) {
        self.txn_state = state;
    }

    /// Gets a session variable value.
    pub fn get_variable(&self, name: &str) -> Option<&str> {
        self.variables.get(name).map(|s| s.as_str())
    }

    /// Sets a session variable. Handles search_path updates by parsing
    /// the value into the search_path vector for the planner.
    pub fn set_variable(&mut self, name: String, value: String) {
        if name == "search_path" {
            self.search_path = parse_search_path(&value);
        }
        self.variables.insert(name, value);
    }

    /// Parameter keys sent during startup, in protocol order.
    const STARTUP_KEYS: [&str; 9] = [
        "server_version",
        "server_encoding",
        "client_encoding",
        "DateStyle",
        "TimeZone",
        "integer_datetimes",
        "standard_conforming_strings",
        "is_superuser",
        "session_authorization",
    ];

    /// Returns all parameter status pairs to send during the startup handshake.
    /// These tell the client about server configuration.
    /// Returns a fixed-size array (all keys are always present from Session::new).
    pub fn startup_parameters(&self) -> [(&str, &str); 9] {
        Self::STARTUP_KEYS.map(|key| {
            (
                key,
                self.variables.get(key).map(|v| v.as_str()).unwrap_or(""),
            )
        })
    }
}

/// Parses a search_path value like '"$user", public, myschema' into a Vec.
fn parse_search_path(value: &str) -> Vec<String> {
    value
        .split(',')
        .map(|s| s.trim().trim_matches('"').to_string())
        .filter(|s| !s.is_empty() && s != "$user")
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_session() -> Session {
        Session::new("testuser".into(), "testdb".into(), DatabaseId(1))
    }

    #[test]
    fn test_new_session_defaults() {
        let session = test_session();
        assert_eq!(session.user, "testuser");
        assert_eq!(session.database, "testdb");
        assert_eq!(session.database_id, DatabaseId(1));
        assert_eq!(session.transaction_state(), TransactionState::Idle);
        assert_eq!(session.get_variable("server_version"), Some("16.0"));
        assert_eq!(session.get_variable("server_encoding"), Some("UTF8"));
        assert_eq!(session.get_variable("integer_datetimes"), Some("on"));
    }

    #[test]
    fn test_set_transaction_state() {
        let mut session = test_session();
        session.set_transaction_state(TransactionState::InTransaction);
        assert_eq!(session.transaction_state(), TransactionState::InTransaction);

        session.set_transaction_state(TransactionState::Failed);
        assert_eq!(session.transaction_state(), TransactionState::Failed);
    }

    #[test]
    fn test_get_set_variable() {
        let mut session = test_session();
        assert_eq!(session.get_variable("nonexistent"), None);

        session.set_variable("TimeZone".into(), "US/Pacific".into());
        assert_eq!(session.get_variable("TimeZone"), Some("US/Pacific"));
    }

    #[test]
    fn test_set_search_path() {
        let mut session = test_session();
        session.set_variable("search_path".into(), "myschema, public".into());
        assert_eq!(session.search_path, vec!["myschema", "public"]);
    }

    #[test]
    fn test_search_path_with_user() {
        let mut session = test_session();
        session.set_variable("search_path".into(), "\"$user\", public, extra".into());
        assert_eq!(session.search_path, vec!["public", "extra"]);
    }

    #[test]
    fn test_startup_parameters() {
        let session = test_session();
        let params = session.startup_parameters();
        assert!(!params.is_empty());

        let names: Vec<&str> = params.iter().map(|(k, _)| *k).collect();
        assert!(names.contains(&"server_version"));
        assert!(names.contains(&"server_encoding"));
        assert!(names.contains(&"client_encoding"));
        assert!(names.contains(&"DateStyle"));
        assert!(names.contains(&"TimeZone"));
        assert!(names.contains(&"integer_datetimes"));
    }

    #[test]
    fn test_parse_search_path() {
        assert_eq!(
            parse_search_path("public, myschema"),
            vec!["public", "myschema"]
        );
        assert_eq!(parse_search_path("public"), vec!["public"]);
        assert_eq!(parse_search_path("\"$user\", public"), vec!["public"]);
    }

    #[test]
    fn test_default_search_path() {
        let session = test_session();
        assert_eq!(session.search_path, vec!["public"]);
    }
}
