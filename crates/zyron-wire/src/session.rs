//! Per-connection session state.
//!
//! Tracks GUC (Grand Unified Configuration) variables, transaction state,
//! and connection identity (user, database) for each client connection.

use std::collections::HashMap;

use zyron_catalog::DatabaseId;
use zyron_common::ZyronError;

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
    /// Security context for privilege checks. None if auth system is not initialized.
    pub security_context: Option<zyron_auth::SecurityContext>,
    /// Per-session circuit breaker registry. Holds named breakers for use with
    /// CIRCUIT_BREAKER_STATUS('name') and ALTER CIRCUIT BREAKER 'name'.
    pub circuit_breakers: std::sync::Arc<zyron_types::resilience::CircuitBreakerRegistry>,
    /// Per-session rate limiter registry, keyed by user-supplied bucket name.
    pub rate_limiters: std::sync::Arc<zyron_types::scheduling::RateLimiterRegistry>,
    /// Per-session quota registry.
    pub quotas: std::sync::Arc<zyron_types::scheduling::QuotaRegistry>,
    /// Per-session sequence state backing currval and lastval. Shared into each
    /// query's execution context so currval('s') returns the value the
    /// session's last nextval('s') produced.
    pub sequence_state: std::sync::Arc<zyron_executor::sequence::SessionSeqState>,
}

impl Session {
    /// Creates a new session with default PostgreSQL-compatible parameters.
    pub fn new(user: String, database: String, database_id: DatabaseId) -> Self {
        Self::with_security_context(user, database, database_id, None)
    }

    /// Creates a new session with an optional security context for privilege checks.
    pub fn with_security_context(
        user: String,
        database: String,
        database_id: DatabaseId,
        security_context: Option<zyron_auth::SecurityContext>,
    ) -> Self {
        // PG wire protocol requires is_superuser parameter for client compatibility.
        // In Zyron, all access control flows through the RBAC/ABAC privilege system.
        // This value is for client display only and does not bypass any checks.
        let superuser_str = "on";

        let variables = HashMap::from([
            ("server_version".into(), String::from("16.0")),
            ("server_encoding".into(), String::from("UTF8")),
            ("client_encoding".into(), String::from("UTF8")),
            ("DateStyle".into(), String::from("ISO, MDY")),
            ("TimeZone".into(), String::from("UTC")),
            ("integer_datetimes".into(), String::from("on")),
            ("standard_conforming_strings".into(), String::from("on")),
            // Empty default: Zyron does not auto-create a user schema. The
            // client must set search_path or fully qualify identifiers.
            ("search_path".into(), String::new()),
            ("is_superuser".into(), String::from(superuser_str)),
            ("session_authorization".into(), user.clone()),
        ]);

        Self {
            variables,
            txn_state: TransactionState::Idle,
            database,
            user,
            database_id,
            search_path: Vec::new(),
            security_context,
            circuit_breakers: std::sync::Arc::new(
                zyron_types::resilience::CircuitBreakerRegistry::new(),
            ),
            rate_limiters: std::sync::Arc::new(zyron_types::scheduling::RateLimiterRegistry::new()),
            quotas: std::sync::Arc::new(zyron_types::scheduling::QuotaRegistry::new()),
            sequence_state: std::sync::Arc::new(zyron_executor::sequence::SessionSeqState::new()),
        }
    }

    /// Stable hash of the session's effective identity for plan-cache
    /// keying. RLS, ABAC, and column-security predicates are a function of
    /// the role, so a plan bound under one identity must never be served to
    /// another. Folds the user name, whether a security context is active, and
    /// the active role set so a secured session never collides with an
    /// unsecured one of the same name nor with the same login under a
    /// different role.
    ///
    /// Row-security policies are loaded at startup and have no live
    /// CREATE/ALTER/DROP path today, so the policy set is fixed for the
    /// process lifetime and need not be in the key. If runtime policy DDL is
    /// added, it must either bump catalog schema_version or fold a policy
    /// epoch into the cache key, otherwise stale plans would survive the change.
    pub fn identity_hash(&self) -> u64 {
        let mut h = zyron_common::hash64(self.user.as_bytes());
        // Row-security predicates are baked into the cached plan per effective
        // role, so the key must change when the active role set changes
        // (SET ROLE, or two sessions of the same login user under different
        // roles). Fold current_role and the effective-role set into the hash.
        if let Some(sc) = self.security_context.as_ref() {
            h ^= 0x9e37_79b9_7f4a_7c15;
            h = h
                .rotate_left(7)
                .wrapping_add(sc.current_role.0 as u64)
                .wrapping_mul(0x0100_0000_01b3);
            for role in &sc.effective_roles {
                h = h.rotate_left(5).wrapping_add(role.0 as u64);
            }
        }
        h
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

    /// Sets a session variable after validating the value for known keys.
    /// search_path is parsed into the planner's search_path vector. An invalid
    /// value for a validated key is rejected so the session never silently keeps
    /// a bad setting. Role identity changes (SET ROLE / SET SESSION
    /// AUTHORIZATION) are applied to security_context separately by the caller
    /// through `apply_role`, which resolves the role name against the role
    /// registry the session does not hold.
    pub fn set_variable(&mut self, name: String, value: String) -> Result<(), ZyronError> {
        match name.to_ascii_lowercase().as_str() {
            "search_path" => {
                self.search_path = parse_search_path(&value);
            }
            "client_encoding" => {
                // The server speaks UTF-8 only. Reject any other client
                // encoding instead of accepting it and then sending UTF-8.
                let v = value.trim().to_ascii_uppercase();
                if v != "UTF8" && v != "UTF-8" && v != "UNICODE" {
                    return Err(ZyronError::ConfigError(format!(
                        "unsupported client_encoding '{value}', only UTF8 is supported"
                    )));
                }
            }
            "timezone" => {
                if value.trim().is_empty() {
                    return Err(ZyronError::ConfigError(
                        "timezone must not be empty".to_string(),
                    ));
                }
            }
            "role" | "session_authorization" => {
                // A role/identity name must be present. The caller resolves it
                // against the role registry and applies it to security_context.
                if value.trim().is_empty() {
                    return Err(ZyronError::ConfigError(format!("{name} must name a role")));
                }
            }
            _ => {}
        }
        self.variables.insert(name, value);
        Ok(())
    }

    /// Applies a resolved role to the session security context so subsequent
    /// privilege checks run under the new role. SET ROLE NONE / RESET ROLE
    /// resets to the session login role. Errors if the target is not in the
    /// session's allowed role set or if no security context is active.
    pub fn apply_role(
        &mut self,
        target: Option<zyron_auth::RoleId>,
        hierarchy: &zyron_auth::RoleHierarchy,
    ) -> Result<(), ZyronError> {
        let sc = self.security_context.as_mut().ok_or_else(|| {
            ZyronError::ConfigError("no security context active for SET ROLE".to_string())
        })?;
        match target {
            Some(role_id) => sc.set_role(role_id, hierarchy),
            None => {
                sc.reset_role(hierarchy);
                Ok(())
            }
        }
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

        session
            .set_variable("TimeZone".into(), "US/Pacific".into())
            .unwrap();
        assert_eq!(session.get_variable("TimeZone"), Some("US/Pacific"));
    }

    #[test]
    fn test_set_search_path() {
        let mut session = test_session();
        session
            .set_variable("search_path".into(), "myschema, public".into())
            .unwrap();
        assert_eq!(session.search_path, vec!["myschema", "public"]);
    }

    #[test]
    fn test_search_path_with_user() {
        let mut session = test_session();
        session
            .set_variable("search_path".into(), "\"$user\", public, extra".into())
            .unwrap();
        assert_eq!(session.search_path, vec!["public", "extra"]);
    }

    #[test]
    fn test_set_variable_rejects_bad_encoding() {
        let mut session = test_session();
        assert!(
            session
                .set_variable("client_encoding".into(), "LATIN1".into())
                .is_err()
        );
        assert!(
            session
                .set_variable("client_encoding".into(), "UTF8".into())
                .is_ok()
        );
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
        assert!(session.search_path.is_empty());
        assert_eq!(session.get_variable("search_path"), Some(""));
    }
}
