//! WORM / retention-lock enforcement helpers. A retention lock makes a table
//! immutable until an expiry time even for admins; the `immutable` flag is a
//! permanent variant. Enforced in the legal-hold DML hook composite.

use zyron_catalog::schema::TableEntry;

use crate::ttl::now_micros;

/// True when the table is currently write-protected by an active retention
/// lock or the permanent immutable flag.
pub fn write_locked(entry: &TableEntry) -> bool {
    if entry.lifecycle.immutable {
        return true;
    }
    let until = entry.lifecycle.retention_lock_until;
    until != 0 && until > now_micros()
}

/// Human-readable reason for a rejected mutation, for the error message.
pub fn lock_reason(entry: &TableEntry) -> String {
    if entry.lifecycle.immutable {
        return "table is immutable (WORM)".to_string();
    }
    format!(
        "retention lock active until epoch-micros {}",
        entry.lifecycle.retention_lock_until
    )
}
