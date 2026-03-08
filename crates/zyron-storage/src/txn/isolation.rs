//! Transaction isolation levels.

/// Isolation level for a transaction.
///
/// Determines when snapshots are refreshed:
/// - ReadCommitted: new snapshot per statement (caller calls refresh_snapshot)
/// - SnapshotIsolation: single snapshot for entire transaction lifetime
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum IsolationLevel {
    /// Each statement sees data committed before the statement began.
    /// New snapshot acquired per statement via TransactionManager::refresh_snapshot().
    ReadCommitted,
    /// All statements see a consistent snapshot taken at BEGIN time.
    /// Prevents dirty reads, non-repeatable reads, and phantom reads.
    #[default]
    SnapshotIsolation,
}

impl std::fmt::Display for IsolationLevel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::ReadCommitted => write!(f, "READ COMMITTED"),
            Self::SnapshotIsolation => write!(f, "SNAPSHOT ISOLATION"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_isolation() {
        assert_eq!(IsolationLevel::default(), IsolationLevel::SnapshotIsolation);
    }

    #[test]
    fn test_display() {
        assert_eq!(IsolationLevel::ReadCommitted.to_string(), "READ COMMITTED");
        assert_eq!(
            IsolationLevel::SnapshotIsolation.to_string(),
            "SNAPSHOT ISOLATION"
        );
    }

    #[test]
    fn test_equality() {
        assert_eq!(IsolationLevel::ReadCommitted, IsolationLevel::ReadCommitted);
        assert_ne!(
            IsolationLevel::ReadCommitted,
            IsolationLevel::SnapshotIsolation
        );
    }

    #[test]
    fn test_copy() {
        let level = IsolationLevel::SnapshotIsolation;
        let copied = level;
        assert_eq!(level, copied);
    }
}
