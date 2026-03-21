//! Server-level session tracking.
//!
//! Tracks active client sessions for connection limit enforcement, idle timeout
//! detection, and administrative queries (pg_stat_activity equivalent).
//! Uses monotonic Instant for idle tracking to avoid system clock adjustment issues.

use std::sync::atomic::{AtomicU8, AtomicU64, Ordering};
use std::time::{Duration, Instant};

/// State of a client session.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum SessionState {
    /// Connected but not executing a query.
    Idle = 0,
    /// Currently executing a query.
    Active = 1,
    /// Inside a transaction block, waiting for next command.
    IdleInTransaction = 2,
}

impl SessionState {
    fn from_u8(val: u8) -> Self {
        match val {
            1 => Self::Active,
            2 => Self::IdleInTransaction,
            _ => Self::Idle,
        }
    }
}

/// Information about an active client session.
pub struct SessionInfo {
    pub process_id: i32,
    pub user: String,
    pub database: String,
    pub connected_at: Instant,
    /// Monotonic nanos since the manager's baseline Instant.
    /// Using Instant avoids issues with system clock adjustments.
    last_activity_nanos: AtomicU64,
    /// Session state (idle, active, in_transaction).
    state: AtomicU8,
}

impl SessionInfo {
    fn new(process_id: i32, user: String, database: String, baseline: Instant) -> Self {
        let nanos = baseline.elapsed().as_nanos() as u64;
        Self {
            process_id,
            user,
            database,
            connected_at: Instant::now(),
            last_activity_nanos: AtomicU64::new(nanos),
            state: AtomicU8::new(SessionState::Idle as u8),
        }
    }

    /// Updates the last activity timestamp to now (monotonic).
    pub fn touch(&self, baseline: Instant) {
        let nanos = baseline.elapsed().as_nanos() as u64;
        self.last_activity_nanos.store(nanos, Ordering::Release);
    }

    /// Returns monotonic nanos of last activity (relative to baseline).
    pub fn last_activity_nanos(&self) -> u64 {
        self.last_activity_nanos.load(Ordering::Acquire)
    }

    /// Sets the session state.
    pub fn set_state(&self, state: SessionState) {
        self.state.store(state as u8, Ordering::Release);
    }

    /// Returns the current session state.
    pub fn state(&self) -> SessionState {
        SessionState::from_u8(self.state.load(Ordering::Acquire))
    }

    /// Returns how long this session has been idle (since last activity).
    pub fn idle_duration(&self, baseline: Instant) -> Duration {
        let last_nanos = self.last_activity_nanos();
        let now_nanos = baseline.elapsed().as_nanos() as u64;
        Duration::from_nanos(now_nanos.saturating_sub(last_nanos))
    }
}

/// Tracks all active sessions on the server.
pub struct SessionManager {
    sessions: scc::HashMap<i32, SessionInfo>,
    max_connections: u32,
    idle_timeout: Duration,
    /// Monotonic baseline for all activity timestamps.
    baseline: Instant,
}

impl SessionManager {
    /// Creates a new session manager.
    pub fn new(max_connections: u32, idle_timeout_secs: u32) -> Self {
        Self {
            sessions: scc::HashMap::new(),
            max_connections,
            idle_timeout: Duration::from_secs(idle_timeout_secs as u64),
            baseline: Instant::now(),
        }
    }

    /// Registers a new session. Returns an error if the connection limit is
    /// reached or if the process_id is already registered.
    /// Uses entry_sync for atomic check-and-insert to prevent TOCTOU races.
    pub fn register(
        &self,
        process_id: i32,
        user: String,
        database: String,
    ) -> std::result::Result<(), String> {
        // Atomic check: reject duplicates and enforce connection limit.
        match self.sessions.entry_sync(process_id) {
            scc::hash_map::Entry::Occupied(_) => {
                return Err(format!("session {} already registered", process_id));
            }
            scc::hash_map::Entry::Vacant(entry) => {
                // Check connection limit after acquiring the vacant entry.
                // This is still not perfectly atomic with respect to concurrent
                // registrations, but the entry lock prevents duplicate IDs and
                // the count check is close enough (brief overshoot by 1-2 is
                // acceptable for connection limits).
                if self.sessions.len() as u32 >= self.max_connections {
                    return Err(format!(
                        "too many connections (max {})",
                        self.max_connections
                    ));
                }
                let info = SessionInfo::new(process_id, user, database, self.baseline);
                entry.insert_entry(info);
            }
        }
        Ok(())
    }

    /// Unregisters a session when the connection closes.
    pub fn unregister(&self, process_id: i32) {
        let _ = self.sessions.remove_sync(&process_id);
    }

    /// Updates the last activity timestamp for a session.
    pub fn touch(&self, process_id: i32) {
        let baseline = self.baseline;
        self.sessions.read_sync(&process_id, |_, info| {
            info.touch(baseline);
        });
    }

    /// Sets the state of a session.
    pub fn set_state(&self, process_id: i32, state: SessionState) {
        self.sessions.read_sync(&process_id, |_, info| {
            info.set_state(state);
        });
    }

    /// Returns the number of active sessions.
    pub fn active_count(&self) -> u32 {
        self.sessions.len() as u32
    }

    /// Returns the maximum connection limit.
    pub fn max_connections(&self) -> u32 {
        self.max_connections
    }

    /// Returns the configured idle timeout.
    pub fn idle_timeout(&self) -> Duration {
        self.idle_timeout
    }

    /// Collects process IDs of sessions that have been idle longer than the timeout.
    /// The caller is responsible for terminating these connections.
    pub fn collect_idle_sessions(&self) -> Vec<i32> {
        let mut idle = Vec::new();
        if self.idle_timeout.is_zero() {
            return idle;
        }
        let baseline = self.baseline;
        let timeout = self.idle_timeout;
        self.sessions.iter_sync(|pid, info| {
            if info.state() == SessionState::Idle && info.idle_duration(baseline) > timeout {
                idle.push(*pid);
            }
            true
        });
        idle
    }

    /// Iterates over all sessions, calling the provided function with each session's info.
    pub fn for_each<F>(&self, mut f: F)
    where
        F: FnMut(&SessionInfo),
    {
        self.sessions.iter_sync(|_, info| {
            f(info);
            true
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_register_unregister() {
        let mgr = SessionManager::new(100, 0);
        mgr.register(1, "admin".into(), "zyron".into()).unwrap();
        assert_eq!(mgr.active_count(), 1);
        mgr.register(2, "user1".into(), "zyron".into()).unwrap();
        assert_eq!(mgr.active_count(), 2);
        mgr.unregister(1);
        assert_eq!(mgr.active_count(), 1);
        mgr.unregister(2);
        assert_eq!(mgr.active_count(), 0);
    }

    #[test]
    fn test_max_connections() {
        let mgr = SessionManager::new(2, 0);
        mgr.register(1, "a".into(), "db".into()).unwrap();
        mgr.register(2, "b".into(), "db".into()).unwrap();
        let result = mgr.register(3, "c".into(), "db".into());
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("too many connections"));
    }

    #[test]
    fn test_duplicate_register_rejected() {
        let mgr = SessionManager::new(100, 0);
        mgr.register(1, "admin".into(), "db".into()).unwrap();
        let result = mgr.register(1, "other".into(), "db".into());
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("already registered"));
    }

    #[test]
    fn test_touch_updates_activity() {
        let mgr = SessionManager::new(100, 0);
        mgr.register(1, "admin".into(), "db".into()).unwrap();
        std::thread::sleep(Duration::from_millis(10));
        mgr.touch(1);
        mgr.sessions.read_sync(&1, |_, info| {
            let idle = info.idle_duration(mgr.baseline);
            assert!(idle < Duration::from_millis(100));
        });
    }

    #[test]
    fn test_session_state() {
        let mgr = SessionManager::new(100, 0);
        mgr.register(1, "admin".into(), "db".into()).unwrap();
        mgr.set_state(1, SessionState::Active);
        mgr.sessions.read_sync(&1, |_, info| {
            assert_eq!(info.state(), SessionState::Active);
        });
        mgr.set_state(1, SessionState::IdleInTransaction);
        mgr.sessions.read_sync(&1, |_, info| {
            assert_eq!(info.state(), SessionState::IdleInTransaction);
        });
    }

    #[test]
    fn test_idle_sessions_no_timeout() {
        let mgr = SessionManager::new(100, 0);
        mgr.register(1, "a".into(), "db".into()).unwrap();
        assert!(mgr.collect_idle_sessions().is_empty());
    }

    #[test]
    fn test_for_each() {
        let mgr = SessionManager::new(100, 0);
        mgr.register(1, "a".into(), "db1".into()).unwrap();
        mgr.register(2, "b".into(), "db2".into()).unwrap();
        let mut count = 0;
        mgr.for_each(|_| count += 1);
        assert_eq!(count, 2);
    }
}
