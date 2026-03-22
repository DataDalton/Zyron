//! Brute force defense with rate limiting, account lockout signaling, and audit.
//!
//! Tracks failed authentication attempts per user and per IP using lock-free
//! sliding window counters. Signals account lockout when thresholds are exceeded.
//! Does not directly modify User state. Returns LockAction for the caller to apply.

use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};
use std::time::Duration;

use zyron_catalog::encoding::{
    read_bool, read_string, read_u8, read_u16, read_u32, read_u64, write_bool, write_string,
    write_u8, write_u16, write_u32, write_u64,
};
use zyron_common::{Result, ZyronError};

use crate::ip_management::{IpBlockSource, IpManager};
use crate::role::RoleId;

/// Returns the current time in seconds since the Unix epoch.
fn now_secs() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}

/// Returns the current time in milliseconds since the Unix epoch.
fn now_ms() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as u64
}

/// Policy controlling brute force defense thresholds and timing.
#[derive(Debug, Clone)]
pub struct BruteForcePolicy {
    pub lockout_threshold: u32,
    pub lockout_duration_secs: u64,
    pub ip_block_threshold: u32,
    pub failure_window_secs: u64,
    pub ip_block_duration_secs: u64,
    pub min_attempt_interval_ms: u64,
    pub lockout_enabled: bool,
}

impl Default for BruteForcePolicy {
    fn default() -> Self {
        Self {
            lockout_threshold: 5,
            lockout_duration_secs: 900,
            ip_block_threshold: 20,
            failure_window_secs: 300,
            ip_block_duration_secs: 3600,
            min_attempt_interval_ms: 100,
            lockout_enabled: true,
        }
    }
}

impl BruteForcePolicy {
    /// Serializes the policy to a byte vector.
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut buf = Vec::with_capacity(40);
        write_u32(&mut buf, self.lockout_threshold);
        write_u64(&mut buf, self.lockout_duration_secs);
        write_u32(&mut buf, self.ip_block_threshold);
        write_u64(&mut buf, self.failure_window_secs);
        write_u64(&mut buf, self.ip_block_duration_secs);
        write_u64(&mut buf, self.min_attempt_interval_ms);
        write_bool(&mut buf, self.lockout_enabled);
        buf
    }

    /// Deserializes the policy from bytes.
    pub fn from_bytes(data: &[u8]) -> Result<Self> {
        let mut offset = 0;
        let lockout_threshold = read_u32(data, &mut offset)?;
        let lockout_duration_secs = read_u64(data, &mut offset)?;
        let ip_block_threshold = read_u32(data, &mut offset)?;
        let failure_window_secs = read_u64(data, &mut offset)?;
        let ip_block_duration_secs = read_u64(data, &mut offset)?;
        let min_attempt_interval_ms = read_u64(data, &mut offset)?;
        let lockout_enabled = read_bool(data, &mut offset)?;
        Ok(Self {
            lockout_threshold,
            lockout_duration_secs,
            ip_block_threshold,
            failure_window_secs,
            ip_block_duration_secs,
            min_attempt_interval_ms,
            lockout_enabled,
        })
    }
}

/// Binds a brute force policy to a specific role and/or database pattern.
#[derive(Debug, Clone)]
pub struct BruteForcePolicyBinding {
    pub policy: BruteForcePolicy,
    pub role_id: Option<RoleId>,
    pub database_pattern: Option<String>,
    pub priority: u16,
}

impl BruteForcePolicyBinding {
    /// Serializes the binding to a byte vector.
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut buf = Vec::with_capacity(64);
        let policy_bytes = self.policy.to_bytes();
        write_u32(&mut buf, policy_bytes.len() as u32);
        buf.extend_from_slice(&policy_bytes);
        match self.role_id {
            Some(rid) => {
                write_bool(&mut buf, true);
                write_u32(&mut buf, rid.0);
            }
            None => {
                write_bool(&mut buf, false);
            }
        }
        match &self.database_pattern {
            Some(pat) => {
                write_bool(&mut buf, true);
                write_string(&mut buf, pat);
            }
            None => {
                write_bool(&mut buf, false);
            }
        }
        write_u16(&mut buf, self.priority);
        buf
    }

    /// Deserializes the binding from bytes.
    pub fn from_bytes(data: &[u8]) -> Result<Self> {
        let mut offset = 0;
        let policy_len = read_u32(data, &mut offset)? as usize;
        if offset + policy_len > data.len() {
            return Err(ZyronError::DecodingFailed(
                "BruteForcePolicyBinding policy data truncated".to_string(),
            ));
        }
        let policy = BruteForcePolicy::from_bytes(&data[offset..offset + policy_len])?;
        offset += policy_len;
        let has_role = read_bool(data, &mut offset)?;
        let role_id = if has_role {
            Some(RoleId(read_u32(data, &mut offset)?))
        } else {
            None
        };
        let has_db = read_bool(data, &mut offset)?;
        let database_pattern = if has_db {
            Some(read_string(data, &mut offset)?)
        } else {
            None
        };
        let priority = read_u16(data, &mut offset)?;
        Ok(Self {
            policy,
            role_id,
            database_pattern,
            priority,
        })
    }
}

/// Sliding window failure tracker using lock-free atomics.
struct AttemptTracker {
    timestamps: Vec<AtomicU64>,
    cursor: AtomicU32,
    consecutive_failures: AtomicU32,
    last_attempt_ms: AtomicU64,
}

impl AttemptTracker {
    fn new(capacity: usize) -> Self {
        let mut timestamps = Vec::with_capacity(capacity);
        for _ in 0..capacity {
            timestamps.push(AtomicU64::new(0));
        }
        Self {
            timestamps,
            cursor: AtomicU32::new(0),
            consecutive_failures: AtomicU32::new(0),
            last_attempt_ms: AtomicU64::new(0),
        }
    }

    fn record(&self, timestamp_secs: u64, timestamp_ms: u64) {
        let pos = self.cursor.fetch_add(1, Ordering::Relaxed) as usize % self.timestamps.len();
        self.timestamps[pos].store(timestamp_secs, Ordering::Relaxed);
        self.consecutive_failures.fetch_add(1, Ordering::Relaxed);
        self.last_attempt_ms.store(timestamp_ms, Ordering::Relaxed);
    }

    fn count_in_window(&self, now_secs: u64, window_secs: u64) -> u32 {
        let cutoff = now_secs.saturating_sub(window_secs);
        let mut count = 0u32;
        for ts in &self.timestamps {
            let t = ts.load(Ordering::Relaxed);
            if t > cutoff && t <= now_secs {
                count += 1;
            }
        }
        count
    }

    fn consecutive(&self) -> u32 {
        self.consecutive_failures.load(Ordering::Relaxed)
    }

    fn reset_consecutive(&self) {
        self.consecutive_failures.store(0, Ordering::Relaxed);
    }

    fn last_attempt_ms(&self) -> u64 {
        self.last_attempt_ms.load(Ordering::Relaxed)
    }
}

/// Result of pre-authentication gate check.
#[derive(Debug, Clone)]
pub enum AuthGate {
    Proceed,
    Delayed(Duration),
    Blocked(String),
}

/// Action the caller should take to lock a user account.
#[derive(Debug, Clone)]
pub enum LockAction {
    LockUser { reason: String, duration_secs: u64 },
}

/// Audit trail entry for an authentication attempt.
#[derive(Debug, Clone)]
pub struct AttemptEntry {
    pub timestamp: u64,
    pub ip: String,
    pub user: String,
    pub database: String,
    pub auth_method: u8,
    pub success: bool,
    pub gate_result: u8,
    pub failure_reason: Option<String>,
}

impl AttemptEntry {
    /// Serializes the entry to a byte vector.
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut buf = Vec::with_capacity(128);
        write_u64(&mut buf, self.timestamp);
        write_string(&mut buf, &self.ip);
        write_string(&mut buf, &self.user);
        write_string(&mut buf, &self.database);
        write_u8(&mut buf, self.auth_method);
        write_bool(&mut buf, self.success);
        write_u8(&mut buf, self.gate_result);
        match &self.failure_reason {
            Some(r) => {
                write_bool(&mut buf, true);
                write_string(&mut buf, r);
            }
            None => {
                write_bool(&mut buf, false);
            }
        }
        buf
    }

    /// Deserializes the entry from bytes.
    pub fn from_bytes(data: &[u8]) -> Result<Self> {
        let mut offset = 0;
        let timestamp = read_u64(data, &mut offset)?;
        let ip = read_string(data, &mut offset)?;
        let user = read_string(data, &mut offset)?;
        let database = read_string(data, &mut offset)?;
        let auth_method = read_u8(data, &mut offset)?;
        let success = read_bool(data, &mut offset)?;
        let gate_result = read_u8(data, &mut offset)?;
        let has_reason = read_bool(data, &mut offset)?;
        let failure_reason = if has_reason {
            Some(read_string(data, &mut offset)?)
        } else {
            None
        };
        Ok(Self {
            timestamp,
            ip,
            user,
            database,
            auth_method,
            success,
            gate_result,
            failure_reason,
        })
    }
}

/// Fixed-capacity circular buffer for audit entries.
struct AuditRing {
    entries: Vec<Option<AttemptEntry>>,
    cursor: usize,
    capacity: usize,
}

impl AuditRing {
    fn new(capacity: usize) -> Self {
        let mut entries = Vec::with_capacity(capacity);
        for _ in 0..capacity {
            entries.push(None);
        }
        Self {
            entries,
            cursor: 0,
            capacity,
        }
    }

    fn push(&mut self, entry: AttemptEntry) {
        let pos = self.cursor % self.capacity;
        self.entries[pos] = Some(entry);
        self.cursor += 1;
    }

    fn recent(&self, limit: usize) -> Vec<AttemptEntry> {
        let count = limit.min(self.cursor).min(self.capacity);
        let mut result = Vec::with_capacity(count);
        for i in 0..count {
            let idx = if self.cursor >= count {
                (self.cursor - count + i) % self.capacity
            } else {
                i % self.capacity
            };
            if let Some(entry) = &self.entries[idx] {
                result.push(entry.clone());
            }
        }
        result
    }
}

/// Brute force defense manager with per-user and per-IP tracking.
pub struct BruteForceManager {
    global_policy: parking_lot::RwLock<BruteForcePolicy>,
    policy_bindings: parking_lot::RwLock<Vec<BruteForcePolicyBinding>>,
    user_attempts: scc::HashMap<String, AttemptTracker>,
    ip_attempts: scc::HashMap<String, AttemptTracker>,
    /// Tracks how many distinct accounts have been locked from each IP.
    /// When an IP causes 2+ account lockouts, it gets auto-blocked.
    /// One lockout could be a user mistake. Two is a pattern.
    locked_accounts_per_ip: scc::HashMap<String, AtomicU32>,
    audit_ring: parking_lot::Mutex<AuditRing>,
}

impl BruteForceManager {
    /// Creates a new BruteForceManager with default policy and 1024-entry audit ring.
    pub fn new() -> Self {
        Self {
            global_policy: parking_lot::RwLock::new(BruteForcePolicy::default()),
            policy_bindings: parking_lot::RwLock::new(Vec::new()),
            user_attempts: scc::HashMap::new(),
            ip_attempts: scc::HashMap::new(),
            locked_accounts_per_ip: scc::HashMap::new(),
            audit_ring: parking_lot::Mutex::new(AuditRing::new(1024)),
        }
    }

    /// Checks whether an authentication attempt should proceed, be delayed, or be blocked.
    /// auth_method values: 0 = Trust, 6 = Certificate (both skip all checks).
    pub fn check_allowed(
        &self,
        ip: &str,
        _user_name: &str,
        _database: &str,
        auth_method: u8,
        ip_manager: &IpManager,
        user_locked: bool,
        locked_reason: Option<&str>,
    ) -> AuthGate {
        // Certificate and Trust skip all checks
        if auth_method == 6 || auth_method == 0 {
            return AuthGate::Proceed;
        }

        // Trusted IPs bypass IP-based checks
        let ip_trusted = ip_manager.is_trusted(ip);

        if !ip_trusted {
            // Check if IP is blocked
            if let Some(block) = ip_manager.is_blocked(ip) {
                return AuthGate::Blocked(format!("IP blocked: {}", block.reason));
            }
        }

        // Check if user account is locked
        if user_locked {
            let reason = locked_reason.unwrap_or("Account locked");
            return AuthGate::Blocked(format!("User locked: {}", reason));
        }

        // Rate limit check on IP
        if !ip_trusted {
            let policy = self.global_policy.read();
            let min_interval = policy.min_attempt_interval_ms;
            if min_interval > 0 {
                let current_ms = now_ms();
                let mut last_ms = 0u64;
                self.ip_attempts.read_sync(ip, |_, tracker| {
                    last_ms = tracker.last_attempt_ms();
                });
                if last_ms > 0 && current_ms.saturating_sub(last_ms) < min_interval {
                    let wait = min_interval - (current_ms - last_ms);
                    return AuthGate::Delayed(Duration::from_millis(wait));
                }
            }
        }

        AuthGate::Proceed
    }

    /// Records a failed authentication attempt. Returns a LockAction if thresholds are exceeded.
    pub fn record_failure(
        &self,
        ip: &str,
        user_name: &str,
        database: &str,
        auth_method: u8,
        reason: &str,
        ip_manager: &IpManager,
    ) -> Option<LockAction> {
        let ts = now_secs();
        let ms = now_ms();
        let policy = self.global_policy.read().clone();

        let user_capacity = (policy.lockout_threshold as usize).max(8);
        let ip_capacity = (policy.ip_block_threshold as usize).max(8);

        // Record in user tracker
        match self.user_attempts.entry_sync(user_name.to_string()) {
            scc::hash_map::Entry::Occupied(occ) => {
                occ.get().record(ts, ms);
            }
            scc::hash_map::Entry::Vacant(vac) => {
                let tracker = AttemptTracker::new(user_capacity);
                tracker.record(ts, ms);
                vac.insert_entry(tracker);
            }
        }

        // Record in IP tracker
        match self.ip_attempts.entry_sync(ip.to_string()) {
            scc::hash_map::Entry::Occupied(occ) => {
                occ.get().record(ts, ms);
            }
            scc::hash_map::Entry::Vacant(vac) => {
                let tracker = AttemptTracker::new(ip_capacity);
                tracker.record(ts, ms);
                vac.insert_entry(tracker);
            }
        }

        // Check user consecutive failures against lockout threshold
        let mut lock_action = None;
        if policy.lockout_enabled {
            self.user_attempts.read_sync(user_name, |_, tracker| {
                if tracker.consecutive() >= policy.lockout_threshold {
                    lock_action = Some(LockAction::LockUser {
                        reason: format!(
                            "Exceeded {} consecutive failed attempts",
                            policy.lockout_threshold
                        ),
                        duration_secs: policy.lockout_duration_secs,
                    });
                }
            });
        }

        // Check IP failures in window
        self.ip_attempts.read_sync(ip, |_, tracker| {
            let ip_failures = tracker.count_in_window(ts, policy.failure_window_secs);
            if ip_failures >= policy.ip_block_threshold {
                let expires = now_secs() + policy.ip_block_duration_secs;
                ip_manager.block_ip(
                    ip.to_string(),
                    expires,
                    format!(
                        "Exceeded {} failures in {} seconds",
                        policy.ip_block_threshold, policy.failure_window_secs
                    ),
                    IpBlockSource::BruteForce,
                );
            }
        });

        // Push audit entry
        {
            let mut ring = self.audit_ring.lock();
            ring.push(AttemptEntry {
                timestamp: ts,
                ip: ip.to_string(),
                user: user_name.to_string(),
                database: database.to_string(),
                auth_method,
                success: false,
                gate_result: 0,
                failure_reason: Some(reason.to_string()),
            });
        }

        lock_action
    }

    /// Records a successful authentication and resets the user's consecutive failure counter.
    pub fn record_success(&self, ip: &str, user_name: &str, database: &str) {
        let ts = now_secs();

        // Reset consecutive failures
        self.user_attempts.read_sync(user_name, |_, tracker| {
            tracker.reset_consecutive();
        });

        // Push audit entry
        {
            let mut ring = self.audit_ring.lock();
            ring.push(AttemptEntry {
                timestamp: ts,
                ip: ip.to_string(),
                user: user_name.to_string(),
                database: database.to_string(),
                auth_method: 0,
                success: true,
                gate_result: 0,
                failure_reason: None,
            });
        }
    }

    /// Called after the caller has locked a user account due to a LockAction.
    /// Tracks how many distinct accounts this IP has caused to lock. If 2 or more
    /// accounts have been locked from the same IP, the IP gets auto-blocked.
    /// One account lockout could be a user mistake. Two is an attack pattern.
    pub fn report_lockout(&self, ip: &str, user_name: &str, ip_manager: &IpManager) {
        let count = match self.locked_accounts_per_ip.entry_sync(ip.to_string()) {
            scc::hash_map::Entry::Occupied(occ) => occ.get().fetch_add(1, Ordering::Relaxed) + 1,
            scc::hash_map::Entry::Vacant(vac) => {
                vac.insert_entry(AtomicU32::new(1));
                1
            }
        };

        if count >= 2 {
            let policy = self.global_policy.read();
            let expires = now_secs() + policy.ip_block_duration_secs;
            ip_manager.block_ip(
                ip.to_string(),
                expires,
                format!(
                    "Auto-blocked: {} accounts locked from this IP (triggered by user '{}')",
                    count, user_name
                ),
                IpBlockSource::BruteForce,
            );
        }
    }

    /// Sets the global brute force policy.
    pub fn set_global_policy(&self, policy: BruteForcePolicy) {
        *self.global_policy.write() = policy;
    }

    /// Adds a policy binding. Bindings are kept sorted by priority (ascending).
    pub fn add_policy_binding(&self, binding: BruteForcePolicyBinding) {
        let mut bindings = self.policy_bindings.write();
        bindings.push(binding);
        bindings.sort_by_key(|b| b.priority);
    }

    /// Removes a policy binding by priority. Returns true if found.
    pub fn remove_policy_binding(&self, priority: u16) -> bool {
        let mut bindings = self.policy_bindings.write();
        let len_before = bindings.len();
        bindings.retain(|b| b.priority != priority);
        bindings.len() < len_before
    }

    /// Resets the consecutive failure counter for a user.
    pub fn reset_user_failures(&self, user_name: &str) {
        self.user_attempts.read_sync(user_name, |_, tracker| {
            tracker.reset_consecutive();
        });
    }

    /// Returns recent audit entries up to the given limit.
    pub fn attempt_log(&self, limit: usize) -> Vec<AttemptEntry> {
        let ring = self.audit_ring.lock();
        ring.recent(limit)
    }

    /// Returns the consecutive failure count for a user.
    pub fn user_failure_count(&self, user_name: &str) -> u32 {
        let mut count = 0u32;
        self.user_attempts.read_sync(user_name, |_, tracker| {
            count = tracker.consecutive();
        });
        count
    }

    /// Returns the failure count for an IP within the current policy window.
    pub fn ip_failure_count(&self, ip: &str) -> u32 {
        let policy = self.global_policy.read();
        let window = policy.failure_window_secs;
        let ts = now_secs();
        let mut count = 0u32;
        self.ip_attempts.read_sync(ip, |_, tracker| {
            count = tracker.count_in_window(ts, window);
        });
        count
    }

    /// Returns the effective policy for a set of roles and database.
    pub fn effective_policy(&self, effective_roles: &[RoleId], database: &str) -> BruteForcePolicy {
        self.resolve_policy(effective_roles, database)
    }

    /// Returns a clone of all policy bindings.
    pub fn policy_bindings(&self) -> Vec<BruteForcePolicyBinding> {
        self.policy_bindings.read().clone()
    }

    /// Loads policy bindings from storage (replaces existing).
    pub fn load_policy_bindings(&self, bindings: Vec<BruteForcePolicyBinding>) {
        let mut store = self.policy_bindings.write();
        *store = bindings;
        store.sort_by_key(|b| b.priority);
    }

    /// Exports policy bindings for persistence.
    pub fn export_policy_bindings(&self) -> Vec<BruteForcePolicyBinding> {
        self.policy_bindings.read().clone()
    }

    /// Resolves the effective policy by checking bindings in priority order.
    /// Returns the first matching binding's policy, or the global policy as fallback.
    fn resolve_policy(&self, effective_roles: &[RoleId], database: &str) -> BruteForcePolicy {
        let bindings = self.policy_bindings.read();
        for binding in bindings.iter() {
            // Check role match
            let role_matches = match binding.role_id {
                Some(rid) => effective_roles.contains(&rid),
                None => true,
            };
            if !role_matches {
                continue;
            }
            // Check database pattern match
            let db_matches = match &binding.database_pattern {
                Some(pattern) => {
                    if pattern == "*" {
                        true
                    } else {
                        pattern == database
                    }
                }
                None => true,
            };
            if db_matches {
                return binding.policy.clone();
            }
        }
        self.global_policy.read().clone()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ip_management::{IpManager, TrustedIpEntry};
    use crate::role::UserId;

    fn make_ip_manager() -> IpManager {
        IpManager::new()
    }

    #[test]
    fn test_check_allowed_normal() {
        let mgr = BruteForceManager::new();
        let ip_mgr = make_ip_manager();
        let gate = mgr.check_allowed("10.0.0.1", "alice", "mydb", 2, &ip_mgr, false, None);
        assert!(matches!(gate, AuthGate::Proceed));
    }

    #[test]
    fn test_check_allowed_user_locked() {
        let mgr = BruteForceManager::new();
        let ip_mgr = make_ip_manager();
        let gate = mgr.check_allowed(
            "10.0.0.1",
            "alice",
            "mydb",
            2,
            &ip_mgr,
            true,
            Some("too many failures"),
        );
        assert!(matches!(gate, AuthGate::Blocked(_)));
    }

    #[test]
    fn test_check_allowed_ip_blocked() {
        let mgr = BruteForceManager::new();
        let ip_mgr = make_ip_manager();
        ip_mgr.block_ip(
            "10.0.0.1".to_string(),
            0,
            "bad actor".to_string(),
            IpBlockSource::Admin(UserId(1)),
        );
        let gate = mgr.check_allowed("10.0.0.1", "alice", "mydb", 2, &ip_mgr, false, None);
        assert!(matches!(gate, AuthGate::Blocked(_)));
    }

    #[test]
    fn test_check_allowed_trusted_ip_bypasses_block() {
        let mgr = BruteForceManager::new();
        let ip_mgr = make_ip_manager();
        ip_mgr.block_ip(
            "10.0.0.1".to_string(),
            0,
            "bad actor".to_string(),
            IpBlockSource::Admin(UserId(1)),
        );
        ip_mgr.add_trusted(TrustedIpEntry {
            ip_or_cidr: "10.0.0.0/24".to_string(),
            added_by: UserId(1),
            added_at: 1000,
            reason: "internal".to_string(),
        });
        let gate = mgr.check_allowed("10.0.0.1", "alice", "mydb", 2, &ip_mgr, false, None);
        assert!(matches!(gate, AuthGate::Proceed));
    }

    #[test]
    fn test_check_allowed_rate_limited() {
        let mgr = BruteForceManager::new();
        let ip_mgr = make_ip_manager();
        // Set a high min_attempt_interval_ms
        mgr.set_global_policy(BruteForcePolicy {
            min_attempt_interval_ms: 60_000, // 60 seconds
            ..BruteForcePolicy::default()
        });
        // Record a failure to set last_attempt_ms
        mgr.record_failure("10.0.0.1", "alice", "mydb", 2, "bad password", &ip_mgr);
        // Next check should be rate limited
        let gate = mgr.check_allowed("10.0.0.1", "alice", "mydb", 2, &ip_mgr, false, None);
        assert!(matches!(gate, AuthGate::Delayed(_)));
    }

    #[test]
    fn test_check_allowed_certificate_skips_checks() {
        let mgr = BruteForceManager::new();
        let ip_mgr = make_ip_manager();
        ip_mgr.block_ip(
            "10.0.0.1".to_string(),
            0,
            "blocked".to_string(),
            IpBlockSource::Admin(UserId(1)),
        );
        // auth_method=6 (Certificate) skips all checks
        let gate = mgr.check_allowed(
            "10.0.0.1",
            "alice",
            "mydb",
            6,
            &ip_mgr,
            true,
            Some("locked"),
        );
        assert!(matches!(gate, AuthGate::Proceed));
    }

    #[test]
    fn test_record_failure_below_threshold() {
        let mgr = BruteForceManager::new();
        let ip_mgr = make_ip_manager();
        let action = mgr.record_failure("10.0.0.1", "alice", "mydb", 2, "wrong pw", &ip_mgr);
        assert!(action.is_none());
    }

    #[test]
    fn test_record_failure_at_threshold() {
        let mgr = BruteForceManager::new();
        let ip_mgr = make_ip_manager();
        mgr.set_global_policy(BruteForcePolicy {
            lockout_threshold: 3,
            lockout_enabled: true,
            ..BruteForcePolicy::default()
        });
        mgr.record_failure("10.0.0.1", "alice", "mydb", 2, "wrong pw", &ip_mgr);
        mgr.record_failure("10.0.0.1", "alice", "mydb", 2, "wrong pw", &ip_mgr);
        let action = mgr.record_failure("10.0.0.1", "alice", "mydb", 2, "wrong pw", &ip_mgr);
        assert!(action.is_some());
        match action.unwrap() {
            LockAction::LockUser { reason, .. } => {
                assert!(reason.contains("3"));
            }
        }
    }

    #[test]
    fn test_record_failure_ip_auto_block() {
        let mgr = BruteForceManager::new();
        let ip_mgr = make_ip_manager();
        mgr.set_global_policy(BruteForcePolicy {
            ip_block_threshold: 3,
            failure_window_secs: 3600,
            lockout_threshold: 100, // high so user lockout does not trigger
            ..BruteForcePolicy::default()
        });
        // Use different users to avoid user lockout, same IP
        mgr.record_failure("10.0.0.1", "alice", "mydb", 2, "wrong", &ip_mgr);
        mgr.record_failure("10.0.0.1", "bob", "mydb", 2, "wrong", &ip_mgr);
        mgr.record_failure("10.0.0.1", "carol", "mydb", 2, "wrong", &ip_mgr);
        // IP should now be blocked
        assert!(ip_mgr.is_blocked("10.0.0.1").is_some());
    }

    #[test]
    fn test_record_success_resets_consecutive() {
        let mgr = BruteForceManager::new();
        let ip_mgr = make_ip_manager();
        mgr.record_failure("10.0.0.1", "alice", "mydb", 2, "wrong", &ip_mgr);
        mgr.record_failure("10.0.0.1", "alice", "mydb", 2, "wrong", &ip_mgr);
        assert_eq!(mgr.user_failure_count("alice"), 2);
        mgr.record_success("10.0.0.1", "alice", "mydb");
        assert_eq!(mgr.user_failure_count("alice"), 0);
    }

    #[test]
    fn test_audit_ring_push_and_recent() {
        let mut ring = AuditRing::new(4);
        for i in 0..3 {
            ring.push(AttemptEntry {
                timestamp: i as u64,
                ip: "10.0.0.1".to_string(),
                user: "alice".to_string(),
                database: "db".to_string(),
                auth_method: 2,
                success: false,
                gate_result: 0,
                failure_reason: Some(format!("attempt {}", i)),
            });
        }
        let recent = ring.recent(10);
        assert_eq!(recent.len(), 3);
        assert_eq!(recent[0].timestamp, 0);
        assert_eq!(recent[2].timestamp, 2);
    }

    #[test]
    fn test_audit_ring_wraps() {
        let mut ring = AuditRing::new(3);
        for i in 0..5 {
            ring.push(AttemptEntry {
                timestamp: i as u64,
                ip: "10.0.0.1".to_string(),
                user: "alice".to_string(),
                database: "db".to_string(),
                auth_method: 2,
                success: false,
                gate_result: 0,
                failure_reason: None,
            });
        }
        let recent = ring.recent(10);
        // Only 3 entries fit, should have timestamps 2, 3, 4
        assert_eq!(recent.len(), 3);
        assert_eq!(recent[0].timestamp, 2);
        assert_eq!(recent[1].timestamp, 3);
        assert_eq!(recent[2].timestamp, 4);
    }

    #[test]
    fn test_attempt_tracker_sliding_window() {
        let tracker = AttemptTracker::new(8);
        // Record failures at various "times"
        tracker.timestamps[0].store(100, Ordering::Relaxed);
        tracker.timestamps[1].store(200, Ordering::Relaxed);
        tracker.timestamps[2].store(300, Ordering::Relaxed);
        tracker.timestamps[3].store(400, Ordering::Relaxed);
        // Count in window [250, 400]
        let count = tracker.count_in_window(400, 150);
        assert_eq!(count, 2); // timestamps 300 and 400
    }

    #[test]
    fn test_attempt_tracker_consecutive_reset() {
        let tracker = AttemptTracker::new(8);
        tracker.record(100, 100000);
        tracker.record(101, 101000);
        assert_eq!(tracker.consecutive(), 2);
        tracker.reset_consecutive();
        assert_eq!(tracker.consecutive(), 0);
    }

    #[test]
    fn test_policy_default() {
        let policy = BruteForcePolicy::default();
        assert_eq!(policy.lockout_threshold, 5);
        assert_eq!(policy.lockout_duration_secs, 900);
        assert_eq!(policy.ip_block_threshold, 20);
        assert_eq!(policy.failure_window_secs, 300);
        assert_eq!(policy.ip_block_duration_secs, 3600);
        assert_eq!(policy.min_attempt_interval_ms, 100);
        assert!(policy.lockout_enabled);
    }

    #[test]
    fn test_policy_roundtrip() {
        let policy = BruteForcePolicy {
            lockout_threshold: 10,
            lockout_duration_secs: 1800,
            ip_block_threshold: 50,
            failure_window_secs: 600,
            ip_block_duration_secs: 7200,
            min_attempt_interval_ms: 250,
            lockout_enabled: false,
        };
        let bytes = policy.to_bytes();
        let decoded = BruteForcePolicy::from_bytes(&bytes).unwrap();
        assert_eq!(decoded.lockout_threshold, 10);
        assert_eq!(decoded.lockout_duration_secs, 1800);
        assert_eq!(decoded.ip_block_threshold, 50);
        assert_eq!(decoded.failure_window_secs, 600);
        assert_eq!(decoded.ip_block_duration_secs, 7200);
        assert_eq!(decoded.min_attempt_interval_ms, 250);
        assert!(!decoded.lockout_enabled);
    }

    #[test]
    fn test_policy_binding_roundtrip() {
        let binding = BruteForcePolicyBinding {
            policy: BruteForcePolicy {
                lockout_threshold: 3,
                ..BruteForcePolicy::default()
            },
            role_id: Some(RoleId(42)),
            database_pattern: Some("production_*".to_string()),
            priority: 10,
        };
        let bytes = binding.to_bytes();
        let decoded = BruteForcePolicyBinding::from_bytes(&bytes).unwrap();
        assert_eq!(decoded.policy.lockout_threshold, 3);
        assert_eq!(decoded.role_id, Some(RoleId(42)));
        assert_eq!(decoded.database_pattern.as_deref(), Some("production_*"));
        assert_eq!(decoded.priority, 10);
    }

    #[test]
    fn test_attempt_entry_roundtrip() {
        let entry = AttemptEntry {
            timestamp: 1700000000,
            ip: "10.0.0.1".to_string(),
            user: "alice".to_string(),
            database: "mydb".to_string(),
            auth_method: 4,
            success: false,
            gate_result: 1,
            failure_reason: Some("invalid password".to_string()),
        };
        let bytes = entry.to_bytes();
        let decoded = AttemptEntry::from_bytes(&bytes).unwrap();
        assert_eq!(decoded.timestamp, entry.timestamp);
        assert_eq!(decoded.ip, entry.ip);
        assert_eq!(decoded.user, entry.user);
        assert_eq!(decoded.database, entry.database);
        assert_eq!(decoded.auth_method, entry.auth_method);
        assert_eq!(decoded.success, entry.success);
        assert_eq!(decoded.gate_result, entry.gate_result);
        assert_eq!(decoded.failure_reason, entry.failure_reason);
    }

    #[test]
    fn test_resolve_policy_global_fallback() {
        let mgr = BruteForceManager::new();
        mgr.set_global_policy(BruteForcePolicy {
            lockout_threshold: 7,
            ..BruteForcePolicy::default()
        });
        let policy = mgr.resolve_policy(&[RoleId(1)], "mydb");
        assert_eq!(policy.lockout_threshold, 7);
    }

    #[test]
    fn test_resolve_policy_role_override() {
        let mgr = BruteForceManager::new();
        mgr.set_global_policy(BruteForcePolicy {
            lockout_threshold: 5,
            ..BruteForcePolicy::default()
        });
        mgr.add_policy_binding(BruteForcePolicyBinding {
            policy: BruteForcePolicy {
                lockout_threshold: 3,
                ..BruteForcePolicy::default()
            },
            role_id: Some(RoleId(42)),
            database_pattern: None,
            priority: 1,
        });
        // Role 42 should get the override
        let policy = mgr.resolve_policy(&[RoleId(42)], "mydb");
        assert_eq!(policy.lockout_threshold, 3);
        // Role 99 should get global fallback
        let policy = mgr.resolve_policy(&[RoleId(99)], "mydb");
        assert_eq!(policy.lockout_threshold, 5);
    }

    #[test]
    fn test_lockout_disabled() {
        let mgr = BruteForceManager::new();
        let ip_mgr = make_ip_manager();
        mgr.set_global_policy(BruteForcePolicy {
            lockout_threshold: 2,
            lockout_enabled: false,
            ip_block_threshold: 100, // high to avoid IP block
            ..BruteForcePolicy::default()
        });
        mgr.record_failure("10.0.0.1", "alice", "mydb", 2, "wrong", &ip_mgr);
        let action = mgr.record_failure("10.0.0.1", "alice", "mydb", 2, "wrong", &ip_mgr);
        assert!(
            action.is_none(),
            "Lockout disabled should not produce LockAction"
        );
    }

    #[test]
    fn test_report_lockout_single_account_no_ip_block() {
        let mgr = BruteForceManager::new();
        let ip_mgr = make_ip_manager();
        // One account lockout from an IP should NOT trigger IP ban.
        mgr.report_lockout("10.0.0.1", "alice", &ip_mgr);
        assert!(ip_mgr.is_blocked("10.0.0.1").is_none());
    }

    #[test]
    fn test_report_lockout_two_accounts_triggers_ip_block() {
        let mgr = BruteForceManager::new();
        let ip_mgr = make_ip_manager();
        // First account lockout, no IP ban.
        mgr.report_lockout("10.0.0.1", "alice", &ip_mgr);
        assert!(ip_mgr.is_blocked("10.0.0.1").is_none());
        // Second account lockout from same IP, IP gets banned.
        mgr.report_lockout("10.0.0.1", "bob", &ip_mgr);
        let block = ip_mgr.is_blocked("10.0.0.1");
        assert!(block.is_some());
        let entry = block.unwrap();
        assert!(entry.reason.contains("2 accounts locked"));
        assert!(entry.reason.contains("bob"));
    }

    #[test]
    fn test_report_lockout_different_ips_no_cross_contamination() {
        let mgr = BruteForceManager::new();
        let ip_mgr = make_ip_manager();
        // One lockout each from different IPs. Neither should be blocked.
        mgr.report_lockout("10.0.0.1", "alice", &ip_mgr);
        mgr.report_lockout("10.0.0.2", "bob", &ip_mgr);
        assert!(ip_mgr.is_blocked("10.0.0.1").is_none());
        assert!(ip_mgr.is_blocked("10.0.0.2").is_none());
    }

    #[test]
    fn test_report_lockout_three_accounts_still_blocked() {
        let mgr = BruteForceManager::new();
        let ip_mgr = make_ip_manager();
        mgr.report_lockout("10.0.0.1", "alice", &ip_mgr);
        mgr.report_lockout("10.0.0.1", "bob", &ip_mgr);
        mgr.report_lockout("10.0.0.1", "carol", &ip_mgr);
        // IP was blocked on the 2nd lockout and stays blocked.
        let block = ip_mgr.is_blocked("10.0.0.1");
        assert!(block.is_some());
        let entry = block.unwrap();
        assert!(entry.reason.contains("2 accounts locked"));
    }

    #[test]
    fn test_full_flow_failures_then_lockout_then_ip_ban() {
        let mgr = BruteForceManager::new();
        let ip_mgr = make_ip_manager();
        mgr.set_global_policy(BruteForcePolicy {
            lockout_threshold: 3,
            lockout_enabled: true,
            ip_block_threshold: 100, // high so raw IP failure count does not trigger
            ..BruteForcePolicy::default()
        });

        // 3 failures on alice from 10.0.0.1 -> lock alice
        for _ in 0..2 {
            let action = mgr.record_failure("10.0.0.1", "alice", "mydb", 2, "wrong", &ip_mgr);
            assert!(action.is_none());
        }
        let action = mgr.record_failure("10.0.0.1", "alice", "mydb", 2, "wrong", &ip_mgr);
        assert!(action.is_some());
        mgr.report_lockout("10.0.0.1", "alice", &ip_mgr);
        // IP not yet blocked (only 1 account locked).
        assert!(ip_mgr.is_blocked("10.0.0.1").is_none());

        // 3 failures on bob from 10.0.0.1 -> lock bob
        for _ in 0..2 {
            let action = mgr.record_failure("10.0.0.1", "bob", "mydb", 2, "wrong", &ip_mgr);
            assert!(action.is_none());
        }
        let action = mgr.record_failure("10.0.0.1", "bob", "mydb", 2, "wrong", &ip_mgr);
        assert!(action.is_some());
        mgr.report_lockout("10.0.0.1", "bob", &ip_mgr);
        // NOW IP should be blocked (2 accounts locked from same IP).
        assert!(ip_mgr.is_blocked("10.0.0.1").is_some());
    }
}
