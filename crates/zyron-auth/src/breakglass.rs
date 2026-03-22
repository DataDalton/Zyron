//! Break-glass emergency access with auto-revocation.
//!
//! Provides time-limited emergency privilege escalation that is audited
//! and automatically revoked when the duration expires.

use crate::role::RoleId;
use zyron_common::{Result, ZyronError};

/// Returns the current Unix timestamp in seconds.
fn now_secs() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}

/// A single break-glass session granting temporary escalated access.
#[derive(Debug, Clone)]
pub struct BreakGlassSession {
    pub role_id: RoleId,
    /// Dormant roles that become active during break-glass.
    pub activated_roles: Vec<RoleId>,
    /// Additional privilege types granted during break-glass (scoped to object in the activation call).
    pub activated_privileges: Vec<crate::privilege::PrivilegeType>,
    /// Elevated clearance during break-glass (None = no change).
    pub elevated_clearance: Option<crate::classification::ClassificationLevel>,
    pub reason: String,
    pub granted_at: u64,
    pub expires_at: u64,
    pub revoked: bool,
}

impl BreakGlassSession {
    /// Serializes the session to bytes.
    /// Layout: role_id(4) + activated_roles_count(4) + activated_roles(N*4)
    ///       + activated_privileges_count(4) + activated_privileges(N*1)
    ///       + elevated_clearance_tag(1) + [clearance(1)]
    ///       + granted_at(8) + expires_at(8) + revoked(1) + reason_len(4) + reason(N)
    pub fn to_bytes(&self) -> Vec<u8> {
        let reason_bytes = self.reason.as_bytes();
        let mut buf = Vec::with_capacity(64 + reason_bytes.len());
        buf.extend_from_slice(&self.role_id.0.to_le_bytes());

        // activated_roles
        buf.extend_from_slice(&(self.activated_roles.len() as u32).to_le_bytes());
        for r in &self.activated_roles {
            buf.extend_from_slice(&r.0.to_le_bytes());
        }

        // activated_privileges
        buf.extend_from_slice(&(self.activated_privileges.len() as u32).to_le_bytes());
        for p in &self.activated_privileges {
            buf.push(*p as u8);
        }

        // elevated_clearance
        match self.elevated_clearance {
            None => buf.push(0),
            Some(level) => {
                buf.push(1);
                buf.push(level as u8);
            }
        }

        buf.extend_from_slice(&self.granted_at.to_le_bytes());
        buf.extend_from_slice(&self.expires_at.to_le_bytes());
        buf.push(if self.revoked { 1 } else { 0 });
        buf.extend_from_slice(&(reason_bytes.len() as u32).to_le_bytes());
        buf.extend_from_slice(reason_bytes);
        buf
    }

    /// Deserializes a session from bytes.
    pub fn from_bytes(data: &[u8]) -> Result<Self> {
        if data.len() < 4 {
            return Err(ZyronError::DecodingFailed(
                "BreakGlassSession too short".to_string(),
            ));
        }
        let mut off = 0;

        let role_id =
            RoleId(u32::from_le_bytes(data[off..off + 4].try_into().map_err(
                |_| ZyronError::DecodingFailed("truncated role_id".to_string()),
            )?));
        off += 4;

        // activated_roles
        if off + 4 > data.len() {
            return Err(ZyronError::DecodingFailed(
                "truncated activated_roles count".to_string(),
            ));
        }
        let roles_count = u32::from_le_bytes(
            data[off..off + 4]
                .try_into()
                .map_err(|_| ZyronError::DecodingFailed("truncated field".to_string()))?,
        ) as usize;
        off += 4;
        let mut activated_roles = Vec::with_capacity(roles_count);
        for _ in 0..roles_count {
            if off + 4 > data.len() {
                return Err(ZyronError::DecodingFailed(
                    "truncated activated_roles".to_string(),
                ));
            }
            activated_roles.push(RoleId(u32::from_le_bytes(
                data[off..off + 4]
                    .try_into()
                    .map_err(|_| ZyronError::DecodingFailed("truncated field".to_string()))?,
            )));
            off += 4;
        }

        // activated_privileges
        if off + 4 > data.len() {
            return Err(ZyronError::DecodingFailed(
                "truncated activated_privileges count".to_string(),
            ));
        }
        let privs_count = u32::from_le_bytes(
            data[off..off + 4]
                .try_into()
                .map_err(|_| ZyronError::DecodingFailed("truncated field".to_string()))?,
        ) as usize;
        off += 4;
        let mut activated_privileges = Vec::with_capacity(privs_count);
        for _ in 0..privs_count {
            if off + 1 > data.len() {
                return Err(ZyronError::DecodingFailed(
                    "truncated activated_privileges".to_string(),
                ));
            }
            activated_privileges.push(crate::privilege::PrivilegeType::from_u8(data[off])?);
            off += 1;
        }

        // elevated_clearance
        if off + 1 > data.len() {
            return Err(ZyronError::DecodingFailed(
                "truncated elevated_clearance tag".to_string(),
            ));
        }
        let elevated_clearance = match data[off] {
            0 => {
                off += 1;
                None
            }
            1 => {
                off += 1;
                if off + 1 > data.len() {
                    return Err(ZyronError::DecodingFailed(
                        "truncated elevated_clearance value".to_string(),
                    ));
                }
                let level = crate::classification::ClassificationLevel::from_u8(data[off])?;
                off += 1;
                Some(level)
            }
            tag => {
                return Err(ZyronError::DecodingFailed(format!(
                    "invalid elevated_clearance tag {}",
                    tag
                )));
            }
        };

        if off + 8 + 8 + 1 + 4 > data.len() {
            return Err(ZyronError::DecodingFailed(
                "truncated session fields".to_string(),
            ));
        }
        let granted_at = u64::from_le_bytes(
            data[off..off + 8]
                .try_into()
                .map_err(|_| ZyronError::DecodingFailed("truncated field".to_string()))?,
        );
        off += 8;
        let expires_at = u64::from_le_bytes(
            data[off..off + 8]
                .try_into()
                .map_err(|_| ZyronError::DecodingFailed("truncated field".to_string()))?,
        );
        off += 8;
        let revoked = data[off] != 0;
        off += 1;
        let reason_len = u32::from_le_bytes(
            data[off..off + 4]
                .try_into()
                .map_err(|_| ZyronError::DecodingFailed("truncated field".to_string()))?,
        ) as usize;
        off += 4;
        if off + reason_len > data.len() {
            return Err(ZyronError::DecodingFailed(
                "BreakGlassSession reason truncated".to_string(),
            ));
        }
        let reason = std::str::from_utf8(&data[off..off + reason_len])
            .map_err(|_| ZyronError::DecodingFailed("Invalid UTF-8 in reason".to_string()))?
            .to_string();
        Ok(Self {
            role_id,
            activated_roles,
            activated_privileges,
            elevated_clearance,
            reason,
            granted_at,
            expires_at,
            revoked,
        })
    }
}

/// Manages break-glass sessions with duration caps and audit logging.
pub struct BreakGlassManager {
    active: scc::HashMap<RoleId, BreakGlassSession>,
    audit_log: parking_lot::Mutex<Vec<BreakGlassSession>>,
    max_duration_secs: u64,
}

impl BreakGlassManager {
    /// Creates a new manager with the given maximum session duration.
    pub fn new(max_duration_secs: u64) -> Self {
        Self {
            active: scc::HashMap::new(),
            audit_log: parking_lot::Mutex::new(Vec::new()),
            max_duration_secs,
        }
    }

    /// Activates a break-glass session for the given role.
    /// The duration is capped at max_duration_secs.
    pub fn activate(
        &self,
        role_id: RoleId,
        activated_roles: Vec<RoleId>,
        activated_privileges: Vec<crate::privilege::PrivilegeType>,
        elevated_clearance: Option<crate::classification::ClassificationLevel>,
        reason: String,
        duration_secs: u64,
    ) -> Result<()> {
        let capped = duration_secs.min(self.max_duration_secs);
        let now = now_secs();
        let session = BreakGlassSession {
            role_id,
            activated_roles,
            activated_privileges,
            elevated_clearance,
            reason,
            granted_at: now,
            expires_at: now + capped,
            revoked: false,
        };
        let audit_copy = session.clone();
        let _ = self.active.insert_sync(role_id, session);
        self.audit_log.lock().push(audit_copy);
        Ok(())
    }

    /// Returns the active session for the role if it exists and has not expired.
    /// Removes expired sessions on access.
    pub fn is_active(&self, role_id: RoleId) -> Option<BreakGlassSession> {
        let now = now_secs();
        let mut result = None;
        let mut expired = false;
        self.active.read_sync(&role_id, |_k, v| {
            if v.revoked || v.expires_at <= now {
                expired = true;
            } else {
                result = Some(v.clone());
            }
        });
        if expired {
            let _ = self.active.remove_sync(&role_id);
        }
        result
    }

    /// Deactivates a break-glass session and marks it revoked in the audit log.
    pub fn deactivate(&self, role_id: RoleId) {
        let _ = self.active.remove_sync(&role_id);
        let mut log = self.audit_log.lock();
        for entry in log.iter_mut().rev() {
            if entry.role_id == role_id && !entry.revoked {
                entry.revoked = true;
                break;
            }
        }
    }

    /// Removes all expired sessions from the active map.
    pub fn cleanup_expired(&self) {
        let now = now_secs();
        let mut to_remove = Vec::new();
        self.active.iter_sync(|k, v| {
            if v.expires_at <= now || v.revoked {
                to_remove.push(*k);
            }
            true
        });
        for key in to_remove {
            let _ = self.active.remove_sync(&key);
        }
    }

    /// Returns a copy of the full audit trail.
    pub fn audit_trail(&self) -> Vec<BreakGlassSession> {
        self.audit_log.lock().clone()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_activate_and_is_active() {
        let mgr = BreakGlassManager::new(3600);
        mgr.activate(
            RoleId(1),
            vec![RoleId(99)],
            vec![],
            None,
            "emergency".to_string(),
            300,
        )
        .expect("activate should succeed");
        let session = mgr.is_active(RoleId(1));
        assert!(session.is_some());
        let s = session.expect("session present");
        assert_eq!(s.activated_roles, vec![RoleId(99)]);
        assert_eq!(s.reason, "emergency");
        assert!(!s.revoked);
    }

    #[test]
    fn test_is_active_returns_none_for_missing() {
        let mgr = BreakGlassManager::new(3600);
        assert!(mgr.is_active(RoleId(42)).is_none());
    }

    #[test]
    fn test_deactivate() {
        let mgr = BreakGlassManager::new(3600);
        mgr.activate(
            RoleId(1),
            vec![RoleId(99)],
            vec![],
            None,
            "fix outage".to_string(),
            600,
        )
        .expect("activate should succeed");
        mgr.deactivate(RoleId(1));
        assert!(mgr.is_active(RoleId(1)).is_none());
        let trail = mgr.audit_trail();
        assert_eq!(trail.len(), 1);
        assert!(trail[0].revoked);
    }

    #[test]
    fn test_duration_capped_at_max() {
        let mgr = BreakGlassManager::new(100);
        mgr.activate(
            RoleId(1),
            vec![RoleId(99)],
            vec![],
            None,
            "test".to_string(),
            9999,
        )
        .expect("activate should succeed");
        let session = mgr.is_active(RoleId(1)).expect("session present");
        // The duration should be capped: expires_at - granted_at <= 100
        assert!(session.expires_at - session.granted_at <= 100);
    }

    #[test]
    fn test_expired_session_removed_on_access() {
        let mgr = BreakGlassManager::new(3600);
        // Manually insert an already-expired session
        let session = BreakGlassSession {
            role_id: RoleId(5),
            activated_roles: vec![RoleId(99)],
            activated_privileges: vec![],
            elevated_clearance: None,
            reason: "past".to_string(),
            granted_at: 1000,
            expires_at: 1001,
            revoked: false,
        };
        let _ = mgr.active.insert_sync(RoleId(5), session);
        // is_active should detect expiry and return None
        assert!(mgr.is_active(RoleId(5)).is_none());
    }

    #[test]
    fn test_audit_trail_records_all_activations() {
        let mgr = BreakGlassManager::new(3600);
        mgr.activate(
            RoleId(1),
            vec![RoleId(99)],
            vec![],
            None,
            "first".to_string(),
            300,
        )
        .expect("activate should succeed");
        mgr.activate(
            RoleId(2),
            vec![RoleId(99)],
            vec![],
            None,
            "second".to_string(),
            300,
        )
        .expect("activate should succeed");
        let trail = mgr.audit_trail();
        assert_eq!(trail.len(), 2);
    }

    #[test]
    fn test_cleanup_expired() {
        let mgr = BreakGlassManager::new(3600);
        // Insert an expired session directly
        let session = BreakGlassSession {
            role_id: RoleId(10),
            activated_roles: vec![RoleId(99)],
            activated_privileges: vec![],
            elevated_clearance: None,
            reason: "old".to_string(),
            granted_at: 100,
            expires_at: 101,
            revoked: false,
        };
        let _ = mgr.active.insert_sync(RoleId(10), session);
        mgr.cleanup_expired();
        assert!(mgr.is_active(RoleId(10)).is_none());
    }

    #[test]
    fn test_session_to_bytes_from_bytes_roundtrip() {
        let session = BreakGlassSession {
            role_id: RoleId(42),
            activated_roles: vec![RoleId(99)],
            activated_privileges: vec![],
            elevated_clearance: None,
            reason: "production incident #1234".to_string(),
            granted_at: 1700000000,
            expires_at: 1700003600,
            revoked: false,
        };
        let bytes = session.to_bytes();
        let restored =
            BreakGlassSession::from_bytes(&bytes).expect("deserialization should succeed");
        assert_eq!(restored.role_id, session.role_id);
        assert_eq!(restored.activated_roles, session.activated_roles);
        assert_eq!(restored.reason, session.reason);
        assert_eq!(restored.granted_at, session.granted_at);
        assert_eq!(restored.expires_at, session.expires_at);
        assert_eq!(restored.revoked, session.revoked);
    }

    #[test]
    fn test_session_from_bytes_too_short() {
        let short = vec![0u8; 10];
        assert!(BreakGlassSession::from_bytes(&short).is_err());
    }
}
