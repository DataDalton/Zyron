// -----------------------------------------------------------------------------
// SecurityContextSnapshot
// -----------------------------------------------------------------------------
//
// Frozen capture of a SecurityContext at the moment a long-lived background
// job (e.g. a streaming job) is created. The snapshot persists through the
// catalog and is rehydrated into a SecurityContext when the job runner starts.
//
// Break-glass fields are deliberately NOT captured. A streaming job outlives
// the creating session, so carrying an elevated clearance forward would turn
// a temporary break-glass window into a permanent privilege escalation.
// The snapshot always records the role's base clearance.

use crate::abac::SessionAttributes;
use crate::classification::ClassificationLevel;
use crate::context::SecurityContext;
use crate::role::{RoleId, UserId};
use crate::session_binding::QueryLimits;
use std::collections::HashMap;
use zyron_catalog::encoding::{
    read_option_string, read_string, read_u8, read_u16, read_u32, read_u64, write_option_string,
    write_string, write_u8, write_u16, write_u32, write_u64,
};
use zyron_common::Result;

// -----------------------------------------------------------------------------
// Struct
// -----------------------------------------------------------------------------

/// Serializable capture of the identity + attribute state of a SecurityContext.
/// Used to persist the creator identity of a streaming job so the background
/// runner can reconstruct an equivalent context after a server restart.
#[derive(Debug, Clone)]
pub struct SecurityContextSnapshot {
    pub user_id: UserId,
    pub current_role: RoleId,
    pub effective_roles: Vec<RoleId>,
    pub clearance: ClassificationLevel,
    pub attributes: SessionAttributes,
    pub captured_at: u64,
}

// -----------------------------------------------------------------------------
// Capture + rehydrate
// -----------------------------------------------------------------------------

impl SecurityContextSnapshot {
    /// Captures the base identity of the given context. Ignores any active
    /// break-glass elevation, impersonation stack, and privilege cache.
    pub fn from_context(ctx: &SecurityContext) -> Self {
        let captured_at = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();

        Self {
            user_id: ctx.user_id,
            current_role: ctx.current_role,
            effective_roles: ctx.effective_roles.clone(),
            clearance: ctx.clearance,
            attributes: ctx.attributes.clone(),
            captured_at,
        }
    }

    /// Rebuilds a SecurityContext from the snapshot. Query limits must be
    /// looked up at reconstruction time from the current QueryLimitStore
    /// because they are not part of the snapshot.
    pub fn into_context(self, query_limits: QueryLimits) -> SecurityContext {
        let effective_roles = self.effective_roles.clone();
        SecurityContext::new(
            self.user_id,
            self.current_role,
            effective_roles.clone(),
            effective_roles,
            self.clearance,
            self.attributes,
            None,
            query_limits,
        )
    }
}

// -----------------------------------------------------------------------------
// Binary encoding
// -----------------------------------------------------------------------------
//
// Layout:
//   u32  user_id
//   u32  current_role
//   u32  effective_roles count
//   u32[n] effective_roles values
//   u8   clearance
//   u64  captured_at
//   ---- attributes ----
//   u32  role_id
//   opt  department (option_string)
//   opt  region (option_string)
//   u8   clearance
//   str  ip_address
//   u64  connection_time
//   u32  custom map count
//   (str, str)[n] custom entries

impl SecurityContextSnapshot {
    /// Serializes the snapshot to a length-prefixed byte blob. Returns the
    /// raw bytes; callers wrap this in whatever outer frame they prefer.
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut buf = Vec::with_capacity(128);
        write_u32(&mut buf, self.user_id.0);
        write_u32(&mut buf, self.current_role.0);

        write_u32(&mut buf, self.effective_roles.len() as u32);
        for role in &self.effective_roles {
            write_u32(&mut buf, role.0);
        }

        write_u8(&mut buf, self.clearance as u8);
        write_u64(&mut buf, self.captured_at);

        write_u32(&mut buf, self.attributes.role_id.0);
        write_option_string(&mut buf, &self.attributes.department);
        write_option_string(&mut buf, &self.attributes.region);
        write_u8(&mut buf, self.attributes.clearance as u8);
        write_string(&mut buf, &self.attributes.ip_address);
        write_u64(&mut buf, self.attributes.connection_time);

        write_u32(&mut buf, self.attributes.custom.len() as u32);
        for (k, v) in &self.attributes.custom {
            write_string(&mut buf, k);
            write_string(&mut buf, v);
        }

        buf
    }

    /// Deserializes a snapshot from bytes. Advances `offset` past the snapshot
    /// so callers can continue reading a larger frame.
    pub fn from_bytes(data: &[u8], offset: &mut usize) -> Result<Self> {
        let user_id = UserId(read_u32(data, offset)?);
        let current_role = RoleId(read_u32(data, offset)?);

        let role_count = read_u32(data, offset)? as usize;
        let mut effective_roles = Vec::with_capacity(role_count);
        for _ in 0..role_count {
            effective_roles.push(RoleId(read_u32(data, offset)?));
        }

        let clearance = ClassificationLevel::from_u8(read_u8(data, offset)?)?;
        let captured_at = read_u64(data, offset)?;

        let attr_role = RoleId(read_u32(data, offset)?);
        let department = read_option_string(data, offset)?;
        let region = read_option_string(data, offset)?;
        let attr_clearance = ClassificationLevel::from_u8(read_u8(data, offset)?)?;
        let ip_address = read_string(data, offset)?;
        let connection_time = read_u64(data, offset)?;

        let custom_count = read_u32(data, offset)? as usize;
        let mut custom = HashMap::with_capacity(custom_count);
        for _ in 0..custom_count {
            let k = read_string(data, offset)?;
            let v = read_string(data, offset)?;
            custom.insert(k, v);
        }

        let attributes = SessionAttributes {
            role_id: attr_role,
            department,
            region,
            clearance: attr_clearance,
            ip_address,
            connection_time,
            custom,
        };

        Ok(Self {
            user_id,
            current_role,
            effective_roles,
            clearance,
            attributes,
            captured_at,
        })
    }
}

// Silence unused-import warnings for helpers kept for future fields.
const _: fn() = || {
    let _ = read_u16;
    let _ = write_u16;
};

// -----------------------------------------------------------------------------
// Tests
// -----------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_attrs() -> SessionAttributes {
        let mut custom = HashMap::new();
        custom.insert("team".to_string(), "ingest".to_string());
        SessionAttributes {
            role_id: RoleId(42),
            department: Some("data".to_string()),
            region: None,
            clearance: ClassificationLevel::Confidential,
            ip_address: "10.1.2.3".to_string(),
            connection_time: 1_700_000_000,
            custom,
        }
    }

    #[test]
    fn roundtrip_snapshot() {
        let snap = SecurityContextSnapshot {
            user_id: UserId(7),
            current_role: RoleId(42),
            effective_roles: vec![RoleId(42), RoleId(100), RoleId(101)],
            clearance: ClassificationLevel::Confidential,
            attributes: sample_attrs(),
            captured_at: 1_700_000_042,
        };

        let bytes = snap.to_bytes();
        let mut off = 0;
        let decoded = SecurityContextSnapshot::from_bytes(&bytes, &mut off).unwrap();
        assert_eq!(off, bytes.len());
        assert_eq!(decoded.user_id, snap.user_id);
        assert_eq!(decoded.current_role, snap.current_role);
        assert_eq!(decoded.effective_roles, snap.effective_roles);
        assert_eq!(decoded.clearance, snap.clearance);
        assert_eq!(decoded.captured_at, snap.captured_at);
        assert_eq!(decoded.attributes.role_id, snap.attributes.role_id);
        assert_eq!(decoded.attributes.department, snap.attributes.department);
        assert_eq!(decoded.attributes.region, snap.attributes.region);
        assert_eq!(decoded.attributes.clearance, snap.attributes.clearance);
        assert_eq!(decoded.attributes.ip_address, snap.attributes.ip_address);
        assert_eq!(
            decoded.attributes.connection_time,
            snap.attributes.connection_time
        );
        assert_eq!(decoded.attributes.custom, snap.attributes.custom);
    }

    #[test]
    fn snapshot_ignores_break_glass_elevation() {
        let mut ctx = SecurityContext::new(
            UserId(1),
            RoleId(10),
            vec![RoleId(10)],
            vec![RoleId(10)],
            ClassificationLevel::Internal,
            sample_attrs(),
            None,
            QueryLimits::default(),
        );
        ctx.break_glass = Some(RoleId(999));
        ctx.break_glass_clearance = Some(ClassificationLevel::Restricted);

        let snap = SecurityContextSnapshot::from_context(&ctx);
        assert_eq!(snap.clearance, ClassificationLevel::Internal);
    }
}
