//! User account management.
//!
//! A User is a login identity with credentials, connection limits, and account
//! status. Users are assigned to Roles for privilege resolution.

use crate::role::{RoleId, UserId};
use zyron_catalog::encoding::{
    read_bool, read_option_string, read_string, read_u8, read_u32, read_u64, write_bool,
    write_option_string, write_string, write_u8, write_u32, write_u64,
};
use zyron_common::{Result, ZyronError};

/// A database user account with credentials and account status.
pub struct User {
    pub id: UserId,
    pub name: String,
    pub password_hash: Option<String>,
    pub api_key_prefix: Option<String>,
    pub api_key_hash: Option<Vec<u8>>,
    pub totp_secret: Option<Vec<u8>>,
    pub connection_limit: i32,
    pub valid_until: Option<u64>,
    pub locked: bool,
    pub locked_at: Option<u64>,
    pub locked_reason: Option<String>,
    pub created_at: u64,
}

impl User {
    /// Serializes the user to a binary byte vector for storage.
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut buf = Vec::with_capacity(128);
        write_u32(&mut buf, self.id.0);
        write_string(&mut buf, &self.name);
        write_option_string(&mut buf, &self.password_hash);
        write_option_string(&mut buf, &self.api_key_prefix);

        // api_key_hash: 0 for None, 1 + 32 bytes for Some.
        match &self.api_key_hash {
            None => write_u8(&mut buf, 0),
            Some(hash) => {
                write_u8(&mut buf, 1);
                buf.extend_from_slice(hash);
            }
        }

        // totp_secret: 0 for None, 1 + u32 len + bytes for Some.
        match &self.totp_secret {
            None => write_u8(&mut buf, 0),
            Some(secret) => {
                write_u8(&mut buf, 1);
                write_u32(&mut buf, secret.len() as u32);
                buf.extend_from_slice(secret);
            }
        }

        // connection_limit stored as i32 via u32 bit cast.
        write_u32(&mut buf, self.connection_limit as u32);

        // valid_until: 0 for None, 1 + u64 for Some.
        match self.valid_until {
            None => write_u8(&mut buf, 0),
            Some(ts) => {
                write_u8(&mut buf, 1);
                write_u64(&mut buf, ts);
            }
        }

        // locked: u8 (0 = false, 1 = true).
        write_u8(&mut buf, if self.locked { 1 } else { 0 });

        // locked_at: 0 tag = None, 1 tag + 8 bytes = Some.
        match self.locked_at {
            None => write_u8(&mut buf, 0),
            Some(ts) => {
                write_u8(&mut buf, 1);
                write_u64(&mut buf, ts);
            }
        }

        // locked_reason: 0 tag = None, 1 tag + len + bytes = Some.
        write_option_string(&mut buf, &self.locked_reason);

        write_u64(&mut buf, self.created_at);
        buf
    }

    /// Deserializes a user from a binary byte slice.
    pub fn from_bytes(data: &[u8]) -> Result<Self> {
        let mut off = 0;
        let id = UserId(read_u32(data, &mut off)?);
        let name = read_string(data, &mut off)?;
        let password_hash = read_option_string(data, &mut off)?;
        let api_key_prefix = read_option_string(data, &mut off)?;

        let api_key_hash = match read_u8(data, &mut off)? {
            0 => None,
            1 => {
                if off + 32 > data.len() {
                    return Err(ZyronError::CatalogCorrupted(
                        "truncated api_key_hash in user".to_string(),
                    ));
                }
                let mut hash = vec![0u8; 32];
                hash.copy_from_slice(&data[off..off + 32]);
                off += 32;
                Some(hash)
            }
            tag => {
                return Err(ZyronError::CatalogCorrupted(format!(
                    "invalid api_key_hash tag {} in user",
                    tag
                )));
            }
        };

        let totp_secret = match read_u8(data, &mut off)? {
            0 => None,
            1 => {
                let len = read_u32(data, &mut off)? as usize;
                if off + len > data.len() {
                    return Err(ZyronError::CatalogCorrupted(
                        "truncated totp_secret in user".to_string(),
                    ));
                }
                let mut secret = vec![0u8; len];
                secret.copy_from_slice(&data[off..off + len]);
                off += len;
                Some(secret)
            }
            tag => {
                return Err(ZyronError::CatalogCorrupted(format!(
                    "invalid totp_secret tag {} in user",
                    tag
                )));
            }
        };

        let connection_limit = read_u32(data, &mut off)? as i32;

        let valid_until = match read_u8(data, &mut off)? {
            0 => None,
            1 => Some(read_u64(data, &mut off)?),
            tag => {
                return Err(ZyronError::CatalogCorrupted(format!(
                    "invalid valid_until tag {} in user",
                    tag
                )));
            }
        };

        let locked = read_u8(data, &mut off)? != 0;

        let locked_at = match read_u8(data, &mut off)? {
            0 => None,
            1 => Some(read_u64(data, &mut off)?),
            tag => {
                return Err(ZyronError::CatalogCorrupted(format!(
                    "invalid locked_at tag {} in user",
                    tag
                )));
            }
        };

        let locked_reason = read_option_string(data, &mut off)?;

        let created_at = read_u64(data, &mut off)?;

        Ok(Self {
            id,
            name,
            password_hash,
            api_key_prefix,
            api_key_hash,
            totp_secret,
            connection_limit,
            valid_until,
            locked,
            locked_at,
            locked_reason,
            created_at,
        })
    }
}

/// Maps a user to a role they are a member of.
pub struct UserRoleMembership {
    pub user_id: UserId,
    pub role_id: RoleId,
    pub admin_option: bool,
    pub inherit: bool,
    pub granted_by: UserId,
}

impl UserRoleMembership {
    /// Serializes the user-role membership to a binary byte vector.
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut buf = Vec::with_capacity(20);
        write_u32(&mut buf, self.user_id.0);
        write_u32(&mut buf, self.role_id.0);
        write_bool(&mut buf, self.admin_option);
        write_bool(&mut buf, self.inherit);
        write_u32(&mut buf, self.granted_by.0);
        buf
    }

    /// Deserializes a user-role membership from a binary byte slice.
    pub fn from_bytes(data: &[u8]) -> Result<Self> {
        let mut off = 0;
        let user_id = UserId(read_u32(data, &mut off)?);
        let role_id = RoleId(read_u32(data, &mut off)?);
        let admin_option = read_bool(data, &mut off)?;
        let inherit = read_bool(data, &mut off)?;
        let granted_by = UserId(read_u32(data, &mut off)?);
        Ok(Self {
            user_id,
            role_id,
            admin_option,
            inherit,
            granted_by,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_user_to_bytes_from_bytes_roundtrip() {
        let user = User {
            id: UserId(42),
            name: "alice".to_string(),
            password_hash: Some("$balloon-sha256$v=1$s=64,t=3,d=3$salt$hash".to_string()),
            api_key_prefix: Some("zyron_ab".to_string()),
            api_key_hash: Some(vec![0xaa; 32]),
            totp_secret: Some(vec![0xbb; 20]),
            connection_limit: 50,
            valid_until: Some(2000000000),
            locked: false,
            locked_at: None,
            locked_reason: None,
            created_at: 1700000000,
        };
        let bytes = user.to_bytes();
        let restored = User::from_bytes(&bytes).expect("from_bytes failed");
        assert_eq!(restored.id, UserId(42));
        assert_eq!(restored.name, "alice");
        assert_eq!(
            restored.password_hash,
            Some("$balloon-sha256$v=1$s=64,t=3,d=3$salt$hash".to_string())
        );
        assert_eq!(restored.api_key_prefix, Some("zyron_ab".to_string()));
        assert_eq!(restored.api_key_hash, Some(vec![0xaa; 32]));
        assert_eq!(restored.totp_secret, Some(vec![0xbb; 20]));
        assert_eq!(restored.connection_limit, 50);
        assert_eq!(restored.valid_until, Some(2000000000));
        assert!(!restored.locked);
        assert_eq!(restored.locked_at, None);
        assert_eq!(restored.locked_reason, None);
        assert_eq!(restored.created_at, 1700000000);
    }

    #[test]
    fn test_user_to_bytes_from_bytes_minimal() {
        let user = User {
            id: UserId(1),
            name: "bob".to_string(),
            password_hash: None,
            api_key_prefix: None,
            api_key_hash: None,
            totp_secret: None,
            connection_limit: -1,
            valid_until: None,
            locked: false,
            locked_at: None,
            locked_reason: None,
            created_at: 1700000000,
        };
        let bytes = user.to_bytes();
        let restored = User::from_bytes(&bytes).expect("from_bytes failed");
        assert_eq!(restored.id, UserId(1));
        assert_eq!(restored.name, "bob");
        assert!(restored.password_hash.is_none());
        assert!(restored.api_key_prefix.is_none());
        assert!(restored.api_key_hash.is_none());
        assert!(restored.totp_secret.is_none());
        assert_eq!(restored.connection_limit, -1);
        assert_eq!(restored.valid_until, None);
        assert!(!restored.locked);
        assert_eq!(restored.locked_at, None);
        assert_eq!(restored.locked_reason, None);
        assert_eq!(restored.created_at, 1700000000);
    }

    #[test]
    fn test_user_locked_roundtrip() {
        let user = User {
            id: UserId(7),
            name: "charlie".to_string(),
            password_hash: Some("hash".to_string()),
            api_key_prefix: None,
            api_key_hash: None,
            totp_secret: None,
            connection_limit: 10,
            valid_until: None,
            locked: true,
            locked_at: Some(1700001000),
            locked_reason: Some("too many failed login attempts".to_string()),
            created_at: 1700000000,
        };
        let bytes = user.to_bytes();
        let restored = User::from_bytes(&bytes).expect("from_bytes failed");
        assert_eq!(restored.id, UserId(7));
        assert_eq!(restored.name, "charlie");
        assert!(restored.locked);
        assert_eq!(restored.locked_at, Some(1700001000));
        assert_eq!(
            restored.locked_reason,
            Some("too many failed login attempts".to_string())
        );
    }

    #[test]
    fn test_user_membership_roundtrip() {
        let m = UserRoleMembership {
            user_id: UserId(10),
            role_id: RoleId(20),
            admin_option: true,
            inherit: false,
            granted_by: UserId(1),
        };
        let bytes = m.to_bytes();
        let restored = UserRoleMembership::from_bytes(&bytes).expect("from_bytes failed");
        assert_eq!(restored.user_id, UserId(10));
        assert_eq!(restored.role_id, RoleId(20));
        assert!(restored.admin_option);
        assert!(!restored.inherit);
        assert_eq!(restored.granted_by, UserId(1));
    }
}
