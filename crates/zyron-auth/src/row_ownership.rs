//! Automatic row-level security based on inserting user.
//!
//! When row ownership is enabled on a table, each row is associated with
//! the role that inserted it. Only the owning role (or designated admin
//! roles) can read, update, or delete the row.

use serde::{Deserialize, Serialize};
use zyron_common::{Result, ZyronError};

use crate::role::RoleId;

/// Configuration for row-level ownership on a single table.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RowOwnershipConfig {
    pub table_id: u32,
    /// Whether row ownership filtering is active for this table.
    pub enabled: bool,
    /// The column name that stores the owning role ID.
    pub owner_column: String,
    /// Roles that bypass ownership checks (can access all rows).
    pub admin_roles: Vec<RoleId>,
}

impl RowOwnershipConfig {
    /// Serializes this config to bytes.
    /// Layout: table_id (4 LE) + enabled (1) + owner_column_len (2 LE) +
    /// owner_column bytes + admin_count (2 LE) + [role_id (4 LE)] per admin.
    pub fn to_bytes(&self) -> Vec<u8> {
        let col_bytes = self.owner_column.as_bytes();
        let admin_count = self.admin_roles.len();
        let size = 4 + 1 + 2 + col_bytes.len() + 2 + admin_count * 4;
        let mut buf = Vec::with_capacity(size);
        buf.extend_from_slice(&self.table_id.to_le_bytes());
        buf.push(if self.enabled { 1 } else { 0 });
        buf.extend_from_slice(&(col_bytes.len() as u16).to_le_bytes());
        buf.extend_from_slice(col_bytes);
        buf.extend_from_slice(&(admin_count as u16).to_le_bytes());
        for role in &self.admin_roles {
            buf.extend_from_slice(&role.0.to_le_bytes());
        }
        buf
    }

    /// Deserializes a RowOwnershipConfig from a byte slice.
    pub fn from_bytes(data: &[u8]) -> Result<Self> {
        if data.len() < 9 {
            return Err(ZyronError::DecodingFailed(
                "RowOwnershipConfig requires at least 9 bytes".to_string(),
            ));
        }
        let table_id = u32::from_le_bytes([data[0], data[1], data[2], data[3]]);
        let enabled = data[4] != 0;
        let col_len = u16::from_le_bytes([data[5], data[6]]) as usize;
        if data.len() < 7 + col_len + 2 {
            return Err(ZyronError::DecodingFailed(
                "RowOwnershipConfig buffer too short for owner_column".to_string(),
            ));
        }
        let owner_column = std::str::from_utf8(&data[7..7 + col_len])
            .map_err(|_| {
                ZyronError::DecodingFailed(
                    "RowOwnershipConfig owner_column is not valid UTF-8".to_string(),
                )
            })?
            .to_string();
        let offset = 7 + col_len;
        let admin_count = u16::from_le_bytes([data[offset], data[offset + 1]]) as usize;
        let admins_start = offset + 2;
        if data.len() < admins_start + admin_count * 4 {
            return Err(ZyronError::DecodingFailed(
                "RowOwnershipConfig buffer too short for admin_roles".to_string(),
            ));
        }
        let mut admin_roles = Vec::with_capacity(admin_count);
        for i in 0..admin_count {
            let pos = admins_start + i * 4;
            let rid = u32::from_le_bytes([data[pos], data[pos + 1], data[pos + 2], data[pos + 3]]);
            admin_roles.push(RoleId(rid));
        }
        Ok(Self {
            table_id,
            enabled,
            owner_column,
            admin_roles,
        })
    }
}

/// In-memory store for row ownership configurations, keyed by table_id.
pub struct RowOwnershipStore {
    configs: scc::HashMap<u32, RowOwnershipConfig>,
}

impl RowOwnershipStore {
    /// Creates an empty row ownership store.
    pub fn new() -> Self {
        Self {
            configs: scc::HashMap::new(),
        }
    }

    /// Returns true if row ownership is enabled for the given table.
    pub fn is_enabled(&self, table_id: u32) -> bool {
        self.configs
            .read_sync(&table_id, |_, config| config.enabled)
            .unwrap_or(false)
    }

    /// Returns a clone of the config for the given table, if one exists.
    pub fn get_config(&self, table_id: u32) -> Option<RowOwnershipConfig> {
        self.configs
            .read_sync(&table_id, |_, config| config.clone())
    }

    /// Enables row ownership for a table by storing or replacing the config.
    pub fn enable(&self, table_id: u32, config: RowOwnershipConfig) {
        match self.configs.entry_sync(table_id) {
            scc::hash_map::Entry::Occupied(mut occ) => {
                *occ.get_mut() = config;
            }
            scc::hash_map::Entry::Vacant(vac) => {
                let _ = vac.insert_entry(config);
            }
        }
    }

    /// Disables row ownership for a table by removing its config.
    pub fn disable(&self, table_id: u32) {
        let _ = self.configs.remove_sync(&table_id);
    }

    /// Checks whether the given role is an admin for the specified table.
    /// Returns false if no config exists or if the role is not in admin_roles.
    pub fn is_admin(&self, table_id: u32, role_id: RoleId) -> bool {
        self.configs
            .read_sync(&table_id, |_, config| config.admin_roles.contains(&role_id))
            .unwrap_or(false)
    }

    /// Loads a batch of configs into the store, replacing any existing entries.
    pub fn load(&self, configs: Vec<RowOwnershipConfig>) {
        for config in configs {
            let tid = config.table_id;
            self.enable(tid, config);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_config(table_id: u32, enabled: bool, admins: Vec<u32>) -> RowOwnershipConfig {
        RowOwnershipConfig {
            table_id,
            enabled,
            owner_column: "owner_id".to_string(),
            admin_roles: admins.into_iter().map(RoleId).collect(),
        }
    }

    // -- Serialization tests --

    #[test]
    fn test_config_roundtrip_no_admins() {
        let config = make_config(42, true, vec![]);
        let bytes = config.to_bytes();
        let decoded = RowOwnershipConfig::from_bytes(&bytes).expect("decode");
        assert_eq!(decoded.table_id, 42);
        assert!(decoded.enabled);
        assert_eq!(decoded.owner_column, "owner_id");
        assert!(decoded.admin_roles.is_empty());
    }

    #[test]
    fn test_config_roundtrip_with_admins() {
        let config = make_config(10, true, vec![100, 200, 300]);
        let bytes = config.to_bytes();
        let decoded = RowOwnershipConfig::from_bytes(&bytes).expect("decode");
        assert_eq!(decoded.table_id, 10);
        assert!(decoded.enabled);
        assert_eq!(decoded.admin_roles.len(), 3);
        assert_eq!(decoded.admin_roles[0], RoleId(100));
        assert_eq!(decoded.admin_roles[1], RoleId(200));
        assert_eq!(decoded.admin_roles[2], RoleId(300));
    }

    #[test]
    fn test_config_roundtrip_disabled() {
        let config = make_config(5, false, vec![]);
        let bytes = config.to_bytes();
        let decoded = RowOwnershipConfig::from_bytes(&bytes).expect("decode");
        assert!(!decoded.enabled);
    }

    #[test]
    fn test_config_from_bytes_too_short() {
        assert!(RowOwnershipConfig::from_bytes(&[0; 5]).is_err());
    }

    #[test]
    fn test_config_from_bytes_truncated_admins() {
        let config = make_config(1, true, vec![100, 200]);
        let mut bytes = config.to_bytes();
        // Truncate to cut off the second admin role.
        bytes.truncate(bytes.len() - 3);
        assert!(RowOwnershipConfig::from_bytes(&bytes).is_err());
    }

    // -- Store tests --

    #[test]
    fn test_store_enable_and_is_enabled() {
        let store = RowOwnershipStore::new();
        assert!(!store.is_enabled(1));

        store.enable(1, make_config(1, true, vec![]));
        assert!(store.is_enabled(1));
    }

    #[test]
    fn test_store_disable() {
        let store = RowOwnershipStore::new();
        store.enable(1, make_config(1, true, vec![]));
        assert!(store.is_enabled(1));

        store.disable(1);
        assert!(!store.is_enabled(1));
    }

    #[test]
    fn test_store_get_config() {
        let store = RowOwnershipStore::new();
        assert!(store.get_config(1).is_none());

        store.enable(1, make_config(1, true, vec![10, 20]));
        let config = store.get_config(1).expect("config should exist");
        assert_eq!(config.table_id, 1);
        assert_eq!(config.admin_roles.len(), 2);
    }

    #[test]
    fn test_store_is_admin() {
        let store = RowOwnershipStore::new();
        store.enable(1, make_config(1, true, vec![100, 200]));

        assert!(store.is_admin(1, RoleId(100)));
        assert!(store.is_admin(1, RoleId(200)));
        assert!(!store.is_admin(1, RoleId(300)));
    }

    #[test]
    fn test_store_is_admin_no_config() {
        let store = RowOwnershipStore::new();
        assert!(!store.is_admin(999, RoleId(1)));
    }

    #[test]
    fn test_store_load_bulk() {
        let store = RowOwnershipStore::new();
        let configs = vec![
            make_config(1, true, vec![10]),
            make_config(2, false, vec![]),
            make_config(3, true, vec![20, 30]),
        ];
        store.load(configs);

        assert!(store.is_enabled(1));
        assert!(!store.is_enabled(2));
        assert!(store.is_enabled(3));
        assert!(store.is_admin(3, RoleId(20)));
    }

    #[test]
    fn test_store_enable_overwrites() {
        let store = RowOwnershipStore::new();
        store.enable(1, make_config(1, true, vec![100]));
        assert!(store.is_admin(1, RoleId(100)));

        store.enable(1, make_config(1, true, vec![200]));
        assert!(!store.is_admin(1, RoleId(100)));
        assert!(store.is_admin(1, RoleId(200)));
    }

    #[test]
    fn test_store_disable_nonexistent() {
        let store = RowOwnershipStore::new();
        // Should not panic.
        store.disable(999);
        assert!(!store.is_enabled(999));
    }

    #[test]
    fn test_config_long_owner_column() {
        let config = RowOwnershipConfig {
            table_id: 1,
            enabled: true,
            owner_column: "a_very_long_column_name_for_ownership".to_string(),
            admin_roles: vec![RoleId(1)],
        };
        let bytes = config.to_bytes();
        let decoded = RowOwnershipConfig::from_bytes(&bytes).expect("decode");
        assert_eq!(
            decoded.owner_column,
            "a_very_long_column_name_for_ownership"
        );
    }
}
