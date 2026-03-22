//! Data classification labels and clearance levels.
//!
//! Each column can be assigned a classification level (Public, Internal,
//! Confidential, Restricted). Roles have a clearance level, and access
//! is granted only when the clearance meets or exceeds the column label.

use serde::{Deserialize, Serialize};
use zyron_common::{Result, ZyronError};

/// Classification levels ordered from least to most sensitive.
/// The numeric ordering matches Ord: Public < Internal < Confidential < Restricted.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[repr(u8)]
pub enum ClassificationLevel {
    Public = 0,
    Internal = 1,
    Confidential = 2,
    Restricted = 3,
}

impl ClassificationLevel {
    /// Converts a u8 value to a ClassificationLevel.
    /// Returns an error if the value is not in the range 0..=3.
    pub fn from_u8(v: u8) -> Result<Self> {
        match v {
            0 => Ok(ClassificationLevel::Public),
            1 => Ok(ClassificationLevel::Internal),
            2 => Ok(ClassificationLevel::Confidential),
            3 => Ok(ClassificationLevel::Restricted),
            _ => Err(ZyronError::Internal(format!(
                "Invalid classification level: {}",
                v
            ))),
        }
    }
}

/// A classification label assigned to a specific column in a table.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ColumnClassification {
    pub table_id: u32,
    pub column_id: u16,
    pub level: ClassificationLevel,
}

impl ColumnClassification {
    /// Serialized size: 4 (table_id) + 2 (column_id) + 1 (level) = 7 bytes.
    pub const SERIALIZED_SIZE: usize = 7;

    /// Serializes this classification to a 7-byte buffer.
    /// Layout: table_id (4 bytes LE), column_id (2 bytes LE), level (1 byte).
    pub fn to_bytes(&self) -> [u8; Self::SERIALIZED_SIZE] {
        let mut buf = [0u8; Self::SERIALIZED_SIZE];
        buf[0..4].copy_from_slice(&self.table_id.to_le_bytes());
        buf[4..6].copy_from_slice(&self.column_id.to_le_bytes());
        buf[6] = self.level as u8;
        buf
    }

    /// Deserializes a classification from a byte slice.
    /// The slice must be at least 7 bytes long.
    pub fn from_bytes(data: &[u8]) -> Result<Self> {
        if data.len() < Self::SERIALIZED_SIZE {
            return Err(ZyronError::DecodingFailed(format!(
                "ColumnClassification requires {} bytes, got {}",
                Self::SERIALIZED_SIZE,
                data.len()
            )));
        }
        let table_id = u32::from_le_bytes([data[0], data[1], data[2], data[3]]);
        let column_id = u16::from_le_bytes([data[4], data[5]]);
        let level = ClassificationLevel::from_u8(data[6])?;
        Ok(Self {
            table_id,
            column_id,
            level,
        })
    }
}

/// In-memory store mapping (table_id, column_id) to classification levels.
/// Thread-safe via scc::HashMap for concurrent read/write access.
pub struct ClassificationStore {
    labels: scc::HashMap<(u32, u16), ClassificationLevel>,
}

impl ClassificationStore {
    /// Creates an empty classification store.
    pub fn new() -> Self {
        Self {
            labels: scc::HashMap::new(),
        }
    }

    /// Checks whether the given clearance level is sufficient to access
    /// the specified column. Returns true if no label is set for the column,
    /// or if the role clearance is >= the column classification.
    pub fn check_clearance(
        &self,
        role_clearance: ClassificationLevel,
        table_id: u32,
        column_id: u16,
    ) -> bool {
        let key = (table_id, column_id);
        match self.labels.read_sync(&key, |_, level| *level) {
            Some(level) => role_clearance >= level,
            None => true,
        }
    }

    /// Sets or updates the classification level for a column.
    pub fn set_classification(&self, table_id: u32, column_id: u16, level: ClassificationLevel) {
        let key = (table_id, column_id);
        match self.labels.entry_sync(key) {
            scc::hash_map::Entry::Occupied(mut occ) => {
                *occ.get_mut() = level;
            }
            scc::hash_map::Entry::Vacant(vac) => {
                let _ = vac.insert_entry(level);
            }
        }
    }

    /// Removes the classification label from a column.
    pub fn drop_classification(&self, table_id: u32, column_id: u16) {
        let key = (table_id, column_id);
        let _ = self.labels.remove_sync(&key);
    }

    /// Returns all classifications for the given table.
    /// Scans every entry and filters by table_id.
    pub fn classifications_for_table(&self, table_id: u32) -> Vec<ColumnClassification> {
        let mut result = Vec::new();
        self.labels.iter_sync(|key, level| {
            if key.0 == table_id {
                result.push(ColumnClassification {
                    table_id: key.0,
                    column_id: key.1,
                    level: *level,
                });
            }
            true
        });
        result
    }

    /// Loads a batch of classifications into the store, replacing any existing entries.
    pub fn load(&self, entries: Vec<ColumnClassification>) {
        for entry in entries {
            self.set_classification(entry.table_id, entry.column_id, entry.level);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_classification_level_from_u8_valid() {
        assert_eq!(
            ClassificationLevel::from_u8(0).ok(),
            Some(ClassificationLevel::Public)
        );
        assert_eq!(
            ClassificationLevel::from_u8(1).ok(),
            Some(ClassificationLevel::Internal)
        );
        assert_eq!(
            ClassificationLevel::from_u8(2).ok(),
            Some(ClassificationLevel::Confidential)
        );
        assert_eq!(
            ClassificationLevel::from_u8(3).ok(),
            Some(ClassificationLevel::Restricted)
        );
    }

    #[test]
    fn test_classification_level_from_u8_invalid() {
        assert!(ClassificationLevel::from_u8(4).is_err());
        assert!(ClassificationLevel::from_u8(255).is_err());
    }

    #[test]
    fn test_classification_level_ordering() {
        assert!(ClassificationLevel::Public < ClassificationLevel::Internal);
        assert!(ClassificationLevel::Internal < ClassificationLevel::Confidential);
        assert!(ClassificationLevel::Confidential < ClassificationLevel::Restricted);
    }

    #[test]
    fn test_column_classification_to_from_bytes_roundtrip() {
        let original = ColumnClassification {
            table_id: 42,
            column_id: 7,
            level: ClassificationLevel::Confidential,
        };
        let bytes = original.to_bytes();
        let decoded = ColumnClassification::from_bytes(&bytes).expect("decode should succeed");
        assert_eq!(decoded.table_id, 42);
        assert_eq!(decoded.column_id, 7);
        assert_eq!(decoded.level, ClassificationLevel::Confidential);
    }

    #[test]
    fn test_column_classification_from_bytes_too_short() {
        let short = [0u8; 5];
        assert!(ColumnClassification::from_bytes(&short).is_err());
    }

    #[test]
    fn test_column_classification_from_bytes_invalid_level() {
        let mut bytes = [0u8; 7];
        bytes[0..4].copy_from_slice(&1u32.to_le_bytes());
        bytes[4..6].copy_from_slice(&1u16.to_le_bytes());
        bytes[6] = 99;
        assert!(ColumnClassification::from_bytes(&bytes).is_err());
    }

    #[test]
    fn test_store_set_and_check_clearance() {
        let store = ClassificationStore::new();
        store.set_classification(1, 0, ClassificationLevel::Confidential);

        // Restricted clearance can access Confidential data.
        assert!(store.check_clearance(ClassificationLevel::Restricted, 1, 0));
        // Confidential clearance can access Confidential data (equal).
        assert!(store.check_clearance(ClassificationLevel::Confidential, 1, 0));
        // Internal clearance cannot access Confidential data.
        assert!(!store.check_clearance(ClassificationLevel::Internal, 1, 0));
        // Public clearance cannot access Confidential data.
        assert!(!store.check_clearance(ClassificationLevel::Public, 1, 0));
    }

    #[test]
    fn test_store_check_clearance_no_label() {
        let store = ClassificationStore::new();
        // No label set, any clearance should pass.
        assert!(store.check_clearance(ClassificationLevel::Public, 999, 0));
    }

    #[test]
    fn test_store_drop_classification() {
        let store = ClassificationStore::new();
        store.set_classification(1, 0, ClassificationLevel::Restricted);
        assert!(!store.check_clearance(ClassificationLevel::Public, 1, 0));

        store.drop_classification(1, 0);
        // After drop, no label means any clearance passes.
        assert!(store.check_clearance(ClassificationLevel::Public, 1, 0));
    }

    #[test]
    fn test_store_classifications_for_table() {
        let store = ClassificationStore::new();
        store.set_classification(1, 0, ClassificationLevel::Public);
        store.set_classification(1, 1, ClassificationLevel::Internal);
        store.set_classification(2, 0, ClassificationLevel::Restricted);

        let mut results = store.classifications_for_table(1);
        results.sort_by_key(|c| c.column_id);
        assert_eq!(results.len(), 2);
        assert_eq!(results[0].column_id, 0);
        assert_eq!(results[0].level, ClassificationLevel::Public);
        assert_eq!(results[1].column_id, 1);
        assert_eq!(results[1].level, ClassificationLevel::Internal);
    }

    #[test]
    fn test_store_classifications_for_table_empty() {
        let store = ClassificationStore::new();
        let results = store.classifications_for_table(999);
        assert!(results.is_empty());
    }

    #[test]
    fn test_store_load_bulk() {
        let store = ClassificationStore::new();
        let entries = vec![
            ColumnClassification {
                table_id: 10,
                column_id: 0,
                level: ClassificationLevel::Internal,
            },
            ColumnClassification {
                table_id: 10,
                column_id: 1,
                level: ClassificationLevel::Confidential,
            },
            ColumnClassification {
                table_id: 20,
                column_id: 0,
                level: ClassificationLevel::Restricted,
            },
        ];
        store.load(entries);

        assert!(store.check_clearance(ClassificationLevel::Internal, 10, 0));
        assert!(!store.check_clearance(ClassificationLevel::Internal, 10, 1));
        assert!(!store.check_clearance(ClassificationLevel::Confidential, 20, 0));
    }

    #[test]
    fn test_store_set_overwrites_existing() {
        let store = ClassificationStore::new();
        store.set_classification(1, 0, ClassificationLevel::Restricted);
        assert!(!store.check_clearance(ClassificationLevel::Public, 1, 0));

        store.set_classification(1, 0, ClassificationLevel::Public);
        assert!(store.check_clearance(ClassificationLevel::Public, 1, 0));
    }

    #[test]
    fn test_column_classification_byte_layout() {
        let entry = ColumnClassification {
            table_id: 0x04030201,
            column_id: 0x0605,
            level: ClassificationLevel::Restricted,
        };
        let bytes = entry.to_bytes();
        assert_eq!(bytes[0], 0x01);
        assert_eq!(bytes[1], 0x02);
        assert_eq!(bytes[2], 0x03);
        assert_eq!(bytes[3], 0x04);
        assert_eq!(bytes[4], 0x05);
        assert_eq!(bytes[5], 0x06);
        assert_eq!(bytes[6], 3);
    }
}
