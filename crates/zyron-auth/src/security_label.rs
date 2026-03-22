//! Mandatory Access Control (MAC) with compartment-based security labels.
//!
//! Extends the classification system (classification.rs) with compartment-based
//! labels for objects and subjects. A subject can access an object only if the
//! subject's level is >= the object's level AND the subject's compartments are
//! a superset of the object's compartments.

use crate::privilege::ObjectType;
use crate::rcu::RcuMap;
use crate::role::RoleId;
use serde::{Deserialize, Serialize};
use zyron_common::{Result, ZyronError};

/// Security classification level for MAC.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[repr(u8)]
pub enum SecurityLevel {
    Unclassified = 0,
    Confidential = 1,
    Secret = 2,
    TopSecret = 3,
}

impl SecurityLevel {
    fn from_u8(v: u8) -> Result<Self> {
        match v {
            0 => Ok(SecurityLevel::Unclassified),
            1 => Ok(SecurityLevel::Confidential),
            2 => Ok(SecurityLevel::Secret),
            3 => Ok(SecurityLevel::TopSecret),
            _ => Err(ZyronError::DecodingFailed(format!(
                "Unknown SecurityLevel value: {}",
                v
            ))),
        }
    }
}

/// A security label combining a level with compartments.
/// Compartments are kept sorted for consistent comparison.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SecurityLabel {
    pub level: SecurityLevel,
    /// Compartments this label belongs to (sorted, deduplicated).
    pub compartments: Vec<String>,
}

impl SecurityLabel {
    /// Creates a new security label with sorted, deduplicated compartments.
    pub fn new(level: SecurityLevel, mut compartments: Vec<String>) -> Self {
        compartments.sort();
        compartments.dedup();
        Self {
            level,
            compartments,
        }
    }

    /// Returns true if this label dominates the other (level >= other.level
    /// AND compartments is a superset of other.compartments).
    pub fn dominates(&self, other: &SecurityLabel) -> bool {
        if self.level < other.level {
            return false;
        }
        // Check superset: all of other's compartments must be in self
        other
            .compartments
            .iter()
            .all(|c| self.compartments.binary_search(c).is_ok())
    }

    /// Serializes the label to bytes.
    /// Layout: level(1) + compartment_count(4) + [len(4) + data(N)]*
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut buf = Vec::with_capacity(5 + self.compartments.len() * 16);
        buf.push(self.level as u8);
        buf.extend_from_slice(&(self.compartments.len() as u32).to_le_bytes());
        for comp in &self.compartments {
            let comp_bytes = comp.as_bytes();
            buf.extend_from_slice(&(comp_bytes.len() as u32).to_le_bytes());
            buf.extend_from_slice(comp_bytes);
        }
        buf
    }

    /// Deserializes a label from bytes. Returns the label and bytes consumed.
    pub fn from_bytes(data: &[u8]) -> Result<(Self, usize)> {
        if data.len() < 5 {
            return Err(ZyronError::DecodingFailed(
                "SecurityLabel data too short".to_string(),
            ));
        }
        let mut pos = 0;
        let level = SecurityLevel::from_u8(data[pos])?;
        pos += 1;
        let count =
            u32::from_le_bytes([data[pos], data[pos + 1], data[pos + 2], data[pos + 3]]) as usize;
        pos += 4;
        let mut compartments = Vec::with_capacity(count);
        for _ in 0..count {
            if data.len() < pos + 4 {
                return Err(ZyronError::DecodingFailed(
                    "SecurityLabel compartment length truncated".to_string(),
                ));
            }
            let len = u32::from_le_bytes([data[pos], data[pos + 1], data[pos + 2], data[pos + 3]])
                as usize;
            pos += 4;
            if data.len() < pos + len {
                return Err(ZyronError::DecodingFailed(
                    "SecurityLabel compartment data truncated".to_string(),
                ));
            }
            let s = std::str::from_utf8(&data[pos..pos + len])
                .map_err(|_| {
                    ZyronError::DecodingFailed("Invalid UTF-8 in compartment".to_string())
                })?
                .to_string();
            pos += len;
            compartments.push(s);
        }
        // Enforce sorted, deduplicated invariant for dominates() binary search.
        compartments.sort();
        compartments.dedup();
        Ok((
            Self {
                level,
                compartments,
            },
            pos,
        ))
    }
}

/// An object-level security label binding.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ObjectSecurityLabel {
    pub object_type: ObjectType,
    pub object_id: u32,
    pub label: SecurityLabel,
}

impl ObjectSecurityLabel {
    /// Serializes to bytes.
    /// Layout: object_type(1) + object_id(4) + label_bytes
    pub fn to_bytes(&self) -> Vec<u8> {
        let label_bytes = self.label.to_bytes();
        let mut buf = Vec::with_capacity(5 + label_bytes.len());
        buf.push(self.object_type as u8);
        buf.extend_from_slice(&self.object_id.to_le_bytes());
        buf.extend_from_slice(&label_bytes);
        buf
    }

    /// Deserializes from bytes.
    pub fn from_bytes(data: &[u8]) -> Result<Self> {
        if data.len() < 6 {
            return Err(ZyronError::DecodingFailed(
                "ObjectSecurityLabel data too short".to_string(),
            ));
        }
        let object_type = ObjectType::from_u8(data[0])?;
        let object_id = u32::from_le_bytes([data[1], data[2], data[3], data[4]]);
        let (label, _) = SecurityLabel::from_bytes(&data[5..])?;
        Ok(Self {
            object_type,
            object_id,
            label,
        })
    }
}

/// A subject-level security label binding.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SubjectSecurityLabel {
    pub role_id: RoleId,
    pub label: SecurityLabel,
}

impl SubjectSecurityLabel {
    /// Serializes to bytes.
    /// Layout: role_id(4) + label_bytes
    pub fn to_bytes(&self) -> Vec<u8> {
        let label_bytes = self.label.to_bytes();
        let mut buf = Vec::with_capacity(4 + label_bytes.len());
        buf.extend_from_slice(&self.role_id.0.to_le_bytes());
        buf.extend_from_slice(&label_bytes);
        buf
    }

    /// Deserializes from bytes.
    pub fn from_bytes(data: &[u8]) -> Result<Self> {
        if data.len() < 5 {
            return Err(ZyronError::DecodingFailed(
                "SubjectSecurityLabel data too short".to_string(),
            ));
        }
        let role_id = RoleId(u32::from_le_bytes([data[0], data[1], data[2], data[3]]));
        let (label, _) = SecurityLabel::from_bytes(&data[4..])?;
        Ok(Self { role_id, label })
    }
}

/// Mandatory Access Control store for comparing subject and object labels.
pub struct MandatoryAccessControl {
    object_labels: RcuMap<(u8, u32), SecurityLabel>,
    subject_labels: RcuMap<RoleId, SecurityLabel>,
}

impl MandatoryAccessControl {
    pub fn new() -> Self {
        Self {
            object_labels: RcuMap::empty_map(),
            subject_labels: RcuMap::empty_map(),
        }
    }

    /// Sets a security label on an object.
    pub fn set_object_label(&self, object_type: ObjectType, object_id: u32, label: SecurityLabel) {
        let key = (object_type as u8, object_id);
        self.object_labels.insert(key, label);
    }

    /// Removes a security label from an object.
    pub fn remove_object_label(&self, object_type: ObjectType, object_id: u32) -> bool {
        self.object_labels.remove(&(object_type as u8, object_id))
    }

    /// Gets the security label for an object.
    pub fn get_object_label(
        &self,
        object_type: ObjectType,
        object_id: u32,
    ) -> Option<SecurityLabel> {
        self.object_labels.get(&(object_type as u8, object_id))
    }

    /// Sets a security label on a subject (role).
    pub fn set_subject_label(&self, role_id: RoleId, label: SecurityLabel) {
        self.subject_labels.insert(role_id, label);
    }

    /// Removes a security label from a subject.
    pub fn remove_subject_label(&self, role_id: RoleId) -> bool {
        self.subject_labels.remove(&role_id)
    }

    /// Gets the security label for a subject.
    pub fn get_subject_label(&self, role_id: RoleId) -> Option<SecurityLabel> {
        self.subject_labels.get(&role_id)
    }

    /// Checks if a subject (role) can access an object.
    /// Returns true if the object has no label, or the subject dominates it.
    /// Returns false if the object has a label but the subject does not.
    pub fn check_access(&self, role_id: RoleId, object_type: ObjectType, object_id: u32) -> bool {
        let key = (object_type as u8, object_id);
        let obj_snap = self.object_labels.load();
        match obj_snap.get(&key) {
            Some(object_label) => {
                let subj_snap = self.subject_labels.load();
                match subj_snap.get(&role_id) {
                    Some(subject_label) => subject_label.dominates(object_label),
                    None => false,
                }
            }
            None => true, // No label on object, access allowed
        }
    }

    /// Bulk-loads object and subject labels from storage.
    pub fn load_object_labels(&self, labels: Vec<ObjectSecurityLabel>) {
        for osl in labels {
            self.set_object_label(osl.object_type, osl.object_id, osl.label);
        }
    }

    /// Bulk-loads subject labels from storage.
    pub fn load_subject_labels(&self, labels: Vec<SubjectSecurityLabel>) {
        for ssl in labels {
            self.set_subject_label(ssl.role_id, ssl.label);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // -- SecurityLabel tests --

    #[test]
    fn test_label_new_sorts_compartments() {
        let label = SecurityLabel::new(
            SecurityLevel::Secret,
            vec!["hr".to_string(), "finance".to_string(), "hr".to_string()],
        );
        assert_eq!(label.compartments, vec!["finance", "hr"]);
    }

    #[test]
    fn test_label_dominates_same_level_same_compartments() {
        let a = SecurityLabel::new(SecurityLevel::Secret, vec!["finance".to_string()]);
        let b = SecurityLabel::new(SecurityLevel::Secret, vec!["finance".to_string()]);
        assert!(a.dominates(&b));
    }

    #[test]
    fn test_label_dominates_higher_level() {
        let a = SecurityLabel::new(SecurityLevel::TopSecret, vec![]);
        let b = SecurityLabel::new(SecurityLevel::Secret, vec![]);
        assert!(a.dominates(&b));
    }

    #[test]
    fn test_label_not_dominates_lower_level() {
        let a = SecurityLabel::new(SecurityLevel::Confidential, vec![]);
        let b = SecurityLabel::new(SecurityLevel::Secret, vec![]);
        assert!(!a.dominates(&b));
    }

    #[test]
    fn test_label_dominates_superset_compartments() {
        let subject = SecurityLabel::new(
            SecurityLevel::Secret,
            vec!["finance".to_string(), "hr".to_string(), "legal".to_string()],
        );
        let object = SecurityLabel::new(
            SecurityLevel::Secret,
            vec!["finance".to_string(), "hr".to_string()],
        );
        assert!(subject.dominates(&object));
    }

    #[test]
    fn test_label_not_dominates_missing_compartment() {
        let subject = SecurityLabel::new(SecurityLevel::TopSecret, vec!["finance".to_string()]);
        let object = SecurityLabel::new(
            SecurityLevel::Secret,
            vec!["finance".to_string(), "hr".to_string()],
        );
        assert!(!subject.dominates(&object));
    }

    #[test]
    fn test_label_dominates_empty_object_compartments() {
        let subject = SecurityLabel::new(SecurityLevel::Secret, vec!["finance".to_string()]);
        let object = SecurityLabel::new(SecurityLevel::Secret, vec![]);
        assert!(subject.dominates(&object));
    }

    // -- Serialization tests --

    #[test]
    fn test_label_roundtrip() {
        let label = SecurityLabel::new(
            SecurityLevel::TopSecret,
            vec!["alpha".to_string(), "bravo".to_string()],
        );
        let bytes = label.to_bytes();
        let (restored, consumed) = SecurityLabel::from_bytes(&bytes).expect("decode");
        assert_eq!(restored, label);
        assert_eq!(consumed, bytes.len());
    }

    #[test]
    fn test_label_roundtrip_empty_compartments() {
        let label = SecurityLabel::new(SecurityLevel::Unclassified, vec![]);
        let bytes = label.to_bytes();
        let (restored, _) = SecurityLabel::from_bytes(&bytes).expect("decode");
        assert_eq!(restored, label);
    }

    #[test]
    fn test_object_security_label_roundtrip() {
        let osl = ObjectSecurityLabel {
            object_type: ObjectType::Table,
            object_id: 42,
            label: SecurityLabel::new(SecurityLevel::Confidential, vec!["finance".to_string()]),
        };
        let bytes = osl.to_bytes();
        let restored = ObjectSecurityLabel::from_bytes(&bytes).expect("decode");
        assert_eq!(restored.object_type, ObjectType::Table);
        assert_eq!(restored.object_id, 42);
        assert_eq!(restored.label, osl.label);
    }

    #[test]
    fn test_subject_security_label_roundtrip() {
        let ssl = SubjectSecurityLabel {
            role_id: RoleId(10),
            label: SecurityLabel::new(
                SecurityLevel::Secret,
                vec!["finance".to_string(), "hr".to_string()],
            ),
        };
        let bytes = ssl.to_bytes();
        let restored = SubjectSecurityLabel::from_bytes(&bytes).expect("decode");
        assert_eq!(restored.role_id, RoleId(10));
        assert_eq!(restored.label, ssl.label);
    }

    #[test]
    fn test_label_from_bytes_too_short() {
        assert!(SecurityLabel::from_bytes(&[0u8; 2]).is_err());
    }

    // -- MandatoryAccessControl tests --

    #[test]
    fn test_mac_no_label_allows() {
        let mac = MandatoryAccessControl::new();
        assert!(mac.check_access(RoleId(1), ObjectType::Table, 42));
    }

    #[test]
    fn test_mac_object_label_no_subject_denies() {
        let mac = MandatoryAccessControl::new();
        mac.set_object_label(
            ObjectType::Table,
            42,
            SecurityLabel::new(SecurityLevel::Confidential, vec![]),
        );
        assert!(!mac.check_access(RoleId(1), ObjectType::Table, 42));
    }

    #[test]
    fn test_mac_subject_dominates_allows() {
        let mac = MandatoryAccessControl::new();
        mac.set_object_label(
            ObjectType::Table,
            42,
            SecurityLabel::new(SecurityLevel::Secret, vec!["finance".to_string()]),
        );
        mac.set_subject_label(
            RoleId(1),
            SecurityLabel::new(
                SecurityLevel::TopSecret,
                vec!["finance".to_string(), "hr".to_string()],
            ),
        );
        assert!(mac.check_access(RoleId(1), ObjectType::Table, 42));
    }

    #[test]
    fn test_mac_subject_not_dominates_denies() {
        let mac = MandatoryAccessControl::new();
        mac.set_object_label(
            ObjectType::Table,
            42,
            SecurityLabel::new(
                SecurityLevel::Secret,
                vec!["finance".to_string(), "hr".to_string()],
            ),
        );
        mac.set_subject_label(
            RoleId(1),
            SecurityLabel::new(SecurityLevel::Secret, vec!["finance".to_string()]),
        );
        assert!(!mac.check_access(RoleId(1), ObjectType::Table, 42));
    }

    #[test]
    fn test_mac_remove_object_label() {
        let mac = MandatoryAccessControl::new();
        mac.set_object_label(
            ObjectType::Table,
            42,
            SecurityLabel::new(SecurityLevel::Secret, vec![]),
        );
        assert!(mac.remove_object_label(ObjectType::Table, 42));
        assert!(mac.check_access(RoleId(1), ObjectType::Table, 42));
    }

    #[test]
    fn test_mac_remove_subject_label() {
        let mac = MandatoryAccessControl::new();
        mac.set_subject_label(RoleId(1), SecurityLabel::new(SecurityLevel::Secret, vec![]));
        assert!(mac.remove_subject_label(RoleId(1)));
        assert!(mac.get_subject_label(RoleId(1)).is_none());
    }

    #[test]
    fn test_mac_load_bulk() {
        let mac = MandatoryAccessControl::new();
        mac.load_object_labels(vec![ObjectSecurityLabel {
            object_type: ObjectType::Table,
            object_id: 42,
            label: SecurityLabel::new(SecurityLevel::Confidential, vec![]),
        }]);
        mac.load_subject_labels(vec![SubjectSecurityLabel {
            role_id: RoleId(1),
            label: SecurityLabel::new(SecurityLevel::Secret, vec![]),
        }]);
        assert!(mac.check_access(RoleId(1), ObjectType::Table, 42));
    }
}
