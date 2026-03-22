//! Object tagging and tag-based grants.
//!
//! Tags are arbitrary string labels attached to database objects (tables,
//! schemas, columns, indexes). The tag store maintains two indexes for
//! bidirectional lookup: by tag name and by object identity.

use serde::{Deserialize, Serialize};
use zyron_common::{Result, ZyronError};

use crate::privilege::ObjectType;

/// A tag attached to a specific database object, optionally at the column level.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ObjectTag {
    pub object_type: ObjectType,
    pub object_id: u32,
    /// If set, the tag applies to a specific column within the object.
    pub column_id: Option<u16>,
    pub tag: String,
}

impl ObjectTag {
    /// Serializes this tag to bytes.
    /// Layout: object_type (1 byte) + object_id (4 LE) + has_column (1) +
    /// [column_id (2 LE)] + tag_len (2 LE) + tag bytes.
    pub fn to_bytes(&self) -> Vec<u8> {
        let tag_bytes = self.tag.as_bytes();
        let col_size = if self.column_id.is_some() { 2 } else { 0 };
        let mut buf = Vec::with_capacity(1 + 4 + 1 + col_size + 2 + tag_bytes.len());
        buf.push(self.object_type as u8);
        buf.extend_from_slice(&self.object_id.to_le_bytes());
        match self.column_id {
            Some(cid) => {
                buf.push(1);
                buf.extend_from_slice(&cid.to_le_bytes());
            }
            None => {
                buf.push(0);
            }
        }
        buf.extend_from_slice(&(tag_bytes.len() as u16).to_le_bytes());
        buf.extend_from_slice(tag_bytes);
        buf
    }

    /// Deserializes an ObjectTag from a byte slice.
    pub fn from_bytes(data: &[u8]) -> Result<Self> {
        if data.len() < 8 {
            return Err(ZyronError::DecodingFailed(
                "ObjectTag requires at least 8 bytes".to_string(),
            ));
        }
        let object_type = ObjectType::from_u8(data[0])?;
        let object_id = u32::from_le_bytes([data[1], data[2], data[3], data[4]]);
        let has_column = data[5];
        let (column_id, offset) = if has_column == 1 {
            if data.len() < 10 {
                return Err(ZyronError::DecodingFailed(
                    "ObjectTag with column requires at least 10 bytes".to_string(),
                ));
            }
            let cid = u16::from_le_bytes([data[6], data[7]]);
            (Some(cid), 8)
        } else {
            (None, 6)
        };
        if data.len() < offset + 2 {
            return Err(ZyronError::DecodingFailed(
                "ObjectTag missing tag length".to_string(),
            ));
        }
        let tag_len = u16::from_le_bytes([data[offset], data[offset + 1]]) as usize;
        if data.len() < offset + 2 + tag_len {
            return Err(ZyronError::DecodingFailed(format!(
                "ObjectTag tag requires {} bytes, got {}",
                tag_len,
                data.len() - offset - 2
            )));
        }
        let tag = std::str::from_utf8(&data[offset + 2..offset + 2 + tag_len])
            .map_err(|_| {
                ZyronError::DecodingFailed("ObjectTag tag is not valid UTF-8".to_string())
            })?
            .to_string();
        Ok(Self {
            object_type,
            object_id,
            column_id,
            tag,
        })
    }
}

/// In-memory store for object tags with two indexes:
/// - by_tag: tag name -> list of (ObjectType, object_id, column_id)
/// - by_object: (object_type as u8, object_id) -> list of tag names
pub struct TagStore {
    by_tag: scc::HashMap<String, Vec<(ObjectType, u32, Option<u16>)>>,
    by_object: scc::HashMap<(u8, u32), Vec<String>>,
}

impl TagStore {
    /// Creates an empty tag store.
    pub fn new() -> Self {
        Self {
            by_tag: scc::HashMap::new(),
            by_object: scc::HashMap::new(),
        }
    }

    /// Attaches a tag to an object. Updates both indexes.
    pub fn tag_object(&self, tag: ObjectTag) -> Result<()> {
        let entry = (tag.object_type, tag.object_id, tag.column_id);
        let obj_key = (tag.object_type as u8, tag.object_id);
        let tag_name = tag.tag;

        // Update by_tag index.
        match self.by_tag.entry_sync(tag_name.clone()) {
            scc::hash_map::Entry::Occupied(mut occ) => {
                let list = occ.get_mut();
                if !list.contains(&entry) {
                    list.push(entry);
                }
            }
            scc::hash_map::Entry::Vacant(vac) => {
                let _ = vac.insert_entry(vec![entry]);
            }
        }

        // Update by_object index.
        match self.by_object.entry_sync(obj_key) {
            scc::hash_map::Entry::Occupied(mut occ) => {
                let list = occ.get_mut();
                if !list.contains(&tag_name) {
                    list.push(tag_name);
                }
            }
            scc::hash_map::Entry::Vacant(vac) => {
                let _ = vac.insert_entry(vec![tag_name]);
            }
        }

        Ok(())
    }

    /// Removes a tag from an object. Returns true if the tag was found and removed.
    pub fn untag_object(&self, object_type: ObjectType, object_id: u32, tag: &str) -> bool {
        let mut removed = false;

        // Remove from by_tag index.
        let _ = self.by_tag.read_sync(tag, |_, list| {
            // We need mutable access, so just check existence here.
            list.iter().any(|e| e.0 == object_type && e.1 == object_id)
        });

        // Use entry to modify by_tag.
        if let scc::hash_map::Entry::Occupied(mut occ) = self.by_tag.entry_sync(tag.to_string()) {
            let list = occ.get_mut();
            let before = list.len();
            list.retain(|e| !(e.0 == object_type && e.1 == object_id));
            if list.len() < before {
                removed = true;
            }
            if list.is_empty() {
                let _ = occ.remove();
            }
        }

        // Remove from by_object index.
        let obj_key = (object_type as u8, object_id);
        if let scc::hash_map::Entry::Occupied(mut occ) = self.by_object.entry_sync(obj_key) {
            let list = occ.get_mut();
            list.retain(|t| t != tag);
            if list.is_empty() {
                let _ = occ.remove();
            }
        }

        removed
    }

    /// Returns all objects that have the given tag.
    pub fn objects_with_tag(&self, tag: &str) -> Vec<(ObjectType, u32, Option<u16>)> {
        self.by_tag
            .read_sync(tag, |_, list| list.clone())
            .unwrap_or_default()
    }

    /// Returns all tags for the given object.
    pub fn tags_for_object(&self, object_type: ObjectType, object_id: u32) -> Vec<String> {
        let key = (object_type as u8, object_id);
        self.by_object
            .read_sync(&key, |_, list| list.clone())
            .unwrap_or_default()
    }

    /// Checks whether the given object has a specific tag.
    pub fn has_tag(&self, object_type: ObjectType, object_id: u32, tag: &str) -> bool {
        self.by_tag
            .read_sync(tag, |_, list| {
                list.iter().any(|e| e.0 == object_type && e.1 == object_id)
            })
            .unwrap_or(false)
    }

    /// Loads a batch of tags into the store.
    pub fn load(&self, tags: Vec<ObjectTag>) {
        for tag in tags {
            // Ignore errors from tag_object during bulk load (duplicates are fine).
            let _ = self.tag_object(tag);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // -- ObjectTag serialization tests --

    #[test]
    fn test_object_tag_roundtrip_no_column() {
        let tag = ObjectTag {
            object_type: ObjectType::Table,
            object_id: 42,
            column_id: None,
            tag: "pii".to_string(),
        };
        let bytes = tag.to_bytes();
        let decoded = ObjectTag::from_bytes(&bytes).expect("decode");
        assert_eq!(decoded.object_type, ObjectType::Table);
        assert_eq!(decoded.object_id, 42);
        assert_eq!(decoded.column_id, None);
        assert_eq!(decoded.tag, "pii");
    }

    #[test]
    fn test_object_tag_roundtrip_with_column() {
        let tag = ObjectTag {
            object_type: ObjectType::Table,
            object_id: 10,
            column_id: Some(5),
            tag: "sensitive".to_string(),
        };
        let bytes = tag.to_bytes();
        let decoded = ObjectTag::from_bytes(&bytes).expect("decode");
        assert_eq!(decoded.object_type, ObjectType::Table);
        assert_eq!(decoded.object_id, 10);
        assert_eq!(decoded.column_id, Some(5));
        assert_eq!(decoded.tag, "sensitive");
    }

    #[test]
    fn test_object_tag_from_bytes_too_short() {
        assert!(ObjectTag::from_bytes(&[0; 3]).is_err());
    }

    // -- TagStore tests --

    #[test]
    fn test_tag_and_query_by_tag() {
        let store = TagStore::new();
        store
            .tag_object(ObjectTag {
                object_type: ObjectType::Table,
                object_id: 1,
                column_id: None,
                tag: "pii".to_string(),
            })
            .expect("tag");
        store
            .tag_object(ObjectTag {
                object_type: ObjectType::Table,
                object_id: 2,
                column_id: None,
                tag: "pii".to_string(),
            })
            .expect("tag");

        let objects = store.objects_with_tag("pii");
        assert_eq!(objects.len(), 2);
    }

    #[test]
    fn test_tag_and_query_by_object() {
        let store = TagStore::new();
        store
            .tag_object(ObjectTag {
                object_type: ObjectType::Table,
                object_id: 1,
                column_id: None,
                tag: "pii".to_string(),
            })
            .expect("tag");
        store
            .tag_object(ObjectTag {
                object_type: ObjectType::Table,
                object_id: 1,
                column_id: None,
                tag: "financial".to_string(),
            })
            .expect("tag");

        let mut tags = store.tags_for_object(ObjectType::Table, 1);
        tags.sort();
        assert_eq!(tags, vec!["financial", "pii"]);
    }

    #[test]
    fn test_has_tag() {
        let store = TagStore::new();
        store
            .tag_object(ObjectTag {
                object_type: ObjectType::Table,
                object_id: 1,
                column_id: None,
                tag: "pii".to_string(),
            })
            .expect("tag");

        assert!(store.has_tag(ObjectType::Table, 1, "pii"));
        assert!(!store.has_tag(ObjectType::Table, 1, "other"));
        assert!(!store.has_tag(ObjectType::Table, 99, "pii"));
    }

    #[test]
    fn test_untag_object() {
        let store = TagStore::new();
        store
            .tag_object(ObjectTag {
                object_type: ObjectType::Table,
                object_id: 1,
                column_id: None,
                tag: "pii".to_string(),
            })
            .expect("tag");

        assert!(store.has_tag(ObjectType::Table, 1, "pii"));
        let removed = store.untag_object(ObjectType::Table, 1, "pii");
        assert!(removed);
        assert!(!store.has_tag(ObjectType::Table, 1, "pii"));

        // Removing again returns false.
        let removed_again = store.untag_object(ObjectType::Table, 1, "pii");
        assert!(!removed_again);
    }

    #[test]
    fn test_untag_nonexistent() {
        let store = TagStore::new();
        let removed = store.untag_object(ObjectType::Table, 999, "nothing");
        assert!(!removed);
    }

    #[test]
    fn test_objects_with_tag_empty() {
        let store = TagStore::new();
        let objects = store.objects_with_tag("nonexistent");
        assert!(objects.is_empty());
    }

    #[test]
    fn test_tags_for_object_empty() {
        let store = TagStore::new();
        let tags = store.tags_for_object(ObjectType::Table, 999);
        assert!(tags.is_empty());
    }

    #[test]
    fn test_duplicate_tag_ignored() {
        let store = TagStore::new();
        let tag = ObjectTag {
            object_type: ObjectType::Table,
            object_id: 1,
            column_id: None,
            tag: "pii".to_string(),
        };
        store.tag_object(tag.clone()).expect("tag");
        store.tag_object(tag).expect("tag");

        let objects = store.objects_with_tag("pii");
        assert_eq!(objects.len(), 1);
    }

    #[test]
    fn test_load_bulk() {
        let store = TagStore::new();
        let tags = vec![
            ObjectTag {
                object_type: ObjectType::Table,
                object_id: 1,
                column_id: None,
                tag: "pii".to_string(),
            },
            ObjectTag {
                object_type: ObjectType::Table,
                object_id: 2,
                column_id: Some(3),
                tag: "pii".to_string(),
            },
            ObjectTag {
                object_type: ObjectType::Schema,
                object_id: 5,
                column_id: None,
                tag: "internal".to_string(),
            },
        ];
        store.load(tags);

        assert_eq!(store.objects_with_tag("pii").len(), 2);
        assert!(store.has_tag(ObjectType::Schema, 5, "internal"));
    }

    #[test]
    fn test_tag_with_column() {
        let store = TagStore::new();
        store
            .tag_object(ObjectTag {
                object_type: ObjectType::Table,
                object_id: 1,
                column_id: Some(7),
                tag: "encrypted".to_string(),
            })
            .expect("tag");

        let objects = store.objects_with_tag("encrypted");
        assert_eq!(objects.len(), 1);
        assert_eq!(objects[0].2, Some(7));
    }
}
