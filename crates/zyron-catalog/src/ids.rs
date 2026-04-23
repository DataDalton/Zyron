//! Catalog identifier types for ZyronDB.
//!
//! Provides strongly-typed wrappers for all catalog object identifiers,
//! preventing accidental mixing of database IDs with table IDs, etc.

use serde::{Deserialize, Serialize};
use std::fmt;
use std::sync::atomic::{AtomicU32, Ordering};

/// Object identifier for catalog entries.
pub type Oid = u32;

/// Starting OID for user-created objects. OIDs below this are reserved for system objects.
pub const USER_OID_START: u32 = 10000;

/// System database ID for the default "zyron" database.
pub const SYSTEM_DATABASE_ID: DatabaseId = DatabaseId(1);

/// Default schema ID for the "public" schema.
pub const DEFAULT_SCHEMA_ID: SchemaId = SchemaId(1);

/// Catalog schema ID for the "zyron_catalog" internal schema.
pub const CATALOG_SCHEMA_ID: SchemaId = SchemaId(2);

/// Atomic counter for allocating globally unique OIDs.
pub struct OidAllocator {
    next: AtomicU32,
}

impl OidAllocator {
    /// Creates a new allocator starting from the given value.
    pub fn new(start: u32) -> Self {
        Self {
            next: AtomicU32::new(start),
        }
    }

    /// Allocates the next OID.
    pub fn next(&self) -> Oid {
        self.next.fetch_add(1, Ordering::Relaxed)
    }

    /// Returns the current counter value without allocating.
    pub fn current(&self) -> Oid {
        self.next.load(Ordering::Relaxed)
    }

    /// Resets the counter to the given value. Used during recovery to restore
    /// the allocator state from the maximum OID found in system tables.
    pub fn reset(&self, val: u32) {
        self.next.store(val, Ordering::Relaxed);
    }
}

macro_rules! define_id {
    ($name:ident, $inner:ty, $label:expr) => {
        #[derive(
            Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize,
        )]
        pub struct $name(pub $inner);

        impl fmt::Display for $name {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                write!(f, "{}({})", $label, self.0)
            }
        }
    };
}

define_id!(DatabaseId, u32, "DatabaseId");
define_id!(SchemaId, u32, "SchemaId");
define_id!(TableId, u32, "TableId");
define_id!(IndexId, u32, "IndexId");
define_id!(SequenceId, u32, "SequenceId");
define_id!(ColumnId, u16, "ColumnId");
define_id!(StreamingJobId, u32, "StreamingJobId");
define_id!(ExternalSourceId, u32, "ExternalSourceId");
define_id!(ExternalSinkId, u32, "ExternalSinkId");
define_id!(PublicationId, u32, "PublicationId");
define_id!(SubscriptionId, u32, "SubscriptionId");
define_id!(EndpointId, u32, "EndpointId");
define_id!(SecurityMapId, u32, "SecurityMapId");

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;

    #[test]
    fn test_oid_allocator_sequential() {
        let alloc = OidAllocator::new(100);
        assert_eq!(alloc.next(), 100);
        assert_eq!(alloc.next(), 101);
        assert_eq!(alloc.next(), 102);
        assert_eq!(alloc.current(), 103);
    }

    #[test]
    fn test_oid_allocator_reset() {
        let alloc = OidAllocator::new(1);
        alloc.next();
        alloc.next();
        alloc.reset(500);
        assert_eq!(alloc.next(), 500);
    }

    #[test]
    fn test_oid_allocator_concurrent() {
        use std::sync::Arc;
        use std::thread;

        let alloc = Arc::new(OidAllocator::new(0));
        let mut handles = Vec::new();

        for _ in 0..8 {
            let alloc = Arc::clone(&alloc);
            handles.push(thread::spawn(move || {
                let mut ids = Vec::new();
                for _ in 0..1000 {
                    ids.push(alloc.next());
                }
                ids
            }));
        }

        let mut all_ids = HashSet::new();
        for h in handles {
            for id in h.join().unwrap() {
                assert!(all_ids.insert(id), "duplicate OID: {id}");
            }
        }
        assert_eq!(all_ids.len(), 8000);
    }

    #[test]
    fn test_id_equality_and_hashing() {
        let t1 = TableId(42);
        let t2 = TableId(42);
        let t3 = TableId(99);

        assert_eq!(t1, t2);
        assert_ne!(t1, t3);

        let mut set = HashSet::new();
        set.insert(t1);
        set.insert(t2);
        set.insert(t3);
        assert_eq!(set.len(), 2);
    }

    #[test]
    fn test_id_copy_clone() {
        let id = DatabaseId(5);
        let copied = id;
        let cloned = id.clone();
        assert_eq!(id, copied);
        assert_eq!(id, cloned);
    }

    #[test]
    fn test_id_display() {
        assert_eq!(DatabaseId(1).to_string(), "DatabaseId(1)");
        assert_eq!(SchemaId(2).to_string(), "SchemaId(2)");
        assert_eq!(TableId(42).to_string(), "TableId(42)");
        assert_eq!(IndexId(100).to_string(), "IndexId(100)");
        assert_eq!(ColumnId(3).to_string(), "ColumnId(3)");
        assert_eq!(SequenceId(7).to_string(), "SequenceId(7)");
    }

    #[test]
    fn test_well_known_constants() {
        assert_eq!(SYSTEM_DATABASE_ID, DatabaseId(1));
        assert_eq!(DEFAULT_SCHEMA_ID, SchemaId(1));
        assert_eq!(CATALOG_SCHEMA_ID, SchemaId(2));
    }

    #[test]
    fn test_id_ordering() {
        assert!(TableId(1) < TableId(2));
        assert!(TableId(100) > TableId(50));
    }
}
