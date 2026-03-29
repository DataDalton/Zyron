//! ID newtypes for pipeline, trigger, function, and related entities.
//!
//! Provides strongly-typed wrappers for all pipeline-subsystem identifiers
//! and an OID allocator starting at 20000 for pipeline objects.

use serde::{Deserialize, Serialize};
use std::fmt;
use std::sync::atomic::{AtomicU32, Ordering};

/// Object identifier for pipeline entities.
pub type Oid = u32;

/// Starting OID for pipeline objects. OIDs below this are reserved
/// for catalog and auth objects.
pub const PIPELINE_OID_START: u32 = 20000;

/// Atomic counter for allocating globally unique OIDs within
/// the pipeline subsystem.
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

    /// Resets the counter to the given value. Used during recovery
    /// to restore the allocator state from the maximum OID found
    /// in pipeline system tables.
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

define_id!(PipelineId, u32, "PipelineId");
define_id!(MaterializedViewId, u32, "MaterializedViewId");
define_id!(TriggerId, u32, "TriggerId");
define_id!(FunctionId, u32, "FunctionId");
define_id!(AggregateId, u32, "AggregateId");
define_id!(ScheduleId, u32, "ScheduleId");
define_id!(EventHandlerId, u32, "EventHandlerId");
define_id!(ProcedureId, u32, "ProcedureId");
define_id!(LineageId, u32, "LineageId");

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;
    use std::sync::Arc;
    use std::thread;

    #[test]
    fn test_pipeline_id_display() {
        let id = PipelineId(42);
        assert_eq!(format!("{}", id), "PipelineId(42)");
    }

    #[test]
    fn test_all_id_display() {
        assert_eq!(PipelineId(1).to_string(), "PipelineId(1)");
        assert_eq!(MaterializedViewId(2).to_string(), "MaterializedViewId(2)");
        assert_eq!(TriggerId(3).to_string(), "TriggerId(3)");
        assert_eq!(FunctionId(4).to_string(), "FunctionId(4)");
        assert_eq!(AggregateId(5).to_string(), "AggregateId(5)");
        assert_eq!(ScheduleId(6).to_string(), "ScheduleId(6)");
        assert_eq!(EventHandlerId(7).to_string(), "EventHandlerId(7)");
        assert_eq!(ProcedureId(8).to_string(), "ProcedureId(8)");
        assert_eq!(LineageId(9).to_string(), "LineageId(9)");
    }

    #[test]
    fn test_id_equality() {
        assert_eq!(TriggerId(1), TriggerId(1));
        assert_ne!(FunctionId(1), FunctionId(2));
    }

    #[test]
    fn test_id_ordering() {
        assert!(ScheduleId(1) < ScheduleId(2));
        assert!(PipelineId(100) > PipelineId(50));
    }

    #[test]
    fn test_id_copy_clone() {
        let id = PipelineId(5);
        let copied = id;
        let cloned = id.clone();
        assert_eq!(id, copied);
        assert_eq!(id, cloned);
    }

    #[test]
    fn test_id_hashing() {
        let mut set = HashSet::new();
        set.insert(FunctionId(1));
        set.insert(FunctionId(1));
        set.insert(FunctionId(2));
        assert_eq!(set.len(), 2);
    }

    #[test]
    fn test_oid_allocator_sequential() {
        let alloc = OidAllocator::new(PIPELINE_OID_START);
        assert_eq!(alloc.next(), 20000);
        assert_eq!(alloc.next(), 20001);
        assert_eq!(alloc.next(), 20002);
        assert_eq!(alloc.current(), 20003);
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

        let mut allIds = HashSet::new();
        for h in handles {
            let ids = h.join().expect("thread should not panic");
            for id in ids {
                assert!(allIds.insert(id), "duplicate OID: {id}");
            }
        }
        assert_eq!(allIds.len(), 8000);
    }
}
