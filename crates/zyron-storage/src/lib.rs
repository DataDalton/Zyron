#![feature(portable_simd)]

//! Storage engine for ZyronDB.
//!
//! This crate provides:
//! - Disk manager for page-level file I/O
//! - Heap pages for variable-length tuple storage
//! - HeapFile manager for coordinating storage operations
//! - Tuple representation and serialization
//! - B+ tree index implementation
//! - Free space map for space management
//! - Column encoding engine for .zyr columnar storage
//! - Columnar file format with bloom filters and zone maps

mod btree;
pub mod checkpoint;
pub mod checkpoint_coordinator;
pub mod columnar;
mod disk;
pub mod encoding;
mod freespace;
mod heap;
mod tuple;
pub mod txn;

pub use btree::{
    BTreeArenaIndex, BTreeIndex, BTreeInternalPage, BTreeLeafPage, BufferedBTreeIndex,
    DeleteResult, InternalEntry, InternalPageHeader, LeafEntry, LeafPageHeader, MAX_KEY_SIZE,
    MIN_FILL_FACTOR,
    checkpoint::{CheckpointConfig, CheckpointTrigger},
};
pub use checkpoint::CheckpointTracker;
pub use checkpoint_coordinator::{
    CheckpointCoordinator, CheckpointCoordinatorConfig, CheckpointResult, CheckpointScheduler,
    CheckpointStats,
};
pub use disk::{DiskManager, DiskManagerConfig};
pub use freespace::{
    ENTRIES_PER_FSM_PAGE, FreeSpaceMap, FsmHeader, FsmPage, category_to_min_space,
    space_to_category,
};
pub use heap::{
    HeapFile, HeapFileConfig, HeapPage, HeapPageHeader, HeapPageIterator, SlotId, TupleSlot,
};
pub use tuple::{Tuple, TupleFlags, TupleHeader, TupleId};
pub use txn::{
    GcStats, IntentLockTable, IsolationLevel, LockTable, MvccGc, NodeLatch, Savepoint, Snapshot,
    Transaction, TransactionManager, TransactionStatus, WaitForGraph,
};
