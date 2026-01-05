//! Storage engine for ZyronDB.
//!
//! This crate provides:
//! - Disk manager for page-level file I/O
//! - Heap pages for variable-length tuple storage
//! - HeapFile manager for coordinating storage operations
//! - Tuple representation and serialization
//! - B+ tree index implementation
//! - Free space map for space management

mod btree;
mod disk;
mod freespace;
mod heap;
mod tuple;

pub use btree::{
    BTreeIndex, BTreeInternalPage, BTreeLeafPage, DeleteResult, InternalEntry,
    InternalPageHeader, LeafEntry, LeafPageHeader, MAX_KEY_SIZE, MIN_FILL_FACTOR,
};
pub use disk::{DiskManager, DiskManagerConfig};
pub use freespace::{
    category_to_min_space, space_to_category, FreeSpaceMap, FsmHeader, FsmPage,
    ENTRIES_PER_FSM_PAGE,
};
pub use heap::{
    HeapFile, HeapFileConfig, HeapPage, HeapPageHeader, HeapPageIterator, SlotId, TupleSlot,
};
pub use tuple::{Tuple, TupleFlags, TupleHeader, TupleId};
