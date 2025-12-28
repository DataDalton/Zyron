//! Storage engine for ZyronDB.
//!
//! This crate provides:
//! - Disk manager for page-level file I/O
//! - Heap pages for variable-length tuple storage
//! - Tuple representation and serialization

mod disk;
mod heap;
mod tuple;

pub use disk::{DiskManager, DiskManagerConfig};
pub use heap::{HeapPage, SlotId, TupleSlot};
pub use tuple::{Tuple, TupleId};
