//! Heap storage implementation.
//!
//! This module provides heap-based tuple storage with:
//! - HeapPage: Slotted page format for variable-length tuples
//! - HeapFile: High-level API coordinating pages, FSM, and disk I/O

pub mod constants;
mod file;
mod page;

pub use file::{HeapFile, HeapFileConfig};
pub use page::{HeapPage, HeapPageHeader, HeapPageIterator, SlotId, TupleSlot};
