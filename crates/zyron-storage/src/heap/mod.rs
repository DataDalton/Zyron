//! Heap storage implementation.
//!
//! This module provides heap-based tuple storage with:
//! - HeapPage: Slotted page format for variable-length tuples
//! - HeapFile: High-level API coordinating pages, FSM, and disk I/O
//! - BufferedHeapWriter: High-throughput batched inserts with zero-copy

pub mod constants;
mod file;
mod page;
mod writer;

pub use file::{HeapFile, HeapFileConfig};
pub use page::{HeapPage, HeapPageHeader, HeapPageIterator, SlotId, TupleSlot};
pub use writer::{BufferedHeapWriter, WriteBufferStats};
