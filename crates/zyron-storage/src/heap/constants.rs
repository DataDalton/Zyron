//! Heap page layout constants.

use zyron_common::page::PageHeader;

/// Size of the heap page header in bytes.
pub const HEAP_HEADER_SIZE: usize = 8;

/// Offset of heap header in page (after PageHeader).
pub const HEAP_HEADER_OFFSET: usize = PageHeader::SIZE;

/// Offset where slot array begins (after PageHeader + HeapPageHeader).
pub const DATA_START: usize = PageHeader::SIZE + HEAP_HEADER_SIZE;

/// Size of a tuple slot entry in bytes.
pub const TUPLE_SLOT_SIZE: usize = 4;

/// Size of the tuple header in bytes.
pub const TUPLE_HEADER_SIZE: usize = 12;
