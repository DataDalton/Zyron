//! Heap page layout constants.

use zyron_common::page::PageHeader;

/// Size of the heap page header in bytes.
pub const HEAP_HEADER_SIZE: usize = 8;

/// Offset of heap header in page (after PageHeader).
pub const HEAP_HEADER_OFFSET: usize = PageHeader::SIZE;

/// Offset where slot array begins (after PageHeader + HeapPageHeader).
pub const DATA_START: usize = PageHeader::SIZE + HEAP_HEADER_SIZE;

/// Size of a tuple slot entry in bytes. Each slot holds the tuple offset and
/// the tuple header, so a scan reads one dense array
pub const TUPLE_SLOT_SIZE: usize = 16;
