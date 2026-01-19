//! In-memory page storage for B+Tree nodes.

use zyron_common::page::PAGE_SIZE;

/// In-memory page storage for B+Tree nodes.
///
/// All pages are stored in RAM in a Vec. Page numbers map directly to Vec indices.
/// This eliminates disk I/O overhead and BufferPool lock contention from the hot path.
pub struct InMemoryPageStore {
    /// Pages stored by page number (index = page_num).
    pages: Vec<Box<[u8; PAGE_SIZE]>>,
}

impl InMemoryPageStore {
    /// Creates a new empty page store.
    pub fn new() -> Self {
        Self { pages: Vec::new() }
    }

    /// Allocates a new page and returns its page number.
    #[inline]
    pub fn allocate(&mut self) -> u32 {
        let page_num = self.pages.len() as u32;
        self.pages.push(Box::new([0u8; PAGE_SIZE]));
        page_num
    }

    /// Gets a page by page number (read-only).
    #[inline]
    pub fn get(&self, page_num: u32) -> Option<&[u8; PAGE_SIZE]> {
        self.pages.get(page_num as usize).map(|p| &**p)
    }

    /// Gets a mutable page by page number.
    #[inline]
    pub fn get_mut(&mut self, page_num: u32) -> Option<&mut [u8; PAGE_SIZE]> {
        self.pages.get_mut(page_num as usize).map(|p| &mut **p)
    }

    /// Writes page data at a specific page number.
    #[inline]
    pub fn write(&mut self, page_num: u32, data: &[u8; PAGE_SIZE]) {
        if let Some(page) = self.pages.get_mut(page_num as usize) {
            page.copy_from_slice(data);
        }
    }
}
