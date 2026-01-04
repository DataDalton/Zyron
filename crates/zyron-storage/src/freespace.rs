//! Free space map for tracking page space availability.
//!
//! The free space map (FSM) tracks how much free space each heap page has,
//! enabling efficient allocation of tuples to pages with sufficient space.
//!
//! Each heap page is represented by a single byte in the FSM, indicating
//! the approximate amount of free space available (quantized to 256 levels).
//!
//! Page layout:
//! ```text
//! +------------------+
//! | Page Header (32) |
//! +------------------+
//! | FSM Header (8)   |
//! +------------------+
//! | Space Entries    |
//! | (1 byte each)    |
//! +------------------+
//! ```

use zyron_common::page::{PageHeader, PageId, PageType, PAGE_SIZE};
use zyron_common::{Result, ZyronError};

/// Number of pages tracked per FSM page.
/// PAGE_SIZE - PageHeader::SIZE - FsmHeader::SIZE
pub const ENTRIES_PER_FSM_PAGE: usize = PAGE_SIZE - PageHeader::SIZE - FsmHeader::SIZE;

/// Header for free space map pages.
///
/// Layout (8 bytes):
/// - first_page_num: 4 bytes (first heap page tracked by this FSM page)
/// - num_entries: 2 bytes
/// - reserved: 2 bytes
#[derive(Debug, Clone, Copy)]
pub struct FsmHeader {
    /// First heap page number tracked by this FSM page.
    pub first_page_num: u32,
    /// Number of valid entries.
    pub num_entries: u16,
    /// Reserved for future use.
    pub reserved: u16,
}

impl FsmHeader {
    /// Size of the FSM header in bytes.
    pub const SIZE: usize = 8;

    /// Offset of FSM header in page (after PageHeader).
    pub const OFFSET: usize = PageHeader::SIZE;

    /// Creates a new FSM header.
    pub fn new(first_page_num: u32) -> Self {
        Self {
            first_page_num,
            num_entries: 0,
            reserved: 0,
        }
    }

    /// Serializes to bytes.
    pub fn to_bytes(&self) -> [u8; Self::SIZE] {
        let mut buf = [0u8; Self::SIZE];
        buf[0..4].copy_from_slice(&self.first_page_num.to_le_bytes());
        buf[4..6].copy_from_slice(&self.num_entries.to_le_bytes());
        buf[6..8].copy_from_slice(&self.reserved.to_le_bytes());
        buf
    }

    /// Deserializes from bytes.
    pub fn from_bytes(buf: &[u8]) -> Self {
        Self {
            first_page_num: u32::from_le_bytes([buf[0], buf[1], buf[2], buf[3]]),
            num_entries: u16::from_le_bytes([buf[4], buf[5]]),
            reserved: u16::from_le_bytes([buf[6], buf[7]]),
        }
    }
}

impl Default for FsmHeader {
    fn default() -> Self {
        Self::new(0)
    }
}

/// Converts actual free space to FSM category (0-255).
///
/// Category 0 means no space, 255 means full page of space.
/// Each category represents PAGE_SIZE / 256 bytes.
pub fn space_to_category(free_space: usize) -> u8 {
    let max_space = PAGE_SIZE - PageHeader::SIZE;
    if free_space >= max_space {
        255
    } else {
        ((free_space * 255) / max_space) as u8
    }
}

/// Converts FSM category back to minimum guaranteed free space.
pub fn category_to_min_space(category: u8) -> usize {
    let max_space = PAGE_SIZE - PageHeader::SIZE;
    (category as usize * max_space) / 255
}

/// Free space map page.
pub struct FsmPage {
    /// Page data buffer.
    data: Box<[u8; PAGE_SIZE]>,
}

impl FsmPage {
    /// Data start offset after headers.
    const DATA_START: usize = PageHeader::SIZE + FsmHeader::SIZE;

    /// Creates a new empty FSM page.
    pub fn new(page_id: PageId, first_page_num: u32) -> Self {
        let mut data = Box::new([0u8; PAGE_SIZE]);

        // Initialize page header
        let page_header = PageHeader::new(page_id, PageType::FreeSpaceMap);
        data[..PageHeader::SIZE].copy_from_slice(&page_header.to_bytes());

        // Initialize FSM header
        let fsm_header = FsmHeader::new(first_page_num);
        let offset = FsmHeader::OFFSET;
        data[offset..offset + FsmHeader::SIZE].copy_from_slice(&fsm_header.to_bytes());

        Self { data }
    }

    /// Creates an FSM page from raw bytes.
    pub fn from_bytes(data: [u8; PAGE_SIZE]) -> Self {
        Self {
            data: Box::new(data),
        }
    }

    /// Returns the raw page data.
    pub fn as_bytes(&self) -> &[u8; PAGE_SIZE] {
        &self.data
    }

    /// Returns the FSM header.
    fn fsm_header(&self) -> FsmHeader {
        let offset = FsmHeader::OFFSET;
        FsmHeader::from_bytes(&self.data[offset..offset + FsmHeader::SIZE])
    }

    /// Writes the FSM header.
    fn set_fsm_header(&mut self, header: FsmHeader) {
        let offset = FsmHeader::OFFSET;
        self.data[offset..offset + FsmHeader::SIZE].copy_from_slice(&header.to_bytes());
    }

    /// Returns the first page number tracked by this FSM page.
    pub fn first_page_num(&self) -> u32 {
        self.fsm_header().first_page_num
    }

    /// Returns the number of entries.
    pub fn num_entries(&self) -> u16 {
        self.fsm_header().num_entries
    }

    /// Gets the space category for a page.
    ///
    /// Returns None if the page is not tracked by this FSM page.
    pub fn get_space(&self, page_num: u32) -> Option<u8> {
        let header = self.fsm_header();
        if page_num < header.first_page_num {
            return None;
        }

        let index = (page_num - header.first_page_num) as usize;
        if index >= ENTRIES_PER_FSM_PAGE || index >= header.num_entries as usize {
            return None;
        }

        Some(self.data[Self::DATA_START + index])
    }

    /// Sets the space category for a page.
    ///
    /// Returns error if the page is not tracked by this FSM page.
    pub fn set_space(&mut self, page_num: u32, category: u8) -> Result<()> {
        let mut header = self.fsm_header();
        if page_num < header.first_page_num {
            return Err(ZyronError::IoError(format!(
                "page {} is before FSM range starting at {}",
                page_num, header.first_page_num
            )));
        }

        let index = (page_num - header.first_page_num) as usize;
        if index >= ENTRIES_PER_FSM_PAGE {
            return Err(ZyronError::IoError(format!(
                "page {} is beyond FSM range",
                page_num
            )));
        }

        self.data[Self::DATA_START + index] = category;

        // Update num_entries if needed
        if index >= header.num_entries as usize {
            header.num_entries = (index + 1) as u16;
            self.set_fsm_header(header);
        }

        Ok(())
    }

    /// Updates the space for a page using actual free space bytes.
    pub fn update_space(&mut self, page_num: u32, free_space: usize) -> Result<()> {
        let category = space_to_category(free_space);
        self.set_space(page_num, category)
    }

    /// Finds a page with at least the requested free space.
    ///
    /// Returns the page number of a suitable page, or None if no page has enough space.
    pub fn find_page_with_space(&self, min_space: usize) -> Option<u32> {
        let header = self.fsm_header();
        let min_category = space_to_category(min_space);

        for i in 0..header.num_entries as usize {
            let category = self.data[Self::DATA_START + i];
            if category >= min_category {
                return Some(header.first_page_num + i as u32);
            }
        }

        None
    }

    /// Returns the last page number tracked by this FSM page.
    pub fn last_page_num(&self) -> Option<u32> {
        let header = self.fsm_header();
        if header.num_entries == 0 {
            None
        } else {
            Some(header.first_page_num + header.num_entries as u32 - 1)
        }
    }

    /// Checks if this FSM page can track the given page number.
    pub fn can_track(&self, page_num: u32) -> bool {
        let header = self.fsm_header();
        page_num >= header.first_page_num
            && (page_num - header.first_page_num) < ENTRIES_PER_FSM_PAGE as u32
    }
}

/// Free space map manager for a heap file.
///
/// Manages multiple FSM pages to track free space across all heap pages.
pub struct FreeSpaceMap {
    /// File ID this FSM is for.
    file_id: u32,
    /// Page ID of the first FSM page.
    first_fsm_page: PageId,
}

impl FreeSpaceMap {
    /// Creates a new free space map.
    pub fn new(file_id: u32, first_fsm_page: PageId) -> Self {
        Self {
            file_id,
            first_fsm_page,
        }
    }

    /// Returns the file ID.
    pub fn file_id(&self) -> u32 {
        self.file_id
    }

    /// Returns the first FSM page ID.
    pub fn first_fsm_page(&self) -> PageId {
        self.first_fsm_page
    }

    /// Computes which FSM page tracks a given heap page.
    pub fn fsm_page_for(&self, heap_page_num: u32) -> u32 {
        heap_page_num / ENTRIES_PER_FSM_PAGE as u32
    }

    /// Computes the FSM page ID for a given FSM page number.
    pub fn fsm_page_id(&self, fsm_page_num: u32) -> PageId {
        PageId::new(
            self.first_fsm_page.file_id,
            self.first_fsm_page.page_num + fsm_page_num,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fsm_header_roundtrip() {
        let header = FsmHeader {
            first_page_num: 1000,
            num_entries: 500,
            reserved: 0,
        };

        let bytes = header.to_bytes();
        let recovered = FsmHeader::from_bytes(&bytes);

        assert_eq!(recovered.first_page_num, 1000);
        assert_eq!(recovered.num_entries, 500);
    }

    #[test]
    fn test_space_to_category() {
        // No space = 0
        assert_eq!(space_to_category(0), 0);

        // Full page = 255
        let max_space = PAGE_SIZE - PageHeader::SIZE;
        assert_eq!(space_to_category(max_space), 255);

        // Half space = ~127
        let half_space = max_space / 2;
        let category = space_to_category(half_space);
        assert!(category >= 126 && category <= 128);
    }

    #[test]
    fn test_category_to_min_space() {
        // Category 0 = 0 space
        assert_eq!(category_to_min_space(0), 0);

        // Category 255 = max space
        let max_space = PAGE_SIZE - PageHeader::SIZE;
        assert_eq!(category_to_min_space(255), max_space);

        // Roundtrip should be conservative (min space <= original)
        let original_space = 1000;
        let category = space_to_category(original_space);
        let min_space = category_to_min_space(category);
        assert!(min_space <= original_space);
    }

    #[test]
    fn test_fsm_page_new() {
        let page = FsmPage::new(PageId::new(0, 0), 100);

        assert_eq!(page.first_page_num(), 100);
        assert_eq!(page.num_entries(), 0);
    }

    #[test]
    fn test_fsm_page_set_get_space() {
        let mut page = FsmPage::new(PageId::new(0, 0), 0);

        page.set_space(5, 128).unwrap();
        assert_eq!(page.get_space(5), Some(128));
        assert_eq!(page.num_entries(), 6); // 0-5 inclusive

        page.set_space(10, 200).unwrap();
        assert_eq!(page.get_space(10), Some(200));
        assert_eq!(page.num_entries(), 11);
    }

    #[test]
    fn test_fsm_page_update_space() {
        let mut page = FsmPage::new(PageId::new(0, 0), 0);

        page.update_space(0, 1000).unwrap();
        let category = page.get_space(0).unwrap();
        assert!(category > 0);

        page.update_space(0, 0).unwrap();
        assert_eq!(page.get_space(0), Some(0));
    }

    #[test]
    fn test_fsm_page_find_page_with_space() {
        let mut page = FsmPage::new(PageId::new(0, 0), 0);

        // Set up pages with varying space
        page.set_space(0, 10).unwrap(); // Little space
        page.set_space(1, 50).unwrap(); // Some space
        page.set_space(2, 200).unwrap(); // Lots of space
        page.set_space(3, 100).unwrap(); // Medium space

        // Find page with at least category 150
        let found = page.find_page_with_space(category_to_min_space(150));
        assert_eq!(found, Some(2));

        // Find page with at least category 50
        let found = page.find_page_with_space(category_to_min_space(50));
        assert_eq!(found, Some(1));

        // Find page with category 0 (any space)
        let found = page.find_page_with_space(0);
        assert_eq!(found, Some(0));

        // No page with category 255
        let found = page.find_page_with_space(category_to_min_space(255));
        assert!(found.is_none());
    }

    #[test]
    fn test_fsm_page_out_of_range() {
        let mut page = FsmPage::new(PageId::new(0, 0), 100);

        // Page before range
        assert!(page.set_space(50, 100).is_err());
        assert_eq!(page.get_space(50), None);

        // Page within range
        assert!(page.set_space(100, 100).is_ok());
        assert!(page.set_space(200, 100).is_ok());
    }

    #[test]
    fn test_fsm_page_last_page_num() {
        let mut page = FsmPage::new(PageId::new(0, 0), 0);

        assert_eq!(page.last_page_num(), None);

        page.set_space(0, 100).unwrap();
        assert_eq!(page.last_page_num(), Some(0));

        page.set_space(10, 100).unwrap();
        assert_eq!(page.last_page_num(), Some(10));
    }

    #[test]
    fn test_fsm_page_can_track() {
        let page = FsmPage::new(PageId::new(0, 0), 100);

        assert!(!page.can_track(50)); // Before range
        assert!(page.can_track(100)); // Start of range
        assert!(page.can_track(100 + ENTRIES_PER_FSM_PAGE as u32 - 1)); // End of range
        assert!(!page.can_track(100 + ENTRIES_PER_FSM_PAGE as u32)); // After range
    }

    #[test]
    fn test_fsm_page_from_bytes() {
        let mut page = FsmPage::new(PageId::new(0, 0), 0);
        page.set_space(5, 150).unwrap();

        let bytes = *page.as_bytes();
        let recovered = FsmPage::from_bytes(bytes);

        assert_eq!(recovered.first_page_num(), 0);
        assert_eq!(recovered.get_space(5), Some(150));
    }

    #[test]
    fn test_entries_per_fsm_page() {
        // Should be able to track many pages per FSM page
        assert!(ENTRIES_PER_FSM_PAGE > 16000);
    }

    #[test]
    fn test_free_space_map_new() {
        let fsm = FreeSpaceMap::new(0, PageId::new(1, 0));

        assert_eq!(fsm.file_id(), 0);
        assert_eq!(fsm.first_fsm_page(), PageId::new(1, 0));
    }

    #[test]
    fn test_free_space_map_fsm_page_for() {
        let fsm = FreeSpaceMap::new(0, PageId::new(1, 0));

        assert_eq!(fsm.fsm_page_for(0), 0);
        assert_eq!(fsm.fsm_page_for(ENTRIES_PER_FSM_PAGE as u32 - 1), 0);
        assert_eq!(fsm.fsm_page_for(ENTRIES_PER_FSM_PAGE as u32), 1);
        assert_eq!(fsm.fsm_page_for(ENTRIES_PER_FSM_PAGE as u32 * 2), 2);
    }

    #[test]
    fn test_free_space_map_fsm_page_id() {
        let fsm = FreeSpaceMap::new(0, PageId::new(1, 100));

        assert_eq!(fsm.fsm_page_id(0), PageId::new(1, 100));
        assert_eq!(fsm.fsm_page_id(1), PageId::new(1, 101));
        assert_eq!(fsm.fsm_page_id(5), PageId::new(1, 105));
    }
}
