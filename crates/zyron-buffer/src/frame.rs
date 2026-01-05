//! Buffer frame management.

use parking_lot::RwLock;
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, Ordering};
use zyron_common::page::{PageId, PAGE_SIZE};

/// Sentinel value indicating no page is loaded in the frame.
const NO_PAGE: u64 = u64::MAX;

/// Unique identifier for a frame in the buffer pool.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct FrameId(pub u32);

impl FrameId {
    /// Invalid frame ID.
    pub const INVALID: FrameId = FrameId(u32::MAX);

    /// Returns true if this is a valid frame ID.
    pub fn is_valid(&self) -> bool {
        *self != Self::INVALID
    }
}

impl std::fmt::Display for FrameId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "frame:{}", self.0)
    }
}

/// A frame in the buffer pool holding a single page.
///
/// Each frame contains:
/// - The actual page data (PAGE_SIZE bytes)
/// - Metadata for buffer management (pin count, dirty flag, etc.)
pub struct BufferFrame {
    /// Frame identifier.
    frame_id: FrameId,
    /// The page currently stored in this frame (packed as u64, NO_PAGE = none).
    /// Layout: upper 32 bits = file_id, lower 32 bits = page_num.
    page_id: AtomicU64,
    /// Page data buffer.
    data: RwLock<Box<[u8; PAGE_SIZE]>>,
    /// Number of users currently accessing this page.
    pin_count: AtomicU32,
    /// Whether the page has been modified.
    is_dirty: AtomicBool,
    /// Reference bit for clock replacement algorithm.
    reference_bit: AtomicBool,
}

impl BufferFrame {
    /// Creates a new empty buffer frame.
    pub fn new(frame_id: FrameId) -> Self {
        Self {
            frame_id,
            page_id: AtomicU64::new(NO_PAGE),
            data: RwLock::new(Box::new([0u8; PAGE_SIZE])),
            pin_count: AtomicU32::new(0),
            is_dirty: AtomicBool::new(false),
            reference_bit: AtomicBool::new(false),
        }
    }

    /// Packs a PageId into a u64.
    #[inline(always)]
    fn pack_page_id(page_id: PageId) -> u64 {
        ((page_id.file_id as u64) << 32) | (page_id.page_num as u64)
    }

    /// Unpacks a u64 into a PageId.
    #[inline(always)]
    fn unpack_page_id(packed: u64) -> PageId {
        PageId {
            file_id: (packed >> 32) as u32,
            page_num: packed as u32,
        }
    }

    /// Returns the frame ID.
    #[inline]
    pub fn frame_id(&self) -> FrameId {
        self.frame_id
    }

    /// Returns the page ID currently stored in this frame.
    #[inline]
    pub fn page_id(&self) -> Option<PageId> {
        let packed = self.page_id.load(Ordering::Acquire);
        if packed == NO_PAGE {
            None
        } else {
            Some(Self::unpack_page_id(packed))
        }
    }

    /// Sets the page ID for this frame.
    #[inline]
    pub fn set_page_id(&self, page_id: Option<PageId>) {
        let packed = match page_id {
            Some(pid) => Self::pack_page_id(pid),
            None => NO_PAGE,
        };
        self.page_id.store(packed, Ordering::Release);
    }

    /// Returns the current pin count.
    #[inline]
    pub fn pin_count(&self) -> u32 {
        self.pin_count.load(Ordering::Acquire)
    }

    /// Increments the pin count and returns the previous pin count.
    /// Returns 0 if the frame was unpinned before this call.
    #[inline]
    pub fn pin(&self) -> u32 {
        let prev = self.pin_count.fetch_add(1, Ordering::AcqRel);
        self.reference_bit.store(true, Ordering::Relaxed);
        prev
    }

    /// Decrements the pin count.
    ///
    /// Returns the new pin count.
    #[inline]
    pub fn unpin(&self) -> u32 {
        let prev = self.pin_count.fetch_sub(1, Ordering::AcqRel);
        if prev == 0 {
            // Underflow protection: restore to 0
            self.pin_count.store(0, Ordering::Release);
            return 0;
        }
        prev - 1
    }

    /// Returns true if this frame is pinned.
    #[inline]
    pub fn is_pinned(&self) -> bool {
        self.pin_count.load(Ordering::Acquire) > 0
    }

    /// Returns true if this frame is dirty.
    #[inline]
    pub fn is_dirty(&self) -> bool {
        self.is_dirty.load(Ordering::Acquire)
    }

    /// Marks this frame as dirty.
    #[inline]
    pub fn set_dirty(&self, dirty: bool) {
        self.is_dirty.store(dirty, Ordering::Release);
    }

    /// Returns the reference bit value.
    #[inline]
    pub fn reference_bit(&self) -> bool {
        self.reference_bit.load(Ordering::Relaxed)
    }

    /// Sets the reference bit.
    #[inline]
    pub fn set_reference_bit(&self, value: bool) {
        self.reference_bit.store(value, Ordering::Relaxed);
    }

    /// Returns true if this frame is empty (no page loaded).
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.page_id.load(Ordering::Acquire) == NO_PAGE
    }

    /// Reads the page data.
    #[inline]
    pub fn read_data(&self) -> parking_lot::RwLockReadGuard<'_, Box<[u8; PAGE_SIZE]>> {
        self.data.read()
    }

    /// Writes to the page data.
    #[inline]
    pub fn write_data(&self) -> parking_lot::RwLockWriteGuard<'_, Box<[u8; PAGE_SIZE]>> {
        self.data.write()
    }

    /// Copies data into the frame.
    #[inline]
    pub fn copy_from(&self, src: &[u8]) {
        let mut data = self.data.write();
        let len = src.len().min(PAGE_SIZE);
        data[..len].copy_from_slice(&src[..len]);
    }

    /// Copies data out of the frame.
    #[inline]
    pub fn copy_to(&self, dst: &mut [u8]) {
        let data = self.data.read();
        let len = dst.len().min(PAGE_SIZE);
        dst[..len].copy_from_slice(&data[..len]);
    }

    /// Resets the frame to empty state.
    #[inline]
    pub fn reset(&self) {
        self.page_id.store(NO_PAGE, Ordering::Release);
        self.pin_count.store(0, Ordering::Release);
        self.is_dirty.store(false, Ordering::Release);
        self.reference_bit.store(false, Ordering::Relaxed);
        // Zero out data for security
        let mut data = self.data.write();
        data.fill(0);
    }
}

impl std::fmt::Debug for BufferFrame {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("BufferFrame")
            .field("frame_id", &self.frame_id)
            .field("page_id", &self.page_id())
            .field("pin_count", &self.pin_count())
            .field("is_dirty", &self.is_dirty())
            .field("reference_bit", &self.reference_bit())
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_frame_id_validity() {
        let valid = FrameId(0);
        let invalid = FrameId::INVALID;

        assert!(valid.is_valid());
        assert!(!invalid.is_valid());
    }

    #[test]
    fn test_frame_id_display() {
        let frame_id = FrameId(42);
        assert_eq!(frame_id.to_string(), "frame:42");
    }

    #[test]
    fn test_buffer_frame_new() {
        let frame = BufferFrame::new(FrameId(0));

        assert_eq!(frame.frame_id(), FrameId(0));
        assert!(frame.page_id().is_none());
        assert_eq!(frame.pin_count(), 0);
        assert!(!frame.is_dirty());
        assert!(!frame.reference_bit());
        assert!(frame.is_empty());
    }

    #[test]
    fn test_buffer_frame_pin_unpin() {
        let frame = BufferFrame::new(FrameId(0));

        assert!(!frame.is_pinned());

        frame.pin();
        assert!(frame.is_pinned());
        assert_eq!(frame.pin_count(), 1);
        assert!(frame.reference_bit());

        frame.pin();
        assert_eq!(frame.pin_count(), 2);

        frame.unpin();
        assert_eq!(frame.pin_count(), 1);
        assert!(frame.is_pinned());

        frame.unpin();
        assert_eq!(frame.pin_count(), 0);
        assert!(!frame.is_pinned());
    }

    #[test]
    fn test_buffer_frame_unpin_underflow() {
        let frame = BufferFrame::new(FrameId(0));

        // Unpin when already at 0 should stay at 0
        frame.unpin();
        assert_eq!(frame.pin_count(), 0);
    }

    #[test]
    fn test_buffer_frame_dirty() {
        let frame = BufferFrame::new(FrameId(0));

        assert!(!frame.is_dirty());

        frame.set_dirty(true);
        assert!(frame.is_dirty());

        frame.set_dirty(false);
        assert!(!frame.is_dirty());
    }

    #[test]
    fn test_buffer_frame_page_id() {
        let frame = BufferFrame::new(FrameId(0));
        let page_id = PageId::new(1, 100);

        assert!(frame.page_id().is_none());
        assert!(frame.is_empty());

        frame.set_page_id(Some(page_id));
        assert_eq!(frame.page_id(), Some(page_id));
        assert!(!frame.is_empty());

        frame.set_page_id(None);
        assert!(frame.page_id().is_none());
        assert!(frame.is_empty());
    }

    #[test]
    fn test_buffer_frame_data_access() {
        let frame = BufferFrame::new(FrameId(0));

        // Write some data
        {
            let mut data = frame.write_data();
            data[0] = 0xAB;
            data[1] = 0xCD;
        }

        // Read it back
        {
            let data = frame.read_data();
            assert_eq!(data[0], 0xAB);
            assert_eq!(data[1], 0xCD);
        }
    }

    #[test]
    fn test_buffer_frame_copy_from_to() {
        let frame = BufferFrame::new(FrameId(0));
        let src = [1u8, 2, 3, 4, 5];

        frame.copy_from(&src);

        let mut dst = [0u8; 5];
        frame.copy_to(&mut dst);

        assert_eq!(dst, src);
    }

    #[test]
    fn test_buffer_frame_reset() {
        let frame = BufferFrame::new(FrameId(0));

        // Set up frame state
        frame.set_page_id(Some(PageId::new(1, 1)));
        frame.pin();
        frame.set_dirty(true);
        frame.set_reference_bit(true);
        {
            let mut data = frame.write_data();
            data[0] = 0xFF;
        }

        // Reset
        frame.reset();

        // Verify all state is cleared
        assert!(frame.page_id().is_none());
        assert_eq!(frame.pin_count(), 0);
        assert!(!frame.is_dirty());
        assert!(!frame.reference_bit());
        assert!(frame.is_empty());

        let data = frame.read_data();
        assert_eq!(data[0], 0);
    }

    #[test]
    fn test_buffer_frame_reference_bit() {
        let frame = BufferFrame::new(FrameId(0));

        assert!(!frame.reference_bit());

        frame.set_reference_bit(true);
        assert!(frame.reference_bit());

        frame.set_reference_bit(false);
        assert!(!frame.reference_bit());
    }

    #[test]
    fn test_buffer_frame_pin_sets_reference() {
        let frame = BufferFrame::new(FrameId(0));

        assert!(!frame.reference_bit());
        frame.pin();
        assert!(frame.reference_bit());
    }

    #[test]
    fn test_buffer_frame_debug() {
        let frame = BufferFrame::new(FrameId(5));
        frame.set_page_id(Some(PageId::new(1, 10)));
        frame.pin();
        frame.set_dirty(true);

        let debug_str = format!("{:?}", frame);
        assert!(debug_str.contains("BufferFrame"));
        assert!(debug_str.contains("frame_id"));
        assert!(debug_str.contains("pin_count"));
    }
}
