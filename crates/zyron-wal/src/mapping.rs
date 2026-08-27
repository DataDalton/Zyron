//! Anonymous page mappings for the WAL ring's two fixed-size arrays.
//!
//! Both arrays are sized to the ring capacity, sixteen megabytes each by
//! default. Requesting them from the allocator as zeroed blocks costs a write
//! to every byte: mimalloc commits a large block and clears it up front, which
//! measures at 2.8ms per sixteen megabytes and leaves the whole region
//! resident before the WAL has logged a single record.
//!
//! Mapping the pages from the operating system instead hands back demand-zero
//! pages. Creating the ring performs no writes, and the mapping occupies only
//! the pages the workload actually reaches, so a WAL that keeps a few kilobytes
//! in flight does not hold thirty-two megabytes resident.

use std::ptr::NonNull;

use zyron_common::{Result, ZyronError};

/// A page-aligned anonymous mapping whose bytes read as zero until written.
///
/// The mapping is released when dropped. Nothing else may hold a pointer into
/// it at that point, which the ring guarantees by owning its mappings for its
/// whole lifetime and joining the flush thread before dropping.
pub(crate) struct ZeroedMapping {
    ptr: NonNull<u8>,
    len: usize,
}

// The mapping is a plain byte region with no interior pointers and no thread
// affinity. Callers coordinate access to its contents; sharing the owner
// across threads is what the ring needs and is safe on its own.
unsafe impl Send for ZeroedMapping {}
unsafe impl Sync for ZeroedMapping {}

impl ZeroedMapping {
    /// Maps `len` zero-filled bytes. `len` must be non-zero.
    pub(crate) fn new(len: usize) -> Result<Self> {
        assert!(len > 0, "zero-length mapping requested");
        let ptr = platform::map(len)?;
        Ok(Self { ptr, len })
    }

    /// Start of the mapping. Valid for `len` bytes for the owner's lifetime.
    #[inline]
    pub(crate) fn as_ptr(&self) -> *mut u8 {
        self.ptr.as_ptr()
    }

    /// Length of the mapping in bytes.
    #[inline]
    pub(crate) fn len(&self) -> usize {
        self.len
    }
}

impl Drop for ZeroedMapping {
    fn drop(&mut self) {
        // Nothing recovers from a failed unmap, and the process is losing the
        // region either way, so the result is dropped rather than panicking in
        // a destructor
        unsafe { platform::unmap(self.ptr, self.len) };
    }
}

#[cfg(windows)]
mod platform {
    use super::*;

    use windows_sys::Win32::System::Memory::{
        MEM_COMMIT, MEM_RELEASE, MEM_RESERVE, PAGE_READWRITE, VirtualAlloc, VirtualFree,
    };

    pub(super) fn map(len: usize) -> Result<NonNull<u8>> {
        // MEM_COMMIT charges the region against the commit limit but leaves
        // the pages demand-zero: the kernel supplies a zeroed page on first
        // touch, so this call writes nothing
        let ptr = unsafe {
            VirtualAlloc(
                std::ptr::null(),
                len,
                MEM_COMMIT | MEM_RESERVE,
                PAGE_READWRITE,
            )
        };
        NonNull::new(ptr.cast::<u8>()).ok_or_else(|| {
            ZyronError::Internal(format!(
                "VirtualAlloc of {} bytes for the WAL ring failed with error {}",
                len,
                std::io::Error::last_os_error()
            ))
        })
    }

    pub(super) unsafe fn unmap(ptr: NonNull<u8>, _len: usize) {
        // MEM_RELEASE frees the whole reservation and requires a zero size
        unsafe { VirtualFree(ptr.as_ptr().cast(), 0, MEM_RELEASE) };
    }
}

#[cfg(not(windows))]
mod platform {
    use super::*;

    pub(super) fn map(len: usize) -> Result<NonNull<u8>> {
        // MAP_ANONYMOUS pages start zeroed and are faulted in on first touch,
        // so this call writes nothing
        let ptr = unsafe {
            libc::mmap(
                std::ptr::null_mut(),
                len,
                libc::PROT_READ | libc::PROT_WRITE,
                libc::MAP_PRIVATE | libc::MAP_ANONYMOUS,
                -1,
                0,
            )
        };
        if ptr == libc::MAP_FAILED {
            return Err(ZyronError::Internal(format!(
                "mmap of {} bytes for the WAL ring failed: {}",
                len,
                std::io::Error::last_os_error()
            )));
        }
        NonNull::new(ptr.cast::<u8>())
            .ok_or_else(|| ZyronError::Internal(format!("mmap of {} bytes returned null", len)))
    }

    pub(super) unsafe fn unmap(ptr: NonNull<u8>, len: usize) {
        unsafe { libc::munmap(ptr.as_ptr().cast(), len) };
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mapping_reads_as_zero_and_holds_writes() {
        const LEN: usize = 4 * 1024 * 1024;
        let map = ZeroedMapping::new(LEN).unwrap();
        assert_eq!(map.len(), LEN);

        let bytes = unsafe { std::slice::from_raw_parts(map.as_ptr(), LEN) };
        assert!(bytes.iter().all(|b| *b == 0), "mapping was not zero filled");

        // Touch the first byte of every page and the last byte of the region,
        // which is where a wrong length would fault
        unsafe {
            for page in 0..LEN / 4096 {
                map.as_ptr().add(page * 4096).write(0xA5);
            }
            map.as_ptr().add(LEN - 1).write(0x5A);
        }
        let bytes = unsafe { std::slice::from_raw_parts(map.as_ptr(), LEN) };
        assert_eq!(bytes[0], 0xA5);
        assert_eq!(bytes[4096], 0xA5);
        assert_eq!(bytes[LEN - 1], 0x5A);
        assert_eq!(bytes[1], 0, "untouched bytes must stay zero");
    }

    #[test]
    fn many_mappings_are_independent() {
        let a = ZeroedMapping::new(1 << 20).unwrap();
        let b = ZeroedMapping::new(1 << 20).unwrap();
        unsafe { a.as_ptr().write(7) };
        assert_eq!(unsafe { b.as_ptr().read() }, 0);
        assert_eq!(unsafe { a.as_ptr().read() }, 7);
    }
}
