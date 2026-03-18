//! Custom zero-copy serialization traits for packed structs.
//!
//! Provides `AsBytes` and `FromBytes` traits for types with stable,
//! padding-free memory layouts (`#[repr(C, packed)]`). Each implementing
//! type must include compile-time const assertions verifying that the
//! struct size equals the sum of its field sizes and alignment is 1.

/// Trait for types that can be safely interpreted as a byte slice.
///
/// # Safety
///
/// The implementing type must be `#[repr(C, packed)]` with no padding bytes.
/// Implementors must include const assertions verifying:
/// - `size_of::<Self>()` equals the sum of all field sizes
/// - `align_of::<Self>()` equals 1
pub unsafe trait AsBytes: Sized {
    /// Returns a byte slice view of the struct's memory.
    #[inline]
    fn as_bytes(&self) -> &[u8] {
        unsafe {
            std::slice::from_raw_parts(
                self as *const Self as *const u8,
                std::mem::size_of::<Self>(),
            )
        }
    }

    /// Writes the struct's bytes into `buf` at the given offset.
    /// Returns the number of bytes written.
    #[inline]
    fn write_to(&self, buf: &mut [u8], offset: usize) -> usize {
        let size = std::mem::size_of::<Self>();
        buf[offset..offset + size].copy_from_slice(self.as_bytes());
        size
    }
}

/// Trait for types that can be safely read from a raw byte slice.
///
/// # Safety
///
/// The implementing type must be `#[repr(C, packed)]` with no padding bytes,
/// and must be valid for any bit pattern (all field types must be integers
/// or other trivially-valid types).
pub unsafe trait FromBytes: Sized {
    /// Reads a value from `buf` at the given byte offset using an unaligned read.
    #[inline]
    fn read_from(buf: &[u8], offset: usize) -> Self {
        assert!(
            offset + std::mem::size_of::<Self>() <= buf.len(),
            "read_from: buffer too small"
        );
        unsafe { std::ptr::read_unaligned(buf.as_ptr().add(offset) as *const Self) }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[repr(C, packed)]
    #[derive(Debug, PartialEq)]
    struct TestHeader {
        a: u64,
        b: u32,
        c: u8,
        d: u8,
        e: u16,
    }

    const _: () = {
        assert!(std::mem::size_of::<TestHeader>() == 8 + 4 + 1 + 1 + 2);
        assert!(std::mem::align_of::<TestHeader>() == 1);
    };

    unsafe impl AsBytes for TestHeader {}
    unsafe impl FromBytes for TestHeader {}

    #[test]
    fn test_roundtrip() {
        let original = TestHeader {
            a: 0x0102030405060708,
            b: 0x090A0B0C,
            c: 0x0D,
            d: 0x0E,
            e: 0x0F10,
        };

        let bytes = original.as_bytes();
        assert_eq!(bytes.len(), 16);

        let restored = TestHeader::read_from(bytes, 0);
        assert_eq!(original, restored);
    }

    #[test]
    fn test_write_to() {
        let header = TestHeader {
            a: 1,
            b: 2,
            c: 3,
            d: 4,
            e: 5,
        };

        let mut buf = [0u8; 32];
        let written = header.write_to(&mut buf, 8);
        assert_eq!(written, 16);

        let restored = TestHeader::read_from(&buf, 8);
        assert_eq!(header, restored);
    }

    #[test]
    #[should_panic(expected = "buffer too small")]
    fn test_read_from_too_small() {
        let buf = [0u8; 4];
        let _ = TestHeader::read_from(&buf, 0);
    }
}
