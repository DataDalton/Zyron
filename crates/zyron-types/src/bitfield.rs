//! Named bitfield operations on u64 values.
//!
//! Provides bit manipulation functions for feature flags, permission masks,
//! and status tracking stored in a single 8-byte column.

use zyron_common::{Result, ZyronError};

/// Sets the bit at the given position (0-63). Returns error if position >= 64.
pub fn bitfield_set(field: u64, position: u8) -> Result<u64> {
    if position >= 64 {
        return Err(ZyronError::ExecutionError(format!(
            "Bit position {} out of range (0-63)",
            position
        )));
    }
    Ok(field | (1u64 << position))
}

/// Clears the bit at the given position (0-63). Returns error if position >= 64.
pub fn bitfield_clear(field: u64, position: u8) -> Result<u64> {
    if position >= 64 {
        return Err(ZyronError::ExecutionError(format!(
            "Bit position {} out of range (0-63)",
            position
        )));
    }
    Ok(field & !(1u64 << position))
}

/// Toggles the bit at the given position (0-63). Returns error if position >= 64.
pub fn bitfield_toggle(field: u64, position: u8) -> Result<u64> {
    if position >= 64 {
        return Err(ZyronError::ExecutionError(format!(
            "Bit position {} out of range (0-63)",
            position
        )));
    }
    Ok(field ^ (1u64 << position))
}

/// Tests whether the bit at the given position is set. Returns error if position >= 64.
pub fn bitfield_test(field: u64, position: u8) -> Result<bool> {
    if position >= 64 {
        return Err(ZyronError::ExecutionError(format!(
            "Bit position {} out of range (0-63)",
            position
        )));
    }
    Ok((field >> position) & 1 == 1)
}

/// Returns the number of set bits (population count).
pub fn bitfield_count(field: u64) -> u32 {
    field.count_ones()
}

/// Returns true if all bits in the mask are set in the field.
pub fn bitfield_all(field: u64, mask: u64) -> bool {
    field & mask == mask
}

/// Returns true if any bit in the mask is set in the field.
pub fn bitfield_any(field: u64, mask: u64) -> bool {
    field & mask != 0
}

/// Bitwise AND of two fields.
pub fn bitfield_and(a: u64, b: u64) -> u64 {
    a & b
}

/// Bitwise OR of two fields.
pub fn bitfield_or(a: u64, b: u64) -> u64 {
    a | b
}

/// Bitwise XOR of two fields.
pub fn bitfield_xor(a: u64, b: u64) -> u64 {
    a ^ b
}

/// Bitwise NOT of a field.
pub fn bitfield_not(field: u64) -> u64 {
    !field
}

/// Returns a Vec of all set bit positions, from lowest to highest.
pub fn bitfield_to_positions(field: u64) -> Vec<u8> {
    let mut positions = Vec::with_capacity(field.count_ones() as usize);
    let mut remaining = field;
    while remaining != 0 {
        let pos = remaining.trailing_zeros() as u8;
        positions.push(pos);
        remaining &= remaining - 1; // clear lowest set bit
    }
    positions
}

/// Creates a bitfield from a slice of bit positions. Positions >= 64 are ignored.
pub fn bitfield_from_positions(positions: &[u8]) -> u64 {
    let mut field = 0u64;
    for &pos in positions {
        if pos < 64 {
            field |= 1u64 << pos;
        }
    }
    field
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_set_and_test() {
        let f = bitfield_set(0, 3).unwrap();
        assert_eq!(f, 8);
        assert!(bitfield_test(f, 3).unwrap());
        assert!(!bitfield_test(f, 2).unwrap());
    }

    #[test]
    fn test_set_multiple() {
        let f = bitfield_set(0, 0).unwrap();
        let f = bitfield_set(f, 7).unwrap();
        let f = bitfield_set(f, 63).unwrap();
        assert!(bitfield_test(f, 0).unwrap());
        assert!(bitfield_test(f, 7).unwrap());
        assert!(bitfield_test(f, 63).unwrap());
        assert!(!bitfield_test(f, 1).unwrap());
    }

    #[test]
    fn test_clear() {
        let f = bitfield_set(0xFF, 3).unwrap();
        let f = bitfield_clear(f, 3).unwrap();
        assert!(!bitfield_test(f, 3).unwrap());
        assert!(bitfield_test(f, 0).unwrap());
    }

    #[test]
    fn test_toggle() {
        let f = bitfield_toggle(0, 5).unwrap();
        assert!(bitfield_test(f, 5).unwrap());
        let f = bitfield_toggle(f, 5).unwrap();
        assert!(!bitfield_test(f, 5).unwrap());
    }

    #[test]
    fn test_out_of_range() {
        assert!(bitfield_set(0, 64).is_err());
        assert!(bitfield_clear(0, 64).is_err());
        assert!(bitfield_toggle(0, 64).is_err());
        assert!(bitfield_test(0, 64).is_err());
        assert!(bitfield_set(0, 255).is_err());
    }

    #[test]
    fn test_count() {
        assert_eq!(bitfield_count(0), 0);
        assert_eq!(bitfield_count(0b1010_1010), 4);
        assert_eq!(bitfield_count(u64::MAX), 64);
    }

    #[test]
    fn test_all() {
        let f = 0b1111u64;
        assert!(bitfield_all(f, 0b1010));
        assert!(bitfield_all(f, 0b1111));
        assert!(!bitfield_all(f, 0b1_0000));
    }

    #[test]
    fn test_any() {
        let f = 0b1010u64;
        assert!(bitfield_any(f, 0b1000));
        assert!(bitfield_any(f, 0b0010));
        assert!(!bitfield_any(f, 0b0101));
    }

    #[test]
    fn test_bitwise_ops() {
        assert_eq!(bitfield_and(0xFF, 0x0F), 0x0F);
        assert_eq!(bitfield_or(0xF0, 0x0F), 0xFF);
        assert_eq!(bitfield_xor(0xFF, 0x0F), 0xF0);
        assert_eq!(bitfield_not(0), u64::MAX);
    }

    #[test]
    fn test_to_positions() {
        assert_eq!(bitfield_to_positions(0), vec![]);
        assert_eq!(bitfield_to_positions(0b1010), vec![1, 3]);
        assert_eq!(bitfield_to_positions(0b1000_0001), vec![0, 7]);
        assert_eq!(bitfield_to_positions(1u64 << 63), vec![63]);
    }

    #[test]
    fn test_from_positions() {
        assert_eq!(bitfield_from_positions(&[]), 0);
        assert_eq!(bitfield_from_positions(&[0, 3, 7]), 0b1000_1001);
        assert_eq!(bitfield_from_positions(&[63]), 1u64 << 63);
    }

    #[test]
    fn test_from_positions_ignores_invalid() {
        assert_eq!(bitfield_from_positions(&[0, 64, 255]), 1);
    }

    #[test]
    fn test_roundtrip_positions() {
        let original: u64 = 0b1010_0110_1001;
        let positions = bitfield_to_positions(original);
        let recovered = bitfield_from_positions(&positions);
        assert_eq!(original, recovered);
    }

    #[test]
    fn test_set_idempotent() {
        let f = bitfield_set(0, 5).unwrap();
        let f2 = bitfield_set(f, 5).unwrap();
        assert_eq!(f, f2);
    }

    #[test]
    fn test_zero_field() {
        assert!(!bitfield_any(0, u64::MAX));
        assert!(bitfield_all(0, 0));
        assert_eq!(bitfield_count(0), 0);
    }
}
