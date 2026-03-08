//! Constant encoding for column segments where every row has the same value.
//! Stores the single value once. Zero per-row storage cost.
//! Predicate evaluation compares against the single stored value,
//! producing an all-ones or all-zeros bitmask in O(1).

use crate::encoding::{Encoding, EncodingType, Predicate};
use zyron_common::{Result, ZyronError};

pub struct ConstantEncoding;

/// Encoded format:
///   [0..4]  value_size: u32 (little-endian)
///   [4..]   value: [u8; value_size]
impl Encoding for ConstantEncoding {
    fn encoding_type(&self) -> EncodingType {
        EncodingType::Constant
    }

    fn encode(&self, data: &[u8], row_count: usize, value_size: usize) -> Result<Vec<u8>> {
        if row_count == 0 {
            let mut out = Vec::with_capacity(4);
            out.extend_from_slice(&0u32.to_le_bytes());
            return Ok(out);
        }

        let expected_len = row_count * value_size;
        if data.len() < expected_len {
            return Err(ZyronError::EncodingFailed(
                "data shorter than expected for constant encoding".to_string(),
            ));
        }

        let first_value = &data[..value_size];

        // Verify all rows have the same value
        for i in 1..row_count {
            let offset = i * value_size;
            if &data[offset..offset + value_size] != first_value {
                return Err(ZyronError::EncodingFailed(
                    "not all values are identical for constant encoding".to_string(),
                ));
            }
        }

        let mut out = Vec::with_capacity(4 + value_size);
        out.extend_from_slice(&(value_size as u32).to_le_bytes());
        out.extend_from_slice(first_value);
        Ok(out)
    }

    fn decode(&self, encoded: &[u8], row_count: usize, value_size: usize) -> Result<Vec<u8>> {
        if row_count == 0 {
            return Ok(Vec::new());
        }

        if encoded.len() < 4 {
            return Err(ZyronError::DecodingFailed(
                "constant encoded data too short".to_string(),
            ));
        }

        let stored_size =
            u32::from_le_bytes([encoded[0], encoded[1], encoded[2], encoded[3]]) as usize;
        if stored_size != value_size {
            return Err(ZyronError::DecodingFailed(format!(
                "constant value_size mismatch: stored {}, expected {}",
                stored_size, value_size
            )));
        }

        if encoded.len() < 4 + value_size {
            return Err(ZyronError::DecodingFailed(
                "constant encoded data truncated".to_string(),
            ));
        }

        let value = &encoded[4..4 + value_size];
        let mut out = Vec::with_capacity(row_count * value_size);
        for _ in 0..row_count {
            out.extend_from_slice(value);
        }
        Ok(out)
    }

    fn eval_predicate(
        &self,
        encoded: &[u8],
        row_count: usize,
        value_size: usize,
        predicate: &Predicate,
    ) -> Result<Vec<u8>> {
        if row_count == 0 {
            return Ok(Vec::new());
        }

        if encoded.len() < 4 + value_size {
            return Err(ZyronError::DecodingFailed(
                "constant encoded data too short for predicate evaluation".to_string(),
            ));
        }

        let value = &encoded[4..4 + value_size];
        let bitmask_len = row_count.div_ceil(8);

        let matches = match predicate {
            Predicate::Equality(target) => value == *target,
            Predicate::Range { low, high } => {
                let above_low = match low {
                    Some(lo) => value >= *lo,
                    None => true,
                };
                let below_high = match high {
                    Some(hi) => value <= *hi,
                    None => true,
                };
                above_low && below_high
            }
            Predicate::In(values) => values.contains(&value),
        };

        if matches {
            // All rows match: set all bits
            let mut bitmask = vec![0xFFu8; bitmask_len];
            // Clear unused trailing bits in the last byte
            let trailing = row_count % 8;
            if trailing != 0 {
                bitmask[bitmask_len - 1] = (1u8 << trailing) - 1;
            }
            Ok(bitmask)
        } else {
            // No rows match
            Ok(vec![0u8; bitmask_len])
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_roundtrip_i32() {
        let enc = ConstantEncoding;
        let value = 42u32.to_le_bytes();
        let mut data = Vec::new();
        for _ in 0..100 {
            data.extend_from_slice(&value);
        }

        let encoded = enc.encode(&data, 100, 4).unwrap();
        // Header (4 bytes) + value (4 bytes) = 8 bytes total
        assert_eq!(encoded.len(), 8);

        let decoded = enc.decode(&encoded, 100, 4).unwrap();
        assert_eq!(decoded, data);
    }

    #[test]
    fn test_encode_non_constant_fails() {
        let enc = ConstantEncoding;
        let mut data = Vec::new();
        data.extend_from_slice(&1u32.to_le_bytes());
        data.extend_from_slice(&2u32.to_le_bytes());

        let result = enc.encode(&data, 2, 4);
        assert!(result.is_err());
    }

    #[test]
    fn test_empty() {
        let enc = ConstantEncoding;
        let encoded = enc.encode(&[], 0, 4).unwrap();
        let decoded = enc.decode(&encoded, 0, 4).unwrap();
        assert!(decoded.is_empty());
    }

    #[test]
    fn test_predicate_equality_match() {
        let enc = ConstantEncoding;
        let value = 42u32.to_le_bytes();
        let mut data = Vec::new();
        for _ in 0..10 {
            data.extend_from_slice(&value);
        }
        let encoded = enc.encode(&data, 10, 4).unwrap();

        let target = 42u32.to_le_bytes();
        let bitmask = enc
            .eval_predicate(&encoded, 10, 4, &Predicate::Equality(&target))
            .unwrap();
        // All 10 rows match: first byte = 0xFF, second byte = 0b00000011
        assert_eq!(bitmask[0], 0xFF);
        assert_eq!(bitmask[1], 0b00000011);
    }

    #[test]
    fn test_predicate_equality_no_match() {
        let enc = ConstantEncoding;
        let value = 42u32.to_le_bytes();
        let mut data = Vec::new();
        for _ in 0..10 {
            data.extend_from_slice(&value);
        }
        let encoded = enc.encode(&data, 10, 4).unwrap();

        let target = 99u32.to_le_bytes();
        let bitmask = enc
            .eval_predicate(&encoded, 10, 4, &Predicate::Equality(&target))
            .unwrap();
        assert_eq!(bitmask, vec![0u8; 2]);
    }
}
