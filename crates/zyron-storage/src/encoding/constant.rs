//! Constant encoding for column segments where every row has the same value.
//! Stores the single value once. Zero per-row storage cost.
//! Predicate evaluation compares against the single stored value,
//! producing an all-ones or all-zeros bitmask in O(1).

use crate::encoding::{Encoding, EncodingType, Predicate, range_admits, slice_rows, varlen_pack};
use zyron_common::{Result, ZyronError};

pub struct ConstantEncoding;

/// Encoded format:
///   [0..4]  stored_len: u32 (little-endian, the constant value's byte length;
///           equals value_size for the fixed-width path, the value length for
///           the variable-length path where value_size == 0)
///   [4..]   value: [u8; stored_len]
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

        let rows = slice_rows(data, row_count, value_size)?;
        let first_value = rows[0];

        // Verify all rows have the same value
        for r in &rows[1..] {
            if *r != first_value {
                return Err(ZyronError::EncodingFailed(
                    "not all values are identical for constant encoding".to_string(),
                ));
            }
        }

        let mut out = Vec::with_capacity(4 + first_value.len());
        out.extend_from_slice(&(first_value.len() as u32).to_le_bytes());
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
        if value_size != 0 && stored_size != value_size {
            return Err(ZyronError::DecodingFailed(format!(
                "constant value_size mismatch: stored {}, expected {}",
                stored_size, value_size
            )));
        }

        if encoded.len() < 4 + stored_size {
            return Err(ZyronError::DecodingFailed(
                "constant encoded data truncated".to_string(),
            ));
        }

        let value = &encoded[4..4 + stored_size];
        if value_size == 0 {
            // Variable-length: reconstruct the canonical buffer.
            let rows: Vec<Option<&[u8]>> = vec![Some(value); row_count];
            return Ok(varlen_pack(&rows));
        }
        let mut out = Vec::with_capacity(row_count * value_size);
        for _ in 0..row_count {
            out.extend_from_slice(value);
        }
        Ok(out)
    }

    /// Every row is the same value, so a range is the value repeated the
    /// number of times the range asks for and the segment size never enters
    fn decode_range(
        &self,
        encoded: &[u8],
        row_count: usize,
        value_size: usize,
        start: usize,
        end: usize,
    ) -> Result<Vec<u8>> {
        let (start, end) = crate::encoding::clamp_range(row_count, start, end);
        self.decode(encoded, end - start, value_size)
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

        if encoded.len() < 4 {
            return Err(ZyronError::DecodingFailed(
                "constant encoded data too short for predicate evaluation".to_string(),
            ));
        }
        let stored_size =
            u32::from_le_bytes([encoded[0], encoded[1], encoded[2], encoded[3]]) as usize;
        if encoded.len() < 4 + stored_size {
            return Err(ZyronError::DecodingFailed(
                "constant encoded data too short for predicate evaluation".to_string(),
            ));
        }

        let value = &encoded[4..4 + stored_size];
        let bitmask_len = row_count.div_ceil(8);

        // The range comparison runs in the column's stored order, which the
        // column's value size selects: little endian numeric for a
        // fixed-width cell, lexicographic for a variable-length one. The
        // stored length is how many bytes this one value occupies, and on a
        // variable-length column it is the length of the string rather than
        // zero, so ordering by it would read the string backwards as a
        // number and admit or reject every row on that reading
        let matches = match predicate {
            Predicate::Equality(target) => value == *target,
            Predicate::Range { low, high } => range_admits(value, value_size, *low, *high),
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

    /// A variable-length column orders by its bytes from the first. Ordering
    /// the stored value by its own byte length instead reads the string from
    /// the last byte, which decides the whole file on the reversed reading:
    /// every row of a file holding one distinct string is returned, or none
    /// of them, and which one it is depends on the string's tail.
    #[test]
    fn a_variable_length_constant_orders_its_range_by_its_bytes() {
        let enc = ConstantEncoding;
        let rows = 2_048usize;
        let views: Vec<Option<&[u8]>> = vec![Some(b"same".as_slice()); rows];
        let raw = crate::encoding::varlen_pack(&views);
        let encoded = enc.encode(&raw, rows, 0).expect("encode");

        // "same" is above "row-99999999" from the first byte, so a range
        // ending there admits nothing. Read from the last byte the four
        // characters are a little endian number and the answer flips
        let low = b"a".to_vec();
        let high = b"row-99999999".to_vec();
        let mask = enc
            .eval_predicate(
                &encoded,
                rows,
                0,
                &Predicate::Range {
                    low: Some(&low),
                    high: Some(&high),
                },
            )
            .expect("eval");
        assert!(
            mask.iter().all(|byte| *byte == 0),
            "a range the value sits above must admit no row"
        );

        // And a range the value does sit inside admits every row
        let high = b"tail".to_vec();
        let mask = enc
            .eval_predicate(
                &encoded,
                rows,
                0,
                &Predicate::Range {
                    low: Some(&low),
                    high: Some(&high),
                },
            )
            .expect("eval");
        assert!(
            mask.iter().all(|byte| *byte == 0xFF),
            "a range the value sits inside must admit every row"
        );

        // The direction that loses rows: "ba" is between "b" and "c" read
        // from the first byte and above both read from the last, so the
        // reversed reading answers that no row matches and the rows the
        // query wanted are dropped rather than merely rechecked
        let views: Vec<Option<&[u8]>> = vec![Some(b"ba".as_slice()); rows];
        let raw = crate::encoding::varlen_pack(&views);
        let encoded = enc.encode(&raw, rows, 0).expect("encode");
        let low = b"b".to_vec();
        let high = b"c".to_vec();
        let mask = enc
            .eval_predicate(
                &encoded,
                rows,
                0,
                &Predicate::Range {
                    low: Some(&low),
                    high: Some(&high),
                },
            )
            .expect("eval");
        assert!(
            mask.iter().all(|byte| *byte == 0xFF),
            "every row holds a value inside the range and none may be dropped"
        );
    }
}
