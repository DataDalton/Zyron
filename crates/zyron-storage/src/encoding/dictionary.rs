//! Dictionary encoding for low-cardinality columns.
//! Builds a sorted dictionary of distinct values, replaces each row value
//! with a bit-packed code index. Code width = ceil(log2(dict_count)) bits,
//! so 10 distinct values use 4-bit codes instead of 32-bit codes.
//! Predicate evaluation resolves the search term to a code via binary search
//! on the dictionary, then scans the code array without decoding.

use crate::encoding::{Encoding, EncodingType, Predicate};
use zyron_common::{Result, ZyronError};

pub struct DictionaryEncoding;

/// Encoded format:
///   [0..4]     value_size: u32
///   [4..8]     dict_count: u32
///   [8..8+dict_count*value_size]  dictionary entries (sorted)
///   [8+dict_count*value_size..]   bit-packed code array (ceil(log2(dict_count)) bits per row)
impl Encoding for DictionaryEncoding {
    fn encoding_type(&self) -> EncodingType {
        EncodingType::Dictionary
    }

    fn encode(&self, data: &[u8], row_count: usize, value_size: usize) -> Result<Vec<u8>> {
        if row_count == 0 {
            let mut out = vec![0u8; 8];
            out[0..4].copy_from_slice(&(value_size as u32).to_le_bytes());
            return Ok(out);
        }

        if data.len() < row_count * value_size {
            return Err(ZyronError::EncodingFailed(
                "data shorter than expected for dictionary encoding".to_string(),
            ));
        }

        // Collect distinct values in sorted order via binary search insertion
        let mut distinct: Vec<&[u8]> = Vec::new();
        for i in 0..row_count {
            let val = &data[i * value_size..(i + 1) * value_size];
            if let Err(pos) = distinct.binary_search_by(|probe| (*probe).cmp(val)) {
                distinct.insert(pos, val);
            }
        }

        if distinct.len() > u32::MAX as usize {
            return Err(ZyronError::EncodingFailed(
                "dictionary cardinality exceeds u32 range".to_string(),
            ));
        }

        let dictCount = distinct.len() as u32;
        let dictSize = dictCount as usize * value_size;

        // Bit width for codes: ceil(log2(dict_count)), minimum 1 bit
        let codeBitWidth = if dictCount <= 1 {
            1u8
        } else {
            (32 - (dictCount - 1).leading_zeros()) as u8
        };

        let totalCodeBits = row_count as u64 * codeBitWidth as u64;
        let packedCodeBytes = ((totalCodeBits + 7) / 8) as usize;

        let mut out = Vec::with_capacity(8 + dictSize + packedCodeBytes);
        out.extend_from_slice(&(value_size as u32).to_le_bytes());
        out.extend_from_slice(&dictCount.to_le_bytes());

        // Write dictionary entries
        for entry in &distinct {
            out.extend_from_slice(entry);
        }

        // Bit-pack the code array
        let mut packed = vec![0u8; packedCodeBytes];
        for i in 0..row_count {
            let val = &data[i * value_size..(i + 1) * value_size];
            let code = distinct
                .binary_search_by(|probe| (*probe).cmp(val))
                .map_err(|_| {
                    ZyronError::EncodingFailed(
                        "value not found in dictionary during encoding".to_string(),
                    )
                })? as u32;
            pack_bits(
                &mut packed,
                i as u64 * codeBitWidth as u64,
                code as u64,
                codeBitWidth,
            );
        }

        out.extend_from_slice(&packed);
        Ok(out)
    }

    fn decode(&self, encoded: &[u8], row_count: usize, value_size: usize) -> Result<Vec<u8>> {
        if row_count == 0 {
            return Ok(Vec::new());
        }

        if encoded.len() < 8 {
            return Err(ZyronError::DecodingFailed(
                "dictionary header too short".to_string(),
            ));
        }

        let storedValueSize =
            u32::from_le_bytes([encoded[0], encoded[1], encoded[2], encoded[3]]) as usize;
        if storedValueSize != value_size {
            return Err(ZyronError::DecodingFailed(format!(
                "dictionary value_size mismatch: stored {}, expected {}",
                storedValueSize, value_size
            )));
        }

        let dictCount =
            u32::from_le_bytes([encoded[4], encoded[5], encoded[6], encoded[7]]) as usize;

        let dictStart = 8;
        let dictEnd = dictStart + dictCount * value_size;

        // Bit width for codes
        let codeBitWidth = if dictCount <= 1 {
            1u8
        } else {
            (32 - (dictCount as u32 - 1).leading_zeros()) as u8
        };

        let packedStart = dictEnd;

        if encoded.len() < packedStart {
            return Err(ZyronError::DecodingFailed(
                "dictionary data truncated".to_string(),
            ));
        }

        let packed = &encoded[packedStart..];
        let mut out = Vec::with_capacity(row_count * value_size);

        for i in 0..row_count {
            let code = unpack_bits(packed, i as u64 * codeBitWidth as u64, codeBitWidth) as usize;

            if code >= dictCount {
                return Err(ZyronError::DecodingFailed(format!(
                    "dictionary code {} out of range (dict_count={})",
                    code, dictCount
                )));
            }

            let valOffset = dictStart + code * value_size;
            out.extend_from_slice(&encoded[valOffset..valOffset + value_size]);
        }

        Ok(out)
    }

    fn eval_predicate(
        &self,
        encoded: &[u8],
        row_count: usize,
        _value_size: usize,
        predicate: &Predicate,
    ) -> Result<Vec<u8>> {
        if row_count == 0 {
            return Ok(Vec::new());
        }

        if encoded.len() < 8 {
            return Err(ZyronError::DecodingFailed(
                "dictionary header too short for predicate evaluation".to_string(),
            ));
        }

        let storedValueSize =
            u32::from_le_bytes([encoded[0], encoded[1], encoded[2], encoded[3]]) as usize;
        let dictCount =
            u32::from_le_bytes([encoded[4], encoded[5], encoded[6], encoded[7]]) as usize;

        let dictStart = 8;
        let dictEnd = dictStart + dictCount * storedValueSize;

        let codeBitWidth = if dictCount <= 1 {
            1u8
        } else {
            (32 - (dictCount as u32 - 1).leading_zeros()) as u8
        };

        let packedStart = dictEnd;

        let bitmaskLen = (row_count + 7) / 8;
        let mut bitmask = vec![0u8; bitmaskLen];

        // Build a set of matching dictionary codes
        let mut matchingCodes = Vec::new();

        match predicate {
            Predicate::Equality(target) => {
                if let Some(code) =
                    dict_binary_search(encoded, dictStart, dictCount, storedValueSize, target)
                {
                    matchingCodes.push(code as u32);
                }
            }
            Predicate::Range { low, high } => {
                for c in 0..dictCount {
                    let offset = dictStart + c * storedValueSize;
                    let entry = &encoded[offset..offset + storedValueSize];
                    let above = match low {
                        Some(lo) => entry >= *lo,
                        None => true,
                    };
                    let below = match high {
                        Some(hi) => entry <= *hi,
                        None => true,
                    };
                    if above && below {
                        matchingCodes.push(c as u32);
                    }
                }
            }
            Predicate::In(values) => {
                for target in *values {
                    if let Some(code) =
                        dict_binary_search(encoded, dictStart, dictCount, storedValueSize, target)
                    {
                        matchingCodes.push(code as u32);
                    }
                }
            }
        }

        if matchingCodes.is_empty() {
            return Ok(bitmask);
        }

        // Scan code array, checking membership
        matchingCodes.sort_unstable();
        let packed = &encoded[packedStart..];
        for i in 0..row_count {
            let code = unpack_bits(packed, i as u64 * codeBitWidth as u64, codeBitWidth) as u32;
            if matchingCodes.binary_search(&code).is_ok() {
                bitmask[i / 8] |= 1 << (i % 8);
            }
        }

        Ok(bitmask)
    }
}

/// Binary search for a value in the sorted dictionary. Returns the code index if found.
fn dict_binary_search(
    encoded: &[u8],
    dict_start: usize,
    dict_count: usize,
    value_size: usize,
    target: &[u8],
) -> Option<usize> {
    let mut lo = 0usize;
    let mut hi = dict_count;

    while lo < hi {
        let mid = lo + (hi - lo) / 2;
        let offset = dict_start + mid * value_size;
        let entry = &encoded[offset..offset + value_size];

        match entry.cmp(target) {
            std::cmp::Ordering::Less => lo = mid + 1,
            std::cmp::Ordering::Equal => return Some(mid),
            std::cmp::Ordering::Greater => hi = mid,
        }
    }

    None
}

/// Packs a u64 value at the given bit offset.
#[inline]
fn pack_bits(packed: &mut [u8], bit_offset: u64, value: u64, bit_width: u8) {
    let byteIdx = (bit_offset / 8) as usize;
    let bitIdx = (bit_offset % 8) as u32;
    let mask = if bit_width >= 64 {
        u64::MAX
    } else {
        (1u64 << bit_width) - 1
    };
    let val = value & mask;
    let shifted = val << bitIdx;
    let shiftedBytes = shifted.to_le_bytes();
    let totalBits = bitIdx + bit_width as u32;
    let bytesNeeded = ((totalBits + 7) / 8) as usize;

    for j in 0..bytesNeeded.min(8) {
        if byteIdx + j < packed.len() {
            packed[byteIdx + j] |= shiftedBytes[j];
        }
    }
}

/// Unpacks a u64 value from the given bit offset.
#[inline]
fn unpack_bits(packed: &[u8], bit_offset: u64, bit_width: u8) -> u64 {
    let byteIdx = (bit_offset / 8) as usize;
    let bitIdx = (bit_offset % 8) as u32;
    let mut buf = [0u8; 9];
    let available = packed.len().saturating_sub(byteIdx).min(9);
    buf[..available].copy_from_slice(&packed[byteIdx..byteIdx + available]);

    let lo = u64::from_le_bytes([
        buf[0], buf[1], buf[2], buf[3], buf[4], buf[5], buf[6], buf[7],
    ]);
    let val = lo >> bitIdx;
    let mask = if bit_width >= 64 {
        u64::MAX
    } else {
        (1u64 << bit_width) - 1
    };

    if bitIdx + bit_width as u32 > 64 {
        let hi = (buf[8] as u64) << (64 - bitIdx);
        (val | hi) & mask
    } else {
        val & mask
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_roundtrip_i32() {
        let enc = DictionaryEncoding;
        let values = [10u32, 20, 30];
        let mut data = Vec::new();
        for i in 0..100 {
            data.extend_from_slice(&values[i % 3].to_le_bytes());
        }

        let encoded = enc.encode(&data, 100, 4).unwrap();
        let decoded = enc.decode(&encoded, 100, 4).unwrap();
        assert_eq!(decoded, data);
    }

    #[test]
    fn test_roundtrip_single_byte() {
        let enc = DictionaryEncoding;
        let mut data = Vec::new();
        for i in 0..50u8 {
            data.push(i % 5);
        }

        let encoded = enc.encode(&data, 50, 1).unwrap();
        let decoded = enc.decode(&encoded, 50, 1).unwrap();
        assert_eq!(decoded, data);
    }

    #[test]
    fn test_empty() {
        let enc = DictionaryEncoding;
        let encoded = enc.encode(&[], 0, 4).unwrap();
        let decoded = enc.decode(&encoded, 0, 4).unwrap();
        assert!(decoded.is_empty());
    }

    #[test]
    fn test_compression_ratio_10_distinct() {
        let enc = DictionaryEncoding;
        let n = 100_000usize;
        let mut data = Vec::with_capacity(n * 4);
        for i in 0..n {
            data.extend_from_slice(&((i % 10) as u32 * 1000).to_le_bytes());
        }

        let encoded = enc.encode(&data, n, 4).unwrap();
        // 10 distinct -> 4-bit codes, packed = 100000*4/8 = 50000 bytes + dict(40) + header(8)
        let ratio = data.len() as f64 / encoded.len() as f64;
        assert!(ratio > 7.0, "expected 7:1+ ratio, got {:.1}:1", ratio);

        let decoded = enc.decode(&encoded, n, 4).unwrap();
        assert_eq!(decoded, data);
    }

    #[test]
    fn test_predicate_equality_found() {
        let enc = DictionaryEncoding;
        let mut data = Vec::new();
        // [10, 20, 10, 30, 20]
        for v in [10u32, 20, 10, 30, 20] {
            data.extend_from_slice(&v.to_le_bytes());
        }

        let encoded = enc.encode(&data, 5, 4).unwrap();
        let target = 10u32.to_le_bytes();
        let bitmask = enc
            .eval_predicate(&encoded, 5, 4, &Predicate::Equality(&target))
            .unwrap();
        // Rows 0 and 2 match: bits 0,2 = 0b00000101
        assert_eq!(bitmask[0], 0b00000101);
    }

    #[test]
    fn test_predicate_equality_not_found() {
        let enc = DictionaryEncoding;
        let mut data = Vec::new();
        for v in [10u32, 20, 30] {
            data.extend_from_slice(&v.to_le_bytes());
        }

        let encoded = enc.encode(&data, 3, 4).unwrap();
        let target = 99u32.to_le_bytes();
        let bitmask = enc
            .eval_predicate(&encoded, 3, 4, &Predicate::Equality(&target))
            .unwrap();
        assert_eq!(bitmask[0], 0);
    }

    #[test]
    fn test_predicate_range() {
        let enc = DictionaryEncoding;
        let mut data = Vec::new();
        for v in [10u32, 20, 30, 40, 50] {
            data.extend_from_slice(&v.to_le_bytes());
        }

        let encoded = enc.encode(&data, 5, 4).unwrap();
        let lo = 20u32.to_le_bytes();
        let hi = 40u32.to_le_bytes();
        let bitmask = enc
            .eval_predicate(
                &encoded,
                5,
                4,
                &Predicate::Range {
                    low: Some(&lo),
                    high: Some(&hi),
                },
            )
            .unwrap();
        // Rows 1,2,3 match (values 20,30,40)
        assert_eq!(bitmask[0], 0b00001110);
    }

    #[test]
    fn test_predicate_in() {
        let enc = DictionaryEncoding;
        let mut data = Vec::new();
        for v in [10u32, 20, 30, 40, 50] {
            data.extend_from_slice(&v.to_le_bytes());
        }

        let encoded = enc.encode(&data, 5, 4).unwrap();
        let v1 = 10u32.to_le_bytes();
        let v2 = 50u32.to_le_bytes();
        let targets: Vec<&[u8]> = vec![&v1, &v2];
        let bitmask = enc
            .eval_predicate(&encoded, 5, 4, &Predicate::In(&targets))
            .unwrap();
        // Rows 0 and 4 match
        assert_eq!(bitmask[0], 0b00010001);
    }
}
