//! Bit-packing encoding with Frame-of-Reference (FoR) for booleans, small integers, and flags.
//! Subtracts the minimum value (base) from all values before packing,
//! reducing bit width. Packs N-bit residuals into contiguous byte arrays,
//! where N is the minimum bits needed to represent (max - min).

use super::unpack::Packed;
use crate::encoding::{
    Encoding, EncodingType, Predicate, bitmask_from_rows, eval_predicate_on_raw,
};
use zyron_common::{Result, ZyronError};

pub struct BitPackEncoding;

/// Encoded format:
///   [0]      bit_width: u8 (1..=64)
///   [1..5]   original_value_size: u32 (bytes per value before packing)
///   [5..13]  base_value: u64 (FoR base, subtracted before packing)
///   [13..]   packed bit array of (value - base_value) residuals
impl Encoding for BitPackEncoding {
    fn encoding_type(&self) -> EncodingType {
        EncodingType::BitPack
    }

    fn encode(&self, data: &[u8], row_count: usize, value_size: usize) -> Result<Vec<u8>> {
        if row_count == 0 {
            let mut out = vec![0u8; 13];
            out[0] = 1;
            out[1..5].copy_from_slice(&(value_size as u32).to_le_bytes());
            return Ok(out);
        }

        if data.len() < row_count * value_size {
            return Err(ZyronError::EncodingFailed(
                "data shorter than expected for bitpack encoding".to_string(),
            ));
        }

        // Find min and max values for FoR base computation
        let mut minVal = u64::MAX;
        let mut maxVal = 0u64;
        for i in 0..row_count {
            let val = read_value_as_u64(data, i, value_size);
            if val < minVal {
                minVal = val;
            }
            if val > maxVal {
                maxVal = val;
            }
        }

        let baseValue = minVal;
        let maxResidual = maxVal - baseValue;
        let bitWidth = if maxResidual == 0 {
            1
        } else {
            64 - maxResidual.leading_zeros()
        } as u8;

        let packedBits = row_count as u64 * bitWidth as u64;
        let packedBytes = packedBits.div_ceil(8) as usize;

        let mut out = Vec::with_capacity(13 + packedBytes);
        out.push(bitWidth);
        out.extend_from_slice(&(value_size as u32).to_le_bytes());
        out.extend_from_slice(&baseValue.to_le_bytes());

        let mut packed = vec![0u8; packedBytes];
        let mut bitOffset: u64 = 0;

        for i in 0..row_count {
            let val = read_value_as_u64(data, i, value_size);
            let residual = val - baseValue;
            pack_value(&mut packed, bitOffset, residual, bitWidth);
            bitOffset += bitWidth as u64;
        }

        out.extend_from_slice(&packed);
        Ok(out)
    }

    /// A whole column is the range of every row, so the two share one path
    fn decode(&self, encoded: &[u8], row_count: usize, value_size: usize) -> Result<Vec<u8>> {
        if row_count == 0 {
            return Ok(Vec::new());
        }
        self.decode_range(encoded, row_count, value_size, 0, row_count)
    }

    /// Every residual is the same width, so row i sits at bit offset
    /// `i * bit_width` and a range starts there rather than at the segment
    /// start. Reading one row costs one unpack whatever the segment holds
    fn decode_range(
        &self,
        encoded: &[u8],
        row_count: usize,
        value_size: usize,
        start: usize,
        end: usize,
    ) -> Result<Vec<u8>> {
        let (start, end) = crate::encoding::clamp_range(row_count, start, end);
        if start == end {
            return Ok(Vec::new());
        }
        if encoded.len() < 13 {
            return Err(ZyronError::DecodingFailed(
                "bitpack header too short".to_string(),
            ));
        }
        let bitWidth = encoded[0];
        if bitWidth == 0 || bitWidth > 64 {
            return Err(ZyronError::DecodingFailed(format!(
                "invalid bit width: {}",
                bitWidth
            )));
        }
        let storedValueSize =
            u32::from_le_bytes([encoded[1], encoded[2], encoded[3], encoded[4]]) as usize;
        if storedValueSize != value_size {
            return Err(ZyronError::DecodingFailed(format!(
                "bitpack value_size mismatch: stored {}, expected {}",
                storedValueSize, value_size
            )));
        }
        let baseValue = u64::from_le_bytes([
            encoded[5],
            encoded[6],
            encoded[7],
            encoded[8],
            encoded[9],
            encoded[10],
            encoded[11],
            encoded[12],
        ]);
        let taken = end - start;
        // SAFETY: the kernel writes every one of the taken slots at
        // value_size bytes each, which is the whole buffer, before anything
        // reads it
        let mut out = unsafe { super::scratch::take_uninit(taken * value_size) };
        // SAFETY: out holds taken slots of value_size bytes
        unsafe {
            Packed::new(&encoded[13..], bitWidth).add_base_into(
                start,
                taken,
                baseValue,
                out.as_mut_ptr(),
                value_size,
            );
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

        if encoded.len() < 13 {
            let decoded = self.decode(encoded, row_count, value_size)?;
            return eval_predicate_on_raw(&decoded, row_count, value_size, predicate);
        }

        let bitWidth = encoded[0];
        let baseValue = u64::from_le_bytes([
            encoded[5],
            encoded[6],
            encoded[7],
            encoded[8],
            encoded[9],
            encoded[10],
            encoded[11],
            encoded[12],
        ]);
        let packed = &encoded[13..];

        let maxResidual = if bitWidth >= 64 {
            u64::MAX
        } else {
            (1u64 << bitWidth) - 1
        };

        // For single-bit booleans with base=0, use direct bitmask operations
        if bitWidth == 1
            && baseValue == 0
            && let Predicate::Equality(target) = predicate
        {
            let targetVal = read_value_as_u64(target, 0, target.len().min(value_size));
            let bitmaskLen = row_count.div_ceil(8);
            let mut bitmask = vec![0u8; bitmaskLen];

            if targetVal == 1 {
                let copyLen = bitmaskLen.min(packed.len());
                bitmask[..copyLen].copy_from_slice(&packed[..copyLen]);
                let trailing = row_count % 8;
                if trailing != 0 && bitmaskLen > 0 {
                    bitmask[bitmaskLen - 1] &= (1u8 << trailing) - 1;
                }
            } else if targetVal == 0 {
                let copyLen = bitmaskLen.min(packed.len());
                for j in 0..copyLen {
                    bitmask[j] = !packed[j];
                }
                let trailing = row_count % 8;
                if trailing != 0 && bitmaskLen > 0 {
                    bitmask[bitmaskLen - 1] &= (1u8 << trailing) - 1;
                }
            }
            return Ok(bitmask);
        }

        // Evaluate predicates on packed residuals by transforming bounds
        // into the FoR domain (subtract base_value from search targets).
        // Residuals are addressed by row rather than walked with a running
        // offset, so eight rows answer independently into one mask byte
        let bitmaskLen = row_count.div_ceil(8);
        let stream = Packed::new(packed, bitWidth);
        let residual_at = |i: usize| stream.at(i);

        Ok(match predicate {
            Predicate::Equality(target) => {
                let targetVal = read_value_as_u64(target, 0, target.len().min(value_size));
                if targetVal < baseValue || targetVal > baseValue.saturating_add(maxResidual) {
                    return Ok(vec![0u8; bitmaskLen]);
                }
                let targetResidual = targetVal - baseValue;
                bitmask_from_rows(row_count, |i| residual_at(i) == targetResidual)
            }
            Predicate::Range { low, high } => {
                let loVal = match low {
                    Some(lo) => read_value_as_u64(lo, 0, lo.len().min(value_size)),
                    None => 0,
                };
                let hiVal = match high {
                    Some(hi) => read_value_as_u64(hi, 0, hi.len().min(value_size)),
                    None => u64::MAX,
                };

                let maxRepresentable = baseValue.saturating_add(maxResidual);

                // Segment-level skip
                if loVal > maxRepresentable || hiVal < baseValue {
                    return Ok(vec![0u8; bitmaskLen]);
                }

                // Segment-level accept
                if loVal <= baseValue && hiVal >= maxRepresentable {
                    let mut bitmask = vec![0xFFu8; bitmaskLen];
                    let trailing = row_count % 8;
                    if trailing != 0 {
                        bitmask[bitmaskLen - 1] = (1u8 << trailing) - 1;
                    }
                    return Ok(bitmask);
                }

                let loResidual = loVal.saturating_sub(baseValue);
                let hiResidual = if hiVal >= baseValue {
                    (hiVal - baseValue).min(maxResidual)
                } else {
                    return Ok(vec![0u8; bitmaskLen]);
                };

                bitmask_from_rows(row_count, |i| {
                    let residual = residual_at(i);
                    residual >= loResidual && residual <= hiResidual
                })
            }
            Predicate::In(values) => {
                let targetResiduals: Vec<u64> = values
                    .iter()
                    .filter_map(|v| {
                        let val = read_value_as_u64(v, 0, v.len().min(value_size));
                        if val >= baseValue && val <= baseValue.saturating_add(maxResidual) {
                            Some(val - baseValue)
                        } else {
                            None
                        }
                    })
                    .collect();
                if targetResiduals.is_empty() {
                    return Ok(vec![0u8; bitmaskLen]);
                }
                bitmask_from_rows(row_count, |i| targetResiduals.contains(&residual_at(i)))
            }
        })
    }
}

/// Reads a value from data at the given row index as a u64.
fn read_value_as_u64(data: &[u8], row: usize, value_size: usize) -> u64 {
    let offset = row * value_size;
    if offset >= data.len() {
        return 0;
    }
    let end = (offset + value_size).min(data.len());
    let slice = &data[offset..end];
    let mut buf = [0u8; 8];
    let copyLen = slice.len().min(8);
    buf[..copyLen].copy_from_slice(&slice[..copyLen]);
    u64::from_le_bytes(buf)
}

/// Packs a value at the given bit offset into the packed array.
#[inline]
fn pack_value(packed: &mut [u8], bit_offset: u64, value: u64, bit_width: u8) {
    let byteIdx = (bit_offset / 8) as usize;
    let bitIdx = (bit_offset % 8) as u32;
    let mask = if bit_width >= 64 {
        u64::MAX
    } else {
        (1u64 << bit_width) - 1
    };
    let val = value & mask;

    let totalBits = bitIdx + bit_width as u32;
    let bytesNeeded = totalBits.div_ceil(8) as usize;

    // shift in u128 so a bitIdx + bit_width that exceeds 64 keeps the spilled
    // high bits instead of dropping them off the top of a u64
    let shifted = (val as u128) << bitIdx;
    let shiftedBytes = shifted.to_le_bytes();
    for j in 0..bytesNeeded.min(16) {
        if byteIdx + j < packed.len() {
            packed[byteIdx + j] |= shiftedBytes[j];
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_roundtrip_boolean() {
        let enc = BitPackEncoding;
        let mut data = Vec::new();
        for i in 0..100u8 {
            data.push(i % 2); // alternating 0,1
        }

        let encoded = enc.encode(&data, 100, 1).unwrap();
        assert!(encoded.len() < data.len());

        let decoded = enc.decode(&encoded, 100, 1).unwrap();
        assert_eq!(decoded, data);
    }

    #[test]
    fn test_roundtrip_small_int() {
        let enc = BitPackEncoding;
        let mut data = Vec::new();
        for i in 0..64u32 {
            data.extend_from_slice(&(i % 16).to_le_bytes());
        }

        let encoded = enc.encode(&data, 64, 4).unwrap();
        let decoded = enc.decode(&encoded, 64, 4).unwrap();
        assert_eq!(decoded, data);
    }

    #[test]
    fn test_roundtrip_u64() {
        let enc = BitPackEncoding;
        let mut data = Vec::new();
        for i in 0..32u64 {
            data.extend_from_slice(&(i * 1000).to_le_bytes());
        }

        let encoded = enc.encode(&data, 32, 8).unwrap();
        let decoded = enc.decode(&encoded, 32, 8).unwrap();
        assert_eq!(decoded, data);
    }

    #[test]
    fn test_pack_unpack_width64_rows() {
        // width 64 rows round-trip the full value at every row
        let values = [
            u64::MAX,
            u64::MAX - 1,
            0x8000_0000_0000_0001,
            0xFEDC_BA98_7654_3210,
            1,
        ];
        let mut packed = vec![0u8; values.len() * 8];
        for (i, &val) in values.iter().enumerate() {
            pack_value(&mut packed, i as u64 * 64, val, 64);
        }
        let stream = Packed::new(&packed, 64);
        for (i, &val) in values.iter().enumerate() {
            assert_eq!(stream.at(i), val, "val {val:#x} at row {i}");
        }
    }

    #[test]
    fn test_pack_unpack_wide_width_spill() {
        // widths where a row's bit offset plus the width crosses a word
        // boundary round-trip, spilled high bits included
        for bitWidth in [58u8, 60, 62, 63, 64] {
            let mask = if bitWidth >= 64 {
                u64::MAX
            } else {
                (1u64 << bitWidth) - 1
            };
            let rows = 16usize;
            let value = |i: usize| 0x9E37_79B9_7F4A_7C15u64.wrapping_mul(i as u64 + 1) & mask;
            let mut packed = vec![0u8; (rows * bitWidth as usize).div_ceil(8)];
            for i in 0..rows {
                pack_value(&mut packed, i as u64 * bitWidth as u64, value(i), bitWidth);
            }
            let stream = Packed::new(&packed, bitWidth);
            for i in 0..rows {
                assert_eq!(stream.at(i), value(i), "width {bitWidth} row {i}");
            }
        }
    }

    #[test]
    fn test_roundtrip_offset_range() {
        // FoR test: values [1000..1100] should only need 7 bits instead of 11
        let enc = BitPackEncoding;
        let mut data = Vec::new();
        for i in 1000..1100u32 {
            data.extend_from_slice(&i.to_le_bytes());
        }

        let encoded = enc.encode(&data, 100, 4).unwrap();
        // bit_width should be 7 (max residual = 99, needs 7 bits)
        assert_eq!(encoded[0], 7);
        // base_value should be 1000
        let base = u64::from_le_bytes([
            encoded[5],
            encoded[6],
            encoded[7],
            encoded[8],
            encoded[9],
            encoded[10],
            encoded[11],
            encoded[12],
        ]);
        assert_eq!(base, 1000);

        let decoded = enc.decode(&encoded, 100, 4).unwrap();
        assert_eq!(decoded, data);
    }

    #[test]
    fn test_empty() {
        let enc = BitPackEncoding;
        let encoded = enc.encode(&[], 0, 4).unwrap();
        let decoded = enc.decode(&encoded, 0, 4).unwrap();
        assert!(decoded.is_empty());
    }

    #[test]
    fn test_predicate_boolean_true() {
        let enc = BitPackEncoding;
        let data: Vec<u8> = (0..8).map(|i| (i + 1) % 2).collect();
        let encoded = enc.encode(&data, 8, 1).unwrap();

        let target = [1u8];
        let bitmask = enc
            .eval_predicate(&encoded, 8, 1, &Predicate::Equality(&target))
            .unwrap();
        assert_eq!(bitmask[0], 0x55);
    }

    #[test]
    fn test_predicate_boolean_false() {
        let enc = BitPackEncoding;
        let data: Vec<u8> = (0..8).map(|i| (i + 1) % 2).collect();
        let encoded = enc.encode(&data, 8, 1).unwrap();

        let target = [0u8];
        let bitmask = enc
            .eval_predicate(&encoded, 8, 1, &Predicate::Equality(&target))
            .unwrap();
        assert_eq!(bitmask[0], 0xAA);
    }

    #[test]
    fn test_predicate_equality_with_base() {
        let enc = BitPackEncoding;
        let mut data = Vec::new();
        for v in [1000u32, 1005, 1010, 1005, 1020] {
            data.extend_from_slice(&v.to_le_bytes());
        }

        let encoded = enc.encode(&data, 5, 4).unwrap();
        let target = 1005u32.to_le_bytes();
        let bitmask = enc
            .eval_predicate(&encoded, 5, 4, &Predicate::Equality(&target))
            .unwrap();
        // Rows 1 and 3 match
        assert_eq!(bitmask[0], 0b00001010);
    }

    #[test]
    fn test_predicate_range_with_base() {
        let enc = BitPackEncoding;
        let mut data = Vec::new();
        for v in [1000u32, 1005, 1010, 1015, 1020] {
            data.extend_from_slice(&v.to_le_bytes());
        }

        let encoded = enc.encode(&data, 5, 4).unwrap();
        let lo = 1005u32.to_le_bytes();
        let hi = 1015u32.to_le_bytes();
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
        // Rows 1,2,3 match (1005,1010,1015)
        assert_eq!(bitmask[0], 0b00001110);
    }

    #[test]
    fn test_compaction_ratio() {
        let enc = BitPackEncoding;
        // 1000 values in range [0, 7] = 3 bits each
        let mut data = Vec::new();
        for i in 0..1000u32 {
            data.extend_from_slice(&(i % 8).to_le_bytes());
        }

        let encoded = enc.encode(&data, 1000, 4).unwrap();
        // 3 bits * 1000 = 375 bytes + 13 header = 388 bytes vs 4000 raw
        assert!(encoded.len() < data.len() / 2);
    }

    #[test]
    fn test_compaction_ratio_offset() {
        let enc = BitPackEncoding;
        // 1000 values in range [10000, 10007] = 3 bits each with FoR
        // Without FoR this would need 14 bits per value
        let mut data = Vec::new();
        for i in 0..1000u32 {
            data.extend_from_slice(&(10000 + i % 8).to_le_bytes());
        }

        let encoded = enc.encode(&data, 1000, 4).unwrap();
        let ratio = data.len() as f64 / encoded.len() as f64;
        // 3 bits * 1000 = 375 bytes + 13 header = 388 bytes vs 4000 raw = 10.3:1
        assert!(
            ratio > 10.0,
            "expected 10:1+ ratio with FoR, got {:.1}:1",
            ratio
        );

        let decoded = enc.decode(&encoded, 1000, 4).unwrap();
        assert_eq!(decoded, data);
    }
}
