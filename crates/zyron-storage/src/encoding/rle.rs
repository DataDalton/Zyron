//! Run-Length Encoding for sorted or repetitive column data.
//! Stores (value, run_length) pairs with varint-encoded run lengths.
//! For sorted data with many repeats, this encodes significantly
//! and supports predicate evaluation directly on the run values
//! without expansion.

use crate::encoding::{Encoding, EncodingType, Predicate};
use zyron_common::{Result, ZyronError};

pub struct RleEncoding;

/// Encoded format:
///   [0..4]    value_size: u32
///   [4..8]    run_count: u32
///   [8..]     runs: [value(value_size bytes) + run_length(varint)] * run_count
///
/// Varint encoding uses LEB128: each byte stores 7 data bits + 1 continuation bit.
/// Values 0-127 use 1 byte, 128-16383 use 2 bytes, etc.
impl Encoding for RleEncoding {
    fn encoding_type(&self) -> EncodingType {
        EncodingType::Rle
    }

    fn encode(&self, data: &[u8], row_count: usize, value_size: usize) -> Result<Vec<u8>> {
        if row_count == 0 {
            let mut out = vec![0u8; 8];
            out[0..4].copy_from_slice(&(value_size as u32).to_le_bytes());
            // run_count = 0
            return Ok(out);
        }

        if data.len() < row_count * value_size {
            return Err(ZyronError::EncodingFailed(
                "data shorter than expected for RLE encoding".to_string(),
            ));
        }

        let mut runs: Vec<u8> = Vec::new();
        let mut runCount = 0u32;

        let mut currentStart = 0;
        let mut currentLen = 1u32;

        for i in 1..row_count {
            let prev = &data[currentStart * value_size..(currentStart + 1) * value_size];
            let curr = &data[i * value_size..(i + 1) * value_size];

            if curr == prev {
                currentLen += 1;
            } else {
                // Emit run: value + varint length
                runs.extend_from_slice(prev);
                encode_varint(currentLen, &mut runs);
                runCount += 1;
                currentStart = i;
                currentLen = 1;
            }
        }

        // Emit final run
        let lastValue = &data[currentStart * value_size..(currentStart + 1) * value_size];
        runs.extend_from_slice(lastValue);
        encode_varint(currentLen, &mut runs);
        runCount += 1;

        let mut out = Vec::with_capacity(8 + runs.len());
        out.extend_from_slice(&(value_size as u32).to_le_bytes());
        out.extend_from_slice(&runCount.to_le_bytes());
        out.extend_from_slice(&runs);

        Ok(out)
    }

    fn decode(&self, encoded: &[u8], row_count: usize, value_size: usize) -> Result<Vec<u8>> {
        if row_count == 0 {
            return Ok(Vec::new());
        }

        if encoded.len() < 8 {
            return Err(ZyronError::DecodingFailed(
                "RLE header too short".to_string(),
            ));
        }

        let storedValueSize =
            u32::from_le_bytes([encoded[0], encoded[1], encoded[2], encoded[3]]) as usize;
        if storedValueSize != value_size {
            return Err(ZyronError::DecodingFailed(format!(
                "RLE value_size mismatch: stored {}, expected {}",
                storedValueSize, value_size
            )));
        }

        let runCount =
            u32::from_le_bytes([encoded[4], encoded[5], encoded[6], encoded[7]]) as usize;

        let totalBytes = row_count * value_size;
        let mut out: Vec<u8> = Vec::with_capacity(totalBytes);
        unsafe {
            out.set_len(totalBytes);
        }
        let outPtr = out.as_mut_ptr();
        let mut writePos = 0usize;
        let mut pos = 8;

        for _ in 0..runCount {
            if pos + value_size > encoded.len() {
                return Err(ZyronError::DecodingFailed("RLE data truncated".to_string()));
            }
            let value = &encoded[pos..pos + value_size];
            pos += value_size;

            let (runLen, bytesRead) = decode_varint(&encoded[pos..])?;
            pos += bytesRead;

            let runBytes = runLen as usize * value_size;
            if writePos + runBytes > totalBytes {
                return Err(ZyronError::DecodingFailed(format!(
                    "RLE row count mismatch: runs exceed expected {} rows",
                    row_count
                )));
            }

            // Write the first value, then use doubling memcpy to fill the rest.
            // LLVM turns large copy_nonoverlapping into SIMD stores or rep movsb,
            // achieving 50-100 GB/s fill bandwidth on modern CPUs.
            unsafe {
                std::ptr::copy_nonoverlapping(value.as_ptr(), outPtr.add(writePos), value_size);
            }
            let mut filled = value_size;
            while filled < runBytes {
                let chunk = filled.min(runBytes - filled);
                unsafe {
                    std::ptr::copy_nonoverlapping(
                        outPtr.add(writePos),
                        outPtr.add(writePos + filled),
                        chunk,
                    );
                }
                filled += chunk;
            }

            writePos += runBytes;
        }

        if writePos != totalBytes {
            return Err(ZyronError::DecodingFailed(format!(
                "RLE row count mismatch: runs sum to {}, expected {}",
                writePos / value_size,
                row_count
            )));
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
                "RLE header too short for predicate evaluation".to_string(),
            ));
        }

        let storedValueSize =
            u32::from_le_bytes([encoded[0], encoded[1], encoded[2], encoded[3]]) as usize;
        let runCount =
            u32::from_le_bytes([encoded[4], encoded[5], encoded[6], encoded[7]]) as usize;

        let bitmaskLen = (row_count + 7) / 8;
        let mut bitmask = vec![0u8; bitmaskLen];
        let mut rowIdx = 0usize;
        let mut pos = 8;

        for _ in 0..runCount {
            if pos + storedValueSize > encoded.len() {
                return Err(ZyronError::DecodingFailed(
                    "RLE data truncated during predicate evaluation".to_string(),
                ));
            }

            let value = &encoded[pos..pos + storedValueSize];
            pos += storedValueSize;

            let (runLen, bytesRead) = decode_varint(&encoded[pos..])?;
            pos += bytesRead;

            let matches = match predicate {
                Predicate::Equality(target) => value == *target,
                Predicate::Range { low, high } => {
                    let above = match low {
                        Some(lo) => value >= *lo,
                        None => true,
                    };
                    let below = match high {
                        Some(hi) => value <= *hi,
                        None => true,
                    };
                    above && below
                }
                Predicate::In(values) => values.iter().any(|v| value == *v),
            };

            if matches {
                // Bulk-set bits for the entire matching run.
                // First set full bytes (8 bits at a time), then handle partial bytes
                // at the start and end of the run.
                let runEnd = (rowIdx + runLen as usize).min(row_count);
                let startBit = rowIdx;
                let endBit = runEnd;

                // Handle partial first byte
                let firstByte = startBit / 8;
                let lastByte = (endBit.saturating_sub(1)) / 8;
                let firstBitInByte = startBit % 8;
                let lastBitInByte = if endBit > 0 { (endBit - 1) % 8 } else { 0 };

                if firstByte == lastByte {
                    // All bits in the same byte
                    for b in firstBitInByte..=lastBitInByte {
                        bitmask[firstByte] |= 1 << b;
                    }
                } else {
                    // Set partial first byte
                    for b in firstBitInByte..8 {
                        bitmask[firstByte] |= 1 << b;
                    }
                    // Set full middle bytes
                    for byte in (firstByte + 1)..lastByte {
                        bitmask[byte] = 0xFF;
                    }
                    // Set partial last byte
                    for b in 0..=lastBitInByte {
                        bitmask[lastByte] |= 1 << b;
                    }
                }
            }

            rowIdx += runLen as usize;
        }

        Ok(bitmask)
    }
}

/// Encodes a u32 value as a LEB128 varint.
#[inline]
fn encode_varint(mut value: u32, output: &mut Vec<u8>) {
    loop {
        let byte = (value & 0x7F) as u8;
        value >>= 7;
        if value == 0 {
            output.push(byte);
            break;
        } else {
            output.push(byte | 0x80);
        }
    }
}

/// Decodes a LEB128 varint from the given byte slice.
/// Returns (value, bytes_consumed).
#[inline]
fn decode_varint(data: &[u8]) -> Result<(u32, usize)> {
    let mut result = 0u32;
    let mut shift = 0u32;
    for (i, &byte) in data.iter().enumerate() {
        result |= ((byte & 0x7F) as u32) << shift;
        if byte & 0x80 == 0 {
            return Ok((result, i + 1));
        }
        shift += 7;
        if shift >= 35 {
            return Err(ZyronError::DecodingFailed(
                "RLE varint too long".to_string(),
            ));
        }
    }
    Err(ZyronError::DecodingFailed(
        "RLE varint truncated".to_string(),
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_roundtrip_i32() {
        let enc = RleEncoding;
        let mut data = Vec::new();
        // 3 runs: [1,1,1,1,1, 2,2,2, 3,3]
        for _ in 0..5 {
            data.extend_from_slice(&1u32.to_le_bytes());
        }
        for _ in 0..3 {
            data.extend_from_slice(&2u32.to_le_bytes());
        }
        for _ in 0..2 {
            data.extend_from_slice(&3u32.to_le_bytes());
        }

        let encoded = enc.encode(&data, 10, 4).unwrap();
        // 3 runs * (4 value + 1 varint) = 15 + 8 header = 23 bytes vs 40 raw
        assert_eq!(encoded.len(), 23);

        let decoded = enc.decode(&encoded, 10, 4).unwrap();
        assert_eq!(decoded, data);
    }

    #[test]
    fn test_no_encoding_benefit_unique_values() {
        let enc = RleEncoding;
        let mut data = Vec::new();
        for i in 0..10u32 {
            data.extend_from_slice(&i.to_le_bytes());
        }

        let encoded = enc.encode(&data, 10, 4).unwrap();
        // 10 runs * (4 + 1) + 8 header = 58 bytes > 40 raw (no benefit)
        assert!(encoded.len() > data.len());

        let decoded = enc.decode(&encoded, 10, 4).unwrap();
        assert_eq!(decoded, data);
    }

    #[test]
    fn test_empty() {
        let enc = RleEncoding;
        let encoded = enc.encode(&[], 0, 4).unwrap();
        let decoded = enc.decode(&encoded, 0, 4).unwrap();
        assert!(decoded.is_empty());
    }

    #[test]
    fn test_single_run() {
        let enc = RleEncoding;
        let mut data = Vec::new();
        for _ in 0..1000 {
            data.extend_from_slice(&42u32.to_le_bytes());
        }

        let encoded = enc.encode(&data, 1000, 4).unwrap();
        // 1 run * (4 value + 2 varint for 1000) + 8 header = 14 bytes vs 4000 raw
        assert_eq!(encoded.len(), 14);

        let decoded = enc.decode(&encoded, 1000, 4).unwrap();
        assert_eq!(decoded, data);
    }

    #[test]
    fn test_large_run_varint() {
        let enc = RleEncoding;
        let mut data = Vec::new();
        let runLen = 100_000u32;
        for _ in 0..runLen {
            data.extend_from_slice(&42u32.to_le_bytes());
        }

        let encoded = enc.encode(&data, runLen as usize, 4).unwrap();
        // 100000 needs 3 varint bytes: 1 run * (4 + 3) + 8 = 15 bytes
        assert_eq!(encoded.len(), 15);

        let decoded = enc.decode(&encoded, runLen as usize, 4).unwrap();
        assert_eq!(decoded, data);
    }

    #[test]
    fn test_predicate_equality() {
        let enc = RleEncoding;
        let mut data = Vec::new();
        for _ in 0..3 {
            data.extend_from_slice(&1u32.to_le_bytes());
        }
        for _ in 0..4 {
            data.extend_from_slice(&2u32.to_le_bytes());
        }
        for _ in 0..3 {
            data.extend_from_slice(&1u32.to_le_bytes());
        }

        let encoded = enc.encode(&data, 10, 4).unwrap();
        let target = 1u32.to_le_bytes();
        let bitmask = enc
            .eval_predicate(&encoded, 10, 4, &Predicate::Equality(&target))
            .unwrap();
        assert_eq!(bitmask[0], 0b10000111);
        assert_eq!(bitmask[1], 0b00000011);
    }

    #[test]
    fn test_predicate_range() {
        let enc = RleEncoding;
        let mut data = Vec::new();
        for _ in 0..3 {
            data.extend_from_slice(&10u32.to_le_bytes());
        }
        for _ in 0..4 {
            data.extend_from_slice(&20u32.to_le_bytes());
        }
        for _ in 0..3 {
            data.extend_from_slice(&30u32.to_le_bytes());
        }

        let encoded = enc.encode(&data, 10, 4).unwrap();
        let lo = 15u32.to_le_bytes();
        let hi = 25u32.to_le_bytes();
        let bitmask = enc
            .eval_predicate(
                &encoded,
                10,
                4,
                &Predicate::Range {
                    low: Some(&lo),
                    high: Some(&hi),
                },
            )
            .unwrap();
        // Only the 20-run (rows 3,4,5,6) matches
        assert_eq!(bitmask[0], 0b01111000);
        assert_eq!(bitmask[1], 0b00000000);
    }

    #[test]
    fn test_varint_roundtrip() {
        // Test varint encoding/decoding for various values
        for val in [0, 1, 127, 128, 255, 16383, 16384, 100000, u32::MAX] {
            let mut buf = Vec::new();
            encode_varint(val, &mut buf);
            let (decoded, _) = decode_varint(&buf).unwrap();
            assert_eq!(decoded, val, "varint roundtrip failed for {}", val);
        }
    }
}
