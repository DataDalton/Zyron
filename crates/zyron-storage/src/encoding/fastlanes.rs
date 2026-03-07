//! FastLanes integer encoding: Frame-of-Reference (FoR) + Delta + bit-packing.
//!
//! For integer columns, subtracts the minimum value (FoR base) from all values,
//! reducing the bit width needed per value. For sorted data, applies delta
//! encoding before bit-packing. Uses unaligned u64 reads for batch unpacking.
//!
//! Based on FastLanes (VLDB 2023), tuned for page-aligned columnar storage.

use crate::encoding::{Encoding, EncodingType, Predicate};
use zyron_common::{Result, ZyronError};

pub struct FastLanesEncoding;

/// Header flags.
const FLAG_DELTA: u8 = 0x01;

/// Encoded format:
///   [0..8]    base_value: u64 (FoR base, little-endian)
///   [8]       bit_width: u8 (bits per packed value after FoR subtraction)
///   [9]       flags: u8 (bit 0 = delta encoding applied)
///   [10..12]  reserved: u16
///   [12..]    packed bit array
impl Encoding for FastLanesEncoding {
    fn encoding_type(&self) -> EncodingType {
        EncodingType::FastLanes
    }

    fn encode(&self, data: &[u8], row_count: usize, value_size: usize) -> Result<Vec<u8>> {
        if row_count == 0 {
            let out = vec![0u8; 12];
            return Ok(out);
        }

        if value_size > 8 {
            return Err(ZyronError::EncodingFailed(
                "FastLanes supports values up to 8 bytes".to_string(),
            ));
        }

        if data.len() < row_count * value_size {
            return Err(ZyronError::EncodingFailed(
                "data shorter than expected for FastLanes encoding".to_string(),
            ));
        }

        // Read all values as u64
        let mut values = Vec::with_capacity(row_count);
        for i in 0..row_count {
            values.push(read_u64_le(data, i * value_size, value_size));
        }

        // Determine if delta encoding is beneficial (sorted or nearly sorted)
        let sorted_count = values.windows(2).filter(|w| w[1] >= w[0]).count();
        let use_delta = row_count > 1 && sorted_count >= (row_count - 1) * 9 / 10;

        // FoR: find minimum value
        let base_value = values.iter().copied().min().unwrap_or(0);

        // Subtract base (FoR transform)
        let mut residuals: Vec<u64> = values.iter().map(|v| v - base_value).collect();

        // Delta encoding on residuals if beneficial
        let flags = if use_delta {
            // Delta: replace each value with the difference from the previous
            // First value stays as-is (already FoR-subtracted)
            for i in (1..residuals.len()).rev() {
                residuals[i] = residuals[i].wrapping_sub(residuals[i - 1]);
            }
            FLAG_DELTA
        } else {
            0
        };

        // Determine bit width from max residual
        let max_residual = residuals.iter().copied().max().unwrap_or(0);
        let bit_width = if max_residual == 0 {
            1
        } else {
            64 - max_residual.leading_zeros()
        } as u8;

        // Pack residuals into bit array
        let packed_bits = row_count as u64 * bit_width as u64;
        let packed_bytes = ((packed_bits + 7) / 8) as usize;
        let mut packed = vec![0u8; packed_bytes];

        for (i, &val) in residuals.iter().enumerate() {
            pack_bits(&mut packed, i as u64 * bit_width as u64, val, bit_width);
        }

        // Build output: header + packed data
        let mut out = Vec::with_capacity(12 + packed_bytes);
        out.extend_from_slice(&base_value.to_le_bytes()); // [0..8]
        out.push(bit_width); // [8]
        out.push(flags); // [9]
        out.extend_from_slice(&0u16.to_le_bytes()); // [10..12] reserved
        out.extend_from_slice(&packed);

        Ok(out)
    }

    fn decode(&self, encoded: &[u8], row_count: usize, value_size: usize) -> Result<Vec<u8>> {
        if row_count == 0 {
            return Ok(Vec::new());
        }

        if encoded.len() < 12 {
            return Err(ZyronError::DecodingFailed(
                "FastLanes header too short".to_string(),
            ));
        }

        let base_value = u64::from_le_bytes([
            encoded[0], encoded[1], encoded[2], encoded[3], encoded[4], encoded[5], encoded[6],
            encoded[7],
        ]);
        let bit_width = encoded[8];
        let flags = encoded[9];
        let use_delta = flags & FLAG_DELTA != 0;

        if bit_width == 0 || bit_width > 64 {
            return Err(ZyronError::DecodingFailed(format!(
                "invalid FastLanes bit width: {}",
                bit_width
            )));
        }

        let packed = &encoded[12..];
        let out_len = row_count * value_size;
        let mut out: Vec<u8> = Vec::with_capacity(out_len);
        unsafe {
            out.set_len(out_len);
        }
        let out_ptr = out.as_mut_ptr();
        let mask: u64 = if bit_width >= 64 {
            u64::MAX
        } else {
            (1u64 << bit_width) - 1
        };
        let bw = bit_width as u64;
        let packed_ptr = packed.as_ptr();
        let packed_len = packed.len();

        if use_delta {
            // Fused single-pass: unpack delta, prefix-sum, and write output
            // in one loop. Eliminates the intermediate residuals Vec (800KB for
            // 100K u64 values) and reduces 3 passes over data to 1.
            let mut accumulator: u64 = 0;
            match value_size {
                4 if bit_width == 1 => {
                    // bit_width=1 specialization: extract 8 deltas per packed byte.
                    // Common for auto-increment PKs and sorted columns with unit step.
                    // Eliminates per-element unpack_inline overhead (u64 read + shift + mask).
                    let out32 = out_ptr as *mut u32;
                    let base32 = base_value as u32;
                    let fullBytes = row_count / 8;

                    for b in 0..fullBytes {
                        let byte = unsafe { *packed_ptr.add(b) };
                        let idx = b * 8;

                        // Unroll 8 bit extractions per byte. Each delta is 0 or 1.
                        // Vec output is pointer-aligned and u32 writes at idx*4 are
                        // always 4-byte aligned, so use aligned write.
                        accumulator += (byte & 1) as u64;
                        unsafe {
                            out32.add(idx).write(accumulator as u32 + base32);
                        }
                        accumulator += ((byte >> 1) & 1) as u64;
                        unsafe {
                            out32.add(idx + 1).write(accumulator as u32 + base32);
                        }
                        accumulator += ((byte >> 2) & 1) as u64;
                        unsafe {
                            out32.add(idx + 2).write(accumulator as u32 + base32);
                        }
                        accumulator += ((byte >> 3) & 1) as u64;
                        unsafe {
                            out32.add(idx + 3).write(accumulator as u32 + base32);
                        }
                        accumulator += ((byte >> 4) & 1) as u64;
                        unsafe {
                            out32.add(idx + 4).write(accumulator as u32 + base32);
                        }
                        accumulator += ((byte >> 5) & 1) as u64;
                        unsafe {
                            out32.add(idx + 5).write(accumulator as u32 + base32);
                        }
                        accumulator += ((byte >> 6) & 1) as u64;
                        unsafe {
                            out32.add(idx + 6).write(accumulator as u32 + base32);
                        }
                        accumulator += ((byte >> 7) & 1) as u64;
                        unsafe {
                            out32.add(idx + 7).write(accumulator as u32 + base32);
                        }
                    }
                    for i in (fullBytes * 8)..row_count {
                        let delta =
                            unpack_inline(packed_ptr, packed_len, i as u64 * bw, bit_width, mask);
                        accumulator = accumulator.wrapping_add(delta);
                        unsafe {
                            out32.add(i).write((accumulator + base_value) as u32);
                        }
                    }
                }
                4 => {
                    // Batch-unpack 4 deltas at a time for instruction-level parallelism
                    // on superscalar CPUs. The prefix-sum is sequential but the 4 unpacks
                    // can overlap in the CPU pipeline.
                    let chunks = row_count / 4;
                    let out32 = out_ptr as *mut u32;

                    for chunk in 0..chunks {
                        let i0 = chunk * 4;
                        let d0 =
                            unpack_inline(packed_ptr, packed_len, i0 as u64 * bw, bit_width, mask);
                        let d1 = unpack_inline(
                            packed_ptr,
                            packed_len,
                            (i0 + 1) as u64 * bw,
                            bit_width,
                            mask,
                        );
                        let d2 = unpack_inline(
                            packed_ptr,
                            packed_len,
                            (i0 + 2) as u64 * bw,
                            bit_width,
                            mask,
                        );
                        let d3 = unpack_inline(
                            packed_ptr,
                            packed_len,
                            (i0 + 3) as u64 * bw,
                            bit_width,
                            mask,
                        );

                        accumulator = accumulator.wrapping_add(d0);
                        let v0 = (accumulator + base_value) as u32;
                        accumulator = accumulator.wrapping_add(d1);
                        let v1 = (accumulator + base_value) as u32;
                        accumulator = accumulator.wrapping_add(d2);
                        let v2 = (accumulator + base_value) as u32;
                        accumulator = accumulator.wrapping_add(d3);
                        let v3 = (accumulator + base_value) as u32;

                        unsafe {
                            out32.add(i0).write(v0);
                            out32.add(i0 + 1).write(v1);
                            out32.add(i0 + 2).write(v2);
                            out32.add(i0 + 3).write(v3);
                        }
                    }
                    for i in (chunks * 4)..row_count {
                        let delta =
                            unpack_inline(packed_ptr, packed_len, i as u64 * bw, bit_width, mask);
                        accumulator = accumulator.wrapping_add(delta);
                        unsafe {
                            out32.add(i).write((accumulator + base_value) as u32);
                        }
                    }
                }
                8 if bit_width == 1 => {
                    // bit_width=1 specialization for u64: extract 8 deltas per byte.
                    let out64 = out_ptr as *mut u64;
                    let fullBytes = row_count / 8;

                    for b in 0..fullBytes {
                        let byte = unsafe { *packed_ptr.add(b) };
                        let idx = b * 8;

                        accumulator += (byte & 1) as u64;
                        unsafe {
                            out64.add(idx).write(accumulator + base_value);
                        }
                        accumulator += ((byte >> 1) & 1) as u64;
                        unsafe {
                            out64.add(idx + 1).write(accumulator + base_value);
                        }
                        accumulator += ((byte >> 2) & 1) as u64;
                        unsafe {
                            out64.add(idx + 2).write(accumulator + base_value);
                        }
                        accumulator += ((byte >> 3) & 1) as u64;
                        unsafe {
                            out64.add(idx + 3).write(accumulator + base_value);
                        }
                        accumulator += ((byte >> 4) & 1) as u64;
                        unsafe {
                            out64.add(idx + 4).write(accumulator + base_value);
                        }
                        accumulator += ((byte >> 5) & 1) as u64;
                        unsafe {
                            out64.add(idx + 5).write(accumulator + base_value);
                        }
                        accumulator += ((byte >> 6) & 1) as u64;
                        unsafe {
                            out64.add(idx + 6).write(accumulator + base_value);
                        }
                        accumulator += ((byte >> 7) & 1) as u64;
                        unsafe {
                            out64.add(idx + 7).write(accumulator + base_value);
                        }
                    }
                    for i in (fullBytes * 8)..row_count {
                        let delta =
                            unpack_inline(packed_ptr, packed_len, i as u64 * bw, bit_width, mask);
                        accumulator = accumulator.wrapping_add(delta);
                        unsafe {
                            out64.add(i).write(accumulator + base_value);
                        }
                    }
                }
                8 => {
                    let out64 = out_ptr as *mut u64;
                    let chunks = row_count / 4;
                    for chunk in 0..chunks {
                        let i0 = chunk * 4;
                        let d0 =
                            unpack_inline(packed_ptr, packed_len, i0 as u64 * bw, bit_width, mask);
                        let d1 = unpack_inline(
                            packed_ptr,
                            packed_len,
                            (i0 + 1) as u64 * bw,
                            bit_width,
                            mask,
                        );
                        let d2 = unpack_inline(
                            packed_ptr,
                            packed_len,
                            (i0 + 2) as u64 * bw,
                            bit_width,
                            mask,
                        );
                        let d3 = unpack_inline(
                            packed_ptr,
                            packed_len,
                            (i0 + 3) as u64 * bw,
                            bit_width,
                            mask,
                        );

                        accumulator = accumulator.wrapping_add(d0);
                        unsafe {
                            out64.add(i0).write(accumulator + base_value);
                        }
                        accumulator = accumulator.wrapping_add(d1);
                        unsafe {
                            out64.add(i0 + 1).write(accumulator + base_value);
                        }
                        accumulator = accumulator.wrapping_add(d2);
                        unsafe {
                            out64.add(i0 + 2).write(accumulator + base_value);
                        }
                        accumulator = accumulator.wrapping_add(d3);
                        unsafe {
                            out64.add(i0 + 3).write(accumulator + base_value);
                        }
                    }
                    for i in (chunks * 4)..row_count {
                        let delta =
                            unpack_inline(packed_ptr, packed_len, i as u64 * bw, bit_width, mask);
                        accumulator = accumulator.wrapping_add(delta);
                        unsafe {
                            out64.add(i).write(accumulator + base_value);
                        }
                    }
                }
                _ => {
                    for i in 0..row_count {
                        let delta =
                            unpack_inline(packed_ptr, packed_len, i as u64 * bw, bit_width, mask);
                        accumulator = accumulator.wrapping_add(delta);
                        let val = (accumulator + base_value).to_le_bytes();
                        unsafe {
                            std::ptr::copy_nonoverlapping(
                                val.as_ptr(),
                                out_ptr.add(i * value_size),
                                value_size,
                            );
                        }
                    }
                }
            }
        } else {
            // No delta: unpack each value and write directly to output.
            // Process 4 values at a time for instruction-level parallelism.
            match value_size {
                4 => {
                    let out32 = out_ptr as *mut u32;
                    let chunks = row_count / 4;
                    for chunk in 0..chunks {
                        let i0 = chunk * 4;
                        let r0 =
                            unpack_inline(packed_ptr, packed_len, i0 as u64 * bw, bit_width, mask);
                        let r1 = unpack_inline(
                            packed_ptr,
                            packed_len,
                            (i0 + 1) as u64 * bw,
                            bit_width,
                            mask,
                        );
                        let r2 = unpack_inline(
                            packed_ptr,
                            packed_len,
                            (i0 + 2) as u64 * bw,
                            bit_width,
                            mask,
                        );
                        let r3 = unpack_inline(
                            packed_ptr,
                            packed_len,
                            (i0 + 3) as u64 * bw,
                            bit_width,
                            mask,
                        );
                        unsafe {
                            out32.add(i0).write((r0 + base_value) as u32);
                            out32.add(i0 + 1).write((r1 + base_value) as u32);
                            out32.add(i0 + 2).write((r2 + base_value) as u32);
                            out32.add(i0 + 3).write((r3 + base_value) as u32);
                        }
                    }
                    for i in (chunks * 4)..row_count {
                        let r =
                            unpack_inline(packed_ptr, packed_len, i as u64 * bw, bit_width, mask);
                        unsafe {
                            out32.add(i).write((r + base_value) as u32);
                        }
                    }
                }
                8 => {
                    let out64 = out_ptr as *mut u64;
                    let chunks = row_count / 4;
                    for chunk in 0..chunks {
                        let i0 = chunk * 4;
                        let r0 =
                            unpack_inline(packed_ptr, packed_len, i0 as u64 * bw, bit_width, mask);
                        let r1 = unpack_inline(
                            packed_ptr,
                            packed_len,
                            (i0 + 1) as u64 * bw,
                            bit_width,
                            mask,
                        );
                        let r2 = unpack_inline(
                            packed_ptr,
                            packed_len,
                            (i0 + 2) as u64 * bw,
                            bit_width,
                            mask,
                        );
                        let r3 = unpack_inline(
                            packed_ptr,
                            packed_len,
                            (i0 + 3) as u64 * bw,
                            bit_width,
                            mask,
                        );
                        unsafe {
                            out64.add(i0).write(r0 + base_value);
                            out64.add(i0 + 1).write(r1 + base_value);
                            out64.add(i0 + 2).write(r2 + base_value);
                            out64.add(i0 + 3).write(r3 + base_value);
                        }
                    }
                    for i in (chunks * 4)..row_count {
                        let r =
                            unpack_inline(packed_ptr, packed_len, i as u64 * bw, bit_width, mask);
                        unsafe {
                            out64.add(i).write(r + base_value);
                        }
                    }
                }
                _ => {
                    for i in 0..row_count {
                        let r =
                            unpack_inline(packed_ptr, packed_len, i as u64 * bw, bit_width, mask);
                        let val = (r + base_value).to_le_bytes();
                        unsafe {
                            std::ptr::copy_nonoverlapping(
                                val.as_ptr(),
                                out_ptr.add(i * value_size),
                                value_size,
                            );
                        }
                    }
                }
            }
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

        if encoded.len() < 12 {
            return Err(ZyronError::DecodingFailed(
                "FastLanes header too short for predicate evaluation".to_string(),
            ));
        }

        let base_value = u64::from_le_bytes([
            encoded[0], encoded[1], encoded[2], encoded[3], encoded[4], encoded[5], encoded[6],
            encoded[7],
        ]);
        let bit_width = encoded[8];
        let flags = encoded[9];
        let use_delta = flags & FLAG_DELTA != 0;

        let packed = &encoded[12..];
        let mask: u64 = if bit_width >= 64 {
            u64::MAX
        } else {
            (1u64 << bit_width) - 1
        };

        // For non-delta FoR encoding, evaluate predicates directly on packed
        // residuals by transforming bounds into the FoR domain.
        if !use_delta {
            let maxResidual = if bit_width >= 64 {
                u64::MAX
            } else {
                (1u64 << bit_width) - 1
            };
            let maxRepresentable = base_value.saturating_add(maxResidual);

            match predicate {
                Predicate::Range { low, high } => {
                    let loVal = match low {
                        Some(lo_bytes) => read_u64_le(lo_bytes, 0, lo_bytes.len().min(8)),
                        None => 0,
                    };
                    let hiVal = match high {
                        Some(hi_bytes) => read_u64_le(hi_bytes, 0, hi_bytes.len().min(8)),
                        None => u64::MAX,
                    };

                    // Segment-level skip: all values outside range
                    if loVal > maxRepresentable || hiVal < base_value {
                        let bitmaskLen = (row_count + 7) / 8;
                        return Ok(vec![0u8; bitmaskLen]);
                    }

                    // Segment-level accept: entire range within bounds
                    if loVal <= base_value && hiVal >= maxRepresentable {
                        let bitmaskLen = (row_count + 7) / 8;
                        let mut bitmask = vec![0xFFu8; bitmaskLen];
                        let trailing = row_count % 8;
                        if trailing != 0 {
                            bitmask[bitmaskLen - 1] = (1u8 << trailing) - 1;
                        }
                        return Ok(bitmask);
                    }

                    // Row-level filtering on residuals
                    let loResidual = if loVal > base_value {
                        loVal - base_value
                    } else {
                        0
                    };
                    let hiResidual = if hiVal >= base_value {
                        (hiVal - base_value).min(maxResidual)
                    } else {
                        return Ok(vec![0u8; (row_count + 7) / 8]);
                    };

                    let bitmaskLen = (row_count + 7) / 8;
                    let mut bitmask = vec![0u8; bitmaskLen];
                    for i in 0..row_count {
                        let residual =
                            unpack_fast(packed, i as u64 * bit_width as u64, bit_width, mask);
                        if residual >= loResidual && residual <= hiResidual {
                            bitmask[i / 8] |= 1 << (i % 8);
                        }
                    }
                    return Ok(bitmask);
                }
                Predicate::Equality(target) => {
                    let targetVal = read_u64_le(target, 0, target.len().min(8));
                    let bitmaskLen = (row_count + 7) / 8;
                    if targetVal < base_value || targetVal > maxRepresentable {
                        return Ok(vec![0u8; bitmaskLen]);
                    }
                    let targetResidual = targetVal - base_value;
                    let mut bitmask = vec![0u8; bitmaskLen];
                    for i in 0..row_count {
                        let residual =
                            unpack_fast(packed, i as u64 * bit_width as u64, bit_width, mask);
                        if residual == targetResidual {
                            bitmask[i / 8] |= 1 << (i % 8);
                        }
                    }
                    return Ok(bitmask);
                }
                Predicate::In(values) => {
                    let targetResiduals: Vec<u64> = values
                        .iter()
                        .filter_map(|v| {
                            let val = read_u64_le(v, 0, v.len().min(8));
                            if val >= base_value && val <= maxRepresentable {
                                Some(val - base_value)
                            } else {
                                None
                            }
                        })
                        .collect();
                    let bitmaskLen = (row_count + 7) / 8;
                    if targetResiduals.is_empty() {
                        return Ok(vec![0u8; bitmaskLen]);
                    }
                    let mut bitmask = vec![0u8; bitmaskLen];
                    for i in 0..row_count {
                        let residual =
                            unpack_fast(packed, i as u64 * bit_width as u64, bit_width, mask);
                        if targetResiduals.contains(&residual) {
                            bitmask[i / 8] |= 1 << (i % 8);
                        }
                    }
                    return Ok(bitmask);
                }
            }
        }

        // For delta-encoded data, evaluate the predicate without full decode.
        let bitmaskLen = (row_count + 7) / 8;
        let mut bitmask = vec![0u8; bitmaskLen];

        // For Range predicates, try the constant-step fast path first.
        // Delta-encoded sequential data has packed values [r0, d, d, d, ...]
        // where r0 is the first FoR-subtracted value and d is the constant step.
        // After prefix sum: value[i] = base + r0 + i*d for i > 0, value[0] = base + r0.
        // This gives O(1) range computation instead of O(N) unpack + prefix sum.
        if let Predicate::Range { low, high } = predicate {
            if row_count >= 2 {
                let r0 = unpack_fast(packed, 0, bit_width, mask);
                let step = unpack_fast(packed, bit_width as u64, bit_width, mask);

                // Spot-check that all deltas from index 1 onward are identical
                let spots = [
                    row_count / 4,
                    row_count / 2,
                    row_count * 3 / 4,
                    row_count - 1,
                ];
                let isConstantStep = spots.iter().all(|&idx| {
                    if idx < 1 || idx >= row_count {
                        return true;
                    }
                    unpack_fast(packed, idx as u64 * bit_width as u64, bit_width, mask) == step
                });

                if isConstantStep && step > 0 {
                    // After prefix sum: ps[0] = r0, ps[i] = r0 + i*step
                    // Original value[i] = base_value + r0 + i * step
                    let loVal = match low {
                        Some(lo) => read_u64_le(lo, 0, lo.len().min(8)),
                        None => 0,
                    };
                    let hiVal = match high {
                        Some(hi) => read_u64_le(hi, 0, hi.len().min(8)),
                        None => u64::MAX,
                    };

                    let firstValue = base_value + r0;
                    let lastValue = firstValue + (row_count as u64 - 1) * step;

                    // Segment-level skip/accept
                    if firstValue > hiVal || lastValue < loVal {
                        return Ok(bitmask);
                    }
                    if firstValue >= loVal && lastValue <= hiVal {
                        for byte in &mut bitmask[..bitmaskLen] {
                            *byte = 0xFF;
                        }
                        let trailing = row_count % 8;
                        if trailing != 0 {
                            bitmask[bitmaskLen - 1] = (1u8 << trailing) - 1;
                        }
                        return Ok(bitmask);
                    }

                    // Compute matching index range analytically
                    let loStart = if loVal <= firstValue {
                        0
                    } else {
                        let diff = loVal - firstValue;
                        ((diff + step - 1) / step) as usize
                    };
                    let hiEnd = if hiVal >= lastValue {
                        row_count
                    } else {
                        let diff = hiVal - firstValue;
                        (diff / step + 1) as usize
                    };
                    let hiEnd = hiEnd.min(row_count);

                    // Bulk-fill bitmask for the matching range
                    if loStart < hiEnd {
                        fill_bitmask_range(&mut bitmask, loStart, hiEnd);
                    }

                    return Ok(bitmask);
                }
            }
        }

        // Full unpack + prefix sum path for non-constant-delta data
        let mut residuals = vec![0u64; row_count];
        unpack_batch(packed, bit_width, mask, row_count, &mut residuals);

        // Prefix sum to reverse delta encoding
        for i in 1..row_count {
            residuals[i] = residuals[i].wrapping_add(residuals[i - 1]);
        }

        // For Range predicates on sorted delta data, use binary search to find
        // the contiguous range of matching rows, then bulk-fill the bitmask.
        // This is O(log N + range_size) instead of O(N) per-row comparison.
        // Uses numeric u64 comparison, consistent with eval_predicate_on_raw.
        if let Predicate::Range { low, high } = predicate {
            // Check if prefix-summed residuals are monotonically non-decreasing.
            // Delta encoding is applied when >= 90% sorted, so spot-check.
            let isSorted = row_count <= 1
                || residuals[row_count - 1] >= residuals[0] && {
                    let step = (row_count / 16).max(1);
                    let mut sorted = true;
                    let mut prev = residuals[0];
                    let mut idx = step;
                    while idx < row_count {
                        if residuals[idx] < prev {
                            sorted = false;
                            break;
                        }
                        prev = residuals[idx];
                        idx += step;
                    }
                    sorted
                };

            if isSorted {
                // Convert bounds to u64 for numeric comparison
                let loVal = match low {
                    Some(lo) => read_u64_le(lo, 0, lo.len().min(8)),
                    None => 0,
                };
                let hiVal = match high {
                    Some(hi) => read_u64_le(hi, 0, hi.len().min(8)),
                    None => u64::MAX,
                };

                // Convert to residual domain
                let loResidual = if loVal > base_value {
                    loVal - base_value
                } else {
                    0
                };
                let hiResidual = if hiVal >= base_value {
                    hiVal - base_value
                } else {
                    return Ok(bitmask);
                };

                // Binary search for the contiguous matching range
                let loStart = residuals.partition_point(|&r| r < loResidual);
                let hiEnd = residuals[loStart..].partition_point(|&r| r <= hiResidual) + loStart;

                fill_bitmask_range(&mut bitmask, loStart, hiEnd);
                return Ok(bitmask);
            }
        }

        // General fallback for non-sorted delta data or non-Range predicates.
        // Uses u64 numeric comparison for consistency with eval_predicate_on_raw.
        match predicate {
            Predicate::Range { low, high } => {
                let loVal = match low {
                    Some(lo) => read_u64_le(lo, 0, lo.len().min(8)),
                    None => 0,
                };
                let hiVal = match high {
                    Some(hi) => read_u64_le(hi, 0, hi.len().min(8)),
                    None => u64::MAX,
                };
                for i in 0..row_count {
                    let v = residuals[i] + base_value;
                    if v >= loVal && v <= hiVal {
                        bitmask[i / 8] |= 1 << (i % 8);
                    }
                }
            }
            Predicate::Equality(target) => {
                let targetVal = read_u64_le(target, 0, target.len().min(8));
                for i in 0..row_count {
                    if residuals[i] + base_value == targetVal {
                        bitmask[i / 8] |= 1 << (i % 8);
                    }
                }
            }
            Predicate::In(values) => {
                let targets: Vec<u64> = values
                    .iter()
                    .map(|v| read_u64_le(v, 0, v.len().min(8)))
                    .collect();
                for i in 0..row_count {
                    let v = residuals[i] + base_value;
                    if targets.contains(&v) {
                        bitmask[i / 8] |= 1 << (i % 8);
                    }
                }
            }
        }

        Ok(bitmask)
    }
}

/// Reads a value of up to 8 bytes from data as a u64 (little-endian).
#[inline]
fn read_u64_le(data: &[u8], offset: usize, size: usize) -> u64 {
    let end = (offset + size).min(data.len());
    let slice = &data[offset..end];
    let mut buf = [0u8; 8];
    let copy_len = slice.len().min(8);
    buf[..copy_len].copy_from_slice(&slice[..copy_len]);
    u64::from_le_bytes(buf)
}

/// Packs a value at the given bit offset.
#[inline]
fn pack_bits(packed: &mut [u8], bit_offset: u64, value: u64, bit_width: u8) {
    let byte_idx = (bit_offset / 8) as usize;
    let bit_idx = (bit_offset % 8) as u32;
    let mask = if bit_width >= 64 {
        u64::MAX
    } else {
        (1u64 << bit_width) - 1
    };
    let val = value & mask;
    let shifted = val << bit_idx;
    let shifted_bytes = shifted.to_le_bytes();
    let total_bits = bit_idx + bit_width as u32;
    let bytes_needed = ((total_bits + 7) / 8) as usize;

    for j in 0..bytes_needed.min(8) {
        if byte_idx + j < packed.len() {
            packed[byte_idx + j] |= shifted_bytes[j];
        }
    }
}

/// Unpacks a single value using unaligned u64 read instead of 9-byte memcpy.
/// The unaligned read is faster on most modern CPUs where unaligned loads
/// execute in a single cycle.
#[inline(always)]
fn unpack_fast(packed: &[u8], bit_offset: u64, bit_width: u8, mask: u64) -> u64 {
    unpack_inline(packed.as_ptr(), packed.len(), bit_offset, bit_width, mask)
}

/// Raw pointer version of unpack_fast. Takes pre-computed pointer and length
/// to avoid repeated slice header access in tight loops. The caller must
/// guarantee packed_ptr points to a valid buffer of packed_len bytes.
#[inline(always)]
fn unpack_inline(
    packed_ptr: *const u8,
    packed_len: usize,
    bit_offset: u64,
    bit_width: u8,
    mask: u64,
) -> u64 {
    let byte_idx = (bit_offset >> 3) as usize;
    let bit_idx = (bit_offset & 7) as u32;

    if byte_idx + 8 <= packed_len {
        let raw = unsafe { (packed_ptr.add(byte_idx) as *const u64).read_unaligned() };
        let val = (raw >> bit_idx) & mask;

        if bit_idx + bit_width as u32 > 64 {
            if byte_idx + 9 <= packed_len {
                let hi = unsafe { *packed_ptr.add(byte_idx + 8) } as u64;
                return (val | (hi << (64 - bit_idx))) & mask;
            }
            // 9th byte unavailable, fall through to safe fallback
        } else {
            return val;
        }
    }

    // Fallback for the last few bytes
    let mut buf = [0u8; 8];
    let available = packed_len.saturating_sub(byte_idx).min(8);
    unsafe {
        std::ptr::copy_nonoverlapping(packed_ptr.add(byte_idx), buf.as_mut_ptr(), available);
    }
    let raw = u64::from_le_bytes(buf);
    (raw >> bit_idx) & mask
}

/// Batch unpacks all values from the packed bit array into a u64 output buffer.
/// Uses unaligned u64 reads for the inner loop.
#[inline]
fn unpack_batch(packed: &[u8], bit_width: u8, mask: u64, count: usize, out: &mut [u64]) {
    let bw = bit_width as u64;
    let packed_ptr = packed.as_ptr();
    let packed_len = packed.len();

    for i in 0..count {
        out[i] = unpack_inline(packed_ptr, packed_len, i as u64 * bw, bit_width, mask);
    }
}

/// Sets bits [start, end) in a bitmask. Handles partial first/last bytes
/// and fills full bytes with 0xFF in the middle.
#[inline]
fn fill_bitmask_range(bitmask: &mut [u8], start: usize, end: usize) {
    if start >= end {
        return;
    }
    let firstByte = start / 8;
    let lastByte = (end - 1) / 8;
    let firstBit = start % 8;
    let lastBit = (end - 1) % 8;

    if firstByte == lastByte {
        for b in firstBit..=lastBit {
            bitmask[firstByte] |= 1 << b;
        }
    } else {
        for b in firstBit..8 {
            bitmask[firstByte] |= 1 << b;
        }
        for byte in (firstByte + 1)..lastByte {
            bitmask[byte] = 0xFF;
        }
        for b in 0..=lastBit {
            bitmask[lastByte] |= 1 << b;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_roundtrip_i32_sequential() {
        let enc = FastLanesEncoding;
        let mut data = Vec::new();
        for i in 100..200u32 {
            data.extend_from_slice(&i.to_le_bytes());
        }

        let encoded = enc.encode(&data, 100, 4).unwrap();
        // Should use delta encoding for sorted data
        assert!(encoded[9] & FLAG_DELTA != 0);
        // Delta of sorted sequence = all 1s, bit_width should be small
        assert!(encoded.len() < data.len());

        let decoded = enc.decode(&encoded, 100, 4).unwrap();
        assert_eq!(decoded, data);
    }

    #[test]
    fn test_roundtrip_i64_random() {
        let enc = FastLanesEncoding;
        let values: Vec<u64> = vec![1000, 5000, 2000, 8000, 3000, 9000, 1500, 7000];
        let mut data = Vec::new();
        for v in &values {
            data.extend_from_slice(&v.to_le_bytes());
        }

        let encoded = enc.encode(&data, 8, 8).unwrap();
        let decoded = enc.decode(&encoded, 8, 8).unwrap();
        assert_eq!(decoded, data);
    }

    #[test]
    fn test_roundtrip_small_values() {
        let enc = FastLanesEncoding;
        let mut data = Vec::new();
        // Values 0..10, FoR base=0, bit_width=4
        for i in 0..10u32 {
            data.extend_from_slice(&i.to_le_bytes());
        }

        let encoded = enc.encode(&data, 10, 4).unwrap();
        let decoded = enc.decode(&encoded, 10, 4).unwrap();
        assert_eq!(decoded, data);
    }

    #[test]
    fn test_roundtrip_constant_values() {
        let enc = FastLanesEncoding;
        let mut data = Vec::new();
        for _ in 0..50 {
            data.extend_from_slice(&42u32.to_le_bytes());
        }

        let encoded = enc.encode(&data, 50, 4).unwrap();
        // All same value: FoR base=42, residuals all 0, bit_width=1
        assert_eq!(encoded[8], 1); // bit_width

        let decoded = enc.decode(&encoded, 50, 4).unwrap();
        assert_eq!(decoded, data);
    }

    #[test]
    fn test_empty() {
        let enc = FastLanesEncoding;
        let encoded = enc.encode(&[], 0, 4).unwrap();
        let decoded = enc.decode(&encoded, 0, 4).unwrap();
        assert!(decoded.is_empty());
    }

    #[test]
    fn test_for_compression_ratio() {
        let enc = FastLanesEncoding;
        let mut data = Vec::new();
        // Strictly ascending values in a narrow range.
        // FoR base = 1_000_000, delta encoding produces all-1 residuals (1 bit each).
        // 1000 values * 1 bit = 125 bytes + 12 byte header = 137 bytes vs 4000 raw.
        for i in 0..1000u32 {
            data.extend_from_slice(&(1_000_000 + i).to_le_bytes());
        }

        let encoded = enc.encode(&data, 1000, 4).unwrap();
        assert!(encoded.len() < data.len());
    }

    #[test]
    fn test_predicate_range_skip() {
        let enc = FastLanesEncoding;
        let mut data = Vec::new();
        for i in 100..200u32 {
            data.extend_from_slice(&i.to_le_bytes());
        }

        let encoded = enc.encode(&data, 100, 4).unwrap();

        // Range [500, 600]: entirely above all values, should skip
        let lo = 500u32.to_le_bytes();
        let hi = 600u32.to_le_bytes();
        let bitmask = enc
            .eval_predicate(
                &encoded,
                100,
                4,
                &Predicate::Range {
                    low: Some(&lo),
                    high: Some(&hi),
                },
            )
            .unwrap();
        // All zeros (no matches)
        assert!(bitmask.iter().all(|&b| b == 0));
    }
}
