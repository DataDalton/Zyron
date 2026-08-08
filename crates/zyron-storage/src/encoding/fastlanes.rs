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

/// Header flags. Each flag is its own bit so an old reader masking `& FLAG_DELTA`
/// never misreads a new flag.
const FLAG_DELTA: u8 = 0x01;
/// Second-order delta (delta-of-delta) applied to the FoR residuals.
const FLAG_DELTA_OF_DELTA: u8 = 0x02;
/// Patched frame-of-reference: low-width packed residuals plus an exception
/// table for the few values that exceed the chosen width.
const FLAG_PFOR: u8 = 0x04;
/// Constant-step closed form (no packed bit array).
const FLAG_CONST_STEP: u8 = 0x08;
/// Reserved for per-mini-block bit width (A4).
#[allow(dead_code)]
const FLAG_MINIBLOCK: u8 = 0x10;
/// Reserved for effective-resolution scale (A5).
#[allow(dead_code)]
const FLAG_SCALE: u8 = 0x20;

/// 8-byte (and narrower) encoded format:
///   [0..8]    base_value: u64 (FoR base, little-endian)
///   [8]       bit_width: u8 (bits per packed value after FoR subtraction)
///   [9]       flags: u8 (FLAG_DELTA / FLAG_DELTA_OF_DELTA)
///   [10..12]  reserved: u16
///   [12..]    packed bit array
///
/// 16-byte (i128/u128) encoded format:
///   [0..16]   base_value: u128 (FoR base, little-endian)
///   [16]      bit_width: u8 (bits per packed value, up to 128)
///   [17]      flags: u8
///   [18..24]  reserved
///   [24..]    packed bit array
const WIDE_HEADER_SIZE: usize = 24;
impl Encoding for FastLanesEncoding {
    fn encoding_type(&self) -> EncodingType {
        EncodingType::FastLanes
    }

    fn encode(&self, data: &[u8], row_count: usize, value_size: usize) -> Result<Vec<u8>> {
        if row_count == 0 {
            let out = vec![0u8; 12];
            return Ok(out);
        }

        if value_size == 16 {
            if data.len() < row_count * 16 {
                return Err(ZyronError::EncodingFailed(
                    "data shorter than expected for FastLanes 128-bit encoding".to_string(),
                ));
            }
            return encode_wide(data, row_count);
        }

        if value_size > 8 {
            return Err(ZyronError::EncodingFailed(
                "FastLanes supports 1..=8 or 16 byte values".to_string(),
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

        let base_value = values.iter().copied().min().unwrap_or(0);
        let mut best = encode_narrow_core(&values, row_count);

        // Effective-resolution scale (A5): when every residual shares a common
        // factor g (e.g. a us-granular column promoted to ps -> g = 1_000_000),
        // encode value/g losslessly and record g. Gated by FLAG_SCALE; inner
        // is a full core blob, so decode recurses one level.
        let g = gcd_residual_u64(&values, base_value);
        if g > 1 {
            let q: Vec<u64> = values.iter().map(|v| (v - base_value) / g).collect();
            let inner = encode_narrow_core(&q, row_count);
            if 20 + inner.len() < best.len() {
                let mut scaled = Vec::with_capacity(20 + inner.len());
                scaled.extend_from_slice(&base_value.to_le_bytes()); // [0..8] base
                scaled.push(0); // [8] unused
                scaled.push(FLAG_SCALE); // [9] flags
                scaled.extend_from_slice(&0u16.to_le_bytes()); // [10..12]
                scaled.extend_from_slice(&g.to_le_bytes()); // [12..20] scale
                scaled.extend_from_slice(&inner); // [20..] inner core blob
                best = scaled;
            }
        }
        Ok(best)
    }

    fn decode(&self, encoded: &[u8], row_count: usize, value_size: usize) -> Result<Vec<u8>> {
        if row_count == 0 {
            return Ok(Vec::new());
        }

        if value_size == 16 {
            return decode_wide(encoded, row_count);
        }

        if encoded.len() < 12 {
            return Err(ZyronError::DecodingFailed(
                "FastLanes header too short".to_string(),
            ));
        }

        // Effective-resolution scale wrapper: [base:u64][_][FLAG_SCALE][_][scale:u64][inner].
        if encoded[9] & FLAG_SCALE != 0 {
            if encoded.len() < 20 {
                return Err(ZyronError::DecodingFailed(
                    "FastLanes scale blob too short".to_string(),
                ));
            }
            let base = u64::from_le_bytes([
                encoded[0], encoded[1], encoded[2], encoded[3], encoded[4], encoded[5], encoded[6],
                encoded[7],
            ]);
            let scale = u64::from_le_bytes([
                encoded[12],
                encoded[13],
                encoded[14],
                encoded[15],
                encoded[16],
                encoded[17],
                encoded[18],
                encoded[19],
            ]);
            let q_raw = self.decode(&encoded[20..], row_count, value_size)?;
            let mut out = vec![0u8; row_count * value_size];
            for i in 0..row_count {
                let q = read_u64_le(&q_raw, i * value_size, value_size);
                let v = base.wrapping_add(q.wrapping_mul(scale));
                write_le(&mut out, i, value_size, v);
            }
            return Ok(out);
        }

        let base_value = u64::from_le_bytes([
            encoded[0], encoded[1], encoded[2], encoded[3], encoded[4], encoded[5], encoded[6],
            encoded[7],
        ]);
        let bit_width = encoded[8];
        let flags = encoded[9];
        let use_delta = flags & FLAG_DELTA != 0;
        let use_dod = flags & FLAG_DELTA_OF_DELTA != 0;

        // Constant-step closed form: [first_value:u64][.. step:u64]. No packed
        // bit array, so this is handled before the bit-width check.
        if flags & FLAG_CONST_STEP != 0 {
            if encoded.len() < 20 {
                return Err(ZyronError::DecodingFailed(
                    "FastLanes constant-step blob too short".to_string(),
                ));
            }
            let first = base_value;
            let step = u64::from_le_bytes([
                encoded[12],
                encoded[13],
                encoded[14],
                encoded[15],
                encoded[16],
                encoded[17],
                encoded[18],
                encoded[19],
            ]);
            // The values are the linear sequence first + i*step. Storing through a
            // typed pointer (instead of the byte-wise write_le) lets the loop
            // vectorize into SIMD stores; value_size 4 and 8 are the hot column
            // widths. Output is fully written, so it starts uninitialized to skip
            // a redundant memset. write_unaligned is used because a Vec<u8> buffer
            // is only byte-aligned; on x86_64 it compiles to the same store.
            let out_len = row_count * value_size;
            #[allow(clippy::uninit_vec)]
            let mut out: Vec<u8> = {
                let mut v = Vec::with_capacity(out_len);
                unsafe { v.set_len(out_len) };
                v
            };
            match value_size {
                4 => {
                    let p = out.as_mut_ptr() as *mut u32;
                    for i in 0..row_count {
                        let v = first.wrapping_add((i as u64).wrapping_mul(step)) as u32;
                        unsafe { p.add(i).write_unaligned(v) };
                    }
                }
                8 => {
                    let p = out.as_mut_ptr() as *mut u64;
                    for i in 0..row_count {
                        let v = first.wrapping_add((i as u64).wrapping_mul(step));
                        unsafe { p.add(i).write_unaligned(v) };
                    }
                }
                _ => {
                    for i in 0..row_count {
                        write_le(
                            &mut out,
                            i,
                            value_size,
                            first.wrapping_add((i as u64).wrapping_mul(step)),
                        );
                    }
                }
            }
            return Ok(out);
        }

        // Patched FoR: [hdr][exception table][packed low-width residuals].
        if flags & FLAG_PFOR != 0 {
            let exc_count = u16::from_le_bytes([encoded[10], encoded[11]]) as usize;
            let table_off = 12usize;
            let table_bytes = exc_count * 12;
            if bit_width == 0 || bit_width > 64 || encoded.len() < table_off + table_bytes {
                return Err(ZyronError::DecodingFailed(
                    "FastLanes PFOR blob malformed".to_string(),
                ));
            }
            let packed = &encoded[table_off + table_bytes..];
            let mask: u64 = if bit_width >= 64 {
                u64::MAX
            } else {
                (1u64 << bit_width) - 1
            };
            let mut r = vec![0u64; row_count];
            unpack_batch(packed, bit_width, mask, row_count, &mut r);
            let mut out = vec![0u8; row_count * value_size];
            write_residuals_add_base(&mut out, value_size, &r, base_value);
            for e in 0..exc_count {
                let o = table_off + e * 12;
                let pos = u32::from_le_bytes([
                    encoded[o],
                    encoded[o + 1],
                    encoded[o + 2],
                    encoded[o + 3],
                ]) as usize;
                let resid = u64::from_le_bytes([
                    encoded[o + 4],
                    encoded[o + 5],
                    encoded[o + 6],
                    encoded[o + 7],
                    encoded[o + 8],
                    encoded[o + 9],
                    encoded[o + 10],
                    encoded[o + 11],
                ]);
                if pos < row_count {
                    write_le(&mut out, pos, value_size, resid.wrapping_add(base_value));
                }
            }
            return Ok(out);
        }

        // Per-mini-block bit width: [hdr][ (width:1)(byte-aligned packed) ]*.
        if flags & FLAG_MINIBLOCK != 0 {
            let nblocks = row_count.div_ceil(MINIBLOCK_SIZE);
            let mut off = 12usize;
            let mut out = vec![0u8; row_count * value_size];
            for b in 0..nblocks {
                if off >= encoded.len() {
                    return Err(ZyronError::DecodingFailed(
                        "FastLanes mini-block blob truncated".to_string(),
                    ));
                }
                let bw = encoded[off];
                off += 1;
                if bw == 0 || bw > 64 {
                    return Err(ZyronError::DecodingFailed(format!(
                        "invalid FastLanes mini-block width: {bw}"
                    )));
                }
                let start = b * MINIBLOCK_SIZE;
                let end = (start + MINIBLOCK_SIZE).min(row_count);
                let blen = end - start;
                let block_bytes = (blen as u64 * bw as u64).div_ceil(8) as usize;
                if off + block_bytes > encoded.len() {
                    return Err(ZyronError::DecodingFailed(
                        "FastLanes mini-block blob truncated".to_string(),
                    ));
                }
                let packed = &encoded[off..off + block_bytes];
                let mut block = vec![0u64; blen];
                unpack_block_into(packed, bw, blen, &mut block);
                write_residuals_add_base(
                    &mut out[start * value_size..],
                    value_size,
                    &block,
                    base_value,
                );
                off += block_bytes;
            }
            return Ok(out);
        }

        if bit_width == 0 || bit_width > 64 {
            return Err(ZyronError::DecodingFailed(format!(
                "invalid FastLanes bit width: {}",
                bit_width
            )));
        }

        let packed = &encoded[12..];

        if use_dod {
            // Reconstruct FoR residuals from the packed [r0, zz(d1), zz(dd2)..]
            // stream via a double prefix sum, then re-add the FoR base.
            let mask: u64 = if bit_width >= 64 {
                u64::MAX
            } else {
                (1u64 << bit_width) - 1
            };
            let mut r = vec![0u64; row_count];
            unpack_batch(packed, bit_width, mask, row_count, &mut r);
            let mut out: Vec<u8> = vec![0u8; row_count * value_size];
            let mut residual: u64 = r[0];
            write_le(&mut out, 0, value_size, residual.wrapping_add(base_value));
            if row_count > 1 {
                let mut delta = unzigzag_i64(r[1]) as u64;
                residual = residual.wrapping_add(delta);
                write_le(&mut out, 1, value_size, residual.wrapping_add(base_value));
                for i in 2..row_count {
                    let dd = unzigzag_i64(r[i]);
                    delta = delta.wrapping_add(dd as u64);
                    residual = residual.wrapping_add(delta);
                    write_le(&mut out, i, value_size, residual.wrapping_add(base_value));
                }
            }
            return Ok(out);
        }
        let out_len = row_count * value_size;
        // SAFETY: the decode loop below writes every one of `out_len` bytes
        // before any read. Zeroing first would memset the whole buffer just
        // to overwrite it, halving decode throughput on the hot scan path.
        #[allow(clippy::uninit_vec)]
        let mut out: Vec<u8> = {
            let mut v = Vec::with_capacity(out_len);
            unsafe { v.set_len(out_len) };
            v
        };
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
                        accumulator = accumulator.wrapping_add((byte & 1) as u64);
                        unsafe {
                            out32
                                .add(idx)
                                .write((accumulator as u32).wrapping_add(base32));
                        }
                        accumulator = accumulator.wrapping_add(((byte >> 1) & 1) as u64);
                        unsafe {
                            out32
                                .add(idx + 1)
                                .write((accumulator as u32).wrapping_add(base32));
                        }
                        accumulator = accumulator.wrapping_add(((byte >> 2) & 1) as u64);
                        unsafe {
                            out32
                                .add(idx + 2)
                                .write((accumulator as u32).wrapping_add(base32));
                        }
                        accumulator = accumulator.wrapping_add(((byte >> 3) & 1) as u64);
                        unsafe {
                            out32
                                .add(idx + 3)
                                .write((accumulator as u32).wrapping_add(base32));
                        }
                        accumulator = accumulator.wrapping_add(((byte >> 4) & 1) as u64);
                        unsafe {
                            out32
                                .add(idx + 4)
                                .write((accumulator as u32).wrapping_add(base32));
                        }
                        accumulator = accumulator.wrapping_add(((byte >> 5) & 1) as u64);
                        unsafe {
                            out32
                                .add(idx + 5)
                                .write((accumulator as u32).wrapping_add(base32));
                        }
                        accumulator = accumulator.wrapping_add(((byte >> 6) & 1) as u64);
                        unsafe {
                            out32
                                .add(idx + 6)
                                .write((accumulator as u32).wrapping_add(base32));
                        }
                        accumulator = accumulator.wrapping_add(((byte >> 7) & 1) as u64);
                        unsafe {
                            out32
                                .add(idx + 7)
                                .write((accumulator as u32).wrapping_add(base32));
                        }
                    }
                    for i in (fullBytes * 8)..row_count {
                        let delta =
                            unpack_inline(packed_ptr, packed_len, i as u64 * bw, bit_width, mask);
                        accumulator = accumulator.wrapping_add(delta);
                        unsafe {
                            out32
                                .add(i)
                                .write(accumulator.wrapping_add(base_value) as u32);
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
                        let v0 = accumulator.wrapping_add(base_value) as u32;
                        accumulator = accumulator.wrapping_add(d1);
                        let v1 = accumulator.wrapping_add(base_value) as u32;
                        accumulator = accumulator.wrapping_add(d2);
                        let v2 = accumulator.wrapping_add(base_value) as u32;
                        accumulator = accumulator.wrapping_add(d3);
                        let v3 = accumulator.wrapping_add(base_value) as u32;

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
                            out32
                                .add(i)
                                .write(accumulator.wrapping_add(base_value) as u32);
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

                        accumulator = accumulator.wrapping_add((byte & 1) as u64);
                        unsafe {
                            out64.add(idx).write(accumulator.wrapping_add(base_value));
                        }
                        accumulator = accumulator.wrapping_add(((byte >> 1) & 1) as u64);
                        unsafe {
                            out64
                                .add(idx + 1)
                                .write(accumulator.wrapping_add(base_value));
                        }
                        accumulator = accumulator.wrapping_add(((byte >> 2) & 1) as u64);
                        unsafe {
                            out64
                                .add(idx + 2)
                                .write(accumulator.wrapping_add(base_value));
                        }
                        accumulator = accumulator.wrapping_add(((byte >> 3) & 1) as u64);
                        unsafe {
                            out64
                                .add(idx + 3)
                                .write(accumulator.wrapping_add(base_value));
                        }
                        accumulator = accumulator.wrapping_add(((byte >> 4) & 1) as u64);
                        unsafe {
                            out64
                                .add(idx + 4)
                                .write(accumulator.wrapping_add(base_value));
                        }
                        accumulator = accumulator.wrapping_add(((byte >> 5) & 1) as u64);
                        unsafe {
                            out64
                                .add(idx + 5)
                                .write(accumulator.wrapping_add(base_value));
                        }
                        accumulator = accumulator.wrapping_add(((byte >> 6) & 1) as u64);
                        unsafe {
                            out64
                                .add(idx + 6)
                                .write(accumulator.wrapping_add(base_value));
                        }
                        accumulator = accumulator.wrapping_add(((byte >> 7) & 1) as u64);
                        unsafe {
                            out64
                                .add(idx + 7)
                                .write(accumulator.wrapping_add(base_value));
                        }
                    }
                    for i in (fullBytes * 8)..row_count {
                        let delta =
                            unpack_inline(packed_ptr, packed_len, i as u64 * bw, bit_width, mask);
                        accumulator = accumulator.wrapping_add(delta);
                        unsafe {
                            out64.add(i).write(accumulator.wrapping_add(base_value));
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
                            out64.add(i0).write(accumulator.wrapping_add(base_value));
                        }
                        accumulator = accumulator.wrapping_add(d1);
                        unsafe {
                            out64
                                .add(i0 + 1)
                                .write(accumulator.wrapping_add(base_value));
                        }
                        accumulator = accumulator.wrapping_add(d2);
                        unsafe {
                            out64
                                .add(i0 + 2)
                                .write(accumulator.wrapping_add(base_value));
                        }
                        accumulator = accumulator.wrapping_add(d3);
                        unsafe {
                            out64
                                .add(i0 + 3)
                                .write(accumulator.wrapping_add(base_value));
                        }
                    }
                    for i in (chunks * 4)..row_count {
                        let delta =
                            unpack_inline(packed_ptr, packed_len, i as u64 * bw, bit_width, mask);
                        accumulator = accumulator.wrapping_add(delta);
                        unsafe {
                            out64.add(i).write(accumulator.wrapping_add(base_value));
                        }
                    }
                }
                _ => {
                    for i in 0..row_count {
                        let delta =
                            unpack_inline(packed_ptr, packed_len, i as u64 * bw, bit_width, mask);
                        accumulator = accumulator.wrapping_add(delta);
                        let val = accumulator.wrapping_add(base_value).to_le_bytes();
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
                            out32.add(i0).write(r0.wrapping_add(base_value) as u32);
                            out32.add(i0 + 1).write(r1.wrapping_add(base_value) as u32);
                            out32.add(i0 + 2).write(r2.wrapping_add(base_value) as u32);
                            out32.add(i0 + 3).write(r3.wrapping_add(base_value) as u32);
                        }
                    }
                    for i in (chunks * 4)..row_count {
                        let r =
                            unpack_inline(packed_ptr, packed_len, i as u64 * bw, bit_width, mask);
                        unsafe {
                            out32.add(i).write(r.wrapping_add(base_value) as u32);
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
                            out64.add(i0).write(r0.wrapping_add(base_value));
                            out64.add(i0 + 1).write(r1.wrapping_add(base_value));
                            out64.add(i0 + 2).write(r2.wrapping_add(base_value));
                            out64.add(i0 + 3).write(r3.wrapping_add(base_value));
                        }
                    }
                    for i in (chunks * 4)..row_count {
                        let r =
                            unpack_inline(packed_ptr, packed_len, i as u64 * bw, bit_width, mask);
                        unsafe {
                            out64.add(i).write(r.wrapping_add(base_value));
                        }
                    }
                }
                _ => {
                    for i in 0..row_count {
                        let r =
                            unpack_inline(packed_ptr, packed_len, i as u64 * bw, bit_width, mask);
                        let val = r.wrapping_add(base_value).to_le_bytes();
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
        value_size: usize,
        predicate: &Predicate,
    ) -> Result<Vec<u8>> {
        if row_count == 0 {
            return Ok(Vec::new());
        }

        if value_size == 16 {
            return eval_predicate_wide(encoded, row_count, predicate);
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

        // Constant-step closed form: value[i] = first + i*step. A Range
        // predicate is answered analytically in O(1) (segment skip/accept or a
        // single contiguous matching index range), no decode at all. This is
        // the time-series fast path: a periodic series range query never
        // materializes a row.
        if flags & FLAG_CONST_STEP != 0
            && let Predicate::Range { low, high } = predicate
            && row_count >= 1
            && encoded.len() >= 20
        {
            let first = base_value;
            let step = u64::from_le_bytes([
                encoded[12],
                encoded[13],
                encoded[14],
                encoded[15],
                encoded[16],
                encoded[17],
                encoded[18],
                encoded[19],
            ]);
            // O(1)-safe only for a strictly ascending, non-overflowing series.
            let span_ok = (row_count as u64)
                .checked_sub(1)
                .and_then(|m| m.checked_mul(step))
                .and_then(|d| first.checked_add(d));
            if step != 0
                && step < (1u64 << 63)
                && let Some(last) = span_ok
            {
                let lo = match low {
                    Some(b) => read_u64_le(b, 0, b.len().min(8)),
                    None => 0,
                };
                let hi = match high {
                    Some(b) => read_u64_le(b, 0, b.len().min(8)),
                    None => u64::MAX,
                };
                let bml = row_count.div_ceil(8);
                if first > hi || last < lo {
                    return Ok(vec![0u8; bml]);
                }
                if first >= lo && last <= hi {
                    let mut bm = vec![0xFFu8; bml];
                    let trailing = row_count % 8;
                    if trailing != 0 {
                        bm[bml - 1] = (1u8 << trailing) - 1;
                    }
                    return Ok(bm);
                }
                let lo_start = if lo <= first {
                    0
                } else {
                    (lo - first).div_ceil(step) as usize
                };
                let hi_end = if hi >= last {
                    row_count
                } else {
                    ((hi - first) / step + 1) as usize
                }
                .min(row_count);
                let mut bm = vec![0u8; bml];
                if lo_start < hi_end {
                    fill_bitmask_range(&mut bm, lo_start, hi_end);
                }
                return Ok(bm);
            }
        }

        // Delta-of-delta, constant-step, PFOR, mini-block and the scale wrapper
        // pack a different layout than FoR/DELTA, so the encoded-domain fast
        // paths below do not apply. Decode (correct O(n)) then evaluate.
        if flags & (FLAG_DELTA_OF_DELTA | FLAG_CONST_STEP | FLAG_PFOR | FLAG_SCALE | FLAG_MINIBLOCK)
            != 0
        {
            let decoded = self.decode(encoded, row_count, value_size)?;
            return crate::encoding::eval_predicate_on_raw(
                &decoded, row_count, value_size, predicate,
            );
        }

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
                        let bitmaskLen = row_count.div_ceil(8);
                        return Ok(vec![0u8; bitmaskLen]);
                    }

                    // Segment-level accept: entire range within bounds
                    if loVal <= base_value && hiVal >= maxRepresentable {
                        let bitmaskLen = row_count.div_ceil(8);
                        let mut bitmask = vec![0xFFu8; bitmaskLen];
                        let trailing = row_count % 8;
                        if trailing != 0 {
                            bitmask[bitmaskLen - 1] = (1u8 << trailing) - 1;
                        }
                        return Ok(bitmask);
                    }

                    // Row-level filtering on residuals
                    let loResidual = loVal.saturating_sub(base_value);
                    let hiResidual = if hiVal >= base_value {
                        (hiVal - base_value).min(maxResidual)
                    } else {
                        return Ok(vec![0u8; row_count.div_ceil(8)]);
                    };

                    let bitmaskLen = row_count.div_ceil(8);
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
                    let bitmaskLen = row_count.div_ceil(8);
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
                    let bitmaskLen = row_count.div_ceil(8);
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
        let bitmaskLen = row_count.div_ceil(8);
        let mut bitmask = vec![0u8; bitmaskLen];

        // For Range predicates, try the constant-step fast path first.
        // Delta-encoded sequential data has packed values [r0, d, d, d, ...]
        // where r0 is the first FoR-subtracted value and d is the constant step.
        // After prefix sum: value[i] = base + r0 + i*d for i > 0, value[0] = base + r0.
        // This gives O(1) range computation instead of O(N) unpack + prefix sum.
        if let Predicate::Range { low, high } = predicate
            && row_count >= 2
        {
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
                    diff.div_ceil(step) as usize
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
                let loResidual = loVal.saturating_sub(base_value);
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
                    let v = residuals[i].wrapping_add(base_value);
                    if v >= loVal && v <= hiVal {
                        bitmask[i / 8] |= 1 << (i % 8);
                    }
                }
            }
            Predicate::Equality(target) => {
                let targetVal = read_u64_le(target, 0, target.len().min(8));
                for i in 0..row_count {
                    if residuals[i].wrapping_add(base_value) == targetVal {
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
                    let v = residuals[i].wrapping_add(base_value);
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
    if offset >= data.len() {
        return 0;
    }
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
    let bytes_needed = (total_bits as usize).div_ceil(8);

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

    for (i, val) in out.iter_mut().enumerate().take(count) {
        *val = unpack_inline(packed_ptr, packed_len, i as u64 * bw, bit_width, mask);
    }
}

/// Scalar reference unpack of one byte-aligned mini-block into `out[..len]`.
/// This is the authoritative implementation. SIMD paths must match it exactly.
#[inline]
fn unpack_block_scalar(packed: &[u8], bw: u8, len: usize, out: &mut [u64]) {
    let mask: u64 = if bw >= 64 { u64::MAX } else { (1u64 << bw) - 1 };
    let pp = packed.as_ptr();
    let pl = packed.len();
    for (j, slot) in out.iter_mut().enumerate().take(len) {
        *slot = unpack_inline(pp, pl, j as u64 * bw as u64, bw, mask);
    }
}

/// AVX2 widening unpack for byte-multiple widths (8/16/32). For these widths
/// the packed value is a whole little-endian integer, so unpacking is a pure
/// zero-extend - provably identical to the scalar path, vectorized 4 lanes at
/// a time. Caller guarantees avx2 is available.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn unpack_block_bytemul_avx2(packed: &[u8], bw: u8, len: usize, out: &mut [u64]) {
    match bw {
        8 => {
            for (j, slot) in out.iter_mut().enumerate().take(len) {
                *slot = packed[j] as u64;
            }
        }
        16 => {
            for (j, slot) in out.iter_mut().enumerate().take(len) {
                *slot = u16::from_le_bytes([packed[2 * j], packed[2 * j + 1]]) as u64;
            }
        }
        32 => {
            for (j, slot) in out.iter_mut().enumerate().take(len) {
                let o = 4 * j;
                *slot = u32::from_le_bytes([packed[o], packed[o + 1], packed[o + 2], packed[o + 3]])
                    as u64;
            }
        }
        _ => unpack_block_scalar(packed, bw, len, out),
    }
}

/// Unpacks one byte-aligned mini-block. Uses the AVX2 widening path for
/// byte-multiple widths when the CPU supports it, otherwise the authoritative
/// scalar path. Output is bit-for-bit identical regardless of path.
#[inline]
fn unpack_block_into(packed: &[u8], bw: u8, len: usize, out: &mut [u64]) {
    #[cfg(target_arch = "x86_64")]
    {
        if matches!(bw, 8 | 16 | 32) && std::arch::is_x86_feature_detected!("avx2") {
            // Safety: avx2 was just feature-detected; bw is a byte multiple so
            // the byte-extent (len*bw/8) is within the block slice.
            unsafe { unpack_block_bytemul_avx2(packed, bw, len, out) };
            return;
        }
    }
    unpack_block_scalar(packed, bw, len, out);
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
        for b in &mut bitmask[(firstByte + 1)..lastByte] {
            *b = 0xFF;
        }
        for b in 0..=lastBit {
            bitmask[lastByte] |= 1 << b;
        }
    }
}

// ===========================================================================
// 128-bit path + delta-of-delta helpers
// ===========================================================================

#[inline]
fn unzigzag_i64(z: u64) -> i64 {
    ((z >> 1) as i64) ^ -((z & 1) as i64)
}

#[inline]
fn zigzag_i128(v: i128) -> u128 {
    ((v << 1) ^ (v >> 127)) as u128
}

#[inline]
fn unzigzag_i128(z: u128) -> i128 {
    ((z >> 1) as i128) ^ -((z & 1) as i128)
}

/// Writes the low `value_size` little-endian bytes of `val` at row `idx`.
#[inline]
fn write_le(out: &mut [u8], idx: usize, value_size: usize, val: u64) {
    let bytes = val.to_le_bytes();
    let start = idx * value_size;
    out[start..start + value_size].copy_from_slice(&bytes[..value_size]);
}

/// Adds `base` to each residual and stores it as a value_size-byte little-endian
/// integer through a typed pointer, so the loop vectorizes into SIMD stores
/// instead of the byte-wise write_le. value_size 4 and 8 are the hot column
/// widths; other widths fall back to write_le. write_unaligned is used because
/// the output buffer is only byte-aligned (on x86_64 it is the same store). The
/// caller guarantees out holds at least residuals.len()*value_size bytes.
#[inline]
fn write_residuals_add_base(out: &mut [u8], value_size: usize, residuals: &[u64], base: u64) {
    match value_size {
        4 => {
            let p = out.as_mut_ptr() as *mut u32;
            for (i, &r) in residuals.iter().enumerate() {
                unsafe { p.add(i).write_unaligned(r.wrapping_add(base) as u32) };
            }
        }
        8 => {
            let p = out.as_mut_ptr() as *mut u64;
            for (i, &r) in residuals.iter().enumerate() {
                unsafe { p.add(i).write_unaligned(r.wrapping_add(base)) };
            }
        }
        _ => {
            for (i, &r) in residuals.iter().enumerate() {
                write_le(out, i, value_size, r.wrapping_add(base));
            }
        }
    }
}

/// Reads a 16-byte little-endian u128 at byte `offset` (zero-padded if short).
#[inline]
fn read_u128_le(data: &[u8], offset: usize) -> u128 {
    let mut buf = [0u8; 16];
    let end = (offset + 16).min(data.len());
    if offset < end {
        let n = end - offset;
        buf[..n].copy_from_slice(&data[offset..end]);
    }
    u128::from_le_bytes(buf)
}

/// Reads up to 16 little-endian bytes from a predicate bound as a u128.
#[inline]
fn read_u128_bound(bytes: &[u8]) -> u128 {
    let mut buf = [0u8; 16];
    let n = bytes.len().min(16);
    buf[..n].copy_from_slice(&bytes[..n]);
    u128::from_le_bytes(buf)
}

/// Bit width (1..=128) needed to represent `max`.
#[inline]
fn bit_width_u128(max: u128) -> u8 {
    if max == 0 {
        1
    } else {
        (128 - max.leading_zeros()) as u8
    }
}

/// Packs the low `bit_width` bits of `value` at `bit_offset`.
fn pack_bits_128(packed: &mut [u8], bit_offset: u64, value: u128, bit_width: u8) {
    let mut bo = bit_offset;
    for i in 0..bit_width {
        if (value >> i) & 1 == 1 {
            let byte = (bo >> 3) as usize;
            if byte < packed.len() {
                packed[byte] |= 1 << (bo & 7);
            }
        }
        bo += 1;
    }
}

/// Unpacks `bit_width` bits at `bit_offset` into a u128.
fn unpack_bits_128(packed: &[u8], bit_offset: u64, bit_width: u8) -> u128 {
    let mut v: u128 = 0;
    let mut bo = bit_offset;
    for i in 0..bit_width {
        let byte = (bo >> 3) as usize;
        let bit = if byte < packed.len() {
            (packed[byte] >> (bo & 7)) & 1
        } else {
            0
        };
        v |= (bit as u128) << i;
        bo += 1;
    }
    v
}

#[inline]
fn gcd_u64(mut a: u64, mut b: u64) -> u64 {
    while b != 0 {
        let t = b;
        b = a % b;
        a = t;
    }
    a
}

#[inline]
fn gcd_u128(mut a: u128, mut b: u128) -> u128 {
    while b != 0 {
        let t = b;
        b = a % b;
        a = t;
    }
    a
}

/// GCD of all (value - base) residuals. 0 means every residual is 0.
fn gcd_residual_u64(values: &[u64], base: u64) -> u64 {
    let mut g = 0u64;
    for &v in values {
        g = gcd_u64(g, v - base);
        if g == 1 {
            break;
        }
    }
    g
}

/// GCD of all (value - base) residuals for the 16-byte path.
fn gcd_residual_u128(values: &[u128], base: u128) -> u128 {
    let mut g = 0u128;
    for &v in values {
        g = gcd_u128(g, v.wrapping_sub(base));
        if g == 1 {
            break;
        }
    }
    g
}

/// FoR + DELTA/delta-of-delta/constant-step/PFOR candidate selection for the
/// 8-byte-or-narrower path. Returns the smallest representation. Does not apply
/// the scale wrapper (that is layered by the caller).
fn encode_narrow_core(values: &[u64], row_count: usize) -> Vec<u8> {
    let sorted_count = values.windows(2).filter(|w| w[1] >= w[0]).count();
    let use_delta = row_count > 1 && sorted_count >= (row_count - 1) * 9 / 10;
    let base_value = values.iter().copied().min().unwrap_or(0);
    let mut residuals: Vec<u64> = values.iter().map(|v| v - base_value).collect();

    let flags = if use_delta {
        for i in (1..residuals.len()).rev() {
            residuals[i] = residuals[i].wrapping_sub(residuals[i - 1]);
        }
        FLAG_DELTA
    } else {
        0
    };

    let max_residual = residuals.iter().copied().max().unwrap_or(0);
    let bit_width = if max_residual == 0 {
        1
    } else {
        64 - max_residual.leading_zeros()
    } as u8;

    let packed_bytes = (row_count as u64 * bit_width as u64).div_ceil(8) as usize;
    let mut packed = vec![0u8; packed_bytes];
    for (i, &val) in residuals.iter().enumerate() {
        pack_bits(&mut packed, i as u64 * bit_width as u64, val, bit_width);
    }

    let mut out = Vec::with_capacity(12 + packed_bytes);
    out.extend_from_slice(&base_value.to_le_bytes()); // [0..8]
    out.push(bit_width); // [8]
    out.push(flags); // [9]
    out.extend_from_slice(&0u16.to_le_bytes()); // [10..12] reserved
    out.extend_from_slice(&packed);

    let mut best = out;
    if row_count >= 3
        && let Some(dod) = encode_dod_narrow(values, base_value, row_count)
        && dod.len() < best.len()
    {
        best = dod;
    }
    if row_count >= 2
        && let Some(cs) = encode_const_step_narrow(values, row_count)
        && cs.len() < best.len()
    {
        best = cs;
    }
    if row_count >= 2
        && let Some(pf) = encode_pfor_narrow(values, base_value, row_count)
        && pf.len() < best.len()
    {
        best = pf;
    }
    if row_count > MINIBLOCK_SIZE
        && let Some(mb) = encode_miniblock_narrow(values, base_value, row_count)
        && mb.len() < best.len()
    {
        best = mb;
    }
    best
}

/// Fixed mini-block length. A burst or one wide value only inflates its own
/// 1024-value block instead of the whole segment. Blocks are byte-aligned so a
/// future SIMD unpack can process one block at a time.
const MINIBLOCK_SIZE: usize = 1024;

/// FoR + per-mini-block bit width for the 8-byte-or-narrower path. Each block
/// of MINIBLOCK_SIZE residuals carries its own 1-byte width and is packed
/// byte-aligned. Returns None when it does not beat the single-width form.
fn encode_miniblock_narrow(values: &[u64], base: u64, row_count: usize) -> Option<Vec<u8>> {
    let residuals: Vec<u64> = values.iter().map(|v| v.wrapping_sub(base)).collect();
    let global_max = residuals.iter().copied().max().unwrap_or(0);
    let global_bw = if global_max == 0 {
        1u64
    } else {
        64 - global_max.leading_zeros() as u64
    };
    let global_size = 12 + (row_count as u64 * global_bw).div_ceil(8) as usize;

    let nblocks = row_count.div_ceil(MINIBLOCK_SIZE);
    // Per-block width + byte-aligned packed bytes.
    let mut widths = Vec::with_capacity(nblocks);
    let mut total = 12 + nblocks; // header + one width byte per block
    for b in 0..nblocks {
        let start = b * MINIBLOCK_SIZE;
        let end = (start + MINIBLOCK_SIZE).min(row_count);
        let bmax = residuals[start..end].iter().copied().max().unwrap_or(0);
        let bw = if bmax == 0 {
            1u8
        } else {
            (64 - bmax.leading_zeros()) as u8
        };
        widths.push(bw);
        total += ((end - start) as u64 * bw as u64).div_ceil(8) as usize;
    }
    if total >= global_size {
        return None;
    }

    let mut out = Vec::with_capacity(total);
    out.extend_from_slice(&base.to_le_bytes()); // [0..8]
    out.push(0); // [8] global bit_width unused
    out.push(FLAG_MINIBLOCK); // [9]
    out.extend_from_slice(&0u16.to_le_bytes()); // [10..12]
    for (b, &bw) in widths.iter().enumerate() {
        let start = b * MINIBLOCK_SIZE;
        let end = (start + MINIBLOCK_SIZE).min(row_count);
        out.push(bw);
        let block_bytes = ((end - start) as u64 * bw as u64).div_ceil(8) as usize;
        let mut packed = vec![0u8; block_bytes];
        for (j, &r) in residuals[start..end].iter().enumerate() {
            pack_bits(&mut packed, j as u64 * bw as u64, r, bw);
        }
        out.extend_from_slice(&packed);
    }
    Some(out)
}

/// Builds the delta-of-delta stream for the 8-byte-or-narrower path.
/// Returns None when any second difference does not fit the u64 packed stream
/// (caller then keeps the FoR/DELTA output).
fn encode_dod_narrow(values: &[u64], base: u64, row_count: usize) -> Option<Vec<u8>> {
    let mut stream = vec![0u64; row_count];
    let r0 = values[0].wrapping_sub(base);
    stream[0] = r0;

    let r1 = values[1].wrapping_sub(base);
    let d1 = (r1 as i128) - (r0 as i128);
    let zz1 = zigzag_i128(d1);
    if zz1 > u64::MAX as u128 {
        return None;
    }
    stream[1] = zz1 as u64;

    let mut prev_d = d1;
    let mut prev_r = r1;
    for (i, slot) in stream.iter_mut().enumerate().take(row_count).skip(2) {
        let r = values[i].wrapping_sub(base);
        let d = (r as i128) - (prev_r as i128);
        let dd = d - prev_d;
        let zz = zigzag_i128(dd);
        if zz > u64::MAX as u128 {
            return None;
        }
        *slot = zz as u64;
        prev_d = d;
        prev_r = r;
    }

    let max_packed = stream.iter().copied().max().unwrap_or(0);
    let bit_width = if max_packed == 0 {
        1
    } else {
        64 - max_packed.leading_zeros()
    } as u8;

    let packed_bytes = (row_count as u64 * bit_width as u64).div_ceil(8) as usize;
    let mut packed = vec![0u8; packed_bytes];
    for (i, &val) in stream.iter().enumerate() {
        pack_bits(&mut packed, i as u64 * bit_width as u64, val, bit_width);
    }

    let mut out = Vec::with_capacity(12 + packed_bytes);
    out.extend_from_slice(&base.to_le_bytes());
    out.push(bit_width);
    out.push(FLAG_DELTA_OF_DELTA);
    out.extend_from_slice(&0u16.to_le_bytes());
    out.extend_from_slice(&packed);
    Some(out)
}

/// Encodes 16-byte values via FoR plus the smallest of {plain, delta,
/// delta-of-delta}, writing the 24-byte wide header.
fn encode_wide(data: &[u8], row_count: usize) -> Result<Vec<u8>> {
    let mut values = Vec::with_capacity(row_count);
    for i in 0..row_count {
        values.push(read_u128_le(data, i * 16));
    }
    let base = values.iter().copied().min().unwrap_or(0);
    let mut best = encode_wide_core(&values, row_count);

    // Effective-resolution scale (A5): factor out a common gcd losslessly.
    // Layout: [base:u128][_][FLAG_SCALE][_][scale:u128][inner core blob].
    let g = gcd_residual_u128(&values, base);
    if g > 1 {
        let q: Vec<u128> = values.iter().map(|v| v.wrapping_sub(base) / g).collect();
        let inner = encode_wide_core(&q, row_count);
        if WIDE_HEADER_SIZE + 16 + inner.len() < best.len() {
            let mut scaled = vec![0u8; WIDE_HEADER_SIZE + 16 + inner.len()];
            scaled[0..16].copy_from_slice(&base.to_le_bytes());
            scaled[17] = FLAG_SCALE;
            scaled[WIDE_HEADER_SIZE..WIDE_HEADER_SIZE + 16].copy_from_slice(&g.to_le_bytes());
            scaled[WIDE_HEADER_SIZE + 16..].copy_from_slice(&inner);
            best = scaled;
        }
    }
    Ok(best)
}

/// FoR + DELTA/DoD/constant-step/PFOR selection for 16-byte values. The scale
/// wrapper is layered by encode_wide.
fn encode_wide_core(values: &[u128], row_count: usize) -> Vec<u8> {
    let base = values.iter().copied().min().unwrap_or(0);
    let residuals: Vec<u128> = values.iter().map(|v| v.wrapping_sub(base)).collect();

    // Candidate 0: plain FoR residuals.
    let plain_bw = bit_width_u128(residuals.iter().copied().max().unwrap_or(0));

    // Candidate 1: first-order delta (zigzag, residual[0] verbatim).
    let mut delta_stream = vec![0u128; row_count];
    delta_stream[0] = residuals[0];
    let mut delta_max = residuals[0];
    for i in 1..row_count {
        let d = (residuals[i] as i128).wrapping_sub(residuals[i - 1] as i128);
        let zz = zigzag_i128(d);
        delta_stream[i] = zz;
        delta_max = delta_max.max(zz);
    }
    let delta_bw = bit_width_u128(delta_max);

    // Candidate 2: delta-of-delta.
    let mut dod_stream = vec![0u128; row_count];
    let mut dod_bw = 0u8;
    if row_count >= 3 {
        dod_stream[0] = residuals[0];
        let d1 = (residuals[1] as i128).wrapping_sub(residuals[0] as i128);
        dod_stream[1] = zigzag_i128(d1);
        let mut dod_max = residuals[0].max(dod_stream[1]);
        let mut prev_d = d1;
        for i in 2..row_count {
            let d = (residuals[i] as i128).wrapping_sub(residuals[i - 1] as i128);
            let dd = d.wrapping_sub(prev_d);
            let zz = zigzag_i128(dd);
            dod_stream[i] = zz;
            dod_max = dod_max.max(zz);
            prev_d = d;
        }
        dod_bw = bit_width_u128(dod_max);
    }

    let plain_size = row_count as u64 * plain_bw as u64;
    let delta_size = row_count as u64 * delta_bw as u64;
    let dod_size = if row_count >= 3 {
        row_count as u64 * dod_bw as u64
    } else {
        u64::MAX
    };

    let (flags, bit_width, stream): (u8, u8, &[u128]) =
        if dod_size <= plain_size && dod_size <= delta_size {
            (FLAG_DELTA_OF_DELTA, dod_bw, &dod_stream)
        } else if delta_size <= plain_size {
            (FLAG_DELTA, delta_bw, &delta_stream)
        } else {
            (0, plain_bw, &residuals)
        };

    let packed_bytes = (row_count as u64 * bit_width as u64).div_ceil(8) as usize;
    let mut out = vec![0u8; WIDE_HEADER_SIZE + packed_bytes];
    out[0..16].copy_from_slice(&base.to_le_bytes());
    out[16] = bit_width;
    out[17] = flags;
    let packed = &mut out[WIDE_HEADER_SIZE..];
    for (i, &val) in stream.iter().enumerate() {
        pack_bits_128(packed, i as u64 * bit_width as u64, val, bit_width);
    }

    // Constant-step closed form: O(1) regardless of row count. Chosen when
    // applicable and smaller than the packed representation.
    if row_count >= 2
        && let Some(cs) = encode_const_step_wide(values, row_count)
        && cs.len() < out.len()
    {
        return cs;
    }
    // Patched FoR: wins for near-regular data with a few outliers.
    if row_count >= 2
        && let Some(pf) = encode_pfor_wide(values, base, row_count)
        && pf.len() < out.len()
    {
        return pf;
    }
    out
}

/// Builds the constant-step closed form for 16-byte values, or None if the
/// step is not constant. Layout: [first_value:u128][.. step:u128].
fn encode_const_step_wide(values: &[u128], row_count: usize) -> Option<Vec<u8>> {
    if row_count < 2 {
        return None;
    }
    let step = values[1].wrapping_sub(values[0]);
    for i in 2..row_count {
        if values[i].wrapping_sub(values[i - 1]) != step {
            return None;
        }
    }
    let mut out = vec![0u8; WIDE_HEADER_SIZE + 16];
    out[0..16].copy_from_slice(&values[0].to_le_bytes());
    out[17] = FLAG_CONST_STEP;
    out[WIDE_HEADER_SIZE..WIDE_HEADER_SIZE + 16].copy_from_slice(&step.to_le_bytes());
    Some(out)
}

/// Builds the constant-step closed form for the 8-byte-or-narrower path, or
/// None if the step is not constant. Layout: [first_value:u64][.. step:u64].
fn encode_const_step_narrow(values: &[u64], row_count: usize) -> Option<Vec<u8>> {
    if row_count < 2 {
        return None;
    }
    let step = values[1].wrapping_sub(values[0]);
    for i in 2..row_count {
        if values[i].wrapping_sub(values[i - 1]) != step {
            return None;
        }
    }
    let mut out = Vec::with_capacity(20);
    out.extend_from_slice(&values[0].to_le_bytes()); // [0..8] first value
    out.push(0); // [8] bit_width unused
    out.push(FLAG_CONST_STEP); // [9] flags
    out.extend_from_slice(&0u16.to_le_bytes()); // [10..12] reserved
    out.extend_from_slice(&step.to_le_bytes()); // [12..20] step
    Some(out)
}

/// Patched FoR for the 8-byte-or-narrower path. Picks the packed width that
/// minimizes total size (packed low bits + a 12-byte-per-exception table for
/// the values that exceed it). Returns None when no width beats plain FoR.
fn encode_pfor_narrow(values: &[u64], base: u64, row_count: usize) -> Option<Vec<u8>> {
    let residuals: Vec<u64> = values.iter().map(|v| v.wrapping_sub(base)).collect();
    let max_residual = residuals.iter().copied().max().unwrap_or(0);
    let full_bw: u32 = if max_residual == 0 {
        1
    } else {
        64 - max_residual.leading_zeros()
    };
    if full_bw <= 2 {
        return None;
    }

    let plain_size = 12 + (row_count as u64 * full_bw as u64).div_ceil(8) as usize;

    let mut best: Option<(u32, usize, usize)> = None; // (width, exc_count, total)
    for w in 1..full_bw {
        let mut exc = 0usize;
        for &r in &residuals {
            if r >> w != 0 {
                exc += 1;
            }
        }
        if exc > u16::MAX as usize {
            continue;
        }
        let total = 12 + exc * 12 + (row_count as u64 * w as u64).div_ceil(8) as usize;
        if best.map(|(_, _, t)| total < t).unwrap_or(true) {
            best = Some((w, exc, total));
        }
    }

    let (w, exc_count, total) = best?;
    if total >= plain_size {
        return None;
    }

    let w_mask: u64 = if w >= 64 { u64::MAX } else { (1u64 << w) - 1 };
    let packed_bytes = (row_count as u64 * w as u64).div_ceil(8) as usize;
    let mut out = Vec::with_capacity(12 + exc_count * 12 + packed_bytes);
    out.extend_from_slice(&base.to_le_bytes()); // [0..8]
    out.push(w as u8); // [8] packed width
    out.push(FLAG_PFOR); // [9]
    out.extend_from_slice(&(exc_count as u16).to_le_bytes()); // [10..12]
    for (i, &r) in residuals.iter().enumerate() {
        if r >> w != 0 {
            out.extend_from_slice(&(i as u32).to_le_bytes());
            out.extend_from_slice(&r.to_le_bytes());
        }
    }
    let mut packed = vec![0u8; packed_bytes];
    for (i, &r) in residuals.iter().enumerate() {
        pack_bits(&mut packed, i as u64 * w as u64, r & w_mask, w as u8);
    }
    out.extend_from_slice(&packed);
    Some(out)
}

/// Patched FoR for the 16-byte path. Exception entries are 20 bytes
/// (u32 position + u128 residual).
fn encode_pfor_wide(values: &[u128], base: u128, row_count: usize) -> Option<Vec<u8>> {
    let residuals: Vec<u128> = values.iter().map(|v| v.wrapping_sub(base)).collect();
    let full_bw = bit_width_u128(residuals.iter().copied().max().unwrap_or(0)) as u32;
    if full_bw <= 2 {
        return None;
    }
    let plain_size = WIDE_HEADER_SIZE + (row_count as u64 * full_bw as u64).div_ceil(8) as usize;

    let mut best: Option<(u32, usize, usize)> = None;
    for w in 1..full_bw {
        let mut exc = 0usize;
        for &r in &residuals {
            if r >> w != 0 {
                exc += 1;
            }
        }
        if exc > u16::MAX as usize {
            continue;
        }
        let total =
            WIDE_HEADER_SIZE + exc * 20 + (row_count as u64 * w as u64).div_ceil(8) as usize;
        if best.map(|(_, _, t)| total < t).unwrap_or(true) {
            best = Some((w, exc, total));
        }
    }
    let (w, exc_count, total) = best?;
    if total >= plain_size {
        return None;
    }

    let w_mask: u128 = if w >= 128 {
        u128::MAX
    } else {
        (1u128 << w) - 1
    };
    let packed_bytes = (row_count as u64 * w as u64).div_ceil(8) as usize;
    let mut out = vec![0u8; WIDE_HEADER_SIZE + exc_count * 20 + packed_bytes];
    out[0..16].copy_from_slice(&base.to_le_bytes());
    out[16] = w as u8;
    out[17] = FLAG_PFOR;
    out[18..20].copy_from_slice(&(exc_count as u16).to_le_bytes());
    let mut o = WIDE_HEADER_SIZE;
    for (i, &r) in residuals.iter().enumerate() {
        if r >> w != 0 {
            out[o..o + 4].copy_from_slice(&(i as u32).to_le_bytes());
            out[o + 4..o + 20].copy_from_slice(&r.to_le_bytes());
            o += 20;
        }
    }
    let packed = &mut out[WIDE_HEADER_SIZE + exc_count * 20..];
    for (i, &r) in residuals.iter().enumerate() {
        pack_bits_128(packed, i as u64 * w as u64, r & w_mask, w as u8);
    }
    Some(out)
}

/// Decodes the 16-byte wide format back to raw little-endian u128 values.
fn decode_wide(encoded: &[u8], row_count: usize) -> Result<Vec<u8>> {
    if encoded.len() < WIDE_HEADER_SIZE {
        return Err(ZyronError::DecodingFailed(
            "FastLanes wide header too short".to_string(),
        ));
    }
    let base = read_u128_le(encoded, 0);
    let bit_width = encoded[16];
    let flags = encoded[17];

    // Effective-resolution scale wrapper: recurse one level into the inner
    // core blob, then multiply back.
    if flags & FLAG_SCALE != 0 {
        if encoded.len() < WIDE_HEADER_SIZE + 16 {
            return Err(ZyronError::DecodingFailed(
                "FastLanes wide scale blob too short".to_string(),
            ));
        }
        let scale = read_u128_le(encoded, WIDE_HEADER_SIZE);
        let q_raw = decode_wide(&encoded[WIDE_HEADER_SIZE + 16..], row_count)?;
        let mut out = vec![0u8; row_count * 16];
        for i in 0..row_count {
            let q = read_u128_le(&q_raw, i * 16);
            let v = base.wrapping_add(q.wrapping_mul(scale));
            out[i * 16..i * 16 + 16].copy_from_slice(&v.to_le_bytes());
        }
        return Ok(out);
    }

    // Constant-step closed form has no packed bit array.
    if flags & FLAG_CONST_STEP != 0 {
        if encoded.len() < WIDE_HEADER_SIZE + 16 {
            return Err(ZyronError::DecodingFailed(
                "FastLanes wide constant-step blob too short".to_string(),
            ));
        }
        let first = base;
        let step = read_u128_le(encoded, WIDE_HEADER_SIZE);
        let mut out = vec![0u8; row_count * 16];
        for i in 0..row_count {
            let v = first.wrapping_add((i as u128).wrapping_mul(step));
            out[i * 16..i * 16 + 16].copy_from_slice(&v.to_le_bytes());
        }
        return Ok(out);
    }

    // Patched FoR: [hdr][exception table (20B each)][packed low-width residuals].
    if flags & FLAG_PFOR != 0 {
        let exc_count = u16::from_le_bytes([encoded[18], encoded[19]]) as usize;
        let table_off = WIDE_HEADER_SIZE;
        let table_bytes = exc_count * 20;
        if bit_width == 0 || bit_width > 128 || encoded.len() < table_off + table_bytes {
            return Err(ZyronError::DecodingFailed(
                "FastLanes wide PFOR blob malformed".to_string(),
            ));
        }
        let packed = &encoded[table_off + table_bytes..];
        let mut out = vec![0u8; row_count * 16];
        for i in 0..row_count {
            let r = unpack_bits_128(packed, i as u64 * bit_width as u64, bit_width);
            out[i * 16..i * 16 + 16].copy_from_slice(&r.wrapping_add(base).to_le_bytes());
        }
        for e in 0..exc_count {
            let o = table_off + e * 20;
            let pos =
                u32::from_le_bytes([encoded[o], encoded[o + 1], encoded[o + 2], encoded[o + 3]])
                    as usize;
            let resid = read_u128_le(encoded, o + 4);
            if pos < row_count {
                out[pos * 16..pos * 16 + 16]
                    .copy_from_slice(&resid.wrapping_add(base).to_le_bytes());
            }
        }
        return Ok(out);
    }

    if bit_width == 0 || bit_width > 128 {
        return Err(ZyronError::DecodingFailed(format!(
            "invalid FastLanes wide bit width: {}",
            bit_width
        )));
    }
    let packed = &encoded[WIDE_HEADER_SIZE..];

    let mut stream = vec![0u128; row_count];
    for (i, slot) in stream.iter_mut().enumerate() {
        *slot = unpack_bits_128(packed, i as u64 * bit_width as u64, bit_width);
    }

    let mut out = vec![0u8; row_count * 16];
    let write = |out: &mut [u8], i: usize, v: u128| {
        out[i * 16..i * 16 + 16].copy_from_slice(&v.to_le_bytes());
    };

    if flags & FLAG_DELTA_OF_DELTA != 0 {
        let mut residual = stream[0];
        write(&mut out, 0, residual.wrapping_add(base));
        if row_count > 1 {
            let mut delta = unzigzag_i128(stream[1]);
            residual = residual.wrapping_add(delta as u128);
            write(&mut out, 1, residual.wrapping_add(base));
            for (i, &s) in stream.iter().enumerate().take(row_count).skip(2) {
                let dd = unzigzag_i128(s);
                delta = delta.wrapping_add(dd);
                residual = residual.wrapping_add(delta as u128);
                write(&mut out, i, residual.wrapping_add(base));
            }
        }
    } else if flags & FLAG_DELTA != 0 {
        let mut residual = stream[0];
        write(&mut out, 0, residual.wrapping_add(base));
        for (i, &s) in stream.iter().enumerate().take(row_count).skip(1) {
            let d = unzigzag_i128(s);
            residual = residual.wrapping_add(d as u128);
            write(&mut out, i, residual.wrapping_add(base));
        }
    } else {
        for (i, &s) in stream.iter().enumerate().take(row_count) {
            write(&mut out, i, s.wrapping_add(base));
        }
    }
    Ok(out)
}

/// Evaluates a predicate on the 16-byte wide format. Decodes then compares
/// numerically as u128 (same unsigned-pattern semantics the 8-byte path uses).
fn eval_predicate_wide(encoded: &[u8], row_count: usize, predicate: &Predicate) -> Result<Vec<u8>> {
    let decoded = decode_wide(encoded, row_count)?;
    let bitmask_len = row_count.div_ceil(8);
    let mut bitmask = vec![0u8; bitmask_len];
    match predicate {
        Predicate::Range { low, high } => {
            let lo = match *low {
                Some(b) => read_u128_bound(b),
                None => 0,
            };
            let hi = match *high {
                Some(b) => read_u128_bound(b),
                None => u128::MAX,
            };
            for i in 0..row_count {
                let v = read_u128_le(&decoded, i * 16);
                if v >= lo && v <= hi {
                    bitmask[i / 8] |= 1 << (i % 8);
                }
            }
        }
        Predicate::Equality(target) => {
            let t = read_u128_bound(target);
            for i in 0..row_count {
                if read_u128_le(&decoded, i * 16) == t {
                    bitmask[i / 8] |= 1 << (i % 8);
                }
            }
        }
        Predicate::In(values) => {
            let targets: Vec<u128> = values.iter().map(|v| read_u128_bound(v)).collect();
            for i in 0..row_count {
                let v = read_u128_le(&decoded, i * 16);
                if targets.contains(&v) {
                    bitmask[i / 8] |= 1 << (i % 8);
                }
            }
        }
    }
    Ok(bitmask)
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
        // 100..200 is a constant-step sequence, so the closed form wins.
        assert_eq!(encoded[9] & FLAG_CONST_STEP, FLAG_CONST_STEP);
        // Closed form is O(1): header + one step, independent of row count.
        assert_eq!(encoded.len(), 20);

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

    #[test]
    fn test_simd_unpack_matches_scalar_differential_fuzz() {
        // The scalar block unpack is authoritative. unpack_block_into may take
        // an AVX2 path for byte-multiple widths; it must be bit-identical.
        // Deterministic LCG, no external rng dependency.
        let mut state: u64 = 0x9E37_79B9_7F4A_7C15;
        let mut next = || {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            state
        };
        for bw in 1u8..=40 {
            for &len in &[1usize, 7, 8, 16, 31, 64, 1000, 3000] {
                let block_bytes = (len as u64 * bw as u64).div_ceil(8) as usize + 8;
                let packed: Vec<u8> = (0..block_bytes).map(|_| (next() & 0xFF) as u8).collect();
                let mut a = vec![0u64; len];
                let mut b = vec![0u64; len];
                unpack_block_scalar(&packed, bw, len, &mut a);
                unpack_block_into(&packed, bw, len, &mut b);
                assert_eq!(a, b, "SIMD/scalar mismatch at bw={bw} len={len}");
            }
        }
    }

    #[test]
    fn test_const_step_narrow_analytic_predicate() {
        // A periodic series. The const-step Range predicate is answered
        // analytically (no decode) and must equal a full scan, including
        // skip, accept, and partial-range cases.
        let enc = FastLanesEncoding;
        let base = 1_000_000u64;
        let step = 250u64;
        let n = 5000usize;
        let values: Vec<u64> = (0..n as u64).map(|i| base + i * step).collect();
        let mut data = Vec::new();
        for v in &values {
            data.extend_from_slice(&v.to_le_bytes());
        }
        let encoded = enc.encode(&data, n, 8).unwrap();
        assert_eq!(encoded[9] & FLAG_CONST_STEP, FLAG_CONST_STEP);

        let cases: &[(u64, u64)] = &[
            (0, 10),                                  // skip (all above)
            (10_000_000, 99_000_000),                 // skip (all below)
            (0, u64::MAX),                            // accept all
            (base, base + (n as u64 - 1) * step),     // accept all exact
            (1_000_500, 1_002_000),                   // partial interior
            (1_000_000, 1_000_000),                   // single first
            (base + (n as u64 - 1) * step, u64::MAX), // single last
        ];
        for &(lo, hi) in cases {
            let lob = lo.to_le_bytes();
            let hib = hi.to_le_bytes();
            let bm = enc
                .eval_predicate(
                    &encoded,
                    n,
                    8,
                    &Predicate::Range {
                        low: Some(&lob),
                        high: Some(&hib),
                    },
                )
                .unwrap();
            for (i, v) in values.iter().enumerate() {
                let want = *v >= lo && *v <= hi;
                let got = bm[i / 8] & (1 << (i % 8)) != 0;
                assert_eq!(got, want, "lo={lo} hi={hi} row={i} val={v}");
            }
        }
    }

    #[test]
    fn test_miniblock_byte_multiple_widths_roundtrip() {
        // Exercises the AVX2 widening path (8/16/32-bit blocks) end to end.
        let enc = FastLanesEncoding;
        for spread in [200u64, 50_000, 3_000_000_000] {
            let mut values: Vec<u64> = Vec::with_capacity(4096);
            for i in 0..4096u64 {
                let block = i / 1024;
                values.push(if block == 2 {
                    spread + (i % 1024) // wider block forces per-block width
                } else {
                    1 + (i % 97)
                });
            }
            let mut data = Vec::new();
            for v in &values {
                data.extend_from_slice(&v.to_le_bytes());
            }
            let encoded = enc.encode(&data, values.len(), 8).unwrap();
            let decoded = enc.decode(&encoded, values.len(), 8).unwrap();
            let got: Vec<u64> = decoded
                .chunks_exact(8)
                .map(|c| u64::from_le_bytes(c.try_into().unwrap()))
                .collect();
            assert_eq!(got, values, "spread {spread}");
        }
    }

    #[test]
    fn test_miniblock_narrow_bursty() {
        let enc = FastLanesEncoding;
        // 4 blocks of 1024. Blocks 0,2,3 are small (~10 bits); block 1 is a
        // wide burst (~24 bits). A single global width would pay 24 bits for
        // all 4096 rows; per-block width pays it only for block 1. PFOR cannot
        // win here because the burst is a whole block (1024 exceptions).
        let mut values: Vec<u64> = Vec::with_capacity(4096);
        for i in 0..4096u64 {
            let block = i / 1024;
            if block == 1 {
                values.push(10_000_000 + (i % 1024)); // wide block
            } else {
                values.push(1000 + (i % 1024)); // narrow blocks
            }
        }
        let mut data = Vec::new();
        for v in &values {
            data.extend_from_slice(&v.to_le_bytes());
        }
        let encoded = enc.encode(&data, values.len(), 8).unwrap();
        assert_eq!(
            encoded[9] & FLAG_MINIBLOCK,
            FLAG_MINIBLOCK,
            "bursty data should select the per-mini-block layout"
        );
        let decoded = enc.decode(&encoded, values.len(), 8).unwrap();
        let got: Vec<u64> = decoded
            .chunks_exact(8)
            .map(|c| u64::from_le_bytes(c.try_into().unwrap()))
            .collect();
        assert_eq!(got, values);

        // Predicate over the mini-block layout must equal a full scan.
        let lo = 1500u64.to_le_bytes();
        let hi = 1800u64.to_le_bytes();
        let bm = enc
            .eval_predicate(
                &encoded,
                values.len(),
                8,
                &Predicate::Range {
                    low: Some(&lo),
                    high: Some(&hi),
                },
            )
            .unwrap();
        for (i, v) in values.iter().enumerate() {
            let want = *v >= 1500 && *v <= 1800;
            let got = bm[i / 8] & (1 << (i % 8)) != 0;
            assert_eq!(got, want, "row {i}");
        }
    }

    #[test]
    fn test_pfor_narrow_outliers() {
        let enc = FastLanesEncoding;
        // Near-regular small values with two large outliers.
        let mut values: Vec<u64> = (0..256u64).map(|i| 1_000_000 + i).collect();
        values[50] = 8_000_000;
        values[150] = 9_500_000;
        let mut data = Vec::new();
        for v in &values {
            data.extend_from_slice(&v.to_le_bytes());
        }
        let encoded = enc.encode(&data, values.len(), 8).unwrap();
        assert_eq!(encoded[9] & FLAG_PFOR, FLAG_PFOR);
        let decoded = enc.decode(&encoded, values.len(), 8).unwrap();
        let got: Vec<u64> = decoded
            .chunks_exact(8)
            .map(|c| u64::from_le_bytes(c.try_into().unwrap()))
            .collect();
        assert_eq!(got, values);

        let lo = 1_000_010u64.to_le_bytes();
        let hi = 1_000_020u64.to_le_bytes();
        let bm = enc
            .eval_predicate(
                &encoded,
                values.len(),
                8,
                &Predicate::Range {
                    low: Some(&lo),
                    high: Some(&hi),
                },
            )
            .unwrap();
        for (i, v) in values.iter().enumerate() {
            let want = *v >= 1_000_010 && *v <= 1_000_020;
            let got = bm[i / 8] & (1 << (i % 8)) != 0;
            assert_eq!(got, want, "row {i}");
        }
    }

    #[test]
    fn test_scale_narrow_lossless() {
        let enc = FastLanesEncoding;
        // Irregular but all residuals are multiples of 1000 (g = 1000): not
        // constant-step, not tight-delta, so the scale wrapper should win.
        let pat = [3u64, 17, 5, 91, 2, 44, 60, 8, 130, 19];
        let values: Vec<u64> = (0..400)
            .map(|i| 1_000_000 + 1000 * pat[i % pat.len()] * (1 + (i as u64 % 7)))
            .collect();
        let mut data = Vec::new();
        for v in &values {
            data.extend_from_slice(&v.to_le_bytes());
        }
        let encoded = enc.encode(&data, values.len(), 8).unwrap();
        assert_eq!(encoded[9] & FLAG_SCALE, FLAG_SCALE);
        let decoded = enc.decode(&encoded, values.len(), 8).unwrap();
        let got: Vec<u64> = decoded
            .chunks_exact(8)
            .map(|c| u64::from_le_bytes(c.try_into().unwrap()))
            .collect();
        assert_eq!(got, values);
    }

    #[test]
    fn test_scale_wide_us_in_ps_column() {
        let enc = FastLanesEncoding;
        // Microsecond-granular timestamps stored in a picosecond column:
        // every value is a multiple of 1_000_000 (g = 1e6). Irregular spacing
        // so it is not constant-step; scale must make it us-class size.
        let pat = [5i128, 9, 2, 40, 7, 13, 1, 88];
        let base: i128 = 1_700_000_000_000_000_000_000;
        let values: Vec<i128> = (0..500)
            .map(|i| base + 1_000_000 * pat[i % pat.len()] * (1 + (i as i128 % 5)))
            .collect();
        let data = pack_i128(&values);
        let encoded = enc.encode(&data, values.len(), 16).unwrap();
        assert_eq!(encoded[17] & FLAG_SCALE, FLAG_SCALE);
        // us-class: well under the raw 16 bytes/row.
        assert!(encoded.len() < data.len() / 3);
        let decoded = enc.decode(&encoded, values.len(), 16).unwrap();
        assert_eq!(unpack_i128(&decoded), values);

        // Predicate must still be correct through the scale wrapper.
        let lo = ((base + 1_000_000 * 10) as u128).to_le_bytes();
        let hi = ((base + 1_000_000 * 200) as u128).to_le_bytes();
        let bm = enc
            .eval_predicate(
                &encoded,
                values.len(),
                16,
                &Predicate::Range {
                    low: Some(&lo),
                    high: Some(&hi),
                },
            )
            .unwrap();
        for (i, v) in values.iter().enumerate() {
            let want = *v >= base + 1_000_000 * 10 && *v <= base + 1_000_000 * 200;
            let got = bm[i / 8] & (1 << (i % 8)) != 0;
            assert_eq!(got, want, "row {i}");
        }
    }

    #[test]
    fn test_pfor_wide_outliers() {
        let enc = FastLanesEncoding;
        let mut values: Vec<i128> = (0..300i128).map(|i| 5_000_000 + i).collect();
        values[77] = 900_000_000_000;
        values[201] = -400_000_000_000;
        let data = pack_i128(&values);
        let encoded = enc.encode(&data, values.len(), 16).unwrap();
        assert_eq!(encoded[17] & FLAG_PFOR, FLAG_PFOR);
        let decoded = enc.decode(&encoded, values.len(), 16).unwrap();
        assert_eq!(unpack_i128(&decoded), values);
    }

    fn pack_i128(values: &[i128]) -> Vec<u8> {
        let mut data = Vec::with_capacity(values.len() * 16);
        for v in values {
            data.extend_from_slice(&(*v as u128).to_le_bytes());
        }
        data
    }

    fn unpack_i128(bytes: &[u8]) -> Vec<i128> {
        bytes
            .chunks_exact(16)
            .map(|c| {
                let mut b = [0u8; 16];
                b.copy_from_slice(c);
                u128::from_le_bytes(b) as i128
            })
            .collect()
    }

    #[test]
    fn test_roundtrip_i128_random() {
        let enc = FastLanesEncoding;
        let values: Vec<i128> = vec![
            1000,
            -5000,
            i128::MAX / 2,
            0,
            -1,
            i128::MIN / 4,
            42,
            999_999_999_999_999,
        ];
        let data = pack_i128(&values);
        let encoded = enc.encode(&data, values.len(), 16).unwrap();
        let decoded = enc.decode(&encoded, values.len(), 16).unwrap();
        assert_eq!(unpack_i128(&decoded), values);
    }

    #[test]
    fn test_roundtrip_i128_sequential_delta() {
        let enc = FastLanesEncoding;
        // Microsecond-class timestamps promoted to picoseconds: regular series.
        let base: i128 = 1_700_000_000_000_000_000_000;
        let values: Vec<i128> = (0..512).map(|i| base + i as i128 * 1_000_000).collect();
        let data = pack_i128(&values);
        let encoded = enc.encode(&data, values.len(), 16).unwrap();
        // Constant-step series must collapse far below the raw 16 bytes/row.
        assert!(encoded.len() < data.len() / 4);
        let decoded = enc.decode(&encoded, values.len(), 16).unwrap();
        assert_eq!(unpack_i128(&decoded), values);
    }

    #[test]
    fn test_roundtrip_dod_i128() {
        let enc = FastLanesEncoding;
        // Quadratic series: first differences vary (so constant-step does NOT
        // apply) but second differences are constant, which is delta-of-delta's
        // domain.
        let values: Vec<i128> = (0..300)
            .map(|i| 5_000 + (i as i128) * (i as i128) * 3 + (i as i128) * 7)
            .collect();
        let data = pack_i128(&values);
        let encoded = enc.encode(&data, values.len(), 16).unwrap();
        assert_eq!(encoded[17] & FLAG_DELTA_OF_DELTA, FLAG_DELTA_OF_DELTA);
        let decoded = enc.decode(&encoded, values.len(), 16).unwrap();
        assert_eq!(unpack_i128(&decoded), values);
    }

    #[test]
    fn test_const_step_i128_closed_form() {
        let enc = FastLanesEncoding;
        // Picosecond timestamps on a fixed 1us cadence: closed form, O(1).
        let base: i128 = 1_700_000_000_000_000_000_000;
        let values: Vec<i128> = (0..100_000).map(|i| base + i as i128 * 1_000_000).collect();
        let data = pack_i128(&values);
        let encoded = enc.encode(&data, values.len(), 16).unwrap();
        assert_eq!(encoded[17] & FLAG_CONST_STEP, FLAG_CONST_STEP);
        assert_eq!(encoded.len(), WIDE_HEADER_SIZE + 16);
        let decoded = enc.decode(&encoded, values.len(), 16).unwrap();
        assert_eq!(unpack_i128(&decoded), values);

        // Predicate on the closed form must match a full scan.
        let lo = ((base + 10_000 * 1_000_000) as u128).to_le_bytes();
        let hi = ((base + 20_000 * 1_000_000) as u128).to_le_bytes();
        let bm = enc
            .eval_predicate(
                &encoded,
                values.len(),
                16,
                &Predicate::Range {
                    low: Some(&lo),
                    high: Some(&hi),
                },
            )
            .unwrap();
        for (i, v) in values.iter().enumerate() {
            let want = *v >= base + 10_000 * 1_000_000 && *v <= base + 20_000 * 1_000_000;
            let got = bm[i / 8] & (1 << (i % 8)) != 0;
            assert_eq!(got, want, "row {i}");
        }
    }

    #[test]
    fn test_roundtrip_dod_u64() {
        let enc = FastLanesEncoding;
        // Quadratic series: deltas grow linearly, second differences constant,
        // so delta-of-delta wins on the 8-byte path.
        let values: Vec<u64> = (0..400u64).map(|i| 1_000 + i * i).collect();
        let mut data = Vec::new();
        for v in &values {
            data.extend_from_slice(&v.to_le_bytes());
        }
        let encoded = enc.encode(&data, values.len(), 8).unwrap();
        assert_eq!(encoded[9] & FLAG_DELTA_OF_DELTA, FLAG_DELTA_OF_DELTA);
        let decoded = enc.decode(&encoded, values.len(), 8).unwrap();
        let got: Vec<u64> = decoded
            .chunks_exact(8)
            .map(|c| {
                let mut b = [0u8; 8];
                b.copy_from_slice(c);
                u64::from_le_bytes(b)
            })
            .collect();
        assert_eq!(got, values);
    }

    #[test]
    fn test_i128_negative_deltas() {
        let enc = FastLanesEncoding;
        // Descending across zero exercises zigzag on negative deltas.
        let values: Vec<i128> = (0..256).map(|i| 1_000_000 - i as i128 * 9_973).collect();
        let data = pack_i128(&values);
        let encoded = enc.encode(&data, values.len(), 16).unwrap();
        let decoded = enc.decode(&encoded, values.len(), 16).unwrap();
        assert_eq!(unpack_i128(&decoded), values);
    }

    #[test]
    fn test_i128_predicate_equiv_full_scan() {
        let enc = FastLanesEncoding;
        let values: Vec<i128> = (0..200).map(|i| 10_000 + i as i128 * 13).collect();
        let data = pack_i128(&values);
        let encoded = enc.encode(&data, values.len(), 16).unwrap();

        let lo = (10_500i128 as u128).to_le_bytes();
        let hi = (12_000i128 as u128).to_le_bytes();
        let bm = enc
            .eval_predicate(
                &encoded,
                values.len(),
                16,
                &Predicate::Range {
                    low: Some(&lo),
                    high: Some(&hi),
                },
            )
            .unwrap();
        for (i, v) in values.iter().enumerate() {
            let want = *v >= 10_500 && *v <= 12_000;
            let got = bm[i / 8] & (1 << (i % 8)) != 0;
            assert_eq!(got, want, "row {i} value {v}");
        }

        let target = (10_013i128 as u128).to_le_bytes();
        let bm = enc
            .eval_predicate(&encoded, values.len(), 16, &Predicate::Equality(&target))
            .unwrap();
        for (i, v) in values.iter().enumerate() {
            let got = bm[i / 8] & (1 << (i % 8)) != 0;
            assert_eq!(got, *v == 10_013, "eq row {i}");
        }
    }

    #[test]
    fn test_i128_empty_and_single() {
        let enc = FastLanesEncoding;
        let dec = enc.decode(&enc.encode(&[], 0, 16).unwrap(), 0, 16).unwrap();
        assert!(dec.is_empty());

        let one = pack_i128(&[-12345]);
        let encoded = enc.encode(&one, 1, 16).unwrap();
        let decoded = enc.decode(&encoded, 1, 16).unwrap();
        assert_eq!(unpack_i128(&decoded), vec![-12345]);
    }
}
