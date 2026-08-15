//! ALP (Adaptive Lossless Predictor) encoding for floating-point columns.
//!
//! Finds a (factor, exponent) pair such that multiplying each float produces
//! an integer. Encodes the integer representation with bit-packing. Values
//! that cannot be losslessly converted are stored as exceptions.
//!
//! Based on ALP (SIGMOD 2024), adapted for Zyron's columnar format.

use crate::encoding::{Encoding, EncodingType, Predicate, eval_predicate_on_raw};
use zyron_common::{Result, ZyronError};

pub struct AlpEncoding;

/// Maximum number of exceptions before falling back to raw encoding.
const MAX_EXCEPTION_RATIO: f64 = 0.1;

/// Delta encoding applied to the integer array.
const FLAG_DELTA: u8 = 0x01;
/// Periodic restart values ahead of the packed array, letting a range decode of
/// the delta stream seed at the nearest boundary instead of replaying the whole
/// prefix.
const FLAG_RESTART: u8 = 0x02;

/// Restart spacing floor, matching the columnar zone map batch size so every
/// restart point lands on a zone boundary.
const RESTART_MIN_SHIFT: u32 = 10;

/// Encoded format:
///   [0..8]    factor: f64 (multiply floats by this to get integers)
///   [8..12]   exponent: i32 (power of 10 used for factor)
///   [12..16]  exception_count: u32
///   [16]      int_bit_width: u8 (bits per integer in main array)
///   [17]      value_size_marker: u8 (4 = f32, 8 = f64)
///   [18]      flags: u8 (FLAG_DELTA / FLAG_RESTART)
///   [19]      restart shift when FLAG_RESTART is set
///   [20..28]  base: i64 (FoR base for unsigned shift)
///   [28..]    restart table when FLAG_RESTART is set, then the bit-packed
///             integer array (delta+FoR if the delta flag is set)
///   after it  exceptions: (row_index: u32 + original_value: value_size) per exception

/// Number of restart boundaries for a row count and spacing. Boundary k covers
/// row `k << shift` counted from 1.
#[inline]
fn restart_count(row_count: usize, shift: u32) -> usize {
    if row_count == 0 || shift >= usize::BITS {
        0
    } else {
        (row_count - 1) >> shift
    }
}

/// Widens the restart spacing until the table costs at most a thirty-second of
/// the packed array, keeping the spacing a power-of-two multiple of the floor.
fn choose_restart_shift(row_count: usize, packed_bytes: usize) -> u32 {
    let budget = (packed_bytes / 32).max(8);
    let mut shift = RESTART_MIN_SHIFT;
    while shift < 31 && restart_count(row_count, shift) * 8 > budget {
        shift += 1;
    }
    shift
}

/// Byte offset of the packed integer array, past any restart table.
#[inline]
fn packed_offset(encoded: &[u8], row_count: usize) -> usize {
    if encoded[18] & FLAG_RESTART == 0 {
        return 28;
    }
    28 + restart_count(row_count, encoded[19] as u32) * 8
}

/// Reads the restart entry covering `start`, returning the seeded running sum
/// and the row decoding resumes at. Falls back to the head of the stream when
/// the range starts before the first boundary or the table is absent.
fn seed_restart(table: &[u8], shift: u32, start: usize) -> (i64, usize) {
    if table.is_empty() || shift >= usize::BITS {
        return (0, 0);
    }
    let k = start >> shift;
    if k == 0 {
        return (0, 0);
    }
    let at = (k - 1) * 8;
    if at + 8 > table.len() {
        return (0, 0);
    }
    let value = i64::from_le_bytes([
        table[at],
        table[at + 1],
        table[at + 2],
        table[at + 3],
        table[at + 4],
        table[at + 5],
        table[at + 6],
        table[at + 7],
    ]);
    (value, k << shift)
}
impl Encoding for AlpEncoding {
    fn encoding_type(&self) -> EncodingType {
        EncodingType::Alp
    }

    fn encode(&self, data: &[u8], row_count: usize, value_size: usize) -> Result<Vec<u8>> {
        if row_count == 0 {
            let mut out = vec![0u8; 28];
            out[17] = value_size as u8;
            return Ok(out);
        }

        if value_size != 4 && value_size != 8 {
            return Err(ZyronError::EncodingFailed(
                "ALP encoding supports f32 (4 bytes) and f64 (8 bytes) only".to_string(),
            ));
        }

        if data.len() < row_count * value_size {
            return Err(ZyronError::EncodingFailed(
                "data shorter than expected for ALP encoding".to_string(),
            ));
        }

        // Read float values as f64 (widen f32 to f64 for uniform processing)
        let floats: Vec<f64> = (0..row_count)
            .map(|i| read_float(data, i, value_size))
            .collect();

        // Find the best (factor, exponent) pair
        let (factor, exponent) = find_best_factor(&floats);

        // Convert floats to integers, tracking exceptions
        let mut integers = Vec::with_capacity(row_count);
        let mut exceptions: Vec<(u32, f64)> = Vec::new();

        for (i, &f) in floats.iter().enumerate() {
            let scaled = f * factor;
            let rounded = scaled.round();

            // Check if conversion is lossless within floating-point tolerance.
            // Uses the same epsilon threshold as find_best_factor to avoid
            // mismatches between the 128-element sample and the full dataset.
            // NaN, infinity, negative zero, and subnormals are always exceptions
            // to preserve their exact bit patterns.
            if f.is_finite()
                && f.to_bits() != 0x8000_0000_0000_0000 // negative zero
                && rounded.is_finite()
                && rounded >= i64::MIN as f64
                && rounded <= i64::MAX as f64
            {
                let int_val = rounded as i64;
                let recovered = int_val as f64 / factor;

                if value_size == 4 {
                    let orig_f32 = f as f32;
                    let recovered_f32 = recovered as f32;
                    if orig_f32.is_finite()
                        && orig_f32.to_bits() != 0x8000_0000 // negative zero f32
                        && (orig_f32 - recovered_f32).abs() <= f32::EPSILON * orig_f32.abs().max(1.0)
                    {
                        integers.push(int_val);
                        continue;
                    }
                } else if (recovered - f).abs() <= f64::EPSILON * f.abs().max(1.0) {
                    integers.push(int_val);
                    continue;
                }
            }

            // Exception: store original float, use 0 as placeholder integer
            exceptions.push((i as u32, f));
            integers.push(0);
        }

        // If too many exceptions, fall back to storing raw
        if exceptions.len() as f64 > row_count as f64 * MAX_EXCEPTION_RATIO && row_count > 10 {
            // Pack as raw with ALP header indicating no compression
            return encode_raw_fallback(data, row_count, value_size);
        }

        // Delta encoding: only when no exceptions (exceptions use placeholder 0s
        // which would corrupt the delta sequence). Check if integers are mostly monotonic.
        let sortedCount = if integers.len() > 1 {
            integers.windows(2).filter(|w| w[1] >= w[0]).count()
        } else {
            0
        };
        let useDelta = exceptions.is_empty()
            && integers.len() > 1
            && sortedCount >= (integers.len() - 1) * 9 / 10;

        // Apply delta encoding if beneficial (reduces bit width for sequential data)
        let mut workingIntegers = integers.clone();
        if useDelta {
            for i in (1..workingIntegers.len()).rev() {
                workingIntegers[i] = workingIntegers[i].wrapping_sub(workingIntegers[i - 1]);
            }
        }

        // Determine bit width for integer array
        let (minInt, maxInt) = workingIntegers
            .iter()
            .fold((i64::MAX, i64::MIN), |(mn, mx), &v| (mn.min(v), mx.max(v)));

        // Shift to unsigned for bit-packing
        let base = minInt;
        let maxUnsigned = (maxInt.wrapping_sub(base)) as u64;
        let intBitWidth = if maxUnsigned == 0 {
            1
        } else {
            64 - maxUnsigned.leading_zeros()
        } as u8;

        // Bit-pack the integers
        let packedBits = row_count as u64 * intBitWidth as u64;
        let packedBytes = (packedBits as usize).div_ceil(8);
        let mut packedMain = vec![0u8; packedBytes];

        for (i, &intVal) in workingIntegers.iter().enumerate() {
            let unsigned = (intVal.wrapping_sub(base)) as u64;
            pack_bits(
                &mut packedMain,
                i as u64 * intBitWidth as u64,
                unsigned,
                intBitWidth,
            );
        }

        // Restart values for the cumulative delta stream. The running sum after
        // row i is the undifferenced integer at row i, so the entry for
        // boundary k is the integer at the row before it
        let restartShift = choose_restart_shift(row_count, packedBytes);
        let restarts = if useDelta {
            restart_count(row_count, restartShift)
        } else {
            0
        };

        // Build output
        let exceptionEntrySize = 4 + value_size; // row_index + original_value
        let totalSize = 28 + restarts * 8 + packedBytes + exceptions.len() * exceptionEntrySize;
        let mut out = Vec::with_capacity(totalSize);

        let mut flags: u8 = if useDelta { FLAG_DELTA } else { 0x00 };
        if restarts > 0 {
            flags |= FLAG_RESTART;
        }

        out.extend_from_slice(&factor.to_le_bytes()); // [0..8]
        out.extend_from_slice(&exponent.to_le_bytes()); // [8..12]
        out.extend_from_slice(&(exceptions.len() as u32).to_le_bytes()); // [12..16]
        out.push(intBitWidth); // [16]
        out.push(value_size as u8); // [17]
        out.push(flags); // [18]
        out.push(restartShift as u8); // [19] restart spacing
        out.extend_from_slice(&base.to_le_bytes()); // [20..28] base for unsigned shift
        for k in 1..=restarts {
            out.extend_from_slice(&integers[(k << restartShift) - 1].to_le_bytes());
        }
        out.extend_from_slice(&packedMain); // packed integers

        // Exceptions
        for &(row_idx, val) in &exceptions {
            out.extend_from_slice(&row_idx.to_le_bytes());
            if value_size == 4 {
                out.extend_from_slice(&(val as f32).to_le_bytes());
            } else {
                out.extend_from_slice(&val.to_le_bytes());
            }
        }

        Ok(out)
    }

    /// Non-delta ALP addresses a row directly: the main array packs every
    /// integer to the same width, so row i is at bit offset
    /// `i * int_bit_width`, and the exception list carries its own row
    /// indices so only the ones inside the range are applied.
    ///
    /// Delta ALP is cumulative, row i being defined against row i-1, so it
    /// seeds its running sum from the restart value at or before `start` and
    /// replays at most one restart spacing rather than the whole prefix.
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
        let take_default = |this: &Self| -> Result<Vec<u8>> {
            let decoded = this.decode(encoded, row_count, value_size)?;
            crate::encoding::slice_decoded(&decoded, row_count, value_size, start, end)
        };
        if encoded.len() < 28 {
            return take_default(self);
        }
        let factor = f64::from_le_bytes([
            encoded[0], encoded[1], encoded[2], encoded[3], encoded[4], encoded[5], encoded[6],
            encoded[7],
        ]);
        let int_bit_width = encoded[16];
        let stored_value_size = encoded[17] as usize;
        let flags = encoded[18];
        // A raw fallback carries no packed array
        if factor == 0.0 || stored_value_size != value_size {
            return take_default(self);
        }
        if int_bit_width == 0 || int_bit_width > 64 {
            return take_default(self);
        }
        let exception_count =
            u32::from_le_bytes([encoded[12], encoded[13], encoded[14], encoded[15]]) as usize;
        let base = i64::from_le_bytes([
            encoded[20],
            encoded[21],
            encoded[22],
            encoded[23],
            encoded[24],
            encoded[25],
            encoded[26],
            encoded[27],
        ]);
        let packed_start = packed_offset(encoded, row_count);
        let packed_bytes = ((row_count as u64 * int_bit_width as u64) as usize).div_ceil(8);
        let packed_end = packed_start + packed_bytes;
        if encoded.len() < packed_end {
            return take_default(self);
        }
        let packed = &encoded[packed_start..packed_end];
        let mask: u64 = if int_bit_width >= 64 {
            u64::MAX
        } else {
            (1u64 << int_bit_width) - 1
        };
        let inv_factor = 1.0 / factor;
        let taken = end - start;
        let mut out = vec![0u8; taken * value_size];
        let write_value = |out: &mut [u8], slot: usize, int_val: i64| {
            let value = int_val as f64 * inv_factor;
            let at = &mut out[slot * value_size..(slot + 1) * value_size];
            if value_size == 4 {
                at.copy_from_slice(&(value as f32).to_le_bytes());
            } else {
                at.copy_from_slice(&value.to_le_bytes());
            }
        };

        if flags & FLAG_DELTA != 0 {
            // The running sum after row i is the undifferenced integer at row
            // i, so a restart entry seeds it directly
            let table = &encoded[28..packed_start];
            let shift = encoded[19] as u32;
            let (mut running, mut row) = seed_restart(table, shift, start);
            while row < start {
                let raw =
                    unpack_bits(packed, row as u64 * int_bit_width as u64, int_bit_width) & mask;
                running = running.wrapping_add(raw as i64 + base);
                row += 1;
            }
            while row < end {
                let raw =
                    unpack_bits(packed, row as u64 * int_bit_width as u64, int_bit_width) & mask;
                running = running.wrapping_add(raw as i64 + base);
                write_value(&mut out, row - start, running);
                row += 1;
            }
            // Delta is only chosen when the column has no exceptions, so the
            // range is complete once the stream is replayed
            return Ok(out);
        }

        for i in 0..taken {
            let row = start + i;
            let raw = unpack_bits(packed, row as u64 * int_bit_width as u64, int_bit_width) & mask;
            write_value(&mut out, i, (raw as i64).wrapping_add(base));
        }

        // Exceptions carry their own row index, so only the ones landing in
        // the range are read back and the rest are stepped over
        let exceptions = &encoded[packed_end..];
        let entry_size = 4 + value_size;
        for e in 0..exception_count {
            let at = e * entry_size;
            if at + entry_size > exceptions.len() {
                return take_default(self);
            }
            let row = u32::from_le_bytes([
                exceptions[at],
                exceptions[at + 1],
                exceptions[at + 2],
                exceptions[at + 3],
            ]) as usize;
            if row < start || row >= end {
                continue;
            }
            let payload = &exceptions[at + 4..at + 4 + value_size];
            out[(row - start) * value_size..(row - start + 1) * value_size]
                .copy_from_slice(payload);
        }
        Ok(out)
    }

    fn decode(&self, encoded: &[u8], row_count: usize, value_size: usize) -> Result<Vec<u8>> {
        if row_count == 0 {
            return Ok(Vec::new());
        }

        if encoded.len() < 28 {
            return Err(ZyronError::DecodingFailed(
                "ALP header too short".to_string(),
            ));
        }

        let factor = f64::from_le_bytes([
            encoded[0], encoded[1], encoded[2], encoded[3], encoded[4], encoded[5], encoded[6],
            encoded[7],
        ]);

        let exception_count =
            u32::from_le_bytes([encoded[12], encoded[13], encoded[14], encoded[15]]) as usize;

        let intBitWidth = encoded[16];
        let storedValueSize = encoded[17] as usize;

        if storedValueSize != value_size {
            return Err(ZyronError::DecodingFailed(format!(
                "ALP value_size mismatch: stored {}, expected {}",
                storedValueSize, value_size
            )));
        }

        // Check for raw fallback (factor == 0.0)
        if factor == 0.0 {
            return decode_raw_fallback(encoded, row_count, value_size);
        }

        let flags = encoded[18];
        let useDelta = flags & FLAG_DELTA != 0;

        let base = i64::from_le_bytes([
            encoded[20],
            encoded[21],
            encoded[22],
            encoded[23],
            encoded[24],
            encoded[25],
            encoded[26],
            encoded[27],
        ]);

        let packedStart = packed_offset(encoded, row_count);
        let packedBits = row_count as u64 * intBitWidth as u64;
        let packedBytes = (packedBits as usize).div_ceil(8);
        let packedEnd = packedStart + packedBytes;

        if encoded.len() < packedEnd {
            return Err(ZyronError::DecodingFailed(
                "ALP packed data truncated".to_string(),
            ));
        }

        let packed = &encoded[packedStart..packedEnd];

        // Reciprocal multiplication instead of division per value.
        // f64 division: ~15-20 cycles. f64 multiplication: ~4-5 cycles.
        let invFactor = 1.0 / factor;

        // Single-pass decode: unpack, reverse delta in-place, convert to float,
        // and write output directly. Eliminates intermediate Vec<i64> allocation.
        let outLen = row_count * value_size;
        // SAFETY: every byte of `out` is written by the decode below before
        // any read; zeroing first would memset the whole buffer only to
        // overwrite it, regressing decode throughput on the scan path.
        #[allow(clippy::uninit_vec)]
        let mut out: Vec<u8> = {
            let mut v = Vec::with_capacity(outLen);
            unsafe { v.set_len(outLen) };
            v
        };
        let outPtr = out.as_mut_ptr();
        let packedPtr = packed.as_ptr();
        let packedLen = packed.len();

        let mask: u64 = if intBitWidth >= 64 {
            u64::MAX
        } else {
            (1u64 << intBitWidth) - 1
        };
        let bw = intBitWidth as u64;
        let mut prevInt: i64 = 0;

        if value_size == 8 && useDelta && intBitWidth == 1 {
            // bit_width=1 delta f64: extract 8 deltas per packed byte.
            // Single-pass: prefix-sum + float convert + write.
            // On modern out-of-order CPUs, the float conversion pipelines
            // behind the integer add without stalling the prefix-sum chain.
            let out64f = outPtr as *mut f64;

            if row_count > 0 {
                let byte0 = unsafe { *packedPtr.add(0) };
                prevInt = (byte0 & 1) as i64 + base;
                unsafe {
                    out64f.add(0).write(prevInt as f64 * invFactor);
                }

                let firstByteTail = 8.min(row_count);
                for bit in 1..firstByteTail {
                    prevInt = prevInt.wrapping_add(((byte0 >> bit) & 1) as i64 + base);
                    unsafe {
                        out64f.add(bit).write(prevInt as f64 * invFactor);
                    }
                }
            }

            let fullBytes = row_count / 8;
            for b in 1..fullBytes {
                let byte = unsafe { *packedPtr.add(b) };
                let idx = b * 8;

                prevInt = prevInt.wrapping_add((byte & 1) as i64 + base);
                unsafe {
                    out64f.add(idx).write(prevInt as f64 * invFactor);
                }
                prevInt = prevInt.wrapping_add(((byte >> 1) & 1) as i64 + base);
                unsafe {
                    out64f.add(idx + 1).write(prevInt as f64 * invFactor);
                }
                prevInt = prevInt.wrapping_add(((byte >> 2) & 1) as i64 + base);
                unsafe {
                    out64f.add(idx + 2).write(prevInt as f64 * invFactor);
                }
                prevInt = prevInt.wrapping_add(((byte >> 3) & 1) as i64 + base);
                unsafe {
                    out64f.add(idx + 3).write(prevInt as f64 * invFactor);
                }
                prevInt = prevInt.wrapping_add(((byte >> 4) & 1) as i64 + base);
                unsafe {
                    out64f.add(idx + 4).write(prevInt as f64 * invFactor);
                }
                prevInt = prevInt.wrapping_add(((byte >> 5) & 1) as i64 + base);
                unsafe {
                    out64f.add(idx + 5).write(prevInt as f64 * invFactor);
                }
                prevInt = prevInt.wrapping_add(((byte >> 6) & 1) as i64 + base);
                unsafe {
                    out64f.add(idx + 6).write(prevInt as f64 * invFactor);
                }
                prevInt = prevInt.wrapping_add(((byte >> 7) & 1) as i64 + base);
                unsafe {
                    out64f.add(idx + 7).write(prevInt as f64 * invFactor);
                }
            }
            for i in (fullBytes * 8).max(8.min(row_count))..row_count {
                let unsigned =
                    unpack_alp_inline(packedPtr, packedLen, i as u64 * bw, intBitWidth, mask);
                prevInt = prevInt.wrapping_add(unsigned as i64 + base);
                unsafe {
                    out64f.add(i).write(prevInt as f64 * invFactor);
                }
            }
        } else if value_size == 8 && useDelta {
            // Generic delta f64: unpack + prefix-sum + float convert in one pass.
            let out64f = outPtr as *mut f64;

            if row_count > 0 {
                let u0 = unpack_alp_inline(packedPtr, packedLen, 0, intBitWidth, mask);
                prevInt = u0 as i64 + base;
                unsafe {
                    out64f.add(0).write(prevInt as f64 * invFactor);
                }
            }

            let remaining = row_count - 1;
            let chunks = remaining / 4;
            for chunk in 0..chunks {
                let i0 = chunk * 4 + 1;
                let u0 = unpack_alp_inline(packedPtr, packedLen, i0 as u64 * bw, intBitWidth, mask);
                let u1 = unpack_alp_inline(
                    packedPtr,
                    packedLen,
                    (i0 + 1) as u64 * bw,
                    intBitWidth,
                    mask,
                );
                let u2 = unpack_alp_inline(
                    packedPtr,
                    packedLen,
                    (i0 + 2) as u64 * bw,
                    intBitWidth,
                    mask,
                );
                let u3 = unpack_alp_inline(
                    packedPtr,
                    packedLen,
                    (i0 + 3) as u64 * bw,
                    intBitWidth,
                    mask,
                );

                prevInt = prevInt.wrapping_add(u0 as i64 + base);
                unsafe {
                    out64f.add(i0).write(prevInt as f64 * invFactor);
                }
                prevInt = prevInt.wrapping_add(u1 as i64 + base);
                unsafe {
                    out64f.add(i0 + 1).write(prevInt as f64 * invFactor);
                }
                prevInt = prevInt.wrapping_add(u2 as i64 + base);
                unsafe {
                    out64f.add(i0 + 2).write(prevInt as f64 * invFactor);
                }
                prevInt = prevInt.wrapping_add(u3 as i64 + base);
                unsafe {
                    out64f.add(i0 + 3).write(prevInt as f64 * invFactor);
                }
            }
            for i in (chunks * 4 + 1)..row_count {
                let unsigned =
                    unpack_alp_inline(packedPtr, packedLen, i as u64 * bw, intBitWidth, mask);
                prevInt = prevInt.wrapping_add(unsigned as i64 + base);
                unsafe {
                    out64f.add(i).write(prevInt as f64 * invFactor);
                }
            }
        } else if value_size == 8 && !useDelta {
            // Fast path for f64 without delta: batch unpack + convert.
            // Process 4 values at a time to help instruction-level parallelism.
            let chunks = row_count / 4;

            for chunk in 0..chunks {
                let i0 = chunk * 4;
                let u0 = unpack_alp_inline(packedPtr, packedLen, i0 as u64 * bw, intBitWidth, mask);
                let u1 = unpack_alp_inline(
                    packedPtr,
                    packedLen,
                    (i0 + 1) as u64 * bw,
                    intBitWidth,
                    mask,
                );
                let u2 = unpack_alp_inline(
                    packedPtr,
                    packedLen,
                    (i0 + 2) as u64 * bw,
                    intBitWidth,
                    mask,
                );
                let u3 = unpack_alp_inline(
                    packedPtr,
                    packedLen,
                    (i0 + 3) as u64 * bw,
                    intBitWidth,
                    mask,
                );

                let f0 = (u0 as i64 + base) as f64 * invFactor;
                let f1 = (u1 as i64 + base) as f64 * invFactor;
                let f2 = (u2 as i64 + base) as f64 * invFactor;
                let f3 = (u3 as i64 + base) as f64 * invFactor;

                unsafe {
                    (outPtr.add(i0 * 8) as *mut f64).write_unaligned(f0);
                    (outPtr.add((i0 + 1) * 8) as *mut f64).write_unaligned(f1);
                    (outPtr.add((i0 + 2) * 8) as *mut f64).write_unaligned(f2);
                    (outPtr.add((i0 + 3) * 8) as *mut f64).write_unaligned(f3);
                }
            }

            for i in (chunks * 4)..row_count {
                let unsigned =
                    unpack_alp_inline(packedPtr, packedLen, i as u64 * bw, intBitWidth, mask);
                let floatVal = (unsigned as i64 + base) as f64 * invFactor;
                unsafe {
                    (outPtr.add(i * 8) as *mut f64).write_unaligned(floatVal);
                }
            }
        } else {
            for i in 0..row_count {
                let unsigned =
                    unpack_alp_inline(packedPtr, packedLen, i as u64 * bw, intBitWidth, mask);
                let mut intVal = unsigned as i64 + base;

                if useDelta {
                    if i > 0 {
                        intVal = intVal.wrapping_add(prevInt);
                    }
                    prevInt = intVal;
                }

                let floatVal = intVal as f64 * invFactor;

                if value_size == 4 {
                    unsafe {
                        (outPtr.add(i * 4) as *mut f32).write_unaligned(floatVal as f32);
                    }
                } else {
                    unsafe {
                        (outPtr.add(i * 8) as *mut f64).write_unaligned(floatVal);
                    }
                }
            }
        }

        // Apply exceptions (overwrite the placeholder values)
        let exceptionEntrySize = 4 + value_size;
        let exceptionsStart = packedEnd;

        for e in 0..exception_count {
            let offset = exceptionsStart + e * exceptionEntrySize;
            if offset + exceptionEntrySize > encoded.len() {
                return Err(ZyronError::DecodingFailed(
                    "ALP exception data truncated".to_string(),
                ));
            }

            let row_idx = u32::from_le_bytes([
                encoded[offset],
                encoded[offset + 1],
                encoded[offset + 2],
                encoded[offset + 3],
            ]) as usize;

            if row_idx >= row_count {
                return Err(ZyronError::DecodingFailed(format!(
                    "ALP exception row index {} out of range",
                    row_idx
                )));
            }

            let val_offset = offset + 4;
            let out_offset = row_idx * value_size;
            out[out_offset..out_offset + value_size]
                .copy_from_slice(&encoded[val_offset..val_offset + value_size]);
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

        // For range predicates, convert bounds through the same factor
        // and filter on the encoded integer array. Exception rows are
        // handled by decoding only those specific values.
        if let Predicate::Range { low, high } = predicate
            && encoded.len() >= 28
        {
            let factor = f64::from_le_bytes([
                encoded[0], encoded[1], encoded[2], encoded[3], encoded[4], encoded[5], encoded[6],
                encoded[7],
            ]);

            // Only apply pushdown when factor > 0 (not raw fallback)
            // and no exceptions to keep logic simple and correct
            if factor > 0.0 {
                let exceptionCount =
                    u32::from_le_bytes([encoded[12], encoded[13], encoded[14], encoded[15]])
                        as usize;
                let intBitWidth = encoded[16];

                let base = i64::from_le_bytes([
                    encoded[20],
                    encoded[21],
                    encoded[22],
                    encoded[23],
                    encoded[24],
                    encoded[25],
                    encoded[26],
                    encoded[27],
                ]);

                let packedStart = packed_offset(encoded, row_count);
                let packedBits = row_count as u64 * intBitWidth as u64;
                let packedBytes = (packedBits as usize).div_ceil(8);
                let packedEnd = packedStart + packedBytes;

                let useDelta = encoded[18] & FLAG_DELTA != 0;

                // Predicate pushdown only works when no delta and no exceptions
                if exceptionCount == 0 && !useDelta && packedEnd <= encoded.len() {
                    // Convert float bounds to integer bounds in the encoded domain
                    let loInt = match low {
                        Some(lo_bytes) => {
                            let fVal = read_float_from_bytes(lo_bytes, value_size);
                            let scaled = fVal * factor;
                            // Ceiling for lower bound: we want integers >= ceil(scaled)
                            Some(scaled.ceil() as i64)
                        }
                        None => None,
                    };

                    let hiInt = match high {
                        Some(hi_bytes) => {
                            let fVal = read_float_from_bytes(hi_bytes, value_size);
                            let scaled = fVal * factor;
                            // Floor for upper bound: we want integers <= floor(scaled)
                            Some(scaled.floor() as i64)
                        }
                        None => None,
                    };

                    let packed = &encoded[packedStart..packedEnd];
                    let bitmaskLen = row_count.div_ceil(8);
                    let mut bitmask = vec![0u8; bitmaskLen];

                    for i in 0..row_count {
                        let unsigned =
                            unpack_bits(packed, i as u64 * intBitWidth as u64, intBitWidth);
                        let intVal = unsigned as i64 + base;

                        let aboveLow = match loInt {
                            Some(lo) => intVal >= lo,
                            None => true,
                        };
                        let belowHigh = match hiInt {
                            Some(hi) => intVal <= hi,
                            None => true,
                        };

                        if aboveLow && belowHigh {
                            bitmask[i / 8] |= 1 << (i % 8);
                        }
                    }

                    return Ok(bitmask);
                }
            }
        }

        // Fall back to decode-then-evaluate for equality, IN predicates,
        // and cases with exceptions
        let decoded = self.decode(encoded, row_count, value_size)?;
        eval_predicate_on_raw(&decoded, row_count, value_size, predicate)
    }
}

/// Reads a float value from a raw byte slice (not indexed by row).
fn read_float_from_bytes(bytes: &[u8], value_size: usize) -> f64 {
    if value_size == 4 && bytes.len() >= 4 {
        f32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]) as f64
    } else if bytes.len() >= 8 {
        f64::from_le_bytes([
            bytes[0], bytes[1], bytes[2], bytes[3], bytes[4], bytes[5], bytes[6], bytes[7],
        ])
    } else {
        0.0
    }
}

/// Reads a float from data at the given row index.
fn read_float(data: &[u8], row: usize, value_size: usize) -> f64 {
    let offset = row * value_size;
    if value_size == 4 {
        f32::from_le_bytes([
            data[offset],
            data[offset + 1],
            data[offset + 2],
            data[offset + 3],
        ]) as f64
    } else {
        f64::from_le_bytes([
            data[offset],
            data[offset + 1],
            data[offset + 2],
            data[offset + 3],
            data[offset + 4],
            data[offset + 5],
            data[offset + 6],
            data[offset + 7],
        ])
    }
}

/// Finds the best factor (power of 10) that minimizes exceptions.
/// Tests powers of 10 from 10^0 to 10^18 and picks the one that
/// produces the fewest exceptions when converting floats to integers.
fn find_best_factor(floats: &[f64]) -> (f64, i32) {
    let sample_size = floats.len().min(128);
    let sample = &floats[..sample_size];

    let mut best_factor = 1.0f64;
    let mut best_exponent = 0i32;
    let mut best_exceptions = sample_size;

    for exp in 0..=18i32 {
        let factor = 10f64.powi(exp);
        let mut exceptions = 0;

        for &f in sample {
            let scaled = f * factor;
            let rounded = scaled.round();
            if !rounded.is_finite() || rounded < i64::MIN as f64 || rounded > i64::MAX as f64 {
                exceptions += 1;
                continue;
            }
            let int_val = rounded as i64;
            let recovered = int_val as f64 / factor;
            if (recovered - f).abs() > f64::EPSILON * f.abs().max(1.0) {
                exceptions += 1;
            }
        }

        if exceptions < best_exceptions {
            best_exceptions = exceptions;
            best_factor = factor;
            best_exponent = exp;
            if exceptions == 0 {
                break;
            }
        }
    }

    (best_factor, best_exponent)
}

/// Encodes data as raw (no encoding) with ALP header indicating fallback.
/// factor=0.0 signals the decoder to read raw values.
fn encode_raw_fallback(data: &[u8], row_count: usize, value_size: usize) -> Result<Vec<u8>> {
    let rawSize = row_count * value_size;
    let mut out = Vec::with_capacity(28 + rawSize);

    out.extend_from_slice(&0.0f64.to_le_bytes()); // factor = 0.0 (signals raw)
    out.extend_from_slice(&0i32.to_le_bytes()); // exponent
    out.extend_from_slice(&0u32.to_le_bytes()); // exception_count
    out.push(0); // int_bit_width (unused)
    out.push(value_size as u8); // value_size_marker
    out.push(0); // flags
    out.push(0); // reserved
    out.extend_from_slice(&0i64.to_le_bytes()); // base (unused)
    out.extend_from_slice(&data[..rawSize]);

    Ok(out)
}

/// Decodes raw fallback data.
fn decode_raw_fallback(encoded: &[u8], row_count: usize, value_size: usize) -> Result<Vec<u8>> {
    let raw_start = 28;
    let raw_size = row_count * value_size;
    if encoded.len() < raw_start + raw_size {
        return Err(ZyronError::DecodingFailed(
            "ALP raw fallback data truncated".to_string(),
        ));
    }
    Ok(encoded[raw_start..raw_start + raw_size].to_vec())
}

/// Unpacks a single value from a bit-packed array using raw pointer reads.
/// Equivalent to unpack_bits but takes pre-computed pointer and length.
#[inline(always)]
fn unpack_alp_inline(
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
            let hi = unsafe { *packed_ptr.add(byte_idx + 8) } as u64;
            return (val | (hi << (64 - bit_idx))) & mask;
        }
        return val;
    }

    let mut buf = [0u8; 8];
    let available = packed_len.saturating_sub(byte_idx).min(8);
    unsafe {
        std::ptr::copy_nonoverlapping(packed_ptr.add(byte_idx), buf.as_mut_ptr(), available);
    }
    let raw = u64::from_le_bytes(buf);
    (raw >> bit_idx) & mask
}

/// Packs a u64 value at the given bit offset.
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

/// Unpacks a u64 value from the given bit offset.
#[inline]
fn unpack_bits(packed: &[u8], bit_offset: u64, bit_width: u8) -> u64 {
    let byte_idx = (bit_offset / 8) as usize;
    let bit_idx = (bit_offset % 8) as u32;
    let mut buf = [0u8; 9];
    let available = packed.len().saturating_sub(byte_idx).min(9);
    buf[..available].copy_from_slice(&packed[byte_idx..byte_idx + available]);

    let lo = u64::from_le_bytes([
        buf[0], buf[1], buf[2], buf[3], buf[4], buf[5], buf[6], buf[7],
    ]);
    let val = lo >> bit_idx;
    let mask = if bit_width >= 64 {
        u64::MAX
    } else {
        (1u64 << bit_width) - 1
    };

    if bit_idx + bit_width as u32 > 64 {
        let hi = (buf[8] as u64) << (64 - bit_idx);
        (val | hi) & mask
    } else {
        val & mask
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A range decode of the delta stream resumes at the restart boundary at
    /// or before it, so the rows replayed ahead of the first requested one are
    /// bounded by the restart spacing rather than by the segment length.
    #[test]
    fn test_delta_range_replays_at_most_one_restart_spacing() {
        let enc = AlpEncoding;
        const ROWS: usize = 40_000;
        let values: Vec<f64> = (0..ROWS).map(|i| (i as f64) * 0.25 + 1.5).collect();
        let data: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
        let encoded = enc.encode(&data, ROWS, 8).unwrap();
        assert_eq!(
            encoded[18] & FLAG_DELTA,
            FLAG_DELTA,
            "an ascending float column packs as a delta stream"
        );
        assert_eq!(
            encoded[18] & FLAG_RESTART,
            FLAG_RESTART,
            "a delta stream this long carries restart points"
        );

        let shift = encoded[19] as u32;
        let table = &encoded[28..packed_offset(&encoded, ROWS)];
        let start = ROWS - 3;
        let (_, resume) = seed_restart(table, shift, start);
        assert!(resume > 0, "the tail resumes past the head of the stream");
        assert!(
            start - resume < (1usize << shift),
            "replayed {} rows, more than the {} row spacing",
            start - resume,
            1usize << shift
        );

        let ranged = enc.decode_range(&encoded, ROWS, 8, start, ROWS).unwrap();
        let got: Vec<f64> = ranged
            .chunks_exact(8)
            .map(|c| f64::from_le_bytes(c.try_into().unwrap()))
            .collect();
        assert_eq!(got, values[start..]);
    }

    #[test]
    fn test_roundtrip_f64_integers() {
        let enc = AlpEncoding;
        let values: Vec<f64> = (0..100).map(|i| i as f64).collect();
        let mut data = Vec::new();
        for v in &values {
            data.extend_from_slice(&v.to_le_bytes());
        }

        let encoded = enc.encode(&data, 100, 8).unwrap();
        let decoded = enc.decode(&encoded, 100, 8).unwrap();

        for i in 0..100 {
            let original = f64::from_le_bytes(data[i * 8..(i + 1) * 8].try_into().unwrap());
            let recovered = f64::from_le_bytes(decoded[i * 8..(i + 1) * 8].try_into().unwrap());
            assert_eq!(original, recovered, "mismatch at row {}", i);
        }
    }

    #[test]
    fn test_roundtrip_f64_decimals() {
        let enc = AlpEncoding;
        let values: Vec<f64> = (0..100).map(|i| i as f64 * 0.01).collect();
        let mut data = Vec::new();
        for v in &values {
            data.extend_from_slice(&v.to_le_bytes());
        }

        let encoded = enc.encode(&data, 100, 8).unwrap();
        let decoded = enc.decode(&encoded, 100, 8).unwrap();

        for i in 0..100 {
            let original = f64::from_le_bytes(data[i * 8..(i + 1) * 8].try_into().unwrap());
            let recovered = f64::from_le_bytes(decoded[i * 8..(i + 1) * 8].try_into().unwrap());
            assert!(
                (original - recovered).abs() < 1e-10,
                "mismatch at row {}: {} vs {}",
                i,
                original,
                recovered
            );
        }
    }

    #[test]
    fn test_roundtrip_f32() {
        let enc = AlpEncoding;
        let values: Vec<f32> = (0..50).map(|i| i as f32 * 1.5).collect();
        let mut data = Vec::new();
        for v in &values {
            data.extend_from_slice(&v.to_le_bytes());
        }

        let encoded = enc.encode(&data, 50, 4).unwrap();
        let decoded = enc.decode(&encoded, 50, 4).unwrap();

        for i in 0..50 {
            let original = f32::from_le_bytes(data[i * 4..(i + 1) * 4].try_into().unwrap());
            let recovered = f32::from_le_bytes(decoded[i * 4..(i + 1) * 4].try_into().unwrap());
            assert_eq!(
                original.to_bits(),
                recovered.to_bits(),
                "mismatch at row {}: {} vs {}",
                i,
                original,
                recovered
            );
        }
    }

    #[test]
    fn test_empty() {
        let enc = AlpEncoding;
        let encoded = enc.encode(&[], 0, 8).unwrap();
        let decoded = enc.decode(&encoded, 0, 8).unwrap();
        assert!(decoded.is_empty());
    }

    #[test]
    fn test_special_values() {
        let enc = AlpEncoding;
        // Includes NaN, infinity, negative zero (these become exceptions)
        let values: Vec<f64> = vec![1.0, f64::NAN, f64::INFINITY, -0.0, 2.5];
        let mut data = Vec::new();
        for v in &values {
            data.extend_from_slice(&v.to_le_bytes());
        }

        let encoded = enc.encode(&data, 5, 8).unwrap();
        let decoded = enc.decode(&encoded, 5, 8).unwrap();

        // Verify bit-exact preservation including NaN and special values
        for i in 0..5 {
            let orig_bits = u64::from_le_bytes(data[i * 8..(i + 1) * 8].try_into().unwrap());
            let dec_bits = u64::from_le_bytes(decoded[i * 8..(i + 1) * 8].try_into().unwrap());
            assert_eq!(orig_bits, dec_bits, "bit mismatch at row {}", i);
        }
    }

    #[test]
    fn test_compression_ratio() {
        let enc = AlpEncoding;
        // 1000 values like 12.34, 56.78, etc. (2 decimal places)
        let mut data = Vec::new();
        for i in 0..1000 {
            let v = (i as f64) * 0.01 + 100.0;
            data.extend_from_slice(&v.to_le_bytes());
        }

        let encoded = enc.encode(&data, 1000, 8).unwrap();
        // Factor=100 converts to integers [10000..10999], bit_width ~14 bits
        // 14 * 1000 / 8 = 1750 bytes + header < 8000 raw
        assert!(encoded.len() < data.len());
    }
}
