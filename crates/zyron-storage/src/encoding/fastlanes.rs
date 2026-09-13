//! FastLanes integer encoding: Frame-of-Reference (FoR) + Delta + bit-packing.
//!
//! For integer columns, subtracts the minimum value (FoR base) from all values,
//! reducing the bit width needed per value. For sorted data, applies delta
//! encoding before bit-packing. Uses unaligned u64 reads for batch unpacking.
//!
//! Based on FastLanes (VLDB 2023), tuned for page-aligned columnar storage.

use super::unpack::Packed;
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
/// Per-mini-block bit width: each block of MINIBLOCK_SIZE residuals carries
/// its own width and is packed byte-aligned.
const FLAG_MINIBLOCK: u8 = 0x10;
/// Effective-resolution scale: residuals share a common factor, stored once
/// with a nested core blob holding the quotients.
const FLAG_SCALE: u8 = 0x20;
/// Periodic restart values ahead of the packed array, so a range decode of a
/// cumulative layout seeds from the nearest boundary rather than replaying
/// the whole prefix. Set alongside FLAG_DELTA or FLAG_DELTA_OF_DELTA.
const FLAG_RESTART: u8 = 0x40;

/// Running values a constant-step decode of the wide layout carries at once.
/// Each covers every fourth row, so the adds of four rows are in flight
/// together instead of queueing behind one carry chain
const LINEAR_LANES: usize = 4;

/// 8-byte (and narrower) encoded format:
///   [0..8]    base_value: u64 (FoR base, little-endian)
///   [8]       bit_width: u8 (bits per packed value after FoR subtraction)
///   [9]       flags: u8 (FLAG_DELTA / FLAG_DELTA_OF_DELTA)
///   [10..12]  reserved: u16, or [10] = restart shift when FLAG_RESTART is set
///   [12..]    restart table when FLAG_RESTART is set, then the packed bit array
///
/// 16-byte (i128/u128) encoded format:
///   [0..16]   base_value: u128 (FoR base, little-endian)
///   [16]      bit_width: u8 (bits per packed value, up to 128)
///   [17]      flags: u8
///   [18]      restart shift when FLAG_RESTART is set
///   [19..24]  reserved
///   [24..]    restart table when FLAG_RESTART is set, then the packed bit array
const WIDE_HEADER_SIZE: usize = 24;

/// Restart spacing floor, matching the columnar zone map batch size so every
/// restart point lands on a zone boundary and a zone-aligned range needs no
/// replay at all.
const RESTART_MIN_SHIFT: u32 = 10;

/// Number of restart boundaries for a row count and spacing. Boundary k covers
/// row `k << shift` counted from 1, so a segment shorter than one spacing
/// carries no table.
#[inline]
fn restart_count(row_count: usize, shift: u32) -> usize {
    if row_count == 0 || shift >= usize::BITS {
        0
    } else {
        (row_count - 1) >> shift
    }
}

/// Widens the restart spacing until the table costs at most a thirty-second of
/// the packed array. The spacing stays a power-of-two multiple of the floor, so
/// restart points remain aligned to zone boundaries at every width.
fn choose_restart_shift(row_count: usize, packed_bytes: usize, entry_size: usize) -> u32 {
    let budget = (packed_bytes / 32).max(entry_size);
    let mut shift = RESTART_MIN_SHIFT;
    while shift < 31 && restart_count(row_count, shift) * entry_size > budget {
        shift += 1;
    }
    shift
}

/// Bits needed to pack residuals up to `max`. A column of identical values
/// still packs one bit per row so the width is never zero.
#[inline]
fn pack_width(max: u64) -> u8 {
    if max == 0 {
        1
    } else {
        (64 - max.leading_zeros()) as u8
    }
}

/// Restart entry width for the narrow layouts. Delta carries one accumulator,
/// delta-of-delta carries the running residual and the running first difference.
#[inline]
fn narrow_restart_entry(flags: u8) -> usize {
    if flags & FLAG_DELTA_OF_DELTA != 0 {
        16
    } else {
        8
    }
}

/// Byte offset of the packed bit array in the narrow layout, past any restart
/// table.
#[inline]
fn narrow_packed_offset(encoded: &[u8], flags: u8, row_count: usize) -> usize {
    if flags & FLAG_RESTART == 0 {
        return 12;
    }
    12 + restart_count(row_count, encoded[10] as u32) * narrow_restart_entry(flags)
}

/// Restart entry width for the 16-byte layouts.
#[inline]
fn wide_restart_entry(flags: u8) -> usize {
    if flags & FLAG_DELTA_OF_DELTA != 0 {
        32
    } else {
        16
    }
}

/// Byte offset of the packed bit array in the 16-byte layout.
#[inline]
fn wide_packed_offset(encoded: &[u8], flags: u8, row_count: usize) -> usize {
    if flags & FLAG_RESTART == 0 {
        return WIDE_HEADER_SIZE;
    }
    WIDE_HEADER_SIZE + restart_count(row_count, encoded[18] as u32) * wide_restart_entry(flags)
}

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

    /// Every layout of this encoding answers a range without materializing the
    /// rows outside it.
    ///
    /// A constant step is a closed form: row i is `first + i * step`, computed
    /// from the header alone. Plain frame of reference, patched frame of
    /// reference and the mini-block form all pack residuals to a known width,
    /// so row i is at a computable bit offset and only the exception entries
    /// landing inside the range are applied.
    ///
    /// Delta and delta-of-delta are cumulative, row i being defined against
    /// row i-1. They carry a table of periodic restart values, so a range
    /// seeds its running state at the boundary at or before `start` and
    /// replays at most one restart spacing instead of the whole prefix.
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
        if value_size == 16 {
            return decode_range_wide(encoded, row_count, start, end);
        }
        decode_narrow(encoded, row_count, value_size, start, end)
    }

    /// A whole column is the range of every row, so the two share one
    /// decode and the kernels behind it
    fn decode(&self, encoded: &[u8], row_count: usize, value_size: usize) -> Result<Vec<u8>> {
        if row_count == 0 {
            return Ok(Vec::new());
        }
        if value_size == 16 {
            return decode_wide(encoded, row_count);
        }
        decode_narrow(encoded, row_count, value_size, 0, row_count)
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

        let packed_off = narrow_packed_offset(encoded, flags, row_count);
        if encoded.len() < packed_off {
            return Err(ZyronError::DecodingFailed(
                "FastLanes restart table truncated".to_string(),
            ));
        }
        let packed = &encoded[packed_off..];

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

                    // The residual is read straight out of the packed array,
                    // eight rows at a time, so a row that survives the zone
                    // maps is answered without materializing the column
                    let residuals = Packed::new(packed, bit_width);
                    return Ok(crate::encoding::bitmask_from_rows(row_count, |i| {
                        let residual = residuals.at(i);
                        residual >= loResidual && residual <= hiResidual
                    }));
                }
                Predicate::Equality(target) => {
                    let targetVal = read_u64_le(target, 0, target.len().min(8));
                    if targetVal < base_value || targetVal > maxRepresentable {
                        return Ok(vec![0u8; row_count.div_ceil(8)]);
                    }
                    let targetResidual = targetVal - base_value;
                    let residuals = Packed::new(packed, bit_width);
                    return Ok(crate::encoding::bitmask_from_rows(row_count, |i| {
                        residuals.at(i) == targetResidual
                    }));
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
                    if targetResiduals.is_empty() {
                        return Ok(vec![0u8; row_count.div_ceil(8)]);
                    }
                    let residuals = Packed::new(packed, bit_width);
                    return Ok(crate::encoding::bitmask_from_rows(row_count, |i| {
                        targetResiduals.contains(&residuals.at(i))
                    }));
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
            let stream = Packed::new(packed, bit_width);
            let r0 = stream.at(0);
            let step = stream.at(1);

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
                stream.at(idx) == step
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

        // The running sum of the deltas is the FoR residual of every row,
        // unpacked and summed in one pass
        let mut residuals = vec![0u64; row_count];
        // SAFETY: the buffer holds row_count eight byte slots
        unsafe {
            Packed::new(packed, bit_width).prefix_into(
                0,
                row_count,
                0,
                0,
                residuals.as_mut_ptr() as *mut u8,
                8,
            );
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
        Ok(match predicate {
            Predicate::Range { low, high } => {
                let loVal = match low {
                    Some(lo) => read_u64_le(lo, 0, lo.len().min(8)),
                    None => 0,
                };
                let hiVal = match high {
                    Some(hi) => read_u64_le(hi, 0, hi.len().min(8)),
                    None => u64::MAX,
                };
                crate::encoding::bitmask_from_rows(row_count, |i| {
                    let v = residuals[i].wrapping_add(base_value);
                    v >= loVal && v <= hiVal
                })
            }
            Predicate::Equality(target) => {
                let targetVal = read_u64_le(target, 0, target.len().min(8));
                crate::encoding::bitmask_from_rows(row_count, |i| {
                    residuals[i].wrapping_add(base_value) == targetVal
                })
            }
            Predicate::In(values) => {
                let targets: Vec<u64> = values
                    .iter()
                    .map(|v| read_u64_le(v, 0, v.len().min(8)))
                    .collect();
                crate::encoding::bitmask_from_rows(row_count, |i| {
                    targets.contains(&residuals[i].wrapping_add(base_value))
                })
            }
        })
    }
}

/// Reads restart entry `k - 1` as a fixed number of little-endian u64 words,
/// where k is the boundary index covering `start`. Returns the seeded words and
/// the row that decoding resumes at, or None when the range starts before the
/// first boundary.
fn seed_narrow_restart<const W: usize>(
    restart: Option<(&[u8], u32)>,
    start: usize,
) -> Option<([u64; W], usize)> {
    let (table, shift) = restart?;
    if shift >= usize::BITS {
        return None;
    }
    let k = start >> shift;
    if k == 0 {
        return None;
    }
    let at = (k - 1) * W * 8;
    if at + W * 8 > table.len() {
        return None;
    }
    let mut words = [0u64; W];
    for (w, slot) in words.iter_mut().enumerate() {
        let o = at + w * 8;
        *slot = u64::from_le_bytes([
            table[o],
            table[o + 1],
            table[o + 2],
            table[o + 3],
            table[o + 4],
            table[o + 5],
            table[o + 6],
            table[o + 7],
        ]);
    }
    Some((words, k << shift))
}

/// Rows `start..end` of a narrow layout as `value_size` byte little endian
/// integers.
///
/// Every layout answers a range without materializing the rows outside it.
/// A constant step is a closed form, row i is `first + i * step` from the
/// header alone. Plain frame of reference, patched frame of reference and
/// the mini-block form pack residuals to a known width, so row i is at a
/// computable bit offset and only the exception entries landing inside the
/// range are applied. Delta and delta-of-delta are cumulative, row i being
/// defined against row i-1. They carry a table of periodic restart values,
/// so a range seeds its running state at the boundary at or before `start`
/// and sums at most one restart spacing instead of the whole prefix.
///
/// The scale layout wraps another core blob, so it unwraps its own header
/// here and recurses on the inner blob
fn decode_narrow(
    encoded: &[u8],
    row_count: usize,
    value_size: usize,
    start: usize,
    end: usize,
) -> Result<Vec<u8>> {
    if !(1..=8).contains(&value_size) {
        return Err(ZyronError::DecodingFailed(format!(
            "FastLanes decodes 1 to 8 or 16 byte values, not {value_size}"
        )));
    }
    if encoded.len() < 12 {
        return Err(ZyronError::DecodingFailed(
            "FastLanes header too short".to_string(),
        ));
    }
    let base_value = read_u64_le(encoded, 0, 8);
    let bit_width = encoded[8];
    let flags = encoded[9];
    let taken = end - start;

    if flags & FLAG_SCALE != 0 {
        if encoded.len() < 20 {
            return Err(ZyronError::DecodingFailed(
                "FastLanes scale blob too short".to_string(),
            ));
        }
        let scale = read_u64_le(encoded, 12, 8);
        let quotients = decode_narrow(&encoded[20..], row_count, value_size, start, end)?;
        // SAFETY: the loop below writes every one of the taken slots at
        // value_size bytes each, which is the whole buffer, before anything
        // reads it
        let mut out = unsafe { super::scratch::take_uninit(taken * value_size) };
        for i in 0..taken {
            let q = read_u64_le(&quotients, i * value_size, value_size);
            write_le(
                &mut out,
                i,
                value_size,
                base_value.wrapping_add(q.wrapping_mul(scale)),
            );
        }
        return Ok(out);
    }

    if flags & FLAG_CONST_STEP != 0 {
        if encoded.len() < 20 {
            return Err(ZyronError::DecodingFailed(
                "FastLanes constant-step blob too short".to_string(),
            ));
        }
        let step = read_u64_le(encoded, 12, 8);
        // SAFETY: store_linear writes every one of the taken slots at
        // value_size bytes each, which is the whole buffer, before anything
        // reads it
        let mut out = unsafe { super::scratch::take_uninit(taken * value_size) };
        store_linear(&mut out, value_size, start, taken, base_value, step);
        return Ok(out);
    }

    if flags & FLAG_MINIBLOCK != 0 {
        return decode_range_miniblock(encoded, row_count, value_size, start, end, base_value);
    }

    if bit_width == 0 || bit_width > 64 {
        return Err(ZyronError::DecodingFailed(format!(
            "invalid FastLanes bit width: {bit_width}"
        )));
    }

    if flags & FLAG_PFOR != 0 {
        let exc_count = u16::from_le_bytes([encoded[10], encoded[11]]) as usize;
        let table_off = 12usize;
        let table_bytes = exc_count * 12;
        if encoded.len() < table_off + table_bytes {
            return Err(ZyronError::DecodingFailed(
                "FastLanes PFOR blob malformed".to_string(),
            ));
        }
        let stream = Packed::new(&encoded[table_off + table_bytes..], bit_width);
        // SAFETY: the kernel writes every one of the taken slots at
        // value_size bytes each, which is the whole buffer, before anything
        // reads it
        let mut out = unsafe { super::scratch::take_uninit(taken * value_size) };
        // SAFETY: out holds taken slots of value_size bytes
        unsafe { stream.add_base_into(start, taken, base_value, out.as_mut_ptr(), value_size) };
        // Exceptions carry their own row index, so the ones outside the
        // range are stepped over rather than decoded
        for e in 0..exc_count {
            let o = table_off + e * 12;
            let pos =
                u32::from_le_bytes([encoded[o], encoded[o + 1], encoded[o + 2], encoded[o + 3]])
                    as usize;
            if pos < start || pos >= end {
                continue;
            }
            let resid = read_u64_le(encoded, o + 4, 8);
            write_le(
                &mut out,
                pos - start,
                value_size,
                resid.wrapping_add(base_value),
            );
        }
        return Ok(out);
    }

    let packed_off = narrow_packed_offset(encoded, flags, row_count);
    if encoded.len() < packed_off {
        return Err(ZyronError::DecodingFailed(
            "FastLanes restart table truncated".to_string(),
        ));
    }
    let restart = if flags & FLAG_RESTART != 0 {
        Some((&encoded[12..packed_off], encoded[10] as u32))
    } else {
        None
    };
    let stream = Packed::new(&encoded[packed_off..], bit_width);

    if flags & FLAG_DELTA_OF_DELTA != 0 {
        return Ok(decode_range_dod(
            stream, base_value, value_size, row_count, restart, start, end,
        ));
    }
    if flags & FLAG_DELTA != 0 {
        return Ok(decode_range_delta(
            stream, base_value, value_size, restart, start, end,
        ));
    }

    // SAFETY: the kernel writes every one of the taken slots at value_size
    // bytes each, which is the whole buffer, before anything reads it
    let mut out = unsafe { super::scratch::take_uninit(taken * value_size) };
    // SAFETY: out holds taken slots of value_size bytes
    unsafe { stream.add_base_into(start, taken, base_value, out.as_mut_ptr(), value_size) };
    Ok(out)
}

/// Stores `first + row * step` for rows `start..start + count` as
/// `value_size` byte little endian integers. The two hot widths go through
/// typed pointers so the loop becomes vector stores
fn store_linear(
    out: &mut [u8],
    value_size: usize,
    start: usize,
    count: usize,
    first: u64,
    step: u64,
) {
    assert!(
        out.len() >= count * value_size,
        "linear fill wider than its buffer"
    );
    // The value at the first row of the range, so the loop below counts
    // from zero
    let first = first.wrapping_add((start as u64).wrapping_mul(step));
    match value_size {
        8 => {
            let p = out.as_mut_ptr() as *mut u64;
            for i in 0..count {
                let v = first.wrapping_add((i as u64).wrapping_mul(step));
                // SAFETY: slot i is inside out, which holds count slots
                unsafe { p.add(i).write_unaligned(v) };
            }
        }
        4 => {
            let p = out.as_mut_ptr() as *mut u32;
            for i in 0..count {
                let v = first.wrapping_add((i as u64).wrapping_mul(step)) as u32;
                // SAFETY: slot i is inside out, which holds count slots
                unsafe { p.add(i).write_unaligned(v) };
            }
        }
        _ => {
            for i in 0..count {
                write_le(
                    out,
                    i,
                    value_size,
                    first.wrapping_add((i as u64).wrapping_mul(step)),
                );
            }
        }
    }
}

/// Range decode for the narrow delta layout. Seeds the running sum at the
/// restart boundary at or before `start`, sums the residuals between that
/// boundary and `start` without writing them, then emits the requested rows
fn decode_range_delta(
    stream: Packed<'_>,
    base_value: u64,
    value_size: usize,
    restart: Option<(&[u8], u32)>,
    start: usize,
    end: usize,
) -> Vec<u8> {
    let (mut accumulator, row) = match seed_narrow_restart::<1>(restart, start) {
        Some((words, at)) => (words[0], at),
        None => (0u64, 0usize),
    };
    if row < start {
        accumulator = accumulator.wrapping_add(stream.sum(row, start - row));
    }
    let taken = end - start;
    // SAFETY: the kernel writes every one of the taken slots at value_size
    // bytes each, which is the whole buffer, before anything reads it
    let mut out = unsafe { super::scratch::take_uninit(taken * value_size) };
    // SAFETY: out holds taken slots of value_size bytes
    unsafe {
        stream.prefix_into(
            start,
            taken,
            accumulator,
            base_value,
            out.as_mut_ptr(),
            value_size,
        );
    }
    out
}

/// Range decode for the narrow delta-of-delta layout. A restart entry carries
/// both running values the double prefix sum needs, the residual and the first
/// difference. Without one the two head rows are replayed verbatim, which is
/// what the layout stores them as. The second differences from the seed row
/// to the end of the range are unpacked in one pass, and the double prefix
/// sum is then a walk over them
fn decode_range_dod(
    stream: Packed<'_>,
    base_value: u64,
    value_size: usize,
    row_count: usize,
    restart: Option<(&[u8], u32)>,
    start: usize,
    end: usize,
) -> Vec<u8> {
    let taken = end - start;
    // SAFETY: every row of start..end is written below at value_size bytes,
    // the head rows by hand and the rest by the walk, before anything reads
    // the buffer
    let mut out = unsafe { super::scratch::take_uninit(taken * value_size) };
    let mut residual: u64;
    let mut delta: u64;
    let row: usize;

    match seed_narrow_restart::<2>(restart, start) {
        Some((words, at)) => {
            residual = words[0];
            delta = words[1];
            row = at;
        }
        None => {
            residual = stream.at(0);
            delta = 0;
            if start == 0 {
                write_le(&mut out, 0, value_size, residual.wrapping_add(base_value));
            }
            if row_count > 1 {
                delta = unzigzag_i64(stream.at(1)) as u64;
                residual = residual.wrapping_add(delta);
                if start <= 1 && end > 1 {
                    write_le(
                        &mut out,
                        1 - start,
                        value_size,
                        residual.wrapping_add(base_value),
                    );
                }
            }
            row = 2;
        }
    }

    if row < end {
        let second = stream.unpack_vec(row, end - row);
        let replay = start.saturating_sub(row);
        for &z in &second[..replay] {
            delta = delta.wrapping_add(unzigzag_i64(z) as u64);
            residual = residual.wrapping_add(delta);
        }
        for (i, &z) in second[replay..].iter().enumerate() {
            delta = delta.wrapping_add(unzigzag_i64(z) as u64);
            residual = residual.wrapping_add(delta);
            write_le(
                &mut out,
                (row + replay + i) - start,
                value_size,
                residual.wrapping_add(base_value),
            );
        }
    }
    out
}

/// Range decode for the mini-block layout. Block widths are walked to reach the
/// byte offset of the block holding `start`, which costs one byte read per
/// skipped block, then only the blocks overlapping the range are unpacked
fn decode_range_miniblock(
    encoded: &[u8],
    row_count: usize,
    value_size: usize,
    start: usize,
    end: usize,
    base_value: u64,
) -> Result<Vec<u8>> {
    let first_block = start / MINIBLOCK_SIZE;
    let last_block = (end - 1) / MINIBLOCK_SIZE;
    // SAFETY: the blocks from first_block to last_block cover start..end
    // without a gap, and each writes its rows at value_size bytes before
    // anything reads the buffer. An error on the way out drops the buffer
    // unread
    let mut out = unsafe { super::scratch::take_uninit((end - start) * value_size) };
    let mut off = 12usize;
    for b in 0..=last_block {
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
        let block_start = b * MINIBLOCK_SIZE;
        let block_end = (block_start + MINIBLOCK_SIZE).min(row_count);
        let block_bytes = ((block_end - block_start) as u64 * bw as u64).div_ceil(8) as usize;
        if off + block_bytes > encoded.len() {
            return Err(ZyronError::DecodingFailed(
                "FastLanes mini-block blob truncated".to_string(),
            ));
        }
        if b >= first_block {
            let lo = start.max(block_start);
            let hi = end.min(block_end);
            let block = Packed::new(&encoded[off..off + block_bytes], bw);
            // SAFETY: rows lo..hi land in slots lo - start onward, which
            // are inside out
            unsafe {
                block.add_base_into(
                    lo - block_start,
                    hi - lo,
                    base_value,
                    out.as_mut_ptr().add((lo - start) * value_size),
                    value_size,
                );
            }
        }
        off += block_bytes;
    }
    Ok(out)
}

/// Reads restart entry `k - 1` from the 16-byte layout table as u128 words.
fn seed_wide_restart<const W: usize>(
    restart: Option<(&[u8], u32)>,
    start: usize,
) -> Option<([u128; W], usize)> {
    let (table, shift) = restart?;
    if shift >= usize::BITS {
        return None;
    }
    let k = start >> shift;
    if k == 0 {
        return None;
    }
    let at = (k - 1) * W * 16;
    if at + W * 16 > table.len() {
        return None;
    }
    let mut words = [0u128; W];
    for (w, slot) in words.iter_mut().enumerate() {
        *slot = read_u128_le(table, at + w * 16);
    }
    Some((words, k << shift))
}

/// Fills `count` little-endian 16-byte slots ascending by `step` from
/// `first`, for a step that fits in 64 bits.
///
/// The low halves are a 64-bit arithmetic sequence and the high half moves
/// only where that sequence wraps, which is once every `u64::MAX / step`
/// rows. Each run between wraps is written as a 64-bit fill beside a
/// constant, so a row costs one narrow add and no carry
fn store_linear_wide(out: &mut [u8], count: usize, first: u128, step: u64) {
    debug_assert!(
        out.len() >= count * 16,
        "wide linear fill wider than its buffer"
    );
    debug_assert!(step != 0, "a wide linear fill needs a step");
    let p = out.as_mut_ptr() as *mut u64;
    // A buffer the allocator handed out is word aligned, and a run written
    // through a slice of words is one the compiler pairs up into wide
    // stores. The unaligned stores answer for a buffer that is not
    let words: Option<&mut [u64]> = if p.align_offset(std::mem::align_of::<u64>()) == 0 {
        // SAFETY: out holds count slots of sixteen bytes, which is count
        // pairs of words, and the pointer is aligned for them
        Some(unsafe { std::slice::from_raw_parts_mut(p, count * 2) })
    } else {
        None
    };
    let mut low = first as u64;
    let mut high = (first >> 64) as u64;
    let mut at = 0usize;
    let mut words = words;
    while at < count {
        // Rows this run covers, the ones before the low half wraps
        let room = ((u64::MAX - low) / step) as usize;
        let run = room.saturating_add(1).min(count - at);
        match &mut words {
            Some(words) => {
                for (i, pair) in words[at * 2..(at + run) * 2]
                    .chunks_exact_mut(2)
                    .enumerate()
                {
                    // Inside the run `low + i * step` is under the wrap by
                    // the way `room` was measured, so neither the product
                    // nor the sum carries
                    pair[0] = (low + i as u64 * step).to_le();
                    pair[1] = high.to_le();
                }
            }
            None => {
                for i in 0..run {
                    let value = low + i as u64 * step;
                    // SAFETY: slot at + i is inside out, which holds count
                    // slots of two words
                    unsafe {
                        p.add((at + i) * 2).write_unaligned(value.to_le());
                        p.add((at + i) * 2 + 1).write_unaligned(high.to_le());
                    }
                }
            }
        }
        at += run;
        let past = (low as u128) + (run as u128) * (step as u128);
        low = past as u64;
        high = high.wrapping_add((past >> 64) as u64);
    }
}

/// Range decode for the 16-byte layouts. Mirrors the narrow path: closed form
/// for a constant step, direct bit addressing for plain and patched frame of
/// reference, and restart-seeded replay for the two cumulative forms.
fn decode_range_wide(
    encoded: &[u8],
    row_count: usize,
    start: usize,
    end: usize,
) -> Result<Vec<u8>> {
    if encoded.len() < WIDE_HEADER_SIZE {
        return Err(ZyronError::DecodingFailed(
            "FastLanes wide header too short".to_string(),
        ));
    }
    let base = read_u128_le(encoded, 0);
    let bit_width = encoded[16];
    let flags = encoded[17];
    let taken = end - start;
    // SAFETY: every layout below writes each of the taken slots, sixteen
    // bytes apiece, which is the whole buffer, before anything reads it.
    // The closed forms and the plain layout fill slots in order, the
    // patched layout fills every slot and then overwrites its exceptions,
    // and the cumulative layouts write each row of start..end as they
    // replay it
    let mut out = unsafe { super::scratch::take_uninit(taken * 16) };
    let write = |out: &mut [u8], i: usize, v: u128| {
        debug_assert!(i < taken, "wide decode slot outside its buffer");
        // SAFETY: slot i is inside out, which holds taken slots of sixteen
        // bytes, and the store is unaligned so the buffer needs no alignment
        unsafe { (out.as_mut_ptr() as *mut u128).add(i).write_unaligned(v) };
    };

    if flags & FLAG_SCALE != 0 {
        if encoded.len() < WIDE_HEADER_SIZE + 16 {
            return Err(ZyronError::DecodingFailed(
                "FastLanes wide scale blob too short".to_string(),
            ));
        }
        let scale = read_u128_le(encoded, WIDE_HEADER_SIZE);
        let quotients =
            decode_range_wide(&encoded[WIDE_HEADER_SIZE + 16..], row_count, start, end)?;
        for i in 0..taken {
            let q = read_u128_le(&quotients, i * 16);
            write(&mut out, i, base.wrapping_add(q.wrapping_mul(scale)));
        }
        // The inner decode's buffer goes back for the next one on this
        // thread rather than to the allocator
        super::scratch::give_back(quotients);
        return Ok(out);
    }

    if flags & FLAG_CONST_STEP != 0 {
        if encoded.len() < WIDE_HEADER_SIZE + 16 {
            return Err(ZyronError::DecodingFailed(
                "FastLanes wide constant-step blob too short".to_string(),
            ));
        }
        let step = read_u128_le(encoded, WIDE_HEADER_SIZE);
        let first = base.wrapping_add((start as u128).wrapping_mul(step));
        match u64::try_from(step) {
            // A step inside 64 bits leaves the high half of the value fixed
            // between the rows where the low half wraps, so each run is a
            // 64-bit sequence beside a constant rather than a carry chain
            Ok(narrow) if narrow != 0 => store_linear_wide(&mut out, taken, first, narrow),
            _ => {
                // Values a lane apart, each carried forward by that many
                // steps, so a row's add waits on the one a lane back rather
                // than on the row before it
                let stride = step.wrapping_mul(LINEAR_LANES as u128);
                let mut lanes: [u128; LINEAR_LANES] =
                    std::array::from_fn(|lane| first.wrapping_add(step.wrapping_mul(lane as u128)));
                let whole = taken - taken % LINEAR_LANES;
                let mut i = 0;
                while i < whole {
                    for (lane, value) in lanes.iter_mut().enumerate() {
                        write(&mut out, i + lane, *value);
                        *value = value.wrapping_add(stride);
                    }
                    i += LINEAR_LANES;
                }
                while i < taken {
                    write(&mut out, i, lanes[i - whole]);
                    i += 1;
                }
            }
        }
        return Ok(out);
    }

    if bit_width == 0 || bit_width > 128 {
        return Err(ZyronError::DecodingFailed(format!(
            "invalid FastLanes wide bit width: {bit_width}"
        )));
    }

    if flags & FLAG_PFOR != 0 {
        let exc_count = u16::from_le_bytes([encoded[18], encoded[19]]) as usize;
        let table_off = WIDE_HEADER_SIZE;
        let table_bytes = exc_count * 20;
        if encoded.len() < table_off + table_bytes {
            return Err(ZyronError::DecodingFailed(
                "FastLanes wide PFOR blob malformed".to_string(),
            ));
        }
        let packed = &encoded[table_off + table_bytes..];
        for i in 0..taken {
            let r = unpack_bits_128(packed, (start + i) as u64 * bit_width as u64, bit_width);
            write(&mut out, i, r.wrapping_add(base));
        }
        for e in 0..exc_count {
            let o = table_off + e * 20;
            let pos =
                u32::from_le_bytes([encoded[o], encoded[o + 1], encoded[o + 2], encoded[o + 3]])
                    as usize;
            if pos < start || pos >= end {
                continue;
            }
            let resid = read_u128_le(encoded, o + 4);
            write(&mut out, pos - start, resid.wrapping_add(base));
        }
        return Ok(out);
    }

    let packed_off = wide_packed_offset(encoded, flags, row_count);
    if encoded.len() < packed_off {
        return Err(ZyronError::DecodingFailed(
            "FastLanes wide restart table truncated".to_string(),
        ));
    }
    let restart = if flags & FLAG_RESTART != 0 {
        Some((&encoded[WIDE_HEADER_SIZE..packed_off], encoded[18] as u32))
    } else {
        None
    };
    let packed = &encoded[packed_off..];
    let at_bit = |row: usize| row as u64 * bit_width as u64;

    if flags & FLAG_DELTA_OF_DELTA != 0 {
        let mut residual: u128;
        let mut delta: i128;
        let mut row: usize;
        match seed_wide_restart::<2>(restart, start) {
            Some((words, at)) => {
                residual = words[0];
                delta = words[1] as i128;
                row = at;
            }
            None => {
                residual = unpack_bits_128(packed, 0, bit_width);
                delta = 0;
                if start == 0 {
                    write(&mut out, 0, residual.wrapping_add(base));
                }
                if row_count > 1 {
                    delta = unzigzag_i128(unpack_bits_128(packed, at_bit(1), bit_width));
                    residual = residual.wrapping_add(delta as u128);
                    if start <= 1 && end > 1 {
                        write(&mut out, 1 - start, residual.wrapping_add(base));
                    }
                }
                row = 2;
            }
        }
        while row < start {
            delta = delta.wrapping_add(unzigzag_i128(unpack_bits_128(
                packed,
                at_bit(row),
                bit_width,
            )));
            residual = residual.wrapping_add(delta as u128);
            row += 1;
        }
        while row < end {
            delta = delta.wrapping_add(unzigzag_i128(unpack_bits_128(
                packed,
                at_bit(row),
                bit_width,
            )));
            residual = residual.wrapping_add(delta as u128);
            write(&mut out, row - start, residual.wrapping_add(base));
            row += 1;
        }
        return Ok(out);
    }

    if flags & FLAG_DELTA != 0 {
        let (mut residual, mut row) = match seed_wide_restart::<1>(restart, start) {
            Some((words, at)) => (words[0], at),
            None => (unpack_bits_128(packed, 0, bit_width), 1usize),
        };
        if row == 1 && start == 0 {
            write(&mut out, 0, residual.wrapping_add(base));
        }
        while row < start {
            residual =
                residual.wrapping_add(
                    unzigzag_i128(unpack_bits_128(packed, at_bit(row), bit_width)) as u128,
                );
            row += 1;
        }
        while row < end {
            residual =
                residual.wrapping_add(
                    unzigzag_i128(unpack_bits_128(packed, at_bit(row), bit_width)) as u128,
                );
            write(&mut out, row - start, residual.wrapping_add(base));
            row += 1;
        }
        return Ok(out);
    }

    for i in 0..taken {
        let s = unpack_bits_128(packed, at_bit(start + i), bit_width);
        write(&mut out, i, s.wrapping_add(base));
    }
    Ok(out)
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
/// The two hot widths copy a fixed length, which is one store
#[inline(always)]
fn write_le(out: &mut [u8], idx: usize, value_size: usize, val: u64) {
    let bytes = val.to_le_bytes();
    match value_size {
        8 => out[idx * 8..idx * 8 + 8].copy_from_slice(&bytes),
        4 => out[idx * 4..idx * 4 + 4].copy_from_slice(&bytes[..4]),
        _ => {
            let start = idx * value_size;
            out[start..start + value_size].copy_from_slice(&bytes[..value_size]);
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
    let base_value = values.iter().copied().min().unwrap_or(0);
    let mut residuals: Vec<u64> = values.iter().map(|v| v - base_value).collect();
    let mut bit_width = pack_width(residuals.iter().copied().max().unwrap_or(0));
    let mut packed_bytes = (row_count as u64 * bit_width as u64).div_ceil(8) as usize;
    let mut flags = 0u8;
    let mut shift = RESTART_MIN_SHIFT;
    let mut restarts = 0usize;

    // Delta wins on data that ascends smoothly, but a column that ascends and
    // drops once wraps that single difference to a full-width value, which
    // makes the delta stream wider than the residuals it replaces. Both forms
    // are measured and the smaller is kept, restart table included
    let sorted_count = values.windows(2).filter(|w| w[1] >= w[0]).count();
    if row_count > 1 && sorted_count >= (row_count - 1) * 9 / 10 {
        let mut delta = residuals.clone();
        for i in (1..delta.len()).rev() {
            delta[i] = delta[i].wrapping_sub(delta[i - 1]);
        }
        let delta_width = pack_width(delta.iter().copied().max().unwrap_or(0));
        let delta_bytes = (row_count as u64 * delta_width as u64).div_ceil(8) as usize;
        let delta_shift = choose_restart_shift(row_count, delta_bytes, 8);
        let delta_restarts = restart_count(row_count, delta_shift);
        if delta_bytes + delta_restarts * 8 < packed_bytes {
            residuals = delta;
            bit_width = delta_width;
            packed_bytes = delta_bytes;
            flags = FLAG_DELTA;
            shift = delta_shift;
            restarts = delta_restarts;
        }
    }

    let mut packed = vec![0u8; packed_bytes];
    for (i, &val) in residuals.iter().enumerate() {
        pack_bits(&mut packed, i as u64 * bit_width as u64, val, bit_width);
    }

    // The entry for restart boundary k holds the running sum reached at the row
    // before it, which is that row's FoR residual
    let mut out = Vec::with_capacity(12 + restarts * 8 + packed_bytes);
    out.extend_from_slice(&base_value.to_le_bytes()); // [0..8]
    out.push(bit_width); // [8]
    out.push(if restarts > 0 {
        flags | FLAG_RESTART
    } else {
        flags
    }); // [9]
    out.push(shift as u8); // [10] restart spacing
    out.push(0); // [11] reserved
    for k in 1..=restarts {
        let row = (k << shift) - 1;
        out.extend_from_slice(&values[row].wrapping_sub(base_value).to_le_bytes());
    }
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

    // The double prefix sum carries two running values, so a restart entry
    // holds the residual and the first difference reached at the row before
    // its boundary
    let shift = choose_restart_shift(row_count, packed_bytes, 16);
    let restarts = restart_count(row_count, shift);
    let mut out = Vec::with_capacity(12 + restarts * 16 + packed_bytes);
    out.extend_from_slice(&base.to_le_bytes());
    out.push(bit_width);
    out.push(if restarts > 0 {
        FLAG_DELTA_OF_DELTA | FLAG_RESTART
    } else {
        FLAG_DELTA_OF_DELTA
    });
    out.push(shift as u8);
    out.push(0);
    for k in 1..=restarts {
        let row = (k << shift) - 1;
        let residual = values[row].wrapping_sub(base);
        let prior = values[row - 1].wrapping_sub(base);
        out.extend_from_slice(&residual.to_le_bytes());
        out.extend_from_slice(&residual.wrapping_sub(prior).to_le_bytes());
    }
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
    // Restart values for the two cumulative streams, mirroring the narrow path
    let entry = wide_restart_entry(flags);
    let shift = choose_restart_shift(row_count, packed_bytes, entry);
    let restarts = if flags & (FLAG_DELTA | FLAG_DELTA_OF_DELTA) != 0 {
        restart_count(row_count, shift)
    } else {
        0
    };
    let table_bytes = restarts * entry;
    let mut out = vec![0u8; WIDE_HEADER_SIZE + table_bytes + packed_bytes];
    out[0..16].copy_from_slice(&base.to_le_bytes());
    out[16] = bit_width;
    out[17] = if restarts > 0 {
        flags | FLAG_RESTART
    } else {
        flags
    };
    out[18] = shift as u8;
    for k in 1..=restarts {
        let row = (k << shift) - 1;
        let o = WIDE_HEADER_SIZE + (k - 1) * entry;
        let residual = residuals[row];
        out[o..o + 16].copy_from_slice(&residual.to_le_bytes());
        if entry == 32 {
            let prior = residuals[row - 1];
            out[o + 16..o + 32].copy_from_slice(&residual.wrapping_sub(prior).to_le_bytes());
        }
    }
    let packed = &mut out[WIDE_HEADER_SIZE + table_bytes..];
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
///
/// A whole column is the range of every row. The range kernel reads each
/// layout from its own bit positions and fills a pooled buffer in place,
/// so the column decode is that call at row zero rather than a second set
/// of kernels holding the residuals in an array of their own first
fn decode_wide(encoded: &[u8], row_count: usize) -> Result<Vec<u8>> {
    if row_count == 0 {
        return Ok(Vec::new());
    }
    decode_range_wide(encoded, row_count, 0, row_count)
}

/// Evaluates a predicate on the 16-byte wide format. Decodes then compares
/// numerically as u128 (same unsigned-pattern semantics the 8-byte path uses).
fn eval_predicate_wide(encoded: &[u8], row_count: usize, predicate: &Predicate) -> Result<Vec<u8>> {
    let decoded = decode_wide(encoded, row_count)?;
    let mask = match predicate {
        Predicate::Range { low, high } => {
            let lo = match *low {
                Some(b) => read_u128_bound(b),
                None => 0,
            };
            let hi = match *high {
                Some(b) => read_u128_bound(b),
                None => u128::MAX,
            };
            crate::encoding::bitmask_from_rows(row_count, |i| {
                let v = read_u128_le(&decoded, i * 16);
                v >= lo && v <= hi
            })
        }
        Predicate::Equality(target) => {
            let t = read_u128_bound(target);
            crate::encoding::bitmask_from_rows(row_count, |i| read_u128_le(&decoded, i * 16) == t)
        }
        Predicate::In(values) => {
            let targets: Vec<u128> = values.iter().map(|v| read_u128_bound(v)).collect();
            crate::encoding::bitmask_from_rows(row_count, |i| {
                targets.contains(&read_u128_le(&decoded, i * 16))
            })
        }
    };
    // The values were read to answer the predicate and nothing holds them
    // after, so the buffer goes back for the next decode on this thread
    super::scratch::give_back(decoded);
    Ok(mask)
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

    /// The shapes the ranged-decode property test relies on must actually
    /// select the layouts they are named for. A shape that quietly fell back
    /// to a different layout would make that test agree for the wrong reason.
    #[test]
    fn test_shapes_select_the_layouts_they_are_named_for() {
        let enc = FastLanesEncoding;
        const ROWS: usize = 4100;

        let ascending: Vec<u8> = (0..ROWS)
            .flat_map(|i| ((i as i64) * 3 + (i as i64) / 50).to_le_bytes())
            .collect();
        let encoded = enc.encode(&ascending, ROWS, 8).unwrap();
        assert_eq!(encoded[9] & FLAG_DELTA, FLAG_DELTA, "ascending is delta");
        assert_eq!(
            encoded[9] & FLAG_RESTART,
            FLAG_RESTART,
            "a delta stream this long carries restart points"
        );

        let quadratic: Vec<u8> = (0..ROWS)
            .flat_map(|i| (((i * i) / 3 + i) as i64).to_le_bytes())
            .collect();
        let encoded = enc.encode(&quadratic, ROWS, 8).unwrap();
        assert_eq!(
            encoded[9] & FLAG_DELTA_OF_DELTA,
            FLAG_DELTA_OF_DELTA,
            "quadratic growth is delta-of-delta"
        );
        assert_eq!(encoded[9] & FLAG_RESTART, FLAG_RESTART);

        let bursty: Vec<u8> = (0..ROWS)
            .flat_map(|i| {
                let v = if (i / 1024) % 3 == 1 {
                    (i as i64) * 1_000_003
                } else {
                    (i % 7) as i64
                };
                v.to_le_bytes()
            })
            .collect();
        let encoded = enc.encode(&bursty, ROWS, 8).unwrap();
        assert_eq!(
            encoded[9] & FLAG_MINIBLOCK,
            FLAG_MINIBLOCK,
            "a burst confined to whole blocks is per-mini-block"
        );

        let with_outliers: Vec<u8> = (0..ROWS)
            .flat_map(|i| {
                let v = if i % 97 == 0 {
                    1i64 << 40
                } else {
                    (i % 13) as i64
                };
                v.to_le_bytes()
            })
            .collect();
        let encoded = enc.encode(&with_outliers, ROWS, 8).unwrap();
        assert_eq!(
            encoded[9] & FLAG_PFOR,
            FLAG_PFOR,
            "scattered outliers are patched frame of reference"
        );

        let wide_ascending: Vec<u8> = (0..ROWS)
            .flat_map(|i| ((i as i128) * 7 + (i as i128) / 40).to_le_bytes())
            .collect();
        let encoded = enc.encode(&wide_ascending, ROWS, 16).unwrap();
        assert_ne!(
            encoded[17] & (FLAG_DELTA | FLAG_DELTA_OF_DELTA),
            0,
            "an ascending 16-byte column is cumulative"
        );
        assert_eq!(encoded[17] & FLAG_RESTART, FLAG_RESTART);
    }

    /// A range decode of a cumulative column resumes at the restart boundary
    /// at or before it, so the rows replayed ahead of the first requested one
    /// are bounded by the restart spacing rather than by the segment length.
    #[test]
    fn test_cumulative_range_replays_at_most_one_restart_spacing() {
        let enc = FastLanesEncoding;
        const ROWS: usize = 40_000;
        let values: Vec<u64> = (0..ROWS as u64).map(|i| i * 3 + i / 50).collect();
        let data: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
        let encoded = enc.encode(&data, ROWS, 8).unwrap();
        assert_eq!(encoded[9] & FLAG_RESTART, FLAG_RESTART);

        let shift = encoded[10] as u32;
        let table_end = narrow_packed_offset(&encoded, encoded[9], ROWS);
        let restart = Some((&encoded[12..table_end], shift));
        let start = ROWS - 3;
        let (_, resume) =
            seed_narrow_restart::<1>(restart, start).expect("a boundary covers the tail");
        assert!(resume > 0, "the tail resumes past the head of the stream");
        assert!(
            start - resume < (1usize << shift),
            "replayed {} rows, more than the {} row spacing",
            start - resume,
            1usize << shift
        );

        let ranged = enc.decode_range(&encoded, ROWS, 8, start, ROWS).unwrap();
        let got: Vec<u64> = ranged
            .chunks_exact(8)
            .map(|c| u64::from_le_bytes(c.try_into().unwrap()))
            .collect();
        assert_eq!(got, values[start..]);
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

    /// A wide constant-step column whose low 64 bits wrap partway through,
    /// which is where the high half of the value moves. Checked whole and
    /// over ranges that open before, on and after each wrap
    #[test]
    fn a_wide_constant_step_carries_across_every_wrap_of_its_low_half() {
        let enc = FastLanesEncoding;
        // Three wraps inside the column, with the first one four rows in
        let step: i128 = (u64::MAX as i128 + 1) / 4;
        let first: i128 = (u64::MAX as i128) - step + 1;
        let values: Vec<i128> = (0..40).map(|i| first + i * step).collect();
        let encoded = enc.encode(&pack_i128(&values), values.len(), 16).unwrap();
        assert_eq!(encoded[17] & FLAG_CONST_STEP, FLAG_CONST_STEP);

        let decoded = enc.decode(&encoded, values.len(), 16).unwrap();
        assert_eq!(unpack_i128(&decoded), values, "the column decodes whole");

        for start in 0..values.len() {
            for end in (start + 1)..=values.len() {
                let part = enc
                    .decode_range(&encoded, values.len(), 16, start, end)
                    .unwrap();
                assert_eq!(
                    unpack_i128(&part),
                    values[start..end],
                    "rows {start}..{end} decode to what the column holds"
                );
            }
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
