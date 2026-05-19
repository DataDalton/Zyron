//! Dictionary encoding for low-cardinality columns.
//! Builds a sorted dictionary of distinct values, replaces each row value
//! with a bit-packed code index. Code width = ceil(log2(dict_count)) bits,
//! so 10 distinct values use 4-bit codes instead of 32-bit codes.
//! Predicate evaluation resolves the search term to a code via binary search
//! on the dictionary, then scans the code array without decoding.

use crate::encoding::{Encoding, EncodingType, Predicate, slice_rows, varlen_pack};
use std::collections::{HashMap, HashSet};
use zyron_common::{Result, ZyronError};

pub struct DictionaryEncoding;

/// Fixed-width encoded format (`value_size > 0`):
///   [0..4]     value_size: u32
///   [4..8]     dict_count: u32
///   [8..8+dict_count*value_size]  dictionary entries (sorted)
///   [8+dict_count*value_size..]   bit-packed code array (ceil(log2(dict_count)) bits per row)
///
/// Variable-length encoded format (`value_size == 0`, signalled by the stored
/// value_size being 0):
///   [0..4]     value_size: u32 = 0
///   [4..8]     dict_count: u32
///   [8..12]    dict_blob_len: u32
///   [12 .. 12+4*(dict_count+1)]   dict offsets: u32 LE, dict_off[0]=0,
///                                 dict_off[dict_count]=dict_blob_len
///   [.. + dict_blob_len]          dict blob: sorted distinct values concatenated
///   [.. ]                         bit-packed code array (row_count codes)
/// Decode reproduces the canonical variable-length buffer. A null row is a
/// zero-length value here; the segment null bitmap stays authoritative so an
/// empty string and a null remain distinct.
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

        if value_size == 0 {
            return encode_varlen(data, row_count);
        }

        if data.len() < row_count * value_size {
            return Err(ZyronError::EncodingFailed(
                "data shorter than expected for dictionary encoding".to_string(),
            ));
        }

        // Dedup in O(n) via a hash set, then sort the distinct set once
        // (O(k log k)). The prior binary-search + Vec::insert was O(k^2) in
        // shifts on high-cardinality columns. The dictionary stays sorted, so
        // the on-disk format and decode/predicate path are unchanged.
        let mut seen: HashSet<&[u8]> = HashSet::with_capacity(row_count.min(4096));
        for i in 0..row_count {
            seen.insert(&data[i * value_size..(i + 1) * value_size]);
        }
        let mut distinct: Vec<&[u8]> = seen.into_iter().collect();
        distinct.sort_unstable();

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
        let packedCodeBytes = totalCodeBits.div_ceil(8) as usize;

        let mut out = Vec::with_capacity(8 + dictSize + packedCodeBytes);
        out.extend_from_slice(&(value_size as u32).to_le_bytes());
        out.extend_from_slice(&dictCount.to_le_bytes());

        // Write dictionary entries
        for entry in &distinct {
            out.extend_from_slice(entry);
        }

        // O(1) per-row code lookup instead of O(log k) binary search per row.
        let code_of: HashMap<&[u8], u32> = distinct
            .iter()
            .enumerate()
            .map(|(idx, v)| (*v, idx as u32))
            .collect();

        // Bit-pack the code array
        let mut packed = vec![0u8; packedCodeBytes];
        for i in 0..row_count {
            let val = &data[i * value_size..(i + 1) * value_size];
            let code = *code_of.get(val).ok_or_else(|| {
                ZyronError::EncodingFailed(
                    "value not found in dictionary during encoding".to_string(),
                )
            })?;
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

        if value_size == 0 {
            return decode_varlen(encoded, row_count);
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

        let headerValueSize =
            u32::from_le_bytes([encoded[0], encoded[1], encoded[2], encoded[3]]) as usize;
        if headerValueSize == 0 {
            return eval_predicate_varlen(encoded, row_count, predicate);
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

        let bitmaskLen = row_count.div_ceil(8);
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

/// Code bit width for a dictionary of `dict_count` entries.
#[inline]
fn code_bit_width(dict_count: usize) -> u8 {
    if dict_count <= 1 {
        1u8
    } else {
        (32 - (dict_count as u32 - 1).leading_zeros()) as u8
    }
}

/// Encodes the canonical variable-length buffer as a variable-length
/// dictionary. Distinct values are stored once, sorted, behind their own
/// u32 offset array; each row becomes a bit-packed code.
fn encode_varlen(data: &[u8], row_count: usize) -> Result<Vec<u8>> {
    let rows = slice_rows(data, row_count, 0)?;

    let mut seen: HashSet<&[u8]> = HashSet::with_capacity(row_count.min(4096));
    for r in &rows {
        seen.insert(*r);
    }
    let mut distinct: Vec<&[u8]> = seen.into_iter().collect();
    distinct.sort_unstable();

    if distinct.len() > u32::MAX as usize {
        return Err(ZyronError::EncodingFailed(
            "dictionary cardinality exceeds u32 range".to_string(),
        ));
    }
    let dict_count = distinct.len();
    let code_of: HashMap<&[u8], u32> = distinct
        .iter()
        .enumerate()
        .map(|(idx, v)| (*v, idx as u32))
        .collect();

    let dict_blob_len: usize = distinct.iter().map(|v| v.len()).sum();
    if dict_blob_len > u32::MAX as usize {
        return Err(ZyronError::EncodingFailed(
            "dictionary blob exceeds u32 range".to_string(),
        ));
    }
    let code_bits = code_bit_width(dict_count);
    let packed_bytes = (row_count as u64 * code_bits as u64).div_ceil(8) as usize;
    let offsets_bytes = 4 * (dict_count + 1);

    let mut out = Vec::with_capacity(12 + offsets_bytes + dict_blob_len + packed_bytes);
    out.extend_from_slice(&0u32.to_le_bytes());
    out.extend_from_slice(&(dict_count as u32).to_le_bytes());
    out.extend_from_slice(&(dict_blob_len as u32).to_le_bytes());

    let mut cursor = 0u32;
    out.extend_from_slice(&cursor.to_le_bytes());
    for v in &distinct {
        cursor += v.len() as u32;
        out.extend_from_slice(&cursor.to_le_bytes());
    }
    for v in &distinct {
        out.extend_from_slice(v);
    }

    let mut packed = vec![0u8; packed_bytes];
    for (i, r) in rows.iter().enumerate() {
        let code = *code_of.get(*r).ok_or_else(|| {
            ZyronError::EncodingFailed("value not found in dictionary during encoding".to_string())
        })?;
        pack_bits(
            &mut packed,
            i as u64 * code_bits as u64,
            code as u64,
            code_bits,
        );
    }
    out.extend_from_slice(&packed);
    Ok(out)
}

/// Reads the variable-length dictionary container header. Returns
/// (dict_count, dict offsets, blob slice, packed code slice).
fn read_varlen_container(encoded: &[u8]) -> Result<(usize, Vec<u32>, &[u8], &[u8])> {
    if encoded.len() < 12 {
        return Err(ZyronError::DecodingFailed(
            "varlen dictionary header too short".to_string(),
        ));
    }
    let dict_count = u32::from_le_bytes([encoded[4], encoded[5], encoded[6], encoded[7]]) as usize;
    let dict_blob_len =
        u32::from_le_bytes([encoded[8], encoded[9], encoded[10], encoded[11]]) as usize;
    let offsets_start = 12;
    let offsets_end = offsets_start + 4 * (dict_count + 1);
    let blob_end = offsets_end + dict_blob_len;
    if encoded.len() < blob_end {
        return Err(ZyronError::DecodingFailed(
            "varlen dictionary blob truncated".to_string(),
        ));
    }
    let mut offsets = Vec::with_capacity(dict_count + 1);
    for i in 0..=dict_count {
        let p = offsets_start + 4 * i;
        offsets.push(u32::from_le_bytes([
            encoded[p],
            encoded[p + 1],
            encoded[p + 2],
            encoded[p + 3],
        ]));
    }
    if offsets[dict_count] as usize != dict_blob_len {
        return Err(ZyronError::DecodingFailed(
            "varlen dictionary offset array inconsistent with blob length".to_string(),
        ));
    }
    let blob = &encoded[offsets_end..blob_end];
    let packed = &encoded[blob_end..];
    Ok((dict_count, offsets, blob, packed))
}

/// Returns dictionary entry `i` as a byte slice.
#[inline]
fn varlen_dict_entry<'a>(blob: &'a [u8], offsets: &[u32], i: usize) -> &'a [u8] {
    &blob[offsets[i] as usize..offsets[i + 1] as usize]
}

/// Decodes a variable-length dictionary directly into the canonical buffer in
/// a single allocation. The prior path built Vec<&[u8]> then Vec<Option<..>>
/// then varlen_pack (three allocations + an extra pass) for a 1M-row column;
/// here the codes are unpacked once to size the output, then again to fill it,
/// writing the header, offset array, and blob in place.
fn decode_varlen(encoded: &[u8], row_count: usize) -> Result<Vec<u8>> {
    let (dict_count, offsets, blob, packed) = read_varlen_container(encoded)?;
    let code_bits = code_bit_width(dict_count);

    let code_at = |i: usize| -> Result<usize> {
        let code = unpack_bits(packed, i as u64 * code_bits as u64, code_bits) as usize;
        if code >= dict_count {
            return Err(ZyronError::DecodingFailed(format!(
                "varlen dictionary code {} out of range (dict_count={})",
                code, dict_count
            )));
        }
        Ok(code)
    };

    // Pass 1: total decoded blob length (validates every code too).
    let mut blob_total: usize = 0;
    for i in 0..row_count {
        let code = code_at(i)?;
        blob_total += varlen_dict_entry(blob, &offsets, code).len();
    }

    let header = 4 + 4 * (row_count + 1);
    let mut out = Vec::with_capacity(header + blob_total);
    out.extend_from_slice(&(row_count as u32).to_le_bytes());

    // Pass 2a: cumulative offset array.
    let mut cursor: u32 = 0;
    out.extend_from_slice(&cursor.to_le_bytes());
    for i in 0..row_count {
        let code = code_at(i)?;
        cursor += varlen_dict_entry(blob, &offsets, code).len() as u32;
        out.extend_from_slice(&cursor.to_le_bytes());
    }
    // Pass 2b: values blob.
    for i in 0..row_count {
        let code = code_at(i)?;
        out.extend_from_slice(varlen_dict_entry(blob, &offsets, code));
    }
    Ok(out)
}

/// Evaluates a predicate on a variable-length dictionary. Comparisons are
/// lexicographic over the distinct entries (correct for string and binary
/// columns), then the bit-packed codes are scanned once.
fn eval_predicate_varlen(
    encoded: &[u8],
    row_count: usize,
    predicate: &Predicate,
) -> Result<Vec<u8>> {
    let (dict_count, offsets, blob, packed) = read_varlen_container(encoded)?;
    let code_bits = code_bit_width(dict_count);
    let bitmask_len = row_count.div_ceil(8);
    let mut bitmask = vec![0u8; bitmask_len];

    // The distinct entries are stored sorted, so equality/IN resolve by
    // binary search and a range scans the contiguous matching prefix.
    let find = |target: &[u8]| -> Option<u32> {
        let mut lo = 0usize;
        let mut hi = dict_count;
        while lo < hi {
            let mid = lo + (hi - lo) / 2;
            match varlen_dict_entry(blob, &offsets, mid).cmp(target) {
                std::cmp::Ordering::Less => lo = mid + 1,
                std::cmp::Ordering::Equal => return Some(mid as u32),
                std::cmp::Ordering::Greater => hi = mid,
            }
        }
        None
    };

    let mut matching: Vec<u32> = Vec::new();
    match predicate {
        Predicate::Equality(target) => {
            if let Some(c) = find(target) {
                matching.push(c);
            }
        }
        Predicate::Range { low, high } => {
            for c in 0..dict_count {
                let entry = varlen_dict_entry(blob, &offsets, c);
                let above = low.map_or(true, |lo| entry >= lo);
                let below = high.map_or(true, |hi| entry <= hi);
                if above && below {
                    matching.push(c as u32);
                }
            }
        }
        Predicate::In(values) => {
            for t in *values {
                if let Some(c) = find(t) {
                    matching.push(c);
                }
            }
        }
    }

    if matching.is_empty() {
        return Ok(bitmask);
    }
    matching.sort_unstable();
    for i in 0..row_count {
        let code = unpack_bits(packed, i as u64 * code_bits as u64, code_bits) as u32;
        if matching.binary_search(&code).is_ok() {
            bitmask[i / 8] |= 1 << (i % 8);
        }
    }
    Ok(bitmask)
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
    let bytesNeeded = totalBits.div_ceil(8) as usize;

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
    use crate::encoding::{eval_predicate_on_raw, varlen_slice_rows};

    #[test]
    fn test_varlen_roundtrip_and_density() {
        let enc = DictionaryEncoding;
        // Low-cardinality category column: 3 distinct values over 3000 rows,
        // plus a null (zero-length) row mixed in.
        let cats: [&[u8]; 3] = [b"pending", b"shipped", b"delivered"];
        let mut vals: Vec<Option<&[u8]>> = Vec::with_capacity(3001);
        for i in 0..3000 {
            vals.push(Some(cats[i % 3]));
        }
        vals.push(None);
        let raw = varlen_pack(&vals);

        let encoded = enc.encode(&raw, vals.len(), 0).unwrap();
        // 3 codes + a null entry -> tiny dict + 2-bit codes, far under raw.
        assert!(
            encoded.len() < raw.len() / 4,
            "varlen dictionary must be dense: {} vs {}",
            encoded.len(),
            raw.len()
        );

        let decoded = enc.decode(&encoded, vals.len(), 0).unwrap();
        let rows = varlen_slice_rows(&decoded, vals.len()).unwrap();
        for i in 0..3000 {
            assert_eq!(rows[i], cats[i % 3]);
        }
        assert_eq!(rows[3000], b"");
    }

    #[test]
    fn test_varlen_predicate_matches_full_scan() {
        let enc = DictionaryEncoding;
        let cats: [&[u8]; 4] = [b"a", b"bb", b"ccc", b"dddd"];
        let vals: Vec<Option<&[u8]>> = (0..500).map(|i| Some(cats[i % 4])).collect();
        let raw = varlen_pack(&vals);
        let encoded = enc.encode(&raw, vals.len(), 0).unwrap();

        let eq = enc
            .eval_predicate(&encoded, vals.len(), 0, &Predicate::Equality(b"ccc"))
            .unwrap();
        let truth =
            eval_predicate_on_raw(&raw, vals.len(), 0, &Predicate::Equality(b"ccc")).unwrap();
        assert_eq!(eq, truth);

        let lo: &[u8] = b"bb";
        let hi: &[u8] = b"ccc";
        let rng = enc
            .eval_predicate(
                &encoded,
                vals.len(),
                0,
                &Predicate::Range {
                    low: Some(lo),
                    high: Some(hi),
                },
            )
            .unwrap();
        let rng_truth = eval_predicate_on_raw(
            &raw,
            vals.len(),
            0,
            &Predicate::Range {
                low: Some(lo),
                high: Some(hi),
            },
        )
        .unwrap();
        assert_eq!(rng, rng_truth);

        let in_set: [&[u8]; 2] = [b"a", b"dddd"];
        let isin = enc
            .eval_predicate(&encoded, vals.len(), 0, &Predicate::In(&in_set))
            .unwrap();
        let isin_truth =
            eval_predicate_on_raw(&raw, vals.len(), 0, &Predicate::In(&in_set)).unwrap();
        assert_eq!(isin, isin_truth);
    }

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
