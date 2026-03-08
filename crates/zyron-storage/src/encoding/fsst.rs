//! FSST (Finite State Symbol Table) encoding for string columns.
//!
//! Builds a symbol table of frequent 1-8 byte substrings from a sample,
//! then encodes strings by replacing substrings with 1-byte symbol codes.
//! Uses delta-encoded bit-packed offsets instead of u32 per row to keep
//! the offset array compact.
//!
//! Based on FSST (VLDB 2020), adapted for Zyron's columnar format.

use crate::encoding::{Encoding, EncodingType, Predicate, eval_predicate_on_raw};
use zyron_common::{Result, ZyronError};

pub struct FsstEncoding;

/// Maximum symbol length (bytes).
const MAX_SYMBOL_LEN: usize = 8;

/// Number of symbol table entries (codes 0..255).
const SYMBOL_TABLE_SIZE: usize = 256;

/// Escape byte: the next byte in compressed output is a literal, not a symbol code.
const ESCAPE_BYTE: u8 = 0xFF;

/// Number of iterative refinement rounds for symbol table building.
const REFINEMENT_ROUNDS: usize = 5;

/// Encoded format:
///   [0..4]     row_count: u32
///   [4..8]     value_size: u32 (original fixed value_size, 0 for variable-length)
///   [8..12]    symbol_count: u32 (number of symbols in table, max 255)
///   [12..13]   offset_bit_width: u8 (bits per delta-encoded offset)
///   [13..14]   reserved: u8
///   [14..14+symbol_table_bytes]  symbol table:
///       Per symbol: length(u8) + bytes(1..8)
///   [symbol_table_end..+packed_offsets_bytes]  bit-packed delta-encoded offsets
///   [offsets_end..]  compressed string data
impl Encoding for FsstEncoding {
    fn encoding_type(&self) -> EncodingType {
        EncodingType::Fsst
    }

    fn encode(&self, data: &[u8], row_count: usize, value_size: usize) -> Result<Vec<u8>> {
        if row_count == 0 {
            let mut out = Vec::with_capacity(14);
            out.extend_from_slice(&0u32.to_le_bytes());
            out.extend_from_slice(&(value_size as u32).to_le_bytes());
            out.extend_from_slice(&0u32.to_le_bytes());
            out.push(0);
            out.push(0);
            return Ok(out);
        }

        // Extract individual strings from the data
        let strings = extract_strings(data, row_count, value_size)?;

        // Build symbol table with iterative refinement on a larger sample
        let sampleSize = strings.len().min(1024);
        let symbolTable = build_symbol_table_iterative(&strings[..sampleSize]);

        // Serialize symbol table
        let mut tableBytes = Vec::new();
        let mut symbolCount = 0u32;
        for sym in &symbolTable {
            if sym.is_empty() {
                break;
            }
            tableBytes.push(sym.len() as u8);
            tableBytes.extend_from_slice(sym);
            symbolCount += 1;
        }

        // Build hash index once, then compress all strings using it
        let symbolIndex = build_symbol_index(&symbolTable, symbolCount as usize);
        let maxSymLen = max_symbol_length(&symbolTable, symbolCount as usize);

        let mut compressedData = Vec::new();
        let mut rowLengths = Vec::with_capacity(row_count);

        for s in &strings {
            let startLen = compressedData.len();
            compress_string_with_index(s, &symbolIndex, maxSymLen, &mut compressedData);
            rowLengths.push((compressedData.len() - startLen) as u32);
        }

        // Delta-encode offsets as cumulative lengths, then bit-pack
        let maxLen = rowLengths.iter().copied().max().unwrap_or(0);
        let offsetBitWidth = if maxLen == 0 {
            1u8
        } else {
            (32 - maxLen.leading_zeros()) as u8
        };

        let totalOffsetBits = row_count as u64 * offsetBitWidth as u64;
        let packedOffsetBytes = (totalOffsetBits as usize).div_ceil(8);
        let mut packedOffsets = vec![0u8; packedOffsetBytes];

        for (i, &len) in rowLengths.iter().enumerate() {
            pack_bits(
                &mut packedOffsets,
                i as u64 * offsetBitWidth as u64,
                len as u64,
                offsetBitWidth,
            );
        }

        // Build output
        let total = 14 + tableBytes.len() + packedOffsetBytes + compressedData.len();
        let mut out = Vec::with_capacity(total);

        out.extend_from_slice(&(row_count as u32).to_le_bytes());
        out.extend_from_slice(&(value_size as u32).to_le_bytes());
        out.extend_from_slice(&symbolCount.to_le_bytes());
        out.push(offsetBitWidth);
        out.push(0); // reserved
        out.extend_from_slice(&tableBytes);
        out.extend_from_slice(&packedOffsets);
        out.extend_from_slice(&compressedData);

        Ok(out)
    }

    fn decode(&self, encoded: &[u8], row_count: usize, value_size: usize) -> Result<Vec<u8>> {
        if row_count == 0 {
            return Ok(Vec::new());
        }

        if encoded.len() < 14 {
            return Err(ZyronError::DecodingFailed(
                "FSST header too short".to_string(),
            ));
        }

        let storedRowCount =
            u32::from_le_bytes([encoded[0], encoded[1], encoded[2], encoded[3]]) as usize;
        let storedValueSize =
            u32::from_le_bytes([encoded[4], encoded[5], encoded[6], encoded[7]]) as usize;
        let symbolCount =
            u32::from_le_bytes([encoded[8], encoded[9], encoded[10], encoded[11]]) as usize;
        let offsetBitWidth = encoded[12];

        if storedRowCount != row_count {
            return Err(ZyronError::DecodingFailed(format!(
                "FSST row count mismatch: stored {}, expected {}",
                storedRowCount, row_count
            )));
        }

        if storedValueSize != value_size {
            return Err(ZyronError::DecodingFailed(format!(
                "FSST value_size mismatch: stored {}, expected {}",
                storedValueSize, value_size
            )));
        }

        // Read symbol table into a packed lookup table for cache-friendly data access.
        // 256 entries x 16 bytes = 4KB total. Symbol lengths stored in a separate
        // 256-byte array that fits in 4 cache lines for L1-hot length lookups.
        let mut pos = 14;
        let mut symbolTable = [[0u8; 16]; SYMBOL_TABLE_SIZE];
        // Separate length array (256 bytes = 4 cache lines) for L1-hot length lookups.
        // Avoids 16-byte stride access pattern of the packed symbol table when only
        // the length is needed.
        let mut symbolLengths = [0u8; SYMBOL_TABLE_SIZE];

        for code in 0..symbolCount {
            if pos >= encoded.len() {
                return Err(ZyronError::DecodingFailed(
                    "FSST symbol table truncated".to_string(),
                ));
            }
            let len = encoded[pos] as usize;
            pos += 1;
            if pos + len > encoded.len() {
                return Err(ZyronError::DecodingFailed(
                    "FSST symbol data truncated".to_string(),
                ));
            }
            // Layout: [0..8] = symbol data
            symbolTable[code][..len].copy_from_slice(&encoded[pos..pos + len]);
            symbolLengths[code] = len as u8;
            pos += len;
        }

        // Read bit-packed delta offsets (per-row compressed lengths)
        let totalOffsetBits = row_count as u64 * offsetBitWidth as u64;
        let packedOffsetBytes = (totalOffsetBits as usize).div_ceil(8);
        let offsetsStart = pos;
        let offsetsEnd = offsetsStart + packedOffsetBytes;

        if offsetsEnd > encoded.len() {
            return Err(ZyronError::DecodingFailed(
                "FSST packed offsets truncated".to_string(),
            ));
        }

        let packedOffsets = &encoded[offsetsStart..offsetsEnd];
        let compressedStart = offsetsEnd;
        let compressed = &encoded[compressedStart..];

        // Pre-allocate output buffer. Decompress directly into it without
        // per-string Vec allocation.
        let outSize = row_count * value_size;
        let mut out: Vec<u8> = Vec::with_capacity(outSize);
        unsafe {
            out.set_len(outSize);
        }
        let outPtr = out.as_mut_ptr();
        let compPtr = compressed.as_ptr();
        let compLen = compressed.len();
        let offsetPtr = packedOffsets.as_ptr();
        let offsetPtrLen = packedOffsets.len();
        let symTablePtr = symbolTable.as_ptr();
        let symLenPtr = symbolLengths.as_ptr();
        let mut cursor = 0usize;
        let offsetMask: u64 = if offsetBitWidth >= 64 {
            u64::MAX
        } else {
            (1u64 << offsetBitWidth) - 1
        };

        for i in 0..row_count {
            // Unpack row length using unaligned u64 read
            let bitOffset = i as u64 * offsetBitWidth as u64;
            let byteIdx = (bitOffset >> 3) as usize;
            let bitIdx = (bitOffset & 7) as u32;

            let rowLen = if byteIdx + 8 <= offsetPtrLen {
                let raw = unsafe { (offsetPtr.add(byteIdx) as *const u64).read_unaligned() };
                ((raw >> bitIdx) & offsetMask) as usize
            } else {
                let mut buf = [0u8; 8];
                let avail = offsetPtrLen.saturating_sub(byteIdx).min(8);
                buf[..avail].copy_from_slice(&packedOffsets[byteIdx..byteIdx + avail]);
                ((u64::from_le_bytes(buf) >> bitIdx) & offsetMask) as usize
            };

            let compEnd = cursor + rowLen;
            if compEnd > compLen {
                return Err(ZyronError::DecodingFailed(
                    "FSST compressed data out of bounds".to_string(),
                ));
            }

            // Decompress using raw pointers. The packed symbol table yields both
            // data (u64 at offset 0) and length (u8 at offset 8) from a single
            // cache line access per symbol.
            let outStart = i * value_size;
            let mut writePos = outStart;
            let writeEnd = outStart + value_size;
            let mut j = cursor;

            while j < compEnd && writePos < writeEnd {
                let byte = unsafe { *compPtr.add(j) };
                j += 1;

                if byte != ESCAPE_BYTE {
                    let code = byte as usize;
                    if code >= symbolCount {
                        return Err(ZyronError::DecodingFailed(format!(
                            "FSST symbol code {} out of range (table size {})",
                            code, symbolCount
                        )));
                    }
                    // Length from separate L1-hot array, data from packed table
                    let symLen = unsafe { *symLenPtr.add(code) } as usize;
                    let entry = unsafe { &*symTablePtr.add(code) };
                    // u64 write covers all symbols (max 8 bytes).
                    if writePos + 8 <= outSize {
                        unsafe {
                            let symWord = (entry.as_ptr() as *const u64).read_unaligned();
                            (outPtr.add(writePos) as *mut u64).write_unaligned(symWord);
                        }
                    } else {
                        let copyLen = symLen.min(writeEnd - writePos);
                        unsafe {
                            std::ptr::copy_nonoverlapping(
                                entry.as_ptr(),
                                outPtr.add(writePos),
                                copyLen,
                            );
                        }
                    }
                    writePos += symLen;
                } else {
                    if j >= compEnd {
                        return Err(ZyronError::DecodingFailed(
                            "FSST escape byte at end of compressed data".to_string(),
                        ));
                    }
                    unsafe {
                        *outPtr.add(writePos) = *compPtr.add(j);
                    }
                    writePos += 1;
                    j += 1;
                }
            }

            cursor = compEnd;
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

        // For equality predicates, compress the search term with the same
        // symbol table and compare compressed bytes directly.
        if let Predicate::Equality(target) = predicate
            && encoded.len() >= 14
        {
            let symbolCount =
                u32::from_le_bytes([encoded[8], encoded[9], encoded[10], encoded[11]]) as usize;
            let offsetBitWidth = encoded[12];

            // Read symbol table
            let mut pos = 14;
            let mut symbolTable: Vec<Vec<u8>> = Vec::with_capacity(symbolCount);
            let mut tableOk = true;
            for _ in 0..symbolCount {
                if pos >= encoded.len() {
                    tableOk = false;
                    break;
                }
                let len = encoded[pos] as usize;
                pos += 1;
                if pos + len > encoded.len() {
                    tableOk = false;
                    break;
                }
                symbolTable.push(encoded[pos..pos + len].to_vec());
                pos += len;
            }

            if tableOk {
                // Compress the search term with the same symbol table
                let mut compressedTarget = Vec::new();
                compress_string(target, &symbolTable, symbolCount, &mut compressedTarget);

                // Read bit-packed offsets
                let totalOffsetBits = row_count as u64 * offsetBitWidth as u64;
                let packedOffsetBytes = (totalOffsetBits as usize).div_ceil(8);
                let offsetsStart = pos;
                let offsetsEnd = offsetsStart + packedOffsetBytes;

                if offsetsEnd <= encoded.len() {
                    let packedOffsets = &encoded[offsetsStart..offsetsEnd];
                    let compressedStart = offsetsEnd;
                    let compressed = &encoded[compressedStart..];

                    let bitmaskLen = row_count.div_ceil(8);
                    let mut bitmask = vec![0u8; bitmaskLen];
                    let mut cursor = 0usize;

                    for i in 0..row_count {
                        let len = unpack_bits(
                            packedOffsets,
                            i as u64 * offsetBitWidth as u64,
                            offsetBitWidth,
                        ) as usize;
                        let end = cursor + len;

                        if end <= compressed.len() {
                            let rowCompressed = &compressed[cursor..end];
                            if rowCompressed == compressedTarget.as_slice() {
                                bitmask[i / 8] |= 1 << (i % 8);
                            }
                        }
                        cursor = end;
                    }

                    return Ok(bitmask);
                }
            }
        }

        // Fall back to decode-then-evaluate for range and IN predicates
        let decoded = self.decode(encoded, row_count, value_size)?;
        eval_predicate_on_raw(&decoded, row_count, value_size, predicate)
    }
}

/// Extracts individual string values from contiguous fixed-size data.
fn extract_strings(data: &[u8], row_count: usize, value_size: usize) -> Result<Vec<&[u8]>> {
    if value_size == 0 {
        return Err(ZyronError::EncodingFailed(
            "FSST requires non-zero value_size".to_string(),
        ));
    }

    if data.len() < row_count * value_size {
        return Err(ZyronError::EncodingFailed(
            "data shorter than expected for FSST encoding".to_string(),
        ));
    }

    let mut strings = Vec::with_capacity(row_count);
    for i in 0..row_count {
        strings.push(&data[i * value_size..(i + 1) * value_size]);
    }
    Ok(strings)
}

/// Builds a symbol table using iterative refinement.
/// Each round compresses the sample with the current table, then rebuilds
/// the table from the compressed output's escape sequences, capturing
/// multi-byte patterns that span previous symbols.
fn build_symbol_table_iterative(sample: &[&[u8]]) -> Vec<Vec<u8>> {
    let mut table = build_symbol_table_from_raw(sample);

    for _ in 0..REFINEMENT_ROUNDS {
        // Compress the sample with the current table
        let symbolCount = table
            .iter()
            .position(|s| s.is_empty())
            .unwrap_or(table.len());
        let mut escapedSegments: Vec<Vec<u8>> = Vec::new();

        for s in sample {
            let mut compressed = Vec::new();
            compress_string(s, &table, symbolCount, &mut compressed);

            // Collect runs of escaped (literal) bytes as candidates for new symbols
            let mut segment = Vec::new();
            let mut i = 0;
            while i < compressed.len() {
                if compressed[i] == ESCAPE_BYTE && i + 1 < compressed.len() {
                    segment.push(compressed[i + 1]);
                    i += 2;
                } else {
                    if segment.len() >= 2 {
                        escapedSegments.push(segment.clone());
                    }
                    segment.clear();
                    i += 1;
                }
            }
            if segment.len() >= 2 {
                escapedSegments.push(segment);
            }
        }

        // Build new candidate table from raw sample plus escape patterns
        let newTable = build_symbol_table_from_raw(sample);

        // Merge: keep symbols from both tables, ranked by frequency
        let mut merged = merge_symbol_tables(&table, &newTable, sample);

        // Add multi-byte escape patterns as new symbols
        let mut escapeFreq: hashbrown::HashMap<Vec<u8>, usize> = hashbrown::HashMap::new();
        for seg in &escapedSegments {
            for len in 2..=MAX_SYMBOL_LEN.min(seg.len()) {
                for start in 0..=seg.len() - len {
                    let substr = seg[start..start + len].to_vec();
                    if !substr.contains(&ESCAPE_BYTE) {
                        *escapeFreq.entry(substr).or_insert(0) += 1;
                    }
                }
            }
        }

        let mut escapeCandidates: Vec<(Vec<u8>, usize)> = escapeFreq.into_iter().collect();
        escapeCandidates.sort_by(|a, b| {
            let ba = a.1 * (a.0.len() - 1);
            let bb = b.1 * (b.0.len() - 1);
            bb.cmp(&ba)
        });

        for (sym, _) in escapeCandidates {
            if merged.len() >= SYMBOL_TABLE_SIZE - 1 {
                break;
            }
            if !merged.contains(&sym) {
                merged.push(sym);
            }
        }

        table = merged;
    }

    table
}

/// Builds a symbol table from raw input strings using frequency analysis.
fn build_symbol_table_from_raw(sample: &[&[u8]]) -> Vec<Vec<u8>> {
    let mut freq: hashbrown::HashMap<Vec<u8>, usize> = hashbrown::HashMap::new();

    // Count substring frequencies for lengths 1..=MAX_SYMBOL_LEN
    for s in sample {
        for len in 1..=MAX_SYMBOL_LEN.min(s.len()) {
            for start in 0..=s.len() - len {
                let substr = &s[start..start + len];
                *freq.entry(substr.to_vec()).or_insert(0) += 1;
            }
        }
    }

    // Sort by benefit: frequency * (length - 1) gives bytes saved
    let mut candidates: Vec<(Vec<u8>, usize)> = freq
        .into_iter()
        .filter(|(sym, count)| {
            let benefit = *count * (sym.len().saturating_sub(1));
            benefit > 0 && !sym.contains(&ESCAPE_BYTE)
        })
        .collect();

    candidates.sort_by(|a, b| {
        let benefitA = a.1 * (a.0.len() - 1);
        let benefitB = b.1 * (b.0.len() - 1);
        benefitB.cmp(&benefitA)
    });

    // Take top symbols (max 254 to reserve 0xFF as escape)
    let maxSymbols = SYMBOL_TABLE_SIZE - 1;
    let mut table: Vec<Vec<u8>> = Vec::with_capacity(maxSymbols);

    for (sym, _) in candidates.into_iter().take(maxSymbols) {
        table.push(sym);
    }

    table
}

/// Merges two symbol tables, keeping the best symbols by benefit on the sample.
fn merge_symbol_tables(a: &[Vec<u8>], b: &[Vec<u8>], sample: &[&[u8]]) -> Vec<Vec<u8>> {
    let mut freq: hashbrown::HashMap<Vec<u8>, usize> = hashbrown::HashMap::new();

    // Count actual occurrences in sample for all candidate symbols
    let mut allSymbols: Vec<&Vec<u8>> = Vec::new();
    for s in a {
        if !s.is_empty() {
            allSymbols.push(s);
        }
    }
    for s in b {
        if !s.is_empty() && !a.contains(s) {
            allSymbols.push(s);
        }
    }

    for sym in &allSymbols {
        let mut count = 0;
        for s in sample {
            let mut pos = 0;
            while pos + sym.len() <= s.len() {
                if &s[pos..pos + sym.len()] == sym.as_slice() {
                    count += 1;
                    pos += sym.len();
                } else {
                    pos += 1;
                }
            }
        }
        if count > 0 {
            freq.insert((*sym).clone(), count);
        }
    }

    let mut ranked: Vec<(Vec<u8>, usize)> = freq.into_iter().collect();
    ranked.sort_by(|a, b| {
        let ba = a.1 * (a.0.len() - 1);
        let bb = b.1 * (b.0.len() - 1);
        bb.cmp(&ba)
    });

    let maxSymbols = SYMBOL_TABLE_SIZE - 1;
    ranked
        .into_iter()
        .take(maxSymbols)
        .map(|(s, _)| s)
        .collect()
}

/// Builds a hash-based lookup index from the symbol table for O(1) substring matching.
/// Returns a HashMap keyed by (length, bytes) for each symbol, mapping to its code index.
fn build_symbol_index(
    symbolTable: &[Vec<u8>],
    symbolCount: usize,
) -> hashbrown::HashMap<Vec<u8>, u8> {
    let mut index = hashbrown::HashMap::with_capacity(symbolCount);
    for (code, sym) in symbolTable.iter().enumerate().take(symbolCount) {
        if !sym.is_empty() {
            index.insert(sym.clone(), code as u8);
        }
    }
    index
}

/// Computes the maximum symbol length in the table.
fn max_symbol_length(symbolTable: &[Vec<u8>], symbolCount: usize) -> usize {
    symbolTable
        .iter()
        .take(symbolCount)
        .map(|s| s.len())
        .max()
        .unwrap_or(0)
}

/// Compresses a single string using the symbol table.
/// Uses greedy longest-match with hash-based lookup: at each position,
/// try substrings from longest to shortest until a symbol match is found.
/// If no symbol matches, emit ESCAPE_BYTE + literal byte.
fn compress_string(
    input: &[u8],
    symbol_table: &[Vec<u8>],
    symbol_count: usize,
    output: &mut Vec<u8>,
) {
    let index = build_symbol_index(symbol_table, symbol_count);
    let maxLen = max_symbol_length(symbol_table, symbol_count);
    compress_string_with_index(input, &index, maxLen, output);
}

/// Inner compression using a pre-built hash index. Avoids rebuilding the
/// index when compressing multiple strings with the same symbol table.
fn compress_string_with_index(
    input: &[u8],
    index: &hashbrown::HashMap<Vec<u8>, u8>,
    maxSymLen: usize,
    output: &mut Vec<u8>,
) {
    let mut i = 0;
    while i < input.len() {
        let mut matched = false;
        let remaining = input.len() - i;
        let tryLen = remaining.min(maxSymLen);

        // Try longest substrings first for greedy longest-match
        for len in (1..=tryLen).rev() {
            let substr = &input[i..i + len];
            if let Some(&code) = index.get(substr) {
                output.push(code);
                i += len;
                matched = true;
                break;
            }
        }

        if !matched {
            output.push(ESCAPE_BYTE);
            output.push(input[i]);
            i += 1;
        }
    }
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
    let bytesNeeded = (totalBits as usize).div_ceil(8);

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
    fn test_roundtrip_fixed_strings() {
        let enc = FsstEncoding;
        // 10 strings, each 8 bytes, padded with zeros
        let strings = [
            b"hello\0\0\0",
            b"world\0\0\0",
            b"hello\0\0\0",
            b"test!\0\0\0",
            b"hello\0\0\0",
            b"world\0\0\0",
            b"hello\0\0\0",
            b"data!\0\0\0",
            b"hello\0\0\0",
            b"world\0\0\0",
        ];

        let mut data = Vec::new();
        for s in &strings {
            data.extend_from_slice(*s);
        }

        let encoded = enc.encode(&data, 10, 8).unwrap();
        let decoded = enc.decode(&encoded, 10, 8).unwrap();
        assert_eq!(decoded, data);
    }

    #[test]
    fn test_roundtrip_repetitive() {
        let enc = FsstEncoding;
        let pattern = b"abcdefgh";
        let mut data = Vec::new();
        for _ in 0..100 {
            data.extend_from_slice(pattern);
        }

        let encoded = enc.encode(&data, 100, 8).unwrap();
        let decoded = enc.decode(&encoded, 100, 8).unwrap();
        assert_eq!(decoded, data);
    }

    #[test]
    fn test_empty() {
        let enc = FsstEncoding;
        let encoded = enc.encode(&[], 0, 8).unwrap();
        let decoded = enc.decode(&encoded, 0, 8).unwrap();
        assert!(decoded.is_empty());
    }

    #[test]
    fn test_single_row() {
        let enc = FsstEncoding;
        let data = b"testdata";

        let encoded = enc.encode(data, 1, 8).unwrap();
        let decoded = enc.decode(&encoded, 1, 8).unwrap();
        assert_eq!(&decoded, data);
    }

    #[test]
    fn test_all_unique_bytes() {
        let enc = FsstEncoding;
        // Each row has unique bytes, minimal encoding opportunity
        let mut data = Vec::new();
        for i in 0..10u8 {
            let row: [u8; 4] = [i * 4, i * 4 + 1, i * 4 + 2, i * 4 + 3];
            data.extend_from_slice(&row);
        }

        let encoded = enc.encode(&data, 10, 4).unwrap();
        let decoded = enc.decode(&encoded, 10, 4).unwrap();
        assert_eq!(decoded, data);
    }

    #[test]
    fn test_compression_ratio() {
        let enc = FsstEncoding;
        let n = 100_000usize;
        let mut data = Vec::with_capacity(n * 32);
        for i in 0..n {
            let base = format!("row_{:05}_padding_abcdefgh", i % 1000);
            let mut val = [0u8; 32];
            let bytes = base.as_bytes();
            val[..bytes.len().min(32)].copy_from_slice(&bytes[..bytes.len().min(32)]);
            data.extend_from_slice(&val);
        }

        let encoded = enc.encode(&data, n, 32).unwrap();
        let ratio = data.len() as f64 / encoded.len() as f64;
        assert!(ratio > 4.0, "expected 4:1+ ratio, got {:.1}:1", ratio);

        let decoded = enc.decode(&encoded, n, 32).unwrap();
        assert_eq!(decoded, data);
    }
}
