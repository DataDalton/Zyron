//! FSST (Finite State Symbol Table) encoding for string columns.
//!
//! Builds a symbol table of frequent 1-8 byte substrings from a sample,
//! then encodes strings by replacing substrings with 1-byte symbol codes.
//! Uses delta-encoded bit-packed offsets instead of u32 per row to keep
//! the offset array compact.
//!
//! Based on FSST (VLDB 2020), adapted for Zyron's columnar format.

use crate::encoding::{
    Encoding, EncodingType, Predicate, bitmask_from_rows, eval_predicate_on_raw, slice_rows,
    varlen_pack,
};
use zyron_common::{Result, ZyronError};

pub struct FsstEncoding;

/// Maximum symbol length (bytes).
const MAX_SYMBOL_LEN: usize = 8;

/// Number of symbol table entries (codes 0..255).
const SYMBOL_TABLE_SIZE: usize = 256;

/// Escape byte: the next byte in compressed output is a literal, not a symbol code.
const ESCAPE_BYTE: u8 = 0xFF;

/// Refinement rounds a symbol table is trained for. A symbol grows by one
/// neighbour a round, so eight rounds reach the longest symbol from single
/// bytes with rounds to spare for the table to settle
const REFINEMENT_ROUNDS: usize = 8;

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

        let strings = extract_strings(data, row_count, value_size)?;
        let table = train_symbol_table(&training_sample(&strings));

        // Serialize symbol table
        let mut tableBytes = Vec::with_capacity(table.len() * (MAX_SYMBOL_LEN + 1));
        for sym in &table {
            tableBytes.push(sym.len);
            tableBytes.extend_from_slice(&sym.bytes()[..sym.len as usize]);
        }
        let symbolCount = table.len() as u32;

        let matcher = Matcher::new(&table);
        let mut compressedData = Vec::with_capacity(data.len());
        let mut rowLengths = Vec::with_capacity(row_count);
        for s in &strings {
            let startLen = compressedData.len();
            matcher.compress(s, &mut compressedData);
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

    /// A variable-length column skips to a row through its offsets, which
    /// are per-row compressed lengths, so reaching row i costs unpacking i
    /// bit fields and expands no symbols. Only the requested rows are
    /// decompressed.
    ///
    /// A fixed-width column takes the decode-and-take default: its rows are
    /// padded to one width after decompression, so the saving is the same
    /// walk with an extra copy, and the default already does the copy
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
        if value_size == 0 && encoded.len() >= 14 {
            return decode_fsst_varlen_range(encoded, row_count, start, end);
        }
        let decoded = self.decode(encoded, row_count, value_size)?;
        crate::encoding::slice_decoded(&decoded, row_count, value_size, start, end)
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

        // Variable-length: each row decompresses to its full original bytes
        // (no fixed cap); emit the canonical variable-length buffer.
        if value_size == 0 {
            return decode_fsst_varlen(encoded, row_count);
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
        // SAFETY: the symbol-expansion loop below writes every byte of `out`
        // before any read; zeroing first would memset the whole buffer only
        // to overwrite it, regressing scan decode throughput.
        #[allow(clippy::uninit_vec)]
        let mut out: Vec<u8> = {
            let mut v = Vec::with_capacity(outSize);
            unsafe { v.set_len(outSize) };
            v
        };
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

            // A table the matcher cannot hold is a table the decoder will
            // refuse too, so the comparison falls through to the decode
            if tableOk && let Some(matcher) = Matcher::from_table_bytes(&symbolTable) {
                // Compress the search term with the same symbol table
                let mut compressedTarget = Vec::new();
                matcher.compress(target, &mut compressedTarget);

                // Read bit-packed offsets
                let totalOffsetBits = row_count as u64 * offsetBitWidth as u64;
                let packedOffsetBytes = (totalOffsetBits as usize).div_ceil(8);
                let offsetsStart = pos;
                let offsetsEnd = offsetsStart + packedOffsetBytes;

                if offsetsEnd <= encoded.len() {
                    let packedOffsets = &encoded[offsetsStart..offsetsEnd];
                    let compressedStart = offsetsEnd;
                    let compressed = &encoded[compressedStart..];

                    // Row lengths are stored, not offsets, so the cursor walks
                    // forward with the rows. The mask builder visits every row
                    // once in ascending order, which is what lets the cursor
                    // live in the closure
                    let mut cursor = 0usize;
                    return Ok(bitmask_from_rows(row_count, |i| {
                        let len = unpack_bits(
                            packedOffsets,
                            i as u64 * offsetBitWidth as u64,
                            offsetBitWidth,
                        ) as usize;
                        let end = cursor + len;
                        let matched = end <= compressed.len()
                            && compressed[cursor..end] == *compressedTarget.as_slice();
                        cursor = end;
                        matched
                    }));
                }
            }
        }

        // Fall back to decode-then-evaluate for range and IN predicates
        let decoded = self.decode(encoded, row_count, value_size)?;
        eval_predicate_on_raw(&decoded, row_count, value_size, predicate)
    }
}

/// Extracts the per-row byte slices. `value_size > 0` is the fixed-width
/// layout; `value_size == 0` is the canonical variable-length buffer. FSST's
/// symbol compression operates on arbitrary-length slices either way.
fn extract_strings(data: &[u8], row_count: usize, value_size: usize) -> Result<Vec<&[u8]>> {
    slice_rows(data, row_count, value_size)
}

/// Decodes an FSST segment whose rows are variable length. Each row is
/// decompressed to its full original bytes and the result is returned as the
/// canonical variable-length buffer so the columnar read path is uniform.
fn decode_fsst_varlen(encoded: &[u8], row_count: usize) -> Result<Vec<u8>> {
    decode_fsst_varlen_range(encoded, row_count, 0, row_count)
}

/// Decompresses rows `start..end` only.
///
/// The packed array holds each row's compressed length, so the byte where a
/// row begins is the sum of the lengths before it. That sum is bit
/// unpacking and nothing else, which means skipping to a row costs the
/// offsets and never the symbol expansion, and only the requested rows are
/// decompressed
fn decode_fsst_varlen_range(
    encoded: &[u8],
    row_count: usize,
    start: usize,
    end: usize,
) -> Result<Vec<u8>> {
    let symbol_count =
        u32::from_le_bytes([encoded[8], encoded[9], encoded[10], encoded[11]]) as usize;
    let offset_bits = encoded[12];
    let mut pos = 14usize;
    let mut symbols: Vec<&[u8]> = Vec::with_capacity(symbol_count);
    for _ in 0..symbol_count {
        if pos >= encoded.len() {
            return Err(ZyronError::DecodingFailed(
                "FSST varlen symbol table truncated".to_string(),
            ));
        }
        let len = encoded[pos] as usize;
        pos += 1;
        if pos + len > encoded.len() {
            return Err(ZyronError::DecodingFailed(
                "FSST varlen symbol data truncated".to_string(),
            ));
        }
        symbols.push(&encoded[pos..pos + len]);
        pos += len;
    }
    let total_bits = row_count as u64 * offset_bits as u64;
    let packed_bytes = (total_bits as usize).div_ceil(8);
    let offsets_start = pos;
    let offsets_end = offsets_start + packed_bytes;
    if offsets_end > encoded.len() {
        return Err(ZyronError::DecodingFailed(
            "FSST varlen offsets truncated".to_string(),
        ));
    }
    let packed = &encoded[offsets_start..offsets_end];
    let compressed = &encoded[offsets_end..];
    let mask: u64 = if offset_bits >= 64 {
        u64::MAX
    } else {
        (1u64 << offset_bits) - 1
    };

    let (start, end) = crate::encoding::clamp_range(row_count, start, end);
    let mut rows: Vec<Vec<u8>> = Vec::with_capacity(end - start);
    let mut cursor = 0usize;
    for i in 0..end {
        let bit_off = i as u64 * offset_bits as u64;
        let byte_idx = (bit_off >> 3) as usize;
        let bit_idx = (bit_off & 7) as u32;
        let row_len = if byte_idx + 8 <= packed.len() {
            let raw = u64::from_le_bytes(packed[byte_idx..byte_idx + 8].try_into().unwrap());
            ((raw >> bit_idx) & mask) as usize
        } else {
            let mut b = [0u8; 8];
            let avail = packed.len().saturating_sub(byte_idx).min(8);
            b[..avail].copy_from_slice(&packed[byte_idx..byte_idx + avail]);
            ((u64::from_le_bytes(b) >> bit_idx) & mask) as usize
        };
        let row_end = cursor + row_len;
        if row_end > compressed.len() {
            return Err(ZyronError::DecodingFailed(
                "FSST varlen compressed data out of bounds".to_string(),
            ));
        }
        // Rows before the range only advance the cursor. Their symbols are
        // never expanded, which is the work this skips
        if i < start {
            cursor = row_end;
            continue;
        }
        let mut out = Vec::with_capacity(row_len);
        let mut j = cursor;
        while j < row_end {
            let byte = compressed[j];
            j += 1;
            if byte == ESCAPE_BYTE {
                if j >= row_end {
                    return Err(ZyronError::DecodingFailed(
                        "FSST varlen escape at row end".to_string(),
                    ));
                }
                out.push(compressed[j]);
                j += 1;
            } else {
                let code = byte as usize;
                if code >= symbol_count {
                    return Err(ZyronError::DecodingFailed(format!(
                        "FSST varlen symbol code {} out of range",
                        code
                    )));
                }
                out.extend_from_slice(symbols[code]);
            }
        }
        rows.push(out);
        cursor = row_end;
    }
    let refs: Vec<Option<&[u8]>> = rows.iter().map(|r| Some(r.as_slice())).collect();
    Ok(varlen_pack(&refs))
}

/// A symbol as the trainer and the matcher hold it: its bytes packed little
/// endian into one word, zero padded, and its length. Two symbols are equal
/// exactly when their bytes are, so a word and a length key a map directly
/// and a candidate matches the input by one masked compare
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
struct Symbol {
    word: u64,
    len: u8,
}

/// Mask of the low `len` bytes of a word, indexed by length
const WORD_MASKS: [u64; MAX_SYMBOL_LEN + 1] = [
    0,
    0xFF,
    0xFFFF,
    0xFF_FFFF,
    0xFFFF_FFFF,
    0xFF_FFFF_FFFF,
    0xFFFF_FFFF_FFFF,
    0xFF_FFFF_FFFF_FFFF,
    u64::MAX,
];

/// Symbols a table holds at most, one code short of the byte range so the
/// escape keeps its value
const MAX_SYMBOLS: usize = SYMBOL_TABLE_SIZE - 1;

/// Bytes the training sample is held to.
///
/// Training is several passes over the sample, so its size is the cost of
/// building the table, and a table learned from this much of a column
/// compresses the rest of it about as well as one learned from all of it
const SAMPLE_BYTES_MAX: usize = 32 << 10;

/// Strings the training sample draws from at most
const SAMPLE_STRINGS_MAX: usize = 1024;

/// Bytes a string contributes to the sample at least, however many strings
/// share the budget, so a long string still shows training a run of it
const SAMPLE_WINDOW_MIN: usize = 64;

/// Bytes of the sample the first table is seeded from.
///
/// The seed counts every run of one to eight bytes, which is eight
/// candidates per byte and the one pass that costs more than the sample is
/// long. A few kilobytes of it are enough to put every alignment of a
/// repeated run in the first table, which is what the seed is for: a table
/// grown from pairs alone holds only the alignments the greedy match
/// happened to produce, and a variable run ahead of a constant one leaves
/// those misaligned for good
const SEED_BYTES_MAX: usize = 4 << 10;

impl Symbol {
    /// The symbol for a run of one to eight bytes
    fn from_bytes(bytes: &[u8]) -> Self {
        let len = bytes.len().min(MAX_SYMBOL_LEN);
        let mut packed = [0u8; 8];
        packed[..len].copy_from_slice(&bytes[..len]);
        Self {
            word: u64::from_le_bytes(packed),
            len: len as u8,
        }
    }

    fn bytes(&self) -> [u8; 8] {
        self.word.to_le_bytes()
    }

    /// This symbol followed by another, when the two fit one symbol
    fn followed_by(self, next: Symbol) -> Option<Symbol> {
        let len = self.len as usize + next.len as usize;
        if len > MAX_SYMBOL_LEN {
            return None;
        }
        Some(Symbol {
            word: self.word | (next.word << (8 * self.len as u32)),
            len: len as u8,
        })
    }

    /// Whether any byte of the symbol is the escape byte, which a symbol
    /// never holds so a code and an escaped literal stay apart
    fn contains_escape(&self) -> bool {
        self.bytes()[..self.len as usize].contains(&ESCAPE_BYTE)
    }
}

/// Greedy longest match over a symbol table.
///
/// Candidates are kept per first byte, longest first, so a position reads
/// one word of input, walks the few symbols that start with its byte and
/// takes the first whose masked word matches. No hashing, no allocation,
/// and no work per position for the symbols that cannot start there
struct Matcher {
    /// Per first byte, the word, length and code of every symbol starting
    /// with it, longest first
    buckets: Vec<Vec<(u64, u8, u8)>>,
}

impl Matcher {
    fn new(table: &[Symbol]) -> Self {
        let mut buckets: Vec<Vec<(u64, u8, u8)>> = vec![Vec::new(); 256];
        for (code, sym) in table.iter().enumerate().take(MAX_SYMBOLS) {
            buckets[(sym.word & 0xFF) as usize].push((sym.word, sym.len, code as u8));
        }
        for bucket in &mut buckets {
            bucket.sort_by(|a, b| b.1.cmp(&a.1).then(a.2.cmp(&b.2)));
        }
        Self { buckets }
    }

    /// A matcher over a table read back from a segment, or None when an
    /// entry is not a symbol the format allows
    fn from_table_bytes(table: &[Vec<u8>]) -> Option<Self> {
        if table.len() > MAX_SYMBOLS
            || table
                .iter()
                .any(|s| s.is_empty() || s.len() > MAX_SYMBOL_LEN)
        {
            return None;
        }
        let symbols: Vec<Symbol> = table.iter().map(|s| Symbol::from_bytes(s)).collect();
        Some(Self::new(&symbols))
    }

    /// The longest symbol matching the input at `at`, as its length and code
    #[inline]
    fn longest_at(&self, input: &[u8], at: usize) -> Option<(u8, u8)> {
        let remaining = input.len() - at;
        let word = word_at(input, at);
        self.buckets[(word & 0xFF) as usize]
            .iter()
            .find(|&&(sym_word, len, _)| {
                len as usize <= remaining && (word & WORD_MASKS[len as usize]) == sym_word
            })
            .map(|&(_, len, code)| (len, code))
    }

    /// Compresses one string, appending codes and escaped literals
    fn compress(&self, input: &[u8], out: &mut Vec<u8>) {
        let mut at = 0;
        while at < input.len() {
            match self.longest_at(input, at) {
                Some((len, code)) => {
                    out.push(code);
                    at += len as usize;
                }
                None => {
                    out.push(ESCAPE_BYTE);
                    out.push(input[at]);
                    at += 1;
                }
            }
        }
    }

    /// Splits one string into the symbols compression would emit, a
    /// literal byte standing as a symbol of its own, and returns the bytes
    /// the compressed form takes
    fn tokenize(&self, input: &[u8], tokens: &mut Vec<Symbol>) -> usize {
        let mut at = 0;
        let mut bytes = 0;
        while at < input.len() {
            match self.longest_at(input, at) {
                Some((len, _)) => {
                    tokens.push(Symbol::from_bytes(&input[at..at + len as usize]));
                    at += len as usize;
                    bytes += 1;
                }
                None => {
                    tokens.push(Symbol::from_bytes(&input[at..at + 1]));
                    at += 1;
                    bytes += 2;
                }
            }
        }
        bytes
    }
}

/// The word of input at `at`, zero padded past the end
#[inline]
fn word_at(input: &[u8], at: usize) -> u64 {
    let mut packed = [0u8; 8];
    let take = (input.len() - at).min(8);
    packed[..take].copy_from_slice(&input[at..at + take]);
    u64::from_le_bytes(packed)
}

/// The table ranked out of a gain per candidate: the top of the ranking by
/// gain, ties shorter then smaller, so the table is a function of the
/// counts alone
fn rank_table(gains: &hashbrown::HashMap<Symbol, u64>) -> Vec<Symbol> {
    let mut ranked: Vec<(Symbol, u64)> = gains
        .iter()
        .filter(|(sym, _)| !sym.contains_escape())
        .map(|(sym, gain)| (*sym, *gain))
        .collect();
    ranked.sort_unstable_by(|a, b| {
        b.1.cmp(&a.1)
            .then(a.0.len.cmp(&b.0.len))
            .then(a.0.word.cmp(&b.0.word))
    });
    ranked
        .into_iter()
        .take(MAX_SYMBOLS)
        .map(|(sym, _)| sym)
        .collect()
}

/// The first table: every run of one to eight bytes in the head of the
/// sample, ranked by the bytes it covers
fn seeded_table(sample: &[&[u8]], gains: &mut hashbrown::HashMap<Symbol, u64>) -> Vec<Symbol> {
    gains.clear();
    let mut budget = SEED_BYTES_MAX;
    for s in sample {
        if budget == 0 {
            break;
        }
        let take = s.len().min(budget);
        budget -= take;
        let s = &s[..take];
        for start in 0..s.len() {
            let word = word_at(s, start);
            for len in 1..=MAX_SYMBOL_LEN.min(s.len() - start) {
                let sym = Symbol {
                    word: word & WORD_MASKS[len],
                    len: len as u8,
                };
                *gains.entry(sym).or_insert(0) += len as u64;
            }
        }
    }
    rank_table(gains)
}

/// The strings training reads: up to the string cap, spread over the
/// column rather than taken from its head, each contributing a window of
/// itself sized so the whole sample stays within the byte budget. The
/// window slides along the column so the sample sees the strings' tails as
/// well as their heads
fn training_sample<'a>(strings: &[&'a [u8]]) -> Vec<&'a [u8]> {
    let step = strings.len().div_ceil(SAMPLE_STRINGS_MAX).max(1);
    let picked = strings.len().div_ceil(step);
    let window = (SAMPLE_BYTES_MAX / picked.max(1)).max(SAMPLE_WINDOW_MIN);
    let mut sample = Vec::with_capacity(picked);
    for (k, s) in strings.iter().step_by(step).enumerate() {
        if s.len() <= window {
            sample.push(*s);
        } else {
            let offset = (k * 61) % (s.len() - window + 1);
            sample.push(&s[offset..offset + window]);
        }
    }
    sample
}

/// The symbol table the sample settles on.
///
/// The first table is seeded from the runs in the sample. Each round then
/// compresses the sample with the table in hand and counts what came out:
/// every symbol by the bytes it covered, and every adjacent pair by the
/// bytes the two would cover as one. The next table is the top of that
/// ranking, so a symbol that is used keeps its place, one that is not
/// falls out, and a pair that is used grows into one symbol. The table
/// kept is the one that compressed the sample smallest
fn train_symbol_table(sample: &[&[u8]]) -> Vec<Symbol> {
    let mut gains: hashbrown::HashMap<Symbol, u64> = hashbrown::HashMap::new();
    let mut table = seeded_table(sample, &mut gains);
    let mut best: Option<(usize, Vec<Symbol>)> = None;
    let mut tokens: Vec<Symbol> = Vec::new();
    for round in 0..=REFINEMENT_ROUNDS {
        let matcher = Matcher::new(&table);
        gains.clear();
        let mut compressed = 0usize;
        for s in sample {
            tokens.clear();
            compressed += matcher.tokenize(s, &mut tokens);
            let mut prev: Option<Symbol> = None;
            for &token in &tokens {
                *gains.entry(token).or_insert(0) += token.len as u64;
                if let Some(joined) = prev.and_then(|p| p.followed_by(token)) {
                    *gains.entry(joined).or_insert(0) += joined.len as u64;
                }
                prev = Some(token);
            }
        }
        if best.as_ref().is_none_or(|(size, _)| compressed < *size) {
            best = Some((compressed, table.clone()));
        }
        if round == REFINEMENT_ROUNDS {
            break;
        }
        table = rank_table(&gains);
    }
    best.map(|(_, table)| table).unwrap_or_default()
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
    fn test_roundtrip_varlen_canonical_buffer() {
        // Variable-length rows of differing lengths, including an empty one.
        let vals: Vec<Option<&[u8]>> = vec![
            Some(b"the quick brown fox".as_slice()),
            Some(b"the quick brown dog".as_slice()),
            Some(b"".as_slice()),
            Some(b"jumps over the lazy fox".as_slice()),
            Some(b"the quick brown fox".as_slice()),
        ];
        let raw = crate::encoding::varlen_pack(&vals);
        let enc = FsstEncoding;
        let encoded = enc.encode(&raw, vals.len(), 0).expect("encode varlen");
        let decoded = enc.decode(&encoded, vals.len(), 0).expect("decode varlen");
        let rows = crate::encoding::varlen_slice_rows(&decoded, vals.len()).expect("slice");
        assert_eq!(rows[0], b"the quick brown fox");
        assert_eq!(rows[1], b"the quick brown dog");
        assert_eq!(rows[2], b"");
        assert_eq!(rows[3], b"jumps over the lazy fox");
        assert_eq!(rows[4], b"the quick brown fox");
    }

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

    /// The symbol table read back from an encoded segment
    fn symbols_of(encoded: &[u8]) -> Vec<Vec<u8>> {
        let count = u32::from_le_bytes([encoded[8], encoded[9], encoded[10], encoded[11]]) as usize;
        let mut pos = 14;
        let mut table = Vec::with_capacity(count);
        for _ in 0..count {
            let len = encoded[pos] as usize;
            table.push(encoded[pos + 1..pos + 1 + len].to_vec());
            pos += 1 + len;
        }
        table
    }

    // A label column, the shape a lake table's text column takes, encodes
    // smaller than raw, decodes to what went in, and encodes to the same
    // bytes every time
    #[test]
    fn test_label_column_roundtrips_and_is_deterministic() {
        let labels: Vec<Vec<u8>> = (0..10_000)
            .map(|i| format!("row-{:08}", i).into_bytes())
            .collect();
        let vals: Vec<Option<&[u8]>> = labels.iter().map(|l| Some(l.as_slice())).collect();
        let raw = crate::encoding::varlen_pack(&vals);
        let enc = FsstEncoding;
        let encoded = enc.encode(&raw, vals.len(), 0).expect("encode");
        assert!(
            encoded.len() * 2 < raw.len(),
            "{} encoded bytes for {} raw",
            encoded.len(),
            raw.len()
        );
        assert_eq!(enc.decode(&encoded, vals.len(), 0).expect("decode"), raw);
        assert_eq!(enc.encode(&raw, vals.len(), 0).expect("encode"), encoded);
        assert!(
            symbols_of(&encoded)
                .iter()
                .all(|s| s.len() <= MAX_SYMBOL_LEN)
        );
    }

    // Strings far longer than the sample budget train on a bounded sample
    // and still round-trip
    #[test]
    fn test_long_strings_train_on_a_bounded_sample() {
        let rows: Vec<Vec<u8>> = (0..200)
            .map(|i| format!("{i:04}-").repeat(300).into_bytes())
            .collect();
        let vals: Vec<Option<&[u8]>> = rows.iter().map(|r| Some(r.as_slice())).collect();
        let raw = crate::encoding::varlen_pack(&vals);
        let enc = FsstEncoding;
        let encoded = enc.encode(&raw, vals.len(), 0).expect("encode");
        assert!(encoded.len() < raw.len());
        assert_eq!(enc.decode(&encoded, vals.len(), 0).expect("decode"), raw);
    }

    // The escape byte never lands inside a symbol, so a run of it is
    // carried as literals and comes back intact
    #[test]
    fn test_symbols_never_hold_the_escape_byte() {
        let rows: Vec<Vec<u8>> = (0..500)
            .map(|i| {
                let mut r = vec![ESCAPE_BYTE, ESCAPE_BYTE, b'a', ESCAPE_BYTE, b'b'];
                r.extend_from_slice(format!("{}", i % 7).as_bytes());
                r
            })
            .collect();
        let vals: Vec<Option<&[u8]>> = rows.iter().map(|r| Some(r.as_slice())).collect();
        let raw = crate::encoding::varlen_pack(&vals);
        let enc = FsstEncoding;
        let encoded = enc.encode(&raw, vals.len(), 0).expect("encode");
        assert!(
            symbols_of(&encoded)
                .iter()
                .all(|s| !s.contains(&ESCAPE_BYTE))
        );
        assert_eq!(enc.decode(&encoded, vals.len(), 0).expect("decode"), raw);
    }

    // The equality fast path compresses the search term with the stored
    // table and compares compressed rows, which must agree with a decode
    #[test]
    fn test_equality_predicate_matches_through_the_symbol_table() {
        let rows: Vec<Vec<u8>> = (0..300)
            .map(|i| format!("item-{:03}-{}", i % 50, i % 3).into_bytes())
            .collect();
        let vals: Vec<Option<&[u8]>> = rows.iter().map(|r| Some(r.as_slice())).collect();
        let raw = crate::encoding::varlen_pack(&vals);
        let enc = FsstEncoding;
        let encoded = enc.encode(&raw, vals.len(), 0).expect("encode");
        let target = b"item-007-1".to_vec();
        let mask = enc
            .eval_predicate(&encoded, vals.len(), 0, &Predicate::Equality(&target))
            .expect("predicate");
        for (row, value) in rows.iter().enumerate() {
            let set = mask[row / 8] & (1 << (row % 8)) != 0;
            assert_eq!(set, *value == target, "row {row}");
        }
    }
}
