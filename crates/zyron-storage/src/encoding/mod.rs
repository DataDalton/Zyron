#![allow(non_snake_case)]
//! Column encoding engine for .zyr columnar storage.
//!
//! Provides type-specific encoding strategies that compact column data
//! and support predicate evaluation on encoded data without full decode.
//!
//! Encoding selection samples column values and picks the smallest output
//! from candidate encodings, with decode speed as a tiebreaker.

mod alp;
mod bitpack;
mod constant;
mod dictionary;
mod fastlanes;
mod fsst;
mod rle;
mod unencoded;

pub use alp::AlpEncoding;
pub use bitpack::BitPackEncoding;
pub use constant::ConstantEncoding;
pub use dictionary::DictionaryEncoding;
pub use fastlanes::FastLanesEncoding;
pub use fsst::FsstEncoding;
pub use rle::RleEncoding;
pub use unencoded::UnencodedEncoding;

use zyron_common::types::TypeId;
use zyron_common::{Result, ZyronError};

/// Column encoding type identifier stored in segment headers.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum EncodingType {
    Unencoded = 0,
    Constant = 1,
    BitPack = 2,
    Rle = 3,
    Dictionary = 4,
    FastLanes = 5,
    Alp = 6,
    Fsst = 7,
}

impl EncodingType {
    pub fn from_u8(v: u8) -> Result<Self> {
        match v {
            0 => Ok(Self::Unencoded),
            1 => Ok(Self::Constant),
            2 => Ok(Self::BitPack),
            3 => Ok(Self::Rle),
            4 => Ok(Self::Dictionary),
            5 => Ok(Self::FastLanes),
            6 => Ok(Self::Alp),
            7 => Ok(Self::Fsst),
            _ => Err(ZyronError::DecodingFailed(format!(
                "unknown encoding type: {}",
                v
            ))),
        }
    }
}

/// Predicate for query-on-encoded evaluation.
/// Encodings that support predicate pushdown can evaluate these
/// directly on encoded data, returning a bitmask of matching rows.
pub enum Predicate<'a> {
    /// Match rows equal to the given value.
    Equality(&'a [u8]),
    /// Match rows within [low, high]. None means unbounded on that side.
    ///
    /// Bounds are cell bytes compared in the column's stored order, which
    /// `range_admits` defines: little endian numeric for a fixed-width
    /// cell, lexicographic for a variable-length one. Every encoding that
    /// resolves a range from its own compact form goes through that, so
    /// two segments of one column answer alike whatever encoding each
    /// picked.
    ///
    /// A float column is the exception and must not be pushed one. Its
    /// stored bytes put negatives above positives and reverse their order,
    /// so byte order is not value order and ALP answers in float order
    /// while an unencoded segment of the same column would not.
    Range {
        low: Option<&'a [u8]>,
        high: Option<&'a [u8]>,
    },
    /// Match rows whose value is in the given set.
    In(&'a [&'a [u8]]),
}

/// Core encoding trait. Each encoding strategy implements this to provide
/// encode, decode, and optional predicate evaluation on encoded data.
pub trait Encoding: Send + Sync {
    /// Returns the encoding type identifier for this implementation.
    fn encoding_type(&self) -> EncodingType;

    /// Encodes raw column data into the encoding's compact format.
    /// `data` contains row_count values laid out contiguously, each of `value_size` bytes.
    /// For variable-length encodings, data is prefixed with a u32 offset array.
    fn encode(&self, data: &[u8], row_count: usize, value_size: usize) -> Result<Vec<u8>>;

    /// Decodes encoded data back to the original raw column format.
    fn decode(&self, encoded: &[u8], row_count: usize, value_size: usize) -> Result<Vec<u8>>;

    /// Evaluates a predicate directly on encoded data, returning a bitmask
    /// where bit i is set if row i matches. The Vec<u8> is a packed bit array
    /// with ceil(row_count / 8) bytes.
    /// Default implementation decodes and evaluates, but encodings should
    /// override this to avoid full decode when possible.
    fn eval_predicate(
        &self,
        encoded: &[u8],
        row_count: usize,
        value_size: usize,
        predicate: &Predicate,
    ) -> Result<Vec<u8>> {
        let decoded = self.decode(encoded, row_count, value_size)?;
        eval_predicate_on_raw(&decoded, row_count, value_size, predicate)
    }

    /// Decodes rows `start..end` in the same layout `decode` produces for a
    /// column of `end - start` rows.
    ///
    /// A point read wants a handful of rows out of a segment holding
    /// millions, and decoding the segment to reach them costs the whole
    /// column. An encoding that can address a row without replaying the
    /// ones before it overrides this and pays for the range alone.
    ///
    /// The default is correct for every encoding and cheaper for none: it
    /// decodes the segment and takes the range out of the result, which is
    /// what a caller would otherwise have written itself
    fn decode_range(
        &self,
        encoded: &[u8],
        row_count: usize,
        value_size: usize,
        start: usize,
        end: usize,
    ) -> Result<Vec<u8>> {
        let decoded = self.decode(encoded, row_count, value_size)?;
        slice_decoded(&decoded, row_count, value_size, start, end)
    }
}

/// Takes rows `start..end` out of a fully decoded column, in the layout the
/// decode produced.
///
/// Fixed-width rows are a byte range. Variable-length rows are repacked
/// into their own canonical buffer, because the offsets in the original are
/// relative to a blob the range does not carry
pub fn slice_decoded(
    decoded: &[u8],
    row_count: usize,
    value_size: usize,
    start: usize,
    end: usize,
) -> Result<Vec<u8>> {
    let (start, end) = clamp_range(row_count, start, end);
    // Every encoding's `decode` answers a zero-row column with no bytes at
    // all rather than an empty canonical buffer, so an empty range does the
    // same and the two stay interchangeable
    if start == end {
        return Ok(Vec::new());
    }
    if value_size > 0 {
        let from = start * value_size;
        let to = end * value_size;
        if to > decoded.len() {
            return Err(ZyronError::DecodingFailed(format!(
                "decoded column of {} bytes cannot supply rows {}..{} at {} bytes each",
                decoded.len(),
                start,
                end,
                value_size
            )));
        }
        return Ok(decoded[from..to].to_vec());
    }
    let rows = varlen_slice_rows(decoded, row_count)?;
    let taken: Vec<Option<&[u8]>> = rows[start..end].iter().map(|r| Some(*r)).collect();
    Ok(varlen_pack(&taken))
}

/// Clamps a requested row range to what the column holds, so a caller
/// asking past the end gets the rows that exist rather than an error
pub fn clamp_range(row_count: usize, start: usize, end: usize) -> (usize, usize) {
    let start = start.min(row_count);
    let end = end.clamp(start, row_count);
    (start, end)
}

/// Canonical variable-length column buffer, used when `value_size == 0`:
///   [0..4]                      row_count: u32 (LE)
///   [4 .. 4 + 4*(row_count+1)]  offsets: u32 LE array. offsets[0] == 0,
///                               offsets[i+1] == end of row i in the blob.
///   [offsets_end ..]            values blob
/// Row i bytes are `blob[offsets[i]..offsets[i+1]]`. A null row is stored as
/// a zero-length slice. The null bitmap held by `ColumnSegment` is the
/// authoritative null marker, so an empty string and a null stay distinct.
pub const VARLEN_VALUE_SIZE: usize = 0;

/// Packs row values into the canonical variable-length buffer. A `None`
/// value is encoded as a zero-length row.
pub fn varlen_pack(values: &[Option<&[u8]>]) -> Vec<u8> {
    let row_count = values.len();
    let blob_len: usize = values.iter().map(|v| v.map_or(0, |b| b.len())).sum();
    let offsets_bytes = 4 * (row_count + 1);
    let mut out = Vec::with_capacity(4 + offsets_bytes + blob_len);
    out.extend_from_slice(&(row_count as u32).to_le_bytes());
    let mut cursor = 0u32;
    out.extend_from_slice(&cursor.to_le_bytes());
    for v in values {
        cursor += v.map_or(0, |b| b.len()) as u32;
        out.extend_from_slice(&cursor.to_le_bytes());
    }
    for v in values {
        if let Some(b) = v {
            out.extend_from_slice(b);
        }
    }
    out
}

/// Reads the row count from a canonical variable-length buffer header.
pub fn varlen_row_count(data: &[u8]) -> Result<usize> {
    if data.len() < 4 {
        return Err(ZyronError::DecodingFailed(
            "varlen buffer shorter than header".to_string(),
        ));
    }
    Ok(u32::from_le_bytes([data[0], data[1], data[2], data[3]]) as usize)
}

/// Borrows each row as a slice from a canonical variable-length buffer.
pub fn varlen_slice_rows(data: &[u8], row_count: usize) -> Result<Vec<&[u8]>> {
    let offsets_start = 4;
    let blob_start = offsets_start + 4 * (row_count + 1);
    if data.len() < blob_start {
        return Err(ZyronError::DecodingFailed(
            "varlen buffer offset array truncated".to_string(),
        ));
    }
    let read_off = |i: usize| -> u32 {
        let p = offsets_start + 4 * i;
        u32::from_le_bytes([data[p], data[p + 1], data[p + 2], data[p + 3]])
    };
    let blob = &data[blob_start..];
    let mut rows = Vec::with_capacity(row_count);
    for i in 0..row_count {
        let lo = read_off(i) as usize;
        let hi = read_off(i + 1) as usize;
        if lo > hi || hi > blob.len() {
            return Err(ZyronError::DecodingFailed(
                "varlen buffer offset out of range".to_string(),
            ));
        }
        rows.push(&blob[lo..hi]);
    }
    Ok(rows)
}

/// Borrows each row as a slice. `value_size > 0` is the fixed-width layout
/// (`row_count` contiguous `value_size`-byte slots). `value_size == 0` is the
/// canonical variable-length layout. Every encoder uses this so the
/// fixed/variable split lives in exactly one place.
pub fn slice_rows(data: &[u8], row_count: usize, value_size: usize) -> Result<Vec<&[u8]>> {
    if value_size == 0 {
        return varlen_slice_rows(data, row_count);
    }
    if data.len() < row_count * value_size {
        return Err(ZyronError::DecodingFailed(
            "data shorter than expected row count".to_string(),
        ));
    }
    let mut rows = Vec::with_capacity(row_count);
    for i in 0..row_count {
        rows.push(&data[i * value_size..(i + 1) * value_size]);
    }
    Ok(rows)
}

/// Orders two cells of one column the way that column stores them.
///
/// A fixed-width cell holds a little endian value, so its last byte is the
/// most significant and a plain slice comparison would rank 256 below 255.
/// A variable-length cell is compared lexicographically, which is already
/// its value order. Bytes past the shorter operand read as zero, so a
/// bound narrower than the cell still compares.
#[inline]
pub fn compare_cell_bytes(a: &[u8], b: &[u8], value_size: usize) -> std::cmp::Ordering {
    if value_size == 0 {
        return a.cmp(b);
    }
    let byte = |s: &[u8], i: usize| -> u8 { s.get(i).copied().unwrap_or(0) };
    for i in (0..value_size.max(a.len()).max(b.len())).rev() {
        match byte(a, i).cmp(&byte(b, i)) {
            std::cmp::Ordering::Equal => continue,
            other => return other,
        }
    }
    std::cmp::Ordering::Equal
}

/// Whether one cell falls inside a Range under the column's stored order.
///
/// Every encoding that resolves a range from its own compact form calls
/// this instead of comparing bytes itself. Two segments of one column
/// choose their encodings independently, so a range that meant
/// lexicographic order in a run length segment and numeric order in a
/// bit-packed one would return different rows for the same query.
#[inline]
pub fn range_admits(
    cell: &[u8],
    value_size: usize,
    low: Option<&[u8]>,
    high: Option<&[u8]>,
) -> bool {
    low.is_none_or(|lo| compare_cell_bytes(cell, lo, value_size) != std::cmp::Ordering::Less)
        && high.is_none_or(|hi| {
            compare_cell_bytes(cell, hi, value_size) != std::cmp::Ordering::Greater
        })
}

/// Builds one row bitmask from a per-row answer, folding eight answers into
/// a byte before storing it.
///
/// Setting the mask a bit at a time reads and writes the same byte on eight
/// consecutive rows, which serializes those rows on that byte and pays a
/// bounds check each time. Eight answers folded in a register leave one
/// store per eight rows, and the eight `admits` calls have no dependency on
/// each other so they pipeline. Every encoding that resolves a predicate
/// row by row builds its mask through here.
///
/// `admits` is called exactly once for every row, in ascending row order, so
/// an encoding whose rows are addressed by walking forward (a stored length
/// per row, for one) can carry its cursor in the closure
#[inline]
pub fn bitmask_from_rows<F: FnMut(usize) -> bool>(row_count: usize, mut admits: F) -> Vec<u8> {
    let mut bitmask = vec![0u8; row_count.div_ceil(8)];
    let full = row_count / 8;
    for (byte_index, slot) in bitmask.iter_mut().enumerate().take(full) {
        let base = byte_index * 8;
        let mut bits = 0u8;
        for lane in 0..8usize {
            bits |= (admits(base + lane) as u8) << lane;
        }
        *slot = bits;
    }
    let tail = full * 8;
    if tail < row_count {
        let mut bits = 0u8;
        for row in tail..row_count {
            bits |= (admits(row) as u8) << (row - tail);
        }
        bitmask[full] = bits;
    }
    bitmask
}

/// Evaluates a predicate on raw (decoded) column data, producing a packed bitmask.
pub fn eval_predicate_on_raw(
    data: &[u8],
    row_count: usize,
    value_size: usize,
    predicate: &Predicate,
) -> Result<Vec<u8>> {
    if value_size == 0 {
        // Slicing either reaches every row the count claims or fails, so the
        // row a mask bit stands for is always there to compare
        let rows = varlen_slice_rows(data, row_count)?;
        return Ok(match predicate {
            Predicate::Equality(target) => bitmask_from_rows(row_count, |i| rows[i] == *target),
            Predicate::Range { low, high } => {
                bitmask_from_rows(row_count, |i| range_admits(rows[i], 0, *low, *high))
            }
            Predicate::In(values) => {
                bitmask_from_rows(row_count, |i| values.iter().any(|t| rows[i] == *t))
            }
        });
    }
    // The widest row this reads is the last one, so the one check that
    // decides every row is made once rather than per row
    if row_count * value_size > data.len() {
        return Err(ZyronError::DecodingFailed(
            "data shorter than expected row count".to_string(),
        ));
    }
    let row = |i: usize| &data[i * value_size..(i + 1) * value_size];

    Ok(match predicate {
        Predicate::Equality(target) => bitmask_from_rows(row_count, |i| row(i) == *target),
        Predicate::Range { low, high } => {
            bitmask_from_rows(row_count, |i| range_admits(row(i), value_size, *low, *high))
        }
        Predicate::In(values) => bitmask_from_rows(row_count, |i| values.contains(&row(i))),
    })
}

/// Cardinality at or above which a dictionary is never chosen, whatever the
/// row count. Both selection paths test `cardinality < 65536`, so this is
/// that constant named once
const DICTIONARY_MAX_CARDINALITY: usize = 65536;

/// Statistics computed from a column sample for encoding selection.
struct ColumnSampleStats {
    /// Distinct values, saturating at the point the exact count stops
    /// changing any decision. Every consumer tests it as an upper bound, so
    /// a saturated value reads as "more than the ceiling" and compares the
    /// same way the true count would
    cardinality: usize,
    run_count: usize,
    all_identical: bool,
}

/// Computes sample statistics from a set of values.
/// Each value is Option<&[u8]> where None represents null.
///
/// The distinct set stops growing once the count can no longer change an
/// answer. Cardinality decides exactly two things here, whether the column
/// is one repeated value and whether it is sparse enough for a dictionary,
/// and past `min(DICTIONARY_MAX_CARDINALITY, rows / 2)` both are already
/// settled. Without the cap a column builds a hash set with one entry per
/// distinct value purely to choose an encoding, so a million row column
/// allocated a million entry set to conclude it would not be dictionary
/// encoded. Selection cost and memory now stay flat in the row count.
///
/// Runs are counted from a comparison against the previous value rather than
/// from the set, so `run_count` stays exact after the set saturates
fn compute_sample_stats(sample: &[Option<&[u8]>]) -> ColumnSampleStats {
    // Two is the floor whatever the ceiling works out to, because
    // `all_identical` still has to tell one distinct value from more than one
    let cap = DICTIONARY_MAX_CARDINALITY.min(sample.len() / 2).max(2);
    let mut distinct = hashbrown::HashSet::new();
    let mut saturated = false;
    let mut run_count = 1usize;
    let mut prev_value: Option<&[u8]> = None;
    let mut null_count = 0usize;

    for val in sample {
        if let Some(v) = val {
            if !saturated {
                distinct.insert(*v);
                saturated = distinct.len() > cap;
            }

            if let Some(prev) = prev_value {
                if *v != prev {
                    run_count += 1;
                }
            }
            prev_value = Some(*v);
        } else {
            null_count += 1;
        }
    }

    // Constant encoding stores the raw buffer's single repeated cell, and a
    // NULL occupies a placeholder cell (zero-filled fixed cell or zero-length
    // row) that differs from any non-null value. So a column is
    // all-identical only when it is all null (every placeholder identical)
    // or has exactly one distinct value and no nulls at all
    let all_identical =
        !saturated && (distinct.is_empty() || (distinct.len() == 1 && null_count == 0));

    ColumnSampleStats {
        cardinality: distinct.len(),
        run_count,
        all_identical,
    }
}

/// Selects the best encoding type for a column based on sampled data and type.
///
/// Uses a two-phase approach:
/// 1. Heuristic selection produces up to 2 candidate encodings based on data statistics.
/// 2. Trial-encode the sample with each candidate and pick the one producing
///    the smallest output, with Unencoded as a fallback if both are larger than raw.
///
/// Heuristic priority:
/// 1. Constant - all values identical (zero per-row cost)
/// 2. BitPack - booleans (1-bit packing)
/// 3. Dictionary - low cardinality (< row_count/2 AND < 65536)
/// 4. RLE - repetitive data (run_count < row_count/10)
/// 5. FastLanes - integer types (FoR + delta + bit-packing)
/// 6. ALP - float types (exponent/mantissa split)
/// 7. FSST - string types (symbol table encoding)
/// 8. Unencoded - fallback
pub fn select_encoding(type_id: TypeId, sample: &[Option<&[u8]>]) -> EncodingType {
    select_encoding_bounded(type_id, sample, TRIAL_ENCODE_ROWS)
}

/// The encoding a column will use, and the bytes selection already produced.
pub struct EncodingChoice {
    pub encoding: EncodingType,
    /// Output of the trial encode, present only when the trial covered every
    /// row and its candidate is the encoding that was chosen.
    ///
    /// A caller holding this must not encode again. These are the bytes a
    /// second encode would produce, from the same buffer through the same
    /// encoder, so re-encoding would spend a full pass to arrive back here
    pub encoded: Option<Vec<u8>>,
}

/// Chooses an encoding for a fixed-width column the caller has already
/// packed into `raw_data`.
///
/// Selection cannot know whether its candidate beats storing the column raw
/// without encoding it, and under `exact` that trial is a full encode of
/// every row. The previous form built its own copy of the packed buffer,
/// encoded into it, compared the size and then discarded both, leaving the
/// caller to pack and encode the same column a second time. Passing the
/// buffer in and the result back means a column is packed once and encoded
/// once.
///
/// `exact` trials every row, which is what a caller holding the whole
/// column wants: the candidate is then chosen exactly when it wins over the
/// data the decoder will actually face. Otherwise the trial is a bounded
/// prefix and no output is returned, because a prefix's bytes are not the
/// column's bytes
pub fn select_encoding_packed(
    type_id: TypeId,
    sample: &[Option<&[u8]>],
    raw_data: &[u8],
    value_size: usize,
    exact: bool,
) -> EncodingChoice {
    let trial_rows = if exact { usize::MAX } else { TRIAL_ENCODE_ROWS };
    select_encoding_inner(type_id, sample, Some((raw_data, value_size)), trial_rows)
}

/// Selects a fixed-width column's encoding with the trial encode run over
/// every row rather than a bounded prefix.
///
/// The statistical heuristics already read the whole column, only the
/// candidate-versus-raw trial is sampled in `select_encoding`. A caller that
/// already holds the full column, such as the lake writer, gets the decision
/// the decoder will actually face: the candidate is chosen exactly when it is
/// smaller over every row, so an unrepresentative prefix can no longer pick
/// an encoding that loses on the rest of the column.
pub fn select_encoding_exact(type_id: TypeId, values: &[Option<&[u8]>]) -> EncodingType {
    select_encoding_bounded(type_id, values, usize::MAX)
}

/// Rows the trial encode compares when the caller does not ask for exact
/// selection. Bounded so a wide column does not encode twice in full.
const TRIAL_ENCODE_ROWS: usize = 1024;

fn select_encoding_bounded(
    type_id: TypeId,
    sample: &[Option<&[u8]>],
    trial_rows: usize,
) -> EncodingType {
    select_encoding_inner(type_id, sample, None, trial_rows).encoding
}

/// Shared selection. `packed` is the caller's already-built raw buffer and
/// the fixed value width it was built at, or None when selection has to
/// build its own to trial with
fn select_encoding_inner(
    type_id: TypeId,
    sample: &[Option<&[u8]>],
    packed: Option<(&[u8], usize)>,
    trial_rows: usize,
) -> EncodingChoice {
    let plain = |encoding| EncodingChoice {
        encoding,
        encoded: None,
    };
    if sample.is_empty() {
        return plain(EncodingType::Unencoded);
    }

    let stats = compute_sample_stats(sample);
    let row_count = sample.len();

    // All values identical (including all-null): constant encoding is always optimal
    if stats.all_identical {
        return plain(EncodingType::Constant);
    }

    // Booleans: bit-pack to 1-bit is always the best choice
    if type_id == TypeId::Boolean {
        return plain(EncodingType::BitPack);
    }

    // Statistical heuristics: Dictionary and RLE are chosen based on
    // data characteristics and take priority. They support predicate
    // pushdown on encoded data, which is worth structural overhead.
    if stats.cardinality < DICTIONARY_MAX_CARDINALITY && stats.cardinality < row_count / 2 {
        return plain(EncodingType::Dictionary);
    }

    if stats.run_count < row_count / 10 {
        return plain(EncodingType::Rle);
    }

    // Type-specific candidate and Unencoded fallback for trial-encode.
    // Temporal and HLC types are integer-backed fixed-width values (Date i32,
    // Time/Timestamp/TimestampTz/Interval i64, ps Timestamp/HLC i128), so they
    // ride the FastLanes FoR+delta/delta-of-delta/const-step path exactly like
    // the integer types. ColumnSegment::build is handed the logical type id,
    // which for a ps column is Timestamp (not Int128), so keying only on
    // is_integer() here would fold every timestamp column Unencoded and the
    // "ps at us-class density" property would never hold. Trial-encode still
    // falls back to Unencoded when FastLanes is not actually smaller.
    let typeCandidate = if type_id.is_integer() || type_id.is_temporal() || type_id == TypeId::Hlc {
        EncodingType::FastLanes
    } else if type_id.is_floating_point() {
        EncodingType::Alp
    } else if type_id.is_string() {
        EncodingType::Fsst
    } else {
        return plain(EncodingType::Unencoded);
    };

    // Trial-encode: compare type-specific encoding against Unencoded
    // to verify it produces a smaller output.
    let valueSize = match packed {
        Some((_, width)) => width,
        None => sample.iter().find_map(|v| v.map(|b| b.len())).unwrap_or(0),
    };

    if valueSize == 0 {
        return plain(typeCandidate);
    }

    let sampleCount = sample.len().min(trial_rows);
    // The caller's buffer already holds exactly what the trial would build,
    // values at `i * valueSize` and null slots zeroed, so a prefix of it is
    // the trial input. Only a caller that supplied none pays to build one
    let mut ownedRaw: Vec<u8> = Vec::new();
    let rawData: &[u8] = match packed {
        Some((buffer, _)) if buffer.len() >= sampleCount * valueSize => {
            &buffer[..sampleCount * valueSize]
        }
        _ => {
            let trialSample = &sample[..sampleCount];
            ownedRaw = vec![0u8; sampleCount * valueSize];
            fill_trial_buffer(&mut ownedRaw, trialSample, valueSize);
            &ownedRaw
        }
    };

    let encoder = create_encoding(typeCandidate);
    match encoder.encode(rawData, sampleCount, valueSize) {
        Ok(encoded) if encoded.len() < rawData.len() => EncodingChoice {
            encoding: typeCandidate,
            // Reusable only when the trial encoded the whole column. A
            // prefix's output describes a prefix and would truncate the
            // column if a caller wrote it out
            encoded: (sampleCount == sample.len()).then_some(encoded),
        },
        _ => plain(EncodingType::Unencoded),
    }
}

/// Packs values into a fixed-width trial buffer, leaving null slots zeroed.
///
/// Matches the layout a segment build produces so the two are byte
/// identical, which is what lets a caller hand its own buffer in
fn fill_trial_buffer(rawData: &mut [u8], sample: &[Option<&[u8]>], valueSize: usize) {
    for (i, val) in sample.iter().enumerate() {
        if let Some(v) = val {
            let start = i * valueSize;
            let end = start + valueSize;
            if v.len() == valueSize && end <= rawData.len() {
                rawData[start..end].copy_from_slice(v);
            }
        }
    }
}

/// Selects the encoding for a variable-length column (`value_size == 0`,
/// canonical buffer). Fixed-width selection in `select_encoding` keys off
/// value width and bit-packing which do not apply to variable-length data.
/// Constant collapses an all-identical column to a single stored value.
/// Unencoded stores the canonical buffer verbatim and round-trips exactly,
/// with predicate pushdown handled by `eval_predicate_on_raw`'s
/// variable-length path. FSST symbol compression of the values blob is the
/// next optimization layered on this selection, not a correctness
/// prerequisite: every variable-length column folds and reads correctly
/// under this policy.
pub fn select_encoding_varlen(_type_id: TypeId, sample: &[Option<&[u8]>]) -> EncodingType {
    if sample.is_empty() {
        return EncodingType::Unencoded;
    }
    let stats = compute_sample_stats(sample);
    if stats.all_identical {
        return EncodingType::Constant;
    }

    // Pick the encoding from a bounded prefix probe instead of fully encoding
    // the whole column twice (Dictionary and FSST) just to choose. The
    // cardinality/run decision uses the full-sample stats above (cheap: a
    // HashSet pass, no encoding); only the expensive trial compression runs
    // on a bounded prefix. The chosen encoder still encodes the full column
    // in build_varlen, so this changes the selection cost, not the encoded
    // output. Unencoded is the always-correct floor, so a candidate is only
    // chosen when it is strictly smaller on the probe. RLE is intentionally
    // not a candidate here: it is a fixed-width encoder and cannot round-trip
    // the canonical variable-length buffer; run-heavy / low-cardinality
    // variable-length columns are already captured densely by the Dictionary
    // candidate (whole-value dedup + bit-packed codes).
    const PROBE_ROWS: usize = 8192;
    let row_count = sample.len();
    let probe = &sample[..row_count.min(PROBE_ROWS)];
    let raw = varlen_pack(probe);
    let probe_rows = probe.len();
    let mut best = EncodingType::Unencoded;
    let mut best_size = raw.len();

    if stats.cardinality < DICTIONARY_MAX_CARDINALITY && stats.cardinality < row_count / 2 {
        let dict = create_encoding(EncodingType::Dictionary);
        if let Ok(enc) = dict.encode(&raw, probe_rows, 0)
            && enc.len() < best_size
        {
            best = EncodingType::Dictionary;
            best_size = enc.len();
        }
    }

    let fsst = create_encoding(EncodingType::Fsst);
    if let Ok(enc) = fsst.encode(&raw, probe_rows, 0)
        && enc.len() < best_size
    {
        best = EncodingType::Fsst;
    }

    best
}

/// Creates an Encoding trait object for the given encoding type.
pub fn create_encoding(encoding_type: EncodingType) -> Box<dyn Encoding> {
    match encoding_type {
        EncodingType::Unencoded => Box::new(UnencodedEncoding),
        EncodingType::Constant => Box::new(ConstantEncoding),
        EncodingType::BitPack => Box::new(BitPackEncoding),
        EncodingType::Rle => Box::new(RleEncoding),
        EncodingType::Dictionary => Box::new(DictionaryEncoding),
        EncodingType::FastLanes => Box::new(FastLanesEncoding),
        EncodingType::Alp => Box::new(AlpEncoding),
        EncodingType::Fsst => Box::new(FsstEncoding),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Capping the distinct set is only sound if it never changes a choice.
    ///
    /// Cardinality reaches two decisions, `all_identical` and the dictionary
    /// threshold, and both are compared against an independent uncapped
    /// count here across the shapes that straddle every boundary the cap
    /// touches: a column with one value, with two, with exactly half its
    /// rows distinct, with one more than half, and with every row distinct.
    /// Run counts are compared too, because they are counted outside the set
    /// and must stay exact after it saturates
    #[test]
    fn capping_cardinality_preserves_every_encoding_decision() {
        fn uncapped(sample: &[Option<&[u8]>]) -> (usize, usize, bool) {
            let mut distinct = std::collections::HashSet::new();
            let mut run_count = 1usize;
            let mut prev: Option<&[u8]> = None;
            let mut nulls = 0usize;
            for val in sample {
                if let Some(v) = val {
                    distinct.insert(*v);
                    if let Some(p) = prev
                        && *v != p
                    {
                        run_count += 1;
                    }
                    prev = Some(*v);
                } else {
                    nulls += 1;
                }
            }
            let all_identical = distinct.is_empty() || (distinct.len() == 1 && nulls == 0);
            (distinct.len(), run_count, all_identical)
        }

        for rows in [0usize, 1, 2, 3, 8, 100, 1000, 4096] {
            for distinct_values in [1usize, 2, 3, 7, 64, 65, 512, 999, 4096] {
                if distinct_values > rows.max(1) {
                    continue;
                }
                for nulls in [0usize, 1] {
                    let owned: Vec<Vec<u8>> = (0..rows)
                        .map(|i| ((i % distinct_values) as u64).to_le_bytes().to_vec())
                        .collect();
                    let sample: Vec<Option<&[u8]>> = owned
                        .iter()
                        .enumerate()
                        .map(|(i, v)| {
                            if nulls == 1 && i % 17 == 0 {
                                None
                            } else {
                                Some(v.as_slice())
                            }
                        })
                        .collect();

                    let capped = compute_sample_stats(&sample);
                    let (true_cardinality, true_runs, true_identical) = uncapped(&sample);

                    assert_eq!(
                        capped.run_count, true_runs,
                        "rows {rows} distinct {distinct_values} nulls {nulls}: runs are                          counted outside the set and must stay exact"
                    );
                    assert_eq!(
                        capped.all_identical, true_identical,
                        "rows {rows} distinct {distinct_values} nulls {nulls}: all_identical                          changed under the cap"
                    );

                    let capped_dict = capped.cardinality < DICTIONARY_MAX_CARDINALITY
                        && capped.cardinality < rows / 2;
                    let true_dict = true_cardinality < DICTIONARY_MAX_CARDINALITY
                        && true_cardinality < rows / 2;
                    assert_eq!(
                        capped_dict, true_dict,
                        "rows {rows} distinct {distinct_values} nulls {nulls}: the dictionary                          decision changed, capped said {} from {} and the true count {} says {}",
                        capped_dict, capped.cardinality, true_cardinality, true_dict
                    );
                }
            }
        }
    }

    /// The whole point of the cap is that a high cardinality column stops
    /// paying for distinct values it will never use. A column where every
    /// row is unique must not retain a set that grows with the row count
    #[test]
    fn a_high_cardinality_column_saturates_instead_of_counting_every_value() {
        let rows = 8192usize;
        let owned: Vec<Vec<u8>> = (0..rows)
            .map(|i| (i as u64).to_le_bytes().to_vec())
            .collect();
        let sample: Vec<Option<&[u8]>> = owned.iter().map(|v| Some(v.as_slice())).collect();
        let stats = compute_sample_stats(&sample);
        assert!(
            stats.cardinality <= rows / 2 + 1,
            "the set kept counting past the point the answer was fixed, reached {}",
            stats.cardinality
        );
        assert!(!stats.all_identical);
        assert!(
            !(stats.cardinality < DICTIONARY_MAX_CARDINALITY && stats.cardinality < rows / 2),
            "an all-distinct column must not be dictionary encoded"
        );
    }
    use super::*;

    /// Every encoding's ranged decode must agree with its full decode over
    /// the same rows, for every range including the empty and the whole.
    ///
    /// The default implementation decodes and takes a slice, so it agrees
    /// by construction. The value of this test is the encodings that
    /// override it to address rows directly: an off-by-one in a bit offset
    /// there returns neighbouring values rather than failing, which is a
    /// wrong answer no caller can detect.
    ///
    /// Both row counts matter. The small one keeps every cumulative layout
    /// below the restart spacing so the head of the stream is replayed, and
    /// the large one crosses several restart boundaries so the seeded replay
    /// is what answers instead.
    #[test]
    fn test_ranged_decode_agrees_with_full_decode_for_every_encoding() {
        for rows in [300usize, 4100] {
            ranged_decode_agrees_with_full_decode(rows);
        }
    }

    fn ranged_decode_agrees_with_full_decode(rows: usize) {
        // Values with a small distinct set, so dictionary and bitpack both
        // apply, and a run structure RLE can use
        let fixed: Vec<u8> = (0..rows)
            .flat_map(|i| (((i / 7) % 11) as i64).to_le_bytes())
            .collect();
        // Shapes that steer FastLanes into each of its layouts, so a fast
        // path that is only correct for one of them cannot pass. Ascending
        // has a step that widens every fiftieth row, which keeps it out of
        // the constant-step closed form and inside the delta stream
        let ascending: Vec<u8> = (0..rows)
            .flat_map(|i| ((i as i64) * 3 + (i as i64) / 50).to_le_bytes())
            .collect();
        let const_step: Vec<u8> = (0..rows)
            .flat_map(|i| ((i as i64) * 8).to_le_bytes())
            .collect();
        // Quadratic growth: first differences rise steadily and second
        // differences stay tiny, which is what delta-of-delta packs
        let quadratic: Vec<u8> = (0..rows)
            .flat_map(|i| (((i * i) / 3 + i) as i64).to_le_bytes())
            .collect();
        let with_outliers: Vec<u8> = (0..rows)
            .flat_map(|i| {
                let v = if i % 97 == 0 {
                    1i64 << 40
                } else {
                    (i % 13) as i64
                };
                v.to_le_bytes()
            })
            .collect();
        // Wide values confined to every third block of a thousand rows, which
        // is what per-mini-block widths exist for
        let bursty: Vec<u8> = (0..rows)
            .flat_map(|i| {
                let v = if (i / 1024) % 3 == 1 {
                    (i as i64) * 1_000_003
                } else {
                    (i % 7) as i64
                };
                v.to_le_bytes()
            })
            .collect();
        let floats: Vec<u8> = (0..rows)
            .flat_map(|i| ((i as f64) * 0.25 + 1.5).to_le_bytes())
            .collect();
        let float_unsorted: Vec<u8> = (0..rows)
            .flat_map(|i| (((i % 37) as f64) * 0.5).to_le_bytes())
            .collect();
        let varlen_rows: Vec<Vec<u8>> = (0..rows)
            .map(|i| format!("value-{:03}", (i / 5) % 17).into_bytes())
            .collect();
        let varlen_refs: Vec<Option<&[u8]>> =
            varlen_rows.iter().map(|v| Some(v.as_slice())).collect();
        let varlen = varlen_pack(&varlen_refs);

        let constant_fixed: Vec<u8> = (0..rows).flat_map(|_| 42i64.to_le_bytes()).collect();

        // The 16-byte layouts mirror the narrow ones and are addressed the
        // same way, so each of them is covered too
        let wide_flat: Vec<u8> = (0..rows)
            .flat_map(|i| (((i / 9) % 5) as i128).to_le_bytes())
            .collect();
        let wide_ascending: Vec<u8> = (0..rows)
            .flat_map(|i| ((i as i128) * 7 + (i as i128) / 40).to_le_bytes())
            .collect();
        let wide_const_step: Vec<u8> = (0..rows)
            .flat_map(|i| ((i as i128) * 4).to_le_bytes())
            .collect();
        let wide_quadratic: Vec<u8> = (0..rows)
            .flat_map(|i| (((i * i) / 5 + 2 * i) as i128).to_le_bytes())
            .collect();
        let wide_outliers: Vec<u8> = (0..rows)
            .flat_map(|i| {
                let v = if i % 89 == 0 {
                    1i128 << 100
                } else {
                    (i % 11) as i128
                };
                v.to_le_bytes()
            })
            .collect();

        let cases: Vec<(EncodingType, &[u8], usize)> = vec![
            (EncodingType::Unencoded, &fixed, 8),
            (EncodingType::BitPack, &fixed, 8),
            (EncodingType::Dictionary, &fixed, 8),
            (EncodingType::Rle, &fixed, 8),
            (EncodingType::Constant, &constant_fixed, 8),
            (EncodingType::Dictionary, &varlen, 0),
            (EncodingType::Unencoded, &varlen, 0),
            (EncodingType::Fsst, &varlen, 0),
            // FastLanes over every shape that steers it to a different
            // layout: flat, ascending (delta), constant step (closed form),
            // quadratic (delta-of-delta), outliers (patched frame of
            // reference) and bursty (per-mini-block widths)
            (EncodingType::FastLanes, &fixed, 8),
            (EncodingType::FastLanes, &ascending, 8),
            (EncodingType::FastLanes, &const_step, 8),
            (EncodingType::FastLanes, &quadratic, 8),
            (EncodingType::FastLanes, &with_outliers, 8),
            (EncodingType::FastLanes, &bursty, 8),
            (EncodingType::FastLanes, &wide_flat, 16),
            (EncodingType::FastLanes, &wide_ascending, 16),
            (EncodingType::FastLanes, &wide_const_step, 16),
            (EncodingType::FastLanes, &wide_quadratic, 16),
            (EncodingType::FastLanes, &wide_outliers, 16),
            (EncodingType::BitPack, &ascending, 8),
            (EncodingType::Rle, &constant_fixed, 8),
            (EncodingType::Alp, &floats, 8),
            (EncodingType::Alp, &float_unsorted, 8),
            (EncodingType::Unencoded, &floats, 8),
        ];

        let mut ranges = vec![
            (0usize, 0usize),
            (0, 1),
            (0, rows),
            (1, 2),
            (7, 8),
            (127, 129),
            (rows - 1, rows),
            (rows / 3, (rows / 3 + 64).min(rows)),
            // Past the end clamps rather than failing
            (rows - 2, rows + 50),
        ];
        // Ranges pinned to the restart spacing: exactly on a boundary, either
        // side of one, and spanning several
        for boundary in [1024usize, 2048, 3072] {
            if boundary < rows {
                ranges.push((boundary, boundary + 1));
                ranges.push((boundary - 1, boundary + 2));
                ranges.push((boundary, rows));
            }
        }
        if rows > 2048 {
            ranges.push((1000, 3000));
            ranges.push((rows - 1024, rows));
        }

        for (kind, data, value_size) in cases {
            let encoding = create_encoding(kind);
            let Ok(encoded) = encoding.encode(data, rows, value_size) else {
                // An encoding that refuses this shape has nothing to check
                continue;
            };
            let full = encoding
                .decode(&encoded, rows, value_size)
                .unwrap_or_else(|e| panic!("{kind:?} value_size {value_size} full decode: {e}"));

            for &(start, end) in &ranges {
                let ranged = encoding
                    .decode_range(&encoded, rows, value_size, start, end)
                    .unwrap_or_else(|e| {
                        panic!("{kind:?} value_size {value_size} range {start}..{end}: {e}")
                    });
                let expected = slice_decoded(&full, rows, value_size, start, end)
                    .expect("slicing the full decode");
                assert_eq!(
                    ranged, expected,
                    "{kind:?} value_size {value_size} rows {rows} disagreed on {start}..{end}"
                );
            }
        }
    }

    #[test]
    fn test_encoding_type_roundtrip() {
        for v in 0..=7u8 {
            let et = EncodingType::from_u8(v).unwrap();
            assert_eq!(et as u8, v);
        }
    }

    #[test]
    fn test_encoding_type_invalid() {
        assert!(EncodingType::from_u8(8).is_err());
        assert!(EncodingType::from_u8(255).is_err());
    }

    #[test]
    fn test_select_constant_all_identical() {
        let val = [1u8, 0, 0, 0];
        let sample: Vec<Option<&[u8]>> = (0..100).map(|_| Some(val.as_slice())).collect();
        assert_eq!(
            select_encoding(TypeId::Int32, &sample),
            EncodingType::Constant
        );
    }

    #[test]
    fn test_one_distinct_value_plus_nulls_is_not_constant() {
        // A NULL occupies a placeholder cell that differs from the value,
        // constant encoding of the raw buffer would fail
        let val = 5i64.to_le_bytes();
        let sample: Vec<Option<&[u8]>> = vec![Some(val.as_slice()), None, Some(val.as_slice())];
        assert_ne!(
            select_encoding(TypeId::Int64, &sample),
            EncodingType::Constant
        );
        let text: Vec<Option<&[u8]>> = vec![Some(b"dave".as_slice()), None];
        assert_ne!(
            select_encoding_varlen(TypeId::Text, &text),
            EncodingType::Constant
        );
        // All null stays constant, every placeholder is identical
        let all_null: Vec<Option<&[u8]>> = vec![None, None, None];
        assert_eq!(
            select_encoding(TypeId::Int64, &all_null),
            EncodingType::Constant
        );
        assert_eq!(
            select_encoding_varlen(TypeId::Text, &all_null),
            EncodingType::Constant
        );
    }

    #[test]
    fn test_select_bitpack_boolean() {
        let t = [1u8];
        let f = [0u8];
        let sample: Vec<Option<&[u8]>> = (0..100)
            .map(|i| {
                if i % 3 == 0 {
                    Some(t.as_slice())
                } else {
                    Some(f.as_slice())
                }
            })
            .collect();
        assert_eq!(
            select_encoding(TypeId::Boolean, &sample),
            EncodingType::BitPack
        );
    }

    #[test]
    fn test_select_dictionary_low_cardinality() {
        let vals: Vec<[u8; 4]> = (0..10u32).map(|v| v.to_le_bytes()).collect();
        let sample: Vec<Option<&[u8]>> = (0..1000).map(|i| Some(vals[i % 10].as_slice())).collect();
        assert_eq!(
            select_encoding(TypeId::Int32, &sample),
            EncodingType::Dictionary
        );
    }

    #[test]
    fn test_select_fastlanes_integer() {
        let vals: Vec<[u8; 4]> = (0..1000u32).map(|v| v.to_le_bytes()).collect();
        let sample: Vec<Option<&[u8]>> = vals.iter().map(|v| Some(v.as_slice())).collect();
        assert_eq!(
            select_encoding(TypeId::Int32, &sample),
            EncodingType::FastLanes
        );
    }

    #[test]
    fn test_select_alp_float() {
        // Values with 2 decimal places encode well with ALP (factor=100).
        let vals: Vec<[u8; 8]> = (0..1000)
            .map(|i| (i as f64 * 0.01 + 100.0).to_le_bytes())
            .collect();
        let sample: Vec<Option<&[u8]>> = vals.iter().map(|v| Some(v.as_slice())).collect();
        assert_eq!(select_encoding(TypeId::Float64, &sample), EncodingType::Alp);
    }

    #[test]
    fn test_select_empty_sample() {
        assert_eq!(select_encoding(TypeId::Int32, &[]), EncodingType::Unencoded);
    }

    #[test]
    fn test_sentinel_i128_column_compacts_and_roundtrips() {
        // System-versioning sys_end pattern: ~every live row holds the same
        // i128 MAX_TIMESTAMP sentinel, a few rows have real end timestamps.
        // The existing Constant/Dictionary/RLE selection already collapses this
        // without a bespoke sub-encoding.
        let sentinel: u128 = 253_402_300_799_000_000u128 * 1_000_000;
        let ended: [u128; 3] = [1_700_000_000_000_000, 1_700_000_005_000_000, 42];
        let mut raw = Vec::new();
        let mut sample: Vec<[u8; 16]> = Vec::with_capacity(2000);
        for i in 0..2000usize {
            let v = if i % 700 == 13 {
                ended[i % 3]
            } else {
                sentinel
            };
            sample.push(v.to_le_bytes());
        }
        let sample_refs: Vec<Option<&[u8]>> = sample.iter().map(|b| Some(b.as_slice())).collect();
        let et = select_encoding(TypeId::Int128, &sample_refs);
        assert!(
            matches!(
                et,
                EncodingType::Constant | EncodingType::Dictionary | EncodingType::Rle
            ),
            "sentinel column should compact, got {:?}",
            et
        );
        for b in &sample {
            raw.extend_from_slice(b);
        }
        let enc = create_encoding(et);
        let encoded = enc.encode(&raw, sample.len(), 16).unwrap();
        assert!(
            encoded.len() < raw.len() / 4,
            "sentinel column must compress hard: {} vs {}",
            encoded.len(),
            raw.len()
        );
        let decoded = enc.decode(&encoded, sample.len(), 16).unwrap();
        assert_eq!(decoded, raw);
    }

    #[test]
    fn test_eval_predicate_on_raw_equality() {
        // 4 rows of i32: [10, 20, 10, 30]
        let mut data = Vec::new();
        data.extend_from_slice(&10u32.to_le_bytes());
        data.extend_from_slice(&20u32.to_le_bytes());
        data.extend_from_slice(&10u32.to_le_bytes());
        data.extend_from_slice(&30u32.to_le_bytes());

        let target = 10u32.to_le_bytes();
        let bitmask = eval_predicate_on_raw(&data, 4, 4, &Predicate::Equality(&target)).unwrap();
        // Rows 0 and 2 match: bits 0 and 2 set = 0b00000101 = 5
        assert_eq!(bitmask[0], 0b00000101);
    }

    #[test]
    fn test_eval_predicate_on_raw_range() {
        let mut data = Vec::new();
        for v in [10u32, 20, 30, 40, 50] {
            data.extend_from_slice(&v.to_le_bytes());
        }
        let lo = 20u32.to_le_bytes();
        let hi = 40u32.to_le_bytes();
        let bitmask = eval_predicate_on_raw(
            &data,
            5,
            4,
            &Predicate::Range {
                low: Some(&lo),
                high: Some(&hi),
            },
        )
        .unwrap();
        // Rows 1,2,3 match: bits 1,2,3 = 0b00001110 = 14
        assert_eq!(bitmask[0], 0b00001110);
    }
}
