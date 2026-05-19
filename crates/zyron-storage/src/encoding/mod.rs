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

/// Evaluates a predicate on raw (decoded) column data, producing a packed bitmask.
pub fn eval_predicate_on_raw(
    data: &[u8],
    row_count: usize,
    value_size: usize,
    predicate: &Predicate,
) -> Result<Vec<u8>> {
    if value_size == 0 {
        // Variable-length: lexicographic comparison over the canonical buffer.
        let rows = varlen_slice_rows(data, row_count)?;
        let bitmask_len = row_count.div_ceil(8);
        let mut bitmask = vec![0u8; bitmask_len];
        for (i, value) in rows.iter().enumerate() {
            let matches = match predicate {
                Predicate::Equality(target) => value == target,
                Predicate::Range { low, high } => {
                    let above_low = low.map_or(true, |lo| *value >= lo);
                    let below_high = high.map_or(true, |hi| *value <= hi);
                    above_low && below_high
                }
                Predicate::In(values) => values.iter().any(|t| value == t),
            };
            if matches {
                bitmask[i / 8] |= 1 << (i % 8);
            }
        }
        return Ok(bitmask);
    }
    let bitmask_len = row_count.div_ceil(8);
    let mut bitmask = vec![0u8; bitmask_len];

    // For integer-sized values (1-8 bytes), use numeric u64 comparison
    // instead of lexicographic byte comparison. LE-encoded integers are
    // not sorted by their byte representation at byte boundaries
    // (e.g., 256 = [0,1,0,0] is lexicographically less than 255 = [255,0,0,0]).
    if value_size <= 8
        && let Predicate::Range { low, high } = predicate
    {
        let lo_val = match low {
            Some(lo) => read_as_u64(lo, value_size),
            None => 0,
        };
        let hi_val = match high {
            Some(hi) => read_as_u64(hi, value_size),
            None => u64::MAX,
        };

        for i in 0..row_count {
            let start = i * value_size;
            let end = start + value_size;
            if end > data.len() {
                return Err(ZyronError::DecodingFailed(
                    "data shorter than expected row count".to_string(),
                ));
            }
            let v = read_as_u64(&data[start..end], value_size);
            if v >= lo_val && v <= hi_val {
                bitmask[i / 8] |= 1 << (i % 8);
            }
        }

        return Ok(bitmask);
    }

    for i in 0..row_count {
        let start = i * value_size;
        let end = start + value_size;
        if end > data.len() {
            return Err(ZyronError::DecodingFailed(
                "data shorter than expected row count".to_string(),
            ));
        }
        let value = &data[start..end];

        let matches = match predicate {
            Predicate::Equality(target) => value == *target,
            Predicate::Range { low, high } => {
                let above_low = match low {
                    Some(lo) => value >= *lo,
                    None => true,
                };
                let below_high = match high {
                    Some(hi) => value <= *hi,
                    None => true,
                };
                above_low && below_high
            }
            Predicate::In(values) => values.contains(&value),
        };

        if matches {
            bitmask[i / 8] |= 1 << (i % 8);
        }
    }

    Ok(bitmask)
}

/// Reads up to 8 bytes from a slice as a u64 (little-endian).
#[inline(always)]
fn read_as_u64(bytes: &[u8], value_size: usize) -> u64 {
    let mut buf = [0u8; 8];
    let len = bytes.len().min(value_size).min(8);
    buf[..len].copy_from_slice(&bytes[..len]);
    u64::from_le_bytes(buf)
}

/// Statistics computed from a column sample for encoding selection.
struct ColumnSampleStats {
    cardinality: usize,
    run_count: usize,
    all_identical: bool,
}

/// Computes sample statistics from a set of values.
/// Each value is Option<&[u8]> where None represents null.
fn compute_sample_stats(sample: &[Option<&[u8]>]) -> ColumnSampleStats {
    let mut distinct = hashbrown::HashSet::new();
    let mut run_count = 1usize;
    let mut prev_value: Option<&[u8]> = None;

    for val in sample {
        if let Some(v) = val {
            distinct.insert(*v);

            if let Some(prev) = prev_value {
                if *v != prev {
                    run_count += 1;
                }
            }
            prev_value = Some(*v);
        }
    }

    ColumnSampleStats {
        cardinality: distinct.len(),
        run_count,
        all_identical: distinct.len() <= 1,
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
    if sample.is_empty() {
        return EncodingType::Unencoded;
    }

    let stats = compute_sample_stats(sample);
    let row_count = sample.len();

    // All values identical (including all-null): constant encoding is always optimal
    if stats.all_identical {
        return EncodingType::Constant;
    }

    // Booleans: bit-pack to 1-bit is always the best choice
    if type_id == TypeId::Boolean {
        return EncodingType::BitPack;
    }

    // Statistical heuristics: Dictionary and RLE are chosen based on
    // data characteristics and take priority. They support predicate
    // pushdown on encoded data, which is worth structural overhead.
    if stats.cardinality < 65536 && stats.cardinality < row_count / 2 {
        return EncodingType::Dictionary;
    }

    if stats.run_count < row_count / 10 {
        return EncodingType::Rle;
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
        return EncodingType::Unencoded;
    };

    // Trial-encode: compare type-specific encoding against Unencoded
    // to verify it produces a smaller output.
    let valueSize = sample.iter().find_map(|v| v.map(|b| b.len())).unwrap_or(0);

    if valueSize == 0 {
        return typeCandidate;
    }

    let sampleCount = sample.len().min(1024);
    let trialSample = &sample[..sampleCount];
    let mut rawData = vec![0u8; sampleCount * valueSize];
    for (i, val) in trialSample.iter().enumerate() {
        if let Some(v) = val {
            let start = i * valueSize;
            let end = start + valueSize;
            if v.len() == valueSize && end <= rawData.len() {
                rawData[start..end].copy_from_slice(v);
            }
        }
    }

    let encoder = create_encoding(typeCandidate);
    if let Ok(encoded) = encoder.encode(&rawData, sampleCount, valueSize)
        && encoded.len() < rawData.len()
    {
        return typeCandidate;
    }

    EncodingType::Unencoded
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

    if stats.cardinality < 65536 && stats.cardinality < row_count / 2 {
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
