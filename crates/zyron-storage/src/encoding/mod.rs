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

/// Evaluates a predicate on raw (decoded) column data, producing a packed bitmask.
pub fn eval_predicate_on_raw(
    data: &[u8],
    row_count: usize,
    value_size: usize,
    predicate: &Predicate,
) -> Result<Vec<u8>> {
    let bitmask_len = (row_count + 7) / 8;
    let mut bitmask = vec![0u8; bitmask_len];

    // For integer-sized values (1-8 bytes), use numeric u64 comparison
    // instead of lexicographic byte comparison. LE-encoded integers are
    // not sorted by their byte representation at byte boundaries
    // (e.g., 256 = [0,1,0,0] is lexicographically less than 255 = [255,0,0,0]).
    if value_size <= 8 {
        if let Predicate::Range { low, high } = predicate {
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
            Predicate::In(values) => values.iter().any(|v| value == *v),
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
    #[allow(dead_code)]
    null_count: usize,
    run_count: usize,
    all_identical: bool,
    #[allow(dead_code)]
    sorted_ratio: f64,
}

/// Computes sample statistics from a set of values.
/// Each value is Option<&[u8]> where None represents null.
fn compute_sample_stats(sample: &[Option<&[u8]>]) -> ColumnSampleStats {
    let mut null_count = 0usize;
    let mut distinct = hashbrown::HashSet::new();
    let mut run_count = 1usize;
    let mut sorted_count = 0usize;
    let mut prev_value: Option<&[u8]> = None;
    let mut non_null_count = 0usize;

    for val in sample {
        match val {
            None => null_count += 1,
            Some(v) => {
                non_null_count += 1;
                distinct.insert(*v);

                if let Some(prev) = prev_value {
                    if *v != prev {
                        run_count += 1;
                    }
                    if *v >= prev {
                        sorted_count += 1;
                    }
                }
                prev_value = Some(*v);
            }
        }
    }

    let sorted_ratio = if non_null_count > 1 {
        sorted_count as f64 / (non_null_count - 1) as f64
    } else {
        1.0
    };

    ColumnSampleStats {
        cardinality: distinct.len(),
        null_count,
        run_count,
        all_identical: distinct.len() <= 1,
        sorted_ratio,
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

    // Type-specific candidate and Unencoded fallback for trial-encode
    let typeCandidate = if type_id.is_integer() {
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
    if let Ok(encoded) = encoder.encode(&rawData, sampleCount, valueSize) {
        if encoded.len() < rawData.len() {
            return typeCandidate;
        }
    }

    EncodingType::Unencoded
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
