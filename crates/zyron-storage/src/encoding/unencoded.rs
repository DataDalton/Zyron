//! Raw unencoded fallback. Stores column data as-is with no transformation.
//! Used when no encoding provides compaction benefit over the raw format.

use crate::encoding::{Encoding, EncodingType};
use zyron_common::Result;

pub struct UnencodedEncoding;

impl Encoding for UnencodedEncoding {
    fn encoding_type(&self) -> EncodingType {
        EncodingType::Unencoded
    }

    fn encode(&self, data: &[u8], _row_count: usize, _value_size: usize) -> Result<Vec<u8>> {
        Ok(data.to_vec())
    }

    fn decode(&self, encoded: &[u8], _row_count: usize, _value_size: usize) -> Result<Vec<u8>> {
        Ok(encoded.to_vec())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_roundtrip() {
        let enc = UnencodedEncoding;
        let data: Vec<u8> = (0..100).map(|i| (i * 7 + 3) as u8).collect();
        let encoded = enc.encode(&data, 25, 4).unwrap();
        let decoded = enc.decode(&encoded, 25, 4).unwrap();
        assert_eq!(data, decoded);
    }

    #[test]
    fn test_empty() {
        let enc = UnencodedEncoding;
        let encoded = enc.encode(&[], 0, 4).unwrap();
        let decoded = enc.decode(&encoded, 0, 4).unwrap();
        assert!(decoded.is_empty());
    }
}
