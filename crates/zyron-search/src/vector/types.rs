//! Core types for vector search indexing and retrieval.
//!
//! Defines vector values, distance metrics, index configurations (HNSW, IVF-PQ),
//! catalog-level serialization for index parameters, and the VectorSearch trait.

use serde::{Deserialize, Serialize};
use zyron_common::{Result, ZyronError};

use super::profile::DataProfile;

/// Unique identifier for a vector entry, matching the DocId scheme from FTS.
pub type VectorId = u64;

/// A dense floating-point vector with a fixed dimension count.
#[derive(Debug, Clone, PartialEq)]
pub struct VectorValue {
    dimensions: u16,
    data: Vec<f32>,
}

impl VectorValue {
    /// Creates a new VectorValue from a data slice.
    /// Returns an error if the data is empty or exceeds u16::MAX elements.
    pub fn new(data: Vec<f32>) -> Result<Self> {
        if data.is_empty() {
            return Err(ZyronError::InvalidParameter {
                name: "data".to_string(),
                value: "empty vector".to_string(),
            });
        }
        if data.len() > u16::MAX as usize {
            return Err(ZyronError::InvalidParameter {
                name: "data.len()".to_string(),
                value: format!("{} exceeds max {}", data.len(), u16::MAX),
            });
        }
        Ok(Self {
            dimensions: data.len() as u16,
            data,
        })
    }

    /// Serializes to bytes: 2-byte LE dimension header followed by dimensions * 4 bytes of LE f32.
    pub fn toBytes(&self) -> Vec<u8> {
        let mut buf = Vec::with_capacity(2 + self.data.len() * 4);
        buf.extend_from_slice(&self.dimensions.to_le_bytes());
        for &v in &self.data {
            buf.extend_from_slice(&v.to_le_bytes());
        }
        buf
    }

    /// Deserializes from bytes produced by `toBytes`.
    /// Validates that the byte slice length matches the encoded dimension count.
    pub fn fromBytes(data: &[u8]) -> Result<Self> {
        if data.len() < 2 {
            return Err(ZyronError::InvalidParameter {
                name: "data".to_string(),
                value: format!("too short: {} bytes, need at least 2", data.len()),
            });
        }
        let dimensions = u16::from_le_bytes([data[0], data[1]]);
        if dimensions == 0 {
            return Err(ZyronError::InvalidParameter {
                name: "dimensions".to_string(),
                value: "0".to_string(),
            });
        }
        let expectedLen = 2 + (dimensions as usize) * 4;
        if data.len() != expectedLen {
            return Err(ZyronError::InvalidParameter {
                name: "data.len()".to_string(),
                value: format!("expected {} bytes, got {}", expectedLen, data.len()),
            });
        }
        let mut values = Vec::with_capacity(dimensions as usize);
        for i in 0..dimensions as usize {
            let offset = 2 + i * 4;
            let bytes = [
                data[offset],
                data[offset + 1],
                data[offset + 2],
                data[offset + 3],
            ];
            values.push(f32::from_le_bytes(bytes));
        }
        Ok(Self {
            dimensions,
            data: values,
        })
    }

    /// Returns the number of dimensions.
    pub fn dimensions(&self) -> u16 {
        self.dimensions
    }

    /// Returns the underlying f32 data as a slice.
    pub fn asSlice(&self) -> &[f32] {
        &self.data
    }
}

/// Distance function used for vector similarity comparisons.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DistanceMetric {
    Cosine,
    Euclidean,
    DotProduct,
    Manhattan,
}

/// Configuration for HNSW (Hierarchical Navigable Small World) graph index.
#[derive(Debug, Clone)]
pub struct HnswConfig {
    /// Max connections per node per layer.
    pub m: u16,
    /// Build-time beam width.
    pub efConstruction: u16,
    /// Query-time beam width.
    pub efSearch: u16,
    /// Distance function for similarity comparisons.
    pub metric: DistanceMetric,
}

impl HnswConfig {
    /// Creates an HNSW config with parameters automatically derived from a DataProfile.
    /// All values are computed from the dataset characteristics, no manual tuning needed.
    pub fn auto(profile: &DataProfile, metric: DistanceMetric) -> Self {
        let m = profile.hnswM();
        let efConstruction = profile.hnswEfConstruction(m);
        let efSearch = profile.hnswEfSearch(m);
        Self {
            m,
            efConstruction,
            efSearch,
            metric,
        }
    }
}

impl Default for HnswConfig {
    fn default() -> Self {
        Self {
            m: 32,
            efConstruction: 200,
            efSearch: 128,
            metric: DistanceMetric::Cosine,
        }
    }
}

/// Configuration for IVF-PQ (Inverted File with Product Quantization) index.
#[derive(Debug, Clone)]
pub struct IvfPqConfig {
    /// Number of Voronoi partitions.
    pub numCentroids: u32,
    /// Number of sub-vector segments for product quantization.
    pub numSubvectors: u16,
    /// Bits per PQ code (determines codebook size as 2^bits_per_code).
    pub bitsPerCode: u8,
    /// Number of partitions probed at query time.
    pub numProbes: u16,
    /// Distance function for similarity comparisons.
    pub metric: DistanceMetric,
}

impl IvfPqConfig {
    /// Creates an IVF-PQ config with parameters automatically derived from a DataProfile.
    /// All values are computed from the dataset characteristics, no manual tuning needed.
    pub fn auto(profile: &DataProfile, metric: DistanceMetric) -> Self {
        let numCentroids = profile.ivfNumCentroids();
        let numSubvectors = profile.ivfNumSubvectors();
        let numProbes = profile.ivfNumProbes(numCentroids);
        // 4-bit codes for very large datasets (faster search, smaller memory).
        // 8-bit codes for everything else (better quantization accuracy).
        let bitsPerCode = if profile.n > 10_000_000 { 4 } else { 8 };
        Self {
            numCentroids,
            numSubvectors,
            bitsPerCode,
            numProbes,
            metric,
        }
    }
}

impl Default for IvfPqConfig {
    fn default() -> Self {
        Self {
            numCentroids: 256,
            numSubvectors: 32,
            bitsPerCode: 8,
            numProbes: 10,
            metric: DistanceMetric::Cosine,
        }
    }
}

/// Serializable index parameters stored in the catalog.
/// Encodes all configuration needed to reconstruct a vector index.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum VectorIndexParams {
    Hnsw {
        m: u16,
        efConstruction: u16,
        efSearch: u16,
        metric: u8,
        dimensions: u16,
    },
    IvfPq {
        numCentroids: u32,
        numSubvectors: u16,
        bitsPerCode: u8,
        numProbes: u16,
        metric: u8,
        dimensions: u16,
    },
}

impl VectorIndexParams {
    /// Serializes to a compact binary format.
    /// Byte 0 is a variant tag (0 = Hnsw, 1 = IvfPq), followed by LE field bytes.
    pub fn toBytes(&self) -> Vec<u8> {
        match self {
            VectorIndexParams::Hnsw {
                m,
                efConstruction,
                efSearch,
                metric,
                dimensions,
            } => {
                // 1 tag + 2 + 2 + 2 + 1 + 2 = 10 bytes
                let mut buf = Vec::with_capacity(10);
                buf.push(0u8);
                buf.extend_from_slice(&m.to_le_bytes());
                buf.extend_from_slice(&efConstruction.to_le_bytes());
                buf.extend_from_slice(&efSearch.to_le_bytes());
                buf.push(*metric);
                buf.extend_from_slice(&dimensions.to_le_bytes());
                buf
            }
            VectorIndexParams::IvfPq {
                numCentroids,
                numSubvectors,
                bitsPerCode,
                numProbes,
                metric,
                dimensions,
            } => {
                // 1 tag + 4 + 2 + 1 + 2 + 1 + 2 = 13 bytes
                let mut buf = Vec::with_capacity(13);
                buf.push(1u8);
                buf.extend_from_slice(&numCentroids.to_le_bytes());
                buf.extend_from_slice(&numSubvectors.to_le_bytes());
                buf.push(*bitsPerCode);
                buf.extend_from_slice(&numProbes.to_le_bytes());
                buf.push(*metric);
                buf.extend_from_slice(&dimensions.to_le_bytes());
                buf
            }
        }
    }

    /// Deserializes from bytes produced by `toBytes`.
    pub fn fromBytes(data: &[u8]) -> Result<Self> {
        if data.is_empty() {
            return Err(ZyronError::InvalidParameter {
                name: "data".to_string(),
                value: "empty".to_string(),
            });
        }
        match data[0] {
            0 => {
                if data.len() != 10 {
                    return Err(ZyronError::InvalidParameter {
                        name: "data.len()".to_string(),
                        value: format!("expected 10 bytes for Hnsw, got {}", data.len()),
                    });
                }
                Ok(VectorIndexParams::Hnsw {
                    m: u16::from_le_bytes([data[1], data[2]]),
                    efConstruction: u16::from_le_bytes([data[3], data[4]]),
                    efSearch: u16::from_le_bytes([data[5], data[6]]),
                    metric: data[7],
                    dimensions: u16::from_le_bytes([data[8], data[9]]),
                })
            }
            1 => {
                if data.len() != 13 {
                    return Err(ZyronError::InvalidParameter {
                        name: "data.len()".to_string(),
                        value: format!("expected 13 bytes for IvfPq, got {}", data.len()),
                    });
                }
                Ok(VectorIndexParams::IvfPq {
                    numCentroids: u32::from_le_bytes([data[1], data[2], data[3], data[4]]),
                    numSubvectors: u16::from_le_bytes([data[5], data[6]]),
                    bitsPerCode: data[7],
                    numProbes: u16::from_le_bytes([data[8], data[9]]),
                    metric: data[10],
                    dimensions: u16::from_le_bytes([data[11], data[12]]),
                })
            }
            tag => Err(ZyronError::InvalidParameter {
                name: "variant tag".to_string(),
                value: format!("unknown tag {tag}"),
            }),
        }
    }
}

/// Trait for vector similarity search indexes.
/// Implementations must be safe to share across threads.
pub trait VectorSearch: Send + Sync {
    /// Finds the k nearest neighbors to the query vector.
    /// Returns pairs of (VectorId, distance) sorted by ascending distance.
    fn search(&self, query: &[f32], k: usize, efSearch: u16) -> Result<Vec<(VectorId, f32)>>;

    /// Inserts a vector with the given identifier.
    fn insert(&self, id: VectorId, vector: &[f32]) -> Result<()>;

    /// Removes a vector by identifier.
    fn delete(&self, id: VectorId) -> Result<()>;

    /// Returns the dimensionality of vectors in this index.
    fn dimensions(&self) -> u16;

    /// Returns the distance metric used by this index.
    fn metric(&self) -> DistanceMetric;

    /// Returns the number of vectors currently stored.
    fn len(&self) -> usize;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn vectorValueConstruction() {
        let v = VectorValue::new(vec![1.0, 2.0, 3.0]).expect("valid vector");
        assert_eq!(v.dimensions(), 3);
        assert_eq!(v.asSlice(), &[1.0, 2.0, 3.0]);
    }

    #[test]
    fn vectorValueEmptyError() {
        let result = VectorValue::new(vec![]);
        assert!(result.is_err());
    }

    #[test]
    fn vectorValueSerializationRoundTrip() {
        let original = VectorValue::new(vec![1.5, -2.25, 0.0, 3.14]).expect("valid vector");
        let bytes = original.toBytes();
        assert_eq!(bytes.len(), 2 + 4 * 4);

        let restored = VectorValue::fromBytes(&bytes).expect("valid deserialization");
        assert_eq!(original.dimensions(), restored.dimensions());
        assert_eq!(original.asSlice(), restored.asSlice());
    }

    #[test]
    fn vectorValueFromBytesTooShort() {
        let result = VectorValue::fromBytes(&[0u8]);
        assert!(result.is_err());
    }

    #[test]
    fn vectorValueFromBytesZeroDimensions() {
        let result = VectorValue::fromBytes(&[0u8, 0u8]);
        assert!(result.is_err());
    }

    #[test]
    fn vectorValueFromBytesSizeMismatch() {
        // Header says 2 dimensions (needs 10 bytes total), but only 6 provided.
        let result = VectorValue::fromBytes(&[2u8, 0u8, 0u8, 0u8, 0u8, 0u8]);
        assert!(result.is_err());
    }

    #[test]
    fn hnswConfigDefaults() {
        let cfg = HnswConfig::default();
        assert_eq!(cfg.m, 32);
        assert_eq!(cfg.efConstruction, 200);
        assert_eq!(cfg.efSearch, 128);
        assert_eq!(cfg.metric, DistanceMetric::Cosine);
    }

    #[test]
    fn ivfPqConfigDefaults() {
        let cfg = IvfPqConfig::default();
        assert_eq!(cfg.numCentroids, 256);
        assert_eq!(cfg.numSubvectors, 32);
        assert_eq!(cfg.bitsPerCode, 8);
        assert_eq!(cfg.numProbes, 10);
        assert_eq!(cfg.metric, DistanceMetric::Cosine);
    }

    #[test]
    fn vectorIndexParamsHnswRoundTrip() {
        let params = VectorIndexParams::Hnsw {
            m: 32,
            efConstruction: 400,
            efSearch: 128,
            metric: 1,
            dimensions: 768,
        };
        let bytes = params.toBytes();
        assert_eq!(bytes.len(), 10);
        assert_eq!(bytes[0], 0); // tag

        let restored = VectorIndexParams::fromBytes(&bytes).expect("valid deserialization");
        assert_eq!(params, restored);
    }

    #[test]
    fn vectorIndexParamsIvfPqRoundTrip() {
        let params = VectorIndexParams::IvfPq {
            numCentroids: 1024,
            numSubvectors: 64,
            bitsPerCode: 4,
            numProbes: 20,
            metric: 2,
            dimensions: 384,
        };
        let bytes = params.toBytes();
        assert_eq!(bytes.len(), 13);
        assert_eq!(bytes[0], 1); // tag

        let restored = VectorIndexParams::fromBytes(&bytes).expect("valid deserialization");
        assert_eq!(params, restored);
    }

    #[test]
    fn vectorIndexParamsFromBytesEmpty() {
        let result = VectorIndexParams::fromBytes(&[]);
        assert!(result.is_err());
    }

    #[test]
    fn vectorIndexParamsFromBytesUnknownTag() {
        let result = VectorIndexParams::fromBytes(&[99u8, 0, 0, 0, 0, 0, 0, 0, 0, 0]);
        assert!(result.is_err());
    }

    #[test]
    fn vectorIndexParamsFromBytesWrongLength() {
        // Tag 0 (Hnsw) but only 5 bytes instead of 10.
        let result = VectorIndexParams::fromBytes(&[0u8, 1, 2, 3, 4]);
        assert!(result.is_err());
    }

    #[test]
    fn distanceMetricEquality() {
        assert_eq!(DistanceMetric::Cosine, DistanceMetric::Cosine);
        assert_ne!(DistanceMetric::Euclidean, DistanceMetric::Manhattan);
    }
}
