//! Vector similarity search module for ZyronDB.
//!
//! Provides SIMD-accelerated distance computation, HNSW approximate nearest
//! neighbor indexing, IVF-PQ quantized indexing for large datasets, hybrid
//! FTS+vector rank fusion, and index lifecycle management.

pub mod ann_index;
pub mod distance;
pub mod hybrid;
pub mod manager;
pub mod memory;
pub mod profile;
pub mod quantized_index;
pub mod types;

pub use ann_index::AnnIndex;
pub use distance::{
    DistFn, batchDistances, computeDistance, computeQuantizationBounds, distWithFn,
    euclideanQuantized, euclideanSmall4, euclideanSmall8, quantizeArena, quantizeVector,
    resolveDistFn, vectorAddInplace, vectorNorm, vectorScaleInplace, vectorSubtract,
};
pub use hybrid::HybridSearch;
pub use manager::{VectorIndex, VectorIndexManager};
pub use memory::{
    DEFAULT_BUDGET_FRACTION, MAX_INDEX_FILE_BYTES, MemoryReservation, VectorMemoryBudget,
    available_system_memory, estimate_hnsw_build_peak, estimate_hnsw_memory,
    estimate_ivfpq_build_peak, estimate_ivfpq_memory, try_alloc_default, try_alloc_filled,
    try_alloc_vec, validate_file_size,
};
pub use profile::{DataProfile, QueryTuner};
pub use quantized_index::IvfPqIndex;
pub use types::{
    DistanceMetric, HnswConfig, IvfPqConfig, VectorId, VectorIndexParams, VectorSearch, VectorValue,
};
