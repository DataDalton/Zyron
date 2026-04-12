//! Hybrid search combining full-text search and vector similarity results.
//!
//! Provides rank fusion methods to merge scored result sets from different
//! retrieval systems into a unified ranking. Uses SIMD-accelerated
//! normalization and blending operations.

use std::collections::HashMap;
use std::sync::OnceLock;

use super::types::VectorId;

// ---------------------------------------------------------------------------
// SIMD function pointer types and selection
// ---------------------------------------------------------------------------

/// Finds the min and max values in an f32 array.
type MinMaxFn = unsafe fn(data: *const f32, len: usize) -> (f32, f32);
static MIN_MAX_FN: OnceLock<MinMaxFn> = OnceLock::new();

/// Normalizes an f32 array in-place: out[i] = (data[i] - min) / range.
type NormalizeFn = unsafe fn(data: *mut f32, len: usize, min: f32, inv_range: f32);
static NORMALIZE_FN: OnceLock<NormalizeFn> = OnceLock::new();

/// Blends two f32 arrays: out[i] = alpha * a[i] + (1 - alpha) * b[i].
type BlendFn = unsafe fn(a: *const f32, b: *const f32, out: *mut f32, len: usize, alpha: f32);
static BLEND_FN: OnceLock<BlendFn> = OnceLock::new();

fn get_min_max_fn() -> MinMaxFn {
    *MIN_MAX_FN.get_or_init(select_min_max_fn)
}
fn get_normalize_fn() -> NormalizeFn {
    *NORMALIZE_FN.get_or_init(select_normalize_fn)
}
fn get_blend_fn() -> BlendFn {
    *BLEND_FN.get_or_init(select_blend_fn)
}

// ---------------------------------------------------------------------------
// AVX-512 implementations
// ---------------------------------------------------------------------------

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
#[allow(unsafe_op_in_unsafe_fn)]
unsafe fn min_max_avx512(data: *const f32, len: usize) -> (f32, f32) {
    use std::arch::x86_64::*;
    let chunks = len / 16;
    let mut vmin = _mm512_set1_ps(f32::INFINITY);
    let mut vmax = _mm512_set1_ps(f32::NEG_INFINITY);
    for i in 0..chunks {
        let v = _mm512_loadu_ps(data.add(i * 16));
        vmin = _mm512_min_ps(vmin, v);
        vmax = _mm512_max_ps(vmax, v);
    }
    let mut lo = _mm512_reduce_min_ps(vmin);
    let mut hi = _mm512_reduce_max_ps(vmax);
    for i in (chunks * 16)..len {
        let val = *data.add(i);
        if val < lo {
            lo = val;
        }
        if val > hi {
            hi = val;
        }
    }
    (lo, hi)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
#[allow(unsafe_op_in_unsafe_fn)]
unsafe fn normalize_avx512(data: *mut f32, len: usize, min: f32, inv_range: f32) {
    use std::arch::x86_64::*;
    let vmin = _mm512_set1_ps(min);
    let vscale = _mm512_set1_ps(inv_range);
    let chunks = len / 16;
    for i in 0..chunks {
        let v = _mm512_loadu_ps(data.add(i * 16) as *const f32);
        let shifted = _mm512_sub_ps(v, vmin);
        let scaled = _mm512_mul_ps(shifted, vscale);
        _mm512_storeu_ps(data.add(i * 16), scaled);
    }
    for i in (chunks * 16)..len {
        let val = *data.add(i);
        *data.add(i) = (val - min) * inv_range;
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
#[allow(unsafe_op_in_unsafe_fn)]
unsafe fn blend_avx512(a: *const f32, b: *const f32, out: *mut f32, len: usize, alpha: f32) {
    use std::arch::x86_64::*;
    let valpha = _mm512_set1_ps(alpha);
    let vbeta = _mm512_set1_ps(1.0 - alpha);
    let chunks = len / 16;
    for i in 0..chunks {
        let va = _mm512_loadu_ps(a.add(i * 16));
        let vb = _mm512_loadu_ps(b.add(i * 16));
        let result = _mm512_add_ps(_mm512_mul_ps(valpha, va), _mm512_mul_ps(vbeta, vb));
        _mm512_storeu_ps(out.add(i * 16), result);
    }
    for i in (chunks * 16)..len {
        *out.add(i) = alpha * *a.add(i) + (1.0 - alpha) * *b.add(i);
    }
}

// ---------------------------------------------------------------------------
// AVX2 implementations
// ---------------------------------------------------------------------------

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[allow(unsafe_op_in_unsafe_fn)]
unsafe fn min_max_avx2(data: *const f32, len: usize) -> (f32, f32) {
    use std::arch::x86_64::*;
    let chunks = len / 8;
    let mut vmin = _mm256_set1_ps(f32::INFINITY);
    let mut vmax = _mm256_set1_ps(f32::NEG_INFINITY);
    for i in 0..chunks {
        let v = _mm256_loadu_ps(data.add(i * 8));
        vmin = _mm256_min_ps(vmin, v);
        vmax = _mm256_max_ps(vmax, v);
    }
    // Horizontal reduce
    let mut arr_min = [0.0f32; 8];
    let mut arr_max = [0.0f32; 8];
    _mm256_storeu_ps(arr_min.as_mut_ptr(), vmin);
    _mm256_storeu_ps(arr_max.as_mut_ptr(), vmax);
    let mut lo = arr_min[0];
    let mut hi = arr_max[0];
    for j in 1..8 {
        if arr_min[j] < lo {
            lo = arr_min[j];
        }
        if arr_max[j] > hi {
            hi = arr_max[j];
        }
    }
    for i in (chunks * 8)..len {
        let val = *data.add(i);
        if val < lo {
            lo = val;
        }
        if val > hi {
            hi = val;
        }
    }
    (lo, hi)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[allow(unsafe_op_in_unsafe_fn)]
unsafe fn normalize_avx2(data: *mut f32, len: usize, min: f32, inv_range: f32) {
    use std::arch::x86_64::*;
    let vmin = _mm256_set1_ps(min);
    let vscale = _mm256_set1_ps(inv_range);
    let chunks = len / 8;
    for i in 0..chunks {
        let v = _mm256_loadu_ps(data.add(i * 8) as *const f32);
        let shifted = _mm256_sub_ps(v, vmin);
        let scaled = _mm256_mul_ps(shifted, vscale);
        _mm256_storeu_ps(data.add(i * 8), scaled);
    }
    for i in (chunks * 8)..len {
        let val = *data.add(i);
        *data.add(i) = (val - min) * inv_range;
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[allow(unsafe_op_in_unsafe_fn)]
unsafe fn blend_avx2(a: *const f32, b: *const f32, out: *mut f32, len: usize, alpha: f32) {
    use std::arch::x86_64::*;
    let valpha = _mm256_set1_ps(alpha);
    let vbeta = _mm256_set1_ps(1.0 - alpha);
    let chunks = len / 8;
    for i in 0..chunks {
        let va = _mm256_loadu_ps(a.add(i * 8));
        let vb = _mm256_loadu_ps(b.add(i * 8));
        let result = _mm256_add_ps(_mm256_mul_ps(valpha, va), _mm256_mul_ps(vbeta, vb));
        _mm256_storeu_ps(out.add(i * 8), result);
    }
    for i in (chunks * 8)..len {
        *out.add(i) = alpha * *a.add(i) + (1.0 - alpha) * *b.add(i);
    }
}

// ---------------------------------------------------------------------------
// NEON implementations
// ---------------------------------------------------------------------------

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
#[allow(unsafe_op_in_unsafe_fn)]
unsafe fn min_max_neon(data: *const f32, len: usize) -> (f32, f32) {
    use std::arch::aarch64::*;
    let chunks = len / 4;
    let mut vmin = vdupq_n_f32(f32::INFINITY);
    let mut vmax = vdupq_n_f32(f32::NEG_INFINITY);
    for i in 0..chunks {
        let v = vld1q_f32(data.add(i * 4));
        vmin = vminq_f32(vmin, v);
        vmax = vmaxq_f32(vmax, v);
    }
    let lo = vminvq_f32(vmin);
    let mut hi = vmaxvq_f32(vmax);
    let mut lo = lo;
    for i in (chunks * 4)..len {
        let val = *data.add(i);
        if val < lo {
            lo = val;
        }
        if val > hi {
            hi = val;
        }
    }
    (lo, hi)
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
#[allow(unsafe_op_in_unsafe_fn)]
unsafe fn normalize_neon(data: *mut f32, len: usize, min: f32, inv_range: f32) {
    use std::arch::aarch64::*;
    let vmin = vdupq_n_f32(min);
    let vscale = vdupq_n_f32(inv_range);
    let chunks = len / 4;
    for i in 0..chunks {
        let v = vld1q_f32(data.add(i * 4) as *const f32);
        let shifted = vsubq_f32(v, vmin);
        let scaled = vmulq_f32(shifted, vscale);
        vst1q_f32(data.add(i * 4), scaled);
    }
    for i in (chunks * 4)..len {
        let val = *data.add(i);
        *data.add(i) = (val - min) * inv_range;
    }
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
#[allow(unsafe_op_in_unsafe_fn)]
unsafe fn blend_neon(a: *const f32, b: *const f32, out: *mut f32, len: usize, alpha: f32) {
    use std::arch::aarch64::*;
    let valpha = vdupq_n_f32(alpha);
    let vbeta = vdupq_n_f32(1.0 - alpha);
    let chunks = len / 4;
    for i in 0..chunks {
        let va = vld1q_f32(a.add(i * 4));
        let vb = vld1q_f32(b.add(i * 4));
        let ra = vmulq_f32(valpha, va);
        let result = vfmaq_f32(ra, vbeta, vb);
        vst1q_f32(out.add(i * 4), result);
    }
    for i in (chunks * 4)..len {
        *out.add(i) = alpha * *a.add(i) + (1.0 - alpha) * *b.add(i);
    }
}

// ---------------------------------------------------------------------------
// Fallback implementations
// ---------------------------------------------------------------------------

#[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
unsafe fn min_max_fallback(data: *const f32, len: usize) -> (f32, f32) {
    let mut lo = f32::INFINITY;
    let mut hi = f32::NEG_INFINITY;
    for i in 0..len {
        let val = *data.add(i);
        if val < lo {
            lo = val;
        }
        if val > hi {
            hi = val;
        }
    }
    (lo, hi)
}

#[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
unsafe fn normalize_fallback(data: *mut f32, len: usize, min: f32, inv_range: f32) {
    for i in 0..len {
        let val = *data.add(i);
        *data.add(i) = (val - min) * inv_range;
    }
}

#[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
unsafe fn blend_fallback(a: *const f32, b: *const f32, out: *mut f32, len: usize, alpha: f32) {
    let beta = 1.0 - alpha;
    for i in 0..len {
        *out.add(i) = alpha * *a.add(i) + beta * *b.add(i);
    }
}

// ---------------------------------------------------------------------------
// Selection functions
// ---------------------------------------------------------------------------

#[cfg(target_arch = "x86_64")]
fn select_min_max_fn() -> MinMaxFn {
    if is_x86_feature_detected!("avx512f") {
        min_max_avx512
    } else {
        min_max_avx2
    }
}

#[cfg(target_arch = "aarch64")]
fn select_min_max_fn() -> MinMaxFn {
    min_max_neon
}

#[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
fn select_min_max_fn() -> MinMaxFn {
    min_max_fallback
}

#[cfg(target_arch = "x86_64")]
fn select_normalize_fn() -> NormalizeFn {
    if is_x86_feature_detected!("avx512f") {
        normalize_avx512
    } else {
        normalize_avx2
    }
}

#[cfg(target_arch = "aarch64")]
fn select_normalize_fn() -> NormalizeFn {
    normalize_neon
}

#[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
fn select_normalize_fn() -> NormalizeFn {
    normalize_fallback
}

#[cfg(target_arch = "x86_64")]
fn select_blend_fn() -> BlendFn {
    if is_x86_feature_detected!("avx512f") {
        blend_avx512
    } else {
        blend_avx2
    }
}

#[cfg(target_arch = "aarch64")]
fn select_blend_fn() -> BlendFn {
    blend_neon
}

#[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
fn select_blend_fn() -> BlendFn {
    blend_fallback
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Hybrid search combining full-text search and vector similarity results.
pub struct HybridSearch;

impl HybridSearch {
    /// Combines FTS relevance scores and vector distance scores using
    /// linear interpolation with SIMD-accelerated normalization and blending.
    ///
    /// `fts_results`: (doc_id, BM25 score) pairs from full-text search.
    /// `vec_results`: (VectorId, distance) pairs from vector search.
    ///   Distances are inverted to similarity (smaller distance = higher score).
    /// `alpha`: weight for vector similarity (0.0 = FTS only, 1.0 = vector only).
    /// `k`: number of results to return.
    ///
    /// Returns (doc_id, combined_score) pairs sorted by descending combined score.
    pub fn linear_combination(
        fts_results: &[(u64, f64)],
        vec_results: &[(VectorId, f32)],
        alpha: f32,
        k: usize,
    ) -> Vec<(u64, f64)> {
        if fts_results.is_empty() && vec_results.is_empty() {
            return Vec::new();
        }

        // Safety cap on input sizes. Prevents unbounded memory growth if a
        // caller passes in huge result sets from an upstream bug or attack.
        // 1M combined results is far beyond any realistic hybrid query.
        const MAX_HYBRID_INPUT: usize = 1_000_000;
        let fts_len = fts_results.len().min(MAX_HYBRID_INPUT);
        let vec_len = vec_results.len().min(MAX_HYBRID_INPUT);
        let fts_slice = &fts_results[..fts_len];
        let vec_slice = &vec_results[..vec_len];

        // Pre-size maps to avoid reallocation during insert.
        let mut ftsMap: HashMap<u64, f32> = HashMap::with_capacity(fts_len);
        let mut vecMap: HashMap<u64, f32> = HashMap::with_capacity(vec_len);

        for &(docId, score) in fts_slice {
            ftsMap.insert(docId, score as f32);
        }
        for &(vecId, dist) in vec_slice {
            vecMap.insert(vecId, dist);
        }

        // Collect all unique doc IDs
        let mut allIds: std::collections::HashSet<u64> = std::collections::HashSet::new();
        for &id in ftsMap.keys() {
            allIds.insert(id);
        }
        for &id in vecMap.keys() {
            allIds.insert(id);
        }

        let n = allIds.len();
        if n == 0 {
            return Vec::new();
        }

        let docIds: Vec<u64> = allIds.into_iter().collect();
        let mut ftsScores: Vec<f32> = Vec::with_capacity(n);
        let mut vecSims: Vec<f32> = Vec::with_capacity(n);

        // Normalize FTS scores to [0, 1] using min-max on the actual FTS results
        let ftsMin = ftsMap.values().copied().fold(f32::INFINITY, f32::min);
        let ftsMax = ftsMap.values().copied().fold(f32::NEG_INFINITY, f32::max);
        let ftsRange = ftsMax - ftsMin;

        // Convert vector distances to similarities: 1 - (dist / maxDist)
        // Only consider docs that actually have vector scores
        let vecMaxDist = vecMap.values().copied().fold(f32::NEG_INFINITY, f32::max);

        for &docId in &docIds {
            // FTS: normalize to [0, 1]. Docs without FTS results get 0.0.
            let ftsNorm = match ftsMap.get(&docId) {
                Some(&score) => {
                    if ftsRange > f32::EPSILON {
                        (score - ftsMin) / ftsRange
                    } else if ftsMax > 0.0 {
                        1.0
                    } else {
                        0.0
                    }
                }
                None => 0.0,
            };
            ftsScores.push(ftsNorm);

            // Vector: convert distance to similarity [0, 1].
            // Docs without vector results get 0.0 similarity.
            let vecSim = match vecMap.get(&docId) {
                Some(&dist) => {
                    if vecMaxDist > f32::EPSILON {
                        1.0 - (dist / vecMaxDist)
                    } else {
                        1.0
                    }
                }
                None => 0.0,
            };
            vecSims.push(vecSim);
        }

        // Blend scores using SIMD: result = alpha * vecSim + (1-alpha) * ftsSim
        let mut blended = vec![0.0f32; n];
        if n > 0 {
            let blend = get_blend_fn();
            unsafe {
                blend(
                    vecSims.as_ptr(),
                    ftsScores.as_ptr(),
                    blended.as_mut_ptr(),
                    n,
                    alpha,
                );
            }
        }

        // Combine into results and sort by descending score
        let mut results: Vec<(u64, f64)> = docIds
            .into_iter()
            .zip(blended.iter())
            .map(|(id, &score)| (id, score as f64))
            .collect();

        results.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        results.truncate(k);
        results
    }
}

/// Normalizes an f32 array to [0, 1] using SIMD min/max and normalization.
fn normalize_array(data: &mut [f32]) {
    if data.len() <= 1 {
        return;
    }

    let min_max = get_min_max_fn();
    let (lo, hi) = unsafe { min_max(data.as_ptr(), data.len()) };

    let range = hi - lo;
    if range < f32::EPSILON {
        // All values are the same, set to 0
        for v in data.iter_mut() {
            *v = 0.0;
        }
        return;
    }

    let inv_range = 1.0 / range;
    let normalize = get_normalize_fn();
    unsafe {
        normalize(data.as_mut_ptr(), data.len(), lo, inv_range);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_empty_inputs() {
        let results = HybridSearch::linear_combination(&[], &[], 0.5, 10);
        assert!(results.is_empty());
    }

    #[test]
    fn test_fts_only() {
        let fts = vec![(1u64, 10.0), (2, 5.0), (3, 8.0)];
        let results = HybridSearch::linear_combination(&fts, &[], 0.0, 10);
        assert_eq!(results.len(), 3);
        // With alpha=0, only FTS scores matter. Doc 1 has highest FTS score.
        assert_eq!(results[0].0, 1);
    }

    #[test]
    fn test_vector_only() {
        let vec_results = vec![(10u64, 0.1f32), (20, 0.9), (30, 0.5)];
        let results = HybridSearch::linear_combination(&[], &vec_results, 1.0, 10);
        assert_eq!(results.len(), 3);
        // With alpha=1, only vector similarity matters. Doc 10 is closest (dist 0.1).
        assert_eq!(results[0].0, 10);
    }

    #[test]
    fn test_hybrid_blend() {
        let fts = vec![(1u64, 10.0), (2, 5.0)];
        let vec_results = vec![(1u64, 0.9f32), (2, 0.1)];
        // Doc 1: high FTS, far vector. Doc 2: low FTS, close vector.
        let results = HybridSearch::linear_combination(&fts, &vec_results, 0.5, 10);
        assert_eq!(results.len(), 2);
    }

    #[test]
    fn test_truncation() {
        let fts: Vec<(u64, f64)> = (0..100).map(|i| (i, i as f64)).collect();
        let results = HybridSearch::linear_combination(&fts, &[], 0.0, 5);
        assert_eq!(results.len(), 5);
    }

    #[test]
    fn test_normalize_array() {
        let mut data = vec![2.0f32, 4.0, 6.0, 8.0, 10.0];
        normalize_array(&mut data);
        assert!((data[0] - 0.0).abs() < 1e-6);
        assert!((data[4] - 1.0).abs() < 1e-6);
        assert!((data[2] - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_normalize_uniform() {
        let mut data = vec![5.0f32; 10];
        normalize_array(&mut data);
        // All same value, should normalize to 0
        for &v in &data {
            assert_eq!(v, 0.0);
        }
    }
}
