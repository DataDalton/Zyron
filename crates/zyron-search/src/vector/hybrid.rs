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
unsafe fn min_max_avx512(data: *const f32, len: usize) -> (f32, f32) {
    use std::arch::x86_64::*;

    // SAFETY: caller guarantees the pointers are valid for len f32s and that the required CPU feature is present
    unsafe {
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
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
unsafe fn normalize_avx512(data: *mut f32, len: usize, min: f32, inv_range: f32) {
    use std::arch::x86_64::*;

    // SAFETY: caller guarantees the pointers are valid for len f32s and that the required CPU feature is present
    unsafe {
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
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
unsafe fn blend_avx512(a: *const f32, b: *const f32, out: *mut f32, len: usize, alpha: f32) {
    use std::arch::x86_64::*;

    // SAFETY: caller guarantees the pointers are valid for len f32s and that the required CPU feature is present
    unsafe {
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
}

// ---------------------------------------------------------------------------
// AVX2 implementations
// ---------------------------------------------------------------------------

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn min_max_avx2(data: *const f32, len: usize) -> (f32, f32) {
    use std::arch::x86_64::*;

    // SAFETY: caller guarantees the pointers are valid for len f32s and that the required CPU feature is present
    unsafe {
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
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn normalize_avx2(data: *mut f32, len: usize, min: f32, inv_range: f32) {
    use std::arch::x86_64::*;

    // SAFETY: caller guarantees the pointers are valid for len f32s and that the required CPU feature is present
    unsafe {
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
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn blend_avx2(a: *const f32, b: *const f32, out: *mut f32, len: usize, alpha: f32) {
    use std::arch::x86_64::*;

    // SAFETY: caller guarantees the pointers are valid for len f32s and that the required CPU feature is present
    unsafe {
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
}

// ---------------------------------------------------------------------------
// NEON implementations
// ---------------------------------------------------------------------------

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn min_max_neon(data: *const f32, len: usize) -> (f32, f32) {
    use std::arch::aarch64::*;

    // SAFETY: caller guarantees the pointers are valid for len f32s and that the required CPU feature is present
    unsafe {
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
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn normalize_neon(data: *mut f32, len: usize, min: f32, inv_range: f32) {
    use std::arch::aarch64::*;

    // SAFETY: caller guarantees the pointers are valid for len f32s and that the required CPU feature is present
    unsafe {
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
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn blend_neon(a: *const f32, b: *const f32, out: *mut f32, len: usize, alpha: f32) {
    use std::arch::aarch64::*;

    // SAFETY: caller guarantees the pointers are valid for len f32s and that the required CPU feature is present
    unsafe {
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
}

// ---------------------------------------------------------------------------
// Fallback implementations
// ---------------------------------------------------------------------------

#[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
unsafe fn min_max_fallback(data: *const f32, len: usize) -> (f32, f32) {
    // SAFETY: caller guarantees the pointers are valid for len f32s and that the required CPU feature is present
    unsafe {
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
}

#[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
unsafe fn normalize_fallback(data: *mut f32, len: usize, min: f32, inv_range: f32) {
    // SAFETY: caller guarantees the pointers are valid for len f32s and that the required CPU feature is present
    unsafe {
        for i in 0..len {
            let val = *data.add(i);
            *data.add(i) = (val - min) * inv_range;
        }
    }
}

#[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
unsafe fn blend_fallback(a: *const f32, b: *const f32, out: *mut f32, len: usize, alpha: f32) {
    // SAFETY: caller guarantees the pointers are valid for len f32s and that the required CPU feature is present
    unsafe {
        let beta = 1.0 - alpha;
        for i in 0..len {
            *out.add(i) = alpha * *a.add(i) + beta * *b.add(i);
        }
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

        // Both score families reach the blend as an affine map of a raw
        // value, so both are gathered raw and transformed by the same SIMD
        // kernel rather than one scalar branch per document.
        //
        // FTS wants (score - min) / range. Vector wants 1 - dist / maxDist,
        // which is (dist - maxDist) * (-1 / maxDist), the same shape with a
        // negative scale.
        //
        // A document missing from either map has to end at 0.0. Gathering it
        // as the value that maps to zero, the minimum for FTS and maxDist for
        // vector, gets that from the transform itself, so the gather carries
        // no per-document branch at all.
        // One pass over each map into a contiguous buffer, then one SIMD
        // pass for both bounds. Folding twice per map walked the hash table
        // twice to get two numbers out of the same values.
        //
        // An empty map keeps the identities a fold would have produced, so a
        // search with hits on only one side behaves exactly as before.
        let minMax = get_min_max_fn();
        let ftsValues: Vec<f32> = ftsMap.values().copied().collect();
        let (ftsMin, ftsMax) = if ftsValues.is_empty() {
            (f32::INFINITY, f32::NEG_INFINITY)
        } else {
            unsafe { minMax(ftsValues.as_ptr(), ftsValues.len()) }
        };
        let ftsRange = ftsMax - ftsMin;

        let vecValues: Vec<f32> = vecMap.values().copied().collect();
        let vecMaxDist = if vecValues.is_empty() {
            f32::NEG_INFINITY
        } else {
            unsafe { minMax(vecValues.as_ptr(), vecValues.len()).1 }
        };

        let mut ftsScores: Vec<f32> = Vec::with_capacity(n);
        let mut vecSims: Vec<f32> = Vec::with_capacity(n);
        for &docId in &docIds {
            ftsScores.push(ftsMap.get(&docId).copied().unwrap_or(ftsMin));
            vecSims.push(vecMap.get(&docId).copied().unwrap_or(vecMaxDist));
        }

        let normalize = get_normalize_fn();

        // Every FTS score equal means min-max says nothing, so a document
        // that has one scores 1.0 and a document that has none still scores
        // 0.0. That distinction is the one thing the transform cannot carry,
        // and it needs the map to tell the two apart
        if ftsRange > f32::EPSILON {
            unsafe {
                normalize(ftsScores.as_mut_ptr(), n, ftsMin, 1.0 / ftsRange);
            }
        } else {
            let present = if ftsMax > 0.0 { 1.0 } else { 0.0 };
            for (slot, &docId) in ftsScores.iter_mut().zip(docIds.iter()) {
                *slot = if ftsMap.contains_key(&docId) {
                    present
                } else {
                    0.0
                };
            }
        }

        // Distances all zero means every hit is exact, so each one is a
        // perfect match and a document with no vector hit is still 0.0
        if vecMaxDist > f32::EPSILON {
            unsafe {
                normalize(vecSims.as_mut_ptr(), n, vecMaxDist, -1.0 / vecMaxDist);
            }
        } else {
            for (slot, &docId) in vecSims.iter_mut().zip(docIds.iter()) {
                *slot = if vecMap.contains_key(&docId) {
                    1.0
                } else {
                    0.0
                };
            }
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

        // Ties break on doc id so the same query over the same data returns
        // the same documents every time. The candidate order coming in is set
        // iteration order, which varies per process, so without a total order
        // here a top-k cut through a tie group would keep different documents
        // across restarts
        results.sort_unstable_by(|a, b| {
            b.1.partial_cmp(&a.1)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| a.0.cmp(&b.0))
        });
        results.truncate(k);
        results
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

    /// The same query over the same data returns the same documents, even
    /// when a top-k cut lands inside a group of equal scores. Candidates
    /// arrive in set iteration order, which is not stable across processes,
    /// so the ordering has to be total rather than score-only
    #[test]
    fn test_top_k_is_stable_when_scores_tie() {
        // Every document scores identically, so the cut is entirely inside a
        // tie group and only the tie-break decides who survives
        let fts: Vec<(u64, f64)> = (0..200u64).map(|i| (i, 1.0)).collect();
        let vectors: Vec<(u64, f32)> = (0..200u64).map(|i| (i, 1.0)).collect();

        let first = HybridSearch::linear_combination(&fts, &vectors, 0.5, 10);
        assert_eq!(first.len(), 10);
        for _ in 0..16 {
            let again = HybridSearch::linear_combination(&fts, &vectors, 0.5, 10);
            assert_eq!(
                first, again,
                "a tied top-k must not depend on candidate arrival order"
            );
        }

        // And a partial tie: distinct scores order by score, the tied tail
        // orders by id
        let mixed: Vec<(u64, f64)> = (0..50u64)
            .map(|i| (i, if i < 5 { 10.0 - i as f64 } else { 0.0 }))
            .collect();
        let ranked = HybridSearch::linear_combination(&mixed, &[], 0.0, 12);
        let ids: Vec<u64> = ranked.iter().map(|&(id, _)| id).collect();
        assert_eq!(&ids[..5], &[0, 1, 2, 3, 4], "scored docs rank by score");
        let tail = &ids[5..];
        let mut sortedTail = tail.to_vec();
        sortedTail.sort_unstable();
        assert_eq!(tail, &sortedTail[..], "the tied tail is ordered by id");
    }

    /// The min/max kernel agrees with a scalar fold over the same values,
    /// including the tail past the last full SIMD lane
    #[test]
    fn test_min_max_kernel_matches_a_scalar_fold() {
        let minMax = get_min_max_fn();
        // Lengths either side of a 4, 8 and 16 lane boundary, so the tail
        // handling is exercised rather than assumed
        for len in [1usize, 3, 4, 7, 8, 15, 16, 17, 31, 33, 100] {
            let data: Vec<f32> = (0..len).map(|i| ((i * 37 % 71) as f32) - 20.0).collect();
            let (lo, hi) = unsafe { minMax(data.as_ptr(), len) };
            let expectedLo = data.iter().copied().fold(f32::INFINITY, f32::min);
            let expectedHi = data.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            assert_eq!(lo, expectedLo, "min at len {len}");
            assert_eq!(hi, expectedHi, "max at len {len}");
        }
    }

    /// The normalize kernel is an affine map, which is what lets the scorer
    /// use it for both the FTS direction and the inverted vector direction
    #[test]
    fn test_normalize_kernel_applies_the_affine_map() {
        let normalize = get_normalize_fn();
        for len in [1usize, 3, 8, 17, 64, 100] {
            let source: Vec<f32> = (0..len).map(|i| i as f32).collect();

            // Ascending: (x - min) / range lands on [0, 1]
            let mut ascending = source.clone();
            let range = (len as f32 - 1.0).max(1.0);
            unsafe { normalize(ascending.as_mut_ptr(), len, 0.0, 1.0 / range) };
            for (i, &v) in ascending.iter().enumerate() {
                assert!(
                    (v - (i as f32 / range)).abs() < 1e-6,
                    "len {len} index {i} got {v}"
                );
            }

            // Descending: a negative scale inverts, which is how a distance
            // becomes a similarity
            let mut inverted = source.clone();
            let top = (len as f32) - 1.0;
            unsafe { normalize(inverted.as_mut_ptr(), len, top, -1.0 / range) };
            for (i, &v) in inverted.iter().enumerate() {
                let want = (top - i as f32) / range;
                assert!((v - want).abs() < 1e-6, "len {len} index {i} got {v}");
            }
        }
    }

    /// The SIMD scoring path returns what the straightforward scalar
    /// formulation returns, including for documents present on only one side
    /// and for the degenerate cases where min-max says nothing
    #[test]
    fn test_hybrid_scores_match_the_scalar_reference() {
        fn reference(
            fts: &[(u64, f64)],
            vectors: &[(u64, f32)],
            alpha: f32,
            k: usize,
        ) -> Vec<(u64, f64)> {
            use std::collections::{HashMap, HashSet};
            let ftsMap: HashMap<u64, f32> = fts.iter().map(|&(id, s)| (id, s as f32)).collect();
            let vecMap: HashMap<u64, f32> = vectors.iter().copied().collect();
            let mut ids: HashSet<u64> = HashSet::new();
            ids.extend(ftsMap.keys());
            ids.extend(vecMap.keys());
            if ids.is_empty() {
                return Vec::new();
            }
            let ftsMin = ftsMap.values().copied().fold(f32::INFINITY, f32::min);
            let ftsMax = ftsMap.values().copied().fold(f32::NEG_INFINITY, f32::max);
            let ftsRange = ftsMax - ftsMin;
            let vecMaxDist = vecMap.values().copied().fold(f32::NEG_INFINITY, f32::max);
            let mut out: Vec<(u64, f64)> = ids
                .into_iter()
                .map(|id| {
                    let f = match ftsMap.get(&id) {
                        Some(&s) => {
                            if ftsRange > f32::EPSILON {
                                (s - ftsMin) / ftsRange
                            } else if ftsMax > 0.0 {
                                1.0
                            } else {
                                0.0
                            }
                        }
                        None => 0.0,
                    };
                    let v = match vecMap.get(&id) {
                        Some(&d) => {
                            if vecMaxDist > f32::EPSILON {
                                1.0 - (d / vecMaxDist)
                            } else {
                                1.0
                            }
                        }
                        None => 0.0,
                    };
                    (id, (alpha * v + (1.0 - alpha) * f) as f64)
                })
                .collect();
            out.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
            out.truncate(k);
            out
        }

        let cases: Vec<(Vec<(u64, f64)>, Vec<(u64, f32)>)> = vec![
            // Overlapping, distinct scores
            (
                (0..40).map(|i| (i, (i * 3 % 17) as f64)).collect(),
                (20..60).map(|i| (i, (i % 11) as f32 * 0.5)).collect(),
            ),
            // Disjoint: every document is missing from one side
            (
                (0..10).map(|i| (i, i as f64 + 1.0)).collect(),
                (100..110).map(|i| (i, i as f32 * 0.01)).collect(),
            ),
            // Every FTS score equal, which makes the range degenerate
            (
                (0..12).map(|i| (i, 7.0)).collect(),
                (5..15).map(|i| (i, 2.0)).collect(),
            ),
            // Every FTS score zero, the other degenerate direction
            (
                (0..12).map(|i| (i, 0.0)).collect(),
                (5..15).map(|i| (i, 2.0)).collect(),
            ),
            // Every distance zero, so every vector hit is exact
            (
                (0..8).map(|i| (i, i as f64)).collect(),
                (4..12).map(|i| (i, 0.0)).collect(),
            ),
            // One side empty
            ((0..6).map(|i| (i, i as f64)).collect(), Vec::new()),
            (Vec::new(), (0..6).map(|i| (i, i as f32)).collect()),
        ];

        // k is larger than any union here on purpose. Truncating would compare
        // which members of a tie group each side happened to keep, and with
        // set iteration feeding an unstable sort that is not a property either
        // implementation promises. What is being checked is the score.
        const KEEP_ALL: usize = 1000;

        for (index, (fts, vectors)) in cases.iter().enumerate() {
            for &alpha in &[0.0f32, 0.35, 0.5, 1.0] {
                let got = HybridSearch::linear_combination(fts, vectors, alpha, KEEP_ALL);
                let want = reference(fts, vectors, alpha, KEEP_ALL);
                assert_eq!(got.len(), want.len(), "case {index} alpha {alpha}");
                let gotScores: std::collections::HashMap<u64, f64> = got.iter().copied().collect();
                for (id, wantScore) in want {
                    let gotScore = gotScores
                        .get(&id)
                        .unwrap_or_else(|| panic!("case {index} alpha {alpha} lost doc {id}"));
                    assert!(
                        (gotScore - wantScore).abs() < 1e-6,
                        "case {index} alpha {alpha} doc {id}: {gotScore} vs {wantScore}"
                    );
                }
            }
        }
    }
}
