//! SIMD-accelerated vector distance computation.
//!
//! Supports four distance metrics: cosine, euclidean, dot product, and manhattan.
//! Each metric dispatches through a function pointer selected at startup based on
//! available CPU features. On x86_64, AVX-512 is preferred with AVX2 as fallback.
//! On aarch64, NEON is used. A scalar loop fallback exists for other platforms.

use std::sync::OnceLock;

use super::types::DistanceMetric;

// ---------------------------------------------------------------------------
// Function pointer type and per-metric OnceLock singletons
// ---------------------------------------------------------------------------

/// Raw distance function pointer type. Resolved once per build to eliminate
/// per-call dispatch overhead (match + OnceLock + indirect call).
pub type DistFn = unsafe fn(a: *const f32, b: *const f32, len: usize) -> f32;

static COSINE_FN: OnceLock<DistFn> = OnceLock::new();
static EUCLIDEAN_FN: OnceLock<DistFn> = OnceLock::new();
static DOT_PRODUCT_FN: OnceLock<DistFn> = OnceLock::new();
static MANHATTAN_FN: OnceLock<DistFn> = OnceLock::new();

fn getCosineFn() -> DistFn {
    *COSINE_FN.get_or_init(selectCosineFn)
}

fn getEuclideanFn() -> DistFn {
    *EUCLIDEAN_FN.get_or_init(selectEuclideanFn)
}

fn getDotProductFn() -> DistFn {
    *DOT_PRODUCT_FN.get_or_init(selectDotProductFn)
}

fn getManhattanFn() -> DistFn {
    *MANHATTAN_FN.get_or_init(selectManhattanFn)
}

// ---------------------------------------------------------------------------
// Public dispatch
// ---------------------------------------------------------------------------

/// Computes the distance between two f32 slices using the given metric.
/// Both slices must have the same length. The caller must guarantee this.
pub fn computeDistance(metric: DistanceMetric, a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len(), "vector lengths must match");
    let len = a.len().min(b.len());
    let f = match metric {
        DistanceMetric::Cosine => getCosineFn(),
        DistanceMetric::Euclidean => getEuclideanFn(),
        DistanceMetric::DotProduct => getDotProductFn(),
        DistanceMetric::Manhattan => getManhattanFn(),
    };
    unsafe { f(a.as_ptr(), b.as_ptr(), len) }
}

/// Resolves a distance metric to a raw function pointer once. Call this at the
/// start of a build or batch operation, then pass the returned DistFn through
/// all hot-path functions to eliminate per-call dispatch overhead.
pub fn resolveDistFn(metric: DistanceMetric) -> DistFn {
    match metric {
        DistanceMetric::Cosine => getCosineFn(),
        DistanceMetric::Euclidean => getEuclideanFn(),
        DistanceMetric::DotProduct => getDotProductFn(),
        DistanceMetric::Manhattan => getManhattanFn(),
    }
}

/// Calls a resolved DistFn from safe code. Zero dispatch overhead.
#[inline(always)]
pub fn distWithFn(f: DistFn, a: &[f32], b: &[f32]) -> f32 {
    let len = a.len().min(b.len());
    unsafe { f(a.as_ptr(), b.as_ptr(), len) }
}

/// Euclidean distance for 4-dimensional subvectors. Fully inlined scalar path
/// that avoids all SIMD setup/teardown overhead. For 4d vectors, the standard
/// SIMD path falls to the scalar tail loop (4 < 8-wide AVX2) with function
/// pointer indirection costing more than the actual computation.
#[inline(always)]
pub fn euclideanSmall4(a: &[f32], b: &[f32]) -> f32 {
    debug_assert!(a.len() >= 4 && b.len() >= 4);
    let d0 = a[0] - b[0];
    let d1 = a[1] - b[1];
    let d2 = a[2] - b[2];
    let d3 = a[3] - b[3];
    (d0 * d0 + d1 * d1 + d2 * d2 + d3 * d3).sqrt()
}

/// Euclidean distance for 8-dimensional subvectors. Inlined scalar path
/// for cases where SIMD width matches but setup overhead dominates.
#[inline(always)]
pub fn euclideanSmall8(a: &[f32], b: &[f32]) -> f32 {
    debug_assert!(a.len() >= 8 && b.len() >= 8);
    let mut sum = 0.0f32;
    for i in 0..8 {
        let d = a[i] - b[i];
        sum += d * d;
    }
    sum.sqrt()
}

/// Computes distances from a query vector to each vector in the batch.
/// Results are appended to `out`. On x86_64, prefetches the next vector
/// while computing the current one.
pub fn batchDistances(
    query: &[f32],
    vectors: &[&[f32]],
    metric: DistanceMetric,
    out: &mut Vec<f32>,
) {
    out.reserve(vectors.len());
    let f = match metric {
        DistanceMetric::Cosine => getCosineFn(),
        DistanceMetric::Euclidean => getEuclideanFn(),
        DistanceMetric::DotProduct => getDotProductFn(),
        DistanceMetric::Manhattan => getManhattanFn(),
    };
    let qPtr = query.as_ptr();
    let qLen = query.len();
    for i in 0..vectors.len() {
        #[cfg(target_arch = "x86_64")]
        {
            if i + 1 < vectors.len() {
                unsafe {
                    use std::arch::x86_64::*;
                    _mm_prefetch(vectors[i + 1].as_ptr() as *const i8, _MM_HINT_T0);
                }
            }
        }
        let v = vectors[i];
        let len = qLen.min(v.len());
        let d = unsafe { f(qPtr, v.as_ptr(), len) };
        out.push(d);
    }
}

// ---------------------------------------------------------------------------
// Scalar quantization for fast approximate distance during beam search
// ---------------------------------------------------------------------------

/// Computes per-dimension min and max values across a set of vectors.
/// Returns (mins, maxs) vectors each of length dims.
pub fn computeQuantizationBounds(vectors: &[f32], dims: usize) -> (Vec<f32>, Vec<f32>) {
    let n = vectors.len() / dims;
    if n == 0 {
        return (vec![0.0; dims], vec![1.0; dims]);
    }
    let mut mins = vec![f32::INFINITY; dims];
    let mut maxs = vec![f32::NEG_INFINITY; dims];
    for i in 0..n {
        let base = i * dims;
        for d in 0..dims {
            let v = vectors[base + d];
            if v < mins[d] {
                mins[d] = v;
            }
            if v > maxs[d] {
                maxs[d] = v;
            }
        }
    }
    // Ensure no zero-range dimensions.
    for d in 0..dims {
        if (maxs[d] - mins[d]).abs() < 1e-10 {
            maxs[d] = mins[d] + 1.0;
        }
    }
    (mins, maxs)
}

/// Quantizes a single f32 vector to u8 using per-dimension min/max scaling.
/// out must have length >= src.len().
pub fn quantizeVector(src: &[f32], mins: &[f32], scales: &[f32], out: &mut [u8]) {
    for i in 0..src.len() {
        out[i] = ((src[i] - mins[i]) * scales[i]).clamp(0.0, 255.0) as u8;
    }
}

/// Quantizes an entire arena of vectors in bulk.
/// Returns the quantized arena and the scale factors (255 / (max - min) per dim).
pub fn quantizeArena(
    arena: &[f32],
    dims: usize,
    mins: &[f32],
    maxs: &[f32],
) -> (Vec<u8>, Vec<f32>) {
    let n = arena.len() / dims;
    let mut scales = vec![0.0f32; dims];
    for d in 0..dims {
        scales[d] = 255.0 / (maxs[d] - mins[d]);
    }
    let mut quantized = vec![0u8; arena.len()];
    for i in 0..n {
        let base = i * dims;
        quantizeVector(
            &arena[base..base + dims],
            mins,
            &scales,
            &mut quantized[base..base + dims],
        );
    }
    (quantized, scales)
}

/// Squared Euclidean distance between two u8 vectors using SIMD integer ops.
/// Returns the sum of squared differences as u32.
/// SIMD path processes 16 bytes per iteration via AVX2 pmaddwd, with a
/// scalar tail loop handling the remainder when dims is not a multiple of
/// the vector width.
#[inline]
pub fn euclideanQuantized(a: &[u8], b: &[u8]) -> u32 {
    debug_assert_eq!(a.len(), b.len());
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            return unsafe { euclideanQuantizedAvx2(a, b) };
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        if std::arch::is_aarch64_feature_detected!("neon") {
            return unsafe { euclideanQuantizedNeon(a, b) };
        }
    }
    euclideanQuantizedScalar(a, b)
}

#[inline]
fn euclideanQuantizedScalar(a: &[u8], b: &[u8]) -> u32 {
    let mut sum = 0u32;
    for i in 0..a.len() {
        let diff = a[i] as i32 - b[i] as i32;
        sum += (diff * diff) as u32;
    }
    sum
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn euclideanQuantizedAvx2(a: &[u8], b: &[u8]) -> u32 {
    use std::arch::x86_64::*;
    let len = a.len();
    let mut acc = _mm256_setzero_si256();
    let chunks = len / 16;

    // Process 16 u8 bytes per iteration. Unpack to i16, compute diff, square via pmaddwd.
    for c in 0..chunks {
        let offset = c * 16;
        // Load 16 bytes from each input (lower 128 bits of 256-bit reg).
        let a16 = _mm_loadu_si128(a.as_ptr().add(offset) as *const _);
        let b16 = _mm_loadu_si128(b.as_ptr().add(offset) as *const _);
        // Zero-extend 16 u8 to 16 i16 (256-bit reg).
        let a256 = _mm256_cvtepu8_epi16(a16);
        let b256 = _mm256_cvtepu8_epi16(b16);
        // Signed difference (i16).
        let diff = _mm256_sub_epi16(a256, b256);
        // Square via pmaddwd: multiplies pairs of i16, adds adjacent pairs to i32.
        // diff * diff gives squared differences as i32 (8 values per 256-bit reg).
        let squared = _mm256_madd_epi16(diff, diff);
        acc = _mm256_add_epi32(acc, squared);
    }

    // Horizontal sum of 8 i32 lanes.
    let hi = _mm256_extracti128_si256(acc, 1);
    let lo = _mm256_castsi256_si128(acc);
    let sum128 = _mm_add_epi32(lo, hi);
    let sum64 = _mm_add_epi32(sum128, _mm_shuffle_epi32(sum128, 0b_01_00_11_10));
    let sum32 = _mm_add_epi32(sum64, _mm_shuffle_epi32(sum64, 0b_00_00_00_01));
    let mut total = _mm_cvtsi128_si32(sum32) as u32;

    // Scalar tail for remaining bytes.
    for i in (chunks * 16)..len {
        let diff = a[i] as i32 - b[i] as i32;
        total += (diff * diff) as u32;
    }
    total
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn euclideanQuantizedNeon(a: &[u8], b: &[u8]) -> u32 {
    use std::arch::aarch64::*;
    let len = a.len();
    let mut acc = vdupq_n_u32(0);
    let chunks = len / 16;

    for c in 0..chunks {
        let offset = c * 16;
        let av = vld1q_u8(a.as_ptr().add(offset));
        let bv = vld1q_u8(b.as_ptr().add(offset));
        // Absolute difference of u8 (preserves ordering as a proxy).
        // Actually compute squared diff: unpack to u16, subtract, multiply.
        let a_lo = vmovl_u8(vget_low_u8(av));
        let a_hi = vmovl_u8(vget_high_u8(av));
        let b_lo = vmovl_u8(vget_low_u8(bv));
        let b_hi = vmovl_u8(vget_high_u8(bv));
        // Signed diff (cast to i16 internally via abdq).
        let d_lo = vabdq_u16(a_lo, b_lo);
        let d_hi = vabdq_u16(a_hi, b_hi);
        // Multiply-accumulate squared diff.
        let sq_lo_lo = vmull_u16(vget_low_u16(d_lo), vget_low_u16(d_lo));
        let sq_lo_hi = vmull_u16(vget_high_u16(d_lo), vget_high_u16(d_lo));
        let sq_hi_lo = vmull_u16(vget_low_u16(d_hi), vget_low_u16(d_hi));
        let sq_hi_hi = vmull_u16(vget_high_u16(d_hi), vget_high_u16(d_hi));
        acc = vaddq_u32(acc, sq_lo_lo);
        acc = vaddq_u32(acc, sq_lo_hi);
        acc = vaddq_u32(acc, sq_hi_lo);
        acc = vaddq_u32(acc, sq_hi_hi);
    }

    let mut total = vaddvq_u32(acc);
    for i in (chunks * 16)..len {
        let diff = a[i] as i32 - b[i] as i32;
        total += (diff * diff) as u32;
    }
    total
}

// ---------------------------------------------------------------------------
// SIMD utility functions for k-means and product quantization
// ---------------------------------------------------------------------------

/// Element-wise addition: dst[i] += src[i]. Both slices must have equal length.
pub fn vectorAddInplace(dst: &mut [f32], src: &[f32]) {
    debug_assert_eq!(dst.len(), src.len());
    let len = dst.len().min(src.len());
    let mut i = 0usize;

    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            unsafe {
                use std::arch::x86_64::*;
                while i + 8 <= len {
                    let va = _mm256_loadu_ps(dst.as_ptr().add(i));
                    let vb = _mm256_loadu_ps(src.as_ptr().add(i));
                    _mm256_storeu_ps(dst.as_mut_ptr().add(i), _mm256_add_ps(va, vb));
                    i += 8;
                }
            }
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        unsafe {
            use std::arch::aarch64::*;
            while i + 4 <= len {
                let va = vld1q_f32(dst.as_ptr().add(i));
                let vb = vld1q_f32(src.as_ptr().add(i));
                vst1q_f32(dst.as_mut_ptr().add(i), vaddq_f32(va, vb));
                i += 4;
            }
        }
    }

    while i < len {
        dst[i] += src[i];
        i += 1;
    }
}

/// Scalar multiplication in place: dst[i] *= scalar.
pub fn vectorScaleInplace(dst: &mut [f32], scalar: f32) {
    let len = dst.len();
    let mut i = 0usize;

    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            unsafe {
                use std::arch::x86_64::*;
                let vs = _mm256_set1_ps(scalar);
                while i + 8 <= len {
                    let va = _mm256_loadu_ps(dst.as_ptr().add(i));
                    _mm256_storeu_ps(dst.as_mut_ptr().add(i), _mm256_mul_ps(va, vs));
                    i += 8;
                }
            }
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        unsafe {
            use std::arch::aarch64::*;
            let vs = vdupq_n_f32(scalar);
            while i + 4 <= len {
                let va = vld1q_f32(dst.as_ptr().add(i));
                vst1q_f32(dst.as_mut_ptr().add(i), vmulq_f32(va, vs));
                i += 4;
            }
        }
    }

    while i < len {
        dst[i] *= scalar;
        i += 1;
    }
}

/// Element-wise subtraction: out[i] = a[i] - b[i]. All slices must have equal length.
pub fn vectorSubtract(a: &[f32], b: &[f32], out: &mut [f32]) {
    debug_assert_eq!(a.len(), b.len());
    debug_assert_eq!(a.len(), out.len());
    let len = a.len().min(b.len()).min(out.len());
    let mut i = 0usize;

    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            unsafe {
                use std::arch::x86_64::*;
                while i + 8 <= len {
                    let va = _mm256_loadu_ps(a.as_ptr().add(i));
                    let vb = _mm256_loadu_ps(b.as_ptr().add(i));
                    _mm256_storeu_ps(out.as_mut_ptr().add(i), _mm256_sub_ps(va, vb));
                    i += 8;
                }
            }
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        unsafe {
            use std::arch::aarch64::*;
            while i + 4 <= len {
                let va = vld1q_f32(a.as_ptr().add(i));
                let vb = vld1q_f32(b.as_ptr().add(i));
                vst1q_f32(out.as_mut_ptr().add(i), vsubq_f32(va, vb));
                i += 4;
            }
        }
    }

    while i < len {
        out[i] = a[i] - b[i];
        i += 1;
    }
}

/// L2 norm: sqrt(sum of squares). Returns 0.0 for empty vectors.
pub fn vectorNorm(v: &[f32]) -> f32 {
    if v.is_empty() {
        return 0.0;
    }
    let len = v.len();
    let mut acc = 0.0f32;
    let mut i = 0usize;

    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            unsafe {
                use std::arch::x86_64::*;
                let mut vacc = _mm256_setzero_ps();
                while i + 8 <= len {
                    let va = _mm256_loadu_ps(v.as_ptr().add(i));
                    vacc = _mm256_fmadd_ps(va, va, vacc);
                    i += 8;
                }
                // Horizontal sum of 8 floats
                let hi = _mm256_extractf128_ps(vacc, 1);
                let lo = _mm256_castps256_ps128(vacc);
                let sum4 = _mm_add_ps(lo, hi);
                let shuf = _mm_movehdup_ps(sum4);
                let sum2 = _mm_add_ps(sum4, shuf);
                let shuf2 = _mm_movehl_ps(sum2, sum2);
                let result = _mm_add_ss(sum2, shuf2);
                acc = _mm_cvtss_f32(result);
            }
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        unsafe {
            use std::arch::aarch64::*;
            let mut vacc = vdupq_n_f32(0.0);
            while i + 4 <= len {
                let va = vld1q_f32(v.as_ptr().add(i));
                vacc = vfmaq_f32(vacc, va, va);
                i += 4;
            }
            acc = vaddvq_f32(vacc);
        }
    }

    while i < len {
        acc += v[i] * v[i];
        i += 1;
    }

    acc.sqrt()
}

// ---------------------------------------------------------------------------
// Selection functions (per metric)
// ---------------------------------------------------------------------------

#[cfg(target_arch = "x86_64")]
fn selectCosineFn() -> DistFn {
    if is_x86_feature_detected!("avx512f") {
        cosineAvx512
    } else {
        cosineAvx2
    }
}

#[cfg(target_arch = "aarch64")]
fn selectCosineFn() -> DistFn {
    cosineNeon
}

#[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
fn selectCosineFn() -> DistFn {
    cosineGeneric
}

#[cfg(target_arch = "x86_64")]
fn selectEuclideanFn() -> DistFn {
    if is_x86_feature_detected!("avx512f") {
        euclideanAvx512
    } else {
        euclideanAvx2
    }
}

#[cfg(target_arch = "aarch64")]
fn selectEuclideanFn() -> DistFn {
    euclideanNeon
}

#[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
fn selectEuclideanFn() -> DistFn {
    euclideanGeneric
}

#[cfg(target_arch = "x86_64")]
fn selectDotProductFn() -> DistFn {
    if is_x86_feature_detected!("avx512f") {
        dotProductAvx512
    } else {
        dotProductAvx2
    }
}

#[cfg(target_arch = "aarch64")]
fn selectDotProductFn() -> DistFn {
    dotProductNeon
}

#[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
fn selectDotProductFn() -> DistFn {
    dotProductGeneric
}

#[cfg(target_arch = "x86_64")]
fn selectManhattanFn() -> DistFn {
    if is_x86_feature_detected!("avx512f") {
        manhattanAvx512
    } else {
        manhattanAvx2
    }
}

#[cfg(target_arch = "aarch64")]
fn selectManhattanFn() -> DistFn {
    manhattanNeon
}

#[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
fn selectManhattanFn() -> DistFn {
    manhattanGeneric
}

// ===========================================================================
// Cosine distance implementations
// ===========================================================================

// Cosine distance = 1 - (a . b) / (||a|| * ||b||)

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
#[allow(unsafe_op_in_unsafe_fn)]
unsafe fn cosineAvx512(a: *const f32, b: *const f32, len: usize) -> f32 {
    use std::arch::x86_64::*;
    let mut dotAcc = _mm512_setzero_ps();
    let mut normAacc = _mm512_setzero_ps();
    let mut normBacc = _mm512_setzero_ps();
    let chunks = len / 16;
    for c in 0..chunks {
        let off = c * 16;
        let va = _mm512_loadu_ps(a.add(off));
        let vb = _mm512_loadu_ps(b.add(off));
        dotAcc = _mm512_fmadd_ps(va, vb, dotAcc);
        normAacc = _mm512_fmadd_ps(va, va, normAacc);
        normBacc = _mm512_fmadd_ps(vb, vb, normBacc);
    }
    let mut dot = _mm512_reduce_add_ps(dotAcc);
    let mut normA = _mm512_reduce_add_ps(normAacc);
    let mut normB = _mm512_reduce_add_ps(normBacc);
    for i in (chunks * 16)..len {
        let va = *a.add(i);
        let vb = *b.add(i);
        dot += va * vb;
        normA += va * va;
        normB += vb * vb;
    }
    let denom = normA.sqrt() * normB.sqrt();
    if denom == 0.0 {
        return 0.0;
    }
    1.0 - dot / denom
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "fma")]
#[allow(unsafe_op_in_unsafe_fn)]
unsafe fn cosineAvx2(a: *const f32, b: *const f32, len: usize) -> f32 {
    use std::arch::x86_64::*;
    let mut dotAcc = _mm256_setzero_ps();
    let mut normAacc = _mm256_setzero_ps();
    let mut normBacc = _mm256_setzero_ps();
    let chunks = len / 8;
    for c in 0..chunks {
        let off = c * 8;
        let va = _mm256_loadu_ps(a.add(off));
        let vb = _mm256_loadu_ps(b.add(off));
        dotAcc = _mm256_fmadd_ps(va, vb, dotAcc);
        normAacc = _mm256_fmadd_ps(va, va, normAacc);
        normBacc = _mm256_fmadd_ps(vb, vb, normBacc);
    }
    let dot = hsum256(dotAcc);
    let mut normA = hsum256(normAacc);
    let mut normB = hsum256(normBacc);
    let mut dotScalar = dot;
    for i in (chunks * 8)..len {
        let va = *a.add(i);
        let vb = *b.add(i);
        dotScalar += va * vb;
        normA += va * va;
        normB += vb * vb;
    }
    let denom = normA.sqrt() * normB.sqrt();
    if denom == 0.0 {
        return 0.0;
    }
    1.0 - dotScalar / denom
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
#[allow(unsafe_op_in_unsafe_fn)]
unsafe fn cosineNeon(a: *const f32, b: *const f32, len: usize) -> f32 {
    use std::arch::aarch64::*;
    let mut dotAcc = vdupq_n_f32(0.0);
    let mut normAacc = vdupq_n_f32(0.0);
    let mut normBacc = vdupq_n_f32(0.0);
    let chunks = len / 4;
    for c in 0..chunks {
        let off = c * 4;
        let va = vld1q_f32(a.add(off));
        let vb = vld1q_f32(b.add(off));
        dotAcc = vfmaq_f32(dotAcc, va, vb);
        normAacc = vfmaq_f32(normAacc, va, va);
        normBacc = vfmaq_f32(normBacc, vb, vb);
    }
    let mut dot = vaddvq_f32(dotAcc);
    let mut normA = vaddvq_f32(normAacc);
    let mut normB = vaddvq_f32(normBacc);
    for i in (chunks * 4)..len {
        let va = *a.add(i);
        let vb = *b.add(i);
        dot += va * vb;
        normA += va * va;
        normB += vb * vb;
    }
    let denom = normA.sqrt() * normB.sqrt();
    if denom == 0.0 {
        return 0.0;
    }
    1.0 - dot / denom
}

#[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
unsafe fn cosineGeneric(a: *const f32, b: *const f32, len: usize) -> f32 {
    let mut dot = 0.0f32;
    let mut normA = 0.0f32;
    let mut normB = 0.0f32;
    for i in 0..len {
        let va = *a.add(i);
        let vb = *b.add(i);
        dot += va * vb;
        normA += va * va;
        normB += vb * vb;
    }
    let denom = normA.sqrt() * normB.sqrt();
    if denom == 0.0 {
        return 0.0;
    }
    1.0 - dot / denom
}

// ===========================================================================
// Euclidean distance implementations
// ===========================================================================

// Euclidean (L2) distance = sqrt(sum((a[i] - b[i])^2))

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
#[allow(unsafe_op_in_unsafe_fn)]
unsafe fn euclideanAvx512(a: *const f32, b: *const f32, len: usize) -> f32 {
    use std::arch::x86_64::*;
    let mut acc = _mm512_setzero_ps();
    let chunks = len / 16;
    for c in 0..chunks {
        let off = c * 16;
        let va = _mm512_loadu_ps(a.add(off));
        let vb = _mm512_loadu_ps(b.add(off));
        let diff = _mm512_sub_ps(va, vb);
        acc = _mm512_fmadd_ps(diff, diff, acc);
    }
    let mut sum = _mm512_reduce_add_ps(acc);
    for i in (chunks * 16)..len {
        let d = *a.add(i) - *b.add(i);
        sum += d * d;
    }
    sum.sqrt()
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "fma")]
#[allow(unsafe_op_in_unsafe_fn)]
unsafe fn euclideanAvx2(a: *const f32, b: *const f32, len: usize) -> f32 {
    use std::arch::x86_64::*;
    let mut acc = _mm256_setzero_ps();
    let chunks = len / 8;
    for c in 0..chunks {
        let off = c * 8;
        let va = _mm256_loadu_ps(a.add(off));
        let vb = _mm256_loadu_ps(b.add(off));
        let diff = _mm256_sub_ps(va, vb);
        acc = _mm256_fmadd_ps(diff, diff, acc);
    }
    let mut sum = hsum256(acc);
    for i in (chunks * 8)..len {
        let d = *a.add(i) - *b.add(i);
        sum += d * d;
    }
    sum.sqrt()
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
#[allow(unsafe_op_in_unsafe_fn)]
unsafe fn euclideanNeon(a: *const f32, b: *const f32, len: usize) -> f32 {
    use std::arch::aarch64::*;
    let mut acc = vdupq_n_f32(0.0);
    let chunks = len / 4;
    for c in 0..chunks {
        let off = c * 4;
        let va = vld1q_f32(a.add(off));
        let vb = vld1q_f32(b.add(off));
        let diff = vsubq_f32(va, vb);
        acc = vfmaq_f32(acc, diff, diff);
    }
    let mut sum = vaddvq_f32(acc);
    for i in (chunks * 4)..len {
        let d = *a.add(i) - *b.add(i);
        sum += d * d;
    }
    sum.sqrt()
}

#[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
unsafe fn euclideanGeneric(a: *const f32, b: *const f32, len: usize) -> f32 {
    let mut sum = 0.0f32;
    for i in 0..len {
        let d = *a.add(i) - *b.add(i);
        sum += d * d;
    }
    sum.sqrt()
}

// ===========================================================================
// Dot product distance implementations
// ===========================================================================

// Negative dot product: -sum(a[i] * b[i]) so that smaller = more similar.

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
#[allow(unsafe_op_in_unsafe_fn)]
unsafe fn dotProductAvx512(a: *const f32, b: *const f32, len: usize) -> f32 {
    use std::arch::x86_64::*;
    let mut acc = _mm512_setzero_ps();
    let chunks = len / 16;
    for c in 0..chunks {
        let off = c * 16;
        let va = _mm512_loadu_ps(a.add(off));
        let vb = _mm512_loadu_ps(b.add(off));
        acc = _mm512_fmadd_ps(va, vb, acc);
    }
    let mut sum = _mm512_reduce_add_ps(acc);
    for i in (chunks * 16)..len {
        sum += *a.add(i) * *b.add(i);
    }
    -sum
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "fma")]
#[allow(unsafe_op_in_unsafe_fn)]
unsafe fn dotProductAvx2(a: *const f32, b: *const f32, len: usize) -> f32 {
    use std::arch::x86_64::*;
    let mut acc = _mm256_setzero_ps();
    let chunks = len / 8;
    for c in 0..chunks {
        let off = c * 8;
        let va = _mm256_loadu_ps(a.add(off));
        let vb = _mm256_loadu_ps(b.add(off));
        acc = _mm256_fmadd_ps(va, vb, acc);
    }
    let mut sum = hsum256(acc);
    for i in (chunks * 8)..len {
        sum += *a.add(i) * *b.add(i);
    }
    -sum
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
#[allow(unsafe_op_in_unsafe_fn)]
unsafe fn dotProductNeon(a: *const f32, b: *const f32, len: usize) -> f32 {
    use std::arch::aarch64::*;
    let mut acc = vdupq_n_f32(0.0);
    let chunks = len / 4;
    for c in 0..chunks {
        let off = c * 4;
        let va = vld1q_f32(a.add(off));
        let vb = vld1q_f32(b.add(off));
        acc = vfmaq_f32(acc, va, vb);
    }
    let mut sum = vaddvq_f32(acc);
    for i in (chunks * 4)..len {
        sum += *a.add(i) * *b.add(i);
    }
    -sum
}

#[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
unsafe fn dotProductGeneric(a: *const f32, b: *const f32, len: usize) -> f32 {
    let mut sum = 0.0f32;
    for i in 0..len {
        sum += *a.add(i) * *b.add(i);
    }
    -sum
}

// ===========================================================================
// Manhattan distance implementations
// ===========================================================================

// Manhattan (L1) distance = sum(|a[i] - b[i]|)

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
#[allow(unsafe_op_in_unsafe_fn)]
unsafe fn manhattanAvx512(a: *const f32, b: *const f32, len: usize) -> f32 {
    use std::arch::x86_64::*;
    // Mask to clear the sign bit of f32 values (0x7FFF_FFFF)
    let signMask = _mm512_castsi512_ps(_mm512_set1_epi32(0x7FFF_FFFFu32 as i32));
    let mut acc = _mm512_setzero_ps();
    let chunks = len / 16;
    for c in 0..chunks {
        let off = c * 16;
        let va = _mm512_loadu_ps(a.add(off));
        let vb = _mm512_loadu_ps(b.add(off));
        let diff = _mm512_sub_ps(va, vb);
        let absDiff = _mm512_and_ps(diff, signMask);
        acc = _mm512_add_ps(acc, absDiff);
    }
    let mut sum = _mm512_reduce_add_ps(acc);
    for i in (chunks * 16)..len {
        sum += (*a.add(i) - *b.add(i)).abs();
    }
    sum
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[allow(unsafe_op_in_unsafe_fn)]
unsafe fn manhattanAvx2(a: *const f32, b: *const f32, len: usize) -> f32 {
    use std::arch::x86_64::*;
    // Sign bit mask for f32: set all bits except the sign bit
    let signBit = _mm256_set1_ps(f32::from_bits(0x8000_0000));
    let mut acc = _mm256_setzero_ps();
    let chunks = len / 8;
    for c in 0..chunks {
        let off = c * 8;
        let va = _mm256_loadu_ps(a.add(off));
        let vb = _mm256_loadu_ps(b.add(off));
        let diff = _mm256_sub_ps(va, vb);
        let absDiff = _mm256_andnot_ps(signBit, diff);
        acc = _mm256_add_ps(acc, absDiff);
    }
    let mut sum = hsum256(acc);
    for i in (chunks * 8)..len {
        sum += (*a.add(i) - *b.add(i)).abs();
    }
    sum
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
#[allow(unsafe_op_in_unsafe_fn)]
unsafe fn manhattanNeon(a: *const f32, b: *const f32, len: usize) -> f32 {
    use std::arch::aarch64::*;
    let mut acc = vdupq_n_f32(0.0);
    let chunks = len / 4;
    for c in 0..chunks {
        let off = c * 4;
        let va = vld1q_f32(a.add(off));
        let vb = vld1q_f32(b.add(off));
        acc = vaddq_f32(acc, vabdq_f32(va, vb));
    }
    let mut sum = vaddvq_f32(acc);
    for i in (chunks * 4)..len {
        sum += (*a.add(i) - *b.add(i)).abs();
    }
    sum
}

#[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
unsafe fn manhattanGeneric(a: *const f32, b: *const f32, len: usize) -> f32 {
    let mut sum = 0.0f32;
    for i in 0..len {
        sum += (*a.add(i) - *b.add(i)).abs();
    }
    sum
}

// ===========================================================================
// AVX2 horizontal sum helper
// ===========================================================================

/// Reduces an __m256 (8 x f32) to a single f32 sum.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[allow(unsafe_op_in_unsafe_fn)]
#[inline]
unsafe fn hsum256(v: std::arch::x86_64::__m256) -> f32 {
    use std::arch::x86_64::*;
    let hi = _mm256_extractf128_ps(v, 1);
    let lo = _mm256_castps256_ps128(v);
    let sum4 = _mm_add_ps(lo, hi);
    let shuf = _mm_movehdup_ps(sum4);
    let sum2 = _mm_add_ps(sum4, shuf);
    let shuf2 = _mm_movehl_ps(sum2, sum2);
    let result = _mm_add_ss(sum2, shuf2);
    _mm_cvtss_f32(result)
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    const EPSILON: f32 = 1e-5;

    fn approxEq(a: f32, b: f32) -> bool {
        (a - b).abs() < EPSILON
    }

    // -----------------------------------------------------------------------
    // Cosine distance
    // -----------------------------------------------------------------------

    #[test]
    fn cosineIdenticalVectors() {
        let a = vec![1.0, 2.0, 3.0, 4.0];
        let d = computeDistance(DistanceMetric::Cosine, &a, &a);
        assert!(
            approxEq(d, 0.0),
            "identical vectors should have cosine distance 0, got {d}"
        );
    }

    #[test]
    fn cosineOrthogonalVectors() {
        let a = vec![1.0, 0.0, 0.0, 0.0];
        let b = vec![0.0, 1.0, 0.0, 0.0];
        let d = computeDistance(DistanceMetric::Cosine, &a, &b);
        assert!(
            approxEq(d, 1.0),
            "orthogonal vectors should have cosine distance 1, got {d}"
        );
    }

    #[test]
    fn cosineAntiParallelVectors() {
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![-1.0, 0.0, 0.0];
        let d = computeDistance(DistanceMetric::Cosine, &a, &b);
        assert!(
            approxEq(d, 2.0),
            "anti-parallel vectors should have cosine distance 2, got {d}"
        );
    }

    #[test]
    fn cosineZeroVectors() {
        let a = vec![0.0, 0.0, 0.0];
        let b = vec![1.0, 2.0, 3.0];
        let d = computeDistance(DistanceMetric::Cosine, &a, &b);
        assert!(
            approxEq(d, 0.0),
            "zero vector should produce distance 0, got {d}"
        );
    }

    // -----------------------------------------------------------------------
    // Euclidean distance
    // -----------------------------------------------------------------------

    #[test]
    fn euclideanIdenticalVectors() {
        let a = vec![3.0, 4.0, 5.0];
        let d = computeDistance(DistanceMetric::Euclidean, &a, &a);
        assert!(
            approxEq(d, 0.0),
            "identical vectors should have euclidean distance 0, got {d}"
        );
    }

    #[test]
    fn euclideanKnownDistance() {
        let a = vec![0.0, 0.0, 0.0];
        let b = vec![3.0, 4.0, 0.0];
        let d = computeDistance(DistanceMetric::Euclidean, &a, &b);
        assert!(approxEq(d, 5.0), "expected euclidean distance 5.0, got {d}");
    }

    #[test]
    fn euclideanUnitVectors() {
        let a = vec![1.0, 0.0];
        let b = vec![0.0, 1.0];
        let expected = std::f32::consts::SQRT_2;
        let d = computeDistance(DistanceMetric::Euclidean, &a, &b);
        assert!(approxEq(d, expected), "expected {expected}, got {d}");
    }

    // -----------------------------------------------------------------------
    // Dot product distance
    // -----------------------------------------------------------------------

    #[test]
    fn dotProductKnownValues() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![4.0, 5.0, 6.0];
        // dot = 1*4 + 2*5 + 3*6 = 32, returned as -32
        let d = computeDistance(DistanceMetric::DotProduct, &a, &b);
        assert!(approxEq(d, -32.0), "expected -32.0, got {d}");
    }

    #[test]
    fn dotProductNegativeReturn() {
        let a = vec![1.0, 1.0];
        let b = vec![1.0, 1.0];
        let d = computeDistance(DistanceMetric::DotProduct, &a, &b);
        assert!(
            d < 0.0,
            "positive dot product should return negative distance, got {d}"
        );
    }

    #[test]
    fn dotProductOrthogonal() {
        let a = vec![1.0, 0.0];
        let b = vec![0.0, 1.0];
        let d = computeDistance(DistanceMetric::DotProduct, &a, &b);
        assert!(
            approxEq(d, 0.0),
            "orthogonal dot product should be 0, got {d}"
        );
    }

    // -----------------------------------------------------------------------
    // Manhattan distance
    // -----------------------------------------------------------------------

    #[test]
    fn manhattanKnownDistance() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![4.0, 6.0, 3.0];
        // |1-4| + |2-6| + |3-3| = 3 + 4 + 0 = 7
        let d = computeDistance(DistanceMetric::Manhattan, &a, &b);
        assert!(approxEq(d, 7.0), "expected manhattan distance 7.0, got {d}");
    }

    #[test]
    fn manhattanIdentical() {
        let a = vec![5.0, -3.0, 7.0];
        let d = computeDistance(DistanceMetric::Manhattan, &a, &a);
        assert!(
            approxEq(d, 0.0),
            "identical vectors should have manhattan distance 0, got {d}"
        );
    }

    #[test]
    fn manhattanNegativeValues() {
        let a = vec![-1.0, -2.0];
        let b = vec![1.0, 2.0];
        // |(-1)-1| + |(-2)-2| = 2 + 4 = 6
        let d = computeDistance(DistanceMetric::Manhattan, &a, &b);
        assert!(approxEq(d, 6.0), "expected 6.0, got {d}");
    }

    // -----------------------------------------------------------------------
    // Batch distances
    // -----------------------------------------------------------------------

    #[test]
    fn batchMatchesIndividual() {
        let query = vec![1.0, 2.0, 3.0, 4.0];
        let v1 = vec![4.0, 3.0, 2.0, 1.0];
        let v2 = vec![1.0, 1.0, 1.0, 1.0];
        let v3 = vec![0.0, 0.0, 0.0, 0.0];
        let vectors: Vec<&[f32]> = vec![&v1, &v2, &v3];

        for metric in [
            DistanceMetric::Cosine,
            DistanceMetric::Euclidean,
            DistanceMetric::DotProduct,
            DistanceMetric::Manhattan,
        ] {
            let mut batchOut = Vec::new();
            batchDistances(&query, &vectors, metric, &mut batchOut);

            for (idx, v) in vectors.iter().enumerate() {
                let individual = computeDistance(metric, &query, v);
                assert!(
                    approxEq(batchOut[idx], individual),
                    "metric {:?} mismatch at index {idx}: batch={}, individual={}",
                    metric,
                    batchOut[idx],
                    individual
                );
            }
        }
    }

    // -----------------------------------------------------------------------
    // Utility functions
    // -----------------------------------------------------------------------

    #[test]
    fn vectorAddInplaceCorrectness() {
        let mut dst = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let src = vec![10.0, 20.0, 30.0, 40.0, 50.0];
        vectorAddInplace(&mut dst, &src);
        assert_eq!(dst, vec![11.0, 22.0, 33.0, 44.0, 55.0]);
    }

    #[test]
    fn vectorScaleInplaceCorrectness() {
        let mut dst = vec![2.0, 4.0, 6.0, 8.0];
        vectorScaleInplace(&mut dst, 0.5);
        assert_eq!(dst, vec![1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn vectorSubtractCorrectness() {
        let a = vec![10.0, 20.0, 30.0];
        let b = vec![1.0, 2.0, 3.0];
        let mut out = vec![0.0; 3];
        vectorSubtract(&a, &b, &mut out);
        assert_eq!(out, vec![9.0, 18.0, 27.0]);
    }

    #[test]
    fn vectorNormCorrectness() {
        let v = vec![3.0, 4.0];
        let n = vectorNorm(&v);
        assert!(approxEq(n, 5.0), "expected norm 5.0, got {n}");
    }

    #[test]
    fn vectorNormEmpty() {
        let n = vectorNorm(&[]);
        assert!(approxEq(n, 0.0), "empty vector norm should be 0, got {n}");
    }

    #[test]
    fn vectorNormUnit() {
        let v = vec![1.0, 0.0, 0.0];
        let n = vectorNorm(&v);
        assert!(approxEq(n, 1.0), "unit vector norm should be 1, got {n}");
    }

    // -----------------------------------------------------------------------
    // Non-aligned dimension tests (tail element processing)
    // -----------------------------------------------------------------------

    #[test]
    fn cosineNonAlignedDimensions() {
        for dim in [3, 7, 13, 127, 128, 1536] {
            let a: Vec<f32> = (0..dim).map(|i| (i as f32) * 0.01 + 0.1).collect();
            let b: Vec<f32> = (0..dim).map(|i| (i as f32) * 0.02 + 0.05).collect();
            let d = computeDistance(DistanceMetric::Cosine, &a, &b);
            assert!(
                d.is_finite(),
                "cosine distance should be finite for dim={dim}, got {d}"
            );
            assert!(
                d >= -EPSILON,
                "cosine distance should be >= 0 for dim={dim}, got {d}"
            );
        }
    }

    #[test]
    fn euclideanNonAlignedDimensions() {
        for dim in [3, 7, 13, 127, 128, 1536] {
            let a: Vec<f32> = (0..dim).map(|i| i as f32).collect();
            let b: Vec<f32> = (0..dim).map(|i| i as f32 + 1.0).collect();
            let d = computeDistance(DistanceMetric::Euclidean, &a, &b);
            // Each element differs by 1.0, so L2 = sqrt(dim)
            let expected = (dim as f32).sqrt();
            assert!(
                approxEq(d, expected),
                "dim={dim}: expected {expected}, got {d}"
            );
        }
    }

    #[test]
    fn dotProductNonAlignedDimensions() {
        for dim in [3, 7, 13, 127, 128, 1536] {
            let a: Vec<f32> = vec![1.0; dim];
            let b: Vec<f32> = vec![2.0; dim];
            let d = computeDistance(DistanceMetric::DotProduct, &a, &b);
            let expected = -(dim as f32 * 2.0);
            assert!(
                approxEq(d, expected),
                "dim={dim}: expected {expected}, got {d}"
            );
        }
    }

    #[test]
    fn manhattanNonAlignedDimensions() {
        for dim in [3, 7, 13, 127, 128, 1536] {
            let a: Vec<f32> = vec![0.0; dim];
            let b: Vec<f32> = vec![1.0; dim];
            let d = computeDistance(DistanceMetric::Manhattan, &a, &b);
            let expected = dim as f32;
            assert!(
                approxEq(d, expected),
                "dim={dim}: expected {expected}, got {d}"
            );
        }
    }

    #[test]
    fn utilityFunctionsNonAlignedDimensions() {
        for dim in [3, 7, 13, 127, 128, 1536] {
            // vectorAddInplace
            let mut dst: Vec<f32> = vec![1.0; dim];
            let src: Vec<f32> = vec![2.0; dim];
            vectorAddInplace(&mut dst, &src);
            assert!(
                dst.iter().all(|&x| approxEq(x, 3.0)),
                "add failed for dim={dim}"
            );

            // vectorScaleInplace
            let mut dst2: Vec<f32> = vec![4.0; dim];
            vectorScaleInplace(&mut dst2, 0.25);
            assert!(
                dst2.iter().all(|&x| approxEq(x, 1.0)),
                "scale failed for dim={dim}"
            );

            // vectorSubtract
            let a: Vec<f32> = vec![5.0; dim];
            let b: Vec<f32> = vec![3.0; dim];
            let mut out = vec![0.0; dim];
            vectorSubtract(&a, &b, &mut out);
            assert!(
                out.iter().all(|&x| approxEq(x, 2.0)),
                "subtract failed for dim={dim}"
            );

            // vectorNorm
            let v: Vec<f32> = vec![1.0; dim];
            let n = vectorNorm(&v);
            let expected = (dim as f32).sqrt();
            assert!(
                approxEq(n, expected),
                "norm failed for dim={dim}: expected {expected}, got {n}"
            );
        }
    }
}
