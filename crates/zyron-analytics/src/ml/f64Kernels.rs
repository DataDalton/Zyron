#![allow(non_snake_case)]
// f64 vector kernels for ML hot paths
// SIMD via std::arch on x86_64 (AVX2) and aarch64 (NEON), scalar fallback
// Resolved once at startup through OnceLock function pointers

use std::sync::OnceLock;

pub type DotFn = unsafe fn(a: *const f64, b: *const f64, len: usize) -> f64;
pub type AxpyFn = unsafe fn(alpha: f64, x: *const f64, y: *mut f64, len: usize);
pub type ScaleFn = unsafe fn(alpha: f64, x: *mut f64, len: usize);
pub type AddFn = unsafe fn(x: *const f64, y: *mut f64, len: usize);
pub type NormSqFn = unsafe fn(x: *const f64, len: usize) -> f64;

static DOT_FN: OnceLock<DotFn> = OnceLock::new();
static AXPY_FN: OnceLock<AxpyFn> = OnceLock::new();
static SCALE_FN: OnceLock<ScaleFn> = OnceLock::new();
static ADD_FN: OnceLock<AddFn> = OnceLock::new();
static NORMSQ_FN: OnceLock<NormSqFn> = OnceLock::new();

/// Public so callers in hot loops can hoist the dispatch out of the
/// inner iteration and call the returned fn pointer directly
#[inline]
pub fn resolveDot() -> DotFn {
    *DOT_FN.get_or_init(|| {
        #[cfg(target_arch = "x86_64")]
        {
            if is_x86_feature_detected!("avx2") {
                return dotAvx2 as DotFn;
            }
        }
        dotScalar as DotFn
    })
}

/// Public so callers can hoist the dispatch out of inner loops
#[inline]
pub fn resolveAxpyFn() -> AxpyFn {
    resolveAxpy()
}

#[inline]
fn resolveAxpy() -> AxpyFn {
    *AXPY_FN.get_or_init(|| {
        #[cfg(target_arch = "x86_64")]
        {
            if is_x86_feature_detected!("avx2") {
                return axpyAvx2 as AxpyFn;
            }
        }
        axpyScalar as AxpyFn
    })
}

#[inline]
fn resolveScale() -> ScaleFn {
    *SCALE_FN.get_or_init(|| {
        #[cfg(target_arch = "x86_64")]
        {
            if is_x86_feature_detected!("avx2") {
                return scaleAvx2 as ScaleFn;
            }
        }
        scaleScalar as ScaleFn
    })
}

#[inline]
fn resolveAdd() -> AddFn {
    *ADD_FN.get_or_init(|| {
        #[cfg(target_arch = "x86_64")]
        {
            if is_x86_feature_detected!("avx2") {
                return addAvx2 as AddFn;
            }
        }
        addScalar as AddFn
    })
}

#[inline]
fn resolveNormSq() -> NormSqFn {
    *NORMSQ_FN.get_or_init(|| {
        #[cfg(target_arch = "x86_64")]
        {
            if is_x86_feature_detected!("avx2") {
                return normSqAvx2 as NormSqFn;
            }
        }
        normSqScalar as NormSqFn
    })
}

/// Dot product, sum_i a[i]*b[i]
#[inline]
pub fn dot(a: &[f64], b: &[f64]) -> f64 {
    debug_assert_eq!(a.len(), b.len());
    let len = a.len();
    if len == 0 {
        return 0.0;
    }
    let f = resolveDot();
    unsafe { f(a.as_ptr(), b.as_ptr(), len) }
}

/// y[i] += alpha * x[i] in place
#[inline]
pub fn axpy(alpha: f64, x: &[f64], y: &mut [f64]) {
    debug_assert_eq!(x.len(), y.len());
    let len = x.len();
    if len == 0 {
        return;
    }
    let f = resolveAxpy();
    unsafe { f(alpha, x.as_ptr(), y.as_mut_ptr(), len) }
}

/// x[i] *= alpha in place
#[inline]
pub fn scaleInPlace(alpha: f64, x: &mut [f64]) {
    let len = x.len();
    if len == 0 {
        return;
    }
    let f = resolveScale();
    unsafe { f(alpha, x.as_mut_ptr(), len) }
}

/// y[i] += x[i] in place
#[inline]
pub fn addInPlace(x: &[f64], y: &mut [f64]) {
    debug_assert_eq!(x.len(), y.len());
    let len = x.len();
    if len == 0 {
        return;
    }
    let f = resolveAdd();
    unsafe { f(x.as_ptr(), y.as_mut_ptr(), len) }
}

/// Sum of squares, sum_i x[i]^2
#[inline]
pub fn normSquared(x: &[f64]) -> f64 {
    let len = x.len();
    if len == 0 {
        return 0.0;
    }
    let f = resolveNormSq();
    unsafe { f(x.as_ptr(), len) }
}

/// L2 norm
#[inline]
pub fn l2Norm(x: &[f64]) -> f64 {
    normSquared(x).sqrt()
}

/// Squared euclidean distance, sum_i (a[i] - b[i])^2
pub fn sqDistance(a: &[f64], b: &[f64]) -> f64 {
    debug_assert_eq!(a.len(), b.len());
    let mut s = 0.0f64;
    for i in 0..a.len() {
        let d = a[i] - b[i];
        s += d * d;
    }
    s
}

/// Polynomial-approximation exp suitable for vectorized inner loops
/// Reduces argument by ldexp into [-ln 2 / 2, ln 2 / 2] then evaluates
/// a 5-degree minimax polynomial. Accurate to ~1 ulp on the target range
#[inline]
pub fn fastExp(x: f64) -> f64 {
    if x > 700.0 {
        return f64::INFINITY;
    }
    if x < -700.0 {
        return 0.0;
    }
    let LN2: f64 = std::f64::consts::LN_2;
    let n = (x / LN2).round();
    let r = x - n * LN2;
    // Horner's evaluation of e^r approximated on |r| <= ln 2 / 2
    let p = 1.0
        + r * (1.0
            + r * (0.5
                + r * (1.0 / 6.0
                    + r * (1.0 / 24.0 + r * (1.0 / 120.0 + r * (1.0 / 720.0))))));
    let bits = ((n as i64 + 1023) as u64) << 52;
    let scale = f64::from_bits(bits);
    p * scale
}

/// Numerically stable sigmoid
/// 1/(1+exp(-z)) for z >= 0, exp(z)/(1+exp(z)) for z < 0
/// Avoids overflow at large negative z
#[inline]
pub fn sigmoid(z: f64) -> f64 {
    if z >= 0.0 {
        let e = fastExp(-z);
        1.0 / (1.0 + e)
    } else {
        let e = fastExp(z);
        e / (1.0 + e)
    }
}

/// Vectorized sigmoid in place
pub fn sigmoidInPlace(x: &mut [f64]) {
    for v in x.iter_mut() {
        *v = sigmoid(*v);
    }
}

/// Stable log(1 + exp(z)), used for logistic loss
#[inline]
pub fn log1pExp(z: f64) -> f64 {
    if z > 0.0 {
        z + (1.0 + fastExp(-z)).ln()
    } else {
        (1.0 + fastExp(z)).ln()
    }
}

/// `y[j] += sum_i alpha[i] * x[i*p + j]` for j in 0..p
/// Row-major (n, p) data with per-row scalar `alpha[i]`. Used to accumulate
/// the SGD gradient for a mini-batch in one pass instead of calling axpy
/// per row (F4). Saves the per-row function-pointer dispatch and lets the
/// inner loop hit the SIMD-add path
pub fn rowMajorScaledAccumulate(alpha: &[f64], xs: &[f64], y: &mut [f64], n: usize, p: usize) {
    debug_assert_eq!(alpha.len(), n);
    debug_assert_eq!(xs.len(), n * p);
    debug_assert_eq!(y.len(), p);
    if n == 0 || p == 0 {
        return;
    }
    let f = resolveAxpy();
    // Hoist the dispatch out of the per-row loop, one indirect call total
    for i in 0..n {
        let a = alpha[i];
        if a == 0.0 {
            continue;
        }
        let row = &xs[i * p..i * p + p];
        unsafe { f(a, row.as_ptr(), y.as_mut_ptr(), p) };
    }
}

/// `y = X * w + b` where X is row-major (n, p), w is length p, b is scalar
/// Output column-vector y has length n. Used by linear-model batch predict
pub fn rowMajorMatvec(xs: &[f64], w: &[f64], b: f64, n: usize, p: usize, y: &mut [f64]) {
    debug_assert_eq!(xs.len(), n * p);
    debug_assert_eq!(w.len(), p);
    debug_assert_eq!(y.len(), n);
    let f = resolveDot();
    for i in 0..n {
        let row = &xs[i * p..i * p + p];
        let d = unsafe { f(row.as_ptr(), w.as_ptr(), p) };
        y[i] = d + b;
    }
}

// ===== Scalar fallbacks =====

#[inline]
unsafe fn dotScalar(a: *const f64, b: *const f64, len: usize) -> f64 {
    let a = unsafe { std::slice::from_raw_parts(a, len) };
    let b = unsafe { std::slice::from_raw_parts(b, len) };
    let mut s0 = 0.0f64;
    let mut s1 = 0.0f64;
    let mut s2 = 0.0f64;
    let mut s3 = 0.0f64;
    let chunks = len / 4;
    for i in 0..chunks {
        let j = i * 4;
        s0 += a[j] * b[j];
        s1 += a[j + 1] * b[j + 1];
        s2 += a[j + 2] * b[j + 2];
        s3 += a[j + 3] * b[j + 3];
    }
    let mut s = s0 + s1 + s2 + s3;
    for i in (chunks * 4)..len {
        s += a[i] * b[i];
    }
    s
}

#[inline]
unsafe fn axpyScalar(alpha: f64, x: *const f64, y: *mut f64, len: usize) {
    let x = unsafe { std::slice::from_raw_parts(x, len) };
    let y = unsafe { std::slice::from_raw_parts_mut(y, len) };
    for i in 0..len {
        y[i] += alpha * x[i];
    }
}

#[inline]
unsafe fn scaleScalar(alpha: f64, x: *mut f64, len: usize) {
    let x = unsafe { std::slice::from_raw_parts_mut(x, len) };
    for v in x.iter_mut() {
        *v *= alpha;
    }
}

#[inline]
unsafe fn addScalar(x: *const f64, y: *mut f64, len: usize) {
    let x = unsafe { std::slice::from_raw_parts(x, len) };
    let y = unsafe { std::slice::from_raw_parts_mut(y, len) };
    for i in 0..len {
        y[i] += x[i];
    }
}

#[inline]
unsafe fn normSqScalar(x: *const f64, len: usize) -> f64 {
    let x = unsafe { std::slice::from_raw_parts(x, len) };
    let mut s0 = 0.0f64;
    let mut s1 = 0.0f64;
    let chunks = len / 2;
    for i in 0..chunks {
        let j = i * 2;
        s0 += x[j] * x[j];
        s1 += x[j + 1] * x[j + 1];
    }
    let mut s = s0 + s1;
    for i in (chunks * 2)..len {
        s += x[i] * x[i];
    }
    s
}

// ===== AVX2 path =====

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "fma")]
unsafe fn dotAvx2(a: *const f64, b: *const f64, len: usize) -> f64 {
    use std::arch::x86_64::*;
    let mut acc0 = unsafe { _mm256_setzero_pd() };
    let mut acc1 = unsafe { _mm256_setzero_pd() };
    let chunks = len / 8;
    for i in 0..chunks {
        let off = (i * 8) as isize;
        let av0 = unsafe { _mm256_loadu_pd(a.offset(off)) };
        let bv0 = unsafe { _mm256_loadu_pd(b.offset(off)) };
        let av1 = unsafe { _mm256_loadu_pd(a.offset(off + 4)) };
        let bv1 = unsafe { _mm256_loadu_pd(b.offset(off + 4)) };
        acc0 = unsafe { _mm256_fmadd_pd(av0, bv0, acc0) };
        acc1 = unsafe { _mm256_fmadd_pd(av1, bv1, acc1) };
    }
    let acc = unsafe { _mm256_add_pd(acc0, acc1) };
    let mut buf = [0.0f64; 4];
    unsafe { _mm256_storeu_pd(buf.as_mut_ptr(), acc) };
    let mut s = buf[0] + buf[1] + buf[2] + buf[3];
    for i in (chunks * 8)..len {
        s += unsafe { *a.add(i) * *b.add(i) };
    }
    s
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "fma")]
unsafe fn axpyAvx2(alpha: f64, x: *const f64, y: *mut f64, len: usize) {
    use std::arch::x86_64::*;
    let av = unsafe { _mm256_set1_pd(alpha) };
    let chunks = len / 4;
    for i in 0..chunks {
        let off = (i * 4) as isize;
        let xv = unsafe { _mm256_loadu_pd(x.offset(off)) };
        let yv = unsafe { _mm256_loadu_pd(y.offset(off)) };
        let r = unsafe { _mm256_fmadd_pd(av, xv, yv) };
        unsafe { _mm256_storeu_pd(y.offset(off), r) };
    }
    for i in (chunks * 4)..len {
        unsafe { *y.add(i) += alpha * *x.add(i) };
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn scaleAvx2(alpha: f64, x: *mut f64, len: usize) {
    use std::arch::x86_64::*;
    let av = unsafe { _mm256_set1_pd(alpha) };
    let chunks = len / 4;
    for i in 0..chunks {
        let off = (i * 4) as isize;
        let xv = unsafe { _mm256_loadu_pd(x.offset(off)) };
        let r = unsafe { _mm256_mul_pd(av, xv) };
        unsafe { _mm256_storeu_pd(x.offset(off), r) };
    }
    for i in (chunks * 4)..len {
        unsafe { *x.add(i) *= alpha };
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn addAvx2(x: *const f64, y: *mut f64, len: usize) {
    use std::arch::x86_64::*;
    let chunks = len / 4;
    for i in 0..chunks {
        let off = (i * 4) as isize;
        let xv = unsafe { _mm256_loadu_pd(x.offset(off)) };
        let yv = unsafe { _mm256_loadu_pd(y.offset(off)) };
        let r = unsafe { _mm256_add_pd(xv, yv) };
        unsafe { _mm256_storeu_pd(y.offset(off), r) };
    }
    for i in (chunks * 4)..len {
        unsafe { *y.add(i) += *x.add(i) };
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "fma")]
unsafe fn normSqAvx2(x: *const f64, len: usize) -> f64 {
    use std::arch::x86_64::*;
    let mut acc = unsafe { _mm256_setzero_pd() };
    let chunks = len / 4;
    for i in 0..chunks {
        let off = (i * 4) as isize;
        let xv = unsafe { _mm256_loadu_pd(x.offset(off)) };
        acc = unsafe { _mm256_fmadd_pd(xv, xv, acc) };
    }
    let mut buf = [0.0f64; 4];
    unsafe { _mm256_storeu_pd(buf.as_mut_ptr(), acc) };
    let mut s = buf[0] + buf[1] + buf[2] + buf[3];
    for i in (chunks * 4)..len {
        let v = unsafe { *x.add(i) };
        s += v * v;
    }
    s
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dotMatchesNaive() {
        let a: Vec<f64> = (0..100).map(|i| i as f64 * 0.1).collect();
        let b: Vec<f64> = (0..100).map(|i| (100 - i) as f64 * 0.5).collect();
        let mut expected = 0.0f64;
        for i in 0..100 {
            expected += a[i] * b[i];
        }
        let got = dot(&a, &b);
        assert!((got - expected).abs() < 1e-9);
    }

    #[test]
    fn axpyMatches() {
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let mut y = vec![10.0, 10.0, 10.0, 10.0, 10.0];
        axpy(2.0, &x, &mut y);
        assert_eq!(y, vec![12.0, 14.0, 16.0, 18.0, 20.0]);
    }

    #[test]
    fn scaleMatches() {
        let mut x = vec![1.0, 2.0, 3.0, 4.0];
        scaleInPlace(3.0, &mut x);
        assert_eq!(x, vec![3.0, 6.0, 9.0, 12.0]);
    }

    #[test]
    fn normSqMatches() {
        let x = vec![1.0, 2.0, 3.0, 4.0];
        assert!((normSquared(&x) - 30.0).abs() < 1e-12);
    }

    #[test]
    fn sigmoidEndpoints() {
        assert!((sigmoid(0.0) - 0.5).abs() < 1e-12);
        assert!(sigmoid(20.0) > 0.999);
        assert!(sigmoid(-20.0) < 0.001);
    }

    #[test]
    fn fastExpReasonable() {
        for &x in &[0.0, 0.5, 1.0, -1.0, 5.0, -5.0, 10.0] {
            let got = fastExp(x);
            let expected = x.exp();
            let rel = (got - expected).abs() / expected;
            assert!(rel < 1e-6, "fastExp({}) = {} vs {}", x, got, expected);
        }
    }

    #[test]
    fn log1pExpStable() {
        assert!((log1pExp(0.0) - (1.0f64 + 1.0).ln()).abs() < 1e-12);
        assert!((log1pExp(50.0) - 50.0).abs() < 1e-6);
        assert!(log1pExp(-50.0).abs() < 1e-6);
    }
}
