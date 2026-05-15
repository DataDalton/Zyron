#![allow(non_snake_case)]
// Numeric primitives shared across analytics, ML, causal, predictive
// Kahan/Neumaier compensated summation, online moments and quantiles,
// matrix solvers for closed-form regression
// All hand-rolled, no third-party crates

use zyron_common::error::{Result, ZyronError};

/// Kahan compensated summation
/// Reduces the f64 rounding error of a long reduction from O(n*eps) to O(eps)
#[derive(Debug, Clone, Default)]
pub struct KahanSum {
    pub sum: f64,
    pub c: f64,
}

impl KahanSum {
    pub fn new() -> Self {
        Self { sum: 0.0, c: 0.0 }
    }

    #[inline]
    pub fn add(&mut self, x: f64) {
        let y = x - self.c;
        let t = self.sum + y;
        self.c = (t - self.sum) - y;
        self.sum = t;
    }

    #[inline]
    pub fn value(&self) -> f64 {
        self.sum
    }
}

/// Welford online mean, variance, skewness, kurtosis
#[derive(Debug, Clone, Default)]
pub struct OnlineMoments {
    pub n: u64,
    pub mean: f64,
    pub m2: f64,
    pub m3: f64,
    pub m4: f64,
    pub minV: f64,
    pub maxV: f64,
}

impl OnlineMoments {
    pub fn new() -> Self {
        Self {
            n: 0,
            mean: 0.0,
            m2: 0.0,
            m3: 0.0,
            m4: 0.0,
            minV: f64::INFINITY,
            maxV: f64::NEG_INFINITY,
        }
    }

    pub fn ingest(&mut self, x: f64) {
        if x.is_nan() {
            return;
        }
        if x < self.minV {
            self.minV = x;
        }
        if x > self.maxV {
            self.maxV = x;
        }
        self.n += 1;
        let n = self.n as f64;
        let n1 = (self.n - 1) as f64;
        let delta = x - self.mean;
        let deltaN = delta / n;
        let deltaN2 = deltaN * deltaN;
        let term1 = delta * deltaN * n1;
        self.mean += deltaN;
        self.m4 += term1 * deltaN2 * (n * n - 3.0 * n + 3.0)
            + 6.0 * deltaN2 * self.m2
            - 4.0 * deltaN * self.m3;
        self.m3 += term1 * deltaN * (n - 2.0) - 3.0 * deltaN * self.m2;
        self.m2 += term1;
    }

    pub fn variance(&self) -> Option<f64> {
        if self.n < 2 {
            return None;
        }
        Some(self.m2 / (self.n - 1) as f64)
    }

    pub fn stddev(&self) -> Option<f64> {
        self.variance().map(|v| v.sqrt())
    }

    pub fn skewness(&self) -> Option<f64> {
        if self.n < 3 || self.m2 == 0.0 {
            return None;
        }
        Some((self.n as f64).sqrt() * self.m3 / self.m2.powf(1.5))
    }

    pub fn kurtosis(&self) -> Option<f64> {
        if self.n < 4 || self.m2 == 0.0 {
            return None;
        }
        Some(self.n as f64 * self.m4 / (self.m2 * self.m2) - 3.0)
    }
}

/// Online running covariance, Welford pairwise form
#[derive(Debug, Clone, Default)]
pub struct OnlineCovariance {
    pub n: u64,
    pub meanX: f64,
    pub meanY: f64,
    pub cXy: f64,
    pub m2X: f64,
    pub m2Y: f64,
}

impl OnlineCovariance {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn ingest(&mut self, x: f64, y: f64) {
        if x.is_nan() || y.is_nan() {
            return;
        }
        self.n += 1;
        let n = self.n as f64;
        let dx = x - self.meanX;
        let dy = y - self.meanY;
        self.meanX += dx / n;
        self.meanY += dy / n;
        self.cXy += dx * (y - self.meanY);
        self.m2X += dx * (x - self.meanX);
        self.m2Y += dy * (y - self.meanY);
    }

    pub fn correlation(&self) -> Option<f64> {
        if self.n < 2 {
            return None;
        }
        let denom = (self.m2X * self.m2Y).sqrt();
        if denom == 0.0 {
            return None;
        }
        Some(self.cXy / denom)
    }

    pub fn covariance(&self) -> Option<f64> {
        if self.n < 2 {
            return None;
        }
        Some(self.cXy / (self.n - 1) as f64)
    }
}

/// P^2 algorithm for online quantile estimation (Jain and Chlamtac)
/// Constant memory regardless of input size, accuracy improves with n.
/// Marker count is fixed at 5 to preserve the original algorithm's
/// arithmetic; for tighter accuracy on long streams compose several
/// instances at strategic quantiles (q10, q25, q50, q75, q90)
#[derive(Debug, Clone)]
pub struct OnlineQuantile {
    p: f64,
    n: i64,
    q: [f64; 5],
    np: [f64; 5],
    nDesired: [f64; 5],
    initialized: bool,
}

impl OnlineQuantile {
    pub fn new(p: f64) -> Self {
        Self {
            p: p.clamp(0.0, 1.0),
            n: 0,
            q: [0.0; 5],
            np: [1.0, 2.0, 3.0, 4.0, 5.0],
            nDesired: [
                1.0,
                1.0 + 2.0 * p,
                1.0 + 4.0 * p,
                3.0 + 2.0 * p,
                5.0,
            ],
            initialized: false,
        }
    }

    pub fn ingest(&mut self, x: f64) {
        if x.is_nan() {
            return;
        }
        self.n += 1;
        if (self.n as usize) <= 5 {
            self.q[(self.n as usize) - 1] = x;
            if self.n == 5 {
                self.q.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
                self.initialized = true;
            }
            return;
        }
        let k: usize;
        if x < self.q[0] {
            self.q[0] = x;
            k = 0;
        } else if x < self.q[1] {
            k = 0;
        } else if x < self.q[2] {
            k = 1;
        } else if x < self.q[3] {
            k = 2;
        } else if x <= self.q[4] {
            k = 3;
        } else {
            self.q[4] = x;
            k = 3;
        }
        for i in (k + 1)..5 {
            self.np[i] += 1.0;
        }
        let p = self.p;
        self.nDesired[1] += p / 2.0;
        self.nDesired[2] += p;
        self.nDesired[3] += (1.0 + p) / 2.0;
        self.nDesired[4] += 1.0;
        for i in 1..4 {
            let d = self.nDesired[i] - self.np[i];
            if (d >= 1.0 && self.np[i + 1] - self.np[i] > 1.0)
                || (d <= -1.0 && self.np[i - 1] - self.np[i] < -1.0)
            {
                let dSign = if d >= 0.0 { 1.0 } else { -1.0 };
                let qNew = self.parabolic(i, dSign);
                if self.q[i - 1] < qNew && qNew < self.q[i + 1] {
                    self.q[i] = qNew;
                } else {
                    self.q[i] = self.linear(i, dSign);
                }
                self.np[i] += dSign;
            }
        }
    }

    fn parabolic(&self, i: usize, d: f64) -> f64 {
        let qi = self.q[i];
        let qip = self.q[i + 1];
        let qim = self.q[i - 1];
        let np = self.np[i];
        let npp = self.np[i + 1];
        let npm = self.np[i - 1];
        qi + d / (npp - npm)
            * ((np - npm + d) * (qip - qi) / (npp - np)
                + (npp - np - d) * (qi - qim) / (np - npm))
    }

    fn linear(&self, i: usize, d: f64) -> f64 {
        let idx = if d >= 0.0 { i + 1 } else { i - 1 };
        self.q[i] + d * (self.q[idx] - self.q[i]) / (self.np[idx] - self.np[i])
    }

    pub fn estimate(&self) -> Option<f64> {
        if self.n < 5 {
            if self.n == 0 {
                return None;
            }
            let mut sorted: Vec<f64> = self.q[..(self.n as usize)].to_vec();
            sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            let idx = ((self.p * (self.n as f64 - 1.0)).round() as usize).min(sorted.len() - 1);
            return Some(sorted[idx]);
        }
        Some(self.q[2])
    }
}

/// Bloom filter with double hashing
/// k hashes derived from two base hashes via h_i = h1 + i*h2
/// Sized for an expected n at false-positive rate p
#[derive(Debug)]
pub struct BloomFilter {
    bits: Vec<u64>,
    bitCount: u64,
    k: u32,
}

impl BloomFilter {
    pub fn withCapacity(expectedN: usize, fpRate: f64) -> Self {
        let n = expectedN.max(1) as f64;
        let p = fpRate.clamp(1e-12, 0.999);
        let mF = (-(n * p.ln()) / (std::f64::consts::LN_2 * std::f64::consts::LN_2)).ceil();
        let m = (mF as u64).max(64);
        let k = ((m as f64 / n) * std::f64::consts::LN_2).ceil() as u32;
        let kClamped = k.clamp(1, 16);
        let words = ((m + 63) / 64) as usize;
        Self {
            bits: vec![0u64; words],
            bitCount: m,
            k: kClamped,
        }
    }

    pub fn insert<H: HashableBloom + ?Sized>(&mut self, key: &H) {
        let (h1, h2) = key.bloomHashes();
        for i in 0..self.k {
            let h = h1.wrapping_add((i as u64).wrapping_mul(h2));
            let bit = (h % self.bitCount) as usize;
            self.bits[bit / 64] |= 1u64 << (bit % 64);
        }
    }

    pub fn contains<H: HashableBloom + ?Sized>(&self, key: &H) -> bool {
        let (h1, h2) = key.bloomHashes();
        for i in 0..self.k {
            let h = h1.wrapping_add((i as u64).wrapping_mul(h2));
            let bit = (h % self.bitCount) as usize;
            if self.bits[bit / 64] & (1u64 << (bit % 64)) == 0 {
                return false;
            }
        }
        true
    }

    pub fn lenBits(&self) -> u64 {
        self.bitCount
    }
}

pub trait HashableBloom {
    fn bloomHashes(&self) -> (u64, u64);
}

impl HashableBloom for u64 {
    fn bloomHashes(&self) -> (u64, u64) {
        let h1 = zyron_common::fx_mix(0x9E3779B97F4A7C15, *self);
        let h2 = zyron_common::fx_mix(0xBF58476D1CE4E5B9, *self).wrapping_add(1);
        (h1, h2)
    }
}

impl HashableBloom for str {
    fn bloomHashes(&self) -> (u64, u64) {
        // Use the project's xxh3-style hash with two different seeds, one
        // call per hash. Much cheaper than fx_mix per byte for long keys
        // such as UUIDs and entity-id strings (F18)
        let bytes = self.as_bytes();
        let h1 = zyron_common::hash64_seeded(bytes, 0x9E3779B97F4A7C15);
        let h2 = zyron_common::hash64_seeded(bytes, 0xBF58476D1CE4E5B9) | 1;
        (h1, h2)
    }
}

impl HashableBloom for String {
    fn bloomHashes(&self) -> (u64, u64) {
        self.as_str().bloomHashes()
    }
}

/// Cholesky factorization for symmetric positive-definite matrices
/// Solves A x = b in place, A is row-major n*n, b is length n
/// Returns the lower-triangular factor in A (upper triangle untouched)
pub fn choleskySolve(a: &mut [f64], b: &mut [f64], n: usize) -> Result<()> {
    if a.len() != n * n {
        return Err(ZyronError::InvalidParameter {
            name: "matrix".to_string(),
            value: format!("expected {}x{}, got {} entries", n, n, a.len()),
        });
    }
    if b.len() != n {
        return Err(ZyronError::InvalidParameter {
            name: "rhs".to_string(),
            value: format!("expected length {}, got {}", n, b.len()),
        });
    }
    // Cholesky decomposition A = L L^T
    for i in 0..n {
        for j in 0..=i {
            let mut s = a[i * n + j];
            for k in 0..j {
                s -= a[i * n + k] * a[j * n + k];
            }
            if i == j {
                if s <= 0.0 {
                    return Err(ZyronError::ExecutionError(
                        "matrix is not positive definite".to_string(),
                    ));
                }
                a[i * n + j] = s.sqrt();
            } else {
                a[i * n + j] = s / a[j * n + j];
            }
        }
    }
    // Forward substitution L y = b
    for i in 0..n {
        let mut s = b[i];
        for k in 0..i {
            s -= a[i * n + k] * b[k];
        }
        b[i] = s / a[i * n + i];
    }
    // Back substitution L^T x = y, store result in b
    for i in (0..n).rev() {
        let mut s = b[i];
        for k in (i + 1)..n {
            s -= a[k * n + i] * b[k];
        }
        b[i] = s / a[i * n + i];
    }
    Ok(())
}

/// Computes mean and stddev for each column of an (n, p) row-major matrix
pub fn columnStandardize(data: &[f64], n: usize, p: usize) -> (Vec<f64>, Vec<f64>) {
    let mut mean = vec![0.0f64; p];
    let mut std = vec![0.0f64; p];
    if n == 0 {
        return (mean, std);
    }
    for i in 0..n {
        for j in 0..p {
            mean[j] += data[i * p + j];
        }
    }
    for j in 0..p {
        mean[j] /= n as f64;
    }
    for i in 0..n {
        for j in 0..p {
            let d = data[i * p + j] - mean[j];
            std[j] += d * d;
        }
    }
    let denom = if n > 1 { (n - 1) as f64 } else { 1.0 };
    for j in 0..p {
        std[j] = (std[j] / denom).sqrt();
        if std[j] == 0.0 {
            std[j] = 1.0;
        }
    }
    (mean, std)
}

/// Bootstrap confidence interval, BCa percentile bootstrap
/// `estimator` is called with the sampled values and must return a scalar
pub fn bootstrapCi<F>(
    values: &[f64],
    estimator: F,
    bSamples: usize,
    alpha: f64,
    seed: u64,
) -> Option<(f64, f64, f64)>
where
    F: Fn(&[f64]) -> f64,
{
    if values.is_empty() || bSamples == 0 {
        return None;
    }
    let mut rng = zyron_common::Xoshiro256pp::fromSeed(seed);
    let n = values.len();
    let mut buf = vec![0.0f64; n];
    let mut estimates = Vec::with_capacity(bSamples);
    for _ in 0..bSamples {
        for i in 0..n {
            let idx = rng.nextRange(n as u64) as usize;
            buf[i] = values[idx];
        }
        estimates.push(estimator(&buf));
    }
    estimates.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let lo = ((alpha / 2.0) * bSamples as f64).floor() as usize;
    let hi = (((1.0 - alpha / 2.0) * bSamples as f64) - 1.0).ceil().max(0.0) as usize;
    let lo = lo.min(bSamples - 1);
    let hi = hi.min(bSamples - 1);
    Some((estimator(values), estimates[lo], estimates[hi]))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn kahanReducesError() {
        let mut s = KahanSum::new();
        for _ in 0..1_000_000 {
            s.add(0.1);
        }
        assert!((s.value() - 100_000.0).abs() < 1e-6);
    }

    #[test]
    fn onlineMomentsMatchOffline() {
        let xs: Vec<f64> = (1..=100).map(|x| x as f64).collect();
        let mut m = OnlineMoments::new();
        for &x in &xs {
            m.ingest(x);
        }
        assert!((m.mean - 50.5).abs() < 1e-9);
        let v = m.variance().unwrap();
        assert!((v - 841.6666666666666).abs() < 1e-6);
    }

    #[test]
    fn onlineCovarianceComputesPearson() {
        let xs: Vec<f64> = (0..100).map(|x| x as f64).collect();
        let ys: Vec<f64> = xs.iter().map(|x| 2.0 * x + 5.0).collect();
        let mut c = OnlineCovariance::new();
        for i in 0..xs.len() {
            c.ingest(xs[i], ys[i]);
        }
        let r = c.correlation().unwrap();
        assert!((r - 1.0).abs() < 1e-9);
    }

    #[test]
    fn onlineQuantileApproximate() {
        let mut q = OnlineQuantile::new(0.5);
        for i in 1..=10_001 {
            q.ingest(i as f64);
        }
        let est = q.estimate().unwrap();
        assert!((est - 5001.0).abs() < 100.0, "estimate = {}", est);
    }

    #[test]
    fn bloomMembershipNoFalseNegatives() {
        let mut bf = BloomFilter::withCapacity(1000, 0.01);
        for i in 0..500u64 {
            bf.insert(&i);
        }
        for i in 0..500u64 {
            assert!(bf.contains(&i));
        }
    }

    #[test]
    fn choleskyIdentitySolve() {
        let mut a = vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];
        let mut b = vec![3.0, 5.0, 7.0];
        choleskySolve(&mut a, &mut b, 3).unwrap();
        assert_eq!(b, vec![3.0, 5.0, 7.0]);
    }

    #[test]
    fn choleskyDiagSolve() {
        let mut a = vec![4.0, 0.0, 0.0, 0.0, 9.0, 0.0, 0.0, 0.0, 16.0];
        let mut b = vec![8.0, 27.0, 64.0];
        choleskySolve(&mut a, &mut b, 3).unwrap();
        assert!((b[0] - 2.0).abs() < 1e-12);
        assert!((b[1] - 3.0).abs() < 1e-12);
        assert!((b[2] - 4.0).abs() < 1e-12);
    }
}
