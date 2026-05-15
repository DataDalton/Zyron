#![allow(non_snake_case)]
// Cooley-Tukey radix-2 FFT, custom implementation, no third-party deps
//
// Provides forward and inverse FFT for power-of-two complex inputs plus
// a real-input wrapper that pads to the next power of two. Used by
// Wiener-Khinchin autocorrelation in predictive.rs when the requested
// (n, maxLag) cost is high enough that O(n log n) beats the direct
// O(n * maxLag) loop

/// Bit-reverse permutation in place. n must be a power of two
fn bitReversePermute(x: &mut [Complex]) {
    let n = x.len();
    let mut j = 0usize;
    for i in 1..n {
        let mut bit = n >> 1;
        while j & bit != 0 {
            j ^= bit;
            bit >>= 1;
        }
        j ^= bit;
        if i < j {
            x.swap(i, j);
        }
    }
}

/// Complex number with f64 real and imaginary parts
#[derive(Debug, Clone, Copy, Default)]
pub struct Complex {
    pub re: f64,
    pub im: f64,
}

impl Complex {
    pub fn new(re: f64, im: f64) -> Self {
        Self { re, im }
    }

    pub fn norm_sq(&self) -> f64 {
        self.re * self.re + self.im * self.im
    }
}

impl std::ops::Add for Complex {
    type Output = Self;
    fn add(self, rhs: Self) -> Self {
        Self {
            re: self.re + rhs.re,
            im: self.im + rhs.im,
        }
    }
}

impl std::ops::Sub for Complex {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self {
        Self {
            re: self.re - rhs.re,
            im: self.im - rhs.im,
        }
    }
}

impl std::ops::Mul for Complex {
    type Output = Self;
    fn mul(self, rhs: Self) -> Self {
        Self {
            re: self.re * rhs.re - self.im * rhs.im,
            im: self.re * rhs.im + self.im * rhs.re,
        }
    }
}

/// Forward radix-2 Cooley-Tukey FFT, in place. Input length must be a
/// power of two. Uses iterative decimation-in-time with precomputed
/// twiddle factors. Sign convention: F[k] = sum_n x[n] * exp(-2 pi i k n / N)
pub fn fft(x: &mut [Complex]) {
    let n = x.len();
    if n <= 1 {
        return;
    }
    debug_assert!(n.is_power_of_two(), "fft input length must be a power of 2");

    bitReversePermute(x);

    let mut size = 2usize;
    while size <= n {
        let half = size >> 1;
        let theta = -2.0 * std::f64::consts::PI / size as f64;
        let wStep = Complex::new(theta.cos(), theta.sin());
        let mut i = 0usize;
        while i < n {
            let mut w = Complex::new(1.0, 0.0);
            for j in 0..half {
                let evenIdx = i + j;
                let oddIdx = i + j + half;
                let t = w * x[oddIdx];
                x[oddIdx] = x[evenIdx] - t;
                x[evenIdx] = x[evenIdx] + t;
                w = w * wStep;
            }
            i += size;
        }
        size <<= 1;
    }
}

/// Inverse FFT, in place. Reuses the forward path with conjugation
/// trick and a final 1/N scaling so f -> ifft(fft(f)) == f
pub fn ifft(x: &mut [Complex]) {
    let n = x.len();
    if n <= 1 {
        return;
    }
    for v in x.iter_mut() {
        v.im = -v.im;
    }
    fft(x);
    let scale = 1.0 / n as f64;
    for v in x.iter_mut() {
        v.re *= scale;
        v.im = -v.im * scale;
    }
}

/// Real-input forward FFT into a freshly allocated complex buffer.
/// Zero-pads the input up to the next power of two so any length is
/// accepted. Returns a buffer of length nextPow2(input.len())
pub fn fftReal(input: &[f64]) -> Vec<Complex> {
    let n = input.len();
    let m = n.next_power_of_two().max(2);
    let mut buf: Vec<Complex> = Vec::with_capacity(m);
    for &x in input {
        buf.push(Complex::new(x, 0.0));
    }
    for _ in n..m {
        buf.push(Complex::new(0.0, 0.0));
    }
    fft(&mut buf);
    buf
}

/// Autocorrelation via the Wiener-Khinchin theorem:
/// `acf[lag] = ifft(|fft(x_centered)|^2)[lag] / variance_n`
///
/// For series of length n, returns biased ACF values for lags
/// 0..=maxLag. Zero-padding to 2*n avoids circular wraparound. Total
/// cost is O(n log n) rather than the naive O(n * maxLag) which is the
/// win for large n and large maxLag
pub fn autocorrelationFft(values: &[f64], maxLag: usize) -> Vec<f64> {
    let n = values.len();
    if n == 0 {
        return Vec::new();
    }
    let lim = maxLag.min(n - 1);
    let mean: f64 = values.iter().sum::<f64>() / n as f64;
    let mut centered: Vec<f64> = values.iter().map(|v| v - mean).collect();
    let var0: f64 = centered.iter().map(|v| v * v).sum();
    if var0 == 0.0 {
        // Zero-variance input is degenerate. Return ACF=1 at every
        // requested lag with the same length as the non-degenerate path
        // so callers get a consistent output shape regardless of input
        return vec![1.0; lim + 1];
    }
    // Pad to 2n then to next power of two so circular convolution
    // produces correct linear autocorrelation for lags 0..n
    let padTarget = (2 * n).next_power_of_two();
    centered.resize(padTarget, 0.0);
    let mut spectrum = fftReal(&centered);
    // |X(f)|^2 = X * conj(X)
    for c in spectrum.iter_mut() {
        let mag = c.norm_sq();
        c.re = mag;
        c.im = 0.0;
    }
    ifft(&mut spectrum);
    let mut out = Vec::with_capacity(lim + 1);
    for lag in 0..=lim {
        out.push(spectrum[lag].re / var0);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approxEqVec(a: &[f64], b: &[f64], tol: f64) -> bool {
        if a.len() != b.len() {
            return false;
        }
        a.iter().zip(b.iter()).all(|(x, y)| (x - y).abs() < tol)
    }

    #[test]
    fn fftIfftRoundTrip() {
        let mut x: Vec<Complex> = (0..16).map(|i| Complex::new(i as f64, 0.0)).collect();
        let orig = x.clone();
        fft(&mut x);
        ifft(&mut x);
        for i in 0..16 {
            assert!((x[i].re - orig[i].re).abs() < 1e-9);
            assert!(x[i].im.abs() < 1e-9);
        }
    }

    #[test]
    fn fftLinearity() {
        let mut x: Vec<Complex> = (0..8).map(|i| Complex::new(i as f64, 0.0)).collect();
        let mut y: Vec<Complex> = (0..8).map(|i| Complex::new((8 - i) as f64, 0.0)).collect();
        let mut sumIn: Vec<Complex> = x
            .iter()
            .zip(y.iter())
            .map(|(a, b)| Complex::new(a.re + b.re, 0.0))
            .collect();
        fft(&mut x);
        fft(&mut y);
        fft(&mut sumIn);
        for i in 0..8 {
            assert!((x[i].re + y[i].re - sumIn[i].re).abs() < 1e-9);
            assert!((x[i].im + y[i].im - sumIn[i].im).abs() < 1e-9);
        }
    }

    #[test]
    fn autocorrelationMatchesDirect() {
        // White noise: ACF should peak at lag 0, drop quickly after
        let values: Vec<f64> = (0..256).map(|i| (i as f64 * 0.1).sin()).collect();
        let direct = naiveAcf(&values, 32);
        let viaFft = autocorrelationFft(&values, 32);
        assert!(
            approxEqVec(&direct, &viaFft, 1e-6),
            "fft acf differs from direct: direct={:?} fft={:?}",
            &direct[..5],
            &viaFft[..5]
        );
    }

    fn naiveAcf(values: &[f64], maxLag: usize) -> Vec<f64> {
        let n = values.len();
        let mean: f64 = values.iter().sum::<f64>() / n as f64;
        let centered: Vec<f64> = values.iter().map(|v| v - mean).collect();
        let var0: f64 = centered.iter().map(|v| v * v).sum();
        let mut out = Vec::with_capacity(maxLag + 1);
        for lag in 0..=maxLag {
            let mut s = 0.0;
            for i in 0..(n - lag) {
                s += centered[i] * centered[i + lag];
            }
            out.push(s / var0);
        }
        out
    }

    #[test]
    fn autocorrelationLag0IsOne() {
        let v: Vec<f64> = (0..64).map(|i| i as f64).collect();
        let acf = autocorrelationFft(&v, 16);
        assert!((acf[0] - 1.0).abs() < 1e-9);
    }
}
