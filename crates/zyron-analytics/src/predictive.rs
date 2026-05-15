#![allow(non_snake_case)]
// Predictive analytics, time series and anomaly detection
// FORECAST, ANOMALY_DETECT, TREND, SEASONALITY_DETECT, ACF, PACF
// ARIMA(p,d,q) configurable order, fit by Hannan-Rissanen two-step

use crate::numeric::{KahanSum, choleskySolve};
use std::collections::HashMap;
use zyron_common::error::{Result, ZyronError};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ForecastMethod {
    ExponentialSmoothing,
    HoltWinters,
    Arima,
    LinearTrend,
    SeasonalDecompose,
}

impl ForecastMethod {
    pub fn fromStr(s: &str) -> Option<Self> {
        let lower = s.to_ascii_lowercase();
        match lower.as_str() {
            "ses" | "exponential_smoothing" | "es" => Some(Self::ExponentialSmoothing),
            "hw" | "holt_winters" | "triple_es" => Some(Self::HoltWinters),
            "arima" => Some(Self::Arima),
            "linear_trend" | "ols" => Some(Self::LinearTrend),
            "seasonal_decompose" | "decompose" => Some(Self::SeasonalDecompose),
            _ => None,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AnomalyMethod {
    ZScore,
    Mad,
    Iqr,
}

impl AnomalyMethod {
    pub fn fromStr(s: &str) -> Option<Self> {
        match s.to_ascii_lowercase().as_str() {
            "z" | "zscore" | "z_score" => Some(Self::ZScore),
            "mad" => Some(Self::Mad),
            "iqr" => Some(Self::Iqr),
            _ => None,
        }
    }
}

#[derive(Debug, Clone)]
pub struct ForecastPoint {
    pub timestampMs: i64,
    pub value: f64,
    pub lower: f64,
    pub upper: f64,
}

#[derive(Debug, Clone)]
pub struct SeasonalComponent {
    pub period: usize,
    pub seasonalIndices: Vec<f64>,
    pub trend: Vec<f64>,
    pub residual: Vec<f64>,
}

/// Linear OLS trend, returns (slope, intercept)
pub fn trend(values: &[f64]) -> (f64, f64) {
    let n = values.len();
    if n < 2 {
        return (0.0, values.first().copied().unwrap_or(0.0));
    }
    let nf = n as f64;
    let mut sumX = 0.0f64;
    let mut sumY = 0.0f64;
    for (i, &y) in values.iter().enumerate() {
        sumX += i as f64;
        sumY += y;
    }
    let meanX = sumX / nf;
    let meanY = sumY / nf;
    let mut num = 0.0f64;
    let mut den = 0.0f64;
    for (i, &y) in values.iter().enumerate() {
        let dx = i as f64 - meanX;
        let dy = y - meanY;
        num += dx * dy;
        den += dx * dx;
    }
    if den == 0.0 {
        (0.0, meanY)
    } else {
        let slope = num / den;
        (slope, meanY - slope * meanX)
    }
}

/// Auto-correlation function up to maxLag
///
/// Auto-switches between two algorithms (F15):
/// - Direct O(n * maxLag) loop when the work is small or maxLag is tiny
/// - Wiener-Khinchin via FFT (O(n log n)) when n * maxLag is large
///
/// The crossover threshold is set where the FFT overhead (twiddle table,
/// padding to next power of two, two complex passes) breaks even with
/// the direct loop. Below it the direct loop wins on constant factors;
/// above it the FFT path dominates by an asymptotic margin
const FFT_ACF_CROSSOVER: usize = 200_000;

pub fn acf(values: &[f64], maxLag: usize) -> Vec<f64> {
    let n = values.len();
    if n == 0 {
        return Vec::new();
    }
    let lim = maxLag.min(n.saturating_sub(1));
    if n.saturating_mul(lim) >= FFT_ACF_CROSSOVER && lim >= 4 {
        return crate::fft::autocorrelationFft(values, lim);
    }
    let mean: f64 = values.iter().sum::<f64>() / n as f64;
    let centered: Vec<f64> = values.iter().map(|v| v - mean).collect();
    let var0: f64 = centered.iter().map(|v| v * v).sum();
    if var0 == 0.0 {
        return vec![1.0; lim + 1];
    }
    let mut out = Vec::with_capacity(lim + 1);
    out.push(1.0);
    for lag in 1..=lim {
        let mut s = 0.0f64;
        for i in 0..(n - lag) {
            s += centered[i] * centered[i + lag];
        }
        out.push(s / var0);
    }
    out
}

/// Partial autocorrelation by Levinson-Durbin recursion on ACF
pub fn pacf(values: &[f64], maxLag: usize) -> Vec<f64> {
    let r = acf(values, maxLag);
    if r.is_empty() {
        return Vec::new();
    }
    let mut phi = vec![0.0f64; maxLag + 1];
    let mut prev = vec![0.0f64; maxLag + 1];
    let mut e = r[0];
    let mut out = vec![1.0f64; maxLag + 1];
    for k in 1..=maxLag {
        if k >= r.len() {
            break;
        }
        let mut num = r[k];
        for j in 1..k {
            num -= prev[j] * r[k - j];
        }
        let kk = if e == 0.0 { 0.0 } else { num / e };
        out[k] = kk;
        phi[k] = kk;
        for j in 1..k {
            phi[j] = prev[j] - kk * prev[k - j];
        }
        e *= 1.0 - kk * kk;
        prev[..=k].copy_from_slice(&phi[..=k]);
    }
    out
}

/// Simple exponential smoothing forecast
/// Fits alpha by minimizing in-sample MSE if not provided
pub fn exponentialSmoothing(values: &[f64], periods: usize, alpha: Option<f64>) -> Vec<f64> {
    if values.is_empty() || periods == 0 {
        return Vec::new();
    }
    let alpha = alpha.unwrap_or_else(|| fitAlpha(values));
    let alpha = alpha.clamp(0.001, 0.999);
    let mut level = values[0];
    for &v in values.iter().skip(1) {
        level = alpha * v + (1.0 - alpha) * level;
    }
    vec![level; periods]
}

fn fitAlpha(values: &[f64]) -> f64 {
    let mut bestAlpha = 0.3;
    let mut bestMse = f64::INFINITY;
    let mut a = 0.05;
    while a < 0.99 {
        let mut level = values[0];
        let mut sse = KahanSum::new();
        for i in 1..values.len() {
            let pred = level;
            sse.add((values[i] - pred).powi(2));
            level = a * values[i] + (1.0 - a) * level;
        }
        if sse.value() < bestMse {
            bestMse = sse.value();
            bestAlpha = a;
        }
        a += 0.05;
    }
    bestAlpha
}

/// Holt-Winters triple exponential smoothing, additive seasonality
pub fn holtWinters(
    values: &[f64],
    periods: usize,
    seasonLength: usize,
    alpha: f64,
    beta: f64,
    gamma: f64,
) -> Vec<f64> {
    let n = values.len();
    if n < seasonLength.max(2) || periods == 0 {
        return Vec::new();
    }
    let alpha = alpha.clamp(0.001, 0.999);
    let beta = beta.clamp(0.001, 0.999);
    let gamma = gamma.clamp(0.001, 0.999);
    // Initialize seasonal indices via average of first season
    let mut s = vec![0.0f64; seasonLength];
    for i in 0..seasonLength {
        s[i] = values[i] - values[..seasonLength].iter().sum::<f64>() / seasonLength as f64;
    }
    let mut level = values[..seasonLength].iter().sum::<f64>() / seasonLength as f64;
    let mut trend = (values[seasonLength] - values[0]) / seasonLength as f64;
    for i in seasonLength..n {
        let prevLevel = level;
        level = alpha * (values[i] - s[i % seasonLength]) + (1.0 - alpha) * (level + trend);
        trend = beta * (level - prevLevel) + (1.0 - beta) * trend;
        s[i % seasonLength] = gamma * (values[i] - level) + (1.0 - gamma) * s[i % seasonLength];
    }
    let mut out = Vec::with_capacity(periods);
    for h in 1..=periods {
        let f = level + h as f64 * trend + s[(n + h - 1) % seasonLength];
        out.push(f);
    }
    out
}

/// Simple linear trend forecast
pub fn linearTrendForecast(values: &[f64], periods: usize) -> Vec<f64> {
    let (slope, intercept) = trend(values);
    let n = values.len();
    (0..periods)
        .map(|h| slope * (n + h) as f64 + intercept)
        .collect()
}

/// Classical additive decomposition into trend, seasonal, residual
pub fn seasonalDecompose(values: &[f64], period: usize) -> SeasonalComponent {
    let n = values.len();
    let mut comp = SeasonalComponent {
        period,
        seasonalIndices: vec![0.0; period],
        trend: vec![0.0; n],
        residual: vec![0.0; n],
    };
    if n < period * 2 || period < 2 {
        return comp;
    }
    // Centered moving average for trend, rolling-window form so cost is
    // O(n) instead of O(n * period) (F14)
    let half = period / 2;
    let mut trend = vec![f64::NAN; n];
    if n >= 2 * half + 1 {
        let win = 2 * half + 1;
        let mut runningSum: f64 = values[..win].iter().sum();
        trend[half] = runningSum / win as f64;
        for i in (half + 1)..(n - half) {
            runningSum += values[i + half] - values[i - half - 1];
            trend[i] = runningSum / win as f64;
        }
    }
    let mut detrended = vec![f64::NAN; n];
    for i in 0..n {
        if !trend[i].is_nan() {
            detrended[i] = values[i] - trend[i];
        }
    }
    let mut seasonalSum = vec![0.0f64; period];
    let mut seasonalCnt = vec![0u64; period];
    for i in 0..n {
        if !detrended[i].is_nan() {
            let phase = i % period;
            seasonalSum[phase] += detrended[i];
            seasonalCnt[phase] += 1;
        }
    }
    for i in 0..period {
        if seasonalCnt[i] > 0 {
            comp.seasonalIndices[i] = seasonalSum[i] / seasonalCnt[i] as f64;
        }
    }
    // Recenter seasonal to mean 0
    let seasonalMean: f64 = comp.seasonalIndices.iter().sum::<f64>() / period as f64;
    for v in comp.seasonalIndices.iter_mut() {
        *v -= seasonalMean;
    }
    let mut tFinal = vec![0.0f64; n];
    for i in 0..n {
        if !trend[i].is_nan() {
            tFinal[i] = trend[i];
        } else {
            tFinal[i] = values[i] - comp.seasonalIndices[i % period];
        }
    }
    comp.trend = tFinal;
    let mut resid = vec![0.0f64; n];
    for i in 0..n {
        resid[i] = values[i] - comp.trend[i] - comp.seasonalIndices[i % period];
    }
    comp.residual = resid;
    comp
}

/// ARIMA(p, d, q) forecast
/// Differencing of order d, AR(p) and MA(q) coefficients fit by
/// Hannan-Rissanen two-step
pub fn arima(values: &[f64], periods: usize, p: usize, d: usize, q: usize) -> Result<Vec<f64>> {
    if p > 5 || d > 5 || q > 5 {
        return Err(ZyronError::InvalidParameter {
            name: "arima_order".to_string(),
            value: format!("(p={}, d={}, q={}) all must be <= 5", p, d, q),
        });
    }
    let needed = p.max(q) + d * 2 + 4;
    if values.len() < needed {
        return Err(ZyronError::InvalidParameter {
            name: "arima_data".to_string(),
            value: format!("need at least {} observations", needed),
        });
    }
    // Apply d-th differencing
    let mut diffed = values.to_vec();
    let mut diffHistory: Vec<Vec<f64>> = Vec::with_capacity(d);
    for _ in 0..d {
        diffHistory.push(diffed.clone());
        let mut nxt = Vec::with_capacity(diffed.len() - 1);
        for i in 1..diffed.len() {
            nxt.push(diffed[i] - diffed[i - 1]);
        }
        diffed = nxt;
    }

    // Step 1: Long-AR estimate of residuals via Yule-Walker on diffed series
    let arOrderLong = (p + q).max(1).min(diffed.len() / 2).max(p);
    let arCoefsLong = yuleWalker(&diffed, arOrderLong);
    let mut residuals = vec![0.0f64; diffed.len()];
    for i in arOrderLong..diffed.len() {
        let mut yhat = 0.0f64;
        for j in 0..arOrderLong {
            yhat += arCoefsLong[j] * diffed[i - j - 1];
        }
        residuals[i] = diffed[i] - yhat;
    }

    // Step 2: OLS regression of diffed on lags of diffed and lags of residuals
    let nReg = diffed.len() - p.max(q);
    let pAr = p;
    let pMa = q;
    let nCols = pAr + pMa;
    let mut arCoef = vec![0.0f64; pAr];
    let mut maCoef = vec![0.0f64; pMa];
    if nCols > 0 && nReg > nCols {
        let start = p.max(q);
        let mut a = vec![0.0f64; nCols * nCols];
        let mut b = vec![0.0f64; nCols];
        for t in start..diffed.len() {
            let mut row = Vec::with_capacity(nCols);
            for k in 0..pAr {
                row.push(diffed[t - k - 1]);
            }
            for k in 0..pMa {
                row.push(residuals[t - k - 1]);
            }
            for j in 0..nCols {
                for k in 0..nCols {
                    a[j * nCols + k] += row[j] * row[k];
                }
                b[j] += row[j] * diffed[t];
            }
        }
        // Symmetrize and add tiny ridge for numerical stability
        for j in 0..nCols {
            a[j * nCols + j] += 1e-6;
            for k in 0..j {
                let avg = 0.5 * (a[j * nCols + k] + a[k * nCols + j]);
                a[j * nCols + k] = avg;
                a[k * nCols + j] = avg;
            }
        }
        if choleskySolve(&mut a, &mut b, nCols).is_ok() {
            for k in 0..pAr {
                arCoef[k] = b[k];
            }
            for k in 0..pMa {
                maCoef[k] = b[pAr + k];
            }
        }
    }

    // Forecast on diffed series, then invert differencing
    let mut history = diffed.clone();
    let mut residHistory = residuals.clone();
    let mut forecastDiffed = Vec::with_capacity(periods);
    for _ in 0..periods {
        let n = history.len();
        let mut yhat = 0.0f64;
        for k in 0..pAr {
            if n >= k + 1 {
                yhat += arCoef[k] * history[n - k - 1];
            }
        }
        for k in 0..pMa {
            if residHistory.len() >= k + 1 {
                yhat += maCoef[k] * residHistory[residHistory.len() - k - 1];
            }
        }
        forecastDiffed.push(yhat);
        history.push(yhat);
        residHistory.push(0.0);
    }

    // Invert differencing using saved last values per stage
    let mut current = forecastDiffed;
    for stage in (0..d).rev() {
        let prior = &diffHistory[stage];
        let mut last = *prior.last().unwrap_or(&0.0);
        let mut undiffed = Vec::with_capacity(current.len());
        for v in current {
            last += v;
            undiffed.push(last);
        }
        current = undiffed;
    }
    Ok(current)
}

fn yuleWalker(values: &[f64], order: usize) -> Vec<f64> {
    if order == 0 || values.is_empty() {
        return Vec::new();
    }
    let r = acf(values, order);
    let mut a = vec![0.0f64; order * order];
    let mut b = vec![0.0f64; order];
    for j in 0..order {
        for k in 0..order {
            let lag = (j as isize - k as isize).unsigned_abs();
            a[j * order + k] = r.get(lag).copied().unwrap_or(0.0);
        }
        b[j] = r.get(j + 1).copied().unwrap_or(0.0);
    }
    for j in 0..order {
        a[j * order + j] += 1e-6;
    }
    if choleskySolve(&mut a, &mut b, order).is_ok() {
        b
    } else {
        vec![0.0; order]
    }
}

/// Convenience FORECAST entry point
pub fn forecast(
    values: &[f64],
    periods: usize,
    method: ForecastMethod,
    extra: &HashMap<String, f64>,
) -> Result<Vec<f64>> {
    match method {
        ForecastMethod::ExponentialSmoothing => {
            let alpha = extra.get("alpha").copied();
            Ok(exponentialSmoothing(values, periods, alpha))
        }
        ForecastMethod::HoltWinters => {
            let season = extra.get("season").copied().unwrap_or(7.0) as usize;
            let alpha = extra.get("alpha").copied().unwrap_or(0.3);
            let beta = extra.get("beta").copied().unwrap_or(0.1);
            let gamma = extra.get("gamma").copied().unwrap_or(0.1);
            Ok(holtWinters(
                values,
                periods,
                season.max(2),
                alpha,
                beta,
                gamma,
            ))
        }
        ForecastMethod::Arima => {
            let p = extra.get("p").copied().unwrap_or(1.0) as usize;
            let d = extra.get("d").copied().unwrap_or(1.0) as usize;
            let q = extra.get("q").copied().unwrap_or(1.0) as usize;
            arima(values, periods, p, d, q)
        }
        ForecastMethod::LinearTrend => Ok(linearTrendForecast(values, periods)),
        ForecastMethod::SeasonalDecompose => {
            let season = extra.get("season").copied().unwrap_or(7.0) as usize;
            let comp = seasonalDecompose(values, season.max(2));
            // Forecast from trend extrapolation plus replayed seasonal
            let (slope, intercept) = trend(&comp.trend);
            let n = values.len();
            let mut out = Vec::with_capacity(periods);
            for h in 0..periods {
                let baseline = slope * (n + h) as f64 + intercept;
                let phase = (n + h) % season.max(2);
                out.push(baseline + comp.seasonalIndices.get(phase).copied().unwrap_or(0.0));
            }
            Ok(out)
        }
    }
}

/// Anomaly detection, returns (index, isAnomaly, score) per input point
pub fn anomalyDetect(
    values: &[f64],
    method: AnomalyMethod,
    threshold: f64,
) -> Vec<(usize, bool, f64)> {
    let n = values.len();
    if n == 0 {
        return Vec::new();
    }
    match method {
        AnomalyMethod::ZScore => {
            let mean: f64 = values.iter().sum::<f64>() / n as f64;
            let var =
                values.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / (n as f64 - 1.0).max(1.0);
            let std = var.sqrt();
            (0..n)
                .map(|i| {
                    let s = if std > 0.0 {
                        (values[i] - mean) / std
                    } else {
                        0.0
                    };
                    (i, s.abs() > threshold, s)
                })
                .collect()
        }
        AnomalyMethod::Mad => {
            let mut sorted: Vec<f64> = values.to_vec();
            sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            let median = sorted[n / 2];
            let mut deviations: Vec<f64> = values.iter().map(|v| (v - median).abs()).collect();
            deviations.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            let mad = deviations[n / 2].max(1e-12);
            (0..n)
                .map(|i| {
                    let s = 0.6745 * (values[i] - median) / mad;
                    (i, s.abs() > threshold, s)
                })
                .collect()
        }
        AnomalyMethod::Iqr => {
            let mut sorted: Vec<f64> = values.to_vec();
            sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            let q1 = sorted[n / 4];
            let q3 = sorted[(3 * n) / 4];
            let iqr = q3 - q1;
            let lo = q1 - threshold * iqr;
            let hi = q3 + threshold * iqr;
            (0..n)
                .map(|i| {
                    let v = values[i];
                    let s = if v < lo {
                        v - lo
                    } else if v > hi {
                        v - hi
                    } else {
                        0.0
                    };
                    (i, v < lo || v > hi, s)
                })
                .collect()
        }
    }
}

/// Detects periodic patterns by inspecting ACF peaks
pub fn seasonalityDetect(values: &[f64], maxPeriod: usize) -> Vec<SeasonalComponent> {
    let r = acf(values, maxPeriod);
    let mut peaks: Vec<usize> = Vec::new();
    for lag in 2..r.len().saturating_sub(1) {
        if r[lag] > 0.3 && r[lag] > r[lag - 1] && r[lag] > r[lag + 1] {
            peaks.push(lag);
        }
    }
    peaks
        .into_iter()
        .map(|p| seasonalDecompose(values, p))
        .collect()
}

/// Cumulative-sum change point detection, returns indices where the
/// running deviation exceeds threshold
pub fn changePoints(values: &[f64], threshold: f64) -> Vec<usize> {
    let n = values.len();
    if n == 0 {
        return Vec::new();
    }
    let mean: f64 = values.iter().sum::<f64>() / n as f64;
    let mut cusum = 0.0f64;
    let mut points = Vec::new();
    for i in 0..n {
        cusum += values[i] - mean;
        if cusum.abs() > threshold {
            points.push(i);
            cusum = 0.0;
        }
    }
    points
}

#[cfg(test)]
mod tests {
    use super::*;
    use zyron_common::Xoshiro256pp;

    #[test]
    fn trendRecoversSlope() {
        let v: Vec<f64> = (0..50).map(|i| 2.0 * i as f64 + 5.0).collect();
        let (slope, intercept) = trend(&v);
        assert!((slope - 2.0).abs() < 1e-9);
        assert!((intercept - 5.0).abs() < 1e-9);
    }

    #[test]
    fn acfLag1ForRandomWalk() {
        let mut rng = Xoshiro256pp::fromSeed(1);
        let n = 1000;
        let mut v = Vec::with_capacity(n);
        let mut x = 0.0;
        for _ in 0..n {
            x += rng.nextNormal();
            v.push(x);
        }
        let r = acf(&v, 5);
        assert!(r[1] > 0.9, "lag-1 acf = {}", r[1]);
    }

    #[test]
    fn linearTrendForecastIncreasing() {
        let v: Vec<f64> = (0..30).map(|i| 2.0 * i as f64).collect();
        let f = linearTrendForecast(&v, 5);
        for i in 1..f.len() {
            assert!(f[i] > f[i - 1]);
        }
    }

    #[test]
    fn arimaProducesPeriods() {
        let mut rng = Xoshiro256pp::fromSeed(7);
        let mut v = Vec::with_capacity(200);
        let mut x = 0.0;
        for _ in 0..200 {
            x = 0.6 * x + rng.nextNormal();
            v.push(x);
        }
        let f = arima(&v, 10, 1, 1, 1).unwrap();
        assert_eq!(f.len(), 10);
        for f in f {
            assert!(f.is_finite());
        }
    }

    #[test]
    fn anomalyDetectsOutlier() {
        let mut v: Vec<f64> = (0..100).map(|_| 1.0).collect();
        v[50] = 100.0;
        let r = anomalyDetect(&v, AnomalyMethod::ZScore, 3.0);
        assert!(r[50].1, "anomaly should flag index 50");
    }

    #[test]
    fn seasonalDecomposeReassembles() {
        let n = 100;
        let mut v = Vec::with_capacity(n);
        for i in 0..n {
            v.push(0.5 * i as f64 + (2.0 * std::f64::consts::PI * (i % 7) as f64 / 7.0).sin());
        }
        let comp = seasonalDecompose(&v, 7);
        for i in 0..n {
            let recon = comp.trend[i] + comp.seasonalIndices[i % 7] + comp.residual[i];
            assert!((v[i] - recon).abs() < 1e-9);
        }
    }
}
