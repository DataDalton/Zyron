//! Statistical functions: regression, correlation, percentiles, forecasting, outlier detection.

use zyron_common::{Result, ZyronError};

/// Linear regression: fits y = slope*x + intercept. Returns (slope, intercept, r_squared).
pub fn linear_regression(x: &[f64], y: &[f64]) -> Result<(f64, f64, f64)> {
    if x.len() != y.len() {
        return Err(ZyronError::ExecutionError(
            "x and y must have same length".into(),
        ));
    }
    let n = x.len();
    if n < 2 {
        return Err(ZyronError::ExecutionError(
            "Need at least 2 data points".into(),
        ));
    }

    let mean_x: f64 = x.iter().sum::<f64>() / n as f64;
    let mean_y: f64 = y.iter().sum::<f64>() / n as f64;

    let mut ss_xy = 0.0;
    let mut ss_xx = 0.0;
    let mut ss_yy = 0.0;

    for i in 0..n {
        let dx = x[i] - mean_x;
        let dy = y[i] - mean_y;
        ss_xy += dx * dy;
        ss_xx += dx * dx;
        ss_yy += dy * dy;
    }

    if ss_xx.abs() < 1e-15 {
        return Err(ZyronError::ExecutionError(
            "All x values are identical".into(),
        ));
    }

    let slope = ss_xy / ss_xx;
    let intercept = mean_y - slope * mean_x;
    let r_squared = if ss_yy.abs() < 1e-15 {
        0.0
    } else {
        (ss_xy * ss_xy) / (ss_xx * ss_yy)
    };

    Ok((slope, intercept, r_squared))
}

/// Pearson correlation coefficient.
pub fn correlation(x: &[f64], y: &[f64]) -> Result<f64> {
    if x.len() != y.len() {
        return Err(ZyronError::ExecutionError(
            "x and y must have same length".into(),
        ));
    }
    if x.len() < 2 {
        return Err(ZyronError::ExecutionError(
            "Need at least 2 data points".into(),
        ));
    }

    let n = x.len() as f64;
    let mean_x: f64 = x.iter().sum::<f64>() / n;
    let mean_y: f64 = y.iter().sum::<f64>() / n;

    let mut num = 0.0;
    let mut ssx = 0.0;
    let mut ssy = 0.0;
    for i in 0..x.len() {
        let dx = x[i] - mean_x;
        let dy = y[i] - mean_y;
        num += dx * dy;
        ssx += dx * dx;
        ssy += dy * dy;
    }

    if ssx == 0.0 || ssy == 0.0 {
        return Ok(0.0);
    }

    Ok(num / (ssx * ssy).sqrt())
}

/// Sample covariance.
pub fn covariance(x: &[f64], y: &[f64]) -> Result<f64> {
    if x.len() != y.len() {
        return Err(ZyronError::ExecutionError(
            "x and y must have same length".into(),
        ));
    }
    if x.len() < 2 {
        return Err(ZyronError::ExecutionError(
            "Need at least 2 data points".into(),
        ));
    }

    let n = x.len() as f64;
    let mean_x: f64 = x.iter().sum::<f64>() / n;
    let mean_y: f64 = y.iter().sum::<f64>() / n;

    let mut cov = 0.0;
    for i in 0..x.len() {
        cov += (x[i] - mean_x) * (y[i] - mean_y);
    }
    Ok(cov / (n - 1.0))
}

/// Z-score: (value - mean) / stddev.
pub fn zscore(value: f64, mean: f64, stddev: f64) -> f64 {
    if stddev == 0.0 {
        return 0.0;
    }
    (value - mean) / stddev
}

/// Exact percentile from a sorted slice.
pub fn percentile(values: &mut [f64], p: f64) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let clamped = p.clamp(0.0, 1.0);
    if values.len() == 1 {
        return values[0];
    }
    let idx = clamped * (values.len() - 1) as f64;
    let lower = idx.floor() as usize;
    let upper = idx.ceil() as usize;
    if lower == upper {
        return values[lower];
    }
    let weight = idx - lower as f64;
    values[lower] * (1.0 - weight) + values[upper] * weight
}

/// Population standard deviation.
pub fn stddev_pop(values: &[f64]) -> f64 {
    variance_pop(values).sqrt()
}

/// Sample standard deviation.
pub fn stddev_sample(values: &[f64]) -> f64 {
    variance_sample(values).sqrt()
}

/// Population variance.
pub fn variance_pop(values: &[f64]) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    let mean: f64 = values.iter().sum::<f64>() / values.len() as f64;
    values.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / values.len() as f64
}

/// Sample variance (n-1 denominator).
pub fn variance_sample(values: &[f64]) -> f64 {
    if values.len() < 2 {
        return 0.0;
    }
    let mean: f64 = values.iter().sum::<f64>() / values.len() as f64;
    values.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (values.len() - 1) as f64
}

/// Skewness (sample).
pub fn skewness(values: &[f64]) -> f64 {
    if values.len() < 2 {
        return 0.0;
    }
    let n = values.len() as f64;
    let mean: f64 = values.iter().sum::<f64>() / n;
    let std = stddev_sample(values);
    if std == 0.0 {
        return 0.0;
    }
    let m3: f64 = values.iter().map(|x| (x - mean).powi(3)).sum::<f64>() / n;
    m3 / std.powi(3)
}

/// Kurtosis (excess kurtosis; normal distribution = 0).
pub fn kurtosis(values: &[f64]) -> f64 {
    if values.len() < 2 {
        return 0.0;
    }
    let n = values.len() as f64;
    let mean: f64 = values.iter().sum::<f64>() / n;
    let var = variance_sample(values);
    if var == 0.0 {
        return 0.0;
    }
    let m4: f64 = values.iter().map(|x| (x - mean).powi(4)).sum::<f64>() / n;
    m4 / var.powi(2) - 3.0
}

/// Exponential smoothing with smoothing factor alpha (0 < alpha <= 1).
pub fn exponential_smoothing(values: &[f64], alpha: f64) -> Vec<f64> {
    if values.is_empty() {
        return Vec::new();
    }
    let alpha = alpha.clamp(0.0, 1.0);
    let mut result = Vec::with_capacity(values.len());
    let mut prev = values[0];
    result.push(prev);
    for &v in &values[1..] {
        let smoothed = alpha * v + (1.0 - alpha) * prev;
        result.push(smoothed);
        prev = smoothed;
    }
    result
}

/// Linear forecast: fits y = m*x + b and projects to future_x.
pub fn forecast_linear(x: &[f64], y: &[f64], future_x: &[f64]) -> Result<Vec<f64>> {
    let (slope, intercept, _) = linear_regression(x, y)?;
    Ok(future_x.iter().map(|&xi| slope * xi + intercept).collect())
}

/// Outlier detection via z-score (returns bool per value; true = outlier).
pub fn outlier_detect_zscore(values: &[f64], threshold: f64) -> Vec<bool> {
    if values.is_empty() {
        return Vec::new();
    }
    let mean: f64 = values.iter().sum::<f64>() / values.len() as f64;
    let std = stddev_pop(values);
    if std == 0.0 {
        return vec![false; values.len()];
    }
    values
        .iter()
        .map(|x| ((x - mean) / std).abs() > threshold)
        .collect()
}

/// Outlier detection via IQR (Tukey's fences, factor is typically 1.5).
pub fn outlier_detect_iqr(values: &[f64], factor: f64) -> Vec<bool> {
    if values.is_empty() {
        return Vec::new();
    }
    let mut sorted = values.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let q1 = percentile(&mut sorted.clone(), 0.25);
    let q3 = percentile(&mut sorted.clone(), 0.75);
    let iqr = q3 - q1;
    let lower = q1 - factor * iqr;
    let upper = q3 + factor * iqr;
    values.iter().map(|x| *x < lower || *x > upper).collect()
}

/// Simple moving average over a sliding window.
pub fn moving_average(values: &[f64], window: usize) -> Vec<f64> {
    if window == 0 || values.is_empty() {
        return values.to_vec();
    }
    let mut result = Vec::with_capacity(values.len());
    for i in 0..values.len() {
        let start = i.saturating_sub(window - 1);
        let slice = &values[start..=i];
        let avg: f64 = slice.iter().sum::<f64>() / slice.len() as f64;
        result.push(avg);
    }
    result
}

/// Weighted moving average with explicit weights.
pub fn weighted_moving_average(values: &[f64], weights: &[f64]) -> Result<Vec<f64>> {
    if weights.is_empty() {
        return Err(ZyronError::ExecutionError("weights cannot be empty".into()));
    }
    let weight_sum: f64 = weights.iter().sum();
    if weight_sum.abs() < 1e-15 {
        return Err(ZyronError::ExecutionError("weights sum to zero".into()));
    }

    let mut result = Vec::with_capacity(values.len());
    for i in 0..values.len() {
        let mut sum = 0.0;
        let mut w_sum = 0.0;
        for (j, &w) in weights.iter().enumerate() {
            if i >= j {
                sum += values[i - j] * w;
                w_sum += w;
            }
        }
        result.push(if w_sum > 0.0 { sum / w_sum } else { 0.0 });
    }
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_linear_regression_perfect() {
        // y = 2x + 1
        let x = vec![0.0, 1.0, 2.0, 3.0, 4.0];
        let y = vec![1.0, 3.0, 5.0, 7.0, 9.0];
        let (slope, intercept, r2) = linear_regression(&x, &y).unwrap();
        assert!((slope - 2.0).abs() < 1e-10);
        assert!((intercept - 1.0).abs() < 1e-10);
        assert!((r2 - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_linear_regression_mismatch() {
        assert!(linear_regression(&[1.0], &[1.0, 2.0]).is_err());
    }

    #[test]
    fn test_correlation_perfect_positive() {
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let y = vec![2.0, 4.0, 6.0, 8.0];
        let r = correlation(&x, &y).unwrap();
        assert!((r - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_correlation_perfect_negative() {
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let y = vec![4.0, 3.0, 2.0, 1.0];
        let r = correlation(&x, &y).unwrap();
        assert!((r - (-1.0)).abs() < 1e-10);
    }

    #[test]
    fn test_correlation_uncorrelated() {
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let y = vec![2.0, 2.0, 2.0, 2.0];
        let r = correlation(&x, &y).unwrap();
        assert_eq!(r, 0.0);
    }

    #[test]
    fn test_covariance() {
        let x = vec![1.0, 2.0, 3.0];
        let y = vec![2.0, 4.0, 6.0];
        let cov = covariance(&x, &y).unwrap();
        assert!(cov > 0.0);
    }

    #[test]
    fn test_zscore() {
        assert_eq!(zscore(110.0, 100.0, 10.0), 1.0);
        assert_eq!(zscore(90.0, 100.0, 10.0), -1.0);
        assert_eq!(zscore(100.0, 100.0, 0.0), 0.0);
    }

    #[test]
    fn test_percentile() {
        let mut data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        assert_eq!(percentile(&mut data, 0.0), 1.0);
        assert_eq!(percentile(&mut data, 1.0), 5.0);
        assert_eq!(percentile(&mut data, 0.5), 3.0);
    }

    #[test]
    fn test_percentile_empty() {
        let mut data = Vec::new();
        assert_eq!(percentile(&mut data, 0.5), 0.0);
    }

    #[test]
    fn test_variance_pop() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        // mean = 3, variance_pop = sum((x-3)^2) / 5 = (4+1+0+1+4)/5 = 2
        assert!((variance_pop(&data) - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_variance_sample() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        // variance_sample = 10/4 = 2.5
        assert!((variance_sample(&data) - 2.5).abs() < 1e-10);
    }

    #[test]
    fn test_stddev_pop() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        assert!((stddev_pop(&data) - 2.0_f64.sqrt()).abs() < 1e-10);
    }

    #[test]
    fn test_skewness_symmetric() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let s = skewness(&data);
        assert!(s.abs() < 0.01); // symmetric data has ~0 skewness
    }

    #[test]
    fn test_skewness_right_skewed() {
        let data = vec![1.0, 1.0, 1.0, 2.0, 10.0];
        let s = skewness(&data);
        assert!(s > 0.0); // right-skewed
    }

    #[test]
    fn test_exponential_smoothing() {
        let data = vec![1.0, 2.0, 3.0, 4.0];
        let smoothed = exponential_smoothing(&data, 0.5);
        assert_eq!(smoothed.len(), 4);
        assert_eq!(smoothed[0], 1.0);
        // smoothed[1] = 0.5*2 + 0.5*1 = 1.5
        assert!((smoothed[1] - 1.5).abs() < 1e-10);
    }

    #[test]
    fn test_forecast_linear() {
        let x = vec![0.0, 1.0, 2.0];
        let y = vec![1.0, 3.0, 5.0]; // y = 2x + 1
        let future = vec![3.0, 4.0];
        let predicted = forecast_linear(&x, &y, &future).unwrap();
        assert!((predicted[0] - 7.0).abs() < 1e-10);
        assert!((predicted[1] - 9.0).abs() < 1e-10);
    }

    #[test]
    fn test_outlier_zscore() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 100.0];
        let outliers = outlier_detect_zscore(&data, 2.0);
        assert!(outliers[5]); // 100 is an outlier
        assert!(!outliers[0]);
    }

    #[test]
    fn test_outlier_iqr() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 100.0];
        let outliers = outlier_detect_iqr(&data, 1.5);
        assert!(outliers[7]); // 100 is an outlier
    }

    #[test]
    fn test_moving_average() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let ma = moving_average(&data, 3);
        assert_eq!(ma[0], 1.0);
        assert_eq!(ma[1], 1.5); // (1+2)/2
        assert_eq!(ma[2], 2.0); // (1+2+3)/3
        assert_eq!(ma[3], 3.0); // (2+3+4)/3
        assert_eq!(ma[4], 4.0); // (3+4+5)/3
    }

    #[test]
    fn test_moving_average_window_zero() {
        let data = vec![1.0, 2.0];
        let ma = moving_average(&data, 0);
        assert_eq!(ma, data);
    }

    #[test]
    fn test_weighted_moving_average() {
        let data = vec![10.0, 20.0, 30.0];
        let weights = vec![0.5, 0.3, 0.2]; // weights for current, prev, prev-prev
        let wma = weighted_moving_average(&data, &weights).unwrap();
        assert_eq!(wma.len(), 3);
    }

    #[test]
    fn test_weighted_moving_average_empty_weights() {
        let data = vec![1.0, 2.0];
        let weights: Vec<f64> = Vec::new();
        assert!(weighted_moving_average(&data, &weights).is_err());
    }

    #[test]
    fn test_kurtosis_normal_like() {
        // Roughly normal data should have kurtosis near 0
        let data: Vec<f64> = (0..100).map(|i| i as f64).collect();
        let k = kurtosis(&data);
        assert!(k > -2.0 && k < 2.0);
    }
}
