//! Time-series functions: bucketing, gap filling, rate/delta/derivative,
//! LTTB downsampling. Uses integer arithmetic on epoch microseconds
//! to avoid floating-point precision issues.

use zyron_common::{Interval, days_from_ymd, ymd_from_days};

/// Floors a timestamp to the bucket boundary using integer division.
/// interval_micros: bucket size in microseconds.
pub fn time_bucket(interval_micros: i64, timestamp_micros: i64) -> i64 {
    if interval_micros <= 0 {
        return timestamp_micros;
    }
    // Use Euclidean division for correct behavior with negative timestamps
    let bucket = timestamp_micros.div_euclid(interval_micros);
    bucket * interval_micros
}

/// Generates all bucket boundary timestamps in the range [start, end).
/// Used for gap filling when source data has missing buckets.
pub fn time_bucket_gapfill(interval_micros: i64, start: i64, end: i64) -> Vec<i64> {
    if interval_micros <= 0 || start >= end {
        return Vec::new();
    }
    let first = time_bucket(interval_micros, start);
    let first = if first < start {
        first + interval_micros
    } else {
        first
    };

    let mut result = Vec::new();
    let mut current = first;
    while current < end {
        result.push(current);
        current += interval_micros;
    }
    result
}

/// Last observation carried forward. For each null value, uses the
/// most recent non-null value. Leading nulls become 0.0.
pub fn locf(values: &[Option<f64>]) -> Vec<f64> {
    let mut result = Vec::with_capacity(values.len());
    let mut last: f64 = 0.0;
    let mut seen = false;
    for v in values {
        match v {
            Some(x) => {
                last = *x;
                seen = true;
                result.push(*x);
            }
            None => {
                if seen {
                    result.push(last);
                } else {
                    result.push(0.0);
                }
            }
        }
    }
    result
}

/// Linear interpolation between known values at given timestamps.
/// For each null, computes the interpolated value using the surrounding
/// non-null values. Leading/trailing nulls get the nearest known value.
pub fn interpolate(timestamps: &[i64], values: &[Option<f64>]) -> Vec<f64> {
    let n = values.len();
    let mut result = vec![0.0f64; n];

    if timestamps.len() != n || n == 0 {
        return result;
    }

    // Find indices of known values
    let known: Vec<usize> = (0..n).filter(|i| values[*i].is_some()).collect();

    if known.is_empty() {
        return result;
    }

    for i in 0..n {
        if let Some(v) = values[i] {
            result[i] = v;
            continue;
        }

        // Find surrounding known values
        let prev = known.iter().rev().find(|&&k| k < i).copied();
        let next = known.iter().find(|&&k| k > i).copied();

        match (prev, next) {
            (Some(p), Some(q)) => {
                let v_p = values[p].unwrap_or(0.0);
                let v_q = values[q].unwrap_or(0.0);
                let t_p = timestamps[p] as f64;
                let t_q = timestamps[q] as f64;
                let t_i = timestamps[i] as f64;
                let ratio = (t_i - t_p) / (t_q - t_p);
                result[i] = v_p + ratio * (v_q - v_p);
            }
            (Some(p), None) => {
                result[i] = values[p].unwrap_or(0.0);
            }
            (None, Some(q)) => {
                result[i] = values[q].unwrap_or(0.0);
            }
            (None, None) => {
                result[i] = 0.0;
            }
        }
    }

    result
}

/// Exponential Moving Average.
/// alpha: smoothing factor (0 < alpha <= 1). Higher alpha = more weight on recent values.
pub fn ema(values: &[f64], alpha: f64) -> Vec<f64> {
    if values.is_empty() {
        return Vec::new();
    }
    let alpha = alpha.clamp(0.0, 1.0);
    let mut result = Vec::with_capacity(values.len());
    let mut prev = values[0];
    result.push(prev);
    for &v in &values[1..] {
        let new_ema = alpha * v + (1.0 - alpha) * prev;
        result.push(new_ema);
        prev = new_ema;
    }
    result
}

/// Rate of change per second.
/// First value is None (no previous point to compare).
pub fn rate(timestamps: &[i64], values: &[f64]) -> Vec<Option<f64>> {
    let n = values.len();
    if n == 0 || timestamps.len() != n {
        return Vec::new();
    }
    let mut result = Vec::with_capacity(n);
    result.push(None);
    for i in 1..n {
        let dt_micros = timestamps[i] - timestamps[i - 1];
        if dt_micros <= 0 {
            result.push(None);
        } else {
            let dv = values[i] - values[i - 1];
            let dt_secs = dt_micros as f64 / 1_000_000.0;
            result.push(Some(dv / dt_secs));
        }
    }
    result
}

/// Difference from the previous value (discrete derivative).
pub fn delta(values: &[f64]) -> Vec<Option<f64>> {
    if values.is_empty() {
        return Vec::new();
    }
    let mut result = Vec::with_capacity(values.len());
    result.push(None);
    for i in 1..values.len() {
        result.push(Some(values[i] - values[i - 1]));
    }
    result
}

/// Time-normalized rate of change (derivative).
/// Same as rate() but with explicit time input.
pub fn derivative(timestamps: &[i64], values: &[f64]) -> Vec<Option<f64>> {
    rate(timestamps, values)
}

/// Largest Triangle Three Buckets downsampling.
/// Reduces the number of points while preserving the visual shape of the data.
/// Returns indices into the original arrays.
pub fn lttb(timestamps: &[f64], values: &[f64], threshold: usize) -> Vec<usize> {
    let n = timestamps.len();
    if n != values.len() {
        return Vec::new();
    }
    if threshold >= n || threshold < 3 {
        return (0..n).collect();
    }

    let mut sampled = Vec::with_capacity(threshold);
    sampled.push(0); // always keep first

    let bucket_size = (n - 2) as f64 / (threshold - 2) as f64;

    let mut a = 0usize;

    for i in 0..threshold - 2 {
        // Bucket range for next point
        let avg_range_start = ((i + 1) as f64 * bucket_size).floor() as usize + 1;
        let avg_range_end = ((i + 2) as f64 * bucket_size).floor() as usize + 1;
        let avg_range_end = avg_range_end.min(n);

        // Compute average point in next bucket
        let avg_range_len = (avg_range_end - avg_range_start).max(1) as f64;
        let mut avg_x = 0.0f64;
        let mut avg_y = 0.0f64;
        for j in avg_range_start..avg_range_end {
            avg_x += timestamps[j];
            avg_y += values[j];
        }
        avg_x /= avg_range_len;
        avg_y /= avg_range_len;

        // Current bucket range
        let range_start = (i as f64 * bucket_size).floor() as usize + 1;
        let range_end = ((i + 1) as f64 * bucket_size).floor() as usize + 1;

        // Point at a (previous selected)
        let point_a_x = timestamps[a];
        let point_a_y = values[a];

        // Find point in current bucket that forms largest triangle
        let mut max_area = -1.0;
        let mut max_idx = range_start;
        for j in range_start..range_end.min(n) {
            let area = ((point_a_x - avg_x) * (values[j] - point_a_y)
                - (point_a_x - timestamps[j]) * (avg_y - point_a_y))
                .abs()
                * 0.5;
            if area > max_area {
                max_area = area;
                max_idx = j;
            }
        }

        sampled.push(max_idx);
        a = max_idx;
    }

    sampled.push(n - 1); // always keep last
    sampled
}

// ---------------------------------------------------------------------------
// Calendar-aware time bucketing
// ---------------------------------------------------------------------------

/// Floors a timestamp to a calendar-aware bucket boundary. Used for buckets
/// expressed as a composite Interval (months/days/nanoseconds).
///
/// Semantics:
/// - Month-based intervals (e.g. INTERVAL '1 month', '1 year', '1 quarter') align
///   to the start of the calendar month (or multi-month boundary). A 1-month
///   bucket for 2024-03-15 returns the micros for 2024-03-01 00:00:00.
/// - Day-based intervals align to the start of the day (floor division of days
///   since epoch).
/// - Sub-day intervals fall back to integer-microsecond bucketing from epoch 0.
///
/// When the interval is purely sub-day (months == 0 && days == 0 && nanoseconds > 0),
/// this behaves like `time_bucket(nanoseconds / 1000, timestamp_micros)` for back-compat.
pub fn time_bucket_calendar(interval: Interval, timestamp_micros: i64) -> i64 {
    // Pure sub-day bucket: reuse the simple integer-division path.
    if interval.months == 0 && interval.days == 0 {
        let interval_micros = interval.nanoseconds / 1_000;
        return time_bucket(interval_micros, timestamp_micros);
    }

    // Month-based bucket: align to calendar month boundary.
    if interval.months != 0 && interval.days == 0 && interval.nanoseconds == 0 {
        let seconds = timestamp_micros.div_euclid(1_000_000);
        let days_since_epoch = seconds.div_euclid(86_400);
        let (year, month, _day) = ymd_from_days(days_since_epoch as i32);

        // Months since epoch-month (year 0, month 1) to align multi-month buckets
        // consistently relative to year boundaries.
        let total_months = (year as i64) * 12 + (month as i64 - 1);
        let bucket_months = interval.months as i64;
        let aligned = total_months.div_euclid(bucket_months) * bucket_months;
        let new_year = aligned.div_euclid(12) as i32;
        let new_month = (aligned.rem_euclid(12) + 1) as u32;
        let days = days_from_ymd(new_year, new_month, 1) as i64;
        return days * 86_400_000_000;
    }

    // Day-based bucket (no months, no sub-day component): align to day boundary.
    if interval.months == 0 && interval.days != 0 && interval.nanoseconds == 0 {
        let seconds = timestamp_micros.div_euclid(1_000_000);
        let days_since_epoch = seconds.div_euclid(86_400);
        let bucket_days = interval.days as i64;
        let aligned = days_since_epoch.div_euclid(bucket_days) * bucket_days;
        return aligned * 86_400_000_000;
    }

    // Mixed intervals (e.g. '1 month 5 days'): align by the most significant
    // component. Default to the month-aligned bucket if months are present.
    if interval.months != 0 {
        time_bucket_calendar(Interval::from_months(interval.months), timestamp_micros)
    } else {
        time_bucket_calendar(Interval::from_days(interval.days), timestamp_micros)
    }
}

/// Generates calendar-aligned bucket boundaries in [start, end) for a given interval.
pub fn time_bucket_gapfill_calendar(
    interval: Interval,
    start_micros: i64,
    end_micros: i64,
) -> Vec<i64> {
    if start_micros >= end_micros {
        return Vec::new();
    }
    let mut result = Vec::new();
    let mut current = time_bucket_calendar(interval, start_micros);
    if current < start_micros {
        // Advance to the next bucket boundary
        current = interval.add_to_timestamp_micros(current);
    }
    while current < end_micros {
        result.push(current);
        let next = interval.add_to_timestamp_micros(current);
        if next <= current {
            // Guard against pathological intervals (zero or negative)
            break;
        }
        current = next;
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_time_bucket() {
        // 1-hour buckets: 1:23:45 -> 1:00:00
        let hour = 3600 * 1_000_000;
        let ts = 1 * hour + 23 * 60 * 1_000_000 + 45 * 1_000_000;
        assert_eq!(time_bucket(hour, ts), hour);
    }

    #[test]
    fn test_time_bucket_exact() {
        let hour = 3600 * 1_000_000;
        assert_eq!(time_bucket(hour, 2 * hour), 2 * hour);
    }

    #[test]
    fn test_time_bucket_negative() {
        let hour = 3600 * 1_000_000;
        // -1 second should bucket to -1 hour (Euclidean division)
        assert_eq!(time_bucket(hour, -1_000_000), -hour);
    }

    #[test]
    fn test_gapfill() {
        let hour = 3600 * 1_000_000;
        let buckets = time_bucket_gapfill(hour, 0, 3 * hour);
        assert_eq!(buckets, vec![0, hour, 2 * hour]);
    }

    #[test]
    fn test_locf() {
        let values = vec![Some(1.0), None, Some(3.0), None, None];
        let result = locf(&values);
        assert_eq!(result, vec![1.0, 1.0, 3.0, 3.0, 3.0]);
    }

    #[test]
    fn test_locf_leading_null() {
        let values = vec![None, Some(5.0)];
        let result = locf(&values);
        assert_eq!(result, vec![0.0, 5.0]);
    }

    #[test]
    fn test_interpolate_linear() {
        let ts = vec![0, 1, 2, 3, 4];
        let values = vec![Some(0.0), None, None, None, Some(4.0)];
        let result = interpolate(&ts, &values);
        assert!((result[1] - 1.0).abs() < 1e-10);
        assert!((result[2] - 2.0).abs() < 1e-10);
        assert!((result[3] - 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_interpolate_edges() {
        let ts = vec![0, 1, 2];
        let values = vec![None, Some(5.0), None];
        let result = interpolate(&ts, &values);
        // Leading and trailing nulls take nearest known value
        assert_eq!(result[0], 5.0);
        assert_eq!(result[2], 5.0);
    }

    #[test]
    fn test_ema() {
        let data = vec![1.0, 2.0, 3.0, 4.0];
        let result = ema(&data, 0.5);
        assert_eq!(result[0], 1.0);
        assert!((result[1] - 1.5).abs() < 1e-10);
        assert!((result[2] - 2.25).abs() < 1e-10);
    }

    #[test]
    fn test_rate() {
        let ts = vec![0i64, 1_000_000, 2_000_000]; // 0s, 1s, 2s
        let values = vec![0.0, 10.0, 25.0];
        let result = rate(&ts, &values);
        assert_eq!(result[0], None);
        assert_eq!(result[1], Some(10.0)); // 10 per second
        assert_eq!(result[2], Some(15.0)); // 15 per second
    }

    #[test]
    fn test_delta() {
        let values = vec![10.0, 15.0, 12.0, 20.0];
        let result = delta(&values);
        assert_eq!(result[0], None);
        assert_eq!(result[1], Some(5.0));
        assert_eq!(result[2], Some(-3.0));
        assert_eq!(result[3], Some(8.0));
    }

    #[test]
    fn test_derivative() {
        let ts = vec![0i64, 1_000_000, 2_000_000];
        let values = vec![0.0, 10.0, 20.0];
        let d = derivative(&ts, &values);
        assert_eq!(d[1], Some(10.0));
        assert_eq!(d[2], Some(10.0));
    }

    #[test]
    fn test_lttb_preserves_endpoints() {
        let ts: Vec<f64> = (0..100).map(|i| i as f64).collect();
        let values: Vec<f64> = (0..100).map(|i| (i as f64).sin()).collect();
        let sampled = lttb(&ts, &values, 10);
        assert_eq!(sampled[0], 0);
        assert_eq!(*sampled.last().unwrap(), 99);
    }

    #[test]
    fn test_lttb_below_threshold() {
        let ts = vec![0.0, 1.0, 2.0];
        let values = vec![1.0, 2.0, 3.0];
        let sampled = lttb(&ts, &values, 10);
        // threshold > n -> return all indices
        assert_eq!(sampled, vec![0, 1, 2]);
    }

    #[test]
    fn test_lttb_sample_count() {
        let ts: Vec<f64> = (0..100).map(|i| i as f64).collect();
        let values: Vec<f64> = (0..100).map(|i| i as f64).collect();
        let sampled = lttb(&ts, &values, 20);
        assert_eq!(sampled.len(), 20);
    }

    #[test]
    fn test_ema_empty() {
        let result = ema(&[], 0.5);
        assert!(result.is_empty());
    }

    #[test]
    fn test_rate_mismatched() {
        let ts = vec![0i64, 1_000_000];
        let values = vec![0.0];
        let result = rate(&ts, &values);
        assert!(result.is_empty());
    }

    // ----- Calendar time bucket tests -----

    fn ymd_micros(y: i32, m: u32, d: u32) -> i64 {
        (zyron_common::days_from_ymd(y, m, d) as i64) * 86_400_000_000
    }

    #[test]
    fn test_time_bucket_calendar_one_month() {
        // 2024-03-15 should bucket to 2024-03-01 with INTERVAL '1 month'
        let ts = ymd_micros(2024, 3, 15) + 12 * 3_600_000_000; // noon
        let bucket = time_bucket_calendar(Interval::from_months(1), ts);
        assert_eq!(bucket, ymd_micros(2024, 3, 1));
    }

    #[test]
    fn test_time_bucket_calendar_one_year() {
        // 2024-07-15 should bucket to 2024-01-01 with INTERVAL '1 year'
        let ts = ymd_micros(2024, 7, 15);
        let bucket = time_bucket_calendar(Interval::from_months(12), ts);
        assert_eq!(bucket, ymd_micros(2024, 1, 1));
    }

    #[test]
    fn test_time_bucket_calendar_quarter() {
        // 2024-05-15 should bucket to 2024-04-01 with INTERVAL '1 quarter' (3 months)
        let ts = ymd_micros(2024, 5, 15);
        let bucket = time_bucket_calendar(Interval::from_months(3), ts);
        assert_eq!(bucket, ymd_micros(2024, 4, 1));
    }

    #[test]
    fn test_time_bucket_calendar_february_leap() {
        // 2024-02-29 (leap) should bucket to 2024-02-01 with INTERVAL '1 month'
        let ts = ymd_micros(2024, 2, 29);
        let bucket = time_bucket_calendar(Interval::from_months(1), ts);
        assert_eq!(bucket, ymd_micros(2024, 2, 1));
    }

    #[test]
    fn test_time_bucket_calendar_day_fallback() {
        // INTERVAL '7 days' = weekly bucket
        let week_iv = Interval::from_days(7);
        // 2024-01-08 is 7 days after epoch-day 19730 (or whatever); verify alignment.
        let ts = ymd_micros(2024, 1, 15);
        let bucket = time_bucket_calendar(week_iv, ts);
        // Bucket should be a multiple of 7 days
        let bucket_days = bucket / 86_400_000_000;
        assert_eq!(bucket_days % 7, 0);
        assert!(bucket <= ts);
    }

    #[test]
    fn test_time_bucket_calendar_sub_day_fallback() {
        // Sub-day intervals use the old integer-division path.
        let one_hour_ns = Interval::from_nanoseconds(3_600_000_000_000);
        let ts = 2 * 3_600_000_000 + 30 * 60_000_000;
        let bucket = time_bucket_calendar(one_hour_ns, ts);
        // Aligned to 2-hour mark (7200 seconds = 7_200_000_000 micros)
        assert_eq!(bucket, 2 * 3_600_000_000);
    }

    #[test]
    fn test_time_bucket_gapfill_calendar_months() {
        let start = ymd_micros(2024, 1, 15);
        let end = ymd_micros(2024, 5, 1);
        let buckets = time_bucket_gapfill_calendar(Interval::from_months(1), start, end);
        // Expected: 2024-02-01, 2024-03-01, 2024-04-01
        assert_eq!(buckets.len(), 3);
        assert_eq!(buckets[0], ymd_micros(2024, 2, 1));
        assert_eq!(buckets[1], ymd_micros(2024, 3, 1));
        assert_eq!(buckets[2], ymd_micros(2024, 4, 1));
    }
}
