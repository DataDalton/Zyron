//! Rate limiting primitives: token bucket and sliding window.
//!
//! These are pure-math functions; storage and atomicity are the caller's responsibility.

/// Token bucket state: (current_tokens, capacity, refill_rate_per_sec, last_refill_micros).
pub type TokenBucket = (f64, f64, f64, i64);

/// Creates a new token bucket with the given capacity and refill rate.
pub fn token_bucket_create(capacity: f64, refill_rate: f64) -> TokenBucket {
    (capacity, capacity, refill_rate, 0)
}

/// Attempts to consume `tokens` from the bucket at the given time.
/// Returns (allowed, updated_bucket).
pub fn token_bucket_consume(
    bucket: TokenBucket,
    tokens: f64,
    now_micros: i64,
) -> (bool, TokenBucket) {
    let (current, capacity, rate, last_refill) = bucket;

    // Refill tokens based on elapsed time
    let elapsed_micros = (now_micros - last_refill).max(0);
    let elapsed_secs = elapsed_micros as f64 / 1_000_000.0;
    let refilled = (current + elapsed_secs * rate).min(capacity);

    if refilled >= tokens {
        let new_current = refilled - tokens;
        (true, (new_current, capacity, rate, now_micros))
    } else {
        (false, (refilled, capacity, rate, now_micros))
    }
}

/// Returns the number of tokens currently available in the bucket
/// (without mutating state).
pub fn token_bucket_available(bucket: TokenBucket, now_micros: i64) -> f64 {
    let (current, capacity, rate, last_refill) = bucket;
    let elapsed_micros = (now_micros - last_refill).max(0);
    let elapsed_secs = elapsed_micros as f64 / 1_000_000.0;
    (current + elapsed_secs * rate).min(capacity)
}

/// Counts events that occurred within `window_micros` before `now_micros`.
/// Requires `timestamps` to be sorted in ascending order.
pub fn sliding_window_count(timestamps: &[i64], window_micros: i64, now_micros: i64) -> usize {
    let cutoff = now_micros - window_micros;
    // Binary search for the first timestamp >= cutoff
    match timestamps.binary_search(&cutoff) {
        Ok(idx) => timestamps.len() - idx,
        Err(idx) => timestamps.len() - idx,
    }
}

/// Returns true if the event count within the window exceeds max_count.
pub fn sliding_window_check(
    timestamps: &[i64],
    window_micros: i64,
    max_count: usize,
    now_micros: i64,
) -> bool {
    sliding_window_count(timestamps, window_micros, now_micros) < max_count
}

/// Returns the window bucket identifier for a given timestamp.
/// Useful for grouping events into fixed-size time windows.
pub fn fixed_window_count(timestamp_micros: i64, window_micros: i64) -> i64 {
    if window_micros <= 0 {
        return 0;
    }
    timestamp_micros.div_euclid(window_micros)
}

/// Leaky bucket: similar to token bucket but drains at a constant rate.
/// State: (current_level, capacity, leak_rate_per_sec, last_update_micros).
pub type LeakyBucket = (f64, f64, f64, i64);

/// Creates a new leaky bucket.
pub fn leaky_bucket_create(capacity: f64, leak_rate: f64) -> LeakyBucket {
    (0.0, capacity, leak_rate, 0)
}

/// Attempts to add `amount` to the leaky bucket.
/// Returns (accepted, updated_bucket).
pub fn leaky_bucket_add(bucket: LeakyBucket, amount: f64, now_micros: i64) -> (bool, LeakyBucket) {
    let (current, capacity, rate, last_update) = bucket;

    // Leak based on elapsed time
    let elapsed_micros = (now_micros - last_update).max(0);
    let elapsed_secs = elapsed_micros as f64 / 1_000_000.0;
    let leaked = (current - elapsed_secs * rate).max(0.0);

    if leaked + amount <= capacity {
        (true, (leaked + amount, capacity, rate, now_micros))
    } else {
        (false, (leaked, capacity, rate, now_micros))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_token_bucket_create() {
        let b = token_bucket_create(10.0, 1.0);
        assert_eq!(b.0, 10.0); // full capacity
        assert_eq!(b.1, 10.0);
        assert_eq!(b.2, 1.0);
    }

    #[test]
    fn test_token_bucket_consume_allowed() {
        let b = token_bucket_create(10.0, 1.0);
        let (allowed, b2) = token_bucket_consume(b, 5.0, 1_000_000);
        assert!(allowed);
        assert_eq!(b2.0, 5.0);
    }

    #[test]
    fn test_token_bucket_consume_rejected() {
        let b = token_bucket_create(10.0, 1.0);
        let (allowed, b2) = token_bucket_consume(b, 20.0, 1_000_000);
        assert!(!allowed);
        // Bucket should still refill though
        assert_eq!(b2.0, 10.0);
    }

    #[test]
    fn test_token_bucket_refill() {
        let b = token_bucket_create(10.0, 1.0);
        // Consume all
        let (_, b) = token_bucket_consume(b, 10.0, 0);
        assert_eq!(b.0, 0.0);
        // Wait 5 seconds - should have 5 tokens
        let (allowed, b2) = token_bucket_consume(b, 5.0, 5_000_000);
        assert!(allowed);
        assert_eq!(b2.0, 0.0); // consumed all 5 refilled
    }

    #[test]
    fn test_token_bucket_cap_at_capacity() {
        let b = token_bucket_create(10.0, 100.0);
        // Wait a long time
        let avail = token_bucket_available(b, 1_000_000_000);
        assert_eq!(avail, 10.0); // Capped at capacity
    }

    #[test]
    fn test_token_bucket_available_no_mutation() {
        let b = token_bucket_create(10.0, 1.0);
        let avail_1 = token_bucket_available(b, 1_000_000);
        let avail_2 = token_bucket_available(b, 1_000_000);
        assert_eq!(avail_1, avail_2);
    }

    #[test]
    fn test_sliding_window_count() {
        let timestamps = vec![0i64, 1_000_000, 2_000_000, 3_000_000, 4_000_000];
        // Count events in last 2 seconds as of t=4s
        let count = sliding_window_count(&timestamps, 2_000_000, 4_000_000);
        assert_eq!(count, 3); // events at 2s, 3s, 4s
    }

    #[test]
    fn test_sliding_window_count_empty() {
        let timestamps: Vec<i64> = Vec::new();
        let count = sliding_window_count(&timestamps, 1_000_000, 5_000_000);
        assert_eq!(count, 0);
    }

    #[test]
    fn test_sliding_window_check_allowed() {
        let timestamps = vec![0i64, 1_000_000];
        assert!(sliding_window_check(&timestamps, 5_000_000, 10, 5_000_000));
    }

    #[test]
    fn test_sliding_window_check_rejected() {
        let timestamps = vec![0i64, 1_000_000, 2_000_000, 3_000_000, 4_000_000];
        // 5 events in 5s, max 3 -> rejected
        assert!(!sliding_window_check(&timestamps, 5_000_000, 3, 4_000_000));
    }

    #[test]
    fn test_fixed_window_count() {
        // 60-second windows
        let w = 60_000_000;
        assert_eq!(fixed_window_count(0, w), 0);
        assert_eq!(fixed_window_count(59_999_999, w), 0);
        assert_eq!(fixed_window_count(60_000_000, w), 1);
        assert_eq!(fixed_window_count(120_000_000, w), 2);
    }

    #[test]
    fn test_fixed_window_negative_time() {
        // div_euclid handles negative correctly
        let w = 60_000_000;
        assert_eq!(fixed_window_count(-60_000_000, w), -1);
    }

    #[test]
    fn test_leaky_bucket_add() {
        let b = leaky_bucket_create(10.0, 1.0);
        let (ok, b2) = leaky_bucket_add(b, 5.0, 0);
        assert!(ok);
        assert_eq!(b2.0, 5.0);
    }

    #[test]
    fn test_leaky_bucket_rejected_at_capacity() {
        let b = leaky_bucket_create(10.0, 1.0);
        let (_, b) = leaky_bucket_add(b, 10.0, 0);
        let (ok, _) = leaky_bucket_add(b, 5.0, 0);
        assert!(!ok); // would exceed capacity
    }

    #[test]
    fn test_leaky_bucket_drains() {
        let b = leaky_bucket_create(10.0, 1.0);
        let (_, b) = leaky_bucket_add(b, 10.0, 0);
        // After 5 seconds, 5 units leaked
        let (ok, b2) = leaky_bucket_add(b, 5.0, 5_000_000);
        assert!(ok);
        assert_eq!(b2.0, 10.0); // 10 - 5 leaked + 5 added
    }
}
