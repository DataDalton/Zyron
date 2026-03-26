//! Equi-height histogram and most-common-values (MCV) for selectivity estimation.
//!
//! EquiHeightHistogram divides sorted sample data into buckets with approximately
//! equal row counts. Range and equality selectivity queries use binary search
//! on bucket boundaries with linear interpolation within partial buckets.
//!
//! ReservoirSampler collects a bounded-memory sample from an arbitrarily large
//! input stream using Algorithm R.

/// Default reservoir sample size for histogram construction.
const RESERVOIR_SIZE: usize = 10_000;

/// Maximum number of most-common-value entries to track.
const MAX_MCV_ENTRIES: usize = 100;

// ---------------------------------------------------------------------------
// Equi-height histogram
// ---------------------------------------------------------------------------

/// Equi-height histogram where each bucket contains approximately the same
/// number of rows. Bucket boundaries enable range and equality selectivity
/// estimation via binary search and interpolation.
#[derive(Debug, Clone)]
pub struct EquiHeightHistogram {
    /// Bucket boundary values (num_buckets + 1 entries, sorted ascending).
    pub bounds: Vec<Vec<u8>>,
    /// Number of rows in each bucket.
    pub row_counts: Vec<u64>,
    /// Number of distinct values in each bucket.
    pub distinct_counts: Vec<u64>,
    /// Total rows represented by this histogram.
    pub total_rows: u64,
    /// True column minimum across all rows (not just the sample).
    /// Used for out-of-range interpolation when a query value falls
    /// below the sample's minimum bound.
    pub col_min: Option<Vec<u8>>,
    /// True column maximum across all rows (not just the sample).
    pub col_max: Option<Vec<u8>>,
}

impl EquiHeightHistogram {
    /// Builds an equi-height histogram from pre-sorted sample values.
    /// Returns None if the input is empty.
    pub fn build_from_sorted(sorted: &[Vec<u8>], num_buckets: u32) -> Option<Self> {
        if sorted.is_empty() {
            return None;
        }

        let n = sorted.len();
        let num_buckets = (num_buckets as usize).min(n).max(1);
        let rows_per_bucket = n / num_buckets;

        let mut bounds = Vec::with_capacity(num_buckets + 1);
        let mut row_counts = Vec::with_capacity(num_buckets);
        let mut distinct_counts = Vec::with_capacity(num_buckets);

        bounds.push(sorted[0].clone());

        for bucket_idx in 0..num_buckets {
            let start = bucket_idx * rows_per_bucket;
            let end = if bucket_idx == num_buckets - 1 {
                n
            } else {
                (bucket_idx + 1) * rows_per_bucket
            };

            let count = (end - start) as u64;
            row_counts.push(count);

            // Count distinct values in this bucket
            let mut distinct = 1u64;
            for i in (start + 1)..end {
                if sorted[i] != sorted[i - 1] {
                    distinct += 1;
                }
            }
            distinct_counts.push(distinct);

            // Upper bound of this bucket
            if end > 0 {
                bounds.push(sorted[end - 1].clone());
            }
        }

        Some(Self {
            bounds,
            row_counts,
            distinct_counts,
            total_rows: n as u64,
            col_min: None,
            col_max: None,
        })
    }

    /// Builds an equi-height histogram with column min/max for out-of-range estimation.
    /// The col_min/col_max represent the true column extent from the full table scan,
    /// while the sorted sample only covers a subset of that range.
    pub fn build_from_sorted_with_bounds(
        sorted: &[Vec<u8>],
        num_buckets: u32,
        col_min: Option<Vec<u8>>,
        col_max: Option<Vec<u8>>,
    ) -> Option<Self> {
        let mut hist = Self::build_from_sorted(sorted, num_buckets)?;
        hist.col_min = col_min;
        hist.col_max = col_max;
        Some(hist)
    }

    /// Returns the number of buckets in this histogram.
    pub fn num_buckets(&self) -> usize {
        self.row_counts.len()
    }

    /// Estimates selectivity for a range predicate [low, high].
    /// Pass None for unbounded sides (e.g., col > X uses low=Some(X), high=None).
    pub fn estimate_range_selectivity(&self, low: Option<&[u8]>, high: Option<&[u8]>) -> f64 {
        if self.total_rows == 0 || self.bounds.is_empty() {
            return 0.5;
        }

        let num_buckets = self.num_buckets();

        // Find the first bucket containing the low bound.
        // None from find_bucket means the value is below the histogram range,
        // so start from the first bucket.
        let start_bucket = if let Some(low_val) = low {
            match self.find_bucket(low_val) {
                Some(b) => b,
                None => 0,
            }
        } else {
            0
        };

        // Find the last bucket containing the high bound.
        // None from find_bucket means the value is below the histogram sample range.
        // If col_min/col_max are available, interpolate uniformly over the gap
        // between the true column minimum and the first sample bound.
        let end_bucket = if let Some(high_val) = high {
            match self.find_bucket(high_val) {
                Some(b) => b,
                None => {
                    // High bound is below the sample range. Check if it falls
                    // within the true column range for uniform interpolation.
                    if let (Some(col_min), Some(col_max)) = (&self.col_min, &self.col_max) {
                        if high_val >= col_min.as_slice() && col_max.as_slice() > col_min.as_slice()
                        {
                            let col_min_val = leading_bytes_as_u64(col_min);
                            let col_max_val = leading_bytes_as_u64(col_max);
                            let high_val_u64 = leading_bytes_as_u64(high_val);
                            let low_val_u64 = low.map(leading_bytes_as_u64).unwrap_or(col_min_val);
                            let range = col_max_val.saturating_sub(col_min_val) as f64;
                            if range > 0.0 {
                                let query_span = high_val_u64
                                    .saturating_sub(low_val_u64.max(col_min_val))
                                    as f64;
                                return (query_span / range).clamp(0.0, 1.0);
                            }
                        }
                    }
                    return 0.0;
                }
            }
        } else {
            num_buckets.saturating_sub(1)
        };

        if start_bucket > end_bucket {
            return 0.0;
        }

        // Sum rows in fully-contained buckets
        let mut matching_rows = 0.0_f64;

        for bucket in start_bucket..=end_bucket {
            if bucket >= num_buckets {
                break;
            }
            let bucket_rows = self.row_counts[bucket] as f64;
            let is_start = bucket == start_bucket && low.is_some();
            let is_end = bucket == end_bucket && high.is_some();

            if is_start && is_end {
                // Both bounds fall in the same bucket: fraction_high - fraction_low
                let frac_low = self.interpolate_within_bucket(bucket, low);
                let frac_high = self.interpolate_within_bucket(bucket, high);
                matching_rows += bucket_rows * (frac_high - frac_low).max(0.0);
            } else if is_start {
                // Partial bucket: estimate fraction above low bound
                let fraction = self.interpolate_within_bucket(bucket, low);
                matching_rows += bucket_rows * (1.0 - fraction);
            } else if is_end {
                // Partial bucket: estimate fraction below high bound
                let fraction = self.interpolate_within_bucket(bucket, high);
                matching_rows += bucket_rows * fraction;
            } else {
                // Fully contained bucket
                matching_rows += bucket_rows;
            }
        }

        (matching_rows / self.total_rows as f64).clamp(0.0, 1.0)
    }

    /// Estimates selectivity for an equality predicate (col = value).
    /// Returns 1/distinct_in_bucket for the bucket containing the value.
    pub fn estimate_equality_selectivity(&self, value: &[u8]) -> f64 {
        if self.total_rows == 0 {
            return 0.0;
        }

        if let Some(bucket) = self.find_bucket(value) {
            if bucket < self.distinct_counts.len() && self.distinct_counts[bucket] > 0 {
                let bucket_fraction = self.row_counts[bucket] as f64 / self.total_rows as f64;
                return bucket_fraction / self.distinct_counts[bucket] as f64;
            }
        }

        // Value outside histogram range
        1.0 / self.total_rows as f64
    }

    /// Estimates cardinality (number of matching rows) for a range predicate.
    pub fn estimate_cardinality(&self, low: Option<&[u8]>, high: Option<&[u8]>) -> u64 {
        let selectivity = self.estimate_range_selectivity(low, high);
        (selectivity * self.total_rows as f64) as u64
    }

    /// Finds which bucket a value falls into using binary search on bounds.
    fn find_bucket(&self, value: &[u8]) -> Option<usize> {
        if self.bounds.len() < 2 {
            return None;
        }

        // Binary search: find the first bound > value
        let pos = self
            .bounds
            .partition_point(|bound| bound.as_slice() <= value);

        if pos == 0 {
            // Value is below the minimum bound, outside histogram range
            return None;
        }

        // pos-1 is the bucket index (bucket i spans bounds[i]..=bounds[i+1])
        let bucket = (pos - 1).min(self.num_buckets().saturating_sub(1));
        Some(bucket)
    }

    /// Estimates what fraction of a bucket's range is below the given value.
    /// Returns 0.0 at the lower bound, 1.0 at the upper bound.
    fn interpolate_within_bucket(&self, bucket: usize, value: Option<&[u8]>) -> f64 {
        let value = match value {
            Some(v) => v,
            None => return 0.5,
        };

        if bucket + 1 >= self.bounds.len() {
            return 0.5;
        }

        let lower = &self.bounds[bucket];
        let upper = &self.bounds[bucket + 1];

        if lower == upper {
            return 0.5;
        }

        // Byte-level linear interpolation using leading bytes
        let lower_val = leading_bytes_as_u64(lower);
        let upper_val = leading_bytes_as_u64(upper);
        let target_val = leading_bytes_as_u64(value);

        if upper_val <= lower_val {
            return 0.5;
        }

        let fraction =
            (target_val.saturating_sub(lower_val)) as f64 / (upper_val - lower_val) as f64;
        fraction.clamp(0.0, 1.0)
    }
}

/// Interprets the first 8 bytes of a byte slice as a big-endian u64 for comparison.
fn leading_bytes_as_u64(data: &[u8]) -> u64 {
    let mut buf = [0u8; 8];
    let len = data.len().min(8);
    buf[..len].copy_from_slice(&data[..len]);
    u64::from_be_bytes(buf)
}

// ---------------------------------------------------------------------------
// Most common values
// ---------------------------------------------------------------------------

/// Tracks the most frequently occurring values and their frequencies.
/// Used for equality selectivity estimation: if a value is in the MCV list,
/// its frequency is used directly rather than the 1/NDV approximation.
#[derive(Debug, Clone)]
pub struct MostCommonValues {
    /// Top-N values sorted by frequency (descending).
    pub values: Vec<Vec<u8>>,
    /// Frequency of each value (fraction of total rows, 0.0 to 1.0).
    pub frequencies: Vec<f64>,
}

impl MostCommonValues {
    /// Builds an MCV list from pre-sorted values.
    /// Counts runs of identical values and keeps the top max_entries.
    pub fn build(sorted: &[Vec<u8>], total: u64, max_entries: usize) -> Self {
        if sorted.is_empty() || total == 0 {
            return Self {
                values: Vec::new(),
                frequencies: Vec::new(),
            };
        }

        let max_entries = max_entries.min(MAX_MCV_ENTRIES);

        // Count runs of identical values
        let mut runs: Vec<(Vec<u8>, u64)> = Vec::new();
        let mut current_value = &sorted[0];
        let mut current_count = 1u64;

        for val in sorted.iter().skip(1) {
            if val == current_value {
                current_count += 1;
            } else {
                runs.push((current_value.clone(), current_count));
                current_value = val;
                current_count = 1;
            }
        }
        runs.push((current_value.clone(), current_count));

        // Sort by count descending, take top N
        runs.sort_by(|a, b| b.1.cmp(&a.1));
        runs.truncate(max_entries);

        let total_f64 = total as f64;
        let (values, frequencies): (Vec<_>, Vec<_>) = runs
            .into_iter()
            .map(|(v, c)| (v, c as f64 / total_f64))
            .unzip();

        Self {
            values,
            frequencies,
        }
    }

    /// Returns the frequency of a value if it appears in the MCV list.
    pub fn frequency_of(&self, value: &[u8]) -> Option<f64> {
        for (i, v) in self.values.iter().enumerate() {
            if v.as_slice() == value {
                return Some(self.frequencies[i]);
            }
        }
        None
    }

    /// Returns the total frequency mass covered by MCV entries.
    pub fn total_frequency(&self) -> f64 {
        self.frequencies.iter().sum()
    }

    /// Returns the number of entries in the MCV list.
    pub fn len(&self) -> usize {
        self.values.len()
    }

    /// Returns true if the MCV list is empty.
    pub fn is_empty(&self) -> bool {
        self.values.is_empty()
    }
}

// ---------------------------------------------------------------------------
// Reservoir sampler
// ---------------------------------------------------------------------------

/// Bounded-memory reservoir sampler using Algorithm R.
/// Collects a uniform random sample of at most `capacity` values
/// from an arbitrarily large input stream.
#[derive(Debug, Clone)]
pub struct ReservoirSampler {
    reservoir: Vec<Vec<u8>>,
    capacity: usize,
    seen: u64,
    /// Simple PRNG state (xorshift64)
    rng_state: u64,
    /// True column minimum across all inserted values.
    col_min: Option<Vec<u8>>,
    /// True column maximum across all inserted values.
    col_max: Option<Vec<u8>>,
}

impl ReservoirSampler {
    /// Creates a new reservoir sampler with the given capacity.
    pub fn new(capacity: usize) -> Self {
        Self {
            reservoir: Vec::with_capacity(capacity.min(RESERVOIR_SIZE)),
            capacity: capacity.min(RESERVOIR_SIZE),
            seen: 0,
            rng_state: 0x1234_5678_9abc_def0,
            col_min: None,
            col_max: None,
        }
    }

    /// Creates a sampler with the default capacity (10,000 values).
    pub fn with_default_capacity() -> Self {
        Self::new(RESERVOIR_SIZE)
    }

    /// Inserts a value into the reservoir.
    /// For the first `capacity` values, all are kept.
    /// After that, each new value replaces a random existing value
    /// with probability capacity/seen.
    pub fn insert(&mut self, value: Vec<u8>) {
        self.seen += 1;

        // Track true column min/max across all values (two byte comparisons per row)
        match &self.col_min {
            None => self.col_min = Some(value.clone()),
            Some(current) if value < *current => self.col_min = Some(value.clone()),
            _ => {}
        }
        match &self.col_max {
            None => self.col_max = Some(value.clone()),
            Some(current) if value > *current => self.col_max = Some(value.clone()),
            _ => {}
        }

        if self.reservoir.len() < self.capacity {
            self.reservoir.push(value);
        } else {
            // Algorithm R: replace with probability capacity/seen
            let idx = self.next_random() % self.seen;
            if (idx as usize) < self.capacity {
                self.reservoir[idx as usize] = value;
            }
        }
    }

    /// Sorts the reservoir and returns the sorted sample.
    pub fn into_sorted(mut self) -> Vec<Vec<u8>> {
        self.reservoir.sort();
        self.reservoir
    }

    /// Sorts the reservoir and returns it along with the true column min/max.
    pub fn into_sorted_with_bounds(mut self) -> (Vec<Vec<u8>>, Option<Vec<u8>>, Option<Vec<u8>>) {
        self.reservoir.sort();
        (self.reservoir, self.col_min, self.col_max)
    }

    /// Returns the true column minimum across all inserted values.
    pub fn col_min(&self) -> Option<&[u8]> {
        self.col_min.as_deref()
    }

    /// Returns the true column maximum across all inserted values.
    pub fn col_max(&self) -> Option<&[u8]> {
        self.col_max.as_deref()
    }

    /// Returns the number of values seen so far (including those not kept).
    pub fn total_seen(&self) -> u64 {
        self.seen
    }

    /// Returns the current number of values in the reservoir.
    pub fn len(&self) -> usize {
        self.reservoir.len()
    }

    /// Returns true if no values have been inserted.
    pub fn is_empty(&self) -> bool {
        self.reservoir.is_empty()
    }

    /// Xorshift64 PRNG for reservoir replacement decisions.
    fn next_random(&mut self) -> u64 {
        let mut x = self.rng_state;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.rng_state = x;
        x
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn make_sorted_values(n: usize) -> Vec<Vec<u8>> {
        (0..n).map(|i| (i as u64).to_be_bytes().to_vec()).collect()
    }

    #[test]
    fn test_histogram_build() {
        let sorted = make_sorted_values(100);
        let hist = EquiHeightHistogram::build_from_sorted(&sorted, 10);
        assert!(hist.is_some());
        let hist = hist.unwrap();
        assert_eq!(hist.num_buckets(), 10);
        assert_eq!(hist.bounds.len(), 11);
        let total: u64 = hist.row_counts.iter().sum();
        assert_eq!(total, 100);
    }

    #[test]
    fn test_histogram_empty() {
        let hist = EquiHeightHistogram::build_from_sorted(&[], 10);
        assert!(hist.is_none());
    }

    #[test]
    fn test_histogram_single_value() {
        let sorted = vec![vec![42u8]];
        let hist = EquiHeightHistogram::build_from_sorted(&sorted, 10);
        assert!(hist.is_some());
        let hist = hist.unwrap();
        assert_eq!(hist.num_buckets(), 1);
        assert_eq!(hist.total_rows, 1);
    }

    #[test]
    fn test_histogram_range_selectivity_full() {
        let sorted = make_sorted_values(1000);
        let hist = EquiHeightHistogram::build_from_sorted(&sorted, 10).unwrap();
        // Full range should return ~1.0
        let sel = hist.estimate_range_selectivity(None, None);
        assert!((sel - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_histogram_range_selectivity_partial() {
        let sorted = make_sorted_values(1000);
        let hist = EquiHeightHistogram::build_from_sorted(&sorted, 10).unwrap();
        // Lower half
        let mid = 500u64.to_be_bytes().to_vec();
        let sel = hist.estimate_range_selectivity(None, Some(&mid));
        assert!(sel > 0.3 && sel < 0.7, "selectivity {}", sel);
    }

    #[test]
    fn test_histogram_equality_selectivity() {
        let sorted = make_sorted_values(1000);
        let hist = EquiHeightHistogram::build_from_sorted(&sorted, 10).unwrap();
        let val = 500u64.to_be_bytes().to_vec();
        let sel = hist.estimate_equality_selectivity(&val);
        // With 1000 distinct values in 10 buckets (~100 distinct per bucket),
        // equality selectivity should be around 0.001
        assert!(sel > 0.0 && sel < 0.1, "equality selectivity {}", sel);
    }

    #[test]
    fn test_histogram_cardinality() {
        let sorted = make_sorted_values(1000);
        let hist = EquiHeightHistogram::build_from_sorted(&sorted, 10).unwrap();
        let card = hist.estimate_cardinality(None, None);
        assert_eq!(card, 1000);
    }

    #[test]
    fn test_mcv_build() {
        let mut sorted = Vec::new();
        // Value 0 appears 50 times, value 1 appears 30 times, rest appear once
        for _ in 0..50 {
            sorted.push(vec![0u8]);
        }
        for _ in 0..30 {
            sorted.push(vec![1u8]);
        }
        for i in 2u8..22 {
            sorted.push(vec![i]);
        }
        sorted.sort();

        let mcv = MostCommonValues::build(&sorted, 100, 10);
        assert!(!mcv.is_empty());
        // Value 0 should be in MCV with frequency 0.50
        let freq0 = mcv.frequency_of(&[0u8]);
        assert!(freq0.is_some());
        assert!((freq0.unwrap() - 0.50).abs() < 0.01);
    }

    #[test]
    fn test_mcv_empty() {
        let mcv = MostCommonValues::build(&[], 0, 10);
        assert!(mcv.is_empty());
        assert_eq!(mcv.total_frequency(), 0.0);
    }

    #[test]
    fn test_mcv_frequency_not_found() {
        let sorted = vec![vec![1u8], vec![2u8], vec![3u8]];
        let mcv = MostCommonValues::build(&sorted, 3, 10);
        assert!(mcv.frequency_of(&[99u8]).is_none());
    }

    #[test]
    fn test_reservoir_sampler_small_input() {
        let mut sampler = ReservoirSampler::new(100);
        for i in 0u8..10 {
            sampler.insert(vec![i]);
        }
        assert_eq!(sampler.len(), 10);
        assert_eq!(sampler.total_seen(), 10);
        let sorted = sampler.into_sorted();
        assert_eq!(sorted.len(), 10);
        // Should be sorted
        for i in 1..sorted.len() {
            assert!(sorted[i] >= sorted[i - 1]);
        }
    }

    #[test]
    fn test_reservoir_sampler_large_input() {
        let mut sampler = ReservoirSampler::new(100);
        for i in 0u64..10_000 {
            sampler.insert(i.to_le_bytes().to_vec());
        }
        assert_eq!(sampler.len(), 100);
        assert_eq!(sampler.total_seen(), 10_000);
    }

    #[test]
    fn test_reservoir_sampler_default_capacity() {
        let sampler = ReservoirSampler::with_default_capacity();
        assert_eq!(sampler.len(), 0);
        assert!(sampler.is_empty());
    }
}
