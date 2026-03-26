//! HyperLogLog probabilistic distinct count estimator.
//!
//! Uses precision 14 (16384 registers) for a standard error of ~0.8%.
//! Suitable for cardinality estimation on large datasets without
//! requiring memory proportional to the number of distinct values.

/// Precision bits. 14 gives 16384 registers and ~0.8% standard error.
const HLL_PRECISION: u8 = 14;
const HLL_REGISTER_COUNT: usize = 1 << HLL_PRECISION;
/// Bias correction constant alpha_m for m = 16384.
const HLL_ALPHA: f64 = 0.7213 / (1.0 + 1.079 / HLL_REGISTER_COUNT as f64);

/// HyperLogLog distinct count estimator with <2% error.
/// Uses 16384 registers (16 KB) regardless of input size.
#[derive(Debug, Clone)]
pub struct HyperLogLog {
    registers: Vec<u8>,
    count: u64,
}

impl HyperLogLog {
    pub fn new() -> Self {
        Self {
            registers: vec![0; HLL_REGISTER_COUNT],
            count: 0,
        }
    }

    /// Inserts a pre-hashed 64-bit value into the estimator.
    pub fn insert(&mut self, hash: u64) {
        self.count += 1;
        // Upper bits select the register, lower bits determine the rank
        let index = (hash >> (64 - HLL_PRECISION)) as usize;
        let remaining = (hash << HLL_PRECISION) | (1 << (HLL_PRECISION - 1));
        let rank = (remaining.leading_zeros() + 1) as u8;
        if rank > self.registers[index] {
            self.registers[index] = rank;
        }
    }

    /// Returns the estimated number of distinct values.
    pub fn cardinality(&self) -> u64 {
        let m = HLL_REGISTER_COUNT as f64;

        // Harmonic mean of 2^(-register)
        let mut sum = 0.0_f64;
        let mut zeros = 0u32;
        for &reg in &self.registers {
            sum += 2.0_f64.powi(-(reg as i32));
            if reg == 0 {
                zeros += 1;
            }
        }

        let raw_estimate = HLL_ALPHA * m * m / sum;

        // Small range correction: use linear counting when many registers are zero
        if raw_estimate <= 2.5 * m && zeros > 0 {
            let linear = m * (m / zeros as f64).ln();
            return linear as u64;
        }

        // No large range correction needed for 64-bit hashes.
        // The 32-bit correction formula is only applicable to 32-bit hash functions.

        raw_estimate as u64
    }

    /// Merges another HyperLogLog into this one (union operation).
    /// Both must use the same precision.
    pub fn merge(&mut self, other: &HyperLogLog) {
        for i in 0..HLL_REGISTER_COUNT {
            if other.registers[i] > self.registers[i] {
                self.registers[i] = other.registers[i];
            }
        }
        self.count += other.count;
    }

    /// Returns the number of values inserted (not distinct count).
    pub fn total_count(&self) -> u64 {
        self.count
    }
}

impl Default for HyperLogLog {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Hash helper
// ---------------------------------------------------------------------------

/// Simple 64-bit hash mixing function for use when callers do not have
/// a pre-computed hash. Uses the finalizer from SplitMix64.
pub fn hash_bytes(data: &[u8]) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for chunk in data.chunks(8) {
        let mut buf = [0u8; 8];
        buf[..chunk.len()].copy_from_slice(chunk);
        h ^= u64::from_le_bytes(buf);
        h = h.wrapping_mul(0x517c_c1b7_2722_0a95);
        h = (h >> 32) | (h << 32);
    }
    // SplitMix64 finalizer
    h ^= h >> 30;
    h = h.wrapping_mul(0xbf58_476d_1ce4_e5b9);
    h ^= h >> 27;
    h = h.wrapping_mul(0x94d0_49bb_1331_11eb);
    h ^= h >> 31;
    h
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_empty_hll() {
        let hll = HyperLogLog::new();
        assert_eq!(hll.cardinality(), 0);
        assert_eq!(hll.total_count(), 0);
    }

    #[test]
    fn test_single_value() {
        let mut hll = HyperLogLog::new();
        hll.insert(hash_bytes(b"hello"));
        assert!(hll.cardinality() >= 1);
        assert_eq!(hll.total_count(), 1);
    }

    #[test]
    fn test_hll_accuracy_1000() {
        let mut hll = HyperLogLog::new();
        let n = 1000u64;
        for i in 0..n {
            hll.insert(hash_bytes(&i.to_le_bytes()));
        }
        let estimate = hll.cardinality();
        let error = (estimate as f64 - n as f64).abs() / n as f64;
        // HLL with precision 14 should be within 5% for 1000 values
        assert!(error < 0.05, "error {:.2}% for n={}", error * 100.0, n);
    }

    #[test]
    fn test_hll_accuracy_100000() {
        let mut hll = HyperLogLog::new();
        let n = 100_000u64;
        for i in 0..n {
            hll.insert(hash_bytes(&i.to_le_bytes()));
        }
        let estimate = hll.cardinality();
        let error = (estimate as f64 - n as f64).abs() / n as f64;
        // Should be within 2% for large cardinalities
        assert!(error < 0.02, "error {:.2}% for n={}", error * 100.0, n);
    }

    #[test]
    fn test_hll_duplicates() {
        let mut hll = HyperLogLog::new();
        // Insert same 100 values 100 times each
        for _ in 0..100 {
            for i in 0u64..100 {
                hll.insert(hash_bytes(&i.to_le_bytes()));
            }
        }
        let estimate = hll.cardinality();
        let error = (estimate as f64 - 100.0).abs() / 100.0;
        assert!(
            error < 0.10,
            "error {:.2}% for 100 distinct values",
            error * 100.0
        );
    }

    #[test]
    fn test_hll_merge() {
        let mut hll1 = HyperLogLog::new();
        let mut hll2 = HyperLogLog::new();

        for i in 0u64..5000 {
            hll1.insert(hash_bytes(&i.to_le_bytes()));
        }
        for i in 5000u64..10000 {
            hll2.insert(hash_bytes(&i.to_le_bytes()));
        }

        hll1.merge(&hll2);
        let estimate = hll1.cardinality();
        let error = (estimate as f64 - 10000.0).abs() / 10000.0;
        assert!(error < 0.05, "error {:.2}% after merge", error * 100.0);
    }

    #[test]
    fn test_hash_bytes_deterministic() {
        let h1 = hash_bytes(b"test");
        let h2 = hash_bytes(b"test");
        assert_eq!(h1, h2);
        let h3 = hash_bytes(b"test2");
        assert_ne!(h1, h3);
    }
}
