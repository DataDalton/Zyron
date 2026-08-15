//! Distinct-value counting for the lake writer.
//!
//! The writer materializes every column anyway, so it could count distinct
//! values exactly with a hash set. It does not, because an exact set over a
//! 256 MB file's column is hundreds of megabytes of transient memory, and
//! the number is only ever compared against thresholds. A HyperLogLog holds
//! 1 KiB regardless of how many rows it sees and lands inside a couple of
//! percent, which is well inside the distance between the thresholds that
//! read it.
//!
//! The sketch is dropped once the writer has its estimate. Only the u64
//! estimate reaches the manifest, so a table with 100k files pays 8 bytes
//! per column per file rather than 1 KiB.

use zyron_common::hash64;

/// Register count, 2^PRECISION. 1024 registers of one byte is the 1 KiB
/// the sketch is budgeted at, and gives roughly 3% relative error
const PRECISION: u32 = 10;
const REGISTERS: usize = 1 << PRECISION;

/// Bias constant for this register count
const ALPHA: f64 = 0.7213 / (1.0 + 1.079 / REGISTERS as f64);

/// Fixed-size distinct-value sketch over hashed cell bytes.
pub struct DistinctSketch {
    /// Largest leading-zero run seen for each register, one byte each
    registers: [u8; REGISTERS],
}

impl Default for DistinctSketch {
    fn default() -> Self {
        Self::new()
    }
}

impl DistinctSketch {
    pub fn new() -> Self {
        Self {
            registers: [0u8; REGISTERS],
        }
    }

    /// Adds one value, identified by its bytes.
    #[inline]
    pub fn insert(&mut self, value: &[u8]) {
        self.insert_hash(hash64(value));
    }

    #[inline]
    fn insert_hash(&mut self, hash: u64) {
        let index = (hash >> (64 - PRECISION)) as usize;
        // The remaining bits decide the register value. Shifting the index
        // out leaves zeros behind, so a hash whose tail is entirely zero
        // saturates at the width that is actually left
        let tail = hash << PRECISION;
        let rho = if tail == 0 {
            (64 - PRECISION) as u8 + 1
        } else {
            tail.leading_zeros() as u8 + 1
        };
        if rho > self.registers[index] {
            self.registers[index] = rho;
        }
    }

    /// Estimated distinct count.
    ///
    /// Small cardinalities use linear counting, which is exact enough to
    /// matter: the difference between 2 distinct values and 200 decides
    /// whether a column is worth ordering by at all.
    pub fn estimate(&self) -> u64 {
        let mut harmonic = 0.0f64;
        let mut zeros = 0usize;
        for &register in self.registers.iter() {
            harmonic += 1.0 / (1u64 << register) as f64;
            if register == 0 {
                zeros += 1;
            }
        }
        let m = REGISTERS as f64;
        let raw = ALPHA * m * m / harmonic;
        if raw <= 2.5 * m && zeros > 0 {
            // Linear counting, exact for cardinalities below the register
            // count where the raw estimator is unreliable
            return (m * (m / zeros as f64).ln()).round() as u64;
        }
        raw.round() as u64
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sketch_of(count: u64) -> DistinctSketch {
        let mut sketch = DistinctSketch::new();
        for value in 0..count {
            sketch.insert(&value.to_le_bytes());
        }
        sketch
    }

    /// Below the point where two values can collide into one register the
    /// count is exact, which is the range where the difference between one
    /// distinct value and three decides whether a column orders anything
    #[test]
    fn test_tiny_cardinalities_are_exact() {
        for count in [0u64, 1, 2, 3, 7, 16] {
            assert_eq!(
                sketch_of(count).estimate(),
                count,
                "linear counting must be exact at {count}"
            );
        }
    }

    /// Past that, register collisions make it an estimate. What has to
    /// hold is that it stays on the right side of the thresholds the
    /// clustering planner compares it against
    #[test]
    fn test_the_low_cardinality_threshold_is_never_crossed_by_error() {
        for count in [100u64, 200, 250] {
            let estimate = sketch_of(count).estimate();
            let error = (estimate as f64 - count as f64).abs() / count as f64;
            assert!(
                error < 0.05,
                "estimate {estimate} for {count} is off by {error}"
            );
            assert!(
                estimate <= 256,
                "{count} distinct values must not read as more than low cardinality"
            );
        }
        // And a column genuinely above the threshold reads as above it
        assert!(sketch_of(400).estimate() > 256);
    }

    /// Large cardinalities only have to land on the right side of the
    /// thresholds the planner compares them against
    #[test]
    fn test_large_cardinalities_land_within_a_few_percent() {
        for count in [10_000u64, 100_000, 1_000_000] {
            let estimate = sketch_of(count).estimate() as f64;
            let error = (estimate - count as f64).abs() / count as f64;
            assert!(
                error < 0.05,
                "estimate {estimate} for {count} is off by {error}"
            );
        }
    }

    /// Repeats are what a low-cardinality column is made of
    #[test]
    fn test_repeats_do_not_inflate_the_count() {
        let mut sketch = DistinctSketch::new();
        for _ in 0..10_000 {
            for value in 0..7u64 {
                sketch.insert(&value.to_le_bytes());
            }
        }
        assert_eq!(sketch.estimate(), 7);
    }

    /// The sketch counts values, not lengths, so byte-identical cells are
    /// one value however long they are
    #[test]
    fn test_varlen_values_count_by_content() {
        let mut sketch = DistinctSketch::new();
        for _ in 0..1000 {
            sketch.insert(b"alice");
            sketch.insert(b"bob");
            sketch.insert(b"");
        }
        assert_eq!(sketch.estimate(), 3);
    }
}
