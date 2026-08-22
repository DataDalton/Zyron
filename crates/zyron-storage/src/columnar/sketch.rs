//! Distinct-value counting for a column segment.
//!
//! A segment build materializes every value anyway, so it could count
//! distinct values exactly with a hash set. It does not, because an exact
//! set over a 256 MB file's column is hundreds of megabytes of transient
//! memory, and the number is only ever compared against thresholds. A
//! HyperLogLog holds a fixed 8 KiB of registers regardless of how many rows
//! it sees and lands inside a couple of percent, which is well inside the
//! distance between the thresholds that read it.
//!
//! Below a certain count an estimator is the wrong instrument. Linear
//! counting is exact only while no two values land in the same register, so
//! whether a small count comes back exact is a property of which hash
//! produced the values rather than of the sketch, and it fails more often
//! the smaller the register array is. The first few hundred distinct hashes
//! are therefore held in an open-addressed table and counted, and the
//! registers answer only once that table fills. The table is transient,
//! sized in kilobytes, and dropped with the sketch.
//!
//! The sketch is dropped once the caller has its estimate. Only the u64
//! estimate reaches a manifest, so a table with 100k files pays 8 bytes per
//! column per file rather than the registers

use zyron_common::hash64;

/// Register count, 2^PRECISION.
///
/// The relative error of the estimator is 1.04 over the square root of the
/// register count, so 1024 registers stand at 3.25% and land outside five
/// percent often enough that whether a column passes a five percent bound
/// is a property of which hash filled the registers. 8192 registers stand
/// at 1.15%, which puts five percent past four standard errors, and they
/// also lift the linear-counting ceiling to twenty thousand distinct values
/// so the range most columns sit in is counted rather than estimated.
///
/// The cost is 8 KiB of registers held while one column is being built and
/// dropped with the sketch. Nothing persists them: the only value that
/// reaches a manifest is the u64 estimate
const PRECISION: u32 = 13;
const REGISTERS: usize = 1 << PRECISION;

/// Bias constant for this register count
const ALPHA: f64 = 0.7213 / (1.0 + 1.079 / REGISTERS as f64);

/// Distinct hashes a sketch holds exactly when the caller names no capacity
/// of its own. The planner's low-cardinality band ends at 256, so every
/// column inside that band reports a count rather than an estimate, and no
/// threshold decision rests on estimator behaviour where it is weakest
pub const EXACT_CAPACITY: usize = 256;

/// Smallest exact table, which is what a sketch built for a handful of
/// distinct values allocates
const EXACT_SLOTS_MIN: usize = 64;

/// Slots needed to hold `capacity` distinct hashes at a load factor of one
/// half, where linear probing averages under two probes.
///
/// Taken up front rather than reached by doubling. Doubling a table that
/// ends up holding five thousand hashes rehashes sixteen thousand entries
/// across eleven reallocations, which measured seventeen times the cost of
/// allocating the final size once and probing a table that never moves
fn slots_for(capacity: usize) -> usize {
    capacity
        .saturating_add(1)
        .saturating_mul(2)
        .next_power_of_two()
        .max(EXACT_SLOTS_MIN)
}

/// Odd constant that spreads the high half of a wide value into the low
/// half before mixing. Multiplication by an odd word is a bijection, so it
/// loses nothing
const GOLDEN: u64 = 0x9E37_79B9_7F4A_7C15;

/// Widths this hashes inline. Past it the shared hash's per-call setup
/// starts to pay for itself against its throughput
const INLINE_HASH_MAX: usize = 16;

/// Mixes one 64-bit word so that every input bit affects every output bit.
///
/// The canonical three-round Murmur3 finalizer, a bijection with avalanche
/// good enough for register selection and for keying the exact table.
/// Sketch registers derived from it persist with segment stats, and
/// mix_finalize_2round is not interchangeable with it
#[inline(always)]
fn mix64(word: u64) -> u64 {
    zyron_common::mix_finalize_3round(word)
}

/// Reads eight little-endian bytes from `offset`, zero-padding past the end
#[inline(always)]
fn load_le_u64(value: &[u8], offset: usize) -> u64 {
    let available = value.len().saturating_sub(offset).min(8);
    let mut word = [0u8; 8];
    if available > 0 {
        word[..available].copy_from_slice(&value[offset..offset + available]);
    }
    u64::from_le_bytes(word)
}

/// Hashes a cell's bytes the way this sketch keys them.
///
/// Nothing outside the sketch depends on which hash it uses. The only value
/// that reaches a manifest is the distinct estimate, so the bytes a column
/// is stored as do not move when this changes, and neither do a bloom
/// filter's. That makes a short inline mix the right primitive for the
/// widths a fixed-size column is made of, where the shared `hash64` pays a
/// dispatch through a function pointer and an eight-lane pipeline setup
/// that eight bytes of input never amortizes.
///
/// A caller that already needs the hash for its own structures can key them
/// by this and hand the result to [`DistinctSketch::insert_hash`], paying
/// for it once
#[inline]
pub fn hash_cell(value: &[u8]) -> u64 {
    if value.len() > INLINE_HASH_MAX {
        return hash64(value);
    }
    let low = load_le_u64(value, 0);
    let high = load_le_u64(value, 8);
    // Length participates so that a shorter value and a longer one that
    // zero-pads to the same words stay distinct
    mix64(low ^ high.wrapping_mul(GOLDEN) ^ (value.len() as u64))
}

/// Two to the negative register for every value a register can hold.
///
/// Built from the exponent field rather than divided out, and read from the
/// table rather than built per register: moving an integer into the floating
/// point domain costs a register crossing, and the estimator does it once
/// per register over the whole array
static REGISTER_WEIGHTS: [f64; 256] = {
    let mut table = [0.0f64; 256];
    let mut value = 0usize;
    while value < 256 {
        table[value] = f64::from_bits((1023 - value as u64) << 52);
        value += 1;
    }
    table
};

/// Distinct-value sketch over hashed cell bytes.
pub struct DistinctSketch {
    /// Largest leading-zero run seen for each register, one byte each
    registers: [u8; REGISTERS],
    /// The first distinct hashes, open addressed with linear probing. A
    /// zero slot is empty, so a hash of zero is carried in its own flag.
    /// Sized once for the capacity and never moved
    exact: Vec<u64>,
    /// One less than the slot count, which is the index mask
    mask: usize,
    /// Distinct hashes counted before the registers take over
    capacity: usize,
    /// Distinct hashes recorded, including a zero if one arrived
    exact_len: usize,
    /// Whether a hash of zero was one of them
    zero_seen: bool,
    /// Set once more distinct hashes arrived than the capacity, after which
    /// only the registers answer
    saturated: bool,
}

impl Default for DistinctSketch {
    fn default() -> Self {
        Self::new()
    }
}

impl DistinctSketch {
    pub fn new() -> Self {
        Self::with_exact_capacity(EXACT_CAPACITY)
    }

    /// Builds a sketch that counts its first `capacity` distinct hashes
    /// exactly before the registers take over.
    ///
    /// A caller that compares the count against a threshold reads an exact
    /// answer up to that capacity rather than an estimate, which is what
    /// lets a segment build decide on a dictionary without keeping a second
    /// set of its own beside this one
    pub fn with_exact_capacity(capacity: usize) -> Self {
        let slots = slots_for(capacity);
        Self {
            registers: [0u8; REGISTERS],
            exact: vec![0u64; slots],
            mask: slots - 1,
            capacity,
            exact_len: 0,
            zero_seen: false,
            saturated: false,
        }
    }

    /// Adds one value, identified by its bytes
    #[inline]
    pub fn insert(&mut self, value: &[u8]) {
        self.insert_hash(hash_cell(value));
    }

    /// Adds one value already hashed with [`hash_cell`]. A caller that keys
    /// its own structures by the same hash pays for it once
    #[inline]
    pub fn insert_hash(&mut self, hash: u64) {
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
        // Nothing but the register moves here. Carrying the estimator's
        // running terms alongside was measured slower: it puts two integer
        // to floating point crossings and an add on a branch every value
        // can take, to save one walk that happens once for the whole column
        if rho > self.registers[index] {
            self.registers[index] = rho;
        }
        if !self.saturated {
            self.record_exact(hash);
        }
    }

    /// Records one hash in the exact table, saturating once it is full.
    ///
    /// The slot is chosen from the low bits and the register from the high
    /// bits, so the two structures index independently off one hash
    #[inline]
    fn record_exact(&mut self, hash: u64) {
        if hash == 0 {
            if !self.zero_seen {
                self.zero_seen = true;
                self.note_insert();
            }
            return;
        }
        // The table holds twice the capacity and stops taking hashes one
        // past it, so it is never more than half full and the probe always
        // reaches an empty slot
        let mut slot = (hash as usize) & self.mask;
        loop {
            let held = self.exact[slot];
            if held == hash {
                return;
            }
            if held == 0 {
                self.exact[slot] = hash;
                self.note_insert();
                return;
            }
            slot = (slot + 1) & self.mask;
        }
    }

    /// Counts one newly seen hash and saturates once the capacity is passed
    #[inline]
    fn note_insert(&mut self) {
        self.exact_len += 1;
        self.saturated = self.exact_len > self.capacity;
    }

    /// Distinct hashes counted exactly, or None once more arrived than the
    /// exact table holds.
    ///
    /// A caller comparing against a threshold inside the table's capacity
    /// reads this rather than [`Self::estimate`], because it is a count
    /// and carries no estimator error at all
    #[inline]
    pub fn exact_count(&self) -> Option<u64> {
        (!self.saturated).then_some(self.exact_len as u64)
    }

    /// Distinct hashes counted, saturating one past the capacity.
    ///
    /// A consumer comparing the count against a ceiling reads a saturated
    /// value as "more than the capacity", which compares the same way the
    /// true count would against any threshold at or below it
    #[inline]
    pub fn counted(&self) -> usize {
        self.exact_len.min(self.capacity + 1)
    }

    /// Estimated distinct count.
    ///
    /// Exact while the table holds every hash seen. Past that the registers
    /// answer, with linear counting covering the range where the raw
    /// estimator is unreliable
    pub fn estimate(&self) -> u64 {
        if let Some(exact) = self.exact_count() {
            return exact;
        }
        // Four running sums rather than one, because a single accumulator
        // makes every add wait on the one before it and the array is long
        // enough for that chain to be the whole cost
        let mut sums = [0.0f64; 4];
        for group in self.registers.chunks_exact(4) {
            sums[0] += REGISTER_WEIGHTS[group[0] as usize];
            sums[1] += REGISTER_WEIGHTS[group[1] as usize];
            sums[2] += REGISTER_WEIGHTS[group[2] as usize];
            sums[3] += REGISTER_WEIGHTS[group[3] as usize];
        }
        let harmonic = (sums[0] + sums[1]) + (sums[2] + sums[3]);
        let zeros = self.registers.iter().filter(|&&r| r == 0).count();

        let m = REGISTERS as f64;
        let raw = ALPHA * m * m / harmonic;
        if raw <= 2.5 * m && zeros > 0 {
            // Linear counting, which the raw estimator is unreliable below
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
                "the exact table must be exact at {count}"
            );
        }
    }

    /// Exactness up to the table's capacity is a property of the table, not
    /// of the hash that filled it. Every count inside the band the planner
    /// calls low cardinality comes back as a count
    #[test]
    fn test_every_cardinality_inside_the_exact_table_is_exact() {
        for count in 0..=EXACT_CAPACITY as u64 {
            let sketch = sketch_of(count);
            assert_eq!(
                sketch.exact_count(),
                Some(count),
                "{count} distinct values must still be counted"
            );
            assert_eq!(sketch.estimate(), count, "{count} must report exactly");
        }
    }

    /// One past the capacity the table gives up and says so, rather than
    /// reporting a count it can no longer keep
    #[test]
    fn test_passing_the_capacity_hands_over_to_the_registers() {
        let sketch = sketch_of(EXACT_CAPACITY as u64 + 1);
        assert_eq!(
            sketch.exact_count(),
            None,
            "the table cannot hold more than its capacity"
        );
        let estimate = sketch.estimate() as f64;
        let error = (estimate - (EXACT_CAPACITY as f64 + 1.0)).abs() / (EXACT_CAPACITY as f64);
        assert!(
            error < 0.05,
            "handover estimate {estimate} is off by {error}"
        );
    }

    /// A hash of zero is the empty-slot marker, so it has to be counted
    /// somewhere other than a slot or it reads as a value never seen
    #[test]
    fn test_a_zero_hash_is_counted_once() {
        let mut sketch = DistinctSketch::new();
        for _ in 0..100 {
            sketch.insert_hash(0);
        }
        assert_eq!(sketch.exact_count(), Some(1));
        sketch.insert_hash(1);
        assert_eq!(sketch.exact_count(), Some(2));
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

    /// The five percent bound a manifest's distinct estimate is held to has
    /// to be a property of the register count, not of the values that
    /// happened to fill it.
    ///
    /// Sequential ids, a strided key and a scattered one are the three
    /// shapes a real column's values take, and the counts walk the range
    /// from just past the exact table to a million. A register array too
    /// small to carry the bound fails somewhere in this grid, which is
    /// where it should be caught rather than in a lake test that builds a
    /// manifest
    #[test]
    fn test_the_five_percent_bound_holds_across_shapes_and_sizes() {
        let shapes: [(&str, fn(u64) -> u64); 3] = [
            ("sequential", |i| i),
            ("strided", |i| i.wrapping_mul(7919)),
            ("scattered", |i| i.wrapping_mul(6_364_136_223_846_793_005)),
        ];
        let counts = [
            300u64, 1_000, 2_600, 5_000, 10_000, 25_000, 50_000, 200_000, 1_000_000,
        ];
        for (shape, generate) in shapes {
            for count in counts {
                let mut sketch = DistinctSketch::new();
                for i in 0..count {
                    sketch.insert(&generate(i).to_le_bytes());
                }
                let estimate = sketch.estimate() as f64;
                let error = (estimate - count as f64).abs() / count as f64;
                assert!(
                    error < 0.05,
                    "{shape} at {count} estimated {estimate}, off by {:.2}%",
                    error * 100.0
                );
            }
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

    /// Values that zero-pad to the same words are still separate values,
    /// which is what folding the length into the hash buys
    #[test]
    fn test_shorter_values_do_not_fold_into_longer_ones() {
        let mut sketch = DistinctSketch::new();
        sketch.insert(b"");
        sketch.insert(&[0u8]);
        sketch.insert(&[0u8, 0]);
        sketch.insert(&[0u8; 8]);
        sketch.insert(&[0u8; 16]);
        assert_eq!(sketch.estimate(), 5);
    }

    /// Hashing a value once and feeding the hash is the same measurement as
    /// handing the sketch the bytes
    #[test]
    fn test_inserting_a_hash_matches_inserting_the_value() {
        let mut by_value = DistinctSketch::new();
        let mut by_hash = DistinctSketch::new();
        for value in 0..5_000u64 {
            let bytes = value.to_le_bytes();
            by_value.insert(&bytes);
            by_hash.insert_hash(hash_cell(&bytes));
        }
        assert_eq!(by_value.estimate(), by_hash.estimate());
    }

    /// Widths past the inline path go to the shared hash, and both halves
    /// of the split have to count the same values the same way
    #[test]
    fn test_values_wider_than_the_inline_path_still_count() {
        let mut sketch = DistinctSketch::new();
        for value in 0..300u64 {
            let mut wide = [0u8; 40];
            wide[..8].copy_from_slice(&value.to_le_bytes());
            sketch.insert(&wide);
        }
        let estimate = sketch.estimate() as f64;
        let error = (estimate - 300.0).abs() / 300.0;
        assert!(
            error < 0.05,
            "estimate {estimate} for 300 is off by {error}"
        );
    }
}
