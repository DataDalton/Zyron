// Single-pass column profiler with bounded memory.
// Numeric stats use Welford's online moments for mean, variance, skewness,
// and kurtosis. Percentiles use a Greenwald-Khanna sketch with epsilon=0.01
// (target 1% accuracy). Distinct count uses HyperLogLog with precision=14
// (~16KB, ~0.81% error). Most common values use a Misra-Gries sketch.

use crate::value::{AnalyticsValue, hash_value_into};
use std::collections::HashMap;
use zyron_common::error::Result;
use zyron_common::mix_finalize_2round;

#[derive(Debug, Clone, Default)]
pub struct NumericRange {
    pub low: f64,
    pub high: f64,
    pub low_inclusive: bool,
    pub high_inclusive: bool,
}

#[derive(Debug, Clone)]
pub struct HistogramBin {
    pub range: NumericRange,
    pub count: u64,
}

#[derive(Debug, Clone, Default)]
pub struct PatternFrequency {
    pub pattern: String,
    pub count: u64,
}

#[derive(Debug, Clone, Default)]
pub struct PercentileSet {
    pub p1: f64,
    pub p5: f64,
    pub p10: f64,
    pub p25: f64,
    pub p50: f64,
    pub p75: f64,
    pub p90: f64,
    pub p95: f64,
    pub p99: f64,
}

#[derive(Debug, Clone, Default)]
pub struct ColumnProfile {
    pub column_name: String,
    pub data_type: String,
    pub null_count: u64,
    pub null_pct: f64,
    pub distinct_count: u64,
    pub distinct_pct: f64,
    pub min: Option<AnalyticsValue>,
    pub max: Option<AnalyticsValue>,
    pub mean: Option<f64>,
    pub median: Option<f64>,
    pub stddev: Option<f64>,
    pub variance: Option<f64>,
    pub skewness: Option<f64>,
    pub kurtosis: Option<f64>,
    pub percentiles: Option<PercentileSet>,
    pub most_common_values: Vec<(AnalyticsValue, u64)>,
    pub histogram: Vec<HistogramBin>,
    pub pattern_frequencies: Vec<PatternFrequency>,
}

#[derive(Debug, Clone, Default)]
pub struct TableProfile {
    pub table_name: String,
    pub row_count: u64,
    pub columns: Vec<ColumnProfile>,
    pub correlation_matrix: Option<crate::correlation::CorrelationMatrix>,
}

// ===== HyperLogLog distinct counter =====
const HLL_PRECISION: u8 = 14;
const HLL_M: usize = 1 << HLL_PRECISION;

pub struct HyperLogLog {
    registers: Vec<u8>,
}

impl HyperLogLog {
    pub fn new() -> Self {
        Self {
            registers: vec![0; HLL_M],
        }
    }

    fn hash64(value: &AnalyticsValue) -> u64 {
        // Project's canonical fx_mix seeded with a fixed HLL constant so
        // this hash space is independent of other callers using hash_value_into
        mix_finalize_2round(hash_value_into(0x9E37_79B9_7F4A_7C15, value))
    }

    pub fn add(&mut self, value: &AnalyticsValue) {
        let h = Self::hash64(value);
        let idx = (h & ((HLL_M as u64) - 1)) as usize;
        let w = h >> HLL_PRECISION;
        // Number of leading zeros in the remaining 64-PRECISION bits + 1
        let lz = if w == 0 {
            64 - HLL_PRECISION + 1
        } else {
            (w.leading_zeros() as u8 - HLL_PRECISION + 1).min(64 - HLL_PRECISION + 1)
        };
        if lz > self.registers[idx] {
            self.registers[idx] = lz;
        }
    }

    /// Cardinality estimate using Ertl's 2017 "Improved" estimator with
    /// the sigma correction for empty registers.
    ///
    /// Replaces the raw HLL form's c[0]*2^0 term with m * sigma(c[0]/m).
    /// Since sigma(x) > x on (0, 1), this enlarges the denominator and
    /// shrinks the estimate, correcting the raw form's known over-count
    /// when a large fraction of registers are empty. Edge cases:
    ///   - all registers zero (c[0] = m): sigma(1) -> infinity, n_hat -> 0
    ///   - no empty registers (c[0] = 0): sigma(0) = 0, so the formula
    ///     collapses to the raw HLL estimator
    ///
    /// Saturated registers (value q+1, possible only at >2^49 distinct
    /// values for our q=50) are included at the raw weight c[q+1]*2^{-(q+1)}
    /// rather than the Ertl tau correction. The tau correction matters
    /// only well outside the cardinality range any practical column
    /// profile will ever reach.
    pub fn estimate(&self) -> u64 {
        let m = HLL_M as f64;
        const Q: usize = 64 - HLL_PRECISION as usize; // 50 for p=14
        const HISTOGRAM_LEN: usize = Q + 2; // covers indices 0..=q+1

        // Build register-value histogram on the stack. Register values are
        // capped at Q+1 by `add`, so this fixed-size array always fits.
        let mut c = [0u32; HISTOGRAM_LEN];
        for &r in &self.registers {
            let idx = (r as usize).min(Q + 1);
            c[idx] += 1;
        }

        // Edge case: empty sketch.
        if c[0] == HLL_M as u32 {
            return 0;
        }

        // Compute z = sum_{k=1}^{q+1} c[k] * 2^{-k} via Horner-style
        // accumulation from the highest k down. After the loop:
        //   y = c[1] + c[2]/2 + c[3]/4 + ... + c[q+1]/2^q
        // and z = y / 2 = sum_{k=1}^{q+1} c[k] * 2^{-k}.
        let mut y = c[Q + 1] as f64;
        for k in (1..=Q).rev() {
            y = y / 2.0 + c[k] as f64;
        }
        let z_no_zeros = y / 2.0;

        // Add the sigma correction for empty registers.
        let z = z_no_zeros + m * hll_sigma(c[0] as f64 / m);

        // alpha_inf = 1 / (2 * ln 2). The m-dependent alpha_m used by the
        // raw estimator converges to this as m grows; for m >= 128 the two
        // agree to within ~0.5 %, and Ertl's analysis is derived against
        // alpha_inf so we use that directly here.
        let alpha_inf = 1.0 / (2.0 * std::f64::consts::LN_2);
        let estimate = alpha_inf * m * m / z;
        estimate.round().max(0.0) as u64
    }
}

/// Ertl's sigma function, evaluated as a power series. Converges very
/// quickly because successive terms square the previous x and contribute
/// geometrically smaller amounts; the loop typically exits after 5-10
/// iterations even for x close to 1.
///
///   sigma(x) = x + sum_{k=1}^infinity x^(2^k) * 2^(k-1)
///            = x + x^2 + 2 * x^4 + 4 * x^8 + 8 * x^16 + ...
fn hll_sigma(x: f64) -> f64 {
    if x == 1.0 {
        return f64::INFINITY;
    }
    if x == 0.0 {
        return 0.0;
    }
    let mut x_pow = x * x; // starts at x^2 (the k=1 term's base)
    let mut weight = 1.0; // starts at 2^0 (the k=1 term's coefficient)
    let mut z = x;
    loop {
        let term = x_pow * weight;
        let z_new = z + term;
        if z_new == z {
            // Converged: term is below f64 epsilon at this magnitude
            return z_new;
        }
        z = z_new;
        x_pow = x_pow * x_pow; // x^(2^(k+1))
        weight = weight * 2.0; // 2^k
    }
}

// ===== Space-Saving top-k sketch (Metwally et al. 2005) =====
//
// Why Space-Saving over Misra-Gries:
// - Per-miss cost: SS does one O(k) min-scan; MG does an O(k) decrement
//   walk PLUS conditional removals (an extra O(k) HashMap remove pass).
// - SS's count of a kept item is an upper bound on its true frequency,
//   which is the same guarantee a top-k caller cares about, plus the
//   gap is bounded by min_count at eviction time.
//
// Layout: a flat Vec<Slot> backs the table; a parallel hash -> Vec-index
// PreHashMap gives O(1) hit lookup. Hits don't touch the Vec (just the
// counter on the slot inside the Vec). Misses with a full table find the
// minimum-counter slot in one Vec scan and overwrite it in place.
pub struct MisraGries {
    capacity: usize,
    slots: Vec<SsSlot>,
    // hash_key -> index into `slots`. Hits look up here, then go straight
    // to slots[i].count without re-hashing.
    index: zyron_common::PreHashMap<u64, u32>,
}

struct SsSlot {
    hash_key: u64,
    value: AnalyticsValue,
    count: u64,
}

impl MisraGries {
    pub fn new(k: usize) -> Self {
        Self {
            capacity: k,
            slots: Vec::with_capacity(k),
            index: zyron_common::PreHashMap::default(),
        }
    }

    pub fn add(&mut self, v: &AnalyticsValue) {
        let key = mg_key(v);
        // Hit fast path: bump the counter on the existing slot
        if let Some(&idx) = self.index.get(&key) {
            self.slots[idx as usize].count += 1;
            return;
        }
        // Miss with room: append a new slot
        if self.slots.len() < self.capacity {
            let idx = self.slots.len() as u32;
            self.slots.push(SsSlot {
                hash_key: key,
                value: v.clone(),
                count: 1,
            });
            self.index.insert(key, idx);
            return;
        }
        // Miss with full table: find the minimum-counter slot and replace
        // its (hash, value) in place; new count = old min + 1. This is the
        // Space-Saving rule.
        let mut min_idx: usize = 0;
        let mut min_count: u64 = self.slots[0].count;
        for (i, slot) in self.slots.iter().enumerate().skip(1) {
            if slot.count < min_count {
                min_count = slot.count;
                min_idx = i;
            }
        }
        let evicted_hash = self.slots[min_idx].hash_key;
        self.index.remove(&evicted_hash);
        self.slots[min_idx] = SsSlot {
            hash_key: key,
            value: v.clone(),
            count: min_count + 1,
        };
        self.index.insert(key, min_idx as u32);
    }

    pub fn into_sorted(self) -> Vec<(AnalyticsValue, u64)> {
        let mut v: Vec<(AnalyticsValue, u64)> =
            self.slots.into_iter().map(|s| (s.value, s.count)).collect();
        v.sort_by(|a, b| b.1.cmp(&a.1));
        v
    }
}

#[inline]
fn mg_key(v: &AnalyticsValue) -> u64 {
    // Distinct seed from HLL so the two hash spaces don't correlate
    mix_finalize_2round(hash_value_into(0xC0FF_EE15_DEAD_BEEF, v))
}

// ===== Streaming percentile reservoir =====
//
// Vitter's Algorithm L (1985) reservoir sampling. Holds at most
// RESERVOIR_CAPACITY samples, with two key properties that matter for
// open-ended profiling:
//
// 1. Per-record cost amortises to O(k * log(n/k) / n). The skip-count
//    formulation does almost nothing on most inserts: a single integer
//    compare against the next replacement index. RNG calls happen only at
//    replacement boundaries.
//
// 2. Memory grows only as samples arrive (Vec::new on construction). A
//    profile of a 10-row column allocates 10 * sizeof(f64) = 80 bytes
//    rather than the full reservoir buffer.
//
// Variance of a reservoir-sample quantile decays as 1/sqrt(k) and is
// independent of n. With k = 65536 the standard deviation of the median
// estimator is roughly 0.4% of the data range on uniform inputs,
// comfortably under the 1% accuracy the profile validation asserts. The
// same accuracy applies whether the column has 10K or 10B rows.
const RESERVOIR_CAPACITY: usize = 65_536;

pub struct Reservoir {
    samples: Vec<f64>,
    n_seen: u64,
    // Algorithm L state, populated once the reservoir first fills
    next_replace_at: u64,
    w: f64,
    rng: ProfilingRng,
    // True if `samples` is currently sorted ascending. Set by quantile()
    // when it sorts in place, cleared by any insert. Lets callers
    // interleave insert and quantile correctly: each quantile call only
    // re-sorts when something has changed since the last one.
    sorted: bool,
}

// SplitMix64-style PRNG, seeded per reservoir so two columns produce
// independent sample streams without sharing state.
struct ProfilingRng(u64);

impl ProfilingRng {
    #[inline]
    fn new(seed: u64) -> Self {
        Self(seed.wrapping_add(0x9E3779B97F4A7C15))
    }
    #[inline]
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E3779B97F4A7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
        z ^ (z >> 31)
    }
    // Uniform double in (0, 1). Top 53 bits, scaled, with a small offset
    // that excludes exactly 0 so .ln() is finite.
    #[inline]
    fn next_f64_open(&mut self) -> f64 {
        let bits = (self.next() >> 11) | 1; // ensure nonzero
        bits as f64 * (1.0f64 / (1u64 << 53) as f64)
    }
}

impl Reservoir {
    pub fn new(seed: u64) -> Self {
        // Vec::new defers any heap allocation until the first push; columns
        // smaller than the reservoir capacity allocate exactly len * 8 bytes
        Self {
            samples: Vec::new(),
            n_seen: 0,
            next_replace_at: u64::MAX,
            w: 0.0,
            rng: ProfilingRng::new(seed),
            sorted: true,
        }
    }

    #[inline]
    pub fn insert(&mut self, v: f64) {
        if v.is_nan() {
            return;
        }
        self.n_seen += 1;
        // Any insert invalidates a prior in-place sort
        self.sorted = false;

        // Fill phase: keep every value verbatim until the reservoir is full
        if self.samples.len() < RESERVOIR_CAPACITY {
            self.samples.push(v);
            if self.samples.len() == RESERVOIR_CAPACITY {
                // Initialise Algorithm L state on the boundary
                self.w = (self.rng.next_f64_open().ln() / RESERVOIR_CAPACITY as f64).exp();
                self.next_replace_at = self.n_seen + self.skip_count();
            }
            return;
        }

        // Skip phase: most calls take this fast exit. A single integer
        // compare decides whether anything else needs to happen.
        if self.n_seen < self.next_replace_at {
            return;
        }

        // Replacement phase: write into a uniformly random slot, then
        // recompute W and the next replacement index
        let j = (self.rng.next() % RESERVOIR_CAPACITY as u64) as usize;
        self.samples[j] = v;
        self.w *= (self.rng.next_f64_open().ln() / RESERVOIR_CAPACITY as f64).exp();
        self.next_replace_at = self.n_seen + self.skip_count();
    }

    #[inline]
    fn skip_count(&mut self) -> u64 {
        let r = self.rng.next_f64_open();
        // log(1 - W) is negative; r.ln() is also negative; ratio is positive
        let one_minus_w = 1.0 - self.w;
        if one_minus_w <= 0.0 {
            // W has saturated to 1.0; no further replacements will land
            return u64::MAX;
        }
        let skip = (r.ln() / one_minus_w.ln()).floor();
        if !skip.is_finite() || skip < 0.0 {
            return u64::MAX;
        }
        skip as u64 + 1
    }

    pub fn quantile(&mut self, q: f64) -> f64 {
        if self.samples.is_empty() {
            return f64::NAN;
        }
        // Sort lazily; skip the resort if the buffer is still sorted from
        // a prior quantile call (no insert in between).
        if !self.sorted {
            self.samples
                .sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            self.sorted = true;
        }
        let pos = (q * (self.samples.len() as f64 - 1.0)).round() as usize;
        self.samples[pos.min(self.samples.len() - 1)]
    }
}

// ===== Welford online moments =====
pub struct OnlineMoments {
    pub n: u64,
    pub mean: f64,
    pub m2: f64,
    pub m3: f64,
    pub m4: f64,
    pub min: f64,
    pub max: f64,
}

impl OnlineMoments {
    pub fn new() -> Self {
        Self {
            n: 0,
            mean: 0.0,
            m2: 0.0,
            m3: 0.0,
            m4: 0.0,
            min: f64::INFINITY,
            max: f64::NEG_INFINITY,
        }
    }

    pub fn ingest(&mut self, x: f64) {
        if x.is_nan() {
            return;
        }
        self.n += 1;
        let n = self.n as f64;
        let delta = x - self.mean;
        let delta_n = delta / n;
        let delta_n2 = delta_n * delta_n;
        let term1 = delta * delta_n * (n - 1.0);
        self.mean += delta_n;
        self.m4 += term1 * delta_n2 * (n * n - 3.0 * n + 3.0) + 6.0 * delta_n2 * self.m2
            - 4.0 * delta_n * self.m3;
        self.m3 += term1 * delta_n * (n - 2.0) - 3.0 * delta_n * self.m2;
        self.m2 += term1;
        if x < self.min {
            self.min = x;
        }
        if x > self.max {
            self.max = x;
        }
    }

    pub fn variance(&self) -> Option<f64> {
        if self.n < 2 {
            None
        } else {
            Some(self.m2 / (self.n as f64 - 1.0))
        }
    }

    pub fn stddev(&self) -> Option<f64> {
        self.variance().map(|v| v.sqrt())
    }

    pub fn skewness(&self) -> Option<f64> {
        if self.n < 3 || self.m2 == 0.0 {
            None
        } else {
            let n = self.n as f64;
            Some((n.sqrt() * self.m3) / self.m2.powf(1.5))
        }
    }

    pub fn kurtosis(&self) -> Option<f64> {
        if self.n < 4 || self.m2 == 0.0 {
            None
        } else {
            let n = self.n as f64;
            // Excess kurtosis
            Some((n * self.m4) / (self.m2 * self.m2) - 3.0)
        }
    }
}

// ===== Pattern detector =====
fn detect_pattern(s: &str) -> &'static str {
    let trimmed = s.trim();
    if trimmed.is_empty() {
        return "empty";
    }
    if is_email(trimmed) {
        return "email";
    }
    if is_uuid(trimmed) {
        return "uuid";
    }
    // Test ipv4 before phone since dotted-quad has digit ratio similar to a
    // phone number (a, b, c, d are all digits separated by dots)
    if is_ipv4(trimmed) {
        return "ipv4";
    }
    if is_url(trimmed) {
        return "url";
    }
    if is_phone(trimmed) {
        return "phone";
    }
    if trimmed.chars().all(|c| c.is_ascii_digit()) {
        return "numeric_string";
    }
    if trimmed.chars().all(|c| c.is_ascii_alphabetic()) {
        return "alpha";
    }
    "other"
}

// All pattern detectors are allocation-free: they walk the input as bytes
// and decide via byte-level checks. None of them call split().collect() or
// chars().filter().collect(). For high-cardinality text columns this saves
// one heap alloc per check, per pattern.

fn is_email(s: &str) -> bool {
    let bytes = s.as_bytes();
    // Find first '@'. Local part must be non-empty, domain must be
    // non-empty and contain at least one '.'
    let at_idx = match bytes.iter().position(|&b| b == b'@') {
        Some(i) => i,
        None => return false,
    };
    if at_idx == 0 || at_idx + 1 >= bytes.len() {
        return false;
    }
    // Reject multiple '@'
    if bytes[at_idx + 1..].iter().any(|&b| b == b'@') {
        return false;
    }
    bytes[at_idx + 1..].iter().any(|&b| b == b'.')
}

fn is_phone(s: &str) -> bool {
    let bytes = s.as_bytes();
    let total = bytes.len();
    if total == 0 {
        return false;
    }
    let mut digits = 0usize;
    for &b in bytes {
        if b.is_ascii_digit() {
            digits += 1;
        }
    }
    digits >= 7 && digits <= 15 && (digits as f64 / total as f64) > 0.5
}

fn is_uuid(s: &str) -> bool {
    let bytes = s.as_bytes();
    if bytes.len() != 36 {
        return false;
    }
    for (i, &b) in bytes.iter().enumerate() {
        match i {
            8 | 13 | 18 | 23 => {
                if b != b'-' {
                    return false;
                }
            }
            _ => {
                if !b.is_ascii_hexdigit() {
                    return false;
                }
            }
        }
    }
    true
}

fn is_ipv4(s: &str) -> bool {
    let bytes = s.as_bytes();
    if bytes.len() < 7 || bytes.len() > 15 {
        return false;
    }
    let mut octets = 0u8;
    let mut current: u32 = 0;
    let mut has_digit = false;
    for &b in bytes {
        if b == b'.' {
            if !has_digit {
                return false;
            }
            octets += 1;
            if octets > 3 {
                return false;
            }
            current = 0;
            has_digit = false;
        } else if b.is_ascii_digit() {
            current = current * 10 + (b - b'0') as u32;
            if current > 255 {
                return false;
            }
            has_digit = true;
        } else {
            return false;
        }
    }
    has_digit && octets == 3
}

fn is_url(s: &str) -> bool {
    s.starts_with("http://") || s.starts_with("https://")
}

// ===== Single-pass profiler =====
//
// Maintenance overhead per record stays bounded regardless of input size:
// - HLL register update is O(1) and the hash is the project's fx_mix
//   over the value's payload bytes
// - Reservoir sampling is O(1) amortised per insert
// - Welford moments are six FMAs
// - min/max compares the value against the running extrema and only clones
//   when a new extremum is observed
// - MisraGries top-k and pattern frequency tracking are O(k) per call.
//   For high-cardinality streams the per-record cost dominates the profile,
//   so after MG_FULL_RATE_RECORDS values we sample these two structures at
//   MG_SAMPLE_STRIDE to bound total work.

const MG_FULL_RATE_RECORDS: u64 = 8_192;
const MG_SAMPLE_STRIDE: u64 = 64;

pub struct ColumnProfiler {
    pub column_name: String,
    pub data_type: String,
    pub seen: u64,
    pub null_count: u64,
    pub hll: HyperLogLog,
    pub reservoir: Reservoir,
    pub moments: OnlineMoments,
    pub mg: MisraGries,
    pub patterns: HashMap<&'static str, u64>,
    pub min_value: Option<AnalyticsValue>,
    pub max_value: Option<AnalyticsValue>,
}

impl ColumnProfiler {
    pub fn new(column_name: String, data_type: String) -> Self {
        // Seed the reservoir's RNG from a hash of the column name so two
        // columns produce uncorrelated sample paths within one profile run
        let mut seed: u64 = 0xA1F0_5C9E_2B73_4DD1;
        for &b in column_name.as_bytes() {
            seed = zyron_common::fx_mix(seed, b as u64);
        }
        Self {
            column_name,
            data_type,
            seen: 0,
            null_count: 0,
            hll: HyperLogLog::new(),
            reservoir: Reservoir::new(seed),
            moments: OnlineMoments::new(),
            mg: MisraGries::new(64),
            patterns: HashMap::new(),
            min_value: None,
            max_value: None,
        }
    }

    #[inline]
    pub fn ingest(&mut self, value: &AnalyticsValue) {
        self.seen += 1;
        if value.is_null() {
            self.null_count += 1;
            return;
        }
        self.hll.add(value);

        // Top-k and pattern frequency: full rate during the first window,
        // then strided sampling. Sampling preserves the most-common values
        // statistically while bounding per-record cost
        let track_topk = self.seen <= MG_FULL_RATE_RECORDS || self.seen % MG_SAMPLE_STRIDE == 0;
        if track_topk {
            self.mg.add(value);
        }

        // min/max: assign_from reuses the existing String allocation when
        // both sides are Text, avoiding a drop+heap-alloc on every new
        // extremum for text columns.
        if let Some(slot) = self.min_value.as_mut() {
            if value.total_cmp(slot) == std::cmp::Ordering::Less {
                slot.assign_from(value);
            }
        } else {
            self.min_value = Some(value.clone());
        }
        if let Some(slot) = self.max_value.as_mut() {
            if value.total_cmp(slot) == std::cmp::Ordering::Greater {
                slot.assign_from(value);
            }
        } else {
            self.max_value = Some(value.clone());
        }

        if let Some(f) = value.as_f64() {
            self.reservoir.insert(f);
            self.moments.ingest(f);
        }

        if track_topk {
            if let Some(s) = value.as_text() {
                let p = detect_pattern(s);
                *self.patterns.entry(p).or_insert(0) += 1;
            }
        }
    }

    pub fn finalise(mut self) -> ColumnProfile {
        let total = self.seen.max(1) as f64;
        let null_pct = self.null_count as f64 / total;
        let distinct = self.hll.estimate().min(self.seen);
        let distinct_pct = distinct as f64 / total;

        let percentiles = if self.moments.n >= 2 {
            Some(PercentileSet {
                p1: self.reservoir.quantile(0.01),
                p5: self.reservoir.quantile(0.05),
                p10: self.reservoir.quantile(0.10),
                p25: self.reservoir.quantile(0.25),
                p50: self.reservoir.quantile(0.50),
                p75: self.reservoir.quantile(0.75),
                p90: self.reservoir.quantile(0.90),
                p95: self.reservoir.quantile(0.95),
                p99: self.reservoir.quantile(0.99),
            })
        } else {
            None
        };

        // Build the histogram from the (now sorted) reservoir samples. Bin
        // counts are scaled by total/sample_count so the histogram reports
        // estimated population counts rather than sample counts.
        let histogram = build_histogram_from_reservoir(
            &self.reservoir.samples,
            self.moments.n,
            self.moments.min,
            self.moments.max,
        );
        let median = percentiles.as_ref().map(|p| p.p50);

        let pattern_frequencies = {
            let mut pf: Vec<PatternFrequency> = self
                .patterns
                .iter()
                .map(|(k, v)| PatternFrequency {
                    pattern: (*k).to_string(),
                    count: *v,
                })
                .collect();
            pf.sort_by(|a, b| b.count.cmp(&a.count));
            pf
        };

        let mean = if self.moments.n > 0 {
            Some(self.moments.mean)
        } else {
            None
        };

        ColumnProfile {
            column_name: self.column_name,
            data_type: self.data_type,
            null_count: self.null_count,
            null_pct,
            distinct_count: distinct,
            distinct_pct,
            min: self.min_value,
            max: self.max_value,
            mean,
            median,
            stddev: self.moments.stddev(),
            variance: self.moments.variance(),
            skewness: self.moments.skewness(),
            kurtosis: self.moments.kurtosis(),
            percentiles,
            most_common_values: self.mg.into_sorted(),
            histogram,
            pattern_frequencies,
        }
    }
}

// 16 equi-width bins between observed min and max, populated by counting
// reservoir samples per bin and scaling to the full record count
fn build_histogram_from_reservoir(
    samples: &[f64],
    total_records: u64,
    lo: f64,
    hi: f64,
) -> Vec<HistogramBin> {
    if samples.len() < 2 || !(lo.is_finite() && hi.is_finite()) || hi == lo {
        return Vec::new();
    }
    let nbins = 16usize;
    let width = (hi - lo) / nbins as f64;
    let mut counts = vec![0u64; nbins];
    for &v in samples {
        if v < lo || v > hi {
            continue;
        }
        let mut idx = ((v - lo) / width).floor() as usize;
        if idx >= nbins {
            idx = nbins - 1;
        }
        counts[idx] += 1;
    }
    let scale = total_records as f64 / samples.len() as f64;
    let mut bins = Vec::with_capacity(nbins);
    for i in 0..nbins {
        let low = lo + i as f64 * width;
        let high = if i + 1 == nbins { hi } else { low + width };
        bins.push(HistogramBin {
            range: NumericRange {
                low,
                high,
                low_inclusive: true,
                high_inclusive: i + 1 == nbins,
            },
            count: (counts[i] as f64 * scale).round() as u64,
        });
    }
    bins
}

pub fn column_profile(
    column_name: &str,
    data_type: &str,
    values: &[AnalyticsValue],
) -> Result<ColumnProfile> {
    let mut prof = ColumnProfiler::new(column_name.to_string(), data_type.to_string());
    for v in values {
        prof.ingest(v);
    }
    Ok(prof.finalise())
}

// Profile an entire table given column-major data. The caller chooses
// whether to provide all rows (full pass) or a sample. Memory is bounded
// by the per-column profiler, regardless of input length.
pub fn profile_table(
    table_name: &str,
    column_names: &[String],
    column_types: &[String],
    columns: &[Vec<AnalyticsValue>],
    include_correlations: bool,
) -> Result<TableProfile> {
    let row_count = columns.first().map(|c| c.len()).unwrap_or(0) as u64;
    let mut col_profiles = Vec::with_capacity(column_names.len());
    for (i, name) in column_names.iter().enumerate() {
        let dtype = column_types
            .get(i)
            .cloned()
            .unwrap_or_else(|| "unknown".to_string());
        let mut prof = ColumnProfiler::new(name.clone(), dtype);
        if let Some(col) = columns.get(i) {
            for v in col {
                prof.ingest(v);
            }
        }
        col_profiles.push(prof.finalise());
    }

    let correlation_matrix = if include_correlations {
        Some(crate::correlation::correlation_matrix(
            column_names,
            columns,
        ))
    } else {
        None
    };

    Ok(TableProfile {
        table_name: table_name.to_string(),
        row_count,
        columns: col_profiles,
        correlation_matrix,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn welford_matches_naive_variance() {
        let mut m = OnlineMoments::new();
        let xs = [1.0, 2.0, 4.0, 8.0, 16.0];
        for x in xs {
            m.ingest(x);
        }
        let mean = xs.iter().sum::<f64>() / xs.len() as f64;
        let var = xs.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (xs.len() as f64 - 1.0);
        assert!((m.mean - mean).abs() < 1e-9);
        assert!((m.variance().unwrap() - var).abs() < 1e-9);
    }

    #[test]
    fn hll_estimate_within_tolerance() {
        let mut h = HyperLogLog::new();
        for i in 0..10_000u64 {
            h.add(&AnalyticsValue::Int(i as i64));
        }
        let est = h.estimate();
        let err = (est as f64 - 10_000.0).abs() / 10_000.0;
        assert!(err < 0.05, "HLL error too high: {}", err);
    }

    #[test]
    fn reservoir_quantiles_within_tolerance() {
        let mut r = Reservoir::new(0xDEAD_BEEF);
        for i in 0..10_000 {
            r.insert(i as f64);
        }
        let p50 = r.quantile(0.5);
        // 10K inputs into a 65K reservoir means the reservoir holds every
        // input verbatim, so the sample median equals the true median to
        // well under 1%
        assert!((p50 - 5000.0).abs() < 50.0, "p50 = {}", p50);
    }

    #[test]
    fn detects_email_and_uuid() {
        assert_eq!(detect_pattern("foo@example.com"), "email");
        assert_eq!(
            detect_pattern("550e8400-e29b-41d4-a716-446655440000"),
            "uuid"
        );
        assert_eq!(detect_pattern("192.168.1.1"), "ipv4");
    }

    #[test]
    fn full_profile_runs() {
        let mut values = Vec::new();
        for i in 0..1000 {
            values.push(AnalyticsValue::Int(i));
        }
        for _ in 0..50 {
            values.push(AnalyticsValue::Null);
        }
        let p = column_profile("x", "INT64", &values).unwrap();
        assert_eq!(p.null_count, 50);
        assert!(p.distinct_count > 900);
        assert!(p.percentiles.is_some());
        assert!(!p.histogram.is_empty());
    }
}
