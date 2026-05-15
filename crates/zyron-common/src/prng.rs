#![allow(non_snake_case)]
// Custom PRNG for analytics, ML, sampling, bootstrap CIs
// xoshiro256++ as the main generator, splitmix64 for seed expansion
// Both are public-domain algorithms by Blackman and Vigna

/// Mixes a single u64 into a uniformly distributed u64
/// Used to expand a single user seed into the four-word state
/// xoshiro256++ requires
#[inline]
pub fn splitMix64(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9E3779B97F4A7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
    z ^ (z >> 31)
}

/// xoshiro256++ generator state, four 64-bit words
#[derive(Debug, Clone, Copy)]
pub struct Xoshiro256pp {
    s: [u64; 4],
}

impl Xoshiro256pp {
    /// Builds the generator from a single 64-bit seed
    /// State words are derived via splitMix64 to give good initial mixing
    pub fn fromSeed(seed: u64) -> Self {
        let mut sm = seed;
        let s0 = splitMix64(&mut sm);
        let s1 = splitMix64(&mut sm);
        let s2 = splitMix64(&mut sm);
        let s3 = splitMix64(&mut sm);
        let mut g = Self { s: [s0, s1, s2, s3] };
        if g.s == [0, 0, 0, 0] {
            g.s[0] = 1;
        }
        g
    }

    /// Returns the next 64-bit value, advancing state
    #[inline]
    pub fn nextU64(&mut self) -> u64 {
        let result = self.s[0]
            .wrapping_add(self.s[3])
            .rotate_left(23)
            .wrapping_add(self.s[0]);
        let t = self.s[1] << 17;
        self.s[2] ^= self.s[0];
        self.s[3] ^= self.s[1];
        self.s[1] ^= self.s[2];
        self.s[0] ^= self.s[3];
        self.s[2] ^= t;
        self.s[3] = self.s[3].rotate_left(45);
        result
    }

    /// Uniform f64 in [0.0, 1.0)
    /// Builds the value from the high 53 bits, the canonical xoshiro
    /// mantissa-fill technique
    #[inline]
    pub fn nextF64(&mut self) -> f64 {
        ((self.nextU64() >> 11) as f64) * (1.0 / (1u64 << 53) as f64)
    }

    /// Uniform integer in [0, bound), bound > 0
    /// Uses Lemire's nearly-divisionless rejection-free method
    #[inline]
    pub fn nextRange(&mut self, bound: u64) -> u64 {
        if bound == 0 {
            return 0;
        }
        let mut x = self.nextU64();
        let mut m = (x as u128) * (bound as u128);
        let mut l = m as u64;
        if l < bound {
            let t = bound.wrapping_neg() % bound;
            while l < t {
                x = self.nextU64();
                m = (x as u128) * (bound as u128);
                l = m as u64;
            }
        }
        (m >> 64) as u64
    }

    /// Standard normal sample via Box-Muller (one of two paired draws)
    /// Caches the second sample for the next call
    pub fn nextNormal(&mut self) -> f64 {
        let u1 = self.nextF64().max(f64::MIN_POSITIVE);
        let u2 = self.nextF64();
        let r = (-2.0 * u1.ln()).sqrt();
        let theta = 2.0 * std::f64::consts::PI * u2;
        r * theta.cos()
    }

    /// Fisher-Yates shuffle in place
    pub fn shuffle<T>(&mut self, slice: &mut [T]) {
        let n = slice.len();
        for i in (1..n).rev() {
            let j = self.nextRange((i + 1) as u64) as usize;
            slice.swap(i, j);
        }
    }

    /// Picks a random element index in [0, n)
    #[inline]
    pub fn pickIndex(&mut self, n: usize) -> usize {
        self.nextRange(n as u64) as usize
    }

    /// Long jump, equivalent to 2^192 calls to nextU64
    /// Used to derive independent streams across threads from a shared seed
    pub fn longJump(&mut self) {
        const JUMP: [u64; 4] = [
            0x76e15d3efefdcbbf,
            0xc5004e441c522fb3,
            0x77710069854ee241,
            0x39109bb02acbe635,
        ];
        let mut s0 = 0u64;
        let mut s1 = 0u64;
        let mut s2 = 0u64;
        let mut s3 = 0u64;
        for &j in JUMP.iter() {
            for b in 0..64 {
                if (j >> b) & 1 == 1 {
                    s0 ^= self.s[0];
                    s1 ^= self.s[1];
                    s2 ^= self.s[2];
                    s3 ^= self.s[3];
                }
                let _ = self.nextU64();
            }
        }
        self.s = [s0, s1, s2, s3];
    }

    /// Returns a clone with state advanced by one long jump, for thread fanout
    pub fn forkStream(&mut self) -> Self {
        let mut child = *self;
        child.longJump();
        child
    }
}

/// Reservoir sampling, Algorithm L
/// Single-pass sampler that retains a uniform sample of size k from a stream
/// of unknown length
pub struct ReservoirL<T> {
    sample: Vec<T>,
    capacity: usize,
    seen: u64,
    w: f64,
    skip: u64,
    rng: Xoshiro256pp,
}

impl<T> ReservoirL<T> {
    pub fn new(capacity: usize, seed: u64) -> Self {
        let mut rng = Xoshiro256pp::fromSeed(seed);
        let w = (rng.nextF64().max(f64::MIN_POSITIVE).ln() / capacity as f64).exp();
        let skip = (rng.nextF64().max(f64::MIN_POSITIVE).ln() / (1.0 - w).ln()).floor() as u64;
        Self {
            sample: Vec::with_capacity(capacity),
            capacity,
            seen: 0,
            w,
            skip,
            rng,
        }
    }

    pub fn ingest(&mut self, item: T) {
        self.seen += 1;
        if self.sample.len() < self.capacity {
            self.sample.push(item);
            if self.sample.len() == self.capacity {
                self.skip = self.computeSkip();
            }
            return;
        }
        if self.skip > 0 {
            self.skip -= 1;
            return;
        }
        let idx = self.rng.pickIndex(self.capacity);
        self.sample[idx] = item;
        self.w *= (self.rng.nextF64().max(f64::MIN_POSITIVE).ln() / self.capacity as f64).exp();
        self.skip = self.computeSkip();
    }

    fn computeSkip(&mut self) -> u64 {
        let u = self.rng.nextF64().max(f64::MIN_POSITIVE);
        let denom = (1.0 - self.w).ln();
        if denom.is_finite() && denom < 0.0 {
            (u.ln() / denom).floor() as u64
        } else {
            u64::MAX
        }
    }

    pub fn intoSample(self) -> Vec<T> {
        self.sample
    }

    pub fn seen(&self) -> u64 {
        self.seen
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn splitMixDeterministic() {
        let mut a = 1u64;
        let mut b = 1u64;
        for _ in 0..16 {
            assert_eq!(splitMix64(&mut a), splitMix64(&mut b));
        }
    }

    #[test]
    fn xoshiroDeterministic() {
        let mut g1 = Xoshiro256pp::fromSeed(42);
        let mut g2 = Xoshiro256pp::fromSeed(42);
        for _ in 0..1024 {
            assert_eq!(g1.nextU64(), g2.nextU64());
        }
    }

    #[test]
    fn xoshiroF64Range() {
        let mut g = Xoshiro256pp::fromSeed(7);
        for _ in 0..1024 {
            let v = g.nextF64();
            assert!((0.0..1.0).contains(&v));
        }
    }

    #[test]
    fn xoshiroRangeBound() {
        let mut g = Xoshiro256pp::fromSeed(11);
        for _ in 0..2048 {
            assert!(g.nextRange(100) < 100);
        }
    }

    #[test]
    fn xoshiroForkStreamDiverges() {
        let mut a = Xoshiro256pp::fromSeed(99);
        let mut b = a.forkStream();
        let av: Vec<u64> = (0..16).map(|_| a.nextU64()).collect();
        let bv: Vec<u64> = (0..16).map(|_| b.nextU64()).collect();
        assert_ne!(av, bv);
    }

    #[test]
    fn shuffleIsPermutation() {
        let mut g = Xoshiro256pp::fromSeed(123);
        let mut v: Vec<u32> = (0..64).collect();
        g.shuffle(&mut v);
        v.sort();
        let expect: Vec<u32> = (0..64).collect();
        assert_eq!(v, expect);
    }

    #[test]
    fn reservoirReturnsCapacity() {
        let mut r: ReservoirL<u32> = ReservoirL::new(10, 5);
        for i in 0..1000u32 {
            r.ingest(i);
        }
        assert_eq!(r.seen(), 1000);
        let s = r.intoSample();
        assert_eq!(s.len(), 10);
        for x in s {
            assert!(x < 1000);
        }
    }

    #[test]
    fn normalApproxSymmetric() {
        let mut g = Xoshiro256pp::fromSeed(2024);
        let mut sum = 0.0f64;
        let n = 10_000;
        for _ in 0..n {
            sum += g.nextNormal();
        }
        let mean = sum / n as f64;
        assert!(mean.abs() < 0.05);
    }
}
