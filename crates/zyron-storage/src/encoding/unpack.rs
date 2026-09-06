//! Bit unpacking of the sequentially packed residual stream FastLanes
//! writes.
//!
//! Value i of a stream packed at `bw` bits occupies bits `i * bw` up to
//! `(i + 1) * bw` of a little endian bit array. Eight consecutive values
//! therefore occupy exactly `bw` bytes, and every eighth value starts on a
//! byte boundary, which is what lets a group of eight be addressed by a
//! byte offset alone.
//!
//! The vector kernels take a group at a time. Each pair of values comes
//! from one 16 byte load starting on the byte that holds the pair's first
//! bit, a byte shuffle lands each value's bytes at the bottom of its own
//! 64 bit lane, and a per lane right shift and a mask finish it. The load
//! offsets and lane controls depend on the width alone, so they are built
//! once per call and reused for every group. A value starts at most seven
//! bits into its lane, which leaves 57 bits for it, so wider residuals
//! take the scalar path, as do the head and tail of a range that do not
//! fill a group and any group whose loads would reach past the end of the
//! stream.
//!
//! Tiers, widest first: AVX-512 holds a group in one register, AVX2 holds
//! it in two, NEON in four, and the scalar path reads one value at a time.
//! Every tier writes the same bytes, which the differential test below
//! checks against the scalar reader on whatever the host supports.

use std::ops::Range;

/// Values in one vector group
const GROUP: usize = 8;

/// Widest residual the 64 bit lane kernels hold. A value starts up to
/// seven bits into its lane, which leaves 57 bits for the value
const LANE_MAX_WIDTH: u8 = 57;

/// A packed residual stream at one width
#[derive(Clone, Copy)]
pub(super) struct Packed<'a> {
    bytes: &'a [u8],
    bw: u8,
    mask: u64,
}

/// The vector tier the host supports, decided once per process
#[derive(Clone, Copy, PartialEq, Eq)]
enum Tier {
    Scalar,
    #[cfg(target_arch = "x86_64")]
    Avx2,
    #[cfg(target_arch = "x86_64")]
    Avx512,
    #[cfg(target_arch = "aarch64")]
    Neon,
}

fn tier() -> Tier {
    use std::sync::atomic::{AtomicU8, Ordering};
    static CACHE: AtomicU8 = AtomicU8::new(u8::MAX);
    let detect = || -> Tier {
        #[cfg(target_arch = "x86_64")]
        {
            if std::arch::is_x86_feature_detected!("avx512f")
                && std::arch::is_x86_feature_detected!("avx512bw")
            {
                return Tier::Avx512;
            }
            if std::arch::is_x86_feature_detected!("avx2") {
                return Tier::Avx2;
            }
        }
        #[cfg(target_arch = "aarch64")]
        {
            if std::arch::is_aarch64_feature_detected!("neon") {
                return Tier::Neon;
            }
        }
        Tier::Scalar
    };
    let decode = |code: u8| -> Option<Tier> {
        match code {
            0 => Some(Tier::Scalar),
            #[cfg(target_arch = "x86_64")]
            1 => Some(Tier::Avx2),
            #[cfg(target_arch = "x86_64")]
            2 => Some(Tier::Avx512),
            #[cfg(target_arch = "aarch64")]
            3 => Some(Tier::Neon),
            _ => None,
        }
    };
    if let Some(t) = decode(CACHE.load(Ordering::Relaxed)) {
        return t;
    }
    let t = detect();
    let code = match t {
        Tier::Scalar => 0,
        #[cfg(target_arch = "x86_64")]
        Tier::Avx2 => 1,
        #[cfg(target_arch = "x86_64")]
        Tier::Avx512 => 2,
        #[cfg(target_arch = "aarch64")]
        Tier::Neon => 3,
    };
    CACHE.store(code, Ordering::Relaxed);
    t
}

impl<'a> Packed<'a> {
    pub(super) fn new(bytes: &'a [u8], bw: u8) -> Self {
        let mask = if bw >= 64 { u64::MAX } else { (1u64 << bw) - 1 };
        Self { bytes, bw, mask }
    }

    /// Residual of value `i`
    #[inline(always)]
    pub(super) fn at(&self, i: usize) -> u64 {
        unpack_inline(
            self.bytes.as_ptr(),
            self.bytes.len(),
            i as u64 * self.bw as u64,
            self.bw,
            self.mask,
        )
    }

    /// The vector tier to run, or None where the width or the value size
    /// leaves only the scalar path
    #[inline]
    fn vector_tier(&self, value_size: usize) -> Option<Tier> {
        if self.bw > LANE_MAX_WIDTH || !matches!(value_size, 4 | 8) {
            return None;
        }
        match tier() {
            Tier::Scalar => None,
            t => Some(t),
        }
    }

    /// Writes `base + residual` for values `first..first + count` as
    /// `value_size` byte little endian integers at `out`.
    ///
    /// # Safety
    /// `out` has room for `count * value_size` bytes
    pub(super) unsafe fn add_base_into(
        &self,
        first: usize,
        count: usize,
        base: u64,
        out: *mut u8,
        value_size: usize,
    ) {
        // SAFETY: each tier was detected on this host, and the caller's
        // contract on `out` is the kernel's
        unsafe {
            match self.vector_tier(value_size) {
                #[cfg(target_arch = "x86_64")]
                Some(Tier::Avx512) => {
                    avx512::add_base_into(self, first, count, base, out, value_size)
                }
                #[cfg(target_arch = "x86_64")]
                Some(Tier::Avx2) => avx2::add_base_into(self, first, count, base, out, value_size),
                #[cfg(target_arch = "aarch64")]
                Some(Tier::Neon) => neon::add_base_into(self, first, count, base, out, value_size),
                _ => self.add_base_scalar(first..first + count, first, base, out, value_size),
            }
        }
    }

    /// Writes `base + seed + residual[first] + .. + residual[i]` for each
    /// value i of `first..first + count`, and returns the running sum after
    /// the last of them.
    ///
    /// # Safety
    /// `out` has room for `count * value_size` bytes
    pub(super) unsafe fn prefix_into(
        &self,
        first: usize,
        count: usize,
        seed: u64,
        base: u64,
        out: *mut u8,
        value_size: usize,
    ) -> u64 {
        // SAFETY: as in add_base_into
        unsafe {
            match self.vector_tier(value_size) {
                #[cfg(target_arch = "x86_64")]
                Some(Tier::Avx512) => {
                    avx512::prefix_into(self, first, count, seed, base, out, value_size)
                }
                #[cfg(target_arch = "x86_64")]
                Some(Tier::Avx2) => {
                    avx2::prefix_into(self, first, count, seed, base, out, value_size)
                }
                #[cfg(target_arch = "aarch64")]
                Some(Tier::Neon) => {
                    neon::prefix_into(self, first, count, seed, base, out, value_size)
                }
                _ => self.prefix_scalar(first..first + count, first, seed, base, out, value_size),
            }
        }
    }

    /// Writes each value of `first..first + count` as a float: the
    /// residual plus `base`, or with `running` the sum so far of those,
    /// converted to f64 the way `as f64` converts and scaled by `scale`,
    /// stored at the column's float width of 4 or 8 bytes. Returns the sum
    /// after the last value when running.
    ///
    /// The conversion is vectorized on AVX2, which every wider x86 tier
    /// also has, and scalar elsewhere.
    ///
    /// # Safety
    /// `out` has room for `count * value_size` bytes
    #[allow(clippy::too_many_arguments)]
    pub(super) unsafe fn floats_into(
        &self,
        first: usize,
        count: usize,
        base: u64,
        running: Option<u64>,
        scale: f64,
        out: *mut u8,
        value_size: usize,
    ) -> u64 {
        // SAFETY: the tier was detected on this host, and the caller's
        // contract on `out` is the kernel's
        unsafe {
            match self.vector_tier(value_size) {
                #[cfg(target_arch = "x86_64")]
                Some(Tier::Avx512) | Some(Tier::Avx2) => {
                    avx2::floats_into(self, first, count, base, running, scale, out, value_size)
                }
                _ => self.floats_scalar(
                    first..first + count,
                    first,
                    base,
                    running,
                    scale,
                    out,
                    value_size,
                ),
            }
        }
    }

    /// Sum of the residuals of values `first..first + count`
    pub(super) fn sum(&self, first: usize, count: usize) -> u64 {
        // SAFETY: each tier was detected on this host
        unsafe {
            match self.vector_tier(8) {
                #[cfg(target_arch = "x86_64")]
                Some(Tier::Avx512) => avx512::sum(self, first, count),
                #[cfg(target_arch = "x86_64")]
                Some(Tier::Avx2) => avx2::sum(self, first, count),
                #[cfg(target_arch = "aarch64")]
                Some(Tier::Neon) => neon::sum(self, first, count),
                _ => self.sum_scalar(first..first + count),
            }
        }
    }

    /// Residuals of values `first..first + count` as a new vector
    pub(super) fn unpack_vec(&self, first: usize, count: usize) -> Vec<u64> {
        let mut out = Vec::new();
        self.unpack_reusing(first, count, &mut out);
        out
    }

    /// Residuals of values `first..first + count` into `out`, replacing
    /// its contents and reusing its allocation
    pub(super) fn unpack_reusing(&self, first: usize, count: usize, out: &mut Vec<u64>) {
        out.clear();
        out.reserve(count);
        // SAFETY: the kernel writes every one of the count slots before the
        // length exposes them
        unsafe {
            self.add_base_into(first, count, 0, out.as_mut_ptr() as *mut u8, 8);
            out.set_len(count);
        }
    }

    /// The groups whose eight values all lie in `first..first + count` and
    /// whose loads, reaching `reach` bytes past a group's start, all lie
    /// inside the stream, with the scalar spans on either side of them
    fn split(
        &self,
        first: usize,
        count: usize,
        reach: usize,
    ) -> (Range<usize>, Range<usize>, Range<usize>) {
        let end = first + count;
        let first_group = first.div_ceil(GROUP);
        let loadable = if self.bytes.len() >= reach {
            (self.bytes.len() - reach) / self.bw as usize + 1
        } else {
            0
        };
        let end_group = (end / GROUP).min(loadable).max(first_group);
        let head = first..(first_group * GROUP).min(end);
        let tail = (end_group * GROUP).max(head.end)..end;
        (head, first_group..end_group, tail)
    }

    /// Value i of `values` goes to slot `i - origin`
    #[inline]
    unsafe fn add_base_scalar(
        &self,
        values: Range<usize>,
        origin: usize,
        base: u64,
        out: *mut u8,
        value_size: usize,
    ) {
        for i in values {
            // SAFETY: slot `i - origin` is inside the caller's buffer
            unsafe { store(out, i - origin, value_size, self.at(i).wrapping_add(base)) };
        }
    }

    #[inline]
    unsafe fn prefix_scalar(
        &self,
        values: Range<usize>,
        origin: usize,
        seed: u64,
        base: u64,
        out: *mut u8,
        value_size: usize,
    ) -> u64 {
        let mut acc = seed;
        for i in values {
            acc = acc.wrapping_add(self.at(i));
            // SAFETY: slot `i - origin` is inside the caller's buffer
            unsafe { store(out, i - origin, value_size, acc.wrapping_add(base)) };
        }
        acc
    }

    #[inline]
    fn sum_scalar(&self, values: Range<usize>) -> u64 {
        values.fold(0u64, |acc, i| acc.wrapping_add(self.at(i)))
    }

    #[inline]
    #[allow(clippy::too_many_arguments)]
    unsafe fn floats_scalar(
        &self,
        values: Range<usize>,
        origin: usize,
        base: u64,
        mut running: Option<u64>,
        scale: f64,
        out: *mut u8,
        value_size: usize,
    ) -> u64 {
        for i in values {
            let with_base = self.at(i).wrapping_add(base);
            let value = match running.as_mut() {
                Some(acc) => {
                    *acc = acc.wrapping_add(with_base);
                    *acc
                }
                None => with_base,
            };
            // SAFETY: slot `i - origin` is inside the caller's buffer
            unsafe { store_float(out, i - origin, value_size, value as i64 as f64 * scale) };
        }
        running.unwrap_or(0)
    }
}

/// Stores `v` at slot `i` at a float width of 4 or 8 bytes
#[inline(always)]
unsafe fn store_float(out: *mut u8, i: usize, value_size: usize, v: f64) {
    // SAFETY: the caller guarantees slot `i` at this width is inside `out`
    unsafe {
        if value_size == 4 {
            (out as *mut f32).add(i).write_unaligned(v as f32);
        } else {
            (out as *mut f64).add(i).write_unaligned(v);
        }
    }
}

/// Stores `v` at slot `i` as a `value_size` byte little endian integer,
/// zero extended past eight bytes
#[inline(always)]
unsafe fn store(out: *mut u8, i: usize, value_size: usize, v: u64) {
    // SAFETY: the caller guarantees slot `i` at this width is inside `out`
    unsafe {
        match value_size {
            8 => (out as *mut u64).add(i).write_unaligned(v),
            4 => (out as *mut u32).add(i).write_unaligned(v as u32),
            2 => (out as *mut u16).add(i).write_unaligned(v as u16),
            1 => out.add(i).write(v as u8),
            _ => {
                let at = out.add(i * value_size);
                let low = value_size.min(8);
                std::ptr::copy_nonoverlapping(v.to_le_bytes().as_ptr(), at, low);
                if value_size > 8 {
                    std::ptr::write_bytes(at.add(8), 0, value_size - 8);
                }
            }
        }
    }
}

/// One value out of the stream by bit offset. The common case is a single
/// unaligned 64 bit load, shift and mask, a value crossing the eighth byte
/// picks up a ninth, and the last few bytes of the stream are read through
/// a zero padded copy so the load never runs past it
#[inline(always)]
fn unpack_inline(
    packed_ptr: *const u8,
    packed_len: usize,
    bit_offset: u64,
    bit_width: u8,
    mask: u64,
) -> u64 {
    let byte_idx = (bit_offset >> 3) as usize;
    let bit_idx = (bit_offset & 7) as u32;

    if byte_idx + 8 <= packed_len {
        // SAFETY: eight bytes from byte_idx are inside the stream
        let raw = unsafe { (packed_ptr.add(byte_idx) as *const u64).read_unaligned() };
        let val = (raw >> bit_idx) & mask;

        if bit_idx + bit_width as u32 > 64 {
            if byte_idx + 9 <= packed_len {
                // SAFETY: the ninth byte is inside the stream
                let hi = unsafe { *packed_ptr.add(byte_idx + 8) } as u64;
                return (val | (hi << (64 - bit_idx))) & mask;
            }
        } else {
            return val;
        }
    }

    let mut buf = [0u8; 8];
    let available = packed_len.saturating_sub(byte_idx).min(8);
    // SAFETY: `available` bytes from byte_idx are inside the stream
    unsafe {
        std::ptr::copy_nonoverlapping(packed_ptr.add(byte_idx), buf.as_mut_ptr(), available);
    }
    let raw = u64::from_le_bytes(buf);
    (raw >> bit_idx) & mask
}

/// Load offset, byte shuffle and lane shifts for one pair of values of a
/// group. Pair `h` holds values `2h` and `2h + 1`, its load starts on the
/// byte holding the first bit of value `2h`, and each value's eight bytes
/// are gathered to the bottom of its own 64 bit lane
#[derive(Clone, Copy)]
struct Pair {
    load: usize,
    shuffle: [u8; 16],
    shift: [u8; 2],
}

impl Pair {
    fn new(bw: usize, h: usize) -> Self {
        let lead = 2 * h * bw / 8;
        let mut shuffle = [0u8; 16];
        let mut shift = [0u8; 2];
        for j in 0..2 {
            let bit = (2 * h + j) * bw;
            let byte = bit / 8 - lead;
            shift[j] = (bit % 8) as u8;
            for k in 0..8 {
                shuffle[8 * j + k] = (byte + k) as u8;
            }
        }
        Self {
            load: lead,
            shuffle,
            shift,
        }
    }

    /// Bytes past a group's start the last pair's load reaches
    fn reach(bw: usize) -> usize {
        Pair::new(bw, GROUP / 2 - 1).load + 16
    }
}

#[cfg(target_arch = "x86_64")]
mod avx2 {
    use super::{GROUP, Packed, Pair};
    use std::arch::x86_64::*;

    /// Lane controls for the two quads of a group
    struct Plan {
        load: [[usize; 2]; 2],
        shuffle: [__m256i; 2],
        shift: [__m256i; 2],
        mask: __m256i,
        reach: usize,
    }

    impl Plan {
        #[target_feature(enable = "avx2")]
        fn new(bw: u8, mask: u64) -> Self {
            let bw = bw as usize;
            let pairs = [0, 1, 2, 3].map(|h| Pair::new(bw, h));
            let quad = |q: usize| {
                let (lo, hi) = (pairs[2 * q], pairs[2 * q + 1]);
                let mut shuffle = [0u8; 32];
                shuffle[..16].copy_from_slice(&lo.shuffle);
                shuffle[16..].copy_from_slice(&hi.shuffle);
                let shift = [
                    lo.shift[0] as i64,
                    lo.shift[1] as i64,
                    hi.shift[0] as i64,
                    hi.shift[1] as i64,
                ];
                // SAFETY: both arrays are 32 bytes on the stack
                unsafe {
                    (
                        [lo.load, hi.load],
                        _mm256_loadu_si256(shuffle.as_ptr() as *const __m256i),
                        _mm256_loadu_si256(shift.as_ptr() as *const __m256i),
                    )
                }
            };
            let (load0, shuffle0, shift0) = quad(0);
            let (load1, shuffle1, shift1) = quad(1);
            Self {
                load: [load0, load1],
                shuffle: [shuffle0, shuffle1],
                shift: [shift0, shift1],
                mask: _mm256_set1_epi64x(mask as i64),
                reach: Pair::reach(bw),
            }
        }

        /// The four residuals of quad `q` of the group starting at `p`, one
        /// per 64 bit lane.
        ///
        /// # Safety
        /// `reach` bytes from `p` are readable
        #[inline]
        #[target_feature(enable = "avx2")]
        unsafe fn quad(&self, p: *const u8, q: usize) -> __m256i {
            // SAFETY: both loads lie within `reach` bytes of `p`
            let (lo, hi) = unsafe {
                (
                    _mm_loadu_si128(p.add(self.load[q][0]) as *const __m128i),
                    _mm_loadu_si128(p.add(self.load[q][1]) as *const __m128i),
                )
            };
            let v = _mm256_inserti128_si256::<1>(_mm256_castsi128_si256(lo), hi);
            let v = _mm256_shuffle_epi8(v, self.shuffle[q]);
            _mm256_and_si256(_mm256_srlv_epi64(v, self.shift[q]), self.mask)
        }
    }

    /// Widest residual the 32 bit lane kernel holds. A value starts up to
    /// seven bits into its lane, which leaves 25 bits for the value, and
    /// the four values of a half then span at most 14 bytes of its load
    const NARROW_MAX_WIDTH: u8 = 25;

    /// Lane controls for a whole group in eight 32 bit lanes. Each half
    /// of the register takes four values from one 16 byte load
    struct Narrow {
        load: [usize; 2],
        shuffle: __m256i,
        shift: __m256i,
        mask: __m256i,
        reach: usize,
    }

    impl Narrow {
        #[target_feature(enable = "avx2")]
        fn new(bw: u8, mask: u64) -> Self {
            let bw = bw as usize;
            let mut load = [0usize; 2];
            let mut shuffle = [0u8; 32];
            let mut shift = [0i32; 8];
            for h in 0..2 {
                let lead = 4 * h * bw / 8;
                load[h] = lead;
                for j in 0..4 {
                    let bit = (4 * h + j) * bw - lead * 8;
                    shift[4 * h + j] = (bit % 8) as i32;
                    for k in 0..4 {
                        shuffle[16 * h + 4 * j + k] = (bit / 8 + k) as u8;
                    }
                }
            }
            // SAFETY: both arrays are 32 bytes on the stack
            let (shuffle, shift) = unsafe {
                (
                    _mm256_loadu_si256(shuffle.as_ptr() as *const __m256i),
                    _mm256_loadu_si256(shift.as_ptr() as *const __m256i),
                )
            };
            Self {
                load,
                shuffle,
                shift,
                mask: _mm256_set1_epi32(mask as i32),
                reach: load[1] + 16,
            }
        }

        /// The eight residuals of the group starting at `p`, one per 32
        /// bit lane.
        ///
        /// # Safety
        /// `reach` bytes from `p` are readable
        #[inline]
        #[target_feature(enable = "avx2")]
        unsafe fn group(&self, p: *const u8) -> __m256i {
            // SAFETY: both loads lie within `reach` bytes of `p`
            let (lo, hi) = unsafe {
                (
                    _mm_loadu_si128(p.add(self.load[0]) as *const __m128i),
                    _mm_loadu_si128(p.add(self.load[1]) as *const __m128i),
                )
            };
            let v = _mm256_inserti128_si256::<1>(_mm256_castsi128_si256(lo), hi);
            let v = _mm256_shuffle_epi8(v, self.shuffle);
            _mm256_and_si256(_mm256_srlv_epi32(v, self.shift), self.mask)
        }
    }

    /// The running sum within eight 32 bit lanes, and the total in every
    /// lane. Eight residuals of at most 25 bits sum below 2^28, so the
    /// lanes never wrap
    #[inline]
    #[target_feature(enable = "avx2")]
    fn prefix8(v: __m256i) -> (__m256i, __m256i) {
        let x = _mm256_add_epi32(v, _mm256_slli_si256::<4>(v));
        let x = _mm256_add_epi32(x, _mm256_slli_si256::<8>(x));
        // The low half's total carries into every lane of the high half
        let carry = _mm256_blend_epi32::<0b1111_0000>(
            _mm256_setzero_si256(),
            _mm256_permutevar8x32_epi32(x, _mm256_set1_epi32(3)),
        );
        let x = _mm256_add_epi32(x, carry);
        (x, _mm256_permutevar8x32_epi32(x, _mm256_set1_epi32(7)))
    }

    /// Eight 32 bit lanes as two quads of 64 bit lanes, zero extended
    #[inline]
    #[target_feature(enable = "avx2")]
    fn widen(x: __m256i) -> (__m256i, __m256i) {
        (
            _mm256_cvtepu32_epi64(_mm256_castsi256_si128(x)),
            _mm256_cvtepu32_epi64(_mm256_extracti128_si256::<1>(x)),
        )
    }

    /// The low 32 bits of each lane of `a` then of `b`, as eight 32 bit
    /// lanes
    #[inline]
    #[target_feature(enable = "avx2")]
    fn low_halves(a: __m256i, b: __m256i) -> __m256i {
        let low_dwords = _mm256_setr_epi32(0, 2, 4, 6, 0, 2, 4, 6);
        _mm256_blend_epi32::<0b1111_0000>(
            _mm256_permutevar8x32_epi32(a, low_dwords),
            _mm256_permutevar8x32_epi32(b, low_dwords),
        )
    }

    /// Stores the two quads of one group at slot `slot`
    #[inline]
    #[target_feature(enable = "avx2")]
    unsafe fn store_group(out: *mut u8, slot: usize, value_size: usize, a: __m256i, b: __m256i) {
        // SAFETY: the group's eight slots are inside the caller's buffer
        unsafe {
            if value_size == 8 {
                let o = (out as *mut u64).add(slot);
                _mm256_storeu_si256(o as *mut __m256i, a);
                _mm256_storeu_si256(o.add(4) as *mut __m256i, b);
            } else {
                let o = (out as *mut u32).add(slot);
                _mm256_storeu_si256(o as *mut __m256i, low_halves(a, b));
            }
        }
    }

    /// The running sum within a quad, and its total in every lane
    #[inline]
    #[target_feature(enable = "avx2")]
    fn prefix4(v: __m256i) -> (__m256i, __m256i) {
        // Within each 128 bit half, lane one picks up lane zero
        let x = _mm256_add_epi64(v, _mm256_slli_si256::<8>(v));
        // The low half's total carries into both lanes of the high half
        let carry = _mm256_blend_epi32::<0b1111_0000>(
            _mm256_setzero_si256(),
            _mm256_permute4x64_epi64::<0b01_01_01_01>(x),
        );
        let x = _mm256_add_epi64(x, carry);
        (x, _mm256_permute4x64_epi64::<0b11_11_11_11>(x))
    }

    #[target_feature(enable = "avx2")]
    pub(super) unsafe fn add_base_into(
        s: &Packed<'_>,
        first: usize,
        count: usize,
        base: u64,
        out: *mut u8,
        value_size: usize,
    ) {
        let base_v = _mm256_set1_epi64x(base as i64);
        let bw = s.bw as usize;
        let p = s.bytes.as_ptr();
        if s.bw <= NARROW_MAX_WIDTH {
            let plan = Narrow::new(s.bw, s.mask);
            let (head, groups, tail) = s.split(first, count, plan.reach);
            // SAFETY: every slot written lies in `first..first + count`,
            // and every group's loads lie inside the stream
            unsafe {
                s.add_base_scalar(head, first, base, out, value_size);
                if value_size == 4 {
                    // The output is the low 32 bits, so the base joins in
                    // the lanes the values already occupy
                    let base32 = _mm256_set1_epi32(base as i32);
                    for gi in groups {
                        let v = _mm256_add_epi32(plan.group(p.add(gi * bw)), base32);
                        let o = (out as *mut u32).add(gi * GROUP - first);
                        _mm256_storeu_si256(o as *mut __m256i, v);
                    }
                } else {
                    for gi in groups {
                        let (a, b) = widen(plan.group(p.add(gi * bw)));
                        store_group(
                            out,
                            gi * GROUP - first,
                            value_size,
                            _mm256_add_epi64(a, base_v),
                            _mm256_add_epi64(b, base_v),
                        );
                    }
                }
                s.add_base_scalar(tail, first, base, out, value_size);
            }
            return;
        }
        let plan = Plan::new(s.bw, s.mask);
        let (head, groups, tail) = s.split(first, count, plan.reach);
        // SAFETY: as above
        unsafe {
            s.add_base_scalar(head, first, base, out, value_size);
            for gi in groups {
                let gp = p.add(gi * bw);
                let a = _mm256_add_epi64(plan.quad(gp, 0), base_v);
                let b = _mm256_add_epi64(plan.quad(gp, 1), base_v);
                store_group(out, gi * GROUP - first, value_size, a, b);
            }
            s.add_base_scalar(tail, first, base, out, value_size);
        }
    }

    #[target_feature(enable = "avx2")]
    pub(super) unsafe fn prefix_into(
        s: &Packed<'_>,
        first: usize,
        count: usize,
        seed: u64,
        base: u64,
        out: *mut u8,
        value_size: usize,
    ) -> u64 {
        let base_v = _mm256_set1_epi64x(base as i64);
        let bw = s.bw as usize;
        let p = s.bytes.as_ptr();
        if s.bw <= NARROW_MAX_WIDTH {
            let plan = Narrow::new(s.bw, s.mask);
            let (head, groups, tail) = s.split(first, count, plan.reach);
            // SAFETY: every slot written lies in `first..first + count`,
            // and every group's loads lie inside the stream
            unsafe {
                let mut acc = s.prefix_scalar(head, first, seed, base, out, value_size);
                if !groups.is_empty() {
                    let mut acc_v = _mm256_set1_epi64x(acc as i64);
                    for gi in groups {
                        // The group's own running sum is independent of
                        // the accumulator, so the chain through the group
                        // is one add
                        let (x, total) = prefix8(plan.group(p.add(gi * bw)));
                        let (a, b) = widen(x);
                        let a = _mm256_add_epi64(a, acc_v);
                        let b = _mm256_add_epi64(b, acc_v);
                        acc_v = _mm256_add_epi64(
                            acc_v,
                            _mm256_cvtepu32_epi64(_mm256_castsi256_si128(total)),
                        );
                        store_group(
                            out,
                            gi * GROUP - first,
                            value_size,
                            _mm256_add_epi64(a, base_v),
                            _mm256_add_epi64(b, base_v),
                        );
                    }
                    acc = _mm256_extract_epi64::<0>(acc_v) as u64;
                }
                return s.prefix_scalar(tail, first, acc, base, out, value_size);
            }
        }
        let plan = Plan::new(s.bw, s.mask);
        let (head, groups, tail) = s.split(first, count, plan.reach);
        // SAFETY: as above
        unsafe {
            let mut acc = s.prefix_scalar(head, first, seed, base, out, value_size);
            if !groups.is_empty() {
                let mut acc_v = _mm256_set1_epi64x(acc as i64);
                for gi in groups {
                    let gp = p.add(gi * bw);
                    // Each quad's own running sum is independent of the
                    // accumulator, so the chain through the group is two
                    // adds rather than eight
                    let (xa, total_a) = prefix4(plan.quad(gp, 0));
                    let (xb, total_b) = prefix4(plan.quad(gp, 1));
                    let a = _mm256_add_epi64(xa, acc_v);
                    acc_v = _mm256_add_epi64(acc_v, total_a);
                    let b = _mm256_add_epi64(xb, acc_v);
                    acc_v = _mm256_add_epi64(acc_v, total_b);
                    store_group(
                        out,
                        gi * GROUP - first,
                        value_size,
                        _mm256_add_epi64(a, base_v),
                        _mm256_add_epi64(b, base_v),
                    );
                }
                acc = _mm256_extract_epi64::<0>(acc_v) as u64;
            }
            s.prefix_scalar(tail, first, acc, base, out, value_size)
        }
    }

    #[target_feature(enable = "avx2")]
    pub(super) unsafe fn sum(s: &Packed<'_>, first: usize, count: usize) -> u64 {
        let plan = Plan::new(s.bw, s.mask);
        let (head, groups, tail) = s.split(first, count, plan.reach);
        let mut total = s.sum_scalar(head);
        if !groups.is_empty() {
            let mut v = _mm256_setzero_si256();
            let bw = s.bw as usize;
            let p = s.bytes.as_ptr();
            for gi in groups {
                // SAFETY: the group's loads lie inside the stream
                let (a, b) = unsafe {
                    let gp = p.add(gi * bw);
                    (plan.quad(gp, 0), plan.quad(gp, 1))
                };
                v = _mm256_add_epi64(v, _mm256_add_epi64(a, b));
            }
            let mut lanes = [0u64; 4];
            // SAFETY: the array is 32 bytes on the stack
            unsafe { _mm256_storeu_si256(lanes.as_mut_ptr() as *mut __m256i, v) };
            for lane in lanes {
                total = total.wrapping_add(lane);
            }
        }
        total.wrapping_add(s.sum_scalar(tail))
    }

    /// Each 64 bit lane as an f64, rounded once the way `as f64` rounds.
    ///
    /// AVX2 has no signed 64 bit conversion, so the lane is split: the low
    /// 32 bits are placed under the exponent of 2^52 and that offset
    /// subtracted, which is exact, and the high 32 bits convert as a
    /// signed 32 bit integer, also exact. Scaling the high part by 2^32 is
    /// exact too, so the one rounding is the final add, which is the
    /// rounding a direct conversion makes
    #[inline]
    #[target_feature(enable = "avx2")]
    fn lanes_to_f64(x: __m256i) -> __m256d {
        let two_52 = 4_503_599_627_370_496.0;
        let low = _mm256_or_si256(
            _mm256_and_si256(x, _mm256_set1_epi64x(0xFFFF_FFFF)),
            _mm256_set1_epi64x(0x4330_0000_0000_0000),
        );
        let low = _mm256_sub_pd(_mm256_castsi256_pd(low), _mm256_set1_pd(two_52));
        let high = _mm256_castsi256_si128(_mm256_permutevar8x32_epi32(
            x,
            _mm256_setr_epi32(1, 3, 5, 7, 1, 3, 5, 7),
        ));
        let high = _mm256_cvtepi32_pd(high);
        _mm256_add_pd(_mm256_mul_pd(high, _mm256_set1_pd(4_294_967_296.0)), low)
    }

    /// Stores two quads of floats at slot `slot`
    #[inline]
    #[target_feature(enable = "avx2")]
    unsafe fn store_floats(out: *mut u8, slot: usize, value_size: usize, a: __m256d, b: __m256d) {
        // SAFETY: the group's eight slots are inside the caller's buffer
        unsafe {
            if value_size == 8 {
                let o = (out as *mut f64).add(slot);
                _mm256_storeu_pd(o, a);
                _mm256_storeu_pd(o.add(4), b);
            } else {
                _mm256_storeu_ps(
                    (out as *mut f32).add(slot),
                    _mm256_set_m128(_mm256_cvtpd_ps(b), _mm256_cvtpd_ps(a)),
                );
            }
        }
    }

    /// Each 64 bit lane as an f64 when every lane lies in `-2^51..2^51`.
    /// Offset by 2^51 the lane is a non negative integer below 2^52, which
    /// placed under the exponent of 2^52 is that float exactly, and taking
    /// the offset back out is exact too
    #[inline]
    #[target_feature(enable = "avx2")]
    fn small_lanes_to_f64(x: __m256i) -> __m256d {
        let biased = _mm256_add_epi64(x, _mm256_set1_epi64x(1 << 51));
        let bits = _mm256_or_si256(biased, _mm256_set1_epi64x(0x4330_0000_0000_0000));
        _mm256_sub_pd(
            _mm256_castsi256_pd(bits),
            _mm256_set1_pd(4_503_599_627_370_496.0 + 2_251_799_813_685_248.0),
        )
    }

    /// Whether every value a call writes lies in the range the short
    /// conversion handles. With the base folded in a residual lies in
    /// `base..=base + mask`, and a running sum moves by at most the larger
    /// magnitude of those per row
    fn fits_short_conversion(
        s: &Packed<'_>,
        count: usize,
        base: u64,
        running: Option<u64>,
    ) -> bool {
        const LIMIT: u64 = 1 << 51;
        let base = base as i64;
        let Some(top) = base.checked_add(s.mask as i64) else {
            return false;
        };
        let step = base.unsigned_abs().max(top.unsigned_abs());
        match running {
            None => step < LIMIT,
            Some(seed) => (count as u64)
                .checked_mul(step)
                .and_then(|span| span.checked_add((seed as i64).unsigned_abs()))
                .is_some_and(|reach| reach < LIMIT),
        }
    }

    #[target_feature(enable = "avx2")]
    #[allow(clippy::too_many_arguments)]
    pub(super) unsafe fn floats_into(
        s: &Packed<'_>,
        first: usize,
        count: usize,
        base: u64,
        running: Option<u64>,
        scale: f64,
        out: *mut u8,
        value_size: usize,
    ) -> u64 {
        let short = fits_short_conversion(s, count, base, running);
        let convert = |x: __m256i| -> __m256d {
            if short {
                small_lanes_to_f64(x)
            } else {
                lanes_to_f64(x)
            }
        };
        let base_v = _mm256_set1_epi64x(base as i64);
        let scale_v = _mm256_set1_pd(scale);
        let bw = s.bw as usize;
        let p = s.bytes.as_ptr();
        if s.bw <= NARROW_MAX_WIDTH {
            let plan = Narrow::new(s.bw, s.mask);
            let (head, groups, tail) = s.split(first, count, plan.reach);
            // SAFETY: every slot written lies in `first..first + count`,
            // and every group's loads lie inside the stream
            unsafe {
                let mut acc = s.floats_scalar(head, first, base, running, scale, out, value_size);
                if !groups.is_empty() {
                    let mut acc_v = _mm256_set1_epi64x(acc as i64);
                    // A running value picks up one base per row, so the
                    // rows of a group add one to eight bases on top of the
                    // running sum of the residuals alone
                    let b = base as i64;
                    let steps_a = _mm256_setr_epi64x(
                        b,
                        b.wrapping_mul(2),
                        b.wrapping_mul(3),
                        b.wrapping_mul(4),
                    );
                    let steps_b = _mm256_setr_epi64x(
                        b.wrapping_mul(5),
                        b.wrapping_mul(6),
                        b.wrapping_mul(7),
                        b.wrapping_mul(8),
                    );
                    let eight = _mm256_set1_epi64x(b.wrapping_mul(8));
                    for gi in groups {
                        let g = plan.group(p.add(gi * bw));
                        let (a, b) = if running.is_some() {
                            let (x, total) = prefix8(g);
                            let (a, b) = widen(x);
                            let a = _mm256_add_epi64(_mm256_add_epi64(a, steps_a), acc_v);
                            let b = _mm256_add_epi64(_mm256_add_epi64(b, steps_b), acc_v);
                            acc_v = _mm256_add_epi64(
                                acc_v,
                                _mm256_add_epi64(
                                    _mm256_cvtepu32_epi64(_mm256_castsi256_si128(total)),
                                    eight,
                                ),
                            );
                            (a, b)
                        } else {
                            let (a, b) = widen(g);
                            (_mm256_add_epi64(a, base_v), _mm256_add_epi64(b, base_v))
                        };
                        store_floats(
                            out,
                            gi * GROUP - first,
                            value_size,
                            _mm256_mul_pd(convert(a), scale_v),
                            _mm256_mul_pd(convert(b), scale_v),
                        );
                    }
                    if running.is_some() {
                        acc = _mm256_extract_epi64::<0>(acc_v) as u64;
                    }
                }
                return s.floats_scalar(
                    tail,
                    first,
                    base,
                    running.map(|_| acc),
                    scale,
                    out,
                    value_size,
                );
            }
        }
        let plan = Plan::new(s.bw, s.mask);
        let (head, groups, tail) = s.split(first, count, plan.reach);
        // SAFETY: as above
        unsafe {
            let mut acc = s.floats_scalar(head, first, base, running, scale, out, value_size);
            if !groups.is_empty() {
                let mut acc_v = _mm256_set1_epi64x(acc as i64);
                for gi in groups {
                    let gp = p.add(gi * bw);
                    let mut a = _mm256_add_epi64(plan.quad(gp, 0), base_v);
                    let mut b = _mm256_add_epi64(plan.quad(gp, 1), base_v);
                    if running.is_some() {
                        let (xa, total_a) = prefix4(a);
                        let (xb, total_b) = prefix4(b);
                        a = _mm256_add_epi64(xa, acc_v);
                        acc_v = _mm256_add_epi64(acc_v, total_a);
                        b = _mm256_add_epi64(xb, acc_v);
                        acc_v = _mm256_add_epi64(acc_v, total_b);
                    }
                    store_floats(
                        out,
                        gi * GROUP - first,
                        value_size,
                        _mm256_mul_pd(convert(a), scale_v),
                        _mm256_mul_pd(convert(b), scale_v),
                    );
                }
                if running.is_some() {
                    acc = _mm256_extract_epi64::<0>(acc_v) as u64;
                }
            }
            s.floats_scalar(
                tail,
                first,
                base,
                running.map(|_| acc),
                scale,
                out,
                value_size,
            )
        }
    }
}

#[cfg(target_arch = "x86_64")]
mod avx512 {
    use super::{GROUP, Packed, Pair};
    use std::arch::x86_64::*;

    /// Lane controls for a whole group in one register
    struct Plan {
        load: [usize; 4],
        shuffle: __m512i,
        shift: __m512i,
        mask: __m512i,
        reach: usize,
    }

    impl Plan {
        #[target_feature(enable = "avx512f,avx512bw")]
        fn new(bw: u8, mask: u64) -> Self {
            let bw = bw as usize;
            let pairs = [0, 1, 2, 3].map(|h| Pair::new(bw, h));
            let mut shuffle = [0u8; 64];
            let mut shift = [0i64; 8];
            for (h, pair) in pairs.iter().enumerate() {
                shuffle[16 * h..16 * h + 16].copy_from_slice(&pair.shuffle);
                shift[2 * h] = pair.shift[0] as i64;
                shift[2 * h + 1] = pair.shift[1] as i64;
            }
            // SAFETY: both arrays are 64 bytes on the stack
            let (shuffle, shift) = unsafe {
                (
                    _mm512_loadu_si512(shuffle.as_ptr() as *const __m512i),
                    _mm512_loadu_si512(shift.as_ptr() as *const __m512i),
                )
            };
            Self {
                load: pairs.map(|pair| pair.load),
                shuffle,
                shift,
                mask: _mm512_set1_epi64(mask as i64),
                reach: Pair::reach(bw),
            }
        }

        /// The eight residuals of the group starting at `p`, one per lane.
        ///
        /// # Safety
        /// `reach` bytes from `p` are readable
        #[inline]
        #[target_feature(enable = "avx512f,avx512bw")]
        unsafe fn group(&self, p: *const u8) -> __m512i {
            // SAFETY: every load lies within `reach` bytes of `p`
            let v = unsafe {
                let l0 = _mm_loadu_si128(p.add(self.load[0]) as *const __m128i);
                let l1 = _mm_loadu_si128(p.add(self.load[1]) as *const __m128i);
                let l2 = _mm_loadu_si128(p.add(self.load[2]) as *const __m128i);
                let l3 = _mm_loadu_si128(p.add(self.load[3]) as *const __m128i);
                let v = _mm512_inserti32x4::<1>(_mm512_castsi128_si512(l0), l1);
                let v = _mm512_inserti32x4::<2>(v, l2);
                _mm512_inserti32x4::<3>(v, l3)
            };
            let v = _mm512_shuffle_epi8(v, self.shuffle);
            _mm512_and_si512(_mm512_srlv_epi64(v, self.shift), self.mask)
        }
    }

    /// Stores one group at slot `slot`
    #[inline]
    #[target_feature(enable = "avx512f,avx512bw")]
    unsafe fn store_group(out: *mut u8, slot: usize, value_size: usize, v: __m512i) {
        // SAFETY: the group's eight slots are inside the caller's buffer
        unsafe {
            if value_size == 8 {
                _mm512_storeu_si512((out as *mut u64).add(slot) as *mut __m512i, v);
            } else {
                _mm256_storeu_si256(
                    (out as *mut u32).add(slot) as *mut __m256i,
                    _mm512_cvtepi64_epi32(v),
                );
            }
        }
    }

    /// The running sum within a group, and its total in every lane
    #[inline]
    #[target_feature(enable = "avx512f,avx512bw")]
    fn prefix8(v: __m512i) -> (__m512i, __m512i) {
        let zero = _mm512_setzero_si512();
        // Three doubling steps, each adding the lanes one, two and four
        // places below
        let x = _mm512_add_epi64(v, _mm512_alignr_epi64::<7>(v, zero));
        let x = _mm512_add_epi64(x, _mm512_alignr_epi64::<6>(x, zero));
        let x = _mm512_add_epi64(x, _mm512_alignr_epi64::<4>(x, zero));
        (x, _mm512_permutexvar_epi64(_mm512_set1_epi64(7), x))
    }

    #[target_feature(enable = "avx512f,avx512bw")]
    pub(super) unsafe fn add_base_into(
        s: &Packed<'_>,
        first: usize,
        count: usize,
        base: u64,
        out: *mut u8,
        value_size: usize,
    ) {
        let plan = Plan::new(s.bw, s.mask);
        let (head, groups, tail) = s.split(first, count, plan.reach);
        // SAFETY: every slot written lies in `first..first + count`, and
        // every group's loads lie inside the stream
        unsafe {
            s.add_base_scalar(head, first, base, out, value_size);
            let base_v = _mm512_set1_epi64(base as i64);
            let bw = s.bw as usize;
            let p = s.bytes.as_ptr();
            for gi in groups {
                let v = _mm512_add_epi64(plan.group(p.add(gi * bw)), base_v);
                store_group(out, gi * GROUP - first, value_size, v);
            }
            s.add_base_scalar(tail, first, base, out, value_size);
        }
    }

    #[target_feature(enable = "avx512f,avx512bw")]
    pub(super) unsafe fn prefix_into(
        s: &Packed<'_>,
        first: usize,
        count: usize,
        seed: u64,
        base: u64,
        out: *mut u8,
        value_size: usize,
    ) -> u64 {
        let plan = Plan::new(s.bw, s.mask);
        let (head, groups, tail) = s.split(first, count, plan.reach);
        // SAFETY: as in add_base_into
        unsafe {
            let mut acc = s.prefix_scalar(head, first, seed, base, out, value_size);
            if !groups.is_empty() {
                let base_v = _mm512_set1_epi64(base as i64);
                let mut acc_v = _mm512_set1_epi64(acc as i64);
                let bw = s.bw as usize;
                let p = s.bytes.as_ptr();
                for gi in groups {
                    let (x, total) = prefix8(plan.group(p.add(gi * bw)));
                    let v = _mm512_add_epi64(x, acc_v);
                    acc_v = _mm512_add_epi64(acc_v, total);
                    store_group(
                        out,
                        gi * GROUP - first,
                        value_size,
                        _mm512_add_epi64(v, base_v),
                    );
                }
                acc = _mm_cvtsi128_si64(_mm512_castsi512_si128(acc_v)) as u64;
            }
            s.prefix_scalar(tail, first, acc, base, out, value_size)
        }
    }

    #[target_feature(enable = "avx512f,avx512bw")]
    pub(super) unsafe fn sum(s: &Packed<'_>, first: usize, count: usize) -> u64 {
        let plan = Plan::new(s.bw, s.mask);
        let (head, groups, tail) = s.split(first, count, plan.reach);
        let mut total = s.sum_scalar(head);
        if !groups.is_empty() {
            let mut v = _mm512_setzero_si512();
            let bw = s.bw as usize;
            let p = s.bytes.as_ptr();
            for gi in groups {
                // SAFETY: the group's loads lie inside the stream
                let g = unsafe { plan.group(p.add(gi * bw)) };
                v = _mm512_add_epi64(v, g);
            }
            total = total.wrapping_add(_mm512_reduce_add_epi64(v) as u64);
        }
        total.wrapping_add(s.sum_scalar(tail))
    }
}

#[cfg(target_arch = "aarch64")]
mod neon {
    use super::{GROUP, Packed, Pair};
    use std::arch::aarch64::*;

    /// Lane controls for the four pairs of a group
    struct Plan {
        load: [usize; 4],
        shuffle: [uint8x16_t; 4],
        /// Negative, because NEON shifts right by a negative left shift
        shift: [int64x2_t; 4],
        mask: uint64x2_t,
        reach: usize,
    }

    impl Plan {
        #[target_feature(enable = "neon")]
        fn new(bw: u8, mask: u64) -> Self {
            let bw = bw as usize;
            let pairs = [0, 1, 2, 3].map(|h| Pair::new(bw, h));
            // SAFETY: the arrays are 16 bytes on the stack
            let shuffle = unsafe { pairs.map(|pair| vld1q_u8(pair.shuffle.as_ptr())) };
            let shift = unsafe {
                pairs.map(|pair| {
                    let neg = [-(pair.shift[0] as i64), -(pair.shift[1] as i64)];
                    vld1q_s64(neg.as_ptr())
                })
            };
            Self {
                load: pairs.map(|pair| pair.load),
                shuffle,
                shift,
                mask: vdupq_n_u64(mask),
                reach: Pair::reach(bw),
            }
        }

        /// The two residuals of pair `h` of the group starting at `p`.
        ///
        /// # Safety
        /// `reach` bytes from `p` are readable
        #[inline]
        #[target_feature(enable = "neon")]
        unsafe fn pair(&self, p: *const u8, h: usize) -> uint64x2_t {
            // SAFETY: the load lies within `reach` bytes of `p`
            let v = unsafe { vld1q_u8(p.add(self.load[h])) };
            let v = vreinterpretq_u64_u8(vqtbl1q_u8(v, self.shuffle[h]));
            vandq_u64(vshlq_u64(v, self.shift[h]), self.mask)
        }
    }

    /// Stores two pairs at slot `slot`
    #[inline]
    #[target_feature(enable = "neon")]
    unsafe fn store_pairs(
        out: *mut u8,
        slot: usize,
        value_size: usize,
        a: uint64x2_t,
        b: uint64x2_t,
    ) {
        // SAFETY: the four slots are inside the caller's buffer
        unsafe {
            if value_size == 8 {
                let o = (out as *mut u64).add(slot);
                vst1q_u64(o, a);
                vst1q_u64(o.add(2), b);
            } else {
                vst1q_u32(
                    (out as *mut u32).add(slot),
                    vcombine_u32(vmovn_u64(a), vmovn_u64(b)),
                );
            }
        }
    }

    /// The running sum within a pair, and its total in both lanes
    #[inline]
    #[target_feature(enable = "neon")]
    fn prefix2(v: uint64x2_t) -> (uint64x2_t, uint64x2_t) {
        let x = vaddq_u64(v, vextq_u64::<1>(vdupq_n_u64(0), v));
        (x, vdupq_laneq_u64::<1>(x))
    }

    #[target_feature(enable = "neon")]
    pub(super) unsafe fn add_base_into(
        s: &Packed<'_>,
        first: usize,
        count: usize,
        base: u64,
        out: *mut u8,
        value_size: usize,
    ) {
        let plan = Plan::new(s.bw, s.mask);
        let (head, groups, tail) = s.split(first, count, plan.reach);
        // SAFETY: every slot written lies in `first..first + count`, and
        // every group's loads lie inside the stream
        unsafe {
            s.add_base_scalar(head, first, base, out, value_size);
            let base_v = vdupq_n_u64(base);
            let bw = s.bw as usize;
            let p = s.bytes.as_ptr();
            for gi in groups {
                let gp = p.add(gi * bw);
                let slot = gi * GROUP - first;
                store_pairs(
                    out,
                    slot,
                    value_size,
                    vaddq_u64(plan.pair(gp, 0), base_v),
                    vaddq_u64(plan.pair(gp, 1), base_v),
                );
                store_pairs(
                    out,
                    slot + 4,
                    value_size,
                    vaddq_u64(plan.pair(gp, 2), base_v),
                    vaddq_u64(plan.pair(gp, 3), base_v),
                );
            }
            s.add_base_scalar(tail, first, base, out, value_size);
        }
    }

    #[target_feature(enable = "neon")]
    pub(super) unsafe fn prefix_into(
        s: &Packed<'_>,
        first: usize,
        count: usize,
        seed: u64,
        base: u64,
        out: *mut u8,
        value_size: usize,
    ) -> u64 {
        let plan = Plan::new(s.bw, s.mask);
        let (head, groups, tail) = s.split(first, count, plan.reach);
        // SAFETY: as in add_base_into
        unsafe {
            let mut acc = s.prefix_scalar(head, first, seed, base, out, value_size);
            if !groups.is_empty() {
                let base_v = vdupq_n_u64(base);
                let mut acc_v = vdupq_n_u64(acc);
                let bw = s.bw as usize;
                let p = s.bytes.as_ptr();
                for gi in groups {
                    let gp = p.add(gi * bw);
                    let slot = gi * GROUP - first;
                    let mut sums = [vdupq_n_u64(0); 4];
                    for (h, sum) in sums.iter_mut().enumerate() {
                        let (x, total) = prefix2(plan.pair(gp, h));
                        *sum = vaddq_u64(vaddq_u64(x, acc_v), base_v);
                        acc_v = vaddq_u64(acc_v, total);
                    }
                    store_pairs(out, slot, value_size, sums[0], sums[1]);
                    store_pairs(out, slot + 4, value_size, sums[2], sums[3]);
                }
                acc = vgetq_lane_u64::<0>(acc_v);
            }
            s.prefix_scalar(tail, first, acc, base, out, value_size)
        }
    }

    #[target_feature(enable = "neon")]
    pub(super) unsafe fn sum(s: &Packed<'_>, first: usize, count: usize) -> u64 {
        let plan = Plan::new(s.bw, s.mask);
        let (head, groups, tail) = s.split(first, count, plan.reach);
        let mut total = s.sum_scalar(head);
        if !groups.is_empty() {
            let mut v = vdupq_n_u64(0);
            let bw = s.bw as usize;
            let p = s.bytes.as_ptr();
            for gi in groups {
                // SAFETY: the group's loads lie inside the stream
                unsafe {
                    let gp = p.add(gi * bw);
                    for h in 0..4 {
                        v = vaddq_u64(v, plan.pair(gp, h));
                    }
                }
            }
            total = total.wrapping_add(vaddvq_u64(v));
        }
        total.wrapping_add(s.sum_scalar(tail))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A stream of `n` values at `bw` bits from a fixed seed, and the
    /// values it holds
    fn stream(bw: u8, n: usize, seed: u64) -> (Vec<u8>, Vec<u64>) {
        let mut state = seed;
        let mut next = move || {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            state
        };
        let mask = if bw >= 64 { u64::MAX } else { (1u64 << bw) - 1 };
        let values: Vec<u64> = (0..n).map(|_| next() & mask).collect();
        let mut packed = vec![0u8; (n as u64 * bw as u64).div_ceil(8) as usize];
        for (i, &v) in values.iter().enumerate() {
            let mut bit = i as u64 * bw as u64;
            let mut left = bw as u32;
            let mut val = v;
            while left > 0 {
                let byte = (bit / 8) as usize;
                let at = (bit % 8) as u32;
                let take = (8 - at).min(left);
                packed[byte] |= ((val & ((1u64 << take) - 1)) as u8) << at;
                val >>= take;
                bit += take as u64;
                left -= take;
            }
        }
        (packed, values)
    }

    /// The slot's integer, checking that a slot wider than eight bytes is
    /// zero above them
    fn read(out: &[u8], value_size: usize, i: usize) -> u64 {
        let slot = &out[i * value_size..(i + 1) * value_size];
        let low = value_size.min(8);
        assert!(slot[low..].iter().all(|b| *b == 0), "high bytes not zero");
        let mut buf = [0u8; 8];
        buf[..low].copy_from_slice(&slot[..low]);
        u64::from_le_bytes(buf)
    }

    /// Every width and value size against the scalar reader, over ranges
    /// that start and end off group boundaries, that fit inside one group,
    /// and that run to the end of the stream where the vector loads must
    /// hand over
    #[test]
    fn kernels_match_the_scalar_reader_at_every_width() {
        for bw in 1u8..=64 {
            let n = 1000 + bw as usize * 7;
            let (packed, values) = stream(bw, n, 0x9E37_79B9_7F4A_7C15 ^ bw as u64);
            let s = Packed::new(&packed, bw);
            for (i, &v) in values.iter().enumerate() {
                assert_eq!(s.at(i), v, "scalar read bw={bw} i={i}");
            }
            let ranges = [
                (0usize, n),
                (0, 1),
                (3, 2),
                (5, 13),
                (7, 64),
                (8, 24),
                (13, 500),
                (n - 9, 9),
                (n - 1, 1),
                (n - 70, 70),
            ];
            for &(first, count) in &ranges {
                for value_size in [1usize, 2, 4, 8, 16] {
                    let base = 0x0123_4567_89AB_CDEFu64;
                    let keep = if value_size >= 8 {
                        u64::MAX
                    } else {
                        (1u64 << (8 * value_size)) - 1
                    };
                    let mut out = vec![0u8; count * value_size];
                    // SAFETY: out holds count * value_size bytes
                    unsafe { s.add_base_into(first, count, base, out.as_mut_ptr(), value_size) };
                    for k in 0..count {
                        assert_eq!(
                            read(&out, value_size, k),
                            values[first + k].wrapping_add(base) & keep,
                            "add_base bw={bw} first={first} count={count} size={value_size} k={k}"
                        );
                    }

                    let seed = 77u64;
                    let mut out = vec![0u8; count * value_size];
                    // SAFETY: out holds count * value_size bytes
                    let got = unsafe {
                        s.prefix_into(first, count, seed, base, out.as_mut_ptr(), value_size)
                    };
                    let mut acc = seed;
                    for k in 0..count {
                        acc = acc.wrapping_add(values[first + k]);
                        assert_eq!(
                            read(&out, value_size, k),
                            acc.wrapping_add(base) & keep,
                            "prefix bw={bw} first={first} count={count} size={value_size} k={k}"
                        );
                    }
                    assert_eq!(got, acc, "prefix total bw={bw} first={first} count={count}");
                }

                let expected = values[first..first + count]
                    .iter()
                    .fold(0u64, |a, &v| a.wrapping_add(v));
                assert_eq!(s.sum(first, count), expected, "sum bw={bw} first={first}");

                assert_eq!(
                    s.unpack_vec(first, count),
                    &values[first..first + count],
                    "unpack_vec bw={bw} first={first}"
                );
            }
        }
    }

    /// The float sink against a scalar reference at every width, with a
    /// base that pushes values past the 2^32 boundary the lane split
    /// crosses, negative values, both running modes and both float widths
    #[test]
    fn float_sink_matches_the_scalar_conversion() {
        for bw in 1u8..=64 {
            let n = 500 + bw as usize * 3;
            let (packed, values) = stream(bw, n, 0xA5A5_5A5A_1234_5678 ^ bw as u64);
            let s = Packed::new(&packed, bw);
            let bases = [0u64, (-5_000_000_000i64) as u64, 1u64 << 40, u64::MAX - 7];
            for &base in &bases {
                for running in [None, Some(12_345u64)] {
                    for value_size in [4usize, 8] {
                        let scale = 0.01f64;
                        let (first, count) = (3usize, n - 5);
                        let mut out = vec![0u8; count * value_size];
                        // SAFETY: out holds count * value_size bytes
                        let got = unsafe {
                            s.floats_into(
                                first,
                                count,
                                base,
                                running,
                                scale,
                                out.as_mut_ptr(),
                                value_size,
                            )
                        };
                        let mut acc = running;
                        for k in 0..count {
                            let with_base = values[first + k].wrapping_add(base);
                            let value = match acc.as_mut() {
                                Some(a) => {
                                    *a = a.wrapping_add(with_base);
                                    *a
                                }
                                None => with_base,
                            };
                            let want = value as i64 as f64 * scale;
                            let slot = &out[k * value_size..(k + 1) * value_size];
                            if value_size == 8 {
                                let mut buf = [0u8; 8];
                                buf.copy_from_slice(slot);
                                assert_eq!(
                                    f64::from_le_bytes(buf).to_bits(),
                                    want.to_bits(),
                                    "f64 bw={bw} base={base:#x} running={running:?} k={k}"
                                );
                            } else {
                                let mut buf = [0u8; 4];
                                buf.copy_from_slice(slot);
                                assert_eq!(
                                    f32::from_le_bytes(buf).to_bits(),
                                    (want as f32).to_bits(),
                                    "f32 bw={bw} base={base:#x} running={running:?} k={k}"
                                );
                            }
                        }
                        assert_eq!(got, acc.unwrap_or(0), "running total bw={bw}");
                    }
                }
            }
        }
    }

    /// A stream too short for even one vector group still decodes
    #[test]
    fn short_streams_take_the_scalar_path() {
        for bw in 1u8..=64 {
            for n in 0..=17 {
                let (packed, values) = stream(bw, n, 3 + bw as u64);
                let s = Packed::new(&packed, bw);
                let mut out = vec![0u8; n * 8];
                // SAFETY: out holds n * 8 bytes
                unsafe { s.add_base_into(0, n, 5, out.as_mut_ptr(), 8) };
                for (k, &v) in values.iter().enumerate() {
                    assert_eq!(read(&out, 8, k), v.wrapping_add(5), "bw={bw} n={n} k={k}");
                }
                assert_eq!(
                    s.sum(0, n),
                    values.iter().fold(0u64, |a, &v| a.wrapping_add(v))
                );
            }
        }
    }
}
