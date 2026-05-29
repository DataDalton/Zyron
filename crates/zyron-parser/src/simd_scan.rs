//! SIMD-accelerated byte scanning for the lexer hot loops.
//!
//! Each public function dispatches at runtime to the widest instruction set
//! the CPU supports, following the project standard:
//! AVX-512 -> AVX2 -> NEON -> scalar. Every SIMD path is verified against the
//! scalar implementation in the unit tests, so the scalar version is the
//! source of truth for behavior.
//!
//! The functions answer "how far does this run extend": find the first byte
//! that ends an identifier, a digit run, a whitespace run, or matches a
//! delimiter. They take a starting offset and return an absolute index into
//! `bytes` (clamped to `bytes.len()` when the run reaches the end).

// ---------------------------------------------------------------------------
// Public dispatch entry points
// ---------------------------------------------------------------------------

/// Returns the index of the first byte at or after `start` that is not an
/// identifier-continuation character (`a-z A-Z 0-9 _`).
#[inline]
pub fn identifier_end(bytes: &[u8], start: usize) -> usize {
    debug_assert!(start <= bytes.len(), "scan start past end of input");
    #[cfg(target_arch = "x86_64")]
    {
        if has_avx512() {
            return unsafe { identifier_end_avx512(bytes, start) };
        }
        if has_avx2() {
            return unsafe { identifier_end_avx2(bytes, start) };
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        if std::arch::is_aarch64_feature_detected!("neon") {
            return unsafe { identifier_end_neon(bytes, start) };
        }
    }
    identifier_end_scalar(bytes, start)
}

/// Returns the index of the first digit-run end at or after `start`
/// (first byte not in `0-9`).
#[inline]
pub fn digits_end(bytes: &[u8], start: usize) -> usize {
    debug_assert!(start <= bytes.len(), "scan start past end of input");
    #[cfg(target_arch = "x86_64")]
    {
        if has_avx512() {
            return unsafe { digits_end_avx512(bytes, start) };
        }
        if has_avx2() {
            return unsafe { digits_end_avx2(bytes, start) };
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        if std::arch::is_aarch64_feature_detected!("neon") {
            return unsafe { digits_end_neon(bytes, start) };
        }
    }
    digits_end_scalar(bytes, start)
}

/// Returns the index of the first ASCII-whitespace-run end at or after
/// `start` (first byte that is not space, tab, CR, or LF).
#[inline]
pub fn whitespace_end(bytes: &[u8], start: usize) -> usize {
    debug_assert!(start <= bytes.len(), "scan start past end of input");
    #[cfg(target_arch = "x86_64")]
    {
        if has_avx2() {
            return unsafe { whitespace_end_avx2(bytes, start) };
        }
    }
    whitespace_end_scalar(bytes, start)
}

/// Returns the index of the first occurrence of `needle` at or after `start`,
/// or `bytes.len()` if not found. Used to locate string-literal terminators.
#[inline]
pub fn find_byte(bytes: &[u8], start: usize, needle: u8) -> usize {
    debug_assert!(start <= bytes.len(), "scan start past end of input");
    #[cfg(target_arch = "x86_64")]
    {
        if has_avx512() {
            return unsafe { find_byte_avx512(bytes, start, needle) };
        }
        if has_avx2() {
            return unsafe { find_byte_avx2(bytes, start, needle) };
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        if std::arch::is_aarch64_feature_detected!("neon") {
            return unsafe { find_byte_neon(bytes, start, needle) };
        }
    }
    find_byte_scalar(bytes, start, needle)
}

// ---------------------------------------------------------------------------
// Feature detection (cached)
// ---------------------------------------------------------------------------

#[cfg(target_arch = "x86_64")]
#[inline]
fn has_avx512() -> bool {
    use std::sync::atomic::{AtomicU8, Ordering};
    static CACHE: AtomicU8 = AtomicU8::new(2); // 2 = unknown
    match CACHE.load(Ordering::Relaxed) {
        0 => false,
        1 => true,
        _ => {
            let v = is_x86_feature_detected!("avx512f") && is_x86_feature_detected!("avx512bw");
            CACHE.store(v as u8, Ordering::Relaxed);
            v
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[inline]
fn has_avx2() -> bool {
    use std::sync::atomic::{AtomicU8, Ordering};
    static CACHE: AtomicU8 = AtomicU8::new(2);
    match CACHE.load(Ordering::Relaxed) {
        0 => false,
        1 => true,
        _ => {
            let v = is_x86_feature_detected!("avx2");
            CACHE.store(v as u8, Ordering::Relaxed);
            v
        }
    }
}

// ---------------------------------------------------------------------------
// Scalar implementations (source of truth)
// ---------------------------------------------------------------------------

#[inline]
fn is_ident_byte(b: u8) -> bool {
    let lower = b | 0x20;
    lower.is_ascii_lowercase() || b.is_ascii_digit() || b == b'_'
}

#[inline]
fn is_ws_byte(b: u8) -> bool {
    matches!(b, b' ' | b'\t' | b'\r' | b'\n')
}

fn identifier_end_scalar(bytes: &[u8], start: usize) -> usize {
    let mut i = start;
    while i < bytes.len() && is_ident_byte(bytes[i]) {
        i += 1;
    }
    i
}

fn digits_end_scalar(bytes: &[u8], start: usize) -> usize {
    let mut i = start;
    while i < bytes.len() && bytes[i].is_ascii_digit() {
        i += 1;
    }
    i
}

fn whitespace_end_scalar(bytes: &[u8], start: usize) -> usize {
    let mut i = start;
    while i < bytes.len() && is_ws_byte(bytes[i]) {
        i += 1;
    }
    i
}

fn find_byte_scalar(bytes: &[u8], start: usize, needle: u8) -> usize {
    let mut i = start;
    while i < bytes.len() && bytes[i] != needle {
        i += 1;
    }
    i
}

// ---------------------------------------------------------------------------
// x86_64 AVX2 / AVX-512
//
// Each function is `unsafe fn` gated by `#[target_feature]`. The single
// body-level `unsafe` block covers the raw unaligned loads and pointer
// arithmetic; the intrinsics themselves are callable without further
// `unsafe` because the enabling target feature is in scope.
// ---------------------------------------------------------------------------

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn find_byte_avx2(bytes: &[u8], start: usize, needle: u8) -> usize {
    use std::arch::x86_64::*;
    unsafe {
        let n = bytes.len();
        let mut i = start;
        let v_needle = _mm256_set1_epi8(needle as i8);
        while i + 32 <= n {
            let chunk = _mm256_loadu_si256(bytes.as_ptr().add(i) as *const __m256i);
            let eq = _mm256_cmpeq_epi8(chunk, v_needle);
            let mask = _mm256_movemask_epi8(eq) as u32;
            if mask != 0 {
                return i + mask.trailing_zeros() as usize;
            }
            i += 32;
        }
        find_byte_scalar(bytes, i, needle)
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512bw")]
unsafe fn find_byte_avx512(bytes: &[u8], start: usize, needle: u8) -> usize {
    use std::arch::x86_64::*;
    unsafe {
        let n = bytes.len();
        let mut i = start;
        let v_needle = _mm512_set1_epi8(needle as i8);
        while i + 64 <= n {
            let chunk = _mm512_loadu_si512(bytes.as_ptr().add(i) as *const __m512i);
            let mask = _mm512_cmpeq_epi8_mask(chunk, v_needle);
            if mask != 0 {
                return i + mask.trailing_zeros() as usize;
            }
            i += 64;
        }
        find_byte_avx2(bytes, i, needle)
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn identifier_end_avx2(bytes: &[u8], start: usize) -> usize {
    use std::arch::x86_64::*;
    unsafe {
        let n = bytes.len();
        let mut i = start;
        let lower_a = _mm256_set1_epi8((b'a' - 1) as i8);
        let lower_z = _mm256_set1_epi8((b'z' + 1) as i8);
        let dig_0 = _mm256_set1_epi8((b'0' - 1) as i8);
        let dig_9 = _mm256_set1_epi8((b'9' + 1) as i8);
        let underscore = _mm256_set1_epi8(b'_' as i8);
        let v0x20 = _mm256_set1_epi8(0x20);
        while i + 32 <= n {
            let chunk = _mm256_loadu_si256(bytes.as_ptr().add(i) as *const __m256i);
            let lower = _mm256_or_si256(chunk, v0x20);
            let is_alpha = _mm256_and_si256(
                _mm256_cmpgt_epi8(lower, lower_a),
                _mm256_cmpgt_epi8(lower_z, lower),
            );
            let is_digit = _mm256_and_si256(
                _mm256_cmpgt_epi8(chunk, dig_0),
                _mm256_cmpgt_epi8(dig_9, chunk),
            );
            let is_us = _mm256_cmpeq_epi8(chunk, underscore);
            let is_ident = _mm256_or_si256(_mm256_or_si256(is_alpha, is_digit), is_us);
            let mask = _mm256_movemask_epi8(is_ident) as u32;
            if mask != 0xFFFF_FFFF {
                return i + (!mask).trailing_zeros() as usize;
            }
            i += 32;
        }
        identifier_end_scalar(bytes, i)
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512bw")]
unsafe fn identifier_end_avx512(bytes: &[u8], start: usize) -> usize {
    use std::arch::x86_64::*;
    unsafe {
        let n = bytes.len();
        let mut i = start;
        let lower_a = _mm512_set1_epi8((b'a' - 1) as i8);
        let lower_z = _mm512_set1_epi8((b'z' + 1) as i8);
        let dig_0 = _mm512_set1_epi8((b'0' - 1) as i8);
        let dig_9 = _mm512_set1_epi8((b'9' + 1) as i8);
        let underscore = _mm512_set1_epi8(b'_' as i8);
        let v0x20 = _mm512_set1_epi8(0x20);
        while i + 64 <= n {
            let chunk = _mm512_loadu_si512(bytes.as_ptr().add(i) as *const __m512i);
            let lower = _mm512_or_si512(chunk, v0x20);
            let is_alpha =
                _mm512_cmpgt_epi8_mask(lower, lower_a) & _mm512_cmpgt_epi8_mask(lower_z, lower);
            let is_digit =
                _mm512_cmpgt_epi8_mask(chunk, dig_0) & _mm512_cmpgt_epi8_mask(dig_9, chunk);
            let is_us = _mm512_cmpeq_epi8_mask(chunk, underscore);
            let is_ident = is_alpha | is_digit | is_us;
            if is_ident != u64::MAX {
                return i + (!is_ident).trailing_zeros() as usize;
            }
            i += 64;
        }
        identifier_end_avx2(bytes, i)
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn digits_end_avx2(bytes: &[u8], start: usize) -> usize {
    use std::arch::x86_64::*;
    unsafe {
        let n = bytes.len();
        let mut i = start;
        let dig_0 = _mm256_set1_epi8((b'0' - 1) as i8);
        let dig_9 = _mm256_set1_epi8((b'9' + 1) as i8);
        while i + 32 <= n {
            let chunk = _mm256_loadu_si256(bytes.as_ptr().add(i) as *const __m256i);
            let is_digit = _mm256_and_si256(
                _mm256_cmpgt_epi8(chunk, dig_0),
                _mm256_cmpgt_epi8(dig_9, chunk),
            );
            let mask = _mm256_movemask_epi8(is_digit) as u32;
            if mask != 0xFFFF_FFFF {
                return i + (!mask).trailing_zeros() as usize;
            }
            i += 32;
        }
        digits_end_scalar(bytes, i)
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512bw")]
unsafe fn digits_end_avx512(bytes: &[u8], start: usize) -> usize {
    use std::arch::x86_64::*;
    unsafe {
        let n = bytes.len();
        let mut i = start;
        let dig_0 = _mm512_set1_epi8((b'0' - 1) as i8);
        let dig_9 = _mm512_set1_epi8((b'9' + 1) as i8);
        while i + 64 <= n {
            let chunk = _mm512_loadu_si512(bytes.as_ptr().add(i) as *const __m512i);
            let is_digit =
                _mm512_cmpgt_epi8_mask(chunk, dig_0) & _mm512_cmpgt_epi8_mask(dig_9, chunk);
            if is_digit != u64::MAX {
                return i + (!is_digit).trailing_zeros() as usize;
            }
            i += 64;
        }
        digits_end_avx2(bytes, i)
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn whitespace_end_avx2(bytes: &[u8], start: usize) -> usize {
    use std::arch::x86_64::*;
    unsafe {
        let n = bytes.len();
        let mut i = start;
        let sp = _mm256_set1_epi8(b' ' as i8);
        let tab = _mm256_set1_epi8(b'\t' as i8);
        let cr = _mm256_set1_epi8(b'\r' as i8);
        let lf = _mm256_set1_epi8(b'\n' as i8);
        while i + 32 <= n {
            let chunk = _mm256_loadu_si256(bytes.as_ptr().add(i) as *const __m256i);
            let is_ws = _mm256_or_si256(
                _mm256_or_si256(_mm256_cmpeq_epi8(chunk, sp), _mm256_cmpeq_epi8(chunk, tab)),
                _mm256_or_si256(_mm256_cmpeq_epi8(chunk, cr), _mm256_cmpeq_epi8(chunk, lf)),
            );
            let mask = _mm256_movemask_epi8(is_ws) as u32;
            if mask != 0xFFFF_FFFF {
                return i + (!mask).trailing_zeros() as usize;
            }
            i += 32;
        }
        whitespace_end_scalar(bytes, i)
    }
}

// ---------------------------------------------------------------------------
// aarch64 NEON
// ---------------------------------------------------------------------------

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn find_byte_neon(bytes: &[u8], start: usize, needle: u8) -> usize {
    use std::arch::aarch64::*;
    unsafe {
        let n = bytes.len();
        let mut i = start;
        let v_needle = vdupq_n_u8(needle);
        while i + 16 <= n {
            let chunk = vld1q_u8(bytes.as_ptr().add(i));
            let eq = vceqq_u8(chunk, v_needle);
            if vmaxvq_u8(eq) != 0 {
                return find_byte_scalar(bytes, i, needle);
            }
            i += 16;
        }
        find_byte_scalar(bytes, i, needle)
    }
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn identifier_end_neon(bytes: &[u8], start: usize) -> usize {
    use std::arch::aarch64::*;
    unsafe {
        let n = bytes.len();
        let mut i = start;
        while i + 16 <= n {
            let chunk = vld1q_u8(bytes.as_ptr().add(i));
            let lower = vorrq_u8(chunk, vdupq_n_u8(0x20));
            let is_alpha = vandq_u8(
                vcgeq_u8(lower, vdupq_n_u8(b'a')),
                vcleq_u8(lower, vdupq_n_u8(b'z')),
            );
            let is_digit = vandq_u8(
                vcgeq_u8(chunk, vdupq_n_u8(b'0')),
                vcleq_u8(chunk, vdupq_n_u8(b'9')),
            );
            let is_us = vceqq_u8(chunk, vdupq_n_u8(b'_'));
            let is_ident = vorrq_u8(vorrq_u8(is_alpha, is_digit), is_us);
            if vminvq_u8(is_ident) == 0 {
                return identifier_end_scalar(bytes, i);
            }
            i += 16;
        }
        identifier_end_scalar(bytes, i)
    }
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn digits_end_neon(bytes: &[u8], start: usize) -> usize {
    use std::arch::aarch64::*;
    unsafe {
        let n = bytes.len();
        let mut i = start;
        while i + 16 <= n {
            let chunk = vld1q_u8(bytes.as_ptr().add(i));
            let is_digit = vandq_u8(
                vcgeq_u8(chunk, vdupq_n_u8(b'0')),
                vcleq_u8(chunk, vdupq_n_u8(b'9')),
            );
            if vminvq_u8(is_digit) == 0 {
                return digits_end_scalar(bytes, i);
            }
            i += 16;
        }
        digits_end_scalar(bytes, i)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identifier_end_matches_scalar() {
        let cases: &[&str] = &[
            "orders_0 VALUES",
            "abcdefghijklmnopqrstuvwxyz0123456789_ABCDEFGHIJKLMNOP rest",
            "x",
            "_underscore123,next",
            "0123456789012345678901234567890123 done",
            "",
            "   leading",
        ];
        for c in cases {
            let b = c.as_bytes();
            for start in 0..=b.len() {
                assert_eq!(
                    identifier_end(b, start),
                    identifier_end_scalar(b, start),
                    "ident mismatch in {:?} at {}",
                    c,
                    start
                );
            }
        }
    }

    #[test]
    fn digits_end_matches_scalar() {
        let cases: &[&str] = &[
            "12345abc",
            "9999999999999999999999999999999999999 x",
            "0.13",
            "",
            "abc",
        ];
        for c in cases {
            let b = c.as_bytes();
            for start in 0..=b.len() {
                assert_eq!(digits_end(b, start), digits_end_scalar(b, start));
            }
        }
    }

    #[test]
    fn whitespace_end_matches_scalar() {
        let cases: &[&str] = &[
            "    \t\r\n   x",
            "no_ws",
            "                                  thirtyplus",
            "",
        ];
        for c in cases {
            let b = c.as_bytes();
            for start in 0..=b.len() {
                assert_eq!(whitespace_end(b, start), whitespace_end_scalar(b, start));
            }
        }
    }

    #[test]
    fn find_byte_matches_scalar() {
        let cases: &[(&str, u8)] = &[
            ("hello 'world' rest", b'\''),
            ("no quote here", b'\''),
            ("aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa'b", b'\''),
            ("", b'x'),
            ("find,the,comma", b','),
        ];
        for (c, needle) in cases {
            let b = c.as_bytes();
            for start in 0..=b.len() {
                assert_eq!(
                    find_byte(b, start, *needle),
                    find_byte_scalar(b, start, *needle),
                    "find_byte mismatch in {:?} needle {} at {}",
                    c,
                    *needle as char,
                    start
                );
            }
        }
    }

    #[test]
    fn long_random_inputs_match_scalar() {
        let mut state: u64 = 0x1234_5678_9abc_def0;
        let mut buf = Vec::with_capacity(4096);
        for _ in 0..4096 {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            buf.push((state >> 33) as u8 & 0x7f);
        }
        for start in [0usize, 1, 7, 31, 63, 100, 1000] {
            assert_eq!(
                identifier_end(&buf, start),
                identifier_end_scalar(&buf, start)
            );
            assert_eq!(digits_end(&buf, start), digits_end_scalar(&buf, start));
            assert_eq!(
                whitespace_end(&buf, start),
                whitespace_end_scalar(&buf, start)
            );
            assert_eq!(
                find_byte(&buf, start, b'A'),
                find_byte_scalar(&buf, start, b'A')
            );
        }
    }

    // Exercises the full 0..=255 byte range (including >=0x80, where the SIMD
    // signed-compare paths must still agree with scalar) and every length /
    // start across the AVX-512(64) -> AVX2(32) -> NEON(16) -> scalar handoff
    // boundaries.
    #[test]
    fn high_bytes_and_lane_boundaries_match_scalar() {
        let mut state: u64 = 0xdead_beef_0bad_f00d;
        let mut next = || {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (state >> 33) as u8 // full 0..=255, not masked to ASCII
        };
        // Lengths spanning each lane width and its +/-1 neighbours.
        for len in 0usize..=130 {
            let buf: Vec<u8> = (0..len).map(|_| next()).collect();
            for start in 0..=len {
                assert_eq!(
                    identifier_end(&buf, start),
                    identifier_end_scalar(&buf, start),
                    "identifier_end len={len} start={start}"
                );
                assert_eq!(
                    digits_end(&buf, start),
                    digits_end_scalar(&buf, start),
                    "digits_end len={len} start={start}"
                );
                assert_eq!(
                    whitespace_end(&buf, start),
                    whitespace_end_scalar(&buf, start),
                    "whitespace_end len={len} start={start}"
                );
                for needle in [b'A', 0u8, 0xFF] {
                    assert_eq!(
                        find_byte(&buf, start, needle),
                        find_byte_scalar(&buf, start, needle),
                        "find_byte len={len} start={start} needle={needle}"
                    );
                }
            }
        }
    }
}
