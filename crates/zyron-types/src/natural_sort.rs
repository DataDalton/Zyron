//! Natural sort, version compare, IP sort, path sort, custom order
//!
//! SortKey produces bytes that compare lexicographically equivalent to the
//! desired natural order. Numeric runs are length prefixed so longer numbers
//! sort after shorter, text runs are tagged distinctly so they sort after
//! numbers at the same position. SortKey uses an inline-small-string
//! representation for keys up to 23 bytes to avoid heap allocation for the
//! common short input case

use std::cmp::Ordering;

// ---------------------------------------------------------------------------
// Inline-small-string SortKey
// ---------------------------------------------------------------------------

const INLINE_CAP: usize = 23;

/// Compact byte buffer that stays inline up to 23 bytes and spills to the heap
/// beyond that. Comparison is byte-lexicographic which by construction matches
/// the natural-sort ordering of the input
#[derive(Clone, Eq)]
pub enum SortKey {
    Inline { len: u8, bytes: [u8; INLINE_CAP] },
    Heap(Vec<u8>),
}

impl SortKey {
    fn from_bytes(bytes: Vec<u8>) -> Self {
        if bytes.len() <= INLINE_CAP {
            let mut buf = [0u8; INLINE_CAP];
            buf[..bytes.len()].copy_from_slice(&bytes);
            SortKey::Inline {
                len: bytes.len() as u8,
                bytes: buf,
            }
        } else {
            SortKey::Heap(bytes)
        }
    }

    pub fn as_bytes(&self) -> &[u8] {
        match self {
            SortKey::Inline { len, bytes } => &bytes[..*len as usize],
            SortKey::Heap(v) => v.as_slice(),
        }
    }

    pub fn into_bytes(self) -> Vec<u8> {
        match self {
            SortKey::Inline { len, bytes } => bytes[..len as usize].to_vec(),
            SortKey::Heap(v) => v,
        }
    }
}

impl PartialEq for SortKey {
    fn eq(&self, other: &Self) -> bool {
        self.as_bytes() == other.as_bytes()
    }
}

impl PartialOrd for SortKey {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for SortKey {
    fn cmp(&self, other: &Self) -> Ordering {
        self.as_bytes().cmp(other.as_bytes())
    }
}

impl std::fmt::Debug for SortKey {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_tuple("SortKey").field(&self.as_bytes()).finish()
    }
}

// ---------------------------------------------------------------------------
// natural_sort_key
// ---------------------------------------------------------------------------

const TAG_NUM: u8 = 0x00;
const TAG_TXT: u8 = 0x01;

/// Builds a sort key whose lexicographic byte ordering matches natural
/// ordering of the input. Numeric runs are encoded as TAG_NUM, varint
/// length, decimal digits with leading zeros stripped. Text runs are encoded
/// as TAG_TXT, raw bytes, then a 0x02 separator
pub fn natural_sort_key(text: &str) -> SortKey {
    natural_sort_key_inner(text, false)
}

/// Case insensitive variant of natural_sort_key. Lowercases ASCII characters
pub fn natural_sort_key_ci(text: &str) -> SortKey {
    natural_sort_key_inner(text, true)
}

fn natural_sort_key_inner(text: &str, fold_case: bool) -> SortKey {
    let bytes = text.as_bytes();
    let mut out = Vec::with_capacity(bytes.len() + 4);
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i].is_ascii_digit() {
            // Skip leading zeros but remember their count separately so that
            // "001" and "1" still produce a stable order, "001" sorting after
            let mut j = i;
            while j < bytes.len() && bytes[j].is_ascii_digit() {
                j += 1;
            }
            let raw = &bytes[i..j];
            let mut k = 0usize;
            while k + 1 < raw.len() && raw[k] == b'0' {
                k += 1;
            }
            let digits = &raw[k..];
            // Length prefix: encode as 4-byte big-endian usize, that gives us
            // up to ~4 billion digits which is overkill but lets us stay
            // simple. Empty digits would encode as length 1 zero
            let nd = digits.len() as u32;
            out.push(TAG_NUM);
            out.extend_from_slice(&nd.to_be_bytes());
            out.extend_from_slice(digits);
            // Tie breaker on leading-zero count, fewer zeros sort first
            out.push(k.min(255) as u8);
            i = j;
        } else {
            let mut j = i;
            while j < bytes.len() && !bytes[j].is_ascii_digit() {
                j += 1;
            }
            out.push(TAG_TXT);
            if fold_case {
                for &b in &bytes[i..j] {
                    out.push(b.to_ascii_lowercase());
                }
            } else {
                out.extend_from_slice(&bytes[i..j]);
            }
            out.push(0x02);
            i = j;
        }
    }
    SortKey::from_bytes(out)
}

/// Direct natural compare without materializing a SortKey. Used by ORDER BY
/// clauses where the key is not persisted
pub fn natural_compare(a: &str, b: &str) -> Ordering {
    natural_compare_inner(a, b, false)
}

pub fn natural_compare_ci(a: &str, b: &str) -> Ordering {
    natural_compare_inner(a, b, true)
}

fn natural_compare_inner(a: &str, b: &str, fold_case: bool) -> Ordering {
    let mut ai = 0usize;
    let mut bi = 0usize;
    let ab = a.as_bytes();
    let bb = b.as_bytes();
    while ai < ab.len() && bi < bb.len() {
        let a_digit = ab[ai].is_ascii_digit();
        let b_digit = bb[bi].is_ascii_digit();
        if a_digit && b_digit {
            // Compare numeric runs by value, leading zeros become a tie breaker
            let aj = ai + ab[ai..].iter().take_while(|c| c.is_ascii_digit()).count();
            let bj = bi + bb[bi..].iter().take_while(|c| c.is_ascii_digit()).count();
            let a_run = &ab[ai..aj];
            let b_run = &bb[bi..bj];
            let a_strip = a_run.iter().position(|c| *c != b'0').unwrap_or(a_run.len() - 1);
            let b_strip = b_run.iter().position(|c| *c != b'0').unwrap_or(b_run.len() - 1);
            let a_digits = &a_run[a_strip..];
            let b_digits = &b_run[b_strip..];
            match a_digits.len().cmp(&b_digits.len()) {
                Ordering::Equal => {}
                other => return other,
            }
            for (x, y) in a_digits.iter().zip(b_digits.iter()) {
                match x.cmp(y) {
                    Ordering::Equal => {}
                    other => return other,
                }
            }
            // Numbers equal in value, more leading zeros sorts later
            match a_strip.cmp(&b_strip) {
                Ordering::Equal => {}
                other => return other,
            }
            ai = aj;
            bi = bj;
        } else if a_digit {
            // Numeric runs sort before text at the same position
            return Ordering::Less;
        } else if b_digit {
            return Ordering::Greater;
        } else {
            let x = if fold_case { ab[ai].to_ascii_lowercase() } else { ab[ai] };
            let y = if fold_case { bb[bi].to_ascii_lowercase() } else { bb[bi] };
            match x.cmp(&y) {
                Ordering::Equal => {}
                other => return other,
            }
            ai += 1;
            bi += 1;
        }
    }
    ab.len().cmp(&bb.len())
}

// ---------------------------------------------------------------------------
// version_compare
// ---------------------------------------------------------------------------

/// Compares two version strings. Splits on '.' and '-', numeric chunks are
/// compared by integer value, alphanumeric chunks by ASCII order. SemVer
/// prerelease semantics: "1.0.0-alpha" < "1.0.0"
/// Returns -1, 0, 1
pub fn version_compare(a: &str, b: &str) -> i32 {
    let (a_main, a_pre) = split_prerelease(a);
    let (b_main, b_pre) = split_prerelease(b);
    let cmp_main = compare_dot_segments(a_main, b_main);
    if cmp_main != Ordering::Equal {
        return ord_to_i32(cmp_main);
    }
    // Main equal, check prerelease. No prerelease > prerelease
    match (a_pre, b_pre) {
        (None, None) => 0,
        (Some(_), None) => -1,
        (None, Some(_)) => 1,
        (Some(ap), Some(bp)) => ord_to_i32(compare_dot_segments(ap, bp)),
    }
}

fn split_prerelease(v: &str) -> (&str, Option<&str>) {
    if let Some(idx) = v.find('-') {
        (&v[..idx], Some(&v[idx + 1..]))
    } else {
        (v, None)
    }
}

fn compare_dot_segments(a: &str, b: &str) -> Ordering {
    let mut ai = a.split('.');
    let mut bi = b.split('.');
    loop {
        match (ai.next(), bi.next()) {
            (None, None) => return Ordering::Equal,
            (None, Some(_)) => return Ordering::Less,
            (Some(_), None) => return Ordering::Greater,
            (Some(a_seg), Some(b_seg)) => {
                let a_num = a_seg.parse::<u64>().ok();
                let b_num = b_seg.parse::<u64>().ok();
                let cmp = match (a_num, b_num) {
                    (Some(x), Some(y)) => x.cmp(&y),
                    (Some(_), None) => Ordering::Less,
                    (None, Some(_)) => Ordering::Greater,
                    (None, None) => natural_compare(a_seg, b_seg),
                };
                if cmp != Ordering::Equal {
                    return cmp;
                }
            }
        }
    }
}

#[inline(always)]
fn ord_to_i32(o: Ordering) -> i32 {
    match o {
        Ordering::Less => -1,
        Ordering::Equal => 0,
        Ordering::Greater => 1,
    }
}

// ---------------------------------------------------------------------------
// IP address sort (E)
// ---------------------------------------------------------------------------

/// Returns a 16-byte sort key for an IPv4 or IPv6 address. IPv4 is mapped to
/// the IPv4-in-IPv6 ::ffff:a.b.c.d form so v4 and v6 sort together correctly
pub fn ip_sort_key(addr: &str) -> Option<[u8; 16]> {
    if let Some(v6) = parse_ipv6(addr) {
        return Some(v6);
    }
    if let Some(v4) = parse_ipv4(addr) {
        let mut key = [0u8; 16];
        key[10] = 0xff;
        key[11] = 0xff;
        key[12..16].copy_from_slice(&v4);
        return Some(key);
    }
    None
}

fn parse_ipv4(s: &str) -> Option<[u8; 4]> {
    let mut parts = s.split('.');
    let mut out = [0u8; 4];
    for slot in out.iter_mut() {
        let p = parts.next()?;
        if p.is_empty() || p.len() > 3 {
            return None;
        }
        let v: u16 = p.parse().ok()?;
        if v > 255 {
            return None;
        }
        *slot = v as u8;
    }
    if parts.next().is_some() {
        return None;
    }
    Some(out)
}

fn parse_ipv6(s: &str) -> Option<[u8; 16]> {
    // Find a '::' double colon, at most one allowed
    let dc = s.find("::");
    if let Some(_extra) = dc.and_then(|p| s[p + 2..].find("::")) {
        return None;
    }
    let (head, tail) = match dc {
        Some(p) => (&s[..p], &s[p + 2..]),
        None => (s, ""),
    };
    let head_groups: Vec<&str> = if head.is_empty() {
        Vec::new()
    } else {
        head.split(':').collect()
    };
    let tail_groups: Vec<&str> = if tail.is_empty() {
        Vec::new()
    } else {
        tail.split(':').collect()
    };
    let total = head_groups.len() + tail_groups.len();
    if total > 8 {
        return None;
    }
    let zeros = 8 - total;
    if dc.is_none() && zeros != 0 {
        return None;
    }
    let mut out = [0u8; 16];
    for (i, g) in head_groups.iter().enumerate() {
        let v = u16::from_str_radix(g, 16).ok()?;
        out[i * 2] = (v >> 8) as u8;
        out[i * 2 + 1] = (v & 0xff) as u8;
    }
    for (j, g) in tail_groups.iter().enumerate() {
        let i = head_groups.len() + zeros + j;
        let v = u16::from_str_radix(g, 16).ok()?;
        out[i * 2] = (v >> 8) as u8;
        out[i * 2 + 1] = (v & 0xff) as u8;
    }
    Some(out)
}

/// Compares two IP address strings. Unparseable addresses sort after parseable
pub fn ip_compare(a: &str, b: &str) -> Ordering {
    match (ip_sort_key(a), ip_sort_key(b)) {
        (Some(x), Some(y)) => x.cmp(&y),
        (Some(_), None) => Ordering::Less,
        (None, Some(_)) => Ordering::Greater,
        (None, None) => a.cmp(b),
    }
}

// ---------------------------------------------------------------------------
// Path sort (E)
// ---------------------------------------------------------------------------

/// Compares two filesystem-style paths component by component using natural
/// compare, so /a/file2 < /a/file10
pub fn path_compare(a: &str, b: &str) -> Ordering {
    let mut ai = a.split(|c| c == '/' || c == '\\');
    let mut bi = b.split(|c| c == '/' || c == '\\');
    loop {
        match (ai.next(), bi.next()) {
            (None, None) => return Ordering::Equal,
            (None, Some(_)) => return Ordering::Less,
            (Some(_), None) => return Ordering::Greater,
            (Some(x), Some(y)) => match natural_compare(x, y) {
                Ordering::Equal => {}
                other => return other,
            },
        }
    }
}

// ---------------------------------------------------------------------------
// Custom order (E)
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, Debug)]
pub enum UnknownPosition {
    First,
    Last,
}

/// Returns the index of value in the order list, or unknown_position
/// resolved to either i32::MIN or i32::MAX so values not in the list sort
/// at the requested end
pub fn custom_order_rank(value: &str, order: &[&str], unknown: UnknownPosition) -> i32 {
    if let Some(idx) = order.iter().position(|v| *v == value) {
        return idx as i32;
    }
    match unknown {
        UnknownPosition::First => i32::MIN,
        UnknownPosition::Last => i32::MAX,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn natural_sort_orders_numerically() {
        let mut items = vec!["file10", "file2", "file1", "file20"];
        items.sort_by(|a, b| natural_compare(a, b));
        assert_eq!(items, vec!["file1", "file2", "file10", "file20"]);
    }

    #[test]
    fn natural_sort_key_round_trip() {
        let mut items: Vec<&str> = vec!["file10", "file2", "file1", "file20"];
        items.sort_by_key(|s| natural_sort_key(s));
        assert_eq!(items, vec!["file1", "file2", "file10", "file20"]);
    }

    #[test]
    fn natural_sort_key_uses_inline_for_short() {
        let k = natural_sort_key("file1");
        assert!(matches!(k, SortKey::Inline { .. }));
    }

    #[test]
    fn natural_sort_key_spills_to_heap_for_long() {
        let s: String = (0..10).map(|_| "abcde").collect();
        let k = natural_sort_key(&s);
        assert!(matches!(k, SortKey::Heap(_)));
    }

    #[test]
    fn natural_sort_case_insensitive() {
        assert_eq!(natural_compare_ci("File2", "file10"), Ordering::Less);
    }

    #[test]
    fn version_compare_basic() {
        assert_eq!(version_compare("1.2.3", "1.10.0"), -1);
        assert_eq!(version_compare("2.0.0", "1.9.9"), 1);
        assert_eq!(version_compare("1.2.3", "1.2.3"), 0);
    }

    #[test]
    fn version_compare_prerelease() {
        assert_eq!(version_compare("1.0.0-alpha", "1.0.0"), -1);
        assert_eq!(version_compare("1.0.0", "1.0.0-alpha"), 1);
        assert_eq!(version_compare("1.0.0-alpha", "1.0.0-beta"), -1);
    }

    #[test]
    fn ip_sort_key_v4() {
        let k = ip_sort_key("10.0.0.1").unwrap();
        assert_eq!(k[12..16], [10, 0, 0, 1]);
        // v4 mapped form, 10..12 must be 0xff 0xff
        assert_eq!(&k[10..12], &[0xff, 0xff]);
    }

    #[test]
    fn ip_compare_v4_orders_numerically() {
        let mut ips = vec!["1.2.3.10", "1.2.3.4", "10.0.0.1", "1.2.4.1"];
        ips.sort_by(|a, b| ip_compare(a, b));
        assert_eq!(ips, vec!["1.2.3.4", "1.2.3.10", "1.2.4.1", "10.0.0.1"]);
    }

    #[test]
    fn ip_sort_key_v6_double_colon() {
        let k = ip_sort_key("::1").unwrap();
        assert_eq!(k[15], 1);
        assert!(k[..15].iter().all(|b| *b == 0));
    }

    #[test]
    fn path_compare_naturally() {
        assert_eq!(path_compare("/a/file2", "/a/file10"), Ordering::Less);
        assert_eq!(path_compare("/a/b", "/a/b/c"), Ordering::Less);
    }

    #[test]
    fn custom_order_known_values() {
        let order = ["urgent", "high", "medium", "low"];
        assert!(custom_order_rank("urgent", &order, UnknownPosition::Last)
            < custom_order_rank("high", &order, UnknownPosition::Last));
        assert_eq!(
            custom_order_rank("nope", &order, UnknownPosition::Last),
            i32::MAX
        );
        assert_eq!(
            custom_order_rank("nope", &order, UnknownPosition::First),
            i32::MIN
        );
    }
}
