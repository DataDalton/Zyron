//! Range types with boundary semantics and set operations.
//!
//! Storage layout (variable length):
//!   byte 0: flags byte
//!     bit 0: empty
//!     bit 1: lower_inclusive
//!     bit 2: upper_inclusive
//!     bit 3: lower_infinite
//!     bit 4: upper_infinite
//!   bytes 1..1+elem_size: lower bound (if not infinite)
//!   bytes next elem_size: upper bound (if not infinite)
//!
//! Elements are compared byte-by-byte in big-endian for integers (caller's choice).

use zyron_common::{Result, ZyronError};

const FLAG_EMPTY: u8 = 0x01;
const FLAG_LOWER_INC: u8 = 0x02;
const FLAG_UPPER_INC: u8 = 0x04;
const FLAG_LOWER_INF: u8 = 0x08;
const FLAG_UPPER_INF: u8 = 0x10;

/// Creates a range from optional lower/upper bounds and inclusivity flags.
pub fn range_create(
    lower: Option<&[u8]>,
    upper: Option<&[u8]>,
    lower_inc: bool,
    upper_inc: bool,
    elem_size: usize,
) -> Result<Vec<u8>> {
    if let (Some(l), Some(u)) = (lower, upper) {
        if l.len() != elem_size || u.len() != elem_size {
            return Err(ZyronError::ExecutionError(format!(
                "Range bound size mismatch: expected {}, got {}/{}",
                elem_size,
                l.len(),
                u.len()
            )));
        }
        // Empty range: lower > upper, or lower == upper with both exclusive
        let cmp = compare_bytes(l, u);
        if cmp > 0 || (cmp == 0 && (!lower_inc || !upper_inc)) {
            return Ok(vec![FLAG_EMPTY]);
        }
    }

    let mut flags = 0u8;
    if lower_inc {
        flags |= FLAG_LOWER_INC;
    }
    if upper_inc {
        flags |= FLAG_UPPER_INC;
    }
    if lower.is_none() {
        flags |= FLAG_LOWER_INF;
    }
    if upper.is_none() {
        flags |= FLAG_UPPER_INF;
    }

    let mut result = Vec::with_capacity(1 + 2 * elem_size);
    result.push(flags);
    if let Some(l) = lower {
        result.extend_from_slice(l);
    }
    if let Some(u) = upper {
        result.extend_from_slice(u);
    }
    Ok(result)
}

/// Returns true if the range is empty.
pub fn range_is_empty(r: &[u8]) -> bool {
    !r.is_empty() && r[0] & FLAG_EMPTY != 0
}

/// Returns the lower bound of the range, or None if infinite or empty.
pub fn range_lower(r: &[u8], elem_size: usize) -> Option<Vec<u8>> {
    if r.is_empty() || range_is_empty(r) {
        return None;
    }
    let flags = r[0];
    if flags & FLAG_LOWER_INF != 0 {
        return None;
    }
    let lower_start = 1;
    if r.len() < lower_start + elem_size {
        return None;
    }
    Some(r[lower_start..lower_start + elem_size].to_vec())
}

/// Returns the upper bound of the range, or None if infinite or empty.
pub fn range_upper(r: &[u8], elem_size: usize) -> Option<Vec<u8>> {
    if r.is_empty() || range_is_empty(r) {
        return None;
    }
    let flags = r[0];
    if flags & FLAG_UPPER_INF != 0 {
        return None;
    }
    let upper_start = if flags & FLAG_LOWER_INF != 0 {
        1
    } else {
        1 + elem_size
    };
    if r.len() < upper_start + elem_size {
        return None;
    }
    Some(r[upper_start..upper_start + elem_size].to_vec())
}

/// Returns true if lower bound is inclusive (or lower is infinite).
pub fn range_lower_inclusive(r: &[u8]) -> bool {
    if r.is_empty() {
        return false;
    }
    r[0] & FLAG_LOWER_INC != 0
}

/// Returns true if upper bound is inclusive (or upper is infinite).
pub fn range_upper_inclusive(r: &[u8]) -> bool {
    if r.is_empty() {
        return false;
    }
    r[0] & FLAG_UPPER_INC != 0
}

/// Checks if a value falls within the range.
pub fn range_contains_value(r: &[u8], value: &[u8], elem_size: usize) -> bool {
    if range_is_empty(r) {
        return false;
    }
    if value.len() != elem_size {
        return false;
    }

    let flags = r[0];

    // Lower bound check
    if flags & FLAG_LOWER_INF == 0 {
        let lower_start = 1;
        if r.len() < lower_start + elem_size {
            return false;
        }
        let lower = &r[lower_start..lower_start + elem_size];
        let cmp = compare_bytes(value, lower);
        if cmp < 0 {
            return false;
        }
        if cmp == 0 && flags & FLAG_LOWER_INC == 0 {
            return false;
        }
    }

    // Upper bound check
    if flags & FLAG_UPPER_INF == 0 {
        let upper_start = if flags & FLAG_LOWER_INF != 0 {
            1
        } else {
            1 + elem_size
        };
        if r.len() < upper_start + elem_size {
            return false;
        }
        let upper = &r[upper_start..upper_start + elem_size];
        let cmp = compare_bytes(value, upper);
        if cmp > 0 {
            return false;
        }
        if cmp == 0 && flags & FLAG_UPPER_INC == 0 {
            return false;
        }
    }

    true
}

/// Checks if `inner` is entirely contained within `outer`.
pub fn range_contains_range(outer: &[u8], inner: &[u8], elem_size: usize) -> bool {
    if range_is_empty(inner) {
        return true; // empty range is contained in everything
    }
    if range_is_empty(outer) {
        return false;
    }

    // Outer lower <= inner lower (considering inclusivity)
    let outer_lower = range_lower(outer, elem_size);
    let inner_lower = range_lower(inner, elem_size);
    match (outer_lower.as_deref(), inner_lower.as_deref()) {
        (Some(ol), Some(il)) => {
            let cmp = compare_bytes(ol, il);
            if cmp > 0 {
                return false;
            }
            if cmp == 0 && !range_lower_inclusive(outer) && range_lower_inclusive(inner) {
                return false;
            }
        }
        (Some(_), None) => return false, // outer has lower bound, inner is -inf
        _ => {}
    }

    // Outer upper >= inner upper
    let outer_upper = range_upper(outer, elem_size);
    let inner_upper = range_upper(inner, elem_size);
    match (outer_upper.as_deref(), inner_upper.as_deref()) {
        (Some(ou), Some(iu)) => {
            let cmp = compare_bytes(ou, iu);
            if cmp < 0 {
                return false;
            }
            if cmp == 0 && !range_upper_inclusive(outer) && range_upper_inclusive(inner) {
                return false;
            }
        }
        (Some(_), None) => return false,
        _ => {}
    }

    true
}

/// Checks if two ranges share any point.
pub fn range_overlaps(a: &[u8], b: &[u8], elem_size: usize) -> bool {
    if range_is_empty(a) || range_is_empty(b) {
        return false;
    }

    // a.lower <= b.upper and b.lower <= a.upper
    let a_lower = range_lower(a, elem_size);
    let a_upper = range_upper(a, elem_size);
    let b_lower = range_lower(b, elem_size);
    let b_upper = range_upper(b, elem_size);

    // Check a.lower <= b.upper
    match (a_lower.as_deref(), b_upper.as_deref()) {
        (Some(al), Some(bu)) => {
            let cmp = compare_bytes(al, bu);
            if cmp > 0 {
                return false;
            }
            if cmp == 0 && (!range_lower_inclusive(a) || !range_upper_inclusive(b)) {
                return false;
            }
        }
        _ => {}
    }

    // Check b.lower <= a.upper
    match (b_lower.as_deref(), a_upper.as_deref()) {
        (Some(bl), Some(au)) => {
            let cmp = compare_bytes(bl, au);
            if cmp > 0 {
                return false;
            }
            if cmp == 0 && (!range_lower_inclusive(b) || !range_upper_inclusive(a)) {
                return false;
            }
        }
        _ => {}
    }

    true
}

/// Returns true if two ranges are adjacent (share a boundary with no gap).
pub fn range_adjacent(a: &[u8], b: &[u8], elem_size: usize) -> bool {
    if range_is_empty(a) || range_is_empty(b) {
        return false;
    }

    // a's upper matches b's lower with complementary inclusivity
    let a_upper = range_upper(a, elem_size);
    let b_lower = range_lower(b, elem_size);
    if let (Some(au), Some(bl)) = (a_upper.as_deref(), b_lower.as_deref()) {
        if compare_bytes(au, bl) == 0 && range_upper_inclusive(a) != range_lower_inclusive(b) {
            return true;
        }
    }

    let b_upper = range_upper(b, elem_size);
    let a_lower = range_lower(a, elem_size);
    if let (Some(bu), Some(al)) = (b_upper.as_deref(), a_lower.as_deref()) {
        if compare_bytes(bu, al) == 0 && range_upper_inclusive(b) != range_lower_inclusive(a) {
            return true;
        }
    }

    false
}

/// Returns the union of two ranges if they overlap or are adjacent.
/// Returns an error if the ranges are disjoint with a gap.
pub fn range_union(a: &[u8], b: &[u8], elem_size: usize) -> Result<Vec<u8>> {
    if range_is_empty(a) {
        return Ok(b.to_vec());
    }
    if range_is_empty(b) {
        return Ok(a.to_vec());
    }

    if !range_overlaps(a, b, elem_size) && !range_adjacent(a, b, elem_size) {
        return Err(ZyronError::ExecutionError(
            "Cannot union disjoint non-adjacent ranges".into(),
        ));
    }

    // Take the minimum lower and maximum upper
    let a_lower = range_lower(a, elem_size);
    let b_lower = range_lower(b, elem_size);
    let a_upper = range_upper(a, elem_size);
    let b_upper = range_upper(b, elem_size);

    let (new_lower, new_lower_inc) = match (a_lower.as_deref(), b_lower.as_deref()) {
        (None, _) | (_, None) => (None, true),
        (Some(al), Some(bl)) => {
            let cmp = compare_bytes(al, bl);
            if cmp < 0 {
                (Some(al.to_vec()), range_lower_inclusive(a))
            } else if cmp > 0 {
                (Some(bl.to_vec()), range_lower_inclusive(b))
            } else {
                (
                    Some(al.to_vec()),
                    range_lower_inclusive(a) || range_lower_inclusive(b),
                )
            }
        }
    };

    let (new_upper, new_upper_inc) = match (a_upper.as_deref(), b_upper.as_deref()) {
        (None, _) | (_, None) => (None, true),
        (Some(au), Some(bu)) => {
            let cmp = compare_bytes(au, bu);
            if cmp > 0 {
                (Some(au.to_vec()), range_upper_inclusive(a))
            } else if cmp < 0 {
                (Some(bu.to_vec()), range_upper_inclusive(b))
            } else {
                (
                    Some(au.to_vec()),
                    range_upper_inclusive(a) || range_upper_inclusive(b),
                )
            }
        }
    };

    range_create(
        new_lower.as_deref(),
        new_upper.as_deref(),
        new_lower_inc,
        new_upper_inc,
        elem_size,
    )
}

/// Returns the intersection of two ranges.
pub fn range_intersection(a: &[u8], b: &[u8], elem_size: usize) -> Result<Vec<u8>> {
    if !range_overlaps(a, b, elem_size) {
        return Ok(vec![FLAG_EMPTY]);
    }

    let a_lower = range_lower(a, elem_size);
    let b_lower = range_lower(b, elem_size);
    let a_upper = range_upper(a, elem_size);
    let b_upper = range_upper(b, elem_size);

    // Take the maximum lower and minimum upper
    let (new_lower, new_lower_inc) = match (a_lower.as_deref(), b_lower.as_deref()) {
        (None, None) => (None, true),
        (Some(v), None) => (Some(v.to_vec()), range_lower_inclusive(a)),
        (None, Some(v)) => (Some(v.to_vec()), range_lower_inclusive(b)),
        (Some(al), Some(bl)) => {
            let cmp = compare_bytes(al, bl);
            if cmp > 0 {
                (Some(al.to_vec()), range_lower_inclusive(a))
            } else if cmp < 0 {
                (Some(bl.to_vec()), range_lower_inclusive(b))
            } else {
                (
                    Some(al.to_vec()),
                    range_lower_inclusive(a) && range_lower_inclusive(b),
                )
            }
        }
    };

    let (new_upper, new_upper_inc) = match (a_upper.as_deref(), b_upper.as_deref()) {
        (None, None) => (None, true),
        (Some(v), None) => (Some(v.to_vec()), range_upper_inclusive(a)),
        (None, Some(v)) => (Some(v.to_vec()), range_upper_inclusive(b)),
        (Some(au), Some(bu)) => {
            let cmp = compare_bytes(au, bu);
            if cmp < 0 {
                (Some(au.to_vec()), range_upper_inclusive(a))
            } else if cmp > 0 {
                (Some(bu.to_vec()), range_upper_inclusive(b))
            } else {
                (
                    Some(au.to_vec()),
                    range_upper_inclusive(a) && range_upper_inclusive(b),
                )
            }
        }
    };

    range_create(
        new_lower.as_deref(),
        new_upper.as_deref(),
        new_lower_inc,
        new_upper_inc,
        elem_size,
    )
}

fn compare_bytes(a: &[u8], b: &[u8]) -> i32 {
    for i in 0..a.len().min(b.len()) {
        if a[i] < b[i] {
            return -1;
        }
        if a[i] > b[i] {
            return 1;
        }
    }
    if a.len() < b.len() {
        -1
    } else if a.len() > b.len() {
        1
    } else {
        0
    }
}

// ---------------------------------------------------------------------------
// SQL text form and overlap
// ---------------------------------------------------------------------------

/// What a range column's 8 byte elements mean, from its declared element
/// type. Elements store as sign flipped big endian order keys so byte
/// order equals numeric order
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RangeElemKind {
    Int,
    Date,
    TimestampMicros,
}

/// Sign flipped big endian encoding of an i64, whose byte order matches
/// numeric order
pub fn order_key_i64(value: i64) -> [u8; 8] {
    ((value as u64) ^ (1u64 << 63)).to_be_bytes()
}

/// Reverses [`order_key_i64`]
pub fn order_key_to_i64(bytes: &[u8]) -> Result<i64> {
    let arr: [u8; 8] = bytes.try_into().map_err(|_| ZyronError::InvalidParameter {
        name: "range_bound".to_string(),
        value: format!("{} bytes", bytes.len()),
    })?;
    Ok((u64::from_be_bytes(arr) ^ (1u64 << 63)) as i64)
}

/// Civil date to days since the epoch
fn days_from_civil(y: i32, m: u32, d: u32) -> i64 {
    let y = y - i32::from(m <= 2);
    let era = if y >= 0 { y } else { y - 399 } / 400;
    let yoe = y - era * 400;
    let mp = (m as i32 + 9) % 12;
    let doy = (153 * mp + 2) / 5 + d as i32 - 1;
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    (era as i64) * 146_097 + doe as i64 - 719_468
}

fn parse_bound_value(text: &str, kind: RangeElemKind) -> Result<i64> {
    let t = text.trim().trim_matches('"');
    let bad = || ZyronError::InvalidParameter {
        name: "range_bound".to_string(),
        value: text.to_string(),
    };
    match kind {
        RangeElemKind::Int => t.parse::<i64>().map_err(|_| bad()),
        RangeElemKind::Date => {
            let mut parts = t.splitn(3, '-');
            let y: i32 = parts.next().and_then(|p| p.parse().ok()).ok_or_else(bad)?;
            let m: u32 = parts.next().and_then(|p| p.parse().ok()).ok_or_else(bad)?;
            let d: u32 = parts.next().and_then(|p| p.parse().ok()).ok_or_else(bad)?;
            if !(1..=12).contains(&m) || !(1..=31).contains(&d) {
                return Err(bad());
            }
            Ok(days_from_civil(y, m, d))
        }
        RangeElemKind::TimestampMicros => {
            let (date_part, time_part) = match t.split_once([' ', 'T']) {
                Some((d, tp)) => (d, Some(tp)),
                None => (t, None),
            };
            let days = parse_bound_value(date_part, RangeElemKind::Date)?;
            let mut micros = days.saturating_mul(86_400_000_000);
            if let Some(time) = time_part {
                let mut hms = time.trim_end_matches('Z').splitn(3, ':');
                let h: i64 = hms.next().and_then(|p| p.parse().ok()).ok_or_else(bad)?;
                let mi: i64 = hms.next().and_then(|p| p.parse().ok()).unwrap_or(0);
                let sec_text = hms.next().unwrap_or("0");
                let sec: f64 = sec_text.parse().map_err(|_| bad())?;
                micros += h * 3_600_000_000 + mi * 60_000_000 + (sec * 1_000_000.0) as i64;
            }
            Ok(micros)
        }
    }
}

/// Parses the SQL text form of a range: `[lower,upper)`, `(lower,upper]`,
/// an omitted bound is infinite, and `empty` is the empty range. Bounds
/// store as order keys so the stored bytes compare in value order
pub fn range_from_text(text: &str, kind: RangeElemKind) -> Result<Vec<u8>> {
    let t = text.trim();
    if t.eq_ignore_ascii_case("empty") {
        return Ok(vec![FLAG_EMPTY]);
    }
    let bad = || ZyronError::InvalidParameter {
        name: "range".to_string(),
        value: text.to_string(),
    };
    let mut chars = t.chars();
    let lower_inc = match chars.next() {
        Some('[') => true,
        Some('(') => false,
        _ => return Err(bad()),
    };
    let upper_inc = match t.chars().last() {
        Some(']') => true,
        Some(')') => false,
        _ => return Err(bad()),
    };
    let inner = &t[1..t.len() - 1];
    let (lower_text, upper_text) = inner.split_once(',').ok_or_else(bad)?;
    let lower = if lower_text.trim().is_empty() {
        None
    } else {
        Some(order_key_i64(parse_bound_value(lower_text, kind)?))
    };
    let upper = if upper_text.trim().is_empty() {
        None
    } else {
        Some(order_key_i64(parse_bound_value(upper_text, kind)?))
    };
    range_create(
        lower.as_ref().map(|b| b.as_slice()),
        upper.as_ref().map(|b| b.as_slice()),
        lower_inc,
        upper_inc,
        8,
    )
}

// ---------------------------------------------------------------------------
// Order preserving index key form
// ---------------------------------------------------------------------------

/// Width of one encoded range bound. Every element kind a range column can
/// declare reaches storage through `order_key_i64`, so one width covers them
/// all and a range column's index keys are fixed width
pub const RANGE_ELEM_SIZE: usize = 8;

/// Bytes one range occupies in index key form
pub const RANGE_INDEX_KEY_LEN: usize = 2 * RANGE_ELEM_SIZE + 4;

// Bound kind markers, ordered so an unbounded low end sorts below every
// finite value and an unbounded high end sorts above every finite value
const KIND_NEG_INF: u8 = 0x00;
const KIND_FINITE: u8 = 0x01;
const KIND_POS_INF: u8 = 0x02;
const KIND_EMPTY: u8 = 0xFF;

/// The index key form of a range: upper bound first, then lower.
///
/// The storage form leads with a flags byte whose inclusivity bits change
/// independently of the values, so memcmp over stored ranges says nothing
/// about which range starts or ends first. This form is byte ordered by
/// upper bound, then by lower, which is what lets an index seek answer an
/// overlap question instead of scanning.
///
/// Upper first is deliberate. Ranges that share a key under a WITHOUT
/// OVERLAPS constraint are pairwise disjoint, so ordering them by upper
/// bound orders them by lower bound as well, and a seek to the first range
/// whose upper bound is past a probe point lands on the only candidate that
/// can still be open there.
///
/// Inclusivity rides after each value, and its direction differs per end. A
/// bound that includes its value starts earlier and so sorts first at the
/// low end, while at the high end an excluded value ends earlier and sorts
/// first. Empty ranges take a marker above every finite value so they
/// gather past the live entries rather than in the middle of them.
pub fn range_index_key(r: &[u8], elem_size: usize) -> Vec<u8> {
    let mut key = vec![0u8; 2 * elem_size + 4];
    if r.is_empty() || range_is_empty(r) {
        key[0] = KIND_EMPTY;
        key[elem_size + 2] = KIND_EMPTY;
        return key;
    }
    // Upper half
    match range_upper(r, elem_size) {
        Some(bytes) => {
            key[0] = KIND_FINITE;
            key[1..1 + elem_size].copy_from_slice(&bytes);
        }
        None => key[0] = KIND_POS_INF,
    }
    key[1 + elem_size] = u8::from(range_upper_inclusive(r));
    // Lower half
    let lo = elem_size + 2;
    match range_lower(r, elem_size) {
        Some(bytes) => {
            key[lo] = KIND_FINITE;
            key[lo + 1..lo + 1 + elem_size].copy_from_slice(&bytes);
        }
        None => key[lo] = KIND_NEG_INF,
    }
    key[lo + 1 + elem_size] = u8::from(!range_lower_inclusive(r));
    key
}

/// Rebuilds the storage form of a range from its index key, so an overlap
/// test reads the index entry rather than fetching the row it points at
pub fn range_from_index_key(key: &[u8], elem_size: usize) -> Option<Vec<u8>> {
    if key.len() != 2 * elem_size + 4 {
        return None;
    }
    if key[0] == KIND_EMPTY {
        return Some(vec![FLAG_EMPTY]);
    }
    let lo = elem_size + 2;
    let upper = match key[0] {
        KIND_FINITE => Some(&key[1..1 + elem_size]),
        KIND_POS_INF => None,
        _ => return None,
    };
    let lower = match key[lo] {
        KIND_FINITE => Some(&key[lo + 1..lo + 1 + elem_size]),
        KIND_NEG_INF => None,
        _ => return None,
    };
    let upper_inc = key[1 + elem_size] != 0;
    let lower_inc = key[lo + 1 + elem_size] == 0;
    range_create(lower, upper, lower_inc, upper_inc, elem_size).ok()
}

/// The seek key for an overlap probe: the lowest index key any range whose
/// upper bound still covers `point` can carry.
///
/// A range that ends exactly at the probe point shares its value, so the
/// inclusivity marker decides. When the probe point is included, a stored
/// range ending there inclusively still meets it and has to be visited;
/// when the probe point is excluded, such a range only touches the excluded
/// endpoint and is skipped
pub fn range_index_probe(point: &[u8], point_inclusive: bool, elem_size: usize) -> Vec<u8> {
    let mut key = vec![0u8; 2 * elem_size + 4];
    key[0] = KIND_FINITE;
    let n = point.len().min(elem_size);
    key[1 + elem_size - n..1 + elem_size].copy_from_slice(&point[point.len() - n..]);
    key[1 + elem_size] = if point_inclusive { 0x01 } else { 0x02 };
    key
}

/// The probe key for a range's own lower end, the seek an overlap check for
/// that range starts from. An unbounded lower end probes from the bottom
pub fn range_overlap_probe(r: &[u8], elem_size: usize) -> Vec<u8> {
    match range_lower(r, elem_size) {
        Some(lower) => range_index_probe(&lower, range_lower_inclusive(r), elem_size),
        None => vec![0u8; 2 * elem_size + 4],
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Use 4-byte BE encoded i32 for testing
    fn i32_bytes(v: i32) -> [u8; 4] {
        // Offset to unsigned for byte-comparable ordering
        let u = (v as i64 - i32::MIN as i64) as u32;
        u.to_be_bytes()
    }

    #[test]
    fn test_create_basic() {
        let lo = i32_bytes(1);
        let hi = i32_bytes(10);
        let r = range_create(Some(&lo), Some(&hi), true, false, 4).unwrap();
        assert!(!range_is_empty(&r));
    }

    #[test]
    fn test_empty_range() {
        let lo = i32_bytes(10);
        let hi = i32_bytes(1);
        let r = range_create(Some(&lo), Some(&hi), true, true, 4).unwrap();
        assert!(range_is_empty(&r));
    }

    #[test]
    fn test_exclusive_equal_is_empty() {
        let v = i32_bytes(5);
        let r = range_create(Some(&v), Some(&v), false, false, 4).unwrap();
        assert!(range_is_empty(&r));
    }

    #[test]
    fn test_contains_value() {
        let lo = i32_bytes(1);
        let hi = i32_bytes(10);
        let r = range_create(Some(&lo), Some(&hi), true, false, 4).unwrap();
        assert!(range_contains_value(&r, &i32_bytes(5), 4));
        assert!(range_contains_value(&r, &i32_bytes(1), 4)); // inclusive lower
        assert!(!range_contains_value(&r, &i32_bytes(10), 4)); // exclusive upper
        assert!(!range_contains_value(&r, &i32_bytes(0), 4));
        assert!(!range_contains_value(&r, &i32_bytes(11), 4));
    }

    #[test]
    fn test_contains_value_unbounded_lower() {
        let hi = i32_bytes(10);
        let r = range_create(None, Some(&hi), true, true, 4).unwrap();
        assert!(range_contains_value(&r, &i32_bytes(-100), 4));
        assert!(range_contains_value(&r, &i32_bytes(10), 4));
        assert!(!range_contains_value(&r, &i32_bytes(11), 4));
    }

    #[test]
    fn test_contains_value_unbounded_upper() {
        let lo = i32_bytes(1);
        let r = range_create(Some(&lo), None, true, true, 4).unwrap();
        assert!(range_contains_value(&r, &i32_bytes(1), 4));
        assert!(range_contains_value(&r, &i32_bytes(1000000), 4));
        assert!(!range_contains_value(&r, &i32_bytes(0), 4));
    }

    #[test]
    fn test_overlaps() {
        let r1 = range_create(Some(&i32_bytes(1)), Some(&i32_bytes(10)), true, true, 4).unwrap();
        let r2 = range_create(Some(&i32_bytes(5)), Some(&i32_bytes(15)), true, true, 4).unwrap();
        assert!(range_overlaps(&r1, &r2, 4));
    }

    #[test]
    fn test_no_overlap() {
        let r1 = range_create(Some(&i32_bytes(1)), Some(&i32_bytes(5)), true, true, 4).unwrap();
        let r2 = range_create(Some(&i32_bytes(10)), Some(&i32_bytes(20)), true, true, 4).unwrap();
        assert!(!range_overlaps(&r1, &r2, 4));
    }

    #[test]
    fn test_touching_exclusive_no_overlap() {
        let r1 = range_create(Some(&i32_bytes(1)), Some(&i32_bytes(5)), true, false, 4).unwrap();
        let r2 = range_create(Some(&i32_bytes(5)), Some(&i32_bytes(10)), true, true, 4).unwrap();
        assert!(!range_overlaps(&r1, &r2, 4));
    }

    #[test]
    fn test_adjacent() {
        let r1 = range_create(Some(&i32_bytes(1)), Some(&i32_bytes(5)), true, false, 4).unwrap();
        let r2 = range_create(Some(&i32_bytes(5)), Some(&i32_bytes(10)), true, true, 4).unwrap();
        assert!(range_adjacent(&r1, &r2, 4));
    }

    #[test]
    fn test_contains_range() {
        let outer =
            range_create(Some(&i32_bytes(0)), Some(&i32_bytes(100)), true, true, 4).unwrap();
        let inner =
            range_create(Some(&i32_bytes(10)), Some(&i32_bytes(20)), true, true, 4).unwrap();
        assert!(range_contains_range(&outer, &inner, 4));
    }

    #[test]
    fn test_not_contains_range() {
        let r1 = range_create(Some(&i32_bytes(10)), Some(&i32_bytes(20)), true, true, 4).unwrap();
        let r2 = range_create(Some(&i32_bytes(15)), Some(&i32_bytes(25)), true, true, 4).unwrap();
        assert!(!range_contains_range(&r1, &r2, 4));
    }

    #[test]
    fn test_union_overlapping() {
        let r1 = range_create(Some(&i32_bytes(1)), Some(&i32_bytes(10)), true, true, 4).unwrap();
        let r2 = range_create(Some(&i32_bytes(5)), Some(&i32_bytes(15)), true, true, 4).unwrap();
        let union = range_union(&r1, &r2, 4).unwrap();
        assert!(range_contains_value(&union, &i32_bytes(1), 4));
        assert!(range_contains_value(&union, &i32_bytes(15), 4));
        assert!(!range_contains_value(&union, &i32_bytes(16), 4));
    }

    #[test]
    fn test_union_disjoint_error() {
        let r1 = range_create(Some(&i32_bytes(1)), Some(&i32_bytes(5)), true, true, 4).unwrap();
        let r2 = range_create(Some(&i32_bytes(10)), Some(&i32_bytes(20)), true, true, 4).unwrap();
        assert!(range_union(&r1, &r2, 4).is_err());
    }

    #[test]
    fn test_intersection_overlapping() {
        let r1 = range_create(Some(&i32_bytes(1)), Some(&i32_bytes(10)), true, true, 4).unwrap();
        let r2 = range_create(Some(&i32_bytes(5)), Some(&i32_bytes(15)), true, true, 4).unwrap();
        let inter = range_intersection(&r1, &r2, 4).unwrap();
        assert!(range_contains_value(&inter, &i32_bytes(7), 4));
        assert!(!range_contains_value(&inter, &i32_bytes(1), 4));
        assert!(!range_contains_value(&inter, &i32_bytes(15), 4));
    }

    #[test]
    fn test_intersection_disjoint() {
        let r1 = range_create(Some(&i32_bytes(1)), Some(&i32_bytes(5)), true, true, 4).unwrap();
        let r2 = range_create(Some(&i32_bytes(10)), Some(&i32_bytes(20)), true, true, 4).unwrap();
        let inter = range_intersection(&r1, &r2, 4).unwrap();
        assert!(range_is_empty(&inter));
    }

    #[test]
    fn test_lower_upper_extraction() {
        let lo = i32_bytes(3);
        let hi = i32_bytes(7);
        let r = range_create(Some(&lo), Some(&hi), true, false, 4).unwrap();
        assert_eq!(range_lower(&r, 4).unwrap(), lo.to_vec());
        assert_eq!(range_upper(&r, 4).unwrap(), hi.to_vec());
        assert!(range_lower_inclusive(&r));
        assert!(!range_upper_inclusive(&r));
    }

    #[test]
    fn test_empty_contained_in_anything() {
        let empty = vec![FLAG_EMPTY];
        let normal =
            range_create(Some(&i32_bytes(0)), Some(&i32_bytes(10)), true, true, 4).unwrap();
        assert!(range_contains_range(&normal, &empty, 4));
    }

    #[test]
    fn test_union_with_empty() {
        let empty = vec![FLAG_EMPTY];
        let normal = range_create(Some(&i32_bytes(1)), Some(&i32_bytes(5)), true, true, 4).unwrap();
        let u = range_union(&normal, &empty, 4).unwrap();
        assert!(range_contains_value(&u, &i32_bytes(3), 4));
    }

    fn day_range(lo: i64, hi: i64) -> Vec<u8> {
        range_create(
            Some(&order_key_i64(lo)),
            Some(&order_key_i64(hi)),
            true,
            false,
            RANGE_ELEM_SIZE,
        )
        .expect("range")
    }

    #[test]
    fn test_index_key_orders_disjoint_ranges_by_position() {
        // Disjoint ranges in ascending order must produce ascending keys,
        // which is what lets an index seek answer an overlap question
        let ranges: Vec<Vec<u8>> = (0..64).map(|i| day_range(i * 10, i * 10 + 5)).collect();
        let keys: Vec<Vec<u8>> = ranges
            .iter()
            .map(|r| range_index_key(r, RANGE_ELEM_SIZE))
            .collect();
        for w in keys.windows(2) {
            assert!(w[0] < w[1], "index keys are not ascending");
        }
        // Negative positions sort below positive ones, so the sign flipped
        // order key survives the reordering
        let neg = range_index_key(&day_range(-100, -90), RANGE_ELEM_SIZE);
        assert!(neg < keys[0]);
    }

    #[test]
    fn test_index_key_orders_inclusivity_within_a_value() {
        let inclusive_upper = range_create(
            Some(&order_key_i64(0)),
            Some(&order_key_i64(10)),
            true,
            true,
            RANGE_ELEM_SIZE,
        )
        .expect("inclusive");
        let exclusive_upper = day_range(0, 10);
        // The range ending before 10 sorts under the one that includes it
        assert!(
            range_index_key(&exclusive_upper, RANGE_ELEM_SIZE)
                < range_index_key(&inclusive_upper, RANGE_ELEM_SIZE)
        );
    }

    #[test]
    fn test_index_key_round_trips() {
        let cases = vec![
            day_range(5, 9),
            range_create(None, Some(&order_key_i64(9)), false, true, RANGE_ELEM_SIZE).expect("a"),
            range_create(Some(&order_key_i64(5)), None, true, false, RANGE_ELEM_SIZE).expect("b"),
            range_create(None, None, false, false, RANGE_ELEM_SIZE).expect("c"),
            vec![FLAG_EMPTY],
        ];
        for original in cases {
            let key = range_index_key(&original, RANGE_ELEM_SIZE);
            assert_eq!(key.len(), RANGE_INDEX_KEY_LEN);
            let back = range_from_index_key(&key, RANGE_ELEM_SIZE).expect("decodes");
            assert_eq!(
                range_is_empty(&back),
                range_is_empty(&original),
                "emptiness changed"
            );
            if !range_is_empty(&original) {
                assert_eq!(
                    range_lower(&back, RANGE_ELEM_SIZE),
                    range_lower(&original, RANGE_ELEM_SIZE)
                );
                assert_eq!(
                    range_upper(&back, RANGE_ELEM_SIZE),
                    range_upper(&original, RANGE_ELEM_SIZE)
                );
                assert_eq!(
                    range_lower_inclusive(&back),
                    range_lower_inclusive(&original)
                );
                assert_eq!(
                    range_upper_inclusive(&back),
                    range_upper_inclusive(&original)
                );
            }
        }
    }

    #[test]
    fn test_probe_selects_exactly_the_ranges_still_open() {
        // A disjoint ladder, then a probe from every candidate range. The
        // first key at or above the probe must be the first stored range
        // that the probe range can still overlap, which is what the
        // enforcement path relies on for its single seek
        let stored: Vec<Vec<u8>> = (0..40).map(|i| day_range(i * 10, i * 10 + 6)).collect();
        let mut keyed: Vec<(Vec<u8>, usize)> = stored
            .iter()
            .enumerate()
            .map(|(i, r)| (range_index_key(r, RANGE_ELEM_SIZE), i))
            .collect();
        keyed.sort();
        for probe_lo in [0i64, 7, 13, 55, 100, 396] {
            let probe = day_range(probe_lo, probe_lo + 4);
            let seek = range_overlap_probe(&probe, RANGE_ELEM_SIZE);
            let first = keyed.iter().find(|(k, _)| k.as_slice() >= seek.as_slice());
            // Every stored range under the seek point ends at or before the
            // probe starts, so none of them can overlap
            for (k, i) in &keyed {
                if k.as_slice() < seek.as_slice() {
                    assert!(
                        !range_overlaps(&stored[*i], &probe, RANGE_ELEM_SIZE),
                        "a range below the seek overlapped the probe"
                    );
                }
            }
            // The true answer, computed the slow way, is found at or after
            // the seek
            let expected = stored
                .iter()
                .position(|s| range_overlaps(s, &probe, RANGE_ELEM_SIZE));
            if let Some(exp) = expected {
                let found = first.expect("a seek hit exists when an overlap exists");
                assert!(found.1 <= exp, "the seek landed past the overlapping range");
            }
        }
    }

    #[test]
    fn test_probe_skips_a_range_that_only_touches_an_excluded_start() {
        // [0,10) stored, probe (10,20): they meet at 10, which neither
        // contains, so the seek must land above the stored entry
        let stored = day_range(0, 10);
        let probe = range_create(
            Some(&order_key_i64(10)),
            Some(&order_key_i64(20)),
            false,
            false,
            RANGE_ELEM_SIZE,
        )
        .expect("probe");
        assert!(!range_overlaps(&stored, &probe, RANGE_ELEM_SIZE));
        let seek = range_overlap_probe(&probe, RANGE_ELEM_SIZE);
        assert!(range_index_key(&stored, RANGE_ELEM_SIZE) < seek);
    }

    #[test]
    fn test_probe_keeps_a_range_that_covers_an_included_start() {
        // [0,10] stored, probe [10,20): they share 10, so the seek must not
        // sort past the stored entry
        let stored = range_create(
            Some(&order_key_i64(0)),
            Some(&order_key_i64(10)),
            true,
            true,
            RANGE_ELEM_SIZE,
        )
        .expect("stored");
        let probe = day_range(10, 20);
        assert!(range_overlaps(&stored, &probe, RANGE_ELEM_SIZE));
        let seek = range_overlap_probe(&probe, RANGE_ELEM_SIZE);
        assert!(range_index_key(&stored, RANGE_ELEM_SIZE) >= seek);
    }

    #[test]
    fn test_sequential_date_ranges_stay_disjoint() {
        // A run of consecutive half open date ranges built from a linear
        // serial, none of which may report overlap with any other
        let day_of = |i: usize| {
            let year = 2000 + i / 336;
            let month = 1 + (i % 336) / 28;
            let day = 1 + i % 28;
            format!("{year:04}-{month:02}-{day:02}")
        };
        let ranges: Vec<Vec<u8>> = (0..200)
            .map(|i| {
                range_from_text(
                    &format!("[{},{})", day_of(i * 2), day_of(i * 2 + 1)),
                    RangeElemKind::Date,
                )
                .expect("valid text")
            })
            .collect();
        for a in 0..ranges.len() {
            for b in (a + 1)..ranges.len() {
                assert!(
                    !range_overlaps(&ranges[a], &ranges[b], 8),
                    "ranges {a} and {b} report overlap"
                );
            }
        }
    }

    #[test]
    fn test_range_from_text_and_overlap() {
        let jan = range_from_text("[2026-01-01,2026-01-05)", RangeElemKind::Date).expect("jan");
        let overlapping =
            range_from_text("[2026-01-04,2026-01-10)", RangeElemKind::Date).expect("mid");
        let disjoint =
            range_from_text("[2026-01-05,2026-01-10)", RangeElemKind::Date).expect("next");
        assert!(range_overlaps(&jan, &overlapping, 8));
        // Half open ranges meeting at the boundary do not overlap
        assert!(!range_overlaps(&jan, &disjoint, 8));
        // Closed ranges meeting at the boundary do
        let closed_a = range_from_text("[1,5]", RangeElemKind::Int).expect("a");
        let closed_b = range_from_text("[5,9]", RangeElemKind::Int).expect("b");
        assert!(range_overlaps(&closed_a, &closed_b, 8));
        // Infinite bounds overlap everything on that side
        let open_ended = range_from_text("[2026-01-04,)", RangeElemKind::Int).is_err();
        assert!(open_ended, "date text under Int kind errors");
        let unbounded = range_from_text("[5,)", RangeElemKind::Int).expect("unbounded");
        assert!(range_overlaps(&unbounded, &closed_b, 8));
        // Empty never overlaps
        let empty = range_from_text("empty", RangeElemKind::Int).expect("empty");
        assert!(!range_overlaps(&empty, &closed_a, 8));
        // Timestamps parse with and without a time of day
        let ts = range_from_text(
            "[2026-01-01 12:30:00,2026-01-02)",
            RangeElemKind::TimestampMicros,
        );
        assert!(ts.is_ok());
        assert!(range_from_text("not a range", RangeElemKind::Int).is_err());
    }
}
