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
}
