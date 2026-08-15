//! Raw cell byte interpretation shared by the writer and reader.
//!
//! Cells use the storage layer's representation: little endian fixed-width
//! values, UTF-8 or raw bytes for variable-length values, NULL carried
//! outside the cell. Ordering is defined per type family, and any type
//! whose byte order does not match its value order is Unordered, it gets
//! no bounds and prunes nothing rather than pruning wrongly

use std::cmp::Ordering;

use zyron_common::TypeId;

use crate::predicate::{LakeValue, compare_i128_to_value, compare_u128_to_value};

// Cell family classification is shared with the fold tier through the
// ordering curves, so it is defined in zyron-common
pub(crate) use zyron_common::curve::{CellFamily, cell_family};

fn signed_from_cell(cell: &[u8]) -> Option<i128> {
    Some(match cell.len() {
        1 => cell[0] as i8 as i128,
        2 => i16::from_le_bytes([cell[0], cell[1]]) as i128,
        4 => {
            let mut a = [0u8; 4];
            a.copy_from_slice(cell);
            i32::from_le_bytes(a) as i128
        }
        8 => {
            let mut a = [0u8; 8];
            a.copy_from_slice(cell);
            i64::from_le_bytes(a) as i128
        }
        16 => {
            let mut a = [0u8; 16];
            a.copy_from_slice(cell);
            i128::from_le_bytes(a)
        }
        _ => return None,
    })
}

fn unsigned_from_cell(cell: &[u8]) -> Option<u128> {
    Some(match cell.len() {
        1 => cell[0] as u128,
        2 => u16::from_le_bytes([cell[0], cell[1]]) as u128,
        4 => {
            let mut a = [0u8; 4];
            a.copy_from_slice(cell);
            u32::from_le_bytes(a) as u128
        }
        8 => {
            let mut a = [0u8; 8];
            a.copy_from_slice(cell);
            u64::from_le_bytes(a) as u128
        }
        16 => {
            let mut a = [0u8; 16];
            a.copy_from_slice(cell);
            u128::from_le_bytes(a)
        }
        _ => return None,
    })
}

fn float_from_cell(cell: &[u8]) -> Option<f64> {
    Some(match cell.len() {
        4 => {
            let mut a = [0u8; 4];
            a.copy_from_slice(cell);
            f32::from_le_bytes(a) as f64
        }
        8 => {
            let mut a = [0u8; 8];
            a.copy_from_slice(cell);
            f64::from_le_bytes(a)
        }
        _ => return None,
    })
}

/// Interprets one cell into a typed value for statistics bounds. None for
/// unordered types and malformed widths, those columns keep no bounds
pub(crate) fn cell_to_value(physical: TypeId, cell: &[u8]) -> Option<LakeValue> {
    match cell_family(physical) {
        CellFamily::Bool => cell.first().map(|b| LakeValue::Bool(*b != 0)),
        CellFamily::SignedInt => signed_from_cell(cell).map(|v| {
            if cell.len() == 16 {
                LakeValue::Int128(v)
            } else {
                LakeValue::Int(v as i64)
            }
        }),
        CellFamily::UnsignedInt => unsigned_from_cell(cell).map(|v| {
            if cell.len() == 16 {
                LakeValue::UInt128(v)
            } else {
                LakeValue::UInt(v as u64)
            }
        }),
        CellFamily::Float => float_from_cell(cell).map(LakeValue::Float),
        CellFamily::Str => std::str::from_utf8(cell)
            .ok()
            .map(|s| LakeValue::Str(s.to_string())),
        CellFamily::Bytes => Some(LakeValue::Bytes(cell.to_vec())),
        CellFamily::Unordered => None,
    }
}

/// A cell's canonical bytes, borrowed when the value already holds them and
/// held inline otherwise, so encoding a constant for a bloom probe never
/// allocates
pub(crate) enum CellBytes<'a> {
    Inline { buf: [u8; 16], len: usize },
    Borrowed(&'a [u8]),
}

impl CellBytes<'_> {
    pub(crate) fn as_slice(&self) -> &[u8] {
        match self {
            Self::Inline { buf, len } => &buf[..*len],
            Self::Borrowed(b) => b,
        }
    }
}

fn inline(bytes: &[u8]) -> CellBytes<'static> {
    let mut buf = [0u8; 16];
    buf[..bytes.len()].copy_from_slice(bytes);
    CellBytes::Inline {
        buf,
        len: bytes.len(),
    }
}

/// Encodes a typed constant into the exact cell bytes the writer stored for
/// this column, or None when no stored cell can be proven byte-identical.
///
/// This is the inverse of `cell_to_value` and it must stay exact: the bytes
/// go to a bloom probe whose false answer drops rows that do exist. Anything
/// whose representation is not pinned down here returns None, which prunes
/// nothing.
pub(crate) fn value_to_cell<'a>(
    physical: TypeId,
    width: usize,
    value: &'a LakeValue,
) -> Option<CellBytes<'a>> {
    match cell_family(physical) {
        CellFamily::Bool => match value {
            LakeValue::Bool(b) => Some(inline(&[*b as u8])),
            _ => None,
        },
        CellFamily::SignedInt => {
            let v = match value {
                LakeValue::Int(v) => *v as i128,
                LakeValue::Int128(v) => *v,
                // A non-negative unsigned constant compares exactly against a
                // signed column, so it encodes when it fits the width
                LakeValue::UInt(v) => *v as i128,
                LakeValue::UInt128(v) => i128::try_from(*v).ok()?,
                _ => return None,
            };
            match width {
                1 => i8::try_from(v).ok().map(|x| inline(&x.to_le_bytes())),
                2 => i16::try_from(v).ok().map(|x| inline(&x.to_le_bytes())),
                4 => i32::try_from(v).ok().map(|x| inline(&x.to_le_bytes())),
                8 => i64::try_from(v).ok().map(|x| inline(&x.to_le_bytes())),
                16 => Some(inline(&v.to_le_bytes())),
                _ => None,
            }
        }
        CellFamily::UnsignedInt => {
            let v = match value {
                LakeValue::UInt(v) => *v as u128,
                LakeValue::UInt128(v) => *v,
                LakeValue::Int(v) => u128::try_from(*v).ok()?,
                LakeValue::Int128(v) => u128::try_from(*v).ok()?,
                _ => return None,
            };
            match width {
                1 => u8::try_from(v).ok().map(|x| inline(&x.to_le_bytes())),
                2 => u16::try_from(v).ok().map(|x| inline(&x.to_le_bytes())),
                4 => u32::try_from(v).ok().map(|x| inline(&x.to_le_bytes())),
                8 => u64::try_from(v).ok().map(|x| inline(&x.to_le_bytes())),
                16 => Some(inline(&v.to_le_bytes())),
                _ => None,
            }
        }
        CellFamily::Float => {
            let v = match value {
                LakeValue::Float(v) => *v,
                _ => return None,
            };
            match width {
                // Only a constant that survives the narrowing round trip has
                // a stored f32 cell it can equal
                4 => {
                    let narrowed = v as f32;
                    if narrowed as f64 == v || (v.is_nan() && narrowed.is_nan()) {
                        Some(inline(&narrowed.to_le_bytes()))
                    } else {
                        None
                    }
                }
                8 => Some(inline(&v.to_le_bytes())),
                _ => None,
            }
        }
        // Char cells are blank padded to their declared width, so a bare
        // literal is not the stored cell
        CellFamily::Str if physical == TypeId::Char => None,
        CellFamily::Str => match value {
            LakeValue::Str(s) => Some(CellBytes::Borrowed(s.as_bytes())),
            _ => None,
        },
        CellFamily::Bytes => match value {
            // A fixed-width identifier only matches a constant of that width
            LakeValue::Bytes(b) if width == 0 || b.len() == width => {
                Some(CellBytes::Borrowed(b.as_slice()))
            }
            _ => None,
        },
        CellFamily::Unordered => None,
    }
}

/// Total order between two cells of one column, used for sort routing.
/// Unordered families and malformed widths fall back to byte order so the
/// sort stays deterministic
pub(crate) fn compare_cells(physical: TypeId, a: &[u8], b: &[u8]) -> Ordering {
    match cell_family(physical) {
        CellFamily::Bool => a.first().cmp(&b.first()),
        CellFamily::SignedInt => match (signed_from_cell(a), signed_from_cell(b)) {
            (Some(x), Some(y)) => x.cmp(&y),
            _ => a.cmp(b),
        },
        CellFamily::UnsignedInt => match (unsigned_from_cell(a), unsigned_from_cell(b)) {
            (Some(x), Some(y)) => x.cmp(&y),
            _ => a.cmp(b),
        },
        CellFamily::Float => match (float_from_cell(a), float_from_cell(b)) {
            (Some(x), Some(y)) => x.total_cmp(&y),
            _ => a.cmp(b),
        },
        CellFamily::Str | CellFamily::Bytes | CellFamily::Unordered => a.cmp(b),
    }
}

/// Compares one cell against a typed constant without allocating, the
/// per-row primitive predicate evaluation runs on. None means the pair is
/// not comparable and the row outcome is unknown
pub(crate) fn compare_cell_to_value(
    physical: TypeId,
    cell: &[u8],
    value: &LakeValue,
) -> Option<Ordering> {
    match cell_family(physical) {
        CellFamily::Bool => match value {
            LakeValue::Bool(v) => cell.first().map(|b| (*b != 0).cmp(v)),
            _ => None,
        },
        CellFamily::SignedInt => compare_i128_to_value(signed_from_cell(cell)?, value),
        CellFamily::UnsignedInt => compare_u128_to_value(unsigned_from_cell(cell)?, value),
        CellFamily::Float => match value {
            LakeValue::Float(v) => float_from_cell(cell).map(|x| x.total_cmp(v)),
            _ => None,
        },
        CellFamily::Str => match value {
            LakeValue::Str(s) => Some(cell.cmp(s.as_bytes())),
            _ => None,
        },
        CellFamily::Bytes => match value {
            LakeValue::Bytes(b) => Some(cell.cmp(b.as_slice())),
            _ => None,
        },
        CellFamily::Unordered => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_signed_widths_and_negatives() {
        let a = (-5i32).to_le_bytes();
        let b = 3i32.to_le_bytes();
        assert_eq!(compare_cells(TypeId::Int32, &a, &b), Ordering::Less);
        assert_eq!(cell_to_value(TypeId::Int32, &a), Some(LakeValue::Int(-5)));
        let ts = (-86_400_000_000i64).to_le_bytes();
        assert_eq!(
            cell_to_value(TypeId::Timestamp, &ts),
            Some(LakeValue::Int(-86_400_000_000))
        );
        let d = i128::MIN.to_le_bytes();
        assert_eq!(
            cell_to_value(TypeId::Decimal, &d),
            Some(LakeValue::Int128(i128::MIN))
        );
    }

    #[test]
    fn test_cell_vs_value_comparisons() {
        let cell = 7u16.to_le_bytes();
        assert_eq!(
            compare_cell_to_value(TypeId::UInt16, &cell, &LakeValue::Int(-1)),
            Some(Ordering::Greater)
        );
        let f = 1.5f32.to_le_bytes();
        assert_eq!(
            compare_cell_to_value(TypeId::Float32, &f, &LakeValue::Float(1.5)),
            Some(Ordering::Equal)
        );
        assert_eq!(
            compare_cell_to_value(TypeId::Varchar, b"apple", &LakeValue::Str("banana".into())),
            Some(Ordering::Less)
        );
        assert_eq!(
            compare_cell_to_value(TypeId::Varchar, b"a", &LakeValue::Int(1)),
            None
        );
        // Unordered types never compare
        assert_eq!(
            compare_cell_to_value(TypeId::Geometry, &[1, 2, 3], &LakeValue::Bytes(vec![1])),
            None
        );
        assert_eq!(cell_to_value(TypeId::Geometry, &[1, 2, 3]), None);
    }

    #[test]
    fn test_malformed_width_is_not_comparable() {
        assert_eq!(
            compare_cell_to_value(TypeId::Int64, &[1, 2, 3], &LakeValue::Int(1)),
            None
        );
        assert_eq!(cell_to_value(TypeId::Int64, &[1, 2, 3]), None);
    }
}
