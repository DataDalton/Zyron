//! Widening a column's stored cells to the type the column declares now.
//!
//! A data file is immutable and its cells are the width the column had
//! when the file was written. A column whose type widened since is read at
//! the width its files carry and widened on the way out, so a type change
//! the stored bytes already fit in never rewrites a file. The widenings
//! are the ones a heap tuple reads through as well, a signed integer into
//! a wider signed one, an unsigned integer into a wider unsigned one, a
//! microsecond instant into a picosecond one, and a decimal whose scale
//! grew. A change of any other kind is not a widening, and reading cells
//! through it is refused rather than guessed at

use zyron_common::curve::{CellFamily, cell_family};
use zyron_common::{TypeId, ZyronError};

use crate::predicate::LakeValue;

/// How stored cells of one shape become cells of a wider one
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Widening {
    /// Sign extended from `from` bytes to `to` bytes
    SignExtend { from: usize, to: usize },
    /// Zero extended from `from` bytes to `to` bytes
    ZeroExtend { from: usize, to: usize },
    /// An eight byte microsecond instant into a sixteen byte picosecond one
    MicrosToPicos,
    /// A sixteen byte scaled decimal multiplied by ten to the power `by`
    DecimalRescale { by: u32 },
}

/// The shape a column's cells take, a logical type and the digits it
/// declares, which together fix the physical type and the scale
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CellShape {
    pub type_id: TypeId,
    pub fractional_digits: Option<u8>,
}

impl CellShape {
    pub fn new(type_id: TypeId, fractional_digits: Option<u8>) -> Self {
        Self {
            type_id,
            fractional_digits,
        }
    }

    /// The type the cell bytes are laid out as
    pub fn physical(&self) -> TypeId {
        TypeId::timestamp_physical_type_id(self.type_id, self.fractional_digits)
    }

    /// Bytes per cell, zero for a variable-length shape
    pub fn width(&self) -> usize {
        self.physical().fixed_size().unwrap_or(0)
    }
}

/// Picoseconds per microsecond, the factor an instant grows by when its
/// column moves from microsecond to picosecond precision
pub const PICOS_PER_MICRO: i128 = 1_000_000;

/// The widening that reads cells written as `from` into the shape `to`,
/// None when the two shapes are the same bytes.
///
/// An error names a pair no widening covers, such as a narrowing or a
/// change of family, so a reader never decodes cells at a width they do
/// not have
pub fn widening_between(from: CellShape, to: CellShape) -> Result<Option<Widening>, ZyronError> {
    if from == to {
        return Ok(None);
    }
    let refuse = || {
        ZyronError::Internal(format!(
            "cells written as {:?} with {:?} digits cannot be read as {:?} with {:?} digits, \
             the change is not a widening",
            from.type_id, from.fractional_digits, to.type_id, to.fractional_digits
        ))
    };
    let (from_physical, to_physical) = (from.physical(), to.physical());
    if from.type_id == to.type_id {
        return match from.type_id {
            TypeId::Decimal => {
                let from_scale = u32::from(from.fractional_digits.unwrap_or(0));
                let to_scale = u32::from(to.fractional_digits.unwrap_or(0));
                match to_scale.checked_sub(from_scale) {
                    Some(0) => Ok(None),
                    Some(by) => Ok(Some(Widening::DecimalRescale { by })),
                    None => Err(refuse()),
                }
            }
            TypeId::Timestamp | TypeId::TimestampTz => {
                let from_digits = from.fractional_digits.unwrap_or(6);
                let to_digits = to.fractional_digits.unwrap_or(6);
                if to_digits < from_digits {
                    return Err(refuse());
                }
                // Six digits and fewer store an eight byte microsecond
                // instant, more store a sixteen byte picosecond one
                let crosses = to_physical == TypeId::Int128 && from_physical != TypeId::Int128;
                Ok(crosses.then_some(Widening::MicrosToPicos))
            }
            // Digits on any other type do not change the stored bytes
            _ => Ok(None),
        };
    }
    let (from_width, to_width) = (from.width(), to.width());
    match (cell_family(from_physical), cell_family(to_physical)) {
        (CellFamily::SignedInt, CellFamily::SignedInt)
            if is_plain_signed(from_physical) && is_plain_signed(to_physical) =>
        {
            if to_width <= from_width {
                return Err(refuse());
            }
            Ok(Some(Widening::SignExtend {
                from: from_width,
                to: to_width,
            }))
        }
        (CellFamily::UnsignedInt, CellFamily::UnsignedInt)
            if is_plain_unsigned(from_physical) && is_plain_unsigned(to_physical) =>
        {
            if to_width <= from_width {
                return Err(refuse());
            }
            Ok(Some(Widening::ZeroExtend {
                from: from_width,
                to: to_width,
            }))
        }
        // A text or byte column widened in its declared bound, or moved to
        // an unbounded type of the same family, keeps every cell as it is
        (CellFamily::Str, CellFamily::Str) | (CellFamily::Bytes, CellFamily::Bytes)
            if from_width == 0 && to_width == 0 =>
        {
            Ok(None)
        }
        _ => Err(refuse()),
    }
}

/// The signed integer types a wider signed integer extends
fn is_plain_signed(t: TypeId) -> bool {
    matches!(
        t,
        TypeId::Int8 | TypeId::Int16 | TypeId::Int32 | TypeId::Int64 | TypeId::Int128
    )
}

/// The unsigned integer types a wider unsigned integer extends
fn is_plain_unsigned(t: TypeId) -> bool {
    matches!(
        t,
        TypeId::UInt8 | TypeId::UInt16 | TypeId::UInt32 | TypeId::UInt64 | TypeId::UInt128
    )
}

/// Widens `rows` fixed width cells laid out back to back in `data`.
///
/// A null cell is stored as zero bytes and widens to zero bytes, so the
/// null bitmap the cells came with stays the authority on nullness. One
/// pass over the cells, writing straight into the wider buffer
pub fn widen_cells(widening: Widening, data: &[u8], rows: usize) -> Result<Vec<u8>, ZyronError> {
    let (from, to) = match widening {
        Widening::SignExtend { from, to } | Widening::ZeroExtend { from, to } => (from, to),
        Widening::MicrosToPicos => (8, 16),
        Widening::DecimalRescale { .. } => (16, 16),
    };
    if data.len() != rows * from {
        return Err(ZyronError::Internal(format!(
            "a column of {rows} cells at {from} bytes each holds {} bytes",
            data.len()
        )));
    }
    let mut out = vec![0u8; rows * to];
    match widening {
        Widening::SignExtend { .. } => {
            for (cell, wide) in data.chunks_exact(from).zip(out.chunks_exact_mut(to)) {
                wide[..from].copy_from_slice(cell);
                if cell[from - 1] & 0x80 != 0 {
                    wide[from..].fill(0xFF);
                }
            }
        }
        Widening::ZeroExtend { .. } => {
            for (cell, wide) in data.chunks_exact(from).zip(out.chunks_exact_mut(to)) {
                wide[..from].copy_from_slice(cell);
            }
        }
        Widening::MicrosToPicos => {
            for (cell, wide) in data.chunks_exact(8).zip(out.chunks_exact_mut(16)) {
                let mut raw = [0u8; 8];
                raw.copy_from_slice(cell);
                let micros = i64::from_le_bytes(raw) as i128;
                wide.copy_from_slice(&(micros * PICOS_PER_MICRO).to_le_bytes());
            }
        }
        Widening::DecimalRescale { by } => {
            let factor = 10i128.checked_pow(by).ok_or_else(|| {
                ZyronError::Internal(format!(
                    "a decimal scale cannot grow by {by} digits inside a sixteen byte value"
                ))
            })?;
            for (cell, wide) in data.chunks_exact(16).zip(out.chunks_exact_mut(16)) {
                let mut raw = [0u8; 16];
                raw.copy_from_slice(cell);
                let scaled = i128::from_le_bytes(raw)
                    .checked_mul(factor)
                    .ok_or_else(|| {
                        ZyronError::Internal(format!(
                            "a stored decimal does not fit its column's scale grown by {by} digits"
                        ))
                    })?;
                wide.copy_from_slice(&scaled.to_le_bytes());
            }
        }
    }
    Ok(out)
}

/// Widens one recorded value, a file's bound or a constant, the way
/// `widen_cells` widens the cells it describes. None for a value that is
/// not of the family the widening moves, which a caller treats as unknown
/// rather than as a value on the wrong scale
pub fn widen_value(widening: Widening, value: &LakeValue) -> Option<LakeValue> {
    let as_i128 = match value {
        LakeValue::Int(v) => *v as i128,
        LakeValue::Int128(v) => *v,
        LakeValue::UInt(v) => *v as i128,
        LakeValue::UInt128(v) => i128::try_from(*v).ok()?,
        _ => return None,
    };
    match widening {
        // A wider integer holds the same value
        Widening::SignExtend { .. } | Widening::ZeroExtend { .. } => Some(value.clone()),
        Widening::MicrosToPicos => as_i128.checked_mul(PICOS_PER_MICRO).map(LakeValue::Int128),
        Widening::DecimalRescale { by } => 10i128
            .checked_pow(by)
            .and_then(|factor| as_i128.checked_mul(factor))
            .map(LakeValue::Int128),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn shape(type_id: TypeId) -> CellShape {
        CellShape::new(type_id, None)
    }

    #[test]
    fn test_integer_widenings_extend_by_sign_and_refuse_the_reverse() {
        assert_eq!(
            widening_between(shape(TypeId::Int32), shape(TypeId::Int64)).expect("widens"),
            Some(Widening::SignExtend { from: 4, to: 8 })
        );
        assert_eq!(
            widening_between(shape(TypeId::UInt8), shape(TypeId::UInt128)).expect("widens"),
            Some(Widening::ZeroExtend { from: 1, to: 16 })
        );
        assert_eq!(
            widening_between(shape(TypeId::Int64), shape(TypeId::Int64)).expect("same"),
            None
        );
        assert!(widening_between(shape(TypeId::Int64), shape(TypeId::Int32)).is_err());
        assert!(widening_between(shape(TypeId::Int32), shape(TypeId::UInt64)).is_err());
        assert!(widening_between(shape(TypeId::Date), shape(TypeId::Int64)).is_err());
        assert!(widening_between(shape(TypeId::Int32), shape(TypeId::Float64)).is_err());
    }

    #[test]
    fn test_text_and_byte_bounds_change_no_cell() {
        assert_eq!(
            widening_between(
                CellShape::new(TypeId::Varchar, None),
                CellShape::new(TypeId::Text, None)
            )
            .expect("same bytes"),
            None
        );
        assert_eq!(
            widening_between(shape(TypeId::Varbinary), shape(TypeId::Bytea)).expect("same bytes"),
            None
        );
        assert!(widening_between(shape(TypeId::Varchar), shape(TypeId::Bytea)).is_err());
    }

    #[test]
    fn test_sign_extension_keeps_every_value() {
        let values: [i32; 4] = [-1, i32::MIN, i32::MAX, 0];
        let mut data = Vec::new();
        for v in values {
            data.extend_from_slice(&v.to_le_bytes());
        }
        let wide = widen_cells(Widening::SignExtend { from: 4, to: 8 }, &data, 4).expect("widens");
        for (i, v) in values.iter().enumerate() {
            let mut raw = [0u8; 8];
            raw.copy_from_slice(&wide[i * 8..(i + 1) * 8]);
            assert_eq!(i64::from_le_bytes(raw), *v as i64);
        }
        let narrow =
            widen_cells(Widening::ZeroExtend { from: 2, to: 4 }, &[0xFF, 0xFF], 1).expect("widens");
        assert_eq!(
            u32::from_le_bytes([narrow[0], narrow[1], narrow[2], narrow[3]]),
            65_535
        );
        assert!(
            widen_cells(Widening::SignExtend { from: 4, to: 8 }, &data[..6], 2).is_err(),
            "a buffer that is not whole cells is refused"
        );
    }

    #[test]
    fn test_instants_and_decimals_rescale_exactly() {
        let micros: i64 = -1_234_567;
        let wide = widen_cells(Widening::MicrosToPicos, &micros.to_le_bytes(), 1).expect("widens");
        let mut raw = [0u8; 16];
        raw.copy_from_slice(&wide);
        assert_eq!(i128::from_le_bytes(raw), micros as i128 * PICOS_PER_MICRO);

        let stored: i128 = -12_345;
        let wide = widen_cells(Widening::DecimalRescale { by: 3 }, &stored.to_le_bytes(), 1)
            .expect("rescales");
        raw.copy_from_slice(&wide);
        assert_eq!(i128::from_le_bytes(raw), stored * 1_000);
        assert!(
            widen_cells(
                Widening::DecimalRescale { by: 2 },
                &i128::MAX.to_le_bytes(),
                1
            )
            .is_err(),
            "a value the grown scale cannot hold is reported"
        );

        assert_eq!(
            widening_between(
                CellShape::new(TypeId::Timestamp, Some(6)),
                CellShape::new(TypeId::Timestamp, Some(9))
            )
            .expect("widens"),
            Some(Widening::MicrosToPicos)
        );
        assert_eq!(
            widening_between(
                CellShape::new(TypeId::Timestamp, Some(3)),
                CellShape::new(TypeId::Timestamp, Some(6))
            )
            .expect("same bytes"),
            None
        );
        assert_eq!(
            widening_between(
                CellShape::new(TypeId::Decimal, Some(2)),
                CellShape::new(TypeId::Decimal, Some(4))
            )
            .expect("rescales"),
            Some(Widening::DecimalRescale { by: 2 })
        );
        assert!(
            widening_between(
                CellShape::new(TypeId::Decimal, Some(4)),
                CellShape::new(TypeId::Decimal, Some(2))
            )
            .is_err()
        );
    }
}
