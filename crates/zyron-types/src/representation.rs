//! Whether every encoded value of one column shape decodes as a valid value
//! of another without loss.
//!
//! A column's stored bytes are written under the shape the column had at the
//! time. Changing the shape therefore asks one question: can the bytes already
//! on disk be read as the new shape. When the answer is yes the change is a
//! catalog write and the decoder widens each value as it reads it. When the
//! answer is no every row has to be re-encoded, which is a rewrite.
//!
//! The answer is total over the whole domain of the source shape, not over the
//! values a particular table happens to hold. A BIGINT column whose values all
//! fit in an INT still answers no, because the shape permits values that do
//! not, and a rule that consulted the data would give two tables the same
//! declaration and different behaviour.

use zyron_common::TypeId;

/// The physical shape of one column: what it stores and how wide.
///
/// `max_length` carries the declared bound a sized type spells out, which is
/// the character count for CHAR and VARCHAR, the byte count for BINARY and
/// VARBINARY, the dimension count for VECTOR, and the precision for DECIMAL.
/// `fractional_digits` carries the digits kept below the point, which is the
/// fractional-second precision of a TIMESTAMP and the scale of a DECIMAL.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Representation {
    pub type_id: TypeId,
    pub max_length: Option<usize>,
    pub fractional_digits: Option<u8>,
    pub nullable: bool,
}

impl Representation {
    /// A shape with no declared bound and no fractional digits.
    pub fn plain(type_id: TypeId, nullable: bool) -> Self {
        Self {
            type_id,
            max_length: None,
            fractional_digits: None,
            nullable,
        }
    }

    /// A sized shape, for the types that declare a bound.
    pub fn sized(type_id: TypeId, max_length: Option<usize>, nullable: bool) -> Self {
        Self {
            type_id,
            max_length,
            fractional_digits: None,
            nullable,
        }
    }

    /// The digits below the point, with the defaults each type applies when
    /// the declaration leaves them out. A TIMESTAMP defaults to 6, which is
    /// microseconds. A DECIMAL defaults to 0.
    fn digits(&self) -> u8 {
        match self.type_id {
            TypeId::Timestamp | TypeId::TimestampTz | TypeId::Time => {
                self.fractional_digits.unwrap_or(6)
            }
            _ => self.fractional_digits.unwrap_or(0),
        }
    }

    /// Digits above the point a DECIMAL can hold, precision minus scale. An
    /// undeclared precision is the widest an i128 carries.
    fn integral_digits(&self) -> usize {
        let precision = self.max_length.unwrap_or(DECIMAL_MAX_PRECISION);
        precision.saturating_sub(self.digits() as usize)
    }
}

/// Widest precision a scaled i128 decimal holds.
const DECIMAL_MAX_PRECISION: usize = 38;

/// Rank of an integer type within its signedness family. A higher rank holds
/// every value of a lower one, which is what makes the widening total.
fn signed_rank(t: TypeId) -> Option<u8> {
    match t {
        TypeId::Int8 => Some(1),
        TypeId::Int16 => Some(2),
        TypeId::Int32 => Some(3),
        TypeId::Int64 => Some(4),
        TypeId::Int128 => Some(5),
        _ => None,
    }
}

fn unsigned_rank(t: TypeId) -> Option<u8> {
    match t {
        TypeId::UInt8 => Some(1),
        TypeId::UInt16 => Some(2),
        TypeId::UInt32 => Some(3),
        TypeId::UInt64 => Some(4),
        TypeId::UInt128 => Some(5),
        _ => None,
    }
}

/// True when the type stores its bytes as text with a declared character
/// bound, so widening the bound leaves every stored value valid.
fn is_text(t: TypeId) -> bool {
    matches!(t, TypeId::Char | TypeId::Varchar | TypeId::Text)
}

/// True when the type stores opaque bytes with a declared length bound.
fn is_binary(t: TypeId) -> bool {
    matches!(t, TypeId::Binary | TypeId::Varbinary | TypeId::Bytea)
}

/// True when the type keeps a declared bound at all. TEXT and BYTEA are
/// unbounded, so they absorb any bound below them.
fn is_unbounded(t: TypeId) -> bool {
    matches!(t, TypeId::Text | TypeId::Bytea)
}

/// True when the bound `to` declares covers every value the bound `from`
/// permits. An unbounded target covers a bounded source; a bounded target
/// covers a bounded source only when it is at least as wide, and never
/// covers an unbounded one.
fn bound_widens(from: &Representation, to: &Representation) -> bool {
    if is_unbounded(to.type_id) {
        return true;
    }
    if is_unbounded(from.type_id) {
        return false;
    }
    match (from.max_length, to.max_length) {
        (Some(f), Some(t)) => t >= f,
        // An undeclared bound on the source permits any width, which a
        // declared bound on the target does not cover
        (None, Some(_)) => false,
        (_, None) => true,
    }
}

/// True exactly when every encoded value of `from` decodes as a valid `to`
/// without loss.
///
/// The four families that answer yes are integer widening within one
/// signedness, text and binary length widening, fractional-digit widening for
/// the two types that declare digits, and relaxing NOT NULL to nullable. An
/// identical pair answers yes. Everything else answers no, including every
/// narrowing, every change of storage family, and every change that would
/// have to reinterpret the bytes rather than extend them.
pub fn representation_compatible(from: Representation, to: Representation) -> bool {
    // Tightening nullability is a constraint the stored rows were never held
    // to, so it cannot be settled by reading them
    if from.nullable && !to.nullable {
        return false;
    }

    if from.type_id == to.type_id {
        return same_type_widens(&from, &to);
    }

    if let (Some(f), Some(t)) = (signed_rank(from.type_id), signed_rank(to.type_id)) {
        return t >= f;
    }
    if let (Some(f), Some(t)) = (unsigned_rank(from.type_id), unsigned_rank(to.type_id)) {
        return t >= f;
    }
    if is_text(from.type_id) && is_text(to.type_id) {
        // A CHAR is stored space padded to its declared width, so reading it
        // as a VARCHAR or TEXT keeps the padding rather than the value the
        // column was given. Only widening within the padded family or into a
        // wider CHAR leaves the bytes meaning what they meant
        if from.type_id == TypeId::Char && to.type_id != TypeId::Char {
            return false;
        }
        if to.type_id == TypeId::Char && from.type_id != TypeId::Char {
            return false;
        }
        return bound_widens(&from, &to);
    }
    if is_binary(from.type_id) && is_binary(to.type_id) {
        if from.type_id == TypeId::Binary && to.type_id != TypeId::Binary {
            return false;
        }
        if to.type_id == TypeId::Binary && from.type_id != TypeId::Binary {
            return false;
        }
        return bound_widens(&from, &to);
    }
    // A TIMESTAMP and a TIMESTAMPTZ hold the same instant at the same width,
    // but the zoned reading attaches an offset the unzoned bytes never
    // carried, so neither direction is a pure widening
    false
}

/// The widening rules for a pair that names one type, where only the declared
/// bound and the digits can differ.
fn same_type_widens(from: &Representation, to: &Representation) -> bool {
    match from.type_id {
        TypeId::Decimal => {
            // Scaling up multiplies the stored i128 by a power of ten, so the
            // digits above the point must still fit afterwards
            to.digits() >= from.digits() && to.integral_digits() >= from.integral_digits()
        }
        TypeId::Timestamp | TypeId::TimestampTz | TypeId::Time => to.digits() >= from.digits(),
        TypeId::Char
        | TypeId::Varchar
        | TypeId::Text
        | TypeId::Binary
        | TypeId::Varbinary
        | TypeId::Bytea
        | TypeId::Vector => bound_widens(from, to),
        // Every other type stores one fixed shape, so naming it twice is the
        // same shape
        _ => true,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn int(t: TypeId) -> Representation {
        Representation::plain(t, true)
    }

    fn varchar(n: usize) -> Representation {
        Representation::sized(TypeId::Varchar, Some(n), true)
    }

    fn decimal(p: usize, s: u8) -> Representation {
        Representation {
            type_id: TypeId::Decimal,
            max_length: Some(p),
            fractional_digits: Some(s),
            nullable: true,
        }
    }

    fn timestamp(p: u8) -> Representation {
        Representation {
            type_id: TypeId::Timestamp,
            max_length: None,
            fractional_digits: Some(p),
            nullable: true,
        }
    }

    #[test]
    fn test_integer_widening_is_compatible_in_one_direction() {
        assert!(representation_compatible(
            int(TypeId::Int32),
            int(TypeId::Int64)
        ));
        assert!(!representation_compatible(
            int(TypeId::Int64),
            int(TypeId::Int32)
        ));
        assert!(representation_compatible(
            int(TypeId::Int8),
            int(TypeId::Int128)
        ));
    }

    #[test]
    fn test_signedness_never_crosses() {
        assert!(!representation_compatible(
            int(TypeId::Int32),
            int(TypeId::UInt64)
        ));
        assert!(!representation_compatible(
            int(TypeId::UInt32),
            int(TypeId::Int64)
        ));
    }

    #[test]
    fn test_text_length_widening() {
        assert!(representation_compatible(varchar(10), varchar(40)));
        assert!(!representation_compatible(varchar(40), varchar(10)));
        assert!(representation_compatible(
            varchar(10),
            Representation::plain(TypeId::Text, true)
        ));
        assert!(!representation_compatible(
            Representation::plain(TypeId::Text, true),
            varchar(40)
        ));
    }

    #[test]
    fn test_timestamp_digit_widening() {
        assert!(representation_compatible(timestamp(3), timestamp(6)));
        assert!(!representation_compatible(timestamp(6), timestamp(3)));
        assert!(representation_compatible(timestamp(6), timestamp(9)));
    }

    #[test]
    fn test_decimal_widening_keeps_the_integral_digits() {
        assert!(representation_compatible(decimal(10, 2), decimal(14, 2)));
        assert!(!representation_compatible(decimal(14, 2), decimal(10, 2)));
        // Scaling 10,2 to 12,4 keeps eight digits above the point
        assert!(representation_compatible(decimal(10, 2), decimal(12, 4)));
        // Scaling 10,2 to 10,4 would drop two digits above the point
        assert!(!representation_compatible(decimal(10, 2), decimal(10, 4)));
    }

    #[test]
    fn test_unrelated_families_never_widen() {
        assert!(!representation_compatible(
            Representation::plain(TypeId::Text, true),
            int(TypeId::Int32)
        ));
        assert!(!representation_compatible(
            int(TypeId::Int32),
            Representation::plain(TypeId::Text, true)
        ));
        assert!(!representation_compatible(
            timestamp(3),
            Representation::plain(TypeId::TimestampTz, true)
        ));
    }

    #[test]
    fn test_nullability_relaxes_but_does_not_tighten() {
        let nullable = Representation::plain(TypeId::Int32, true);
        let not_null = Representation::plain(TypeId::Int32, false);
        assert!(representation_compatible(not_null, nullable));
        assert!(!representation_compatible(nullable, not_null));
    }

    #[test]
    fn test_char_padding_does_not_cross_into_varchar() {
        let ch = Representation::sized(TypeId::Char, Some(10), true);
        assert!(!representation_compatible(ch, varchar(40)));
        assert!(representation_compatible(
            ch,
            Representation::sized(TypeId::Char, Some(40), true)
        ));
    }
}
