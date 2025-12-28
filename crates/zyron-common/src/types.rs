//! Type identifiers for ZyronDB data types.

use serde::{Deserialize, Serialize};

/// Identifier for all supported data types in ZyronDB.
///
/// Type IDs are stored in tuple headers and catalog metadata
/// to identify the type of each column value.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[repr(u8)]
pub enum TypeId {
    // Null type
    Null = 0,

    // Boolean
    Boolean = 1,

    // Integer types
    Int8 = 10,
    Int16 = 11,
    Int32 = 12,
    Int64 = 13,
    Int128 = 14,

    // Unsigned integer types
    UInt8 = 20,
    UInt16 = 21,
    UInt32 = 22,
    UInt64 = 23,
    UInt128 = 24,

    // Floating point types
    Float32 = 30,
    Float64 = 31,

    // Fixed-precision decimal
    Decimal = 40,

    // String types
    Char = 50,
    Varchar = 51,
    Text = 52,

    // Binary types
    Binary = 60,
    Varbinary = 61,
    Blob = 62,

    // Date/Time types
    Date = 70,
    Time = 71,
    Timestamp = 72,
    TimestampTz = 73,
    Interval = 74,

    // UUID
    Uuid = 80,

    // JSON
    Json = 90,
    Jsonb = 91,

    // Array (element type stored separately)
    Array = 100,

    // Composite/Struct (field types stored in catalog)
    Composite = 110,
}

impl TypeId {
    /// Returns the fixed byte size for this type, or None for variable-length types.
    pub fn fixed_size(&self) -> Option<usize> {
        match self {
            TypeId::Null => Some(0),
            TypeId::Boolean => Some(1),

            TypeId::Int8 | TypeId::UInt8 => Some(1),
            TypeId::Int16 | TypeId::UInt16 => Some(2),
            TypeId::Int32 | TypeId::UInt32 | TypeId::Float32 => Some(4),
            TypeId::Int64 | TypeId::UInt64 | TypeId::Float64 => Some(8),
            TypeId::Int128 | TypeId::UInt128 | TypeId::Decimal => Some(16),

            TypeId::Date => Some(4),
            TypeId::Time => Some(8),
            TypeId::Timestamp | TypeId::TimestampTz => Some(8),
            TypeId::Interval => Some(16),

            TypeId::Uuid => Some(16),

            // Variable-length types
            TypeId::Char
            | TypeId::Varchar
            | TypeId::Text
            | TypeId::Binary
            | TypeId::Varbinary
            | TypeId::Blob
            | TypeId::Json
            | TypeId::Jsonb
            | TypeId::Array
            | TypeId::Composite => None,
        }
    }

    /// Returns true if this type has a fixed byte size.
    pub fn is_fixed_size(&self) -> bool {
        self.fixed_size().is_some()
    }

    /// Returns true if this type is a numeric type.
    pub fn is_numeric(&self) -> bool {
        matches!(
            self,
            TypeId::Int8
                | TypeId::Int16
                | TypeId::Int32
                | TypeId::Int64
                | TypeId::Int128
                | TypeId::UInt8
                | TypeId::UInt16
                | TypeId::UInt32
                | TypeId::UInt64
                | TypeId::UInt128
                | TypeId::Float32
                | TypeId::Float64
                | TypeId::Decimal
        )
    }

    /// Returns true if this type is an integer type (signed or unsigned).
    pub fn is_integer(&self) -> bool {
        matches!(
            self,
            TypeId::Int8
                | TypeId::Int16
                | TypeId::Int32
                | TypeId::Int64
                | TypeId::Int128
                | TypeId::UInt8
                | TypeId::UInt16
                | TypeId::UInt32
                | TypeId::UInt64
                | TypeId::UInt128
        )
    }

    /// Returns true if this type is a floating-point type.
    pub fn is_floating_point(&self) -> bool {
        matches!(self, TypeId::Float32 | TypeId::Float64)
    }

    /// Returns true if this type is a string type.
    pub fn is_string(&self) -> bool {
        matches!(self, TypeId::Char | TypeId::Varchar | TypeId::Text)
    }

    /// Returns true if this type is a binary type.
    pub fn is_binary(&self) -> bool {
        matches!(self, TypeId::Binary | TypeId::Varbinary | TypeId::Blob)
    }

    /// Returns true if this type is a temporal type.
    pub fn is_temporal(&self) -> bool {
        matches!(
            self,
            TypeId::Date
                | TypeId::Time
                | TypeId::Timestamp
                | TypeId::TimestampTz
                | TypeId::Interval
        )
    }
}

impl std::fmt::Display for TypeId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let name = match self {
            TypeId::Null => "NULL",
            TypeId::Boolean => "BOOLEAN",
            TypeId::Int8 => "INT8",
            TypeId::Int16 => "INT16",
            TypeId::Int32 => "INT32",
            TypeId::Int64 => "INT64",
            TypeId::Int128 => "INT128",
            TypeId::UInt8 => "UINT8",
            TypeId::UInt16 => "UINT16",
            TypeId::UInt32 => "UINT32",
            TypeId::UInt64 => "UINT64",
            TypeId::UInt128 => "UINT128",
            TypeId::Float32 => "FLOAT32",
            TypeId::Float64 => "FLOAT64",
            TypeId::Decimal => "DECIMAL",
            TypeId::Char => "CHAR",
            TypeId::Varchar => "VARCHAR",
            TypeId::Text => "TEXT",
            TypeId::Binary => "BINARY",
            TypeId::Varbinary => "VARBINARY",
            TypeId::Blob => "BLOB",
            TypeId::Date => "DATE",
            TypeId::Time => "TIME",
            TypeId::Timestamp => "TIMESTAMP",
            TypeId::TimestampTz => "TIMESTAMPTZ",
            TypeId::Interval => "INTERVAL",
            TypeId::Uuid => "UUID",
            TypeId::Json => "JSON",
            TypeId::Jsonb => "JSONB",
            TypeId::Array => "ARRAY",
            TypeId::Composite => "COMPOSITE",
        };
        write!(f, "{}", name)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fixed_size_integers() {
        assert_eq!(TypeId::Int8.fixed_size(), Some(1));
        assert_eq!(TypeId::Int16.fixed_size(), Some(2));
        assert_eq!(TypeId::Int32.fixed_size(), Some(4));
        assert_eq!(TypeId::Int64.fixed_size(), Some(8));
        assert_eq!(TypeId::Int128.fixed_size(), Some(16));

        assert_eq!(TypeId::UInt8.fixed_size(), Some(1));
        assert_eq!(TypeId::UInt16.fixed_size(), Some(2));
        assert_eq!(TypeId::UInt32.fixed_size(), Some(4));
        assert_eq!(TypeId::UInt64.fixed_size(), Some(8));
        assert_eq!(TypeId::UInt128.fixed_size(), Some(16));
    }

    #[test]
    fn test_fixed_size_floats() {
        assert_eq!(TypeId::Float32.fixed_size(), Some(4));
        assert_eq!(TypeId::Float64.fixed_size(), Some(8));
    }

    #[test]
    fn test_fixed_size_temporal() {
        assert_eq!(TypeId::Date.fixed_size(), Some(4));
        assert_eq!(TypeId::Time.fixed_size(), Some(8));
        assert_eq!(TypeId::Timestamp.fixed_size(), Some(8));
        assert_eq!(TypeId::TimestampTz.fixed_size(), Some(8));
        assert_eq!(TypeId::Interval.fixed_size(), Some(16));
    }

    #[test]
    fn test_fixed_size_other() {
        assert_eq!(TypeId::Null.fixed_size(), Some(0));
        assert_eq!(TypeId::Boolean.fixed_size(), Some(1));
        assert_eq!(TypeId::Decimal.fixed_size(), Some(16));
        assert_eq!(TypeId::Uuid.fixed_size(), Some(16));
    }

    #[test]
    fn test_variable_size_types() {
        assert_eq!(TypeId::Char.fixed_size(), None);
        assert_eq!(TypeId::Varchar.fixed_size(), None);
        assert_eq!(TypeId::Text.fixed_size(), None);
        assert_eq!(TypeId::Binary.fixed_size(), None);
        assert_eq!(TypeId::Varbinary.fixed_size(), None);
        assert_eq!(TypeId::Blob.fixed_size(), None);
        assert_eq!(TypeId::Json.fixed_size(), None);
        assert_eq!(TypeId::Jsonb.fixed_size(), None);
        assert_eq!(TypeId::Array.fixed_size(), None);
        assert_eq!(TypeId::Composite.fixed_size(), None);
    }

    #[test]
    fn test_is_fixed_size() {
        assert!(TypeId::Int64.is_fixed_size());
        assert!(TypeId::Boolean.is_fixed_size());
        assert!(TypeId::Uuid.is_fixed_size());

        assert!(!TypeId::Varchar.is_fixed_size());
        assert!(!TypeId::Text.is_fixed_size());
        assert!(!TypeId::Array.is_fixed_size());
    }

    #[test]
    fn test_is_numeric() {
        // Signed integers
        assert!(TypeId::Int8.is_numeric());
        assert!(TypeId::Int16.is_numeric());
        assert!(TypeId::Int32.is_numeric());
        assert!(TypeId::Int64.is_numeric());
        assert!(TypeId::Int128.is_numeric());

        // Unsigned integers
        assert!(TypeId::UInt8.is_numeric());
        assert!(TypeId::UInt16.is_numeric());
        assert!(TypeId::UInt32.is_numeric());
        assert!(TypeId::UInt64.is_numeric());
        assert!(TypeId::UInt128.is_numeric());

        // Floats and decimal
        assert!(TypeId::Float32.is_numeric());
        assert!(TypeId::Float64.is_numeric());
        assert!(TypeId::Decimal.is_numeric());

        // Non-numeric types
        assert!(!TypeId::Boolean.is_numeric());
        assert!(!TypeId::Varchar.is_numeric());
        assert!(!TypeId::Date.is_numeric());
        assert!(!TypeId::Uuid.is_numeric());
    }

    #[test]
    fn test_is_integer() {
        assert!(TypeId::Int32.is_integer());
        assert!(TypeId::Int64.is_integer());
        assert!(TypeId::UInt32.is_integer());
        assert!(TypeId::UInt64.is_integer());

        assert!(!TypeId::Float64.is_integer());
        assert!(!TypeId::Decimal.is_integer());
        assert!(!TypeId::Varchar.is_integer());
    }

    #[test]
    fn test_is_floating_point() {
        assert!(TypeId::Float32.is_floating_point());
        assert!(TypeId::Float64.is_floating_point());

        assert!(!TypeId::Int64.is_floating_point());
        assert!(!TypeId::Decimal.is_floating_point());
    }

    #[test]
    fn test_is_string() {
        assert!(TypeId::Char.is_string());
        assert!(TypeId::Varchar.is_string());
        assert!(TypeId::Text.is_string());

        assert!(!TypeId::Binary.is_string());
        assert!(!TypeId::Json.is_string());
        assert!(!TypeId::Int32.is_string());
    }

    #[test]
    fn test_is_binary() {
        assert!(TypeId::Binary.is_binary());
        assert!(TypeId::Varbinary.is_binary());
        assert!(TypeId::Blob.is_binary());

        assert!(!TypeId::Text.is_binary());
        assert!(!TypeId::Jsonb.is_binary());
    }

    #[test]
    fn test_is_temporal() {
        assert!(TypeId::Date.is_temporal());
        assert!(TypeId::Time.is_temporal());
        assert!(TypeId::Timestamp.is_temporal());
        assert!(TypeId::TimestampTz.is_temporal());
        assert!(TypeId::Interval.is_temporal());

        assert!(!TypeId::Int64.is_temporal());
        assert!(!TypeId::Varchar.is_temporal());
    }

    #[test]
    fn test_display() {
        assert_eq!(TypeId::Null.to_string(), "NULL");
        assert_eq!(TypeId::Boolean.to_string(), "BOOLEAN");
        assert_eq!(TypeId::Int64.to_string(), "INT64");
        assert_eq!(TypeId::Float64.to_string(), "FLOAT64");
        assert_eq!(TypeId::Varchar.to_string(), "VARCHAR");
        assert_eq!(TypeId::Timestamp.to_string(), "TIMESTAMP");
        assert_eq!(TypeId::TimestampTz.to_string(), "TIMESTAMPTZ");
        assert_eq!(TypeId::Uuid.to_string(), "UUID");
        assert_eq!(TypeId::Jsonb.to_string(), "JSONB");
        assert_eq!(TypeId::Array.to_string(), "ARRAY");
        assert_eq!(TypeId::Composite.to_string(), "COMPOSITE");
    }

    #[test]
    fn test_repr_u8_values() {
        assert_eq!(TypeId::Null as u8, 0);
        assert_eq!(TypeId::Boolean as u8, 1);
        assert_eq!(TypeId::Int8 as u8, 10);
        assert_eq!(TypeId::UInt8 as u8, 20);
        assert_eq!(TypeId::Float32 as u8, 30);
        assert_eq!(TypeId::Decimal as u8, 40);
        assert_eq!(TypeId::Char as u8, 50);
        assert_eq!(TypeId::Binary as u8, 60);
        assert_eq!(TypeId::Date as u8, 70);
        assert_eq!(TypeId::Uuid as u8, 80);
        assert_eq!(TypeId::Json as u8, 90);
        assert_eq!(TypeId::Array as u8, 100);
        assert_eq!(TypeId::Composite as u8, 110);
    }

    #[test]
    fn test_clone_copy() {
        let t1 = TypeId::Int64;
        let t2 = t1; // Copy
        let t3 = t1.clone(); // Clone
        assert_eq!(t1, t2);
        assert_eq!(t1, t3);
    }

    #[test]
    fn test_hash() {
        use std::collections::HashSet;

        let mut set = HashSet::new();
        set.insert(TypeId::Int32);
        set.insert(TypeId::Int64);
        set.insert(TypeId::Int32); // Duplicate

        assert_eq!(set.len(), 2);
        assert!(set.contains(&TypeId::Int32));
        assert!(set.contains(&TypeId::Int64));
    }

    #[test]
    fn test_serde_roundtrip() {
        use serde_json;

        let original = TypeId::Timestamp;
        let serialized = serde_json::to_string(&original).unwrap();
        let deserialized: TypeId = serde_json::from_str(&serialized).unwrap();
        assert_eq!(original, deserialized);
    }

    #[test]
    fn test_all_types_have_display() {
        let all_types = [
            TypeId::Null,
            TypeId::Boolean,
            TypeId::Int8,
            TypeId::Int16,
            TypeId::Int32,
            TypeId::Int64,
            TypeId::Int128,
            TypeId::UInt8,
            TypeId::UInt16,
            TypeId::UInt32,
            TypeId::UInt64,
            TypeId::UInt128,
            TypeId::Float32,
            TypeId::Float64,
            TypeId::Decimal,
            TypeId::Char,
            TypeId::Varchar,
            TypeId::Text,
            TypeId::Binary,
            TypeId::Varbinary,
            TypeId::Blob,
            TypeId::Date,
            TypeId::Time,
            TypeId::Timestamp,
            TypeId::TimestampTz,
            TypeId::Interval,
            TypeId::Uuid,
            TypeId::Json,
            TypeId::Jsonb,
            TypeId::Array,
            TypeId::Composite,
        ];

        for type_id in all_types {
            let display = type_id.to_string();
            assert!(!display.is_empty(), "TypeId {:?} has empty display", type_id);
        }
    }
}
