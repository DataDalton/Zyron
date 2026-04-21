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
    Bytea = 62,

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

    // Vector (fixed-dimension float array for similarity search)
    Vector = 120,

    // Geospatial (WKB binary, variable length)
    Geometry = 130,

    // Matrix (row-major f64 array with dimension header, variable length)
    Matrix = 140,

    // Color (RGBA packed into u32, 4 bytes)
    Color = 150,

    // Semantic versioning (packed u64: major|minor|patch|pre-release flag)
    SemVer = 160,

    // Network types
    Inet = 170,
    Cidr = 171,
    MacAddr = 172,

    // Currency-aware fixed-point money (i64 minor units + u16 currency code)
    Money = 180,

    // Range (element type stored separately in catalog, variable length)
    Range = 190,

    // Probabilistic data structures (variable length binary)
    HyperLogLog = 210,
    BloomFilter = 211,
    TDigest = 212,
    CountMinSketch = 213,

    // Bitfield (named bit positions in u64)
    Bitfield = 220,

    // Unit-aware quantity (f64 value + u16 unit identifier)
    Quantity = 240,
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

            // Extended fixed-size types
            TypeId::Color => Some(4),
            TypeId::SemVer | TypeId::Bitfield => Some(8),
            TypeId::MacAddr => Some(6),
            TypeId::Inet | TypeId::Cidr => Some(18),
            TypeId::Money | TypeId::Quantity => Some(10),

            // Variable-length types
            TypeId::Char
            | TypeId::Varchar
            | TypeId::Text
            | TypeId::Binary
            | TypeId::Varbinary
            | TypeId::Bytea
            | TypeId::Json
            | TypeId::Jsonb
            | TypeId::Array
            | TypeId::Composite
            | TypeId::Vector
            | TypeId::Geometry
            | TypeId::Matrix
            | TypeId::Range
            | TypeId::HyperLogLog
            | TypeId::BloomFilter
            | TypeId::TDigest
            | TypeId::CountMinSketch => None,
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

    /// Maps a discriminant byte back to a TypeId. Returns None for bytes
    /// that do not correspond to a known variant.
    pub fn from_u8(byte: u8) -> Option<Self> {
        Some(match byte {
            0 => TypeId::Null,
            1 => TypeId::Boolean,
            10 => TypeId::Int8,
            11 => TypeId::Int16,
            12 => TypeId::Int32,
            13 => TypeId::Int64,
            14 => TypeId::Int128,
            20 => TypeId::UInt8,
            21 => TypeId::UInt16,
            22 => TypeId::UInt32,
            23 => TypeId::UInt64,
            24 => TypeId::UInt128,
            30 => TypeId::Float32,
            31 => TypeId::Float64,
            40 => TypeId::Decimal,
            50 => TypeId::Char,
            51 => TypeId::Varchar,
            52 => TypeId::Text,
            60 => TypeId::Binary,
            61 => TypeId::Varbinary,
            62 => TypeId::Bytea,
            70 => TypeId::Date,
            71 => TypeId::Time,
            72 => TypeId::Timestamp,
            73 => TypeId::TimestampTz,
            74 => TypeId::Interval,
            80 => TypeId::Uuid,
            90 => TypeId::Json,
            91 => TypeId::Jsonb,
            100 => TypeId::Array,
            110 => TypeId::Composite,
            120 => TypeId::Vector,
            130 => TypeId::Geometry,
            140 => TypeId::Matrix,
            150 => TypeId::Color,
            160 => TypeId::SemVer,
            170 => TypeId::Inet,
            171 => TypeId::Cidr,
            172 => TypeId::MacAddr,
            180 => TypeId::Money,
            190 => TypeId::Range,
            210 => TypeId::HyperLogLog,
            211 => TypeId::BloomFilter,
            212 => TypeId::TDigest,
            213 => TypeId::CountMinSketch,
            220 => TypeId::Bitfield,
            240 => TypeId::Quantity,
            _ => return None,
        })
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
        matches!(self, TypeId::Binary | TypeId::Varbinary | TypeId::Bytea)
    }

    /// Returns true if this type is one of the extended types (spatial,
    /// temporal, probabilistic, etc.) that sit outside the core SQL type set.
    pub fn is_extended_type(&self) -> bool {
        matches!(
            self,
            TypeId::Geometry
                | TypeId::Matrix
                | TypeId::Color
                | TypeId::SemVer
                | TypeId::Inet
                | TypeId::Cidr
                | TypeId::MacAddr
                | TypeId::Money
                | TypeId::Range
                | TypeId::HyperLogLog
                | TypeId::BloomFilter
                | TypeId::TDigest
                | TypeId::CountMinSketch
                | TypeId::Bitfield
                | TypeId::Quantity
        )
    }

    /// Returns true if this type is a probabilistic data structure.
    pub fn is_probabilistic(&self) -> bool {
        matches!(
            self,
            TypeId::HyperLogLog | TypeId::BloomFilter | TypeId::TDigest | TypeId::CountMinSketch
        )
    }

    /// Returns true if this type is a network address type.
    pub fn is_network(&self) -> bool {
        matches!(self, TypeId::Inet | TypeId::Cidr | TypeId::MacAddr)
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
            TypeId::Bytea => "BYTEA",
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
            TypeId::Vector => "VECTOR",
            TypeId::Geometry => "GEOMETRY",
            TypeId::Matrix => "MATRIX",
            TypeId::Color => "COLOR",
            TypeId::SemVer => "SEMVER",
            TypeId::Inet => "INET",
            TypeId::Cidr => "CIDR",
            TypeId::MacAddr => "MACADDR",
            TypeId::Money => "MONEY",
            TypeId::Range => "RANGE",
            TypeId::HyperLogLog => "HYPERLOGLOG",
            TypeId::BloomFilter => "BLOOMFILTER",
            TypeId::TDigest => "TDIGEST",
            TypeId::CountMinSketch => "COUNTMINSKETCH",
            TypeId::Bitfield => "BITFIELD",
            TypeId::Quantity => "QUANTITY",
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
        assert_eq!(TypeId::Bytea.fixed_size(), None);
        assert_eq!(TypeId::Json.fixed_size(), None);
        assert_eq!(TypeId::Jsonb.fixed_size(), None);
        assert_eq!(TypeId::Array.fixed_size(), None);
        assert_eq!(TypeId::Composite.fixed_size(), None);
        assert_eq!(TypeId::Vector.fixed_size(), None);
        assert_eq!(TypeId::Geometry.fixed_size(), None);
        assert_eq!(TypeId::Matrix.fixed_size(), None);
        assert_eq!(TypeId::Range.fixed_size(), None);
        assert_eq!(TypeId::HyperLogLog.fixed_size(), None);
        assert_eq!(TypeId::BloomFilter.fixed_size(), None);
        assert_eq!(TypeId::TDigest.fixed_size(), None);
        assert_eq!(TypeId::CountMinSketch.fixed_size(), None);
    }

    #[test]
    fn test_extended_fixed_size_types() {
        assert_eq!(TypeId::Color.fixed_size(), Some(4));
        assert_eq!(TypeId::SemVer.fixed_size(), Some(8));
        assert_eq!(TypeId::Bitfield.fixed_size(), Some(8));
        assert_eq!(TypeId::MacAddr.fixed_size(), Some(6));
        assert_eq!(TypeId::Inet.fixed_size(), Some(18));
        assert_eq!(TypeId::Cidr.fixed_size(), Some(18));
        assert_eq!(TypeId::Money.fixed_size(), Some(10));
        assert_eq!(TypeId::Quantity.fixed_size(), Some(10));
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
        assert!(TypeId::Bytea.is_binary());

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
        assert_eq!(TypeId::Vector.to_string(), "VECTOR");
        assert_eq!(TypeId::Geometry.to_string(), "GEOMETRY");
        assert_eq!(TypeId::Matrix.to_string(), "MATRIX");
        assert_eq!(TypeId::Color.to_string(), "COLOR");
        assert_eq!(TypeId::SemVer.to_string(), "SEMVER");
        assert_eq!(TypeId::Inet.to_string(), "INET");
        assert_eq!(TypeId::Cidr.to_string(), "CIDR");
        assert_eq!(TypeId::MacAddr.to_string(), "MACADDR");
        assert_eq!(TypeId::Money.to_string(), "MONEY");
        assert_eq!(TypeId::Range.to_string(), "RANGE");
        assert_eq!(TypeId::HyperLogLog.to_string(), "HYPERLOGLOG");
        assert_eq!(TypeId::BloomFilter.to_string(), "BLOOMFILTER");
        assert_eq!(TypeId::TDigest.to_string(), "TDIGEST");
        assert_eq!(TypeId::CountMinSketch.to_string(), "COUNTMINSKETCH");
        assert_eq!(TypeId::Bitfield.to_string(), "BITFIELD");
        assert_eq!(TypeId::Quantity.to_string(), "QUANTITY");
    }

    #[test]
    fn test_is_extended_type() {
        assert!(TypeId::Geometry.is_extended_type());
        assert!(TypeId::Matrix.is_extended_type());
        assert!(TypeId::Color.is_extended_type());
        assert!(TypeId::SemVer.is_extended_type());
        assert!(TypeId::Money.is_extended_type());
        assert!(TypeId::HyperLogLog.is_extended_type());
        assert!(TypeId::Bitfield.is_extended_type());
        assert!(TypeId::Quantity.is_extended_type());

        assert!(!TypeId::Int64.is_extended_type());
        assert!(!TypeId::Varchar.is_extended_type());
        assert!(!TypeId::Vector.is_extended_type());
    }

    #[test]
    fn test_is_probabilistic() {
        assert!(TypeId::HyperLogLog.is_probabilistic());
        assert!(TypeId::BloomFilter.is_probabilistic());
        assert!(TypeId::TDigest.is_probabilistic());
        assert!(TypeId::CountMinSketch.is_probabilistic());

        assert!(!TypeId::Geometry.is_probabilistic());
        assert!(!TypeId::Int64.is_probabilistic());
    }

    #[test]
    fn test_is_network() {
        assert!(TypeId::Inet.is_network());
        assert!(TypeId::Cidr.is_network());
        assert!(TypeId::MacAddr.is_network());

        assert!(!TypeId::Varchar.is_network());
        assert!(!TypeId::Geometry.is_network());
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
        assert_eq!(TypeId::Vector as u8, 120);
        assert_eq!(TypeId::Geometry as u8, 130);
        assert_eq!(TypeId::Matrix as u8, 140);
        assert_eq!(TypeId::Color as u8, 150);
        assert_eq!(TypeId::SemVer as u8, 160);
        assert_eq!(TypeId::Inet as u8, 170);
        assert_eq!(TypeId::Cidr as u8, 171);
        assert_eq!(TypeId::MacAddr as u8, 172);
        assert_eq!(TypeId::Money as u8, 180);
        assert_eq!(TypeId::Range as u8, 190);
        assert_eq!(TypeId::HyperLogLog as u8, 210);
        assert_eq!(TypeId::BloomFilter as u8, 211);
        assert_eq!(TypeId::TDigest as u8, 212);
        assert_eq!(TypeId::CountMinSketch as u8, 213);
        assert_eq!(TypeId::Bitfield as u8, 220);
        assert_eq!(TypeId::Quantity as u8, 240);
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
            TypeId::Bytea,
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
            TypeId::Vector,
            TypeId::Geometry,
            TypeId::Matrix,
            TypeId::Color,
            TypeId::SemVer,
            TypeId::Inet,
            TypeId::Cidr,
            TypeId::MacAddr,
            TypeId::Money,
            TypeId::Range,
            TypeId::HyperLogLog,
            TypeId::BloomFilter,
            TypeId::TDigest,
            TypeId::CountMinSketch,
            TypeId::Bitfield,
            TypeId::Quantity,
        ];

        for type_id in all_types {
            let display = type_id.to_string();
            assert!(
                !display.is_empty(),
                "TypeId {:?} has empty display",
                type_id
            );
        }
    }
}
