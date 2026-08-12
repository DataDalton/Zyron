//! Lake table schema with stable column ids.
//!
//! Column identity is the id, never the name or position. Rename rewrites
//! the name string, reorder permutes the vec, drop removes the entry, and
//! none of them touch a data file because every .zyr column is addressed
//! by column id. Ids are never reused, `next_column_id` only grows.
//!
//! The binary section layout follows the manifest specification frame,
//! one u32 column count then per-column records. The per-column metadata
//! blob is binary rather than JSON, matching the binary log deviation

use zyron_common::{TypeId, ZyronError};

use crate::codec::Cursor;

/// One column of a lake table schema
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LakeColumn {
    /// Stable identity, allocated once and never reused
    pub id: u32,
    pub name: String,
    pub type_id: TypeId,
    pub nullable: bool,
    /// TIMESTAMP(p) fractional-second precision 0..=12. None means the
    /// default 6. p>6 stores i128 picoseconds
    pub ts_precision: Option<u8>,
    /// Original timezone offset in seconds for a single-zone TIMESTAMPTZ
    /// column, reattached on display. None means unknown, display UTC
    pub tz_offset_secs: Option<i32>,
    /// Declared VARCHAR(n)/CHAR(n) limit
    pub max_length: Option<u32>,
    /// Default expression SQL text, applied on ADD COLUMN and on inserts
    /// that omit the column
    pub default_expr: Option<String>,
}

impl LakeColumn {
    /// Physical storage TypeId. TIMESTAMP(p)/TIMESTAMPTZ(p) with p>6
    /// stores i128 picoseconds, everything else is its logical type
    pub fn physical_type_id(&self) -> TypeId {
        TypeId::timestamp_physical_type_id(self.type_id, self.ts_precision)
    }
}

/// A whole-table schema at one schema version
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LakeSchema {
    /// Monotone schema version, bumped by every schema change commit
    pub schema_id: u64,
    /// Next column id to allocate, strictly greater than every id ever used
    pub next_column_id: u32,
    /// Columns in user-visible order
    pub columns: Vec<LakeColumn>,
}

// Metadata blob presence flags
const META_TS_PRECISION: u8 = 1 << 0;
const META_TZ_OFFSET: u8 = 1 << 1;
const META_MAX_LENGTH: u8 = 1 << 2;
const META_DEFAULT_EXPR: u8 = 1 << 3;
const META_KNOWN_MASK: u8 = META_TS_PRECISION | META_TZ_OFFSET | META_MAX_LENGTH | META_DEFAULT_EXPR;

impl LakeSchema {
    /// Builds a validated schema. `next_column_id` is derived as one past
    /// the highest column id
    pub fn new(schema_id: u64, columns: Vec<LakeColumn>) -> Result<Self, ZyronError> {
        let next_column_id = columns
            .iter()
            .map(|c| c.id)
            .max()
            .map(|m| m + 1)
            .unwrap_or(0);
        let schema = Self {
            schema_id,
            next_column_id,
            columns,
        };
        schema.validate()?;
        Ok(schema)
    }

    /// Checks every invariant the codec and the writer rely on
    pub fn validate(&self) -> Result<(), ZyronError> {
        if self.columns.is_empty() {
            return Err(ZyronError::Internal(
                "lake schema must have at least one column".into(),
            ));
        }
        for (i, col) in self.columns.iter().enumerate() {
            if col.name.is_empty() {
                return Err(ZyronError::Internal(format!(
                    "lake schema column id {} has an empty name",
                    col.id
                )));
            }
            if col.name.len() > u16::MAX as usize {
                return Err(ZyronError::Internal(format!(
                    "lake schema column \"{}\" name exceeds {} bytes",
                    &col.name[..64.min(col.name.len())],
                    u16::MAX
                )));
            }
            if let Some(expr) = &col.default_expr {
                if expr.len() > u16::MAX as usize {
                    return Err(ZyronError::Internal(format!(
                        "lake schema column \"{}\" default expression exceeds {} bytes",
                        col.name,
                        u16::MAX
                    )));
                }
            }
            if let Some(p) = col.ts_precision {
                if p > 12 {
                    return Err(ZyronError::Internal(format!(
                        "lake schema column \"{}\" timestamp precision {} exceeds 12",
                        col.name, p
                    )));
                }
            }
            if col.id >= self.next_column_id {
                return Err(ZyronError::Internal(format!(
                    "lake schema column \"{}\" id {} is not below next_column_id {}",
                    col.name, col.id, self.next_column_id
                )));
            }
            for other in &self.columns[i + 1..] {
                if other.id == col.id {
                    return Err(ZyronError::Internal(format!(
                        "lake schema duplicate column id {}",
                        col.id
                    )));
                }
                if other.name == col.name {
                    return Err(ZyronError::Internal(format!(
                        "lake schema duplicate column name \"{}\"",
                        col.name
                    )));
                }
            }
        }
        Ok(())
    }

    pub fn column_by_id(&self, id: u32) -> Option<&LakeColumn> {
        self.columns.iter().find(|c| c.id == id)
    }

    pub fn column_by_name(&self, name: &str) -> Option<&LakeColumn> {
        self.columns.iter().find(|c| c.name == name)
    }

    /// Position of a column id in user-visible order
    pub fn ordinal_of_id(&self, id: u32) -> Option<usize> {
        self.columns.iter().position(|c| c.id == id)
    }

    /// Serializes the schema section. Layout: schema_id u64, next_column_id
    /// u32, column count u32, then per column: id u32, name u16 len + UTF-8,
    /// type id u16, nullable u8, metadata u32 len + flagged binary fields.
    /// All integers little endian. Infallible because `validate` bounds
    /// every length at construction
    pub fn encode_into(&self, buf: &mut Vec<u8>) {
        buf.extend_from_slice(&self.schema_id.to_le_bytes());
        buf.extend_from_slice(&self.next_column_id.to_le_bytes());
        buf.extend_from_slice(&(self.columns.len() as u32).to_le_bytes());
        for col in &self.columns {
            buf.extend_from_slice(&col.id.to_le_bytes());
            buf.extend_from_slice(&(col.name.len() as u16).to_le_bytes());
            buf.extend_from_slice(col.name.as_bytes());
            buf.extend_from_slice(&(col.type_id as u16).to_le_bytes());
            buf.push(col.nullable as u8);

            let mut flags = 0u8;
            let mut meta_len = 1usize;
            if col.ts_precision.is_some() {
                flags |= META_TS_PRECISION;
                meta_len += 1;
            }
            if col.tz_offset_secs.is_some() {
                flags |= META_TZ_OFFSET;
                meta_len += 4;
            }
            if col.max_length.is_some() {
                flags |= META_MAX_LENGTH;
                meta_len += 4;
            }
            if let Some(expr) = &col.default_expr {
                flags |= META_DEFAULT_EXPR;
                meta_len += 2 + expr.len();
            }
            buf.extend_from_slice(&(meta_len as u32).to_le_bytes());
            buf.push(flags);
            if let Some(p) = col.ts_precision {
                buf.push(p);
            }
            if let Some(secs) = col.tz_offset_secs {
                buf.extend_from_slice(&secs.to_le_bytes());
            }
            if let Some(n) = col.max_length {
                buf.extend_from_slice(&n.to_le_bytes());
            }
            if let Some(expr) = &col.default_expr {
                buf.extend_from_slice(&(expr.len() as u16).to_le_bytes());
                buf.extend_from_slice(expr.as_bytes());
            }
        }
    }

    /// Parses one schema section from the front of `bytes`, returning the
    /// schema and the number of bytes consumed. `ctx` names the enclosing
    /// file for error messages
    pub fn decode(bytes: &[u8], ctx: &str) -> Result<(Self, usize), ZyronError> {
        let mut r = Cursor::new(bytes, ctx);
        let schema_id = r.u64()?;
        let next_column_id = r.u32()?;
        let count = r.u32()? as usize;
        // Each column record is at least 13 bytes, bounds the count against
        // the buffer so a corrupt count cannot drive a huge preallocation
        if count > bytes.len() / 13 {
            return Err(r.corrupt(format!("schema column count {} exceeds section size", count)));
        }
        let mut columns = Vec::with_capacity(count);
        for _ in 0..count {
            let id = r.u32()?;
            let name_len = r.u16()? as usize;
            let name = r.utf8(name_len, "column name")?;
            let type_raw = r.u16()?;
            let type_id = u8::try_from(type_raw)
                .ok()
                .and_then(TypeId::from_u8)
                .ok_or_else(|| {
                    r.corrupt(format!(
                        "column \"{}\" has unknown type id {}",
                        name, type_raw
                    ))
                })?;
            let nullable = match r.u8()? {
                0 => false,
                1 => true,
                v => {
                    return Err(r.corrupt(format!(
                        "column \"{}\" has invalid nullable byte {}",
                        name, v
                    )));
                }
            };
            let meta_len = r.u32()? as usize;
            let meta_end = r.pos().checked_add(meta_len).ok_or_else(|| {
                r.corrupt(format!("column \"{}\" metadata length overflows", name))
            })?;
            if meta_end > bytes.len() {
                return Err(r.corrupt(format!(
                    "column \"{}\" metadata length {} exceeds section",
                    name, meta_len
                )));
            }
            let flags = r.u8()?;
            if flags & !META_KNOWN_MASK != 0 {
                return Err(r.corrupt(format!(
                    "column \"{}\" has unknown metadata flags {:#04x}",
                    name, flags
                )));
            }
            let ts_precision = if flags & META_TS_PRECISION != 0 {
                Some(r.u8()?)
            } else {
                None
            };
            let tz_offset_secs = if flags & META_TZ_OFFSET != 0 {
                Some(r.i32()?)
            } else {
                None
            };
            let max_length = if flags & META_MAX_LENGTH != 0 {
                Some(r.u32()?)
            } else {
                None
            };
            let default_expr = if flags & META_DEFAULT_EXPR != 0 {
                let len = r.u16()? as usize;
                Some(r.utf8(len, "default expression")?)
            } else {
                None
            };
            if r.pos() != meta_end {
                return Err(r.corrupt(format!(
                    "column \"{}\" metadata length {} does not match its fields",
                    name, meta_len
                )));
            }
            columns.push(LakeColumn {
                id,
                name,
                type_id,
                nullable,
                ts_precision,
                tz_offset_secs,
                max_length,
                default_expr,
            });
        }
        let schema = Self {
            schema_id,
            next_column_id,
            columns,
        };
        schema.validate().map_err(|e| ZyronError::ManifestCorrupted {
            path: ctx.to_string(),
            reason: e.to_string(),
        })?;
        Ok((schema, r.pos()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn plain(id: u32, name: &str, type_id: TypeId) -> LakeColumn {
        LakeColumn {
            id,
            name: name.to_string(),
            type_id,
            nullable: true,
            ts_precision: None,
            tz_offset_secs: None,
            max_length: None,
            default_expr: None,
        }
    }

    fn sample() -> LakeSchema {
        LakeSchema::new(
            7,
            vec![
                LakeColumn {
                    id: 0,
                    name: "id".into(),
                    type_id: TypeId::Int64,
                    nullable: false,
                    ts_precision: None,
                    tz_offset_secs: None,
                    max_length: None,
                    default_expr: None,
                },
                LakeColumn {
                    id: 1,
                    name: "name".into(),
                    type_id: TypeId::Varchar,
                    nullable: true,
                    ts_precision: None,
                    tz_offset_secs: None,
                    max_length: Some(255),
                    default_expr: Some("'anonymous'".into()),
                },
                LakeColumn {
                    id: 2,
                    name: "created".into(),
                    type_id: TypeId::TimestampTz,
                    nullable: false,
                    ts_precision: Some(9),
                    tz_offset_secs: Some(-18000),
                    max_length: None,
                    default_expr: None,
                },
            ],
        )
        .expect("valid schema")
    }

    #[test]
    fn test_roundtrip_preserves_every_field_and_order() {
        let schema = sample();
        let mut buf = Vec::new();
        schema.encode_into(&mut buf);
        let (decoded, consumed) = LakeSchema::decode(&buf, "test").expect("decodes");
        assert_eq!(decoded, schema);
        assert_eq!(consumed, buf.len());
    }

    #[test]
    fn test_decode_consumes_only_its_section() {
        let schema = sample();
        let mut buf = Vec::new();
        schema.encode_into(&mut buf);
        let section_len = buf.len();
        buf.extend_from_slice(&[0xAB; 32]);
        let (decoded, consumed) = LakeSchema::decode(&buf, "test").expect("decodes");
        assert_eq!(decoded, schema);
        assert_eq!(consumed, section_len);
    }

    #[test]
    fn test_new_derives_next_column_id_and_rejects_duplicates() {
        let s = LakeSchema::new(
            1,
            vec![plain(3, "a", TypeId::Int32), plain(9, "b", TypeId::Text)],
        )
        .expect("valid");
        assert_eq!(s.next_column_id, 10);

        let dup_id = LakeSchema::new(
            1,
            vec![plain(3, "a", TypeId::Int32), plain(3, "b", TypeId::Text)],
        );
        assert!(dup_id.is_err());

        let dup_name = LakeSchema::new(
            1,
            vec![plain(3, "a", TypeId::Int32), plain(4, "a", TypeId::Text)],
        );
        assert!(dup_name.is_err());

        assert!(LakeSchema::new(1, vec![]).is_err());
    }

    #[test]
    fn test_lookups_by_id_name_and_ordinal() {
        let schema = sample();
        assert_eq!(schema.column_by_id(2).map(|c| c.name.as_str()), Some("created"));
        assert_eq!(schema.column_by_name("name").map(|c| c.id), Some(1));
        assert_eq!(schema.ordinal_of_id(1), Some(1));
        assert_eq!(schema.column_by_id(99), None);
        assert_eq!(schema.ordinal_of_id(99), None);
    }

    #[test]
    fn test_physical_type_routes_high_precision_timestamps_to_int128() {
        let schema = sample();
        let created = schema.column_by_name("created").expect("exists");
        assert_eq!(created.physical_type_id(), TypeId::Int128);
        let id = schema.column_by_name("id").expect("exists");
        assert_eq!(id.physical_type_id(), TypeId::Int64);
    }

    #[test]
    fn test_decode_rejects_corruption() {
        let schema = sample();
        let mut buf = Vec::new();
        schema.encode_into(&mut buf);

        for cut in [0, 4, 8, 12, 20, buf.len() - 1] {
            assert!(
                LakeSchema::decode(&buf[..cut], "test").is_err(),
                "truncation at {} must fail",
                cut
            );
        }

        // Unknown type id in the first column record. The record starts
        // after schema_id u64 + next_column_id u32 + count u32, the type
        // sits after id u32 + name len u16 + 2 name bytes
        let type_off = 8 + 4 + 4 + 4 + 2 + 2;
        let mut bad_type = buf.clone();
        bad_type[type_off] = 0xFF;
        bad_type[type_off + 1] = 0xFF;
        assert!(LakeSchema::decode(&bad_type, "test").is_err());

        // Nullable byte outside 0 or 1
        let mut bad_null = buf.clone();
        bad_null[type_off + 2] = 7;
        assert!(LakeSchema::decode(&bad_null, "test").is_err());

        // Unknown metadata flag bit
        let mut bad_flags = buf.clone();
        bad_flags[type_off + 2 + 1 + 4] |= 0x80;
        assert!(LakeSchema::decode(&bad_flags, "test").is_err());
    }

    #[test]
    fn test_decode_rejects_oversized_column_count() {
        let mut buf = Vec::new();
        buf.extend_from_slice(&1u64.to_le_bytes());
        buf.extend_from_slice(&1u32.to_le_bytes());
        buf.extend_from_slice(&u32::MAX.to_le_bytes());
        assert!(LakeSchema::decode(&buf, "test").is_err());
    }
}
