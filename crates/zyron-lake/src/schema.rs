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
    pub fractional_digits: Option<u8>,
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
        TypeId::timestamp_physical_type_id(self.type_id, self.fractional_digits)
    }
}

/// A column whose values are computed from an expression rather than
/// supplied by the writer's caller as a table column.
///
/// This is what lets clustering target an expression. A cluster key names
/// a column id, pruning reads statistics by column id, and the prune index
/// and the overlap sweep both address columns, so an expression that is to
/// be clustered on and pruned by has to be a column. Giving it one means
/// every one of those paths works on it unchanged, and the writer computes
/// its bounds and bloom in the pass it already makes over the batch.
///
/// The hash is the identity. The planner canonicalizes an expression and
/// hashes the canonical form, so two spellings of the same expression
/// reach the same column and matching a query against it is a u64 compare
/// rather than an AST walk. The SQL is what the canonical form renders
/// back to, carried so the column can be shown and recomputed
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DerivedColumn {
    /// The schema column holding this expression's values
    pub column_id: u32,
    pub canonical_hash: u64,
    pub sql: String,
    /// Columns the expression reads, ascending. Recorded rather than
    /// re-derived from the SQL so dropping a column can refuse when an
    /// expression still depends on it, which is the case that would
    /// otherwise leave a derived column nothing can recompute
    pub source_columns: Vec<u32>,
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
    /// Columns computed from an expression, by the id of the column that
    /// holds their values. Empty on a table nobody clusters by expression
    pub derived: Vec<DerivedColumn>,
}

// Metadata blob presence flags
const META_TS_PRECISION: u8 = 1 << 0;
const META_TZ_OFFSET: u8 = 1 << 1;
const META_MAX_LENGTH: u8 = 1 << 2;
const META_DEFAULT_EXPR: u8 = 1 << 3;
const META_KNOWN_MASK: u8 =
    META_TS_PRECISION | META_TZ_OFFSET | META_MAX_LENGTH | META_DEFAULT_EXPR;

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
            derived: Vec::new(),
        };
        schema.validate()?;
        Ok(schema)
    }

    /// Builds a validated schema carrying expression columns.
    ///
    /// `next_column_id` still comes from the highest id present, and a
    /// derived entry names a column that has to be in `columns`, because
    /// its values are stored like any other column's
    pub fn with_derived(
        schema_id: u64,
        columns: Vec<LakeColumn>,
        derived: Vec<DerivedColumn>,
    ) -> Result<Self, ZyronError> {
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
            derived,
        };
        schema.validate()?;
        Ok(schema)
    }

    /// The expression a column computes, or None for a stored column
    pub fn derived_for(&self, column_id: u32) -> Option<&DerivedColumn> {
        self.derived.iter().find(|d| d.column_id == column_id)
    }

    /// The column holding an expression's values, matched by canonical
    /// hash. This is how a query's expression finds the column carrying
    /// the statistics that can prune it
    pub fn column_by_derived_hash(&self, canonical_hash: u64) -> Option<&LakeColumn> {
        self.derived
            .iter()
            .find(|d| d.canonical_hash == canonical_hash)
            .and_then(|d| self.column_by_id(d.column_id))
    }

    /// True when a column exists to carry an expression rather than
    /// because the table declared it. These are not part of the table's
    /// user-visible shape
    pub fn is_derived(&self, column_id: u32) -> bool {
        self.derived.iter().any(|d| d.column_id == column_id)
    }

    /// The first expression column that reads this column, if any.
    ///
    /// Dropping a column an expression reads would leave that expression
    /// with nothing to recompute from, so a drop consults this and refuses
    /// rather than leaving a column whose next insert cannot fill it
    pub fn derived_depending_on(&self, column_id: u32) -> Option<&DerivedColumn> {
        self.derived
            .iter()
            .find(|d| d.source_columns.contains(&column_id))
    }

    /// Columns the table declared, in order, without the expression
    /// columns clustering added
    pub fn user_columns(&self) -> impl Iterator<Item = &LakeColumn> {
        self.columns.iter().filter(|c| !self.is_derived(c.id))
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
            if let Some(p) = col.fractional_digits {
                // A decimal scale reaches 38, the widest an i128 holds. The
                // timestamp precisions stop at 12 (picoseconds)
                let (bound, what) = if col.type_id == TypeId::Decimal {
                    (38, "decimal scale")
                } else {
                    (12, "timestamp precision")
                };
                if p > bound {
                    return Err(ZyronError::Internal(format!(
                        "lake schema column \"{}\" {} {} exceeds {}",
                        col.name, what, p, bound
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
        for (i, derived) in self.derived.iter().enumerate() {
            if self.column_by_id(derived.column_id).is_none() {
                return Err(ZyronError::Internal(format!(
                    "lake schema derived expression names column id {}, which is not in the schema",
                    derived.column_id
                )));
            }
            if derived.sql.is_empty() {
                return Err(ZyronError::Internal(format!(
                    "lake schema derived column id {} has an empty expression",
                    derived.column_id
                )));
            }
            if derived.sql.len() > u16::MAX as usize {
                return Err(ZyronError::Internal(format!(
                    "lake schema derived column id {} expression exceeds {} bytes",
                    derived.column_id,
                    u16::MAX
                )));
            }
            for source in &derived.source_columns {
                if *source == derived.column_id {
                    return Err(ZyronError::Internal(format!(
                        "lake schema derived column id {} reads itself",
                        derived.column_id
                    )));
                }
                if self.column_by_id(*source).is_none() {
                    return Err(ZyronError::Internal(format!(
                        "lake schema derived column id {} reads column id {}, which is not in the schema",
                        derived.column_id, source
                    )));
                }
            }
            for other in &self.derived[i + 1..] {
                if other.column_id == derived.column_id {
                    return Err(ZyronError::Internal(format!(
                        "lake schema has two expressions for column id {}",
                        derived.column_id
                    )));
                }
                // Two columns computing the same expression would split its
                // statistics and let a query prune by whichever it matched
                if other.canonical_hash == derived.canonical_hash {
                    return Err(ZyronError::Internal(format!(
                        "lake schema has two columns for expression \"{}\"",
                        derived.sql
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
            if col.fractional_digits.is_some() {
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
            if let Some(p) = col.fractional_digits {
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
        // Expression columns follow the column records, so a schema that
        // has none costs four bytes
        buf.extend_from_slice(&(self.derived.len() as u32).to_le_bytes());
        for derived in &self.derived {
            buf.extend_from_slice(&derived.column_id.to_le_bytes());
            buf.extend_from_slice(&derived.canonical_hash.to_le_bytes());
            buf.extend_from_slice(&(derived.sql.len() as u16).to_le_bytes());
            buf.extend_from_slice(derived.sql.as_bytes());
            buf.extend_from_slice(&(derived.source_columns.len() as u16).to_le_bytes());
            for source in &derived.source_columns {
                buf.extend_from_slice(&source.to_le_bytes());
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
            return Err(r.corrupt(format!(
                "schema column count {} exceeds section size",
                count
            )));
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
            let fractional_digits = if flags & META_TS_PRECISION != 0 {
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
                fractional_digits,
                tz_offset_secs,
                max_length,
                default_expr,
            });
        }
        let derived_count = r.u32()? as usize;
        // A record is at least 14 bytes, so a corrupt count cannot drive a
        // large preallocation
        if derived_count > bytes.len() / 14 {
            return Err(r.corrupt(format!(
                "schema derived column count {} exceeds section size",
                derived_count
            )));
        }
        let mut derived = Vec::with_capacity(derived_count);
        for _ in 0..derived_count {
            let column_id = r.u32()?;
            let canonical_hash = r.u64()?;
            let sql_len = r.u16()? as usize;
            let sql = r.utf8(sql_len, "derived column expression")?;
            let source_count = r.u16()? as usize;
            let mut source_columns = Vec::with_capacity(source_count);
            for _ in 0..source_count {
                source_columns.push(r.u32()?);
            }
            derived.push(DerivedColumn {
                column_id,
                canonical_hash,
                sql,
                source_columns,
            });
        }
        let schema = Self {
            schema_id,
            next_column_id,
            columns,
            derived,
        };
        schema
            .validate()
            .map_err(|e| ZyronError::ManifestCorrupted {
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
            fractional_digits: None,
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
                    fractional_digits: None,
                    tz_offset_secs: None,
                    max_length: None,
                    default_expr: None,
                },
                LakeColumn {
                    id: 1,
                    name: "name".into(),
                    type_id: TypeId::Varchar,
                    nullable: true,
                    fractional_digits: None,
                    tz_offset_secs: None,
                    max_length: Some(255),
                    default_expr: Some("'anonymous'".into()),
                },
                LakeColumn {
                    id: 2,
                    name: "created".into(),
                    type_id: TypeId::TimestampTz,
                    nullable: false,
                    fractional_digits: Some(9),
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
        assert_eq!(
            schema.column_by_id(2).map(|c| c.name.as_str()),
            Some("created")
        );
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

    fn derived_schema() -> LakeSchema {
        LakeSchema::with_derived(
            3,
            vec![
                plain(0, "id", TypeId::Int64),
                plain(1, "ts", TypeId::Timestamp),
                plain(2, "ts_day", TypeId::Timestamp),
            ],
            vec![DerivedColumn {
                column_id: 2,
                canonical_hash: 0x9E37_79B9_7F4A_7C15,
                sql: "date_trunc('day', ts)".into(),
                source_columns: vec![1],
            }],
        )
        .expect("valid derived schema")
    }

    /// An expression column has to survive the codec intact, because the
    /// hash is how a query finds the column carrying statistics it can
    /// prune by, and a lost source list would let a drop break it
    #[test]
    fn test_roundtrip_preserves_derived_columns() {
        let schema = derived_schema();
        let mut buf = Vec::new();
        schema.encode_into(&mut buf);
        let (back, consumed) = LakeSchema::decode(&buf, "test").expect("decodes");
        assert_eq!(consumed, buf.len());
        assert_eq!(back, schema);
        assert_eq!(back.derived.len(), 1);
        assert_eq!(back.derived[0].source_columns, vec![1]);
    }

    /// A schema with no expression columns pays four bytes for the section
    /// and still round trips
    #[test]
    fn test_roundtrip_without_derived_columns() {
        let schema = sample();
        let mut buf = Vec::new();
        schema.encode_into(&mut buf);
        let (back, consumed) = LakeSchema::decode(&buf, "test").expect("decodes");
        assert_eq!(consumed, buf.len());
        assert!(back.derived.is_empty());
    }

    #[test]
    fn test_derived_lookups_answer_by_hash_and_by_column() {
        let schema = derived_schema();
        let found = schema
            .column_by_derived_hash(0x9E37_79B9_7F4A_7C15)
            .expect("hash finds its column");
        assert_eq!(found.id, 2);
        assert!(schema.column_by_derived_hash(1).is_none());
        assert!(schema.is_derived(2));
        assert!(!schema.is_derived(1));
        assert_eq!(
            schema.derived_for(2).map(|d| d.sql.as_str()),
            Some("date_trunc('day', ts)")
        );
        // The table declared two columns, clustering added the third
        let user: Vec<&str> = schema.user_columns().map(|c| c.name.as_str()).collect();
        assert_eq!(user, vec!["id", "ts"]);
    }

    /// Dropping a column an expression reads has to be refusable, so the
    /// dependency has to be answerable from the schema alone
    #[test]
    fn test_derived_dependency_is_reported_for_its_sources_only() {
        let schema = derived_schema();
        assert_eq!(
            schema.derived_depending_on(1).map(|d| d.column_id),
            Some(2),
            "ts is read by the expression"
        );
        assert!(schema.derived_depending_on(0).is_none());
        assert!(
            schema.derived_depending_on(2).is_none(),
            "the column holding the values is not one of its sources"
        );
    }

    #[test]
    fn test_validate_rejects_malformed_derived_columns() {
        let columns = || {
            vec![
                plain(0, "id", TypeId::Int64),
                plain(1, "ts", TypeId::Timestamp),
                plain(2, "ts_day", TypeId::Timestamp),
            ]
        };
        let derived = |d: DerivedColumn| LakeSchema::with_derived(1, columns(), vec![d]);

        // Names a column that is not in the schema
        assert!(
            derived(DerivedColumn {
                column_id: 9,
                canonical_hash: 1,
                sql: "x".into(),
                source_columns: vec![1],
            })
            .is_err()
        );
        // Reads a column that is not in the schema
        assert!(
            derived(DerivedColumn {
                column_id: 2,
                canonical_hash: 1,
                sql: "x".into(),
                source_columns: vec![9],
            })
            .is_err()
        );
        // Reads itself, which has no evaluation order
        assert!(
            derived(DerivedColumn {
                column_id: 2,
                canonical_hash: 1,
                sql: "x".into(),
                source_columns: vec![2],
            })
            .is_err()
        );
        // An empty expression names nothing
        assert!(
            derived(DerivedColumn {
                column_id: 2,
                canonical_hash: 1,
                sql: String::new(),
                source_columns: vec![1],
            })
            .is_err()
        );
        // Two columns holding the same expression would split its statistics
        assert!(
            LakeSchema::with_derived(
                1,
                columns(),
                vec![
                    DerivedColumn {
                        column_id: 1,
                        canonical_hash: 7,
                        sql: "x".into(),
                        source_columns: vec![0],
                    },
                    DerivedColumn {
                        column_id: 2,
                        canonical_hash: 7,
                        sql: "x".into(),
                        source_columns: vec![0],
                    },
                ],
            )
            .is_err()
        );
    }
}
