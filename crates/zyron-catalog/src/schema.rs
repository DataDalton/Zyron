//! Catalog entry structs for databases, schemas, tables, columns, indexes, and constraints.
//!
//! Each entry type has binary serialization via to_bytes/from_bytes for storage
//! in heap file system tables.

use crate::encoding::*;
use crate::ids::*;
use serde::{Deserialize, Serialize};
use zyron_common::{Result, TypeId};

// ---------------------------------------------------------------------------
// DatabaseEntry
// ---------------------------------------------------------------------------

/// Catalog entry for a database.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DatabaseEntry {
    pub id: DatabaseId,
    pub name: String,
    pub owner: String,
    pub created_at: u64,
}

impl DatabaseEntry {
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut buf = Vec::with_capacity(64);
        write_u32(&mut buf, self.id.0);
        write_string(&mut buf, &self.name);
        write_string(&mut buf, &self.owner);
        write_u64(&mut buf, self.created_at);
        buf
    }

    pub fn from_bytes(data: &[u8]) -> Result<Self> {
        let mut off = 0;
        let id = DatabaseId(read_u32(data, &mut off)?);
        let name = read_string(data, &mut off)?;
        let owner = read_string(data, &mut off)?;
        let created_at = read_u64(data, &mut off)?;
        Ok(Self {
            id,
            name,
            owner,
            created_at,
        })
    }
}

// ---------------------------------------------------------------------------
// SchemaEntry
// ---------------------------------------------------------------------------

/// Catalog entry for a schema within a database.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SchemaEntry {
    pub id: SchemaId,
    pub database_id: DatabaseId,
    pub name: String,
    pub owner: String,
}

impl SchemaEntry {
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut buf = Vec::with_capacity(64);
        write_u32(&mut buf, self.id.0);
        write_u32(&mut buf, self.database_id.0);
        write_string(&mut buf, &self.name);
        write_string(&mut buf, &self.owner);
        buf
    }

    pub fn from_bytes(data: &[u8]) -> Result<Self> {
        let mut off = 0;
        let id = SchemaId(read_u32(data, &mut off)?);
        let database_id = DatabaseId(read_u32(data, &mut off)?);
        let name = read_string(data, &mut off)?;
        let owner = read_string(data, &mut off)?;
        Ok(Self {
            id,
            database_id,
            name,
            owner,
        })
    }
}

// ---------------------------------------------------------------------------
// ColumnEntry
// ---------------------------------------------------------------------------

/// Catalog entry for a column within a table.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ColumnEntry {
    pub id: ColumnId,
    pub table_id: TableId,
    pub name: String,
    pub type_id: TypeId,
    pub ordinal: u16,
    pub nullable: bool,
    pub default_expr: Option<String>,
    pub max_length: Option<usize>,
}

impl ColumnEntry {
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut buf = Vec::with_capacity(64);
        write_u16(&mut buf, self.id.0);
        write_u32(&mut buf, self.table_id.0);
        write_string(&mut buf, &self.name);
        write_u8(&mut buf, self.type_id as u8);
        write_u16(&mut buf, self.ordinal);
        write_bool(&mut buf, self.nullable);
        write_option_string(&mut buf, &self.default_expr);
        write_option_usize(&mut buf, &self.max_length);
        buf
    }

    pub fn from_bytes(data: &[u8]) -> Result<Self> {
        let mut off = 0;
        let id = ColumnId(read_u16(data, &mut off)?);
        let table_id = TableId(read_u32(data, &mut off)?);
        let name = read_string(data, &mut off)?;
        let type_id = type_id_from_u8(read_u8(data, &mut off)?)?;
        let ordinal = read_u16(data, &mut off)?;
        let nullable = read_bool(data, &mut off)?;
        let default_expr = read_option_string(data, &mut off)?;
        let max_length = read_option_usize(data, &mut off)?;
        Ok(Self {
            id,
            table_id,
            name,
            type_id,
            ordinal,
            nullable,
            default_expr,
            max_length,
        })
    }
}

// ---------------------------------------------------------------------------
// ConstraintEntry
// ---------------------------------------------------------------------------

/// Type of table constraint.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[repr(u8)]
pub enum ConstraintType {
    PrimaryKey = 0,
    Unique = 1,
    ForeignKey = 2,
    Check = 3,
    NotNull = 4,
}

/// Catalog entry for a table constraint.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConstraintEntry {
    pub name: String,
    pub constraint_type: ConstraintType,
    pub columns: Vec<ColumnId>,
    pub ref_table_id: Option<TableId>,
    pub ref_columns: Vec<ColumnId>,
    pub check_expr: Option<String>,
}

impl ConstraintEntry {
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut buf = Vec::with_capacity(64);
        write_string(&mut buf, &self.name);
        write_u8(&mut buf, self.constraint_type as u8);
        write_u16(&mut buf, self.columns.len() as u16);
        for col in &self.columns {
            write_u16(&mut buf, col.0);
        }
        match self.ref_table_id {
            None => write_u8(&mut buf, 0),
            Some(t) => {
                write_u8(&mut buf, 1);
                write_u32(&mut buf, t.0);
            }
        }
        write_u16(&mut buf, self.ref_columns.len() as u16);
        for col in &self.ref_columns {
            write_u16(&mut buf, col.0);
        }
        write_option_string(&mut buf, &self.check_expr);
        buf
    }

    pub fn from_bytes(data: &[u8], offset: &mut usize) -> Result<Self> {
        let name = read_string(data, offset)?;
        let ct = read_u8(data, offset)?;
        let constraint_type = constraint_type_from_u8(ct)?;
        let col_count = read_u16(data, offset)? as usize;
        let mut columns = Vec::with_capacity(col_count);
        for _ in 0..col_count {
            columns.push(ColumnId(read_u16(data, offset)?));
        }
        let has_ref = read_u8(data, offset)?;
        let ref_table_id = if has_ref != 0 {
            Some(TableId(read_u32(data, offset)?))
        } else {
            None
        };
        let ref_col_count = read_u16(data, offset)? as usize;
        let mut ref_columns = Vec::with_capacity(ref_col_count);
        for _ in 0..ref_col_count {
            ref_columns.push(ColumnId(read_u16(data, offset)?));
        }
        let check_expr = read_option_string(data, offset)?;
        Ok(Self {
            name,
            constraint_type,
            columns,
            ref_table_id,
            ref_columns,
            check_expr,
        })
    }
}

// ---------------------------------------------------------------------------
// TableEntry
// ---------------------------------------------------------------------------

/// Catalog entry for a table.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TableEntry {
    pub id: TableId,
    pub schema_id: SchemaId,
    pub name: String,
    pub heap_file_id: u32,
    pub fsm_file_id: u32,
    pub columns: Vec<ColumnEntry>,
    pub constraints: Vec<ConstraintEntry>,
    pub created_at: u64,
    /// Whether table-level versioning is enabled for time travel queries.
    pub versioning_enabled: bool,
    /// SCD type configured on this table (None, 1-4, or 6).
    pub scd_type: Option<u8>,
    /// Whether this table is system-versioned (automatic sys_start/sys_end).
    pub system_versioned: bool,
    /// For SCD Type 4: the companion history table's table_id.
    pub history_table_id: Option<u32>,
}

impl TableEntry {
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut buf = Vec::with_capacity(256);
        write_u32(&mut buf, self.id.0);
        write_u32(&mut buf, self.schema_id.0);
        write_string(&mut buf, &self.name);
        write_u32(&mut buf, self.heap_file_id);
        write_u32(&mut buf, self.fsm_file_id);
        write_u64(&mut buf, self.created_at);

        // Columns
        write_u16(&mut buf, self.columns.len() as u16);
        for col in &self.columns {
            let col_bytes = col.to_bytes();
            write_u32(&mut buf, col_bytes.len() as u32);
            buf.extend_from_slice(&col_bytes);
        }

        // Constraints
        write_u16(&mut buf, self.constraints.len() as u16);
        for con in &self.constraints {
            let con_bytes = con.to_bytes();
            write_u32(&mut buf, con_bytes.len() as u32);
            buf.extend_from_slice(&con_bytes);
        }

        // Versioning fields (appended for backward compatibility)
        buf.push(if self.versioning_enabled { 1 } else { 0 });
        buf.push(self.scd_type.unwrap_or(0));
        buf.push(if self.system_versioned { 1 } else { 0 });
        write_u32(&mut buf, self.history_table_id.unwrap_or(0));

        buf
    }

    pub fn from_bytes(data: &[u8]) -> Result<Self> {
        let mut off = 0;
        let id = TableId(read_u32(data, &mut off)?);
        let schema_id = SchemaId(read_u32(data, &mut off)?);
        let name = read_string(data, &mut off)?;
        let heap_file_id = read_u32(data, &mut off)?;
        let fsm_file_id = read_u32(data, &mut off)?;
        let created_at = read_u64(data, &mut off)?;

        // Columns
        let col_count = read_u16(data, &mut off)? as usize;
        let mut columns = Vec::with_capacity(col_count);
        for _ in 0..col_count {
            let col_len = read_u32(data, &mut off)? as usize;
            let col_data = &data[off..off + col_len];
            columns.push(ColumnEntry::from_bytes(col_data)?);
            off += col_len;
        }

        // Constraints
        let con_count = read_u16(data, &mut off)? as usize;
        let mut constraints = Vec::with_capacity(con_count);
        for _ in 0..con_count {
            let con_len = read_u32(data, &mut off)? as usize;
            let con_start = off;
            let mut con_off = con_start;
            constraints.push(ConstraintEntry::from_bytes(data, &mut con_off)?);
            off += con_len;
        }

        // Versioning fields (backward compatible: default to false/None if missing)
        let versioning_enabled = if off < data.len() {
            let v = data[off];
            off += 1;
            v != 0
        } else {
            false
        };
        let scd_type = if off < data.len() {
            let v = data[off];
            off += 1;
            if v == 0 { None } else { Some(v) }
        } else {
            None
        };
        let system_versioned = if off < data.len() {
            let v = data[off];
            off += 1;
            v != 0
        } else {
            false
        };
        let history_table_id = if off + 4 <= data.len() {
            let v = read_u32(data, &mut off)?;
            if v == 0 { None } else { Some(v) }
        } else {
            None
        };

        Ok(Self {
            id,
            schema_id,
            name,
            heap_file_id,
            fsm_file_id,
            columns,
            constraints,
            created_at,
            versioning_enabled,
            scd_type,
            system_versioned,
            history_table_id,
        })
    }
}

// ---------------------------------------------------------------------------
// IndexEntry
// ---------------------------------------------------------------------------

/// Type of index.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[repr(u8)]
pub enum IndexType {
    BTree = 0,
    Fulltext = 1,
    Vector = 2,
}

/// A column participating in an index.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IndexColumnEntry {
    pub column_id: ColumnId,
    pub ordinal: u16,
    pub descending: bool,
}

/// Catalog entry for an index.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IndexEntry {
    pub id: IndexId,
    pub table_id: TableId,
    pub schema_id: SchemaId,
    pub name: String,
    pub columns: Vec<IndexColumnEntry>,
    pub unique: bool,
    pub index_file_id: u32,
    pub index_type: IndexType,
}

impl IndexEntry {
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut buf = Vec::with_capacity(128);
        write_u32(&mut buf, self.id.0);
        write_u32(&mut buf, self.table_id.0);
        write_u32(&mut buf, self.schema_id.0);
        write_string(&mut buf, &self.name);
        write_u16(&mut buf, self.columns.len() as u16);
        for col in &self.columns {
            write_u16(&mut buf, col.column_id.0);
            write_u16(&mut buf, col.ordinal);
            write_bool(&mut buf, col.descending);
        }
        write_bool(&mut buf, self.unique);
        write_u32(&mut buf, self.index_file_id);
        write_u8(&mut buf, self.index_type as u8);
        buf
    }

    pub fn from_bytes(data: &[u8]) -> Result<Self> {
        let mut off = 0;
        let id = IndexId(read_u32(data, &mut off)?);
        let table_id = TableId(read_u32(data, &mut off)?);
        let schema_id = SchemaId(read_u32(data, &mut off)?);
        let name = read_string(data, &mut off)?;
        let col_count = read_u16(data, &mut off)? as usize;
        let mut columns = Vec::with_capacity(col_count);
        for _ in 0..col_count {
            let column_id = ColumnId(read_u16(data, &mut off)?);
            let ordinal = read_u16(data, &mut off)?;
            let descending = read_bool(data, &mut off)?;
            columns.push(IndexColumnEntry {
                column_id,
                ordinal,
                descending,
            });
        }
        let unique = read_bool(data, &mut off)?;
        let index_file_id = read_u32(data, &mut off)?;
        let it = read_u8(data, &mut off)?;
        let index_type = index_type_from_u8(it)?;
        Ok(Self {
            id,
            table_id,
            schema_id,
            name,
            columns,
            unique,
            index_file_id,
            index_type,
        })
    }
}

// ---------------------------------------------------------------------------
// Enum conversion helpers
// ---------------------------------------------------------------------------

fn type_id_from_u8(val: u8) -> Result<TypeId> {
    match val {
        0 => Ok(TypeId::Null),
        1 => Ok(TypeId::Boolean),
        10 => Ok(TypeId::Int8),
        11 => Ok(TypeId::Int16),
        12 => Ok(TypeId::Int32),
        13 => Ok(TypeId::Int64),
        14 => Ok(TypeId::Int128),
        20 => Ok(TypeId::UInt8),
        21 => Ok(TypeId::UInt16),
        22 => Ok(TypeId::UInt32),
        23 => Ok(TypeId::UInt64),
        24 => Ok(TypeId::UInt128),
        30 => Ok(TypeId::Float32),
        31 => Ok(TypeId::Float64),
        40 => Ok(TypeId::Decimal),
        50 => Ok(TypeId::Char),
        51 => Ok(TypeId::Varchar),
        52 => Ok(TypeId::Text),
        60 => Ok(TypeId::Binary),
        61 => Ok(TypeId::Varbinary),
        62 => Ok(TypeId::Bytea),
        70 => Ok(TypeId::Date),
        71 => Ok(TypeId::Time),
        72 => Ok(TypeId::Timestamp),
        73 => Ok(TypeId::TimestampTz),
        74 => Ok(TypeId::Interval),
        80 => Ok(TypeId::Uuid),
        90 => Ok(TypeId::Json),
        91 => Ok(TypeId::Jsonb),
        100 => Ok(TypeId::Array),
        110 => Ok(TypeId::Composite),
        120 => Ok(TypeId::Vector),
        _ => Err(zyron_common::ZyronError::CatalogCorrupted(format!(
            "unknown TypeId value: {val}"
        ))),
    }
}

fn constraint_type_from_u8(val: u8) -> Result<ConstraintType> {
    match val {
        0 => Ok(ConstraintType::PrimaryKey),
        1 => Ok(ConstraintType::Unique),
        2 => Ok(ConstraintType::ForeignKey),
        3 => Ok(ConstraintType::Check),
        4 => Ok(ConstraintType::NotNull),
        _ => Err(zyron_common::ZyronError::CatalogCorrupted(format!(
            "unknown ConstraintType value: {val}"
        ))),
    }
}

fn index_type_from_u8(val: u8) -> Result<IndexType> {
    match val {
        0 => Ok(IndexType::BTree),
        1 => Ok(IndexType::Fulltext),
        2 => Ok(IndexType::Vector),
        _ => Err(zyron_common::ZyronError::CatalogCorrupted(format!(
            "unknown IndexType value: {val}"
        ))),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_database_entry_roundtrip() {
        let entry = DatabaseEntry {
            id: DatabaseId(1),
            name: "testdb".to_string(),
            owner: "admin".to_string(),
            created_at: 1700000000,
        };
        let bytes = entry.to_bytes();
        let decoded = DatabaseEntry::from_bytes(&bytes).unwrap();
        assert_eq!(decoded.id, entry.id);
        assert_eq!(decoded.name, entry.name);
        assert_eq!(decoded.owner, entry.owner);
        assert_eq!(decoded.created_at, entry.created_at);
    }

    #[test]
    fn test_schema_entry_roundtrip() {
        let entry = SchemaEntry {
            id: SchemaId(5),
            database_id: DatabaseId(1),
            name: "public".to_string(),
            owner: "admin".to_string(),
        };
        let bytes = entry.to_bytes();
        let decoded = SchemaEntry::from_bytes(&bytes).unwrap();
        assert_eq!(decoded.id, entry.id);
        assert_eq!(decoded.database_id, entry.database_id);
        assert_eq!(decoded.name, entry.name);
        assert_eq!(decoded.owner, entry.owner);
    }

    #[test]
    fn test_column_entry_roundtrip() {
        let entry = ColumnEntry {
            id: ColumnId(0),
            table_id: TableId(10),
            name: "user_id".to_string(),
            type_id: TypeId::Int64,
            ordinal: 0,
            nullable: false,
            default_expr: None,
            max_length: None,
        };
        let bytes = entry.to_bytes();
        let decoded = ColumnEntry::from_bytes(&bytes).unwrap();
        assert_eq!(decoded.id, entry.id);
        assert_eq!(decoded.name, entry.name);
        assert_eq!(decoded.type_id, TypeId::Int64);
        assert_eq!(decoded.nullable, false);
        assert_eq!(decoded.default_expr, None);
    }

    #[test]
    fn test_column_entry_with_optionals() {
        let entry = ColumnEntry {
            id: ColumnId(1),
            table_id: TableId(10),
            name: "email".to_string(),
            type_id: TypeId::Varchar,
            ordinal: 1,
            nullable: true,
            default_expr: Some("'unknown'".to_string()),
            max_length: Some(255),
        };
        let bytes = entry.to_bytes();
        let decoded = ColumnEntry::from_bytes(&bytes).unwrap();
        assert_eq!(decoded.default_expr, Some("'unknown'".to_string()));
        assert_eq!(decoded.max_length, Some(255));
        assert_eq!(decoded.nullable, true);
    }

    #[test]
    fn test_table_entry_roundtrip() {
        let entry = TableEntry {
            id: TableId(100),
            schema_id: SchemaId(1),
            name: "users".to_string(),
            heap_file_id: 200,
            fsm_file_id: 201,
            columns: vec![
                ColumnEntry {
                    id: ColumnId(0),
                    table_id: TableId(100),
                    name: "id".to_string(),
                    type_id: TypeId::Int64,
                    ordinal: 0,
                    nullable: false,
                    default_expr: None,
                    max_length: None,
                },
                ColumnEntry {
                    id: ColumnId(1),
                    table_id: TableId(100),
                    name: "name".to_string(),
                    type_id: TypeId::Varchar,
                    ordinal: 1,
                    nullable: true,
                    default_expr: None,
                    max_length: Some(100),
                },
            ],
            constraints: vec![ConstraintEntry {
                name: "pk_users".to_string(),
                constraint_type: ConstraintType::PrimaryKey,
                columns: vec![ColumnId(0)],
                ref_table_id: None,
                ref_columns: vec![],
                check_expr: None,
            }],
            created_at: 1700000000,
            versioning_enabled: false,
            scd_type: None,
            system_versioned: false,
            history_table_id: None,
        };
        let bytes = entry.to_bytes();
        let decoded = TableEntry::from_bytes(&bytes).unwrap();
        assert_eq!(decoded.id, entry.id);
        assert_eq!(decoded.name, "users");
        assert_eq!(decoded.columns.len(), 2);
        assert_eq!(decoded.columns[0].name, "id");
        assert_eq!(decoded.columns[1].name, "name");
        assert_eq!(decoded.columns[1].max_length, Some(100));
        assert_eq!(decoded.constraints.len(), 1);
        assert_eq!(
            decoded.constraints[0].constraint_type,
            ConstraintType::PrimaryKey
        );
    }

    #[test]
    fn test_constraint_entry_foreign_key() {
        let entry = ConstraintEntry {
            name: "fk_orders_user".to_string(),
            constraint_type: ConstraintType::ForeignKey,
            columns: vec![ColumnId(2)],
            ref_table_id: Some(TableId(100)),
            ref_columns: vec![ColumnId(0)],
            check_expr: None,
        };
        let bytes = entry.to_bytes();
        let mut off = 0;
        let decoded = ConstraintEntry::from_bytes(&bytes, &mut off).unwrap();
        assert_eq!(decoded.constraint_type, ConstraintType::ForeignKey);
        assert_eq!(decoded.ref_table_id, Some(TableId(100)));
        assert_eq!(decoded.ref_columns.len(), 1);
    }

    #[test]
    fn test_index_entry_roundtrip() {
        let entry = IndexEntry {
            id: IndexId(1),
            table_id: TableId(100),
            schema_id: SchemaId(1),
            name: "idx_users_email".to_string(),
            columns: vec![IndexColumnEntry {
                column_id: ColumnId(1),
                ordinal: 0,
                descending: false,
            }],
            unique: true,
            index_file_id: 10000,
            index_type: IndexType::BTree,
        };
        let bytes = entry.to_bytes();
        let decoded = IndexEntry::from_bytes(&bytes).unwrap();
        assert_eq!(decoded.id, entry.id);
        assert_eq!(decoded.name, "idx_users_email");
        assert_eq!(decoded.unique, true);
        assert_eq!(decoded.index_type, IndexType::BTree);
        assert_eq!(decoded.columns.len(), 1);
        assert_eq!(decoded.columns[0].column_id, ColumnId(1));
    }

    #[test]
    fn test_all_type_ids_roundtrip() {
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
        ];
        for tid in all_types {
            let val = tid as u8;
            let decoded = type_id_from_u8(val).unwrap();
            assert_eq!(decoded, tid, "roundtrip failed for {tid:?}");
        }
    }

    #[test]
    fn test_all_constraint_types_roundtrip() {
        let all = [
            ConstraintType::PrimaryKey,
            ConstraintType::Unique,
            ConstraintType::ForeignKey,
            ConstraintType::Check,
            ConstraintType::NotNull,
        ];
        for ct in all {
            let decoded = constraint_type_from_u8(ct as u8).unwrap();
            assert_eq!(decoded, ct);
        }
    }

    #[test]
    fn test_all_index_types_roundtrip() {
        let all = [IndexType::BTree, IndexType::Fulltext, IndexType::Vector];
        for it in all {
            let decoded = index_type_from_u8(it as u8).unwrap();
            assert_eq!(decoded, it);
        }
    }

    #[test]
    fn test_empty_table_roundtrip() {
        let entry = TableEntry {
            id: TableId(1),
            schema_id: SchemaId(1),
            name: "empty".to_string(),
            heap_file_id: 200,
            fsm_file_id: 201,
            columns: vec![],
            constraints: vec![],
            created_at: 0,
            versioning_enabled: false,
            scd_type: None,
            system_versioned: false,
            history_table_id: None,
        };
        let bytes = entry.to_bytes();
        let decoded = TableEntry::from_bytes(&bytes).unwrap();
        assert_eq!(decoded.columns.len(), 0);
        assert_eq!(decoded.constraints.len(), 0);
    }
}
