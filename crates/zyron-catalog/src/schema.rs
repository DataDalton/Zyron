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
    /// Whether change data feed (CDC) is enabled for this table.
    pub cdf_enabled: bool,
    /// CDF retention in days (0 = unlimited).
    pub cdf_retention_days: u32,
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
        buf.push(if self.cdf_enabled { 1 } else { 0 });
        write_u32(&mut buf, self.cdf_retention_days);

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
        let cdf_enabled = if off < data.len() {
            let v = data[off];
            off += 1;
            v != 0
        } else {
            false
        };
        let cdf_retention_days = if off + 4 <= data.len() {
            read_u32(data, &mut off)?
        } else {
            0
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
            cdf_enabled,
            cdf_retention_days,
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
    Spatial = 3,
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
    pub parameters: Option<Vec<u8>>,
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
        match &self.parameters {
            Some(params) => {
                write_u8(&mut buf, 1);
                write_u32(&mut buf, params.len() as u32);
                buf.extend_from_slice(params);
            }
            None => {
                write_u8(&mut buf, 0);
            }
        }
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
        let parameters = if off < data.len() {
            let has_params = read_u8(data, &mut off)?;
            if has_params == 1 {
                let param_len = read_u32(data, &mut off)? as usize;
                if off + param_len > data.len() {
                    return Err(zyron_common::ZyronError::CatalogCorrupted(
                        "IndexEntry parameters truncated".to_string(),
                    ));
                }
                let params = data[off..off + param_len].to_vec();
                off += param_len;
                Some(params)
            } else {
                None
            }
        } else {
            None
        };
        Ok(Self {
            id,
            table_id,
            schema_id,
            name,
            columns,
            unique,
            index_file_id,
            index_type,
            parameters,
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
        3 => Ok(IndexType::Spatial),
        _ => Err(zyron_common::ZyronError::CatalogCorrupted(format!(
            "unknown IndexType value: {val}"
        ))),
    }
}

// ---------------------------------------------------------------------------
// StreamingJobEntry
// ---------------------------------------------------------------------------

/// Runtime status of a streaming job.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[repr(u8)]
pub enum StreamingJobStatus {
    Active = 0,
    Paused = 1,
    Failed = 2,
}

impl StreamingJobStatus {
    pub fn from_u8(val: u8) -> Result<Self> {
        match val {
            0 => Ok(StreamingJobStatus::Active),
            1 => Ok(StreamingJobStatus::Paused),
            2 => Ok(StreamingJobStatus::Failed),
            _ => Err(zyron_common::ZyronError::CatalogCorrupted(format!(
                "unknown StreamingJobStatus value: {val}"
            ))),
        }
    }
}

/// Write mode for a streaming job sink. Mirrors the parser's StreamingWriteMode
/// so the catalog crate does not depend on parser types.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[repr(u8)]
pub enum CatalogStreamingWriteMode {
    Append = 0,
    Upsert = 1,
}

impl CatalogStreamingWriteMode {
    pub fn from_u8(val: u8) -> Result<Self> {
        match val {
            0 => Ok(CatalogStreamingWriteMode::Append),
            1 => Ok(CatalogStreamingWriteMode::Upsert),
            _ => Err(zyron_common::ZyronError::CatalogCorrupted(format!(
                "unknown CatalogStreamingWriteMode value: {val}"
            ))),
        }
    }
}

/// Catalog entry for a streaming job. The creator's security context is
/// persisted as opaque bytes produced by SecurityContextSnapshot::to_bytes.
#[derive(Debug, Clone)]
pub struct StreamingJobEntry {
    pub id: StreamingJobId,
    pub name: String,
    pub source_table_id: TableId,
    pub target_table_id: TableId,
    pub source_schema_id: SchemaId,
    pub target_schema_id: SchemaId,
    pub select_sql: String,
    pub write_mode: CatalogStreamingWriteMode,
    pub status: StreamingJobStatus,
    pub creator_snapshot_bytes: Vec<u8>,
    pub created_at: u64,
    pub last_error: Option<String>,
}

impl StreamingJobEntry {
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut buf = Vec::with_capacity(128);
        write_u32(&mut buf, self.id.0);
        write_string(&mut buf, &self.name);
        write_u32(&mut buf, self.source_table_id.0);
        write_u32(&mut buf, self.target_table_id.0);
        write_u32(&mut buf, self.source_schema_id.0);
        write_u32(&mut buf, self.target_schema_id.0);
        write_string(&mut buf, &self.select_sql);
        write_u8(&mut buf, self.write_mode as u8);
        write_u8(&mut buf, self.status as u8);
        write_u32(&mut buf, self.creator_snapshot_bytes.len() as u32);
        buf.extend_from_slice(&self.creator_snapshot_bytes);
        write_u64(&mut buf, self.created_at);
        write_option_string(&mut buf, &self.last_error);
        buf
    }

    pub fn from_bytes(data: &[u8]) -> Result<Self> {
        let mut off = 0;
        let id = StreamingJobId(read_u32(data, &mut off)?);
        let name = read_string(data, &mut off)?;
        let source_table_id = TableId(read_u32(data, &mut off)?);
        let target_table_id = TableId(read_u32(data, &mut off)?);
        let source_schema_id = SchemaId(read_u32(data, &mut off)?);
        let target_schema_id = SchemaId(read_u32(data, &mut off)?);
        let select_sql = read_string(data, &mut off)?;
        let write_mode = CatalogStreamingWriteMode::from_u8(read_u8(data, &mut off)?)?;
        let status = StreamingJobStatus::from_u8(read_u8(data, &mut off)?)?;
        let snap_len = read_u32(data, &mut off)? as usize;
        if off + snap_len > data.len() {
            return Err(zyron_common::ZyronError::CatalogCorrupted(format!(
                "streaming job snapshot length {snap_len} exceeds remaining data at offset {off}"
            )));
        }
        let creator_snapshot_bytes = Vec::from(&data[off..off + snap_len]);
        off += snap_len;
        let created_at = read_u64(data, &mut off)?;
        let last_error = read_option_string(data, &mut off)?;
        Ok(Self {
            id,
            name,
            source_table_id,
            target_table_id,
            source_schema_id,
            target_schema_id,
            select_sql,
            write_mode,
            status,
            creator_snapshot_bytes,
            created_at,
            last_error,
        })
    }
}

// ---------------------------------------------------------------------------
// External source and sink catalog entries
// ---------------------------------------------------------------------------

// Mirror of the streaming-layer backend kind enum. Duplicated here so the
// catalog crate does not depend on zyron-streaming.
#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ExternalBackend {
    File = 0,
    S3 = 1,
    Gcs = 2,
    Azure = 3,
    Http = 4,
    Zyron = 5,
}

impl ExternalBackend {
    pub fn from_u8(val: u8) -> Result<Self> {
        match val {
            0 => Ok(ExternalBackend::File),
            1 => Ok(ExternalBackend::S3),
            2 => Ok(ExternalBackend::Gcs),
            3 => Ok(ExternalBackend::Azure),
            4 => Ok(ExternalBackend::Http),
            5 => Ok(ExternalBackend::Zyron),
            _ => Err(zyron_common::ZyronError::CatalogCorrupted(format!(
                "unknown ExternalBackend value: {val}"
            ))),
        }
    }
}

// Mirror of the streaming-layer format enum.
#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ExternalFormat {
    Json = 0,
    JsonLines = 1,
    Csv = 2,
    Parquet = 3,
    ArrowIpc = 4,
    Avro = 5,
}

impl ExternalFormat {
    pub fn from_u8(val: u8) -> Result<Self> {
        match val {
            0 => Ok(ExternalFormat::Json),
            1 => Ok(ExternalFormat::JsonLines),
            2 => Ok(ExternalFormat::Csv),
            3 => Ok(ExternalFormat::Parquet),
            4 => Ok(ExternalFormat::ArrowIpc),
            5 => Ok(ExternalFormat::Avro),
            _ => Err(zyron_common::ZyronError::CatalogCorrupted(format!(
                "unknown ExternalFormat value: {val}"
            ))),
        }
    }
}

// Ingestion mode for an external source.
#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ExternalMode {
    OneShot = 0,
    Scheduled = 1,
    Watch = 2,
}

impl ExternalMode {
    pub fn from_u8(val: u8) -> Result<Self> {
        match val {
            0 => Ok(ExternalMode::OneShot),
            1 => Ok(ExternalMode::Scheduled),
            2 => Ok(ExternalMode::Watch),
            _ => Err(zyron_common::ZyronError::CatalogCorrupted(format!(
                "unknown ExternalMode value: {val}"
            ))),
        }
    }
}

// Catalog classification mirror. Avoids a dependency on zyron-auth from the
// catalog crate. Values match ClassificationLevel Public=0 .. Restricted=3.
#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CatalogClassification {
    Public = 0,
    Internal = 1,
    Confidential = 2,
    Restricted = 3,
}

impl CatalogClassification {
    pub fn from_u8(val: u8) -> Result<Self> {
        match val {
            0 => Ok(CatalogClassification::Public),
            1 => Ok(CatalogClassification::Internal),
            2 => Ok(CatalogClassification::Confidential),
            3 => Ok(CatalogClassification::Restricted),
            _ => Err(zyron_common::ZyronError::CatalogCorrupted(format!(
                "unknown CatalogClassification value: {val}"
            ))),
        }
    }
}

/// Catalog entry for an external data source.
#[derive(Debug, Clone)]
pub struct ExternalSourceEntry {
    pub id: ExternalSourceId,
    pub schema_id: SchemaId,
    pub name: String,
    pub backend: ExternalBackend,
    pub uri: String,
    pub format: ExternalFormat,
    pub mode: ExternalMode,
    // Non-empty only when mode == Scheduled.
    pub schedule_cron: Option<String>,
    // Non-secret config, key-value pairs.
    pub options: Vec<(String, String)>,
    // Column layout declared at CREATE time or inferred from the first
    // matching file. Empty when the source's column shape is unknown at
    // CREATE time.
    pub columns: Vec<(String, TypeId)>,
    // None if no credentials are needed (local file, public http).
    pub credential_key_id: Option<u32>,
    pub credential_ciphertext: Option<Vec<u8>>,
    pub classification: CatalogClassification,
    pub tags: Vec<String>,
    pub owner_role_id: u32,
    pub created_at: u64,
}

impl ExternalSourceEntry {
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut buf = Vec::with_capacity(128);
        write_u32(&mut buf, self.id.0);
        write_u32(&mut buf, self.schema_id.0);
        write_string(&mut buf, &self.name);
        write_u8(&mut buf, self.backend as u8);
        write_string(&mut buf, &self.uri);
        write_u8(&mut buf, self.format as u8);
        write_u8(&mut buf, self.mode as u8);
        write_option_string(&mut buf, &self.schedule_cron);
        write_u32(&mut buf, self.options.len() as u32);
        for (k, v) in &self.options {
            write_string(&mut buf, k);
            write_string(&mut buf, v);
        }
        write_u32(&mut buf, self.columns.len() as u32);
        for (name, type_id) in &self.columns {
            write_string(&mut buf, name);
            write_u8(&mut buf, *type_id as u8);
        }
        // Credentials: both fields present together (both Some, or both None).
        let has_cred = self.credential_key_id.is_some() && self.credential_ciphertext.is_some();
        write_u8(&mut buf, has_cred as u8);
        if has_cred {
            write_u32(&mut buf, self.credential_key_id.unwrap());
            let ct = self.credential_ciphertext.as_ref().unwrap();
            write_u32(&mut buf, ct.len() as u32);
            buf.extend_from_slice(ct);
        }
        write_u8(&mut buf, self.classification as u8);
        write_u32(&mut buf, self.tags.len() as u32);
        for t in &self.tags {
            write_string(&mut buf, t);
        }
        write_u32(&mut buf, self.owner_role_id);
        write_u64(&mut buf, self.created_at);
        buf
    }

    pub fn from_bytes(data: &[u8]) -> Result<Self> {
        let mut off = 0;
        let id = ExternalSourceId(read_u32(data, &mut off)?);
        let schema_id = SchemaId(read_u32(data, &mut off)?);
        let name = read_string(data, &mut off)?;
        let backend = ExternalBackend::from_u8(read_u8(data, &mut off)?)?;
        let uri = read_string(data, &mut off)?;
        let format = ExternalFormat::from_u8(read_u8(data, &mut off)?)?;
        let mode = ExternalMode::from_u8(read_u8(data, &mut off)?)?;
        let schedule_cron = read_option_string(data, &mut off)?;
        let opt_count = read_u32(data, &mut off)? as usize;
        let mut options = Vec::with_capacity(opt_count);
        for _ in 0..opt_count {
            let k = read_string(data, &mut off)?;
            let v = read_string(data, &mut off)?;
            options.push((k, v));
        }
        let col_count = read_u32(data, &mut off)? as usize;
        let mut columns: Vec<(String, TypeId)> = Vec::with_capacity(col_count);
        for _ in 0..col_count {
            let name = read_string(data, &mut off)?;
            let type_id = type_id_from_u8(read_u8(data, &mut off)?)?;
            columns.push((name, type_id));
        }
        let has_cred = read_u8(data, &mut off)?;
        let (credential_key_id, credential_ciphertext) = if has_cred != 0 {
            let kid = read_u32(data, &mut off)?;
            let ct_len = read_u32(data, &mut off)? as usize;
            if off + ct_len > data.len() {
                return Err(zyron_common::ZyronError::CatalogCorrupted(format!(
                    "external source credential length {ct_len} exceeds remaining data at offset {off}"
                )));
            }
            let ct = data[off..off + ct_len].to_vec();
            off += ct_len;
            (Some(kid), Some(ct))
        } else {
            (None, None)
        };
        let classification = CatalogClassification::from_u8(read_u8(data, &mut off)?)?;
        let tag_count = read_u32(data, &mut off)? as usize;
        let mut tags = Vec::with_capacity(tag_count);
        for _ in 0..tag_count {
            tags.push(read_string(data, &mut off)?);
        }
        let owner_role_id = read_u32(data, &mut off)?;
        let created_at = read_u64(data, &mut off)?;
        Ok(Self {
            id,
            schema_id,
            name,
            backend,
            uri,
            format,
            mode,
            schedule_cron,
            options,
            columns,
            credential_key_id,
            credential_ciphertext,
            classification,
            tags,
            owner_role_id,
            created_at,
        })
    }
}

/// Catalog entry for an external data sink.
#[derive(Debug, Clone)]
pub struct ExternalSinkEntry {
    pub id: ExternalSinkId,
    pub schema_id: SchemaId,
    pub name: String,
    pub backend: ExternalBackend,
    pub uri: String,
    pub format: ExternalFormat,
    pub options: Vec<(String, String)>,
    // Column layout declared at CREATE time. Empty when unknown.
    pub columns: Vec<(String, TypeId)>,
    pub credential_key_id: Option<u32>,
    pub credential_ciphertext: Option<Vec<u8>>,
    pub classification: CatalogClassification,
    pub tags: Vec<String>,
    pub owner_role_id: u32,
    pub created_at: u64,
}

impl ExternalSinkEntry {
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut buf = Vec::with_capacity(128);
        write_u32(&mut buf, self.id.0);
        write_u32(&mut buf, self.schema_id.0);
        write_string(&mut buf, &self.name);
        write_u8(&mut buf, self.backend as u8);
        write_string(&mut buf, &self.uri);
        write_u8(&mut buf, self.format as u8);
        write_u32(&mut buf, self.options.len() as u32);
        for (k, v) in &self.options {
            write_string(&mut buf, k);
            write_string(&mut buf, v);
        }
        write_u32(&mut buf, self.columns.len() as u32);
        for (name, type_id) in &self.columns {
            write_string(&mut buf, name);
            write_u8(&mut buf, *type_id as u8);
        }
        let has_cred = self.credential_key_id.is_some() && self.credential_ciphertext.is_some();
        write_u8(&mut buf, has_cred as u8);
        if has_cred {
            write_u32(&mut buf, self.credential_key_id.unwrap());
            let ct = self.credential_ciphertext.as_ref().unwrap();
            write_u32(&mut buf, ct.len() as u32);
            buf.extend_from_slice(ct);
        }
        write_u8(&mut buf, self.classification as u8);
        write_u32(&mut buf, self.tags.len() as u32);
        for t in &self.tags {
            write_string(&mut buf, t);
        }
        write_u32(&mut buf, self.owner_role_id);
        write_u64(&mut buf, self.created_at);
        buf
    }

    pub fn from_bytes(data: &[u8]) -> Result<Self> {
        let mut off = 0;
        let id = ExternalSinkId(read_u32(data, &mut off)?);
        let schema_id = SchemaId(read_u32(data, &mut off)?);
        let name = read_string(data, &mut off)?;
        let backend = ExternalBackend::from_u8(read_u8(data, &mut off)?)?;
        let uri = read_string(data, &mut off)?;
        let format = ExternalFormat::from_u8(read_u8(data, &mut off)?)?;
        let opt_count = read_u32(data, &mut off)? as usize;
        let mut options = Vec::with_capacity(opt_count);
        for _ in 0..opt_count {
            let k = read_string(data, &mut off)?;
            let v = read_string(data, &mut off)?;
            options.push((k, v));
        }
        let col_count = read_u32(data, &mut off)? as usize;
        let mut columns: Vec<(String, TypeId)> = Vec::with_capacity(col_count);
        for _ in 0..col_count {
            let cname = read_string(data, &mut off)?;
            let type_id = type_id_from_u8(read_u8(data, &mut off)?)?;
            columns.push((cname, type_id));
        }
        let has_cred = read_u8(data, &mut off)?;
        let (credential_key_id, credential_ciphertext) = if has_cred != 0 {
            let kid = read_u32(data, &mut off)?;
            let ct_len = read_u32(data, &mut off)? as usize;
            if off + ct_len > data.len() {
                return Err(zyron_common::ZyronError::CatalogCorrupted(format!(
                    "external sink credential length {ct_len} exceeds remaining data at offset {off}"
                )));
            }
            let ct = data[off..off + ct_len].to_vec();
            off += ct_len;
            (Some(kid), Some(ct))
        } else {
            (None, None)
        };
        let classification = CatalogClassification::from_u8(read_u8(data, &mut off)?)?;
        let tag_count = read_u32(data, &mut off)? as usize;
        let mut tags = Vec::with_capacity(tag_count);
        for _ in 0..tag_count {
            tags.push(read_string(data, &mut off)?);
        }
        let owner_role_id = read_u32(data, &mut off)?;
        let created_at = read_u64(data, &mut off)?;
        Ok(Self {
            id,
            schema_id,
            name,
            backend,
            uri,
            format,
            options,
            columns,
            credential_key_id,
            credential_ciphertext,
            classification,
            tags,
            owner_role_id,
            created_at,
        })
    }
}

// ---------------------------------------------------------------------------
// Publication, Subscription, Endpoint, SecurityMap entries (Zyron-to-Zyron)
// ---------------------------------------------------------------------------

/// Row serialization format for publications.
#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum RowFormat {
    Binary = 0,
    Text = 1,
}

impl RowFormat {
    pub fn from_u8(v: u8) -> Result<Self> {
        match v {
            0 => Ok(RowFormat::Binary),
            1 => Ok(RowFormat::Text),
            _ => Err(zyron_common::ZyronError::CatalogCorrupted(format!(
                "unknown RowFormat value: {v}"
            ))),
        }
    }
}

/// Catalog entry for a publication. A publication exposes one or more tables
/// for remote consumption with optional predicate, column projection, and
/// row-level security rules.
#[derive(Debug, Clone)]
pub struct PublicationEntry {
    pub id: PublicationId,
    pub schema_id: SchemaId,
    pub name: String,
    pub change_feed: bool,
    pub row_format: RowFormat,
    pub retention_days: u32,
    pub retain_until_advance: bool,
    pub max_rows_per_sec: Option<u64>,
    pub max_bytes_per_sec: Option<u64>,
    pub max_concurrent_subscribers: Option<u32>,
    pub classification: CatalogClassification,
    pub allow_initial_snapshot: bool,
    pub where_predicate: Option<String>,
    pub columns_projection: Vec<String>,
    pub rls_using_predicate: Option<String>,
    pub tags: Vec<String>,
    pub schema_fingerprint: [u8; 32],
    pub owner_role_id: u32,
    pub created_at: u64,
}

#[inline]
fn write_option_u64(buf: &mut Vec<u8>, val: &Option<u64>) {
    match val {
        None => write_u8(buf, 0),
        Some(v) => {
            write_u8(buf, 1);
            write_u64(buf, *v);
        }
    }
}

#[inline]
fn read_option_u64(data: &[u8], offset: &mut usize) -> Result<Option<u64>> {
    let tag = read_u8(data, offset)?;
    match tag {
        0 => Ok(None),
        1 => Ok(Some(read_u64(data, offset)?)),
        _ => Err(zyron_common::ZyronError::CatalogCorrupted(format!(
            "invalid option tag {tag} reading u64"
        ))),
    }
}

#[inline]
fn write_option_u32(buf: &mut Vec<u8>, val: &Option<u32>) {
    match val {
        None => write_u8(buf, 0),
        Some(v) => {
            write_u8(buf, 1);
            write_u32(buf, *v);
        }
    }
}

#[inline]
fn read_option_u32(data: &[u8], offset: &mut usize) -> Result<Option<u32>> {
    let tag = read_u8(data, offset)?;
    match tag {
        0 => Ok(None),
        1 => Ok(Some(read_u32(data, offset)?)),
        _ => Err(zyron_common::ZyronError::CatalogCorrupted(format!(
            "invalid option tag {tag} reading u32"
        ))),
    }
}

impl PublicationEntry {
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut buf = Vec::with_capacity(256);
        write_u32(&mut buf, self.id.0);
        write_u32(&mut buf, self.schema_id.0);
        write_string(&mut buf, &self.name);
        write_bool(&mut buf, self.change_feed);
        write_u8(&mut buf, self.row_format as u8);
        write_u32(&mut buf, self.retention_days);
        write_bool(&mut buf, self.retain_until_advance);
        write_option_u64(&mut buf, &self.max_rows_per_sec);
        write_option_u64(&mut buf, &self.max_bytes_per_sec);
        write_option_u32(&mut buf, &self.max_concurrent_subscribers);
        write_u8(&mut buf, self.classification as u8);
        write_bool(&mut buf, self.allow_initial_snapshot);
        write_option_string(&mut buf, &self.where_predicate);
        write_u32(&mut buf, self.columns_projection.len() as u32);
        for c in &self.columns_projection {
            write_string(&mut buf, c);
        }
        write_option_string(&mut buf, &self.rls_using_predicate);
        write_u32(&mut buf, self.tags.len() as u32);
        for t in &self.tags {
            write_string(&mut buf, t);
        }
        buf.extend_from_slice(&self.schema_fingerprint);
        write_u32(&mut buf, self.owner_role_id);
        write_u64(&mut buf, self.created_at);
        buf
    }

    pub fn from_bytes(data: &[u8]) -> Result<Self> {
        let mut off = 0;
        let id = PublicationId(read_u32(data, &mut off)?);
        let schema_id = SchemaId(read_u32(data, &mut off)?);
        let name = read_string(data, &mut off)?;
        let change_feed = read_bool(data, &mut off)?;
        let row_format = RowFormat::from_u8(read_u8(data, &mut off)?)?;
        let retention_days = read_u32(data, &mut off)?;
        let retain_until_advance = read_bool(data, &mut off)?;
        let max_rows_per_sec = read_option_u64(data, &mut off)?;
        let max_bytes_per_sec = read_option_u64(data, &mut off)?;
        let max_concurrent_subscribers = read_option_u32(data, &mut off)?;
        let classification = CatalogClassification::from_u8(read_u8(data, &mut off)?)?;
        let allow_initial_snapshot = read_bool(data, &mut off)?;
        let where_predicate = read_option_string(data, &mut off)?;
        let col_count = read_u32(data, &mut off)? as usize;
        let mut columns_projection = Vec::with_capacity(col_count);
        for _ in 0..col_count {
            columns_projection.push(read_string(data, &mut off)?);
        }
        let rls_using_predicate = read_option_string(data, &mut off)?;
        let tag_count = read_u32(data, &mut off)? as usize;
        let mut tags = Vec::with_capacity(tag_count);
        for _ in 0..tag_count {
            tags.push(read_string(data, &mut off)?);
        }
        if off + 32 > data.len() {
            return Err(zyron_common::ZyronError::CatalogCorrupted(
                "publication schema_fingerprint truncated".to_string(),
            ));
        }
        let mut schema_fingerprint = [0u8; 32];
        schema_fingerprint.copy_from_slice(&data[off..off + 32]);
        off += 32;
        let owner_role_id = read_u32(data, &mut off)?;
        let created_at = read_u64(data, &mut off)?;
        Ok(Self {
            id,
            schema_id,
            name,
            change_feed,
            row_format,
            retention_days,
            retain_until_advance,
            max_rows_per_sec,
            max_bytes_per_sec,
            max_concurrent_subscribers,
            classification,
            allow_initial_snapshot,
            where_predicate,
            columns_projection,
            rls_using_predicate,
            tags,
            schema_fingerprint,
            owner_role_id,
            created_at,
        })
    }
}

/// Junction entry that ties a publication to one of its member tables,
/// optionally with per-table predicate and column projection overrides.
#[derive(Debug, Clone)]
pub struct PublicationTableEntry {
    pub id: u32,
    pub publication_id: PublicationId,
    pub table_id: TableId,
    pub where_predicate: Option<String>,
    pub columns: Vec<String>,
    pub created_at: u64,
}

impl PublicationTableEntry {
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut buf = Vec::with_capacity(64);
        write_u32(&mut buf, self.id);
        write_u32(&mut buf, self.publication_id.0);
        write_u32(&mut buf, self.table_id.0);
        write_option_string(&mut buf, &self.where_predicate);
        write_u32(&mut buf, self.columns.len() as u32);
        for c in &self.columns {
            write_string(&mut buf, c);
        }
        write_u64(&mut buf, self.created_at);
        buf
    }

    pub fn from_bytes(data: &[u8]) -> Result<Self> {
        let mut off = 0;
        let id = read_u32(data, &mut off)?;
        let publication_id = PublicationId(read_u32(data, &mut off)?);
        let table_id = TableId(read_u32(data, &mut off)?);
        let where_predicate = read_option_string(data, &mut off)?;
        let col_count = read_u32(data, &mut off)? as usize;
        let mut columns = Vec::with_capacity(col_count);
        for _ in 0..col_count {
            columns.push(read_string(data, &mut off)?);
        }
        let created_at = read_u64(data, &mut off)?;
        Ok(Self {
            id,
            publication_id,
            table_id,
            where_predicate,
            columns,
            created_at,
        })
    }
}

/// Delivery mode for a subscription.
#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SubscriptionMode {
    Pull = 0,
    Push = 1,
    Snapshot = 2,
}

impl SubscriptionMode {
    pub fn from_u8(v: u8) -> Result<Self> {
        match v {
            0 => Ok(SubscriptionMode::Pull),
            1 => Ok(SubscriptionMode::Push),
            2 => Ok(SubscriptionMode::Snapshot),
            _ => Err(zyron_common::ZyronError::CatalogCorrupted(format!(
                "unknown SubscriptionMode value: {v}"
            ))),
        }
    }
}

/// Runtime state for a subscription.
#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SubscriptionState {
    Active = 0,
    Paused = 1,
    Failed = 2,
}

impl SubscriptionState {
    pub fn from_u8(v: u8) -> Result<Self> {
        match v {
            0 => Ok(SubscriptionState::Active),
            1 => Ok(SubscriptionState::Paused),
            2 => Ok(SubscriptionState::Failed),
            _ => Err(zyron_common::ZyronError::CatalogCorrupted(format!(
                "unknown SubscriptionState value: {v}"
            ))),
        }
    }
}

/// Catalog entry for a subscription to a remote publication.
#[derive(Debug, Clone)]
pub struct SubscriptionEntry {
    pub id: SubscriptionId,
    pub publication_id: PublicationId,
    pub consumer_id: String,
    pub consumer_role_id: u32,
    pub last_seen_lsn: u64,
    pub last_poll_at: u64,
    pub schema_pin: [u8; 32],
    pub mode: SubscriptionMode,
    pub state: SubscriptionState,
    pub last_error: Option<String>,
    pub created_at: u64,
    // Explicit source binding. When Some, resume_subscription uses a direct
    // get_external_source_by_id lookup. When None, the heuristic that scans
    // every external source looking for a matching publication name is used.
    // Older entries serialized before this field was added decode as None.
    pub source_id: Option<ExternalSourceId>,
}

impl SubscriptionEntry {
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut buf = Vec::with_capacity(128);
        write_u32(&mut buf, self.id.0);
        write_u32(&mut buf, self.publication_id.0);
        write_string(&mut buf, &self.consumer_id);
        write_u32(&mut buf, self.consumer_role_id);
        write_u64(&mut buf, self.last_seen_lsn);
        write_u64(&mut buf, self.last_poll_at);
        buf.extend_from_slice(&self.schema_pin);
        write_u8(&mut buf, self.mode as u8);
        write_u8(&mut buf, self.state as u8);
        write_option_string(&mut buf, &self.last_error);
        write_u64(&mut buf, self.created_at);
        // Append-only trailer, zero tag means absent. Readers that know the
        // field decode it, older readers that stop at created_at ignore it.
        match self.source_id {
            Some(sid) => {
                write_u8(&mut buf, 1);
                write_u32(&mut buf, sid.0);
            }
            None => {
                write_u8(&mut buf, 0);
            }
        }
        buf
    }

    pub fn from_bytes(data: &[u8]) -> Result<Self> {
        let mut off = 0;
        let id = SubscriptionId(read_u32(data, &mut off)?);
        let publication_id = PublicationId(read_u32(data, &mut off)?);
        let consumer_id = read_string(data, &mut off)?;
        let consumer_role_id = read_u32(data, &mut off)?;
        let last_seen_lsn = read_u64(data, &mut off)?;
        let last_poll_at = read_u64(data, &mut off)?;
        if off + 32 > data.len() {
            return Err(zyron_common::ZyronError::CatalogCorrupted(
                "subscription schema_pin truncated".to_string(),
            ));
        }
        let mut schema_pin = [0u8; 32];
        schema_pin.copy_from_slice(&data[off..off + 32]);
        off += 32;
        let mode = SubscriptionMode::from_u8(read_u8(data, &mut off)?)?;
        let state = SubscriptionState::from_u8(read_u8(data, &mut off)?)?;
        let last_error = read_option_string(data, &mut off)?;
        let created_at = read_u64(data, &mut off)?;
        // Optional source_id trailer. Older entries have no trailing bytes,
        // treat that as None rather than a decode error.
        let source_id = if off < data.len() {
            let tag = read_u8(data, &mut off)?;
            match tag {
                0 => None,
                1 => Some(ExternalSourceId(read_u32(data, &mut off)?)),
                _ => {
                    return Err(zyron_common::ZyronError::CatalogCorrupted(format!(
                        "unknown SubscriptionEntry.source_id tag: {tag}"
                    )));
                }
            }
        } else {
            None
        };
        Ok(Self {
            id,
            publication_id,
            consumer_id,
            consumer_role_id,
            last_seen_lsn,
            last_poll_at,
            schema_pin,
            mode,
            state,
            last_error,
            created_at,
            source_id,
        })
    }
}

/// Dynamic endpoint kinds: REST, WebSocket, or Server-Sent Events.
#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum EndpointKind {
    Rest = 0,
    WebSocket = 1,
    Sse = 2,
}

impl EndpointKind {
    pub fn from_u8(v: u8) -> Result<Self> {
        match v {
            0 => Ok(EndpointKind::Rest),
            1 => Ok(EndpointKind::WebSocket),
            2 => Ok(EndpointKind::Sse),
            _ => Err(zyron_common::ZyronError::CatalogCorrupted(format!(
                "unknown EndpointKind value: {v}"
            ))),
        }
    }
}

/// HTTP methods accepted by a REST endpoint.
#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum HttpMethod {
    Get = 0,
    Post = 1,
    Put = 2,
    Delete = 3,
    Patch = 4,
    Head = 5,
    Options = 6,
}

impl HttpMethod {
    pub fn from_u8(v: u8) -> Result<Self> {
        match v {
            0 => Ok(HttpMethod::Get),
            1 => Ok(HttpMethod::Post),
            2 => Ok(HttpMethod::Put),
            3 => Ok(HttpMethod::Delete),
            4 => Ok(HttpMethod::Patch),
            5 => Ok(HttpMethod::Head),
            6 => Ok(HttpMethod::Options),
            _ => Err(zyron_common::ZyronError::CatalogCorrupted(format!(
                "unknown HttpMethod value: {v}"
            ))),
        }
    }
}

/// Authentication mode for an endpoint.
#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum EndpointAuthMode {
    None = 0,
    Jwt = 1,
    ApiKey = 2,
    OAuth2 = 3,
    Basic = 4,
    Mtls = 5,
}

impl EndpointAuthMode {
    pub fn from_u8(v: u8) -> Result<Self> {
        match v {
            0 => Ok(EndpointAuthMode::None),
            1 => Ok(EndpointAuthMode::Jwt),
            2 => Ok(EndpointAuthMode::ApiKey),
            3 => Ok(EndpointAuthMode::OAuth2),
            4 => Ok(EndpointAuthMode::Basic),
            5 => Ok(EndpointAuthMode::Mtls),
            _ => Err(zyron_common::ZyronError::CatalogCorrupted(format!(
                "unknown EndpointAuthMode value: {v}"
            ))),
        }
    }
}

/// Output encoding for a REST endpoint response.
#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum EndpointOutputFormat {
    Json = 0,
    JsonLines = 1,
    Csv = 2,
    Parquet = 3,
    ArrowIpc = 4,
}

impl EndpointOutputFormat {
    pub fn from_u8(v: u8) -> Result<Self> {
        match v {
            0 => Ok(EndpointOutputFormat::Json),
            1 => Ok(EndpointOutputFormat::JsonLines),
            2 => Ok(EndpointOutputFormat::Csv),
            3 => Ok(EndpointOutputFormat::Parquet),
            4 => Ok(EndpointOutputFormat::ArrowIpc),
            _ => Err(zyron_common::ZyronError::CatalogCorrupted(format!(
                "unknown EndpointOutputFormat value: {v}"
            ))),
        }
    }
}

/// Message encoding for a streaming endpoint (WebSocket or SSE).
#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum EndpointMessageFormat {
    Json = 0,
    JsonLines = 1,
    Protobuf = 2,
}

impl EndpointMessageFormat {
    pub fn from_u8(v: u8) -> Result<Self> {
        match v {
            0 => Ok(EndpointMessageFormat::Json),
            1 => Ok(EndpointMessageFormat::JsonLines),
            2 => Ok(EndpointMessageFormat::Protobuf),
            _ => Err(zyron_common::ZyronError::CatalogCorrupted(format!(
                "unknown EndpointMessageFormat value: {v}"
            ))),
        }
    }
}

/// Policy applied when a streaming endpoint client falls behind.
#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum BackpressurePolicy {
    DropOldest = 0,
    CloseSlow = 1,
    Block = 2,
}

impl BackpressurePolicy {
    pub fn from_u8(v: u8) -> Result<Self> {
        match v {
            0 => Ok(BackpressurePolicy::DropOldest),
            1 => Ok(BackpressurePolicy::CloseSlow),
            2 => Ok(BackpressurePolicy::Block),
            _ => Err(zyron_common::ZyronError::CatalogCorrupted(format!(
                "unknown BackpressurePolicy value: {v}"
            ))),
        }
    }
}

/// Rate limit measurement window.
#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum RateLimitPeriod {
    Second = 0,
    Minute = 1,
    Hour = 2,
    Day = 3,
}

impl RateLimitPeriod {
    pub fn from_u8(v: u8) -> Result<Self> {
        match v {
            0 => Ok(RateLimitPeriod::Second),
            1 => Ok(RateLimitPeriod::Minute),
            2 => Ok(RateLimitPeriod::Hour),
            3 => Ok(RateLimitPeriod::Day),
            _ => Err(zyron_common::ZyronError::CatalogCorrupted(format!(
                "unknown RateLimitPeriod value: {v}"
            ))),
        }
    }
}

/// Rate limit application scope.
#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum RateLimitScope {
    Global = 0,
    PerIp = 1,
    PerUser = 2,
    PerApiKey = 3,
}

impl RateLimitScope {
    pub fn from_u8(v: u8) -> Result<Self> {
        match v {
            0 => Ok(RateLimitScope::Global),
            1 => Ok(RateLimitScope::PerIp),
            2 => Ok(RateLimitScope::PerUser),
            3 => Ok(RateLimitScope::PerApiKey),
            _ => Err(zyron_common::ZyronError::CatalogCorrupted(format!(
                "unknown RateLimitScope value: {v}"
            ))),
        }
    }
}

/// Rate limit settings attached to an endpoint.
#[derive(Debug, Clone)]
pub struct RateLimitSpec {
    pub count: u64,
    pub period: RateLimitPeriod,
    pub scope: RateLimitScope,
}

/// Catalog entry for a dynamic HTTP or streaming endpoint.
#[derive(Debug, Clone)]
pub struct EndpointEntry {
    pub id: EndpointId,
    pub schema_id: SchemaId,
    pub name: String,
    pub kind: EndpointKind,
    pub path: String,
    pub methods: Vec<HttpMethod>,
    pub sql_body: String,
    pub backed_publication_id: Option<PublicationId>,
    pub auth_mode: EndpointAuthMode,
    pub required_scopes: Vec<String>,
    pub output_format: Option<EndpointOutputFormat>,
    pub cors_origins: Vec<String>,
    pub rate_limit: Option<RateLimitSpec>,
    pub cache_seconds: Option<u32>,
    pub timeout_seconds: Option<u32>,
    pub max_request_body_kb: Option<u32>,
    pub message_format: Option<EndpointMessageFormat>,
    pub heartbeat_seconds: Option<u32>,
    pub backpressure: Option<BackpressurePolicy>,
    pub max_connections: Option<u32>,
    pub enabled: bool,
    pub owner_role_id: u32,
    pub created_at: u64,
}

impl EndpointEntry {
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut buf = Vec::with_capacity(256);
        write_u32(&mut buf, self.id.0);
        write_u32(&mut buf, self.schema_id.0);
        write_string(&mut buf, &self.name);
        write_u8(&mut buf, self.kind as u8);
        write_string(&mut buf, &self.path);
        write_u32(&mut buf, self.methods.len() as u32);
        for m in &self.methods {
            write_u8(&mut buf, *m as u8);
        }
        write_string(&mut buf, &self.sql_body);
        match &self.backed_publication_id {
            None => write_u8(&mut buf, 0),
            Some(p) => {
                write_u8(&mut buf, 1);
                write_u32(&mut buf, p.0);
            }
        }
        write_u8(&mut buf, self.auth_mode as u8);
        write_u32(&mut buf, self.required_scopes.len() as u32);
        for s in &self.required_scopes {
            write_string(&mut buf, s);
        }
        match &self.output_format {
            None => write_u8(&mut buf, 0),
            Some(f) => {
                write_u8(&mut buf, 1);
                write_u8(&mut buf, *f as u8);
            }
        }
        write_u32(&mut buf, self.cors_origins.len() as u32);
        for o in &self.cors_origins {
            write_string(&mut buf, o);
        }
        match &self.rate_limit {
            None => write_u8(&mut buf, 0),
            Some(r) => {
                write_u8(&mut buf, 1);
                write_u64(&mut buf, r.count);
                write_u8(&mut buf, r.period as u8);
                write_u8(&mut buf, r.scope as u8);
            }
        }
        write_option_u32(&mut buf, &self.cache_seconds);
        write_option_u32(&mut buf, &self.timeout_seconds);
        write_option_u32(&mut buf, &self.max_request_body_kb);
        match &self.message_format {
            None => write_u8(&mut buf, 0),
            Some(f) => {
                write_u8(&mut buf, 1);
                write_u8(&mut buf, *f as u8);
            }
        }
        write_option_u32(&mut buf, &self.heartbeat_seconds);
        match &self.backpressure {
            None => write_u8(&mut buf, 0),
            Some(b) => {
                write_u8(&mut buf, 1);
                write_u8(&mut buf, *b as u8);
            }
        }
        write_option_u32(&mut buf, &self.max_connections);
        write_bool(&mut buf, self.enabled);
        write_u32(&mut buf, self.owner_role_id);
        write_u64(&mut buf, self.created_at);
        buf
    }

    pub fn from_bytes(data: &[u8]) -> Result<Self> {
        let mut off = 0;
        let id = EndpointId(read_u32(data, &mut off)?);
        let schema_id = SchemaId(read_u32(data, &mut off)?);
        let name = read_string(data, &mut off)?;
        let kind = EndpointKind::from_u8(read_u8(data, &mut off)?)?;
        let path = read_string(data, &mut off)?;
        let method_count = read_u32(data, &mut off)? as usize;
        let mut methods = Vec::with_capacity(method_count);
        for _ in 0..method_count {
            methods.push(HttpMethod::from_u8(read_u8(data, &mut off)?)?);
        }
        let sql_body = read_string(data, &mut off)?;
        let backed_publication_id = match read_u8(data, &mut off)? {
            0 => None,
            1 => Some(PublicationId(read_u32(data, &mut off)?)),
            t => {
                return Err(zyron_common::ZyronError::CatalogCorrupted(format!(
                    "invalid backed_publication_id tag {t}"
                )));
            }
        };
        let auth_mode = EndpointAuthMode::from_u8(read_u8(data, &mut off)?)?;
        let scope_count = read_u32(data, &mut off)? as usize;
        let mut required_scopes = Vec::with_capacity(scope_count);
        for _ in 0..scope_count {
            required_scopes.push(read_string(data, &mut off)?);
        }
        let output_format = match read_u8(data, &mut off)? {
            0 => None,
            1 => Some(EndpointOutputFormat::from_u8(read_u8(data, &mut off)?)?),
            t => {
                return Err(zyron_common::ZyronError::CatalogCorrupted(format!(
                    "invalid output_format tag {t}"
                )));
            }
        };
        let origin_count = read_u32(data, &mut off)? as usize;
        let mut cors_origins = Vec::with_capacity(origin_count);
        for _ in 0..origin_count {
            cors_origins.push(read_string(data, &mut off)?);
        }
        let rate_limit = match read_u8(data, &mut off)? {
            0 => None,
            1 => {
                let count = read_u64(data, &mut off)?;
                let period = RateLimitPeriod::from_u8(read_u8(data, &mut off)?)?;
                let scope = RateLimitScope::from_u8(read_u8(data, &mut off)?)?;
                Some(RateLimitSpec {
                    count,
                    period,
                    scope,
                })
            }
            t => {
                return Err(zyron_common::ZyronError::CatalogCorrupted(format!(
                    "invalid rate_limit tag {t}"
                )));
            }
        };
        let cache_seconds = read_option_u32(data, &mut off)?;
        let timeout_seconds = read_option_u32(data, &mut off)?;
        let max_request_body_kb = read_option_u32(data, &mut off)?;
        let message_format = match read_u8(data, &mut off)? {
            0 => None,
            1 => Some(EndpointMessageFormat::from_u8(read_u8(data, &mut off)?)?),
            t => {
                return Err(zyron_common::ZyronError::CatalogCorrupted(format!(
                    "invalid message_format tag {t}"
                )));
            }
        };
        let heartbeat_seconds = read_option_u32(data, &mut off)?;
        let backpressure = match read_u8(data, &mut off)? {
            0 => None,
            1 => Some(BackpressurePolicy::from_u8(read_u8(data, &mut off)?)?),
            t => {
                return Err(zyron_common::ZyronError::CatalogCorrupted(format!(
                    "invalid backpressure tag {t}"
                )));
            }
        };
        let max_connections = read_option_u32(data, &mut off)?;
        let enabled = read_bool(data, &mut off)?;
        let owner_role_id = read_u32(data, &mut off)?;
        let created_at = read_u64(data, &mut off)?;
        Ok(Self {
            id,
            schema_id,
            name,
            kind,
            path,
            methods,
            sql_body,
            backed_publication_id,
            auth_mode,
            required_scopes,
            output_format,
            cors_origins,
            rate_limit,
            cache_seconds,
            timeout_seconds,
            max_request_body_kb,
            message_format,
            heartbeat_seconds,
            backpressure,
            max_connections,
            enabled,
            owner_role_id,
            created_at,
        })
    }
}

/// Identity binding source for a security map entry.
#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum SecurityMapKind {
    K8sSa = 0,
    Jwt = 1,
    MtlsSubject = 2,
    MtlsFingerprint = 3,
}

impl SecurityMapKind {
    pub fn from_u8(v: u8) -> Result<Self> {
        match v {
            0 => Ok(SecurityMapKind::K8sSa),
            1 => Ok(SecurityMapKind::Jwt),
            2 => Ok(SecurityMapKind::MtlsSubject),
            3 => Ok(SecurityMapKind::MtlsFingerprint),
            _ => Err(zyron_common::ZyronError::CatalogCorrupted(format!(
                "unknown SecurityMapKind value: {v}"
            ))),
        }
    }
}

/// Persistent binding from an external identity to a local role.
#[derive(Debug, Clone)]
pub struct SecurityMapEntry {
    pub id: SecurityMapId,
    pub kind: SecurityMapKind,
    pub key: String,
    pub role_id: u32,
    pub created_at: u64,
}

impl SecurityMapEntry {
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut buf = Vec::with_capacity(64);
        write_u32(&mut buf, self.id.0);
        write_u8(&mut buf, self.kind as u8);
        write_string(&mut buf, &self.key);
        write_u32(&mut buf, self.role_id);
        write_u64(&mut buf, self.created_at);
        buf
    }

    pub fn from_bytes(data: &[u8]) -> Result<Self> {
        let mut off = 0;
        let id = SecurityMapId(read_u32(data, &mut off)?);
        let kind = SecurityMapKind::from_u8(read_u8(data, &mut off)?)?;
        let key = read_string(data, &mut off)?;
        let role_id = read_u32(data, &mut off)?;
        let created_at = read_u64(data, &mut off)?;
        Ok(Self {
            id,
            kind,
            key,
            role_id,
            created_at,
        })
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
            cdf_enabled: false,
            cdf_retention_days: 0,
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
            parameters: None,
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
            cdf_enabled: false,
            cdf_retention_days: 0,
        };
        let bytes = entry.to_bytes();
        let decoded = TableEntry::from_bytes(&bytes).unwrap();
        assert_eq!(decoded.columns.len(), 0);
        assert_eq!(decoded.constraints.len(), 0);
    }

    #[test]
    fn test_streaming_job_entry_roundtrip() {
        let entry = StreamingJobEntry {
            id: StreamingJobId(42),
            name: "orders_to_warehouse".to_string(),
            source_table_id: TableId(100),
            target_table_id: TableId(200),
            source_schema_id: SchemaId(1),
            target_schema_id: SchemaId(2),
            select_sql: "SELECT id, amount FROM orders WHERE amount > 0".to_string(),
            write_mode: CatalogStreamingWriteMode::Upsert,
            status: StreamingJobStatus::Paused,
            creator_snapshot_bytes: vec![0x01, 0x02, 0x03, 0xde, 0xad, 0xbe, 0xef],
            created_at: 1_700_000_000,
            last_error: Some("connection reset".to_string()),
        };
        let bytes = entry.to_bytes();
        let decoded = StreamingJobEntry::from_bytes(&bytes).unwrap();
        assert_eq!(decoded.id, entry.id);
        assert_eq!(decoded.name, entry.name);
        assert_eq!(decoded.source_table_id, entry.source_table_id);
        assert_eq!(decoded.target_table_id, entry.target_table_id);
        assert_eq!(decoded.source_schema_id, entry.source_schema_id);
        assert_eq!(decoded.target_schema_id, entry.target_schema_id);
        assert_eq!(decoded.select_sql, entry.select_sql);
        assert_eq!(decoded.write_mode, entry.write_mode);
        assert_eq!(decoded.status, entry.status);
        assert_eq!(decoded.creator_snapshot_bytes, entry.creator_snapshot_bytes);
        assert_eq!(decoded.created_at, entry.created_at);
        assert_eq!(decoded.last_error, entry.last_error);
    }

    #[test]
    fn test_external_source_entry_roundtrip() {
        let entry = ExternalSourceEntry {
            id: ExternalSourceId(77),
            schema_id: SchemaId(3),
            name: "orders_source".to_string(),
            backend: ExternalBackend::S3,
            uri: "s3://bucket/prefix/".to_string(),
            format: ExternalFormat::Parquet,
            mode: ExternalMode::Scheduled,
            schedule_cron: Some("0 */5 * * *".to_string()),
            options: vec![
                ("region".to_string(), "us-west-2".to_string()),
                ("batch_size".to_string(), "1024".to_string()),
            ],
            columns: vec![
                ("order_id".to_string(), TypeId::Int64),
                ("amount".to_string(), TypeId::Decimal),
            ],
            credential_key_id: Some(42),
            credential_ciphertext: Some(vec![0xaa, 0xbb, 0xcc, 0xdd, 0xee]),
            classification: CatalogClassification::Confidential,
            tags: vec!["pii".to_string(), "prod".to_string()],
            owner_role_id: 9,
            created_at: 1_700_000_000,
        };
        let bytes = entry.to_bytes();
        let decoded = ExternalSourceEntry::from_bytes(&bytes).unwrap();
        assert_eq!(decoded.id, entry.id);
        assert_eq!(decoded.schema_id, entry.schema_id);
        assert_eq!(decoded.name, entry.name);
        assert_eq!(decoded.backend, entry.backend);
        assert_eq!(decoded.uri, entry.uri);
        assert_eq!(decoded.format, entry.format);
        assert_eq!(decoded.mode, entry.mode);
        assert_eq!(decoded.schedule_cron, entry.schedule_cron);
        assert_eq!(decoded.options, entry.options);
        assert_eq!(decoded.columns, entry.columns);
        assert_eq!(decoded.credential_key_id, entry.credential_key_id);
        assert_eq!(decoded.credential_ciphertext, entry.credential_ciphertext);
        assert_eq!(decoded.classification, entry.classification);
        assert_eq!(decoded.tags, entry.tags);
        assert_eq!(decoded.owner_role_id, entry.owner_role_id);
        assert_eq!(decoded.created_at, entry.created_at);
    }

    #[test]
    fn test_external_sink_entry_roundtrip() {
        let entry = ExternalSinkEntry {
            id: ExternalSinkId(88),
            schema_id: SchemaId(4),
            name: "warehouse_sink".to_string(),
            backend: ExternalBackend::Gcs,
            uri: "gs://bucket/out/".to_string(),
            format: ExternalFormat::JsonLines,
            options: vec![
                ("compression".to_string(), "gzip".to_string()),
                ("flush_ms".to_string(), "500".to_string()),
            ],
            columns: vec![
                ("event_id".to_string(), TypeId::Int64),
                ("name".to_string(), TypeId::Text),
            ],
            credential_key_id: Some(7),
            credential_ciphertext: Some(vec![0x01, 0x02, 0x03, 0x04]),
            classification: CatalogClassification::Restricted,
            tags: vec!["export".to_string()],
            owner_role_id: 3,
            created_at: 1_700_000_123,
        };
        let bytes = entry.to_bytes();
        let decoded = ExternalSinkEntry::from_bytes(&bytes).unwrap();
        assert_eq!(decoded.id, entry.id);
        assert_eq!(decoded.schema_id, entry.schema_id);
        assert_eq!(decoded.name, entry.name);
        assert_eq!(decoded.backend, entry.backend);
        assert_eq!(decoded.uri, entry.uri);
        assert_eq!(decoded.format, entry.format);
        assert_eq!(decoded.options, entry.options);
        assert_eq!(decoded.columns, entry.columns);
        assert_eq!(decoded.credential_key_id, entry.credential_key_id);
        assert_eq!(decoded.credential_ciphertext, entry.credential_ciphertext);
        assert_eq!(decoded.classification, entry.classification);
        assert_eq!(decoded.tags, entry.tags);
        assert_eq!(decoded.owner_role_id, entry.owner_role_id);
        assert_eq!(decoded.created_at, entry.created_at);
    }

    #[test]
    fn test_publication_entry_roundtrip() {
        let mut fp = [0u8; 32];
        for (i, b) in fp.iter_mut().enumerate() {
            *b = i as u8;
        }
        let entry = PublicationEntry {
            id: PublicationId(101),
            schema_id: SchemaId(1),
            name: "orders_pub".to_string(),
            change_feed: true,
            row_format: RowFormat::Text,
            retention_days: 14,
            retain_until_advance: true,
            max_rows_per_sec: Some(100_000),
            max_bytes_per_sec: Some(50 * 1024 * 1024),
            max_concurrent_subscribers: Some(16),
            classification: CatalogClassification::Confidential,
            allow_initial_snapshot: true,
            where_predicate: Some("amount > 0".to_string()),
            columns_projection: vec!["id".to_string(), "amount".to_string()],
            rls_using_predicate: Some("tenant_id = current_tenant()".to_string()),
            tags: vec!["pii".to_string(), "prod".to_string()],
            schema_fingerprint: fp,
            owner_role_id: 7,
            created_at: 1_700_000_000,
        };
        let bytes = entry.to_bytes();
        let decoded = PublicationEntry::from_bytes(&bytes).unwrap();
        assert_eq!(decoded.id, entry.id);
        assert_eq!(decoded.schema_id, entry.schema_id);
        assert_eq!(decoded.name, entry.name);
        assert_eq!(decoded.change_feed, entry.change_feed);
        assert_eq!(decoded.row_format, entry.row_format);
        assert_eq!(decoded.retention_days, entry.retention_days);
        assert_eq!(decoded.retain_until_advance, entry.retain_until_advance);
        assert_eq!(decoded.max_rows_per_sec, entry.max_rows_per_sec);
        assert_eq!(decoded.max_bytes_per_sec, entry.max_bytes_per_sec);
        assert_eq!(
            decoded.max_concurrent_subscribers,
            entry.max_concurrent_subscribers
        );
        assert_eq!(decoded.classification, entry.classification);
        assert_eq!(decoded.allow_initial_snapshot, entry.allow_initial_snapshot);
        assert_eq!(decoded.where_predicate, entry.where_predicate);
        assert_eq!(decoded.columns_projection, entry.columns_projection);
        assert_eq!(decoded.rls_using_predicate, entry.rls_using_predicate);
        assert_eq!(decoded.tags, entry.tags);
        assert_eq!(decoded.schema_fingerprint, entry.schema_fingerprint);
        assert_eq!(decoded.owner_role_id, entry.owner_role_id);
        assert_eq!(decoded.created_at, entry.created_at);
    }

    #[test]
    fn test_publication_table_entry_roundtrip() {
        let entry = PublicationTableEntry {
            id: 42,
            publication_id: PublicationId(101),
            table_id: TableId(500),
            where_predicate: Some("status = 'active'".to_string()),
            columns: vec!["a".to_string(), "b".to_string()],
            created_at: 1_700_000_100,
        };
        let bytes = entry.to_bytes();
        let decoded = PublicationTableEntry::from_bytes(&bytes).unwrap();
        assert_eq!(decoded.id, entry.id);
        assert_eq!(decoded.publication_id, entry.publication_id);
        assert_eq!(decoded.table_id, entry.table_id);
        assert_eq!(decoded.where_predicate, entry.where_predicate);
        assert_eq!(decoded.columns, entry.columns);
        assert_eq!(decoded.created_at, entry.created_at);
    }

    #[test]
    fn test_subscription_entry_roundtrip() {
        let mut pin = [0u8; 32];
        pin[0] = 0xab;
        pin[31] = 0xcd;
        let entry = SubscriptionEntry {
            id: SubscriptionId(9),
            publication_id: PublicationId(101),
            consumer_id: "consumer-alpha".to_string(),
            consumer_role_id: 3,
            last_seen_lsn: 9876543210,
            last_poll_at: 1_700_000_200,
            schema_pin: pin,
            mode: SubscriptionMode::Push,
            state: SubscriptionState::Failed,
            last_error: Some("timeout".to_string()),
            created_at: 1_700_000_150,
            source_id: Some(ExternalSourceId(73)),
        };
        let bytes = entry.to_bytes();
        let decoded = SubscriptionEntry::from_bytes(&bytes).unwrap();
        assert_eq!(decoded.id, entry.id);
        assert_eq!(decoded.publication_id, entry.publication_id);
        assert_eq!(decoded.consumer_id, entry.consumer_id);
        assert_eq!(decoded.consumer_role_id, entry.consumer_role_id);
        assert_eq!(decoded.last_seen_lsn, entry.last_seen_lsn);
        assert_eq!(decoded.last_poll_at, entry.last_poll_at);
        assert_eq!(decoded.schema_pin, entry.schema_pin);
        assert_eq!(decoded.mode, entry.mode);
        assert_eq!(decoded.state, entry.state);
        assert_eq!(decoded.last_error, entry.last_error);
        assert_eq!(decoded.created_at, entry.created_at);
        assert_eq!(decoded.source_id, entry.source_id);
    }

    #[test]
    fn test_subscription_entry_decodes_older_layout_without_source_id() {
        // An older-layout byte sequence ends at created_at with no trailing
        // source_id tag. Decoding must succeed and return source_id = None.
        let mut pin = [0u8; 32];
        pin[5] = 0x11;
        let mut buf = Vec::new();
        write_u32(&mut buf, 7);
        write_u32(&mut buf, 8);
        write_string(&mut buf, "c");
        write_u32(&mut buf, 0);
        write_u64(&mut buf, 100);
        write_u64(&mut buf, 200);
        buf.extend_from_slice(&pin);
        write_u8(&mut buf, SubscriptionMode::Pull as u8);
        write_u8(&mut buf, SubscriptionState::Active as u8);
        write_option_string(&mut buf, &None);
        write_u64(&mut buf, 300);
        let decoded = SubscriptionEntry::from_bytes(&buf).unwrap();
        assert_eq!(decoded.source_id, None);
        assert_eq!(decoded.id.0, 7);
    }

    #[test]
    fn test_endpoint_entry_roundtrip_full() {
        let entry = EndpointEntry {
            id: EndpointId(55),
            schema_id: SchemaId(1),
            name: "orders_api".to_string(),
            kind: EndpointKind::Rest,
            path: "/api/orders/$id".to_string(),
            methods: vec![HttpMethod::Get, HttpMethod::Post],
            sql_body: "SELECT * FROM orders WHERE id = $id".to_string(),
            backed_publication_id: None,
            auth_mode: EndpointAuthMode::Jwt,
            required_scopes: vec!["orders.read".to_string()],
            output_format: Some(EndpointOutputFormat::Json),
            cors_origins: vec!["https://example.com".to_string()],
            rate_limit: Some(RateLimitSpec {
                count: 1000,
                period: RateLimitPeriod::Minute,
                scope: RateLimitScope::PerUser,
            }),
            cache_seconds: Some(30),
            timeout_seconds: Some(10),
            max_request_body_kb: Some(256),
            message_format: None,
            heartbeat_seconds: None,
            backpressure: None,
            max_connections: None,
            enabled: true,
            owner_role_id: 4,
            created_at: 1_700_000_300,
        };
        let bytes = entry.to_bytes();
        let decoded = EndpointEntry::from_bytes(&bytes).unwrap();
        assert_eq!(decoded.id, entry.id);
        assert_eq!(decoded.name, entry.name);
        assert_eq!(decoded.kind, entry.kind);
        assert_eq!(decoded.path, entry.path);
        assert_eq!(decoded.methods, entry.methods);
        assert_eq!(decoded.sql_body, entry.sql_body);
        assert_eq!(decoded.backed_publication_id, entry.backed_publication_id);
        assert_eq!(decoded.auth_mode, entry.auth_mode);
        assert_eq!(decoded.required_scopes, entry.required_scopes);
        assert_eq!(decoded.output_format, entry.output_format);
        assert_eq!(decoded.cors_origins, entry.cors_origins);
        assert!(decoded.rate_limit.is_some());
        let rl = decoded.rate_limit.as_ref().unwrap();
        assert_eq!(rl.count, 1000);
        assert_eq!(rl.period, RateLimitPeriod::Minute);
        assert_eq!(rl.scope, RateLimitScope::PerUser);
        assert_eq!(decoded.cache_seconds, entry.cache_seconds);
        assert_eq!(decoded.timeout_seconds, entry.timeout_seconds);
        assert_eq!(decoded.max_request_body_kb, entry.max_request_body_kb);
        assert_eq!(decoded.enabled, entry.enabled);
        assert_eq!(decoded.owner_role_id, entry.owner_role_id);
        assert_eq!(decoded.created_at, entry.created_at);
    }

    #[test]
    fn test_endpoint_entry_roundtrip_streaming() {
        let entry = EndpointEntry {
            id: EndpointId(66),
            schema_id: SchemaId(2),
            name: "events_stream".to_string(),
            kind: EndpointKind::WebSocket,
            path: "/ws/events".to_string(),
            methods: vec![],
            sql_body: "SELECT * FROM events".to_string(),
            backed_publication_id: Some(PublicationId(101)),
            auth_mode: EndpointAuthMode::ApiKey,
            required_scopes: vec![],
            output_format: None,
            cors_origins: vec![],
            rate_limit: None,
            cache_seconds: None,
            timeout_seconds: None,
            max_request_body_kb: None,
            message_format: Some(EndpointMessageFormat::JsonLines),
            heartbeat_seconds: Some(30),
            backpressure: Some(BackpressurePolicy::CloseSlow),
            max_connections: Some(1024),
            enabled: false,
            owner_role_id: 8,
            created_at: 1_700_000_400,
        };
        let bytes = entry.to_bytes();
        let decoded = EndpointEntry::from_bytes(&bytes).unwrap();
        assert_eq!(decoded.kind, entry.kind);
        assert_eq!(decoded.backed_publication_id, entry.backed_publication_id);
        assert_eq!(decoded.message_format, entry.message_format);
        assert_eq!(decoded.heartbeat_seconds, entry.heartbeat_seconds);
        assert_eq!(decoded.backpressure, entry.backpressure);
        assert_eq!(decoded.max_connections, entry.max_connections);
        assert_eq!(decoded.enabled, entry.enabled);
    }

    #[test]
    fn test_security_map_entry_roundtrip() {
        let entry = SecurityMapEntry {
            id: SecurityMapId(300),
            kind: SecurityMapKind::Jwt,
            key: "https://issuer.example.com|sub-123".to_string(),
            role_id: 12,
            created_at: 1_700_000_500,
        };
        let bytes = entry.to_bytes();
        let decoded = SecurityMapEntry::from_bytes(&bytes).unwrap();
        assert_eq!(decoded.id, entry.id);
        assert_eq!(decoded.kind, entry.kind);
        assert_eq!(decoded.key, entry.key);
        assert_eq!(decoded.role_id, entry.role_id);
        assert_eq!(decoded.created_at, entry.created_at);
    }
}
