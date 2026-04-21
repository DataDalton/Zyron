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
}

impl ExternalBackend {
    pub fn from_u8(val: u8) -> Result<Self> {
        match val {
            0 => Ok(ExternalBackend::File),
            1 => Ok(ExternalBackend::S3),
            2 => Ok(ExternalBackend::Gcs),
            3 => Ok(ExternalBackend::Azure),
            4 => Ok(ExternalBackend::Http),
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
}
