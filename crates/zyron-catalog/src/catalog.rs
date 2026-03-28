//! Central catalog manager for ZyronDB.
//!
//! Coordinates DDL operations with WAL logging, cache updates,
//! and storage persistence. All DDL operations are crash-safe
//! through WAL integration.

use crate::cache::CatalogCache;
use crate::ids::*;
use crate::resolver::NameResolver;
use crate::schema::*;
use crate::stats::{ColumnStats, TableStats};
use crate::storage::CatalogStorage;
use parking_lot::RwLock;
use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use zyron_common::{Result, ZyronError};
use zyron_parser::ast::{ColumnConstraint, ColumnDef, DataType, TableConstraint};
use zyron_wal::record::Lsn;
use zyron_wal::writer::WalWriter;

/// DDL operation type prefixes for WAL payloads.
const DDL_CREATE_DATABASE: u8 = 0x01;
const DDL_DROP_DATABASE: u8 = 0x02;
const DDL_CREATE_SCHEMA: u8 = 0x03;
const DDL_DROP_SCHEMA: u8 = 0x04;
const DDL_CREATE_TABLE: u8 = 0x05;
const DDL_DROP_TABLE: u8 = 0x06;
const DDL_CREATE_INDEX: u8 = 0x07;
const DDL_DROP_INDEX: u8 = 0x08;

/// Central catalog manager.
pub struct Catalog {
    storage: Arc<dyn CatalogStorage>,
    cache: Arc<CatalogCache>,
    wal: Arc<WalWriter>,
    oid_allocator: OidAllocator,
    stats: RwLock<HashMap<TableId, (TableStats, Vec<ColumnStats>)>>,
}

impl Catalog {
    /// Creates a new catalog. Bootstraps system tables on first init.
    pub async fn new(
        storage: Arc<dyn CatalogStorage>,
        cache: Arc<CatalogCache>,
        wal: Arc<WalWriter>,
    ) -> Result<Self> {
        let catalog = Self {
            storage,
            cache,
            wal,
            oid_allocator: OidAllocator::new(USER_OID_START),
            stats: RwLock::new(HashMap::new()),
        };

        if !catalog.storage.is_bootstrapped().await? {
            catalog.storage.bootstrap().await?;
        }

        catalog.load().await?;
        Ok(catalog)
    }

    /// Loads all catalog data from storage into cache and recovers OID counter.
    /// Runs all 4 storage scans concurrently to minimize cold-start latency.
    pub async fn load(&self) -> Result<()> {
        self.cache.invalidate_all();

        let (databases, schemas, tables, indexes) = tokio::try_join!(
            self.storage.load_databases(),
            self.storage.load_schemas(),
            self.storage.load_tables(),
            self.storage.load_indexes(),
        )?;

        let mut max_oid: u32 = USER_OID_START;

        for db in databases {
            if db.id.0 >= max_oid {
                max_oid = db.id.0 + 1;
            }
            self.cache.put_database(db);
        }

        for schema in schemas {
            if schema.id.0 >= max_oid {
                max_oid = schema.id.0 + 1;
            }
            self.cache.put_schema(schema);
        }

        for table in tables {
            if table.id.0 >= max_oid {
                max_oid = table.id.0 + 1;
            }
            self.cache.put_table(table);
        }

        for index in indexes {
            if index.id.0 >= max_oid {
                max_oid = index.id.0 + 1;
            }
            self.cache.put_index(index);
        }

        self.oid_allocator.reset(max_oid);
        Ok(())
    }

    /// Allocates the next OID.
    pub fn next_oid(&self) -> Oid {
        self.oid_allocator.next()
    }

    /// Creates a NameResolver bound to the given database and search path.
    pub fn resolver(&self, database_id: DatabaseId, search_path: Vec<String>) -> NameResolver {
        NameResolver::new(
            database_id,
            search_path,
            Arc::clone(&self.cache),
            Arc::clone(&self.storage),
        )
    }

    // --- Database operations ---

    pub async fn create_database(&self, name: &str, owner: &str) -> Result<DatabaseId> {
        if self.cache.get_database_by_name(name).is_some() {
            return Err(ZyronError::DatabaseAlreadyExists(name.to_string()));
        }

        let id = DatabaseId(self.oid_allocator.next());
        let now = current_timestamp();
        let entry = DatabaseEntry {
            id,
            name: name.to_string(),
            owner: owner.to_string(),
            created_at: now,
        };

        self.log_ddl(DDL_CREATE_DATABASE, &entry.to_bytes())?;
        self.storage.store_database(&entry).await?;
        self.cache.put_database(entry);
        Ok(id)
    }

    pub async fn drop_database(&self, name: &str) -> Result<()> {
        let db = self
            .cache
            .get_database_by_name(name)
            .ok_or_else(|| ZyronError::DatabaseNotFound(name.to_string()))?;

        let id = db.id;
        let mut payload = vec![0u8; 4];
        payload[..4].copy_from_slice(&id.0.to_le_bytes());
        self.log_ddl(DDL_DROP_DATABASE, &payload)?;
        self.storage.delete_database(id).await?;
        self.cache.invalidate_database(id);
        Ok(())
    }

    pub fn get_database(&self, name: &str) -> Result<Arc<DatabaseEntry>> {
        self.cache
            .get_database_by_name(name)
            .ok_or_else(|| ZyronError::DatabaseNotFound(name.to_string()))
    }

    // --- Schema operations ---

    pub async fn create_schema(
        &self,
        db_id: DatabaseId,
        name: &str,
        owner: &str,
    ) -> Result<SchemaId> {
        if self.cache.get_schema_by_name(db_id, name).is_some() {
            return Err(ZyronError::SchemaAlreadyExists(name.to_string()));
        }

        let id = SchemaId(self.oid_allocator.next());
        let entry = SchemaEntry {
            id,
            database_id: db_id,
            name: name.to_string(),
            owner: owner.to_string(),
        };

        self.log_ddl(DDL_CREATE_SCHEMA, &entry.to_bytes())?;
        self.storage.store_schema(&entry).await?;
        self.cache.put_schema(entry);
        Ok(id)
    }

    pub async fn drop_schema(&self, db_id: DatabaseId, name: &str) -> Result<()> {
        let schema = self
            .cache
            .get_schema_by_name(db_id, name)
            .ok_or_else(|| ZyronError::SchemaNotFound(name.to_string()))?;

        let id = schema.id;
        let mut payload = vec![0u8; 4];
        payload[..4].copy_from_slice(&id.0.to_le_bytes());
        self.log_ddl(DDL_DROP_SCHEMA, &payload)?;
        self.storage.delete_schema(id).await?;
        self.cache.invalidate_schema(id);
        Ok(())
    }

    pub fn get_schema(&self, db_id: DatabaseId, name: &str) -> Result<Arc<SchemaEntry>> {
        self.cache
            .get_schema_by_name(db_id, name)
            .ok_or_else(|| ZyronError::SchemaNotFound(name.to_string()))
    }

    // --- Table operations ---

    pub async fn create_table(
        &self,
        schema_id: SchemaId,
        name: &str,
        column_defs: &[ColumnDef],
        table_constraints: &[TableConstraint],
    ) -> Result<TableId> {
        if self.cache.get_table_by_name(schema_id, name).is_some() {
            return Err(ZyronError::TableAlreadyExists(name.to_string()));
        }

        if column_defs.len() > u16::MAX as usize {
            return Err(ZyronError::Internal(format!(
                "table has {} columns, max is {}",
                column_defs.len(),
                u16::MAX
            )));
        }

        // Validate no duplicate column names
        let mut seen_names = HashSet::with_capacity(column_defs.len());
        for def in column_defs {
            if !seen_names.insert(&def.name) {
                return Err(ZyronError::Internal(format!(
                    "duplicate column name: {}",
                    def.name
                )));
            }
        }

        let table_id = TableId(self.oid_allocator.next());
        let (heap_file_id, fsm_file_id) = self.storage.next_heap_file_id();
        let now = current_timestamp();

        // Convert parser ColumnDefs to catalog ColumnEntries
        let columns = convert_column_defs(table_id, column_defs)?;

        // Convert parser constraints to catalog ConstraintEntries
        let mut constraints = convert_table_constraints(table_constraints, &columns)?;

        // Extract inline column constraints (PrimaryKey, Unique, NotNull, Check, References)
        for (i, col_def) in column_defs.iter().enumerate() {
            for cc in &col_def.constraints {
                let col_id = ColumnId(i as u16);
                match cc {
                    ColumnConstraint::PrimaryKey => {
                        constraints.push(ConstraintEntry {
                            name: format!("pk_{}_{}", name, col_def.name),
                            constraint_type: ConstraintType::PrimaryKey,
                            columns: vec![col_id],
                            ref_table_id: None,
                            ref_columns: vec![],
                            check_expr: None,
                        });
                    }
                    ColumnConstraint::Unique => {
                        constraints.push(ConstraintEntry {
                            name: format!("uq_{}_{}", name, col_def.name),
                            constraint_type: ConstraintType::Unique,
                            columns: vec![col_id],
                            ref_table_id: None,
                            ref_columns: vec![],
                            check_expr: None,
                        });
                    }
                    ColumnConstraint::NotNull => {
                        constraints.push(ConstraintEntry {
                            name: format!("nn_{}_{}", name, col_def.name),
                            constraint_type: ConstraintType::NotNull,
                            columns: vec![col_id],
                            ref_table_id: None,
                            ref_columns: vec![],
                            check_expr: None,
                        });
                    }
                    ColumnConstraint::Check(expr) => {
                        constraints.push(ConstraintEntry {
                            name: format!("ck_{}_{}", name, col_def.name),
                            constraint_type: ConstraintType::Check,
                            columns: vec![col_id],
                            ref_table_id: None,
                            ref_columns: vec![],
                            check_expr: Some(format!("{:?}", expr)),
                        });
                    }
                    ColumnConstraint::References {
                        table: _,
                        column: _,
                    } => {
                        // Foreign key references need the referenced table to be resolved.
                        // Store the constraint with the reference info as a string for now.
                        // Full resolution happens at query time.
                        constraints.push(ConstraintEntry {
                            name: format!("fk_{}_{}", name, col_def.name),
                            constraint_type: ConstraintType::ForeignKey,
                            columns: vec![col_id],
                            ref_table_id: None,
                            ref_columns: vec![],
                            check_expr: None,
                        });
                    }
                    ColumnConstraint::Default(_) => {
                        // Default values are already captured in ColumnEntry.default_expr
                    }
                }
            }
        }

        let entry = TableEntry {
            id: table_id,
            schema_id,
            name: name.to_string(),
            heap_file_id,
            fsm_file_id,
            columns,
            constraints,
            created_at: now,
            versioning_enabled: false,
            scd_type: None,
            system_versioned: false,
            history_table_id: None,
            cdf_enabled: false,
            cdf_retention_days: 0,
        };

        self.log_ddl(DDL_CREATE_TABLE, &entry.to_bytes())?;
        self.storage.store_table(&entry).await?;
        self.cache.put_table(entry);
        Ok(table_id)
    }

    pub async fn drop_table(&self, schema_id: SchemaId, name: &str) -> Result<()> {
        let table = self
            .cache
            .get_table_by_name(schema_id, name)
            .ok_or_else(|| ZyronError::TableNotFound(name.to_string()))?;

        let id = table.id;
        let mut payload = vec![0u8; 4];
        payload[..4].copy_from_slice(&id.0.to_le_bytes());
        self.log_ddl(DDL_DROP_TABLE, &payload)?;
        self.storage.delete_table(id).await?;
        self.cache.invalidate_table(id);
        Ok(())
    }

    pub fn get_table(&self, schema_id: SchemaId, name: &str) -> Result<Arc<TableEntry>> {
        self.cache
            .get_table_by_name(schema_id, name)
            .ok_or_else(|| ZyronError::TableNotFound(name.to_string()))
    }

    pub fn get_table_by_id(&self, id: TableId) -> Result<Arc<TableEntry>> {
        self.cache
            .get_table(id)
            .ok_or_else(|| ZyronError::TableNotFound(format!("id={}", id.0)))
    }

    pub fn list_tables(&self, schema_id: SchemaId) -> Vec<Arc<TableEntry>> {
        self.cache.list_tables(schema_id)
    }

    /// Returns all cached tables across all schemas.
    pub fn list_all_tables(&self) -> Vec<Arc<TableEntry>> {
        self.cache.list_all_tables()
    }

    // --- Index operations ---

    pub async fn create_index(
        &self,
        table_id: TableId,
        schema_id: SchemaId,
        name: &str,
        column_names: &[String],
        unique: bool,
        index_type: IndexType,
    ) -> Result<IndexId> {
        // Check for duplicate index name in cache
        let existing = self.cache.get_indexes_for_table(table_id);
        for idx in &existing {
            if idx.name == name {
                return Err(ZyronError::IndexAlreadyExists(name.to_string()));
            }
        }

        let table = self.get_table_by_id(table_id)?;
        let index_id = IndexId(self.oid_allocator.next());
        let index_file_id = self.storage.next_index_file_id();

        // Resolve column names to ColumnIds
        let mut columns = Vec::with_capacity(column_names.len());
        for (ordinal, col_name) in column_names.iter().enumerate() {
            let col = table
                .columns
                .iter()
                .find(|c| c.name == *col_name)
                .ok_or_else(|| ZyronError::ColumnNotFound(col_name.clone()))?;
            columns.push(IndexColumnEntry {
                column_id: col.id,
                ordinal: ordinal as u16,
                descending: false,
            });
        }

        let entry = IndexEntry {
            id: index_id,
            table_id,
            schema_id,
            name: name.to_string(),
            columns,
            unique,
            index_file_id,
            index_type,
        };

        self.log_ddl(DDL_CREATE_INDEX, &entry.to_bytes())?;
        self.storage.store_index(&entry).await?;
        self.cache.put_index(entry);
        Ok(index_id)
    }

    pub async fn drop_index(&self, table_id: TableId, name: &str) -> Result<()> {
        let indexes = self.cache.get_indexes_for_table(table_id);
        let idx = indexes
            .iter()
            .find(|i| i.name == name)
            .ok_or_else(|| ZyronError::IndexNotFound(name.to_string()))?;

        let id = idx.id;
        let mut payload = vec![0u8; 4];
        payload[..4].copy_from_slice(&id.0.to_le_bytes());
        self.log_ddl(DDL_DROP_INDEX, &payload)?;
        self.storage.delete_index(id).await?;
        self.cache.invalidate_index(id);
        Ok(())
    }

    pub fn get_indexes_for_table(&self, table_id: TableId) -> Vec<Arc<IndexEntry>> {
        self.cache.get_indexes_for_table(table_id)
    }

    // --- Statistics ---

    /// Stores pre-computed statistics for a table.
    pub fn put_stats(
        &self,
        table_id: TableId,
        table_stats: TableStats,
        column_stats: Vec<ColumnStats>,
    ) {
        self.stats
            .write()
            .insert(table_id, (table_stats, column_stats));
    }

    /// Retrieves statistics for a table.
    pub fn get_stats(&self, table_id: TableId) -> Option<(TableStats, Vec<ColumnStats>)> {
        self.stats.read().get(&table_id).cloned()
    }

    // --- WAL integration ---

    /// Logs a DDL operation to the WAL as a transactional insert.
    /// Returns the commit LSN.
    fn log_ddl(&self, ddl_type: u8, entry_bytes: &[u8]) -> Result<Lsn> {
        let txn_id = self.wal.allocate_txn_id();
        let begin_lsn = self.wal.log_begin(txn_id)?;

        // Build DDL payload: 1-byte type prefix + entry bytes
        let mut payload = Vec::with_capacity(1 + entry_bytes.len());
        payload.push(ddl_type);
        payload.extend_from_slice(entry_bytes);

        let insert_lsn = self.wal.log_insert(txn_id, begin_lsn, &payload)?;
        let commit_lsn = self.wal.log_commit(txn_id, insert_lsn)?;
        Ok(commit_lsn)
    }
}

// ---------------------------------------------------------------------------
// Conversion helpers
// ---------------------------------------------------------------------------

/// Converts parser ColumnDefs to catalog ColumnEntries.
/// Column count must already be validated to fit in u16.
fn convert_column_defs(table_id: TableId, defs: &[ColumnDef]) -> Result<Vec<ColumnEntry>> {
    let mut entries = Vec::with_capacity(defs.len());
    for (i, def) in defs.iter().enumerate() {
        let type_id = def.data_type.to_type_id();
        let max_length = extract_max_length(&def.data_type);
        let nullable = def.nullable.unwrap_or(true);
        let default_expr = def.default.as_ref().map(|e| format!("{:?}", e));

        entries.push(ColumnEntry {
            id: ColumnId(i as u16),
            table_id,
            name: def.name.clone(),
            type_id,
            ordinal: i as u16,
            nullable,
            default_expr,
            max_length,
        });
    }
    Ok(entries)
}

/// Extracts the max_length parameter from sized data types.
fn extract_max_length(dt: &DataType) -> Option<usize> {
    match dt {
        DataType::Char(n)
        | DataType::Varchar(n)
        | DataType::Binary(n)
        | DataType::Varbinary(n)
        | DataType::Vector(n) => *n,
        _ => None,
    }
}

/// Converts parser TableConstraints to catalog ConstraintEntries.
fn convert_table_constraints(
    constraints: &[TableConstraint],
    columns: &[ColumnEntry],
) -> Result<Vec<ConstraintEntry>> {
    let mut result = Vec::with_capacity(constraints.len());
    for tc in constraints {
        let entry = match tc {
            TableConstraint::PrimaryKey(col_names) => ConstraintEntry {
                name: format!("pk_{}", col_names.join("_")),
                constraint_type: ConstraintType::PrimaryKey,
                columns: resolve_column_ids(col_names, columns)?,
                ref_table_id: None,
                ref_columns: vec![],
                check_expr: None,
            },
            TableConstraint::Unique(col_names) => ConstraintEntry {
                name: format!("uq_{}", col_names.join("_")),
                constraint_type: ConstraintType::Unique,
                columns: resolve_column_ids(col_names, columns)?,
                ref_table_id: None,
                ref_columns: vec![],
                check_expr: None,
            },
            TableConstraint::Check(expr) => ConstraintEntry {
                name: "ck_table".to_string(),
                constraint_type: ConstraintType::Check,
                columns: vec![],
                ref_table_id: None,
                ref_columns: vec![],
                check_expr: Some(format!("{:?}", expr)),
            },
            TableConstraint::ForeignKey {
                columns: col_names,
                ref_table: _,
                ref_columns: _,
            } => ConstraintEntry {
                name: format!("fk_{}", col_names.join("_")),
                constraint_type: ConstraintType::ForeignKey,
                columns: resolve_column_ids(col_names, columns)?,
                ref_table_id: None,
                ref_columns: vec![],
                check_expr: None,
            },
        };
        result.push(entry);
    }
    Ok(result)
}

/// Resolves column names to ColumnIds. Returns an error if any column name is not found.
fn resolve_column_ids(names: &[String], columns: &[ColumnEntry]) -> Result<Vec<ColumnId>> {
    let mut ids = Vec::with_capacity(names.len());
    for name in names {
        let col = columns
            .iter()
            .find(|c| c.name == *name)
            .ok_or_else(|| ZyronError::ColumnNotFound(name.clone()))?;
        ids.push(col.id);
    }
    Ok(ids)
}

fn current_timestamp() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}

#[cfg(test)]
mod tests {
    use super::*;
    use zyron_common::TypeId;

    #[test]
    fn test_convert_column_defs() {
        let defs = vec![
            ColumnDef {
                name: "id".to_string(),
                data_type: DataType::BigInt,
                nullable: Some(false),
                default: None,
                constraints: vec![ColumnConstraint::PrimaryKey],
            },
            ColumnDef {
                name: "email".to_string(),
                data_type: DataType::Varchar(Some(255)),
                nullable: None,
                default: None,
                constraints: vec![],
            },
        ];

        let cols = convert_column_defs(TableId(1), &defs).unwrap();
        assert_eq!(cols.len(), 2);
        assert_eq!(cols[0].name, "id");
        assert_eq!(cols[0].type_id, TypeId::Int64);
        assert_eq!(cols[0].nullable, false);
        assert_eq!(cols[0].ordinal, 0);
        assert_eq!(cols[1].name, "email");
        assert_eq!(cols[1].type_id, TypeId::Varchar);
        assert_eq!(cols[1].nullable, true);
        assert_eq!(cols[1].max_length, Some(255));
    }

    #[test]
    fn test_extract_max_length() {
        assert_eq!(extract_max_length(&DataType::Varchar(Some(100))), Some(100));
        assert_eq!(extract_max_length(&DataType::Char(None)), None);
        assert_eq!(extract_max_length(&DataType::Vector(Some(128))), Some(128));
        assert_eq!(extract_max_length(&DataType::Int), None);
        assert_eq!(extract_max_length(&DataType::Text), None);
    }

    #[test]
    fn test_convert_table_constraints() {
        let cols = vec![
            ColumnEntry {
                id: ColumnId(0),
                table_id: TableId(1),
                name: "a".to_string(),
                type_id: TypeId::Int32,
                ordinal: 0,
                nullable: false,
                default_expr: None,
                max_length: None,
            },
            ColumnEntry {
                id: ColumnId(1),
                table_id: TableId(1),
                name: "b".to_string(),
                type_id: TypeId::Int32,
                ordinal: 1,
                nullable: false,
                default_expr: None,
                max_length: None,
            },
        ];
        let tcs = vec![
            TableConstraint::PrimaryKey(vec!["a".to_string()]),
            TableConstraint::Unique(vec!["a".to_string(), "b".to_string()]),
        ];
        let result = convert_table_constraints(&tcs, &cols).unwrap();
        assert_eq!(result.len(), 2);
        assert_eq!(result[0].constraint_type, ConstraintType::PrimaryKey);
        assert_eq!(result[0].columns, vec![ColumnId(0)]);
        assert_eq!(result[1].constraint_type, ConstraintType::Unique);
        assert_eq!(result[1].columns, vec![ColumnId(0), ColumnId(1)]);
    }
}
