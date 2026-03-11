//! Name resolution for SQL identifiers.
//!
//! Resolves unqualified table and column names using a configurable search path,
//! with cache-first lookups and storage fallback.

use crate::cache::CatalogCache;
use crate::ids::*;
use crate::schema::*;
use crate::storage::CatalogStorage;
use std::sync::Arc;
use zyron_common::{Result, ZyronError};

/// Resolves unqualified SQL names using a search path.
pub struct NameResolver {
    database_id: DatabaseId,
    search_path: Vec<String>,
    cache: Arc<CatalogCache>,
    storage: Arc<dyn CatalogStorage>,
}

impl NameResolver {
    /// Creates a new resolver with the given search path.
    pub fn new(
        database_id: DatabaseId,
        search_path: Vec<String>,
        cache: Arc<CatalogCache>,
        storage: Arc<dyn CatalogStorage>,
    ) -> Self {
        Self {
            database_id,
            search_path,
            cache,
            storage,
        }
    }

    /// Resolves a possibly-qualified table name to a TableEntry.
    ///
    /// If schema_name is Some, looks up that schema directly.
    /// If None, iterates the search path and returns the first match.
    /// Cache-hit fast path is fully synchronous (no nested async futures).
    /// Only falls through to async storage lookups on cache miss.
    pub async fn resolve_table(
        &self,
        schema_name: Option<&str>,
        table_name: &str,
    ) -> Result<Arc<TableEntry>> {
        // Fast path: pure cache lookups with no async calls.
        // Avoids creating nested futures for the common case where
        // schemas and tables are already cached.
        if let Some(schema) = schema_name {
            if let Some(schema_entry) = self.cache.get_schema_by_name(self.database_id, schema) {
                if let Some(table) = self.cache.get_table_by_name(schema_entry.id, table_name) {
                    return Ok(table);
                }
                // Schema cached but table not, fall through to storage
                return self.find_table_in_schema(schema_entry.id, table_name).await;
            }
            // Schema not cached, full async fallback
            let schema_entry = self.resolve_schema(schema).await?;
            return self.find_table_in_schema(schema_entry.id, table_name).await;
        }

        for path_schema in &self.search_path {
            if let Some(schema_entry) = self.cache.get_schema_by_name(self.database_id, path_schema) {
                if let Some(table) = self.cache.get_table_by_name(schema_entry.id, table_name) {
                    return Ok(table);
                }
            }
        }

        // Cache miss on all search path entries, try async storage fallback
        for path_schema in &self.search_path {
            if let Ok(schema_entry) = self.resolve_schema(path_schema).await {
                if let Ok(table) = self.find_table_in_schema(schema_entry.id, table_name).await {
                    return Ok(table);
                }
            }
        }

        Err(ZyronError::TableNotFound(table_name.to_string()))
    }

    /// Resolves a column name within a table.
    pub fn resolve_column<'a>(
        &self,
        table: &'a TableEntry,
        column_name: &str,
    ) -> Result<&'a ColumnEntry> {
        table
            .columns
            .iter()
            .find(|c| c.name == column_name)
            .ok_or_else(|| ZyronError::ColumnNotFound(column_name.to_string()))
    }

    /// Resolves a schema name to a SchemaEntry.
    pub async fn resolve_schema(&self, name: &str) -> Result<Arc<SchemaEntry>> {
        // Check cache first
        if let Some(entry) = self.cache.get_schema_by_name(self.database_id, name) {
            return Ok(entry);
        }

        // Fall back to storage
        let schemas = self.storage.load_schemas().await?;
        for schema in schemas {
            if schema.database_id == self.database_id && schema.name == name {
                self.cache.put_schema(schema.clone());
                return Ok(Arc::new(schema));
            }
        }

        Err(ZyronError::SchemaNotFound(name.to_string()))
    }

    /// Sets the search path.
    pub fn set_search_path(&mut self, path: Vec<String>) {
        self.search_path = path;
    }

    /// Returns the current search path.
    pub fn search_path(&self) -> &[String] {
        &self.search_path
    }

    /// Returns the database ID this resolver is bound to.
    pub fn database_id(&self) -> DatabaseId {
        self.database_id
    }

    /// Finds a table by name within a specific schema.
    async fn find_table_in_schema(
        &self,
        schema_id: SchemaId,
        table_name: &str,
    ) -> Result<Arc<TableEntry>> {
        // Check cache first
        if let Some(entry) = self.cache.get_table_by_name(schema_id, table_name) {
            return Ok(entry);
        }

        // Fall back to storage
        let tables = self.storage.load_tables().await?;
        for table in tables {
            if table.schema_id == schema_id && table.name == table_name {
                self.cache.put_table(table.clone());
                return Ok(Arc::new(table));
            }
        }

        Err(ZyronError::TableNotFound(table_name.to_string()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::storage::CatalogStorage;
    use async_trait::async_trait;
    use zyron_common::TypeId;
    use zyron_storage::TupleId;

    /// Mock storage for resolver tests.
    struct MockStorage {
        schemas: Vec<SchemaEntry>,
        tables: Vec<TableEntry>,
    }

    #[async_trait]
    impl CatalogStorage for MockStorage {
        async fn load_databases(&self) -> Result<Vec<DatabaseEntry>> {
            Ok(vec![])
        }
        async fn store_database(&self, _: &DatabaseEntry) -> Result<TupleId> {
            unimplemented!()
        }
        async fn delete_database(&self, _: DatabaseId) -> Result<bool> {
            unimplemented!()
        }
        async fn load_schemas(&self) -> Result<Vec<SchemaEntry>> {
            Ok(self.schemas.clone())
        }
        async fn store_schema(&self, _: &SchemaEntry) -> Result<TupleId> {
            unimplemented!()
        }
        async fn delete_schema(&self, _: SchemaId) -> Result<bool> {
            unimplemented!()
        }
        async fn load_tables(&self) -> Result<Vec<TableEntry>> {
            Ok(self.tables.clone())
        }
        async fn store_table(&self, _: &TableEntry) -> Result<TupleId> {
            unimplemented!()
        }
        async fn delete_table(&self, _: TableId) -> Result<bool> {
            unimplemented!()
        }
        async fn load_columns(&self, _: TableId) -> Result<Vec<ColumnEntry>> {
            Ok(vec![])
        }
        async fn store_columns(&self, _: &[ColumnEntry]) -> Result<Vec<TupleId>> {
            Ok(vec![])
        }
        async fn delete_columns(&self, _: TableId) -> Result<usize> {
            Ok(0)
        }
        async fn load_indexes(&self) -> Result<Vec<IndexEntry>> {
            Ok(vec![])
        }
        async fn store_index(&self, _: &IndexEntry) -> Result<TupleId> {
            unimplemented!()
        }
        async fn delete_index(&self, _: IndexId) -> Result<bool> {
            unimplemented!()
        }
        async fn is_bootstrapped(&self) -> Result<bool> {
            Ok(true)
        }
        async fn bootstrap(&self) -> Result<()> {
            Ok(())
        }
        fn next_heap_file_id(&self) -> (u32, u32) {
            (200, 201)
        }
        fn next_index_file_id(&self) -> u32 {
            10000
        }
    }

    fn make_mock() -> (Arc<CatalogCache>, Arc<MockStorage>) {
        let cache = Arc::new(CatalogCache::new(100, 100));
        let storage = Arc::new(MockStorage {
            schemas: vec![
                SchemaEntry {
                    id: SchemaId(1),
                    database_id: DatabaseId(1),
                    name: "public".to_string(),
                    owner: "system".to_string(),
                },
                SchemaEntry {
                    id: SchemaId(2),
                    database_id: DatabaseId(1),
                    name: "analytics".to_string(),
                    owner: "system".to_string(),
                },
            ],
            tables: vec![
                TableEntry {
                    id: TableId(10),
                    schema_id: SchemaId(1),
                    name: "users".to_string(),
                    heap_file_id: 200,
                    fsm_file_id: 201,
                    columns: vec![
                        ColumnEntry {
                            id: ColumnId(0),
                            table_id: TableId(10),
                            name: "id".to_string(),
                            type_id: TypeId::Int64,
                            ordinal: 0,
                            nullable: false,
                            default_expr: None,
                            max_length: None,
                        },
                        ColumnEntry {
                            id: ColumnId(1),
                            table_id: TableId(10),
                            name: "email".to_string(),
                            type_id: TypeId::Varchar,
                            ordinal: 1,
                            nullable: true,
                            default_expr: None,
                            max_length: Some(255),
                        },
                    ],
                    constraints: vec![],
                    created_at: 0,
                },
                TableEntry {
                    id: TableId(20),
                    schema_id: SchemaId(2),
                    name: "events".to_string(),
                    heap_file_id: 202,
                    fsm_file_id: 203,
                    columns: vec![],
                    constraints: vec![],
                    created_at: 0,
                },
            ],
        });
        (cache, storage)
    }

    #[tokio::test]
    async fn test_resolve_unqualified_table() {
        let (cache, storage) = make_mock();
        let resolver = NameResolver::new(DatabaseId(1), vec!["public".to_string()], cache, storage);

        let table = resolver.resolve_table(None, "users").await.unwrap();
        assert_eq!(table.id, TableId(10));
        assert_eq!(table.name, "users");
    }

    #[tokio::test]
    async fn test_resolve_qualified_table() {
        let (cache, storage) = make_mock();
        let resolver = NameResolver::new(DatabaseId(1), vec!["public".to_string()], cache, storage);

        let table = resolver
            .resolve_table(Some("analytics"), "events")
            .await
            .unwrap();
        assert_eq!(table.id, TableId(20));
    }

    #[tokio::test]
    async fn test_resolve_table_not_found() {
        let (cache, storage) = make_mock();
        let resolver = NameResolver::new(DatabaseId(1), vec!["public".to_string()], cache, storage);

        let result = resolver.resolve_table(None, "nonexistent").await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_resolve_column() {
        let (cache, storage) = make_mock();
        let resolver = NameResolver::new(DatabaseId(1), vec!["public".to_string()], cache, storage);

        let table = resolver.resolve_table(None, "users").await.unwrap();
        let col = resolver.resolve_column(&table, "email").unwrap();
        assert_eq!(col.type_id, TypeId::Varchar);
        assert_eq!(col.max_length, Some(255));

        let result = resolver.resolve_column(&table, "nonexistent");
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_search_path_order() {
        let (cache, storage) = make_mock();
        // analytics first in search path, events is only in analytics
        let resolver = NameResolver::new(
            DatabaseId(1),
            vec!["analytics".to_string(), "public".to_string()],
            cache,
            storage,
        );

        let table = resolver.resolve_table(None, "events").await.unwrap();
        assert_eq!(table.schema_id, SchemaId(2));

        // users is only in public, should still be found
        let table = resolver.resolve_table(None, "users").await.unwrap();
        assert_eq!(table.schema_id, SchemaId(1));
    }

    #[tokio::test]
    async fn test_resolve_schema_not_found() {
        let (cache, storage) = make_mock();
        let resolver = NameResolver::new(DatabaseId(1), vec!["public".to_string()], cache, storage);

        let result = resolver.resolve_schema("nonexistent").await;
        assert!(result.is_err());
    }

    #[test]
    fn test_search_path_get_set() {
        let cache = Arc::new(CatalogCache::new(10, 10));
        let storage: Arc<dyn CatalogStorage> = Arc::new(MockStorage {
            schemas: vec![],
            tables: vec![],
        });
        let mut resolver =
            NameResolver::new(DatabaseId(1), vec!["public".to_string()], cache, storage);

        assert_eq!(resolver.search_path(), &["public"]);
        resolver.set_search_path(vec!["myschema".to_string(), "public".to_string()]);
        assert_eq!(resolver.search_path(), &["myschema", "public"]);
    }
}
