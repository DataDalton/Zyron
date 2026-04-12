//! Graph schema lifecycle management.
//!
//! GraphManager is the central registry for graph schemas, providing
//! creation, lookup, deletion, and CSR cache management. Mirrors the
//! FtsManager pattern from the text search module.

use std::sync::Arc;

use zyron_common::{Result, ZyronError};

use super::schema::{GraphSchema, NodeId};
use super::storage::{CompactGraph, CsrCache};

/// Central manager for graph schemas and their cached CSR representations.
pub struct GraphManager {
    /// Map from graph schema name to schema metadata.
    schemas: scc::HashMap<String, Arc<GraphSchema>>,
    /// CSR cache for algorithm execution.
    csr_cache: CsrCache,
}

impl GraphManager {
    /// Creates a new empty GraphManager.
    pub fn new() -> Self {
        Self {
            schemas: scc::HashMap::new(),
            csr_cache: CsrCache::new(),
        }
    }

    /// Registers a graph schema. Returns an error if a schema with the
    /// same name already exists.
    pub fn create_schema(&self, schema: GraphSchema) -> Result<Arc<GraphSchema>> {
        let name = schema.name.clone();
        let arc = Arc::new(schema);
        match self.schemas.insert_sync(name, arc.clone()) {
            Ok(_) => Ok(arc),
            Err(_) => Err(ZyronError::GraphSchemaAlreadyExists(arc.name.clone())),
        }
    }

    /// Removes a graph schema by name. Also invalidates any cached CSR.
    /// Returns an error if the schema does not exist.
    pub fn drop_schema(&self, name: &str) -> Result<()> {
        if self.schemas.remove_sync(name).is_none() {
            return Err(ZyronError::GraphSchemaNotFound(name.to_string()));
        }
        self.csr_cache.invalidate(name);
        Ok(())
    }

    /// Returns the graph schema with the given name, if it exists.
    pub fn get_schema(&self, name: &str) -> Option<Arc<GraphSchema>> {
        self.schemas.read_sync(name, |_, v| v.clone())
    }

    /// Returns all registered graph schema names.
    pub fn schema_names(&self) -> Vec<String> {
        let mut names = Vec::new();
        self.schemas.iter_sync(|k, _| {
            names.push(k.clone());
            true
        });
        names
    }

    /// Builds a CSR representation from edge data and caches it.
    /// If a cached CSR already exists for this schema, returns it.
    pub fn get_or_build_csr(
        &self,
        schema_name: &str,
        edge_data: &[(NodeId, NodeId, Option<f64>)],
    ) -> Result<Arc<CompactGraph>> {
        // Check cache first
        if let Some(cached) = self.csr_cache.get(schema_name) {
            return Ok(cached);
        }

        // Verify schema exists
        if self.get_schema(schema_name).is_none() {
            return Err(ZyronError::GraphSchemaNotFound(schema_name.to_string()));
        }

        // Build and cache
        let graph = Arc::new(CompactGraph::build(edge_data));
        self.csr_cache
            .insert(schema_name.to_string(), graph.clone());
        Ok(graph)
    }

    /// Stores a pre-built CSR in the cache.
    pub fn cache_csr(&self, schema_name: &str, graph: Arc<CompactGraph>) {
        self.csr_cache.insert(schema_name.to_string(), graph);
    }

    /// Returns the cached CSR for the given schema, if available.
    pub fn get_cached_csr(&self, schema_name: &str) -> Option<Arc<CompactGraph>> {
        self.csr_cache.get(schema_name)
    }

    /// Invalidates the cached CSR for a schema. Called when DML operations
    /// modify the backing node or edge tables.
    pub fn invalidate_csr(&self, schema_name: &str) {
        self.csr_cache.invalidate(schema_name);
    }

    /// Populates the in-memory registry from a list of serialized schemas.
    pub fn load_schemas(&self, schemas: Vec<GraphSchema>) -> Result<()> {
        for schema in schemas {
            let name = schema.name.clone();
            let arc = Arc::new(schema);
            let _ = self.schemas.insert_sync(name, arc);
        }
        Ok(())
    }

    /// Returns the number of registered graph schemas.
    pub fn schema_count(&self) -> usize {
        self.schemas.len()
    }

    /// Persists all graph schemas to individual files in the given directory.
    /// Each schema is written as `<name>.zygraph` using GraphSchema::to_bytes.
    pub fn save_all(&self, dir: &std::path::Path) -> Result<()> {
        std::fs::create_dir_all(dir).map_err(|e| {
            ZyronError::GraphSchemaNotFound(format!("failed to create graph dir {:?}: {e}", dir))
        })?;

        // Remove stale files for schemas that no longer exist. Re-check the
        // schema presence right before remove to narrow the TOCTOU window
        // with concurrent DDL.
        let entries = std::fs::read_dir(dir)
            .map_err(|e| ZyronError::GraphSchemaNotFound(format!("read_dir {:?}: {e}", dir)))?;
        for entry in entries.flatten() {
            let path = entry.path();
            if path.extension().map(|e| e == "zygraph").unwrap_or(false) {
                let stem = path.file_stem().and_then(|s| s.to_str()).unwrap_or("");
                if self.get_schema(stem).is_none() {
                    let _ = std::fs::remove_file(&path);
                }
            }
        }

        // Write each schema. Collect the first failure to return after the
        // loop so every schema still gets a write attempt.
        let mut first_err: Option<ZyronError> = None;
        self.schemas.iter_sync(|name, schema| {
            let path = dir.join(format!("{}.zygraph", name));
            let bytes = schema.to_bytes();
            if let Err(e) = std::fs::write(&path, &bytes) {
                if first_err.is_none() {
                    first_err = Some(ZyronError::GraphSchemaNotFound(format!(
                        "write {:?}: {e}",
                        path
                    )));
                }
            }
            true
        });
        match first_err {
            Some(e) => Err(e),
            None => Ok(()),
        }
    }

    /// Loads all graph schemas from .zygraph files in the given directory.
    /// A missing directory is not an error (first startup). Corrupt or
    /// unreadable files abort the load with a descriptive error.
    pub fn load_all(&self, dir: &std::path::Path) -> Result<()> {
        let entries = match std::fs::read_dir(dir) {
            Ok(e) => e,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(()),
            Err(e) => {
                return Err(ZyronError::GraphSchemaNotFound(format!(
                    "read_dir {:?}: {e}",
                    dir
                )));
            }
        };
        for entry in entries.flatten() {
            let path = entry.path();
            if path.extension().map(|e| e == "zygraph").unwrap_or(false) {
                let bytes = std::fs::read(&path).map_err(|e| {
                    ZyronError::GraphSchemaNotFound(format!("read {:?}: {e}", path))
                })?;
                let schema = GraphSchema::from_bytes(&bytes).map_err(|e| {
                    ZyronError::GraphSchemaNotFound(format!("parse {:?}: {e}", path))
                })?;
                let name = schema.name.clone();
                let _ = self.schemas.insert_sync(name, Arc::new(schema));
            }
        }
        Ok(())
    }
}

impl Default for GraphManager {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_schema(name: &str) -> GraphSchema {
        GraphSchema::new(name.to_string(), 1)
    }

    #[test]
    fn test_create_and_get_schema() {
        let mgr = GraphManager::new();
        let schema = make_schema("social");
        mgr.create_schema(schema).expect("create");

        let retrieved = mgr.get_schema("social");
        assert!(retrieved.is_some());
        assert_eq!(retrieved.expect("schema").name, "social");
    }

    #[test]
    fn test_duplicate_schema_error() {
        let mgr = GraphManager::new();
        mgr.create_schema(make_schema("social")).expect("first");
        let err = mgr.create_schema(make_schema("social"));
        assert!(err.is_err());
    }

    #[test]
    fn test_drop_schema() {
        let mgr = GraphManager::new();
        mgr.create_schema(make_schema("social")).expect("create");
        mgr.drop_schema("social").expect("drop");
        assert!(mgr.get_schema("social").is_none());
    }

    #[test]
    fn test_drop_nonexistent_error() {
        let mgr = GraphManager::new();
        let err = mgr.drop_schema("nonexistent");
        assert!(err.is_err());
    }

    #[test]
    fn test_csr_caching() {
        let mgr = GraphManager::new();
        mgr.create_schema(make_schema("graph1")).expect("create");

        let edges = vec![(1u64, 2u64, None), (2, 3, None)];
        let csr = mgr.get_or_build_csr("graph1", &edges).expect("build");
        assert_eq!(csr.node_count, 3);

        // Second call should return cached version
        let cached = mgr.get_or_build_csr("graph1", &edges).expect("cached");
        assert_eq!(Arc::as_ptr(&csr), Arc::as_ptr(&cached));
    }

    #[test]
    fn test_invalidate_csr() {
        let mgr = GraphManager::new();
        mgr.create_schema(make_schema("graph1")).expect("create");

        let edges = vec![(1u64, 2u64, None)];
        mgr.get_or_build_csr("graph1", &edges).expect("build");

        mgr.invalidate_csr("graph1");
        assert!(mgr.get_cached_csr("graph1").is_none());
    }

    #[test]
    fn test_schema_names() {
        let mgr = GraphManager::new();
        mgr.create_schema(make_schema("alpha")).expect("a");
        mgr.create_schema(make_schema("beta")).expect("b");

        let mut names = mgr.schema_names();
        names.sort();
        assert_eq!(names, vec!["alpha", "beta"]);
    }

    #[test]
    fn test_load_schemas() {
        let mgr = GraphManager::new();
        let schemas = vec![make_schema("g1"), make_schema("g2")];
        mgr.load_schemas(schemas).expect("load");
        assert_eq!(mgr.schema_count(), 2);
    }
}
