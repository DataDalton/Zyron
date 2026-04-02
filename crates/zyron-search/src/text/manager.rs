//! FTS index manager. Central coordination point that owns all live
//! full-text search index instances for a server. Handles creation,
//! deletion, lookup, and persistence lifecycle.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use zyron_common::{Result, ZyronError};

use super::analyzer::{Analyzer, StandardAnalyzer};
use super::inverted_index::InvertedIndex;
use super::scoring::{Bm25Scorer, RelevanceScorer};

/// Manages all live FTS index instances for the server.
pub struct FtsManager {
    /// Map from catalog IndexId -> live InvertedIndex.
    indexes: scc::HashMap<u32, Arc<InvertedIndex>>,
    /// Reverse map: table_id -> list of index_ids. Updated on create/drop
    /// to avoid O(n) scans of all indexes on every DML operation.
    table_indexes: scc::HashMap<u32, Vec<u32>>,
    /// Default analyzer for indexes without explicit configuration.
    default_analyzer: Arc<dyn Analyzer>,
    /// Default scorer for search queries.
    default_scorer: Arc<dyn RelevanceScorer>,
    /// Data directory for .zyfts file storage.
    data_dir: Option<PathBuf>,
}

impl FtsManager {
    /// Creates a new FTS manager with default analyzer and scorer.
    pub fn new() -> Self {
        Self {
            indexes: scc::HashMap::new(),
            table_indexes: scc::HashMap::new(),
            default_analyzer: Arc::new(StandardAnalyzer),
            default_scorer: Arc::new(Bm25Scorer::default()),
            data_dir: None,
        }
    }

    /// Creates a new FTS manager with a data directory for persistence.
    pub fn with_data_dir(data_dir: PathBuf) -> Self {
        Self {
            indexes: scc::HashMap::new(),
            table_indexes: scc::HashMap::new(),
            default_analyzer: Arc::new(StandardAnalyzer),
            default_scorer: Arc::new(Bm25Scorer::default()),
            data_dir: Some(data_dir),
        }
    }

    /// Creates a new FTS index instance and registers it.
    pub fn create_index(
        &self,
        index_id: u32,
        table_id: u32,
        column_ids: Vec<u16>,
    ) -> Result<Arc<InvertedIndex>> {
        let index = Arc::new(InvertedIndex::new(index_id, table_id, column_ids));

        match self.indexes.insert_sync(index_id, Arc::clone(&index)) {
            Ok(_) => {
                // Update the table -> indexes reverse map.
                if self
                    .table_indexes
                    .update_sync(&table_id, |_, ids| ids.push(index_id))
                    .is_none()
                {
                    let _ = self.table_indexes.insert_sync(table_id, vec![index_id]);
                }
                Ok(index)
            }
            Err(_) => Err(ZyronError::FtsIndexAlreadyExists(format!(
                "IndexId({index_id})"
            ))),
        }
    }

    /// Drops an FTS index instance and removes it from the registry.
    pub fn drop_index(&self, index_id: u32) -> Result<()> {
        // Find and remove from the index map, capturing the table_id.
        let table_id = self.indexes.read_sync(&index_id, |_, idx| idx.table_id);
        match self.indexes.remove_sync(&index_id) {
            Some(_) => {
                // Remove from the table -> indexes reverse map.
                if let Some(tid) = table_id {
                    self.table_indexes.update_sync(&tid, |_, ids| {
                        ids.retain(|&id| id != index_id);
                    });
                }
                // Remove the .zyfts file if it exists.
                if let Some(ref dir) = self.data_dir {
                    let path = fts_file_path(dir, index_id);
                    let _ = std::fs::remove_file(&path);
                }
                Ok(())
            }
            None => Err(ZyronError::FtsIndexNotFound(format!("IndexId({index_id})"))),
        }
    }

    /// Returns the live FTS index for the given catalog IndexId.
    pub fn get_index(&self, index_id: u32) -> Option<Arc<InvertedIndex>> {
        self.indexes.read_sync(&index_id, |_, idx| Arc::clone(idx))
    }

    /// Returns the default analyzer.
    pub fn default_analyzer(&self) -> Arc<dyn Analyzer> {
        Arc::clone(&self.default_analyzer)
    }

    /// Returns the default scorer.
    pub fn default_scorer(&self) -> Arc<dyn RelevanceScorer> {
        Arc::clone(&self.default_scorer)
    }

    /// Loads all FTS indexes from disk. Called during server startup.
    /// Takes catalog index entries to determine which .zyfts files to load.
    pub fn load_indexes(
        &self,
        data_dir: &Path,
        entries: &[(u32, u32, Vec<u16>)], // (index_id, table_id, column_ids)
    ) -> Result<()> {
        for (index_id, table_id, column_ids) in entries {
            let path = fts_file_path(data_dir, *index_id);
            if path.exists() {
                let index =
                    InvertedIndex::load_from_file(&path, *index_id, *table_id, column_ids.clone())?;
                let _ = self.indexes.insert_sync(*index_id, Arc::new(index));
            } else {
                // Create an empty index if the file doesn't exist yet
                let index = InvertedIndex::new(*index_id, *table_id, column_ids.clone());
                let _ = self.indexes.insert_sync(*index_id, Arc::new(index));
            }
            // Maintain table -> indexes reverse map.
            if self
                .table_indexes
                .update_sync(table_id, |_, ids| ids.push(*index_id))
                .is_none()
            {
                let _ = self.table_indexes.insert_sync(*table_id, vec![*index_id]);
            }
        }
        Ok(())
    }

    /// Saves all FTS indexes to disk. Called during graceful shutdown.
    pub fn save_all(&self, data_dir: &Path) -> Result<()> {
        let _ = std::fs::create_dir_all(data_dir);

        let mut errors = Vec::new();
        self.indexes.iter_sync(|index_id, index| {
            let path = fts_file_path(data_dir, *index_id);
            if let Err(e) = index.save_to_file(&path) {
                errors.push(format!("index_{index_id}: {e}"));
            }
            true
        });

        if errors.is_empty() {
            Ok(())
        } else {
            Err(ZyronError::FtsIndexCorrupted {
                index: "save_all".to_string(),
                reason: errors.join(", "),
            })
        }
    }

    /// Returns the number of registered FTS indexes.
    pub fn index_count(&self) -> usize {
        let mut count = 0;
        self.indexes.iter_sync(|_, _| {
            count += 1;
            true
        });
        count
    }

    /// Returns all FTS index IDs for a given table. O(1) lookup via reverse map.
    pub fn indexes_for_table(&self, table_id: u32) -> Vec<u32> {
        self.table_indexes
            .read_sync(&table_id, |_, ids| ids.clone())
            .unwrap_or_default()
    }
}

impl Default for FtsManager {
    fn default() -> Self {
        Self::new()
    }
}

/// Constructs the file path for an FTS index file.
fn fts_file_path(data_dir: &Path, index_id: u32) -> PathBuf {
    data_dir.join(format!("fts_{index_id}.zyfts"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::text::analyzer::SimpleAnalyzer;
    use crate::text::query::FtsQuery;

    #[test]
    fn test_create_and_get_index() {
        let mgr = FtsManager::new();
        let index = mgr.create_index(1, 100, vec![1, 2]).unwrap();
        assert_eq!(index.index_id, 1);
        assert_eq!(index.table_id, 100);

        let retrieved = mgr.get_index(1).unwrap();
        assert_eq!(retrieved.index_id, 1);
    }

    #[test]
    fn test_create_duplicate_error() {
        let mgr = FtsManager::new();
        mgr.create_index(1, 100, vec![1]).unwrap();
        assert!(mgr.create_index(1, 100, vec![1]).is_err());
    }

    #[test]
    fn test_drop_index() {
        let mgr = FtsManager::new();
        mgr.create_index(1, 100, vec![1]).unwrap();
        mgr.drop_index(1).unwrap();
        assert!(mgr.get_index(1).is_none());
    }

    #[test]
    fn test_drop_nonexistent_error() {
        let mgr = FtsManager::new();
        assert!(mgr.drop_index(999).is_err());
    }

    #[test]
    fn test_index_count() {
        let mgr = FtsManager::new();
        assert_eq!(mgr.index_count(), 0);

        mgr.create_index(1, 100, vec![1]).unwrap();
        mgr.create_index(2, 101, vec![2]).unwrap();
        assert_eq!(mgr.index_count(), 2);
    }

    #[test]
    fn test_indexes_for_table() {
        let mgr = FtsManager::new();
        mgr.create_index(1, 100, vec![1]).unwrap();
        mgr.create_index(2, 100, vec![2]).unwrap();
        mgr.create_index(3, 200, vec![1]).unwrap();

        let table_100 = mgr.indexes_for_table(100);
        assert_eq!(table_100.len(), 2);

        let table_200 = mgr.indexes_for_table(200);
        assert_eq!(table_200.len(), 1);
    }

    #[test]
    fn test_save_and_load() {
        let dir = std::env::temp_dir().join("zyron_fts_mgr_test");
        let _ = std::fs::create_dir_all(&dir);

        let mgr = FtsManager::new();
        let index = mgr.create_index(42, 100, vec![1, 2]).unwrap();

        // Add some data
        let analyzer = SimpleAnalyzer;
        index
            .add_document(1, "hello world search", &analyzer)
            .unwrap();
        index
            .add_document(2, "search engine optimization", &analyzer)
            .unwrap();

        // Save
        mgr.save_all(&dir).unwrap();
        assert!(dir.join("fts_42.zyfts").exists());

        // Load into a new manager
        let mgr2 = FtsManager::new();
        mgr2.load_indexes(&dir, &[(42, 100, vec![1, 2])]).unwrap();

        let loaded = mgr2.get_index(42).unwrap();
        assert_eq!(loaded.doc_count(), 2);

        let scorer = crate::text::scoring::Bm25Scorer::default();
        let results = loaded
            .search(
                &FtsQuery::Term("search".to_string()),
                &analyzer,
                &scorer,
                10,
            )
            .unwrap();
        assert_eq!(results.len(), 2);

        // Cleanup
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_load_missing_file_creates_empty() {
        let dir = std::env::temp_dir().join("zyron_fts_mgr_empty_test");
        let _ = std::fs::create_dir_all(&dir);

        let mgr = FtsManager::new();
        mgr.load_indexes(&dir, &[(99, 100, vec![1])]).unwrap();

        let index = mgr.get_index(99).unwrap();
        assert_eq!(index.doc_count(), 0);

        let _ = std::fs::remove_dir_all(&dir);
    }
}
