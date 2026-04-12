//! Vector index lifecycle management.
//!
//! VectorIndexManager is the central registry for vector indexes, providing
//! creation, lookup, deletion, persistence, and table-to-index reverse lookups.
//! Mirrors the FtsManager pattern from the text search module.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use parking_lot::Mutex;
use zyron_common::{Result, ZyronError};

use super::ann_index::AnnIndex;
use super::memory::VectorMemoryBudget;
use super::types::{DistanceMetric, HnswConfig, VectorId, VectorSearch};

/// Wrapper enum for different vector index implementations.
/// Allows the manager to store both HNSW and IVF-PQ indexes uniformly.
pub enum VectorIndex {
    Hnsw(AnnIndex),
}

impl VectorSearch for VectorIndex {
    fn search(&self, query: &[f32], k: usize, efSearch: u16) -> Result<Vec<(VectorId, f32)>> {
        match self {
            VectorIndex::Hnsw(idx) => idx.search(query, k, efSearch),
        }
    }

    fn insert(&self, id: VectorId, vector: &[f32]) -> Result<()> {
        match self {
            VectorIndex::Hnsw(idx) => idx.insert(id, vector),
        }
    }

    fn delete(&self, id: VectorId) -> Result<()> {
        match self {
            VectorIndex::Hnsw(idx) => idx.delete(id),
        }
    }

    fn dimensions(&self) -> u16 {
        match self {
            VectorIndex::Hnsw(idx) => idx.dimensions(),
        }
    }

    fn metric(&self) -> DistanceMetric {
        match self {
            VectorIndex::Hnsw(idx) => idx.metric(),
        }
    }

    fn len(&self) -> usize {
        match self {
            VectorIndex::Hnsw(idx) => idx.len(),
        }
    }
}

impl VectorIndex {
    /// Returns the column_id this index is built on. DML maintenance uses
    /// this to extract the correct vector when a table has multiple vector
    /// columns with separate indexes.
    pub fn column_id(&self) -> u16 {
        match self {
            VectorIndex::Hnsw(idx) => idx.columnId,
        }
    }
}

/// Central manager for vector indexes, providing lifecycle management
/// and table-to-index reverse lookups for DML maintenance.
///
/// Concurrent build admission is gated exclusively by the memory budget.
/// There is no fixed count cap: the number of concurrent builds scales
/// with available memory, and excess submissions park in the budget's
/// FIFO wait queue until memory frees.
pub struct VectorIndexManager {
    /// Map from index_id to live vector index instance.
    indexes: scc::HashMap<u32, Arc<VectorIndex>>,
    /// Reverse map from table_id to list of index_ids for O(1) DML lookup.
    tableIndexes: scc::HashMap<u32, Vec<u32>>,
    /// Directory for persisting vector index files.
    dataDir: Option<PathBuf>,
    /// Global memory budget for all vector indexes and queries.
    budget: Arc<VectorMemoryBudget>,
    /// Per-index memory reservations kept alive for as long as the index exists.
    per_index_reservations: Mutex<Vec<(u32, super::memory::MemoryReservation)>>,
}

impl VectorIndexManager {
    /// Creates a new manager without a persistence directory, with a default
    /// memory budget sized to 50% of available system RAM.
    pub fn new() -> Self {
        Self::with_budget(Arc::new(VectorMemoryBudget::default()))
    }

    /// Creates a new manager with a specific memory budget.
    pub fn with_budget(budget: Arc<VectorMemoryBudget>) -> Self {
        Self {
            indexes: scc::HashMap::new(),
            tableIndexes: scc::HashMap::new(),
            dataDir: None,
            budget,
            per_index_reservations: Mutex::new(Vec::new()),
        }
    }

    /// Creates a new manager with a persistence directory for .zyvec files.
    pub fn with_data_dir(dir: PathBuf) -> Self {
        Self {
            indexes: scc::HashMap::new(),
            tableIndexes: scc::HashMap::new(),
            dataDir: Some(dir),
            budget: Arc::new(VectorMemoryBudget::default()),
            per_index_reservations: Mutex::new(Vec::new()),
        }
    }

    /// Returns the shared memory budget for use by index builders.
    pub fn budget(&self) -> &Arc<VectorMemoryBudget> {
        &self.budget
    }

    /// Registers a per-index memory reservation to be held as long as the
    /// index exists. Called after successfully reserving from the budget.
    pub fn register_reservation(
        &self,
        indexId: u32,
        reservation: super::memory::MemoryReservation,
    ) {
        let mut guard = self.per_index_reservations.lock();
        guard.push((indexId, reservation));
    }

    /// Releases the per-index reservation when an index is dropped.
    fn release_reservation(&self, indexId: u32) {
        let mut guard = self.per_index_reservations.lock();
        guard.retain(|(id, _)| *id != indexId);
    }

    /// Creates a new HNSW vector index and registers it.
    pub fn create_index(
        &self,
        indexId: u32,
        tableId: u32,
        columnId: u16,
        dimensions: u16,
        config: HnswConfig,
    ) -> Result<Arc<VectorIndex>> {
        let index = AnnIndex::new(indexId, tableId, columnId, dimensions, config);
        let wrapped = Arc::new(VectorIndex::Hnsw(index));

        match self.indexes.insert_sync(indexId, Arc::clone(&wrapped)) {
            Ok(_) => {}
            Err(_) => {
                return Err(ZyronError::Internal(format!(
                    "vector index {} already exists",
                    indexId
                )));
            }
        }

        // Update reverse table map
        let tableId = match &wrapped.as_ref() {
            VectorIndex::Hnsw(idx) => idx.tableId,
        };
        if self
            .tableIndexes
            .update_sync(&tableId, |_, ids| ids.push(indexId))
            .is_none()
        {
            let _ = self.tableIndexes.insert_sync(tableId, vec![indexId]);
        }

        Ok(wrapped)
    }

    /// Drops a vector index by ID. Removes the .zyvec file if persistence is configured.
    pub fn drop_index(&self, indexId: u32) -> Result<()> {
        let tableId = self
            .indexes
            .read_sync(&indexId, |_, idx| match idx.as_ref() {
                VectorIndex::Hnsw(a) => a.tableId,
            });

        match self.indexes.remove_sync(&indexId) {
            Some(_) => {}
            None => {
                return Err(ZyronError::VectorIndexNotFound(format!(
                    "IndexId({})",
                    indexId
                )));
            }
        }

        // Update reverse table map
        if let Some(tid) = tableId {
            if let Some(mut ids) = self.tableIndexes.read_sync(&tid, |_, ids| ids.clone()) {
                ids.retain(|&id| id != indexId);
                if ids.is_empty() {
                    let _ = self.tableIndexes.remove_sync(&tid);
                } else {
                    let _ = self.tableIndexes.insert_sync(tid, ids);
                }
            }
        }

        // Remove persistence file
        if let Some(ref dir) = self.dataDir {
            let path = dir.join(format!("vec_{}.zyvec", indexId));
            if path.exists() {
                std::fs::remove_file(&path).map_err(|e| {
                    ZyronError::IoError(format!("failed to remove {}: {}", path.display(), e))
                })?;
            }
        }

        // Release any memory reservation held for this index.
        self.release_reservation(indexId);

        Ok(())
    }

    /// Returns the vector index with the given ID, if it exists.
    pub fn get_index(&self, indexId: u32) -> Option<Arc<VectorIndex>> {
        self.indexes.read_sync(&indexId, |_, idx| Arc::clone(idx))
    }

    /// Returns all index IDs for the given table. Used by DML operators
    /// to maintain vector indexes on INSERT/UPDATE/DELETE.
    pub fn indexes_for_table(&self, tableId: u32) -> Vec<u32> {
        self.tableIndexes
            .read_sync(&tableId, |_, ids| ids.clone())
            .unwrap_or_default()
    }

    /// Loads vector indexes from disk during server startup.
    /// Each entry provides (indexId, tableId, columnId, dimensions, config).
    pub fn load_indexes(
        &self,
        dataDir: &Path,
        entries: &[(u32, u32, u16, u16, HnswConfig)],
    ) -> Result<()> {
        for &(indexId, tableId, columnId, dimensions, ref config) in entries {
            let path = dataDir.join(format!("vec_{}.zyvec", indexId));
            let index = if path.exists() {
                AnnIndex::loadFromFile(&path, indexId, tableId, columnId)?
            } else {
                AnnIndex::new(indexId, tableId, columnId, dimensions, config.clone())
            };

            let wrapped = Arc::new(VectorIndex::Hnsw(index));
            let _ = self.indexes.insert_sync(indexId, wrapped);

            if self
                .tableIndexes
                .update_sync(&tableId, |_, ids| {
                    if !ids.contains(&indexId) {
                        ids.push(indexId);
                    }
                })
                .is_none()
            {
                let _ = self.tableIndexes.insert_sync(tableId, vec![indexId]);
            }
        }
        Ok(())
    }

    /// Saves all indexes to disk. Called during graceful shutdown.
    pub fn save_all(&self, dataDir: &Path) -> Result<()> {
        if !dataDir.exists() {
            std::fs::create_dir_all(dataDir).map_err(|e| {
                ZyronError::IoError(format!(
                    "failed to create vector dir {}: {}",
                    dataDir.display(),
                    e
                ))
            })?;
        }

        self.indexes.iter_sync(|&indexId, idx| {
            let path = dataDir.join(format!("vec_{}.zyvec", indexId));
            match idx.as_ref() {
                VectorIndex::Hnsw(ann) => {
                    if let Err(e) = ann.saveToFile(&path) {
                        eprintln!("failed to save vector index {}: {}", indexId, e);
                    }
                }
            }
            true
        });

        Ok(())
    }

    /// Returns the number of registered vector indexes.
    pub fn index_count(&self) -> usize {
        self.indexes.len()
    }
}

impl Default for VectorIndexManager {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_create_and_get_index() {
        let mgr = VectorIndexManager::new();
        let config = HnswConfig::default();
        let idx = mgr.create_index(1, 100, 0, 128, config).expect("create");
        assert_eq!(idx.dimensions(), 128);

        let retrieved = mgr.get_index(1);
        assert!(retrieved.is_some());
    }

    #[test]
    fn test_drop_index() {
        let mgr = VectorIndexManager::new();
        mgr.create_index(1, 100, 0, 64, HnswConfig::default())
            .expect("create");
        mgr.drop_index(1).expect("drop");
        assert!(mgr.get_index(1).is_none());
    }

    #[test]
    fn test_drop_nonexistent() {
        let mgr = VectorIndexManager::new();
        assert!(mgr.drop_index(999).is_err());
    }

    #[test]
    fn test_indexes_for_table() {
        let mgr = VectorIndexManager::new();
        let cfg = HnswConfig::default();
        mgr.create_index(1, 100, 0, 64, cfg.clone()).expect("1");
        mgr.create_index(2, 100, 1, 64, cfg.clone()).expect("2");
        mgr.create_index(3, 200, 0, 64, cfg).expect("3");

        let ids = mgr.indexes_for_table(100);
        assert_eq!(ids.len(), 2);
        assert!(ids.contains(&1));
        assert!(ids.contains(&2));

        let ids2 = mgr.indexes_for_table(200);
        assert_eq!(ids2, vec![3]);

        assert!(mgr.indexes_for_table(999).is_empty());
    }

    #[test]
    fn test_index_count() {
        let mgr = VectorIndexManager::new();
        assert_eq!(mgr.index_count(), 0);
        mgr.create_index(1, 100, 0, 64, HnswConfig::default())
            .expect("create");
        assert_eq!(mgr.index_count(), 1);
    }

    #[test]
    fn managerExposesBudgetForReservations() {
        let budget = Arc::new(VectorMemoryBudget::with_limit(10_000));
        let mgr = VectorIndexManager::with_budget(Arc::clone(&budget));

        // Reserving through the manager's budget reflects in the budget.
        let r = mgr.budget().try_reserve(1_000).expect("reserve");
        assert_eq!(budget.reserved_bytes(), 1_000);
        drop(r);
        assert_eq!(budget.reserved_bytes(), 0);
    }

    #[test]
    fn perIndexReservationReleasedOnDrop() {
        let budget = Arc::new(VectorMemoryBudget::with_limit(10_000_000));
        let mgr = VectorIndexManager::with_budget(Arc::clone(&budget));

        mgr.create_index(1, 100, 0, 64, HnswConfig::default())
            .expect("create");
        // Manually register a reservation mimicking a real build.
        let reservation = budget.try_reserve(1_000_000).expect("reserve");
        mgr.register_reservation(1, reservation);
        assert_eq!(budget.reserved_bytes(), 1_000_000);

        mgr.drop_index(1).expect("drop");
        assert_eq!(budget.reserved_bytes(), 0);
    }
}
