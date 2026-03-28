//! Inbound CDC ingestion from external sources (Kafka, S3).
//!
//! Consumes change events from upstream systems and applies them to local
//! tables transactionally. Failed records are routed to a dead letter queue.

use std::fs::{self, File};
use std::io::{Read, Write};
use std::path::{Path, PathBuf};

use scc::HashMap as SccHashMap;
use serde::{Deserialize, Serialize};
use zyron_common::{Result, ZyronError};

use crate::cdc_stream::OutputFormat;

// ---------------------------------------------------------------------------
// CdcIngestSource
// ---------------------------------------------------------------------------

/// Source configuration for inbound CDC ingestion.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CdcIngestSource {
    Kafka {
        brokers: String,
        topic: String,
        group_id: String,
        start_offset: Option<String>,
    },
    S3 {
        bucket: String,
        prefix: String,
        region: String,
        format: OutputFormat,
    },
}

// ---------------------------------------------------------------------------
// OnConflict
// ---------------------------------------------------------------------------

/// Conflict resolution strategy for ingested records.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum OnConflict {
    /// Skip conflicting records silently.
    Skip,
    /// Update existing rows with incoming values.
    Update,
    /// Return an error on conflict.
    Error,
}

// ---------------------------------------------------------------------------
// CdcIngestConfig
// ---------------------------------------------------------------------------

/// Configuration for an inbound CDC ingestion job.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CdcIngestConfig {
    pub name: String,
    pub source: CdcIngestSource,
    pub target_table_id: u32,
    pub primary_key_columns: Vec<String>,
    pub on_conflict: OnConflict,
    pub dead_letter_table_id: Option<u32>,
    pub batch_size: usize,
    pub active: bool,
}

// ---------------------------------------------------------------------------
// IngestCheckpoint
// ---------------------------------------------------------------------------

/// Tracks ingestion progress for resumption after restart.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IngestCheckpoint {
    pub name: String,
    pub last_source_offset: String,
    pub records_applied: u64,
    pub records_failed: u64,
}

// ---------------------------------------------------------------------------
// IngestStatus
// ---------------------------------------------------------------------------

/// Runtime status of a CDC ingest job.
#[derive(Debug, Clone)]
pub struct IngestStatus {
    pub name: String,
    pub active: bool,
    pub records_applied: u64,
    pub records_failed: u64,
    pub dead_letter_count: u64,
    pub last_offset: String,
}

// ---------------------------------------------------------------------------
// CdcIngestManager
// ---------------------------------------------------------------------------

/// Manages inbound CDC ingestion jobs.
pub struct CdcIngestManager {
    configs: SccHashMap<String, CdcIngestConfig>,
    checkpoints: SccHashMap<String, IngestCheckpoint>,
    state_file: PathBuf,
}

impl CdcIngestManager {
    /// Opens or creates the ingest manager, loading persisted state.
    pub fn new(data_dir: &Path) -> Result<Self> {
        let state_file = data_dir.join(".zyingests");

        let configs = SccHashMap::new();
        let checkpoints = SccHashMap::new();

        if state_file.exists() {
            let mut file = File::open(&state_file)?;
            let mut data = Vec::new();
            file.read_to_end(&mut data)?;
            if !data.is_empty() {
                let list: Vec<CdcIngestConfig> = serde_json::from_slice(&data).map_err(|e| {
                    ZyronError::CdcIngestError(format!("failed to parse ingest state: {e}"))
                })?;
                for config in list {
                    let _ = configs.insert_sync(config.name.clone(), config);
                }
            }
        }

        // Load checkpoints if available.
        let cp_file = data_dir.join(".zyingest_checkpoints");
        if cp_file.exists() {
            let mut file = File::open(&cp_file)?;
            let mut data = Vec::new();
            file.read_to_end(&mut data)?;
            if !data.is_empty() {
                let list: Vec<IngestCheckpoint> = serde_json::from_slice(&data).map_err(|e| {
                    ZyronError::CdcIngestError(format!("failed to parse ingest checkpoints: {e}"))
                })?;
                for cp in list {
                    let _ = checkpoints.insert_sync(cp.name.clone(), cp);
                }
            }
        }

        Ok(Self {
            configs,
            checkpoints,
            state_file,
        })
    }

    /// Creates a new inbound CDC ingestion job.
    pub fn create_ingest(&self, config: CdcIngestConfig) -> Result<()> {
        if self
            .configs
            .insert_sync(config.name.clone(), config)
            .is_err()
        {
            return Err(ZyronError::CdcIngestError("ingest already exists".into()));
        }
        self.persist()?;
        Ok(())
    }

    /// Drops an inbound CDC ingestion job.
    pub fn drop_ingest(&self, name: &str) -> Result<()> {
        self.configs
            .remove_sync(name)
            .ok_or_else(|| ZyronError::CdcIngestError(format!("ingest not found: {name}")))?;
        let _ = self.checkpoints.remove_sync(name);
        self.persist()?;
        Ok(())
    }

    /// Lists all inbound CDC ingestion jobs.
    pub fn list_ingests(&self) -> Vec<CdcIngestConfig> {
        let mut result = Vec::new();
        self.configs.iter_sync(|_name, config| {
            result.push(config.clone());
            true
        });
        result
    }

    /// Gets an ingest job by name.
    pub fn get_ingest(&self, name: &str) -> Result<CdcIngestConfig> {
        self.configs
            .read_sync(name, |_, config| config.clone())
            .ok_or_else(|| ZyronError::CdcIngestError(format!("ingest not found: {name}")))
    }

    /// Gets the checkpoint for an ingest job.
    pub fn get_checkpoint(&self, name: &str) -> Option<IngestCheckpoint> {
        self.checkpoints.read_sync(name, |_, cp| cp.clone())
    }

    /// Updates the checkpoint for an ingest job.
    pub fn update_checkpoint(&self, checkpoint: IngestCheckpoint) -> Result<()> {
        match self.checkpoints.entry_sync(checkpoint.name.clone()) {
            scc::hash_map::Entry::Occupied(mut o) => {
                *o.get_mut() = checkpoint;
            }
            scc::hash_map::Entry::Vacant(v) => {
                v.insert_entry(checkpoint);
            }
        }
        self.persist_checkpoints()?;
        Ok(())
    }

    /// Returns the status of an ingest job.
    pub fn get_ingest_status(&self, name: &str) -> Result<IngestStatus> {
        let config = self.get_ingest(name)?;
        let cp = self.get_checkpoint(name);

        Ok(IngestStatus {
            name: name.to_string(),
            active: config.active,
            records_applied: cp.as_ref().map(|c| c.records_applied).unwrap_or(0),
            records_failed: cp.as_ref().map(|c| c.records_failed).unwrap_or(0),
            dead_letter_count: 0,
            last_offset: cp
                .as_ref()
                .map(|c| c.last_source_offset.clone())
                .unwrap_or_default(),
        })
    }

    /// Removes all ingest jobs targeting the given table_id.
    pub fn remove_ingests_for_table(&self, table_id: u32) -> Result<Vec<String>> {
        let mut to_remove = Vec::new();
        self.configs.iter_sync(|name, config| {
            if config.target_table_id == table_id {
                to_remove.push(name.clone());
            }
            true
        });

        for name in &to_remove {
            let _ = self.configs.remove_sync(name);
            let _ = self.checkpoints.remove_sync(name);
        }

        if !to_remove.is_empty() {
            self.persist()?;
        }

        Ok(to_remove)
    }

    /// Persists ingest configurations to disk using atomic rename.
    fn persist(&self) -> Result<()> {
        let configs = self.list_ingests();
        let data = serde_json::to_vec(&configs).map_err(|e| {
            ZyronError::CdcIngestError(format!("failed to serialize ingest state: {e}"))
        })?;

        let tmp_path = self.state_file.with_extension("zyingests.tmp");
        {
            let mut tmp = File::create(&tmp_path)?;
            tmp.write_all(&data)?;
            tmp.sync_all()?;
        }

        fs::rename(&tmp_path, &self.state_file)?;
        Ok(())
    }

    /// Persists ingest checkpoints to disk.
    fn persist_checkpoints(&self) -> Result<()> {
        let mut cps = Vec::new();
        self.checkpoints.iter_sync(|_name, cp| {
            cps.push(cp.clone());
            true
        });

        let data = serde_json::to_vec(&cps).map_err(|e| {
            ZyronError::CdcIngestError(format!("failed to serialize ingest checkpoints: {e}"))
        })?;

        let cp_file = self
            .state_file
            .parent()
            .map(|p| p.join(".zyingest_checkpoints"));
        if let Some(cp_path) = cp_file {
            let tmp_path = cp_path.with_extension("tmp");
            {
                let mut tmp = File::create(&tmp_path)?;
                tmp.write_all(&data)?;
                tmp.sync_all()?;
            }
            fs::rename(&tmp_path, &cp_path)?;
        }

        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    fn sample_config() -> CdcIngestConfig {
        CdcIngestConfig {
            name: "test_ingest".into(),
            source: CdcIngestSource::Kafka {
                brokers: "localhost:9092".into(),
                topic: "upstream_cdc".into(),
                group_id: "zyrondb_ingest".into(),
                start_offset: None,
            },
            target_table_id: 42,
            primary_key_columns: vec!["id".into()],
            on_conflict: OnConflict::Update,
            dead_letter_table_id: None,
            batch_size: 1000,
            active: true,
        }
    }

    #[test]
    fn test_create_and_list_ingests() {
        let tmp = TempDir::new().unwrap();
        let mgr = CdcIngestManager::new(tmp.path()).unwrap();

        mgr.create_ingest(sample_config()).unwrap();
        let ingests = mgr.list_ingests();
        assert_eq!(ingests.len(), 1);
        assert_eq!(ingests[0].name, "test_ingest");
    }

    #[test]
    fn test_create_duplicate_fails() {
        let tmp = TempDir::new().unwrap();
        let mgr = CdcIngestManager::new(tmp.path()).unwrap();

        mgr.create_ingest(sample_config()).unwrap();
        assert!(mgr.create_ingest(sample_config()).is_err());
    }

    #[test]
    fn test_drop_ingest() {
        let tmp = TempDir::new().unwrap();
        let mgr = CdcIngestManager::new(tmp.path()).unwrap();

        mgr.create_ingest(sample_config()).unwrap();
        mgr.drop_ingest("test_ingest").unwrap();
        assert!(mgr.list_ingests().is_empty());
    }

    #[test]
    fn test_persistence() {
        let tmp = TempDir::new().unwrap();

        {
            let mgr = CdcIngestManager::new(tmp.path()).unwrap();
            mgr.create_ingest(sample_config()).unwrap();
        }

        let mgr = CdcIngestManager::new(tmp.path()).unwrap();
        assert_eq!(mgr.list_ingests().len(), 1);
    }

    #[test]
    fn test_checkpoint_update() {
        let tmp = TempDir::new().unwrap();
        let mgr = CdcIngestManager::new(tmp.path()).unwrap();

        mgr.create_ingest(sample_config()).unwrap();

        let cp = IngestCheckpoint {
            name: "test_ingest".into(),
            last_source_offset: "offset:100".into(),
            records_applied: 50,
            records_failed: 2,
        };
        mgr.update_checkpoint(cp).unwrap();

        let status = mgr.get_ingest_status("test_ingest").unwrap();
        assert_eq!(status.records_applied, 50);
        assert_eq!(status.records_failed, 2);
        assert_eq!(status.last_offset, "offset:100");
    }

    #[test]
    fn test_remove_ingests_for_table() {
        let tmp = TempDir::new().unwrap();
        let mgr = CdcIngestManager::new(tmp.path()).unwrap();

        let mut c1 = sample_config();
        c1.name = "i1".into();
        c1.target_table_id = 42;

        let mut c2 = sample_config();
        c2.name = "i2".into();
        c2.target_table_id = 43;

        mgr.create_ingest(c1).unwrap();
        mgr.create_ingest(c2).unwrap();

        let removed = mgr.remove_ingests_for_table(42).unwrap();
        assert_eq!(removed, vec!["i1"]);
        assert_eq!(mgr.list_ingests().len(), 1);
    }
}
