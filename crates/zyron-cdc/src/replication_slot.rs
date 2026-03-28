//! Replication slots for WAL-based change streaming.
//!
//! Each slot tracks a consumer's confirmed LSN position, preventing WAL
//! segments from being deleted until the consumer has processed them.
//! Slot state is persisted via atomic rename for crash safety.

use std::fs::{self, File};
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

use scc::HashMap as SccHashMap;
use serde::{Deserialize, Serialize};
use zyron_common::{Result, ZyronError};
use zyron_wal::Lsn;

use crate::decoder::DecoderPlugin;

// ---------------------------------------------------------------------------
// SlotSnapshotMode
// ---------------------------------------------------------------------------

/// Snapshot mode for slot creation.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum SlotSnapshotMode {
    /// No initial snapshot.
    None,
    /// Export a snapshot of current table state at slot creation.
    ExportSnapshot,
    /// Use an existing named snapshot.
    UseSnapshot(String),
}

// ---------------------------------------------------------------------------
// SlotLagConfig
// ---------------------------------------------------------------------------

/// Configuration for replication slot lag monitoring.
#[derive(Debug, Clone)]
pub struct SlotLagConfig {
    /// Maximum allowed lag in bytes before a slot is deactivated (default 1 GB).
    pub max_lag_bytes: u64,
    /// Lag threshold in bytes that triggers a warning (default 512 MB).
    pub warn_lag_bytes: u64,
}

impl Default for SlotLagConfig {
    fn default() -> Self {
        Self {
            max_lag_bytes: 1024 * 1024 * 1024, // 1 GB
            warn_lag_bytes: 512 * 1024 * 1024, // 512 MB
        }
    }
}

// ---------------------------------------------------------------------------
// SlotLagWarning
// ---------------------------------------------------------------------------

/// Warning issued when a slot's lag approaches the configured limit.
#[derive(Debug, Clone)]
pub struct SlotLagWarning {
    pub slot_name: String,
    pub lag_bytes: u64,
    pub max_lag_bytes: u64,
}

// ---------------------------------------------------------------------------
// ReplicationSlot
// ---------------------------------------------------------------------------

/// A replication slot that tracks WAL consumption position.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReplicationSlot {
    pub name: String,
    pub plugin: DecoderPlugin,
    pub confirmed_lsn: u64,
    pub restart_lsn: u64,
    pub active: bool,
    pub created_at: i64,
    pub table_filter: Option<Vec<u32>>,
    pub snapshot_mode: SlotSnapshotMode,
}

// ---------------------------------------------------------------------------
// SlotManager
// ---------------------------------------------------------------------------

/// Manages replication slots with persistent state and lag monitoring.
pub struct SlotManager {
    slots: SccHashMap<String, ReplicationSlot>,
    state_file: PathBuf,
    pub lag_config: SlotLagConfig,
    dirty: AtomicBool,
}

impl SlotManager {
    /// Opens or creates the slot manager. Loads persisted state from the
    /// .zyslots file with crash recovery (prefers .zyslots over .zyslots.tmp).
    pub fn open(data_dir: &Path, lag_config: SlotLagConfig) -> Result<Self> {
        let state_file = data_dir.join(".zyslots");
        let tmp_file = data_dir.join(".zyslots.tmp");

        // Crash recovery: if only tmp exists, rename to final.
        if tmp_file.exists() && !state_file.exists() {
            fs::rename(&tmp_file, &state_file)?;
        } else if tmp_file.exists() {
            // Both exist: temp write did not complete, remove temp.
            let _ = fs::remove_file(&tmp_file);
        }

        let slots = SccHashMap::new();

        if state_file.exists() {
            let mut file = File::open(&state_file)?;
            let mut data = Vec::new();
            file.read_to_end(&mut data)?;

            if !data.is_empty() {
                let slot_list: Vec<ReplicationSlot> =
                    serde_json::from_slice(&data).map_err(|e| {
                        ZyronError::CdcStreamError(format!("failed to parse slot state: {e}"))
                    })?;
                for slot in slot_list {
                    let _ = slots.insert_sync(slot.name.clone(), slot);
                }
            }
        }

        Ok(Self {
            slots,
            state_file,
            lag_config,
            dirty: AtomicBool::new(false),
        })
    }

    /// Creates a new replication slot.
    pub fn create_slot(
        &self,
        name: &str,
        plugin: DecoderPlugin,
        table_filter: Option<Vec<u32>>,
    ) -> Result<ReplicationSlot> {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_micros() as i64;

        let slot = ReplicationSlot {
            name: name.to_string(),
            plugin,
            confirmed_lsn: 0,
            restart_lsn: 0,
            active: true,
            created_at: now,
            table_filter,
            snapshot_mode: SlotSnapshotMode::None,
        };

        if self
            .slots
            .insert_sync(name.to_string(), slot.clone())
            .is_err()
        {
            return Err(ZyronError::SlotAlreadyExists(name.to_string()));
        }

        self.persist()?;
        Ok(slot)
    }

    /// Creates a slot with an initial snapshot LSN for baseline data.
    pub fn create_slot_with_snapshot(
        &self,
        name: &str,
        plugin: DecoderPlugin,
        table_filter: Option<Vec<u32>>,
        snapshot_lsn: Lsn,
    ) -> Result<ReplicationSlot> {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_micros() as i64;

        let slot = ReplicationSlot {
            name: name.to_string(),
            plugin,
            confirmed_lsn: snapshot_lsn.0,
            restart_lsn: snapshot_lsn.0,
            active: true,
            created_at: now,
            table_filter,
            snapshot_mode: SlotSnapshotMode::ExportSnapshot,
        };

        if self
            .slots
            .insert_sync(name.to_string(), slot.clone())
            .is_err()
        {
            return Err(ZyronError::SlotAlreadyExists(name.to_string()));
        }

        self.persist()?;
        Ok(slot)
    }

    /// Drops a replication slot.
    pub fn drop_slot(&self, name: &str) -> Result<()> {
        self.slots
            .remove_sync(name)
            .ok_or_else(|| ZyronError::SlotNotFound(name.to_string()))?;
        self.persist()?;
        Ok(())
    }

    /// Lists all replication slots.
    pub fn list_slots(&self) -> Vec<ReplicationSlot> {
        let mut result = Vec::new();
        self.slots.iter_sync(|_name, slot| {
            result.push(slot.clone());
            true
        });
        result
    }

    /// Gets a slot by name.
    pub fn get_slot(&self, name: &str) -> Result<ReplicationSlot> {
        self.slots
            .read_sync(name, |_, slot| slot.clone())
            .ok_or_else(|| ZyronError::SlotNotFound(name.to_string()))
    }

    /// Advances the confirmed LSN for a slot.
    pub fn advance_slot(&self, name: &str, lsn: Lsn) -> Result<()> {
        match self.slots.entry_sync(name.to_string()) {
            scc::hash_map::Entry::Occupied(mut entry) => {
                let slot = entry.get_mut();
                slot.confirmed_lsn = lsn.0;
                slot.restart_lsn = lsn.0;
            }
            scc::hash_map::Entry::Vacant(_) => {
                return Err(ZyronError::SlotNotFound(name.to_string()));
            }
        }
        self.dirty.store(true, Ordering::Release);
        Ok(())
    }

    /// Returns the minimum restart_lsn across all active slots.
    /// WAL segments below this LSN must not be deleted.
    pub fn min_restart_lsn(&self) -> Option<Lsn> {
        let mut min_lsn: Option<u64> = None;
        self.slots.iter_sync(|_name, slot| {
            if slot.active && slot.restart_lsn > 0 {
                min_lsn = Some(match min_lsn {
                    Some(current) => current.min(slot.restart_lsn),
                    None => slot.restart_lsn,
                });
            }
            true
        });
        min_lsn.map(Lsn)
    }

    /// Calculates the lag in bytes between a slot's confirmed_lsn and the current WAL head.
    pub fn slot_lag_bytes(&self, name: &str, current_wal_lsn: Lsn) -> Result<u64> {
        let slot = self.get_slot(name)?;
        Ok(current_wal_lsn.0.saturating_sub(slot.confirmed_lsn))
    }

    /// Returns warnings for slots approaching or exceeding the configured lag limit.
    pub fn check_lag_warnings(&self, current_wal_lsn: Lsn) -> Vec<SlotLagWarning> {
        let mut warnings = Vec::new();
        self.slots.iter_sync(|_name, slot| {
            if slot.active {
                let lag = current_wal_lsn.0.saturating_sub(slot.confirmed_lsn);
                if lag >= self.lag_config.warn_lag_bytes {
                    warnings.push(SlotLagWarning {
                        slot_name: slot.name.clone(),
                        lag_bytes: lag,
                        max_lag_bytes: self.lag_config.max_lag_bytes,
                    });
                }
            }
            true
        });
        warnings
    }

    /// Deactivates slots that exceed the maximum lag. Returns names of deactivated slots.
    pub fn deactivate_lagging_slots(&self, current_wal_lsn: Lsn) -> Result<Vec<String>> {
        let mut deactivated = Vec::new();
        self.slots.iter_sync(|_name, slot| {
            if slot.active {
                let lag = current_wal_lsn.0.saturating_sub(slot.confirmed_lsn);
                if lag >= self.lag_config.max_lag_bytes {
                    deactivated.push(slot.name.clone());
                }
            }
            true
        });

        for name in &deactivated {
            if let scc::hash_map::Entry::Occupied(mut entry) = self.slots.entry_sync(name.clone()) {
                entry.get_mut().active = false;
            }
        }

        if !deactivated.is_empty() {
            self.persist()?;
        }

        Ok(deactivated)
    }

    /// Removes all slots that filter exclusively to the given table_id.
    pub fn remove_slots_for_table(&self, table_id: u32) -> Result<Vec<String>> {
        let mut to_remove = Vec::new();
        self.slots.iter_sync(|name, slot| {
            if let Some(ref filter) = slot.table_filter {
                if filter.len() == 1 && filter[0] == table_id {
                    to_remove.push(name.clone());
                }
            }
            true
        });

        for name in &to_remove {
            let _ = self.slots.remove_sync(name);
        }

        if !to_remove.is_empty() {
            self.persist()?;
        }

        Ok(to_remove)
    }

    /// Persists slot state if any advance has occurred since the last flush.
    /// Called periodically by checkpoint or background workers.
    pub fn flush_if_dirty(&self) -> Result<()> {
        if self.dirty.swap(false, Ordering::AcqRel) {
            self.persist()?;
        }
        Ok(())
    }

    /// Persists slot state to disk using atomic rename.
    pub fn persist(&self) -> Result<()> {
        let slots = self.list_slots();
        let data = serde_json::to_vec(&slots).map_err(|e| {
            ZyronError::CdcStreamError(format!("failed to serialize slot state: {e}"))
        })?;

        let tmp_path = self
            .state_file
            .parent()
            .unwrap_or(Path::new("."))
            .join(".zyslots.tmp");
        {
            let mut tmp = File::create(&tmp_path)?;
            tmp.write_all(&data)?;
            tmp.sync_all()?;
        }

        fs::rename(&tmp_path, &self.state_file)?;
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Retention hook helper
// ---------------------------------------------------------------------------

/// Creates a WAL retention hook closure that returns the minimum restart_lsn
/// across all active replication slots.
pub fn create_retention_hook(
    slot_manager: Arc<SlotManager>,
) -> Arc<dyn Fn() -> Option<Lsn> + Send + Sync> {
    Arc::new(move || slot_manager.min_restart_lsn())
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[test]
    fn test_create_and_list_slots() {
        let tmp = TempDir::new().unwrap();
        let mgr = SlotManager::open(tmp.path(), SlotLagConfig::default()).unwrap();

        let slot = mgr
            .create_slot("test_slot", DecoderPlugin::ZyronCdc, None)
            .unwrap();
        assert_eq!(slot.name, "test_slot");
        assert!(slot.active);
        assert_eq!(slot.confirmed_lsn, 0);

        let slots = mgr.list_slots();
        assert_eq!(slots.len(), 1);
    }

    #[test]
    fn test_create_duplicate_slot_fails() {
        let tmp = TempDir::new().unwrap();
        let mgr = SlotManager::open(tmp.path(), SlotLagConfig::default()).unwrap();

        mgr.create_slot("s1", DecoderPlugin::Debezium, None)
            .unwrap();
        let result = mgr.create_slot("s1", DecoderPlugin::Debezium, None);
        assert!(result.is_err());
    }

    #[test]
    fn test_drop_slot() {
        let tmp = TempDir::new().unwrap();
        let mgr = SlotManager::open(tmp.path(), SlotLagConfig::default()).unwrap();

        mgr.create_slot("s1", DecoderPlugin::ZyronCdc, None)
            .unwrap();
        mgr.drop_slot("s1").unwrap();
        assert!(mgr.list_slots().is_empty());

        assert!(mgr.drop_slot("nonexistent").is_err());
    }

    #[test]
    fn test_advance_slot() {
        let tmp = TempDir::new().unwrap();
        let mgr = SlotManager::open(tmp.path(), SlotLagConfig::default()).unwrap();

        mgr.create_slot("s1", DecoderPlugin::ZyronCdc, None)
            .unwrap();
        mgr.advance_slot("s1", Lsn(1000)).unwrap();

        let slot = mgr.get_slot("s1").unwrap();
        assert_eq!(slot.confirmed_lsn, 1000);
    }

    #[test]
    fn test_min_restart_lsn() {
        let tmp = TempDir::new().unwrap();
        let mgr = SlotManager::open(tmp.path(), SlotLagConfig::default()).unwrap();

        assert!(mgr.min_restart_lsn().is_none());

        mgr.create_slot("s1", DecoderPlugin::ZyronCdc, None)
            .unwrap();
        mgr.advance_slot("s1", Lsn(1000)).unwrap();

        mgr.create_slot("s2", DecoderPlugin::ZyronCdc, None)
            .unwrap();
        mgr.advance_slot("s2", Lsn(500)).unwrap();

        assert_eq!(mgr.min_restart_lsn(), Some(Lsn(500)));
    }

    #[test]
    fn test_slot_lag_bytes() {
        let tmp = TempDir::new().unwrap();
        let mgr = SlotManager::open(tmp.path(), SlotLagConfig::default()).unwrap();

        mgr.create_slot("s1", DecoderPlugin::ZyronCdc, None)
            .unwrap();
        mgr.advance_slot("s1", Lsn(1000)).unwrap();

        let lag = mgr.slot_lag_bytes("s1", Lsn(5000)).unwrap();
        assert_eq!(lag, 4000);
    }

    #[test]
    fn test_lag_warnings() {
        let tmp = TempDir::new().unwrap();
        let config = SlotLagConfig {
            max_lag_bytes: 1000,
            warn_lag_bytes: 500,
        };
        let mgr = SlotManager::open(tmp.path(), config).unwrap();

        mgr.create_slot("s1", DecoderPlugin::ZyronCdc, None)
            .unwrap();
        mgr.advance_slot("s1", Lsn(100)).unwrap();

        let warnings = mgr.check_lag_warnings(Lsn(700));
        assert_eq!(warnings.len(), 1);
        assert_eq!(warnings[0].slot_name, "s1");

        let warnings = mgr.check_lag_warnings(Lsn(200));
        assert!(warnings.is_empty());
    }

    #[test]
    fn test_deactivate_lagging_slots() {
        let tmp = TempDir::new().unwrap();
        let config = SlotLagConfig {
            max_lag_bytes: 1000,
            warn_lag_bytes: 500,
        };
        let mgr = SlotManager::open(tmp.path(), config).unwrap();

        mgr.create_slot("s1", DecoderPlugin::ZyronCdc, None)
            .unwrap();
        mgr.advance_slot("s1", Lsn(100)).unwrap();

        let deactivated = mgr.deactivate_lagging_slots(Lsn(1200)).unwrap();
        assert_eq!(deactivated, vec!["s1"]);

        let slot = mgr.get_slot("s1").unwrap();
        assert!(!slot.active);
    }

    #[test]
    fn test_persistence_and_recovery() {
        let tmp = TempDir::new().unwrap();

        {
            let mgr = SlotManager::open(tmp.path(), SlotLagConfig::default()).unwrap();
            mgr.create_slot("s1", DecoderPlugin::Debezium, Some(vec![1, 2]))
                .unwrap();
            mgr.advance_slot("s1", Lsn(5000)).unwrap();
            mgr.flush_if_dirty().unwrap();
        }

        // Reopen and verify state persisted.
        let mgr = SlotManager::open(tmp.path(), SlotLagConfig::default()).unwrap();
        let slot = mgr.get_slot("s1").unwrap();
        assert_eq!(slot.plugin, DecoderPlugin::Debezium);
        assert_eq!(slot.confirmed_lsn, 5000);
        assert_eq!(slot.table_filter, Some(vec![1, 2]));
    }

    #[test]
    fn test_crash_recovery_tmp_only() {
        let tmp = TempDir::new().unwrap();

        // Write state to .zyslots.tmp (simulating crash during persist).
        let tmp_file = tmp.path().join(".zyslots.tmp");
        let slot = ReplicationSlot {
            name: "recovered".into(),
            plugin: DecoderPlugin::ZyronCdc,
            confirmed_lsn: 999,
            restart_lsn: 999,
            active: true,
            created_at: 0,
            table_filter: None,
            snapshot_mode: SlotSnapshotMode::None,
        };
        let data = serde_json::to_vec(&vec![slot]).unwrap();
        fs::write(&tmp_file, &data).unwrap();

        // Open: should recover from tmp file.
        let mgr = SlotManager::open(tmp.path(), SlotLagConfig::default()).unwrap();
        let slot = mgr.get_slot("recovered").unwrap();
        assert_eq!(slot.confirmed_lsn, 999);
    }

    #[test]
    fn test_remove_slots_for_table() {
        let tmp = TempDir::new().unwrap();
        let mgr = SlotManager::open(tmp.path(), SlotLagConfig::default()).unwrap();

        mgr.create_slot("s1", DecoderPlugin::ZyronCdc, Some(vec![42]))
            .unwrap();
        mgr.create_slot("s2", DecoderPlugin::ZyronCdc, Some(vec![42, 43]))
            .unwrap();
        mgr.create_slot("s3", DecoderPlugin::ZyronCdc, None)
            .unwrap();

        let removed = mgr.remove_slots_for_table(42).unwrap();
        // Only s1 should be removed (single-table filter matching 42).
        assert_eq!(removed, vec!["s1"]);
        assert_eq!(mgr.list_slots().len(), 2);
    }

    #[test]
    fn test_create_slot_with_snapshot() {
        let tmp = TempDir::new().unwrap();
        let mgr = SlotManager::open(tmp.path(), SlotLagConfig::default()).unwrap();

        let slot = mgr
            .create_slot_with_snapshot("snap_slot", DecoderPlugin::ZyronCdc, None, Lsn(5000))
            .unwrap();
        assert_eq!(slot.confirmed_lsn, 5000);
        assert_eq!(slot.restart_lsn, 5000);
        assert_eq!(slot.snapshot_mode, SlotSnapshotMode::ExportSnapshot);
    }

    #[test]
    fn test_retention_hook() {
        let tmp = TempDir::new().unwrap();
        let mgr = Arc::new(SlotManager::open(tmp.path(), SlotLagConfig::default()).unwrap());

        mgr.create_slot("s1", DecoderPlugin::ZyronCdc, None)
            .unwrap();
        mgr.advance_slot("s1", Lsn(1000)).unwrap();

        let hook = create_retention_hook(mgr.clone());
        assert_eq!(hook(), Some(Lsn(1000)));
    }
}
