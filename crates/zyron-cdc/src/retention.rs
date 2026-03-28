//! CDC retention management for change data feed files.
//!
//! Enforces retention policies by purging old change records and compacting
//! redundant preimage/postimage pairs.

use std::sync::Arc;

use scc::HashMap as SccHashMap;
use serde::{Deserialize, Serialize};
use zyron_common::{Result, ZyronError};

use crate::change_feed::{CdfRegistry, ChangeType};

// ---------------------------------------------------------------------------
// CdcRetentionPolicy
// ---------------------------------------------------------------------------

/// Retention policy for a table's change data feed.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CdcRetentionPolicy {
    pub table_id: u32,
    pub retention_days: u32,
    pub compaction_enabled: bool,
}

// ---------------------------------------------------------------------------
// RetentionStats
// ---------------------------------------------------------------------------

/// Statistics from a retention enforcement run.
#[derive(Debug, Clone, Default)]
pub struct RetentionStats {
    pub tables_processed: u32,
    pub records_purged: u64,
    pub records_compacted: u64,
    pub bytes_reclaimed: u64,
}

// ---------------------------------------------------------------------------
// CompactionStats
// ---------------------------------------------------------------------------

/// Statistics from a compaction run on a single table.
#[derive(Debug, Clone, Default)]
pub struct CompactionStats {
    pub records_merged: u64,
    pub records_removed: u64,
    pub new_file_size: u64,
}

// ---------------------------------------------------------------------------
// CdcRetentionManager
// ---------------------------------------------------------------------------

/// Manages retention enforcement across all CDF-enabled tables.
pub struct CdcRetentionManager {
    policies: SccHashMap<u32, CdcRetentionPolicy>,
    cdf_registry: Arc<CdfRegistry>,
}

impl CdcRetentionManager {
    pub fn new(cdf_registry: Arc<CdfRegistry>) -> Self {
        Self {
            policies: SccHashMap::new(),
            cdf_registry,
        }
    }

    /// Sets or updates a retention policy for a table.
    pub fn set_policy(&self, policy: CdcRetentionPolicy) -> Result<()> {
        match self.policies.entry_sync(policy.table_id) {
            scc::hash_map::Entry::Occupied(mut o) => {
                *o.get_mut() = policy;
            }
            scc::hash_map::Entry::Vacant(v) => {
                v.insert_entry(policy);
            }
        }
        Ok(())
    }

    /// Removes the retention policy for a table.
    pub fn remove_policy(&self, table_id: u32) -> Result<()> {
        let _ = self.policies.remove_sync(&table_id);
        Ok(())
    }

    /// Returns the retention policy for a table, if set.
    pub fn get_policy(&self, table_id: u32) -> Option<CdcRetentionPolicy> {
        self.policies
            .read_sync(&table_id, |_, policy| policy.clone())
    }

    /// Enforces retention for all tables with policies. Called by the background worker.
    pub fn enforce_all(&self) -> Result<RetentionStats> {
        let mut stats = RetentionStats::default();

        let mut policies = Vec::new();
        self.policies.iter_sync(|_id, policy| {
            policies.push(policy.clone());
            true
        });

        for policy in &policies {
            if let Some(feed) = self.cdf_registry.get_feed(policy.table_id) {
                stats.tables_processed += 1;

                // Calculate the minimum version to retain based on retention_days.
                // For now, use a simple approach: purge records older than
                // retention_days worth of versions. The actual version cutoff
                // should be based on timestamps, but this requires scanning
                // the CDF file. We use the time index for timestamp-based purge.
                let retention_micros = policy.retention_days as i64 * 24 * 60 * 60 * 1_000_000;
                let now_micros = std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap_or_default()
                    .as_micros() as i64;
                let cutoff_ts = now_micros - retention_micros;

                // Find the maximum version before the cutoff timestamp without
                // materializing all expired records into memory.
                if let Some(max_version) = feed.max_version_before_time(cutoff_ts) {
                    let size_before = feed.file_size_bytes();
                    let purged = feed.purge_before_version(max_version + 1)?;
                    let size_after = feed.file_size_bytes();
                    stats.records_purged += purged;
                    stats.bytes_reclaimed += size_before.saturating_sub(size_after);
                }

                // Run compaction if enabled.
                if policy.compaction_enabled {
                    let compaction = self.compact_change_log(policy.table_id)?;
                    stats.records_compacted += compaction.records_removed;
                }
            }
        }

        Ok(stats)
    }

    /// Compacts the change log for a single table by removing redundant
    /// preimage/postimage pairs for records that have been updated multiple times.
    /// Keeps only the latest state for each primary key within a version window.
    /// Compacts the change log for a single table.
    ///
    /// For each primary key group:
    /// - If the last event is a Delete and the first is an Insert (with no
    ///   preceding preimage), the row was created and destroyed within this
    ///   window. All records for this PK are eliminated entirely.
    /// - Otherwise, intermediate preimage/postimage pairs are collapsed:
    ///   only the first preimage and the last postimage/insert are kept.
    /// - Delete, SchemaChange, and Truncate records are always kept unless
    ///   the entire PK group is eliminated by insert+delete cancellation.
    pub fn compact_change_log(&self, table_id: u32) -> Result<CompactionStats> {
        let feed = self
            .cdf_registry
            .get_feed(table_id)
            .ok_or(ZyronError::CdcFeedNotEnabled { table_id })?;

        let all_records = feed.query_changes(0, u64::MAX)?;
        if all_records.is_empty() {
            return Ok(CompactionStats::default());
        }

        // Sort indices by primary key to group without HashMap or key cloning.
        let mut sorted_indices: Vec<usize> = (0..all_records.len()).collect();
        sorted_indices.sort_by(|&a, &b| {
            all_records[a]
                .primary_key_data
                .cmp(&all_records[b].primary_key_data)
        });

        let mut keep_indices: Vec<bool> = vec![false; all_records.len()];
        let mut group_start = 0;

        while group_start < sorted_indices.len() {
            let pk = &all_records[sorted_indices[group_start]].primary_key_data;
            let mut group_end = group_start + 1;
            while group_end < sorted_indices.len()
                && all_records[sorted_indices[group_end]].primary_key_data == *pk
            {
                group_end += 1;
            }

            let group = &sorted_indices[group_start..group_end];

            if group.len() == 1 {
                keep_indices[group[0]] = true;
            } else {
                // Check for insert+delete cancellation: if the first event
                // created the row (Insert) and the last event removed it
                // (Delete), the entire group can be dropped.
                let first_type = all_records[group[0]].change_type;
                let last_type = all_records[group[group.len() - 1]].change_type;
                let is_insert_then_delete =
                    first_type == ChangeType::Insert && last_type == ChangeType::Delete;

                if is_insert_then_delete {
                    // Row was born and died within this window. Drop all records.
                    // (SchemaChange/Truncate within the group are also dropped
                    // since they refer to a row that no longer exists.)
                } else if group.len() == 2 {
                    // Two records, not an insert+delete pair. Keep both.
                    keep_indices[group[0]] = true;
                    keep_indices[group[1]] = true;
                } else {
                    // Collapse intermediate preimage/postimage pairs.
                    let mut first_preimage: Option<usize> = None;
                    let mut last_postimage: Option<usize> = None;

                    for &idx in group {
                        match all_records[idx].change_type {
                            ChangeType::UpdatePreimage => {
                                if first_preimage.is_none() {
                                    first_preimage = Some(idx);
                                }
                            }
                            ChangeType::Insert | ChangeType::UpdatePostimage => {
                                last_postimage = Some(idx);
                            }
                            ChangeType::Delete
                            | ChangeType::SchemaChange
                            | ChangeType::Truncate => {
                                keep_indices[idx] = true;
                            }
                        }
                    }

                    if let Some(idx) = first_preimage {
                        keep_indices[idx] = true;
                    }
                    if let Some(idx) = last_postimage {
                        keep_indices[idx] = true;
                    }
                }
            }

            group_start = group_end;
        }

        let kept_count = keep_indices.iter().filter(|&&k| k).count();
        let records_removed = (all_records.len() - kept_count) as u64;

        if records_removed == 0 {
            return Ok(CompactionStats::default());
        }

        // Build kept records in original version order (enumerate preserves
        // the original index order from all_records, which is version-ordered).
        let kept_records: Vec<_> = all_records
            .into_iter()
            .enumerate()
            .filter(|(i, _)| keep_indices[*i])
            .map(|(_, r)| r)
            .collect();

        // Clear the file entirely, then write back only the kept records.
        feed.purge_before_version(u64::MAX)?;
        if !kept_records.is_empty() {
            feed.append_batch(&kept_records)?;
        }

        Ok(CompactionStats {
            records_merged: 0,
            records_removed,
            new_file_size: feed.file_size_bytes(),
        })
    }

    /// Purges changes older than the given version for a single table.
    pub fn purge_old_changes(&self, table_id: u32, min_version: u64) -> Result<u64> {
        let feed = self
            .cdf_registry
            .get_feed(table_id)
            .ok_or(ZyronError::CdcFeedNotEnabled { table_id })?;
        feed.purge_before_version(min_version)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::change_feed::{ChangeRecord, ChangeType};
    use tempfile::TempDir;

    fn make_record(version: u64, ts: i64, change_type: ChangeType, pk: u8) -> ChangeRecord {
        ChangeRecord {
            change_type,
            commit_version: version,
            commit_timestamp: ts,
            table_id: 1,
            txn_id: 100,
            schema_version: 1,
            row_data: vec![pk, 1, 2, 3],
            primary_key_data: vec![pk],
            is_last_in_txn: true,
        }
    }

    #[test]
    fn test_set_and_get_policy() {
        let tmp = TempDir::new().unwrap();
        let registry = Arc::new(CdfRegistry::new(tmp.path().to_path_buf()));
        let mgr = CdcRetentionManager::new(registry);

        let policy = CdcRetentionPolicy {
            table_id: 1,
            retention_days: 30,
            compaction_enabled: true,
        };
        mgr.set_policy(policy.clone()).unwrap();

        let got = mgr.get_policy(1).unwrap();
        assert_eq!(got.retention_days, 30);
        assert!(got.compaction_enabled);

        mgr.remove_policy(1).unwrap();
        assert!(mgr.get_policy(1).is_none());
    }

    #[test]
    fn test_purge_old_changes() {
        let tmp = TempDir::new().unwrap();
        let registry = Arc::new(CdfRegistry::new(tmp.path().to_path_buf()));
        let feed = registry.enable_for_table(1, 30).unwrap();

        let records: Vec<ChangeRecord> = (1..=10)
            .map(|i| make_record(i, i as i64 * 1000, ChangeType::Insert, 1))
            .collect();
        feed.append_batch(&records).unwrap();

        let mgr = CdcRetentionManager::new(registry.clone());
        let purged = mgr.purge_old_changes(1, 6).unwrap();
        assert_eq!(purged, 5);
    }

    #[test]
    fn test_purge_nonexistent_table() {
        let tmp = TempDir::new().unwrap();
        let registry = Arc::new(CdfRegistry::new(tmp.path().to_path_buf()));
        let mgr = CdcRetentionManager::new(registry);

        let result = mgr.purge_old_changes(999, 1);
        assert!(result.is_err());
    }
}
