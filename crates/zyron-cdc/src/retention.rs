//! CDC retention management for change data feed files.
//!
//! Enforces retention policies by purging old change records and compacting
//! redundant preimage/postimage pairs.

use std::collections::HashMap;
use std::sync::Arc;

use scc::HashMap as SccHashMap;
use serde::{Deserialize, Serialize};
use zyron_common::{Result, ZyronError};

use crate::change_feed::{CdfRegistry, ChangeRecord, ChangeType};

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
    /// Records a byte cap took ahead of their retention
    pub records_capped: u64,
    /// Tables whose feed reached its byte cap this pass
    pub tables_capped: u32,
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

/// What one key's records in the sealed segments amount to, gathered in
/// one pass over them so the rewrite decides each record on its own
/// without holding the others
#[derive(Default)]
struct KeyHistory {
    count: u64,
    first: Option<ChangeType>,
    last: Option<ChangeType>,
    /// The version and ordinal of the key's first update preimage
    first_preimage: Option<(u64, u64)>,
    /// The version and ordinal of the key's last insert or update
    /// postimage
    last_postimage: Option<(u64, u64)>,
}

impl KeyHistory {
    fn observe(&mut self, record: &ChangeRecord) {
        let at = (record.commit_version, record.change_ordinal);
        if self.first.is_none() {
            self.first = Some(record.change_type);
        }
        self.last = Some(record.change_type);
        self.count += 1;
        match record.change_type {
            ChangeType::UpdatePreimage => {
                if self.first_preimage.is_none() {
                    self.first_preimage = Some(at);
                }
            }
            ChangeType::Insert | ChangeType::UpdatePostimage => self.last_postimage = Some(at),
            ChangeType::Delete | ChangeType::SchemaChange | ChangeType::Truncate => {}
        }
    }

    /// Whether the record stays. A key with one record keeps it, a row
    /// born and deleted inside the segments loses every record, two
    /// records stay as they are, and a longer history keeps its deletes,
    /// its first preimage and its last postimage
    fn keeps(&self, record: &ChangeRecord) -> bool {
        if self.count == 1 {
            return true;
        }
        if self.first == Some(ChangeType::Insert) && self.last == Some(ChangeType::Delete) {
            return false;
        }
        if self.count == 2 {
            return true;
        }
        let at = (record.commit_version, record.change_ordinal);
        match record.change_type {
            ChangeType::Delete | ChangeType::SchemaChange | ChangeType::Truncate => true,
            ChangeType::UpdatePreimage => self.first_preimage == Some(at),
            ChangeType::Insert | ChangeType::UpdatePostimage => self.last_postimage == Some(at),
        }
    }
}

// ---------------------------------------------------------------------------
// CdcRetentionManager
// ---------------------------------------------------------------------------

/// Manages retention enforcement across all CDF-enabled tables.
pub struct CdcRetentionManager {
    policies: SccHashMap<u32, CdcRetentionPolicy>,
    cdf_registry: Arc<CdfRegistry>,
    /// The longest retention any feed is enforced at, in microseconds. Zero
    /// leaves each feed's own window in force
    max_retention_micros: i64,
    /// Bytes a feed may hold before its oldest changes are purged ahead of
    /// their retention. Zero sets no cap
    max_bytes_per_table: u64,
}

impl CdcRetentionManager {
    pub fn new(cdf_registry: Arc<CdfRegistry>) -> Self {
        Self {
            policies: SccHashMap::new(),
            cdf_registry,
            max_retention_micros: 0,
            max_bytes_per_table: 0,
        }
    }

    /// Sets the caps every feed stays under, whatever its own settings say.
    ///
    /// A feed must never stop its table accepting writes, so a feed at its
    /// byte cap purges oldest first rather than refusing anything
    pub fn with_caps(mut self, max_retention_micros: i64, max_bytes_per_table: u64) -> Self {
        self.max_retention_micros = max_retention_micros.max(0);
        self.max_bytes_per_table = max_bytes_per_table;
        self
    }

    /// The retention a feed is enforced at, its own window, held under the
    /// cap when one is set
    pub fn effective_retention_micros(&self, feed_retention_micros: i64) -> i64 {
        match (feed_retention_micros > 0, self.max_retention_micros > 0) {
            (true, true) => feed_retention_micros.min(self.max_retention_micros),
            (true, false) => feed_retention_micros,
            (false, true) => self.max_retention_micros,
            (false, false) => 0,
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

    /// Enforces age retention for every registered feed, using the explicit
    /// policy when one is set and the feed's own retention window otherwise.
    /// The hold LSN is the slowest consumer's confirmed position, records
    /// above it never age out, so a lagging replication slot or subscriber
    /// never loses changes it has not confirmed. Failures are collected per
    /// table so one bad feed never hides the rest
    pub fn enforce_all(&self, hold_lsn: Option<u64>) -> (RetentionStats, Vec<(u32, ZyronError)>) {
        let mut stats = RetentionStats::default();
        let mut failures = Vec::new();

        let now_micros = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_micros() as i64;

        for table_id in self.cdf_registry.table_ids() {
            let Some(feed) = self.cdf_registry.get_feed(table_id) else {
                continue;
            };
            let policy = self.get_policy(table_id);
            // An explicit policy is written in days, the feed's own window
            // in microseconds, and the cap holds either one down
            let own = match policy.as_ref() {
                Some(p) => p.retention_days as i64 * crate::change_feed::MICROS_PER_DAY,
                None => feed.config().retention_micros,
            };
            let retention_micros = self.effective_retention_micros(own);
            let size_before = feed.file_size_bytes();
            let mut touched = false;

            // Zero means the feed keeps everything until a cap or an explicit
            // truncation removes it, there is no age window to enforce
            if retention_micros > 0 {
                touched = true;
                let cutoff_ts = now_micros - retention_micros;
                match feed.purge_retention(cutoff_ts, hold_lsn) {
                    Ok(purged) => stats.records_purged += purged,
                    Err(e) => {
                        failures.push((table_id, e));
                        continue;
                    }
                }
            }

            // A feed over its byte cap gives up its oldest changes ahead of
            // their retention. The table keeps accepting writes throughout
            if self.max_bytes_per_table > 0 && feed.file_size_bytes() > self.max_bytes_per_table {
                touched = true;
                match feed.enforce_byte_cap(self.max_bytes_per_table) {
                    Ok(capped) => {
                        stats.records_capped += capped;
                        if capped > 0 {
                            stats.tables_capped += 1;
                        }
                    }
                    Err(e) => {
                        failures.push((table_id, e));
                        continue;
                    }
                }
            }
            if !touched {
                continue;
            }
            stats.tables_processed += 1;
            stats.bytes_reclaimed += size_before.saturating_sub(feed.file_size_bytes());

            if policy.as_ref().is_some_and(|p| p.compaction_enabled) {
                match self.compact_change_log(table_id) {
                    Ok(compaction) => stats.records_compacted += compaction.records_removed,
                    Err(e) => failures.push((table_id, e)),
                }
            }
        }

        (stats, failures)
    }

    /// Compacts the sealed segments of a table's feed, dropping the records
    /// a key's history makes redundant.
    ///
    /// For each primary key, over the sealed segments: a row created and
    /// deleted inside them loses every record, a longer history keeps its
    /// deletes, its first update preimage and its last insert or update
    /// postimage, and a key with one or two records keeps them. Two passes
    /// over the segments, each a segment at a time, the first gathering
    /// what every key's history amounts to and the second rewriting each
    /// segment in place with the records that stay, so the pass holds one
    /// segment and one entry per key rather than the feed. The open segment
    /// is left to a later pass, once it seals. A purge running meanwhile
    /// ends the pass, and the next cycle plans it again
    pub fn compact_change_log(&self, table_id: u32) -> Result<CompactionStats> {
        let feed = self
            .cdf_registry
            .get_feed(table_id)
            .ok_or(ZyronError::CdcFeedNotEnabled { table_id })?;

        let plan = feed.plan_compaction();
        if plan.record_count == 0 {
            return Ok(CompactionStats::default());
        }
        let mut keys: HashMap<Vec<u8>, KeyHistory> = HashMap::new();
        let walked = feed.visit_planned(&plan, &mut |record| {
            match keys.get_mut(&record.primary_key_data) {
                Some(history) => history.observe(record),
                None => {
                    let mut history = KeyHistory::default();
                    history.observe(record);
                    keys.insert(record.primary_key_data.clone(), history);
                }
            }
            Ok(())
        })?;
        if !walked {
            return Ok(CompactionStats::default());
        }
        let removed = feed.compact_planned(&plan, &mut |record| {
            keys.get(&record.primary_key_data)
                .is_none_or(|history| history.keeps(record))
        })?;
        let Some(records_removed) = removed else {
            return Ok(CompactionStats::default());
        };
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
            change_ordinal: 0,
            schema_version: 1,
            row_data: vec![pk, 1, 2, 3],
            primary_key_data: vec![pk],
            is_last_in_txn: true,
            projected: false,
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

    /// A compaction collapses each key's history in the sealed segments by
    /// the rules, leaves the open segment alone, and keeps every position
    /// naming the place it named before
    #[test]
    fn test_compaction_collapses_each_keys_history_in_the_sealed_segments() {
        let tmp = TempDir::new().unwrap();
        let registry = Arc::new(CdfRegistry::new(tmp.path().to_path_buf()));
        let feed = registry.enable_for_table(2, 30).unwrap();
        // Key 1 is inserted and updated twice, key 2 is born and deleted,
        // key 3 is inserted once
        let sealed = vec![
            make_record(1, 1_000, ChangeType::Insert, 1),
            make_record(2, 2_000, ChangeType::UpdatePreimage, 1),
            make_record(2, 2_000, ChangeType::UpdatePostimage, 1),
            make_record(3, 3_000, ChangeType::UpdatePreimage, 1),
            make_record(3, 3_000, ChangeType::UpdatePostimage, 1),
            make_record(4, 4_000, ChangeType::Insert, 2),
            make_record(5, 5_000, ChangeType::Delete, 2),
            make_record(6, 6_000, ChangeType::Insert, 3),
        ];
        feed.append_batch(&sealed).unwrap();
        feed.seal_open_segment().unwrap();
        // An update of key 3 in the open segment, which the pass leaves
        feed.append_batch(&[
            make_record(7, 7_000, ChangeType::UpdatePreimage, 3),
            make_record(7, 7_000, ChangeType::UpdatePostimage, 3),
        ])
        .unwrap();
        let before_at_six = feed.records_at_or_below(6);

        let mgr = CdcRetentionManager::new(Arc::clone(&registry));
        let stats = mgr.compact_change_log(2).unwrap();
        assert_eq!(
            stats.records_removed, 5,
            "key 1 loses three records and key 2 both of its own"
        );
        assert_eq!(feed.record_count(), 5);
        let kept: Vec<(u64, ChangeType, u8)> = feed
            .query_changes(0, u64::MAX)
            .unwrap()
            .iter()
            .map(|r| (r.commit_version, r.change_type, r.primary_key_data[0]))
            .collect();
        assert_eq!(
            kept,
            vec![
                (2, ChangeType::UpdatePreimage, 1),
                (3, ChangeType::UpdatePostimage, 1),
                (6, ChangeType::Insert, 3),
                (7, ChangeType::UpdatePreimage, 3),
                (7, ChangeType::UpdatePostimage, 3),
            ]
        );
        assert_eq!(
            feed.records_at_or_below(6),
            before_at_six,
            "the numbering a position counts in is unchanged"
        );
        // A second pass finds nothing left to drop in the sealed segment
        let again = mgr.compact_change_log(2).unwrap();
        assert_eq!(again.records_removed, 0);
    }
}
