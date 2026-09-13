// -----------------------------------------------------------------------------
// Publication retention enforcer.
//
// Walks the catalog publication list on a fixed interval and truncates the
// upstream CDF buffer for each publication whose retention window has passed.
// When retain_until_subscribers_advance is true the effective retention point
// is the minimum of the configured retention age and the slowest active
// subscriber's last_seen_lsn, so a lagging consumer holds the window open.
// -----------------------------------------------------------------------------

use std::sync::Arc;
use std::sync::atomic::AtomicBool;
use std::time::Duration;

use tracing::info;
use zyron_catalog::Catalog;

/// Polling interval between retention sweeps.
pub const DEFAULT_INTERVAL_SECS: u64 = 3600;

pub async fn publication_retention_loop(
    catalog: Arc<Catalog>,
    cdc_registry: Option<Arc<zyron_cdc::CdfRegistry>>,
    shutdown: Arc<AtomicBool>,
    wake: Arc<tokio::sync::Notify>,
    interval_secs: u64,
    metrics: Option<Arc<zyron_common::LabeledMetrics>>,
) {
    let mut ticker = tokio::time::interval(Duration::from_secs(interval_secs.max(60)));
    while super::tick_until_shutdown(&mut ticker, &shutdown, &wake).await {
        run_retention_sweep(&catalog, cdc_registry.as_deref(), metrics.as_deref()).await;
    }
}

// -----------------------------------------------------------------------------
// Single-pass sweep used by the loop and by tests
// -----------------------------------------------------------------------------

/// Iterates every publication and truncates each member table's CDF up to the
/// computed retention point. Returns the total number of records removed
/// across all tables touched during this sweep.
pub async fn run_retention_sweep(
    catalog: &Catalog,
    cdc_registry: Option<&zyron_cdc::CdfRegistry>,
    metrics: Option<&zyron_common::LabeledMetrics>,
) -> u64 {
    let mut grand_total: u64 = 0;
    let publications = catalog.list_publications();
    for pub_entry in publications {
        let (cutoff_secs, hold_lsn) = compute_retention_bounds(catalog, &pub_entry);
        // Records older than this commit-timestamp cutoff age out. Change
        // records stamp commit_timestamp in microseconds
        let cutoff_micros = cutoff_secs as i64 * 1_000_000;
        if let Some(m) = metrics {
            // Lag is how far behind wall-clock now the age-based truncation
            // horizon sits
            let lag = current_secs().saturating_sub(cutoff_secs);
            m.pubRetentionLagSet(&pub_entry.id.0.to_string(), lag);
        }
        let mut total_removed: u64 = 0;
        if let Some(reg) = cdc_registry {
            let tables = catalog.get_publication_tables(pub_entry.id);
            for pt in tables {
                match reg
                    .truncate_retention(pt.table_id.0, cutoff_micros, hold_lsn)
                    .await
                {
                    Ok(removed) => {
                        total_removed = total_removed.saturating_add(removed);
                    }
                    Err(e) => {
                        tracing::warn!(
                            target: "zyron::retention",
                            publication = %pub_entry.name,
                            table_id = pt.table_id.0,
                            error = %e,
                            "publication retention truncate failed"
                        );
                    }
                }
            }
        }
        grand_total = grand_total.saturating_add(total_removed);
        info!(
            target: "zyron::retention",
            publication = %pub_entry.name,
            cutoff_secs,
            hold_lsn,
            records_removed = total_removed,
            "publication retention sweep complete"
        );
    }
    grand_total
}

/// Returns the age cutoff in epoch seconds plus the hold LSN. Records older
/// than the cutoff age out of the CDF. When the publication holds records
/// until subscribers advance, the hold LSN is the slowest active
/// subscriber's confirmed position: records above it stay regardless of
/// age, in the LSN units the feed's commit versions actually use.
fn compute_retention_bounds(
    catalog: &Catalog,
    pub_entry: &zyron_catalog::PublicationEntry,
) -> (u64, Option<u64>) {
    let cutoff_secs = current_secs().saturating_sub(pub_entry.retention_days as u64 * 86400);
    if !pub_entry.retain_until_advance {
        return (cutoff_secs, None);
    }
    let mut min_lsn = u64::MAX;
    for sub in catalog.list_publication_subscribers(pub_entry.id) {
        if sub.state == zyron_catalog::SubscriptionState::Active {
            min_lsn = min_lsn.min(sub.last_seen_lsn);
        }
    }
    if min_lsn == u64::MAX {
        (cutoff_secs, None)
    } else {
        (cutoff_secs, Some(min_lsn))
    }
}

fn current_secs() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}

#[cfg(test)]
mod tests {
    use super::*;

    use zyron_buffer::{BufferPool, BufferPoolConfig};
    use zyron_catalog::{
        Catalog, CatalogCache, CatalogClassification, HeapCatalogStorage, PublicationEntry,
        PublicationId, PublicationTableEntry, RowFormat, SchemaId, TableId,
    };
    use zyron_cdc::{CdfRegistry, ChangeRecord, ChangeType};
    use zyron_storage::{DiskManager, DiskManagerConfig};
    use zyron_wal::{WalWriter, WalWriterConfig};

    #[test]
    fn default_interval_is_one_hour() {
        assert_eq!(DEFAULT_INTERVAL_SECS, 3600);
    }

    #[test]
    fn current_secs_monotonic() {
        let a = current_secs();
        let b = current_secs();
        assert!(b >= a);
    }

    // -------------------------------------------------------------------
    // Helpers for retention sweep tests
    // -------------------------------------------------------------------

    async fn build_catalog(tmp: &tempfile::TempDir) -> Arc<Catalog> {
        let data_dir = tmp.path().join("data");
        let wal_dir = tmp.path().join("wal");
        std::fs::create_dir_all(&data_dir).unwrap();
        std::fs::create_dir_all(&wal_dir).unwrap();

        let wal = Arc::new(
            WalWriter::new(WalWriterConfig {
                wal_dir,
                segment_size: 1024 * 1024,
                fsync_enabled: false,
                ring_buffer_capacity: 1 * 1024 * 1024,
            })
            .unwrap(),
        );

        let disk = Arc::new(
            DiskManager::new(DiskManagerConfig {
                data_dir,
                fsync_enabled: false,
                ..Default::default()
            })
            .await
            .unwrap(),
        );

        let pool = Arc::new(BufferPool::new(BufferPoolConfig { num_frames: 256 }));
        let storage = Arc::new(HeapCatalogStorage::new(disk, pool).unwrap());
        let cache = Arc::new(CatalogCache::new(64, 32));
        Arc::new(Catalog::new(storage, cache, wal).await.unwrap())
    }

    fn make_record(version: u64) -> ChangeRecord {
        ChangeRecord {
            change_type: ChangeType::Insert,
            commit_version: version,
            commit_timestamp: version as i64 * 1000,
            table_id: 0,
            txn_id: 1,
            schema_version: 1,
            change_ordinal: 0,
            row_data: vec![1, 2, 3],
            primary_key_data: vec![version as u8],
            is_last_in_txn: true,
            projected: false,
        }
    }

    fn make_pub(id: u32, retention_days: u32) -> PublicationEntry {
        PublicationEntry {
            id: PublicationId(id),
            schema_id: SchemaId(1),
            name: format!("pub_{id}"),
            change_feed: true,
            row_format: RowFormat::Text,
            retention_days,
            retain_until_advance: false,
            max_rows_per_sec: None,
            max_bytes_per_sec: None,
            max_concurrent_subscribers: None,
            classification: CatalogClassification::Internal,
            allow_initial_snapshot: false,
            where_predicate: None,
            columns_projection: Vec::new(),
            rls_using_predicate: None,
            tags: Vec::new(),
            schema_fingerprint: [0u8; 32],
            owner_role_id: 0,
            created_at: 0,
        }
    }

    #[tokio::test]
    async fn retention_loop_calls_truncate() {
        let tmp = tempfile::TempDir::new().unwrap();
        let catalog = build_catalog(&tmp).await;

        // retain_until_advance=false, retention_days=0 means cutoff=current
        // epoch seconds. Every commit_version<=that is eligible. We pick
        // small versions (1..=10) that are far below the current time.
        let pub_id = catalog.create_publication(make_pub(0, 0)).await.unwrap();

        let cdf_registry = Arc::new(CdfRegistry::new(tmp.path().join("cdf_root")));
        let table_id: u32 = 4242;
        let feed = cdf_registry.enable_for_table(table_id, 30).unwrap();

        let records: Vec<ChangeRecord> = (1..=10).map(make_record).collect();
        feed.append_batch(&records).unwrap();
        assert_eq!(feed.record_count(), 10);

        catalog
            .add_publication_table(PublicationTableEntry {
                id: 0,
                publication_id: pub_id,
                table_id: TableId(table_id),
                where_predicate: None,
                columns: Vec::new(),
                created_at: 0,
            })
            .await
            .unwrap();

        let removed = run_retention_sweep(&catalog, Some(&cdf_registry), None).await;
        assert_eq!(removed, 10);
        assert_eq!(feed.record_count(), 0);
    }

    #[tokio::test]
    async fn retention_loop_idempotent() {
        let tmp = tempfile::TempDir::new().unwrap();
        let catalog = build_catalog(&tmp).await;

        let pub_id = catalog.create_publication(make_pub(0, 0)).await.unwrap();
        let cdf_registry = Arc::new(CdfRegistry::new(tmp.path().join("cdf_root")));
        let table_id: u32 = 7777;
        let feed = cdf_registry.enable_for_table(table_id, 30).unwrap();

        let records: Vec<ChangeRecord> = (1..=5).map(make_record).collect();
        feed.append_batch(&records).unwrap();

        catalog
            .add_publication_table(PublicationTableEntry {
                id: 0,
                publication_id: pub_id,
                table_id: TableId(table_id),
                where_predicate: None,
                columns: Vec::new(),
                created_at: 0,
            })
            .await
            .unwrap();

        let first = run_retention_sweep(&catalog, Some(&cdf_registry), None).await;
        assert_eq!(first, 5);
        let second = run_retention_sweep(&catalog, Some(&cdf_registry), None).await;
        assert_eq!(second, 0);
        assert_eq!(feed.record_count(), 0);
    }
}
