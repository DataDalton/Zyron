//! Source connectors for ingesting data into streaming jobs.
//!
//! ZyronTableSource pulls change records directly from a ZyronDB change data
//! feed and yields CdfChange items. ZyronSourceAdapter is the trait object the
//! runner uses to pull rows from a remote Zyron publication.

use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use zyron_common::{Result, ZyronError};

// ---------------------------------------------------------------------------
// ZyronTableSource
// ---------------------------------------------------------------------------

/// Pulls change records directly from a ZyronDB change data feed.
/// Each call to read_batch advances the last_version watermark to the
/// max commit_version observed. Returns a Vec of CdfChange items.
pub struct ZyronTableSource {
    table_id: u32,
    cdf: Arc<zyron_cdc::ChangeDataFeed>,
    last_version: AtomicU64,
}

/// One row pulled from the CDF, exposing the raw row_data payload and
/// the change type for downstream filter and project stages.
#[derive(Debug, Clone)]
pub struct CdfChange {
    pub commit_version: u64,
    pub commit_timestamp: i64,
    pub change_type: zyron_cdc::ChangeType,
    pub row_data: Vec<u8>,
    pub primary_key_data: Vec<u8>,
}

impl ZyronTableSource {
    /// Resolves the CDF for the given table through the registry. Returns an
    /// error if the table has no active feed.
    pub fn new(table_id: u32, cdc_registry: Arc<zyron_cdc::CdfRegistry>) -> Result<Self> {
        let cdf = cdc_registry.get_feed(table_id).ok_or_else(|| {
            ZyronError::StreamingError(format!(
                "no change data feed registered for table {table_id}"
            ))
        })?;
        Ok(Self {
            table_id,
            cdf,
            last_version: AtomicU64::new(0),
        })
    }

    /// Returns the table id this source reads from.
    pub fn table_id(&self) -> u32 {
        self.table_id
    }

    /// Returns the last commit_version seen by this source.
    pub fn last_version(&self) -> u64 {
        self.last_version.load(Ordering::Acquire)
    }

    /// Pulls up to `max_rows` new change records from the CDF, advancing the
    /// last_version watermark. Returns an empty Vec if no new records exist.
    pub fn read_batch(&self, max_rows: usize) -> Result<Vec<CdfChange>> {
        if max_rows == 0 {
            return Ok(Vec::new());
        }
        let start = self.last_version.load(Ordering::Acquire).saturating_add(1);
        let records = self.cdf.query_changes(start, u64::MAX)?;

        if records.is_empty() {
            return Ok(Vec::new());
        }

        let mut out: Vec<CdfChange> = Vec::with_capacity(records.len().min(max_rows));
        let mut max_seen = self.last_version.load(Ordering::Acquire);
        for rec in records.into_iter().take(max_rows) {
            if rec.commit_version > max_seen {
                max_seen = rec.commit_version;
            }
            out.push(CdfChange {
                commit_version: rec.commit_version,
                commit_timestamp: rec.commit_timestamp,
                change_type: rec.change_type,
                row_data: rec.row_data,
                primary_key_data: rec.primary_key_data,
            });
        }
        self.last_version.store(max_seen, Ordering::Release);
        Ok(out)
    }
}

// ---------------------------------------------------------------------------
// ZyronSourceAdapter trait
// ---------------------------------------------------------------------------

/// Adapter trait for sources that pull rows from a remote Zyron publication
/// and forward them to a streaming job as CdfChange batches. The concrete
/// implementation lives in the zyron-wire crate as ZyronSourceClient. The
/// trait object form lets the runner dispatch to the remote source without
/// taking a build dependency on zyron-wire.
#[async_trait::async_trait]
pub trait ZyronSourceAdapter: Send + Sync {
    /// Runs the source loop. The adapter pulls rows from its remote, converts
    /// them to CdfChange batches, and invokes on_batch for each batch. The
    /// loop exits when shutdown is set or the remote signals end-of-stream.
    /// start_lsn is the LSN to resume from, zero means start from earliest.
    async fn run(
        &self,
        start_lsn: u64,
        on_batch: Box<dyn Fn(Vec<CdfChange>) -> zyron_common::Result<()> + Send + Sync>,
        shutdown: std::sync::Arc<std::sync::atomic::AtomicBool>,
    ) -> zyron_common::Result<()>;

    /// Closes the adapter and releases any held resources.
    async fn close(&self) -> zyron_common::Result<()>;
}
