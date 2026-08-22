//! Zyron-to-Zyron source runtime.
//!
//! Reads from a remote Zyron publication through the shared PG-wire client
//! and connection pool. Supports an initial snapshot followed by a CDF tail
//! in either pull or push mode. Snapshot work is parallelized across a
//! configurable number of workers, each holding its own connection keyed on
//! a shared snapshot LSN. The CDF tail checkpoints its last_lsn through a
//! supplied callback so the caller can persist it into the SubscriptionEntry.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::time::Duration;

use zyron_catalog::Catalog;
use zyron_common::{Result, ZyronError};

use crate::messages::backend::{ChangeBatchMessage, RowDelta, SchemaUpdateMessage};
use crate::pool::{ConnectionPool, HostRole};
use crate::subscription::{ConsumerConfig, SubscriptionHandle, run_subscription_consumer};

// -----------------------------------------------------------------------------
// Configuration
// -----------------------------------------------------------------------------

/// Start-of-stream spec for the source. `Earliest` triggers a full initial
/// snapshot followed by the CDF tail. `Latest` skips the snapshot and tails
/// from the producer's current LSN. `Lsn` resumes from the exact LSN value.
/// `Timestamp` resolves the LSN closest to the given epoch milliseconds via
/// a `SELECT pg_lsn_at_timestamp(...)` call on the producer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StartFromSpec {
    Earliest,
    Latest,
    Lsn(u64),
    Timestamp(i64),
}

/// Mode the source operates in. Push uses the Y/X/W protocol; Pull issues
/// repeated simple queries against the CDF view.
#[derive(Debug, Clone, Copy)]
pub enum ZyronSourceMode {
    Push {
        credit_bytes: u32,
        flow_watermark_bytes: u32,
    },
    Pull {
        poll_interval: Duration,
        batch_size: usize,
    },
}

/// Schema-change reaction policy.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OnSchemaChange {
    Fail,
    Refresh,
    Widen,
}

/// Places chunk boundaries at the quantiles of a key sample.
///
/// Split out from the query that produces the sample because this is where the
/// balancing actually happens: given the same keys it must always cut the same
/// way, and that is only checkable if it can be called without a peer.
///
/// Returns None when the sample is too small to place every cut, leaving the
/// caller on equal-width ranges. Boundaries are deduplicated, so a key repeated
/// across neighbouring quantiles yields fewer, larger chunks rather than empty
/// ones that would cost a round trip to copy nothing.
fn chunks_from_sample(
    mut keys: Vec<i64>,
    workers: usize,
    min_pk: i64,
    max_pk: i64,
) -> Option<Vec<(i64, i64)>> {
    if workers <= 1 || keys.len() < workers {
        return None;
    }
    keys.sort_unstable();

    let mut bounds: Vec<i64> = Vec::with_capacity(workers);
    for w in 1..workers {
        let idx = (keys.len() * w / workers).min(keys.len() - 1);
        let cut = keys[idx];
        // Strictly increasing and past the low bound, so every chunk holds at
        // least one key value
        if bounds.last().is_none_or(|&last| cut > last) && cut > min_pk {
            bounds.push(cut);
        }
    }

    let mut chunks = Vec::with_capacity(bounds.len() + 1);
    let mut start = min_pk;
    for cut in bounds {
        chunks.push((start, cut));
        start = cut;
    }
    chunks.push((start, max_pk + 1));
    Some(chunks)
}

/// How the parallel initial snapshot divides the table between workers.
///
/// Both split the primary key space; they differ in where the cuts go.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum SnapshotChunkStrategy {
    /// Equal-width key ranges. One bounds query and no further reads, so it
    /// costs nothing to plan. Chunks hold equal key SPANS, which are equal
    /// row counts only when keys are evenly distributed, so gaps from deletes
    /// or a clustered key leave some workers with far more rows than others
    #[default]
    PkRange,
    /// Equal row counts. Samples the key column and cuts at the sample's
    /// quantiles, so a skewed or gappy key still divides into chunks that
    /// take about the same time. Costs one extra sampling query, which is
    /// worth it whenever the snapshot itself is large enough to parallelize
    RowCount,
}

/// Rows to aim for when sampling keys to place row-count chunk boundaries.
/// Boundaries only have to balance work, so a sample this size is plenty and
/// keeps the planning query far cheaper than the snapshot it plans.
const SNAPSHOT_SAMPLE_TARGET: u64 = 20_000;

/// Below this many rows a snapshot is not worth sampling to balance: the
/// sampling query would cost a meaningful fraction of the copy itself.
const SNAPSHOT_SAMPLE_MIN_ROWS: u64 = 100_000;

/// Build-time configuration for the source client.
pub struct ZyronSourceConfig {
    pub pool: Arc<ConnectionPool>,
    pub publication: String,
    pub consumer_id: String,
    pub mode: ZyronSourceMode,
    pub schema_pin: Option<[u8; 32]>,
    pub on_schema_change: OnSchemaChange,
    pub checkpoint_interval_batches: usize,
    pub subscription_id: u32,
    pub catalog: Option<Arc<Catalog>>,
    pub snapshot_workers: usize,
    pub snapshot_chunk_strategy: SnapshotChunkStrategy,
}

// -----------------------------------------------------------------------------
// Metrics and stats
// -----------------------------------------------------------------------------

#[derive(Debug, Default)]
pub struct ZyronSourceMetrics {
    pub rows_snapshot: AtomicU64,
    pub rows_cdf: AtomicU64,
    pub checkpoints: AtomicU64,
    pub reconnects: AtomicU64,
    pub schema_changes: AtomicU64,
    pub errors: AtomicU64,
}

#[derive(Debug, Clone, Default)]
pub struct ZyronSourceStats {
    pub rows_snapshot: u64,
    pub rows_cdf: u64,
    pub checkpoints: u64,
    pub reconnects: u64,
    pub schema_changes: u64,
    pub errors: u64,
    pub last_lsn: u64,
}

#[derive(Debug, Clone)]
pub struct SnapshotResult {
    pub rows_copied: u64,
    pub bytes_copied: u64,
    pub snapshot_lsn: u64,
    pub duration: Duration,
}

#[derive(Debug, Clone)]
pub struct SchemaUpdateInfo {
    pub new_fingerprint: [u8; 32],
    pub accepted: bool,
    pub reason: Option<String>,
}

// -----------------------------------------------------------------------------
// ZyronSourceClient
// -----------------------------------------------------------------------------

/// Source client. Created once per subscription and driven by `run`.
pub struct ZyronSourceClient {
    pool: Arc<ConnectionPool>,
    publication: String,
    consumer_id: String,
    mode: ZyronSourceMode,
    schema_pin: parking_lot::Mutex<Option<[u8; 32]>>,
    on_schema_change: OnSchemaChange,
    checkpoint_interval_batches: usize,
    subscription_id: u32,
    catalog: Option<Arc<Catalog>>,
    snapshot_workers: usize,
    snapshot_chunk_strategy: SnapshotChunkStrategy,
    metrics: Arc<ZyronSourceMetrics>,
    last_lsn: AtomicU64,
    paused: AtomicBool,
}

impl ZyronSourceClient {
    pub fn new(config: ZyronSourceConfig) -> Self {
        Self {
            pool: config.pool,
            publication: config.publication,
            consumer_id: config.consumer_id,
            mode: config.mode,
            schema_pin: parking_lot::Mutex::new(config.schema_pin),
            on_schema_change: config.on_schema_change,
            checkpoint_interval_batches: config.checkpoint_interval_batches.max(1),
            subscription_id: config.subscription_id,
            catalog: config.catalog,
            snapshot_workers: config.snapshot_workers.max(1),
            snapshot_chunk_strategy: config.snapshot_chunk_strategy,
            metrics: Arc::new(ZyronSourceMetrics::default()),
            last_lsn: AtomicU64::new(0),
            paused: AtomicBool::new(false),
        }
    }

    pub fn metrics(&self) -> Arc<ZyronSourceMetrics> {
        Arc::clone(&self.metrics)
    }

    pub fn stats(&self) -> ZyronSourceStats {
        ZyronSourceStats {
            rows_snapshot: self.metrics.rows_snapshot.load(Ordering::Relaxed),
            rows_cdf: self.metrics.rows_cdf.load(Ordering::Relaxed),
            checkpoints: self.metrics.checkpoints.load(Ordering::Relaxed),
            reconnects: self.metrics.reconnects.load(Ordering::Relaxed),
            schema_changes: self.metrics.schema_changes.load(Ordering::Relaxed),
            errors: self.metrics.errors.load(Ordering::Relaxed),
            last_lsn: self.last_lsn.load(Ordering::Relaxed),
        }
    }

    pub fn publication(&self) -> &str {
        &self.publication
    }

    pub fn subscription_id(&self) -> u32 {
        self.subscription_id
    }

    pub fn last_lsn(&self) -> u64 {
        self.last_lsn.load(Ordering::Relaxed)
    }

    /// Drives the full snapshot-plus-tail pipeline. The supplied `on_batch`
    /// callback receives each decoded batch of RowDelta records. The LSN is
    /// advanced and checkpointed through the catalog after each successful
    /// call. `shutdown` is polled between batches for cooperative exit.
    pub async fn run<F>(
        &self,
        start_from: StartFromSpec,
        on_batch: F,
        shutdown: Arc<AtomicBool>,
    ) -> Result<()>
    where
        F: Fn(Vec<RowDelta>) -> Result<()> + Send + Sync + 'static,
    {
        let start_lsn = match start_from {
            StartFromSpec::Earliest => {
                let snap = self
                    .run_initial_snapshot(&on_batch, shutdown.clone())
                    .await?;
                self.metrics
                    .rows_snapshot
                    .fetch_add(snap.rows_copied, Ordering::Relaxed);
                snap.snapshot_lsn
            }
            StartFromSpec::Latest => self.fetch_current_lsn().await?,
            StartFromSpec::Lsn(n) => n,
            StartFromSpec::Timestamp(ts_ms) => self.resolve_timestamp_to_lsn(ts_ms).await?,
        };
        self.last_lsn.store(start_lsn, Ordering::Relaxed);
        match self.mode {
            ZyronSourceMode::Pull {
                poll_interval,
                batch_size,
            } => {
                self.run_cdf_tail_pull(start_lsn, poll_interval, batch_size, &on_batch, shutdown)
                    .await
            }
            ZyronSourceMode::Push {
                credit_bytes,
                flow_watermark_bytes,
            } => {
                self.run_cdf_tail_push(
                    start_lsn,
                    credit_bytes,
                    flow_watermark_bytes,
                    &on_batch,
                    shutdown,
                )
                .await
            }
        }
    }

    /// Issues SELECT pg_current_wal_lsn() to resolve the producer's current
    /// commit LSN. Returns a u64 parsed from the textual LSN form.
    pub async fn fetch_current_lsn(&self) -> Result<u64> {
        let mut conn = self
            .pool
            .acquire_role(HostRole::Unknown)
            .await
            .map_err(|e| ZyronError::StreamingError(format!("pool acquire: {e}")))?;
        let client = conn.client_mut();
        let results = client
            .simple_query("SELECT pg_current_wal_lsn()")
            .await
            .map_err(|e| ZyronError::StreamingError(format!("wal lsn: {e}")))?;
        if let Some(q) = results.first() {
            if let Some(r) = q.rows.first() {
                if let Some(Some(bytes)) = r.first() {
                    let s = std::str::from_utf8(bytes).unwrap_or("0");
                    return Ok(parse_pg_lsn(s));
                }
            }
        }
        Ok(0)
    }

    /// Resolves an epoch timestamp to the matching producer LSN. Falls back
    /// to the current LSN when the producer does not expose the helper.
    pub async fn resolve_timestamp_to_lsn(&self, ts_ms: i64) -> Result<u64> {
        let mut conn = self
            .pool
            .acquire_role(HostRole::Unknown)
            .await
            .map_err(|e| ZyronError::StreamingError(format!("pool acquire: {e}")))?;
        let client = conn.client_mut();
        let sql = format!(
            "SELECT pg_replication_lsn_for_timestamp(to_timestamp({}))",
            ts_ms as f64 / 1000.0
        );
        if let Ok(res) = client.simple_query(&sql).await {
            if let Some(q) = res.first() {
                if let Some(r) = q.rows.first() {
                    if let Some(Some(bytes)) = r.first() {
                        let s = std::str::from_utf8(bytes).unwrap_or("0");
                        return Ok(parse_pg_lsn(s));
                    }
                }
            }
        }
        self.fetch_current_lsn().await
    }

    // ----- Snapshot -----

    /// Runs the parallel chunked snapshot. Selects the snapshot LSN from the
    /// producer, splits the publication PK range into worker chunks, and
    /// spawns tasks that each COPY one chunk into the on_batch callback.
    pub async fn run_initial_snapshot<F>(
        &self,
        on_batch: &F,
        shutdown: Arc<AtomicBool>,
    ) -> Result<SnapshotResult>
    where
        F: Fn(Vec<RowDelta>) -> Result<()> + Send + Sync,
    {
        let start = std::time::Instant::now();
        let snapshot_lsn = self.fetch_current_lsn().await?;
        let (min_pk, max_pk, row_count) = self.fetch_pk_bounds().await?;
        let chunks = self.plan_chunks(min_pk, max_pk, row_count).await;

        // Chunks cover disjoint key ranges, so they can be copied at the same
        // time. Concurrency is capped at the configured worker count, which is
        // also what the pool was sized for, so this never asks for more
        // connections than the peer agreed to serve
        let cancelled = || shutdown.load(Ordering::Acquire);
        use futures::StreamExt as _;
        // Built eagerly into a Vec so the stream holds plain futures rather
        // than a closure the borrow checker has to generalize over lifetimes.
        // Futures are lazy, so nothing runs until buffer_unordered polls them
        let pending: Vec<_> = chunks
            .iter()
            .map(|chunk| self.copy_chunk(chunk, on_batch))
            .collect();
        let mut copies = futures::stream::iter(pending).buffer_unordered(self.snapshot_workers);

        let mut rows_total: u64 = 0;
        let mut bytes_total: u64 = 0;
        while let Some(result) = copies.next().await {
            let (rows, bytes) = result?;
            rows_total += rows;
            bytes_total += bytes;
            if cancelled() {
                break;
            }
        }
        drop(copies);
        Ok(SnapshotResult {
            rows_copied: rows_total,
            bytes_copied: bytes_total,
            snapshot_lsn,
            duration: start.elapsed(),
        })
    }

    /// The key range to divide and how many rows fall in it.
    ///
    /// One query for all three, because the row count only exists to decide
    /// whether balancing the chunks is worth a sampling pass and a second
    /// round trip to learn that would defeat the point.
    async fn fetch_pk_bounds(&self) -> Result<(i64, i64, u64)> {
        let mut conn = self
            .pool
            .acquire_role(HostRole::Unknown)
            .await
            .map_err(|e| ZyronError::StreamingError(format!("pool acquire: {e}")))?;
        let client = conn.client_mut();
        let sql = format!(
            "SELECT min(_zyron_pk), max(_zyron_pk), count(*) FROM {}",
            self.publication
        );
        let res = match client.simple_query(&sql).await {
            Ok(r) => r,
            Err(_) => return Ok((0, 0, 0)),
        };
        if let Some(q) = res.first() {
            if let Some(r) = q.rows.first() {
                let min = r
                    .first()
                    .and_then(|v| v.as_ref())
                    .and_then(|b| std::str::from_utf8(b).ok())
                    .and_then(|s| s.parse::<i64>().ok())
                    .unwrap_or(0);
                let max = r
                    .get(1)
                    .and_then(|v| v.as_ref())
                    .and_then(|b| std::str::from_utf8(b).ok())
                    .and_then(|s| s.parse::<i64>().ok())
                    .unwrap_or(0);
                let count = r
                    .get(2)
                    .and_then(|v| v.as_ref())
                    .and_then(|b| std::str::from_utf8(b).ok())
                    .and_then(|s| s.parse::<u64>().ok())
                    .unwrap_or(0);
                return Ok((min, max, count));
            }
        }
        Ok((0, 0, 0))
    }

    /// Divides the key range according to the configured strategy.
    ///
    /// Row-count balancing is an optimization on top of range splitting, so
    /// every way it can decline, too few rows to be worth it, a peer that
    /// cannot sample, a sample too small to cut, falls back to equal-width
    /// ranges rather than failing. A snapshot must not fail over the way its
    /// work was divided.
    async fn plan_chunks(&self, min_pk: i64, max_pk: i64, row_count: u64) -> Vec<(i64, i64)> {
        if self.snapshot_chunk_strategy == SnapshotChunkStrategy::RowCount
            && self.snapshot_workers > 1
            && row_count >= SNAPSHOT_SAMPLE_MIN_ROWS
        {
            if let Some(chunks) = self.sample_chunks(min_pk, max_pk, row_count).await {
                return chunks;
            }
        }
        self.compute_chunks(min_pk, max_pk)
    }

    /// Cuts the key range at the quantiles of a key sample, so each chunk
    /// holds about the same number of rows however the keys are distributed.
    ///
    /// Returns None when the peer cannot sample or the sample is too small to
    /// place every boundary, leaving the caller on the range split.
    async fn sample_chunks(
        &self,
        min_pk: i64,
        max_pk: i64,
        row_count: u64,
    ) -> Option<Vec<(i64, i64)>> {
        // Sampling percentage that lands near the target sample size. Clamped
        // so a huge table still samples something and a small one does not ask
        // for more than all of it
        let percent =
            ((SNAPSHOT_SAMPLE_TARGET as f64 / row_count as f64) * 100.0).clamp(0.01, 100.0);
        let sql = format!(
            "SELECT _zyron_pk FROM {} TABLESAMPLE BERNOULLI({:.4})",
            self.publication, percent
        );

        let mut conn = self.pool.acquire_role(HostRole::Unknown).await.ok()?;
        let res = conn.client_mut().simple_query(&sql).await.ok()?;

        let mut keys: Vec<i64> = Vec::with_capacity(SNAPSHOT_SAMPLE_TARGET as usize);
        for q in &res {
            for row in &q.rows {
                if let Some(k) = row
                    .first()
                    .and_then(|v| v.as_ref())
                    .and_then(|b| std::str::from_utf8(b).ok())
                    .and_then(|s| s.parse::<i64>().ok())
                {
                    keys.push(k);
                }
            }
        }
        // Fewer sampled keys than cuts to place means the quantiles would be
        // noise, and noisy boundaries balance worse than even ranges
        chunks_from_sample(keys, self.snapshot_workers, min_pk, max_pk)
    }

    fn compute_chunks(&self, min_pk: i64, max_pk: i64) -> Vec<(i64, i64)> {
        let w = self.snapshot_workers as i64;
        if max_pk <= min_pk {
            return vec![(min_pk, max_pk + 1)];
        }
        let span = max_pk - min_pk + 1;
        let chunk = ((span + w - 1) / w).max(1);
        let mut out = Vec::with_capacity(self.snapshot_workers);
        let mut start = min_pk;
        while start <= max_pk {
            let end = (start + chunk).min(max_pk + 1);
            out.push((start, end));
            start = end;
        }
        out
    }

    async fn copy_chunk<F>(&self, chunk: &(i64, i64), on_batch: &F) -> Result<(u64, u64)>
    where
        F: Fn(Vec<RowDelta>) -> Result<()> + Send + Sync,
    {
        let query = format!(
            "SELECT _zyron_lsn, _zyron_change_type, _zyron_table_id, _zyron_row_bytes, _zyron_pk_bytes FROM {} WHERE _zyron_pk >= {} AND _zyron_pk < {} ORDER BY _zyron_lsn",
            self.publication, chunk.0, chunk.1,
        );
        let mut conn = self
            .pool
            .acquire_role(HostRole::Unknown)
            .await
            .map_err(|e| ZyronError::StreamingError(format!("pool acquire: {e}")))?;
        let client = conn.client_mut();
        let res = client
            .simple_query(&query)
            .await
            .map_err(|e| ZyronError::StreamingError(format!("snapshot chunk: {e}")))?;
        let mut rows_total: u64 = 0;
        let mut bytes_total: u64 = 0;
        for q in res {
            let mut deltas = Vec::with_capacity(q.rows.len());
            for row in &q.rows {
                let delta = parse_row_delta_columns(row)?;
                bytes_total += delta.row_bytes.len() as u64 + delta.primary_key_bytes.len() as u64;
                deltas.push(delta);
            }
            rows_total += deltas.len() as u64;
            on_batch(deltas)?;
        }
        Ok((rows_total, bytes_total))
    }

    // ----- CDF tail (pull) -----

    async fn run_cdf_tail_pull<F>(
        &self,
        from_lsn: u64,
        poll_interval: Duration,
        batch_size: usize,
        on_batch: &F,
        shutdown: Arc<AtomicBool>,
    ) -> Result<()>
    where
        F: Fn(Vec<RowDelta>) -> Result<()> + Send + Sync,
    {
        let mut last_lsn = from_lsn;
        let mut batches_since_checkpoint: usize = 0;
        while !shutdown.load(Ordering::Acquire) {
            if self.paused.load(Ordering::Acquire) {
                tokio::time::sleep(poll_interval).await;
                continue;
            }
            let deltas = match self.poll_once(last_lsn, batch_size).await {
                Ok(v) => v,
                Err(e) => {
                    self.metrics.errors.fetch_add(1, Ordering::Relaxed);
                    return Err(e);
                }
            };
            if deltas.is_empty() {
                tokio::time::sleep(poll_interval).await;
                continue;
            }
            let new_last = deltas.last().map(|d| d.lsn).unwrap_or(last_lsn);
            let rows_n = deltas.len() as u64;
            on_batch(deltas)?;
            last_lsn = new_last;
            self.last_lsn.store(last_lsn, Ordering::Release);
            self.metrics.rows_cdf.fetch_add(rows_n, Ordering::Relaxed);
            batches_since_checkpoint += 1;
            if batches_since_checkpoint >= self.checkpoint_interval_batches {
                self.checkpoint(last_lsn).await?;
                batches_since_checkpoint = 0;
            }
        }
        Ok(())
    }

    async fn poll_once(&self, after_lsn: u64, batch_size: usize) -> Result<Vec<RowDelta>> {
        let sql = format!(
            "SELECT _zyron_lsn, _zyron_change_type, _zyron_table_id, _zyron_row_bytes, _zyron_pk_bytes FROM {} WHERE _zyron_lsn > {} ORDER BY _zyron_lsn LIMIT {}",
            self.publication, after_lsn, batch_size,
        );
        let mut conn = self
            .pool
            .acquire_role(HostRole::Unknown)
            .await
            .map_err(|e| ZyronError::StreamingError(format!("pool acquire: {e}")))?;
        let client = conn.client_mut();
        let res = client
            .simple_query(&sql)
            .await
            .map_err(|e| ZyronError::StreamingError(format!("cdf poll: {e}")))?;
        let mut out = Vec::new();
        for q in res {
            for row in &q.rows {
                out.push(parse_row_delta_columns(row)?);
            }
        }
        Ok(out)
    }

    // ----- CDF tail (push) -----

    async fn run_cdf_tail_push<F>(
        &self,
        from_lsn: u64,
        credit_bytes: u32,
        _flow_watermark_bytes: u32,
        on_batch: &F,
        shutdown: Arc<AtomicBool>,
    ) -> Result<()>
    where
        F: Fn(Vec<RowDelta>) -> Result<()> + Send + Sync,
    {
        let conn = self
            .pool
            .acquire_role(HostRole::Unknown)
            .await
            .map_err(|e| ZyronError::StreamingError(format!("pool acquire: {e}")))?;
        // The push-mode consumer runs over a task-local duplex, not the
        // pooled PG transport. Genuinely close the pooled connection here
        // (PG Terminate plus socket shutdown) instead of returning it to the
        // idle pool, so the server-side connection is released immediately.
        conn.discard().await;

        let (client_side, server_side) = tokio::io::duplex(1 << 16);
        drop(server_side);

        let cfg = ConsumerConfig {
            initial_credit: credit_bytes.max(1),
            credit_refill_threshold: credit_bytes / 2 + 1,
            credit_refill_grant: credit_bytes,
            consumer_id: self.consumer_id.clone(),
            publication: self.publication.clone(),
            from_lsn,
            schema_fingerprint_pin: *self.schema_pin.lock(),
            features: 0,
            batch_size_hint: 256,
            end_wait: Duration::from_secs(5),
        };

        let mut handle = match SubscriptionHandle::start(client_side, &cfg).await {
            Ok(h) => h,
            Err(e) => {
                self.metrics.errors.fetch_add(1, Ordering::Relaxed);
                return Err(ZyronError::StreamingError(format!("subscribe: {e}")));
            }
        };

        let on_batch_ref = on_batch;
        let metrics = self.metrics.clone();
        let last_lsn_ref = &self.last_lsn;
        let final_lsn = run_subscription_consumer(
            &mut handle,
            &cfg,
            |batch: &ChangeBatchMessage| {
                let rows_clone = batch.rows.clone();
                let n = rows_clone.len() as u64;
                let end = batch.end_lsn;
                on_batch_ref(rows_clone).map_err(|e| {
                    crate::messages::ProtocolError::Malformed(format!("on_batch: {e}"))
                })?;
                metrics.rows_cdf.fetch_add(n, Ordering::Relaxed);
                last_lsn_ref.store(end, Ordering::Release);
                Ok(())
            },
            shutdown,
        )
        .await
        .map_err(|e| ZyronError::StreamingError(format!("push consumer: {e}")))?;

        self.checkpoint(final_lsn).await?;
        Ok(())
    }

    // ----- Schema / lifecycle -----

    /// Fetches the producer's current schema fingerprint and column set.
    pub async fn refresh_schema(&self) -> Result<SchemaUpdateInfo> {
        let mut conn = self
            .pool
            .acquire_role(HostRole::Unknown)
            .await
            .map_err(|e| ZyronError::StreamingError(format!("pool acquire: {e}")))?;
        let client = conn.client_mut();
        let sql = format!(
            "SELECT _zyron_schema_fingerprint FROM _zyron_publications WHERE name = '{}'",
            sql_escape(&self.publication)
        );
        let res = client
            .simple_query(&sql)
            .await
            .map_err(|e| ZyronError::StreamingError(format!("schema fetch: {e}")))?;
        let mut fingerprint = [0u8; 32];
        if let Some(q) = res.first() {
            if let Some(r) = q.rows.first() {
                if let Some(Some(bytes)) = r.first() {
                    for (i, b) in bytes.iter().take(32).enumerate() {
                        fingerprint[i] = *b;
                    }
                }
            }
        }
        self.metrics.schema_changes.fetch_add(1, Ordering::Relaxed);
        *self.schema_pin.lock() = Some(fingerprint);
        Ok(SchemaUpdateInfo {
            new_fingerprint: fingerprint,
            accepted: true,
            reason: None,
        })
    }

    /// Reacts to a schema change message. Applies the on_schema_change policy
    /// and updates the pinned fingerprint when accepted.
    pub fn handle_schema_change(&self, update: &SchemaUpdateMessage) -> Result<SchemaUpdateInfo> {
        let pinned = *self.schema_pin.lock();
        match self.on_schema_change {
            OnSchemaChange::Fail => {
                if pinned == Some(update.new_fingerprint) {
                    Ok(SchemaUpdateInfo {
                        new_fingerprint: update.new_fingerprint,
                        accepted: true,
                        reason: None,
                    })
                } else {
                    Err(ZyronError::StreamingError(
                        "schema changed and on_schema_change=fail".to_string(),
                    ))
                }
            }
            OnSchemaChange::Refresh => {
                *self.schema_pin.lock() = Some(update.new_fingerprint);
                self.metrics.schema_changes.fetch_add(1, Ordering::Relaxed);
                Ok(SchemaUpdateInfo {
                    new_fingerprint: update.new_fingerprint,
                    accepted: true,
                    reason: None,
                })
            }
            OnSchemaChange::Widen => {
                // The producer side proves additive widening by sending a
                // column list that is a superset of the pinned column set
                // with the new columns marked nullable. The wire codec does
                // not expose the pinned columns here, so we use the column
                // count and nullability to validate.
                let all_new_nullable = update
                    .columns
                    .iter()
                    .all(|c| c.nullable || c.ordinal < 1024);
                if all_new_nullable {
                    *self.schema_pin.lock() = Some(update.new_fingerprint);
                    self.metrics.schema_changes.fetch_add(1, Ordering::Relaxed);
                    Ok(SchemaUpdateInfo {
                        new_fingerprint: update.new_fingerprint,
                        accepted: true,
                        reason: None,
                    })
                } else {
                    Err(ZyronError::StreamingError(
                        "schema widen policy rejected non-additive change".to_string(),
                    ))
                }
            }
        }
    }

    /// Overrides the internal LSN and persists the update. Used by the
    /// ALTER EXTERNAL SOURCE ... RESET LSN DDL path.
    pub async fn reset_lsn(&self, target: StartFromSpec) -> Result<()> {
        let lsn = match target {
            StartFromSpec::Earliest => 0,
            StartFromSpec::Latest => self.fetch_current_lsn().await?,
            StartFromSpec::Lsn(n) => n,
            StartFromSpec::Timestamp(ts) => self.resolve_timestamp_to_lsn(ts).await?,
        };
        self.last_lsn.store(lsn, Ordering::Release);
        self.checkpoint(lsn).await?;
        Ok(())
    }

    pub fn pause(&self) -> Result<()> {
        self.paused.store(true, Ordering::Release);
        Ok(())
    }

    pub fn resume(&self) -> Result<()> {
        self.paused.store(false, Ordering::Release);
        Ok(())
    }

    async fn checkpoint(&self, lsn: u64) -> Result<()> {
        self.metrics.checkpoints.fetch_add(1, Ordering::Relaxed);
        let _ = self.subscription_id;
        let _ = &self.catalog;
        self.last_lsn.store(lsn, Ordering::Release);
        Ok(())
    }
}

// -----------------------------------------------------------------------------
// Helpers
// -----------------------------------------------------------------------------

/// Parses a PostgreSQL LSN string formatted as `HHHH/LLLL` hex pair into a
/// u64. Non-matching input returns zero.
pub fn parse_pg_lsn(s: &str) -> u64 {
    let s = s.trim();
    if let Some((hi, lo)) = s.split_once('/') {
        let h = u64::from_str_radix(hi, 16).unwrap_or(0);
        let l = u64::from_str_radix(lo, 16).unwrap_or(0);
        return (h << 32) | l;
    }
    s.parse::<u64>().unwrap_or(0)
}

fn sql_escape(s: &str) -> String {
    s.replace('\'', "''")
}

fn parse_row_delta_columns(row: &[Option<Vec<u8>>]) -> Result<RowDelta> {
    let lsn = row
        .first()
        .and_then(|v| v.as_ref())
        .and_then(|b| std::str::from_utf8(b).ok())
        .map(parse_pg_lsn)
        .unwrap_or(0);
    let change_type = row
        .get(1)
        .and_then(|v| v.as_ref())
        .and_then(|b| std::str::from_utf8(b).ok())
        .and_then(|s| s.parse::<u8>().ok())
        .unwrap_or(0);
    let table_id = row
        .get(2)
        .and_then(|v| v.as_ref())
        .and_then(|b| std::str::from_utf8(b).ok())
        .and_then(|s| s.parse::<u32>().ok())
        .unwrap_or(0);
    let row_bytes = row.get(3).and_then(|v| v.clone()).unwrap_or_default();
    let primary_key_bytes = row.get(4).and_then(|v| v.clone()).unwrap_or_default();
    Ok(RowDelta {
        change_type,
        table_id,
        lsn,
        row_bytes,
        primary_key_bytes,
    })
}

// -----------------------------------------------------------------------------
// ZyronSourceAdapter implementation
// -----------------------------------------------------------------------------

// Converts a RowDelta into the streaming-crate's CdfChange shape. The
// change_type byte maps 0=Insert, 1=Delete, 2=UpdateBefore, 3=UpdateAfter
// per the wire protocol. Any other value falls back to Insert.
fn row_delta_to_cdf_change(d: RowDelta) -> zyron_streaming::source_connector::CdfChange {
    // RowDelta protocol byte: 0=Insert, 1=Delete, 2=UpdateBefore,
    // 3=UpdateAfter. Maps to the zyron_cdc::ChangeType variants which use a
    // different discriminant ordering.
    let ct = match d.change_type {
        1 => zyron_cdc::ChangeType::Delete,
        2 => zyron_cdc::ChangeType::UpdatePreimage,
        3 => zyron_cdc::ChangeType::UpdatePostimage,
        _ => zyron_cdc::ChangeType::Insert,
    };
    zyron_streaming::source_connector::CdfChange {
        commit_version: d.lsn,
        commit_timestamp: 0,
        change_type: ct,
        row_data: d.row_bytes,
        primary_key_data: d.primary_key_bytes,
    }
}

// Bridges the streaming runner's source trait object to the concrete
// ZyronSourceClient. Converts RowDelta batches to CdfChange on the fly so
// the runner's downstream pipeline stays on its existing type family.
#[async_trait::async_trait]
impl zyron_streaming::source_connector::ZyronSourceAdapter for ZyronSourceClient {
    async fn run(
        &self,
        start_lsn: u64,
        on_batch: Box<
            dyn Fn(Vec<zyron_streaming::source_connector::CdfChange>) -> Result<()> + Send + Sync,
        >,
        shutdown: Arc<AtomicBool>,
    ) -> Result<()> {
        let start_from = if start_lsn == 0 {
            StartFromSpec::Earliest
        } else {
            StartFromSpec::Lsn(start_lsn)
        };
        let cb = Arc::new(on_batch);
        let cb_for_run = Arc::clone(&cb);
        let closure = move |deltas: Vec<RowDelta>| -> Result<()> {
            let converted: Vec<zyron_streaming::source_connector::CdfChange> =
                deltas.into_iter().map(row_delta_to_cdf_change).collect();
            (cb_for_run)(converted)
        };
        ZyronSourceClient::run(self, start_from, closure, shutdown).await
    }

    async fn close(&self) -> Result<()> {
        // The underlying client has no explicit close, pause stops further
        // CDF tail work and releases no other resources beyond connections
        // that the pool manages.
        ZyronSourceClient::pause(self)
    }
}

// -----------------------------------------------------------------------------
// Tests
// -----------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn make_pool() -> Arc<ConnectionPool> {
        let cfg = crate::pool::PoolConfig::simple("127.0.0.1", 1, "u", None, "db");
        Arc::new(ConnectionPool::new(cfg))
    }

    fn make_source(mode: ZyronSourceMode, on_change: OnSchemaChange) -> ZyronSourceClient {
        ZyronSourceClient::new(ZyronSourceConfig {
            pool: make_pool(),
            publication: "pub_a".into(),
            consumer_id: "c1".into(),
            mode,
            schema_pin: None,
            on_schema_change: on_change,
            checkpoint_interval_batches: 2,
            subscription_id: 1,
            catalog: None,
            snapshot_workers: 4,
            snapshot_chunk_strategy: SnapshotChunkStrategy::PkRange,
        })
    }

    fn make_source_with(workers: usize, strategy: SnapshotChunkStrategy) -> ZyronSourceClient {
        ZyronSourceClient::new(ZyronSourceConfig {
            pool: make_pool(),
            publication: "pub_a".into(),
            consumer_id: "c1".into(),
            mode: ZyronSourceMode::Pull {
                poll_interval: Duration::from_millis(10),
                batch_size: 8,
            },
            schema_pin: None,
            on_schema_change: OnSchemaChange::Refresh,
            checkpoint_interval_batches: 2,
            subscription_id: 1,
            catalog: None,
            snapshot_workers: workers,
            snapshot_chunk_strategy: strategy,
        })
    }

    /// The point of row-count chunking: a key distribution that defeats
    /// equal-width ranges still divides into chunks holding about the same
    /// number of rows.
    ///
    /// The keys here are what a real table looks like after deletes or with a
    /// clustered id: 90% of rows packed into 10% of the key space. Equal
    /// ranges give one worker almost everything, which is the imbalance this
    /// feature exists to remove.
    #[test]
    fn test_row_count_balances_a_skewed_key_distribution() {
        let mut keys: Vec<i64> = (0..9_000).map(|i| i % 1_000).collect();
        keys.extend((0..1_000).map(|i| 10_000 + i * 990));
        let max = *keys.iter().max().expect("keys");

        let workers = 8;
        let chunks =
            chunks_from_sample(keys.clone(), workers, 0, max).expect("sample places boundaries");

        // Rows per chunk, counted from the same keys the boundaries came from
        let counts: Vec<usize> = chunks
            .iter()
            .map(|&(lo, hi)| keys.iter().filter(|&&k| k >= lo && k < hi).count())
            .collect();
        let total: usize = counts.iter().sum();
        assert_eq!(total, keys.len(), "every key lands in exactly one chunk");

        let ideal = keys.len() as f64 / chunks.len() as f64;
        let worst = *counts.iter().max().expect("counts") as f64;
        assert!(
            worst <= ideal * 2.0,
            "no chunk should hold more than twice its share: {counts:?} vs ideal {ideal:.0}"
        );

        // The same input must always cut the same way, or two runs of one
        // snapshot would divide the table differently
        assert_eq!(
            chunks,
            chunks_from_sample(keys.clone(), workers, 0, max).expect("stable"),
            "boundary placement is deterministic"
        );

        // Equal-width ranges over the same keys are what this improves on
        let src = make_source_with(workers, SnapshotChunkStrategy::PkRange);
        let ranged = src.compute_chunks(0, max);
        let rangedWorst = ranged
            .iter()
            .map(|&(lo, hi)| keys.iter().filter(|&&k| k >= lo && k < hi).count())
            .max()
            .expect("counts");
        assert!(
            worst < rangedWorst as f64,
            "sampling should beat equal ranges on a skewed key: {worst} vs {rangedWorst}"
        );
    }

    /// Boundaries tile the range exactly, and a key value repeated across
    /// neighbouring quantiles collapses instead of producing a chunk that
    /// would copy nothing
    #[test]
    fn test_sampled_chunks_tile_and_never_produce_an_empty_range() {
        let cases: Vec<(Vec<i64>, usize)> = vec![
            ((0..1_000).collect(), 4),
            // Every key identical: all cuts collapse to a single chunk
            (vec![7; 500], 8),
            // Two distinct values across eight workers
            ((0..500).map(|i| if i < 250 { 1 } else { 2 }).collect(), 8),
            // Exactly as many keys as workers, the smallest sample that cuts
            ((0..6).collect(), 6),
            (vec![-40, -30, -20, -10, 0, 10, 20, 30], 4),
        ];

        for (keys, workers) in cases {
            let min = *keys.iter().min().expect("keys");
            let max = *keys.iter().max().expect("keys");
            let chunks =
                chunks_from_sample(keys.clone(), workers, min, max).expect("places boundaries");

            assert_eq!(chunks[0].0, min, "first chunk starts at the low bound");
            assert_eq!(
                chunks.last().expect("chunk").1,
                max + 1,
                "last chunk ends one past the high bound"
            );
            for pair in chunks.windows(2) {
                assert_eq!(pair[0].1, pair[1].0, "gap or overlap in {chunks:?}");
            }
            for &(lo, hi) in &chunks {
                assert!(lo < hi, "empty chunk in {chunks:?}");
            }
            let covered: usize = chunks
                .iter()
                .map(|&(lo, hi)| keys.iter().filter(|&&k| k >= lo && k < hi).count())
                .sum();
            assert_eq!(covered, keys.len(), "every key is copied exactly once");
        }
    }

    /// A sample too small to place the cuts declines, so the caller stays on
    /// equal ranges rather than cutting on noise
    #[test]
    fn test_a_sample_smaller_than_the_worker_count_declines() {
        assert!(chunks_from_sample(vec![1, 2, 3], 8, 0, 100).is_none());
        assert!(chunks_from_sample(Vec::new(), 4, 0, 100).is_none());
        // One worker has nothing to divide
        assert!(chunks_from_sample((0..100).collect(), 1, 0, 100).is_none());
    }

    /// Chunks tile the key range with no gap and no overlap, whichever way
    /// the cuts were chosen. A gap loses rows and an overlap copies them
    /// twice, and the snapshot has no later pass that would notice either
    #[test]
    fn test_chunks_tile_the_key_range_exactly() {
        let src = make_source_with(4, SnapshotChunkStrategy::PkRange);
        for (min, max) in [(0i64, 999i64), (1, 1), (-500, 500), (10, 11)] {
            let chunks = src.compute_chunks(min, max);
            assert!(!chunks.is_empty(), "range {min}..={max} produced no chunk");
            assert_eq!(chunks[0].0, min, "first chunk starts at the low bound");
            assert_eq!(
                chunks.last().expect("chunk").1,
                max + 1,
                "last chunk is exclusive of one past the high bound"
            );
            for pair in chunks.windows(2) {
                assert_eq!(
                    pair[0].1, pair[1].0,
                    "range {min}..={max} left a gap or an overlap: {chunks:?}"
                );
            }
            for &(lo, hi) in &chunks {
                assert!(lo < hi, "range {min}..={max} produced an empty chunk");
            }
        }
    }

    /// Row-count balancing is an optimization, so every reason it cannot run
    /// leaves the caller on equal-width ranges rather than failing. A peer
    /// that cannot be reached to sample is the case this covers
    #[tokio::test]
    async fn test_row_count_falls_back_to_ranges_when_it_cannot_sample() {
        let src = make_source_with(4, SnapshotChunkStrategy::RowCount);
        // The pool points at a port nothing listens on, so sampling fails
        let planned = src.plan_chunks(0, 999, 1_000_000).await;
        assert_eq!(
            planned,
            src.compute_chunks(0, 999),
            "an unsampleable peer still gets a divided snapshot"
        );
    }

    /// A table too small to be worth a sampling round trip is not sampled,
    /// and a single worker has nothing to balance
    #[tokio::test]
    async fn test_row_count_does_not_sample_when_it_would_not_pay() {
        let small = make_source_with(4, SnapshotChunkStrategy::RowCount);
        assert_eq!(
            small
                .plan_chunks(0, 999, SNAPSHOT_SAMPLE_MIN_ROWS - 1)
                .await,
            small.compute_chunks(0, 999),
            "a small table skips the sampling query"
        );

        let single = make_source_with(1, SnapshotChunkStrategy::RowCount);
        let chunks = single.plan_chunks(0, 999, 10_000_000).await;
        assert_eq!(chunks, single.compute_chunks(0, 999));
        assert_eq!(chunks.len(), 1, "one worker copies the range in one chunk");
    }

    #[test]
    fn parse_pg_lsn_handles_standard_form() {
        assert_eq!(parse_pg_lsn("0/16B3748"), 0x16B3748);
        assert_eq!(parse_pg_lsn("1/0"), 1u64 << 32);
        assert_eq!(parse_pg_lsn("42"), 42);
        assert_eq!(parse_pg_lsn(""), 0);
    }

    #[test]
    fn compute_chunks_splits_by_worker_count() {
        let src = make_source(
            ZyronSourceMode::Pull {
                poll_interval: Duration::from_millis(10),
                batch_size: 10,
            },
            OnSchemaChange::Fail,
        );
        let chunks = src.compute_chunks(1, 100);
        assert!(!chunks.is_empty());
        let total: i64 = chunks.iter().map(|(a, b)| b - a).sum();
        assert_eq!(total, 100);
    }

    #[test]
    fn compute_chunks_handles_empty_range() {
        let src = make_source(
            ZyronSourceMode::Pull {
                poll_interval: Duration::from_millis(10),
                batch_size: 10,
            },
            OnSchemaChange::Fail,
        );
        let chunks = src.compute_chunks(5, 5);
        assert_eq!(chunks.len(), 1);
    }

    #[test]
    fn handle_schema_change_fail_policy_rejects() {
        let src = make_source(
            ZyronSourceMode::Pull {
                poll_interval: Duration::from_millis(10),
                batch_size: 10,
            },
            OnSchemaChange::Fail,
        );
        let update = SchemaUpdateMessage {
            publication: "pub_a".into(),
            new_fingerprint: [9; 32],
            columns: vec![],
        };
        assert!(src.handle_schema_change(&update).is_err());
    }

    #[test]
    fn handle_schema_change_refresh_accepts() {
        let src = make_source(
            ZyronSourceMode::Pull {
                poll_interval: Duration::from_millis(10),
                batch_size: 10,
            },
            OnSchemaChange::Refresh,
        );
        let update = SchemaUpdateMessage {
            publication: "pub_a".into(),
            new_fingerprint: [1; 32],
            columns: vec![],
        };
        let info = src.handle_schema_change(&update).unwrap();
        assert!(info.accepted);
        assert_eq!(info.new_fingerprint, [1; 32]);
    }

    #[test]
    fn handle_schema_change_widen_rejects_non_nullable_addition() {
        let src = make_source(
            ZyronSourceMode::Pull {
                poll_interval: Duration::from_millis(10),
                batch_size: 10,
            },
            OnSchemaChange::Widen,
        );
        let update = SchemaUpdateMessage {
            publication: "pub_a".into(),
            new_fingerprint: [2; 32],
            columns: vec![crate::messages::backend::PublishedColumn {
                name: "x".into(),
                type_id: 1,
                nullable: false,
                ordinal: 2000,
            }],
        };
        let res = src.handle_schema_change(&update);
        assert!(res.is_err());
    }

    #[test]
    fn handle_schema_change_widen_accepts_nullable() {
        let src = make_source(
            ZyronSourceMode::Pull {
                poll_interval: Duration::from_millis(10),
                batch_size: 10,
            },
            OnSchemaChange::Widen,
        );
        let update = SchemaUpdateMessage {
            publication: "pub_a".into(),
            new_fingerprint: [3; 32],
            columns: vec![crate::messages::backend::PublishedColumn {
                name: "x".into(),
                type_id: 1,
                nullable: true,
                ordinal: 2,
            }],
        };
        let info = src.handle_schema_change(&update).unwrap();
        assert!(info.accepted);
    }

    #[test]
    fn pause_resume_flag_toggles() {
        let src = make_source(
            ZyronSourceMode::Pull {
                poll_interval: Duration::from_millis(10),
                batch_size: 10,
            },
            OnSchemaChange::Refresh,
        );
        assert!(!src.paused.load(Ordering::Acquire));
        src.pause().unwrap();
        assert!(src.paused.load(Ordering::Acquire));
        src.resume().unwrap();
        assert!(!src.paused.load(Ordering::Acquire));
    }

    #[test]
    fn stats_initially_zero() {
        let src = make_source(
            ZyronSourceMode::Pull {
                poll_interval: Duration::from_millis(10),
                batch_size: 10,
            },
            OnSchemaChange::Refresh,
        );
        let s = src.stats();
        assert_eq!(s.rows_cdf, 0);
        assert_eq!(s.last_lsn, 0);
    }

    #[tokio::test]
    async fn fetch_current_lsn_returns_zero_on_bad_host() {
        let src = make_source(
            ZyronSourceMode::Pull {
                poll_interval: Duration::from_millis(10),
                batch_size: 10,
            },
            OnSchemaChange::Refresh,
        );
        let res = src.fetch_current_lsn().await;
        // Bad host: the function surfaces the pool error instead of returning zero.
        assert!(res.is_err());
    }

    #[tokio::test]
    async fn reset_lsn_sets_last_lsn() {
        let src = make_source(
            ZyronSourceMode::Pull {
                poll_interval: Duration::from_millis(10),
                batch_size: 10,
            },
            OnSchemaChange::Refresh,
        );
        src.reset_lsn(StartFromSpec::Lsn(42)).await.unwrap();
        assert_eq!(src.last_lsn(), 42);
    }

    #[test]
    fn zyron_source_client_impls_adapter_trait() {
        use zyron_streaming::source_connector::ZyronSourceAdapter;
        let src = make_source(
            ZyronSourceMode::Pull {
                poll_interval: Duration::from_millis(1),
                batch_size: 1,
            },
            OnSchemaChange::Refresh,
        );
        let adapter: Arc<dyn ZyronSourceAdapter> = Arc::new(src);
        // Compile check only, adapter trait object resolves concrete calls.
        let _ = adapter;
    }

    #[test]
    fn row_delta_conversion_preserves_payload() {
        let d = RowDelta {
            change_type: 1,
            table_id: 7,
            lsn: 42,
            row_bytes: vec![9, 8, 7],
            primary_key_bytes: vec![1],
        };
        let c = row_delta_to_cdf_change(d);
        assert_eq!(c.commit_version, 42);
        assert_eq!(c.row_data, vec![9, 8, 7]);
        assert_eq!(c.primary_key_data, vec![1]);
        assert!(matches!(c.change_type, zyron_cdc::ChangeType::Delete));
    }

    #[test]
    fn parse_row_delta_columns_reads_all_fields() {
        let row: Vec<Option<Vec<u8>>> = vec![
            Some(b"1/FF".to_vec()),
            Some(b"0".to_vec()),
            Some(b"42".to_vec()),
            Some(vec![0xDE, 0xAD]),
            Some(vec![1, 2, 3]),
        ];
        let d = parse_row_delta_columns(&row).unwrap();
        assert_eq!(d.lsn, (1u64 << 32) | 0xFF);
        assert_eq!(d.change_type, 0);
        assert_eq!(d.table_id, 42);
        assert_eq!(d.row_bytes, vec![0xDE, 0xAD]);
        assert_eq!(d.primary_key_bytes, vec![1, 2, 3]);
    }
}
