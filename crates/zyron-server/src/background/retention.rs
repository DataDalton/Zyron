//! Retention worker: enforces TTL, archive, and soft-delete purge by driving
//! real SQL through the planner + executor so legal holds, WORM/retention
//! locks, and the soft-delete rewrite are all honored automatically (the same
//! enforcement path interactive DML uses). It does not re-implement row
//! decoding or MVCC.
//!
//! Each cycle:
//!  1. Reloads active legal holds from the catalog into a lock-free registry.
//!  2. Loads retention policies; for each TTL policy resolves the table and
//!     runs `DELETE FROM t WHERE <col> < <cutoff>` (Archive first copies the
//!     matching rows to the configured object store, then deletes).
//!  3. Purges soft-deleted rows whose grace window elapsed via
//!     `DELETE ... HARD`, which bypasses the soft-delete rewrite.
//!  4. Records a retention job row and a tamper-evident compliance entry.
//!
//! Legal-hold / WORM violations surface as errors from the DML hook; the
//! worker logs and skips that table without aborting the whole cycle.

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, OnceLock};
use std::thread::{self, JoinHandle};
use std::time::Duration;

use tracing::{debug, info, warn};

use zyron_buffer::BufferPool;
use zyron_catalog::Catalog;
use zyron_storage::DiskManager;
use zyron_storage::txn::{IsolationLevel, TransactionManager};
use zyron_wal::WalWriter;

use crate::hooks::LegalHoldDmlHook;

#[derive(Debug, Clone)]
pub struct RetentionWorkerConfig {
    /// Seconds between retention cycles.
    pub interval_secs: u64,
}

impl Default for RetentionWorkerConfig {
    fn default() -> Self {
        Self { interval_secs: 60 }
    }
}

pub struct RetentionStats {
    pub cycles_completed: AtomicU64,
    pub rows_deleted: AtomicU64,
    pub rows_archived: AtomicU64,
    pub rows_purged: AtomicU64,
}

impl RetentionStats {
    fn new() -> Self {
        Self {
            cycles_completed: AtomicU64::new(0),
            rows_deleted: AtomicU64::new(0),
            rows_archived: AtomicU64::new(0),
            rows_purged: AtomicU64::new(0),
        }
    }
}

pub struct RetentionWorker {
    shutdown: Arc<AtomicBool>,
    waker: Arc<OnceLock<thread::Thread>>,
    thread: Option<JoinHandle<()>>,
    stats: Arc<RetentionStats>,
    /// Installed once the server state exists, which is after the workers
    /// start. The age-tiering pass drives the wire relocation and needs it,
    /// cycles before installation skip that pass
    server_state: Arc<OnceLock<Arc<zyron_wire::connection::ServerState>>>,
}

struct WorkerCtx {
    catalog: Arc<Catalog>,
    txn_manager: Arc<TransactionManager>,
    wal: Arc<WalWriter>,
    buffer_pool: Arc<BufferPool>,
    disk_manager: Arc<DiskManager>,
    legal_holds: Arc<zyron_lifecycle::legal_hold::LegalHoldRegistry>,
    server_state: Arc<OnceLock<Arc<zyron_wire::connection::ServerState>>>,
}

impl RetentionWorker {
    #[allow(clippy::too_many_arguments)]
    pub fn start(
        catalog: Arc<Catalog>,
        txn_manager: Arc<TransactionManager>,
        wal: Arc<WalWriter>,
        buffer_pool: Arc<BufferPool>,
        disk_manager: Arc<DiskManager>,
        config: RetentionWorkerConfig,
    ) -> Self {
        let shutdown = Arc::new(AtomicBool::new(false));
        let waker = Arc::new(OnceLock::new());
        let stats = Arc::new(RetentionStats::new());
        let server_state: Arc<OnceLock<Arc<zyron_wire::connection::ServerState>>> =
            Arc::new(OnceLock::new());

        let t_shutdown = Arc::clone(&shutdown);
        let t_waker = Arc::clone(&waker);
        let t_stats = Arc::clone(&stats);
        let t_server_state = Arc::clone(&server_state);

        let handle = thread::Builder::new()
            .name("zyron-retention".into())
            .spawn(move || {
                let _ = t_waker.set(thread::current());
                let runtime = match tokio::runtime::Builder::new_current_thread()
                    .enable_all()
                    .build()
                {
                    Ok(rt) => rt,
                    Err(e) => {
                        warn!("retention worker: runtime build failed: {e}");
                        return;
                    }
                };
                let wc = WorkerCtx {
                    catalog,
                    txn_manager,
                    wal,
                    buffer_pool,
                    disk_manager,
                    legal_holds: Arc::new(zyron_lifecycle::legal_hold::LegalHoldRegistry::new()),
                    server_state: t_server_state,
                };
                let interval = Duration::from_secs(config.interval_secs.max(1));
                loop {
                    thread::park_timeout(interval);
                    if t_shutdown.load(Ordering::Acquire) {
                        return;
                    }
                    runtime.block_on(Self::run_cycle(&wc, &t_stats));
                }
            })
            .expect("failed to spawn retention worker thread");

        Self {
            shutdown,
            waker,
            thread: Some(handle),
            stats,
            server_state,
        }
    }

    /// Installs the server state once it exists, enabling the age-tiering
    /// pass from the next cycle on
    pub fn install_server_state(&self, state: Arc<zyron_wire::connection::ServerState>) {
        let _ = self.server_state.set(state);
    }

    pub fn stats(&self) -> Arc<RetentionStats> {
        Arc::clone(&self.stats)
    }

    pub fn shutdown(&mut self) {
        self.shutdown.store(true, Ordering::Release);
        if let Some(t) = self.waker.get() {
            t.unpark();
        }
        if let Some(h) = self.thread.take() {
            let _ = h.join();
        }
    }

    async fn run_cycle(wc: &WorkerCtx, stats: &RetentionStats) {
        // Reload holds from the catalog (source of truth) so the DML hook
        // enforces the current state this cycle.
        if let Ok(holds) = wc.catalog.load_legal_holds().await {
            wc.legal_holds.reload(&holds);
        }

        let policies = match wc.catalog.load_retention_policies().await {
            Ok(p) => p,
            Err(e) => {
                debug!("retention: load policies failed: {e}");
                Vec::new()
            }
        };

        let now_us = now_micros();
        let mut deleted_total = 0u64;
        let mut archived_total = 0u64;

        for pol in policies.iter().filter(|p| p.kind == 0) {
            let table = match wc
                .catalog
                .get_table_by_id(zyron_catalog::TableId(pol.table_id))
            {
                Ok(t) => t,
                Err(_) => continue,
            };
            let lc = &table.lifecycle;
            // Resolve the comparison column and cutoff.
            let (col_id, cutoff) =
                if zyron_catalog::schema::LifecycleConfig::column_is_set(lc.retention_column_id) {
                    (lc.retention_column_id, now_us)
                } else if zyron_catalog::schema::LifecycleConfig::column_is_set(lc.ttl_column_id)
                    && lc.ttl_seconds > 0
                {
                    (
                        lc.ttl_column_id,
                        now_us - lc.ttl_seconds.saturating_mul(1_000_000),
                    )
                } else {
                    continue;
                };
            let col_name = match table.columns.iter().find(|c| c.id.0 as u32 == col_id) {
                Some(c) => c.name.clone(),
                None => continue,
            };
            let predicate = format!("\"{}\" < {}", col_name, cutoff);

            // Archive action: copy matching rows out before deleting.
            if pol.action == 1 && !lc.archive_destination.is_empty() {
                match Self::archive_matching(wc, &table.name, &predicate, &lc.archive_destination)
                    .await
                {
                    Ok(n) => archived_total += n,
                    Err(e) => {
                        warn!("retention archive for '{}' failed: {e}", table.name);
                        continue;
                    }
                }
            }

            let sql = format!("DELETE FROM \"{}\" WHERE {}", table.name, predicate);
            match Self::run_dml(wc, &sql).await {
                Ok(n) => {
                    deleted_total += n;
                    Self::record_job(wc, pol.table_id, 0, n, "ttl delete").await;
                }
                Err(e) => {
                    // Legal hold / WORM violations land here; skip, do not abort.
                    warn!("retention delete for '{}' skipped: {e}", table.name);
                }
            }
        }

        // Age tiering: cold_after and archive_after are declarations, so
        // they run every cycle rather than waiting for a manual
        // RUN RETENTION JOB. The relocation drives the same wire pass the
        // manual job does, and it needs the server state, which is
        // installed shortly after startup, so the first cycle or two may
        // skip it
        if let Some(server) = wc.server_state.get() {
            for t in wc.catalog.list_all_tables() {
                let lc = &t.lifecycle;
                if lc.cold_after_seconds <= 0 && lc.archive_after_seconds <= 0 {
                    continue;
                }
                match zyron_wire::lifecycle_dispatch::run_age_tiering(server, &t, now_us, false)
                    .await
                {
                    Ok((segments, rows)) if segments > 0 => {
                        info!(
                            table = %t.name,
                            segments,
                            rows,
                            "age tiering relocated segments"
                        );
                        Self::record_job(wc, t.id.0, 2, rows, "age tiering").await;
                    }
                    Ok(_) => {}
                    Err(e) => warn!("age tiering for '{}' failed: {e}", t.name),
                }
            }
        }

        // Soft-delete purge: physically remove tombstoned rows past the grace.
        let purged = Self::purge_soft_deleted(wc, now_us).await;

        stats.cycles_completed.fetch_add(1, Ordering::Relaxed);
        stats
            .rows_deleted
            .fetch_add(deleted_total, Ordering::Relaxed);
        stats
            .rows_archived
            .fetch_add(archived_total, Ordering::Relaxed);
        stats.rows_purged.fetch_add(purged, Ordering::Relaxed);
        if deleted_total + archived_total + purged > 0 {
            info!(
                "retention cycle: {} deleted, {} archived, {} purged",
                deleted_total, archived_total, purged
            );
        }
    }

    /// Purges soft-deleted rows whose deleted_at is older than the table's
    /// purge grace window, using HARD delete (bypasses the soft-delete
    /// rewrite). Recycle window and legal holds are still enforced.
    async fn purge_soft_deleted(wc: &WorkerCtx, now_us: i64) -> u64 {
        let tables = wc.catalog.list_all_tables();
        let mut purged = 0u64;
        for t in &tables {
            let lc = &t.lifecycle;
            if !lc.soft_delete_enabled {
                continue;
            }
            let grace = lc.purge_grace_seconds.max(lc.recycle_window_seconds);
            if grace <= 0 {
                continue;
            }
            let is_del = match t
                .columns
                .iter()
                .find(|c| c.id.0 as u32 == lc.soft_delete_is_deleted_col_id)
            {
                Some(c) => c.name.clone(),
                None => continue,
            };
            let del_at = match t
                .columns
                .iter()
                .find(|c| c.id.0 as u32 == lc.soft_delete_deleted_at_col_id)
            {
                Some(c) => c.name.clone(),
                None => continue,
            };
            let cutoff = now_us - grace.saturating_mul(1_000_000);
            let sql = format!(
                "DELETE FROM \"{}\" WHERE \"{}\" = true AND \"{}\" < {} HARD",
                t.name, is_del, del_at, cutoff
            );
            match Self::run_dml(wc, &sql).await {
                Ok(n) => {
                    if n > 0 {
                        purged += n;
                        Self::record_job(wc, t.id.0, 3, n, "soft-delete purge").await;
                    }
                }
                Err(e) => warn!("purge for '{}' skipped: {e}", t.name),
            }
        }
        purged
    }

    /// Selects rows matching `predicate`, serializes them to newline records,
    /// and writes them to the archive object store. Returns rows archived.
    async fn archive_matching(
        wc: &WorkerCtx,
        table: &str,
        predicate: &str,
        destination: &str,
    ) -> Result<u64, String> {
        let sql = format!(
            "SELECT * FROM \"{}\" WHERE {} INCLUDING DELETED",
            table, predicate
        );
        let batches = Self::run_query(wc, &sql).await?;
        let mut rows: Vec<Vec<u8>> = Vec::new();
        for b in &batches {
            for r in 0..b.num_rows {
                let mut fields: Vec<String> = Vec::with_capacity(b.columns.len());
                for c in 0..b.columns.len() {
                    fields.push(format!("{:?}", b.column(c).get_scalar(r)));
                }
                rows.push(fields.join("\u{1f}").into_bytes());
            }
        }
        if rows.is_empty() {
            return Ok(0);
        }
        let n = rows.len() as u64;
        zyron_lifecycle::archive::archive_rows(destination, &rows)
            .await
            .map_err(|e| format!("archive write: {e}"))?;
        Ok(n)
    }

    /// Plans and executes a DML statement in its own transaction with the
    /// legal-hold / WORM enforcement hook attached. Returns rows affected.
    async fn run_dml(wc: &WorkerCtx, sql: &str) -> Result<u64, String> {
        let stmts = zyron_parser::parse(sql).map_err(|e| format!("parse: {e}"))?;
        let stmt = stmts.into_iter().next().ok_or("empty statement")?;
        let plan = zyron_planner::plan(
            &wc.catalog,
            zyron_catalog::DatabaseId(1),
            vec!["public".to_string()],
            stmt,
            None,
        )
        .await
        .map_err(|e| format!("plan: {e}"))?;

        let mut txn = wc
            .txn_manager
            .begin(IsolationLevel::ReadCommitted)
            .map_err(|e| format!("begin: {e}"))?;
        let snapshot = txn.snapshot.clone();
        let txn_id = u32::try_from(txn.txn_id).map_err(|_| "txn id overflow".to_string())?;

        let mut ctx = zyron_executor::context::ExecutionContext::new(
            Arc::clone(&wc.catalog),
            Arc::clone(&wc.wal),
            Arc::clone(&wc.buffer_pool),
            Arc::clone(&wc.disk_manager),
            txn_id,
            snapshot,
        );
        ctx.dml_hook = Some(Arc::new(LegalHoldDmlHook::new(
            Arc::clone(&wc.legal_holds),
            Arc::clone(&wc.catalog),
        )) as Arc<dyn zyron_executor::context::DmlHook>);
        let ctx = Arc::new(ctx);

        let result = zyron_executor::execute(plan, &ctx).await;
        match result {
            Ok(batches) => {
                wc.txn_manager
                    .commit(&mut txn)
                    .await
                    .map_err(|e| format!("commit: {e}"))?;
                let affected: u64 = batches.iter().map(|b| b.num_rows as u64).sum();
                Ok(affected)
            }
            Err(e) => {
                let _ = wc.txn_manager.abort(&mut txn);
                Err(format!("execute: {e}"))
            }
        }
    }

    /// Plans and executes a read-only query in an aborted transaction.
    async fn run_query(
        wc: &WorkerCtx,
        sql: &str,
    ) -> Result<Vec<zyron_executor::batch::DataBatch>, String> {
        let stmts = zyron_parser::parse(sql).map_err(|e| format!("parse: {e}"))?;
        let stmt = stmts.into_iter().next().ok_or("empty statement")?;
        let plan = zyron_planner::plan(
            &wc.catalog,
            zyron_catalog::DatabaseId(1),
            vec!["public".to_string()],
            stmt,
            None,
        )
        .await
        .map_err(|e| format!("plan: {e}"))?;
        let mut txn = wc
            .txn_manager
            .begin(IsolationLevel::ReadCommitted)
            .map_err(|e| format!("begin: {e}"))?;
        let snapshot = txn.snapshot.clone();
        let txn_id = u32::try_from(txn.txn_id).map_err(|_| "txn id overflow".to_string())?;
        let ctx = Arc::new(zyron_executor::context::ExecutionContext::new(
            Arc::clone(&wc.catalog),
            Arc::clone(&wc.wal),
            Arc::clone(&wc.buffer_pool),
            Arc::clone(&wc.disk_manager),
            txn_id,
            snapshot,
        ));
        let result = zyron_executor::execute(plan, &ctx).await;
        let _ = wc.txn_manager.abort(&mut txn);
        result.map_err(|e| format!("execute: {e}"))
    }

    async fn record_job(wc: &WorkerCtx, table_id: u32, kind: u8, rows: u64, detail: &str) {
        let now = now_micros();
        let _ = wc
            .catalog
            .store_retention_job(&zyron_catalog::schema::RetentionJobEntry {
                job_id: now as u64,
                table_id,
                kind,
                scheduled_at: now,
                started_at: now,
                finished_at: now,
                rows_affected: rows,
                status: 2,
                detail: detail.to_string(),
            })
            .await;
        let _ = wc
            .catalog
            .append_compliance_log(zyron_catalog::schema::ComplianceLogEntry {
                event_id: 0,
                event_type: if kind == 3 { 10 } else { 0 },
                subject: format!("table:{table_id}"),
                table_id,
                ts: now,
                detail: format!("{detail}: {rows} rows"),
                prev_hash: 0,
                entry_hash: 0,
            })
            .await;
    }
}

fn now_micros() -> i64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_micros() as i64)
        .unwrap_or(0)
}
