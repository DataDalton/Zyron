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
//! On a group the leader runs the expiring steps and commits each delete
//! through the group, so every member expires the same rows in the same
//! entry. Age tiering moves a node's own columnar segments between its own
//! tiers and changes no visible data, so every member runs it for itself.
//!
//! Legal-hold / WORM violations surface as errors from the DML hook; the
//! worker logs and skips that table without aborting the whole cycle.

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, OnceLock};
use std::thread::{self, JoinHandle};
use std::time::Duration;

use tracing::{debug, info, warn};

use zyron_lifecycle::legal_hold::LegalHoldRegistry;
use zyron_storage::txn::IsolationLevel;
use zyron_wire::connection::ServerState;

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
    pub fn new() -> Self {
        Self {
            cycles_completed: AtomicU64::new(0),
            rows_deleted: AtomicU64::new(0),
            rows_archived: AtomicU64::new(0),
            rows_purged: AtomicU64::new(0),
        }
    }
}

impl Default for RetentionStats {
    fn default() -> Self {
        Self::new()
    }
}

pub struct RetentionWorker {
    shutdown: Arc<AtomicBool>,
    waker: Arc<OnceLock<thread::Thread>>,
    thread: Option<JoinHandle<()>>,
    stats: Arc<RetentionStats>,
    /// Installed once the server state exists, which is after the workers
    /// start. Every pass writes and reads through it, cycles before
    /// installation run nothing
    server_state: Arc<OnceLock<Arc<ServerState>>>,
}

impl RetentionWorker {
    pub fn start(
        config: RetentionWorkerConfig,
        authority: crate::background::authority::WriteAuthority,
    ) -> Self {
        let shutdown = Arc::new(AtomicBool::new(false));
        let waker = Arc::new(OnceLock::new());
        let stats = Arc::new(RetentionStats::new());
        let server_state: Arc<OnceLock<Arc<ServerState>>> = Arc::new(OnceLock::new());

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
                let legal_holds = Arc::new(LegalHoldRegistry::new());
                let interval = Duration::from_secs(config.interval_secs.max(1));
                loop {
                    thread::park_timeout(interval);
                    if t_shutdown.load(Ordering::Acquire) {
                        return;
                    }
                    // Every pass writes through the server's registries and
                    // records the rows it expires for the tables' feeds, so
                    // a cycle that runs before the server state is installed
                    // waits for the next one
                    let Some(server) = t_server_state.get() else {
                        debug!("retention: the server state is not installed yet, cycle skipped");
                        continue;
                    };
                    // Expiring rows is a write the leader of a group commits
                    // through the group, so a follower expires nothing and
                    // applies what the leader expired. Tiering is every
                    // member's own
                    let expires = authority.may_write();
                    runtime.block_on(run_retention_cycle(server, &legal_holds, &t_stats, expires));
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

    /// Installs the server state once it exists, enabling every pass from
    /// the next cycle on
    pub fn install_server_state(&self, state: Arc<ServerState>) {
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
}

/// What one cycle runs against, the server every pass writes and reads
/// through, and the holds the DML hook enforces
struct CycleCtx<'a> {
    server: &'a Arc<ServerState>,
    legal_holds: &'a Arc<LegalHoldRegistry>,
}

/// One retention cycle against `server`. `expires` says whether this node
/// may expire rows this cycle, which the leader of a group and a node in no
/// group may. The worker drives this on its interval, and a test drives it
/// directly to prove what a cycle leaves on every member of a group
pub async fn run_retention_cycle(
    server: &Arc<ServerState>,
    legal_holds: &Arc<LegalHoldRegistry>,
    stats: &RetentionStats,
    expires: bool,
) {
    let cx = CycleCtx {
        server,
        legal_holds,
    };
    let catalog = &server.catalog;
    let now_us = now_micros();
    let mut deleted_total = 0u64;
    let mut archived_total = 0u64;

    if expires {
        // Reload holds from the catalog (source of truth) so the DML hook
        // enforces the current state this cycle.
        if let Ok(holds) = catalog.load_legal_holds().await {
            legal_holds.reload(&holds);
        }

        let policies = match catalog.load_retention_policies().await {
            Ok(p) => p,
            Err(e) => {
                debug!("retention: load policies failed: {e}");
                Vec::new()
            }
        };

        for pol in policies.iter().filter(|p| p.kind == 0) {
            let table = match catalog.get_table_by_id(zyron_catalog::TableId(pol.table_id)) {
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
                match archive_matching(
                    &cx,
                    table_ns(&cx, table.schema_id),
                    &table.name,
                    &predicate,
                    &lc.archive_destination,
                )
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
            match run_dml(&cx, table_ns(&cx, table.schema_id), &sql).await {
                Ok(n) => {
                    if n > 0 {
                        deleted_total += n;
                        record_job(&cx, pol.table_id, 0, n, "ttl delete").await;
                    }
                }
                Err(e) => {
                    // Legal hold / WORM violations land here; skip, do not abort.
                    warn!("retention delete for '{}' skipped: {e}", table.name);
                }
            }
        }
    }

    // Age tiering runs every cycle rather than waiting for a manual RUN
    // RETENTION JOB, because cold_after and archive_after are declarations.
    // The relocation moves this node's own segments between its own tiers
    // and changes no visible data, so every member of a group runs it for
    // itself
    for t in catalog.list_all_tables() {
        let lc = &t.lifecycle;
        if lc.cold_after_seconds <= 0 && lc.archive_after_seconds <= 0 {
            continue;
        }
        match zyron_wire::lifecycle_dispatch::run_age_tiering(server, &t, now_us, false).await {
            Ok((segments, rows)) if segments > 0 => {
                info!(
                    table = %t.name,
                    segments,
                    rows,
                    "age tiering relocated segments"
                );
                record_job(&cx, t.id.0, 2, rows, "age tiering").await;
            }
            Ok(_) => {}
            Err(e) => warn!("age tiering for '{}' failed: {e}", t.name),
        }
    }

    // Soft-delete purge, removing tombstoned rows past the grace
    let purged = if expires {
        purge_soft_deleted(&cx, now_us).await
    } else {
        0
    };

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
/// rewrite). Recycle window and legal holds are still enforced
async fn purge_soft_deleted(cx: &CycleCtx<'_>, now_us: i64) -> u64 {
    let tables = cx.server.catalog.list_all_tables();
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
        match run_dml(cx, table_ns(cx, t.schema_id), &sql).await {
            Ok(n) => {
                if n > 0 {
                    purged += n;
                    record_job(cx, t.id.0, 3, n, "soft-delete purge").await;
                }
            }
            Err(e) => warn!("purge for '{}' skipped: {e}", t.name),
        }
    }
    purged
}

/// Selects rows matching `predicate`, serializes them to newline records,
/// and writes them to the archive object store. Returns rows archived
async fn archive_matching(
    cx: &CycleCtx<'_>,
    ns: (zyron_catalog::DatabaseId, Vec<String>),
    table: &str,
    predicate: &str,
    destination: &str,
) -> Result<u64, String> {
    let sql = format!(
        "SELECT * FROM \"{}\" WHERE {} INCLUDING DELETED",
        table, predicate
    );
    let batches = run_query(cx, ns, &sql).await?;
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

/// The namespace that resolves a retention target, the table's own
/// schema, never an implicit default
fn table_ns(
    cx: &CycleCtx<'_>,
    schema_id: zyron_catalog::SchemaId,
) -> (zyron_catalog::DatabaseId, Vec<String>) {
    match cx.server.catalog.get_schema_by_id(schema_id) {
        Ok(s) => (s.database_id, vec![s.name.clone()]),
        Err(_) => (
            zyron_catalog::DatabaseId(1),
            zyron_catalog::default_search_path(),
        ),
    }
}

/// Plans and executes a DML statement in its own transaction with the
/// legal-hold / WORM enforcement hook attached, committing through the
/// group when this node leads one. Returns rows affected
async fn run_dml(
    cx: &CycleCtx<'_>,
    ns: (zyron_catalog::DatabaseId, Vec<String>),
    sql: &str,
) -> Result<u64, String> {
    let server = cx.server;
    let stmts = zyron_parser::parse(sql).map_err(|e| format!("parse: {e}"))?;
    let stmt = stmts.into_iter().next().ok_or("empty statement")?;
    let plan = zyron_planner::plan(&server.catalog, ns.0, ns.1, stmt, None)
        .await
        .map_err(|e| format!("plan: {e}"))?;

    let mut txn = server
        .txn_manager
        .begin(IsolationLevel::ReadCommitted)
        .map_err(|e| format!("begin: {e}"))?;
    let snapshot = txn.snapshot.clone();
    let txn_id = txn.txn_id;

    // The rows retention expires are the table's changes, recorded for its
    // feed, landed through the same registries a client's own DELETE runs
    // against, and captured for the group the same way
    let mut ctx = server.statement_context(txn_id, snapshot);
    ctx.dml_hook = Some(Arc::new(LegalHoldDmlHook::new(
        Arc::clone(cx.legal_holds),
        Arc::clone(&server.catalog),
    )) as Arc<dyn zyron_executor::context::DmlHook>);
    let changeset = server.replication.as_ref().map(|r| r.changeset(txn_id));
    ctx.replication = changeset.clone();
    let ctx = Arc::new(ctx);

    let result = zyron_executor::execute(plan, &ctx).await;
    if ctx.wrote_wal() {
        txn.mark_wrote_data();
    }
    match result {
        Ok(batches) => {
            zyron_wire::ddl_dispatch::commit_generated(server, txn, changeset, &[])
                .await
                .map_err(|e| format!("commit: {e}"))?;
            Ok(affected_rows(&batches))
        }
        Err(e) => {
            zyron_wire::ddl_dispatch::abandon_generated(server, &mut txn, changeset.as_deref());
            Err(format!("execute: {e}"))
        }
    }
}

/// The rows a write statement affected, read off the count it answers with.
/// A write answers one row holding the count, so the row count of the
/// answer is not the number of rows written
fn affected_rows(batches: &[zyron_executor::batch::DataBatch]) -> u64 {
    batches
        .iter()
        .filter(|b| b.num_rows > 0)
        .filter_map(|b| b.columns.first())
        .map(|column| match column.data.get_scalar(0) {
            zyron_executor::column::ScalarValue::Int64(count) => count.max(0) as u64,
            zyron_executor::column::ScalarValue::Int32(count) => count.max(0) as u64,
            _ => 0,
        })
        .sum()
}

/// Plans and executes a read-only query in an aborted transaction
async fn run_query(
    cx: &CycleCtx<'_>,
    ns: (zyron_catalog::DatabaseId, Vec<String>),
    sql: &str,
) -> Result<Vec<zyron_executor::batch::DataBatch>, String> {
    let server = cx.server;
    let stmts = zyron_parser::parse(sql).map_err(|e| format!("parse: {e}"))?;
    let stmt = stmts.into_iter().next().ok_or("empty statement")?;
    let plan = zyron_planner::plan(&server.catalog, ns.0, ns.1, stmt, None)
        .await
        .map_err(|e| format!("plan: {e}"))?;
    let mut txn = server
        .txn_manager
        .begin(IsolationLevel::ReadCommitted)
        .map_err(|e| format!("begin: {e}"))?;
    let snapshot = txn.snapshot.clone();
    let txn_id = txn.txn_id;
    let ctx = Arc::new(server.statement_context(txn_id, snapshot));
    let result = zyron_executor::execute(plan, &ctx).await;
    let _ = server.txn_manager.abort(&mut txn);
    result.map_err(|e| format!("execute: {e}"))
}

/// Records what a pass did in this node's own job history and its
/// compliance log. The history is the record of the node that ran the
/// pass, so it is written where the pass ran
async fn record_job(cx: &CycleCtx<'_>, table_id: u32, kind: u8, rows: u64, detail: &str) {
    let now = now_micros();
    let catalog = &cx.server.catalog;
    let _ = catalog
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
    let _ = catalog
        .append_compliance_log(zyron_catalog::schema::ComplianceLogEntry {
            event_id: 0,
            event_type: if kind == 3 { 10 } else { 0 },
            subject: format!("table:{table_id}"),
            table_id,
            ts: now,
            detail: format!("{detail}: {rows} rows"),
            prev_hash: 0,
            entry_hash: 0,
            record_version: zyron_lifecycle::format::AUDIT_RECORD_VERSION_BYTE,
        })
        .await;
}

fn now_micros() -> i64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_micros() as i64)
        .unwrap_or(0)
}
