//! ZyronDB server integration.
//!
//! Orchestrates all subsystems into a complete database server: configuration,
//! session management, background workers, metrics, health endpoints, and
//! graceful shutdown.

pub mod background;
pub mod backup;
pub mod config;
pub mod gateway;
pub mod health;
pub mod hooks;
pub mod io_stats;
pub mod metrics;
pub mod session;
pub mod signal;

use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Instant;

use tracing::{error, info, warn};

use zyron_buffer::{
    BackgroundWriter, BackgroundWriterConfig, BufferPool, BufferPoolConfig, WriteFn,
};
use zyron_catalog::{Catalog, CatalogCache, HeapCatalogStorage};
use zyron_storage::checkpoint::CheckpointTracker;
use zyron_storage::txn::TransactionManager;
use zyron_storage::{DiskManager, DiskManagerConfig};
use zyron_wal::{RecoveryManager, WalWriter, WalWriterConfig};
use zyron_wire::connection::ServerState;

use crate::background::BackgroundWorkers;
use crate::background::checkpoint::CheckpointWorkerConfig;
use crate::background::stats::StatsCollectorConfig;
use crate::background::vacuum::VacuumWorkerConfig;
use crate::config::ZyronConfig;
use crate::health::HealthState;
use crate::metrics::MetricsRegistry;
use crate::session::SessionManager;

/// CLI options parsed from command-line arguments.
pub struct CliOptions {
    pub config_path: Option<PathBuf>,
    pub data_dir: Option<PathBuf>,
    pub port: Option<u16>,
    pub host: Option<String>,
    pub log_level: Option<String>,
    pub foreground: bool,
    pub single_user: bool,
    pub skip_recovery: bool,
}

impl Default for CliOptions {
    fn default() -> Self {
        Self {
            config_path: None,
            data_dir: None,
            port: None,
            host: None,
            log_level: None,
            foreground: false,
            single_user: false,
            skip_recovery: false,
        }
    }
}

/// Parses command-line arguments into CliOptions.
/// Returns None if --help or --version was requested (already printed).
pub fn parse_cli_args() -> Option<CliOptions> {
    let args: Vec<String> = std::env::args().collect();
    let mut opts = CliOptions::default();
    let mut i = 1;

    while i < args.len() {
        match args[i].as_str() {
            "--help" | "-h" => {
                print_usage();
                return None;
            }
            "--version" | "-v" => {
                println!("zyrondb-server {}", env!("CARGO_PKG_VERSION"));
                return None;
            }
            "--config" => {
                i += 1;
                if i >= args.len() {
                    eprintln!("--config requires a value");
                    std::process::exit(1);
                }
                opts.config_path = Some(PathBuf::from(&args[i]));
            }
            "--data-dir" => {
                i += 1;
                if i >= args.len() {
                    eprintln!("--data-dir requires a value");
                    std::process::exit(1);
                }
                opts.data_dir = Some(PathBuf::from(&args[i]));
            }
            "--port" => {
                i += 1;
                if i >= args.len() {
                    eprintln!("--port requires a value");
                    std::process::exit(1);
                }
                opts.port = Some(args[i].parse().expect("invalid port number"));
            }
            "--host" => {
                i += 1;
                if i >= args.len() {
                    eprintln!("--host requires a value");
                    std::process::exit(1);
                }
                opts.host = Some(args[i].clone());
            }
            "--log-level" => {
                i += 1;
                if i >= args.len() {
                    eprintln!("--log-level requires a value");
                    std::process::exit(1);
                }
                opts.log_level = Some(args[i].clone());
            }
            "--foreground" | "-f" => {
                opts.foreground = true;
            }
            "--single-user" => {
                opts.single_user = true;
            }
            "--skip-recovery" => {
                opts.skip_recovery = true;
            }
            other => {
                eprintln!("Unknown argument: {}", other);
                print_usage();
                std::process::exit(1);
            }
        }
        i += 1;
    }

    Some(opts)
}

fn print_usage() {
    println!(
        "Usage: zyrondb-server [OPTIONS]

Options:
  --config <path>       Path to zyrondb.toml config file
  --data-dir <path>     Data directory override
  --port <number>       Listen port override
  --host <addr>         Bind address override
  --log-level <level>   Log level: debug, info, warn, error
  --foreground, -f      Run in foreground
  --single-user         Single-connection mode for maintenance
  --skip-recovery       Skip WAL replay (emergency use only)
  --version, -v         Print version and exit
  --help, -h            Print this help and exit"
    );
}

/// The ZyronDB server. Owns all subsystems and coordinates lifecycle.
pub struct Server {
    config: ZyronConfig,
    session_mgr: Arc<SessionManager>,
    health_state: Arc<HealthState>,
    shutdown: Arc<AtomicBool>,
}

impl Server {
    /// Initializes the server with the given configuration and CLI options.
    /// Performs WAL recovery if needed.
    pub async fn init(mut config: ZyronConfig, opts: &CliOptions) -> zyron_common::Result<Self> {
        // Apply CLI overrides
        if let Some(ref dir) = opts.data_dir {
            config.storage.data_dir = dir.clone();
        }
        if let Some(port) = opts.port {
            config.server.port = port;
        }
        if let Some(ref host) = opts.host {
            config.server.host = host.clone();
        }
        if let Some(ref level) = opts.log_level {
            config.logging.level = level.clone();
        }
        if opts.single_user {
            config.server.max_connections = 1;
        }

        // Initialize tracing
        let log_level = match config.logging.level.as_str() {
            "debug" => tracing::Level::DEBUG,
            "warn" => tracing::Level::WARN,
            "error" => tracing::Level::ERROR,
            _ => tracing::Level::INFO,
        };
        tracing_subscriber::fmt()
            .with_max_level(log_level)
            .with_target(false)
            .init();

        let session_mgr = Arc::new(SessionManager::new(
            config.server.max_connections,
            config.server.connection_timeout_secs,
        ));
        let metrics = Arc::new(MetricsRegistry::new(session_mgr.clone()));
        let health_state = Arc::new(HealthState::new(metrics));
        let shutdown = Arc::new(AtomicBool::new(false));

        Ok(Self {
            config,
            session_mgr,
            health_state,
            shutdown,
        })
    }

    /// Runs the server. This is the main entry point that blocks until shutdown.
    pub async fn run(self) -> zyron_common::Result<()> {
        let start_time = Instant::now();
        info!("ZyronDB starting");

        // 1. Create data and WAL directories
        let data_dir = &self.config.storage.data_dir;
        let wal_dir = self.config.wal_dir();
        tokio::fs::create_dir_all(data_dir).await.map_err(|e| {
            zyron_common::ZyronError::Internal(format!(
                "Failed to create data dir {}: {}",
                data_dir.display(),
                e
            ))
        })?;
        tokio::fs::create_dir_all(&wal_dir).await.map_err(|e| {
            zyron_common::ZyronError::Internal(format!(
                "Failed to create WAL dir {}: {}",
                wal_dir.display(),
                e
            ))
        })?;

        // 2. Create DiskManager
        let disk_manager = Arc::new(
            DiskManager::new(DiskManagerConfig {
                data_dir: data_dir.clone(),
                fsync_enabled: self.config.wal.sync_mode == "fsync",
            })
            .await?,
        );

        // 3. Create BufferPool + BackgroundWriter
        let buffer_pool = Arc::new(BufferPool::new(BufferPoolConfig {
            num_frames: self.config.storage.buffer_pool_size / self.config.storage.page_size,
        }));

        let dm_for_bg = Arc::clone(&disk_manager);
        let write_fn: WriteFn =
            Arc::new(move |page_id, data| dm_for_bg.write_page_sync(page_id, data));
        let background_writer = Arc::new(BackgroundWriter::new(
            Arc::clone(&buffer_pool),
            write_fn,
            BackgroundWriterConfig::default(),
        ));

        // 4. Create WalWriter
        let wal = Arc::new(WalWriter::new(WalWriterConfig {
            wal_dir: wal_dir.clone(),
            segment_size: self.config.wal.segment_size as u32,
            fsync_enabled: self.config.wal.sync_mode == "fsync",
            ring_buffer_capacity: 256 * 1024, // 256 KB ring buffer
        })?);

        // 5. WAL recovery
        let recovery_result = if self.config.storage.data_dir.exists() {
            info!("Running WAL recovery");
            let recovery_mgr = RecoveryManager::new(&wal_dir)?;
            let result = recovery_mgr.recover()?;
            info!(
                "WAL recovery complete: {} redo records, {} undo transactions",
                result.redo_records.len(),
                result.undo_txns.len()
            );
            Some(result)
        } else {
            None
        };

        // Determine starting txn_id from recovery
        let start_txn_id = recovery_result
            .as_ref()
            .and_then(|r| r.last_lsn.map(|_| 1u64))
            .unwrap_or(1);

        // 6. Create Catalog
        let catalog_storage = Arc::new(HeapCatalogStorage::new(
            Arc::clone(&disk_manager),
            Arc::clone(&buffer_pool),
        )?);
        let catalog_cache = Arc::new(CatalogCache::new(1024, 256));
        let catalog =
            Arc::new(Catalog::new(catalog_storage, catalog_cache, Arc::clone(&wal)).await?);

        // Load catalog entries from disk
        catalog.load().await?;

        // 7. Create TransactionManager
        let txn_manager = Arc::new(TransactionManager::with_start_txn_id(
            Arc::clone(&wal),
            start_txn_id,
        ));

        // 8. Create SecurityManager with heap-backed auth storage
        let auth_storage: Arc<dyn zyron_auth::storage::AuthStorage> = Arc::new(
            zyron_auth::HeapAuthStorage::new(Arc::clone(&disk_manager), Arc::clone(&buffer_pool))?,
        );
        let mut security_manager = zyron_auth::SecurityManager::new(auth_storage).await?;

        // Apply auth config from the config file
        {
            let auth_cfg = &self.config.auth;

            // WebAuthn relying party config
            if let Some(ref rp_id) = auth_cfg.webauthn_rp_id {
                security_manager.webauthn_rp_config.rp_id = rp_id.clone();
            }
            if let Some(ref rp_name) = auth_cfg.webauthn_rp_name {
                security_manager.webauthn_rp_config.rp_name = rp_name.clone();
            }
            if let Some(ref origin) = auth_cfg.webauthn_origin {
                security_manager.webauthn_rp_config.origin = origin.clone();
            }
            if let Some(timeout) = auth_cfg.webauthn_challenge_timeout {
                security_manager.webauthn_rp_config.challenge_timeout_secs = timeout;
            }
            // JWT authentication config
            if let Some(ref secret) = auth_cfg.jwt_secret {
                security_manager.jwt_secret = Some(secret.as_bytes().to_vec());
            }
            if let Some(ref alg) = auth_cfg.jwt_algorithm {
                security_manager.jwt_algorithm = match alg.as_str() {
                    "HS384" => zyron_auth::JwtAlgorithm::Hs384,
                    "HS512" => zyron_auth::JwtAlgorithm::Hs512,
                    _ => zyron_auth::JwtAlgorithm::Hs256,
                };
            }
            if let Some(ref issuer) = auth_cfg.jwt_issuer {
                security_manager.jwt_issuer = Some(issuer.clone());
            }

            let policy = zyron_auth::BruteForcePolicy {
                lockout_threshold: auth_cfg.lockout_threshold.unwrap_or(5),
                lockout_duration_secs: auth_cfg.lockout_duration_secs.unwrap_or(900),
                ip_block_threshold: auth_cfg.ip_block_threshold.unwrap_or(50),
                failure_window_secs: auth_cfg.failure_window_secs.unwrap_or(600),
                ip_block_duration_secs: auth_cfg.ip_block_duration_secs.unwrap_or(3600),
                min_attempt_interval_ms: auth_cfg.min_attempt_interval_ms.unwrap_or(100),
                lockout_enabled: auth_cfg.brute_force_enabled.unwrap_or(true),
            };
            security_manager.brute_force.set_global_policy(policy);
        }

        // 9. Create CheckpointTracker
        let tracker = Arc::new(CheckpointTracker::new());

        // 10. Initialize subsystem managers (before background workers so workers can use them)
        let cdc_registry_arc = Arc::new(zyron_cdc::CdfRegistry::new(data_dir.clone()));
        let slot_mgr_arc =
            zyron_cdc::SlotManager::open(&data_dir, zyron_cdc::SlotLagConfig::default())
                .ok()
                .map(Arc::new);
        let pub_mgr_arc = zyron_cdc::PublicationManager::open(&data_dir)
            .ok()
            .map(Arc::new);
        let cdc_stream_mgr_arc = zyron_cdc::CdcStreamManager::new(&data_dir)
            .ok()
            .map(Arc::new);
        let cdc_ingest_mgr_arc = zyron_cdc::CdcIngestManager::new(&data_dir)
            .ok()
            .map(Arc::new);

        let trigger_mgr_arc = Arc::new(zyron_pipeline::trigger::TriggerManager::new());
        let udf_reg_arc = Arc::new(zyron_pipeline::udf::UdfRegistry::new());
        let uda_reg_arc = Arc::new(zyron_pipeline::aggregate::UdaRegistry::new());
        let proc_reg_arc = Arc::new(zyron_pipeline::stored_procedure::ProcedureRegistry::new());
        let pipeline_mgr_arc = Arc::new(zyron_pipeline::pipeline::PipelineManager::new());
        let sched_mgr_arc = Arc::new(zyron_pipeline::schedule::ScheduleManager::new());
        let event_disp_arc = Arc::new(zyron_pipeline::event_handler::EventDispatcher::new());
        let mv_mgr_arc =
            Arc::new(zyron_pipeline::materialized_view::MaterializedViewManager::new());

        let stream_mgr_arc = Arc::new(parking_lot::Mutex::new(
            zyron_streaming::job::StreamJobManager::new(),
        ));
        let branch_mgr_arc = Arc::new(zyron_versioning::BranchManager::new(data_dir.clone()));
        let notif_arc = Arc::new(zyron_wire::notifications::NotificationChannels::new());

        // Full-text search manager: load persisted FTS indexes from catalog.
        let fts_mgr_arc = {
            let fts_dir = data_dir.join("fts");
            let _ = std::fs::create_dir_all(&fts_dir);
            let mgr = zyron_search::FtsManager::with_data_dir(fts_dir.clone());
            let mut fts_entries: Vec<(u32, u32, Vec<u16>)> = Vec::new();
            for table in catalog.list_all_tables() {
                for idx in catalog.get_indexes_for_table(table.id) {
                    if idx.index_type == zyron_catalog::IndexType::Fulltext {
                        let col_ids: Vec<u16> = idx.columns.iter().map(|c| c.column_id.0).collect();
                        fts_entries.push((idx.id.0, table.id.0, col_ids));
                    }
                }
            }
            if !fts_entries.is_empty() {
                if let Err(e) = mgr.load_indexes(&fts_dir, &fts_entries) {
                    error!("FTS index loading failed: {e}");
                }
            }
            Arc::new(mgr)
        };

        // Vector index manager: load persisted vector indexes from catalog.
        let vec_mgr_arc = {
            let vec_dir = data_dir.join("vector");
            let _ = std::fs::create_dir_all(&vec_dir);
            let mgr = zyron_search::vector::VectorIndexManager::with_data_dir(vec_dir.clone());
            let mut vec_entries: Vec<(u32, u32, u16, u16, zyron_search::vector::HnswConfig)> =
                Vec::new();
            for table in catalog.list_all_tables() {
                for idx in catalog.get_indexes_for_table(table.id) {
                    if idx.index_type == zyron_catalog::IndexType::Vector {
                        let col_id = idx.columns.first().map(|c| c.column_id.0).unwrap_or(0);
                        let (dims, config) = if let Some(ref param_bytes) = idx.parameters {
                            match zyron_search::vector::VectorIndexParams::fromBytes(param_bytes) {
                                Ok(zyron_search::vector::VectorIndexParams::Hnsw {
                                    m,
                                    efConstruction,
                                    efSearch,
                                    metric,
                                    dimensions,
                                }) => {
                                    let dist = match metric {
                                        1 => zyron_search::vector::DistanceMetric::Euclidean,
                                        2 => zyron_search::vector::DistanceMetric::DotProduct,
                                        3 => zyron_search::vector::DistanceMetric::Manhattan,
                                        _ => zyron_search::vector::DistanceMetric::Cosine,
                                    };
                                    (
                                        dimensions,
                                        zyron_search::vector::HnswConfig {
                                            m,
                                            efConstruction,
                                            efSearch,
                                            metric: dist,
                                        },
                                    )
                                }
                                _ => (0, zyron_search::vector::HnswConfig::default()),
                            }
                        } else {
                            (0, zyron_search::vector::HnswConfig::default())
                        };
                        vec_entries.push((idx.id.0, table.id.0, col_id, dims, config));
                    }
                }
            }
            if !vec_entries.is_empty() {
                if let Err(e) = mgr.load_indexes(&vec_dir, &vec_entries) {
                    error!("Vector index loading failed: {e}");
                }
            }
            Arc::new(mgr)
        };

        // Graph schema manager. Load persisted schemas from disk.
        let graph_mgr_arc = {
            let graph_dir = data_dir.join("graph");
            let _ = std::fs::create_dir_all(&graph_dir);
            let mgr = zyron_search::graph::GraphManager::new();
            if let Err(e) = mgr.load_all(&graph_dir) {
                error!("Graph schema loading failed: {e}");
            }
            Arc::new(mgr)
        };

        // -------------------------------------------------------------------
        // Spatial index manager: load persisted R-trees or rebuild from the
        // base table for every Spatial catalog entry. Snapshot loads are a
        // fast-path optimization. When a snapshot is missing, stale, or fails
        // the checksum, recovery falls back to a full heap scan and re-inserts
        // every geometry so queries and DML maintenance stay correct.
        // -------------------------------------------------------------------
        let spatial_mgr_arc = {
            let spatial_dir = data_dir.join("spatial");
            let _ = std::fs::create_dir_all(&spatial_dir);
            let mgr = zyron_types::spatial_index::SpatialIndexManager::new();

            for table in catalog.list_all_tables() {
                for idx in catalog.get_indexes_for_table(table.id) {
                    if idx.index_type != zyron_catalog::IndexType::Spatial {
                        continue;
                    }
                    let (dims, srid) = parse_spatial_params(idx.parameters.as_deref());
                    mgr.create_index(idx.id.0, dims, srid);

                    let saved = spatial_dir.join(format!("{}.rtree", idx.id.0));
                    if saved.exists() {
                        match mgr.restore_from_file(idx.id.0, &saved) {
                            Ok(()) => {
                                info!(
                                    target: "zyron::recovery",
                                    "spatial index {} restored from disk",
                                    idx.id.0
                                );
                                continue;
                            }
                            Err(e) => {
                                warn!(
                                    target: "zyron::recovery",
                                    "spatial index {} disk load failed ({}), rebuilding from table",
                                    idx.id.0,
                                    e
                                );
                            }
                        }
                    }

                    if let Err(e) = rebuild_spatial_index_from_table(
                        &mgr,
                        idx.id.0,
                        dims,
                        &buffer_pool,
                        &disk_manager,
                        idx.as_ref(),
                        table.as_ref(),
                    )
                    .await
                    {
                        error!(
                            target: "zyron::recovery",
                            "spatial index {} rebuild failed: {}",
                            idx.id.0,
                            e
                        );
                    }
                }
            }

            Arc::new(mgr)
        };

        // 11. Start background workers
        let ckpt_config = CheckpointWorkerConfig {
            wal_bytes_threshold: self.config.checkpoint.wal_bytes_threshold,
            max_interval_secs: self.config.checkpoint.max_interval_secs as u64,
            min_interval_secs: self.config.checkpoint.min_interval_secs as u64,
        };
        let vacuum_config = VacuumWorkerConfig {
            interval_secs: self.config.vacuum.interval_secs,
            ..VacuumWorkerConfig::default()
        };
        let mut background = BackgroundWorkers::start(
            Arc::clone(&catalog),
            Arc::clone(&wal),
            Arc::clone(&buffer_pool),
            background_writer,
            Arc::clone(&disk_manager),
            Arc::clone(&txn_manager),
            tracker,
            ckpt_config,
            StatsCollectorConfig::default(),
            vacuum_config,
            wal_dir,
            None, // WAL archiving disabled unless configured
            Some(Arc::clone(&cdc_registry_arc)),
            Some(Arc::clone(&stream_mgr_arc)),
        );

        // Extract background worker stats for ServerState
        let ckpt_stats = background.checkpoint_stats();
        let vac_stats = background.vacuum_stats();

        // Build config callbacks for ServerState
        let config_for_lookup = self.config.clone();
        let config_for_all = self.config.clone();
        let data_dir_for_alter = data_dir.clone();

        let session_mgr_for_view = Arc::clone(&self.session_mgr);
        let ckpt_wake = {
            let _ckpt_ref = background.checkpoint();
            let waker: Option<Arc<dyn Fn() + Send + Sync>> = None;
            // CheckpointWorker does not expose a wake method via Arc. Skip for now.
            waker
        };

        // -------------------------------------------------------------------
        // Gateway router + registrar.
        // The router holds compiled routes. The registrar wraps the router in
        // a trait object so ddl_dispatch can mutate it without depending on
        // zyron-server. Startup recovery below populates it from the catalog.
        // -------------------------------------------------------------------
        let gateway_router = Arc::new(crate::gateway::router::Router::new());
        let endpoint_registrar: Arc<
            dyn zyron_wire::EndpointRegistrar,
        > = Arc::new(crate::gateway::router::CatalogEndpointRegistrar::new(
            Arc::clone(&gateway_router),
        ));

        // Wrap the SecurityManager in an Arc once so both ServerState and the
        // admin executor share the same instance.
        let security_manager_arc = Arc::new(security_manager);

        // Build ServerState for zyron-wire
        let server_state = Arc::new(ServerState {
            catalog: Arc::clone(&catalog),
            wal: Arc::clone(&wal),
            buffer_pool: Arc::clone(&buffer_pool),
            disk_manager: Arc::clone(&disk_manager),
            txn_manager: Arc::clone(&txn_manager),
            security_manager: Some(Arc::clone(&security_manager_arc)),
            key_store: {
                // Derive a stable master key from the data-dir path. Survives
                // restarts for the same data directory. An ops deployment
                // replaces this with a KMS-backed KeyStore via the trait.
                use sha2::{Digest, Sha256};
                let mut hasher = Sha256::new();
                hasher.update(data_dir.to_string_lossy().as_bytes());
                hasher.update(b"external-credentials-kek-v1");
                let digest = hasher.finalize();
                let mut key = [0u8; 32];
                key.copy_from_slice(&digest);
                Arc::new(zyron_auth::LocalKeyStore::new(key))
            },
            config_lookup: Some(Arc::new(move |key: &str| -> Option<String> {
                config_for_lookup.get_config_value(key)
            })),
            config_all: Some(Arc::new(move || -> Vec<(String, String, String)> {
                config_for_all.all_config_entries()
            })),
            data_dir: data_dir.clone(),
            session_info_collector: Some(Arc::new(
                move || -> Vec<zyron_wire::stat_views::SessionRow> {
                    let mut rows = Vec::new();
                    session_mgr_for_view.for_each(|info| {
                        rows.push(zyron_wire::stat_views::SessionRow {
                            pid: info.process_id,
                            user_name: info.user.clone(),
                            database: info.database.clone(),
                            state: format!("{:?}", info.state()),
                            connected_at_secs: info.connected_at.elapsed().as_secs(),
                            last_activity_secs: info.last_activity_nanos() / 1_000_000_000,
                        });
                    });
                    rows
                },
            )),
            checkpoint_stats: Some({
                let stats = Arc::clone(&ckpt_stats);
                Arc::new(move || -> (u64, u64, u64) {
                    (
                        stats
                            .checkpoints_completed
                            .load(std::sync::atomic::Ordering::Relaxed),
                        stats
                            .total_segments_deleted
                            .load(std::sync::atomic::Ordering::Relaxed),
                        stats
                            .last_checkpoint_lsn
                            .load(std::sync::atomic::Ordering::Relaxed),
                    )
                })
            }),
            vacuum_stats: Some({
                let stats = Arc::clone(&vac_stats);
                Arc::new(move || -> (u64, u64, u64) {
                    (
                        stats
                            .cycles_completed
                            .load(std::sync::atomic::Ordering::Relaxed),
                        stats
                            .tuples_reclaimed
                            .load(std::sync::atomic::Ordering::Relaxed),
                        stats
                            .pages_scanned
                            .load(std::sync::atomic::Ordering::Relaxed),
                    )
                })
            }),
            checkpoint_wake: ckpt_wake,
            alter_system_set: Some(Arc::new(
                move |key: &str, value: &str| -> std::result::Result<(), String> {
                    crate::config::ZyronConfig::write_auto_conf(&data_dir_for_alter, key, value)
                        .map_err(|e| e.to_string())
                },
            )),
            cdc_feed_stats: {
                let reg = Arc::clone(&cdc_registry_arc);
                Some(Arc::new(move || -> Vec<(u32, u64, u64, u32)> {
                    reg.list_feeds()
                }))
            },
            cdc_slot_stats: {
                let mgr = slot_mgr_arc.clone();
                Some(Arc::new(
                    move || -> Vec<(String, String, u64, u64, bool, u64)> {
                        mgr.as_ref()
                            .map(|m| {
                                m.list_slots()
                                    .into_iter()
                                    .map(|s| {
                                        (
                                            s.name,
                                            format!("{:?}", s.plugin),
                                            s.confirmed_lsn,
                                            s.restart_lsn,
                                            s.active,
                                            0u64,
                                        )
                                    })
                                    .collect()
                            })
                            .unwrap_or_default()
                    },
                ))
            },
            cdc_stream_stats: {
                let mgr = cdc_stream_mgr_arc.clone();
                Some(Arc::new(move || -> Vec<(String, u32, bool, String)> {
                    mgr.as_ref()
                        .map(|m| {
                            m.list_streams()
                                .into_iter()
                                .map(|s| (s.name, s.table_id, s.active, s.slot_name))
                                .collect()
                        })
                        .unwrap_or_default()
                }))
            },
            cdc_ingest_stats: {
                let mgr = cdc_ingest_mgr_arc.clone();
                Some(Arc::new(move || -> Vec<(String, u32, bool, u64, u64)> {
                    mgr.as_ref()
                        .map(|m| {
                            m.list_ingests()
                                .into_iter()
                                .map(|i| (i.name, i.target_table_id, i.active, 0u64, 0u64))
                                .collect()
                        })
                        .unwrap_or_default()
                }))
            },
            // CDC managers
            cdc_registry: Some(Arc::clone(&cdc_registry_arc)),
            slot_manager: slot_mgr_arc,
            publication_manager: pub_mgr_arc,
            cdc_stream_manager: cdc_stream_mgr_arc,
            cdc_ingest_manager: cdc_ingest_mgr_arc,
            // Pipeline managers
            trigger_manager: Some(Arc::clone(&trigger_mgr_arc)),
            udf_registry: Some(udf_reg_arc),
            uda_registry: Some(uda_reg_arc),
            procedure_registry: Some(proc_reg_arc),
            pipeline_manager: Some(pipeline_mgr_arc),
            schedule_manager: Some(sched_mgr_arc),
            event_dispatcher: Some(event_disp_arc),
            mv_manager: Some(mv_mgr_arc),
            // Streaming
            stream_job_manager: Some(stream_mgr_arc),
            // Versioning
            branch_manager: Some(branch_mgr_arc),
            // Search indexes
            fts_manager: Some(Arc::clone(&fts_mgr_arc)),
            vector_manager: Some(Arc::clone(&vec_mgr_arc)),
            graph_manager: Some(Arc::clone(&graph_mgr_arc)),
            spatial_manager: Some(Arc::clone(&spatial_mgr_arc)),
            // DML hooks: CDC capture + trigger dispatch
            cdc_hook: Some(Arc::new(hooks::CdcHookBridge::new(
                Arc::clone(&cdc_registry_arc),
                Arc::clone(&trigger_mgr_arc),
            )) as Arc<dyn zyron_executor::context::CdcHook>),
            dml_hook: Some(
                Arc::new(hooks::DmlHookBridge::new(Arc::clone(&trigger_mgr_arc)))
                    as Arc<dyn zyron_executor::context::DmlHook>,
            ),
            // Notification channels
            notification_channels: Some(notif_arc),
            // TLS upgrade support (disabled by default; enable via config).
            tls_mode: zyron_wire::tls::TlsMode::Disabled,
            tls_acceptor: None,
            endpoint_registrar: Some(Arc::clone(&endpoint_registrar)),
            subscription_runtimes: Arc::new(scc::HashMap::new()),
        });

        // -------------------------------------------------------------------
        // Install the admin executor on the shared HealthState so /admin/*
        // HTTP routes run against the live catalog, security manager,
        // endpoint registrar, and CDC registry.
        // -------------------------------------------------------------------
        {
            let admin_executor = Arc::new(
                crate::gateway::AdminExecutor::new(
                    Arc::clone(&catalog),
                    Some(Arc::clone(&security_manager_arc)),
                    Some(Arc::clone(&endpoint_registrar)),
                    Some(Arc::clone(&cdc_registry_arc)),
                )
                .with_storage(
                    Arc::clone(&disk_manager),
                    Arc::clone(&buffer_pool),
                    Arc::clone(&server_state.key_store),
                ),
            );
            self.health_state.set_admin_executor(admin_executor);
        }

        // -------------------------------------------------------------------
        // Install the dynamic endpoint SQL executor on the HealthState so
        // registered REST endpoints run against the real catalog, buffer
        // pool, disk manager, WAL, transaction manager, and security manager.
        // -------------------------------------------------------------------
        {
            let endpoint_executor = Arc::new(crate::gateway::endpoint_exec::EndpointExecutor::new(
                Arc::clone(&catalog),
                Arc::clone(&buffer_pool),
                Arc::clone(&disk_manager),
                Arc::clone(&wal),
                Arc::clone(&txn_manager),
                Some(Arc::clone(&security_manager_arc)),
            ));
            self.health_state.set_endpoint_executor(endpoint_executor);
        }

        // -------------------------------------------------------------------
        // External endpoint probes.
        // Walks the catalog external source and sink lists and verifies each
        // one can be opened. Probe failures log a warning and do not abort
        // startup so a single broken endpoint does not block the server.
        // -------------------------------------------------------------------
        verify_external_endpoints(&server_state).await;

        // -------------------------------------------------------------------
        // Streaming-job recovery.
        // Walks the catalog streaming-job list and respawns every Active job
        // so restarts do not lose running pipelines. Paused jobs stay paused
        // and only resume via ALTER STREAMING JOB RESUME.
        // -------------------------------------------------------------------
        if let Err(e) = recover_streaming_jobs(&server_state).await {
            error!("streaming job recovery failed: {}", e);
        }

        // -------------------------------------------------------------------
        // Zyron-to-Zyron startup recovery.
        // Walks the catalog publication, subscription, and endpoint lists so
        // restarts do not lose declarative state. Subscriptions that fail to
        // resume are marked Failed with the error stored on the entry.
        // -------------------------------------------------------------------
        recover_zyron_to_zyron(&server_state).await;

        // -------------------------------------------------------------------
        // Spawn Zyron-to-Zyron background workers. Each loop honors the
        // shared shutdown flag and exits cleanly when the server stops.
        // -------------------------------------------------------------------
        {
            use std::time::Duration;
            let catalog_ret = Arc::clone(&catalog);
            let cdc_reg_ret = Some(Arc::clone(&cdc_registry_arc));
            let sh_ret = Arc::clone(&self.shutdown);
            tokio::spawn(async move {
                background::publication_retention::publication_retention_loop(
                    catalog_ret,
                    cdc_reg_ret,
                    sh_ret,
                    background::publication_retention::DEFAULT_INTERVAL_SECS,
                )
                .await;
            });

            let catalog_reap = Arc::clone(&catalog);
            let sh_reap = Arc::clone(&self.shutdown);
            tokio::spawn(async move {
                background::dead_subscriber_reaper::dead_subscriber_reaper_loop(
                    catalog_reap,
                    sh_reap,
                    background::dead_subscriber_reaper::DEFAULT_INTERVAL_SECS,
                    Duration::from_secs(3600),
                )
                .await;
            });

            let sh_cred = Arc::clone(&self.shutdown);
            tokio::spawn(async move {
                background::credential_refresh::credential_refresh_loop(
                    sh_cred,
                    background::credential_refresh::DEFAULT_INTERVAL_SECS,
                    Duration::from_secs(
                        background::credential_refresh::DEFAULT_REFRESH_WINDOW_SECS,
                    ),
                    |_win| {},
                )
                .await;
            });

            let sh_dlq = Arc::clone(&self.shutdown);
            tokio::spawn(async move {
                background::dlq_ttl::dlq_ttl_loop(
                    sh_dlq,
                    background::dlq_ttl::DEFAULT_INTERVAL_SECS,
                    30,
                    |_cutoff| {},
                )
                .await;
            });

            let sh_host = Arc::clone(&self.shutdown);
            tokio::spawn(async move {
                background::host_health::host_health_monitor_loop(
                    sh_host,
                    background::host_health::DEFAULT_INTERVAL_SECS,
                    || {},
                )
                .await;
            });
        }

        // 11. Start health/metrics HTTP server
        let health_port = self.config.metrics.port;
        let health_shutdown = Arc::clone(&self.shutdown);
        let health_state = Arc::clone(&self.health_state);
        tokio::spawn(async move {
            health::start_health_server(health_port, health_state, health_shutdown).await;
        });

        // Mark startup complete
        self.health_state.mark_startup_complete();
        let startup_duration = start_time.elapsed();
        info!(
            "ZyronDB started in {:.1}ms on {}:{}",
            startup_duration.as_secs_f64() * 1000.0,
            self.config.server.host,
            self.config.server.port,
        );

        // 12. Start wire protocol server (runs accept loop until shutdown)
        self.health_state.mark_accepting();

        let wire_config = self.config.to_server_config();
        // Retain a reference for shutdown so the streaming-job stop path can
        // still reach the manager after the wire server task takes ownership.
        let state_for_shutdown = Arc::clone(&server_state);
        let wire_handle = tokio::spawn(async move {
            if let Err(e) = zyron_wire::start_server(&wire_config, server_state).await {
                error!("Wire protocol server error: {}", e);
            }
        });

        // 13. Wait for shutdown signal
        let reason = signal::wait_for_shutdown().await;
        info!("Shutdown signal received: {:?}", reason);

        // Graceful shutdown sequence
        self.shutdown.store(true, Ordering::Release);

        // Wait briefly for active queries to complete
        let drain_start = Instant::now();
        while self.session_mgr.active_count() > 0 && drain_start.elapsed().as_secs() < 5 {
            tokio::time::sleep(std::time::Duration::from_millis(100)).await;
        }

        if self.session_mgr.active_count() > 0 {
            warn!(
                "Shutdown timeout: {} sessions still active, aborting",
                self.session_mgr.active_count()
            );
        }

        // Persist FTS indexes to disk before stopping workers.
        {
            let fts_dir = self.config.storage.data_dir.join("fts");
            if let Err(e) = fts_mgr_arc.save_all(&fts_dir) {
                error!("FTS index persistence failed during shutdown: {e}");
            }
        }

        // Persist vector indexes to disk before stopping workers.
        {
            let vec_dir = self.config.storage.data_dir.join("vector");
            if let Err(e) = vec_mgr_arc.save_all(&vec_dir) {
                error!("Vector index persistence failed during shutdown: {e}");
            }
        }

        // Persist graph schemas to disk before stopping workers.
        {
            let graph_dir = self.config.storage.data_dir.join("graph");
            if let Err(e) = graph_mgr_arc.save_all(&graph_dir) {
                error!("Graph schema persistence failed during shutdown: {e}");
            }
        }

        // Persist spatial R-trees to disk before stopping workers. Save
        // errors are logged per index and do not abort shutdown, startup
        // rebuilds from the base table when a snapshot is missing.
        {
            let spatial_dir = self.config.storage.data_dir.join("spatial");
            let errors = spatial_mgr_arc.save_all(&spatial_dir);
            for (idx_id, err) in errors {
                warn!(
                    target: "zyron::shutdown",
                    "spatial index {} save failed: {}",
                    idx_id,
                    err
                );
            }
        }

        // Stop all running streaming jobs before shutting down background
        // workers so runner threads exit cleanly. shutdown_all drains the
        // handle map and joins every runner thread. External sink runners
        // call flush on their sink before breaking out of the loop so any
        // buffered rows are written out before the process exits.
        if let Some(ref mgr) = state_for_shutdown.stream_job_manager {
            let stopped = mgr.lock().shutdown_all();
            if stopped > 0 {
                info!("stopped {} streaming runner(s) during shutdown", stopped);
            }
        }

        // Stop every subscription runtime spawned by recovery or runtime DDL.
        // The tokio tasks cooperatively observe shutdown via their ZyronSource
        // pull loop, so abort is the definitive signal here. A short per-task
        // timeout bounds the wait so one stuck adapter cannot block process
        // exit. The adapter already persists last_seen_lsn into the catalog
        // as batches flow, so aborting mid-batch is safe: the next start
        // resumes from the last checkpointed LSN.
        {
            let mut keys: Vec<zyron_catalog::SubscriptionId> = Vec::new();
            state_for_shutdown.subscription_runtimes.iter_sync(|k, _| {
                keys.push(*k);
                true
            });
            let mut drained: Vec<(
                zyron_catalog::SubscriptionId,
                tokio::task::JoinHandle<()>,
            )> = Vec::with_capacity(keys.len());
            for k in keys {
                if let Some((_, h)) = state_for_shutdown
                    .subscription_runtimes
                    .remove_sync(&k)
                {
                    drained.push((k, h));
                }
            }
            let count = drained.len();
            for (id, handle) in drained {
                handle.abort();
                let waited = tokio::time::timeout(
                    std::time::Duration::from_millis(500),
                    handle,
                )
                .await;
                if let Err(_elapsed) = waited {
                    warn!(
                        target: "zyron::shutdown",
                        subscription_id = id.0,
                        "subscription task did not stop within timeout"
                    );
                }
            }
            if count > 0 {
                info!(
                    target: "zyron::shutdown",
                    "stopped {} subscription runtime(s) during shutdown",
                    count
                );
            }
        }

        // Stop background workers (runs final checkpoint)
        background.shutdown();

        // Abort the wire server
        wire_handle.abort();

        info!("ZyronDB shut down");
        Ok(())
    }
}

// -----------------------------------------------------------------------------
// Zyron-to-Zyron startup recovery
// -----------------------------------------------------------------------------

/// Re-warms Zyron-to-Zyron catalog state after a restart. Publications carry
/// no runtime state, subscriptions marked Failed are left as-is for the reaper
/// or an admin resume, and endpoints stay enabled so the gateway picks them
/// up on its next router rebuild. Security map entries are reloaded into the
/// in-memory auth store so request paths can resolve external identities.
pub async fn recover_zyron_to_zyron(state: &Arc<zyron_wire::connection::ServerState>) {
    let publications = state.catalog.list_publications();
    info!(
        target: "zyron::recovery",
        count = publications.len(),
        "loaded publications"
    );

    let subscriptions = state.catalog.list_subscriptions();
    info!(
        target: "zyron::recovery",
        count = subscriptions.len(),
        "loaded subscriptions"
    );

    // -------------------------------------------------------------------
    // Subscription resumption runs inside streaming-job recovery. A Zyron
    // subscription only has a meaningful destination when it is part of a
    // streaming job, so recover_streaming_jobs handles reopening the
    // connection, re-binding the job, and dispatching rows into the target
    // sink. Recovering the subscription here on its own would advance the
    // catalog LSN without delivering rows anywhere, silently dropping data
    // across restart. Standalone recovery is intentionally absent.
    // -------------------------------------------------------------------
    #[allow(clippy::collapsible_if)]
    {
        let active_count = subscriptions
            .iter()
            .filter(|s| s.state == zyron_catalog::SubscriptionState::Active)
            .count();
        if active_count > 0 {
            info!(
                target: "zyron::recovery",
                active_count,
                "subscriptions will resume through streaming job recovery"
            );
        }
    }

    let endpoints = state.catalog.list_endpoints();
    let enabled = endpoints.iter().filter(|e| e.enabled).count();
    info!(
        target: "zyron::recovery",
        total = endpoints.len(),
        enabled,
        "loaded endpoints"
    );

    // Register every enabled endpoint with the live gateway router. A failed
    // registration is logged and the loop continues so one bad entry does
    // not block the rest of the server.
    if let Some(ref registrar) = state.endpoint_registrar {
        let mut registered = 0usize;
        let mut failed = 0usize;
        for entry in &endpoints {
            if !entry.enabled {
                continue;
            }
            match registrar.register(entry).await {
                Ok(()) => registered += 1,
                Err(e) => {
                    failed += 1;
                    warn!(
                        target: "zyron::recovery",
                        name = %entry.name,
                        path = %entry.path,
                        error = %e,
                        "endpoint registration failed"
                    );
                }
            }
        }
        info!(
            target: "zyron::recovery",
            registered,
            failed,
            "gateway router populated from catalog"
        );
    }

    if let Some(ref sm) = state.security_manager {
        let entries: Vec<zyron_auth::SecurityMapEntry> = state
            .catalog
            .list_security_maps()
            .iter()
            .map(|e| zyron_auth::SecurityMapEntry {
                kind: map_cat_to_auth_kind(e.kind),
                key: e.key.clone(),
                role: zyron_auth::RoleId(e.role_id),
            })
            .collect();
        sm.security_map.load(entries);
    }

    for entry in state.catalog.list_external_sources() {
        if !matches!(entry.backend, zyron_catalog::ExternalBackend::Zyron) {
            continue;
        }
        info!(
            target: "zyron::recovery",
            name = %entry.name,
            uri = %entry.uri,
            "remote Zyron source observed"
        );
    }
}

// -----------------------------------------------------------------------------
// Catalog to auth security-map kind conversion
// -----------------------------------------------------------------------------

/// Maps the catalog's SecurityMapKind enum to the auth crate's equivalent.
/// Required because the two crates own parallel enum definitions so the
/// catalog does not take a dependency on auth.
fn map_cat_to_auth_kind(
    k: zyron_catalog::SecurityMapKind,
) -> zyron_auth::security_map::SecurityMapKind {
    match k {
        zyron_catalog::SecurityMapKind::K8sSa => zyron_auth::security_map::SecurityMapKind::K8sSa,
        zyron_catalog::SecurityMapKind::Jwt => zyron_auth::security_map::SecurityMapKind::Jwt,
        zyron_catalog::SecurityMapKind::MtlsSubject => {
            zyron_auth::security_map::SecurityMapKind::MtlsSubject
        }
        zyron_catalog::SecurityMapKind::MtlsFingerprint => {
            zyron_auth::security_map::SecurityMapKind::MtlsFingerprint
        }
    }
}

// -----------------------------------------------------------------------------
// Streaming-job startup recovery
// -----------------------------------------------------------------------------

/// Walks the catalog streaming-job list and respawns every Active job. Jobs in
/// the Paused or Failed state are skipped. A job that fails to respawn is
/// transitioned to Failed with the error string stored in last_error.
async fn recover_streaming_jobs(
    state: &Arc<zyron_wire::connection::ServerState>,
) -> zyron_common::Result<()> {
    let jobs = state.catalog.list_streaming_jobs();
    if jobs.is_empty() {
        return Ok(());
    }

    if state.stream_job_manager.is_none() {
        info!("streaming job manager not configured, skipping recovery");
        return Ok(());
    }

    let mut recovered = 0usize;
    let mut skipped = 0usize;
    let mut failed = 0usize;

    for entry in jobs {
        if entry.status != zyron_catalog::StreamingJobStatus::Active {
            skipped += 1;
            continue;
        }
        match respawn_streaming_job(&entry, state).await {
            Ok(()) => recovered += 1,
            Err(e) => {
                warn!("streaming job {} failed to restart: {}", entry.name, e);
                failed += 1;
                let _ = state
                    .catalog
                    .update_streaming_job_status(
                        entry.id,
                        zyron_catalog::StreamingJobStatus::Failed,
                        Some(e.to_string()),
                    )
                    .await;
            }
        }
    }

    info!(
        "streaming job recovery: {} respawned, {} skipped, {} failed",
        recovered, skipped, failed
    );
    Ok(())
}

/// Re-parses the stored CREATE STREAMING JOB SQL, re-binds it through the
/// planner, rehydrates the creator security context from the stored snapshot,
/// and hands the bound plan to the shared wire-level dispatch helper. The
/// helper covers every topology: Zyron-to-Zyron, external-to-Zyron,
/// Zyron-to-external, and external-to-external.
async fn respawn_streaming_job(
    entry: &zyron_catalog::StreamingJobEntry,
    state: &Arc<zyron_wire::connection::ServerState>,
) -> zyron_common::Result<()> {
    use zyron_common::ZyronError;

    let statements = zyron_parser::parse(&entry.select_sql)
        .map_err(|e| ZyronError::PlanError(format!("recovery parse failed: {e}")))?;
    let stmt = statements
        .into_iter()
        .find(|s| matches!(s, zyron_parser::Statement::CreateStreamingJob(_)))
        .ok_or_else(|| {
            ZyronError::PlanError("recovery: stored SQL has no CREATE STREAMING JOB".to_string())
        })?;

    let resolver = state.catalog.resolver(
        zyron_catalog::SYSTEM_DATABASE_ID,
        vec!["public".to_string()],
    );
    let mut binder = zyron_planner::Binder::new(resolver, &state.catalog);
    let bound = binder.bind(stmt).await?;
    let bsj = match bound {
        zyron_planner::BoundStatement::CreateStreamingJob(b) => b,
        _ => {
            return Err(ZyronError::PlanError(
                "recovery: unexpected bound variant".to_string(),
            ));
        }
    };

    // Rehydrate the creator security context from the stored snapshot.
    let security_manager = state.security_manager.as_ref().cloned().ok_or_else(|| {
        ZyronError::AuthenticationFailed("security manager not configured".to_string())
    })?;
    let mut off = 0usize;
    let snap =
        zyron_auth::SecurityContextSnapshot::from_bytes(&entry.creator_snapshot_bytes, &mut off)?;
    let limits = security_manager
        .query_limits
        .get_limits(&snap.effective_roles);
    let security_ctx = snap.into_context(limits);

    let spec = zyron_wire::ddl_dispatch::lower_bsj_to_spec(&bsj)?;

    let cdc_registry = state
        .cdc_registry
        .as_ref()
        .cloned()
        .ok_or_else(|| ZyronError::StreamingError("CDC registry not configured".to_string()))?;

    zyron_wire::ddl_dispatch::spawn_bound_streaming_job(
        &bsj,
        entry,
        spec,
        security_ctx,
        security_manager,
        cdc_registry,
        state,
    )
    .map_err(|e| match e {
        zyron_wire::messages::ProtocolError::Database(err) => err,
        other => ZyronError::StreamingError(format!("recovery dispatch failed: {other}")),
    })?;

    tracing::info!(
        target: "zyron::audit",
        event = "StreamingJobRespawned",
        job_id = entry.id.0,
        job_name = %entry.name,
        reason = "startup-recovery",
    );
    Ok(())
}

// ---------------------------------------------------------------------------
// External endpoint probes
// ---------------------------------------------------------------------------

/// Walks the catalog external source and sink lists and calls the streaming
/// probe helpers. A probe construct the OpenDAL operator locally, it does
/// not touch the remote endpoint. Probe failures log a warning and do not
/// abort startup. Credentials that must be unsealed are emitted as an audit
/// event so operators can trace credential access at boot.
async fn verify_external_endpoints(state: &Arc<zyron_wire::connection::ServerState>) {
    for entry in state.catalog.list_external_sources() {
        let creds = match unseal_entry_credentials_for_probe(
            entry.credential_key_id,
            entry.credential_ciphertext.as_deref(),
            state,
            &entry.name,
            true,
        ) {
            Ok(m) => m,
            Err(e) => {
                warn!(
                    "external source '{}' credential unseal failed: {}",
                    entry.name, e
                );
                continue;
            }
        };
        if let Err(e) = zyron_streaming::external_source::probe_external_source(&entry, creds) {
            warn!("external source '{}' may be unavailable: {}", entry.name, e);
        }
    }
    for entry in state.catalog.list_external_sinks() {
        let creds = match unseal_entry_credentials_for_probe(
            entry.credential_key_id,
            entry.credential_ciphertext.as_deref(),
            state,
            &entry.name,
            false,
        ) {
            Ok(m) => m,
            Err(e) => {
                warn!(
                    "external sink '{}' credential unseal failed: {}",
                    entry.name, e
                );
                continue;
            }
        };
        if let Err(e) = zyron_streaming::external_sink::ExternalRowSink::probe(&entry, creds) {
            warn!("external sink '{}' may be unavailable: {}", entry.name, e);
        }
    }
}

/// Probe-side credential unsealer. Emits an audit event so credential reads
/// during startup recovery are visible to operators.
fn unseal_entry_credentials_for_probe(
    key_id: Option<u32>,
    ciphertext: Option<&[u8]>,
    state: &Arc<zyron_wire::connection::ServerState>,
    object_name: &str,
    is_source: bool,
) -> zyron_common::Result<std::collections::HashMap<String, String>> {
    match (key_id, ciphertext) {
        (Some(kid), Some(ct)) => {
            let sealed = zyron_auth::SealedCredentials {
                key_id: kid,
                ciphertext: ct.to_vec(),
            };
            let opened = zyron_auth::open_credentials(&sealed, state.key_store.as_ref())?;
            tracing::info!(
                target: "zyron::audit",
                event = "ExternalCredentialRead",
                key_id = kid,
                object = %object_name,
                kind = if is_source { "source" } else { "sink" },
                reason = "startup-probe",
            );
            Ok(opened)
        }
        _ => Ok(std::collections::HashMap::new()),
    }
}

// -----------------------------------------------------------------------------
// Spatial index startup recovery
// -----------------------------------------------------------------------------

/// Extracts dims and srid from an IndexEntry parameters blob. Layout written
/// by handle_create_spatial_index is [u8 dims][u32 srid little-endian]. When
/// the blob is missing or too short the function falls back to a 2D, srid=0
/// index so recovery still runs against entries written before the header
/// was introduced.
fn parse_spatial_params(params: Option<&[u8]>) -> (u8, u32) {
    match params {
        Some(b) if b.len() >= 5 => {
            let dims = b[0].clamp(1, 4);
            let srid = u32::from_le_bytes([b[1], b[2], b[3], b[4]]);
            (dims, srid)
        }
        _ => (2, 0),
    }
}

/// Rebuilds a spatial index by scanning the base table's heap file, decoding
/// the geometry column of every live tuple, and inserting (mbr, rowid) pairs
/// into the live R-tree. Used when no snapshot file exists or when the saved
/// snapshot failed to load. Visibility filtering is deferred to query time,
/// this pass simply repopulates the tree with every reachable geometry.
async fn rebuild_spatial_index_from_table(
    mgr: &zyron_types::spatial_index::SpatialIndexManager,
    index_id: u32,
    dims: u8,
    buffer_pool: &Arc<BufferPool>,
    disk: &Arc<DiskManager>,
    idx: &zyron_catalog::schema::IndexEntry,
    table: &zyron_catalog::schema::TableEntry,
) -> zyron_common::Result<()> {
    use zyron_common::ZyronError;

    let col_id = idx.columns.first().map(|c| c.column_id).ok_or_else(|| {
        ZyronError::ExecutionError(format!("spatial index {} has no indexed column", index_id))
    })?;

    let col_ordinal = table
        .columns
        .iter()
        .position(|c| c.id == col_id)
        .ok_or_else(|| {
            ZyronError::ExecutionError(format!(
                "spatial index {} references unknown column {}",
                index_id, col_id.0
            ))
        })?;

    let heap = zyron_storage::HeapFile::new(
        Arc::clone(disk),
        Arc::clone(buffer_pool),
        zyron_storage::HeapFileConfig {
            heap_file_id: table.heap_file_id,
            fsm_file_id: table.fsm_file_id,
        },
    )?;
    heap.init_cache().await?;

    let guard = heap.scan()?;
    let mut count: u64 = 0;
    guard.for_each(|tid, view| {
        let data = view.data;
        let Some(geom_bytes) = column_bytes_at(data, col_ordinal, &table.columns) else {
            return;
        };
        let Ok(geom) = zyron_types::geospatial::decode_wkb(geom_bytes) else {
            return;
        };
        let mbr = zyron_types::spatial_index::mbr_from_geometry(&geom, dims);
        let rowid = match zyron_search::encode_doc_id(tid.page_id.page_num, tid.slot_id) {
            Ok(r) => r,
            Err(_) => return,
        };
        if mgr.insert_mbr(index_id, mbr, rowid).is_ok() {
            count += 1;
        }
    });

    info!(
        target: "zyron::recovery",
        "spatial index {} rebuilt from table scan, rows {}",
        index_id,
        count
    );
    Ok(())
}

/// Extracts the raw bytes of the column at `ordinal` from an NSM-encoded
/// tuple data slice. Skips preceding columns using each type's fixed size or
/// the 4-byte length prefix used for variable-length columns. Returns None
/// when the column is null or when the tuple data is shorter than expected.
fn column_bytes_at<'a>(
    data: &'a [u8],
    ordinal: usize,
    columns: &[zyron_catalog::ColumnEntry],
) -> Option<&'a [u8]> {
    let num_cols = columns.len();
    if ordinal >= num_cols {
        return None;
    }
    let null_bitmap_len = (num_cols + 7) / 8;
    if data.len() < null_bitmap_len {
        return None;
    }
    let null_bitmap = &data[..null_bitmap_len];
    let mut offset = null_bitmap_len;

    for (i, col) in columns.iter().enumerate() {
        let is_null = (null_bitmap[i / 8] >> (i % 8)) & 1 == 1;

        if let Some(fixed_size) = col.type_id.fixed_size() {
            if offset + fixed_size > data.len() {
                return None;
            }
            if i == ordinal {
                if is_null {
                    return None;
                }
                return Some(&data[offset..offset + fixed_size]);
            }
            offset += fixed_size;
        } else {
            if offset + 4 > data.len() {
                return None;
            }
            let len = u32::from_le_bytes([
                data[offset],
                data[offset + 1],
                data[offset + 2],
                data[offset + 3],
            ]) as usize;
            offset += 4;
            if offset + len > data.len() {
                return None;
            }
            if i == ordinal {
                if is_null {
                    return None;
                }
                return Some(&data[offset..offset + len]);
            }
            offset += len;
        }
    }
    None
}

// -----------------------------------------------------------------------------
// Subscription recovery tests
// -----------------------------------------------------------------------------
