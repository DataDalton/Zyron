//! ZyronDB server integration.
//!
//! Orchestrates all subsystems into a complete database server: configuration,
//! session management, background workers, metrics, health endpoints, and
//! graceful shutdown.

pub mod background;
pub mod config;
pub mod health;
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

        // Build ServerState for zyron-wire
        let server_state = Arc::new(ServerState {
            catalog: Arc::clone(&catalog),
            wal: Arc::clone(&wal),
            buffer_pool: Arc::clone(&buffer_pool),
            disk_manager: Arc::clone(&disk_manager),
            txn_manager: Arc::clone(&txn_manager),
            security_manager: Some(Arc::new(security_manager)),
        });

        // 9. Create CheckpointTracker
        let tracker = Arc::new(CheckpointTracker::new());

        // 10. Start background workers
        let ckpt_config = CheckpointWorkerConfig {
            wal_bytes_threshold: self.config.checkpoint.wal_bytes_threshold,
            max_interval_secs: self.config.checkpoint.max_interval_secs as u64,
            min_interval_secs: self.config.checkpoint.min_interval_secs as u64,
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
            VacuumWorkerConfig::default(),
            wal_dir,
            None, // WAL archiving disabled unless configured
        );

        // 11. Start health/metrics HTTP server
        let health_port = self.config.server.health_port;
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

        // Stop background workers (runs final checkpoint)
        background.shutdown();

        // Abort the wire server
        wire_handle.abort();

        info!("ZyronDB shut down");
        Ok(())
    }
}
