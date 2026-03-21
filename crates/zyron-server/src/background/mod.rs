//! Background task scheduler coordinating all maintenance workers.
//!
//! Provides a single struct that starts and stops all background workers
//! in the correct order. On shutdown, runs a final checkpoint for
//! zero-replay restart.

pub mod checkpoint;
pub mod stats;
pub mod vacuum;
pub mod wal_archiver;

use std::path::PathBuf;
use std::sync::Arc;

use tracing::info;

use zyron_buffer::BufferPool;
use zyron_catalog::Catalog;
use zyron_storage::CheckpointCoordinator;
use zyron_storage::DiskManager;
use zyron_storage::checkpoint::CheckpointTracker;
use zyron_storage::txn::TransactionManager;
use zyron_wal::WalWriter;

use zyron_buffer::BackgroundWriter;

use self::checkpoint::{CheckpointWorker, CheckpointWorkerConfig};
use self::stats::{StatsCollector, StatsCollectorConfig};
use self::vacuum::{VacuumWorker, VacuumWorkerConfig};
use self::wal_archiver::{WalArchiver, WalArchiverConfig};

/// Coordinates all background maintenance workers.
pub struct BackgroundWorkers {
    checkpoint: CheckpointWorker,
    stats: StatsCollector,
    vacuum: VacuumWorker,
    wal_archiver: Option<WalArchiver>,
}

impl BackgroundWorkers {
    /// Starts all background workers.
    pub fn start(
        catalog: Arc<Catalog>,
        wal: Arc<WalWriter>,
        buffer_pool: Arc<BufferPool>,
        background_writer: Arc<BackgroundWriter>,
        disk_manager: Arc<DiskManager>,
        txn_manager: Arc<TransactionManager>,
        tracker: Arc<CheckpointTracker>,
        ckpt_config: CheckpointWorkerConfig,
        stats_config: StatsCollectorConfig,
        vacuum_config: VacuumWorkerConfig,
        wal_dir: PathBuf,
        archive_dir: Option<PathBuf>,
    ) -> Self {
        info!("Starting background workers");

        // Checkpoint coordinator (from zyron-storage)
        let coord_config = zyron_storage::CheckpointCoordinatorConfig {
            checkpoint_timeout_secs: 60,
            checkpoint_interval_secs: ckpt_config.max_interval_secs,
            max_wal_segments: 8,
        };
        let coordinator = Arc::new(CheckpointCoordinator::new(
            buffer_pool.clone(),
            wal.clone(),
            background_writer,
            tracker,
            coord_config,
        ));

        let checkpoint = CheckpointWorker::start(coordinator, wal.clone(), ckpt_config);

        let stats = StatsCollector::start(catalog.clone(), stats_config);

        let vacuum = VacuumWorker::start(
            catalog,
            txn_manager,
            disk_manager,
            buffer_pool,
            wal,
            vacuum_config,
        );

        let wal_archiver = archive_dir.map(|dir| {
            WalArchiver::start(WalArchiverConfig {
                wal_dir,
                archive_dir: dir,
                retention_count: 100,
                interval_secs: 30,
            })
        });

        info!("All background workers started");

        Self {
            checkpoint,
            stats,
            vacuum,
            wal_archiver,
        }
    }

    /// Returns a reference to the checkpoint worker (for stats access).
    pub fn checkpoint(&self) -> &CheckpointWorker {
        &self.checkpoint
    }

    /// Gracefully shuts down all workers.
    /// Runs a final checkpoint before stopping the checkpoint worker.
    pub fn shutdown(&mut self) {
        info!("Shutting down background workers");

        // Run final checkpoint for zero-replay restart
        if let Err(e) = self.checkpoint.final_checkpoint() {
            tracing::error!(
                "Final checkpoint failed during shutdown: {}. WAL replay will be needed on restart.",
                e
            );
        }

        // Stop workers in reverse dependency order
        if let Some(ref mut archiver) = self.wal_archiver {
            archiver.shutdown();
        }
        self.vacuum.shutdown();
        self.stats.shutdown();
        self.checkpoint.shutdown();

        info!("All background workers stopped");
    }
}
