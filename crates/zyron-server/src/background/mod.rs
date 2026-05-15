//! Background task scheduler coordinating all maintenance workers.
//!
//! Provides a single struct that starts and stops all background workers
//! in the correct order. On shutdown, runs a final checkpoint for
//! zero-replay restart.

pub mod cdc_writer;
pub mod checkpoint;
pub mod credential_refresh;
pub mod dead_subscriber_reaper;
pub mod dlq_ttl;
pub mod feature_materialization;
pub mod feature_materialization_impl;
pub mod host_health;
pub mod mv_refresh;
pub mod publication_retention;
pub mod quota_gossip;
pub mod stats;
pub mod stream_monitor;
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

use self::cdc_writer::{CdcWriter, CdcWriterConfig};
use self::checkpoint::{CheckpointWorker, CheckpointWorkerConfig};
use self::feature_materialization::{
    FeatureMaterializationConfig, FeatureMaterializationWorker,
};
use self::mv_refresh::{MvRefreshConfig, MvRefreshWorker};
use self::quota_gossip::{
    NoopTransport, QuotaGossipConfig, QuotaGossipTransport, QuotaGossipWorker,
};
use self::stats::{StatsCollector, StatsCollectorConfig};
use self::stream_monitor::{StreamMonitor, StreamMonitorConfig};
use self::vacuum::{VacuumWorker, VacuumWorkerConfig};
use self::wal_archiver::{WalArchiver, WalArchiverConfig};

use zyron_types::scheduling::QuotaRegistry;

/// Coordinates all background maintenance workers.
pub struct BackgroundWorkers {
    checkpoint: CheckpointWorker,
    stats: StatsCollector,
    vacuum: VacuumWorker,
    wal_archiver: Option<WalArchiver>,
    cdc_writer: CdcWriter,
    mv_refresh: MvRefreshWorker,
    feature_materialization: FeatureMaterializationWorker,
    stream_monitor: StreamMonitor,
    quota_gossip: Option<QuotaGossipWorker>,
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
        cdc_registry: Option<Arc<zyron_cdc::CdfRegistry>>,
        stream_job_manager: Option<Arc<parking_lot::Mutex<zyron_streaming::job::StreamJobManager>>>,
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

        let stats = StatsCollector::start(
            catalog.clone(),
            disk_manager.clone(),
            buffer_pool.clone(),
            stats_config,
        );

        let catalog_for_mv = catalog.clone();
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

        let cdc_writer =
            CdcWriter::start_with_registry(CdcWriterConfig::default(), cdc_registry.clone());
        let mv_refresh =
            MvRefreshWorker::start_with_catalog(MvRefreshConfig::default(), Some(catalog_for_mv));
        let feature_materialization =
            FeatureMaterializationWorker::start(FeatureMaterializationConfig::default());
        let stream_monitor = StreamMonitor::start_with_manager(
            StreamMonitorConfig::default(),
            stream_job_manager.clone(),
        );

        info!("All background workers started");

        Self {
            checkpoint,
            stats,
            vacuum,
            wal_archiver,
            cdc_writer,
            mv_refresh,
            feature_materialization,
            stream_monitor,
            quota_gossip: None,
        }
    }

    /// Returns the feature materialization worker stats Arc
    pub fn feature_materialization_stats(
        &self,
    ) -> Arc<feature_materialization::FeatureMaterializationStats> {
        self.feature_materialization.stats()
    }

    /// Attaches a QuotaGossip worker to the running set. Call after `start`
    /// when the server has assembled its QuotaRegistry and (optional) peer
    /// transport. Pass NoopTransport to enable the worker as a no-op while
    /// reserving the gossip schedule, or a real transport once peers are wired
    pub fn attach_quota_gossip(
        &mut self,
        registry: Arc<QuotaRegistry>,
        transport: Arc<dyn QuotaGossipTransport>,
        config: QuotaGossipConfig,
    ) {
        if self.quota_gossip.is_none() {
            self.quota_gossip = Some(QuotaGossipWorker::start(registry, transport, config));
            info!("QuotaGossip worker attached");
        }
    }

    /// Convenience: attach with the default no-op transport. Useful when the
    /// server is single-node or peer transport is not yet configured
    pub fn attach_quota_gossip_default(&mut self, registry: Arc<QuotaRegistry>) {
        self.attach_quota_gossip(
            registry,
            Arc::new(NoopTransport),
            QuotaGossipConfig::default(),
        );
    }

    /// Returns a reference to the checkpoint worker (for stats access).
    pub fn checkpoint(&self) -> &CheckpointWorker {
        &self.checkpoint
    }

    /// Returns the checkpoint worker stats Arc.
    pub fn checkpoint_stats(&self) -> Arc<checkpoint::CheckpointWorkerStats> {
        Arc::clone(self.checkpoint.stats())
    }

    /// Returns the vacuum worker stats Arc.
    pub fn vacuum_stats(&self) -> Arc<vacuum::VacuumStats> {
        Arc::clone(self.vacuum.stats())
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
        if let Some(ref mut gossip) = self.quota_gossip {
            gossip.shutdown();
        }
        self.stream_monitor.shutdown();
        self.feature_materialization.shutdown();
        self.mv_refresh.shutdown();
        self.cdc_writer.shutdown();
        if let Some(ref mut archiver) = self.wal_archiver {
            archiver.shutdown();
        }
        self.vacuum.shutdown();
        self.stats.shutdown();
        self.checkpoint.shutdown();

        info!("All background workers stopped");
    }
}
