//! Background task scheduler coordinating all maintenance workers.
//!
//! Provides a single struct that starts and stops all background workers
//! in the correct order. On shutdown, runs a final checkpoint for
//! zero-replay restart.

pub mod cdc_ingest;
pub mod cdc_stream_pump;
pub mod cdc_writer;
pub mod checkpoint;
pub mod compaction;
pub mod credential_refresh;
pub mod dead_subscriber_reaper;
pub mod dlq_ttl;
pub mod feature_materialization;
pub mod feature_materialization_impl;
pub mod host_health;
pub mod lake_clustering;
pub mod lake_follower;
pub mod mv_refresh;
pub mod publication_retention;
pub mod quota_gossip;
pub mod recycle_reaper;
pub mod retention;
pub mod schedule;
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
use self::compaction::{CompactionWorker, CompactionWorkerConfig};
use self::feature_materialization::{FeatureMaterializationConfig, FeatureMaterializationWorker};
use self::mv_refresh::{MvRefreshConfig, MvRefreshWorker};
use self::quota_gossip::{
    NoopTransport, QuotaGossipConfig, QuotaGossipTransport, QuotaGossipWorker,
};
use self::retention::{RetentionWorker, RetentionWorkerConfig};
use self::schedule::{ScheduleWorker, ScheduleWorkerConfig};
use self::stats::{StatsCollector, StatsCollectorConfig};
use self::stream_monitor::{StreamMonitor, StreamMonitorConfig};
use self::vacuum::{VacuumWorker, VacuumWorkerConfig};
use self::wal_archiver::{WalArchiver, WalArchiverConfig};
use crate::metrics::MetricsRegistry;

use zyron_types::scheduling::QuotaRegistry;

/// Coordinates all background maintenance workers.
pub struct BackgroundWorkers {
    checkpoint: CheckpointWorker,
    stats: StatsCollector,
    vacuum: VacuumWorker,
    compaction: CompactionWorker,
    retention: RetentionWorker,
    schedule: ScheduleWorker,
    wal_archiver: Option<WalArchiver>,
    cdc_writer: CdcWriter,
    mv_refresh: MvRefreshWorker,
    feature_materialization: FeatureMaterializationWorker,
    stream_monitor: StreamMonitor,
    quota_gossip: Option<QuotaGossipWorker>,
    lake_clustering: Option<self::lake_clustering::LakeClusteringWorker>,
    lake_follower: Option<self::lake_follower::LakeFollowerWorker>,
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
        compaction_config: CompactionWorkerConfig,
        metrics: Option<Arc<MetricsRegistry>>,
        wal_dir: PathBuf,
        data_dir: PathBuf,
        archive_dir: Option<PathBuf>,
        cdc_registry: Option<Arc<zyron_cdc::CdfRegistry>>,
        stream_job_manager: Option<Arc<parking_lot::Mutex<zyron_streaming::job::StreamJobManager>>>,
        btree_indexes: Arc<scc::HashMap<u32, Arc<zyron_storage::BTreeIndex>>>,
        doc_registry: Arc<zyron_common::DocRegistry>,
        table_io_stats: Arc<zyron_common::TableIOStatsRegistry>,
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

        // Persist the commit-status map and the retention clock as part of every
        // checkpoint, after the page flush and before WAL truncation, so an
        // aborted transaction's status and the time-to-LSN samples survive a
        // crash even once their WAL is reclaimed.
        {
            let clog_status = Arc::clone(txn_manager.status_map());
            let retention_clock = Arc::clone(txn_manager.retention_clock());
            let clog_dir = data_dir.clone();
            coordinator.set_pre_truncate_hook(Arc::new(move || {
                clog_status.persist(&clog_dir)?;
                retention_clock.persist(&clog_dir)
            }));
        }

        let checkpoint = CheckpointWorker::start(coordinator, wal.clone(), ckpt_config);

        let stats = StatsCollector::start(
            catalog.clone(),
            disk_manager.clone(),
            buffer_pool.clone(),
            stats_config,
        );

        let catalog_for_mv = catalog.clone();
        let retention = RetentionWorker::start(
            catalog.clone(),
            txn_manager.clone(),
            wal.clone(),
            buffer_pool.clone(),
            disk_manager.clone(),
            RetentionWorkerConfig::default(),
        );
        let schedule = ScheduleWorker::start(
            catalog.clone(),
            txn_manager.clone(),
            wal.clone(),
            buffer_pool.clone(),
            disk_manager.clone(),
            ScheduleWorkerConfig::default(),
        );
        let compaction = CompactionWorker::start(
            catalog.clone(),
            txn_manager.clone(),
            disk_manager.clone(),
            buffer_pool.clone(),
            wal.clone(),
            metrics,
            compaction_config,
            doc_registry,
            Arc::clone(&btree_indexes),
        );
        let vacuum = VacuumWorker::start(
            catalog,
            txn_manager,
            disk_manager,
            buffer_pool,
            wal,
            btree_indexes,
            table_io_stats,
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
            compaction,
            retention,
            schedule,
            wal_archiver,
            cdc_writer,
            mv_refresh,
            feature_materialization,
            stream_monitor,
            quota_gossip: None,
            lake_clustering: None,
            lake_follower: None,
        }
    }

    /// Attaches the lake follower. Returns without starting a thread off
    /// the lake tier, where there are no lake tables to follow
    pub fn attach_lake_follower(
        &mut self,
        mode: zyron_common::DeploymentMode,
        catalog: Arc<Catalog>,
        peers: Arc<parking_lot::RwLock<Arc<zyron_common::PeerRegistry>>>,
        config: self::lake_follower::LakeFollowerConfig,
    ) {
        if self.lake_follower.is_none() {
            self.lake_follower =
                self::lake_follower::LakeFollowerWorker::start(mode, catalog, peers, config);
        }
    }

    /// Returns the follower's counters, None off the lake tier
    pub fn lake_follower_stats(&self) -> Option<Arc<self::lake_follower::LakeFollowerStats>> {
        self.lake_follower.as_ref().map(|w| Arc::clone(w.stats()))
    }

    /// Attaches the Adaptive Clustering worker. Returns without starting a
    /// thread on a node that does not run the lake tier, so a db node pays
    /// nothing for a tier it does not host
    pub fn attach_lake_clustering(
        &mut self,
        mode: zyron_common::DeploymentMode,
        catalog: Arc<Catalog>,
        metrics: Option<Arc<zyron_common::LabeledMetrics>>,
        config: self::lake_clustering::LakeClusteringConfig,
    ) {
        if self.lake_clustering.is_none() {
            self.lake_clustering =
                self::lake_clustering::LakeClusteringWorker::start(mode, catalog, metrics, config);
        }
    }

    /// Returns the clustering worker's counters, None off the lake tier
    pub fn lake_clustering_stats(&self) -> Option<Arc<self::lake_clustering::LakeClusteringStats>> {
        self.lake_clustering.as_ref().map(|w| Arc::clone(w.stats()))
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

    /// Returns a cloneable handle that forces a checkpoint and blocks until it
    /// completes. Used to wire the SQL CHECKPOINT command.
    pub fn checkpoint_trigger(&self) -> checkpoint::CheckpointTrigger {
        self.checkpoint.trigger()
    }

    /// Returns the checkpoint worker stats Arc.
    pub fn checkpoint_stats(&self) -> Arc<checkpoint::CheckpointWorkerStats> {
        Arc::clone(self.checkpoint.stats())
    }

    /// Returns the vacuum worker stats Arc.
    pub fn vacuum_stats(&self) -> Arc<vacuum::VacuumStats> {
        Arc::clone(self.vacuum.stats())
    }

    /// Returns the compaction worker stats Arc.
    pub fn compaction_stats(&self) -> Arc<compaction::CompactionStats> {
        Arc::clone(self.compaction.stats())
    }

    /// Gracefully shuts down all workers.
    /// Runs a final checkpoint before stopping the checkpoint worker.
    pub fn shutdown(&mut self) {
        info!("Shutting down background workers");

        // Stop compaction before the final checkpoint so no fold transition
        // is in flight while the checkpoint runs.
        self.compaction.shutdown();

        // Run final checkpoint for zero-replay restart
        if let Err(e) = self.checkpoint.final_checkpoint() {
            tracing::error!(
                "Final checkpoint failed during shutdown: {}. WAL replay will be needed on restart.",
                e
            );
        }

        // Stop workers in reverse dependency order
        if let Some(ref mut follower) = self.lake_follower {
            follower.shutdown();
        }
        if let Some(ref mut clustering) = self.lake_clustering {
            clustering.shutdown();
        }
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
        self.schedule.shutdown();
        self.retention.shutdown();
        self.vacuum.shutdown();
        self.stats.shutdown();
        self.checkpoint.shutdown();

        info!("All background workers stopped");
    }
}
