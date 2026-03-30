//! Stream job management: lifecycle, configuration, and execution.
//!
//! StreamJobManager tracks running streaming jobs via scc::HashMap.
//! Each StreamJob owns an OperatorChain and CheckpointCoordinator.
//! Jobs transition through Created -> Running -> Paused/Failed/Cancelled
//! states. Savepoints allow snapshotting a running job for later resume.

use std::sync::Arc;
use std::sync::atomic::{AtomicU8, AtomicU32, Ordering};

use zyron_common::{Result, ZyronError};

use crate::checkpoint::{CheckpointConfig, CheckpointId, StreamCheckpoint};
use crate::stream_operator::OperatorChain;

// ---------------------------------------------------------------------------
// StreamJobId
// ---------------------------------------------------------------------------

/// Unique identifier for a streaming job, allocated from an atomic counter.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct StreamJobId(pub u32);

impl StreamJobId {
    pub fn new(id: u32) -> Self {
        Self(id)
    }

    pub fn as_u32(&self) -> u32 {
        self.0
    }
}

impl std::fmt::Display for StreamJobId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "job-{}", self.0)
    }
}

// ---------------------------------------------------------------------------
// ProcessingGuarantee
// ---------------------------------------------------------------------------

/// Processing guarantee level for a streaming job.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProcessingGuarantee {
    /// Each record processed at least once. Duplicates possible on failure.
    AtLeastOnce,
    /// Each record processed exactly once. Requires checkpoint coordination.
    ExactlyOnce,
}

// ---------------------------------------------------------------------------
// StreamJobStatus
// ---------------------------------------------------------------------------

/// Current status of a streaming job.
#[derive(Debug, Clone, PartialEq)]
pub enum StreamJobStatus {
    Created,
    Running,
    Paused,
    Failed { reason: String },
    Cancelled,
}

impl StreamJobStatus {
    pub fn is_active(&self) -> bool {
        matches!(self, StreamJobStatus::Running | StreamJobStatus::Paused)
    }

    fn to_u8(&self) -> u8 {
        match self {
            StreamJobStatus::Created => 0,
            StreamJobStatus::Running => 1,
            StreamJobStatus::Paused => 2,
            StreamJobStatus::Failed { .. } => 3,
            StreamJobStatus::Cancelled => 4,
        }
    }

    fn from_u8(v: u8) -> Self {
        match v {
            0 => StreamJobStatus::Created,
            1 => StreamJobStatus::Running,
            2 => StreamJobStatus::Paused,
            3 => StreamJobStatus::Failed {
                reason: String::new(),
            },
            4 => StreamJobStatus::Cancelled,
            _ => StreamJobStatus::Created,
        }
    }
}

impl std::fmt::Display for StreamJobStatus {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            StreamJobStatus::Created => write!(f, "CREATED"),
            StreamJobStatus::Running => write!(f, "RUNNING"),
            StreamJobStatus::Paused => write!(f, "PAUSED"),
            StreamJobStatus::Failed { reason } => write!(f, "FAILED: {reason}"),
            StreamJobStatus::Cancelled => write!(f, "CANCELLED"),
        }
    }
}

// ---------------------------------------------------------------------------
// StreamJobConfig
// ---------------------------------------------------------------------------

/// Configuration for a streaming job.
#[derive(Debug, Clone)]
pub struct StreamJobConfig {
    pub name: String,
    pub checkpoint_config: CheckpointConfig,
    pub processing_guarantee: ProcessingGuarantee,
    pub parallelism: u32,
}

impl StreamJobConfig {
    pub fn new(name: String) -> Self {
        Self {
            name,
            checkpoint_config: CheckpointConfig::default(),
            processing_guarantee: ProcessingGuarantee::AtLeastOnce,
            parallelism: 1,
        }
    }

    pub fn with_guarantee(mut self, guarantee: ProcessingGuarantee) -> Self {
        self.processing_guarantee = guarantee;
        self
    }

    pub fn with_parallelism(mut self, parallelism: u32) -> Self {
        self.parallelism = parallelism.max(1);
        self
    }

    pub fn with_checkpoint_config(mut self, config: CheckpointConfig) -> Self {
        self.checkpoint_config = config;
        self
    }
}

// ---------------------------------------------------------------------------
// StreamJob
// ---------------------------------------------------------------------------

/// A running streaming job. Owns the operator chain and checkpoint coordinator.
/// Status is stored as an AtomicU8 for lock-free reads. The error message
/// for the Failed state is stored in a separate Mutex, only locked on failure.
pub struct StreamJob {
    pub id: StreamJobId,
    pub config: StreamJobConfig,
    status: AtomicU8,
    error_message: parking_lot::Mutex<Option<String>>,
    pub operator_chain: OperatorChain,
    /// Completed checkpoints.
    pub checkpoints: parking_lot::Mutex<Vec<StreamCheckpoint>>,
    /// Start time in milliseconds.
    pub start_time_ms: std::sync::atomic::AtomicI64,
}

impl StreamJob {
    pub fn new(id: StreamJobId, config: StreamJobConfig, operator_chain: OperatorChain) -> Self {
        Self {
            id,
            config,
            status: AtomicU8::new(StreamJobStatus::Created.to_u8()),
            error_message: parking_lot::Mutex::new(None),
            operator_chain,
            checkpoints: parking_lot::Mutex::new(Vec::new()),
            start_time_ms: std::sync::atomic::AtomicI64::new(0),
        }
    }

    /// Returns the current job status. Lock-free for non-Failed states.
    /// For the Failed state, locks the error_message mutex to read the reason.
    pub fn status(&self) -> StreamJobStatus {
        let code = self.status.load(Ordering::Acquire);
        if code == 3 {
            let reason = self.error_message.lock().clone().unwrap_or_default();
            StreamJobStatus::Failed { reason }
        } else {
            StreamJobStatus::from_u8(code)
        }
    }

    /// Transitions the job to Running.
    pub fn start(&self, current_time_ms: i64) -> Result<()> {
        let current = self.status.load(Ordering::Acquire);
        match current {
            0 | 2 => {
                // Created or Paused -> Running
                self.status.store(1, Ordering::Release);
                self.start_time_ms.store(current_time_ms, Ordering::Relaxed);
                Ok(())
            }
            _ => Err(ZyronError::StreamingError(format!(
                "cannot start job in state {}",
                StreamJobStatus::from_u8(current)
            ))),
        }
    }

    /// Pauses the job.
    pub fn pause(&self) -> Result<()> {
        let current = self.status.load(Ordering::Acquire);
        if current == 1 {
            // Running -> Paused
            self.status.store(2, Ordering::Release);
            Ok(())
        } else {
            Err(ZyronError::StreamingError(format!(
                "cannot pause job in state {}",
                StreamJobStatus::from_u8(current)
            )))
        }
    }

    /// Cancels the job.
    pub fn cancel(&self) -> Result<()> {
        let current = self.status.load(Ordering::Acquire);
        if current == 1 || current == 2 {
            // Running or Paused -> Cancelled
            self.status.store(4, Ordering::Release);
            Ok(())
        } else {
            Err(ZyronError::StreamingError(format!(
                "cannot cancel job in state {}",
                StreamJobStatus::from_u8(current)
            )))
        }
    }

    /// Marks the job as failed.
    pub fn fail(&self, reason: String) {
        *self.error_message.lock() = Some(reason);
        self.status.store(3, Ordering::Release);
    }

    /// Adds a completed checkpoint.
    pub fn add_checkpoint(&self, checkpoint: StreamCheckpoint) {
        self.checkpoints.lock().push(checkpoint);
    }

    /// Returns the latest checkpoint ID, if any.
    pub fn latest_checkpoint(&self) -> Option<CheckpointId> {
        self.checkpoints.lock().last().map(|c| c.checkpoint_id)
    }
}

// ---------------------------------------------------------------------------
// StreamJobManager
// ---------------------------------------------------------------------------

/// Manages the lifecycle of all streaming jobs.
/// Uses scc::HashMap for lock-free concurrent job tracking.
pub struct StreamJobManager {
    jobs: scc::HashMap<u32, Arc<StreamJob>>,
    next_id: AtomicU32,
}

impl StreamJobManager {
    pub fn new() -> Self {
        Self {
            jobs: scc::HashMap::new(),
            next_id: AtomicU32::new(1),
        }
    }

    /// Creates and registers a new streaming job. Returns the job ID.
    pub fn create_job(
        &self,
        config: StreamJobConfig,
        operator_chain: OperatorChain,
    ) -> Result<StreamJobId> {
        let id_val = self.next_id.fetch_add(1, Ordering::Relaxed);
        let job_id = StreamJobId::new(id_val);
        let job = StreamJob::new(job_id, config, operator_chain);
        let _ = self.jobs.insert_sync(id_val, Arc::new(job));
        Ok(job_id)
    }

    /// Starts a job.
    pub fn start(&self, job_id: StreamJobId, current_time_ms: i64) -> Result<()> {
        let job = self.get_job(job_id)?;
        job.start(current_time_ms)
    }

    /// Stops (cancels) a job.
    pub fn stop(&self, job_id: StreamJobId) -> Result<()> {
        let job = self.get_job(job_id)?;
        job.cancel()
    }

    /// Pauses a running job.
    pub fn pause(&self, job_id: StreamJobId) -> Result<()> {
        let job = self.get_job(job_id)?;
        job.pause()
    }

    /// Resumes a paused job.
    pub fn resume(&self, job_id: StreamJobId, current_time_ms: i64) -> Result<()> {
        let job = self.get_job(job_id)?;
        job.start(current_time_ms)
    }

    /// Returns the status of a job. Reads the AtomicU8 status without locking.
    pub fn status(&self, job_id: StreamJobId) -> Result<StreamJobStatus> {
        let job = self.get_job(job_id)?;
        Ok(job.status())
    }

    /// Lists all jobs with their names and statuses.
    pub fn list(&self) -> Vec<(StreamJobId, String, StreamJobStatus)> {
        let mut result = Vec::new();
        self.jobs.iter_sync(|_, job| {
            result.push((job.id, job.config.name.clone(), job.status()));
            true
        });
        result
    }

    /// Returns the number of tracked jobs.
    pub fn job_count(&self) -> usize {
        let mut count = 0;
        self.jobs.iter_sync(|_, _| {
            count += 1;
            true
        });
        count
    }

    fn get_job(&self, job_id: StreamJobId) -> Result<Arc<StreamJob>> {
        let result = self.jobs.read_sync(&job_id.0, |_, v| Arc::clone(v));
        result.ok_or_else(|| ZyronError::StreamingError(format!("job {} not found", job_id)))
    }
}

impl Default for StreamJobManager {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::stream_operator::OperatorChain;

    #[test]
    fn test_job_lifecycle() {
        let job = StreamJob::new(
            StreamJobId::new(1),
            StreamJobConfig::new("test-job".into()),
            OperatorChain::new(),
        );

        assert_eq!(job.status(), StreamJobStatus::Created);

        job.start(1000).expect("start should succeed");
        assert_eq!(job.status(), StreamJobStatus::Running);

        job.pause().expect("pause should succeed");
        assert_eq!(job.status(), StreamJobStatus::Paused);

        job.start(2000).expect("resume should succeed");
        assert_eq!(job.status(), StreamJobStatus::Running);

        job.cancel().expect("cancel should succeed");
        assert_eq!(job.status(), StreamJobStatus::Cancelled);
    }

    #[test]
    fn test_job_fail() {
        let job = StreamJob::new(
            StreamJobId::new(2),
            StreamJobConfig::new("failing-job".into()),
            OperatorChain::new(),
        );
        job.start(1000).expect("start should succeed");
        job.fail("out of memory".into());
        assert!(matches!(job.status(), StreamJobStatus::Failed { .. }));
    }

    #[test]
    fn test_job_manager_create_and_start() {
        let manager = StreamJobManager::new();
        let config = StreamJobConfig::new("test-job".into());
        let chain = OperatorChain::new();

        let job_id = manager
            .create_job(config, chain)
            .expect("create should succeed");

        let status = manager.status(job_id).expect("status should succeed");
        assert_eq!(status, StreamJobStatus::Created);

        manager.start(job_id, 1000).expect("start should succeed");
        let status = manager.status(job_id).expect("status should succeed");
        assert_eq!(status, StreamJobStatus::Running);
    }

    #[test]
    fn test_job_manager_list() {
        let manager = StreamJobManager::new();
        let config1 = StreamJobConfig::new("job-1".into());
        let config2 = StreamJobConfig::new("job-2".into());

        manager
            .create_job(config1, OperatorChain::new())
            .expect("create should succeed");
        manager
            .create_job(config2, OperatorChain::new())
            .expect("create should succeed");

        let jobs = manager.list();
        assert_eq!(jobs.len(), 2);
    }

    #[test]
    fn test_job_manager_stop() {
        let manager = StreamJobManager::new();
        let config = StreamJobConfig::new("stoppable-job".into());
        let job_id = manager
            .create_job(config, OperatorChain::new())
            .expect("create should succeed");

        manager.start(job_id, 1000).expect("start should succeed");
        manager.stop(job_id).expect("stop should succeed");

        let status = manager.status(job_id).expect("status should succeed");
        assert_eq!(status, StreamJobStatus::Cancelled);
    }

    #[test]
    fn test_job_not_found() {
        let manager = StreamJobManager::new();
        let result = manager.status(StreamJobId::new(999));
        assert!(result.is_err());
    }

    #[test]
    fn test_job_config() {
        let config = StreamJobConfig::new("configured-job".into())
            .with_guarantee(ProcessingGuarantee::ExactlyOnce)
            .with_parallelism(4);

        assert_eq!(
            config.processing_guarantee,
            ProcessingGuarantee::ExactlyOnce
        );
        assert_eq!(config.parallelism, 4);
    }

    #[test]
    fn test_stream_job_id_display() {
        let id = StreamJobId::new(42);
        assert_eq!(format!("{id}"), "job-42");
    }
}
