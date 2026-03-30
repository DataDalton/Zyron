//! Barrier-based checkpointing for exactly-once stream processing.
//!
//! Implements the Chandy-Lamport barrier protocol: the CheckpointCoordinator
//! injects CheckpointBarrier messages into source SPSC channels. Each
//! operator snapshots its state on barrier arrival and acks via atomic
//! counter. When all acks are received, the checkpoint is complete.
//! BarrierAligner handles multi-input operators by buffering records
//! from channels where the barrier has already arrived.

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};

use zyron_common::{Result, ZyronError};

use crate::spsc::SpscSender;
use crate::state::StateSnapshot;

// ---------------------------------------------------------------------------
// CheckpointId
// ---------------------------------------------------------------------------

/// Monotonically increasing checkpoint identifier.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct CheckpointId(pub u64);

impl CheckpointId {
    pub fn new(id: u64) -> Self {
        Self(id)
    }

    pub fn as_u64(&self) -> u64 {
        self.0
    }
}

impl std::fmt::Display for CheckpointId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "ckpt-{}", self.0)
    }
}

// ---------------------------------------------------------------------------
// CheckpointBarrier
// ---------------------------------------------------------------------------

/// Barrier message injected into the stream to trigger a checkpoint.
/// Operators snapshot their state when they receive this barrier.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CheckpointBarrier {
    pub checkpoint_id: CheckpointId,
    pub timestamp_ms: i64,
}

impl CheckpointBarrier {
    pub fn new(checkpoint_id: CheckpointId, timestamp_ms: i64) -> Self {
        Self {
            checkpoint_id,
            timestamp_ms,
        }
    }
}

// ---------------------------------------------------------------------------
// StreamCheckpoint
// ---------------------------------------------------------------------------

/// A completed checkpoint containing all operator state snapshots
/// and source offsets needed for exactly-once recovery.
#[derive(Debug, Clone)]
pub struct StreamCheckpoint {
    pub checkpoint_id: CheckpointId,
    pub timestamp_ms: i64,
    /// Per-operator state snapshots. (operator_id, snapshot).
    pub operator_states: Vec<(u32, StateSnapshot)>,
    /// Per-source offsets for replay. (source_id, serialized_offset).
    pub source_offsets: Vec<(u32, Vec<u8>)>,
    /// Global watermark at the time of the checkpoint.
    pub watermark_at_checkpoint: i64,
}

impl StreamCheckpoint {
    pub fn new(checkpoint_id: CheckpointId, timestamp_ms: i64) -> Self {
        Self {
            checkpoint_id,
            timestamp_ms,
            operator_states: Vec::new(),
            source_offsets: Vec::new(),
            watermark_at_checkpoint: i64::MIN,
        }
    }

    /// Total size in bytes of all serialized state in this checkpoint.
    pub fn size_bytes(&self) -> usize {
        let state_size: usize = self
            .operator_states
            .iter()
            .map(|(_, s)| s.size_bytes())
            .sum();
        let offset_size: usize = self.source_offsets.iter().map(|(_, o)| o.len()).sum();
        state_size + offset_size
    }
}

// ---------------------------------------------------------------------------
// CheckpointConfig
// ---------------------------------------------------------------------------

/// Configuration for the checkpoint coordinator.
#[derive(Debug, Clone)]
pub struct CheckpointConfig {
    /// Interval between checkpoints in milliseconds.
    pub interval_ms: u64,
    /// Timeout for a single checkpoint in milliseconds.
    pub timeout_ms: u64,
    /// Minimum pause between checkpoints in milliseconds.
    pub min_pause_ms: u64,
    /// Maximum number of concurrent in-flight checkpoints.
    pub max_concurrent: u32,
}

impl CheckpointConfig {
    pub fn new(interval_ms: u64) -> Self {
        Self {
            interval_ms,
            timeout_ms: 60_000,
            min_pause_ms: 0,
            max_concurrent: 1,
        }
    }
}

impl Default for CheckpointConfig {
    fn default() -> Self {
        Self::new(30_000) // 30 seconds default interval.
    }
}

// ---------------------------------------------------------------------------
// CheckpointCoordinator
// ---------------------------------------------------------------------------

/// Coordinates checkpoint barriers across the streaming DAG.
/// Injects barriers into source SPSC channels and tracks ack completion
/// via atomic counters, requiring zero mutex on the ack path.
pub struct CheckpointCoordinator {
    /// Next checkpoint ID, allocated via fetch_add.
    next_id: AtomicU64,
    /// SPSC senders to inject barriers into source operators.
    barrier_senders: Vec<SpscSender<CheckpointBarrier>>,
    /// Per-checkpoint ack counters. Index = checkpoint_id % max_concurrent.
    ack_counters: Vec<AtomicU32>,
    /// Total number of operators that must ack each checkpoint.
    total_operators: u32,
    /// Checkpoint configuration.
    config: CheckpointConfig,
    /// Completed checkpoints stored for reference.
    completed: parking_lot::Mutex<Vec<StreamCheckpoint>>,
    /// Timestamp of the last completed checkpoint.
    last_completed_ms: AtomicU64,
}

impl CheckpointCoordinator {
    /// Creates a coordinator with barrier injection channels.
    pub fn new(
        barrier_senders: Vec<SpscSender<CheckpointBarrier>>,
        total_operators: u32,
        config: CheckpointConfig,
    ) -> Self {
        let max_concurrent = config.max_concurrent.max(1) as usize;
        let mut ack_counters = Vec::with_capacity(max_concurrent);
        for _ in 0..max_concurrent {
            ack_counters.push(AtomicU32::new(0));
        }
        Self {
            next_id: AtomicU64::new(1),
            barrier_senders,
            ack_counters,
            total_operators,
            config,
            completed: parking_lot::Mutex::new(Vec::new()),
            last_completed_ms: AtomicU64::new(0),
        }
    }

    /// Triggers a new checkpoint by injecting barriers into all sources.
    /// Returns the checkpoint ID.
    pub fn trigger_checkpoint(&self, timestamp_ms: i64) -> Result<CheckpointId> {
        // Check minimum pause.
        let last = self.last_completed_ms.load(Ordering::Relaxed);
        if timestamp_ms > 0 && last > 0 {
            let elapsed = (timestamp_ms as u64).saturating_sub(last);
            if elapsed < self.config.min_pause_ms {
                return Err(ZyronError::CheckpointError(
                    "minimum pause between checkpoints not elapsed".into(),
                ));
            }
        }

        let id = CheckpointId::new(self.next_id.fetch_add(1, Ordering::Relaxed));
        let barrier = CheckpointBarrier::new(id, timestamp_ms);

        // Reset ack counter for this checkpoint.
        let slot = (id.0 as usize) % self.ack_counters.len();
        self.ack_counters[slot].store(0, Ordering::Relaxed);

        // Inject barrier into each source.
        for sender in &self.barrier_senders {
            sender.send(barrier);
        }

        Ok(id)
    }

    /// Called by an operator after it has snapshotted its state.
    /// Returns true if this was the final ack (checkpoint complete).
    pub fn acknowledge(&self, checkpoint_id: CheckpointId) -> bool {
        let slot = (checkpoint_id.0 as usize) % self.ack_counters.len();
        let prev = self.ack_counters[slot].fetch_add(1, Ordering::AcqRel);
        prev + 1 >= self.total_operators
    }

    /// Registers a completed checkpoint.
    pub fn complete_checkpoint(&self, checkpoint: StreamCheckpoint) {
        self.last_completed_ms
            .store(checkpoint.timestamp_ms as u64, Ordering::Relaxed);
        let mut completed = self.completed.lock();
        completed.push(checkpoint);
    }

    /// Returns the list of completed checkpoint IDs.
    pub fn completed_checkpoints(&self) -> Vec<CheckpointId> {
        let completed = self.completed.lock();
        completed.iter().map(|c| c.checkpoint_id).collect()
    }

    /// Returns the total number of operators.
    pub fn total_operators(&self) -> u32 {
        self.total_operators
    }

    /// Returns the current ack count for a checkpoint.
    pub fn ack_count(&self, checkpoint_id: CheckpointId) -> u32 {
        let slot = (checkpoint_id.0 as usize) % self.ack_counters.len();
        self.ack_counters[slot].load(Ordering::Relaxed)
    }

    /// Returns the checkpoint configuration.
    pub fn config(&self) -> &CheckpointConfig {
        &self.config
    }
}

// ---------------------------------------------------------------------------
// BarrierAligner
// ---------------------------------------------------------------------------

/// Handles barrier alignment for multi-input operators.
/// For operators with multiple input channels (e.g., joins), the aligner
/// buffers records from channels where the barrier has arrived until all
/// channels have delivered the barrier.
pub struct BarrierAligner {
    /// Number of input channels.
    input_count: usize,
    /// Per-channel flag: true if the barrier has arrived on this channel.
    barrier_received: Vec<bool>,
    /// Number of channels that have received the barrier.
    received_count: usize,
    /// Current checkpoint being aligned (None if not in alignment).
    current_checkpoint: Option<CheckpointId>,
}

impl BarrierAligner {
    pub fn new(input_count: usize) -> Self {
        Self {
            input_count,
            barrier_received: vec![false; input_count],
            received_count: 0,
            current_checkpoint: None,
        }
    }

    /// Called when a barrier arrives on a specific input channel.
    /// Returns true if all channels have now received the barrier
    /// (alignment complete, operator should snapshot).
    pub fn on_barrier(&mut self, input_idx: usize, checkpoint_id: CheckpointId) -> bool {
        if input_idx >= self.input_count {
            return false;
        }

        // If this is a new checkpoint, reset state.
        if self.current_checkpoint != Some(checkpoint_id) {
            self.reset();
            self.current_checkpoint = Some(checkpoint_id);
        }

        if !self.barrier_received[input_idx] {
            self.barrier_received[input_idx] = true;
            self.received_count += 1;
        }

        self.received_count >= self.input_count
    }

    /// Returns true if records from the given channel should be buffered
    /// (barrier already received on this channel, waiting for other channels).
    #[inline]
    pub fn should_buffer(&self, input_idx: usize) -> bool {
        if input_idx >= self.input_count {
            return false;
        }
        self.barrier_received[input_idx] && self.received_count < self.input_count
    }

    /// Returns true if alignment is in progress.
    pub fn is_aligning(&self) -> bool {
        self.received_count > 0 && self.received_count < self.input_count
    }

    /// Resets the aligner for a new checkpoint.
    pub fn reset(&mut self) {
        for flag in &mut self.barrier_received {
            *flag = false;
        }
        self.received_count = 0;
        self.current_checkpoint = None;
    }
}

// ---------------------------------------------------------------------------
// CheckpointStorage trait
// ---------------------------------------------------------------------------

/// Trait for persisting checkpoint data.
pub trait CheckpointStorage: Send + Sync {
    /// Saves a checkpoint.
    fn save(&self, checkpoint: &StreamCheckpoint) -> Result<()>;

    /// Loads a checkpoint by ID.
    fn load(&self, checkpoint_id: CheckpointId) -> Result<StreamCheckpoint>;

    /// Lists all stored checkpoint IDs.
    fn list(&self) -> Result<Vec<CheckpointId>>;

    /// Deletes a checkpoint.
    fn delete(&self, checkpoint_id: CheckpointId) -> Result<()>;
}

// ---------------------------------------------------------------------------
// LocalCheckpointStorage
// ---------------------------------------------------------------------------

/// Checkpoint storage backed by the local filesystem.
/// Each checkpoint is stored as a directory containing serialized state files.
pub struct LocalCheckpointStorage {
    base_dir: PathBuf,
}

impl LocalCheckpointStorage {
    pub fn new(base_dir: &Path) -> Result<Self> {
        std::fs::create_dir_all(base_dir).map_err(|e| {
            ZyronError::CheckpointError(format!("failed to create checkpoint dir: {e}"))
        })?;
        Ok(Self {
            base_dir: base_dir.to_path_buf(),
        })
    }

    fn checkpoint_path(&self, checkpoint_id: CheckpointId) -> PathBuf {
        self.base_dir.join(format!("ckpt_{}", checkpoint_id.0))
    }
}

impl CheckpointStorage for LocalCheckpointStorage {
    fn save(&self, checkpoint: &StreamCheckpoint) -> Result<()> {
        let path = self.checkpoint_path(checkpoint.checkpoint_id);
        std::fs::create_dir_all(&path).map_err(|e| {
            ZyronError::CheckpointError(format!("failed to create checkpoint dir: {e}"))
        })?;

        // Serialize metadata.
        let mut meta = Vec::new();
        meta.extend_from_slice(&checkpoint.checkpoint_id.0.to_le_bytes());
        meta.extend_from_slice(&checkpoint.timestamp_ms.to_le_bytes());
        meta.extend_from_slice(&checkpoint.watermark_at_checkpoint.to_le_bytes());
        meta.extend_from_slice(&(checkpoint.operator_states.len() as u32).to_le_bytes());
        meta.extend_from_slice(&(checkpoint.source_offsets.len() as u32).to_le_bytes());

        std::fs::write(path.join("meta"), &meta).map_err(|e| {
            ZyronError::CheckpointError(format!("failed to write checkpoint metadata: {e}"))
        })?;

        // Serialize operator states.
        for (op_id, snapshot) in &checkpoint.operator_states {
            let mut state_data = Vec::new();
            state_data.extend_from_slice(&snapshot.snapshot_id.to_le_bytes());
            state_data.extend_from_slice(&(snapshot.data.len() as u32).to_le_bytes());
            for (ns, key, val) in &snapshot.data {
                state_data.extend_from_slice(&(ns.len() as u32).to_le_bytes());
                state_data.extend_from_slice(ns);
                state_data.extend_from_slice(&(key.len() as u32).to_le_bytes());
                state_data.extend_from_slice(key);
                state_data.extend_from_slice(&(val.len() as u32).to_le_bytes());
                state_data.extend_from_slice(val);
            }
            std::fs::write(path.join(format!("op_{op_id}")), &state_data).map_err(|e| {
                ZyronError::CheckpointError(format!("failed to write operator state: {e}"))
            })?;
        }

        // Serialize source offsets.
        for (src_id, offset) in &checkpoint.source_offsets {
            std::fs::write(path.join(format!("src_{src_id}")), offset).map_err(|e| {
                ZyronError::CheckpointError(format!("failed to write source offset: {e}"))
            })?;
        }

        Ok(())
    }

    fn load(&self, checkpoint_id: CheckpointId) -> Result<StreamCheckpoint> {
        let path = self.checkpoint_path(checkpoint_id);
        if !path.exists() {
            return Err(ZyronError::CheckpointError(format!(
                "checkpoint {checkpoint_id} not found"
            )));
        }

        let meta = std::fs::read(path.join("meta")).map_err(|e| {
            ZyronError::CheckpointError(format!("failed to read checkpoint metadata: {e}"))
        })?;

        if meta.len() < 28 {
            return Err(ZyronError::CheckpointError(
                "corrupted checkpoint metadata".into(),
            ));
        }

        let id = u64::from_le_bytes([
            meta[0], meta[1], meta[2], meta[3], meta[4], meta[5], meta[6], meta[7],
        ]);
        let timestamp_ms = i64::from_le_bytes([
            meta[8], meta[9], meta[10], meta[11], meta[12], meta[13], meta[14], meta[15],
        ]);
        let watermark = i64::from_le_bytes([
            meta[16], meta[17], meta[18], meta[19], meta[20], meta[21], meta[22], meta[23],
        ]);
        let num_operators = u32::from_le_bytes([meta[24], meta[25], meta[26], meta[27]]);

        let mut checkpoint = StreamCheckpoint::new(CheckpointId::new(id), timestamp_ms);
        checkpoint.watermark_at_checkpoint = watermark;

        // Load operator states.
        for op_id in 0..num_operators {
            let op_path = path.join(format!("op_{op_id}"));
            if op_path.exists() {
                let state_data = std::fs::read(&op_path).map_err(|e| {
                    ZyronError::CheckpointError(format!("failed to read operator state: {e}"))
                })?;
                if state_data.len() >= 12 {
                    let snapshot_id = u64::from_le_bytes([
                        state_data[0],
                        state_data[1],
                        state_data[2],
                        state_data[3],
                        state_data[4],
                        state_data[5],
                        state_data[6],
                        state_data[7],
                    ]);
                    let entry_count = u32::from_le_bytes([
                        state_data[8],
                        state_data[9],
                        state_data[10],
                        state_data[11],
                    ]) as usize;

                    let mut data = Vec::with_capacity(entry_count);
                    let mut pos = 12;
                    for _ in 0..entry_count {
                        if pos + 4 > state_data.len() {
                            return Err(ZyronError::CheckpointError(
                                "truncated checkpoint: missing namespace length".into(),
                            ));
                        }
                        let ns_len = u32::from_le_bytes([
                            state_data[pos],
                            state_data[pos + 1],
                            state_data[pos + 2],
                            state_data[pos + 3],
                        ]) as usize;
                        pos += 4;
                        if pos + ns_len > state_data.len() {
                            return Err(ZyronError::CheckpointError(
                                "corrupt checkpoint: namespace length exceeds file size".into(),
                            ));
                        }
                        let ns = state_data[pos..pos + ns_len].to_vec();
                        pos += ns_len;

                        if pos + 4 > state_data.len() {
                            return Err(ZyronError::CheckpointError(
                                "truncated checkpoint: missing key length".into(),
                            ));
                        }
                        let key_len = u32::from_le_bytes([
                            state_data[pos],
                            state_data[pos + 1],
                            state_data[pos + 2],
                            state_data[pos + 3],
                        ]) as usize;
                        pos += 4;
                        if pos + key_len > state_data.len() {
                            return Err(ZyronError::CheckpointError(
                                "corrupt checkpoint: key length exceeds file size".into(),
                            ));
                        }
                        let key = state_data[pos..pos + key_len].to_vec();
                        pos += key_len;

                        if pos + 4 > state_data.len() {
                            return Err(ZyronError::CheckpointError(
                                "truncated checkpoint: missing value length".into(),
                            ));
                        }
                        let val_len = u32::from_le_bytes([
                            state_data[pos],
                            state_data[pos + 1],
                            state_data[pos + 2],
                            state_data[pos + 3],
                        ]) as usize;
                        pos += 4;
                        if pos + val_len > state_data.len() {
                            return Err(ZyronError::CheckpointError(
                                "corrupt checkpoint: value length exceeds file size".into(),
                            ));
                        }
                        let val = state_data[pos..pos + val_len].to_vec();
                        pos += val_len;

                        data.push((ns, key, val));
                    }
                    checkpoint
                        .operator_states
                        .push((op_id, StateSnapshot { data, snapshot_id }));
                }
            }
        }

        Ok(checkpoint)
    }

    fn list(&self) -> Result<Vec<CheckpointId>> {
        let mut ids = Vec::new();
        let entries = std::fs::read_dir(&self.base_dir)
            .map_err(|e| ZyronError::CheckpointError(format!("failed to list checkpoints: {e}")))?;

        for entry in entries {
            let entry = entry.map_err(|e| {
                ZyronError::CheckpointError(format!("failed to read dir entry: {e}"))
            })?;
            let name = entry.file_name();
            let name_str = name.to_string_lossy();
            if let Some(id_str) = name_str.strip_prefix("ckpt_") {
                if let Ok(id) = id_str.parse::<u64>() {
                    ids.push(CheckpointId::new(id));
                }
            }
        }

        ids.sort();
        Ok(ids)
    }

    fn delete(&self, checkpoint_id: CheckpointId) -> Result<()> {
        let path = self.checkpoint_path(checkpoint_id);
        if path.exists() {
            std::fs::remove_dir_all(&path).map_err(|e| {
                ZyronError::CheckpointError(format!("failed to delete checkpoint: {e}"))
            })?;
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::spsc::spsc_channel;

    #[test]
    fn test_checkpoint_id_ordering() {
        let a = CheckpointId::new(1);
        let b = CheckpointId::new(2);
        assert!(a < b);
        assert_eq!(a, CheckpointId::new(1));
    }

    #[test]
    fn test_checkpoint_barrier() {
        let barrier = CheckpointBarrier::new(CheckpointId::new(5), 12345);
        assert_eq!(barrier.checkpoint_id.as_u64(), 5);
        assert_eq!(barrier.timestamp_ms, 12345);
    }

    #[test]
    fn test_checkpoint_coordinator_trigger_and_ack() {
        let (tx1, _rx1) = spsc_channel::<CheckpointBarrier>(8);
        let (tx2, _rx2) = spsc_channel::<CheckpointBarrier>(8);
        let config = CheckpointConfig::new(10_000);
        let coordinator = CheckpointCoordinator::new(vec![tx1, tx2], 3, config);

        let id = coordinator
            .trigger_checkpoint(1000)
            .expect("trigger should succeed");
        assert_eq!(id.as_u64(), 1);

        // First two acks should not complete.
        assert!(!coordinator.acknowledge(id));
        assert!(!coordinator.acknowledge(id));
        // Third ack should complete (total_operators = 3).
        assert!(coordinator.acknowledge(id));
    }

    #[test]
    fn test_barrier_aligner_two_inputs() {
        let mut aligner = BarrierAligner::new(2);
        let ckpt = CheckpointId::new(1);

        // Barrier on input 0 should not complete.
        assert!(!aligner.on_barrier(0, ckpt));
        assert!(aligner.is_aligning());
        // Input 0 should buffer, input 1 should not.
        assert!(aligner.should_buffer(0));
        assert!(!aligner.should_buffer(1));

        // Barrier on input 1 should complete alignment.
        assert!(aligner.on_barrier(1, ckpt));
        assert!(!aligner.is_aligning());
    }

    #[test]
    fn test_barrier_aligner_single_input() {
        let mut aligner = BarrierAligner::new(1);
        let ckpt = CheckpointId::new(1);
        // Single input should complete immediately.
        assert!(aligner.on_barrier(0, ckpt));
    }

    #[test]
    fn test_stream_checkpoint_size() {
        let mut checkpoint = StreamCheckpoint::new(CheckpointId::new(1), 1000);
        checkpoint.operator_states.push((
            0,
            StateSnapshot {
                data: vec![(b"ns".to_vec(), b"k".to_vec(), b"v".to_vec())],
                snapshot_id: 1,
            },
        ));
        checkpoint.source_offsets.push((0, vec![1, 2, 3, 4]));
        assert_eq!(checkpoint.size_bytes(), 4 + 4); // state(2+1+1) + offset(4)
    }

    #[test]
    fn test_local_checkpoint_storage_save_load() {
        let dir = tempfile::tempdir().expect("failed to create temp dir");
        let storage = LocalCheckpointStorage::new(dir.path()).expect("failed to create storage");

        let mut ckpt = StreamCheckpoint::new(CheckpointId::new(1), 5000);
        ckpt.watermark_at_checkpoint = 4000;
        ckpt.operator_states.push((
            0,
            StateSnapshot {
                data: vec![(b"ns".to_vec(), b"key1".to_vec(), b"val1".to_vec())],
                snapshot_id: 1,
            },
        ));
        ckpt.source_offsets.push((0, vec![0, 0, 0, 8]));

        storage.save(&ckpt).expect("save should succeed");

        let ids = storage.list().expect("list should succeed");
        assert_eq!(ids.len(), 1);
        assert_eq!(ids[0], CheckpointId::new(1));

        let loaded = storage
            .load(CheckpointId::new(1))
            .expect("load should succeed");
        assert_eq!(loaded.checkpoint_id, CheckpointId::new(1));
        assert_eq!(loaded.timestamp_ms, 5000);
        assert_eq!(loaded.watermark_at_checkpoint, 4000);
        assert_eq!(loaded.operator_states.len(), 1);

        storage
            .delete(CheckpointId::new(1))
            .expect("delete should succeed");
        let ids = storage.list().expect("list should succeed");
        assert!(ids.is_empty());
    }

    #[test]
    fn test_checkpoint_config_default() {
        let config = CheckpointConfig::default();
        assert_eq!(config.interval_ms, 30_000);
        assert_eq!(config.timeout_ms, 60_000);
        assert_eq!(config.max_concurrent, 1);
    }
}
