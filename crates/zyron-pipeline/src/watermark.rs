//! Watermark tracking for incremental pipeline processing.

use crate::ids::PipelineId;
use zyron_common::Result;

/// Strategy for detecting incremental changes in source data.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IncrementalStrategy {
    Timestamp,
    Version,
    Sequence,
}

/// A watermark records the last-processed position for an incremental pipeline stage.
#[derive(Debug, Clone)]
pub struct Watermark {
    pub pipeline_id: PipelineId,
    pub stage_name: String,
    pub column_name: String,
    pub current_value: Vec<u8>,
    pub strategy: IncrementalStrategy,
    pub updated_at: i64,
}

/// Tracks watermarks per pipeline stage for incremental refresh.
/// Uses (pipeline_id, stage_name) tuple key to avoid format string allocation.
pub struct WatermarkTracker {
    watermarks: scc::HashMap<(u32, String), Watermark>,
}

impl WatermarkTracker {
    pub fn new() -> Self {
        Self {
            watermarks: scc::HashMap::new(),
        }
    }

    /// Get the current watermark for a pipeline stage.
    pub fn get_watermark(&self, pipeline_id: PipelineId, stage: &str) -> Option<Watermark> {
        let key = (pipeline_id.0, stage.to_string());
        self.watermarks.read_sync(&key, |_k, v| v.clone())
    }

    /// Advance the watermark for a pipeline stage to a new value.
    pub fn advance_watermark(
        &self,
        pipeline_id: PipelineId,
        stage: &str,
        new_value: Vec<u8>,
    ) -> Result<()> {
        let key = (pipeline_id.0, stage.to_string());
        self.watermarks
            .entry_sync(key)
            .and_modify(|w| {
                w.current_value = new_value.clone();
                w.updated_at = current_timestamp();
            })
            .or_insert_with(|| Watermark {
                pipeline_id,
                stage_name: stage.to_string(),
                column_name: String::new(),
                current_value: new_value,
                strategy: IncrementalStrategy::Timestamp,
                updated_at: current_timestamp(),
            });
        Ok(())
    }

    /// Reset the watermark. If value is None, removes the watermark (forces full refresh).
    /// If value is Some, sets to that value (for backfill from a specific point).
    pub fn reset_watermark(
        &self,
        pipeline_id: PipelineId,
        stage: &str,
        value: Option<Vec<u8>>,
    ) -> Result<()> {
        let key = (pipeline_id.0, stage.to_string());
        match value {
            None => {
                let _ = self.watermarks.remove_sync(&key);
                Ok(())
            }
            Some(v) => self.advance_watermark(pipeline_id, stage, v),
        }
    }

    /// Set the column name and strategy for a watermark.
    pub fn configure_watermark(
        &self,
        pipeline_id: PipelineId,
        stage: &str,
        column_name: &str,
        strategy: IncrementalStrategy,
    ) -> Result<()> {
        let key = (pipeline_id.0, stage.to_string());
        self.watermarks
            .entry_sync(key)
            .and_modify(|w| {
                w.column_name = column_name.to_string();
                w.strategy = strategy;
            })
            .or_insert_with(|| Watermark {
                pipeline_id,
                stage_name: stage.to_string(),
                column_name: column_name.to_string(),
                current_value: Vec::new(),
                strategy,
                updated_at: current_timestamp(),
            });
        Ok(())
    }

    /// Return the number of tracked watermarks.
    pub fn watermark_count(&self) -> usize {
        self.watermarks.len()
    }
}

fn current_timestamp() -> i64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs() as i64)
        .unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_watermark_advance_and_get() {
        let tracker = WatermarkTracker::new();
        let pid = PipelineId(1);

        assert!(tracker.get_watermark(pid, "stage1").is_none());

        tracker
            .advance_watermark(pid, "stage1", vec![10, 20])
            .expect("advance");
        let wm = tracker.get_watermark(pid, "stage1").expect("should exist");
        assert_eq!(wm.current_value, vec![10, 20]);
        assert_eq!(wm.pipeline_id, pid);
    }

    #[test]
    fn test_watermark_reset_to_none() {
        let tracker = WatermarkTracker::new();
        let pid = PipelineId(2);

        tracker
            .advance_watermark(pid, "s1", vec![1])
            .expect("advance");
        assert_eq!(tracker.watermark_count(), 1);

        tracker.reset_watermark(pid, "s1", None).expect("reset");
        assert!(tracker.get_watermark(pid, "s1").is_none());
        assert_eq!(tracker.watermark_count(), 0);
    }

    #[test]
    fn test_watermark_reset_to_value() {
        let tracker = WatermarkTracker::new();
        let pid = PipelineId(3);

        tracker
            .advance_watermark(pid, "s1", vec![100])
            .expect("advance");
        tracker
            .reset_watermark(pid, "s1", Some(vec![50]))
            .expect("backfill");

        let wm = tracker.get_watermark(pid, "s1").expect("should exist");
        assert_eq!(wm.current_value, vec![50]);
    }

    #[test]
    fn test_watermark_configure() {
        let tracker = WatermarkTracker::new();
        let pid = PipelineId(4);

        tracker
            .configure_watermark(pid, "bronze", "updated_at", IncrementalStrategy::Timestamp)
            .expect("configure");
        let wm = tracker.get_watermark(pid, "bronze").expect("should exist");
        assert_eq!(wm.column_name, "updated_at");
        assert_eq!(wm.strategy, IncrementalStrategy::Timestamp);
    }

    #[test]
    fn test_multiple_pipelines() {
        let tracker = WatermarkTracker::new();
        tracker
            .advance_watermark(PipelineId(1), "s1", vec![1])
            .expect("p1");
        tracker
            .advance_watermark(PipelineId(2), "s1", vec![2])
            .expect("p2");
        tracker
            .advance_watermark(PipelineId(1), "s2", vec![3])
            .expect("p1s2");

        assert_eq!(tracker.watermark_count(), 3);

        let wm = tracker.get_watermark(PipelineId(2), "s1").expect("p2s1");
        assert_eq!(wm.current_value, vec![2]);
    }
}
