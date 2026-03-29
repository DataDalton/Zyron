//! Pipeline SLA monitoring.
//!
//! Tracks pipeline run durations and data staleness against configured
//! thresholds, recording breach events when limits are exceeded.

use crate::ids::PipelineId;
use zyron_common::Result;

/// SLA configuration for a pipeline, specifying maximum allowed
/// duration and data staleness.
#[derive(Clone, Debug)]
pub struct PipelineSlaConfig {
    /// Maximum allowed run duration in milliseconds. If a pipeline
    /// run exceeds this, it is recorded as a duration breach.
    pub maxDurationMs: Option<u64>,
    /// Maximum allowed staleness in milliseconds. Staleness measures
    /// the gap between the current time and the last successful run.
    /// If exceeded, it is recorded as a staleness breach.
    pub maxStalenessMs: Option<u64>,
}

/// A single SLA check result for a pipeline run.
#[derive(Clone, Debug)]
pub struct SlaRecord {
    pub pipelineId: PipelineId,
    /// Epoch timestamp (milliseconds) when the pipeline run started.
    pub runAt: i64,
    /// Actual run duration in milliseconds.
    pub durationMs: u64,
    /// True if this run breached the SLA.
    pub breached: bool,
    /// Human-readable reason for the breach, if any.
    pub breachReason: Option<String>,
}

/// Tracks SLA records per pipeline and provides breach detection.
pub struct SlaTracker {
    /// Maps pipeline ID (as u32) to its history of SLA records.
    history: scc::HashMap<u32, Vec<SlaRecord>>,
}

impl SlaTracker {
    /// Creates an empty SLA tracker.
    pub fn new() -> Self {
        Self {
            history: scc::HashMap::new(),
        }
    }

    /// Checks a pipeline run against its SLA config. Returns a
    /// SlaRecord with breached=true and a reason if any threshold
    /// was exceeded. Returns None if no SLA was breached.
    ///
    /// durationMs is the actual run duration. stalenessMs is the
    /// time since the last successful run (0 if not applicable).
    pub fn checkSla(
        &self,
        pipelineId: PipelineId,
        config: &PipelineSlaConfig,
        durationMs: u64,
        stalenessMs: u64,
    ) -> Option<SlaRecord> {
        let mut reasons = Vec::new();

        if let Some(maxDuration) = config.maxDurationMs {
            if durationMs > maxDuration {
                reasons.push(format!(
                    "duration {}ms exceeded limit {}ms",
                    durationMs, maxDuration
                ));
            }
        }

        if let Some(maxStaleness) = config.maxStalenessMs {
            if stalenessMs > maxStaleness {
                reasons.push(format!(
                    "staleness {}ms exceeded limit {}ms",
                    stalenessMs, maxStaleness
                ));
            }
        }

        if reasons.is_empty() {
            return None;
        }

        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_millis() as i64)
            .unwrap_or(0);

        Some(SlaRecord {
            pipelineId,
            runAt: now,
            durationMs,
            breached: true,
            breachReason: Some(reasons.join(", ")),
        })
    }

    /// Records an SLA check result for a pipeline.
    pub fn record(&self, pipelineId: PipelineId, record: SlaRecord) -> Result<()> {
        let key = pipelineId.0;
        let entry = self.history.entry_sync(key);
        match entry {
            scc::hash_map::Entry::Occupied(mut occ) => {
                occ.get_mut().push(record);
            }
            scc::hash_map::Entry::Vacant(vac) => {
                vac.insert_entry(vec![record]);
            }
        }
        Ok(())
    }

    /// Returns the SLA check history for a pipeline, ordered by
    /// insertion time. Returns an empty vector if no records exist.
    pub fn getHistory(&self, pipelineId: PipelineId) -> Vec<SlaRecord> {
        let key = pipelineId.0;
        self.history
            .read_sync(&key, |_k, v| v.clone())
            .unwrap_or_default()
    }

    /// Returns the count of breached records for a pipeline.
    pub fn breachCount(&self, pipelineId: PipelineId) -> usize {
        let key = pipelineId.0;
        self.history
            .read_sync(&key, |_k, v| v.iter().filter(|r| r.breached).count())
            .unwrap_or(0)
    }

    /// Returns the most recent SLA record for a pipeline, if any.
    pub fn lastRecord(&self, pipelineId: PipelineId) -> Option<SlaRecord> {
        let key = pipelineId.0;
        self.history
            .read_sync(&key, |_k, v| v.last().cloned())
            .flatten()
    }

    /// Clears all history for a pipeline.
    pub fn clearHistory(&self, pipelineId: PipelineId) -> Result<()> {
        let key = pipelineId.0;
        let _ = self.history.remove_sync(&key);
        Ok(())
    }

    /// Returns the total number of pipelines with recorded SLA history.
    pub fn trackedPipelineCount(&self) -> usize {
        let mut count = 0;
        self.history.iter_sync(|_k, _v| {
            count += 1;
            true
        });
        count
    }
}

impl Default for SlaTracker {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn makeConfig(maxDurationMs: Option<u64>, maxStalenessMs: Option<u64>) -> PipelineSlaConfig {
        PipelineSlaConfig {
            maxDurationMs,
            maxStalenessMs,
        }
    }

    fn makeRecord(pipelineId: PipelineId, durationMs: u64, breached: bool) -> SlaRecord {
        SlaRecord {
            pipelineId,
            runAt: 1000,
            durationMs,
            breached,
            breachReason: if breached {
                Some("test breach".to_string())
            } else {
                None
            },
        }
    }

    #[test]
    fn test_check_sla_no_breach() {
        let tracker = SlaTracker::new();
        let config = makeConfig(Some(5000), Some(10000));
        let result = tracker.checkSla(PipelineId(1), &config, 3000, 5000);
        assert!(result.is_none());
    }

    #[test]
    fn test_check_sla_duration_breach() {
        let tracker = SlaTracker::new();
        let config = makeConfig(Some(5000), None);
        let result = tracker.checkSla(PipelineId(1), &config, 6000, 0);
        assert!(result.is_some());
        let record = result.expect("should be breach");
        assert!(record.breached);
        assert!(
            record
                .breachReason
                .as_ref()
                .expect("reason")
                .contains("duration")
        );
    }

    #[test]
    fn test_check_sla_staleness_breach() {
        let tracker = SlaTracker::new();
        let config = makeConfig(None, Some(10000));
        let result = tracker.checkSla(PipelineId(1), &config, 1000, 15000);
        assert!(result.is_some());
        let record = result.expect("should be breach");
        assert!(record.breached);
        assert!(
            record
                .breachReason
                .as_ref()
                .expect("reason")
                .contains("staleness")
        );
    }

    #[test]
    fn test_check_sla_both_breached() {
        let tracker = SlaTracker::new();
        let config = makeConfig(Some(5000), Some(10000));
        let result = tracker.checkSla(PipelineId(1), &config, 6000, 15000);
        assert!(result.is_some());
        let record = result.expect("should be breach");
        let reason = record.breachReason.expect("reason");
        assert!(reason.contains("duration"));
        assert!(reason.contains("staleness"));
    }

    #[test]
    fn test_check_sla_no_limits() {
        let tracker = SlaTracker::new();
        let config = makeConfig(None, None);
        let result = tracker.checkSla(PipelineId(1), &config, 999999, 999999);
        assert!(result.is_none());
    }

    #[test]
    fn test_record_and_get_history() {
        let tracker = SlaTracker::new();
        let pid = PipelineId(1);
        tracker
            .record(pid, makeRecord(pid, 3000, false))
            .expect("record");
        tracker
            .record(pid, makeRecord(pid, 6000, true))
            .expect("record");

        let history = tracker.getHistory(pid);
        assert_eq!(history.len(), 2);
        assert!(!history[0].breached);
        assert!(history[1].breached);
    }

    #[test]
    fn test_get_history_empty() {
        let tracker = SlaTracker::new();
        let history = tracker.getHistory(PipelineId(999));
        assert!(history.is_empty());
    }

    #[test]
    fn test_breach_count() {
        let tracker = SlaTracker::new();
        let pid = PipelineId(1);
        tracker
            .record(pid, makeRecord(pid, 1000, false))
            .expect("record");
        tracker
            .record(pid, makeRecord(pid, 2000, true))
            .expect("record");
        tracker
            .record(pid, makeRecord(pid, 3000, true))
            .expect("record");

        assert_eq!(tracker.breachCount(pid), 2);
    }

    #[test]
    fn test_last_record() {
        let tracker = SlaTracker::new();
        let pid = PipelineId(1);
        assert!(tracker.lastRecord(pid).is_none());

        tracker
            .record(pid, makeRecord(pid, 1000, false))
            .expect("record");
        tracker
            .record(pid, makeRecord(pid, 5000, true))
            .expect("record");

        let last = tracker.lastRecord(pid).expect("should have record");
        assert_eq!(last.durationMs, 5000);
        assert!(last.breached);
    }

    #[test]
    fn test_clear_history() {
        let tracker = SlaTracker::new();
        let pid = PipelineId(1);
        tracker
            .record(pid, makeRecord(pid, 1000, false))
            .expect("record");
        tracker.clearHistory(pid).expect("clear");

        assert!(tracker.getHistory(pid).is_empty());
        assert_eq!(tracker.trackedPipelineCount(), 0);
    }

    #[test]
    fn test_tracked_pipeline_count() {
        let tracker = SlaTracker::new();
        assert_eq!(tracker.trackedPipelineCount(), 0);

        tracker
            .record(PipelineId(1), makeRecord(PipelineId(1), 1000, false))
            .expect("record");
        tracker
            .record(PipelineId(2), makeRecord(PipelineId(2), 2000, false))
            .expect("record");

        assert_eq!(tracker.trackedPipelineCount(), 2);
    }

    #[test]
    fn test_exact_threshold_no_breach() {
        let tracker = SlaTracker::new();
        let config = makeConfig(Some(5000), Some(10000));
        // Exactly at the threshold should not breach.
        let result = tracker.checkSla(PipelineId(1), &config, 5000, 10000);
        assert!(result.is_none());
    }

    #[test]
    fn test_multiple_pipelines_independent() {
        let tracker = SlaTracker::new();
        let pid1 = PipelineId(1);
        let pid2 = PipelineId(2);

        tracker
            .record(pid1, makeRecord(pid1, 1000, false))
            .expect("record");
        tracker
            .record(pid2, makeRecord(pid2, 2000, true))
            .expect("record");

        assert_eq!(tracker.getHistory(pid1).len(), 1);
        assert_eq!(tracker.getHistory(pid2).len(), 1);
        assert_eq!(tracker.breachCount(pid1), 0);
        assert_eq!(tracker.breachCount(pid2), 1);
    }
}
