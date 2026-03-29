//! Statistical drift detection using Population Stability Index (PSI).

use zyron_common::Result;

/// Severity of detected distribution drift.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DriftSeverity {
    None,
    Warning,
    Critical,
}

/// Result of a drift detection analysis.
#[derive(Debug, Clone)]
pub struct DriftResult {
    pub column: String,
    pub psi_score: f64,
    pub severity: DriftSeverity,
    pub baseline_bucket_count: usize,
    pub current_bucket_count: usize,
}

/// A histogram snapshot for a column at a point in time.
#[derive(Debug, Clone)]
pub struct HistogramSnapshot {
    pub column: String,
    pub bucket_boundaries: Vec<f64>,
    pub bucket_counts: Vec<u64>,
    pub total_count: u64,
    pub recorded_at: i64,
}

impl HistogramSnapshot {
    /// Convert bucket counts to proportions (each bucket / total).
    pub fn proportions(&self) -> Vec<f64> {
        if self.total_count == 0 {
            return vec![0.0; self.bucket_counts.len()];
        }
        self.bucket_counts
            .iter()
            .map(|&c| c as f64 / self.total_count as f64)
            .collect()
    }
}

/// Detects distribution drift by comparing current histograms to historical baselines.
pub struct DriftDetector {
    snapshots: scc::HashMap<String, Vec<HistogramSnapshot>>,
}

impl DriftDetector {
    pub fn new() -> Self {
        Self {
            snapshots: scc::HashMap::new(),
        }
    }

    /// Record a histogram snapshot for a column.
    pub fn record_snapshot(
        &self,
        table_name: &str,
        column: &str,
        snapshot: HistogramSnapshot,
    ) -> Result<()> {
        let key = format!("{}:{}", table_name, column);
        self.snapshots
            .entry_sync(key)
            .and_modify(|v| v.push(snapshot.clone()))
            .or_insert_with(|| vec![snapshot]);
        Ok(())
    }

    /// Detect drift between current histogram and historical baseline within the time window.
    /// Returns None if no drift detected or insufficient baseline data.
    pub fn detect_drift(
        &self,
        table_name: &str,
        column: &str,
        current: &HistogramSnapshot,
        window_secs: u64,
    ) -> Option<DriftResult> {
        let key = format!("{}:{}", table_name, column);

        // Read baseline proportions from historical snapshots within the time window
        let baseline_data: Option<(Vec<f64>, usize)> = self
            .snapshots
            .read_sync(&key, |_k, history| {
                let cutoff = current.recorded_at - window_secs as i64;
                let window_snapshots: Vec<&HistogramSnapshot> = history
                    .iter()
                    .filter(|s| s.recorded_at >= cutoff && s.recorded_at < current.recorded_at)
                    .collect();

                if window_snapshots.is_empty() {
                    return None;
                }

                let snap_count = window_snapshots.len();
                let num_buckets = current.bucket_counts.len();

                // Average proportions across all snapshots in window
                let mut avg = vec![0.0f64; num_buckets];
                for snap in &window_snapshots {
                    let props = snap.proportions();
                    for (i, &p) in props.iter().enumerate() {
                        if i < num_buckets {
                            avg[i] += p;
                        }
                    }
                }
                let count = snap_count as f64;
                for v in &mut avg {
                    *v /= count;
                }
                Some((avg, snap_count))
            })
            .flatten();

        let (baseline, baseline_count) = match baseline_data {
            Some(d) => d,
            None => return None,
        };
        if baseline_count == 0 {
            return None;
        }

        let current_proportions = current.proportions();
        let psi = calculate_psi(&baseline, &current_proportions);

        let severity = if psi > 0.25 {
            DriftSeverity::Critical
        } else if psi > 0.10 {
            DriftSeverity::Warning
        } else {
            return None;
        };

        Some(DriftResult {
            column: column.to_string(),
            psi_score: psi,
            severity,
            baseline_bucket_count: baseline.len(),
            current_bucket_count: current_proportions.len(),
        })
    }

    /// Get all historical snapshots for a column.
    pub fn get_snapshots(&self, table_name: &str, column: &str) -> Vec<HistogramSnapshot> {
        let key = format!("{}:{}", table_name, column);
        self.snapshots
            .read_sync(&key, |_k, v| v.clone())
            .unwrap_or_default()
    }
}

/// Calculate Population Stability Index between baseline and current distributions.
/// PSI = sum( (current_i - baseline_i) * ln(current_i / baseline_i) )
/// Small epsilon added to avoid ln(0).
pub fn calculate_psi(baseline: &[f64], current: &[f64]) -> f64 {
    let epsilon = 1e-10;
    let len = baseline.len().min(current.len());
    let mut psi = 0.0;

    for i in 0..len {
        let b = baseline[i].max(epsilon);
        let c = current[i].max(epsilon);
        psi += (c - b) * (c / b).ln();
    }

    psi
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_psi_identical_distributions() {
        let dist = vec![0.25, 0.25, 0.25, 0.25];
        let psi = calculate_psi(&dist, &dist);
        assert!(
            psi.abs() < 1e-8,
            "PSI of identical distributions should be ~0, got {}",
            psi
        );
    }

    #[test]
    fn test_psi_shifted_distribution() {
        let baseline = vec![0.25, 0.25, 0.25, 0.25];
        let shifted = vec![0.10, 0.10, 0.40, 0.40];
        let psi = calculate_psi(&baseline, &shifted);
        assert!(
            psi > 0.10,
            "Shifted distribution should have PSI > 0.10, got {}",
            psi
        );
    }

    #[test]
    fn test_psi_with_zeros() {
        let baseline = vec![0.0, 0.5, 0.5];
        let current = vec![0.3, 0.3, 0.4];
        let psi = calculate_psi(&baseline, &current);
        assert!(psi.is_finite(), "PSI with zeros should be finite");
    }

    #[test]
    fn test_histogram_proportions() {
        let snap = HistogramSnapshot {
            column: "price".to_string(),
            bucket_boundaries: vec![0.0, 10.0, 20.0, 30.0],
            bucket_counts: vec![100, 200, 300, 400],
            total_count: 1000,
            recorded_at: 1000,
        };
        let props = snap.proportions();
        assert_eq!(props.len(), 4);
        assert!((props[0] - 0.1).abs() < 0.001);
        assert!((props[1] - 0.2).abs() < 0.001);
        assert!((props[2] - 0.3).abs() < 0.001);
        assert!((props[3] - 0.4).abs() < 0.001);
    }

    #[test]
    fn test_drift_detector_no_baseline() {
        let detector = DriftDetector::new();
        let current = HistogramSnapshot {
            column: "val".to_string(),
            bucket_boundaries: vec![0.0, 50.0, 100.0],
            bucket_counts: vec![500, 500],
            total_count: 1000,
            recorded_at: 2000,
        };
        let result = detector.detect_drift("tbl", "val", &current, 3600);
        assert!(result.is_none());
    }

    #[test]
    fn test_drift_detector_with_baseline() {
        let detector = DriftDetector::new();

        // Record baseline snapshots with uniform distribution
        for t in 0..5 {
            let snap = HistogramSnapshot {
                column: "val".to_string(),
                bucket_boundaries: vec![0.0, 50.0, 100.0],
                bucket_counts: vec![500, 500],
                total_count: 1000,
                recorded_at: 1000 + t,
            };
            detector
                .record_snapshot("tbl", "val", snap)
                .expect("record");
        }

        // Current has a very different distribution
        let current = HistogramSnapshot {
            column: "val".to_string(),
            bucket_boundaries: vec![0.0, 50.0, 100.0],
            bucket_counts: vec![50, 950],
            total_count: 1000,
            recorded_at: 2000,
        };

        let result = detector.detect_drift("tbl", "val", &current, 3600);
        assert!(result.is_some(), "should detect drift");
        let drift = result.expect("drift result");
        assert!(drift.psi_score > 0.10);
        assert!(
            drift.severity == DriftSeverity::Warning || drift.severity == DriftSeverity::Critical
        );
    }

    #[test]
    fn test_drift_detector_no_drift() {
        let detector = DriftDetector::new();

        for t in 0..5 {
            let snap = HistogramSnapshot {
                column: "val".to_string(),
                bucket_boundaries: vec![0.0, 50.0, 100.0],
                bucket_counts: vec![500, 500],
                total_count: 1000,
                recorded_at: 1000 + t,
            };
            detector
                .record_snapshot("tbl", "val", snap)
                .expect("record");
        }

        // Current is the same distribution
        let current = HistogramSnapshot {
            column: "val".to_string(),
            bucket_boundaries: vec![0.0, 50.0, 100.0],
            bucket_counts: vec![501, 499],
            total_count: 1000,
            recorded_at: 2000,
        };

        let result = detector.detect_drift("tbl", "val", &current, 3600);
        assert!(
            result.is_none(),
            "no drift for nearly identical distribution"
        );
    }
}
