//! Lock-free CDC metrics for observability.
//!
//! All counters use AtomicU64 for contention-free updates on the write path.
//! Metrics are exposed in Prometheus text format.

use std::sync::atomic::{AtomicU64, Ordering};

use scc::HashMap as SccHashMap;

// ---------------------------------------------------------------------------
// CdcMetrics
// ---------------------------------------------------------------------------

/// Lock-free CDC metrics registry.
pub struct CdcMetrics {
    // CDF metrics
    pub cdf_records_appended: AtomicU64,
    pub cdf_records_queried: AtomicU64,
    pub cdf_bytes_written: AtomicU64,

    // Replication slot metrics
    pub slot_changes_decoded: AtomicU64,
    pub slot_lag_bytes: SccHashMap<String, AtomicU64>,

    // Outbound stream metrics
    pub stream_records_sent: AtomicU64,
    pub stream_send_failures: AtomicU64,
    pub stream_retry_count: AtomicU64,

    // Inbound ingest metrics
    pub ingest_records_applied: AtomicU64,
    pub ingest_records_failed: AtomicU64,
    pub ingest_dead_letter_count: AtomicU64,

    // Retention metrics
    pub retention_records_purged: AtomicU64,
    pub retention_bytes_reclaimed: AtomicU64,
}

impl CdcMetrics {
    pub fn new() -> Self {
        Self {
            cdf_records_appended: AtomicU64::new(0),
            cdf_records_queried: AtomicU64::new(0),
            cdf_bytes_written: AtomicU64::new(0),
            slot_changes_decoded: AtomicU64::new(0),
            slot_lag_bytes: SccHashMap::new(),
            stream_records_sent: AtomicU64::new(0),
            stream_send_failures: AtomicU64::new(0),
            stream_retry_count: AtomicU64::new(0),
            ingest_records_applied: AtomicU64::new(0),
            ingest_records_failed: AtomicU64::new(0),
            ingest_dead_letter_count: AtomicU64::new(0),
            retention_records_purged: AtomicU64::new(0),
            retention_bytes_reclaimed: AtomicU64::new(0),
        }
    }

    /// Updates the lag metric for a specific slot.
    pub fn update_slot_lag(&self, slot_name: &str, lag_bytes: u64) {
        // Fast path: slot exists, update atomic without allocating a String key.
        if self
            .slot_lag_bytes
            .read_sync(slot_name, |_, v| {
                v.store(lag_bytes, Ordering::Relaxed);
            })
            .is_some()
        {
            return;
        }
        // Cold path: first time seeing this slot.
        let _ = self
            .slot_lag_bytes
            .insert_sync(slot_name.to_string(), AtomicU64::new(lag_bytes));
    }

    /// Removes the lag metric for a dropped slot.
    pub fn remove_slot_lag(&self, slot_name: &str) {
        let _ = self.slot_lag_bytes.remove_sync(slot_name);
    }

    /// Renders all CDC metrics in Prometheus text exposition format.
    pub fn render_prometheus(&self) -> String {
        let mut out = String::with_capacity(2048);

        // CDF metrics
        Self::write_counter(
            &mut out,
            "zyrondb_cdc_cdf_records_appended_total",
            "Total change records written to CDF files",
            self.cdf_records_appended.load(Ordering::Relaxed),
        );
        Self::write_counter(
            &mut out,
            "zyrondb_cdc_cdf_records_queried_total",
            "Total records read via table_changes()",
            self.cdf_records_queried.load(Ordering::Relaxed),
        );
        Self::write_counter(
            &mut out,
            "zyrondb_cdc_cdf_bytes_written_total",
            "Total bytes written to CDF files",
            self.cdf_bytes_written.load(Ordering::Relaxed),
        );

        // Slot metrics
        Self::write_counter(
            &mut out,
            "zyrondb_cdc_slot_changes_decoded_total",
            "Total changes decoded through replication slots",
            self.slot_changes_decoded.load(Ordering::Relaxed),
        );

        // Per-slot lag (gauge)
        out.push_str("# HELP zyrondb_cdc_slot_lag_bytes Replication slot lag in bytes\n");
        out.push_str("# TYPE zyrondb_cdc_slot_lag_bytes gauge\n");
        self.slot_lag_bytes.iter_sync(|name, lag| {
            let val = lag.load(Ordering::Relaxed);
            out.push_str(&format!(
                "zyrondb_cdc_slot_lag_bytes{{slot=\"{name}\"}} {val}\n"
            ));
            true
        });

        // Stream metrics
        Self::write_counter(
            &mut out,
            "zyrondb_cdc_stream_records_sent_total",
            "Total records sent to sinks",
            self.stream_records_sent.load(Ordering::Relaxed),
        );
        Self::write_counter(
            &mut out,
            "zyrondb_cdc_stream_send_failures_total",
            "Total sink write failures",
            self.stream_send_failures.load(Ordering::Relaxed),
        );
        Self::write_counter(
            &mut out,
            "zyrondb_cdc_stream_retry_count_total",
            "Total retries across all streams",
            self.stream_retry_count.load(Ordering::Relaxed),
        );

        // Ingest metrics
        Self::write_counter(
            &mut out,
            "zyrondb_cdc_ingest_records_applied_total",
            "Total records ingested",
            self.ingest_records_applied.load(Ordering::Relaxed),
        );
        Self::write_counter(
            &mut out,
            "zyrondb_cdc_ingest_records_failed_total",
            "Total ingest failures",
            self.ingest_records_failed.load(Ordering::Relaxed),
        );
        Self::write_counter(
            &mut out,
            "zyrondb_cdc_ingest_dead_letter_count_total",
            "Total dead letter records",
            self.ingest_dead_letter_count.load(Ordering::Relaxed),
        );

        // Retention metrics
        Self::write_counter(
            &mut out,
            "zyrondb_cdc_retention_records_purged_total",
            "Total records purged by retention",
            self.retention_records_purged.load(Ordering::Relaxed),
        );
        Self::write_counter(
            &mut out,
            "zyrondb_cdc_retention_bytes_reclaimed_total",
            "Total bytes freed by retention",
            self.retention_bytes_reclaimed.load(Ordering::Relaxed),
        );

        out
    }

    fn write_counter(out: &mut String, name: &str, help: &str, value: u64) {
        out.push_str(&format!("# HELP {name} {help}\n"));
        out.push_str(&format!("# TYPE {name} counter\n"));
        out.push_str(&format!("{name} {value}\n"));
    }
}

impl Default for CdcMetrics {
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

    #[test]
    fn test_metrics_initial_values() {
        let m = CdcMetrics::new();
        assert_eq!(m.cdf_records_appended.load(Ordering::Relaxed), 0);
        assert_eq!(m.stream_records_sent.load(Ordering::Relaxed), 0);
    }

    #[test]
    fn test_increment_counters() {
        let m = CdcMetrics::new();
        m.cdf_records_appended.fetch_add(10, Ordering::Relaxed);
        m.stream_records_sent.fetch_add(5, Ordering::Relaxed);
        assert_eq!(m.cdf_records_appended.load(Ordering::Relaxed), 10);
        assert_eq!(m.stream_records_sent.load(Ordering::Relaxed), 5);
    }

    #[test]
    fn test_slot_lag_update() {
        let m = CdcMetrics::new();
        m.update_slot_lag("slot1", 1000);
        m.update_slot_lag("slot2", 2000);

        let mut total = 0u64;
        m.slot_lag_bytes.iter_sync(|_name, lag| {
            total += lag.load(Ordering::Relaxed);
            true
        });
        assert_eq!(total, 3000);

        m.update_slot_lag("slot1", 500);
        let lag1 = m
            .slot_lag_bytes
            .read_sync("slot1", |_, v| v.load(Ordering::Relaxed))
            .unwrap_or(0);
        assert_eq!(lag1, 500);

        m.remove_slot_lag("slot1");
        assert!(m.slot_lag_bytes.read_sync("slot1", |_, _| ()).is_none());
    }

    #[test]
    fn test_prometheus_render() {
        let m = CdcMetrics::new();
        m.cdf_records_appended.fetch_add(42, Ordering::Relaxed);
        m.update_slot_lag("test_slot", 1024);

        let output = m.render_prometheus();
        assert!(output.contains("zyrondb_cdc_cdf_records_appended_total 42"));
        assert!(output.contains("zyrondb_cdc_slot_lag_bytes{slot=\"test_slot\"} 1024"));
        assert!(output.contains("# TYPE zyrondb_cdc_cdf_records_appended_total counter"));
        assert!(output.contains("# TYPE zyrondb_cdc_slot_lag_bytes gauge"));
    }

    #[test]
    fn test_default_trait() {
        let m = CdcMetrics::default();
        assert_eq!(m.cdf_records_appended.load(Ordering::Relaxed), 0);
    }
}
