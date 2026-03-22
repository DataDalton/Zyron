//! Session IP binding, query complexity limits, and recurring time windows.
//!
//! TimeWindow represents a recurring schedule (hours + day-of-week bitmask).
//! SessionBinding ties a role to a specific IP address.
//! QueryLimits constrains resource usage per role.

use crate::role::RoleId;
use serde::{Deserialize, Serialize};
use zyron_common::{Result, ZyronError};

/// A recurring time window defined by hour range and day-of-week bitmask.
/// Days bitmask: Mon=1, Tue=2, Wed=4, Thu=8, Fri=16, Sat=32, Sun=64.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct TimeWindow {
    pub start_hour: u8,
    pub end_hour: u8,
    /// Bitmask of active days. Mon=1, Tue=2, Wed=4, Thu=8, Fri=16, Sat=32, Sun=64.
    pub days: u8,
}

impl TimeWindow {
    /// Checks if the given Unix timestamp falls within this time window.
    /// Handles hour wraparound (e.g., start=22, end=6 covers 22:00 to 06:00).
    pub fn is_active(&self, timestamp: u64) -> bool {
        let hour = ((timestamp / 3600) % 24) as u8;
        // Days since epoch. Epoch (1970-01-01) was Thursday.
        // +3 makes Monday=0, Tuesday=1, ..., Sunday=6.
        let days_since_epoch = timestamp / 86400;
        let day_of_week = (days_since_epoch + 3) % 7;
        let day_bit = 1u8 << day_of_week;

        if self.days & day_bit == 0 {
            return false;
        }

        if self.start_hour <= self.end_hour {
            // Normal range, e.g. 9-17
            hour >= self.start_hour && hour < self.end_hour
        } else {
            // Wraparound range, e.g. 22-6 means 22..24 or 0..6
            hour >= self.start_hour || hour < self.end_hour
        }
    }

    /// Returns a TimeWindow for business hours: Monday through Friday, 9:00 to 17:00.
    pub fn business_hours() -> Self {
        Self {
            start_hour: 9,
            end_hour: 17,
            days: 0b0001_1111, // Mon-Fri = 31
        }
    }

    /// Serializes to 3 bytes: start_hour, end_hour, days.
    pub fn to_bytes(&self) -> Vec<u8> {
        vec![self.start_hour, self.end_hour, self.days]
    }

    /// Deserializes from 3 bytes.
    pub fn from_bytes(data: &[u8]) -> Result<Self> {
        if data.len() < 3 {
            return Err(ZyronError::DecodingFailed(
                "TimeWindow data too short".to_string(),
            ));
        }
        Ok(Self {
            start_hour: data[0],
            end_hour: data[1],
            days: data[2],
        })
    }
}

/// Binds a session to a specific IP address for the duration of the connection.
#[derive(Debug, Clone)]
pub struct SessionBinding {
    pub role_id: RoleId,
    pub bound_ip: Option<String>,
    pub require_binding: bool,
}

/// Per-role limits on query resource usage.
#[derive(Debug, Clone)]
pub struct QueryLimits {
    pub role_id: RoleId,
    pub max_scan_rows: Option<u64>,
    pub max_result_rows: Option<u64>,
    pub max_execution_time_ms: Option<u64>,
    pub max_memory_bytes: Option<u64>,
    pub max_temp_bytes: Option<u64>,
    pub allow_full_scan: bool,
}

impl Default for QueryLimits {
    fn default() -> Self {
        Self {
            role_id: RoleId(0),
            max_scan_rows: None,
            max_result_rows: None,
            max_execution_time_ms: None,
            max_memory_bytes: None,
            max_temp_bytes: None,
            allow_full_scan: true,
        }
    }
}

/// Stores and merges query limits for multiple roles.
pub struct QueryLimitStore {
    limits: scc::HashMap<RoleId, QueryLimits>,
}

impl QueryLimitStore {
    pub fn new() -> Self {
        Self {
            limits: scc::HashMap::new(),
        }
    }

    /// Merges limits from all effective roles. The most restrictive value wins:
    /// smallest Some value for numeric limits, false if any role disallows full scans.
    pub fn get_limits(&self, effective_roles: &[RoleId]) -> QueryLimits {
        let mut merged = QueryLimits::default();
        for role_id in effective_roles {
            self.limits.read_sync(role_id, |_k, v| {
                merged.max_scan_rows = merge_min_option(merged.max_scan_rows, v.max_scan_rows);
                merged.max_result_rows =
                    merge_min_option(merged.max_result_rows, v.max_result_rows);
                merged.max_execution_time_ms =
                    merge_min_option(merged.max_execution_time_ms, v.max_execution_time_ms);
                merged.max_memory_bytes =
                    merge_min_option(merged.max_memory_bytes, v.max_memory_bytes);
                merged.max_temp_bytes = merge_min_option(merged.max_temp_bytes, v.max_temp_bytes);
                if !v.allow_full_scan {
                    merged.allow_full_scan = false;
                }
            });
        }
        merged
    }

    /// Sets (or replaces) limits for a role.
    pub fn set_limits(&self, limits: QueryLimits) {
        let _ = self.limits.insert_sync(limits.role_id, limits);
    }

    /// Bulk-loads limits, replacing any existing entries.
    pub fn load(&self, all_limits: Vec<QueryLimits>) {
        for limits in all_limits {
            let _ = self.limits.insert_sync(limits.role_id, limits);
        }
    }
}

/// Returns the minimum of two Option<u64> values where Some beats None.
fn merge_min_option(a: Option<u64>, b: Option<u64>) -> Option<u64> {
    match (a, b) {
        (Some(va), Some(vb)) => Some(va.min(vb)),
        (Some(va), None) => Some(va),
        (None, Some(vb)) => Some(vb),
        (None, None) => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Helper: returns a timestamp for a known day and hour.
    // 2024-01-01 00:00:00 UTC is a Monday. Epoch + 19723 days.
    // 19723 * 86400 = 1704067200
    fn monday_at_hour(hour: u8) -> u64 {
        1704067200 + (hour as u64) * 3600
    }

    fn tuesday_at_hour(hour: u8) -> u64 {
        monday_at_hour(hour) + 86400
    }

    fn saturday_at_hour(hour: u8) -> u64 {
        monday_at_hour(hour) + 5 * 86400
    }

    fn sunday_at_hour(hour: u8) -> u64 {
        monday_at_hour(hour) + 6 * 86400
    }

    #[test]
    fn test_business_hours_monday_10am() {
        let bh = TimeWindow::business_hours();
        assert!(bh.is_active(monday_at_hour(10)));
    }

    #[test]
    fn test_business_hours_monday_8am_before_start() {
        let bh = TimeWindow::business_hours();
        assert!(!bh.is_active(monday_at_hour(8)));
    }

    #[test]
    fn test_business_hours_monday_17_at_end() {
        let bh = TimeWindow::business_hours();
        // end_hour=17 means 17:00 is NOT included (hour < end_hour)
        assert!(!bh.is_active(monday_at_hour(17)));
    }

    #[test]
    fn test_business_hours_saturday_rejected() {
        let bh = TimeWindow::business_hours();
        assert!(!bh.is_active(saturday_at_hour(12)));
    }

    #[test]
    fn test_business_hours_sunday_rejected() {
        let bh = TimeWindow::business_hours();
        assert!(!bh.is_active(sunday_at_hour(10)));
    }

    #[test]
    fn test_business_hours_tuesday_noon() {
        let bh = TimeWindow::business_hours();
        assert!(bh.is_active(tuesday_at_hour(12)));
    }

    #[test]
    fn test_wraparound_window() {
        // Night shift: 22:00 to 06:00, every day
        let tw = TimeWindow {
            start_hour: 22,
            end_hour: 6,
            days: 0b0111_1111, // all days
        };
        assert!(tw.is_active(monday_at_hour(23)));
        assert!(tw.is_active(monday_at_hour(0)));
        assert!(tw.is_active(monday_at_hour(3)));
        assert!(!tw.is_active(monday_at_hour(10)));
        assert!(!tw.is_active(monday_at_hour(6)));
    }

    #[test]
    fn test_time_window_to_bytes_from_bytes() {
        let tw = TimeWindow::business_hours();
        let bytes = tw.to_bytes();
        let restored = TimeWindow::from_bytes(&bytes).expect("decode should succeed");
        assert_eq!(tw, restored);
    }

    #[test]
    fn test_time_window_from_bytes_too_short() {
        assert!(TimeWindow::from_bytes(&[1, 2]).is_err());
    }

    #[test]
    fn test_query_limits_default() {
        let ql = QueryLimits::default();
        assert!(ql.allow_full_scan);
        assert!(ql.max_scan_rows.is_none());
        assert_eq!(ql.role_id, RoleId(0));
    }

    #[test]
    fn test_query_limits_merge_most_restrictive() {
        let store = QueryLimitStore::new();
        store.set_limits(QueryLimits {
            role_id: RoleId(1),
            max_scan_rows: Some(1000),
            max_result_rows: None,
            max_execution_time_ms: Some(5000),
            max_memory_bytes: None,
            max_temp_bytes: None,
            allow_full_scan: true,
        });
        store.set_limits(QueryLimits {
            role_id: RoleId(2),
            max_scan_rows: Some(500),
            max_result_rows: Some(100),
            max_execution_time_ms: Some(10000),
            max_memory_bytes: None,
            max_temp_bytes: None,
            allow_full_scan: false,
        });

        let merged = store.get_limits(&[RoleId(1), RoleId(2)]);
        assert_eq!(merged.max_scan_rows, Some(500));
        assert_eq!(merged.max_result_rows, Some(100));
        assert_eq!(merged.max_execution_time_ms, Some(5000));
        assert!(!merged.allow_full_scan);
    }

    #[test]
    fn test_query_limits_merge_no_roles() {
        let store = QueryLimitStore::new();
        let merged = store.get_limits(&[]);
        assert!(merged.allow_full_scan);
        assert!(merged.max_scan_rows.is_none());
    }

    #[test]
    fn test_session_binding_fields() {
        let sb = SessionBinding {
            role_id: RoleId(5),
            bound_ip: Some("10.0.0.1".to_string()),
            require_binding: true,
        };
        assert_eq!(sb.role_id, RoleId(5));
        assert_eq!(sb.bound_ip.as_deref(), Some("10.0.0.1"));
        assert!(sb.require_binding);
    }

    #[test]
    fn test_query_limit_store_load() {
        let store = QueryLimitStore::new();
        let limits = vec![
            QueryLimits {
                role_id: RoleId(10),
                max_scan_rows: Some(2000),
                ..Default::default()
            },
            QueryLimits {
                role_id: RoleId(20),
                max_result_rows: Some(50),
                ..Default::default()
            },
        ];
        store.load(limits);
        let merged = store.get_limits(&[RoleId(10), RoleId(20)]);
        assert_eq!(merged.max_scan_rows, Some(2000));
        assert_eq!(merged.max_result_rows, Some(50));
    }
}
