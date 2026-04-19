//! Materialized view management with incremental maintenance and staleness-aware queries.

use crate::ids::MaterializedViewId;
use crate::pipeline::RefreshMode;
use std::sync::Arc;
use zyron_common::{Result, ZyronError};

/// A materialized view backed by a physical table.
#[derive(Debug, Clone)]
pub struct MaterializedView {
    pub id: MaterializedViewId,
    pub name: String,
    pub query_sql: String,
    pub backing_table_id: u32,
    pub refresh_mode: RefreshMode,
    pub last_refreshed: Option<i64>,
    pub max_staleness_ms: Option<u64>,
}

/// Tracks running aggregate state for incremental maintenance of simple aggregate MVs.
#[derive(Debug, Clone)]
pub struct IncrementalAggState {
    pub group_key: Vec<u8>,
    pub count: i64,
    pub sum: f64,
    pub min: Option<f64>,
    pub max: Option<f64>,
}

impl IncrementalAggState {
    pub fn new(group_key: Vec<u8>) -> Self {
        Self {
            group_key,
            count: 0,
            sum: 0.0,
            min: None,
            max: None,
        }
    }

    /// Apply a delta (positive for inserts, negative for deletes).
    pub fn apply_delta(&mut self, value: f64, delta_count: i64) {
        self.count += delta_count;
        self.sum += value * delta_count as f64;

        if delta_count > 0 {
            self.min = Some(self.min.map_or(value, |m: f64| m.min(value)));
            self.max = Some(self.max.map_or(value, |m: f64| m.max(value)));
        }
        // For deletes of min/max boundary values, a full recompute is needed.
        // The caller should check if the deleted value equals min or max.
    }

    /// Derive the current average from maintained count and sum.
    pub fn avg(&self) -> Option<f64> {
        if self.count > 0 {
            Some(self.sum / self.count as f64)
        } else {
            None
        }
    }

    /// Check if a deleted value was the boundary min or max, requiring recompute.
    pub fn needs_recompute_min(&self, deleted_value: f64) -> bool {
        self.min
            .map_or(false, |m| (m - deleted_value).abs() < f64::EPSILON)
    }

    pub fn needs_recompute_max(&self, deleted_value: f64) -> bool {
        self.max
            .map_or(false, |m| (m - deleted_value).abs() < f64::EPSILON)
    }
}

/// Manages materialized view lifecycle, refresh, and staleness tracking.
pub struct MaterializedViewManager {
    views: scc::HashMap<String, Arc<MaterializedView>>,
}

impl MaterializedViewManager {
    pub fn new() -> Self {
        Self {
            views: scc::HashMap::new(),
        }
    }

    /// Register a new materialized view.
    pub fn create_mv(&self, mv: MaterializedView) -> Result<()> {
        let name = mv.name.clone();
        let arc = Arc::new(mv);
        if self.views.insert_sync(name.clone(), arc).is_err() {
            return Err(ZyronError::MaterializedViewAlreadyExists(name));
        }
        Ok(())
    }

    /// Remove a materialized view.
    pub fn drop_mv(&self, name: &str) -> Result<()> {
        if self.views.remove_sync(name).is_none() {
            return Err(ZyronError::MaterializedViewNotFound(name.to_string()));
        }
        Ok(())
    }

    /// Get a materialized view by name.
    pub fn get_mv(&self, name: &str) -> Option<Arc<MaterializedView>> {
        self.views.read_sync(name, |_k, v| Arc::clone(v))
    }

    /// Check if a materialized view is stale based on its max_staleness_ms setting.
    pub fn is_stale(&self, name: &str, current_time_ms: i64) -> bool {
        let mv = match self.get_mv(name) {
            Some(mv) => mv,
            None => return false,
        };

        let max_staleness = match mv.max_staleness_ms {
            Some(ms) => ms,
            None => return false,
        };

        let last_refreshed = match mv.last_refreshed {
            Some(t) => t,
            None => return true,
        };

        let age_ms = (current_time_ms - last_refreshed).max(0) as u64;
        age_ms > max_staleness
    }

    /// Update the last_refreshed timestamp for a materialized view.
    pub fn mark_refreshed(&self, name: &str, timestamp: i64) -> Result<()> {
        let current = self
            .get_mv(name)
            .ok_or_else(|| ZyronError::MaterializedViewNotFound(name.to_string()))?;
        let mut updated = (*current).clone();
        updated.last_refreshed = Some(timestamp);
        let _ = self.views.entry_sync(name.to_string()).and_modify(|v| {
            *v = Arc::new(updated);
        });
        Ok(())
    }

    /// List all registered materialized views.
    pub fn list_views(&self) -> Vec<Arc<MaterializedView>> {
        let mut result = Vec::new();
        self.views.iter_sync(|_k, v| {
            result.push(Arc::clone(v));
            true
        });
        result
    }

    /// Return the number of managed materialized views.
    pub fn view_count(&self) -> usize {
        self.views.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_mv(name: &str) -> MaterializedView {
        MaterializedView {
            id: MaterializedViewId(1),
            name: name.to_string(),
            query_sql: "SELECT count(*) FROM orders GROUP BY region".to_string(),
            backing_table_id: 200,
            refresh_mode: RefreshMode::Incremental,
            last_refreshed: Some(1000),
            max_staleness_ms: Some(5000),
        }
    }

    #[test]
    fn test_mv_crud() {
        let mgr = MaterializedViewManager::new();
        mgr.create_mv(test_mv("mv1")).expect("create");
        assert_eq!(mgr.view_count(), 1);

        let mv = mgr.get_mv("mv1").expect("should exist");
        assert_eq!(mv.query_sql, "SELECT count(*) FROM orders GROUP BY region");

        mgr.drop_mv("mv1").expect("drop");
        assert_eq!(mgr.view_count(), 0);
    }

    #[test]
    fn test_mv_duplicate() {
        let mgr = MaterializedViewManager::new();
        mgr.create_mv(test_mv("mv1")).expect("first");
        let result = mgr.create_mv(test_mv("mv1"));
        assert!(matches!(
            result,
            Err(ZyronError::MaterializedViewAlreadyExists(_))
        ));
    }

    #[test]
    fn test_staleness_check() {
        let mgr = MaterializedViewManager::new();
        mgr.create_mv(test_mv("mv1")).expect("create");

        // Not stale yet (current = 5999, last = 1000, max = 5000)
        assert!(!mgr.is_stale("mv1", 5999));

        // Stale (current = 7000, last = 1000, max = 5000, age = 6000 > 5000)
        assert!(mgr.is_stale("mv1", 7000));
    }

    #[test]
    fn test_mark_refreshed() {
        let mgr = MaterializedViewManager::new();
        mgr.create_mv(test_mv("mv1")).expect("create");
        mgr.mark_refreshed("mv1", 9000).expect("refresh");

        let mv = mgr.get_mv("mv1").expect("exists");
        assert_eq!(mv.last_refreshed, Some(9000));

        // No longer stale at 13000 (age = 4000 < 5000)
        assert!(!mgr.is_stale("mv1", 13000));
    }

    #[test]
    fn test_incremental_agg_state() {
        let mut state = IncrementalAggState::new(vec![1]);

        // Insert 3 values: 10, 20, 30
        state.apply_delta(10.0, 1);
        state.apply_delta(20.0, 1);
        state.apply_delta(30.0, 1);

        assert_eq!(state.count, 3);
        assert!((state.sum - 60.0).abs() < 0.001);
        assert!((state.avg().expect("avg") - 20.0).abs() < 0.001);
        assert_eq!(state.min, Some(10.0));
        assert_eq!(state.max, Some(30.0));
    }

    #[test]
    fn test_incremental_agg_delete() {
        let mut state = IncrementalAggState::new(vec![1]);
        state.apply_delta(10.0, 1);
        state.apply_delta(20.0, 1);
        state.apply_delta(30.0, 1);

        // Delete the row with value 20
        state.apply_delta(20.0, -1);
        assert_eq!(state.count, 2);
        assert!((state.sum - 40.0).abs() < 0.001);
        assert!((state.avg().expect("avg") - 20.0).abs() < 0.001);

        // Deleting 10.0 means min needs recompute
        assert!(state.needs_recompute_min(10.0));
        assert!(!state.needs_recompute_min(20.0));
    }

    #[test]
    fn test_no_staleness_without_config() {
        let mgr = MaterializedViewManager::new();
        let mut mv = test_mv("mv1");
        mv.max_staleness_ms = None;
        mgr.create_mv(mv).expect("create");

        // Without max_staleness set, never considered stale
        assert!(!mgr.is_stale("mv1", 999999));
    }
}
