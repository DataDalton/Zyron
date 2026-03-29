//! Auto-materialize advisor: tracks query patterns and recommends materialized views.

use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

/// Execution statistics for a query pattern.
#[derive(Debug, Clone)]
pub struct QueryStats {
    pub fingerprint: u64,
    pub normalized_sql: String,
    pub execution_count: u64,
    pub total_cost_us: u64,
    pub avg_time_us: u64,
    pub last_seen: i64,
}

/// A recommendation to create a materialized view.
#[derive(Debug, Clone)]
pub struct MvRecommendation {
    pub query_fingerprint: u64,
    pub suggested_name: String,
    pub suggested_sql: String,
    pub estimated_savings_per_hour_ms: u64,
    pub refresh_recommendation: String,
}

/// Tracks query execution patterns for MV recommendation.
pub struct QueryPatternTracker {
    patterns: scc::HashMap<u64, QueryStats>,
}

impl QueryPatternTracker {
    pub fn new() -> Self {
        Self {
            patterns: scc::HashMap::new(),
        }
    }

    /// Record a query execution. Updates existing pattern or creates a new one.
    pub fn record_query(
        &self,
        fingerprint: u64,
        sql: &str,
        execution_time_us: u64,
        timestamp: i64,
    ) {
        self.patterns
            .entry_sync(fingerprint)
            .and_modify(|stats| {
                stats.execution_count += 1;
                stats.total_cost_us += execution_time_us;
                stats.avg_time_us = stats.total_cost_us / stats.execution_count;
                stats.last_seen = timestamp;
            })
            .or_insert_with(|| QueryStats {
                fingerprint,
                normalized_sql: sql.to_string(),
                execution_count: 1,
                total_cost_us: execution_time_us,
                avg_time_us: execution_time_us,
                last_seen: timestamp,
            });
    }

    /// Get stats for a specific query fingerprint.
    pub fn get_stats(&self, fingerprint: u64) -> Option<QueryStats> {
        self.patterns.read_sync(&fingerprint, |_k, v| v.clone())
    }

    /// Get all tracked query patterns.
    pub fn all_patterns(&self) -> Vec<QueryStats> {
        let mut result = Vec::new();
        self.patterns.iter_sync(|_k, v| {
            result.push(v.clone());
            true
        });
        result
    }

    /// Return the number of tracked patterns.
    pub fn pattern_count(&self) -> usize {
        self.patterns.len()
    }
}

/// Analyzes query patterns and generates materialized view recommendations.
pub struct MaterializeAdvisor {
    pub tracker: QueryPatternTracker,
    pub min_execution_count: u64,
    pub min_avg_time_us: u64,
}

impl MaterializeAdvisor {
    pub fn new(min_count: u64, min_avg_us: u64) -> Self {
        Self {
            tracker: QueryPatternTracker::new(),
            min_execution_count: min_count,
            min_avg_time_us: min_avg_us,
        }
    }

    /// Analyze tracked patterns and return recommendations for queries that exceed thresholds.
    pub fn analyze(&self) -> Vec<MvRecommendation> {
        let patterns = self.tracker.all_patterns();
        let mut recommendations = Vec::new();

        for stats in &patterns {
            if stats.execution_count >= self.min_execution_count
                && stats.avg_time_us >= self.min_avg_time_us
            {
                let savings_us = stats.execution_count * stats.avg_time_us;
                let savings_ms = savings_us / 1000;

                let name = format!("mv_auto_{:x}", stats.fingerprint);
                let sql = format!(
                    "CREATE MATERIALIZED VIEW {} AS {}",
                    name, stats.normalized_sql
                );

                // Simple heuristic: if query contains GROUP BY, recommend incremental
                let refresh = if stats.normalized_sql.to_lowercase().contains("group by") {
                    "incremental".to_string()
                } else {
                    "full".to_string()
                };

                recommendations.push(MvRecommendation {
                    query_fingerprint: stats.fingerprint,
                    suggested_name: name,
                    suggested_sql: sql,
                    estimated_savings_per_hour_ms: savings_ms,
                    refresh_recommendation: refresh,
                });
            }
        }

        // Sort by estimated savings descending
        recommendations.sort_by(|a, b| {
            b.estimated_savings_per_hour_ms
                .cmp(&a.estimated_savings_per_hour_ms)
        });
        recommendations
    }

    /// Compute a fingerprint for a SQL query by normalizing and hashing.
    pub fn query_fingerprint(sql: &str) -> u64 {
        let normalized: String = sql
            .to_lowercase()
            .split_whitespace()
            .collect::<Vec<&str>>()
            .join(" ");
        let mut hasher = DefaultHasher::new();
        normalized.hash(&mut hasher);
        hasher.finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_query_fingerprint_normalization() {
        let fp1 = MaterializeAdvisor::query_fingerprint("SELECT  *  FROM  orders");
        let fp2 = MaterializeAdvisor::query_fingerprint("select * from orders");
        assert_eq!(fp1, fp2);
    }

    #[test]
    fn test_query_fingerprint_different() {
        let fp1 = MaterializeAdvisor::query_fingerprint("SELECT * FROM orders");
        let fp2 = MaterializeAdvisor::query_fingerprint("SELECT * FROM users");
        assert_ne!(fp1, fp2);
    }

    #[test]
    fn test_pattern_tracker_record() {
        let tracker = QueryPatternTracker::new();
        let fp = 12345u64;

        tracker.record_query(fp, "SELECT * FROM t", 1000, 100);
        tracker.record_query(fp, "SELECT * FROM t", 2000, 200);
        tracker.record_query(fp, "SELECT * FROM t", 3000, 300);

        let stats = tracker.get_stats(fp).expect("should exist");
        assert_eq!(stats.execution_count, 3);
        assert_eq!(stats.total_cost_us, 6000);
        assert_eq!(stats.avg_time_us, 2000);
        assert_eq!(stats.last_seen, 300);
    }

    #[test]
    fn test_advisor_no_recommendations() {
        let advisor = MaterializeAdvisor::new(50, 100_000);

        // Only 3 executions, below min_execution_count of 50
        let fp = MaterializeAdvisor::query_fingerprint("SELECT * FROM orders");
        for i in 0..3 {
            advisor
                .tracker
                .record_query(fp, "SELECT * FROM orders", 200_000, i);
        }

        let recs = advisor.analyze();
        assert!(recs.is_empty());
    }

    #[test]
    fn test_advisor_with_recommendation() {
        let advisor = MaterializeAdvisor::new(5, 1000);

        let fp = MaterializeAdvisor::query_fingerprint(
            "SELECT region, count(*) FROM orders GROUP BY region",
        );
        for i in 0..10 {
            advisor.tracker.record_query(
                fp,
                "SELECT region, count(*) FROM orders GROUP BY region",
                5000,
                i,
            );
        }

        let recs = advisor.analyze();
        assert_eq!(recs.len(), 1);
        assert!(recs[0].suggested_sql.contains("CREATE MATERIALIZED VIEW"));
        assert_eq!(recs[0].refresh_recommendation, "incremental");
        assert!(recs[0].estimated_savings_per_hour_ms > 0);
    }

    #[test]
    fn test_advisor_sorted_by_savings() {
        let advisor = MaterializeAdvisor::new(2, 100);

        let fp1 = 111u64;
        let fp2 = 222u64;

        // fp1: 5 runs at 500us each = 2500us total savings
        for i in 0..5 {
            advisor
                .tracker
                .record_query(fp1, "SELECT * FROM small", 500, i);
        }
        // fp2: 10 runs at 2000us each = 20000us total savings (higher)
        for i in 0..10 {
            advisor
                .tracker
                .record_query(fp2, "SELECT * FROM big", 2000, i);
        }

        let recs = advisor.analyze();
        assert_eq!(recs.len(), 2);
        // Higher savings should come first
        assert_eq!(recs[0].query_fingerprint, fp2);
    }
}
