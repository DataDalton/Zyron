// Cohort retention analysis. Single pass over the event stream to assign
// each user to a cohort and record activity in each subsequent period.

use crate::value::{AnalyticsValue, VerifiedKeyMap, hash_value_128};
use zyron_common::error::{Result, ZyronError};
use zyron_common::{fx_mix, mix_finalize_2round};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CohortPeriod {
    Day,
    Week,
    Month,
    Quarter,
}

impl CohortPeriod {
    pub fn from_str_ci(s: &str) -> Option<Self> {
        match s.to_ascii_lowercase().as_str() {
            "day" | "daily" | "d" => Some(CohortPeriod::Day),
            "week" | "weekly" | "w" => Some(CohortPeriod::Week),
            "month" | "monthly" | "mo" | "m" => Some(CohortPeriod::Month),
            "quarter" | "quarterly" | "q" => Some(CohortPeriod::Quarter),
            _ => None,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CohortType {
    // Group user by the period of their first event
    FirstEvent,
    // Group user by an attribute string carried in the event
    Attribute,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CohortMetric {
    ActiveUsers,
    Revenue,
    Custom(String),
}

#[derive(Debug, Clone)]
pub struct CohortDefinition {
    pub user_id_column: String,
    pub event_time_column: String,
    pub cohort_period: CohortPeriod,
    pub analysis_period: CohortPeriod,
    pub cohort_type: CohortType,
}

#[derive(Debug, Clone)]
pub struct CohortAnalysis {
    pub definition: CohortDefinition,
    pub metric: CohortMetric,
}

#[derive(Debug, Clone)]
pub struct CohortRow {
    pub cohort_label: String,
    // Indexed by period offset 0..periods, NULL for periods where we have no data
    pub period_values: Vec<f64>,
}

#[derive(Debug, Clone)]
pub struct CohortResult {
    pub cohorts: Vec<CohortRow>,
    pub periods: u32,
}

// One input event for the cohort analyser
#[derive(Debug, Clone)]
pub struct CohortEvent {
    pub user_id: AnalyticsValue,
    pub event_time_ms: i64,
    pub revenue: Option<f64>,
    pub custom_value: Option<f64>,
    pub attribute: Option<String>,
}

// Bucket a timestamp into a coarse period index. The unit is one period
// boundary, e.g. day -> floor(ts/86_400_000).
fn period_bucket(ts_ms: i64, period: CohortPeriod) -> i64 {
    use crate::value::MS_PER_DAY;
    match period {
        CohortPeriod::Day => ts_ms.div_euclid(MS_PER_DAY),
        CohortPeriod::Week => ts_ms.div_euclid(MS_PER_DAY * 7),
        CohortPeriod::Month => {
            // Calendar month: convert to (y, m) and pack
            let days = ts_ms.div_euclid(MS_PER_DAY);
            let (y, m, _) = civil_from_days(days);
            (y as i64) * 12 + (m as i64 - 1)
        }
        CohortPeriod::Quarter => {
            let days = ts_ms.div_euclid(MS_PER_DAY);
            let (y, m, _) = civil_from_days(days);
            let q = (m - 1) / 3;
            (y as i64) * 4 + q as i64
        }
    }
}

// Same calendar arithmetic as period_compare. Re-declared here so this
// module is self-contained when only cohort code is touched.
fn civil_from_days(days: i64) -> (i32, u32, u32) {
    let z = days + 719_468;
    let era = if z >= 0 { z } else { z - 146_096 } / 146_097;
    let doe = (z - era * 146_097) as u64;
    let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146_096) / 365;
    let y = yoe as i64 + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    let y = if m <= 2 { y + 1 } else { y };
    (y as i32, m as u32, d as u32)
}

fn label_for_bucket(period: CohortPeriod, bucket: i64) -> String {
    match period {
        CohortPeriod::Day => {
            let (y, m, d) = civil_from_days(bucket);
            format!("{:04}-{:02}-{:02}", y, m, d)
        }
        CohortPeriod::Week => {
            let days = bucket * 7;
            let (y, m, d) = civil_from_days(days);
            format!("{:04}-W{:02}-{:02}", y, m, d)
        }
        CohortPeriod::Month => {
            let y = bucket.div_euclid(12) as i32;
            let m = bucket.rem_euclid(12) as u32 + 1;
            format!("{:04}-{:02}", y, m)
        }
        CohortPeriod::Quarter => {
            let y = bucket.div_euclid(4) as i32;
            let q = bucket.rem_euclid(4) + 1;
            format!("{:04}-Q{}", y, q)
        }
    }
}

// Per-user cached cohort assignment. Built lazily on the user's first
// event under the time-sorted contract, then reused for all later events.
struct CohortSlot {
    label: String,
    label_hash: u64,
    start_bucket: i64,
}

/// Streaming cohort analyser: ingest events one at a time without
/// requiring a materialised slice. Use directly from scan-driven
/// pipelines so peak memory is independent of source row count.
pub struct CohortAnalyser {
    analysis: CohortAnalysis,
    periods: u32,
    cohort_for_user: VerifiedKeyMap<AnalyticsValue, CohortSlot>,
    active_users: VerifiedKeyMap<(), VerifiedKeyMap<(), ()>>,
    numeric_sum: VerifiedKeyMap<(), f64>,
}

const COHORT_SEED_USER_LOW: u64 = 0xA1F0_42E5_88B3_CC91;
const COHORT_SEED_USER_HIGH: u64 = 0x6CD1_8492_50A7_F3B6;
const COHORT_SEED_LABEL: u64 = 0x4D8E_27A3_15F0_BB02;
const COHORT_SEED_LABEL_HIGH: u64 = 0x71A8_D532_6BCE_4F08;

#[inline]
fn user_hash_128(u: &AnalyticsValue) -> (u64, u64) {
    let (lo, hi) = hash_value_128(COHORT_SEED_USER_LOW, COHORT_SEED_USER_HIGH, u);
    (mix_finalize_2round(lo), mix_finalize_2round(hi))
}

impl CohortAnalyser {
    pub fn new(analysis: CohortAnalysis, periods: u32) -> Result<Self> {
        if periods == 0 {
            return Err(ZyronError::ExecutionError(
                "cohort periods must be >= 1".into(),
            ));
        }
        Ok(Self {
            analysis,
            periods,
            cohort_for_user: VerifiedKeyMap::new(),
            active_users: VerifiedKeyMap::new(),
            numeric_sum: VerifiedKeyMap::new(),
        })
    }

    pub fn ingest(&mut self, ev: &CohortEvent) {
        let def = &self.analysis.definition;
        let (uh_low, uh_high) = user_hash_128(&ev.user_id);
        let slot = self.cohort_for_user.entry_or_insert(
            uh_low,
            uh_high,
            || ev.user_id.clone(),
            || {
                let label = match &def.cohort_type {
                    CohortType::FirstEvent => {
                        let bucket = period_bucket(ev.event_time_ms, def.cohort_period);
                        label_for_bucket(def.cohort_period, bucket)
                    }
                    CohortType::Attribute => {
                        ev.attribute.clone().unwrap_or_else(|| "<none>".to_string())
                    }
                };
                let mut h = COHORT_SEED_LABEL;
                for &b in label.as_bytes() {
                    h = fx_mix(h, b as u64);
                }
                let label_hash = h;
                let start_bucket = period_bucket(ev.event_time_ms, def.analysis_period);
                CohortSlot {
                    label,
                    label_hash,
                    start_bucket,
                }
            },
        );
        let slot_label_hash = slot.label_hash;
        let slot_start_bucket = slot.start_bucket;

        let event_bucket = period_bucket(ev.event_time_ms, def.analysis_period);
        let offset = event_bucket - slot_start_bucket;
        if offset < 0 || offset >= self.periods as i64 {
            return;
        }
        let offset_u32 = offset as u32;
        let ckey_low = mix_finalize_2round(fx_mix(slot_label_hash, offset_u32 as u64));
        let ckey_high = mix_finalize_2round(fx_mix(
            slot_label_hash ^ COHORT_SEED_LABEL_HIGH,
            offset_u32 as u64,
        ));
        match self.analysis.metric {
            CohortMetric::ActiveUsers => {
                let bucket = self.active_users.entry_or_insert(
                    ckey_low,
                    ckey_high,
                    || (),
                    VerifiedKeyMap::new,
                );
                bucket.entry_or_insert(uh_low, uh_high, || (), || ());
            }
            CohortMetric::Revenue => {
                if let Some(r) = ev.revenue {
                    let bucket =
                        self.numeric_sum
                            .entry_or_insert(ckey_low, ckey_high, || (), || 0.0);
                    *bucket += r;
                }
            }
            CohortMetric::Custom(_) => {
                if let Some(c) = ev.custom_value {
                    let bucket =
                        self.numeric_sum
                            .entry_or_insert(ckey_low, ckey_high, || (), || 0.0);
                    *bucket += c;
                }
            }
        }
    }

    pub fn finalise(self) -> CohortResult {
        let analysis = self.analysis;
        let periods = self.periods;
        let cohort_for_user = self.cohort_for_user;
        let active_users = self.active_users;
        let numeric_sum = self.numeric_sum;

        // Assemble result rows in label order. Labels are read out of the
        // already-built cohort_for_user map.
        let mut labels: Vec<String> = cohort_for_user
            .iter()
            .map(|(_, slot)| slot.label.clone())
            .collect();
        labels.sort();
        labels.dedup();

        let mut rows = Vec::with_capacity(labels.len());
        for label in labels {
            let mut h = COHORT_SEED_LABEL;
            for &b in label.as_bytes() {
                h = fx_mix(h, b as u64);
            }
            let label_hash = h;
            let mut period_values = vec![0.0f64; periods as usize];
            for offset in 0..periods {
                let ckey_low = mix_finalize_2round(fx_mix(label_hash, offset as u64));
                let ckey_high =
                    mix_finalize_2round(fx_mix(label_hash ^ COHORT_SEED_LABEL_HIGH, offset as u64));
                let value = match analysis.metric {
                    CohortMetric::ActiveUsers => active_users
                        .get(ckey_low, ckey_high)
                        .map(|set| set.len() as f64)
                        .unwrap_or(0.0),
                    _ => numeric_sum.get(ckey_low, ckey_high).copied().unwrap_or(0.0),
                };
                period_values[offset as usize] = value;
            }
            rows.push(CohortRow {
                cohort_label: label,
                period_values,
            });
        }

        CohortResult {
            cohorts: rows,
            periods,
        }
    }
}

/// Slice convenience wrapper over `CohortAnalyser`. Use the analyser
/// directly if events arrive from a streaming source.
///
/// Input contract: events for any one user must arrive in non-decreasing
/// `event_time_ms` order. With the contract met, this function makes a
/// single pass over `events`.
///
/// Returns a (cohort x period) matrix of metric values. For ActiveUsers
/// the cell value is the unique user count active in that period offset;
/// for Revenue/Custom it is the sum of those values.
pub fn retention_analysis(
    events: &[CohortEvent],
    analysis: &CohortAnalysis,
    periods: u32,
) -> Result<CohortResult> {
    let mut analyser = CohortAnalyser::new(analysis.clone(), periods)?;
    for ev in events {
        analyser.ingest(ev);
    }
    Ok(analyser.finalise())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::value::MS_PER_DAY;

    fn user(s: &str) -> AnalyticsValue {
        AnalyticsValue::Text(s.to_string())
    }

    #[test]
    fn first_event_cohort_active_users() {
        let events = vec![
            CohortEvent {
                user_id: user("u1"),
                event_time_ms: 0,
                revenue: None,
                custom_value: None,
                attribute: None,
            },
            CohortEvent {
                user_id: user("u1"),
                event_time_ms: MS_PER_DAY * 5,
                revenue: None,
                custom_value: None,
                attribute: None,
            },
            CohortEvent {
                user_id: user("u2"),
                event_time_ms: 0,
                revenue: None,
                custom_value: None,
                attribute: None,
            },
        ];
        let def = CohortDefinition {
            user_id_column: "user_id".into(),
            event_time_column: "event_time".into(),
            cohort_period: CohortPeriod::Day,
            analysis_period: CohortPeriod::Day,
            cohort_type: CohortType::FirstEvent,
        };
        let result = retention_analysis(
            &events,
            &CohortAnalysis {
                definition: def,
                metric: CohortMetric::ActiveUsers,
            },
            7,
        )
        .unwrap();
        assert_eq!(result.cohorts.len(), 1);
        let cohort = &result.cohorts[0];
        assert_eq!(cohort.period_values[0], 2.0);
        assert_eq!(cohort.period_values[5], 1.0);
    }

    #[test]
    fn revenue_metric_aggregates_per_period() {
        let events = vec![
            CohortEvent {
                user_id: user("a"),
                event_time_ms: 0,
                revenue: Some(10.0),
                custom_value: None,
                attribute: None,
            },
            CohortEvent {
                user_id: user("a"),
                event_time_ms: MS_PER_DAY,
                revenue: Some(5.0),
                custom_value: None,
                attribute: None,
            },
        ];
        let def = CohortDefinition {
            user_id_column: "user_id".into(),
            event_time_column: "event_time".into(),
            cohort_period: CohortPeriod::Day,
            analysis_period: CohortPeriod::Day,
            cohort_type: CohortType::FirstEvent,
        };
        let r = retention_analysis(
            &events,
            &CohortAnalysis {
                definition: def,
                metric: CohortMetric::Revenue,
            },
            3,
        )
        .unwrap();
        assert_eq!(r.cohorts[0].period_values[0], 10.0);
        assert_eq!(r.cohorts[0].period_values[1], 5.0);
    }
}
