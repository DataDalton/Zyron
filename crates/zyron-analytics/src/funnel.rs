// Funnel analysis. For each user, find the longest ordered prefix of the
// configured steps such that successive matches occur within the window.
// Reports per-step user count, conversion rate, drop-off, and avg time.

use crate::value::{AnalyticsValue, VerifiedKeyMap, hash_value_128};
use zyron_common::error::{Result, ZyronError};
use zyron_common::mix_finalize_2round;

#[derive(Debug, Clone)]
pub struct FunnelStep {
    pub name: String,
    // Caller decides which events satisfy the predicate, e.g. by tagging
    // events with the step name. The runtime here matches by name.
    pub event_match: String,
}

#[derive(Debug, Clone)]
pub struct FunnelConfig {
    pub steps: Vec<FunnelStep>,
    pub window_ms: i64,
    pub user_id_column: String,
    pub event_time_column: String,
}

#[derive(Debug, Clone)]
pub struct FunnelEvent {
    pub user_id: AnalyticsValue,
    pub event_time_ms: i64,
    pub event_name: String,
}

#[derive(Debug, Clone)]
pub struct StepResult {
    pub name: String,
    pub users_count: u64,
    pub conversion_rate: f64,
    pub drop_off_rate: f64,
    pub avg_time_to_next_ms: Option<f64>,
}

#[derive(Debug, Clone)]
pub struct FunnelResult {
    pub steps: Vec<StepResult>,
    pub overall_conversion: f64,
    pub total_users: u64,
}

/// Streaming funnel analyser: ingest events one at a time without
/// requiring a materialised slice. Use this directly in scan-driven
/// pipelines (executor heap scan) so peak memory is independent of the
/// source table's row count.
pub struct FunnelAnalyser {
    config: FunnelConfig,
    state: VerifiedKeyMap<(), UserFunnelState>,
    next_time_sum: Vec<i64>,
    next_time_n: Vec<u64>,
}

#[derive(Clone, Copy)]
struct UserFunnelState {
    anchor_time: i64,
    prev_step_time: i64,
    step: u8,
    frozen: bool,
}

impl FunnelAnalyser {
    pub fn new(config: FunnelConfig) -> Result<Self> {
        if config.steps.is_empty() {
            return Err(ZyronError::ExecutionError(
                "funnel must have at least one step".into(),
            ));
        }
        if config.window_ms <= 0 {
            return Err(ZyronError::ExecutionError(
                "funnel must have a positive window".into(),
            ));
        }
        let n_steps = config.steps.len();
        Ok(Self {
            config,
            state: VerifiedKeyMap::new(),
            next_time_sum: vec![0i64; n_steps],
            next_time_n: vec![0u64; n_steps],
        })
    }

    pub fn ingest(&mut self, ev: &FunnelEvent) {
        const SEED_LOW: u64 = 0x91A7_B432_558E_DD13;
        const SEED_HIGH: u64 = 0x4B83_A92F_C015_5E27;
        let (lo, hi) = hash_value_128(SEED_LOW, SEED_HIGH, &ev.user_id);
        let h_low = mix_finalize_2round(lo);
        let h_high = mix_finalize_2round(hi);
        let n_steps = self.config.steps.len();
        let s = self.state.entry_or_insert(
            h_low,
            h_high,
            || (),
            || UserFunnelState {
                anchor_time: 0,
                prev_step_time: 0,
                step: 0,
                frozen: false,
            },
        );
        if s.frozen {
            return;
        }
        if s.step == 0 {
            if ev.event_name == self.config.steps[0].event_match {
                s.anchor_time = ev.event_time_ms;
                s.prev_step_time = ev.event_time_ms;
                s.step = 1;
                if (s.step as usize) == n_steps {
                    s.frozen = true;
                }
            }
            return;
        }
        if ev.event_time_ms - s.anchor_time > self.config.window_ms {
            s.frozen = true;
            return;
        }
        if ev.event_name == self.config.steps[s.step as usize].event_match {
            self.next_time_sum[s.step as usize - 1] += ev.event_time_ms - s.prev_step_time;
            self.next_time_n[s.step as usize - 1] += 1;
            s.prev_step_time = ev.event_time_ms;
            s.step += 1;
            if (s.step as usize) == n_steps {
                s.frozen = true;
            }
        }
    }

    pub fn finalise(self) -> FunnelResult {
        let n_steps = self.config.steps.len();
        let total_users = self.state.len() as u64;
        let mut step_users = vec![0u64; n_steps];
        for (_, s) in self.state.iter() {
            for i in 0..(s.step as usize) {
                step_users[i] += 1;
            }
        }
        let entry = step_users.first().copied().unwrap_or(0).max(1) as f64;
        let mut steps_out = Vec::with_capacity(n_steps);
        for s in 0..n_steps {
            let users = step_users[s];
            let prev = if s == 0 { users } else { step_users[s - 1] };
            let prev_f = prev.max(1) as f64;
            let conversion = if s == 0 { 1.0 } else { users as f64 / prev_f };
            let drop_off = if s == 0 { 0.0 } else { 1.0 - conversion };
            let avg_time = if self.next_time_n[s] > 0 {
                Some(self.next_time_sum[s] as f64 / self.next_time_n[s] as f64)
            } else {
                None
            };
            steps_out.push(StepResult {
                name: self.config.steps[s].name.clone(),
                users_count: users,
                conversion_rate: conversion,
                drop_off_rate: drop_off,
                avg_time_to_next_ms: avg_time,
            });
        }
        let overall = step_users.last().copied().unwrap_or(0) as f64 / entry;
        FunnelResult {
            steps: steps_out,
            overall_conversion: overall,
            total_users,
        }
    }
}

/// Slice convenience wrapper over `FunnelAnalyser`. Use the analyser
/// directly if events arrive from a streaming source.
///
/// Contract: events for any one user must arrive in non-decreasing
/// event_time_ms order. Per-user out-of-order inputs are accepted but
/// produce best-effort results.
pub fn funnel_analysis(events: &[FunnelEvent], config: &FunnelConfig) -> Result<FunnelResult> {
    let mut analyser = FunnelAnalyser::new(config.clone())?;
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
    fn three_step_funnel() {
        let events = vec![
            FunnelEvent {
                user_id: user("a"),
                event_time_ms: 0,
                event_name: "view".into(),
            },
            FunnelEvent {
                user_id: user("a"),
                event_time_ms: 60_000,
                event_name: "cart".into(),
            },
            FunnelEvent {
                user_id: user("a"),
                event_time_ms: 120_000,
                event_name: "buy".into(),
            },
            FunnelEvent {
                user_id: user("b"),
                event_time_ms: 0,
                event_name: "view".into(),
            },
            FunnelEvent {
                user_id: user("b"),
                event_time_ms: 60_000,
                event_name: "cart".into(),
            },
            FunnelEvent {
                user_id: user("c"),
                event_time_ms: 0,
                event_name: "view".into(),
            },
        ];
        let cfg = FunnelConfig {
            steps: vec![
                FunnelStep {
                    name: "view".into(),
                    event_match: "view".into(),
                },
                FunnelStep {
                    name: "cart".into(),
                    event_match: "cart".into(),
                },
                FunnelStep {
                    name: "buy".into(),
                    event_match: "buy".into(),
                },
            ],
            window_ms: MS_PER_DAY,
            user_id_column: "user_id".into(),
            event_time_column: "event_time".into(),
        };
        let r = funnel_analysis(&events, &cfg).unwrap();
        assert_eq!(r.steps[0].users_count, 3);
        assert_eq!(r.steps[1].users_count, 2);
        assert_eq!(r.steps[2].users_count, 1);
        assert!((r.overall_conversion - 1.0 / 3.0).abs() < 1e-9);
        assert!(r.steps[1].avg_time_to_next_ms.is_some());
    }

    #[test]
    fn window_excludes_late_events() {
        let events = vec![
            FunnelEvent {
                user_id: user("a"),
                event_time_ms: 0,
                event_name: "view".into(),
            },
            FunnelEvent {
                user_id: user("a"),
                event_time_ms: MS_PER_DAY * 10,
                event_name: "buy".into(),
            },
        ];
        let cfg = FunnelConfig {
            steps: vec![
                FunnelStep {
                    name: "view".into(),
                    event_match: "view".into(),
                },
                FunnelStep {
                    name: "buy".into(),
                    event_match: "buy".into(),
                },
            ],
            window_ms: MS_PER_DAY,
            user_id_column: "u".into(),
            event_time_column: "t".into(),
        };
        let r = funnel_analysis(&events, &cfg).unwrap();
        assert_eq!(r.steps[0].users_count, 1);
        assert_eq!(r.steps[1].users_count, 0);
    }
}
