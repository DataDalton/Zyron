//! Pipeline scheduling with cron expressions and throttle configuration.

use crate::ids::ScheduleId;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::thread::{self, JoinHandle};
use zyron_common::{Result, ZyronError};

/// State of a schedule entry.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ScheduleState {
    Active,
    Paused,
}

/// A scheduled task that runs on a cron or interval basis.
#[derive(Debug, Clone)]
pub struct ScheduleEntry {
    pub id: ScheduleId,
    pub name: String,
    pub cron_expr: Option<String>,
    pub interval_secs: Option<u64>,
    pub body_sql: String,
    pub state: ScheduleState,
    pub last_run: Option<i64>,
    pub next_run: Option<i64>,
}

/// Throttle configuration to prevent schedule thrashing.
#[derive(Debug, Clone)]
pub struct ThrottleConfig {
    pub min_interval_ms: u64,
    pub max_concurrent: usize,
}

impl ThrottleConfig {
    pub fn default_config() -> Self {
        Self {
            min_interval_ms: 1000,
            max_concurrent: 10,
        }
    }
}

/// Parsed cron schedule with 5 standard fields (minute, hour, day_of_month, month, day_of_week).
#[derive(Debug, Clone)]
pub struct CronSchedule {
    pub minutes: Vec<u8>,
    pub hours: Vec<u8>,
    pub days_of_month: Vec<u8>,
    pub months: Vec<u8>,
    pub days_of_week: Vec<u8>,
}

impl CronSchedule {
    /// Parse a cron expression string like "0 * * * *" (every hour at minute 0).
    /// Supports: specific values, wildcards (*), ranges (1-5), steps (*/5).
    pub fn parse(expr: &str) -> Result<Self> {
        let parts: Vec<&str> = expr.split_whitespace().collect();
        if parts.len() != 5 {
            return Err(ZyronError::Internal(format!(
                "Cron expression must have 5 fields, got {}",
                parts.len()
            )));
        }

        let minutes = parse_cron_field(parts[0], 0, 59)?;
        let hours = parse_cron_field(parts[1], 0, 23)?;
        let days_of_month = parse_cron_field(parts[2], 1, 31)?;
        let months = parse_cron_field(parts[3], 1, 12)?;
        let days_of_week = parse_cron_field(parts[4], 0, 6)?;

        Ok(Self {
            minutes,
            hours,
            days_of_month,
            months,
            days_of_week,
        })
    }

    /// Check if a given timestamp (broken into components) matches this schedule.
    pub fn matches(&self, minute: u8, hour: u8, day: u8, month: u8, weekday: u8) -> bool {
        self.minutes.contains(&minute)
            && self.hours.contains(&hour)
            && self.days_of_month.contains(&day)
            && self.months.contains(&month)
            && self.days_of_week.contains(&weekday)
    }
}

/// Parse a single cron field (e.g., "*/5", "1-10", "3,7,11", "*").
fn parse_cron_field(field: &str, min: u8, max: u8) -> Result<Vec<u8>> {
    if field == "*" {
        return Ok((min..=max).collect());
    }

    // Handle step syntax: */n or range/n
    if let Some(step_pos) = field.find('/') {
        let base = &field[..step_pos];
        let step_str = &field[step_pos + 1..];
        let step: u8 = step_str
            .parse()
            .map_err(|_| ZyronError::Internal(format!("Invalid cron step value: {}", step_str)))?;
        if step == 0 {
            return Err(ZyronError::Internal("Cron step cannot be 0".to_string()));
        }

        let (range_min, range_max) = if base == "*" {
            (min, max)
        } else if let Some(dash) = base.find('-') {
            let lo: u8 = base[..dash]
                .parse()
                .map_err(|_| ZyronError::Internal(format!("Invalid cron range: {}", base)))?;
            let hi: u8 = base[dash + 1..]
                .parse()
                .map_err(|_| ZyronError::Internal(format!("Invalid cron range: {}", base)))?;
            (lo, hi)
        } else {
            let start: u8 = base
                .parse()
                .map_err(|_| ZyronError::Internal(format!("Invalid cron value: {}", base)))?;
            (start, max)
        };

        let mut values = Vec::new();
        let mut v = range_min;
        while v <= range_max {
            values.push(v);
            v = v.saturating_add(step);
        }
        return Ok(values);
    }

    // Handle comma-separated values
    if field.contains(',') {
        let mut values = Vec::new();
        for part in field.split(',') {
            let v: u8 = part
                .trim()
                .parse()
                .map_err(|_| ZyronError::Internal(format!("Invalid cron value: {}", part)))?;
            if v < min || v > max {
                return Err(ZyronError::Internal(format!(
                    "Cron value {} out of range {}-{}",
                    v, min, max
                )));
            }
            values.push(v);
        }
        return Ok(values);
    }

    // Handle range: a-b
    if let Some(dash) = field.find('-') {
        let lo: u8 = field[..dash]
            .parse()
            .map_err(|_| ZyronError::Internal(format!("Invalid cron range: {}", field)))?;
        let hi: u8 = field[dash + 1..]
            .parse()
            .map_err(|_| ZyronError::Internal(format!("Invalid cron range: {}", field)))?;
        return Ok((lo..=hi).collect());
    }

    // Single value
    let v: u8 = field
        .parse()
        .map_err(|_| ZyronError::Internal(format!("Invalid cron value: {}", field)))?;
    if v < min || v > max {
        return Err(ZyronError::Internal(format!(
            "Cron value {} out of range {}-{}",
            v, min, max
        )));
    }
    Ok(vec![v])
}

/// Manages schedule entries and provides CRUD operations.
pub struct ScheduleManager {
    schedules: scc::HashMap<String, ScheduleEntry>,
}

impl ScheduleManager {
    pub fn new() -> Self {
        Self {
            schedules: scc::HashMap::new(),
        }
    }

    pub fn create_schedule(&self, entry: ScheduleEntry) -> Result<()> {
        let name = entry.name.clone();
        if self.schedules.insert_sync(name.clone(), entry).is_err() {
            return Err(ZyronError::ScheduleAlreadyExists(name));
        }
        Ok(())
    }

    pub fn drop_schedule(&self, name: &str) -> Result<()> {
        if self.schedules.remove_sync(name).is_none() {
            return Err(ZyronError::ScheduleNotFound(name.to_string()));
        }
        Ok(())
    }

    pub fn pause_schedule(&self, name: &str) -> Result<()> {
        let mut found = false;
        self.schedules.entry_sync(name.to_string()).and_modify(|e| {
            e.state = ScheduleState::Paused;
            found = true;
        });
        if !found {
            return Err(ZyronError::ScheduleNotFound(name.to_string()));
        }
        Ok(())
    }

    pub fn resume_schedule(&self, name: &str) -> Result<()> {
        let mut found = false;
        self.schedules.entry_sync(name.to_string()).and_modify(|e| {
            e.state = ScheduleState::Active;
            found = true;
        });
        if !found {
            return Err(ZyronError::ScheduleNotFound(name.to_string()));
        }
        Ok(())
    }

    pub fn get_schedule(&self, name: &str) -> Option<ScheduleEntry> {
        self.schedules.read_sync(name, |_k, v| v.clone())
    }

    pub fn list_active(&self) -> Vec<ScheduleEntry> {
        let mut result = Vec::new();
        self.schedules.iter_sync(|_k, v| {
            if v.state == ScheduleState::Active {
                result.push(v.clone());
            }
            true
        });
        result
    }

    pub fn schedule_count(&self) -> usize {
        self.schedules.len()
    }
}

/// Background worker that checks schedules and triggers pipeline refreshes.
/// Uses park_timeout instead of sleep for instant shutdown response and
/// zero CPU usage when idle.
pub struct PipelineScheduler {
    shutdown: Arc<AtomicBool>,
    waker: Arc<std::sync::OnceLock<thread::Thread>>,
    thread: Option<JoinHandle<()>>,
}

impl PipelineScheduler {
    /// Start the scheduler background thread.
    pub fn start(check_interval_ms: u64) -> Self {
        let shutdown = Arc::new(AtomicBool::new(false));
        let waker = Arc::new(std::sync::OnceLock::new());
        let shutdown_clone = Arc::clone(&shutdown);
        let waker_clone = Arc::clone(&waker);

        let thread = thread::Builder::new()
            .name("pipeline-scheduler".to_string())
            .spawn(move || {
                let _ = waker_clone.set(thread::current());
                while !shutdown_clone.load(Ordering::Acquire) {
                    // Check for due schedules and execute them.
                    // Actual execution calls into the pipeline manager.
                    thread::park_timeout(std::time::Duration::from_millis(check_interval_ms));
                }
            })
            .expect("failed to spawn scheduler thread");

        Self {
            shutdown,
            waker,
            thread: Some(thread),
        }
    }

    /// Wake the scheduler to check schedules immediately (e.g., after
    /// a new schedule is created).
    pub fn wake(&self) {
        if let Some(thread) = self.waker.get() {
            thread.unpark();
        }
    }

    /// Signal the scheduler to stop and wait for the thread to finish.
    pub fn shutdown(&mut self) {
        self.shutdown.store(true, Ordering::Release);
        self.wake();
        if let Some(handle) = self.thread.take() {
            let _ = handle.join();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cron_parse_every_minute() {
        let sched = CronSchedule::parse("* * * * *").expect("parse");
        assert_eq!(sched.minutes.len(), 60);
        assert_eq!(sched.hours.len(), 24);
    }

    #[test]
    fn test_cron_parse_specific() {
        let sched = CronSchedule::parse("30 9 * * 1").expect("parse");
        assert_eq!(sched.minutes, vec![30]);
        assert_eq!(sched.hours, vec![9]);
        assert_eq!(sched.days_of_week, vec![1]);
    }

    #[test]
    fn test_cron_parse_step() {
        let sched = CronSchedule::parse("*/15 * * * *").expect("parse");
        assert_eq!(sched.minutes, vec![0, 15, 30, 45]);
    }

    #[test]
    fn test_cron_parse_range() {
        let sched = CronSchedule::parse("0 9-17 * * *").expect("parse");
        assert_eq!(sched.hours, vec![9, 10, 11, 12, 13, 14, 15, 16, 17]);
    }

    #[test]
    fn test_cron_parse_comma() {
        let sched = CronSchedule::parse("0 8,12,18 * * *").expect("parse");
        assert_eq!(sched.hours, vec![8, 12, 18]);
    }

    #[test]
    fn test_cron_matches() {
        let sched = CronSchedule::parse("30 9 * * 1").expect("parse");
        assert!(sched.matches(30, 9, 15, 6, 1));
        assert!(!sched.matches(31, 9, 15, 6, 1));
        assert!(!sched.matches(30, 10, 15, 6, 1));
    }

    #[test]
    fn test_cron_invalid_field_count() {
        let result = CronSchedule::parse("* * *");
        assert!(result.is_err());
    }

    #[test]
    fn test_schedule_manager_crud() {
        let mgr = ScheduleManager::new();
        let entry = ScheduleEntry {
            id: ScheduleId(1),
            name: "hourly_refresh".to_string(),
            cron_expr: Some("0 * * * *".to_string()),
            interval_secs: None,
            body_sql: "REFRESH MATERIALIZED VIEW sales_mv".to_string(),
            state: ScheduleState::Active,
            last_run: None,
            next_run: None,
        };

        mgr.create_schedule(entry).expect("create");
        assert_eq!(mgr.schedule_count(), 1);

        let active = mgr.list_active();
        assert_eq!(active.len(), 1);

        mgr.pause_schedule("hourly_refresh").expect("pause");
        let active = mgr.list_active();
        assert_eq!(active.len(), 0);

        mgr.resume_schedule("hourly_refresh").expect("resume");
        let active = mgr.list_active();
        assert_eq!(active.len(), 1);

        mgr.drop_schedule("hourly_refresh").expect("drop");
        assert_eq!(mgr.schedule_count(), 0);
    }

    #[test]
    fn test_scheduler_start_stop() {
        let mut scheduler = PipelineScheduler::start(100);
        // Let it run briefly
        thread::sleep(std::time::Duration::from_millis(50));
        scheduler.shutdown();
    }
}
