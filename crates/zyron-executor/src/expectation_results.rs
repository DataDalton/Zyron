//! Recent expectation evaluation outcomes, backing
//! `zyron_sys.expectation.results`.
//!
//! Every statement that evaluates a table's expectations records one
//! outcome per expectation here. The store keeps a bounded window of the
//! most recent evaluations so the view answers from live memory without
//! any storage cost on the write path beyond an append.

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Mutex, OnceLock};

/// One expectation evaluated over one statement's rows
#[derive(Debug, Clone)]
pub struct ExpectationOutcome {
    pub table_id: u32,
    pub expectation_name: String,
    pub evaluated_at_micros: i64,
    pub rows_checked: u64,
    pub violations: u64,
    pub passed: bool,
    pub action: String,
}

const WINDOW: usize = 1024;

struct Store {
    ring: Mutex<Vec<ExpectationOutcome>>,
    next: AtomicU64,
}

static STORE: OnceLock<Store> = OnceLock::new();

fn store() -> &'static Store {
    STORE.get_or_init(|| Store {
        ring: Mutex::new(Vec::with_capacity(WINDOW)),
        next: AtomicU64::new(0),
    })
}

/// Appends one outcome, evicting the oldest past the window
pub fn record(outcome: ExpectationOutcome) {
    let s = store();
    let slot = s.next.fetch_add(1, Ordering::Relaxed) as usize % WINDOW;
    let mut ring = match s.ring.lock() {
        Ok(guard) => guard,
        Err(poisoned) => poisoned.into_inner(),
    };
    if ring.len() < WINDOW {
        ring.push(outcome);
    } else {
        ring[slot] = outcome;
    }
}

/// The current window, newest first
pub fn snapshot() -> Vec<ExpectationOutcome> {
    let s = store();
    let ring = match s.ring.lock() {
        Ok(guard) => guard,
        Err(poisoned) => poisoned.into_inner(),
    };
    let mut out: Vec<ExpectationOutcome> = ring.clone();
    out.sort_by(|a, b| b.evaluated_at_micros.cmp(&a.evaluated_at_micros));
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_record_and_snapshot_orders_newest_first() {
        record(ExpectationOutcome {
            table_id: 900_001,
            expectation_name: "older".to_string(),
            evaluated_at_micros: 10,
            rows_checked: 5,
            violations: 0,
            passed: true,
            action: "warn".to_string(),
        });
        record(ExpectationOutcome {
            table_id: 900_001,
            expectation_name: "newer".to_string(),
            evaluated_at_micros: 20,
            rows_checked: 5,
            violations: 2,
            passed: false,
            action: "warn".to_string(),
        });
        let snap = snapshot();
        let ours: Vec<_> = snap.iter().filter(|o| o.table_id == 900_001).collect();
        assert!(ours.len() >= 2);
        assert_eq!(ours[0].expectation_name, "newer");
        assert!(!ours[0].passed);
    }
}
