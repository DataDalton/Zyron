//! A slow query must not be able to make a fast one slow.
//!
//! This is the property a single admission queue cannot have. One queue means
//! one ordering, and a point lookup behind a hundred five-second reports waits
//! for the reports. Every workaround for that at the application layer is a
//! second connection pool, which is an admission controller built by somebody
//! with less information.
//!
//! So the queue is per class, the class comes from what the query is priced at,
//! and the objective comes from the class. A report and a lookup are never in
//! the same queue, so the report cannot be in front of the lookup.
//!
//! Run: cargo test -p zyron-pressure --test class_isolation_test

use std::time::Instant;

use zyron_pressure::pressure::{AdmitDecision, WorkloadClass};
use zyron_pressure::pressure_control::PressureController;

/// A report: well past the interactive objective, so it is Bulk.
const REPORT_SECONDS: f64 = 5.0;

/// A lookup: inside the interactive objective and above the bypass threshold,
/// so it goes through admission as an Interactive query rather than skipping
/// the decision entirely.
const LOOKUP_SECONDS: f64 = 0.050;

/// Reports run at once in the scenario the spec names.
const REPORTS: usize = 100;

/// Lookups that arrive while those reports are running.
const LOOKUPS: usize = 10;

fn controller() -> PressureController {
    PressureController::new(1, 64 * 1024 * 1024 * 1024)
}

/// The pricing decides the class, and these two prices have to land on
/// different sides of the boundary or nothing below is testing isolation.
#[test]
fn a_report_and_a_lookup_are_classified_apart() {
    assert_eq!(
        WorkloadClass::classify(REPORT_SECONDS, false),
        WorkloadClass::Bulk
    );
    assert_eq!(
        WorkloadClass::classify(LOOKUP_SECONDS, false),
        WorkloadClass::Interactive
    );
    assert!(LOOKUP_SECONDS > WorkloadClass::BYPASS_WORK_SECONDS);
}

/// A hundred reports saturating the node do not delay a single lookup.
#[test]
fn a_saturated_bulk_class_does_not_delay_an_interactive_query() {
    let c = controller();

    let mut queued_reports = 0;
    for _ in 0..REPORTS {
        match c.admit(REPORT_SECONDS, false, None).1 {
            AdmitDecision::Delay(_) => queued_reports += 1,
            AdmitDecision::Shed { .. } => panic!("the node refused a report it should have held"),
            _ => {}
        }
    }
    assert!(
        queued_reports > 0,
        "the reports never filled the bulk class, so nothing is being isolated from"
    );
    assert!(
        c.counters(WorkloadClass::Bulk).queued_count() > 0,
        "the bulk queue is empty"
    );

    // Every lookup goes through admission with the bulk class full behind it
    let started = Instant::now();
    for i in 0..LOOKUPS {
        let (class, decision) = c.admit(LOOKUP_SECONDS, false, None);
        assert_eq!(class, WorkloadClass::Interactive);
        assert!(
            matches!(decision, AdmitDecision::Admit | AdmitDecision::Bypass),
            "lookup {i} was {decision:?} while only reports were queued"
        );
        c.complete(class, LOOKUP_SECONDS, LOOKUP_SECONDS, None);
    }
    let elapsed = started.elapsed();

    assert!(
        elapsed < std::time::Duration::from_millis(200),
        "{LOOKUPS} lookups took {elapsed:?} behind {REPORTS} reports"
    );
    assert_eq!(
        c.counters(WorkloadClass::Interactive).queued_count(),
        0,
        "an interactive query queued behind bulk work"
    );
    assert_eq!(c.counters(WorkloadClass::Interactive).shed_total(), 0);
}

/// The same load with the classes collapsed: every query priced as bulk, one
/// queue, and the lookups wait.
///
/// This is the revert proof. Without it the test above passes on a node that
/// has no isolation at all and simply is not busy.
#[test]
fn collapsing_the_classes_puts_the_lookups_behind_the_reports() {
    let c = controller();
    for _ in 0..REPORTS {
        c.admit(REPORT_SECONDS, false, None);
    }

    // The lookups now enter the same class as the reports, which is what a
    // single queue would do to them
    let mut delayed = 0;
    for _ in 0..LOOKUPS {
        // Priced as bulk work, which is how they would be treated if the
        // classifier did not separate them
        let (class, decision) = c.admit(REPORT_SECONDS, false, None);
        assert_eq!(class, WorkloadClass::Bulk);
        if matches!(decision, AdmitDecision::Delay(_)) {
            delayed += 1;
        }
    }
    assert_eq!(
        delayed, LOOKUPS,
        "collapsing the classes did not queue the lookups, so the isolation test above \
         would pass with the isolation removed"
    );
}

/// Each class has its own ceiling, and pressure in one is not pressure in
/// another. A shared ceiling is the other way isolation is lost.
#[test]
fn the_classes_carry_separate_ceilings_and_separate_pressure() {
    let c = controller();
    for _ in 0..REPORTS {
        c.admit(REPORT_SECONDS, false, None);
    }
    let bulk = c.counters(WorkloadClass::Bulk);
    let interactive = c.counters(WorkloadClass::Interactive);

    assert!(bulk.pressure_seconds() > 0.0);
    assert_eq!(
        interactive.pressure_seconds(),
        0.0,
        "bulk work registered as interactive pressure"
    );
    assert_eq!(interactive.in_flight(), 0);
    assert!(bulk.in_flight() > 0);
    // The objectives differ by orders of magnitude, which is the reason the
    // classes exist at all
    assert!(WorkloadClass::Bulk.slo_seconds() > WorkloadClass::Interactive.slo_seconds() * 100.0);
}

/// A query cheap enough that deciding about it costs more than running it
/// skips the decision, even while its class is refusing everything else.
///
/// This is what keeps a point lookup at point-lookup latency on a node under
/// pressure, and it is the reason admission overhead does not show up on the
/// hot path.
#[test]
fn a_query_below_the_bypass_threshold_runs_while_the_node_is_full() {
    let c = controller();
    // Fill the interactive class past its ceiling and its queue
    let depth = WorkloadClass::Interactive.queue_depth();
    for _ in 0..depth + 64 {
        c.admit(LOOKUP_SECONDS, false, None);
    }
    assert!(
        c.counters(WorkloadClass::Interactive).shed_total() > 0,
        "the class never reached the point of refusing work"
    );

    let tiny = WorkloadClass::BYPASS_WORK_SECONDS / 2.0;
    let (class, decision) = c.admit(tiny, false, None);
    assert_eq!(class, WorkloadClass::Interactive);
    assert!(
        matches!(decision, AdmitDecision::Bypass),
        "a query below the bypass threshold was {decision:?} on a shedding node"
    );
}

/// Background work has no objective, so it never breaches and never triggers a
/// response, however much of it there is.
#[test]
fn background_work_never_provokes_the_controller() {
    let c = controller();
    for _ in 0..500 {
        let (class, _) = c.admit(120.0, true, None);
        assert_eq!(class, WorkloadClass::Background);
    }
    let background = c.counters(WorkloadClass::Background);
    assert!(background.pressure_seconds() > 0.0);
    assert!(
        !background.snapshot(WorkloadClass::Background).breaching(),
        "background work reported a breach against an objective it does not have"
    );
}
