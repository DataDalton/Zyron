//! Two memory counters that must not be confused for each other.
//!
//! A query's budget answers "has this one query asked for more than it is
//! allowed". It never gives anything back, and that is correct: the question
//! is about the whole query, so a reservation that was released partway
//! through would let a query that peaked over its limit report that it stayed
//! under. The budget dies with the query, which is when the answer stops
//! mattering.
//!
//! The node's gauge answers a different question: "is this machine about to
//! run out of memory". That one is about right now, so a query that finished
//! has to give its bytes back or the node would believe it was full of work
//! that ended hours ago and would refuse everything.
//!
//! Both behaviours are correct and they are opposites, which is exactly why
//! this is worth a test rather than a comment. Making the budget release, or
//! making the gauge not, are both one-line changes that look like fixes.
//!
//! Run: cargo test -p zyron-executor --test memory_accounting_test

use zyron_executor::QueryMemoryBudget;
use zyron_pressure::pressure::{MemoryReservation, NodeMemoryGauge};

/// The query budget accumulates and never gives anything back.
#[test]
fn the_query_budget_never_releases() {
    let budget = QueryMemoryBudget::new(1_000);
    budget.reserve(400).expect("fits");
    assert_eq!(budget.used(), 400);
    budget.reserve(400).expect("fits");
    assert_eq!(budget.used(), 800);

    // There is no release, by design: a query that peaked over its limit must
    // not be able to report that it stayed under
    budget.reserve(400).expect_err("over the limit");
    assert_eq!(
        budget.used(),
        800,
        "a refused reservation must not be charged"
    );

    // And nothing about finishing with the budget hands anything back
    drop(budget);
}

/// The node gauge gives bytes back, because it describes the machine now.
#[test]
fn the_node_gauge_releases_what_a_finished_query_held() {
    let gauge = NodeMemoryGauge::new(1_000);
    assert!(gauge.try_reserve(600));
    assert_eq!(gauge.reserved(), 600);

    assert!(!gauge.try_reserve(500), "the gauge overcommitted");
    assert_eq!(gauge.reserved(), 600, "a refused reservation took bytes");
    assert_eq!(gauge.rejections(), 1);

    gauge.release(600);
    assert_eq!(
        gauge.reserved(),
        0,
        "the gauge kept a finished query's bytes"
    );
    assert_eq!(gauge.peak(), 600, "the peak is still recorded");

    // The node is usable again, which is the whole difference from the budget
    assert!(gauge.try_reserve(900));
}

/// A reservation that goes out of scope gives its bytes back, so a query that
/// errors or is cancelled does not leak node memory.
#[test]
fn a_dropped_reservation_returns_its_bytes() {
    let gauge = NodeMemoryGauge::new(1_000);
    {
        let held = MemoryReservation::acquire(&gauge, 700).expect("fits");
        assert_eq!(held.bytes(), 700);
        assert_eq!(gauge.reserved(), 700);
        assert!(
            MemoryReservation::acquire(&gauge, 500).is_none(),
            "the gauge handed out bytes it did not have"
        );
    }
    assert_eq!(gauge.reserved(), 0);
    assert!(MemoryReservation::acquire(&gauge, 900).is_some());
}

/// The two counters are independent. Charging one must not move the other, or
/// a node would refuse work because of a query's own accounting, or a query
/// would pass its limit because the node happened to be quiet.
#[test]
fn the_two_counters_do_not_see_each_other() {
    let gauge = NodeMemoryGauge::new(10_000);
    let budget = QueryMemoryBudget::new(1_000);

    budget.reserve(500).expect("fits");
    assert_eq!(gauge.reserved(), 0, "a query budget moved the node gauge");

    assert!(gauge.try_reserve(5_000));
    assert_eq!(budget.used(), 500, "the node gauge moved a query budget");

    gauge.release(5_000);
    assert_eq!(budget.used(), 500);
    assert_eq!(gauge.reserved(), 0);
}

/// Running the same query shape repeatedly leaves the node where it started,
/// which is what a releasing gauge means in practice and what a non-releasing
/// one would break within an hour of serving.
#[test]
fn repeated_queries_leave_the_node_gauge_where_it_started() {
    let gauge = NodeMemoryGauge::new(4_096);
    for _ in 0..1_000 {
        let budget = QueryMemoryBudget::new(2_048);
        let held = MemoryReservation::acquire(&gauge, 1_024).expect("fits");
        budget.reserve(1_024).expect("fits");
        assert_eq!(gauge.reserved(), 1_024);
        drop(held);
    }
    assert_eq!(gauge.reserved(), 0);
    assert_eq!(gauge.peak(), 1_024);
    assert_eq!(gauge.rejections(), 0);
}
