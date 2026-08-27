//! The classifier's inputs, and whether they can move at all.
//!
//! Three of the seven bottleneck kinds exist to stop the controller from
//! answering saturation with capacity that makes it worse. Every one of them
//! is gated on a signal, and a signal nothing produces is a branch that never
//! fires: the controller would behave exactly like the threshold scaler it
//! replaced, and nothing would ever say so.
//!
//! What is checked here is the signal arithmetic and the sketch. That the
//! engine actually calls the producers is checked where the producers live:
//! the lock table, the transaction manager, and the durability notifier.
//!
//! Run: cargo test -p zyron-pressure --test contention_signal_test

use zyron_pressure::capability::ContentionSignals;
use zyron_pressure::pressure::BottleneckKind;
use zyron_pressure::pressure_control::PressureController;

/// A rate needs both halves. Aborts with no commits is a node doing nothing
/// but failing, and commits with no aborts is a node with no contention.
#[test]
fn the_conflict_rate_is_aborts_over_outcomes() {
    let signals = ContentionSignals::new();
    assert_eq!(signals.occ_abort_rate(), 0.0, "no traffic is not conflict");

    for _ in 0..90 {
        signals.record_commit();
    }
    for _ in 0..10 {
        signals.record_conflict_abort();
    }
    assert!((signals.occ_abort_rate() - 0.10).abs() < 1e-9);

    signals.roll_window();
    assert_eq!(
        signals.occ_abort_rate(),
        0.0,
        "the rate survived its own window"
    );
}

/// Writers waiting is a level, not a rate. It counts up and down with the
/// writers, and it does not reset when the window does.
#[test]
fn writers_waiting_counts_up_and_down() {
    let signals = ContentionSignals::new();
    assert_eq!(signals.writes_waiting(), 0);

    signals.enter_write_wait();
    signals.enter_write_wait();
    assert_eq!(signals.writes_waiting(), 2);

    signals.roll_window();
    assert_eq!(
        signals.writes_waiting(),
        2,
        "a writer still parked stopped being counted at a window boundary"
    );

    signals.leave_write_wait();
    signals.leave_write_wait();
    assert_eq!(signals.writes_waiting(), 0);
}

/// A decrement that outran its increment must not wrap. Four billion writers
/// waiting would hold the node in the fsync classification forever, which
/// masks every rung that adds compute.
#[test]
fn the_writer_count_never_wraps_below_zero() {
    let signals = ContentionSignals::new();
    signals.leave_write_wait();
    signals.leave_write_wait();
    assert_eq!(signals.writes_waiting(), 0);
    signals.enter_write_wait();
    assert_eq!(signals.writes_waiting(), 1);
}

/// A commit that batched with others cost a share of a device round trip. One
/// that did not paid for a whole one, which is what fsync bound means.
#[test]
fn the_group_commit_rate_is_batched_over_all() {
    let signals = ContentionSignals::new();
    assert_eq!(
        signals.group_commit_hit_rate(),
        1.0,
        "a node that has not committed is not fsync bound"
    );

    for _ in 0..3 {
        signals.record_group_commit(true);
    }
    for _ in 0..7 {
        signals.record_group_commit(false);
    }
    assert!((signals.group_commit_hit_rate() - 0.30).abs() < 1e-9);
}

/// One extent taking most of the writes is visible in the share, and a spread
/// across many is not.
#[test]
fn a_hot_key_is_visible_and_an_even_spread_is_not() {
    let skewed = ContentionSignals::new();
    // Nine in ten writes to one row, which is a hot partition
    for i in 0..20_000u64 {
        skewed.record_key(if i % 10 == 0 { i } else { 1 });
    }
    assert!(
        skewed.hot_key_share() > 0.30,
        "a nine in ten skew read as {}",
        skewed.hot_key_share()
    );

    let even = ContentionSignals::new();
    for i in 0..20_000u64 {
        even.record_key(i);
    }
    assert!(
        even.hot_key_share() < 0.10,
        "twenty thousand distinct keys read as skewed at {}",
        even.hot_key_share()
    );
}

/// The sketch reports a lower bound, so a key it merely suspects is hot does
/// not mask provisioning that would have helped.
#[test]
fn the_sketch_never_over_reports_a_cold_key() {
    let signals = ContentionSignals::new();
    // Far more distinct keys than the sketch has slots, none of them hot
    for i in 0..200_000u64 {
        signals.record_key(i);
    }
    assert!(
        signals.hot_key_share() < 0.05,
        "a long tail of distinct keys was reported as a hot partition at {}",
        signals.hot_key_share()
    );
}

/// A skew that has gone away stops being reported, and one that has not keeps
/// being reported across a window boundary.
#[test]
fn the_sketch_follows_a_skew_that_moves() {
    // A hot extent among many, which is what skew means. One extent on its
    // own is not skew, it is a workload with nowhere to spread
    let hammer = |signals: &ContentionSignals| {
        for i in 0..20_000u64 {
            signals.record_key(if i % 4 == 0 { i } else { 7 });
        }
    };

    let signals = ContentionSignals::new();
    hammer(&signals);
    let while_hot = signals.hot_key_share();
    assert!(while_hot > 0.30, "{while_hot}");

    // The window ends. The extent is still hot, so it is still reported
    signals.roll_window();
    hammer(&signals);
    assert!(
        signals.hot_key_share() > 0.30,
        "an extent that stayed hot was forgotten at the window boundary"
    );

    // Traffic moves off it entirely and stays off
    for _ in 0..12 {
        signals.roll_window();
    }
    for i in 0..40_000u64 {
        signals.record_key(1_000_000 + i);
    }
    assert!(
        signals.hot_key_share() < 0.10,
        "a skew that ended was still reported at {}",
        signals.hot_key_share()
    );
}

/// Each signal on its own drives the classifier to its own answer, and the
/// order between them is the order that matters: a node that is both busy and
/// conflicted is conflicted.
#[test]
fn each_signal_reaches_its_own_classification() {
    let conflicted = PressureController::new(1, 64 * 1024 * 1024 * 1024);
    for _ in 0..80 {
        conflicted.contention().record_commit();
    }
    for _ in 0..20 {
        conflicted.contention().record_conflict_abort();
    }
    assert_eq!(
        conflicted.classify_bottleneck(),
        BottleneckKind::OccContention
    );

    let skewed = PressureController::new(2, 64 * 1024 * 1024 * 1024);
    for i in 0..20_000u64 {
        skewed
            .contention()
            .record_key(if i % 10 == 0 { i } else { 1 });
    }
    assert_eq!(skewed.classify_bottleneck(), BottleneckKind::HotPartition);

    let waiting = PressureController::new(3, 64 * 1024 * 1024 * 1024);
    waiting.contention().enter_write_wait();
    for _ in 0..10 {
        waiting.contention().record_group_commit(false);
    }
    assert_eq!(waiting.classify_bottleneck(), BottleneckKind::FsyncBound);

    let full = PressureController::new(4, 4 * 1024 * 1024 * 1024);
    let ceiling = full.memory().ceiling();
    assert!(full.memory().try_reserve(ceiling - ceiling / 20));
    assert_eq!(full.classify_bottleneck(), BottleneckKind::Memory);

    // Conflict outranks everything, because treating a conflicted node as
    // merely busy is what turns contention into collapse
    let both = PressureController::new(5, 4 * 1024 * 1024 * 1024);
    for _ in 0..80 {
        both.contention().record_commit();
    }
    for _ in 0..20 {
        both.contention().record_conflict_abort();
    }
    let ceiling = both.memory().ceiling();
    assert!(both.memory().try_reserve(ceiling - ceiling / 20));
    assert_eq!(both.classify_bottleneck(), BottleneckKind::OccContention);
}

/// The memory gauge measures the share query execution may take, not the
/// machine. Against the machine it would read as full headroom on a node
/// minutes from an out-of-memory kill.
#[test]
fn the_gauge_measures_a_share_of_the_machine() {
    let machine = 64u64 * 1024 * 1024 * 1024;
    let c = PressureController::new(1, machine);
    let ceiling = c.memory().ceiling();
    assert!(ceiling > 0);
    assert!(
        ceiling < machine / 2,
        "query execution was given {ceiling} of a {machine} byte machine"
    );
}

/// A workload writing one extent is not skewed, it is a workload with one
/// partition. Reporting skew there would mask provisioning permanently on
/// every small table, and provisioning is not what fails to help: there is
/// simply nothing to spread.
#[test]
fn one_partition_is_not_skew() {
    let signals = ContentionSignals::new();
    for _ in 0..20_000u64 {
        signals.record_key(7);
    }
    assert_eq!(
        signals.hot_key_share(),
        0.0,
        "a single extent was reported as a hot partition"
    );

    // Enough distinct extents for concentration to mean something, and the
    // same concentration is now reported
    let spread = ContentionSignals::new();
    for i in 0..20_000u64 {
        spread.record_key(if i % 8 == 0 { i } else { 7 });
    }
    assert!(
        spread.hot_key_share() > 0.30,
        "a hot extent among many read as {}",
        spread.hot_key_share()
    );
}

/// Idle cores are not evidence of slow storage on their own.
///
/// A node holding queries back at its own admission ceiling looks identical
/// from the parallel budget: permits free, work outstanding. The two want
/// opposite responses, so the classifier requires that the work is in flight
/// and that the node is actually reading pages.
#[test]
fn a_throttled_node_is_not_mistaken_for_a_slow_device() {
    use std::time::Duration;
    use zyron_pressure::pressure::{ParallelCapacity, WorkloadClass};

    // Queries queued behind the ceiling, none running, no reads. The parallel
    // budget is idle and there is work outstanding, which is exactly the shape
    // a slow device produces
    let capacity: &'static ParallelCapacity = Box::leak(Box::new(ParallelCapacity::with_total(8)));
    let throttled = PressureController::with_capacity(1, 64 * 1024 * 1024 * 1024, capacity);
    for _ in 0..64 {
        throttled.counters(WorkloadClass::Bulk).enqueue(5.0);
    }
    assert!(throttled.capacity().headroom_fraction() > 0.4);
    assert_ne!(
        throttled.classify_bottleneck(),
        BottleneckKind::Io,
        "a node throttling itself was diagnosed as storage bound"
    );

    // The same idle budget, but the work is running and pages are being read
    let reading_capacity: &'static ParallelCapacity =
        Box::leak(Box::new(ParallelCapacity::with_total(8)));
    let reading = PressureController::with_capacity(2, 64 * 1024 * 1024 * 1024, reading_capacity);
    for _ in 0..8 {
        reading.counters(WorkloadClass::Bulk).start_direct(5.0);
    }
    reading.record_page_read(Duration::from_micros(400), 16_384, 64);
    assert_eq!(reading.classify_bottleneck(), BottleneckKind::Io);
}

/// Running queries with idle cores and no reads at all is not storage either.
/// Whatever they are waiting on, more workers is not the answer.
#[test]
fn running_queries_that_read_nothing_are_not_storage_bound() {
    use zyron_pressure::pressure::{ParallelCapacity, WorkloadClass};

    let capacity: &'static ParallelCapacity = Box::leak(Box::new(ParallelCapacity::with_total(8)));
    let c = PressureController::with_capacity(3, 64 * 1024 * 1024 * 1024, capacity);
    for _ in 0..8 {
        c.counters(WorkloadClass::Bulk).start_direct(5.0);
    }
    assert_eq!(c.classify_bottleneck(), BottleneckKind::None);
}

/// The page read count is the whole traffic, not the sampled slice, or a node
/// reading steadily would look like one reading occasionally.
#[test]
fn page_reads_are_counted_in_whole_reads() {
    use std::time::Duration;
    use zyron_pressure::capability::PAGE_READ_SAMPLE_WEIGHT;

    let c = PressureController::new(1, 8 * 1024 * 1024 * 1024);
    assert_eq!(c.contention().page_reads(), 0);
    c.record_page_read(Duration::from_micros(80), 16_384, PAGE_READ_SAMPLE_WEIGHT);
    assert_eq!(c.contention().page_reads(), PAGE_READ_SAMPLE_WEIGHT);
    c.contention().roll_window();
    assert_eq!(
        c.contention().page_reads(),
        0,
        "the read count survived its own window"
    );
}
