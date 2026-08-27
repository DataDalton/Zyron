//! Scaling on where the load is going, not where it has been.
//!
//! A node that asks for hardware when it breaches spends the whole provision
//! latency breaching. On a cloud that is ninety seconds of missed objective
//! that arrives too late to prevent, every time. The only scale-out that is on
//! time is one triggered by the trend.
//!
//! So the controller fits the arrival rate over the last minute, runs the
//! queue forward by exactly one provision latency, and sizes the warm pool
//! against the demand growth that lands inside that window: the part of the
//! burst that provisioning cannot answer, because it arrives before the
//! hardware does.
//!
//! Run: cargo test -p zyron-pressure --test projection_horizon_test

use std::time::{Duration, Instant};

use zyron_pressure::pressure::WorkloadClass;
use zyron_pressure::pressure_control::PressureController;

/// The controller's own tick period, which is one measurement window.
const WINDOW: Duration = Duration::from_millis(100);

/// A minute of windows, which is the span the trend is fitted over.
const WINDOWS: u32 = 600;

/// Priced below the bypass threshold so every arrival is admitted without
/// queueing. The subject here is the arrival trend, and a queue forming would
/// measure the admission control instead.
const TINY_WORK_SECONDS: f64 = 0.0005;

/// Drives the controller through a minute of traffic whose arrival rate rises
/// linearly, and returns the controller holding that history.
fn ramp_traffic(base_qps: f64, slope_qps_per_s: f64) -> PressureController {
    let controller = PressureController::new(1, 64 * 1024 * 1024 * 1024);
    let start = Instant::now();
    for window in 0..WINDOWS {
        let elapsed = window as f64 * WINDOW.as_secs_f64();
        let rate = base_qps + slope_qps_per_s * elapsed;
        let arrivals = (rate * WINDOW.as_secs_f64()).round() as u32;
        for _ in 0..arrivals {
            let (class, _) = controller.admit(TINY_WORK_SECONDS, false, None);
            controller.complete(class, TINY_WORK_SECONDS, TINY_WORK_SECONDS, None);
        }
        controller.tick(start + WINDOW * (window + 1));
    }
    controller
}

/// The warm pool covers exactly the growth that arrives while a node is being
/// created, because that is the demand provisioning is too slow to answer.
#[test]
fn the_warm_pool_is_one_provision_latency_of_growth() {
    let slope = 20.0;
    let horizon = Duration::from_secs(30);
    let controller = ramp_traffic(500.0, slope);

    let trend = controller.arrival_trend(
        WorkloadClass::Interactive,
        zyron_pressure::projection::TREND_WINDOW,
    );
    assert!(
        trend.trustworthy(),
        "a minute of windows produced no usable trend: {trend:?}"
    );
    assert!(
        (trend.derivative_qps_per_s - slope).abs() / slope < 0.15,
        "fitted slope {} against an applied {slope}",
        trend.derivative_qps_per_s
    );

    let projection = controller.project(WorkloadClass::Interactive, horizon, 64);
    let expected = slope * horizon.as_secs_f64();
    let error = (projection.projected_arrival_rate_delta - expected).abs() / expected;
    assert!(
        error < 0.15,
        "warm pool sized against {} where one provision latency of growth is {expected}: \
         {projection:?}",
        projection.projected_arrival_rate_delta
    );
    assert!(
        projection.warm_pool_nodes >= 1,
        "a rising load kept no warm capacity: {projection:?}"
    );
    assert_eq!(projection.horizon, horizon);
}

/// A longer provision latency means more of the burst lands before the
/// hardware does, so more of it has to already be running.
#[test]
fn a_slower_provisioner_needs_a_larger_warm_pool() {
    let controller = ramp_traffic(500.0, 20.0);
    let quick = controller.project(WorkloadClass::Interactive, Duration::from_secs(15), 256);
    let slow = controller.project(WorkloadClass::Interactive, Duration::from_secs(120), 256);

    assert!(
        slow.projected_arrival_rate_delta > quick.projected_arrival_rate_delta * 3.0,
        "eight times the latency did not need more warm capacity: {} against {}",
        slow.projected_arrival_rate_delta,
        quick.projected_arrival_rate_delta
    );
    assert!(slow.warm_pool_nodes > quick.warm_pool_nodes);
}

/// A flat load keeps nothing warm, whatever the operator would have paid for.
#[test]
fn a_steady_load_keeps_no_warm_pool() {
    let controller = ramp_traffic(800.0, 0.0);
    let projection = controller.project(WorkloadClass::Interactive, Duration::from_secs(90), 128);
    assert_eq!(
        projection.warm_pool_nodes, 0,
        "a flat load paid for idle capacity: {projection:?}"
    );
    assert!(
        projection.arrival_rate_derivative.abs() < 1.0,
        "a flat load fitted a slope of {}",
        projection.arrival_rate_derivative
    );
}

/// The projection leads the present reading. A node whose load is climbing
/// sees the breach before it happens, which is the entire reason the
/// projection exists.
#[test]
fn the_projection_sees_a_breach_the_present_reading_does_not() {
    // Rising hard, and starting from a rate the node is currently keeping up
    // with, so nothing is wrong yet
    let controller = ramp_traffic(200.0, 60.0);
    let now = controller
        .counters(WorkloadClass::Interactive)
        .pressure_seconds();
    assert!(
        now < WorkloadClass::Interactive.slo_seconds(),
        "the node is already breaching, so the test proves nothing about leading"
    );
    let projection = controller.project(WorkloadClass::Interactive, Duration::from_secs(90), 64);
    assert!(
        projection.projected_pressure_seconds > projection.current_pressure_seconds,
        "the projection did not lead the present: {projection:?}"
    );
    assert!(
        projection.arrival_rate_derivative > 0.0,
        "a load climbing at 60 a second fitted a flat trend: {projection:?}"
    );
}

/// The configured value bounds what the projection asks for and never raises
/// it, so an operator setting a warm pool is setting a spend limit rather than
/// a target to fill.
#[test]
fn the_configured_warm_pool_only_ever_bounds_the_answer() {
    // A horizon several times the trend window, so the growth that lands
    // before capacity arrives is several nodes worth rather than one
    let controller = ramp_traffic(200.0, 200.0);
    let horizon = Duration::from_secs(300);
    let unbounded = controller.project(WorkloadClass::Interactive, horizon, 4096);
    let bounded = controller.project(WorkloadClass::Interactive, horizon, 2);

    assert_eq!(bounded.warm_pool_nodes, 2);
    assert_eq!(bounded.warm_pool_uncapped, unbounded.warm_pool_uncapped);
    assert!(unbounded.warm_pool_nodes > 2, "{unbounded:?}");
}

/// A node that has just started has no trend and says so, rather than fitting
/// a line through two points and provisioning against it.
#[test]
fn a_node_with_no_history_projects_nothing() {
    let controller = PressureController::new(1, 8 * 1024 * 1024 * 1024);
    let projection = controller.project(WorkloadClass::Interactive, Duration::from_secs(90), 32);
    assert!(!projection.trend_trustworthy);
    assert_eq!(projection.warm_pool_nodes, 0);
    assert_eq!(projection.projected_arrival_rate_delta, 0.0);
}

/// The published horizon and cap are what every reader projects against, so
/// the view, the endpoint, and the ladder cannot disagree about the same node.
#[test]
fn the_published_horizon_is_what_the_shared_projection_uses() {
    let controller = ramp_traffic(400.0, 25.0);
    controller.set_provision_horizon(Duration::from_secs(45));
    controller.set_warm_pool_cap(7);

    let shared = controller.projection(WorkloadClass::Interactive);
    let explicit = controller.project(WorkloadClass::Interactive, Duration::from_secs(45), 7);
    assert_eq!(shared, explicit);
    assert_eq!(shared.warm_pool_cap, 7);
    assert_eq!(shared.horizon, Duration::from_secs(45));
}
