//! Where pressure will be by the time new capacity could arrive.
//!
//! Reacting to pressure is always late. A node that asks for hardware the
//! moment it breaches its objective spends the whole provision latency
//! breaching, and on a cloud that is a minute and a half of missed objective
//! that no amount of local relief recovers. The only way a scale-out is on
//! time is if it is triggered by where the load is going rather than where it
//! is.
//!
//! So the trigger is the projection: fit the arrival rate and its slope over
//! the recent past, run the queue forward by exactly one provision latency,
//! and compare that against the objective. A rising load that has not
//! breached yet still asks for capacity, and a spike that is already
//! flattening does not.
//!
//! The warm pool falls out of the same arithmetic. Capacity that has to be
//! created cannot answer a burst that arrives faster than it can be created,
//! so the part of the demand growth that lands inside one provision latency is
//! the part that has to already be running. That is a derived number, not a
//! configured one: the configured value is a ceiling on what an operator will
//! pay to keep idle, never a target to fill.

use std::time::Duration;

use crate::pressure::WorkloadClass;

/// How far back the fit looks.
///
/// Long enough that one slow window does not turn into a trend, short enough
/// that a genuine ramp is visible while it is still a ramp. A minute of
/// hundred-millisecond windows is six hundred points, which is far more than
/// the fit needs and costs nothing to walk.
pub const TREND_WINDOW: Duration = Duration::from_secs(60);

/// Fewest windows that can produce a slope worth acting on.
///
/// Two points always fit a line perfectly, which is exactly the situation
/// where the line means nothing. A node that has just started reports no
/// trend rather than an imaginary one.
pub const MIN_TREND_SAMPLES: usize = 8;

/// One observation of how fast work arrived.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ArrivalSample {
    /// Seconds before now. Positive, and larger means older
    pub age_seconds: f64,
    /// Arrivals per second over the window this sample closed
    pub rate_qps: f64,
}

/// The fitted arrival behaviour.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct ArrivalTrend {
    /// Fitted rate at this instant, which is steadier than the last window's
    /// raw reading
    pub rate_qps: f64,
    /// Change in arrival rate per second. Positive means the load is growing
    pub derivative_qps_per_s: f64,
    /// Seconds the fit actually spanned
    pub observed_seconds: f64,
    pub samples: usize,
}

impl ArrivalTrend {
    /// Whether the fit rests on enough windows to act on.
    pub fn trustworthy(&self) -> bool {
        self.samples >= MIN_TREND_SAMPLES && self.observed_seconds > 0.0
    }

    /// Arrival rate expected after `horizon`, floored at zero because a
    /// falling trend runs out of load rather than going negative.
    pub fn rate_at(&self, horizon: Duration) -> f64 {
        (self.rate_qps + self.derivative_qps_per_s * horizon.as_secs_f64()).max(0.0)
    }
}

/// Fits rate against time by least squares.
///
/// Least squares rather than a difference of endpoints, because the endpoints
/// of a hundred-millisecond window are two of the noisiest numbers the node
/// produces, and a scale-out decision taken from their difference would be a
/// decision taken from noise.
pub fn fit_arrival_trend(samples: &[ArrivalSample]) -> ArrivalTrend {
    let n = samples.len();
    if n == 0 {
        return ArrivalTrend::default();
    }
    // Time runs forward for the fit, so a positive slope means growth
    let mut sum_t = 0.0;
    let mut sum_r = 0.0;
    let mut oldest = 0.0f64;
    for s in samples {
        let t = -s.age_seconds;
        sum_t += t;
        sum_r += s.rate_qps;
        oldest = oldest.max(s.age_seconds);
    }
    let mean_t = sum_t / n as f64;
    let mean_r = sum_r / n as f64;

    let mut covariance = 0.0;
    let mut variance = 0.0;
    for s in samples {
        let dt = -s.age_seconds - mean_t;
        covariance += dt * (s.rate_qps - mean_r);
        variance += dt * dt;
    }

    let slope = if variance > 0.0 {
        covariance / variance
    } else {
        0.0
    };
    // Evaluate the fitted line at t = 0, which is now
    let rate_now = mean_r + slope * (0.0 - mean_t);

    ArrivalTrend {
        rate_qps: rate_now.max(0.0),
        derivative_qps_per_s: slope,
        observed_seconds: oldest,
        samples: n,
    }
}

/// Everything the projection needs that is not the trend.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ProjectionInputs {
    pub class: WorkloadClass,
    /// Work the class is holding right now, queued plus in flight
    pub outstanding_work_seconds: f64,
    /// Completions per second the class is achieving
    pub service_capacity_qps: f64,
    /// Mean measured service time of one query in this class
    pub mean_service_seconds: f64,
    /// How long capacity takes to become useful here
    pub provision_latency: Duration,
    /// Largest warm pool the operator will pay for
    pub warm_pool_cap: u32,
}

/// Where the class will be one provision latency from now.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PressureProjection {
    pub class: WorkloadClass,
    pub horizon: Duration,
    pub current_pressure_seconds: f64,
    pub projected_pressure_seconds: f64,
    pub slo_seconds: f64,
    pub arrival_rate_qps: f64,
    pub arrival_rate_derivative: f64,
    /// Extra arrivals per second expected to exist by the time capacity could
    /// arrive. This is the quantity the warm pool sizes against, because it is
    /// the demand growth that provisioning cannot answer in time
    pub projected_arrival_rate_delta: f64,
    pub service_capacity_qps: f64,
    /// Nodes that have to already be running to absorb that growth, bounded by
    /// what the operator will pay for
    pub warm_pool_nodes: u32,
    /// What the warm pool would have been without the cap, so a deployment
    /// whose cap is binding can see that it is
    pub warm_pool_uncapped: u32,
    pub warm_pool_cap: u32,
    pub trend_trustworthy: bool,
}

impl PressureProjection {
    /// Whether the objective is expected to be missed by the horizon, which is
    /// the only justification for asking for hardware.
    pub fn breaching_at_horizon(&self) -> bool {
        self.slo_seconds.is_finite() && self.projected_pressure_seconds > self.slo_seconds
    }

    /// How much of the objective the projected reading uses.
    pub fn projected_utilization(&self) -> f64 {
        if !self.slo_seconds.is_finite() || self.slo_seconds <= 0.0 {
            return 0.0;
        }
        self.projected_pressure_seconds / self.slo_seconds
    }
}

/// Runs the queue forward by one provision latency.
///
/// The queue is treated as a fluid: arrivals accumulate at the fitted rate,
/// which is itself moving at the fitted slope, and the node drains at the rate
/// it has been measured to drain at. Integrating the difference over the
/// horizon gives the work that will be outstanding when capacity could arrive,
/// and dividing by the drain rate puts it back into the same seconds the
/// objective is written in.
///
/// A node draining faster than it fills projects to no backlog rather than to
/// a negative one, because the queue empties and stops.
pub fn project(inputs: &ProjectionInputs, trend: &ArrivalTrend) -> PressureProjection {
    let horizon = inputs.provision_latency;
    let h = horizon.as_secs_f64();
    let capacity = inputs.service_capacity_qps;

    let current_pressure = if capacity > 0.0 {
        inputs.outstanding_work_seconds / capacity
    } else {
        inputs.outstanding_work_seconds
    };

    // Only a trend backed by enough windows is allowed to move the projection.
    // Without one the honest answer is that the future looks like the present
    let (rate, slope) = if trend.trustworthy() {
        (trend.rate_qps, trend.derivative_qps_per_s)
    } else {
        (trend.rate_qps, 0.0)
    };

    // Net arrivals over the horizon, with the rate itself moving
    let net_arrivals = (rate - capacity) * h + 0.5 * slope * h * h;
    let added_work = net_arrivals.max(0.0) * inputs.mean_service_seconds;
    let projected_work = (inputs.outstanding_work_seconds + added_work).max(0.0);
    let projected_pressure = if capacity > 0.0 {
        projected_work / capacity
    } else {
        projected_work
    };

    let delta = (slope * h).max(0.0);
    let per_node_capacity = capacity.max(f64::MIN_POSITIVE);
    let uncapped = if delta > 0.0 && capacity > 0.0 {
        (delta / per_node_capacity).ceil().min(u32::MAX as f64) as u32
    } else {
        0
    };

    PressureProjection {
        class: inputs.class,
        horizon,
        current_pressure_seconds: current_pressure,
        projected_pressure_seconds: projected_pressure,
        slo_seconds: inputs.class.slo_seconds(),
        arrival_rate_qps: rate,
        arrival_rate_derivative: slope,
        projected_arrival_rate_delta: delta,
        service_capacity_qps: capacity,
        warm_pool_nodes: uncapped.min(inputs.warm_pool_cap),
        warm_pool_uncapped: uncapped,
        warm_pool_cap: inputs.warm_pool_cap,
        trend_trustworthy: trend.trustworthy(),
    }
}

/// How long the mesh expects to stay as quiet as it is.
///
/// The mirror of the projection, and the number a reclaim decision needs: a
/// node is only worth giving back if the quiet is going to outlast the cost of
/// getting it back. A falling arrival rate has a point where it would reach
/// the current service capacity, and that is how long the quiet lasts.
///
/// Flat or rising load predicts no idle window at all, which is the correct
/// answer: nothing about the trend says the mesh will need less.
pub fn predicted_idle_window(trend: &ArrivalTrend, service_capacity_qps: f64) -> Duration {
    if !trend.trustworthy() || trend.derivative_qps_per_s >= 0.0 {
        return Duration::ZERO;
    }
    let headroom = service_capacity_qps - trend.rate_qps;
    if headroom <= 0.0 {
        return Duration::ZERO;
    }
    // Rate is falling, so it will not reach capacity from below. The window is
    // bounded by how long it takes the remaining load to reach zero
    let seconds = trend.rate_qps / trend.derivative_qps_per_s.abs();
    if !seconds.is_finite() || seconds <= 0.0 {
        return Duration::ZERO;
    }
    Duration::from_secs_f64(seconds.min(u32::MAX as f64))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Samples for a rate that climbs linearly, newest first, exactly as the
    /// history ring hands them over.
    fn ramp(from: f64, per_second: f64, seconds: f64, step: f64) -> Vec<ArrivalSample> {
        let mut out = Vec::new();
        let mut age = 0.0;
        while age <= seconds {
            out.push(ArrivalSample {
                age_seconds: age,
                rate_qps: from + per_second * (seconds - age),
            });
            age += step;
        }
        out
    }

    #[test]
    fn a_flat_load_has_no_slope() {
        let flat: Vec<ArrivalSample> = (0..600)
            .map(|i| ArrivalSample {
                age_seconds: i as f64 * 0.1,
                rate_qps: 1000.0,
            })
            .collect();
        let trend = fit_arrival_trend(&flat);
        assert!(trend.derivative_qps_per_s.abs() < 1e-6, "{trend:?}");
        assert!((trend.rate_qps - 1000.0).abs() < 1e-6);
        assert!(trend.trustworthy());
    }

    #[test]
    fn a_ramp_recovers_its_own_slope() {
        let trend = fit_arrival_trend(&ramp(500.0, 20.0, 60.0, 0.1));
        assert!(
            (trend.derivative_qps_per_s - 20.0).abs() < 0.01,
            "{trend:?}"
        );
        // The fitted value at now is the top of the ramp
        assert!((trend.rate_qps - 1700.0).abs() < 1.0, "{trend:?}");
    }

    /// One outlying window must not become a trend.
    #[test]
    fn a_single_spike_barely_moves_the_slope() {
        let mut samples: Vec<ArrivalSample> = (0..600)
            .map(|i| ArrivalSample {
                age_seconds: i as f64 * 0.1,
                rate_qps: 1000.0,
            })
            .collect();
        samples[0].rate_qps = 50_000.0;
        let trend = fit_arrival_trend(&samples);
        let projected = trend.rate_at(Duration::from_secs(30));
        assert!(
            projected < 10_000.0,
            "one window dragged the projection to {projected}"
        );
    }

    #[test]
    fn too_few_windows_produce_no_trend() {
        let trend = fit_arrival_trend(&ramp(0.0, 100.0, 0.3, 0.1));
        assert!(!trend.trustworthy());
        let inputs = ProjectionInputs {
            class: WorkloadClass::Interactive,
            outstanding_work_seconds: 1.0,
            service_capacity_qps: 100.0,
            mean_service_seconds: 0.01,
            provision_latency: Duration::from_secs(30),
            warm_pool_cap: 8,
        };
        let projection = project(&inputs, &trend);
        assert_eq!(projection.warm_pool_nodes, 0);
        assert!(!projection.trend_trustworthy);
    }

    /// The warm pool is the growth that lands inside one provision latency.
    #[test]
    fn the_warm_pool_covers_one_provision_latency_of_growth() {
        let trend = fit_arrival_trend(&ramp(500.0, 20.0, 60.0, 0.1));
        let inputs = ProjectionInputs {
            class: WorkloadClass::Interactive,
            outstanding_work_seconds: 2.0,
            service_capacity_qps: 600.0,
            mean_service_seconds: 0.002,
            provision_latency: Duration::from_secs(30),
            warm_pool_cap: 32,
        };
        let projection = project(&inputs, &trend);
        let expected = 20.0 * 30.0;
        let error = (projection.projected_arrival_rate_delta - expected).abs() / expected;
        assert!(error < 0.15, "{projection:?}");
        // 600 qps a node, 600 qps of growth, so one node
        assert_eq!(projection.warm_pool_nodes, 1);
    }

    /// The configured value bounds the answer and never raises it.
    #[test]
    fn the_configured_warm_pool_is_a_ceiling_not_a_target() {
        let trend = fit_arrival_trend(&ramp(100.0, 200.0, 60.0, 0.1));
        let mut inputs = ProjectionInputs {
            class: WorkloadClass::Interactive,
            outstanding_work_seconds: 0.0,
            service_capacity_qps: 100.0,
            mean_service_seconds: 0.01,
            provision_latency: Duration::from_secs(30),
            warm_pool_cap: 4,
        };
        let capped = project(&inputs, &trend);
        assert_eq!(capped.warm_pool_nodes, 4);
        assert!(capped.warm_pool_uncapped > 4);

        // A quiet node with a generous cap still keeps nothing warm
        let flat = fit_arrival_trend(
            &(0..600)
                .map(|i| ArrivalSample {
                    age_seconds: i as f64 * 0.1,
                    rate_qps: 10.0,
                })
                .collect::<Vec<_>>(),
        );
        inputs.warm_pool_cap = 64;
        assert_eq!(project(&inputs, &flat).warm_pool_nodes, 0);
    }

    /// A load that is already growing past capacity projects a breach before
    /// it has breached, which is the whole point.
    #[test]
    fn a_rising_load_breaches_at_the_horizon_before_it_breaches_now() {
        // Arrivals sit exactly at capacity now and are climbing, so nothing is
        // wrong yet and something will be
        let trend = fit_arrival_trend(&ramp(200.0, 5.0, 60.0, 0.1));
        let inputs = ProjectionInputs {
            class: WorkloadClass::Interactive,
            outstanding_work_seconds: 0.4,
            service_capacity_qps: 500.0,
            mean_service_seconds: 0.004,
            provision_latency: Duration::from_secs(90),
            warm_pool_cap: 16,
        };
        let projection = project(&inputs, &trend);
        assert!(
            projection.current_pressure_seconds < projection.slo_seconds,
            "the test needs a node that is not breaching yet: {projection:?}"
        );
        assert!(
            projection.breaching_at_horizon(),
            "the projection missed a ramp that overtakes capacity: {projection:?}"
        );
    }

    /// A queue that is draining projects to empty, not to negative work.
    #[test]
    fn a_draining_queue_projects_to_no_backlog() {
        let falling = fit_arrival_trend(&ramp(1000.0, -10.0, 60.0, 0.1));
        let inputs = ProjectionInputs {
            class: WorkloadClass::Bulk,
            outstanding_work_seconds: 5.0,
            service_capacity_qps: 900.0,
            mean_service_seconds: 0.002,
            provision_latency: Duration::from_secs(60),
            warm_pool_cap: 8,
        };
        let projection = project(&inputs, &falling);
        assert_eq!(projection.warm_pool_nodes, 0);
        assert!(projection.projected_pressure_seconds >= 0.0);
        assert!(
            projection.projected_pressure_seconds <= projection.current_pressure_seconds + 1e-9
        );
    }

    #[test]
    fn an_idle_window_is_only_predicted_when_load_is_falling() {
        let rising = fit_arrival_trend(&ramp(100.0, 10.0, 60.0, 0.1));
        assert_eq!(predicted_idle_window(&rising, 1000.0), Duration::ZERO);

        let falling = fit_arrival_trend(&ramp(1000.0, -10.0, 60.0, 0.1));
        let window = predicted_idle_window(&falling, 900.0);
        // Rate is around 400 and falling at 10 a second
        assert!(
            window > Duration::from_secs(20) && window < Duration::from_secs(80),
            "{window:?}"
        );

        // Falling but still above what the node can serve is not quiet
        assert_eq!(predicted_idle_window(&falling, 100.0), Duration::ZERO);
    }
}
