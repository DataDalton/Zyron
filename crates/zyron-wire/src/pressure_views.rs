//! The `zyron_sys.pressure.*` views.
//!
//! Everything the controller decides is readable as SQL, including the things
//! it decided not to do. A controller nobody can interrogate is a controller
//! nobody can trust, and the failure mode being designed against here is an
//! operator watching latency climb with no way to ask why the node did not
//! react, or why it reacted the way it did.
//!
//! Read-only, computed on read. The one thing on this schema that is stored is
//! the calibration cache, and it is stored because its whole purpose is to
//! outlive the process.

use zyron_pressure::capability::{NodeCapabilities, OperatorKind};
use zyron_pressure::pressure::WorkloadClass;
use zyron_pressure::pressure_control::PressureController;

use crate::messages::backend::FieldDescription;
use crate::types::{PG_INT4_OID, PG_INT8_OID, PG_TEXT_OID};

/// Every view on the schema, in the order they are documented.
pub const PRESSURE_VIEW_NAMES: &[&str] = &[
    "zyron_sys.pressure.node_capabilities",
    "zyron_sys.pressure.current",
    "zyron_sys.pressure.history",
    "zyron_sys.pressure.admissions",
    "zyron_sys.pressure.classes",
    "zyron_sys.pressure.calibration_cache",
    "zyron_sys.pressure.learning_ledger",
    "zyron_sys.pressure.capacity_projection",
    "zyron_sys.pressure.node_state",
    "zyron_sys.pressure.tenants",
    "zyron_sys.pressure.projection",
    "zyron_sys.pressure.provisioner",
    "zyron_sys.pressure.hot_set",
    "zyron_sys.pressure.query_shapes",
    "zyron_sys.pressure.spill_stats",
];

/// Whether a name is one of ours.
pub fn is_pressure_view(name: &str) -> bool {
    PRESSURE_VIEW_NAMES
        .iter()
        .any(|v| v.eq_ignore_ascii_case(name))
}

type ViewRows = (Vec<FieldDescription>, Vec<Vec<Option<Vec<u8>>>>);

fn field(name: &str, oid: i32, size: i16) -> FieldDescription {
    FieldDescription {
        name: name.to_string(),
        table_oid: 0,
        column_attr: 0,
        type_oid: oid,
        type_size: size,
        type_modifier: -1,
        format: 0,
    }
}

fn text(v: impl std::fmt::Display) -> Option<Vec<u8>> {
    Some(v.to_string().into_bytes())
}

/// Renders a float with enough places to read a microsecond in a seconds
/// column, which is the resolution the signal is actually kept at.
fn seconds(v: f64) -> Option<Vec<u8>> {
    if !v.is_finite() {
        return Some(b"unbounded".to_vec());
    }
    Some(format!("{v:.6}").into_bytes())
}

/// Dispatches a read. None when the name is not one of ours.
pub fn query_pressure_view(
    name: &str,
    capabilities: Option<&NodeCapabilities>,
) -> Option<ViewRows> {
    let controller = PressureController::global();
    let lower = name.to_ascii_lowercase();
    match lower.as_str() {
        "zyron_sys.pressure.node_capabilities" => Some(build_node_capabilities(capabilities)),
        "zyron_sys.pressure.current" => Some(build_current(controller)),
        "zyron_sys.pressure.history" => Some(build_history(controller)),
        "zyron_sys.pressure.admissions" => Some(build_admissions(controller)),
        "zyron_sys.pressure.classes" => Some(build_classes()),
        "zyron_sys.pressure.calibration_cache" => {
            Some(build_calibration_cache(controller, capabilities))
        }
        "zyron_sys.pressure.learning_ledger" => Some(build_learning_ledger(controller)),
        "zyron_sys.pressure.capacity_projection" => {
            Some(build_capacity_projection(controller, capabilities))
        }
        "zyron_sys.pressure.node_state" => Some(build_node_state(controller)),
        "zyron_sys.pressure.tenants" => Some(build_tenants(controller)),
        "zyron_sys.pressure.projection" => Some(build_projection(controller)),
        "zyron_sys.pressure.provisioner" => Some(build_provisioner(controller)),
        "zyron_sys.pressure.hot_set" => Some(build_hot_set(controller, capabilities)),
        "zyron_sys.pressure.query_shapes" => Some(build_query_shapes(controller)),
        "zyron_sys.pressure.spill_stats" => Some(build_spill_stats()),
        _ => None,
    }
}

/// What the node measured about the machine it is on. One row per mount.
fn build_node_capabilities(capabilities: Option<&NodeCapabilities>) -> ViewRows {
    let fields = vec![
        field("node_id", PG_INT8_OID, 8),
        field("fingerprint", PG_TEXT_OID, -1),
        field("cpu_model", PG_TEXT_OID, -1),
        field("core_count", PG_INT4_OID, 4),
        field("mem_total_bytes", PG_INT8_OID, 8),
        field("mem_available_bytes", PG_INT8_OID, 8),
        field("numa_nodes", PG_INT4_OID, 4),
        field("simd_level", PG_TEXT_OID, -1),
        field("connection_ceiling", PG_INT4_OID, 4),
        field("mount_path", PG_TEXT_OID, -1),
        field("mount_device_kind", PG_TEXT_OID, -1),
        field("mount_total_bytes", PG_INT8_OID, 8),
        field("mount_available_bytes", PG_INT8_OID, 8),
        field("read_p50_ns", PG_INT8_OID, 8),
        field("read_p99_ns", PG_INT8_OID, 8),
        field("read_throughput_mb_s", PG_TEXT_OID, -1),
        field("mount_sample_count", PG_INT8_OID, 8),
        field("calibration_inherited", PG_TEXT_OID, -1),
        field("probed_at_us", PG_INT8_OID, 8),
    ];
    let Some(caps) = capabilities else {
        return (fields, Vec::new());
    };
    let ceiling = PressureController::global().connections().ceiling();
    let rows = caps
        .mounts
        .iter()
        .map(|mount| {
            vec![
                text(caps.node_id),
                text(caps.fingerprint),
                text(&caps.inputs.cpu_model),
                text(caps.core_count),
                text(caps.mem_total_bytes),
                text(caps.mem_available_bytes),
                text(caps.numa_nodes),
                text(caps.simd_level.as_str()),
                text(ceiling),
                text(&mount.path),
                text(mount.device_kind.as_str()),
                text(mount.total_bytes),
                text(mount.available_bytes),
                text(mount.read_p50_ns),
                text(mount.read_p99_ns),
                text(format!("{:.2}", mount.read_throughput_mb_s)),
                text(mount.sample_count),
                text(caps.inherited),
                text(caps.probed_at_us),
            ]
        })
        .collect();
    (fields, rows)
}

/// Live pressure, one row per class.
fn build_current(controller: &PressureController) -> ViewRows {
    let fields = vec![
        field("node_id", PG_INT8_OID, 8),
        field("class", PG_TEXT_OID, -1),
        field("pressure_seconds", PG_TEXT_OID, -1),
        field("slo_seconds", PG_TEXT_OID, -1),
        field("slo_utilization", PG_TEXT_OID, -1),
        field("queued_work_seconds", PG_TEXT_OID, -1),
        field("active_work_seconds", PG_TEXT_OID, -1),
        field("queued_count", PG_INT4_OID, 4),
        field("in_flight", PG_INT4_OID, 4),
        field("ceiling", PG_INT4_OID, 4),
        field("service_capacity_qps", PG_TEXT_OID, -1),
        field("calibration_error", PG_TEXT_OID, -1),
        field("bottleneck_kind", PG_TEXT_OID, -1),
        field("actuator_level", PG_TEXT_OID, -1),
        field("admit_recommendation", PG_TEXT_OID, -1),
        field("admitted_total", PG_INT8_OID, 8),
        field("delayed_total", PG_INT8_OID, 8),
        field("shed_total", PG_INT8_OID, 8),
        field("bypassed_total", PG_INT8_OID, 8),
        field("updated_at_us", PG_INT8_OID, 8),
    ];
    let snapshot = controller.snapshot();
    let rows = snapshot
        .classes
        .iter()
        .map(|row| {
            // What a caller arriving right now would be told, which is more
            // use than the raw counters when reading this to explain a refusal
            let recommendation = if row.in_flight < row.ceiling {
                "admit"
            } else if (row.queued_count as usize) < row.class.queue_depth() {
                "delay"
            } else {
                "shed"
            };
            vec![
                text(snapshot.node_id),
                text(row.class.as_str()),
                seconds(row.pressure_seconds),
                seconds(row.slo_seconds),
                text(format!("{:.4}", row.slo_utilization())),
                seconds(row.queued_work_seconds),
                seconds(row.active_work_seconds),
                text(row.queued_count),
                text(row.in_flight),
                text(row.ceiling),
                text(format!("{:.3}", row.service_capacity_qps)),
                text(format!("{:.3}", row.calibration_error)),
                text(row.bottleneck.as_str()),
                text(row.actuator_level.as_str()),
                text(recommendation),
                text(row.admitted_total),
                text(row.delayed_total),
                text(row.shed_total),
                text(row.bypassed_total),
                text(snapshot.updated_at_us),
            ]
        })
        .collect();
    (fields, rows)
}

/// The history ring, newest first, one row per class per sample.
fn build_history(controller: &PressureController) -> ViewRows {
    let fields = vec![
        field("at_us", PG_INT8_OID, 8),
        field("class", PG_TEXT_OID, -1),
        field("pressure_seconds", PG_TEXT_OID, -1),
        field("queued_work_seconds", PG_TEXT_OID, -1),
        field("active_work_seconds", PG_TEXT_OID, -1),
        field("in_flight", PG_INT4_OID, 4),
        field("service_capacity_qps", PG_TEXT_OID, -1),
        field("bottleneck_kind", PG_TEXT_OID, -1),
        field("actuator_level", PG_TEXT_OID, -1),
    ];
    let mut rows = Vec::new();
    for sample in controller.history(usize::MAX) {
        for class in WorkloadClass::ALL {
            let i = class.index();
            rows.push(vec![
                text(sample.at_us),
                text(class.as_str()),
                seconds(sample.pressure_us[i] as f64 / 1e6),
                seconds(sample.queued_us[i] as f64 / 1e6),
                seconds(sample.active_us[i] as f64 / 1e6),
                text(sample.in_flight[i]),
                text(format!(
                    "{:.3}",
                    sample.capacity_milli_qps[i] as f64 / 1000.0
                )),
                text(sample.bottleneck.as_str()),
                text(sample.actuator.as_str()),
            ]);
        }
    }
    (fields, rows)
}

/// Recent admission decisions and why each was made.
fn build_admissions(controller: &PressureController) -> ViewRows {
    let fields = vec![
        field("at_us", PG_INT8_OID, 8),
        field("class", PG_TEXT_OID, -1),
        field("decision", PG_TEXT_OID, -1),
        field("estimated_work_seconds", PG_TEXT_OID, -1),
        field("queue_wait_us", PG_INT8_OID, 8),
        field("tenant_id", PG_TEXT_OID, -1),
        field("reason", PG_TEXT_OID, -1),
    ];
    let rows = controller
        .admissions(usize::MAX)
        .into_iter()
        .map(|r| {
            vec![
                text(r.at_us),
                text(r.class.as_str()),
                text(r.decision),
                seconds(r.estimated_work_seconds),
                text(r.queue_wait_us),
                r.tenant_id.as_deref().map(|t| t.as_bytes().to_vec()),
                text(r.reason),
            ]
        })
        .collect();
    (fields, rows)
}

/// The class definitions, so the objectives the controller measures against
/// are readable rather than folded into its behaviour.
fn build_classes() -> ViewRows {
    let fields = vec![
        field("class", PG_TEXT_OID, -1),
        field("slo_seconds", PG_TEXT_OID, -1),
        field("queue_depth", PG_INT4_OID, 4),
        field("max_estimated_work_seconds", PG_TEXT_OID, -1),
        field("bypass_below_seconds", PG_TEXT_OID, -1),
    ];
    let rows = WorkloadClass::ALL
        .into_iter()
        .map(|class| {
            let upper = match class {
                WorkloadClass::Interactive => seconds(WorkloadClass::INTERACTIVE_MAX_WORK_SECONDS),
                _ => Some(b"unbounded".to_vec()),
            };
            vec![
                text(class.as_str()),
                seconds(class.slo_seconds()),
                text(class.queue_depth()),
                upper,
                seconds(WorkloadClass::BYPASS_WORK_SECONDS),
            ]
        })
        .collect();
    (fields, rows)
}

/// The persisted calibration, one row per operator kind.
fn build_calibration_cache(
    controller: &PressureController,
    capabilities: Option<&NodeCapabilities>,
) -> ViewRows {
    let fields = vec![
        field("fingerprint", PG_TEXT_OID, -1),
        field("operator_kind", PG_TEXT_OID, -1),
        field("unit", PG_TEXT_OID, -1),
        field("ns_per_unit", PG_TEXT_OID, -1),
        field("sample_count", PG_INT8_OID, 8),
        field("measured", PG_TEXT_OID, -1),
        field("local_ns_per_unit", PG_TEXT_OID, -1),
        field("local_sample_count", PG_INT8_OID, 8),
        field("fleet_ns_per_unit", PG_TEXT_OID, -1),
        field("fleet_sample_count", PG_INT8_OID, 8),
        field("recent_sample_count", PG_INT8_OID, 8),
        field("probe_candidate", PG_TEXT_OID, -1),
    ];
    let fingerprint = capabilities
        .map(|c| c.fingerprint.to_hex())
        .unwrap_or_else(|| "unknown".to_string());
    let accumulator = controller.coefficients();
    // Three columns for what looks like one number, because they answer
    // different questions. The effective value is what plans are priced
    // against, the local one is what this node can vouch for, and the fleet
    // one is what it inherited from siblings of the same hardware shape. An
    // operator debugging a mispriced plan needs to know which of those moved
    let effective = accumulator.effective();
    let local = accumulator.snapshot();
    let fleet = accumulator.fleet();
    let candidates = accumulator.probe_candidates();
    let rows = OperatorKind::ALL
        .into_iter()
        .map(|kind| {
            vec![
                text(&fingerprint),
                text(kind.as_str()),
                text(if kind.unit_is_page() { "page" } else { "row" }),
                text(format!("{:.3}", effective.get(kind))),
                text(effective.samples(kind)),
                text(effective.is_measured(kind)),
                text(format!("{:.3}", local.get(kind))),
                text(local.samples(kind)),
                text(format!("{:.3}", fleet.get(kind))),
                text(fleet.samples(kind)),
                text(accumulator.recent_samples(kind)),
                text(candidates.contains(&kind)),
            ]
        })
        .collect();
    (fields, rows)
}

/// What each exploration step past the believed knee produced.
fn build_learning_ledger(controller: &PressureController) -> ViewRows {
    let fields = vec![
        field("at_us", PG_INT8_OID, 8),
        field("class", PG_TEXT_OID, -1),
        field("ceiling_before", PG_INT4_OID, 4),
        field("ceiling_after", PG_INT4_OID, 4),
        field("throughput_before_qps", PG_TEXT_OID, -1),
        field("throughput_after_qps", PG_TEXT_OID, -1),
        field("pressure_seconds", PG_TEXT_OID, -1),
        field("kept", PG_TEXT_OID, -1),
    ];
    let rows = controller
        .ledger(usize::MAX)
        .into_iter()
        .map(|e| {
            vec![
                text(e.at_us),
                text(e.class.as_str()),
                text(e.ceiling_before),
                text(e.ceiling_after),
                text(format!("{:.3}", e.throughput_before)),
                text(format!("{:.3}", e.throughput_after)),
                seconds(e.pressure_seconds),
                text(e.kept),
            ]
        })
        .collect();
    (fields, rows)
}

/// What hardware a multiple of the current workload would need.
///
/// Falls out of the rest of the substrate rather than being modelled
/// separately: the arrival rate is measured, the knee is measured, and the
/// coefficients say what a row costs on this hardware, so the answer is
/// arithmetic over three things already known. This is the report an on-prem
/// operator needs, because their ceiling is hardware they have to buy rather
/// than a quota they can raise.
fn build_capacity_projection(
    controller: &PressureController,
    capabilities: Option<&NodeCapabilities>,
) -> ViewRows {
    let fields = vec![
        field("multiple", PG_TEXT_OID, -1),
        field("class", PG_TEXT_OID, -1),
        field("observed_qps", PG_TEXT_OID, -1),
        field("projected_qps", PG_TEXT_OID, -1),
        field("projected_pressure_seconds", PG_TEXT_OID, -1),
        field("slo_seconds", PG_TEXT_OID, -1),
        field("meets_slo", PG_TEXT_OID, -1),
        field("cores_now", PG_INT4_OID, 4),
        field("cores_needed", PG_INT4_OID, 4),
        field("memory_now_bytes", PG_INT8_OID, 8),
        field("memory_needed_bytes", PG_INT8_OID, 8),
    ];
    let cores_now = capabilities.map(|c| c.core_count).unwrap_or(0);
    let memory_now = capabilities.map(|c| c.mem_total_bytes).unwrap_or(0);
    let snapshot = controller.snapshot();

    let mut rows = Vec::new();
    // Doubling is the question people actually ask, and the two neighbours
    // around it show whether the answer is near a cliff
    for multiple in [1.0f64, 2.0, 4.0, 10.0] {
        for row in &snapshot.classes {
            let observed = row.service_capacity_qps;
            let projected_qps = observed * multiple;
            // Work scales with arrivals; capacity does not, until hardware
            // does. Pressure is therefore the current backlog times the
            // multiple, against the capacity that exists today
            let projected_pressure = if observed > 0.0 {
                (row.queued_work_seconds + row.active_work_seconds) * multiple / observed
            } else {
                (row.queued_work_seconds + row.active_work_seconds) * multiple
            };
            let meets = !row.slo_seconds.is_finite() || projected_pressure <= row.slo_seconds;
            // Cores scale with the multiple only once the objective is missed;
            // below that the hardware already in place carries it
            let cores_needed = if meets {
                cores_now
            } else {
                ((cores_now as f64) * multiple).ceil() as u32
            };
            let memory_needed = if meets {
                memory_now
            } else {
                ((memory_now as f64) * multiple) as u64
            };
            rows.push(vec![
                text(format!("{multiple:.0}x")),
                text(row.class.as_str()),
                text(format!("{observed:.3}")),
                text(format!("{projected_qps:.3}")),
                seconds(projected_pressure),
                seconds(row.slo_seconds),
                text(meets),
                text(cores_now),
                text(cores_needed),
                text(memory_now),
                text(memory_needed),
            ]);
        }
    }
    (fields, rows)
}

/// The single rollup a mesh scheduler compares between nodes.
fn build_node_state(controller: &PressureController) -> ViewRows {
    let fields = vec![
        field("node_id", PG_INT8_OID, 8),
        field("overall_utilization", PG_TEXT_OID, -1),
        field("bottleneck_kind", PG_TEXT_OID, -1),
        field("actuator_level", PG_TEXT_OID, -1),
        field("parallel_permits_total", PG_INT4_OID, 4),
        field("parallel_permits_available", PG_INT4_OID, 4),
        field("parallel_scale_pct", PG_INT4_OID, 4),
        field("parallel_base_total", PG_INT4_OID, 4),
        field("parallel_growth_pct", PG_INT4_OID, 4),
        field("query_memory_pct", PG_INT4_OID, 4),
        field("contention_occ_abort_rate", PG_TEXT_OID, -1),
        field("contention_hot_key_share", PG_TEXT_OID, -1),
        field("contention_group_commit_rate", PG_TEXT_OID, -1),
        field("contention_writes_waiting", PG_INT4_OID, 4),
        field("contention_page_reads", PG_INT8_OID, 8),
        field("memory_reserved_bytes", PG_INT8_OID, 8),
        field("memory_ceiling_bytes", PG_INT8_OID, 8),
        field("connections_live", PG_INT4_OID, 4),
        field("connections_ceiling", PG_INT4_OID, 4),
        field("connections_refused_total", PG_INT8_OID, 8),
        field("ticks", PG_INT8_OID, 8),
        field("sequence", PG_INT8_OID, 8),
        field("updated_at_us", PG_INT8_OID, 8),
    ];
    let snapshot = controller.snapshot();
    let connections = controller.connections();
    let contention = controller.contention();
    let row = vec![
        text(snapshot.node_id),
        text(format!("{:.4}", snapshot.overall_utilization)),
        text(snapshot.bottleneck.as_str()),
        text(snapshot.actuator_level.as_str()),
        text(snapshot.parallel_permits_total),
        text(snapshot.parallel_permits_available),
        text(controller.capacity().scale_pct()),
        text(controller.capacity().base_total()),
        text(controller.capacity().growth_pct()),
        text(controller.query_memory_pct()),
        // The classifier's own inputs, so an operator asking why the node
        // chose a rung sees the readings it chose from rather than only the
        // conclusion
        text(format!("{:.4}", contention.occ_abort_rate())),
        text(format!("{:.4}", contention.hot_key_share())),
        text(format!("{:.4}", contention.group_commit_hit_rate())),
        text(contention.writes_waiting()),
        text(contention.page_reads()),
        text(snapshot.memory_reserved_bytes),
        text(snapshot.memory_ceiling_bytes),
        text(connections.live()),
        text(connections.ceiling()),
        text(connections.refused_total()),
        text(controller.ticks()),
        text(controller.sequence()),
        text(snapshot.updated_at_us),
    ];
    (fields, vec![row])
}

/// Pressure broken out by whoever caused it.
fn build_tenants(controller: &PressureController) -> ViewRows {
    let fields = vec![
        field("tenant_id", PG_TEXT_OID, -1),
        field("queued_work_seconds", PG_TEXT_OID, -1),
        field("active_work_seconds", PG_TEXT_OID, -1),
        field("total_work_seconds", PG_TEXT_OID, -1),
        field("completed", PG_INT8_OID, 8),
        field("completed_work_seconds", PG_TEXT_OID, -1),
    ];
    let rows = controller
        .tenant_pressure()
        .into_iter()
        .map(|t| {
            vec![
                text(&t.tenant_id),
                seconds(t.queued_work_seconds),
                seconds(t.active_work_seconds),
                seconds(t.total_work_seconds()),
                text(t.completed),
                seconds(t.completed_work_seconds),
            ]
        })
        .collect();
    (fields, rows)
}

/// Where each class will be one provision latency from now, and how much
/// capacity would have to already be running to meet it.
///
/// Separate from capacity_projection, which answers a hypothetical about a
/// multiple of today's workload. This one answers what the node expects to
/// happen, from the trend it has actually observed, and it is the reading a
/// scale-out decision is taken against.
fn build_projection(controller: &PressureController) -> ViewRows {
    let fields = vec![
        field("class", PG_TEXT_OID, -1),
        field("horizon_seconds", PG_TEXT_OID, -1),
        field("current_pressure_seconds", PG_TEXT_OID, -1),
        field("projected_pressure_seconds", PG_TEXT_OID, -1),
        field("slo_seconds", PG_TEXT_OID, -1),
        field("breaching_at_horizon", PG_TEXT_OID, -1),
        field("arrival_rate_qps", PG_TEXT_OID, -1),
        field("arrival_rate_derivative", PG_TEXT_OID, -1),
        field("projected_arrival_rate_delta", PG_TEXT_OID, -1),
        field("service_capacity_qps", PG_TEXT_OID, -1),
        field("warm_pool_nodes", PG_INT4_OID, 4),
        field("warm_pool_uncapped", PG_INT4_OID, 4),
        field("warm_pool_cap", PG_INT4_OID, 4),
        field("trend_trustworthy", PG_TEXT_OID, -1),
        field("predicted_idle_seconds", PG_TEXT_OID, -1),
    ];
    let rows = WorkloadClass::ALL
        .into_iter()
        .map(|class| {
            let p = controller.projection(class);
            vec![
                text(class.as_str()),
                seconds(p.horizon.as_secs_f64()),
                seconds(p.current_pressure_seconds),
                seconds(p.projected_pressure_seconds),
                seconds(p.slo_seconds),
                text(p.breaching_at_horizon()),
                text(format!("{:.3}", p.arrival_rate_qps)),
                text(format!("{:.3}", p.arrival_rate_derivative)),
                text(format!("{:.3}", p.projected_arrival_rate_delta)),
                text(format!("{:.3}", p.service_capacity_qps)),
                text(p.warm_pool_nodes),
                text(p.warm_pool_uncapped),
                text(p.warm_pool_cap),
                text(p.trend_trustworthy),
                seconds(controller.predicted_idle_window(class).as_secs_f64()),
            ]
        })
        .collect();
    (fields, rows)
}

/// What this node can do about capacity.
///
/// Reports the driver rather than the configured mode, which is the same
/// information arriving by the route the rest of the system uses: an operator
/// reading this sees what was actually selected, including the case where a
/// mode was configured and its control plane is not reachable from here.
fn build_provisioner(controller: &PressureController) -> ViewRows {
    let fields = vec![
        field("driver", PG_TEXT_OID, -1),
        field("can_provision", PG_TEXT_OID, -1),
        field("can_reclaim", PG_TEXT_OID, -1),
        field("can_scale_to_zero", PG_TEXT_OID, -1),
        field("min_nodes", PG_INT4_OID, 4),
        field("max_nodes", PG_INT4_OID, 4),
        field("billing_interval_seconds", PG_INT8_OID, 8),
        field("provision_latency_seconds", PG_TEXT_OID, -1),
        field("latency_is_measured", PG_TEXT_OID, -1),
        field("completed_provisions", PG_INT8_OID, 8),
        field("ladder_can_take_warm_node", PG_TEXT_OID, -1),
        field("ladder_can_provision", PG_TEXT_OID, -1),
    ];
    let registry = zyron_pressure::provisioner::ProvisionerRegistry::global();
    let driver = registry.active();
    let capabilities = driver.capabilities();
    let completed = registry.completed_provisions(driver.kind());
    let reach = controller.mesh_reach();
    let rows = vec![vec![
        text(driver.kind().as_str()),
        text(capabilities.can_provision),
        text(capabilities.can_reclaim),
        text(capabilities.can_scale_to_zero),
        text(capabilities.min_nodes),
        text(capabilities.max_nodes),
        text(capabilities.billing_interval.as_secs()),
        seconds(controller.provision_horizon().as_secs_f64()),
        text(completed > 0),
        text(completed),
        text(reach.can_take_warm_node),
        text(reach.can_provision),
    ]];
    (fields, rows)
}

/// The working-set manifest, and what resuming from it would cost.
///
/// The resume estimate is the number scale-to-zero is judged on, and it is
/// deliberately built from checkpoint staleness and working-set size rather
/// than from write-ahead log length: a node resuming from a clean checkpoint
/// does not replay the log, so log length says nothing about how long it takes
/// to come back.
fn build_hot_set(
    controller: &PressureController,
    capabilities: Option<&NodeCapabilities>,
) -> ViewRows {
    let fields = vec![
        field("pages", PG_INT4_OID, 4),
        field("query_shapes", PG_INT8_OID, 8),
        field("generated_us", PG_INT8_OID, 8),
        field("persisted", PG_TEXT_OID, -1),
        field("page_read_p50_ns", PG_INT8_OID, 8),
        field("read_throughput_mb_s", PG_TEXT_OID, -1),
        field("estimated_reload_seconds", PG_TEXT_OID, -1),
        field("checkpoint_age_seconds", PG_TEXT_OID, -1),
        field("checkpoint_clean", PG_TEXT_OID, -1),
        field("wal_bytes_since_checkpoint", PG_INT8_OID, 8),
        field("estimated_resume_seconds", PG_TEXT_OID, -1),
        field("may_scale_to_zero", PG_TEXT_OID, -1),
        field("blocked_by", PG_TEXT_OID, -1),
    ];
    let status = controller.hot_set_status();
    let mount = capabilities.and_then(|c| c.mounts.first());
    let read_p50_ns = mount.map(|m| m.read_p50_ns).unwrap_or(0);
    let read_throughput_mb_s = mount.map(|m| m.read_throughput_mb_s).unwrap_or(0.0);
    let checkpoint = controller.checkpoint_state();
    let checkpoint_age = controller.checkpoint_age();
    let inputs = zyron_pressure::provisioner::ScaleToZeroInputs {
        checkpoint_age,
        checkpoint_clean: checkpoint.clean,
        wal_bytes_since_checkpoint: checkpoint.wal_bytes_since,
        hot_set_persisted: status.persisted,
        hot_set_pages: status.pages,
        page_bytes: zyron_common::page::PAGE_SIZE as u64,
        read_throughput_bytes_per_s: read_throughput_mb_s * 1_000_000.0,
        page_read_p50: std::time::Duration::from_nanos(read_p50_ns),
    };
    let estimate = zyron_pressure::provisioner::estimate_resume(&inputs);
    // The same call the trigger makes, so what an operator reads here is what
    // a scale to zero would actually decide rather than a second opinion
    let readiness = zyron_pressure::provisioner::assert_scale_to_zero_ready(&inputs);
    let blocked_by = match &readiness {
        Ok(_) => String::new(),
        Err(e) => e.to_string(),
    };
    let rows = vec![vec![
        text(status.pages),
        text(status.shapes),
        text(status.generated_us),
        text(status.persisted),
        text(read_p50_ns),
        text(format!("{read_throughput_mb_s:.1}")),
        seconds(estimate.hot_set_reload.as_secs_f64()),
        seconds(checkpoint_age.as_secs_f64()),
        text(checkpoint.clean),
        text(checkpoint.wal_bytes_since),
        seconds(estimate.total.as_secs_f64()),
        text(readiness.is_ok()),
        text(blocked_by),
    ]];
    (fields, rows)
}

/// The query shapes the node has been serving, heaviest first.
///
/// The same list a draining node hands to its survivors, so what a handover
/// would carry is readable before the handover happens.
fn build_query_shapes(controller: &PressureController) -> ViewRows {
    let fields = vec![
        field("fingerprint", PG_TEXT_OID, -1),
        field("executions", PG_INT8_OID, 8),
        field("mean_work_seconds", PG_TEXT_OID, -1),
        field("total_work_seconds", PG_TEXT_OID, -1),
    ];
    let rows = controller
        .hot_queries(QUERY_SHAPE_VIEW_LIMIT)
        .into_iter()
        .map(|q| {
            let mean = q.mean_work_us as f64 / 1e6;
            vec![
                text(format!("{:016x}", q.fingerprint)),
                text(q.executions),
                seconds(mean),
                seconds(mean * q.executions as f64),
            ]
        })
        .collect();
    (fields, rows)
}

/// What spilling has cost this node.
///
/// One row, because the question is about the node rather than about a query:
/// a query's own spilling is in its plan. What an operator wants from here is
/// whether this node is spilling at all, how much of the quota is in use, and
/// whether anything was refused, because a refusal is a query that spilling
/// did not save and is the signal that the quota is too small or the workload
/// too large.
fn build_spill_stats() -> ViewRows {
    use std::sync::atomic::Ordering;

    let fields = vec![
        field("files_created", PG_INT8_OID, 8),
        field("files_deleted", PG_INT8_OID, 8),
        field("files_live", PG_INT8_OID, 8),
        field("bytes_written", PG_INT8_OID, 8),
        field("bytes_read", PG_INT8_OID, 8),
        field("batches_written", PG_INT8_OID, 8),
        field("batches_read", PG_INT8_OID, 8),
        field("live_bytes", PG_INT8_OID, 8),
        field("peak_live_bytes", PG_INT8_OID, 8),
        field("sorts_spilled", PG_INT8_OID, 8),
        field("joins_spilled", PG_INT8_OID, 8),
        field("aggregates_spilled", PG_INT8_OID, 8),
        field("runs_written", PG_INT8_OID, 8),
        field("partitions_spilled", PG_INT8_OID, 8),
        field("blocked_passes", PG_INT8_OID, 8),
        field("quota_refusals", PG_INT8_OID, 8),
    ];
    let stats = zyron_executor::spill::SpillStats::global();
    let created = stats.files_created.load(Ordering::Relaxed);
    let deleted = stats.files_deleted.load(Ordering::Relaxed);
    let rows = vec![vec![
        text(created),
        text(deleted),
        text(created.saturating_sub(deleted)),
        text(stats.bytes_written.load(Ordering::Relaxed)),
        text(stats.bytes_read.load(Ordering::Relaxed)),
        text(stats.batches_written.load(Ordering::Relaxed)),
        text(stats.batches_read.load(Ordering::Relaxed)),
        text(stats.live_bytes.load(Ordering::Relaxed)),
        text(stats.peak_live_bytes.load(Ordering::Relaxed)),
        text(stats.sorts_spilled.load(Ordering::Relaxed)),
        text(stats.joins_spilled.load(Ordering::Relaxed)),
        text(stats.aggregates_spilled.load(Ordering::Relaxed)),
        text(stats.runs_merged.load(Ordering::Relaxed)),
        text(stats.partitions_spilled.load(Ordering::Relaxed)),
        text(stats.blocked_passes.load(Ordering::Relaxed)),
        text(stats.quota_refusals.load(Ordering::Relaxed)),
    ]];
    (fields, rows)
}

/// Query shapes the view returns. Enough to see the workload, bounded so a
/// node tracking thousands of generated statements does not return all of
/// them to a operator who asked what it was busy with.
const QUERY_SHAPE_VIEW_LIMIT: usize = 256;

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn column(fields: &[FieldDescription], name: &str) -> usize {
        fields
            .iter()
            .position(|f| f.name == name)
            .unwrap_or_else(|| panic!("no column {name}"))
    }

    fn cell(row: &[Option<Vec<u8>>], idx: usize) -> String {
        String::from_utf8(row[idx].clone().unwrap_or_default()).expect("utf8")
    }

    #[test]
    fn every_named_view_answers() {
        let caps = NodeCapabilities::probe(1, None);
        for name in PRESSURE_VIEW_NAMES {
            let built = query_pressure_view(name, Some(&caps));
            assert!(built.is_some(), "{name} returned nothing");
            let (fields, rows) = built.expect("view");
            assert!(!fields.is_empty(), "{name} has no columns");
            // Every row has to be the width the schema declares, or a client
            // decoding it reads one column's bytes as another's
            for row in &rows {
                assert_eq!(row.len(), fields.len(), "{name} row width");
            }
        }
    }

    #[test]
    fn an_unknown_name_is_not_ours() {
        assert!(!is_pressure_view("zyron_sys.pressure.nope"));
        assert!(!is_pressure_view("zyron_stat_wal"));
        assert!(is_pressure_view("zyron_sys.pressure.current"));
        // Names are matched the way SQL matches them
        assert!(is_pressure_view("ZYRON_SYS.PRESSURE.CURRENT"));
        assert!(query_pressure_view("zyron_sys.pressure.nope", None).is_none());
    }

    #[test]
    fn current_reports_one_row_per_class_with_its_objective() {
        let (fields, rows) = query_pressure_view("zyron_sys.pressure.current", None).expect("view");
        assert_eq!(rows.len(), WorkloadClass::COUNT);
        let class_col = column(&fields, "class");
        let slo_col = column(&fields, "slo_seconds");
        let names: Vec<String> = rows.iter().map(|r| cell(r, class_col)).collect();
        assert_eq!(names, vec!["interactive", "bulk", "background"]);
        // Background has no objective and must say so rather than printing a
        // number that would compare wrong
        assert_eq!(cell(&rows[2], slo_col), "unbounded");
        assert_eq!(cell(&rows[0], slo_col), "0.100000");
    }

    #[test]
    fn classes_publish_the_thresholds_the_controller_uses() {
        let (fields, rows) = query_pressure_view("zyron_sys.pressure.classes", None).expect("view");
        let depth_col = column(&fields, "queue_depth");
        let bypass_col = column(&fields, "bypass_below_seconds");
        assert_eq!(rows.len(), WorkloadClass::COUNT);
        assert_eq!(
            cell(&rows[0], depth_col),
            WorkloadClass::Interactive.queue_depth().to_string()
        );
        assert_eq!(cell(&rows[0], bypass_col), "0.001000");
    }

    #[test]
    fn node_state_is_one_row_the_mesh_can_compare() {
        let (fields, rows) =
            query_pressure_view("zyron_sys.pressure.node_state", None).expect("view");
        assert_eq!(rows.len(), 1);
        let ceiling = column(&fields, "connections_ceiling");
        assert!(
            cell(&rows[0], ceiling).parse::<u32>().expect("number") >= 1,
            "the node reported a ceiling of zero connections"
        );
    }

    #[test]
    fn calibration_cache_covers_every_operator_kind() {
        let caps = NodeCapabilities::probe(1, None);
        let (fields, rows) =
            query_pressure_view("zyron_sys.pressure.calibration_cache", Some(&caps)).expect("view");
        assert_eq!(rows.len(), OperatorKind::COUNT);
        let unit = column(&fields, "unit");
        let kind = column(&fields, "operator_kind");
        // A page-unit kind must not be labelled as rows, or the number reads
        // three orders of magnitude wrong
        let page_read = rows
            .iter()
            .find(|r| cell(r, kind) == "page_read")
            .expect("page_read row");
        assert_eq!(cell(page_read, unit), "page");
        let seq_scan = rows
            .iter()
            .find(|r| cell(r, kind) == "seq_scan")
            .expect("seq_scan row");
        assert_eq!(cell(seq_scan, unit), "row");
    }

    #[test]
    fn node_capabilities_is_empty_rather_than_wrong_without_a_probe() {
        let (_, rows) =
            query_pressure_view("zyron_sys.pressure.node_capabilities", None).expect("view");
        assert!(rows.is_empty(), "unprobed capabilities invented a row");
    }

    #[test]
    fn capacity_projection_answers_the_doubling_question() {
        let caps = NodeCapabilities::probe(1, None);
        let (fields, rows) =
            query_pressure_view("zyron_sys.pressure.capacity_projection", Some(&caps))
                .expect("view");
        let multiple = column(&fields, "multiple");
        let cores_needed = column(&fields, "cores_needed");
        let cores_now = column(&fields, "cores_now");
        let doubled: Vec<_> = rows.iter().filter(|r| cell(r, multiple) == "2x").collect();
        assert_eq!(doubled.len(), WorkloadClass::COUNT);
        for row in doubled {
            let now: u32 = cell(row, cores_now).parse().expect("cores");
            let needed: u32 = cell(row, cores_needed).parse().expect("cores");
            assert!(
                needed >= now,
                "twice the workload asked for fewer cores than the node has"
            );
        }
    }
}
