//! The mesh registers with the pressure ladder, and the rungs do what they
//! say.
//!
//! Three things are being checked and they are different. That the seam is
//! connected: the ladder can see that something claims the two mesh rungs.
//! That an unreachable rung says so: a node with no scheduler answers that the
//! rung cannot be performed, with a reason, so the controller carries on to
//! shedding instead of sitting on a rung that changes nothing. And that a
//! reachable one performs: with a scheduler holding a warm node, claiming it
//! actually takes it out of the pool.
//!
//! The scheduler is installed once per process, so the whole sequence is one
//! test rather than several that would race each other over which of them
//! installed it.
//!
//! Run: cargo test -p zyron-mesh --test actuator_registration_test

use std::sync::Arc;

use zyron_mesh::rpc::{
    BeginDrainRequest, CancelProvisioningRequest, DrainStatus, DrainStatusRequest, HotSetChunk,
    HotSetManifestRequest, MeshFuture, MeshRpc, NodeAck, NodeStatus, NodeStatusRequest,
    PrefetchRequest, PrefetchStatus, RelocateSessionRequest, RelocationOutcome, RestartRequest,
    RollbackRequest, SetClusterSettingRequest, StageReleaseRequest,
};
use zyron_mesh::{MeshActuator, MeshScheduler, NodeRef, WarmPool};
use zyron_pressure::extension::{ActuatorExtension, ActuatorResult, ExtensionRegistry};
use zyron_pressure::pressure::{ActuatorDecision, ActuatorLevel, BottleneckKind, WorkloadClass};

/// The rungs the mesh owns.
const MESH_RUNGS: [ActuatorLevel; 2] = [ActuatorLevel::WarmPoolTake, ActuatorLevel::ProvisionNode];

fn decision(level: ActuatorLevel) -> ActuatorDecision {
    ActuatorDecision {
        class: WorkloadClass::Interactive,
        level,
        bottleneck: BottleneckKind::Cpu,
        dop_scale_pct: 100,
        delay: std::time::Duration::ZERO,
        pressure_seconds: 2.0,
        slo_seconds: 1.0,
        reason: "test",
    }
}

/// A transport nothing calls. The rungs under test never reach the network:
/// claiming a warm node is local, and provisioning talks to a driver.
struct Silent;

impl MeshRpc for Silent {
    fn begin_drain(&self, r: BeginDrainRequest) -> MeshFuture<'_, DrainStatus> {
        Box::pin(async move { Ok(DrainStatus::idle(r.target, r.sequence)) })
    }
    fn drain_status(&self, r: DrainStatusRequest) -> MeshFuture<'_, DrainStatus> {
        Box::pin(async move { Ok(DrainStatus::idle(r.target, r.sequence)) })
    }
    fn hot_set_manifest(&self, r: HotSetManifestRequest) -> MeshFuture<'_, HotSetChunk> {
        Box::pin(async move {
            Ok(HotSetChunk {
                source: r.target,
                sequence: r.sequence,
                chunk: 0,
                chunks_total: 1,
                page_ids: Vec::new(),
            })
        })
    }
    fn prefetch(&self, r: PrefetchRequest) -> MeshFuture<'_, PrefetchStatus> {
        Box::pin(async move {
            Ok(PrefetchStatus {
                target: r.target,
                sequence: r.sequence,
                pages_read: 0,
                pages_skipped: 0,
                budget_exhausted: false,
            })
        })
    }
    fn relocate_session(&self, _r: RelocateSessionRequest) -> MeshFuture<'_, RelocationOutcome> {
        Box::pin(async move { Ok(RelocationOutcome::Ended) })
    }
    fn cancel_provisioning(&self, _r: CancelProvisioningRequest) -> MeshFuture<'_, ()> {
        Box::pin(async move { Ok(()) })
    }
    fn node_status(&self, r: NodeStatusRequest) -> MeshFuture<'_, NodeStatus> {
        Box::pin(async move {
            Ok(NodeStatus {
                target: r.target,
                sequence: r.sequence,
                version: "0.12.0".into(),
                staged_version: String::new(),
                draining: false,
                accepting: true,
                queries_in_flight: 0,
                sessions_attached: 0,
                transactions_open: 0,
                p50_latency_us: 0,
                p99_latency_us: 0,
                throughput_milli_per_sec: 0,
                error_rate_ppm: 0,
                queries_in_window: 0,
                uptime_secs: 0,
            })
        })
    }
    fn stage_release(&self, r: StageReleaseRequest) -> MeshFuture<'_, NodeAck> {
        Box::pin(async move {
            Ok(NodeAck {
                target: r.target,
                sequence: r.sequence,
                accepted: true,
                detail: String::new(),
            })
        })
    }
    fn set_cluster_setting(&self, r: SetClusterSettingRequest) -> MeshFuture<'_, NodeAck> {
        Box::pin(async move {
            Ok(NodeAck {
                target: r.target,
                sequence: r.sequence,
                accepted: true,
                detail: String::new(),
            })
        })
    }
    fn restart_into_staged(&self, r: RestartRequest) -> MeshFuture<'_, NodeAck> {
        Box::pin(async move {
            Ok(NodeAck {
                target: r.target,
                sequence: r.sequence,
                accepted: true,
                detail: String::new(),
            })
        })
    }
    fn rollback_to_previous(&self, r: RollbackRequest) -> MeshFuture<'_, NodeAck> {
        Box::pin(async move {
            Ok(NodeAck {
                target: r.target,
                sequence: r.sequence,
                accepted: true,
                detail: String::new(),
            })
        })
    }
}

/// The whole seam, in the order a node lives it.
#[test]
fn the_mesh_rungs_report_honestly_before_and_after_a_scheduler_exists() {
    // Registering puts the rungs on the ladder, which is what lets the
    // controller see that something claims them
    zyron_mesh::register();
    let registry = ExtensionRegistry::global();
    for level in MESH_RUNGS {
        assert!(
            registry.handles(level),
            "{level:?} is not claimed by anything after the mesh registered"
        );
    }

    // With no scheduler installed, both answer that they cannot be reached
    // and say why. That answer is what sends the ladder on to shedding
    for level in MESH_RUNGS {
        match registry.actuate(&decision(level)) {
            ActuatorResult::NotAvailable { reason } => {
                assert!(
                    reason.contains("no mesh scheduler is installed"),
                    "{level:?} gave a reason that does not say why: {reason}"
                );
                assert!(
                    reason.contains(level.as_str()),
                    "{level:?} gave a reason that does not name the rung: {reason}"
                );
            }
            other => panic!("{level:?} answered {other:?} with no scheduler behind it"),
        }
    }

    // A scheduler holding one warm node
    let pool = Arc::new(WarmPool::new(4));
    assert!(pool.offer(NodeRef::new(9, "warm-9")));
    let scheduler = Arc::new(MeshScheduler::new(
        NodeRef::new(1, "local"),
        Arc::new(Silent),
        Arc::clone(&pool),
    ));
    assert!(
        zyron_mesh::install_scheduler(Arc::clone(&scheduler)),
        "the scheduler would not install"
    );

    // Claiming the warm node now performs, and the pool is one shorter for it
    match registry.actuate(&decision(ActuatorLevel::WarmPoolTake)) {
        ActuatorResult::Applied { detail } => {
            assert!(detail.contains("warm-9"), "{detail}");
        }
        other => panic!("claiming a warm node answered {other:?}"),
    }
    assert!(
        pool.is_empty(),
        "the rung reported claiming a node and left it in the pool"
    );

    // And with the pool empty it goes back to being unavailable, naming the
    // pool rather than the missing scheduler
    match registry.actuate(&decision(ActuatorLevel::WarmPoolTake)) {
        ActuatorResult::NotAvailable { reason } => {
            assert!(reason.contains("warm pool is empty"), "{reason}");
        }
        other => panic!("an empty pool answered {other:?}"),
    }

    // Provisioning with no driver installed reports that it cannot add
    // capacity, which is the honest answer on a node with no control plane
    match registry.actuate(&decision(ActuatorLevel::ProvisionNode)) {
        ActuatorResult::NotAvailable { reason } => {
            assert!(reason.contains("cannot add capacity"), "{reason}");
        }
        other => panic!("provisioning with no driver answered {other:?}"),
    }

    // Installing a second scheduler is refused rather than replacing the
    // first, because two would each hold a pool and each ask the provisioner
    assert!(
        !zyron_mesh::install_scheduler(scheduler),
        "a second scheduler was installed over the first"
    );
}

/// A rung the node performs itself is never offered to the mesh.
#[test]
fn the_local_rungs_stay_local() {
    zyron_mesh::register();
    for level in [
        ActuatorLevel::ReduceDop,
        ActuatorLevel::DelayAdmission,
        ActuatorLevel::TrimMemory,
        ActuatorLevel::ForceSpill,
        ActuatorLevel::Shed,
        ActuatorLevel::GrowWorkers,
    ] {
        assert_eq!(
            MeshActuator.try_actuate(&decision(level)),
            ActuatorResult::NotApplicable,
            "the mesh claimed {level:?}, which the node performs itself"
        );
    }
}

/// A rung nothing can perform must not strand the ladder.
///
/// This is what the whole seam exists for: the controller climbs to the top of
/// what it owns, finds the mesh rungs unreachable, and the honest answer is to
/// refuse work rather than wait for capacity that is not coming. Checked
/// against the ladder itself rather than restated, so a change to the rung
/// order cannot leave this asserting about the wrong one.
#[test]
fn shedding_stays_below_the_rungs_the_mesh_owns() {
    for level in MESH_RUNGS {
        assert!(
            ActuatorLevel::Shed < level,
            "{level:?} is at or below shedding, so an unreachable mesh would strand the ladder"
        );
    }
    assert!(
        !ExtensionRegistry::global().handles(ActuatorLevel::Shed),
        "shedding was claimed by an extension, which would make it unreachable too"
    );
}

/// The controller asks on rung entry and keeps what it was told, so a view can
/// show why a rung it climbed to did nothing.
#[test]
fn the_controller_records_what_the_mesh_told_it() {
    zyron_mesh::register();
    let controller = zyron_pressure::PressureController::new(0, 1 << 30);
    controller.apply(&decision(ActuatorLevel::ProvisionNode));
    let (level, detail) = controller
        .last_extension_answer()
        .expect("the controller reached the rung and kept no answer");
    assert_eq!(level, ActuatorLevel::ProvisionNode);
    assert!(!detail.is_empty(), "the answer was recorded empty");
}

/// A rung the node performs itself is never handed to an extension by the
/// controller.
#[test]
fn a_local_rung_is_never_offered_to_an_extension() {
    zyron_mesh::register();
    let controller = zyron_pressure::PressureController::new(0, 1 << 30);
    controller.apply(&decision(ActuatorLevel::ReduceDop));
    assert!(
        controller.last_extension_answer().is_none(),
        "a rung the node owns was handed to an extension"
    );
}
