//! The order the controller reaches for its levers in, and what stops it.
//!
//! A ladder is only a ladder if the cheap rungs are genuinely tried first. The
//! failure this guards against is subtle and expensive: a controller that
//! reaches straight for capacity looks fine on a graph, because the pressure
//! does come down, and costs money every time a burst could have been absorbed
//! by handing one query fewer workers.
//!
//! The gating is the other half. Three of the bottleneck kinds get worse when
//! capacity is added, and adding it is exactly what an ungated ladder does
//! when the cheap rungs fail to help. Conflict is the worst of them: more
//! concurrency means more aborts means more apparent load, which asks for more
//! concurrency.
//!
//! Run: cargo test -p zyron-pressure --test actuator_ladder_test

use std::time::Instant;

use zyron_pressure::pressure::{
    ActuatorLevel, BottleneckKind, ClassPressure, ParallelCapacity, WorkloadClass,
};
use zyron_pressure::pressure_control::{MeshReach, PressureController};

/// The controller's own settling period. A test that waited less would be
/// asking the ladder to climb before it has decided the rung failed.
const SETTLE: std::time::Duration = std::time::Duration::from_millis(500);

/// The fraction of its requested workers a query keeps once parallelism is
/// pulled back.
const REDUCED_DOP_PCT: u16 = 75;

fn controller(reach: MeshReach) -> PressureController {
    let c = PressureController::new(1, 64 * 1024 * 1024 * 1024);
    c.set_mesh_reach(reach);
    c
}

/// A controller whose parallel budget nothing else is spending.
///
/// The process budget is one account, so two tests applying rungs to it at
/// once would each be reading the other's effect. Leaked because the
/// controller holds it for the process lifetime, which for a test binary is
/// the test.
fn isolated_controller(reach: MeshReach, workers: u32) -> PressureController {
    let capacity: &'static ParallelCapacity =
        Box::leak(Box::new(ParallelCapacity::with_total(workers)));
    let c = PressureController::with_capacity(1, 64 * 1024 * 1024 * 1024, capacity);
    c.set_mesh_reach(reach);
    c
}

/// Decides and puts the decision into force, which is what the tick does.
fn step(c: &PressureController, pressure: &ClassPressure, at: Instant) -> ActuatorLevel {
    let decision = c.choose_actuator(pressure, BottleneckKind::None, at);
    c.apply(&decision);
    decision.level
}

/// The same, against a named bottleneck.
fn step_against(
    c: &PressureController,
    pressure: &ClassPressure,
    bottleneck: BottleneckKind,
    at: Instant,
) -> ActuatorLevel {
    let decision = c.choose_actuator(pressure, bottleneck, at);
    c.apply(&decision);
    decision.level
}

/// A class past its objective, with nothing else about it that would steer
/// the choice.
///
/// Priced against the class's own objective rather than at a fixed number,
/// because the objectives are three orders of magnitude apart and one constant
/// would breach Interactive while leaving Bulk comfortable.
fn breaching(class: WorkloadClass) -> ClassPressure {
    ClassPressure {
        class,
        queued_work_seconds: 10.0,
        active_work_seconds: 2.0,
        queued_count: 40,
        in_flight: 8,
        service_capacity_qps: 10.0,
        mean_service_seconds: 0.05,
        pressure_seconds: class.slo_seconds() * 1.5,
        calibration_error: 1.0,
        ceiling: 8,
        slo_seconds: class.slo_seconds(),
        admitted_total: 0,
        delayed_total: 0,
        shed_total: 0,
        bypassed_total: 0,
        bottleneck: BottleneckKind::None,
        actuator_level: ActuatorLevel::Steady,
    }
}

/// Every rung the controller may take against this bottleneck, cheapest
/// first. Worked out from the rule rather than written down, so a change to
/// the rule cannot leave a test asserting the old one.
fn legal_ladder(bottleneck: BottleneckKind, reach: MeshReach) -> Vec<ActuatorLevel> {
    ActuatorLevel::LADDER
        .into_iter()
        .filter(|l| l.legal_for(bottleneck) && reach.allows(*l))
        .collect()
}

/// The cheapest legal reachable rung.
fn expected_rung(bottleneck: BottleneckKind, reach: MeshReach) -> ActuatorLevel {
    // Nothing legal is left, and refusing work is the honest remaining answer
    legal_ladder(bottleneck, reach)
        .first()
        .copied()
        .unwrap_or(ActuatorLevel::Shed)
}

/// The ladder is ordered by what a rung costs to apply and to undo.
#[test]
fn the_ladder_is_in_ascending_cost_order() {
    let expected = [
        ActuatorLevel::ReduceDop,
        ActuatorLevel::DelayAdmission,
        ActuatorLevel::TrimMemory,
        ActuatorLevel::ForceSpill,
        ActuatorLevel::Shed,
        ActuatorLevel::GrowWorkers,
        ActuatorLevel::WarmPoolTake,
        ActuatorLevel::ProvisionNode,
    ];
    assert_eq!(ActuatorLevel::LADDER, expected);
    for pair in ActuatorLevel::LADDER.windows(2) {
        assert!(
            pair[0].code() < pair[1].code(),
            "{:?} is not cheaper than {:?}",
            pair[0],
            pair[1]
        );
    }
    // Steady sits below every rung, so a calm class compares as doing less
    // than the cheapest response
    assert!(ActuatorLevel::Steady < ActuatorLevel::LADDER[0]);
}

/// Walking the ladder: a rung that leaves the class breaching is replaced by
/// the next one, in order, all the way to provisioning.
///
/// This is the sequence the spec names, driven the way the controller actually
/// drives it. The pressure never comes down, which is the case the ladder
/// exists for: each rung is given its settling period, fails to help, and the
/// next one takes over.
#[test]
fn the_controller_climbs_the_ladder_one_rung_at_a_time() {
    // Two classifications with different legal sets, so the order is checked
    // on a path that includes growth and one that does not
    for bottleneck in [BottleneckKind::None, BottleneckKind::Io] {
        let c = controller(MeshReach::FULL);
        // Just over the objective, so the step size stays at one rung and the
        // whole ladder is walked rather than jumped
        let mut pressure = breaching(WorkloadClass::Interactive);
        pressure.pressure_seconds = pressure.slo_seconds * 1.5;

        let start = Instant::now();
        let mut sequence = Vec::new();
        for i in 0..ActuatorLevel::LADDER.len() {
            // One settling period per rung, which is what the controller waits
            // before deciding a lever did not work
            let now = start + SETTLE * (i as u32) * 2;
            let level = c.choose_actuator(&pressure, bottleneck, now).level;
            if sequence.last() != Some(&level) {
                sequence.push(level);
            }
        }

        assert_eq!(
            sequence,
            legal_ladder(bottleneck, MeshReach::FULL),
            "the ladder was not climbed in order against {bottleneck:?}"
        );
    }
}

/// The cheap relief stays applied while the ladder climbs past it, or the
/// dearer rung has to cover work the cheaper one was already handling.
#[test]
fn climbing_does_not_release_the_relief_already_in_force() {
    let c = controller(MeshReach::FULL);
    let mut pressure = breaching(WorkloadClass::Interactive);
    pressure.pressure_seconds = pressure.slo_seconds * 1.5;

    let start = Instant::now();
    let mut seen_above_reduce = false;
    for step in 0..ActuatorLevel::LADDER.len() {
        let now = start + SETTLE * (step as u32) * 2;
        let decision = c.choose_actuator(&pressure, BottleneckKind::None, now);
        assert_eq!(
            decision.dop_scale_pct, REDUCED_DOP_PCT,
            "parallelism was handed back at rung {:?}",
            decision.level
        );
        if decision.level > ActuatorLevel::DelayAdmission {
            seen_above_reduce = true;
            assert!(
                decision.delay > std::time::Duration::ZERO,
                "the admission delay was released at rung {:?}",
                decision.level
            );
        }
    }
    assert!(seen_above_reduce, "the walk never left the cheap rungs");
}

/// A node many times over its objective does not spend a settling period per
/// rung on its way to shedding.
#[test]
fn a_severe_breach_climbs_faster_than_a_mild_one() {
    let mild = controller(MeshReach::FULL);
    let severe = controller(MeshReach::FULL);
    let mut small = breaching(WorkloadClass::Interactive);
    small.pressure_seconds = small.slo_seconds * 1.2;
    let mut large = breaching(WorkloadClass::Interactive);
    large.pressure_seconds = large.slo_seconds * 64.0;

    let start = Instant::now();
    let mut mild_level = ActuatorLevel::Steady;
    let mut severe_level = ActuatorLevel::Steady;
    for step in 0..3 {
        let now = start + SETTLE * (step as u32) * 2;
        mild_level = mild
            .choose_actuator(&small, BottleneckKind::None, now)
            .level;
        severe_level = severe
            .choose_actuator(&large, BottleneckKind::None, now)
            .level;
    }
    assert!(
        severe_level > mild_level,
        "a breach 64x the objective reached {severe_level:?} while a mild one reached {mild_level:?}"
    );
    assert!(
        severe_level >= ActuatorLevel::Shed,
        "a node 64 times over its objective was still not shedding: {severe_level:?}"
    );
}

/// Relief is given back one rung at a time. Releasing it all at once puts the
/// node straight back into the breach it just left.
#[test]
fn relief_is_withdrawn_gradually_rather_than_all_at_once() {
    let c = controller(MeshReach::FULL);
    let mut pressure = breaching(WorkloadClass::Interactive);
    pressure.pressure_seconds = pressure.slo_seconds * 1.5;

    let start = Instant::now();
    // Climb to the middle of the ladder
    let mut climbed = ActuatorLevel::Steady;
    for step in 0..4 {
        let now = start + SETTLE * (step as u32) * 2;
        climbed = c
            .choose_actuator(&pressure, BottleneckKind::None, now)
            .level;
    }
    assert!(climbed >= ActuatorLevel::TrimMemory, "{climbed:?}");

    // The class recovers, and the ladder comes down a rung at a time
    let mut calm = pressure;
    calm.pressure_seconds = calm.slo_seconds * 0.1;
    let mut descent = vec![climbed];
    let mut now = start + SETTLE * 8;
    for _ in 0..ActuatorLevel::LADDER.len() + 1 {
        now += SETTLE * 2;
        let level = c.choose_actuator(&calm, BottleneckKind::None, now).level;
        if descent.last() != Some(&level) {
            descent.push(level);
        }
    }
    assert_eq!(
        *descent.last().expect("a descent"),
        ActuatorLevel::Steady,
        "the ladder never came all the way down: {descent:?}"
    );
    for pair in descent.windows(2) {
        assert!(
            pair[1] < pair[0],
            "the descent went up at {:?} -> {:?}",
            pair[0],
            pair[1]
        );
    }
    assert!(
        descent.len() > 2,
        "relief was released in one step: {descent:?}"
    );
}

/// Whatever the situation, the rung chosen is the cheapest one that is both
/// legal for the bottleneck and reachable in this deployment.
///
/// Exhaustive over both dials, which is stronger than walking one path: it
/// catches a gating rule that is right for the case somebody thought about and
/// wrong for a combination nobody did.
#[test]
fn the_chosen_rung_is_always_the_cheapest_legal_reachable_one() {
    let reaches = [
        MeshReach::NONE,
        MeshReach::FULL,
        MeshReach {
            can_take_warm_node: true,
            can_provision: false,
        },
        MeshReach {
            can_take_warm_node: false,
            can_provision: true,
        },
    ];
    for reach in reaches {
        for bottleneck in BottleneckKind::ALL {
            for class in [WorkloadClass::Interactive, WorkloadClass::Bulk] {
                // A controller per combination. The rung a class sits on
                // persists across ticks by design, so reusing one here would
                // be asking where a ladder that has already been climbed
                // starts
                let c = controller(reach);
                let decision = c.choose_actuator(&breaching(class), bottleneck, Instant::now());
                assert_eq!(
                    decision.level,
                    expected_rung(bottleneck, reach),
                    "class {class:?} bottleneck {bottleneck:?} reach {reach:?}"
                );
                assert!(
                    !decision.reason.is_empty(),
                    "a decision with no stated reason is not auditable"
                );
            }
        }
    }
}

/// Conflict rises with concurrency, so every rung that adds concurrency is
/// masked and the ones that remove it are all that is left.
#[test]
fn conflict_never_reaches_a_rung_that_adds_concurrency() {
    let c = controller(MeshReach::FULL);
    let decision = c.choose_actuator(
        &breaching(WorkloadClass::Interactive),
        BottleneckKind::OccContention,
        Instant::now(),
    );
    assert!(
        matches!(
            decision.level,
            ActuatorLevel::ReduceDop | ActuatorLevel::DelayAdmission | ActuatorLevel::Shed
        ),
        "picked {:?} against conflict",
        decision.level
    );
    for level in [
        ActuatorLevel::GrowWorkers,
        ActuatorLevel::WarmPoolTake,
        ActuatorLevel::ProvisionNode,
        ActuatorLevel::TrimMemory,
        ActuatorLevel::ForceSpill,
    ] {
        assert!(
            !level.legal_for(BottleneckKind::OccContention),
            "{level:?} is legal against conflict, which would make the conflict worse"
        );
    }
}

/// A hot key does not spread across nodes, so the mesh rungs are masked while
/// the skew stands. Growth is masked with them: more work in flight against
/// one row is more work queued behind the same row. So is spilling, which
/// buys memory nobody is short of and pays for it in device time.
#[test]
fn a_hot_key_masks_the_rungs_that_would_pile_onto_the_same_row() {
    for level in ActuatorLevel::LADDER {
        let legal = level.legal_for(BottleneckKind::HotPartition);
        let masked = matches!(
            level,
            ActuatorLevel::ForceSpill
                | ActuatorLevel::GrowWorkers
                | ActuatorLevel::WarmPoolTake
                | ActuatorLevel::ProvisionNode
        );
        assert_eq!(
            legal,
            !masked,
            "{level:?} against a hot partition should be {}",
            if masked { "masked" } else { "allowed" }
        );
    }
    // The relief that does help is still reachable
    for level in [
        ActuatorLevel::ReduceDop,
        ActuatorLevel::DelayAdmission,
        ActuatorLevel::Shed,
    ] {
        assert!(level.legal_for(BottleneckKind::HotPartition));
    }
}

/// Growth is the one rung that is legal against exactly one classification.
///
/// It raises in-flight work past what the cores afford, which only ever helps
/// when the work is parked on storage rather than computing. Every other
/// diagnosis makes it a way of adding competition.
#[test]
fn growth_is_legal_only_where_the_cores_are_idle() {
    for bottleneck in BottleneckKind::ALL {
        assert_eq!(
            ActuatorLevel::GrowWorkers.legal_for(bottleneck),
            bottleneck == BottleneckKind::Io,
            "growth against {bottleneck:?}"
        );
    }
}

/// A deployment that cannot provision must not answer pressure with a rung
/// nothing will act on. It sheds instead, which is a true statement about what
/// the node is doing.
#[test]
fn a_node_with_no_provisioner_never_names_a_mesh_rung() {
    let c = controller(MeshReach::NONE);
    for bottleneck in BottleneckKind::ALL {
        let decision =
            c.choose_actuator(&breaching(WorkloadClass::Bulk), bottleneck, Instant::now());
        assert!(
            decision.level < ActuatorLevel::WarmPoolTake,
            "a node with nowhere to grow chose {:?} against {bottleneck:?}",
            decision.level
        );
    }
}

/// Writers waiting on a device are not waiting on compute, so the rungs that
/// add compute are masked and spilling is reached instead.
#[test]
fn an_fsync_bound_node_is_not_given_more_compute() {
    let c = controller(MeshReach::FULL);
    let decision = c.choose_actuator(
        &breaching(WorkloadClass::Bulk),
        BottleneckKind::FsyncBound,
        Instant::now(),
    );
    assert_eq!(decision.level, ActuatorLevel::ReduceDop);
    for level in [
        ActuatorLevel::GrowWorkers,
        ActuatorLevel::WarmPoolTake,
        ActuatorLevel::ProvisionNode,
    ] {
        assert!(!level.legal_for(BottleneckKind::FsyncBound));
    }
}

/// A class inside its objective provokes nothing at all, whatever is
/// happening elsewhere on the node.
#[test]
fn nothing_is_done_about_a_class_that_is_meeting_its_objective() {
    let c = controller(MeshReach::FULL);
    let mut calm = breaching(WorkloadClass::Interactive);
    calm.pressure_seconds = 0.001;
    for bottleneck in BottleneckKind::ALL {
        assert_eq!(
            c.choose_actuator(&calm, bottleneck, Instant::now()).level,
            ActuatorLevel::Steady,
            "acted on a calm class because of {bottleneck:?}"
        );
    }
}

/// Every rung the controller names locally must change something the engine
/// reads. A rung that is chosen, logged, and applied to nothing is the failure
/// this whole ladder exists to avoid.
#[test]
fn every_local_rung_moves_something_the_engine_reads() {
    use zyron_pressure::pressure::{
        PARALLEL_GROWTH_MAX_PCT, PARALLEL_SCALE_FULL_PCT, QUERY_MEMORY_FULL_PCT,
        QUERY_MEMORY_MIN_PCT, SPILL_THRESHOLD_FORCED_PCT, SPILL_THRESHOLD_FULL_PCT,
    };

    // Storage bound, which is the one classification whose legal set covers
    // both the memory rung and the growth rung
    let bottleneck = BottleneckKind::Io;
    let c = isolated_controller(MeshReach::FULL, 8);
    let capacity = c.capacity();
    let base = capacity.base_total();
    let mut pressure = breaching(WorkloadClass::Interactive);
    pressure.pressure_seconds = pressure.slo_seconds * 1.5;

    let start = Instant::now();
    let mut reached = Vec::new();
    // Walk the ladder, recording what each rung actually changed
    for i in 0..ActuatorLevel::LADDER.len() {
        let level = step_against(&c, &pressure, bottleneck, start + SETTLE * (i as u32) * 2);
        reached.push((
            level,
            capacity.scale_pct(),
            c.query_memory_pct(),
            capacity.growth_pct(),
            c.spill_threshold_pct(),
        ));
    }
    let climbed: Vec<ActuatorLevel> = reached.iter().map(|r| r.0).collect();
    assert!(
        climbed.contains(&ActuatorLevel::GrowWorkers),
        "the walk never reached the growth rung: {climbed:?}"
    );

    // Each lever is in force exactly from its own rung upward, and each is
    // checked against the same legality rule the controller applies
    for (level, scale, memory, growth, spill) in &reached {
        let dop_expected = if *level >= ActuatorLevel::ReduceDop
            && ActuatorLevel::ReduceDop.legal_for(bottleneck)
        {
            REDUCED_DOP_PCT as u32
        } else {
            PARALLEL_SCALE_FULL_PCT
        };
        assert_eq!(*scale, dop_expected, "parallel scale at {level:?}");

        let memory_expected = if *level >= ActuatorLevel::TrimMemory
            && ActuatorLevel::TrimMemory.legal_for(bottleneck)
        {
            QUERY_MEMORY_MIN_PCT
        } else {
            QUERY_MEMORY_FULL_PCT
        };
        assert_eq!(*memory, memory_expected, "query memory share at {level:?}");

        let spill_expected = if *level >= ActuatorLevel::ForceSpill
            && ActuatorLevel::ForceSpill.legal_for(bottleneck)
        {
            SPILL_THRESHOLD_FORCED_PCT
        } else {
            SPILL_THRESHOLD_FULL_PCT
        };
        assert_eq!(*spill, spill_expected, "spill point at {level:?}");

        let growth_expected = if *level >= ActuatorLevel::GrowWorkers
            && ActuatorLevel::GrowWorkers.legal_for(bottleneck)
        {
            PARALLEL_GROWTH_MAX_PCT
        } else {
            PARALLEL_SCALE_FULL_PCT
        };
        assert_eq!(*growth, growth_expected, "parallel budget at {level:?}");
    }
    assert_eq!(
        capacity.total(),
        base * 2,
        "growth did not reach the two times cap"
    );

    // And everything is handed back when the class recovers
    let mut calm = pressure;
    calm.pressure_seconds = calm.slo_seconds * 0.01;
    let mut now = start + SETTLE * 64;
    for _ in 0..ActuatorLevel::LADDER.len() + 2 {
        now += SETTLE * 2;
        step_against(&c, &calm, bottleneck, now);
    }
    assert_eq!(capacity.scale_pct(), PARALLEL_SCALE_FULL_PCT);
    assert_eq!(c.query_memory_pct(), QUERY_MEMORY_FULL_PCT);
    assert_eq!(c.spill_threshold_pct(), SPILL_THRESHOLD_FULL_PCT);
    assert_eq!(
        capacity.total(),
        base,
        "the grown budget was never given back"
    );
}

/// The trim rung hands a smaller allowance to a query starting now, and the
/// arithmetic is a share of what the session configured rather than a
/// constant the operator never set.
#[test]
fn the_trim_rung_scales_the_configured_allowance() {
    use zyron_pressure::pressure::{QUERY_MEMORY_FULL_PCT, QUERY_MEMORY_MIN_PCT};

    let c = isolated_controller(MeshReach::NONE, 8);
    let configured = 1_024u64 * 1024 * 1024;
    assert_eq!(c.query_memory_allowance(configured), configured);

    let mut pressure = breaching(WorkloadClass::Bulk);
    pressure.pressure_seconds = pressure.slo_seconds * 1.5;
    let start = Instant::now();
    for i in 0..4 {
        step_against(
            &c,
            &pressure,
            BottleneckKind::Memory,
            start + SETTLE * (i as u32) * 2,
        );
    }
    assert_eq!(c.query_memory_pct(), QUERY_MEMORY_MIN_PCT);
    let trimmed = c.query_memory_allowance(configured);
    assert_eq!(
        trimmed,
        configured * QUERY_MEMORY_MIN_PCT as u64 / QUERY_MEMORY_FULL_PCT as u64
    );

    // A session with no configured limit is not given one by the trim
    assert_eq!(c.query_memory_allowance(0), 0);
}

/// Growth is refused wherever the cores are the constraint, because adding
/// in-flight workers there only adds competition for them.
#[test]
fn the_budget_never_grows_against_a_bottleneck_that_is_not_waiting() {
    use zyron_pressure::pressure::PARALLEL_SCALE_FULL_PCT;

    // Every classification except the one that means the cores are idle
    for bottleneck in BottleneckKind::ALL
        .into_iter()
        .filter(|b| *b != BottleneckKind::Io)
    {
        let c = isolated_controller(MeshReach::FULL, 8);
        let mut pressure = breaching(WorkloadClass::Bulk);
        pressure.pressure_seconds = pressure.slo_seconds * 64.0;
        let start = Instant::now();
        for i in 0..8 {
            step_against(&c, &pressure, bottleneck, start + SETTLE * (i as u32) * 2);
        }
        assert_eq!(
            c.capacity().growth_pct(),
            PARALLEL_SCALE_FULL_PCT,
            "the budget grew against {bottleneck:?}"
        );
    }
}
