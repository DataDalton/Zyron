//! The mesh's answer to the two rungs it owns.
//!
//! The ladder climbs to a rung, asks whatever registered for it, and takes the
//! answer. This is what answers, and what it does depends on whether a
//! scheduler has been installed:
//!
//! - With one, claiming a warm node takes a node out of the pool, and
//!   provisioning asks the driver for capacity. Both report what they did.
//! - Without one, both report that the rung cannot be reached, and the ladder
//!   carries on to shedding.
//!
//! The second case is not a placeholder. A single node, or one whose
//! deployment has no control plane this process can reach, genuinely cannot
//! perform these, and telling the controller so is what makes it refuse work
//! instead of waiting for capacity that is not coming.

use std::sync::{Arc, OnceLock};

use zyron_pressure::extension::{ActuatorExtension, ActuatorResult, ExtensionRegistry};
use zyron_pressure::pressure::{ActuatorDecision, ActuatorLevel};
use zyron_pressure::provisioner::ProvisionRequest;

use crate::scheduler::MeshScheduler;

/// Why the mesh rungs cannot be performed when nothing is installed.
const NOT_INSTALLED: &str = "no mesh scheduler is installed on this node";

/// The rungs this crate answers for.
const MESH_RUNGS: &[ActuatorLevel] = &[ActuatorLevel::WarmPoolTake, ActuatorLevel::ProvisionNode];

/// The scheduler the actuator drives, once one exists.
static SCHEDULER: OnceLock<Arc<MeshScheduler>> = OnceLock::new();

/// Answers the mesh rungs on behalf of the scheduler.
pub struct MeshActuator;

impl MeshActuator {
    /// The installed scheduler, or None on a node that has no mesh.
    pub fn scheduler() -> Option<&'static Arc<MeshScheduler>> {
        SCHEDULER.get()
    }
}

impl ActuatorExtension for MeshActuator {
    fn name(&self) -> &'static str {
        "zyron-mesh"
    }

    fn handles(&self) -> &'static [ActuatorLevel] {
        MESH_RUNGS
    }

    fn try_actuate(&self, decision: &ActuatorDecision) -> ActuatorResult {
        if !MESH_RUNGS.contains(&decision.level) {
            return ActuatorResult::NotApplicable;
        }
        let Some(scheduler) = MeshActuator::scheduler() else {
            return ActuatorResult::NotAvailable {
                reason: format!(
                    "{}: {} cannot be reached",
                    NOT_INSTALLED,
                    decision.level.as_str()
                ),
            };
        };

        match decision.level {
            ActuatorLevel::WarmPoolTake => match scheduler.take_warm_node() {
                Some(node) => ActuatorResult::Applied {
                    detail: format!("claimed the warm node {}", node.name),
                },
                None => ActuatorResult::NotAvailable {
                    reason: "the warm pool is empty, so there is no node to claim".into(),
                },
            },
            ActuatorLevel::ProvisionNode => {
                // One node per decision. The ladder stays on this rung while
                // pressure holds, so a shortage that needs two nodes asks
                // twice and each ask is measured against what the first one
                // did, which is what stops a spike from buying a fleet
                let request = ProvisionRequest {
                    nodes: 1,
                    class: decision.class,
                    pressure_seconds: decision.pressure_seconds,
                    projected_pressure_seconds: decision.pressure_seconds,
                    requested_by: scheduler.local().node_id,
                };
                match scheduler.provision(&request) {
                    Ok(ticket) => ActuatorResult::Applied {
                        detail: format!(
                            "asked for {} node(s), ticket {}, expected in {}s",
                            ticket.nodes,
                            ticket.external_id,
                            ticket.expected.as_secs()
                        ),
                    },
                    Err(reason) => ActuatorResult::NotAvailable { reason },
                }
            }
            _ => ActuatorResult::NotApplicable,
        }
    }
}

static MESH_ACTUATOR: MeshActuator = MeshActuator;

/// Registers the mesh with the pressure ladder.
///
/// Called once from the server at startup, before any scheduler exists.
/// Registering an extension that reports the rungs as unreachable is not the
/// same as registering nothing: the ladder asks either way and gets the same
/// verdict, but with this installed the reason names the mesh, so an operator
/// reading the pressure views is told that the rung exists and why it did not
/// fire rather than being left to work out that nothing on the node claims it.
pub fn register() {
    ExtensionRegistry::global().install(&MESH_ACTUATOR);
}

/// Installs the scheduler the rungs drive.
///
/// Returns false when one is already installed. Two schedulers on one node
/// would each hold their own warm pool and each ask the provisioner, so the
/// second is refused rather than replacing the first.
pub fn install_scheduler(scheduler: Arc<MeshScheduler>) -> bool {
    SCHEDULER.set(scheduler).is_ok()
}

#[cfg(test)]
mod tests {
    use super::*;
    use zyron_pressure::pressure::{BottleneckKind, WorkloadClass};

    fn decision(level: ActuatorLevel) -> ActuatorDecision {
        ActuatorDecision {
            class: WorkloadClass::Interactive,
            level,
            bottleneck: BottleneckKind::Cpu,
            dop_scale_pct: 100,
            delay: std::time::Duration::ZERO,
            pressure_seconds: 1.0,
            slo_seconds: 0.5,
            reason: "test",
        }
    }

    /// A rung the node performs itself is never claimed by the mesh.
    #[test]
    fn a_local_rung_is_not_the_mesh_actuators_business() {
        for level in [
            ActuatorLevel::ReduceDop,
            ActuatorLevel::TrimMemory,
            ActuatorLevel::ForceSpill,
            ActuatorLevel::Shed,
            ActuatorLevel::GrowWorkers,
        ] {
            assert_eq!(
                MeshActuator.try_actuate(&decision(level)),
                ActuatorResult::NotApplicable,
                "the mesh claimed {level:?}"
            );
        }
    }

    /// Both mesh rungs answer, and whichever answer they give names the rung
    /// or what was done.
    ///
    /// Written to hold whether or not a scheduler has been installed by
    /// another test in this process, because the installation is a process
    /// global and the tests share one.
    #[test]
    fn both_mesh_rungs_answer_for_themselves() {
        for level in [ActuatorLevel::WarmPoolTake, ActuatorLevel::ProvisionNode] {
            let answer = MeshActuator.try_actuate(&decision(level));
            assert_ne!(
                answer,
                ActuatorResult::NotApplicable,
                "{level:?} was not answered by the mesh"
            );
            let detail = answer.detail().expect("an answer says something");
            assert!(!detail.is_empty(), "{level:?} answered with nothing");
        }
    }
}
