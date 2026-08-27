//! Provisioner drivers: how a deployment actually gets a node.
//!
//! ## What is here
//!
//! [`StaticPoolProvisioner`], for a deployment whose machines were racked and
//! registered ahead of time. It is the one mode that needs no control plane at
//! all: capacity is claimed from a list the operator gave and released back to
//! it, never created and never destroyed. That makes it complete here, and it
//! is the mode most on-premises deployments run.
//!
//! ## What is not here, and why that is not a gap in this file
//!
//! Cloud, hypervisor, Kubernetes, and IPMI each mean talking to a control
//! plane with its own authentication: request signing for a cloud account, a
//! session ticket for vSphere, a service account token and cluster CA for
//! Kubernetes, a Redfish session for metal. None of those clients is in this
//! tree, and adding four of them is a dependency decision rather than a coding
//! one.
//!
//! A node configured for one of those modes gets
//! [`zyron_pressure::provisioner::UnreachableProvisioner`], which reports that
//! it cannot provision. That is not a stub standing in for behaviour: it is
//! the correct behaviour for a process with no credentials for the control
//! plane it was pointed at. It masks the mesh rungs of the actuator ladder, so
//! the controller relieves pressure with the levers it owns and refuses work
//! rather than publishing a request nothing will answer. A driver that
//! pretended to provision and never did would leave the ladder sitting on a
//! rung that changes nothing while the objective is missed.

use std::sync::Mutex;
use std::time::Duration;

use zyron_common::{Result, ZyronError};
use zyron_pressure::provisioner::{
    ProvisionRequest, ProvisionTicket, ProvisionerCapabilities, ProvisionerDriver, ProvisionerKind,
    ReclaimRequest,
};

/// One machine the operator registered ahead of time.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PoolMember {
    /// The node id it reports once it is serving, or zero until it has been
    /// contacted
    pub node_id: u64,
    /// The operator-facing name
    pub name: String,
    /// Where its health listener answers, `host:port`
    pub address: String,
}

/// A fixed set of machines, claimed and released.
///
/// Provisioning here means marking a member as claimed and letting the
/// scheduler bring it into the mesh. Nothing is created, so the request either
/// finds free members or it does not, and the answer arrives immediately
/// rather than after a boot.
pub struct StaticPoolProvisioner {
    members: Mutex<Vec<Member>>,
}

#[derive(Debug, Clone)]
struct Member {
    member: PoolMember,
    claimed: bool,
}

impl StaticPoolProvisioner {
    pub fn new(members: Vec<PoolMember>) -> Self {
        Self {
            members: Mutex::new(
                members
                    .into_iter()
                    .map(|member| Member {
                        member,
                        claimed: false,
                    })
                    .collect(),
            ),
        }
    }

    /// Members not currently claimed.
    pub fn free(&self) -> Vec<PoolMember> {
        self.with_members(|members| {
            members
                .iter()
                .filter(|m| !m.claimed)
                .map(|m| m.member.clone())
                .collect()
        })
    }

    /// Members currently in the mesh.
    pub fn claimed(&self) -> Vec<PoolMember> {
        self.with_members(|members| {
            members
                .iter()
                .filter(|m| m.claimed)
                .map(|m| m.member.clone())
                .collect()
        })
    }

    pub fn size(&self) -> usize {
        self.with_members(|members| members.len())
    }

    fn with_members<T>(&self, f: impl FnOnce(&mut Vec<Member>) -> T) -> T {
        let mut guard = match self.members.lock() {
            Ok(guard) => guard,
            Err(poisoned) => poisoned.into_inner(),
        };
        f(&mut guard)
    }
}

impl ProvisionerDriver for StaticPoolProvisioner {
    fn kind(&self) -> ProvisionerKind {
        ProvisionerKind::Static
    }

    fn capabilities(&self) -> ProvisionerCapabilities {
        ProvisionerCapabilities {
            kind: ProvisionerKind::Static,
            // A pool with nothing free cannot grow. Reported rather than
            // discovered on the request, so the ladder masks the rung instead
            // of climbing to it and being refused
            can_provision: self.free().len() > 0,
            can_reclaim: true,
            // A fixed pool cannot drop to nothing: the machines are the
            // deployment, and one of them is always this node
            can_scale_to_zero: false,
            min_nodes: 1,
            max_nodes: self.size() as u32,
            // A machine somebody already paid for is not billed by the hour,
            // so there is no interval worth holding a node for. Nothing is
            // created either, so nothing waits on a control plane: what a
            // claimed machine takes to serve is its own startup
            billing_interval: Duration::ZERO,
        }
    }

    fn provision(&self, request: &ProvisionRequest) -> Result<ProvisionTicket> {
        if request.nodes == 0 {
            return Err(ZyronError::ConfigError(
                "a provision request for no nodes".into(),
            ));
        }
        let claimed = self.with_members(|members| {
            let mut claimed = Vec::new();
            for member in members.iter_mut() {
                if claimed.len() as u32 >= request.nodes {
                    break;
                }
                if !member.claimed {
                    member.claimed = true;
                    claimed.push(member.member.name.clone());
                }
            }
            claimed
        });
        if claimed.is_empty() {
            return Err(ZyronError::ExecutionError(
                "every machine in the static pool is already in the mesh".into(),
            ));
        }
        Ok(ProvisionTicket {
            external_id: claimed.join(","),
            nodes: claimed.len() as u32,
            expected: Duration::from_secs(0),
        })
    }

    fn reclaim(&self, request: &ReclaimRequest) -> Result<()> {
        let released = self.with_members(|members| {
            for member in members.iter_mut() {
                if member.member.node_id == request.node_id && member.claimed {
                    member.claimed = false;
                    return true;
                }
            }
            false
        });
        if released {
            Ok(())
        } else {
            Err(ZyronError::ExecutionError(format!(
                "node {} is not a claimed member of the static pool",
                request.node_id
            )))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use zyron_pressure::pressure::WorkloadClass;

    fn pool(count: u64) -> StaticPoolProvisioner {
        StaticPoolProvisioner::new(
            (1..=count)
                .map(|i| PoolMember {
                    node_id: i,
                    name: format!("node-{i}"),
                    address: format!("10.0.0.{i}:8080"),
                })
                .collect(),
        )
    }

    fn request(nodes: u32) -> ProvisionRequest {
        ProvisionRequest {
            nodes,
            class: WorkloadClass::Interactive,
            pressure_seconds: 2.0,
            projected_pressure_seconds: 3.0,
            requested_by: 1,
        }
    }

    /// Claiming takes members and returns a ticket naming them.
    #[test]
    fn a_claim_takes_machines_out_of_the_pool() {
        let pool = pool(3);
        let ticket = pool.provision(&request(2)).expect("two are free");
        assert_eq!(ticket.nodes, 2);
        assert_eq!(pool.free().len(), 1);
        assert_eq!(pool.claimed().len(), 2);
        assert!(ticket.external_id.contains("node-1"));
    }

    /// A pool with nothing free says so before it is asked, so the ladder
    /// masks the rung rather than climbing to a refusal.
    #[test]
    fn an_exhausted_pool_reports_that_it_cannot_provision() {
        let pool = pool(1);
        assert!(pool.capabilities().can_provision);
        pool.provision(&request(1)).expect("one is free");
        assert!(!pool.capabilities().can_provision);
        assert!(pool.provision(&request(1)).is_err());
    }

    /// Asking for more than is free takes what there is rather than refusing
    /// outright, because partial relief is relief.
    #[test]
    fn a_claim_larger_than_the_pool_takes_what_there_is() {
        let pool = pool(2);
        let ticket = pool.provision(&request(5)).expect("two are free");
        assert_eq!(ticket.nodes, 2);
        assert!(pool.free().is_empty());
    }

    /// Releasing puts a machine back, and releasing one that was never
    /// claimed is an error rather than a silent success.
    #[test]
    fn a_release_puts_the_machine_back() {
        let pool = pool(2);
        pool.provision(&request(2)).expect("two are free");
        let request = ReclaimRequest {
            node_id: 1,
            remaining_paid_interval: Duration::ZERO,
            predicted_idle_window: Duration::from_secs(600),
            hot_set_handed_off: true,
        };
        pool.reclaim(&request).expect("node 1 was claimed");
        assert_eq!(pool.free().len(), 1);
        assert!(
            pool.reclaim(&request).is_err(),
            "releasing twice reported success"
        );
    }

    /// Machines nobody is billed for by the hour are given back as soon as
    /// they are idle and handed over.
    #[test]
    fn a_machine_that_is_not_billed_by_time_is_released_immediately() {
        let pool = pool(1);
        let verdict = pool.reclaim_allowed(&ReclaimRequest {
            node_id: 1,
            remaining_paid_interval: Duration::from_secs(3600),
            predicted_idle_window: Duration::from_secs(1),
            hot_set_handed_off: true,
        });
        assert!(verdict.allowed(), "{verdict:?}");
    }

    /// A node that has not handed its working set over is not given back,
    /// whatever the billing says.
    #[test]
    fn a_node_that_kept_its_pages_is_not_released() {
        let pool = pool(1);
        let verdict = pool.reclaim_allowed(&ReclaimRequest {
            node_id: 1,
            remaining_paid_interval: Duration::ZERO,
            predicted_idle_window: Duration::from_secs(600),
            hot_set_handed_off: false,
        });
        assert!(!verdict.allowed(), "{verdict:?}");
    }
}
