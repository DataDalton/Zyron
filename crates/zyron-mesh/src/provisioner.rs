//! The provisioning interface, from the mesh's side.
//!
//! ## Why this re-exports rather than defines
//!
//! The plan for this crate was that the driver interface would be defined
//! here, on the reasoning that acquiring hardware is a mesh concern and the
//! pressure substrate had no drivers to move. The second half of that turned
//! out not to hold: `zyron-pressure` already carries a full provisioner, with
//! a driver trait, a kind registry, the mesh config section, and the
//! scale-to-zero readiness check, all of it in use.
//!
//! Moving it up here would put `zyron-wire` in the position of depending on
//! this crate, because the `zyron_sys.pressure.*` views read provision
//! latency and scale-to-zero readiness, and the wire layer is supposed to have
//! no mesh concerns at all. Splitting the module in half so the views keep
//! their half would be a refactor of the pressure internals, which is exactly
//! what the move was meant to avoid.
//!
//! So the definition stays where the controller and the views can reach it,
//! and this module is where the mesh names it. Phase 21.1's drivers implement
//! [`ProvisionerDriver`] and install themselves through
//! [`ProvisionerRegistry`], both from here, so a reader looking for
//! provisioning in the mesh crate finds it.

pub use zyron_pressure::provisioner::{
    MeshSection, ProvisionRequest, ProvisionTicket, ProvisionerCapabilities, ProvisionerDriver,
    ProvisionerKind, ProvisionerRegistry, ReclaimRequest, ReclaimVerdict, ResumeEstimate,
    ScaleToZeroInputs, UnreachableProvisioner, assert_scale_to_zero_ready, estimate_resume,
};

#[cfg(test)]
mod tests {
    use super::*;

    /// With no driver installed the registry hands back one that refuses,
    /// which is what makes the top rung answer honestly on a node that cannot
    /// provision.
    #[test]
    fn the_default_driver_cannot_provision() {
        let driver = ProvisionerRegistry::new().active();
        assert!(!driver.capabilities().can_provision);
    }
}
