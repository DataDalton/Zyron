//! Where the ladder hands off to something this crate cannot reach.
//!
//! The rungs below shedding are all things one node does to itself: fewer
//! workers, held arrivals, a smaller memory ceiling, earlier spilling. The
//! two above it are not. Claiming a warm node and asking a provisioner for
//! hardware both require talking to something outside this process, and the
//! crate that knows how to do that sits above this one.
//!
//! So the ladder does not perform them. It asks, and takes the answer:
//!
//! - `Applied` means the extension did it, and the controller stays on that
//!   rung while it settles.
//! - `NotApplicable` means the extension exists and this is not its rung.
//! - `NotAvailable` means nothing can perform it here, which is the answer
//!   when no extension is registered at all.
//!
//! The last one is the important one, because it is the default. A node with
//! no mesh registered climbs to the rung, is told the rung is not there, and
//! carries on to shedding, which is exactly what a single node should do when
//! it is out of local relief. Nothing waits on a reply that is not coming.

use std::sync::RwLock;

use crate::pressure::{ActuatorDecision, ActuatorLevel};

/// What an extension did with a rung it was offered.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ActuatorResult {
    /// The rung was performed
    Applied {
        /// What was done, for the decision record and the views
        detail: String,
    },
    /// This extension does not handle this rung
    NotApplicable,
    /// Nothing here can perform this rung
    NotAvailable {
        /// Why, so an operator reading the views is told rather than left to
        /// infer it from a rung that never takes effect
        reason: String,
    },
}

impl ActuatorResult {
    pub fn applied(&self) -> bool {
        matches!(self, ActuatorResult::Applied { .. })
    }

    /// What to show an operator, or None when the extension had nothing to
    /// say about this rung.
    pub fn detail(&self) -> Option<&str> {
        match self {
            ActuatorResult::Applied { detail } => Some(detail),
            ActuatorResult::NotAvailable { reason } => Some(reason),
            ActuatorResult::NotApplicable => None,
        }
    }
}

/// Something that can perform a rung this crate cannot.
///
/// One method, because the ladder has one question. An implementation that
/// covers several rungs answers for each of them and returns
/// `NotApplicable` for the rest.
pub trait ActuatorExtension: Send + Sync {
    /// A short name, for the decision record.
    fn name(&self) -> &'static str;

    /// The rungs this extension answers for, so the registry can skip it
    /// without a call.
    fn handles(&self) -> &'static [ActuatorLevel];

    /// Performs the rung, or says why it did not.
    fn try_actuate(&self, decision: &ActuatorDecision) -> ActuatorResult;
}

/// The extensions registered on this node.
///
/// A registry rather than one slot, because the two rungs above shedding are
/// separate capabilities that a deployment can have separately: a fixed pool
/// with a warm spare can claim it and can never provision, and a cloud
/// account with no warm pool is the other way round.
pub struct ExtensionRegistry {
    extensions: RwLock<Vec<&'static dyn ActuatorExtension>>,
}

static REGISTRY: ExtensionRegistry = ExtensionRegistry {
    extensions: RwLock::new(Vec::new()),
};

impl ExtensionRegistry {
    /// The process registry, which is what the controller asks.
    pub fn global() -> &'static ExtensionRegistry {
        &REGISTRY
    }

    /// Registers an extension. Called once, at startup, by whatever crate
    /// can perform the rungs it names.
    pub fn install(&self, extension: &'static dyn ActuatorExtension) {
        let mut guard = match self.extensions.write() {
            Ok(guard) => guard,
            Err(poisoned) => poisoned.into_inner(),
        };
        if guard.iter().any(|e| e.name() == extension.name()) {
            return;
        }
        guard.push(extension);
    }

    /// Whether anything registered handles this rung.
    ///
    /// Read while choosing, so a rung nothing can perform is not chosen and
    /// then reported as in force. The controller keeps climbing instead.
    pub fn handles(&self, level: ActuatorLevel) -> bool {
        let guard = match self.extensions.read() {
            Ok(guard) => guard,
            Err(poisoned) => poisoned.into_inner(),
        };
        guard.iter().any(|e| e.handles().contains(&level))
    }

    /// Offers a decision to every extension that handles its rung, in the
    /// order they registered, and returns the first real answer.
    ///
    /// With nothing registered this is `NotAvailable`, which is the whole
    /// point: the ladder gets a straight answer instead of a rung that
    /// silently does nothing.
    pub fn actuate(&self, decision: &ActuatorDecision) -> ActuatorResult {
        let guard = match self.extensions.read() {
            Ok(guard) => guard,
            Err(poisoned) => poisoned.into_inner(),
        };
        for extension in guard.iter() {
            if !extension.handles().contains(&decision.level) {
                continue;
            }
            match extension.try_actuate(decision) {
                ActuatorResult::NotApplicable => continue,
                answer => return answer,
            }
        }
        ActuatorResult::NotAvailable {
            reason: format!(
                "nothing on this node performs {}, so local relief is all there is",
                decision.level.as_str()
            ),
        }
    }

    /// Removes every registered extension. For tests, which would otherwise
    /// see whatever an earlier test in the same process installed.
    pub fn clear(&self) {
        let mut guard = match self.extensions.write() {
            Ok(guard) => guard,
            Err(poisoned) => poisoned.into_inner(),
        };
        guard.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pressure::BottleneckKind;

    fn decision(level: ActuatorLevel) -> ActuatorDecision {
        ActuatorDecision {
            class: crate::pressure::WorkloadClass::Interactive,
            level,
            bottleneck: BottleneckKind::Cpu,
            dop_scale_pct: 100,
            delay: std::time::Duration::ZERO,
            pressure_seconds: 1.0,
            slo_seconds: 0.5,
            reason: "test",
        }
    }

    struct Warm;

    impl ActuatorExtension for Warm {
        fn name(&self) -> &'static str {
            "test-warm"
        }
        fn handles(&self) -> &'static [ActuatorLevel] {
            &[ActuatorLevel::WarmPoolTake]
        }
        fn try_actuate(&self, _decision: &ActuatorDecision) -> ActuatorResult {
            ActuatorResult::Applied {
                detail: "took a warm node".into(),
            }
        }
    }

    static WARM: Warm = Warm;

    /// With nothing registered the answer is that the rung is not there,
    /// which is what sends the ladder on to shedding.
    #[test]
    fn an_empty_registry_says_the_rung_is_not_available() {
        let registry = ExtensionRegistry {
            extensions: RwLock::new(Vec::new()),
        };
        let answer = registry.actuate(&decision(ActuatorLevel::ProvisionNode));
        assert!(matches!(answer, ActuatorResult::NotAvailable { .. }));
        assert!(!registry.handles(ActuatorLevel::ProvisionNode));
    }

    /// A registered extension answers for its own rungs and not for others.
    #[test]
    fn an_extension_answers_only_for_the_rungs_it_handles() {
        let registry = ExtensionRegistry {
            extensions: RwLock::new(Vec::new()),
        };
        registry.install(&WARM);
        assert!(registry.handles(ActuatorLevel::WarmPoolTake));
        assert!(!registry.handles(ActuatorLevel::ProvisionNode));
        assert!(
            registry
                .actuate(&decision(ActuatorLevel::WarmPoolTake))
                .applied()
        );
        assert!(
            !registry
                .actuate(&decision(ActuatorLevel::ProvisionNode))
                .applied()
        );
    }

    /// Installing the same extension twice leaves one, so a startup path that
    /// runs again does not double every answer.
    #[test]
    fn installing_twice_registers_once() {
        let registry = ExtensionRegistry {
            extensions: RwLock::new(Vec::new()),
        };
        registry.install(&WARM);
        registry.install(&WARM);
        let guard = registry.extensions.read().expect("registry");
        assert_eq!(guard.len(), 1);
    }
}
