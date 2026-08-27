//! How a node acquires and releases hardware, and the one place that reads
//! how this node was registered.
//!
//! The controller decides *that* capacity is needed and how urgently. It never
//! decides *how* to get it, because that is the only part of the substrate
//! that differs between a cloud account, a vSphere cluster, a Kubernetes
//! namespace, a rack of IPMI-managed metal, and a fixed pool of machines
//! somebody already racked. Keeping the mode in one module is what stops a
//! static deployment from quietly losing its elasticity: every other module
//! sees a driver with capabilities, not a mode with special cases.
//!
//! Two things live here that look like policy rather than plumbing, and both
//! belong to whoever owns the hardware rather than to the controller:
//!
//! - Whether a node may be reclaimed at all. A cloud node inside an interval
//!   that has already been paid for costs nothing to keep and everything to
//!   re-provision, so reclaiming it early is a loss. A node in a rack has no
//!   interval, so the question does not arise.
//! - What has to be true before a node may go to zero. Resume time is what
//!   the operator experiences, and resume time is decided by how stale the
//!   checkpoint is and how much of the working set has to be read back, not
//!   by how much write-ahead log accumulated.

use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Duration;

use serde::{Deserialize, Serialize};

use zyron_common::error::{Result, ZyronError};

// ---------------------------------------------------------------------------
// Modes and drivers
// ---------------------------------------------------------------------------

/// How nodes come into existence in this deployment.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ProvisionerKind {
    /// Nothing can be added or removed. A single node, or a mesh whose
    /// membership is managed entirely outside Zyron
    None,
    /// A fixed set of machines registered ahead of time. Capacity is claimed
    /// from the pool and released back to it, never created or destroyed
    Static,
    /// A cloud account. Instances are created on demand and billed by time
    Cloud,
    /// vSphere, Proxmox, KVM, or OpenStack. Instances are cloned from a
    /// template against a finite host pool
    Hypervisor,
    /// A Kubernetes namespace. Capacity is a replica count, and the scheduler
    /// decides where it lands
    Kubernetes,
    /// Bare metal reached over IPMI or Redfish. Capacity is a machine that has
    /// to boot
    Ipmi,
}

impl ProvisionerKind {
    pub const ALL: [ProvisionerKind; 6] = [
        ProvisionerKind::None,
        ProvisionerKind::Static,
        ProvisionerKind::Cloud,
        ProvisionerKind::Hypervisor,
        ProvisionerKind::Kubernetes,
        ProvisionerKind::Ipmi,
    ];

    pub const COUNT: usize = 6;

    pub const fn index(self) -> usize {
        self as usize
    }

    pub const fn as_str(self) -> &'static str {
        match self {
            ProvisionerKind::None => "none",
            ProvisionerKind::Static => "static",
            ProvisionerKind::Cloud => "cloud",
            ProvisionerKind::Hypervisor => "hypervisor",
            ProvisionerKind::Kubernetes => "kubernetes",
            ProvisionerKind::Ipmi => "ipmi",
        }
    }

    /// Parses a configured mode name, case-insensitive.
    pub fn parse(s: &str) -> Option<Self> {
        match s.trim().to_ascii_lowercase().as_str() {
            "none" | "" => Some(ProvisionerKind::None),
            "static" => Some(ProvisionerKind::Static),
            "cloud" => Some(ProvisionerKind::Cloud),
            "hypervisor" => Some(ProvisionerKind::Hypervisor),
            "kubernetes" | "k8s" => Some(ProvisionerKind::Kubernetes),
            "ipmi" | "redfish" => Some(ProvisionerKind::Ipmi),
            _ => None,
        }
    }

    /// How long capacity of this kind typically takes to become useful,
    /// before this deployment has measured it for itself.
    ///
    /// A prior, not a constant: the first completed provision replaces it, and
    /// every later one moves it. It exists because the projection needs a
    /// horizon on the very first scale-out, when nothing has been measured
    /// and refusing to project would mean never scaling out in time.
    pub const fn latency_prior(self) -> Duration {
        match self {
            // Nothing is provisioned, so there is no horizon to project over
            ProvisionerKind::None => Duration::ZERO,
            // The machine is already running. Claiming it is a mesh message
            ProvisionerKind::Static => Duration::from_secs(2),
            // Instance create, boot, join
            ProvisionerKind::Cloud => Duration::from_secs(90),
            // Template clone dominates, and it reads a disk image
            ProvisionerKind::Hypervisor => Duration::from_secs(120),
            // Image pull is the variable part. A warm node schedules in
            // seconds, a cold one waits on the registry
            ProvisionerKind::Kubernetes => Duration::from_secs(30),
            // Power on, POST, network boot, join
            ProvisionerKind::Ipmi => Duration::from_secs(420),
        }
    }

    /// Whether nodes of this kind are billed by elapsed time.
    ///
    /// Decides whether reclaiming early can lose money. A rack is paid for
    /// whether or not the node is running.
    pub const fn billed_by_time(self) -> bool {
        matches!(self, ProvisionerKind::Cloud)
    }
}

/// What a driver can actually do, which is what every other module reads
/// instead of reading the mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct ProvisionerCapabilities {
    pub kind: ProvisionerKind,
    /// Whether new capacity can be created. False for a fixed pool and for a
    /// deployment whose control plane is unreachable or unconfigured
    pub can_provision: bool,
    /// Whether capacity can be handed back
    pub can_reclaim: bool,
    /// Whether the deployment may drop to no serving nodes at all
    pub can_scale_to_zero: bool,
    /// Smallest and largest mesh this deployment permits. Zero max means the
    /// driver does not bound it
    pub min_nodes: u32,
    pub max_nodes: u32,
    /// Billing interval capacity is charged in, zero when it is not billed by
    /// time. Reclaiming inside an interval that is already paid for is a loss
    pub billing_interval: Duration,
}

impl ProvisionerCapabilities {
    /// The capability set of a driver that cannot reach its control plane.
    ///
    /// Everything is refused rather than attempted, because a driver that
    /// tries and fails leaves the controller waiting out a provision latency
    /// for capacity that was never coming.
    pub const fn unreachable(kind: ProvisionerKind) -> Self {
        Self {
            kind,
            can_provision: false,
            can_reclaim: false,
            can_scale_to_zero: false,
            min_nodes: 1,
            max_nodes: 1,
            billing_interval: Duration::ZERO,
        }
    }
}

/// What the controller is asking for, and why.
#[derive(Debug, Clone, PartialEq)]
pub struct ProvisionRequest {
    /// How many nodes beyond the current mesh
    pub nodes: u32,
    /// The class whose objective is being missed
    pub class: crate::pressure::WorkloadClass,
    /// Pressure that triggered the request, in seconds
    pub pressure_seconds: f64,
    /// Pressure expected by the time the capacity arrives, which is the
    /// number the request is actually justified by
    pub projected_pressure_seconds: f64,
    /// The node asking
    pub requested_by: u64,
}

/// What the controller is giving back.
#[derive(Debug, Clone, PartialEq)]
pub struct ReclaimRequest {
    pub node_id: u64,
    /// Time left in an interval already paid for. Zero when capacity is not
    /// billed by time
    pub remaining_paid_interval: Duration,
    /// How long the mesh expects to stay this quiet, from the arrival trend
    pub predicted_idle_window: Duration,
    /// Whether the node has handed its working set to a survivor
    pub hot_set_handed_off: bool,
}

/// A provision in progress.
#[derive(Debug, Clone, PartialEq)]
pub struct ProvisionTicket {
    /// Opaque to Zyron, meaningful to the control plane
    pub external_id: String,
    pub nodes: u32,
    /// What the driver expects this one to take, which the caller records
    /// against the actual on completion
    pub expected: Duration,
}

/// The interface between wanting capacity and getting it.
///
/// Every method is synchronous and expected to return promptly: a driver talks
/// to a control plane, it does not wait for the node to boot. The waiting is
/// the caller's, against `expected`.
pub trait ProvisionerDriver: Send + Sync {
    fn kind(&self) -> ProvisionerKind;

    fn capabilities(&self) -> ProvisionerCapabilities;

    /// Asks for capacity. Returns as soon as the control plane has accepted
    /// the request, not when the node is serving.
    fn provision(&self, request: &ProvisionRequest) -> Result<ProvisionTicket>;

    /// Hands capacity back.
    fn reclaim(&self, request: &ReclaimRequest) -> Result<()>;

    /// Whether this request may proceed given what it would cost.
    ///
    /// The default is the rule that holds for every kind: capacity billed by
    /// time is kept until the interval it is already charged for runs out,
    /// unless the mesh expects to stay quiet past that point. Capacity that is
    /// not billed by time is released as soon as it is idle.
    fn reclaim_allowed(&self, request: &ReclaimRequest) -> ReclaimVerdict {
        let capabilities = self.capabilities();
        if !capabilities.can_reclaim {
            return ReclaimVerdict::Refused("this deployment does not reclaim nodes");
        }
        if !request.hot_set_handed_off {
            return ReclaimVerdict::Refused(
                "the node has not handed its working set to a survivor",
            );
        }
        if capabilities.billing_interval.is_zero() {
            return ReclaimVerdict::Allowed;
        }
        if request.remaining_paid_interval > request.predicted_idle_window {
            return ReclaimVerdict::HoldUntilPaidIntervalEnds(request.remaining_paid_interval);
        }
        ReclaimVerdict::Allowed
    }
}

/// The answer to whether a node may be given back now.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReclaimVerdict {
    Allowed,
    /// Keeping it costs nothing more, and giving it back would mean paying to
    /// create it again inside the interval already charged for
    HoldUntilPaidIntervalEnds(Duration),
    Refused(&'static str),
}

impl ReclaimVerdict {
    pub fn allowed(self) -> bool {
        matches!(self, ReclaimVerdict::Allowed)
    }

    pub fn as_str(self) -> &'static str {
        match self {
            ReclaimVerdict::Allowed => "allowed",
            ReclaimVerdict::HoldUntilPaidIntervalEnds(_) => "hold_until_paid_interval_ends",
            ReclaimVerdict::Refused(_) => "refused",
        }
    }
}

/// A driver for a deployment whose control plane this process cannot reach.
///
/// Not a placeholder for a missing implementation: it is the correct behaviour
/// for a node configured for a mode whose credentials, endpoint, or client are
/// not present. It reports that it cannot provision, which masks the mesh
/// rungs of the actuator ladder, so the controller relieves pressure with the
/// levers it owns instead of publishing a request nothing will answer.
#[derive(Debug, Clone, Copy)]
pub struct UnreachableProvisioner {
    kind: ProvisionerKind,
}

impl UnreachableProvisioner {
    pub const fn new(kind: ProvisionerKind) -> Self {
        Self { kind }
    }

    fn explain(&self) -> ZyronError {
        ZyronError::ConfigError(format!(
            "node_registration_mode is '{}' but no {} provisioner is installed in this process, \
             so capacity cannot be changed from here",
            self.kind.as_str(),
            self.kind.as_str()
        ))
    }
}

impl ProvisionerDriver for UnreachableProvisioner {
    fn kind(&self) -> ProvisionerKind {
        self.kind
    }

    fn capabilities(&self) -> ProvisionerCapabilities {
        ProvisionerCapabilities::unreachable(self.kind)
    }

    fn provision(&self, _request: &ProvisionRequest) -> Result<ProvisionTicket> {
        Err(self.explain())
    }

    fn reclaim(&self, _request: &ReclaimRequest) -> Result<()> {
        Err(self.explain())
    }
}

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// The `[mesh]` section of the config file.
///
/// It lives beside the driver rather than beside the rest of the config
/// because its one field is the mode, and the whole point of this module is
/// that nothing outside it reads the mode.
#[derive(Debug, Clone, PartialEq, Deserialize, Serialize)]
#[serde(default)]
pub struct MeshSection {
    /// How this node's deployment acquires hardware: `none`, `static`,
    /// `cloud`, `hypervisor`, `kubernetes`, or `ipmi`.
    pub node_registration_mode: String,
    /// Largest warm pool the operator will pay to keep idle. A cap on the
    /// size the projection asks for, never a target: a node that projects no
    /// need keeps no warm pool whatever this says
    pub warm_pool_max_nodes: u32,
    /// Overrides the measured provision latency, for a deployment whose
    /// control plane is slower than what this node has observed. Zero uses
    /// the measurement
    pub provision_latency_secs: u64,
    /// How many page identifiers a draining node hands to its survivors
    pub hot_set_pages: u32,
    /// How many query shapes a draining node hands to its survivors
    pub hot_set_queries: u32,
}

impl Default for MeshSection {
    fn default() -> Self {
        Self {
            node_registration_mode: "none".into(),
            warm_pool_max_nodes: 0,
            provision_latency_secs: 0,
            hot_set_pages: crate::hot_set::DEFAULT_HOT_PAGES,
            hot_set_queries: crate::hot_set::DEFAULT_HOT_QUERIES,
        }
    }
}

impl MeshSection {
    /// The driver this deployment's mode selects.
    ///
    /// Exists so no caller outside this module has to name the mode. Every
    /// consumer wants the driver, and giving them the driver is what keeps the
    /// mode from being read a second time somewhere that would branch on it.
    pub fn select_driver(&self) -> Arc<dyn ProvisionerDriver> {
        ProvisionerRegistry::global().select(&self.node_registration_mode)
    }

    /// Rejects a mode nobody implements rather than falling back to one
    /// silently, because falling back means an operator who asked for
    /// elasticity gets none and is never told.
    pub fn validate(&self) -> Result<()> {
        if ProvisionerKind::parse(&self.node_registration_mode).is_none() {
            return Err(ZyronError::InvalidParameter {
                name: "mesh.node_registration_mode".into(),
                value: self.node_registration_mode.clone(),
            });
        }
        if self.hot_set_pages > crate::hot_set::MAX_HOT_PAGES {
            return Err(ZyronError::InvalidParameter {
                name: "mesh.hot_set_pages".into(),
                value: self.hot_set_pages.to_string(),
            });
        }
        if self.hot_set_queries > crate::hot_set::MAX_HOT_QUERIES {
            return Err(ZyronError::InvalidParameter {
                name: "mesh.hot_set_queries".into(),
                value: self.hot_set_queries.to_string(),
            });
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Registry
// ---------------------------------------------------------------------------

/// Holds the installed driver and what provisioning has actually cost.
///
/// The measured latency is kept here rather than inside a driver so it
/// survives a driver being replaced, and so a deployment that installs its
/// real driver after startup inherits what the process already observed.
pub struct ProvisionerRegistry {
    driver: std::sync::RwLock<Arc<dyn ProvisionerDriver>>,
    /// Smoothed provision latency in milliseconds, per kind. Zero means
    /// nothing has completed yet and the prior stands
    measured_ms: [AtomicU64; ProvisionerKind::COUNT],
    completed: [AtomicU64; ProvisionerKind::COUNT],
}

/// Weight given to the newest completed provision.
///
/// High, because provision latency is dominated by whatever the control plane
/// is doing right now: an image pull that got slow is the number that matters
/// for the next request, not the average of a quiet afternoon.
const LATENCY_BLEND: f64 = 0.3;

static REGISTRY: std::sync::OnceLock<ProvisionerRegistry> = std::sync::OnceLock::new();

impl ProvisionerRegistry {
    pub fn global() -> &'static ProvisionerRegistry {
        REGISTRY.get_or_init(ProvisionerRegistry::new)
    }

    pub fn new() -> Self {
        Self {
            driver: std::sync::RwLock::new(Arc::new(UnreachableProvisioner::new(
                ProvisionerKind::None,
            ))),
            measured_ms: std::array::from_fn(|_| AtomicU64::new(0)),
            completed: std::array::from_fn(|_| AtomicU64::new(0)),
        }
    }

    /// Chooses the driver for a configured mode.
    ///
    /// This function is the only reader of the registration mode in the whole
    /// substrate. An unknown mode is not a fallback: it produces a driver that
    /// refuses everything and says why, so the deployment fails loudly at the
    /// first scale-out rather than silently never scaling.
    pub fn select(&self, node_registration_mode: &str) -> Arc<dyn ProvisionerDriver> {
        let kind = ProvisionerKind::parse(node_registration_mode).unwrap_or(ProvisionerKind::None);
        let installed = self.driver.read().ok().map(|d| Arc::clone(&d));
        match installed {
            Some(driver) if driver.kind() == kind => driver,
            _ => Arc::new(UnreachableProvisioner::new(kind)),
        }
    }

    /// Installs the driver a deployment's control-plane client provides.
    pub fn install(&self, driver: Arc<dyn ProvisionerDriver>) {
        if let Ok(mut slot) = self.driver.write() {
            *slot = driver;
        }
    }

    /// The driver currently installed, without consulting the mode.
    pub fn active(&self) -> Arc<dyn ProvisionerDriver> {
        self.driver
            .read()
            .ok()
            .map(|d| Arc::clone(&d))
            .unwrap_or_else(|| Arc::new(UnreachableProvisioner::new(ProvisionerKind::None)))
    }

    /// Records how long a completed provision actually took.
    pub fn record_provision_latency(&self, kind: ProvisionerKind, elapsed: Duration) {
        let i = kind.index();
        let observed = elapsed.as_millis().min(u64::MAX as u128) as u64;
        let prior = self.measured_ms[i].load(Ordering::Relaxed);
        let next = if prior == 0 {
            observed
        } else {
            (prior as f64 * (1.0 - LATENCY_BLEND) + observed as f64 * LATENCY_BLEND) as u64
        };
        self.measured_ms[i].store(next, Ordering::Relaxed);
        self.completed[i].fetch_add(1, Ordering::Relaxed);
    }

    /// How long provisioning takes here: measured when anything has completed,
    /// the kind's prior otherwise.
    pub fn provision_latency(&self, kind: ProvisionerKind) -> Duration {
        let measured = self.measured_ms[kind.index()].load(Ordering::Relaxed);
        if measured == 0 {
            kind.latency_prior()
        } else {
            Duration::from_millis(measured)
        }
    }

    /// How many provisions this process has seen complete for a kind.
    pub fn completed_provisions(&self, kind: ProvisionerKind) -> u64 {
        self.completed[kind.index()].load(Ordering::Relaxed)
    }

    /// Replaces the measured latency outright, for an operator override.
    pub fn set_provision_latency(&self, kind: ProvisionerKind, latency: Duration) {
        self.measured_ms[kind.index()].store(
            latency.as_millis().min(u64::MAX as u128) as u64,
            Ordering::Relaxed,
        );
    }
}

impl Default for ProvisionerRegistry {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Scale to zero
// ---------------------------------------------------------------------------

/// What has to be true before a node may stop serving entirely.
#[derive(Debug, Clone, PartialEq)]
pub struct ScaleToZeroInputs {
    /// How old the newest complete checkpoint is
    pub checkpoint_age: Duration,
    /// Whether that checkpoint closed cleanly. A torn checkpoint is not a
    /// checkpoint: resuming from it means replaying the log after all
    pub checkpoint_clean: bool,
    /// Write-ahead log written since that checkpoint. Not part of the resume
    /// estimate, and deliberately so: it is recorded to make visible how far
    /// the two have diverged
    pub wal_bytes_since_checkpoint: u64,
    /// Whether the working-set manifest has been written where a resuming
    /// node will find it
    pub hot_set_persisted: bool,
    /// Pages the manifest names
    pub hot_set_pages: u32,
    /// Bytes in one page
    pub page_bytes: u64,
    /// Measured sequential read throughput in bytes per second.
    ///
    /// The manifest is sorted, so the reload is a sequential read and this is
    /// the rate it will actually run at. Zero when nothing has been measured
    /// yet, which falls back to the per-page latency below
    pub read_throughput_bytes_per_s: f64,
    /// Measured page read latency, used when throughput has not been observed.
    ///
    /// One page at a time is the slowest the reload can be, and over-stating
    /// the resume cost is the safe direction to be wrong in: it delays a scale
    /// to zero rather than promising a resume that does not arrive
    pub page_read_p50: Duration,
}

/// What resuming from zero will cost.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ResumeEstimate {
    /// Time to open the checkpoint, which grows with how stale it is
    pub checkpoint_freshness: Duration,
    /// Time to read the working set back in
    pub hot_set_reload: Duration,
    pub total: Duration,
}

/// How much of a stale checkpoint's age shows up as resume time.
///
/// Opening a checkpoint is not free and not proportional to its age, but a
/// stale one carries more catalog and page-table delta to apply. Measured as
/// a fraction rather than modelled, because the alternative is a constant that
/// is wrong for every deployment.
const CHECKPOINT_AGE_TO_RESUME: f64 = 0.02;

/// Floor on checkpoint open time, so a node that checkpointed a moment ago is
/// not estimated to resume instantly.
const CHECKPOINT_OPEN_FLOOR: Duration = Duration::from_millis(50);

/// Decides whether a node may go to zero, and says what waking it will cost.
///
/// Refuses on either missing precondition. Both refusals exist because the
/// operator's experience of scale-to-zero is entirely the resume, and both
/// missing pieces turn a fast resume into a slow one without warning: without
/// a clean checkpoint the node replays the log, and without a manifest it
/// serves the first minutes of traffic from an empty buffer pool.
pub fn assert_scale_to_zero_ready(inputs: &ScaleToZeroInputs) -> Result<ResumeEstimate> {
    if !inputs.checkpoint_clean {
        return Err(ZyronError::ConfigError(format!(
            "scale to zero refused: the newest checkpoint did not close cleanly, so resuming \
             would replay {} bytes of write-ahead log instead of opening a checkpoint",
            inputs.wal_bytes_since_checkpoint
        )));
    }
    if !inputs.hot_set_persisted {
        return Err(ZyronError::ConfigError(
            "scale to zero refused: the working-set manifest has not been persisted, so a \
             resumed node would serve from an empty buffer pool"
                .into(),
        ));
    }
    Ok(estimate_resume(inputs))
}

/// Resume time, which is checkpoint staleness plus working-set reload and
/// explicitly not the length of the write-ahead log.
pub fn estimate_resume(inputs: &ScaleToZeroInputs) -> ResumeEstimate {
    let checkpoint_freshness =
        Duration::from_secs_f64(inputs.checkpoint_age.as_secs_f64() * CHECKPOINT_AGE_TO_RESUME)
            .max(CHECKPOINT_OPEN_FLOOR);

    // The manifest is sorted, so the reload is a sequential read and the
    // measured throughput is what it will run at. Nothing here is a guess
    // about how many reads are in flight: that is the device's business and it
    // is already reflected in the throughput the node observed
    let bytes = inputs.hot_set_pages as f64 * inputs.page_bytes as f64;
    let hot_set_reload = if inputs.read_throughput_bytes_per_s > 0.0 {
        Duration::from_secs_f64(bytes / inputs.read_throughput_bytes_per_s)
    } else {
        Duration::from_secs_f64(inputs.hot_set_pages as f64 * inputs.page_read_p50.as_secs_f64())
    };

    ResumeEstimate {
        checkpoint_freshness,
        hot_set_reload,
        total: checkpoint_freshness + hot_set_reload,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pressure::WorkloadClass;

    fn request() -> ProvisionRequest {
        ProvisionRequest {
            nodes: 1,
            class: WorkloadClass::Interactive,
            pressure_seconds: 0.4,
            projected_pressure_seconds: 1.2,
            requested_by: 7,
        }
    }

    #[test]
    fn every_mode_name_round_trips() {
        for kind in ProvisionerKind::ALL {
            assert_eq!(ProvisionerKind::parse(kind.as_str()), Some(kind));
        }
        assert_eq!(
            ProvisionerKind::parse("K8s"),
            Some(ProvisionerKind::Kubernetes)
        );
        assert_eq!(
            ProvisionerKind::parse("redfish"),
            Some(ProvisionerKind::Ipmi)
        );
        assert_eq!(ProvisionerKind::parse("ec2"), None);
    }

    /// An unknown mode must not quietly become a working one.
    #[test]
    fn an_unknown_mode_is_refused_by_the_config() {
        let section = MeshSection {
            node_registration_mode: "ec2-spot".into(),
            ..MeshSection::default()
        };
        assert!(section.validate().is_err());
        assert!(MeshSection::default().validate().is_ok());
    }

    /// A driver that cannot reach its control plane says so instead of
    /// accepting a request nothing will answer.
    #[test]
    fn an_unreachable_driver_refuses_rather_than_pretending() {
        let driver = UnreachableProvisioner::new(ProvisionerKind::Kubernetes);
        assert!(!driver.capabilities().can_provision);
        let err = driver.provision(&request()).unwrap_err().to_string();
        assert!(err.contains("kubernetes"), "{err}");
    }

    /// Selecting for a mode with no matching driver installed gives a driver
    /// of that kind that refuses, not a driver of some other kind.
    #[test]
    fn select_returns_a_driver_of_the_configured_kind() {
        let registry = ProvisionerRegistry::new();
        for kind in ProvisionerKind::ALL {
            let driver = registry.select(kind.as_str());
            assert_eq!(driver.kind(), kind);
            assert!(!driver.capabilities().can_provision);
        }
    }

    /// An installed driver of one kind must not be handed out for another.
    #[test]
    fn select_will_not_hand_out_a_driver_for_a_different_mode() {
        struct Fake;
        impl ProvisionerDriver for Fake {
            fn kind(&self) -> ProvisionerKind {
                ProvisionerKind::Cloud
            }
            fn capabilities(&self) -> ProvisionerCapabilities {
                ProvisionerCapabilities {
                    kind: ProvisionerKind::Cloud,
                    can_provision: true,
                    can_reclaim: true,
                    can_scale_to_zero: true,
                    min_nodes: 0,
                    max_nodes: 64,
                    billing_interval: Duration::from_secs(3600),
                }
            }
            fn provision(&self, _r: &ProvisionRequest) -> Result<ProvisionTicket> {
                Ok(ProvisionTicket {
                    external_id: "i-1".into(),
                    nodes: 1,
                    expected: Duration::from_secs(90),
                })
            }
            fn reclaim(&self, _r: &ReclaimRequest) -> Result<()> {
                Ok(())
            }
        }
        let registry = ProvisionerRegistry::new();
        registry.install(Arc::new(Fake));
        assert!(registry.select("cloud").capabilities().can_provision);
        assert!(!registry.select("kubernetes").capabilities().can_provision);
    }

    /// Cloud capacity inside a paid interval is kept, on-prem capacity is not.
    #[test]
    fn reclaim_waits_out_a_paid_interval_and_never_waits_without_one() {
        struct Billed(bool);
        impl ProvisionerDriver for Billed {
            fn kind(&self) -> ProvisionerKind {
                if self.0 {
                    ProvisionerKind::Cloud
                } else {
                    ProvisionerKind::Static
                }
            }
            fn capabilities(&self) -> ProvisionerCapabilities {
                ProvisionerCapabilities {
                    kind: self.kind(),
                    can_provision: true,
                    can_reclaim: true,
                    can_scale_to_zero: true,
                    min_nodes: 0,
                    max_nodes: 8,
                    billing_interval: if self.0 {
                        Duration::from_secs(3600)
                    } else {
                        Duration::ZERO
                    },
                }
            }
            fn provision(&self, _r: &ProvisionRequest) -> Result<ProvisionTicket> {
                Ok(ProvisionTicket {
                    external_id: String::new(),
                    nodes: 1,
                    expected: Duration::ZERO,
                })
            }
            fn reclaim(&self, _r: &ReclaimRequest) -> Result<()> {
                Ok(())
            }
        }

        let mut req = ReclaimRequest {
            node_id: 3,
            remaining_paid_interval: Duration::from_secs(2000),
            predicted_idle_window: Duration::from_secs(600),
            hot_set_handed_off: true,
        };

        let cloud = Billed(true);
        assert!(matches!(
            cloud.reclaim_allowed(&req),
            ReclaimVerdict::HoldUntilPaidIntervalEnds(_)
        ));
        // Quiet for longer than the interval left, so giving it back is free
        req.predicted_idle_window = Duration::from_secs(4000);
        assert!(cloud.reclaim_allowed(&req).allowed());

        // A rack has no interval, so the question never arises
        req.predicted_idle_window = Duration::from_secs(1);
        assert!(Billed(false).reclaim_allowed(&req).allowed());
    }

    /// A node that has not handed off its working set is not reclaimed, or the
    /// survivors serve the drain from an empty pool.
    #[test]
    fn reclaim_refuses_before_the_working_set_moves() {
        let driver = UnreachableProvisioner::new(ProvisionerKind::Static);
        let req = ReclaimRequest {
            node_id: 1,
            remaining_paid_interval: Duration::ZERO,
            predicted_idle_window: Duration::from_secs(600),
            hot_set_handed_off: false,
        };
        assert!(!driver.reclaim_allowed(&req).allowed());
    }

    /// Measured latency replaces the prior on the first completion and is
    /// blended after that.
    #[test]
    fn provision_latency_is_measured_once_anything_completes() {
        let registry = ProvisionerRegistry::new();
        assert_eq!(
            registry.provision_latency(ProvisionerKind::Cloud),
            ProvisionerKind::Cloud.latency_prior()
        );
        registry.record_provision_latency(ProvisionerKind::Cloud, Duration::from_secs(30));
        assert_eq!(
            registry.provision_latency(ProvisionerKind::Cloud),
            Duration::from_secs(30)
        );
        registry.record_provision_latency(ProvisionerKind::Cloud, Duration::from_secs(60));
        let blended = registry.provision_latency(ProvisionerKind::Cloud);
        assert!(
            blended > Duration::from_secs(30) && blended < Duration::from_secs(60),
            "{blended:?}"
        );
        assert_eq!(registry.completed_provisions(ProvisionerKind::Cloud), 2);
    }

    /// The two preconditions are refusals, not warnings.
    #[test]
    fn scale_to_zero_refuses_without_a_clean_checkpoint_or_a_manifest() {
        let mut inputs = ScaleToZeroInputs {
            checkpoint_age: Duration::from_secs(300),
            checkpoint_clean: false,
            wal_bytes_since_checkpoint: 900_000,
            hot_set_persisted: true,
            hot_set_pages: 10_000,
            page_bytes: 16_384,
            read_throughput_bytes_per_s: 2_000_000_000.0,
            page_read_p50: Duration::from_micros(80),
        };
        assert!(assert_scale_to_zero_ready(&inputs).is_err());

        inputs.checkpoint_clean = true;
        inputs.hot_set_persisted = false;
        assert!(assert_scale_to_zero_ready(&inputs).is_err());

        inputs.hot_set_persisted = true;
        assert!(assert_scale_to_zero_ready(&inputs).is_ok());
    }

    /// Resume time follows checkpoint staleness and working-set size, and
    /// moving the log length does not move it.
    #[test]
    fn resume_time_ignores_the_write_ahead_log() {
        let base = ScaleToZeroInputs {
            checkpoint_age: Duration::from_secs(600),
            checkpoint_clean: true,
            wal_bytes_since_checkpoint: 0,
            hot_set_persisted: true,
            hot_set_pages: 32_000,
            page_bytes: 16_384,
            // Two gigabytes a second, which is a plain NVMe read
            read_throughput_bytes_per_s: 2_000_000_000.0,
            page_read_p50: Duration::from_micros(100),
        };
        let quiet = estimate_resume(&base);

        let mut busy = base.clone();
        busy.wal_bytes_since_checkpoint = 8 * 1024 * 1024 * 1024;
        assert_eq!(estimate_resume(&busy), quiet);

        // 32000 pages of 16KB at two gigabytes a second
        assert!(
            (quiet.hot_set_reload.as_secs_f64() - 0.262144).abs() < 1e-6,
            "{:?}",
            quiet.hot_set_reload
        );
        assert!(quiet.total > quiet.hot_set_reload);

        // With no throughput measured yet the estimate falls back to reading
        // one page at a time, which is slower and is the safe direction
        let mut unmeasured = base.clone();
        unmeasured.read_throughput_bytes_per_s = 0.0;
        assert!(
            estimate_resume(&unmeasured).hot_set_reload > quiet.hot_set_reload,
            "the unmeasured fallback was optimistic"
        );

        let mut stale = base.clone();
        stale.checkpoint_age = Duration::from_secs(7200);
        assert!(estimate_resume(&stale).total > quiet.total);
    }
}
