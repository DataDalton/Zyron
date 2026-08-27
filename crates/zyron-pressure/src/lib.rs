//! How one node adapts to its own load.
//!
//! Everything here answers a question about this machine: what it can do, what
//! it is currently doing, what is limiting it, and which of the responses it
//! owns is the cheapest one that helps. Nothing here talks to another node.
//!
//! ## Why this is its own crate
//!
//! It was inside `zyron-common`, which every other crate depends on. That put
//! a control loop with a background tick, a hardware probe, and a persisted
//! learning ledger underneath the type definitions, which meant a change to
//! how the node measures itself rebuilt the whole tree, and it meant the error
//! type could reach the controller. One direction of that reach was real:
//! producing a transaction conflict is the event the conflict signal is made
//! of. That call now goes through [`zyron_common::conflict_signal`], which
//! this crate installs itself into, so the dependency points the way the crate
//! graph does.
//!
//! ## What is not here
//!
//! Cross-node coordination. Claiming a warm node and asking for hardware are
//! rungs on the ladder this crate owns, but performing them is the mesh's
//! work, and the mesh sits above this crate. The seam is
//! [`ActuatorExtension`]: the ladder climbs to a rung it cannot reach itself,
//! asks whatever is registered, and treats an absent or unwilling extension
//! the way it treats any rung that did not help, which is by carrying on up
//! to shedding.

pub mod calibration_store;
pub mod capability;
pub mod extension;
pub mod hot_set;
pub mod pressure;
pub mod pressure_control;
pub mod projection;
pub mod provisioner;

pub use calibration_store::{CALIBRATION_FILE, CalibrationCache, CalibrationEntry};
pub use capability::{
    CALIBRATION_KEY_PREFIX, CalibrationField, CoefficientAccumulator, ContentionSignals,
    DeviceKind, FingerprintInputs, HardwareFingerprint, LatencyHistogram,
    MIN_SAMPLES_FOR_CONFIDENCE, MountCapability, NodeCapabilities, OperatorCoefficients,
    OperatorKind, SimdLevel, calibration_key, decode_coefficient, encode_coefficient,
    parse_calibration_key,
};
pub use extension::{ActuatorExtension, ActuatorResult, ExtensionRegistry};
pub use hot_set::{
    DEFAULT_HOT_PAGES, DEFAULT_HOT_QUERIES, HOT_SET_FILE, HotQuery, HotSetManifest, MAX_HOT_PAGES,
    MAX_HOT_QUERIES, PrefetchReport,
};
pub use pressure::{
    ActuatorDecision, ActuatorLevel, AdmitDecision, BottleneckKind, ClassCounters, ClassPressure,
    MemoryReservation, NodeMemoryGauge, NodePressure, PARALLEL_SCALE_FULL_PCT,
    PARALLEL_SCALE_MIN_PCT, ParallelCapacity, PressureField, WorkloadClass,
};
pub use pressure_control::{
    AdmissionRecord, CheckpointState, ConnectionGauge, ConnectionSlot, HistorySample, HotSetStatus,
    LedgerEntry, MeshReach, PressureController, TenantPressure,
};
pub use projection::{
    ArrivalSample, ArrivalTrend, MIN_TREND_SAMPLES, PressureProjection, ProjectionInputs,
    TREND_WINDOW, fit_arrival_trend, predicted_idle_window,
};
pub use provisioner::{
    MeshSection, ProvisionRequest, ProvisionTicket, ProvisionerCapabilities, ProvisionerDriver,
    ProvisionerKind, ProvisionerRegistry, ReclaimRequest, ReclaimVerdict, ResumeEstimate,
    ScaleToZeroInputs, UnreachableProvisioner, assert_scale_to_zero_ready, estimate_resume,
};
