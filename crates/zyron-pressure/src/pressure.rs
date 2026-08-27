//! Work-time pressure signal.
//!
//! Queue length says nothing when a 50us point lookup and a 30s scan share a
//! queue. Queued work-time does. Every query carries an estimated cost in
//! seconds from the planner, so the queue holds a known amount of work rather
//! than a count of unknowns, and dividing that by measured service capacity
//! gives a pressure in seconds that compares directly against a latency
//! objective expressed in the same unit.
//!
//! The signal is per workload class per node. Classes never share a queue,
//! because a controller that averages a bimodal workload serves neither half.

use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};
use std::time::Duration;

use serde::{Deserialize, Serialize};

// ---------------------------------------------------------------------------
// Workload class
// ---------------------------------------------------------------------------

/// The three service classes a query can land in. Fixed, not user extensible:
/// each one carries a latency objective the controller measures against, and
/// an objective nobody has measured is not an objective.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum WorkloadClass {
    /// Sub-100ms work, the shape a dashboard or an application request makes
    Interactive,
    /// Analytical work measured in seconds, tolerant of queueing
    Bulk,
    /// Work with no caller waiting on it, yielding to everything else
    Background,
}

impl WorkloadClass {
    /// Every class, in the order their indices run.
    pub const ALL: [WorkloadClass; 3] = [
        WorkloadClass::Interactive,
        WorkloadClass::Bulk,
        WorkloadClass::Background,
    ];

    /// Number of distinct classes, the width of every per-class array.
    pub const COUNT: usize = 3;

    /// Dense index for array storage.
    pub const fn index(self) -> usize {
        match self {
            WorkloadClass::Interactive => 0,
            WorkloadClass::Bulk => 1,
            WorkloadClass::Background => 2,
        }
    }

    /// The class at a dense index, None when the index is out of range.
    pub const fn from_index(i: usize) -> Option<Self> {
        match i {
            0 => Some(WorkloadClass::Interactive),
            1 => Some(WorkloadClass::Bulk),
            2 => Some(WorkloadClass::Background),
            _ => None,
        }
    }

    /// Name as it appears in system views and gossip keys.
    pub const fn as_str(self) -> &'static str {
        match self {
            WorkloadClass::Interactive => "interactive",
            WorkloadClass::Bulk => "bulk",
            WorkloadClass::Background => "background",
        }
    }

    /// Parses a class name, case insensitive.
    pub fn parse(s: &str) -> Option<Self> {
        match s.trim().to_ascii_lowercase().as_str() {
            "interactive" => Some(WorkloadClass::Interactive),
            "bulk" => Some(WorkloadClass::Bulk),
            "background" => Some(WorkloadClass::Background),
            _ => None,
        }
    }

    /// Latency objective in seconds. Pressure is denominated the same way, so
    /// the comparison needs no conversion and no tuning constant.
    /// Background has no objective, it runs on whatever is left.
    pub const fn slo_seconds(self) -> f64 {
        match self {
            WorkloadClass::Interactive => 0.100,
            WorkloadClass::Bulk => 30.0,
            WorkloadClass::Background => f64::INFINITY,
        }
    }

    /// How many queries may wait before arrivals are shed. Interactive holds
    /// the shallowest queue because a deep queue on a 100ms objective is
    /// already a breach by the time it drains.
    pub const fn queue_depth(self) -> usize {
        match self {
            WorkloadClass::Interactive => 1024,
            WorkloadClass::Bulk => 256,
            WorkloadClass::Background => 64,
        }
    }

    /// Estimated cost below which a query skips admission entirely. Deciding
    /// costs more than running at this size.
    pub const BYPASS_WORK_SECONDS: f64 = 0.001;

    /// Estimated cost at or above which a query is no longer Interactive.
    pub const INTERACTIVE_MAX_WORK_SECONDS: f64 = 0.100;

    /// Picks the class for a query from its estimated cost. An explicit
    /// Background request overrides, because a caller declaring that nobody
    /// is waiting knows something the estimate does not.
    pub fn classify(estimated_work_seconds: f64, requested_background: bool) -> Self {
        if requested_background {
            return WorkloadClass::Background;
        }
        if estimated_work_seconds < Self::INTERACTIVE_MAX_WORK_SECONDS {
            WorkloadClass::Interactive
        } else {
            WorkloadClass::Bulk
        }
    }
}

// ---------------------------------------------------------------------------
// Bottleneck
// ---------------------------------------------------------------------------

/// What is actually limiting the node. The controller classifies before it
/// acts, because most saturation has a cause that adding capacity makes worse
/// rather than better.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BottleneckKind {
    /// Nothing is saturated
    None,
    /// The parallel work pool is saturated and the queue is growing
    Cpu,
    /// Storage has no headroom while cores sit idle waiting on it
    Io,
    /// Working set is close to the node memory gauge ceiling
    Memory,
    /// Transactions are aborting on conflict. More concurrency produces more
    /// conflict, so capacity is the wrong lever and reducing it is the right
    /// one
    OccContention,
    /// A small key range is taking a large share of traffic. Capacity does
    /// not spread a hot key, only redistribution does
    HotPartition,
    /// Writers are waiting on durability rather than on compute
    FsyncBound,
}

impl BottleneckKind {
    /// Every classification, so a rule about them can be checked against all
    /// of them rather than against the ones somebody remembered.
    pub const ALL: [BottleneckKind; 7] = [
        BottleneckKind::None,
        BottleneckKind::Cpu,
        BottleneckKind::Io,
        BottleneckKind::Memory,
        BottleneckKind::OccContention,
        BottleneckKind::HotPartition,
        BottleneckKind::FsyncBound,
    ];

    pub const COUNT: usize = 7;

    pub const fn as_str(self) -> &'static str {
        match self {
            BottleneckKind::None => "none",
            BottleneckKind::Cpu => "cpu",
            BottleneckKind::Io => "io",
            BottleneckKind::Memory => "memory",
            BottleneckKind::OccContention => "occ_contention",
            BottleneckKind::HotPartition => "hot_partition",
            BottleneckKind::FsyncBound => "fsync_bound",
        }
    }

    /// Dense encoding for the gossip payload.
    pub const fn code(self) -> u8 {
        match self {
            BottleneckKind::None => 0,
            BottleneckKind::Cpu => 1,
            BottleneckKind::Io => 2,
            BottleneckKind::Memory => 3,
            BottleneckKind::OccContention => 4,
            BottleneckKind::HotPartition => 5,
            BottleneckKind::FsyncBound => 6,
        }
    }

    pub const fn from_code(c: u8) -> Self {
        match c {
            1 => BottleneckKind::Cpu,
            2 => BottleneckKind::Io,
            3 => BottleneckKind::Memory,
            4 => BottleneckKind::OccContention,
            5 => BottleneckKind::HotPartition,
            6 => BottleneckKind::FsyncBound,
            _ => BottleneckKind::None,
        }
    }
}

// ---------------------------------------------------------------------------
// Actuator ladder
// ---------------------------------------------------------------------------

/// Responses to pressure, ordered by what they cost to apply and to undo.
/// The controller takes the lowest rung that brings pressure under the
/// objective, so provisioning hardware is reached only after everything free
/// has been tried.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ActuatorLevel {
    /// Pressure is under the objective, nothing to do
    Steady = 0,
    /// Hand the next query fewer parallel workers. Microseconds, fully
    /// reversible, costs one query some latency and nobody else anything
    ReduceDop = 1,
    /// Hold arrivals in the class queue
    DelayAdmission = 2,
    /// Hand newly admitted queries a smaller memory ceiling.
    ///
    /// Costs nothing to a query already running and no IO to anyone. A new
    /// query gets less room, which with somewhere to spill means it spills
    /// sooner and with nowhere to spill means an over-large one fails early
    /// naming its own limit
    TrimMemory = 3,
    /// Move the point at which materializing operators start writing to disk
    /// down below what the query may hold.
    ///
    /// Distinct from the rung below it, which lowers the ceiling: this lowers
    /// only the spill point, so a query keeps the room it needs for its result
    /// while its sorts, joins, and aggregates give their working set back to
    /// the node early. Real IO, which is why it sits above trimming and why it
    /// is masked wherever the device is what is short
    ForceSpill = 4,
    /// Refuse arrivals past the class queue depth
    Shed = 5,
    /// Add plan-drive capacity within the node ceiling
    GrowWorkers = 6,
    /// Claim a node the mesh already has running
    WarmPoolTake = 7,
    /// Ask the provisioner for hardware. Tens of seconds, and undoing it
    /// throws away a warm cache
    ProvisionNode = 8,
}

impl ActuatorLevel {
    /// The ladder in the order the controller climbs it.
    pub const LADDER: [ActuatorLevel; 8] = [
        ActuatorLevel::ReduceDop,
        ActuatorLevel::DelayAdmission,
        ActuatorLevel::TrimMemory,
        ActuatorLevel::ForceSpill,
        ActuatorLevel::Shed,
        ActuatorLevel::GrowWorkers,
        ActuatorLevel::WarmPoolTake,
        ActuatorLevel::ProvisionNode,
    ];

    pub const fn as_str(self) -> &'static str {
        match self {
            ActuatorLevel::Steady => "steady",
            ActuatorLevel::ReduceDop => "reduce_dop",
            ActuatorLevel::DelayAdmission => "delay_admission",
            ActuatorLevel::TrimMemory => "trim_memory",
            ActuatorLevel::ForceSpill => "force_spill",
            ActuatorLevel::Shed => "shed",
            ActuatorLevel::GrowWorkers => "grow_workers",
            ActuatorLevel::WarmPoolTake => "warm_pool_take",
            ActuatorLevel::ProvisionNode => "provision_node",
        }
    }

    pub const fn code(self) -> u8 {
        self as u8
    }

    pub const fn from_code(c: u8) -> Self {
        match c {
            1 => ActuatorLevel::ReduceDop,
            2 => ActuatorLevel::DelayAdmission,
            3 => ActuatorLevel::TrimMemory,
            4 => ActuatorLevel::ForceSpill,
            5 => ActuatorLevel::Shed,
            6 => ActuatorLevel::GrowWorkers,
            7 => ActuatorLevel::WarmPoolTake,
            8 => ActuatorLevel::ProvisionNode,
            _ => ActuatorLevel::Steady,
        }
    }

    /// Whether this rung may be used against a given bottleneck.
    ///
    /// The gating is the part that separates this from a threshold: three of
    /// the bottleneck kinds get worse when capacity is added, and one of them
    /// gets worse when parallelism is reduced.
    pub const fn legal_for(self, bottleneck: BottleneckKind) -> bool {
        match bottleneck {
            // Conflict rises with concurrency. Only the rungs that remove
            // concurrency help, everything above them adds it
            BottleneckKind::OccContention => matches!(
                self,
                ActuatorLevel::ReduceDop | ActuatorLevel::DelayAdmission | ActuatorLevel::Shed
            ),
            // A hot key does not spread across more nodes, so provisioning is
            // masked until the skew clears. Growth is masked for the same
            // reason it is under conflict: more work in flight against one row
            // is more work queued behind the same row
            BottleneckKind::HotPartition => !matches!(
                self,
                ActuatorLevel::ForceSpill
                    | ActuatorLevel::GrowWorkers
                    | ActuatorLevel::WarmPoolTake
                    | ActuatorLevel::ProvisionNode
            ),
            // Writers are waiting on the device, not on compute, so nothing
            // that adds device work is allowed and spilling is exactly that
            BottleneckKind::FsyncBound => !matches!(
                self,
                ActuatorLevel::ForceSpill
                    | ActuatorLevel::GrowWorkers
                    | ActuatorLevel::WarmPoolTake
                    | ActuatorLevel::ProvisionNode
            ),
            // Cores are already idle waiting on storage, so taking parallelism
            // away frees nothing, and putting more work in flight is the one
            // situation where it helps: the extra workers park on the device
            // rather than competing for a core. Spilling is masked for the
            // same reason writers are: it is more work for the device that is
            // already the constraint
            BottleneckKind::Io => {
                !matches!(self, ActuatorLevel::ReduceDop | ActuatorLevel::ForceSpill)
            }
            // Trimming what a query may hold is the point, and so is spilling
            // what it holds anyway. Growing workers grows the working set,
            // which is the opposite
            BottleneckKind::Memory => !matches!(self, ActuatorLevel::GrowWorkers),
            // Every rung except growth. Raising in-flight work past the core
            // count on a node whose cores are the constraint buys nothing but
            // context switches, and the rung that answers a shortage of cores
            // is provisioning, which is still reachable
            BottleneckKind::Cpu => {
                !matches!(self, ActuatorLevel::ForceSpill | ActuatorLevel::GrowWorkers)
            }
            // Nothing identified as limiting. Growth is the one rung held
            // back, because it is only ever right against a specific
            // diagnosis and this is the absence of one
            BottleneckKind::None => {
                !matches!(self, ActuatorLevel::ForceSpill | ActuatorLevel::GrowWorkers)
            }
        }
    }
}

/// One controller decision, recorded whether or not it changed anything.
#[derive(Debug, Clone, PartialEq)]
pub struct ActuatorDecision {
    pub class: WorkloadClass,
    pub level: ActuatorLevel,
    pub bottleneck: BottleneckKind,
    /// Fraction of the requested worker count a query may still take, in
    /// percent. 100 when the level is not ReduceDop
    pub dop_scale_pct: u16,
    /// How long an arrival waits before it is reconsidered
    pub delay: Duration,
    /// Pressure that produced the decision
    pub pressure_seconds: f64,
    /// The objective it was measured against
    pub slo_seconds: f64,
    pub reason: &'static str,
}

impl ActuatorDecision {
    pub fn steady(class: WorkloadClass, pressure_seconds: f64) -> Self {
        Self {
            class,
            level: ActuatorLevel::Steady,
            bottleneck: BottleneckKind::None,
            dop_scale_pct: 100,
            delay: Duration::ZERO,
            pressure_seconds,
            slo_seconds: class.slo_seconds(),
            reason: "pressure under objective",
        }
    }
}

// ---------------------------------------------------------------------------
// Admission
// ---------------------------------------------------------------------------

/// What the controller tells a connection to do with a query it is holding.
#[derive(Debug, Clone, PartialEq)]
pub enum AdmitDecision {
    /// Run it
    Admit,
    /// Run it without consulting the controller, the query is too small for
    /// the decision to pay for itself
    Bypass,
    /// Hold it for this long, then ask again
    Delay(Duration),
    /// Refuse it
    Shed { reason: String },
}

impl AdmitDecision {
    pub const fn admitted(&self) -> bool {
        matches!(self, AdmitDecision::Admit | AdmitDecision::Bypass)
    }

    pub const fn as_str(&self) -> &'static str {
        match self {
            AdmitDecision::Admit => "admit",
            AdmitDecision::Bypass => "bypass",
            AdmitDecision::Delay(_) => "delay",
            AdmitDecision::Shed { .. } => "shed",
        }
    }
}

// ---------------------------------------------------------------------------
// Per class signal
// ---------------------------------------------------------------------------

/// How much of the newest window's mean service time is taken.
///
/// Low, because the value is multiplied by a whole provision horizon of
/// projected arrivals: a single window of unusually cheap queries would halve
/// the projected work and cancel a scale-out that was needed.
const MEAN_SERVICE_BLEND: f64 = 0.2;

/// Live per-class counters. Read on the admission path, so every field is an
/// atomic and no lock is taken to sample the whole set.
#[derive(Debug)]
pub struct ClassCounters {
    /// Estimated work-seconds sitting in the queue, in microseconds
    queued_work_us: AtomicU64,
    /// Estimated work-seconds remaining across in-flight queries, in micros
    active_work_us: AtomicU64,
    /// Queries waiting
    queued_count: AtomicU32,
    /// Queries running
    in_flight: AtomicU32,
    /// Completions in the current measurement window
    window_completions: AtomicU64,
    /// Work-microseconds actually served in the current window
    window_served_us: AtomicU64,
    /// Measured throughput, queries per second, scaled by 1000
    service_capacity_milli_qps: AtomicU64,
    /// Mean measured service time of one query, in microseconds. Smoothed
    /// across windows, because the projection multiplies it by a whole
    /// horizon of arrivals and one quiet window would halve the answer
    mean_service_us: AtomicU64,
    /// Ratio of actual to estimated cost across the recent window, scaled by
    /// 1000. 1000 means the estimate is exactly right
    calibration_error_milli: AtomicU64,
    /// Concurrency ceiling the controller currently allows
    ceiling: AtomicU32,
    /// Admissions, delays and sheds since start
    admitted_total: AtomicU64,
    delayed_total: AtomicU64,
    shed_total: AtomicU64,
    bypassed_total: AtomicU64,
}

impl Default for ClassCounters {
    fn default() -> Self {
        Self::new()
    }
}

impl ClassCounters {
    pub fn new() -> Self {
        Self {
            queued_work_us: AtomicU64::new(0),
            active_work_us: AtomicU64::new(0),
            queued_count: AtomicU32::new(0),
            in_flight: AtomicU32::new(0),
            window_completions: AtomicU64::new(0),
            window_served_us: AtomicU64::new(0),
            service_capacity_milli_qps: AtomicU64::new(0),
            mean_service_us: AtomicU64::new(0),
            calibration_error_milli: AtomicU64::new(1000),
            ceiling: AtomicU32::new(0),
            admitted_total: AtomicU64::new(0),
            delayed_total: AtomicU64::new(0),
            shed_total: AtomicU64::new(0),
            bypassed_total: AtomicU64::new(0),
        }
    }

    /// Adds a query's estimate to the queue.
    pub fn enqueue(&self, work_seconds: f64) {
        self.queued_work_us
            .fetch_add(seconds_to_us(work_seconds), Ordering::Relaxed);
        self.queued_count.fetch_add(1, Ordering::Relaxed);
    }

    /// Moves a query from queued to in flight.
    pub fn start(&self, work_seconds: f64) {
        let us = seconds_to_us(work_seconds);
        sub_saturating(&self.queued_work_us, us);
        sub_saturating_u32(&self.queued_count, 1);
        self.active_work_us.fetch_add(us, Ordering::Relaxed);
        self.in_flight.fetch_add(1, Ordering::Relaxed);
    }

    /// Admits a query that never queued.
    pub fn start_direct(&self, work_seconds: f64) {
        self.active_work_us
            .fetch_add(seconds_to_us(work_seconds), Ordering::Relaxed);
        self.in_flight.fetch_add(1, Ordering::Relaxed);
    }

    /// Retires a query, recording what it actually cost against what it was
    /// estimated to cost.
    pub fn complete(&self, estimated_seconds: f64, actual_seconds: f64) {
        sub_saturating(&self.active_work_us, seconds_to_us(estimated_seconds));
        sub_saturating_u32(&self.in_flight, 1);
        self.window_completions.fetch_add(1, Ordering::Relaxed);
        self.window_served_us
            .fetch_add(seconds_to_us(actual_seconds), Ordering::Relaxed);
    }

    /// Drops a query that was queued and then refused or cancelled.
    pub fn abandon_queued(&self, work_seconds: f64) {
        sub_saturating(&self.queued_work_us, seconds_to_us(work_seconds));
        sub_saturating_u32(&self.queued_count, 1);
    }

    pub fn record_admit(&self) {
        self.admitted_total.fetch_add(1, Ordering::Relaxed);
    }
    pub fn record_delay(&self) {
        self.delayed_total.fetch_add(1, Ordering::Relaxed);
    }
    pub fn record_shed(&self) {
        self.shed_total.fetch_add(1, Ordering::Relaxed);
    }
    pub fn record_bypass(&self) {
        self.bypassed_total.fetch_add(1, Ordering::Relaxed);
    }

    /// Closes the measurement window and republishes service capacity.
    /// Returns the completions the window saw.
    pub fn roll_window(&self, window: Duration) -> u64 {
        let completions = self.window_completions.swap(0, Ordering::Relaxed);
        let served_us = self.window_served_us.swap(0, Ordering::Relaxed);
        let secs = window.as_secs_f64().max(f64::MIN_POSITIVE);
        let qps = completions as f64 / secs;
        self.service_capacity_milli_qps
            .store((qps * 1000.0) as u64, Ordering::Relaxed);
        if completions > 0 {
            let mean_actual_us = served_us as f64 / completions as f64;
            let previous = self.mean_service_us.load(Ordering::Relaxed);
            let next = if previous == 0 {
                mean_actual_us
            } else {
                previous as f64 * (1.0 - MEAN_SERVICE_BLEND) + mean_actual_us * MEAN_SERVICE_BLEND
            };
            self.mean_service_us.store(next as u64, Ordering::Relaxed);
        }
        completions
    }

    /// Records the ratio of measured to estimated cost. One means the cost
    /// model is telling the truth.
    pub fn set_calibration_error(&self, ratio: f64) {
        let clamped = ratio.clamp(0.001, 1000.0);
        self.calibration_error_milli
            .store((clamped * 1000.0) as u64, Ordering::Relaxed);
    }

    pub fn set_ceiling(&self, ceiling: u32) {
        self.ceiling.store(ceiling, Ordering::Relaxed);
    }

    pub fn ceiling(&self) -> u32 {
        self.ceiling.load(Ordering::Relaxed)
    }

    pub fn in_flight(&self) -> u32 {
        self.in_flight.load(Ordering::Relaxed)
    }

    pub fn queued_count(&self) -> u32 {
        self.queued_count.load(Ordering::Relaxed)
    }

    pub fn queued_work_seconds(&self) -> f64 {
        us_to_seconds(self.queued_work_us.load(Ordering::Relaxed))
    }

    pub fn active_work_seconds(&self) -> f64 {
        us_to_seconds(self.active_work_us.load(Ordering::Relaxed))
    }

    pub fn service_capacity_qps(&self) -> f64 {
        self.service_capacity_milli_qps.load(Ordering::Relaxed) as f64 / 1000.0
    }

    /// Mean measured cost of one query in this class. Zero until a window has
    /// closed with completions in it.
    pub fn mean_service_seconds(&self) -> f64 {
        us_to_seconds(self.mean_service_us.load(Ordering::Relaxed))
    }

    /// Queries that arrived, however they were answered.
    ///
    /// Delayed arrivals are not counted here: a query that queued and then ran
    /// is recorded as admitted when it leaves the queue, so counting the delay
    /// as well would count the same arrival twice.
    pub fn arrivals_total(&self) -> u64 {
        self.admitted_total()
            .saturating_add(self.shed_total())
            .saturating_add(self.bypassed_total())
    }

    pub fn calibration_error(&self) -> f64 {
        self.calibration_error_milli.load(Ordering::Relaxed) as f64 / 1000.0
    }

    pub fn admitted_total(&self) -> u64 {
        self.admitted_total.load(Ordering::Relaxed)
    }
    pub fn delayed_total(&self) -> u64 {
        self.delayed_total.load(Ordering::Relaxed)
    }
    pub fn shed_total(&self) -> u64 {
        self.shed_total.load(Ordering::Relaxed)
    }
    pub fn bypassed_total(&self) -> u64 {
        self.bypassed_total.load(Ordering::Relaxed)
    }

    /// Pressure in seconds: the work the class is holding divided by the rate
    /// it is being served at. Both halves are measured, neither is declared.
    ///
    /// With no measured capacity yet the answer is the raw work outstanding,
    /// which is the honest reading before the first window closes: that much
    /// work exists and nothing is known to be draining it.
    pub fn pressure_seconds(&self) -> f64 {
        let work = self.queued_work_seconds() + self.active_work_seconds();
        let capacity = self.service_capacity_qps();
        if capacity <= 0.0 {
            return work;
        }
        work / capacity
    }

    /// Point-in-time copy for a view or a gossip frame.
    pub fn snapshot(&self, class: WorkloadClass) -> ClassPressure {
        ClassPressure {
            class,
            queued_work_seconds: self.queued_work_seconds(),
            active_work_seconds: self.active_work_seconds(),
            queued_count: self.queued_count(),
            in_flight: self.in_flight(),
            service_capacity_qps: self.service_capacity_qps(),
            mean_service_seconds: self.mean_service_seconds(),
            pressure_seconds: self.pressure_seconds(),
            calibration_error: self.calibration_error(),
            ceiling: self.ceiling(),
            slo_seconds: class.slo_seconds(),
            admitted_total: self.admitted_total(),
            delayed_total: self.delayed_total(),
            shed_total: self.shed_total(),
            bypassed_total: self.bypassed_total(),
            bottleneck: BottleneckKind::None,
            actuator_level: ActuatorLevel::Steady,
        }
    }
}

/// A class's state at one instant.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ClassPressure {
    pub class: WorkloadClass,
    pub queued_work_seconds: f64,
    pub active_work_seconds: f64,
    pub queued_count: u32,
    pub in_flight: u32,
    pub service_capacity_qps: f64,
    /// Mean measured cost of one query in this class
    pub mean_service_seconds: f64,
    pub pressure_seconds: f64,
    pub calibration_error: f64,
    pub ceiling: u32,
    pub slo_seconds: f64,
    pub admitted_total: u64,
    pub delayed_total: u64,
    pub shed_total: u64,
    pub bypassed_total: u64,
    pub bottleneck: BottleneckKind,
    pub actuator_level: ActuatorLevel,
}

impl ClassPressure {
    /// How much of the objective the class is currently consuming. Above one
    /// is a breach. Background has no objective so it never breaches.
    pub fn slo_utilization(&self) -> f64 {
        if !self.slo_seconds.is_finite() || self.slo_seconds <= 0.0 {
            return 0.0;
        }
        self.pressure_seconds / self.slo_seconds
    }

    pub fn breaching(&self) -> bool {
        self.slo_utilization() > 1.0
    }
}

/// Every class on one node, plus the node level rollup the mesh reads.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct NodePressure {
    pub node_id: u64,
    pub classes: Vec<ClassPressure>,
    /// Highest SLO utilization across classes that have an objective. This is
    /// the single number a mesh scheduler compares between nodes
    pub overall_utilization: f64,
    pub bottleneck: BottleneckKind,
    pub actuator_level: ActuatorLevel,
    /// Permits the parallel work pool is handing out right now
    pub parallel_permits_total: u32,
    pub parallel_permits_available: u32,
    /// Node memory gauge, bytes currently reserved by running queries
    pub memory_reserved_bytes: u64,
    pub memory_ceiling_bytes: u64,
    pub updated_at_us: i64,
}

impl NodePressure {
    /// Recomputes the rollup from the per-class rows.
    pub fn recompute_overall(&mut self) {
        self.overall_utilization = self
            .classes
            .iter()
            .map(ClassPressure::slo_utilization)
            .fold(0.0f64, f64::max);
    }
}

// ---------------------------------------------------------------------------
// Gossip encoding
// ---------------------------------------------------------------------------

/// Field slots carried in a gossip frame, one key per slot per class.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PressureField {
    PressureMicros,
    QueuedWorkMicros,
    ActiveWorkMicros,
    ServiceCapacityMilliQps,
    InFlight,
    Ceiling,
    CalibrationErrorMilli,
    /// Bottleneck code in the low byte, actuator code in the next
    StatusCodes,
}

impl PressureField {
    pub const ALL: [PressureField; 8] = [
        PressureField::PressureMicros,
        PressureField::QueuedWorkMicros,
        PressureField::ActiveWorkMicros,
        PressureField::ServiceCapacityMilliQps,
        PressureField::InFlight,
        PressureField::Ceiling,
        PressureField::CalibrationErrorMilli,
        PressureField::StatusCodes,
    ];

    pub const fn as_str(self) -> &'static str {
        match self {
            PressureField::PressureMicros => "pressure_us",
            PressureField::QueuedWorkMicros => "queued_us",
            PressureField::ActiveWorkMicros => "active_us",
            PressureField::ServiceCapacityMilliQps => "capacity_mqps",
            PressureField::InFlight => "in_flight",
            PressureField::Ceiling => "ceiling",
            PressureField::CalibrationErrorMilli => "calib_milli",
            PressureField::StatusCodes => "status",
        }
    }

    pub fn parse(s: &str) -> Option<Self> {
        Self::ALL.into_iter().find(|f| f.as_str() == s)
    }
}

/// Key prefix every pressure record shares, so a receiver can tell pressure
/// keys apart from the quota keys already travelling on the same transport.
pub const PRESSURE_KEY_PREFIX: &str = "zpressure";

/// Builds the gossip key for one field of one class on one node.
pub fn pressure_key(node_id: u64, class: WorkloadClass, field: PressureField) -> String {
    format!(
        "{}/{}/{}/{}",
        PRESSURE_KEY_PREFIX,
        node_id,
        class.as_str(),
        field.as_str()
    )
}

/// Splits a gossip key back into its parts. None when the key is not one of
/// ours, which is how quota keys pass through untouched.
pub fn parse_pressure_key(key: &str) -> Option<(u64, WorkloadClass, PressureField)> {
    let mut parts = key.split('/');
    if parts.next()? != PRESSURE_KEY_PREFIX {
        return None;
    }
    let node_id: u64 = parts.next()?.parse().ok()?;
    let class = WorkloadClass::parse(parts.next()?)?;
    let field = PressureField::parse(parts.next()?)?;
    if parts.next().is_some() {
        return None;
    }
    Some((node_id, class, field))
}

/// Packs a sequence number and a payload into one monotonically increasing
/// value.
///
/// The transport converges by taking the maximum of the local and remote
/// value for a key, which only converges on something that never decreases.
/// Pressure itself rises and falls, so the sequence number occupies the high
/// half and carries the payload with it: a newer tick always compares greater
/// than an older one whatever the payload did.
pub const fn encode_versioned(seq: u32, payload: u32) -> u64 {
    ((seq as u64) << 32) | payload as u64
}

/// Splits a packed value back into its sequence number and payload.
pub const fn decode_versioned(value: u64) -> (u32, u32) {
    ((value >> 32) as u32, value as u32)
}

/// Saturating conversion of a payload that must survive a 32 bit slot.
pub fn clamp_payload(v: u64) -> u32 {
    v.min(u32::MAX as u64) as u32
}

/// Packs the bottleneck and actuator codes into one payload word.
pub const fn encode_status(bottleneck: BottleneckKind, actuator: ActuatorLevel) -> u32 {
    (bottleneck.code() as u32) | ((actuator.code() as u32) << 8)
}

/// Unpacks a status payload.
pub const fn decode_status(payload: u32) -> (BottleneckKind, ActuatorLevel) {
    (
        BottleneckKind::from_code((payload & 0xff) as u8),
        ActuatorLevel::from_code(((payload >> 8) & 0xff) as u8),
    )
}

// ---------------------------------------------------------------------------
// Parallel capacity
// ---------------------------------------------------------------------------

/// The machine's budget for intra-query parallel work, and the one place that
/// budget is accounted.
///
/// It lives here rather than beside the thread pool because two crates need
/// it and they sit on opposite sides of a dependency edge: the planner decides
/// how many workers to ask for, and the executor owns the threads that serve
/// them. A number the planner cannot see is a number the planner will guess.
///
/// Nothing ever waits on this. A caller takes what is free and runs the rest
/// of its work itself, which is what keeps a nested fan-out from deadlocking
/// against permits its own parent is holding.
#[derive(Debug)]
pub struct ParallelCapacity {
    total: AtomicU32,
    /// What the machine affords before any growth. Kept so growth can be
    /// undone exactly rather than by re-probing, which would drift
    base_total: AtomicU32,
    in_use: AtomicU32,
    /// Percent of a request that may be granted, the controller's cheapest
    /// lever and the only one that acts in microseconds
    scale_pct: AtomicU32,
    granted_total: AtomicU64,
    trimmed_total: AtomicU64,
    serial_fallbacks: AtomicU64,
}

/// Full scale, meaning requests pass through unreduced.
pub const PARALLEL_SCALE_FULL_PCT: u32 = 100;

/// Full memory ceiling, meaning a query gets the whole allowance it asked for.
pub const QUERY_MEMORY_FULL_PCT: u32 = 100;

/// Smallest share of its allowance a query may be handed under memory pressure.
///
/// A quarter, which is enough that an ordinary query still runs and an
/// unbounded one fails on its own limit rather than on the node's. Below this
/// the rung stops relieving pressure and starts refusing ordinary work, which
/// is what the rung above it is for.
pub const QUERY_MEMORY_MIN_PCT: u32 = 25;

/// What the spill point is, as a percentage of what a query may hold, when
/// nothing is forcing operators to spill early.
pub const SPILL_THRESHOLD_FULL_PCT: u32 = 100;

/// What it becomes on the ForceSpill rung. A quarter, so a materializing
/// operator gives most of its working set back to the node and still has room
/// to assemble output batches without writing one to disk per batch.
pub const SPILL_THRESHOLD_FORCED_PCT: u32 = 25;

/// Ceiling on how far the parallel budget may grow past what the cores afford.
///
/// Twice, which is the point where the memory each in-flight worker holds
/// starts to be the binding constraint rather than the thread it is not
/// occupying. Growth only ever applies when the work is waiting on storage, so
/// the extra workers are parked rather than competing for cores.
pub const PARALLEL_GROWTH_MAX_PCT: u32 = 200;

/// Floor the scale cannot fall below, so the lever can always reduce
/// parallelism and can never switch it off.
pub const PARALLEL_SCALE_MIN_PCT: u32 = 10;

static PARALLEL_CAPACITY: std::sync::OnceLock<ParallelCapacity> = std::sync::OnceLock::new();

impl ParallelCapacity {
    /// A standalone budget. The process uses `global`; this exists so a test
    /// can hold an account of a known size instead of asserting against
    /// whatever else the binary is running.
    pub fn with_total(total: u32) -> Self {
        Self::new(total)
    }

    fn new(total: u32) -> Self {
        Self {
            total: AtomicU32::new(total.max(1)),
            base_total: AtomicU32::new(total.max(1)),
            in_use: AtomicU32::new(0),
            scale_pct: AtomicU32::new(PARALLEL_SCALE_FULL_PCT),
            granted_total: AtomicU64::new(0),
            trimmed_total: AtomicU64::new(0),
            serial_fallbacks: AtomicU64::new(0),
        }
    }

    /// The process-wide budget, sized from the machine on first use.
    pub fn global() -> &'static ParallelCapacity {
        PARALLEL_CAPACITY.get_or_init(|| {
            let cores = std::thread::available_parallelism()
                .map(|n| n.get())
                .unwrap_or(4);
            Self::new(cores as u32)
        })
    }

    /// Sets what the machine affords, once at startup from whatever the node
    /// measured or the operator configured.
    ///
    /// Also resets growth, because a new base makes any growth measured
    /// against the old one meaningless.
    pub fn set_total(&self, total: u32) {
        let total = total.max(1);
        self.total.store(total, Ordering::Relaxed);
        self.base_total.store(total, Ordering::Relaxed);
    }

    /// What the machine affords before growth.
    pub fn base_total(&self) -> u32 {
        self.base_total.load(Ordering::Relaxed).max(1)
    }

    /// Raises the in-flight parallel budget above what the machine's cores
    /// afford, as a percentage of the base.
    ///
    /// Only ever correct when the work is waiting rather than computing. A
    /// task parked on storage has released its thread, so the core count
    /// bounds nothing and a budget pinned to it leaves the device idle. The
    /// classifier is what decides that, which is why this rung is illegal
    /// against every bottleneck where the cores are the constraint.
    ///
    /// Clamped at twice the base. Past that the memory each in-flight worker
    /// holds becomes the constraint, and the rung that answers memory pressure
    /// is below this one.
    pub fn set_growth_pct(&self, pct: u32) {
        let pct = pct.clamp(PARALLEL_SCALE_FULL_PCT, PARALLEL_GROWTH_MAX_PCT);
        let base = self.base_total() as u64;
        let grown = (base * pct as u64 / PARALLEL_SCALE_FULL_PCT as u64).max(1);
        self.total
            .store(grown.min(u32::MAX as u64) as u32, Ordering::Relaxed);
    }

    /// How far above the base the budget currently sits, in percent.
    pub fn growth_pct(&self) -> u32 {
        let base = self.base_total() as u64;
        ((self.total() as u64 * PARALLEL_SCALE_FULL_PCT as u64) / base.max(1)) as u32
    }

    pub fn total(&self) -> u32 {
        self.total.load(Ordering::Relaxed)
    }

    pub fn in_use(&self) -> u32 {
        self.in_use.load(Ordering::Relaxed)
    }

    pub fn available(&self) -> u32 {
        self.total().saturating_sub(self.in_use())
    }

    pub fn headroom_fraction(&self) -> f64 {
        let total = self.total();
        if total == 0 {
            return 0.0;
        }
        self.available() as f64 / total as f64
    }

    /// True when the budget is fully committed, which is the CPU half of the
    /// bottleneck classification.
    pub fn saturated(&self) -> bool {
        self.available() == 0
    }

    pub fn set_scale_pct(&self, pct: u32) {
        self.scale_pct.store(
            pct.clamp(PARALLEL_SCALE_MIN_PCT, PARALLEL_SCALE_FULL_PCT),
            Ordering::Relaxed,
        );
    }

    pub fn scale_pct(&self) -> u32 {
        self.scale_pct.load(Ordering::Relaxed)
    }

    /// How many workers a request for `requested` should plan for, taking
    /// nothing. Reflects both the controller's scale and what the rest of the
    /// machine is doing, so the same query planned in a quiet moment and in a
    /// storm gets different answers.
    pub fn advise(&self, requested: usize) -> usize {
        if requested <= 1 {
            return 1;
        }
        // Clamp to what exists before applying the scale, not after. A
        // request for sixty-four workers on a sixteen unit machine is already
        // going to be cut to sixteen, so scaling the request first would let
        // the reduce-parallelism lever pull from sixty-four to sixteen and
        // change nothing at all
        let affordable = requested
            .min(self.available().max(1) as usize)
            .min(self.total() as usize)
            .max(1);
        affordable
            .saturating_mul(self.scale_pct() as usize)
            .div_ceil(PARALLEL_SCALE_FULL_PCT as usize)
            .max(1)
    }

    /// Takes up to `requested` units and reports how many were obtained.
    /// Zero means the caller runs serially.
    pub fn try_take(&self, requested: usize) -> u32 {
        if requested == 0 {
            return 0;
        }
        let want = self.advise(requested) as u32;
        if want == 0 {
            return 0;
        }
        let total = self.total();
        let mut cur = self.in_use.load(Ordering::Relaxed);
        loop {
            let free = total.saturating_sub(cur);
            let take = want.min(free);
            if take == 0 {
                self.serial_fallbacks.fetch_add(1, Ordering::Relaxed);
                return 0;
            }
            match self.in_use.compare_exchange_weak(
                cur,
                cur + take,
                Ordering::AcqRel,
                Ordering::Relaxed,
            ) {
                Ok(_) => {
                    self.granted_total.fetch_add(take as u64, Ordering::Relaxed);
                    if (take as usize) < requested {
                        self.trimmed_total.fetch_add(1, Ordering::Relaxed);
                    }
                    return take;
                }
                Err(observed) => cur = observed,
            }
        }
    }

    /// Returns units. Saturates at zero so a double release cannot invent
    /// capacity the machine does not have.
    pub fn give_back(&self, units: u32) {
        if units == 0 {
            return;
        }
        let mut cur = self.in_use.load(Ordering::Relaxed);
        loop {
            let next = cur.saturating_sub(units);
            match self
                .in_use
                .compare_exchange_weak(cur, next, Ordering::AcqRel, Ordering::Relaxed)
            {
                Ok(_) => return,
                Err(observed) => cur = observed,
            }
        }
    }

    pub fn granted_total(&self) -> u64 {
        self.granted_total.load(Ordering::Relaxed)
    }

    pub fn trimmed_total(&self) -> u64 {
        self.trimmed_total.load(Ordering::Relaxed)
    }

    pub fn serial_fallbacks(&self) -> u64 {
        self.serial_fallbacks.load(Ordering::Relaxed)
    }
}

// ---------------------------------------------------------------------------
// Node memory gauge
// ---------------------------------------------------------------------------

/// Bytes reserved by running queries across the node.
///
/// Distinct from the per-query budget, which records a high-water mark for one
/// query and never gives anything back. Admission needs to know what is in use
/// right now, so this one releases when a query retires.
#[derive(Debug)]
pub struct NodeMemoryGauge {
    reserved: AtomicU64,
    ceiling: AtomicU64,
    /// Highest reservation seen, for the view
    peak: AtomicU64,
    rejections: AtomicU64,
}

impl NodeMemoryGauge {
    pub fn new(ceiling_bytes: u64) -> Self {
        Self {
            reserved: AtomicU64::new(0),
            ceiling: AtomicU64::new(ceiling_bytes),
            peak: AtomicU64::new(0),
            rejections: AtomicU64::new(0),
        }
    }

    /// Takes bytes if they fit. Returns false without reserving when they do
    /// not, so the caller can spill or wait rather than push the node over.
    pub fn try_reserve(&self, bytes: u64) -> bool {
        let ceiling = self.ceiling.load(Ordering::Relaxed);
        let mut cur = self.reserved.load(Ordering::Relaxed);
        loop {
            let next = cur.saturating_add(bytes);
            if next > ceiling {
                self.rejections.fetch_add(1, Ordering::Relaxed);
                return false;
            }
            match self.reserved.compare_exchange_weak(
                cur,
                next,
                Ordering::AcqRel,
                Ordering::Relaxed,
            ) {
                Ok(_) => {
                    self.peak.fetch_max(next, Ordering::Relaxed);
                    return true;
                }
                Err(observed) => cur = observed,
            }
        }
    }

    /// Gives bytes back. Saturates at zero rather than wrapping, so a double
    /// release cannot manufacture headroom.
    pub fn release(&self, bytes: u64) {
        sub_saturating(&self.reserved, bytes);
    }

    pub fn reserved(&self) -> u64 {
        self.reserved.load(Ordering::Relaxed)
    }

    pub fn ceiling(&self) -> u64 {
        self.ceiling.load(Ordering::Relaxed)
    }

    pub fn peak(&self) -> u64 {
        self.peak.load(Ordering::Relaxed)
    }

    pub fn rejections(&self) -> u64 {
        self.rejections.load(Ordering::Relaxed)
    }

    pub fn set_ceiling(&self, bytes: u64) {
        self.ceiling.store(bytes, Ordering::Relaxed);
    }

    /// Free bytes, floored at zero.
    pub fn headroom(&self) -> u64 {
        self.ceiling().saturating_sub(self.reserved())
    }

    /// Fraction of the ceiling still free.
    pub fn headroom_fraction(&self) -> f64 {
        let ceiling = self.ceiling();
        if ceiling == 0 {
            return 0.0;
        }
        self.headroom() as f64 / ceiling as f64
    }
}

/// Releases a node memory reservation when the query that took it ends,
/// including on an early return or a panic unwinding through the executor.
#[derive(Debug)]
pub struct MemoryReservation<'a> {
    gauge: &'a NodeMemoryGauge,
    bytes: u64,
}

impl<'a> MemoryReservation<'a> {
    /// Takes bytes from the gauge, None when they do not fit.
    pub fn acquire(gauge: &'a NodeMemoryGauge, bytes: u64) -> Option<Self> {
        if gauge.try_reserve(bytes) {
            Some(Self { gauge, bytes })
        } else {
            None
        }
    }

    pub fn bytes(&self) -> u64 {
        self.bytes
    }
}

impl Drop for MemoryReservation<'_> {
    fn drop(&mut self) {
        self.gauge.release(self.bytes);
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn seconds_to_us(seconds: f64) -> u64 {
    if !seconds.is_finite() || seconds <= 0.0 {
        return 0;
    }
    (seconds * 1_000_000.0) as u64
}

fn us_to_seconds(us: u64) -> f64 {
    us as f64 / 1_000_000.0
}

fn sub_saturating(cell: &AtomicU64, amount: u64) {
    let mut cur = cell.load(Ordering::Relaxed);
    loop {
        let next = cur.saturating_sub(amount);
        match cell.compare_exchange_weak(cur, next, Ordering::AcqRel, Ordering::Relaxed) {
            Ok(_) => return,
            Err(observed) => cur = observed,
        }
    }
}

fn sub_saturating_u32(cell: &AtomicU32, amount: u32) {
    let mut cur = cell.load(Ordering::Relaxed);
    loop {
        let next = cur.saturating_sub(amount);
        match cell.compare_exchange_weak(cur, next, Ordering::AcqRel, Ordering::Relaxed) {
            Ok(_) => return,
            Err(observed) => cur = observed,
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn classes_round_trip_their_index_and_name() {
        for class in WorkloadClass::ALL {
            assert_eq!(WorkloadClass::from_index(class.index()), Some(class));
            assert_eq!(WorkloadClass::parse(class.as_str()), Some(class));
        }
        assert_eq!(WorkloadClass::from_index(WorkloadClass::COUNT), None);
    }

    #[test]
    fn classification_follows_estimated_cost() {
        assert_eq!(
            WorkloadClass::classify(0.0005, false),
            WorkloadClass::Interactive
        );
        assert_eq!(
            WorkloadClass::classify(0.099, false),
            WorkloadClass::Interactive
        );
        assert_eq!(WorkloadClass::classify(0.100, false), WorkloadClass::Bulk);
        assert_eq!(WorkloadClass::classify(120.0, false), WorkloadClass::Bulk);
        // An explicit background request wins over the estimate either way
        assert_eq!(
            WorkloadClass::classify(0.0001, true),
            WorkloadClass::Background
        );
        assert_eq!(
            WorkloadClass::classify(500.0, true),
            WorkloadClass::Background
        );
    }

    /// The gating is the whole point of the classifier, so each bottleneck's
    /// forbidden rungs are pinned rather than left to the match arm reading
    /// right.
    #[test]
    fn occ_contention_forbids_every_rung_that_adds_concurrency() {
        let legal: Vec<_> = ActuatorLevel::LADDER
            .into_iter()
            .filter(|l| l.legal_for(BottleneckKind::OccContention))
            .collect();
        assert_eq!(
            legal,
            vec![
                ActuatorLevel::ReduceDop,
                ActuatorLevel::DelayAdmission,
                ActuatorLevel::Shed
            ]
        );
        // Named explicitly: provisioning under conflict makes conflict worse
        assert!(!ActuatorLevel::ProvisionNode.legal_for(BottleneckKind::OccContention));
        assert!(!ActuatorLevel::GrowWorkers.legal_for(BottleneckKind::OccContention));
    }

    #[test]
    fn hot_partition_masks_only_the_mesh_rungs() {
        assert!(!ActuatorLevel::ProvisionNode.legal_for(BottleneckKind::HotPartition));
        assert!(!ActuatorLevel::WarmPoolTake.legal_for(BottleneckKind::HotPartition));
        assert!(ActuatorLevel::ReduceDop.legal_for(BottleneckKind::HotPartition));
        assert!(ActuatorLevel::Shed.legal_for(BottleneckKind::HotPartition));
    }

    #[test]
    fn io_bound_forbids_reducing_parallelism() {
        // Cores are idle waiting on the device, taking workers away frees
        // nothing that was scarce
        assert!(!ActuatorLevel::ReduceDop.legal_for(BottleneckKind::Io));
        assert!(ActuatorLevel::TrimMemory.legal_for(BottleneckKind::Io));
    }

    #[test]
    fn memory_bound_forbids_growing_workers() {
        assert!(!ActuatorLevel::GrowWorkers.legal_for(BottleneckKind::Memory));
        assert!(ActuatorLevel::TrimMemory.legal_for(BottleneckKind::Memory));
    }

    #[test]
    fn ladder_is_ordered_by_cost() {
        let mut previous = 0u8;
        for level in ActuatorLevel::LADDER {
            assert!(
                level.code() > previous,
                "{} is out of order",
                level.as_str()
            );
            previous = level.code();
        }
    }

    #[test]
    fn pressure_is_work_over_measured_capacity() {
        let counters = ClassCounters::new();
        counters.enqueue(2.0);
        counters.enqueue(3.0);
        // No capacity measured yet, so the honest reading is the raw work
        assert!((counters.pressure_seconds() - 5.0).abs() < 1e-9);

        // Ten queries a second draining five work-seconds is half a second of
        // pressure
        counters.window_completions.store(10, Ordering::Relaxed);
        counters.roll_window(Duration::from_secs(1));
        assert!((counters.service_capacity_qps() - 10.0).abs() < 1e-6);
        assert!((counters.pressure_seconds() - 0.5).abs() < 1e-6);
    }

    #[test]
    fn counters_track_a_query_through_its_whole_life() {
        let counters = ClassCounters::new();
        counters.enqueue(0.25);
        assert_eq!(counters.queued_count(), 1);
        assert!((counters.queued_work_seconds() - 0.25).abs() < 1e-9);

        counters.start(0.25);
        assert_eq!(counters.queued_count(), 0);
        assert_eq!(counters.in_flight(), 1);
        assert!(counters.queued_work_seconds().abs() < 1e-9);
        assert!((counters.active_work_seconds() - 0.25).abs() < 1e-9);

        counters.complete(0.25, 0.30);
        assert_eq!(counters.in_flight(), 0);
        assert!(counters.active_work_seconds().abs() < 1e-9);
    }

    /// A completion that overshoots its estimate must not wrap the counter
    /// into a huge positive number, which would read as permanent pressure.
    #[test]
    fn over_release_saturates_at_zero() {
        let counters = ClassCounters::new();
        counters.start_direct(0.1);
        counters.complete(5.0, 5.0);
        assert!(counters.active_work_seconds().abs() < 1e-9);
        assert_eq!(counters.in_flight(), 0);
        counters.complete(1.0, 1.0);
        assert!(counters.active_work_seconds().abs() < 1e-9);
        assert_eq!(counters.in_flight(), 0);
    }

    #[test]
    fn versioned_encoding_is_monotone_in_the_sequence() {
        // The payload falling must not make the packed value fall, or the
        // max merge would keep the stale reading forever
        let older = encode_versioned(7, 900_000);
        let newer = encode_versioned(8, 1);
        assert!(newer > older);
        assert_eq!(decode_versioned(newer), (8, 1));
        assert_eq!(decode_versioned(older), (7, 900_000));
    }

    #[test]
    fn pressure_keys_round_trip_and_reject_foreign_keys() {
        for class in WorkloadClass::ALL {
            for field in PressureField::ALL {
                let key = pressure_key(42, class, field);
                assert_eq!(parse_pressure_key(&key), Some((42, class, field)));
            }
        }
        // A quota key sharing the transport must not be read as pressure
        assert_eq!(parse_pressure_key("tenant/acme/rows"), None);
        assert_eq!(parse_pressure_key("zpressure/42/interactive"), None);
        assert_eq!(parse_pressure_key("zpressure/42/nosuch/pressure_us"), None);
    }

    #[test]
    fn status_codes_round_trip() {
        for bottleneck in [
            BottleneckKind::None,
            BottleneckKind::Cpu,
            BottleneckKind::Io,
            BottleneckKind::Memory,
            BottleneckKind::OccContention,
            BottleneckKind::HotPartition,
            BottleneckKind::FsyncBound,
        ] {
            for actuator in ActuatorLevel::LADDER {
                let packed = encode_status(bottleneck, actuator);
                assert_eq!(decode_status(packed), (bottleneck, actuator));
            }
        }
    }

    #[test]
    fn memory_gauge_releases_what_the_budget_never_does() {
        let gauge = NodeMemoryGauge::new(1000);
        assert!(gauge.try_reserve(600));
        assert_eq!(gauge.reserved(), 600);
        assert!(!gauge.try_reserve(500));
        assert_eq!(gauge.reserved(), 600, "a refused reservation takes nothing");
        assert_eq!(gauge.rejections(), 1);
        gauge.release(600);
        assert_eq!(gauge.reserved(), 0);
        assert_eq!(gauge.peak(), 600);
    }

    #[test]
    fn memory_reservation_gives_bytes_back_when_it_drops() {
        let gauge = NodeMemoryGauge::new(1000);
        {
            let held = MemoryReservation::acquire(&gauge, 400).expect("fits");
            assert_eq!(held.bytes(), 400);
            assert_eq!(gauge.reserved(), 400);
        }
        assert_eq!(gauge.reserved(), 0);
        assert!(MemoryReservation::acquire(&gauge, 2000).is_none());
    }

    #[test]
    fn overall_utilization_takes_the_worst_class_with_an_objective() {
        let mut node = NodePressure {
            node_id: 1,
            classes: vec![
                ClassPressure {
                    class: WorkloadClass::Interactive,
                    pressure_seconds: 0.050,
                    slo_seconds: 0.100,
                    ..blank(WorkloadClass::Interactive)
                },
                ClassPressure {
                    class: WorkloadClass::Bulk,
                    pressure_seconds: 45.0,
                    slo_seconds: 30.0,
                    ..blank(WorkloadClass::Bulk)
                },
                ClassPressure {
                    class: WorkloadClass::Background,
                    pressure_seconds: 9000.0,
                    slo_seconds: f64::INFINITY,
                    ..blank(WorkloadClass::Background)
                },
            ],
            overall_utilization: 0.0,
            bottleneck: BottleneckKind::None,
            actuator_level: ActuatorLevel::Steady,
            parallel_permits_total: 8,
            parallel_permits_available: 8,
            memory_reserved_bytes: 0,
            memory_ceiling_bytes: 0,
            updated_at_us: 0,
        };
        node.recompute_overall();
        assert!((node.overall_utilization - 1.5).abs() < 1e-9);
        // Background carries no objective, so its enormous backlog cannot
        // make the node look breached
        assert!(!node.classes[2].breaching());
    }

    fn blank(class: WorkloadClass) -> ClassPressure {
        ClassPressure {
            class,
            queued_work_seconds: 0.0,
            active_work_seconds: 0.0,
            queued_count: 0,
            in_flight: 0,
            service_capacity_qps: 0.0,
            mean_service_seconds: 0.0,
            pressure_seconds: 0.0,
            calibration_error: 1.0,
            ceiling: 0,
            slo_seconds: class.slo_seconds(),
            admitted_total: 0,
            delayed_total: 0,
            shed_total: 0,
            bypassed_total: 0,
            bottleneck: BottleneckKind::None,
            actuator_level: ActuatorLevel::Steady,
        }
    }
}
