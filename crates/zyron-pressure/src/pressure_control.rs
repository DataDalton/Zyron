//! The pressure controller: admission, bottleneck classification, and the
//! actuator ladder.
//!
//! It replaces threshold-with-cooldown scaling. A threshold encodes a belief
//! about where the machine's limit is, and that belief is wrong on every
//! machine except the one it was measured on. This controller measures instead:
//! it watches what throughput does as concurrency rises, keeps the point where
//! throughput stopped improving, and treats that as the limit.
//!
//! Two things make it more than a feedback loop.
//!
//! The signal is denominated in seconds of work rather than in a queue length,
//! so it compares directly against a latency objective without a conversion
//! constant, and a queue of cheap queries and a queue of expensive ones are not
//! confused for each other.
//!
//! And the response is gated on what is actually saturated. Most saturation has
//! a cause that capacity makes worse: conflict rises with concurrency, a hot key
//! does not spread, and a writer waiting on a disk is not waiting on a core. The
//! classifier runs before the ladder, so those cases get the lever that helps
//! rather than the lever that is easiest to reach for.
//!
//! Nothing here reads the core count. Query concurrency is not bounded by cores,
//! because a query parked on storage has released its thread, and a controller
//! that assumes otherwise idles the machine.

use std::sync::Mutex;
use std::sync::atomic::{AtomicBool, AtomicI64, AtomicU32, AtomicU64, Ordering};
use std::time::{Duration, Instant};

use crate::capability::{CoefficientAccumulator, ContentionSignals, LatencyHistogram};
use crate::hot_set::HotQuery;
use crate::pressure::{
    ActuatorDecision, ActuatorLevel, AdmitDecision, BottleneckKind, ClassCounters, ClassPressure,
    NodeMemoryGauge, NodePressure, ParallelCapacity, WorkloadClass,
};
use crate::projection::{
    ArrivalSample, ArrivalTrend, PressureProjection, ProjectionInputs, TREND_WINDOW,
    fit_arrival_trend, project as project_pressure,
};
use crate::provisioner::ProvisionerCapabilities;

// ---------------------------------------------------------------------------
// Tuning that is not tuning
// ---------------------------------------------------------------------------

/// Where the concurrency ceiling starts before anything has been measured.
///
/// Not a limit and not derived from the hardware: the knee finder moves it
/// within the first few windows of real traffic. It exists only because an
/// additive-increase search has to start somewhere, and starting at one would
/// make the first seconds after boot artificially slow.
const INITIAL_CEILING: u32 = 8;

/// Ceiling below which the controller will not go. One query at a time is a
/// stalled node, not a throttled one.
const MIN_CEILING: u32 = 1;

/// Throughput must beat the best seen by this much to count as an improvement
/// rather than as measurement noise.
const IMPROVEMENT_MARGIN: f64 = 1.02;

/// Multiplicative back-off applied when the objective is breached.
const BACKOFF_FACTOR: f64 = 0.8;

/// Multiplicative growth applied while throughput is still improving.
const GROWTH_FACTOR: f64 = 1.25;

/// Conflict-abort share above which capacity is the wrong answer.
const OCC_ABORT_THRESHOLD: f64 = 0.05;

/// Share of key traffic on the hottest keys above which the load is skewed
/// rather than heavy.
const HOT_KEY_THRESHOLD: f64 = 0.30;

/// Group-commit hit rate below which writers are paying a device round trip
/// each, provided some are actually waiting.
const GROUP_COMMIT_THRESHOLD: f64 = 0.50;

/// Free parallel capacity below which the node is considered CPU bound.
const CPU_HEADROOM_THRESHOLD: f64 = 0.10;

/// Free parallel capacity above which the cores are idle enough that a storage
/// wait, not compute, is the constraint.
const IO_IDLE_CPU_THRESHOLD: f64 = 0.40;

/// Free query-execution memory below which the node is memory bound.
const MEMORY_HEADROOM_THRESHOLD: f64 = 0.10;

/// Share of the machine's memory that query execution may hold at once.
///
/// Not the whole machine. The buffer pool takes its own quarter, connections
/// are bounded against another, and the operating system, the write-ahead log,
/// and the catalog all need room. A gauge measured against total memory would
/// read as full headroom on a node that is minutes from an out-of-memory kill,
/// which is the same as having no gauge.
const QUERY_MEMORY_SHARE: f64 = 0.25;

/// Fraction of the objective under which the node is considered to have
/// headroom worth exploring into.
const EXPLORE_UTILIZATION: f64 = 0.50;

/// How long that headroom must hold before the controller spends any of it on
/// finding out whether the knee has moved.
const EXPLORE_QUIET_PERIOD: Duration = Duration::from_secs(300);

/// How far past the believed knee an exploration step reaches.
const EXPLORE_STEP_PCT: u32 = 10;

/// Parallelism scale applied when the ReduceDop rung fires.
const REDUCED_DOP_PCT: u32 = 75;

/// How long a delayed arrival waits before it is reconsidered.
const ADMISSION_DELAY: Duration = Duration::from_millis(5);

/// Ticks of history kept, at the publish cadence. One hour at 100ms.
const HISTORY_CAPACITY: usize = 36_000;

/// Admission decisions kept for the view.
const ADMISSIONS_CAPACITY: usize = 8_192;

/// Exploration outcomes kept.
const LEDGER_CAPACITY: usize = 256;

/// How long a rung stays in force before the controller judges whether it
/// worked.
///
/// The pressure signal is rebuilt once per measurement window, and one window
/// of a moved number is noise. Five is enough for the effect of a lever to be
/// visible above that, and short enough that a node genuinely in trouble
/// reaches shedding in a few seconds rather than a few minutes.
///
/// Deliberately not scaled by the class objective. What is being waited for is
/// the measurement settling, which is a property of the signal and identical
/// for every class, not of how fast that class is expected to answer.
const LADDER_SETTLE: Duration = Duration::from_millis(500);

/// How long a bottleneck classification suppresses the rungs it forbids, so a
/// single quiet window cannot immediately re-enable provisioning against a hot
/// partition that has not actually gone away.
const CLASSIFICATION_HOLD: Duration = Duration::from_secs(60);

// ---------------------------------------------------------------------------
// Connection admission
// ---------------------------------------------------------------------------

/// Bytes an idle connection is assumed to hold: its read and write buffers
/// plus session state.
///
/// A measurement would be better and is what the passive collector will
/// eventually supply. Until then this errs high, because the failure mode of
/// guessing low is an out-of-memory kill and the failure mode of guessing high
/// is refusing a connection the node could have served.
const CONNECTION_BYTES_ESTIMATE: u64 = 64 * 1024;

/// Share of node memory connections may occupy. The rest belongs to query
/// working sets, which is where the memory actually earns anything.
const CONNECTION_MEMORY_SHARE: f64 = 0.25;

/// How many clients may be connected at once.
///
/// Not a configured count. A connection costs memory, so the limit is what the
/// memory affords, and on a machine with more memory it is automatically
/// higher. The failure this replaces was a fixed hundred that no deployment
/// could raise past its own hardware.
#[derive(Debug)]
pub struct ConnectionGauge {
    live: AtomicU32,
    ceiling: AtomicU32,
    accepted_total: AtomicU64,
    refused_total: AtomicU64,
    peak: AtomicU32,
}

impl ConnectionGauge {
    /// Derives the ceiling from the memory the node reported.
    pub fn from_memory(mem_total_bytes: u64) -> Self {
        let budget = (mem_total_bytes as f64 * CONNECTION_MEMORY_SHARE) as u64;
        let ceiling = (budget / CONNECTION_BYTES_ESTIMATE).clamp(1, u32::MAX as u64) as u32;
        Self {
            live: AtomicU32::new(0),
            ceiling: AtomicU32::new(ceiling),
            accepted_total: AtomicU64::new(0),
            refused_total: AtomicU64::new(0),
            peak: AtomicU32::new(0),
        }
    }

    /// Takes a slot, or reports that the node has no memory left for one.
    pub fn try_accept(&self) -> bool {
        let ceiling = self.ceiling.load(Ordering::Relaxed);
        let mut cur = self.live.load(Ordering::Relaxed);
        loop {
            if cur >= ceiling {
                self.refused_total.fetch_add(1, Ordering::Relaxed);
                return false;
            }
            match self
                .live
                .compare_exchange_weak(cur, cur + 1, Ordering::AcqRel, Ordering::Relaxed)
            {
                Ok(_) => {
                    self.accepted_total.fetch_add(1, Ordering::Relaxed);
                    self.peak.fetch_max(cur + 1, Ordering::Relaxed);
                    return true;
                }
                Err(observed) => cur = observed,
            }
        }
    }

    /// Gives a slot back when a connection closes.
    pub fn release(&self) {
        let mut cur = self.live.load(Ordering::Relaxed);
        loop {
            let next = cur.saturating_sub(1);
            match self
                .live
                .compare_exchange_weak(cur, next, Ordering::AcqRel, Ordering::Relaxed)
            {
                Ok(_) => return,
                Err(observed) => cur = observed,
            }
        }
    }

    pub fn live(&self) -> u32 {
        self.live.load(Ordering::Relaxed)
    }

    pub fn ceiling(&self) -> u32 {
        self.ceiling.load(Ordering::Relaxed)
    }

    pub fn set_ceiling(&self, ceiling: u32) {
        self.ceiling.store(ceiling.max(1), Ordering::Relaxed);
    }

    pub fn accepted_total(&self) -> u64 {
        self.accepted_total.load(Ordering::Relaxed)
    }

    pub fn refused_total(&self) -> u64 {
        self.refused_total.load(Ordering::Relaxed)
    }

    pub fn peak(&self) -> u32 {
        self.peak.load(Ordering::Relaxed)
    }
}

/// Releases a connection slot when the connection ends, including on an error
/// path or a panic unwinding out of the connection task.
#[derive(Debug)]
pub struct ConnectionSlot {
    gauge: &'static ConnectionGauge,
}

impl ConnectionSlot {
    /// Takes a slot from the process controller, None when the node is full.
    pub fn acquire() -> Option<Self> {
        let gauge = PressureController::global().connections();
        if gauge.try_accept() {
            Some(Self { gauge })
        } else {
            None
        }
    }
}

impl Drop for ConnectionSlot {
    fn drop(&mut self) {
        self.gauge.release();
    }
}

// ---------------------------------------------------------------------------
// Controller
// ---------------------------------------------------------------------------

/// Per-class adaptive state. Not atomic: only the controller tick touches it,
/// under one lock, at the publish cadence.
#[derive(Debug, Clone)]
struct ClassKnee {
    /// Concurrency the controller currently allows
    ceiling: u32,
    /// Best throughput seen, and the concurrency it was seen at
    best_throughput: f64,
    best_concurrency: u32,
    /// When the class last sat comfortably under its objective
    calm_since: Option<Instant>,
    /// Set while an exploration step is outstanding, so its outcome can be
    /// scored against what the ceiling was before it
    exploring_from: Option<u32>,
    /// Smoothed ratio of measured cost to estimated cost
    calibration_error: f64,
    /// Estimated work charged for completions in the current window, so the
    /// ratio has both halves
    window_estimated_seconds: f64,
    window_actual_seconds: f64,
    /// The rung currently in force for this class, and when it was last moved.
    /// Held per class because the classes breach independently and a rung
    /// applied for one of them says nothing about the others
    rung: ActuatorLevel,
    rung_since: Option<Instant>,
}

impl ClassKnee {
    fn new() -> Self {
        Self {
            ceiling: INITIAL_CEILING,
            best_throughput: 0.0,
            best_concurrency: 0,
            calm_since: None,
            exploring_from: None,
            calibration_error: 1.0,
            window_estimated_seconds: 0.0,
            window_actual_seconds: 0.0,
            rung: ActuatorLevel::Steady,
            rung_since: None,
        }
    }
}

/// Query shapes tracked at once.
///
/// Far above the number of distinct statements a served application has, and
/// below what a node generating statement text per request would produce. The
/// bound exists so the second case cannot grow the map without limit, not to
/// ration the first.
const SHAPE_CAPACITY: usize = 4_096;

/// One entry in the history ring.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct HistorySample {
    /// Wall clock, for a reader that wants to line this up against a log line
    pub at_us: i64,
    /// Microseconds since the controller started.
    ///
    /// What the trend fit measures ages against, because a wall clock can be
    /// stepped and a step would turn a minute of history into a slope that
    /// never happened. A scale-out taken from that slope would provision
    /// against a clock correction
    pub elapsed_us: u64,
    pub pressure_us: [u32; WorkloadClass::COUNT],
    pub queued_us: [u32; WorkloadClass::COUNT],
    pub active_us: [u32; WorkloadClass::COUNT],
    pub in_flight: [u16; WorkloadClass::COUNT],
    pub capacity_milli_qps: [u32; WorkloadClass::COUNT],
    /// Queries that arrived during the window this sample closed. The
    /// projection differentiates this, so it is a count per window rather
    /// than a running total: a total would need the reader to reconstruct the
    /// window boundaries the controller already knows
    pub arrivals: [u32; WorkloadClass::COUNT],
    /// How long that window was, so a tick that ran late does not read as a
    /// burst of arrivals
    pub window_us: u32,
    pub bottleneck: BottleneckKind,
    pub actuator: ActuatorLevel,
}

impl HistorySample {
    /// Arrivals per second for one class over this sample's window.
    ///
    pub fn arrival_rate(&self, class: WorkloadClass) -> f64 {
        if self.window_us == 0 {
            return 0.0;
        }
        self.arrivals[class.index()] as f64 * 1e6 / self.window_us as f64
    }
}

/// One admission decision, kept for the view.
#[derive(Debug, Clone)]
pub struct AdmissionRecord {
    pub at_us: i64,
    pub class: WorkloadClass,
    pub decision: &'static str,
    pub estimated_work_seconds: f64,
    pub queue_wait_us: u64,
    pub tenant_id: Option<String>,
    pub reason: &'static str,
}

/// One exploration step and what it produced.
#[derive(Debug, Clone)]
pub struct LedgerEntry {
    pub at_us: i64,
    pub class: WorkloadClass,
    pub ceiling_before: u32,
    pub ceiling_after: u32,
    pub throughput_before: f64,
    pub throughput_after: f64,
    pub pressure_seconds: f64,
    pub kept: bool,
}

/// Per-tenant pressure attribution, so scale-out cost lands on whoever caused
/// it and a noisy neighbour is visible rather than inferred.
#[derive(Debug, Default, Clone)]
struct TenantWork {
    queued_us: u64,
    active_us: u64,
    completed: u64,
    completed_us: u64,
}

/// What one query shape has cost, for the working-set manifest.
#[derive(Debug, Default, Clone, Copy)]
struct ShapeWork {
    executions: u64,
    work_us: u64,
}

/// The node's pressure controller.
pub struct PressureController {
    node_id: u64,
    counters: [ClassCounters; WorkloadClass::COUNT],
    memory: NodeMemoryGauge,
    contention: ContentionSignals,
    coefficients: CoefficientAccumulator,
    read_latency: LatencyHistogram,
    capacity: &'static ParallelCapacity,
    connections: ConnectionGauge,

    /// Adaptive state, touched only by the tick
    knees: Mutex<[ClassKnee; WorkloadClass::COUNT]>,
    history: Mutex<Ring<HistorySample>>,
    admissions: Mutex<Ring<AdmissionRecord>>,
    ledger: Mutex<Ring<LedgerEntry>>,
    tenants: Mutex<std::collections::HashMap<String, TenantWork>>,
    /// What the installed provisioner can reach, packed into two bits so the
    /// ladder can consult it without taking a lock
    mesh_reach: AtomicU32,
    /// Arrival totals as of the previous tick, so the history ring can carry a
    /// per-window count without every reader re-deriving it
    last_arrivals: [AtomicU64; WorkloadClass::COUNT],
    /// The query shapes this node has been serving, for the working-set
    /// manifest a draining node hands to its survivors
    shapes: Mutex<std::collections::HashMap<u64, ShapeWork>>,
    /// How long capacity takes to arrive here and how much of it the operator
    /// will pay to keep idle. Published by the loop that owns the driver, so
    /// the views, the endpoint, and the ladder all project against one number
    /// rather than each deriving their own
    provision_horizon_ms: AtomicU64,
    warm_pool_cap: AtomicU32,
    /// State of the last working-set manifest, which is one of the two
    /// conditions on going to zero
    hot_set_pages: AtomicU32,
    hot_set_generated_us: AtomicI64,
    hot_set_persisted: AtomicBool,
    /// When the last checkpoint completed and whether it closed cleanly.
    ///
    /// Published here rather than read from the checkpoint worker, because
    /// three things need it and a reader that reached into the worker would be
    /// a second source of truth for the number scale to zero is judged on
    checkpoint_at_us: AtomicI64,
    checkpoint_clean: AtomicBool,
    wal_bytes_since_checkpoint: AtomicU64,

    /// Last classification and when it was made, so a forbidden rung stays
    /// forbidden for the hold period rather than for one window
    bottleneck: AtomicU32,
    bottleneck_at: Mutex<Option<Instant>>,
    actuator: AtomicU32,

    /// When the controller started, which is what history ages are measured
    /// against
    started: Instant,
    /// Share of its configured allowance a query starting now may hold
    query_memory_pct: AtomicU32,
    /// Share of that allowance a materializing operator may hold before it
    /// starts writing to disk
    spill_threshold_pct: AtomicU32,
    /// What one query is configured to be allowed to hold, before either of
    /// the two shares above are applied. Zero means unlimited, which is the
    /// default and means nothing spills
    configured_query_memory: AtomicU64,
    /// What an extension said the last time the ladder reached a rung this
    /// node cannot perform itself, and which rung that was. Kept so a view
    /// can show why a rung the controller climbed to did nothing
    last_extension: Mutex<Option<(ActuatorLevel, String)>>,
    /// Monotonic tick counter, also the gossip sequence number
    sequence: AtomicU32,
    last_tick: Mutex<Option<Instant>>,
    ticks: AtomicU64,
}

/// The process controller. One node, one controller, and the wire layer and
/// the background loop both need it, so it lives here rather than threaded
/// through a state struct that thirty-odd call sites build by hand.
static CONTROLLER: std::sync::OnceLock<PressureController> = std::sync::OnceLock::new();

/// Forwards a conflict abort from the error type into this node's signal.
///
/// A separate zero-sized type rather than the controller itself, because the
/// controller is behind a OnceLock and a sink that borrowed it would have to
/// be installed after initialization rather than during it.
struct ControllerConflictSink;

impl zyron_common::conflict_signal::ConflictSink for ControllerConflictSink {
    fn record_conflict_abort(&self) {
        PressureController::global()
            .contention()
            .record_conflict_abort();
    }
}

static CONFLICT_SINK: ControllerConflictSink = ControllerConflictSink;

impl PressureController {
    /// Installs the process controller. Returns false when one already exists,
    /// which happens when something touched it before startup wiring ran.
    pub fn init(node_id: u64, memory_ceiling_bytes: u64) -> bool {
        CONTROLLER
            .set(Self::new(node_id, memory_ceiling_bytes))
            .is_ok()
    }

    /// The process controller, built with a memory ceiling read from the
    /// machine if startup has not installed one yet.
    pub fn global() -> &'static PressureController {
        CONTROLLER.get_or_init(|| {
            // Producing a transaction conflict is the event the conflict
            // signal is made of, and the error type that produces it lives
            // below this crate. Installing here rather than at startup means
            // the count is connected as soon as anything asks the controller
            // anything, which is before any query has run
            zyron_common::conflict_signal::install_conflict_sink(&CONFLICT_SINK);
            let mut system = sysinfo::System::new();
            system.refresh_memory();
            Self::new(0, system.total_memory())
        })
    }

    /// Connects this crate to the error type that reports conflicts.
    ///
    /// Called from startup as well as lazily from , because a node
    /// that has not touched the controller yet can still abort a transaction,
    /// and a conflict that nothing counted is a conflict the classifier will
    /// not see.
    pub fn install_signals() {
        zyron_common::conflict_signal::install_conflict_sink(&CONFLICT_SINK);
    }

    pub fn new(node_id: u64, memory_ceiling_bytes: u64) -> Self {
        Self::with_capacity(node_id, memory_ceiling_bytes, ParallelCapacity::global())
    }

    /// A controller accounting parallel work against a budget of its own.
    ///
    /// The process has one budget and one controller, so this exists for the
    /// tests: two controllers sharing the global account would each see the
    /// other's grants and neither would be measuring what it thinks.
    pub fn with_capacity(
        node_id: u64,
        memory_ceiling_bytes: u64,
        capacity: &'static ParallelCapacity,
    ) -> Self {
        let counters: [ClassCounters; WorkloadClass::COUNT] =
            std::array::from_fn(|_| ClassCounters::new());
        // Publish the starting ceiling before the first tick. Left at zero a
        // peer reading this node would see one that admits nothing, which is
        // the opposite of what a freshly started node is doing
        for class in WorkloadClass::ALL {
            counters[class.index()].set_ceiling(INITIAL_CEILING);
        }
        Self {
            node_id,
            counters,
            memory: NodeMemoryGauge::new((memory_ceiling_bytes as f64 * QUERY_MEMORY_SHARE) as u64),
            contention: ContentionSignals::new(),
            coefficients: CoefficientAccumulator::new(),
            read_latency: LatencyHistogram::new(),
            capacity,
            connections: ConnectionGauge::from_memory(memory_ceiling_bytes),
            knees: Mutex::new(std::array::from_fn(|_| ClassKnee::new())),
            history: Mutex::new(Ring::new(HISTORY_CAPACITY)),
            admissions: Mutex::new(Ring::new(ADMISSIONS_CAPACITY)),
            ledger: Mutex::new(Ring::new(LEDGER_CAPACITY)),
            tenants: Mutex::new(std::collections::HashMap::new()),
            // Nothing until a provisioner says otherwise. A node that claims
            // it can reach hardware it cannot would climb to a rung that
            // never relieves anything and stop shedding
            mesh_reach: AtomicU32::new(MeshReach::NONE.bits()),
            last_arrivals: std::array::from_fn(|_| AtomicU64::new(0)),
            shapes: Mutex::new(std::collections::HashMap::new()),
            provision_horizon_ms: AtomicU64::new(0),
            warm_pool_cap: AtomicU32::new(0),
            hot_set_pages: AtomicU32::new(0),
            hot_set_generated_us: AtomicI64::new(0),
            hot_set_persisted: AtomicBool::new(false),
            checkpoint_at_us: AtomicI64::new(0),
            checkpoint_clean: AtomicBool::new(false),
            wal_bytes_since_checkpoint: AtomicU64::new(0),
            bottleneck: AtomicU32::new(BottleneckKind::None.code() as u32),
            bottleneck_at: Mutex::new(None),
            actuator: AtomicU32::new(ActuatorLevel::Steady.code() as u32),
            query_memory_pct: AtomicU32::new(crate::pressure::QUERY_MEMORY_FULL_PCT),
            spill_threshold_pct: AtomicU32::new(crate::pressure::SPILL_THRESHOLD_FULL_PCT),
            configured_query_memory: AtomicU64::new(0),
            last_extension: Mutex::new(None),
            started: Instant::now(),
            sequence: AtomicU32::new(0),
            last_tick: Mutex::new(None),
            ticks: AtomicU64::new(0),
        }
    }

    pub fn node_id(&self) -> u64 {
        self.node_id
    }

    pub fn counters(&self, class: WorkloadClass) -> &ClassCounters {
        &self.counters[class.index()]
    }

    pub fn memory(&self) -> &NodeMemoryGauge {
        &self.memory
    }

    pub fn contention(&self) -> &ContentionSignals {
        &self.contention
    }

    pub fn coefficients(&self) -> &CoefficientAccumulator {
        &self.coefficients
    }

    pub fn read_latency(&self) -> &LatencyHistogram {
        &self.read_latency
    }

    pub fn capacity(&self) -> &'static ParallelCapacity {
        self.capacity
    }

    pub fn connections(&self) -> &ConnectionGauge {
        &self.connections
    }

    pub fn sequence(&self) -> u32 {
        self.sequence.load(Ordering::Relaxed)
    }

    pub fn ticks(&self) -> u64 {
        self.ticks.load(Ordering::Relaxed)
    }

    pub fn bottleneck(&self) -> BottleneckKind {
        BottleneckKind::from_code(self.bottleneck.load(Ordering::Relaxed) as u8)
    }

    pub fn actuator(&self) -> ActuatorLevel {
        ActuatorLevel::from_code(self.actuator.load(Ordering::Relaxed) as u8)
    }

    pub fn ceiling(&self, class: WorkloadClass) -> u32 {
        self.knees
            .lock()
            .map(|k| k[class.index()].ceiling)
            .unwrap_or(INITIAL_CEILING)
    }

    // -----------------------------------------------------------------------
    // Admission
    // -----------------------------------------------------------------------

    /// Decides what to do with an arriving query.
    ///
    /// Cheap queries never reach the decision: below the bypass threshold the
    /// bookkeeping costs more than running the query, so they run.
    pub fn admit(
        &self,
        estimated_work_seconds: f64,
        requested_background: bool,
        tenant_id: Option<&str>,
    ) -> (WorkloadClass, AdmitDecision) {
        let class = WorkloadClass::classify(estimated_work_seconds, requested_background);
        let counters = self.counters(class);

        if estimated_work_seconds < WorkloadClass::BYPASS_WORK_SECONDS {
            counters.record_bypass();
            counters.start_direct(estimated_work_seconds);
            self.charge_tenant_active(tenant_id, estimated_work_seconds);
            return (class, AdmitDecision::Bypass);
        }

        let in_flight = counters.in_flight();
        let ceiling = self.ceiling(class);
        if in_flight < ceiling {
            counters.record_admit();
            counters.start_direct(estimated_work_seconds);
            self.charge_tenant_active(tenant_id, estimated_work_seconds);
            self.record_admission(
                class,
                "admit",
                estimated_work_seconds,
                0,
                tenant_id,
                "under ceiling",
            );
            return (class, AdmitDecision::Admit);
        }

        // Past the ceiling, so the query waits. Past the queue as well and it
        // is refused, because a queue deeper than the objective can drain is
        // a breach that has not been reported yet
        let queued = counters.queued_count() as usize;
        if queued >= class.queue_depth() {
            counters.record_shed();
            let reason = format!(
                "node is shedding {} work: {} queued at ceiling {}, objective {:.3}s",
                class.as_str(),
                queued,
                ceiling,
                class.slo_seconds()
            );
            self.record_admission(
                class,
                "shed",
                estimated_work_seconds,
                0,
                tenant_id,
                "queue depth reached",
            );
            return (class, AdmitDecision::Shed { reason });
        }

        counters.record_delay();
        counters.enqueue(estimated_work_seconds);
        self.charge_tenant_queued(tenant_id, estimated_work_seconds);
        self.record_admission(
            class,
            "delay",
            estimated_work_seconds,
            0,
            tenant_id,
            "at ceiling",
        );
        (class, AdmitDecision::Delay(ADMISSION_DELAY))
    }

    /// Re-decides for a query that has already waited once. Returns Admit as
    /// soon as the class has room, so a delayed query does not re-enter the
    /// queue behind arrivals that came after it.
    pub fn retry_queued(
        &self,
        class: WorkloadClass,
        estimated_work_seconds: f64,
        waited: Duration,
        tenant_id: Option<&str>,
    ) -> AdmitDecision {
        let counters = self.counters(class);
        if counters.in_flight() < self.ceiling(class) {
            counters.start(estimated_work_seconds);
            counters.record_admit();
            self.charge_tenant_active(tenant_id, estimated_work_seconds);
            self.record_admission(
                class,
                "admit",
                estimated_work_seconds,
                waited.as_micros() as u64,
                tenant_id,
                "left the queue",
            );
            return AdmitDecision::Admit;
        }
        AdmitDecision::Delay(ADMISSION_DELAY)
    }

    /// Drops a query that queued and then gave up or was cancelled.
    pub fn abandon_queued(&self, class: WorkloadClass, estimated_work_seconds: f64) {
        self.counters(class).abandon_queued(estimated_work_seconds);
    }

    /// Records one timed page read into both the storage histogram and the
    /// page-read coefficient.
    ///
    /// Called from the read path under sampling, so the weight says how many
    /// untimed reads this one stands for and the coefficient ends up
    /// describing the whole traffic rather than the sampled slice.
    #[inline]
    pub fn record_page_read(&self, elapsed: Duration, bytes: u64, weight: u64) {
        let nanos = elapsed.as_nanos().min(u64::MAX as u128) as u64;
        self.read_latency.record(nanos, bytes);
        // The whole traffic, not the sampled slice, because the classifier
        // asks whether the node is reading rather than how often it samples
        self.contention.record_page_reads(weight);
        self.coefficients.record(
            crate::capability::OperatorKind::PageRead,
            weight,
            nanos.saturating_mul(weight),
        );
    }

    /// Records one operator batch: how many rows it moved and how long it took.
    ///
    /// Called once per batch rather than once per row, so a clock pair here is
    /// spread over thousands of rows and needs no sampling.
    #[inline]
    pub fn record_operator(
        &self,
        kind: crate::capability::OperatorKind,
        units: u64,
        elapsed: Duration,
    ) {
        self.coefficients.record_elapsed(kind, units, elapsed);
    }

    /// Folds what the accumulator has gathered into the live coefficients.
    /// Driven by the calibration collector rather than by the query path.
    pub fn drain_calibration(&self) -> crate::capability::DrainSummary {
        self.coefficients.drain()
    }

    /// Retires a query and feeds the cost model what it actually cost.
    pub fn complete(
        &self,
        class: WorkloadClass,
        estimated_work_seconds: f64,
        actual_seconds: f64,
        tenant_id: Option<&str>,
    ) {
        self.counters(class)
            .complete(estimated_work_seconds, actual_seconds);
        if let Ok(mut knees) = self.knees.lock() {
            let knee = &mut knees[class.index()];
            knee.window_estimated_seconds += estimated_work_seconds;
            knee.window_actual_seconds += actual_seconds;
        }
        if let Some(tenant) = tenant_id {
            if let Ok(mut tenants) = self.tenants.lock() {
                let entry = tenants.entry(tenant.to_string()).or_default();
                entry.active_us = entry
                    .active_us
                    .saturating_sub(seconds_to_us(estimated_work_seconds));
                entry.completed += 1;
                entry.completed_us += seconds_to_us(actual_seconds);
            }
        }
    }

    fn charge_tenant_active(&self, tenant_id: Option<&str>, seconds: f64) {
        if let Some(tenant) = tenant_id {
            if let Ok(mut tenants) = self.tenants.lock() {
                tenants.entry(tenant.to_string()).or_default().active_us += seconds_to_us(seconds);
            }
        }
    }

    fn charge_tenant_queued(&self, tenant_id: Option<&str>, seconds: f64) {
        if let Some(tenant) = tenant_id {
            if let Ok(mut tenants) = self.tenants.lock() {
                tenants.entry(tenant.to_string()).or_default().queued_us += seconds_to_us(seconds);
            }
        }
    }

    fn record_admission(
        &self,
        class: WorkloadClass,
        decision: &'static str,
        estimated_work_seconds: f64,
        queue_wait_us: u64,
        tenant_id: Option<&str>,
        reason: &'static str,
    ) {
        if let Ok(mut ring) = self.admissions.lock() {
            ring.push(AdmissionRecord {
                at_us: now_micros(),
                class,
                decision,
                estimated_work_seconds,
                queue_wait_us,
                tenant_id: tenant_id.map(str::to_string),
                reason,
            });
        }
    }

    // -----------------------------------------------------------------------
    // Bottleneck classification
    // -----------------------------------------------------------------------

    /// Decides what is actually limiting the node.
    ///
    /// Order matters. The kinds that make capacity counterproductive are tested
    /// first, because a node that is both busy and conflicted is conflicted:
    /// treating it as merely busy is what turns a contention problem into a
    /// collapse.
    pub fn classify_bottleneck(&self) -> BottleneckKind {
        if self.contention.occ_abort_rate() > OCC_ABORT_THRESHOLD {
            return BottleneckKind::OccContention;
        }
        if self.contention.hot_key_share() > HOT_KEY_THRESHOLD {
            return BottleneckKind::HotPartition;
        }
        if self.contention.writes_waiting() > 0
            && self.contention.group_commit_hit_rate() < GROUP_COMMIT_THRESHOLD
        {
            return BottleneckKind::FsyncBound;
        }
        if self.memory.headroom_fraction() < MEMORY_HEADROOM_THRESHOLD {
            return BottleneckKind::Memory;
        }

        let cpu_headroom = self.capacity.headroom_fraction();
        if cpu_headroom < CPU_HEADROOM_THRESHOLD {
            return BottleneckKind::Cpu;
        }
        // Cores free while queries are running and the node is reading pages.
        // All three are required. Idle cores alone say nothing: a node holding
        // queries back at its admission ceiling looks exactly the same and
        // wants the opposite response, so the discriminator is that the work
        // is in flight rather than queued, and that storage is actually being
        // read rather than merely suspected
        if cpu_headroom > IO_IDLE_CPU_THRESHOLD
            && self.in_flight_work_seconds() > 0.0
            && self.contention.page_reads() > 0
        {
            return BottleneckKind::Io;
        }
        BottleneckKind::None
    }

    /// Work belonging to queries that are running.
    ///
    /// Queued work is excluded on purpose. A query still in the admission
    /// queue is not waiting on storage, it is waiting on this node's own
    /// ceiling, and counting it would classify a throttled node as one whose
    /// device is slow.
    fn in_flight_work_seconds(&self) -> f64 {
        WorkloadClass::ALL
            .into_iter()
            .map(|c| self.counters(c).active_work_seconds())
            .sum()
    }

    /// The classification currently in force, which holds for a period after
    /// it is made so one quiet window cannot re-enable a rung against a
    /// condition that has not cleared.
    fn effective_bottleneck(&self, observed: BottleneckKind, now: Instant) -> BottleneckKind {
        let held = self.bottleneck();
        let mut at = match self.bottleneck_at.lock() {
            Ok(guard) => guard,
            Err(_) => return observed,
        };
        // A fresh non-trivial reading always wins and restarts the hold
        if observed != BottleneckKind::None {
            *at = Some(now);
            return observed;
        }
        match (*at, held) {
            (Some(when), held) if held != BottleneckKind::None => {
                if now.duration_since(when) < CLASSIFICATION_HOLD {
                    held
                } else {
                    *at = None;
                    BottleneckKind::None
                }
            }
            _ => BottleneckKind::None,
        }
    }

    // -----------------------------------------------------------------------
    // Actuator ladder
    // -----------------------------------------------------------------------

    /// The cheapest rung that is both legal against this bottleneck and
    /// reachable in this deployment.
    ///
    /// A rung is skipped when the classifier says it would not help or would
    /// make the situation worse, which is what stops the loop from
    /// provisioning hardware against a conflict problem. A rung this
    /// deployment cannot reach is skipped for a different reason: it is not a
    /// response, it is a request nobody answers.
    pub fn lowest_rung(&self, bottleneck: BottleneckKind, reach: MeshReach) -> ActuatorLevel {
        ActuatorLevel::LADDER
            .into_iter()
            .find(|level| level.legal_for(bottleneck) && reach.allows(*level))
            // Shedding is legal under every classification, so this is
            // unreachable in practice and is the honest answer if a future
            // classification forbids everything
            .unwrap_or(ActuatorLevel::Shed)
    }

    /// Decides what the node is doing about the worst breaching class.
    ///
    /// The ladder is climbed, not indexed. A rung is applied and then given
    /// time to work; if the class is still breaching when that time is up, the
    /// next legal rung replaces it. Without that, the controller would apply
    /// its cheapest lever, watch it fail to help, and apply it again forever:
    /// the ladder would have exactly one rung and the node would never shed.
    ///
    /// How fast it climbs follows how badly the objective is being missed. A
    /// node a little over spends a few windows on each lever, which is what
    /// gives the cheap ones a fair chance. A node many times over reaches
    /// shedding in one step, because walking politely up a ladder while the
    /// queue grows is how a breach becomes a collapse.
    ///
    /// Coming down is one rung at a time and never a jump to nothing. Relief
    /// released all at once puts the node straight back into the breach it
    /// just left, which is the oscillation this design exists to avoid.
    pub fn choose_actuator(
        &self,
        worst: &ClassPressure,
        bottleneck: BottleneckKind,
        now: Instant,
    ) -> ActuatorDecision {
        let reach = self.mesh_reach();
        let level = self.advance_rung(worst, bottleneck, reach, now);
        if level == ActuatorLevel::Steady {
            return ActuatorDecision::steady(worst.class, worst.pressure_seconds);
        }
        ActuatorDecision {
            class: worst.class,
            level,
            bottleneck,
            // Relief already applied stays applied while the ladder climbs
            // past it. Taking the cheap lever back to reach for a dearer one
            // would mean the dearer one has to cover both
            dop_scale_pct: if level >= ActuatorLevel::ReduceDop
                && ActuatorLevel::ReduceDop.legal_for(bottleneck)
            {
                REDUCED_DOP_PCT as u16
            } else {
                100
            },
            delay: if level >= ActuatorLevel::DelayAdmission
                && ActuatorLevel::DelayAdmission.legal_for(bottleneck)
            {
                ADMISSION_DELAY
            } else {
                Duration::ZERO
            },
            pressure_seconds: worst.pressure_seconds,
            slo_seconds: worst.slo_seconds,
            reason: reason_for(level, bottleneck),
        }
    }

    /// Moves the class's rung and returns where it landed.
    fn advance_rung(
        &self,
        worst: &ClassPressure,
        bottleneck: BottleneckKind,
        reach: MeshReach,
        now: Instant,
    ) -> ActuatorLevel {
        let floor = self.lowest_rung(bottleneck, reach);
        let Ok(mut knees) = self.knees.lock() else {
            // Without the state there is no ladder to climb, and the cheapest
            // legal rung is the correct standing answer
            return if worst.breaching() {
                floor
            } else {
                ActuatorLevel::Steady
            };
        };
        let knee = &mut knees[worst.class.index()];
        let settled = knee
            .rung_since
            .map(|since| now.saturating_duration_since(since) >= LADDER_SETTLE)
            .unwrap_or(true);

        if !worst.breaching() {
            if knee.rung != ActuatorLevel::Steady && settled {
                knee.rung = rung_below(knee.rung, bottleneck, reach);
                knee.rung_since = Some(now);
            }
            return knee.rung;
        }

        // A rung the classifier has since forbidden, or one below the floor,
        // is replaced immediately: continuing to hold it would be acting on a
        // classification that no longer stands
        if knee.rung < floor || !knee.rung.legal_for(bottleneck) || !reach.allows(knee.rung) {
            knee.rung = floor;
            knee.rung_since = Some(now);
            return knee.rung;
        }
        if settled {
            let steps = escalation_steps(worst.slo_utilization());
            let next = rung_above(knee.rung, steps, bottleneck, reach);
            if next != knee.rung {
                knee.rung = next;
                knee.rung_since = Some(now);
            }
        }
        knee.rung
    }

    /// Applies a decision to the levers the node actually owns. The mesh rungs
    /// are published rather than applied, because this phase owns the signal
    /// and not the provisioner.
    /// Puts a decision into force.
    ///
    /// Separate from choosing it, because choosing moves this controller's own
    /// rung state and acting moves budgets the whole process shares. The tick
    /// does both in order. A caller that wants to know what the controller
    /// would do without doing it calls `choose_actuator` alone, and one that
    /// is driving the controller calls both.
    pub fn apply(&self, decision: &ActuatorDecision) {
        let level = decision.level;

        // The cheap levers stay in force while the ladder climbs past them, so
        // each is set from the level rather than switched on by it
        self.capacity.set_scale_pct(decision.dop_scale_pct as u32);
        self.query_memory_pct.store(
            if level >= ActuatorLevel::TrimMemory
                && ActuatorLevel::TrimMemory.legal_for(decision.bottleneck)
            {
                crate::pressure::QUERY_MEMORY_MIN_PCT
            } else {
                crate::pressure::QUERY_MEMORY_FULL_PCT
            },
            Ordering::Relaxed,
        );
        self.spill_threshold_pct.store(
            if level >= ActuatorLevel::ForceSpill
                && ActuatorLevel::ForceSpill.legal_for(decision.bottleneck)
            {
                crate::pressure::SPILL_THRESHOLD_FORCED_PCT
            } else {
                crate::pressure::SPILL_THRESHOLD_FULL_PCT
            },
            Ordering::Relaxed,
        );
        // Growth is not cumulative with the rungs below it: it is the answer
        // to a node whose cores are idle waiting on storage, and it is illegal
        // wherever that is not the case
        self.capacity.set_growth_pct(
            if level >= ActuatorLevel::GrowWorkers
                && ActuatorLevel::GrowWorkers.legal_for(decision.bottleneck)
            {
                crate::pressure::PARALLEL_GROWTH_MAX_PCT
            } else {
                crate::pressure::PARALLEL_SCALE_FULL_PCT
            },
        );

        // Delay and shed are enforced at admission. The rungs above them are
        // not this node's to perform, so they are offered to whatever
        // registered for them, once as the ladder enters the rung rather than
        // on every tick that leaves it there: asking a provisioner for
        // hardware ten times a second because the rung is still in force is
        // how a control loop buys a fleet
        let previous = ActuatorLevel::from_code(
            self.actuator.swap(level.code() as u32, Ordering::Relaxed) as u8,
        );
        if level != previous && crate::extension::ExtensionRegistry::global().handles(level) {
            let answer = crate::extension::ExtensionRegistry::global().actuate(decision);
            if let Some(detail) = answer.detail() {
                let mut slot = match self.last_extension.lock() {
                    Ok(slot) => slot,
                    Err(poisoned) => poisoned.into_inner(),
                };
                *slot = Some((level, detail.to_string()));
            }
        }
    }

    /// What an extension said the last time the ladder reached a rung this
    /// node does not perform itself, and which rung that was.
    ///
    /// None when the ladder has never climbed that far, or when nothing was
    /// registered for the rung it reached. A view showing this is showing an
    /// operator why a rung the node climbed to changed nothing.
    pub fn last_extension_answer(&self) -> Option<(ActuatorLevel, String)> {
        let slot = match self.last_extension.lock() {
            Ok(slot) => slot,
            Err(poisoned) => poisoned.into_inner(),
        };
        slot.clone()
    }

    /// Share of its configured allowance a newly admitted query may hold.
    ///
    /// Read when a query's memory budget is built, so a query already running
    /// keeps what it was promised: tightening under a query that has already
    /// materialized half a hash table would fail work that was within its
    /// limit when it started.
    pub fn query_memory_pct(&self) -> u32 {
        self.query_memory_pct.load(Ordering::Relaxed)
    }

    /// The allowance a query starting now should be given, from what the
    /// session configured.
    pub fn query_memory_allowance(&self, configured_bytes: u64) -> u64 {
        if configured_bytes == 0 {
            return 0;
        }
        let pct = self.query_memory_pct() as u64;
        (configured_bytes.saturating_mul(pct) / crate::pressure::QUERY_MEMORY_FULL_PCT as u64)
            .max(1)
    }

    /// Records what the node configured one query to be allowed to hold.
    ///
    /// Set once at startup from `query.max_memory_bytes`. Zero, the default,
    /// means unlimited: nothing spills and nothing fails on a budget.
    pub fn set_configured_query_memory(&self, bytes: u64) {
        self.configured_query_memory.store(bytes, Ordering::Relaxed);
    }

    /// What a materializing operator in a query starting now may hold before
    /// it writes to disk, or zero when nothing bounds it.
    ///
    /// Both rungs are applied, because both are in force by the time the
    /// query runs. Read by the planner, which needs to know whether the plan
    /// it is costing would spill, and that answer is node state: the same
    /// query planned on a node under memory pressure gets a different plan
    /// from one planned on an idle node, which is the point of measuring
    /// pressure at all.
    pub fn working_memory_bytes(&self) -> u64 {
        let configured = self.configured_query_memory.load(Ordering::Relaxed);
        if configured == 0 {
            return 0;
        }
        self.spill_threshold(self.query_memory_allowance(configured))
    }

    /// Share of a query's allowance its materializing operators may hold
    /// before they write to disk.
    pub fn spill_threshold_pct(&self) -> u32 {
        self.spill_threshold_pct.load(Ordering::Relaxed)
    }

    /// Where a query's operators should start spilling, from what that query
    /// was allowed to hold.
    ///
    /// Read as the operator tree is built rather than stored on the query, so
    /// the rung reaches the query that is starting now. Below the allowance
    /// only while the ForceSpill rung is in force, and equal to it otherwise,
    /// which is the point at which an operator would have failed before
    /// spilling existed.
    pub fn spill_threshold(&self, allowance_bytes: u64) -> u64 {
        if allowance_bytes == 0 {
            return 0;
        }
        let pct = self.spill_threshold_pct() as u64;
        (allowance_bytes.saturating_mul(pct) / crate::pressure::SPILL_THRESHOLD_FULL_PCT as u64)
            .max(1)
    }

    // -----------------------------------------------------------------------
    // Tick
    // -----------------------------------------------------------------------

    /// Advances the controller by one window.
    ///
    /// Closes the measurement windows, moves each class's ceiling toward where
    /// throughput actually stopped improving, classifies what is limiting the
    /// node, picks and applies a rung, and publishes the result.
    pub fn tick(&self, now: Instant) -> ActuatorDecision {
        let window = {
            let mut last = self.last_tick.lock().expect("tick clock");
            let elapsed = last.map(|t| now.duration_since(t));
            *last = Some(now);
            elapsed.unwrap_or(Duration::from_millis(100))
        };
        let window = if window.is_zero() {
            Duration::from_millis(100)
        } else {
            window
        };

        for class in WorkloadClass::ALL {
            self.counters(class).roll_window(window);
        }
        self.update_knees(now);

        let observed = self.classify_bottleneck();
        let bottleneck = self.effective_bottleneck(observed, now);
        self.bottleneck
            .store(bottleneck.code() as u32, Ordering::Relaxed);

        let snapshot = self.snapshot_with(bottleneck, self.actuator());
        let worst = worst_breaching(&snapshot).unwrap_or_else(|| snapshot.classes[0].clone());
        let decision = self.choose_actuator(&worst, bottleneck, now);
        self.apply(&decision);

        self.push_history(&snapshot, now, window, bottleneck, decision.level);
        self.contention.roll_window();
        self.sequence.fetch_add(1, Ordering::Relaxed);
        self.ticks.fetch_add(1, Ordering::Relaxed);
        decision
    }

    /// Moves each class's ceiling toward the measured knee.
    ///
    /// Growth continues while throughput improves. A breach backs off
    /// multiplicatively. When neither holds and the class has been calm for
    /// long enough, one exploration step reaches past the believed knee to find
    /// out whether the machine, the workload, or the storage has changed since
    /// the knee was found.
    fn update_knees(&self, now: Instant) {
        let mut knees = match self.knees.lock() {
            Ok(k) => k,
            Err(_) => return,
        };
        for class in WorkloadClass::ALL {
            let counters = self.counters(class);
            let knee = &mut knees[class.index()];
            let throughput = counters.service_capacity_qps();
            let concurrency = counters.in_flight();
            let pressure = counters.pressure_seconds();
            let slo = class.slo_seconds();
            let utilization = if slo.is_finite() && slo > 0.0 {
                pressure / slo
            } else {
                0.0
            };

            // Score an outstanding exploration before deciding anything else
            if let Some(before) = knee.exploring_from.take() {
                let kept = throughput >= knee.best_throughput * IMPROVEMENT_MARGIN;
                if let Ok(mut ledger) = self.ledger.lock() {
                    ledger.push(LedgerEntry {
                        at_us: now_micros(),
                        class,
                        ceiling_before: before,
                        ceiling_after: knee.ceiling,
                        throughput_before: knee.best_throughput,
                        throughput_after: throughput,
                        pressure_seconds: pressure,
                        kept,
                    });
                }
                if kept {
                    knee.best_throughput = throughput;
                    knee.best_concurrency = concurrency;
                } else {
                    knee.ceiling = before.max(MIN_CEILING);
                }
            }

            // Decide whether this window improved on the best before
            // recording it. Recording first would make the comparison below
            // compare the new best against itself, and the ceiling would
            // never grow
            let improving = throughput > knee.best_throughput * IMPROVEMENT_MARGIN;
            if improving {
                knee.best_throughput = throughput;
                knee.best_concurrency = concurrency;
            }

            // Calibration error over the window, which is what tells the cost
            // model whether it is lying
            if knee.window_estimated_seconds > 0.0 {
                let ratio = knee.window_actual_seconds / knee.window_estimated_seconds;
                knee.calibration_error = knee.calibration_error * 0.8 + ratio * 0.2;
                counters.set_calibration_error(knee.calibration_error);
            }
            knee.window_estimated_seconds = 0.0;
            knee.window_actual_seconds = 0.0;

            if utilization > 1.0 {
                // Breaching. Pull back, and stop counting as calm
                let reduced = (knee.ceiling as f64 * BACKOFF_FACTOR) as u32;
                knee.ceiling = reduced.max(MIN_CEILING);
                knee.calm_since = None;
            } else if throughput > 0.0 && (improving || concurrency >= knee.ceiling) {
                // Still improving, or pressed against the ceiling without
                // breaching, so there may be more to have. An idle class is
                // excluded: no throughput is not evidence that a higher
                // ceiling would produce any
                knee.ceiling = grow_ceiling(knee.ceiling, class);
                knee.calm_since = None;
            } else if utilization < EXPLORE_UTILIZATION {
                let calm_since = *knee.calm_since.get_or_insert(now);
                if now.duration_since(calm_since) >= EXPLORE_QUIET_PERIOD {
                    let before = knee.ceiling;
                    let step = before.saturating_mul(EXPLORE_STEP_PCT).div_ceil(100).max(1);
                    knee.ceiling = before.saturating_add(step).min(max_ceiling(class));
                    knee.exploring_from = Some(before);
                    knee.calm_since = None;
                }
            } else {
                knee.calm_since = None;
            }

            counters.set_ceiling(knee.ceiling);
        }
    }

    /// Forces an exploration step for a class, used to prove the ledger records
    /// one without waiting out the quiet period.
    pub fn force_exploration(&self, class: WorkloadClass) {
        if let Ok(mut knees) = self.knees.lock() {
            let knee = &mut knees[class.index()];
            knee.calm_since = Some(Instant::now() - EXPLORE_QUIET_PERIOD - Duration::from_secs(1));
        }
    }

    fn push_history(
        &self,
        snapshot: &NodePressure,
        now: Instant,
        window: Duration,
        bottleneck: BottleneckKind,
        actuator: ActuatorLevel,
    ) {
        let mut sample = HistorySample {
            at_us: snapshot.updated_at_us,
            elapsed_us: now
                .saturating_duration_since(self.started)
                .as_micros()
                .min(u64::MAX as u128) as u64,
            pressure_us: [0; WorkloadClass::COUNT],
            queued_us: [0; WorkloadClass::COUNT],
            active_us: [0; WorkloadClass::COUNT],
            in_flight: [0; WorkloadClass::COUNT],
            capacity_milli_qps: [0; WorkloadClass::COUNT],
            arrivals: [0; WorkloadClass::COUNT],
            window_us: window.as_micros().min(u32::MAX as u128) as u32,
            bottleneck,
            actuator,
        };
        for row in &snapshot.classes {
            let i = row.class.index();
            sample.pressure_us[i] = clamp_u32(row.pressure_seconds * 1e6);
            sample.queued_us[i] = clamp_u32(row.queued_work_seconds * 1e6);
            sample.active_us[i] = clamp_u32(row.active_work_seconds * 1e6);
            sample.in_flight[i] = row.in_flight.min(u16::MAX as u32) as u16;
            sample.capacity_milli_qps[i] = clamp_u32(row.service_capacity_qps * 1000.0);

            let total = self.counters(row.class).arrivals_total();
            let previous = self.last_arrivals[i].swap(total, Ordering::Relaxed);
            sample.arrivals[i] = total.saturating_sub(previous).min(u32::MAX as u64) as u32;
        }
        if let Ok(mut ring) = self.history.lock() {
            ring.push(sample);
        }
    }

    // -----------------------------------------------------------------------
    // Snapshots
    // -----------------------------------------------------------------------

    /// The node's current state, as the views and the gossip frame see it.
    pub fn snapshot(&self) -> NodePressure {
        self.snapshot_with(self.bottleneck(), self.actuator())
    }

    fn snapshot_with(&self, bottleneck: BottleneckKind, actuator: ActuatorLevel) -> NodePressure {
        let classes: Vec<ClassPressure> = WorkloadClass::ALL
            .into_iter()
            .map(|class| {
                let mut row = self.counters(class).snapshot(class);
                row.bottleneck = bottleneck;
                row.actuator_level = actuator;
                row
            })
            .collect();
        let mut node = NodePressure {
            node_id: self.node_id,
            classes,
            overall_utilization: 0.0,
            bottleneck,
            actuator_level: actuator,
            parallel_permits_total: self.capacity.total(),
            parallel_permits_available: self.capacity.available(),
            memory_reserved_bytes: self.memory.reserved(),
            memory_ceiling_bytes: self.memory.ceiling(),
            updated_at_us: now_micros(),
        };
        node.recompute_overall();
        node
    }

    /// History newest first, at most `limit` samples.
    pub fn history(&self, limit: usize) -> Vec<HistorySample> {
        self.history
            .lock()
            .map(|r| r.recent(limit))
            .unwrap_or_default()
    }

    /// Admission decisions newest first.
    pub fn admissions(&self, limit: usize) -> Vec<AdmissionRecord> {
        self.admissions
            .lock()
            .map(|r| r.recent(limit))
            .unwrap_or_default()
    }

    /// Exploration outcomes newest first.
    pub fn ledger(&self, limit: usize) -> Vec<LedgerEntry> {
        self.ledger
            .lock()
            .map(|r| r.recent(limit))
            .unwrap_or_default()
    }

    /// Pressure attributed per tenant, so the cost of relieving it lands on
    /// whoever created it.
    pub fn tenant_pressure(&self) -> Vec<TenantPressure> {
        let tenants = match self.tenants.lock() {
            Ok(t) => t,
            Err(_) => return Vec::new(),
        };
        let mut out: Vec<TenantPressure> = tenants
            .iter()
            .map(|(tenant, work)| TenantPressure {
                tenant_id: tenant.clone(),
                queued_work_seconds: work.queued_us as f64 / 1e6,
                active_work_seconds: work.active_us as f64 / 1e6,
                completed: work.completed,
                completed_work_seconds: work.completed_us as f64 / 1e6,
            })
            .collect();
        out.sort_by(|a, b| {
            b.total_work_seconds()
                .partial_cmp(&a.total_work_seconds())
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        out
    }

    // -----------------------------------------------------------------------
    // Projection
    // -----------------------------------------------------------------------

    /// Which mesh rungs this deployment can reach, as the ladder sees them.
    pub fn mesh_reach(&self) -> MeshReach {
        MeshReach::from_bits(self.mesh_reach.load(Ordering::Relaxed))
    }

    /// How long capacity takes to become useful here.
    ///
    /// Zero until the loop that owns the driver publishes one, which is the
    /// honest reading for a deployment that provisions nothing: there is no
    /// horizon to project over, so the projection is the present.
    pub fn provision_horizon(&self) -> Duration {
        Duration::from_millis(self.provision_horizon_ms.load(Ordering::Relaxed))
    }

    pub fn set_provision_horizon(&self, horizon: Duration) {
        self.provision_horizon_ms.store(
            horizon.as_millis().min(u64::MAX as u128) as u64,
            Ordering::Relaxed,
        );
    }

    /// Largest warm pool the operator will pay for. A ceiling, never a target.
    pub fn warm_pool_cap(&self) -> u32 {
        self.warm_pool_cap.load(Ordering::Relaxed)
    }

    pub fn set_warm_pool_cap(&self, cap: u32) {
        self.warm_pool_cap.store(cap, Ordering::Relaxed);
    }

    /// The projection every reader shares, against the published horizon and
    /// cap.
    pub fn projection(&self, class: WorkloadClass) -> PressureProjection {
        self.project(class, self.provision_horizon(), self.warm_pool_cap())
    }

    /// Records what the installed provisioner can actually do.
    ///
    /// Called by the loop that owns the driver rather than read from the
    /// registry here, so the ladder has one source of truth per tick and a
    /// controller under test is not steered by process-wide state.
    pub fn set_mesh_reach(&self, reach: MeshReach) {
        self.mesh_reach.store(reach.bits(), Ordering::Relaxed);
    }

    /// Fits how fast work has been arriving for a class.
    ///
    /// Walks the history newest first and stops at the window edge, so the
    /// cost is the window rather than the ring.
    pub fn arrival_trend(&self, class: WorkloadClass, window: Duration) -> ArrivalTrend {
        let samples = match self.history.lock() {
            Ok(ring) => ring.recent(HISTORY_CAPACITY),
            Err(_) => return ArrivalTrend::default(),
        };
        let Some(newest) = samples.first() else {
            return ArrivalTrend::default();
        };
        let horizon_us = window.as_micros().min(u64::MAX as u128) as u64;
        let mut points = Vec::with_capacity(samples.len().min(1024));
        for sample in &samples {
            let age_us = newest.elapsed_us.saturating_sub(sample.elapsed_us);
            if age_us > horizon_us {
                break;
            }
            points.push(ArrivalSample {
                age_seconds: age_us as f64 / 1e6,
                rate_qps: sample.arrival_rate(class),
            });
        }
        fit_arrival_trend(&points)
    }

    /// Where a class will be one provision latency from now.
    ///
    /// The horizon is passed in rather than read from the provisioner, because
    /// the caller is the loop that already holds the driver and reading it
    /// twice invites the two answers to differ inside one decision.
    pub fn project(
        &self,
        class: WorkloadClass,
        horizon: Duration,
        warm_pool_cap: u32,
    ) -> PressureProjection {
        let counters = self.counters(class);
        let trend = self.arrival_trend(class, TREND_WINDOW);
        let inputs = ProjectionInputs {
            class,
            outstanding_work_seconds: counters.queued_work_seconds()
                + counters.active_work_seconds(),
            service_capacity_qps: counters.service_capacity_qps(),
            mean_service_seconds: counters.mean_service_seconds(),
            provision_latency: horizon,
            warm_pool_cap,
        };
        project_pressure(&inputs, &trend)
    }

    /// How long the node expects to stay as quiet as it is, which is what a
    /// reclaim decision weighs against the cost of getting the node back.
    pub fn predicted_idle_window(&self, class: WorkloadClass) -> Duration {
        let trend = self.arrival_trend(class, TREND_WINDOW);
        crate::projection::predicted_idle_window(
            &trend,
            self.counters(class).service_capacity_qps(),
        )
    }

    // -----------------------------------------------------------------------
    // Query shapes
    // -----------------------------------------------------------------------

    /// Records that a query of a given shape ran and what it was priced at.
    ///
    /// Keyed by the plan's structural hash, so two runs of the same statement
    /// with different parameters are one shape. Bounded by eviction of the
    /// cheapest shape rather than by refusing new ones, because a node whose
    /// traffic has changed would otherwise keep describing the traffic it used
    /// to serve.
    pub fn record_shape(&self, fingerprint: u64, estimated_work_seconds: f64) {
        if fingerprint == 0 {
            return;
        }
        let Ok(mut shapes) = self.shapes.lock() else {
            return;
        };
        let entry = shapes.entry(fingerprint).or_default();
        entry.executions += 1;
        entry.work_us = entry
            .work_us
            .saturating_add(seconds_to_us(estimated_work_seconds));
        if shapes.len() > SHAPE_CAPACITY {
            evict_cheapest_shape(&mut shapes);
        }
    }

    /// The query shapes worth handing to a survivor, heaviest first.
    pub fn hot_queries(&self, limit: usize) -> Vec<HotQuery> {
        let Ok(shapes) = self.shapes.lock() else {
            return Vec::new();
        };
        let mut out: Vec<HotQuery> = shapes
            .iter()
            .map(|(fingerprint, work)| HotQuery {
                fingerprint: *fingerprint,
                executions: work.executions.min(u32::MAX as u64) as u32,
                mean_work_us: (work.work_us / work.executions.max(1)).min(u32::MAX as u64) as u32,
            })
            .collect();
        out.sort_unstable_by_key(|q| {
            std::cmp::Reverse(q.executions as u64 * q.mean_work_us as u64)
        });
        out.truncate(limit);
        out
    }

    /// How many distinct shapes the node is tracking.
    pub fn shape_count(&self) -> usize {
        self.shapes.lock().map(|s| s.len()).unwrap_or(0)
    }

    /// Records that a working-set manifest was written.
    pub fn record_hot_set(&self, pages: u32, generated_us: i64, persisted: bool) {
        self.hot_set_pages.store(pages, Ordering::Relaxed);
        self.hot_set_generated_us
            .store(generated_us, Ordering::Relaxed);
        self.hot_set_persisted.store(persisted, Ordering::Relaxed);
    }

    /// Records the outcome of a checkpoint.
    ///
    /// A failed one is recorded too, and as unclean, because the question this
    /// answers is whether a resuming node would open a checkpoint or replay a
    /// log. Leaving the previous success standing would answer it wrongly.
    pub fn record_checkpoint(&self, at_us: i64, clean: bool) {
        self.checkpoint_at_us.store(at_us, Ordering::Release);
        self.checkpoint_clean.store(clean, Ordering::Release);
        if clean {
            self.wal_bytes_since_checkpoint.store(0, Ordering::Release);
        }
    }

    /// Records write-ahead log written since the last checkpoint.
    ///
    /// Not part of the resume estimate, and deliberately so. It is recorded so
    /// an operator can see how far the two have diverged, and so a refusal can
    /// say what replaying would cost.
    pub fn record_wal_since_checkpoint(&self, bytes: u64) {
        self.wal_bytes_since_checkpoint
            .store(bytes, Ordering::Relaxed);
    }

    /// What the last checkpoint did.
    pub fn checkpoint_state(&self) -> CheckpointState {
        CheckpointState {
            at_us: self.checkpoint_at_us.load(Ordering::Acquire),
            // A checkpoint that never ran is not a clean one, whatever the
            // flag says, because there is nothing to open
            clean: self.checkpoint_clean.load(Ordering::Acquire)
                && self.checkpoint_at_us.load(Ordering::Acquire) > 0,
            wal_bytes_since: self.wal_bytes_since_checkpoint.load(Ordering::Relaxed),
        }
    }

    /// How stale the checkpoint is, which is half of what a resume costs.
    pub fn checkpoint_age(&self) -> Duration {
        let state = self.checkpoint_state();
        if state.at_us <= 0 {
            return Duration::ZERO;
        }
        let now_us = now_micros();
        Duration::from_micros(now_us.saturating_sub(state.at_us).max(0) as u64)
    }

    /// What the last manifest holds, for the view and for the scale-to-zero
    /// precondition.
    pub fn hot_set_status(&self) -> HotSetStatus {
        HotSetStatus {
            pages: self.hot_set_pages.load(Ordering::Relaxed),
            generated_us: self.hot_set_generated_us.load(Ordering::Relaxed),
            persisted: self.hot_set_persisted.load(Ordering::Relaxed),
            shapes: self.shape_count(),
        }
    }
}

/// What the last checkpoint did, as every reader sees it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct CheckpointState {
    /// Wall clock of the last completed checkpoint. Zero when none has run
    pub at_us: i64,
    /// Whether a resuming node would open a checkpoint rather than replay
    pub clean: bool,
    /// Log written since, for the refusal message rather than the estimate
    pub wal_bytes_since: u64,
}

/// The working-set manifest as the node last wrote it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct HotSetStatus {
    pub pages: u32,
    pub generated_us: i64,
    /// False until a manifest has actually reached the disk. A node that has
    /// built one in memory and not written it cannot go to zero, because the
    /// copy that matters is the one a resuming node reads
    pub persisted: bool,
    /// Query shapes the node is tracking
    pub shapes: usize,
}

/// Which mesh rungs of the ladder this deployment can reach.
///
/// Derived from what the installed provisioner reports rather than from how
/// the node was registered, so a mode whose control plane is unreachable
/// behaves exactly like a mode that has none: the controller relieves pressure
/// with the levers it owns and does not publish a request nothing will answer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MeshReach {
    /// Whether a node the mesh already has running can be claimed
    pub can_take_warm_node: bool,
    /// Whether new hardware can be created
    pub can_provision: bool,
}

impl MeshReach {
    /// A single node with nowhere to grow, which is what a deployment with no
    /// provisioner installed is.
    pub const NONE: MeshReach = MeshReach {
        can_take_warm_node: false,
        can_provision: false,
    };

    /// Both mesh rungs reachable.
    pub const FULL: MeshReach = MeshReach {
        can_take_warm_node: true,
        can_provision: true,
    };

    pub fn from_capabilities(capabilities: &ProvisionerCapabilities) -> Self {
        Self {
            // A mesh of one has no other node to claim, whatever the driver
            // is otherwise capable of
            can_take_warm_node: capabilities.max_nodes > 1,
            can_provision: capabilities.can_provision,
        }
    }

    pub const fn allows(self, level: ActuatorLevel) -> bool {
        match level {
            ActuatorLevel::WarmPoolTake => self.can_take_warm_node,
            ActuatorLevel::ProvisionNode => self.can_provision,
            _ => true,
        }
    }

    const fn bits(self) -> u32 {
        (self.can_take_warm_node as u32) | ((self.can_provision as u32) << 1)
    }

    const fn from_bits(bits: u32) -> Self {
        Self {
            can_take_warm_node: bits & 1 != 0,
            can_provision: bits & 2 != 0,
        }
    }
}

/// Drops the shape that accounts for the least work.
///
/// Called only when the map is over its bound, and the bound is high enough
/// that this runs on a node whose statement text is generated per request
/// rather than on one serving a fixed set of queries.
fn evict_cheapest_shape(shapes: &mut std::collections::HashMap<u64, ShapeWork>) {
    let victim = shapes
        .iter()
        .min_by_key(|(_, work)| work.work_us)
        .map(|(fingerprint, _)| *fingerprint);
    if let Some(fingerprint) = victim {
        shapes.remove(&fingerprint);
    }
}

/// One tenant's share of the node's work.
#[derive(Debug, Clone, PartialEq)]
pub struct TenantPressure {
    pub tenant_id: String,
    pub queued_work_seconds: f64,
    pub active_work_seconds: f64,
    pub completed: u64,
    pub completed_work_seconds: f64,
}

impl TenantPressure {
    pub fn total_work_seconds(&self) -> f64 {
        self.queued_work_seconds + self.active_work_seconds
    }
}

/// How many rungs to climb in one step, from how badly the objective is
/// being missed.
///
/// One rung per settling period while the miss is small, doubling the step for
/// every doubling of the overshoot. A node at ten times its objective is not
/// going to be rescued by handing one query fewer workers, and the windows
/// spent finding that out are windows of queue growth.
fn escalation_steps(utilization: f64) -> usize {
    if !utilization.is_finite() || utilization <= 1.0 {
        return 1;
    }
    let steps = utilization.log2().floor() as i64 + 1;
    (steps.max(1) as usize).min(ActuatorLevel::LADDER.len())
}

/// The next legal reachable rung above `from`, at most `steps` of them.
fn rung_above(
    from: ActuatorLevel,
    steps: usize,
    bottleneck: BottleneckKind,
    reach: MeshReach,
) -> ActuatorLevel {
    let mut landed = from;
    let mut taken = 0usize;
    for level in ActuatorLevel::LADDER {
        if level <= from {
            continue;
        }
        if !level.legal_for(bottleneck) || !reach.allows(level) {
            continue;
        }
        landed = level;
        taken += 1;
        if taken >= steps {
            break;
        }
    }
    landed
}

/// The next legal reachable rung below `from`, or Steady when there is none.
fn rung_below(from: ActuatorLevel, bottleneck: BottleneckKind, reach: MeshReach) -> ActuatorLevel {
    let mut landed = ActuatorLevel::Steady;
    for level in ActuatorLevel::LADDER {
        if level >= from {
            break;
        }
        if level.legal_for(bottleneck) && reach.allows(level) {
            landed = level;
        }
    }
    landed
}

/// The class furthest past its objective, if any is past it.
fn worst_breaching(snapshot: &NodePressure) -> Option<ClassPressure> {
    snapshot
        .classes
        .iter()
        .filter(|c| c.breaching())
        .max_by(|a, b| {
            a.slo_utilization()
                .partial_cmp(&b.slo_utilization())
                .unwrap_or(std::cmp::Ordering::Equal)
        })
        .cloned()
}

/// The largest concurrency a class may be allowed.
///
/// Bounded by the class's own queue depth rather than by a separate literal:
/// admitting more at once than the queue could ever hold makes the queue
/// vestigial, and past that point the ceiling has stopped being a control.
fn max_ceiling(class: WorkloadClass) -> u32 {
    class.queue_depth().min(u32::MAX as usize) as u32
}

/// One growth step, saturating rather than wrapping and stopping at the
/// class's bound.
fn grow_ceiling(current: u32, class: WorkloadClass) -> u32 {
    let scaled = (current as f64 * GROWTH_FACTOR).ceil();
    let scaled = if scaled.is_finite() && scaled < u32::MAX as f64 {
        scaled as u32
    } else {
        u32::MAX
    };
    scaled
        .max(current.saturating_add(1))
        .min(max_ceiling(class))
}

fn reason_for(level: ActuatorLevel, bottleneck: BottleneckKind) -> &'static str {
    match (level, bottleneck) {
        (ActuatorLevel::ReduceDop, BottleneckKind::OccContention) => {
            "conflict rises with concurrency, so parallelism comes down"
        }
        (ActuatorLevel::ReduceDop, _) => "narrowing parallelism is the cheapest relief",
        (ActuatorLevel::DelayAdmission, _) => "holding arrivals until the class drains",
        (ActuatorLevel::TrimMemory, BottleneckKind::Memory) => {
            "the node is at its memory ceiling, so new queries get a smaller share of it"
        }
        (ActuatorLevel::TrimMemory, _) => "handing new queries a smaller memory ceiling",
        (ActuatorLevel::ForceSpill, _) => {
            "sorts, joins, and aggregates write to disk sooner so the node gets its memory back"
        }
        (ActuatorLevel::Shed, BottleneckKind::OccContention) => {
            "refusing work, conflict makes added capacity counterproductive"
        }
        (ActuatorLevel::Shed, _) => "refusing work past the queue depth",
        (ActuatorLevel::GrowWorkers, _) => {
            "cores are idle waiting on storage, so more work goes in flight"
        }
        (ActuatorLevel::WarmPoolTake, _) => "claiming a node the mesh already has running",
        (ActuatorLevel::ProvisionNode, _) => "local relief is exhausted, hardware is needed",
        (ActuatorLevel::Steady, _) => "pressure under objective",
    }
}

// ---------------------------------------------------------------------------
// Ring buffer
// ---------------------------------------------------------------------------

/// Fixed-capacity ring. Bounded storage is the point: the history views must
/// cost the same after a month as after a minute.
#[derive(Debug)]
struct Ring<T> {
    items: Vec<T>,
    next: usize,
    capacity: usize,
}

impl<T: Clone> Ring<T> {
    fn new(capacity: usize) -> Self {
        Self {
            items: Vec::with_capacity(capacity.min(1024)),
            next: 0,
            capacity,
        }
    }

    fn push(&mut self, item: T) {
        if self.items.len() < self.capacity {
            self.items.push(item);
            self.next = self.items.len() % self.capacity;
        } else {
            self.items[self.next] = item;
            self.next = (self.next + 1) % self.capacity;
        }
    }

    /// Newest first.
    fn recent(&self, limit: usize) -> Vec<T> {
        let count = limit.min(self.items.len());
        let mut out = Vec::with_capacity(count);
        for i in 0..count {
            let idx = (self.next + self.items.len() - 1 - i) % self.items.len();
            out.push(self.items[idx].clone());
        }
        out
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn seconds_to_us(seconds: f64) -> u64 {
    if !seconds.is_finite() || seconds <= 0.0 {
        return 0;
    }
    (seconds * 1e6) as u64
}

fn clamp_u32(v: f64) -> u32 {
    if !v.is_finite() || v <= 0.0 {
        return 0;
    }
    v.min(u32::MAX as f64) as u32
}

fn now_micros() -> i64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_micros() as i64)
        .unwrap_or(0)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn controller() -> PressureController {
        PressureController::new(1, 1 << 30)
    }

    #[test]
    fn a_tiny_query_bypasses_the_decision_entirely() {
        let c = controller();
        let (class, decision) = c.admit(0.0001, false, None);
        assert_eq!(class, WorkloadClass::Interactive);
        assert_eq!(decision, AdmitDecision::Bypass);
        assert_eq!(c.counters(class).bypassed_total(), 1);
        // It still counts as in flight, or the node would not know it is busy
        assert_eq!(c.counters(class).in_flight(), 1);
    }

    /// A shedding node must still run the queries too small for the decision
    /// to pay for itself.
    #[test]
    fn the_bypass_survives_a_shedding_class() {
        let c = controller();
        let class = WorkloadClass::Interactive;
        // Fill the ceiling and the whole queue
        let ceiling = c.ceiling(class);
        for _ in 0..ceiling {
            assert_eq!(c.admit(0.01, false, None).1, AdmitDecision::Admit);
        }
        for _ in 0..class.queue_depth() {
            assert!(matches!(
                c.admit(0.01, false, None).1,
                AdmitDecision::Delay(_)
            ));
        }
        assert!(matches!(
            c.admit(0.01, false, None).1,
            AdmitDecision::Shed { .. }
        ));
        // And yet
        assert_eq!(c.admit(0.0001, false, None).1, AdmitDecision::Bypass);
    }

    #[test]
    fn admission_fills_the_ceiling_then_queues_then_sheds() {
        let c = controller();
        let class = WorkloadClass::Interactive;
        let ceiling = c.ceiling(class);
        for _ in 0..ceiling {
            assert_eq!(c.admit(0.01, false, None).1, AdmitDecision::Admit);
        }
        assert!(matches!(
            c.admit(0.01, false, None).1,
            AdmitDecision::Delay(_)
        ));
        assert_eq!(c.counters(class).queued_count(), 1);
    }

    #[test]
    fn a_queued_query_is_admitted_once_the_class_drains() {
        let c = controller();
        let class = WorkloadClass::Interactive;
        for _ in 0..c.ceiling(class) {
            c.admit(0.01, false, None);
        }
        let (_, queued) = c.admit(0.01, false, None);
        assert!(matches!(queued, AdmitDecision::Delay(_)));

        // Still full
        assert!(matches!(
            c.retry_queued(class, 0.01, Duration::from_millis(5), None),
            AdmitDecision::Delay(_)
        ));

        c.complete(class, 0.01, 0.01, None);
        assert_eq!(
            c.retry_queued(class, 0.01, Duration::from_millis(10), None),
            AdmitDecision::Admit
        );
        assert_eq!(c.counters(class).queued_count(), 0);
    }

    #[test]
    fn work_time_moves_the_signal_by_what_the_query_costs() {
        let c = controller();
        let class = WorkloadClass::Bulk;
        // Bulk, so it is queued behind an already-full ceiling and lands in
        // queued work rather than active
        let before = c.counters(class).queued_work_seconds();
        for _ in 0..c.ceiling(class) {
            c.admit(0.5, false, None);
        }
        c.admit(0.5, false, None);
        let after = c.counters(class).queued_work_seconds();
        assert!(
            (after - before - 0.5).abs() < 1e-6,
            "queue moved by {} not 0.5",
            after - before
        );
    }

    #[test]
    fn classes_are_isolated_from_each_other() {
        let c = controller();
        // Saturate bulk completely
        for _ in 0..c.ceiling(WorkloadClass::Bulk) + WorkloadClass::Bulk.queue_depth() as u32 {
            c.admit(5.0, false, None);
        }
        assert!(matches!(
            c.admit(5.0, false, None).1,
            AdmitDecision::Shed { .. }
        ));
        // Interactive is untouched by it
        assert_eq!(c.admit(0.01, false, None).1, AdmitDecision::Admit);
    }

    // -- bottleneck classification ------------------------------------------

    #[test]
    fn conflict_outranks_being_busy() {
        let c = controller();
        // Look busy
        let taken = c.capacity().try_take(c.capacity().total() as usize);
        // And conflicted
        for _ in 0..90 {
            c.contention().record_commit();
        }
        for _ in 0..10 {
            c.contention().record_conflict_abort();
        }
        assert_eq!(c.classify_bottleneck(), BottleneckKind::OccContention);
        c.capacity().give_back(taken);
    }

    #[test]
    fn a_conflicted_node_is_never_told_to_add_capacity() {
        let c = controller();
        for _ in 0..80 {
            c.contention().record_commit();
        }
        for _ in 0..20 {
            c.contention().record_conflict_abort();
        }
        let breaching = ClassPressure {
            pressure_seconds: 10.0,
            slo_seconds: 0.1,
            ..blank(WorkloadClass::Interactive)
        };
        let decision = c.choose_actuator(&breaching, BottleneckKind::OccContention, Instant::now());
        assert!(
            matches!(
                decision.level,
                ActuatorLevel::ReduceDop | ActuatorLevel::DelayAdmission | ActuatorLevel::Shed
            ),
            "picked {:?} against conflict",
            decision.level
        );
    }

    #[test]
    fn a_hot_partition_masks_the_mesh_rungs() {
        let c = controller();
        c.contention().record_key_touches(40, 100);
        assert_eq!(c.classify_bottleneck(), BottleneckKind::HotPartition);
        let breaching = ClassPressure {
            pressure_seconds: 10.0,
            slo_seconds: 0.1,
            ..blank(WorkloadClass::Interactive)
        };
        let decision = c.choose_actuator(&breaching, BottleneckKind::HotPartition, Instant::now());
        assert!(decision.level < ActuatorLevel::WarmPoolTake);
    }

    #[test]
    fn writers_waiting_on_the_device_are_not_a_compute_shortage() {
        let c = controller();
        c.contention().set_writes_waiting(4);
        for _ in 0..10 {
            c.contention().record_group_commit(false);
        }
        assert_eq!(c.classify_bottleneck(), BottleneckKind::FsyncBound);
    }

    #[test]
    fn a_full_memory_gauge_classifies_as_memory() {
        // The gauge measures the share of the machine query execution may
        // take, not the machine, so the reservation is sized from the ceiling
        // rather than from the number handed to the constructor
        let c = PressureController::new(1, 1000);
        let ceiling = c.memory().ceiling();
        assert!(ceiling > 0, "the query memory share rounded to nothing");
        assert!(c.memory().try_reserve(ceiling - ceiling / 20));
        assert_eq!(c.classify_bottleneck(), BottleneckKind::Memory);
        let breaching = ClassPressure {
            pressure_seconds: 10.0,
            slo_seconds: 0.1,
            ..blank(WorkloadClass::Bulk)
        };
        // Growing the worker count grows the working set, so it is refused
        let decision = c.choose_actuator(&breaching, BottleneckKind::Memory, Instant::now());
        assert_ne!(decision.level, ActuatorLevel::GrowWorkers);
    }

    #[test]
    fn a_classification_holds_rather_than_flapping() {
        let c = controller();
        let now = Instant::now();
        c.contention().record_key_touches(40, 100);
        let held = c.effective_bottleneck(BottleneckKind::HotPartition, now);
        assert_eq!(held, BottleneckKind::HotPartition);
        c.bottleneck.store(
            BottleneckKind::HotPartition.code() as u32,
            Ordering::Relaxed,
        );

        // One quiet window does not clear it
        let soon = now + Duration::from_secs(5);
        assert_eq!(
            c.effective_bottleneck(BottleneckKind::None, soon),
            BottleneckKind::HotPartition
        );
        // Past the hold it does
        let later = now + CLASSIFICATION_HOLD + Duration::from_secs(1);
        assert_eq!(
            c.effective_bottleneck(BottleneckKind::None, later),
            BottleneckKind::None
        );
    }

    // -- ladder -------------------------------------------------------------

    #[test]
    fn the_ladder_is_climbed_from_the_cheapest_rung() {
        let c = controller();
        let breaching = ClassPressure {
            pressure_seconds: 1.0,
            slo_seconds: 0.1,
            ..blank(WorkloadClass::Interactive)
        };
        // With nothing forbidden, the first rung is the cheapest one
        let decision = c.choose_actuator(&breaching, BottleneckKind::None, Instant::now());
        assert_eq!(decision.level, ActuatorLevel::ReduceDop);
        assert_eq!(decision.dop_scale_pct, REDUCED_DOP_PCT as u16);
    }

    #[test]
    fn io_bound_skips_the_rung_that_would_not_help() {
        let c = controller();
        let breaching = ClassPressure {
            pressure_seconds: 1.0,
            slo_seconds: 0.1,
            ..blank(WorkloadClass::Interactive)
        };
        let decision = c.choose_actuator(&breaching, BottleneckKind::Io, Instant::now());
        // Taking workers away frees nothing when the cores are already waiting
        assert_ne!(decision.level, ActuatorLevel::ReduceDop);
        assert_eq!(decision.level, ActuatorLevel::DelayAdmission);
    }

    #[test]
    fn a_class_under_its_objective_provokes_nothing() {
        let c = controller();
        let calm = ClassPressure {
            pressure_seconds: 0.01,
            slo_seconds: 0.1,
            ..blank(WorkloadClass::Interactive)
        };
        let decision = c.choose_actuator(&calm, BottleneckKind::None, Instant::now());
        assert_eq!(decision.level, ActuatorLevel::Steady);
    }

    #[test]
    fn background_never_breaches_and_never_actuates() {
        let c = controller();
        let enormous = ClassPressure {
            pressure_seconds: 100_000.0,
            slo_seconds: f64::INFINITY,
            ..blank(WorkloadClass::Background)
        };
        assert!(!enormous.breaching());
        assert_eq!(
            c.choose_actuator(&enormous, BottleneckKind::None, Instant::now())
                .level,
            ActuatorLevel::Steady
        );
    }

    #[test]
    fn applying_reduce_dop_moves_the_parallel_scale() {
        let c = controller();
        c.capacity().set_scale_pct(100);
        let breaching = ClassPressure {
            pressure_seconds: 1.0,
            slo_seconds: 0.1,
            ..blank(WorkloadClass::Interactive)
        };
        let decision = c.choose_actuator(&breaching, BottleneckKind::None, Instant::now());
        c.apply(&decision);
        assert_eq!(c.capacity().scale_pct(), REDUCED_DOP_PCT);
        // And a steady decision puts it back
        c.apply(&ActuatorDecision::steady(WorkloadClass::Interactive, 0.0));
        assert_eq!(
            c.capacity().scale_pct(),
            crate::pressure::PARALLEL_SCALE_FULL_PCT
        );
    }

    // -- knee ---------------------------------------------------------------

    #[test]
    fn the_ceiling_grows_while_throughput_improves() {
        let c = controller();
        let class = WorkloadClass::Interactive;
        let start = c.ceiling(class);
        let mut now = Instant::now();
        for i in 1..=5 {
            // Rising completions mean rising measured throughput
            for _ in 0..(i * 20) {
                c.counters(class).start_direct(0.001);
                c.counters(class).complete(0.001, 0.001);
            }
            now += Duration::from_millis(100);
            c.tick(now);
        }
        assert!(
            c.ceiling(class) > start,
            "ceiling stayed at {start} while throughput rose"
        );
    }

    #[test]
    fn a_breach_pulls_the_ceiling_back() {
        let c = controller();
        let class = WorkloadClass::Interactive;
        let mut now = Instant::now();
        // Grow it first
        for i in 1..=5 {
            for _ in 0..(i * 20) {
                c.counters(class).start_direct(0.001);
                c.counters(class).complete(0.001, 0.001);
            }
            now += Duration::from_millis(100);
            c.tick(now);
        }
        let grown = c.ceiling(class);

        // Now pile on work with no completions, so pressure runs past the
        // objective
        for _ in 0..200 {
            c.counters(class).enqueue(1.0);
        }
        now += Duration::from_millis(100);
        c.tick(now);
        assert!(
            c.ceiling(class) < grown,
            "ceiling stayed at {grown} through a breach"
        );
        assert!(c.ceiling(class) >= MIN_CEILING);
    }

    #[test]
    fn the_ceiling_never_falls_below_one() {
        let c = controller();
        let class = WorkloadClass::Interactive;
        let mut now = Instant::now();
        for _ in 0..200 {
            for _ in 0..50 {
                c.counters(class).enqueue(10.0);
            }
            now += Duration::from_millis(100);
            c.tick(now);
        }
        assert!(c.ceiling(class) >= MIN_CEILING);
    }

    /// The controller must never conclude anything from the core count.
    /// Concurrency is not core bound: a query parked on storage has released
    /// its thread.
    #[test]
    fn the_starting_ceiling_is_not_the_core_count() {
        let c = controller();
        let cores = std::thread::available_parallelism()
            .map(|n| n.get() as u32)
            .unwrap_or(4);
        assert_eq!(c.ceiling(WorkloadClass::Interactive), INITIAL_CEILING);
        if cores != INITIAL_CEILING {
            assert_ne!(c.ceiling(WorkloadClass::Interactive), cores);
        }
    }

    #[test]
    fn exploration_is_recorded_in_the_ledger() {
        let c = controller();
        let class = WorkloadClass::Interactive;
        let mut now = Instant::now();
        // Settle: a little traffic, well under the objective
        for _ in 0..3 {
            c.counters(class).start_direct(0.0005);
            c.counters(class).complete(0.0005, 0.0005);
            now += Duration::from_millis(100);
            c.tick(now);
        }
        let before = c.ceiling(class);
        c.force_exploration(class);
        now += Duration::from_millis(100);
        c.tick(now);
        let stepped = c.ceiling(class);
        assert!(stepped > before, "no step past {before}");

        // The next tick scores it and writes the ledger entry
        now += Duration::from_millis(100);
        c.tick(now);
        let ledger = c.ledger(8);
        assert!(!ledger.is_empty(), "exploration left no record");
        assert_eq!(ledger[0].class, class);
        assert_eq!(ledger[0].ceiling_before, before);
    }

    // -- attribution and rings ----------------------------------------------

    #[test]
    fn pressure_is_attributed_to_the_tenant_that_caused_it() {
        let c = controller();
        for _ in 0..3 {
            c.admit(0.5, false, Some("acme"));
        }
        c.admit(0.5, false, Some("globex"));
        let rows = c.tenant_pressure();
        assert_eq!(rows.len(), 2);
        assert_eq!(rows[0].tenant_id, "acme", "the heavier tenant sorts first");
        assert!((rows[0].total_work_seconds() - 1.5).abs() < 1e-6);
        assert!((rows[1].total_work_seconds() - 0.5).abs() < 1e-6);
    }

    #[test]
    fn admissions_are_recorded_with_their_reason() {
        let c = controller();
        c.admit(0.01, false, Some("acme"));
        let records = c.admissions(4);
        assert_eq!(records.len(), 1);
        assert_eq!(records[0].decision, "admit");
        assert_eq!(records[0].tenant_id.as_deref(), Some("acme"));
    }

    #[test]
    fn history_is_bounded_and_newest_first() {
        let c = controller();
        let mut now = Instant::now();
        for _ in 0..5 {
            now += Duration::from_millis(100);
            c.tick(now);
        }
        let history = c.history(3);
        assert_eq!(history.len(), 3);
        assert!(history[0].at_us >= history[1].at_us);
    }

    #[test]
    fn a_ring_overwrites_instead_of_growing() {
        let mut ring: Ring<u32> = Ring::new(4);
        for i in 0..10 {
            ring.push(i);
        }
        assert_eq!(ring.recent(100).len(), 4);
        assert_eq!(ring.recent(4), vec![9, 8, 7, 6]);
        assert_eq!(ring.recent(100).len(), 4);
    }

    #[test]
    fn a_tick_advances_the_gossip_sequence() {
        let c = controller();
        let before = c.sequence();
        c.tick(Instant::now());
        assert_eq!(c.sequence(), before + 1);
        assert_eq!(c.ticks(), 1);
    }

    #[test]
    fn the_calibration_error_follows_what_queries_actually_cost() {
        let c = controller();
        let class = WorkloadClass::Bulk;
        let mut now = Instant::now();
        // Every query costs twice its estimate
        for _ in 0..20 {
            for _ in 0..10 {
                c.counters(class).start_direct(1.0);
                c.complete(class, 1.0, 2.0, None);
            }
            now += Duration::from_millis(100);
            c.tick(now);
        }
        let error = c.counters(class).calibration_error();
        assert!(
            error > 1.5,
            "estimates were half the truth but error reads {error}"
        );
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
