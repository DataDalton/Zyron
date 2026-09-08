//! What a running online DDL operation is doing right now.
//!
//! An index build, a constraint validation and a shadow rewrite all run while
//! the table stays open to everyone else, so the only way to know one is in
//! flight is to ask. Each registers a row here when it starts and the row goes
//! away when it finishes, successfully or not, because a row that outlived its
//! operation would read as an operation that never ends.
//!
//! The row is held in memory rather than in the catalog: it describes work in
//! this process, and a node that restarts has no such work in flight, so
//! nothing here needs to survive a restart.

use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use zyron_pressure::{ActuatorLevel, BottleneckKind, PressureController};

/// Which online operation a progress row belongs to.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DdlOperation {
    CreateIndex,
    Reindex,
    ValidateConstraint,
    ShadowRewrite,
}

impl DdlOperation {
    pub const fn as_str(self) -> &'static str {
        match self {
            DdlOperation::CreateIndex => "create_index",
            DdlOperation::Reindex => "reindex",
            DdlOperation::ValidateConstraint => "validate_constraint",
            DdlOperation::ShadowRewrite => "shadow_rewrite",
        }
    }
}

/// Where the operation is in its sequence.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DdlPhase {
    /// Writing the catalog entry that makes writers maintain the new thing
    Publishing,
    /// Waiting for the transactions that were running at publication to end
    WaitingOldTxns,
    /// Reading the rows that predate publication
    Scanning,
    /// Writing the sorted keys into the tree
    Loading,
    /// Applying what committed after the scan snapshot
    CatchingUp,
    /// Installing the finished result
    Swapping,
    /// Standing down until the node has room again
    PausedOnPressure,
}

impl DdlPhase {
    pub const fn as_str(self) -> &'static str {
        match self {
            DdlPhase::Publishing => "publishing",
            DdlPhase::WaitingOldTxns => "waiting_old_txns",
            DdlPhase::Scanning => "scanning",
            DdlPhase::Loading => "loading",
            DdlPhase::CatchingUp => "catching_up",
            DdlPhase::Swapping => "swapping",
            DdlPhase::PausedOnPressure => "paused_on_pressure",
        }
    }

    const fn code(self) -> u8 {
        match self {
            DdlPhase::Publishing => 0,
            DdlPhase::WaitingOldTxns => 1,
            DdlPhase::Scanning => 2,
            DdlPhase::Loading => 3,
            DdlPhase::CatchingUp => 4,
            DdlPhase::Swapping => 5,
            DdlPhase::PausedOnPressure => 6,
        }
    }

    const fn from_code(c: u8) -> DdlPhase {
        match c {
            0 => DdlPhase::Publishing,
            1 => DdlPhase::WaitingOldTxns,
            2 => DdlPhase::Scanning,
            3 => DdlPhase::Loading,
            4 => DdlPhase::CatchingUp,
            5 => DdlPhase::Swapping,
            _ => DdlPhase::PausedOnPressure,
        }
    }
}

/// One operation's live state, read by the view and written by the operation.
#[derive(Debug)]
pub struct DdlProgress {
    pub id: u64,
    pub table: String,
    pub operation: DdlOperation,
    /// The object being built or validated, so a table with two builds in
    /// flight says which row is which
    pub object: String,
    pub issuing_session: String,
    pub started_at_secs: u64,
    phase: AtomicU64,
    rows_done: AtomicU64,
    rows_total_estimate: AtomicU64,
    bytes_spilled: AtomicU64,
    /// The pressure signal that paused the operation, empty when running
    pause_signal: parking_lot::RwLock<Option<String>>,
}

impl DdlProgress {
    /// Moves the operation to a phase, which is what the view reports.
    pub fn set_phase(&self, phase: DdlPhase) {
        self.phase.store(phase.code() as u64, Ordering::Release);
        if phase != DdlPhase::PausedOnPressure {
            *self.pause_signal.write() = None;
        }
    }

    pub fn phase(&self) -> DdlPhase {
        DdlPhase::from_code(self.phase.load(Ordering::Acquire) as u8)
    }

    /// Adds to the rows this operation has finished with. Only ever grows, so
    /// a reader watching the view sees a number that moves one way.
    pub fn add_rows(&self, n: u64) {
        self.rows_done.fetch_add(n, Ordering::Relaxed);
    }

    pub fn rows_done(&self) -> u64 {
        self.rows_done.load(Ordering::Relaxed)
    }

    pub fn set_rows_total_estimate(&self, n: u64) {
        self.rows_total_estimate.store(n, Ordering::Relaxed);
    }

    pub fn rows_total_estimate(&self) -> u64 {
        self.rows_total_estimate.load(Ordering::Relaxed)
    }

    pub fn add_bytes_spilled(&self, n: u64) {
        self.bytes_spilled.fetch_add(n, Ordering::Relaxed);
    }

    pub fn bytes_spilled(&self) -> u64 {
        self.bytes_spilled.load(Ordering::Relaxed)
    }

    /// Records that the operation has stood down, and what it stood down for.
    pub fn pause(&self, signal: String) {
        *self.pause_signal.write() = Some(signal);
        self.phase
            .store(DdlPhase::PausedOnPressure.code() as u64, Ordering::Release);
    }

    /// Returns to a running phase after a pause.
    pub fn resume(&self, phase: DdlPhase) {
        *self.pause_signal.write() = None;
        self.phase.store(phase.code() as u64, Ordering::Release);
    }

    pub fn pause_signal(&self) -> Option<String> {
        self.pause_signal.read().clone()
    }
}

/// Every online DDL operation running in this process.
#[derive(Default)]
pub struct DdlProgressRegistry {
    rows: scc::HashMap<u64, Arc<DdlProgress>>,
    next_id: AtomicU64,
}

impl DdlProgressRegistry {
    pub fn new() -> Self {
        Self {
            rows: scc::HashMap::new(),
            next_id: AtomicU64::new(1),
        }
    }

    /// Registers an operation and returns a handle that removes the row when
    /// it is dropped.
    ///
    /// Dropping is what removes it, rather than a call at the end of the
    /// happy path, because an operation that fails halfway is exactly the one
    /// whose row must not be left behind.
    pub fn begin(
        self: &Arc<Self>,
        table: &str,
        object: &str,
        operation: DdlOperation,
        issuing_session: &str,
    ) -> DdlProgressHandle {
        let id = self.next_id.fetch_add(1, Ordering::Relaxed);
        let progress = Arc::new(DdlProgress {
            id,
            table: table.to_string(),
            operation,
            object: object.to_string(),
            issuing_session: issuing_session.to_string(),
            started_at_secs: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_secs())
                .unwrap_or(0),
            phase: AtomicU64::new(DdlPhase::Publishing.code() as u64),
            rows_done: AtomicU64::new(0),
            rows_total_estimate: AtomicU64::new(0),
            bytes_spilled: AtomicU64::new(0),
            pause_signal: parking_lot::RwLock::new(None),
        });
        let _ = self.rows.insert_sync(id, Arc::clone(&progress));
        DdlProgressHandle {
            registry: Arc::clone(self),
            progress,
        }
    }

    /// Every live operation, oldest first.
    pub fn rows(&self) -> Vec<Arc<DdlProgress>> {
        let mut out = Vec::new();
        self.rows.iter_sync(|_, v| {
            out.push(Arc::clone(v));
            true
        });
        out.sort_by_key(|r| r.id);
        out
    }

    /// The row an index build registered, by index name, so the index view
    /// can point at it.
    pub fn row_for_object(&self, table: &str, object: &str) -> Option<Arc<DdlProgress>> {
        self.rows()
            .into_iter()
            .find(|r| r.table == table && r.object == object)
    }

    fn end(&self, id: u64) {
        let _ = self.rows.remove_sync(&id);
    }
}

/// The live row for one operation, removed from the registry on drop.
pub struct DdlProgressHandle {
    registry: Arc<DdlProgressRegistry>,
    progress: Arc<DdlProgress>,
}

impl DdlProgressHandle {
    pub fn progress(&self) -> &Arc<DdlProgress> {
        &self.progress
    }
}

impl std::ops::Deref for DdlProgressHandle {
    type Target = DdlProgress;

    fn deref(&self) -> &DdlProgress {
        &self.progress
    }
}

impl Drop for DdlProgressHandle {
    fn drop(&mut self) {
        self.registry.end(self.progress.id);
    }
}

/// How many consecutive elevated readings turn a yield into a pause.
///
/// One reading is a spike, which a yield already absorbs. A run of them is the
/// node telling the build it has no room, and standing down is the only
/// response that gives the room back.
const SUSTAINED_CHECKS: u32 = 3;

/// How long a paused build waits before looking again.
const PAUSE_POLL: std::time::Duration = std::time::Duration::from_millis(200);

/// Runs a build's between-batch pressure check.
///
/// A build is background work, so it yields the moment the node reaches for
/// any actuator above reducing parallelism, and stands down entirely once the
/// node has been asking for several checks running. It never takes the
/// foreground path: the check is a read of the controller's own state, not an
/// admission request.
pub struct BuildPacer {
    elevated_streak: u32,
    running_phase: DdlPhase,
}

impl BuildPacer {
    pub fn new(running_phase: DdlPhase) -> Self {
        Self {
            elevated_streak: 0,
            running_phase,
        }
    }

    /// The phase the pacer returns to after a pause.
    pub fn set_running_phase(&mut self, phase: DdlPhase) {
        self.running_phase = phase;
    }

    /// Called between batches. Yields under pressure and blocks while the
    /// node stays under it.
    pub async fn between_batches(&mut self, progress: &DdlProgress) {
        let controller = PressureController::global();
        let level = controller.actuator();
        if level <= ActuatorLevel::ReduceDop {
            if self.elevated_streak > 0 {
                self.elevated_streak = 0;
                progress.resume(self.running_phase);
            }
            tokio::task::yield_now().await;
            return;
        }

        self.elevated_streak = self.elevated_streak.saturating_add(1);
        if self.elevated_streak < SUSTAINED_CHECKS {
            tokio::task::yield_now().await;
            return;
        }

        progress.pause(pressure_signal(controller.bottleneck(), level));
        loop {
            tokio::time::sleep(PAUSE_POLL).await;
            let controller = PressureController::global();
            let level = controller.actuator();
            if level <= ActuatorLevel::ReduceDop {
                self.elevated_streak = 0;
                progress.resume(self.running_phase);
                return;
            }
            progress.pause(pressure_signal(controller.bottleneck(), level));
        }
    }
}

/// The signal a paused build reports, which names what the node is short of
/// and what it did about it.
fn pressure_signal(bottleneck: BottleneckKind, level: ActuatorLevel) -> String {
    format!("{} / {}", bottleneck.as_str(), level.as_str())
}
