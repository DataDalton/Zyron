// -----------------------------------------------------------------------------
// Adaptive Clustering maintenance.
//
// Runs clustering passes, compaction and log collapse for lake tables whose
// schedule asks for them. The decisions all live in zyron-lake: this worker
// chooses which table to look at and when, hands the pass what it needs, and
// records what came back.
//
// **It is woken by change, not by a clock.** Everything the worker decides is
// a function of a table's manifest, and a manifest changes only when a commit
// publishes a version. A commit therefore records its head in the node's
// maintenance signal and the worker drains that record, so a table nobody
// wrote costs nothing at all and a table that just committed is looked at as
// soon as the spacing floor allows rather than up to a full interval later.
// Two things genuinely move without a commit and each has its own deadline:
// version retention expires on the wall clock, and read traffic changes what
// layout the workload wants without writing anything. The second is found by
// one sweep of the workload observer per epoch, which costs the observer's
// fixed size however many tables the node hosts.
//
// **A table is followed until it settles.** One pass rewrites at most
// `max_inputs` files, so a table far from its target shape needs several. An
// evaluation that made progress and left work behind re-arms the table
// instead of dropping it back onto a timer, which is what lets repair keep up
// with ingest. One that made no progress backs off, because a gate that
// refused this layout will refuse it again until something changes.
//
// **Rewrites are paced.** Following a table until it settles would otherwise
// let background work take the disk from queries, so every rewrite is charged
// against a node-wide byte budget and a table that cannot be admitted waits
// for the budget rather than being skipped.
//
// Two things it deliberately does not do. It never overrides an operator:
// under Force the declared spec is applied and the gate's verdict is
// reported rather than enforced, and under Auto and Hybrid the gate is
// binding so a pass can only improve a layout or leave it alone. And it
// never blocks a query: passes stage their candidates outside the active
// file set and touch it only through one metadata commit.
// -----------------------------------------------------------------------------

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use tracing::{debug, info, warn};

use zyron_catalog::Catalog;
use zyron_common::{ClusterDecision, DeploymentMode, LabeledMetrics};
use zyron_lake::{
    Decision, GateConfig, LakePaths, ResumeOptions, TablePassOptions, TransactionLog,
};

/// Longest a table waits to be looked at with nothing driving it. Reaching
/// this means neither a commit nor read evidence nor an expiring retention
/// window asked for anything, so it is a floor under staleness rather than a
/// working cadence
pub const DEFAULT_INTERVAL_SECS: u64 = 300;

/// Shortest gap between two evaluations of one table.
///
/// A table under load commits continuously and every commit is a reason to
/// look, so without a floor the worker would evaluate per commit and spend
/// more on deciding than on the work. This is also the pace a table with a
/// backlog is followed at
pub const DEFAULT_MIN_SPACING_SECS: u64 = 5;

/// Tables one wake evaluates before yielding. A node whose whole table set
/// commits at once still hands the runtime back between batches
pub const DEFAULT_MAX_TABLES_PER_WAKE: usize = 64;

/// Bytes background rewrites may move per second, averaged over the burst.
///
/// This is the whole reason following a table until it settles is safe:
/// clustering and compaction read and rewrite whole files, and a table far
/// from its shape would otherwise take the disk from the queries the rewrite
/// exists to speed up
pub const DEFAULT_REWRITE_BYTES_PER_SEC: u64 = 64 * 1024 * 1024;

/// Bytes the rewrite budget may bank while nothing is rewriting, so an idle
/// node can repair a table promptly instead of metering it from zero
pub const DEFAULT_REWRITE_BURST_BYTES: u64 = 1024 * 1024 * 1024;

/// Versions a table's log accumulates between checkpoints. Reconstruction
/// replays from the newest checkpoint, so this bounds both the replay
/// length of a cache miss and the number of version files GC must retain
const CHECKPOINT_EVERY: u64 = 64;

/// Times an evaluation that made no progress doubles its own wait before the
/// wait stops growing. Four doublings turns the default interval into a bit
/// over an hour, which is long enough that a permanently refused layout is
/// free and short enough that a changed workload is still picked up
const MAX_BACKOFF_DOUBLINGS: u32 = 4;

#[derive(Debug, Clone)]
pub struct LakeClusteringConfig {
    pub interval_secs: u64,
    /// Input files one pass rewrites, which is what bounds its memory
    pub max_inputs: usize,
    pub target_rows_per_file: u64,
    pub gate: GateConfig,
    pub data_dir: PathBuf,
    /// Shortest gap between two evaluations of one table
    pub min_spacing_secs: u64,
    /// Tables one wake evaluates before yielding
    pub max_tables_per_wake: usize,
    /// Bytes background rewrites may move per second, None to leave them
    /// unmetered
    pub rewrite_bytes_per_sec: Option<u64>,
    /// Bytes the rewrite budget may bank while nothing is rewriting
    pub rewrite_burst_bytes: u64,
}

impl LakeClusteringConfig {
    pub fn new(data_dir: PathBuf) -> Self {
        Self {
            interval_secs: DEFAULT_INTERVAL_SECS,
            max_inputs: zyron_lake::DEFAULT_MAX_INPUTS,
            target_rows_per_file: zyron_lake::DEFAULT_ROWS_PER_FILE,
            gate: GateConfig::default(),
            data_dir,
            min_spacing_secs: DEFAULT_MIN_SPACING_SECS,
            max_tables_per_wake: DEFAULT_MAX_TABLES_PER_WAKE,
            rewrite_bytes_per_sec: Some(DEFAULT_REWRITE_BYTES_PER_SEC),
            rewrite_burst_bytes: DEFAULT_REWRITE_BURST_BYTES,
        }
    }

    fn min_spacing(&self) -> Duration {
        Duration::from_secs(self.min_spacing_secs.max(1))
    }
}

/// What the worker has done since startup.
#[derive(Debug, Default)]
pub struct LakeClusteringStats {
    pub passes: AtomicU64,
    pub accepted: AtomicU64,
    pub refused: AtomicU64,
    pub files_rewritten: AtomicU64,
    pub bytes_written: AtomicU64,
    pub resumed: AtomicU64,
    /// Times the worker woke, whether or not it found anything
    pub wakes: AtomicU64,
    /// Tables looked at. The gap between this and `wakes` times the table
    /// count is what the change signal saves
    pub evaluations: AtomicU64,
    /// Evaluations that found the table unchanged since the last one, which
    /// is a spurious wake rather than work
    pub unchanged: AtomicU64,
    /// Rewrites the byte budget held back, so a node that is metering its
    /// own maintenance says so rather than looking idle
    pub budget_deferrals: AtomicU64,
}

pub struct LakeClusteringWorker {
    shutdown: Arc<AtomicBool>,
    stats: Arc<LakeClusteringStats>,
    handle: Option<tokio::task::JoinHandle<()>>,
}

impl LakeClusteringWorker {
    /// Starts the worker, or returns None on a node that does not run the
    /// lake tier. A db node must not pay for a thread it can never use
    pub fn start(
        mode: DeploymentMode,
        catalog: Arc<Catalog>,
        metrics: Option<Arc<LabeledMetrics>>,
        config: LakeClusteringConfig,
    ) -> Option<Self> {
        if !mode.runs_lake_tier() {
            return None;
        }
        let shutdown = Arc::new(AtomicBool::new(false));
        let stats = Arc::new(LakeClusteringStats::default());
        let handle = tokio::spawn(clustering_loop(
            Arc::clone(&shutdown),
            Arc::clone(&stats),
            catalog,
            metrics,
            config,
        ));
        info!("lake clustering worker started");
        Some(Self {
            shutdown,
            stats,
            handle: Some(handle),
        })
    }

    pub fn stats(&self) -> &Arc<LakeClusteringStats> {
        &self.stats
    }

    pub fn shutdown(&mut self) {
        self.shutdown.store(true, Ordering::Release);
        if let Some(handle) = self.handle.take() {
            handle.abort();
        }
    }
}

/// What the worker remembers about one table between wakes.
///
/// All of it is in memory. Losing it on restart costs one evaluation per
/// table, which is what startup does anyway, so nothing here is durable
/// state the node depends on
#[derive(Debug)]
struct TableState {
    /// Published version the last evaluation read. Everything the worker
    /// decides comes from the manifest at this version, so an unchanged
    /// version means an unchanged decision and the evaluation can stop
    /// before it reconstructs anything
    seen_version: u64,
    /// A commit published since the last evaluation
    dirty: bool,
    /// The soonest this table may be evaluated again, the floor that stops
    /// a table under continuous ingest being evaluated per commit
    not_before: Instant,
    /// When this table must be evaluated with no commit behind it, None
    /// when nothing time driven is outstanding and a commit is the only
    /// thing that will bring it back
    due: Option<Instant>,
    /// Consecutive evaluations that made no progress, what the backoff
    /// grows on
    quiet_rounds: u32,
    /// Newest version the log has a manifest checkpoint at, read from the
    /// hint file once and tracked from there
    last_checkpoint: Option<u64>,
    /// Oldest version time travel still needs. The retention floor only
    /// rises, so the next evaluation walks up from here rather than down
    /// from the head over history it already judged
    retain_min: Option<u64>,
    /// When the table was last evaluated, what fair share orders on
    last_evaluated: Instant,
}

impl TableState {
    fn new(now: Instant) -> Self {
        Self {
            seen_version: 0,
            dirty: false,
            not_before: now,
            due: Some(now),
            quiet_rounds: 0,
            last_checkpoint: None,
            retain_min: None,
            last_evaluated: now,
        }
    }

    /// The soonest this table wants to be looked at, None when nothing does.
    ///
    /// A published commit wants an evaluation immediately and a deadline
    /// wants one at its own instant, and both are held back to the spacing
    /// floor rather than allowed to jump it
    fn next_action(&self) -> Option<Instant> {
        let wanted = if self.dirty {
            Some(self.not_before)
        } else {
            self.due
        };
        wanted.map(|at| at.max(self.not_before))
    }

    fn is_ready(&self, now: Instant) -> bool {
        self.next_action().is_some_and(|at| at <= now)
    }

    /// Brings a deadline forward, leaving an earlier one alone
    fn schedule(&mut self, at: Instant) {
        self.due = Some(match self.due {
            Some(existing) => existing.min(at),
            None => at,
        });
    }
}

/// Bytes background rewrites may move, as a bucket that refills with time.
///
/// A rewrite is admitted on the balance it finds and charged what it turned
/// out to move, because its cost is not known until it is done. That lets
/// one large pass overdraw, which the next refill pays back, rather than
/// requiring the worker to predict a rewrite before running it
struct RewriteBudget {
    bytes_per_sec: Option<u64>,
    burst: i64,
    /// Signed, because a rewrite is charged what it turned out to move and
    /// may overdraw. Carrying the debt is what makes the pacing hold: a
    /// pass that moved a gigabyte waits out a gigabyte of refill, where a
    /// balance clamped at zero would forgive it after one tick
    available: i64,
    refilled: Instant,
}

impl RewriteBudget {
    fn new(config: &LakeClusteringConfig, now: Instant) -> Self {
        let burst = config.rewrite_burst_bytes.min(i64::MAX as u64).max(1) as i64;
        Self {
            bytes_per_sec: config.rewrite_bytes_per_sec.filter(|rate| *rate > 0),
            burst,
            available: burst,
            refilled: now,
        }
    }

    fn refill(&mut self, now: Instant) {
        let Some(rate) = self.bytes_per_sec else {
            return;
        };
        let elapsed = now.saturating_duration_since(self.refilled).as_micros();
        // Refills in whole microseconds of credit, so a wake that lands
        // sooner than one byte of refill leaves the clock where it was and
        // the credit is not rounded away
        let gained = (elapsed as u64).saturating_mul(rate) / 1_000_000;
        if gained > 0 {
            self.available = self
                .available
                .saturating_add(gained.min(i64::MAX as u64) as i64)
                .min(self.burst);
            self.refilled = now;
        }
    }

    fn admits(&mut self, now: Instant) -> bool {
        if self.bytes_per_sec.is_none() {
            return true;
        }
        self.refill(now);
        self.available > 0
    }

    fn charge(&mut self, bytes: u64) {
        if self.bytes_per_sec.is_some() {
            self.available = self
                .available
                .saturating_sub(bytes.min(i64::MAX as u64) as i64);
        }
    }

    /// How long until the bucket carries anything again.
    ///
    /// Capped at the time an empty bucket takes to refill, so however large
    /// a single rewrite turned out to be, background maintenance is never
    /// held off for longer than that
    fn ready_in(&self) -> Duration {
        match self.bytes_per_sec {
            Some(rate) if self.available <= 0 => {
                let deficit = 1i64.saturating_sub(self.available) as u64;
                let micros = deficit.saturating_mul(1_000_000) / rate.max(1);
                let cap = Duration::from_secs((self.burst as u64) / rate.max(1) + 1);
                Duration::from_micros(micros).clamp(Duration::from_millis(1), cap)
            }
            _ => Duration::ZERO,
        }
    }
}

async fn clustering_loop(
    shutdown: Arc<AtomicBool>,
    stats: Arc<LakeClusteringStats>,
    catalog: Arc<Catalog>,
    metrics: Option<Arc<LabeledMetrics>>,
    config: LakeClusteringConfig,
) {
    // Passes a crash left half done are finished or unwound before any new
    // one starts, so a resumed pass never races a fresh one over the same
    // staging directory
    let mut states: HashMap<u32, TableState> = HashMap::new();
    let startup = Instant::now();
    for (name, log) in lake_tables(&catalog, &config.data_dir) {
        match zyron_lake::resume_cluster_passes(
            &log,
            pass_attempt(),
            &[],
            &ResumeOptions::default(),
        ) {
            Ok(outcomes) => {
                for outcome in outcomes {
                    stats.resumed.fetch_add(1, Ordering::Relaxed);
                    info!(
                        table = %name,
                        pass_id = outcome.pass_id,
                        version = ?outcome.version,
                        "resumed an interrupted clustering pass"
                    );
                }
            }
            Err(e) => warn!(table = %name, error = %e, "clustering resume failed"),
        }
        // Nothing is known about a table until it has been looked at once,
        // so every table the node holds open is due at startup
        if let Some(id) = log.paths().table_id() {
            states.entry(id).or_insert_with(|| TableState::new(startup));
        }
    }

    // A commit wakes the worker rather than a clock finding the commit
    // later. One worker owns the record, so a second installation would be
    // two workers dividing one node's work between them without knowing it
    let notify = Arc::new(tokio::sync::Notify::new());
    let signal = zyron_lake::maintenance_signal();
    {
        let waker = Arc::clone(&notify);
        if !signal.set_waker(Box::new(move || waker.notify_one())) {
            warn!(
                "the lake maintenance signal already has a waker, this worker will run on its \
                 deadlines alone"
            );
        }
    }
    let mut seen_generation = signal.generation();

    let pass_counter = AtomicU64::new(1);
    let mut budget = RewriteBudget::new(&config, startup);
    // Read traffic moves what layout a table wants without committing
    // anything, and the observer decays on an epoch, so once per epoch is
    // both the soonest the answer can differ and the cheapest way to ask
    let evidence_period = Duration::from_secs(zyron_lake::workload::EPOCH_SECONDS.max(1));
    let mut evidence_checked = startup;

    loop {
        if shutdown.load(Ordering::Acquire) {
            break;
        }
        stats.wakes.fetch_add(1, Ordering::Relaxed);
        let now = Instant::now();

        // Every head that published since the last drain. A generation that
        // has not moved means nothing committed anywhere, so the record is
        // not even locked
        let generation = signal.generation();
        if generation != seen_generation {
            let drained = signal.drain();
            seen_generation = generation;
            if drained.overflowed {
                // More heads committed than the signal tracks, so the list
                // is no longer the whole truth and the worker falls back to
                // what it would have done on a timer
                debug!("lake maintenance signal overflowed, enumerating every open table");
                for (_, log) in lake_tables(&catalog, &config.data_dir) {
                    if let Some(id) = log.paths().table_id() {
                        states
                            .entry(id)
                            .or_insert_with(|| TableState::new(now))
                            .dirty = true;
                    }
                }
            }
            for head in drained.heads {
                // Branch heads commit independently of the table they forked
                // from and are not maintained on this schedule, so a branch
                // commit is recorded and then left alone
                if !head.main_head {
                    continue;
                }
                states
                    .entry(head.table_id)
                    .or_insert_with(|| TableState::new(now))
                    .dirty = true;
            }
        }

        if now.saturating_duration_since(evidence_checked) >= evidence_period {
            evidence_checked = now;
            for table_id in zyron_lake::observer().tables_with_evidence(zyron_lake::current_epoch())
            {
                states
                    .entry(table_id)
                    .or_insert_with(|| TableState::new(now))
                    .schedule(now);
            }
        }

        // Oldest waiting first, so a node with more ready tables than one
        // wake evaluates cannot starve any of them
        let mut ready: Vec<u32> = states
            .iter()
            .filter(|(_, state)| state.is_ready(now))
            .map(|(id, _)| *id)
            .collect();
        ready.sort_unstable_by_key(|id| {
            states
                .get(id)
                .map(|state| state.last_evaluated)
                .unwrap_or(now)
        });
        ready.truncate(config.max_tables_per_wake.max(1));

        for table_id in ready {
            if shutdown.load(Ordering::Acquire) {
                break;
            }
            let Some((name, log)) = resolve_table(&catalog, &config.data_dir, table_id) else {
                // The table was dropped, or its log is not open on this
                // node, so there is nothing here to maintain
                states.remove(&table_id);
                continue;
            };
            let Some(state) = states.get_mut(&table_id) else {
                continue;
            };
            let pass_id = pass_counter.fetch_add(1, Ordering::Relaxed);
            evaluate_table(
                &catalog,
                &name,
                &log,
                table_id,
                pass_id,
                state,
                &mut budget,
                &stats,
                metrics.as_deref(),
                &config,
            )
            .await;
        }

        if shutdown.load(Ordering::Acquire) {
            break;
        }
        let now = Instant::now();
        let wake_at = next_wake(&states, evidence_checked + evidence_period, now, &config);
        tokio::select! {
            _ = notify.notified() => {}
            _ = tokio::time::sleep_until(tokio::time::Instant::from_std(wake_at)) => {}
        }
    }
}

/// When the worker has something to do next.
///
/// The evidence sweep always has a deadline, so this is never unbounded, and
/// a node with nothing else outstanding sleeps exactly until that sweep
/// unless a commit wakes it first. The floor stops a wake that could not
/// finish its ready set from spinning
fn next_wake(
    states: &HashMap<u32, TableState>,
    evidence_at: Instant,
    now: Instant,
    config: &LakeClusteringConfig,
) -> Instant {
    let mut earliest = evidence_at;
    for state in states.values() {
        if let Some(at) = state.next_action() {
            earliest = earliest.min(at);
        }
    }
    // A ceiling on the sleep itself, not on how long a table may stay
    // quiet. A settled table with nothing driving it is meant to wait for a
    // commit, and this only keeps the worker from sleeping past a deadline
    // that moved while it slept
    let sleep_ceiling = now + Duration::from_secs(config.interval_secs.max(1));
    earliest
        .min(sleep_ceiling)
        .max(now + Duration::from_millis(10))
}

/// What one evaluation found, which is what decides when the table is looked
/// at next
#[derive(Debug, Default, Clone, Copy)]
struct TableVerdict {
    /// A version landed, so the manifest the next evaluation reads differs
    /// from the one this evaluation judged
    progressed: bool,
    /// Work the table still wants that this evaluation did not finish
    backlog: bool,
    /// Drift is over the threshold the table itself set, which is the table
    /// saying its layout is costing every query right now
    urgent: bool,
    /// The table's own bound on how often it may be passed
    repair_interval: Duration,
    /// A rewrite was held back by the byte budget rather than by anything
    /// about the table
    budget_bound: bool,
    /// How long until version retention can collect more history, None when
    /// only a commit will change that
    gc_in: Option<Duration>,
    /// Version of the manifest this evaluation drew its conclusions from,
    /// None when it did not get far enough to read one
    judged_version: Option<u64>,
}

/// Looks at one table and decides when to look at it again.
#[allow(clippy::too_many_arguments)]
async fn evaluate_table(
    catalog: &Catalog,
    name: &str,
    log: &TransactionLog,
    table_id: u32,
    pass_id: u64,
    state: &mut TableState,
    budget: &mut RewriteBudget,
    stats: &LakeClusteringStats,
    metrics: Option<&LabeledMetrics>,
    config: &LakeClusteringConfig,
) {
    let now = Instant::now();
    // A deadline is the table asking to be looked at with no commit behind
    // it, which is exactly the case the unchanged-version shortcut must not
    // swallow: an expiring retention window and stale read evidence both
    // move without publishing a version
    let deadline_fired = state.due.is_some_and(|at| at <= now);
    state.dirty = false;
    state.due = None;
    state.last_evaluated = now;
    state.not_before = now + config.min_spacing();
    stats.evaluations.fetch_add(1, Ordering::Relaxed);

    let version = log.latest_version();
    if version == state.seen_version && state.retain_min.is_some() && !deadline_fired {
        // Nothing published since the last evaluation and the retention
        // floor was already established, so every answer this evaluation
        // could reach is the answer the last one reached. This is the whole
        // point of the change signal: no manifest is reconstructed, no file
        // set is swept, and the table costs one atomic load
        stats.unchanged.fetch_add(1, Ordering::Relaxed);
        return;
    }

    let mut verdict = TableVerdict {
        repair_interval: Duration::from_secs(config.interval_secs.max(1)),
        ..TableVerdict::default()
    };
    // A step that failed leaves the table in a state nothing here judged, so
    // it is brought back on the backoff rather than left waiting for a
    // commit that may never come
    let mut failed = false;
    match collapse_log(catalog, log, state) {
        Ok(gc_in) => verdict.gc_in = gc_in,
        Err(e) => {
            warn!(table = %name, error = %e, "lake log maintenance failed");
            failed = true;
        }
    }

    match run_one_table(
        catalog, name, log, table_id, pass_id, budget, stats, metrics, config,
    )
    .await
    {
        Ok(pass) => {
            verdict.progressed = pass.progressed;
            verdict.backlog = pass.backlog;
            verdict.urgent = pass.urgent;
            verdict.budget_bound = pass.budget_bound;
            verdict.repair_interval = pass.repair_interval;
            verdict.judged_version = pass.judged_version;
        }
        Err(e) => {
            warn!(table = %name, error = %e, "clustering pass failed");
            failed = true;
        }
    }
    if failed {
        verdict.backlog = true;
        verdict.progressed = false;
    }

    // Only a version whose consequences this evaluation actually read counts
    // as considered. A commit that landed while it ran is left to the next
    // one rather than skipped as already seen
    if let Some(judged) = verdict.judged_version {
        state.seen_version = judged;
    }
    reschedule(state, &verdict, budget, config, Instant::now());
}

/// Decides when a table is looked at again from what the evaluation found.
///
/// Four outcomes, and the order they are tested in is the order of urgency.
/// A rewrite the budget held back is not the table's fault, so it waits for
/// the budget and nothing else. A table over its own drift threshold is
/// being served badly now and is followed at the spacing floor. A table with
/// a backlog that just made progress is followed at its own interval, which
/// is how repair keeps up with ingest instead of losing 16 files a pass to a
/// five minute clock. A table that made no progress backs off, because
/// whatever refused this layout will refuse it again until something moves.
///
/// A settled table gets no deadline at all. It comes back when it commits
fn reschedule(
    state: &mut TableState,
    verdict: &TableVerdict,
    budget: &RewriteBudget,
    config: &LakeClusteringConfig,
    now: Instant,
) {
    if let Some(gc_in) = verdict.gc_in {
        state.schedule(now + gc_in);
    }
    if verdict.budget_bound {
        state.schedule(now + budget.ready_in().max(config.min_spacing()));
        return;
    }
    if !verdict.backlog {
        state.quiet_rounds = 0;
        return;
    }
    if verdict.urgent {
        state.quiet_rounds = 0;
        state.schedule(now + config.min_spacing());
        return;
    }
    if verdict.progressed {
        state.quiet_rounds = 0;
        state.schedule(now + verdict.repair_interval);
        return;
    }
    let doublings = state.quiet_rounds.min(MAX_BACKOFF_DOUBLINGS);
    state.quiet_rounds = state.quiet_rounds.saturating_add(1);
    let wait = verdict
        .repair_interval
        .saturating_mul(1u32 << doublings)
        .max(config.min_spacing());
    state.schedule(now + wait);
}

/// Collapses one table's log: writes a manifest checkpoint at the published
/// head once enough versions accumulated since the last one, then removes
/// version files older than what the table's time-travel window and the
/// newest checkpoint still need. A table with no retention window promises
/// no history, so only the published head has to stay reconstructable.
///
/// Returns how long until retention can collect more history, None when only
/// a commit will change that. Both halves are cheap to repeat because the
/// checkpoint boundary and the retention floor are both carried forward: the
/// hint file is read once per table rather than once per tick, and the floor
/// walks up over the versions that have expired since the last look instead
/// of down over the whole history every time
fn collapse_log(
    catalog: &Catalog,
    log: &TransactionLog,
    state: &mut TableState,
) -> Result<Option<Duration>, zyron_common::ZyronError> {
    let latest = log.latest_version();
    if latest == 0 {
        return Ok(None);
    }
    let last_checkpoint = match state.last_checkpoint {
        Some(v) => v,
        None => std::fs::read_to_string(log.paths().last_checkpoint_hint())
            .ok()
            .and_then(|text| text.trim().parse::<u64>().ok())
            .unwrap_or(0),
    };
    let last_checkpoint = if latest >= last_checkpoint + CHECKPOINT_EVERY {
        log.checkpoint(latest)?;
        latest
    } else {
        last_checkpoint
    };
    state.last_checkpoint = Some(last_checkpoint);

    let retention_secs = log
        .paths()
        .table_id()
        .and_then(|id| catalog.get_table_by_id(zyron_catalog::TableId(id)).ok())
        .map(|entry| entry.time_travel_retention_secs)
        .unwrap_or(0);
    let now_us = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_micros() as i64)
        .unwrap_or(0);
    let floor_us = now_us.saturating_sub((retention_secs as i64).saturating_mul(1_000_000));
    let retain_min = match state.retain_min {
        Some(from) => advance_retain_min(log, from, latest, floor_us)?,
        None => oldest_needed_version(log, latest, floor_us)?,
    };
    state.retain_min = Some(retain_min);
    let removed = log.gc_versions(retain_min)?;
    if removed > 0 {
        debug!(
            table_root = %log.paths().root().display(),
            removed,
            retain_min,
            "collected lake log versions"
        );
    }
    Ok(next_gc_deadline(
        log,
        retain_min,
        latest,
        retention_secs,
        now_us,
    ))
}

/// How long until the retention window releases more history.
///
/// The oldest version the window still needs stops being needed when the
/// version above it ages past the window, so that version's own timestamp
/// plus the window is the exact instant, read from one small header. A floor
/// already at the head has nothing above it to wait for, so only a commit
/// changes anything and no deadline is set
fn next_gc_deadline(
    log: &TransactionLog,
    retain_min: u64,
    latest: u64,
    retention_secs: u64,
    now_us: i64,
) -> Option<Duration> {
    if retain_min >= latest {
        return None;
    }
    let path = log.paths().version_file(retain_min + 1);
    let header = zyron_lake::transaction_log::read_commit_header(&path).ok()?;
    let expires_us = header
        .timestamp_us
        .saturating_add((retention_secs as i64).saturating_mul(1_000_000));
    Some(Duration::from_micros(
        expires_us.saturating_sub(now_us).max(0) as u64,
    ))
}

/// The oldest version any query inside the retention window can resolve:
/// AS OF a timestamp at the floor resolves to the newest version committed
/// at or before it, so that version and everything after must survive.
/// Walks version headers from the head downwards and stops at the first
/// one at or before the floor, reading one small header per version.
///
/// Used for the first look at a table, where there is no floor to walk up
/// from. Later looks use `advance_retain_min`
fn oldest_needed_version(
    log: &TransactionLog,
    latest: u64,
    floor_us: i64,
) -> Result<u64, zyron_common::ZyronError> {
    let mut retain_min = 1;
    for v in (1..=latest).rev() {
        let path = log.paths().version_file(v);
        let header = match zyron_lake::transaction_log::read_commit_header(&path) {
            Ok(h) => h,
            // already collected below here, nothing older is needed
            Err(_) if !path.exists() => {
                retain_min = v + 1;
                break;
            }
            Err(e) => return Err(e),
        };
        retain_min = v;
        if header.timestamp_us <= floor_us {
            break;
        }
    }
    Ok(retain_min)
}

/// The same answer as `oldest_needed_version`, reached from a floor already
/// known to be at or below it.
///
/// The retention floor only rises, because wall clock time only moves one
/// way, so a floor established earlier is still a valid lower bound and the
/// versions between it and the answer are exactly the ones that expired
/// since. Walking up over those costs one header per newly expired version
/// instead of one per version of retained history on every look, which is
/// what turns log collapse from O(history) per evaluation into O(expired)
fn advance_retain_min(
    log: &TransactionLog,
    from: u64,
    latest: u64,
    floor_us: i64,
) -> Result<u64, zyron_common::ZyronError> {
    let mut retain_min = from.max(1);
    while retain_min < latest {
        let next = retain_min + 1;
        let path = log.paths().version_file(next);
        let header = match zyron_lake::transaction_log::read_commit_header(&path) {
            Ok(h) => h,
            // Already collected, so nothing at or below it is needed
            Err(_) if !path.exists() => {
                retain_min = next;
                continue;
            }
            Err(e) => return Err(e),
        };
        if header.timestamp_us > floor_us {
            break;
        }
        retain_min = next;
    }
    Ok(retain_min)
}

/// Rewrites one table when it has drifted far enough from its target
/// shape, without being asked.
///
/// Two thresholds, both table properties, both read from the manifest with
/// no IO: `auto_compact_small_file_ratio` and `auto_compact_dead_row_ratio`.
/// Crossing either is enough, and crossing both runs one compaction rather
/// than two, because one rewrite settles both.
///
/// This runs on the clustering schedule, so a table set to OnDemand is not
/// compacted unasked either. That is the same promise the schedule already
/// makes: nothing rewrites this table unless someone says so.
///
/// A failure is logged and swallowed. Compaction is maintenance, and a
/// table that could not be compacted this tick is a table that reads a
/// little more until the next one, not a table that should stop the loop.
///
/// Returns the bytes the rewrite moved, which is what the caller charges
/// against the node's rewrite budget
fn auto_compact(
    name: &str,
    log: &TransactionLog,
    table_id: u32,
    config: &LakeClusteringConfig,
) -> u64 {
    let Ok(before) = log.latest_manifest() else {
        return 0;
    };
    let need = before.compaction_need(config.target_rows_per_file);
    let Some(trigger) = need.trigger else {
        return 0;
    };

    let dead_before = need.pending_deleted_rows;
    let outcome = match zyron_lake::optimize(
        log,
        pass_attempt(),
        table_id as u64,
        config.target_rows_per_file,
    ) {
        Ok(o) => o,
        Err(e) => {
            warn!(
                table = %name,
                trigger = %trigger,
                error = %e,
                "auto compaction failed, the table keeps the shape it had"
            );
            return 0;
        }
    };
    let files_after = log
        .latest_manifest()
        .map(|m| m.entries.len())
        .unwrap_or(need.total_files);
    let dead_after = log
        .latest_manifest()
        .map(|m| m.pending_deleted_rows())
        .unwrap_or(dead_before);

    zyron_lake::compaction_history::compaction_history().record(
        zyron_lake::compaction_history::CompactionRecord {
            table_id,
            table_name: name.to_string(),
            trigger,
            triggered_at_us: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_micros() as i64)
                .unwrap_or(0),
            files_before: need.total_files,
            files_after,
            dead_rows_reclaimed: dead_before.saturating_sub(dead_after),
            small_file_ratio_milli: zyron_lake::compaction_history::ratio_milli(
                need.small_files as u64,
                need.total_files as u64,
            ),
            dead_row_ratio_milli: zyron_lake::compaction_history::ratio_milli(
                need.pending_deleted_rows,
                need.total_rows,
            ),
            version: outcome.version,
        },
    );
    info!(
        table = %name,
        trigger = %trigger,
        files_before = need.total_files,
        files_after,
        version = ?outcome.version,
        "compacted a table that drifted from its target shape"
    );
    outcome.bytes_rewritten
}

/// What the pass half of an evaluation found
#[derive(Debug, Default, Clone, Copy)]
struct PassSummary {
    progressed: bool,
    backlog: bool,
    urgent: bool,
    budget_bound: bool,
    repair_interval: Duration,
    /// Version of the manifest the conclusions were drawn from. A commit
    /// that landed after it was read is not covered by them, so recording
    /// this rather than whatever the head reads afterwards is what stops a
    /// concurrent write being skipped as already considered
    judged_version: Option<u64>,
}

#[allow(clippy::too_many_arguments)]
async fn run_one_table(
    catalog: &Catalog,
    name: &str,
    log: &TransactionLog,
    table_id: u32,
    pass_id: u64,
    budget: &mut RewriteBudget,
    stats: &LakeClusteringStats,
    metrics: Option<&LabeledMetrics>,
    config: &LakeClusteringConfig,
) -> Result<PassSummary, zyron_common::ZyronError> {
    let manifest = log.latest_manifest()?;
    let schedule = manifest.clustering_schedule();
    let mut summary = PassSummary {
        repair_interval: Duration::from_secs(
            manifest
                .cluster_repair_interval_secs(config.interval_secs)
                .max(1),
        ),
        judged_version: Some(manifest.snapshot_id),
        ..PassSummary::default()
    };
    if let Some(metrics) = metrics {
        metrics.clusteringPolicySet(name, manifest.clustering_mode(), schedule);
    }
    // OnDemand means exactly that: only OPTIMIZE starts a pass
    if !schedule.runs_in_background() {
        return Ok(summary);
    }
    // A rewrite the node cannot afford right now is deferred rather than
    // shrunk, because a half sized pass costs a whole read of its inputs
    if !budget.admits(Instant::now()) {
        stats.budget_deferrals.fetch_add(1, Ordering::Relaxed);
        summary.budget_bound = true;
        summary.backlog = true;
        return Ok(summary);
    }
    let before_version = log.latest_version();

    // Index runs first, because it is cheap to decide and a fragmented
    // index costs every probe until it is collapsed. Each write commit
    // appends its own index file over whatever keys it touched, so their
    // ranges overlap and a probe stops being able to prune to one file.
    // The check reads the manifest and opens nothing
    match zyron_lake::operations::compact_indexes_if_fragmented(
        log,
        pass_attempt(),
        table_id as u64,
    ) {
        Ok(Some(version)) => tracing::info!(
            target: "zyron::lake",
            table = name,
            version,
            "compacted fragmented index runs"
        ),
        Ok(None) => {}
        Err(e) => tracing::warn!(
            target: "zyron::lake",
            table = name,
            error = %e,
            "index compaction failed, probes keep using the runs they have"
        ),
    }
    // What a plan is costed against before this evaluation touches
    // anything. A pass that moves it has to evict the plans that priced the
    // old layout, and this is the one caller that can move it with no
    // statement behind it, so nothing else would notice
    let epoch_before = log.latest_manifest().map(|m| m.clustering_epoch()).ok();

    // Compaction before clustering, for the same reason OPTIMIZE runs its
    // delete pass first: a rewrite that drops logically deleted rows and
    // merges undersized files leaves the clustering pass fewer rows and
    // fewer inputs to carry
    budget.charge(auto_compact(name, log, table_id, config));

    // The layout decision itself lives in zyron-lake, so a scheduled pass
    // and an OPTIMIZE ... CLUSTER pass choose from the same evidence
    let started = Instant::now();
    let report = zyron_lake::run_table_cluster_pass(
        log,
        pass_attempt(),
        table_id,
        &TablePassOptions {
            pass_id,
            target_rows_per_file: config.target_rows_per_file,
            max_inputs: config.max_inputs,
            gate: config.gate,
            max_proposed_keys: zyron_lake::DEFAULT_MAX_PROPOSED_KEYS,
        },
    )?;
    if let Some(metrics) = metrics {
        metrics.clusteringWorkloadWindowSet(name, report.evidence_columns as u64);
        metrics.clusteringPendingProposalsSet(name, u64::from(report.proposal_pending));
    }
    // Planning reads the catalog, so the layout the files now carry has to
    // be recorded there or every later plan is judged against keys no file
    // is sorted by. Skipped when nothing changed, which is most evaluations
    match catalog
        .set_active_cluster_keys(zyron_catalog::TableId(table_id), &report.active_keys)
        .await
    {
        Ok(true) => tracing::info!(
            target: "zyron::lake",
            table = name,
            keys = report.active_keys.len(),
            "recorded the layout the table is now clustered by"
        ),
        Ok(false) => {}
        Err(e) => tracing::warn!(
            target: "zyron::lake",
            table = name,
            error = %e,
            "recording the active cluster keys failed, plans will be judged against the \
             previous layout"
        ),
    }
    if log.latest_manifest().map(|m| m.clustering_epoch()).ok() != epoch_before {
        catalog.bump_schema_version();
    }

    // What the table still wants, read from the manifest the evaluation
    // leaves behind rather than the one it started from, so following a
    // table until it settles is judged on the shape it now has
    let after = log.latest_manifest()?;
    summary.judged_version = Some(after.snapshot_id);
    summary.progressed = after.snapshot_id != before_version;
    summary.repair_interval = Duration::from_secs(
        after
            .cluster_repair_interval_secs(config.interval_secs)
            .max(1),
    );
    let drifted = zyron_lake::drifted_file_count(&after);
    summary.urgent = drifted > after.cluster_repair_urgency_threshold();
    summary.backlog = drifted > 0
        || after
            .compaction_need(config.target_rows_per_file)
            .trigger
            .is_some();
    if summary.urgent {
        info!(
            table = %name,
            drifted,
            threshold = after.cluster_repair_urgency_threshold(),
            "repairing ahead of the interval, the layout has drifted far enough to be costing \
             every query"
        );
    }

    // Nothing declared and nothing measured worth ordering by
    let Some(outcome) = report.outcome else {
        return Ok(summary);
    };
    if outcome.inputs == 0 {
        return Ok(summary);
    }
    budget.charge(outcome.bytes_written);

    stats.passes.fetch_add(1, Ordering::Relaxed);
    stats
        .files_rewritten
        .fetch_add(outcome.inputs as u64, Ordering::Relaxed);
    stats
        .bytes_written
        .fetch_add(outcome.bytes_written, Ordering::Relaxed);
    if outcome.version.is_some() {
        stats.accepted.fetch_add(1, Ordering::Relaxed);
    } else {
        stats.refused.fetch_add(1, Ordering::Relaxed);
    }

    if let Some(metrics) = metrics {
        // The delta the gate computed, whether or not it was binding, so a
        // Force table that is being served badly still says so
        let delta = match &outcome.decision {
            Decision::Accept { delta } => *delta,
            Decision::BelowThreshold { delta, .. } => *delta,
            Decision::Worse { delta, .. } => *delta,
            _ => 0.0,
        };
        metrics.clusteringPass(
            name,
            decision_label(&outcome.decision, outcome.version.is_some()),
            outcome.inputs as u64,
            outcome.outputs as u64,
            outcome.bytes_written,
            delta,
            started.elapsed().as_micros() as u64,
        );
        if let Some(rate) = report.measured_skip_rate {
            metrics.clusteringSkipRateSet(name, rate);
        }
    }

    debug!(
        table = %name,
        pass_id,
        inputs = outcome.inputs,
        outputs = outcome.outputs,
        version = ?outcome.version,
        decision = ?outcome.decision,
        "clustering pass finished"
    );
    Ok(summary)
}

/// The metric label for what happened. A pass the gate refused but Force
/// applied anyway is recorded as accepted, because that is what happened
/// to the files, and the refusal reason still reaches the skip-rate delta
fn decision_label(decision: &Decision, committed: bool) -> ClusterDecision {
    if committed {
        return ClusterDecision::Accepted;
    }
    match decision {
        Decision::Accept { .. } => ClusterDecision::Accepted,
        Decision::Worse { .. } => ClusterDecision::RejectedWorse,
        Decision::BelowThreshold { .. } => ClusterDecision::RejectedBelowThreshold,
        Decision::AnchorConflict { .. } => ClusterDecision::RejectedAnchorConflict,
        Decision::ReplayDiverged { .. } => ClusterDecision::RejectedReplayDiverged,
    }
}

/// Every lake table whose log this node already holds open. A table with no
/// open log has never been touched since startup, so there is nothing to
/// cluster and opening it here would be startup IO charged to a timer
fn lake_tables(catalog: &Catalog, data_dir: &Path) -> Vec<(String, Arc<TransactionLog>)> {
    let mut out = Vec::new();
    for table in catalog.list_all_tables() {
        if !table.lake.is_lake() {
            continue;
        }
        let paths = LakePaths::new(data_dir, table.id.0);
        if let Some(log) = TransactionLog::lookup_shared(&paths) {
            out.push((table.name.clone(), log));
        }
    }
    out
}

/// One table by id, without enumerating the catalog.
///
/// The change signal names the tables that moved, so the worker resolves
/// exactly those. Enumerating every table to find the two that committed is
/// the cost the signal exists to remove
fn resolve_table(
    catalog: &Catalog,
    data_dir: &Path,
    table_id: u32,
) -> Option<(String, Arc<TransactionLog>)> {
    let entry = catalog
        .get_table_by_id(zyron_catalog::TableId(table_id))
        .ok()?;
    if !entry.lake.is_lake() {
        return None;
    }
    let paths = LakePaths::new(data_dir, table_id);
    let log = TransactionLog::lookup_shared(&paths)?;
    Some((entry.name.clone(), log))
}

/// Maintenance commits stand alone. `db_txn_id` zero publishes immediately
/// rather than waiting on a database transaction that does not exist
fn pass_attempt() -> zyron_lake::CommitAttempt<'static> {
    zyron_lake::CommitAttempt {
        operation: zyron_lake::OperationKind::Optimize,
        db_txn_id: 0,
        commit_lsn: 0,
        timestamp_us: SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_micros() as i64)
            .unwrap_or(0),
        read_predicate: None,
        read_version: 0,
        audit: None,
        deadline: None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_schema() -> zyron_lake::LakeSchema {
        zyron_lake::LakeSchema::new(
            1,
            vec![zyron_lake::LakeColumn {
                id: 0,
                name: "id".into(),
                type_id: zyron_common::TypeId::Int64,
                nullable: false,
                fractional_digits: None,
                tz_offset_secs: None,
                max_length: None,
                default_expr: None,
            }],
        )
        .expect("schema")
    }

    fn stamped_attempt(timestamp_us: i64) -> zyron_lake::CommitAttempt<'static> {
        zyron_lake::CommitAttempt {
            operation: zyron_lake::OperationKind::Append,
            db_txn_id: 0,
            commit_lsn: 0,
            timestamp_us,
            read_predicate: None,
            read_version: 0,
            audit: None,
            deadline: None,
        }
    }

    fn add_file(partition_id: u64) -> zyron_lake::LogEntry {
        zyron_lake::LogEntry::AddFile(zyron_lake::PartitionEntry {
            partition_id,
            size_bytes: 8,
            row_count: 1,
            added_version: 0,
            cluster_spec_id: 0,
            column_stats: std::sync::Arc::new(vec![]),
            delete_predicate_ids: vec![],
        })
    }

    /// A table state as `evaluate_table` leaves it just before it decides
    /// when to come back: the deadline that brought it here is spent and
    /// the spacing floor is already behind it
    fn evaluated_state(now: Instant) -> TableState {
        let mut state = TableState::new(now);
        state.dirty = false;
        state.due = None;
        state.not_before = now;
        state
    }

    fn six_version_log(dir: &std::path::Path) -> TransactionLog {
        let log = TransactionLog::create(
            LakePaths::new(dir, 9),
            zyron_lake::CommitAttempt {
                operation: zyron_lake::OperationKind::SchemaChange,
                ..stamped_attempt(1_000)
            },
            &test_schema(),
            None,
            &std::collections::BTreeMap::new(),
        )
        .expect("create");
        // versions 2 to 6 at timestamps 2000, 3000, 4000, 5000, 6000
        for i in 0..5u64 {
            log.commit(stamped_attempt(2_000 + i as i64 * 1_000), |_| {
                Ok(vec![add_file(0x100 + i)])
            })
            .expect("append");
        }
        log
    }

    /// The retention floor decides which version history GC may take: the
    /// newest version at or before the floor still serves AS OF queries at
    /// the floor, so it and everything after survive, and with no window
    /// only the head does
    #[test]
    fn test_oldest_needed_version_honors_the_floor() {
        let dir = tempfile::tempdir().expect("tempdir");
        let log = six_version_log(dir.path());
        let latest = log.latest_version();
        assert_eq!(latest, 6);

        // Floor inside the history: version 4 (ts 4000) serves AS OF 4500
        let retain = oldest_needed_version(&log, latest, 4_500).expect("floor mid history");
        assert_eq!(retain, 4);
        // Floor after every commit: only the head is needed
        let retain = oldest_needed_version(&log, latest, 10_000).expect("floor after history");
        assert_eq!(retain, 6);
        // Floor before every commit: everything is needed
        let retain = oldest_needed_version(&log, latest, 500).expect("floor before history");
        assert_eq!(retain, 1);

        // The checkpoint plus GC pass built on that answer reclaims the
        // history below it and keeps everything at or above readable
        log.checkpoint(6).expect("checkpoint");
        let removed = log.gc_versions(6).expect("gc");
        assert_eq!(
            removed, 6,
            "history at and below the checkpoint is reclaimed"
        );
        let m = log
            .manifest_at(6)
            .expect("head reconstructs from the checkpoint");
        assert_eq!(m.entries.len(), 5);
        // A collected tail is reported as the boundary above it rather
        // than replayed
        let retain = oldest_needed_version(&log, latest, 500).expect("collected tail");
        assert_eq!(retain, 7, "the walk stops at the collected boundary");
    }

    /// The whole point of carrying the floor forward is that it reaches the
    /// same answer as the full walk. A cheaper answer that disagreed would
    /// either collect history a query still needs or keep it forever
    #[test]
    fn test_advancing_the_floor_agrees_with_walking_from_the_head() {
        let dir = tempfile::tempdir().expect("tempdir");
        let log = six_version_log(dir.path());
        let latest = log.latest_version();
        for floor in [0i64, 500, 1_000, 2_500, 4_000, 4_500, 6_000, 10_000] {
            let full = oldest_needed_version(&log, latest, floor).expect("full walk");
            // Every valid lower bound has to reach the same answer, which
            // is what makes carrying the previous answer forward safe
            for from in 1..=full {
                let advanced = advance_retain_min(&log, from, latest, floor).expect("advance");
                assert_eq!(
                    advanced, full,
                    "floor {} from {} walked to {} but the full walk says {}",
                    floor, from, advanced, full
                );
            }
        }
    }

    /// The floor only rises, so a series of widening floors has to walk
    /// forward monotonically rather than ever reopening collected history
    #[test]
    fn test_the_floor_only_rises_as_the_window_expires() {
        let dir = tempfile::tempdir().expect("tempdir");
        let log = six_version_log(dir.path());
        let latest = log.latest_version();
        let mut floor_version = 1u64;
        let mut seen = Vec::new();
        for floor in [500i64, 2_500, 4_500, 10_000] {
            floor_version =
                advance_retain_min(&log, floor_version, latest, floor).expect("advance");
            seen.push(floor_version);
        }
        assert_eq!(seen, vec![1, 2, 4, 6]);
    }

    /// A db node must not pay for a lake thread it can never use
    #[test]
    fn test_the_worker_does_not_start_off_the_lake_tier() {
        assert!(!DeploymentMode::Db.runs_lake_tier());
        assert!(DeploymentMode::Lake.runs_lake_tier());
        assert!(DeploymentMode::Unified.runs_lake_tier());
    }

    /// A settled table gets no deadline, which is what makes an idle node
    /// free. Everything else is a reason to come back, and the reason
    /// decides how soon
    #[test]
    fn test_a_settled_table_waits_for_a_commit_and_a_busy_one_does_not() {
        let config = LakeClusteringConfig::new(PathBuf::from("/lake"));
        let now = Instant::now();
        let budget = RewriteBudget::new(&config, now);
        let interval = Duration::from_secs(300);

        let settled = TableVerdict {
            repair_interval: interval,
            ..TableVerdict::default()
        };
        let mut state = evaluated_state(now);
        reschedule(&mut state, &settled, &budget, &config, now);
        assert_eq!(
            state.due, None,
            "nothing outstanding means nothing but a commit brings the table back"
        );
        assert!(!state.is_ready(now), "and it is not ready on its own");

        // Over its own drift threshold the table is being served badly now,
        // so it is followed at the spacing floor rather than its interval
        let urgent = TableVerdict {
            progressed: true,
            backlog: true,
            urgent: true,
            repair_interval: interval,
            ..TableVerdict::default()
        };
        let mut state = evaluated_state(now);
        reschedule(&mut state, &urgent, &budget, &config, now);
        assert_eq!(state.due, Some(now + config.min_spacing()));

        // A backlog that is making progress is followed at the table's own
        // interval, which is the operator's bound on how often it is passed
        let working = TableVerdict {
            progressed: true,
            backlog: true,
            repair_interval: interval,
            ..TableVerdict::default()
        };
        let mut state = evaluated_state(now);
        reschedule(&mut state, &working, &budget, &config, now);
        assert_eq!(state.due, Some(now + interval));
    }

    /// A pass that changes nothing will keep changing nothing until the
    /// table or the workload moves, so retrying it at the same pace burns
    /// the node down for no reason
    #[test]
    fn test_an_evaluation_that_makes_no_progress_backs_off_and_stops_growing() {
        let config = LakeClusteringConfig::new(PathBuf::from("/lake"));
        let now = Instant::now();
        let budget = RewriteBudget::new(&config, now);
        let interval = Duration::from_secs(300);
        let refused = TableVerdict {
            progressed: false,
            backlog: true,
            repair_interval: interval,
            ..TableVerdict::default()
        };
        let mut state = evaluated_state(now);
        let mut waits = Vec::new();
        for _ in 0..7 {
            state.due = None;
            reschedule(&mut state, &refused, &budget, &config, now);
            waits.push(state.due.expect("a backlog always comes back") - now);
        }
        assert_eq!(
            waits,
            vec![
                interval,
                interval * 2,
                interval * 4,
                interval * 8,
                interval * 16,
                interval * 16,
                interval * 16,
            ],
            "the wait doubles and then stops, it does not grow without bound"
        );

        // One evaluation that gets somewhere clears the whole backoff
        let working = TableVerdict {
            progressed: true,
            backlog: true,
            repair_interval: interval,
            ..TableVerdict::default()
        };
        state.due = None;
        reschedule(&mut state, &working, &budget, &config, now);
        assert_eq!(state.due, Some(now + interval));
        assert_eq!(state.quiet_rounds, 0);
    }

    /// A retention window that is about to release history has to be waited
    /// for exactly, and it never delays work the table wants sooner
    #[test]
    fn test_retention_sets_its_own_deadline_without_delaying_repair() {
        let config = LakeClusteringConfig::new(PathBuf::from("/lake"));
        let now = Instant::now();
        let budget = RewriteBudget::new(&config, now);
        let verdict = TableVerdict {
            repair_interval: Duration::from_secs(300),
            gc_in: Some(Duration::from_secs(90)),
            ..TableVerdict::default()
        };
        let mut state = evaluated_state(now);
        reschedule(&mut state, &verdict, &budget, &config, now);
        assert_eq!(
            state.due,
            Some(now + Duration::from_secs(90)),
            "a settled table still comes back when its window expires"
        );

        // Urgent repair is sooner than the window, and the sooner of the two
        // is what the table waits for
        let urgent = TableVerdict {
            progressed: true,
            backlog: true,
            urgent: true,
            gc_in: Some(Duration::from_secs(90)),
            repair_interval: Duration::from_secs(300),
            ..TableVerdict::default()
        };
        let mut state = evaluated_state(now);
        reschedule(&mut state, &urgent, &budget, &config, now);
        assert_eq!(state.due, Some(now + config.min_spacing()));
    }

    /// The budget is what makes following a table until it settles safe, so
    /// it has to actually stop admitting and then actually recover
    #[test]
    fn test_the_rewrite_budget_meters_and_refills() {
        let mut config = LakeClusteringConfig::new(PathBuf::from("/lake"));
        config.rewrite_bytes_per_sec = Some(1_000_000);
        config.rewrite_burst_bytes = 4_000_000;
        let start = Instant::now();
        let mut budget = RewriteBudget::new(&config, start);
        assert!(budget.admits(start), "a fresh budget carries its burst");
        budget.charge(4_000_000);
        assert!(!budget.admits(start), "a spent budget stops admitting");
        assert!(budget.ready_in() > Duration::ZERO);
        // One second of credit at a megabyte a second
        assert!(
            budget.admits(start + Duration::from_secs(1)),
            "time refills it"
        );
        // A rewrite may overdraw, because its cost is only known once it
        // has run, and the debt is paid back rather than forgiven
        budget.charge(3_000_000);
        assert!(
            !budget.admits(start + Duration::from_secs(2)),
            "one second of refill does not clear a two megabyte debt"
        );
        assert!(
            budget.admits(start + Duration::from_secs(4)),
            "three more seconds of refill does"
        );
        // However far a single pass overdraws, the wait is capped at the
        // time an empty bucket takes to refill
        budget.charge(u64::MAX);
        assert!(budget.ready_in() <= Duration::from_secs(5));

        // Unmetered is a real setting and admits without consulting a clock
        config.rewrite_bytes_per_sec = None;
        let mut unmetered = RewriteBudget::new(&config, start);
        unmetered.charge(u64::MAX);
        assert!(unmetered.admits(start));
    }

    /// The sleep is bounded by the evidence sweep even with no table
    /// outstanding, and never lands in the past
    #[test]
    fn test_the_wake_is_bounded_and_never_in_the_past() {
        let config = LakeClusteringConfig::new(PathBuf::from("/lake"));
        let now = Instant::now();
        let evidence_at = now + Duration::from_secs(300);

        let empty: HashMap<u32, TableState> = HashMap::new();
        assert_eq!(next_wake(&empty, evidence_at, now, &config), evidence_at);

        // A table due sooner pulls the wake forward
        let mut states = HashMap::new();
        let mut state = TableState::new(now);
        state.due = Some(now + Duration::from_secs(7));
        state.not_before = now;
        states.insert(1u32, state);
        assert_eq!(
            next_wake(&states, evidence_at, now, &config),
            now + Duration::from_secs(7)
        );

        // A table already overdue does not produce a wake in the past, which
        // would spin the loop
        let mut states = HashMap::new();
        let mut state = TableState::new(now);
        state.due = Some(now - Duration::from_secs(60));
        state.not_before = now - Duration::from_secs(60);
        states.insert(1u32, state);
        assert!(next_wake(&states, evidence_at, now, &config) > now);
    }

    /// A commit is a reason to look and the spacing floor is the answer to
    /// a table that commits continuously
    #[test]
    fn test_the_spacing_floor_holds_a_commit_storm_back() {
        let now = Instant::now();
        let mut state = TableState::new(now);
        state.due = None;
        state.not_before = now + Duration::from_secs(5);
        state.dirty = true;
        assert!(
            !state.is_ready(now),
            "a commit inside the floor waits for the floor"
        );
        assert_eq!(state.next_action(), Some(now + Duration::from_secs(5)));
        assert!(state.is_ready(now + Duration::from_secs(5)));
    }

    /// A refusal has to reach the metric as the reason it was refused, and
    /// a Force pass that committed reads as accepted because that is what
    /// happened to the files
    #[test]
    fn test_decision_labels_report_what_happened() {
        assert_eq!(
            decision_label(&Decision::Accept { delta: 0.5 }, true),
            ClusterDecision::Accepted
        );
        assert_eq!(
            decision_label(
                &Decision::BelowThreshold {
                    delta: 0.0,
                    required: 0.05
                },
                true
            ),
            ClusterDecision::Accepted,
            "Force applied it, so the files did change"
        );
        assert_eq!(
            decision_label(
                &Decision::BelowThreshold {
                    delta: 0.0,
                    required: 0.05
                },
                false
            ),
            ClusterDecision::RejectedBelowThreshold
        );
        assert_eq!(
            decision_label(
                &Decision::Worse {
                    class: 0,
                    delta: -0.2
                },
                false
            ),
            ClusterDecision::RejectedWorse
        );
        assert_eq!(
            decision_label(
                &Decision::AnchorConflict {
                    expected: vec![1],
                    found: vec![0]
                },
                false
            ),
            ClusterDecision::RejectedAnchorConflict
        );
        assert_eq!(
            decision_label(
                &Decision::ReplayDiverged {
                    classes: 2,
                    tolerance: 0.25
                },
                false
            ),
            ClusterDecision::RejectedReplayDiverged
        );
    }
}
