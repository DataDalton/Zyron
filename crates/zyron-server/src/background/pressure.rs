//! The pressure controller's clock, calibration collector, and gossip publisher.
//!
//! Three loops on different timescales run from one thread, because they are
//! the same loop observed at different rates and splitting them across threads
//! would only add a way for them to disagree about what the node is doing.
//!
//! The fast tick advances the controller: it closes the measurement windows,
//! moves each class's concurrency ceiling toward where throughput actually
//! stopped improving, classifies what is limiting the node, and applies the
//! cheapest legal response. This is also the cadence the signal is published
//! at, because the node's own knee detection is the mesh's scale-out trigger
//! and two independent heuristics would fight: the node would throttle while
//! the mesh saw idle cores and scaled in.
//!
//! The calibration drain folds what the operators have measured into the live
//! coefficients, on a slower cadence, so a query never pays for the fold.
//!
//! The persist pass writes the coefficients back under this node's hardware
//! fingerprint, which is what lets the next node of the same shape start
//! calibrated instead of guessing.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

use zyron_buffer::BufferPool;
use zyron_pressure::capability::{
    CalibrationField, NodeCapabilities, OperatorCoefficients, OperatorKind, calibration_key,
    decode_coefficient, encode_coefficient, parse_calibration_key,
};
use zyron_pressure::hot_set::HotSetManifest;
use zyron_pressure::pressure::{
    PressureField, WorkloadClass, clamp_payload, encode_status, encode_versioned, pressure_key,
};
use zyron_pressure::pressure_control::{MeshReach, PressureController};
use zyron_pressure::provisioner::{MeshSection, ProvisionerDriver};
use zyron_types::scheduling::QuotaRegistry;

/// How often the controller advances and republishes.
///
/// Fast enough that a burst is seen while it is still a burst, slow enough
/// that the loop itself is not part of the load. Everything the tick reads is
/// an atomic, so a pass costs microseconds.
const TICK_INTERVAL: Duration = Duration::from_millis(100);

/// How often measured operator costs are folded into the live coefficients.
const CALIBRATION_INTERVAL: Duration = Duration::from_secs(30);

/// How often the coefficients are written back to the calibration cache.
///
/// Far slower than the drain, because the value of persisting is that the next
/// node of this hardware shape starts calibrated, and that does not need the
/// last thirty seconds of drift.
const PERSIST_INTERVAL: Duration = Duration::from_secs(300);

/// How long the node serves before it will consider an active probe.
///
/// The rule is that nothing calibration-related runs at startup, and this is
/// what enforces it. Five minutes is also exactly the window the recent-sample
/// count covers, so the first pass is taken with a full window of real
/// evidence behind it and probes only what that traffic did not reach.
const PROBE_STARTUP_GRACE: Duration = Duration::from_secs(300);

/// How often the stale coefficients are refreshed.
const PROBE_INTERVAL: Duration = Duration::from_secs(300);

/// How often the working-set manifest is written.
///
/// The manifest is what a resuming node reads instead of warming from cold, so
/// it has to be recent enough to be worth reading. A pass writes the resident
/// page identifiers, which is a bulk sequential write of a few hundred
/// kilobytes, so this is far cheaper than the cold start it prevents.
const HOT_SET_INTERVAL: Duration = Duration::from_secs(300);

/// What the worker has done, for the views and for the tests.
#[derive(Debug, Default)]
pub struct PressureWorkerStats {
    pub ticks: AtomicU64,
    pub calibration_drains: AtomicU64,
    pub persists: AtomicU64,
    pub gossip_publishes: AtomicU64,
    pub gossip_keys_written: AtomicU64,
    /// Probe passes run, and the operator kinds they measured
    pub probe_passes: AtomicU64,
    pub probed_kinds: AtomicU64,
    /// Probe passes skipped because the node was busy
    pub probes_deferred: AtomicU64,
    pub hot_set_writes: AtomicU64,
    pub hot_set_pages: AtomicU64,
    /// Calibration records published, and peer records folded in
    pub calibration_keys_written: AtomicU64,
    pub peer_calibrations_adopted: AtomicU64,
}

/// Where the calibration cache lives, so the worker can persist without
/// knowing how it is stored.
pub trait CalibrationStore: Send + Sync {
    /// Writes this node's coefficients under its hardware fingerprint.
    fn persist(
        &self,
        capabilities: &NodeCapabilities,
    ) -> std::result::Result<(), zyron_common::ZyronError>;
}

/// The calibration cache as a file beside the node identity.
pub struct FileCalibrationStore {
    data_dir: std::path::PathBuf,
    node_id: u64,
}

impl FileCalibrationStore {
    pub fn new(data_dir: impl Into<std::path::PathBuf>, node_id: u64) -> Self {
        Self {
            data_dir: data_dir.into(),
            node_id,
        }
    }

    /// Reads back what this hardware shape was last measured to cost, so a
    /// node that has run here before starts calibrated.
    ///
    /// Returns false when the fingerprint is unknown, which is the case for a
    /// genuinely new shape of machine. That node serves immediately anyway,
    /// on cold start values, and replaces them from measurement within the
    /// first windows of traffic. Nothing waits on a probe.
    pub fn adopt_into(data_dir: &std::path::Path, capabilities: &mut NodeCapabilities) -> bool {
        let cache = zyron_pressure::CalibrationCache::load(data_dir);
        let Some(entry) = cache.get(capabilities.fingerprint) else {
            return false;
        };
        let coefficients = entry.coefficients.clone();
        PressureController::global()
            .coefficients()
            .seed(&coefficients);
        capabilities.adopt(coefficients);
        true
    }
}

impl CalibrationStore for FileCalibrationStore {
    fn persist(
        &self,
        capabilities: &NodeCapabilities,
    ) -> std::result::Result<(), zyron_common::ZyronError> {
        let mut cache = zyron_pressure::CalibrationCache::load(&self.data_dir);
        // Replace rather than merge: this node seeded itself from the same
        // file at startup, so the value being written already contains what
        // is on disk and merging would count that evidence twice
        cache.set(zyron_pressure::CalibrationEntry {
            fingerprint: capabilities.fingerprint,
            coefficients: capabilities.coefficients.clone(),
            updated_us: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_micros() as i64)
                .unwrap_or(0),
            source_node_id: self.node_id,
        });
        cache.persist(&self.data_dir)
    }
}

/// What the loop needs beyond the controller itself.
pub struct PressureInputs {
    pub capabilities: NodeCapabilities,
    pub quota_registry: Option<Arc<QuotaRegistry>>,
    pub store: Option<Arc<dyn CalibrationStore>>,
    /// Read for the working-set manifest, never written
    pub buffer_pool: Option<Arc<BufferPool>>,
    pub data_dir: std::path::PathBuf,
    pub mesh: MeshSection,
}

/// Runs the controller.
pub struct PressureWorker {
    shutdown: Arc<AtomicBool>,
    handle: Option<JoinHandle<()>>,
    stats: Arc<PressureWorkerStats>,
}

impl PressureWorker {
    /// Starts the loop. The capabilities carry the fingerprint the persisted
    /// calibration is keyed by.
    pub fn start(inputs: PressureInputs) -> Self {
        let shutdown = Arc::new(AtomicBool::new(false));
        let stats = Arc::new(PressureWorkerStats::default());

        let loop_shutdown = Arc::clone(&shutdown);
        let loop_stats = Arc::clone(&stats);
        let handle = thread::Builder::new()
            .name("zyron-pressure".into())
            .spawn(move || {
                worker_loop(inputs, loop_shutdown, loop_stats);
            })
            .expect("spawn pressure worker");

        Self {
            shutdown,
            handle: Some(handle),
            stats,
        }
    }

    pub fn stats(&self) -> &Arc<PressureWorkerStats> {
        &self.stats
    }

    /// Stops the loop and waits for it, so a shutdown does not race a persist
    /// half-written.
    pub fn stop(&mut self) {
        self.shutdown.store(true, Ordering::Release);
        if let Some(handle) = self.handle.take() {
            let _ = handle.join();
        }
    }
}

impl Drop for PressureWorker {
    fn drop(&mut self) {
        self.stop();
    }
}

fn worker_loop(inputs: PressureInputs, shutdown: Arc<AtomicBool>, stats: Arc<PressureWorkerStats>) {
    let PressureInputs {
        mut capabilities,
        quota_registry,
        store,
        buffer_pool,
        data_dir,
        mesh,
    } = inputs;
    let controller = PressureController::global();
    let started = Instant::now();
    let mut last_calibration = Instant::now();
    let mut last_persist = Instant::now();
    let mut last_probe = Instant::now();
    let mut last_hot_set = Instant::now();
    // Peer evidence already folded in, so a record that has not moved since
    // the last round is not counted a second time
    let mut folded: std::collections::HashMap<(u64, usize), u64> = std::collections::HashMap::new();

    // The driver is chosen once. Its capabilities are what gates the mesh
    // rungs of the ladder, so a deployment with nothing installed relieves
    // pressure locally instead of publishing a request nothing answers
    let driver = mesh.select_driver();
    controller.set_mesh_reach(MeshReach::from_capabilities(&driver.capabilities()));
    controller.set_warm_pool_cap(mesh.warm_pool_max_nodes);
    describe_provisioner(driver.as_ref());

    while !shutdown.load(Ordering::Acquire) {
        let now = Instant::now();
        controller.tick(now);
        stats.ticks.fetch_add(1, Ordering::Relaxed);

        if let Some(registry) = quota_registry.as_ref() {
            let written = publish_pressure(controller, registry);
            stats.gossip_publishes.fetch_add(1, Ordering::Relaxed);
            stats
                .gossip_keys_written
                .fetch_add(written as u64, Ordering::Relaxed);
        }

        if now.duration_since(last_calibration) >= CALIBRATION_INTERVAL {
            controller.drain_calibration();
            refresh_measured_capabilities(controller, &mut capabilities);
            if let Some(registry) = quota_registry.as_ref() {
                // Published and read on the same cadence, so a node that just
                // folded a peer's evidence republishes the result and the
                // fleet converges rather than each node holding its own answer
                let written = publish_calibration(controller, registry, &capabilities);
                stats
                    .calibration_keys_written
                    .fetch_add(written as u64, Ordering::Relaxed);
                let adopted =
                    adopt_peer_calibration(controller, registry, &capabilities, &mut folded);
                stats
                    .peer_calibrations_adopted
                    .fetch_add(adopted as u64, Ordering::Relaxed);
            }
            last_calibration = now;
            stats.calibration_drains.fetch_add(1, Ordering::Relaxed);
        }

        if now.duration_since(last_persist) >= PERSIST_INTERVAL {
            if let Some(store) = store.as_ref() {
                capabilities.coefficients = controller.coefficients().snapshot();
                match store.persist(&capabilities) {
                    Ok(()) => {
                        stats.persists.fetch_add(1, Ordering::Relaxed);
                    }
                    Err(e) => {
                        tracing::warn!(error = %e, "calibration cache write failed");
                    }
                }
            }
            last_persist = now;
        }

        // Republished every tick, because the measured latency moves as
        // provisions complete and a projection against a stale horizon is a
        // projection to the wrong instant
        controller.set_provision_horizon(effective_horizon(driver.as_ref(), &mesh));

        if now.duration_since(started) >= PROBE_STARTUP_GRACE
            && now.duration_since(last_probe) >= PROBE_INTERVAL
        {
            run_probe_pass(controller, &stats);
            last_probe = now;
        }

        if now.duration_since(last_hot_set) >= HOT_SET_INTERVAL {
            write_hot_set(controller, buffer_pool.as_deref(), &data_dir, &mesh, &stats);
            last_hot_set = now;
        }

        // Sleep the remainder of the interval rather than a fixed amount, so
        // a slow pass does not push every later tick out behind it
        let spent = Instant::now().duration_since(now);
        if let Some(rest) = TICK_INTERVAL.checked_sub(spent) {
            thread::sleep(rest);
        }
    }

    // A manifest written on the way out is the one a resuming node reads, so
    // a clean shutdown is what makes the next start warm
    write_hot_set(controller, buffer_pool.as_deref(), &data_dir, &mesh, &stats);
}

/// How long capacity takes to arrive, measured unless the operator overrode it.
///
/// An override exists because a control plane can be slower than anything this
/// node has yet observed, and a deployment that knows that should not have to
/// wait for a missed objective to teach the node.
fn effective_horizon(driver: &dyn ProvisionerDriver, mesh: &MeshSection) -> Duration {
    if mesh.provision_latency_secs > 0 {
        return Duration::from_secs(mesh.provision_latency_secs);
    }
    zyron_pressure::provisioner::ProvisionerRegistry::global().provision_latency(driver.kind())
}

/// Says once what this node can do about capacity, because the answer decides
/// how the ladder behaves and an operator debugging a node that never scales
/// should not have to infer it.
fn describe_provisioner(driver: &dyn ProvisionerDriver) {
    let capabilities = driver.capabilities();
    tracing::info!(
        provisioner = driver.kind().as_str(),
        can_provision = capabilities.can_provision,
        can_reclaim = capabilities.can_reclaim,
        max_nodes = capabilities.max_nodes,
        "pressure controller mesh reach"
    );
}

/// Refreshes the operator costs real traffic has stopped exercising.
///
/// Skipped whenever the node is doing anything about pressure. A probe is
/// worth a slice of an idle thread and never worth a slice of a busy one, and
/// a node under load is measuring those coefficients from real work anyway.
fn run_probe_pass(controller: &PressureController, stats: &PressureWorkerStats) {
    if controller.actuator() != zyron_pressure::pressure::ActuatorLevel::Steady {
        stats.probes_deferred.fetch_add(1, Ordering::Relaxed);
        return;
    }
    let summary = zyron_executor::probe::probe_stale_kinds(zyron_executor::probe::PROBE_BUDGET);
    if summary.measured.is_empty() {
        return;
    }
    stats.probe_passes.fetch_add(1, Ordering::Relaxed);
    stats
        .probed_kinds
        .fetch_add(summary.measured.len() as u64, Ordering::Relaxed);
    tracing::debug!(
        kinds = summary.measured.len(),
        elapsed_ms = summary.total_elapsed.as_millis() as u64,
        "refreshed operator costs real traffic had stopped exercising"
    );
}

/// Writes down what the node is holding, so a survivor or a resumed instance
/// can start warm.
fn write_hot_set(
    controller: &PressureController,
    buffer_pool: Option<&BufferPool>,
    data_dir: &std::path::Path,
    mesh: &MeshSection,
    stats: &PressureWorkerStats,
) {
    let Some(pool) = buffer_pool else {
        return;
    };
    let generated_us = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_micros() as i64)
        .unwrap_or(0);
    let manifest = HotSetManifest::build(
        controller.node_id(),
        generated_us,
        pool.hot_pages(mesh.hot_set_pages as usize),
        mesh.hot_set_pages,
        controller.hot_queries(mesh.hot_set_queries as usize),
        mesh.hot_set_queries,
    );
    let pages = manifest.pages.len() as u32;
    match manifest.persist(data_dir) {
        Ok(()) => {
            controller.record_hot_set(pages, generated_us, true);
            stats.hot_set_writes.fetch_add(1, Ordering::Relaxed);
            stats.hot_set_pages.store(pages as u64, Ordering::Relaxed);
        }
        Err(e) => {
            // Recorded as not persisted, which is what blocks scale to zero.
            // Reporting a manifest that is not on disk would let a node stop
            // and resume into an empty pool
            controller.record_hot_set(pages, generated_us, false);
            tracing::warn!(error = %e, "working set manifest write failed");
        }
    }
}

/// Folds measured read latency into the mount description, so the device kind
/// reported is the one the reads actually saw rather than the one the OS
/// claimed.
fn refresh_measured_capabilities(
    controller: &PressureController,
    capabilities: &mut NodeCapabilities,
) {
    let histogram = controller.read_latency();
    let samples = histogram.count();
    if samples == 0 {
        return;
    }
    let p50 = histogram.quantile_nanos(0.50);
    let p99 = histogram.quantile_nanos(0.99);
    let throughput = histogram.throughput_mb_s();
    if let Some(mount) = capabilities.mounts.first_mut() {
        mount.read_p50_ns = p50;
        mount.read_p99_ns = p99;
        mount.read_throughput_mb_s = throughput;
        mount.sample_count = mount.sample_count.saturating_add(samples);
        // Measured latency outranks whatever the OS said the device was. A
        // network volume presented as a local disk only tells the truth here
        mount.device_kind = zyron_pressure::capability::DeviceKind::from_read_latency_ns(p50);
    }
    histogram.reset();
}

/// Publishes this node's pressure into the shared quota registry.
///
/// The registry converges by taking the maximum of the local and remote value
/// for a key, which only converges on a value that never decreases. Pressure
/// rises and falls, so each record carries the tick counter in its high half
/// and the reading in its low half: a newer tick always compares greater
/// whatever the reading did, and the max merge therefore keeps the newest
/// rather than the largest.
fn publish_pressure(controller: &PressureController, registry: &QuotaRegistry) -> usize {
    let snapshot = controller.snapshot();
    let sequence = controller.sequence();
    let node_id = controller.node_id();
    let mut written = 0usize;

    for row in &snapshot.classes {
        let values = [
            (
                PressureField::PressureMicros,
                clamp_payload((row.pressure_seconds * 1e6) as u64),
            ),
            (
                PressureField::QueuedWorkMicros,
                clamp_payload((row.queued_work_seconds * 1e6) as u64),
            ),
            (
                PressureField::ActiveWorkMicros,
                clamp_payload((row.active_work_seconds * 1e6) as u64),
            ),
            (
                PressureField::ServiceCapacityMilliQps,
                clamp_payload((row.service_capacity_qps * 1000.0) as u64),
            ),
            (PressureField::InFlight, row.in_flight),
            (PressureField::Ceiling, row.ceiling),
            (
                PressureField::CalibrationErrorMilli,
                clamp_payload((row.calibration_error * 1000.0) as u64),
            ),
            (
                PressureField::StatusCodes,
                encode_status(row.bottleneck, row.actuator_level),
            ),
        ];
        for (field, payload) in values {
            let key = pressure_key(node_id, row.class, field);
            let packed = encode_versioned(sequence, payload);
            // Written through the same monotone-max merge peers converge on,
            // so a locally published record and one that arrived by gossip are
            // indistinguishable to a reader
            registry.merge_remote(&[(key, packed)]);
            written += 1;
        }
    }
    written
}

/// Publishes what this node has measured, under its hardware fingerprint.
///
/// Keyed by shape rather than by node, because that is what makes the
/// measurement worth sharing: a node joining a fleet of identical machines can
/// price its first query against what its siblings already learned instead of
/// spending its first minutes guessing.
///
/// Only kinds with evidence behind them are published. A cold-start constant
/// republished as a measurement would spread a guess through the fleet dressed
/// as a reading.
pub fn publish_calibration(
    controller: &PressureController,
    registry: &QuotaRegistry,
    capabilities: &NodeCapabilities,
) -> usize {
    let coefficients = controller.coefficients().snapshot();
    let sequence = controller.sequence();
    let node_id = controller.node_id();
    let mut written = 0usize;
    for kind in OperatorKind::ALL {
        let i = kind.index();
        let samples = coefficients.sample_count[i];
        if samples == 0 {
            continue;
        }
        let pairs = [
            (
                CalibrationField::CentiNanosPerUnit,
                encode_coefficient(coefficients.ns_per_unit[i]),
            ),
            (
                CalibrationField::SampleCount,
                zyron_pressure::pressure::clamp_payload(samples),
            ),
        ];
        for (field, payload) in pairs {
            let key = calibration_key(capabilities.fingerprint, node_id, kind, field);
            registry.merge_remote(&[(key, encode_versioned(sequence, payload))]);
            written += 1;
        }
    }
    written
}

/// Folds in what other nodes of this hardware shape have measured.
///
/// Each peer's evidence is counted exactly once. `folded` remembers how much
/// of each peer's sample count has already been taken, and only the growth
/// since then is merged, because a weighted average that re-counted the same
/// evidence every round would let one busy peer drown out every other node
/// simply by being read more often.
///
/// Returns how many peer records moved this node's numbers.
pub fn adopt_peer_calibration(
    controller: &PressureController,
    registry: &QuotaRegistry,
    capabilities: &NodeCapabilities,
    folded: &mut std::collections::HashMap<(u64, usize), u64>,
) -> usize {
    let node_id = controller.node_id();
    let mut peers: std::collections::HashMap<(u64, usize), (f64, u64)> =
        std::collections::HashMap::new();

    for (key, packed) in registry.snapshot() {
        let Some((fingerprint, peer, kind, field)) = parse_calibration_key(&key) else {
            continue;
        };
        // A different shape of machine measured something about a different
        // machine, and adopting it would price this node's plans against
        // hardware it is not running on
        if fingerprint != capabilities.fingerprint || peer == node_id {
            continue;
        }
        let (_, payload) = zyron_pressure::pressure::decode_versioned(packed);
        let slot = peers.entry((peer, kind.index())).or_insert((0.0, 0));
        match field {
            CalibrationField::CentiNanosPerUnit => slot.0 = decode_coefficient(payload),
            CalibrationField::SampleCount => slot.1 = payload as u64,
        }
    }

    let mut fresh = OperatorCoefficients::cold_start();
    let mut adopted = 0usize;
    for ((peer, kind_index), (value, samples)) in peers {
        if samples == 0 || value <= 0.0 {
            continue;
        }
        let already = folded.get(&(peer, kind_index)).copied().unwrap_or(0);
        let Some(new_evidence) = samples.checked_sub(already) else {
            // The peer restarted and its count went backwards. Its history is
            // gone, so this round's reading is taken whole rather than treated
            // as a negative amount of evidence
            folded.insert((peer, kind_index), samples);
            continue;
        };
        if new_evidence == 0 {
            continue;
        }
        folded.insert((peer, kind_index), samples);
        let mut one_peer = OperatorCoefficients::cold_start();
        one_peer.ns_per_unit[kind_index] = value;
        one_peer.sample_count[kind_index] = new_evidence;
        fresh.merge(&one_peer);
        adopted += 1;
    }

    if adopted > 0 {
        controller.coefficients().adopt(&fresh);
    }
    adopted
}

/// Whether this node may stop serving entirely, and what waking it will cost.
///
/// The trigger belongs to whatever decides a node is no longer needed. What
/// belongs here is the answer, built from state the node actually holds rather
/// than from an operator's belief about it: a checkpoint that closed cleanly
/// and a working-set manifest on disk.
///
/// Both are refusals rather than warnings, because the operator's experience
/// of scale to zero is entirely the resume. Without the checkpoint the node
/// replays its log. Without the manifest it serves its first minutes from an
/// empty buffer pool. Either one turns a resume that takes a second into one
/// that takes minutes, and neither announces itself.
pub fn scale_to_zero_readiness(
    controller: &PressureController,
    capabilities: &NodeCapabilities,
) -> zyron_common::Result<zyron_pressure::provisioner::ResumeEstimate> {
    zyron_pressure::provisioner::assert_scale_to_zero_ready(&scale_to_zero_inputs(
        controller,
        capabilities,
    ))
}

/// What resuming would cost, whether or not it is currently allowed.
///
/// Separate from the assertion so an operator can see the number before the
/// node is anywhere near idle, which is when the answer is worth acting on.
pub fn resume_estimate(
    controller: &PressureController,
    capabilities: &NodeCapabilities,
) -> zyron_pressure::provisioner::ResumeEstimate {
    zyron_pressure::provisioner::estimate_resume(&scale_to_zero_inputs(controller, capabilities))
}

fn scale_to_zero_inputs(
    controller: &PressureController,
    capabilities: &NodeCapabilities,
) -> zyron_pressure::provisioner::ScaleToZeroInputs {
    let hot_set = controller.hot_set_status();
    let checkpoint = controller.checkpoint_state();
    // What the mount was measured to deliver, because the reload is a
    // sequential read of the pages the manifest names and nothing else
    let mount = capabilities.mounts.first();
    let read_throughput_bytes_per_s = mount
        .map(|m| m.read_throughput_mb_s * 1_000_000.0)
        .unwrap_or(0.0);
    let page_read_p50 = mount
        .map(|m| Duration::from_nanos(m.read_p50_ns))
        .unwrap_or(Duration::ZERO);

    zyron_pressure::provisioner::ScaleToZeroInputs {
        checkpoint_age: controller.checkpoint_age(),
        checkpoint_clean: checkpoint.clean,
        wal_bytes_since_checkpoint: checkpoint.wal_bytes_since,
        hot_set_persisted: hot_set.persisted,
        hot_set_pages: hot_set.pages,
        page_bytes: zyron_common::page::PAGE_SIZE as u64,
        read_throughput_bytes_per_s,
        page_read_p50,
    }
}

/// Reads a node's pressure back out of the registry, for the mesh view.
///
/// Returns the newest reading each field carries, along with the sequence it
/// was published at, so a stale peer is visible as one whose sequence has
/// stopped moving rather than as one that looks idle.
pub fn read_pressure(
    registry: &QuotaRegistry,
    node_id: u64,
    class: WorkloadClass,
) -> Vec<(PressureField, u32, u32)> {
    let mut out = Vec::with_capacity(PressureField::ALL.len());
    for field in PressureField::ALL {
        let key = pressure_key(node_id, class, field);
        let result = registry.check(&key, u64::MAX);
        if result.used == 0 {
            continue;
        }
        let (sequence, payload) = zyron_pressure::pressure::decode_versioned(result.used);
        out.push((field, sequence, payload));
    }
    out
}

/// Every node the registry has heard pressure from.
pub fn known_pressure_nodes(registry: &QuotaRegistry) -> Vec<u64> {
    let mut nodes: Vec<u64> = registry
        .snapshot()
        .into_iter()
        .filter_map(|(key, _)| {
            zyron_pressure::pressure::parse_pressure_key(&key).map(|(node, _, _)| node)
        })
        .collect();
    nodes.sort_unstable();
    nodes.dedup();
    nodes
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use zyron_pressure::pressure::decode_versioned;

    #[test]
    fn publishing_writes_every_field_of_every_class() {
        let controller = PressureController::global();
        let registry = QuotaRegistry::new();
        let written = publish_pressure(controller, &registry);
        assert_eq!(
            written,
            WorkloadClass::COUNT * PressureField::ALL.len(),
            "a field or a class went unpublished"
        );
        let nodes = known_pressure_nodes(&registry);
        assert_eq!(nodes, vec![controller.node_id()]);
    }

    /// The transport converges by taking a maximum, so a falling reading must
    /// still overwrite a rising one. The sequence in the high half is what
    /// makes that true, and this is the test that would catch its removal.
    #[test]
    fn a_newer_reading_wins_even_when_the_value_fell() {
        let registry = QuotaRegistry::new();
        let key = pressure_key(9, WorkloadClass::Interactive, PressureField::PressureMicros);

        registry.merge_remote(&[(key.clone(), encode_versioned(1, 900_000))]);
        registry.merge_remote(&[(key.clone(), encode_versioned(2, 25))]);

        let stored = registry.check(&key, u64::MAX).used;
        let (sequence, payload) = decode_versioned(stored);
        assert_eq!(sequence, 2, "the older tick survived");
        assert_eq!(payload, 25, "the stale reading survived");
    }

    #[test]
    fn an_older_reading_never_overwrites_a_newer_one() {
        let registry = QuotaRegistry::new();
        let key = pressure_key(9, WorkloadClass::Bulk, PressureField::InFlight);
        registry.merge_remote(&[(key.clone(), encode_versioned(10, 5))]);
        // A frame that arrived late out of order
        registry.merge_remote(&[(key.clone(), encode_versioned(3, 999))]);
        let (sequence, payload) = decode_versioned(registry.check(&key, u64::MAX).used);
        assert_eq!(sequence, 10);
        assert_eq!(payload, 5);
    }

    #[test]
    fn reading_back_recovers_what_was_published() {
        let controller = PressureController::global();
        let registry = QuotaRegistry::new();
        publish_pressure(controller, &registry);
        let rows = read_pressure(&registry, controller.node_id(), WorkloadClass::Interactive);
        // Fields whose payload is zero are indistinguishable from absent in a
        // max-merged registry, so this asserts the ones that always carry a
        // value rather than requiring all eight
        assert!(
            rows.iter().any(|(f, _, _)| *f == PressureField::Ceiling),
            "the ceiling did not survive the round trip"
        );
    }

    #[test]
    fn quota_keys_sharing_the_transport_are_not_read_as_pressure() {
        let registry = QuotaRegistry::new();
        registry
            .increment("tenant/acme/rows", 500, u64::MAX)
            .expect("quota");
        assert!(
            known_pressure_nodes(&registry).is_empty(),
            "a quota key was mistaken for a node"
        );
    }

    #[test]
    fn measured_latency_replaces_what_the_os_claimed_the_device_was() {
        let controller = PressureController::global();
        let mut capabilities = NodeCapabilities::probe(1, None);
        capabilities.mounts = vec![zyron_pressure::capability::MountCapability::unmeasured(
            "/data".to_string(),
            0,
            0,
        )];
        for _ in 0..256 {
            controller.record_page_read(Duration::from_micros(20), 8192, 1);
        }
        refresh_measured_capabilities(controller, &mut capabilities);
        let mount = &capabilities.mounts[0];
        assert!(mount.sample_count >= 256);
        assert_eq!(
            mount.device_kind,
            zyron_pressure::capability::DeviceKind::Nvme,
            "20us reads were not classified from the measurement"
        );
        assert!(mount.read_p50_ns > 0);
    }

    /// The whole point of persisting: a node that has run on this hardware
    /// before starts from what it measured, not from an assumption, and does
    /// so without running a probe.
    #[test]
    fn a_known_fingerprint_is_inherited_at_startup() {
        let dir = tempfile::tempdir().expect("tempdir");
        let mut capabilities = NodeCapabilities::probe(1, None);

        // Nothing on disk yet, so there is nothing to inherit
        assert!(!FileCalibrationStore::adopt_into(
            dir.path(),
            &mut capabilities
        ));
        assert!(!capabilities.inherited);

        // A previous run of this hardware shape measured a sort at 12.5ns
        let mut measured = capabilities.coefficients.clone();
        measured.ns_per_unit[zyron_pressure::capability::OperatorKind::Sort.index()] = 12.5;
        measured.sample_count[zyron_pressure::capability::OperatorKind::Sort.index()] = 250_000;
        let mut written = capabilities.clone();
        written.coefficients = measured;
        FileCalibrationStore::new(dir.path(), 1)
            .persist(&written)
            .expect("persist");

        let mut fresh = NodeCapabilities::probe(2, None);
        assert_eq!(
            fresh.fingerprint, capabilities.fingerprint,
            "the same machine must fingerprint the same"
        );
        let started = Instant::now();
        assert!(FileCalibrationStore::adopt_into(dir.path(), &mut fresh));
        let elapsed = started.elapsed();

        assert!(fresh.inherited);
        assert!(
            (fresh
                .coefficients
                .get(zyron_pressure::capability::OperatorKind::Sort)
                - 12.5)
                .abs()
                < 1e-9,
            "the measured value was not adopted"
        );
        // Reading a file, not running a benchmark
        assert!(
            elapsed < Duration::from_millis(200),
            "adopting took {elapsed:?}, which is a probe rather than a read"
        );
    }

    /// A shape nobody has measured must still serve, on cold start values,
    /// rather than blocking on a probe.
    #[test]
    fn an_unknown_fingerprint_starts_immediately_on_cold_values() {
        let dir = tempfile::tempdir().expect("tempdir");
        let mut capabilities = NodeCapabilities::probe(1, None);
        capabilities.fingerprint = zyron_pressure::capability::HardwareFingerprint(0xDEADBEEF);
        let started = Instant::now();
        let inherited = FileCalibrationStore::adopt_into(dir.path(), &mut capabilities);
        assert!(!inherited);
        assert!(started.elapsed() < Duration::from_millis(200));
        // Cold start values, not zeroes, so nothing prices as free
        for kind in zyron_pressure::capability::OperatorKind::ALL {
            assert!(capabilities.coefficients.get(kind) > 0.0);
        }
    }

    #[test]
    fn the_worker_ticks_and_stops_cleanly() {
        let capabilities = NodeCapabilities::probe(1, None);
        let registry = Arc::new(QuotaRegistry::new());
        let mut worker = PressureWorker::start(PressureInputs {
            capabilities,
            quota_registry: Some(Arc::clone(&registry)),
            store: None,
            // No pool and a scratch directory, so the loop is exercised
            // without a manifest write reaching anything the test owns
            buffer_pool: None,
            data_dir: std::env::temp_dir().join("zyron_pressure_worker_test"),
            mesh: MeshSection::default(),
        });
        // Two intervals is enough for at least one tick to have landed
        thread::sleep(TICK_INTERVAL * 3);
        let ticks = worker.stats().ticks.load(Ordering::Relaxed);
        worker.stop();
        assert!(ticks >= 1, "the worker never ticked");
        assert!(!known_pressure_nodes(&registry).is_empty());
    }
}
