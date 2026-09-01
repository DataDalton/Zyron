//! Node capability probe and the calibration it feeds.
//!
//! Constants measured on one machine are wrong on the next. A four core cloud
//! VM on network storage and a 128 core box on local NVMe differ by two orders
//! of magnitude on the numbers that decide batch sizes, prefetch depth and
//! concurrency, so the node measures itself instead of being told.
//!
//! Measurement is passive: the coefficients come from the real work the node
//! is already doing, recorded as (rows, elapsed) pairs on the operator hot
//! paths. Active probing exists only for an operator kind that has not run
//! often enough to have an opinion, and never at startup, because a node that
//! benchmarks its disk before accepting a connection has no fast cold start.
//!
//! Calibration is keyed by a hardware fingerprint and persisted, so a node
//! that has seen this shape of machine before starts with what it learned and
//! a mesh becomes a calibration cache for its own fleet.

use std::sync::atomic::{AtomicU32, AtomicU64, AtomicUsize, Ordering};

use serde::{Deserialize, Serialize};

use zyron_common::checksum::hash64;

// ---------------------------------------------------------------------------
// Fingerprint
// ---------------------------------------------------------------------------

/// Identifies a hardware shape closely enough that calibration measured on one
/// machine transfers to another, and no more closely than that.
///
/// Two nodes of the same cloud instance type hash the same and share what they
/// learn. A different core count, a different memory size, or a different
/// storage device gives a different hash, because those are the differences
/// that move the coefficients.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub struct HardwareFingerprint(pub u64);

impl HardwareFingerprint {
    /// Renders as the 16 hex digits used in the catalog and in views.
    pub fn to_hex(self) -> String {
        format!("{:016x}", self.0)
    }

    /// Parses the hex form. None on anything else.
    pub fn parse_hex(s: &str) -> Option<Self> {
        u64::from_str_radix(s.trim(), 16).ok().map(Self)
    }
}

impl std::fmt::Display for HardwareFingerprint {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:016x}", self.0)
    }
}

/// The inputs the fingerprint is computed from, kept so a view can explain
/// why two nodes did or did not match.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct FingerprintInputs {
    pub cpu_model: String,
    pub core_count: u32,
    /// Rounded to whole gigabytes, so a few megabytes of reporting drift
    /// between two identical machines does not split them apart
    pub ram_gb: u32,
    pub numa_nodes: u32,
    pub simd_level: SimdLevel,
    /// Mount path and the device kind behind it, sorted by path
    pub mounts: Vec<(String, DeviceKind)>,
}

impl FingerprintInputs {
    /// Hashes the inputs in a fixed order, so the same machine always produces
    /// the same value across restarts and across processes.
    pub fn fingerprint(&self) -> HardwareFingerprint {
        let mut buf = Vec::with_capacity(128);
        buf.extend_from_slice(self.cpu_model.as_bytes());
        buf.push(0);
        buf.extend_from_slice(&self.core_count.to_le_bytes());
        buf.extend_from_slice(&self.ram_gb.to_le_bytes());
        buf.extend_from_slice(&self.numa_nodes.to_le_bytes());
        buf.push(self.simd_level as u8);
        let mut mounts = self.mounts.clone();
        mounts.sort();
        for (path, kind) in &mounts {
            buf.extend_from_slice(path.as_bytes());
            buf.push(b'=');
            buf.push(kind.code());
            buf.push(0);
        }
        HardwareFingerprint(hash64(&buf))
    }
}

/// Widest vector unit the build can actually use on this CPU.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SimdLevel {
    Scalar = 0,
    Sse2 = 1,
    Avx2 = 2,
    Avx512 = 3,
    Neon = 4,
}

impl SimdLevel {
    pub const fn as_str(self) -> &'static str {
        match self {
            SimdLevel::Scalar => "scalar",
            SimdLevel::Sse2 => "sse2",
            SimdLevel::Avx2 => "avx2",
            SimdLevel::Avx512 => "avx512",
            SimdLevel::Neon => "neon",
        }
    }

    /// Widest unit this process can issue, decided at runtime rather than at
    /// build time so a binary built for a baseline still reports the truth.
    pub fn detect() -> Self {
        #[cfg(target_arch = "x86_64")]
        {
            if std::arch::is_x86_feature_detected!("avx512f") {
                return SimdLevel::Avx512;
            }
            if std::arch::is_x86_feature_detected!("avx2") {
                return SimdLevel::Avx2;
            }
            return SimdLevel::Sse2;
        }
        #[cfg(target_arch = "aarch64")]
        {
            return SimdLevel::Neon;
        }
        #[allow(unreachable_code)]
        SimdLevel::Scalar
    }
}

/// What kind of device is behind a mount. The read latency curve differs by
/// two orders of magnitude across these, which is why prefetch depth and
/// batch size cannot be constants.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DeviceKind {
    Unknown,
    Hdd,
    Ssd,
    Nvme,
    /// Network attached, whether a SAN, an NFS mount or a cloud volume
    Network,
}

impl DeviceKind {
    pub const fn as_str(self) -> &'static str {
        match self {
            DeviceKind::Unknown => "unknown",
            DeviceKind::Hdd => "hdd",
            DeviceKind::Ssd => "ssd",
            DeviceKind::Nvme => "nvme",
            DeviceKind::Network => "network",
        }
    }

    pub const fn code(self) -> u8 {
        match self {
            DeviceKind::Unknown => b'u',
            DeviceKind::Hdd => b'h',
            DeviceKind::Ssd => b's',
            DeviceKind::Nvme => b'n',
            DeviceKind::Network => b'w',
        }
    }

    /// Classification from a measured 4K read latency, used once passive
    /// samples exist. The boundaries are wide because the point is to pick a
    /// prefetch strategy, not to name the product.
    pub fn from_read_latency_ns(p50_ns: u64) -> Self {
        match p50_ns {
            0 => DeviceKind::Unknown,
            n if n < 50_000 => DeviceKind::Nvme,
            n if n < 500_000 => DeviceKind::Ssd,
            n if n < 3_000_000 => DeviceKind::Network,
            _ => DeviceKind::Hdd,
        }
    }
}

// ---------------------------------------------------------------------------
// Operator coefficients
// ---------------------------------------------------------------------------

/// The work units the cost model prices. Each one is measured separately
/// because their per-row costs differ by more than an order of magnitude and a
/// single blended number would misprice every plan that is not average.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum OperatorKind {
    SeqScan,
    IndexScan,
    Filter,
    Project,
    HashJoinBuild,
    HashJoinProbe,
    Sort,
    Aggregate,
    Window,
    SetOp,
    LakeScan,
    /// Per page rather than per row
    PageRead,
    /// Per page rather than per row
    WalFsync,
}

impl OperatorKind {
    pub const ALL: [OperatorKind; 13] = [
        OperatorKind::SeqScan,
        OperatorKind::IndexScan,
        OperatorKind::Filter,
        OperatorKind::Project,
        OperatorKind::HashJoinBuild,
        OperatorKind::HashJoinProbe,
        OperatorKind::Sort,
        OperatorKind::Aggregate,
        OperatorKind::Window,
        OperatorKind::SetOp,
        OperatorKind::LakeScan,
        OperatorKind::PageRead,
        OperatorKind::WalFsync,
    ];

    pub const COUNT: usize = 13;

    pub const fn index(self) -> usize {
        self as usize
    }

    pub const fn from_index(i: usize) -> Option<Self> {
        if i < Self::COUNT {
            Some(Self::ALL[i])
        } else {
            None
        }
    }

    pub const fn as_str(self) -> &'static str {
        match self {
            OperatorKind::SeqScan => "seq_scan",
            OperatorKind::IndexScan => "index_scan",
            OperatorKind::Filter => "filter",
            OperatorKind::Project => "project",
            OperatorKind::HashJoinBuild => "hash_join_build",
            OperatorKind::HashJoinProbe => "hash_join_probe",
            OperatorKind::Sort => "sort",
            OperatorKind::Aggregate => "aggregate",
            OperatorKind::Window => "window",
            OperatorKind::SetOp => "set_op",
            OperatorKind::LakeScan => "lake_scan",
            OperatorKind::PageRead => "page_read",
            OperatorKind::WalFsync => "wal_fsync",
        }
    }

    pub fn parse(s: &str) -> Option<Self> {
        Self::ALL.into_iter().find(|k| k.as_str() == s)
    }

    /// Whether the unit is a page rather than a row, which the views label
    /// and the cost model multiplies differently.
    pub const fn unit_is_page(self) -> bool {
        matches!(self, OperatorKind::PageRead | OperatorKind::WalFsync)
    }

    /// The value used before this kind has been measured on this hardware.
    ///
    /// These are starting points that get replaced by measurement within
    /// seconds of real traffic, not tuned constants. They are deliberately
    /// pessimistic: over-estimating cost makes the first queries look
    /// expensive, which errs toward admitting fewer rather than overcommitting
    /// a machine nobody has measured yet.
    pub const fn cold_start_ns_per_unit(self) -> u64 {
        match self {
            OperatorKind::SeqScan => 40,
            OperatorKind::IndexScan => 200,
            OperatorKind::Filter => 10,
            OperatorKind::Project => 8,
            OperatorKind::HashJoinBuild => 60,
            OperatorKind::HashJoinProbe => 45,
            OperatorKind::Sort => 90,
            OperatorKind::Aggregate => 55,
            OperatorKind::Window => 120,
            OperatorKind::SetOp => 70,
            OperatorKind::LakeScan => 35,
            OperatorKind::PageRead => 120_000,
            OperatorKind::WalFsync => 200_000,
        }
    }
}

/// Measured cost per unit for every operator kind, plus how much evidence sits
/// behind each number.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct OperatorCoefficients {
    /// Nanoseconds per row, or per page for the page-unit kinds
    pub ns_per_unit: [f64; OperatorKind::COUNT],
    /// Units observed per kind. A coefficient with few samples behind it is
    /// still a guess, and the merge uses this to decide which side wins
    pub sample_count: [u64; OperatorKind::COUNT],
}

impl Default for OperatorCoefficients {
    fn default() -> Self {
        Self::cold_start()
    }
}

impl OperatorCoefficients {
    /// Starting values, with zero evidence recorded so any measurement at all
    /// outweighs them.
    pub fn cold_start() -> Self {
        let mut ns_per_unit = [0.0f64; OperatorKind::COUNT];
        for kind in OperatorKind::ALL {
            ns_per_unit[kind.index()] = kind.cold_start_ns_per_unit() as f64;
        }
        Self {
            ns_per_unit,
            sample_count: [0; OperatorKind::COUNT],
        }
    }

    pub fn get(&self, kind: OperatorKind) -> f64 {
        self.ns_per_unit[kind.index()]
    }

    pub fn samples(&self, kind: OperatorKind) -> u64 {
        self.sample_count[kind.index()]
    }

    /// Whether a kind has enough evidence that active probing would add
    /// nothing.
    pub fn is_measured(&self, kind: OperatorKind) -> bool {
        self.sample_count[kind.index()] >= MIN_SAMPLES_FOR_CONFIDENCE
    }

    /// Folds another node's measurements in, weighting each side by the
    /// evidence behind it. Commutative and idempotent enough for gossip: two
    /// nodes that merge each other converge on the same weighted mean.
    pub fn merge(&mut self, other: &OperatorCoefficients) {
        for kind in OperatorKind::ALL {
            let i = kind.index();
            let mine = self.sample_count[i];
            let theirs = other.sample_count[i];
            if theirs == 0 {
                continue;
            }
            if mine == 0 {
                self.ns_per_unit[i] = other.ns_per_unit[i];
                self.sample_count[i] = theirs;
                continue;
            }
            let total = mine.saturating_add(theirs);
            let weighted = (self.ns_per_unit[i] * mine as f64
                + other.ns_per_unit[i] * theirs as f64)
                / total as f64;
            self.ns_per_unit[i] = weighted;
            self.sample_count[i] = total;
        }
    }

    /// Estimated seconds to push `units` through an operator of this kind.
    pub fn estimate_seconds(&self, kind: OperatorKind, units: f64) -> f64 {
        if !units.is_finite() || units <= 0.0 {
            return 0.0;
        }
        units * self.get(kind) / 1e9
    }
}

/// Sample count past which a coefficient is treated as measured rather than
/// assumed. Below it the passive collector keeps folding new evidence in at
/// full weight, above it the value has stabilised.
pub const MIN_SAMPLES_FOR_CONFIDENCE: u64 = 100;

// ---------------------------------------------------------------------------
// Live passive collector
// ---------------------------------------------------------------------------

/// Accumulates (units, elapsed) pairs from the operator hot paths.
///
/// Written on every batch, so the write has to be two relaxed atomic adds and
/// nothing else. The exponential decay that keeps the number current lives in
/// the drain, which runs on the collector's own cadence rather than in the
/// query's path.
#[derive(Debug)]
pub struct CoefficientAccumulator {
    units: [AtomicU64; OperatorKind::COUNT],
    nanos: [AtomicU64; OperatorKind::COUNT],
    /// Live value, nanoseconds per unit scaled by 1000 so it fits an integer
    /// without losing sub-nanosecond resolution on the cheap operators
    current_milli_ns: [AtomicU64; OperatorKind::COUNT],
    total_samples: [AtomicU64; OperatorKind::COUNT],
    /// Units seen in each of the last few drains, so a kind that used to be
    /// well measured and has since gone quiet is visible as quiet. Total
    /// evidence cannot say that: it only ever grows
    recent_units: [[AtomicU64; RECENT_BUCKETS]; OperatorKind::COUNT],
    recent_cursor: AtomicUsize,
    /// What other nodes of this hardware shape have measured, kept apart from
    /// what this node measured itself.
    ///
    /// The separation is what stops evidence echoing around a mesh. If a node
    /// published the value it had adopted, its peers would adopt that back as
    /// though it were new evidence, and after a few rounds a single
    /// measurement would look like a hundred. What is published is only ever
    /// what this node saw with its own operators
    fleet_milli_ns: [AtomicU64; OperatorKind::COUNT],
    fleet_samples: [AtomicU64; OperatorKind::COUNT],
}

/// Drains the recent window spans.
///
/// The drain runs every thirty seconds and the observation window is five
/// minutes, so ten buckets hold exactly that window with no arithmetic on
/// timestamps and no clock read on the write path.
const RECENT_BUCKETS: usize = 10;

impl Default for CoefficientAccumulator {
    fn default() -> Self {
        Self::new()
    }
}

impl CoefficientAccumulator {
    pub fn new() -> Self {
        let accumulator = Self {
            units: std::array::from_fn(|_| AtomicU64::new(0)),
            nanos: std::array::from_fn(|_| AtomicU64::new(0)),
            current_milli_ns: std::array::from_fn(|_| AtomicU64::new(0)),
            total_samples: std::array::from_fn(|_| AtomicU64::new(0)),
            recent_units: std::array::from_fn(|_| std::array::from_fn(|_| AtomicU64::new(0))),
            recent_cursor: AtomicUsize::new(0),
            fleet_milli_ns: std::array::from_fn(|_| AtomicU64::new(0)),
            fleet_samples: std::array::from_fn(|_| AtomicU64::new(0)),
        };
        for kind in OperatorKind::ALL {
            accumulator.current_milli_ns[kind.index()].store(
                kind.cold_start_ns_per_unit().saturating_mul(1000),
                Ordering::Relaxed,
            );
        }
        accumulator
    }

    /// Seeds the live values from a persisted or inherited calibration, so a
    /// node that found its fingerprint in the cache starts calibrated.
    pub fn seed(&self, coefficients: &OperatorCoefficients) {
        for kind in OperatorKind::ALL {
            let i = kind.index();
            let scaled = (coefficients.ns_per_unit[i] * 1000.0).round();
            let scaled = if scaled.is_finite() && scaled >= 0.0 {
                scaled as u64
            } else {
                kind.cold_start_ns_per_unit().saturating_mul(1000)
            };
            self.current_milli_ns[i].store(scaled, Ordering::Relaxed);
            self.total_samples[i].store(coefficients.sample_count[i], Ordering::Relaxed);
        }
    }

    /// Records one batch measured as a span rather than a nanosecond count.
    ///
    /// The clamp lives here rather than at each caller so a span longer than
    /// the counter can hold saturates in one place instead of wrapping in
    /// whichever caller forgot it.
    #[inline]
    pub fn record_elapsed(&self, kind: OperatorKind, units: u64, elapsed: std::time::Duration) {
        self.record(kind, units, elapsed.as_nanos().min(u64::MAX as u128) as u64);
    }

    /// Records one batch of real work. Two relaxed adds, called from inside
    /// operators, so it must stay this cheap.
    #[inline]
    pub fn record(&self, kind: OperatorKind, units: u64, elapsed_nanos: u64) {
        if units == 0 {
            return;
        }
        let i = kind.index();
        self.units[i].fetch_add(units, Ordering::Relaxed);
        self.nanos[i].fetch_add(elapsed_nanos, Ordering::Relaxed);
    }

    /// Folds what has accumulated since the last call into the live values and
    /// clears the accumulator.
    ///
    /// New evidence is blended rather than substituted, at a weight that falls
    /// as the sample count rises: an unmeasured kind moves straight to what
    /// was just observed, a well measured one moves a fraction of the way. The
    /// floor on that fraction is what lets the value follow a device that is
    /// aging or a SAN that got busy instead of freezing on the first reading.
    pub fn drain(&self) -> DrainSummary {
        let mut updated = 0usize;
        let bucket = self.recent_cursor.load(Ordering::Relaxed) % RECENT_BUCKETS;
        for kind in OperatorKind::ALL {
            let i = kind.index();
            let units = self.units[i].swap(0, Ordering::Relaxed);
            // Written for every kind, including the ones that saw nothing.
            // Skipping the quiet ones would leave a stale count in the bucket
            // and a kind that stopped being exercised would keep reporting the
            // traffic it had five minutes ago
            self.recent_units[i][bucket].store(units, Ordering::Relaxed);
            if units == 0 {
                continue;
            }
            let nanos = self.nanos[i].swap(0, Ordering::Relaxed);
            let observed_milli_ns = ((nanos as f64 / units as f64) * 1000.0) as u64;
            let prior_samples = self.total_samples[i].fetch_add(units, Ordering::Relaxed);
            let current = self.current_milli_ns[i].load(Ordering::Relaxed);
            let next = if prior_samples == 0 {
                observed_milli_ns
            } else {
                let weight = blend_weight(prior_samples);
                (current as f64 * (1.0 - weight) + observed_milli_ns as f64 * weight) as u64
            };
            self.current_milli_ns[i].store(next, Ordering::Relaxed);
            updated += 1;
        }
        self.recent_cursor.store(bucket + 1, Ordering::Relaxed);
        DrainSummary {
            kinds_updated: updated,
        }
    }

    /// Nanoseconds per unit for one kind, as currently believed.
    pub fn get(&self, kind: OperatorKind) -> f64 {
        self.current_milli_ns[kind.index()].load(Ordering::Relaxed) as f64 / 1000.0
    }

    pub fn samples(&self, kind: OperatorKind) -> u64 {
        self.total_samples[kind.index()].load(Ordering::Relaxed)
    }

    /// Kinds no amount of traffic has yet measured well enough to trust.
    ///
    /// Counts every sample the node has ever taken, including what it
    /// inherited, so a node that started calibrated reports nothing here.
    pub fn undersampled(&self) -> Vec<OperatorKind> {
        OperatorKind::ALL
            .into_iter()
            .filter(|k| self.samples(*k) < MIN_SAMPLES_FOR_CONFIDENCE)
            .collect()
    }

    /// Units a kind has been measured over in the recent window.
    ///
    /// Only complete drains are counted, so what is accumulating right now is
    /// not included. That makes the answer lag by up to one drain, which is
    /// the correct direction to be wrong in: a kind is never declared busy on
    /// the strength of samples that have not been folded in yet.
    pub fn recent_samples(&self, kind: OperatorKind) -> u64 {
        let i = kind.index();
        self.recent_units[i]
            .iter()
            .map(|b| b.load(Ordering::Relaxed))
            .fold(0u64, |acc, v| acc.saturating_add(v))
    }

    /// Kinds real traffic has not exercised lately, which is the only thing
    /// that justifies spending time on an active probe.
    ///
    /// Recency rather than lifetime evidence, because the value being defended
    /// is that the coefficient describes this machine now. A kind measured
    /// exhaustively last week on a volume that has since been migrated is
    /// exactly the case a lifetime count calls settled and a recency count
    /// calls stale.
    pub fn probe_candidates(&self) -> Vec<OperatorKind> {
        OperatorKind::ALL
            .into_iter()
            .filter(|k| self.recent_samples(*k) < MIN_SAMPLES_FOR_CONFIDENCE)
            .collect()
    }

    /// What this node measured itself.
    ///
    /// The copy that is published to peers and persisted, because it is the
    /// only part this node can vouch for.
    pub fn snapshot(&self) -> OperatorCoefficients {
        let mut out = OperatorCoefficients::cold_start();
        for kind in OperatorKind::ALL {
            let i = kind.index();
            out.ns_per_unit[i] = self.get(kind);
            out.sample_count[i] = self.samples(kind);
        }
        out
    }

    /// Folds a peer's evidence into the fleet view.
    ///
    /// Weighted by how much traffic stands behind each side, so a node that
    /// measured a kind over millions of rows outweighs one that saw it twice.
    /// Never touches the local values, which is what keeps what this node
    /// publishes to what this node observed.
    pub fn adopt(&self, contribution: &OperatorCoefficients) {
        for kind in OperatorKind::ALL {
            let i = kind.index();
            let theirs = contribution.sample_count[i];
            if theirs == 0 {
                continue;
            }
            let mine = self.fleet_samples[i].load(Ordering::Relaxed);
            let their_value = contribution.ns_per_unit[i];
            let next = if mine == 0 {
                their_value
            } else {
                let my_value = self.fleet_milli_ns[i].load(Ordering::Relaxed) as f64 / 1000.0;
                let total = mine.saturating_add(theirs);
                (my_value * mine as f64 + their_value * theirs as f64) / total as f64
            };
            self.fleet_milli_ns[i].store((next * 1000.0) as u64, Ordering::Relaxed);
            self.fleet_samples[i].store(mine.saturating_add(theirs), Ordering::Relaxed);
        }
    }

    /// What the fleet has measured, without this node's own contribution.
    pub fn fleet(&self) -> OperatorCoefficients {
        let mut out = OperatorCoefficients::cold_start();
        for kind in OperatorKind::ALL {
            let i = kind.index();
            let samples = self.fleet_samples[i].load(Ordering::Relaxed);
            if samples == 0 {
                continue;
            }
            out.ns_per_unit[i] = self.fleet_milli_ns[i].load(Ordering::Relaxed) as f64 / 1000.0;
            out.sample_count[i] = samples;
        }
        out
    }

    /// What the node should price a plan against: what it measured, weighted
    /// against what the rest of its hardware shape measured.
    ///
    /// This is what makes a freshly added node useful immediately. With no
    /// local evidence the answer is entirely the fleet's, and as the node
    /// serves traffic its own measurement takes over.
    pub fn effective(&self) -> OperatorCoefficients {
        let mut out = self.snapshot();
        out.merge(&self.fleet());
        out
    }
}

// ---------------------------------------------------------------------------
// Gossip keys
// ---------------------------------------------------------------------------

/// Prefix on every calibration record in the shared registry.
pub const CALIBRATION_KEY_PREFIX: &str = "zcalib";

/// Scale applied to a coefficient before it is packed into a gossip payload.
///
/// The payload is thirty-two bits. Nanoseconds per unit is a fraction for the
/// cheap operators and tens of thousands for a page read, so it is carried in
/// hundredths of a nanosecond: that keeps two decimal places on a
/// sub-nanosecond filter and still holds a forty-millisecond fsync without
/// saturating.
pub const COEFFICIENT_SCALE: f64 = 100.0;

/// The two halves of a calibration record.
///
/// Both are needed because the value alone cannot be merged. Two nodes
/// measuring the same hardware disagree slightly, and the right answer is the
/// average weighted by how much traffic each one saw, which is what the sample
/// count carries.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum CalibrationField {
    /// Hundredths of a nanosecond per unit
    CentiNanosPerUnit,
    /// Batches this node has measured the kind over
    SampleCount,
}

impl CalibrationField {
    pub const ALL: [CalibrationField; 2] = [
        CalibrationField::CentiNanosPerUnit,
        CalibrationField::SampleCount,
    ];

    pub const fn as_str(self) -> &'static str {
        match self {
            CalibrationField::CentiNanosPerUnit => "v",
            CalibrationField::SampleCount => "n",
        }
    }

    pub fn parse(s: &str) -> Option<Self> {
        match s {
            "v" => Some(CalibrationField::CentiNanosPerUnit),
            "n" => Some(CalibrationField::SampleCount),
            _ => None,
        }
    }
}

/// The registry key one node's measurement of one operator kind lives under.
///
/// Keyed by hardware shape first and by node second. The shape is what makes
/// the measurement transferable, and the node is what keeps two publishers
/// from overwriting each other: a merge that took the maximum of a shared key
/// would keep whichever node had ticked more, which says nothing about which
/// measurement is better.
pub fn calibration_key(
    fingerprint: HardwareFingerprint,
    node_id: u64,
    kind: OperatorKind,
    field: CalibrationField,
) -> String {
    format!(
        "{CALIBRATION_KEY_PREFIX}:{}:{node_id}:{}:{}",
        fingerprint.to_hex(),
        kind.index(),
        field.as_str()
    )
}

/// Reads a calibration key back. None when the key belongs to something else.
pub fn parse_calibration_key(
    key: &str,
) -> Option<(HardwareFingerprint, u64, OperatorKind, CalibrationField)> {
    let mut parts = key.split(':');
    if parts.next()? != CALIBRATION_KEY_PREFIX {
        return None;
    }
    let fingerprint = HardwareFingerprint(u64::from_str_radix(parts.next()?, 16).ok()?);
    let node_id = parts.next()?.parse::<u64>().ok()?;
    let kind = OperatorKind::from_index(parts.next()?.parse::<usize>().ok()?)?;
    let field = CalibrationField::parse(parts.next()?)?;
    if parts.next().is_some() {
        return None;
    }
    Some((fingerprint, node_id, kind, field))
}

/// Packs a coefficient for the gossip payload, saturating rather than wrapping.
pub fn encode_coefficient(ns_per_unit: f64) -> u32 {
    let scaled = ns_per_unit * COEFFICIENT_SCALE;
    if !scaled.is_finite() || scaled <= 0.0 {
        return 0;
    }
    scaled.min(u32::MAX as f64) as u32
}

/// Unpacks a coefficient from a gossip payload.
pub fn decode_coefficient(payload: u32) -> f64 {
    payload as f64 / COEFFICIENT_SCALE
}

/// What one drain pass changed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DrainSummary {
    pub kinds_updated: usize,
}

/// How much of a new observation to take, given how much evidence already
/// stands behind the current value.
///
/// Falls as evidence accumulates so the value settles, and floors well above
/// zero so it never stops tracking. A device that slows down after a year must
/// still move the number.
fn blend_weight(prior_samples: u64) -> f64 {
    const FLOOR: f64 = 0.02;
    let weight = 1.0 / (1.0 + prior_samples as f64 / MIN_SAMPLES_FOR_CONFIDENCE as f64);
    weight.max(FLOOR)
}

// ---------------------------------------------------------------------------
// Mount and node capabilities
// ---------------------------------------------------------------------------

/// Measured behaviour of one storage mount.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MountCapability {
    pub path: String,
    pub device_kind: DeviceKind,
    pub total_bytes: u64,
    pub available_bytes: u64,
    /// Measured 4K read latency. Zero until passive samples arrive
    pub read_p50_ns: u64,
    pub read_p99_ns: u64,
    pub write_p50_ns: u64,
    pub write_p99_ns: u64,
    /// Sustained read throughput in megabytes per second, measured
    pub read_throughput_mb_s: f64,
    pub sample_count: u64,
}

impl MountCapability {
    pub fn unmeasured(path: String, total_bytes: u64, available_bytes: u64) -> Self {
        Self {
            path,
            device_kind: DeviceKind::Unknown,
            total_bytes,
            available_bytes,
            read_p50_ns: 0,
            read_p99_ns: 0,
            write_p50_ns: 0,
            write_p99_ns: 0,
            read_throughput_mb_s: 0.0,
            sample_count: 0,
        }
    }

    /// Whether enough reads have been seen to classify the device.
    pub fn is_measured(&self) -> bool {
        self.sample_count >= MIN_SAMPLES_FOR_CONFIDENCE
    }
}

/// Everything the node knows about the machine it is running on.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct NodeCapabilities {
    pub node_id: u64,
    pub fingerprint: HardwareFingerprint,
    pub inputs: FingerprintInputs,
    pub probed_at_us: i64,
    pub core_count: u32,
    pub mem_total_bytes: u64,
    pub mem_available_bytes: u64,
    pub numa_nodes: u32,
    pub simd_level: SimdLevel,
    pub mounts: Vec<MountCapability>,
    pub coefficients: OperatorCoefficients,
    /// True when the fingerprint was found in the calibration cache and the
    /// coefficients came from there rather than from cold start values
    pub inherited: bool,
}

impl NodeCapabilities {
    /// Reads the machine. Cheap: counts, sizes and feature bits, no timing
    /// loop and no file written, so this is safe to call before the first
    /// connection is accepted.
    pub fn probe(node_id: u64, data_dir: Option<&std::path::Path>) -> Self {
        let core_count = std::thread::available_parallelism()
            .map(|n| n.get() as u32)
            .unwrap_or(1);

        let mut system = sysinfo::System::new();
        system.refresh_memory();
        let cpus = sysinfo::System::new_all();
        let cpu_model = cpus
            .cpus()
            .first()
            .map(|c| c.brand().trim().to_string())
            .filter(|s| !s.is_empty())
            .unwrap_or_else(|| "unknown".to_string());
        let mem_total_bytes = system.total_memory();
        let mem_available_bytes = system.available_memory();

        let mounts = probe_mounts(data_dir);
        let numa_nodes = probe_numa_nodes(core_count);
        let simd_level = SimdLevel::detect();

        let inputs = FingerprintInputs {
            cpu_model,
            core_count,
            ram_gb: (mem_total_bytes / (1024 * 1024 * 1024)) as u32,
            numa_nodes,
            simd_level,
            mounts: mounts
                .iter()
                .map(|m| (m.path.clone(), m.device_kind))
                .collect(),
        };
        let fingerprint = inputs.fingerprint();

        Self {
            node_id,
            fingerprint,
            inputs,
            probed_at_us: now_micros(),
            core_count,
            mem_total_bytes,
            mem_available_bytes,
            numa_nodes,
            simd_level,
            mounts,
            coefficients: OperatorCoefficients::cold_start(),
            inherited: false,
        }
    }

    /// Replaces the cold start coefficients with ones learned on this hardware
    /// shape, whether they came from this node's own history or from a peer
    /// that already measured the same fingerprint.
    pub fn adopt(&mut self, coefficients: OperatorCoefficients) {
        self.coefficients = coefficients;
        self.inherited = true;
    }

    /// The mount the data directory sits on, which is the one whose latency
    /// decides page read cost.
    pub fn primary_mount(&self) -> Option<&MountCapability> {
        self.mounts.first()
    }
}

/// Lists the mounts worth reporting, with the data directory's mount first so
/// it is the one the cost model reads.
fn probe_mounts(data_dir: Option<&std::path::Path>) -> Vec<MountCapability> {
    let disks = sysinfo::Disks::new_with_refreshed_list();
    let mut out: Vec<MountCapability> = disks
        .list()
        .iter()
        .map(|d| {
            let path = d.mount_point().to_string_lossy().to_string();
            let mut cap = MountCapability::unmeasured(path, d.total_space(), d.available_space());
            // The kind reported here is a starting guess from what the OS
            // says. Measured read latency replaces it once samples arrive,
            // because the OS answer says nothing about a network volume
            // presented as a local disk
            cap.device_kind = if d.is_removable() {
                DeviceKind::Unknown
            } else {
                match d.kind() {
                    sysinfo::DiskKind::SSD => DeviceKind::Ssd,
                    sysinfo::DiskKind::HDD => DeviceKind::Hdd,
                    _ => DeviceKind::Unknown,
                }
            };
            cap
        })
        .collect();

    out.sort_by(|a, b| a.path.cmp(&b.path));

    // Put the mount holding the data directory first: it is the one whose
    // latency the page reads actually pay
    if let Some(dir) = data_dir {
        let dir_str = dir.to_string_lossy().to_string();
        if let Some(best) = out
            .iter()
            .enumerate()
            .filter(|(_, m)| dir_str.starts_with(&m.path))
            .max_by_key(|(_, m)| m.path.len())
            .map(|(i, _)| i)
        {
            out.swap(0, best);
        }
    }
    out
}

/// NUMA node count. Linux exposes it directly; elsewhere a single node is
/// reported, which is right for every machine small enough not to have the
/// interface.
fn probe_numa_nodes(_core_count: u32) -> u32 {
    #[cfg(target_os = "linux")]
    {
        if let Ok(entries) = std::fs::read_dir("/sys/devices/system/node") {
            let count = entries
                .filter_map(|e| e.ok())
                .filter(|e| {
                    e.file_name()
                        .to_string_lossy()
                        .strip_prefix("node")
                        .map(|rest| rest.chars().all(|c| c.is_ascii_digit()) && !rest.is_empty())
                        .unwrap_or(false)
                })
                .count();
            if count > 0 {
                return count as u32;
            }
        }
    }
    1
}

fn now_micros() -> i64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_micros() as i64)
        .unwrap_or(0)
}

// ---------------------------------------------------------------------------
// Latency histogram for passive storage measurement
// ---------------------------------------------------------------------------

/// Log-bucketed read latency histogram, written from the page read path.
///
/// Buckets are powers of two of nanoseconds, so a sample costs one leading
/// zero count and one relaxed add, and the whole structure is 64 counters.
/// Exact quantiles are not the point: telling 20 microseconds from 2
/// milliseconds is.
#[derive(Debug)]
pub struct LatencyHistogram {
    buckets: [AtomicU64; 64],
    total: AtomicU64,
    sum_nanos: AtomicU64,
    bytes: AtomicU64,
}

impl Default for LatencyHistogram {
    fn default() -> Self {
        Self::new()
    }
}

impl LatencyHistogram {
    pub fn new() -> Self {
        Self {
            buckets: std::array::from_fn(|_| AtomicU64::new(0)),
            total: AtomicU64::new(0),
            sum_nanos: AtomicU64::new(0),
            bytes: AtomicU64::new(0),
        }
    }

    /// Records one operation. Called on the page read path, so it is a leading
    /// zero count and three relaxed adds.
    #[inline]
    pub fn record(&self, nanos: u64, bytes: u64) {
        let bucket = 63 - nanos.max(1).leading_zeros() as usize;
        self.buckets[bucket].fetch_add(1, Ordering::Relaxed);
        self.total.fetch_add(1, Ordering::Relaxed);
        self.sum_nanos.fetch_add(nanos, Ordering::Relaxed);
        self.bytes.fetch_add(bytes, Ordering::Relaxed);
    }

    pub fn count(&self) -> u64 {
        self.total.load(Ordering::Relaxed)
    }

    pub fn total_bytes(&self) -> u64 {
        self.bytes.load(Ordering::Relaxed)
    }

    pub fn mean_nanos(&self) -> u64 {
        let n = self.count();
        if n == 0 {
            return 0;
        }
        self.sum_nanos.load(Ordering::Relaxed) / n
    }

    /// Upper edge of the bucket the requested quantile falls in.
    ///
    /// Bucket resolution means the answer is within a factor of two of the
    /// true quantile, which is the resolution the decisions downstream act on.
    pub fn quantile_nanos(&self, q: f64) -> u64 {
        let total = self.count();
        if total == 0 {
            return 0;
        }
        let target = (total as f64 * q.clamp(0.0, 1.0)).ceil() as u64;
        let mut seen = 0u64;
        for (i, bucket) in self.buckets.iter().enumerate() {
            seen += bucket.load(Ordering::Relaxed);
            if seen >= target {
                return 1u64 << i;
            }
        }
        1u64 << 63
    }

    /// Throughput implied by everything recorded, in megabytes per second.
    pub fn throughput_mb_s(&self) -> f64 {
        let nanos = self.sum_nanos.load(Ordering::Relaxed);
        if nanos == 0 {
            return 0.0;
        }
        let bytes = self.total_bytes() as f64;
        bytes / (nanos as f64 / 1e9) / (1024.0 * 1024.0)
    }

    /// Clears every counter, used when a window closes.
    pub fn reset(&self) {
        for bucket in &self.buckets {
            bucket.store(0, Ordering::Relaxed);
        }
        self.total.store(0, Ordering::Relaxed);
        self.sum_nanos.store(0, Ordering::Relaxed);
        self.bytes.store(0, Ordering::Relaxed);
    }
}

// ---------------------------------------------------------------------------
// Passive collection
// ---------------------------------------------------------------------------

/// One in this many page reads is timed.
///
/// A cached page read costs a microsecond or two, and a pair of clock reads
/// costs tens of nanoseconds, so timing every one would spend a few percent of
/// the read path on measuring it. A latency histogram does not need every
/// sample, it needs enough of them, and one in sixty-four leaves the
/// distribution intact while putting a counter increment and a branch on the
/// path instead of two clock reads.
const PAGE_READ_SAMPLE_MASK: u64 = 63;

static PAGE_READ_COUNTER: AtomicU64 = AtomicU64::new(0);

/// Whether this page read should be timed, and the clock reading to time it
/// from. None on the calls that are not sampled, which is most of them.
///
/// The counter is scrambled rather than strided. A scan reads pages in runs,
/// and the first read of a run behaves differently from the rest: it is the
/// one that misses the cache and starts the readahead. A plain every-sixty-
/// fourth sample would take the same position in every run and measure that
/// position's latency as the device's, which is how a p50 ends up describing
/// a read nobody does.
#[inline]
pub fn sample_page_read() -> Option<std::time::Instant> {
    let n = PAGE_READ_COUNTER.fetch_add(1, Ordering::Relaxed);
    if scramble(n) & PAGE_READ_SAMPLE_MASK == 0 {
        Some(std::time::Instant::now())
    } else {
        None
    }
}

/// How many reads one sample stands for, so the coefficient reflects the whole
/// traffic rather than the sampled slice.
pub const PAGE_READ_SAMPLE_WEIGHT: u64 = PAGE_READ_SAMPLE_MASK + 1;

// ---------------------------------------------------------------------------
// Contention and skew signals
// ---------------------------------------------------------------------------

/// Counters the bottleneck classifier reads to tell one kind of saturation
/// from another. Each is fed from the subsystem that owns the truth.
#[derive(Debug)]
pub struct ContentionSignals {
    txn_commits: AtomicU64,
    txn_aborts_conflict: AtomicU64,
    wal_group_commit_hits: AtomicU64,
    wal_group_commit_misses: AtomicU64,
    writes_waiting: AtomicU32,
    /// Traffic to the hottest one percent of keys, and to everything, in the
    /// current window
    hot_key_touches: AtomicU64,
    total_key_touches: AtomicU64,
    /// Which keys the traffic is landing on, bounded and sampled
    hot_keys: std::sync::Mutex<HotKeySketch>,
    key_sample_counter: AtomicU64,
    /// Pages read from storage in the current window.
    ///
    /// What separates a node whose cores are idle because storage is slow from
    /// one whose cores are idle because admission is holding queries back.
    /// Both look identical from the parallel budget, and they want opposite
    /// responses: the first wants more work in flight, the second wants none
    window_page_reads: AtomicU64,
}

/// One write in this many reaches the hot-extent sketch, on average.
///
/// The row write path already pays for a WAL append, so a sampled sketch
/// update beside it is not measurable. Sixteen is low enough that a skew
/// concentrated on one extent is visible within a window at any realistic
/// write rate, and high enough that the sketch's lock is never contended.
const KEY_SAMPLE_STRIDE: u64 = 16;
const KEY_SAMPLE_MASK: u64 = KEY_SAMPLE_STRIDE - 1;

/// Distinct extents a window must touch before concentration means anything.
///
/// Skew is a statement about how traffic is distributed across partitions, and
/// a workload writing one page has no distribution to be skewed. Reporting one
/// there would mask provisioning on every small table permanently, which is
/// worse than missing a skew that has nowhere to spread anyway.
const MIN_DISTINCT_FOR_SKEW: usize = 8;

/// Spreads a counter's low bits so a sample stride cannot align with a
/// periodic workload.
///
/// The finalizer from splitmix64, which is three instructions and mixes every
/// input bit into every output bit. Taking the counter's low bits directly
/// would sample exactly the indices a round-robin writer puts its cold keys
/// on.
#[inline]
const fn scramble(n: u64) -> u64 {
    let mut z = n.wrapping_add(0x9E37_79B9_7F4A_7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// Keys the sketch tracks at once.
///
/// Space-Saving over a fixed set of slots: enough that a genuine hot key is
/// never evicted by the long tail, small enough that the whole thing is two
/// cache lines and the scan to find the victim is a handful of comparisons.
const HOT_KEY_SLOTS: usize = 64;

/// Finds the heaviest key in an unbounded stream using bounded memory.
///
/// Space-Saving: a new key takes the lightest slot and inherits its count as
/// its error bound, so the count is an over-estimate and `count - error` is a
/// guaranteed lower bound on the true frequency. The lower bound is what gets
/// reported, because over-detecting a hot partition masks provisioning that
/// might have helped, and under-detecting only costs a window.
#[derive(Debug)]
struct HotKeySketch {
    keys: [u64; HOT_KEY_SLOTS],
    counts: [u64; HOT_KEY_SLOTS],
    errors: [u64; HOT_KEY_SLOTS],
    occupied: usize,
}

impl HotKeySketch {
    fn new() -> Self {
        Self {
            keys: [0; HOT_KEY_SLOTS],
            counts: [0; HOT_KEY_SLOTS],
            errors: [0; HOT_KEY_SLOTS],
            occupied: 0,
        }
    }

    fn observe(&mut self, key: u64) {
        for i in 0..self.occupied {
            if self.keys[i] == key {
                self.counts[i] += 1;
                return;
            }
        }
        if self.occupied < HOT_KEY_SLOTS {
            let i = self.occupied;
            self.keys[i] = key;
            self.counts[i] = 1;
            self.errors[i] = 0;
            self.occupied += 1;
            return;
        }
        // Full: the lightest slot is displaced, and the new key inherits its
        // count as the error it carries
        let mut victim = 0usize;
        for i in 1..HOT_KEY_SLOTS {
            if self.counts[i] < self.counts[victim] {
                victim = i;
            }
        }
        self.keys[victim] = key;
        self.errors[victim] = self.counts[victim];
        self.counts[victim] += 1;
    }

    /// Distinct keys currently tracked.
    fn distinct(&self) -> usize {
        self.occupied
    }

    /// The heaviest key's guaranteed minimum count.
    fn hottest_lower_bound(&self) -> u64 {
        (0..self.occupied)
            .map(|i| self.counts[i].saturating_sub(self.errors[i]))
            .max()
            .unwrap_or(0)
    }

    /// Halves every count, so the sketch follows a skew that moves without
    /// forgetting one that has not.
    fn decay(&mut self) {
        for i in 0..self.occupied {
            self.counts[i] /= 2;
            self.errors[i] /= 2;
        }
        // A key that decayed to nothing is no longer evidence of anything, and
        // holding its slot would keep a genuinely hot key out
        let mut i = 0;
        while i < self.occupied {
            if self.counts[i] == 0 {
                let last = self.occupied - 1;
                self.keys[i] = self.keys[last];
                self.counts[i] = self.counts[last];
                self.errors[i] = self.errors[last];
                self.occupied -= 1;
            } else {
                i += 1;
            }
        }
    }
}

impl Default for ContentionSignals {
    fn default() -> Self {
        Self::new()
    }
}

impl ContentionSignals {
    pub fn new() -> Self {
        Self {
            txn_commits: AtomicU64::new(0),
            txn_aborts_conflict: AtomicU64::new(0),
            wal_group_commit_hits: AtomicU64::new(0),
            wal_group_commit_misses: AtomicU64::new(0),
            writes_waiting: AtomicU32::new(0),
            hot_key_touches: AtomicU64::new(0),
            total_key_touches: AtomicU64::new(0),
            hot_keys: std::sync::Mutex::new(HotKeySketch::new()),
            key_sample_counter: AtomicU64::new(0),
            window_page_reads: AtomicU64::new(0),
        }
    }

    pub fn record_commit(&self) {
        self.txn_commits.fetch_add(1, Ordering::Relaxed);
    }

    pub fn record_conflict_abort(&self) {
        self.txn_aborts_conflict.fetch_add(1, Ordering::Relaxed);
    }

    pub fn record_group_commit(&self, joined_a_group: bool) {
        if joined_a_group {
            self.wal_group_commit_hits.fetch_add(1, Ordering::Relaxed);
        } else {
            self.wal_group_commit_misses.fetch_add(1, Ordering::Relaxed);
        }
    }

    pub fn set_writes_waiting(&self, n: u32) {
        self.writes_waiting.store(n, Ordering::Relaxed);
    }

    /// A writer has started waiting on the device.
    ///
    /// Counted rather than set, because the number wanted is how many are
    /// waiting at once and no single caller knows that. Paired with
    /// `leave_write_wait`, which every path out of the wait must reach.
    #[inline]
    pub fn enter_write_wait(&self) {
        self.writes_waiting.fetch_add(1, Ordering::Relaxed);
    }

    /// A writer has stopped waiting, whether it was satisfied or errored.
    #[inline]
    pub fn leave_write_wait(&self) {
        // Saturating, because a decrement that outran its increment would
        // wrap to four billion writers waiting and hold the node in the
        // fsync-bound classification permanently
        let mut current = self.writes_waiting.load(Ordering::Relaxed);
        while current > 0 {
            match self.writes_waiting.compare_exchange_weak(
                current,
                current - 1,
                Ordering::Relaxed,
                Ordering::Relaxed,
            ) {
                Ok(_) => return,
                Err(observed) => current = observed,
            }
        }
    }

    /// Records pages read from storage, in whole reads rather than samples.
    #[inline]
    pub fn record_page_reads(&self, pages: u64) {
        self.window_page_reads.fetch_add(pages, Ordering::Relaxed);
    }

    /// Pages read from storage in the current window.
    pub fn page_reads(&self) -> u64 {
        self.window_page_reads.load(Ordering::Relaxed)
    }

    /// Records that a key was written.
    ///
    /// Sampled, because this is called from the row write path. One in
    /// `KEY_SAMPLE_STRIDE` touches reaches the sketch and each one counts for
    /// the whole stride, so the share it reports describes all the traffic
    /// rather than the sampled slice.
    #[inline]
    pub fn record_key(&self, key_hash: u64) {
        let n = self.key_sample_counter.fetch_add(1, Ordering::Relaxed);
        // Scrambled rather than every sixteenth call. A plain stride aliases
        // with any periodic write pattern, and round-robin over partitions is
        // one of the most common shapes there is: a workload whose hot key
        // lands on the unsampled indices would be invisible, which is the one
        // failure this signal cannot afford
        if scramble(n) & KEY_SAMPLE_MASK != 0 {
            return;
        }
        self.total_key_touches
            .fetch_add(KEY_SAMPLE_STRIDE, Ordering::Relaxed);
        if let Ok(mut sketch) = self.hot_keys.lock() {
            sketch.observe(key_hash);
        }
    }

    /// Records a batch of key touches directly, for a caller that already
    /// knows the split. Used by the tests that pin the classifier's thresholds.
    pub fn record_key_touches(&self, hot: u64, total: u64) {
        self.hot_key_touches.fetch_add(hot, Ordering::Relaxed);
        self.total_key_touches.fetch_add(total, Ordering::Relaxed);
    }

    /// Share of transaction outcomes that were conflict aborts. Rising means
    /// concurrency is producing conflict, and more capacity would produce more
    /// of it.
    pub fn occ_abort_rate(&self) -> f64 {
        let aborts = self.txn_aborts_conflict.load(Ordering::Relaxed);
        let commits = self.txn_commits.load(Ordering::Relaxed);
        let total = aborts + commits;
        if total == 0 {
            return 0.0;
        }
        aborts as f64 / total as f64
    }

    /// Share of commits that batched with another. Low with writers waiting
    /// means the node is paying a device round trip per commit.
    pub fn group_commit_hit_rate(&self) -> f64 {
        let hits = self.wal_group_commit_hits.load(Ordering::Relaxed);
        let misses = self.wal_group_commit_misses.load(Ordering::Relaxed);
        let total = hits + misses;
        if total == 0 {
            return 1.0;
        }
        hits as f64 / total as f64
    }

    pub fn writes_waiting(&self) -> u32 {
        self.writes_waiting.load(Ordering::Relaxed)
    }

    /// Share of key traffic landing on the hottest key. A high share does not
    /// spread across more nodes.
    ///
    /// Takes the larger of what the sketch found and what a caller reported
    /// directly, so a test that pins the value and a node measuring its own
    /// traffic both answer the same question.
    pub fn hot_key_share(&self) -> f64 {
        let total = self.total_key_touches.load(Ordering::Relaxed);
        if total == 0 {
            return 0.0;
        }
        let reported = self.hot_key_touches.load(Ordering::Relaxed);
        let sketched = self
            .hot_keys
            .lock()
            .map(|s| {
                if s.distinct() < MIN_DISTINCT_FOR_SKEW {
                    0
                } else {
                    s.hottest_lower_bound().saturating_mul(KEY_SAMPLE_STRIDE)
                }
            })
            .unwrap_or(0);
        reported.max(sketched) as f64 / total as f64
    }

    /// Clears the windowed counters. The classifier reads rates over a window,
    /// so the window has to end.
    pub fn roll_window(&self) {
        self.txn_commits.store(0, Ordering::Relaxed);
        self.txn_aborts_conflict.store(0, Ordering::Relaxed);
        self.wal_group_commit_hits.store(0, Ordering::Relaxed);
        self.wal_group_commit_misses.store(0, Ordering::Relaxed);
        self.hot_key_touches.store(0, Ordering::Relaxed);
        self.total_key_touches.store(0, Ordering::Relaxed);
        self.window_page_reads.store(0, Ordering::Relaxed);
        // The sketch is halved rather than cleared. A key that is hot stays
        // hot across a window boundary, and clearing would make the classifier
        // forget a skew every window and rediscover it every window, which
        // reads as flapping rather than as a hot partition
        if let Ok(mut sketch) = self.hot_keys.lock() {
            sketch.decay();
        }
        // Writers currently waiting is a level, not a rate, so it survives the
        // window: the ones still parked are still parked
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn inputs(cpu: &str, cores: u32, ram_gb: u32) -> FingerprintInputs {
        FingerprintInputs {
            cpu_model: cpu.to_string(),
            core_count: cores,
            ram_gb,
            numa_nodes: 1,
            simd_level: SimdLevel::Avx2,
            mounts: vec![("/data".to_string(), DeviceKind::Nvme)],
        }
    }

    #[test]
    fn identical_hardware_fingerprints_identically() {
        let a = inputs("Xeon Platinum 8375C", 32, 128);
        let b = inputs("Xeon Platinum 8375C", 32, 128);
        assert_eq!(a.fingerprint(), b.fingerprint());
    }

    #[test]
    fn fingerprint_splits_on_anything_that_moves_the_coefficients() {
        let base = inputs("Xeon Platinum 8375C", 32, 128);
        let base_fp = base.fingerprint();
        // Twice the cores is a different machine for calibration purposes
        assert_ne!(
            inputs("Xeon Platinum 8375C", 64, 128).fingerprint(),
            base_fp
        );
        // So is twice the memory
        assert_ne!(
            inputs("Xeon Platinum 8375C", 32, 256).fingerprint(),
            base_fp
        );
        // So is a different CPU
        assert_ne!(inputs("EPYC 7R13", 32, 128).fingerprint(), base_fp);

        let mut different_disk = base.clone();
        different_disk.mounts = vec![("/data".to_string(), DeviceKind::Network)];
        assert_ne!(different_disk.fingerprint(), base_fp);
    }

    #[test]
    fn mount_order_does_not_change_the_fingerprint() {
        let mut a = inputs("cpu", 8, 32);
        a.mounts = vec![
            ("/a".to_string(), DeviceKind::Nvme),
            ("/b".to_string(), DeviceKind::Ssd),
        ];
        let mut b = inputs("cpu", 8, 32);
        b.mounts = vec![
            ("/b".to_string(), DeviceKind::Ssd),
            ("/a".to_string(), DeviceKind::Nvme),
        ];
        assert_eq!(a.fingerprint(), b.fingerprint());
    }

    #[test]
    fn fingerprint_hex_round_trips() {
        let fp = inputs("cpu", 4, 16).fingerprint();
        assert_eq!(HardwareFingerprint::parse_hex(&fp.to_hex()), Some(fp));
        assert_eq!(fp.to_hex().len(), 16);
    }

    #[test]
    fn accumulator_moves_toward_what_was_measured() {
        let acc = CoefficientAccumulator::new();
        let cold = OperatorKind::SeqScan.cold_start_ns_per_unit() as f64;
        assert!((acc.get(OperatorKind::SeqScan) - cold).abs() < 1e-6);

        // First evidence replaces the assumption outright
        acc.record(OperatorKind::SeqScan, 1_000, 5_000);
        acc.drain();
        assert!(
            (acc.get(OperatorKind::SeqScan) - 5.0).abs() < 0.01,
            "got {}",
            acc.get(OperatorKind::SeqScan)
        );

        // Later evidence moves it part of the way, so one odd batch cannot
        // throw the estimate
        for _ in 0..50 {
            acc.record(OperatorKind::SeqScan, 1_000, 9_000);
            acc.drain();
        }
        let settled = acc.get(OperatorKind::SeqScan);
        assert!(
            settled > 5.0 && settled <= 9.1,
            "should track toward 9, got {settled}"
        );
    }

    /// The floor on the blend weight is what lets a slowing device move the
    /// number after millions of samples, so it is pinned.
    #[test]
    fn blend_weight_never_reaches_zero() {
        assert!(blend_weight(u64::MAX / 2) >= 0.02);
        assert!(blend_weight(0) > 0.9);
        assert!(blend_weight(MIN_SAMPLES_FOR_CONFIDENCE) < blend_weight(0));
    }

    #[test]
    fn accumulator_tracks_a_device_that_gets_slower() {
        let acc = CoefficientAccumulator::new();
        for _ in 0..500 {
            acc.record(OperatorKind::PageRead, 100, 100 * 20_000);
            acc.drain();
        }
        let fast = acc.get(OperatorKind::PageRead);
        assert!((fast - 20_000.0).abs() < 2_000.0, "got {fast}");

        // The volume degrades by 5x and the estimate has to follow it
        for _ in 0..500 {
            acc.record(OperatorKind::PageRead, 100, 100 * 100_000);
            acc.drain();
        }
        let slow = acc.get(OperatorKind::PageRead);
        assert!(slow > fast * 3.0, "stuck at {slow} after {fast}");
    }

    #[test]
    fn seeding_starts_a_node_calibrated() {
        let mut learned = OperatorCoefficients::cold_start();
        learned.ns_per_unit[OperatorKind::Sort.index()] = 12.5;
        learned.sample_count[OperatorKind::Sort.index()] = 50_000;

        let acc = CoefficientAccumulator::new();
        acc.seed(&learned);
        assert!((acc.get(OperatorKind::Sort) - 12.5).abs() < 0.01);
        assert_eq!(acc.samples(OperatorKind::Sort), 50_000);
        assert!(!acc.undersampled().contains(&OperatorKind::Sort));
    }

    #[test]
    fn merge_weights_each_side_by_its_evidence() {
        let mut a = OperatorCoefficients::cold_start();
        a.ns_per_unit[OperatorKind::Sort.index()] = 10.0;
        a.sample_count[OperatorKind::Sort.index()] = 100;

        let mut b = OperatorCoefficients::cold_start();
        b.ns_per_unit[OperatorKind::Sort.index()] = 20.0;
        b.sample_count[OperatorKind::Sort.index()] = 300;

        a.merge(&b);
        // Three quarters of the evidence says 20, so the answer sits at 17.5
        assert!((a.get(OperatorKind::Sort) - 17.5).abs() < 1e-9);
        assert_eq!(a.samples(OperatorKind::Sort), 400);
    }

    #[test]
    fn merge_ignores_a_peer_with_no_evidence() {
        let mut a = OperatorCoefficients::cold_start();
        a.ns_per_unit[OperatorKind::Filter.index()] = 3.0;
        a.sample_count[OperatorKind::Filter.index()] = 1_000;
        let before = a.get(OperatorKind::Filter);

        a.merge(&OperatorCoefficients::cold_start());
        assert!((a.get(OperatorKind::Filter) - before).abs() < 1e-9);
        assert_eq!(a.samples(OperatorKind::Filter), 1_000);
    }

    #[test]
    fn undersampled_kinds_are_the_ones_with_no_evidence_behind_them() {
        let acc = CoefficientAccumulator::new();
        assert_eq!(acc.undersampled().len(), OperatorKind::COUNT);
        acc.record(OperatorKind::Sort, MIN_SAMPLES_FOR_CONFIDENCE * 2, 1_000);
        acc.drain();
        assert!(!acc.undersampled().contains(&OperatorKind::Sort));
    }

    /// Recent traffic takes a kind off the probe list, and going quiet for a
    /// whole window puts it back on.
    #[test]
    fn probe_candidates_follow_recent_traffic_not_lifetime_evidence() {
        let acc = CoefficientAccumulator::new();
        assert_eq!(acc.probe_candidates().len(), OperatorKind::COUNT);

        acc.record(OperatorKind::Sort, MIN_SAMPLES_FOR_CONFIDENCE * 2, 1_000);
        acc.drain();
        assert!(!acc.probe_candidates().contains(&OperatorKind::Sort));
        assert_eq!(acc.recent_samples(OperatorKind::Sort), 200);

        // Quiet for a whole observation window, so the evidence is stale even
        // though the lifetime count still says it is well measured
        for _ in 0..RECENT_BUCKETS {
            acc.drain();
        }
        assert_eq!(acc.recent_samples(OperatorKind::Sort), 0);
        assert!(acc.probe_candidates().contains(&OperatorKind::Sort));
        assert!(!acc.undersampled().contains(&OperatorKind::Sort));
    }

    /// A node that inherited calibration must not probe on its first drain.
    #[test]
    fn a_seeded_node_trusts_what_it_inherited() {
        let mut learned = OperatorCoefficients::cold_start();
        for kind in OperatorKind::ALL {
            learned.ns_per_unit[kind.index()] = 7.0;
            learned.sample_count[kind.index()] = MIN_SAMPLES_FOR_CONFIDENCE * 10;
        }
        let acc = CoefficientAccumulator::new();
        acc.seed(&learned);
        assert!(acc.undersampled().is_empty());
    }

    #[test]
    fn histogram_separates_a_microsecond_from_a_millisecond() {
        let hist = LatencyHistogram::new();
        for _ in 0..99 {
            hist.record(20_000, 4096);
        }
        hist.record(4_000_000, 4096);
        assert_eq!(hist.count(), 100);

        let p50 = hist.quantile_nanos(0.50);
        let p99 = hist.quantile_nanos(0.999);
        assert!(p50 >= 16_384 && p50 <= 32_768, "p50 {p50}");
        assert!(p99 >= 2_097_152, "p99 {p99}");
        assert!(hist.throughput_mb_s() > 0.0);

        hist.reset();
        assert_eq!(hist.count(), 0);
        assert_eq!(hist.quantile_nanos(0.5), 0);
    }

    #[test]
    fn device_kind_follows_measured_latency() {
        assert_eq!(DeviceKind::from_read_latency_ns(20_000), DeviceKind::Nvme);
        assert_eq!(DeviceKind::from_read_latency_ns(200_000), DeviceKind::Ssd);
        assert_eq!(
            DeviceKind::from_read_latency_ns(1_500_000),
            DeviceKind::Network
        );
        assert_eq!(DeviceKind::from_read_latency_ns(8_000_000), DeviceKind::Hdd);
        assert_eq!(DeviceKind::from_read_latency_ns(0), DeviceKind::Unknown);
    }

    #[test]
    fn contention_signals_report_rates_over_their_window() {
        let signals = ContentionSignals::new();
        for _ in 0..90 {
            signals.record_commit();
        }
        for _ in 0..10 {
            signals.record_conflict_abort();
        }
        assert!((signals.occ_abort_rate() - 0.10).abs() < 1e-9);

        signals.roll_window();
        assert_eq!(signals.occ_abort_rate(), 0.0);

        // No commits at all is not a conflict problem
        assert!((signals.group_commit_hit_rate() - 1.0).abs() < 1e-9);
        signals.record_group_commit(true);
        signals.record_group_commit(false);
        assert!((signals.group_commit_hit_rate() - 0.5).abs() < 1e-9);

        signals.record_key_touches(30, 100);
        assert!((signals.hot_key_share() - 0.30).abs() < 1e-9);
    }

    #[test]
    fn probe_reads_the_machine_without_writing_anything() {
        let caps = NodeCapabilities::probe(7, None);
        assert_eq!(caps.node_id, 7);
        assert!(caps.core_count >= 1);
        assert!(caps.mem_total_bytes > 0);
        assert!(caps.numa_nodes >= 1);
        assert_ne!(caps.fingerprint.0, 0);
        assert!(!caps.inherited);
        // Cold start values, nothing measured yet
        for kind in OperatorKind::ALL {
            assert_eq!(caps.coefficients.samples(kind), 0);
        }
    }

    #[test]
    fn probe_is_stable_across_calls_on_one_machine() {
        let a = NodeCapabilities::probe(1, None);
        let b = NodeCapabilities::probe(2, None);
        assert_eq!(
            a.fingerprint, b.fingerprint,
            "the same machine must fingerprint the same twice"
        );
    }

    #[test]
    fn adopting_calibration_marks_it_inherited() {
        let mut caps = NodeCapabilities::probe(1, None);
        let mut learned = OperatorCoefficients::cold_start();
        learned.ns_per_unit[OperatorKind::SeqScan.index()] = 3.25;
        learned.sample_count[OperatorKind::SeqScan.index()] = 1_000_000;
        caps.adopt(learned);
        assert!(caps.inherited);
        assert!((caps.coefficients.get(OperatorKind::SeqScan) - 3.25).abs() < 1e-9);
    }

    #[test]
    fn estimate_scales_linearly_with_units() {
        let mut coefficients = OperatorCoefficients::cold_start();
        coefficients.ns_per_unit[OperatorKind::SeqScan.index()] = 50.0;
        // A million rows at 50ns each is 50 milliseconds
        let est = coefficients.estimate_seconds(OperatorKind::SeqScan, 1_000_000.0);
        assert!((est - 0.050).abs() < 1e-9);
        assert_eq!(
            coefficients.estimate_seconds(OperatorKind::SeqScan, 0.0),
            0.0
        );
        assert_eq!(
            coefficients.estimate_seconds(OperatorKind::SeqScan, f64::NAN),
            0.0
        );
    }

    #[test]
    fn operator_kinds_round_trip() {
        for kind in OperatorKind::ALL {
            assert_eq!(OperatorKind::from_index(kind.index()), Some(kind));
            assert_eq!(OperatorKind::parse(kind.as_str()), Some(kind));
        }
        assert_eq!(OperatorKind::from_index(OperatorKind::COUNT), None);
        assert!(OperatorKind::PageRead.unit_is_page());
        assert!(!OperatorKind::SeqScan.unit_is_page());
    }
}
