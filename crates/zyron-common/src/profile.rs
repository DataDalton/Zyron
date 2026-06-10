//! Wall-clock phase profiler for the commit and query hot paths.
//!
//! Compile-gated by the `profile` cargo feature. Without the feature (the
//! default, i.e. production builds) every hook is a zero-sized no-op the
//! optimizer elides entirely, so the binary carries no instrumentation. With
//! `--features profile` the hooks are compiled in and further gated at runtime
//! by the ZYRON_PROFILE env var, so a profile-enabled build still runs clean
//! unless the var is set.
//!
//! When active, each phase accumulates a call count and total wall-clock
//! nanoseconds. Wall clock (not CPU time) is deliberate: the durable-commit path
//! spends much of its time parked on the flush thread and blocked in the device
//! write, which a CPU sampling profiler would not attribute.
//!
//! Distortion control: counters are thread-local. Each thread accumulates into
//! its own leaked atomic array, so 256 concurrent committers never contend on a
//! shared counter (which would itself dominate the measurement). `dump` sums
//! every registered thread array.

/// Pipeline phases, grouped by layer. Defined unconditionally so instrumentation
/// call sites compile in any build; the recording machinery behind them is
/// compiled only under the `profile` feature.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(usize)]
pub enum Phase {
    // Wire / server, per statement
    WireRecvParse = 0,
    WirePlan,
    WireExecSetup,
    WireExecute,
    WireSend,
    WireAutoCommit,

    // Transaction commit path
    TxnBegin,
    CommitRecordAppend,
    DurabilityWait,
    LockRelease,
    ProcArrayRelease,

    // WAL append (producer side)
    WalReserve,
    WalSerialize,
    WalPublish,
    WalBackpressure,

    // WAL flush thread (single consumer)
    FlushWaitWork,
    FlushAdvance,
    FlushDrain,
    FlushChecksum,
    FlushSegWrite,
    FlushFsync,
    FlushWake,
    // Accumulates records-per-flush (a count, not nanoseconds): the "ns" field
    // sums batch sizes so ns/count reads as the average commit-batch size.
    FlushBatchRecords,
}

#[cfg(feature = "profile")]
mod imp {
    use super::Phase;
    use std::sync::Mutex;
    use std::sync::OnceLock;
    use std::sync::atomic::{AtomicU64, Ordering};
    use std::time::Instant;

    impl Phase {
        /// Every phase, in discriminant order. COUNT and all indexing derive
        /// from this, so adding a phase to the enum and here cannot drift apart.
        const ALL: [Phase; 23] = [
            Phase::WireRecvParse,
            Phase::WirePlan,
            Phase::WireExecSetup,
            Phase::WireExecute,
            Phase::WireSend,
            Phase::WireAutoCommit,
            Phase::TxnBegin,
            Phase::CommitRecordAppend,
            Phase::DurabilityWait,
            Phase::LockRelease,
            Phase::ProcArrayRelease,
            Phase::WalReserve,
            Phase::WalSerialize,
            Phase::WalPublish,
            Phase::WalBackpressure,
            Phase::FlushWaitWork,
            Phase::FlushAdvance,
            Phase::FlushDrain,
            Phase::FlushChecksum,
            Phase::FlushSegWrite,
            Phase::FlushFsync,
            Phase::FlushWake,
            Phase::FlushBatchRecords,
        ];

        const COUNT: usize = Phase::ALL.len();

        fn name(self) -> &'static str {
            match self {
                Phase::WireRecvParse => "wire.recv_parse",
                Phase::WirePlan => "wire.plan",
                Phase::WireExecSetup => "wire.exec_setup",
                Phase::WireExecute => "wire.execute",
                Phase::WireSend => "wire.send",
                Phase::WireAutoCommit => "wire.auto_commit",
                Phase::TxnBegin => "txn.begin",
                Phase::CommitRecordAppend => "txn.commit_record_append",
                Phase::DurabilityWait => "txn.durability_wait",
                Phase::LockRelease => "txn.lock_release",
                Phase::ProcArrayRelease => "txn.proc_array_release",
                Phase::WalReserve => "wal.reserve",
                Phase::WalSerialize => "wal.serialize",
                Phase::WalPublish => "wal.publish",
                Phase::WalBackpressure => "wal.backpressure",
                Phase::FlushWaitWork => "flush.wait_work",
                Phase::FlushAdvance => "flush.advance_committed",
                Phase::FlushDrain => "flush.drain_copy",
                Phase::FlushChecksum => "flush.checksum_backfill",
                Phase::FlushSegWrite => "flush.seg_write",
                Phase::FlushFsync => "flush.fsync",
                Phase::FlushWake => "flush.wake",
                Phase::FlushBatchRecords => "flush.batch_records(avg)",
            }
        }
    }

    // ALL must list phases in discriminant order, since the per-phase slot index
    // is the discriminant and the report indexes ALL by position. This fails the
    // build if the two ever diverge.
    const _: () = {
        let mut i = 0;
        while i < Phase::ALL.len() {
            assert!(Phase::ALL[i] as usize == i);
            i += 1;
        }
    };

    /// Two u64 slots per phase, interleaved: [count, accum, count, accum, ...].
    const SLOTS: usize = Phase::COUNT * 2;

    /// Registry of every thread's leaked counter array. Pointers are stored as
    /// usize; the arrays are leaked so they outlive their threads and stay valid
    /// for the dumper to read.
    static REGISTRY: Mutex<Vec<usize>> = Mutex::new(Vec::new());

    #[inline]
    fn enabled() -> bool {
        static ON: OnceLock<bool> = OnceLock::new();
        *ON.get_or_init(|| std::env::var("ZYRON_PROFILE").is_ok())
    }

    thread_local! {
        static LOCAL: *const [AtomicU64; SLOTS] = register_thread();
    }

    fn register_thread() -> *const [AtomicU64; SLOTS] {
        let arr: Box<[AtomicU64; SLOTS]> = Box::new(std::array::from_fn(|_| AtomicU64::new(0)));
        let ptr = Box::into_raw(arr) as *const [AtomicU64; SLOTS];
        // Leaked on purpose: the array must outlive the thread so dump can read it.
        if let Ok(mut reg) = REGISTRY.lock() {
            reg.push(ptr as usize);
        }
        ptr
    }

    #[inline]
    fn add(phase: Phase, value: u64) {
        LOCAL.with(|&ptr| {
            // SAFETY: ptr is this thread's own leaked array, valid for the
            // process lifetime. Atomic ops make the concurrent read in dump
            // well-defined.
            let arr = unsafe { &*ptr };
            let base = phase as usize * 2;
            arr[base].fetch_add(1, Ordering::Relaxed);
            arr[base + 1].fetch_add(value, Ordering::Relaxed);
        });
    }

    /// Records `nanos` against `phase` (and one call). No-op when the env gate
    /// is unset.
    #[inline]
    pub fn record(phase: Phase, nanos: u64) {
        if !enabled() {
            return;
        }
        add(phase, nanos);
    }

    /// Records a raw value (e.g. a batch size) against `phase`. Read back via the
    /// average column. No-op when the env gate is unset.
    #[inline]
    pub fn record_value(phase: Phase, value: u64) {
        if !enabled() {
            return;
        }
        add(phase, value);
    }

    /// RAII span: records elapsed wall-clock against `phase` on drop. Holds no
    /// timestamp and does nothing when the env gate is unset.
    pub struct Span {
        phase: Phase,
        start: Option<Instant>,
    }

    impl Drop for Span {
        #[inline]
        fn drop(&mut self) {
            if let Some(s) = self.start {
                add(self.phase, s.elapsed().as_nanos() as u64);
            }
        }
    }

    /// Starts a timing span for `phase`. The span records on drop.
    #[inline]
    pub fn scope(phase: Phase) -> Span {
        Span {
            phase,
            start: if enabled() {
                Some(Instant::now())
            } else {
                None
            },
        }
    }

    /// Returns true if profiling is active (feature on and env gate set).
    #[inline]
    pub fn is_enabled() -> bool {
        enabled()
    }

    /// Sums every registered thread array and returns (count, accum) per phase.
    fn totals() -> [(u64, u64); Phase::COUNT] {
        let mut out = [(0u64, 0u64); Phase::COUNT];
        if let Ok(reg) = REGISTRY.lock() {
            for &p in reg.iter() {
                let arr = unsafe { &*(p as *const [AtomicU64; SLOTS]) };
                for i in 0..Phase::COUNT {
                    out[i].0 += arr[i * 2].load(Ordering::Relaxed);
                    out[i].1 += arr[i * 2 + 1].load(Ordering::Relaxed);
                }
            }
        }
        out
    }

    /// Formats the accumulated profile as a table: phase, calls, total ms,
    /// ns/call. The batch-size phase is shown as an average rather than a time.
    pub fn report() -> String {
        let t = totals();
        let mut s = String::new();
        s.push_str("=== ZYRON_PROFILE phase breakdown (wall-clock) ===\n");
        s.push_str(&format!(
            "{:<28} {:>12} {:>12} {:>12}\n",
            "phase", "calls", "total_ms", "ns/call"
        ));
        for (i, phase) in Phase::ALL.iter().enumerate() {
            let phase = *phase;
            let (calls, accum) = t[i];
            if calls == 0 {
                continue;
            }
            if matches!(phase, Phase::FlushBatchRecords) {
                s.push_str(&format!(
                    "{:<28} {:>12} {:>12} {:>12}\n",
                    phase.name(),
                    calls,
                    "-",
                    format!("{} avg", accum / calls)
                ));
            } else {
                s.push_str(&format!(
                    "{:<28} {:>12} {:>12.3} {:>12}\n",
                    phase.name(),
                    calls,
                    accum as f64 / 1_000_000.0,
                    accum / calls
                ));
            }
        }
        s
    }

    /// Prints the report to stderr, prefixed by `label`. No-op when the env gate
    /// is unset.
    pub fn dump(label: &str) {
        if !enabled() {
            return;
        }
        eprintln!("[{}]\n{}", label, report());
    }

    /// Zeroes every registered thread array, so a following measurement window
    /// starts clean. Used by benchmarks to report one table per concurrency
    /// level.
    pub fn reset() {
        if !enabled() {
            return;
        }
        if let Ok(reg) = REGISTRY.lock() {
            for &p in reg.iter() {
                let arr = unsafe { &*(p as *const [AtomicU64; SLOTS]) };
                for a in arr.iter() {
                    a.store(0, Ordering::Relaxed);
                }
            }
        }
    }
}

#[cfg(not(feature = "profile"))]
mod imp {
    use super::Phase;

    /// Zero-sized no-op span. Constructing and dropping it emits no code.
    pub struct Span;

    #[inline(always)]
    pub fn scope(_phase: Phase) -> Span {
        Span
    }

    #[inline(always)]
    pub fn record(_phase: Phase, _nanos: u64) {}

    #[inline(always)]
    pub fn record_value(_phase: Phase, _value: u64) {}

    #[inline(always)]
    pub fn is_enabled() -> bool {
        false
    }

    #[inline(always)]
    pub fn report() -> String {
        String::new()
    }

    #[inline(always)]
    pub fn dump(_label: &str) {}

    #[inline(always)]
    pub fn reset() {}
}

pub use imp::{Span, dump, is_enabled, record, record_value, report, reset, scope};
