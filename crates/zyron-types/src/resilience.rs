//! Circuit breaker, retry, timeout, hedged-request helpers
//!
//! CircuitBreaker is count based with three states (Closed, Open, HalfOpen)
//! and uses atomic counters so concurrent record_success/record_failure are
//! lock free in the common path. EmaCircuitBreaker is a rolling EMA variant
//! that opens on combined error-rate and latency-p95. CircuitBreakerRegistry
//! holds named breakers keyed by string and lives on a Session for
//! per-connection state
//!
//! retry runs a closure with exponential backoff and jitter
//! timeout_blocking runs a sync closure on a dedicated scoped thread and
//! cancels on duration expiry via a crossbeam channel select
//! hedged kicks off a duplicate after the configured p95 elapses

use std::sync::Arc;
use std::sync::atomic::{AtomicU8, AtomicU32, AtomicU64, Ordering};
use std::time::{Duration, Instant};

use rand::RngExt;
use zyron_common::{Result, ZyronError};

// ---------------------------------------------------------------------------
// State
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum CircuitState {
    Closed = 0,
    Open = 1,
    HalfOpen = 2,
}

impl CircuitState {
    fn from_u8(v: u8) -> Self {
        match v {
            1 => CircuitState::Open,
            2 => CircuitState::HalfOpen,
            _ => CircuitState::Closed,
        }
    }
}

#[derive(Debug, Clone)]
pub struct CircuitStatus {
    pub state: CircuitState,
    pub failure_count: u32,
    pub success_count: u32,
    pub failure_threshold: u32,
    pub reset_timeout_ms: u64,
    pub half_open_max_calls: u32,
}

// ---------------------------------------------------------------------------
// Count-based CircuitBreaker
// ---------------------------------------------------------------------------

pub struct CircuitBreaker {
    state: AtomicU8,
    failure_count: AtomicU32,
    success_count: AtomicU32,
    failure_threshold: AtomicU32,
    success_threshold: AtomicU32,
    reset_timeout: AtomicU64,
    half_open_max_calls: AtomicU32,
    half_open_inflight: AtomicU32,
    last_open_ms: AtomicU64,
    baseline: Instant,
}

impl CircuitBreaker {
    pub fn new(failure_threshold: u32, reset_timeout: Duration, half_open_max_calls: u32) -> Self {
        Self {
            state: AtomicU8::new(CircuitState::Closed as u8),
            failure_count: AtomicU32::new(0),
            success_count: AtomicU32::new(0),
            failure_threshold: AtomicU32::new(failure_threshold.max(1)),
            success_threshold: AtomicU32::new(half_open_max_calls.max(1)),
            reset_timeout: AtomicU64::new(reset_timeout.as_millis() as u64),
            half_open_max_calls: AtomicU32::new(half_open_max_calls.max(1)),
            half_open_inflight: AtomicU32::new(0),
            last_open_ms: AtomicU64::new(0),
            baseline: Instant::now(),
        }
    }

    fn now_ms(&self) -> u64 {
        self.baseline.elapsed().as_millis() as u64
    }

    pub fn state(&self) -> CircuitState {
        CircuitState::from_u8(self.state.load(Ordering::Acquire))
    }

    pub fn snapshot(&self) -> CircuitStatus {
        CircuitStatus {
            state: self.state(),
            failure_count: self.failure_count.load(Ordering::Acquire),
            success_count: self.success_count.load(Ordering::Acquire),
            failure_threshold: self.failure_threshold.load(Ordering::Acquire),
            reset_timeout_ms: self.reset_timeout.load(Ordering::Acquire),
            half_open_max_calls: self.half_open_max_calls.load(Ordering::Acquire),
        }
    }

    /// Returns Ok(()) if the caller may proceed. Caller must invoke
    /// record_success or record_failure once the operation completes. Open
    /// breakers transition to HalfOpen when the reset timeout has elapsed
    pub fn try_acquire(&self) -> Result<()> {
        match self.state() {
            CircuitState::Closed => Ok(()),
            CircuitState::HalfOpen => {
                let cur = self.half_open_inflight.fetch_add(1, Ordering::AcqRel);
                if cur >= self.half_open_max_calls.load(Ordering::Acquire) {
                    self.half_open_inflight.fetch_sub(1, Ordering::AcqRel);
                    return Err(ZyronError::ExecutionError(
                        "circuit breaker open: half-open quota exhausted".to_string(),
                    ));
                }
                Ok(())
            }
            CircuitState::Open => {
                let last = self.last_open_ms.load(Ordering::Acquire);
                let now = self.now_ms();
                let timeout = self.reset_timeout.load(Ordering::Acquire);
                if now.saturating_sub(last) >= timeout {
                    // Try to transition to HalfOpen
                    if self
                        .state
                        .compare_exchange(
                            CircuitState::Open as u8,
                            CircuitState::HalfOpen as u8,
                            Ordering::AcqRel,
                            Ordering::Acquire,
                        )
                        .is_ok()
                    {
                        self.half_open_inflight.store(0, Ordering::Release);
                        self.success_count.store(0, Ordering::Release);
                        self.half_open_inflight.fetch_add(1, Ordering::AcqRel);
                        return Ok(());
                    }
                    // Lost the race, retry the dispatch
                    self.try_acquire()
                } else {
                    Err(ZyronError::ExecutionError(
                        "circuit breaker open".to_string(),
                    ))
                }
            }
        }
    }

    pub fn record_success(&self) {
        match self.state() {
            CircuitState::HalfOpen => {
                self.half_open_inflight.fetch_sub(1, Ordering::AcqRel);
                let prev = self.success_count.fetch_add(1, Ordering::AcqRel);
                if prev + 1 >= self.success_threshold.load(Ordering::Acquire) {
                    // Close the breaker
                    self.state
                        .store(CircuitState::Closed as u8, Ordering::Release);
                    self.failure_count.store(0, Ordering::Release);
                    self.success_count.store(0, Ordering::Release);
                }
            }
            CircuitState::Closed => {
                // Reset failure count on a clean run, optional but matches
                // standard behaviour for count-based breakers
                self.failure_count.store(0, Ordering::Release);
            }
            CircuitState::Open => {}
        }
    }

    pub fn record_failure(&self) {
        match self.state() {
            CircuitState::HalfOpen => {
                self.half_open_inflight.fetch_sub(1, Ordering::AcqRel);
                self.state
                    .store(CircuitState::Open as u8, Ordering::Release);
                self.last_open_ms.store(self.now_ms(), Ordering::Release);
            }
            CircuitState::Closed => {
                let prev = self.failure_count.fetch_add(1, Ordering::AcqRel);
                if prev + 1 >= self.failure_threshold.load(Ordering::Acquire) {
                    self.state
                        .store(CircuitState::Open as u8, Ordering::Release);
                    self.last_open_ms.store(self.now_ms(), Ordering::Release);
                }
            }
            CircuitState::Open => {}
        }
    }

    pub fn set_failure_threshold(&self, threshold: u32) {
        self.failure_threshold
            .store(threshold.max(1), Ordering::Release);
    }

    pub fn set_reset_timeout(&self, timeout: Duration) {
        self.reset_timeout
            .store(timeout.as_millis() as u64, Ordering::Release);
    }

    pub fn set_half_open_max_calls(&self, max_calls: u32) {
        let m = max_calls.max(1);
        self.half_open_max_calls.store(m, Ordering::Release);
        self.success_threshold.store(m, Ordering::Release);
    }

    pub fn reset(&self) {
        self.state
            .store(CircuitState::Closed as u8, Ordering::Release);
        self.failure_count.store(0, Ordering::Release);
        self.success_count.store(0, Ordering::Release);
        self.half_open_inflight.store(0, Ordering::Release);
    }
}

// ---------------------------------------------------------------------------
// Adaptive (EMA) CircuitBreaker (item M)
// ---------------------------------------------------------------------------

/// Adaptive breaker driven by EMA of error rate and EMA of latency. Opens
/// when error_ema > error_threshold OR latency_ema > latency_threshold
pub struct EmaCircuitBreaker {
    state: AtomicU8,
    error_ema_x1000: AtomicU32,
    latency_ema_us: AtomicU64,
    alpha_x1000: AtomicU32,
    error_threshold_x1000: AtomicU32,
    latency_threshold_us: AtomicU64,
    reset_timeout_ms: AtomicU64,
    last_open_ms: AtomicU64,
    baseline: Instant,
}

impl EmaCircuitBreaker {
    /// alpha is 0..1, expressed as parts per thousand. error_threshold is the
    /// fraction of failed requests over recent activity above which the
    /// breaker opens, expressed as parts per thousand. latency_threshold is
    /// the EMA latency in microseconds above which the breaker opens
    pub fn new(
        alpha_per_mille: u32,
        error_threshold_per_mille: u32,
        latency_threshold: Duration,
        reset_timeout: Duration,
    ) -> Self {
        Self {
            state: AtomicU8::new(CircuitState::Closed as u8),
            error_ema_x1000: AtomicU32::new(0),
            latency_ema_us: AtomicU64::new(0),
            alpha_x1000: AtomicU32::new(alpha_per_mille.min(1000).max(1)),
            error_threshold_x1000: AtomicU32::new(error_threshold_per_mille.min(1000)),
            latency_threshold_us: AtomicU64::new(latency_threshold.as_micros() as u64),
            reset_timeout_ms: AtomicU64::new(reset_timeout.as_millis() as u64),
            last_open_ms: AtomicU64::new(0),
            baseline: Instant::now(),
        }
    }

    fn now_ms(&self) -> u64 {
        self.baseline.elapsed().as_millis() as u64
    }

    pub fn state(&self) -> CircuitState {
        CircuitState::from_u8(self.state.load(Ordering::Acquire))
    }

    /// Records the outcome of an operation. error is true on failure, latency
    /// is observed wall-clock duration
    pub fn record(&self, error: bool, latency: Duration) {
        let alpha = self.alpha_x1000.load(Ordering::Acquire) as u64;
        // EMA on error_x1000 = alpha * sample + (1-alpha) * old. sample is
        // 0 or 1000
        let sample_e = if error { 1000u64 } else { 0u64 };
        loop {
            let cur = self.error_ema_x1000.load(Ordering::Acquire) as u64;
            let new = (alpha * sample_e + (1000 - alpha) * cur) / 1000;
            if self
                .error_ema_x1000
                .compare_exchange_weak(cur as u32, new as u32, Ordering::AcqRel, Ordering::Acquire)
                .is_ok()
            {
                break;
            }
        }
        let sample_l = latency.as_micros() as u64;
        loop {
            let cur = self.latency_ema_us.load(Ordering::Acquire);
            let new = (alpha * sample_l + (1000 - alpha) * cur) / 1000;
            if self
                .latency_ema_us
                .compare_exchange_weak(cur, new, Ordering::AcqRel, Ordering::Acquire)
                .is_ok()
            {
                break;
            }
        }
        self.maybe_open();
    }

    fn maybe_open(&self) {
        if self.state() == CircuitState::Open {
            return;
        }
        let err = self.error_ema_x1000.load(Ordering::Acquire);
        let lat = self.latency_ema_us.load(Ordering::Acquire);
        let err_thresh = self.error_threshold_x1000.load(Ordering::Acquire);
        let lat_thresh = self.latency_threshold_us.load(Ordering::Acquire);
        if err >= err_thresh || lat >= lat_thresh {
            self.state
                .store(CircuitState::Open as u8, Ordering::Release);
            self.last_open_ms.store(self.now_ms(), Ordering::Release);
        }
    }

    pub fn try_acquire(&self) -> Result<()> {
        match self.state() {
            CircuitState::Closed | CircuitState::HalfOpen => Ok(()),
            CircuitState::Open => {
                let elapsed = self
                    .now_ms()
                    .saturating_sub(self.last_open_ms.load(Ordering::Acquire));
                if elapsed >= self.reset_timeout_ms.load(Ordering::Acquire) {
                    self.state
                        .store(CircuitState::HalfOpen as u8, Ordering::Release);
                    Ok(())
                } else {
                    Err(ZyronError::ExecutionError(
                        "ema circuit breaker open".to_string(),
                    ))
                }
            }
        }
    }

    pub fn close(&self) {
        self.state
            .store(CircuitState::Closed as u8, Ordering::Release);
        self.error_ema_x1000.store(0, Ordering::Release);
        self.latency_ema_us.store(0, Ordering::Release);
    }
}

// ---------------------------------------------------------------------------
// Registry
// ---------------------------------------------------------------------------

pub struct CircuitBreakerRegistry {
    inner: scc::HashMap<String, Arc<CircuitBreaker>>,
}

impl Default for CircuitBreakerRegistry {
    fn default() -> Self {
        Self::new()
    }
}

impl CircuitBreakerRegistry {
    pub fn new() -> Self {
        Self {
            inner: scc::HashMap::new(),
        }
    }

    pub fn get_or_create(
        &self,
        name: &str,
        failure_threshold: u32,
        reset_timeout: Duration,
        half_open_max_calls: u32,
    ) -> Arc<CircuitBreaker> {
        if let Some(found) = self.inner.read_sync(name, |_, b| Arc::clone(b)) {
            return found;
        }
        let new_b = Arc::new(CircuitBreaker::new(
            failure_threshold,
            reset_timeout,
            half_open_max_calls,
        ));
        match self.inner.entry_sync(name.to_string()) {
            scc::hash_map::Entry::Occupied(o) => Arc::clone(o.get()),
            scc::hash_map::Entry::Vacant(v) => {
                v.insert_entry(Arc::clone(&new_b));
                new_b
            }
        }
    }

    pub fn get(&self, name: &str) -> Option<Arc<CircuitBreaker>> {
        self.inner.read_sync(name, |_, b| Arc::clone(b))
    }

    pub fn list(&self) -> Vec<(String, CircuitStatus)> {
        let mut out = Vec::new();
        self.inner.iter_sync(|k, v| {
            out.push((k.clone(), v.snapshot()));
            true
        });
        out
    }
}

// ---------------------------------------------------------------------------
// retry
// ---------------------------------------------------------------------------

/// Runs action up to max_attempts times. Backoff doubles each attempt up to
/// max_delay, with +/- 25 percent jitter. Returns the final error if all
/// attempts fail
pub fn retry<T, E, F>(
    mut action: F,
    max_attempts: u32,
    base_delay: Duration,
    max_delay: Duration,
) -> std::result::Result<T, E>
where
    F: FnMut() -> std::result::Result<T, E>,
{
    let mut delay = base_delay;
    let mut last_err: Option<E> = None;
    for attempt in 0..max_attempts {
        match action() {
            Ok(v) => return Ok(v),
            Err(e) => {
                last_err = Some(e);
                if attempt + 1 >= max_attempts {
                    break;
                }
                let jittered = jitter(delay);
                std::thread::sleep(jittered);
                delay = (delay * 2).min(max_delay);
            }
        }
    }
    Err(last_err.expect("at least one attempt was made"))
}

fn jitter(d: Duration) -> Duration {
    let mut rng = rand::rng();
    let factor: f64 = rng.random_range(0.75..1.25);
    let nanos = (d.as_nanos() as f64 * factor) as u64;
    Duration::from_nanos(nanos)
}

// ---------------------------------------------------------------------------
// timeout_blocking
// ---------------------------------------------------------------------------

/// Runs action on a scoped thread, returning its result if it finishes
/// before duration. Returns ExecutionError on timeout. Does not interrupt the
/// underlying action; the user is expected to make actions cooperative if
/// they need true cancellation
pub fn timeout_blocking<T, F>(action: F, duration: Duration) -> Result<T>
where
    T: Send + 'static,
    F: FnOnce() -> Result<T> + Send + 'static,
{
    let (tx, rx) = crossbeam::channel::bounded::<Result<T>>(1);
    std::thread::spawn(move || {
        let _ = tx.send(action());
    });
    match rx.recv_timeout(duration) {
        Ok(result) => result,
        Err(_) => Err(ZyronError::ExecutionError(format!(
            "operation timed out after {} ms",
            duration.as_millis()
        ))),
    }
}

// ---------------------------------------------------------------------------
// hedged (item N)
// ---------------------------------------------------------------------------

/// Kicks off action in the foreground. If it has not returned after p95,
/// kicks off a duplicate, then returns whichever wins
pub fn hedged<T, F>(action: F, p95: Duration, max_total: Duration) -> Result<T>
where
    T: Send + 'static,
    F: Fn() -> Result<T> + Send + Sync + 'static,
{
    let action = Arc::new(action);
    let (tx, rx) = crossbeam::channel::bounded::<Result<T>>(2);
    let action_a = Arc::clone(&action);
    let tx_a = tx.clone();
    std::thread::spawn(move || {
        let _ = tx_a.send(action_a());
    });
    // Wait p95 then optionally hedge
    let first = rx.recv_timeout(p95);
    if let Ok(v) = first {
        return v;
    }
    let action_b = Arc::clone(&action);
    let tx_b = tx.clone();
    std::thread::spawn(move || {
        let _ = tx_b.send(action_b());
    });
    drop(tx);
    let remaining = max_total.saturating_sub(p95);
    rx.recv_timeout(remaining)
        .map_err(|_| ZyronError::ExecutionError("hedged operation timed out".to_string()))?
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn closed_passes_through() {
        let cb = CircuitBreaker::new(3, Duration::from_secs(1), 1);
        assert_eq!(cb.state(), CircuitState::Closed);
        assert!(cb.try_acquire().is_ok());
    }

    #[test]
    fn opens_after_threshold() {
        let cb = CircuitBreaker::new(3, Duration::from_millis(50), 1);
        cb.record_failure();
        cb.record_failure();
        assert_eq!(cb.state(), CircuitState::Closed);
        cb.record_failure();
        assert_eq!(cb.state(), CircuitState::Open);
        assert!(cb.try_acquire().is_err());
    }

    #[test]
    fn half_open_then_closes_on_success() {
        let cb = CircuitBreaker::new(2, Duration::from_millis(20), 1);
        cb.record_failure();
        cb.record_failure();
        assert_eq!(cb.state(), CircuitState::Open);
        std::thread::sleep(Duration::from_millis(30));
        assert!(cb.try_acquire().is_ok());
        assert_eq!(cb.state(), CircuitState::HalfOpen);
        cb.record_success();
        assert_eq!(cb.state(), CircuitState::Closed);
    }

    #[test]
    fn half_open_failure_reopens() {
        let cb = CircuitBreaker::new(1, Duration::from_millis(10), 1);
        cb.record_failure();
        std::thread::sleep(Duration::from_millis(20));
        assert!(cb.try_acquire().is_ok());
        cb.record_failure();
        assert_eq!(cb.state(), CircuitState::Open);
    }

    #[test]
    fn registry_round_trip() {
        let reg = CircuitBreakerRegistry::new();
        let b = reg.get_or_create("api", 3, Duration::from_secs(1), 1);
        assert!(Arc::ptr_eq(
            &b,
            &reg.get_or_create("api", 3, Duration::from_secs(1), 1)
        ));
        assert_eq!(reg.get("api").unwrap().state(), CircuitState::Closed);
        let listed = reg.list();
        assert_eq!(listed.len(), 1);
        assert_eq!(listed[0].0, "api");
    }

    #[test]
    fn ema_breaker_opens_on_high_error_rate() {
        let cb =
            EmaCircuitBreaker::new(500, 200, Duration::from_secs(1), Duration::from_millis(50));
        for _ in 0..10 {
            cb.record(true, Duration::from_millis(1));
        }
        assert_eq!(cb.state(), CircuitState::Open);
    }

    #[test]
    fn ema_breaker_opens_on_high_latency() {
        let cb = EmaCircuitBreaker::new(
            900,
            900, // very tolerant on errors
            Duration::from_millis(100),
            Duration::from_millis(50),
        );
        // Repeated high latency observations push the EMA up
        for _ in 0..5 {
            cb.record(false, Duration::from_millis(500));
        }
        assert_eq!(cb.state(), CircuitState::Open);
    }

    #[test]
    fn retry_eventually_succeeds() {
        let mut count = 0;
        let r: std::result::Result<i32, &'static str> = retry(
            || {
                count += 1;
                if count < 3 { Err("transient") } else { Ok(42) }
            },
            5,
            Duration::from_micros(1),
            Duration::from_millis(1),
        );
        assert_eq!(r, Ok(42));
        assert_eq!(count, 3);
    }

    #[test]
    fn retry_exhausts_attempts() {
        let mut count = 0;
        let r: std::result::Result<i32, &'static str> = retry(
            || {
                count += 1;
                Err("nope")
            },
            3,
            Duration::from_micros(1),
            Duration::from_millis(1),
        );
        assert_eq!(r, Err("nope"));
        assert_eq!(count, 3);
    }

    #[test]
    fn timeout_returns_value() {
        let r: Result<i32> = timeout_blocking(|| Ok(7), Duration::from_millis(500));
        assert_eq!(r.unwrap(), 7);
    }

    #[test]
    fn timeout_returns_error_on_overrun() {
        let r: Result<i32> = timeout_blocking(
            || {
                std::thread::sleep(Duration::from_millis(200));
                Ok(0)
            },
            Duration::from_millis(20),
        );
        assert!(r.is_err());
    }

    #[test]
    fn hedged_returns_first_completion() {
        let r: Result<i32> = hedged(
            || Ok(99),
            Duration::from_millis(50),
            Duration::from_millis(500),
        );
        assert_eq!(r.unwrap(), 99);
    }
}
