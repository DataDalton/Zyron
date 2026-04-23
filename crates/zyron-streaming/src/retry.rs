// -----------------------------------------------------------------------------
// Retry policy, error classification, and circuit breaker
// -----------------------------------------------------------------------------
//
// RetryConfig drives exponential-backoff retry loops in Zyron-to-Zyron sink
// clients. classify_pg_error and classify_io_error sort errors into Transient
// (retry with backoff) and Fatal (fail fast, route to DLQ). CircuitBreaker
// tracks a rolling window of successes and failures and opens on threshold
// breach, blocks calls during cooldown, and probes recovery in half-open.

use std::collections::VecDeque;
use std::sync::atomic::{AtomicU64, AtomicU8, Ordering};
use std::time::{Duration, Instant};

use parking_lot::Mutex;

// -----------------------------------------------------------------------------
// RetryConfig
// -----------------------------------------------------------------------------

/// Parameters for exponential-backoff retry loops. `initial_backoff` doubles
/// on each failure up to `max_backoff`. When `jitter` is true, each sleep is
/// multiplied by a random factor in [1.0, 1.5).
#[derive(Debug, Clone, Copy)]
pub struct RetryConfig {
    pub max_attempts: u32,
    pub initial_backoff: Duration,
    pub max_backoff: Duration,
    pub jitter: bool,
}

impl Default for RetryConfig {
    fn default() -> Self {
        Self {
            max_attempts: 5,
            initial_backoff: Duration::from_millis(100),
            max_backoff: Duration::from_secs(30),
            jitter: true,
        }
    }
}

impl RetryConfig {
    /// Computes the backoff duration for the given zero-based attempt index.
    pub fn backoff_for_attempt(&self, attempt: u32) -> Duration {
        let capped = attempt.min(20);
        let base = self.initial_backoff.saturating_mul(1u32 << capped);
        let limited = if base > self.max_backoff {
            self.max_backoff
        } else {
            base
        };
        if self.jitter {
            // Scale by [1.0, 1.5). Uses a cheap xor-shift seeded from the
            // attempt count and the current instant nanos to avoid a crate
            // dependency on rand from this module.
            let seed = attempt as u64
                ^ Instant::now().elapsed().as_nanos() as u64
                ^ std::process::id() as u64;
            let mut x = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15).wrapping_add(1);
            x ^= x >> 30;
            x = x.wrapping_mul(0xBF58_476D_1CE4_E5B9);
            x ^= x >> 27;
            let frac = ((x >> 32) as f64) / (u32::MAX as f64); // 0.0 .. 1.0
            let mult = 1.0 + frac * 0.5;
            let nanos = (limited.as_nanos() as f64 * mult) as u64;
            Duration::from_nanos(nanos)
        } else {
            limited
        }
    }
}

// -----------------------------------------------------------------------------
// Error classification
// -----------------------------------------------------------------------------

/// Classification of an error surfaced by the remote connection layer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ErrorClass {
    /// Safe to retry after backoff. Network hiccups, serialization, deadlocks.
    Transient,
    /// No retry. Auth denial, schema mismatch, unrecoverable protocol state.
    Fatal,
}

/// Classifies a PostgreSQL SQLSTATE code into Transient or Fatal. The rules
/// mirror the spec's retry policy for Zyron-to-Zyron sinks.
pub fn classify_sqlstate(code: &str) -> ErrorClass {
    // Connection exception class 08.
    if code.starts_with("08") {
        return ErrorClass::Transient;
    }
    // Insufficient resources class 53.
    if code.starts_with("53") {
        return ErrorClass::Transient;
    }
    // Serialization failure, deadlock, cannot connect now.
    if code == "40001" || code == "40P01" || code == "57P03" {
        return ErrorClass::Transient;
    }
    // Invalid authorization class 28.
    if code.starts_with("28") {
        return ErrorClass::Fatal;
    }
    // Privilege or syntax classes 42, 23.
    if code.starts_with("42") || code.starts_with("23") {
        return ErrorClass::Fatal;
    }
    // Default to fatal. Callers that want to retry on unknown codes can
    // wrap this function.
    ErrorClass::Fatal
}

/// Classifies an io::Error. Network drops, resets, timeouts, and closures
/// are transient. Everything else is fatal.
pub fn classify_io_error(err: &std::io::Error) -> ErrorClass {
    use std::io::ErrorKind;
    match err.kind() {
        ErrorKind::ConnectionRefused
        | ErrorKind::ConnectionReset
        | ErrorKind::ConnectionAborted
        | ErrorKind::TimedOut
        | ErrorKind::WouldBlock
        | ErrorKind::Interrupted
        | ErrorKind::BrokenPipe
        | ErrorKind::UnexpectedEof
        | ErrorKind::NotConnected
        | ErrorKind::HostUnreachable
        | ErrorKind::NetworkUnreachable
        | ErrorKind::NetworkDown => ErrorClass::Transient,
        _ => ErrorClass::Fatal,
    }
}

/// Classifies a free-form error message by checking for embedded SQLSTATE
/// tokens like `code: 40001` or by falling back to substring heuristics for
/// TLS, connect, and timeout cases.
pub fn classify_message(msg: &str) -> ErrorClass {
    let lower = msg.to_ascii_lowercase();
    if lower.contains("tls handshake") || lower.contains("cert invalid") {
        return ErrorClass::Fatal;
    }
    if lower.contains("timeout")
        || lower.contains("timed out")
        || lower.contains("connection refused")
        || lower.contains("connection reset")
        || lower.contains("connection closed")
        || lower.contains("broken pipe")
    {
        return ErrorClass::Transient;
    }
    // Look for a 5-character SQLSTATE pattern after "code:".
    if let Some(idx) = lower.find("code:") {
        let rest = &msg[idx + 5..].trim_start();
        let candidate: String = rest.chars().take(5).collect();
        if candidate.len() == 5 && candidate.chars().all(|c| c.is_ascii_alphanumeric()) {
            return classify_sqlstate(&candidate);
        }
    }
    ErrorClass::Fatal
}

// -----------------------------------------------------------------------------
// Circuit breaker
// -----------------------------------------------------------------------------

/// Lifecycle state for the circuit breaker.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[repr(u8)]
pub enum CircuitState {
    #[default]
    Closed = 0,
    Open = 1,
    HalfOpen = 2,
}

impl CircuitState {
    pub fn from_u8(v: u8) -> Self {
        match v {
            0 => CircuitState::Closed,
            1 => CircuitState::Open,
            _ => CircuitState::HalfOpen,
        }
    }
}

/// Fixed-length ring buffer of recent attempt outcomes.
#[derive(Debug)]
struct RollingWindow {
    slots: VecDeque<bool>,
    capacity: usize,
    failures: usize,
}

impl RollingWindow {
    fn new(capacity: usize) -> Self {
        Self {
            slots: VecDeque::with_capacity(capacity.max(1)),
            capacity: capacity.max(1),
            failures: 0,
        }
    }

    fn record(&mut self, success: bool) {
        if self.slots.len() == self.capacity {
            if let Some(old) = self.slots.pop_front() {
                if !old {
                    self.failures = self.failures.saturating_sub(1);
                }
            }
        }
        self.slots.push_back(success);
        if !success {
            self.failures += 1;
        }
    }

    fn error_rate(&self) -> f64 {
        if self.slots.is_empty() {
            return 0.0;
        }
        self.failures as f64 / self.slots.len() as f64
    }

    fn len(&self) -> usize {
        self.slots.len()
    }
}

/// Rolling-window circuit breaker. Opens when the error rate in the window
/// meets or exceeds `threshold` after at least `window_size` samples have
/// been recorded. While open, calls are rejected until `cooldown` has
/// elapsed since the last trip, at which point the breaker transitions to
/// HalfOpen and allows a single probe.
pub struct CircuitBreaker {
    threshold: f64,
    window_size: usize,
    cooldown: Duration,
    outcomes: Mutex<RollingWindow>,
    state: AtomicU8,
    last_trip_epoch_ms: AtomicU64,
}

impl CircuitBreaker {
    pub fn new(threshold: f64, window_size: usize, cooldown: Duration) -> Self {
        Self {
            threshold,
            window_size: window_size.max(1),
            cooldown,
            outcomes: Mutex::new(RollingWindow::new(window_size.max(1))),
            state: AtomicU8::new(CircuitState::Closed as u8),
            last_trip_epoch_ms: AtomicU64::new(0),
        }
    }

    /// Returns the current breaker state.
    pub fn state(&self) -> CircuitState {
        CircuitState::from_u8(self.state.load(Ordering::Acquire))
    }

    /// Returns true if a caller is permitted to attempt the guarded operation
    /// right now. Transitions Open to HalfOpen when the cooldown has elapsed.
    pub fn should_attempt(&self) -> bool {
        match self.state() {
            CircuitState::Closed => true,
            CircuitState::HalfOpen => true,
            CircuitState::Open => {
                let last = self.last_trip_epoch_ms.load(Ordering::Acquire);
                let now = now_epoch_ms();
                if now.saturating_sub(last) >= self.cooldown.as_millis() as u64 {
                    self.state
                        .store(CircuitState::HalfOpen as u8, Ordering::Release);
                    true
                } else {
                    false
                }
            }
        }
    }

    /// Records a successful attempt. Closes a HalfOpen breaker.
    pub fn record_success(&self) {
        self.outcomes.lock().record(true);
        if self.state() == CircuitState::HalfOpen {
            self.state
                .store(CircuitState::Closed as u8, Ordering::Release);
        }
    }

    /// Records a failed attempt. Opens the breaker when the rolling error
    /// rate meets or exceeds the configured threshold over at least
    /// window_size samples. In HalfOpen, a single failure re-opens.
    pub fn record_failure(&self) {
        let (len, rate) = {
            let mut w = self.outcomes.lock();
            w.record(false);
            (w.len(), w.error_rate())
        };
        let should_open = match self.state() {
            CircuitState::HalfOpen => true,
            CircuitState::Closed => len >= self.window_size && rate >= self.threshold,
            CircuitState::Open => false,
        };
        if should_open {
            self.state
                .store(CircuitState::Open as u8, Ordering::Release);
            self.last_trip_epoch_ms
                .store(now_epoch_ms(), Ordering::Release);
        }
    }
}

fn now_epoch_ms() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0)
}

// -----------------------------------------------------------------------------
// Tests
// -----------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn classify_transient_sqlstates() {
        assert_eq!(classify_sqlstate("08000"), ErrorClass::Transient);
        assert_eq!(classify_sqlstate("08006"), ErrorClass::Transient);
        assert_eq!(classify_sqlstate("53100"), ErrorClass::Transient);
        assert_eq!(classify_sqlstate("40001"), ErrorClass::Transient);
        assert_eq!(classify_sqlstate("40P01"), ErrorClass::Transient);
        assert_eq!(classify_sqlstate("57P03"), ErrorClass::Transient);
    }

    #[test]
    fn classify_fatal_sqlstates() {
        assert_eq!(classify_sqlstate("28000"), ErrorClass::Fatal);
        assert_eq!(classify_sqlstate("42501"), ErrorClass::Fatal);
        assert_eq!(classify_sqlstate("23505"), ErrorClass::Fatal);
        assert_eq!(classify_sqlstate("XX000"), ErrorClass::Fatal);
    }

    #[test]
    fn classify_io_errors() {
        let e = std::io::Error::new(std::io::ErrorKind::ConnectionReset, "reset");
        assert_eq!(classify_io_error(&e), ErrorClass::Transient);
        let e = std::io::Error::new(std::io::ErrorKind::TimedOut, "to");
        assert_eq!(classify_io_error(&e), ErrorClass::Transient);
        let e = std::io::Error::new(std::io::ErrorKind::PermissionDenied, "no");
        assert_eq!(classify_io_error(&e), ErrorClass::Fatal);
    }

    #[test]
    fn classify_message_heuristics() {
        assert_eq!(
            classify_message("connect error: connection refused"),
            ErrorClass::Transient
        );
        assert_eq!(
            classify_message("TLS handshake failed: cert invalid"),
            ErrorClass::Fatal
        );
        assert_eq!(
            classify_message("server error code: 40001 serialization"),
            ErrorClass::Transient
        );
        assert_eq!(
            classify_message("server error code: 42501 no privilege"),
            ErrorClass::Fatal
        );
    }

    #[test]
    fn retry_backoff_progression() {
        let cfg = RetryConfig {
            max_attempts: 5,
            initial_backoff: Duration::from_millis(100),
            max_backoff: Duration::from_secs(10),
            jitter: false,
        };
        assert_eq!(cfg.backoff_for_attempt(0), Duration::from_millis(100));
        assert_eq!(cfg.backoff_for_attempt(1), Duration::from_millis(200));
        assert_eq!(cfg.backoff_for_attempt(2), Duration::from_millis(400));
        // Cap kicks in before overflow.
        let capped = cfg.backoff_for_attempt(20);
        assert_eq!(capped, Duration::from_secs(10));
    }

    #[test]
    fn retry_backoff_with_jitter_bounds() {
        let cfg = RetryConfig {
            max_attempts: 3,
            initial_backoff: Duration::from_millis(100),
            max_backoff: Duration::from_secs(10),
            jitter: true,
        };
        for attempt in 0..4 {
            let base = cfg.backoff_for_attempt(attempt);
            // Jitter multiplies the capped base by [1.0, 1.5).
            let min_expected = match attempt {
                0 => Duration::from_millis(100),
                1 => Duration::from_millis(200),
                2 => Duration::from_millis(400),
                _ => Duration::from_millis(800),
            };
            assert!(base >= min_expected);
            let max_expected = min_expected.mul_f64(1.5);
            assert!(base <= max_expected + Duration::from_micros(1));
        }
    }

    #[test]
    fn circuit_breaker_starts_closed() {
        let cb = CircuitBreaker::new(0.5, 10, Duration::from_secs(1));
        assert_eq!(cb.state(), CircuitState::Closed);
        assert!(cb.should_attempt());
    }

    #[test]
    fn circuit_breaker_opens_on_threshold() {
        let cb = CircuitBreaker::new(0.5, 4, Duration::from_secs(1));
        // Trip check only fires inside record_failure, so the window must
        // reach size on a failing call.
        cb.record_success();
        cb.record_success();
        cb.record_failure();
        cb.record_failure();
        // Two failures out of four = 0.5 error rate, meets threshold.
        assert_eq!(cb.state(), CircuitState::Open);
        assert!(!cb.should_attempt());
    }

    #[test]
    fn circuit_breaker_stays_closed_below_threshold() {
        let cb = CircuitBreaker::new(0.5, 4, Duration::from_secs(1));
        cb.record_success();
        cb.record_success();
        cb.record_success();
        cb.record_failure();
        assert_eq!(cb.state(), CircuitState::Closed);
    }

    #[test]
    fn circuit_breaker_half_open_after_cooldown() {
        let cb = CircuitBreaker::new(0.5, 2, Duration::from_millis(30));
        cb.record_failure();
        cb.record_failure();
        assert_eq!(cb.state(), CircuitState::Open);
        std::thread::sleep(Duration::from_millis(40));
        assert!(cb.should_attempt());
        assert_eq!(cb.state(), CircuitState::HalfOpen);
    }

    #[test]
    fn circuit_breaker_half_open_success_closes() {
        let cb = CircuitBreaker::new(0.5, 2, Duration::from_millis(10));
        cb.record_failure();
        cb.record_failure();
        std::thread::sleep(Duration::from_millis(20));
        assert!(cb.should_attempt());
        cb.record_success();
        assert_eq!(cb.state(), CircuitState::Closed);
    }

    #[test]
    fn circuit_breaker_half_open_failure_reopens() {
        let cb = CircuitBreaker::new(0.5, 2, Duration::from_millis(10));
        cb.record_failure();
        cb.record_failure();
        std::thread::sleep(Duration::from_millis(20));
        assert!(cb.should_attempt());
        cb.record_failure();
        assert_eq!(cb.state(), CircuitState::Open);
    }
}
