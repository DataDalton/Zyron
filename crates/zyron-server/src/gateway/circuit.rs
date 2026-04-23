// -----------------------------------------------------------------------------
// Per-endpoint circuit breaker.
//
// Three states: Closed (allow traffic), Open (reject), HalfOpen (probe). The
// breaker flips to Open after `threshold` consecutive failures. After
// `open_for`, the next request probes the endpoint. A probe success closes the
// breaker. A probe failure reopens it.
// -----------------------------------------------------------------------------

use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};
use std::time::{Duration, Instant};

/// State of the breaker at a point in time.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CircuitState {
    Closed,
    Open,
    HalfOpen,
}

/// Circuit breaker with atomic state transitions.
pub struct CircuitBreaker {
    consecutive_failures: AtomicU32,
    state_tag: AtomicU32, // 0=Closed, 1=Open, 2=HalfOpen
    open_until_micros: AtomicU64,
    threshold: u32,
    open_for: Duration,
    start: Instant,
}

impl CircuitBreaker {
    pub fn new(threshold: u32, open_for: Duration) -> Self {
        Self {
            consecutive_failures: AtomicU32::new(0),
            state_tag: AtomicU32::new(0),
            open_until_micros: AtomicU64::new(0),
            threshold,
            open_for,
            start: Instant::now(),
        }
    }

    pub fn state(&self) -> CircuitState {
        match self.state_tag.load(Ordering::Acquire) {
            0 => CircuitState::Closed,
            1 => CircuitState::Open,
            _ => CircuitState::HalfOpen,
        }
    }

    /// Returns true when the caller may attempt the downstream call.
    pub fn should_attempt(&self) -> bool {
        match self.state() {
            CircuitState::Closed | CircuitState::HalfOpen => true,
            CircuitState::Open => {
                let now = self.start.elapsed().as_micros() as u64;
                let until = self.open_until_micros.load(Ordering::Acquire);
                if now >= until {
                    self.state_tag.store(2, Ordering::Release);
                    true
                } else {
                    false
                }
            }
        }
    }

    pub fn record_success(&self) {
        self.consecutive_failures.store(0, Ordering::Release);
        self.state_tag.store(0, Ordering::Release);
    }

    pub fn record_failure(&self) {
        let prior = self.consecutive_failures.fetch_add(1, Ordering::AcqRel);
        if prior + 1 >= self.threshold {
            let now = self.start.elapsed().as_micros() as u64;
            let until = now + self.open_for.as_micros() as u64;
            self.open_until_micros.store(until, Ordering::Release);
            self.state_tag.store(1, Ordering::Release);
        }
    }

    /// Snapshot used by metrics and the _health endpoint.
    pub fn snapshot(&self) -> (CircuitState, u32) {
        (
            self.state(),
            self.consecutive_failures.load(Ordering::Relaxed),
        )
    }
}

impl std::fmt::Debug for CircuitBreaker {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CircuitBreaker")
            .field("state", &self.state())
            .field("threshold", &self.threshold)
            .field(
                "failures",
                &self.consecutive_failures.load(Ordering::Relaxed),
            )
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::thread::sleep;

    #[test]
    fn closed_by_default() {
        let cb = CircuitBreaker::new(3, Duration::from_millis(50));
        assert!(cb.should_attempt());
        assert_eq!(cb.state(), CircuitState::Closed);
    }

    #[test]
    fn opens_after_threshold() {
        let cb = CircuitBreaker::new(2, Duration::from_secs(30));
        cb.record_failure();
        assert_eq!(cb.state(), CircuitState::Closed);
        cb.record_failure();
        assert_eq!(cb.state(), CircuitState::Open);
        assert!(!cb.should_attempt());
    }

    #[test]
    fn half_open_after_timeout() {
        let cb = CircuitBreaker::new(1, Duration::from_millis(10));
        cb.record_failure();
        assert_eq!(cb.state(), CircuitState::Open);
        sleep(Duration::from_millis(20));
        assert!(cb.should_attempt());
        assert_eq!(cb.state(), CircuitState::HalfOpen);
    }

    #[test]
    fn success_resets() {
        let cb = CircuitBreaker::new(3, Duration::from_secs(1));
        cb.record_failure();
        cb.record_failure();
        cb.record_success();
        assert_eq!(cb.state(), CircuitState::Closed);
        assert_eq!(cb.snapshot().1, 0);
    }
}
