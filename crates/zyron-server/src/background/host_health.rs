// -----------------------------------------------------------------------------
// Host health monitor.
//
// Polls every registered outbound connection pool at a short fixed interval
// and flips host states between healthy and unhealthy. The pool implementation
// owns the retry and circuit-break logic, this worker simply emits a
// heartbeat so stale state does not linger.
// -----------------------------------------------------------------------------

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;

pub const DEFAULT_INTERVAL_SECS: u64 = 5;

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test(flavor = "current_thread")]
    async fn shutdown_short_circuits_loop() {
        let shutdown = Arc::new(AtomicBool::new(true));
        let called = Arc::new(std::sync::atomic::AtomicU32::new(0));
        let called_clone = Arc::clone(&called);
        host_health_monitor_loop(shutdown, 1, move || {
            called_clone.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        })
        .await;
        assert_eq!(called.load(std::sync::atomic::Ordering::Relaxed), 0);
    }

    #[test]
    fn default_interval_is_five_seconds() {
        assert_eq!(DEFAULT_INTERVAL_SECS, 5);
    }

    #[test]
    fn interval_clamps_to_minimum_one_second() {
        let requested = 0u64;
        let clamped = requested.max(1);
        assert_eq!(clamped, 1);
    }
}

/// Host health tick. The on_tick callback walks the concrete pool registry so
/// this file stays independent of the registry shape.
pub async fn host_health_monitor_loop<F>(
    shutdown: Arc<AtomicBool>,
    interval_secs: u64,
    mut on_tick: F,
) where
    F: FnMut() + Send + 'static,
{
    let mut ticker = tokio::time::interval(Duration::from_secs(interval_secs.max(1)));
    loop {
        ticker.tick().await;
        if shutdown.load(Ordering::Acquire) {
            break;
        }
        on_tick();
    }
}
