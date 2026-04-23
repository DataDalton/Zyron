// -----------------------------------------------------------------------------
// Credential cache refresh worker.
//
// Periodically asks the credential cache to refresh every entry whose
// expiration falls inside the refresh window. Keeping refreshes off the
// critical path avoids expiry-induced stalls at query time.
// -----------------------------------------------------------------------------

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;

pub const DEFAULT_INTERVAL_SECS: u64 = 60;
pub const DEFAULT_REFRESH_WINDOW_SECS: u64 = 900;

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;
    use std::sync::atomic::AtomicBool;

    #[tokio::test(flavor = "current_thread")]
    async fn loop_exits_on_shutdown() {
        let shutdown = Arc::new(AtomicBool::new(true));
        let called = Arc::new(std::sync::atomic::AtomicU32::new(0));
        let called_clone = Arc::clone(&called);
        credential_refresh_loop(
            shutdown,
            15,
            std::time::Duration::from_secs(60),
            move |_| {
                called_clone.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            },
        )
        .await;
        assert_eq!(called.load(std::sync::atomic::Ordering::Relaxed), 0);
    }
}

/// Credential cache refresh tick. The inner probe closure is passed in so the
/// worker stays agnostic of the specific provider registry.
pub async fn credential_refresh_loop<F>(
    shutdown: Arc<AtomicBool>,
    interval_secs: u64,
    refresh_window: Duration,
    mut on_tick: F,
) where
    F: FnMut(Duration) + Send + 'static,
{
    let mut ticker = tokio::time::interval(Duration::from_secs(interval_secs.max(15)));
    loop {
        ticker.tick().await;
        if shutdown.load(Ordering::Acquire) {
            break;
        }
        on_tick(refresh_window);
    }
}
