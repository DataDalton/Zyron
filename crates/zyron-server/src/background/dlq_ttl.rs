// -----------------------------------------------------------------------------
// Dead-letter queue TTL sweeper.
//
// Once per day evicts rows older than the configured TTL window from every
// registered DLQ so failed replication work does not accumulate forever.
// -----------------------------------------------------------------------------

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;

pub const DEFAULT_INTERVAL_SECS: u64 = 86400;

/// Background TTL sweeper. The on_tick callback receives the cutoff timestamp
/// computed from ttl_days and owns the eviction implementation so this loop
/// stays generic over the DLQ backend.
pub async fn dlq_ttl_loop<F>(
    shutdown: Arc<AtomicBool>,
    interval_secs: u64,
    ttl_days: u32,
    mut on_tick: F,
) where
    F: FnMut(u64) + Send + 'static,
{
    let mut ticker = tokio::time::interval(Duration::from_secs(interval_secs.max(3600)));
    loop {
        ticker.tick().await;
        if shutdown.load(Ordering::Acquire) {
            break;
        }
        let cutoff = current_secs().saturating_sub(ttl_days as u64 * 86400);
        on_tick(cutoff);
    }
}

fn current_secs() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ttl_cutoff_computation() {
        let now = current_secs();
        let ttl_days: u32 = 30;
        let cutoff = now.saturating_sub(ttl_days as u64 * 86400);
        assert!(cutoff < now);
        assert!(now - cutoff == 30 * 86400);
    }

    #[test]
    fn default_interval_one_day() {
        assert_eq!(DEFAULT_INTERVAL_SECS, 86400);
    }
}
