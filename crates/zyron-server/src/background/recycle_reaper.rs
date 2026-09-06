// -----------------------------------------------------------------------------
// Recycle-bin reaper.
//
// Tables dropped with a configured recycle window are soft-dropped: hidden from
// lookups but kept intact so UNDROP TABLE can restore them. This worker finds
// soft-dropped tables whose window has elapsed and finalizes the drop, removing
// the catalog entry plus the table's index entries and reclaiming the backing
// heap, FSM, and index files along with their in-memory index handles.
// -----------------------------------------------------------------------------

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;

use tracing::{info, warn};
use zyron_wire::connection::ServerState;

pub const DEFAULT_INTERVAL_SECS: u64 = 60;

/// At most one reaper pass runs at a time so two ticks cannot double-finalize
/// the same table.
static REAPER_PASS_IN_PROGRESS: AtomicBool = AtomicBool::new(false);

struct ReaperPassGuard;
impl Drop for ReaperPassGuard {
    fn drop(&mut self) {
        REAPER_PASS_IN_PROGRESS.store(false, Ordering::Release);
    }
}

pub async fn recycle_reaper_loop(
    server: Arc<ServerState>,
    shutdown: Arc<AtomicBool>,
    wake: Arc<tokio::sync::Notify>,
    interval_secs: u64,
) {
    let mut ticker = tokio::time::interval(Duration::from_secs(interval_secs.max(10)));
    while super::tick_until_shutdown(&mut ticker, &shutdown, &wake).await {
        run_reaper_once(&server).await;
    }
}

/// Finalizes every soft-dropped table whose recycle window has elapsed.
/// Returns the number of tables purged.
pub async fn run_reaper_once(server: &Arc<ServerState>) -> usize {
    if REAPER_PASS_IN_PROGRESS
        .compare_exchange(false, true, Ordering::Acquire, Ordering::Relaxed)
        .is_err()
    {
        return 0;
    }
    let _guard = ReaperPassGuard;
    let now = current_secs();
    let mut purged = 0usize;

    for entry in server.catalog.list_dropped_tables() {
        let Some(dropped_at) = entry.dropped_at else {
            continue;
        };
        let window = entry.lifecycle.recycle_window_seconds.max(0) as u64;
        if now < dropped_at.saturating_add(window) {
            continue;
        }
        // The shared finalize path removes the catalog rows and reclaims
        // the storage: index handles and files, heap and FSM files, the
        // lake and columnar tiers. It is the same path DROP SCHEMA CASCADE
        // purges through, so the two cannot drift apart.
        match zyron_wire::ddl_dispatch::finalize_recycled_table(server, entry.id).await {
            Ok(true) => {
                info!(
                    target: "zyron::recycle",
                    table_id = entry.id.0,
                    name = %entry.name,
                    "purged recycled table after window elapsed"
                );
                purged += 1;
            }
            Ok(false) => {}
            Err(e) => {
                warn!(
                    target: "zyron::recycle",
                    table_id = entry.id.0,
                    "failed to finalize recycled table: {e:?}"
                );
            }
        }
    }
    purged
}

fn current_secs() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}
