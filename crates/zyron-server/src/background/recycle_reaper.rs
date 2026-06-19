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
    interval_secs: u64,
) {
    let mut ticker = tokio::time::interval(Duration::from_secs(interval_secs.max(10)));
    loop {
        ticker.tick().await;
        if shutdown.load(Ordering::Acquire) {
            break;
        }
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
        if finalize_one(server, entry.id).await {
            purged += 1;
        }
    }
    purged
}

/// Finalizes a single recycled table: drops its index entries and table entry
/// from the catalog, then reclaims the in-memory index handles and on-disk
/// heap, FSM, and index files. Returns true when the table was purged.
async fn finalize_one(server: &Arc<ServerState>, table_id: zyron_catalog::TableId) -> bool {
    // Capture the index set before finalize removes the catalog entries.
    let indexes = server.catalog.get_indexes_for_table(table_id);

    let entry = match server.catalog.finalize_dropped_table(table_id).await {
        Ok(Some(e)) => e,
        Ok(None) => return false,
        Err(e) => {
            warn!(
                target: "zyron::recycle",
                table_id = table_id.0,
                "failed to finalize recycled table: {e}"
            );
            return false;
        }
    };

    // Drop in-memory index handles and delete index files so the checkpoint
    // worker does not resurrect a file for a now-gone index.
    for idx in &indexes {
        match idx.index_type {
            zyron_catalog::IndexType::BTree => {
                let _ = server.btree_indexes.remove_async(&idx.id.0).await;
            }
            zyron_catalog::IndexType::Fulltext => {
                if let Some(m) = &server.fts_manager {
                    let _ = m.drop_index(idx.id.0);
                }
            }
            zyron_catalog::IndexType::Vector => {
                if let Some(m) = &server.vector_manager {
                    let _ = m.drop_index(idx.id.0);
                }
            }
            zyron_catalog::IndexType::Spatial => {
                if let Some(m) = &server.spatial_manager {
                    m.drop_index(idx.id.0);
                }
            }
        }
        if let Err(e) = server.disk_manager.delete_file(idx.index_file_id).await {
            warn!(
                target: "zyron::recycle",
                index_file_id = idx.index_file_id,
                "failed to remove index file: {e}"
            );
        }
    }

    // Reclaim the heap and FSM files the recycle bin was holding.
    let _ = server.heap_files.remove_async(&entry.heap_file_id).await;
    if let Err(e) = server.disk_manager.delete_file(entry.heap_file_id).await {
        warn!(
            target: "zyron::recycle",
            heap_file_id = entry.heap_file_id,
            "failed to remove heap file: {e}"
        );
    }
    if let Err(e) = server.disk_manager.delete_file(entry.fsm_file_id).await {
        warn!(
            target: "zyron::recycle",
            fsm_file_id = entry.fsm_file_id,
            "failed to remove FSM file: {e}"
        );
    }

    info!(
        target: "zyron::recycle",
        table_id = table_id.0,
        name = %entry.name,
        "purged recycled table after window elapsed"
    );
    true
}

fn current_secs() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}
