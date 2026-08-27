//! Reads the pages a departing node asked this one to warm.
//!
//! The mesh call queues page ids and returns, because reading a quarter of a
//! gigabyte inside a request handler would hold the connection open for the
//! length of the read and do it on the listener's thread. This is what picks
//! the queue up.
//!
//! ## Why it never evicts
//!
//! The buffer pool's prefetch declines pages once the free frames are down to
//! its reserve, and that is the property that makes this safe to run while the
//! node is serving. A prefetch that displaced resident pages would trade this
//! node's own warm set, which is definitely being used, for a guess about
//! traffic it is about to inherit. Warming is worth doing only while it is
//! free.
//!
//! ## Why it polls
//!
//! A handover happens once per scale-in, so there is nothing to wake on for
//! hours at a time and a notification channel would be a channel that is
//! almost always empty. The interval is short enough that the pages arrive
//! before the traffic does, which is the only deadline this has.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;

use zyron_buffer::BufferPool;
use zyron_common::page::PageId;
use zyron_storage::DiskManager;

use crate::mesh_node::ServerMeshNode;

/// How often the queue is looked at.
pub const DEFAULT_INTERVAL_MS: u64 = 200;

/// Drains the prefetch queue into the buffer pool until shutdown.
pub async fn mesh_prefetch_loop(
    node: Arc<ServerMeshNode>,
    pool: Arc<BufferPool>,
    disk: Arc<DiskManager>,
    shutdown: Arc<AtomicBool>,
    interval_ms: u64,
) {
    let interval = Duration::from_millis(interval_ms.max(1));
    while !shutdown.load(Ordering::Relaxed) {
        tokio::time::sleep(interval).await;
        let queued = node.take_prefetch_queue();
        if queued.is_empty() {
            continue;
        }
        let pages: Vec<PageId> = queued.into_iter().map(PageId::from_u64).collect();
        // The pool decides page by page whether it has room, so the read has
        // to happen inside its loop rather than ahead of it: reading every
        // page first would pull in the ones it is about to decline. The
        // synchronous read is what makes that possible, and this runs on a
        // blocking-tolerant worker so the runtime keeps serving
        let disk_reader = Arc::clone(&disk);
        let warm_pool = Arc::clone(&pool);
        let report = tokio::task::block_in_place(move || {
            warm_pool.prefetch(&pages, move |page_id| {
                disk_reader
                    .read_page_sync(page_id)
                    .ok()
                    .map(|bytes| bytes.to_vec())
            })
        });
        tracing::info!(
            requested = report.requested,
            loaded = report.loaded,
            already_resident = report.already_resident,
            declined = report.declined,
            failed = report.failed,
            coverage = report.coverage(),
            "warmed pages handed over by a departing node"
        );
    }
}
