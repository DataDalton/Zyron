// -----------------------------------------------------------------------------
// Inbound CDC ingest worker.
//
// Polls every active CDC ingest job for new source records (Kafka or S3),
// decodes them, and applies them to the target table. The sweep logic lives in
// zyron-wire's run_due_ingests so it is shared with tests; this worker is the
// timer that drives it. It runs on a dedicated thread with an inner
// current-thread runtime because the sweep both awaits the executor and
// performs synchronous source network IO. The thread exits within one interval
// after the shutdown flag is set.
// -----------------------------------------------------------------------------

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::thread::{self, JoinHandle};
use std::time::Duration;

use tracing::warn;

use zyron_wire::connection::ServerState;

pub const DEFAULT_INTERVAL_SECS: u64 = 5;

/// Spawns the ingest worker thread and returns its join handle.
pub fn start(
    server: Arc<ServerState>,
    shutdown: Arc<AtomicBool>,
    interval_secs: u64,
) -> JoinHandle<()> {
    thread::Builder::new()
        .name("zyron-cdc-ingest".into())
        .spawn(move || {
            let runtime = match tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()
            {
                Ok(rt) => rt,
                Err(e) => {
                    warn!("cdc ingest worker: runtime build failed: {e}");
                    return;
                }
            };
            let interval = Duration::from_secs(interval_secs.max(1));
            loop {
                thread::park_timeout(interval);
                if shutdown.load(Ordering::Acquire) {
                    return;
                }
                if server.cdc_ingest_manager.is_none() {
                    continue;
                }
                let report = runtime.block_on(zyron_wire::ddl_dispatch::run_due_ingests(&server));
                if report.failed > 0 {
                    warn!(
                        "cdc ingest worker: {} record(s) failed this cycle",
                        report.failed
                    );
                }
            }
        })
        .expect("failed to spawn cdc ingest worker thread")
}
