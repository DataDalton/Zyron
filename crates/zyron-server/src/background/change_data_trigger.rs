// -----------------------------------------------------------------------------
// Change data trigger loop.
//
// Runs the pipelines whose ON CHANGE DATA trigger fires, the stream they
// watch holds at least MIN ROWS pending changes, or MAX WAIT has passed with
// at least one. The pending count is read off the feeds' counters, so a
// tick costs one counter compare per triggered pipeline and touches no
// position. The tick is short so a run starts within a second of the
// condition being met.
// -----------------------------------------------------------------------------

use std::sync::Arc;
use std::sync::atomic::AtomicBool;
use std::time::Duration;

use tracing::warn;

use zyron_wire::change_data_trigger::{TriggerWatch, run_due_pipelines};
use zyron_wire::connection::ServerState;

pub const DEFAULT_INTERVAL_SECS: u64 = 1;

pub async fn change_data_trigger_loop(
    server: Arc<ServerState>,
    shutdown: Arc<AtomicBool>,
    wake: Arc<tokio::sync::Notify>,
    interval_secs: u64,
) {
    let mut ticker = tokio::time::interval(Duration::from_secs(interval_secs.max(1)));
    let mut watch = TriggerWatch::new();
    while super::tick_until_shutdown(&mut ticker, &shutdown, &wake).await {
        run_once(&server, &mut watch).await;
    }
}

/// One pass. On a node standing alone or leading its group, runs every
/// pipeline whose trigger fires, and answers with how many ran.
///
/// A pipeline's stages write rows, which reach the rest of a group the way
/// any statement's do, from the member that leads. A follower's pass
/// decides nothing so a run happens once
pub async fn run_once(server: &Arc<ServerState>, watch: &mut TriggerWatch) -> usize {
    if !server.raft.as_ref().map_or(true, |raft| raft.is_leader()) {
        return 0;
    }
    // A stage moves its stream's position in a commit every member applies.
    // While a member runs a binary that does not read the advance, every
    // run would do its work and be refused at commit, so runs wait until
    // the group carries it
    if server
        .replication
        .as_ref()
        .is_some_and(|router| !router.carries_stream_advance())
    {
        return 0;
    }
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_micros() as i64)
        .unwrap_or(0);
    let outcomes = run_due_pipelines(server, watch, now).await;
    for (name, outcome) in &outcomes {
        if let Err(e) = outcome {
            warn!(
                target: "zyron::cdc",
                pipeline = %name,
                "a pipeline run on change data failed: {e}"
            );
        }
    }
    outcomes.len()
}
