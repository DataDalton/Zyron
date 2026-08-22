// -----------------------------------------------------------------------------
// Lake follower.
//
// Keeps every table that declared a leader caught up with it. One poll per
// table per tick: read the versions it lacks, apply them, record freshness.
//
// Two properties this worker must not break. A follower never blocks a
// reader: a leader that is unreachable leaves the table exactly where it
// was, readable and honestly stale, rather than stalled. And a follower
// never invents state: a version that will not decode or will not apply
// stops that table's catch-up and is reported, because a replica that is
// neither the leader's state nor its own is worse than one that is behind.
// -----------------------------------------------------------------------------

use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::time::Duration;

use tracing::{debug, info, warn};

use zyron_catalog::Catalog;
use zyron_common::{DeploymentMode, PeerRegistry};
use zyron_lake::{AllCommitted, LakePaths, TransactionLog};

/// Seconds between polls. Replication lag is measured in versions, and a
/// poll that finds nothing costs one round trip, so this is the tradeoff
/// between staleness and chatter rather than a correctness knob.
pub const DEFAULT_INTERVAL_SECS: u64 = 10;

/// Versions fetched in one poll. A follower far behind catches up over
/// several ticks rather than holding one connection open for all of it.
pub const DEFAULT_BATCH: usize = 256;

#[derive(Debug, Clone)]
pub struct LakeFollowerConfig {
    pub interval_secs: u64,
    pub batch: usize,
    pub data_dir: PathBuf,
    /// User the follower connects to its leaders as
    pub user: String,
    pub database: String,
}

#[derive(Debug, Default)]
pub struct LakeFollowerStats {
    pub polls: AtomicU64,
    pub versions_applied: AtomicU64,
    pub unreachable: AtomicU64,
    pub failed: AtomicU64,
}

pub struct LakeFollowerWorker {
    shutdown: Arc<AtomicBool>,
    stats: Arc<LakeFollowerStats>,
    handle: Option<tokio::task::JoinHandle<()>>,
}

impl LakeFollowerWorker {
    /// Starts the worker, or returns None off the lake tier. A db node
    /// holds no lake tables, so it has nothing to follow.
    pub fn start(
        mode: DeploymentMode,
        catalog: Arc<Catalog>,
        peers: Arc<parking_lot::RwLock<Arc<PeerRegistry>>>,
        config: LakeFollowerConfig,
    ) -> Option<Self> {
        if !mode.runs_lake_tier() {
            return None;
        }
        let shutdown = Arc::new(AtomicBool::new(false));
        let stats = Arc::new(LakeFollowerStats::default());
        let handle = tokio::spawn(follower_loop(
            Arc::clone(&shutdown),
            Arc::clone(&stats),
            catalog,
            peers,
            config,
        ));
        info!("lake follower worker started");
        Some(Self {
            shutdown,
            stats,
            handle: Some(handle),
        })
    }

    pub fn stats(&self) -> &Arc<LakeFollowerStats> {
        &self.stats
    }

    pub fn shutdown(&mut self) {
        self.shutdown.store(true, Ordering::Release);
        if let Some(handle) = self.handle.take() {
            handle.abort();
        }
    }
}

async fn follower_loop(
    shutdown: Arc<AtomicBool>,
    stats: Arc<LakeFollowerStats>,
    catalog: Arc<Catalog>,
    peers: Arc<parking_lot::RwLock<Arc<PeerRegistry>>>,
    config: LakeFollowerConfig,
) {
    let mut ticker = tokio::time::interval(Duration::from_secs(config.interval_secs.max(1)));
    loop {
        ticker.tick().await;
        if shutdown.load(Ordering::Acquire) {
            break;
        }
        for (table, peer_name, remote) in followed_tables(&catalog) {
            if shutdown.load(Ordering::Acquire) {
                break;
            }
            let Some(address) = peers.read().get(&peer_name).map(|p| p.address.clone()) else {
                // The peer was dropped while the table still names it. The
                // table stays where it is rather than being reset, because
                // what it applied is real
                warn!(
                    table = %table.1,
                    peer = %peer_name,
                    "a followed table names a peer that is no longer declared"
                );
                continue;
            };
            stats.polls.fetch_add(1, Ordering::Relaxed);
            match poll_one(&table, &address, &remote, &config).await {
                Ok(0) => {}
                Ok(applied) => {
                    stats.versions_applied.fetch_add(applied, Ordering::Relaxed);
                    debug!(table = %table.1, applied, "follower caught up");
                }
                Err(FollowError::Unreachable(reason)) => {
                    stats.unreachable.fetch_add(1, Ordering::Relaxed);
                    debug!(table = %table.1, peer = %peer_name, %reason, "leader unreachable");
                }
                Err(FollowError::Fatal(reason)) => {
                    stats.failed.fetch_add(1, Ordering::Relaxed);
                    warn!(table = %table.1, peer = %peer_name, %reason, "follower stopped");
                }
            }
        }
    }
}

/// Why a poll did not finish. The distinction is the point: a leader that
/// is down is expected and costs nothing, while a version that will not
/// apply means this table cannot catch up until someone looks at it.
enum FollowError {
    Unreachable(String),
    Fatal(String),
}

/// One table's catch-up. Returns how many versions it applied.
async fn poll_one(
    table: &(u32, String),
    address: &str,
    remote: &str,
    config: &LakeFollowerConfig,
) -> Result<u64, FollowError> {
    let paths = LakePaths::new(&config.data_dir, table.0);
    let log = TransactionLog::open_shared(paths.clone(), &AllCommitted)
        .map_err(|e| FollowError::Fatal(format!("opening the local log: {e}")))?;
    let from = log.latest_version();

    let versions = zyron_wire::peer_probe::fetch_remote_versions(
        address,
        &config.user,
        &config.database,
        remote,
        from,
        config.batch,
    )
    .await
    .map_err(|e| FollowError::Unreachable(e.to_string()))?;
    if versions.is_empty() {
        return Ok(0);
    }
    // Applying is local and deterministic, so a failure here is not the
    // leader being away, it is this replica being unable to continue
    let applied = zyron_lake::apply_versions(&log, &versions)
        .map_err(|e| FollowError::Fatal(e.to_string()))?;
    Ok(applied)
}

/// Every lake table that declared a leader, as (table id, local name,
/// peer) plus the remote table name.
fn followed_tables(catalog: &Catalog) -> Vec<((u32, String), String, String)> {
    let mut out = Vec::new();
    for table in catalog.list_all_tables() {
        if !table.lake.is_lake() {
            continue;
        }
        if let Some((peer, remote)) = table.lake.follows() {
            out.push((
                (table.id.0, table.name.clone()),
                peer.to_string(),
                remote.to_string(),
            ));
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A db node holds no lake tables, so it starts no follower thread
    #[test]
    fn test_the_worker_does_not_start_off_the_lake_tier() {
        assert!(!DeploymentMode::Db.runs_lake_tier());
        assert!(DeploymentMode::Lake.runs_lake_tier());
        assert!(DeploymentMode::Unified.runs_lake_tier());
    }

    /// The two failure kinds are treated differently on purpose: an
    /// unreachable leader is expected and cheap, a version that will not
    /// apply means this replica is stuck and someone has to look
    #[test]
    fn test_failure_kinds_are_distinguished() {
        let stats = LakeFollowerStats::default();
        for error in [
            FollowError::Unreachable("connection refused".into()),
            FollowError::Unreachable("timed out".into()),
            FollowError::Fatal("version 7 does not decode".into()),
        ] {
            match error {
                FollowError::Unreachable(_) => {
                    stats.unreachable.fetch_add(1, Ordering::Relaxed);
                }
                FollowError::Fatal(_) => {
                    stats.failed.fetch_add(1, Ordering::Relaxed);
                }
            }
        }
        assert_eq!(stats.unreachable.load(Ordering::Relaxed), 2);
        assert_eq!(stats.failed.load(Ordering::Relaxed), 1);
    }
}
