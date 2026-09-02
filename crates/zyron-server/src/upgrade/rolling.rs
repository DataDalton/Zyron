//! Rolling upgrade coordination.
//!
//! Nodes go one at a time, followers first, the leader last. Each node
//! drains, restarts on the new binary, and is watched against the health
//! baseline the cluster had before the sequence started. A node that does
//! not recover inside the threshold is rolled back on its own and the whole
//! sequence pauses, because a second node failing the same way is a pattern
//! and cascading through it would take the cluster down rather than one node.
//!
//! Leadership moves only after every follower is healthy on the new binary,
//! so the node that decides for the group is the last one to change

use std::sync::Arc;

use zyron_common::format::{
    HealthBaseline, HealthThreshold, HealthVerdict, NodeUpgradeState, UpgradeBoard, UpgradePhase,
};
use zyron_common::{Result, ZyronError};

/// What a node in the sequence needs
#[derive(Debug, Clone)]
pub struct NodePlan {
    pub node_id: String,
    pub is_leader: bool,
}

/// What the coordinator does to a node. Implemented by the server for a
/// real cluster and by a test for a simulated one, so the sequencing logic
/// is exercised without a cluster
#[async_trait::async_trait]
pub trait NodeDriver: Send + Sync {
    /// Stops accepting new work and lets in-flight work finish
    async fn drain(&self, node_id: &str) -> Result<()>;

    /// Restarts the node on the staged binary
    async fn restart(&self, node_id: &str, to_version: &str) -> Result<()>;

    /// Reads the node's current health
    async fn observe(&self, node_id: &str) -> Result<HealthBaseline>;

    /// Puts the node back on the previous binary
    async fn rollback(&self, node_id: &str, to_version: &str) -> Result<()>;

    /// Moves leadership away from a node
    async fn transfer_leadership(&self, from_node: &str) -> Result<()>;

    /// Waits before the next health observation
    async fn wait(&self, secs: u64);

    /// Unix seconds now
    fn now_secs(&self) -> u64;
}

/// How the sequence ended
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RollingOutcome {
    /// Every node is on the new binary
    Completed { nodes_upgraded: u32 },
    /// A node failed its health check, was rolled back, and the sequence
    /// stopped for an operator
    PausedAfterRollback {
        node_id: String,
        reason: String,
        nodes_upgraded: u32,
    },
    /// An operator paused the sequence
    PausedByOperator { nodes_upgraded: u32 },
}

impl RollingOutcome {
    pub fn nodes_upgraded(&self) -> u32 {
        match self {
            RollingOutcome::Completed { nodes_upgraded }
            | RollingOutcome::PausedAfterRollback { nodes_upgraded, .. }
            | RollingOutcome::PausedByOperator { nodes_upgraded } => *nodes_upgraded,
        }
    }

    pub fn completed(&self) -> bool {
        matches!(self, RollingOutcome::Completed { .. })
    }
}

/// Settings the sequence runs under
#[derive(Debug, Clone, Copy)]
pub struct RollingSettings {
    /// Seconds a node has to reach its baseline after restarting
    pub health_recovery_timeout_secs: u64,
    /// Seconds between health observations
    pub health_poll_interval_secs: u64,
    pub threshold: HealthThreshold,
    /// Whether a failing node is put back on the previous binary
    pub rollback_on_health_fail: bool,
}

impl Default for RollingSettings {
    fn default() -> Self {
        Self {
            health_recovery_timeout_secs: 5 * 60,
            health_poll_interval_secs: 5,
            threshold: HealthThreshold::default(),
            rollback_on_health_fail: true,
        }
    }
}

/// Runs the sequence
pub async fn run(
    driver: &dyn NodeDriver,
    board: &UpgradeBoard,
    nodes: &[NodePlan],
    from_version: &str,
    to_version: &str,
    baseline: HealthBaseline,
    settings: RollingSettings,
) -> Result<RollingOutcome> {
    if nodes.is_empty() {
        return Err(ZyronError::UpgradeRefused(
            "the rolling sequence was given no nodes to upgrade".to_string(),
        ));
    }

    // Followers first, the leader last, so the node that decides for the
    // group changes only once every other node is proven healthy
    let mut ordered: Vec<&NodePlan> = nodes.iter().collect();
    ordered.sort_by_key(|node| node.is_leader);

    let started = driver.now_secs();
    let mut upgraded = 0u32;

    for node in ordered {
        if board.settings().paused {
            publish(
                board,
                node,
                from_version,
                to_version,
                UpgradePhase::Paused,
                driver.now_secs(),
                started,
                "paused by operator before this node started",
            );
            return Ok(RollingOutcome::PausedByOperator {
                nodes_upgraded: upgraded,
            });
        }

        publish(
            board,
            node,
            from_version,
            to_version,
            UpgradePhase::Rolling,
            driver.now_secs(),
            started,
            "draining",
        );
        driver.drain(&node.node_id).await?;

        if node.is_leader {
            publish(
                board,
                node,
                from_version,
                to_version,
                UpgradePhase::Rolling,
                driver.now_secs(),
                started,
                "transferring leadership",
            );
            driver.transfer_leadership(&node.node_id).await?;
        }

        publish(
            board,
            node,
            from_version,
            to_version,
            UpgradePhase::Rolling,
            driver.now_secs(),
            started,
            "restarting on the new binary",
        );
        driver.restart(&node.node_id, to_version).await?;

        match watch_health(driver, node, &baseline, settings).await? {
            HealthVerdict::Healthy => {
                upgraded += 1;
                publish(
                    board,
                    node,
                    from_version,
                    to_version,
                    UpgradePhase::Completed,
                    driver.now_secs(),
                    started,
                    "healthy on the new binary",
                );
            }
            verdict => {
                let reason = verdict.reason();
                if settings.rollback_on_health_fail {
                    publish(
                        board,
                        node,
                        from_version,
                        to_version,
                        UpgradePhase::RollingBack,
                        driver.now_secs(),
                        started,
                        &format!("unhealthy, {reason}"),
                    );
                    driver.rollback(&node.node_id, from_version).await?;
                }
                // The whole sequence stops here. A second node failing the
                // same way is a pattern, and rolling through it would take
                // the cluster down rather than one node
                publish(
                    board,
                    node,
                    from_version,
                    to_version,
                    UpgradePhase::Paused,
                    driver.now_secs(),
                    started,
                    &format!("rolled back and paused, {reason}"),
                );
                board.update_settings(|settings| settings.paused = true);
                return Ok(RollingOutcome::PausedAfterRollback {
                    node_id: node.node_id.clone(),
                    reason,
                    nodes_upgraded: upgraded,
                });
            }
        }
    }

    Ok(RollingOutcome::Completed {
        nodes_upgraded: upgraded,
    })
}

/// Watches one node until it is healthy or the recovery window closes
async fn watch_health(
    driver: &dyn NodeDriver,
    node: &NodePlan,
    baseline: &HealthBaseline,
    settings: RollingSettings,
) -> Result<HealthVerdict> {
    let deadline = driver.now_secs() + settings.health_recovery_timeout_secs;
    let mut last;
    loop {
        let observed = driver.observe(&node.node_id).await?;
        last = baseline.judge(&observed, settings.threshold);
        if last.is_healthy() {
            return Ok(last);
        }
        if driver.now_secs() >= deadline {
            return Ok(last);
        }
        driver.wait(settings.health_poll_interval_secs).await;
    }
}

/// Publishes one node's state to the board
#[allow(clippy::too_many_arguments)]
fn publish(
    board: &UpgradeBoard,
    node: &NodePlan,
    from_version: &str,
    to_version: &str,
    phase: UpgradePhase,
    now: u64,
    started: u64,
    message: &str,
) {
    board.set_node_state(NodeUpgradeState {
        node_id: node.node_id.clone(),
        from_version: from_version.to_string(),
        to_version: to_version.to_string(),
        phase,
        started_at_secs: started,
        updated_at_secs: now,
        is_leader: node.is_leader,
        message: message.to_string(),
    });
}

/// Captures the cluster's health baseline before a sequence starts
pub async fn capture_baseline(
    driver: &dyn NodeDriver,
    nodes: &[NodePlan],
) -> Result<HealthBaseline> {
    if nodes.is_empty() {
        return Ok(HealthBaseline::default());
    }
    let mut p50 = 0u64;
    let mut p99 = 0u64;
    let mut throughput = 0.0f64;
    let mut error_rate = 0.0f64;
    let mut connections = 0u64;
    for node in nodes {
        let observed = driver.observe(&node.node_id).await?;
        // The baseline takes the worst latency and the total throughput, so
        // a node that was already the slow one is not asked to beat itself
        p50 = p50.max(observed.p50_latency_us);
        p99 = p99.max(observed.p99_latency_us);
        throughput += observed.throughput_per_sec;
        error_rate = error_rate.max(observed.error_rate);
        connections += observed.active_connections;
    }
    Ok(HealthBaseline {
        p50_latency_us: p50,
        p99_latency_us: p99,
        throughput_per_sec: throughput / nodes.len() as f64,
        error_rate,
        active_connections: connections,
    })
}

/// A driver over a simulated cluster, used by the rolling upgrade tests and
/// by `zyron-ctl upgrade check` to walk the sequence without touching a
/// node
pub struct SimulatedCluster {
    /// Health each node reports, in the order it is asked
    pub observations: parking_lot::Mutex<Vec<(String, HealthBaseline)>>,
    pub drained: parking_lot::Mutex<Vec<String>>,
    pub restarted: parking_lot::Mutex<Vec<String>>,
    pub rolled_back: parking_lot::Mutex<Vec<String>>,
    pub leadership_transfers: parking_lot::Mutex<Vec<String>>,
    clock: std::sync::atomic::AtomicU64,
}

impl SimulatedCluster {
    pub fn new(observations: Vec<(String, HealthBaseline)>) -> Arc<Self> {
        Arc::new(Self {
            observations: parking_lot::Mutex::new(observations),
            drained: parking_lot::Mutex::new(Vec::new()),
            restarted: parking_lot::Mutex::new(Vec::new()),
            rolled_back: parking_lot::Mutex::new(Vec::new()),
            leadership_transfers: parking_lot::Mutex::new(Vec::new()),
            clock: std::sync::atomic::AtomicU64::new(1_000),
        })
    }
}

#[async_trait::async_trait]
impl NodeDriver for SimulatedCluster {
    async fn drain(&self, node_id: &str) -> Result<()> {
        self.drained.lock().push(node_id.to_string());
        Ok(())
    }

    async fn restart(&self, node_id: &str, _to_version: &str) -> Result<()> {
        self.restarted.lock().push(node_id.to_string());
        Ok(())
    }

    async fn observe(&self, node_id: &str) -> Result<HealthBaseline> {
        let mut observations = self.observations.lock();
        match observations.iter().position(|(id, _)| id == node_id) {
            Some(index) => Ok(observations.remove(index).1),
            None => Ok(HealthBaseline {
                p50_latency_us: 100,
                p99_latency_us: 1_000,
                throughput_per_sec: 10_000.0,
                error_rate: 0.0,
                active_connections: 10,
            }),
        }
    }

    async fn rollback(&self, node_id: &str, _to_version: &str) -> Result<()> {
        self.rolled_back.lock().push(node_id.to_string());
        Ok(())
    }

    async fn transfer_leadership(&self, from_node: &str) -> Result<()> {
        self.leadership_transfers.lock().push(from_node.to_string());
        Ok(())
    }

    async fn wait(&self, secs: u64) {
        self.clock
            .fetch_add(secs.max(1), std::sync::atomic::Ordering::Relaxed);
    }

    fn now_secs(&self) -> u64 {
        self.clock.load(std::sync::atomic::Ordering::Relaxed)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn healthy() -> HealthBaseline {
        HealthBaseline {
            p50_latency_us: 100,
            p99_latency_us: 1_000,
            throughput_per_sec: 10_000.0,
            error_rate: 0.0,
            active_connections: 10,
        }
    }

    fn nodes() -> Vec<NodePlan> {
        vec![
            NodePlan {
                node_id: "node-1".to_string(),
                is_leader: true,
            },
            NodePlan {
                node_id: "node-2".to_string(),
                is_leader: false,
            },
            NodePlan {
                node_id: "node-3".to_string(),
                is_leader: false,
            },
        ]
    }

    #[tokio::test]
    async fn test_a_healthy_cluster_upgrades_followers_then_the_leader() {
        let driver = SimulatedCluster::new(Vec::new());
        let board = UpgradeBoard::new();
        let outcome = run(
            driver.as_ref(),
            &board,
            &nodes(),
            "0.11.0",
            "0.12.0",
            healthy(),
            RollingSettings::default(),
        )
        .await
        .expect("runs");
        assert_eq!(outcome, RollingOutcome::Completed { nodes_upgraded: 3 });

        let restarted = driver.restarted.lock().clone();
        assert_eq!(
            restarted,
            vec![
                "node-2".to_string(),
                "node-3".to_string(),
                "node-1".to_string()
            ],
            "the leader restarts last"
        );
        assert_eq!(
            driver.leadership_transfers.lock().clone(),
            vec!["node-1".to_string()],
            "leadership moves once, off the leader"
        );
        assert!(driver.rolled_back.lock().is_empty());

        let states = board.node_states();
        assert_eq!(states.len(), 3);
        assert!(states.iter().all(|s| s.phase == UpgradePhase::Completed));
        assert_eq!(board.cluster_phase(), UpgradePhase::Completed);
    }

    #[tokio::test]
    async fn test_an_unhealthy_node_is_rolled_back_and_the_sequence_pauses() {
        // node-2 answers slowly every time it is asked
        let slow = HealthBaseline {
            p99_latency_us: 60_000,
            ..healthy()
        };
        let driver = SimulatedCluster::new(vec![
            ("node-2".to_string(), slow),
            ("node-2".to_string(), slow),
            ("node-2".to_string(), slow),
        ]);
        let board = UpgradeBoard::new();
        let settings = RollingSettings {
            health_recovery_timeout_secs: 10,
            health_poll_interval_secs: 5,
            ..RollingSettings::default()
        };
        let outcome = run(
            driver.as_ref(),
            &board,
            &nodes(),
            "0.11.0",
            "0.12.0",
            healthy(),
            settings,
        )
        .await
        .expect("runs");

        match outcome {
            RollingOutcome::PausedAfterRollback {
                node_id,
                reason,
                nodes_upgraded,
            } => {
                assert_eq!(node_id, "node-2");
                assert_eq!(nodes_upgraded, 0);
                assert!(reason.contains("p99 latency"), "{reason}");
            }
            other => panic!("expected a rollback and a pause, got {other:?}"),
        }
        assert_eq!(
            driver.rolled_back.lock().clone(),
            vec!["node-2".to_string()]
        );
        assert!(
            !driver.restarted.lock().contains(&"node-3".to_string()),
            "the sequence stops rather than cascading"
        );
        assert!(board.settings().paused, "the whole sequence is paused");
        assert_eq!(board.cluster_phase(), UpgradePhase::Paused);
    }

    #[tokio::test]
    async fn test_an_operator_pause_stops_the_sequence_at_the_next_node() {
        let driver = SimulatedCluster::new(Vec::new());
        let board = UpgradeBoard::new();
        board.update_settings(|settings| settings.paused = true);
        let outcome = run(
            driver.as_ref(),
            &board,
            &nodes(),
            "0.11.0",
            "0.12.0",
            healthy(),
            RollingSettings::default(),
        )
        .await
        .expect("runs");
        assert_eq!(
            outcome,
            RollingOutcome::PausedByOperator { nodes_upgraded: 0 }
        );
        assert!(driver.restarted.lock().is_empty());
    }

    #[tokio::test]
    async fn test_a_sequence_with_no_nodes_is_refused() {
        let driver = SimulatedCluster::new(Vec::new());
        let board = UpgradeBoard::new();
        let err = run(
            driver.as_ref(),
            &board,
            &[],
            "0.11.0",
            "0.12.0",
            healthy(),
            RollingSettings::default(),
        )
        .await
        .expect_err("refused");
        assert!(err.to_string().contains("no nodes"), "{err}");
    }

    #[tokio::test]
    async fn test_the_baseline_takes_the_worst_latency_across_nodes() {
        let driver = SimulatedCluster::new(vec![
            (
                "node-1".to_string(),
                HealthBaseline {
                    p99_latency_us: 500,
                    ..healthy()
                },
            ),
            (
                "node-2".to_string(),
                HealthBaseline {
                    p99_latency_us: 2_000,
                    ..healthy()
                },
            ),
            (
                "node-3".to_string(),
                HealthBaseline {
                    p99_latency_us: 900,
                    ..healthy()
                },
            ),
        ]);
        let baseline = capture_baseline(driver.as_ref(), &nodes())
            .await
            .expect("captures");
        assert_eq!(baseline.p99_latency_us, 2_000);
        assert!(baseline.throughput_per_sec > 0.0);
    }

    #[tokio::test]
    async fn test_rollback_can_be_turned_off_and_the_sequence_still_pauses() {
        let slow = HealthBaseline {
            p99_latency_us: 60_000,
            ..healthy()
        };
        let driver = SimulatedCluster::new(vec![("node-2".to_string(), slow)]);
        let board = UpgradeBoard::new();
        let settings = RollingSettings {
            health_recovery_timeout_secs: 0,
            rollback_on_health_fail: false,
            ..RollingSettings::default()
        };
        let outcome = run(
            driver.as_ref(),
            &board,
            &nodes(),
            "0.11.0",
            "0.12.0",
            healthy(),
            settings,
        )
        .await
        .expect("runs");
        assert!(matches!(
            outcome,
            RollingOutcome::PausedAfterRollback { .. }
        ));
        assert!(driver.rolled_back.lock().is_empty());
    }
}
