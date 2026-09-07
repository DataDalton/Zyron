//! Deciding which node takes work, which node gives it up, and when the mesh
//! grows or shrinks.
//!
//! ## What a scale-in has to do before it removes a node
//!
//! Removing a node is not deleting it. The data survives, so nothing is lost
//! in the sense that matters for correctness, but every page that node had in
//! memory is now a page a survivor has never read, and the minutes after a
//! scale-in are slower than the minutes before it. That is the cost that makes
//! operators turn scale-in off, so the sequence here is built around removing
//! it:
//!
//! 1. Tell the node to stop taking new work and finish what it has.
//! 2. While it finishes, read what it had in memory and hand that to the
//!    survivors, so they read those pages while the departing node is still
//!    serving from them.
//! 3. Only once it is drained and the handover is done, ask the provisioner
//!    to take it back.
//!
//! Step two is what makes step three cheap. Doing it after the node is gone
//! would be reading the manifest of a node that no longer answers.
//!
//! ## What it refuses to do
//!
//! Every decision here is guarded by the provisioner's own capabilities and by
//! the reclaim rule that belongs to the driver, not to this file. A scheduler
//! that decided for itself when a node could be given back would be second
//! guessing the one component that knows whether the node is billed by the
//! hour.

use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Duration;

use zyron_pressure::extension::ExtensionRegistry;
use zyron_pressure::pressure::ActuatorLevel;
use zyron_pressure::provisioner::{
    ProvisionRequest, ProvisionTicket, ProvisionerDriver, ProvisionerRegistry, ReclaimRequest,
    ReclaimVerdict,
};

use crate::pool::WarmPool;
use crate::rpc::{
    BeginDrainRequest, DrainStatus, DrainStatusRequest, HotSetManifestRequest, MeshRpc,
    MeshRpcError, NodeRef, PrefetchRequest,
};

/// How often a drain is polled while it finishes.
///
/// A drain takes as long as its longest running query, which is seconds to
/// minutes. Polling faster than this would ask a node that is busy finishing
/// work to answer a question instead.
pub const DRAIN_POLL_INTERVAL: Duration = Duration::from_millis(500);

/// Bytes a survivor may spend reading a departing node's pages.
///
/// A prefetch that evicts what the survivor is already serving from is worse
/// than no prefetch, so the budget is what the handover may cost, not what the
/// manifest happens to name.
pub const PREFETCH_BYTE_BUDGET: u64 = 256 * 1024 * 1024;

/// What a scale-in did.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DrainOutcome {
    /// The node finished its work, handed its pages over, and was reclaimed
    Reclaimed { node: NodeRef },
    /// It finished and handed over, and the driver is keeping it because the
    /// interval it is already paid for has not run out
    HeldUntilPaidFor { node: NodeRef, remaining: Duration },
    /// It did not finish inside the deadline, so nothing was taken away
    StillBusy { node: NodeRef, in_flight: u32 },
    /// The driver refused, and this is why
    Refused { node: NodeRef, reason: String },
}

/// What the scheduler has done, for the mesh views.
#[derive(Debug, Default)]
pub struct SchedulerStats {
    pub warm_takes: AtomicU64,
    pub provisions: AtomicU64,
    pub drains_started: AtomicU64,
    pub drains_completed: AtomicU64,
    pub pages_handed_over: AtomicU64,
    pub refusals: AtomicU64,
}

/// Places work, grows the mesh, and shrinks it.
pub struct MeshScheduler {
    rpc: Arc<dyn MeshRpc>,
    pool: Arc<WarmPool>,
    stats: Arc<SchedulerStats>,
    /// Monotonic call sequence, so a retried request is recognizable as the
    /// same one by the node answering it
    sequence: AtomicU64,
    /// This node, named on every request it makes
    local: NodeRef,
}

impl MeshScheduler {
    pub fn new(local: NodeRef, rpc: Arc<dyn MeshRpc>, pool: Arc<WarmPool>) -> Self {
        Self {
            rpc,
            pool,
            stats: Arc::new(SchedulerStats::default()),
            sequence: AtomicU64::new(1),
            local,
        }
    }

    pub fn stats(&self) -> Arc<SchedulerStats> {
        Arc::clone(&self.stats)
    }

    pub fn pool(&self) -> &WarmPool {
        &self.pool
    }

    /// The transport this scheduler speaks through, so the upgrade driver
    /// reaches the same nodes through the same address book
    pub fn rpc(&self) -> Arc<dyn MeshRpc> {
        Arc::clone(&self.rpc)
    }

    /// A sequence for a call made on this scheduler's behalf
    pub fn next_call_sequence(&self) -> u64 {
        self.next_sequence()
    }

    fn next_sequence(&self) -> u64 {
        self.sequence.fetch_add(1, Ordering::Relaxed)
    }

    /// Drains one node and hands its working set to the survivors, without
    /// taking the node away.
    ///
    /// What a restart needs, and the first half of a scale-in. The deadline
    /// covers the drain. Returns where the drain stood when it finished or
    /// when the deadline ran out, and whether anything was handed over
    pub async fn drain_for_restart(
        &self,
        node: &NodeRef,
        survivors: &[NodeRef],
        deadline: Duration,
    ) -> Result<(DrainStatus, bool), MeshRpcError> {
        self.stats.drains_started.fetch_add(1, Ordering::Relaxed);
        let sequence = self.next_sequence();

        let mut status = self
            .rpc
            .begin_drain(BeginDrainRequest {
                target: node.clone(),
                sequence,
                deadline_ms: deadline.as_millis().min(u32::MAX as u128) as u32,
                relocate_sessions: !survivors.is_empty(),
            })
            .await?;
        tracing::info!(
            node = %node.name,
            in_flight = status.queries_in_flight,
            "node is draining"
        );

        // The handover runs while the node is still serving, so a survivor is
        // reading those pages from storage at the same time the departing node
        // is answering from memory
        let handed_over = self.hand_over_hot_set(node, survivors).await?;

        let started = std::time::Instant::now();
        while !status.drained {
            if started.elapsed() >= deadline {
                tracing::warn!(
                    node = %node.name,
                    in_flight = status.queries_in_flight,
                    "drain did not finish inside its deadline"
                );
                return Ok((status, handed_over));
            }
            tokio::time::sleep(DRAIN_POLL_INTERVAL).await;
            match self
                .rpc
                .drain_status(DrainStatusRequest {
                    target: node.clone(),
                    sequence,
                })
                .await
            {
                Ok(next) => status = next,
                // A node that blinks mid-drain is still draining. Only a
                // refusal ends the attempt, because a refusal will not change
                Err(e) if e.transient() => continue,
                Err(e) => return Err(e),
            }
        }
        self.stats.drains_completed.fetch_add(1, Ordering::Relaxed);
        Ok((status, handed_over))
    }

    /// The driver this deployment's registration mode selects.
    fn driver(&self) -> Arc<dyn ProvisionerDriver> {
        ProvisionerRegistry::global().active()
    }

    /// Whether the scheduler can place work.
    ///
    /// True once it has somewhere to place it: a warm pool with a node in it,
    /// or a provisioner that can make one. A scheduler with neither is a
    /// single node, and says so rather than accepting requests it cannot
    /// answer.
    pub fn ready(&self) -> bool {
        !self.pool.is_empty() || self.driver().capabilities().can_provision
    }

    /// Whether the pressure ladder's mesh rungs can be reached from here.
    ///
    /// Reads the registry rather than answering from this type, because the
    /// ladder reads the registry too: one answer, one place, and no chance of
    /// the scheduler claiming a rung the controller cannot see.
    pub fn rungs_reachable(&self) -> bool {
        let registry = ExtensionRegistry::global();
        registry.handles(ActuatorLevel::WarmPoolTake)
            || registry.handles(ActuatorLevel::ProvisionNode)
    }

    /// Claims a node the mesh already has running.
    ///
    /// The cheap half of growing: no control plane, no boot, just a node that
    /// was kept idle for this. Returns None when the pool is empty, which
    /// sends the ladder on to provisioning.
    pub fn take_warm_node(&self) -> Option<NodeRef> {
        let node = self.pool.take()?;
        self.stats.warm_takes.fetch_add(1, Ordering::Relaxed);
        tracing::info!(node = %node.name, "claimed a warm node");
        Some(node)
    }

    /// Asks the provisioner for capacity.
    ///
    /// Returns as soon as the control plane accepts, not when the node
    /// serves. What it costs is recorded against the driver so the projection
    /// that decides when to ask next is measured rather than configured.
    pub fn provision(&self, request: &ProvisionRequest) -> Result<ProvisionTicket, String> {
        let driver = self.driver();
        let capabilities = driver.capabilities();
        if !capabilities.can_provision {
            self.stats.refusals.fetch_add(1, Ordering::Relaxed);
            return Err(format!(
                "the {} provisioner cannot add capacity from this node",
                driver.kind().as_str()
            ));
        }
        let started = std::time::Instant::now();
        match driver.provision(request) {
            Ok(ticket) => {
                ProvisionerRegistry::global()
                    .record_provision_latency(driver.kind(), started.elapsed());
                self.stats.provisions.fetch_add(1, Ordering::Relaxed);
                tracing::info!(
                    nodes = ticket.nodes,
                    ticket = %ticket.external_id,
                    expected_secs = ticket.expected.as_secs(),
                    "asked the provisioner for capacity"
                );
                Ok(ticket)
            }
            Err(e) => {
                self.stats.refusals.fetch_add(1, Ordering::Relaxed);
                Err(e.to_string())
            }
        }
    }

    /// Takes a node out of the mesh, in the order that keeps the survivors
    /// warm.
    ///
    /// The deadline covers the drain, not the handover: a node that finishes
    /// its work still gets its pages handed over, because the point of the
    /// handover is the minutes after it leaves and abandoning it would spend
    /// the drain and keep none of the benefit.
    pub async fn drain_and_reclaim(
        &self,
        node: NodeRef,
        survivors: &[NodeRef],
        deadline: Duration,
    ) -> Result<DrainOutcome, MeshRpcError> {
        let (status, handed_over) = self.drain_for_restart(&node, survivors, deadline).await?;
        if !status.drained {
            return Ok(DrainOutcome::StillBusy {
                node,
                in_flight: status.queries_in_flight,
            });
        }

        let driver = self.driver();
        let request = ReclaimRequest {
            node_id: node.node_id,
            remaining_paid_interval: self.pool.remaining_paid_interval(&node),
            predicted_idle_window: self.pool.predicted_idle_window(),
            hot_set_handed_off: handed_over,
        };
        match driver.reclaim_allowed(&request) {
            ReclaimVerdict::Allowed => match driver.reclaim(&request) {
                Ok(()) => {
                    self.pool.forget(&node);
                    tracing::info!(node = %node.name, "node reclaimed");
                    Ok(DrainOutcome::Reclaimed { node })
                }
                Err(e) => {
                    self.stats.refusals.fetch_add(1, Ordering::Relaxed);
                    Ok(DrainOutcome::Refused {
                        node,
                        reason: e.to_string(),
                    })
                }
            },
            ReclaimVerdict::HoldUntilPaidIntervalEnds(remaining) => {
                // Drained and idle but still paid for, so it goes back to the
                // warm pool rather than being thrown away and bought again
                self.pool.offer(node.clone());
                Ok(DrainOutcome::HeldUntilPaidFor { node, remaining })
            }
            ReclaimVerdict::Refused(reason) => {
                self.stats.refusals.fetch_add(1, Ordering::Relaxed);
                Ok(DrainOutcome::Refused {
                    node,
                    reason: reason.to_string(),
                })
            }
        }
    }

    /// Reads a departing node's working set and asks the survivors to warm it.
    ///
    /// Returns whether anything was handed over. False is not a failure: a
    /// node with nothing resident, or a mesh with no survivor to hand it to,
    /// has nothing to do here, and the reclaim rule reads that as the
    /// handover being complete only when it genuinely was.
    async fn hand_over_hot_set(
        &self,
        node: &NodeRef,
        survivors: &[NodeRef],
    ) -> Result<bool, MeshRpcError> {
        if survivors.is_empty() {
            return Ok(false);
        }
        let sequence = self.next_sequence();
        let mut chunk_index = 0u32;
        let mut chunks_total = 1u32;
        let mut handed = 0u64;

        while chunk_index < chunks_total {
            let chunk = match self
                .rpc
                .hot_set_manifest(HotSetManifestRequest {
                    target: node.clone(),
                    sequence,
                    chunk: chunk_index,
                })
                .await
            {
                Ok(chunk) => chunk,
                // A node too busy to answer for its manifest is still a node
                // worth draining. The handover is an optimization and the
                // drain is not, so this reports what it managed rather than
                // failing the scale-in
                Err(e) => {
                    tracing::warn!(
                        node = %node.name,
                        error = %e,
                        "could not read the departing node's working set"
                    );
                    break;
                }
            };
            chunks_total = chunk.chunks_total.max(1);
            chunk_index += 1;
            if chunk.page_ids.is_empty() {
                continue;
            }

            // Every survivor warms the whole chunk. Which of them will be
            // asked for a given page is not known here, and reading a page
            // that is never asked for costs one read while missing one costs
            // a stall on the first query that wants it
            let budget = PREFETCH_BYTE_BUDGET / survivors.len().max(1) as u64;
            for survivor in survivors {
                match self
                    .rpc
                    .prefetch(PrefetchRequest {
                        target: survivor.clone(),
                        sequence,
                        page_ids: chunk.page_ids.clone(),
                        byte_budget: budget,
                    })
                    .await
                {
                    Ok(status) => handed += status.pages_read as u64,
                    Err(e) => tracing::warn!(
                        survivor = %survivor.name,
                        error = %e,
                        "a survivor did not take the handover"
                    ),
                }
            }
        }

        self.stats
            .pages_handed_over
            .fetch_add(handed, Ordering::Relaxed);
        Ok(handed > 0)
    }

    /// The node this scheduler runs on.
    pub fn local(&self) -> &NodeRef {
        &self.local
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rpc::{
        CancelProvisioningRequest, HotSetChunk, MeshFuture, NodeAck, NodeStatus, NodeStatusRequest,
        PrefetchStatus, RelocateSessionRequest, RelocationOutcome, RestartRequest, RollbackRequest,
        SetClusterSettingRequest, StageReleaseRequest,
    };
    use std::sync::Mutex;

    /// A peer that answers from a script, so a drain can be walked without a
    /// network.
    struct Scripted {
        /// Drain answers, taken in order, the last one repeating
        drain: Mutex<Vec<DrainStatus>>,
        pages: Vec<u64>,
        prefetched: Mutex<Vec<(String, usize)>>,
    }

    impl Scripted {
        fn new(drain: Vec<DrainStatus>, pages: Vec<u64>) -> Arc<Self> {
            Arc::new(Self {
                drain: Mutex::new(drain),
                pages,
                prefetched: Mutex::new(Vec::new()),
            })
        }

        fn next_drain(&self) -> DrainStatus {
            let mut queue = self.drain.lock().expect("script");
            if queue.len() > 1 {
                queue.remove(0)
            } else {
                queue[0].clone()
            }
        }
    }

    impl MeshRpc for Scripted {
        fn begin_drain(&self, _r: BeginDrainRequest) -> MeshFuture<'_, DrainStatus> {
            Box::pin(async move { Ok(self.next_drain()) })
        }
        fn drain_status(&self, _r: DrainStatusRequest) -> MeshFuture<'_, DrainStatus> {
            Box::pin(async move { Ok(self.next_drain()) })
        }
        fn hot_set_manifest(&self, r: HotSetManifestRequest) -> MeshFuture<'_, HotSetChunk> {
            Box::pin(async move {
                Ok(HotSetChunk {
                    source: r.target,
                    sequence: r.sequence,
                    chunk: r.chunk,
                    chunks_total: 1,
                    page_ids: self.pages.clone(),
                })
            })
        }
        fn prefetch(&self, r: PrefetchRequest) -> MeshFuture<'_, PrefetchStatus> {
            Box::pin(async move {
                self.prefetched
                    .lock()
                    .expect("script")
                    .push((r.target.name.clone(), r.page_ids.len()));
                Ok(PrefetchStatus {
                    pages_read: r.page_ids.len() as u32,
                    pages_skipped: 0,
                    budget_exhausted: false,
                    target: r.target,
                    sequence: r.sequence,
                })
            })
        }
        fn relocate_session(
            &self,
            _r: RelocateSessionRequest,
        ) -> MeshFuture<'_, RelocationOutcome> {
            Box::pin(async move { Ok(RelocationOutcome::Ended) })
        }
        fn cancel_provisioning(&self, _r: CancelProvisioningRequest) -> MeshFuture<'_, ()> {
            Box::pin(async move { Ok(()) })
        }
        fn node_status(&self, r: NodeStatusRequest) -> MeshFuture<'_, NodeStatus> {
            Box::pin(async move {
                let drain = self.next_drain();
                Ok(NodeStatus {
                    target: r.target,
                    sequence: r.sequence,
                    version: "0.12.0".into(),
                    staged_version: String::new(),
                    draining: !drain.drained,
                    accepting: drain.drained,
                    queries_in_flight: drain.queries_in_flight,
                    sessions_attached: drain.sessions_attached,
                    transactions_open: drain.transactions_open,
                    p50_latency_us: 0,
                    p99_latency_us: 0,
                    throughput_milli_per_sec: 0,
                    error_rate_ppm: 0,
                    queries_in_window: 0,
                    uptime_secs: 0,
                    upgrade: None,
                })
            })
        }
        fn stage_release(&self, r: StageReleaseRequest) -> MeshFuture<'_, NodeAck> {
            Box::pin(async move {
                Ok(NodeAck {
                    target: r.target,
                    sequence: r.sequence,
                    accepted: true,
                    detail: String::new(),
                })
            })
        }
        fn set_cluster_setting(&self, r: SetClusterSettingRequest) -> MeshFuture<'_, NodeAck> {
            Box::pin(async move {
                Ok(NodeAck {
                    target: r.target,
                    sequence: r.sequence,
                    accepted: true,
                    detail: String::new(),
                })
            })
        }
        fn restart_into_staged(&self, r: RestartRequest) -> MeshFuture<'_, NodeAck> {
            Box::pin(async move {
                Ok(NodeAck {
                    target: r.target,
                    sequence: r.sequence,
                    accepted: true,
                    detail: String::new(),
                })
            })
        }
        fn rollback_to_previous(&self, r: RollbackRequest) -> MeshFuture<'_, NodeAck> {
            Box::pin(async move {
                Ok(NodeAck {
                    target: r.target,
                    sequence: r.sequence,
                    accepted: true,
                    detail: String::new(),
                })
            })
        }
    }

    fn node(id: u64, name: &str) -> NodeRef {
        NodeRef::new(id, name)
    }

    fn scheduler(rpc: Arc<dyn MeshRpc>) -> MeshScheduler {
        MeshScheduler::new(node(1, "local"), rpc, Arc::new(WarmPool::new(0)))
    }

    /// A node still running work when the deadline passes is left alone.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn a_node_that_will_not_finish_is_not_taken_away() {
        let busy = DrainStatus {
            target: node(2, "leaving"),
            sequence: 1,
            queries_in_flight: 4,
            sessions_attached: 1,
            transactions_open: 1,
            drained: false,
        };
        let rpc = Scripted::new(vec![busy], vec![1, 2, 3]);
        let scheduler = scheduler(rpc);
        let outcome = scheduler
            .drain_and_reclaim(
                node(2, "leaving"),
                &[node(3, "survivor")],
                Duration::from_millis(600),
            )
            .await
            .expect("the drain answers");
        assert!(
            matches!(outcome, DrainOutcome::StillBusy { in_flight: 4, .. }),
            "{outcome:?}"
        );
    }

    /// The handover happens before the node is gone, and every survivor gets
    /// the pages.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn the_survivors_are_warmed_before_the_node_leaves() {
        let idle = DrainStatus::idle(node(2, "leaving"), 1);
        let rpc = Scripted::new(vec![idle], vec![10, 11, 12, 13]);
        let scheduler = scheduler(Arc::clone(&rpc) as Arc<dyn MeshRpc>);
        let outcome = scheduler
            .drain_and_reclaim(
                node(2, "leaving"),
                &[node(3, "a"), node(4, "b")],
                Duration::from_secs(5),
            )
            .await
            .expect("the drain answers");

        let prefetched = rpc.prefetched.lock().expect("script").clone();
        assert_eq!(prefetched.len(), 2, "not every survivor was warmed");
        for (name, pages) in &prefetched {
            assert_eq!(*pages, 4, "{name} was handed {pages} of 4 pages");
        }
        assert_eq!(
            scheduler.stats().pages_handed_over.load(Ordering::Relaxed),
            8
        );
        // With no provisioner installed the default driver refuses, which is
        // the honest answer on a node that cannot give hardware back
        assert!(
            matches!(outcome, DrainOutcome::Refused { .. }),
            "{outcome:?}"
        );
    }

    /// With nowhere to place work the scheduler says so.
    #[test]
    fn a_scheduler_with_no_pool_and_no_provisioner_is_not_ready() {
        let rpc = Scripted::new(vec![DrainStatus::idle(node(2, "x"), 1)], Vec::new());
        assert!(!scheduler(rpc).ready());
    }
}
