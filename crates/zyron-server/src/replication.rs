//! The consensus side of ordinary writes.
//!
//! ## The applier is the commit authority
//!
//! On every node, an entry is handled in one of two ways. If this process is
//! still holding the transaction that produced it, the applier finishes that
//! transaction's local commit. Otherwise it replays the rows the entry
//! carries. Same loop, same order, one branch:
//!
//! - a leader takes the first branch, so its transactions commit in log order
//!   and the whole group passes through the same sequence of states rather
//!   than merely arriving at the same last one
//! - a follower takes the second
//! - a leader that restarted takes the second for its own entries, because the
//!   process holding those transactions is gone, and rebuilds them from the
//!   log with no special case anywhere
//!
//! ## One durability wait, not two
//!
//! The commit record is written after the entry is committed by the group and
//! the caller is answered without waiting for the local flush. The raft log
//! already holds the transaction on a majority, so if the local record is lost
//! the applied index is behind and the entry is replayed on the way back up.
//! That is why the applied index lives in the commit record itself: a separate
//! file could survive when the record did not, and the transaction would be
//! lost while the node believed it had applied it.
//!
//! ## A transaction that outgrows one entry
//!
//! Chunks are proposed while the transaction is still running and staged by
//! followers into an open local transaction, committed when the chunk marked
//! last arrives. A leader that dies part way leaves those staged transactions
//! behind on every node, and the no-op a new leader appends is what clears
//! them: every node applies it at the same position, so every node abandons
//! exactly the same set.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use zyron_buffer::BufferPool;
use zyron_catalog::Catalog;
use zyron_common::{Result, ZyronError};
use zyron_executor::context::ExecutionContext;
use zyron_executor::replication::{
    ChangesetChunk, ChangesetHeader, ChangesetOp, ChangesetReader, ChangesetSink, Origin,
};
use zyron_raft::{RaftCommand, RaftNode};
use zyron_storage::txn::{IsolationLevel, Transaction, TransactionManager};
use zyron_storage::{DiskManager, Snapshot};
use zyron_wal::WalWriter;

/// What the engine needs to run a transaction, in one place, so the applier
/// and the proposer can both be handed it.
#[derive(Clone)]
pub struct EngineHandles {
    pub catalog: Arc<Catalog>,
    pub wal: Arc<WalWriter>,
    pub buffer_pool: Arc<BufferPool>,
    pub disk_manager: Arc<DiskManager>,
    pub txn_manager: Arc<TransactionManager>,
    pub data_dir: PathBuf,
}

// ---------------------------------------------------------------------------
// Transactions this node is holding open for the group
// ---------------------------------------------------------------------------

/// What this node is waiting to do when one of its own entries comes up.
pub enum PendingCommit {
    /// Finish a transaction that has already run here.
    ///
    /// The transaction is handed over so the applier commits it in log order
    /// rather than leaving each connection to commit whenever its own wait
    /// happens to finish. That is what makes this node's transactions become
    /// visible in the group's order
    Commit {
        txn: Transaction,
        done: tokio::sync::oneshot::Sender<Result<Transaction>>,
    },
    /// Hand a schema change back to the connection that asked for it, when its
    /// entry comes up.
    ///
    /// A schema change is agreed before it runs anywhere, so the object ids
    /// the catalog hands out are the same on every node without being
    /// replicated. It still has to run on the connection rather than in the
    /// applier, because that is where the session is and where the reply the
    /// client is waiting for comes from.
    ///
    /// If the connection has gone, the applier runs it here like any other
    /// node would, so a client that disconnected mid-statement cannot leave
    /// this node the only one without the change
    Statement {
        turn: tokio::sync::oneshot::Sender<tokio::sync::oneshot::Sender<Result<()>>>,
    },
    /// A transaction still running in this process whose chunks have started
    /// going out.
    ///
    /// Registered by the proposer at the first streamed chunk, so when that
    /// chunk comes back through the log the applier can tell a live
    /// transaction, whose writes are already here, from a dead one it has to
    /// replay. Upgraded to [`PendingCommit::Commit`] when the transaction
    /// commits, and cleared by the abort entry when it rolls back
    Running,
}

/// Transactions this process proposed and has not yet seen commit.
///
/// Keyed by origin rather than by index, because a transaction is registered
/// before it is proposed and so before it has an index
#[derive(Default)]
pub struct PendingRegistry {
    inner: parking_lot::Mutex<HashMap<Origin, PendingCommit>>,
}

impl PendingRegistry {
    pub fn register(&self, origin: Origin, pending: PendingCommit) {
        self.inner.lock().insert(origin, pending);
    }

    pub fn take(&self, origin: &Origin) -> Option<PendingCommit> {
        self.inner.lock().remove(origin)
    }

    /// Marks a transaction as live in this process without disturbing a
    /// commit already registered for it
    pub fn register_running_if_absent(&self, origin: Origin) {
        self.inner
            .lock()
            .entry(origin)
            .or_insert(PendingCommit::Running);
    }

    /// Whether this process still holds the transaction, in any state
    pub fn contains(&self, origin: &Origin) -> bool {
        self.inner.lock().contains_key(origin)
    }

    /// Fails every waiter, for a node that is stepping down or shutting down
    /// with proposals outstanding.
    ///
    /// Dropping a waiter's channel is the failure: both shapes report a
    /// dropped channel as work that did not happen
    pub fn fail_all(&self, reason: &str) {
        let taken: Vec<PendingCommit> = self.inner.lock().drain().map(|(_, v)| v).collect();
        drop(taken);
        if !reason.is_empty() {
            tracing::warn!(reason, "outstanding proposals were failed");
        }
    }

    pub fn len(&self) -> usize {
        self.inner.lock().len()
    }

    pub fn is_empty(&self) -> bool {
        self.inner.lock().is_empty()
    }
}

// ---------------------------------------------------------------------------
// Staging a transaction that arrived from another node
// ---------------------------------------------------------------------------

/// A transaction being rebuilt from chunks that have not finished arriving.
struct StagedTxn {
    txn: Transaction,
    ctx: Arc<ExecutionContext>,
    /// Term of the entry that opened it. A no-op from a later term abandons
    /// it, which is how a leader that died part way through a bulk load stops
    /// leaving an open transaction on every node
    term: u64,
    /// Next chunk number expected, so a gap is refused rather than applied
    /// around
    next_chunk: u32,
}

/// What the applier needs from the server around it.
///
/// The applier is built before the server state exists, because the server
/// state holds the seam a connection reaches the applier through. So it is
/// handed one of these once that state is built, and refuses to apply
/// anything until it has been.
///
/// Both of these have to come from the server rather than from the engine
/// handles: a replayed write maintains the same indexes and feeds the same
/// change feed as the write that produced it, and every one of those lives in
/// a registry the server owns
pub trait DdlRunner: Send + Sync {
    /// A context wired the way a statement's context is wired
    fn apply_context(&self, txn_id: u64, snapshot: Snapshot) -> Arc<ExecutionContext>;

    /// `apply_txn_id` is the transaction this node is replaying the statement
    /// under. An online build waits for the transactions that were running
    /// when it published, and this one cannot end until the statement returns,
    /// so the build is told to hold it out of that wait
    fn run<'a>(
        &'a self,
        sql: &'a str,
        context: &'a zyron_executor::replication::StatementContext,
        apply_txn_id: u64,
    ) -> std::pin::Pin<Box<dyn std::future::Future<Output = Result<()>> + Send + 'a>>;
}

/// Applies changesets and finishes locally originated transactions.
pub struct ChangesetMachine {
    pub(crate) engine: EngineHandles,
    /// This node's consensus id, used to recognise its own entries
    node_id: u64,
    /// This process's life. An entry from a previous life of the same node is
    /// replayed rather than finished, because the transaction that produced it
    /// died with that process
    epoch: u64,
    pending: Arc<PendingRegistry>,
    staging: parking_lot::Mutex<HashMap<Origin, StagedTxn>>,
    /// Every transaction whose chunks span entries and whose completing entry
    /// has not applied yet, mapped to the index of its first chunk and the
    /// term it opened in.
    ///
    /// This is what the replay floor is computed from: a crash loses these
    /// transactions' staged writes, so recovery has to start below the first
    /// chunk of the oldest of them, and compaction must not discard it
    open_txns: parking_lot::Mutex<HashMap<Origin, (u64, u64)>>,
    /// Transactions committed here before a restart whose entries sit above
    /// the recovered replay floor. Replay passes their entries again, and
    /// staging them again would double their rows, so they are skipped and
    /// each is forgotten when its final chunk goes by
    recovered: parking_lot::Mutex<HashMap<Origin, u64>>,
    /// Where every agreed commit record sits in the write-ahead log, oldest
    /// first, keyed by the entry it committed.
    ///
    /// This is what pins WAL retention: a restart reads these records to know
    /// which transactions happened, so a checkpoint may only reclaim one once
    /// a raft snapshot covers its entry and the snapshot point itself becomes
    /// the recovery anchor
    agreed_records: parking_lot::Mutex<std::collections::VecDeque<(zyron_wal::record::Lsn, u64)>>,
    /// Transactions replayed from other nodes, for the operator views
    applied_remote: AtomicU64,
    /// Transactions this node originated and committed through the group
    applied_local: AtomicU64,
    /// Set once the server state exists. A schema change that arrives before
    /// it is refused rather than half applied
    ddl: std::sync::OnceLock<Arc<dyn DdlRunner>>,
}

impl ChangesetMachine {
    pub fn new(engine: EngineHandles, node_id: u64, epoch: u64) -> Self {
        Self {
            engine,
            node_id,
            epoch,
            pending: Arc::new(PendingRegistry::default()),
            staging: parking_lot::Mutex::new(HashMap::new()),
            open_txns: parking_lot::Mutex::new(HashMap::new()),
            recovered: parking_lot::Mutex::new(HashMap::new()),
            agreed_records: parking_lot::Mutex::new(std::collections::VecDeque::new()),
            applied_remote: AtomicU64::new(0),
            applied_local: AtomicU64::new(0),
            ddl: std::sync::OnceLock::new(),
        }
    }

    /// Hands the applier the schema change path, once the server it needs is
    /// built. Called exactly once, before the node serves anything
    pub fn attach_ddl_runner(&self, runner: Arc<dyn DdlRunner>) {
        let _ = self.ddl.set(runner);
    }

    pub fn pending(&self) -> &Arc<PendingRegistry> {
        &self.pending
    }

    pub fn epoch(&self) -> u64 {
        self.epoch
    }

    pub fn node_id(&self) -> u64 {
        self.node_id
    }

    pub fn applied_remote(&self) -> u64 {
        self.applied_remote.load(Ordering::Relaxed)
    }

    pub fn applied_local(&self) -> u64 {
        self.applied_local.load(Ordering::Relaxed)
    }

    /// Transactions from other nodes still waiting for the chunk that
    /// completes them
    pub fn staged(&self) -> usize {
        self.staging.lock().len()
    }

    /// Transactions spanning entries whose completing entry has not applied,
    /// each of which pins the replay floor and the log at its first chunk
    pub fn open_spanning(&self) -> usize {
        self.open_txns.lock().len()
    }

    /// Handles one `Data` entry.
    ///
    /// Three shapes, and the branch between them is which of them this node
    /// is: it holds the transaction that produced this and finishes it, it is
    /// waiting for work it proposed and has not done, or it holds nothing and
    /// replays what the entry describes
    pub async fn apply_data(&self, index: u64, term: u64, payload: &[u8]) -> Result<()> {
        let (header, reader) = ChangesetReader::open(payload)?;
        let origin = header.origin;

        // A transaction committed here before a restart. Replay passes its
        // entries again on the way back up, and staging them again would
        // double its rows, so they go by untouched
        if self.skip_recovered(&origin, header.is_last()) {
            return Ok(());
        }

        // A transaction that spans entries pins the replay floor and the log
        // at its first chunk from the moment that chunk applies until the
        // entry that completes or abandons it
        if header.chunk_seq == 0 && !header.is_last() {
            self.open_txns.lock().entry(origin).or_insert((index, term));
        }
        if header.is_last() {
            self.open_txns.lock().remove(&origin);
        }

        let mine = origin.node == self.node_id && origin.epoch == self.epoch;
        if mine {
            if !header.is_last() {
                if self.pending.contains(&origin) {
                    // The transaction is still running in this process and
                    // its writes are already here, so the chunk asks nothing
                    // of the applier until the entry that completes it
                    return Ok(());
                }
                // The transaction died without finishing, a step-down failed
                // it or its connection dropped, so this node replays its
                // entries the way any other node would
                return self.replay_entry(index, term, header, reader).await;
            }
            match self.pending.take(&origin) {
                Some(PendingCommit::Commit { txn, done }) => {
                    return self.finish_local(index, header, txn, done).await;
                }
                Some(PendingCommit::Statement { turn }) => {
                    let (done, answered) = tokio::sync::oneshot::channel();
                    if turn.send(done).is_ok() {
                        if let Ok(outcome) = answered.await {
                            self.applied_local.fetch_add(1, Ordering::Relaxed);
                            if let Err(e) = outcome {
                                tracing::warn!(
                                    error = %e,
                                    "a schema change the group agreed to was refused here"
                                );
                            }
                            return Ok(());
                        }
                    }
                    // The connection is gone, so this node runs it the way
                    // every other node is about to
                    return self.replay_entry(index, term, header, reader).await;
                }
                // A final chunk with no commit waiting on it: the caller gave
                // up or was never going to ask, and the entry is replayed the
                // way a dead process's would be
                Some(PendingCommit::Running) | None => {}
            }
        }

        // Either another node's, or this node's from a previous life, or this
        // node's whose transaction is gone. The process holding the
        // transaction is not going to finish it, so it is replayed, and that
        // is the whole of what a restarted leader needs
        self.replay_entry(index, term, header, reader).await
    }

    /// Whether this entry belongs to a transaction already committed here
    /// before a restart, forgetting it once its final chunk has gone by
    fn skip_recovered(&self, origin: &Origin, last: bool) -> bool {
        let mut recovered = self.recovered.lock();
        if recovered.is_empty() || !recovered.contains_key(origin) {
            return false;
        }
        if last {
            recovered.remove(origin);
        }
        true
    }

    /// Seeds the recovered state read back from the write-ahead log, called
    /// once at startup before anything applies.
    ///
    /// The records' positions go into the retention pin as well, so the
    /// checkpoint that runs five minutes after a restart cannot reclaim the
    /// records a second restart would still need
    pub fn set_recovered(&self, committed: &[zyron_wal::RecoveredCommit]) {
        let mut recovered = self.recovered.lock();
        let mut records = self.agreed_records.lock();
        for commit in committed {
            recovered.insert(
                Origin {
                    node: commit.stamp.origin_node,
                    epoch: commit.stamp.origin_epoch,
                    txn: commit.stamp.origin_txn,
                },
                commit.stamp.index,
            );
            records.push_back((commit.lsn, commit.stamp.index));
        }
    }

    /// The oldest write-ahead log position a checkpoint must retain for this
    /// node's own restart, or None when the raft snapshot covers everything.
    ///
    /// `snapshot_point` is the entry the raft log is compacted to. A restart
    /// replays from at least there, so an agreed commit at or below it will
    /// never be asked about again and its record may go. Everything above it
    /// is exactly what a restart reads to avoid committing twice
    pub fn wal_retention_pin(&self, snapshot_point: u64) -> Option<zyron_wal::record::Lsn> {
        let mut records = self.agreed_records.lock();
        while let Some((_, index)) = records.front() {
            if *index > snapshot_point {
                break;
            }
            records.pop_front();
        }
        records.front().map(|(lsn, _)| *lsn)
    }

    /// Books one agreed commit's record position for the retention pin.
    ///
    /// The floor this commit recorded also prunes: a restart reads the floor
    /// from the newest surviving record and skips only entries above it, so
    /// an older record whose entry sits at or below this floor will never be
    /// asked about again. Pruning here rather than only at checkpoint time is
    /// what keeps the list at a handful of entries however rarely a
    /// checkpoint runs, instead of one entry per commit forever
    fn record_agreed(&self, lsn: zyron_wal::record::Lsn, index: u64, floor: u64) {
        let mut records = self.agreed_records.lock();
        while let Some((_, front_index)) = records.front() {
            if *front_index > floor {
                break;
            }
            records.pop_front();
        }
        records.push_back((lsn, index));
    }

    /// The highest entry index below which nothing would need replay after a
    /// crash at this moment. An open spanning transaction pins it at the
    /// entry before its first chunk, because a crash loses its staged writes
    /// and they come back only by replaying it from the start
    fn replay_floor(&self, index: u64) -> u64 {
        let open = self.open_txns.lock();
        match open
            .values()
            .map(|(first, _)| first.saturating_sub(1))
            .min()
        {
            Some(pinned) => pinned.min(index),
            None => index,
        }
    }

    /// The highest index the raft log may be compacted to without stranding a
    /// restart of this node, which is the same floor the commit records carry
    pub fn retain_floor(&self) -> u64 {
        self.open_txns
            .lock()
            .values()
            .map(|(first, _)| first.saturating_sub(1))
            .min()
            .unwrap_or(u64::MAX)
    }

    /// Carries out what an entry describes, on a node that did not produce it.
    async fn replay_entry(
        &self,
        index: u64,
        term: u64,
        header: ChangesetHeader,
        reader: ChangesetReader<'_>,
    ) -> Result<()> {
        if header.is_abort() {
            self.abandon(header.origin);
            return Ok(());
        }
        self.stage(index, term, header, reader).await
    }

    /// Finishes a transaction this process proposed.
    ///
    /// The commit record is written here and the caller is answered without
    /// waiting for it to reach the disk. The entry is already on a majority's
    /// log, so the durability the caller was promised is already provided, and
    /// a lost commit record leaves the applied index behind and the entry is
    /// replayed on the way back up
    async fn finish_local(
        &self,
        index: u64,
        header: ChangesetHeader,
        mut txn: Transaction,
        done: tokio::sync::oneshot::Sender<Result<Transaction>>,
    ) -> Result<()> {
        let txn_id = txn.txn_id();
        let stamp = self.agreed_stamp(index, header.origin);
        let outcome = self.engine.txn_manager.commit_agreed(&mut txn, &stamp);
        let answer = match outcome {
            Ok(lsn) => {
                self.record_agreed(lsn, index, stamp.floor);
                // Lake versions this transaction staged become visible with it
                match zyron_lake::publish_txn(&self.engine.data_dir, txn_id) {
                    Ok(_) => Ok(txn),
                    Err(e) => Err(e),
                }
            }
            Err(e) => Err(e),
        };
        let failed = answer.is_err();
        let error = answer.as_ref().err().map(|e| e.to_string());
        let _ = done.send(answer);
        self.applied_local.fetch_add(1, Ordering::Relaxed);
        if failed {
            // A commit that cannot be completed locally is not something this
            // node can apply around: the group has agreed the transaction
            // happened, so a node that cannot record it holds different data
            return Err(ZyronError::Internal(format!(
                "a committed transaction could not be recorded locally: {}",
                error.unwrap_or_default()
            )));
        }
        Ok(())
    }

    /// Replays one chunk into the transaction it belongs to.
    async fn stage(
        &self,
        index: u64,
        term: u64,
        header: ChangesetHeader,
        reader: ChangesetReader<'_>,
    ) -> Result<()> {
        let origin = header.origin;
        let (txn_id, ctx) = {
            let mut staging = self.staging.lock();
            match staging.get(&origin) {
                Some(staged) => {
                    if staged.next_chunk != header.chunk_seq {
                        return Err(ZyronError::RaftLogCorrupted {
                            index,
                            reason: format!(
                                "chunk {} of a transaction arrived where {} was expected",
                                header.chunk_seq, staged.next_chunk
                            ),
                        });
                    }
                    (staged.txn.txn_id(), Arc::clone(&staged.ctx))
                }
                None => {
                    if header.chunk_seq != 0 {
                        return Err(ZyronError::RaftLogCorrupted {
                            index,
                            reason: format!(
                                "a transaction starts at chunk {} rather than its first",
                                header.chunk_seq
                            ),
                        });
                    }
                    let Some(host) = self.ddl.get() else {
                        return Err(ZyronError::Internal(
                            "a changeset reached the applier before this node finished starting"
                                .into(),
                        ));
                    };
                    let txn = self
                        .engine
                        .txn_manager
                        .begin(IsolationLevel::ReadCommitted)?;
                    let txn_id = txn.txn_id();
                    let ctx = host.apply_context(txn_id, txn.snapshot.clone());
                    staging.insert(
                        origin,
                        StagedTxn {
                            txn,
                            ctx: Arc::clone(&ctx),
                            term,
                            next_chunk: 0,
                        },
                    );
                    (txn_id, ctx)
                }
            }
        };

        let outcome = self.replay(&ctx, origin, reader).await;
        if let Err(e) = outcome {
            self.abandon(origin);
            return Err(e);
        }

        let mut staging = self.staging.lock();
        let Some(staged) = staging.get_mut(&origin) else {
            return Err(ZyronError::Internal(
                "a staged transaction vanished while its chunk was being applied".into(),
            ));
        };
        staged.next_chunk = header.chunk_seq + 1;
        if !header.is_last() {
            return Ok(());
        }
        let Some(mut staged) = staging.remove(&origin) else {
            return Err(ZyronError::Internal(
                "a staged transaction vanished as it completed".into(),
            ));
        };
        drop(staging);

        let stamp = self.agreed_stamp(index, origin);
        let lsn = self
            .engine
            .txn_manager
            .commit_agreed(&mut staged.txn, &stamp)?;
        self.record_agreed(lsn, index, stamp.floor);
        zyron_lake::publish_txn(&self.engine.data_dir, txn_id)?;
        self.applied_remote.fetch_add(1, Ordering::Relaxed);
        Ok(())
    }

    /// The stamp one agreed commit writes into its record: the entry that
    /// completed it, the replay floor true at this moment, and whose
    /// transaction it was
    fn agreed_stamp(&self, index: u64, origin: Origin) -> zyron_wal::AgreedCommit {
        zyron_wal::AgreedCommit {
            index,
            floor: self.replay_floor(index),
            origin_node: origin.node,
            origin_epoch: origin.epoch,
            origin_txn: origin.txn,
        }
    }

    /// Applies the operations of one chunk.
    async fn replay(
        &self,
        ctx: &Arc<ExecutionContext>,
        origin: Origin,
        reader: ChangesetReader<'_>,
    ) -> Result<()> {
        for op in reader {
            let op = op?;
            match &op {
                ChangesetOp::Insert { .. }
                | ChangesetOp::Delete { .. }
                | ChangesetOp::Update { .. } => {
                    zyron_executor::replication::apply::apply_op(ctx, &op).await?;
                }
                ChangesetOp::Truncate { table_id } => {
                    self.apply_truncate(ctx, *table_id).await?;
                }
                ChangesetOp::LakeVersion {
                    table_id,
                    version,
                    version_file,
                } => {
                    self.apply_lake_version(origin, *table_id, *version, version_file)?;
                }
                ChangesetOp::Sequence {
                    sequence_id,
                    last_value,
                } => {
                    self.apply_sequence(*sequence_id, *last_value).await?;
                }
                ChangesetOp::Ddl {
                    sql,
                    user,
                    database,
                    search_path,
                    actor_role_id,
                } => {
                    let context = zyron_executor::replication::StatementContext {
                        user: (*user).to_string(),
                        database: (*database).to_string(),
                        search_path: search_path.iter().map(|s| (*s).to_string()).collect(),
                        actor_role_id: *actor_role_id,
                    };
                    self.apply_ddl(ctx, sql, &context).await?;
                }
            }
        }
        Ok(())
    }

    async fn apply_truncate(&self, ctx: &Arc<ExecutionContext>, table_id: u32) -> Result<()> {
        let entry = ctx.get_table_entry(zyron_catalog::TableId(table_id))?;
        let sql = format!("TRUNCATE TABLE {}", quote_ident(&entry.name));
        self.run_dispatched(&sql, &self.local_context(), ctx.txn_id)
            .await
    }

    /// Applies a lake commit by replaying the leader's version file into this
    /// node's own log.
    ///
    /// The entries name data files by partition id, and on shared storage
    /// those files are already readable here, so this moves metadata and
    /// copies no data at all
    fn apply_lake_version(
        &self,
        origin: Origin,
        table_id: u32,
        version: u64,
        version_file: &[u8],
    ) -> Result<()> {
        let txn_id = {
            let staging = self.staging.lock();
            let Some(staged) = staging.get(&origin) else {
                return Err(ZyronError::Internal(
                    "a lake commit names a transaction that is not staged here".into(),
                ));
            };
            staged.txn.txn_id()
        };
        let paths = zyron_lake::LakePaths::new(&self.engine.data_dir, table_id);
        let data =
            zyron_lake::VersionFileData::decode(version_file, &format!("version {version}"))?;
        let followed = zyron_lake::FollowedVersion {
            version,
            timestamp_us: data.header.timestamp_us,
            operation: data.header.operation,
            entries: data.entries,
        };
        let log =
            zyron_lake::TransactionLog::open_shared(paths.clone(), &zyron_lake::AllCommitted)?;
        // Staged under this node's transaction, so the rows it names become
        // visible with the transaction rather than the moment the chunk lands
        zyron_lake::apply_versions_under(&log, std::slice::from_ref(&followed), txn_id)?;
        Ok(())
    }

    /// Moves a sequence's counter to where the leader left it.
    ///
    /// The values themselves are already in the rows, so nothing here draws a
    /// number. This exists so a node elected later does not hand out numbers
    /// that are already in the table
    async fn apply_sequence(&self, sequence_id: u32, last_value: i64) -> Result<()> {
        self.engine
            .catalog
            .set_sequence_reserved(sequence_id, last_value)
            .await
    }

    /// Runs a statement through the dispatcher, which is where schema changes
    /// are carried out.
    ///
    /// Every node runs it, this one included, from the same applied position,
    /// so the object ids the catalog hands out match without being shipped and
    /// validation, file creation and index build all happen where the objects
    /// they describe are going to live
    async fn run_dispatched(
        &self,
        sql: &str,
        context: &zyron_executor::replication::StatementContext,
        apply_txn_id: u64,
    ) -> Result<()> {
        let Some(runner) = self.ddl.get() else {
            return Err(ZyronError::Internal(
                "a schema change reached the applier before this node finished starting".into(),
            ));
        };
        runner.run(sql, context, apply_txn_id).await
    }

    /// The context a change this node makes on its own behalf runs under
    fn local_context(&self) -> zyron_executor::replication::StatementContext {
        zyron_executor::replication::StatementContext {
            user: "zyron".to_string(),
            database: "zyron".to_string(),
            search_path: zyron_catalog::default_search_path(),
            // The node acting for itself, so there is no operator role behind
            // it and nothing to carry
            actor_role_id: None,
        }
    }

    async fn apply_ddl(
        &self,
        ctx: &Arc<ExecutionContext>,
        sql: &str,
        context: &zyron_executor::replication::StatementContext,
    ) -> Result<()> {
        // A schema change the group agreed to and this node then refused, for
        // a name that is already taken or one that is not there, is refused
        // the same way on every node and leaves them all in the same state.
        // That is an answer rather than a failure, and stopping the apply loop
        // over it would let one client's mistake take the group down.
        //
        // A refusal that is not deterministic, a full disk say, does diverge
        // this node, and it is caught by the first entry that depends on what
        // this one was supposed to make: that one fails to apply and the node
        // stops rather than carrying on wrong
        if let Err(e) = self.run_dispatched(sql, context, ctx.txn_id).await {
            tracing::warn!(
                error = %e,
                statement = %sql,
                "a schema change the group agreed to was refused here"
            );
        }
        Ok(())
    }

    /// Throws away a staged transaction and everything it had written.
    fn abandon(&self, origin: Origin) {
        self.open_txns.lock().remove(&origin);
        let staged = self.staging.lock().remove(&origin);
        if let Some(mut staged) = staged {
            let txn_id = staged.txn.txn_id();
            if let Err(e) = self.engine.txn_manager.abort(&mut staged.txn) {
                tracing::warn!(error = %e, "a staged transaction could not be abandoned cleanly");
            }
            zyron_lake::abandon_txn(&self.engine.data_dir, txn_id);
        }
    }

    /// Abandons every staged transaction from before `term`.
    ///
    /// Called when a term-start no-op applies. Every node applies that no-op at
    /// the same position in the log, so every node abandons exactly the same
    /// set and no node is left holding a transaction the others discarded
    pub fn abandon_before_term(&self, term: u64) {
        let stale: Vec<Origin> = self
            .staging
            .lock()
            .iter()
            .filter(|(_, staged)| staged.term < term)
            .map(|(origin, _)| *origin)
            .collect();
        for origin in stale {
            tracing::info!(
                node = origin.node,
                txn = origin.txn,
                "abandoning a transaction whose leader did not finish it"
            );
            self.abandon(origin);
        }
        // Spanning transactions from before the term whose chunks stopped
        // coming stop pinning the replay floor here too, whether or not a
        // chunk of theirs was ever staged
        self.open_txns.lock().retain(|_, (_, t)| *t >= term);
    }
}

fn quote_ident(name: &str) -> String {
    format!("\"{}\"", name.replace('"', "\"\""))
}

// ---------------------------------------------------------------------------
// Proposing
// ---------------------------------------------------------------------------

/// One chunk on its way to the log, with who is waiting for it.
struct QueuedChunk {
    chunk: ChangesetChunk,
    /// Answered with the index the chunk landed at. Only the final chunk of a
    /// transaction carries one, because only its caller is waiting
    answer: Option<tokio::sync::oneshot::Sender<Result<u64>>>,
}

/// Turns sealed chunks into log entries.
///
/// Chunks go through one queue rather than being proposed where they are
/// sealed, for two reasons. A chunk is sealed inside an operator, which is
/// synchronous, and proposing is not. And draining the queue lets a burst of
/// chunks from different transactions become one append and one fsync, which
/// is the same group commit the log already does for everything else
pub struct ChunkProposer {
    tx: tokio::sync::mpsc::UnboundedSender<QueuedChunk>,
    node: Arc<RaftNode>,
    /// Where a streaming transaction announces itself, so the applier can
    /// tell its live chunks from a dead process's
    pending: Arc<PendingRegistry>,
}

impl ChunkProposer {
    pub fn start(node: Arc<RaftNode>, pending: Arc<PendingRegistry>) -> Arc<Self> {
        let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel::<QueuedChunk>();
        let proposer = Arc::new(Self {
            tx,
            node: Arc::clone(&node),
            pending,
        });
        tokio::spawn(async move {
            let mut batch: Vec<QueuedChunk> = Vec::with_capacity(64);
            let mut answers: Vec<Option<tokio::sync::oneshot::Sender<Result<u64>>>> =
                Vec::with_capacity(64);
            while let Some(first) = rx.recv().await {
                batch.push(first);
                // Everything already queued joins this append
                while let Ok(next) = rx.try_recv() {
                    batch.push(next);
                    if batch.len() >= 1024 {
                        break;
                    }
                }
                // Payloads move into the commands rather than being copied.
                // A chunk is up to the configured chunk size, and cloning
                // every one of them would put the whole replication stream
                // through the allocator a second time
                let mut commands: Vec<RaftCommand> = Vec::with_capacity(batch.len());
                for queued in batch.drain(..) {
                    commands.push(RaftCommand::Data {
                        payload: queued.chunk.payload,
                    });
                    answers.push(queued.answer);
                }
                let outcome = node.propose_batch_detached(commands);
                match outcome {
                    Ok(first_index) => {
                        for (offset, answer) in answers.drain(..).enumerate() {
                            if let Some(answer) = answer {
                                let _ = answer.send(Ok(first_index + offset as u64));
                            }
                        }
                    }
                    Err(e) => {
                        for answer in answers.drain(..).flatten() {
                            let _ = answer.send(Err(clone_error(&e)));
                        }
                    }
                }
            }
        });
        proposer
    }

    /// Queues the final chunk and answers with the index it took.
    pub fn propose_final(
        &self,
        chunk: ChangesetChunk,
    ) -> Result<tokio::sync::oneshot::Receiver<Result<u64>>> {
        let (answer, wait) = tokio::sync::oneshot::channel();
        self.tx
            .send(QueuedChunk {
                chunk,
                answer: Some(answer),
            })
            .map_err(|_| {
                ZyronError::Internal("the consensus proposer is no longer running".into())
            })?;
        Ok(wait)
    }

    pub fn node(&self) -> &Arc<RaftNode> {
        &self.node
    }
}

impl ChangesetSink for ChunkProposer {
    fn emit(&self, chunk: ChangesetChunk) -> Result<()> {
        // A chunk that does not complete its transaction means the
        // transaction is live in this process, and the applier has to know
        // that before the chunk can possibly come back through the log,
        // which is why the announcement precedes the send
        if !chunk.last {
            self.pending.register_running_if_absent(chunk.origin);
        }
        self.tx
            .send(QueuedChunk {
                chunk,
                answer: None,
            })
            .map_err(|_| ZyronError::Internal("the consensus proposer is no longer running".into()))
    }

    fn barrier_index(&self) -> u64 {
        self.node.last_applied()
    }
}

/// Copies an error across the channel boundary.
///
/// The proposer answers many waiters from one failure and `ZyronError` is not
/// `Clone`, so the shape that matters to a caller is preserved and the rest is
/// carried as text
fn clone_error(e: &ZyronError) -> ZyronError {
    match e {
        ZyronError::NotLeader { leader } => ZyronError::NotLeader { leader: *leader },
        ZyronError::ConsensusTimeout {
            operation,
            elapsed_ms,
        } => ZyronError::ConsensusTimeout {
            operation: operation.clone(),
            elapsed_ms: *elapsed_ms,
        },
        other => ZyronError::Internal(other.to_string()),
    }
}

/// Where a node keeps the pieces of replication a connection has to reach.
pub struct ReplicationHandle {
    pub node: Arc<RaftNode>,
    pub machine: Arc<ChangesetMachine>,
    pub proposer: Arc<ChunkProposer>,
    /// Bytes a transaction buffers before a chunk goes out
    pub chunk_bytes: usize,
    /// This process's life, stamped into every transaction it originates
    pub epoch: u64,
    /// How long a caller waits for its transaction to reach the front of the
    /// log and commit
    pub propose_timeout: std::time::Duration,
    /// Whether every member of the group runs a binary that reads the actor
    /// role off a replicated schema change.
    ///
    /// False until the upgrade service has seen the group's version floor
    /// reach [`ACTOR_ROLE_INTRODUCED_IN`], because a member that does not know
    /// the operation refuses the whole entry and stops applying. False is the
    /// answer that behaves the way the release before this one did, so the
    /// wrong answer costs an owner column and never a stalled member
    pub group_carries_actor_role: Arc<std::sync::atomic::AtomicBool>,
}

/// The release whose applier reads the actor role off a schema change. The
/// leader holds the field back until every member of the group runs this or
/// later
pub const ACTOR_ROLE_INTRODUCED_IN: zyron_common::format::BinaryVersion =
    zyron_common::format::BinaryVersion::new(0, 13, 0);

impl ReplicationHandle {
    /// The origin stamp for one transaction.
    pub fn origin(&self, txn_id: u64) -> Origin {
        Origin {
            node: self.node.id(),
            epoch: self.epoch,
            txn: txn_id,
        }
    }

    /// A changeset for one transaction to accumulate into.
    pub fn changeset(&self, txn_id: u64) -> Arc<zyron_executor::replication::TxnChangeset> {
        Arc::new(zyron_executor::replication::TxnChangeset::new(
            self.origin(txn_id),
            self.chunk_bytes,
            Arc::clone(&self.proposer) as Arc<dyn ChangesetSink>,
        ))
    }
}

/// Reads the version files a transaction has staged, so a lake commit can be
/// replicated by what it did rather than by the rows it touched.
///
/// One capture point covers append, delete, update, optimize and schema
/// change, because all of them stage the same way
pub fn capture_pending_lake_versions(
    data_dir: &Path,
    txn_id: u64,
    changeset: &zyron_executor::replication::TxnChangeset,
) -> Result<()> {
    for (root, version) in zyron_lake::pending_versions(data_dir, txn_id) {
        let paths = zyron_lake::LakePaths::from_root(&root);
        let Some(table_id) = paths.table_id() else {
            return Err(ZyronError::Internal(format!(
                "a lake log at {} is not named for a table, so it cannot be replicated",
                root.display()
            )));
        };
        let bytes = std::fs::read(paths.version_file(version)).map_err(|e| {
            ZyronError::IoError(format!(
                "read version {version} of {} for replication: {e}",
                root.display()
            ))
        })?;
        changeset.capture_lake_version(table_id, version, &bytes)?;
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// The seam the wire protocol reaches all of this through
// ---------------------------------------------------------------------------

impl zyron_wire::connection::ReplicationRouter for ReplicationHandle {
    fn changeset(&self, txn_id: u64) -> Arc<zyron_executor::replication::TxnChangeset> {
        ReplicationHandle::changeset(self, txn_id)
    }

    fn capture_lake(
        &self,
        txn_id: u64,
        changeset: &zyron_executor::replication::TxnChangeset,
    ) -> Result<()> {
        capture_pending_lake_versions(&self.machine.engine.data_dir, txn_id, changeset)
    }

    fn commit<'a>(
        &'a self,
        txn: Transaction,
        changeset: Arc<zyron_executor::replication::TxnChangeset>,
    ) -> std::pin::Pin<Box<dyn std::future::Future<Output = Result<Transaction>> + Send + 'a>> {
        Box::pin(async move {
            if !self.node.is_leader() {
                return Err(ZyronError::NotLeader {
                    leader: self.node.leader_id(),
                });
            }
            let origin = changeset.origin();
            // Read after every lock is held, which is what makes it sound: an
            // entry proposed between this index and this one was still holding
            // its own locks, so it cannot have touched the same rows and the
            // two may be applied in either order
            let barrier = self.node.last_applied();
            let Some(chunk) = changeset.seal(barrier) else {
                return Err(ZyronError::Internal(
                    "a transaction with changes sealed to nothing".into(),
                ));
            };

            let (done, wait) = tokio::sync::oneshot::channel();
            self.machine
                .pending()
                .register(origin, PendingCommit::Commit { txn, done });

            let queued = match self.proposer.propose_final(chunk) {
                Ok(queued) => queued,
                Err(e) => {
                    // Never proposed, so nothing is waiting on the applier
                    if let Some(pending) = self.machine.pending().take(&origin) {
                        drop(pending);
                    }
                    return Err(e);
                }
            };
            match queued.await {
                Ok(Ok(_index)) => {}
                Ok(Err(e)) => {
                    if let Some(pending) = self.machine.pending().take(&origin) {
                        drop(pending);
                    }
                    return Err(e);
                }
                Err(_) => {
                    if let Some(pending) = self.machine.pending().take(&origin) {
                        drop(pending);
                    }
                    return Err(ZyronError::Internal(
                        "the consensus proposer stopped before answering".into(),
                    ));
                }
            }

            match tokio::time::timeout(self.propose_timeout, wait).await {
                Ok(Ok(outcome)) => outcome,
                Ok(Err(_)) => Err(ZyronError::Internal(
                    "a proposed transaction was dropped before it committed".into(),
                )),
                Err(_) => {
                    // The local transaction is released rather than left
                    // registered, because an entry that was truncated by a
                    // new leader would otherwise leave it open forever,
                    // pinning every snapshot behind it. If the entry does
                    // commit later the applier finds nothing registered and
                    // replays it, which is the same shape as a leader that
                    // crashed after proposing. Either way the caller is told
                    // it does not know
                    if let Some(pending) = self.machine.pending().take(&origin) {
                        drop(pending);
                    }
                    Err(ZyronError::ConsensusTimeout {
                        operation: format!("commit of transaction {}", origin.txn),
                        elapsed_ms: self.propose_timeout.as_millis() as u64,
                    })
                }
            }
        })
    }

    fn begin_statement<'a>(
        &'a self,
        sql: &'a str,
        context: &'a zyron_executor::replication::StatementContext,
    ) -> std::pin::Pin<
        Box<
            dyn std::future::Future<Output = Result<tokio::sync::oneshot::Sender<Result<()>>>>
                + Send
                + 'a,
        >,
    > {
        Box::pin(begin_statement_inner(self, sql, context))
    }

    fn abort(&self, changeset: &zyron_executor::replication::TxnChangeset) {
        let barrier = self.node.last_applied();
        let Some(chunk) = changeset.seal_abort(barrier) else {
            return;
        };
        if let Err(e) = self.proposer.emit(chunk) {
            tracing::warn!(error = %e, "a rolled back transaction could not be withdrawn from the group");
        }
    }
}

/// Puts a schema change to the group and hands back the reply channel once
/// this node's turn comes up.
///
/// The statement is agreed before it runs anywhere, which is what makes the
/// object ids the catalog allocates match across the group without any of them
/// being replicated: every node runs it from the same applied position and
/// nothing else advances that allocator
async fn begin_statement_inner(
    handle: &ReplicationHandle,
    sql: &str,
    context: &zyron_executor::replication::StatementContext,
) -> Result<tokio::sync::oneshot::Sender<Result<()>>> {
    if !handle.node.is_leader() {
        return Err(ZyronError::NotLeader {
            leader: handle.node.leader_id(),
        });
    }
    // A schema change gets a transaction id of its own so it can be told apart
    // from the transactions running alongside it
    let txn_id = handle
        .node
        .id()
        .wrapping_mul(0x9E37_79B9_7F4A_7C15)
        .wrapping_add(STATEMENT_SEQUENCE.fetch_add(1, Ordering::Relaxed));
    let origin = handle.origin(txn_id);
    let changeset = handle.changeset(txn_id);
    // The actor role travels only once the whole group reads it. A member on
    // an earlier release refuses an operation tag it does not know and stops
    // applying, so this is held back rather than sent and hoped for
    let held_back;
    let context = match context.actor_role_id {
        Some(_) if !handle.group_carries_actor_role.load(Ordering::Relaxed) => {
            held_back = zyron_executor::replication::StatementContext {
                actor_role_id: None,
                ..context.clone()
            };
            &held_back
        }
        _ => context,
    };
    changeset.capture_ddl(sql, context)?;
    let barrier = handle.node.last_applied();
    let Some(chunk) = changeset.seal(barrier) else {
        return Err(ZyronError::Internal(
            "a schema change sealed to nothing".into(),
        ));
    };

    let (turn, wait) = tokio::sync::oneshot::channel();
    handle
        .machine
        .pending()
        .register(origin, PendingCommit::Statement { turn });

    let queued = match handle.proposer.propose_final(chunk) {
        Ok(queued) => queued,
        Err(e) => {
            handle.machine.pending().take(&origin);
            return Err(e);
        }
    };
    match queued.await {
        Ok(Ok(_)) => {}
        Ok(Err(e)) => {
            handle.machine.pending().take(&origin);
            return Err(e);
        }
        Err(_) => {
            handle.machine.pending().take(&origin);
            return Err(ZyronError::Internal(
                "the consensus proposer stopped before answering".into(),
            ));
        }
    }

    match tokio::time::timeout(handle.propose_timeout, wait).await {
        Ok(Ok(done)) => Ok(done),
        Ok(Err(_)) => Err(ZyronError::Internal(
            "a proposed schema change was dropped before its turn".into(),
        )),
        Err(_) => {
            // Released so a truncated entry does not leave a turn registered
            // forever. If the entry commits later the applier finds nothing
            // and runs the statement itself, the way it would for a dead
            // connection
            handle.machine.pending().take(&origin);
            Err(ZyronError::ConsensusTimeout {
                operation: "a schema change".into(),
                elapsed_ms: handle.propose_timeout.as_millis() as u64,
            })
        }
    }
}

/// Distinguishes one schema change from the next on the same node.
static STATEMENT_SEQUENCE: AtomicU64 = AtomicU64::new(1);

/// Runs a schema change through the dispatcher, with no session and no
/// enclosing transaction.
///
/// That is exactly the shape a replayed change has: it belongs to no
/// connection, it carries no session state that could make it mean something
/// different here, and the transaction it needs is its own
pub struct DispatchedDdl {
    server: std::sync::Weak<zyron_wire::connection::ServerState>,
    catalog: Arc<Catalog>,
    wal: Arc<WalWriter>,
    buffer_pool: Arc<BufferPool>,
    disk_manager: Arc<DiskManager>,
}

impl DispatchedDdl {
    pub fn new(server: &Arc<zyron_wire::connection::ServerState>) -> Arc<Self> {
        Arc::new(Self {
            server: Arc::downgrade(server),
            catalog: Arc::clone(&server.catalog),
            wal: Arc::clone(&server.wal),
            buffer_pool: Arc::clone(&server.buffer_pool),
            disk_manager: Arc::clone(&server.disk_manager),
        })
    }
}

impl DdlRunner for DispatchedDdl {
    fn apply_context(&self, txn_id: u64, snapshot: Snapshot) -> Arc<ExecutionContext> {
        match self.server.upgrade() {
            Some(server) => Arc::new(server.apply_context(txn_id, snapshot)),
            // The server is gone, so nothing this context produced could be
            // served anyway. A bare context still refuses every write it is
            // asked for, loudly, rather than writing rows into a node that is
            // shutting down
            None => Arc::new(ExecutionContext::new(
                Arc::clone(&self.catalog),
                Arc::clone(&self.wal),
                Arc::clone(&self.buffer_pool),
                Arc::clone(&self.disk_manager),
                txn_id,
                snapshot,
            )),
        }
    }

    fn run<'a>(
        &'a self,
        sql: &'a str,
        context: &'a zyron_executor::replication::StatementContext,
        apply_txn_id: u64,
    ) -> std::pin::Pin<Box<dyn std::future::Future<Output = Result<()>> + Send + 'a>> {
        Box::pin(async move {
            let Some(server) = self.server.upgrade() else {
                return Err(ZyronError::Internal(
                    "a schema change arrived after this node stopped serving".into(),
                ));
            };
            let statements = zyron_parser::parse(sql)?;
            for statement in statements {
                // The originator's own session, rebuilt, so an unqualified
                // name resolves here to the object it named there and an
                // object this creates is owned by the same user
                let mut session = Some(zyron_wire::session::Session::new(
                    context.user.clone(),
                    context.database.clone(),
                    zyron_catalog::DatabaseId(1),
                ));
                if let Some(session) = session.as_mut() {
                    session.search_path = context.search_path.clone();
                    // The role the statement ran under, carried rather than
                    // enforced. The group agreed the statement before it ran
                    // anywhere, so re-deciding here whether it was allowed
                    // could have this node refuse what the others carried out.
                    // What this settles is who owns what it creates
                    session.replicated_actor = context.actor_role_id;
                    // An online build this statement runs cannot wait for the
                    // transaction the statement is being replayed under
                    session.apply_txn_id = Some(apply_txn_id);
                }
                let mut txn = None;
                let mut branch = None;
                let handled = zyron_wire::ddl_dispatch::try_handle_ddl_utility(
                    &statement,
                    &server,
                    &mut session,
                    &mut txn,
                    &mut branch,
                    sql,
                )
                .await;
                match handled {
                    Some(Ok(_)) => {}
                    Some(Err(e)) => return Err(ZyronError::Internal(e.to_string())),
                    None => {
                        return Err(ZyronError::Internal(format!(
                            "the dispatcher did not carry out a replicated statement: {sql}"
                        )));
                    }
                }
                if let Some(mut txn) = txn {
                    if txn.wrote_data() {
                        server.txn_manager.commit(&mut txn).await?;
                    } else {
                        server.txn_manager.commit_read_only(&mut txn)?;
                    }
                }
            }
            Ok(())
        })
    }
}
