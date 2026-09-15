//! Verifiable tables on the wire.
//!
//! Three paths meet here. The write path hands the rows of a verified table
//! to a per-transaction accumulator as they are written. The commit path
//! links what the accumulator holds onto each table's chain, in the
//! transaction's own log chain ahead of its commit record, so an entry
//! reaches the chain exactly when the rows it covers reach the table. The
//! read path walks a chain, reads the rows of the commits it is asked to
//! check back through one pass over the table, and holds the result against
//! the anchors taken over the chain.
//!
//! On a node that leads no group the connection links the entry itself. On a
//! member of one the hash travels in the changeset and every member links
//! the entry as it applies the changeset, in the log's order, so every
//! member's chain stands at the same head for the same commit.

use std::collections::HashMap;
use std::sync::Arc;
use std::sync::atomic::AtomicU32;

use zyron_catalog::TableId;
use zyron_common::{Result, ZyronError};
use zyron_executor::context::{CommitChainSink, ExecutionContext};
use zyron_lifecycle::verify::anchor::{Anchor, ExportedAnchor};
use zyron_lifecycle::verify::{
    self, ChainHash, ChainRegistry, CommitFields, CommitHash, CommitRows, PendingChainWrites,
    PendingTable, RowHashes, RowMode, RowRequest, RowsHasher, SetHasher, VerifyOutcome, VerifyRun,
};

use crate::connection::ServerState;

/// Commits a sampled run reads the rows of when nothing names a number
pub const DEFAULT_SAMPLE: u64 = 64;

/// Pages a pass over a table holds at a time. A verification and a genesis
/// read the table in windows of this many pages, pinning what the pool
/// holds and carrying the rest without putting it in the pool, so a table
/// larger than the pool is read with a bounded footprint and without
/// evicting the pages the statements the node is serving are using
const SCAN_WINDOW_PAGES: u32 = 1024;

/// Walks every page of a heap in windows, handing each window's guard to
/// `visit` and stopping when it answers false or when `cancelled` says so
fn for_each_window(
    heap: &zyron_storage::HeapFile,
    cancelled: &dyn Fn() -> bool,
    mut visit: impl FnMut(&zyron_storage::ScanGuard<'_>) -> bool,
) -> Result<()> {
    let pages = heap.page_count();
    let mut first = 0u32;
    while first < pages {
        if cancelled() {
            return Err(ZyronError::Internal("Query cancelled".into()));
        }
        let guard = heap.scan_window(first, SCAN_WINDOW_PAGES, false)?;
        let more = visit(&guard);
        drop(guard);
        if !more {
            break;
        }
        first = first.saturating_add(SCAN_WINDOW_PAGES);
    }
    Ok(())
}

fn now_micros() -> i64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_micros() as i64)
        .unwrap_or(0)
}

// ---------------------------------------------------------------------------
// The write path
// ---------------------------------------------------------------------------

/// Takes the rows a statement wrote into a verified table.
///
/// One per connection, shared with every context the connection builds, so
/// a write inside a trigger body or a procedure lands in the same
/// transaction's accumulator as one at the top level. Whether a table's
/// rows are chained is the registry's answer, so a table becomes verifiable
/// for the write path at one instant on this node whatever entry a
/// statement resolved
pub struct ChainWriteSink {
    pending: Arc<PendingChainWrites>,
    registry: Arc<ChainRegistry>,
}

impl ChainWriteSink {
    pub fn new(pending: Arc<PendingChainWrites>, registry: Arc<ChainRegistry>) -> Self {
        Self { pending, registry }
    }
}

impl CommitChainSink for ChainWriteSink {
    fn open_write(&self, table_id: u32) -> Option<Arc<AtomicU32>> {
        if self.registry.chained(table_id).is_some() {
            Some(self.pending.cursor(table_id))
        } else {
            self.pending.touched(table_id);
            None
        }
    }

    fn rows_removed(&self, table_id: u32) {
        self.pending.removed(table_id);
    }

    fn rows(
        &self,
        table_id: u32,
        schema_epoch: u16,
        rows: &mut dyn Iterator<Item = &[u8]>,
        first_position: u64,
        last_position: u64,
    ) {
        self.pending
            .record(table_id, schema_epoch, rows, first_position, last_position);
    }
}

/// The sink over one accumulator, built once per connection or per
/// generated body. None on a node with no chains, which is what keeps the
/// write path of a node that holds no verified table unchanged
pub fn chain_sink(
    server: &ServerState,
    pending: &Arc<PendingChainWrites>,
) -> Option<Arc<dyn CommitChainSink>> {
    let registry = server.chain_registry.as_ref()?;
    Some(Arc::new(ChainWriteSink::new(
        Arc::clone(pending),
        Arc::clone(registry),
    )) as Arc<dyn CommitChainSink>)
}

/// Installs the accumulator a statement's writes hash into.
///
/// Set on every context, so a write inside a subquery, a trigger body or a
/// generated body reaches the same transaction's chain entry as one at the
/// top level
pub fn install_chain_writes(
    server: &ServerState,
    ctx: &mut ExecutionContext,
    pending: &Arc<PendingChainWrites>,
) {
    ctx.chain_sink = chain_sink(server, pending);
}

/// Tells the registry which tables the catalog holds as verified.
///
/// Run once at start, before anything writes, so a table verified before
/// the restart chains every transaction of this process. A table the
/// catalog records no genesis position for is asked its chain's first
/// entry, which is where a genesis written at a start lands
pub fn register_chained_tables(registry: &ChainRegistry, catalog: &zyron_catalog::Catalog) {
    for entry in catalog.list_all_tables() {
        if !entry.lifecycle.verified {
            continue;
        }
        let mut genesis_at = entry.lifecycle.genesis_at;
        if genesis_at == 0
            && let Ok(chain) = registry.chain(entry.id.0)
            && let Ok(first) = chain.read_range(0, 0)
            && first.first().is_some_and(|entry| entry.genesis)
        {
            genesis_at = 1;
        }
        registry.chain_from(entry.id.0, 0, algorithm_of(&entry), genesis_at);
    }
}

/// Covers the rows the compliance log held before its chain began.
///
/// The log is registered as a verified table at every start and its rows
/// from before the chain existed have no entry, so the start that finds
/// the chain empty writes the genesis over what the log holds. Nothing is
/// running yet, so the fence takes in every row and the entry lands first
pub async fn cover_compliance_log(server: &Arc<ServerState>) -> Result<()> {
    let Some(registry) = server.chain_registry.as_ref() else {
        return Ok(());
    };
    let Some(table) = compliance_log_table(server) else {
        return Ok(());
    };
    let chain = registry.chain(table.id.0)?;
    if chain.head().commits > 0 {
        return Ok(());
    }
    let algorithm = algorithm_of(&table);
    registry.chain_from(table.id.0, server.txn_manager.next_txn_id(), algorithm, 0);
    match write_genesis(server, &table, None).await {
        Ok(Some(_)) => Ok(()),
        Ok(None) => {
            // An empty log has nothing to cover, and every transaction of
            // this process is chained from its first commit
            registry.chain_from(table.id.0, 0, algorithm, 0);
            Ok(())
        }
        Err(e) => {
            registry.chain_from(table.id.0, 0, algorithm, 0);
            Err(e)
        }
    }
}

// ---------------------------------------------------------------------------
// The commit path
// ---------------------------------------------------------------------------

/// Closes what a committing transaction wrote into chained tables.
///
/// One hash per table, or a refusal: a transaction that started before a
/// table became verifiable and wrote to it, or that removed rows from a
/// chained table, cannot be chained and does not commit
pub fn take_commit_chains(
    server: &ServerState,
    txn_id: u64,
    pending: &Arc<PendingChainWrites>,
) -> Result<Vec<PendingTable>> {
    if pending.is_empty() {
        return Ok(Vec::new());
    }
    let Some(registry) = server.chain_registry.as_ref() else {
        // A table cannot be chained without a registry, so nothing could
        // have been hashed. Clearing keeps the connection's accumulator
        // from carrying a stale record into the next transaction
        pending.clear();
        return Ok(Vec::new());
    };
    pending.take(txn_id, registry)
}

/// Links what a committing transaction wrote onto each table's chain, on a
/// node that leads no group.
///
/// Called with the transaction still open, so the entry is written into its
/// own log chain ahead of its commit record: a stop before the commit
/// record leaves neither the rows nor the entry, and one after it leaves
/// both. The version is read under the chain's lock, so versions rise
/// along the chain
pub fn link_commit_chains(
    server: &ServerState,
    txn: &mut zyron_storage::txn::Transaction,
    tables: &[PendingTable],
) -> Result<Vec<CommitHash>> {
    if tables.is_empty() {
        return Ok(Vec::new());
    }
    let Some(registry) = server.chain_registry.as_ref() else {
        return Err(ZyronError::Internal(
            "a transaction hashed rows for a verified table and this node holds no chains".into(),
        ));
    };
    let at = now_micros();
    let mut written = Vec::with_capacity(tables.len());
    for table in tables {
        let chain = registry.chain(table.table_id)?;
        let linked = chain.append(
            CommitFields {
                txn_id: txn.txn_id,
                rows_hash: table.rows_hash,
                row_count: table.row_count,
                commit_ts: at,
                algorithm_id: table.algorithm_id,
                genesis: false,
            },
            || server.wal.next_lsn().0,
        )?;
        let lsn = server.wal.log_commit_chain_entry(
            txn.txn_id,
            txn.last_lsn(),
            linked.table_id,
            linked.sequence,
            &linked.encode(),
        )?;
        txn.set_last_lsn(lsn);
        txn.mark_wrote_data();
        written.push(linked);
    }
    Ok(written)
}

/// Closes and links a transaction's chained writes in one step, for a
/// commit on a node that leads no group
pub fn log_commit_chains(
    server: &ServerState,
    txn: &mut zyron_storage::txn::Transaction,
    pending: &Arc<PendingChainWrites>,
) -> Result<Vec<CommitHash>> {
    let tables = take_commit_chains(server, txn.txn_id, pending)?;
    link_commit_chains(server, txn, &tables)
}

/// Puts the entries a committed transaction linked onto their chains.
///
/// Called once the commit record is written, which is the point from which
/// the rows the entries cover are there. Until then the entries are held,
/// so a stop between the link and the commit leaves the chains exactly
/// where they were
pub fn publish_commit_chains(server: &ServerState, entries: &[CommitHash]) {
    let Some(registry) = server.chain_registry.as_ref() else {
        return;
    };
    for entry in entries {
        let Ok(chain) = registry.chain(entry.table_id) else {
            continue;
        };
        if let Err(e) = chain.publish(entry.txn_id) {
            // The commit stands, and the log holds the entry, so recovery
            // puts it on the chain. Reported rather than failing a commit
            // the client has already been told about
            tracing::error!(
                target: "zyron::verify",
                table_id = entry.table_id,
                error = %e,
                "a committed transaction's chain entry could not be written, the log holds it \
                 until the next start"
            );
        }
    }
}

/// Drops the entries a transaction linked and did not commit.
pub fn discard_commit_chains(server: &ServerState, entries: &[CommitHash]) {
    let Some(registry) = server.chain_registry.as_ref() else {
        return;
    };
    for entry in entries {
        let Ok(chain) = registry.chain(entry.table_id) else {
            continue;
        };
        if let Err(e) = chain.discard(entry.txn_id) {
            tracing::error!(
                target: "zyron::verify",
                table_id = entry.table_id,
                error = %e,
                "an uncommitted transaction's chain entry could not be dropped"
            );
        }
    }
}

/// Links one entry a changeset carries onto this member's chain, as the
/// applier applies the changeset.
///
/// The version is the entry's index and the instant its proposal instant,
/// both the same on every member, and the head is the one this member's
/// chain stands at, which the log's order makes the same as well. The
/// record goes into the applying transaction's own log chain ahead of its
/// commit record, and the entry is published once that commit is written
#[allow(clippy::too_many_arguments)]
pub fn link_applied_chain(
    registry: &ChainRegistry,
    wal: &zyron_wal::WalWriter,
    txn: &mut zyron_storage::txn::Transaction,
    table_id: u32,
    rows_hash: ChainHash,
    row_count: u64,
    algorithm_id: u16,
    index: u64,
    timestamp_us: i64,
) -> Result<CommitHash> {
    let chain = registry.chain(table_id)?;
    let linked = chain.append(
        CommitFields {
            txn_id: txn.txn_id,
            rows_hash,
            row_count,
            commit_ts: timestamp_us,
            algorithm_id,
            genesis: false,
        },
        || index,
    )?;
    let lsn = wal.log_commit_chain_entry(
        txn.txn_id,
        txn.last_lsn(),
        linked.table_id,
        linked.sequence,
        &linked.encode(),
    )?;
    txn.set_last_lsn(lsn);
    txn.mark_wrote_data();
    Ok(linked)
}

/// Publishes or discards the entries an applied transaction linked, by
/// how its commit went
pub fn settle_applied_chains(registry: &ChainRegistry, entries: &[CommitHash], committed: bool) {
    for entry in entries {
        let Ok(chain) = registry.chain(entry.table_id) else {
            continue;
        };
        let outcome = if committed {
            chain.publish(entry.txn_id).map(|_| ())
        } else {
            chain.discard(entry.txn_id).map(|_| ())
        };
        if let Err(e) = outcome {
            tracing::error!(
                target: "zyron::verify",
                table_id = entry.table_id,
                committed,
                error = %e,
                "an applied transaction's chain entry could not be settled"
            );
        }
    }
}

/// The scheme a table's chain links with, falling back to the one a chain
/// is started with when the table records none
pub fn algorithm_of(entry: &zyron_catalog::TableEntry) -> u16 {
    if entry.lifecycle.chain_algorithm == 0 {
        verify::DEFAULT_CHAIN_ALGORITHM
    } else {
        entry.lifecycle.chain_algorithm
    }
}

/// Puts back the chain entries a restart found in the log.
///
/// Run before anything reads a chain, with the entries of the transactions
/// that committed, in the log's order
pub fn restore_chain_entries(
    registry: &ChainRegistry,
    records: &[zyron_wal::LogRecord],
) -> Result<usize> {
    if records.is_empty() {
        return Ok(0);
    }
    let mut entries = Vec::with_capacity(records.len());
    for record in records {
        let logged = zyron_wal::CommitChainEntry::decode(&record.payload)?;
        entries.push(verify::LoggedEntry::decode(
            logged.table_id,
            logged.sequence,
            logged.record,
        )?);
    }
    verify::restore_logged_entries(registry, &entries)
}

// ---------------------------------------------------------------------------
// The compliance log
// ---------------------------------------------------------------------------

/// Appends one compliance event, extending the log's chain with it.
///
/// The row and the chain entry are one transaction, so an event that is in
/// the log is covered by the chain and one that is not is in neither. This
/// is the one mechanism every hash-chained log uses: nothing here is
/// particular to compliance beyond which table the row lands in
pub async fn append_audit(
    server: &ServerState,
    entry: zyron_catalog::schema::ComplianceLogEntry,
) -> Result<()> {
    let Some(table) = compliance_log_table(server) else {
        // The log's table is not registered on this node, so the row is
        // written where it has always been written and nothing chains it
        return server.catalog.append_compliance_log(entry).await;
    };
    let Some(registry) = server.chain_registry.as_ref() else {
        return server.catalog.append_compliance_log(entry).await;
    };
    let mut txn = server
        .txn_manager
        .begin(zyron_storage::IsolationLevel::ReadCommitted)?;
    let txn_id = txn.txn_id();
    let pending = Arc::new(PendingChainWrites::new());
    let sink = ChainWriteSink::new(Arc::clone(&pending), Arc::clone(registry));
    let cursor = sink.open_write(table.id.0);
    let (stored, tid) = server
        .catalog
        .append_compliance_log_under(entry, txn_id, cursor.as_deref())
        .await?;
    txn.mark_wrote_data();
    if cursor.is_some() {
        let bytes = stored.to_bytes();
        let position = tid.order_key();
        sink.rows(
            table.id.0,
            table.schema_epoch,
            &mut std::iter::once(bytes.as_slice()),
            position,
            position,
        );
    }
    let chained = log_commit_chains(server, &mut txn, &pending)?;
    match server.txn_manager.commit(&mut txn).await {
        Ok(()) => {
            publish_commit_chains(server, &chained);
            Ok(())
        }
        Err(e) => {
            discard_commit_chains(server, &chained);
            Err(e)
        }
    }
}

/// The compliance log's table, once it is registered.
pub fn compliance_log_table(server: &ServerState) -> Option<Arc<zyron_catalog::TableEntry>> {
    let database = server
        .catalog
        .get_database(zyron_catalog::system_catalog::SYSTEM_CATALOG_NAME)
        .ok()?;
    let schema = server
        .catalog
        .get_schema(
            database.id,
            zyron_catalog::system_catalog::COMPLIANCE_SCHEMA,
        )
        .ok()?;
    server.catalog.compliance_log_entry(schema.id)
}

// ---------------------------------------------------------------------------
// Reading a commit's rows back
// ---------------------------------------------------------------------------

/// Reads the rows of named commits out of a table's heap.
///
/// One pass over the table however many commits are named, because a
/// sampled verification of a table with a hundred thousand commits would
/// otherwise read the table once per sampled commit. A commit's rows are
/// taken in the order the heap stores them, which is the order the write
/// path hashed them in. The genesis set is every row the snapshot sees that
/// is stamped below the fence, whatever order it is stored in
pub struct HeapCommitRows {
    heap: Arc<zyron_storage::HeapFile>,
    /// What the table holds now, which decides the genesis set
    snapshot: zyron_storage::txn::Snapshot,
    /// Where the genesis set's sorted runs go when it does not fit in memory
    spill_dir: std::path::PathBuf,
    /// Reports true when the caller has cancelled, checked between pages so
    /// a verification of a large table stops where it was asked to
    cancelled: Arc<dyn Fn() -> bool + Send + Sync>,
}

impl HeapCommitRows {
    pub fn new(
        heap: Arc<zyron_storage::HeapFile>,
        snapshot: zyron_storage::txn::Snapshot,
        spill_dir: std::path::PathBuf,
        cancelled: Arc<dyn Fn() -> bool + Send + Sync>,
    ) -> Self {
        Self {
            heap,
            snapshot,
            spill_dir,
            cancelled,
        }
    }
}

impl CommitRows for HeapCommitRows {
    fn hash_commits(&self, request: &RowRequest) -> Result<RowHashes> {
        // One hasher per named transaction, found by the stamp a row carries.
        // A commit's rows sit together in the heap, so the transaction a row
        // belongs to is nearly always the one the row before it belonged to,
        // and whether that transaction is wanted or not, the answer for the
        // row before it is the answer for this one without touching the map
        let mut hashers: Vec<RowsHasher> = Vec::with_capacity(request.txn_ids.len());
        let mut by_txn: HashMap<u64, usize> = HashMap::with_capacity(request.txn_ids.len());
        for txn_id in &request.txn_ids {
            if !by_txn.contains_key(txn_id) {
                by_txn.insert(*txn_id, hashers.len());
                hashers.push(RowsHasher::new());
            }
        }
        let mut genesis = request
            .genesis_fence
            .map(|_| SetHasher::new(&self.spill_dir));
        let fence = request.genesis_fence.unwrap_or(0);
        let mut last: Option<(u64, Option<usize>)> = None;
        let mut failed: Option<ZyronError> = None;

        for_each_window(&self.heap, &*self.cancelled, |guard| {
            guard.try_for_each_page(|page_id, data| {
                if (self.cancelled)() {
                    failed = Some(ZyronError::Internal("Query cancelled".into()));
                    return false;
                }
                zyron_storage::try_for_each_tuple_in_page(page_id, data, &mut |_tid, view| {
                    let xmin = view.header.xmin;
                    let slot = match last {
                        Some((held, slot)) if held == xmin => slot,
                        _ => {
                            let found = by_txn.get(&xmin).copied();
                            last = Some((xmin, found));
                            found
                        }
                    };
                    if let Some(slot) = slot {
                        hashers[slot].row(view.header.schema_epoch, view.data);
                    }
                    if let Some(set) = genesis.as_mut()
                        && xmin < fence
                        && self.snapshot.is_visible(xmin, view.header.xmax)
                        && let Err(e) = set.row(view.header.schema_epoch, view.data)
                    {
                        failed = Some(e);
                        return false;
                    }
                    true
                })
            });
            failed.is_none()
        })?;
        if let Some(e) = failed {
            return Err(e);
        }
        let mut out = RowHashes {
            by_txn: HashMap::with_capacity(by_txn.len()),
            genesis: None,
        };
        let mut finished: Vec<Option<(ChainHash, u64)>> = hashers
            .into_iter()
            .map(|hasher| {
                let rows = hasher.rows();
                Some((hasher.finish(), rows))
            })
            .collect();
        for (txn_id, slot) in by_txn {
            if let Some(hashed) = finished[slot].take() {
                out.by_txn.insert(txn_id, hashed);
            }
        }
        if let Some(set) = genesis {
            let rows = set.rows();
            out.genesis = Some((set.finish()?, rows));
        }
        Ok(out)
    }
}

// ---------------------------------------------------------------------------
// Anchoring
// ---------------------------------------------------------------------------

/// The head of one table's chain as an anchor names it, None for a table
/// with nothing to anchor
fn anchor_of(
    registry: &ChainRegistry,
    entry: &zyron_catalog::TableEntry,
    taken_at: i64,
) -> Result<Option<Anchor>> {
    if !entry.lifecycle.verified {
        return Ok(None);
    }
    let chain = registry.chain(entry.id.0)?;
    let head = chain.head();
    if head.commits == 0 {
        return Ok(None);
    }
    // The entries are on the device before the head is anchored, so an
    // anchor never names a head the chain cannot show
    chain.sync()?;
    Ok(Some(Anchor {
        table_id: entry.id.0,
        table_name: entry.name.clone(),
        sequence: head.commits - 1,
        commit_version: head.head_version,
        head_hash: head.head_hash,
        taken_at,
    }))
}

/// Writes the audit event that records an anchor
async fn audit_anchor(server: &ServerState, anchor: &Anchor) -> Result<()> {
    append_audit(
        server,
        zyron_catalog::schema::ComplianceLogEntry {
            event_id: 0,
            event_type: zyron_lifecycle::compliance::event::CHAIN_ANCHORED,
            subject: anchor.table_name.clone(),
            table_id: anchor.table_id,
            ts: anchor.taken_at,
            detail: anchor.describe(),
            record_version: zyron_lifecycle::format::AUDIT_RECORD_VERSION_BYTE,
        },
    )
    .await?;
    tracing::info!(
        target: "zyron::audit",
        event = "ChainAnchored",
        table = %anchor.table_name,
        version = anchor.commit_version,
        head = %verify::hex(&anchor.head_hash),
    );
    Ok(())
}

/// Records the head of one table's chain.
///
/// Written into the audit log as well as into the anchor store, so what a
/// verification is held against and what an auditor reads are one fact
pub async fn anchor_table(server: &ServerState, table_id: u32) -> Result<Option<Anchor>> {
    let Some(registry) = server.chain_registry.as_ref() else {
        return Ok(None);
    };
    let entry = server.catalog.get_table_by_id(TableId(table_id))?;
    let Some(anchor) = anchor_of(registry, &entry, now_micros())? else {
        return Ok(None);
    };
    registry.anchors().anchor(anchor.clone())?;
    audit_anchor(server, &anchor).await?;
    Ok(Some(anchor))
}

/// The principal whose key signs an exported anchor, which is the one the
/// release artifacts are signed under
pub const ANCHOR_SIGNING_PRINCIPAL: &str = "release";

/// Signs an anchor so an operator can hold it outside the cluster.
///
/// The signature is what makes an exported anchor stronger than one held
/// inside: whoever rewrites a chain can produce a head, and cannot produce
/// this signature over the head the chain used to have
pub fn export_anchor(anchor: &Anchor) -> Result<ExportedAnchor> {
    let store = zyron_auth::signature::principal_keys();
    let scheme = match store.current(ANCHOR_SIGNING_PRINCIPAL) {
        Some(key) => key.scheme_name,
        None => {
            let issued = store.issue(
                ANCHOR_SIGNING_PRINCIPAL,
                "Ed25519",
                now_micros() as u64 / 1_000_000,
            )?;
            issued.scheme_name
        }
    };
    let signature = store.sign(ANCHOR_SIGNING_PRINCIPAL, &anchor.signed_bytes())?;
    Ok(ExportedAnchor {
        anchor: anchor.clone(),
        scheme,
        signature,
    })
}

/// Whether an exported anchor was signed by this cluster and names the head
/// the chain shows at its position.
///
/// Both halves have to hold. An artifact that matches the chain and carries
/// no valid signature is one whoever rewrote the chain wrote themselves
pub fn check_exported_anchor(
    registry: &ChainRegistry,
    exported: &ExportedAnchor,
) -> Result<(bool, bool)> {
    let store = zyron_auth::signature::principal_keys();
    let signed = match store.current(ANCHOR_SIGNING_PRINCIPAL) {
        Some(key) => {
            let material = key.verifying_material()?;
            zyron_auth::signature::verify_with(
                &material,
                &exported.anchor.signed_bytes(),
                &exported.signature,
            )
            .unwrap_or(false)
        }
        None => false,
    };
    let chain = registry.chain(exported.anchor.table_id)?;
    let agrees = chain
        .entry_hash_at(exported.anchor.sequence)?
        .is_some_and(|found| exported.agrees_with(exported.anchor.sequence, &found));
    Ok((signed, agrees))
}

/// Anchors every verified table whose head has moved since its last anchor.
///
/// The anchors of one pass reach the store in one write, then each is
/// recorded in the audit log. Answers with what was anchored, for the
/// worker to report
pub async fn anchor_due_tables(server: &ServerState) -> Vec<Anchor> {
    let Some(registry) = server.chain_registry.as_ref() else {
        return Vec::new();
    };
    let taken_at = now_micros();
    let mut due = Vec::new();
    for entry in server.catalog.list_all_tables() {
        if !entry.lifecycle.verified {
            continue;
        }
        let Ok(chain) = registry.chain(entry.id.0) else {
            continue;
        };
        let head = chain.head();
        if head.commits == 0 {
            continue;
        }
        if registry
            .anchors()
            .for_table(entry.id.0)
            .is_some_and(|held| held.sequence + 1 >= head.commits)
        {
            continue;
        }
        match anchor_of(registry, &entry, taken_at) {
            Ok(Some(anchor)) => due.push(anchor),
            Ok(None) => {}
            Err(e) => tracing::warn!(
                target: "zyron::verify",
                table = %entry.name,
                error = %e,
                "the head of a verified table could not be read for anchoring"
            ),
        }
    }
    if due.is_empty() {
        return due;
    }
    if let Err(e) = registry.anchors().anchor_all(due.iter().cloned()) {
        tracing::warn!(
            target: "zyron::verify",
            tables = due.len(),
            error = %e,
            "the anchors of this pass could not be written down, none was taken"
        );
        return Vec::new();
    }
    let mut taken = Vec::with_capacity(due.len());
    for anchor in due {
        match audit_anchor(server, &anchor).await {
            Ok(()) => taken.push(anchor),
            Err(e) => tracing::warn!(
                target: "zyron::verify",
                table = %anchor.table_name,
                error = %e,
                "an anchor is held and its audit event could not be written"
            ),
        }
    }
    taken
}

/// Tables whose head has gone unanchored for longer than twice the
/// interval, which is what `chain_not_anchored` fires on
pub fn unanchored_tables(server: &ServerState, interval_secs: u64) -> Vec<(u32, String)> {
    let Some(registry) = server.chain_registry.as_ref() else {
        return Vec::new();
    };
    let now = now_micros();
    let mut overdue = Vec::new();
    for entry in server.catalog.list_all_tables() {
        if !entry.lifecycle.verified {
            continue;
        }
        let Ok(chain) = registry.chain(entry.id.0) else {
            continue;
        };
        let head = chain.head();
        let held = registry.anchors().for_table(entry.id.0);
        if zyron_lifecycle::verify::anchor::anchor_overdue(
            head.commits,
            head.head_ts,
            held.as_ref(),
            interval_secs,
            now,
        ) {
            overdue.push((entry.id.0, entry.name.clone()));
        }
    }
    overdue
}

// ---------------------------------------------------------------------------
// The genesis entry
// ---------------------------------------------------------------------------

/// Covers the rows a table already holds with a genesis entry.
///
/// Written when a populated table becomes verifiable, after the registry
/// has fenced the table, so the chain covers the table from the moment it
/// became verifiable rather than from its next commit. The rows before the
/// fence are covered as one set: the entry states the hash of the set and
/// its count and names the fence, which is what a verification reads the
/// set back by.
///
/// Nothing is waited for. A transaction from before the fence that wrote to
/// the table is refused at its commit, so what the snapshot sees stamped
/// below the fence is exactly the set, and the statement holds up no
/// commit on a member of a group. `agreed` is the log entry the statement
/// runs under on such a member, whose index and instant every member
/// writes, so every member's genesis is the same entry
pub async fn write_genesis(
    server: &Arc<ServerState>,
    entry: &zyron_catalog::TableEntry,
    agreed: Option<(u64, i64)>,
) -> Result<Option<CommitHash>> {
    let Some(registry) = server.chain_registry.as_ref() else {
        return Ok(None);
    };
    let Some(chained) = registry.chained(entry.id.0) else {
        return Err(ZyronError::Internal(format!(
            "table '{}' is not fenced for chaining, so its genesis set has no edge",
            entry.name
        )));
    };
    let fence = chained.fence;

    let heap = crate::connection::table_heap(server, entry).await?;
    let mut reader = server
        .txn_manager
        .begin(zyron_storage::IsolationLevel::ReadCommitted)?;
    let snapshot = reader.snapshot.clone();
    let spill = verify::spill_dir(registry.data_dir());
    let hashed = tokio::task::spawn_blocking(move || -> Result<(ChainHash, u64)> {
        let mut set = SetHasher::new(spill);
        let mut failed: Option<ZyronError> = None;
        for_each_window(&heap, &|| false, |guard| {
            guard.try_for_each_page(|page_id, data| {
                zyron_storage::try_for_each_tuple_in_page(page_id, data, &mut |_tid, view| {
                    let xmin = view.header.xmin;
                    if xmin < fence && snapshot.is_visible(xmin, view.header.xmax) {
                        if let Err(e) = set.row(view.header.schema_epoch, view.data) {
                            failed = Some(e);
                            return false;
                        }
                    }
                    true
                })
            });
            failed.is_none()
        })?;
        if let Some(e) = failed {
            return Err(e);
        }
        let rows = set.rows();
        Ok((set.finish()?, rows))
    })
    .await
    .map_err(|e| {
        ZyronError::Internal(format!(
            "the genesis set of table '{}' was not hashed, {e}",
            entry.name
        ))
    })?;
    server.txn_manager.commit_read_only(&mut reader)?;
    let (rows_hash, rows) = hashed?;
    // A table with no rows has nothing to cover as a set, so its chain
    // begins at its first commit rather than at an entry over nothing
    if rows == 0 {
        return Ok(None);
    }
    let (version, at) = match agreed {
        Some((index, timestamp_us)) => (index, timestamp_us),
        None => (server.wal.next_lsn().0, now_micros()),
    };
    let chain = registry.chain(entry.id.0)?;
    let linked = chain.append(
        CommitFields {
            txn_id: fence,
            rows_hash,
            row_count: rows,
            commit_ts: at,
            algorithm_id: chained.algorithm_id,
            genesis: true,
        },
        || version,
    )?;
    chain.publish(fence)?;
    chain.sync()?;
    registry.set_genesis_at(entry.id.0, linked.sequence + 1);
    Ok(Some(linked))
}

// ---------------------------------------------------------------------------
// Running a verification
// ---------------------------------------------------------------------------

/// Charges a verification to the background class for as long as it runs.
///
/// A verification reads pages and hashes rows, which is work the node has
/// to be able to shed against the statements it is serving. Background is
/// the class with no objective of its own, so a verification runs on
/// whatever is left and a writer keeps its own throughput
struct BackgroundWork {
    started: std::time::Instant,
    estimated_seconds: f64,
}

impl BackgroundWork {
    fn start() -> Self {
        let controller = zyron_pressure::pressure_control::PressureController::global();
        // Priced as work with no objective rather than estimated, because
        // what a walk costs is the chain's length and the table's size, and
        // the class it lands in is the same either way
        let estimated_seconds = 1.0;
        controller.admit(estimated_seconds, true, None);
        Self {
            started: std::time::Instant::now(),
            estimated_seconds,
        }
    }
}

impl Drop for BackgroundWork {
    fn drop(&mut self) {
        zyron_pressure::pressure_control::PressureController::global().complete(
            zyron_pressure::WorkloadClass::Background,
            self.estimated_seconds,
            self.started.elapsed().as_secs_f64(),
            None,
        );
    }
}

/// What a verification was asked to do.
#[derive(Debug, Clone, Copy)]
pub struct VerifyRequest {
    pub table_id: u32,
    pub from_version: Option<u64>,
    pub to_version: Option<u64>,
    pub mode: RowMode,
    pub sample: u64,
}

/// Walks one table's chain and records the run.
///
/// The walk takes no lock on the table and reads its pages the way any
/// other reader does, so a writer keeps writing while it runs. It is
/// cancellable at every page, which is what lets a full verification of a
/// large table be stopped
pub async fn run_verify(
    server: &ServerState,
    request: VerifyRequest,
    actor: u32,
    actor_name: &str,
    cancelled: Arc<dyn Fn() -> bool + Send + Sync>,
) -> Result<VerifyOutcome> {
    let Some(registry) = server.chain_registry.as_ref() else {
        return Err(ZyronError::Internal(
            "this node holds no commit chains, so there is nothing to verify".to_string(),
        ));
    };
    let entry = server.catalog.get_table_by_id(TableId(request.table_id))?;
    let chain = registry.chain(request.table_id)?;
    let head = chain.head();

    // A version range names commits, and a chain is walked by position, so
    // the range is resolved to the positions it covers before the walk
    let (from, to) = resolve_range(&chain, request.from_version, request.to_version)?;
    let anchors: Vec<Anchor> = registry
        .anchors()
        .for_table(request.table_id)
        .into_iter()
        .collect();

    let heap = crate::connection::table_heap(server, &entry).await?;
    // The read runs under a transaction of its own, so what the table holds
    // now is one snapshot for the whole pass and the rows it reads are not
    // reclaimed under it
    let mut reader = server
        .txn_manager
        .begin(zyron_storage::IsolationLevel::ReadCommitted)?;
    let rows = HeapCommitRows::new(
        heap,
        reader.snapshot.clone(),
        verify::spill_dir(registry.data_dir()),
        Arc::clone(&cancelled),
    );
    let started = std::time::Instant::now();
    let started_at = now_micros();
    let sample = if request.sample == 0 {
        DEFAULT_SAMPLE
    } else {
        request.sample
    };

    // The walk reads pages and hashes rows, which is work a runtime worker
    // must not be held by. Run on a blocking thread at the background
    // class, so a verification never takes throughput from the statements
    // the node is serving
    let table_id = request.table_id;
    let mode = request.mode;
    let walked = tokio::task::spawn_blocking(move || {
        let _class = BackgroundWork::start();
        verify::walk(&chain, &rows, from, to, mode, sample, &anchors, &*cancelled)
    })
    .await
    .map_err(|e| {
        ZyronError::Internal(format!(
            "the verification of table {table_id} did not finish, {e}"
        ))
    });
    server.txn_manager.commit_read_only(&mut reader)?;
    let outcome = walked??;

    let run = VerifyRun {
        table_id: request.table_id,
        table_name: entry.name.clone(),
        actor,
        actor_name: actor_name.to_string(),
        from_version: request.from_version.unwrap_or(0),
        to_version: request.to_version.unwrap_or(head.head_version),
        mode: outcome.mode,
        commits_checked: outcome.commits_checked,
        rows_checked: outcome.rows_checked,
        anchors_checked: outcome.anchors_checked,
        intact: outcome.intact,
        finding: outcome
            .failure
            .as_ref()
            .map(|failure| failure.to_string())
            .unwrap_or_default(),
        started_at,
        duration_micros: started.elapsed().as_micros() as u64,
    };
    registry.runs().record(run)?;

    tracing::info!(
        target: "zyron::audit",
        event = "VerifyRun",
        table = %entry.name,
        range = %format!("{from}..={to}"),
        mode = outcome.mode.label(),
        outcome = if outcome.intact { "intact" } else { "not intact" },
        chain_micros = outcome.chain_micros,
        rows_micros = outcome.rows_micros,
        total_micros = started.elapsed().as_micros() as u64,
        by = actor,
    );
    if let Some(failure) = outcome.failure.as_ref() {
        tracing::warn!(
            target: "zyron::audit",
            event = "VerifyFailed",
            table = %entry.name,
            commit_version = failure.commit_version(),
            reason = failure.kind(),
            detail = %failure,
        );
    }
    Ok(outcome)
}

// ---------------------------------------------------------------------------
// VERIFY TABLE
// ---------------------------------------------------------------------------

/// Runs `VERIFY TABLE`.
///
/// SELECT on the table is what lets a reader ask at all, because a
/// verification states what the table holds. Reading every row back costs
/// the whole table, so `rows => 'all'` takes MANAGE_VERIFICATION on top.
/// The statement publishes itself as the connection's cancellable
/// statement, so a cancel request or CANCEL BACKEND stops the walk at the
/// next page
pub async fn handle_verify_table(
    stmt: &zyron_parser::ast::VerifyTableStatement,
    server: &Arc<ServerState>,
    session: &mut Option<crate::session::Session>,
) -> std::result::Result<crate::ddl_dispatch::DdlResult, crate::messages::ProtocolError> {
    use crate::ddl_dispatch::DdlResult;
    use crate::messages::ProtocolError;

    let (schema_id, name) =
        crate::ddl_dispatch::resolve_qualified_name(&stmt.table, server, session)?;
    let entry = server
        .catalog
        .get_table(schema_id, &name)
        .map_err(ProtocolError::Database)?;

    if !entry.lifecycle.verified {
        return Err(ProtocolError::Database(ZyronError::Internal(format!(
            "table '{}' is not verified, so it carries no commit chain to walk. \
             ALTER TABLE {} SET (immutable = true, verified = true) starts one",
            stmt.table, stmt.table
        ))));
    }

    crate::ddl_dispatch::check_ddl_privilege(
        server,
        session,
        zyron_auth::PrivilegeType::Select,
        zyron_auth::ObjectType::Table,
        entry.id.0,
    )?;

    let mode = match stmt.rows {
        zyron_parser::ast::VerifyRowMode::All => RowMode::All,
        zyron_parser::ast::VerifyRowMode::Sampled => RowMode::Sampled,
    };
    if mode == RowMode::All {
        crate::ddl_dispatch::check_ddl_privilege(
            server,
            session,
            zyron_auth::PrivilegeType::ManageVerification,
            zyron_auth::ObjectType::Table,
            entry.id.0,
        )?;
    }

    let from_version = literal_version(stmt.from_version.as_ref(), "FROM VERSION")?;
    let to_version = literal_version(stmt.to_version.as_ref(), "TO VERSION")?;
    let sample = literal_version(stmt.sample.as_ref(), "sample")?.unwrap_or_else(|| {
        crate::lifecycle_dispatch::cdc_setting(server, "verify.default_sample").max(DEFAULT_SAMPLE)
    });

    let actor = crate::ddl_dispatch::actor_role_id(session);
    let actor_name = session
        .as_ref()
        .map(|s| s.user.clone())
        .unwrap_or_else(|| "system".to_string());

    // The statement's cancel flag lives on a context of its own, published
    // under the connection's backend id and secret the way a query's is
    let cancel_txn = server
        .txn_manager
        .begin(zyron_storage::IsolationLevel::ReadCommitted)
        .map_err(ProtocolError::Database)?;
    let ctx = Arc::new(server.statement_context(cancel_txn.txn_id(), cancel_txn.snapshot.clone()));
    if let Some(s) = session.as_ref()
        && s.process_id != 0
    {
        server
            .cancel_registry
            .upsert_sync(s.process_id, (s.secret_key, Arc::downgrade(&ctx)));
    }
    let flag = Arc::clone(&ctx);
    let cancelled: Arc<dyn Fn() -> bool + Send + Sync> = Arc::new(move || flag.is_cancelled());

    let outcome = run_verify(
        server,
        VerifyRequest {
            table_id: entry.id.0,
            from_version,
            to_version,
            mode,
            sample,
        },
        actor,
        &actor_name,
        cancelled,
    )
    .await;
    let mut cancel_txn = cancel_txn;
    server
        .txn_manager
        .commit_read_only(&mut cancel_txn)
        .map_err(ProtocolError::Database)?;
    let outcome = outcome.map_err(ProtocolError::Database)?;

    let columns = vec![
        ("table".to_string(), crate::types::PG_TEXT_OID),
        ("commits_checked".to_string(), crate::types::PG_INT8_OID),
        ("rows_checked".to_string(), crate::types::PG_INT8_OID),
        ("anchors_checked".to_string(), crate::types::PG_INT8_OID),
        ("mode".to_string(), crate::types::PG_TEXT_OID),
        ("intact".to_string(), crate::types::PG_TEXT_OID),
        ("detail".to_string(), crate::types::PG_TEXT_OID),
    ];
    let rows = vec![vec![
        entry.name.clone(),
        outcome.commits_checked.to_string(),
        outcome.rows_checked.to_string(),
        outcome.anchors_checked.to_string(),
        outcome.mode.label().to_string(),
        outcome.intact.to_string(),
        outcome.summary(),
    ]];
    Ok(DdlResult::Rows {
        tag: "VERIFY TABLE".to_string(),
        columns,
        rows,
    })
}

/// Reads a whole number a clause carries, refusing anything else rather
/// than reading it as a default
fn literal_version(
    expr: Option<&zyron_parser::Expr>,
    clause: &str,
) -> std::result::Result<Option<u64>, crate::messages::ProtocolError> {
    use crate::messages::ProtocolError;
    let Some(expr) = expr else {
        return Ok(None);
    };
    match expr {
        zyron_parser::Expr::Literal(zyron_parser::ast::LiteralValue::Integer(n)) if *n >= 0 => {
            Ok(Some(*n as u64))
        }
        other => Err(ProtocolError::Database(ZyronError::Internal(format!(
            "{clause} takes a whole number, found {other:?}"
        )))),
    }
}

/// The chain positions a version range covers.
///
/// A chain is walked by position and a range names versions, so the ends
/// are resolved by binary search over the chain, whose versions rise along
/// it. A range naming versions the chain does not reach covers nothing
/// rather than covering everything
fn resolve_range(
    chain: &verify::CommitChain,
    from_version: Option<u64>,
    to_version: Option<u64>,
) -> Result<(u64, u64)> {
    let head = chain.head();
    if head.commits == 0 {
        return Ok((0, 0));
    }
    let last = head.commits - 1;
    let from = match from_version {
        None => 0,
        Some(version) => chain.first_at_or_above(version)?,
    };
    let to = match to_version {
        None => last,
        Some(version) => chain.last_at_or_below(version)?.unwrap_or(0),
    };
    Ok((from, to))
}
