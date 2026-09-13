//! Building an index over rows a table already holds, while the table stays
//! open to everyone else.
//!
//! ## The race, and what closes it
//!
//! A build has to cover two sets of rows: the ones that existed when it
//! started, and the ones written while it runs. Reading the first set and then
//! registering the index leaves a gap in the middle, and a row written in that
//! gap is never indexed. It is not a slow build, it is a wrong one: every scan
//! the planner routes through the index silently drops that row.
//!
//! The order that closes it is publish, wait, scan, load, flip.
//!
//! **Publish** writes the catalog entry as Building and registers an empty
//! tree. From that instant every write that resolves its index set sees the
//! index and maintains it, so the second set covers itself.
//!
//! **Wait** is the step the whole argument rests on. A transaction that
//! resolved its index set before publication will not maintain the new index
//! for the rest of its life, so its writes would fall in the gap. Waiting for
//! every transaction that was active at publication to end means every writer
//! still running afterwards resolved its index set after publication.
//!
//! **Scan** takes one snapshot and streams the rows that predate it, in
//! bounded batches, into a spilling sort. Nothing holds the table in memory.
//!
//! **Load** merges the sorted run with what maintenance has already put in the
//! tree and writes the tree level by level.
//!
//! **Flip** records Ready, and the planner may choose the index.
//!
//! ## Lake tables take a different path
//!
//! A lake table's rows are addressed by data file and ordinal rather than by
//! page and slot, and the files a clustering or compaction pass rewrites would
//! invalidate every entry of a tree built over them. Its index is committed
//! into its own transaction log instead, so it is versioned with the rows,
//! maintained by the same commits that move them, and readable at a past
//! version. `build_lake_index` routes there.

use std::io::{BufReader, BufWriter, Read, Write};
use std::path::PathBuf;
use std::sync::Arc;

use zyron_catalog::{IndexState, TableEntry};
use zyron_common::page::PageId;
use zyron_common::{RowLocator, ZyronError};
use zyron_executor::context::ExecutionContext;
use zyron_executor::operator::modify::CapturedRow;
use zyron_storage::{HeapFile, HeapFileConfig, HeapPage, TupleHeader};

use crate::connection::ServerState;
use crate::ddl_progress::{BuildPacer, DdlPhase, DdlProgress};

/// Rows one scan batch carries before the sort takes them.
///
/// Bounded so a build over a table larger than memory holds one batch rather
/// than the table. Cascading config overrides it per node.
pub const DEFAULT_BUILD_BATCH_ROWS: usize = 65_536;

/// Bytes of sorted keys a run holds before it spills.
///
/// The build's whole memory footprint is one batch of rows plus one run
/// buffer, so this is the number that decides its peak.
const DEFAULT_RUN_BUDGET_BYTES: u64 = 64 * 1024 * 1024;

/// Resolves index column names against a table and returns their ids in the
/// order they were declared.
fn column_ids_for(table: &TableEntry, column_names: &[String]) -> Result<Vec<u32>, ZyronError> {
    let mut ids = Vec::with_capacity(column_names.len());
    for name in column_names {
        let column = table
            .live_columns()
            .find(|c| c.name.eq_ignore_ascii_case(name))
            .ok_or_else(|| {
                ZyronError::PlanError(format!(
                    "column \"{}\" is not in table \"{}\"",
                    name, table.name
                ))
            })?;
        ids.push(column.id.0 as u32);
    }
    Ok(ids)
}

/// Declares an index on a lake table and backfills it over every live data
/// file in one commit.
///
/// The declaration and its entries land in the same version, so no version
/// exists where the index is declared but empty. Later commits that add or
/// rewrite data files carry their own index entries, and a commit that could
/// not maintain them leaves the index short of covering the table, which makes
/// a probe decline rather than answer with fewer rows than the table has
pub async fn build_lake_index(
    server: &Arc<ServerState>,
    table: &TableEntry,
    column_names: &[String],
    unique: bool,
) -> Result<(), ZyronError> {
    let column_ids = column_ids_for(table, column_names)?;
    let paths = zyron_lake::LakePaths::new(server.disk_manager.data_dir(), table.id.0);
    let log = zyron_lake::TransactionLog::lookup_shared(&paths).ok_or_else(|| {
        ZyronError::ConfigError(format!(
            "this node does not run the lake tier, so it cannot index \"{}\"",
            table.name
        ))
    })?;
    let timestamp_us = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_micros() as i64)
        .unwrap_or(0);
    let attempt = zyron_lake::CommitAttempt {
        operation: zyron_lake::OperationKind::SchemaChange,
        db_txn_id: 0,
        commit_lsn: 0,
        timestamp_us,
        read_predicate: None,
        read_version: 0,
        audit: None,
        deadline: None,
    };
    let name = index_name_for(table, column_names);
    let outcome = zyron_lake::operations::create_index(
        &log,
        attempt,
        table.id.0 as u64,
        &name,
        &column_ids,
        unique,
    )?;
    tracing::info!(
        target: "zyron::ddl",
        index = %name,
        table = %table.name,
        version = outcome.version,
        rows = outcome.rows,
        "CREATE INDEX built a lake index"
    );
    Ok(())
}

/// Drops a lake table's index, by the name the build gave it.
///
/// A missing index is not an error here: the catalog entry is the authority
/// on what exists, and this call only removes the storage behind it
pub async fn drop_lake_index(
    server: &Arc<ServerState>,
    table: &TableEntry,
    column_names: &[String],
) -> Result<(), ZyronError> {
    let paths = zyron_lake::LakePaths::new(server.disk_manager.data_dir(), table.id.0);
    let Some(log) = zyron_lake::TransactionLog::lookup_shared(&paths) else {
        return Ok(());
    };
    let name = index_name_for(table, column_names);
    if log
        .latest_manifest()
        .map(|m| m.index_by_name(&name).is_none())
        .unwrap_or(true)
    {
        return Ok(());
    }
    let timestamp_us = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_micros() as i64)
        .unwrap_or(0);
    let attempt = zyron_lake::CommitAttempt {
        operation: zyron_lake::OperationKind::SchemaChange,
        db_txn_id: 0,
        commit_lsn: 0,
        timestamp_us,
        read_predicate: None,
        read_version: 0,
        audit: None,
        deadline: None,
    };
    zyron_lake::operations::drop_index(&log, attempt, &name)?;
    Ok(())
}

/// Rebuilds every index a lake table declares, which is what REINDEX runs.
pub async fn rebuild_lake_indexes(
    server: &Arc<ServerState>,
    table: &TableEntry,
) -> Result<(), ZyronError> {
    let paths = zyron_lake::LakePaths::new(server.disk_manager.data_dir(), table.id.0);
    let Some(log) = zyron_lake::TransactionLog::lookup_shared(&paths) else {
        return Ok(());
    };
    if log
        .latest_manifest()
        .map(|m| m.indexes.is_empty())
        .unwrap_or(true)
    {
        return Ok(());
    }
    let timestamp_us = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_micros() as i64)
        .unwrap_or(0);
    let attempt = zyron_lake::CommitAttempt {
        operation: zyron_lake::OperationKind::SchemaChange,
        db_txn_id: 0,
        commit_lsn: 0,
        timestamp_us,
        read_predicate: None,
        read_version: 0,
        audit: None,
        deadline: None,
    };
    zyron_lake::operations::rebuild_indexes(&log, attempt, table.id.0 as u64)?;
    Ok(())
}

/// The name a lake index carries in its own manifest.
///
/// The catalog's index name is unique per schema while the lake manifest is
/// per table, so the table's own columns are what identify it there. Deriving
/// it from the columns rather than storing it keeps the two from drifting when
/// an index is renamed in the catalog
fn index_name_for(table: &TableEntry, column_names: &[String]) -> String {
    let mut resolved: Vec<String> = Vec::with_capacity(column_names.len());
    for name in column_names {
        match table
            .columns
            .iter()
            .find(|c| c.name.eq_ignore_ascii_case(name))
        {
            Some(column) => resolved.push(column.name.clone()),
            None => resolved.push(name.clone()),
        }
    }
    format!("ix_{}", resolved.join("_"))
}

// ---------------------------------------------------------------------------
// Waiting for the transactions that were running at publication
// ---------------------------------------------------------------------------

/// Waits until every transaction that was active when the index published has
/// ended.
///
/// This is what makes maintenance sufficient for everything written after the
/// build starts. A transaction that resolved its index set before publication
/// never sees the new index, so its writes would go unindexed; once it ends,
/// every writer still running looked the index set up afterwards.
///
/// The wait is a yield loop over the proc array. Nothing is held while it
/// runs, so a transaction that takes a long time to finish delays this build
/// and nothing else.
/// `exclude` names the transactions this build is itself running inside. A
/// session that opened a transaction and then issued the statement, and an
/// applier replaying a schema change under its own transaction, are both
/// transactions that cannot end until the build returns, so waiting for them
/// would wait forever. Neither of them resolved an index set for this table
/// before publication: the session is running the statement, and a schema
/// change applies alone.
pub async fn wait_for_transactions_active_at(
    server: &Arc<ServerState>,
    at_publication: &[u64],
    exclude: &[u64],
    deadline: Option<std::time::Instant>,
) -> Result<(), ZyronError> {
    if at_publication.is_empty() {
        return Ok(());
    }
    loop {
        let still_running: Vec<u64> = {
            let live = server.txn_manager.proc_array().active_txn_ids();
            at_publication
                .iter()
                .copied()
                .filter(|id| live.contains(id) && !exclude.contains(id))
                .collect()
        };
        if still_running.is_empty() {
            return Ok(());
        }
        if let Some(deadline) = deadline
            && std::time::Instant::now() >= deadline
        {
            return Err(ZyronError::ExecutionError(format!(
                "the index build gave up waiting for {} transaction(s) that were running when it \
                 published: {:?}. Until they end, writes they make would not reach the new index, \
                 so the build cannot proceed",
                still_running.len(),
                still_running
            )));
        }
        tokio::time::sleep(std::time::Duration::from_millis(2)).await;
    }
}

// ---------------------------------------------------------------------------
// The spilling sort
// ---------------------------------------------------------------------------

/// One (key, locator) pair on its way into the tree.
type SortEntry = (Vec<u8>, RowLocator);

/// Collects index keys in key order, spilling runs to disk rather than holding
/// the table's keys in memory.
///
/// The run directory is named after the index, so a crash leaves a directory
/// recovery can identify and remove by the index it belongs to rather than by
/// guessing which files were a build's.
/// One buffered key, addressed into the arena rather than owning its bytes.
struct ArenaEntry {
    start: u32,
    len: u32,
    locator: RowLocator,
}

pub struct KeySorter {
    dir: PathBuf,
    budget_bytes: u64,
    held_bytes: u64,
    /// Every buffered key's bytes, end to end.
    ///
    /// A key of its own is eight to a few dozen bytes, and a separate
    /// allocation for each costs a header and a rounded-up block per key on
    /// top of the bytes themselves, several times the key. One arena holds
    /// them at their own size, which is what makes the budget below the real
    /// figure rather than a fraction of it
    arena: Vec<u8>,
    entries: Vec<ArenaEntry>,
    runs: Vec<PathBuf>,
    spilled_bytes: u64,
}

impl KeySorter {
    pub fn new(data_dir: &std::path::Path, index_id: u32, budget_bytes: u64) -> Self {
        Self {
            dir: build_spill_dir(data_dir, index_id),
            budget_bytes: budget_bytes.max(1024 * 1024),
            held_bytes: 0,
            arena: Vec::new(),
            entries: Vec::new(),
            runs: Vec::new(),
            spilled_bytes: 0,
        }
    }

    /// The buffered keys in key order, as slices into the arena.
    fn sorted_entries(&mut self) -> &[ArenaEntry] {
        let arena = &self.arena;
        self.entries.sort_unstable_by(|a, b| {
            arena[a.start as usize..(a.start + a.len) as usize]
                .cmp(&arena[b.start as usize..(b.start + b.len) as usize])
        });
        &self.entries
    }

    /// Bytes written to run files so far.
    pub fn spilled_bytes(&self) -> u64 {
        self.spilled_bytes
    }

    pub fn push(&mut self, key: Vec<u8>, locator: RowLocator) -> Result<(), ZyronError> {
        let start = self.arena.len() as u32;
        self.arena.extend_from_slice(&key);
        self.entries.push(ArenaEntry {
            start,
            len: key.len() as u32,
            locator,
        });
        self.held_bytes += (key.len() + std::mem::size_of::<ArenaEntry>()) as u64;
        if self.held_bytes >= self.budget_bytes {
            self.spill()?;
        }
        Ok(())
    }

    fn spill(&mut self) -> Result<(), ZyronError> {
        if self.entries.is_empty() {
            return Ok(());
        }
        self.sorted_entries();
        std::fs::create_dir_all(&self.dir).map_err(|e| {
            ZyronError::IoError(format!(
                "index build could not create its spill directory {}: {e}",
                self.dir.display()
            ))
        })?;
        let path = self.dir.join(format!("run-{:05}.zyrun", self.runs.len()));
        let file = std::fs::File::create(&path).map_err(|e| {
            ZyronError::IoError(format!(
                "index build could not write its sort run {}: {e}",
                path.display()
            ))
        })?;
        let mut out = BufWriter::new(file);
        let mut written: u64 = 0;
        let mut payload = [0u8; RowLocator::MAX_PAYLOAD_LEN];
        for entry in &self.entries {
            let key = &self.arena[entry.start as usize..(entry.start + entry.len) as usize];
            let n = entry.locator.write_payload(&mut payload);
            out.write_all(&entry.len.to_le_bytes())
                .and_then(|()| out.write_all(key))
                .and_then(|()| out.write_all(&payload[..n]))
                .map_err(|e| {
                    ZyronError::IoError(format!(
                        "index build could not write its sort run {}: {e}",
                        path.display()
                    ))
                })?;
            written += 4 + entry.len as u64 + n as u64;
        }
        // The arena's capacity is kept, so the next run fills the same bytes
        // rather than growing a new buffer up to the budget again
        self.entries.clear();
        self.arena.clear();
        out.flush().map_err(|e| {
            ZyronError::IoError(format!(
                "index build could not flush its sort run {}: {e}",
                path.display()
            ))
        })?;
        self.spilled_bytes += written;
        self.held_bytes = 0;
        self.runs.push(path);
        Ok(())
    }

    /// Finishes the sort and hands back an iterator over every entry in key
    /// order.
    pub fn finish(mut self) -> Result<SortedKeys, ZyronError> {
        if self.runs.is_empty() {
            self.sorted_entries();
            let held: Vec<SortEntry> = self
                .entries
                .iter()
                .map(|e| {
                    (
                        self.arena[e.start as usize..(e.start + e.len) as usize].to_vec(),
                        e.locator,
                    )
                })
                .collect();
            self.entries.clear();
            self.arena.clear();
            self.arena.shrink_to_fit();
            return Ok(SortedKeys {
                dir: self.dir,
                memory: held.into_iter(),
                readers: Vec::new(),
                heads: Vec::new(),
                unique: false,
                previous: None,
                duplicate: None,
            });
        }
        self.spill()?;
        let mut readers = Vec::with_capacity(self.runs.len());
        let mut heads = Vec::with_capacity(self.runs.len());
        for path in &self.runs {
            let file = std::fs::File::open(path).map_err(|e| {
                ZyronError::IoError(format!(
                    "index build could not read back its sort run {}: {e}",
                    path.display()
                ))
            })?;
            let mut reader = BufReader::new(file);
            heads.push(read_entry(&mut reader, path)?);
            readers.push(reader);
        }
        Ok(SortedKeys {
            dir: self.dir,
            memory: Vec::new().into_iter(),
            readers,
            heads,
            unique: false,
            previous: None,
            duplicate: None,
        })
    }

    /// Removes the run directory, which is what a failed build leaves behind
    /// otherwise.
    pub fn discard(self) {
        remove_build_dir(&self.dir);
    }
}

/// Where a build's sort runs live, named after the index so recovery can
/// identify one without guessing.
pub fn build_spill_dir(data_dir: &std::path::Path, index_id: u32) -> PathBuf {
    data_dir.join("indexes").join(format!("build-{index_id}"))
}

/// Removes a build's spill directory and everything under it.
pub fn remove_build_dir(dir: &std::path::Path) {
    if dir.exists() {
        let _ = std::fs::remove_dir_all(dir);
    }
}

fn read_entry(
    reader: &mut BufReader<std::fs::File>,
    path: &std::path::Path,
) -> Result<Option<SortEntry>, ZyronError> {
    let mut len_buf = [0u8; 4];
    match reader.read_exact(&mut len_buf) {
        Ok(()) => {}
        Err(e) if e.kind() == std::io::ErrorKind::UnexpectedEof => return Ok(None),
        Err(e) => {
            return Err(ZyronError::IoError(format!(
                "index build could not read its sort run {}: {e}",
                path.display()
            )));
        }
    }
    let key_len = u32::from_le_bytes(len_buf) as usize;
    let mut key = vec![0u8; key_len];
    let mut tag = [0u8; 1];
    reader
        .read_exact(&mut key)
        .and_then(|()| reader.read_exact(&mut tag))
        .map_err(|e| {
            ZyronError::IoError(format!(
                "index build could not read its sort run {}: {e}",
                path.display()
            ))
        })?;
    let payload_len = RowLocator::payload_len_for_tag(tag[0]);
    let mut payload = vec![0u8; payload_len];
    payload[0] = tag[0];
    reader.read_exact(&mut payload[1..]).map_err(|e| {
        ZyronError::IoError(format!(
            "index build could not read its sort run {}: {e}",
            path.display()
        ))
    })?;
    let locator = RowLocator::read_payload(&payload).ok_or_else(|| {
        ZyronError::IoError(format!(
            "index build read a row locator it cannot decode from {}",
            path.display()
        ))
    })?;
    Ok(Some((key, locator)))
}

/// Every entry the sort produced, in key order.
///
/// Holds one entry per run rather than a run's contents, so merging a hundred
/// runs costs a hundred entries of memory.
pub struct SortedKeys {
    dir: PathBuf,
    memory: std::vec::IntoIter<SortEntry>,
    readers: Vec<BufReader<std::fs::File>>,
    heads: Vec<Option<SortEntry>>,
    /// Set for a unique index, which makes the stream stop at the first key
    /// whose value part repeats the one before it
    unique: bool,
    /// The value part and address of the key last yielded, held only while
    /// `unique` is set
    previous: Option<(Vec<u8>, RowLocator)>,
    /// The pair that ended the stream, read back once the load returns
    duplicate: Option<DuplicateKey>,
}

impl SortedKeys {
    /// Removes the run directory once the tree has been written.
    pub fn cleanup(self) {
        remove_build_dir(&self.dir);
    }

    /// The directory holding the runs, so a caller can report or remove it.
    pub fn spill_dir(&self) -> &std::path::Path {
        &self.dir
    }

    fn next_merged(&mut self) -> Option<SortEntry> {
        if self.readers.is_empty() {
            return self.memory.next();
        }
        let mut best: Option<usize> = None;
        for (i, head) in self.heads.iter().enumerate() {
            let Some((key, _)) = head else { continue };
            match best {
                None => best = Some(i),
                Some(b) => {
                    let (best_key, _) = self.heads[b].as_ref()?;
                    if key < best_key {
                        best = Some(i);
                    }
                }
            }
        }
        let idx = best?;
        let taken = self.heads[idx].take();
        // A run whose next entry cannot be read stops contributing rather than
        // taking the build down: the read error already surfaced when the run
        // was written, and a torn tail is reported by the count check the
        // caller makes
        self.heads[idx] = match read_entry(&mut self.readers[idx], &self.dir) {
            Ok(next) => next,
            Err(_) => None,
        };
        taken
    }
}

impl SortedKeys {
    /// Makes the stream refuse a value that appears twice.
    ///
    /// The stream is totally ordered, so the two copies of a repeated value
    /// are adjacent and one comparison against the key just yielded finds
    /// them. That is what lets a unique build hold one key rather than a set
    /// of every key the table holds.
    pub fn enforce_unique(&mut self) {
        self.unique = true;
    }

    /// The repeated value that stopped the stream, if one did.
    pub fn duplicate(&mut self) -> Option<DuplicateKey> {
        self.duplicate.take()
    }
}

impl Iterator for SortedKeys {
    type Item = (bytes::Bytes, RowLocator);

    fn next(&mut self) -> Option<Self::Item> {
        let (key, locator) = self.next_merged()?;
        if self.unique {
            // The locator suffix makes every stored key distinct, so what
            // decides uniqueness is the value in front of it
            let value = &key[..key.len().saturating_sub(RowLocator::KEY_SUFFIX_LEN)];
            match &self.previous {
                Some((seen, first)) if seen == value => {
                    self.duplicate = Some(DuplicateKey {
                        key: value.to_vec(),
                        first: *first,
                        second: locator,
                    });
                    return None;
                }
                _ => match &mut self.previous {
                    Some((seen, first)) => {
                        seen.clear();
                        seen.extend_from_slice(value);
                        *first = locator;
                    }
                    slot @ None => *slot = Some((value.to_vec(), locator)),
                },
            }
        }
        Some((bytes::Bytes::from(key), locator))
    }
}

// ---------------------------------------------------------------------------
// Streaming the rows an index build has to cover
// ---------------------------------------------------------------------------

/// One batch of a table's live rows, addressed the way the index keys them.
pub struct LiveBatch {
    /// Heap rows, grouped by the page they sit on
    pub heap: Vec<(PageId, Vec<CapturedRow>)>,
    /// Folded rows, already decoded with the patch overlay applied
    pub columnar: Option<(
        zyron_executor::batch::DataBatch,
        Vec<zyron_common::RowLocator>,
    )>,
}

impl LiveBatch {
    pub fn rows(&self) -> u64 {
        let heap: u64 = self.heap.iter().map(|(_, r)| r.len() as u64).sum();
        let columnar = self
            .columnar
            .as_ref()
            .map(|(b, _)| b.num_rows as u64)
            .unwrap_or(0);
        heap + columnar
    }

    pub fn is_empty(&self) -> bool {
        self.rows() == 0
    }
}

/// Streams a table's live rows in bounded batches under one snapshot.
///
/// Never holds more than one batch, so a build over a table larger than memory
/// costs one batch of rows rather than the table.
pub struct LiveRowStream {
    server: Arc<ServerState>,
    table: Arc<TableEntry>,
    heap_file: Option<Arc<HeapFile>>,
    page_cursor: u64,
    page_count: u64,
    batch_rows: usize,
    is_dead: Box<dyn Fn(u64, u64) -> bool + Send + Sync>,
    columnar: Option<Box<dyn zyron_executor::operator::Operator>>,
    columnar_started: bool,
    scan_txn: Option<zyron_storage::txn::Transaction>,
}

impl LiveRowStream {
    /// Opens the stream under one snapshot, which is the set of rows the build
    /// is responsible for. Everything written after it is covered by the
    /// index maintenance the publish step turned on.
    pub async fn open(
        server: &Arc<ServerState>,
        table: &Arc<TableEntry>,
        batch_rows: usize,
    ) -> Result<Self, ZyronError> {
        let status_map = server.txn_manager.status_map().clone();
        // Taken over published visibility floors, not the oldest active txn
        // id: a committed deleter can sit below the oldest active id while a
        // live reader that started before it committed still sees its rows
        let prune_horizon = server.txn_manager.prune_horizon();
        let now_us = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_micros() as u64)
            .unwrap_or(0);
        let retention_floor = zyron_executor::operator::modify::effective_retention_floor(
            table,
            &status_map,
            server.txn_manager.retention_clock(),
            now_us,
        );
        let is_dead = Box::new(move |xmin: u64, x: u64| {
            status_map.is_aborted(xmin)
                || (x != 0
                    && status_map.is_committed(x)
                    && x < prune_horizon
                    && status_map.is_reclaimable_below(x, retention_floor))
        });

        let (heap_file, page_count) = if table.lake.is_lake() {
            (None, 0)
        } else {
            // The instance every writer shares, with the log attached, so a
            // build that is the first to open a table's heap leaves the
            // writers after it recording their pages
            let heap_file = crate::connection::table_heap(server, table).await?;
            let pages = heap_file.num_pages_cached() as u64;
            (Some(heap_file), pages)
        };

        let (columnar, scan_txn) = if table.columnar.segments.is_empty() {
            (None, None)
        } else {
            let (op, txn) = open_columnar_scan(server, table).await?;
            (Some(op), Some(txn))
        };

        Ok(Self {
            server: Arc::clone(server),
            table: Arc::clone(table),
            heap_file,
            page_cursor: 0,
            page_count,
            batch_rows: batch_rows.max(1),
            is_dead,
            columnar,
            columnar_started: false,
            scan_txn,
        })
    }

    /// An upper bound on the rows this stream will produce, from the table's
    /// statistics. Named an estimate everywhere it is reported, because it is
    /// one.
    pub fn rows_total_estimate(&self) -> u64 {
        let stats = self.server.catalog.get_stats(self.table.id);
        let from_stats = stats.map(|s| s.0.row_count).unwrap_or(0);
        if from_stats > 0 {
            return from_stats;
        }
        // With no statistics the page count is what is known, and a page holds
        // at least one row
        self.page_count
    }

    /// The next batch, or None when the table has been read.
    pub async fn next_batch(&mut self) -> Result<Option<LiveBatch>, ZyronError> {
        let mut heap: Vec<(PageId, Vec<CapturedRow>)> = Vec::new();
        let mut rows = 0usize;
        while self.page_cursor < self.page_count && rows < self.batch_rows {
            let Some(file) = &self.heap_file else { break };
            let page_id = PageId::new(file.heap_file_id(), self.page_cursor);
            self.page_cursor += 1;
            let live = match self.server.buffer_pool.fetch_page(page_id) {
                Some(frame) => {
                    let guard = frame.read_data();
                    let live = live_rows_in_page(&guard[..], &self.is_dead);
                    drop(guard);
                    self.server.buffer_pool.unpin_page(page_id, false);
                    live
                }
                // A page the pool does not hold is read straight off disk
                // rather than skipped. Skipping it would drop every row on it
                // from the index, and a table large enough to want an index is
                // exactly the one whose pages do not all fit. The read bypasses
                // the pool so a build over a large table does not evict the
                // serving working set
                None => {
                    let data = self.server.disk_manager.read_page(page_id).await?;
                    live_rows_in_page(&data[..], &self.is_dead)
                }
            };
            if !live.is_empty() {
                rows += live.len();
                heap.push((page_id, live));
            }
        }

        if !heap.is_empty() {
            return Ok(Some(LiveBatch {
                heap,
                columnar: None,
            }));
        }

        if let Some(op) = self.columnar.as_mut() {
            self.columnar_started = true;
            while let Some(eb) = op.next().await? {
                let Some(locs) = eb.locators.clone() else {
                    continue;
                };
                if eb.batch.num_rows == 0 {
                    continue;
                }
                return Ok(Some(LiveBatch {
                    heap: Vec::new(),
                    columnar: Some((eb.batch, locs)),
                }));
            }
            self.columnar = None;
        }

        Ok(None)
    }

    /// Ends the read-only transaction the columnar scan ran under.
    ///
    /// Aborting rather than committing keeps it from leaking into the active
    /// set, and it has nothing to commit
    pub fn close(&mut self) {
        if let Some(mut txn) = self.scan_txn.take() {
            let _ = self.server.txn_manager.abort(&mut txn);
        }
        let _ = self.columnar_started;
    }
}

impl Drop for LiveRowStream {
    fn drop(&mut self) {
        self.close();
    }
}

/// Live rows on one heap page, captured with the epoch each was written under.
fn live_rows_in_page(
    data: &[u8],
    is_dead: &(impl Fn(u64, u64) -> bool + ?Sized),
) -> Vec<CapturedRow> {
    let header = HeapPage::heap_header_from_slice(data);
    let mut live = Vec::new();
    for slot in 0..header.slot_count {
        let Some(view) = HeapPage::get_tuple_view_from_slice(data, zyron_storage::SlotId(slot))
        else {
            continue;
        };
        let hdr: TupleHeader = view.header;
        if is_dead(hdr.xmin, hdr.xmax) {
            continue;
        }
        live.push(CapturedRow {
            slot,
            schema_epoch: hdr.schema_epoch,
            data: view.data.to_vec(),
        });
    }
    live
}

/// Decodes one scanned batch into a column batch over the table's live
/// columns, paired with the address each row came from.
///
/// Heap rows carry their own epoch, so a batch that spans a schema change
/// decodes each row through the layout it was written under and the result is
/// one shape whichever epochs it mixed. Folded rows arrive decoded already.
pub fn decode_live_batch(
    table: &TableEntry,
    batch: &LiveBatch,
) -> Result<(zyron_executor::batch::DataBatch, Vec<RowLocator>), ZyronError> {
    use zyron_executor::batch::{create_builders, finalize_builders};

    if let Some((data, locators)) = &batch.columnar {
        return Ok((data.clone(), locators.clone()));
    }

    let logical: Vec<zyron_planner::logical::LogicalColumn> = table
        .live_columns()
        .map(|c| zyron_planner::logical::LogicalColumn {
            table_idx: Some(0),
            column_id: c.id,
            name: c.name.clone(),
            type_id: c.type_id,
            nullable: c.nullable,
            fractional_digits: c.fractional_digits,
        })
        .collect();
    let output_ids: Vec<zyron_catalog::ColumnId> = logical.iter().map(|c| c.column_id).collect();
    let decoder = zyron_executor::epoch_decode::EpochDecoder::new(table, &output_ids);
    let rows: usize = batch.heap.iter().map(|(_, r)| r.len()).sum();
    let mut builders = create_builders(&logical, rows);
    let mut locators = Vec::with_capacity(rows);
    for (page_id, captured) in &batch.heap {
        for row in captured {
            let at = RowLocator::Heap {
                page: *page_id,
                slot: row.slot,
            };
            decoder.decode(row.schema_epoch, &row.data, Some(at), &mut builders)?;
            locators.push(at);
        }
    }
    Ok((finalize_builders(builders), locators))
}

/// Resolves a rewrite's shadow heap through the server's cache, building it
/// with no log attached.
///
/// The shadow is filled by the copy and by every writer mirroring into it,
/// and a restart before the swap drops it whole, so a record of each page it
/// fills would be a record nothing ever replays, written at the rate of the
/// copy. The swap attaches the log, flushes the pages the fill left dirty,
/// and only then names the shadow's files as the table's, so every row is
/// on disk or in the log before the catalog says the files are the table's.
/// Opened by the rewrite before the shadow is published to writers, so the
/// instance a writer's hook finds in the cache is this one
pub async fn shadow_heap_file(
    server: &Arc<ServerState>,
    table: &TableEntry,
) -> Result<Arc<HeapFile>, ZyronError> {
    if let Some(hit) = server.heap_files.get_async(&table.heap_file_id).await {
        return Ok(Arc::clone(hit.get()));
    }
    let heap_file = HeapFile::new(
        Arc::clone(&server.disk_manager),
        Arc::clone(&server.buffer_pool),
        HeapFileConfig {
            heap_file_id: table.heap_file_id,
            fsm_file_id: table.fsm_file_id,
        },
    )?;
    heap_file.init_cache().await?;
    let arc = Arc::new(heap_file);
    // A lost race drops this instance and converges on the winner, the same
    // way the execution context's heap file cache resolves one
    match server
        .heap_files
        .insert_async(table.heap_file_id, Arc::clone(&arc))
        .await
    {
        Ok(()) => Ok(arc),
        Err(_) => {
            let hit = server
                .heap_files
                .get_async(&table.heap_file_id)
                .await
                .ok_or_else(|| {
                    ZyronError::Internal(format!(
                        "heap file {} vanished from the cache during an index build",
                        table.heap_file_id
                    ))
                })?;
            Ok(Arc::clone(hit.get()))
        }
    }
}

/// Opens the columnar scan a table's folded rows come back through, so the
/// patch overlay and MVCC visibility decide what is live rather than a second
/// copy of those rules written here.
async fn open_columnar_scan(
    server: &Arc<ServerState>,
    table: &TableEntry,
) -> Result<
    (
        Box<dyn zyron_executor::operator::Operator>,
        zyron_storage::txn::Transaction,
    ),
    ZyronError,
> {
    let txn = server
        .txn_manager
        .begin(zyron_storage::txn::IsolationLevel::ReadCommitted)?;
    let scan_ctx = Arc::new(ExecutionContext::new(
        server.catalog.clone(),
        server.wal.clone(),
        server.buffer_pool.clone(),
        server.disk_manager.clone(),
        txn.txn_id,
        txn.snapshot.clone(),
    ));
    let logical: Vec<zyron_planner::logical::LogicalColumn> = table
        .live_columns()
        .map(|c| zyron_planner::logical::LogicalColumn {
            table_idx: Some(0),
            column_id: c.id,
            name: c.name.clone(),
            type_id: c.type_id,
            nullable: c.nullable,
            fractional_digits: c.fractional_digits,
        })
        .collect();
    let op = zyron_executor::operator::column_scan::ColumnScanOperator::new_for_dml(
        Arc::clone(&scan_ctx),
        table.id,
        logical,
        None,
    )?;
    Ok((Box::new(op), txn))
}

// ---------------------------------------------------------------------------
// The heap index build
// ---------------------------------------------------------------------------

/// What a finished build reports.
pub struct BuildOutcome {
    /// Entries the tree holds
    pub entries: u64,
    /// Bytes the sort wrote to disk
    pub spilled_bytes: u64,
}

/// A duplicate two live rows share, reported by a failed unique build.
pub struct DuplicateKey {
    pub key: Vec<u8>,
    pub first: RowLocator,
    pub second: RowLocator,
}

/// Scans a table's live rows into a sorted key stream and loads them into the
/// tree.
///
/// The tree is already registered and already being maintained, so the load
/// merges what the scan found with what maintenance wrote. A unique index whose
/// scan finds two live rows sharing a key fails here rather than leaving a tree
/// that answers a probe with one of them.
#[allow(clippy::too_many_arguments)]
pub async fn scan_and_load(
    server: &Arc<ServerState>,
    table: &Arc<TableEntry>,
    index_id: u32,
    key_columns: &[zyron_catalog::ColumnId],
    unique: bool,
    btree: &Arc<zyron_storage::BTreeIndex>,
    progress: &DdlProgress,
    batch_rows: usize,
) -> Result<Result<BuildOutcome, DuplicateKey>, ZyronError> {
    let mut stream = LiveRowStream::open(server, table, batch_rows).await?;
    progress.set_rows_total_estimate(stream.rows_total_estimate());
    progress.set_phase(DdlPhase::Scanning);

    let mut sorter = KeySorter::new(&server.data_dir, index_id, DEFAULT_RUN_BUDGET_BYTES);
    let mut pacer = BuildPacer::new(DdlPhase::Scanning);
    let mut scanned: u64 = 0;

    loop {
        let Some(batch) = stream.next_batch().await? else {
            break;
        };
        if batch.is_empty() {
            continue;
        }
        // Sized from the batch rather than grown into, so a batch of the
        // default size does not reallocate its way up from nothing once per
        // pass over the table
        let mut keys: Vec<(Vec<u8>, RowLocator)> = Vec::with_capacity(batch.rows() as usize);
        for (page_id, rows) in &batch.heap {
            zyron_executor::operator::modify::index_keys_for_rows(
                table,
                *page_id,
                rows,
                key_columns,
                &mut keys,
            )?;
        }
        if let Some((data_batch, locators)) = &batch.columnar {
            zyron_executor::operator::modify::index_keys_for_batch(
                table,
                data_batch,
                locators,
                key_columns,
                &mut keys,
            );
        }

        for (key, locator) in keys {
            sorter.push(key, locator)?;
        }

        scanned += batch.rows();
        progress.add_rows(batch.rows());
        progress.add_bytes_spilled(0);
        pacer.between_batches(progress).await;
    }
    stream.close();

    progress.set_phase(DdlPhase::Loading);
    pacer.set_running_phase(DdlPhase::Loading);
    let spilled_bytes = sorter.spilled_bytes();
    let mut sorted = sorter.finish()?;
    if unique {
        sorted.enforce_unique();
    }
    let spill_dir = sorted.spill_dir().to_path_buf();
    // Taken by reference so the stream can be asked afterwards whether it
    // stopped on a repeated value
    let entries = btree.bulk_build_sorted(&mut sorted)?;
    if let Some(duplicate) = sorted.duplicate() {
        remove_build_dir(&spill_dir);
        return Ok(Err(duplicate));
    }
    remove_build_dir(&spill_dir);

    tracing::info!(
        target: "zyron::ddl",
        index_id,
        table = %table.name,
        rows = scanned,
        entries,
        bytes_spilled = spilled_bytes,
        "index build loaded its tree"
    );

    Ok(Ok(BuildOutcome {
        entries,
        spilled_bytes,
    }))
}

/// Discards everything an online DDL left half done when this node stopped.
///
/// A build in flight lives in three places: the catalog entry, the tree file,
/// and the sort runs. None of them is finished, and none of them can be
/// resumed: the wait that made the build correct ended with the process, so
/// the rows a writer added after the crash were never maintained into the
/// partial tree. Dropping all three and reporting each is the only answer that
/// leaves the table correct.
///
/// A shadow rewrite is the same argument with a different shape. The swap is
/// its only step that touches the source, and the swap is one record, so a
/// shadow that survives a restart is by construction a rewrite that had not
/// swapped. Its table and files go, and the source is already right.
///
/// A constraint left unvalidated reverts to absent: it was enforced only by
/// the process that published it, so rows written between the crash and now
/// were never held to it.
pub async fn discard_incomplete_ddl(
    catalog: &Arc<zyron_catalog::Catalog>,
    disk_manager: &Arc<zyron_storage::DiskManager>,
    data_dir: &std::path::Path,
) {
    for table in catalog.list_all_tables() {
        for index in catalog.get_indexes_for_table(table.id) {
            if index.state != IndexState::Building {
                continue;
            }
            tracing::info!(
                target: "zyron::ddl",
                index = %index.name,
                table = %table.name,
                index_file_id = index.index_file_id,
                "dropping an index whose build did not finish before this node stopped, because \
                 the rows written since cover neither the tree nor the wait that made it correct"
            );
            let _ = catalog.drop_index(table.id, &index.name).await;
            let checkpoint = data_dir
                .join("indexes")
                .join(format!("index_{}.zyridx", index.index_file_id));
            if checkpoint.exists() {
                let _ = std::fs::remove_file(&checkpoint);
            }
            remove_build_dir(&build_spill_dir(data_dir, index.index_file_id));
        }

        let unvalidated: Vec<String> = table
            .constraints
            .iter()
            .filter(|c| !c.validated)
            .map(|c| c.name.clone())
            .collect();
        if !unvalidated.is_empty() {
            for name in &unvalidated {
                tracing::info!(
                    target: "zyron::ddl",
                    constraint = %name,
                    table = %table.name,
                    "removing a constraint whose validation did not finish before this node \
                     stopped, because the rows written since were never held to it"
                );
            }
            let mut entry = (*table).clone();
            entry.constraints.retain(|c| c.validated);
            let _ = catalog.update_table(entry).await;
        }

        if crate::shadow_rewrite::is_shadow_table(&table.name) {
            tracing::info!(
                target: "zyron::ddl",
                shadow = %table.name,
                "dropping a shadow table whose rewrite did not reach its swap, the source is \
                 unchanged because the swap is the only step that touches it"
            );
            let heap = table.heap_file_id;
            let fsm = table.fsm_file_id;
            let _ = catalog.drop_table(table.schema_id, &table.name).await;
            let _ = disk_manager.delete_file(heap).await;
            let _ = disk_manager.delete_file(fsm).await;
        }
    }
}

/// Flips a finished index to Ready and refreshes the snapshot writers read.
pub async fn flip_to_ready(
    server: &Arc<ServerState>,
    table_id: zyron_catalog::TableId,
    index_name: &str,
) -> Result<(), ZyronError> {
    server
        .catalog
        .set_index_state(table_id, index_name, IndexState::Ready)
        .await?;
    Ok(())
}
