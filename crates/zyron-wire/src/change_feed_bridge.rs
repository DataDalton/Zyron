//! Reading a table's recorded changes, whichever store holds them.
//!
//! A heap table records its changes into its own change files as the DML
//! runs. A lake table records none, because its transaction log already says which
//! files each commit added and removed, so the same change records are
//! derived from the log on demand and no byte is written twice.
//!
//! Both arrive at the executor through one trait, so the change scan operator
//! is the same for either format. This is also where the planner reads what it
//! needs to resolve `LATEST`, refuse a range retention has reclaimed, and tell
//! EXPLAIN how many change files a scan will open

use std::sync::Arc;

use zyron_catalog::{Catalog, TableId};
use zyron_cdc::change_feed::{CdfRegistry, ChangeRange};
use zyron_common::{Result, ZyronError};
use zyron_executor::context::{
    ChangeColumnBlock, ChangeColumnSource, ChangeCursor, ChangeFeedReader, ChangeGroupShape,
    ChangeRecordHeads, ChangeRowRef, ChangeScanStats, ChangeSegment, ChangeWindow, FeedKey,
    SourceBoundary, WindowBound,
};
use zyron_planner::ChangeFeedFacts;

/// Reads change records for the executor and answers the planner's questions
/// about a feed
pub struct ChangeFeedBridge {
    feeds: Arc<CdfRegistry>,
    catalog: Arc<Catalog>,
    /// Where a lake table's transaction logs live
    data_dir: std::path::PathBuf,
    /// The branches of the database, which name the head a lake table's
    /// branch keeps its versions under. None on a server without branches,
    /// where no read names one
    branches: Option<Arc<zyron_versioning::BranchManager>>,
}

impl ChangeFeedBridge {
    pub fn new(
        feeds: Arc<CdfRegistry>,
        catalog: Arc<Catalog>,
        data_dir: std::path::PathBuf,
        branches: Option<Arc<zyron_versioning::BranchManager>>,
    ) -> Self {
        Self {
            feeds,
            catalog,
            data_dir,
            branches,
        }
    }

    /// The name a branch keeps its lake heads under
    fn branch_name(&self, branch: u64) -> Result<String> {
        let Some(branches) = self.branches.as_ref() else {
            return Err(ZyronError::CdcStreamError(format!(
                "branch {branch} cannot be read on a server without branches"
            )));
        };
        Ok(branches
            .get_branch(zyron_versioning::BranchId(branch))?
            .name)
    }

    /// The log a source's changes are derived from, the table's own or the
    /// head a branch keeps on it.
    ///
    /// A branch that has not forked the table yet forks it now, at the
    /// table's current version, the way a branch's feed on a heap table
    /// opens at the table's current version when the branch has written
    /// nothing there, so a read inside the branch knows where the table's
    /// history ends and the branch's begins
    fn lake_log_on(
        &self,
        table_id: u32,
        branch: Option<u64>,
    ) -> Result<Arc<zyron_lake::TransactionLog>> {
        let Some(branch) = branch else {
            return self.lake_log(table_id);
        };
        let name = self.branch_name(branch)?;
        let paths = zyron_lake::LakePaths::new(&self.data_dir, table_id);
        match zyron_lake::open_branch_shared(&paths, &name) {
            Ok(log) => Ok(log),
            Err(ZyronError::BranchNotFound(_)) => {
                let main = self.lake_log(table_id)?;
                let created_us = std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .map(|d| d.as_micros() as i64)
                    .unwrap_or(0);
                match zyron_lake::create_branch(&main, &name, None, created_us) {
                    // A write on the branch forking the same table at the
                    // same instant is the same outcome
                    Ok(_) | Err(ZyronError::BranchAlreadyExists(_)) => {}
                    Err(e) => return Err(e),
                }
                zyron_lake::open_branch_shared(&paths, &name)
            }
            Err(e) => Err(e),
        }
    }

    /// The source a lake table's feed registered for the table, or for a
    /// branch's head on it, registering the branch's on first use. A
    /// branch's source is opened from what the branch holds on disk, so
    /// one left unregistered by a restart is registered here again
    fn lake_source_on(
        &self,
        table_id: u32,
        branch: Option<u64>,
    ) -> Result<Arc<dyn zyron_cdc::DerivedChangeSource>> {
        let Some(branch) = branch else {
            return self.lake_source(table_id);
        };
        if let Some(source) = self.feeds.derived_on(table_id, Some(branch)) {
            return Ok(source);
        }
        let table = self.catalog.get_table_by_id(TableId(table_id))?;
        if !table.cdf_enabled {
            return Err(ZyronError::CdcStreamError(format!(
                "lake table {} has no change data feed, so a branch of it records no changes",
                table.name
            )));
        }
        let log = self.lake_log_on(table_id, Some(branch))?;
        register_lake_branch_source(
            &self.feeds,
            &self.data_dir,
            table_id,
            branch,
            log,
            table.cdf.before_image,
        )?;
        self.feeds
            .derived_on(table_id, Some(branch))
            .ok_or_else(|| {
                ZyronError::Internal(format!(
                    "the source of branch {branch} on lake table {table_id} was not registered"
                ))
            })
    }

    /// True when the table keeps its rows in the lake rather than the heap
    fn is_lake(&self, table_id: u32) -> bool {
        self.catalog
            .get_table_by_id(TableId(table_id))
            .map(|table| table.lake.is_lake())
            .unwrap_or(false)
    }

    /// The transaction log a lake table's changes are derived from.
    ///
    /// Looked up in the shared registry rather than opened, so a change read
    /// sees exactly the log the writers on this node are committing through
    fn lake_log(&self, table_id: u32) -> Result<Arc<zyron_lake::TransactionLog>> {
        let paths = zyron_lake::LakePaths::new(&self.data_dir, table_id);
        zyron_lake::TransactionLog::lookup_shared(&paths).ok_or_else(|| {
            ZyronError::CdcStreamError(format!(
                "lake table {table_id} has no open transaction log on this node, so its \
                 changes cannot be read"
            ))
        })
    }

    /// The source a lake table's feed registered, which keeps the record
    /// index a stream position is counted in
    fn lake_source(&self, table_id: u32) -> Result<Arc<dyn zyron_cdc::DerivedChangeSource>> {
        self.feeds.derived(table_id).ok_or_else(|| {
            ZyronError::CdcStreamError(format!(
                "lake table {table_id} has no change data feed registered on this node, so \
                 its changes cannot be read"
            ))
        })
    }

    /// The window a read asks for, as a feed range
    fn range_of(window: &ChangeWindow) -> ChangeRange {
        let mut range = ChangeRange {
            start_version: window.from_exclusive.saturating_add(1),
            end_version: window.to_inclusive,
            start_timestamp: window.from_timestamp,
            end_timestamp: window.to_timestamp,
            change_types: window.change_types,
        };
        if range.end_version < range.start_version {
            range.end_version = range.start_version.saturating_sub(1);
        }
        range
    }

    /// The version a lake table's feed began at, zero when the table is
    /// not known
    fn lake_first_version(&self, table_id: u32) -> u64 {
        self.catalog
            .get_table_by_id(TableId(table_id))
            .map(|table| table.cdf.first_version)
            .unwrap_or(0)
    }

    /// Whether a record sits after the position a resumed read left off at
    #[inline]
    fn after_resume(resume: Option<(u64, u64)>, version: u64, ordinal: u64) -> bool {
        match resume {
            None => true,
            Some((at_version, at_ordinal)) => {
                version > at_version || (version == at_version && ordinal > at_ordinal)
            }
        }
    }
}

/// A read of a heap table's window in progress, over the feed's own scan
struct HeapCursor {
    scan: zyron_cdc::change_feed::SegmentScan,
    /// The files the scan reaches and the ones it pruned, known once it
    /// planned
    stats: ChangeScanStats,
}

/// One feed record as the scan hands it over
#[inline]
fn row_of(record: zyron_cdc::ChangeRecordRef<'_>) -> ChangeRowRef<'_> {
    ChangeRowRef {
        table_id: record.table_id,
        change_type: record.change_type as u8,
        commit_version: record.commit_version,
        commit_timestamp: record.commit_timestamp,
        txn_id: record.txn_id,
        change_ordinal: record.change_ordinal,
        schema_epoch: record.schema_version as u16,
        projected: record.projected,
        row_data: record.row_data,
    }
}

impl ChangeCursor for HeapCursor {
    fn next(&mut self, visit: &mut dyn FnMut(ChangeRowRef<'_>) -> Result<bool>) -> Result<bool> {
        self.scan.next(&mut |record| visit(row_of(record)))
    }

    fn stats(&self) -> ChangeScanStats {
        self.stats
    }

    fn take_segments(&mut self) -> Vec<Box<dyn ChangeSegment>> {
        self.scan
            .take_parts()
            .into_iter()
            .map(|part| Box::new(HeapSegment { part }) as Box<dyn ChangeSegment>)
            .collect()
    }
}

/// One file of a heap table's feed, loaded and walked on its own
struct HeapSegment {
    part: zyron_cdc::change_feed::SegmentPart,
}

impl ChangeSegment for HeapSegment {
    fn visit(&mut self, visit: &mut dyn FnMut(ChangeRowRef<'_>) -> Result<bool>) -> Result<()> {
        self.part.visit(&mut |record| visit(row_of(record)))
    }

    fn columns(&mut self) -> Result<Option<Box<dyn ChangeColumnSource>>> {
        let Some(sliced) = self.part.columns()? else {
            return Ok(None);
        };
        Ok(Some(Box::new(SlicedFile::new(sliced))))
    }
}

/// A sealed file in its column-sliced form, as the scan reads it
struct SlicedFile {
    segment: zyron_cdc::segment_columns::ColumnarSegment,
    heads: ChangeRecordHeads,
    groups: Vec<ChangeGroupShape>,
}

impl SlicedFile {
    fn new(segment: zyron_cdc::segment_columns::ColumnarSegment) -> Self {
        let source = segment.heads();
        let heads = ChangeRecordHeads {
            change_types: source.change_types.clone(),
            versions: source.versions.clone(),
            timestamps: source.timestamps.clone(),
            txn_ids: source.txn_ids.clone(),
            ordinals: source.ordinals.clone(),
            group_of: source.group_of.clone(),
        };
        let groups = segment
            .groups()
            .iter()
            .map(|group| ChangeGroupShape {
                epoch: group.epoch as u16,
                projected: group.projected,
                rows: group.rows,
                types: group.types.clone(),
            })
            .collect();
        Self {
            segment,
            heads,
            groups,
        }
    }
}

impl ChangeColumnSource for SlicedFile {
    fn heads(&self) -> &ChangeRecordHeads {
        &self.heads
    }

    fn groups(&self) -> &[ChangeGroupShape] {
        &self.groups
    }

    fn column_nulls(&self, group: usize, column: usize) -> Result<Vec<u8>> {
        self.segment.column_nulls(group, column)
    }

    fn column_values_into(&self, group: usize, column: usize, out: &mut [u8]) -> Result<()> {
        self.segment.column_values_into(group, column, out)
    }

    fn column(&self, group: usize, column: usize) -> Result<ChangeColumnBlock> {
        Ok(match self.segment.column(group, column)? {
            zyron_cdc::segment_columns::ColumnBlock::Fixed {
                width,
                nulls,
                values,
            } => ChangeColumnBlock::Fixed {
                width,
                nulls,
                values,
            },
            zyron_cdc::segment_columns::ColumnBlock::Varlen {
                nulls,
                offsets,
                bytes,
            } => ChangeColumnBlock::Varlen {
                nulls,
                offsets,
                bytes,
            },
        })
    }
}

/// A read of a lake table's window in progress, derived from its log one
/// commit version at a time, so a read of many versions holds one
/// version's changes at once rather than the whole range's
struct LakeCursor {
    log: Arc<zyron_lake::TransactionLog>,
    /// The feed's source, which says under which setting each version's
    /// records were counted and are derived
    source: Arc<dyn zyron_cdc::DerivedChangeSource>,
    table: Arc<zyron_catalog::TableEntry>,
    window: ChangeWindow,
    resume: Option<(u64, u64)>,
    /// The next version to derive, and the last one the window reaches
    next_version: u64,
    to_version: u64,
    /// The version derived last, and how far its records were handed over
    held: crate::lake_changes::LakeChangeRows,
    held_at: usize,
    stats: ChangeScanStats,
}

/// Whether a derived record is inside the window and past the position a
/// resumed read left off at
#[inline]
fn lake_admits(
    window: &ChangeWindow,
    resume: Option<(u64, u64)>,
    record: &crate::lake_changes::LakeChangeRow,
) -> bool {
    if !ChangeFeedBridge::after_resume(resume, record.commit_version, record.change_ordinal) {
        return false;
    }
    if let Some(mask) = window.change_types
        && mask & zyron_common::change_type_bit(record.change_type as u8) == 0
    {
        return false;
    }
    record.commit_timestamp >= window.from_timestamp
        && record.commit_timestamp <= window.to_timestamp
}

/// Hands the admitted records of one derived version to the visitor from
/// `from`, answering with how many were passed over or handed and whether
/// the visitor stopped
fn visit_lake_rows(
    rows: &crate::lake_changes::LakeChangeRows,
    from: usize,
    window: &ChangeWindow,
    resume: Option<(u64, u64)>,
    visit: &mut dyn FnMut(ChangeRowRef<'_>) -> Result<bool>,
) -> Result<(usize, bool)> {
    let mut at = from;
    while at < rows.len() {
        let record = rows.row(at);
        at += 1;
        if !lake_admits(window, resume, &record) {
            continue;
        }
        let keep_going = visit(ChangeRowRef {
            table_id: record.table_id,
            change_type: record.change_type as u8,
            commit_version: record.commit_version,
            commit_timestamp: record.commit_timestamp,
            txn_id: record.txn_id,
            change_ordinal: record.change_ordinal,
            schema_epoch: record.schema_version as u16,
            projected: false,
            row_data: record.row_data,
        })?;
        if !keep_going {
            return Ok((at, true));
        }
    }
    Ok((at, false))
}

impl ChangeCursor for LakeCursor {
    fn next(&mut self, visit: &mut dyn FnMut(ChangeRowRef<'_>) -> Result<bool>) -> Result<bool> {
        loop {
            let (at, stopped) =
                visit_lake_rows(&self.held, self.held_at, &self.window, self.resume, visit)?;
            self.held_at = at;
            if stopped {
                return Ok(true);
            }
            if self.next_version > self.to_version {
                return Ok(false);
            }
            let version = self.next_version;
            self.next_version += 1;
            self.held = crate::lake_changes::lake_change_rows(
                &self.log,
                &self.table,
                version,
                version,
                self.source.preimages_at(version),
            )?;
            self.held_at = 0;
        }
    }

    fn stats(&self) -> ChangeScanStats {
        self.stats
    }

    fn take_segments(&mut self) -> Vec<Box<dyn ChangeSegment>> {
        let mut parts: Vec<Box<dyn ChangeSegment>> = Vec::new();
        // The version derived last is handed over from where it stopped,
        // then every version left, each derived by whichever thread runs it
        if self.held_at < self.held.len() {
            parts.push(Box::new(LakeSegment {
                log: Arc::clone(&self.log),
                source: Arc::clone(&self.source),
                table: Arc::clone(&self.table),
                window: self.window,
                resume: self.resume,
                version: 0,
                held: Some((std::mem::take(&mut self.held), self.held_at)),
            }));
        }
        self.held_at = 0;
        for version in self.next_version..=self.to_version {
            parts.push(Box::new(LakeSegment {
                log: Arc::clone(&self.log),
                source: Arc::clone(&self.source),
                table: Arc::clone(&self.table),
                window: self.window,
                resume: self.resume,
                version,
                held: None,
            }));
        }
        self.next_version = self.to_version.saturating_add(1);
        parts
    }
}

/// One commit version of a lake table's log, derived and walked on its own
struct LakeSegment {
    log: Arc<zyron_lake::TransactionLog>,
    /// The feed's source, which says under which setting the version's
    /// records were counted and are derived
    source: Arc<dyn zyron_cdc::DerivedChangeSource>,
    table: Arc<zyron_catalog::TableEntry>,
    window: ChangeWindow,
    resume: Option<(u64, u64)>,
    version: u64,
    /// Rows already derived, with how far they were handed over, for the
    /// version a cursor stood inside when its parts were taken
    held: Option<(crate::lake_changes::LakeChangeRows, usize)>,
}

impl ChangeSegment for LakeSegment {
    fn columns(&mut self) -> Result<Option<Box<dyn ChangeColumnSource>>> {
        Ok(None)
    }

    fn visit(&mut self, visit: &mut dyn FnMut(ChangeRowRef<'_>) -> Result<bool>) -> Result<()> {
        let (rows, from) = match self.held.take() {
            Some(held) => held,
            None => (
                crate::lake_changes::lake_change_rows(
                    &self.log,
                    &self.table,
                    self.version,
                    self.version,
                    self.source.preimages_at(self.version),
                )?,
                0,
            ),
        };
        visit_lake_rows(&rows, from, &self.window, self.resume, visit)?;
        Ok(())
    }
}

impl ChangeFeedBridge {
    /// Opens a read of a lake table's window, derived from its log
    fn open_lake_window(
        &self,
        window: &ChangeWindow,
        resume: Option<(u64, u64)>,
    ) -> Result<Box<dyn ChangeCursor>> {
        let table = self.catalog.get_table_by_id(TableId(window.table_id))?;
        let log = self.lake_log_on(window.table_id, window.branch)?;
        let source = self.lake_source_on(window.table_id, window.branch)?;
        // The feed's changes begin above the version it was turned on at
        let from = window
            .from_exclusive
            .max(table.cdf.first_version)
            .saturating_add(1);
        // A read that resumes inside the range starts at the version it
        // stopped in, every version below it having been handed over. A
        // version whose every record was handed over is not derived again,
        // which also keeps a read whose position stands at the retention
        // floor from reaching below the floor for that version's base
        let (next_version, resume) = match resume {
            Some((at_version, at_ordinal)) => {
                let whole = source.records_before(at_version, at_ordinal.saturating_add(1))?
                    >= source.records_at_or_below(at_version)?;
                if whole {
                    (from.max(at_version.saturating_add(1)), None)
                } else {
                    (from.max(at_version), resume)
                }
            }
            None => (from, None),
        };
        // The window's upper bound is whatever the planner resolved, which
        // for an open-ended read is the largest version there is, so the
        // read ends at the newest commit the log holds rather than walking
        // versions that do not exist
        let to_version = window.to_inclusive.min(log.latest_version());
        // A lake change is derived from a commit, so the versions the read
        // walks are what a file count means here
        let versions = to_version.saturating_add(1).saturating_sub(next_version) as usize;
        Ok(Box::new(LakeCursor {
            log,
            source,
            table,
            window: *window,
            resume,
            next_version,
            to_version,
            held: crate::lake_changes::LakeChangeRows::default(),
            held_at: 0,
            stats: ChangeScanStats {
                files_opened: versions,
                files_pruned: 0,
            },
        }))
    }
}

/// A lake table as a source of derived changes, which is its transaction log
struct LakeChangeSource {
    log: Arc<zyron_lake::TransactionLog>,
    /// The records each version above the feed's first yields, counted the
    /// way the feed's own index counts a heap table's, taken from the log
    /// as versions are asked about and kept on disk from then on. A
    /// committed version never changes, so what was counted once stays
    /// counted, and it has to, because retention reclaims the versions it
    /// was counted from
    index: parking_lot::Mutex<zyron_cdc::DerivedRecordIndex>,
}

/// What one version yields, with its commit's transaction resolved to the
/// database transaction a reader's snapshot judges it by. A commit made
/// under an intent whose owner is not recorded is handed over with the
/// intent itself, which is what ties its commits to each other
fn counted_version(
    log: &zyron_lake::TransactionLog,
    yielded: zyron_lake::VersionYield,
) -> zyron_cdc::CountedVersion {
    let txn_id = zyron_lake::owning_txn_of(log.paths(), yielded.db_txn_id, yielded.owner_txn_id);
    zyron_cdc::CountedVersion {
        records: yielded.records,
        first_timestamp: yielded.timestamp_us,
        txn_id,
        span_txn: if txn_id != 0 {
            txn_id
        } else {
            yielded.db_txn_id
        },
    }
}

impl LakeChangeSource {
    /// The index built through `version`, or through the log's newest
    /// version when that is lower. Each version the index lacks costs one
    /// read of what its commit changed, paid once.
    ///
    /// A version whose changes cannot be read stops the extension there,
    /// the index never records it as yielding nothing, and the error is
    /// the answer, so a position is never moved by a count that stopped
    /// short of what a read handed over
    fn index_through(
        &self,
        version: u64,
    ) -> Result<parking_lot::MutexGuard<'_, zyron_cdc::DerivedRecordIndex>> {
        let mut index = self.index.lock();
        let through = version.min(self.log.latest_version());
        if through > index.built_through() {
            index.extend(through, |version, before_images| {
                Ok(
                    zyron_lake::change_records_at(&self.log, version, before_images)?
                        .map(|yielded| counted_version(&self.log, yielded)),
                )
            })?;
        }
        Ok(index)
    }

    /// The index built through the log's newest version
    fn whole_index(&self) -> Result<parking_lot::MutexGuard<'_, zyron_cdc::DerivedRecordIndex>> {
        self.index_through(self.log.latest_version())
    }

    /// The index as far as it reaches, for an answer that is a reading
    /// rather than a position. What the index could not count is reported
    /// and the reading covers what it did
    fn index_so_far(&self) -> parking_lot::MutexGuard<'_, zyron_cdc::DerivedRecordIndex> {
        match self.whole_index() {
            Ok(index) => index,
            Err(e) => {
                tracing::warn!(
                    error = %e,
                    "the changes of a lake version could not be counted for its feed's index"
                );
                self.index.lock()
            }
        }
    }
}

impl zyron_cdc::DerivedChangeSource for LakeChangeSource {
    fn latest_version(&self) -> u64 {
        self.log.latest_version()
    }

    fn oldest_readable_version(&self) -> Option<u64> {
        // A position is stale when AS OF that version would fail, so the
        // oldest readable position is the oldest version the log can still
        // stand at. A log whose directory cannot be listed reports no floor,
        // and a read below the real one fails at the read naming the
        // version rather than being passed over here
        match self.log.oldest_readable_version() {
            Ok(oldest) => oldest,
            Err(e) => {
                tracing::warn!(
                    target: "zyron::cdc",
                    error = %e,
                    "the oldest readable version of a lake table's log could not be read"
                );
                None
            }
        }
    }

    fn pending_after(&self, position: u64) -> u64 {
        let index = self.index_so_far();
        index
            .records()
            .saturating_sub(index.records_at_or_below(position))
    }

    fn first_timestamp_after(&self, position: u64) -> Option<i64> {
        let index = self.index_so_far();
        zyron_cdc::version_index::first_after(index.versions(), position)
            .map(|entry| entry.first_timestamp)
    }

    fn version_at_timestamp(&self, timestamp: i64) -> u64 {
        zyron_lake::resolve_version(&self.log, zyron_lake::TimeTravelSpec::Timestamp(timestamp))
            .unwrap_or(0)
    }

    fn records_at_or_below(&self, version: u64) -> Result<u64> {
        Ok(self.index_through(version)?.records_at_or_below(version))
    }

    fn records_before(&self, version: u64, ordinal: u64) -> Result<u64> {
        let index = self.index_through(version)?;
        Ok(zyron_cdc::version_index::records_before(
            index.versions(),
            version,
            ordinal,
        ))
    }

    fn cursor_at_count(&self, count: u64) -> Result<Option<(u64, u64)>> {
        let index = self.whole_index()?;
        Ok(zyron_cdc::version_index::cursor_at_count(
            index.versions(),
            count,
        ))
    }

    fn version_at_count(&self, count: u64) -> u64 {
        // A member re-addressing a replicated count answers from what it
        // has counted, a version no later than the one the count reaches,
        // so a read from there repeats records rather than skipping any
        let index = self.index_so_far();
        zyron_cdc::version_index::version_at_count(index.versions(), count)
    }

    fn bounded_cut(&self, consumed: u64, max_rows: u64) -> Result<Option<u64>> {
        let index = self.whole_index()?;
        Ok(zyron_cdc::version_index::bounded_cut(
            index.versions(),
            consumed,
            max_rows,
        ))
    }

    fn preimages_at(&self, version: u64) -> bool {
        self.index.lock().preimages_at(version)
    }

    fn boundary(&self, ended: &dyn Fn(u64) -> bool) -> Result<zyron_cdc::DerivedBoundary> {
        // The published head is read first, so a commit that publishes
        // while the index is extended lies above it and waits for the next
        // read, where the reader of a heap source holds it open too
        let latest = self.log.latest_version();
        let mut index = self.index_through(latest)?;
        let first_open = index.retire_in_flight(latest, ended);
        Ok(zyron_cdc::DerivedBoundary {
            latest,
            records: index.records_at_or_below(latest),
            first_open,
        })
    }

    fn writers_above(&self, head: u64, aborted: &dyn Fn(u64) -> bool) -> Result<Vec<u64>> {
        // A version above the published head is pending under a
        // transaction the log names, or published since the head was read,
        // in which case its commit header names the transaction. A file
        // that vanished on the way was abandoned with its transaction
        let pending: Vec<(u64, u64)> = self.log.pending_versions();
        let created = self.log.head_version();
        let mut writers = Vec::new();
        for version in head.saturating_add(1)..=created {
            let txn_id = match pending.iter().find(|(v, _)| *v == version) {
                Some((_, txn)) => zyron_lake::owning_txn(self.log.paths(), *txn),
                None => match self.log.commit_header(version) {
                    Ok(header) => zyron_lake::owning_txn_of(
                        self.log.paths(),
                        header.db_txn_id,
                        header.owner_txn_id,
                    ),
                    Err(ZyronError::Io(e)) if e.kind() == std::io::ErrorKind::NotFound => {
                        continue;
                    }
                    Err(e) => return Err(e),
                },
            };
            if txn_id != 0 && !aborted(txn_id) && !writers.contains(&txn_id) {
                writers.push(txn_id);
            }
        }
        Ok(writers)
    }

    fn txns_in(&self, from_exclusive: u64, to_inclusive: u64) -> Vec<u64> {
        self.index.lock().txns_in(from_exclusive, to_inclusive)
    }

    fn span_of(&self, txn_id: u64) -> Option<(u64, u64)> {
        self.index.lock().span_of(txn_id)
    }
}

/// Registers a lake table's transaction log as the source of its changes,
/// so a stream over it has a position the runtime can judge.
///
/// Called when the table's feed is turned on and again when the node
/// starts, because the log is opened per process
pub fn register_lake_source(
    feeds: &zyron_cdc::CdfRegistry,
    data_dir: &std::path::Path,
    table_id: u32,
    before_image: bool,
    first_version: u64,
) -> Result<()> {
    let paths = zyron_lake::LakePaths::new(data_dir, table_id);
    let log = zyron_lake::TransactionLog::lookup_shared(&paths).ok_or_else(|| {
        ZyronError::CdcStreamError(format!(
            "lake table {table_id} has no open transaction log on this node, so its \
             changes cannot be followed"
        ))
    })?;
    let mut index =
        zyron_cdc::DerivedRecordIndex::open(data_dir, table_id, first_version, before_image)?;
    // A setting that changed since the index was written takes effect
    // above the versions committed so far, each of those counted under the
    // setting it was committed under
    index.set_before_images(
        log.latest_version(),
        before_image,
        |version, before_images| {
            Ok(zyron_lake::change_records_at(&log, version, before_images)?
                .map(|yielded| counted_version(&log, yielded)))
        },
    )?;
    feeds.register_derived(
        table_id,
        Arc::new(LakeChangeSource {
            log,
            index: parking_lot::Mutex::new(index),
        }),
    );
    Ok(())
}

/// Registers the head a branch keeps on a lake table as the source of the
/// branch's changes of the table, counted into an index of the branch's
/// own under the branch's directory of the table's feed, beginning at the
/// version the branch forked the table at.
///
/// Called when a branch forks a table whose feed is on, when the feed is
/// turned on for a table with branches, when the node starts, and on the
/// first read of a branch left unregistered by any of those
pub fn register_lake_branch_source(
    feeds: &zyron_cdc::CdfRegistry,
    data_dir: &std::path::Path,
    table_id: u32,
    branch_id: u64,
    log: Arc<zyron_lake::TransactionLog>,
    before_image: bool,
) -> Result<()> {
    let dir = zyron_cdc::ChangeDataFeed::branch_dir(data_dir, table_id, branch_id);
    let mut index = zyron_cdc::DerivedRecordIndex::open_in(&dir, log.branch_base(), before_image)?;
    index.set_before_images(
        log.latest_version(),
        before_image,
        |version, before_images| {
            Ok(zyron_lake::change_records_at(&log, version, before_images)?
                .map(|yielded| counted_version(&log, yielded)))
        },
    )?;
    feeds.register_derived_on(
        table_id,
        Some(branch_id),
        Arc::new(LakeChangeSource {
            log,
            index: parking_lot::Mutex::new(index),
        }),
    );
    Ok(())
}

/// Registers the head of every branch that has forked a lake table, by the
/// branch's id in the branch manager, answering with how many. A head
/// whose name the manager does not know is a branch of the table alone,
/// which no stream is created on, and is passed over
pub fn register_lake_branch_sources(
    feeds: &zyron_cdc::CdfRegistry,
    data_dir: &std::path::Path,
    table_id: u32,
    before_image: bool,
    branches: &zyron_versioning::BranchManager,
) -> Result<usize> {
    let paths = zyron_lake::LakePaths::new(data_dir, table_id);
    let mut registered = 0usize;
    for info in zyron_lake::list_branches(&paths)? {
        let Ok(entry) = branches.get_branch_by_name(&info.name) else {
            continue;
        };
        let log = zyron_lake::open_branch_shared(&paths, &info.name)?;
        register_lake_branch_source(feeds, data_dir, table_id, entry.id.0, log, before_image)?;
        registered += 1;
    }
    Ok(registered)
}

/// Registers the head a branch keeps on a lake table as the source of the
/// branch's changes, forking the table onto the branch first when the
/// branch has not touched it, for the branch DDL and the stream DDL that
/// name a branch
pub fn open_lake_branch_source(
    server: &Arc<crate::connection::ServerState>,
    table_id: u32,
    branch_id: u64,
) -> Result<()> {
    let bridge = bridge_for(server).ok_or_else(|| {
        ZyronError::CdcStreamError(format!(
            "lake table {table_id} has no change data feed registry on this node"
        ))
    })?;
    bridge.lake_source_on(table_id, Some(branch_id)).map(|_| ())
}

/// Opens the feed of every heap table whose change data feed is on, the
/// way the table's own settings describe it.
///
/// Run once at startup, after the catalog has loaded, so a feed a previous
/// life of the process was recording into is recording again before the
/// first write lands and readable before the first read arrives. A feed
/// that stayed closed would record nothing and every stream over the table
/// would read nothing, with no error anywhere. A lake table's source is
/// its log, registered by `reopen_lake_sources` once the logs are open
pub fn reopen_feeds(catalog: &Catalog, feeds: &zyron_cdc::CdfRegistry) -> Result<usize> {
    let mut opened = 0usize;
    for table in catalog.list_all_tables() {
        if !table.cdf_enabled || table.dropped_at.is_some() || table.lake.is_lake() {
            continue;
        }
        feeds.enable_with_config(
            table.id.0,
            crate::lifecycle_dispatch::feed_config_of(&table),
        )?;
        opened += 1;
    }
    Ok(opened)
}

/// Registers the log of every lake table whose change data feed is on as
/// the source of its changes, and the head of every branch that has forked
/// such a table as the source of the branch's.
///
/// Run once at startup after the lake logs are open, because the source is
/// the open log itself. A table whose log this node does not run, the way
/// a member without the lake tier stands, leaves nothing to follow here and
/// is passed over. Any other failure stops the caller, since a source left
/// unregistered would leave every stream over the table reading nothing
pub fn reopen_lake_sources(
    catalog: &Catalog,
    feeds: &zyron_cdc::CdfRegistry,
    branches: Option<&zyron_versioning::BranchManager>,
) -> Result<usize> {
    let Some(data_dir) = catalog.data_dir() else {
        return Ok(0);
    };
    let mut registered = 0usize;
    for table in catalog.list_all_tables() {
        if !table.cdf_enabled || table.dropped_at.is_some() || !table.lake.is_lake() {
            continue;
        }
        let paths = zyron_lake::LakePaths::new(data_dir, table.id.0);
        if zyron_lake::TransactionLog::lookup_shared(&paths).is_none() {
            tracing::debug!(
                table = %table.name,
                "the lake table's log is not open on this node, so its changes are not followed here"
            );
            continue;
        }
        register_lake_source(
            feeds,
            data_dir,
            table.id.0,
            table.cdf.before_image,
            table.cdf.first_version,
        )?;
        registered += 1;
        if let Some(branches) = branches {
            registered += register_lake_branch_sources(
                feeds,
                data_dir,
                table.id.0,
                table.cdf.before_image,
                branches,
            )?;
        }
    }
    Ok(registered)
}

impl ChangeFeedReader for ChangeFeedBridge {
    fn open_window(
        &self,
        window: &ChangeWindow,
        resume: Option<(u64, u64)>,
    ) -> Result<Box<dyn ChangeCursor>> {
        if self.is_lake(window.table_id) {
            return self.open_lake_window(window, resume);
        }
        let Some(feed) = self.feeds.feed_on(window.table_id, window.branch) else {
            return Err(ZyronError::CdcStreamError(format!(
                "table {} has no change data feed open on this node",
                window.table_id
            )));
        };
        let scan = feed.open_scan(&Self::range_of(window), resume);
        let stats = ChangeScanStats {
            files_opened: scan.plan().segments.len(),
            files_pruned: scan.plan().pruned,
        };
        Ok(Box::new(HeapCursor { scan, stats }))
    }

    fn plan_window(&self, window: &ChangeWindow) -> Result<ChangeScanStats> {
        if self.is_lake(window.table_id) {
            return Ok(ChangeScanStats::default());
        }
        let Some(feed) = self.feeds.feed_on(window.table_id, window.branch) else {
            return Ok(ChangeScanStats::default());
        };
        let plan = feed.plan_read(&Self::range_of(window));
        Ok(ChangeScanStats {
            files_opened: plan.segments.len(),
            files_pruned: plan.pruned,
        })
    }

    /// What every source holds, read so that a transaction which wrote a
    /// heap table and a lake table is handed over to both at once.
    ///
    /// A lake commit becomes visible to readers when its transaction
    /// publishes it, a step after the commit record that makes the
    /// transaction's heap rows visible, and a read that landed between the
    /// two would take the rows and leave the commit for the next read.
    /// So each lake source is read first, at its published head, and the
    /// heap feeds are read after it, at one instant under their locks, with
    /// every transaction that has a lake commit above that head held open
    /// whatever the snapshot says of it. Such a commit lies above what this
    /// read takes from the lake, so its transaction's rows wait with it
    fn boundaries(
        &self,
        sources: &[FeedKey],
        outcome: &dyn Fn(u64) -> zyron_storage::txn::TxnStatus,
    ) -> Result<Vec<SourceBoundary>> {
        use zyron_storage::txn::TxnStatus;
        let ended = |txn_id: u64| outcome(txn_id) != TxnStatus::Active;
        let aborted = |txn_id: u64| outcome(txn_id) == TxnStatus::Aborted;
        let mut lake: Vec<(FeedKey, Arc<dyn zyron_cdc::DerivedChangeSource>)> = Vec::new();
        let mut heap: Vec<(u32, Option<u64>)> = Vec::new();
        for key in sources {
            if self.is_lake(key.table_id) {
                lake.push((*key, self.lake_source_on(key.table_id, key.branch)?));
            } else {
                heap.push((key.table_id, key.branch));
            }
        }
        // The transactions with a lake commit above the head readers see,
        // which every source holds open, the lake sources included, so a
        // transaction whose commits publish one table at a time is handed
        // over by none of them until every commit is visible. Gathered
        // before any boundary is read and before the heap feeds are
        // locked, since gathering them reads commit headers off the lake
        // logs and a read under the feed locks would hold every writer of
        // those feeds behind it. A boundary read after this sees a head at
        // or above the one gathered against, and a commit that published
        // in between is one of these, held until the next read
        let mut late_held: Vec<u64> = Vec::new();
        for (_, source) in &lake {
            for txn_id in source.writers_above(source.latest_version(), &aborted)? {
                if !late_held.contains(&txn_id) {
                    late_held.push(txn_id);
                }
            }
        }
        let ended_held = |txn_id: u64| ended(txn_id) && !late_held.contains(&txn_id);
        let lake: Vec<(FeedKey, zyron_cdc::DerivedBoundary)> = lake
            .iter()
            .map(|(key, source)| {
                source
                    .boundary(&ended_held)
                    .map(|boundary| (*key, boundary))
            })
            .collect::<Result<_>>()?;
        let held = self.feeds.boundaries(&heap, &ended_held);
        let mut out = Vec::with_capacity(sources.len());
        for key in sources {
            if let Some((_, boundary)) = lake.iter().find(|(lake_key, _)| lake_key == key) {
                out.push(SourceBoundary {
                    source: *key,
                    latest: boundary.latest,
                    records: boundary.records,
                    first_open: boundary.first_open,
                    lake: true,
                });
                continue;
            }
            let boundary = held
                .iter()
                .find(|(held_key, _)| *held_key == (key.table_id, key.branch))
                .map(|(_, b)| *b)
                .unwrap_or(zyron_cdc::FeedBoundary {
                    latest: 0,
                    records: 0,
                    first_open: None,
                });
            out.push(SourceBoundary {
                source: *key,
                latest: boundary.latest,
                records: boundary.records,
                first_open: boundary.first_open,
                lake: false,
            });
        }
        Ok(out)
    }

    fn records_before(&self, source: FeedKey, version: u64, ordinal: u64) -> Result<u64> {
        if self.is_lake(source.table_id) {
            return self
                .lake_source_on(source.table_id, source.branch)?
                .records_before(version, ordinal);
        }
        Ok(self
            .feeds
            .feed_on(source.table_id, source.branch)
            .map(|feed| feed.records_before(version, ordinal))
            .unwrap_or(0))
    }

    fn cursor_at_count(&self, source: FeedKey, count: u64) -> Result<Option<(u64, u64)>> {
        if self.is_lake(source.table_id) {
            return self
                .lake_source_on(source.table_id, source.branch)?
                .cursor_at_count(count);
        }
        Ok(self
            .feeds
            .feed_on(source.table_id, source.branch)
            .and_then(|feed| feed.cursor_at_count(count)))
    }

    fn bounded_cut(&self, sources: &[(FeedKey, u64, u64)], max_rows: u64) -> Result<Option<u64>> {
        // A lake source's cut is the version holding the record at the
        // bound. Moving it past the transactions inside the read is what
        // `align_windows` does, across every source at once
        let lake_cut = sources
            .iter()
            .filter(|(key, _, _)| self.is_lake(key.table_id))
            .map(|(key, _, consumed)| -> Result<Option<u64>> {
                self.lake_source_on(key.table_id, key.branch)?
                    .bounded_cut(*consumed, max_rows)
            })
            .collect::<Result<Vec<Option<u64>>>>()?
            .into_iter()
            .flatten()
            .min();
        let heap: Vec<((u32, Option<u64>), u64, u64)> = sources
            .iter()
            .filter(|(key, _, _)| !self.is_lake(key.table_id))
            .map(|(key, from, consumed)| ((key.table_id, key.branch), *from, *consumed))
            .collect();
        let heap_cut = if heap.is_empty() {
            None
        } else {
            self.feeds.bounded_cut(&heap, max_rows)
        };
        Ok(match (lake_cut, heap_cut) {
            (Some(a), Some(b)) => Some(a.min(b)),
            (a, b) => a.or(b),
        })
    }

    fn align_windows(&self, windows: &[WindowBound]) -> Result<Vec<u64>> {
        let reads: Vec<zyron_cdc::SourceRead> = windows
            .iter()
            .map(|window| zyron_cdc::SourceRead {
                table_id: window.source.table_id,
                branch: window.source.branch,
                from_exclusive: window.from_exclusive,
                to_inclusive: window.to_inclusive,
                limit: window.limit,
            })
            .collect();
        Ok(self.feeds.align_windows(&reads))
    }

    fn records_at_or_below(&self, source: FeedKey, version: u64) -> Result<u64> {
        if self.is_lake(source.table_id) {
            return self
                .lake_source_on(source.table_id, source.branch)?
                .records_at_or_below(version);
        }
        Ok(self
            .feeds
            .feed_on(source.table_id, source.branch)
            .map(|feed| feed.records_at_or_below(version))
            .unwrap_or(0))
    }

    fn branch_point(&self, table_id: u32, branch: u64) -> Result<Option<u64>> {
        // A lake table's changes come from its log, and a branch's changes
        // of it from the head the branch keeps, which begins at the version
        // the branch forked the table at. A branch that never forked the
        // table has nothing of its own, so a read inside it reads the table
        if self.is_lake(table_id) {
            let name = self.branch_name(branch)?;
            let paths = zyron_lake::LakePaths::new(&self.data_dir, table_id);
            return match zyron_lake::open_branch_shared(&paths, &name) {
                Ok(log) => Ok(Some(log.branch_base())),
                Err(ZyronError::BranchNotFound(_)) => Ok(None),
                Err(e) => Err(e),
            };
        }
        Ok(self
            .feeds
            .branch_feed(table_id, branch)
            .map(|feed| feed.branch_point()))
    }

    fn latest_version(&self, source: FeedKey) -> Result<u64> {
        if self.is_lake(source.table_id) {
            return Ok(self
                .lake_log_on(source.table_id, source.branch)?
                .latest_version());
        }
        Ok(self
            .feeds
            .feed_on(source.table_id, source.branch)
            .and_then(|feed| feed.latest_version())
            .unwrap_or(0))
    }
}

impl ChangeFeedFacts for ChangeFeedBridge {
    fn feed_enabled(&self, table_id: u32) -> bool {
        // A lake table records no separate feed. Its changes come from its
        // transaction log, so the table's own flag is what says whether a
        // consumer may read them
        if self.is_lake(table_id) {
            return self
                .catalog
                .get_table_by_id(TableId(table_id))
                .map(|table| table.cdf_enabled)
                .unwrap_or(false);
        }
        self.feeds
            .get_feed(table_id)
            .map(|feed| feed.is_enabled())
            .unwrap_or(false)
    }

    fn version_range(&self, table_id: u32) -> Option<(u64, u64)> {
        if self.is_lake(table_id) {
            let log = self.lake_log(table_id).ok()?;
            let head = log.latest_version();
            if head == 0 {
                return None;
            }
            // Version one creates the table and changes nothing, so the
            // oldest change a lake table can carry is at version two, and
            // the feed's own begin above the version it was turned on at
            let first = self.lake_first_version(table_id).saturating_add(1);
            return Some((2u64.max(first).min(head), head));
        }
        let feed = self.feeds.get_feed(table_id)?;
        match (feed.oldest_version(), feed.latest_version()) {
            (Some(oldest), Some(newest)) => Some((oldest, newest)),
            _ => None,
        }
    }

    fn purge_floor(&self, table_id: u32) -> u64 {
        if self.is_lake(table_id) {
            // A lake table's changes go exactly as far back as time travel
            // does: a position is stale when AS OF that version would fail,
            // so the floor is the oldest version the log can still stand at
            let Ok(log) = self.lake_log(table_id) else {
                return 0;
            };
            let head = log.latest_version();
            for version in 1..=head {
                if log.manifest_at(version).is_ok() {
                    return if version == 1 { 0 } else { version };
                }
            }
            return 0;
        }
        self.feeds
            .get_feed(table_id)
            .map(|feed| feed.purge_floor())
            .unwrap_or(0)
    }

    fn version_at_timestamp(&self, table_id: u32, timestamp: i64) -> u64 {
        if self.is_lake(table_id) {
            // The log dates each commit, and time travel already resolves an
            // instant to the last commit at or before it
            let Ok(log) = self.lake_log(table_id) else {
                return 0;
            };
            return zyron_lake::resolve_version(
                &log,
                zyron_lake::TimeTravelSpec::Timestamp(timestamp),
            )
            .unwrap_or(0);
        }
        let Some(feed) = self.feeds.get_feed(table_id) else {
            return 0;
        };
        let mut resolved = 0u64;
        let _ = feed.scan_range(&ChangeRange::timestamps(i64::MIN, timestamp), |record| {
            resolved = resolved.max(record.commit_version);
            Ok(())
        });
        resolved
    }

    fn rows_in_window(&self, table_id: u32, from_exclusive: u64, to_inclusive: u64) -> u64 {
        if self.is_lake(table_id) {
            // The feed's record index counts each commit in the window
            // once, whatever the window is asked about afterwards
            let Ok(source) = self.lake_source(table_id) else {
                return 0;
            };
            return match (
                source.records_at_or_below(to_inclusive),
                source.records_at_or_below(from_exclusive),
            ) {
                (Ok(to), Ok(from)) => to.saturating_sub(from),
                _ => 0,
            };
        }
        let Some(feed) = self.feeds.get_feed(table_id) else {
            return 0;
        };
        feed.records_at_or_below(to_inclusive)
            .saturating_sub(feed.records_at_or_below(from_exclusive))
    }

    fn files_for_window(
        &self,
        table_id: u32,
        from_exclusive: u64,
        to_inclusive: u64,
    ) -> (usize, usize) {
        if self.is_lake(table_id) {
            // A lake change read walks one version file per commit in the
            // window, and the commits outside it are what the bounds pruned
            let Ok(log) = self.lake_log(table_id) else {
                return (0, 0);
            };
            let head = log.latest_version();
            if head < 2 {
                return (0, 0);
            }
            let first = from_exclusive
                .max(self.lake_first_version(table_id))
                .saturating_add(1)
                .max(2);
            let last = to_inclusive.min(head);
            let opened = if first > last {
                0
            } else {
                (last - first + 1) as usize
            };
            let total = (head - 1) as usize;
            return (opened, total.saturating_sub(opened));
        }
        let Some(feed) = self.feeds.get_feed(table_id) else {
            return (0, 0);
        };
        let plan = feed.plan_read(&ChangeRange::versions(
            from_exclusive.saturating_add(1),
            to_inclusive,
        ));
        (plan.segments.len(), plan.pruned)
    }

    fn recorded_columns(&self, table_id: u32) -> Vec<u16> {
        let Ok(table) = self.catalog.get_table_by_id(TableId(table_id)) else {
            return Vec::new();
        };
        table.cdf.recorded_columns().iter().map(|id| id.0).collect()
    }

    fn before_image(&self, table_id: u32) -> bool {
        self.feeds
            .get_feed(table_id)
            .map(|feed| feed.config().before_image)
            .unwrap_or_else(|| {
                self.catalog
                    .get_table_by_id(TableId(table_id))
                    .map(|table| table.cdf.before_image)
                    .unwrap_or(true)
            })
    }
}

/// The bridges built on this node, one per data directory.
///
/// A bridge holds only references, so building one per statement would be an
/// allocation per statement for nothing. Keyed by the directory rather than
/// held once for the process, because a test binary builds more than one
/// server and each has its own catalog and its own feeds
static BRIDGES: std::sync::OnceLock<scc::HashMap<std::path::PathBuf, Arc<ChangeFeedBridge>>> =
    std::sync::OnceLock::new();

/// The bridge for one server, built on first use.
///
/// A bridge held for the directory is reused only while it reads the
/// server's own registry and catalog. A server opened again over the same
/// directory inside one process, which is what a restart in a test binary
/// is, has a registry of its own, and a bridge over the earlier one would
/// read feeds nothing writes to any more
///
/// None where CDC is not enabled, which is the same option every other CDC
/// path on the server tests
pub fn bridge_for(server: &crate::connection::ServerState) -> Option<Arc<ChangeFeedBridge>> {
    let feeds = server.cdc_registry.as_ref()?;
    let data_dir = server.disk_manager.data_dir().to_path_buf();
    let bridges = BRIDGES.get_or_init(scc::HashMap::new);
    if let Some(held) = bridges.read_sync(&data_dir, |_, bridge| Arc::clone(bridge)) {
        let same_branches = match (&held.branches, &server.branch_manager) {
            (Some(a), Some(b)) => Arc::ptr_eq(a, b),
            (None, None) => true,
            _ => false,
        };
        if Arc::ptr_eq(&held.feeds, feeds)
            && Arc::ptr_eq(&held.catalog, &server.catalog)
            && same_branches
        {
            return Some(held);
        }
    }
    let built = Arc::new(ChangeFeedBridge::new(
        Arc::clone(feeds),
        Arc::clone(&server.catalog),
        data_dir.clone(),
        server.branch_manager.clone(),
    ));
    let _ = bridges.upsert_sync(data_dir, Arc::clone(&built));
    Some(built)
}

/// Installs what a change read needs on an execution context.
///
/// The reader answers `table_changes` and a stream read, and the position
/// locks are what make a transactional consume exclusive. Both are set on
/// every context, so a change read inside a subquery or a trigger body works
/// exactly as one at the top level does
pub fn install_change_reads(
    server: &Arc<crate::connection::ServerState>,
    ctx: &mut zyron_executor::context::ExecutionContext,
    advances: &Arc<parking_lot::Mutex<Vec<zyron_executor::context::PendingStreamAdvance>>>,
) {
    if let Some(bridge) = bridge_for(server) {
        ctx.change_feed = Some(bridge as Arc<dyn ChangeFeedReader>);
    }
    ctx.stream_position_locks = Some(Arc::clone(server.txn_manager.stream_positions()));
    // The connection's own list, so an advance a statement records is still
    // there when the transaction it belongs to commits
    ctx.pending_stream_advances = Arc::clone(advances);
}

/// The change kinds a mask admits, by name, for a caller rendering one
pub fn kinds_of(mask: u8) -> Vec<&'static str> {
    zyron_common::CHANGE_TYPE_CODES
        .iter()
        .filter(|code| mask & zyron_common::change_type_bit(**code) != 0)
        .filter_map(|code| zyron_common::change_type_label(*code))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_a_resumed_read_starts_after_the_position_it_left() {
        assert!(ChangeFeedBridge::after_resume(None, 1, 0));
        assert!(!ChangeFeedBridge::after_resume(Some((5, 2)), 5, 2));
        assert!(!ChangeFeedBridge::after_resume(Some((5, 2)), 5, 1));
        assert!(!ChangeFeedBridge::after_resume(Some((5, 2)), 4, 9));
        assert!(ChangeFeedBridge::after_resume(Some((5, 2)), 5, 3));
        assert!(ChangeFeedBridge::after_resume(Some((5, 2)), 6, 0));
    }

    #[test]
    fn test_a_window_becomes_an_open_ended_feed_range() {
        let window = ChangeWindow {
            table_id: 1,
            branch: None,
            from_exclusive: 4,
            to_inclusive: 9,
            from_timestamp: i64::MIN,
            to_timestamp: i64::MAX,
            change_types: None,
        };
        let range = ChangeFeedBridge::range_of(&window);
        assert_eq!(
            range.start_version, 5,
            "the window is open at its lower end"
        );
        assert_eq!(range.end_version, 9);
    }

    #[test]
    fn test_an_empty_window_admits_nothing() {
        let window = ChangeWindow {
            table_id: 1,
            branch: None,
            from_exclusive: 9,
            to_inclusive: 9,
            from_timestamp: i64::MIN,
            to_timestamp: i64::MAX,
            change_types: None,
        };
        let range = ChangeFeedBridge::range_of(&window);
        assert!(range.end_version < range.start_version);
    }

    #[test]
    fn test_a_mask_names_the_kinds_it_admits() {
        let mask = zyron_common::change_type_bit(zyron_common::CHANGE_TYPE_INSERT)
            | zyron_common::change_type_bit(zyron_common::CHANGE_TYPE_DELETE);
        assert_eq!(kinds_of(mask), vec!["insert", "delete"]);
    }
}
