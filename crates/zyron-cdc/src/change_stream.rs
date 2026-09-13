//! A change stream is a named, durable position over one or more change feeds.
//!
//! A stream holds a commit version per source table. A read returns every
//! change above that position up to the reader's boundary, and a commit
//! records the version it read to. That is the whole model. The position is
//! catalog state, it moves in the consumer's own transaction, and a rollback
//! leaves it where it was.
//!
//! A change stream is neither a queue nor a Phase 17 stream. A queue hands a
//! message to one consumer and forgets it, and a stream re-reads a range as
//! long as the feed retains it. A Phase 17 stream carries watermarks and
//! subscribers over a live query, and a change stream carries a position over
//! a table's recorded changes

pub mod multi;
pub mod view;

use std::collections::HashMap;
use std::sync::Arc;

use zyron_catalog::{ChangeStreamEntry, ChangeStreamMode, ChangeStreamOrigin, ChangeStreamSource};
use zyron_common::{Result, ZyronError};

use crate::change_feed::{CdfRegistry, ChangeDataFeed, ChangeRange, ChangeType, ReadPlan};

/// Why a stream cannot be read
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum StaleReason {
    /// The source table's feed was turned off. The stream keeps its position
    /// so an operator can see what turning the feed off broke
    FeedDisabled { table_id: u32 },
    /// Retention reclaimed changes the position still names
    RetentionPassed {
        table_id: u32,
        position: u64,
        oldest_available: u64,
    },
    /// The source table is gone
    SourceDropped { table_id: u32 },
}

impl StaleReason {
    /// The word the catalog entry and the observability views record
    pub fn code(&self) -> &'static str {
        match self {
            StaleReason::FeedDisabled { .. } => "feed_disabled",
            StaleReason::RetentionPassed { .. } => "retention_passed",
            StaleReason::SourceDropped { .. } => "source_dropped",
        }
    }

    /// The sentence a refused read carries
    pub fn message(&self, stream: &str) -> String {
        match self {
            StaleReason::FeedDisabled { table_id } => format!(
                "ChangeStreamStale: change stream '{stream}' is stale, the change data feed on \
                 table {table_id} is disabled. Turn it back on with ALTER TABLE ... SET \
                 (change_data_feed = true), then move the stream with ALTER CHANGE STREAM \
                 {stream} RESET TO VERSION <n>"
            ),
            StaleReason::RetentionPassed {
                table_id,
                position,
                oldest_available,
            } => format!(
                "ChangeStreamStale: change stream '{stream}' is stale, its position on table \
                 {table_id} is version {position} and the oldest change still held is version \
                 {oldest_available}. Move it with ALTER CHANGE STREAM {stream} RESET TO VERSION \
                 {oldest_available}"
            ),
            StaleReason::SourceDropped { table_id } => format!(
                "ChangeStreamStale: change stream '{stream}' is stale, its source table \
                 {table_id} was dropped"
            ),
        }
    }
}

/// Why a stream needs looking at without having lost its place
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AttentionReason {
    /// The stream's COLUMNS list names a column the table no longer has
    DroppedColumn { column: String },
    /// A narrowing change went through with ACKNOWLEDGE STREAM BREAK
    AcknowledgedBreak { detail: String },
}

impl AttentionReason {
    pub fn code(&self) -> &'static str {
        match self {
            AttentionReason::DroppedColumn { .. } => "dropped_column",
            AttentionReason::AcknowledgedBreak { .. } => "acknowledged_break",
        }
    }

    pub fn message(&self, stream: &str) -> String {
        match self {
            AttentionReason::DroppedColumn { column } => format!(
                "change stream '{stream}' names column '{column}', which the source table no \
                 longer has. It keeps its position and yields again once ALTER CHANGE STREAM \
                 {stream} drops that column from its COLUMNS list"
            ),
            AttentionReason::AcknowledgedBreak { detail } => format!(
                "change stream '{stream}' reads a table whose type change was acknowledged as a \
                 stream break: {detail}"
            ),
        }
    }
}

/// The window one source table contributes to a read
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SourceWindow {
    pub table_id: u32,
    /// The branch whose feed is read, None for the table's own
    pub branch: Option<u64>,
    /// Changes above this version are returned
    pub from_exclusive: u64,
    /// Changes at or below this version are returned
    pub to_inclusive: u64,
    /// Feed records at or below `to_inclusive`, counted from the feed's
    /// creation. This is what a position advance replicates as
    pub consumed_to: u64,
}

impl SourceWindow {
    /// The feed range this window reads
    pub fn range(&self) -> ChangeRange {
        ChangeRange::versions(self.from_exclusive.saturating_add(1), self.to_inclusive)
    }

    /// True when the window covers no version at all
    pub fn is_empty(&self) -> bool {
        self.to_inclusive <= self.from_exclusive
    }
}

/// Everything a read of one stream needs, resolved before a file is opened
#[derive(Debug, Clone)]
pub struct StreamReadPlan {
    pub stream_id: u32,
    pub mode: ChangeStreamMode,
    /// One window per source table, ascending by table id
    pub windows: Vec<SourceWindow>,
    /// True when this read yields the source's existing rows as inserts
    /// rather than reading the feed
    pub initial_rows: bool,
    /// Change files the windows will open, summed over the sources
    pub files_opened: usize,
    /// Change files the windows pruned
    pub files_pruned: usize,
    /// Records the surviving files hold, an upper bound on the answer
    pub candidate_records: u64,
}

impl StreamReadPlan {
    /// The position the stream advances to when this read commits
    pub fn advance_to(&self) -> Vec<(u32, u64)> {
        self.windows
            .iter()
            .map(|w| (w.table_id, w.to_inclusive))
            .collect()
    }

    /// The version and the consumed record count each source advances to
    pub fn advance_positions(&self) -> Vec<(u32, u64, u64)> {
        self.windows
            .iter()
            .map(|w| (w.table_id, w.to_inclusive, w.consumed_to))
            .collect()
    }

    /// True when nothing at all is pending
    pub fn is_empty(&self) -> bool {
        !self.initial_rows && self.windows.iter().all(|w| w.is_empty())
    }

    /// The change kinds a read in this mode admits
    pub fn admitted_kinds(&self) -> Option<Vec<ChangeType>> {
        match self.mode {
            ChangeStreamMode::Standard => None,
            // An append-only stream skips updates and deletes at read time,
            // which is cheaper than reading and discarding them and is the
            // right semantics for a bronze layer
            ChangeStreamMode::AppendOnly => Some(vec![ChangeType::Insert]),
        }
    }
}

/// What a stream's observability row reports
#[derive(Debug, Clone)]
pub struct StreamStatus {
    pub stream_id: u32,
    pub name: String,
    pub sources: Vec<u32>,
    pub position: Vec<(u32, u64)>,
    pub mode: ChangeStreamMode,
    pub stale: bool,
    pub stale_reason: String,
    pub needs_attention: bool,
    pub attention_reason: String,
    /// Records above the position, summed over the sources and read from the
    /// feeds' counters rather than by scanning any change file
    pub pending_rows: u64,
    pub pending_versions: u64,
    /// Age in seconds of the oldest unconsumed change, zero when none pends
    pub lag_seconds: i64,
    pub last_advanced_at: i64,
    pub last_advanced_by: u32,
    pub owner_id: u32,
}

// ---------------------------------------------------------------------------
// ChangeStreamRuntime
// ---------------------------------------------------------------------------

/// Resolves stream reads, staleness and lag against the feeds on this node.
///
/// Holds no state of its own. The stream's definition and position live in the
/// catalog and its changes live in the feeds, so this is the logic between
/// them rather than a third copy of either
pub struct ChangeStreamRuntime {
    feeds: Arc<CdfRegistry>,
}

impl ChangeStreamRuntime {
    pub fn new(feeds: Arc<CdfRegistry>) -> Self {
        Self { feeds }
    }

    /// The feed registry this runtime reads
    pub fn feeds(&self) -> &Arc<CdfRegistry> {
        &self.feeds
    }

    /// The feed for one source table, or an error naming the ALTER that turns
    /// it on
    pub fn feed_of(&self, table_id: u32) -> Result<Arc<ChangeDataFeed>> {
        self.feed_on(table_id, None)
    }

    /// The feed a source names, the table's own or a branch's on it, or an
    /// error naming the ALTER that turns the table's on
    pub fn feed_on(&self, table_id: u32, branch: Option<u64>) -> Result<Arc<ChangeDataFeed>> {
        match self.feeds.feed_on(table_id, branch) {
            Some(feed) if feed.is_enabled() => Ok(feed),
            Some(_) | None => Err(ZyronError::CdcStreamError(format!(
                "table {table_id} has no change data feed. Enable it with \
                 ALTER TABLE <table> SET (change_data_feed = true)"
            ))),
        }
    }

    /// The feed a stream reads on one of its sources, the branch's when the
    /// stream was created on a branch, the table's own otherwise
    pub fn source_feed(
        &self,
        entry: &ChangeStreamEntry,
        table_id: u32,
    ) -> Option<Arc<ChangeDataFeed>> {
        self.feeds.feed_on(table_id, entry.branch)
    }

    /// Decides whether a stream can be read, and why not when it cannot.
    ///
    /// Checked at read and by the background sweeper, so staleness surfaces
    /// before someone waits on it rather than at the moment they do
    pub fn staleness(&self, entry: &ChangeStreamEntry) -> Option<StaleReason> {
        for table_id in entry.source.table_ids() {
            let position = entry.position_of(table_id);
            let Some(feed) = self.source_feed(entry, table_id) else {
                // A source whose changes come from its own store is stale
                // exactly when that store can no longer stand at the
                // position, which for a lake table is when time travel to
                // it fails
                if let Some(derived) = self.feeds.derived_on(table_id, entry.branch) {
                    // A position is exclusive, so one standing just below
                    // the oldest version still reads that version
                    match derived.oldest_readable_version() {
                        Some(oldest) if position.saturating_add(1) < oldest => {
                            return Some(StaleReason::RetentionPassed {
                                table_id,
                                position,
                                oldest_available: oldest,
                            });
                        }
                        _ => continue,
                    }
                }
                return Some(StaleReason::SourceDropped { table_id });
            };
            if !feed.is_enabled() {
                return Some(StaleReason::FeedDisabled { table_id });
            }
            // An append-only stream never reads an update or a delete, so
            // retention reclaiming them takes nothing it would have yielded
            if entry.mode == ChangeStreamMode::AppendOnly {
                continue;
            }
            let floor = feed.purge_floor();
            if floor > 0 && position < floor {
                return Some(StaleReason::RetentionPassed {
                    table_id,
                    position,
                    oldest_available: feed.oldest_version().unwrap_or(floor.saturating_add(1)),
                });
            }
        }
        None
    }

    /// The version each source reads to, which is its own newest change.
    ///
    /// A transaction that touched a fact and its dimension is never returned
    /// by halves because a reader holds every source below the first change
    /// of any transaction that has not ended, see [`multi::boundary_below`].
    /// A source that has stopped changing reads to its own newest rather
    /// than holding the others back
    pub fn boundary(&self, entry: &ChangeStreamEntry) -> Result<HashMap<u32, u64>> {
        multi::boundary(self, entry)
    }

    /// The boundary with open transactions held out. `ended` says whether a
    /// transaction is over
    pub fn boundary_below(
        &self,
        entry: &ChangeStreamEntry,
        ended: &dyn Fn(u64) -> bool,
    ) -> Result<HashMap<u32, u64>> {
        multi::boundary_below(self, entry, ended)
    }

    /// Resolves what one read will return.
    ///
    /// `snapshot_bound` caps each source at the version the reader's
    /// transaction may see, so a consume never returns a change its own
    /// snapshot could not
    pub fn plan_read(
        &self,
        entry: &ChangeStreamEntry,
        snapshot_bound: Option<u64>,
    ) -> Result<StreamReadPlan> {
        if let Some(reason) = self.staleness(entry) {
            return Err(ZyronError::CdcStreamError(reason.message(&entry.name)));
        }

        let initial_rows = entry.created_from == ChangeStreamOrigin::InitialRows
            && entry.position.iter().all(|p| p.version == 0);

        let boundary = self.boundary(entry)?;
        let kinds = match entry.mode {
            ChangeStreamMode::Standard => None,
            ChangeStreamMode::AppendOnly => Some(vec![ChangeType::Insert]),
        };

        let mut windows = Vec::new();
        let mut files_opened = 0usize;
        let mut files_pruned = 0usize;
        let mut candidate_records = 0u64;
        for table_id in entry.source.table_ids() {
            let mut to = boundary.get(&table_id).copied().unwrap_or(0);
            if let Some(cap) = snapshot_bound {
                to = to.min(cap);
            }
            let from = entry.position_of(table_id);
            let to_inclusive = to.max(from);
            let consumed_to = match self.source_feed(entry, table_id) {
                Some(feed) => feed.records_at_or_below(to_inclusive),
                None => entry.consumed_of(table_id),
            };
            let window = SourceWindow {
                table_id,
                branch: entry.branch,
                from_exclusive: from,
                to_inclusive,
                consumed_to,
            };
            if !window.is_empty() {
                let feed = self.feed_on(table_id, entry.branch)?;
                let mut range = window.range();
                if let Some(kinds) = &kinds {
                    range = range.with_change_types(kinds);
                }
                let ReadPlan {
                    segments,
                    pruned,
                    candidate_records: candidates,
                } = feed.plan_read(&range);
                files_opened += segments.len();
                files_pruned += pruned;
                candidate_records += candidates;
            }
            windows.push(window);
        }
        windows.sort_by_key(|w| w.table_id);

        Ok(StreamReadPlan {
            stream_id: entry.id,
            mode: entry.mode,
            windows,
            initial_rows,
            files_opened,
            files_pruned,
            candidate_records,
        })
    }

    /// Reads one stream's changes, in (version, ordinal) order per source.
    ///
    /// The records come back paired with the table they came from, which is
    /// what a multi-table stream renders as `_source_table`
    pub fn read(&self, plan: &StreamReadPlan) -> Result<Vec<(u32, crate::ChangeRecord)>> {
        let kinds = plan.admitted_kinds();
        let mut out = Vec::new();
        for window in &plan.windows {
            if window.is_empty() {
                continue;
            }
            let feed = self.feed_on(window.table_id, window.branch)?;
            let mut range = window.range();
            if let Some(kinds) = &kinds {
                range = range.with_change_types(kinds);
            }
            let table_id = window.table_id;
            feed.read_range_into(&range, |record| {
                out.push((table_id, record));
                Ok(())
            })?;
        }
        // One consistent order across sources, the commit version first, then
        // the position inside that commit, then the table so a tie is stable
        out.sort_by(|a, b| {
            a.1.commit_version
                .cmp(&b.1.commit_version)
                .then(a.1.change_ordinal.cmp(&b.1.change_ordinal))
                .then(a.0.cmp(&b.0))
        });
        Ok(out)
    }

    /// Records above a stream's position, from the feeds' counters
    pub fn pending_rows(&self, entry: &ChangeStreamEntry) -> u64 {
        entry
            .source
            .table_ids()
            .into_iter()
            .filter_map(|table_id| {
                let position = entry.position_of(table_id);
                match self.source_feed(entry, table_id) {
                    Some(feed) => Some(feed.pending_after(position)),
                    None => self
                        .feeds
                        .derived_on(table_id, entry.branch)
                        .map(|derived| derived.pending_after(position)),
                }
            })
            .sum()
    }

    /// Commit versions above a stream's position, from the feeds' counters
    pub fn pending_versions(&self, entry: &ChangeStreamEntry) -> u64 {
        entry
            .source
            .table_ids()
            .into_iter()
            .filter_map(|table_id| {
                let position = entry.position_of(table_id);
                match self.source_feed(entry, table_id) {
                    Some(feed) => Some(feed.pending_versions_after(position)),
                    None => self
                        .feeds
                        .derived_on(table_id, entry.branch)
                        .map(|derived| derived.latest_version().saturating_sub(position)),
                }
            })
            .sum()
    }

    /// Age in seconds of the oldest change the stream has not consumed
    pub fn lag_seconds(&self, entry: &ChangeStreamEntry, now_micros: i64) -> i64 {
        let oldest = entry
            .source
            .table_ids()
            .into_iter()
            .filter_map(|table_id| {
                let position = entry.position_of(table_id);
                match self.source_feed(entry, table_id) {
                    Some(feed) => feed.first_timestamp_after(position),
                    None => self
                        .feeds
                        .derived_on(table_id, entry.branch)
                        .and_then(|derived| derived.first_timestamp_after(position)),
                }
            })
            .min();
        match oldest {
            Some(ts) => now_micros.saturating_sub(ts).max(0) / 1_000_000,
            None => 0,
        }
    }

    /// The observability row for one stream
    pub fn status(&self, entry: &ChangeStreamEntry, now_micros: i64) -> StreamStatus {
        StreamStatus {
            stream_id: entry.id,
            name: entry.name.clone(),
            sources: entry.source.table_ids(),
            position: entry
                .position
                .iter()
                .map(|p| (p.table_id, p.version))
                .collect(),
            mode: entry.mode,
            stale: entry.stale,
            stale_reason: entry.stale_reason.clone(),
            needs_attention: entry.needs_attention,
            attention_reason: entry.attention_reason.clone(),
            pending_rows: self.pending_rows(entry),
            pending_versions: self.pending_versions(entry),
            lag_seconds: self.lag_seconds(entry, now_micros),
            last_advanced_at: entry.last_advanced_at,
            last_advanced_by: entry.last_advanced_by,
            owner_id: entry.owner_id,
        }
    }

    /// Marks the streams a sweep found stale, answering with the entries that
    /// changed so the caller makes them durable.
    ///
    /// The compare is one position against one purge floor per source, so a
    /// sweep over ten thousand streams reads counters and never a change file
    pub fn sweep(&self, entries: &[Arc<ChangeStreamEntry>]) -> Vec<ChangeStreamEntry> {
        let mut changed = Vec::new();
        for entry in entries {
            match self.staleness(entry) {
                Some(reason) => {
                    if entry.stale && entry.stale_reason == reason.code() {
                        continue;
                    }
                    let mut updated = ChangeStreamEntry::clone(entry);
                    updated.stale = true;
                    updated.stale_reason = reason.code().to_string();
                    changed.push(updated);
                }
                None => {
                    if !entry.stale {
                        continue;
                    }
                    let mut updated = ChangeStreamEntry::clone(entry);
                    updated.stale = false;
                    updated.stale_reason.clear();
                    changed.push(updated);
                }
            }
        }
        changed
    }
}

// ---------------------------------------------------------------------------
// Position moves
// ---------------------------------------------------------------------------

/// Builds the entry a committed read advances to
pub fn advanced(
    entry: &ChangeStreamEntry,
    plan: &StreamReadPlan,
    by_role: u32,
    at_micros: i64,
) -> ChangeStreamEntry {
    let mut next = entry.clone();
    for (table_id, version, consumed) in plan.advance_positions() {
        next.set_position(table_id, version, consumed);
    }
    next.last_advanced_at = at_micros;
    next.last_advanced_by = by_role;
    next
}

/// Re-addresses a replicated position in this member's own feed versions.
///
/// The consumed count is the same number on every member, and each member's
/// version index turns it back into the version that names that place here.
/// A member with no feed for a source keeps the count and leaves the version
/// alone, which is what a follower that has not opened the feed yet holds
pub fn localize_positions(runtime: &ChangeStreamRuntime, entry: &mut ChangeStreamEntry) {
    let branch = entry.branch;
    for slot in entry.position.iter_mut() {
        if let Some(feed) = runtime.feeds().feed_on(slot.table_id, branch) {
            slot.version = feed.version_at_count(slot.consumed);
        } else if let Some(derived) = runtime.feeds().derived_on(slot.table_id, branch) {
            slot.version = derived.version_at_count(slot.consumed);
        }
    }
}

/// Builds the entry an `ALTER CHANGE STREAM ... RESET` produces.
///
/// A target outside what the sources still hold is refused rather than
/// silently clamped, because a position that skipped changes yields a target
/// that is quietly wrong
pub fn reset_to(
    runtime: &ChangeStreamRuntime,
    entry: &ChangeStreamEntry,
    target: ResetTarget,
    by_role: u32,
    at_micros: i64,
) -> Result<ChangeStreamEntry> {
    let mut next = entry.clone();
    // A reset names where the stream reads from, so the existing rows a
    // SHOW INITIAL ROWS stream had yet to yield are not what it yields next
    next.initial_rows_pending = false;
    for table_id in entry.source.table_ids() {
        if runtime.feeds().feed_on(table_id, entry.branch).is_none() {
            if let Some(derived) = runtime.feeds().derived_on(table_id, entry.branch) {
                let oldest = derived.oldest_readable_version().unwrap_or(0);
                let latest = derived.latest_version();
                // A position is exclusive, so the earliest one stands just
                // below the oldest version and reads it, the way a heap
                // feed's does
                let version = match target {
                    ResetTarget::Version(v) => v,
                    ResetTarget::Timestamp(ts) => derived.version_at_timestamp(ts),
                    ResetTarget::Earliest => oldest.saturating_sub(1),
                    ResetTarget::Latest => latest,
                };
                if version.saturating_add(1) < oldest {
                    return Err(ZyronError::CdcStreamError(format!(
                        "cannot reset change stream '{}' to version {version} on table \
                         {table_id}: the oldest version its log still stands at is {oldest}. \
                         Retention reclaimed everything below it",
                        entry.name
                    )));
                }
                next.set_position(table_id, version, derived.records_at_or_below(version)?);
                continue;
            }
        }
        let feed = runtime.feed_on(table_id, entry.branch)?;
        let version = match target {
            ResetTarget::Version(v) => v,
            ResetTarget::Timestamp(ts) => {
                // The last version at or before the timestamp, so a read from
                // here returns every change committed after that instant
                feed.version_at_or_before(ts)
            }
            ResetTarget::Earliest => feed.oldest_version().unwrap_or(0).saturating_sub(1),
            ResetTarget::Latest => feed.latest_version().unwrap_or(0),
        };
        let floor = feed.purge_floor();
        if floor > 0 && version < floor {
            let oldest = feed.oldest_version().unwrap_or(floor.saturating_add(1));
            return Err(ZyronError::CdcStreamError(format!(
                "cannot reset change stream '{}' to version {version} on table {table_id}: the \
                 oldest change still held is version {oldest}. Retention reclaimed everything \
                 below it",
                entry.name
            )));
        }
        let latest = feed.latest_version().unwrap_or(0);
        if version > latest {
            return Err(ZyronError::CdcStreamError(format!(
                "cannot reset change stream '{}' to version {version} on table {table_id}: the \
                 newest change held is version {latest}",
                entry.name
            )));
        }
        next.set_position(table_id, version, feed.records_at_or_below(version));
    }
    next.stale = false;
    next.stale_reason.clear();
    next.last_advanced_at = at_micros;
    next.last_advanced_by = by_role;
    Ok(next)
}

/// Where a reset moves a stream to
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ResetTarget {
    Version(u64),
    Timestamp(i64),
    /// The oldest change the sources still hold
    Earliest,
    /// The newest change the sources hold, so the next read yields nothing
    Latest,
}

/// The position a newly created stream starts at, per source table, as a
/// version paired with the record count at or below it
pub fn initial_position(
    runtime: &ChangeStreamRuntime,
    source: &ChangeStreamSource,
    origin: ChangeStreamOrigin,
    branch: Option<u64>,
) -> Result<Vec<(u32, u64, u64)>> {
    let mut out = Vec::new();
    for table_id in source.table_ids() {
        if runtime.feeds().feed_on(table_id, branch).is_none() {
            if let Some(derived) = runtime.feeds().derived_on(table_id, branch) {
                let version = match origin {
                    ChangeStreamOrigin::Now => derived.latest_version(),
                    ChangeStreamOrigin::Version(v) => v,
                    ChangeStreamOrigin::Timestamp(ts) => derived.version_at_timestamp(ts),
                    ChangeStreamOrigin::InitialRows => 0,
                };
                out.push((table_id, version, derived.records_at_or_below(version)?));
                continue;
            }
        }
        let version = match origin {
            // The source's current version, so a stream created now yields
            // nothing until the next change. On a branch that has recorded
            // nothing yet, that is the version the branch was taken at
            ChangeStreamOrigin::Now => runtime
                .feeds
                .feed_on(table_id, branch)
                .and_then(|feed| feed.latest_version().or(Some(feed.branch_point())))
                .unwrap_or(0),
            ChangeStreamOrigin::Version(v) => v,
            ChangeStreamOrigin::Timestamp(ts) => {
                runtime.feed_on(table_id, branch)?.version_at_or_before(ts)
            }
            // The first read yields the table's existing rows, so the feed
            // position starts at zero and moves to the source's current
            // version once that read commits
            ChangeStreamOrigin::InitialRows => 0,
        };
        let consumed = runtime
            .feeds
            .feed_on(table_id, branch)
            .map(|feed| feed.records_at_or_below(version))
            .unwrap_or(0);
        out.push((table_id, version, consumed));
    }
    Ok(out)
}

// ---------------------------------------------------------------------------
// Alert templates
// ---------------------------------------------------------------------------

/// One alert this phase declares, for the collector to pick up.
///
/// Declared here and dispatched through the notification channels the server
/// already holds. Nothing collects them on a schedule yet, which is what the
/// alert collector adds
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AlertTemplate {
    /// The name a subscription names
    pub name: &'static str,
    /// The condition the template fires on
    pub condition: &'static str,
    /// The setting whose value the condition compares against, empty when the
    /// condition takes no threshold
    pub threshold_setting: &'static str,
    /// The value the threshold takes when nothing sets it
    pub default_threshold: &'static str,
    pub summary: &'static str,
}

/// Every alert the change feed and change stream surfaces declare
pub const ALERT_TEMPLATES: &[AlertTemplate] = &[
    AlertTemplate {
        name: "cdc_stream_lag",
        condition: "pending_rows or lag_seconds over the threshold",
        threshold_setting: "cdc_stream_lag_rows",
        default_threshold: "1000000",
        summary: "A change stream is falling behind its source",
    },
    AlertTemplate {
        name: "cdc_stream_stale",
        condition: "a stream's position names changes the feed no longer holds",
        threshold_setting: "",
        default_threshold: "",
        summary: "A change stream cannot be read until it is reset",
    },
    AlertTemplate {
        name: "cdc_stream_needs_attention",
        condition: "a stream names a column its source no longer has",
        threshold_setting: "",
        default_threshold: "",
        summary: "A change stream yields nothing until its definition is corrected",
    },
    AlertTemplate {
        name: "cdf_retention_pressure",
        condition: "the oldest change is within the margin of a stream's position",
        threshold_setting: "cdf_retention_margin",
        default_threshold: "1h",
        summary: "Retention is about to reclaim changes a stream has not consumed",
    },
    AlertTemplate {
        name: "cdc_apply_failed",
        condition: "an APPLY CHANGES run ended with an error",
        threshold_setting: "",
        default_threshold: "",
        summary: "A declarative apply did not reach its target",
    },
];

/// One alert instance an evaluation produced
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AlertFiring {
    pub template: &'static str,
    pub stream_id: u32,
    pub stream: String,
    pub detail: String,
}

/// Decides which alerts a stream's state fires.
///
/// The lag reading comes from the feed counters the status row already
/// carries, so evaluating every stream on a node opens no change file
pub fn evaluate_alerts(
    status: &StreamStatus,
    lag_rows_threshold: u64,
    lag_seconds_threshold: i64,
) -> Vec<AlertFiring> {
    let mut out = Vec::new();
    if status.stale {
        out.push(AlertFiring {
            template: "cdc_stream_stale",
            stream_id: status.stream_id,
            stream: status.name.clone(),
            detail: status.stale_reason.clone(),
        });
    }
    if status.needs_attention {
        out.push(AlertFiring {
            template: "cdc_stream_needs_attention",
            stream_id: status.stream_id,
            stream: status.name.clone(),
            detail: status.attention_reason.clone(),
        });
    }
    if status.pending_rows > lag_rows_threshold {
        out.push(AlertFiring {
            template: "cdc_stream_lag",
            stream_id: status.stream_id,
            stream: status.name.clone(),
            detail: format!("{} rows pending", status.pending_rows),
        });
    } else if status.lag_seconds > lag_seconds_threshold {
        out.push(AlertFiring {
            template: "cdc_stream_lag",
            stream_id: status.stream_id,
            stream: status.name.clone(),
            detail: format!("{} seconds behind", status.lag_seconds),
        });
    }
    out
}

/// Decides whether a feed is close enough to reclaiming a stream's unconsumed
/// changes to raise retention pressure
pub fn retention_pressure(
    runtime: &ChangeStreamRuntime,
    entry: &ChangeStreamEntry,
    margin_micros: i64,
    now_micros: i64,
) -> Option<AlertFiring> {
    for table_id in entry.source.table_ids() {
        let Some(feed) = runtime.source_feed(entry, table_id) else {
            continue;
        };
        let position = entry.position_of(table_id);
        // A feed a byte cap has already purged takes its oldest changes
        // next, so a stream whose unconsumed changes begin at the oldest
        // one held is one purge from losing them
        if feed.cap_purges() > 0
            && feed
                .oldest_version()
                .is_some_and(|oldest| position < oldest)
        {
            return Some(AlertFiring {
                template: "cdf_retention_pressure",
                stream_id: entry.id,
                stream: entry.name.clone(),
                detail: format!(
                    "table {table_id} is at its byte cap and the next purge takes the oldest \
                     changes this stream has not consumed"
                ),
            });
        }
        let config = feed.config();
        if config.retention_micros <= 0 {
            continue;
        }
        let Some(oldest) = feed.first_timestamp_after(position) else {
            continue;
        };
        let expires_at = oldest.saturating_add(config.retention_micros);
        if expires_at.saturating_sub(now_micros) <= margin_micros {
            return Some(AlertFiring {
                template: "cdf_retention_pressure",
                stream_id: entry.id,
                stream: entry.name.clone(),
                detail: format!(
                    "the oldest change table {table_id} holds for this stream expires within the \
                     configured margin"
                ),
            });
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::change_feed::{ChangeRecord, FeedConfig};
    use tempfile::TempDir;
    use zyron_catalog::{DatabaseId, SchemaId, StreamPosition};

    fn entry(
        name: &str,
        source: ChangeStreamSource,
        positions: &[(u32, u64)],
    ) -> ChangeStreamEntry {
        ChangeStreamEntry {
            id: 1,
            catalog_id: DatabaseId(1),
            schema_id: SchemaId(1),
            name: name.to_string(),
            source,
            position: positions
                .iter()
                .map(|(table_id, version)| StreamPosition {
                    table_id: *table_id,
                    version: *version,
                    consumed: 0,
                })
                .collect(),
            created_at: 0,
            created_from: ChangeStreamOrigin::Now,
            mode: ChangeStreamMode::Standard,
            predicate: None,
            columns: None,
            owner_id: 1,
            last_advanced_at: 0,
            last_advanced_by: 0,
            stale: false,
            stale_reason: String::new(),
            needs_attention: false,
            attention_reason: String::new(),
            initial_rows_pending: false,
            branch: None,
        }
    }

    fn record(table_id: u32, version: u64, kind: ChangeType) -> ChangeRecord {
        ChangeRecord {
            change_type: kind,
            commit_version: version,
            commit_timestamp: version as i64 * 1_000_000,
            table_id,
            txn_id: version,
            change_ordinal: 0,
            schema_version: 1,
            row_data: vec![1, 2, 3],
            primary_key_data: Vec::new(),
            is_last_in_txn: true,
            projected: false,
        }
    }

    fn runtime(tmp: &TempDir) -> ChangeStreamRuntime {
        ChangeStreamRuntime::new(Arc::new(CdfRegistry::new(tmp.path().to_path_buf())))
    }

    #[test]
    fn test_a_stream_created_now_yields_nothing_until_the_next_change() {
        let tmp = TempDir::new().expect("temp dir");
        let rt = runtime(&tmp);
        let feed = rt.feeds().enable_for_table(1, 7).expect("enables");
        feed.append_batch(&[record(1, 5, ChangeType::Insert)])
            .expect("appends");

        let position = initial_position(
            &rt,
            &ChangeStreamSource::Table(1),
            ChangeStreamOrigin::Now,
            None,
        )
        .expect("resolves");
        assert_eq!(position, vec![(1, 5, 1)]);

        let starting: Vec<(u32, u64)> = position.iter().map(|(t, v, _)| (*t, *v)).collect();
        let stream = entry("s", ChangeStreamSource::Table(1), &starting);
        let plan = rt.plan_read(&stream, None).expect("plans");
        assert!(plan.is_empty());

        feed.append_batch(&[record(1, 6, ChangeType::Insert)])
            .expect("appends");
        let plan = rt.plan_read(&stream, None).expect("plans");
        assert!(!plan.is_empty());
        assert_eq!(plan.advance_to(), vec![(1, 6)]);
        assert_eq!(rt.read(&plan).expect("reads").len(), 1);
    }

    #[test]
    fn test_append_only_skips_updates_and_deletes() {
        let tmp = TempDir::new().expect("temp dir");
        let rt = runtime(&tmp);
        let feed = rt.feeds().enable_for_table(2, 7).expect("enables");
        feed.append_batch(&[
            record(2, 1, ChangeType::Insert),
            record(2, 2, ChangeType::UpdatePreimage),
            record(2, 2, ChangeType::UpdatePostimage),
            record(2, 3, ChangeType::Delete),
            record(2, 4, ChangeType::Insert),
        ])
        .expect("appends");

        let mut stream = entry("s", ChangeStreamSource::Table(2), &[(2, 0)]);
        stream.mode = ChangeStreamMode::AppendOnly;
        let plan = rt.plan_read(&stream, None).expect("plans");
        let rows = rt.read(&plan).expect("reads");
        assert_eq!(rows.len(), 2);
        assert!(
            rows.iter()
                .all(|(_, r)| r.change_type == ChangeType::Insert)
        );
    }

    #[test]
    fn test_retention_past_the_position_is_stale_and_names_both_versions() {
        let tmp = TempDir::new().expect("temp dir");
        let rt = runtime(&tmp);
        let feed = rt.feeds().enable_for_table(3, 7).expect("enables");
        for v in 1..=20u64 {
            feed.append_batch(&[record(3, v, ChangeType::Insert)])
                .expect("appends");
        }
        let stream = entry("silver", ChangeStreamSource::Table(3), &[(3, 2)]);
        assert!(rt.staleness(&stream).is_none());

        feed.purge_before_version(10).expect("purges");
        let reason = rt.staleness(&stream).expect("stale");
        let text = reason.message("silver");
        assert!(text.contains("version 2"), "{text}");
        assert!(text.contains("version 10"), "{text}");
        assert!(text.contains("RESET TO VERSION"), "{text}");
        assert!(rt.plan_read(&stream, None).is_err());
    }

    #[test]
    fn test_an_append_only_stream_survives_a_purge_of_updates() {
        let tmp = TempDir::new().expect("temp dir");
        let rt = runtime(&tmp);
        let feed = rt.feeds().enable_for_table(4, 7).expect("enables");
        for v in 1..=20u64 {
            feed.append_batch(&[record(4, v, ChangeType::Insert)])
                .expect("appends");
        }
        let mut stream = entry("bronze", ChangeStreamSource::Table(4), &[(4, 2)]);
        stream.mode = ChangeStreamMode::AppendOnly;
        feed.purge_before_version(10).expect("purges");
        assert!(rt.staleness(&stream).is_none());
    }

    #[test]
    fn test_a_disabled_feed_makes_a_stream_stale_without_dropping_it() {
        let tmp = TempDir::new().expect("temp dir");
        let rt = runtime(&tmp);
        let feed = rt.feeds().enable_for_table(5, 7).expect("enables");
        feed.append_batch(&[record(5, 1, ChangeType::Insert)])
            .expect("appends");
        let stream = entry("s", ChangeStreamSource::Table(5), &[(5, 0)]);
        feed.disable();
        let reason = rt.staleness(&stream).expect("stale");
        assert_eq!(reason.code(), "feed_disabled");
        assert!(reason.message("s").contains("change_data_feed = true"));
    }

    #[test]
    fn test_reset_outside_retention_is_refused() {
        let tmp = TempDir::new().expect("temp dir");
        let rt = runtime(&tmp);
        let feed = rt.feeds().enable_for_table(6, 7).expect("enables");
        for v in 1..=20u64 {
            feed.append_batch(&[record(6, v, ChangeType::Insert)])
                .expect("appends");
        }
        feed.purge_before_version(10).expect("purges");
        let stream = entry("s", ChangeStreamSource::Table(6), &[(6, 12)]);

        let refused = reset_to(&rt, &stream, ResetTarget::Version(3), 1, 0)
            .expect_err("a reset below the oldest change is refused");
        assert!(refused.to_string().contains("oldest change still held"));

        let moved = reset_to(&rt, &stream, ResetTarget::Version(11), 1, 42).expect("resets");
        assert_eq!(moved.position_of(6), 11);
        assert!(!moved.stale);
        assert_eq!(moved.last_advanced_at, 42);
    }

    #[test]
    fn test_pending_and_lag_read_the_counters() {
        let tmp = TempDir::new().expect("temp dir");
        let rt = runtime(&tmp);
        let feed = rt
            .feeds()
            .enable_with_config(7, FeedConfig::default())
            .expect("enables");
        for v in 1..=50u64 {
            feed.append_batch(&[record(7, v, ChangeType::Insert)])
                .expect("appends");
        }
        let stream = entry("s", ChangeStreamSource::Table(7), &[(7, 20)]);
        assert_eq!(rt.pending_rows(&stream), 30);
        assert_eq!(rt.pending_versions(&stream), 30);
        // Version 21's timestamp is 21 seconds, and the clock reads 121
        assert_eq!(rt.lag_seconds(&stream, 121_000_000), 100);
    }

    #[test]
    fn test_the_sweeper_flags_a_stream_before_a_reader_arrives() {
        let tmp = TempDir::new().expect("temp dir");
        let rt = runtime(&tmp);
        let feed = rt.feeds().enable_for_table(8, 7).expect("enables");
        for v in 1..=20u64 {
            feed.append_batch(&[record(8, v, ChangeType::Insert)])
                .expect("appends");
        }
        let stream = Arc::new(entry("s", ChangeStreamSource::Table(8), &[(8, 2)]));
        assert!(rt.sweep(&[Arc::clone(&stream)]).is_empty());

        feed.purge_before_version(10).expect("purges");
        let flagged = rt.sweep(&[stream]);
        assert_eq!(flagged.len(), 1);
        assert!(flagged[0].stale);
        assert_eq!(flagged[0].stale_reason, "retention_passed");
    }

    #[test]
    fn test_alert_templates_fire_on_their_conditions() {
        let tmp = TempDir::new().expect("temp dir");
        let rt = runtime(&tmp);
        let feed = rt.feeds().enable_for_table(9, 7).expect("enables");
        for v in 1..=10u64 {
            feed.append_batch(&[record(9, v, ChangeType::Insert)])
                .expect("appends");
        }
        let mut stream = entry("s", ChangeStreamSource::Table(9), &[(9, 0)]);
        let status = rt.status(&stream, 1_000_000_000);
        let fired = evaluate_alerts(&status, 5, 10_000);
        assert_eq!(fired.len(), 1);
        assert_eq!(fired[0].template, "cdc_stream_lag");

        stream.stale = true;
        stream.stale_reason = "retention_passed".into();
        stream.needs_attention = true;
        stream.attention_reason = "column 'x' was dropped".into();
        let status = rt.status(&stream, 1_000_000_000);
        let fired = evaluate_alerts(&status, u64::MAX, i64::MAX);
        let names: Vec<&str> = fired.iter().map(|f| f.template).collect();
        assert!(names.contains(&"cdc_stream_stale"));
        assert!(names.contains(&"cdc_stream_needs_attention"));
    }

    #[test]
    fn test_every_declared_template_has_a_summary() {
        assert_eq!(ALERT_TEMPLATES.len(), 5);
        for template in ALERT_TEMPLATES {
            assert!(!template.name.is_empty());
            assert!(!template.condition.is_empty());
            assert!(!template.summary.is_empty());
        }
    }
}
