//! Post-upgrade migration.
//!
//! Three kinds of state move after a node comes up on a new binary. Files
//! move according to their format's policy: eager sweeps now, lazy waits for
//! the file's next write, coexist never moves. Catalog rows move through the
//! schema evolution registry, one transaction per table so a partial failure
//! rolls that table back and leaves the rest alone. User-authored objects
//! move through the rewriter, under the tenant's policy.
//!
//! Everything here is budgeted and reports progress, because a sweep over a
//! large table is a thing an operator has to be able to watch and pause

use std::path::{Path, PathBuf};
use std::sync::Arc;

use zyron_common::format::migration::{MigrationBoard, MigrationProgress};
use zyron_common::format::rewrite::{RewriteStatus, UserObjectRewritePolicy};
use zyron_common::format::{
    CatalogSchemaRegistry, FormatKind, FormatRegistry, FormatVersion, MigrationPolicy,
    RewriteRecord, UpgradeBoard, envelope, migration,
};
use zyron_common::{Result, ZyronError};
use zyron_parser::rewriter;

use super::compat_gate::UserObject;

/// What a sweep is allowed to spend
#[derive(Debug, Clone, Copy)]
pub struct MigrationBudget {
    /// Seconds one format's sweep may run for
    pub time_secs: u64,
    /// Multiple of a file's size the sweep may occupy on disk while it works
    pub disk_multiple: f64,
    /// Fraction of node memory the sweep may hold
    pub memory_fraction: f64,
    /// Bytes free on the volume the sweep writes to, zero when unmeasured.
    /// A file whose size times the disk multiple exceeds this stops the
    /// sweep
    pub disk_free_bytes: u64,
    /// Bytes of memory on the node, zero when unmeasured. A file larger
    /// than the memory fraction of this stops the sweep, because a migration
    /// holds the whole file
    pub node_memory_bytes: u64,
}

impl Default for MigrationBudget {
    fn default() -> Self {
        Self {
            time_secs: 6 * 3_600,
            disk_multiple: 2.0,
            memory_fraction: 0.25,
            disk_free_bytes: 0,
            node_memory_bytes: 0,
        }
    }
}

impl MigrationBudget {
    /// Whether one file of this size fits the disk and memory budgets
    pub fn affords(&self, file_bytes: u64) -> std::result::Result<(), String> {
        if self.disk_free_bytes > 0 {
            let needed = (file_bytes as f64 * self.disk_multiple) as u64;
            if needed > self.disk_free_bytes {
                return Err(format!(
                    "a {file_bytes} byte file needs {needed} bytes free at {} times its size \
                     and the volume has {}",
                    self.disk_multiple, self.disk_free_bytes
                ));
            }
        }
        if self.node_memory_bytes > 0 {
            let ceiling = (self.node_memory_bytes as f64 * self.memory_fraction) as u64;
            if file_bytes > ceiling {
                return Err(format!(
                    "a {file_bytes} byte file exceeds the {ceiling} bytes the sweep may hold, \
                     {} of node memory",
                    self.memory_fraction
                ));
            }
        }
        Ok(())
    }
}

/// What one format's sweep did
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SweepResult {
    pub kind: FormatKind,
    pub files_scanned: u64,
    pub files_migrated: u64,
    pub files_skipped: u64,
    pub failures: u64,
    /// True when the budget ran out before the sweep finished
    pub budget_exhausted: bool,
}

/// Sweeps one format's files forward.
///
/// Only the eager policy sweeps. Lazy leaves the file for its next write and
/// coexist leaves it forever, so calling this for either is a no-op that
/// reports what it did not do rather than pretending to work
pub fn sweep_format(
    registry: &FormatRegistry,
    board: &MigrationBoard,
    kind: FormatKind,
    directory: &Path,
    budget: MigrationBudget,
    now_secs: u64,
) -> Result<SweepResult> {
    let entry = registry
        .get(kind)
        .ok_or_else(|| ZyronError::FormatRegistry(format!("format `{kind}` is not registered")))?;
    let current = entry.registration.writer_current_version;
    let mut result = SweepResult {
        kind,
        files_scanned: 0,
        files_migrated: 0,
        files_skipped: 0,
        failures: 0,
        budget_exhausted: false,
    };
    if entry.registration.migration_policy != MigrationPolicy::Eager {
        return Ok(result);
    }

    // Sizes came from the directory read, so the totals cost no syscalls
    let files = collect_files(directory);
    let total_bytes: u64 = files.iter().map(|(_, size)| *size).sum();
    let progress = board.start(MigrationProgress::new(
        kind,
        entry.registration.reader_supported_versions.oldest,
        current,
        MigrationPolicy::Eager,
        now_secs,
    ));
    progress.set_totals(files.len() as u64, total_bytes);

    // The budget is wall clock from the moment the sweep starts. `now_secs`
    // is the caller's clock for progress reporting, which a test moves
    // independently, so the two are never mixed
    let deadline = elapsed_now().saturating_add(budget.time_secs);
    for (path, size) in files {
        if progress.is_paused() {
            result.budget_exhausted = true;
            break;
        }
        if elapsed_now() >= deadline {
            result.budget_exhausted = true;
            break;
        }
        if let Err(reason) = budget.affords(size) {
            tracing::warn!(
                path = %path.display(),
                reason = %reason,
                "format migration stopped at this file, the budget does not cover it"
            );
            result.budget_exhausted = true;
            break;
        }
        match migrate_file(registry, kind, current, &path, size) {
            // Another format's file was never a candidate, so it counts
            // neither as scanned nor as skipped
            Ok(FileOutcome::NotThisFormat) => {}
            Ok(FileOutcome::Migrated(written)) => {
                result.files_scanned += 1;
                result.files_migrated += 1;
                progress.record_file(written);
            }
            Ok(FileOutcome::Skipped) => {
                result.files_scanned += 1;
                result.files_skipped += 1;
                progress.record_file(0);
            }
            Err(e) => {
                result.failures += 1;
                progress.record_failure();
                tracing::warn!(
                    path = %path.display(),
                    error = %e,
                    "format migration skipped this file"
                );
            }
        }
    }
    progress.finish(elapsed_now());
    Ok(result)
}

/// Moves one file forward, returning whether it changed
/// What one file's visit produced
enum FileOutcome {
    /// The file is not this format, so it was never a candidate
    NotThisFormat,
    /// Already at the current version
    Skipped,
    /// Rewritten, carrying the byte count written
    Migrated(u64),
}

fn migrate_file(
    registry: &FormatRegistry,
    kind: FormatKind,
    current: FormatVersion,
    path: &Path,
    size_hint: u64,
) -> Result<FileOutcome> {
    use std::io::Read;

    // One open serves both the format check and the read. Peeking through a
    // separate handle costs an open and close per file on top of the read
    // that follows it, which is the sweep's largest avoidable cost. A file
    // of another format still pays only the eight byte peek
    let mut file = std::fs::File::open(path).map_err(ZyronError::Io)?;
    let mut head = [0u8; zyron_common::format::ENVELOPE_PEEK_LEN];
    if file.read_exact(&mut head).is_err() {
        return Ok(FileOutcome::NotThisFormat);
    }
    let version = match envelope::peek(&head) {
        Ok((found, version)) if found == kind => version,
        _ => return Ok(FileOutcome::NotThisFormat),
    };
    // A file already at the writer's version has nothing to move, and its
    // body stays unread. Reading it would verify a checksum a file still
    // being written cannot carry, and would make every sweep a full read of
    // the data directory
    if version == current {
        return Ok(FileOutcome::Skipped);
    }
    let mut bytes = Vec::with_capacity(size_hint.max(head.len() as u64) as usize);
    bytes.extend_from_slice(&head);
    file.read_to_end(&mut bytes).map_err(ZyronError::Io)?;
    drop(file);

    let opened = migration::open_as(registry, &bytes, kind)?;
    if !opened.path.needs_migration() {
        return Ok(FileOutcome::Skipped);
    }
    // The bytes at the current version, re-wrapped for an envelope framed
    // format and returned whole by the migration for one that owns its
    // trailer
    let rewritten = opened.reencode();
    let written = rewritten.len() as u64;
    let tmp = path.with_extension("zymig.tmp");
    std::fs::write(&tmp, &rewritten).map_err(ZyronError::Io)?;
    std::fs::rename(&tmp, path).map_err(ZyronError::Io)?;
    // The size is what was just written, so progress does not pay a stat
    // to re-learn what this call already knows
    Ok(FileOutcome::Migrated(written))
}

/// Every file in a directory tree, with its size. Which ones are this
/// format is decided in the visit, which has each file open anyway
fn collect_files(directory: &Path) -> Vec<(PathBuf, u64)> {
    let mut out = Vec::new();
    let mut stack = vec![directory.to_path_buf()];
    while let Some(current) = stack.pop() {
        let Ok(entries) = std::fs::read_dir(&current) else {
            continue;
        };
        for entry in entries.flatten() {
            // The directory read already knows the kind and the size, so
            // both come from the entry. Asking the path instead costs a
            // stat per file on top of the read the sweep is about to do
            let Ok(file_type) = entry.file_type() else {
                continue;
            };
            let path = entry.path();
            if file_type.is_dir() {
                stack.push(path);
                continue;
            }
            // The format check happens in the visit, which has the file open
            // anyway. Doing it here would open every file twice
            let size = entry.metadata().map(|m| m.len()).unwrap_or(0);
            out.push((path, size));
        }
    }
    out.sort();
    out
}

fn elapsed_now() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

// ---------------------------------------------------------------------------
// Catalog schema migration
// ---------------------------------------------------------------------------

/// Reads and writes the rows of one catalog table.
///
/// Implemented over the catalog's own storage for a real migration, and over
/// an in-memory table for the tests, so the runner's transaction shape is
/// exercised without a catalog
pub trait CatalogTableStore: Send + Sync {
    /// The schema version the stored rows are at
    fn stored_version(&self, catalog_table: &str) -> FormatVersion;

    /// Every row of a table, as its stored bytes
    fn rows(&self, catalog_table: &str) -> Result<Vec<Vec<u8>>>;

    /// Replaces every row of a table and records its new schema version.
    ///
    /// The whole call is one transaction: either every row lands at the new
    /// version and the version is recorded, or nothing changes
    fn replace_rows(
        &self,
        catalog_table: &str,
        rows: Vec<Vec<u8>>,
        version: FormatVersion,
    ) -> Result<()>;
}

/// What one table's migration did
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TableMigration {
    pub catalog_table: String,
    pub from_version: FormatVersion,
    pub to_version: FormatVersion,
    pub rows_migrated: u64,
    pub reversible: bool,
}

/// Migrates every catalog table whose stored version is behind.
///
/// A table that fails is reported and left untouched, and the run continues:
/// one table's migrator refusing a row must not leave the other tables half
/// migrated with no record of which
pub fn migrate_catalog(
    registry: &CatalogSchemaRegistry,
    store: &dyn CatalogTableStore,
) -> (Vec<TableMigration>, Vec<String>) {
    let mut migrated = Vec::new();
    let mut failures = Vec::new();
    for table in registry.tables() {
        let name = table.registration.catalog_table;
        let stored = store.stored_version(name);
        let target = table.registration.current_schema_version;
        if stored >= target {
            continue;
        }
        let rows = match store.rows(name) {
            Ok(rows) => rows,
            Err(e) => {
                failures.push(format!("{name} rows unreadable, {e}"));
                continue;
            }
        };
        let mut rewritten = Vec::with_capacity(rows.len());
        let mut failed = None;
        for mut row in rows {
            if let Err(e) = table.migrate_row(stored, &mut row) {
                failed = Some(e);
                break;
            }
            rewritten.push(row);
        }
        if let Some(reason) = failed {
            failures.push(format!("{name} left at {stored}, {reason}"));
            continue;
        }
        let count = rewritten.len() as u64;
        if let Err(e) = store.replace_rows(name, rewritten, target) {
            failures.push(format!("{name} left at {stored}, {e}"));
            continue;
        }
        migrated.push(TableMigration {
            catalog_table: name.to_string(),
            from_version: stored,
            to_version: target,
            rows_migrated: count,
            reversible: table.reversible_from(stored),
        });
    }
    (migrated, failures)
}

// ---------------------------------------------------------------------------
// User object rewrites
// ---------------------------------------------------------------------------

/// What the rewrite pass did
#[derive(Debug, Clone, Default)]
pub struct RewritePass {
    pub applied: u32,
    pub queued: u32,
    pub blocked: u32,
    /// The rewritten SQL per object, for the caller to persist
    pub rewritten: Vec<(String, String)>,
}

/// Runs every registered rewriter over a set of objects and publishes the
/// queue to the upgrade board
pub fn rewrite_objects(
    board: &UpgradeBoard,
    objects: &[UserObject],
    policy: UserObjectRewritePolicy,
    now_secs: u64,
) -> RewritePass {
    let mut pass = RewritePass::default();
    let mut records: Vec<RewriteRecord> = Vec::new();
    for object in objects {
        let applied = match rewriter::apply(&object.name, object.kind, &object.sql, policy) {
            Ok(applied) => applied,
            Err(e) => {
                pass.blocked += 1;
                records.push(RewriteRecord {
                    object_name: object.name.clone(),
                    object_kind: object.kind,
                    rewriter_name: "parse".to_string(),
                    category: zyron_common::format::rewrite::RewriteCategory::Unsafe,
                    status: RewriteStatus::Failed,
                    before_hash: zyron_common::hash32(object.sql.as_bytes()),
                    after_hash: 0,
                    acknowledged_by: String::new(),
                    updated_at_secs: now_secs,
                    diff: e.to_string(),
                });
                continue;
            }
        };
        for proposal in &applied.applied {
            pass.applied += 1;
            records.push(record_of(proposal, RewriteStatus::Applied, now_secs));
        }
        for proposal in &applied.deferred {
            let disposition = proposal.disposition(policy);
            let status = if disposition.blocks_upgrade() {
                pass.blocked += 1;
                RewriteStatus::Blocked
            } else {
                pass.queued += 1;
                RewriteStatus::Pending
            };
            records.push(record_of(proposal, status, now_secs));
        }
        if !applied.applied.is_empty() {
            pass.rewritten.push((object.name.clone(), applied.sql));
        }
    }
    board.set_rewrites(records);
    pass
}

fn record_of(
    proposal: &rewriter::ProposedRewrite,
    status: RewriteStatus,
    now_secs: u64,
) -> RewriteRecord {
    RewriteRecord {
        object_name: proposal.object_name.clone(),
        object_kind: proposal.object_kind,
        rewriter_name: proposal.rewriter_name.to_string(),
        category: proposal.category,
        status,
        before_hash: proposal.before_hash(),
        after_hash: proposal.after_hash(),
        acknowledged_by: String::new(),
        updated_at_secs: now_secs,
        diff: proposal.diff.clone(),
    }
}

/// An in-memory catalog table store, used by the migration tests and by the
/// dry-run path that has to prove a migrator before running it for real
#[derive(Debug, Default)]
pub struct InMemoryCatalogStore {
    tables: parking_lot::Mutex<Vec<(String, FormatVersion, Vec<Vec<u8>>)>>,
}

impl InMemoryCatalogStore {
    pub fn new() -> Arc<Self> {
        Arc::new(Self::default())
    }

    /// Seeds one table
    pub fn seed(&self, catalog_table: &str, version: FormatVersion, rows: Vec<Vec<u8>>) {
        let mut tables = self.tables.lock();
        tables.retain(|(name, _, _)| name != catalog_table);
        tables.push((catalog_table.to_string(), version, rows));
    }

    /// Whether a table's rows are what the caller expects
    pub fn rows_of(&self, catalog_table: &str) -> Vec<Vec<u8>> {
        self.tables
            .lock()
            .iter()
            .find(|(name, _, _)| name == catalog_table)
            .map(|(_, _, rows)| rows.clone())
            .unwrap_or_default()
    }
}

impl CatalogTableStore for InMemoryCatalogStore {
    fn stored_version(&self, catalog_table: &str) -> FormatVersion {
        self.tables
            .lock()
            .iter()
            .find(|(name, _, _)| name == catalog_table)
            .map(|(_, version, _)| *version)
            .unwrap_or(FormatVersion::V1)
    }

    fn rows(&self, catalog_table: &str) -> Result<Vec<Vec<u8>>> {
        Ok(self.rows_of(catalog_table))
    }

    fn replace_rows(
        &self,
        catalog_table: &str,
        rows: Vec<Vec<u8>>,
        version: FormatVersion,
    ) -> Result<()> {
        let mut tables = self.tables.lock();
        match tables.iter_mut().find(|(name, _, _)| name == catalog_table) {
            Some(slot) => {
                slot.1 = version;
                slot.2 = rows;
            }
            None => tables.push((catalog_table.to_string(), version, rows)),
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use zyron_common::format::catalog_evolution::{
        CatalogSchemaEvolution, CatalogTableRegistration,
    };
    use zyron_common::format::registry::{
        DeprecationStatus, FormatFixture, FormatMigrator, FormatRegistration,
    };
    use zyron_common::format::rewrite::ObjectKind;
    use zyron_common::format::version::VersionWindow;

    fn append_marker(body: &[u8]) -> std::result::Result<Vec<u8>, String> {
        let mut out = body.to_vec();
        out.extend_from_slice(b"-v2");
        Ok(out)
    }

    fn strip_marker(body: &[u8]) -> std::result::Result<Vec<u8>, String> {
        match body.strip_suffix(b"-v2") {
            Some(rest) => Ok(rest.to_vec()),
            None => Err("body has no v2 marker".to_string()),
        }
    }

    fn registry(policy: MigrationPolicy) -> FormatRegistry {
        let registrations: Vec<FormatRegistration> = zyron_common::format::ALL_FORMAT_KINDS
            .iter()
            .copied()
            .map(|kind| {
                let bumped = kind == FormatKind::StatisticsFile;
                FormatRegistration {
                    kind,
                    writer_current_version: if bumped {
                        FormatVersion::new(1, 1)
                    } else {
                        FormatVersion::V1
                    },
                    reader_supported_versions: if bumped {
                        VersionWindow::new(FormatVersion::V1, FormatVersion::new(1, 1))
                    } else {
                        VersionWindow::single(FormatVersion::V1)
                    },
                    migration_policy: if bumped {
                        policy
                    } else {
                        MigrationPolicy::Lazy
                    },
                    migration_reversible: true,
                    binary_version_gate: "0.11.0",
                    deprecation_status: DeprecationStatus::Active,
                    retirement_date: if bumped { Some("2027-01-01") } else { None },
                    downgrade_write_supported: false,
                    notes: "test",
                }
            })
            .collect();
        let migrators = [FormatMigrator {
            kind: FormatKind::StatisticsFile,
            from: FormatVersion::V1,
            to: FormatVersion::new(1, 1),
            reversible: true,
            forward: append_marker,
            backward: Some(strip_marker),
            no_body_change: false,
            description: "appends the v2 marker",
        }];
        let fixtures = [FormatFixture {
            kind: FormatKind::StatisticsFile,
            version: FormatVersion::V1,
            bytes: b"",
            path: "fixtures/v1.bin",
        }];
        FormatRegistry::from_parts(&registrations, &migrators, &fixtures).expect("loads")
    }

    #[test]
    fn test_an_eager_sweep_moves_every_old_file_forward() {
        let dir = tempfile::tempdir().expect("tempdir");
        for i in 0..5 {
            let bytes = envelope::encode(
                FormatKind::StatisticsFile,
                FormatVersion::V1,
                format!("body {i}").as_bytes(),
            );
            std::fs::write(dir.path().join(format!("{i}.zysts")), bytes).expect("writes");
        }
        // One file is already current and must be left alone
        let current = envelope::encode(
            FormatKind::StatisticsFile,
            FormatVersion::new(1, 1),
            b"already current",
        );
        std::fs::write(dir.path().join("current.zysts"), current).expect("writes");

        let registry = registry(MigrationPolicy::Eager);
        let board = MigrationBoard::new();
        let result = sweep_format(
            &registry,
            &board,
            FormatKind::StatisticsFile,
            dir.path(),
            MigrationBudget::default(),
            0,
        )
        .expect("sweeps");
        assert_eq!(result.files_scanned, 6);
        assert_eq!(result.files_migrated, 5);
        assert_eq!(result.files_skipped, 1);
        assert_eq!(result.failures, 0);
        assert!(!result.budget_exhausted);

        let moved = std::fs::read(dir.path().join("0.zysts")).expect("reads");
        let opened =
            migration::open_as(&registry, &moved, FormatKind::StatisticsFile).expect("opens");
        assert_eq!(opened.version, FormatVersion::new(1, 1));
        assert_eq!(opened.body.as_ref(), b"body 0-v2");

        let runs = board.runs_for(FormatKind::StatisticsFile);
        assert_eq!(runs.len(), 1);
        assert!(runs[0].is_finished());
        assert_eq!(runs[0].files_done(), 6);
    }

    #[test]
    fn test_a_current_file_is_skipped_without_its_body_being_read() {
        // A file still being written, the open WAL segment for one, carries
        // a checksum that does not hold yet. At the current version it is
        // not a candidate, so the sweep decides from the header alone
        let dir = tempfile::tempdir().expect("tempdir");
        let mut bytes = envelope::encode(
            FormatKind::StatisticsFile,
            FormatVersion::new(1, 1),
            b"a body whose checksum is stale",
        );
        let last = bytes.len() - 1;
        bytes[last] ^= 0xff;
        std::fs::write(dir.path().join("live.zysts"), &bytes).expect("writes");
        assert!(
            migration::open_as(
                &registry(MigrationPolicy::Eager),
                &bytes,
                FormatKind::StatisticsFile
            )
            .is_err()
        );

        let registry = registry(MigrationPolicy::Eager);
        let board = MigrationBoard::new();
        let result = sweep_format(
            &registry,
            &board,
            FormatKind::StatisticsFile,
            dir.path(),
            MigrationBudget::default(),
            0,
        )
        .expect("sweeps");
        assert_eq!(result.files_scanned, 1);
        assert_eq!(result.files_skipped, 1);
        assert_eq!(result.failures, 0);
        assert_eq!(
            std::fs::read(dir.path().join("live.zysts")).expect("reads"),
            bytes
        );
    }

    #[test]
    fn test_a_sweep_stops_at_a_file_the_disk_or_memory_budget_does_not_cover() {
        let dir = tempfile::tempdir().expect("tempdir");
        for i in 0..3 {
            let bytes = envelope::encode(
                FormatKind::StatisticsFile,
                FormatVersion::V1,
                format!("body {i}").as_bytes(),
            );
            std::fs::write(dir.path().join(format!("{i}.zysts")), bytes).expect("writes");
        }
        let registry = registry(MigrationPolicy::Eager);
        let board = MigrationBoard::new();
        // Every file is a few dozen bytes, and the volume reports eight free
        let result = sweep_format(
            &registry,
            &board,
            FormatKind::StatisticsFile,
            dir.path(),
            MigrationBudget {
                disk_free_bytes: 8,
                ..MigrationBudget::default()
            },
            0,
        )
        .expect("sweeps");
        assert!(result.budget_exhausted);
        assert_eq!(result.files_migrated, 0);
        let untouched = std::fs::read(dir.path().join("0.zysts")).expect("reads");
        let opened =
            migration::open_as(&registry, &untouched, FormatKind::StatisticsFile).expect("opens");
        assert_eq!(opened.version, FormatVersion::V1, "nothing moved");

        let budget = MigrationBudget {
            node_memory_bytes: 1_000,
            memory_fraction: 0.25,
            ..MigrationBudget::default()
        };
        assert!(budget.affords(250).is_ok());
        let err = budget.affords(251).expect_err("over the fraction");
        assert!(err.contains("250 bytes"), "{err}");
        assert!(
            MigrationBudget::default().affords(u64::MAX).is_ok(),
            "unmeasured is unbounded"
        );
    }

    #[test]
    fn test_a_lazy_format_is_not_swept() {
        let dir = tempfile::tempdir().expect("tempdir");
        let bytes = envelope::encode(FormatKind::StatisticsFile, FormatVersion::V1, b"body");
        std::fs::write(dir.path().join("a.zysts"), &bytes).expect("writes");
        let registry = registry(MigrationPolicy::Lazy);
        let board = MigrationBoard::new();
        let result = sweep_format(
            &registry,
            &board,
            FormatKind::StatisticsFile,
            dir.path(),
            MigrationBudget::default(),
            0,
        )
        .expect("sweeps");
        assert_eq!(result.files_scanned, 0);
        assert_eq!(result.files_migrated, 0);
        assert_eq!(
            std::fs::read(dir.path().join("a.zysts")).expect("reads"),
            bytes,
            "a lazy file is untouched until it is written"
        );
    }

    #[test]
    fn test_a_coexist_format_is_not_swept() {
        let dir = tempfile::tempdir().expect("tempdir");
        let bytes = envelope::encode(FormatKind::StatisticsFile, FormatVersion::V1, b"body");
        std::fs::write(dir.path().join("a.zysts"), &bytes).expect("writes");
        let registry = registry(MigrationPolicy::Coexist);
        let board = MigrationBoard::new();
        let result = sweep_format(
            &registry,
            &board,
            FormatKind::StatisticsFile,
            dir.path(),
            MigrationBudget::default(),
            0,
        )
        .expect("sweeps");
        assert_eq!(result.files_migrated, 0);
        assert_eq!(
            std::fs::read(dir.path().join("a.zysts")).expect("reads"),
            bytes
        );
    }

    #[test]
    fn test_a_file_of_another_format_is_left_alone() {
        let dir = tempfile::tempdir().expect("tempdir");
        let foreign = envelope::encode(FormatKind::HeapPage, FormatVersion::V1, b"not ours");
        std::fs::write(dir.path().join("foreign.bin"), &foreign).expect("writes");
        let registry = registry(MigrationPolicy::Eager);
        let board = MigrationBoard::new();
        let result = sweep_format(
            &registry,
            &board,
            FormatKind::StatisticsFile,
            dir.path(),
            MigrationBudget::default(),
            0,
        )
        .expect("sweeps");
        assert_eq!(result.files_scanned, 0);
        assert_eq!(
            std::fs::read(dir.path().join("foreign.bin")).expect("reads"),
            foreign
        );
    }

    fn add_field(row: &mut Vec<u8>) -> std::result::Result<(), String> {
        row.extend_from_slice(b"|default");
        Ok(())
    }

    fn drop_field(row: &mut Vec<u8>) -> std::result::Result<(), String> {
        match row.len().checked_sub(8) {
            Some(cut) if &row[cut..] == b"|default" => {
                row.truncate(cut);
                Ok(())
            }
            _ => Err("row has no added field".to_string()),
        }
    }

    fn refuse(_row: &mut Vec<u8>) -> std::result::Result<(), String> {
        Err("this migrator refuses every row".to_string())
    }

    fn catalog_registry(forward: zyron_common::format::RowMigrateFn) -> CatalogSchemaRegistry {
        CatalogSchemaRegistry::from_parts(
            &[CatalogTableRegistration {
                catalog_table: "zyron_sys.auth.groups",
                current_schema_version: FormatVersion::new(1, 1),
                introduced_in_binary_version: "0.11.0",
                doc: "groups and the roles they carry",
            }],
            &[CatalogSchemaEvolution {
                catalog_table: "zyron_sys.auth.groups",
                from_version: FormatVersion::V1,
                to_version: FormatVersion::new(1, 1),
                migration_function_ref: "test::add_field",
                reversible: true,
                introduced_in_binary_version: "0.11.0",
                forward,
                backward: Some(drop_field),
                description: "adds the source column with a default",
            }],
        )
        .expect("loads")
    }

    #[test]
    fn test_every_row_of_a_behind_table_is_migrated_in_one_transaction() {
        let registry = catalog_registry(add_field);
        let store = InMemoryCatalogStore::new();
        store.seed(
            "zyron_sys.auth.groups",
            FormatVersion::V1,
            vec![b"admins".to_vec(), b"readers".to_vec()],
        );
        let (migrated, failures) = migrate_catalog(&registry, store.as_ref());
        assert!(failures.is_empty(), "{failures:?}");
        assert_eq!(migrated.len(), 1);
        assert_eq!(migrated[0].rows_migrated, 2);
        assert!(migrated[0].reversible);
        assert_eq!(
            store.stored_version("zyron_sys.auth.groups"),
            FormatVersion::new(1, 1)
        );
        assert_eq!(
            store.rows_of("zyron_sys.auth.groups"),
            vec![b"admins|default".to_vec(), b"readers|default".to_vec()]
        );
    }

    #[test]
    fn test_a_table_already_at_the_current_version_is_skipped() {
        let registry = catalog_registry(add_field);
        let store = InMemoryCatalogStore::new();
        store.seed(
            "zyron_sys.auth.groups",
            FormatVersion::new(1, 1),
            vec![b"admins|default".to_vec()],
        );
        let (migrated, failures) = migrate_catalog(&registry, store.as_ref());
        assert!(migrated.is_empty());
        assert!(failures.is_empty());
    }

    #[test]
    fn test_a_refusing_migrator_leaves_the_table_untouched() {
        let registry = catalog_registry(refuse);
        let store = InMemoryCatalogStore::new();
        store.seed(
            "zyron_sys.auth.groups",
            FormatVersion::V1,
            vec![b"admins".to_vec()],
        );
        let (migrated, failures) = migrate_catalog(&registry, store.as_ref());
        assert!(migrated.is_empty());
        assert_eq!(failures.len(), 1);
        assert!(failures[0].contains("refuses every row"), "{failures:?}");
        assert_eq!(
            store.stored_version("zyron_sys.auth.groups"),
            FormatVersion::V1,
            "the version is not bumped when the rows did not move"
        );
        assert_eq!(
            store.rows_of("zyron_sys.auth.groups"),
            vec![b"admins".to_vec()],
            "the rows are untouched"
        );
    }

    #[test]
    fn test_the_rewrite_pass_applies_queues_and_blocks() {
        let board = UpgradeBoard::new();
        let objects = vec![
            UserObject {
                name: "safe_view".to_string(),
                kind: ObjectKind::View,
                sql: "CREATE VIEW safe_view AS SELECT a FROM warehouse_x".to_string(),
            },
            UserObject {
                name: "widened".to_string(),
                kind: ObjectKind::View,
                sql: "CREATE VIEW widened AS SELECT old_agg(a) FROM t".to_string(),
            },
            UserObject {
                name: "broken".to_string(),
                kind: ObjectKind::View,
                sql: "CREATE VIEW broken AS SELECT gone_fn(a) FROM t".to_string(),
            },
        ];
        let pass = rewrite_objects(&board, &objects, UserObjectRewritePolicy::AutoSafe, 1_000);
        assert_eq!(pass.applied, 1);
        assert_eq!(pass.queued, 1);
        assert_eq!(pass.blocked, 1);
        assert_eq!(pass.rewritten.len(), 1);
        assert!(pass.rewritten[0].1.contains("compute_x"));

        let records = board.rewrites();
        assert_eq!(records.len(), 3);
        assert!(records.iter().any(|r| r.status == RewriteStatus::Applied));
        assert!(records.iter().any(|r| r.status == RewriteStatus::Pending));
        assert!(records.iter().any(|r| r.status == RewriteStatus::Blocked));

        let moved = board.acknowledge(
            zyron_common::format::rewrite::RewriteCategory::Ambiguous,
            "admin",
            2_000,
        );
        assert_eq!(moved, 1);
        assert!(
            board
                .rewrites()
                .iter()
                .any(|r| r.status == RewriteStatus::Acknowledged && r.acknowledged_by == "admin")
        );
    }

    #[test]
    fn test_an_unparseable_object_is_recorded_as_failed() {
        let board = UpgradeBoard::new();
        let objects = vec![UserObject {
            name: "bad".to_string(),
            kind: ObjectKind::View,
            sql: "NOT SQL".to_string(),
        }];
        let pass = rewrite_objects(&board, &objects, UserObjectRewritePolicy::AutoSafe, 0);
        assert_eq!(pass.blocked, 1);
        assert_eq!(board.rewrites()[0].status, RewriteStatus::Failed);
    }
}
