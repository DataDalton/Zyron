//! Reader dispatch and migration application.
//!
//! Opening a file peeks its envelope and compares the version it carries
//! against the version the writer emits. A match takes the current reader
//! path and costs one integer compare. A version behind the current one
//! dispatches to that version's reader, which is the chain of registered
//! migrators from it up to the current version followed by the current
//! reader. A version the binary has no reader for fails closed, naming the
//! version and the upgrade path.
//!
//! Nothing here scans a directory. Migration runs when a policy says to run
//! it, on the file being touched for lazy, on a budgeted background sweep
//! for eager, and never for coexist

use std::borrow::Cow;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

use super::envelope::{self, EnvelopeError};
use super::kind::{FormatKind, Framing};
use super::registry::{FormatEntry, FormatMigrator, FormatRegistry, MigrationPolicy};
use super::version::FormatVersion;

/// Which reader a file's version dispatches to
#[derive(Debug, Clone, PartialEq)]
pub enum ReaderPath {
    /// The version matches the writer, the current reader handles it with no
    /// transformation
    Current,
    /// The version is behind, this chain of migrators runs before the
    /// current reader sees the body
    Migrate {
        from: FormatVersion,
        to: FormatVersion,
        chain: Vec<FormatMigrator>,
    },
}

impl ReaderPath {
    /// The version whose reader this path names, which is the file's own
    /// version in both cases
    pub fn dispatched_version(&self, current: FormatVersion) -> FormatVersion {
        match self {
            ReaderPath::Current => current,
            ReaderPath::Migrate { from, .. } => *from,
        }
    }

    #[inline]
    pub fn needs_migration(&self) -> bool {
        matches!(self, ReaderPath::Migrate { .. })
    }
}

/// What a migration produced and whether it has to be written back
#[derive(Debug, Clone)]
pub struct MigratedBody {
    /// The body at the current writer version, or the whole file for a
    /// format that owns its trailer
    pub body: Vec<u8>,
    pub from: FormatVersion,
    pub to: FormatVersion,
    /// True when every step could be undone, which decides downgrade
    /// eligibility for the file
    pub reversible: bool,
}

/// A file opened through the substrate
#[derive(Debug, Clone)]
pub struct OpenedFile<'a> {
    pub kind: FormatKind,
    /// The version the file was written at
    pub version: FormatVersion,
    /// The version the current writer emits
    pub current_version: FormatVersion,
    pub path: ReaderPath,
    /// The body at the current version, borrowed when no migration ran.
    /// The whole file for a format that owns its trailer
    pub body: Cow<'a, [u8]>,
    pub policy: MigrationPolicy,
    /// How the kind is framed, which decides what `body` holds and how it
    /// is written back
    pub framing: Framing,
}

impl OpenedFile<'_> {
    /// Whether the migrated bytes should be written back now.
    ///
    /// Lazy writes back when the file is next modified, which the caller
    /// signals by passing `modifying`. Eager writes back from the sweep.
    /// Coexist never writes back
    pub fn should_write_back(&self, modifying: bool) -> bool {
        if !self.path.needs_migration() {
            return false;
        }
        match self.policy {
            MigrationPolicy::Lazy => modifying,
            MigrationPolicy::Eager => true,
            MigrationPolicy::Coexist => false,
        }
    }

    /// The bytes to write back at the current version.
    ///
    /// An envelope framed file is re-wrapped around the current body. A
    /// file that owns its trailer is the body itself, which its migrators
    /// returned whole with the format's own checksums in place
    pub fn reencode(&self) -> Vec<u8> {
        match self.framing {
            Framing::OwnTrailer => self.body.to_vec(),
            _ => envelope::encode(self.kind, self.current_version, &self.body),
        }
    }
}

/// Runs the chain that takes a body from one version to another
pub fn migrate_body(
    entry: &FormatEntry,
    from: FormatVersion,
    to: FormatVersion,
    body: &[u8],
) -> Result<MigratedBody, MigrationError> {
    let chain = entry
        .plan(from, to)
        .map_err(|e| MigrationError::Planning(e.to_string()))?;
    migrate_body_with_chain(entry, &chain, from, to, body)
}

/// Runs an already planned chain.
///
/// A caller that resolved the reader path already holds the chain, and
/// planning it a second time allocates another vector and rescans the
/// migrator list for the same answer
pub fn migrate_body_with_chain(
    entry: &FormatEntry,
    chain: &[FormatMigrator],
    from: FormatVersion,
    to: FormatVersion,
    body: &[u8],
) -> Result<MigratedBody, MigrationError> {
    // Borrowed until a step actually rewrites the body. Copying up front
    // costs a full body memcpy that the first rewriting step throws away,
    // because `forward` allocates its own output from a borrowed input
    let mut current: Cow<'_, [u8]> = Cow::Borrowed(body);
    let mut reversible = true;
    for step in chain {
        if !step.no_body_change {
            current =
                Cow::Owned(
                    (step.forward)(&current).map_err(|reason| MigrationError::Step {
                        kind: entry.registration.kind,
                        from: step.from,
                        to: step.to,
                        reason,
                    })?,
                );
        }
        reversible &= step.no_body_change || step.reversible;
    }
    Ok(MigratedBody {
        body: current.into_owned(),
        from,
        to,
        reversible,
    })
}

/// Runs the chain backwards, which is what a downgrade does
pub fn migrate_body_backward(
    entry: &FormatEntry,
    from: FormatVersion,
    to: FormatVersion,
    body: &[u8],
) -> Result<MigratedBody, MigrationError> {
    let chain = entry
        .plan(to, from)
        .map_err(|e| MigrationError::Planning(e.to_string()))?;
    let mut current: Cow<'_, [u8]> = Cow::Borrowed(body);
    for step in chain.iter().rev() {
        if step.no_body_change {
            continue;
        }
        let Some(backward) = step.backward else {
            return Err(MigrationError::OneWay {
                kind: entry.registration.kind,
                from: step.from,
                to: step.to,
                description: step.description,
            });
        };
        current = Cow::Owned(backward(&current).map_err(|reason| MigrationError::Step {
            kind: entry.registration.kind,
            from: step.to,
            to: step.from,
            reason,
        })?);
    }
    Ok(MigratedBody {
        body: current.into_owned(),
        from,
        to,
        reversible: true,
    })
}

/// What went wrong opening or migrating a file
#[derive(Debug, Clone)]
pub enum MigrationError {
    /// The envelope itself would not parse
    Envelope(EnvelopeError),
    /// No registration for the kind, which a loaded registry makes
    /// impossible and a hand-built one does not
    Unregistered { kind: FormatKind },
    /// The chain could not be planned
    Planning(String),
    /// One step refused the body it was given
    Step {
        kind: FormatKind,
        from: FormatVersion,
        to: FormatVersion,
        reason: String,
    },
    /// A downgrade asked to undo a step that has no inverse
    OneWay {
        kind: FormatKind,
        from: FormatVersion,
        to: FormatVersion,
        description: &'static str,
    },
}

impl std::fmt::Display for MigrationError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            MigrationError::Envelope(e) => write!(f, "{e}"),
            MigrationError::Unregistered { kind } => {
                write!(f, "format `{kind}` is not registered in this binary")
            }
            MigrationError::Planning(reason) => write!(f, "{reason}"),
            MigrationError::Step {
                kind,
                from,
                to,
                reason,
            } => write!(
                f,
                "migrating a {kind} body from {from} to {to} failed, {reason}"
            ),
            MigrationError::OneWay {
                kind,
                from,
                to,
                description,
            } => write!(
                f,
                "the {kind} migration from {from} to {to} is one way, so a downgrade past \
                 it is not possible. The migration is `{description}`. Restore from the \
                 pre-upgrade backup snapshot instead"
            ),
        }
    }
}

impl std::error::Error for MigrationError {}

impl From<EnvelopeError> for MigrationError {
    fn from(value: EnvelopeError) -> Self {
        MigrationError::Envelope(value)
    }
}

/// Resolves which reader a version dispatches to.
///
/// This is the enum match on the hot path. A version equal to the writer's
/// returns `Current` without touching the migrator list
#[inline]
pub fn reader_path(
    entry: &FormatEntry,
    version: FormatVersion,
) -> Result<ReaderPath, EnvelopeError> {
    let registration = &entry.registration;
    if version == registration.writer_current_version {
        return Ok(ReaderPath::Current);
    }
    let window = registration.reader_supported_versions;
    if !window.contains(version) {
        return Err(EnvelopeError::UnknownVersion {
            kind: registration.kind,
            found: version,
            oldest_supported: window.oldest,
            newest_supported: window.newest,
        });
    }
    if version > registration.writer_current_version {
        // Inside the window but ahead of the writer, which happens on a node
        // that has not been upgraded yet reading a file a newer peer wrote
        return Err(EnvelopeError::UnknownVersion {
            kind: registration.kind,
            found: version,
            oldest_supported: window.oldest,
            newest_supported: registration.writer_current_version,
        });
    }
    let chain = entry
        .plan(version, registration.writer_current_version)
        .map_err(|_| EnvelopeError::UnknownVersion {
            kind: registration.kind,
            found: version,
            oldest_supported: window.oldest,
            newest_supported: window.newest,
        })?;
    Ok(ReaderPath::Migrate {
        from: version,
        to: registration.writer_current_version,
        chain,
    })
}

/// Opens a file, dispatching by version and migrating it forward when the
/// version is behind. The kind comes off the file's own header
pub fn open<'a>(
    registry: &FormatRegistry,
    bytes: &'a [u8],
) -> Result<OpenedFile<'a>, MigrationError> {
    let (kind, _) = envelope::peek(bytes)?;
    open_as(registry, bytes, kind)
}

/// Opens a file and refuses one of a different kind.
///
/// An envelope framed file has both its checksums verified and its body
/// carved out here. A format that owns its trailer has only its header
/// read, because its trailer is its own to verify, and the whole file is
/// what its migrators take and return
pub fn open_as<'a>(
    registry: &FormatRegistry,
    bytes: &'a [u8],
    expected: FormatKind,
) -> Result<OpenedFile<'a>, MigrationError> {
    let entry = registry
        .get(expected)
        .ok_or(MigrationError::Unregistered { kind: expected })?;
    if expected.framing().migrates_whole_file() {
        let (header, _) = envelope::decode_header(bytes)?;
        if header.kind != expected {
            return Err(EnvelopeError::MagicMismatch {
                expected,
                found: header.kind.magic(),
            }
            .into());
        }
        return open_parsed(entry, expected, header.version, bytes);
    }
    let parsed = envelope::decode_as(bytes, expected)?;
    open_parsed(entry, expected, parsed.header.version, parsed.body)
}

fn open_parsed<'a>(
    entry: &FormatEntry,
    kind: FormatKind,
    version: FormatVersion,
    body: &'a [u8],
) -> Result<OpenedFile<'a>, MigrationError> {
    let path = reader_path(entry, version)?;
    let current_version = entry.registration.writer_current_version;
    let body = match &path {
        ReaderPath::Current => Cow::Borrowed(body),
        // The chain was planned resolving the path, so it is handed straight
        // to the runner rather than planned a second time
        ReaderPath::Migrate { from, to, chain } => {
            Cow::Owned(migrate_body_with_chain(entry, chain, *from, *to, body)?.body)
        }
    };
    Ok(OpenedFile {
        kind,
        version,
        current_version,
        path,
        body,
        policy: entry.registration.migration_policy,
        framing: kind.framing(),
    })
}

// ---------------------------------------------------------------------------
// Sweep progress
// ---------------------------------------------------------------------------

/// Where one format's migration sweep has got to.
///
/// Counters are atomic so the sweep updates them without a lock and the
/// catalog view reads them without stalling the sweep
#[derive(Debug)]
pub struct MigrationProgress {
    pub kind: FormatKind,
    pub from: FormatVersion,
    pub to: FormatVersion,
    pub policy: MigrationPolicy,
    files_total: AtomicU64,
    files_done: AtomicU64,
    bytes_total: AtomicU64,
    bytes_done: AtomicU64,
    started_at_secs: AtomicU64,
    finished_at_secs: AtomicU64,
    paused: AtomicBool,
    failures: AtomicU64,
}

impl MigrationProgress {
    pub fn new(
        kind: FormatKind,
        from: FormatVersion,
        to: FormatVersion,
        policy: MigrationPolicy,
        started_at_secs: u64,
    ) -> Self {
        Self {
            kind,
            from,
            to,
            policy,
            files_total: AtomicU64::new(0),
            files_done: AtomicU64::new(0),
            bytes_total: AtomicU64::new(0),
            bytes_done: AtomicU64::new(0),
            started_at_secs: AtomicU64::new(started_at_secs),
            finished_at_secs: AtomicU64::new(0),
            paused: AtomicBool::new(false),
            failures: AtomicU64::new(0),
        }
    }

    pub fn set_totals(&self, files: u64, bytes: u64) {
        self.files_total.store(files, Ordering::Relaxed);
        self.bytes_total.store(bytes, Ordering::Relaxed);
    }

    pub fn record_file(&self, bytes: u64) {
        self.files_done.fetch_add(1, Ordering::Relaxed);
        self.bytes_done.fetch_add(bytes, Ordering::Relaxed);
    }

    pub fn record_failure(&self) {
        self.failures.fetch_add(1, Ordering::Relaxed);
    }

    pub fn finish(&self, at_secs: u64) {
        self.finished_at_secs.store(at_secs, Ordering::Relaxed);
    }

    pub fn set_paused(&self, paused: bool) {
        self.paused.store(paused, Ordering::Relaxed);
    }

    pub fn is_paused(&self) -> bool {
        self.paused.load(Ordering::Relaxed)
    }

    pub fn is_finished(&self) -> bool {
        self.finished_at_secs.load(Ordering::Relaxed) != 0
    }

    pub fn files_total(&self) -> u64 {
        self.files_total.load(Ordering::Relaxed)
    }

    pub fn files_done(&self) -> u64 {
        self.files_done.load(Ordering::Relaxed)
    }

    pub fn bytes_total(&self) -> u64 {
        self.bytes_total.load(Ordering::Relaxed)
    }

    pub fn bytes_done(&self) -> u64 {
        self.bytes_done.load(Ordering::Relaxed)
    }

    pub fn bytes_remaining(&self) -> u64 {
        self.bytes_total().saturating_sub(self.bytes_done())
    }

    pub fn failures(&self) -> u64 {
        self.failures.load(Ordering::Relaxed)
    }

    pub fn started_at_secs(&self) -> u64 {
        self.started_at_secs.load(Ordering::Relaxed)
    }

    pub fn finished_at_secs(&self) -> u64 {
        self.finished_at_secs.load(Ordering::Relaxed)
    }

    /// Completion as a percentage of files, 100 when there was nothing to do
    pub fn percent_complete(&self) -> f64 {
        let total = self.files_total();
        if total == 0 {
            return 100.0;
        }
        (self.files_done() as f64 / total as f64) * 100.0
    }

    /// Seconds still expected, from the rate achieved so far, or None while
    /// no file has finished
    pub fn eta_secs(&self, now_secs: u64) -> Option<u64> {
        let done = self.files_done();
        if done == 0 {
            return None;
        }
        let total = self.files_total();
        if done >= total {
            return Some(0);
        }
        let elapsed = now_secs.saturating_sub(self.started_at_secs());
        if elapsed == 0 {
            return None;
        }
        let rate = done as f64 / elapsed as f64;
        if rate <= 0.0 {
            return None;
        }
        Some(((total - done) as f64 / rate).ceil() as u64)
    }
}

/// Every migration the node is running or has run since it started.
///
/// The list is touched once per migration start and once per catalog read,
/// never on a query path, so a plain mutex is the right shape here
#[derive(Debug, Default)]
pub struct MigrationBoard {
    runs: std::sync::Mutex<Vec<Arc<MigrationProgress>>>,
}

impl MigrationBoard {
    pub fn new() -> Self {
        Self::default()
    }

    /// Registers a run and hands back the handle the sweep updates
    pub fn start(&self, progress: MigrationProgress) -> Arc<MigrationProgress> {
        let handle = Arc::new(progress);
        if let Ok(mut runs) = self.runs.lock() {
            runs.push(Arc::clone(&handle));
        }
        handle
    }

    /// Every run, newest last
    pub fn runs(&self) -> Vec<Arc<MigrationProgress>> {
        self.runs
            .lock()
            .map(|runs| runs.clone())
            .unwrap_or_default()
    }

    /// Runs for one format kind
    pub fn runs_for(&self, kind: FormatKind) -> Vec<Arc<MigrationProgress>> {
        self.runs()
            .into_iter()
            .filter(|run| run.kind == kind)
            .collect()
    }

    /// Pauses or resumes every run
    pub fn set_paused(&self, paused: bool) {
        for run in self.runs() {
            run.set_paused(paused);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::format::kind::ALL_FORMAT_KINDS;
    use crate::format::registry::{
        DeprecationStatus, FormatFixture, FormatRegistration, MigrationPolicy,
    };
    use crate::format::version::VersionWindow;

    fn append_marker(body: &[u8]) -> Result<Vec<u8>, String> {
        let mut out = body.to_vec();
        out.push(b'2');
        Ok(out)
    }

    fn strip_marker(body: &[u8]) -> Result<Vec<u8>, String> {
        match body.split_last() {
            Some((b'2', rest)) => Ok(rest.to_vec()),
            _ => Err("body does not end with the v2 marker".to_string()),
        }
    }

    fn registrations(kind: FormatKind, policy: MigrationPolicy) -> Vec<FormatRegistration> {
        ALL_FORMAT_KINDS
            .iter()
            .copied()
            .map(|k| FormatRegistration {
                kind: k,
                writer_current_version: if k == kind {
                    FormatVersion::new(1, 1)
                } else {
                    FormatVersion::V1
                },
                reader_supported_versions: if k == kind {
                    VersionWindow::new(FormatVersion::V1, FormatVersion::new(1, 1))
                } else {
                    VersionWindow::single(FormatVersion::V1)
                },
                migration_policy: if k == kind {
                    policy
                } else {
                    MigrationPolicy::Lazy
                },
                migration_reversible: true,
                binary_version_gate: "0.11.0",
                deprecation_status: DeprecationStatus::Active,
                retirement_date: if k == kind { Some("2027-06-01") } else { None },
                downgrade_write_supported: false,
                notes: "test",
            })
            .collect()
    }

    fn registry_with(
        kind: FormatKind,
        policy: MigrationPolicy,
        reversible: bool,
    ) -> FormatRegistry {
        let migrators = [FormatMigrator {
            kind,
            from: FormatVersion::V1,
            to: FormatVersion::new(1, 1),
            reversible,
            forward: append_marker,
            backward: if reversible { Some(strip_marker) } else { None },
            no_body_change: false,
            description: "appends the v2 marker",
        }];
        let fixtures = [FormatFixture {
            kind,
            version: FormatVersion::V1,
            bytes: b"",
            path: "fixtures/v1.bin",
        }];
        FormatRegistry::from_parts(&registrations(kind, policy), &migrators, &fixtures)
            .expect("loads")
    }

    #[test]
    fn test_current_version_dispatches_to_the_current_reader() {
        let registry = registry_with(FormatKind::HeapPage, MigrationPolicy::Lazy, true);
        let bytes = envelope::encode(FormatKind::HeapPage, FormatVersion::new(1, 1), b"body2");
        let opened = open(&registry, &bytes).expect("opens");
        assert_eq!(opened.path, ReaderPath::Current);
        assert_eq!(opened.body.as_ref(), b"body2");
        assert!(!opened.should_write_back(true));
    }

    #[test]
    fn test_old_version_dispatches_to_its_reader_and_migrates() {
        let registry = registry_with(FormatKind::HeapPage, MigrationPolicy::Lazy, true);
        let bytes = envelope::encode(FormatKind::HeapPage, FormatVersion::V1, b"body");
        let opened = open(&registry, &bytes).expect("opens");
        assert!(opened.path.needs_migration());
        assert_eq!(
            opened.path.dispatched_version(FormatVersion::new(1, 1)),
            FormatVersion::V1
        );
        assert_eq!(opened.body.as_ref(), b"body2");
        assert!(opened.should_write_back(true));
        assert!(!opened.should_write_back(false));
    }

    #[test]
    fn test_eager_writes_back_without_a_modification() {
        let registry = registry_with(FormatKind::HeapPage, MigrationPolicy::Eager, true);
        let bytes = envelope::encode(FormatKind::HeapPage, FormatVersion::V1, b"body");
        let opened = open(&registry, &bytes).expect("opens");
        assert!(opened.should_write_back(false));
        let reencoded = opened.reencode();
        let reopened = open(&registry, &reencoded).expect("opens");
        assert_eq!(reopened.path, ReaderPath::Current);
        assert_eq!(reopened.body.as_ref(), b"body2");
    }

    #[test]
    fn test_coexist_never_writes_back() {
        let registry = registry_with(FormatKind::HeapPage, MigrationPolicy::Coexist, true);
        let bytes = envelope::encode(FormatKind::HeapPage, FormatVersion::V1, b"body");
        let opened = open(&registry, &bytes).expect("opens");
        assert!(!opened.should_write_back(true));
        assert_eq!(opened.body.as_ref(), b"body2");
    }

    #[test]
    fn test_unknown_version_fails_closed_naming_the_version() {
        let registry = registry_with(FormatKind::HeapPage, MigrationPolicy::Lazy, true);
        let bytes = envelope::encode(FormatKind::HeapPage, FormatVersion::new(99, 0), b"body");
        let err = open(&registry, &bytes).expect_err("refuses");
        let text = err.to_string();
        assert!(text.contains("99.0"), "{text}");
        assert!(text.contains("Upgrade through"), "{text}");
    }

    #[test]
    fn test_backward_migration_undoes_a_reversible_step() {
        let registry = registry_with(FormatKind::HeapPage, MigrationPolicy::Lazy, true);
        let entry = registry.get(FormatKind::HeapPage).expect("registered");
        let back =
            migrate_body_backward(entry, FormatVersion::new(1, 1), FormatVersion::V1, b"body2")
                .expect("undoes");
        assert_eq!(back.body, b"body");
    }

    #[test]
    fn test_backward_migration_refuses_a_one_way_step() {
        let registry = registry_with(FormatKind::HeapPage, MigrationPolicy::Lazy, false);
        let entry = registry.get(FormatKind::HeapPage).expect("registered");
        let err =
            migrate_body_backward(entry, FormatVersion::new(1, 1), FormatVersion::V1, b"body2")
                .expect_err("refuses");
        assert!(err.to_string().contains("one way"), "{err}");
    }

    #[test]
    fn test_open_as_refuses_a_foreign_file() {
        let registry = registry_with(FormatKind::HeapPage, MigrationPolicy::Lazy, true);
        let bytes = envelope::encode(FormatKind::Fsm, FormatVersion::V1, b"body");
        assert!(open_as(&registry, &bytes, FormatKind::HeapPage).is_err());
    }

    #[test]
    fn test_progress_tracks_percent_and_eta() {
        let progress = MigrationProgress::new(
            FormatKind::HeapPage,
            FormatVersion::V1,
            FormatVersion::new(1, 1),
            MigrationPolicy::Eager,
            1_000,
        );
        progress.set_totals(10, 1_000);
        assert_eq!(progress.percent_complete(), 0.0);
        assert_eq!(progress.eta_secs(1_000), None);
        for _ in 0..5 {
            progress.record_file(100);
        }
        assert_eq!(progress.percent_complete(), 50.0);
        assert_eq!(progress.eta_secs(1_005), Some(5));
        assert_eq!(progress.bytes_remaining(), 500);
        progress.finish(1_010);
        assert!(progress.is_finished());
    }

    #[test]
    fn test_board_collects_runs_and_pauses_them() {
        let board = MigrationBoard::new();
        let run = board.start(MigrationProgress::new(
            FormatKind::Fsm,
            FormatVersion::V1,
            FormatVersion::new(1, 1),
            MigrationPolicy::Eager,
            0,
        ));
        assert_eq!(board.runs().len(), 1);
        assert_eq!(board.runs_for(FormatKind::Fsm).len(), 1);
        assert!(board.runs_for(FormatKind::HeapPage).is_empty());
        board.set_paused(true);
        assert!(run.is_paused());
    }
}
