//! The format registry.
//!
//! Each format module submits one `FormatRegistration` describing what it
//! writes today, what it can read, and how files behind the current version
//! move forward. The registrations are collected once at startup into a
//! dense table indexed by format kind, so a lookup on the read path is an
//! array index rather than a map probe.
//!
//! A registration is data, not behavior. The transformations live in
//! `FormatMigrator` submissions beside it, and the corpus that proves them
//! lives in `FormatFixture` submissions beside those. The startup check and
//! the release check both read the same three collections, so a format that
//! passes one passes the other

use std::fmt;

use super::kind::{ALL_FORMAT_KINDS, FormatKind};
use super::version::{FormatVersion, VersionWindow};

/// What happens to files still on an older version after an upgrade
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MigrationPolicy {
    /// A budgeted background sweep rewrites them
    Eager,
    /// They are rewritten the next time they are modified
    Lazy,
    /// Both versions stay valid indefinitely, nothing is rewritten
    Coexist,
}

impl MigrationPolicy {
    pub const fn label(self) -> &'static str {
        match self {
            MigrationPolicy::Eager => "eager",
            MigrationPolicy::Lazy => "lazy",
            MigrationPolicy::Coexist => "coexist",
        }
    }

    pub fn parse(text: &str) -> Option<MigrationPolicy> {
        match text.to_ascii_lowercase().as_str() {
            "eager" => Some(MigrationPolicy::Eager),
            "lazy" => Some(MigrationPolicy::Lazy),
            "coexist" => Some(MigrationPolicy::Coexist),
            _ => None,
        }
    }
}

impl fmt::Display for MigrationPolicy {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.label())
    }
}

/// Where a format version sits in its lifecycle
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DeprecationStatus {
    /// Written by the current binary
    Active,
    /// Still read, no longer written, scheduled for reader removal
    Deprecating,
    /// Reader removal is due, the release check fails while its code is in
    /// the tree
    Retired,
}

impl DeprecationStatus {
    pub const fn label(self) -> &'static str {
        match self {
            DeprecationStatus::Active => "active",
            DeprecationStatus::Deprecating => "deprecating",
            DeprecationStatus::Retired => "retired",
        }
    }
}

impl fmt::Display for DeprecationStatus {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.label())
    }
}

/// One format's declaration of what it writes and reads
#[derive(Debug, Clone, Copy)]
pub struct FormatRegistration {
    pub kind: FormatKind,
    /// The version every new file of this kind is written at
    pub writer_current_version: FormatVersion,
    /// The inclusive span of versions this binary carries readers for
    pub reader_supported_versions: VersionWindow,
    pub migration_policy: MigrationPolicy,
    /// Whether every migration inside the reader window can be undone
    pub migration_reversible: bool,
    /// The Zyron version that introduced `writer_current_version`
    pub binary_version_gate: &'static str,
    pub deprecation_status: DeprecationStatus,
    /// ISO 8601 date the oldest supported reader is deleted on, or None
    /// while the format has only ever had one version
    pub retirement_date: Option<&'static str>,
    /// Whether a writer can emit an older version on request, which backup
    /// export for an older cluster needs
    pub downgrade_write_supported: bool,
    /// One line naming what changed at the current version
    pub notes: &'static str,
}

inventory::collect!(FormatRegistration);

/// A transformation from one format version to the next.
///
/// For an envelope framed format both directions take and return the
/// envelope body. The substrate re-wraps the result, so a migrator never
/// touches magic, version, or checksums. For a format that owns its
/// trailer they take and return the whole file, header and trailer
/// included, because the format's own checksums cover the header and only
/// its writer can stamp them
pub type MigrateFn = fn(&[u8]) -> Result<Vec<u8>, String>;

#[derive(Debug, Clone, Copy)]
pub struct FormatMigrator {
    pub kind: FormatKind,
    pub from: FormatVersion,
    pub to: FormatVersion,
    /// Whether `backward` is present and total
    pub reversible: bool,
    pub forward: MigrateFn,
    pub backward: Option<MigrateFn>,
    /// True when the version bump changes nothing in the body, which is how
    /// a bump that only widens a header flag declares itself rather than
    /// shipping an identity function nobody can tell from a mistake
    pub no_body_change: bool,
    pub description: &'static str,
}

/// Two migrators are the same when they cover the same step of the same
/// format with the same reversibility. The function pointers are deliberately
/// not compared: their addresses carry no meaning across codegen units, and
/// a step is identified by where it moves a body from and to
impl PartialEq for FormatMigrator {
    fn eq(&self, other: &Self) -> bool {
        self.kind == other.kind
            && self.from == other.from
            && self.to == other.to
            && self.reversible == other.reversible
            && self.no_body_change == other.no_body_change
            && self.description == other.description
    }
}

impl Eq for FormatMigrator {}

inventory::collect!(FormatMigrator);

/// A recorded file of one format version, kept so every reader in the window
/// is exercised against bytes an older writer actually produced
#[derive(Debug, Clone, Copy)]
pub struct FormatFixture {
    pub kind: FormatKind,
    pub version: FormatVersion,
    /// The complete envelope, header and body and footer
    pub bytes: &'static [u8],
    /// Where the fixture lives, printed by the release check
    pub path: &'static str,
}

inventory::collect!(FormatFixture);

/// Why the registry refused to load
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RegistryError {
    /// A format kind submitted no registration
    Missing { kind: FormatKind },
    /// Two registrations name the same format kind
    Duplicate { kind: FormatKind },
    /// The writer's current version is outside the reader window, so the
    /// binary cannot read what it writes
    WriterOutsideWindow {
        kind: FormatKind,
        writer: FormatVersion,
        window: VersionWindow,
    },
    /// The window is inverted
    InvertedWindow {
        kind: FormatKind,
        window: VersionWindow,
    },
    /// A version step inside the reader window has no migrator and is not
    /// marked as needing none
    MissingMigrator {
        kind: FormatKind,
        from: FormatVersion,
        to: FormatVersion,
    },
    /// Two migrators cover the same step
    DuplicateMigrator {
        kind: FormatKind,
        from: FormatVersion,
        to: FormatVersion,
    },
    /// A migrator claims to be reversible but carries no backward function
    ReversibleWithoutBackward {
        kind: FormatKind,
        from: FormatVersion,
        to: FormatVersion,
    },
    /// A migrator steps between versions that are not adjacent
    NonAdjacentMigrator {
        kind: FormatKind,
        from: FormatVersion,
        to: FormatVersion,
    },
    /// A step marked as changing nothing in the body, on a format whose
    /// checksums the substrate cannot restamp
    RestampOnOwnTrailer {
        kind: FormatKind,
        from: FormatVersion,
        to: FormatVersion,
    },
    /// A supported version behind the current one has no fixture
    MissingFixture {
        kind: FormatKind,
        version: FormatVersion,
    },
    /// A registration declares more than one version but no retirement date
    /// for the oldest reader
    MissingRetirementDate { kind: FormatKind },
    /// A registration's retirement date is not an ISO 8601 calendar date
    BadRetirementDate {
        kind: FormatKind,
        value: &'static str,
    },
}

impl fmt::Display for RegistryError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            RegistryError::Missing { kind } => write!(
                f,
                "format `{kind}` has no registration. Every format kind must submit a \
                 FormatRegistration naming its writer_current_version and \
                 reader_supported_versions before the server can start"
            ),
            RegistryError::Duplicate { kind } => {
                write!(f, "format `{kind}` submitted more than one registration")
            }
            RegistryError::WriterOutsideWindow {
                kind,
                writer,
                window,
            } => write!(
                f,
                "format `{kind}` writes version {writer} but its reader window is {window}, \
                 so the binary cannot read what it writes"
            ),
            RegistryError::InvertedWindow { kind, window } => write!(
                f,
                "format `{kind}` declares reader window {window}, whose oldest version is \
                 newer than its newest"
            ),
            RegistryError::MissingMigrator { kind, from, to } => write!(
                f,
                "format `{kind}` has no migrator for {from} to {to}. Add \
                 migrations/v{}_{}_to_v{}_{}.rs beside the format and submit a \
                 FormatMigrator, or mark the step no_body_change when the bump changes \
                 nothing on disk",
                from.major, from.minor, to.major, to.minor
            ),
            RegistryError::DuplicateMigrator { kind, from, to } => {
                write!(f, "format `{kind}` has two migrators for {from} to {to}")
            }
            RegistryError::ReversibleWithoutBackward { kind, from, to } => write!(
                f,
                "format `{kind}` migrator {from} to {to} is marked reversible but carries \
                 no backward function"
            ),
            RegistryError::NonAdjacentMigrator { kind, from, to } => write!(
                f,
                "format `{kind}` migrator steps {from} to {to}, which are not adjacent \
                 versions. Migrators move one version at a time and the substrate chains them"
            ),
            RegistryError::RestampOnOwnTrailer { kind, from, to } => write!(
                f,
                "format `{kind}` migrator {from} to {to} is marked no_body_change, but a \
                 {kind} file owns its trailer and its checksums cover the header the \
                 substrate would restamp. Give the step a forward function that rewrites \
                 the file whole"
            ),
            RegistryError::MissingFixture { kind, version } => write!(
                f,
                "format `{kind}` supports reading version {version} but ships no fixture \
                 for it. Add fixtures/v{}_{}.bin beside the format, written by the release \
                 that wrote that version, and submit a FormatFixture",
                version.major, version.minor
            ),
            RegistryError::MissingRetirementDate { kind } => write!(
                f,
                "format `{kind}` carries more than one reader version but declares no \
                 retirement_date for the oldest"
            ),
            RegistryError::BadRetirementDate { kind, value } => write!(
                f,
                "format `{kind}` declares retirement_date `{value}`, which is not an \
                 ISO 8601 calendar date"
            ),
        }
    }
}

impl std::error::Error for RegistryError {}

/// Everything the substrate knows about one format, gathered from the three
/// collections
#[derive(Debug, Clone)]
pub struct FormatEntry {
    pub registration: FormatRegistration,
    /// Migrators for this kind, sorted by the version they start at
    pub migrators: Vec<FormatMigrator>,
    /// Fixtures for this kind, sorted by version
    pub fixtures: Vec<FormatFixture>,
}

impl FormatEntry {
    /// The migrator for one adjacent step, or None when the step is not
    /// registered
    pub fn migrator(&self, from: FormatVersion) -> Option<&FormatMigrator> {
        self.migrators.iter().find(|m| m.from == from)
    }

    /// Whether every step from `from` up to the current writer version can
    /// be undone
    pub fn reversible_from(&self, from: FormatVersion) -> bool {
        let target = self.registration.writer_current_version;
        self.migrators
            .iter()
            .filter(|m| m.from >= from && m.to <= target)
            .all(|m| m.no_body_change || m.reversible)
    }

    /// The chain of migrators that takes a body from `from` to `to`, or an
    /// error naming the first step that is missing
    pub fn plan(
        &self,
        from: FormatVersion,
        to: FormatVersion,
    ) -> Result<Vec<FormatMigrator>, RegistryError> {
        let mut chain = Vec::new();
        let mut cursor = from;
        while cursor < to {
            let step = self.migrators.iter().find(|m| m.from == cursor).ok_or(
                RegistryError::MissingMigrator {
                    kind: self.registration.kind,
                    from: cursor,
                    to,
                },
            )?;
            cursor = step.to;
            chain.push(*step);
        }
        Ok(chain)
    }
}

/// The loaded registry, one dense slot per format kind
#[derive(Debug)]
pub struct FormatRegistry {
    entries: Vec<Option<FormatEntry>>,
}

impl FormatRegistry {
    /// Collects every submitted registration, migrator, and fixture, and
    /// checks the whole set before handing back a registry.
    ///
    /// A failure here is fatal at startup on purpose. A binary that cannot
    /// say which versions it reads has no safe way to open a data directory
    pub fn load() -> Result<FormatRegistry, RegistryError> {
        let registrations: Vec<FormatRegistration> = inventory::iter::<FormatRegistration>
            .into_iter()
            .copied()
            .collect();
        let migrators: Vec<FormatMigrator> = inventory::iter::<FormatMigrator>
            .into_iter()
            .copied()
            .collect();
        let fixtures: Vec<FormatFixture> = inventory::iter::<FormatFixture>
            .into_iter()
            .copied()
            .collect();
        Self::from_parts(&registrations, &migrators, &fixtures)
    }

    /// Builds a registry from explicit parts, which is what the release
    /// check and the tests use so they can feed a deliberately broken set
    pub fn from_parts(
        registrations: &[FormatRegistration],
        migrators: &[FormatMigrator],
        fixtures: &[FormatFixture],
    ) -> Result<FormatRegistry, RegistryError> {
        let mut entries: Vec<Option<FormatEntry>> = vec![None; ALL_FORMAT_KINDS.len()];

        for registration in registrations {
            let slot = &mut entries[registration.kind.index()];
            if slot.is_some() {
                return Err(RegistryError::Duplicate {
                    kind: registration.kind,
                });
            }
            *slot = Some(FormatEntry {
                registration: *registration,
                migrators: Vec::new(),
                fixtures: Vec::new(),
            });
        }

        for migrator in migrators {
            let entry = entries[migrator.kind.index()]
                .as_mut()
                .ok_or(RegistryError::Missing {
                    kind: migrator.kind,
                })?;
            if entry.migrators.iter().any(|m| m.from == migrator.from) {
                return Err(RegistryError::DuplicateMigrator {
                    kind: migrator.kind,
                    from: migrator.from,
                    to: migrator.to,
                });
            }
            if migrator.reversible && migrator.backward.is_none() && !migrator.no_body_change {
                return Err(RegistryError::ReversibleWithoutBackward {
                    kind: migrator.kind,
                    from: migrator.from,
                    to: migrator.to,
                });
            }
            if !is_adjacent(migrator.from, migrator.to) {
                return Err(RegistryError::NonAdjacentMigrator {
                    kind: migrator.kind,
                    from: migrator.from,
                    to: migrator.to,
                });
            }
            entry.migrators.push(*migrator);
        }

        for fixture in fixtures {
            let entry = entries[fixture.kind.index()]
                .as_mut()
                .ok_or(RegistryError::Missing { kind: fixture.kind })?;
            entry.fixtures.push(*fixture);
        }

        for slot in entries.iter_mut().flatten() {
            slot.migrators.sort_by_key(|m| m.from);
            slot.fixtures.sort_by_key(|fx| fx.version);
            validate_entry(slot)?;
        }

        Ok(FormatRegistry { entries })
    }

    /// Refuses a registry that does not cover every format kind.
    ///
    /// Loading collects whatever the linked crates submitted, which is how a
    /// crate's own tests exercise the registry without linking the whole
    /// server. Completeness is a startup gate rather than a load-time one:
    /// a server that cannot say which versions it reads for some format has
    /// no safe way to open a data directory, so it refuses to start
    pub fn verify_complete(&self) -> Result<(), RegistryError> {
        for kind in ALL_FORMAT_KINDS {
            if self.entries[kind.index()].is_none() {
                return Err(RegistryError::Missing { kind: *kind });
            }
        }
        Ok(())
    }

    /// Format kinds with no registration in this binary
    pub fn missing(&self) -> Vec<FormatKind> {
        ALL_FORMAT_KINDS
            .iter()
            .copied()
            .filter(|kind| self.entries[kind.index()].is_none())
            .collect()
    }

    /// The entry for one kind, or None when this binary registered none
    #[inline]
    pub fn get(&self, kind: FormatKind) -> Option<&FormatEntry> {
        self.entries
            .get(kind.index())
            .and_then(|slot| slot.as_ref())
    }

    /// The version new files of this kind are written at
    #[inline]
    pub fn writer_version(&self, kind: FormatKind) -> Option<FormatVersion> {
        self.get(kind)
            .map(|e| e.registration.writer_current_version)
    }

    /// Whether this binary carries a reader for a version
    #[inline]
    pub fn can_read(&self, kind: FormatKind, version: FormatVersion) -> bool {
        self.get(kind)
            .map(|e| e.registration.reader_supported_versions.contains(version))
            .unwrap_or(false)
    }

    /// Every entry, in allocation order
    pub fn entries(&self) -> impl Iterator<Item = &FormatEntry> {
        self.entries.iter().flatten()
    }

    /// How many kinds are loaded
    pub fn len(&self) -> usize {
        self.entries.iter().flatten().count()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

/// Whether `to` is the version immediately after `from`.
///
/// A minor step inside a major line, or the first minor of the next major
/// line, which is what a major consolidation produces
fn is_adjacent(from: FormatVersion, to: FormatVersion) -> bool {
    if to <= from {
        return false;
    }
    if to.major == from.major {
        return to.minor == from.minor + 1;
    }
    to.major == from.major + 1 && to.minor == 0
}

/// Checks one entry against the rules the startup gate and the release check
/// share
fn validate_entry(entry: &FormatEntry) -> Result<(), RegistryError> {
    let registration = &entry.registration;
    let window = registration.reader_supported_versions;
    if window.oldest > window.newest {
        return Err(RegistryError::InvertedWindow {
            kind: registration.kind,
            window,
        });
    }
    if !window.contains(registration.writer_current_version) {
        return Err(RegistryError::WriterOutsideWindow {
            kind: registration.kind,
            writer: registration.writer_current_version,
            window,
        });
    }

    // Every step inside the window needs a migrator, so a file at any
    // supported version can be moved to the current one
    let mut cursor = window.oldest;
    while cursor < registration.writer_current_version {
        let step = entry.migrators.iter().find(|m| m.from == cursor).ok_or(
            RegistryError::MissingMigrator {
                kind: registration.kind,
                from: cursor,
                to: registration.writer_current_version,
            },
        )?;
        // A format that owns its trailer cannot be restamped by the
        // substrate, its checksums cover the header, so a step for it has
        // to rewrite the file
        if step.no_body_change && registration.kind.framing().migrates_whole_file() {
            return Err(RegistryError::RestampOnOwnTrailer {
                kind: registration.kind,
                from: step.from,
                to: step.to,
            });
        }
        cursor = step.to;
    }

    // Every version behind the current one needs a fixture, so its reader is
    // exercised against bytes an older writer produced rather than against
    // bytes the current writer round-tripped
    for version in window.iter() {
        if version >= registration.writer_current_version {
            continue;
        }
        if !entry.fixtures.iter().any(|fx| fx.version == version) {
            return Err(RegistryError::MissingFixture {
                kind: registration.kind,
                version,
            });
        }
    }

    // A format carrying more than one reader must say when the oldest goes
    if window.oldest != window.newest {
        match registration.retirement_date {
            None => {
                return Err(RegistryError::MissingRetirementDate {
                    kind: registration.kind,
                });
            }
            Some(date) if !is_iso_date(date) => {
                return Err(RegistryError::BadRetirementDate {
                    kind: registration.kind,
                    value: date,
                });
            }
            Some(_) => {}
        }
    } else if let Some(date) = registration.retirement_date {
        if !is_iso_date(date) {
            return Err(RegistryError::BadRetirementDate {
                kind: registration.kind,
                value: date,
            });
        }
    }

    Ok(())
}

/// Whether a string is an ISO 8601 calendar date, `YYYY-MM-DD`
pub fn is_iso_date(text: &str) -> bool {
    let bytes = text.as_bytes();
    if bytes.len() != 10 || bytes[4] != b'-' || bytes[7] != b'-' {
        return false;
    }
    let digits_at = |range: std::ops::Range<usize>| bytes[range].iter().all(u8::is_ascii_digit);
    if !digits_at(0..4) || !digits_at(5..7) || !digits_at(8..10) {
        return false;
    }
    let month: u32 = text[5..7].parse().unwrap_or(0);
    let day: u32 = text[8..10].parse().unwrap_or(0);
    (1..=12).contains(&month) && (1..=31).contains(&day)
}

impl FormatKind {
    /// Dense index of this kind, which is its slot in the registry table
    #[inline]
    pub const fn index(self) -> usize {
        self as usize
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn identity(body: &[u8]) -> Result<Vec<u8>, String> {
        Ok(body.to_vec())
    }

    fn base(kind: FormatKind) -> FormatRegistration {
        FormatRegistration {
            kind,
            writer_current_version: FormatVersion::V1,
            reader_supported_versions: VersionWindow::single(FormatVersion::V1),
            migration_policy: MigrationPolicy::Lazy,
            migration_reversible: true,
            binary_version_gate: "0.11.0",
            deprecation_status: DeprecationStatus::Active,
            retirement_date: None,
            downgrade_write_supported: false,
            notes: "test registration",
        }
    }

    fn full_set() -> Vec<FormatRegistration> {
        ALL_FORMAT_KINDS.iter().copied().map(base).collect()
    }

    #[test]
    fn test_index_matches_allocation_order() {
        for (i, kind) in ALL_FORMAT_KINDS.iter().enumerate() {
            assert_eq!(kind.index(), i, "{kind} index does not match its slot");
        }
    }

    #[test]
    fn test_full_set_loads() {
        let registry = FormatRegistry::from_parts(&full_set(), &[], &[]).expect("loads");
        assert_eq!(registry.len(), ALL_FORMAT_KINDS.len());
        assert_eq!(
            registry.writer_version(FormatKind::HeapPage),
            Some(FormatVersion::V1)
        );
        assert!(registry.can_read(FormatKind::HeapPage, FormatVersion::V1));
        assert!(!registry.can_read(FormatKind::HeapPage, FormatVersion::new(9, 0)));
    }

    #[test]
    fn test_a_missing_registration_refuses_the_startup_gate() {
        let mut set = full_set();
        set.retain(|r| r.kind != FormatKind::MvccClog);
        let registry = FormatRegistry::from_parts(&set, &[], &[]).expect("loads what is there");
        assert_eq!(registry.missing(), vec![FormatKind::MvccClog]);
        match registry.verify_complete() {
            Err(RegistryError::Missing { kind }) => {
                assert_eq!(kind, FormatKind::MvccClog);
                let text = RegistryError::Missing { kind }.to_string();
                assert!(text.contains("mvcc_clog"), "{text}");
                assert!(text.contains("before the server can start"), "{text}");
            }
            other => panic!("expected Missing, got {other:?}"),
        }
    }

    #[test]
    fn test_a_complete_set_passes_the_startup_gate() {
        let registry = FormatRegistry::from_parts(&full_set(), &[], &[]).expect("loads");
        assert!(registry.missing().is_empty());
        registry.verify_complete().expect("complete");
    }

    #[test]
    fn test_a_duplicate_registration_refuses_the_load() {
        let mut set = full_set();
        set.push(base(FormatKind::HeapPage));
        assert!(matches!(
            FormatRegistry::from_parts(&set, &[], &[]),
            Err(RegistryError::Duplicate { .. })
        ));
    }

    #[test]
    fn test_a_version_bump_without_a_migrator_refuses_the_load() {
        let mut set = full_set();
        for registration in set.iter_mut() {
            if registration.kind == FormatKind::HeapPage {
                registration.writer_current_version = FormatVersion::new(1, 1);
                registration.reader_supported_versions =
                    VersionWindow::new(FormatVersion::V1, FormatVersion::new(1, 1));
                registration.retirement_date = Some("2027-01-01");
            }
        }
        match FormatRegistry::from_parts(&set, &[], &[]) {
            Err(RegistryError::MissingMigrator { kind, from, .. }) => {
                assert_eq!(kind, FormatKind::HeapPage);
                assert_eq!(from, FormatVersion::V1);
            }
            other => panic!("expected MissingMigrator, got {other:?}"),
        }
    }

    #[test]
    fn test_a_version_bump_without_a_fixture_refuses_the_load() {
        let mut set = full_set();
        for registration in set.iter_mut() {
            if registration.kind == FormatKind::HeapPage {
                registration.writer_current_version = FormatVersion::new(1, 1);
                registration.reader_supported_versions =
                    VersionWindow::new(FormatVersion::V1, FormatVersion::new(1, 1));
                registration.retirement_date = Some("2027-01-01");
            }
        }
        let migrators = [FormatMigrator {
            kind: FormatKind::HeapPage,
            from: FormatVersion::V1,
            to: FormatVersion::new(1, 1),
            reversible: true,
            forward: identity,
            backward: Some(identity),
            no_body_change: false,
            description: "test step",
        }];
        match FormatRegistry::from_parts(&set, &migrators, &[]) {
            Err(RegistryError::MissingFixture { kind, version }) => {
                assert_eq!(kind, FormatKind::HeapPage);
                assert_eq!(version, FormatVersion::V1);
            }
            other => panic!("expected MissingFixture, got {other:?}"),
        }
    }

    #[test]
    fn test_a_non_adjacent_migrator_refuses_the_load() {
        let migrators = [FormatMigrator {
            kind: FormatKind::HeapPage,
            from: FormatVersion::V1,
            to: FormatVersion::new(1, 3),
            reversible: false,
            forward: identity,
            backward: None,
            no_body_change: false,
            description: "skips two",
        }];
        assert!(matches!(
            FormatRegistry::from_parts(&full_set(), &migrators, &[]),
            Err(RegistryError::NonAdjacentMigrator { .. })
        ));
    }

    #[test]
    fn test_reversible_without_backward_refuses_the_load() {
        let migrators = [FormatMigrator {
            kind: FormatKind::HeapPage,
            from: FormatVersion::V1,
            to: FormatVersion::new(1, 1),
            reversible: true,
            forward: identity,
            backward: None,
            no_body_change: false,
            description: "claims reversible",
        }];
        assert!(matches!(
            FormatRegistry::from_parts(&full_set(), &migrators, &[]),
            Err(RegistryError::ReversibleWithoutBackward { .. })
        ));
    }

    #[test]
    fn test_writer_outside_reader_window_refuses_the_load() {
        let mut set = full_set();
        for registration in set.iter_mut() {
            if registration.kind == FormatKind::Fsm {
                registration.writer_current_version = FormatVersion::new(2, 0);
            }
        }
        assert!(matches!(
            FormatRegistry::from_parts(&set, &[], &[]),
            Err(RegistryError::WriterOutsideWindow { .. })
        ));
    }

    #[test]
    fn test_multi_version_window_without_retirement_date_refuses_the_load() {
        let mut set = full_set();
        for registration in set.iter_mut() {
            if registration.kind == FormatKind::Fsm {
                registration.writer_current_version = FormatVersion::new(1, 1);
                registration.reader_supported_versions =
                    VersionWindow::new(FormatVersion::V1, FormatVersion::new(1, 1));
            }
        }
        let migrators = [FormatMigrator {
            kind: FormatKind::Fsm,
            from: FormatVersion::V1,
            to: FormatVersion::new(1, 1),
            reversible: true,
            forward: identity,
            backward: Some(identity),
            no_body_change: false,
            description: "step",
        }];
        let fixtures = [FormatFixture {
            kind: FormatKind::Fsm,
            version: FormatVersion::V1,
            bytes: b"",
            path: "fixtures/v1.bin",
        }];
        assert!(matches!(
            FormatRegistry::from_parts(&set, &migrators, &fixtures),
            Err(RegistryError::MissingRetirementDate { .. })
        ));
    }

    #[test]
    fn test_adjacency_covers_major_consolidation() {
        assert!(is_adjacent(
            FormatVersion::new(1, 0),
            FormatVersion::new(1, 1)
        ));
        assert!(is_adjacent(
            FormatVersion::new(1, 9),
            FormatVersion::new(2, 0)
        ));
        assert!(!is_adjacent(
            FormatVersion::new(1, 0),
            FormatVersion::new(1, 2)
        ));
        assert!(!is_adjacent(
            FormatVersion::new(2, 0),
            FormatVersion::new(1, 0)
        ));
    }

    #[test]
    fn test_iso_date_check() {
        assert!(is_iso_date("2027-01-01"));
        assert!(is_iso_date("2027-12-31"));
        assert!(!is_iso_date("2027-13-01"));
        assert!(!is_iso_date("2027-1-1"));
        assert!(!is_iso_date("not a date"));
    }
}
