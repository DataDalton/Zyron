//! Format, signature, and upgrade substrate.
//!
//! Every persistent thing Zyron writes carries an envelope naming its format
//! kind and version, every signed artifact carries the identity of the
//! scheme that signed it, every catalog table carries a schema version, and
//! every user-authored SQL object can be rewritten forward by a registered
//! transform. The registries below are what makes each of those a lookup
//! rather than an assumption.
//!
//! The model is cutover plus automatic migration. On upgrade the substrate
//! moves existing state to the new form; once the window closes the old path
//! is deleted rather than carried as a shim. The release check refuses a
//! breaking change without its migrator, fixture, and registry entry, and
//! refuses reader code that has outlived its retirement date

pub mod catalog_evolution;
pub mod deprecation;
pub mod envelope;
pub mod kind;
pub mod migration;
pub mod record_version;
pub mod registry;
pub mod reserved;
pub mod rewrite;
pub mod scan_migration;
pub mod scheme;
pub mod stamp;
pub mod text_envelope;
pub mod upgrade;
pub mod version;
pub mod wire_version;

pub use catalog_evolution::{
    CatalogEvolutionError, CatalogSchemaEvolution, CatalogSchemaRegistry, CatalogTableEvolution,
    CatalogTableRegistration, RowMigrateFn,
};
pub use deprecation::{
    BinaryVersion, DEFAULT_WARNING_RATE_LIMIT_PER_HOUR, DeprecatedItemKind, DeprecationRecord,
    DeprecationRegistry, DeprecationRegistryError, DeprecationStage, EmittedWarning,
    MigrationGuide, WarningLog, WarningRateLimiter, WarningReportRow,
};
pub use envelope::{
    ENVELOPE_FOOTER_LEN, ENVELOPE_HEADER_LEN, ENVELOPE_MIN_LEN, ENVELOPE_PEEK_LEN, Envelope,
    EnvelopeError, EnvelopeHeader,
};
pub use kind::{ALL_FORMAT_KINDS, FormatKind, Framing, MAGIC_ALLOCATIONS, MagicAllocation};
pub use migration::{
    MigratedBody, MigrationBoard, MigrationError, MigrationProgress, OpenedFile, ReaderPath,
};

pub use record_version::{
    RECORD_VERSION_ABSENT, RECORD_VERSION_MAX_INLINE, RECORD_VERSION_WIDE_ESCAPE, RecordTag,
    RecordVersion, WideRecordVersion,
};
pub use registry::{
    DeprecationStatus, FormatEntry, FormatFixture, FormatMigrator, FormatRegistration,
    FormatRegistry, MigrateFn, MigrationPolicy, RegistryError,
};
pub use rewrite::{
    ALL_OBJECT_KINDS, ObjectKind, RewriteCategory, RewriteClassification, RewriteDisposition,
    RewriteStatus, UserObjectRewritePolicy,
};
pub use scan_migration::{
    CatalogScanMigration, FormatDocumentation, catalog_scan_for, catalog_scan_migrations,
    documentation_for, downgrade_refusal,
};
pub use scheme::{
    ALL_ARTIFACT_KINDS, ArtifactKind, ArtifactSchemeBinding, SchemeCategory, SchemeError, SchemeId,
    SchemeIdentifierEncoding, SchemeRegistry, SchemeStatus, SignatureSchemeRegistration,
};
pub use stamp::{FORMAT_STAMP_LEN, FormatStamp};
pub use text_envelope::{TEXT_ENVELOPE_SECTION, TextEnvelopeError};
pub use upgrade::{
    HealthBaseline, HealthThreshold, HealthVerdict, MaintenanceSchedule, MaintenanceWindow,
    NodeUpgradeState, ReleaseEntry, ReleaseManifest, RewriteRecord, UpgradeBoard, UpgradeChannel,
    UpgradeHistoryEntry, UpgradeOutcome, UpgradePhase, UpgradeSettings,
};
pub use version::{FormatVersion, VersionWindow};
pub use wire_version::{
    WireProtocol, WireProtocolVersion, WireVersionError, WireVersionRegistry, WireVersionStatus,
};

use crate::error::{Result, ZyronError};

/// Everything the substrate loads at startup, in one place.
///
/// A server builds this before it opens a data directory. A failure here is
/// fatal, because a binary that cannot say which versions it reads has no
/// safe way to open anything
pub struct FormatSubstrate {
    pub formats: FormatRegistry,
    pub schemes: SchemeRegistry,
    pub catalog_schemas: CatalogSchemaRegistry,
    pub deprecations: DeprecationRegistry,
    pub wire_versions: WireVersionRegistry,
    pub migrations: MigrationBoard,
    pub warning_limiter: WarningRateLimiter,
    pub warning_log: WarningLog,
}

impl FormatSubstrate {
    /// Collects every registration and checks the whole set
    pub fn load() -> Result<FormatSubstrate> {
        let formats =
            FormatRegistry::load().map_err(|e| ZyronError::FormatRegistry(e.to_string()))?;
        let catalog_schemas =
            CatalogSchemaRegistry::load().map_err(|e| ZyronError::FormatRegistry(e.to_string()))?;
        let deprecations =
            DeprecationRegistry::load().map_err(|e| ZyronError::FormatRegistry(e.to_string()))?;
        Ok(FormatSubstrate {
            formats,
            schemes: SchemeRegistry::load(),
            catalog_schemas,
            deprecations,
            wire_versions: WireVersionRegistry::load(),
            migrations: MigrationBoard::new(),
            warning_limiter: WarningRateLimiter::default(),
            warning_log: WarningLog::default(),
        })
    }

    /// Refuses a substrate that does not cover every format kind, which is
    /// the gate a server passes before it opens a data directory
    pub fn verify_complete(&self) -> Result<()> {
        self.formats
            .verify_complete()
            .map_err(|e| ZyronError::FormatRegistry(e.to_string()))
    }

    /// What using a deprecated item does right now.
    ///
    /// The stage is decided by the running version, and a warning is rate
    /// limited per item per tenant so a loop over a deprecated call does not
    /// fill the log with one sentence. The use is counted either way, so the
    /// report shows what the limiter hid
    pub fn check_deprecated_use(
        &self,
        item_id: &str,
        tenant: &str,
        now_secs: u64,
    ) -> DeprecationOutcome {
        let Some(record) = self.deprecations.find(item_id) else {
            return DeprecationOutcome::Allowed;
        };
        let running = BinaryVersion::parse(env!("CARGO_PKG_VERSION")).unwrap_or_default();
        match record.stage(running) {
            DeprecationStage::Supported => DeprecationOutcome::Allowed,
            DeprecationStage::Warning => {
                let message = record.guidance();
                if !self.warning_limiter.allow(item_id, tenant, now_secs) {
                    return DeprecationOutcome::AllowedSilently;
                }
                self.warning_log.record(EmittedWarning {
                    item_id: record.item_id.to_string(),
                    item_kind: record.item_kind,
                    tenant: tenant.to_string(),
                    at_secs: now_secs,
                    message: message.clone(),
                });
                DeprecationOutcome::Warned(message)
            }
            DeprecationStage::Erroring => DeprecationOutcome::Refused(record.guidance()),
            // After removal the item is gone from the parser and the
            // runtime, so nothing can reach this with a live use. It is
            // reachable through a support query against the registry, which
            // is why the record is kept forever
            DeprecationStage::Removed => DeprecationOutcome::Removed(record.guidance()),
        }
    }

    /// Scans a statement's text for every deprecated item that is a SQL
    /// surface, and reports what each use does.
    ///
    /// Text matching is what a scan over a parsed statement cannot do: a
    /// deprecated item may be a keyword the current parser no longer has a
    /// node for, and the point of the warn window is that the use still
    /// parses. A match is on a whole word so `pipeline_runs` does not trip
    /// a deprecation of `PIPELINE`
    pub fn scan_sql_for_deprecations(
        &self,
        sql: &str,
        tenant: &str,
        now_secs: u64,
    ) -> Vec<(&'static str, DeprecationOutcome)> {
        // This runs on every statement, so it does no work at all when there
        // is nothing to look for. Uppercasing the statement to search it for
        // an empty set of items is the entire cost of the scan in the common
        // case, and the registry is small enough that checking it is free
        let records = self.deprecations.records();
        if !records
            .iter()
            .any(|record| record.item_kind.is_user_authored_sql())
        {
            return Vec::new();
        }
        let upper = sql.to_ascii_uppercase();
        let mut found = Vec::new();
        for record in records {
            if !record.item_kind.is_user_authored_sql() {
                continue;
            }
            // `upper` is already uppercase, so the needle is matched
            // case-insensitively rather than allocating a copy of every
            // item id on every statement
            if !contains_word_ignore_case(&upper, record.item_id) {
                continue;
            }
            let outcome = self.check_deprecated_use(record.item_id, tenant, now_secs);
            if !matches!(outcome, DeprecationOutcome::Allowed) {
                found.push((record.item_id, outcome));
            }
        }
        found
    }

    /// Format kinds whose writer is a live subsystem rather than a reserved
    /// magic
    pub fn live_format_kinds(&self) -> Vec<FormatKind> {
        ALL_FORMAT_KINDS
            .iter()
            .copied()
            .filter(|kind| !reserved::RESERVED_KINDS.contains(kind))
            .collect()
    }
}

/// What a use of a deprecated item does
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DeprecationOutcome {
    /// The item is not deprecated, or is not yet in its warn window
    Allowed,
    /// In the warn window, but this tenant has already had its warnings for
    /// this item this hour
    AllowedSilently,
    /// In the warn window, and this use is warned about
    Warned(String),
    /// Past the warn window, the use is an error
    Refused(String),
    /// Past removal. Nothing live can reach this, the registry keeps the
    /// record so a support question has an answer
    Removed(String),
}

impl DeprecationOutcome {
    /// Whether the use may proceed
    pub fn permitted(&self) -> bool {
        matches!(
            self,
            DeprecationOutcome::Allowed
                | DeprecationOutcome::AllowedSilently
                | DeprecationOutcome::Warned(_)
        )
    }

    /// The message, when there is one
    pub fn message(&self) -> Option<&str> {
        match self {
            DeprecationOutcome::Warned(text)
            | DeprecationOutcome::Refused(text)
            | DeprecationOutcome::Removed(text) => Some(text),
            _ => None,
        }
    }
}

/// Whether a haystack holds a needle as a whole word.
///
/// Both are already upper case at the call site. A word boundary is
/// anything that is not alphanumeric or an underscore, which is what keeps
/// a deprecation of `PIPELINE` off `pipeline_runs`
/// `contains_word` where the haystack is already uppercase and the needle is
/// matched without regard to case.
///
/// Avoids allocating an uppercased copy of the needle on a path that runs
/// once per registered item per statement
fn contains_word_ignore_case(upper_haystack: &str, needle: &str) -> bool {
    if needle.is_empty() {
        return false;
    }
    let bytes = upper_haystack.as_bytes();
    let needle_bytes = needle.as_bytes();
    let len = needle_bytes.len();
    if len > bytes.len() {
        return false;
    }
    for start in 0..=(bytes.len() - len) {
        let matches = bytes[start..start + len]
            .iter()
            .zip(needle_bytes)
            .all(|(h, n)| *h == n.to_ascii_uppercase());
        if !matches {
            continue;
        }
        let end = start + len;
        let before_ok = start == 0 || !is_word_byte(bytes[start - 1]);
        let after_ok = end >= bytes.len() || !is_word_byte(bytes[end]);
        if before_ok && after_ok {
            return true;
        }
    }
    false
}

#[inline]
fn is_word_byte(byte: u8) -> bool {
    byte.is_ascii_alphanumeric() || byte == b'_'
}

/// The process-wide substrate, loaded once
static SUBSTRATE: std::sync::OnceLock<FormatSubstrate> = std::sync::OnceLock::new();

/// The substrate for this process, loading it on first use.
///
/// Every registration is compiled in, so loading is pure and deterministic
/// and two threads racing to load produce the same thing. The server loads
/// it explicitly at startup so a bad registration set is a startup failure
/// rather than a failure at the first read
pub fn substrate() -> Result<&'static FormatSubstrate> {
    if let Some(loaded) = SUBSTRATE.get() {
        return Ok(loaded);
    }
    let loaded = FormatSubstrate::load()?;
    Ok(SUBSTRATE.get_or_init(|| loaded))
}

/// The node's upgrade board, which the orchestrator writes and the catalog
/// views, the CLI, and the DDL surface read
static UPGRADE_BOARD: std::sync::OnceLock<upgrade::UpgradeBoard> = std::sync::OnceLock::new();

/// The upgrade board for this process
pub fn upgrade_board() -> &'static upgrade::UpgradeBoard {
    UPGRADE_BOARD.get_or_init(upgrade::UpgradeBoard::new)
}

impl From<EnvelopeError> for ZyronError {
    fn from(value: EnvelopeError) -> Self {
        ZyronError::FormatEnvelope(value.to_string())
    }
}

impl From<MigrationError> for ZyronError {
    fn from(value: MigrationError) -> Self {
        ZyronError::FormatMigration(value.to_string())
    }
}

impl From<RegistryError> for ZyronError {
    fn from(value: RegistryError) -> Self {
        ZyronError::FormatRegistry(value.to_string())
    }
}

impl From<SchemeError> for ZyronError {
    fn from(value: SchemeError) -> Self {
        ZyronError::SignatureScheme(value.to_string())
    }
}

impl From<WireVersionError> for ZyronError {
    fn from(value: WireVersionError) -> Self {
        ZyronError::WireProtocolVersion(value.to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::format::deprecation::{DeprecatedItemKind, DeprecationRecord};

    /// A substrate holding one deprecated item at each stage, so emission is
    /// exercised without a release having to deprecate anything
    fn substrate_with(records: Vec<DeprecationRecord>) -> FormatSubstrate {
        FormatSubstrate {
            formats: FormatRegistry::from_parts(&[], &[], &[]).expect("loads"),
            schemes: SchemeRegistry::from_parts(Vec::new(), Vec::new()),
            catalog_schemas: CatalogSchemaRegistry::from_parts(&[], &[]).expect("loads"),
            deprecations: DeprecationRegistry::from_records(records).expect("loads"),
            wire_versions: WireVersionRegistry::from_versions(Vec::new()),
            migrations: MigrationBoard::new(),
            warning_limiter: WarningRateLimiter::new(2),
            warning_log: WarningLog::default(),
        }
    }

    fn record(
        item_id: &'static str,
        deprecated: &'static str,
        error_since: &'static str,
        removed: &'static str,
    ) -> DeprecationRecord {
        DeprecationRecord {
            item_kind: DeprecatedItemKind::DdlKeyword,
            item_id,
            deprecated_since_version: deprecated,
            warn_until_version: error_since,
            error_since_version: error_since,
            removed_since_version: removed,
            replacement_ref: Some("CREATE WORKFLOW"),
            migration_guide_url: Some("/docs/sql/workflows.md"),
            no_guide_required: false,
            summary: "pipelines became workflows",
            before_example: "CREATE PIPELINE p",
            after_example: "CREATE WORKFLOW w",
            migration_snippet: "SELECT 1",
            common_pitfall: "a stage is not a task",
        }
    }

    #[test]
    fn test_an_item_before_its_window_is_allowed_silently() {
        let substrate = substrate_with(vec![record(
            "CREATE PIPELINE",
            "99.0.0",
            "99.1.0",
            "99.2.0",
        )]);
        assert_eq!(
            substrate.check_deprecated_use("CREATE PIPELINE", "t", 0),
            DeprecationOutcome::Allowed
        );
    }

    #[test]
    fn test_an_item_in_its_warn_window_warns_then_goes_quiet() {
        let substrate =
            substrate_with(vec![record("CREATE PIPELINE", "0.0.1", "99.0.0", "99.1.0")]);
        // The limiter allows two per hour in this substrate
        assert!(matches!(
            substrate.check_deprecated_use("CREATE PIPELINE", "t", 0),
            DeprecationOutcome::Warned(_)
        ));
        assert!(matches!(
            substrate.check_deprecated_use("CREATE PIPELINE", "t", 0),
            DeprecationOutcome::Warned(_)
        ));
        assert_eq!(
            substrate.check_deprecated_use("CREATE PIPELINE", "t", 0),
            DeprecationOutcome::AllowedSilently,
            "the use still works, it is only the warning that stops"
        );
        assert_eq!(substrate.warning_log.report(0).len(), 1);
        assert_eq!(substrate.warning_log.report(0)[0].count, 2);
        assert_eq!(substrate.warning_limiter.suppressed(), 1);
    }

    #[test]
    fn test_the_limiter_budgets_per_tenant() {
        let substrate =
            substrate_with(vec![record("CREATE PIPELINE", "0.0.1", "99.0.0", "99.1.0")]);
        for _ in 0..5 {
            substrate.check_deprecated_use("CREATE PIPELINE", "tenant-a", 0);
        }
        assert!(matches!(
            substrate.check_deprecated_use("CREATE PIPELINE", "tenant-b", 0),
            DeprecationOutcome::Warned(_)
        ));
        let report = substrate.warning_log.report(0);
        assert_eq!(report.len(), 1);
        assert_eq!(report[0].tenants.len(), 2);
    }

    #[test]
    fn test_an_item_past_its_warn_window_is_refused() {
        let substrate = substrate_with(vec![record("CREATE PIPELINE", "0.0.1", "0.0.2", "99.0.0")]);
        let outcome = substrate.check_deprecated_use("CREATE PIPELINE", "t", 0);
        assert!(!outcome.permitted());
        assert!(
            outcome
                .message()
                .expect("carries guidance")
                .contains("CREATE WORKFLOW")
        );
    }

    #[test]
    fn test_a_scan_finds_a_deprecated_keyword_on_a_word_boundary() {
        let substrate =
            substrate_with(vec![record("CREATE PIPELINE", "0.0.1", "99.0.0", "99.1.0")]);
        let found = substrate.scan_sql_for_deprecations(
            "create pipeline p AS (STAGE s AS SELECT 1)",
            "t",
            0,
        );
        assert_eq!(found.len(), 1);
        assert_eq!(found[0].0, "CREATE PIPELINE");

        // A name that merely contains the word does not trip it
        let clean = substrate.scan_sql_for_deprecations(
            "SELECT * FROM zyron_sys.stat.pipeline_runs",
            "t",
            0,
        );
        assert!(clean.is_empty(), "{clean:?}");
    }

    #[test]
    fn test_word_matching_respects_boundaries() {
        assert!(contains_word_ignore_case(
            "CREATE PIPELINE P",
            "CREATE PIPELINE"
        ));
        assert!(contains_word_ignore_case(
            "X CREATE PIPELINE",
            "CREATE PIPELINE"
        ));
        assert!(!contains_word_ignore_case("PIPELINE_RUNS", "PIPELINE"));
        assert!(!contains_word_ignore_case("MYPIPELINE", "PIPELINE"));
        assert!(contains_word_ignore_case("A.PIPELINE.B", "PIPELINE"));
        assert!(!contains_word_ignore_case("", "PIPELINE"));
        assert!(!contains_word_ignore_case("PIPELINE", ""));
    }

    /// The allocation-free matcher agrees with the one it replaced on every
    /// boundary case, and additionally ignores the needle's case
    #[test]
    fn test_case_insensitive_word_matching_agrees() {
        let cases = [
            ("CREATE PIPELINE P", "CREATE PIPELINE", true),
            ("X CREATE PIPELINE", "CREATE PIPELINE", true),
            ("PIPELINE_RUNS", "PIPELINE", false),
            ("MYPIPELINE", "PIPELINE", false),
            ("A.PIPELINE.B", "PIPELINE", true),
            ("", "PIPELINE", false),
            ("PIPELINE", "", false),
        ];
        for (haystack, needle, expected) in cases {
            assert_eq!(
                contains_word_ignore_case(haystack, needle),
                expected,
                "{haystack:?} vs {needle:?}"
            );
        }
        // The haystack is uppercase by construction, the needle need not be
        assert!(contains_word_ignore_case("DROP PIPELINE X", "pipeline"));
        assert!(contains_word_ignore_case("DROP PIPELINE X", "PiPeLiNe"));
        assert!(!contains_word_ignore_case("MYPIPELINE", "pipeline"));
    }

    #[test]
    fn test_errors_convert_into_zyron_errors() {
        let envelope: ZyronError = EnvelopeError::UnknownMagic { magic: *b"QQQQ" }.into();
        assert!(envelope.to_string().contains("QQQQ"));
        let registry: ZyronError = RegistryError::Missing {
            kind: FormatKind::HeapPage,
        }
        .into();
        assert!(registry.to_string().contains("heap_page"));
        let scheme: ZyronError = SchemeError::UnknownScheme {
            named: "HS256".to_string(),
        }
        .into();
        assert!(scheme.to_string().contains("HS256"));
    }
}
