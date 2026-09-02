//! The deprecation lifecycle.
//!
//! Anything Zyron takes away announces itself first. A deprecation record
//! names the version the item was deprecated in, the version warnings stop
//! at, the version use becomes an error at, and the version the item leaves
//! the tree at. The parser, the runtime, and the config loader all resolve
//! an item against the running version and act on the stage it lands in.
//!
//! After removal the item is gone from the parser and the runtime, so using
//! it produces an ordinary unknown-token error. The record stays in the
//! registry forever, which is how a support question about a name that no
//! longer parses gets an answer

use std::fmt;
use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};

/// A Zyron release version, `major.minor.patch`
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
pub struct BinaryVersion {
    pub major: u32,
    pub minor: u32,
    pub patch: u32,
}

impl BinaryVersion {
    pub const fn new(major: u32, minor: u32, patch: u32) -> Self {
        Self {
            major,
            minor,
            patch,
        }
    }

    /// Parses `major.minor.patch`, ignoring any pre-release or build suffix
    /// after the patch digits
    pub fn parse(text: &str) -> Option<BinaryVersion> {
        let cleaned = text.trim().trim_start_matches('v');
        let core = cleaned.split(['-', '+']).next().unwrap_or(cleaned);
        let mut parts = core.split('.');
        let major = parts.next()?.parse().ok()?;
        let minor = parts.next()?.parse().ok()?;
        let patch = parts.next().unwrap_or("0").parse().ok()?;
        if parts.next().is_some() {
            return None;
        }
        Some(BinaryVersion {
            major,
            minor,
            patch,
        })
    }

    /// Whether this is a major-version step up from another, which is what
    /// decides the default pre-upgrade backup snapshot
    pub fn is_major_step_from(&self, other: &BinaryVersion) -> bool {
        self.major > other.major
    }

    /// The minor version this many steps ahead, which is what the default
    /// deprecation cadence is expressed in
    pub const fn plus_minor(self, steps: u32) -> BinaryVersion {
        BinaryVersion {
            major: self.major,
            minor: self.minor + steps,
            patch: 0,
        }
    }
}

impl fmt::Display for BinaryVersion {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}.{}.{}", self.major, self.minor, self.patch)
    }
}

/// What kind of thing is being deprecated
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DeprecatedItemKind {
    SqlSyntax,
    DdlKeyword,
    Function,
    FeatureFlag,
    ConfigKey,
    WireMessage,
    FormatVersion,
}

impl DeprecatedItemKind {
    pub const fn label(self) -> &'static str {
        match self {
            DeprecatedItemKind::SqlSyntax => "sql_syntax",
            DeprecatedItemKind::DdlKeyword => "ddl_keyword",
            DeprecatedItemKind::Function => "function",
            DeprecatedItemKind::FeatureFlag => "feature_flag",
            DeprecatedItemKind::ConfigKey => "config_key",
            DeprecatedItemKind::WireMessage => "wire_message",
            DeprecatedItemKind::FormatVersion => "format_version",
        }
    }

    /// Whether an item of this kind is authored by a user in SQL, which is
    /// what decides whether it needs a registered rewriter
    #[inline]
    pub const fn is_user_authored_sql(self) -> bool {
        matches!(
            self,
            DeprecatedItemKind::SqlSyntax
                | DeprecatedItemKind::DdlKeyword
                | DeprecatedItemKind::Function
        )
    }
}

impl fmt::Display for DeprecatedItemKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.label())
    }
}

/// Where an item sits relative to the running version
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DeprecationStage {
    /// Behind the deprecation, nothing is said
    Supported,
    /// Use warns, rate limited per tenant
    Warning,
    /// Use raises an error, the item still parses
    Erroring,
    /// The item is gone from the parser and the runtime
    Removed,
}

impl DeprecationStage {
    pub const fn label(self) -> &'static str {
        match self {
            DeprecationStage::Supported => "supported",
            DeprecationStage::Warning => "warning",
            DeprecationStage::Erroring => "erroring",
            DeprecationStage::Removed => "removed",
        }
    }
}

impl fmt::Display for DeprecationStage {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.label())
    }
}

/// One deprecated item's lifecycle record
#[derive(Debug, Clone, Copy)]
pub struct DeprecationRecord {
    pub item_kind: DeprecatedItemKind,
    /// The name as a user would write it, `CREATE PIPELINE`, `warehouse`,
    /// `wal_v3`
    pub item_id: &'static str,
    pub deprecated_since_version: &'static str,
    pub warn_until_version: &'static str,
    pub error_since_version: &'static str,
    pub removed_since_version: &'static str,
    /// The successor, when there is one
    pub replacement_ref: Option<&'static str>,
    /// Where the migration guide lives
    pub migration_guide_url: Option<&'static str>,
    /// Set when the item genuinely needs no guide, which is how a record
    /// without a URL passes the release check
    pub no_guide_required: bool,
    /// The sentence the warning and the guide both open with
    pub summary: &'static str,
    /// How the item was written before, used to build the guide
    pub before_example: &'static str,
    /// How it is written now
    pub after_example: &'static str,
    /// A snippet that finds or rewrites affected objects
    pub migration_snippet: &'static str,
    /// What people get wrong when they migrate
    pub common_pitfall: &'static str,
}

inventory::collect!(DeprecationRecord);

impl DeprecationRecord {
    /// Where this item sits for a running binary
    pub fn stage(&self, running: BinaryVersion) -> DeprecationStage {
        let removed = BinaryVersion::parse(self.removed_since_version);
        let erroring = BinaryVersion::parse(self.error_since_version);
        let deprecated = BinaryVersion::parse(self.deprecated_since_version);
        if let Some(removed) = removed {
            if running >= removed {
                return DeprecationStage::Removed;
            }
        }
        if let Some(erroring) = erroring {
            if running >= erroring {
                return DeprecationStage::Erroring;
            }
        }
        if let Some(deprecated) = deprecated {
            if running >= deprecated {
                return DeprecationStage::Warning;
            }
        }
        DeprecationStage::Supported
    }

    /// The message the warning and the error both carry
    pub fn guidance(&self) -> String {
        let mut message = format!(
            "`{}` is deprecated since {}",
            self.item_id, self.deprecated_since_version
        );
        if let Some(replacement) = self.replacement_ref {
            message.push_str(&format!(", use `{replacement}` instead"));
        }
        message.push_str(&format!(
            ". It errors from {} and is removed in {}",
            self.error_since_version, self.removed_since_version
        ));
        if let Some(url) = self.migration_guide_url {
            message.push_str(&format!(". Migration guide {url}"));
        }
        message
    }

    /// The migration guide, built from the record at deprecation time so a
    /// guide can never drift from the lifecycle dates beside it
    pub fn migration_guide(&self) -> MigrationGuide {
        let mut body = String::with_capacity(512);
        body.push_str(&format!("# {}\n\n", self.item_id));
        body.push_str(self.summary);
        body.push_str("\n\n## Lifecycle\n\n");
        body.push_str(&format!(
            "- Deprecated in {}\n- Warns until {}\n- Errors from {}\n- Removed in {}\n",
            self.deprecated_since_version,
            self.warn_until_version,
            self.error_since_version,
            self.removed_since_version
        ));
        if let Some(replacement) = self.replacement_ref {
            body.push_str(&format!("- Replaced by `{replacement}`\n"));
        }
        body.push_str("\n## Before\n\n```sql\n");
        body.push_str(self.before_example);
        body.push_str("\n```\n\n## After\n\n```sql\n");
        body.push_str(self.after_example);
        body.push_str("\n```\n\n## Migration script\n\n```sql\n");
        body.push_str(self.migration_snippet);
        body.push_str("\n```\n\n## Common pitfalls\n\n");
        body.push_str(self.common_pitfall);
        body.push('\n');
        MigrationGuide {
            item_id: self.item_id,
            item_kind: self.item_kind,
            title: self.summary,
            body,
            url: self.migration_guide_url,
        }
    }
}

/// A generated migration guide, published in
/// `zyron_sys.deprecation.migration_guides`
#[derive(Debug, Clone)]
pub struct MigrationGuide {
    pub item_id: &'static str,
    pub item_kind: DeprecatedItemKind,
    pub title: &'static str,
    pub body: String,
    pub url: Option<&'static str>,
}

/// Why the deprecation registry refused to load
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DeprecationRegistryError {
    DuplicateItem {
        item_id: &'static str,
    },
    UnparseableVersion {
        item_id: &'static str,
        field: &'static str,
        value: &'static str,
    },
    OutOfOrderLifecycle {
        item_id: &'static str,
    },
    MissingGuide {
        item_id: &'static str,
    },
}

impl fmt::Display for DeprecationRegistryError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            DeprecationRegistryError::DuplicateItem { item_id } => {
                write!(f, "deprecation item `{item_id}` is registered twice")
            }
            DeprecationRegistryError::UnparseableVersion {
                item_id,
                field,
                value,
            } => write!(
                f,
                "deprecation item `{item_id}` declares {field} = `{value}`, which is not a \
                 major.minor.patch version"
            ),
            DeprecationRegistryError::OutOfOrderLifecycle { item_id } => write!(
                f,
                "deprecation item `{item_id}` has lifecycle versions out of order. They must \
                 run deprecated_since <= warn_until <= error_since <= removed_since"
            ),
            DeprecationRegistryError::MissingGuide { item_id } => write!(
                f,
                "deprecation item `{item_id}` carries no migration_guide_url and is not \
                 marked no_guide_required"
            ),
        }
    }
}

impl std::error::Error for DeprecationRegistryError {}

/// Every registered deprecation, checked once at startup
#[derive(Debug, Default)]
pub struct DeprecationRegistry {
    records: Vec<DeprecationRecord>,
}

impl DeprecationRegistry {
    pub fn load() -> Result<DeprecationRegistry, DeprecationRegistryError> {
        let records: Vec<DeprecationRecord> = inventory::iter::<DeprecationRecord>
            .into_iter()
            .copied()
            .collect();
        DeprecationRegistry::from_records(records)
    }

    pub fn from_records(
        records: Vec<DeprecationRecord>,
    ) -> Result<DeprecationRegistry, DeprecationRegistryError> {
        let mut seen: Vec<&'static str> = Vec::with_capacity(records.len());
        for record in &records {
            if seen
                .iter()
                .any(|id| id.eq_ignore_ascii_case(record.item_id))
            {
                return Err(DeprecationRegistryError::DuplicateItem {
                    item_id: record.item_id,
                });
            }
            seen.push(record.item_id);

            let fields = [
                ("deprecated_since_version", record.deprecated_since_version),
                ("warn_until_version", record.warn_until_version),
                ("error_since_version", record.error_since_version),
                ("removed_since_version", record.removed_since_version),
            ];
            let mut parsed = Vec::with_capacity(4);
            for (field, value) in fields {
                match BinaryVersion::parse(value) {
                    Some(v) => parsed.push(v),
                    None => {
                        return Err(DeprecationRegistryError::UnparseableVersion {
                            item_id: record.item_id,
                            field,
                            value,
                        });
                    }
                }
            }
            if parsed.windows(2).any(|pair| pair[0] > pair[1]) {
                return Err(DeprecationRegistryError::OutOfOrderLifecycle {
                    item_id: record.item_id,
                });
            }
            if record.migration_guide_url.is_none() && !record.no_guide_required {
                return Err(DeprecationRegistryError::MissingGuide {
                    item_id: record.item_id,
                });
            }
        }
        Ok(DeprecationRegistry { records })
    }

    pub fn records(&self) -> &[DeprecationRecord] {
        &self.records
    }

    /// One item by id, case insensitively
    pub fn find(&self, item_id: &str) -> Option<&DeprecationRecord> {
        self.records
            .iter()
            .find(|r| r.item_id.eq_ignore_ascii_case(item_id))
    }

    /// Every item at a given stage for a running version
    pub fn at_stage(
        &self,
        running: BinaryVersion,
        stage: DeprecationStage,
    ) -> Vec<&DeprecationRecord> {
        self.records
            .iter()
            .filter(|r| r.stage(running) == stage)
            .collect()
    }

    /// Every generated guide, which is what
    /// `zyron_sys.deprecation.migration_guides` serves
    pub fn guides(&self) -> Vec<MigrationGuide> {
        self.records.iter().map(|r| r.migration_guide()).collect()
    }
}

// ---------------------------------------------------------------------------
// Warning rate limiting
// ---------------------------------------------------------------------------

/// Default warnings allowed per item per tenant per hour
pub const DEFAULT_WARNING_RATE_LIMIT_PER_HOUR: u32 = 10;

/// Seconds in the rate limit window
const WINDOW_SECS: u64 = 3_600;

/// Number of buckets. A power of two so the index is a mask rather than a
/// remainder, and wide enough that two live items rarely share one
const BUCKET_COUNT: usize = 1 << 12;

/// One bucket, holding the key it is currently counting for.
///
/// A bucket is claimed by the first key that lands on it in a window. A
/// second key colliding with it inside the same window shares the budget,
/// which under-reports rather than over-reports, and is the right way round
/// for a limiter whose job is to stop log floods
#[derive(Debug, Default)]
struct WarningBucket {
    key: AtomicU64,
    window_start: AtomicU64,
    count: AtomicU32,
}

/// Counts deprecation warnings per item per tenant per hour.
///
/// The check is a hash, a mask, and three relaxed atomic loads, so a parse
/// that touches a deprecated item pays a bucket lookup rather than a lock
#[derive(Debug)]
pub struct WarningRateLimiter {
    buckets: Vec<WarningBucket>,
    limit: AtomicU32,
    suppressed: AtomicU64,
    emitted: AtomicU64,
}

impl Default for WarningRateLimiter {
    fn default() -> Self {
        Self::new(DEFAULT_WARNING_RATE_LIMIT_PER_HOUR)
    }
}

impl WarningRateLimiter {
    pub fn new(limit_per_hour: u32) -> Self {
        let mut buckets = Vec::with_capacity(BUCKET_COUNT);
        buckets.resize_with(BUCKET_COUNT, WarningBucket::default);
        Self {
            buckets,
            limit: AtomicU32::new(limit_per_hour),
            suppressed: AtomicU64::new(0),
            emitted: AtomicU64::new(0),
        }
    }

    pub fn set_limit(&self, limit_per_hour: u32) {
        self.limit.store(limit_per_hour, Ordering::Relaxed);
    }

    pub fn limit(&self) -> u32 {
        self.limit.load(Ordering::Relaxed)
    }

    /// Whether a warning for this item and tenant should be emitted now.
    ///
    /// Counts the use either way, so the suppressed total is the real number
    /// of uses the limiter hid
    pub fn allow(&self, item_id: &str, tenant: &str, now_secs: u64) -> bool {
        let key = key_of(item_id, tenant);
        let bucket = &self.buckets[(key as usize) & (BUCKET_COUNT - 1)];
        let window = now_secs / WINDOW_SECS;

        let claimed = bucket.key.load(Ordering::Relaxed);
        let started = bucket.window_start.load(Ordering::Relaxed);
        if claimed != key || started != window {
            bucket.key.store(key, Ordering::Relaxed);
            bucket.window_start.store(window, Ordering::Relaxed);
            bucket.count.store(1, Ordering::Relaxed);
            self.emitted.fetch_add(1, Ordering::Relaxed);
            return true;
        }

        let seen = bucket.count.fetch_add(1, Ordering::Relaxed);
        if seen < self.limit.load(Ordering::Relaxed) {
            self.emitted.fetch_add(1, Ordering::Relaxed);
            true
        } else {
            self.suppressed.fetch_add(1, Ordering::Relaxed);
            false
        }
    }

    /// Uses that were counted but not warned about
    pub fn suppressed(&self) -> u64 {
        self.suppressed.load(Ordering::Relaxed)
    }

    /// Warnings that were emitted
    pub fn emitted(&self) -> u64 {
        self.emitted.load(Ordering::Relaxed)
    }
}

/// Folds an item and tenant into one bucket key, never zero so a claimed
/// bucket is distinguishable from a fresh one
#[inline]
fn key_of(item_id: &str, tenant: &str) -> u64 {
    let mut hasher = crate::checksum::Hasher::new();
    hasher.update(item_id.as_bytes());
    hasher.update(b"\x1f");
    hasher.update(tenant.as_bytes());
    hasher.finish32() as u64 | 1
}

/// One emitted warning, held for the trailing-window view
#[derive(Debug, Clone)]
pub struct EmittedWarning {
    pub item_id: String,
    pub item_kind: DeprecatedItemKind,
    pub tenant: String,
    pub at_secs: u64,
    pub message: String,
}

/// The trailing window of emitted warnings, which
/// `zyron_sys.upgrade.deprecation_warnings` and
/// `zyron-ctl deprecation report` both read
#[derive(Debug)]
pub struct WarningLog {
    entries: std::sync::Mutex<Vec<EmittedWarning>>,
    retain_secs: u64,
    capacity: usize,
}

impl Default for WarningLog {
    fn default() -> Self {
        // A day of warnings is what the report asks for, capped so a runaway
        // workload cannot grow the log without bound
        Self::new(86_400, 16_384)
    }
}

impl WarningLog {
    pub fn new(retain_secs: u64, capacity: usize) -> Self {
        Self {
            entries: std::sync::Mutex::new(Vec::new()),
            retain_secs,
            capacity,
        }
    }

    pub fn record(&self, warning: EmittedWarning) {
        let Ok(mut entries) = self.entries.lock() else {
            return;
        };
        let cutoff = warning.at_secs.saturating_sub(self.retain_secs);
        entries.retain(|e| e.at_secs >= cutoff);
        if entries.len() >= self.capacity {
            let drop_to = entries.len() + 1 - self.capacity;
            entries.drain(..drop_to);
        }
        entries.push(warning);
    }

    /// Warnings emitted in the trailing window ending now
    pub fn since(&self, from_secs: u64) -> Vec<EmittedWarning> {
        self.entries
            .lock()
            .map(|entries| {
                entries
                    .iter()
                    .filter(|e| e.at_secs >= from_secs)
                    .cloned()
                    .collect()
            })
            .unwrap_or_default()
    }

    /// Per-item counts in the trailing window, with the tenants that hit
    /// each item
    pub fn report(&self, from_secs: u64) -> Vec<WarningReportRow> {
        let entries = self.since(from_secs);
        let mut rows: Vec<WarningReportRow> = Vec::new();
        for entry in entries {
            match rows.iter_mut().find(|r| r.item_id == entry.item_id) {
                Some(row) => {
                    row.count += 1;
                    if !row.tenants.contains(&entry.tenant) {
                        row.tenants.push(entry.tenant);
                    }
                }
                None => rows.push(WarningReportRow {
                    item_id: entry.item_id,
                    item_kind: entry.item_kind,
                    count: 1,
                    tenants: vec![entry.tenant],
                }),
            }
        }
        rows.sort_by(|a, b| b.count.cmp(&a.count).then(a.item_id.cmp(&b.item_id)));
        for row in rows.iter_mut() {
            row.tenants.sort();
        }
        rows
    }
}

/// One row of the deprecation report
#[derive(Debug, Clone)]
pub struct WarningReportRow {
    pub item_id: String,
    pub item_kind: DeprecatedItemKind,
    pub count: u64,
    pub tenants: Vec<String>,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn record() -> DeprecationRecord {
        DeprecationRecord {
            item_kind: DeprecatedItemKind::DdlKeyword,
            item_id: "CREATE PIPELINE",
            deprecated_since_version: "1.0.0",
            warn_until_version: "1.2.0",
            error_since_version: "1.3.0",
            removed_since_version: "1.4.0",
            replacement_ref: Some("CREATE WORKFLOW"),
            migration_guide_url: Some("/docs/sql/create-workflow.md"),
            no_guide_required: false,
            summary: "Pipelines became workflows",
            before_example: "CREATE PIPELINE p AS (STAGE s AS SELECT 1)",
            after_example: "CREATE WORKFLOW w AS (TASK s AS SELECT 1)",
            migration_snippet: "SELECT name FROM zyron_sys.core.pipelines",
            common_pitfall: "A stage name is not a task name, rename it explicitly",
        }
    }

    #[test]
    fn test_binary_version_parsing() {
        assert_eq!(
            BinaryVersion::parse("1.2.3"),
            Some(BinaryVersion::new(1, 2, 3))
        );
        assert_eq!(
            BinaryVersion::parse("v0.11.0"),
            Some(BinaryVersion::new(0, 11, 0))
        );
        assert_eq!(
            BinaryVersion::parse("2.0"),
            Some(BinaryVersion::new(2, 0, 0))
        );
        assert_eq!(
            BinaryVersion::parse("1.2.3-rc1"),
            Some(BinaryVersion::new(1, 2, 3))
        );
        assert_eq!(BinaryVersion::parse("1.2.3.4"), None);
        assert_eq!(BinaryVersion::parse("nope"), None);
        assert!(BinaryVersion::new(2, 0, 0).is_major_step_from(&BinaryVersion::new(1, 9, 9)));
        assert!(!BinaryVersion::new(1, 9, 0).is_major_step_from(&BinaryVersion::new(1, 2, 0)));
    }

    #[test]
    fn test_stage_walks_the_lifecycle() {
        let record = record();
        assert_eq!(
            record.stage(BinaryVersion::new(0, 9, 0)),
            DeprecationStage::Supported
        );
        assert_eq!(
            record.stage(BinaryVersion::new(1, 1, 0)),
            DeprecationStage::Warning
        );
        assert_eq!(
            record.stage(BinaryVersion::new(1, 3, 0)),
            DeprecationStage::Erroring
        );
        assert_eq!(
            record.stage(BinaryVersion::new(1, 4, 0)),
            DeprecationStage::Removed
        );
        assert_eq!(
            record.stage(BinaryVersion::new(9, 0, 0)),
            DeprecationStage::Removed
        );
    }

    #[test]
    fn test_guidance_names_the_replacement_and_the_guide() {
        let text = record().guidance();
        assert!(text.contains("CREATE WORKFLOW"), "{text}");
        assert!(text.contains("removed in 1.4.0"), "{text}");
        assert!(text.contains("/docs/sql/create-workflow.md"), "{text}");
    }

    #[test]
    fn test_guide_carries_before_after_and_snippet() {
        let guide = record().migration_guide();
        assert!(
            guide.body.contains("CREATE PIPELINE p AS"),
            "{}",
            guide.body
        );
        assert!(
            guide.body.contains("CREATE WORKFLOW w AS"),
            "{}",
            guide.body
        );
        assert!(
            guide.body.contains("zyron_sys.core.pipelines"),
            "{}",
            guide.body
        );
        assert!(guide.body.contains("Common pitfalls"), "{}", guide.body);
    }

    #[test]
    fn test_registry_rejects_a_duplicate() {
        let err = DeprecationRegistry::from_records(vec![record(), record()])
            .expect_err("duplicate refused");
        assert!(matches!(
            err,
            DeprecationRegistryError::DuplicateItem { .. }
        ));
    }

    #[test]
    fn test_registry_rejects_out_of_order_lifecycle() {
        let mut bad = record();
        bad.error_since_version = "1.0.0";
        bad.warn_until_version = "1.2.0";
        let err = DeprecationRegistry::from_records(vec![bad]).expect_err("out of order refused");
        assert!(matches!(
            err,
            DeprecationRegistryError::OutOfOrderLifecycle { .. }
        ));
    }

    #[test]
    fn test_registry_rejects_a_record_without_a_guide() {
        let mut bad = record();
        bad.migration_guide_url = None;
        let err = DeprecationRegistry::from_records(vec![bad]).expect_err("no guide refused");
        assert!(matches!(err, DeprecationRegistryError::MissingGuide { .. }));

        let mut allowed = record();
        allowed.migration_guide_url = None;
        allowed.no_guide_required = true;
        assert!(DeprecationRegistry::from_records(vec![allowed]).is_ok());
    }

    #[test]
    fn test_rate_limiter_allows_the_budget_then_suppresses() {
        let limiter = WarningRateLimiter::new(10);
        let mut allowed = 0;
        for _ in 0..100 {
            if limiter.allow("CREATE PIPELINE", "tenant-a", 0) {
                allowed += 1;
            }
        }
        assert_eq!(allowed, 10);
        assert_eq!(limiter.emitted(), 10);
        assert_eq!(limiter.suppressed(), 90);
    }

    #[test]
    fn test_rate_limiter_budgets_per_tenant() {
        let limiter = WarningRateLimiter::new(2);
        assert!(limiter.allow("item", "tenant-a", 0));
        assert!(limiter.allow("item", "tenant-a", 0));
        assert!(!limiter.allow("item", "tenant-a", 0));
        assert!(limiter.allow("item", "tenant-b", 0));
        assert!(limiter.allow("item", "tenant-b", 0));
        assert!(!limiter.allow("item", "tenant-b", 0));
    }

    #[test]
    fn test_rate_limiter_resets_each_hour() {
        let limiter = WarningRateLimiter::new(1);
        assert!(limiter.allow("item", "t", 0));
        assert!(!limiter.allow("item", "t", 3_599));
        assert!(limiter.allow("item", "t", 3_600));
    }

    #[test]
    fn test_warning_log_reports_counts_and_tenants() {
        let log = WarningLog::new(86_400, 100);
        for tenant in ["a", "a", "b"] {
            log.record(EmittedWarning {
                item_id: "CREATE PIPELINE".to_string(),
                item_kind: DeprecatedItemKind::DdlKeyword,
                tenant: tenant.to_string(),
                at_secs: 1_000,
                message: "deprecated".to_string(),
            });
        }
        log.record(EmittedWarning {
            item_id: "warehouse".to_string(),
            item_kind: DeprecatedItemKind::SqlSyntax,
            tenant: "a".to_string(),
            at_secs: 1_000,
            message: "deprecated".to_string(),
        });
        let rows = log.report(0);
        assert_eq!(rows.len(), 2);
        assert_eq!(rows[0].item_id, "CREATE PIPELINE");
        assert_eq!(rows[0].count, 3);
        assert_eq!(rows[0].tenants, vec!["a".to_string(), "b".to_string()]);
        assert_eq!(rows[1].count, 1);
    }

    #[test]
    fn test_warning_log_drops_entries_past_the_window() {
        let log = WarningLog::new(60, 100);
        log.record(EmittedWarning {
            item_id: "old".to_string(),
            item_kind: DeprecatedItemKind::ConfigKey,
            tenant: "t".to_string(),
            at_secs: 0,
            message: String::new(),
        });
        log.record(EmittedWarning {
            item_id: "new".to_string(),
            item_kind: DeprecatedItemKind::ConfigKey,
            tenant: "t".to_string(),
            at_secs: 1_000,
            message: String::new(),
        });
        let rows = log.report(0);
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0].item_id, "new");
    }
}
