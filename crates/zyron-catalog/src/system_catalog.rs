//! The `zyron_sys` system catalog.
//!
//! `zyron_sys` is a catalog in its own right, sitting beside the user
//! catalogs rather than inside one. Every system view, table function, and
//! system table lives at the three-part name `zyron_sys.<schema>.<object>`,
//! grouped by the subsystem that produces it.
//!
//! This module is the single registry of those names. The catalog and schema
//! rows are created from it at startup, the wire layer dispatches reads
//! through it, and the name resolver asks it what a failed lookup was
//! probably reaching for. Nothing here resolves a retired name: a name that
//! is not in the registry does not exist, and the closest canonical name is
//! computed by similarity at the point of failure so it can be printed in
//! the error without ever being followed.

use std::collections::HashSet;

use zyron_common::{Result, ZyronError};

use crate::catalog::Catalog;
use crate::ids::DatabaseId;

/// Name of the system catalog. A catalog, not a schema: its children are
/// schemas, and its objects are addressed with all three parts.
pub const SYSTEM_CATALOG_NAME: &str = "zyron_sys";

/// Owner recorded on the catalog and every schema in it.
pub const SYSTEM_CATALOG_OWNER: &str = "system";

/// Schemas of the `zyron_sys` catalog, in the order they are documented.
///
/// Order is load-bearing in one place: when two canonical names score the
/// same against an unknown name, the one whose schema appears first here
/// wins, so a hint is the same on every run.
pub const SYSTEM_SCHEMAS: &[&str] = &[
    // Core / SQL surface
    "core",
    "sql",
    "session",
    // Storage tier, holding what the heap, btree, columnar, lake, snapshot,
    // checkpoint, vacuum, and gc layers expose, plus the on-disk format
    // registry, the per-version layouts, and in-flight format migrations
    "storage",
    // Signature scheme registry and the per-artifact-kind mapping
    "crypto",
    // Deprecation lifecycle records and the guides generated from them
    "deprecation",
    // Auto-upgrade state, history, and the work an upgrade left to do
    "upgrade",
    // Wire protocol versions and their transition state
    "wire",
    // Distribution / consensus
    "cluster",
    "raft",
    "replica",
    "dr",
    "federation",
    // Query + cache
    "query",
    "cache",
    // Mesh / compute / cloud
    "mesh",
    "compute",
    "cloud",
    // Data / versioning
    "time_travel",
    "branch",
    "schema",
    "semantic",
    "contract",
    "expectation",
    "data_drift",
    "schema_drift",
    "external_table",
    "data_placement",
    // Streaming / CDC
    "streaming",
    "cdc",
    // Security. Policy and secret data are split because they carry
    // different permission models
    "security",
    "secret",
    // Observability
    "audit",
    "alert",
    "notification",
    "stat",
    "trace",
    "log",
    "metric",
    "pressure",
    // Governance / compliance / retention
    "governance",
    "compliance",
    "retention",
    // Operations
    "tenant",
    "usage",
    "cost",
    "chargeback",
    "noisy_neighbor",
    "deploy",
    "migration",
    "backup",
    "platform_update",
    // Zyron-branded subsystems
    "app",
    "workflow",
    "volume",
    "dashboard",
    // ML / search
    "ml",
    "search",
    "vector",
    "graph",
    // Sustainability
    "sustainability",
];

/// Search path a session starts with. The two system entries come first so a
/// bare `tables` reads `zyron_sys.core.tables` and a bare `activity` reads
/// `zyron_sys.stat.activity`. There is deliberately NO default user schema:
/// Zyron never creates or assumes a schema for user tables, so a session
/// reaches its own objects by creating a schema, setting the search path to
/// it, or qualifying names. A bare name that is not a system entity fails to
/// resolve rather than landing in an implicit namespace.
pub const DEFAULT_SEARCH_PATH: &[&str] =
    &["zyron_sys.core", "zyron_sys.stat", "information_schema"];

/// The default search path as owned strings, for planner and context call
/// sites. Statement execution that has no session namespace uses exactly
/// this, never an implicit user schema, so an unqualified user-table name in
/// session-less SQL fails loudly instead of resolving somewhere surprising.
pub fn default_search_path() -> Vec<String> {
    DEFAULT_SEARCH_PATH.iter().map(|s| s.to_string()).collect()
}

/// What a registered system entity is.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SystemObjectKind {
    /// Read with a plain SELECT, computed on read.
    View,
    /// Called with arguments in a FROM clause.
    TableFunction,
}

impl SystemObjectKind {
    pub fn label(&self) -> &'static str {
        match self {
            SystemObjectKind::View => "VIEW",
            SystemObjectKind::TableFunction => "TABLE FUNCTION",
        }
    }
}

/// One registered entity of the system catalog.
#[derive(Debug, Clone, Copy)]
pub struct SystemObject {
    pub schema: &'static str,
    pub object: &'static str,
    pub kind: SystemObjectKind,
    /// One line describing what the entity answers, surfaced by
    /// `zyron_sys.core.system_view_documentation`.
    pub doc: &'static str,
}

impl SystemObject {
    /// The three-part name this entity answers to. The only name it answers
    /// to.
    pub fn canonical_name(&self) -> String {
        format!("{}.{}.{}", SYSTEM_CATALOG_NAME, self.schema, self.object)
    }
}

/// Every entity the system catalog serves.
///
/// A subsystem that ships a new system view adds its row here under one of
/// the schemas above; the wire dispatch and the documentation view are both
/// driven from this list, so an entity that is registered but unreachable,
/// or reachable but unregistered, fails the registry tests.
pub const SYSTEM_OBJECTS: &[SystemObject] = &[
    // -----------------------------------------------------------------------
    // core
    // -----------------------------------------------------------------------
    SystemObject {
        schema: "core",
        object: "databases",
        kind: SystemObjectKind::View,
        doc: "Every catalog on this node, including zyron_sys itself",
    },
    SystemObject {
        schema: "core",
        object: "schemas",
        kind: SystemObjectKind::View,
        doc: "Every schema, with the catalog that holds it",
    },
    SystemObject {
        schema: "core",
        object: "tables",
        kind: SystemObjectKind::View,
        doc: "Every live table, with its schema, storage format, and column count",
    },
    SystemObject {
        schema: "core",
        object: "columns",
        kind: SystemObjectKind::View,
        doc: "Every column of every live table, in ordinal order",
    },
    SystemObject {
        schema: "core",
        object: "system_view_documentation",
        kind: SystemObjectKind::View,
        doc: "One row per registered system entity: catalog, schema, object, kind, docstring",
    },
    // -----------------------------------------------------------------------
    // storage
    // -----------------------------------------------------------------------
    SystemObject {
        schema: "storage",
        object: "indexes",
        kind: SystemObjectKind::View,
        doc: "Every index, with its table, type, uniqueness, and key columns",
    },
    SystemObject {
        schema: "storage",
        object: "clustering_status",
        kind: SystemObjectKind::View,
        doc: "Per-table lake clustering mode, schedule, skip rate, and last pass",
    },
    SystemObject {
        schema: "storage",
        object: "derived_columns",
        kind: SystemObjectKind::View,
        doc: "Derived columns declared on lake tables and the expressions behind them",
    },
    SystemObject {
        schema: "storage",
        object: "auto_compaction_history",
        kind: SystemObjectKind::View,
        doc: "Compaction passes the lake maintenance worker has run per table",
    },
    SystemObject {
        schema: "storage",
        object: "table_freshness",
        kind: SystemObjectKind::View,
        doc: "How far behind each lake table's maintenance state is from its latest commit",
    },
    SystemObject {
        schema: "storage",
        object: "lake_log",
        kind: SystemObjectKind::View,
        doc: "Raw lake transaction log entries per table version",
    },
    SystemObject {
        schema: "storage",
        object: "variant_shredding_stats",
        kind: SystemObjectKind::View,
        doc: "Observed JSON paths per variant column with occurrences, coverage, and shredding state",
    },
    SystemObject {
        schema: "storage",
        object: "format_registry",
        kind: SystemObjectKind::View,
        doc: "Every on-disk format, the version written, the versions readable, and the migration policy",
    },
    SystemObject {
        schema: "storage",
        object: "format_documentation",
        kind: SystemObjectKind::View,
        doc: "Per-format on-disk layout with byte offsets, generated from the format modules",
    },
    SystemObject {
        schema: "storage",
        object: "format_migrations",
        kind: SystemObjectKind::View,
        doc: "Format migrations in flight, with progress, bytes remaining, and estimated completion",
    },
    SystemObject {
        schema: "storage",
        object: "catalog_schema_evolution",
        kind: SystemObjectKind::View,
        doc: "Per-catalog-table schema version history and the migration function behind each step",
    },
    // -----------------------------------------------------------------------
    // crypto
    // -----------------------------------------------------------------------
    SystemObject {
        schema: "crypto",
        object: "scheme_registry",
        kind: SystemObjectKind::View,
        doc: "Every registered signature scheme with its numeric tag, status, and retirement date",
    },
    SystemObject {
        schema: "crypto",
        object: "artifact_scheme_map",
        kind: SystemObjectKind::View,
        doc: "Per-artifact-kind current scheme, the outgoing scheme during a rotation, and the overlap end",
    },
    // -----------------------------------------------------------------------
    // deprecation
    // -----------------------------------------------------------------------
    SystemObject {
        schema: "deprecation",
        object: "registry",
        kind: SystemObjectKind::View,
        doc: "Every deprecated item with its warn, error, and removal versions and its replacement",
    },
    SystemObject {
        schema: "deprecation",
        object: "migration_guides",
        kind: SystemObjectKind::View,
        doc: "Migration guide generated for each deprecated item, with before and after examples",
    },
    // -----------------------------------------------------------------------
    // upgrade
    // -----------------------------------------------------------------------
    SystemObject {
        schema: "upgrade",
        object: "state",
        kind: SystemObjectKind::View,
        doc: "Current upgrade phase per node, with the versions it is moving between",
    },
    SystemObject {
        schema: "upgrade",
        object: "history",
        kind: SystemObjectKind::View,
        doc: "Past upgrades with their outcome, the work each ran, and whether it can be undone",
    },
    SystemObject {
        schema: "upgrade",
        object: "format_migrations",
        kind: SystemObjectKind::View,
        doc: "Format migrations this upgrade started, with per-format progress",
    },
    SystemObject {
        schema: "upgrade",
        object: "user_object_rewrites",
        kind: SystemObjectKind::View,
        doc: "User-authored objects an upgrade would rewrite, their class, and where each stands",
    },
    SystemObject {
        schema: "upgrade",
        object: "deprecation_warnings",
        kind: SystemObjectKind::View,
        doc: "Deprecation warnings emitted in the trailing window, per item and tenant",
    },
    // -----------------------------------------------------------------------
    // wire
    // -----------------------------------------------------------------------
    SystemObject {
        schema: "wire",
        object: "protocol_versions",
        kind: SystemObjectKind::View,
        doc: "The client, mesh, and consensus protocol versions, which are accepted now, and \
              when each was introduced",
    },
    // -----------------------------------------------------------------------
    // stat
    // -----------------------------------------------------------------------
    SystemObject {
        schema: "stat",
        object: "activity",
        kind: SystemObjectKind::View,
        doc: "One row per live session with its state and last activity",
    },
    SystemObject {
        schema: "stat",
        object: "tables",
        kind: SystemObjectKind::View,
        doc: "Per-table tuple and IO counters accumulated since startup",
    },
    SystemObject {
        schema: "stat",
        object: "indexes",
        kind: SystemObjectKind::View,
        doc: "Per-index scan and tuple-read counters",
    },
    SystemObject {
        schema: "stat",
        object: "wal",
        kind: SystemObjectKind::View,
        doc: "WAL writer counters: bytes written, records, flushes, current LSN",
    },
    SystemObject {
        schema: "stat",
        object: "bgwriter",
        kind: SystemObjectKind::View,
        doc: "Checkpoint and vacuum worker progress counters",
    },
    SystemObject {
        schema: "stat",
        object: "cdc_feeds",
        kind: SystemObjectKind::View,
        doc: "Change data feed size and retention per captured table",
    },
    SystemObject {
        schema: "stat",
        object: "replication_slots",
        kind: SystemObjectKind::View,
        doc: "Replication slot positions, activity, and WAL lag",
    },
    SystemObject {
        schema: "stat",
        object: "cdc_streams",
        kind: SystemObjectKind::View,
        doc: "Outbound CDC streams and the slots they read",
    },
    SystemObject {
        schema: "stat",
        object: "cdc_ingests",
        kind: SystemObjectKind::View,
        doc: "Inbound CDC ingests with applied and failed record counts",
    },
    SystemObject {
        schema: "stat",
        object: "streaming_jobs",
        kind: SystemObjectKind::View,
        doc: "Runtime state of streaming jobs held by the job manager",
    },
    SystemObject {
        schema: "stat",
        object: "trigger_executions",
        kind: SystemObjectKind::View,
        doc: "Registered triggers with their timing, events, and enabled state",
    },
    SystemObject {
        schema: "stat",
        object: "pipeline_runs",
        kind: SystemObjectKind::View,
        doc: "Pipeline definitions with their stage counts and last run outcome",
    },
    SystemObject {
        schema: "stat",
        object: "branches",
        kind: SystemObjectKind::View,
        doc: "Data branches known to the branch manager",
    },
    SystemObject {
        schema: "stat",
        object: "publications",
        kind: SystemObjectKind::View,
        doc: "Publications with their change feed, retention, and classification",
    },
    SystemObject {
        schema: "stat",
        object: "subscriptions",
        kind: SystemObjectKind::View,
        doc: "Subscriptions with their mode, state, and last seen LSN",
    },
    SystemObject {
        schema: "stat",
        object: "endpoints",
        kind: SystemObjectKind::View,
        doc: "HTTP endpoints registered by DDL, with path and auth mode",
    },
    SystemObject {
        schema: "stat",
        object: "dead_letters",
        kind: SystemObjectKind::View,
        doc: "Pending dead-letter rows per streaming sink target",
    },
    SystemObject {
        schema: "stat",
        object: "zyron_sinks",
        kind: SystemObjectKind::View,
        doc: "Zyron-to-Zyron sinks with their target and delivery state",
    },
    SystemObject {
        schema: "stat",
        object: "zyron_sources",
        kind: SystemObjectKind::View,
        doc: "Zyron-to-Zyron sources with their upstream and read position",
    },
    SystemObject {
        schema: "stat",
        object: "summary",
        kind: SystemObjectKind::View,
        doc: "One row per server-wide counter: WAL, checkpoint, vacuum, and               cumulative tuple activity",
    },
    SystemObject {
        schema: "stat",
        object: "credential_cache",
        kind: SystemObjectKind::View,
        doc: "External credential cache hits, misses, and refreshes per provider",
    },
    // -----------------------------------------------------------------------
    // security
    // -----------------------------------------------------------------------
    SystemObject {
        schema: "security",
        object: "users",
        kind: SystemObjectKind::View,
        doc: "Every account, with whether it may log in, whether it is a               superuser, and when its password expires",
    },
    // -----------------------------------------------------------------------
    // time_travel
    // -----------------------------------------------------------------------
    SystemObject {
        schema: "time_travel",
        object: "table_history",
        kind: SystemObjectKind::View,
        doc: "One row per committed version of a lake table",
    },
    SystemObject {
        schema: "time_travel",
        object: "version_details",
        kind: SystemObjectKind::View,
        doc: "Operation, metrics, and audit fields of one lake table version",
    },
    SystemObject {
        schema: "time_travel",
        object: "version_files",
        kind: SystemObjectKind::View,
        doc: "Data files added and removed by one lake table version",
    },
    SystemObject {
        schema: "time_travel",
        object: "diff_versions",
        kind: SystemObjectKind::View,
        doc: "File-level difference between two versions of a lake table",
    },
    SystemObject {
        schema: "time_travel",
        object: "schema_at_version",
        kind: SystemObjectKind::View,
        doc: "Column list a lake table had at a given version",
    },
    SystemObject {
        schema: "time_travel",
        object: "version_lineage",
        kind: SystemObjectKind::View,
        doc: "Chain of versions a lake table version descends from",
    },
    // -----------------------------------------------------------------------
    // branch
    // -----------------------------------------------------------------------
    SystemObject {
        schema: "branch",
        object: "lake_branches",
        kind: SystemObjectKind::View,
        doc: "Lake branches with their base version and head",
    },
    // -----------------------------------------------------------------------
    // mesh
    // -----------------------------------------------------------------------
    SystemObject {
        schema: "mesh",
        object: "nodes",
        kind: SystemObjectKind::View,
        doc: "Nodes this one has been told about, with their address and reachability",
    },
    // -----------------------------------------------------------------------
    // query
    // -----------------------------------------------------------------------
    SystemObject {
        schema: "query",
        object: "recommend_indexes",
        kind: SystemObjectKind::TableFunction,
        doc: "Indexes the planner's workload tracker would create, with the evidence for each. \
              Takes an optional schema name filter",
    },
    // -----------------------------------------------------------------------
    // sql
    // -----------------------------------------------------------------------
    SystemObject {
        schema: "sql",
        object: "triggers",
        kind: SystemObjectKind::View,
        doc: "Every trigger with its table, timing, events, and body function",
    },
    // -----------------------------------------------------------------------
    // session
    // -----------------------------------------------------------------------
    SystemObject {
        schema: "session",
        object: "prepared_statements",
        kind: SystemObjectKind::View,
        doc: "Prepared statements of every live connection, keyed by the pid \
              shown in zyron_sys.stat.activity",
    },
    // -----------------------------------------------------------------------
    // expectation
    // -----------------------------------------------------------------------
    SystemObject {
        schema: "expectation",
        object: "results",
        kind: SystemObjectKind::View,
        doc: "Recent expectation evaluations per table with pass or fail, \
              violation counts, and the action taken",
    },
    // -----------------------------------------------------------------------
    // cost
    // -----------------------------------------------------------------------
    SystemObject {
        schema: "cost",
        object: "currency_rates",
        kind: SystemObjectKind::View,
        doc: "Currency conversion rates CONVERT_CURRENCY reads. Writable: \
              INSERT VALUES upserts (from_currency, to_currency, rate_date, \
              rate) rows, DELETE without a predicate clears the table",
    },
    // -----------------------------------------------------------------------
    // retention
    // -----------------------------------------------------------------------
    SystemObject {
        schema: "retention",
        object: "storage_by_age",
        kind: SystemObjectKind::View,
        doc: "Row age distribution per retention managed table, with byte \
              estimates from the ANALYZE average row size",
    },
    SystemObject {
        schema: "retention",
        object: "upcoming_actions",
        kind: SystemObjectKind::View,
        doc: "The next action each retention policy will take and how many \
              rows it would touch now",
    },
    SystemObject {
        schema: "retention",
        object: "savings_estimate",
        kind: SystemObjectKind::View,
        doc: "Rows and bytes an immediate purge of expired data would reclaim \
              per table",
    },
    SystemObject {
        schema: "retention",
        object: "compliance_summary",
        kind: SystemObjectKind::View,
        doc: "Latest retention job outcome per policy",
    },
    // -----------------------------------------------------------------------
    // ml
    // -----------------------------------------------------------------------
    SystemObject {
        schema: "ml",
        object: "feature_groups",
        kind: SystemObjectKind::View,
        doc: "Feature groups in the feature store, with entity key and refresh settings",
    },
    SystemObject {
        schema: "ml",
        object: "feature_definitions",
        kind: SystemObjectKind::View,
        doc: "Individual features within each group, with type and transform expression",
    },
    SystemObject {
        schema: "ml",
        object: "models",
        kind: SystemObjectKind::View,
        doc: "Trained models installed in the inference cache",
    },
    // -----------------------------------------------------------------------
    // compliance
    // -----------------------------------------------------------------------
    SystemObject {
        schema: "compliance",
        object: "legal_holds",
        kind: SystemObjectKind::View,
        doc: "Legal holds with the table and predicate each protects",
    },
    SystemObject {
        schema: "compliance",
        object: "report",
        kind: SystemObjectKind::TableFunction,
        doc: "report(kind) returns the compliance report for one of \
              retention, legal_hold, audit, or events",
    },
    // -----------------------------------------------------------------------
    // streaming
    // -----------------------------------------------------------------------
    SystemObject {
        schema: "streaming",
        object: "watermarks",
        kind: SystemObjectKind::View,
        doc: "Current watermark per streaming source and when it last advanced",
    },
    SystemObject {
        schema: "streaming",
        object: "checkpoint_history",
        kind: SystemObjectKind::View,
        doc: "Completed and failed streaming checkpoints with duration and size",
    },
    SystemObject {
        schema: "streaming",
        object: "backpressure",
        kind: SystemObjectKind::View,
        doc: "Per-operator queue occupancy and backpressure ratio",
    },
    SystemObject {
        schema: "streaming",
        object: "operator_metrics",
        kind: SystemObjectKind::View,
        doc: "Per-operator record counts, processing time, and watermark",
    },
    SystemObject {
        schema: "streaming",
        object: "jobs",
        kind: SystemObjectKind::View,
        doc: "Streaming job definitions from the catalog, with source, target, and last error",
    },
    // -----------------------------------------------------------------------
    // cdc
    // -----------------------------------------------------------------------
    SystemObject {
        schema: "cdc",
        object: "create_replication_slot",
        kind: SystemObjectKind::TableFunction,
        doc: "create_replication_slot(name, plugin) creates a slot pinned at the \
              current WAL head and returns its starting position",
    },
    SystemObject {
        schema: "cdc",
        object: "logical_slot_get_changes",
        kind: SystemObjectKind::TableFunction,
        doc: "logical_slot_get_changes(name [, upto_n]) consumes decoded changes \
              from a slot and advances it past them",
    },
    // -----------------------------------------------------------------------
    // pressure
    // -----------------------------------------------------------------------
    SystemObject {
        schema: "pressure",
        object: "node_capabilities",
        kind: SystemObjectKind::View,
        doc: "What the node measured about the machine it runs on, one row per mount",
    },
    SystemObject {
        schema: "pressure",
        object: "current",
        kind: SystemObjectKind::View,
        doc: "The pressure signal right now, at both timescales",
    },
    SystemObject {
        schema: "pressure",
        object: "history",
        kind: SystemObjectKind::View,
        doc: "Recent pressure samples the controller decided against",
    },
    SystemObject {
        schema: "pressure",
        object: "admissions",
        kind: SystemObjectKind::View,
        doc: "Admission decisions with the reason each query was let in or refused",
    },
    SystemObject {
        schema: "pressure",
        object: "classes",
        kind: SystemObjectKind::View,
        doc: "Workload classes the controller separates queries into",
    },
    SystemObject {
        schema: "pressure",
        object: "calibration_cache",
        kind: SystemObjectKind::View,
        doc: "Calibration the node measured or inherited, keyed by hardware fingerprint",
    },
    SystemObject {
        schema: "pressure",
        object: "learning_ledger",
        kind: SystemObjectKind::View,
        doc: "Every actuator move the controller made and what followed it",
    },
    SystemObject {
        schema: "pressure",
        object: "capacity_projection",
        kind: SystemObjectKind::View,
        doc: "What the node projects it can still take before the knee",
    },
    SystemObject {
        schema: "pressure",
        object: "node_state",
        kind: SystemObjectKind::View,
        doc: "Rung the actuator ladder is on and what the classifier blames",
    },
    SystemObject {
        schema: "pressure",
        object: "tenants",
        kind: SystemObjectKind::View,
        doc: "Per-tenant share of the node's work and refusals",
    },
    SystemObject {
        schema: "pressure",
        object: "projection",
        kind: SystemObjectKind::View,
        doc: "Projected headroom per resource under the current mix",
    },
    SystemObject {
        schema: "pressure",
        object: "provisioner",
        kind: SystemObjectKind::View,
        doc: "What the provisioner was asked for and what it returned",
    },
    SystemObject {
        schema: "pressure",
        object: "hot_set",
        kind: SystemObjectKind::View,
        doc: "Pages and tables the node is keeping resident",
    },
    SystemObject {
        schema: "pressure",
        object: "query_shapes",
        kind: SystemObjectKind::View,
        doc: "Query shapes the controller has fingerprinted and what they cost",
    },
    SystemObject {
        schema: "pressure",
        object: "spill_stats",
        kind: SystemObjectKind::View,
        doc: "What spilling has cost this node, by operator kind",
    },
];

// ---------------------------------------------------------------------------
// Lookup
// ---------------------------------------------------------------------------

/// The entity a fully three-part name addresses, or None when nothing does.
pub fn find(name: &str) -> Option<&'static SystemObject> {
    let (catalog, rest) = name.split_once('.')?;
    if !catalog.eq_ignore_ascii_case(SYSTEM_CATALOG_NAME) {
        return None;
    }
    let (schema, object) = rest.split_once('.')?;
    if object.contains('.') {
        return None;
    }
    SYSTEM_OBJECTS
        .iter()
        .find(|o| o.schema.eq_ignore_ascii_case(schema) && o.object.eq_ignore_ascii_case(object))
}

/// Whether a name addresses a registered system entity.
pub fn is_system_object(name: &str) -> bool {
    find(name).is_some()
}

/// Whether a name is under the system catalog at all, canonical or not.
///
/// A name in this space that `find` does not recognize is a name that was
/// retired or never existed, and it is refused here rather than handed to
/// the planner, which would look for a user table of that name.
pub fn is_system_catalog_name(name: &str) -> bool {
    match name.split_once('.') {
        Some((catalog, _)) => catalog.eq_ignore_ascii_case(SYSTEM_CATALOG_NAME),
        None => false,
    }
}

/// The entity an unqualified name reaches through a search path.
///
/// Only `zyron_sys.<schema>` entries can match: a bare name resolves against
/// the system catalog when the path puts a system schema ahead of the user
/// schema that would otherwise claim it, which is what makes `SELECT * FROM
/// tables` read `zyron_sys.core.tables` under the default path.
pub fn resolve_in_search_path<S: AsRef<str>>(
    object: &str,
    search_path: &[S],
) -> Option<&'static SystemObject> {
    if object.contains('.') {
        return None;
    }
    for entry in search_path {
        let entry = entry.as_ref();
        let Some((catalog, schema)) = entry.split_once('.') else {
            continue;
        };
        if !catalog.eq_ignore_ascii_case(SYSTEM_CATALOG_NAME) {
            continue;
        }
        if let Some(found) = SYSTEM_OBJECTS.iter().find(|o| {
            o.schema.eq_ignore_ascii_case(schema) && o.object.eq_ignore_ascii_case(object)
        }) {
            return Some(found);
        }
    }
    None
}

/// Whether a search path names a system schema anywhere in it.
pub fn search_path_reaches_system<S: AsRef<str>>(search_path: &[S]) -> bool {
    search_path.iter().any(|e| {
        e.as_ref()
            .split_once('.')
            .map(|(c, _)| c.eq_ignore_ascii_case(SYSTEM_CATALOG_NAME))
            .unwrap_or(false)
    })
}

/// A RelationNotFound for a name that resolved to nothing.
///
/// No suggestion is attached. A name that is not registered does not exist,
/// and the engine keeps nothing that could describe what it used to be.
pub fn relation_not_found(name: &str) -> ZyronError {
    ZyronError::relation_not_found(name)
}

// ---------------------------------------------------------------------------
// Registration
// ---------------------------------------------------------------------------

/// Brings the `zyron_sys` catalog and its schemas into existence.
pub struct SystemCatalog;

impl SystemCatalog {
    /// Registers the `zyron_sys` catalog and every schema in it, then records
    /// the schema ids on the catalog so DDL against them is refused.
    ///
    /// Idempotent: a restart finds the rows already there and only re-derives
    /// the id set. Called once at server startup, before any connection is
    /// accepted.
    pub async fn init(catalog: &Catalog) -> Result<DatabaseId> {
        let database_id = match catalog.get_database(SYSTEM_CATALOG_NAME) {
            Ok(existing) => existing.id,
            Err(_) => {
                catalog
                    .create_system_database(SYSTEM_CATALOG_NAME, SYSTEM_CATALOG_OWNER)
                    .await?
            }
        };

        let mut schema_ids = HashSet::with_capacity(SYSTEM_SCHEMAS.len());
        for name in SYSTEM_SCHEMAS {
            let id = match catalog.get_schema(database_id, name) {
                Ok(existing) => existing.id,
                Err(_) => {
                    catalog
                        .create_system_schema(database_id, name, SYSTEM_CATALOG_OWNER)
                        .await?
                }
            };
            schema_ids.insert(id);
        }

        catalog.adopt_system_catalog(database_id, schema_ids);
        Ok(database_id)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_schema_list_is_unique() {
        let mut seen = HashSet::new();
        for schema in SYSTEM_SCHEMAS {
            assert!(seen.insert(*schema), "duplicate schema `{}`", schema);
        }
    }

    #[test]
    fn test_schema_count_matches_documented_layout() {
        // 56 documented subsystem schemas, `pressure` for the
        // self-calibration substrate, and four for the format substrate:
        // crypto, deprecation, upgrade, wire
        assert_eq!(SYSTEM_SCHEMAS.len(), 61);
    }

    #[test]
    fn test_every_object_lands_in_a_registered_schema() {
        for object in SYSTEM_OBJECTS {
            assert!(
                SYSTEM_SCHEMAS.contains(&object.schema),
                "`{}` names schema `{}`, which is not registered",
                object.canonical_name(),
                object.schema
            );
        }
    }

    #[test]
    fn test_canonical_names_are_unique() {
        let mut seen = HashSet::new();
        for object in SYSTEM_OBJECTS {
            let name = object.canonical_name();
            assert!(seen.insert(name.clone()), "duplicate object `{}`", name);
        }
    }

    #[test]
    fn test_find_is_case_insensitive_and_three_part_only() {
        assert!(find("zyron_sys.core.databases").is_some());
        assert!(find("ZYRON_SYS.CORE.DATABASES").is_some());
        assert!(find("zyron_sys.databases").is_none());
        assert!(find("core.databases").is_none());
        assert!(find("some_table").is_none());
        assert!(find("zyron_sys.core.databases.extra").is_none());
    }

    #[test]
    fn test_resolve_in_search_path_default() {
        let path: Vec<String> = DEFAULT_SEARCH_PATH.iter().map(|s| s.to_string()).collect();
        let tables = resolve_in_search_path("tables", &path).expect("tables resolves");
        assert_eq!(tables.canonical_name(), "zyron_sys.core.tables");
        let activity = resolve_in_search_path("activity", &path).expect("activity resolves");
        assert_eq!(activity.canonical_name(), "zyron_sys.stat.activity");
        // Not on the default path: reachable only with its schema qualifier
        assert!(resolve_in_search_path("watermarks", &path).is_none());
        assert!(resolve_in_search_path("nothing_here", &path).is_none());
    }

    #[test]
    fn test_resolve_in_search_path_ignores_user_schemas() {
        let path = vec!["zyron_test".to_string(), "myschema".to_string()];
        assert!(resolve_in_search_path("tables", &path).is_none());
    }

    /// A name that resolved to nothing says so, and says nothing else.
    #[test]
    fn test_relation_not_found_message_shape() {
        assert_eq!(
            relation_not_found("some_table").to_string(),
            "relation `some_table` does not exist"
        );
    }

    #[test]
    fn test_is_system_catalog_name() {
        assert!(is_system_catalog_name("zyron_sys.core.tables"));
        assert!(is_system_catalog_name("zyron_sys.clustering_status"));
        assert!(!is_system_catalog_name("some_table"));
        assert!(!is_system_catalog_name("zyron_test.users"));
    }
}
