//! The format substrate's read path.
//!
//! Fourteen views over the format registry, the signature scheme registry,
//! the deprecation lifecycle, the upgrade board, and the wire protocol
//! registry. Every one of them is computed from the registries themselves
//! rather than from stored rows, so what a view reports and what the binary
//! actually does are the same thing by construction

use zyron_common::ZyronError;
use zyron_common::format::deprecation::{BinaryVersion, DeprecationStage};
use zyron_common::format::registry::MigrationPolicy;
use zyron_common::format::{ALL_FORMAT_KINDS, FormatKind, FormatSubstrate, MAGIC_ALLOCATIONS};

use crate::system_views::{ViewRows, make_field};
use crate::types::{PG_BOOL_OID, PG_INT4_OID, PG_INT8_OID, PG_TEXT_OID};

/// The version this binary reports itself as, which the deprecation stage
/// and the upgrade views are computed against
pub fn running_version() -> BinaryVersion {
    BinaryVersion::parse(env!("CARGO_PKG_VERSION")).unwrap_or_default()
}

fn text(value: impl Into<String>) -> Option<Vec<u8>> {
    Some(value.into().into_bytes())
}

fn number(value: impl ToString) -> Option<Vec<u8>> {
    Some(value.to_string().into_bytes())
}

fn boolean(value: bool) -> Option<Vec<u8>> {
    Some(value.to_string().into_bytes())
}

fn now_secs() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

/// Dispatches a read of one of the substrate's views
pub fn build(schema: &str, object: &str) -> Result<ViewRows, ZyronError> {
    let substrate = zyron_common::format::substrate()?;
    Ok(match (schema, object) {
        ("storage", "format_registry") => build_format_registry(substrate),
        ("storage", "format_documentation") => build_format_documentation(substrate),
        ("storage", "format_migrations") => build_format_migrations(substrate),
        ("storage", "catalog_schema_evolution") => build_catalog_schema_evolution(substrate),
        ("crypto", "scheme_registry") => build_scheme_registry(substrate),
        ("crypto", "artifact_scheme_map") => build_artifact_scheme_map(substrate),
        ("deprecation", "registry") => build_deprecation_registry(substrate),
        ("deprecation", "migration_guides") => build_migration_guides(substrate),
        ("upgrade", "state") => build_upgrade_state(),
        ("upgrade", "history") => build_upgrade_history(),
        ("upgrade", "format_migrations") => build_upgrade_format_migrations(substrate),
        ("upgrade", "user_object_rewrites") => build_user_object_rewrites(),
        ("upgrade", "deprecation_warnings") => build_deprecation_warnings(substrate),
        ("wire", "protocol_versions") => build_protocol_versions(substrate),
        _ => {
            return Err(ZyronError::Internal(format!(
                "`zyron_sys.{schema}.{object}` is registered but has no builder"
            )));
        }
    })
}

/// Whether a name addresses one of these views, which is what sends a read
/// here rather than to the general catalog builders
pub fn owns(schema: &str, object: &str) -> bool {
    matches!(
        (schema, object),
        ("storage", "format_registry")
            | ("storage", "format_documentation")
            | ("storage", "format_migrations")
            | ("storage", "catalog_schema_evolution")
            | ("crypto", "scheme_registry")
            | ("crypto", "artifact_scheme_map")
            | ("deprecation", "registry")
            | ("deprecation", "migration_guides")
            | ("upgrade", "state")
            | ("upgrade", "history")
            | ("upgrade", "format_migrations")
            | ("upgrade", "user_object_rewrites")
            | ("upgrade", "deprecation_warnings")
            | ("wire", "protocol_versions")
    )
}

// ---------------------------------------------------------------------------
// storage
// ---------------------------------------------------------------------------

fn build_format_registry(substrate: &FormatSubstrate) -> ViewRows {
    let fields = vec![
        make_field("format_kind", PG_TEXT_OID, -1),
        make_field("magic", PG_TEXT_OID, -1),
        make_field("owner", PG_TEXT_OID, -1),
        make_field("writer_current_version", PG_TEXT_OID, -1),
        make_field("reader_oldest_version", PG_TEXT_OID, -1),
        make_field("reader_newest_version", PG_TEXT_OID, -1),
        make_field("migration_policy", PG_TEXT_OID, -1),
        make_field("migration_reversible", PG_BOOL_OID, 1),
        make_field("binary_version_gate", PG_TEXT_OID, -1),
        make_field("deprecation_status", PG_TEXT_OID, -1),
        make_field("retirement_date", PG_TEXT_OID, -1),
        make_field("downgrade_write_supported", PG_BOOL_OID, 1),
        make_field("migrator_count", PG_INT4_OID, 4),
        make_field("fixture_count", PG_INT4_OID, 4),
        make_field("notes", PG_TEXT_OID, -1),
    ];
    let mut rows = Vec::with_capacity(ALL_FORMAT_KINDS.len());
    for entry in substrate.formats.entries() {
        let registration = &entry.registration;
        rows.push(vec![
            text(registration.kind.catalog_name()),
            text(registration.kind.magic_str()),
            text(registration.kind.owner()),
            text(registration.writer_current_version.to_string()),
            text(registration.reader_supported_versions.oldest.to_string()),
            text(registration.reader_supported_versions.newest.to_string()),
            text(registration.migration_policy.label()),
            boolean(registration.migration_reversible),
            text(registration.binary_version_gate),
            text(registration.deprecation_status.label()),
            registration.retirement_date.map(|d| d.as_bytes().to_vec()),
            boolean(registration.downgrade_write_supported),
            number(entry.migrators.len()),
            number(entry.fixtures.len()),
            text(registration.notes),
        ]);
    }
    rows.sort_by(|a, b| a[0].cmp(&b[0]));
    (fields, rows)
}

fn build_format_documentation(substrate: &FormatSubstrate) -> ViewRows {
    let fields = vec![
        make_field("format_kind", PG_TEXT_OID, -1),
        make_field("version", PG_TEXT_OID, -1),
        make_field("framing", PG_TEXT_OID, -1),
        make_field("header_bytes", PG_INT4_OID, 4),
        make_field("footer_bytes", PG_INT4_OID, 4),
        make_field("layout", PG_TEXT_OID, -1),
        make_field("is_current", PG_BOOL_OID, 1),
    ];
    let mut rows = Vec::new();
    for entry in substrate.formats.entries() {
        let registration = &entry.registration;
        let kind = registration.kind;
        let framing = kind.framing();
        for version in registration.reader_supported_versions.iter() {
            rows.push(vec![
                text(kind.catalog_name()),
                text(version.to_string()),
                text(framing.label()),
                number(framing.header_bytes()),
                number(framing.footer_bytes()),
                text(framing.layout()),
                boolean(version == registration.writer_current_version),
            ]);
        }
    }
    rows.sort_by(|a, b| a[0].cmp(&b[0]).then(a[1].cmp(&b[1])));
    (fields, rows)
}

fn build_format_migrations(substrate: &FormatSubstrate) -> ViewRows {
    let fields = vec![
        make_field("format_kind", PG_TEXT_OID, -1),
        make_field("from_version", PG_TEXT_OID, -1),
        make_field("to_version", PG_TEXT_OID, -1),
        make_field("policy", PG_TEXT_OID, -1),
        make_field("files_total", PG_INT8_OID, 8),
        make_field("files_done", PG_INT8_OID, 8),
        make_field("bytes_remaining", PG_INT8_OID, 8),
        make_field("percent_complete", PG_TEXT_OID, -1),
        make_field("eta_secs", PG_INT8_OID, 8),
        make_field("failures", PG_INT8_OID, 8),
        make_field("paused", PG_BOOL_OID, 1),
        make_field("finished", PG_BOOL_OID, 1),
        make_field("started_at_secs", PG_INT8_OID, 8),
    ];
    let now = now_secs();
    let rows = substrate
        .migrations
        .runs()
        .into_iter()
        .map(|run| {
            vec![
                text(run.kind.catalog_name()),
                text(run.from.to_string()),
                text(run.to.to_string()),
                text(run.policy.label()),
                number(run.files_total()),
                number(run.files_done()),
                number(run.bytes_remaining()),
                text(format!("{:.1}", run.percent_complete())),
                run.eta_secs(now).map(|s| s.to_string().into_bytes()),
                number(run.failures()),
                boolean(run.is_paused()),
                boolean(run.is_finished()),
                number(run.started_at_secs()),
            ]
        })
        .collect();
    (fields, rows)
}

fn build_catalog_schema_evolution(substrate: &FormatSubstrate) -> ViewRows {
    let fields = vec![
        make_field("catalog_table", PG_TEXT_OID, -1),
        make_field("current_schema_version", PG_TEXT_OID, -1),
        make_field("from_version", PG_TEXT_OID, -1),
        make_field("to_version", PG_TEXT_OID, -1),
        make_field("migration_function_ref", PG_TEXT_OID, -1),
        make_field("reversible", PG_BOOL_OID, 1),
        make_field("introduced_in_binary_version", PG_TEXT_OID, -1),
        make_field("description", PG_TEXT_OID, -1),
    ];
    let mut rows = Vec::new();
    for table in substrate.catalog_schemas.tables() {
        if table.steps.is_empty() {
            rows.push(vec![
                text(table.registration.catalog_table),
                text(table.registration.current_schema_version.to_string()),
                None,
                None,
                None,
                None,
                text(table.registration.introduced_in_binary_version),
                text(table.registration.doc),
            ]);
            continue;
        }
        for step in &table.steps {
            rows.push(vec![
                text(table.registration.catalog_table),
                text(table.registration.current_schema_version.to_string()),
                text(step.from_version.to_string()),
                text(step.to_version.to_string()),
                text(step.migration_function_ref),
                boolean(step.reversible),
                text(step.introduced_in_binary_version),
                text(step.description),
            ]);
        }
    }
    rows.sort_by(|a, b| a[0].cmp(&b[0]).then(a[2].cmp(&b[2])));
    (fields, rows)
}

// ---------------------------------------------------------------------------
// crypto
// ---------------------------------------------------------------------------

fn build_scheme_registry(substrate: &FormatSubstrate) -> ViewRows {
    let fields = vec![
        make_field("scheme_name", PG_TEXT_OID, -1),
        make_field("scheme_id", PG_INT4_OID, 4),
        make_field("category", PG_TEXT_OID, -1),
        make_field("status", PG_TEXT_OID, -1),
        make_field("first_available_version", PG_TEXT_OID, -1),
        make_field("retirement_date", PG_TEXT_OID, -1),
        make_field("last_valid_artifact_expiry", PG_INT8_OID, 8),
        make_field("can_sign", PG_BOOL_OID, 1),
        make_field("can_verify", PG_BOOL_OID, 1),
        make_field("notes", PG_TEXT_OID, -1),
    ];
    let rows = substrate
        .schemes
        .schemes()
        .iter()
        .map(|scheme| {
            vec![
                text(scheme.scheme_name),
                number(scheme.scheme_id.0),
                text(scheme.category.label()),
                text(scheme.status.label()),
                text(scheme.first_available_version),
                scheme.retirement_date.map(|d| d.as_bytes().to_vec()),
                number(
                    substrate
                        .schemes
                        .last_valid_artifact_expiry(scheme.scheme_name),
                ),
                boolean(scheme.status.can_sign()),
                boolean(scheme.status.can_verify()),
                text(scheme.notes),
            ]
        })
        .collect();
    (fields, rows)
}

fn build_artifact_scheme_map(substrate: &FormatSubstrate) -> ViewRows {
    let fields = vec![
        make_field("artifact_kind", PG_TEXT_OID, -1),
        make_field("identifier_encoding", PG_TEXT_OID, -1),
        make_field("current_scheme", PG_TEXT_OID, -1),
        make_field("deprecating_scheme", PG_TEXT_OID, -1),
        make_field("overlap_end_secs", PG_INT8_OID, 8),
        make_field("rotation_in_progress", PG_BOOL_OID, 1),
    ];
    let now = now_secs();
    let rows = substrate
        .schemes
        .bindings()
        .into_iter()
        .map(|binding| {
            vec![
                text(binding.artifact_kind.catalog_name()),
                text(binding.artifact_kind.identifier_encoding().label()),
                text(binding.current_scheme.clone()),
                binding.deprecating_scheme.clone().map(|s| s.into_bytes()),
                binding.overlap_end_secs.map(|s| s.to_string().into_bytes()),
                boolean(binding.rotation_in_progress(now)),
            ]
        })
        .collect();
    (fields, rows)
}

// ---------------------------------------------------------------------------
// deprecation
// ---------------------------------------------------------------------------

fn build_deprecation_registry(substrate: &FormatSubstrate) -> ViewRows {
    let fields = vec![
        make_field("item_kind", PG_TEXT_OID, -1),
        make_field("item_id", PG_TEXT_OID, -1),
        make_field("deprecated_since_version", PG_TEXT_OID, -1),
        make_field("warn_until_version", PG_TEXT_OID, -1),
        make_field("error_since_version", PG_TEXT_OID, -1),
        make_field("removed_since_version", PG_TEXT_OID, -1),
        make_field("stage", PG_TEXT_OID, -1),
        make_field("replacement_ref", PG_TEXT_OID, -1),
        make_field("migration_guide_url", PG_TEXT_OID, -1),
        make_field("summary", PG_TEXT_OID, -1),
    ];
    let running = running_version();
    let mut rows: Vec<_> = substrate
        .deprecations
        .records()
        .iter()
        .map(|record| {
            vec![
                text(record.item_kind.label()),
                text(record.item_id),
                text(record.deprecated_since_version),
                text(record.warn_until_version),
                text(record.error_since_version),
                text(record.removed_since_version),
                text(record.stage(running).label()),
                record.replacement_ref.map(|r| r.as_bytes().to_vec()),
                record.migration_guide_url.map(|u| u.as_bytes().to_vec()),
                text(record.summary),
            ]
        })
        .collect();
    rows.sort_by(|a, b| a[1].cmp(&b[1]));
    (fields, rows)
}

fn build_migration_guides(substrate: &FormatSubstrate) -> ViewRows {
    let fields = vec![
        make_field("item_id", PG_TEXT_OID, -1),
        make_field("item_kind", PG_TEXT_OID, -1),
        make_field("title", PG_TEXT_OID, -1),
        make_field("url", PG_TEXT_OID, -1),
        make_field("body", PG_TEXT_OID, -1),
    ];
    let mut rows: Vec<_> = substrate
        .deprecations
        .guides()
        .into_iter()
        .map(|guide| {
            vec![
                text(guide.item_id),
                text(guide.item_kind.label()),
                text(guide.title),
                guide.url.map(|u| u.as_bytes().to_vec()),
                text(guide.body),
            ]
        })
        .collect();
    rows.sort_by(|a, b| a[0].cmp(&b[0]));
    (fields, rows)
}

// ---------------------------------------------------------------------------
// upgrade
// ---------------------------------------------------------------------------

fn build_upgrade_state() -> ViewRows {
    let fields = vec![
        make_field("node_id", PG_TEXT_OID, -1),
        make_field("phase", PG_TEXT_OID, -1),
        make_field("from_version", PG_TEXT_OID, -1),
        make_field("to_version", PG_TEXT_OID, -1),
        make_field("is_leader", PG_BOOL_OID, 1),
        make_field("started_at_secs", PG_INT8_OID, 8),
        make_field("updated_at_secs", PG_INT8_OID, 8),
        make_field("message", PG_TEXT_OID, -1),
    ];
    let board = zyron_common::format::upgrade_board();
    let mut rows: Vec<Vec<Option<Vec<u8>>>> = board
        .node_states()
        .into_iter()
        .map(|state| {
            vec![
                text(state.node_id),
                text(state.phase.label()),
                text(state.from_version),
                text(state.to_version),
                boolean(state.is_leader),
                number(state.started_at_secs),
                number(state.updated_at_secs),
                text(state.message),
            ]
        })
        .collect();
    if rows.is_empty() {
        // A node that has never upgraded still answers, so SHOW UPGRADE
        // STATE never comes back empty and leaves the operator guessing
        let settings = board.settings();
        rows.push(vec![
            text("this-node"),
            text(zyron_common::format::UpgradePhase::Idle.label()),
            text(env!("CARGO_PKG_VERSION")),
            text(env!("CARGO_PKG_VERSION")),
            boolean(false),
            number(0),
            number(0),
            text(format!(
                "no upgrade in progress, channel {}, auto_upgrade_enabled {}",
                settings.channel, settings.auto_upgrade_enabled
            )),
        ]);
    }
    (fields, rows)
}

fn build_upgrade_history() -> ViewRows {
    let fields = vec![
        make_field("upgrade_id", PG_INT8_OID, 8),
        make_field("from_version", PG_TEXT_OID, -1),
        make_field("to_version", PG_TEXT_OID, -1),
        make_field("channel", PG_TEXT_OID, -1),
        make_field("outcome", PG_TEXT_OID, -1),
        make_field("started_at_secs", PG_INT8_OID, 8),
        make_field("finished_at_secs", PG_INT8_OID, 8),
        make_field("nodes_upgraded", PG_INT4_OID, 4),
        make_field("format_migrations_run", PG_INT4_OID, 4),
        make_field("catalog_migrations_run", PG_INT4_OID, 4),
        make_field("rewrites_applied", PG_INT4_OID, 4),
        make_field("reversible", PG_BOOL_OID, 1),
        make_field("detail", PG_TEXT_OID, -1),
    ];
    let rows = zyron_common::format::upgrade_board()
        .history(usize::MAX)
        .into_iter()
        .map(|entry| {
            vec![
                number(entry.upgrade_id),
                text(entry.from_version),
                text(entry.to_version),
                text(entry.channel),
                text(entry.outcome.label()),
                number(entry.started_at_secs),
                number(entry.finished_at_secs),
                number(entry.nodes_upgraded),
                number(entry.format_migrations_run),
                number(entry.catalog_migrations_run),
                number(entry.rewrites_applied),
                boolean(entry.reversible),
                text(entry.detail),
            ]
        })
        .collect();
    (fields, rows)
}

fn build_upgrade_format_migrations(substrate: &FormatSubstrate) -> ViewRows {
    // The upgrade view answers the same question the storage one does, from
    // the same board, with the policy that decided whether a format was
    // swept at all rather than left for its next write
    let (mut fields, mut rows) = build_format_migrations(substrate);
    fields.push(make_field("swept_by_upgrade", PG_BOOL_OID, 1));
    for row in rows.iter_mut() {
        let eager = row
            .get(3)
            .and_then(|v| v.as_ref())
            .map(|v| v.as_slice() == MigrationPolicy::Eager.label().as_bytes())
            .unwrap_or(false);
        row.push(boolean(eager));
    }
    (fields, rows)
}

fn build_user_object_rewrites() -> ViewRows {
    let fields = vec![
        make_field("object_name", PG_TEXT_OID, -1),
        make_field("object_kind", PG_TEXT_OID, -1),
        make_field("rewriter", PG_TEXT_OID, -1),
        make_field("category", PG_TEXT_OID, -1),
        make_field("status", PG_TEXT_OID, -1),
        make_field("before_hash", PG_INT8_OID, 8),
        make_field("after_hash", PG_INT8_OID, 8),
        make_field("acknowledged_by", PG_TEXT_OID, -1),
        make_field("updated_at_secs", PG_INT8_OID, 8),
        make_field("diff", PG_TEXT_OID, -1),
    ];
    let mut rows: Vec<_> = zyron_common::format::upgrade_board()
        .rewrites()
        .into_iter()
        .map(|record| {
            vec![
                text(record.object_name),
                text(record.object_kind.catalog_name()),
                text(record.rewriter_name),
                text(record.category.label()),
                text(record.status.label()),
                number(record.before_hash),
                number(record.after_hash),
                text(record.acknowledged_by),
                number(record.updated_at_secs),
                text(record.diff),
            ]
        })
        .collect();
    rows.sort_by(|a, b| a[0].cmp(&b[0]));
    (fields, rows)
}

fn build_deprecation_warnings(substrate: &FormatSubstrate) -> ViewRows {
    let fields = vec![
        make_field("item_id", PG_TEXT_OID, -1),
        make_field("item_kind", PG_TEXT_OID, -1),
        make_field("uses_warned", PG_INT8_OID, 8),
        make_field("tenants", PG_TEXT_OID, -1),
        make_field("rate_limit_per_hour", PG_INT4_OID, 4),
        make_field("suppressed_total", PG_INT8_OID, 8),
    ];
    let limit = substrate.warning_limiter.limit();
    let suppressed = substrate.warning_limiter.suppressed();
    let rows = substrate
        .warning_log
        .report(0)
        .into_iter()
        .map(|row| {
            vec![
                text(row.item_id),
                text(row.item_kind.label()),
                number(row.count),
                text(row.tenants.join(",")),
                number(limit),
                number(suppressed),
            ]
        })
        .collect();
    (fields, rows)
}

// ---------------------------------------------------------------------------
// wire
// ---------------------------------------------------------------------------

fn build_protocol_versions(substrate: &FormatSubstrate) -> ViewRows {
    let fields = vec![
        make_field("version", PG_INT4_OID, 4),
        make_field("status", PG_TEXT_OID, -1),
        make_field("accepted", PG_BOOL_OID, 1),
        make_field("introduced_in_binary_version", PG_TEXT_OID, -1),
        make_field("retired_in_binary_version", PG_TEXT_OID, -1),
        make_field("notes", PG_TEXT_OID, -1),
    ];
    let rows = substrate
        .wire_versions
        .versions()
        .iter()
        .map(|version| {
            vec![
                number(version.version),
                text(version.status.label()),
                boolean(version.status.is_accepted()),
                text(version.introduced_in_binary_version),
                version
                    .retired_in_binary_version
                    .map(|v| v.as_bytes().to_vec()),
                text(version.notes),
            ]
        })
        .collect();
    (fields, rows)
}

/// Whether the magic byte allocation table holds a duplicate, which the
/// release check reads and the startup check refuses on
pub fn duplicate_magics() -> Vec<(FormatKind, FormatKind)> {
    let mut duplicates = Vec::new();
    for (i, left) in MAGIC_ALLOCATIONS.iter().enumerate() {
        for right in MAGIC_ALLOCATIONS.iter().skip(i + 1) {
            if left.magic == right.magic {
                duplicates.push((left.kind, right.kind));
            }
        }
    }
    duplicates
}

/// Whether a deprecated item is past its warn window, which the parser and
/// the runtime both ask before acting on a use
pub fn deprecation_stage(item_id: &str) -> Option<DeprecationStage> {
    let substrate = zyron_common::format::substrate().ok()?;
    substrate
        .deprecations
        .find(item_id)
        .map(|record| record.stage(running_version()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_every_registered_view_has_a_builder() {
        for object in zyron_catalog::SYSTEM_OBJECTS {
            if !owns(object.schema, object.object) {
                continue;
            }
            let built = build(object.schema, object.object)
                .unwrap_or_else(|e| panic!("{} has no builder, {e}", object.canonical_name()));
            assert!(
                !built.0.is_empty(),
                "{} produced no columns",
                object.canonical_name()
            );
        }
    }

    #[test]
    fn test_owns_covers_exactly_the_substrate_views() {
        let owned: Vec<String> = zyron_catalog::SYSTEM_OBJECTS
            .iter()
            .filter(|o| owns(o.schema, o.object))
            .map(|o| o.canonical_name())
            .collect();
        assert_eq!(owned.len(), 14, "{owned:?}");
    }

    /// The registry lists what this binary registered. Every format kind is
    /// covered in the server binary, which links every crate; this crate
    /// links a subset, so the view reports that subset and the startup gate
    /// is what requires the whole set
    #[test]
    fn test_the_registry_view_lists_what_is_registered() {
        let substrate = zyron_common::format::substrate().expect("loads");
        let (_, rows) = build_format_registry(substrate);
        assert_eq!(rows.len(), substrate.formats.len());
        assert!(rows.len() <= ALL_FORMAT_KINDS.len());
        assert!(
            rows.iter().any(|row| row[0] == text("heap_page")),
            "the storage formats this crate links are listed"
        );
    }

    #[test]
    fn test_no_duplicate_magics() {
        assert!(duplicate_magics().is_empty());
    }

    #[test]
    fn test_upgrade_state_always_answers() {
        let (_, rows) = build_upgrade_state();
        assert!(!rows.is_empty(), "an idle node still reports its state");
    }
}
