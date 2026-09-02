//! Downgrade eligibility.
//!
//! A downgrade is available only when every migration the upgrade ran can be
//! undone. One one-way step anywhere, in a format, a catalog table, or a
//! user object, and the answer is no, with that step named. A pre-upgrade
//! snapshot is the escape hatch when the answer is no, which is why the
//! snapshot is on by default for a major bump

use zyron_common::format::rewrite::RewriteStatus;
use zyron_common::format::{
    CatalogSchemaRegistry, FormatKind, FormatRegistry, FormatVersion, UpgradeBoard,
};

/// One reason a downgrade is refused
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum OneWayStep {
    /// A format migration that cannot be undone
    Format {
        kind: FormatKind,
        from: FormatVersion,
        to: FormatVersion,
        description: String,
    },
    /// A catalog schema migration that cannot be undone
    Catalog {
        catalog_table: String,
        from: FormatVersion,
        to: FormatVersion,
        description: String,
    },
    /// A user object that was rewritten and cannot be put back
    UserObject {
        object_name: String,
        rewriter: String,
    },
}

impl std::fmt::Display for OneWayStep {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            OneWayStep::Format {
                kind,
                from,
                to,
                description,
            } => write!(
                f,
                "the {kind} migration from {from} to {to} is one way, `{description}`"
            ),
            OneWayStep::Catalog {
                catalog_table,
                from,
                to,
                description,
            } => write!(
                f,
                "the {catalog_table} migration from {from} to {to} is one way, `{description}`"
            ),
            OneWayStep::UserObject {
                object_name,
                rewriter,
            } => write!(
                f,
                "`{object_name}` was rewritten by `{rewriter}`, which has no inverse"
            ),
        }
    }
}

/// Whether a downgrade may run, and why not when it may not
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct DowngradeReport {
    pub from_version: String,
    pub to_version: String,
    pub one_way_steps: Vec<OneWayStep>,
    /// Whether a pre-upgrade snapshot is available to restore instead
    pub snapshot_available: bool,
}

impl DowngradeReport {
    pub fn eligible(&self) -> bool {
        self.one_way_steps.is_empty()
    }

    /// The message an operator gets when the answer is no
    pub fn refusal(&self) -> String {
        let named = self
            .one_way_steps
            .iter()
            .map(|step| step.to_string())
            .collect::<Vec<_>>()
            .join("; ");
        let escape = if self.snapshot_available {
            "Restore from the pre-upgrade backup snapshot instead"
        } else {
            "No pre-upgrade snapshot was taken, so there is no way back from here"
        };
        format!(
            "downgrade from {} to {} is blocked, {named}. {escape}",
            self.from_version, self.to_version
        )
    }
}

/// Works out whether a downgrade is possible
pub fn evaluate(
    formats: &FormatRegistry,
    catalog: &CatalogSchemaRegistry,
    board: &UpgradeBoard,
    from_version: &str,
    to_version: &str,
    migrated_formats: &[(FormatKind, FormatVersion)],
    migrated_tables: &[(String, FormatVersion)],
    snapshot_available: bool,
) -> DowngradeReport {
    let mut report = DowngradeReport {
        from_version: from_version.to_string(),
        to_version: to_version.to_string(),
        snapshot_available,
        one_way_steps: Vec::new(),
    };

    for (kind, from) in migrated_formats {
        let Some(entry) = formats.get(*kind) else {
            continue;
        };
        if entry.reversible_from(*from) {
            continue;
        }
        let step = entry
            .migrators
            .iter()
            .find(|m| m.from >= *from && !m.reversible && !m.no_body_change);
        report.one_way_steps.push(OneWayStep::Format {
            kind: *kind,
            from: *from,
            to: entry.registration.writer_current_version,
            description: step
                .map(|s| s.description.to_string())
                .unwrap_or_else(|| "no inverse is registered".to_string()),
        });
    }

    for (name, from) in migrated_tables {
        let Some(table) = catalog.table(name) else {
            continue;
        };
        if table.reversible_from(*from) {
            continue;
        }
        let step = table.first_one_way_step(*from);
        report.one_way_steps.push(OneWayStep::Catalog {
            catalog_table: name.clone(),
            from: *from,
            to: table.registration.current_schema_version,
            description: step
                .map(|s| s.description.to_string())
                .unwrap_or_else(|| "no inverse is registered".to_string()),
        });
    }

    // A user object that was rewritten and accepted as broken cannot be put
    // back, because what it referenced is gone
    for record in board.rewrites() {
        if record.status == RewriteStatus::AcceptedBroken {
            report.one_way_steps.push(OneWayStep::UserObject {
                object_name: record.object_name.clone(),
                rewriter: record.rewriter_name.clone(),
            });
        }
    }

    report
}

#[cfg(test)]
mod tests {
    use super::*;
    use zyron_common::format::catalog_evolution::{
        CatalogSchemaEvolution, CatalogTableRegistration,
    };
    use zyron_common::format::registry::{
        DeprecationStatus, FormatFixture, FormatMigrator, FormatRegistration, MigrationPolicy,
    };
    use zyron_common::format::rewrite::{ObjectKind, RewriteCategory};
    use zyron_common::format::version::VersionWindow;
    use zyron_common::format::{ALL_FORMAT_KINDS, RewriteRecord};

    fn identity(body: &[u8]) -> std::result::Result<Vec<u8>, String> {
        Ok(body.to_vec())
    }

    fn format_registry(reversible: bool) -> FormatRegistry {
        let registrations: Vec<FormatRegistration> = ALL_FORMAT_KINDS
            .iter()
            .copied()
            .map(|kind| {
                let bumped = kind == FormatKind::HeapPage;
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
                    migration_policy: MigrationPolicy::Lazy,
                    migration_reversible: reversible,
                    binary_version_gate: "0.11.0",
                    deprecation_status: DeprecationStatus::Active,
                    retirement_date: if bumped { Some("2027-01-01") } else { None },
                    downgrade_write_supported: false,
                    notes: "test",
                }
            })
            .collect();
        let migrators = [FormatMigrator {
            kind: FormatKind::HeapPage,
            from: FormatVersion::V1,
            to: FormatVersion::new(1, 1),
            reversible,
            forward: identity,
            backward: if reversible { Some(identity) } else { None },
            no_body_change: false,
            description: "drops the retired slot array",
        }];
        let fixtures = [FormatFixture {
            kind: FormatKind::HeapPage,
            version: FormatVersion::V1,
            bytes: b"",
            path: "fixtures/v1.bin",
        }];
        FormatRegistry::from_parts(&registrations, &migrators, &fixtures).expect("loads")
    }

    fn add_field(row: &mut Vec<u8>) -> std::result::Result<(), String> {
        row.push(b'x');
        Ok(())
    }

    fn drop_field(row: &mut Vec<u8>) -> std::result::Result<(), String> {
        row.pop();
        Ok(())
    }

    fn catalog_registry(reversible: bool) -> CatalogSchemaRegistry {
        CatalogSchemaRegistry::from_parts(
            &[CatalogTableRegistration {
                catalog_table: "zyron_sys.auth.groups",
                current_schema_version: FormatVersion::new(1, 1),
                introduced_in_binary_version: "0.11.0",
                doc: "groups",
            }],
            &[CatalogSchemaEvolution {
                catalog_table: "zyron_sys.auth.groups",
                from_version: FormatVersion::V1,
                to_version: FormatVersion::new(1, 1),
                migration_function_ref: "test::add_field",
                reversible,
                introduced_in_binary_version: "0.11.0",
                forward: add_field,
                backward: if reversible { Some(drop_field) } else { None },
                description: "folds two columns into one",
            }],
        )
        .expect("loads")
    }

    #[test]
    fn test_all_reversible_migrations_allow_a_downgrade() {
        let report = evaluate(
            &format_registry(true),
            &catalog_registry(true),
            &UpgradeBoard::new(),
            "0.12.0",
            "0.11.0",
            &[(FormatKind::HeapPage, FormatVersion::V1)],
            &[("zyron_sys.auth.groups".to_string(), FormatVersion::V1)],
            true,
        );
        assert!(report.eligible());
        assert!(report.one_way_steps.is_empty());
    }

    #[test]
    fn test_a_one_way_format_migration_blocks_and_is_named() {
        let report = evaluate(
            &format_registry(false),
            &catalog_registry(true),
            &UpgradeBoard::new(),
            "0.12.0",
            "0.11.0",
            &[(FormatKind::HeapPage, FormatVersion::V1)],
            &[],
            true,
        );
        assert!(!report.eligible());
        let text = report.refusal();
        assert!(text.contains("heap_page"), "{text}");
        assert!(text.contains("drops the retired slot array"), "{text}");
        assert!(
            text.contains("Restore from the pre-upgrade backup snapshot"),
            "{text}"
        );
    }

    #[test]
    fn test_a_one_way_catalog_migration_blocks_and_is_named() {
        let report = evaluate(
            &format_registry(true),
            &catalog_registry(false),
            &UpgradeBoard::new(),
            "0.12.0",
            "0.11.0",
            &[],
            &[("zyron_sys.auth.groups".to_string(), FormatVersion::V1)],
            false,
        );
        assert!(!report.eligible());
        let text = report.refusal();
        assert!(text.contains("zyron_sys.auth.groups"), "{text}");
        assert!(text.contains("folds two columns into one"), "{text}");
        assert!(text.contains("no way back"), "{text}");
    }

    #[test]
    fn test_an_accepted_broken_object_blocks() {
        let board = UpgradeBoard::new();
        board.set_rewrites(vec![RewriteRecord {
            object_name: "legacy_view".to_string(),
            object_kind: ObjectKind::View,
            rewriter_name: "removed_feature".to_string(),
            category: RewriteCategory::Unsafe,
            status: RewriteStatus::AcceptedBroken,
            before_hash: 1,
            after_hash: 2,
            acknowledged_by: "admin".to_string(),
            updated_at_secs: 0,
            diff: String::new(),
        }]);
        let report = evaluate(
            &format_registry(true),
            &catalog_registry(true),
            &board,
            "0.12.0",
            "0.11.0",
            &[],
            &[],
            true,
        );
        assert!(!report.eligible());
        assert!(report.refusal().contains("legacy_view"));
    }
}
