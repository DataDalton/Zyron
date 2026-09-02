//! The format registry and the catalog schema evolution registry.
//!
//! Covers validation items 4, 15, and 16 from the catalog's side: the
//! startup gate refusing an incomplete registration set, a catalog table
//! migrating every row when its schema version is bumped, and a one-way
//! catalog migration blocking a downgrade with the step named

use zyron_common::format::catalog_evolution::{
    CatalogEvolutionError, CatalogSchemaEvolution, CatalogSchemaRegistry, CatalogTableRegistration,
};
use zyron_common::format::registry::{
    DeprecationStatus, FormatFixture, FormatMigrator, FormatRegistration, MigrationPolicy,
    RegistryError,
};
use zyron_common::format::version::VersionWindow;
use zyron_common::format::{ALL_FORMAT_KINDS, FormatKind, FormatRegistry, FormatVersion};

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

/// Item 4. A complete set passes the startup gate, every kind reporting a
/// non-empty writer version and reader window
#[test]
fn a_complete_registration_set_passes_the_startup_gate() {
    let registry = FormatRegistry::from_parts(&full_set(), &[], &[]).expect("loads");
    registry.verify_complete().expect("complete");
    assert!(registry.missing().is_empty());
    assert_eq!(registry.len(), ALL_FORMAT_KINDS.len());
    for kind in ALL_FORMAT_KINDS {
        let entry = registry.get(*kind).expect("registered");
        assert_eq!(entry.registration.writer_current_version, FormatVersion::V1);
        assert!(
            entry
                .registration
                .reader_supported_versions
                .contains(FormatVersion::V1)
        );
        assert!(!entry.registration.binary_version_gate.is_empty());
    }
}

/// Item 4, revert-proved. Removing one registration makes the startup gate
/// refuse, naming the format and what is missing
#[test]
fn removing_a_registration_makes_startup_refuse() {
    for missing in [
        FormatKind::HeapPage,
        FormatKind::WalSegment,
        FormatKind::MvccClog,
        FormatKind::AppImageBundle,
    ] {
        let mut set = full_set();
        set.retain(|r| r.kind != missing);
        let registry = FormatRegistry::from_parts(&set, &[], &[]).expect("loads a subset");
        assert_eq!(registry.missing(), vec![missing]);
        match registry.verify_complete() {
            Err(RegistryError::Missing { kind }) => {
                assert_eq!(kind, missing);
                let text = RegistryError::Missing { kind }.to_string();
                assert!(text.contains(missing.catalog_name()), "{text}");
                assert!(text.contains("writer_current_version"), "{text}");
                assert!(text.contains("before the server can start"), "{text}");
            }
            other => panic!("expected Missing for {missing}, got {other:?}"),
        }
    }
}

/// A registration whose writer sits outside its reader window is refused at
/// load, because the binary could not read what it writes
#[test]
fn a_writer_outside_its_reader_window_is_refused() {
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

/// A version bump with no migrator is refused at load, and the message says
/// what to add
#[test]
fn a_bump_without_a_migrator_is_refused() {
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
        Err(err @ RegistryError::MissingMigrator { .. }) => {
            let text = err.to_string();
            assert!(text.contains("heap_page"), "{text}");
            assert!(text.contains("migrations/v0_to_v1.rs"), "{text}");
            assert!(text.contains("no_body_change"), "{text}");
        }
        other => panic!("expected MissingMigrator, got {other:?}"),
    }
}

/// A version bump with a migrator but no fixture is refused, and the message
/// says where the fixture goes
#[test]
fn a_bump_without_a_fixture_is_refused() {
    fn identity(body: &[u8]) -> Result<Vec<u8>, String> {
        Ok(body.to_vec())
    }
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
        Err(err @ RegistryError::MissingFixture { .. }) => {
            let text = err.to_string();
            assert!(text.contains("heap_page"), "{text}");
            assert!(text.contains("fixtures/v0.bin"), "{text}");
        }
        other => panic!("expected MissingFixture, got {other:?}"),
    }

    // Adding the fixture makes the same set load
    let fixtures = [FormatFixture {
        kind: FormatKind::HeapPage,
        version: FormatVersion::V1,
        bytes: b"",
        path: "fixtures/v1.bin",
    }];
    FormatRegistry::from_parts(&set, &migrators, &fixtures).expect("loads with the fixture");
}

/// The registry this binary carries loads and holds every catalog table
#[test]
fn the_catalog_schema_registry_loads_with_every_table() {
    let registry = CatalogSchemaRegistry::load().expect("loads");
    assert_eq!(
        registry.tables().len(),
        zyron_catalog::catalog_schema::REGISTERED_TABLE_COUNT
    );
    for name in zyron_catalog::catalog_schema::table_names() {
        let table = registry.table(name).unwrap_or_else(|| panic!("{name}"));
        assert_eq!(
            table.registration.current_schema_version,
            zyron_catalog::catalog_schema::CATALOG_SCHEMA_VERSION
        );
        // Nothing has been bumped yet, so every table plans an empty chain
        assert!(table.plan(FormatVersion::V1).expect("plans").is_empty());
    }
}

fn add_source_column(row: &mut Vec<u8>) -> Result<(), String> {
    row.extend_from_slice(b"|local");
    Ok(())
}

fn drop_source_column(row: &mut Vec<u8>) -> Result<(), String> {
    match row.len().checked_sub(6) {
        Some(cut) if &row[cut..] == b"|local" => {
            row.truncate(cut);
            Ok(())
        }
        _ => Err("row does not end with the source column".to_string()),
    }
}

fn groups(reversible: bool) -> CatalogSchemaRegistry {
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
            migration_function_ref: "zyron_catalog::tables::auth::groups::migrate_v1_to_v2",
            reversible,
            introduced_in_binary_version: "0.11.0",
            forward: add_source_column,
            backward: if reversible {
                Some(drop_source_column)
            } else {
                None
            },
            description: "adds the source column with a default",
        }],
    )
    .expect("loads")
}

/// Item 15. Bumping a catalog table's schema version and registering the
/// step migrates every row, and each row comes out with the new field
#[test]
fn a_catalog_schema_bump_migrates_every_row() {
    let registry = groups(true);
    let table = registry.table("zyron_sys.auth.groups").expect("registered");
    assert_eq!(table.plan(FormatVersion::V1).expect("plans").len(), 1);

    let mut rows: Vec<Vec<u8>> = vec![b"admins".to_vec(), b"readers".to_vec(), b"writers".to_vec()];
    for row in rows.iter_mut() {
        let now = table.migrate_row(FormatVersion::V1, row).expect("migrates");
        assert_eq!(now, FormatVersion::new(1, 1));
    }
    assert_eq!(
        rows,
        vec![
            b"admins|local".to_vec(),
            b"readers|local".to_vec(),
            b"writers|local".to_vec()
        ],
        "every row carries the new field with its default"
    );

    // A row already at the current version is left alone
    let mut current = b"admins|local".to_vec();
    table
        .migrate_row(FormatVersion::new(1, 1), &mut current)
        .expect("no work");
    assert_eq!(current, b"admins|local");
}

/// Item 16. A migration marked one way blocks the reverse and names the
/// migration in the refusal
#[test]
fn a_one_way_catalog_migration_blocks_the_downgrade() {
    let registry = groups(false);
    let table = registry.table("zyron_sys.auth.groups").expect("registered");
    assert!(!table.reversible_from(FormatVersion::V1));

    let step = table
        .first_one_way_step(FormatVersion::V1)
        .expect("names the step");
    assert_eq!(step.from_version, FormatVersion::V1);
    assert!(!step.reversible);

    let mut row = b"admins|local".to_vec();
    let err = table
        .revert_row(FormatVersion::V1, &mut row)
        .expect_err("blocked");
    assert!(err.contains("one way"), "{err}");
    assert!(err.contains("zyron_sys.auth.groups"), "{err}");
    assert!(err.contains("adds the source column"), "{err}");
    assert_eq!(row, b"admins|local", "the row is untouched");
}

/// A reversible migration can be undone, which is what makes the downgrade
/// path available
#[test]
fn a_reversible_catalog_migration_can_be_undone() {
    let registry = groups(true);
    let table = registry.table("zyron_sys.auth.groups").expect("registered");
    assert!(table.reversible_from(FormatVersion::V1));
    assert!(table.first_one_way_step(FormatVersion::V1).is_none());

    let mut row = b"admins".to_vec();
    table
        .migrate_row(FormatVersion::V1, &mut row)
        .expect("forward");
    assert_eq!(row, b"admins|local");
    table.revert_row(FormatVersion::V1, &mut row).expect("back");
    assert_eq!(row, b"admins");
}

/// A step naming a table nobody registered is refused, so a migrator cannot
/// be orphaned by a rename
#[test]
fn a_step_for_an_unregistered_table_is_refused() {
    let err = CatalogSchemaRegistry::from_parts(
        &[],
        &[CatalogSchemaEvolution {
            catalog_table: "zyron_sys.auth.nothing",
            from_version: FormatVersion::V1,
            to_version: FormatVersion::new(1, 1),
            migration_function_ref: "test",
            reversible: true,
            introduced_in_binary_version: "0.11.0",
            forward: add_source_column,
            backward: Some(drop_source_column),
            description: "orphan",
        }],
    )
    .expect_err("refused");
    assert!(matches!(
        err,
        CatalogEvolutionError::UnregisteredTable { .. }
    ));
}

/// A table whose current version cannot be reached from the first version is
/// refused at load, so no row can be stranded
#[test]
fn a_table_with_a_gap_in_its_chain_is_refused() {
    let err = CatalogSchemaRegistry::from_parts(
        &[CatalogTableRegistration {
            catalog_table: "zyron_sys.auth.groups",
            current_schema_version: FormatVersion::new(1, 2),
            introduced_in_binary_version: "0.11.0",
            doc: "groups",
        }],
        &[CatalogSchemaEvolution {
            catalog_table: "zyron_sys.auth.groups",
            from_version: FormatVersion::new(1, 1),
            to_version: FormatVersion::new(1, 2),
            migration_function_ref: "test",
            reversible: true,
            introduced_in_binary_version: "0.11.0",
            forward: add_source_column,
            backward: Some(drop_source_column),
            description: "the second step, with the first missing",
        }],
    )
    .expect_err("refused");
    assert!(matches!(err, CatalogEvolutionError::MissingStep { .. }));
}
