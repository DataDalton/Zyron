//! Catalog table schema registrations.
//!
//! Every catalog table the server persists declares the schema version its
//! rows are written at. A release that changes a table's row shape bumps the
//! version here and registers the step beside it, and the migration runner
//! moves the stored rows on the next start.
//!
//! Registering a table with no steps is not a formality. It is what makes a
//! later bump a one-line change with a checked path from every earlier
//! version, and what lets `zyron_sys.storage.catalog_schema_evolution`
//! answer for every table rather than only the ones that have changed

use zyron_common::format::catalog_evolution::{CatalogSchemaEvolution, CatalogTableRegistration};
use zyron_common::format::version::FormatVersion;

/// The Zyron version these tables were last reshaped in
const GATE: &str = "0.11.0";

/// The Zyron version the table and index rows gained their schema epoch and
/// build state
const EPOCH_GATE: &str = "0.15.0";

/// The version every catalog table's rows are written at today, for the tables
/// that have not been reshaped since
pub const CATALOG_SCHEMA_VERSION: FormatVersion = FormatVersion::V1;

/// The version a table row is written at.
///
/// V2 carries the schema epoch the table stamps into every tuple, the layout
/// each recorded epoch describes, and the layout its unstamped rows read
/// through. A V1 row cannot be defaulted into V2: without the recorded layout
/// there is nothing to read its existing tuples with.
pub const TABLE_SCHEMA_VERSION: FormatVersion = FormatVersion::new(2, 0);

/// The version an index row is written at. V2 carries the build state.
pub const INDEX_SCHEMA_VERSION: FormatVersion = FormatVersion::new(2, 0);

/// One persistent catalog table
struct Table {
    name: &'static str,
    doc: &'static str,
}

/// Position of `zyron_sys.core.tables` in `TABLES`, checked by a test rather
/// than trusted, because the version below is attached by position
const TABLES_INDEX: usize = 2;

/// Position of `zyron_sys.storage.indexes` in `TABLES`
const INDEXES_INDEX: usize = 4;

/// The version one table's rows are written at.
///
/// Two tables have moved past the shared version, so the row shape a
/// migration has to produce is per table rather than per release.
const fn version_of(index: usize) -> FormatVersion {
    match index {
        TABLES_INDEX => TABLE_SCHEMA_VERSION,
        INDEXES_INDEX => INDEX_SCHEMA_VERSION,
        _ => CATALOG_SCHEMA_VERSION,
    }
}

/// The Zyron version one table's current row shape was introduced in.
const fn gate_of(index: usize) -> &'static str {
    match index {
        TABLES_INDEX | INDEXES_INDEX => EPOCH_GATE,
        _ => GATE,
    }
}

/// Every catalog table the server persists rows for.
///
/// The names are the canonical three-part names of the system catalog, which
/// is what the evolution view reports and what a migration step names
const TABLES: &[Table] = &[
    Table {
        name: "zyron_sys.core.databases",
        doc: "Every catalog on this node",
    },
    Table {
        name: "zyron_sys.core.schemas",
        doc: "Every schema, with the catalog that holds it",
    },
    Table {
        name: "zyron_sys.core.tables",
        doc: "Every live table with its storage format",
    },
    Table {
        name: "zyron_sys.core.columns",
        doc: "Every column of every live table",
    },
    Table {
        name: "zyron_sys.storage.indexes",
        doc: "Every index with its type and key columns",
    },
    Table {
        name: "zyron_sys.core.views",
        doc: "View definitions with their stored SQL",
    },
    Table {
        name: "zyron_sys.core.materialized_views",
        doc: "Materialized view definitions and their backing tables",
    },
    Table {
        name: "zyron_sys.core.functions",
        doc: "SQL scalar functions and their bodies",
    },
    Table {
        name: "zyron_sys.core.aggregates",
        doc: "User-defined aggregates and their state functions",
    },
    Table {
        name: "zyron_sys.core.procedures",
        doc: "Stored procedures and their bodies",
    },
    Table {
        name: "zyron_sys.core.triggers",
        doc: "Triggers with their timing, event, and action",
    },
    Table {
        name: "zyron_sys.core.sequences",
        doc: "Sequences with their bounds and cursor state",
    },
    Table {
        name: "zyron_sys.core.comments",
        doc: "Comments attached to catalog objects",
    },
    Table {
        name: "zyron_sys.core.types",
        doc: "User-defined types and their storage and checks",
    },
    Table {
        name: "zyron_sys.core.collations",
        doc: "Collations with their locale and provider",
    },
    Table {
        name: "zyron_sys.streaming.jobs",
        doc: "Streaming jobs with their query and write mode",
    },
    Table {
        name: "zyron_sys.external_table.sources",
        doc: "External sources with their backend and format",
    },
    Table {
        name: "zyron_sys.external_table.sinks",
        doc: "External sinks with their backend and format",
    },
    Table {
        name: "zyron_sys.cdc.publications",
        doc: "Publications and the tables they carry",
    },
    Table {
        name: "zyron_sys.cdc.subscriptions",
        doc: "Subscriptions with their mode and state",
    },
    Table {
        name: "zyron_sys.core.endpoints",
        doc: "HTTP endpoints with their path, method, and body",
    },
    Table {
        name: "zyron_sys.security.security_maps",
        doc: "External identity to role mappings",
    },
    Table {
        name: "zyron_sys.compliance.legal_holds",
        doc: "Legal holds and what they cover",
    },
    Table {
        name: "zyron_sys.retention.policies",
        doc: "Retention policies per table",
    },
    Table {
        name: "zyron_sys.retention.jobs",
        doc: "Retention job runs and what they removed",
    },
    Table {
        name: "zyron_sys.compliance.log",
        doc: "The tamper-evident audit chain",
    },
    Table {
        name: "zyron_sys.core.schedules",
        doc: "Schedules with their interval and statement",
    },
    Table {
        name: "zyron_sys.core.pipelines",
        doc: "Pipelines and their stages",
    },
    Table {
        name: "zyron_sys.core.event_handlers",
        doc: "Event handlers and what they react to",
    },
    Table {
        name: "zyron_sys.time_travel.version_tags",
        doc: "Named versions of a table",
    },
    Table {
        name: "zyron_sys.search.analyzers",
        doc: "Text analyzers with their tokenizer and filter chains",
    },
    Table {
        name: "zyron_sys.search.synonym_dictionaries",
        doc: "Synonym dictionaries and their rules",
    },
    Table {
        name: "zyron_sys.core.resilience_policies",
        doc: "Bulkheads and retry policies",
    },
    Table {
        name: "zyron_sys.cdc.change_streams",
        doc: "Change streams with their sources, position and mode",
    },
];

/// Submits one table's registration.
///
/// A macro rather than a loop because `inventory::submit!` takes a value at
/// item scope, and a table list that has to be edited in two places is a
/// list that drifts
macro_rules! register_table {
    ($index:expr) => {
        inventory::submit! {
            CatalogTableRegistration {
                catalog_table: TABLES[$index].name,
                current_schema_version: version_of($index),
                introduced_in_binary_version: gate_of($index),
                doc: TABLES[$index].doc,
            }
        }
    };
}

register_table!(0);
register_table!(1);
register_table!(2);
register_table!(3);
register_table!(4);
register_table!(5);
register_table!(6);
register_table!(7);
register_table!(8);
register_table!(9);
register_table!(10);
register_table!(11);
register_table!(12);
register_table!(13);
register_table!(14);
register_table!(15);
register_table!(16);
register_table!(17);
register_table!(18);
register_table!(19);
register_table!(20);
register_table!(21);
register_table!(22);
register_table!(23);
register_table!(24);
register_table!(25);
register_table!(26);
register_table!(27);
register_table!(28);
register_table!(29);
register_table!(30);
register_table!(31);
register_table!(32);
register_table!(33);

/// How many catalog tables are registered
pub const REGISTERED_TABLE_COUNT: usize = TABLES.len();

/// Every registered catalog table name
pub fn table_names() -> Vec<&'static str> {
    TABLES.iter().map(|table| table.name).collect()
}

/// The version one catalog table's rows are written at, by name, None for
/// a name that is not a registered table. The tables reshaped since the
/// shared version stand above it and the rest at it
pub fn table_version(name: &str) -> Option<FormatVersion> {
    TABLES
        .iter()
        .position(|table| table.name == name)
        .map(version_of)
}

// ---------------------------------------------------------------------------
// Schema epoch evolution
// ---------------------------------------------------------------------------

/// Fills a table row's schema epoch fields from the columns it already names.
///
/// This is the catalog half of the heap page version bump. A tuple written
/// before the bump carries epoch 0, and epoch 0 means "read through the layout
/// recorded here", so a table that reached this binary without one has rows
/// nothing can decode. Recording the current column list as both the pre-stamp
/// layout and epoch 1 is exactly right: nothing has changed the shape, so the
/// layout the old rows were written under is the layout the table declares.
///
/// Re-running it changes nothing, because a row that already carries an epoch
/// is left alone.
fn fill_table_schema_epoch(row: &mut Vec<u8>) -> std::result::Result<(), String> {
    let mut entry = crate::schema::TableEntry::from_bytes(row)
        .map_err(|e| format!("a table row did not decode, {e}"))?;
    if entry.schema_epoch != 0 || !entry.schema_epochs.is_empty() {
        return Ok(());
    }
    let layout = entry.current_physical_columns();
    entry.pre_stamp_columns = layout.clone();
    entry.schema_epoch = 1;
    entry.schema_epochs = vec![crate::schema::EpochColumns {
        epoch: 1,
        columns: layout,
    }];
    *row = entry.to_bytes();
    Ok(())
}

/// Marks an index row complete.
///
/// Every index that reached this binary finished being built, because there
/// was no way to record an unfinished one. Ready is what that says.
fn fill_index_state(row: &mut Vec<u8>) -> std::result::Result<(), String> {
    let mut entry = crate::schema::IndexEntry::from_bytes(row)
        .map_err(|e| format!("an index row did not decode, {e}"))?;
    entry.state = crate::schema::IndexState::Ready;
    *row = entry.to_bytes();
    Ok(())
}

inventory::submit! {
    CatalogSchemaEvolution {
        catalog_table: TABLES[TABLES_INDEX].name,
        from_version: CATALOG_SCHEMA_VERSION,
        to_version: TABLE_SCHEMA_VERSION,
        migration_function_ref: "zyron_catalog::catalog_schema::fill_table_schema_epoch",
        // The layout a table's unstamped rows read through cannot be
        // reconstructed once the table's columns move on, so there is no way
        // back
        reversible: false,
        introduced_in_binary_version: EPOCH_GATE,
        forward: fill_table_schema_epoch,
        backward: None,
        description: "records the column layout every tuple stamped 0 decodes through, and \
                      makes it epoch 1",
    }
}

inventory::submit! {
    CatalogSchemaEvolution {
        catalog_table: TABLES[INDEXES_INDEX].name,
        from_version: CATALOG_SCHEMA_VERSION,
        to_version: INDEX_SCHEMA_VERSION,
        migration_function_ref: "zyron_catalog::catalog_schema::fill_index_state",
        reversible: false,
        introduced_in_binary_version: EPOCH_GATE,
        forward: fill_index_state,
        backward: None,
        description: "marks every index that predates online builds complete",
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;

    #[test]
    fn test_every_table_is_registered_once() {
        for name in table_names() {
            let count = inventory::iter::<CatalogTableRegistration>
                .into_iter()
                .filter(|r| r.catalog_table == name)
                .count();
            assert_eq!(count, 1, "`{name}` submitted {count} registrations");
        }
    }

    #[test]
    fn test_the_registration_list_covers_the_table_list() {
        let registered: HashSet<&str> = inventory::iter::<CatalogTableRegistration>
            .into_iter()
            .map(|r| r.catalog_table)
            .collect();
        for name in table_names() {
            assert!(registered.contains(name), "`{name}` is not registered");
        }
        assert_eq!(REGISTERED_TABLE_COUNT, 34);
    }

    #[test]
    fn test_names_are_unique_and_three_part() {
        let mut seen = HashSet::new();
        for name in table_names() {
            assert!(seen.insert(name), "duplicate table `{name}`");
            assert_eq!(
                name.split('.').count(),
                3,
                "`{name}` is not a three-part name"
            );
            assert!(
                name.starts_with("zyron_sys."),
                "`{name}` is not in zyron_sys"
            );
        }
    }

    #[test]
    fn test_the_registry_loads_with_every_table() {
        let registry = zyron_common::format::CatalogSchemaRegistry::load().expect("loads");
        for (index, name) in table_names().into_iter().enumerate() {
            let table = registry
                .table(name)
                .unwrap_or_else(|| panic!("`{name}` is in the registry"));
            assert_eq!(table.registration.current_schema_version, version_of(index));
            if index == TABLES_INDEX || index == INDEXES_INDEX {
                assert_eq!(
                    table.steps.len(),
                    1,
                    "`{name}` moved past the shared version and needs its step"
                );
            } else {
                assert!(
                    table.steps.is_empty(),
                    "`{name}` is at its first version, so it has no steps yet"
                );
            }
        }
    }

    #[test]
    fn test_the_versioned_tables_sit_where_their_indices_say() {
        assert_eq!(TABLES[TABLES_INDEX].name, "zyron_sys.core.tables");
        assert_eq!(TABLES[INDEXES_INDEX].name, "zyron_sys.storage.indexes");
    }

    /// A table row that reached this binary without an epoch is unreadable,
    /// so the step has to produce one, and running it twice has to leave the
    /// first answer alone
    #[test]
    fn test_the_table_step_records_a_layout_and_is_idempotent() {
        use crate::ids::{ColumnId, SchemaId, TableId};
        use crate::schema::{ColumnEntry, TableEntry};

        let mut entry = TableEntry {
            id: TableId(7),
            schema_id: SchemaId(1),
            name: "t".to_string(),
            heap_file_id: 10,
            fsm_file_id: 11,
            columns: vec![ColumnEntry {
                id: ColumnId(0),
                table_id: TableId(7),
                name: "a".to_string(),
                type_id: zyron_common::TypeId::Int32,
                ordinal: 0,
                nullable: true,
                default_expr: None,
                max_length: None,
                fractional_digits: None,
                tz_offset_secs: None,
                element_type: None,
                attrs: Default::default(),
                absent_value: None,
                dropped: false,
            }],
            constraints: Vec::new(),
            created_at: 0,
            versioning_enabled: false,
            scd_type: None,
            system_versioned: false,
            history_table_id: None,
            cdf_enabled: false,
            cdf_retention_days: 0,
            lifecycle: Default::default(),
            columnar: Default::default(),
            dropped_at: None,
            expectations: Vec::new(),
            time_travel_retention_secs: 0,
            lake: Default::default(),
            cluster: Default::default(),
            foreign: Default::default(),
            schema_epoch: 0,
            schema_epochs: Vec::new(),
            pre_stamp_columns: Vec::new(),
            cdf: Default::default(),
        };
        entry.schema_epoch = 0;
        let mut row = entry.to_bytes();

        fill_table_schema_epoch(&mut row).expect("first run");
        let after = TableEntry::from_bytes(&row).expect("decodes");
        assert_eq!(after.schema_epoch, 1);
        assert_eq!(after.schema_epochs.len(), 1);
        assert_eq!(after.pre_stamp_columns.len(), 1);
        assert_eq!(after.pre_stamp_columns[0].column_id, ColumnId(0));

        let before_second = row.clone();
        fill_table_schema_epoch(&mut row).expect("second run");
        assert_eq!(row, before_second, "the step is not idempotent");
    }
}
