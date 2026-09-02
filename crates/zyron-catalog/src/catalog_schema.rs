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

use zyron_common::format::catalog_evolution::CatalogTableRegistration;
use zyron_common::format::version::FormatVersion;

/// The Zyron version these tables were last reshaped in
const GATE: &str = "0.11.0";

/// The version every catalog table's rows are written at today
pub const CATALOG_SCHEMA_VERSION: FormatVersion = FormatVersion::V1;

/// One persistent catalog table
struct Table {
    name: &'static str,
    doc: &'static str,
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
                current_schema_version: CATALOG_SCHEMA_VERSION,
                introduced_in_binary_version: GATE,
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

/// How many catalog tables are registered
pub const REGISTERED_TABLE_COUNT: usize = TABLES.len();

/// Every registered catalog table name
pub fn table_names() -> Vec<&'static str> {
    TABLES.iter().map(|table| table.name).collect()
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
        assert_eq!(REGISTERED_TABLE_COUNT, 33);
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
        for name in table_names() {
            let table = registry
                .table(name)
                .unwrap_or_else(|| panic!("`{name}` is in the registry"));
            assert_eq!(
                table.registration.current_schema_version,
                CATALOG_SCHEMA_VERSION
            );
            assert!(
                table.steps.is_empty(),
                "`{name}` is at its first version, so it has no steps yet"
            );
        }
    }
}
