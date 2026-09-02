//! Catalog schema evolution.
//!
//! Every catalog table carries a schema version the storage layer manages,
//! not the user. When a release changes a catalog table's row shape, it
//! registers the new version beside the table and a function that rewrites a
//! row from the previous shape into it. On upgrade the substrate walks the
//! registry, migrates every table whose stored version is behind, and bumps
//! the stored version inside the same transaction.
//!
//! The registration is data. The runner that applies it lives with the
//! catalog storage, because only that layer knows how to read and write a
//! row

use std::fmt;

use super::version::FormatVersion;

/// Rewrites one catalog row in place.
///
/// The row arrives as its stored bytes and leaves as the bytes of the next
/// schema version. A failure names what the row was missing, and the caller
/// rolls the whole table back
pub type RowMigrateFn = fn(&mut Vec<u8>) -> Result<(), String>;

/// One catalog table's current schema shape
#[derive(Debug, Clone, Copy)]
pub struct CatalogTableRegistration {
    /// The three-part name, `zyron_sys.auth.groups`
    pub catalog_table: &'static str,
    /// The version rows are written at today
    pub current_schema_version: FormatVersion,
    /// The Zyron version that introduced `current_schema_version`
    pub introduced_in_binary_version: &'static str,
    /// One line naming what the table holds
    pub doc: &'static str,
}

inventory::collect!(CatalogTableRegistration);

/// One step in a catalog table's schema history
#[derive(Debug, Clone, Copy)]
pub struct CatalogSchemaEvolution {
    pub catalog_table: &'static str,
    /// The version this step starts from
    pub from_version: FormatVersion,
    /// The version this step produces
    pub to_version: FormatVersion,
    /// The fully qualified path of the function, printed by the registry
    /// view so an operator can find it in the tree
    pub migration_function_ref: &'static str,
    pub reversible: bool,
    pub introduced_in_binary_version: &'static str,
    pub forward: RowMigrateFn,
    pub backward: Option<RowMigrateFn>,
    /// One line naming what the step changes
    pub description: &'static str,
}

inventory::collect!(CatalogSchemaEvolution);

/// Why the catalog schema registry refused to load
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CatalogEvolutionError {
    DuplicateTable {
        catalog_table: &'static str,
    },
    UnregisteredTable {
        catalog_table: &'static str,
    },
    DuplicateStep {
        catalog_table: &'static str,
        from_version: FormatVersion,
    },
    MissingStep {
        catalog_table: &'static str,
        from_version: FormatVersion,
        to_version: FormatVersion,
    },
    NonAdjacentStep {
        catalog_table: &'static str,
        from_version: FormatVersion,
        to_version: FormatVersion,
    },
    ReversibleWithoutBackward {
        catalog_table: &'static str,
        from_version: FormatVersion,
    },
    StepPastCurrent {
        catalog_table: &'static str,
        to_version: FormatVersion,
        current: FormatVersion,
    },
}

impl fmt::Display for CatalogEvolutionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CatalogEvolutionError::DuplicateTable { catalog_table } => {
                write!(f, "catalog table `{catalog_table}` is registered twice")
            }
            CatalogEvolutionError::UnregisteredTable { catalog_table } => write!(
                f,
                "a schema evolution step names catalog table `{catalog_table}`, which has no \
                 registration"
            ),
            CatalogEvolutionError::DuplicateStep {
                catalog_table,
                from_version,
            } => write!(
                f,
                "catalog table `{catalog_table}` has two evolution steps starting at \
                 {from_version}"
            ),
            CatalogEvolutionError::MissingStep {
                catalog_table,
                from_version,
                to_version,
            } => write!(
                f,
                "catalog table `{catalog_table}` has no evolution step from {from_version} \
                 toward {to_version}. Add the migration function and submit a \
                 CatalogSchemaEvolution beside the table"
            ),
            CatalogEvolutionError::NonAdjacentStep {
                catalog_table,
                from_version,
                to_version,
            } => write!(
                f,
                "catalog table `{catalog_table}` has an evolution step from {from_version} \
                 to {to_version}, which are not adjacent versions"
            ),
            CatalogEvolutionError::ReversibleWithoutBackward {
                catalog_table,
                from_version,
            } => write!(
                f,
                "catalog table `{catalog_table}` marks its step at {from_version} reversible \
                 but carries no backward function"
            ),
            CatalogEvolutionError::StepPastCurrent {
                catalog_table,
                to_version,
                current,
            } => write!(
                f,
                "catalog table `{catalog_table}` has an evolution step producing \
                 {to_version}, past its registered current version {current}"
            ),
        }
    }
}

impl std::error::Error for CatalogEvolutionError {}

/// One table's registration with its ordered evolution steps
#[derive(Debug, Clone)]
pub struct CatalogTableEvolution {
    pub registration: CatalogTableRegistration,
    pub steps: Vec<CatalogSchemaEvolution>,
}

impl CatalogTableEvolution {
    /// The chain that takes a row from a stored version to the current one
    pub fn plan(
        &self,
        from: FormatVersion,
    ) -> Result<Vec<CatalogSchemaEvolution>, CatalogEvolutionError> {
        let target = self.registration.current_schema_version;
        let mut chain = Vec::new();
        let mut cursor = from;
        while cursor < target {
            let step = self.steps.iter().find(|s| s.from_version == cursor).ok_or(
                CatalogEvolutionError::MissingStep {
                    catalog_table: self.registration.catalog_table,
                    from_version: cursor,
                    to_version: target,
                },
            )?;
            cursor = step.to_version;
            chain.push(*step);
        }
        Ok(chain)
    }

    /// Whether every step from a stored version up to the current one can be
    /// undone, which is what decides downgrade eligibility
    pub fn reversible_from(&self, from: FormatVersion) -> bool {
        self.plan(from)
            .map(|chain| chain.iter().all(|s| s.reversible))
            .unwrap_or(false)
    }

    /// The first one-way step between a stored version and the current one
    pub fn first_one_way_step(&self, from: FormatVersion) -> Option<CatalogSchemaEvolution> {
        self.plan(from).ok()?.into_iter().find(|s| !s.reversible)
    }

    /// Applies the chain to one row's bytes
    pub fn migrate_row(
        &self,
        from: FormatVersion,
        row: &mut Vec<u8>,
    ) -> Result<FormatVersion, String> {
        let chain = self.plan(from).map_err(|e| e.to_string())?;
        for step in &chain {
            (step.forward)(row).map_err(|reason| {
                format!(
                    "{} migrating {} from {} to {}, {reason}",
                    step.migration_function_ref,
                    self.registration.catalog_table,
                    step.from_version,
                    step.to_version
                )
            })?;
        }
        Ok(self.registration.current_schema_version)
    }

    /// Undoes the chain, which is what a downgrade runs
    pub fn revert_row(
        &self,
        to: FormatVersion,
        row: &mut Vec<u8>,
    ) -> Result<FormatVersion, String> {
        let chain = self.plan(to).map_err(|e| e.to_string())?;
        for step in chain.iter().rev() {
            let Some(backward) = step.backward else {
                return Err(format!(
                    "the {} migration from {} to {} is one way, `{}`",
                    self.registration.catalog_table,
                    step.from_version,
                    step.to_version,
                    step.description
                ));
            };
            backward(row).map_err(|reason| {
                format!(
                    "{} reverting {} from {} to {}, {reason}",
                    step.migration_function_ref,
                    self.registration.catalog_table,
                    step.to_version,
                    step.from_version
                )
            })?;
        }
        Ok(to)
    }
}

/// Every catalog table's schema evolution, checked once at startup
#[derive(Debug, Default)]
pub struct CatalogSchemaRegistry {
    tables: Vec<CatalogTableEvolution>,
}

impl CatalogSchemaRegistry {
    pub fn load() -> Result<CatalogSchemaRegistry, CatalogEvolutionError> {
        let tables: Vec<CatalogTableRegistration> = inventory::iter::<CatalogTableRegistration>
            .into_iter()
            .copied()
            .collect();
        let steps: Vec<CatalogSchemaEvolution> = inventory::iter::<CatalogSchemaEvolution>
            .into_iter()
            .copied()
            .collect();
        CatalogSchemaRegistry::from_parts(&tables, &steps)
    }

    pub fn from_parts(
        tables: &[CatalogTableRegistration],
        steps: &[CatalogSchemaEvolution],
    ) -> Result<CatalogSchemaRegistry, CatalogEvolutionError> {
        let mut built: Vec<CatalogTableEvolution> = Vec::with_capacity(tables.len());
        for registration in tables {
            if built.iter().any(|t| {
                t.registration
                    .catalog_table
                    .eq_ignore_ascii_case(registration.catalog_table)
            }) {
                return Err(CatalogEvolutionError::DuplicateTable {
                    catalog_table: registration.catalog_table,
                });
            }
            built.push(CatalogTableEvolution {
                registration: *registration,
                steps: Vec::new(),
            });
        }

        for step in steps {
            let table = built
                .iter_mut()
                .find(|t| {
                    t.registration
                        .catalog_table
                        .eq_ignore_ascii_case(step.catalog_table)
                })
                .ok_or(CatalogEvolutionError::UnregisteredTable {
                    catalog_table: step.catalog_table,
                })?;
            if table
                .steps
                .iter()
                .any(|s| s.from_version == step.from_version)
            {
                return Err(CatalogEvolutionError::DuplicateStep {
                    catalog_table: step.catalog_table,
                    from_version: step.from_version,
                });
            }
            if !is_adjacent(step.from_version, step.to_version) {
                return Err(CatalogEvolutionError::NonAdjacentStep {
                    catalog_table: step.catalog_table,
                    from_version: step.from_version,
                    to_version: step.to_version,
                });
            }
            if step.reversible && step.backward.is_none() {
                return Err(CatalogEvolutionError::ReversibleWithoutBackward {
                    catalog_table: step.catalog_table,
                    from_version: step.from_version,
                });
            }
            if step.to_version > table.registration.current_schema_version {
                return Err(CatalogEvolutionError::StepPastCurrent {
                    catalog_table: step.catalog_table,
                    to_version: step.to_version,
                    current: table.registration.current_schema_version,
                });
            }
            table.steps.push(*step);
        }

        for table in built.iter_mut() {
            table.steps.sort_by_key(|s| s.from_version);
            // A table whose current version is past 1.0 has to be reachable
            // from 1.0, so a row written by any prior release can be moved
            table.plan(FormatVersion::V1)?;
        }

        built.sort_by(|a, b| {
            a.registration
                .catalog_table
                .cmp(b.registration.catalog_table)
        });
        Ok(CatalogSchemaRegistry { tables: built })
    }

    pub fn tables(&self) -> &[CatalogTableEvolution] {
        &self.tables
    }

    pub fn table(&self, catalog_table: &str) -> Option<&CatalogTableEvolution> {
        self.tables.iter().find(|t| {
            t.registration
                .catalog_table
                .eq_ignore_ascii_case(catalog_table)
        })
    }

    /// Tables whose stored version is behind the registered current one
    pub fn tables_needing_migration<'a>(
        &'a self,
        stored: &'a dyn Fn(&str) -> FormatVersion,
    ) -> Vec<&'a CatalogTableEvolution> {
        self.tables
            .iter()
            .filter(|t| {
                stored(t.registration.catalog_table) < t.registration.current_schema_version
            })
            .collect()
    }
}

/// Whether `to` follows `from` with no gap
fn is_adjacent(from: FormatVersion, to: FormatVersion) -> bool {
    if to <= from {
        return false;
    }
    if to.major == from.major {
        return to.minor == from.minor + 1;
    }
    to.major == from.major + 1 && to.minor == 0
}

#[cfg(test)]
mod tests {
    use super::*;

    fn add_field(row: &mut Vec<u8>) -> Result<(), String> {
        row.extend_from_slice(b"|default");
        Ok(())
    }

    fn drop_field(row: &mut Vec<u8>) -> Result<(), String> {
        match row.len().checked_sub(8) {
            Some(cut) if &row[cut..] == b"|default" => {
                row.truncate(cut);
                Ok(())
            }
            _ => Err("row does not end with the added field".to_string()),
        }
    }

    fn groups() -> CatalogTableRegistration {
        CatalogTableRegistration {
            catalog_table: "zyron_sys.auth.groups",
            current_schema_version: FormatVersion::new(1, 1),
            introduced_in_binary_version: "0.11.0",
            doc: "Groups and the roles they carry",
        }
    }

    fn step(reversible: bool) -> CatalogSchemaEvolution {
        CatalogSchemaEvolution {
            catalog_table: "zyron_sys.auth.groups",
            from_version: FormatVersion::V1,
            to_version: FormatVersion::new(1, 1),
            migration_function_ref: "zyron_catalog::tables::auth::groups::migrate_v1_to_v2",
            reversible,
            introduced_in_binary_version: "0.11.0",
            forward: add_field,
            backward: if reversible { Some(drop_field) } else { None },
            description: "adds the source column with a default",
        }
    }

    #[test]
    fn test_registry_loads_and_plans() {
        let registry =
            CatalogSchemaRegistry::from_parts(&[groups()], &[step(true)]).expect("loads");
        let table = registry.table("zyron_sys.auth.groups").expect("registered");
        assert_eq!(table.plan(FormatVersion::V1).expect("plans").len(), 1);
        assert!(
            table
                .plan(FormatVersion::new(1, 1))
                .expect("plans")
                .is_empty()
        );
        assert!(table.reversible_from(FormatVersion::V1));
    }

    #[test]
    fn test_row_migrates_and_reverts() {
        let registry =
            CatalogSchemaRegistry::from_parts(&[groups()], &[step(true)]).expect("loads");
        let table = registry.table("zyron_sys.auth.groups").expect("registered");
        let mut row = b"admins".to_vec();
        let now = table
            .migrate_row(FormatVersion::V1, &mut row)
            .expect("migrates");
        assert_eq!(now, FormatVersion::new(1, 1));
        assert_eq!(row, b"admins|default");
        table
            .revert_row(FormatVersion::V1, &mut row)
            .expect("reverts");
        assert_eq!(row, b"admins");
    }

    #[test]
    fn test_one_way_step_blocks_the_revert() {
        let registry =
            CatalogSchemaRegistry::from_parts(&[groups()], &[step(false)]).expect("loads");
        let table = registry.table("zyron_sys.auth.groups").expect("registered");
        assert!(!table.reversible_from(FormatVersion::V1));
        let one_way = table
            .first_one_way_step(FormatVersion::V1)
            .expect("names the step");
        assert_eq!(one_way.from_version, FormatVersion::V1);
        let mut row = b"admins|default".to_vec();
        let err = table
            .revert_row(FormatVersion::V1, &mut row)
            .expect_err("blocked");
        assert!(err.contains("one way"), "{err}");
        assert!(err.contains("zyron_sys.auth.groups"), "{err}");
    }

    #[test]
    fn test_missing_step_refuses_the_load() {
        let err = CatalogSchemaRegistry::from_parts(&[groups()], &[]).expect_err("refused");
        assert!(matches!(err, CatalogEvolutionError::MissingStep { .. }));
    }

    #[test]
    fn test_step_for_unregistered_table_refuses_the_load() {
        let mut orphan = step(true);
        orphan.catalog_table = "zyron_sys.auth.nothing";
        assert!(matches!(
            CatalogSchemaRegistry::from_parts(&[groups()], &[step(true), orphan]),
            Err(CatalogEvolutionError::UnregisteredTable { .. })
        ));
    }

    #[test]
    fn test_step_past_current_refuses_the_load() {
        let mut ahead = step(true);
        ahead.from_version = FormatVersion::new(1, 1);
        ahead.to_version = FormatVersion::new(1, 2);
        assert!(matches!(
            CatalogSchemaRegistry::from_parts(&[groups()], &[step(true), ahead]),
            Err(CatalogEvolutionError::StepPastCurrent { .. })
        ));
    }

    #[test]
    fn test_tables_needing_migration_reads_the_stored_version() {
        let registry =
            CatalogSchemaRegistry::from_parts(&[groups()], &[step(true)]).expect("loads");
        let behind = |_: &str| FormatVersion::V1;
        assert_eq!(registry.tables_needing_migration(&behind).len(), 1);
        let current = |_: &str| FormatVersion::new(1, 1);
        assert!(registry.tables_needing_migration(&current).is_empty());
    }
}
