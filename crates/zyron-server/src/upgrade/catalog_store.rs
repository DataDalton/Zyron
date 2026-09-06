//! The catalog's stored rows, as the schema migration runner reads and
//! writes them.
//!
//! The runner is synchronous and works table by table. The catalog storage
//! is asynchronous, so this store blocks on it from the blocking thread the
//! runner is given. Each table's stored version is what the upgrade journal
//! recorded for it, and a replacement is journaled first: the rows as they
//! were go into the journal, the heap is rewritten, and the journal records
//! the new version and drops the pre-image. A crash between the first and
//! the last of those is undone on the next start from the pre-image

use std::sync::Arc;

use zyron_catalog::Catalog;
use zyron_common::format::FormatVersion;
use zyron_common::{Result, ZyronError};

use super::journal::{CatalogPreimage, Journal, rows_from_hex, rows_to_hex};
use super::migrations::CatalogTableStore;

/// The catalog rows behind the migration runner
pub struct HeapCatalogTableStore {
    catalog: Arc<Catalog>,
    journal: Arc<Journal>,
    /// The runtime the storage's futures run on, entered from the blocking
    /// thread the migration runner uses
    runtime: tokio::runtime::Handle,
}

impl HeapCatalogTableStore {
    /// Builds the store. Called on a runtime thread, whose handle is what
    /// the blocking calls later enter
    pub fn new(catalog: Arc<Catalog>, journal: Arc<Journal>) -> Self {
        Self {
            catalog,
            journal,
            runtime: tokio::runtime::Handle::current(),
        }
    }

    /// Puts back the rows of a table whose replacement did not finish, if
    /// the journal holds any. Run once at start, before any migration
    pub async fn restore_preimage(catalog: &Catalog, journal: &Journal) -> Result<Option<String>> {
        let Some(preimage) = journal.read().catalog_preimage else {
            return Ok(None);
        };
        let rows = rows_from_hex(&preimage.rows_hex)?;
        catalog
            .storage()
            .replace_raw_rows(&preimage.catalog_table, rows)
            .await?;
        journal.update(|j| {
            j.set_catalog_version(&preimage.catalog_table, preimage.version);
            j.catalog_preimage = None;
        })?;
        Ok(Some(preimage.catalog_table))
    }
}

impl CatalogTableStore for HeapCatalogTableStore {
    fn stored_version(&self, catalog_table: &str) -> FormatVersion {
        match self.journal.read().catalog_version(catalog_table) {
            Some(version) => FormatVersion::from_u32(version),
            None => FormatVersion::V1,
        }
    }

    fn rows(&self, catalog_table: &str) -> Result<Vec<Vec<u8>>> {
        let catalog = Arc::clone(&self.catalog);
        let table = catalog_table.to_string();
        self.runtime
            .block_on(async move { catalog.storage().raw_rows(&table).await })
    }

    fn replace_rows(
        &self,
        catalog_table: &str,
        rows: Vec<Vec<u8>>,
        version: FormatVersion,
    ) -> Result<()> {
        let before = self.rows(catalog_table)?;
        let before_version = self.stored_version(catalog_table);
        let table = catalog_table.to_string();
        self.journal.update(|j| {
            j.catalog_preimage = Some(CatalogPreimage {
                catalog_table: table.clone(),
                version: before_version.as_u32(),
                rows_hex: rows_to_hex(&before),
            });
        })?;
        let catalog = Arc::clone(&self.catalog);
        let table_for_write = table.clone();
        let written = self.runtime.block_on(async move {
            catalog
                .storage()
                .replace_raw_rows(&table_for_write, rows)
                .await
        });
        if let Err(e) = written {
            // The heap may hold both sets now. The pre-image stays in the
            // journal so the next start puts the old rows back
            return Err(ZyronError::Internal(format!(
                "{table} was not replaced, {e}. The rows as they were are held in the \
                 upgrade journal and are restored on the next start"
            )));
        }
        self.journal.update(|j| {
            j.set_catalog_version(&table, version.as_u32());
            j.catalog_preimage = None;
        })
    }
}
