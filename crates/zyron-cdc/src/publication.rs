//! Multi-table publications for CDC subscriptions.
//!
//! A publication groups multiple tables into a single subscription that
//! can be consumed through a replication slot. Cross-table ordering is
//! guaranteed because WAL records are globally ordered by LSN.

use std::fs::{self, File};
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use scc::HashMap as SccHashMap;
use serde::{Deserialize, Serialize};
use zyron_common::{Result, ZyronError};

// ---------------------------------------------------------------------------
// Publication
// ---------------------------------------------------------------------------

/// A named publication that groups tables for CDC subscriptions.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Publication {
    pub name: String,
    pub table_ids: Vec<u32>,
    pub all_tables: bool,
    pub include_ddl: bool,
    pub created_at: i64,
}

// ---------------------------------------------------------------------------
// PublicationManager
// ---------------------------------------------------------------------------

/// Manages publications with persistent state.
pub struct PublicationManager {
    publications: SccHashMap<String, Publication>,
    state_file: PathBuf,
}

impl PublicationManager {
    /// Opens or creates the publication manager, loading persisted state.
    pub fn open(data_dir: &Path) -> Result<Self> {
        let state_file = data_dir.join(".zypubs");

        let publications = SccHashMap::new();

        if state_file.exists() {
            let mut file = File::open(&state_file)?;
            let mut data = Vec::new();
            file.read_to_end(&mut data)?;
            if !data.is_empty() {
                let list: Vec<Publication> = serde_json::from_slice(&data).map_err(|e| {
                    ZyronError::CdcStreamError(format!("failed to parse publication state: {e}"))
                })?;
                for pub_entry in list {
                    let _ = publications.insert_sync(pub_entry.name.clone(), pub_entry);
                }
            }
        }

        Ok(Self {
            publications,
            state_file,
        })
    }

    /// Creates a new publication.
    pub fn create_publication(
        &self,
        name: &str,
        table_ids: Vec<u32>,
        all_tables: bool,
        include_ddl: bool,
    ) -> Result<Publication> {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_micros() as i64;

        let publication = Publication {
            name: name.to_string(),
            table_ids,
            all_tables,
            include_ddl,
            created_at: now,
        };

        if self
            .publications
            .insert_sync(name.to_string(), publication.clone())
            .is_err()
        {
            return Err(ZyronError::CdcStreamError(format!(
                "publication already exists: {name}"
            )));
        }

        self.persist()?;
        Ok(publication)
    }

    /// Drops a publication.
    pub fn drop_publication(&self, name: &str) -> Result<()> {
        self.publications
            .remove_sync(name)
            .ok_or_else(|| ZyronError::CdcStreamError(format!("publication not found: {name}")))?;
        self.persist()?;
        Ok(())
    }

    /// Adds a table to an existing publication.
    pub fn alter_publication_add_table(&self, name: &str, table_id: u32) -> Result<()> {
        match self.publications.entry_sync(name.to_string()) {
            scc::hash_map::Entry::Occupied(mut entry) => {
                let pub_entry = entry.get_mut();
                if !pub_entry.table_ids.contains(&table_id) {
                    pub_entry.table_ids.push(table_id);
                }
            }
            scc::hash_map::Entry::Vacant(_) => {
                return Err(ZyronError::CdcStreamError(format!(
                    "publication not found: {name}"
                )));
            }
        }
        self.persist()?;
        Ok(())
    }

    /// Removes a table from a publication.
    pub fn alter_publication_drop_table(&self, name: &str, table_id: u32) -> Result<()> {
        match self.publications.entry_sync(name.to_string()) {
            scc::hash_map::Entry::Occupied(mut entry) => {
                entry.get_mut().table_ids.retain(|&id| id != table_id);
            }
            scc::hash_map::Entry::Vacant(_) => {
                return Err(ZyronError::CdcStreamError(format!(
                    "publication not found: {name}"
                )));
            }
        }
        self.persist()?;
        Ok(())
    }

    /// Lists all publications.
    pub fn list_publications(&self) -> Vec<Publication> {
        let mut result = Vec::new();
        self.publications.iter_sync(|_name, pub_entry| {
            result.push(pub_entry.clone());
            true
        });
        result
    }

    /// Gets a publication by name.
    pub fn get_publication(&self, name: &str) -> Result<Publication> {
        self.publications
            .read_sync(name, |_, pub_entry| pub_entry.clone())
            .ok_or_else(|| ZyronError::CdcStreamError(format!("publication not found: {name}")))
    }

    /// Returns the table IDs for a publication.
    pub fn get_tables_for_publication(&self, name: &str) -> Result<Vec<u32>> {
        let pub_entry = self.get_publication(name)?;
        Ok(pub_entry.table_ids)
    }

    /// Fast check: returns true if a table is in any publication.
    pub fn is_table_published(&self, table_id: u32) -> bool {
        let mut published = false;
        self.publications.iter_sync(|_name, pub_entry| {
            if pub_entry.all_tables || pub_entry.table_ids.contains(&table_id) {
                published = true;
                return false;
            }
            true
        });
        published
    }

    /// Persists publication state to disk using atomic rename.
    fn persist(&self) -> Result<()> {
        let pubs = self.list_publications();
        let data = serde_json::to_vec(&pubs).map_err(|e| {
            ZyronError::CdcStreamError(format!("failed to serialize publication state: {e}"))
        })?;

        let tmp_path = self.state_file.with_extension("zypubs.tmp");
        {
            let mut tmp = File::create(&tmp_path)?;
            tmp.write_all(&data)?;
            tmp.sync_all()?;
        }

        fs::rename(&tmp_path, &self.state_file)?;
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[test]
    fn test_create_and_list_publications() {
        let tmp = TempDir::new().unwrap();
        let mgr = PublicationManager::open(tmp.path()).unwrap();

        let pub_entry = mgr
            .create_publication("pub1", vec![1, 2, 3], false, false)
            .unwrap();
        assert_eq!(pub_entry.name, "pub1");
        assert_eq!(pub_entry.table_ids, vec![1, 2, 3]);

        let pubs = mgr.list_publications();
        assert_eq!(pubs.len(), 1);
    }

    #[test]
    fn test_create_duplicate_fails() {
        let tmp = TempDir::new().unwrap();
        let mgr = PublicationManager::open(tmp.path()).unwrap();

        mgr.create_publication("pub1", vec![1], false, false)
            .unwrap();
        assert!(
            mgr.create_publication("pub1", vec![2], false, false)
                .is_err()
        );
    }

    #[test]
    fn test_drop_publication() {
        let tmp = TempDir::new().unwrap();
        let mgr = PublicationManager::open(tmp.path()).unwrap();

        mgr.create_publication("pub1", vec![1], false, false)
            .unwrap();
        mgr.drop_publication("pub1").unwrap();
        assert!(mgr.list_publications().is_empty());

        assert!(mgr.drop_publication("nonexistent").is_err());
    }

    #[test]
    fn test_alter_add_table() {
        let tmp = TempDir::new().unwrap();
        let mgr = PublicationManager::open(tmp.path()).unwrap();

        mgr.create_publication("pub1", vec![1, 2], false, false)
            .unwrap();
        mgr.alter_publication_add_table("pub1", 3).unwrap();

        let tables = mgr.get_tables_for_publication("pub1").unwrap();
        assert_eq!(tables, vec![1, 2, 3]);

        // Adding duplicate table is a no-op.
        mgr.alter_publication_add_table("pub1", 3).unwrap();
        let tables = mgr.get_tables_for_publication("pub1").unwrap();
        assert_eq!(tables, vec![1, 2, 3]);
    }

    #[test]
    fn test_alter_drop_table() {
        let tmp = TempDir::new().unwrap();
        let mgr = PublicationManager::open(tmp.path()).unwrap();

        mgr.create_publication("pub1", vec![1, 2, 3], false, false)
            .unwrap();
        mgr.alter_publication_drop_table("pub1", 2).unwrap();

        let tables = mgr.get_tables_for_publication("pub1").unwrap();
        assert_eq!(tables, vec![1, 3]);
    }

    #[test]
    fn test_all_tables_publication() {
        let tmp = TempDir::new().unwrap();
        let mgr = PublicationManager::open(tmp.path()).unwrap();

        mgr.create_publication("all_pub", vec![], true, true)
            .unwrap();

        assert!(mgr.is_table_published(42));
        assert!(mgr.is_table_published(999));
    }

    #[test]
    fn test_is_table_published() {
        let tmp = TempDir::new().unwrap();
        let mgr = PublicationManager::open(tmp.path()).unwrap();

        mgr.create_publication("pub1", vec![1, 2], false, false)
            .unwrap();

        assert!(mgr.is_table_published(1));
        assert!(mgr.is_table_published(2));
        assert!(!mgr.is_table_published(3));
    }

    #[test]
    fn test_persistence() {
        let tmp = TempDir::new().unwrap();

        {
            let mgr = PublicationManager::open(tmp.path()).unwrap();
            mgr.create_publication("pub1", vec![1, 2], false, true)
                .unwrap();
        }

        let mgr = PublicationManager::open(tmp.path()).unwrap();
        let pub_entry = mgr.get_publication("pub1").unwrap();
        assert_eq!(pub_entry.table_ids, vec![1, 2]);
        assert!(pub_entry.include_ddl);
    }
}
