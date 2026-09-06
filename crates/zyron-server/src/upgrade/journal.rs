//! The durable half of the upgrade board.
//!
//! The board lives in memory and dies with the process. What has to outlive
//! a restart, which an upgrade is, lives here: the board's rows, history,
//! and rewrite queue, the restart the node is in the middle of, the last
//! upgrade that completed, which a rollback is measured against, the schema
//! version each catalog table's rows are stored at, and the rows of a
//! catalog table mid-replacement. The file is rewritten whole on every
//! change through a temp file and a rename, so a crash mid-write leaves the
//! previous journal in place.
//!
//! Settings are carried in the snapshot too, but the config is what seeds
//! them at boot: `ALTER SYSTEM SET` persists an upgrade setting under the
//! `[upgrade]` section, so the config already holds the latest value

use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};
use zyron_common::format::catalog_evolution::CatalogSchemaRegistry;
use zyron_common::format::registry::FormatRegistry;
use zyron_common::format::upgrade::UpgradeBoardSnapshot;
use zyron_common::format::{FormatKind, HealthBaseline, UpgradeBoard, envelope, migration};
use zyron_common::{Result, ZyronError};

use crate::format::UPGRADE_JOURNAL_FORMAT_VERSION;

/// The journal's file name under the data directory
pub const JOURNAL_FILE: &str = "upgrade.journal";

/// What kind of restart the node is in the middle of
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum RestartKind {
    /// Restarting on a newly activated binary
    Upgrade,
    /// Restarting on the binary that ran before
    Rollback,
}

/// A restart this node armed and has not yet accounted for.
///
/// Written before the process exits to restart, read by the process that
/// comes up, which judges the outcome against the baseline carried here
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PendingRestart {
    pub kind: RestartKind,
    pub upgrade_id: u64,
    pub node_id: String,
    pub from_version: String,
    pub to_version: String,
    pub channel: String,
    /// When the sequence this restart belongs to started
    pub started_at_secs: u64,
    pub requested_at_secs: u64,
    /// The cluster's health before the sequence, which the node that
    /// comes up is judged against
    pub baseline: HealthBaseline,
    /// Nodes already on the new binary when this one restarted
    pub nodes_upgraded_before: u32,
    pub nodes_total: u32,
    /// Formats the gate expects to move, as (kind, from, to) with the
    /// versions as their integer form
    pub format_migrations: Vec<(String, u32, u32)>,
    /// Whether every planned migration can be undone
    pub reversible: bool,
    /// The physical backup taken before the sequence, when one was
    pub snapshot_path: Option<String>,
    /// The binary path that was activated, so a failed restart knows what
    /// to put back
    pub live_path: String,
    /// Set once this restart was undone, so the binary that comes back
    /// knows it is the previous one on purpose, and why
    pub rolled_back_reason: Option<String>,
    /// Whether the node that comes back pauses upgrades, which a rollback
    /// after a failed health check does and a rollback an operator asked
    /// for does not
    pub pause_on_return: bool,
    /// Who asked for it, empty for the controller
    pub actor: String,
    /// Whether another node's coordinator drove this restart and watches
    /// the outcome, in which case the node that comes up records it and
    /// does not judge itself
    pub coordinated: bool,
}

/// The last upgrade that finished on this node, which a rollback is
/// measured against
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CompletedUpgrade {
    pub upgrade_id: u64,
    pub from_version: String,
    pub to_version: String,
    pub finished_at_secs: u64,
    /// Formats the eager sweep moved, as (kind, version they were at)
    pub migrated_formats: Vec<(String, u32)>,
    /// Catalog tables the schema migration moved, as (table, version they
    /// were at)
    pub migrated_tables: Vec<(String, u32)>,
    pub snapshot_path: Option<String>,
}

/// The rows of one catalog table held while they are being replaced, so a
/// crash mid-replacement is undone on the next start rather than leaving
/// two versions of the rows in the heap
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CatalogPreimage {
    pub catalog_table: String,
    /// The version the rows were at
    pub version: u32,
    /// Each row as lower-case hex
    pub rows_hex: Vec<String>,
}

/// Everything the journal holds
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct UpgradeJournal {
    pub board: UpgradeBoardSnapshot,
    pub restart: Option<PendingRestart>,
    pub last_completed: Option<CompletedUpgrade>,
    /// The schema version each catalog table's rows are stored at, seeded
    /// with every registered table at the version this binary writes the
    /// first time the journal is created
    pub catalog_versions: Vec<(String, u32)>,
    pub catalog_preimage: Option<CatalogPreimage>,
}

impl UpgradeJournal {
    /// The stored version of one catalog table, or None when the journal
    /// has never recorded it
    pub fn catalog_version(&self, catalog_table: &str) -> Option<u32> {
        self.catalog_versions
            .iter()
            .find(|(name, _)| name.eq_ignore_ascii_case(catalog_table))
            .map(|(_, version)| *version)
    }

    /// Records the stored version of one catalog table
    pub fn set_catalog_version(&mut self, catalog_table: &str, version: u32) {
        match self
            .catalog_versions
            .iter_mut()
            .find(|(name, _)| name.eq_ignore_ascii_case(catalog_table))
        {
            Some(slot) => slot.1 = version,
            None => self
                .catalog_versions
                .push((catalog_table.to_string(), version)),
        }
    }
}

/// Where the journal lives
pub fn journal_path(data_dir: &Path) -> PathBuf {
    data_dir.join(JOURNAL_FILE)
}

/// Reads the journal, or an empty one when the node has never written it.
///
/// The file goes through the format registry like every other envelope, so
/// a journal written at an older version is migrated on the way in
pub fn load(registry: &FormatRegistry, data_dir: &Path) -> Result<UpgradeJournal> {
    let path = journal_path(data_dir);
    let bytes = match std::fs::read(&path) {
        Ok(bytes) => bytes,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(UpgradeJournal::default()),
        Err(e) => return Err(ZyronError::Io(e)),
    };
    let opened = migration::open_as(registry, &bytes, FormatKind::UpgradeJournal)?;
    serde_json::from_slice(&opened.body).map_err(|e| {
        ZyronError::Internal(format!(
            "the upgrade journal at {} is not readable, {e}",
            path.display()
        ))
    })
}

/// Writes the journal whole, through a temp file and a rename
pub fn save(data_dir: &Path, journal: &UpgradeJournal) -> Result<()> {
    let path = journal_path(data_dir);
    let body = serde_json::to_vec(journal)
        .map_err(|e| ZyronError::Internal(format!("upgrade journal encode, {e}")))?;
    let bytes = envelope::encode(
        FormatKind::UpgradeJournal,
        UPGRADE_JOURNAL_FORMAT_VERSION,
        &body,
    );
    std::fs::create_dir_all(data_dir).map_err(ZyronError::Io)?;
    let tmp = path.with_extension("journal.tmp");
    if let Err(e) = std::fs::write(&tmp, &bytes) {
        let _ = std::fs::remove_file(&tmp);
        return Err(ZyronError::Io(e));
    }
    if let Err(e) = std::fs::rename(&tmp, &path) {
        let _ = std::fs::remove_file(&tmp);
        return Err(ZyronError::Io(e));
    }
    Ok(())
}

/// The journal as the running node holds it, saved on every change
pub struct Journal {
    data_dir: PathBuf,
    state: parking_lot::Mutex<UpgradeJournal>,
}

impl Journal {
    /// Opens the journal, seeding the catalog versions the registry knows
    /// and this journal has never recorded at the version this binary
    /// writes
    pub fn open(
        registry: &FormatRegistry,
        catalog: &CatalogSchemaRegistry,
        data_dir: &Path,
    ) -> Result<Journal> {
        let mut state = load(registry, data_dir)?;
        let mut seeded = false;
        for table in catalog.tables() {
            let name = table.registration.catalog_table;
            if state.catalog_version(name).is_none() {
                state.set_catalog_version(name, table.registration.current_schema_version.as_u32());
                seeded = true;
            }
        }
        if seeded {
            save(data_dir, &state)?;
        }
        Ok(Journal {
            data_dir: data_dir.to_path_buf(),
            state: parking_lot::Mutex::new(state),
        })
    }

    pub fn data_dir(&self) -> &Path {
        &self.data_dir
    }

    /// A copy of what the journal holds
    pub fn read(&self) -> UpgradeJournal {
        self.state.lock().clone()
    }

    /// Changes the journal and writes it. The change is kept in memory only
    /// when the write succeeds, so memory and disk never disagree
    pub fn update(&self, change: impl FnOnce(&mut UpgradeJournal)) -> Result<()> {
        let mut state = self.state.lock();
        let mut next = state.clone();
        change(&mut next);
        save(&self.data_dir, &next)?;
        *state = next;
        Ok(())
    }

    /// Copies the board into the journal and writes it
    pub fn sync_board(&self, board: &UpgradeBoard) -> Result<()> {
        let snapshot = board.snapshot();
        self.update(|journal| journal.board = snapshot)
    }

    /// Puts the journal's board rows, history, and rewrite queue on the
    /// board. Settings come from the config afterwards
    pub fn restore_board(&self, board: &UpgradeBoard) {
        board.restore(self.state.lock().board.clone());
    }
}

/// Encodes catalog rows for the pre-image
pub fn rows_to_hex(rows: &[Vec<u8>]) -> Vec<String> {
    rows.iter()
        .map(|row| super::feed::encode_hex(row))
        .collect()
}

/// Decodes catalog rows from the pre-image, refusing one that is not hex
pub fn rows_from_hex(rows: &[String]) -> Result<Vec<Vec<u8>>> {
    rows.iter()
        .map(|row| {
            super::feed::decode_hex(row).ok_or_else(|| {
                ZyronError::Internal("a catalog pre-image row is not hex".to_string())
            })
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use zyron_common::format::{UpgradeOutcome, UpgradePhase};

    fn registries() -> &'static zyron_common::format::FormatSubstrate {
        zyron_common::format::substrate().expect("the server registers every format")
    }

    #[test]
    fn test_a_journal_round_trips_through_its_envelope() {
        let substrate = registries();
        let dir = tempfile::tempdir().expect("tempdir");
        let journal = Journal::open(&substrate.formats, &substrate.catalog_schemas, dir.path())
            .expect("opens");
        assert!(
            journal
                .read()
                .catalog_version("zyron_sys.core.tables")
                .is_some(),
            "every registered table is seeded"
        );

        let board = UpgradeBoard::new();
        board.set_node_state(zyron_common::format::NodeUpgradeState {
            node_id: "node-1".into(),
            from_version: "0.11.0".into(),
            to_version: "0.12.0".into(),
            phase: UpgradePhase::Rolling,
            started_at_secs: 10,
            updated_at_secs: 11,
            is_leader: true,
            message: "restarting".into(),
        });
        board.push_history(zyron_common::format::UpgradeHistoryEntry {
            upgrade_id: 1,
            from_version: "0.10.0".into(),
            to_version: "0.11.0".into(),
            channel: "stable".into(),
            started_at_secs: 1,
            finished_at_secs: 2,
            outcome: UpgradeOutcome::Completed,
            nodes_upgraded: 1,
            format_migrations_run: 0,
            catalog_migrations_run: 0,
            rewrites_applied: 0,
            reversible: true,
            detail: "done".into(),
        });
        journal.sync_board(&board).expect("saves");
        journal
            .update(|j| {
                j.restart = Some(PendingRestart {
                    kind: RestartKind::Upgrade,
                    upgrade_id: 2,
                    node_id: "node-1".into(),
                    from_version: "0.11.0".into(),
                    to_version: "0.12.0".into(),
                    channel: "stable".into(),
                    started_at_secs: 10,
                    requested_at_secs: 12,
                    baseline: HealthBaseline {
                        p50_latency_us: 100,
                        p99_latency_us: 900,
                        throughput_per_sec: 50.0,
                        error_rate: 0.0,
                        active_connections: 3,
                        queries_in_window: 3_000,
                    },
                    nodes_upgraded_before: 0,
                    nodes_total: 1,
                    format_migrations: vec![("wal_segment".into(), 65536, 65537)],
                    reversible: true,
                    snapshot_path: None,
                    live_path: "/srv/zyron-server".into(),
                    rolled_back_reason: None,
                    pause_on_return: false,
                    actor: String::new(),
                    coordinated: false,
                });
            })
            .expect("saves");

        let bytes = std::fs::read(journal_path(dir.path())).expect("written");
        let (kind, _) = envelope::peek(&bytes).expect("an envelope");
        assert_eq!(kind, FormatKind::UpgradeJournal);

        let reopened = Journal::open(&substrate.formats, &substrate.catalog_schemas, dir.path())
            .expect("reopens");
        let state = reopened.read();
        assert_eq!(state.board.nodes.len(), 1);
        assert_eq!(state.board.history.len(), 1);
        assert_eq!(
            state.restart.as_ref().map(|r| r.to_version.as_str()),
            Some("0.12.0")
        );
        let restored = UpgradeBoard::new();
        reopened.restore_board(&restored);
        assert_eq!(restored.node_states()[0].phase, UpgradePhase::Rolling);
        assert_eq!(restored.history(10)[0].upgrade_id, 1);
    }

    #[test]
    fn test_a_failed_write_leaves_memory_untouched() {
        let substrate = registries();
        let dir = tempfile::tempdir().expect("tempdir");
        let journal = Journal::open(&substrate.formats, &substrate.catalog_schemas, dir.path())
            .expect("opens");
        // A directory where the journal path is cannot be renamed over
        std::fs::remove_file(journal_path(dir.path())).expect("removes");
        std::fs::create_dir_all(journal_path(dir.path())).expect("blocks the path");
        let err = journal
            .update(|j| {
                j.last_completed = Some(CompletedUpgrade {
                    upgrade_id: 9,
                    from_version: "a".into(),
                    to_version: "b".into(),
                    finished_at_secs: 1,
                    migrated_formats: Vec::new(),
                    migrated_tables: Vec::new(),
                    snapshot_path: None,
                })
            })
            .expect_err("the rename fails");
        assert!(!err.to_string().is_empty());
        assert!(journal.read().last_completed.is_none());
    }

    #[test]
    fn test_pre_image_rows_round_trip_as_hex() {
        let rows = vec![vec![0u8, 1, 255], Vec::new(), vec![16]];
        let hex = rows_to_hex(&rows);
        assert_eq!(hex, vec!["0001ff", "", "10"]);
        assert_eq!(rows_from_hex(&hex).expect("decodes"), rows);
        assert!(rows_from_hex(&["zz".to_string()]).is_err());
    }
}
