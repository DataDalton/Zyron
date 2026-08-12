//! Columnar patch log.
//!
//! UPDATE or DELETE of a columnar-resident row never rewrites a .zyr and
//! never round-trips through the heap. It appends an epoch-tagged entry to a
//! per-table append-only `<table_id>.zyrpatch` file: a value patch (a new
//! version of one column for one sys_rowid) or a supersede (a delete marker
//! for one sys_rowid). The scan resolves these in the encoded domain using
//! the same Snapshot::is_visible(xmin, xmax) oracle as the heap.
//!
//! The file is the durable source of truth (fsynced on append). Each record
//! carries the WAL LSN of the matching ColumnarPatch / ColumnarSupersede
//! record and a CRC. Recovery replays a WAL patch record only when its LSN
//! exceeds the file's persisted high-water, so a clean restart never
//! re-appends an already-durable patch (the log and overlay stay bounded).
//! A torn tail from a crash fails CRC and is truncated.

use std::collections::HashMap;
use std::fs::{File, OpenOptions};
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, OnceLock, RwLock, RwLockReadGuard, RwLockWriteGuard};

use zyron_common::{Result, ZyronError};

use super::constants::{
    PATCH_KIND_BRANCH_CLEAR, PATCH_KIND_BRANCH_COPY, PATCH_KIND_REVOKE_SUPERSEDE,
    PATCH_KIND_REVOKE_VALUE, PATCH_KIND_SUPERSEDE, PATCH_KIND_VALUE, ZYRPATCH_MAGIC,
};

/// Fixed record header size: kind(1) branch_id(8) file_id(8) sys_rowid(8)
/// epoch(8) lsn(8) col_id(4) val_len(4). Value bytes then a 4-byte CRC
/// follow. branch_id 0 is the main line.
const REC_HEADER: usize = 49;

/// One value patch: a new version of `column_id` for `sys_rowid`, created by
/// transaction `patch_xid`.
#[derive(Debug, Clone)]
pub struct ValuePatch {
    pub patch_xid: u64,
    pub column_id: u32,
    pub value: Vec<u8>,
}

/// Resolved overlay for one columnar logical row.
#[derive(Debug, Default, Clone)]
pub struct RowOverlay {
    /// Supersede transaction ids (DELETE / re-version markers). The row is
    /// invisible to a snapshot that sees any of these as committed.
    pub supersedes: Vec<u64>,
    /// Per-column value patch chain, append order preserved.
    pub patches: HashMap<u32, Vec<ValuePatch>>,
}

/// file_id -> (sys_rowid -> overlay). Nesting by file_id makes the
/// per-segment scan questions (does this file have any overlay, snapshot
/// this file's overlay) O(rows-in-file) instead of O(rows-in-table).
type OverlayMap = HashMap<u64, HashMap<u64, Arc<RowOverlay>>>;

/// Branch scoped overlay state. Main line rows live in `main` so main line
/// reads pay no branch cost. A branch write copies the touched row from
/// main on first touch and shadows it thereafter, mirroring heap page COW
/// at row granularity, so later main line writes to a branch touched row
/// stay invisible to the branch exactly like a COW copied heap page.
struct Overlays {
    main: OverlayMap,
    branches: HashMap<u64, OverlayMap>,
}

/// Per-table append-only patch store.
pub struct PatchStore {
    file: RwLock<File>,
    overlay: RwLock<Overlays>,
    /// Max WAL LSN of any record durably in the file. Recovery skips a WAL
    /// patch whose LSN is at or below this, so replay never duplicates an
    /// already-persisted record.
    persisted_lsn: AtomicU64,
}

fn record_checksum(bytes: &[u8]) -> u32 {
    zyron_common::hash32(bytes)
}

/// Recovers the guarded data on a poisoned lock instead of erroring. The
/// overlay and file structures are append-only and a panic cannot leave them
/// half-mutated, so reading through poison returns correct data. Failing the
/// read instead (returning empty / None) would silently hide deletes and
/// patches, which is a wrong-answer bug, so this fails functional, not closed.
fn read_through<'a, T>(lock: &'a RwLock<T>) -> RwLockReadGuard<'a, T> {
    lock.read().unwrap_or_else(|e| e.into_inner())
}
fn write_through<'a, T>(lock: &'a RwLock<T>) -> RwLockWriteGuard<'a, T> {
    lock.write().unwrap_or_else(|e| e.into_inner())
}

impl PatchStore {
    /// Opens or creates the patch file at `path`, loading existing records
    /// into the in-memory overlay, recovering the persisted LSN high-water,
    /// and truncating a torn tail.
    pub fn open(path: &Path) -> Result<Self> {
        let mut file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .open(path)
            .map_err(|e| {
                ZyronError::IoError(format!(
                    "failed to open patch file {}: {}",
                    path.display(),
                    e
                ))
            })?;

        let len = file
            .metadata()
            .map_err(|e| ZyronError::IoError(format!("patch metadata: {}", e)))?
            .len();

        let mut overlay = Overlays {
            main: HashMap::new(),
            branches: HashMap::new(),
        };
        let mut hwm: u64 = 0;

        if len == 0 {
            file.write_all(&ZYRPATCH_MAGIC)
                .map_err(|e| ZyronError::IoError(format!("patch header write: {}", e)))?;
            file.sync_all()
                .map_err(|e| ZyronError::IoError(format!("patch header fsync: {}", e)))?;
        } else {
            let mut buf = Vec::new();
            file.seek(SeekFrom::Start(0))
                .map_err(|e| ZyronError::IoError(format!("patch seek: {}", e)))?;
            file.read_to_end(&mut buf)
                .map_err(|e| ZyronError::IoError(format!("patch read: {}", e)))?;
            if buf.len() < 8 || buf[..8] != ZYRPATCH_MAGIC {
                return Err(ZyronError::IoError(format!(
                    "patch file {} has bad magic",
                    path.display()
                )));
            }
            let mut pos = 8usize;
            let mut valid_end = 8usize;
            loop {
                if pos == buf.len() {
                    break; // clean end of log
                }
                if pos + REC_HEADER > buf.len() {
                    // A partial header at the very end is an unwritten torn
                    // tail. Truncate it. A partial header anywhere a full
                    // record could still follow cannot occur, the file only
                    // grows by whole fsynced records.
                    break;
                }
                let kind = buf[pos];
                let branch = u64::from_le_bytes(buf[pos + 1..pos + 9].try_into().unwrap());
                let file_id = u64::from_le_bytes(buf[pos + 9..pos + 17].try_into().unwrap());
                let sys_rowid = u64::from_le_bytes(buf[pos + 17..pos + 25].try_into().unwrap());
                let epoch = u64::from_le_bytes(buf[pos + 25..pos + 33].try_into().unwrap());
                let lsn = u64::from_le_bytes(buf[pos + 33..pos + 41].try_into().unwrap());
                let col_id = u32::from_le_bytes(buf[pos + 41..pos + 45].try_into().unwrap());
                let val_len =
                    u32::from_le_bytes(buf[pos + 45..pos + 49].try_into().unwrap()) as usize;
                let rec_end = pos + REC_HEADER + val_len + 4;
                if rec_end > buf.len() {
                    break; // torn tail, record does not fit the file
                }
                let stored_crc = u32::from_le_bytes(
                    buf[pos + REC_HEADER + val_len..rec_end].try_into().unwrap(),
                );
                if record_checksum(&buf[pos..pos + REC_HEADER + val_len]) != stored_crc {
                    // A CRC mismatch is a torn tail only when nothing valid
                    // follows. If a later region parses as a well formed
                    // record, this is interior corruption: truncating here
                    // would silently drop committed patches that follow. Fail
                    // recovery so the operator can restore from backup rather
                    // than lose data.
                    if Self::well_formed_record_follows(&buf, rec_end) {
                        return Err(ZyronError::IoError(format!(
                            "patch file {} has interior corruption at offset {}, a later record is intact so this is not a torn tail",
                            path.display(),
                            pos
                        )));
                    }
                    break; // torn / corrupt tail
                }
                let value = buf[pos + REC_HEADER..pos + REC_HEADER + val_len].to_vec();
                Self::apply_overlay(
                    &mut overlay,
                    branch,
                    kind,
                    file_id,
                    sys_rowid,
                    epoch,
                    col_id,
                    value,
                );
                if lsn > hwm {
                    hwm = lsn;
                }
                pos = rec_end;
                valid_end = rec_end;
            }
            if (valid_end as u64) < len {
                // Truncate the torn tail so the next append is well formed.
                file.set_len(valid_end as u64)
                    .map_err(|e| ZyronError::IoError(format!("patch truncate: {}", e)))?;
                file.sync_all()
                    .map_err(|e| ZyronError::IoError(format!("patch truncate fsync: {}", e)))?;
            }
            file.seek(SeekFrom::End(0))
                .map_err(|e| ZyronError::IoError(format!("patch seek end: {}", e)))?;
        }

        Ok(Self {
            file: RwLock::new(file),
            overlay: RwLock::new(overlay),
            persisted_lsn: AtomicU64::new(hwm),
        })
    }

    /// Scans from `from` for any record whose CRC validates. A match means a
    /// well formed record region follows a corrupt one, so the corruption is
    /// interior, not a torn tail. The scan starts at the record boundary the
    /// corrupt record claims and then slides byte by byte, because a flipped
    /// length field could mis-size that boundary. A CRC over the record bytes
    /// makes a coincidental match astronomically unlikely.
    fn well_formed_record_follows(buf: &[u8], from: usize) -> bool {
        let mut scan = from;
        while scan + REC_HEADER + 4 <= buf.len() {
            let val_len =
                u32::from_le_bytes(buf[scan + 45..scan + 49].try_into().unwrap()) as usize;
            let rec_end = scan + REC_HEADER + val_len + 4;
            if rec_end <= buf.len() {
                let stored_crc = u32::from_le_bytes(
                    buf[scan + REC_HEADER + val_len..rec_end]
                        .try_into()
                        .unwrap(),
                );
                if record_checksum(&buf[scan..scan + REC_HEADER + val_len]) == stored_crc {
                    return true;
                }
            }
            scan += 1;
        }
        false
    }

    #[allow(clippy::too_many_arguments)]
    fn apply_overlay(
        overlays: &mut Overlays,
        branch: u64,
        kind: u8,
        file_id: u64,
        sys_rowid: u64,
        epoch: u64,
        col_id: u32,
        value: Vec<u8>,
    ) {
        if kind == PATCH_KIND_BRANCH_CLEAR {
            overlays.branches.remove(&branch);
            return;
        }
        // Explicit copy record written before a branch's first write to a
        // row. Idempotent, the copy is skipped when the branch already
        // shadows the row, so concurrent first writers cannot duplicate it
        if kind == PATCH_KIND_BRANCH_COPY {
            if branch == 0 {
                return;
            }
            let bmap = overlays.branches.entry(branch).or_default();
            let present = bmap
                .get(&file_id)
                .is_some_and(|rows| rows.contains_key(&sys_rowid));
            if !present {
                if let Some(mainRow) = overlays
                    .main
                    .get(&file_id)
                    .and_then(|rows| rows.get(&sys_rowid))
                {
                    bmap.entry(file_id)
                        .or_default()
                        .insert(sys_rowid, Arc::clone(mainRow));
                }
            }
            return;
        }
        let overlay = if branch == 0 {
            &mut overlays.main
        } else {
            overlays.branches.entry(branch).or_default()
        };
        // Revokes remove the newest matching entry and never create rows,
        // so a revoke replayed against an already trimmed log is a no-op
        if kind == PATCH_KIND_REVOKE_SUPERSEDE || kind == PATCH_KIND_REVOKE_VALUE {
            let Some(rows) = overlay.get_mut(&file_id) else {
                return;
            };
            let Some(arc_row) = rows.get_mut(&sys_rowid) else {
                return;
            };
            let row = Arc::make_mut(arc_row);
            if kind == PATCH_KIND_REVOKE_SUPERSEDE {
                if let Some(p) = row.supersedes.iter().rposition(|&x| x == epoch) {
                    row.supersedes.remove(p);
                }
            } else if let Some(chain) = row.patches.get_mut(&col_id) {
                if let Some(p) = chain.iter().rposition(|v| v.patch_xid == epoch) {
                    chain.remove(p);
                }
                if chain.is_empty() {
                    row.patches.remove(&col_id);
                }
            }
            if row.supersedes.is_empty() && row.patches.is_empty() {
                rows.remove(&sys_rowid);
                if rows.is_empty() {
                    overlay.remove(&file_id);
                }
            }
            return;
        }
        let row = Arc::make_mut(
            overlay
                .entry(file_id)
                .or_default()
                .entry(sys_rowid)
                .or_insert_with(|| Arc::new(RowOverlay::default())),
        );
        if kind == PATCH_KIND_SUPERSEDE {
            row.supersedes.push(epoch);
        } else if kind == PATCH_KIND_VALUE {
            row.patches.entry(col_id).or_default().push(ValuePatch {
                patch_xid: epoch,
                column_id: col_id,
                value,
            });
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn append_record(
        &self,
        branch: u64,
        kind: u8,
        file_id: u64,
        sys_rowid: u64,
        epoch: u64,
        lsn: u64,
        col_id: u32,
        value: &[u8],
    ) -> Result<()> {
        let mut rec = Vec::with_capacity(REC_HEADER + value.len() + 4);
        rec.push(kind);
        rec.extend_from_slice(&branch.to_le_bytes());
        rec.extend_from_slice(&file_id.to_le_bytes());
        rec.extend_from_slice(&sys_rowid.to_le_bytes());
        rec.extend_from_slice(&epoch.to_le_bytes());
        rec.extend_from_slice(&lsn.to_le_bytes());
        rec.extend_from_slice(&col_id.to_le_bytes());
        rec.extend_from_slice(&(value.len() as u32).to_le_bytes());
        rec.extend_from_slice(value);
        let crc = record_checksum(&rec);
        rec.extend_from_slice(&crc.to_le_bytes());

        {
            let mut f = write_through(&self.file);
            f.write_all(&rec)
                .map_err(|e| ZyronError::IoError(format!("patch append: {}", e)))?;
            f.sync_all()
                .map_err(|e| ZyronError::IoError(format!("patch fsync: {}", e)))?;
        }

        // Update the in-memory overlay BEFORE advancing the persisted LSN
        // high-water. A reader that observes the new high-water must also see
        // the patch in the live overlay. Advancing the high-water first would
        // expose a window where the patch reads as persisted but absent from
        // the overlay, so a concurrent scan would miss a durable patch.
        let mut ov = write_through(&self.overlay);
        Self::apply_overlay(
            &mut ov,
            branch,
            kind,
            file_id,
            sys_rowid,
            epoch,
            col_id,
            value.to_vec(),
        );
        self.persisted_lsn.fetch_max(lsn, Ordering::AcqRel);
        drop(ov);
        Ok(())
    }

    /// Appends a value patch for one column of one columnar row on the given
    /// branch, 0 for the main line. `lsn` is the WAL LSN of the matching
    /// ColumnarPatch record.
    #[allow(clippy::too_many_arguments)]
    pub fn append_value_patch(
        &self,
        branch: u64,
        file_id: u64,
        sys_rowid: u64,
        column_id: u32,
        patch_xid: u64,
        lsn: u64,
        value: &[u8],
    ) -> Result<()> {
        self.copy_branch_row_on_first_touch(branch, file_id, sys_rowid, lsn)?;
        self.append_record(
            branch,
            PATCH_KIND_VALUE,
            file_id,
            sys_rowid,
            patch_xid,
            lsn,
            column_id,
            value,
        )
    }

    /// Appends a supersede (delete) marker for one columnar row on the given
    /// branch, 0 for the main line. `lsn` is the WAL LSN of the matching
    /// ColumnarSupersede record.
    pub fn append_supersede(
        &self,
        branch: u64,
        file_id: u64,
        sys_rowid: u64,
        supersede_xid: u64,
        lsn: u64,
    ) -> Result<()> {
        self.copy_branch_row_on_first_touch(branch, file_id, sys_rowid, lsn)?;
        self.append_record(
            branch,
            PATCH_KIND_SUPERSEDE,
            file_id,
            sys_rowid,
            supersede_xid,
            lsn,
            0,
            &[],
        )
    }

    /// Writes the explicit branch row copy record before a branch's first
    /// write to a row, so replay reconstructs the same shadowed state with
    /// no implicit copying
    fn copy_branch_row_on_first_touch(
        &self,
        branch: u64,
        file_id: u64,
        sys_rowid: u64,
        lsn: u64,
    ) -> Result<()> {
        if branch == 0 {
            return Ok(());
        }
        let touched = {
            let ov = read_through(&self.overlay);
            ov.branches
                .get(&branch)
                .and_then(|m| m.get(&file_id))
                .is_some_and(|rows| rows.contains_key(&sys_rowid))
        };
        if touched {
            return Ok(());
        }
        self.append_record(
            branch,
            PATCH_KIND_BRANCH_COPY,
            file_id,
            sys_rowid,
            0,
            lsn,
            0,
            &[],
        )
    }

    /// Discards every overlay entry of one branch, the columnar side of
    /// DROP BRANCH, and of MERGE BRANCH after the rows fold into main
    pub fn clear_branch(&self, branch: u64, lsn: u64) -> Result<()> {
        self.append_record(branch, PATCH_KIND_BRANCH_CLEAR, 0, 0, 0, lsn, 0, &[])
    }

    /// Revokes the newest value patch written by `patch_xid` for one column
    /// of one row, the undo of `append_value_patch` for ROLLBACK TO
    /// SAVEPOINT. `lsn` is the WAL LSN of the matching revoke record.
    #[allow(clippy::too_many_arguments)]
    pub fn revoke_value_patch(
        &self,
        branch: u64,
        file_id: u64,
        sys_rowid: u64,
        column_id: u32,
        patch_xid: u64,
        lsn: u64,
    ) -> Result<()> {
        self.append_record(
            branch,
            PATCH_KIND_REVOKE_VALUE,
            file_id,
            sys_rowid,
            patch_xid,
            lsn,
            column_id,
            &[],
        )
    }

    /// Revokes the newest supersede written by `supersede_xid` for one row,
    /// the undo of `append_supersede` for ROLLBACK TO SAVEPOINT. `lsn` is
    /// the WAL LSN of the matching revoke record.
    pub fn revoke_supersede(
        &self,
        branch: u64,
        file_id: u64,
        sys_rowid: u64,
        supersede_xid: u64,
        lsn: u64,
    ) -> Result<()> {
        self.append_record(
            branch,
            PATCH_KIND_REVOKE_SUPERSEDE,
            file_id,
            sys_rowid,
            supersede_xid,
            lsn,
            0,
            &[],
        )
    }

    /// WAL-first supersede: logs the ColumnarSupersede record, then appends
    /// the marker to the durable patch log at the returned LSN. Every
    /// supersede writer goes through here so the WAL payload layout and the
    /// log-before-append discipline live in one place. Returns the WAL LSN
    pub fn supersede_logged(
        &self,
        wal: &zyron_wal::WalWriter,
        table_id: u64,
        branch: u64,
        file_id: u64,
        sys_rowid: u64,
        xid: u64,
    ) -> Result<u64> {
        let pl = super::wal_payload::ColumnarSupersedePayload {
            table_id,
            branch,
            file_id,
            sys_rowid,
            xid,
        }
        .encode();
        let lsn = wal.log_columnar_supersede(&pl)?;
        self.append_supersede(branch, file_id, sys_rowid, xid, lsn.0)?;
        Ok(lsn.0)
    }

    /// WAL-first value patch: logs the ColumnarPatch record, then appends the
    /// patch to the durable log at the returned LSN. Returns the WAL LSN
    #[allow(clippy::too_many_arguments)]
    pub fn patch_logged(
        &self,
        wal: &zyron_wal::WalWriter,
        table_id: u64,
        branch: u64,
        file_id: u64,
        sys_rowid: u64,
        column_id: u32,
        xid: u64,
        value: &[u8],
    ) -> Result<u64> {
        let pl = super::wal_payload::ColumnarValuePatchPayload {
            table_id,
            branch,
            file_id,
            sys_rowid,
            column_id,
            xid,
        }
        .encode_with_value(value);
        let lsn = wal.log_columnar_patch(&pl)?;
        self.append_value_patch(branch, file_id, sys_rowid, column_id, xid, lsn.0, value)?;
        Ok(lsn.0)
    }

    /// WAL-first supersede revoke, the undo of supersede_logged for ROLLBACK
    /// TO SAVEPOINT. Returns the WAL LSN
    pub fn revoke_supersede_logged(
        &self,
        wal: &zyron_wal::WalWriter,
        table_id: u64,
        branch: u64,
        file_id: u64,
        sys_rowid: u64,
        xid: u64,
    ) -> Result<u64> {
        let pl = super::wal_payload::ColumnarSupersedePayload {
            table_id,
            branch,
            file_id,
            sys_rowid,
            xid,
        }
        .encode();
        let lsn = wal.log_columnar_supersede_revoke(&pl)?;
        self.revoke_supersede(branch, file_id, sys_rowid, xid, lsn.0)?;
        Ok(lsn.0)
    }

    /// WAL-first value patch revoke, the undo of patch_logged for ROLLBACK TO
    /// SAVEPOINT. Returns the WAL LSN
    #[allow(clippy::too_many_arguments)]
    pub fn revoke_patch_logged(
        &self,
        wal: &zyron_wal::WalWriter,
        table_id: u64,
        branch: u64,
        file_id: u64,
        sys_rowid: u64,
        column_id: u32,
        xid: u64,
    ) -> Result<u64> {
        let pl = super::wal_payload::ColumnarPatchRevokePayload {
            table_id,
            branch,
            file_id,
            sys_rowid,
            column_id,
            xid,
        }
        .encode();
        let lsn = wal.log_columnar_patch_revoke(&pl)?;
        self.revoke_value_patch(branch, file_id, sys_rowid, column_id, xid, lsn.0)?;
        Ok(lsn.0)
    }

    /// WAL-first branch clear, the columnar side of DROP BRANCH and of MERGE
    /// BRANCH after fold-in. Returns the WAL LSN
    pub fn clear_branch_logged(
        &self,
        wal: &zyron_wal::WalWriter,
        table_id: u64,
        branch: u64,
    ) -> Result<u64> {
        let pl = super::wal_payload::ColumnarBranchClearPayload { table_id, branch }.encode();
        let lsn = wal.log_columnar_branch_clear(&pl)?;
        self.clear_branch(branch, lsn.0)?;
        Ok(lsn.0)
    }

    /// The max WAL LSN durably recorded in the patch file. Recovery replays a
    /// WAL patch only when its LSN is strictly greater than this.
    pub fn max_persisted_lsn(&self) -> u64 {
        self.persisted_lsn.load(Ordering::Acquire)
    }

    /// Returns the main line overlay for one row, if any patches exist.
    pub fn row_overlay(&self, file_id: u64, sys_rowid: u64) -> Option<Arc<RowOverlay>> {
        let ov = read_through(&self.overlay);
        ov.main.get(&file_id)?.get(&sys_rowid).cloned()
    }

    /// Returns the overlay for one row as seen from a branch. A branch
    /// touched row resolves to its branch copy, an untouched row falls
    /// through to the live main line, mirroring heap COW page routing.
    pub fn row_overlay_on(
        &self,
        branch: u64,
        file_id: u64,
        sys_rowid: u64,
    ) -> Option<Arc<RowOverlay>> {
        let ov = read_through(&self.overlay);
        if branch != 0 {
            if let Some(row) = ov
                .branches
                .get(&branch)
                .and_then(|m| m.get(&file_id))
                .and_then(|rows| rows.get(&sys_rowid))
            {
                return Some(Arc::clone(row));
            }
        }
        ov.main.get(&file_id)?.get(&sys_rowid).cloned()
    }

    /// True when the store holds no entries (fast path for clean tables).
    pub fn is_empty(&self) -> bool {
        let ov = read_through(&self.overlay);
        ov.main.values().all(|m| m.is_empty())
            && ov
                .branches
                .values()
                .all(|b| b.values().all(|m| m.is_empty()))
    }

    /// True when any main line overlay entry exists for `file_id`. O(1) on
    /// the nested map. Lets the scan and the metadata-aggregate path decide
    /// per segment whether the header fast path is valid, instead of
    /// disabling it table-wide.
    pub fn file_has_overlay(&self, file_id: u64) -> bool {
        read_through(&self.overlay)
            .main
            .get(&file_id)
            .map(|m| !m.is_empty())
            .unwrap_or(false)
    }

    /// True when `file_id` has overlay entries visible from a branch, on
    /// the branch itself or on the main line it reads through to.
    pub fn file_has_overlay_on(&self, branch: u64, file_id: u64) -> bool {
        let ov = read_through(&self.overlay);
        let on_branch = branch != 0
            && ov
                .branches
                .get(&branch)
                .and_then(|m| m.get(&file_id))
                .is_some_and(|rows| !rows.is_empty());
        on_branch
            || ov
                .main
                .get(&file_id)
                .map(|m| !m.is_empty())
                .unwrap_or(false)
    }

    /// Snapshots the main line overlay for one file into a local map under a
    /// single read-lock acquisition. O(rows-in-file). The scan then resolves
    /// per row from the local map with no per-row locking.
    pub fn file_overlay(&self, file_id: u64) -> std::collections::HashMap<u64, Arc<RowOverlay>> {
        match read_through(&self.overlay).main.get(&file_id) {
            Some(m) => m.iter().map(|(r, v)| (*r, Arc::clone(v))).collect(),
            None => std::collections::HashMap::new(),
        }
    }

    /// Snapshots the overlay for one file as seen from a branch: main line
    /// rows overlaid by the branch's own copies of touched rows.
    pub fn file_overlay_on(
        &self,
        branch: u64,
        file_id: u64,
    ) -> std::collections::HashMap<u64, Arc<RowOverlay>> {
        let ov = read_through(&self.overlay);
        let mut out: std::collections::HashMap<u64, Arc<RowOverlay>> = match ov.main.get(&file_id) {
            Some(m) => m.iter().map(|(r, v)| (*r, Arc::clone(v))).collect(),
            None => std::collections::HashMap::new(),
        };
        if branch != 0 {
            if let Some(rows) = ov.branches.get(&branch).and_then(|m| m.get(&file_id)) {
                for (r, v) in rows {
                    out.insert(*r, Arc::clone(v));
                }
            }
        }
        out
    }

    /// Returns every main line (file_id, sys_rowid) that has overlay
    /// entries, so the merge can decide which segments are worth rewriting.
    pub fn rows_with_overlay(&self) -> Vec<(u64, u64)> {
        let ov = read_through(&self.overlay);
        let mut out = Vec::new();
        for (fid, rows) in ov.main.iter() {
            for rid in rows.keys() {
                out.push((*fid, *rid));
            }
        }
        out
    }

    /// Every branch overlay row for one file, so a merge can re-append the
    /// branch state against the file that replaces it.
    pub fn branch_rows_for_file(&self, file_id: u64) -> Vec<(u64, u64, Arc<RowOverlay>)> {
        let ov = read_through(&self.overlay);
        let mut out = Vec::new();
        for (branch, files) in ov.branches.iter() {
            if let Some(rows) = files.get(&file_id) {
                for (rid, row) in rows {
                    out.push((*branch, *rid, Arc::clone(row)));
                }
            }
        }
        out
    }

    /// Every overlay row of one branch across all files, so MERGE BRANCH can
    /// fold the branch state into the main line.
    pub fn branch_overlay_rows(&self, branch: u64) -> Vec<(u64, u64, Arc<RowOverlay>)> {
        let ov = read_through(&self.overlay);
        let mut out = Vec::new();
        if let Some(files) = ov.branches.get(&branch) {
            for (fid, rows) in files {
                for (rid, row) in rows {
                    out.push((*fid, *rid, Arc::clone(row)));
                }
            }
        }
        out
    }

    /// Collapses redundant value-patch history to bound overlay memory
    /// between merges. A patch is collapsible when it is settled
    /// (`patch_xid < horizon`) and `reclaimable(patch_xid)` is true, meaning no
    /// retained version still needs to read the value before it. For each column
    /// chain only the newest collapsible patch is kept (every reader sees at
    /// least that one and resolution picks the newest visible); all other
    /// patches, including any a retained version still needs, are preserved.
    /// Supersedes and rows are never removed here: a settled supersede is the
    /// delete marker the merge needs to physically drop or carry forward the
    /// dead row, and the merge clears it once it completes. In-memory only.
    pub fn trim_below(&self, horizon: u64, reclaimable: impl Fn(u64) -> bool) {
        let mut ov = write_through(&self.overlay);
        for rows in ov.main.values_mut() {
            for arc in rows.values_mut() {
                let collapsible =
                    |p: &ValuePatch| p.patch_xid < horizon && reclaimable(p.patch_xid);
                let needs_trim = arc
                    .patches
                    .values()
                    .any(|chain| chain.iter().filter(|p| collapsible(p)).count() > 1);
                if !needs_trim {
                    continue;
                }
                let row = Arc::make_mut(arc);
                for chain in row.patches.values_mut() {
                    // Index of the newest collapsible patch.
                    let mut newest_below: Option<usize> = None;
                    for (i, p) in chain.iter().enumerate() {
                        if collapsible(p) {
                            match newest_below {
                                Some(j) if chain[j].patch_xid > p.patch_xid => {}
                                _ => newest_below = Some(i),
                            }
                        }
                    }
                    let Some(keep) = newest_below else { continue };
                    let mut idx = 0usize;
                    chain.retain(|p| {
                        let k = idx;
                        idx += 1;
                        !collapsible(p) || k == keep
                    });
                }
            }
        }
    }

    /// Drops every entry for `file_id` after a merge has folded that file's
    /// patches into a fresh segment. O(1) removal of the file's nested map.
    /// Rewrites the patch file without the dropped entries so the log shrinks
    /// with churn. Surviving records are written with the current persisted
    /// LSN high-water so it never regresses across the rewrite, which is what
    /// prevents recovery from resurrecting a merged-away file's WAL patches.
    /// The rewrite is atomic via a temp file plus rename.
    pub fn drop_file(&self, file_id: u64, path: &Path) -> Result<()> {
        let mut ov = write_through(&self.overlay);
        ov.main.remove(&file_id);
        for files in ov.branches.values_mut() {
            files.remove(&file_id);
        }
        let hwm = self.persisted_lsn.load(Ordering::Acquire);

        // Rebuild the on-disk log from the surviving overlay entries, main
        // line first so branch copy-on-touch replays correctly.
        let mut buf = Vec::with_capacity(8);
        buf.extend_from_slice(&ZYRPATCH_MAGIC);
        for (fid, rows) in ov.main.iter() {
            for (rid, row) in rows.iter() {
                for s in &row.supersedes {
                    Self::push_record(
                        &mut buf,
                        0,
                        PATCH_KIND_SUPERSEDE,
                        *fid,
                        *rid,
                        *s,
                        hwm,
                        0,
                        &[],
                    );
                }
                for (col, chain) in &row.patches {
                    for p in chain {
                        Self::push_record(
                            &mut buf,
                            0,
                            PATCH_KIND_VALUE,
                            *fid,
                            *rid,
                            p.patch_xid,
                            hwm,
                            *col,
                            &p.value,
                        );
                    }
                }
            }
        }
        // Branch rows serialize as their exact entries. Replay rebuilds the
        // same shadowed state because copies are explicit records and the
        // rewrite dumps final state, so no copy records are needed here
        for (branch, files) in ov.branches.iter() {
            for (fid, rows) in files.iter() {
                for (rid, row) in rows.iter() {
                    for s in &row.supersedes {
                        Self::push_record(
                            &mut buf,
                            *branch,
                            PATCH_KIND_SUPERSEDE,
                            *fid,
                            *rid,
                            *s,
                            hwm,
                            0,
                            &[],
                        );
                    }
                    for (col, chain) in &row.patches {
                        for p in chain {
                            Self::push_record(
                                &mut buf,
                                *branch,
                                PATCH_KIND_VALUE,
                                *fid,
                                *rid,
                                p.patch_xid,
                                hwm,
                                *col,
                                &p.value,
                            );
                        }
                    }
                }
            }
        }
        let tmp = path.with_extension("zyrpatch.tmp");
        {
            let mut f = OpenOptions::new()
                .write(true)
                .create(true)
                .truncate(true)
                .open(&tmp)
                .map_err(|e| ZyronError::IoError(format!("patch rewrite open: {}", e)))?;
            f.write_all(&buf)
                .map_err(|e| ZyronError::IoError(format!("patch rewrite write: {}", e)))?;
            f.sync_all()
                .map_err(|e| ZyronError::IoError(format!("patch rewrite fsync: {}", e)))?;
        }
        std::fs::rename(&tmp, path)
            .map_err(|e| ZyronError::IoError(format!("patch rewrite rename: {}", e)))?;
        let mut nf = OpenOptions::new()
            .read(true)
            .write(true)
            .open(path)
            .map_err(|e| ZyronError::IoError(format!("patch reopen: {}", e)))?;
        nf.seek(SeekFrom::End(0))
            .map_err(|e| ZyronError::IoError(format!("patch reseek: {}", e)))?;
        let mut fl = write_through(&self.file);
        *fl = nf;
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    #[allow(clippy::too_many_arguments)]
    fn push_record(
        buf: &mut Vec<u8>,
        branch: u64,
        kind: u8,
        file_id: u64,
        sys_rowid: u64,
        epoch: u64,
        lsn: u64,
        col_id: u32,
        value: &[u8],
    ) {
        let start = buf.len();
        buf.push(kind);
        buf.extend_from_slice(&branch.to_le_bytes());
        buf.extend_from_slice(&file_id.to_le_bytes());
        buf.extend_from_slice(&sys_rowid.to_le_bytes());
        buf.extend_from_slice(&epoch.to_le_bytes());
        buf.extend_from_slice(&lsn.to_le_bytes());
        buf.extend_from_slice(&col_id.to_le_bytes());
        buf.extend_from_slice(&(value.len() as u32).to_le_bytes());
        buf.extend_from_slice(value);
        let crc = record_checksum(&buf[start..]);
        buf.extend_from_slice(&crc.to_le_bytes());
    }
}

/// Process-wide manager keyed by the columnar directory. A single database
/// instance per process owns one directory, so the directory is a stable
/// identity shared by the compaction worker, the DML path, and scans without
/// threading a handle through every ExecutionContext constructor.
pub struct ColumnarPatchManager {
    dir: PathBuf,
    stores: RwLock<HashMap<u64, Arc<PatchStore>>>,
}

static GLOBAL: OnceLock<RwLock<HashMap<PathBuf, Arc<ColumnarPatchManager>>>> = OnceLock::new();

impl ColumnarPatchManager {
    fn new(dir: PathBuf) -> Self {
        Self {
            dir,
            stores: RwLock::new(HashMap::new()),
        }
    }

    /// Returns the process-global manager for `columnar_dir`, creating it on
    /// first use.
    pub fn global(columnar_dir: &Path) -> Arc<ColumnarPatchManager> {
        // Canonicalize so the compaction worker (config.columnar_dir), the
        // DML path (segment_path.parent()), recovery (data_dir/columnar) and
        // scans all resolve to the same key. A divergent key would split the
        // overlay and silently lose patches. Falls back to the raw path when
        // the dir does not exist yet (the manager is created before the
        // first fold makes the directory).
        let key =
            std::fs::canonicalize(columnar_dir).unwrap_or_else(|_| columnar_dir.to_path_buf());
        let map = GLOBAL.get_or_init(|| RwLock::new(HashMap::new()));
        {
            if let Ok(r) = map.read()
                && let Some(m) = r.get(&key)
            {
                return Arc::clone(m);
            }
        }
        let mut w = map.write().expect("patch manager registry poisoned");
        Arc::clone(
            w.entry(key.clone())
                .or_insert_with(|| Arc::new(ColumnarPatchManager::new(key))),
        )
    }

    /// Returns the per-table patch store, opening it on first use.
    pub fn store(&self, table_id: u64) -> Result<Arc<PatchStore>> {
        {
            let r = read_through(&self.stores);
            if let Some(s) = r.get(&table_id) {
                return Ok(Arc::clone(s));
            }
        }
        std::fs::create_dir_all(&self.dir)
            .map_err(|e| ZyronError::IoError(format!("create columnar dir: {}", e)))?;
        let path = self.dir.join(format!("{}.zyrpatch", table_id));
        let store = Arc::new(PatchStore::open(&path)?);
        let mut w = write_through(&self.stores);
        Ok(Arc::clone(w.entry(table_id).or_insert(store)))
    }

    /// Evicts a dropped table's store from the process-wide registry and
    /// removes its patch file. The removal can fail while another handle
    /// still holds the file open, the overlays are already dropped at that
    /// point so a leftover file carries no row data. A missing file is fine
    pub fn remove_store(&self, table_id: u64, patch_path: &Path) -> std::io::Result<()> {
        {
            let mut w = write_through(&self.stores);
            w.remove(&table_id);
        }
        match std::fs::remove_file(patch_path) {
            Ok(()) => Ok(()),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(()),
            Err(e) => Err(e),
        }
    }

    /// Resolves the patch store for a table from any of its segment paths.
    /// The parent directory of a .zyr segment is the columnar dir shared by
    /// scans, DML, maintenance and recovery, so every caller converges on the
    /// same manager and store
    pub fn store_for_segment(table_id: u64, segment_path: &Path) -> Result<Arc<PatchStore>> {
        let dir = segment_path.parent().ok_or_else(|| {
            ZyronError::Internal(format!(
                "segment path {} has no parent directory",
                segment_path.display()
            ))
        })?;
        Self::global(dir).store(table_id)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[test]
    fn test_patch_roundtrip_and_reload() {
        let dir = tempdir().expect("tmp");
        let path = dir.path().join("7.zyrpatch");
        {
            let s = PatchStore::open(&path).expect("open");
            s.append_value_patch(0, 1, 42, 3, 100, 10, b"new-value")
                .expect("patch");
            s.append_supersede(0, 1, 43, 105, 11).expect("supersede");
            let o = s.row_overlay(1, 42).expect("overlay");
            assert_eq!(o.patches[&3][0].value, b"new-value");
            assert_eq!(o.patches[&3][0].patch_xid, 100);
            let d = s.row_overlay(1, 43).expect("overlay");
            assert_eq!(d.supersedes, vec![105]);
            assert_eq!(s.max_persisted_lsn(), 11);
        }
        // Reload from disk recovers the overlay and the LSN high-water.
        let s = PatchStore::open(&path).expect("reopen");
        assert_eq!(
            s.row_overlay(1, 42).expect("o").patches[&3][0].value,
            b"new-value"
        );
        assert_eq!(s.row_overlay(1, 43).expect("d").supersedes, vec![105]);
        assert_eq!(s.max_persisted_lsn(), 11);
    }

    #[test]
    fn test_torn_tail_truncated() {
        let dir = tempdir().expect("tmp");
        let path = dir.path().join("9.zyrpatch");
        {
            let s = PatchStore::open(&path).expect("open");
            s.append_value_patch(0, 2, 1, 0, 10, 5, b"abc")
                .expect("patch");
        }
        // Append garbage simulating a torn write.
        {
            let mut f = OpenOptions::new().append(true).open(&path).expect("open");
            f.write_all(&[0xFFu8; 9]).expect("garbage");
        }
        let s = PatchStore::open(&path).expect("reopen");
        assert_eq!(s.row_overlay(2, 1).expect("o").patches[&0][0].value, b"abc");
        // A fresh well-formed append still works after truncation.
        s.append_supersede(0, 2, 1, 20, 6).expect("supersede");
        assert_eq!(s.row_overlay(2, 1).expect("o").supersedes, vec![20]);
    }

    #[test]
    fn test_interior_corruption_rejected() {
        // Two well formed records. Corrupting the first record's value byte
        // must fail recovery, not silently truncate and drop the second
        // committed record.
        let dir = tempdir().expect("tmp");
        let path = dir.path().join("11.zyrpatch");
        {
            let s = PatchStore::open(&path).expect("open");
            s.append_value_patch(0, 1, 1, 0, 10, 5, b"first")
                .expect("p1");
            s.append_value_patch(0, 1, 2, 0, 11, 6, b"second")
                .expect("p2");
        }
        // Flip a byte inside the first record's value, after the magic header.
        // The first value "first" starts at offset 8 + REC_HEADER.
        {
            use std::io::{Read as _, Seek as _, SeekFrom, Write as _};
            let mut f = OpenOptions::new()
                .read(true)
                .write(true)
                .open(&path)
                .expect("open");
            let corrupt_at = (8 + REC_HEADER) as u64;
            f.seek(SeekFrom::Start(corrupt_at)).unwrap();
            let mut b = [0u8; 1];
            f.read_exact(&mut b).unwrap();
            b[0] ^= 0xFF;
            f.seek(SeekFrom::Start(corrupt_at)).unwrap();
            f.write_all(&b).unwrap();
            f.sync_all().unwrap();
        }
        let err = PatchStore::open(&path);
        assert!(
            err.is_err(),
            "interior corruption must fail recovery instead of truncating committed records"
        );
    }

    #[test]
    fn test_overlay_visible_when_persisted_lsn_advances() {
        // After an append returns, the persisted LSN high-water and the live
        // overlay must agree: a reader seeing the new high-water must find the
        // patch in the overlay.
        let dir = tempdir().expect("tmp");
        let path = dir.path().join("12.zyrpatch");
        let s = PatchStore::open(&path).expect("open");
        s.append_value_patch(0, 1, 5, 2, 100, 42, b"x")
            .expect("patch");
        assert_eq!(s.max_persisted_lsn(), 42);
        let o = s
            .row_overlay(1, 5)
            .expect("overlay present once persisted lsn reflects the patch");
        assert_eq!(o.patches[&2][0].value, b"x");
    }

    #[test]
    fn test_trim_below_collapses_old_versions() {
        let dir = tempdir().expect("tmp");
        let path = dir.path().join("3.zyrpatch");
        let s = PatchStore::open(&path).expect("open");
        // Three committed versions of one column, all below the horizon.
        s.append_value_patch(0, 1, 7, 0, 100, 1, b"v1").expect("p1");
        s.append_value_patch(0, 1, 7, 0, 101, 2, b"v2").expect("p2");
        s.append_value_patch(0, 1, 7, 0, 102, 3, b"v3").expect("p3");
        // A separate row deleted below the horizon keeps its supersede: the
        // merge still needs that marker to physically drop the dead row.
        s.append_supersede(0, 1, 8, 50, 4).expect("sup");
        // All patches reclaimable (no retention): collapse to the newest below.
        s.trim_below(200, |_| true);
        let o = s.row_overlay(1, 7).expect("row 7 kept");
        assert_eq!(o.patches[&0].len(), 1, "old versions collapsed");
        assert_eq!(o.patches[&0][0].patch_xid, 102, "newest below horizon kept");
        let d = s.row_overlay(1, 8).expect("superseded row kept for merge");
        assert_eq!(d.supersedes, vec![50], "supersede marker preserved");
    }

    #[test]
    fn test_trim_below_keeps_unreclaimable_patches() {
        let dir = tempdir().expect("tmp");
        let path = dir.path().join("4.zyrpatch");
        let s = PatchStore::open(&path).expect("open");
        s.append_value_patch(0, 1, 7, 0, 100, 1, b"v1").expect("p1");
        s.append_value_patch(0, 1, 7, 0, 101, 2, b"v2").expect("p2");
        s.append_value_patch(0, 1, 7, 0, 102, 3, b"v3").expect("p3");
        // A retention floor keeps patches with xid >= 101 (still within the
        // window); only the older ones may collapse. The newest reclaimable
        // (100) plus all unreclaimable (101, 102) survive.
        s.trim_below(200, |xid| xid < 101);
        let o = s.row_overlay(1, 7).expect("row 7 kept");
        let mut xids: Vec<u64> = o.patches[&0].iter().map(|p| p.patch_xid).collect();
        xids.sort();
        assert_eq!(xids, vec![100, 101, 102], "within-window patches preserved");
    }
}
