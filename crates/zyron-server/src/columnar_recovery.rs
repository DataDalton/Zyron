//! Columnar crash-recovery reconcile.
//!
//! Run once at startup after WAL recovery and before serving. Reconciles the
//! columnar fold and patch records so the invariants hold across a crash:
//!
//! - CompactionBegin with no matching CompactionEnd: the fold did not commit.
//!   The .zyr is an orphan; delete it. The heap was never modified, so it
//!   stays authoritative. No double count, no loss.
//! - CompactionEnd present: the fold committed. Idempotently redo the apply:
//!   register the segment if absent and zero the folded heap slots. Both
//!   steps are idempotent, so any crash during the original apply is closed.
//! - ColumnarPatch / ColumnarSupersede: replay into the patch store so a
//!   crash between the WAL log and the patch-file append loses nothing. The
//!   patch file is fsynced per append, so replay is at worst a harmless
//!   duplicate (newest-xid-wins resolution is unaffected).
//! - Segment tier reconcile: a relocation renames the segment file before
//!   the registry write, so a crash between the two leaves the catalog
//!   naming a path with no file behind it. The registration is repaired
//!   from the tier directory that actually holds the file, and the staging
//!   files and stale copies an interrupted move can leave are swept.

use std::collections::{HashMap, HashSet};
use std::path::Path;

use tracing::{error, info, warn};

use zyron_catalog::schema::ColumnarSegmentEntry;
use zyron_catalog::{Catalog, TableId};
use zyron_common::StorageTier;
use zyron_common::page::{PAGE_SIZE, PageId};
use zyron_storage::columnar::{
    ColumnarBranchClearPayload, ColumnarPatchManager, ColumnarPatchRevokePayload,
    ColumnarSupersedePayload, ColumnarValuePatchPayload, columnar_root_for_segment,
    tier_segment_dir,
};

/// WAL patch record kinds collected for replay into the patch store
enum PatchKind {
    Value,
    Supersede,
    RevokeValue,
    RevokeSupersede,
    BranchClear,
}
use zyron_storage::{DiskManager, HeapPage, TupleSlot};
use zyron_wal::reader::WalReader;
use zyron_wal::record::LogRecordType;

fn rd_u64(b: &[u8], off: usize) -> u64 {
    u64::from_le_bytes(b[off..off + 8].try_into().unwrap())
}
fn rd_u32(b: &[u8], off: usize) -> u32 {
    u32::from_le_bytes(b[off..off + 4].try_into().unwrap())
}

struct EndRec {
    table_id: u64,
    file_id: u64,
    row_count: u64,
    base_rowid: u64,
    next_rowid: u64,
    xmin_lo: u64,
    xmin_hi: u64,
    path: String,
    rid_path: String,
}

fn parse_end(p: &[u8]) -> Option<EndRec> {
    if p.len() < 68 {
        return None;
    }
    let table_id = rd_u64(p, 0);
    let file_id = rd_u64(p, 8);
    let _file_size = rd_u64(p, 16);
    let row_count = rd_u64(p, 24);
    let base_rowid = rd_u64(p, 32);
    let next_rowid = rd_u64(p, 40);
    let xmin_lo = rd_u64(p, 48);
    let xmin_hi = rd_u64(p, 56);
    let path_len = rd_u32(p, 64) as usize;
    let mut off = 68;
    if off + path_len + 4 > p.len() {
        return None;
    }
    let path = String::from_utf8_lossy(&p[off..off + path_len]).into_owned();
    off += path_len;
    let rp_len = rd_u32(p, off) as usize;
    off += 4;
    if off + rp_len > p.len() {
        return None;
    }
    let rid_path = String::from_utf8_lossy(&p[off..off + rp_len]).into_owned();
    Some(EndRec {
        table_id,
        file_id,
        row_count,
        base_rowid,
        next_rowid,
        xmin_lo,
        xmin_hi,
        path,
        rid_path,
    })
}

/// Reconciles columnar fold and patch state from the WAL. Idempotent.
pub async fn reconcile_columnar(
    wal_dir: &Path,
    catalog: &Catalog,
    disk: &DiskManager,
    columnar_dir: &Path,
) -> zyron_common::Result<()> {
    // No WAL yet means no fold or patch records to replay. The tier
    // reconcile at the end still runs, it reads only the catalog and the
    // filesystem
    let records = match WalReader::new(wal_dir) {
        Ok(r) => r.scan_all_trusted(),
        Err(_) => Vec::new(),
    };

    let mut begins: HashMap<String, u64> = HashMap::new(); // path -> table_id
    let mut ends: Vec<EndRec> = Vec::new();
    let mut end_paths: HashSet<String> = HashSet::new();
    // (is_supersede, wal_lsn, payload)
    let mut patch_payloads: Vec<(PatchKind, u64, Vec<u8>)> = Vec::new();
    // (table_id, old_file_id) merged away by a committed MergeEnd. A stale
    // CompactionEnd for one of these must not resurrect the old segment.
    let mut merged_away: HashSet<(u64, u64)> = HashSet::new();
    let mut merge_begin_paths: HashMap<String, ()> = HashMap::new();
    let mut merge_end_paths: HashSet<String> = HashSet::new();

    for rec in &records {
        match rec.record_type {
            LogRecordType::CompactionBegin => {
                if rec.payload.len() >= 8 {
                    let tid = rd_u64(&rec.payload, 0);
                    let path = String::from_utf8_lossy(&rec.payload[8..]).into_owned();
                    begins.insert(path, tid);
                }
            }
            LogRecordType::CompactionEnd => {
                if let Some(e) = parse_end(&rec.payload) {
                    end_paths.insert(e.path.clone());
                    ends.push(e);
                }
            }
            LogRecordType::ColumnarPatch => {
                patch_payloads.push((PatchKind::Value, rec.lsn.0, rec.payload.to_vec()));
            }
            LogRecordType::ColumnarSupersede => {
                patch_payloads.push((PatchKind::Supersede, rec.lsn.0, rec.payload.to_vec()));
            }
            LogRecordType::ColumnarPatchRevoke => {
                patch_payloads.push((PatchKind::RevokeValue, rec.lsn.0, rec.payload.to_vec()));
            }
            LogRecordType::ColumnarSupersedeRevoke => {
                patch_payloads.push((PatchKind::RevokeSupersede, rec.lsn.0, rec.payload.to_vec()));
            }
            LogRecordType::ColumnarBranchClear => {
                patch_payloads.push((PatchKind::BranchClear, rec.lsn.0, rec.payload.to_vec()));
            }
            LogRecordType::MergeBegin => {
                if rec.payload.len() >= 8 {
                    let path = String::from_utf8_lossy(&rec.payload[8..]).into_owned();
                    merge_begin_paths.insert(path, ());
                }
            }
            LogRecordType::MergeEnd => {
                // Two shapes. Whole-segment-died: table_id(8) old_file_id(8).
                // Normal: table_id(8) new_file_id(8) old_file_id(8)
                //   path_len(4) path.
                if rec.payload.len() == 16 {
                    let tid = rd_u64(&rec.payload, 0);
                    let old_fid = rd_u64(&rec.payload, 8);
                    merged_away.insert((tid, old_fid));
                } else if rec.payload.len() >= 24 {
                    let tid = rd_u64(&rec.payload, 0);
                    let old_fid = rd_u64(&rec.payload, 16);
                    merged_away.insert((tid, old_fid));
                    let plen = rd_u32(&rec.payload, 24) as usize;
                    if 28 + plen <= rec.payload.len() {
                        let path =
                            String::from_utf8_lossy(&rec.payload[28..28 + plen]).into_owned();
                        merge_end_paths.insert(path);
                    }
                }
            }
            _ => {}
        }
    }

    // Committed folds: idempotently register and zero the folded heap slots.
    for e in &ends {
        // A segment that a committed MergeEnd replaced must not be
        // resurrected by its now-stale CompactionEnd.
        if merged_away.contains(&(e.table_id, e.file_id)) {
            continue;
        }
        let tid = TableId(e.table_id as u32);
        let te = match catalog.get_table_by_id(tid) {
            Ok(t) => t,
            Err(_) => continue,
        };
        let mut entry = te.as_ref().clone();
        let already = entry
            .columnar
            .segments
            .iter()
            .any(|s| s.path == e.path || s.file_id == e.file_id);
        if !already {
            entry.columnar.segments.push(ColumnarSegmentEntry {
                file_id: e.file_id,
                path: e.path.clone(),
                row_count: e.row_count,
                sys_rowid_lo: e.base_rowid,
                sys_rowid_hi: e.next_rowid.saturating_sub(1),
                sys_xmin_lo: e.xmin_lo,
                sys_xmin_hi: e.xmin_hi,
                // Recovery re-registers a file the fold already wrote. The
                // spec it was written under is not in the WAL record, and
                // reading it back out of the file would be guesswork, so it
                // registers as unclustered and the next pass picks it up
                cluster_spec_id: 0,
                // Recovery reads the file where it found it, and the fold
                // writes into the columnar root, so a re-registered segment
                // is hot
                storage_tier: 0,
            });
            entry.columnar.next_rowid = entry.columnar.next_rowid.max(e.next_rowid);
            entry.columnar.next_file_id = entry.columnar.next_file_id.max(e.file_id + 1);
            if let Err(err) = catalog.update_table(entry).await {
                warn!("columnar recovery: registry update failed: {}", err);
                continue;
            }
            info!(
                "columnar recovery: re-registered segment {} ({} rows)",
                e.path, e.row_count
            );
        }
        // Idempotent heap delete of the folded RIDs, read from the sidecar.
        // If the sidecar is gone the live apply already completed durably
        // (heap pages written, registry persisted), so there is nothing to
        // redo.
        let rid_path = std::path::Path::new(&e.rid_path);
        if let Ok(rids) =
            crate::background::compaction::CompactionWorker::read_rid_sidecar(rid_path)
        {
            let mut pages: HashMap<(u32, u64), [u8; PAGE_SIZE]> = HashMap::new();
            let mut dirty: HashSet<(u32, u64)> = HashSet::new();
            for (pid, slot, folded_xmin) in &rids {
                let key = (pid.file_id, pid.page_num);
                let page = match pages.get_mut(&key) {
                    Some(p) => p,
                    None => {
                        let data = match disk.read_page(*pid).await {
                            Ok(d) => d,
                            Err(_) => continue,
                        };
                        pages.entry(key).or_insert(data)
                    }
                };
                // Empty only when the slot still holds the folded tuple. An
                // already emptied or reused slot (different xmin) is left
                // intact so a redo never destroys an unrelated row.
                let Some(tuple_slot) = HeapPage::live_slot_in_slice(&page[..], *slot) else {
                    continue;
                };
                if tuple_slot.header.xmin != *folded_xmin {
                    continue;
                }
                let so = HeapPage::DATA_START + (*slot as usize) * TupleSlot::SIZE;
                page[so] = 0;
                page[so + 1] = 0;
                dirty.insert(key);
            }
            for ((fid, pnum), data) in &pages {
                if !dirty.contains(&(*fid, *pnum)) {
                    continue;
                }
                let _ = disk.write_page(PageId::new(*fid, *pnum), data).await;
            }
            // Apply complete and durable; the sidecar is no longer needed.
            let _ = std::fs::remove_file(rid_path);
        }
    }

    // Uncommitted folds: delete the orphan .zyr (and any temp). The heap was
    // never touched, so it stays authoritative.
    for (path, _tid) in &begins {
        if end_paths.contains(path) {
            continue;
        }
        let _ = std::fs::remove_file(path);
        let tmp = Path::new(path).with_extension("zyr.tmp");
        let _ = std::fs::remove_file(tmp);
        // The RID sidecar shares the stem with extension .zyrrids.
        let rids = Path::new(path).with_extension("zyrrids");
        let _ = std::fs::remove_file(rids);
        info!("columnar recovery: discarded uncommitted segment {}", path);
    }

    // Uncommitted merges: a MergeBegin with no committed MergeEnd left an
    // orphan output. The input segments stay authoritative (registry still
    // points at them), so discard the partial merged file.
    for (path, _) in &merge_begin_paths {
        if merge_end_paths.contains(path) {
            continue;
        }
        let _ = std::fs::remove_file(path);
        let tmp = Path::new(path).with_extension("zyr.tmp");
        let _ = std::fs::remove_file(tmp);
        info!("columnar recovery: discarded uncommitted merge {}", path);
    }

    // Replay patch/supersede records into the patch store. Layouts match the
    // DML writers in operator/modify.rs.
    // A WAL patch is replayed only when its LSN exceeds the patch file's
    // persisted high-water. The .zyrpatch is fsynced per append, so a record
    // already in the file (lsn <= high-water) is durable and re-appending it
    // would duplicate it in the file and the overlay on every restart. The
    // window that needs replay is exactly a crash between the WAL log and the
    // patch-file fsync, which leaves lsn > high-water.
    if !patch_payloads.is_empty() {
        let mgr = ColumnarPatchManager::global(columnar_dir);
        for (kind, wal_lsn, p) in &patch_payloads {
            match kind {
                PatchKind::Supersede => {
                    let Some(d) = ColumnarSupersedePayload::decode(p) else {
                        continue;
                    };
                    if let Ok(store) = mgr.store(d.table_id) {
                        if *wal_lsn <= store.max_persisted_lsn() {
                            continue;
                        }
                        let _ = store.append_supersede(
                            d.branch,
                            d.file_id,
                            d.sys_rowid,
                            d.xid,
                            *wal_lsn,
                        );
                    }
                }
                PatchKind::Value => {
                    let Some((d, val)) = ColumnarValuePatchPayload::decode(p) else {
                        continue;
                    };
                    if let Ok(store) = mgr.store(d.table_id) {
                        if *wal_lsn <= store.max_persisted_lsn() {
                            continue;
                        }
                        let _ = store.append_value_patch(
                            d.branch,
                            d.file_id,
                            d.sys_rowid,
                            d.column_id,
                            d.xid,
                            *wal_lsn,
                            val,
                        );
                    }
                }
                PatchKind::RevokeSupersede => {
                    let Some(d) = ColumnarSupersedePayload::decode(p) else {
                        continue;
                    };
                    if let Ok(store) = mgr.store(d.table_id) {
                        if *wal_lsn <= store.max_persisted_lsn() {
                            continue;
                        }
                        let _ = store.revoke_supersede(
                            d.branch,
                            d.file_id,
                            d.sys_rowid,
                            d.xid,
                            *wal_lsn,
                        );
                    }
                }
                PatchKind::BranchClear => {
                    let Some(d) = ColumnarBranchClearPayload::decode(p) else {
                        continue;
                    };
                    if let Ok(store) = mgr.store(d.table_id) {
                        if *wal_lsn <= store.max_persisted_lsn() {
                            continue;
                        }
                        let _ = store.clear_branch(d.branch, *wal_lsn);
                    }
                }
                PatchKind::RevokeValue => {
                    let Some(d) = ColumnarPatchRevokePayload::decode(p) else {
                        continue;
                    };
                    if let Ok(store) = mgr.store(d.table_id) {
                        if *wal_lsn <= store.max_persisted_lsn() {
                            continue;
                        }
                        let _ = store.revoke_value_patch(
                            d.branch,
                            d.file_id,
                            d.sys_rowid,
                            d.column_id,
                            d.xid,
                            *wal_lsn,
                        );
                    }
                }
            }
        }
    }

    // Runs after the fold and merge reconcile so it sees the final registry
    reconcile_segment_tiers(catalog).await;

    Ok(())
}

/// Points every columnar segment registration at the tier directory that
/// actually holds its file, and sweeps the debris an interrupted relocation
/// can leave behind.
///
/// A relocation renames the segment file first and persists the registry
/// second, so a crash between the two leaves the catalog naming a path with
/// no file behind it while the bytes sit complete in another tier directory.
/// Segment file names are unique and a rename is atomic, so the disk state
/// is unambiguous and the registration is repaired from it. The cross-device
/// fallback can also strand a staging file or a duplicate copy, either of
/// which blocks a later relocation from placing the file under that name,
/// so both are removed. A registration whose file is found on no tier is
/// reported and left in place: dropping it would silently discard the rows
/// it serves, while restoring the file makes them readable again.
pub async fn reconcile_segment_tiers(catalog: &Catalog) {
    const TIERS: [StorageTier; 4] = [
        StorageTier::Hot,
        StorageTier::Warm,
        StorageTier::Cold,
        StorageTier::Archive,
    ];
    for table in catalog.list_all_tables() {
        if table.columnar.segments.is_empty() {
            continue;
        }
        // Cloned lazily on the first repair so an already-consistent table
        // costs no registry write
        let mut repaired: Option<zyron_catalog::schema::TableEntry> = None;
        for (idx, seg) in table.columnar.segments.iter().enumerate() {
            let recorded = std::path::PathBuf::from(&seg.path);
            let Some(root) = columnar_root_for_segment(&recorded).map(Path::to_path_buf) else {
                warn!(
                    "tier reconcile: segment path {} of table {} has no parent directory",
                    seg.path, table.name
                );
                continue;
            };
            let Some(file_name) = recorded.file_name().map(std::ffi::OsStr::to_os_string) else {
                warn!(
                    "tier reconcile: segment path {} of table {} names no file",
                    seg.path, table.name
                );
                continue;
            };
            // A staging file is never referenced by the catalog and only
            // blocks the next move from placing a file, so it goes first
            for t in TIERS {
                let staging = tier_segment_dir(&root, t.name())
                    .join(&file_name)
                    .with_extension("zyr.moving");
                if staging.exists() {
                    match std::fs::remove_file(&staging) {
                        Ok(()) => info!(
                            "tier reconcile: removed staging leftover {}",
                            staging.display()
                        ),
                        Err(e) => warn!(
                            "tier reconcile: could not remove staging leftover {}: {}",
                            staging.display(),
                            e
                        ),
                    }
                }
            }
            if recorded.exists() {
                // The recorded path is authoritative. A same-named file on
                // another tier is a stale copy from an interrupted
                // cross-device move, and under that name it would make the
                // next relocation unable to place the file
                for t in TIERS {
                    let other = tier_segment_dir(&root, t.name()).join(&file_name);
                    if other != recorded && other.is_file() {
                        match std::fs::remove_file(&other) {
                            Ok(()) => {
                                info!("tier reconcile: removed stale copy {}", other.display())
                            }
                            Err(e) => warn!(
                                "tier reconcile: could not remove stale copy {}: {}",
                                other.display(),
                                e
                            ),
                        }
                    }
                }
                // The tier byte drives planner costing and the already-there
                // check, so it has to agree with the directory the file is in
                let actual = TIERS
                    .into_iter()
                    .find(|t| {
                        recorded.parent() == Some(tier_segment_dir(&root, t.name()).as_path())
                    })
                    .unwrap_or(StorageTier::Hot);
                if seg.storage_tier != actual as u8 {
                    let entry = repaired.get_or_insert_with(|| table.as_ref().clone());
                    entry.columnar.segments[idx].storage_tier = actual as u8;
                    info!(
                        "tier reconcile: segment {} of table {} sits on tier {}, tier byte corrected",
                        seg.path,
                        table.name,
                        actual.name()
                    );
                }
                continue;
            }
            // The recorded path is gone: the file moved and the crash hit
            // before the registry write. Adopt the location that holds the
            // file, hottest first so a duplicate resolves to the cheaper read
            let mut found: Vec<(StorageTier, std::path::PathBuf)> = Vec::new();
            for t in TIERS {
                let candidate = tier_segment_dir(&root, t.name()).join(&file_name);
                if candidate.is_file() {
                    found.push((t, candidate));
                }
            }
            if found.is_empty() {
                error!(
                    "tier reconcile: segment {} of table {} is missing from every tier directory, \
                     its rows cannot be read until the file is restored",
                    seg.path, table.name
                );
                continue;
            }
            for (_, extra) in &found[1..] {
                match std::fs::remove_file(extra) {
                    Ok(()) => info!("tier reconcile: removed duplicate copy {}", extra.display()),
                    Err(e) => warn!(
                        "tier reconcile: could not remove duplicate copy {}: {}",
                        extra.display(),
                        e
                    ),
                }
            }
            let (adopt_tier, adopt_path) = &found[0];
            let entry = repaired.get_or_insert_with(|| table.as_ref().clone());
            entry.columnar.segments[idx].path = adopt_path.to_string_lossy().into_owned();
            entry.columnar.segments[idx].storage_tier = *adopt_tier as u8;
            info!(
                "tier reconcile: segment of table {} adopted at {} on tier {}",
                table.name,
                adopt_path.display(),
                adopt_tier.name()
            );
        }
        if let Some(entry) = repaired {
            if let Err(e) = catalog.update_table(entry).await {
                warn!(
                    "tier reconcile: registry update for table {} failed: {}",
                    table.name, e
                );
            }
        }
    }
}
