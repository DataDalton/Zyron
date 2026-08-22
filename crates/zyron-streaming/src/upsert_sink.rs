// -----------------------------------------------------------------------------
// Upsert sink for streaming jobs
// -----------------------------------------------------------------------------
//
// Implements UPSERT semantics for a Zyron table target: incoming rows are
// matched against existing rows by primary key. Inserts with a new PK are
// added, inserts or update-postimages with an existing PK replace the prior
// row (delete + insert), and delete change types remove the row when present.
//
// The sink looks up existing rows by PK using an in-memory hash map keyed to
// RowLocator, rebuilt at construction time by scanning the target heap and
// every columnar segment. Heap rows are replaced with an in-place delete,
// columnar rows with a WAL-logged supersede on the main branch, so upserts
// against folded rows update instead of duplicating. The map is guarded by a
// mutex because writes are serialized per sink. If the target has more than
// MEMORY_MAP_WARN_THRESHOLD live rows, a warning is logged so operators know
// the in-memory state has grown large.

use std::collections::HashMap;
use std::sync::Arc;

use parking_lot::Mutex as PlMutex;
use zyron_catalog::schema::ConstraintType;
use zyron_catalog::{Catalog, TableId};
use zyron_common::{Result, RowLocator, TypeId, ZyronError};
use zyron_storage::TupleId;

use crate::row_codec::{StreamValue, decode_fixed, decode_row, decode_varlen};
use crate::source_connector::CdfChange;

/// Threshold for warning that the in-memory upsert map is large. A larger
/// map still works but signals a use case that would benefit from a real
/// persistent primary-key index.
const MEMORY_MAP_WARN_THRESHOLD: usize = 10_000_000;

/// Result of deleting the row a prior locator addresses. Stale means the
/// locator no longer names a live row, which happens after a background fold
/// hands heap rows off to the columnar tier or a segment merge relocates them
enum PriorOutcome {
    Applied,
    Stale,
}

// -----------------------------------------------------------------------------
// ZyronUpsertSink
// -----------------------------------------------------------------------------

/// Sink that applies UPSERT write mode to a Zyron table. Incoming CdfChange
/// records whose change_type is Insert or UpdatePostimage overwrite the
/// existing row with a matching primary key, or insert when no match exists.
/// Delete and UpdatePreimage change types remove the row when present.
pub struct ZyronUpsertSink {
    target_table_id: u32,
    // Source-side ordinals of the primary-key columns. The source row encoded
    // in CdfChange.row_data is decoded once per row, and the values at these
    // ordinals are re-encoded into the lookup key.
    target_pk_ordinals: Vec<u16>,
    target_types: Vec<TypeId>,
    catalog: Arc<Catalog>,
    heap: Arc<zyron_storage::HeapFile>,
    txn_manager: Arc<zyron_storage::txn::TransactionManager>,
    wal: Arc<zyron_wal::WalWriter>,
    security_ctx: Arc<PlMutex<zyron_auth::SecurityContext>>,
    security_manager: Arc<zyron_auth::SecurityManager>,
    // The same per-table write counters DML maintains, so stat views count
    // streamed rows and the background workers' activity gates see them
    io_stats: Arc<zyron_common::TableIOStatsRegistry>,
    // In-memory PK to RowLocator map covering heap-resident and folded rows,
    // populated at construction time and healed on stale hits during writes
    memory_map: PlMutex<HashMap<Vec<u8>, RowLocator>>,
}

impl ZyronUpsertSink {
    /// Builds a new upsert sink for the given target table. Scans the target
    /// heap and every columnar segment once to populate the in-memory PK map
    /// so subsequent upserts can locate existing rows in O(1). Heap tuples
    /// are decoded with the target row codec, columnar rows are decoded from
    /// their segments with committed patch overlay values applied.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        target_table_id: u32,
        target_pk_ordinals: Vec<u16>,
        target_types: Vec<TypeId>,
        catalog: Arc<Catalog>,
        heap: Arc<zyron_storage::HeapFile>,
        txn_manager: Arc<zyron_storage::txn::TransactionManager>,
        wal: Arc<zyron_wal::WalWriter>,
        security_ctx: Arc<PlMutex<zyron_auth::SecurityContext>>,
        security_manager: Arc<zyron_auth::SecurityManager>,
        io_stats: Arc<zyron_common::TableIOStatsRegistry>,
    ) -> Result<Self> {
        if target_pk_ordinals.is_empty() {
            return Err(ZyronError::StreamingError(
                "upsert sink requires at least one primary key ordinal".to_string(),
            ));
        }
        for ord in &target_pk_ordinals {
            if (*ord as usize) >= target_types.len() {
                return Err(ZyronError::StreamingError(format!(
                    "upsert pk ordinal {} out of range for target arity {}",
                    ord,
                    target_types.len()
                )));
            }
        }

        // Validate the target table still has a PK declared in the catalog.
        let target_entry = catalog.get_table_by_id(TableId(target_table_id))?;
        let has_pk = target_entry
            .constraints
            .iter()
            .any(|c| c.constraint_type == ConstraintType::PrimaryKey);
        if !has_pk {
            return Err(ZyronError::StreamingError(format!(
                "upsert target table {} has no primary key constraint",
                target_table_id
            )));
        }

        // Build the in-memory PK to RowLocator map from both storage tiers.
        let map = build_pk_map(
            &heap,
            &catalog,
            target_table_id,
            &target_pk_ordinals,
            &target_types,
            txn_manager.status_map(),
            0,
        )?;
        if map.len() > MEMORY_MAP_WARN_THRESHOLD {
            tracing::warn!(
                target_table_id,
                live_rows = map.len(),
                "upsert sink memory map exceeds {MEMORY_MAP_WARN_THRESHOLD} rows, consider a persistent PK index"
            );
        }

        Ok(Self {
            target_table_id,
            target_pk_ordinals,
            target_types,
            catalog,
            heap,
            txn_manager,
            wal,
            security_ctx,
            security_manager,
            io_stats,
            memory_map: PlMutex::new(map),
        })
    }

    /// Returns the target table id configured for this sink.
    pub fn target_table_id(&self) -> u32 {
        self.target_table_id
    }

    /// Returns the current number of entries in the in-memory PK map.
    /// Used by tests to verify insert and delete behavior.
    pub fn live_row_count(&self) -> usize {
        self.memory_map.lock().len()
    }

    /// Applies a batch of CdfChanges to the target table with UPSERT semantics.
    /// Runs one INSERT and one DELETE privilege check up front, opens a single
    /// transaction, processes every change, and commits. Any storage error
    /// aborts the transaction and returns the error.
    pub fn write_batch(&self, records: Vec<CdfChange>) -> Result<()> {
        if records.is_empty() {
            return Ok(());
        }

        // Privilege check: UPSERT touches both INSERT and DELETE surfaces, so
        // both privileges are required on the target table.
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0);
        {
            let mut ctx = self.security_ctx.lock();
            let insert_ok = ctx.has_privilege(
                &self.security_manager.privilege_store,
                zyron_auth::privilege::PrivilegeType::Insert,
                zyron_auth::privilege::ObjectType::Table,
                self.target_table_id,
                None,
                now,
            );
            let delete_ok = ctx.has_privilege(
                &self.security_manager.privilege_store,
                zyron_auth::privilege::PrivilegeType::Delete,
                zyron_auth::privilege::ObjectType::Table,
                self.target_table_id,
                None,
                now,
            );
            if !insert_ok || !delete_ok {
                return Err(ZyronError::PermissionDenied(format!(
                    "streaming upsert sink lacks INSERT or DELETE on table {}",
                    self.target_table_id
                )));
            }
        }

        // Verify the target table still exists at write time.
        let _target = self
            .catalog
            .get_table_by_id(TableId(self.target_table_id))?;

        // Begin a transaction for the entire batch.
        let mut txn = self
            .txn_manager
            .begin(zyron_storage::txn::IsolationLevel::SnapshotIsolation)?;
        let txn_id_u32 = match u32::try_from(txn.txn_id) {
            Ok(v) => v,
            Err(_) => {
                let _ = self.txn_manager.abort(&mut txn);
                return Err(ZyronError::Internal(
                    "txn_id exceeds u32::MAX in upsert sink".to_string(),
                ));
            }
        };

        let rt = match tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
        {
            Ok(r) => r,
            Err(e) => {
                let _ = self.txn_manager.abort(&mut txn);
                return Err(ZyronError::Internal(format!(
                    "failed to build tokio runtime for upsert sink: {e}"
                )));
            }
        };

        // Row counts for the table's write counters, recorded only when the
        // transaction commits so an aborted batch counts nothing
        let mut applied_inserts = 0u64;
        let mut applied_updates = 0u64;
        let mut applied_deletes = 0u64;
        // Apply each record while holding the map mutex so reads and writes
        // to the index are sequentially consistent within this batch.
        let result: Result<()> = (|| {
            let mut map_guard = self.memory_map.lock();
            let mut rebuilt = false;
            for change in records {
                // Decode the source row once to pull the PK values. For Delete
                // and UpdatePreimage, the row_data still carries the image
                // needed to derive the PK. If row_data is empty, skip.
                if change.row_data.is_empty() {
                    continue;
                }
                let row = decode_row(&change.row_data, &self.target_types)?;
                let key = encode_pk(&row, &self.target_pk_ordinals, &self.target_types)?;

                match change.change_type {
                    zyron_cdc::ChangeType::Insert | zyron_cdc::ChangeType::UpdatePostimage => {
                        // Delete the prior row if one exists, in whichever
                        // storage tier it lives.
                        let replaced = self.delete_existing(
                            &rt,
                            &mut map_guard,
                            &key,
                            txn.txn_id,
                            &mut rebuilt,
                        )?;
                        // Insert the new row.
                        let tuple = zyron_storage::Tuple::new(change.row_data.clone(), txn_id_u32);
                        let new_id =
                            rt.block_on(async { self.heap.insert_batch(&[tuple]).await })?;
                        // insert_batch returns Vec<TupleId>. Use the single id.
                        let tuple_id = new_id.into_iter().next().ok_or_else(|| {
                            ZyronError::Internal("upsert insert returned no tuple id".to_string())
                        })?;
                        map_guard.insert(key, tuple_id.locator());
                        // A replacement leaves a superseded version behind,
                        // which is what the update counter tracks
                        if replaced {
                            applied_updates += 1;
                        } else {
                            applied_inserts += 1;
                        }
                    }
                    zyron_cdc::ChangeType::Delete | zyron_cdc::ChangeType::UpdatePreimage => {
                        if self.delete_existing(
                            &rt,
                            &mut map_guard,
                            &key,
                            txn.txn_id,
                            &mut rebuilt,
                        )? {
                            applied_deletes += 1;
                        }
                        map_guard.remove(&key);
                    }
                    // Schema changes and truncates are structural events,
                    // not per-row mutations the sink should re-apply.
                    zyron_cdc::ChangeType::SchemaChange | zyron_cdc::ChangeType::Truncate => {
                        continue;
                    }
                }
            }
            Ok(())
        })();

        match result {
            Ok(()) => {
                self.txn_manager.commit_blocking(&mut txn)?;
                let stats = self.io_stats.get_or_create(self.target_table_id);
                if applied_inserts > 0 {
                    stats.record_inserts(applied_inserts);
                }
                if applied_updates > 0 {
                    stats.record_updates(applied_updates);
                }
                if applied_deletes > 0 {
                    stats.record_deletes(applied_deletes);
                }
                Ok(())
            }
            Err(e) => {
                let _ = self.txn_manager.abort(&mut txn);
                Err(e)
            }
        }
    }

    /// Looks up and deletes the existing row for a key, healing map staleness
    /// caused by background folds and segment merges. On a stale locator the
    /// map is rebuilt from current storage once per batch and the delete is
    /// retried against the fresh locator. Returns true when a live prior row
    /// was deleted, false when no row holds the key
    fn delete_existing(
        &self,
        rt: &tokio::runtime::Runtime,
        map: &mut HashMap<Vec<u8>, RowLocator>,
        key: &[u8],
        txn_id: u64,
        rebuilt: &mut bool,
    ) -> Result<bool> {
        let Some(prior) = map.get(key).copied() else {
            return Ok(false);
        };
        if matches!(self.delete_prior(rt, prior, txn_id)?, PriorOutcome::Applied) {
            return Ok(true);
        }
        if *rebuilt {
            // Map is already current for this batch, so the row vanished
            // through concurrent DML between rebuild and delete. Surface it so
            // the runner retries the batch instead of silently duplicating
            return Err(ZyronError::StreamingError(format!(
                "upsert prior row vanished during write on table {}, retry batch",
                self.target_table_id
            )));
        }
        *map = build_pk_map(
            &self.heap,
            &self.catalog,
            self.target_table_id,
            &self.target_pk_ordinals,
            &self.target_types,
            self.txn_manager.status_map(),
            txn_id,
        )?;
        *rebuilt = true;
        let Some(fresh) = map.get(key).copied() else {
            return Ok(false);
        };
        match self.delete_prior(rt, fresh, txn_id)? {
            PriorOutcome::Applied => Ok(true),
            PriorOutcome::Stale => Err(ZyronError::StreamingError(format!(
                "upsert prior row vanished during write on table {}, retry batch",
                self.target_table_id
            ))),
        }
    }

    /// Deletes the row a locator addresses under the batch transaction. Heap
    /// rows are deleted in place. Columnar rows get a WAL-logged supersede on
    /// the main branch that becomes visible when the batch transaction
    /// commits. Returns Stale when the locator no longer names a live row
    fn delete_prior(
        &self,
        rt: &tokio::runtime::Runtime,
        loc: RowLocator,
        txn_id: u64,
    ) -> Result<PriorOutcome> {
        match loc {
            RowLocator::Heap { .. } => {
                let tid = TupleId::from_locator(loc).ok_or_else(|| {
                    ZyronError::Internal("heap locator failed TupleId conversion".to_string())
                })?;
                let deleted = rt.block_on(async { self.heap.delete(tid).await })?;
                Ok(if deleted {
                    PriorOutcome::Applied
                } else {
                    PriorOutcome::Stale
                })
            }
            RowLocator::Columnar { file_id, sys_rowid } => {
                // Re-fetch the entry so a background merge that relocated the
                // segment is detected against current registry state
                let entry = self
                    .catalog
                    .get_table_by_id(TableId(self.target_table_id))?;
                if !entry.columnar.segments.iter().any(|s| s.file_id == file_id) {
                    return Ok(PriorOutcome::Stale);
                }
                let store = zyron_storage::columnar::ColumnarPatchManager::store_for_segment(
                    self.target_table_id as u64,
                    std::path::Path::new(&entry.columnar.segments[0].path),
                )?;
                store.supersede_logged(
                    &self.wal,
                    self.target_table_id as u64,
                    0,
                    file_id,
                    sys_rowid,
                    txn_id,
                )?;
                Ok(PriorOutcome::Applied)
            }
            RowLocator::Lake { .. } => Err(ZyronError::Internal(
                "lake locator has no upsert delete path".to_string(),
            )),
        }
    }
}

// -----------------------------------------------------------------------------
// Test-only map-level driver
// -----------------------------------------------------------------------------

/// Simulates the upsert state machine purely against the in-memory PK map.
/// Used by unit tests to verify insert, update, and delete sequencing without
/// standing up a real HeapFile and TransactionManager. The closure argument
/// mirrors the storage side effects: it is consulted for every row that would
/// be deleted and produces the locator assigned to every inserted row.
#[cfg(test)]
pub(crate) fn apply_upsert_to_map_for_test(
    records: &[CdfChange],
    pk_ordinals: &[u16],
    target_types: &[TypeId],
    map: &mut HashMap<Vec<u8>, RowLocator>,
    mut next_id: impl FnMut() -> RowLocator,
) -> Result<Vec<(zyron_cdc::ChangeType, Option<RowLocator>)>> {
    let mut trace = Vec::new();
    for change in records {
        if change.row_data.is_empty() {
            continue;
        }
        let row = decode_row(&change.row_data, target_types)?;
        let key = encode_pk(&row, pk_ordinals, target_types)?;
        match change.change_type {
            zyron_cdc::ChangeType::Insert | zyron_cdc::ChangeType::UpdatePostimage => {
                let prior = map.get(&key).copied();
                let new_id = next_id();
                map.insert(key, new_id);
                trace.push((change.change_type, prior));
            }
            zyron_cdc::ChangeType::Delete | zyron_cdc::ChangeType::UpdatePreimage => {
                let prior = map.remove(&key);
                trace.push((change.change_type, prior));
            }
            zyron_cdc::ChangeType::SchemaChange | zyron_cdc::ChangeType::Truncate => {}
        }
    }
    Ok(trace)
}

// -----------------------------------------------------------------------------
// Helpers
// -----------------------------------------------------------------------------

/// Builds the PK to RowLocator map from both storage tiers. The heap pass
/// decodes each live tuple against target_types. The columnar pass decodes
/// each segment's PK columns with committed overlay patches applied and skips
/// rows with an effective supersede. current_txn marks the caller's own
/// in-flight transaction whose writes must count as applied, 0 when building
/// outside a transaction
fn build_pk_map(
    heap: &zyron_storage::HeapFile,
    catalog: &Catalog,
    target_table_id: u32,
    pk_ordinals: &[u16],
    target_types: &[TypeId],
    status_map: &zyron_storage::TxnStatusMap,
    current_txn: u64,
) -> Result<HashMap<Vec<u8>, RowLocator>> {
    let mut map: HashMap<Vec<u8>, RowLocator> = HashMap::new();
    let heap_errors = add_heap_rows(&mut map, heap, pk_ordinals, target_types)?;
    let columnar_errors = add_columnar_rows(
        &mut map,
        catalog,
        target_table_id,
        pk_ordinals,
        target_types,
        status_map,
        current_txn,
    )?;
    let errors = heap_errors + columnar_errors;
    if errors > 0 {
        tracing::warn!(
            errors,
            "upsert sink skipped {errors} rows during PK map build"
        );
    }
    Ok(map)
}

/// Scans the target heap and inserts a Heap locator per live tuple. Corrupt
/// or undecodable tuples are counted and skipped so one bad row does not fail
/// sink construction. Returns the skipped row count
fn add_heap_rows(
    map: &mut HashMap<Vec<u8>, RowLocator>,
    heap: &zyron_storage::HeapFile,
    pk_ordinals: &[u16],
    target_types: &[TypeId],
) -> Result<usize> {
    let guard = heap.scan()?;
    let pk_ordinals_local = pk_ordinals.to_vec();
    let target_types_local = target_types.to_vec();
    let mut errors: usize = 0;
    guard.for_each(|tuple_id, view| {
        if view.is_deleted() {
            return;
        }
        let row = match decode_row(view.data, &target_types_local) {
            Ok(r) => r,
            Err(_) => {
                errors += 1;
                return;
            }
        };
        let key = match encode_pk(&row, &pk_ordinals_local, &target_types_local) {
            Ok(k) => k,
            Err(_) => {
                errors += 1;
                return;
            }
        };
        // Later tuples with the same key (for example, from earlier in-place
        // updates that never got vacuumed) overwrite earlier ones. The live
        // MVCC visibility rules are not consulted here; this scan is a
        // best-effort starting state.
        map.insert(key, tuple_id.locator());
    });
    drop(guard);
    Ok(errors)
}

/// Decodes every columnar segment's PK columns and inserts a Columnar locator
/// per live folded row. A row is live when neither its base supersede stamp
/// nor any overlay supersede is effective, where effective means committed or
/// written by current_txn. The newest effective value patch per PK column
/// wins over the base cell so a folded row that was updated in place keys by
/// its current value. Returns the skipped row count
fn add_columnar_rows(
    map: &mut HashMap<Vec<u8>, RowLocator>,
    catalog: &Catalog,
    target_table_id: u32,
    pk_ordinals: &[u16],
    target_types: &[TypeId],
    status_map: &zyron_storage::TxnStatusMap,
    current_txn: u64,
) -> Result<usize> {
    let entry = catalog.get_table_by_id(TableId(target_table_id))?;
    if entry.columnar.segments.is_empty() {
        return Ok(0);
    }
    let store = zyron_storage::columnar::ColumnarPatchManager::store_for_segment(
        target_table_id as u64,
        std::path::Path::new(&entry.columnar.segments[0].path),
    )?;
    let effective = |x: u64| x == current_txn || status_map.is_committed(x);

    // Resolve the PK column descriptors by target ordinal
    let mut pk_cols = Vec::with_capacity(pk_ordinals.len());
    for ord in pk_ordinals {
        let col = entry
            .columns
            .iter()
            .find(|c| c.ordinal == *ord)
            .ok_or_else(|| {
                ZyronError::StreamingError(format!(
                    "pk ordinal {} not found in table {} columns",
                    ord, target_table_id
                ))
            })?;
        pk_cols.push(col.clone());
    }
    let pk_types: Vec<TypeId> = pk_ordinals
        .iter()
        .map(|o| target_types[*o as usize])
        .collect();

    let mut errors = 0usize;
    for seg in &entry.columnar.segments {
        let reader = zyron_storage::columnar::ZyrFileReader::open(std::path::Path::new(&seg.path))?;
        let row_count = reader.header().row_count as usize;
        if row_count == 0 {
            continue;
        }
        let (rowid_b, _) =
            reader.decode_column(zyron_storage::columnar::SYS_COL_ROWID, row_count, 8)?;
        let (super_b, _) =
            reader.decode_column(zyron_storage::columnar::SYS_COL_SUPERSEDE, row_count, 8)?;
        if rowid_b.len() < row_count * 8 || super_b.len() < row_count * 8 {
            return Err(ZyronError::InvalidZyrFile(format!(
                "system column decode short for segment {}",
                seg.file_id
            )));
        }
        let read_u64 = |b: &[u8], i: usize| -> u64 {
            let mut w = [0u8; 8];
            w.copy_from_slice(&b[i * 8..i * 8 + 8]);
            u64::from_le_bytes(w)
        };

        // Decode each PK column once per segment
        let mut pk_dec: Vec<(Vec<u8>, Vec<u8>, usize, bool)> = Vec::with_capacity(pk_cols.len());
        for col in &pk_cols {
            let phys = col.physical_type_id();
            let vs = phys.fixed_size().unwrap_or(0);
            let (bytes, nulls) = reader.decode_column(col.id.0 as u32, row_count, vs)?;
            if vs > 0 && bytes.len() < row_count * vs {
                return Err(ZyronError::InvalidZyrFile(format!(
                    "column {} decode short for segment {}",
                    col.id.0, seg.file_id
                )));
            }
            pk_dec.push((bytes, nulls, vs, vs == 0));
        }
        let mut varlen_rows: Vec<Option<Vec<&[u8]>>> = Vec::with_capacity(pk_dec.len());
        for (b, _, _, isv) in &pk_dec {
            varlen_rows.push(if *isv {
                Some(zyron_storage::encoding::varlen_slice_rows(b, row_count)?)
            } else {
                None
            });
        }

        // One overlay snapshot per segment, then plain local map lookups
        let overlay = store.file_overlay(seg.file_id);
        for r in 0..row_count {
            // Base supersede stamps are carried forward by merge and were
            // committed when carried, the row is dead
            if read_u64(&super_b, r) != 0 {
                continue;
            }
            let rid = read_u64(&rowid_b, r);
            let ov = overlay.get(&rid);
            if let Some(o) = ov {
                if o.supersedes.iter().any(|x| effective(*x)) {
                    continue;
                }
            }
            let mut vals: Vec<StreamValue> = Vec::with_capacity(pk_cols.len());
            let mut bad = false;
            for (ci, col) in pk_cols.iter().enumerate() {
                let phys = col.physical_type_id();
                let patch = ov
                    .and_then(|o| o.patches.get(&(col.id.0 as u32)))
                    .and_then(|chain| {
                        chain
                            .iter()
                            .filter(|p| effective(p.patch_xid))
                            .max_by_key(|p| p.patch_xid)
                    });
                let decoded = if let Some(p) = patch {
                    columnar_cell_to_value(phys, &p.value)
                } else {
                    let (bytes, nulls, vs, isv) = &pk_dec[ci];
                    let is_null = nulls.get(r / 8).is_some_and(|b| (b >> (r % 8)) & 1 == 1);
                    if is_null {
                        bad = true;
                        break;
                    }
                    if *isv {
                        match varlen_rows[ci].as_ref().and_then(|rows| rows.get(r)) {
                            Some(cell) => columnar_cell_to_value(phys, cell),
                            None => {
                                bad = true;
                                break;
                            }
                        }
                    } else {
                        columnar_cell_to_value(phys, &bytes[r * vs..(r + 1) * vs])
                    }
                };
                match decoded {
                    Ok(v) => vals.push(v),
                    Err(_) => {
                        bad = true;
                        break;
                    }
                }
            }
            if bad {
                errors += 1;
                continue;
            }
            let key = match crate::row_codec::encode_row(&vals, &pk_types) {
                Ok(k) => k,
                Err(_) => {
                    errors += 1;
                    continue;
                }
            };
            map.insert(
                key,
                RowLocator::Columnar {
                    file_id: seg.file_id,
                    sys_rowid: rid,
                },
            );
        }
    }
    Ok(errors)
}

/// Bridges a raw columnar cell to the StreamValue the row codec would produce
/// for the same logical value, so heap-built and columnar-built keys agree
fn columnar_cell_to_value(phys: TypeId, cell: &[u8]) -> Result<StreamValue> {
    match phys.fixed_size() {
        Some(size) => {
            if cell.len() < size {
                return Err(ZyronError::StreamingError(format!(
                    "columnar cell shorter than {size} bytes for {phys:?}"
                )));
            }
            decode_fixed(phys, cell)
        }
        None => decode_varlen(phys, cell),
    }
}

/// Re-encodes the PK columns of a decoded row into a deterministic byte key.
/// Uses the same NSM layout as row_codec::encode_row but only over the PK
/// subset. The result is suitable for direct hash map use because two rows
/// with identical PK values produce identical byte sequences.
fn encode_pk(row: &[StreamValue], pk_ordinals: &[u16], target_types: &[TypeId]) -> Result<Vec<u8>> {
    let mut pk_values = Vec::with_capacity(pk_ordinals.len());
    let mut pk_types = Vec::with_capacity(pk_ordinals.len());
    for ord in pk_ordinals {
        let idx = *ord as usize;
        if idx >= row.len() || idx >= target_types.len() {
            return Err(ZyronError::StreamingError(format!(
                "pk ordinal {} out of row arity {}",
                idx,
                row.len()
            )));
        }
        pk_values.push(row[idx].clone());
        pk_types.push(target_types[idx]);
    }
    crate::row_codec::encode_row(&pk_values, &pk_types)
}

// -----------------------------------------------------------------------------
// Tests
// -----------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use zyron_common::TypeId;

    #[test]
    fn test_encode_pk_deterministic() {
        let row = vec![StreamValue::I64(42), StreamValue::Utf8("hello".to_string())];
        let types = vec![TypeId::Int64, TypeId::Varchar];
        let k1 = encode_pk(&row, &[0u16], &types).expect("encode pk");
        let k2 = encode_pk(&row, &[0u16], &types).expect("encode pk");
        assert_eq!(k1, k2);
    }

    #[test]
    fn test_encode_pk_distinguishes_values() {
        let row_a = vec![StreamValue::I64(1), StreamValue::I64(2)];
        let row_b = vec![StreamValue::I64(2), StreamValue::I64(1)];
        let types = vec![TypeId::Int64, TypeId::Int64];
        let ka = encode_pk(&row_a, &[0u16], &types).expect("encode pk");
        let kb = encode_pk(&row_b, &[0u16], &types).expect("encode pk");
        assert_ne!(ka, kb);
    }

    #[test]
    fn test_encode_pk_composite() {
        let row = vec![
            StreamValue::I64(1),
            StreamValue::I64(2),
            StreamValue::I64(3),
        ];
        let types = vec![TypeId::Int64, TypeId::Int64, TypeId::Int64];
        let k = encode_pk(&row, &[0u16, 2u16], &types).expect("encode pk");
        let k_swap = encode_pk(&row, &[2u16, 0u16], &types).expect("encode pk");
        assert_ne!(k, k_swap);
    }

    #[test]
    fn test_columnar_cell_matches_row_codec_value() {
        // A fixed cell and a varlen cell must produce the same key bytes the
        // heap path would build for the same logical values
        let id_cell = 42i64.to_le_bytes();
        let name_cell = b"hello".to_vec();
        let id_val = columnar_cell_to_value(TypeId::Int64, &id_cell).expect("fixed cell");
        let name_val = columnar_cell_to_value(TypeId::Varchar, &name_cell).expect("varlen cell");
        let types = vec![TypeId::Int64, TypeId::Varchar];
        let heap_row = vec![StreamValue::I64(42), StreamValue::Utf8("hello".to_string())];
        let heap_key = encode_pk(&heap_row, &[0u16, 1u16], &types).expect("heap key");
        let columnar_key =
            crate::row_codec::encode_row(&[id_val, name_val], &types).expect("columnar key");
        assert_eq!(heap_key, columnar_key);
    }

    fn make_change(
        ct: zyron_cdc::ChangeType,
        values: &[StreamValue],
        types: &[TypeId],
    ) -> CdfChange {
        let row_data = crate::row_codec::encode_row(values, types).expect("encode row");
        CdfChange {
            commit_version: 0,
            commit_timestamp: 0,
            change_type: ct,
            row_data,
            primary_key_data: Vec::new(),
        }
    }

    fn next_id_gen() -> impl FnMut() -> RowLocator {
        use zyron_common::PageId;
        let mut counter: u16 = 0;
        move || {
            let id = TupleId::new(PageId::new(1, 0), counter);
            counter += 1;
            id.locator()
        }
    }

    #[test]
    fn test_upsert_inserts_new_rows() {
        let types = vec![TypeId::Int64, TypeId::Int64];
        let pk = [0u16];
        let records = vec![
            make_change(
                zyron_cdc::ChangeType::Insert,
                &[StreamValue::I64(1), StreamValue::I64(10)],
                &types,
            ),
            make_change(
                zyron_cdc::ChangeType::Insert,
                &[StreamValue::I64(2), StreamValue::I64(20)],
                &types,
            ),
            make_change(
                zyron_cdc::ChangeType::Insert,
                &[StreamValue::I64(3), StreamValue::I64(30)],
                &types,
            ),
        ];
        let mut map: HashMap<Vec<u8>, RowLocator> = HashMap::new();
        let trace = apply_upsert_to_map_for_test(&records, &pk, &types, &mut map, next_id_gen())
            .expect("apply ok");
        assert_eq!(map.len(), 3);
        // All three events should have seen no prior row.
        for (_, prior) in &trace {
            assert!(prior.is_none());
        }
    }

    #[test]
    fn test_upsert_updates_existing_rows() {
        let types = vec![TypeId::Int64, TypeId::Int64];
        let pk = [0u16];
        let records = vec![
            make_change(
                zyron_cdc::ChangeType::Insert,
                &[StreamValue::I64(1), StreamValue::I64(10)],
                &types,
            ),
            make_change(
                zyron_cdc::ChangeType::UpdatePostimage,
                &[StreamValue::I64(1), StreamValue::I64(99)],
                &types,
            ),
        ];
        let mut map: HashMap<Vec<u8>, RowLocator> = HashMap::new();
        let trace = apply_upsert_to_map_for_test(&records, &pk, &types, &mut map, next_id_gen())
            .expect("apply ok");
        // One insert, then update should see a prior row.
        assert_eq!(trace.len(), 2);
        assert!(trace[0].1.is_none());
        assert!(trace[1].1.is_some());
        assert_eq!(map.len(), 1);
    }

    #[test]
    fn test_upsert_update_replaces_a_columnar_prior() {
        // A folded row's Columnar locator must be seen as the prior and
        // replaced by the new insert's locator
        let types = vec![TypeId::Int64, TypeId::Int64];
        let pk = [0u16];
        let seed_row = vec![StreamValue::I64(7), StreamValue::I64(70)];
        let seed_key = encode_pk(&seed_row, &pk, &types).expect("seed key");
        let mut map: HashMap<Vec<u8>, RowLocator> = HashMap::new();
        map.insert(
            seed_key.clone(),
            RowLocator::Columnar {
                file_id: 3,
                sys_rowid: 12,
            },
        );
        let records = vec![make_change(
            zyron_cdc::ChangeType::UpdatePostimage,
            &[StreamValue::I64(7), StreamValue::I64(71)],
            &types,
        )];
        let trace = apply_upsert_to_map_for_test(&records, &pk, &types, &mut map, next_id_gen())
            .expect("apply ok");
        assert!(matches!(
            trace[0].1,
            Some(RowLocator::Columnar {
                file_id: 3,
                sys_rowid: 12
            })
        ));
        assert!(matches!(map.get(&seed_key), Some(RowLocator::Heap { .. })));
    }

    #[test]
    fn test_upsert_handles_delete_change_type() {
        let types = vec![TypeId::Int64, TypeId::Int64];
        let pk = [0u16];
        let records = vec![
            make_change(
                zyron_cdc::ChangeType::Insert,
                &[StreamValue::I64(1), StreamValue::I64(10)],
                &types,
            ),
            make_change(
                zyron_cdc::ChangeType::Delete,
                &[StreamValue::I64(1), StreamValue::I64(10)],
                &types,
            ),
        ];
        let mut map: HashMap<Vec<u8>, RowLocator> = HashMap::new();
        let trace = apply_upsert_to_map_for_test(&records, &pk, &types, &mut map, next_id_gen())
            .expect("apply ok");
        assert!(trace[1].1.is_some(), "delete should see existing row");
        assert_eq!(map.len(), 0);
    }

    #[test]
    fn test_upsert_delete_missing_is_noop() {
        let types = vec![TypeId::Int64, TypeId::Int64];
        let pk = [0u16];
        let records = vec![make_change(
            zyron_cdc::ChangeType::Delete,
            &[StreamValue::I64(42), StreamValue::I64(0)],
            &types,
        )];
        let mut map: HashMap<Vec<u8>, RowLocator> = HashMap::new();
        let trace = apply_upsert_to_map_for_test(&records, &pk, &types, &mut map, next_id_gen())
            .expect("apply ok");
        assert!(trace[0].1.is_none());
        assert_eq!(map.len(), 0);
    }

    #[test]
    fn test_encode_pk_out_of_range() {
        let row = vec![StreamValue::I64(1)];
        let types = vec![TypeId::Int64];
        let res = encode_pk(&row, &[5u16], &types);
        assert!(res.is_err());
    }
}
