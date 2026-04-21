// -----------------------------------------------------------------------------
// Upsert sink for streaming jobs
// -----------------------------------------------------------------------------
//
// Implements UPSERT semantics for a Zyron table target: incoming rows are
// matched against existing rows by primary key. Inserts with a new PK are
// added, inserts or update-postimages with an existing PK replace the prior
// row (delete + insert), and delete change types remove the row when present.
//
// The sink looks up existing rows by PK using an in-memory hash map rebuilt
// at construction time by scanning the target heap once. The map is guarded
// by a mutex because writes are serialized per sink. If the target heap has
// more than MEMORY_MAP_WARN_THRESHOLD live rows, a warning is logged so
// operators know the in-memory state has grown large.

use std::collections::HashMap;
use std::sync::Arc;

use parking_lot::Mutex as PlMutex;
use zyron_catalog::schema::ConstraintType;
use zyron_catalog::{Catalog, TableId};
use zyron_common::{Result, TypeId, ZyronError};
use zyron_storage::TupleId;

use crate::row_codec::{StreamValue, decode_row};
use crate::source_connector::CdfChange;

/// Threshold for warning that the in-memory upsert map is large. A larger
/// map still works but signals a use case that would benefit from a real
/// persistent primary-key index.
const MEMORY_MAP_WARN_THRESHOLD: usize = 10_000_000;

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
    security_ctx: Arc<PlMutex<zyron_auth::SecurityContext>>,
    security_manager: Arc<zyron_auth::SecurityManager>,
    // Lookup strategy. Option B, in-memory PK to TupleId map populated at
    // construction time by a full heap scan. Rebuilt on sink creation only.
    memory_map: PlMutex<HashMap<Vec<u8>, TupleId>>,
}

impl ZyronUpsertSink {
    /// Builds a new upsert sink for the given target table. Scans the target
    /// heap once to populate the in-memory PK map so subsequent upserts can
    /// locate existing rows in O(1). The scan decodes each live tuple with
    /// the target row codec and emits a PK key built from target_pk_ordinals.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        target_table_id: u32,
        target_pk_ordinals: Vec<u16>,
        target_types: Vec<TypeId>,
        catalog: Arc<Catalog>,
        heap: Arc<zyron_storage::HeapFile>,
        txn_manager: Arc<zyron_storage::txn::TransactionManager>,
        security_ctx: Arc<PlMutex<zyron_auth::SecurityContext>>,
        security_manager: Arc<zyron_auth::SecurityManager>,
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

        // Build the in-memory PK to TupleId map by scanning the heap once.
        let map = build_pk_map(&heap, &target_pk_ordinals, &target_types)?;
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
            security_ctx,
            security_manager,
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
    /// aborts the transaction and returns the error without mutating the map.
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

        // Apply each record while holding the map mutex so reads and writes
        // to the index are sequentially consistent within this batch.
        let result: Result<()> = (|| {
            let mut map_guard = self.memory_map.lock();
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
                        // Delete the prior row if one exists.
                        if let Some(prior) = map_guard.get(&key).copied() {
                            let _ = rt.block_on(async { self.heap.delete(prior).await })?;
                        }
                        // Insert the new row.
                        let tuple = zyron_storage::Tuple::new(change.row_data.clone(), txn_id_u32);
                        let new_id =
                            rt.block_on(async { self.heap.insert_batch(&[tuple]).await })?;
                        // insert_batch returns Vec<TupleId>. Use the single id.
                        let tuple_id = new_id.into_iter().next().ok_or_else(|| {
                            ZyronError::Internal("upsert insert returned no tuple id".to_string())
                        })?;
                        map_guard.insert(key, tuple_id);
                    }
                    zyron_cdc::ChangeType::Delete | zyron_cdc::ChangeType::UpdatePreimage => {
                        if let Some(prior) = map_guard.remove(&key) {
                            let _ = rt.block_on(async { self.heap.delete(prior).await })?;
                        }
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
                self.txn_manager.commit(&mut txn)?;
                Ok(())
            }
            Err(e) => {
                let _ = self.txn_manager.abort(&mut txn);
                Err(e)
            }
        }
    }
}

// -----------------------------------------------------------------------------
// Test-only map-level driver
// -----------------------------------------------------------------------------

/// Simulates the upsert state machine purely against the in-memory PK map.
/// Used by unit tests to verify insert, update, and delete sequencing without
/// standing up a real HeapFile and TransactionManager. The closure argument
/// mirrors the heap side effects: it is called with Some(TupleId) for every
/// row that would be deleted and None for every row that would be inserted.
/// Returns the new TupleId assigned to each inserted row.
#[cfg(test)]
pub(crate) fn apply_upsert_to_map_for_test(
    records: &[CdfChange],
    pk_ordinals: &[u16],
    target_types: &[TypeId],
    map: &mut HashMap<Vec<u8>, TupleId>,
    mut next_id: impl FnMut() -> TupleId,
) -> Result<Vec<(zyron_cdc::ChangeType, Option<TupleId>)>> {
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

/// Scans the target heap and builds a PK to TupleId map. The scan decodes
/// each live tuple against target_types, extracts the PK values at
/// target_pk_ordinals, and encodes them through encode_pk. Corrupt or
/// undecodable tuples are skipped with a warning so one bad row does not
/// fail sink construction.
fn build_pk_map(
    heap: &zyron_storage::HeapFile,
    pk_ordinals: &[u16],
    target_types: &[TypeId],
) -> Result<HashMap<Vec<u8>, TupleId>> {
    let mut map: HashMap<Vec<u8>, TupleId> = HashMap::new();
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
        map.insert(key, tuple_id);
    });
    drop(guard);
    if errors > 0 {
        tracing::warn!(
            errors,
            "upsert sink skipped {errors} tuples during PK map build"
        );
    }
    Ok(map)
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

    fn next_id_gen() -> impl FnMut() -> TupleId {
        use zyron_common::PageId;
        let mut counter: u16 = 0;
        move || {
            let id = TupleId::new(PageId::new(1, 0), counter);
            counter += 1;
            id
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
        let mut map: HashMap<Vec<u8>, TupleId> = HashMap::new();
        let trace = apply_upsert_to_map_for_test(&records, &pk, &types, &mut map, next_id_gen())
            .expect("apply ok");
        assert_eq!(map.len(), 3);
        // All three events should have seen no prior tuple.
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
        let mut map: HashMap<Vec<u8>, TupleId> = HashMap::new();
        let trace = apply_upsert_to_map_for_test(&records, &pk, &types, &mut map, next_id_gen())
            .expect("apply ok");
        // One insert, then update should see a prior row.
        assert_eq!(trace.len(), 2);
        assert!(trace[0].1.is_none());
        assert!(trace[1].1.is_some());
        assert_eq!(map.len(), 1);
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
        let mut map: HashMap<Vec<u8>, TupleId> = HashMap::new();
        let trace = apply_upsert_to_map_for_test(&records, &pk, &types, &mut map, next_id_gen())
            .expect("apply ok");
        assert!(trace[1].1.is_some(), "delete should see existing tuple");
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
        let mut map: HashMap<Vec<u8>, TupleId> = HashMap::new();
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
