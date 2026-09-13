// -----------------------------------------------------------------------------
// Outbound CDC stream pump.
//
// Drives every active CDC stream: reads change records the table's change feed
// has accumulated past the change stream the outbound stream consumes, decodes
// each row against the table schema, and delivers them to the configured sink
// (Kafka, S3, or webhook). The change stream's position moves in a commit of
// the pass's own once the sink has taken the records, under the position lock
// every consumer of that stream takes. Delivery runs on a blocking thread
// because the network sinks perform synchronous IO through a bridged runtime.
// -----------------------------------------------------------------------------

use std::sync::Arc;
use std::sync::atomic::AtomicBool;
use std::time::Duration;

use tracing::warn;

use zyron_catalog::{ColumnEntry, TableEntry, TableId};
use zyron_cdc::cdc_stream::build_sink;
use zyron_cdc::decoder::DecodedChange;
use zyron_cdc::{ChangeRecord, ChangeType};
use zyron_common::TypeId;
use zyron_executor::batch::ColumnBuilder;
use zyron_executor::epoch_decode::{EpochDecoder, ProjectedEpochDecoder};
use zyron_wire::connection::ServerState;

/// How many of a stream's sink batches one page of a lake table's backlog
/// holds. A pass derives the backlog a page at a time, so what is in
/// memory at once is a few batches rather than every pending version
const LAKE_PAGE_BATCHES: usize = 4;

use crate::replication::DdlRunner;

pub const DEFAULT_INTERVAL_SECS: u64 = 1;

pub async fn cdc_stream_pump_loop(
    server: Arc<ServerState>,
    shutdown: Arc<AtomicBool>,
    wake: Arc<tokio::sync::Notify>,
    interval_secs: u64,
) {
    let mut ticker = tokio::time::interval(Duration::from_secs(interval_secs.max(1)));
    while super::tick_until_shutdown(&mut ticker, &shutdown, &wake).await {
        run_pump_once(&server).await;
    }
}

/// Runs one delivery pass over every active stream. Returns the total number
/// of records delivered across all streams this pass.
///
/// Each stream delivers from the change stream it consumes and moves that
/// stream's position when its pass is done, under the same position lock a
/// SQL consumer of the stream takes, so an outbound stream and a transactional
/// consumer of one stream never take the same change
pub async fn run_pump_once(server: &Arc<ServerState>) -> u64 {
    let (Some(stream_mgr), Some(registry)) = (&server.cdc_stream_manager, &server.cdc_registry)
    else {
        return 0;
    };
    let streams: Vec<(zyron_cdc::cdc_stream::CdcOutputStream, Arc<TableEntry>)> = stream_mgr
        .list_streams()
        .into_iter()
        .filter(|stream| stream.active)
        // A soft-dropped or missing table is skipped, nothing to decode against
        .filter_map(|stream| {
            let table = server
                .catalog
                .get_table_by_id(TableId(stream.table_id))
                .ok()?;
            Some((stream, table))
        })
        .collect();

    // A delivery record this node still holds beside a change stream that
    // exists is retired on every member, whichever member created the stream
    for (stream, table) in &streams {
        if change_stream_of(server, stream, table).is_some() {
            if let Err(e) = retire_delivery_record(server, stream) {
                warn!(
                    target: "zyron::cdc",
                    stream = %stream.name,
                    "CDC stream delivery record could not be retired: {e}"
                );
            }
        }
    }

    // A group's outbound streams are driven by the member that leads it,
    // because the position moves in a commit the group agrees and a sink
    // written once per member would carry every change that many times
    if !server.raft.as_ref().map_or(true, |raft| raft.is_leader()) {
        return 0;
    }
    // A position moves in a commit every member applies. While a member
    // runs a binary that does not read the advance, the commit would be
    // refused after the sink took the records and the next pass would hand
    // them over again, so nothing is delivered until the group carries it
    if server
        .replication
        .as_ref()
        .is_some_and(|router| !router.carries_stream_advance())
    {
        return 0;
    }

    let mut total = 0u64;
    for (stream, table) in &streams {
        match deliver_one(server, registry, stream, table).await {
            Ok(n) => total += n,
            Err(e) => warn!(
                target: "zyron::cdc",
                table_id = table.id.0,
                stream = %stream.name,
                "CDC stream delivery failed: {e}"
            ),
        }
    }
    total
}

/// The change stream an outbound stream consumes, the one of that name in
/// the table's schema, or one of that name elsewhere that reads the table
fn change_stream_of(
    server: &Arc<ServerState>,
    stream: &zyron_cdc::cdc_stream::CdcOutputStream,
    table: &TableEntry,
) -> Option<Arc<zyron_catalog::ChangeStreamEntry>> {
    server
        .catalog
        .get_change_stream(table.schema_id, &stream.change_stream)
        .or_else(|| {
            server
                .catalog
                .change_streams_on_table(table.id.0)
                .into_iter()
                .find(|entry| entry.name == stream.change_stream)
        })
}

/// The replication slot an outbound stream recorded its delivery in before
/// its position was a change stream, when this node still holds it
fn delivery_slot(
    server: &Arc<ServerState>,
    stream: &zyron_cdc::cdc_stream::CdcOutputStream,
) -> Option<zyron_cdc::ReplicationSlot> {
    let slots = server.slot_manager.as_ref()?;
    slots.get_slot(&format!("{}_slot", stream.name)).ok()
}

/// Drops the delivery slot an outbound stream held, when this node still
/// holds one, so nothing pins the log for a record nothing reads
fn retire_delivery_record(
    server: &Arc<ServerState>,
    stream: &zyron_cdc::cdc_stream::CdcOutputStream,
) -> zyron_common::Result<()> {
    if delivery_slot(server, stream).is_none() {
        return Ok(());
    }
    if let Some(slots) = server.slot_manager.as_ref() {
        slots.drop_slot(&format!("{}_slot", stream.name))?;
    }
    Ok(())
}

/// Puts an outbound stream whose state names a change stream that does not
/// exist yet, or that still holds the delivery slot it is carried over
/// from, where it can deliver, and answers with that stream.
///
/// A stream state written before positions were change streams recorded a
/// replication slot of its own. The state migrator names the change stream
/// without creating it, because creating a catalog object is this node's own
/// work when it stands alone and the group's when it is a member. Alone, the
/// stream is created here at the version the slot had reached. In a group it
/// is created by the statements every member runs, at the source's current
/// version and then moved to the count the slot's version names in this
/// member's feed, which is the same place on every member. The slot is
/// retired once the stream holds its place, so nothing pins the log for a
/// record nothing reads, and a slot found still held on a later pass means
/// the move or the retirement did not finish, so it is finished here before
/// anything is delivered past the place the slot names
async fn carry_over_delivery_record(
    server: &Arc<ServerState>,
    stream: &zyron_cdc::cdc_stream::CdcOutputStream,
    table: &Arc<TableEntry>,
) -> zyron_common::Result<Arc<zyron_catalog::ChangeStreamEntry>> {
    let implicit = zyron_cdc::cdc_stream::implicit_change_stream_name(&stream.name);
    let existing = change_stream_of(server, stream, table);
    if stream.change_stream != implicit {
        // A slot under a stream that consumes a change stream of its own
        // naming was not written by the carry-over, so it says nothing
        // about where that stream stands
        return existing.ok_or_else(|| {
            zyron_common::ZyronError::CdcStreamError(format!(
                "outbound stream '{}' consumes change stream '{}', which does not exist",
                stream.name, stream.change_stream
            ))
        });
    }
    let slot = delivery_slot(server, stream);
    let schema = server.catalog.get_schema_by_id(table.schema_id)?;
    match server.replication.as_ref() {
        None => {
            // A stream already here was created at the slot's version by
            // an earlier pass whose retirement of the slot did not finish
            if existing.is_none() {
                let origin = match &slot {
                    Some(slot) => zyron_catalog::ChangeStreamOrigin::Version(slot.confirmed_lsn),
                    None => zyron_catalog::ChangeStreamOrigin::Now,
                };
                zyron_wire::change_stream_dispatch::create_implicit_stream_at(
                    server,
                    schema.database_id,
                    0,
                    &implicit,
                    table,
                    origin,
                )
                .await?;
            }
        }
        Some(router) => {
            let database = server
                .catalog
                .list_databases()
                .into_iter()
                .find(|db| db.id == schema.database_id)
                .map(|db| db.name.clone())
                .ok_or_else(|| {
                    zyron_common::ZyronError::Internal(format!(
                        "schema '{}' belongs to no database this node holds",
                        schema.name
                    ))
                })?;
            let context = zyron_executor::replication::StatementContext {
                user: "zyron".to_string(),
                database,
                search_path: zyron_catalog::default_search_path(),
                actor_role_id: None,
            };
            let qualified = format!("{}.{}", schema.name, implicit);
            if existing.is_none() {
                let create = zyron_parser::Statement::CreateChangeStream(Box::new(
                    zyron_parser::ast::CreateChangeStreamStatement {
                        name: qualified.clone(),
                        if_not_exists: false,
                        target: zyron_parser::ast::ChangeStreamTarget::Table(format!(
                            "{}.{}",
                            schema.name, table.name
                        )),
                        start: zyron_parser::ast::ChangeStreamStart::Now,
                        append_only: false,
                        predicate: None,
                        columns: Vec::new(),
                    },
                ));
                run_agreed(server, router.as_ref(), &create, &context).await?;
            }
            // The count the slot's version names in this member's feed, which
            // is where every member's copy of the stream is moved to. A slot
            // still held after the stream was created means the move did
            // not finish, and moving again lands on the same count
            let consumed = match (&slot, server.cdc_registry.as_ref()) {
                (Some(slot), Some(feeds)) => feeds
                    .get_feed(table.id.0)
                    .map(|feed| feed.records_at_or_below(slot.confirmed_lsn)),
                _ => None,
            };
            if let Some(consumed) = consumed {
                let reset = zyron_parser::Statement::AlterChangeStream(Box::new(
                    zyron_parser::ast::AlterChangeStreamStatement {
                        name: qualified,
                        action: zyron_parser::ast::AlterChangeStreamAction::ResetToPosition(
                            zyron_parser::ast::Expr::Literal(
                                zyron_parser::ast::LiteralValue::Integer(consumed as i64),
                            ),
                        ),
                    },
                ));
                run_agreed(server, router.as_ref(), &reset, &context).await?;
            }
        }
    }
    retire_delivery_record(server, stream)?;
    change_stream_of(server, stream, table).ok_or_else(|| {
        zyron_common::ZyronError::CdcStreamError(format!(
            "change stream '{implicit}' was not found after being created for outbound stream \
             '{}'",
            stream.name
        ))
    })
}

/// Runs one schema statement on a member of a group. Agreed with the group
/// first, then run here on this node's turn, the way a connection runs one
async fn run_agreed(
    server: &Arc<ServerState>,
    router: &dyn zyron_wire::connection::ReplicationRouter,
    statement: &zyron_parser::Statement,
    context: &zyron_executor::replication::StatementContext,
) -> zyron_common::Result<()> {
    let sql = zyron_parser::unparse::statement_to_sql(statement)
        .map_err(|e| zyron_common::ZyronError::Internal(e.to_string()))?;
    let agreed = router.begin_statement(&sql, context).await?;
    let entry = (agreed.index, agreed.timestamp_us);
    // The originating node runs the statement in no transaction of its
    // own, which is what zero says here, as the entry the group agreed
    let runner = crate::replication::DispatchedDdl::new(server);
    let outcome = runner.run(&sql, context, 0, entry).await;
    let report = match &outcome {
        Ok(()) => Ok(()),
        Err(e) => Err(zyron_common::ZyronError::Internal(e.to_string())),
    };
    let _ = agreed.done.send(report);
    outcome
}

/// One delivery pass for one stream
async fn deliver_one(
    server: &Arc<ServerState>,
    registry: &Arc<zyron_cdc::CdfRegistry>,
    stream: &zyron_cdc::cdc_stream::CdcOutputStream,
    table: &Arc<TableEntry>,
) -> zyron_common::Result<u64> {
    // A delivery slot still held means the carry-over from it never
    // finished, so it is finished before anything is delivered past the
    // place the slot names
    let entry = match change_stream_of(server, stream, table) {
        Some(entry) if delivery_slot(server, stream).is_none() => entry,
        _ => carry_over_delivery_record(server, stream, table).await?,
    };
    if entry.stale {
        return Err(zyron_common::ZyronError::CdcStreamError(format!(
            "outbound stream '{}' consumes change stream '{}', which is stale ({}). Reset it \
             with ALTER CHANGE STREAM {} RESET",
            stream.name, stream.change_stream, entry.stale_reason, stream.change_stream
        )));
    }

    // The pass runs as a transaction of its own, which is what holds the
    // position lock and what the advance record is written under. The
    // commit takes the same path a generated statement's does, so on a
    // member of a group the advance is agreed before the position moves
    let mut txn = server
        .txn_manager
        .begin(zyron_storage::txn::IsolationLevel::ReadCommitted)?;
    let txn_id = txn.txn_id;
    let changeset = server.replication.as_ref().map(|r| r.changeset(txn_id));
    let outcome = deliver_locked(server, registry, stream, table, &entry, &mut txn).await;
    let pass = match outcome {
        Ok(pass) => pass,
        Err(e) => {
            let _ = server.txn_manager.abort(&mut txn);
            return Err(e);
        }
    };
    if !pass.moved {
        // Nothing landed since the last pass, so the position stands where
        // it is and no advance is written or agreed for it. The transaction
        // read and wrote nothing, and the position it locked is released
        // with it
        let outcome = server.txn_manager.commit_read_only(&mut txn);
        zyron_wire::change_stream_dispatch::release_stream_positions(server, txn_id);
        outcome?;
        return Ok(0);
    }
    let advance = zyron_executor::context::PendingStreamAdvance {
        stream_id: entry.id,
        positions: vec![(table.id.0, pass.version, pass.consumed)],
    };
    let committed = zyron_wire::ddl_dispatch::commit_generated(
        server,
        txn,
        changeset,
        std::slice::from_ref(&advance),
    )
    .await;
    if let Err(e) = committed {
        // A commit that failed after the advance was logged still holds the
        // position. One that succeeded released it as the advance installed
        zyron_wire::change_stream_dispatch::release_stream_positions(server, txn_id);
        return Err(match e {
            zyron_wire::messages::ProtocolError::Database(e) => e,
            other => zyron_common::ZyronError::Internal(other.to_string()),
        });
    }
    Ok(pass.delivered)
}

/// Where a pass left the stream
struct PassOutcome {
    delivered: u64,
    version: u64,
    consumed: u64,
    /// Whether the pass delivered anything or found the position past
    /// where it stood, which is what earns an advance record
    moved: bool,
}

/// Delivers under the position lock and answers with where the position
/// moves to
async fn deliver_locked(
    server: &Arc<ServerState>,
    registry: &Arc<zyron_cdc::CdfRegistry>,
    stream: &zyron_cdc::cdc_stream::CdcOutputStream,
    table: &Arc<TableEntry>,
    entry: &zyron_catalog::ChangeStreamEntry,
    txn: &mut zyron_storage::txn::Transaction,
) -> zyron_common::Result<PassOutcome> {
    server
        .txn_manager
        .stream_positions()
        .lock_wait(txn.txn_id, entry.id as u64)
        .await?;
    // The position as it stands now that the lock is held, which a consumer
    // that committed while this pass waited may have moved
    let entry = server
        .catalog
        .get_change_stream_by_id(entry.id)
        .ok_or_else(|| {
            zyron_common::ZyronError::CdcStreamError(format!(
                "change stream '{}' was dropped while a delivery pass waited for it",
                stream.change_stream
            ))
        })?;
    let start_version = entry.position_of(table.id.0);
    let start_consumed = entry.consumed_of(table.id.0);

    // The whole entry, not just its columns. A row image is read through
    // the layout the epoch in the entry names
    let decoder = Arc::new(RowDecoder::new(table));
    let sink = build_sink(stream);
    let snapshot = server.txn_manager.refresh_snapshot(txn);
    let decision = move |txn_id: u64| match snapshot.txn_outcome(txn_id) {
        zyron_storage::txn::TxnStatus::Committed => zyron_cdc::TxnDecision::Committed,
        zyron_storage::txn::TxnStatus::Aborted => zyron_cdc::TxnDecision::Aborted,
        zyron_storage::txn::TxnStatus::Active => zyron_cdc::TxnDecision::InFlight,
    };
    let stream = stream.clone();

    // A lake table has no change file. Its transaction log is the change
    // record, so the records are derived from the log and driven through
    // the same batching and sink path
    if table.lake.is_lake() {
        let paths = zyron_lake::LakePaths::new(server.disk_manager.data_dir(), table.id.0);
        let Some(log) = zyron_lake::TransactionLog::lookup_shared(&paths) else {
            return Err(zyron_common::ZyronError::CdcStreamError(format!(
                "lake table '{}' has no open transaction log on this node",
                table.name
            )));
        };
        // The feed's source keeps the record index a stream position is
        // counted in, and says under which setting each commit's records
        // are derived
        let Some(source) = registry.derived(table.id.0) else {
            return Err(zyron_common::ZyronError::CdcStreamError(format!(
                "lake table '{}' has no change data feed registered on this node",
                table.name
            )));
        };
        let table = Arc::clone(table);
        let derive_under = Arc::clone(&source);
        let decoder = Arc::clone(&decoder);
        let pass = tokio::task::spawn_blocking(move || {
            // Each commit is derived under the setting it was counted
            // under, so the records match the index whatever the setting
            // was changed to since. The backlog is derived a page of
            // versions at a time, sized to a few of the sink's batches, so
            // a stream far behind never holds every pending change at once
            let page_records = stream.batch_size.max(1) * LAKE_PAGE_BATCHES;
            let latest = log.latest_version();
            let mut delivered = 0u64;
            let mut complete = start_version;
            let mut next = start_version + 1;
            while next <= latest {
                let mut changes = Vec::new();
                while next <= latest && changes.len() < page_records {
                    changes.extend(zyron_wire::lake_changes::lake_change_records(
                        &log,
                        &table,
                        next,
                        next,
                        derive_under.preimages_at(next),
                    )?);
                    next += 1;
                }
                let page_end = next - 1;
                let last_recorded = changes.last().map(|change| change.commit_version);
                let pass = zyron_cdc::drive_stream_changes(
                    &stream,
                    changes,
                    complete,
                    sink.as_ref(),
                    |rec| decoder.decode_change(rec),
                    // Lake change records derive from committed transaction
                    // log versions, an undecided change never appears there
                    &|_| zyron_cdc::TxnDecision::Committed,
                )?;
                delivered += pass.delivered;
                // A version that yielded no record is complete with the page
                // it was derived in, and a page handed over whole moves the
                // position to its end
                if last_recorded.is_none_or(|last| pass.complete_version >= last) {
                    complete = page_end;
                } else {
                    complete = pass.complete_version;
                    break;
                }
            }
            Ok::<_, zyron_common::ZyronError>(zyron_cdc::cdc_stream::DeliveryPass {
                delivered,
                complete_version: complete,
            })
        })
        .await
        .map_err(|e| zyron_common::ZyronError::Internal(format!("CDC pump task: {e}")))??;
        let consumed = source.records_at_or_below(pass.complete_version)?;
        return Ok(PassOutcome {
            delivered: pass.delivered,
            version: pass.complete_version,
            consumed,
            moved: pass.delivered > 0
                || pass.complete_version != start_version
                || consumed != start_consumed,
        });
    }

    let Some(feed) = registry.get_feed(table.id.0) else {
        return Err(zyron_common::ZyronError::CdcStreamError(format!(
            "table '{}' has no change data feed open on this node",
            table.name
        )));
    };
    // Heap change records land in the feed at DML execution time, inside
    // the transaction. The reader's snapshot is the authority on which of
    // those transactions actually committed, so rolled back changes never
    // reach a sink and an undecided transaction holds delivery until it
    // resolves
    let counted = Arc::clone(&feed);
    let pass = tokio::task::spawn_blocking(move || {
        zyron_cdc::drive_stream_once(
            &stream,
            feed.as_ref(),
            start_version,
            sink.as_ref(),
            |rec| decoder.decode_change(rec),
            &decision,
        )
    })
    .await
    .map_err(|e| zyron_common::ZyronError::Internal(format!("CDC pump task: {e}")))??;
    let consumed = counted.records_at_or_below(pass.complete_version);
    Ok(PassOutcome {
        delivered: pass.delivered,
        version: pass.complete_version,
        consumed,
        moved: pass.delivered > 0
            || pass.complete_version != start_version
            || consumed != start_consumed,
    })
}

/// Reads each change record's row image through the layout the record was
/// written under.
///
/// Built once per pass. The plan for every epoch the table ever wrote is
/// resolved up front, so a record costs one lookup by the epoch it carries,
/// and a record written under an earlier layout than the table has now reads
/// its later columns as the value a row written before them holds. A record
/// flagged as holding the feed's column subset decodes through the subset in
/// force at its epoch
struct RowDecoder {
    /// The columns each delivered row names, in the order they are reported
    columns: Vec<ColumnEntry>,
    decoder: EpochDecoder,
    /// Present when the feed ever recorded a column subset
    projected: Option<ProjectedEpochDecoder>,
    table_name: String,
    table_id: u32,
}

impl RowDecoder {
    fn new(table: &TableEntry) -> Self {
        let columns: Vec<ColumnEntry> = table.live_column_list();
        let output_ids: Vec<zyron_catalog::ColumnId> = columns.iter().map(|c| c.id).collect();
        let decoder = EpochDecoder::new(table, &output_ids);
        let projected = (!table.cdf.column_sets.is_empty())
            .then(|| ProjectedEpochDecoder::new(table, &output_ids, false));
        Self {
            columns,
            decoder,
            projected,
            table_name: table.name.clone(),
            table_id: table.id.0,
        }
    }

    /// Builds a DecodedChange from a raw change record. Insert and update
    /// post-images populate new_values, delete and update pre-images populate
    /// old_values, and schema or truncate markers carry no row image
    fn decode_change(&self, rec: &ChangeRecord) -> zyron_common::Result<DecodedChange> {
        let (old_values, new_values) = match rec.change_type {
            ChangeType::Insert | ChangeType::UpdatePostimage => (None, Some(self.pairs(rec)?)),
            ChangeType::Delete | ChangeType::UpdatePreimage => (Some(self.pairs(rec)?), None),
            ChangeType::SchemaChange | ChangeType::Truncate => (None, None),
        };
        Ok(DecodedChange {
            table_name: self.table_name.clone(),
            table_id: rec.table_id,
            operation: rec.change_type,
            old_values,
            new_values,
            commit_lsn: rec.commit_version,
            commit_timestamp: rec.commit_timestamp,
            txn_id: rec.txn_id,
            is_last_in_txn: rec.is_last_in_txn,
            schema_version: rec.schema_version,
        })
    }

    /// Decodes one record's row image into (column name, value) pairs.
    ///
    /// The record carries the epoch its writer stamped, which names the
    /// layout the bytes follow. A record whose epoch names no layout, or
    /// whose bytes do not span that layout, is reported rather than read at
    /// offsets that would land in the middle of a column, and the pass
    /// holds its position at the record so nothing past it is taken
    fn pairs(&self, rec: &ChangeRecord) -> zyron_common::Result<Vec<(String, String)>> {
        if self.columns.is_empty() || rec.row_data.is_empty() {
            return Ok(Vec::new());
        }
        let mut builders: Vec<ColumnBuilder> = self
            .columns
            .iter()
            .map(|c| {
                let phys = TypeId::timestamp_physical_type_id(c.type_id, c.fractional_digits);
                if phys != c.type_id || c.fractional_digits.is_some() {
                    ColumnBuilder::new_ts(c.type_id, phys, c.fractional_digits, 1)
                } else {
                    ColumnBuilder::new(c.type_id, 1)
                }
            })
            .collect();
        let epoch = rec.schema_version as u16;
        if rec.projected {
            match &self.projected {
                Some(projected) => projected.decode(epoch, &rec.row_data, &mut builders)?,
                None => {
                    return Err(zyron_common::ZyronError::CdcDecoderError(format!(
                        "table \"{}\" (id {}) has a change record at version {} flagged as \
                         holding a column subset, and the table's change data feed never \
                         recorded one, so the record cannot be decoded",
                        self.table_name, self.table_id, rec.commit_version
                    )));
                }
            }
        } else {
            let Some(plan) = self.decoder.plan(epoch) else {
                return Err(self.decoder.unknown_epoch(epoch, None));
            };
            if !plan.spans(&rec.row_data) {
                return Err(zyron_common::ZyronError::CdcDecoderError(format!(
                    "table \"{}\" (id {}) has a change record of {} bytes at version {} \
                     stamped with schema epoch {epoch} that is shorter than the layout the \
                     table wrote at that epoch",
                    self.table_name,
                    self.table_id,
                    rec.row_data.len(),
                    rec.commit_version
                )));
            }
            plan.decode_into(&rec.row_data, &mut builders);
        }

        let mut pairs = Vec::with_capacity(self.columns.len());
        for (col, builder) in self.columns.iter().zip(builders) {
            let column = builder.finish();
            let scalar = column.get_scalar(0);
            pairs.push((col.name.clone(), format!("{scalar}")));
        }
        Ok(pairs)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use zyron_catalog::{ColumnId, TableId};

    fn col(id: u16, name: &str, type_id: TypeId, max_length: Option<usize>) -> ColumnEntry {
        ColumnEntry {
            id: ColumnId(id),
            table_id: TableId(1),
            name: name.to_string(),
            type_id,
            ordinal: id,
            nullable: true,
            default_expr: None,
            max_length,
            fractional_digits: None,
            tz_offset_secs: None,
            element_type: None,
            attrs: Default::default(),
            absent_value: None,
            dropped: false,
        }
    }

    /// A table over `columns`, sealed at its first epoch, which is the layout
    /// a change record's row image is read through.
    fn table_of(columns: Vec<ColumnEntry>) -> TableEntry {
        let mut entry = TableEntry {
            id: TableId(1),
            schema_id: zyron_catalog::SchemaId(1),
            name: "t".to_string(),
            heap_file_id: 1,
            fsm_file_id: 2,
            columns,
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
        entry.seal_initial_epoch();
        entry
    }

    /// Encodes one NSM row: null bitmap then fixed-size values inline and
    /// variable-length values as a 4-byte length prefix plus bytes.
    fn encode(values: &[(bool, Vec<u8>, bool)]) -> Vec<u8> {
        let n = values.len();
        let bitmap_len = (n + 7) / 8;
        let mut out = vec![0u8; bitmap_len];
        for (i, (is_null, _, _)) in values.iter().enumerate() {
            if *is_null {
                out[i / 8] |= 1 << (i % 8);
            }
        }
        for (_, bytes, varlen) in values {
            if *varlen {
                out.extend_from_slice(&(bytes.len() as u32).to_le_bytes());
            }
            out.extend_from_slice(bytes);
        }
        out
    }

    /// A record of `change_type` whose writer stamped `epoch`
    fn record(change_type: ChangeType, epoch: u16, row: Vec<u8>, projected: bool) -> ChangeRecord {
        ChangeRecord {
            change_type,
            commit_version: 5,
            commit_timestamp: 100,
            table_id: 1,
            txn_id: 3,
            change_ordinal: 0,
            schema_version: epoch as u32,
            row_data: row,
            primary_key_data: Vec::new(),
            is_last_in_txn: true,
            projected,
        }
    }

    #[test]
    fn decodes_fixed_and_varlen_columns() {
        let columns = vec![
            col(0, "id", TypeId::Int64, None),
            col(1, "name", TypeId::Varchar, Some(255)),
        ];
        let row = encode(&[
            (false, 42i64.to_le_bytes().to_vec(), false),
            (false, b"hi".to_vec(), true),
        ]);
        let table = table_of(columns);
        let pairs = RowDecoder::new(&table)
            .pairs(&record(ChangeType::Insert, table.schema_epoch, row, false))
            .expect("decodes");
        assert_eq!(pairs.len(), 2);
        assert_eq!(pairs[0], ("id".to_string(), "42".to_string()));
        assert_eq!(pairs[1], ("name".to_string(), "hi".to_string()));
    }

    #[test]
    fn short_row_is_reported() {
        let columns = vec![col(0, "id", TypeId::Int64, None)];
        let table = table_of(columns);
        // One bitmap byte and no value bytes cannot span an Int64 column
        let err = RowDecoder::new(&table)
            .pairs(&record(
                ChangeType::Insert,
                table.schema_epoch,
                vec![0u8],
                false,
            ))
            .expect_err("a row shorter than its layout is reported");
        assert!(
            matches!(err, zyron_common::ZyronError::CdcDecoderError(_)),
            "{err}"
        );
    }

    #[test]
    fn empty_row_image_yields_no_pairs() {
        let columns = vec![col(0, "id", TypeId::Int64, None)];
        let table = table_of(columns);
        let pairs = RowDecoder::new(&table)
            .pairs(&record(
                ChangeType::Insert,
                table.schema_epoch,
                Vec::new(),
                false,
            ))
            .expect("an empty image names no value");
        assert!(pairs.is_empty());
    }

    #[test]
    fn delete_record_populates_old_values() {
        let columns = vec![col(0, "id", TypeId::Int64, None)];
        let row = encode(&[(false, 7i64.to_le_bytes().to_vec(), false)]);
        let table = table_of(columns);
        let decoded = RowDecoder::new(&table)
            .decode_change(&record(ChangeType::Delete, table.schema_epoch, row, false))
            .expect("decodes");
        assert!(decoded.new_values.is_none());
        assert_eq!(
            decoded.old_values.expect("a delete carries its old values")[0],
            ("id".to_string(), "7".to_string())
        );
    }

    /// A record written before a column was added reads through the layout
    /// of its own epoch, and the later column reads as the value a row
    /// written before it holds
    #[test]
    fn record_of_an_earlier_epoch_reads_through_its_own_layout() {
        let mut table = table_of(vec![col(0, "id", TypeId::Int64, None)]);
        let written_under = table.schema_epoch;
        let row = encode(&[(false, 9i64.to_le_bytes().to_vec(), false)]);
        // The column's DEFAULT, recorded as the bytes a row written before
        // the column reads
        let mut added = col(1, "note", TypeId::Varchar, Some(64));
        added.absent_value = Some(b"none".to_vec());
        table.columns.push(added);
        table.push_schema_epoch(table.current_physical_columns());
        assert_ne!(table.schema_epoch, written_under);

        let decoder = RowDecoder::new(&table);
        let pairs = decoder
            .pairs(&record(
                ChangeType::Insert,
                written_under,
                row.clone(),
                false,
            ))
            .expect("an earlier epoch's record decodes through its own layout");
        assert_eq!(pairs.len(), 2);
        assert_eq!(pairs[0], ("id".to_string(), "9".to_string()));
        assert_eq!(
            pairs[1],
            ("note".to_string(), "none".to_string()),
            "a column added after the record was written reads its recorded default"
        );

        // The same bytes stamped with the current epoch do not span the
        // two-column layout and are reported, never read past their end
        let err = decoder
            .pairs(&record(ChangeType::Insert, table.schema_epoch, row, false))
            .expect_err("a short row at the current epoch is reported");
        assert!(
            matches!(err, zyron_common::ZyronError::CdcDecoderError(_)),
            "{err}"
        );
    }

    #[test]
    fn unknown_epoch_is_reported() {
        let table = table_of(vec![col(0, "id", TypeId::Int64, None)]);
        let row = encode(&[(false, 1i64.to_le_bytes().to_vec(), false)]);
        let err = RowDecoder::new(&table)
            .pairs(&record(ChangeType::Insert, 7, row, false))
            .expect_err("an epoch the table never wrote is reported");
        assert!(
            matches!(err, zyron_common::ZyronError::CatalogCorrupted(_)),
            "{err}"
        );
    }

    #[test]
    fn projected_record_without_a_column_subset_is_reported() {
        let table = table_of(vec![col(0, "id", TypeId::Int64, None)]);
        let row = encode(&[(false, 1i64.to_le_bytes().to_vec(), false)]);
        let err = RowDecoder::new(&table)
            .pairs(&record(ChangeType::Insert, table.schema_epoch, row, true))
            .expect_err("a subset record on a feed that recorded none is reported");
        assert!(
            matches!(err, zyron_common::ZyronError::CdcDecoderError(_)),
            "{err}"
        );
    }
}
