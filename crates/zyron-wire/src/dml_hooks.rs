//! The DML hook that records a statement's rows into the change feeds.
//!
//! `CdcHookBridge` implements the `CdcHook` trait. Triggers run through the
//! executor's trigger dispatch from the catalog's entries and have no part
//! here.
//!
//! A statement hands the bridge every row it wrote at once, and the bridge
//! appends them to the feed in one batch with the rows borrowed from the
//! statement's own tuples. One lock acquisition, one encode pass and one
//! write per statement, and one copy of each row straight into the feed's
//! buffer, is what keeps the per-row cost of an enabled feed to the
//! record's own encoding.
//!
//! A feed configured with a column subset records each row narrowed to the
//! subset plus the table's key columns. The rows of one write are projected
//! into one buffer, walked once each, and the records borrow from it, so a
//! narrowed feed costs a copy of the kept bytes and nothing is decoded

use std::sync::Arc;

use zyron_catalog::{Catalog, TableId};
use zyron_cdc::change_feed::{ChangeHeader, RowChange, RowLayout};
use zyron_cdc::schema_evolution::RowProjection;
use zyron_cdc::{CdfRegistry, ChangeDataFeed, ChangeRecord, ChangeType};
use zyron_common::{Result, ZyronError};
use zyron_executor::context::CdcHook;

/// Bridges DML AFTER events to CDC change feed capture
pub struct CdcHookBridge {
    cdc_registry: Arc<CdfRegistry>,
    /// Resolves the schema epoch a table stamps into its tuples right now, so
    /// every change record says which layout its bytes were written under
    catalog: Option<Arc<Catalog>>,
}

impl CdcHookBridge {
    pub fn new(cdc_registry: Arc<CdfRegistry>) -> Self {
        Self {
            cdc_registry,
            catalog: None,
        }
    }

    /// Hands the bridge the catalog it reads schema epochs from
    pub fn with_catalog(mut self, catalog: Arc<Catalog>) -> Self {
        self.catalog = Some(catalog);
        self
    }

    /// The epoch a table's writes carry, which is the layout its change
    /// records decode through, and the projection the feed's column subset
    /// applies to each row. The epoch is zero where no catalog is installed,
    /// which is the epoch a tuple written before stamping reads under
    fn shape_of(&self, table_id: u32) -> Result<WriteShape> {
        let Some(table) = self
            .catalog
            .as_ref()
            .and_then(|catalog| catalog.get_table_by_id(TableId(table_id)).ok())
        else {
            return Ok(WriteShape {
                epoch: 0,
                table: None,
                projection: None,
            });
        };
        let recorded = table.cdf.recorded_columns();
        if recorded.is_empty() {
            return Ok(WriteShape {
                epoch: table.schema_epoch as u32,
                table: Some(table),
                projection: None,
            });
        }
        // The current epoch's layout is what the statement encoded its rows
        // under, so a table without it cannot say what its rows hold, and
        // recording them whole would contradict what the feed is configured
        // to record
        let Some(layout) = table.physical_columns_for_epoch(table.schema_epoch) else {
            return Err(ZyronError::CatalogCorrupted(format!(
                "table \"{}\" (id {}) has no recorded layout for its current schema epoch {}, \
                 so its change data feed cannot narrow the rows it records",
                table.name, table.id.0, table.schema_epoch
            )));
        };
        let projection = RowProjection::new(layout, recorded);
        let narrowed = table
            .projected_columns_for_epoch(table.schema_epoch, recorded)
            .unwrap_or_default();
        Ok(WriteShape {
            epoch: table.schema_epoch as u32,
            table: Some(table),
            projection: Some((projection, narrowed)),
        })
    }

    /// Records the changes of one write, narrowed to the feed's column subset
    /// when it has one.
    ///
    /// `changes` names each row with its kind and whether it closes the
    /// transaction. A feed recording every column borrows the rows as they
    /// are. One recording a subset projects them into one buffer first and
    /// the records borrow from that
    fn record<'a>(
        &self,
        feed: &ChangeDataFeed,
        table_id: u32,
        version: u64,
        timestamp: i64,
        txn_id: u64,
        changes: impl ExactSizeIterator<Item = (ChangeType, &'a [u8], bool)>,
    ) -> Result<()> {
        let shape = self.shape_of(table_id)?;
        let header = ChangeHeader {
            commit_version: version,
            commit_timestamp: timestamp,
            txn_id,
            schema_version: shape.epoch,
        };
        let Some((projection, narrowed)) = shape.projection else {
            // The rows are the table's tuples, laid out as its current epoch
            // says, which is what the feed slices them by when it seals
            let layout = RowLayout {
                columns: shape
                    .table
                    .as_ref()
                    .and_then(|table| table.physical_columns_for_epoch(table.schema_epoch)),
                projected: false,
            };
            let rows: Vec<RowChange<'a>> = changes
                .map(|(kind, row, last)| RowChange::of(kind, row, last))
                .collect();
            return feed.append_rows(&header, layout, &rows);
        };
        let mut buffer = Vec::with_capacity(changes.len() * 32);
        let mut spans = Vec::with_capacity(changes.len());
        for (kind, row, last) in changes {
            let start = buffer.len();
            projection.project(row, &mut buffer)?;
            spans.push((kind, start, buffer.len(), last));
        }
        let rows: Vec<RowChange<'_>> = spans
            .iter()
            .map(|(kind, start, end, last)| {
                RowChange::projected(*kind, &buffer[*start..*end], *last)
            })
            .collect();
        let layout = RowLayout {
            columns: Some(&narrowed),
            projected: true,
        };
        feed.append_rows(&header, layout, &rows)
    }
}

/// What one write's records on a table carry, the epoch that names the
/// layout, the table whose layout it is, and the projection narrowing each
/// row with the layout it leaves when the feed records a column subset
struct WriteShape {
    epoch: u32,
    table: Option<Arc<zyron_catalog::TableEntry>>,
    projection: Option<(RowProjection, Vec<zyron_catalog::PhysicalColumn>)>,
}

impl CdcHookBridge {
    /// The feed a write on a table is recorded in, the branch's when the
    /// write landed on a branch, opened on the first such write, and the
    /// table's own otherwise. None when the table records no changes
    fn feed_for(
        &self,
        table_id: u32,
        branch: Option<u64>,
    ) -> Result<Option<std::sync::Arc<zyron_cdc::ChangeDataFeed>>> {
        match branch {
            None => Ok(self.cdc_registry.get_feed(table_id)),
            Some(branch) => {
                if self.cdc_registry.get_feed(table_id).is_none() {
                    return Ok(None);
                }
                self.cdc_registry
                    .open_branch_feed(table_id, branch)
                    .map(Some)
            }
        }
    }
}

impl CdcHook for CdcHookBridge {
    fn records_rows_of(&self, table_id: u32, _branch: Option<u64>) -> bool {
        // A table whose changes go into a feed of its own records the
        // images, on the table or on any branch of it, since a branch's
        // feed opens on its first write. A lake table's changes are derived
        // from its log and a table without a feed records nothing, so
        // neither needs them
        self.cdc_registry.get_feed(table_id).is_some()
    }

    fn on_insert(
        &self,
        table_id: u32,
        tuples: &[&[u8]],
        version: u64,
        timestamp: i64,
        txn_id: u64,
        is_last_in_txn: bool,
        branch: Option<u64>,
    ) -> Result<()> {
        if let Some(feed) = self.feed_for(table_id, branch)? {
            let last = tuples.len().saturating_sub(1);
            self.record(
                &feed,
                table_id,
                version,
                timestamp,
                txn_id,
                tuples
                    .iter()
                    .enumerate()
                    .map(|(i, tuple)| (ChangeType::Insert, *tuple, is_last_in_txn && i == last)),
            )?;
        }
        Ok(())
    }

    fn on_delete(
        &self,
        table_id: u32,
        old_data: &[&[u8]],
        version: u64,
        timestamp: i64,
        txn_id: u64,
        is_last_in_txn: bool,
        branch: Option<u64>,
    ) -> Result<()> {
        if let Some(feed) = self.feed_for(table_id, branch)? {
            let last = old_data.len().saturating_sub(1);
            self.record(
                &feed,
                table_id,
                version,
                timestamp,
                txn_id,
                old_data
                    .iter()
                    .enumerate()
                    .map(|(i, old)| (ChangeType::Delete, *old, is_last_in_txn && i == last)),
            )?;
        }
        Ok(())
    }

    fn on_update(
        &self,
        table_id: u32,
        old_data: &[&[u8]],
        new_data: &[&[u8]],
        version: u64,
        timestamp: i64,
        txn_id: u64,
        is_last_in_txn: bool,
        branch: Option<u64>,
    ) -> Result<()> {
        if let Some(feed) = self.feed_for(table_id, branch)? {
            // The preimage and the postimage of one row are adjacent under
            // one version, so a consumer reading in ordinal order sees the
            // pair together. A feed configured without before images drops
            // the preimage as it writes, so nothing here decides that
            let paired = old_data.len().min(new_data.len());
            // The transaction's last record is the postimage of the last
            // pair written, so a consumer waiting for a whole transaction
            // sees its end whatever the two sides count
            let last = paired.saturating_sub(1);
            self.record(
                &feed,
                table_id,
                version,
                timestamp,
                txn_id,
                (0..paired * 2).map(|at| {
                    let i = at / 2;
                    if at % 2 == 0 {
                        (ChangeType::UpdatePreimage, old_data[i], false)
                    } else {
                        (
                            ChangeType::UpdatePostimage,
                            new_data[i],
                            is_last_in_txn && i == last,
                        )
                    }
                }),
            )?;
        }
        Ok(())
    }

    fn on_truncate(
        &self,
        table_id: u32,
        version: u64,
        timestamp: i64,
        txn_id: u64,
        branch: Option<u64>,
    ) -> Result<()> {
        if let Some(feed) = self.feed_for(table_id, branch)? {
            let record = ChangeRecord {
                change_type: ChangeType::Truncate,
                commit_version: version,
                commit_timestamp: timestamp,
                table_id,
                txn_id,
                change_ordinal: 0,
                schema_version: self.shape_of(table_id)?.epoch,
                row_data: Vec::new(),
                primary_key_data: Vec::new(),
                is_last_in_txn: true,
                projected: false,
            };
            feed.append_change(&record)?;
        }
        Ok(())
    }
}

// Legal-hold / WORM enforcement hooks live beside this so the
// per-connection executor and lifecycle DDL dispatch share one
// implementation. Re-exported here for the server's hook wiring
pub use crate::dml_enforce::{CompositeDmlHook, LegalHoldDmlHook};
