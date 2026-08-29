//! Recording what a statement did, from inside the operators that did it.
//!
//! Capture sits next to the change-feed notification in every DML operator, so
//! anything that writes rows is caught once wherever it came from: a
//! statement, a procedure body, a `MERGE`, a `COPY`, a trigger's own writes.
//!
//! The rows are encoded straight into the buffer that becomes the log record.
//! Old row images are the exception, because a delete has to name rows that
//! are about to stop existing and the identity it names them by is decided
//! per table rather than per row.

use std::sync::Arc;

use zyron_catalog::TableEntry;
use zyron_common::Result;

use crate::batch::{DataBatch, encode_row};
use crate::context::ExecutionContext;
use crate::operator::modify::{encode_btree_index_key_into, index_key_columns};

use super::changeset::ReplicaIdentity;

/// Row images split by whether the table's replica identity could name them.
///
/// A unique index over a nullable column leaves rows with a null in the key
/// out of the index entirely, which is what SQL requires and what makes those
/// rows unfindable by a key probe. They are named by their whole image
/// instead, in an operation of their own, rather than being dropped or
/// silently mismatched
pub struct IdentityImages {
    /// Encoded index keys, one per row that the identity index covers
    pub keyed: Vec<Vec<u8>>,
    /// Row positions the keys belong to, aligned with `keyed`
    pub keyed_rows: Vec<usize>,
    /// Whole row encodings for rows the identity index does not cover
    pub imaged: Vec<Vec<u8>>,
    /// Row positions the images belong to, aligned with `imaged`
    pub imaged_rows: Vec<usize>,
}

impl IdentityImages {
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.keyed.is_empty() && self.imaged.is_empty()
    }
}

/// Names every row of `batch` the way the applier will look it up.
///
/// With a replica identity this is the encoded index key, which the applier
/// probes directly. Without one it is the whole encoded row, which the applier
/// matches during a scan. A table with no index makes the leader's own
/// `DELETE` a scan, so the second case is the same cost class the work was
/// already in rather than a penalty replication adds
pub fn identity_images(
    table: &TableEntry,
    identity: &ReplicaIdentity,
    batch: &DataBatch,
) -> IdentityImages {
    let mut out = IdentityImages {
        keyed: Vec::new(),
        keyed_rows: Vec::new(),
        imaged: Vec::new(),
        imaged_rows: Vec::new(),
    };
    match identity {
        ReplicaIdentity::Key { columns, .. } => {
            let ids: Vec<zyron_catalog::ColumnId> = columns
                .iter()
                .map(|c| zyron_catalog::ColumnId(*c))
                .collect();
            let Some(key_cols) = index_key_columns(table, &ids) else {
                // The index names a column the table no longer has, so the
                // key cannot be built and every row falls back to its image
                return whole_images(table, batch);
            };
            out.keyed.reserve(batch.num_rows);
            out.keyed_rows.reserve(batch.num_rows);
            let mut key = Vec::with_capacity(64);
            for row in 0..batch.num_rows {
                if encode_btree_index_key_into(batch, row, &key_cols, &mut key) {
                    out.keyed.push(key.clone());
                    out.keyed_rows.push(row);
                } else {
                    out.imaged.push(encode_row(batch, row, &table.columns));
                    out.imaged_rows.push(row);
                }
            }
        }
        ReplicaIdentity::FullImage => return whole_images(table, batch),
    }
    out
}

fn whole_images(table: &TableEntry, batch: &DataBatch) -> IdentityImages {
    let mut imaged = Vec::with_capacity(batch.num_rows);
    let mut imaged_rows = Vec::with_capacity(batch.num_rows);
    for row in 0..batch.num_rows {
        imaged.push(encode_row(batch, row, &table.columns));
        imaged_rows.push(row);
    }
    IdentityImages {
        keyed: Vec::new(),
        keyed_rows: Vec::new(),
        imaged,
        imaged_rows,
    }
}

/// The identity a table replicates by, resolved from the live catalog.
pub fn identity_of(ctx: &ExecutionContext, table: &TableEntry) -> ReplicaIdentity {
    let indexes = ctx.catalog.index_snapshot(table.id);
    ReplicaIdentity::of(table, &indexes)
}

/// Records rows entering a table, when this node is replicating.
///
/// A no-op on a node outside a consensus group, which is what keeps the check
/// on the write path down to one `Option` test
#[inline]
pub fn capture_insert(ctx: &ExecutionContext, table: &TableEntry, batch: &DataBatch) -> Result<()> {
    let Some(set) = ctx.replication.as_ref() else {
        return Ok(());
    };
    if ctx.replication_apply {
        // These rows arrived through consensus. Sending them back would be
        // this node proposing the group's own decision to the group
        return Ok(());
    }
    set.capture_insert(table, batch)
}

/// Records rows leaving a table.
#[inline]
pub fn capture_delete(ctx: &ExecutionContext, table: &TableEntry, batch: &DataBatch) -> Result<()> {
    let Some(set) = ctx.replication.as_ref() else {
        return Ok(());
    };
    if ctx.replication_apply {
        return Ok(());
    }
    let identity = identity_of(ctx, table);
    let images = identity_images(table, &identity, batch);
    if !images.keyed.is_empty() {
        set.capture_delete(table, &identity, &images.keyed)?;
    }
    if !images.imaged.is_empty() {
        set.capture_delete(table, &ReplicaIdentity::FullImage, &images.imaged)?;
    }
    Ok(())
}

/// Records rows changing, carrying both the identity of the old row and the
/// whole new one.
#[inline]
pub fn capture_update(
    ctx: &ExecutionContext,
    table: &TableEntry,
    old_batch: &DataBatch,
    new_batch: &DataBatch,
) -> Result<()> {
    let Some(set) = ctx.replication.as_ref() else {
        return Ok(());
    };
    if ctx.replication_apply {
        return Ok(());
    }
    if old_batch.num_rows != new_batch.num_rows {
        return Err(zyron_common::ZyronError::Internal(format!(
            "an update produced {} new rows from {} old ones",
            new_batch.num_rows, old_batch.num_rows
        )));
    }
    let identity = identity_of(ctx, table);
    let images = identity_images(table, &identity, old_batch);
    if !images.keyed.is_empty() {
        let rows: Vec<u32> = images.keyed_rows.iter().map(|r| *r as u32).collect();
        set.capture_update(table, &identity, &images.keyed, &new_batch.take(&rows))?;
    }
    if !images.imaged.is_empty() {
        let rows: Vec<u32> = images.imaged_rows.iter().map(|r| *r as u32).collect();
        set.capture_update(
            table,
            &ReplicaIdentity::FullImage,
            &images.imaged,
            &new_batch.take(&rows),
        )?;
    }
    Ok(())
}

/// Records a lake commit by its version file, which is the whole description
/// of what the commit did.
#[inline]
pub fn capture_lake_version(
    ctx: &ExecutionContext,
    table_id: u32,
    version: u64,
    version_file: &[u8],
) -> Result<()> {
    let Some(set) = ctx.replication.as_ref() else {
        return Ok(());
    };
    if ctx.replication_apply {
        return Ok(());
    }
    set.capture_lake_version(table_id, version, version_file)
}

/// Records how far a sequence has been drawn, so a follower elected later does
/// not hand out numbers that are already in the table.
#[inline]
pub fn capture_sequence(ctx: &ExecutionContext, sequence_id: u32, last_value: i64) -> Result<()> {
    let Some(set) = ctx.replication.as_ref() else {
        return Ok(());
    };
    if ctx.replication_apply {
        return Ok(());
    }
    set.capture_sequence(sequence_id, last_value)
}

/// The changeset a context is writing into, when it has one.
#[inline]
pub fn changeset(ctx: &ExecutionContext) -> Option<&Arc<super::changeset::TxnChangeset>> {
    if ctx.replication_apply {
        return None;
    }
    ctx.replication.as_ref()
}
