//! Reads a table's recorded changes as rows.
//!
//! One operator serves `table_changes(...)` and a read of a named change
//! stream. Both walk a window of one or more change feeds and produce the
//! source's own columns beside the change metadata. A stream read takes its
//! position lock before the first record and records the advance its commit
//! will make, which is the only difference between them.
//!
//! Records are decoded straight into column builders through the layout each
//! one was written under, so a change older than a column reads that column's
//! recorded absent value and a change that still carries a dropped column is
//! walked past. A record a narrowed feed wrote carries a flag and decodes
//! through the epoch's layout narrowed to the columns recorded at the time.
//! Nothing is materialized per record. The reader hands over the row bytes
//! borrowed from whatever holds them.
//!
//! A window over several change files is decoded across the parallel pool's
//! threads, one file per worker at a time, and the batches are handed over
//! in file order, so a read of ten million changes costs the machine's cores
//! rather than one. A window of one file, or a read on a machine with no
//! parallel capacity to spare, decodes on the operator's own thread through
//! the same row assembly

use std::collections::VecDeque;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use zyron_common::{Result, ZyronError};
use zyron_planner::logical::{
    ChangeMetadataColumn, ChangeScanSpec, ChangeScanWindow, LogicalColumn,
};

use crate::batch::{
    ColumnBuilder, DataBatch, create_builders, decode_fixed_scalar, decode_varlen_scalar,
    finalize_builders,
};
use crate::column::{Column, ScalarValue};
use crate::context::{
    CHANGE_NO_GROUP, ChangeColumnBlock, ChangeColumnSource, ChangeRecordHeads, ChangeRowRef,
    ChangeSegment, ChangeWindow, ExecutionContext, FeedKey, WindowBound,
};
use crate::epoch_decode::{
    EpochDecoder, EpochPlan, ProjectedEpochDecoder, Step, Widen, apply_widen,
};
use crate::operator::{ExecutionBatch, Operator, OperatorResult, apply_column_security};
use crate::parallel_pool::DopGrant;

/// Produces the change rows a window holds
pub struct ChangeScanOperator {
    ctx: Arc<ExecutionContext>,
    spec: ChangeScanSpec,
    output_columns: Vec<LogicalColumn>,
    batch_size: usize,
    /// Which window the scan is reading
    window_at: usize,
    /// The read of the current window in progress, which keeps its place
    /// between batches so a window costs one pass over its files
    read: Option<WindowRead>,
    /// Where each window's read starts, one per window, the record the
    /// stream's position names, so a change recorded after it at a version
    /// the position has already reached is still read. None reads from the
    /// window's lower version bound
    window_starts: Vec<Option<(u64, u64)>>,
    /// Batches decoded and not yet filtered, each with the source table its
    /// rows came from
    pending: std::collections::VecDeque<(zyron_catalog::TableId, DataBatch)>,
    /// The schema a decoded batch has before the filter-only columns are
    /// dropped: the output data columns, the filter-only ones, then the
    /// metadata
    decoded_columns: Vec<LogicalColumn>,
    /// Change files opened across every window so far
    files_opened: usize,
    /// True once the position lock is held, so a second call does not retake it
    locked: bool,
    /// True once a read inside a branch has had its windows split between
    /// the table's feed and the branch's
    routed: bool,
    /// The source's existing rows, for a stream created with SHOW INITIAL
    /// ROWS on its first read. A real scan rather than a second read path, so
    /// visibility, column security and both storage formats behave exactly as
    /// they do for any other read of the table
    initial: Option<Box<dyn Operator>>,
    finished: bool,
}

/// The columns one window decodes into, and the decoders that fill them
struct WindowDecoder {
    decoder: EpochDecoder,
    /// What a record flagged as holding the feed's column subset decodes
    /// through, built when the table's feed has ever recorded one
    projected: Option<ProjectedEpochDecoder>,
    /// The data columns in decode order, the output ones then the ones only
    /// the predicate reads
    data_columns: Vec<LogicalColumn>,
    /// Builder positions of columns this window's table does not have,
    /// which every change of it yields as NULL
    absent: Vec<usize>,
    /// The table the window reads, for what a refused record names
    table_name: String,
}

/// One window's read in progress, on the operator's thread or across the
/// pool's
enum WindowRead {
    /// The cursor keeps its place between batches and the operator's own
    /// thread decodes what it hands over
    Serial {
        cursor: Box<dyn crate::context::ChangeCursor>,
        decoder: WindowDecoder,
    },
    /// The window's files decoded by the pool's threads, handed over in
    /// file order
    Parallel(SegmentPipeline),
}

/// Turns one window's records into rows, one at a time, wherever the
/// records are read.
///
/// The one place a record becomes a row. A read on the operator's thread
/// and every worker decoding a file of a window across cores feed records
/// through this, so the two paths cannot disagree about a row
struct RowAssembler<'d> {
    decoder: &'d WindowDecoder,
    snapshot: &'d zyron_storage::txn::Snapshot,
    metadata: &'d [ChangeMetadataColumn],
    builders: Vec<ColumnBuilder>,
    metadata_builders: MetadataBuilders,
    produced: usize,
    batch_size: usize,
    source_table: i64,
    /// The first change whose transaction has not ended. Nothing at or
    /// after it is read this time, and the window closes just before it,
    /// because a change that commits later at a lower version than the
    /// position would never be consumed. A stream read resolves its
    /// windows below every such change before it starts, so this is what
    /// a `table_changes` read over a live table meets
    in_flight_at: Option<(u64, u64)>,
    /// The outcome of the transaction the last record belonged to. A
    /// commit's records are adjacent in the feed, so nearly every record's
    /// outcome is the one just resolved
    last_txn: Option<(u64, zyron_storage::txn::TxnStatus)>,
}

impl<'d> RowAssembler<'d> {
    fn new(
        decoder: &'d WindowDecoder,
        snapshot: &'d zyron_storage::txn::Snapshot,
        metadata: &'d [ChangeMetadataColumn],
        batch_size: usize,
        source_table: i64,
    ) -> Self {
        Self {
            decoder,
            snapshot,
            metadata,
            builders: create_builders(&decoder.data_columns, batch_size),
            metadata_builders: MetadataBuilders::new(metadata, batch_size),
            produced: 0,
            batch_size,
            source_table,
            in_flight_at: None,
            last_txn: None,
        }
    }

    /// Takes one record. Answers false once the batch is full or the read
    /// met a change whose transaction has not ended, which `in_flight_at`
    /// tells apart
    #[inline]
    fn take(&mut self, row: ChangeRowRef<'_>) -> Result<bool> {
        let outcome = match self.last_txn {
            Some((txn_id, outcome)) if txn_id == row.txn_id => outcome,
            _ => {
                let outcome = self.snapshot.txn_outcome(row.txn_id);
                self.last_txn = Some((row.txn_id, outcome));
                outcome
            }
        };
        match outcome {
            zyron_storage::txn::TxnStatus::Committed => {}
            // A rolled back transaction's changes were never part of the
            // table, so the feed's copy of them is skipped
            zyron_storage::txn::TxnStatus::Aborted => return Ok(true),
            zyron_storage::txn::TxnStatus::Active => {
                self.in_flight_at = Some((row.commit_version, row.change_ordinal));
                return Ok(false);
            }
        }
        // A truncate names no row, so its data columns read NULL and only
        // the metadata says what happened
        if row.change_type == zyron_common::CHANGE_TYPE_TRUNCATE || row.row_data.is_empty() {
            for builder in self.builders.iter_mut() {
                builder.push_null();
            }
        } else {
            if row.projected {
                match self.decoder.projected.as_ref() {
                    Some(projected) => {
                        projected.decode(row.schema_epoch, row.row_data, &mut self.builders)?
                    }
                    None => return Err(self.unexpected_projection(&row)),
                }
            } else {
                let mut cursor = self.decoder.decoder.cursor();
                cursor.decode(row.schema_epoch, row.row_data, None, &mut self.builders)?;
            }
            for at in &self.decoder.absent {
                self.builders[*at].push_null();
            }
        }
        self.metadata_builders.push(&row, self.source_table);
        self.produced += 1;
        Ok(self.produced < self.batch_size)
    }

    /// Reports a record flagged as holding a column subset on a table whose
    /// feed never recorded one, which no writer of this table produced
    fn unexpected_projection(&self, row: &ChangeRowRef<'_>) -> ZyronError {
        ZyronError::CdcDecoderError(format!(
            "table '{}' has a change record at version {} ordinal {} that carries a column \
             subset, and its change data feed has never recorded one, so the record cannot be \
             decoded",
            self.decoder.table_name, row.commit_version, row.change_ordinal
        ))
    }

    /// The rows taken so far as one batch, leaving the assembler ready for
    /// the next
    fn flush(&mut self) -> DataBatch {
        let builders = std::mem::replace(
            &mut self.builders,
            create_builders(&self.decoder.data_columns, self.batch_size),
        );
        let metadata = std::mem::replace(
            &mut self.metadata_builders,
            MetadataBuilders::new(self.metadata, self.batch_size),
        );
        self.produced = 0;
        Self::batch_of(builders, metadata)
    }

    /// The rows taken so far as one batch, None when there are none
    fn finish(self) -> Option<DataBatch> {
        if self.produced == 0 {
            return None;
        }
        Some(Self::batch_of(self.builders, self.metadata_builders))
    }

    fn batch_of(builders: Vec<ColumnBuilder>, metadata: MetadataBuilders) -> DataBatch {
        let mut columns: Vec<Column> = finalize_builders(builders).columns;
        columns.extend(metadata.finish());
        DataBatch::new(columns)
    }
}

/// What every worker decoding one window's files shares
struct SegmentDecodeShared {
    decoder: WindowDecoder,
    snapshot: zyron_storage::txn::Snapshot,
    metadata: Vec<ChangeMetadataColumn>,
    batch_size: usize,
    source_table: i64,
    /// The window and the position the read resumes after, which is what
    /// a record read from its header columns is judged against
    window: ChangeWindow,
    resume: Option<(u64, u64)>,
    /// The index of the first file a worker met an unfinished transaction
    /// in, `usize::MAX` while none has. A worker on a later file stops,
    /// because nothing past that change is handed over this time. A worker
    /// on an earlier file finishes, because everything before the change
    /// is
    stop_before: AtomicUsize,
    /// Files being decoded right now, which is what says whether a worker
    /// the grant allows is idle. A file decoded and not yet handed over
    /// is not counted, so a fast worker takes the next file while a slow
    /// one still holds an earlier one
    active: AtomicUsize,
}

/// Counts one file's decode as active for as long as it runs, however it
/// ends
struct ActiveDecode<'a>(&'a AtomicUsize);

impl Drop for ActiveDecode<'_> {
    fn drop(&mut self) {
        self.0.fetch_sub(1, Ordering::AcqRel);
    }
}

/// What one worker decoded out of one file
struct SegmentOutput {
    batches: Vec<DataBatch>,
    /// The first change of an unfinished transaction the file holds, at
    /// which the read stops
    in_flight_at: Option<(u64, u64)>,
}

impl SegmentDecodeShared {
    /// Whether a record is inside the window and past the resume point,
    /// judged from its header alone
    #[inline]
    fn admits(&self, version: u64, timestamp: i64, change_type: u8, ordinal: u64) -> bool {
        if version <= self.window.from_exclusive || version > self.window.to_inclusive {
            return false;
        }
        if timestamp < self.window.from_timestamp || timestamp > self.window.to_timestamp {
            return false;
        }
        if let Some(mask) = self.window.change_types
            && mask & zyron_common::change_type_bit(change_type) == 0
        {
            return false;
        }
        match self.resume {
            None => true,
            Some((at_version, at_ordinal)) => {
                version > at_version || (version == at_version && ordinal > at_ordinal)
            }
        }
    }

    /// The plan the rows of one layout group decode through
    fn plan_for(&self, epoch: u16, projected: bool) -> Result<&EpochPlan> {
        if projected {
            match self.decoder.projected.as_ref() {
                Some(decoder) => decoder.plan(epoch),
                None => Err(ZyronError::CdcDecoderError(format!(
                    "table '{}' has change records that carry a column subset, and its change \
                     data feed has never recorded one, so the records cannot be decoded",
                    self.decoder.table_name
                ))),
            }
        } else {
            self.decoder
                .decoder
                .plan(epoch)
                .ok_or_else(|| self.decoder.decoder.unknown_epoch(epoch, None))
        }
    }
}

/// Decodes one file of a window into batches, on whichever thread runs it
fn decode_segment(
    index: usize,
    mut segment: Box<dyn ChangeSegment>,
    shared: &SegmentDecodeShared,
) -> Result<SegmentOutput> {
    let _active = ActiveDecode(&shared.active);
    let _segment = zyron_common::profile::scope(zyron_common::profile::Phase::ChangeScanSegment);
    if let Some(source) = segment.columns()? {
        return decode_column_segment(index, source.as_ref(), shared);
    }
    let mut assembler = RowAssembler::new(
        &shared.decoder,
        &shared.snapshot,
        &shared.metadata,
        shared.batch_size,
        shared.source_table,
    );
    let mut batches = Vec::new();
    segment.visit(&mut |row| {
        if shared.stop_before.load(Ordering::Relaxed) <= index {
            return Ok(false);
        }
        if assembler.take(row)? {
            return Ok(true);
        }
        if assembler.in_flight_at.is_some() {
            return Ok(false);
        }
        let _finish = zyron_common::profile::scope(zyron_common::profile::Phase::ChangeScanFinish);
        batches.push(assembler.flush());
        Ok(true)
    })?;
    let in_flight_at = assembler.in_flight_at;
    if in_flight_at.is_some() {
        shared.stop_before.fetch_min(index, Ordering::Relaxed);
    }
    let _finish = zyron_common::profile::scope(zyron_common::profile::Phase::ChangeScanFinish);
    if let Some(batch) = assembler.finish() {
        batches.push(batch);
    }
    Ok(SegmentOutput {
        batches,
        in_flight_at,
    })
}

/// Rows a batch out of a column-sliced file holds at most. A file's rows
/// come out as whole column appends, so a batch this size costs no more
/// per row than a small one and each batch is one set of builders
/// allocated and one hand-off through the pipeline, which is why a file
/// of the target segment size makes a few batches rather than dozens
const SLICED_BATCH_ROWS: usize = 16 * 1024;

/// One layout group of a column-sliced file as a decode reads it, the plan
/// its rows follow and the columns decoded so far, each decoded on the
/// first run that reads it and never when no step reads it
struct GroupDecode<'p> {
    plan: &'p EpochPlan,
    columns: Vec<Option<ChangeColumnBlock>>,
}

/// Decodes one column-sliced file of a window into batches.
///
/// The records are judged from their header columns alone, and the rows
/// of the admitted ones are gathered column by column out of the blocks
/// the plan takes, a run of consecutive rows at a time, so a read of two
/// columns of forty decodes two blocks and a read of all forty appends
/// each one whole rather than walking every row
fn decode_column_segment(
    index: usize,
    source: &dyn ChangeColumnSource,
    shared: &SegmentDecodeShared,
) -> Result<SegmentOutput> {
    let heads = source.heads();
    let groups = source.groups();
    let records = heads.versions.len();

    // The records the window admits, in order, up to the first change of
    // a transaction that has not ended
    let mut admitted: Vec<u32> = Vec::with_capacity(records);
    let mut in_flight_at: Option<(u64, u64)> = None;
    let mut last_txn: Option<(u64, zyron_storage::txn::TxnStatus)> = None;
    for record in 0..records {
        if shared.stop_before.load(Ordering::Relaxed) <= index {
            break;
        }
        if !shared.admits(
            heads.versions[record],
            heads.timestamps[record],
            heads.change_types[record],
            heads.ordinals[record],
        ) {
            continue;
        }
        let txn_id = heads.txn_ids[record];
        let outcome = match last_txn {
            Some((held, outcome)) if held == txn_id => outcome,
            _ => {
                let outcome = shared.snapshot.txn_outcome(txn_id);
                last_txn = Some((txn_id, outcome));
                outcome
            }
        };
        match outcome {
            zyron_storage::txn::TxnStatus::Committed => admitted.push(record as u32),
            zyron_storage::txn::TxnStatus::Aborted => {}
            zyron_storage::txn::TxnStatus::Active => {
                in_flight_at = Some((heads.versions[record], heads.ordinals[record]));
                break;
            }
        }
    }
    if in_flight_at.is_some() {
        shared.stop_before.fetch_min(index, Ordering::Relaxed);
    }

    // Each record's row is the next of its group in record order, so the
    // position of every record's row in its group's blocks is one count
    // per group over the records
    let mut next_row = vec![0u32; groups.len()];
    let mut row_in_group = vec![0u32; records];
    for record in 0..records {
        let group = heads.group_of[record];
        if group != CHANGE_NO_GROUP {
            let g = group as usize;
            if g >= groups.len() || next_row[g] as usize >= groups[g].rows {
                return Err(ZyronError::CdcDecoderError(format!(
                    "table '{}' has a column-sliced change file whose records name more rows                      than its layout groups hold, so the file cannot be decoded",
                    shared.decoder.table_name
                )));
            }
            row_in_group[record] = next_row[g];
            next_row[g] += 1;
        }
    }

    let mut decodes: Vec<Option<GroupDecode<'_>>> = (0..groups.len()).map(|_| None).collect();
    let rows_per_batch = shared.batch_size.max(SLICED_BATCH_ROWS);
    let mut batches = Vec::with_capacity(admitted.len().div_ceil(rows_per_batch));
    let width = shared.decoder.data_columns.len();
    for chunk in admitted.chunks(rows_per_batch) {
        let mut builders = create_builders(&shared.decoder.data_columns, chunk.len());
        let mut metadata = MetadataBuilders::new(&shared.metadata, chunk.len());
        let mut at = 0usize;
        while at < chunk.len() {
            // One run of consecutive records whose rows share a layout
            let group = heads.group_of[chunk[at] as usize];
            let mut end = at + 1;
            while end < chunk.len() && heads.group_of[chunk[end] as usize] == group {
                end += 1;
            }
            let run = &chunk[at..end];
            if group == CHANGE_NO_GROUP {
                // A truncate names no row, so its data columns read NULL and
                // only the metadata says what happened
                for builder in builders.iter_mut() {
                    for _ in 0..run.len() {
                        builder.push_null();
                    }
                }
            } else {
                let g = group as usize;
                if decodes[g].is_none() {
                    let shape = &groups[g];
                    let plan = shared.plan_for(shape.epoch, shape.projected)?;
                    check_group_layout(plan, &shape.types, &shared.decoder.table_name)?;
                    decodes[g] = Some(GroupDecode {
                        plan,
                        columns: (0..shape.types.len()).map(|_| None).collect(),
                    });
                }
                let Some(decode) = decodes[g].as_mut() else {
                    continue;
                };
                let rows: Vec<u32> = run.iter().map(|r| row_in_group[*r as usize]).collect();
                append_group_run(source, g, decode, &rows, &mut builders, width)?;
                // A column of the scan this window's table does not have
                // is in no group's blocks, so its cells read NULL here the
                // way the row path fills them
                for at in &shared.decoder.absent {
                    for _ in 0..rows.len() {
                        builders[*at].push_null();
                    }
                }
            }
            for record in run {
                metadata.push_head(heads, *record as usize, shared.source_table);
            }
            at = end;
        }
        let _finish = zyron_common::profile::scope(zyron_common::profile::Phase::ChangeScanFinish);
        batches.push(RowAssembler::batch_of(builders, metadata));
    }
    Ok(SegmentOutput {
        batches,
        in_flight_at,
    })
}

/// Reports a group whose columns are not the ones the plan walks, which
/// no writer of this table produced
fn check_group_layout(
    plan: &EpochPlan,
    types: &[zyron_common::TypeId],
    table_name: &str,
) -> Result<()> {
    let steps = plan.steps();
    let matches = steps.len() == types.len()
        && steps
            .iter()
            .zip(types)
            .all(|(step, physical)| step.physical() == *physical);
    if matches {
        return Ok(());
    }
    Err(ZyronError::CdcDecoderError(format!(
        "table '{}' has a column-sliced change file whose columns do not match the layout its \
         records were written under, so the file cannot be decoded",
        table_name
    )))
}

/// Appends the rows `rows` of group `g` to the builders, column by
/// column, decoding each block the plan takes on the first run that reads
/// it. `width` is how many builders hold data columns
fn append_group_run(
    source: &dyn ChangeColumnSource,
    g: usize,
    decode: &mut GroupDecode<'_>,
    rows: &[u32],
    builders: &mut [ColumnBuilder],
    width: usize,
) -> Result<()> {
    let contiguous = rows
        .last()
        .is_some_and(|last| *last as usize - rows[0] as usize + 1 == rows.len());
    let group_rows = source.groups().get(g).map(|shape| shape.rows).unwrap_or(0);
    for (column, step) in decode.plan.steps().iter().enumerate() {
        let Step::Take {
            builder,
            physical,
            logical,
            widen,
            encrypted,
        } = step
        else {
            continue;
        };
        let b = *builder as usize;
        if b >= width {
            continue;
        }
        // A run over the whole group of a fixed column without a NULL is
        // decoded straight into the builder's storage, which is the shape
        // a read of a file's every row takes, so the values are written
        // once. Anything else decodes the block and gathers from it
        if decode.columns[column].is_none()
            && *widen == Widen::None
            && physical.fixed_size().is_some()
            && contiguous
            && rows[0] == 0
            && rows.len() == group_rows
        {
            let nulls = source.column_nulls(g, column)?;
            if nulls.iter().all(|byte| *byte == 0) {
                match builders[b].extend_fixed_filled(*physical, rows.len(), |out| {
                    source.column_values_into(g, column, out)
                }) {
                    Some(Ok(())) => continue,
                    Some(Err(e)) => return Err(e),
                    None => {}
                }
            }
        }
        if decode.columns[column].is_none() {
            decode.columns[column] = Some(source.column(g, column)?);
        }
        let Some(block) = decode.columns[column].as_ref() else {
            continue;
        };
        match block {
            ChangeColumnBlock::Fixed {
                width: cell_width,
                values,
                ..
            } => {
                if *widen == Widen::None {
                    let first = rows[0] as usize;
                    let span = first..first + rows.len();
                    if contiguous && !block.any_null(span.clone()) {
                        let bytes = &values[first * cell_width..(first + rows.len()) * cell_width];
                        if builders[b].extend_fixed_run(*physical, bytes, rows.len()) {
                            continue;
                        }
                    }
                    let cells = rows.iter().map(|r| {
                        let r = *r as usize;
                        (!block.is_null(r)).then(|| block.cell(r))
                    });
                    if builders[b].extend_fixed(*physical, cells) {
                        continue;
                    }
                }
                // A pairing the typed appends do not carry, or a value that
                // widens on the way in, goes through the scalar it decodes to
                for r in rows {
                    let r = *r as usize;
                    if block.is_null(r) {
                        builders[b].push_null();
                    } else {
                        let scalar = decode_fixed_scalar(*physical, block.cell(r));
                        builders[b].push_owned(apply_widen(*widen, scalar));
                    }
                }
            }
            ChangeColumnBlock::Varlen { .. } => {
                for r in rows {
                    let r = *r as usize;
                    if block.is_null(r) {
                        builders[b].push_null();
                    } else {
                        let cell = block.cell(r);
                        let scalar = if *encrypted {
                            ScalarValue::Binary(cell.to_vec())
                        } else {
                            decode_varlen_scalar(*logical, cell)
                        };
                        builders[b].push_owned(scalar);
                    }
                }
            }
        }
    }
    for (builder, value) in decode.plan.absent() {
        let b = *builder as usize;
        if b >= width {
            continue;
        }
        for _ in rows {
            builders[b].push(value);
        }
    }
    Ok(())
}

/// How many files past the ones being decoded may stand decoded and not
/// yet handed over, per worker the grant allows. A file whose worker ran
/// slow holds the head while faster workers finish the files behind it,
/// and this is how far they may run ahead before they wait for it, which
/// bounds the memory those files hold
const DECODED_FILES_AHEAD_PER_WORKER: usize = 3;

/// A window's files decoded across the pool's threads and handed over in
/// file order.
///
/// As many files are decoded at once as the pool granted workers, the
/// oldest is awaited first, and the files decoded ahead of the head are
/// bounded, so the batches come out in the order the records were
/// recorded and the memory held is bounded by the grant
struct SegmentPipeline {
    shared: Arc<SegmentDecodeShared>,
    queued: VecDeque<Box<dyn ChangeSegment>>,
    /// Files being decoded, oldest first, with the index each one has in
    /// the window
    running: VecDeque<crate::parallel_pool::JoinHandle<Result<SegmentOutput>>>,
    /// Batches of the file at the head not yet handed over
    ready: VecDeque<DataBatch>,
    next_index: usize,
    width: usize,
    /// Where the read met a change whose transaction has not ended, once
    /// every batch before it has been handed over
    stopped_at: Option<(u64, u64)>,
    _grant: DopGrant,
}

impl SegmentPipeline {
    fn new(
        segments: Vec<Box<dyn ChangeSegment>>,
        shared: SegmentDecodeShared,
        grant: DopGrant,
    ) -> Self {
        Self {
            shared: Arc::new(shared),
            queued: segments.into(),
            running: VecDeque::new(),
            ready: VecDeque::new(),
            next_index: 0,
            width: grant.workers().max(1),
            stopped_at: None,
            _grant: grant,
        }
    }

    /// The next batch in record order, None once the window is exhausted
    /// or stopped at an unfinished transaction
    async fn next_batch(&mut self) -> Result<Option<DataBatch>> {
        loop {
            if self.stopped_at.is_none() {
                // Every worker the grant allows is kept busy, so a file is
                // handed out as soon as one finishes rather than once the
                // finished one's batches have all been taken, up to the
                // bound on files decoded ahead of the head
                let ahead = self
                    .width
                    .saturating_mul(DECODED_FILES_AHEAD_PER_WORKER + 1);
                while self.shared.active.load(Ordering::Acquire) < self.width
                    && self.running.len() < ahead
                {
                    let Some(segment) = self.queued.pop_front() else {
                        break;
                    };
                    self.shared.active.fetch_add(1, Ordering::AcqRel);
                    let index = self.next_index;
                    self.next_index += 1;
                    let shared = Arc::clone(&self.shared);
                    // The shared pool, not the current runtime. On the
                    // serving path the current runtime drives one
                    // connection, so spawning there would put every worker
                    // on the one thread
                    self.running
                        .push_back(crate::parallel_pool::spawn(async move {
                            decode_segment(index, segment, &shared)
                        }));
                }
            }
            if let Some(batch) = self.ready.pop_front() {
                return Ok(Some(batch));
            }
            if self.stopped_at.is_some() {
                return Ok(None);
            }
            let Some(head) = self.running.pop_front() else {
                return Ok(None);
            };
            let awaited =
                zyron_common::profile::scope(zyron_common::profile::Phase::ChangeScanAwait);
            let output = head.await.map_err(|e| {
                ZyronError::ExecutionError(format!("a change file's decode ended early: {e}"))
            })??;
            drop(awaited);
            self.ready.extend(output.batches);
            if let Some(at) = output.in_flight_at {
                // Nothing past the change is handed over this time, so the
                // files behind it are dropped undecoded, and the workers on
                // them see the stop and finish early
                self.stopped_at = Some(at);
                self.queued.clear();
                self.running.clear();
            }
        }
    }
}

impl ChangeScanOperator {
    pub fn new(
        ctx: Arc<ExecutionContext>,
        spec: ChangeScanSpec,
        output_columns: Vec<LogicalColumn>,
        initial: Option<Box<dyn Operator>>,
    ) -> Self {
        let batch_size = ctx.batch_size.max(1);
        let metadata_start = output_columns.len().saturating_sub(spec.metadata.len());
        let decoded_columns = spec
            .data_columns
            .iter()
            .chain(spec.filter_columns.iter())
            .cloned()
            .chain(output_columns[metadata_start..].iter().cloned())
            .collect();
        Self {
            ctx,
            spec,
            output_columns,
            batch_size,
            window_at: 0,
            read: None,
            window_starts: Vec::new(),
            pending: std::collections::VecDeque::new(),
            decoded_columns,
            files_opened: 0,
            locked: false,
            routed: false,
            initial,
            finished: false,
        }
    }

    /// The data columns a record decodes into, output ones first
    fn decoded_data_columns(&self) -> Vec<LogicalColumn> {
        self.spec
            .data_columns
            .iter()
            .chain(self.spec.filter_columns.iter())
            .cloned()
            .collect()
    }

    /// Change files this scan opened, for the query profile
    pub fn files_opened(&self) -> usize {
        self.files_opened
    }

    /// The reader this scan reads through
    fn reader(&self) -> Result<&Arc<dyn crate::context::ChangeFeedReader>> {
        self.ctx.change_feed.as_ref().ok_or_else(|| {
            ZyronError::ExecutionError(
                "this node records no change data feeds, so there are no changes to read"
                    .to_string(),
            )
        })
    }

    /// Takes the position lock a transactional consume holds until it ends,
    /// then resolves the windows against what stands at that instant.
    ///
    /// A peek skips the lock, which is what lets a dashboard read what is
    /// pending while a consumer holds the position. Every stream read
    /// re-resolves after the lock, because a consumer that waited for another
    /// consumer's commit must read from where that commit left the position,
    /// and the plan was made before the wait
    async fn take_position_lock(&mut self) -> Result<()> {
        if self.locked {
            return Ok(());
        }
        self.locked = true;
        let Some(stream) = self.spec.stream.clone() else {
            return Ok(());
        };
        if !stream.peek {
            if let Some(locks) = &self.ctx.stream_position_locks {
                locks
                    .lock_wait(self.ctx.txn_id, stream.stream_id as u64)
                    .await?;
            }
        }
        self.resolve_stream_windows(stream.stream_id)
    }

    /// Points each window at the stream's current position and the boundary
    /// its sources have reached.
    ///
    /// The sources are read at one instant, so the boundary is one version
    /// for every heap source, the newest change any of them holds, held
    /// back to just below the first change of any transaction that has not
    /// ended. A transaction that touched two sources is then wholly inside
    /// the windows or wholly outside them, and no window reaches a change
    /// that may yet commit or roll back. A lake source's versions are its
    /// own commit sequence, so its window ends at its own newest commit,
    /// held back the same way below the first commit of a transaction that
    /// has not ended as of this read's snapshot.
    ///
    /// Each window starts at the record the position's consumed count
    /// names rather than at its version, so a change recorded after the
    /// last read at a version the position had already reached is read
    /// rather than passed over
    fn resolve_stream_windows(&mut self, stream_id: u32) -> Result<()> {
        let entry = self
            .ctx
            .catalog
            .get_change_stream_by_id(stream_id)
            .ok_or_else(|| {
                ZyronError::ExecutionError(format!(
                    "change stream {stream_id} was dropped while a read of it was planned"
                ))
            })?;
        let reader = Arc::clone(self.reader()?);
        // A stream created on a branch reads the branch's feeds, whatever
        // branch the session reading it is on
        let branch = entry.branch;
        let keys: Vec<FeedKey> = self
            .spec
            .windows
            .iter()
            .map(|w| FeedKey {
                table_id: w.table_id.0,
                branch,
            })
            .collect();
        let snapshot = &self.ctx.snapshot;
        let outcome = |txn_id: u64| snapshot.txn_outcome(txn_id);
        let boundaries = reader.boundaries(&keys, &outcome)?;
        let heap_latest = boundaries
            .iter()
            .filter(|b| !b.lake)
            .map(|b| b.latest)
            .max()
            .unwrap_or(0);
        let cut = boundaries
            .iter()
            .filter(|b| !b.lake)
            .filter_map(|b| b.first_open)
            .min();
        let heap_limit = match cut {
            Some(first_open) => heap_latest.min(first_open.saturating_sub(1)),
            None => heap_latest,
        };
        // A bounded read ends at the one version the reader names across
        // every heap source, so a transaction that touched two of them is
        // wholly inside the read or wholly after it. A lake source's
        // versions are its own commit sequence rather than the node's
        // change clock, so it is bounded on its own below, and the windows
        // are then aligned so a transaction with commits in several
        // sources is whole in all of them
        let max_rows = self.spec.stream.as_ref().and_then(|s| s.max_rows);
        let is_lake = |table_id: u32| {
            boundaries
                .iter()
                .find(|b| b.source.table_id == table_id)
                .is_some_and(|b| b.lake)
        };
        let bound = match max_rows {
            Some(max_rows) => {
                let sources: Vec<(FeedKey, u64, u64)> = self
                    .spec
                    .windows
                    .iter()
                    .filter(|w| !is_lake(w.table_id.0))
                    .map(|w| {
                        let table_id = w.table_id.0;
                        (
                            FeedKey { table_id, branch },
                            entry.position_of(table_id),
                            entry.consumed_of(table_id),
                        )
                    })
                    .collect();
                if sources.is_empty() {
                    None
                } else {
                    reader.bounded_cut(&sources, max_rows)?
                }
            }
            None => None,
        };
        let shared = match bound {
            Some(bound) => heap_limit.min(bound),
            None => heap_limit,
        };
        let mut bounds = Vec::with_capacity(self.spec.windows.len());
        for window in self.spec.windows.iter() {
            let table_id = window.table_id.0;
            let key = FeedKey { table_id, branch };
            let boundary = boundaries.iter().find(|b| b.source == key);
            let position = entry.position_of(table_id);
            let consumed = entry.consumed_of(table_id);
            let lake = boundary.map(|b| b.lake).unwrap_or(false);
            // A lake source's read ends at its own newest commit, held
            // below the first commit of a transaction still open as of this
            // read, and bounded by its own record count
            let (to, limit) = if lake {
                let latest = boundary.map(|b| b.latest).unwrap_or(0);
                let latest = match boundary.and_then(|b| b.first_open) {
                    Some(first_open) => latest.min(first_open.saturating_sub(1)),
                    None => latest,
                };
                let own_bound = match max_rows {
                    Some(max_rows) => reader.bounded_cut(&[(key, position, consumed)], max_rows)?,
                    None => None,
                };
                let to = match own_bound {
                    Some(bound) => latest.min(bound),
                    None => latest,
                };
                (to.max(position), latest.max(position))
            } else {
                (shared.max(position), heap_limit.max(position))
            };
            bounds.push(WindowBound {
                source: key,
                from_exclusive: position,
                to_inclusive: to,
                limit,
            });
        }
        let aligned = reader.align_windows(&bounds)?;
        let mut starts = Vec::with_capacity(self.spec.windows.len());
        for (window, to) in self.spec.windows.iter_mut().zip(aligned) {
            let table_id = window.table_id.0;
            let key = FeedKey { table_id, branch };
            window.branch = branch;
            let position = entry.position_of(table_id);
            let consumed = entry.consumed_of(table_id);
            let start = reader.cursor_at_count(key, consumed)?;
            window.from_exclusive = match start {
                Some((version, _)) => version.saturating_sub(1),
                None => position,
            };
            window.to_inclusive = to;
            window.consumed_to = reader.records_at_or_below(key, to)?.max(consumed);
            starts.push(start);
        }
        // A stream this transaction has already read is read again at the
        // window that read recorded, so every read of it inside one
        // transaction yields the same changes whatever has landed since
        let held = self
            .ctx
            .pending_stream_advances
            .lock()
            .iter()
            .find(|held| held.stream_id == stream_id)
            .map(|held| held.positions.clone());
        if let Some(positions) = held {
            for window in self.spec.windows.iter_mut() {
                let table_id = window.table_id.0;
                if let Some((_, to, consumed_to)) =
                    positions.iter().find(|(table, _, _)| *table == table_id)
                {
                    window.to_inclusive = *to;
                    window.consumed_to = *consumed_to;
                }
            }
        }
        self.window_starts = starts;
        // The first read of a stream created with SHOW INITIAL ROWS yields
        // the existing rows. Once that read commits, the feed is what the
        // stream reads from, and a read planned before that commit and run
        // after it drops the seed scan it was built with
        self.spec.initial_rows = self.spec.initial_rows && entry.initial_rows_pending;
        if !self.spec.initial_rows {
            self.initial = None;
        }
        Ok(())
    }

    /// Routes a `table_changes` read made inside a branch.
    ///
    /// The branch's history is the table's up to the version the branch was
    /// taken at and the branch's own after it, so a window over both is
    /// split there. The part at or below the branch point reads the table's
    /// feed and the part above it reads the branch's. A bound written as
    /// LATEST reaches the branch's newest change. A branch that has recorded
    /// nothing on the table reads the table's feed as it stands
    fn route_branch(&mut self) -> Result<()> {
        if self.spec.stream.is_some() {
            return Ok(());
        }
        let Some(branch) = self.ctx.active_branch_id else {
            return Ok(());
        };
        let reader = Arc::clone(self.reader()?);
        let mut routed = Vec::with_capacity(self.spec.windows.len() * 2);
        for window in self.spec.windows.drain(..) {
            if window.branch.is_some() {
                routed.push(window);
                continue;
            }
            let table_id = window.table_id.0;
            let Some(point) = reader.branch_point(table_id, branch)? else {
                routed.push(window);
                continue;
            };
            let key = FeedKey {
                table_id,
                branch: Some(branch),
            };
            let to = if window.open_ended {
                reader.latest_version(key)?.max(window.to_inclusive)
            } else {
                window.to_inclusive
            };
            if window.from_exclusive < point {
                let mut parent = window.clone();
                parent.to_inclusive = to.min(point);
                routed.push(parent);
            }
            if to > point {
                let mut own = window.clone();
                own.branch = Some(branch);
                own.from_exclusive = window.from_exclusive.max(point);
                own.to_inclusive = to;
                routed.push(own);
            }
        }
        self.spec.windows = routed;
        Ok(())
    }

    /// Stops a window short of a change whose transaction has not ended,
    /// so the position never moves past a change that may yet commit or
    /// roll back.
    ///
    /// The change is named by version and ordinal. Everything before it in
    /// the feed has been handed over, so the position lands just before it:
    /// on the previous version when it is the first change of its version,
    /// and on its own version otherwise
    fn truncate_window(&mut self, at: usize, version: u64, ordinal: u64) -> Result<()> {
        let reader = Arc::clone(self.reader()?);
        let Some(window) = self.spec.windows.get_mut(at) else {
            return Ok(());
        };
        let key = FeedKey {
            table_id: window.table_id.0,
            branch: window.branch,
        };
        let consumed_to = reader.records_before(key, version, ordinal)?;
        let to = if ordinal == 0 {
            version.saturating_sub(1)
        } else {
            version
        };
        if consumed_to < window.consumed_to || to < window.to_inclusive {
            window.to_inclusive = to.min(window.to_inclusive);
            window.consumed_to = consumed_to.min(window.consumed_to);
        }
        Ok(())
    }

    /// Records the advance this statement's commit will make
    fn record_advance(&self) {
        let Some(stream) = &self.spec.stream else {
            return;
        };
        if stream.peek {
            return;
        }
        let positions = self
            .spec
            .windows
            .iter()
            .map(|w| (w.table_id.0, w.to_inclusive, w.consumed_to))
            .collect();
        let advance = crate::context::PendingStreamAdvance {
            stream_id: stream.stream_id,
            positions,
        };
        let mut pending = self.ctx.pending_stream_advances.lock();
        // One advance per stream per statement, so reading a stream twice in
        // one transaction records the same move once
        if let Some(held) = pending
            .iter_mut()
            .find(|held| held.stream_id == advance.stream_id)
        {
            *held = advance;
        } else {
            pending.push(advance);
        }
    }

    /// Builds the decoder one window's records read through
    fn window_decoder(&self, window: &ChangeScanWindow) -> Result<WindowDecoder> {
        let table = self.ctx.get_table_entry(window.table_id)?;
        // The scan addresses its columns by the first source's ids, and each
        // source's changes decode through that source's own ids for the
        // same names
        let data_columns = self.decoded_data_columns();
        let own = zyron_planner::change_scan::columns_on(&table, &data_columns);
        let mut ids: Vec<zyron_catalog::ColumnId> = Vec::with_capacity(own.len());
        let mut absent = Vec::new();
        for (at, column) in own.iter().enumerate() {
            match column {
                Some(column) => ids.push(column.column_id),
                None => {
                    // An id no column of the table has, so the decoder walks
                    // past nothing for it and the scan fills it itself
                    ids.push(zyron_catalog::ColumnId(u16::MAX));
                    absent.push(at);
                }
            }
        }
        let decoder = if self.spec.as_of_change {
            EpochDecoder::new_with_dropped(&table, &ids)
        } else {
            EpochDecoder::new(&table, &ids)
        };
        let projected = (!table.cdf.column_sets.is_empty())
            .then(|| ProjectedEpochDecoder::new(&table, &ids, self.spec.as_of_change));
        Ok(WindowDecoder {
            decoder,
            projected,
            data_columns,
            absent,
            table_name: table.name.clone(),
        })
    }

    /// Opens one window's read.
    ///
    /// A window over several files is decoded across the pool's threads
    /// when the machine has parallel capacity to spare, one file per worker
    /// at a time. Otherwise the cursor is read on this thread
    fn open_window(&mut self, window: &ChangeScanWindow) -> Result<WindowRead> {
        let resume = self.window_starts.get(self.window_at).copied().flatten();
        let mut opened = self
            .reader()?
            .open_window(&Self::read_window_of(window), resume)?;
        // A window's files are opened once per pass over it, which is what
        // the read costs whatever the batch size, and its decoder is built
        // once with it
        let stats = opened.stats();
        self.files_opened += stats.files_opened;
        let decoder = self.window_decoder(window)?;
        if stats.files_opened >= 2 {
            let grant = crate::parallel_pool::reserve(stats.files_opened);
            if grant.is_parallel() {
                let segments = opened.take_segments();
                let shared = SegmentDecodeShared {
                    decoder,
                    snapshot: self.ctx.snapshot.clone(),
                    metadata: self.spec.metadata.clone(),
                    batch_size: self.batch_size,
                    source_table: window.table_id.0 as i64,
                    window: Self::read_window_of(window),
                    resume,
                    stop_before: AtomicUsize::new(usize::MAX),
                    active: AtomicUsize::new(0),
                };
                return Ok(WindowRead::Parallel(SegmentPipeline::new(
                    segments, shared, grant,
                )));
            }
        }
        Ok(WindowRead::Serial {
            cursor: opened,
            decoder,
        })
    }

    /// Turns one window into the shape the reader takes
    fn read_window_of(window: &ChangeScanWindow) -> ChangeWindow {
        ChangeWindow {
            table_id: window.table_id.0,
            branch: window.branch,
            from_exclusive: window.from_exclusive,
            to_inclusive: window.to_inclusive,
            from_timestamp: window.from_timestamp,
            to_timestamp: window.to_timestamp,
            change_types: window.change_types,
        }
    }

    /// Reads up to one batch of changes out of the current window.
    ///
    /// Answers false when the window is exhausted, which is what moves the
    /// scan on to the next source
    async fn fill_one_batch(&mut self) -> Result<bool> {
        let Some(window) = self.spec.windows.get(self.window_at).cloned() else {
            return Ok(false);
        };
        if self.read.is_none() {
            self.read = Some(self.open_window(&window)?);
        }
        let batch_size = self.batch_size;
        let source_table = window.table_id.0 as i64;
        let snapshot = &self.ctx.snapshot;
        let metadata = self.spec.metadata.as_slice();
        // What the window's read produced this time, whether it can produce
        // more, and where it met an unfinished transaction
        let (batch, more, in_flight_at) = match self.read.as_mut() {
            Some(WindowRead::Serial { cursor, decoder }) => {
                let mut assembler =
                    RowAssembler::new(decoder, snapshot, metadata, batch_size, source_table);
                let more = cursor.next(&mut |row| assembler.take(row))?;
                let in_flight_at = assembler.in_flight_at;
                (assembler.finish(), more, in_flight_at)
            }
            Some(WindowRead::Parallel(pipeline)) => match pipeline.next_batch().await? {
                Some(batch) => (Some(batch), true, None),
                None => (None, false, pipeline.stopped_at),
            },
            None => return Ok(false),
        };
        if let Some((version, ordinal)) = in_flight_at {
            let at = self.window_at;
            self.truncate_window(at, version, ordinal)?;
            self.record_advance();
        }

        let Some(batch) = batch else {
            self.window_at += 1;
            self.read = None;
            return Ok(self.window_at < self.spec.windows.len());
        };

        if !more || in_flight_at.is_some() {
            // The window ran out, or closed at an unfinished transaction, so
            // the next call reads the next source rather than asking this
            // one again
            self.window_at += 1;
            self.read = None;
        }

        self.pending.push_back((window.table_id, batch));
        Ok(true)
    }

    /// Turns one batch of the source's existing rows into insert-shaped
    /// changes, at the version the stream was created against
    fn shape_initial(&self, batch: DataBatch) -> DataBatch {
        let rows = batch.num_rows;
        let version = self
            .spec
            .windows
            .first()
            .map(|w| w.to_inclusive)
            .unwrap_or(0);
        let source_table = self
            .spec
            .windows
            .first()
            .map(|w| w.table_id.0 as i64)
            .unwrap_or(0);
        let mut columns = batch.columns;
        for column in &self.spec.metadata {
            let mut builder = ColumnBuilder::new(column.type_id(), rows);
            for _ in 0..rows {
                let value = match column {
                    ChangeMetadataColumn::ChangeType => ScalarValue::Utf8(
                        zyron_common::change_type_label(zyron_common::CHANGE_TYPE_INSERT)
                            .unwrap_or("insert")
                            .to_string(),
                    ),
                    ChangeMetadataColumn::CommitVersion => ScalarValue::Int64(version as i64),
                    // Every seeded row arrives at once, so they share the
                    // creation version's own instant rather than each
                    // carrying a moment nothing recorded
                    ChangeMetadataColumn::CommitTimestamp => ScalarValue::Int64(0),
                    ChangeMetadataColumn::CommitTxnId => ScalarValue::Int64(self.ctx.txn_id as i64),
                    ChangeMetadataColumn::ChangeOrdinal => ScalarValue::Int64(0),
                    ChangeMetadataColumn::SourceTable => ScalarValue::Int64(source_table),
                };
                builder.push_owned(value);
            }
            columns.push(builder.finish());
        }
        DataBatch::new(columns)
    }

    /// Applies the scan's own predicate to a decoded batch and shapes what
    /// survives into the output.
    ///
    /// The predicate runs over the real values, with the filter-only columns
    /// still present. Those are dropped afterward and column security is
    /// applied to the rows that leave, so masking never changes which rows
    /// the stream yields
    fn filter(&self, source: zyron_catalog::TableId, batch: DataBatch) -> Result<DataBatch> {
        let filtered = match &self.spec.predicate {
            Some(predicate) if batch.num_rows > 0 => {
                let mask_column = crate::expr::evaluate(
                    predicate,
                    &batch,
                    &self.decoded_columns,
                    &self.ctx.params,
                )?;
                let mask = crate::compute::column_to_mask(&mask_column);
                if mask.iter().all(|keep| *keep) {
                    batch
                } else {
                    batch.filter(&mask)
                }
            }
            _ => batch,
        };
        let shaped = if self.spec.filter_columns.is_empty() {
            filtered
        } else {
            let data = self.spec.data_columns.len();
            let extra = self.spec.filter_columns.len();
            let mut columns = filtered.columns;
            columns.drain(data..data + extra);
            DataBatch::new(columns)
        };
        Ok(apply_column_security(
            &self.ctx,
            source.0,
            &self.output_columns,
            shaped,
        ))
    }
}

/// One builder per metadata column, filled as records arrive
struct MetadataBuilders {
    columns: Vec<ChangeMetadataColumn>,
    builders: Vec<ColumnBuilder>,
}

impl MetadataBuilders {
    fn new(columns: &[ChangeMetadataColumn], capacity: usize) -> Self {
        let builders = columns
            .iter()
            .map(|column| ColumnBuilder::new(column.type_id(), capacity))
            .collect();
        Self {
            columns: columns.to_vec(),
            builders,
        }
    }

    #[inline]
    fn push(&mut self, row: &ChangeRowRef<'_>, source_table: i64) {
        for (i, column) in self.columns.iter().enumerate() {
            let value = match column {
                ChangeMetadataColumn::ChangeType => ScalarValue::Utf8(
                    zyron_common::change_type_label(row.change_type)
                        .unwrap_or("unknown")
                        .to_string(),
                ),
                ChangeMetadataColumn::CommitVersion => {
                    ScalarValue::Int64(row.commit_version as i64)
                }
                ChangeMetadataColumn::CommitTimestamp => ScalarValue::Int64(row.commit_timestamp),
                ChangeMetadataColumn::CommitTxnId => ScalarValue::Int64(row.txn_id as i64),
                ChangeMetadataColumn::ChangeOrdinal => {
                    ScalarValue::Int64(row.change_ordinal as i64)
                }
                ChangeMetadataColumn::SourceTable => ScalarValue::Int64(source_table),
            };
            self.builders[i].push_owned(value);
        }
    }

    /// One record out of a column-sliced file's header columns
    #[inline]
    fn push_head(&mut self, heads: &ChangeRecordHeads, record: usize, source_table: i64) {
        for (i, column) in self.columns.iter().enumerate() {
            let builder = &mut self.builders[i];
            match column {
                ChangeMetadataColumn::ChangeType => builder.push_owned(ScalarValue::Utf8(
                    zyron_common::change_type_label(heads.change_types[record])
                        .unwrap_or("unknown")
                        .to_string(),
                )),
                ChangeMetadataColumn::CommitVersion => {
                    push_i64(builder, heads.versions[record] as i64)
                }
                ChangeMetadataColumn::CommitTimestamp => {
                    push_i64(builder, heads.timestamps[record])
                }
                ChangeMetadataColumn::CommitTxnId => {
                    push_i64(builder, heads.txn_ids[record] as i64)
                }
                ChangeMetadataColumn::ChangeOrdinal => {
                    push_i64(builder, heads.ordinals[record] as i64)
                }
                ChangeMetadataColumn::SourceTable => push_i64(builder, source_table),
            }
        }
    }

    fn finish(self) -> Vec<Column> {
        self.builders
            .into_iter()
            .map(|builder| builder.finish())
            .collect()
    }
}

/// Appends one whole number into a builder of a 64 bit integer column
/// without building a scalar for it
#[inline]
fn push_i64(builder: &mut ColumnBuilder, value: i64) {
    if !builder.push_fixed(zyron_common::TypeId::Int64, &value.to_le_bytes()) {
        builder.push_owned(ScalarValue::Int64(value));
    }
}

impl Operator for ChangeScanOperator {
    fn next(&mut self) -> OperatorResult<'_> {
        Box::pin(async move {
            if !self.routed {
                self.route_branch()?;
                self.routed = true;
            }
            self.take_position_lock().await?;
            // The advance is recorded once the scan is under way rather than
            // when it ends, so a statement that reads part of a stream and
            // then fails still leaves the position where it was. The record
            // is discarded with the transaction
            self.record_advance();

            let mut timer = crate::calibrate::BatchTimer::start(
                zyron_pressure::capability::OperatorKind::SeqScan,
            );
            loop {
                if let Some((source, batch)) = self.pending.pop_front() {
                    let filtered = self.filter(source, batch)?;
                    if filtered.num_rows == 0 {
                        continue;
                    }
                    timer.rows(filtered.num_rows as u64);
                    return Ok(Some(ExecutionBatch::new(filtered)));
                }
                if self.finished {
                    return Ok(None);
                }
                self.ctx.check_cancelled()?;
                // A stream created with SHOW INITIAL ROWS yields the source's
                // existing rows and nothing else on its first read. The feed
                // holds no record of them, which is what enabling a feed on a
                // populated table means, so a real scan is what produces them
                if let Some(initial) = self.initial.as_mut() {
                    match initial.next().await? {
                        Some(batch) => {
                            let source = self
                                .spec
                                .windows
                                .first()
                                .map(|w| w.table_id)
                                .unwrap_or(zyron_catalog::TableId(0));
                            let shaped = self.shape_initial(batch.batch);
                            self.pending.push_back((source, shaped));
                            continue;
                        }
                        None => {
                            self.initial = None;
                            self.finished = true;
                            continue;
                        }
                    }
                }
                if !self.fill_one_batch().await? {
                    self.finished = true;
                }
            }
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_metadata_builders_fill_one_value_per_column() {
        let columns = ChangeMetadataColumn::multi_table();
        let mut builders = MetadataBuilders::new(columns, 4);
        let row = ChangeRowRef {
            table_id: 9,
            change_type: zyron_common::CHANGE_TYPE_UPDATE_POSTIMAGE,
            commit_version: 42,
            commit_timestamp: 1_700_000_000_000_000,
            txn_id: 7,
            change_ordinal: 1,
            schema_epoch: 1,
            projected: false,
            row_data: &[1, 2, 3],
        };
        builders.push(&row, 9);
        let produced = builders.finish();
        assert_eq!(produced.len(), columns.len());
        for column in &produced {
            assert_eq!(column.len(), 1);
        }
        assert_eq!(
            produced[0].get_scalar(0),
            ScalarValue::Utf8("update_postimage".to_string())
        );
        assert_eq!(produced[1].get_scalar(0), ScalarValue::Int64(42));
        assert_eq!(produced[4].get_scalar(0), ScalarValue::Int64(1));
        assert_eq!(produced[5].get_scalar(0), ScalarValue::Int64(9));
    }
}
