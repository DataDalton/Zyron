//! The partitioned hash join: what a hash join becomes when neither side
//! fits in the query's memory budget.
//!
//! A hash join needs one whole side resident, because a probe row can match
//! any build row. When the smaller side is still too large for that, the way
//! out is to stop treating the join as one problem. Rows that hash to
//! different partitions can never match each other, so splitting both inputs
//! by the same hash of their join keys turns one join that does not fit into
//! sixteen that do, and their concatenated output is the same answer in a
//! different order.
//!
//! ## Reflow
//!
//! A partition is only useful if its build side fits. When one does not, it
//! is read back and split again under a different seed, and its children take
//! its place in the queue. Two things stop that from running forever: a depth
//! limit, and a check for whether the split moved any rows at all. A partition
//! whose rows stay together under two independent hashes holds one key value,
//! and no third hash will separate it.
//!
//! ## The partition that cannot be split
//!
//! That partition is passed over in blocks: as much of the build side as fits,
//! joined against the whole probe side, then the next block. Build-side outer
//! rows are correct per block, because a block sees every probe row. Probe-side
//! outer rows are not, because a probe row that matched block three must not be
//! emitted as unmatched by block one, so the blocks record which probe rows
//! matched and a final pass emits the ones that never did.
//!
//! ## Why partitions run one at a time
//!
//! Each partition is independent and could run concurrently. It deliberately
//! does not: a join spills because the node is short of memory, and running
//! sixteen partitions at once would hold sixteen build tables instead of one,
//! which is the memory the partitioning was for.

use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use zyron_common::{Result, TypeId, ZyronError};
use zyron_parser::ast::JoinType;
use zyron_planner::binder::BoundExpr;
use zyron_planner::logical::LogicalColumn;

use crate::batch::{BATCH_SIZE, DataBatch};
use crate::column::{Column, ColumnData, NullBitmap};
use crate::operator::join::{HashJoinOperator, bucket_of, hash_keys, join_key_common_type};
use crate::operator::{ExecutionBatch, Operator, OperatorResult};
use crate::spill::{SpillDirectory, SpillReader, SpillStats, SpillWriter};

/// Partitions each side splits into at one level.
///
/// Sixteen rather than a number derived from the input size, because the input
/// size is not known when the split starts: partitioning begins the moment the
/// budget is passed, which is before the rest of the input has been read.
/// Reflow covers whatever sixteen was not enough for, and each level multiplies
/// the count, so two levels is two hundred and fifty six partitions.
pub(crate) const GRACE_PARTITIONS: usize = 16;

/// Levels of re-partitioning before a partition is passed over in blocks.
const MAX_REFLOW_DEPTH: u32 = 4;

// ---------------------------------------------------------------------------
// Probe match bits
// ---------------------------------------------------------------------------

/// One bit per probe row, set when that row matched something.
///
/// Only a blocked pass needs this. Everywhere else a probe row's fate is
/// decided by the one build table it is probed against, and the join emits it
/// immediately.
pub struct ProbeMatchBits {
    words: Vec<AtomicU64>,
    rows: u64,
}

impl ProbeMatchBits {
    fn new(rows: u64) -> Self {
        Self {
            words: (0..rows.div_ceil(64)).map(|_| AtomicU64::new(0)).collect(),
            rows,
        }
    }

    /// Records that a probe row matched. Called from the probe loop, so it
    /// stays a single relaxed or.
    #[inline]
    pub fn set(&self, row: u64) {
        if row >= self.rows {
            return;
        }
        self.words[(row / 64) as usize].fetch_or(1u64 << (row % 64), Ordering::Relaxed);
    }

    #[inline]
    fn is_set(&self, row: u64) -> bool {
        if row >= self.rows {
            return false;
        }
        self.words[(row / 64) as usize].load(Ordering::Relaxed) & (1u64 << (row % 64)) != 0
    }
}

// ---------------------------------------------------------------------------
// Sources
// ---------------------------------------------------------------------------

/// Yields batches held in memory, then stops.
pub(crate) struct BatchListSource {
    batches: Vec<DataBatch>,
    idx: usize,
}

impl BatchListSource {
    pub(crate) fn new(batches: Vec<DataBatch>) -> Self {
        Self { batches, idx: 0 }
    }
}

impl Operator for BatchListSource {
    fn next(&mut self) -> OperatorResult<'_> {
        Box::pin(async move {
            if self.idx >= self.batches.len() {
                return Ok(None);
            }
            let batch = std::mem::replace(&mut self.batches[self.idx], DataBatch::new(Vec::new()));
            self.idx += 1;
            Ok(Some(ExecutionBatch::new(batch)))
        })
    }
}

/// Reads one spill file back as an operator.
///
/// The reader is shared because a blocked pass reads the same probe file once
/// per build block, handing it to a new join each time and rewinding it in
/// between. Nothing holds the lock across an await.
pub(crate) struct SpillSource {
    reader: Arc<std::sync::Mutex<SpillReader>>,
}

impl SpillSource {
    pub(crate) fn new(reader: Arc<std::sync::Mutex<SpillReader>>) -> Self {
        Self { reader }
    }
}

impl Operator for SpillSource {
    fn next(&mut self) -> OperatorResult<'_> {
        Box::pin(async move {
            let mut guard = self
                .reader
                .lock()
                .map_err(|_| ZyronError::ExecutionError("spill reader lock poisoned".into()))?;
            match guard.read_batch()? {
                Some(batch) => Ok(Some(ExecutionBatch::new(batch))),
                None => Ok(None),
            }
        })
    }
}

/// Yields batches already read, then everything the operator behind them has
/// left.
///
/// A join that read part of an input before deciding it could not hold it
/// hands both halves on as one input, so the decision costs a re-read of
/// nothing.
pub(crate) struct PrefixSource {
    prefix: Vec<DataBatch>,
    idx: usize,
    rest: Box<dyn Operator>,
}

impl PrefixSource {
    pub(crate) fn new(prefix: Vec<DataBatch>, rest: Box<dyn Operator>) -> Self {
        Self {
            prefix,
            idx: 0,
            rest,
        }
    }
}

impl Operator for PrefixSource {
    fn next(&mut self) -> OperatorResult<'_> {
        Box::pin(async move {
            if self.idx < self.prefix.len() {
                let batch =
                    std::mem::replace(&mut self.prefix[self.idx], DataBatch::new(Vec::new()));
                self.idx += 1;
                if self.idx >= self.prefix.len() {
                    self.prefix.clear();
                    self.idx = 0;
                }
                return Ok(Some(ExecutionBatch::new(batch)));
            }
            self.rest.next().await
        })
    }
}

/// Reads several spill files back in sequence as one stream.
///
/// The probe side of a join whose build side fits after all: the rows were
/// partitioned before that was known, and partitions are only meaningful when
/// both sides are split, so they are read back as the one relation they are.
pub(crate) struct MultiSpillSource {
    readers: Vec<SpillReader>,
    idx: usize,
}

impl MultiSpillSource {
    pub(crate) fn new(readers: Vec<SpillReader>) -> Self {
        Self { readers, idx: 0 }
    }
}

impl Operator for MultiSpillSource {
    fn next(&mut self) -> OperatorResult<'_> {
        Box::pin(async move {
            while self.idx < self.readers.len() {
                match self.readers[self.idx].read_batch()? {
                    Some(batch) => return Ok(Some(ExecutionBatch::new(batch))),
                    None => self.idx += 1,
                }
            }
            Ok(None)
        })
    }
}

/// Reads a spill file back, keeping only the rows whose bit is clear.
///
/// The last stage of a blocked pass: every build block has seen every probe
/// row, so a clear bit is a probe row that matched nothing and a probe-outer
/// join owes it a null-padded output row.
struct UnmatchedProbeSource {
    reader: Arc<std::sync::Mutex<SpillReader>>,
    bits: Arc<ProbeMatchBits>,
    ordinal: u64,
}

impl Operator for UnmatchedProbeSource {
    fn next(&mut self) -> OperatorResult<'_> {
        Box::pin(async move {
            loop {
                let batch = {
                    let mut guard = self.reader.lock().map_err(|_| {
                        ZyronError::ExecutionError("spill reader lock poisoned".into())
                    })?;
                    match guard.read_batch()? {
                        Some(batch) => batch,
                        None => return Ok(None),
                    }
                };
                let base = self.ordinal;
                self.ordinal += batch.num_rows as u64;
                let keep: Vec<u32> = (0..batch.num_rows)
                    .filter(|&row| !self.bits.is_set(base + row as u64))
                    .map(|row| row as u32)
                    .collect();
                if keep.is_empty() {
                    continue;
                }
                if keep.len() == batch.num_rows {
                    return Ok(Some(ExecutionBatch::new(batch)));
                }
                let columns = batch.columns.iter().map(|c| c.take(&keep)).collect();
                return Ok(Some(ExecutionBatch::new(DataBatch::new(columns))));
            }
        })
    }
}

// ---------------------------------------------------------------------------
// Partitioning
// ---------------------------------------------------------------------------

/// Mixes a hash so a second pass over the same rows splits them differently.
///
/// The splitmix64 finalizer. Rows in one partition already agree on the low
/// bits of their first hash, so re-partitioning on that hash would put them
/// all together again however many times it ran.
#[inline]
fn mix(mut x: u64) -> u64 {
    x ^= x >> 30;
    x = x.wrapping_mul(0xbf58476d1ce4e5b9);
    x ^= x >> 27;
    x = x.wrapping_mul(0x94d049bb133111eb);
    x ^ (x >> 31)
}

/// Rows on their way to one partition's spill file.
struct PartitionAccumulator {
    columns: Vec<Column>,
    rows: usize,
}

impl PartitionAccumulator {
    fn new() -> Self {
        Self {
            columns: Vec::new(),
            rows: 0,
        }
    }

    /// Appends the named rows of a batch, gathered per column rather than one
    /// row at a time, so partitioning costs a copy and no dispatch per row.
    fn push_gathered(&mut self, batch: &DataBatch, indices: &[u32], capacity: usize) {
        if self.columns.is_empty() {
            self.columns = batch
                .columns
                .iter()
                .map(|c| {
                    Column::with_nulls_ts(
                        ColumnData::with_capacity_for(c.type_id, c.fractional_digits, capacity),
                        NullBitmap::empty(),
                        c.type_id,
                        c.fractional_digits,
                    )
                })
                .collect();
        }
        for (out, src) in self.columns.iter_mut().zip(&batch.columns) {
            out.data.gather_from(&src.data, indices);
            out.nulls.gather_from(&src.nulls, indices);
        }
        self.rows += indices.len();
    }

    /// Hands over what has accumulated, leaving builders of the same shape.
    fn take(&mut self) -> DataBatch {
        let fresh: Vec<Column> = self
            .columns
            .iter()
            .map(|c| {
                Column::with_nulls_ts(
                    ColumnData::with_capacity_for(
                        c.type_id,
                        c.fractional_digits,
                        self.rows.min(BATCH_SIZE),
                    ),
                    NullBitmap::empty(),
                    c.type_id,
                    c.fractional_digits,
                )
            })
            .collect();
        self.rows = 0;
        DataBatch::new(std::mem::replace(&mut self.columns, fresh))
    }
}

/// One side of one partition, on disk.
pub(crate) struct PartitionSide {
    reader: Option<SpillReader>,
    rows: usize,
    bytes: u64,
}

impl PartitionSide {
    /// Hands over the file, for a side read back whole rather than partition
    /// by partition.
    pub(crate) fn into_reader(self) -> Option<SpillReader> {
        self.reader
    }

    fn empty() -> Self {
        Self {
            reader: None,
            rows: 0,
            bytes: 0,
        }
    }
}

/// Writes rows to one spill file per partition, given the hash to route by.
///
/// Knows nothing about where the hash came from. The join hashes its join
/// keys and the aggregate hashes its grouping keys, and both want the same
/// thing done with the answer.
pub(crate) struct PartitionWriterSet {
    dir: Arc<SpillDirectory>,
    writers: Vec<Option<SpillWriter>>,
    accumulators: Vec<PartitionAccumulator>,
    buckets: Vec<Vec<u32>>,
    rows: Vec<usize>,
    routed: usize,
    /// Rows one partition buffers before its batch is written. Derived from
    /// the first input batch so every partition's buffer together stays inside
    /// the budget that made the operator spill in the first place
    rows_cap: usize,
}

impl PartitionWriterSet {
    pub(crate) fn new(dir: Arc<SpillDirectory>, threshold_bytes: u64) -> Self {
        Self {
            dir,
            writers: (0..GRACE_PARTITIONS).map(|_| None).collect(),
            accumulators: (0..GRACE_PARTITIONS)
                .map(|_| PartitionAccumulator::new())
                .collect(),
            buckets: vec![Vec::new(); GRACE_PARTITIONS],
            rows: vec![0; GRACE_PARTITIONS],
            routed: 0,
            rows_cap: threshold_bytes.max(1) as usize,
        }
    }

    /// Routes every row of a batch.
    pub(crate) fn push_all(&mut self, batch: &DataBatch, hashes: &[u64], seed: u64) -> Result<()> {
        if batch.num_rows == 0 {
            return Ok(());
        }
        self.set_rows_cap(batch);
        for (row, &hash) in hashes.iter().enumerate() {
            self.buckets[bucket_of(mix(hash ^ seed), GRACE_PARTITIONS)].push(row as u32);
        }
        self.routed += batch.num_rows;
        self.drain_buckets(batch)
    }

    /// Routes the named rows and leaves the rest where they are.
    ///
    /// The aggregate's case: a row whose group is already resident is folded
    /// into it, and only the rows that would have created a new group go to
    /// disk.
    pub(crate) fn push_selected(
        &mut self,
        batch: &DataBatch,
        hashes: &[u64],
        rows: &[u32],
        seed: u64,
    ) -> Result<()> {
        if rows.is_empty() {
            return Ok(());
        }
        self.set_rows_cap(batch);
        for &row in rows {
            let hash = hashes[row as usize];
            self.buckets[bucket_of(mix(hash ^ seed), GRACE_PARTITIONS)].push(row);
        }
        self.routed += rows.len();
        self.drain_buckets(batch)
    }

    /// Gathers each partition's rows out of the batch and writes what is full.
    fn drain_buckets(&mut self, batch: &DataBatch) -> Result<()> {
        for p in 0..GRACE_PARTITIONS {
            if self.buckets[p].is_empty() {
                continue;
            }
            let cap = self.rows_cap;
            self.accumulators[p].push_gathered(batch, &self.buckets[p], cap);
            self.rows[p] += self.buckets[p].len();
            self.buckets[p].clear();
            if self.accumulators[p].rows >= self.rows_cap {
                self.flush(p)?;
            }
        }
        Ok(())
    }

    /// Rows written out so far.
    pub(crate) fn routed(&self) -> usize {
        self.routed
    }

    /// Sizes the per-partition buffer the first time rows arrive.
    ///
    /// Every partition holds one of these at once, so the row count is the
    /// budget divided by the partition count and the width of a row, capped at
    /// an output batch. A tiny budget gives small batches on disk, which is
    /// slower to read back and is the correct trade when the alternative is
    /// spending the budget on the buffers that were supposed to save it.
    fn set_rows_cap(&mut self, batch: &DataBatch) {
        if self.rows_cap <= BATCH_SIZE {
            return;
        }
        let budget = self.rows_cap as u64;
        let row_bytes = (batch.approx_bytes() / batch.num_rows.max(1) as u64).max(1);
        let per_partition = budget / (2 * GRACE_PARTITIONS as u64);
        self.rows_cap = ((per_partition / row_bytes) as usize).clamp(1, BATCH_SIZE);
    }

    /// Writes one partition's buffered rows out.
    fn flush(&mut self, partition: usize) -> Result<()> {
        if self.accumulators[partition].rows == 0 {
            return Ok(());
        }
        let batch = self.accumulators[partition].take();
        if self.writers[partition].is_none() {
            self.writers[partition] = Some(self.dir.create()?);
            SpillStats::global()
                .partitions_spilled
                .fetch_add(1, Ordering::Relaxed);
        }
        let writer = self.writers[partition]
            .as_mut()
            .ok_or_else(|| ZyronError::ExecutionError("partition writer went missing".into()))?;
        writer.write_batch(&batch)
    }

    /// Closes every partition and returns them ready to read.
    pub(crate) fn finish(mut self) -> Result<Vec<PartitionSide>> {
        for p in 0..GRACE_PARTITIONS {
            self.flush(p)?;
        }
        let mut out = Vec::with_capacity(GRACE_PARTITIONS);
        for (p, writer) in self.writers.into_iter().enumerate() {
            match writer {
                Some(writer) => {
                    let reader = writer.finish()?;
                    out.push(PartitionSide {
                        rows: self.rows[p],
                        bytes: reader.bytes(),
                        reader: Some(reader),
                    });
                }
                None => out.push(PartitionSide::empty()),
            }
        }
        Ok(out)
    }
}

/// Splits one join input by the hash of its join keys.
pub(crate) struct SidePartitioner {
    keys: Vec<BoundExpr>,
    schema: Vec<LogicalColumn>,
    common_types: Vec<TypeId>,
    align_scales: Vec<Option<u8>>,
    seed: u64,
    writers: PartitionWriterSet,
}

impl SidePartitioner {
    pub(crate) fn new(
        dir: Arc<SpillDirectory>,
        keys: Vec<BoundExpr>,
        schema: Vec<LogicalColumn>,
        common_types: Vec<TypeId>,
        align_scales: Vec<Option<u8>>,
        seed: u64,
        threshold_bytes: u64,
    ) -> Self {
        Self {
            keys,
            schema,
            common_types,
            align_scales,
            seed,
            writers: PartitionWriterSet::new(dir, threshold_bytes),
        }
    }

    /// Routes one batch's rows to their partitions.
    pub(crate) fn push(&mut self, batch: &DataBatch) -> Result<()> {
        if batch.num_rows == 0 {
            return Ok(());
        }
        let hashes = hash_keys(
            batch,
            &self.keys,
            &self.schema,
            &self.common_types,
            &self.align_scales,
        )?;
        self.writers.push_all(batch, &hashes, self.seed)
    }

    /// Closes every partition and returns them ready to read.
    pub(crate) fn finish(self) -> Result<Vec<PartitionSide>> {
        self.writers.finish()
    }
}

/// Per-key type both sides cast to before hashing, from the bound expressions
/// alone.
///
/// Declared types rather than the types the batches turn out to hold, because
/// the two sides are partitioned at different times and a partition boundary
/// only means anything when both sides drew it the same way. Casting to a
/// narrower declared type can put unequal values in one partition, which the
/// join inside that partition rejects, and can never separate equal ones,
/// which is the property partitioning depends on.
pub(crate) fn static_common_types(left: &[BoundExpr], right: &[BoundExpr]) -> Vec<TypeId> {
    left.iter()
        .zip(right.iter())
        .map(|(l, r)| join_key_common_type(l.type_id(), r.type_id()))
        .collect()
}

/// Per-key decimal scale both sides align to before hashing.
pub(crate) fn static_align_scales(left: &[BoundExpr], right: &[BoundExpr]) -> Vec<Option<u8>> {
    left.iter()
        .zip(right.iter())
        .map(|(l, r)| {
            let ld = l.type_id() == TypeId::Decimal;
            let rd = r.type_id() == TypeId::Decimal;
            if !ld && !rd {
                return None;
            }
            let ls = if ld {
                l.fractional_digits().unwrap_or(0)
            } else {
                0
            };
            let rs = if rd {
                r.fractional_digits().unwrap_or(0)
            } else {
                0
            };
            Some(ls.max(rs))
        })
        .collect()
}

// ---------------------------------------------------------------------------
// The driver
// ---------------------------------------------------------------------------

/// Everything a partition needs to build the same join over its own rows.
pub(crate) struct JoinSpec {
    pub(crate) join_type: JoinType,
    pub(crate) left_keys: Vec<BoundExpr>,
    pub(crate) right_keys: Vec<BoundExpr>,
    pub(crate) remaining_condition: Option<BoundExpr>,
    pub(crate) left_schema: Vec<LogicalColumn>,
    pub(crate) right_schema: Vec<LogicalColumn>,
}

/// Both sides of one partition.
struct Partition {
    left: PartitionSide,
    right: PartitionSide,
    depth: u32,
}

/// Runs a join partition by partition.
pub(crate) struct GraceJoin {
    spec: Arc<JoinSpec>,
    dir: Arc<SpillDirectory>,
    threshold: u64,
    /// Partitions not yet run. Taken from the back, so a partition split by
    /// reflow is run before its siblings and its files are freed sooner
    pending: Vec<Partition>,
    current: Option<Box<dyn Operator>>,
    blocked: Option<BlockedPass>,
}

impl GraceJoin {
    pub(crate) fn new(
        spec: JoinSpec,
        dir: Arc<SpillDirectory>,
        threshold: u64,
        left: Vec<PartitionSide>,
        right: Vec<PartitionSide>,
    ) -> Self {
        let pending = left
            .into_iter()
            .zip(right)
            .map(|(left, right)| Partition {
                left,
                right,
                depth: 0,
            })
            .collect();
        SpillStats::global()
            .joins_spilled
            .fetch_add(1, Ordering::Relaxed);
        Self {
            spec: Arc::new(spec),
            dir,
            threshold,
            pending,
            current: None,
            blocked: None,
        }
    }

    /// The next batch of the joined result, or None when every partition has
    /// been run.
    pub(crate) async fn next(&mut self) -> Result<Option<DataBatch>> {
        loop {
            if let Some(op) = self.current.as_mut() {
                match op.next().await? {
                    Some(eb) => return Ok(Some(eb.batch)),
                    None => {
                        self.current = None;
                        continue;
                    }
                }
            }
            if let Some(blocked) = self.blocked.as_mut() {
                match blocked.next_stage()? {
                    Some(op) => {
                        self.current = Some(op);
                        continue;
                    }
                    None => {
                        self.blocked = None;
                        continue;
                    }
                }
            }
            match self.pending.pop() {
                Some(partition) => self.start(partition)?,
                None => return Ok(None),
            }
        }
    }

    /// Decides how one partition runs: directly, split again, or in blocks.
    fn start(&mut self, partition: Partition) -> Result<()> {
        if partition.left.rows == 0 && partition.right.rows == 0 {
            return Ok(());
        }
        // The smaller side builds. Both sides are already measured on disk, so
        // this is the count the in-memory join has to drain both inputs to get
        let build_is_right = partition.right.bytes <= partition.left.bytes;
        let build_bytes = partition.left.bytes.min(partition.right.bytes);

        if build_bytes <= self.threshold {
            self.current = Some(self.partition_join(partition, build_is_right)?);
            return Ok(());
        }
        if partition.depth >= MAX_REFLOW_DEPTH {
            SpillStats::global()
                .blocked_passes
                .fetch_add(1, Ordering::Relaxed);
            self.blocked = Some(BlockedPass::new(
                Arc::clone(&self.spec),
                partition,
                build_is_right,
                self.threshold,
            )?);
            return Ok(());
        }
        self.reflow(partition)
    }

    /// Splits a partition whose build side still does not fit and queues the
    /// pieces in its place.
    fn reflow(&mut self, partition: Partition) -> Result<()> {
        let seed = mix(partition.depth as u64 + 1);
        let common = static_common_types(&self.spec.left_keys, &self.spec.right_keys);
        let align = static_align_scales(&self.spec.left_keys, &self.spec.right_keys);
        let parent_left_rows = partition.left.rows;
        let parent_right_rows = partition.right.rows;

        let left = self.repartition(
            partition.left,
            &self.spec.left_keys.clone(),
            &self.spec.left_schema.clone(),
            &common,
            &align,
            seed,
        )?;
        let right = self.repartition(
            partition.right,
            &self.spec.right_keys.clone(),
            &self.spec.right_schema.clone(),
            &common,
            &align,
            seed,
        )?;

        for (left, right) in left.into_iter().zip(right) {
            if left.rows == 0 && right.rows == 0 {
                continue;
            }
            // A child holding everything its parent held is a partition that
            // two independent hashes could not separate, which is one key
            // value. Sending it round again would read and write it for
            // nothing, so it goes straight to the blocked pass
            let split = left.rows < parent_left_rows || right.rows < parent_right_rows;
            let depth = if split {
                partition.depth + 1
            } else {
                MAX_REFLOW_DEPTH
            };
            self.pending.push(Partition { left, right, depth });
        }
        Ok(())
    }

    /// Reads one side of a partition back and splits it under a new seed.
    fn repartition(
        &self,
        side: PartitionSide,
        keys: &[BoundExpr],
        schema: &[LogicalColumn],
        common: &[TypeId],
        align: &[Option<u8>],
        seed: u64,
    ) -> Result<Vec<PartitionSide>> {
        let Some(mut reader) = side.reader else {
            return Ok((0..GRACE_PARTITIONS)
                .map(|_| PartitionSide::empty())
                .collect());
        };
        let mut parts = SidePartitioner::new(
            Arc::clone(&self.dir),
            keys.to_vec(),
            schema.to_vec(),
            common.to_vec(),
            align.to_vec(),
            seed,
            self.threshold,
        );
        while let Some(batch) = reader.read_batch()? {
            parts.push(&batch)?;
        }
        // The parent's file goes here, so the children never coexist with it
        // against the spill quota
        drop(reader);
        parts.finish()
    }

    /// Builds the join for one partition, reading both sides from disk.
    fn partition_join(
        &self,
        partition: Partition,
        build_is_right: bool,
    ) -> Result<Box<dyn Operator>> {
        let left = side_source(partition.left);
        let right = side_source(partition.right);
        let mut op = HashJoinOperator::new(
            left,
            right,
            self.spec.join_type,
            self.spec.left_keys.clone(),
            self.spec.right_keys.clone(),
            self.spec.remaining_condition.clone(),
            self.spec.left_schema.clone(),
            self.spec.right_schema.clone(),
        );
        op.build_from(build_is_right);
        Ok(Box::new(op))
    }
}

/// Wraps one partition side as an operator, empty sides included.
fn side_source(side: PartitionSide) -> Box<dyn Operator> {
    match side.reader {
        Some(reader) => Box::new(SpillSource::new(Arc::new(std::sync::Mutex::new(reader)))),
        None => Box::new(BatchListSource::new(Vec::new())),
    }
}

// ---------------------------------------------------------------------------
// Blocked pass
// ---------------------------------------------------------------------------

/// A partition whose build side does not fit and will not split, passed over
/// in blocks.
struct BlockedPass {
    spec: Arc<JoinSpec>,
    build: SpillReader,
    /// None when the partition has no probe rows, which still leaves the
    /// build side owing its outer rows
    probe: Option<Arc<std::sync::Mutex<SpillReader>>>,
    build_is_right: bool,
    /// Set only when the probe side is the outer one, which is the only case
    /// where a row's fate depends on more than one block
    bits: Option<Arc<ProbeMatchBits>>,
    threshold: u64,
    build_done: bool,
    unmatched_done: bool,
}

impl BlockedPass {
    fn new(
        spec: Arc<JoinSpec>,
        partition: Partition,
        build_is_right: bool,
        threshold: u64,
    ) -> Result<Self> {
        let (build_side, probe_side) = if build_is_right {
            (partition.right, partition.left)
        } else {
            (partition.left, partition.right)
        };
        let build = build_side.reader.ok_or_else(|| {
            ZyronError::ExecutionError(
                "a partition too large to fit has no rows to read, which cannot both be true"
                    .into(),
            )
        })?;
        let probe_rows = probe_side.rows as u64;
        let probe = probe_side
            .reader
            .map(|reader| Arc::new(std::sync::Mutex::new(reader)));
        let probe_outer = probe.is_some()
            && match (spec.join_type, build_is_right) {
                (JoinType::Full, _) => true,
                (JoinType::Right, false) => true,
                (JoinType::Left, true) => true,
                _ => false,
            };
        Ok(Self {
            spec,
            build,
            probe,
            build_is_right,
            bits: probe_outer.then(|| Arc::new(ProbeMatchBits::new(probe_rows))),
            threshold,
            build_done: false,
            unmatched_done: !probe_outer,
        })
    }

    /// The join for the next build block, then the unmatched-probe pass, then
    /// nothing.
    fn next_stage(&mut self) -> Result<Option<Box<dyn Operator>>> {
        if !self.build_done {
            let mut block = Vec::new();
            let mut bytes = 0u64;
            while bytes < self.threshold {
                match self.build.read_batch()? {
                    Some(batch) => {
                        bytes += batch.approx_bytes();
                        block.push(batch);
                    }
                    None => {
                        self.build_done = true;
                        break;
                    }
                }
            }
            if !block.is_empty() {
                self.rewind_probe()?;
                return Ok(Some(self.block_join(block)));
            }
        }
        if !self.unmatched_done {
            self.unmatched_done = true;
            if let Some(bits) = self.bits.clone() {
                self.rewind_probe()?;
                return Ok(Some(self.unmatched_join(bits)));
            }
        }
        Ok(None)
    }

    fn rewind_probe(&self) -> Result<()> {
        match &self.probe {
            Some(probe) => probe
                .lock()
                .map_err(|_| ZyronError::ExecutionError("spill reader lock poisoned".into()))?
                .rewind(),
            None => Ok(()),
        }
    }

    /// A fresh operator over the probe side, or an empty one when the
    /// partition has no probe rows.
    fn probe_source(&self) -> Box<dyn Operator> {
        match &self.probe {
            Some(probe) => Box::new(SpillSource::new(Arc::clone(probe))),
            None => Box::new(BatchListSource::new(Vec::new())),
        }
    }

    /// One build block against the whole probe side.
    ///
    /// The join type keeps build-side outer rows, which are correct here
    /// because this block sees every probe row, and drops probe-side outer
    /// rows, which are not, because the other blocks have not run yet.
    fn block_join(&self, block: Vec<DataBatch>) -> Box<dyn Operator> {
        let build: Box<dyn Operator> = Box::new(BatchListSource::new(block));
        let probe: Box<dyn Operator> = self.probe_source();
        let (left, right) = if self.build_is_right {
            (probe, build)
        } else {
            (build, probe)
        };
        let join_type = match (self.spec.join_type, self.build_is_right) {
            (JoinType::Full, false) => JoinType::Left,
            (JoinType::Full, true) => JoinType::Right,
            (JoinType::Right, false) => JoinType::Inner,
            (JoinType::Left, true) => JoinType::Inner,
            (other, _) => other,
        };
        let mut op = HashJoinOperator::new(
            left,
            right,
            join_type,
            self.spec.left_keys.clone(),
            self.spec.right_keys.clone(),
            self.spec.remaining_condition.clone(),
            self.spec.left_schema.clone(),
            self.spec.right_schema.clone(),
        );
        op.build_from(self.build_is_right);
        if let Some(bits) = &self.bits {
            op.record_probe_matches(Arc::clone(bits));
        }
        Box::new(op)
    }

    /// The probe rows no block matched, null-padded on the build side.
    ///
    /// An empty build side against a probe-outer join is exactly that
    /// output, so this reuses the join's own assembly rather than a second
    /// copy of it.
    fn unmatched_join(&self, bits: Arc<ProbeMatchBits>) -> Box<dyn Operator> {
        let build: Box<dyn Operator> = Box::new(BatchListSource::new(Vec::new()));
        let probe: Box<dyn Operator> = match &self.probe {
            Some(reader) => Box::new(UnmatchedProbeSource {
                reader: Arc::clone(reader),
                bits,
                ordinal: 0,
            }),
            None => Box::new(BatchListSource::new(Vec::new())),
        };
        let (left, right) = if self.build_is_right {
            (probe, build)
        } else {
            (build, probe)
        };
        let join_type = if self.build_is_right {
            JoinType::Left
        } else {
            JoinType::Right
        };
        let mut op = HashJoinOperator::new(
            left,
            right,
            join_type,
            self.spec.left_keys.clone(),
            self.spec.right_keys.clone(),
            self.spec.remaining_condition.clone(),
            self.spec.left_schema.clone(),
            self.spec.right_schema.clone(),
        );
        op.build_from(self.build_is_right);
        Box::new(op)
    }
}
