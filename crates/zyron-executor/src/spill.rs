//! Spill files: what a materializing operator writes when it runs out of
//! memory, instead of failing.
//!
//! Before this existed, a sort or a hash join that outgrew its memory budget
//! returned an error naming the budget. That is a correct thing to do only if
//! the alternative is worse, and the alternative is the one every other
//! database chose decades ago: put the excess on disk and read it back in
//! order. A query that is too big should be slow, not impossible.
//!
//! ## What a spill file is
//!
//! A sequence of encoded batches, written once, read once, in order. Nothing
//! seeks, nothing updates, nothing is read twice. That shape is what decides
//! the design:
//!
//! - **Page structured, but not in the shared buffer pool.** The file is
//!   written and read in `PAGE_SIZE` blocks so the IO is aligned and each
//!   reader holds one block rather than a run. It deliberately does not go
//!   through the shared page table: spill happens exactly when the node is
//!   short of memory, and inserting write-once pages there would evict the
//!   real data pages that are still being used to make room for pages nobody
//!   will read twice. The buffers are charged against the node memory gauge
//!   instead, so the memory a spill costs is memory the node knows about.
//! - **A raw encoding rather than the columnar format.** The `.zyr` writer
//!   builds statistics, dictionaries, and a footer, all of which pay off over
//!   many reads. A spill file is read once and deleted, so every byte spent
//!   compressing it is a byte spent for nothing.
//!
//! ## Lifetime
//!
//! A file deletes itself when its handle drops, which covers the query
//! finishing, erroring, being cancelled, and panicking. Files left by a
//! process that died are removed when the directory is opened, because
//! nothing else will ever read them and the disk they hold is otherwise lost
//! until an operator notices.
//!
//! ## Quota
//!
//! Spilling is bounded. A query that would take the node past the spill quota
//! is refused with an error naming the quota, which is the same honesty the
//! memory budget gives: unbounded spill is a full disk, and a full disk stops
//! the whole node rather than one query.

use std::fs::{File, OpenOptions};
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use zyron_common::page::PAGE_SIZE;
use zyron_common::{Result, RowLocator, TypeId, ZyronError};

use crate::batch::DataBatch;
use crate::column::{Column, ColumnData, NullBitmap};

/// Directory name under the data directory.
pub const SPILL_DIR: &str = "spill";

/// Marks a file as one of ours, so the startup purge cannot delete something
/// that merely shares the directory.
const MAGIC: [u8; 8] = *b"ZYSPILL\x00";

/// Format version. A spill file never outlives the process that wrote it, so
/// this exists to catch a leftover from a different build rather than to
/// support one.
const VERSION: u32 = 1;

/// Bytes buffered before a write reaches the device.
///
/// One page. Larger buffers would cut syscalls further, and the reason not to
/// is that a k-way merge holds one of these per run: at sixty-four runs a
/// megabyte buffer is sixty-four megabytes of the memory the operator spilled
/// to escape.
const BUFFER_BYTES: usize = PAGE_SIZE;

// ---------------------------------------------------------------------------
// Statistics
// ---------------------------------------------------------------------------

/// What spilling has cost this node, for `zyron_sys.pressure.spill_stats`.
///
/// Process wide rather than per query, because the question an operator asks
/// is whether this node is spilling, and the question a query asks is answered
/// by its own plan.
#[derive(Debug, Default)]
pub struct SpillStats {
    pub files_created: AtomicU64,
    pub files_deleted: AtomicU64,
    pub bytes_written: AtomicU64,
    pub bytes_read: AtomicU64,
    pub batches_written: AtomicU64,
    pub batches_read: AtomicU64,
    /// Sorts and joins that had to spill at all
    pub sorts_spilled: AtomicU64,
    pub joins_spilled: AtomicU64,
    pub aggregates_spilled: AtomicU64,
    /// Runs and partitions merged back in
    pub runs_merged: AtomicU64,
    pub partitions_spilled: AtomicU64,
    /// Partitions that no hash would split, passed over in blocks instead.
    /// Nonzero means join key skew severe enough that one key value does not
    /// fit in the budget on either side
    pub blocked_passes: AtomicU64,
    /// Refusals, which are the cases where spilling did not save the query
    pub quota_refusals: AtomicU64,
    /// Bytes currently on disk, and the most there has ever been
    pub live_bytes: AtomicU64,
    pub peak_live_bytes: AtomicU64,
}

static STATS: SpillStats = SpillStats::new();

impl SpillStats {
    pub const fn new() -> Self {
        Self {
            files_created: AtomicU64::new(0),
            files_deleted: AtomicU64::new(0),
            bytes_written: AtomicU64::new(0),
            bytes_read: AtomicU64::new(0),
            batches_written: AtomicU64::new(0),
            batches_read: AtomicU64::new(0),
            sorts_spilled: AtomicU64::new(0),
            joins_spilled: AtomicU64::new(0),
            aggregates_spilled: AtomicU64::new(0),
            runs_merged: AtomicU64::new(0),
            partitions_spilled: AtomicU64::new(0),
            blocked_passes: AtomicU64::new(0),
            quota_refusals: AtomicU64::new(0),
            live_bytes: AtomicU64::new(0),
            peak_live_bytes: AtomicU64::new(0),
        }
    }

    /// The process counters, which is what the view reads.
    pub fn global() -> &'static SpillStats {
        &STATS
    }

    fn add_live(&self, bytes: u64) {
        let now = self.live_bytes.fetch_add(bytes, Ordering::Relaxed) + bytes;
        self.peak_live_bytes.fetch_max(now, Ordering::Relaxed);
    }

    fn release_live(&self, bytes: u64) {
        let mut current = self.live_bytes.load(Ordering::Relaxed);
        loop {
            let next = current.saturating_sub(bytes);
            match self.live_bytes.compare_exchange_weak(
                current,
                next,
                Ordering::Relaxed,
                Ordering::Relaxed,
            ) {
                Ok(_) => return,
                Err(observed) => current = observed,
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Directory and quota
// ---------------------------------------------------------------------------

/// Where spill files live, and how much of the disk they may take.
#[derive(Debug)]
pub struct SpillDirectory {
    root: PathBuf,
    quota_bytes: u64,
    reserved: AtomicU64,
    next_id: AtomicU64,
}

impl SpillDirectory {
    /// Opens the spill directory, removing anything a previous process left.
    ///
    /// The purge is the only garbage collection this needs. A spill file is
    /// meaningful to exactly one operator in one process, so a file that
    /// outlived its process is unreadable by definition, and leaving it costs
    /// disk that nothing will ever reclaim.
    pub fn open(data_dir: &Path, quota_bytes: u64) -> Result<Self> {
        let root = data_dir.join(SPILL_DIR);
        std::fs::create_dir_all(&root)?;
        let purged = purge(&root);
        if purged > 0 {
            tracing::info!(
                files = purged,
                "removed spill files left by a previous process"
            );
        }
        Ok(Self {
            root,
            quota_bytes,
            reserved: AtomicU64::new(0),
            next_id: AtomicU64::new(0),
        })
    }

    pub fn root(&self) -> &Path {
        &self.root
    }

    pub fn quota_bytes(&self) -> u64 {
        self.quota_bytes
    }

    /// Bytes currently held by live spill files.
    pub fn reserved_bytes(&self) -> u64 {
        self.reserved.load(Ordering::Relaxed)
    }

    /// Files this directory has handed out since it opened.
    ///
    /// Per directory rather than the process-wide counter the view reads, so
    /// one query's spilling can be measured while others are running.
    pub fn files_created(&self) -> u64 {
        self.next_id.load(Ordering::Relaxed)
    }

    /// Takes quota for a write, or refuses.
    ///
    /// Refusing is the honest answer rather than writing anyway: a full disk
    /// stops every query on the node, and one query failing with a message
    /// naming the quota is strictly better than that.
    fn take_quota(&self, bytes: u64) -> Result<()> {
        let mut current = self.reserved.load(Ordering::Relaxed);
        loop {
            let next = current.saturating_add(bytes);
            if next > self.quota_bytes {
                STATS.quota_refusals.fetch_add(1, Ordering::Relaxed);
                return Err(ZyronError::ExecutionError(format!(
                    "spilling this query would take {} bytes past the node's spill quota of {} \
                     bytes, raise query.spill_quota_bytes or narrow the query",
                    next - self.quota_bytes,
                    self.quota_bytes
                )));
            }
            match self.reserved.compare_exchange_weak(
                current,
                next,
                Ordering::Relaxed,
                Ordering::Relaxed,
            ) {
                Ok(_) => {
                    STATS.add_live(bytes);
                    return Ok(());
                }
                Err(observed) => current = observed,
            }
        }
    }

    fn release_quota(&self, bytes: u64) {
        let mut current = self.reserved.load(Ordering::Relaxed);
        loop {
            let next = current.saturating_sub(bytes);
            match self.reserved.compare_exchange_weak(
                current,
                next,
                Ordering::Relaxed,
                Ordering::Relaxed,
            ) {
                Ok(_) => {
                    STATS.release_live(bytes);
                    return;
                }
                Err(observed) => current = observed,
            }
        }
    }

    /// Opens a new spill file for writing.
    pub fn create(self: &Arc<Self>) -> Result<SpillWriter> {
        let id = self.next_id.fetch_add(1, Ordering::Relaxed);
        let path = self.root.join(format!("{}_{id}.spill", std::process::id()));
        let file = OpenOptions::new()
            .create(true)
            .write(true)
            .truncate(true)
            .open(&path)?;
        STATS.files_created.fetch_add(1, Ordering::Relaxed);
        let mut writer = SpillWriter {
            file: Some(file),
            handle: SpillHandle {
                path,
                dir: Arc::clone(self),
                bytes: 0,
            },
            buffer: Vec::with_capacity(BUFFER_BYTES),
            batches: 0,
        };
        writer.buffer.extend_from_slice(&MAGIC);
        writer.buffer.extend_from_slice(&VERSION.to_le_bytes());
        Ok(writer)
    }
}

/// Removes every spill file under a directory, returning how many went.
fn purge(root: &Path) -> usize {
    let Ok(entries) = std::fs::read_dir(root) else {
        return 0;
    };
    let mut removed = 0;
    for entry in entries.flatten() {
        let path = entry.path();
        if path.extension().and_then(|e| e.to_str()) == Some("spill")
            && std::fs::remove_file(&path).is_ok()
        {
            removed += 1;
        }
    }
    removed
}

// ---------------------------------------------------------------------------
// Handle
// ---------------------------------------------------------------------------

/// Owns a spill file's existence. Deleting it is this type's only job.
///
/// A handle rather than a call at the end of the operator, because the end of
/// an operator is not one place: it finishes, it errors, it is cancelled, and
/// it can be dropped mid-batch by a limit above it. Every one of those paths
/// runs a destructor and none of them reliably runs cleanup code.
#[derive(Debug)]
pub struct SpillHandle {
    path: PathBuf,
    dir: Arc<SpillDirectory>,
    bytes: u64,
}

impl SpillHandle {
    pub fn path(&self) -> &Path {
        &self.path
    }

    pub fn bytes(&self) -> u64 {
        self.bytes
    }
}

impl Drop for SpillHandle {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.path);
        self.dir.release_quota(self.bytes);
        STATS.files_deleted.fetch_add(1, Ordering::Relaxed);
    }
}

// ---------------------------------------------------------------------------
// Writer
// ---------------------------------------------------------------------------

/// Appends encoded batches to a spill file.
///
/// No destructor of its own: the handle it holds has one, and an abandoned
/// writer's bytes are exactly as worthless as a finished one's once the
/// operator is gone. A `Drop` here would only stop `finish` from handing the
/// handle to the reader.
pub struct SpillWriter {
    file: Option<File>,
    handle: SpillHandle,
    buffer: Vec<u8>,
    batches: u64,
}

impl SpillWriter {
    /// Appends one batch.
    pub fn write_batch(&mut self, batch: &DataBatch) -> Result<()> {
        self.write_batch_with_locators(batch, None)
    }

    /// Appends one batch along with the storage rows it came from.
    ///
    /// Locators travel with the batch because a sort feeding a row-locking
    /// operator has to name the rows it ordered, and a sort that could not
    /// spill in that case would leave the one query shape that most needs to.
    pub fn write_batch_with_locators(
        &mut self,
        batch: &DataBatch,
        locators: Option<&[RowLocator]>,
    ) -> Result<()> {
        let start = self.buffer.len();
        encode_batch(batch, locators, &mut self.buffer)?;
        let grew = (self.buffer.len() - start) as u64;
        self.handle.dir.take_quota(grew)?;
        self.handle.bytes += grew;
        self.batches += 1;
        STATS.batches_written.fetch_add(1, Ordering::Relaxed);
        // Flush whole buffers as they fill, so a batch larger than the buffer
        // is written in page-sized pieces rather than held whole
        while self.buffer.len() >= BUFFER_BYTES {
            self.flush_front(BUFFER_BYTES)?;
        }
        Ok(())
    }

    /// Writes the first `n` buffered bytes and keeps the remainder.
    fn flush_front(&mut self, n: usize) -> Result<()> {
        let file = self
            .file
            .as_mut()
            .ok_or_else(|| ZyronError::ExecutionError("spill writer already closed".into()))?;
        file.write_all(&self.buffer[..n])?;
        self.buffer.drain(..n);
        STATS.bytes_written.fetch_add(n as u64, Ordering::Relaxed);
        Ok(())
    }

    pub fn batches(&self) -> u64 {
        self.batches
    }

    /// Finishes the file and returns a reader over it.
    ///
    /// No fsync. A spill file is meaningless to any process but this one, so
    /// durability across a crash would be paying for a guarantee whose value
    /// is zero: after a crash the file is deleted unread.
    pub fn finish(mut self) -> Result<SpillReader> {
        let remaining = self.buffer.len();
        if remaining > 0 {
            self.flush_front(remaining)?;
        }
        self.file = None;
        let handle = self.handle;
        // The handle moves into the reader, so the file lives exactly as long
        // as something can still read it
        SpillReader::open(handle, self.batches)
    }
}

// ---------------------------------------------------------------------------
// Reader
// ---------------------------------------------------------------------------

/// Reads batches back in the order they were written.
pub struct SpillReader {
    file: File,
    handle: SpillHandle,
    buffer: Vec<u8>,
    /// Read position inside the buffer
    cursor: usize,
    remaining_batches: u64,
    /// Batches the file holds, which is what a rewind restores
    total_batches: u64,
    exhausted: bool,
}

impl SpillReader {
    fn open(handle: SpillHandle, batches: u64) -> Result<Self> {
        let mut file = File::open(handle.path())?;
        let mut header = [0u8; MAGIC.len() + 4];
        file.read_exact(&mut header)?;
        if header[..MAGIC.len()] != MAGIC {
            return Err(ZyronError::ExecutionError(
                "spill file does not carry the expected magic".into(),
            ));
        }
        let version = u32::from_le_bytes(
            header[MAGIC.len()..]
                .try_into()
                .map_err(|_| ZyronError::ExecutionError("spill header".into()))?,
        );
        if version != VERSION {
            return Err(ZyronError::ExecutionError(format!(
                "spill file is version {version}, this build writes {VERSION}"
            )));
        }
        Ok(Self {
            file,
            handle,
            buffer: Vec::with_capacity(BUFFER_BYTES),
            cursor: 0,
            remaining_batches: batches,
            total_batches: batches,
            exhausted: batches == 0,
        })
    }

    /// Reads the file again from its first batch.
    ///
    /// A spill file is normally read once, and a join that has to pass one
    /// side over the other in blocks is the exception: the same probe rows
    /// are read against every block of the build side. Nothing is rewritten,
    /// so a rewind costs a seek and the reads that follow it.
    pub fn rewind(&mut self) -> Result<()> {
        self.file.seek(SeekFrom::Start((MAGIC.len() + 4) as u64))?;
        self.buffer.clear();
        self.cursor = 0;
        self.remaining_batches = self.total_batches;
        self.exhausted = self.total_batches == 0;
        Ok(())
    }

    pub fn batches(&self) -> u64 {
        self.remaining_batches
    }

    pub fn bytes(&self) -> u64 {
        self.handle.bytes()
    }

    /// Reads the next batch, or None at the end.
    pub fn read_batch(&mut self) -> Result<Option<DataBatch>> {
        Ok(self.read_batch_with_locators()?.map(|(batch, _)| batch))
    }

    /// Reads the next batch and whatever locators were written with it.
    pub fn read_batch_with_locators(
        &mut self,
    ) -> Result<Option<(DataBatch, Option<Vec<RowLocator>>)>> {
        if self.exhausted || self.remaining_batches == 0 {
            return Ok(None);
        }
        let pair = decode_batch(self)?;
        self.remaining_batches -= 1;
        if self.remaining_batches == 0 {
            self.exhausted = true;
        }
        STATS.batches_read.fetch_add(1, Ordering::Relaxed);
        Ok(Some(pair))
    }

    /// Makes at least `need` bytes available at the cursor, or fails.
    fn fill(&mut self, need: usize) -> Result<()> {
        if self.buffer.len() - self.cursor >= need {
            return Ok(());
        }
        // Compact what is left to the front, then top up
        self.buffer.drain(..self.cursor);
        self.cursor = 0;
        while self.buffer.len() < need {
            let want = need.max(BUFFER_BYTES) - self.buffer.len();
            let base = self.buffer.len();
            self.buffer.resize(base + want, 0);
            let read = self.file.read(&mut self.buffer[base..])?;
            self.buffer.truncate(base + read);
            if read == 0 {
                return Err(ZyronError::ExecutionError(
                    "spill file ended inside a batch".into(),
                ));
            }
            STATS.bytes_read.fetch_add(read as u64, Ordering::Relaxed);
        }
        Ok(())
    }

    fn take(&mut self, n: usize) -> Result<&[u8]> {
        self.fill(n)?;
        let start = self.cursor;
        self.cursor += n;
        Ok(&self.buffer[start..start + n])
    }

    fn take_u8(&mut self) -> Result<u8> {
        Ok(self.take(1)?[0])
    }

    fn take_u16(&mut self) -> Result<u16> {
        let b = self.take(2)?;
        Ok(u16::from_le_bytes([b[0], b[1]]))
    }

    fn take_u32(&mut self) -> Result<u32> {
        let b = self.take(4)?;
        Ok(u32::from_le_bytes([b[0], b[1], b[2], b[3]]))
    }

    fn take_u64(&mut self) -> Result<u64> {
        let b = self.take(8)?;
        let mut out = [0u8; 8];
        out.copy_from_slice(b);
        Ok(u64::from_le_bytes(out))
    }
}

// ---------------------------------------------------------------------------
// Batch codec
// ---------------------------------------------------------------------------

/// Writes a batch as `[rows][columns]` then each column's type, nullability,
/// and payload.
fn encode_batch(
    batch: &DataBatch,
    locators: Option<&[RowLocator]>,
    out: &mut Vec<u8>,
) -> Result<()> {
    out.extend_from_slice(&(batch.num_rows as u32).to_le_bytes());
    out.extend_from_slice(&(batch.columns.len() as u16).to_le_bytes());
    match locators {
        Some(locs) if locs.len() == batch.num_rows => {
            out.push(1);
            for loc in locs {
                encode_locator(*loc, out);
            }
        }
        Some(locs) => {
            return Err(ZyronError::ExecutionError(format!(
                "spill batch has {} rows but {} locators",
                batch.num_rows,
                locs.len()
            )));
        }
        None => out.push(0),
    }
    for column in &batch.columns {
        out.push(column.type_id as u8);
        // Precision travels with the value, because a picosecond timestamp
        // read back as a plain integer compares wrongly against everything
        match column.fractional_digits {
            Some(d) => {
                out.push(1);
                out.push(d);
            }
            None => {
                out.push(0);
                out.push(0);
            }
        }
        if column.nulls.any_null() {
            out.push(1);
            let words = column.nulls.words();
            out.extend_from_slice(&(words.len() as u32).to_le_bytes());
            for word in words {
                out.extend_from_slice(&word.to_le_bytes());
            }
        } else {
            out.push(0);
        }
        encode_column_data(&column.data, out);
    }
    Ok(())
}

/// Fixed-width payloads go out as raw little-endian elements, variable-width
/// ones as a length table followed by the bytes.
fn encode_column_data(data: &ColumnData, out: &mut Vec<u8>) {
    macro_rules! fixed {
        ($v:expr, $to:ident) => {{
            out.extend_from_slice(&($v.len() as u32).to_le_bytes());
            for item in $v {
                out.extend_from_slice(&item.$to());
            }
        }};
    }
    match data {
        ColumnData::Boolean(v) => {
            out.extend_from_slice(&(v.len() as u32).to_le_bytes());
            for item in v {
                out.push(*item as u8);
            }
        }
        ColumnData::Int8(v) => fixed!(v, to_le_bytes),
        ColumnData::Int16(v) => fixed!(v, to_le_bytes),
        ColumnData::Int32(v) => fixed!(v, to_le_bytes),
        ColumnData::Int64(v) => fixed!(v, to_le_bytes),
        ColumnData::Int128(v) => fixed!(v, to_le_bytes),
        ColumnData::UInt8(v) => fixed!(v, to_le_bytes),
        ColumnData::UInt16(v) => fixed!(v, to_le_bytes),
        ColumnData::UInt32(v) => fixed!(v, to_le_bytes),
        ColumnData::UInt64(v) => fixed!(v, to_le_bytes),
        ColumnData::Float32(v) => fixed!(v, to_le_bytes),
        ColumnData::Float64(v) => fixed!(v, to_le_bytes),
        ColumnData::FixedBinary16(v) => {
            out.extend_from_slice(&(v.len() as u32).to_le_bytes());
            for item in v {
                out.extend_from_slice(item);
            }
        }
        ColumnData::Interval(v) => {
            out.extend_from_slice(&(v.len() as u32).to_le_bytes());
            for item in v {
                out.extend_from_slice(&item.months.to_le_bytes());
                out.extend_from_slice(&item.days.to_le_bytes());
                out.extend_from_slice(&item.nanoseconds.to_le_bytes());
            }
        }
        ColumnData::Utf8(v) => {
            out.extend_from_slice(&(v.len() as u32).to_le_bytes());
            for item in v {
                out.extend_from_slice(&(item.len() as u32).to_le_bytes());
            }
            for item in v {
                out.extend_from_slice(item.as_bytes());
            }
        }
        ColumnData::Binary(v) => {
            out.extend_from_slice(&(v.len() as u32).to_le_bytes());
            for item in v {
                out.extend_from_slice(&(item.len() as u32).to_le_bytes());
            }
            for item in v {
                out.extend_from_slice(item);
            }
        }
    }
}

fn decode_batch(reader: &mut SpillReader) -> Result<(DataBatch, Option<Vec<RowLocator>>)> {
    let rows = reader.take_u32()? as usize;
    let ncols = reader.take_u16()? as usize;
    let locators = if reader.take_u8()? == 1 {
        let mut locs = Vec::with_capacity(rows);
        for _ in 0..rows {
            locs.push(decode_locator(reader)?);
        }
        Some(locs)
    } else {
        None
    };
    let mut columns = Vec::with_capacity(ncols);
    for _ in 0..ncols {
        let type_id = TypeId::from_u8(reader.take_u8()?).ok_or_else(|| {
            ZyronError::ExecutionError("spill file names a type this build does not have".into())
        })?;
        let has_digits = reader.take_u8()? == 1;
        let digits = reader.take_u8()?;
        let fractional_digits = if has_digits { Some(digits) } else { None };

        let nulls = if reader.take_u8()? == 1 {
            let word_count = reader.take_u32()? as usize;
            let mut words = Vec::with_capacity(word_count);
            for _ in 0..word_count {
                words.push(reader.take_u64()?);
            }
            NullBitmap::from_words(words, rows)
        } else {
            NullBitmap::none(rows)
        };

        let data = decode_column_data(reader, type_id)?;
        columns.push(Column {
            data,
            nulls,
            type_id,
            fractional_digits,
        });
    }
    Ok((
        DataBatch {
            columns,
            num_rows: rows,
            resolved: Vec::new(),
        },
        locators,
    ))
}

/// A locator as a tag and two words, which is every form it takes.
fn encode_locator(loc: RowLocator, out: &mut Vec<u8>) {
    let (tag, a, b) = match loc {
        RowLocator::Heap { page, slot } => (1u8, page.as_u64(), slot as u64),
        RowLocator::Columnar { file_id, sys_rowid } => (2, file_id, sys_rowid),
        RowLocator::Lake { file_id, ordinal } => (3, file_id, ordinal),
    };
    out.push(tag);
    out.extend_from_slice(&a.to_le_bytes());
    out.extend_from_slice(&b.to_le_bytes());
}

fn decode_locator(reader: &mut SpillReader) -> Result<RowLocator> {
    let tag = reader.take_u8()?;
    let a = reader.take_u64()?;
    let b = reader.take_u64()?;
    Ok(match tag {
        1 => {
            let page = zyron_common::page::PageId::from_u64(a);
            RowLocator::Heap {
                page,
                slot: b as u16,
            }
        }
        2 => RowLocator::Columnar {
            file_id: a,
            sys_rowid: b,
        },
        3 => RowLocator::Lake {
            file_id: a,
            ordinal: b,
        },
        other => {
            return Err(ZyronError::ExecutionError(format!(
                "spill file names row locator kind {other}, which this build does not have"
            )));
        }
    })
}

fn decode_column_data(reader: &mut SpillReader, type_id: TypeId) -> Result<ColumnData> {
    let len = reader.take_u32()? as usize;
    macro_rules! fixed {
        ($variant:ident, $ty:ty, $width:expr) => {{
            let mut v: Vec<$ty> = Vec::with_capacity(len);
            for _ in 0..len {
                let b = reader.take($width)?;
                let mut raw = [0u8; $width];
                raw.copy_from_slice(b);
                v.push(<$ty>::from_le_bytes(raw));
            }
            ColumnData::$variant(v)
        }};
    }
    let data = match ColumnData::empty(type_id) {
        ColumnData::Boolean(_) => {
            let mut v = Vec::with_capacity(len);
            for _ in 0..len {
                v.push(reader.take_u8()? != 0);
            }
            ColumnData::Boolean(v)
        }
        ColumnData::Int8(_) => fixed!(Int8, i8, 1),
        ColumnData::Int16(_) => fixed!(Int16, i16, 2),
        ColumnData::Int32(_) => fixed!(Int32, i32, 4),
        ColumnData::Int64(_) => fixed!(Int64, i64, 8),
        ColumnData::Int128(_) => fixed!(Int128, i128, 16),
        ColumnData::UInt8(_) => fixed!(UInt8, u8, 1),
        ColumnData::UInt16(_) => fixed!(UInt16, u16, 2),
        ColumnData::UInt32(_) => fixed!(UInt32, u32, 4),
        ColumnData::UInt64(_) => fixed!(UInt64, u64, 8),
        ColumnData::Float32(_) => fixed!(Float32, f32, 4),
        ColumnData::Float64(_) => fixed!(Float64, f64, 8),
        ColumnData::FixedBinary16(_) => {
            let mut v = Vec::with_capacity(len);
            for _ in 0..len {
                let b = reader.take(16)?;
                let mut raw = [0u8; 16];
                raw.copy_from_slice(b);
                v.push(raw);
            }
            ColumnData::FixedBinary16(v)
        }
        ColumnData::Interval(_) => {
            let mut v = Vec::with_capacity(len);
            for _ in 0..len {
                let b = reader.take(16)?;
                let months = i32::from_le_bytes([b[0], b[1], b[2], b[3]]);
                let days = i32::from_le_bytes([b[4], b[5], b[6], b[7]]);
                let nanoseconds =
                    i64::from_le_bytes([b[8], b[9], b[10], b[11], b[12], b[13], b[14], b[15]]);
                v.push(zyron_common::Interval {
                    months,
                    days,
                    nanoseconds,
                });
            }
            ColumnData::Interval(v)
        }
        ColumnData::Utf8(_) => {
            let lengths = read_lengths(reader, len)?;
            let mut v = Vec::with_capacity(len);
            for size in lengths {
                let bytes = reader.take(size)?.to_vec();
                v.push(String::from_utf8(bytes).map_err(|_| {
                    ZyronError::ExecutionError("spill file holds invalid utf8".into())
                })?);
            }
            ColumnData::Utf8(v)
        }
        ColumnData::Binary(_) => {
            let lengths = read_lengths(reader, len)?;
            let mut v = Vec::with_capacity(len);
            for size in lengths {
                v.push(reader.take(size)?.to_vec());
            }
            ColumnData::Binary(v)
        }
    };
    Ok(data)
}

fn read_lengths(reader: &mut SpillReader, count: usize) -> Result<Vec<usize>> {
    let mut out = Vec::with_capacity(count);
    for _ in 0..count {
        out.push(reader.take_u32()? as usize);
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn scratch(name: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("zyron_spill_{}_{name}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).expect("scratch");
        dir
    }

    fn directory(name: &str, quota: u64) -> Arc<SpillDirectory> {
        Arc::new(SpillDirectory::open(&scratch(name), quota).expect("open"))
    }

    /// Every column type survives the trip, values and nulls together.
    #[test]
    fn every_column_type_round_trips() {
        let dir = directory("round_trip", 64 * 1024 * 1024);
        let mut nulls = NullBitmap::none(3);
        nulls.set_null(1);

        let batch = DataBatch::new(vec![
            Column::new(
                ColumnData::Boolean(vec![true, false, true]),
                TypeId::Boolean,
            ),
            Column::new(ColumnData::Int64(vec![-1, 0, i64::MAX]), TypeId::Int64),
            Column::new(
                ColumnData::Int128(vec![i128::MIN, 0, i128::MAX]),
                TypeId::Int128,
            ),
            Column::new(
                ColumnData::Float64(vec![f64::MIN, 0.5, f64::MAX]),
                TypeId::Float64,
            ),
            Column::new(
                ColumnData::Utf8(vec!["".into(), "a longer value".into(), "ünïcodé".into()]),
                TypeId::Text,
            ),
            Column::new(
                ColumnData::Binary(vec![vec![], vec![0xFF; 300], vec![1, 2, 3]]),
                TypeId::Bytea,
            ),
            Column::new(
                ColumnData::FixedBinary16(vec![[0u8; 16], [7u8; 16], [255u8; 16]]),
                TypeId::Uuid,
            ),
            Column {
                data: ColumnData::Interval(vec![
                    zyron_common::Interval {
                        months: -3,
                        days: 4,
                        nanoseconds: -5,
                    },
                    zyron_common::Interval::ZERO,
                    zyron_common::Interval {
                        months: i32::MAX,
                        days: i32::MIN,
                        nanoseconds: i64::MAX,
                    },
                ]),
                nulls: nulls.clone(),
                type_id: TypeId::Interval,
                fractional_digits: Some(9),
            },
        ]);

        let mut writer = dir.create().expect("create");
        writer.write_batch(&batch).expect("write");
        let mut reader = writer.finish().expect("finish");
        let back = reader.read_batch().expect("read").expect("a batch");

        assert_eq!(back.num_rows, batch.num_rows);
        assert_eq!(back.columns.len(), batch.columns.len());
        for (i, (before, after)) in batch.columns.iter().zip(back.columns.iter()).enumerate() {
            assert_eq!(after.type_id, before.type_id, "column {i} type");
            assert_eq!(
                after.fractional_digits, before.fractional_digits,
                "column {i} precision"
            );
            assert_eq!(
                format!("{:?}", after.data),
                format!("{:?}", before.data),
                "column {i} values"
            );
            for row in 0..batch.num_rows {
                assert_eq!(
                    after.nulls.is_null(row),
                    before.nulls.is_null(row),
                    "column {i} row {row} nullness"
                );
            }
        }
        assert!(reader.read_batch().expect("end").is_none());
    }

    /// Batches come back in the order they went in, which is what a merge
    /// depends on.
    #[test]
    fn batches_read_back_in_order() {
        let dir = directory("order", 64 * 1024 * 1024);
        let mut writer = dir.create().expect("create");
        for i in 0..64i64 {
            let batch = DataBatch::new(vec![Column::new(
                ColumnData::Int64(vec![i, i + 1, i + 2]),
                TypeId::Int64,
            )]);
            writer.write_batch(&batch).expect("write");
        }
        let mut reader = writer.finish().expect("finish");
        for i in 0..64i64 {
            let batch = reader.read_batch().expect("read").expect("a batch");
            match &batch.columns[0].data {
                ColumnData::Int64(v) => assert_eq!(v.as_slice(), &[i, i + 1, i + 2]),
                other => panic!("wrong type back: {other:?}"),
            }
        }
        assert!(reader.read_batch().expect("end").is_none());
    }

    /// A batch larger than the write buffer is written in pieces and comes
    /// back whole.
    #[test]
    fn a_batch_larger_than_the_buffer_survives() {
        let dir = directory("large", 64 * 1024 * 1024);
        let rows: Vec<i64> = (0..200_000).collect();
        let batch = DataBatch::new(vec![Column::new(
            ColumnData::Int64(rows.clone()),
            TypeId::Int64,
        )]);
        assert!(batch.approx_bytes() > BUFFER_BYTES as u64 * 10);

        let mut writer = dir.create().expect("create");
        writer.write_batch(&batch).expect("write");
        let mut reader = writer.finish().expect("finish");
        let back = reader.read_batch().expect("read").expect("a batch");
        match &back.columns[0].data {
            ColumnData::Int64(v) => assert_eq!(v.as_slice(), rows.as_slice()),
            other => panic!("wrong type back: {other:?}"),
        }
    }

    /// The file goes when the handle does, on every path out.
    #[test]
    fn the_file_is_removed_when_the_reader_drops() {
        let dir = directory("lifetime", 64 * 1024 * 1024);
        let batch = DataBatch::new(vec![Column::new(
            ColumnData::Int64(vec![1, 2, 3]),
            TypeId::Int64,
        )]);
        let path;
        {
            let mut writer = dir.create().expect("create");
            writer.write_batch(&batch).expect("write");
            let reader = writer.finish().expect("finish");
            path = reader.handle.path().to_path_buf();
            assert!(path.is_file());
            assert!(dir.reserved_bytes() > 0);
        }
        assert!(!path.exists(), "the spill file outlived its reader");
        assert_eq!(dir.reserved_bytes(), 0, "the quota was not given back");
    }

    /// An abandoned writer takes its file with it, which is the cancelled and
    /// errored query path.
    #[test]
    fn an_abandoned_writer_removes_its_file() {
        let dir = directory("abandoned", 64 * 1024 * 1024);
        let batch = DataBatch::new(vec![Column::new(
            ColumnData::Int64(vec![1, 2, 3]),
            TypeId::Int64,
        )]);
        let path;
        {
            let mut writer = dir.create().expect("create");
            writer.write_batch(&batch).expect("write");
            path = writer.handle.path().to_path_buf();
            assert!(path.is_file());
        }
        assert!(!path.exists(), "an abandoned spill file was left behind");
        assert_eq!(dir.reserved_bytes(), 0);
    }

    /// Spilling is bounded. Past the quota the query is told so, rather than
    /// the node filling its disk and stopping every query on it.
    #[test]
    fn the_quota_refuses_rather_than_filling_the_disk() {
        let dir = directory("quota", 4096);
        let batch = DataBatch::new(vec![Column::new(
            ColumnData::Int64((0..4096).collect()),
            TypeId::Int64,
        )]);
        let mut writer = dir.create().expect("create");
        let refused = writer.write_batch(&batch).expect_err("past the quota");
        let text = refused.to_string();
        assert!(text.contains("spill quota"), "{text}");
        assert!(text.contains("4096"), "{text}");
    }

    /// A file left by a process that died is removed when the directory opens,
    /// because nothing will ever read it and the disk is otherwise lost.
    #[test]
    fn leftovers_are_purged_when_the_directory_opens() {
        let root = scratch("purge");
        let stale = root.join(SPILL_DIR);
        std::fs::create_dir_all(&stale).expect("dir");
        let orphan = stale.join("99999_0.spill");
        std::fs::write(&orphan, b"leftover").expect("write");
        let unrelated = stale.join("keep.txt");
        std::fs::write(&unrelated, b"not ours").expect("write");

        let _dir = SpillDirectory::open(&root, 1024).expect("open");
        assert!(!orphan.exists(), "a stale spill file survived startup");
        assert!(
            unrelated.exists(),
            "the purge removed a file that was not a spill file"
        );
    }

    /// An empty batch is a legal thing to write and reads back as one.
    #[test]
    fn an_empty_batch_round_trips() {
        let dir = directory("empty", 1024 * 1024);
        let batch = DataBatch::new(vec![Column::new(ColumnData::Int64(vec![]), TypeId::Int64)]);
        let mut writer = dir.create().expect("create");
        writer.write_batch(&batch).expect("write");
        let mut reader = writer.finish().expect("finish");
        let back = reader.read_batch().expect("read").expect("a batch");
        assert_eq!(back.num_rows, 0);
        assert_eq!(back.columns.len(), 1);
    }
}
