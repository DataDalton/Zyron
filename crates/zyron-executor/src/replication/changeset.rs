//! What a transaction ships to the rest of the group, and how it is written.
//!
//! ## Rows, not statements and not pages
//!
//! A statement re-run on another node is not the same statement: `now()`,
//! `random()` and a sequence draw all answer differently, so replaying SQL
//! diverges. Pages are worse: this engine's buffer pool tracks dirty pages
//! globally rather than per transaction, and one page holds rows from several
//! concurrent writers, so a page image ships uncommitted data.
//!
//! What travels is the rows themselves, in exactly the encoding
//! [`crate::batch::encode_row`] produces and
//! [`crate::batch::decode_tuple_into_builders`] consumes. A follower writes
//! bytes it never has to interpret, so a `BYTEA` column, a `TIMESTAMP(9)` and
//! a `DECIMAL` all cross without a single conversion that could round.
//!
//! ## The layout is written to be read without copying
//!
//! Operators encode straight into the buffer that becomes the log record, and
//! the applier reads row images as slices borrowed from the record it already
//! holds. From the leader's column batch to the follower's heap page there is
//! one copy in each direction and no intermediate structure.
//!
//! ## A large transaction streams
//!
//! Buffering a bulk load until commit would mean holding all of it, so once a
//! transaction passes the chunk size its buffer is sealed and proposed while
//! it is still executing. Followers stage the chunks into an open local
//! transaction and commit it when the chunk carrying [`FLAG_LAST`] arrives.
//! A leader that dies part way leaves those staged transactions behind, and
//! the term-start no-op is what aborts them: every node applies it at the same
//! position, so every node discards the same set.

use std::sync::Arc;

use zyron_catalog::{ColumnEntry, TableEntry};
use zyron_common::{Result, ZyronError};

use crate::batch::{DataBatch, encode_row_into};

/// Format of the payload carried by one `Data` entry.
pub const FORMAT_VERSION: u8 = 1;

/// This chunk completes the transaction, which commits when it applies
pub const FLAG_LAST: u8 = 0x01;
/// This chunk abandons the transaction, discarding every chunk before it
pub const FLAG_ABORT: u8 = 0x02;
/// Nothing may apply alongside this one. Set for schema changes, whose effects
/// every later entry is encoded against
pub const FLAG_BARRIER: u8 = 0x04;

/// version 1, flags 1, reserved 2, chunk_seq 4, node 8, epoch 8, txn 8,
/// barrier 8, timestamp 8
pub const HEADER_LEN: usize = 48;

const OP_INSERT: u8 = 1;
const OP_DELETE: u8 = 2;
const OP_UPDATE: u8 = 3;
const OP_TRUNCATE: u8 = 4;
const OP_LAKE_VERSION: u8 = 5;
const OP_SEQUENCE: u8 = 6;
const OP_DDL: u8 = 10;

/// Written where an index id would go when rows are matched by their whole
/// image instead
const NO_INDEX: u32 = u32::MAX;

/// Which node produced a transaction, and which of its lives.
///
/// The epoch separates a leader's proposals from the same leader's proposals
/// after a restart. Without it a node coming back up would recognise its own
/// pre-crash transactions as ones it still holds open, and finish committing
/// work whose locks and buffers died with the process
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Origin {
    pub node: u64,
    pub epoch: u64,
    pub txn: u64,
}

/// How the applier finds the rows an update or a delete names.
///
/// A key probe is what a table with a primary key or any unique index gets. A
/// whole-row match is what everything else gets, and it is not a fallback so
/// much as the same cost class the leader was already in: a table with no
/// index makes `DELETE` a scan there too
#[derive(Debug, Clone)]
pub enum ReplicaIdentity {
    /// Probe this index with the encoded key
    Key { index_id: u32, columns: Vec<u16> },
    /// Match the whole encoded row
    FullImage,
}

impl ReplicaIdentity {
    /// The identity a table replicates by.
    ///
    /// A primary key is preferred, then any unique index. Uniqueness is what
    /// matters rather than which constraint declared it, because the applier
    /// needs one row per probe and nothing else
    pub fn of(table: &TableEntry, indexes: &zyron_catalog::TableIndexSnapshot) -> Self {
        let primary: Option<&zyron_catalog::ConstraintEntry> = table
            .constraints
            .iter()
            .find(|c| c.constraint_type == zyron_catalog::ConstraintType::PrimaryKey);
        if let Some(pk) = primary {
            let want: Vec<u16> = pk.columns.iter().map(|c| c.0).collect();
            if let Some(spec) = indexes
                .btree
                .iter()
                .find(|s| s.unique && s.columns.iter().map(|c| c.0).eq(want.iter().copied()))
            {
                return ReplicaIdentity::Key {
                    index_id: spec.id.0,
                    columns: want,
                };
            }
        }
        if let Some(spec) = indexes.btree.iter().find(|s| s.unique) {
            return ReplicaIdentity::Key {
                index_id: spec.id.0,
                columns: spec.columns.iter().map(|c| c.0).collect(),
            };
        }
        ReplicaIdentity::FullImage
    }

    #[inline]
    fn index_id(&self) -> u32 {
        match self {
            ReplicaIdentity::Key { index_id, .. } => *index_id,
            ReplicaIdentity::FullImage => NO_INDEX,
        }
    }
}

// ---------------------------------------------------------------------------
// Writing
// ---------------------------------------------------------------------------

#[inline]
fn put_u16(buf: &mut Vec<u8>, v: u16) {
    buf.extend_from_slice(&v.to_le_bytes());
}

#[inline]
fn put_u32(buf: &mut Vec<u8>, v: u32) {
    buf.extend_from_slice(&v.to_le_bytes());
}

#[inline]
fn put_u64(buf: &mut Vec<u8>, v: u64) {
    buf.extend_from_slice(&v.to_le_bytes());
}

#[inline]
fn put_bytes(buf: &mut Vec<u8>, v: &[u8]) {
    put_u32(buf, v.len() as u32);
    buf.extend_from_slice(v);
}

/// Encodes one row straight into `buf` behind its length, so a row never
/// exists as its own allocation on the way to the log
fn put_row(buf: &mut Vec<u8>, batch: &DataBatch, row: usize, columns: &[ColumnEntry]) {
    let at = buf.len();
    buf.extend_from_slice(&0u32.to_le_bytes());
    encode_row_into(buf, batch, row, columns);
    let len = (buf.len() - at - 4) as u32;
    buf[at..at + 4].copy_from_slice(&len.to_le_bytes());
}

/// Where the buffer stood when a savepoint was taken.
///
/// A rollback to the savepoint cuts the buffer back to this, so what was
/// captured after it never reaches the group at all. That is the whole
/// treatment of savepoints in replication: nothing is shipped for the
/// savepoint itself, and a follower only ever sees what survived
struct SavepointMark {
    name: String,
    payload_len: usize,
    ops: usize,
}

/// The bytes of one chunk under construction.
pub struct ChangesetBuffer {
    payload: Vec<u8>,
    ops: usize,
    chunk_seq: u32,
    origin: Origin,
    /// Open savepoints, oldest first. While any is open the buffer must not
    /// stream: a chunk already proposed cannot be cut back
    marks: Vec<SavepointMark>,
}

impl ChangesetBuffer {
    fn new(origin: Origin) -> Self {
        let mut payload = Vec::with_capacity(8 * 1024);
        payload.resize(HEADER_LEN, 0);
        Self {
            payload,
            ops: 0,
            chunk_seq: 0,
            origin,
            marks: Vec::new(),
        }
    }

    #[inline]
    fn is_empty(&self) -> bool {
        self.ops == 0
    }

    #[inline]
    fn len(&self) -> usize {
        self.payload.len()
    }

    /// Stamps the header and hands the bytes over, leaving the buffer ready
    /// for the next chunk of the same transaction
    fn seal(&mut self, flags: u8, barrier_index: u64, timestamp_us: i64) -> Vec<u8> {
        let header = &mut self.payload[..HEADER_LEN];
        header[0] = FORMAT_VERSION;
        header[1] = flags;
        header[2..4].copy_from_slice(&0u16.to_le_bytes());
        header[4..8].copy_from_slice(&self.chunk_seq.to_le_bytes());
        header[8..16].copy_from_slice(&self.origin.node.to_le_bytes());
        header[16..24].copy_from_slice(&self.origin.epoch.to_le_bytes());
        header[24..32].copy_from_slice(&self.origin.txn.to_le_bytes());
        header[32..40].copy_from_slice(&barrier_index.to_le_bytes());
        header[40..48].copy_from_slice(&timestamp_us.to_le_bytes());

        self.chunk_seq += 1;
        self.ops = 0;
        // A commit implicitly releases whatever savepoints are still open
        self.marks.clear();
        let mut next = Vec::with_capacity(self.payload.capacity().min(1 << 20));
        next.resize(HEADER_LEN, 0);
        std::mem::replace(&mut self.payload, next)
    }
}

// ---------------------------------------------------------------------------
// Reading
// ---------------------------------------------------------------------------

/// The fixed part of one chunk.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ChangesetHeader {
    pub flags: u8,
    pub chunk_seq: u32,
    pub origin: Origin,
    /// Everything at or below this index must have applied before this chunk
    /// may. It is the leader's applied index at the moment the transaction
    /// finished acquiring locks, so anything proposed in between held locks of
    /// its own and cannot touch the same rows
    pub barrier_index: u64,
    pub timestamp_us: i64,
}

impl ChangesetHeader {
    #[inline]
    pub fn is_last(&self) -> bool {
        self.flags & FLAG_LAST != 0
    }

    #[inline]
    pub fn is_abort(&self) -> bool {
        self.flags & FLAG_ABORT != 0
    }

    #[inline]
    pub fn is_barrier(&self) -> bool {
        self.flags & FLAG_BARRIER != 0
    }
}

/// One row image and how many rows it stands for.
///
/// A table with no unique index can hold rows that are byte for byte the same,
/// and deleting one of them must delete exactly one. The rows are
/// indistinguishable, so which one is irrelevant, but how many is not
#[derive(Debug, Clone, Copy)]
pub struct RowImage<'a> {
    pub bytes: &'a [u8],
    pub multiplicity: u32,
}

/// One operation read back out of a chunk, borrowing from it.
#[derive(Debug)]
pub enum ChangesetOp<'a> {
    Insert {
        table_id: u32,
        columns: u16,
        rows: Vec<&'a [u8]>,
    },
    Delete {
        table_id: u32,
        columns: u16,
        index_id: Option<u32>,
        rows: Vec<RowImage<'a>>,
    },
    Update {
        table_id: u32,
        columns: u16,
        index_id: Option<u32>,
        rows: Vec<(&'a [u8], &'a [u8])>,
    },
    Truncate {
        table_id: u32,
    },
    LakeVersion {
        table_id: u32,
        version: u64,
        version_file: &'a [u8],
    },
    Sequence {
        sequence_id: u32,
        last_value: i64,
    },
    Ddl {
        sql: &'a str,
        /// Who ran it, so an object it creates is owned by the same user
        /// everywhere
        user: &'a str,
        /// The database it ran in
        database: &'a str,
        /// Schemas it resolved unqualified names against. Without this
        /// `CREATE TABLE t` would mean `public.t` on one node and something
        /// else on another, and the two would never notice
        search_path: Vec<&'a str>,
    },
}

/// Walks the operations in one chunk without copying any of them.
pub struct ChangesetReader<'a> {
    data: &'a [u8],
    at: usize,
}

impl<'a> ChangesetReader<'a> {
    /// Reads the header and positions at the first operation
    pub fn open(data: &'a [u8]) -> Result<(ChangesetHeader, Self)> {
        if data.len() < HEADER_LEN {
            return Err(bad("changeset is shorter than its header"));
        }
        if data[0] != FORMAT_VERSION {
            return Err(bad(&format!(
                "changeset format version {} is not {FORMAT_VERSION}",
                data[0]
            )));
        }
        let header = ChangesetHeader {
            flags: data[1],
            chunk_seq: u32::from_le_bytes([data[4], data[5], data[6], data[7]]),
            origin: Origin {
                node: read_u64(data, 8),
                epoch: read_u64(data, 16),
                txn: read_u64(data, 24),
            },
            barrier_index: read_u64(data, 32),
            timestamp_us: read_u64(data, 40) as i64,
        };
        Ok((
            header,
            Self {
                data,
                at: HEADER_LEN,
            },
        ))
    }

    fn u8(&mut self) -> Result<u8> {
        let v = *self
            .data
            .get(self.at)
            .ok_or_else(|| bad("changeset ends inside an operation tag"))?;
        self.at += 1;
        Ok(v)
    }

    fn u16(&mut self) -> Result<u16> {
        if self.at + 2 > self.data.len() {
            return Err(bad("changeset ends inside a two byte field"));
        }
        let v = u16::from_le_bytes([self.data[self.at], self.data[self.at + 1]]);
        self.at += 2;
        Ok(v)
    }

    fn u32(&mut self) -> Result<u32> {
        if self.at + 4 > self.data.len() {
            return Err(bad("changeset ends inside a four byte field"));
        }
        let v = u32::from_le_bytes([
            self.data[self.at],
            self.data[self.at + 1],
            self.data[self.at + 2],
            self.data[self.at + 3],
        ]);
        self.at += 4;
        Ok(v)
    }

    fn u64(&mut self) -> Result<u64> {
        if self.at + 8 > self.data.len() {
            return Err(bad("changeset ends inside an eight byte field"));
        }
        let v = read_u64(self.data, self.at);
        self.at += 8;
        Ok(v)
    }

    fn bytes(&mut self) -> Result<&'a [u8]> {
        let len = self.u32()? as usize;
        if self.at + len > self.data.len() {
            return Err(bad("changeset ends inside a payload"));
        }
        let out = &self.data[self.at..self.at + len];
        self.at += len;
        Ok(out)
    }

    fn text(&mut self) -> Result<&'a str> {
        let raw = self.bytes()?;
        std::str::from_utf8(raw).map_err(|_| bad("changeset carries text that is not utf8"))
    }

    /// Rows a count field claims, refused when the chunk cannot hold them.
    ///
    /// A corrupt count would otherwise reserve gigabytes before the read that
    /// proves it wrong. Every row costs at least its four byte length prefix
    fn rows_claimed(&self, count: u32) -> Result<usize> {
        let count = count as usize;
        if count.saturating_mul(4) > self.data.len() - self.at {
            return Err(bad(&format!(
                "changeset claims {count} rows that the chunk cannot hold"
            )));
        }
        Ok(count)
    }
}

impl<'a> Iterator for ChangesetReader<'a> {
    type Item = Result<ChangesetOp<'a>>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.at >= self.data.len() {
            return None;
        }
        Some(self.read_op())
    }
}

impl<'a> ChangesetReader<'a> {
    fn read_op(&mut self) -> Result<ChangesetOp<'a>> {
        let tag = self.u8()?;
        match tag {
            OP_INSERT => {
                let table_id = self.u32()?;
                let columns = self.u16()?;
                let count = self.u32()?;
                let count = self.rows_claimed(count)?;
                let mut rows = Vec::with_capacity(count);
                for _ in 0..count {
                    rows.push(self.bytes()?);
                }
                Ok(ChangesetOp::Insert {
                    table_id,
                    columns,
                    rows,
                })
            }
            OP_DELETE => {
                let table_id = self.u32()?;
                let columns = self.u16()?;
                let index_id = self.u32()?;
                let count = self.u32()?;
                let count = self.rows_claimed(count)?;
                let mut rows = Vec::with_capacity(count);
                for _ in 0..count {
                    let bytes = self.bytes()?;
                    let multiplicity = self.u32()?;
                    rows.push(RowImage {
                        bytes,
                        multiplicity,
                    });
                }
                Ok(ChangesetOp::Delete {
                    table_id,
                    columns,
                    index_id: (index_id != NO_INDEX).then_some(index_id),
                    rows,
                })
            }
            OP_UPDATE => {
                let table_id = self.u32()?;
                let columns = self.u16()?;
                let index_id = self.u32()?;
                let count = self.u32()?;
                let count = self.rows_claimed(count)?;
                let mut rows = Vec::with_capacity(count);
                for _ in 0..count {
                    let old = self.bytes()?;
                    let new = self.bytes()?;
                    rows.push((old, new));
                }
                Ok(ChangesetOp::Update {
                    table_id,
                    columns,
                    index_id: (index_id != NO_INDEX).then_some(index_id),
                    rows,
                })
            }
            OP_TRUNCATE => Ok(ChangesetOp::Truncate {
                table_id: self.u32()?,
            }),
            OP_LAKE_VERSION => Ok(ChangesetOp::LakeVersion {
                table_id: self.u32()?,
                version: self.u64()?,
                version_file: self.bytes()?,
            }),
            OP_SEQUENCE => Ok(ChangesetOp::Sequence {
                sequence_id: self.u32()?,
                last_value: self.u64()? as i64,
            }),
            OP_DDL => {
                let sql = self.text()?;
                let user = self.text()?;
                let database = self.text()?;
                let count = self.u32()?;
                let count = self.rows_claimed(count)?;
                let mut search_path = Vec::with_capacity(count);
                for _ in 0..count {
                    search_path.push(self.text()?);
                }
                Ok(ChangesetOp::Ddl {
                    sql,
                    user,
                    database,
                    search_path,
                })
            }
            other => Err(bad(&format!(
                "changeset operation tag {other} is not known to this build"
            ))),
        }
    }
}

#[inline]
fn read_u64(data: &[u8], at: usize) -> u64 {
    u64::from_le_bytes([
        data[at],
        data[at + 1],
        data[at + 2],
        data[at + 3],
        data[at + 4],
        data[at + 5],
        data[at + 6],
        data[at + 7],
    ])
}

fn bad(reason: &str) -> ZyronError {
    ZyronError::EncodingFailed(format!("replication changeset is malformed: {reason}"))
}

// ---------------------------------------------------------------------------
// Capture
// ---------------------------------------------------------------------------

/// What a statement meant where it was written.
///
/// A schema change is deterministic given the same catalog state and the same
/// way of reading names. The second half is this: the same `CREATE TABLE t`
/// makes a different table under a different search path, and two nodes that
/// disagreed about which one would never find out
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StatementContext {
    pub user: String,
    pub database: String,
    pub search_path: Vec<String>,
}

/// One chunk on its way to the log.
pub struct ChangesetChunk {
    pub origin: Origin,
    pub payload: Vec<u8>,
    /// Whether this chunk completes the transaction
    pub last: bool,
}

/// Where sealed chunks go.
///
/// Implemented by the layer that holds the consensus node. Keeping it a trait
/// is what lets the executor build changesets without depending on the
/// consensus crate
pub trait ChangesetSink: Send + Sync {
    /// Queues one chunk for proposal. Returns once it is queued, not once it
    /// is committed, so a large transaction replicates while it is still
    /// running
    fn emit(&self, chunk: ChangesetChunk) -> Result<()>;

    /// The index every entry of this transaction must wait behind, read when
    /// the transaction seals rather than when it started
    fn barrier_index(&self) -> u64;
}

/// The effects of one transaction, accumulating as it runs.
///
/// Held behind an `Arc` by the connection and cloned into every statement's
/// context, so a nested execution appends to the same set as the statement
/// that started it. Operators of one statement can run in parallel, and the
/// rows they touch are disjoint by construction, so the order two of them
/// append in does not change what the transaction did
pub struct TxnChangeset {
    buffer: parking_lot::Mutex<ChangesetBuffer>,
    origin: Origin,
    chunk_bytes: usize,
    sink: Arc<dyn ChangesetSink>,
    /// Set once a chunk has gone, so the transaction cannot be abandoned by
    /// simply dropping its buffer
    streamed: std::sync::atomic::AtomicBool,
    /// Set when nothing may apply alongside this transaction, which is what a
    /// schema change needs: every entry after it is encoded against the schema
    /// it makes
    barrier: std::sync::atomic::AtomicBool,
}

impl TxnChangeset {
    pub fn new(origin: Origin, chunk_bytes: usize, sink: Arc<dyn ChangesetSink>) -> Self {
        Self {
            buffer: parking_lot::Mutex::new(ChangesetBuffer::new(origin)),
            origin,
            chunk_bytes: chunk_bytes.max(64 * 1024),
            sink,
            streamed: std::sync::atomic::AtomicBool::new(false),
            barrier: std::sync::atomic::AtomicBool::new(false),
        }
    }

    #[inline]
    pub fn origin(&self) -> Origin {
        self.origin
    }

    /// Whether anything captured is still standing, which is what decides if
    /// the transaction has to reach consensus at all.
    ///
    /// Read from the buffer rather than kept as a flag, because a rollback to
    /// a savepoint can cut every captured operation back out, and a flag
    /// would keep saying yes about a transaction with nothing left to agree
    pub fn is_dirty(&self) -> bool {
        let buffer = self.buffer.lock();
        buffer.ops > 0 || buffer.chunk_seq > 0
    }

    /// Whether a chunk has already been proposed, so the transaction cannot be
    /// abandoned by simply dropping its buffer
    #[inline]
    pub fn has_streamed(&self) -> bool {
        self.streamed.load(std::sync::atomic::Ordering::Relaxed)
    }

    /// Bytes currently held, for the memory gauge
    pub fn buffered_bytes(&self) -> usize {
        self.buffer.lock().len()
    }

    /// Seals and emits when the buffer has grown past the chunk size.
    ///
    /// Called after every operation, so a bulk load replicates as it runs
    /// instead of arriving as one entry the size of the load
    fn maybe_stream(&self, buffer: &mut ChangesetBuffer) -> Result<()> {
        if buffer.len() < self.chunk_bytes {
            return Ok(());
        }
        // An open savepoint holds the buffer here whatever its size, because
        // a chunk already proposed cannot be cut back by a rollback. The
        // buffer flushes the moment the last mark closes
        if !buffer.marks.is_empty() {
            return Ok(());
        }
        let payload = buffer.seal(0, self.sink.barrier_index(), now_us());
        self.streamed
            .store(true, std::sync::atomic::Ordering::Relaxed);
        self.sink.emit(ChangesetChunk {
            origin: self.origin,
            payload,
            last: false,
        })
    }

    pub fn capture_insert(&self, table: &TableEntry, batch: &DataBatch) -> Result<()> {
        if batch.num_rows == 0 {
            return Ok(());
        }
        let mut buffer = self.buffer.lock();
        let buf = &mut buffer.payload;
        buf.push(OP_INSERT);
        put_u32(buf, table.id.0);
        put_u16(buf, table.columns.len() as u16);
        put_u32(buf, batch.num_rows as u32);
        for row in 0..batch.num_rows {
            put_row(buf, batch, row, &table.columns);
        }
        buffer.ops += 1;
        self.maybe_stream(&mut buffer)
    }

    /// Records rows leaving a table.
    ///
    /// `key_images` is the encoded index key per row when the table has a
    /// replica identity, and the whole row otherwise. Identical images are
    /// folded into one entry carrying a count, which is both smaller and the
    /// only way a duplicate row deletes the right number of times
    pub fn capture_delete(
        &self,
        table: &TableEntry,
        identity: &ReplicaIdentity,
        images: &[Vec<u8>],
    ) -> Result<()> {
        if images.is_empty() {
            return Ok(());
        }
        let folded = fold(images);
        let mut buffer = self.buffer.lock();
        let buf = &mut buffer.payload;
        buf.push(OP_DELETE);
        put_u32(buf, table.id.0);
        put_u16(buf, table.columns.len() as u16);
        put_u32(buf, identity.index_id());
        put_u32(buf, folded.len() as u32);
        for (image, multiplicity) in &folded {
            put_bytes(buf, image);
            put_u32(buf, *multiplicity);
        }
        buffer.ops += 1;
        self.maybe_stream(&mut buffer)
    }

    /// Records rows changing in place.
    ///
    /// Applied as a delete of the old image followed by an insert of the new
    /// one, so this carries both. Reusing those two paths rather than a third
    /// keeps one implementation of index maintenance rather than two, and the
    /// visible result is the same because a row's identity here is its key or
    /// its contents, never the slot it happens to occupy
    pub fn capture_update(
        &self,
        table: &TableEntry,
        identity: &ReplicaIdentity,
        old_images: &[Vec<u8>],
        new_batch: &DataBatch,
    ) -> Result<()> {
        if old_images.is_empty() {
            return Ok(());
        }
        if old_images.len() != new_batch.num_rows {
            return Err(ZyronError::Internal(format!(
                "replication capture was given {} old rows against {} new ones",
                old_images.len(),
                new_batch.num_rows
            )));
        }
        let mut buffer = self.buffer.lock();
        let buf = &mut buffer.payload;
        buf.push(OP_UPDATE);
        put_u32(buf, table.id.0);
        put_u16(buf, table.columns.len() as u16);
        put_u32(buf, identity.index_id());
        put_u32(buf, old_images.len() as u32);
        for (row, old) in old_images.iter().enumerate() {
            put_bytes(buf, old);
            put_row(buf, new_batch, row, &table.columns);
        }
        buffer.ops += 1;
        self.maybe_stream(&mut buffer)
    }

    /// Records a whole table emptying.
    ///
    /// Shipped as itself rather than as a delete per row, because that is what
    /// it costs on the leader and shipping a billion row images to describe an
    /// O(1) operation would be absurd
    pub fn capture_truncate(&self, table_id: u32) -> Result<()> {
        let mut buffer = self.buffer.lock();
        let buf = &mut buffer.payload;
        buf.push(OP_TRUNCATE);
        put_u32(buf, table_id);
        buffer.ops += 1;
        self.maybe_stream(&mut buffer)
    }

    /// Records one lake commit.
    ///
    /// The version file is the whole description of the commit: which data
    /// files entered the table, which left, which predicates were recorded. On
    /// shared storage the data files are already readable by every node, so
    /// this is the entire cost of replicating a lake write however many rows
    /// it added
    pub fn capture_lake_version(
        &self,
        table_id: u32,
        version: u64,
        version_file: &[u8],
    ) -> Result<()> {
        let mut buffer = self.buffer.lock();
        let buf = &mut buffer.payload;
        buf.push(OP_LAKE_VERSION);
        put_u32(buf, table_id);
        put_u64(buf, version);
        put_bytes(buf, version_file);
        buffer.ops += 1;
        self.maybe_stream(&mut buffer)
    }

    /// Records how far a sequence was drawn.
    ///
    /// The values themselves are already in the row images, so a follower
    /// never draws. It still has to move its counter, or the day it is elected
    /// it would hand out numbers that are already in the table
    pub fn capture_sequence(&self, sequence_id: u32, last_value: i64) -> Result<()> {
        let mut buffer = self.buffer.lock();
        let buf = &mut buffer.payload;
        buf.push(OP_SEQUENCE);
        put_u32(buf, sequence_id);
        put_u64(buf, last_value as u64);
        buffer.ops += 1;
        self.maybe_stream(&mut buffer)
    }

    /// Records a schema change as the statement itself.
    ///
    /// A schema change is agreed before it runs and then runs on every node,
    /// this one included, from the same applied position. So the object ids
    /// the catalog hands out are the same everywhere without being shipped,
    /// and the statement is otherwise deterministic: validation, file creation
    /// and index build all happen the same way wherever it runs.
    ///
    /// The chunk is marked as a barrier, because every entry after it is
    /// encoded against the schema it makes
    pub fn capture_ddl(&self, sql: &str, context: &StatementContext) -> Result<()> {
        self.barrier
            .store(true, std::sync::atomic::Ordering::Relaxed);
        let mut buffer = self.buffer.lock();
        let buf = &mut buffer.payload;
        buf.push(OP_DDL);
        put_bytes(buf, sql.as_bytes());
        put_bytes(buf, context.user.as_bytes());
        put_bytes(buf, context.database.as_bytes());
        put_u32(buf, context.search_path.len() as u32);
        for schema in &context.search_path {
            put_bytes(buf, schema.as_bytes());
        }
        buffer.ops += 1;
        // Deliberately never streamed. A statement is one unit: it is handed
        // back to the originating connection whole when its turn comes, and
        // replayed whole when that connection is gone. Streaming it would put
        // the statement in one chunk and the completion in another, and a
        // replay that skipped the first would find nothing in the second. The
        // largest statement fits a single entry with room to spare
        Ok(())
    }

    /// Marks the point a savepoint was taken, so a rollback to it can cut
    /// everything captured after it out of what the group is told.
    ///
    /// From this moment the buffer stops streaming: a chunk already proposed
    /// cannot be cut back. The cost is that a transaction holds what it
    /// writes inside a savepoint scope in memory, which is bounded by the
    /// scope rather than by the transaction
    pub fn mark_savepoint(&self, name: &str) {
        let mut buffer = self.buffer.lock();
        let mark = SavepointMark {
            name: name.to_string(),
            payload_len: buffer.payload.len(),
            ops: buffer.ops,
        };
        buffer.marks.push(mark);
    }

    /// Cuts the buffer back to where the named savepoint was taken.
    ///
    /// The rows the transaction wrote after it have already been reversed on
    /// this node, so cutting them here means the group simply never hears of
    /// them. Marks taken after the named one are dropped, the named one
    /// stays, exactly as the savepoint itself survives a rollback to it.
    ///
    /// Sequence draws inside the cut are cut with it. Their values are in no
    /// row, so a node elected later handing them out again conflicts with
    /// nothing
    pub fn rollback_to_savepoint(&self, name: &str) -> Result<()> {
        let mut buffer = self.buffer.lock();
        let Some(at) = buffer.marks.iter().rposition(|m| m.name == name) else {
            // The engine already refused an unknown savepoint name, so an
            // unmatched mark here means the two went out of step
            return Err(ZyronError::Internal(format!(
                "a rollback names savepoint {name}, which the changeset holds no mark for"
            )));
        };
        let (payload_len, ops) = (buffer.marks[at].payload_len, buffer.marks[at].ops);
        buffer.payload.truncate(payload_len);
        buffer.ops = ops;
        buffer.marks.truncate(at + 1);
        Ok(())
    }

    /// Closes the named savepoint mark and every mark taken after it, keeping
    /// what was captured. The buffer streams again once no mark is open
    pub fn release_savepoint(&self, name: &str) -> Result<()> {
        let mut buffer = self.buffer.lock();
        let Some(at) = buffer.marks.iter().rposition(|m| m.name == name) else {
            return Err(ZyronError::Internal(format!(
                "a release names savepoint {name}, which the changeset holds no mark for"
            )));
        };
        buffer.marks.truncate(at);
        self.maybe_stream(&mut buffer)
    }

    /// Seals the last chunk and hands it over.
    ///
    /// The barrier is read here rather than when the transaction began,
    /// because it is only sound once every lock is held: an entry proposed
    /// between the barrier and this one was still holding its own locks, so it
    /// cannot have touched the same rows and the two may apply in either order
    pub fn seal(&self, barrier_index: u64) -> Option<ChangesetChunk> {
        let mut buffer = self.buffer.lock();
        if buffer.is_empty() && buffer.chunk_seq == 0 {
            return None;
        }
        let mut flags = FLAG_LAST;
        if self.barrier.load(std::sync::atomic::Ordering::Relaxed) {
            flags |= FLAG_BARRIER;
        }
        let payload = buffer.seal(flags, barrier_index, now_us());
        Some(ChangesetChunk {
            origin: self.origin,
            payload,
            last: true,
        })
    }

    /// Builds the chunk that abandons a transaction whose earlier chunks are
    /// already staged on every follower
    pub fn seal_abort(&self, barrier_index: u64) -> Option<ChangesetChunk> {
        if !self.has_streamed() {
            return None;
        }
        let mut buffer = self.buffer.lock();
        buffer.payload.truncate(HEADER_LEN);
        buffer.ops = 0;
        let payload = buffer.seal(FLAG_LAST | FLAG_ABORT, barrier_index, now_us());
        Some(ChangesetChunk {
            origin: self.origin,
            payload,
            last: true,
        })
    }
}

/// Collapses runs of identical images into one entry with a count.
///
/// Images arrive in the order the scan produced them, and rows that encode
/// identically are adjacent only by luck, so this is a map rather than a run
/// length pass. The common case is every image distinct, which costs one hash
/// per row and no allocation past the map itself
fn fold(images: &[Vec<u8>]) -> Vec<(&[u8], u32)> {
    let mut out: Vec<(&[u8], u32)> = Vec::with_capacity(images.len());
    let mut seen: hashbrown::HashMap<&[u8], usize> =
        hashbrown::HashMap::with_capacity(images.len());
    for image in images {
        match seen.entry_ref(image.as_slice()) {
            hashbrown::hash_map::EntryRef::Occupied(slot) => {
                out[*slot.get()].1 += 1;
            }
            hashbrown::hash_map::EntryRef::Vacant(slot) => {
                slot.insert(out.len());
                out.push((image.as_slice(), 1));
            }
        }
    }
    out
}

fn now_us() -> i64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_micros() as i64
}

#[cfg(test)]
mod tests {
    use super::*;

    struct NullSink;

    impl ChangesetSink for NullSink {
        fn emit(&self, _chunk: ChangesetChunk) -> Result<()> {
            Ok(())
        }

        fn barrier_index(&self) -> u64 {
            0
        }
    }

    struct CollectingSink {
        chunks: parking_lot::Mutex<Vec<ChangesetChunk>>,
    }

    impl ChangesetSink for CollectingSink {
        fn emit(&self, chunk: ChangesetChunk) -> Result<()> {
            self.chunks.lock().push(chunk);
            Ok(())
        }

        fn barrier_index(&self) -> u64 {
            7
        }
    }

    fn origin() -> Origin {
        Origin {
            node: 11,
            epoch: 2,
            txn: 500,
        }
    }

    #[test]
    fn a_sealed_chunk_reads_back_with_its_header() {
        let set = TxnChangeset::new(origin(), 1 << 20, Arc::new(NullSink));
        set.capture_truncate(9).expect("truncate");
        set.capture_sequence(3, -42).expect("sequence");
        let chunk = set.seal(1234).expect("chunk");

        let (header, reader) = ChangesetReader::open(&chunk.payload).expect("open");
        assert_eq!(header.origin, origin());
        assert_eq!(header.barrier_index, 1234);
        assert!(header.is_last());
        assert!(!header.is_abort());
        assert!(!header.is_barrier());
        assert_eq!(header.chunk_seq, 0);

        let ops: Vec<_> = reader.map(|op| op.expect("op")).collect();
        assert_eq!(ops.len(), 2);
        match &ops[0] {
            ChangesetOp::Truncate { table_id } => assert_eq!(*table_id, 9),
            other => panic!("expected a truncate, got {other:?}"),
        }
        match &ops[1] {
            ChangesetOp::Sequence {
                sequence_id,
                last_value,
            } => {
                assert_eq!(*sequence_id, 3);
                assert_eq!(*last_value, -42);
            }
            other => panic!("expected a sequence, got {other:?}"),
        }
    }

    #[test]
    fn identical_images_fold_into_one_entry_with_a_count() {
        let images = [
            b"a".to_vec(),
            b"b".to_vec(),
            b"a".to_vec(),
            b"a".to_vec(),
            b"c".to_vec(),
        ];
        let folded = fold(&images);
        assert_eq!(folded.len(), 3);
        assert_eq!(folded[0], (&b"a"[..], 3));
        assert_eq!(folded[1], (&b"b"[..], 1));
        assert_eq!(folded[2], (&b"c"[..], 1));
    }

    /// A rollback to a savepoint cuts everything captured after it out of
    /// the buffer, so the group never hears of rows the transaction unwrote
    #[test]
    fn a_rollback_to_a_savepoint_cuts_the_buffer_back() {
        let set = TxnChangeset::new(origin(), 1 << 20, Arc::new(NullSink));
        set.capture_truncate(1).expect("keep");
        set.mark_savepoint("s1");
        set.capture_truncate(2).expect("discard");
        set.capture_sequence(3, 30).expect("discard");
        set.rollback_to_savepoint("s1").expect("rollback");
        set.capture_truncate(4).expect("keep");

        let chunk = set.seal(0).expect("chunk");
        let (_, reader) = ChangesetReader::open(&chunk.payload).expect("open");
        let tables: Vec<u32> = reader
            .map(|op| match op.expect("op") {
                ChangesetOp::Truncate { table_id } => table_id,
                other => panic!("expected a truncate, got {other:?}"),
            })
            .collect();
        assert_eq!(tables, vec![1, 4]);
    }

    /// The savepoint itself survives a rollback to it, the newest mark with
    /// a name shadows an older one, and a release re-exposes the older mark
    /// the way the engine's own savepoint stack does
    #[test]
    fn savepoint_marks_shadow_and_release_like_the_engine() {
        let set = TxnChangeset::new(origin(), 1 << 20, Arc::new(NullSink));
        set.capture_truncate(1).expect("capture");
        set.mark_savepoint("s");
        set.capture_truncate(2).expect("capture");
        set.mark_savepoint("s");
        set.capture_truncate(3).expect("capture");
        // The newest mark with the name wins, so table 3 goes, 1 and 2 stay
        set.rollback_to_savepoint("s").expect("rollback");
        set.capture_truncate(4).expect("capture");
        // The release drops the newest mark, re-exposing the older one
        set.release_savepoint("s").expect("release");
        set.rollback_to_savepoint("s")
            .expect("the older shadowed mark stands again");

        let chunk = set.seal(0).expect("chunk");
        let (_, reader) = ChangesetReader::open(&chunk.payload).expect("open");
        let tables: Vec<u32> = reader
            .map(|op| match op.expect("op") {
                ChangesetOp::Truncate { table_id } => table_id,
                other => panic!("expected a truncate, got {other:?}"),
            })
            .collect();
        assert_eq!(tables, vec![1]);
        assert!(
            set.rollback_to_savepoint("s").is_err(),
            "sealing the transaction implicitly released every mark"
        );
    }

    /// A transaction whose captures were all cut back has nothing to agree,
    /// and reports so, rather than a flag insisting it is still dirty
    #[test]
    fn a_fully_rolled_back_transaction_is_not_dirty() {
        let set = TxnChangeset::new(origin(), 1 << 20, Arc::new(NullSink));
        set.mark_savepoint("s");
        set.capture_truncate(1).expect("capture");
        assert!(set.is_dirty());
        set.rollback_to_savepoint("s").expect("rollback");
        assert!(!set.is_dirty());
        assert!(set.seal(0).is_none(), "nothing survived to seal");
    }

    /// An open savepoint holds the buffer however large it grows, because a
    /// chunk already proposed cannot be cut back. The release flushes it
    #[test]
    fn an_open_savepoint_suppresses_streaming_until_released() {
        let sink = Arc::new(CollectingSink {
            chunks: parking_lot::Mutex::new(Vec::new()),
        });
        let set = TxnChangeset::new(origin(), 64 * 1024, sink.clone());
        set.mark_savepoint("s");
        for version in 0..200u64 {
            set.capture_lake_version(1, version, &[7u8; 1024])
                .expect("lake version");
        }
        assert!(
            !set.has_streamed(),
            "a chunk went out that a rollback could no longer cut back"
        );
        set.release_savepoint("s").expect("release");
        assert!(
            set.has_streamed(),
            "the release did not flush the buffer past the chunk size"
        );
        assert!(!sink.chunks.lock().is_empty());
    }

    #[test]
    fn a_transaction_past_the_chunk_size_streams_while_it_runs() {
        let sink = Arc::new(CollectingSink {
            chunks: parking_lot::Mutex::new(Vec::new()),
        });
        // Small enough that a handful of operations crosses it
        let set = TxnChangeset::new(origin(), 64 * 1024, sink.clone());
        assert!(!set.has_streamed());
        // Four hundred kilobytes of lake metadata against a sixty four
        // kilobyte chunk, so several chunks go before the transaction ends
        for version in 0..400u64 {
            set.capture_lake_version(1, version, &[7u8; 1024])
                .expect("lake version");
        }
        assert!(
            set.has_streamed(),
            "a transaction past the chunk size must not wait for commit"
        );
        let streamed = sink.chunks.lock().len();
        assert!(streamed >= 1, "no chunk was emitted");

        let last = set.seal(99).expect("final chunk");
        let (header, _) = ChangesetReader::open(&last.payload).expect("open");
        assert!(header.is_last());
        assert_eq!(
            header.chunk_seq, streamed as u32,
            "chunks must be numbered without a gap"
        );
        assert_eq!(header.barrier_index, 99);

        // Every chunk before it carries the barrier the sink reported and is
        // not marked final
        for chunk in sink.chunks.lock().iter() {
            let (header, _) = ChangesetReader::open(&chunk.payload).expect("open");
            assert!(!header.is_last());
            assert_eq!(header.barrier_index, 7);
        }
    }

    #[test]
    fn an_abort_only_exists_once_something_has_been_staged() {
        let sink = Arc::new(CollectingSink {
            chunks: parking_lot::Mutex::new(Vec::new()),
        });
        let set = TxnChangeset::new(origin(), 1 << 30, sink.clone());
        set.capture_truncate(1).expect("truncate");
        assert!(
            set.seal_abort(5).is_none(),
            "a transaction that never streamed is abandoned by dropping it"
        );

        let set = TxnChangeset::new(origin(), 64 * 1024, sink);
        for version in 0..400u64 {
            set.capture_lake_version(1, version, &[7u8; 1024])
                .expect("lake version");
        }
        let abort = set.seal_abort(5).expect("abort chunk");
        let (header, mut reader) = ChangesetReader::open(&abort.payload).expect("open");
        assert!(header.is_abort() && header.is_last());
        assert!(reader.next().is_none(), "an abort carries no operations");
    }

    #[test]
    fn a_truncated_chunk_is_refused_rather_than_read_short() {
        let set = TxnChangeset::new(origin(), 1 << 20, Arc::new(NullSink));
        set.capture_lake_version(4, 12, &[9u8; 64]).expect("lake");
        let chunk = set.seal(0).expect("chunk");

        // A chunk that is only a header carries no operations and is valid,
        // so the interesting cuts start one byte into the first one
        for cut in HEADER_LEN + 1..chunk.payload.len() {
            let (_, reader) = ChangesetReader::open(&chunk.payload[..cut]).expect("header");
            let outcome: Result<Vec<_>> = reader.collect();
            assert!(
                outcome.is_err(),
                "a chunk cut at {cut} decoded as though it were whole"
            );
        }

        let (_, reader) = ChangesetReader::open(&chunk.payload).expect("header");
        let ops: Vec<_> = reader.map(|op| op.expect("op")).collect();
        match &ops[0] {
            ChangesetOp::LakeVersion {
                table_id,
                version,
                version_file,
            } => {
                assert_eq!(*table_id, 4);
                assert_eq!(*version, 12);
                assert_eq!(*version_file, &[9u8; 64][..]);
            }
            other => panic!("expected a lake version, got {other:?}"),
        }
    }

    #[test]
    fn a_wrong_format_version_is_refused() {
        let set = TxnChangeset::new(origin(), 1 << 20, Arc::new(NullSink));
        set.capture_truncate(1).expect("truncate");
        let mut chunk = set.seal(0).expect("chunk");
        chunk.payload[0] = FORMAT_VERSION + 1;
        assert!(ChangesetReader::open(&chunk.payload).is_err());
    }

    #[test]
    fn a_row_count_larger_than_the_chunk_is_refused() {
        let mut payload = vec![0u8; HEADER_LEN];
        payload[0] = FORMAT_VERSION;
        payload.push(OP_INSERT);
        payload.extend_from_slice(&1u32.to_le_bytes());
        payload.extend_from_slice(&1u16.to_le_bytes());
        payload.extend_from_slice(&u32::MAX.to_le_bytes());
        let (_, mut reader) = ChangesetReader::open(&payload).expect("header");
        assert!(reader.next().expect("an op").is_err());
    }
}
