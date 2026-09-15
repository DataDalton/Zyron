//! The commit hash chain of a verifiable table.
//!
//! A verified table carries one chain entry per committing transaction that
//! wrote to it. The entry names the transaction's rows by their hash and
//! links to the entry before it, so the sequence of entries states what the
//! table held at every commit and any later edit of a stored row, removal of
//! a row, or removal of a whole commit contradicts it.
//!
//! ## Where the work happens
//!
//! Hashing the rows is the expensive half and it runs while the statement
//! writes, on whatever thread the statement runs on, into a per-transaction
//! accumulator ([`PendingChainWrites`]). Linking is the serial half and it
//! runs at commit under the table's chain lock: read the head, hash the
//! entry's fields, push the record. Sixteen writers therefore hash in
//! parallel and queue only for the link, and the link touches no device.
//!
//! ## What a chain entry costs
//!
//! [`CHAIN_RECORD_LEN`] bytes on disk per commit, a fixed record with no
//! framing of its own. `prev_hash` is not stored: it is the entry before it,
//! and the genesis entry's is zero. The algorithm is stored per entry rather
//! than read from the table, so an entry states what hashed it.
//!
//! ## What the link covers
//!
//! `entry_hash` is over every field that states what the commit wrote and
//! where it stands in the chain. It is not over `txn_id`. The transaction id
//! is the stamp a member's own rows carry and so the key a verification on
//! that member reads them back by, and every member of a group stamps its
//! rows with an id of its own. Leaving it out of the link is what lets every
//! member hold the same chain over the same rows, and an entry whose id was
//! edited still fails: the rows read back by the edited id do not hash to
//! what the entry recorded.
//!
//! ## Canonical encoding
//!
//! A row is hashed as the schema epoch it was written under, its length,
//! then its stored bytes. The epoch is part of what is hashed, so a column
//! added later mints a new epoch and leaves every earlier entry verifiable
//! against the rows it covered. A commit's rows are hashed in the order they
//! are stored, which the write path keeps equal to the order they were
//! written by giving each transaction its own insertion cursor.
//!
//! ## The genesis entry
//!
//! A table that already holds rows when it becomes verifiable gets one entry
//! over them as a set. Its `rows_hash` is over the ascending sorted list of
//! the rows' digests rather than over the rows in stored order, because the
//! members of a group store the same rows in different orders and the set
//! is what they share. The entry names the transaction id fence the table
//! was chained at: every row stamped below it is in the set, every
//! transaction at or above it is chained on its own.

pub mod anchor;

use std::collections::{BinaryHeap, HashMap, VecDeque};
use std::fs::{File, OpenOptions};
use std::io::{BufReader, BufWriter, Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::AtomicU32;

use parking_lot::{Mutex, RwLock};
use sha2::{Digest, Sha256};
use zyron_common::format::envelope;
use zyron_common::format::{FormatKind, FormatVersion};
use zyron_common::{Result, ZyronError};

/// Version the chain file is written at
pub const CHAIN_FORMAT_VERSION: FormatVersion = FormatVersion::V1;

/// Bytes one chain entry takes on disk.
///
/// A fixed record so a chain of n commits is exactly n records long, the
/// head is the last record, and the storage a table's chain costs is its
/// commit count times this
pub const CHAIN_RECORD_LEN: usize = 96;

/// Bytes of the envelope header the chain file opens with
const CHAIN_HEADER_LEN: usize = envelope::ENVELOPE_HEADER_LEN;

/// A hash as the chain carries it
pub type ChainHash = [u8; 32];

/// The hash a genesis entry links to, which no entry produces
pub const NO_PREVIOUS: ChainHash = [0u8; 32];

/// The registered scheme a chain links with when a table becomes verified
pub const DEFAULT_CHAIN_ALGORITHM: u16 = 4;

/// The scheme name the default algorithm's tag resolves to
pub const DEFAULT_CHAIN_ALGORITHM_NAME: &str = "SHA-256";

/// Record flag: the entry covers the rows the table held when it became
/// verifiable, as a set, and names the transaction id fence rather than a
/// transaction
const ENTRY_FLAG_GENESIS: u8 = 1;

/// Records a chain file write buffers before they reach the device.
///
/// A committed entry lands in memory under the chain lock and the buffer is
/// written when it fills, when the chain is read, and when it is flushed.
/// The log is what makes an entry durable, so nothing waits on this
const WRITE_BUFFER_RECORDS: usize = 512;

/// Records one bulk read of a chain takes at a time, so a walk over a chain
/// of millions of entries holds a bounded window of them
pub const READ_WINDOW_RECORDS: u64 = 8192;

/// Commits one pass over a table hashes the rows of at a time under a full
/// verification. Bounds the hashers held in memory, so a chain longer than
/// this is read back in this many commits per pass over the table
pub const REHASH_WINDOW_COMMITS: u64 = 1 << 20;

/// Digests the genesis set hasher holds in memory before it sorts them and
/// writes the run to a file. A set larger than this is sorted in runs and
/// merged, so the memory it takes is bounded whatever the table holds
const SET_RUN_DIGESTS: usize = 4 << 20;

// ---------------------------------------------------------------------------
// One entry
// ---------------------------------------------------------------------------

/// What a commit states about itself, which the chain links onto its head
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CommitFields {
    /// The transaction whose stamp the rows carry, or for a genesis entry
    /// the transaction id fence the table was chained at
    pub txn_id: u64,
    /// Over the canonical encoding of every row the commit wrote, in the
    /// order they are stored, or for a genesis entry over the sorted
    /// digests of the rows it covers
    pub rows_hash: ChainHash,
    pub row_count: u64,
    /// Microseconds since the epoch
    pub commit_ts: i64,
    pub algorithm_id: u16,
    /// Whether the entry covers the rows the table held when it became
    /// verifiable, as a set
    pub genesis: bool,
}

/// One committing transaction's entry in a table's chain.
///
/// `prev_hash` is carried rather than stored: a walk holds the entry before
/// it and a genesis entry links to [`NO_PREVIOUS`]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CommitHash {
    pub table_id: u32,
    /// The chain position, counted from zero
    pub sequence: u64,
    /// The log position the commit's rows were written at, which on a
    /// member of a group is the index of the entry that carried them
    pub commit_version: u64,
    /// The transaction that wrote them, which is the stamp its rows carry
    /// and so what a verification reads them back by. For a genesis entry
    /// the transaction id fence: every row stamped below it is covered
    pub txn_id: u64,
    pub prev_hash: ChainHash,
    /// Over the canonical encoding of every row the commit wrote, in the
    /// order they are stored
    pub rows_hash: ChainHash,
    /// Over prev_hash, table_id, commit_version, rows_hash, row_count,
    /// commit_ts, algorithm_id and the record flags
    pub entry_hash: ChainHash,
    pub row_count: u64,
    /// Microseconds since the epoch
    pub commit_ts: i64,
    pub algorithm_id: u16,
    /// Whether the entry is the genesis over the rows the table already held
    pub genesis: bool,
}

impl CommitHash {
    /// Builds the entry that follows `prev`, computing its link.
    pub fn link(
        table_id: u32,
        sequence: u64,
        prev_hash: ChainHash,
        commit_version: u64,
        fields: CommitFields,
    ) -> Self {
        let mut entry = Self {
            table_id,
            sequence,
            commit_version,
            txn_id: fields.txn_id,
            prev_hash,
            rows_hash: fields.rows_hash,
            entry_hash: NO_PREVIOUS,
            row_count: fields.row_count,
            commit_ts: fields.commit_ts,
            algorithm_id: fields.algorithm_id,
            genesis: fields.genesis,
        };
        entry.entry_hash = entry.compute_entry_hash();
        entry
    }

    /// Recomputes the link over what the entry states, which is what a walk
    /// compares against the stored `entry_hash`.
    ///
    /// `txn_id` is not covered. It is the key the rows are read back by on
    /// the member that holds them, and each member stamps its rows with an
    /// id of its own, so the link is over what the rows are rather than
    /// over where one member keeps them
    pub fn compute_entry_hash(&self) -> ChainHash {
        sha256_of(&[
            &self.prev_hash,
            &self.table_id.to_le_bytes(),
            &self.commit_version.to_le_bytes(),
            &self.rows_hash,
            &self.row_count.to_le_bytes(),
            &self.commit_ts.to_le_bytes(),
            &self.algorithm_id.to_le_bytes(),
            &[self.flags()],
        ])
    }

    fn flags(&self) -> u8 {
        if self.genesis { ENTRY_FLAG_GENESIS } else { 0 }
    }

    /// The fields the entry was linked from
    pub fn fields(&self) -> CommitFields {
        CommitFields {
            txn_id: self.txn_id,
            rows_hash: self.rows_hash,
            row_count: self.row_count,
            commit_ts: self.commit_ts,
            algorithm_id: self.algorithm_id,
            genesis: self.genesis,
        }
    }

    /// The fixed record the chain file holds.
    ///
    /// Layout: commit_version, commit_ts, txn_id, row_count, algorithm_id,
    /// the record flags, one reserved byte that keeps the hashes aligned,
    /// then rows_hash and entry_hash
    pub fn encode(&self) -> [u8; CHAIN_RECORD_LEN] {
        let mut out = [0u8; CHAIN_RECORD_LEN];
        self.encode_into(&mut out);
        out
    }

    /// Writes the fixed record into a caller's buffer
    #[inline]
    pub fn encode_into(&self, out: &mut [u8; CHAIN_RECORD_LEN]) {
        out[0..8].copy_from_slice(&self.commit_version.to_le_bytes());
        out[8..16].copy_from_slice(&self.commit_ts.to_le_bytes());
        out[16..24].copy_from_slice(&self.txn_id.to_le_bytes());
        out[24..28].copy_from_slice(&(self.row_count.min(u32::MAX as u64) as u32).to_le_bytes());
        out[28..30].copy_from_slice(&self.algorithm_id.to_le_bytes());
        out[30] = self.flags();
        out[31] = 0;
        out[32..64].copy_from_slice(&self.rows_hash);
        out[64..96].copy_from_slice(&self.entry_hash);
    }

    /// Reads one record back, given where it sits and what precedes it.
    pub fn decode(
        table_id: u32,
        sequence: u64,
        prev_hash: ChainHash,
        record: &[u8; CHAIN_RECORD_LEN],
    ) -> Self {
        let word = |at: usize| -> u64 {
            let mut buf = [0u8; 8];
            buf.copy_from_slice(&record[at..at + 8]);
            u64::from_le_bytes(buf)
        };
        let mut rows_hash = [0u8; 32];
        rows_hash.copy_from_slice(&record[32..64]);
        let mut entry_hash = [0u8; 32];
        entry_hash.copy_from_slice(&record[64..96]);
        Self {
            table_id,
            sequence,
            commit_version: word(0),
            txn_id: word(16),
            prev_hash,
            rows_hash,
            entry_hash,
            row_count: u32::from_le_bytes([record[24], record[25], record[26], record[27]]) as u64,
            commit_ts: word(8) as i64,
            algorithm_id: u16::from_le_bytes([record[28], record[29]]),
            genesis: record[30] & ENTRY_FLAG_GENESIS != 0,
        }
    }
}

/// The most rows one commit may cover, which is what the fixed record's row
/// count holds. A commit above it is refused rather than recorded short
pub const MAX_COMMIT_ROWS: u64 = u32::MAX as u64;

/// The SHA-256 initial state, as the standard fixes it
const SHA256_INITIAL: [u32; 8] = [
    0x6a09_e667,
    0xbb67_ae85,
    0x3c6e_f372,
    0xa54f_f53a,
    0x510e_527f,
    0x9b05_688c,
    0x1f83_d9ab,
    0x5be0_cd19,
];

thread_local! {
    /// The state a digest is compressed into, one heap slot per thread.
    ///
    /// The compression stores its result with vector stores and the digest
    /// is read back from it a word at a time. On the processors this runs
    /// on, that read costs two orders of magnitude more when the state is a
    /// stack local than when it sits in a heap slot, so every digest this
    /// module produces is compressed into this slot rather than into a
    /// local of its own
    static DIGEST_STATE: std::cell::RefCell<Box<[u32; 8]>> =
        std::cell::RefCell::new(Box::new(SHA256_INITIAL));
}

/// SHA-256 over the concatenation of `parts`, driving the standard padding
/// onto the block compression directly.
///
/// The digest is exactly what the streaming hasher produces over the same
/// bytes. Whole blocks go to the compression as they fill, so a part of any
/// length costs one compression per block and no copy beyond the block in
/// hand, and the tail takes the one or two padded blocks the length calls
/// for
pub fn sha256_of(parts: &[&[u8]]) -> ChainHash {
    DIGEST_STATE.with(|slot| {
        let mut state = slot.borrow_mut();
        **state = SHA256_INITIAL;
        sha256_into(&mut state, parts)
    })
}

fn sha256_into(state: &mut [u32; 8], parts: &[&[u8]]) -> ChainHash {
    let mut block =
        sha2::digest::generic_array::GenericArray::<u8, sha2::digest::typenum::U64>::default();
    let mut filled = 0usize;
    let mut total = 0u64;
    for part in parts {
        total += part.len() as u64;
        let mut rest: &[u8] = part;
        if filled > 0 {
            let take = rest.len().min(64 - filled);
            block[filled..filled + take].copy_from_slice(&rest[..take]);
            filled += take;
            rest = &rest[take..];
            if filled == 64 {
                sha2::compress256(state, std::slice::from_ref(&block));
                filled = 0;
            }
        }
        let whole = rest.len() / 64 * 64;
        if whole > 0 {
            for chunk in rest[..whole].chunks_exact(64) {
                let full = sha2::digest::generic_array::GenericArray::from_slice(chunk);
                sha2::compress256(state, std::slice::from_ref(full));
            }
            rest = &rest[whole..];
        }
        if !rest.is_empty() {
            block[..rest.len()].copy_from_slice(rest);
            filled = rest.len();
        }
    }
    // The delimiter, zeros to the length field, then the bit length. A tail
    // past the space the length needs takes one more block
    block[filled] = 0x80;
    for byte in &mut block[filled + 1..] {
        *byte = 0;
    }
    if filled + 1 > 56 {
        sha2::compress256(state, std::slice::from_ref(&block));
        for byte in block.iter_mut() {
            *byte = 0;
        }
    }
    block[56..64].copy_from_slice(&(total * 8).to_be_bytes());
    sha2::compress256(state, std::slice::from_ref(&block));
    let mut out = [0u8; 32];
    for (chunk, word) in out.chunks_exact_mut(4).zip(state.iter()) {
        chunk.copy_from_slice(&word.to_be_bytes());
    }
    out
}

const HEX_DIGITS: &[u8; 16] = b"0123456789abcdef";

/// Renders a hash the way a view, an error and an exported anchor spell it.
pub fn hex(hash: &ChainHash) -> String {
    let mut out = Vec::with_capacity(64);
    for byte in hash {
        out.push(HEX_DIGITS[(byte >> 4) as usize]);
        out.push(HEX_DIGITS[(byte & 0x0f) as usize]);
    }
    // Every byte pushed is an ASCII digit
    String::from_utf8(out).unwrap_or_default()
}

/// Reads a hash back from the spelling `hex` produces.
pub fn from_hex(text: &str) -> Option<ChainHash> {
    let bytes = text.as_bytes();
    if bytes.len() != 64 {
        return None;
    }
    let mut out = [0u8; 32];
    for (i, pair) in bytes.chunks_exact(2).enumerate() {
        let hi = (pair[0] as char).to_digit(16)?;
        let lo = (pair[1] as char).to_digit(16)?;
        out[i] = ((hi << 4) | lo) as u8;
    }
    Some(out)
}

// ---------------------------------------------------------------------------
// Hashing the rows a commit wrote
// ---------------------------------------------------------------------------

/// The running hash of one table's rows inside one transaction.
///
/// Rows go in as they are written, so a commit of a million rows holds one
/// hash state rather than a million rows. The hash is over the canonical
/// encoding: each row's schema epoch, its length, then its stored bytes
#[derive(Debug)]
pub struct RowsHasher {
    hasher: Sha256,
    rows: u64,
    bytes: u64,
}

impl Default for RowsHasher {
    fn default() -> Self {
        Self::new()
    }
}

impl RowsHasher {
    pub fn new() -> Self {
        Self {
            hasher: Sha256::new(),
            rows: 0,
            bytes: 0,
        }
    }

    /// The six bytes that prefix a row in the canonical encoding
    #[inline]
    fn prefix(schema_epoch: u16, row: &[u8]) -> [u8; 6] {
        let epoch = schema_epoch.to_le_bytes();
        let len = (row.len() as u32).to_le_bytes();
        [epoch[0], epoch[1], len[0], len[1], len[2], len[3]]
    }

    /// Takes one row in the layout its epoch records.
    #[inline]
    pub fn row(&mut self, schema_epoch: u16, row: &[u8]) {
        self.hasher.update(Self::prefix(schema_epoch, row));
        self.hasher.update(row);
        self.rows += 1;
        self.bytes += row.len() as u64;
    }

    /// The digest of one row on its own, which is what the genesis set is
    /// sorted and hashed over
    #[inline]
    pub fn row_digest(schema_epoch: u16, row: &[u8]) -> ChainHash {
        sha256_of(&[&Self::prefix(schema_epoch, row), row])
    }

    pub fn rows(&self) -> u64 {
        self.rows
    }

    pub fn bytes(&self) -> u64 {
        self.bytes
    }

    /// The hash of everything taken so far.
    pub fn finish(self) -> ChainHash {
        self.hasher.finalize().into()
    }
}

/// Hashes a run of rows in one call, for a verify that recomputes what a
/// commit covered.
pub fn hash_rows<'a>(rows: impl Iterator<Item = (u16, &'a [u8])>) -> (ChainHash, u64, u64) {
    let mut hasher = RowsHasher::new();
    for (epoch, row) in rows {
        hasher.row(epoch, row);
    }
    let (count, bytes) = (hasher.rows(), hasher.bytes());
    (hasher.finish(), count, bytes)
}

/// The hash of a set of rows, whatever order they arrive in.
///
/// Each row's digest is collected and the hash is over the ascending sorted
/// list of digests, so two members of a group that store the same rows in
/// different orders produce the same hash. The digests are held in memory
/// up to [`SET_RUN_DIGESTS`], then sorted and written as a run, and the runs
/// are merged at the end, so the memory a set of any size takes is bounded
pub struct SetHasher {
    spill_dir: PathBuf,
    /// Digests held in memory before a run is written
    run_capacity: usize,
    held: Vec<ChainHash>,
    runs: Vec<PathBuf>,
    rows: u64,
    bytes: u64,
}

impl SetHasher {
    /// Starts a set whose runs, if it needs any, are written under
    /// `spill_dir`
    pub fn new(spill_dir: impl Into<PathBuf>) -> Self {
        Self::with_run_capacity(spill_dir, SET_RUN_DIGESTS)
    }

    /// Starts a set that holds `run_capacity` digests in memory before it
    /// sorts them and writes the run
    pub fn with_run_capacity(spill_dir: impl Into<PathBuf>, run_capacity: usize) -> Self {
        Self {
            spill_dir: spill_dir.into(),
            run_capacity: run_capacity.max(1),
            held: Vec::new(),
            runs: Vec::new(),
            rows: 0,
            bytes: 0,
        }
    }

    /// Takes one row into the set.
    #[inline]
    pub fn row(&mut self, schema_epoch: u16, row: &[u8]) -> Result<()> {
        self.held.push(RowsHasher::row_digest(schema_epoch, row));
        self.rows += 1;
        self.bytes += row.len() as u64;
        if self.held.len() >= self.run_capacity {
            self.spill()?;
        }
        Ok(())
    }

    pub fn rows(&self) -> u64 {
        self.rows
    }

    pub fn bytes(&self) -> u64 {
        self.bytes
    }

    fn spill(&mut self) -> Result<()> {
        self.held.sort_unstable();
        std::fs::create_dir_all(&self.spill_dir).map_err(|e| {
            ZyronError::Internal(format!(
                "the verification directory {} could not be created, {e}",
                self.spill_dir.display()
            ))
        })?;
        let path = self.spill_dir.join(format!(
            "set-{}-{}.run",
            std::process::id(),
            self.runs.len()
        ));
        let mut file = BufWriter::new(File::create(&path).map_err(|e| {
            ZyronError::Internal(format!("{} could not be written, {e}", path.display()))
        })?);
        for digest in &self.held {
            file.write_all(digest).map_err(|e| {
                ZyronError::Internal(format!("{} could not be written, {e}", path.display()))
            })?;
        }
        file.flush().map_err(|e| {
            ZyronError::Internal(format!("{} could not be written, {e}", path.display()))
        })?;
        self.held.clear();
        self.runs.push(path);
        Ok(())
    }

    /// The hash of the set, over its digests in ascending order.
    pub fn finish(mut self) -> Result<ChainHash> {
        self.held.sort_unstable();
        let mut hasher = Sha256::new();
        if self.runs.is_empty() {
            for digest in &self.held {
                hasher.update(digest);
            }
            return Ok(hasher.finalize().into());
        }
        // The in-memory tail is one more run, merged with the ones on disk
        let mut readers: Vec<RunReader> = Vec::with_capacity(self.runs.len() + 1);
        for path in &self.runs {
            readers.push(RunReader::open(path)?);
        }
        let tail = std::mem::take(&mut self.held);
        readers.push(RunReader::held(tail));
        let mut heap: BinaryHeap<std::cmp::Reverse<(ChainHash, usize)>> =
            BinaryHeap::with_capacity(readers.len());
        for (index, reader) in readers.iter_mut().enumerate() {
            if let Some(digest) = reader.next()? {
                heap.push(std::cmp::Reverse((digest, index)));
            }
        }
        while let Some(std::cmp::Reverse((digest, index))) = heap.pop() {
            hasher.update(digest);
            if let Some(next) = readers[index].next()? {
                heap.push(std::cmp::Reverse((next, index)));
            }
        }
        Ok(hasher.finalize().into())
    }
}

impl Drop for SetHasher {
    fn drop(&mut self) {
        for path in &self.runs {
            // A run that cannot be removed is litter rather than loss, and
            // the next set under this directory writes runs of its own
            let _ = std::fs::remove_file(path);
        }
    }
}

/// One sorted run of digests, on disk or the in-memory tail
enum RunReader {
    File(BufReader<File>),
    Held(std::vec::IntoIter<ChainHash>),
}

impl RunReader {
    fn open(path: &Path) -> Result<Self> {
        let file = File::open(path).map_err(|e| {
            ZyronError::Internal(format!("{} could not be read back, {e}", path.display()))
        })?;
        Ok(Self::File(BufReader::with_capacity(1 << 16, file)))
    }

    fn held(digests: Vec<ChainHash>) -> Self {
        Self::Held(digests.into_iter())
    }

    fn next(&mut self) -> Result<Option<ChainHash>> {
        match self {
            Self::Held(iter) => Ok(iter.next()),
            Self::File(reader) => {
                let mut digest = [0u8; 32];
                match reader.read_exact(&mut digest) {
                    Ok(()) => Ok(Some(digest)),
                    Err(e) if e.kind() == std::io::ErrorKind::UnexpectedEof => Ok(None),
                    Err(e) => Err(ZyronError::Internal(format!(
                        "a sorted run of row digests could not be read back, {e}"
                    ))),
                }
            }
        }
    }
}

/// One table's running hash inside one transaction, with the position the
/// last run of rows ended at.
#[derive(Debug)]
struct TableRows {
    hasher: RowsHasher,
    /// The highest row position this transaction has taken so far, so a run
    /// arriving below it is caught rather than hashed out of order
    last_position: Option<u64>,
    /// Set when a run arrived out of order, which the commit reports
    out_of_order: bool,
    /// Where this transaction's next rows in the table go. Seeded by the
    /// heap from the writing thread's own tail page and only ever moved
    /// forward, so every run the transaction stores lands above the one
    /// before it whatever thread stores it
    cursor: Arc<AtomicU32>,
}

impl Default for TableRows {
    fn default() -> Self {
        Self {
            hasher: RowsHasher::new(),
            last_position: None,
            out_of_order: false,
            cursor: Arc::new(AtomicU32::new(u32::MAX)),
        }
    }
}

#[derive(Debug, Default)]
struct PendingInner {
    /// Per chained table, what this transaction has written to it
    tables: HashMap<u32, TableRows>,
    /// Every table this transaction wrote to without hashing, which is every
    /// table the registry did not chain at the time of the write
    touched: Vec<u32>,
    /// Every table this transaction stamped rows deleted in
    removed: Vec<u32>,
}

/// What one transaction owes each verified table it wrote to.
///
/// Held on the execution context and filled by the write path, so the
/// expensive hashing happens where the rows already are and the commit path
/// finds one hash per table ready to link.
///
/// The rows are hashed in the order they are written, and the transaction's
/// own insertion cursor keeps that equal to the order they are stored, which
/// is the order a verification reads them back in. A run that still lands
/// below one already taken is recorded and fails the commit rather than
/// producing an entry a verification would contradict
#[derive(Debug, Default)]
pub struct PendingChainWrites {
    inner: Mutex<PendingInner>,
}

/// What one table's rows hashed to inside one transaction
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PendingTable {
    pub table_id: u32,
    pub rows_hash: ChainHash,
    pub row_count: u64,
    pub algorithm_id: u16,
}

impl PendingChainWrites {
    pub fn new() -> Self {
        Self::default()
    }

    /// A write to a table whose rows the registry does not chain.
    ///
    /// Recorded so a transaction that wrote to a table before it became
    /// verifiable is refused at commit rather than committing rows the
    /// genesis entry does not cover
    pub fn touched(&self, table_id: u32) {
        let mut inner = self.inner.lock();
        if !inner.touched.contains(&table_id) {
            inner.touched.push(table_id);
        }
    }

    /// Rows of a table were stamped deleted inside this transaction.
    ///
    /// Recorded for every table. A chain covers rows that stay, so if the
    /// table's commits turn out to be chained at commit, the transaction is
    /// refused rather than chained over rows it removed
    pub fn removed(&self, table_id: u32) {
        let mut inner = self.inner.lock();
        if !inner.removed.contains(&table_id) {
            inner.removed.push(table_id);
        }
    }

    /// Where this transaction's next rows in a chained table go.
    ///
    /// One cursor per table per transaction, handed to the heap so every run
    /// the transaction stores lands above the one before it
    pub fn cursor(&self, table_id: u32) -> Arc<AtomicU32> {
        let mut inner = self.inner.lock();
        Arc::clone(&inner.tables.entry(table_id).or_default().cursor)
    }

    /// Takes a run of rows one write put into a chained table.
    ///
    /// `first_position` and `last_position` are where the run's first and
    /// last row are stored, which is what orders the runs of one commit
    pub fn record(
        &self,
        table_id: u32,
        schema_epoch: u16,
        rows: &mut dyn Iterator<Item = &[u8]>,
        first_position: u64,
        last_position: u64,
    ) {
        let mut inner = self.inner.lock();
        let held = inner.tables.entry(table_id).or_default();
        let mut any = false;
        for row in rows {
            held.hasher.row(schema_epoch, row);
            any = true;
        }
        if !any {
            return;
        }
        if held
            .last_position
            .is_some_and(|last| first_position <= last)
        {
            held.out_of_order = true;
        }
        held.last_position = Some(last_position);
    }

    /// Whether this transaction wrote to any table at all
    pub fn is_empty(&self) -> bool {
        let inner = self.inner.lock();
        inner.tables.is_empty() && inner.touched.is_empty() && inner.removed.is_empty()
    }

    /// Closes every table's hash and hands back what the commit links, one
    /// entry per table ordered by table id so two members of a group link
    /// the same tables in the same order.
    ///
    /// `txn_id` is held against the fence each chained table was marked
    /// with. A transaction that started before a table became verifiable
    /// and wrote to it, hashed or not, is refused: its rows are neither in
    /// the genesis set nor covered by an entry of their own
    pub fn take(&self, txn_id: u64, registry: &ChainRegistry) -> Result<Vec<PendingTable>> {
        let mut inner = self.inner.lock();
        let held = std::mem::take(&mut *inner);
        drop(inner);
        for table_id in &held.touched {
            if registry.chained(*table_id).is_some() {
                return Err(straddled(*table_id, txn_id));
            }
        }
        for table_id in &held.removed {
            if registry.chained(*table_id).is_some() {
                return Err(ZyronError::Internal(format!(
                    "transaction {txn_id} removed rows from table {table_id}, whose commits are \
                     chained over the rows they wrote, so the transaction cannot be chained and \
                     is rolled back"
                )));
            }
        }
        let mut out = Vec::with_capacity(held.tables.len());
        for (table_id, rows) in held.tables {
            let Some(chained) = registry.chained(table_id) else {
                return Err(ZyronError::Internal(format!(
                    "transaction {txn_id} hashed rows for table {table_id}, whose commits this \
                     node does not chain"
                )));
            };
            if txn_id < chained.fence {
                return Err(straddled(table_id, txn_id));
            }
            if rows.out_of_order {
                return Err(ZyronError::Internal(format!(
                    "a write to verified table {table_id} stored rows below ones it had already \
                     written, so the chain entry would cover them in an order a verification \
                     cannot read them back in"
                )));
            }
            if rows.hasher.rows() == 0 {
                continue;
            }
            out.push(PendingTable {
                table_id,
                row_count: rows.hasher.rows(),
                rows_hash: rows.hasher.finish(),
                algorithm_id: chained.algorithm_id,
            });
        }
        out.sort_by_key(|pending| pending.table_id);
        Ok(out)
    }

    /// Drops what a rolled back transaction had accumulated.
    pub fn clear(&self) {
        let mut inner = self.inner.lock();
        inner.tables.clear();
        inner.touched.clear();
        inner.removed.clear();
    }
}

fn straddled(table_id: u32, txn_id: u64) -> ZyronError {
    ZyronError::Internal(format!(
        "table {table_id} became verifiable after transaction {txn_id} started and the \
         transaction had written to it, so its rows are covered by neither the genesis entry \
         nor an entry of their own. The transaction is rolled back, and run again it is chained"
    ))
}

// ---------------------------------------------------------------------------
// The chain of one table
// ---------------------------------------------------------------------------

/// What the chain of one table stands at.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ChainHead {
    /// Entries in the chain, which is also the sequence the next entry takes
    pub commits: u64,
    /// The last entry's link, or [`NO_PREVIOUS`] for an empty chain
    pub head_hash: ChainHash,
    /// The version the last entry covered
    pub head_version: u64,
    /// Microseconds the last entry was written at
    pub head_ts: i64,
}

impl Default for ChainHead {
    fn default() -> Self {
        Self {
            commits: 0,
            head_hash: NO_PREVIOUS,
            head_version: 0,
            head_ts: 0,
        }
    }
}

impl ChainHead {
    /// Bytes the chain's entries take, which is what storage attribution
    /// charges the table's owner.
    pub fn bytes(&self) -> u64 {
        self.commits * CHAIN_RECORD_LEN as u64
    }
}

/// One verified table's chain, its file and the head it stands at.
#[derive(Debug)]
pub struct CommitChain {
    table_id: u32,
    path: PathBuf,
    state: Mutex<ChainState>,
    /// The handle reads go through, opened on the first read and kept, so
    /// a walk, an anchor check and a range resolution open nothing. Held
    /// apart from the state so a read of the file blocks no commit
    read_handle: Mutex<Option<File>>,
}

#[derive(Debug)]
struct ChainState {
    /// What the next entry links onto, the entries not yet committed
    /// included. Internal: what a reader sees is `durable`
    linking: ChainHead,
    /// What the chain holds, which is what a verification walks, an anchor
    /// names and a view reports
    durable: ChainHead,
    /// The file, opened for append on the first committed entry, with the
    /// records not yet written to it
    file: Option<BufWriter<File>>,
    /// Entries linked by a transaction that has not committed yet, in the
    /// order they were linked.
    ///
    /// The file only ever takes entries whose transaction committed, so a
    /// process that stops between the link and the commit leaves the file
    /// exactly where it was. The log carries the entry either way, and
    /// recovery puts back the ones whose transaction committed
    pending: VecDeque<PendingEntry>,
}

#[derive(Debug)]
struct PendingEntry {
    txn_id: u64,
    entry: CommitHash,
    /// True once the transaction that linked it committed
    committed: bool,
}

impl CommitChain {
    /// Opens the chain of one table, reading its head back from the file.
    ///
    /// A file whose length is not a whole number of records was cut by a
    /// stop between the write and the flush. The partial tail is cut off,
    /// because a record the chain never finished writing covers no commit
    /// and the next record has to land on a record boundary
    pub fn open(dir: &Path, table_id: u32) -> Result<Self> {
        let path = chain_path(dir, table_id);
        let head = if path.exists() {
            read_head(&path, table_id)?
        } else {
            ChainHead::default()
        };
        Ok(Self {
            table_id,
            path,
            state: Mutex::new(ChainState {
                linking: head,
                durable: head,
                file: None,
                pending: VecDeque::new(),
            }),
            read_handle: Mutex::new(None),
        })
    }

    pub fn table_id(&self) -> u32 {
        self.table_id
    }

    pub fn path(&self) -> &Path {
        &self.path
    }

    /// What the chain holds now.
    ///
    /// The entries of transactions that have committed. One linked by a
    /// transaction still in flight is not here, because a chain states what
    /// happened and that transaction has not happened yet
    pub fn head(&self) -> ChainHead {
        self.state.lock().durable
    }

    /// Links one commit onto the chain.
    ///
    /// The whole of the serial section: read the head, hash the entry's own
    /// fields, hold the record until the commit. `commit_version` is read
    /// under the lock, so versions rise along the chain and a version range
    /// resolves to one run of positions. The rows were hashed before this
    /// was called and the transaction's durability is waited for after it
    /// returns
    pub fn append(
        &self,
        fields: CommitFields,
        commit_version: impl FnOnce() -> u64,
    ) -> Result<CommitHash> {
        if fields.row_count > MAX_COMMIT_ROWS {
            return Err(ZyronError::Internal(format!(
                "a commit wrote {} rows to table {}, more than the {MAX_COMMIT_ROWS} a chain \
                 entry can state, so the entry would not cover what the commit wrote",
                fields.row_count, self.table_id
            )));
        }
        let mut state = self.state.lock();
        let version = commit_version().max(state.linking.head_version);
        let entry = CommitHash::link(
            self.table_id,
            state.linking.commits,
            state.linking.head_hash,
            version,
            fields,
        );
        state.pending.push_back(PendingEntry {
            txn_id: fields.txn_id,
            entry,
            committed: false,
        });
        state.linking = head_after(&entry);
        Ok(entry)
    }

    /// Puts the entries one committed transaction linked into the file.
    ///
    /// Called once the transaction's commit record is written, which is the
    /// point from which the rows the entry covers are there. An entry whose
    /// predecessor is still uncommitted waits for it, so the file is always
    /// a whole chain from its first entry to its last. Every entry that is
    /// ready goes in one write
    pub fn publish(&self, txn_id: u64) -> Result<usize> {
        let mut state = self.state.lock();
        for held in state.pending.iter_mut() {
            if held.txn_id == txn_id {
                held.committed = true;
            }
        }
        let mut written = 0usize;
        while state.pending.front().is_some_and(|held| held.committed) {
            let Some(held) = state.pending.pop_front() else {
                break;
            };
            Self::write_record(&self.path, &mut state, &held.entry)?;
            written += 1;
        }
        Ok(written)
    }

    /// Drops the entries a transaction linked and did not commit.
    ///
    /// The head goes back to where it stood when they were linked, which is
    /// only sound while they are the newest entries. Entries linked after
    /// them chained onto them, so discarding one underneath would leave a
    /// chain nothing can walk, and that is reported rather than written
    pub fn discard(&self, txn_id: u64) -> Result<usize> {
        let mut state = self.state.lock();
        let first = state.pending.iter().position(|held| held.txn_id == txn_id);
        let Some(first) = first else {
            return Ok(0);
        };
        if state
            .pending
            .iter()
            .skip(first)
            .any(|held| held.txn_id != txn_id)
        {
            return Err(ZyronError::Internal(format!(
                "the commit chain of table {} holds entries linked after the one transaction \
                 {txn_id} did not commit, so discarding it would leave a chain that cannot be \
                 walked",
                self.table_id
            )));
        }
        let dropped = state.pending.len() - first;
        state.pending.truncate(first);
        state.linking = match state.pending.back() {
            Some(held) => head_after(&held.entry),
            None => state.durable,
        };
        Ok(dropped)
    }

    /// Entries linked and not yet committed, for the views and the tests
    pub fn pending_commits(&self) -> usize {
        self.state.lock().pending.len()
    }

    /// Puts back an entry this node's log holds from before a restart.
    ///
    /// The entry keeps the link it arrived with rather than being relinked,
    /// and the link is recomputed against the head this chain stands at:
    /// an entry that does not follow the head is refused, because writing
    /// it would leave a chain whose walk fails, and a refusal here names
    /// the gap while the entry is still in hand
    pub fn adopt(&self, entry: &CommitHash) -> Result<bool> {
        let mut state = self.state.lock();
        if entry.sequence < state.durable.commits {
            // Already here, which is what a replayed log record looks like
            return Ok(false);
        }
        if entry.sequence != state.durable.commits {
            return Err(ZyronError::Internal(format!(
                "a commit chain entry for table {} is at position {} and this node's chain \
                 stands at {}, so adopting it would leave a gap the chain cannot be walked over",
                entry.table_id, entry.sequence, state.durable.commits
            )));
        }
        if entry.prev_hash != state.durable.head_hash {
            return Err(ZyronError::Internal(format!(
                "a commit chain entry for table {} links to {} and this node's chain head is {}",
                entry.table_id,
                hex(&entry.prev_hash),
                hex(&state.durable.head_hash)
            )));
        }
        if entry.compute_entry_hash() != entry.entry_hash {
            return Err(ZyronError::Internal(format!(
                "a commit chain entry for table {} at position {} does not hash to the link it \
                 carries, so it is not an entry of this chain",
                entry.table_id, entry.sequence
            )));
        }
        if !state.pending.is_empty() {
            return Err(ZyronError::Internal(format!(
                "a commit chain entry for table {} arrived while {} linked entr(ies) wait for \
                 their commit, so its place in the chain is not settled",
                entry.table_id,
                state.pending.len()
            )));
        }
        Self::write_record(&self.path, &mut state, entry)?;
        state.linking = state.durable;
        Ok(true)
    }

    fn write_record(path: &Path, state: &mut ChainState, entry: &CommitHash) -> Result<()> {
        let mut record = [0u8; CHAIN_RECORD_LEN];
        entry.encode_into(&mut record);
        if state.file.is_none() {
            state.file = Some(BufWriter::with_capacity(
                WRITE_BUFFER_RECORDS * CHAIN_RECORD_LEN,
                open_for_append(path)?,
            ));
        }
        let Some(file) = state.file.as_mut() else {
            return Err(ZyronError::Internal(format!(
                "the commit chain at {} has no open file",
                path.display()
            )));
        };
        file.write_all(&record).map_err(|e| {
            ZyronError::Internal(format!(
                "a commit chain entry could not be written to {}, {e}",
                path.display()
            ))
        })?;
        state.durable = head_after(entry);
        if state.linking.commits < state.durable.commits {
            state.linking = state.durable;
        }
        Ok(())
    }

    /// Writes the records held in memory to the file, without waiting for
    /// the device. Run before anything reads the file
    fn flush(&self) -> Result<()> {
        let mut state = self.state.lock();
        let Some(file) = state.file.as_mut() else {
            return Ok(());
        };
        file.flush().map_err(|e| {
            ZyronError::Internal(format!(
                "the commit chain at {} could not be written, {e}",
                self.path.display()
            ))
        })
    }

    /// Puts the entries on the device.
    ///
    /// Called once the transaction that wrote them is durable, so an entry
    /// is on disk no later than the rows it covers
    pub fn sync(&self) -> Result<()> {
        let mut state = self.state.lock();
        let Some(file) = state.file.as_mut() else {
            return Ok(());
        };
        file.flush().map_err(|e| {
            ZyronError::Internal(format!(
                "the commit chain at {} could not be written, {e}",
                self.path.display()
            ))
        })?;
        file.get_ref().sync_data().map_err(|e| {
            ZyronError::Internal(format!(
                "the commit chain at {} could not be flushed, {e}",
                self.path.display()
            ))
        })
    }

    /// The read handle, with everything published written to the file
    /// ahead of it. None while the chain has no file
    fn reader(&self) -> Result<Option<ChainReader<'_>>> {
        self.flush()?;
        let mut handle = self.read_handle.lock();
        if handle.is_none() {
            if !self.path.exists() {
                return Ok(None);
            }
            *handle = Some(open_for_read(&self.path)?);
        }
        Ok(Some(ChainReader {
            table_id: self.table_id,
            handle,
        }))
    }

    /// Reads the entries in a sequence range, oldest first.
    ///
    /// `from` and `to` are chain positions and `to` is inclusive. A range
    /// past the head ends at the head. A range longer than
    /// [`READ_WINDOW_RECORDS`] is read in windows of that many records, so
    /// the file is read in a few large reads rather than one per record
    pub fn read_range(&self, from: u64, to: u64) -> Result<Vec<CommitHash>> {
        let head = self.head();
        if head.commits == 0 || from >= head.commits {
            return Ok(Vec::new());
        }
        let last = to.min(head.commits - 1);
        if last < from {
            return Ok(Vec::new());
        }
        let Some(mut reader) = self.reader()? else {
            return Ok(Vec::new());
        };
        let mut prev = reader.link_before(from)?;
        let mut out = Vec::with_capacity((last - from + 1) as usize);
        let mut at = from;
        while at <= last {
            let window = (last - at + 1).min(READ_WINDOW_RECORDS);
            let records = reader.read_records(at, window)?;
            if records.is_empty() {
                break;
            }
            for record in records.chunks_exact(CHAIN_RECORD_LEN) {
                let Ok(record) = <&[u8; CHAIN_RECORD_LEN]>::try_from(record) else {
                    break;
                };
                let entry = CommitHash::decode(self.table_id, at, prev, record);
                prev = entry.entry_hash;
                out.push(entry);
                at += 1;
            }
            if (records.len() / CHAIN_RECORD_LEN) as u64 != window {
                break;
            }
        }
        Ok(out)
    }

    /// The link the entry at one position carries, None past the end
    pub fn entry_hash_at(&self, sequence: u64) -> Result<Option<ChainHash>> {
        if sequence >= self.head().commits {
            return Ok(None);
        }
        let Some(mut reader) = self.reader()? else {
            return Ok(None);
        };
        Ok(reader
            .record_at(sequence)?
            .map(|record| entry_hash_of(&record)))
    }

    /// The first position whose version is at or above `version`, or the
    /// commit count when no entry reaches it.
    ///
    /// Versions rise along the chain, so this is a binary search over the
    /// file that reads one record per probe
    pub fn first_at_or_above(&self, version: u64) -> Result<u64> {
        let commits = self.head().commits;
        if commits == 0 {
            return Ok(0);
        }
        let Some(mut reader) = self.reader()? else {
            return Ok(0);
        };
        let (mut low, mut high) = (0u64, commits);
        while low < high {
            let mid = low + (high - low) / 2;
            match reader.record_at(mid)? {
                Some(record) if version_of(&record) < version => low = mid + 1,
                Some(_) => high = mid,
                None => high = mid,
            }
        }
        Ok(low)
    }

    /// The last position whose version is at or below `version`, None when
    /// the first entry is already above it
    pub fn last_at_or_below(&self, version: u64) -> Result<Option<u64>> {
        let commits = self.head().commits;
        if commits == 0 {
            return Ok(None);
        }
        let Some(mut reader) = self.reader()? else {
            return Ok(None);
        };
        let (mut low, mut high) = (0u64, commits);
        while low < high {
            let mid = low + (high - low) / 2;
            match reader.record_at(mid)? {
                Some(record) if version_of(&record) <= version => low = mid + 1,
                Some(_) => high = mid,
                None => high = mid,
            }
        }
        Ok(low.checked_sub(1))
    }

    /// Removes the chain with the table it covered.
    pub fn remove(&self) -> Result<()> {
        let mut state = self.state.lock();
        state.file = None;
        state.linking = ChainHead::default();
        state.durable = ChainHead::default();
        state.pending.clear();
        *self.read_handle.lock() = None;
        if self.path.exists() {
            std::fs::remove_file(&self.path).map_err(|e| {
                ZyronError::Internal(format!(
                    "the commit chain at {} could not be removed, {e}",
                    self.path.display()
                ))
            })?;
        }
        Ok(())
    }
}

/// One read of a chain file through the chain's kept handle
struct ChainReader<'a> {
    table_id: u32,
    handle: parking_lot::MutexGuard<'a, Option<File>>,
}

impl ChainReader<'_> {
    fn file(&mut self) -> Result<&mut File> {
        self.handle.as_mut().ok_or_else(|| {
            ZyronError::Internal(format!(
                "the commit chain of table {} lost its read handle mid-read",
                self.table_id
            ))
        })
    }

    /// Reads up to `count` records from `sequence` in one read
    fn read_records(&mut self, sequence: u64, count: u64) -> Result<Vec<u8>> {
        let file = self.file()?;
        let at = CHAIN_HEADER_LEN as u64 + sequence * CHAIN_RECORD_LEN as u64;
        file.seek(SeekFrom::Start(at)).map_err(|e| {
            ZyronError::Internal(format!("a commit chain entry could not be reached, {e}"))
        })?;
        let mut buf = vec![0u8; (count as usize) * CHAIN_RECORD_LEN];
        let mut filled = 0usize;
        while filled < buf.len() {
            match file.read(&mut buf[filled..]) {
                Ok(0) => break,
                Ok(n) => filled += n,
                Err(e) if e.kind() == std::io::ErrorKind::Interrupted => continue,
                Err(e) => {
                    return Err(ZyronError::Internal(format!(
                        "a commit chain entry could not be read, {e}"
                    )));
                }
            }
        }
        buf.truncate(filled - filled % CHAIN_RECORD_LEN);
        Ok(buf)
    }

    /// Reads one record by its chain position, None past the end
    fn record_at(&mut self, sequence: u64) -> Result<Option<[u8; CHAIN_RECORD_LEN]>> {
        let file = self.file()?;
        read_record_from(file, sequence)
    }

    /// The link the entry at `sequence` points back to
    fn link_before(&mut self, sequence: u64) -> Result<ChainHash> {
        if sequence == 0 {
            return Ok(NO_PREVIOUS);
        }
        self.record_at(sequence - 1)?
            .map(|record| entry_hash_of(&record))
            .ok_or_else(|| {
                ZyronError::Internal(format!(
                    "the commit chain of table {} is shorter than the entry before position \
                     {sequence}",
                    self.table_id
                ))
            })
    }
}

/// Where the chain stands once `entry` is on it
fn head_after(entry: &CommitHash) -> ChainHead {
    ChainHead {
        commits: entry.sequence + 1,
        head_hash: entry.entry_hash,
        head_version: entry.commit_version,
        head_ts: entry.commit_ts,
    }
}

/// The file one table's chain is written to.
pub fn chain_path(dir: &Path, table_id: u32) -> PathBuf {
    dir.join("verify").join(format!("{table_id}.zvch"))
}

/// The directory a table's verification writes its temporary runs under
pub fn spill_dir(dir: &Path) -> PathBuf {
    dir.join("verify")
}

fn entry_hash_of(record: &[u8; CHAIN_RECORD_LEN]) -> ChainHash {
    let mut hash = [0u8; 32];
    hash.copy_from_slice(&record[64..96]);
    hash
}

fn version_of(record: &[u8; CHAIN_RECORD_LEN]) -> u64 {
    let mut word = [0u8; 8];
    word.copy_from_slice(&record[0..8]);
    u64::from_le_bytes(word)
}

fn open_for_append(path: &Path) -> Result<File> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).map_err(|e| {
            ZyronError::Internal(format!(
                "the verification directory {} could not be created, {e}",
                parent.display()
            ))
        })?;
    }
    let fresh = !path.exists();
    // A plain write handle positioned at the end rather than an append-mode
    // one. Every write goes through this one handle in order, so the
    // position stays at the end, and reads go through a handle of their own
    let mut file = OpenOptions::new()
        .create(true)
        .write(true)
        .open(path)
        .map_err(|e| {
            ZyronError::Internal(format!(
                "the commit chain at {} could not be opened, {e}",
                path.display()
            ))
        })?;
    if fresh {
        let header = envelope::encode_header(
            FormatKind::VerifiableCommitChain,
            CHAIN_FORMAT_VERSION,
            0,
            &[],
        );
        file.write_all(&header).map_err(|e| {
            ZyronError::Internal(format!(
                "the commit chain at {} could not be started, {e}",
                path.display()
            ))
        })?;
    } else {
        file.seek(SeekFrom::End(0)).map_err(|e| {
            ZyronError::Internal(format!(
                "the end of the commit chain at {} could not be reached, {e}",
                path.display()
            ))
        })?;
    }
    Ok(file)
}

fn open_for_read(path: &Path) -> Result<File> {
    File::open(path).map_err(|e| {
        ZyronError::Internal(format!(
            "the commit chain at {} could not be read, {e}",
            path.display()
        ))
    })
}

fn read_record_from(file: &mut File, sequence: u64) -> Result<Option<[u8; CHAIN_RECORD_LEN]>> {
    let at = CHAIN_HEADER_LEN as u64 + sequence * CHAIN_RECORD_LEN as u64;
    file.seek(SeekFrom::Start(at)).map_err(|e| {
        ZyronError::Internal(format!("a commit chain entry could not be reached, {e}"))
    })?;
    let mut record = [0u8; CHAIN_RECORD_LEN];
    match file.read_exact(&mut record) {
        Ok(()) => Ok(Some(record)),
        Err(e) if e.kind() == std::io::ErrorKind::UnexpectedEof => Ok(None),
        Err(e) => Err(ZyronError::Internal(format!(
            "a commit chain entry could not be read, {e}"
        ))),
    }
}

/// Reads the head back from a chain file, cutting off a partial record a
/// stop left at its end.
fn read_head(path: &Path, table_id: u32) -> Result<ChainHead> {
    let length = std::fs::metadata(path)
        .map_err(|e| {
            ZyronError::Internal(format!(
                "the commit chain at {} could not be measured, {e}",
                path.display()
            ))
        })?
        .len();
    if length < CHAIN_HEADER_LEN as u64 {
        return Ok(ChainHead::default());
    }
    let mut file = open_for_read(path)?;
    let mut header = vec![0u8; CHAIN_HEADER_LEN];
    file.read_exact(&mut header).map_err(|e| {
        ZyronError::Internal(format!(
            "the commit chain at {} has no readable header, {e}",
            path.display()
        ))
    })?;
    let (kind, _) = envelope::peek(&header)?;
    if kind != FormatKind::VerifiableCommitChain {
        return Err(ZyronError::Internal(format!(
            "{} holds a {} rather than a commit chain",
            path.display(),
            kind.catalog_name()
        )));
    }
    let body = length - CHAIN_HEADER_LEN as u64;
    let commits = body / CHAIN_RECORD_LEN as u64;
    let whole = CHAIN_HEADER_LEN as u64 + commits * CHAIN_RECORD_LEN as u64;
    if whole != length {
        // The tail is a record the chain never finished writing. It covers
        // no commit, and the next record has to start on a record boundary
        drop(file);
        OpenOptions::new()
            .write(true)
            .open(path)
            .and_then(|file| file.set_len(whole))
            .map_err(|e| {
                ZyronError::Internal(format!(
                    "the partial record at the end of the commit chain at {} could not be cut \
                     off, {e}",
                    path.display()
                ))
            })?;
        file = open_for_read(path)?;
    }
    if commits == 0 {
        return Ok(ChainHead::default());
    }
    let Some(record) = read_record_from(&mut file, commits - 1)? else {
        return Ok(ChainHead::default());
    };
    // The head's own link is what the next entry points back to. Its
    // prev_hash is not needed to publish the head, so it is read as the
    // genesis value and the walk supplies the real one
    let entry = CommitHash::decode(table_id, commits - 1, NO_PREVIOUS, &record);
    Ok(ChainHead {
        commits,
        head_hash: entry.entry_hash,
        head_version: entry.commit_version,
        head_ts: entry.commit_ts,
    })
}

// ---------------------------------------------------------------------------
// The chains this node holds
// ---------------------------------------------------------------------------

/// What the registry holds about one table whose commits it chains
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ChainedTable {
    /// The transaction id the table was chained from. Every transaction at
    /// or above it that writes to the table is chained, and one below it
    /// that wrote to the table is refused at commit. Zero for a table that
    /// was chained before this process started, which no transaction of
    /// this process precedes
    pub fence: u64,
    /// The registered scheme the chain links with
    pub algorithm_id: u16,
    /// One past the genesis entry's position, zero when the chain has none
    pub genesis_at: u64,
}

impl ChainedTable {
    /// The genesis entry's position, if the chain has one
    pub fn genesis_sequence(&self) -> Option<u64> {
        self.genesis_at.checked_sub(1)
    }
}

/// Every verified table's chain on this node.
#[derive(Debug)]
pub struct ChainRegistry {
    dir: PathBuf,
    chains: Mutex<HashMap<u32, Arc<CommitChain>>>,
    /// The tables whose commits are chained, which is what the write path
    /// asks before it hashes a row
    chained: RwLock<HashMap<u32, ChainedTable>>,
    anchors: anchor::AnchorStore,
    runs: RunLog,
}

impl ChainRegistry {
    /// Opens the registry over a data directory, putting back the anchors
    /// and the verification history it holds. The tables it chains are
    /// registered by whoever holds the catalog, before anything writes
    pub fn open(dir: impl Into<PathBuf>) -> Result<Self> {
        let dir = dir.into();
        let anchors = anchor::AnchorStore::open(&dir)?;
        let runs = RunLog::open(&dir)?;
        Ok(Self {
            dir,
            chains: Mutex::new(HashMap::new()),
            chained: RwLock::new(HashMap::new()),
            anchors,
            runs,
        })
    }

    pub fn data_dir(&self) -> &Path {
        &self.dir
    }

    pub fn anchors(&self) -> &anchor::AnchorStore {
        &self.anchors
    }

    pub fn runs(&self) -> &RunLog {
        &self.runs
    }

    /// Marks a table's commits as chained from `fence` on.
    ///
    /// This is the moment a table becomes verifiable for the write path:
    /// every transaction at or above the fence hashes what it writes
    pub fn chain_from(&self, table_id: u32, fence: u64, algorithm_id: u16, genesis_at: u64) {
        self.chained.write().insert(
            table_id,
            ChainedTable {
                fence,
                algorithm_id,
                genesis_at,
            },
        );
    }

    /// Records where a table's genesis entry landed
    pub fn set_genesis_at(&self, table_id: u32, genesis_at: u64) {
        if let Some(held) = self.chained.write().get_mut(&table_id) {
            held.genesis_at = genesis_at;
        }
    }

    /// Whether a table's commits are chained, and from which transaction
    pub fn chained(&self, table_id: u32) -> Option<ChainedTable> {
        self.chained.read().get(&table_id).copied()
    }

    /// Stops chaining a table's commits, which is only sound while its
    /// chain holds none
    pub fn unchain(&self, table_id: u32) {
        self.chained.write().remove(&table_id);
    }

    /// Every table whose commits are chained, ordered by id
    pub fn chained_tables(&self) -> Vec<(u32, ChainedTable)> {
        let mut out: Vec<(u32, ChainedTable)> = self
            .chained
            .read()
            .iter()
            .map(|(id, held)| (*id, *held))
            .collect();
        out.sort_unstable_by_key(|(id, _)| *id);
        out
    }

    /// The chain of one table, opened on first use.
    pub fn chain(&self, table_id: u32) -> Result<Arc<CommitChain>> {
        if let Some(chain) = self.chains.lock().get(&table_id) {
            return Ok(Arc::clone(chain));
        }
        let opened = Arc::new(CommitChain::open(&self.dir, table_id)?);
        let mut chains = self.chains.lock();
        let held = chains
            .entry(table_id)
            .or_insert_with(|| Arc::clone(&opened));
        Ok(Arc::clone(held))
    }

    /// The chain of one table when it is already open, without opening one
    /// for a table that has none.
    pub fn opened(&self, table_id: u32) -> Option<Arc<CommitChain>> {
        self.chains.lock().get(&table_id).map(Arc::clone)
    }

    /// Puts every open chain's entries on the device.
    ///
    /// Run by the checkpoint before it records the boundary, so every entry
    /// the log stops carrying is already in the file it belongs to
    pub fn sync_all(&self) -> Result<()> {
        let held: Vec<Arc<CommitChain>> = self.chains.lock().values().map(Arc::clone).collect();
        for chain in held {
            chain.sync()?;
        }
        Ok(())
    }

    /// Every table with an open chain, for the views.
    pub fn table_ids(&self) -> Vec<u32> {
        let mut ids: Vec<u32> = self.chains.lock().keys().copied().collect();
        ids.sort_unstable();
        ids
    }

    /// Drops a table's chain with the table.
    pub fn remove(&self, table_id: u32) -> Result<()> {
        self.chained.write().remove(&table_id);
        let held = self.chains.lock().remove(&table_id);
        if let Some(chain) = held {
            chain.remove()?;
        } else {
            let path = chain_path(&self.dir, table_id);
            if path.exists() {
                std::fs::remove_file(&path).map_err(|e| {
                    ZyronError::Internal(format!(
                        "the commit chain at {} could not be removed, {e}",
                        path.display()
                    ))
                })?;
            }
        }
        self.anchors.forget(table_id)?;
        Ok(())
    }
}

/// One chain entry a log record carries, as the chain adopts it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LoggedEntry {
    pub table_id: u32,
    pub sequence: u64,
    pub record: [u8; CHAIN_RECORD_LEN],
}

impl LoggedEntry {
    /// Reads the payload of one logged entry.
    pub fn decode(table_id: u32, sequence: u64, record: &[u8]) -> Result<Self> {
        let Ok(record) = <[u8; CHAIN_RECORD_LEN]>::try_from(record) else {
            return Err(ZyronError::Internal(format!(
                "a logged commit chain entry for table {table_id} holds {} bytes rather than \
                 the {CHAIN_RECORD_LEN} a chain record takes",
                record.len()
            )));
        };
        Ok(Self {
            table_id,
            sequence,
            record,
        })
    }

    /// The entry the record describes, given what it follows.
    pub fn entry(&self, prev_hash: ChainHash) -> CommitHash {
        CommitHash::decode(self.table_id, self.sequence, prev_hash, &self.record)
    }
}

/// Puts logged chain entries back into their chain files.
///
/// Called at start before anything reads a chain, with the entries of the
/// transactions that committed, in log order. An entry the chain already
/// holds is passed over, so replaying a log twice changes nothing, and an
/// entry that would leave a gap fails the start rather than producing a
/// chain that cannot be walked
pub fn restore_logged_entries(registry: &ChainRegistry, entries: &[LoggedEntry]) -> Result<usize> {
    let mut restored = 0usize;
    for logged in entries {
        let chain = registry.chain(logged.table_id)?;
        let head = chain.head();
        if logged.sequence < head.commits {
            continue;
        }
        let entry = logged.entry(head.head_hash);
        if chain.adopt(&entry)? {
            restored += 1;
        }
    }
    Ok(restored)
}

// ---------------------------------------------------------------------------
// Verification
// ---------------------------------------------------------------------------

/// How much of a table a verification recomputed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RowMode {
    /// Every commit's rows were read back and rehashed, the genesis set
    /// included
    All,
    /// The chain was walked whole and the rows of a sample of its commits
    /// were read back and rehashed. The genesis set is not read, because
    /// reading it back is reading the whole of what the table held
    Sampled,
}

impl RowMode {
    pub fn label(self) -> &'static str {
        match self {
            RowMode::All => "all",
            RowMode::Sampled => "sampled",
        }
    }

    /// Reads the word `WITH (rows => ...)` carries.
    pub fn parse(word: &str) -> Option<RowMode> {
        match word.trim().to_ascii_lowercase().as_str() {
            "all" => Some(RowMode::All),
            "sampled" => Some(RowMode::Sampled),
            _ => None,
        }
    }
}

/// Why a verification found the table not intact.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum VerifyFailure {
    /// A commit's rows no longer hash to what its entry recorded
    RowContentMismatch {
        commit_version: u64,
        sequence: u64,
        expected: ChainHash,
        found: ChainHash,
    },
    /// A commit's rows are fewer or more than its entry recorded
    RowCountMismatch {
        commit_version: u64,
        sequence: u64,
        expected: u64,
        found: u64,
    },
    /// An entry does not link to the one before it
    BrokenLink {
        commit_version: u64,
        sequence: u64,
        expected: ChainHash,
        found: ChainHash,
    },
    /// An entry's own link is not what its contents hash to
    EntryHashMismatch { commit_version: u64, sequence: u64 },
    /// The chain is shorter than an anchor says it was, or stands at a
    /// different head where the anchor named one
    AnchorContradiction {
        sequence: u64,
        anchored_version: u64,
        anchored_head: ChainHash,
        found_head: Option<ChainHash>,
        anchored_at: i64,
    },
    /// A commit the chain names has no entry
    MissingCommit { sequence: u64 },
}

impl VerifyFailure {
    /// The chain position the failure was found at.
    pub fn sequence(&self) -> u64 {
        match self {
            VerifyFailure::RowContentMismatch { sequence, .. }
            | VerifyFailure::RowCountMismatch { sequence, .. }
            | VerifyFailure::BrokenLink { sequence, .. }
            | VerifyFailure::EntryHashMismatch { sequence, .. }
            | VerifyFailure::AnchorContradiction { sequence, .. }
            | VerifyFailure::MissingCommit { sequence } => *sequence,
        }
    }

    /// The commit the failure was found at, zero where the failure is the
    /// absence of one.
    pub fn commit_version(&self) -> u64 {
        match self {
            VerifyFailure::RowContentMismatch { commit_version, .. }
            | VerifyFailure::RowCountMismatch { commit_version, .. }
            | VerifyFailure::BrokenLink { commit_version, .. }
            | VerifyFailure::EntryHashMismatch { commit_version, .. } => *commit_version,
            VerifyFailure::AnchorContradiction {
                anchored_version, ..
            } => *anchored_version,
            VerifyFailure::MissingCommit { .. } => 0,
        }
    }

    /// The word a result and an alert name the failure by.
    pub fn kind(&self) -> &'static str {
        match self {
            VerifyFailure::RowContentMismatch { .. } => "row_content_mismatch",
            VerifyFailure::RowCountMismatch { .. } => "row_count_mismatch",
            VerifyFailure::BrokenLink { .. } => "broken_link",
            VerifyFailure::EntryHashMismatch { .. } => "entry_hash_mismatch",
            VerifyFailure::AnchorContradiction { .. } => "anchor_contradiction",
            VerifyFailure::MissingCommit { .. } => "missing_commit",
        }
    }
}

impl std::fmt::Display for VerifyFailure {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            VerifyFailure::RowContentMismatch {
                commit_version,
                expected,
                found,
                ..
            } => write!(
                f,
                "the rows of commit {commit_version} hash to {} and its chain entry recorded {}",
                hex(found),
                hex(expected)
            ),
            VerifyFailure::RowCountMismatch {
                commit_version,
                expected,
                found,
                ..
            } => write!(
                f,
                "commit {commit_version} covers {found} row(s) and its chain entry recorded \
                 {expected}"
            ),
            VerifyFailure::BrokenLink {
                commit_version,
                expected,
                found,
                ..
            } => write!(
                f,
                "the entry for commit {commit_version} links to {} and the entry before it \
                 hashes to {}",
                hex(found),
                hex(expected)
            ),
            VerifyFailure::EntryHashMismatch { commit_version, .. } => write!(
                f,
                "the entry for commit {commit_version} does not hash to the link it carries"
            ),
            VerifyFailure::AnchorContradiction {
                anchored_version,
                anchored_head,
                found_head,
                anchored_at,
                ..
            } => match found_head {
                Some(found) => write!(
                    f,
                    "the anchor taken at epoch-micros {anchored_at} names head {} at version \
                     {anchored_version} and the chain stands at {} there",
                    hex(anchored_head),
                    hex(found)
                ),
                None => write!(
                    f,
                    "the anchor taken at epoch-micros {anchored_at} names head {} at version \
                     {anchored_version} and the chain no longer reaches that commit",
                    hex(anchored_head)
                ),
            },
            VerifyFailure::MissingCommit { sequence } => {
                write!(f, "the chain has no entry at position {sequence}")
            }
        }
    }
}

/// What one verification found.
#[derive(Debug, Clone, PartialEq)]
pub struct VerifyOutcome {
    pub table_id: u32,
    pub commits_checked: u64,
    pub rows_checked: u64,
    pub anchors_checked: u64,
    pub mode: RowMode,
    /// Commits whose rows were read back and rehashed, which under
    /// [`RowMode::All`] is every commit checked
    pub commits_rehashed: u64,
    /// Whether the walk passed a genesis entry whose set it did not read
    /// back, which a sampled pass does
    pub genesis_not_read: bool,
    pub intact: bool,
    pub failure: Option<VerifyFailure>,
    /// The first and last chain position the walk covered
    pub from_sequence: u64,
    pub to_sequence: u64,
    /// Microseconds spent reading the chain's entries and recomputing their
    /// links
    pub chain_micros: u64,
    /// Microseconds spent in the passes over the table that read the rows
    /// back and rehashed them
    pub rows_micros: u64,
}

impl VerifyOutcome {
    /// The sentence a result set and an audit record state the run by.
    ///
    /// A pass always says which mode ran and over how many rows, so a
    /// sampled pass is never read as a full one
    pub fn summary(&self) -> String {
        let scope = match self.mode {
            RowMode::All => format!(
                "{} commit(s) checked and every one of their {} row(s) rehashed",
                self.commits_checked, self.rows_checked
            ),
            RowMode::Sampled => {
                let genesis = if self.genesis_not_read {
                    " and the genesis set"
                } else {
                    ""
                };
                format!(
                    "{} commit(s) checked, {} of them sampled and {} row(s) rehashed, the rows \
                     of the rest{genesis} not read",
                    self.commits_checked, self.commits_rehashed, self.rows_checked
                )
            }
        };
        match (&self.failure, self.intact) {
            (Some(failure), _) => format!("{scope}, and {failure}"),
            (None, true) => format!("{scope}, {} anchor(s) agree", self.anchors_checked),
            (None, false) => scope,
        }
    }
}

/// What a walk asks a table for in one pass over it
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RowRequest {
    /// The transactions whose rows are rehashed in stored order, by the
    /// stamp their rows carry
    pub txn_ids: Vec<u64>,
    /// The fence of a genesis entry whose set is rehashed: every row stamped
    /// below it that the table holds, as a set
    pub genesis_fence: Option<u64>,
}

/// What one pass over a table found for a request
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct RowHashes {
    /// Per requested transaction, the hash and count of its rows. A
    /// transaction with no rows found answers with a hasher that took none
    pub by_txn: HashMap<u64, (ChainHash, u64)>,
    /// The genesis set's hash and count, when one was asked for
    pub genesis: Option<(ChainHash, u64)>,
}

/// Where the rows of one commit come from, so the walk reads them without
/// knowing how the table stores them.
pub trait CommitRows {
    /// Hashes the rows of every named commit in one pass over the table.
    ///
    /// One pass rather than one per commit, because a sampled verification
    /// of a table with a hundred thousand commits would otherwise read the
    /// table a hundred thousand times
    fn hash_commits(&self, request: &RowRequest) -> Result<RowHashes>;
}

/// Which commits of a range a sampled pass rehashes.
///
/// The first, the last and an even spread between them, so a sample always
/// covers the ends of the range a reader asked about
pub fn sample_positions(from: u64, to: u64, sample: u64) -> Vec<u64> {
    if to < from {
        return Vec::new();
    }
    let span = to - from + 1;
    if sample == 0 {
        return Vec::new();
    }
    if sample >= span {
        return (from..=to).collect();
    }
    if sample == 1 {
        return vec![from];
    }
    let mut out = Vec::with_capacity(sample as usize);
    for i in 0..sample {
        // Spread across the span with both ends included
        let offset = (i as u128 * (span as u128 - 1)) / (sample as u128 - 1);
        let position = from + offset as u64;
        if out.last() != Some(&position) {
            out.push(position);
        }
    }
    out
}

/// Walks a chain and recomputes what it states.
///
/// Every entry's own link is recomputed whatever the mode, so a chain whose
/// entries were rewritten is caught by the walk alone. The rows are read
/// back for every commit under [`RowMode::All`] and for the sampled
/// positions otherwise, in windows of [`REHASH_WINDOW_COMMITS`] commits
/// per pass over the table. The anchors are checked last, because an anchor
/// a shorter chain contradicts is the case a walk of the chain alone cannot
/// see
#[allow(clippy::too_many_arguments)]
pub fn walk(
    chain: &CommitChain,
    rows: &dyn CommitRows,
    from: u64,
    to: u64,
    mode: RowMode,
    sample: u64,
    anchors: &[anchor::Anchor],
    cancelled: &dyn Fn() -> bool,
) -> Result<VerifyOutcome> {
    let head = chain.head();
    let table_id = chain.table_id();
    let last = to.min(head.commits.saturating_sub(1));
    let mut outcome = VerifyOutcome {
        table_id,
        commits_checked: 0,
        rows_checked: 0,
        anchors_checked: 0,
        mode,
        commits_rehashed: 0,
        genesis_not_read: false,
        intact: true,
        failure: None,
        from_sequence: from,
        to_sequence: last,
        chain_micros: 0,
        rows_micros: 0,
    };
    if head.commits == 0 || from >= head.commits {
        // Nothing in range. The anchors still decide, because an anchor
        // over a chain that now holds nothing is the truncation case
        check_anchors(&mut outcome, chain, anchors)?;
        return Ok(outcome);
    }

    // Which positions have their rows read back. Under `all` that is every
    // one of them, and the rows of a window of commits are read in one pass
    let sampled: Option<std::collections::HashSet<u64>> = match mode {
        RowMode::All => None,
        RowMode::Sampled => Some(sample_positions(from, last, sample).into_iter().collect()),
    };

    let mut prev: Option<ChainHash> = None;
    let mut at = from;
    while at <= last {
        if cancelled() {
            return Err(ZyronError::Internal("Query cancelled".into()));
        }
        let window_last = (at + REHASH_WINDOW_COMMITS - 1).min(last);
        let chain_started = std::time::Instant::now();
        let entries = chain.read_range(at, window_last)?;
        outcome.chain_micros += chain_started.elapsed().as_micros() as u64;
        if entries.len() as u64 != window_last - at + 1 {
            outcome.intact = false;
            outcome.failure = Some(VerifyFailure::MissingCommit {
                sequence: at + entries.len() as u64,
            });
            return Ok(outcome);
        }
        let mut request = RowRequest {
            txn_ids: Vec::new(),
            genesis_fence: None,
        };
        for entry in &entries {
            let wanted = sampled
                .as_ref()
                .is_none_or(|positions| positions.contains(&entry.sequence));
            if !wanted {
                continue;
            }
            if entry.genesis {
                match mode {
                    RowMode::All => request.genesis_fence = Some(entry.txn_id),
                    RowMode::Sampled => outcome.genesis_not_read = true,
                }
            } else {
                request.txn_ids.push(entry.txn_id);
            }
        }
        let hashed = if request.txn_ids.is_empty() && request.genesis_fence.is_none() {
            RowHashes::default()
        } else {
            let rows_started = std::time::Instant::now();
            let hashed = rows.hash_commits(&request)?;
            outcome.rows_micros += rows_started.elapsed().as_micros() as u64;
            hashed
        };

        let links_started = std::time::Instant::now();
        for entry in &entries {
            if cancelled() {
                return Err(ZyronError::Internal("Query cancelled".into()));
            }
            let expected_prev = prev.unwrap_or(entry.prev_hash);
            if entry.prev_hash != expected_prev {
                outcome.intact = false;
                outcome.failure = Some(VerifyFailure::BrokenLink {
                    commit_version: entry.commit_version,
                    sequence: entry.sequence,
                    expected: expected_prev,
                    found: entry.prev_hash,
                });
                return Ok(outcome);
            }
            if entry.compute_entry_hash() != entry.entry_hash {
                outcome.intact = false;
                outcome.failure = Some(VerifyFailure::EntryHashMismatch {
                    commit_version: entry.commit_version,
                    sequence: entry.sequence,
                });
                return Ok(outcome);
            }
            outcome.commits_checked += 1;

            let found = if entry.genesis {
                hashed.genesis
            } else {
                hashed.by_txn.get(&entry.txn_id).copied()
            };
            if let Some((found, found_rows)) = found {
                outcome.rows_checked += found_rows;
                outcome.commits_rehashed += 1;
                if found_rows != entry.row_count {
                    outcome.intact = false;
                    outcome.failure = Some(VerifyFailure::RowCountMismatch {
                        commit_version: entry.commit_version,
                        sequence: entry.sequence,
                        expected: entry.row_count,
                        found: found_rows,
                    });
                    return Ok(outcome);
                }
                if found != entry.rows_hash {
                    outcome.intact = false;
                    outcome.failure = Some(VerifyFailure::RowContentMismatch {
                        commit_version: entry.commit_version,
                        sequence: entry.sequence,
                        expected: entry.rows_hash,
                        found,
                    });
                    return Ok(outcome);
                }
            }
            prev = Some(entry.entry_hash);
        }
        outcome.chain_micros += links_started.elapsed().as_micros() as u64;
        at = window_last + 1;
    }

    check_anchors(&mut outcome, chain, anchors)?;
    Ok(outcome)
}

/// Holds the chain against every anchor taken over it.
///
/// An anchor names the head the chain stood at at a version. A chain that no
/// longer reaches that version, or that stands at a different head there, is
/// the truncation a walk of the chain alone reads as a shorter, consistent
/// chain
fn check_anchors(
    outcome: &mut VerifyOutcome,
    chain: &CommitChain,
    anchors: &[anchor::Anchor],
) -> Result<()> {
    let head = chain.head();
    for anchor in anchors {
        if anchor.table_id != chain.table_id() {
            continue;
        }
        outcome.anchors_checked += 1;
        if !outcome.intact {
            continue;
        }
        if anchor.sequence >= head.commits {
            outcome.intact = false;
            outcome.failure = Some(VerifyFailure::AnchorContradiction {
                sequence: anchor.sequence,
                anchored_version: anchor.commit_version,
                anchored_head: anchor.head_hash,
                found_head: None,
                anchored_at: anchor.taken_at,
            });
            continue;
        }
        let found = chain.entry_hash_at(anchor.sequence)?;
        if found != Some(anchor.head_hash) {
            outcome.intact = false;
            outcome.failure = Some(VerifyFailure::AnchorContradiction {
                sequence: anchor.sequence,
                anchored_version: anchor.commit_version,
                anchored_head: anchor.head_hash,
                found_head: found,
                anchored_at: anchor.taken_at,
            });
        }
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// What was verified, and when
// ---------------------------------------------------------------------------

/// One verification this node ran.
#[derive(Debug, Clone, PartialEq)]
pub struct VerifyRun {
    pub table_id: u32,
    pub table_name: String,
    /// The role that asked for it
    pub actor: u32,
    pub actor_name: String,
    pub from_version: u64,
    pub to_version: u64,
    pub mode: RowMode,
    pub commits_checked: u64,
    pub rows_checked: u64,
    pub anchors_checked: u64,
    pub intact: bool,
    /// What was found, empty for a run that found nothing wrong
    pub finding: String,
    pub started_at: i64,
    pub duration_micros: u64,
}

impl VerifyRun {
    fn encode(&self, out: &mut Vec<u8>) {
        out.extend_from_slice(&self.table_id.to_le_bytes());
        write_text(out, &self.table_name);
        out.extend_from_slice(&self.actor.to_le_bytes());
        write_text(out, &self.actor_name);
        out.extend_from_slice(&self.from_version.to_le_bytes());
        out.extend_from_slice(&self.to_version.to_le_bytes());
        out.push(match self.mode {
            RowMode::All => 1,
            RowMode::Sampled => 0,
        });
        out.extend_from_slice(&self.commits_checked.to_le_bytes());
        out.extend_from_slice(&self.rows_checked.to_le_bytes());
        out.extend_from_slice(&self.anchors_checked.to_le_bytes());
        out.push(if self.intact { 1 } else { 0 });
        write_text(out, &self.finding);
        out.extend_from_slice(&self.started_at.to_le_bytes());
        out.extend_from_slice(&self.duration_micros.to_le_bytes());
    }

    fn decode(data: &[u8], at: &mut usize) -> Result<Self> {
        Ok(Self {
            table_id: read_u32(data, at)?,
            table_name: read_text(data, at)?,
            actor: read_u32(data, at)?,
            actor_name: read_text(data, at)?,
            from_version: read_u64(data, at)?,
            to_version: read_u64(data, at)?,
            mode: if read_u8(data, at)? == 1 {
                RowMode::All
            } else {
                RowMode::Sampled
            },
            commits_checked: read_u64(data, at)?,
            rows_checked: read_u64(data, at)?,
            anchors_checked: read_u64(data, at)?,
            intact: read_u8(data, at)? == 1,
            finding: read_text(data, at)?,
            started_at: read_u64(data, at)? as i64,
            duration_micros: read_u64(data, at)?,
        })
    }
}

/// The verifications this node ran, newest first.
///
/// Held whole and rewritten on each run, bounded to the most recent
/// [`RUN_HISTORY`] so a node that verifies on a schedule does not grow a
/// file without end
#[derive(Debug)]
pub struct RunLog {
    path: PathBuf,
    runs: Mutex<VecDeque<VerifyRun>>,
}

/// Runs the history keeps
pub const RUN_HISTORY: usize = 512;

/// Version the run log is written at
pub const RUN_LOG_FORMAT_VERSION: FormatVersion = FormatVersion::V1;

impl RunLog {
    fn open(dir: &Path) -> Result<Self> {
        let path = dir.join("verify").join("runs.zvrl");
        let runs = if path.exists() {
            let bytes = std::fs::read(&path).map_err(|e| {
                ZyronError::Internal(format!(
                    "the verification history at {} could not be read, {e}",
                    path.display()
                ))
            })?;
            let body = envelope::decode_as(&bytes, FormatKind::VerificationRunLog)?;
            let mut at = 0usize;
            let count = read_u32(&body.body, &mut at)? as usize;
            let mut runs = VecDeque::with_capacity(count.min(RUN_HISTORY));
            for _ in 0..count {
                runs.push_back(VerifyRun::decode(&body.body, &mut at)?);
            }
            runs
        } else {
            VecDeque::new()
        };
        Ok(Self {
            path,
            runs: Mutex::new(runs),
        })
    }

    /// Records one run and writes the history down.
    ///
    /// A run that could not be written down is an error rather than a
    /// warning: a verification nobody can read afterwards is not evidence.
    /// The history is held under its lock while it is written, so two runs
    /// that finish together are both in the file
    pub fn record(&self, run: VerifyRun) -> Result<()> {
        let mut runs = self.runs.lock();
        runs.push_front(run);
        runs.truncate(RUN_HISTORY);
        let mut body = Vec::with_capacity(256 * runs.len() + 4);
        body.extend_from_slice(&(runs.len() as u32).to_le_bytes());
        for run in runs.iter() {
            run.encode(&mut body);
        }
        let bytes = envelope::encode(
            FormatKind::VerificationRunLog,
            RUN_LOG_FORMAT_VERSION,
            &body,
        );
        write_replacing(&self.path, &bytes)
    }

    /// Every run the history holds, newest first.
    pub fn all(&self) -> Vec<VerifyRun> {
        self.runs.lock().iter().cloned().collect()
    }

    /// The most recent run over one table.
    pub fn latest_for(&self, table_id: u32) -> Option<VerifyRun> {
        self.runs
            .lock()
            .iter()
            .find(|run| run.table_id == table_id)
            .cloned()
    }
}

// ---------------------------------------------------------------------------
// Shared encoding helpers
// ---------------------------------------------------------------------------

pub(crate) fn write_text(out: &mut Vec<u8>, text: &str) {
    out.extend_from_slice(&(text.len() as u32).to_le_bytes());
    out.extend_from_slice(text.as_bytes());
}

pub(crate) fn read_text(data: &[u8], at: &mut usize) -> Result<String> {
    let len = read_u32(data, at)? as usize;
    if data.len() < *at + len {
        return Err(ZyronError::Internal(
            "a verification record ends inside one of its fields".to_string(),
        ));
    }
    let text = String::from_utf8(data[*at..*at + len].to_vec())
        .map_err(|e| ZyronError::Internal(format!("a verification record holds bad text, {e}")))?;
    *at += len;
    Ok(text)
}

pub(crate) fn read_u8(data: &[u8], at: &mut usize) -> Result<u8> {
    if data.len() < *at + 1 {
        return Err(ZyronError::Internal(
            "a verification record ends inside one of its fields".to_string(),
        ));
    }
    let value = data[*at];
    *at += 1;
    Ok(value)
}

pub(crate) fn read_u32(data: &[u8], at: &mut usize) -> Result<u32> {
    if data.len() < *at + 4 {
        return Err(ZyronError::Internal(
            "a verification record ends inside one of its fields".to_string(),
        ));
    }
    let mut buf = [0u8; 4];
    buf.copy_from_slice(&data[*at..*at + 4]);
    *at += 4;
    Ok(u32::from_le_bytes(buf))
}

pub(crate) fn read_u64(data: &[u8], at: &mut usize) -> Result<u64> {
    if data.len() < *at + 8 {
        return Err(ZyronError::Internal(
            "a verification record ends inside one of its fields".to_string(),
        ));
    }
    let mut buf = [0u8; 8];
    buf.copy_from_slice(&data[*at..*at + 8]);
    *at += 8;
    Ok(u64::from_le_bytes(buf))
}

pub(crate) fn read_hash(data: &[u8], at: &mut usize) -> Result<ChainHash> {
    if data.len() < *at + 32 {
        return Err(ZyronError::Internal(
            "a verification record ends inside one of its hashes".to_string(),
        ));
    }
    let mut hash = [0u8; 32];
    hash.copy_from_slice(&data[*at..*at + 32]);
    *at += 32;
    Ok(hash)
}

/// Writes a file by writing a temporary beside it and renaming, so a stop
/// leaves either the file as it was or the file as it is meant to be.
/// Callers serialize their writes to one path under their own lock
pub(crate) fn write_replacing(path: &Path, bytes: &[u8]) -> Result<()> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).map_err(|e| {
            ZyronError::Internal(format!(
                "the verification directory {} could not be created, {e}",
                parent.display()
            ))
        })?;
    }
    let temporary = path.with_extension("writing");
    {
        let mut file = File::create(&temporary).map_err(|e| {
            ZyronError::Internal(format!("{} could not be written, {e}", temporary.display()))
        })?;
        file.write_all(bytes).map_err(|e| {
            ZyronError::Internal(format!("{} could not be written, {e}", temporary.display()))
        })?;
        file.sync_all().map_err(|e| {
            ZyronError::Internal(format!("{} could not be flushed, {e}", temporary.display()))
        })?;
    }
    std::fs::rename(&temporary, path).map_err(|e| {
        ZyronError::Internal(format!("{} could not be put in place, {e}", path.display()))
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rows(n: u64) -> (ChainHash, u64) {
        let mut hasher = RowsHasher::new();
        for i in 0..n {
            hasher.row(1, &i.to_le_bytes());
        }
        (hasher.finish(), n)
    }

    fn fields(txn_id: u64, rows_hash: ChainHash, row_count: u64, commit_ts: i64) -> CommitFields {
        CommitFields {
            txn_id,
            rows_hash,
            row_count,
            commit_ts,
            algorithm_id: 4,
            genesis: false,
        }
    }

    fn registry_chaining(dir: &Path, table_id: u32, fence: u64) -> ChainRegistry {
        let registry = ChainRegistry::open(dir.to_path_buf()).expect("opens");
        registry.chain_from(table_id, fence, 4, 0);
        registry
    }

    #[test]
    fn test_a_chain_record_is_the_documented_size() {
        let entry = CommitHash::link(1, 0, NO_PREVIOUS, 10, fields(55, [7u8; 32], 3, 100));
        assert_eq!(entry.encode().len(), CHAIN_RECORD_LEN);
        assert_eq!(CHAIN_RECORD_LEN, 96);
    }

    #[test]
    fn test_an_entry_round_trips_through_its_record() {
        let entry = CommitHash::link(9, 4, [3u8; 32], 77, fields(88, [8u8; 32], 21, 1234));
        let record = entry.encode();
        let back = CommitHash::decode(9, 4, [3u8; 32], &record);
        assert_eq!(back, entry);
        assert_eq!(back.compute_entry_hash(), back.entry_hash);
        let genesis = CommitHash::link(
            9,
            0,
            NO_PREVIOUS,
            1,
            CommitFields {
                genesis: true,
                ..fields(500, [2u8; 32], 9, 5)
            },
        );
        let back = CommitHash::decode(9, 0, NO_PREVIOUS, &genesis.encode());
        assert!(back.genesis);
        assert_eq!(back, genesis);
    }

    /// The link covers every field that states what the commit wrote and
    /// where it stands, and not the transaction id, which is the key one
    /// member reads the rows back by
    #[test]
    fn test_the_link_covers_every_field_but_the_transaction() {
        let base = CommitHash::link(1, 0, NO_PREVIOUS, 10, fields(40, [1u8; 32], 5, 50));
        let moved_version = CommitHash::link(1, 0, NO_PREVIOUS, 11, fields(40, [1u8; 32], 5, 50));
        let moved_rows = CommitHash::link(1, 0, NO_PREVIOUS, 10, fields(40, [2u8; 32], 5, 50));
        let moved_count = CommitHash::link(1, 0, NO_PREVIOUS, 10, fields(40, [1u8; 32], 6, 50));
        let moved_ts = CommitHash::link(1, 0, NO_PREVIOUS, 10, fields(40, [1u8; 32], 5, 51));
        let moved_table = CommitHash::link(2, 0, NO_PREVIOUS, 10, fields(40, [1u8; 32], 5, 50));
        let moved_algorithm = CommitHash::link(
            1,
            0,
            NO_PREVIOUS,
            10,
            CommitFields {
                algorithm_id: 5,
                ..fields(40, [1u8; 32], 5, 50)
            },
        );
        let moved_prev = CommitHash::link(1, 0, [6u8; 32], 10, fields(40, [1u8; 32], 5, 50));
        let moved_flag = CommitHash::link(
            1,
            0,
            NO_PREVIOUS,
            10,
            CommitFields {
                genesis: true,
                ..fields(40, [1u8; 32], 5, 50)
            },
        );
        for other in [
            moved_version,
            moved_rows,
            moved_count,
            moved_ts,
            moved_table,
            moved_algorithm,
            moved_prev,
            moved_flag,
        ] {
            assert_ne!(base.entry_hash, other.entry_hash);
        }
        let moved_txn = CommitHash::link(1, 0, NO_PREVIOUS, 10, fields(41, [1u8; 32], 5, 50));
        assert_eq!(base.entry_hash, moved_txn.entry_hash);
    }

    /// The digest driven onto the block compression is the standard's, for
    /// every length around the padding boundaries and for bytes split
    /// across parts any way at all
    #[test]
    fn test_the_padded_digest_is_the_streaming_digest() {
        let bytes: Vec<u8> = (0..300u32).map(|n| (n * 7 % 251) as u8).collect();
        for len in [
            0usize, 1, 31, 55, 56, 57, 63, 64, 65, 95, 119, 120, 127, 128, 129, 300,
        ] {
            let expected: ChainHash = Sha256::digest(&bytes[..len]).into();
            assert_eq!(sha256_of(&[&bytes[..len]]), expected, "one part of {len}");
            let (head, tail) = bytes[..len].split_at(len / 3);
            assert_eq!(sha256_of(&[head, tail]), expected, "two parts of {len}");
            let parts: Vec<&[u8]> = bytes[..len].chunks(7).collect();
            assert_eq!(sha256_of(&parts), expected, "many parts of {len}");
        }
        let empty: ChainHash = Sha256::digest([]).into();
        assert_eq!(sha256_of(&[]), empty);
    }

    #[test]
    fn test_the_epoch_is_part_of_what_a_row_hashes_to() {
        let mut one = RowsHasher::new();
        one.row(1, b"abc");
        let mut two = RowsHasher::new();
        two.row(2, b"abc");
        assert_ne!(one.finish(), two.finish());
    }

    #[test]
    fn test_row_boundaries_are_part_of_the_hash() {
        let mut one = RowsHasher::new();
        one.row(1, b"ab");
        one.row(1, b"c");
        let mut two = RowsHasher::new();
        two.row(1, b"abc");
        assert_ne!(
            one.finish(),
            two.finish(),
            "two rows must not hash as one row of their bytes"
        );
    }

    /// The genesis set hashes the same whatever order its rows arrive in,
    /// and differently from the same rows as a sequence
    #[test]
    fn test_a_set_hashes_the_same_in_any_order() {
        let dir = tempfile::TempDir::new().expect("temp dir");
        let mut forward = SetHasher::new(dir.path());
        let mut backward = SetHasher::new(dir.path());
        let mut sequence = RowsHasher::new();
        for i in 0..100u32 {
            forward.row(1, &i.to_le_bytes()).expect("takes");
            sequence.row(1, &i.to_le_bytes());
        }
        for i in (0..100u32).rev() {
            backward.row(1, &i.to_le_bytes()).expect("takes");
        }
        let forward = forward.finish().expect("finishes");
        let backward = backward.finish().expect("finishes");
        assert_eq!(forward, backward);
        assert_ne!(forward, sequence.finish());
    }

    /// A set larger than one in-memory run is sorted in runs on disk and
    /// merged, and hashes the same as one that fit in memory
    #[test]
    fn test_a_spilled_set_hashes_the_same_as_a_held_one() {
        let dir = tempfile::TempDir::new().expect("temp dir");
        let run = 1_000usize;
        let count = run as u32 * 2 + 17;
        let mut spilled = SetHasher::with_run_capacity(dir.path().join("spilled"), run);
        for i in 0..count {
            spilled.row(1, &i.to_le_bytes()).expect("takes");
        }
        // The same digests, sorted in memory in one run
        let mut digests: Vec<ChainHash> = (0..count)
            .map(|i| RowsHasher::row_digest(1, &i.to_le_bytes()))
            .collect();
        digests.sort_unstable();
        let mut hasher = Sha256::new();
        for digest in &digests {
            hasher.update(digest);
        }
        let held: ChainHash = hasher.finalize().into();
        assert_eq!(spilled.rows(), count as u64);
        assert_eq!(spilled.finish().expect("finishes"), held);
        let leftover = std::fs::read_dir(dir.path().join("spilled"))
            .map(|entries| entries.count())
            .unwrap_or(0);
        assert_eq!(leftover, 0, "the runs are removed once the set is hashed");
    }

    #[test]
    fn test_a_chain_reopens_at_the_head_it_left() {
        let dir = tempfile::TempDir::new().expect("temp dir");
        let chain = CommitChain::open(dir.path(), 7).expect("opens");
        assert_eq!(chain.head().commits, 0);
        let (hash, count) = rows(4);
        let first = chain
            .append(fields(11, hash, count, 1_000), || 100)
            .expect("appends");
        let second = chain
            .append(fields(12, hash, count, 2_000), || 200)
            .expect("appends");
        // The entries reach the file when their transactions commit, which
        // is what a reader and a reopen see
        assert_eq!(chain.head().commits, 0, "nothing has committed yet");
        chain.publish(11).expect("publishes");
        chain.publish(12).expect("publishes");
        chain.sync().expect("flushes");
        assert_eq!(second.prev_hash, first.entry_hash);
        assert_eq!(chain.head().commits, 2);

        drop(chain);
        let again = CommitChain::open(dir.path(), 7).expect("reopens");
        assert_eq!(again.head().commits, 2);
        assert_eq!(again.head().head_hash, second.entry_hash);
        assert_eq!(again.head().head_version, 200);
    }

    #[test]
    fn test_a_chain_file_holds_one_record_per_commit() {
        let dir = tempfile::TempDir::new().expect("temp dir");
        let chain = CommitChain::open(dir.path(), 3).expect("opens");
        let (hash, count) = rows(2);
        for version in 1..=10u64 {
            chain
                .append(fields(version, hash, count, version as i64), || version)
                .expect("appends");
            chain.publish(version).expect("publishes");
        }
        chain.sync().expect("flushes");
        let length = std::fs::metadata(chain.path()).expect("measures").len();
        assert_eq!(
            length,
            CHAIN_HEADER_LEN as u64 + 10 * CHAIN_RECORD_LEN as u64
        );
        assert_eq!(chain.head().bytes(), 10 * 96);
    }

    /// A version is read under the chain lock and never falls below the
    /// head's, so versions rise along the chain and a range resolves by
    /// binary search
    #[test]
    fn test_versions_rise_along_the_chain_and_resolve_a_range() {
        let dir = tempfile::TempDir::new().expect("temp dir");
        let chain = CommitChain::open(dir.path(), 4).expect("opens");
        let (hash, count) = rows(1);
        let versions = [10u64, 20, 20, 5, 40];
        for (i, version) in versions.iter().enumerate() {
            let txn = i as u64 + 1;
            chain
                .append(fields(txn, hash, count, txn as i64), || *version)
                .expect("appends");
            chain.publish(txn).expect("publishes");
        }
        let read = chain.read_range(0, 4).expect("reads");
        let stored: Vec<u64> = read.iter().map(|e| e.commit_version).collect();
        assert_eq!(stored, vec![10, 20, 20, 20, 40]);
        assert_eq!(chain.first_at_or_above(20).expect("resolves"), 1);
        assert_eq!(chain.first_at_or_above(21).expect("resolves"), 4);
        assert_eq!(chain.first_at_or_above(41).expect("resolves"), 5);
        assert_eq!(chain.last_at_or_below(20).expect("resolves"), Some(3));
        assert_eq!(chain.last_at_or_below(9).expect("resolves"), None);
        assert_eq!(chain.last_at_or_below(100).expect("resolves"), Some(4));
    }

    #[test]
    fn test_a_range_read_carries_the_link_of_the_entry_before_it() {
        let dir = tempfile::TempDir::new().expect("temp dir");
        let chain = CommitChain::open(dir.path(), 5).expect("opens");
        let (hash, count) = rows(1);
        let mut written = Vec::new();
        for version in 1..=6u64 {
            written.push(
                chain
                    .append(fields(version, hash, count, version as i64), || version)
                    .expect("appends"),
            );
            chain.publish(version).expect("publishes");
        }
        let read = chain.read_range(2, 4).expect("reads");
        assert_eq!(read.len(), 3);
        assert_eq!(read[0].prev_hash, written[1].entry_hash);
        for entry in &read {
            assert_eq!(entry.compute_entry_hash(), entry.entry_hash);
        }
        assert_eq!(
            chain.entry_hash_at(3).expect("reads"),
            Some(written[3].entry_hash)
        );
        assert_eq!(chain.entry_hash_at(6).expect("reads"), None);
    }

    /// A chain longer than one read window is read whole, in windows
    #[test]
    fn test_a_long_range_is_read_in_windows() {
        let dir = tempfile::TempDir::new().expect("temp dir");
        let chain = CommitChain::open(dir.path(), 6).expect("opens");
        let (hash, count) = rows(1);
        let total = READ_WINDOW_RECORDS * 2 + 5;
        for version in 1..=total {
            chain
                .append(fields(version, hash, count, version as i64), || version)
                .expect("appends");
            chain.publish(version).expect("publishes");
        }
        let read = chain.read_range(3, total - 1).expect("reads");
        assert_eq!(read.len() as u64, total - 3);
        let mut prev = read[0].prev_hash;
        for entry in &read {
            assert_eq!(entry.prev_hash, prev);
            prev = entry.entry_hash;
        }
    }

    /// A stop between a write and its flush leaves part of a record at the
    /// end of the file. The reopen cuts it off, so the next record lands on
    /// a record boundary and the chain stays readable
    #[test]
    fn test_a_partial_record_at_the_end_is_cut_off_on_reopen() {
        let dir = tempfile::TempDir::new().expect("temp dir");
        let chain = CommitChain::open(dir.path(), 8).expect("opens");
        let (hash, count) = rows(1);
        chain
            .append(fields(1, hash, count, 1), || 1)
            .expect("appends");
        chain.publish(1).expect("publishes");
        chain.sync().expect("flushes");
        let path = chain.path().to_path_buf();
        drop(chain);
        {
            let mut file = OpenOptions::new()
                .append(true)
                .open(&path)
                .expect("opens for append");
            file.write_all(&[0xAB; 40]).expect("writes a partial tail");
        }
        let again = CommitChain::open(dir.path(), 8).expect("reopens");
        assert_eq!(again.head().commits, 1);
        let length = std::fs::metadata(&path).expect("measures").len();
        assert_eq!(length, CHAIN_HEADER_LEN as u64 + CHAIN_RECORD_LEN as u64);
        again
            .append(fields(2, hash, count, 2), || 2)
            .expect("appends");
        again.publish(2).expect("publishes");
        let read = again.read_range(0, 1).expect("reads");
        assert_eq!(read.len(), 2);
        assert_eq!(read[1].compute_entry_hash(), read[1].entry_hash);
    }

    #[test]
    fn test_adopting_an_entry_out_of_position_is_refused() {
        let dir = tempfile::TempDir::new().expect("temp dir");
        let chain = CommitChain::open(dir.path(), 11).expect("opens");
        let (hash, count) = rows(1);
        chain
            .append(fields(1, hash, count, 1), || 1)
            .expect("appends");
        chain.publish(1).expect("publishes");
        let ahead = CommitHash::link(11, 5, [9u8; 32], 9, fields(9, hash, count, 9));
        let refused = chain.adopt(&ahead).expect_err("refused");
        assert!(refused.to_string().contains("gap"), "{refused}");
    }

    /// An entry at the right position that does not hash to the link it
    /// carries is not an entry of this chain
    #[test]
    fn test_adopting_an_entry_that_does_not_link_is_refused() {
        let dir = tempfile::TempDir::new().expect("temp dir");
        let chain = CommitChain::open(dir.path(), 13).expect("opens");
        let (hash, count) = rows(1);
        let mut forged = CommitHash::link(13, 0, NO_PREVIOUS, 1, fields(1, hash, count, 1));
        forged.row_count += 1;
        let refused = chain.adopt(&forged).expect_err("refused");
        assert!(refused.to_string().contains("does not hash"), "{refused}");
        assert_eq!(chain.head().commits, 0);
    }

    #[test]
    fn test_adopting_an_entry_already_held_changes_nothing() {
        let dir = tempfile::TempDir::new().expect("temp dir");
        let chain = CommitChain::open(dir.path(), 12).expect("opens");
        let (hash, count) = rows(1);
        let first = chain
            .append(fields(1, hash, count, 1), || 1)
            .expect("appends");
        chain.publish(1).expect("publishes");
        assert!(!chain.adopt(&first).expect("adopts"), "already held");
        assert_eq!(chain.head().commits, 1);
    }

    #[test]
    fn test_a_sample_covers_both_ends_of_its_range() {
        assert_eq!(sample_positions(0, 9, 3), vec![0, 4, 9]);
        assert_eq!(sample_positions(0, 2, 10), vec![0, 1, 2]);
        assert_eq!(sample_positions(5, 5, 4), vec![5]);
        assert!(sample_positions(0, 9, 0).is_empty());
        let spread = sample_positions(0, 99, 10);
        assert_eq!(spread.first(), Some(&0));
        assert_eq!(spread.last(), Some(&99));
    }

    /// A verification reads a commit's rows back in the order they are
    /// stored. A run hashed after one that sits above it would hash a
    /// sequence no read can reproduce, so the commit is refused rather than
    /// leaving an entry that a verification would contradict
    #[test]
    fn test_a_run_stored_below_one_already_taken_fails_the_commit() {
        let dir = tempfile::TempDir::new().expect("temp dir");
        let registry = registry_chaining(dir.path(), 1, 0);
        let pending = PendingChainWrites::new();
        pending.record(1, 1, &mut [b"a".as_slice()].into_iter(), 100, 199);
        pending.record(1, 1, &mut [b"b".as_slice()].into_iter(), 50, 60);
        let refused = pending.take(7, &registry).expect_err("refused");
        assert!(refused.to_string().contains("order"), "{refused}");
    }

    #[test]
    fn test_runs_stored_in_ascending_order_are_taken() {
        let dir = tempfile::TempDir::new().expect("temp dir");
        let registry = registry_chaining(dir.path(), 1, 0);
        let pending = PendingChainWrites::new();
        pending.record(1, 1, &mut [b"a".as_slice()].into_iter(), 0, 9);
        pending.record(1, 1, &mut [b"b".as_slice()].into_iter(), 10, 19);
        let taken = pending.take(7, &registry).expect("takes");
        assert_eq!(taken.len(), 1);
        assert_eq!(taken[0].row_count, 2);
        assert_eq!(taken[0].algorithm_id, 4);
    }

    /// A transaction that started before the table was chained and wrote
    /// to it is refused at commit, whether it hashed the rows or not
    #[test]
    fn test_a_transaction_from_before_the_fence_is_refused() {
        let dir = tempfile::TempDir::new().expect("temp dir");
        let registry = registry_chaining(dir.path(), 1, 100);
        let hashed = PendingChainWrites::new();
        hashed.record(1, 1, &mut [b"a".as_slice()].into_iter(), 0, 0);
        let refused = hashed.take(99, &registry).expect_err("refused");
        assert!(
            refused.to_string().contains("became verifiable"),
            "{refused}"
        );

        let touched = PendingChainWrites::new();
        touched.touched(1);
        assert!(!touched.is_empty());
        let refused = touched.take(99, &registry).expect_err("refused");
        assert!(
            refused.to_string().contains("became verifiable"),
            "{refused}"
        );

        let after = PendingChainWrites::new();
        after.record(1, 1, &mut [b"a".as_slice()].into_iter(), 0, 0);
        assert_eq!(after.take(100, &registry).expect("takes").len(), 1);

        // A table the registry does not chain is written freely
        let elsewhere = PendingChainWrites::new();
        elsewhere.touched(2);
        assert!(elsewhere.take(1, &registry).expect("takes").is_empty());
        assert!(elsewhere.is_empty());
    }

    /// A chain covers rows that stay, so a transaction that removed rows
    /// from a chained table is refused whatever its id, and one that removed
    /// rows from an unchained table is not
    #[test]
    fn test_a_transaction_that_removed_rows_from_a_chained_table_is_refused() {
        let dir = tempfile::TempDir::new().expect("temp dir");
        let registry = registry_chaining(dir.path(), 1, 0);
        let pending = PendingChainWrites::new();
        pending.removed(1);
        assert!(!pending.is_empty());
        let refused = pending.take(500, &registry).expect_err("refused");
        assert!(refused.to_string().contains("removed rows"), "{refused}");

        let elsewhere = PendingChainWrites::new();
        elsewhere.removed(2);
        assert!(elsewhere.take(500, &registry).expect("takes").is_empty());
    }

    #[test]
    fn test_a_commit_above_what_a_record_states_is_refused() {
        let dir = tempfile::TempDir::new().expect("temp dir");
        let chain = CommitChain::open(dir.path(), 21).expect("opens");
        let refused = chain
            .append(fields(1, [0u8; 32], MAX_COMMIT_ROWS + 1, 1), || 1)
            .expect_err("refused");
        assert!(
            refused.to_string().contains("chain entry can state"),
            "{refused}"
        );
        assert_eq!(chain.head().commits, 0);
    }

    #[test]
    fn test_hex_round_trips() {
        let hash = [0xabu8; 32];
        assert_eq!(from_hex(&hex(&hash)), Some(hash));
        assert_eq!(from_hex("short"), None);
        let mut mixed = [0u8; 32];
        mixed[0] = 0x01;
        mixed[31] = 0xfe;
        let text = hex(&mixed);
        assert!(text.starts_with("01"));
        assert!(text.ends_with("fe"));
        assert_eq!(from_hex(&text), Some(mixed));
    }

    #[test]
    fn test_pending_writes_hand_back_one_hash_per_table_in_table_order() {
        let dir = tempfile::TempDir::new().expect("temp dir");
        let registry = registry_chaining(dir.path(), 9, 0);
        registry.chain_from(2, 0, 4, 0);
        let pending = PendingChainWrites::new();
        assert!(pending.is_empty());
        pending.record(9, 1, &mut [b"a".as_slice()].into_iter(), 0, 0);
        pending.record(
            2,
            1,
            &mut [b"b".as_slice(), b"c".as_slice()].into_iter(),
            0,
            1,
        );
        assert!(!pending.is_empty());
        let taken = pending.take(5, &registry).expect("takes");
        assert_eq!(taken.len(), 2);
        assert_eq!(taken[0].table_id, 2);
        assert_eq!(taken[0].row_count, 2);
        assert_eq!(taken[1].table_id, 9);
        assert!(pending.is_empty());
    }

    /// The cursor a transaction stores its rows through is one per table
    /// and starts unplaced, so the heap seeds it from the writing thread
    #[test]
    fn test_one_cursor_per_table_per_transaction() {
        let pending = PendingChainWrites::new();
        let first = pending.cursor(4);
        let again = pending.cursor(4);
        let other = pending.cursor(5);
        assert!(Arc::ptr_eq(&first, &again));
        assert!(!Arc::ptr_eq(&first, &other));
        assert_eq!(first.load(std::sync::atomic::Ordering::Relaxed), u32::MAX);
        pending.clear();
        assert!(pending.is_empty());
    }
}
