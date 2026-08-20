//! Re-clustering: a compaction that changes row order rather than row
//! content.
//!
//! One invariant carries the safety of the whole module: no path here
//! opens an existing `data/*.zyr` for write. A pass reads its inputs,
//! writes candidates into `_staging/pass_<id>/`, scores them against the
//! inputs with the feedback gate, and only on accept renames them into
//! `data/` and commits one version removing the inputs and adding the
//! outputs. On reject the staging directory is unlinked and the active
//! file set is byte identical to what it was before the pass started, so
//! a refused proposal costs disk and CPU and nothing else.
//!
//! The rewrite also applies whatever delete predicates attached to the
//! inputs, because it has to read through their survivor masks anyway,
//! and retires the predicates no surviving file references afterwards.
//!
//! Ordering. Rows already under the target spec are merged rather than
//! sorted: each input is already ordered by that spec's curve, so a k-way
//! merge over the leading eight bytes of the ordering key produces the
//! global order in one pass with no comparison sort. Rows under a
//! different spec have no such order to exploit and are sorted on the
//! full key. The merge only decides which rows share an output file,
//! never the order inside one, because the writer sorts every batch it is
//! given on the exact full key.
//!
//! Durability. A pass is resumable from two append-only structures, both
//! CRC'd per record and fsynced at the point that matters:
//!
//! * `clustering/pass_<id>.zycluster` describes the pass: a 128-byte
//!   header, then 64-byte records carrying the target keys, the chosen
//!   inputs and every state transition.
//! * `_staging/pass_<id>/_staged.zyent` describes the staged file set:
//!   one encoded manifest entry per finalized output, fsynced as the
//!   output finalizes. Entries are variable length because they carry
//!   column statistics and value blooms, which is why they live beside
//!   the files they describe rather than in the fixed-width record log.
//!
//! A torn trailing record in either fails its CRC and is ignored, so a
//! crash mid-append costs the last record and nothing before it.
//!
//! Resume dispatches on the last recorded state: `Committing` inspects
//! the log to see whether the commit landed, `Staged` re-runs the gate,
//! `Running` continues at the first output the sidecar does not record.
//! There is no separate gating state because a crash during gating and a
//! crash after staging recover identically, by scoring again.
//!
//! Time travel is unaffected. `RemoveFile` is logical, the input files
//! stay on disk until vacuum passes their retention floor, and every file
//! records the cluster spec id it was written under, so reading a past
//! version reads that version's layout and its clustering metadata.

use std::cell::RefCell;
use std::cmp::Ordering;
use std::collections::{BTreeMap, BTreeSet};
use std::fs::{self, File, OpenOptions};
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::rc::Rc;

use zyron_common::ZyronError;
use zyron_storage::columnar::MergeScanIterator;

use crate::codec::{Cursor, corrupt};
use crate::curve::{normalize_component, ordering_key};
use crate::feedback::{Decision, GateConfig, PredicateClass, evaluate};
use crate::index;
use crate::manifest::{
    ClusterKey, ClusterMode, ClusterSpec, ClusterStrategy, ManifestFile, PartitionEntry,
    decode_partition_entry, encode_partition_entry,
};
use crate::operations::{allocate_partition_id, allocate_unused_partition_id};
use crate::paths::data_file_name;
use crate::predicate::LakeValue;
use crate::reader::{DecodedColumn, LakeFileReader};
use crate::transaction_log::{CommitAttempt, LogEntry, OperationKind, TransactionLog};
use crate::writer::{ColumnData, WriteRequest, write_data_file_at};

const CHECKPOINT_MAGIC: [u8; 8] = *b"ZYCLUSTR";
const CHECKPOINT_FORMAT_VERSION: u32 = 1;
const CHECKPOINT_HEADER_LEN: usize = 128;
const CHECKPOINT_RECORD_LEN: usize = 64;

/// Sidecar inside a pass staging directory holding one encoded manifest
/// entry per finalized output
const STAGED_ENTRIES_NAME: &str = "_staged.zyent";

/// Default rows per output file. Large enough that a pass produces few
/// files, small enough that one output batch is a bounded allocation
pub const DEFAULT_ROWS_PER_FILE: u64 = 1_048_576;

/// Default number of input files one pass rewrites. This is what bounds
/// the memory a pass holds, because every input is decoded once and kept
/// for the duration of the rewrite
pub const DEFAULT_MAX_INPUTS: usize = 16;

/// How far a pass got, durably. Each value means "this much is on disk",
/// so recovery reads the last one and continues from there
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PassState {
    /// Inputs chosen, outputs being written into staging
    Running,
    /// Every output written, fsynced and recorded
    Staged,
    /// Gate accepted, renames and the commit are in flight
    Committing,
    /// Version committed
    Done,
    /// Gate refused, the active set was never touched
    Rejected,
}

impl PassState {
    fn to_u8(self) -> u8 {
        match self {
            PassState::Running => 1,
            PassState::Staged => 2,
            PassState::Committing => 3,
            PassState::Done => 4,
            PassState::Rejected => 5,
        }
    }

    fn from_u8(v: u8) -> Option<Self> {
        Some(match v {
            1 => PassState::Running,
            2 => PassState::Staged,
            3 => PassState::Committing,
            4 => PassState::Done,
            5 => PassState::Rejected,
            _ => return None,
        })
    }
}

/// Fixed 128-byte checkpoint header at offset 0
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct PassHeader {
    pass_id: u64,
    /// Manifest version the pass planned against, what resume replans on
    base_version: u64,
    table_id: u64,
    target_rows_per_file: u64,
    created_us: i64,
    target_spec_id: u32,
}

impl PassHeader {
    fn to_bytes(self) -> [u8; CHECKPOINT_HEADER_LEN] {
        let mut buf = [0u8; CHECKPOINT_HEADER_LEN];
        buf[0..8].copy_from_slice(&CHECKPOINT_MAGIC);
        buf[8..12].copy_from_slice(&CHECKPOINT_FORMAT_VERSION.to_le_bytes());
        buf[16..24].copy_from_slice(&self.pass_id.to_le_bytes());
        buf[24..32].copy_from_slice(&self.base_version.to_le_bytes());
        buf[32..40].copy_from_slice(&self.table_id.to_le_bytes());
        buf[40..48].copy_from_slice(&self.target_rows_per_file.to_le_bytes());
        buf[48..56].copy_from_slice(&self.created_us.to_le_bytes());
        buf[56..60].copy_from_slice(&self.target_spec_id.to_le_bytes());
        let crc = header_crc(&buf);
        buf[12..16].copy_from_slice(&crc.to_le_bytes());
        buf
    }

    fn from_bytes(bytes: &[u8], ctx: &str) -> Result<Self, ZyronError> {
        if bytes.len() < CHECKPOINT_HEADER_LEN {
            return Err(corrupt(
                ctx,
                format!(
                    "clustering checkpoint is {} bytes, needs {}",
                    bytes.len(),
                    CHECKPOINT_HEADER_LEN
                ),
            ));
        }
        if bytes[0..8] != CHECKPOINT_MAGIC {
            return Err(corrupt(ctx, "clustering checkpoint magic mismatch".into()));
        }
        let version = u32::from_le_bytes([bytes[8], bytes[9], bytes[10], bytes[11]]);
        if version != CHECKPOINT_FORMAT_VERSION {
            return Err(corrupt(
                ctx,
                format!("unsupported clustering checkpoint version {}", version),
            ));
        }
        let stored = u32::from_le_bytes([bytes[12], bytes[13], bytes[14], bytes[15]]);
        if stored != header_crc(&bytes[..CHECKPOINT_HEADER_LEN]) {
            return Err(corrupt(ctx, "clustering checkpoint header CRC".into()));
        }
        let mut r = Cursor::new(&bytes[16..CHECKPOINT_HEADER_LEN], ctx);
        Ok(Self {
            pass_id: r.u64()?,
            base_version: r.u64()?,
            table_id: r.u64()?,
            target_rows_per_file: r.u64()?,
            created_us: r.i64()?,
            target_spec_id: r.u32()?,
        })
    }
}

/// CRC over everything but the CRC field itself
fn header_crc(buf: &[u8]) -> u32 {
    let mut hasher = crc32fast::Hasher::new();
    hasher.update(&buf[0..12]);
    hasher.update(&buf[16..CHECKPOINT_HEADER_LEN]);
    hasher.finalize()
}

/// One 64-byte checkpoint record
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum PassRecord {
    /// One key of the target spec, in declaration order
    Key {
        column_id: u32,
        strategy: ClusterStrategy,
        param: u32,
    },
    /// One input file the pass rewrites
    Input { partition_id: u64 },
    /// A state transition, the last one is where recovery starts
    State(PassState),
}

impl PassRecord {
    fn to_bytes(self) -> [u8; CHECKPOINT_RECORD_LEN] {
        let mut buf = [0u8; CHECKPOINT_RECORD_LEN];
        match self {
            PassRecord::Key {
                column_id,
                strategy,
                param,
            } => {
                buf[0] = 1;
                buf[4..8].copy_from_slice(&column_id.to_le_bytes());
                buf[8..16].copy_from_slice(&(strategy.to_u8() as u64).to_le_bytes());
                buf[16..24].copy_from_slice(&(param as u64).to_le_bytes());
            }
            PassRecord::Input { partition_id } => {
                buf[0] = 2;
                buf[8..16].copy_from_slice(&partition_id.to_le_bytes());
            }
            PassRecord::State(state) => {
                buf[0] = 3;
                buf[4..8].copy_from_slice(&(state.to_u8() as u32).to_le_bytes());
            }
        }
        let crc = crc32fast::hash(&buf[..CHECKPOINT_RECORD_LEN - 4]);
        buf[CHECKPOINT_RECORD_LEN - 4..].copy_from_slice(&crc.to_le_bytes());
        buf
    }

    /// Decodes one record, or None when the bytes are a torn tail. A torn
    /// record is the expected outcome of a crash mid-append, not an error
    fn from_bytes(bytes: &[u8]) -> Option<Self> {
        if bytes.len() != CHECKPOINT_RECORD_LEN {
            return None;
        }
        let stored = u32::from_le_bytes([
            bytes[CHECKPOINT_RECORD_LEN - 4],
            bytes[CHECKPOINT_RECORD_LEN - 3],
            bytes[CHECKPOINT_RECORD_LEN - 2],
            bytes[CHECKPOINT_RECORD_LEN - 1],
        ]);
        if stored != crc32fast::hash(&bytes[..CHECKPOINT_RECORD_LEN - 4]) {
            return None;
        }
        let a = u32::from_le_bytes([bytes[4], bytes[5], bytes[6], bytes[7]]);
        let mut b_bytes = [0u8; 8];
        b_bytes.copy_from_slice(&bytes[8..16]);
        let b = u64::from_le_bytes(b_bytes);
        let mut c_bytes = [0u8; 8];
        c_bytes.copy_from_slice(&bytes[16..24]);
        let c = u64::from_le_bytes(c_bytes);
        Some(match bytes[0] {
            1 => PassRecord::Key {
                column_id: a,
                strategy: ClusterStrategy::from_u8(b as u8)?,
                param: c as u32,
            },
            2 => PassRecord::Input { partition_id: b },
            3 => PassRecord::State(PassState::from_u8(a as u8)?),
            _ => return None,
        })
    }
}

/// Append handle on one pass checkpoint
struct PassCheckpoint {
    path: PathBuf,
    file: File,
}

impl PassCheckpoint {
    fn create(path: &Path, header: &PassHeader) -> Result<Self, ZyronError> {
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)?;
        }
        let mut file = OpenOptions::new().write(true).create_new(true).open(path)?;
        file.write_all(&header.to_bytes())?;
        file.sync_all()?;
        Ok(Self {
            path: path.to_path_buf(),
            file,
        })
    }

    fn open_append(path: &Path) -> Result<Self, ZyronError> {
        let file = OpenOptions::new().append(true).open(path)?;
        Ok(Self {
            path: path.to_path_buf(),
            file,
        })
    }

    /// Appends one record and makes it durable. Every caller depends on
    /// the record being on disk before the action it announces starts
    fn append(&mut self, record: PassRecord) -> Result<(), ZyronError> {
        self.file.write_all(&record.to_bytes())?;
        self.file.sync_all()?;
        Ok(())
    }

    fn discard(self) -> Result<(), ZyronError> {
        drop(self.file);
        match fs::remove_file(&self.path) {
            Ok(()) => Ok(()),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(()),
            Err(e) => Err(e.into()),
        }
    }
}

/// What one checkpoint file says, records past a torn tail dropped
struct CheckpointContents {
    header: PassHeader,
    target: ClusterSpec,
    inputs: Vec<u64>,
    state: PassState,
}

fn read_checkpoint(path: &Path) -> Result<CheckpointContents, ZyronError> {
    let bytes = fs::read(path)?;
    let ctx = path.display().to_string();
    let header = PassHeader::from_bytes(&bytes, &ctx)?;
    let mut keys = Vec::new();
    let mut inputs = Vec::new();
    let mut state = PassState::Running;
    let mut offset = CHECKPOINT_HEADER_LEN;
    while offset + CHECKPOINT_RECORD_LEN <= bytes.len() {
        let Some(record) = PassRecord::from_bytes(&bytes[offset..offset + CHECKPOINT_RECORD_LEN])
        else {
            break;
        };
        match record {
            PassRecord::Key {
                column_id,
                strategy,
                param,
            } => keys.push(crate::manifest::ClusterKey {
                column_id,
                strategy,
                param,
            }),
            PassRecord::Input { partition_id } => inputs.push(partition_id),
            PassRecord::State(s) => state = s,
        }
        offset += CHECKPOINT_RECORD_LEN;
    }
    Ok(CheckpointContents {
        header,
        target: ClusterSpec {
            spec_id: header.target_spec_id,
            keys,
        },
        inputs,
        state,
    })
}

/// Appends one finalized output's manifest entry to the staging sidecar.
/// Length prefix, payload, payload CRC, fsynced, so the entry is durable
/// before the pass moves on to the next output
fn append_staged_entry(staging: &Path, entry: &PartitionEntry) -> Result<(), ZyronError> {
    let mut payload = Vec::new();
    encode_partition_entry(entry, &mut payload);
    let mut record = Vec::with_capacity(payload.len() + 8);
    record.extend_from_slice(&(payload.len() as u32).to_le_bytes());
    record.extend_from_slice(&payload);
    record.extend_from_slice(&crc32fast::hash(&payload).to_le_bytes());
    let mut file = OpenOptions::new()
        .append(true)
        .create(true)
        .open(staging.join(STAGED_ENTRIES_NAME))?;
    file.write_all(&record)?;
    file.sync_all()?;
    Ok(())
}

/// Reads back every fully written staged entry. A torn trailing record
/// ends the read, so the outputs it describes are the ones that exist
fn read_staged_entries(staging: &Path) -> Result<Vec<PartitionEntry>, ZyronError> {
    let path = staging.join(STAGED_ENTRIES_NAME);
    let mut bytes = Vec::new();
    match File::open(&path) {
        Ok(mut f) => {
            f.read_to_end(&mut bytes)?;
        }
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(Vec::new()),
        Err(e) => return Err(e.into()),
    }
    let ctx = path.display().to_string();
    let mut entries = Vec::new();
    let mut offset = 0usize;
    while offset + 8 <= bytes.len() {
        let len = u32::from_le_bytes([
            bytes[offset],
            bytes[offset + 1],
            bytes[offset + 2],
            bytes[offset + 3],
        ]) as usize;
        let payload_start = offset + 4;
        let crc_start = payload_start + len;
        if crc_start + 4 > bytes.len() {
            break;
        }
        let payload = &bytes[payload_start..crc_start];
        let stored = u32::from_le_bytes([
            bytes[crc_start],
            bytes[crc_start + 1],
            bytes[crc_start + 2],
            bytes[crc_start + 3],
        ]);
        if stored != crc32fast::hash(payload) {
            break;
        }
        let mut r = Cursor::new(payload, &ctx);
        entries.push(decode_partition_entry(&mut r)?);
        offset = crc_start + 4;
    }
    Ok(entries)
}

/// What a pass is allowed to do and what it must respect
#[derive(Debug, Clone)]
pub struct ClusterPassOptions {
    /// Names the staging directory and the checkpoint, unique per pass
    pub pass_id: u64,
    /// Rows per output file, how the global order is chunked
    pub target_rows_per_file: u64,
    /// Most input files one pass rewrites
    pub max_inputs: usize,
    /// Key prefix the operator pinned. A candidate that does not lead
    /// with it is refused before anything is scored
    pub anchors: Vec<u32>,
    pub gate: GateConfig,
    /// Whether the gate's judgement is binding.
    ///
    /// It is, under Auto and Hybrid, which is what guarantees measurement
    /// can only improve a layout or leave it alone. Under Force the
    /// operator has already decided, on knowledge measurement does not
    /// have, so the pass applies what was declared and the gate's verdict
    /// is reported rather than enforced. An anchor conflict stays fatal
    /// either way: that is a bug, not a preference
    pub gated: bool,
}

impl ClusterPassOptions {
    pub fn new(pass_id: u64) -> Self {
        Self {
            pass_id,
            target_rows_per_file: DEFAULT_ROWS_PER_FILE,
            max_inputs: DEFAULT_MAX_INPUTS,
            anchors: Vec::new(),
            gate: GateConfig::default(),
            gated: true,
        }
    }
}

/// What a pass did. `inputs` of zero means every file already carries the
/// target spec and there was nothing to re-cluster
#[derive(Debug, Clone, PartialEq)]
pub struct ClusterPassOutcome {
    pub pass_id: u64,
    pub decision: Decision,
    /// Version the accepted pass committed, None when it was refused
    pub version: Option<u64>,
    pub inputs: usize,
    pub outputs: usize,
    pub rows_written: u64,
    pub bytes_read: u64,
    pub bytes_written: u64,
    /// True when the pass was picked up from a checkpoint rather than
    /// started fresh
    pub resumed: bool,
    /// True when the global order came from merging inputs already
    /// ordered by the target curve rather than from a comparison sort
    pub merged: bool,
}

impl ClusterPassOutcome {
    fn idle(pass_id: u64, gate: &GateConfig) -> Self {
        Self {
            pass_id,
            decision: Decision::BelowThreshold {
                delta: 0.0,
                required: gate.min_improvement,
            },
            version: None,
            inputs: 0,
            outputs: 0,
            rows_written: 0,
            bytes_read: 0,
            bytes_written: 0,
            resumed: false,
            merged: false,
        }
    }
}

/// Everything a pass needs that does not change between its stages
struct PassContext<'a> {
    log: &'a TransactionLog,
    table_id: u64,
    base: &'a ManifestFile,
    target: &'a ClusterSpec,
    inputs: &'a [u64],
    options: &'a ClusterPassOptions,
    staging: PathBuf,
    resumed: bool,
}

/// Files the pass rewrites.
///
/// Two things make a file wrong for the target layout. A file under
/// another spec was laid out by a different rule, so it is a candidate
/// unconditionally. A file already under the target spec is wrong only
/// when its leading key range overlaps another such file's, because a
/// clustered set is exactly one whose files do not overlap. That second
/// case is the drift new appends introduce, and it is what the merge
/// path exists for: every input is already ordered by the target curve,
/// so the global order costs a merge rather than a sort.
///
/// Smallest first, so a pass bounded by `max_inputs` takes the files
/// whose rewrite costs least and coalesces the most.
fn select_inputs(base: &ManifestFile, target: &ClusterSpec, max_inputs: usize) -> Vec<u64> {
    let mut candidates = repair_candidates(base, target);
    candidates.sort_unstable();
    candidates.truncate(max_inputs);
    let mut inputs: Vec<u64> = candidates.into_iter().map(|(_, id)| id).collect();
    // Commit order and replan determinism both want partition order
    inputs.sort_unstable();
    inputs
}

/// Every file a repair pass would rewrite if nothing bounded it, as
/// (size, partition id) pairs.
///
/// Two kinds of file are in here. One written under an older spec is drift
/// by definition: it is ordered by a layout the table has moved on from.
/// One written under the current spec whose leading key range overlaps
/// another's is drift too, because overlapping ranges are exactly what
/// stops a predicate reaching one file.
///
/// The urgency check and the pass share this, so the threshold an operator
/// sets means the same thing as the work the pass will do. Two definitions
/// of "needs repair" would let a table trip a fast lane that then found
/// nothing to rewrite
fn repair_candidates(base: &ManifestFile, target: &ClusterSpec) -> Vec<(u64, u64)> {
    let mut candidates: Vec<(u64, u64)> = Vec::new();
    let mut aligned: Vec<&PartitionEntry> = Vec::new();
    for entry in &base.entries {
        if entry.cluster_spec_id == target.spec_id {
            aligned.push(entry);
        } else {
            candidates.push((entry.size_bytes, entry.partition_id));
        }
    }
    if let Some(lead) = target.keys.first() {
        let drifted = overlapping_files(&aligned, lead.column_id);
        for entry in &aligned {
            if drifted.contains(&entry.partition_id) {
                candidates.push((entry.size_bytes, entry.partition_id));
            }
        }
    }
    candidates
}

/// How many files a repair pass has outstanding on this table.
///
/// Read from the manifest with no IO, so a maintenance tick can ask it for
/// every table before deciding which ones are worth a pass. Zero when the
/// table declares no layout, because a table with no keys has nothing to
/// have drifted from
pub fn drifted_file_count(manifest: &ManifestFile) -> usize {
    if manifest.cluster_spec.keys.is_empty() {
        return 0;
    }
    repair_candidates(manifest, &manifest.cluster_spec).len()
}

/// Partition ids whose leading key range overlaps another entry's, by an
/// interval sweep over the manifest with no IO.
///
/// A file with no bounds on the key column is ignored rather than
/// treated as overlapping everything. Absent bounds mean the key does
/// not order that file's rows at all, usually because they are all NULL,
/// so it is not evidence of drift and rewriting it would not move it
fn overlapping_files(entries: &[&PartitionEntry], column_id: u32) -> BTreeSet<u64> {
    let mut hits = BTreeSet::new();
    if entries.len() < 2 {
        return hits;
    }
    let mut ranged: Vec<(&LakeValue, &LakeValue, u64)> = Vec::with_capacity(entries.len());
    for entry in entries {
        let Some(stats) = entry.stats_for(column_id) else {
            continue;
        };
        if let (Some(min), Some(max)) = (stats.bounds.min.as_ref(), stats.bounds.max.as_ref()) {
            ranged.push((min, max, entry.partition_id));
        }
    }
    ranged.sort_by(|a, b| a.0.compare(b.0).unwrap_or(Ordering::Equal));
    // Sweep in min order carrying the highest max seen. An entry whose
    // min is not above that max shares values with the entry that set it
    let mut high: Option<(&LakeValue, u64)> = None;
    for (min, max, partition_id) in &ranged {
        if let Some((high_max, high_id)) = high {
            if min.compare(high_max) != Some(Ordering::Greater) {
                hits.insert(*partition_id);
                hits.insert(high_id);
            }
        }
        let replace = high
            .map(|(high_max, _)| max.compare(high_max) == Some(Ordering::Greater))
            .unwrap_or(true);
        if replace {
            high = Some((max, *partition_id));
        }
    }
    hits
}

/// Runs one clustering pass against the newest published version.
///
/// The pass is refused unless the feedback gate finds it a net
/// improvement on every recorded predicate class, so a caller can run
/// this speculatively: the worst outcome is wasted work, never a worse
/// layout.
pub fn run_cluster_pass(
    log: &TransactionLog,
    attempt: CommitAttempt<'_>,
    table_id: u64,
    target: &ClusterSpec,
    classes: &[PredicateClass],
    options: &ClusterPassOptions,
) -> Result<ClusterPassOutcome, ZyronError> {
    let base = log.latest_manifest()?;
    let inputs = select_inputs(&base, target, options.max_inputs);
    if inputs.is_empty() {
        return Ok(ClusterPassOutcome::idle(options.pass_id, &options.gate));
    }

    let checkpoint_path = log.paths().clustering_pass(options.pass_id);
    if checkpoint_path.exists() {
        return Err(ZyronError::ClusteringRejected(format!(
            "clustering pass {} is already in flight",
            options.pass_id
        )));
    }
    let staging = log.paths().staging_dir(options.pass_id);
    // A staging directory with no checkpoint is debris from a crash
    // before the pass was announced, and holds nothing anyone references
    if staging.exists() {
        fs::remove_dir_all(&staging)?;
    }
    fs::create_dir_all(&staging)?;

    let header = PassHeader {
        pass_id: options.pass_id,
        base_version: base.snapshot_id,
        table_id,
        target_rows_per_file: options.target_rows_per_file.max(1),
        created_us: attempt.timestamp_us,
        target_spec_id: target.spec_id,
    };
    let mut checkpoint = PassCheckpoint::create(&checkpoint_path, &header)?;
    for key in &target.keys {
        checkpoint.append(PassRecord::Key {
            column_id: key.column_id,
            strategy: key.strategy,
            param: key.param,
        })?;
    }
    for partition_id in &inputs {
        checkpoint.append(PassRecord::Input {
            partition_id: *partition_id,
        })?;
    }
    checkpoint.append(PassRecord::State(PassState::Running))?;

    let ctx = PassContext {
        log,
        table_id,
        base: &base,
        target,
        inputs: &inputs,
        options,
        staging,
        resumed: false,
    };
    drive_pass(&ctx, attempt, classes, checkpoint, Vec::new())
}

/// Most keys a proposal may carry. Past four the tail keys refine rows a
/// file already holds together and stop buying pruning
pub const DEFAULT_MAX_PROPOSED_KEYS: usize = 4;

/// What a table-level pass is allowed to spend and how far it may reach
#[derive(Debug, Clone)]
pub struct TablePassOptions {
    /// Names the staging directory and the checkpoint, unique per pass
    pub pass_id: u64,
    pub target_rows_per_file: u64,
    pub max_inputs: usize,
    pub gate: GateConfig,
    pub max_proposed_keys: usize,
}

impl TablePassOptions {
    pub fn new(pass_id: u64) -> Self {
        Self {
            pass_id,
            target_rows_per_file: DEFAULT_ROWS_PER_FILE,
            max_inputs: DEFAULT_MAX_INPUTS,
            gate: GateConfig::default(),
            max_proposed_keys: DEFAULT_MAX_PROPOSED_KEYS,
        }
    }
}

/// What one table-level pass decided, and the evidence behind it.
///
/// The measurements come back with the outcome because the caller reports
/// them and recomputing them would read the observer a second time at a
/// different epoch, which would make the metric disagree with the decision
/// it is supposed to describe
#[derive(Debug, Clone)]
pub struct TablePassReport {
    /// None when nothing was declared and nothing measured is worth
    /// ordering by, so no pass was started and no files were read
    pub outcome: Option<ClusterPassOutcome>,
    pub mode: ClusterMode,
    /// Columns the workload window carried, what the proposal was made from
    pub evidence_columns: usize,
    /// Mean measured skip rate over those columns, None until some scan has
    /// reported one
    pub measured_skip_rate: Option<f64>,
    /// True when measurement wants a layout the table is not running yet:
    /// a spec was proposed and it did not reach the files, either because
    /// no pass ran or because the gate refused the one that did. One
    /// proposal is outstanding per table at a time, so this is the count
    pub proposal_pending: bool,
    /// Keys the table is laid out by once this pass is done: the target if
    /// the pass committed it, the spec already in force otherwise.
    ///
    /// The caller mirrors these into the catalog, because planning reads
    /// the catalog and a plan judged against a declared key a pass replaced
    /// would claim a layout the files do not have
    pub active_keys: Vec<ClusterKey>,
}

/// Chooses a target layout for one table and runs a pass against it.
///
/// This is the whole decision in one place: what the mode allows, what the
/// workload window holds, which predicate classes the gate replays, and the
/// pass itself. The background worker and `OPTIMIZE ... CLUSTER` both enter
/// here, so an operator-driven pass and a scheduled one make the same
/// choices from the same evidence rather than drifting apart.
///
/// Whether a pass may start at all is the caller's decision. The clustering
/// schedule governs the background worker, and it does not govern OPTIMIZE,
/// which is the operator asking for a pass directly
pub fn run_table_cluster_pass(
    log: &TransactionLog,
    attempt: CommitAttempt<'_>,
    table_id: u32,
    options: &TablePassOptions,
) -> Result<TablePassReport, ZyronError> {
    let manifest = log.latest_manifest()?;
    let mode = manifest.clustering_mode();
    let now = crate::workload::current_epoch();
    let observer = crate::workload::observer();
    let evidence = crate::planner::evidence_from_manifest(&manifest, observer, table_id, now);
    let anchors = manifest.clustering_anchors();

    let measured: Vec<f64> = evidence
        .iter()
        .filter_map(|c| crate::planner::measured_skip_rate(observer, table_id, c.column_id, now))
        .collect();
    let mut report = TablePassReport {
        outcome: None,
        mode,
        evidence_columns: evidence.len(),
        measured_skip_rate: if measured.is_empty() {
            None
        } else {
            Some(measured.iter().sum::<f64>() / measured.len() as f64)
        },
        proposal_pending: false,
        active_keys: manifest.cluster_spec.keys.clone(),
    };

    // Under Force the target is what the operator declared. Under Auto and
    // Hybrid measurement proposes, and a proposal identical to the current
    // keys is not a new spec, it is the same layout still being applied to
    // files that have not reached it yet
    let target = if mode == ClusterMode::Force {
        manifest.cluster_spec.clone()
    } else {
        let proposal = crate::planner::propose(&evidence, &anchors, options.max_proposed_keys);
        if proposal == manifest.cluster_spec.keys {
            manifest.cluster_spec.clone()
        } else {
            ClusterSpec {
                spec_id: manifest.cluster_spec.spec_id.saturating_add(1),
                keys: proposal,
            }
        }
    };
    // A target carrying a spec id the manifest does not is measurement
    // asking for a layout the table is not running yet. It stops being
    // outstanding only when a pass commits it
    let proposed_new_spec = target.spec_id != manifest.cluster_spec.spec_id;
    report.proposal_pending = proposed_new_spec;
    if target.keys.is_empty() {
        // Nothing declared and nothing measured worth ordering by
        return Ok(report);
    }

    let classes = crate::planner::predicate_classes(&manifest, &evidence, observer, table_id, now);
    let pass_options = ClusterPassOptions {
        pass_id: options.pass_id,
        target_rows_per_file: options.target_rows_per_file,
        // The table's own bound when it names one. A table taking constant
        // small writes wants a pass that keeps up; one that is mostly read
        // wants a pass that never competes with queries for long, and the
        // node default cannot be right for both
        max_inputs: manifest.cluster_repair_max_inputs(options.max_inputs),
        anchors,
        gate: options.gate,
        gated: mode != ClusterMode::Force,
    };
    let outcome = run_cluster_pass(
        log,
        attempt,
        table_id as u64,
        &target,
        &classes,
        &pass_options,
    )?;
    // A committed pass is the proposal landing, so nothing is outstanding
    report.proposal_pending = proposed_new_spec && outcome.version.is_none();
    // The target became the layout only if the pass committed it. A refused
    // pass leaves the files ordered the way they already were
    if outcome.version.is_some() {
        report.active_keys = target.keys.clone();
    }
    report.outcome = Some(outcome);
    Ok(report)
}

/// Carries a pass from wherever it is to a committed version or a clean
/// refusal. `staged` holds the outputs an earlier attempt already wrote
fn drive_pass(
    ctx: &PassContext<'_>,
    attempt: CommitAttempt<'_>,
    classes: &[PredicateClass],
    mut checkpoint: PassCheckpoint,
    staged: Vec<PartitionEntry>,
) -> Result<ClusterPassOutcome, ZyronError> {
    let staged = match stage_outputs(ctx, staged) {
        Ok(v) => v,
        Err(e) => {
            // Nothing in data/ was touched, so the only cleanup is the
            // speculative work the pass itself created
            let _ = fs::remove_dir_all(&ctx.staging);
            let _ = checkpoint.discard();
            return Err(e);
        }
    };
    checkpoint.append(PassRecord::State(PassState::Staged))?;
    finish_pass(ctx, attempt, classes, checkpoint, staged)
}

/// Everything staging produced, carried to the gate
struct StagedPass {
    entries: Vec<PartitionEntry>,
    bytes_read: u64,
    /// The global order came from a merge rather than a sort
    merged: bool,
}

/// Rows of one input the pass carries forward
struct InputRows {
    /// Source row index of every surviving row
    rows: Vec<u32>,
    /// Indices into `rows` whose key has no NULL component, in file order
    ranked: Vec<u32>,
    /// Ordering key per entry of `ranked`, `key_len` bytes each
    keys: Vec<u8>,
    /// Indices into `rows` with a NULL key component. They have no
    /// position on the curve, so they go to the tail exactly as the
    /// writer places them inside a file
    nulls: Vec<u32>,
}

/// Writes every output the plan calls for that is not already staged.
/// Returns the full staged set and the bytes the pass read
fn stage_outputs(
    ctx: &PassContext<'_>,
    mut staged: Vec<PartitionEntry>,
) -> Result<StagedPass, ZyronError> {
    let schema = &ctx.base.schema;
    let sort_keys: Vec<u32> = ctx.target.keys.iter().map(|k| k.column_id).collect();
    let sort_strategies: Vec<ClusterStrategy> =
        ctx.target.keys.iter().map(|k| k.strategy).collect();
    let bloom_columns = ctx.base.bloom_columns();
    let curve = sort_strategies
        .first()
        .copied()
        .unwrap_or(ClusterStrategy::RangePartition);
    let key_len = sort_keys.len() * 8;

    // Every input decoded once. This is what bounds a pass to max_inputs
    let mut decoded: Vec<Vec<DecodedColumn>> = Vec::with_capacity(ctx.inputs.len());
    let mut per_input: Vec<InputRows> = Vec::with_capacity(ctx.inputs.len());
    let mut bytes_read = 0u64;
    for partition_id in ctx.inputs {
        let entry = ctx.base.entry_for(*partition_id).ok_or_else(|| {
            ZyronError::ClusteringRejected(format!(
                "clustering input partition {:#x} is not in the base manifest",
                partition_id
            ))
        })?;
        let reader = LakeFileReader::open(ctx.log.paths(), *partition_id)?;
        let keep = reader.delete_survivors(schema, ctx.base, entry)?;
        let columns: Vec<DecodedColumn> = schema
            .columns
            .iter()
            .map(|c| reader.read_column(c))
            .collect::<Result<_, _>>()?;
        bytes_read += entry.size_bytes;

        // Key columns in declaration order, matched to the decoded set
        let mut key_columns = Vec::with_capacity(sort_keys.len());
        for key_id in &sort_keys {
            let col = schema.column_by_id(*key_id).ok_or_else(|| {
                ZyronError::ClusteringRejected(format!(
                    "cluster key column {} is not in the schema",
                    key_id
                ))
            })?;
            let index = schema
                .columns
                .iter()
                .position(|c| c.id == *key_id)
                .ok_or_else(|| {
                    ZyronError::ClusteringRejected(format!(
                        "cluster key column {} has no decoded data",
                        key_id
                    ))
                })?;
            key_columns.push((col.physical_type_id(), index));
        }

        let mut rows = Vec::new();
        let mut ranked = Vec::new();
        let mut keys = Vec::new();
        let mut nulls = Vec::new();
        let mut axes = Vec::with_capacity(key_columns.len());
        for row in 0..reader.row_count() {
            if keep[row / 8] & (1 << (row % 8)) == 0 {
                continue;
            }
            let local = rows.len() as u32;
            rows.push(row as u32);
            if key_columns.is_empty() {
                ranked.push(local);
                continue;
            }
            axes.clear();
            let mut any_null = false;
            for (physical, index) in &key_columns {
                match columns[*index].cell(row) {
                    Some(cell) => axes.push(normalize_component(*physical, cell)),
                    None => {
                        any_null = true;
                        axes.push(0);
                    }
                }
            }
            if any_null {
                nulls.push(local);
            } else {
                ranked.push(local);
                keys.extend_from_slice(&ordering_key(curve, &axes));
            }
        }
        decoded.push(columns);
        per_input.push(InputRows {
            rows,
            ranked,
            keys,
            nulls,
        });
    }

    // Inputs already under the target spec are each ordered by its curve,
    // so a merge produces the global order without a comparison sort.
    // A spec change leaves nothing to exploit and has to sort
    let mergeable = key_len > 0
        && ctx.inputs.iter().all(|partition_id| {
            ctx.base
                .entry_for(*partition_id)
                .map(|e| e.cluster_spec_id == ctx.target.spec_id)
                .unwrap_or(false)
        });
    let mut order: Vec<(u32, u32)> = if mergeable {
        merge_order(&per_input, key_len)?
    } else {
        sort_order(&per_input, key_len)
    };
    // Null keys have no place on any curve, so they trail every ranked
    // row, which is where the writer puts them inside a file too
    for (file, input) in per_input.iter().enumerate() {
        for local in &input.nulls {
            order.push((file as u32, *local));
        }
    }

    let target_rows = ctx.options.target_rows_per_file.max(1) as usize;
    // Resume skips exactly the rows the recorded outputs already hold
    let mut position: usize = staged.iter().map(|e| e.row_count as usize).sum();
    if position > order.len() {
        return Err(ZyronError::ClusteringRejected(format!(
            "clustering pass {} staged {} rows, the replanned order holds {}",
            ctx.options.pass_id,
            position,
            order.len()
        )));
    }
    let mut used: BTreeSet<u64> = staged.iter().map(|e| e.partition_id).collect();
    while position < order.len() {
        let end = (position + target_rows).min(order.len());
        let chunk = &order[position..end];
        let mut batch: Vec<ColumnData> = schema
            .columns
            .iter()
            .map(|c| ColumnData {
                column_id: c.id,
                cells: Vec::with_capacity(chunk.len()),
            })
            .collect();
        for (file, local) in chunk {
            let row = per_input[*file as usize].rows[*local as usize] as usize;
            for (slot, column) in batch.iter_mut().zip(decoded[*file as usize].iter()) {
                slot.cells.push(column.cell(row).map(|c| c.to_vec()));
            }
        }
        let partition_id = loop {
            let candidate = allocate_partition_id(ctx.base);
            if !used.contains(&candidate) {
                break candidate;
            }
        };
        used.insert(partition_id);
        let entry = write_data_file_at(
            &ctx.staging,
            schema,
            &WriteRequest {
                partition_id,
                columns: &batch,
                sort_keys: &sort_keys,
                sort_strategies: &sort_strategies,
                cluster_spec_id: ctx.target.spec_id,
                table_id: ctx.table_id,
                bloom_columns: &bloom_columns,
                index_id: None,
            },
        )?
        .entry;
        append_staged_entry(&ctx.staging, &entry)?;
        staged.push(entry);
        position = end;
    }
    Ok(StagedPass {
        entries: staged,
        bytes_read,
        merged: mergeable,
    })
}

/// Global order by k-way merge over the leading eight bytes of each
/// row's ordering key. Every input is already sorted by that key, so the
/// merge is exact on the prefix and ties fall into adjacent output files,
/// which is the only thing the global order decides
fn merge_order(per_input: &[InputRows], key_len: usize) -> Result<Vec<(u32, u32)>, ZyronError> {
    let prefix = key_len.min(8);
    let mut columns = Vec::with_capacity(per_input.len());
    let mut row_counts = Vec::with_capacity(per_input.len());
    for input in per_input {
        let count = input.ranked.len();
        let mut column = Vec::with_capacity(count * 8);
        for index in 0..count {
            let start = index * key_len;
            let mut buf = [0u8; 8];
            buf[..prefix].copy_from_slice(&input.keys[start..start + prefix]);
            // Ordering keys are big endian so byte order is value order.
            // The merge reads little endian, so the value round trips
            // through the encoding it expects
            column.extend_from_slice(&u64::from_be_bytes(buf).to_le_bytes());
        }
        columns.push(column);
        row_counts.push(count);
    }
    let total: usize = row_counts.iter().sum();
    let mut iter = MergeScanIterator::new(columns, 8, row_counts)?;
    let mut order = Vec::with_capacity(total);
    while let Some((file, index)) = iter.next() {
        order.push((file as u32, per_input[file].ranked[index]));
    }
    Ok(order)
}

/// Global order by sorting every row on its full ordering key. Stable, so
/// rows sharing a key keep input order and a replan is deterministic
fn sort_order(per_input: &[InputRows], key_len: usize) -> Vec<(u32, u32)> {
    let total: usize = per_input.iter().map(|i| i.ranked.len()).sum();
    let mut positions: Vec<(u32, u32)> = Vec::with_capacity(total);
    for (file, input) in per_input.iter().enumerate() {
        for index in 0..input.ranked.len() {
            positions.push((file as u32, index as u32));
        }
    }
    if key_len > 0 {
        positions.sort_by(|a, b| {
            let left = key_slice(per_input, *a, key_len);
            let right = key_slice(per_input, *b, key_len);
            left.cmp(right)
        });
    }
    positions
        .into_iter()
        .map(|(file, index)| (file, per_input[file as usize].ranked[index as usize]))
        .collect()
}

#[inline]
fn key_slice(per_input: &[InputRows], position: (u32, u32), key_len: usize) -> &[u8] {
    let start = position.1 as usize * key_len;
    &per_input[position.0 as usize].keys[start..start + key_len]
}

/// Scores the staged set against the inputs and either commits it or
/// unlinks it
fn finish_pass(
    ctx: &PassContext<'_>,
    attempt: CommitAttempt<'_>,
    classes: &[PredicateClass],
    mut checkpoint: PassCheckpoint,
    staged: StagedPass,
) -> Result<ClusterPassOutcome, ZyronError> {
    let StagedPass {
        entries: staged,
        bytes_read,
        merged,
    } = staged;
    let current: Vec<PartitionEntry> = ctx
        .inputs
        .iter()
        .filter_map(|id| ctx.base.entry_for(*id).cloned())
        .collect();
    let candidate_keys: Vec<u32> = ctx.target.keys.iter().map(|k| k.column_id).collect();
    let decision = evaluate(
        &current,
        &staged,
        &ctx.base.schema,
        classes,
        &ctx.options.anchors,
        &candidate_keys,
        ctx.options.gate,
    );
    let rows_written: u64 = staged.iter().map(|e| e.row_count).sum();
    let bytes_written: u64 = staged.iter().map(|e| e.size_bytes).sum();
    let mut outcome = ClusterPassOutcome {
        pass_id: ctx.options.pass_id,
        decision: decision.clone(),
        version: None,
        inputs: ctx.inputs.len(),
        outputs: staged.len(),
        rows_written,
        bytes_read,
        bytes_written,
        resumed: ctx.resumed,
        merged,
    };
    let accepted = match &decision {
        Decision::Accept { .. } => true,
        Decision::AnchorConflict { .. } => false,
        _ => !ctx.options.gated,
    };
    if !accepted {
        checkpoint.append(PassRecord::State(PassState::Rejected))?;
        remove_staging(&ctx.staging)?;
        checkpoint.discard()?;
        return Ok(outcome);
    }
    checkpoint.append(PassRecord::State(PassState::Committing))?;
    let version = commit_pass(ctx, attempt, &staged)?;
    checkpoint.append(PassRecord::State(PassState::Done))?;
    remove_staging(&ctx.staging)?;
    checkpoint.discard()?;
    outcome.version = Some(version);
    Ok(outcome)
}

/// Unlinks a pass staging directory, tolerating one already gone so a
/// crash between the unlink and the checkpoint delete is not an error
fn remove_staging(staging: &Path) -> Result<(), ZyronError> {
    match fs::remove_dir_all(staging) {
        Ok(()) => Ok(()),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(()),
        Err(e) => Err(e.into()),
    }
}

/// Renames the accepted candidates into `data/` and commits one version
/// removing the inputs and adding them.
///
/// A crash between the renames and the commit leaves data files no
/// version references, which is precisely what vacuum reclaims, so the
/// window costs disk rather than correctness
fn commit_pass(
    ctx: &PassContext<'_>,
    attempt: CommitAttempt<'_>,
    staged: &[PartitionEntry],
) -> Result<u64, ZyronError> {
    fs::create_dir_all(ctx.log.paths().data_dir())?;
    for entry in staged {
        let from = ctx.staging.join(data_file_name(entry.partition_id));
        let to = ctx.log.paths().data_file(entry.partition_id);
        if from.exists() {
            fs::rename(&from, &to)?;
        } else if !to.exists() {
            return Err(ZyronError::ClusteringRejected(format!(
                "clustering pass {} lost staged output {:#x}",
                ctx.options.pass_id, entry.partition_id
            )));
        }
    }

    // What the outputs were built from. A concurrent commit that changed
    // any of it invalidates rows already written, so the pass loses
    // rather than committing a rewrite of state that moved
    let planned: BTreeMap<u64, Vec<u64>> = ctx
        .inputs
        .iter()
        .filter_map(|id| {
            ctx.base
                .entry_for(*id)
                .map(|e| (*id, e.delete_predicate_ids.clone()))
        })
        .collect();
    let pass_id = ctx.options.pass_id;
    let target = ctx.target.clone();
    let paths = ctx.log.paths().clone();
    let table_id = ctx.table_id;
    // Index files this attempt wrote. A retry builds against a different
    // base and writes fresh ones, so the losing attempt's files are
    // unlinked rather than left for vacuum to find
    let written_index_files: Rc<RefCell<Vec<PathBuf>>> = Rc::new(RefCell::new(Vec::new()));
    let staged_index = Rc::clone(&written_index_files);
    let mut attempt = attempt;
    attempt.operation = OperationKind::Optimize;
    let result = ctx.log.commit(attempt, move |base| {
        for path in staged_index.borrow_mut().drain(..) {
            let _ = fs::remove_file(path);
        }
        for (partition_id, predicates) in &planned {
            match base.entry_for(*partition_id) {
                Some(entry) if entry.delete_predicate_ids == *predicates => {}
                _ => {
                    return Err(ZyronError::ClusteringRejected(format!(
                        "clustering pass {} planned against partition {:#x} that another commit changed",
                        pass_id, partition_id
                    )))
                }
            }
        }
        let mut entries: Vec<LogEntry> = planned
            .keys()
            .map(|partition_id| LogEntry::RemoveFile {
                partition_id: *partition_id,
            })
            .collect();
        if target.spec_id > base.cluster_spec.spec_id {
            entries.push(LogEntry::SetClusterSpec(target.clone()));
        }
        // Outputs carry no delete predicate: the rewrite already applied
        // every predicate that attached to their inputs
        for entry in staged {
            entries.push(LogEntry::AddFile(entry.clone()));
        }
        // A predicate the inputs were the last carriers of has nothing
        // left to remove, the rewrite applied it for good
        let mut retire: BTreeSet<u64> = planned.values().flatten().copied().collect();
        for file in &base.entries {
            if !planned.contains_key(&file.partition_id) {
                for id in &file.delete_predicate_ids {
                    retire.remove(id);
                }
            }
        }
        for id in retire {
            entries.push(LogEntry::RemoveDeletePredicate { id });
        }

        // The pass moved every input row into a new file, so index entries
        // addressing an input are stale. Their files are dropped and the
        // outputs are indexed in this same commit, which keeps coverage
        // complete and stops the index declining after a clustering run
        if !base.indexes.is_empty() {
            let removed: Vec<u64> = planned.keys().copied().collect();
            let (drops, orphaned) = index::stale_index_files(base, &removed);
            entries.extend(drops);
            let mut used: Vec<u64> = staged.iter().map(|e| e.partition_id).collect();
            for spec in &base.indexes {
                let mut batch = index::IndexBatch::new(spec);
                // Outputs carry no delete predicate, so every row is live
                for entry in staged {
                    index::entries_for_file(&paths, base, spec, entry, &mut batch)?;
                }
                for orphan in &orphaned {
                    let Some(entry) = base.entry_for(*orphan) else {
                        continue;
                    };
                    index::entries_for_file(&paths, base, spec, entry, &mut batch)?;
                }
                for file in index::write_index_files(
                    &paths,
                    &base.schema,
                    spec,
                    table_id,
                    batch,
                    &mut || {
                        let id = allocate_unused_partition_id(base, &used);
                        used.push(id);
                        id
                    },
                )? {
                    staged_index
                        .borrow_mut()
                        .push(paths.index_file(spec.index_id, file.file.partition_id));
                    entries.push(LogEntry::AddIndexFile(file));
                }
            }
        }
        Ok(entries)
    });
    if result.is_err() {
        for path in written_index_files.borrow_mut().drain(..) {
            let _ = fs::remove_file(path);
        }
    }
    result
}

/// What a resumed pass is judged and pinned by. Everything else comes
/// from the checkpoint, because a resume must replan exactly what the
/// crashed pass planned
#[derive(Debug, Clone)]
pub struct ResumeOptions {
    pub anchors: Vec<u32>,
    pub gate: GateConfig,
    /// See `ClusterPassOptions::gated`
    pub gated: bool,
}

impl Default for ResumeOptions {
    fn default() -> Self {
        Self {
            anchors: Vec::new(),
            gate: GateConfig::default(),
            gated: true,
        }
    }
}

/// Finishes or unwinds every clustering pass a crash left behind, and
/// unlinks staging directories no checkpoint claims.
///
/// Returns one outcome per pass it resolved. A pass whose base version
/// the log can no longer reconstruct cannot be replanned, so its
/// speculative work is unlinked and the reason is logged rather than
/// returned as an outcome that did nothing
pub fn resume_cluster_passes(
    log: &TransactionLog,
    attempt: CommitAttempt<'_>,
    classes: &[PredicateClass],
    options: &ResumeOptions,
) -> Result<Vec<ClusterPassOutcome>, ZyronError> {
    let mut outcomes = Vec::new();
    let mut claimed: BTreeSet<u64> = BTreeSet::new();
    for path in checkpoint_files(log)? {
        let contents = match read_checkpoint(&path) {
            Ok(c) => c,
            Err(e) => {
                tracing::warn!(
                    checkpoint = %path.display(),
                    error = %e,
                    "clustering checkpoint unreadable, discarding the pass"
                );
                let _ = fs::remove_file(&path);
                continue;
            }
        };
        claimed.insert(contents.header.pass_id);
        if let Some(outcome) = resume_one(log, attempt, classes, options, &path, contents)? {
            outcomes.push(outcome);
        }
    }
    // Staging directories no checkpoint claims are debris from a crash
    // before the pass was announced. Nothing references them
    for (pass_id, dir) in staging_dirs(log)? {
        if !claimed.contains(&pass_id) {
            remove_staging(&dir)?;
        }
    }
    Ok(outcomes)
}

fn resume_one(
    log: &TransactionLog,
    attempt: CommitAttempt<'_>,
    classes: &[PredicateClass],
    options: &ResumeOptions,
    path: &Path,
    contents: CheckpointContents,
) -> Result<Option<ClusterPassOutcome>, ZyronError> {
    let pass_id = contents.header.pass_id;
    let staging = log.paths().staging_dir(pass_id);
    let pass_options = ClusterPassOptions {
        pass_id,
        target_rows_per_file: contents.header.target_rows_per_file,
        max_inputs: contents.inputs.len().max(1),
        anchors: options.anchors.clone(),
        gate: options.gate,
        gated: options.gated,
    };

    // Terminal states mean the work was decided and only the unlink was
    // lost, so there is nothing to judge again
    if matches!(contents.state, PassState::Done | PassState::Rejected) {
        remove_staging(&staging)?;
        fs::remove_file(path)?;
        return Ok(None);
    }

    let base = match log.manifest_at(contents.header.base_version) {
        Ok(m) => m,
        Err(e) => {
            tracing::warn!(
                pass_id,
                base_version = contents.header.base_version,
                error = %e,
                "clustering pass cannot be replanned, discarding it"
            );
            remove_staging(&staging)?;
            fs::remove_file(path)?;
            return Ok(None);
        }
    };
    let staged = read_staged_entries(&staging)?;

    // A commit that already landed leaves its outputs in the live
    // manifest. Recognizing that is what makes the commit idempotent
    if contents.state == PassState::Committing {
        let live = log.latest_manifest()?;
        if !staged.is_empty()
            && staged
                .iter()
                .all(|e| live.entry_for(e.partition_id).is_some())
        {
            let version = staged
                .first()
                .and_then(|e| live.entry_for(e.partition_id))
                .map(|e| e.added_version);
            remove_staging(&staging)?;
            fs::remove_file(path)?;
            return Ok(Some(ClusterPassOutcome {
                pass_id,
                decision: Decision::Accept { delta: 0.0 },
                version,
                inputs: contents.inputs.len(),
                outputs: staged.len(),
                rows_written: staged.iter().map(|e| e.row_count).sum(),
                bytes_read: 0,
                bytes_written: staged.iter().map(|e| e.size_bytes).sum(),
                resumed: true,
                merged: false,
            }));
        }
    }

    let ctx = PassContext {
        log,
        table_id: contents.header.table_id,
        base: &base,
        target: &contents.target,
        inputs: &contents.inputs,
        options: &pass_options,
        staging,
        resumed: true,
    };
    let mut checkpoint = PassCheckpoint::open_append(path)?;
    if contents.state == PassState::Committing {
        // Staged, accepted, and interrupted before the commit landed.
        // The gate already said yes, so finish what it decided
        let bytes_written: u64 = staged.iter().map(|e| e.size_bytes).sum();
        let rows_written: u64 = staged.iter().map(|e| e.row_count).sum();
        let version = commit_pass(&ctx, attempt, &staged)?;
        checkpoint.append(PassRecord::State(PassState::Done))?;
        remove_staging(&ctx.staging)?;
        checkpoint.discard()?;
        return Ok(Some(ClusterPassOutcome {
            pass_id,
            decision: Decision::Accept { delta: 0.0 },
            version: Some(version),
            inputs: contents.inputs.len(),
            outputs: staged.len(),
            rows_written,
            bytes_read: 0,
            bytes_written,
            resumed: true,
            merged: false,
        }));
    }
    drive_pass(&ctx, attempt, classes, checkpoint, staged).map(Some)
}

/// Every `.zycluster` checkpoint the table's log directory holds
fn checkpoint_files(log: &TransactionLog) -> Result<Vec<PathBuf>, ZyronError> {
    let dir = log.paths().log_dir().join("clustering");
    let mut found = Vec::new();
    let listing = match fs::read_dir(&dir) {
        Ok(l) => l,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(found),
        Err(e) => return Err(e.into()),
    };
    for dirent in listing {
        let dirent = dirent?;
        let name = dirent.file_name();
        let Some(name) = name.to_str() else { continue };
        if name.starts_with("pass_") && name.ends_with(".zycluster") {
            found.push(dirent.path());
        }
    }
    found.sort();
    Ok(found)
}

/// Every `_staging/pass_<id>` directory, with the id it names
fn staging_dirs(log: &TransactionLog) -> Result<Vec<(u64, PathBuf)>, ZyronError> {
    let dir = log.paths().root().join("_staging");
    let mut found = Vec::new();
    let listing = match fs::read_dir(&dir) {
        Ok(l) => l,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(found),
        Err(e) => return Err(e.into()),
    };
    for dirent in listing {
        let dirent = dirent?;
        let name = dirent.file_name();
        let Some(name) = name.to_str() else { continue };
        let Some(id) = name.strip_prefix("pass_").and_then(|n| n.parse().ok()) else {
            continue;
        };
        found.push((id, dirent.path()));
    }
    found.sort();
    Ok(found)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::manifest::ClusterKey;
    use crate::operations::append_rows;
    use crate::paths::LakePaths;
    use crate::predicate::{CompareOp, LakePredicate, PruneDecision};
    use crate::schema::{LakeColumn, LakeSchema};
    use std::collections::BTreeMap;
    use zyron_common::TypeId;

    /// A manifest with two files whose leading key ranges do not overlap,
    /// both written under spec three
    fn repair_manifest() -> ManifestFile {
        let bounds = |min: i64, max: i64| crate::predicate::ColumnBounds {
            min: Some(crate::predicate::LakeValue::Int(min)),
            max: Some(crate::predicate::LakeValue::Int(max)),
            null_count: 0,
            row_count: 100,
        };
        let entry = |partition_id: u64, min: i64, max: i64| PartitionEntry {
            partition_id,
            size_bytes: 4096,
            row_count: 100,
            added_version: 1,
            cluster_spec_id: 3,
            column_stats: std::sync::Arc::new(vec![crate::manifest::ColumnStatsEntry {
                ndv: Some(100),
                column_id: 0,
                bounds: bounds(min, max),
                bloom: None,
                size_bytes: Some(4096),
            }]),
            delete_predicate_ids: Vec::new(),
        };
        ManifestFile {
            snapshot_id: 9,
            parent_snapshot_id: 8,
            timestamp_us: 0,
            schema: schema(),
            cluster_spec: ClusterSpec {
                spec_id: 3,
                keys: vec![ClusterKey {
                    column_id: 0,
                    strategy: ClusterStrategy::RangePartition,
                    param: 0,
                }],
            },
            entries: vec![entry(1, 0, 99), entry(2, 100, 199)],
            delete_predicates: Vec::new(),
            properties: BTreeMap::new(),
            indexes: Vec::new(),
            index_files: Vec::new(),
        }
    }

    fn schema() -> LakeSchema {
        LakeSchema::new(
            1,
            vec![
                LakeColumn {
                    id: 0,
                    name: "a".into(),
                    type_id: TypeId::Int64,
                    nullable: false,
                    fractional_digits: None,
                    tz_offset_secs: None,
                    max_length: None,
                    default_expr: None,
                },
                LakeColumn {
                    id: 1,
                    name: "b".into(),
                    type_id: TypeId::Int64,
                    nullable: true,
                    fractional_digits: None,
                    tz_offset_secs: None,
                    max_length: None,
                    default_expr: None,
                },
            ],
        )
        .expect("schema")
    }

    fn attempt() -> CommitAttempt<'static> {
        CommitAttempt {
            operation: OperationKind::Append,
            db_txn_id: 0,
            commit_lsn: 1,
            timestamp_us: 1_754_700_000_000_000,
            read_predicate: None,
            read_version: 0,
            audit: None,
        }
    }

    fn new_log(dir: &Path, table_id: u32) -> TransactionLog {
        let mut create = attempt();
        create.operation = OperationKind::SchemaChange;
        TransactionLog::create(
            LakePaths::new(dir, table_id),
            create,
            &schema(),
            None,
            &BTreeMap::new(),
        )
        .expect("create")
    }

    fn batch(rows: &[(i64, Option<i64>)]) -> Vec<ColumnData> {
        vec![
            ColumnData {
                column_id: 0,
                cells: rows
                    .iter()
                    .map(|(a, _)| Some(a.to_le_bytes().to_vec()))
                    .collect(),
            },
            ColumnData {
                column_id: 1,
                cells: rows
                    .iter()
                    .map(|(_, b)| b.map(|v| v.to_le_bytes().to_vec()))
                    .collect(),
            },
        ]
    }

    fn cluster_on(column_id: u32, spec_id: u32) -> ClusterSpec {
        ClusterSpec {
            spec_id,
            keys: vec![ClusterKey {
                column_id,
                strategy: ClusterStrategy::RangePartition,
                param: 0,
            }],
        }
    }

    /// A range, deliberately: a value bloom already prunes equality on a
    /// low cardinality column whatever the layout is, so only a range
    /// predicate measures what clustering actually changes
    fn below(column_id: u32, value: i64) -> LakePredicate {
        LakePredicate::Compare {
            column_id,
            op: CompareOp::Lt,
            value: LakeValue::Int(value),
        }
    }

    fn class(predicate: LakePredicate, weight: f64) -> PredicateClass {
        PredicateClass {
            predicate,
            weight,
            measured_skip_rate: None,
        }
    }

    /// Four files each holding the whole value range, so nothing on `a`
    /// can be skipped until they are re-clustered
    fn interleaved_table(log: &TransactionLog) {
        for offset in 0..4i64 {
            let rows: Vec<(i64, Option<i64>)> =
                (0..16i64).map(|i| (i * 4 + offset, Some(i))).collect();
            append_rows(log, attempt(), 9, &batch(&rows)).expect("append");
        }
    }

    /// A pass that helps commits, and the outputs carry the target spec
    #[test]
    fn test_an_accepted_pass_rewrites_the_layout_and_commits_one_version() {
        let dir = tempfile::tempdir().expect("tempdir");
        let log = new_log(dir.path(), 9);
        interleaved_table(&log);
        let before = log.latest_manifest().expect("manifest");
        assert_eq!(before.entries.len(), 4);

        let target = cluster_on(0, 1);
        let mut options = ClusterPassOptions::new(1);
        options.target_rows_per_file = 16;
        let outcome = run_cluster_pass(
            &log,
            attempt(),
            9,
            &target,
            &[class(below(0, 8), 1.0)],
            &options,
        )
        .expect("pass");

        match outcome.decision {
            Decision::Accept { delta } => assert!(delta > 0.0, "delta {delta}"),
            other => panic!("expected accept, got {other:?}"),
        }
        assert!(
            !outcome.merged,
            "the spec changed, so there was no existing order to merge"
        );
        assert_eq!(outcome.inputs, 4);
        assert_eq!(outcome.outputs, 4);
        assert_eq!(outcome.rows_written, 64);
        assert!(outcome.version.is_some());

        let after = log.latest_manifest().expect("manifest");
        assert_eq!(after.entries.len(), 4);
        assert_eq!(after.cluster_spec, target);
        for entry in &after.entries {
            assert_eq!(entry.cluster_spec_id, 1);
        }
        // Disjoint ranges are the whole point: a range on `a` now opens
        // one file where before it opened every file
        let before_matching = before
            .entries
            .iter()
            .filter(|e| before.prune_file(&below(0, 8), e) != PruneDecision::CannotMatch)
            .count();
        assert_eq!(before_matching, 4);
        let matching = after
            .entries
            .iter()
            .filter(|e| after.prune_file(&below(0, 8), e) != PruneDecision::CannotMatch)
            .count();
        assert_eq!(matching, 1);

        // Every row survived the rewrite
        let mut values = Vec::new();
        for entry in &after.entries {
            let reader = LakeFileReader::open(log.paths(), entry.partition_id).expect("open");
            let column = reader
                .read_column(after.schema.column_by_id(0).expect("a"))
                .expect("column");
            for row in 0..reader.row_count() {
                let mut buf = [0u8; 8];
                buf.copy_from_slice(column.cell(row).expect("cell"));
                values.push(i64::from_le_bytes(buf));
            }
        }
        values.sort_unstable();
        assert_eq!(values, (0..64i64).collect::<Vec<_>>());

        // Staging and the checkpoint are both reclaimed
        assert!(!log.paths().staging_dir(1).exists());
        assert!(!log.paths().clustering_pass(1).exists());
    }

    /// The invariant the module exists to hold: a refused pass leaves
    /// every active data file byte identical
    #[test]
    fn test_a_refused_pass_leaves_the_active_files_byte_identical() {
        let dir = tempfile::tempdir().expect("tempdir");
        let log = new_log(dir.path(), 11);
        interleaved_table(&log);
        let before = log.latest_manifest().expect("manifest");
        let digests: Vec<(u64, Vec<u8>)> = before
            .entries
            .iter()
            .map(|e| {
                (
                    e.partition_id,
                    fs::read(log.paths().data_file(e.partition_id)).expect("read"),
                )
            })
            .collect();

        // No recorded workload means no evidence, and no evidence is not
        // an improvement
        let outcome = run_cluster_pass(
            &log,
            attempt(),
            11,
            &cluster_on(0, 1),
            &[],
            &ClusterPassOptions::new(2),
        )
        .expect("pass");
        assert!(matches!(outcome.decision, Decision::BelowThreshold { .. }));
        assert_eq!(outcome.version, None);

        let after = log.latest_manifest().expect("manifest");
        assert_eq!(after.snapshot_id, before.snapshot_id);
        assert_eq!(after.cluster_spec.spec_id, before.cluster_spec.spec_id);
        for (partition_id, bytes) in digests {
            let now = fs::read(log.paths().data_file(partition_id)).expect("read");
            assert_eq!(now, bytes, "partition {partition_id:#x} was rewritten");
        }
        assert!(!log.paths().staging_dir(2).exists());
        assert!(!log.paths().clustering_pass(2).exists());
    }

    /// An accepted pass removes its inputs logically. Their bytes stay
    /// exactly as they were until vacuum passes the retention floor,
    /// which is what makes reading a past version read that version
    #[test]
    fn test_an_accepted_pass_never_modifies_an_input_file() {
        let dir = tempfile::tempdir().expect("tempdir");
        let log = new_log(dir.path(), 25);
        interleaved_table(&log);
        let before = log.latest_manifest().expect("manifest");
        let version_before = before.snapshot_id;
        let digests: Vec<(u64, Vec<u8>)> = before
            .entries
            .iter()
            .map(|e| {
                (
                    e.partition_id,
                    fs::read(log.paths().data_file(e.partition_id)).expect("read"),
                )
            })
            .collect();

        let mut options = ClusterPassOptions::new(6);
        options.target_rows_per_file = 16;
        let outcome = run_cluster_pass(
            &log,
            attempt(),
            25,
            &cluster_on(0, 1),
            &[class(below(0, 8), 1.0)],
            &options,
        )
        .expect("pass");
        assert!(matches!(outcome.decision, Decision::Accept { .. }));

        for (partition_id, bytes) in &digests {
            let now = fs::read(log.paths().data_file(*partition_id)).expect("read");
            assert_eq!(&now, bytes, "input {partition_id:#x} was opened for write");
        }
        // And the past version still resolves to them
        let past = log.manifest_at(version_before).expect("past manifest");
        let past_ids: Vec<u64> = past.entries.iter().map(|e| e.partition_id).collect();
        let mut expected: Vec<u64> = digests.iter().map(|(id, _)| *id).collect();
        expected.sort_unstable();
        assert_eq!(past_ids, expected);
        assert_eq!(past.cluster_spec.spec_id, 0);
    }

    /// Force is the operator having decided. The gate reports what it
    /// would have said and the declared layout is still applied
    #[test]
    fn test_an_ungated_pass_applies_a_declared_layout_the_gate_would_refuse() {
        let dir = tempfile::tempdir().expect("tempdir");
        let log = new_log(dir.path(), 31);
        interleaved_table(&log);

        let mut options = ClusterPassOptions::new(11);
        options.target_rows_per_file = 16;
        options.gated = false;
        // No evidence at all, which is exactly what a gated pass refuses
        let outcome =
            run_cluster_pass(&log, attempt(), 31, &cluster_on(0, 1), &[], &options).expect("pass");
        assert!(
            matches!(outcome.decision, Decision::BelowThreshold { .. }),
            "the verdict is reported honestly, got {:?}",
            outcome.decision
        );
        assert!(
            outcome.version.is_some(),
            "a declared layout is applied whatever the gate would have said"
        );
        let after = log.latest_manifest().expect("manifest");
        assert_eq!(after.cluster_spec.spec_id, 1);

        // An anchor conflict is still fatal, ungated or not
        let mut options = ClusterPassOptions::new(12);
        options.target_rows_per_file = 16;
        options.gated = false;
        options.anchors = vec![1];
        let refused =
            run_cluster_pass(&log, attempt(), 31, &cluster_on(0, 2), &[], &options).expect("pass");
        assert!(matches!(refused.decision, Decision::AnchorConflict { .. }));
        assert_eq!(refused.version, None);
    }

    /// An anchor the proposal drops is refused before any file is read
    #[test]
    fn test_a_proposal_that_breaks_an_anchor_is_refused() {
        let dir = tempfile::tempdir().expect("tempdir");
        let log = new_log(dir.path(), 13);
        interleaved_table(&log);
        let mut options = ClusterPassOptions::new(3);
        options.anchors = vec![1];
        let outcome = run_cluster_pass(
            &log,
            attempt(),
            13,
            &cluster_on(0, 1),
            &[class(below(0, 8), 1.0)],
            &options,
        )
        .expect("pass");
        match outcome.decision {
            Decision::AnchorConflict { expected, found } => {
                assert_eq!(expected, vec![1]);
                assert_eq!(found, vec![0]);
            }
            other => panic!("expected an anchor conflict, got {other:?}"),
        }
        assert_eq!(outcome.version, None);
    }

    /// Appends after a clustering pass land under the target spec but
    /// overlap the clustered files, which is the drift the merge path
    /// exists to fold back in without a sort
    #[test]
    fn test_a_later_append_that_overlaps_the_clustered_set_is_folded_back_in() {
        let dir = tempfile::tempdir().expect("tempdir");
        let log = new_log(dir.path(), 27);
        interleaved_table(&log);
        let target = cluster_on(0, 1);
        let mut options = ClusterPassOptions::new(1);
        options.target_rows_per_file = 16;
        run_cluster_pass(
            &log,
            attempt(),
            27,
            &target,
            &[class(below(0, 8), 1.0)],
            &options,
        )
        .expect("first pass");

        // A fresh append spanning the whole range: written under the
        // target spec, and overlapping every clustered file
        let rows: Vec<(i64, Option<i64>)> = (0..8i64).map(|i| (i * 8, Some(i))).collect();
        append_rows(&log, attempt(), 27, &batch(&rows)).expect("append");
        let drifted = log.latest_manifest().expect("manifest");
        assert_eq!(drifted.entries.len(), 5);
        for entry in &drifted.entries {
            assert_eq!(entry.cluster_spec_id, 1);
        }
        let matching = drifted
            .entries
            .iter()
            .filter(|e| drifted.prune_file(&below(0, 8), e) != PruneDecision::CannotMatch)
            .count();
        assert_eq!(matching, 2, "the append reopened the range");

        // Every input already carries the target spec, so this pass takes
        // the merge path rather than a comparison sort
        let mut options = ClusterPassOptions::new(2);
        options.target_rows_per_file = 16;
        let outcome = run_cluster_pass(
            &log,
            attempt(),
            27,
            &target,
            &[class(below(0, 8), 1.0)],
            &options,
        )
        .expect("second pass");
        assert_eq!(outcome.inputs, 5);
        assert!(
            outcome.merged,
            "every input already carried the target spec, so the order is a merge"
        );
        assert!(matches!(outcome.decision, Decision::Accept { .. }));
        assert_eq!(outcome.rows_written, 72);

        let after = log.latest_manifest().expect("manifest");
        let matching = after
            .entries
            .iter()
            .filter(|e| after.prune_file(&below(0, 8), e) != PruneDecision::CannotMatch)
            .count();
        assert_eq!(matching, 1);

        // The merge order is the value order, so every output file holds
        // a contiguous run and reading them in key order reproduces the
        // whole table sorted
        let mut values = Vec::new();
        for entry in &after.entries {
            let reader = LakeFileReader::open(log.paths(), entry.partition_id).expect("open");
            let column = reader
                .read_column(after.schema.column_by_id(0).expect("a"))
                .expect("column");
            for row in 0..reader.row_count() {
                let mut buf = [0u8; 8];
                buf.copy_from_slice(column.cell(row).expect("cell"));
                values.push(i64::from_le_bytes(buf));
            }
        }
        assert_eq!(values.len(), 72);
        let mut expected: Vec<i64> = (0..64i64).collect();
        expected.extend((0..8i64).map(|i| i * 8));
        expected.sort_unstable();
        values.sort_unstable();
        assert_eq!(values, expected);
    }

    /// The full loop with nothing hand-fed: scans observe, the manifest
    /// supplies the shape, the planner proposes from both, and the gate
    /// judges the proposal against what was actually measured
    #[test]
    fn test_a_pass_driven_only_by_observed_scans_improves_the_layout() {
        use crate::planner::{evidence_from_manifest, predicate_classes, propose};
        use crate::workload::{WorkloadObserver, observe_scan};

        let dir = tempfile::tempdir().expect("tempdir");
        let log = new_log(dir.path(), 29);
        interleaved_table(&log);
        let base = log.latest_manifest().expect("manifest");

        // A workload that filters on `a` by range, observed the way the
        // scan path observes it: bytes considered and bytes the statistics
        // let it skip, which on this layout is none of them
        let observer = WorkloadObserver::new();
        let bytes: u64 = base.entries.iter().map(|e| e.size_bytes).sum();
        let epoch = 1000u16;
        for _ in 0..20 {
            // What one scan of `a < 8` reports: every byte considered,
            // none skipped, 64 rows decoded and 8 returned
            crate::workload::observe_for_test(&observer, 29, &below(0, 8), bytes, 0, 64, 8, epoch);
        }
        // The process-wide entry points are the ones the scan path calls
        observe_scan(29, &below(0, 8), bytes, 0, epoch);
        crate::workload::observe_scan_result(29, &below(0, 8), 64, 8, epoch);

        let evidence = evidence_from_manifest(&base, &observer, 29, epoch);
        let column_a = evidence
            .iter()
            .find(|c| c.column_id == 0)
            .expect("column a has evidence");
        assert_eq!(column_a.ndv, 16, "each file holds sixteen distinct values");
        assert_eq!(column_a.row_count, 16);
        assert!(column_a.range_weight > 0.0, "range scans were observed");
        assert_eq!(column_a.equality_weight, 0.0);
        assert_eq!(
            crate::planner::measured_skip_rate(&observer, 29, 0, epoch),
            Some(0.0),
            "the current layout let nothing be skipped"
        );
        assert_eq!(
            crate::planner::measured_selectivity(&observer, 29, 0, epoch),
            Some(0.125),
            "eight of sixty four rows came back"
        );

        let keys = propose(&evidence, &[], 2);
        assert!(
            keys.iter().any(|k| k.column_id == 0),
            "the column the workload filters on must be proposed"
        );
        let classes = predicate_classes(&base, &evidence, &observer, 29, epoch);
        assert!(!classes.is_empty());

        let target = ClusterSpec {
            spec_id: 1,
            keys: keys.clone(),
        };
        let mut options = ClusterPassOptions::new(9);
        options.target_rows_per_file = 16;
        let outcome =
            run_cluster_pass(&log, attempt(), 29, &target, &classes, &options).expect("pass");
        match outcome.decision {
            Decision::Accept { delta } => assert!(delta > 0.0, "delta {delta}"),
            other => panic!("expected accept from measured evidence, got {other:?}"),
        }

        let after = log.latest_manifest().expect("manifest");
        let matching = after
            .entries
            .iter()
            .filter(|e| after.prune_file(&below(0, 8), e) != PruneDecision::CannotMatch)
            .count();
        assert!(matching < 4, "the observed range predicate now skips files");
    }

    /// One node writes a dataset and the others read it. A second writer
    /// is refused rather than merged, because merging forces last-write-wins
    /// or a CRDT, and neither can express an invariant like `balance >= 0`.
    ///
    /// Each log carries the node it writes as, so this drives two writers
    /// without touching the process identity every other test shares.
    #[test]
    fn test_a_second_writer_to_a_dataset_is_refused() {
        use crate::transaction_log::{
            LogEntry, WRITER_NODE_PROPERTY, transfer_writer, writer_node,
        };

        const NODE_A: u64 = 0xAAAA_AAAA_AAAA_AAAA;
        const NODE_B: u64 = 0xBBBB_BBBB_BBBB_BBBB;

        let dir = tempfile::tempdir().expect("tempdir");
        let log = new_log(dir.path(), 41);
        log.set_writer_identity(NODE_A);
        // The table was created with no identity, so the first write claims it
        append_rows(&log, attempt(), 41, &batch(&[(1, Some(1))])).expect("owner writes");
        assert_eq!(
            writer_node(&log.latest_manifest().expect("manifest")),
            Some(NODE_A)
        );

        // Node B is refused, and the message names both nodes
        log.set_writer_identity(NODE_B);
        let refused = append_rows(&log, attempt(), 41, &batch(&[(2, Some(2))]))
            .expect_err("a second writer must be refused");
        let message = refused.to_string();
        assert!(message.contains("aaaaaaaaaaaaaaaa"), "{message}");
        assert!(message.contains("bbbbbbbbbbbbbbbb"), "{message}");

        // The refusal wrote nothing
        let after = log.latest_manifest().expect("manifest");
        assert_eq!(after.entries.len(), 1, "the refused write left no file");

        // A data write cannot take ownership as a side effect. Only a commit
        // that is exactly a handover counts as one, which is the door the
        // rule leaves open and the only one
        let smuggled = log.commit(attempt(), |_| {
            Ok(vec![
                LogEntry::SetProperty {
                    key: WRITER_NODE_PROPERTY.to_string(),
                    value: NODE_B.to_string(),
                },
                LogEntry::SetProperty {
                    key: "unrelated".to_string(),
                    value: "x".to_string(),
                },
            ])
        });
        assert!(
            smuggled.is_err(),
            "ownership must not change as a side effect of another write"
        );

        // Ownership transfers explicitly, and then node B may write
        transfer_writer(&log, attempt(), NODE_B).expect("transfer");
        assert_eq!(
            writer_node(&log.latest_manifest().expect("manifest")),
            Some(NODE_B)
        );
        append_rows(&log, attempt(), 41, &batch(&[(2, Some(2))])).expect("new owner writes");

        // And node A is now the one refused
        log.set_writer_identity(NODE_A);
        assert!(append_rows(&log, attempt(), 41, &batch(&[(3, Some(3))])).is_err());

        // No identity established means no enforcement: a tool operating on
        // its own data directory is not a second writer
        log.set_writer_identity(0);
        append_rows(&log, attempt(), 41, &batch(&[(4, Some(4))])).expect("tool writes");
        // and it does not steal ownership by writing
        assert_eq!(
            writer_node(&log.latest_manifest().expect("manifest")),
            Some(NODE_B)
        );
    }

    /// Transferring to node zero would leave the dataset owned by nobody
    #[test]
    fn test_a_transfer_needs_a_real_node() {
        let dir = tempfile::tempdir().expect("tempdir");
        let log = new_log(dir.path(), 43);
        assert!(crate::transaction_log::transfer_writer(&log, attempt(), 0).is_err());
    }

    /// A clustered set whose files do not overlap has no drift to fold
    #[test]
    fn test_a_table_already_under_the_target_spec_is_idle() {
        let dir = tempfile::tempdir().expect("tempdir");
        let log = new_log(dir.path(), 15);
        interleaved_table(&log);
        let target = cluster_on(0, 1);
        let mut options = ClusterPassOptions::new(4);
        options.target_rows_per_file = 16;
        run_cluster_pass(
            &log,
            attempt(),
            15,
            &target,
            &[class(below(0, 8), 1.0)],
            &options,
        )
        .expect("first pass");

        let idle = run_cluster_pass(
            &log,
            attempt(),
            15,
            &target,
            &[class(below(0, 8), 1.0)],
            &ClusterPassOptions::new(5),
        )
        .expect("second pass");
        assert_eq!(idle.inputs, 0);
        assert_eq!(idle.outputs, 0);
        assert_eq!(idle.version, None);
    }

    /// A crash after staging leaves the outputs on disk. Resume must
    /// score and commit them rather than rewrite them
    #[test]
    fn test_resume_finishes_a_pass_interrupted_after_staging() {
        let dir = tempfile::tempdir().expect("tempdir");
        let log = new_log(dir.path(), 17);
        interleaved_table(&log);
        let base = log.latest_manifest().expect("manifest");
        let target = cluster_on(0, 1);
        let inputs: Vec<u64> = base.entries.iter().map(|e| e.partition_id).collect();

        // Stage by hand, exactly as run_cluster_pass would, then stop
        let staging = log.paths().staging_dir(7);
        fs::create_dir_all(&staging).expect("staging");
        let header = PassHeader {
            pass_id: 7,
            base_version: base.snapshot_id,
            table_id: 17,
            target_rows_per_file: 16,
            created_us: attempt().timestamp_us,
            target_spec_id: target.spec_id,
        };
        let path = log.paths().clustering_pass(7);
        let mut checkpoint = PassCheckpoint::create(&path, &header).expect("checkpoint");
        for key in &target.keys {
            checkpoint
                .append(PassRecord::Key {
                    column_id: key.column_id,
                    strategy: key.strategy,
                    param: key.param,
                })
                .expect("key");
        }
        for partition_id in &inputs {
            checkpoint
                .append(PassRecord::Input {
                    partition_id: *partition_id,
                })
                .expect("input");
        }
        checkpoint
            .append(PassRecord::State(PassState::Running))
            .expect("state");
        let options = ClusterPassOptions {
            pass_id: 7,
            target_rows_per_file: 16,
            max_inputs: inputs.len(),
            anchors: Vec::new(),
            gate: GateConfig::default(),
            gated: true,
        };
        let ctx = PassContext {
            log: &log,
            table_id: 17,
            base: &base,
            target: &target,
            inputs: &inputs,
            options: &options,
            staging: staging.clone(),
            resumed: false,
        };
        let staged = stage_outputs(&ctx, Vec::new()).expect("stage").entries;
        checkpoint
            .append(PassRecord::State(PassState::Staged))
            .expect("staged");
        drop(checkpoint);
        assert_eq!(staged.len(), 4);
        let staged_ids: Vec<u64> = staged.iter().map(|e| e.partition_id).collect();

        let outcomes = resume_cluster_passes(
            &log,
            attempt(),
            &[class(below(0, 8), 1.0)],
            &ResumeOptions::default(),
        )
        .expect("resume");
        assert_eq!(outcomes.len(), 1);
        assert!(outcomes[0].resumed);
        assert!(matches!(outcomes[0].decision, Decision::Accept { .. }));
        assert!(outcomes[0].version.is_some());

        // The committed files are the ones that were already staged, so
        // resume scored the existing work instead of redoing it
        let after = log.latest_manifest().expect("manifest");
        let live: Vec<u64> = after.entries.iter().map(|e| e.partition_id).collect();
        assert_eq!(live, {
            let mut ids = staged_ids;
            ids.sort_unstable();
            ids
        });
        assert!(!staging.exists());
        assert!(!path.exists());
    }

    /// Resume must also unwind, not only finish. A checkpoint whose base
    /// version the log cannot reconstruct is unresumable
    #[test]
    fn test_resume_unlinks_staging_no_checkpoint_claims() {
        let dir = tempfile::tempdir().expect("tempdir");
        let log = new_log(dir.path(), 19);
        interleaved_table(&log);
        let orphan = log.paths().staging_dir(42);
        fs::create_dir_all(&orphan).expect("orphan");
        fs::write(orphan.join("p-00000000000000ff.zyr"), b"debris").expect("debris");

        let outcomes =
            resume_cluster_passes(&log, attempt(), &[], &ResumeOptions::default()).expect("resume");
        assert!(outcomes.is_empty());
        assert!(!orphan.exists());
    }

    /// The rewrite reads through the survivor mask, so a delete recorded
    /// as a predicate is applied for good and the predicate retires
    #[test]
    fn test_a_pass_applies_attached_delete_predicates_and_retires_them() {
        let dir = tempfile::tempdir().expect("tempdir");
        let log = new_log(dir.path(), 21);
        interleaved_table(&log);
        let deleted = crate::operations::delete_where(
            &log,
            attempt(),
            &LakePredicate::Compare {
                column_id: 0,
                op: CompareOp::Lt,
                value: LakeValue::Int(8),
            },
            "a < 8",
        )
        .expect("delete");
        assert!(deleted.predicate_recorded);

        let mut options = ClusterPassOptions::new(8);
        options.target_rows_per_file = 16;
        let outcome = run_cluster_pass(
            &log,
            attempt(),
            21,
            &cluster_on(0, 1),
            &[class(below(0, 22), 1.0)],
            &options,
        )
        .expect("pass");
        assert!(matches!(outcome.decision, Decision::Accept { .. }));
        assert_eq!(outcome.rows_written, 56);

        let after = log.latest_manifest().expect("manifest");
        assert!(after.delete_predicates.is_empty());
        for entry in &after.entries {
            assert!(entry.delete_predicate_ids.is_empty());
        }
        let rows: u64 = after.entries.iter().map(|e| e.row_count).sum();
        assert_eq!(rows, 56);
    }

    /// A torn trailing record is what a crash mid-append looks like, and
    /// it must cost the last record rather than the whole checkpoint
    #[test]
    fn test_a_torn_checkpoint_record_is_ignored() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("pass_1.zycluster");
        let header = PassHeader {
            pass_id: 1,
            base_version: 4,
            table_id: 9,
            target_rows_per_file: 64,
            created_us: 17,
            target_spec_id: 3,
        };
        let mut checkpoint = PassCheckpoint::create(&path, &header).expect("create");
        checkpoint
            .append(PassRecord::Input { partition_id: 0xAB })
            .expect("input");
        checkpoint
            .append(PassRecord::State(PassState::Running))
            .expect("state");
        drop(checkpoint);

        // Half of one more record, exactly what a crash leaves
        let mut file = OpenOptions::new().append(true).open(&path).expect("append");
        file.write_all(&[0u8; CHECKPOINT_RECORD_LEN / 2])
            .expect("torn");
        drop(file);

        let contents = read_checkpoint(&path).expect("read");
        assert_eq!(contents.header, header);
        assert_eq!(contents.inputs, vec![0xAB]);
        assert_eq!(contents.state, PassState::Running);

        // A whole record whose CRC does not hold is equally ignored
        let mut file = OpenOptions::new().append(true).open(&path).expect("append");
        file.write_all(&[7u8; CHECKPOINT_RECORD_LEN]).expect("bad");
        drop(file);
        let contents = read_checkpoint(&path).expect("read");
        assert_eq!(contents.state, PassState::Running);
    }

    /// The staging sidecar is what resume reads instead of re-deriving
    /// statistics, so it has to round trip an entry exactly
    #[test]
    fn test_staged_entries_round_trip_and_stop_at_a_torn_record() {
        let dir = tempfile::tempdir().expect("tempdir");
        let log = new_log(dir.path(), 23);
        append_rows(&log, attempt(), 23, &batch(&[(1, Some(2)), (3, None)])).expect("append");
        let manifest = log.latest_manifest().expect("manifest");
        let entry = manifest.entries[0].clone();

        let staging = dir.path().join("staging");
        fs::create_dir_all(&staging).expect("staging");
        append_staged_entry(&staging, &entry).expect("append entry");
        assert_eq!(read_staged_entries(&staging).expect("read"), vec![entry]);

        let mut file = OpenOptions::new()
            .append(true)
            .open(staging.join(STAGED_ENTRIES_NAME))
            .expect("open");
        file.write_all(&[9u8; 6]).expect("torn");
        drop(file);
        assert_eq!(read_staged_entries(&staging).expect("read").len(), 1);
    }

    /// Merge and sort must produce the same global order when both are
    /// legal, or a coalescing pass and a re-cluster pass would disagree
    /// about the layout they produce
    #[test]
    fn test_merge_and_sort_agree_on_the_global_order() {
        let per_input = vec![
            InputRows {
                rows: vec![0, 1, 2],
                ranked: vec![0, 1, 2],
                keys: [10u64, 30, 50]
                    .iter()
                    .flat_map(|v| v.to_be_bytes())
                    .collect(),
                nulls: Vec::new(),
            },
            InputRows {
                rows: vec![0, 1],
                ranked: vec![0, 1],
                keys: [20u64, 40].iter().flat_map(|v| v.to_be_bytes()).collect(),
                nulls: Vec::new(),
            },
        ];
        let merged = merge_order(&per_input, 8).expect("merge");
        let sorted = sort_order(&per_input, 8);
        assert_eq!(
            merged,
            vec![(0, 0), (1, 0), (0, 1), (1, 1), (0, 2)],
            "merge must interleave the two runs by key"
        );
        assert_eq!(merged, sorted);
    }

    /// The urgency signal and the pass have to agree on what needs repair,
    /// or a table trips the fast lane and the pass it triggers finds
    /// nothing to rewrite
    #[test]
    fn test_the_drift_count_is_what_a_pass_would_rewrite() {
        let mut manifest = repair_manifest();
        // Two files at an older spec, which is drift by definition
        for entry in &mut manifest.entries {
            entry.cluster_spec_id = 1;
        }
        assert_eq!(drifted_file_count(&manifest), 2);
        assert_eq!(
            select_inputs(&manifest, &manifest.cluster_spec, 16).len(),
            2,
            "the pass has to rewrite exactly what the signal counted"
        );

        // Brought up to the current spec with disjoint ranges, so nothing
        // is drifted any more
        for entry in &mut manifest.entries {
            entry.cluster_spec_id = 3;
        }
        assert_eq!(
            drifted_file_count(&manifest),
            0,
            "files at the current spec whose ranges do not overlap are the layout working"
        );

        // A table with no layout has nothing to have drifted from
        manifest.cluster_spec = ClusterSpec::none();
        assert_eq!(drifted_file_count(&manifest), 0);
    }

    /// A bound on how much one pass rewrites is a bound on how long it runs
    /// and how much it writes, and the right value differs per table
    #[test]
    fn test_repair_tuning_comes_from_the_table_then_the_node() {
        let mut manifest = repair_manifest();
        assert_eq!(manifest.cluster_repair_max_inputs(16), 16);
        assert_eq!(manifest.cluster_repair_interval_secs(300), 300);
        assert_eq!(
            manifest.cluster_repair_urgency_threshold(),
            crate::manifest::DEFAULT_CLUSTER_REPAIR_URGENCY_THRESHOLD
        );

        manifest.properties.insert(
            crate::manifest::CLUSTER_REPAIR_MAX_INPUTS_PROPERTY.into(),
            "4".into(),
        );
        manifest.properties.insert(
            crate::manifest::CLUSTER_REPAIR_INTERVAL_SECS_PROPERTY.into(),
            "900".into(),
        );
        manifest.properties.insert(
            crate::manifest::CLUSTER_REPAIR_URGENCY_THRESHOLD_PROPERTY.into(),
            "2".into(),
        );
        assert_eq!(manifest.cluster_repair_max_inputs(16), 4);
        assert_eq!(manifest.cluster_repair_interval_secs(300), 900);
        assert_eq!(manifest.cluster_repair_urgency_threshold(), 2);

        // A pass that rewrote nothing is not a bound, it is a stall
        manifest.properties.insert(
            crate::manifest::CLUSTER_REPAIR_MAX_INPUTS_PROPERTY.into(),
            "0".into(),
        );
        assert_eq!(
            manifest.cluster_repair_max_inputs(16),
            1,
            "a bound of zero would leave a drifted table drifted forever"
        );

        // A threshold of zero asks for a pass as soon as any file needs one
        manifest.properties.insert(
            crate::manifest::CLUSTER_REPAIR_URGENCY_THRESHOLD_PROPERTY.into(),
            "0".into(),
        );
        assert_eq!(manifest.cluster_repair_urgency_threshold(), 0);
    }

    /// A value nobody can read falls back rather than stopping maintenance,
    /// the same as every other maintenance threshold
    #[test]
    fn test_an_unreadable_repair_setting_falls_back() {
        let mut manifest = repair_manifest();
        manifest.properties.insert(
            crate::manifest::CLUSTER_REPAIR_INTERVAL_SECS_PROPERTY.into(),
            "soon".into(),
        );
        assert_eq!(manifest.cluster_repair_interval_secs(300), 300);
    }
}
