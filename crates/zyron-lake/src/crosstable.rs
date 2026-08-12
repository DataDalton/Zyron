//! Cross-table atomic commit.
//!
//! Each table's log commits atomically on its own. Two tables committed by
//! one statement are not atomic together unless something outside both logs
//! records whether the pair happened, which is what an intent file is.
//!
//! The protocol is the pending-version machinery with the intent file
//! playing the part the database's commit record plays for an ordinary
//! transaction:
//!
//!   1. `prepare` writes `<data_dir>/lake/_txn/<seq>.intent` marked
//!      PREPARING and fsyncs it.
//!   2. Each table commits its version under the intent's transaction id.
//!      Those versions exist on disk and are invisible to readers.
//!   3. `commit` rewrites the intent as COMMITTED and fsyncs. **That write
//!      is the commit point for every participant at once.**
//!   4. The versions publish and the intent file is removed.
//!
//! Recovery reads the leftover intents: COMMITTED means step 3 completed, so
//! every participant's version is legitimate; PREPARING means it did not, so
//! every participant's version is discarded. A crash between two tables can
//! therefore only produce all or none, never one.
//!
//! Intent transaction ids carry a high bit so they can never be confused
//! with a database transaction id, and `IntentAware` routes each id to the
//! oracle that owns it.

use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

use zyron_common::ZyronError;

use crate::paths::txn_intent_file;
use crate::transaction_log::CommitStatus;

/// Marks a transaction id as an intent sequence rather than a database
/// transaction, so the two namespaces can never collide.
pub const INTENT_TXN_FLAG: u64 = 1 << 63;

const INTENT_MAGIC: [u8; 4] = *b"ZYTX";
const INTENT_LEN: usize = 24;
const STATE_PREPARING: u8 = 1;
const STATE_COMMITTED: u8 = 2;

/// How far a cross-table transaction got.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IntentState {
    /// Participants may have written versions, none of them count
    Preparing,
    /// Every participant's version counts
    Committed,
}

/// Next sequence this process hands out, primed from disk on first use.
static NEXT_SEQ: AtomicU64 = AtomicU64::new(0);

fn intent_dir(data_dir: &Path) -> PathBuf {
    data_dir.join("lake").join("_txn")
}

fn encode_intent(seq: u64, state: u8, timestamp_us: i64) -> [u8; INTENT_LEN] {
    let mut buf = [0u8; INTENT_LEN];
    buf[0..4].copy_from_slice(&INTENT_MAGIC);
    buf[4] = state;
    buf[5..13].copy_from_slice(&seq.to_le_bytes());
    buf[13..21].copy_from_slice(&timestamp_us.to_le_bytes()[..8]);
    let crc = zyron_common::hash32(&buf[0..21]);
    buf[21..24].copy_from_slice(&crc.to_le_bytes()[..3]);
    buf
}

fn decode_intent(bytes: &[u8], ctx: &str) -> Result<(u64, IntentState), ZyronError> {
    if bytes.len() != INTENT_LEN || bytes[0..4] != INTENT_MAGIC {
        return Err(ZyronError::ManifestCorrupted {
            path: ctx.to_string(),
            reason: "commit intent is not a commit intent".into(),
        });
    }
    let crc = zyron_common::hash32(&bytes[0..21]);
    if bytes[21..24] != crc.to_le_bytes()[..3] {
        return Err(ZyronError::ManifestCorrupted {
            path: ctx.to_string(),
            reason: "commit intent checksum mismatch".into(),
        });
    }
    let state = match bytes[4] {
        STATE_PREPARING => IntentState::Preparing,
        STATE_COMMITTED => IntentState::Committed,
        other => {
            return Err(ZyronError::ManifestCorrupted {
                path: ctx.to_string(),
                reason: format!("unknown commit intent state {}", other),
            });
        }
    };
    let mut b = [0u8; 8];
    b.copy_from_slice(&bytes[5..13]);
    Ok((u64::from_le_bytes(b), state))
}

/// The state of one intent-backed transaction.
///
/// An absent intent means the transaction finished: a commit removes the
/// file only after publishing, and an abort removes the participants'
/// version files before removing the file, so nothing that reads a version
/// header can find an id whose abort left the file behind.
pub fn intent_state(data_dir: &Path, txn_id: u64) -> Option<IntentState> {
    if txn_id & INTENT_TXN_FLAG == 0 {
        return None;
    }
    let seq = txn_id & !INTENT_TXN_FLAG;
    let path = txn_intent_file(data_dir, seq);
    let bytes = fs::read(&path).ok()?;
    decode_intent(&bytes, &path.to_string_lossy())
        .ok()
        .map(|(_, state)| state)
}

/// Routes a transaction id to the oracle that owns it: an intent-flagged id
/// to the intent files, everything else to the database's commit status.
pub struct IntentAware<'a> {
    pub inner: &'a dyn CommitStatus,
    pub data_dir: PathBuf,
}

impl<'a> IntentAware<'a> {
    pub fn new(inner: &'a dyn CommitStatus, data_dir: impl Into<PathBuf>) -> Self {
        Self {
            inner,
            data_dir: data_dir.into(),
        }
    }
}

impl CommitStatus for IntentAware<'_> {
    fn is_committed(&self, db_txn_id: u64) -> bool {
        if db_txn_id & INTENT_TXN_FLAG == 0 {
            return self.inner.is_committed(db_txn_id);
        }
        match intent_state(&self.data_dir, db_txn_id) {
            Some(IntentState::Committed) => true,
            Some(IntentState::Preparing) => false,
            // The transaction finished and cleaned up, so the versions
            // carrying this id are the ones it committed
            None => true,
        }
    }
}

/// A commit spanning several lake tables.
pub struct CrossTableTxn {
    data_dir: PathBuf,
    seq: u64,
    timestamp_us: i64,
    prepared: bool,
    /// Versions this transaction wrote, in the order they were written. The
    /// transaction publishes and abandons through this list rather than a
    /// process-wide registry, so it works for any caller holding the logs
    participants: Vec<(std::sync::Arc<crate::transaction_log::TransactionLog>, u64)>,
}

impl CrossTableTxn {
    /// Allocates a sequence. Nothing is written until `prepare`.
    pub fn begin(data_dir: impl Into<PathBuf>, timestamp_us: i64) -> Result<Self, ZyronError> {
        let data_dir = data_dir.into();
        let seq = allocate_seq(&data_dir)?;
        Ok(Self {
            data_dir,
            seq,
            timestamp_us,
            prepared: false,
            participants: Vec::new(),
        })
    }

    /// Records a version one participant just wrote under this transaction.
    ///
    /// The caller has both the log and the version the commit returned, and
    /// this is what `commit` publishes and `abort` discards.
    pub fn record(
        &mut self,
        log: std::sync::Arc<crate::transaction_log::TransactionLog>,
        version: u64,
    ) {
        self.participants.push((log, version));
    }

    /// How many versions this transaction has written.
    pub fn participant_count(&self) -> usize {
        self.participants.len()
    }

    /// The transaction id every participant commits under.
    pub fn txn_id(&self) -> u64 {
        self.seq | INTENT_TXN_FLAG
    }

    pub fn sequence(&self) -> u64 {
        self.seq
    }

    /// Records the intent before any participant writes, so a crash from
    /// here on is recoverable to all or none.
    pub fn prepare(&mut self) -> Result<(), ZyronError> {
        let dir = intent_dir(&self.data_dir);
        fs::create_dir_all(&dir)?;
        let path = txn_intent_file(&self.data_dir, self.seq);
        write_intent(&path, self.seq, STATE_PREPARING, self.timestamp_us)?;
        self.prepared = true;
        Ok(())
    }

    /// Marks the intent committed and publishes every participant.
    ///
    /// The intent write is the commit point: once it lands, recovery keeps
    /// every participant's version whether or not this call gets to publish
    /// them.
    pub fn commit(
        &self,
    ) -> Result<Vec<std::sync::Arc<crate::transaction_log::TransactionLog>>, ZyronError> {
        if !self.prepared {
            return Err(ZyronError::Internal(
                "cross-table commit without a prepared intent".into(),
            ));
        }
        let path = txn_intent_file(&self.data_dir, self.seq);
        write_intent(&path, self.seq, STATE_COMMITTED, self.timestamp_us)?;

        // Versions registered as pending under this id publish first, which
        // is how a statement executed through the engine joins. Ascending,
        // so a reader never sees a later version before the one it was
        // built on
        let mut published = crate::transaction_log::publish_txn(&self.data_dir, self.txn_id())?;
        // Then anything a direct caller recorded on this transaction and the
        // registry did not already cover
        let mut ordered: Vec<&(std::sync::Arc<crate::transaction_log::TransactionLog>, u64)> =
            self.participants.iter().collect();
        ordered.sort_by_key(|(_, version)| *version);
        for (log, version) in ordered {
            if published.iter().any(|p| std::sync::Arc::ptr_eq(p, log)) {
                continue;
            }
            log.publish(*version)?;
            published.push(std::sync::Arc::clone(log));
        }
        // Removing the intent is cleanup. A crash before it leaves a
        // COMMITTED intent whose participants are already durable
        fs::remove_file(&path)?;
        Ok(published)
    }

    /// Discards every participant's version, then the intent.
    ///
    /// Versions go first: an intent removed while its versions survive would
    /// read as a finished transaction and make them visible.
    pub fn abort(&self) -> Vec<std::sync::Arc<crate::transaction_log::TransactionLog>> {
        // Anything registered as pending under this id goes first
        let mut touched = crate::transaction_log::abandon_txn(&self.data_dir, self.txn_id());
        // Newest first, so a discarded version is never the base of one that
        // still exists
        let mut ordered: Vec<&(std::sync::Arc<crate::transaction_log::TransactionLog>, u64)> =
            self.participants.iter().collect();
        ordered.sort_by_key(|(_, version)| std::cmp::Reverse(*version));
        for (log, version) in ordered {
            if touched.iter().any(|p| std::sync::Arc::ptr_eq(p, log)) {
                continue;
            }
            if let Err(e) = log.abandon(*version) {
                tracing::warn!(
                    version,
                    error = %e,
                    "cross-table abort could not discard a version, recovery will"
                );
            }
            touched.push(std::sync::Arc::clone(log));
        }
        // The versions are gone, so the intent can go: an intent removed
        // while its versions survive would read as a finished transaction
        if self.prepared {
            let _ = fs::remove_file(txn_intent_file(&self.data_dir, self.seq));
        }
        touched
    }
}

fn write_intent(path: &Path, seq: u64, state: u8, timestamp_us: i64) -> Result<(), ZyronError> {
    let bytes = encode_intent(seq, state, timestamp_us);
    let mut file = fs::OpenOptions::new()
        .write(true)
        .create(true)
        .truncate(true)
        .open(path)?;
    file.write_all(&bytes)?;
    file.sync_all()?;
    Ok(())
}

/// Hands out the next sequence, primed from the highest intent on disk so a
/// restart never reuses one a surviving intent still owns.
fn allocate_seq(data_dir: &Path) -> Result<u64, ZyronError> {
    let mut next = NEXT_SEQ.load(Ordering::Acquire);
    if next == 0 {
        let mut highest = 0u64;
        let dir = intent_dir(data_dir);
        if dir.exists() {
            for dirent in fs::read_dir(&dir)? {
                let dirent = dirent?;
                let name = dirent.file_name();
                let Some(name) = name.to_str() else { continue };
                if let Some(seq) = name
                    .strip_suffix(".intent")
                    .and_then(|s| s.parse::<u64>().ok())
                {
                    highest = highest.max(seq);
                }
            }
        }
        // Racing callers converge on the same floor, then each takes its own
        // sequence from the counter below
        let _ = NEXT_SEQ.compare_exchange(
            0,
            highest + 1,
            Ordering::AcqRel,
            Ordering::Acquire,
        );
        next = NEXT_SEQ.load(Ordering::Acquire);
    }
    Ok(NEXT_SEQ.fetch_add(1, Ordering::AcqRel).max(next))
}

/// What startup recovery did with the leftover intents.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct IntentRecovery {
    /// Transactions whose commit point had landed, kept
    pub committed: Vec<u64>,
    /// Transactions that never reached their commit point, discarded
    pub discarded: Vec<u64>,
}

/// Clears the intent files a crash left behind.
///
/// Runs before any lake log opens, so the log's own recovery sees the final
/// answer for every intent-flagged transaction. A COMMITTED intent is
/// removed after its participants are already durable, a PREPARING one is
/// removed and its participants' versions are discarded by each log's own
/// recovery, which reads the same answer through `IntentAware`.
pub fn recover_intents(data_dir: &Path) -> Result<IntentRecovery, ZyronError> {
    let dir = intent_dir(data_dir);
    let mut report = IntentRecovery::default();
    if !dir.exists() {
        return Ok(report);
    }
    let mut entries: Vec<(u64, PathBuf, IntentState)> = Vec::new();
    for dirent in fs::read_dir(&dir)? {
        let dirent = dirent?;
        let path = dirent.path();
        let bytes = match fs::read(&path) {
            Ok(b) => b,
            Err(_) => continue,
        };
        match decode_intent(&bytes, &path.to_string_lossy()) {
            Ok((seq, state)) => entries.push((seq, path, state)),
            // An unreadable intent cannot say its transaction committed, so
            // it is treated as never having reached the commit point
            Err(e) => {
                tracing::warn!(path = %path.display(), error = %e, "discarding unreadable commit intent");
                fs::remove_file(&path)?;
            }
        }
    }
    entries.sort_by_key(|(seq, _, _)| *seq);

    // A PREPARING intent must outlive this pass: each log's recovery asks
    // whether its transaction committed, and the answer has to stay "no"
    // until those version files are gone. The logs run after this returns,
    // so PREPARING intents are removed on the second pass below only once
    // their versions cannot be reached
    for (seq, _, state) in &entries {
        match state {
            IntentState::Committed => report.committed.push(*seq),
            IntentState::Preparing => report.discarded.push(*seq),
        }
    }
    Ok(report)
}

/// Removes the intent files recovery already accounted for.
///
/// Called after every lake log has opened, so a PREPARING intent's version
/// files are gone by the time its intent is. Removing it earlier would let
/// a log that opens later read "absent means committed" and resurrect them.
pub fn clear_recovered_intents(
    data_dir: &Path,
    recovery: &IntentRecovery,
) -> Result<(), ZyronError> {
    for seq in recovery.committed.iter().chain(recovery.discarded.iter()) {
        let path = txn_intent_file(data_dir, *seq);
        match fs::remove_file(&path) {
            Ok(()) => {}
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => {}
            Err(e) => {
                return Err(ZyronError::IoError(format!(
                    "cannot clear commit intent {}: {}",
                    path.display(),
                    e
                )));
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::operations::append_rows;
    use crate::paths::LakePaths;
    use crate::schema::{LakeColumn, LakeSchema};
    use crate::transaction_log::{AllCommitted, CommitAttempt, OperationKind, TransactionLog};
    use crate::writer::ColumnData;
    use std::collections::BTreeMap;
    use zyron_common::TypeId;

    fn schema() -> LakeSchema {
        LakeSchema::new(
            1,
            vec![LakeColumn {
                id: 0,
                name: "id".into(),
                type_id: TypeId::Int64,
                nullable: false,
                ts_precision: None,
                tz_offset_secs: None,
                max_length: None,
                default_expr: None,
            }],
        )
        .expect("schema")
    }

    fn attempt(db_txn_id: u64) -> CommitAttempt<'static> {
        CommitAttempt {
            operation: OperationKind::Append,
            db_txn_id,
            commit_lsn: 1,
            timestamp_us: 1_000,
            read_predicate: None,
            audit: None,
        }
    }

    fn rows(ids: &[i64]) -> Vec<ColumnData> {
        vec![ColumnData {
            column_id: 0,
            cells: ids.iter().map(|v| Some(v.to_le_bytes().to_vec())).collect(),
        }]
    }

    fn new_table(data_dir: &Path, table_id: u32) -> TransactionLog {
        TransactionLog::create(
            LakePaths::new(data_dir, table_id),
            CommitAttempt {
                operation: OperationKind::SchemaChange,
                ..attempt(0)
            },
            &schema(),
            None,
            &BTreeMap::new(),
        )
        .expect("create")
    }

    fn live_rows(log: &TransactionLog) -> u64 {
        log.latest_manifest()
            .expect("manifest")
            .entries
            .iter()
            .map(|e| e.row_count)
            .sum()
    }

    #[test]
    fn test_both_tables_become_visible_at_the_intent_commit() {
        let dir = tempfile::TempDir::new().expect("temp dir");
        let a = new_table(dir.path(), 21);
        let b = new_table(dir.path(), 22);

        let mut txn = CrossTableTxn::begin(dir.path(), 5_000).expect("begin");
        txn.prepare().expect("prepare");
        let a = std::sync::Arc::new(a);
        let b = std::sync::Arc::new(b);
        let va = append_rows(&a, attempt(txn.txn_id()), 21, &rows(&[1, 2])).expect("a");
        let vb = append_rows(&b, attempt(txn.txn_id()), 22, &rows(&[3])).expect("b");
        txn.record(std::sync::Arc::clone(&a), va.version);
        txn.record(std::sync::Arc::clone(&b), vb.version);

        // Written, invisible: neither table's head moved
        assert_eq!(live_rows(&a), 0);
        assert_eq!(live_rows(&b), 0);

        txn.commit().expect("commit");
        assert_eq!(live_rows(&a), 2);
        assert_eq!(live_rows(&b), 1);
        assert!(!txn_intent_file(dir.path(), txn.sequence()).exists());
    }

    #[test]
    fn test_crosstable_crash_between_intent_and_second_table_discards_both() {
        let dir = tempfile::TempDir::new().expect("temp dir");
        let a = new_table(dir.path(), 23);
        let b = new_table(dir.path(), 24);
        let a_paths = a.paths().clone();
        let b_paths = b.paths().clone();

        let mut txn = CrossTableTxn::begin(dir.path(), 5_000).expect("begin");
        txn.prepare().expect("prepare");
        append_rows(&a, attempt(txn.txn_id()), 23, &rows(&[1, 2])).expect("a");
        // The process dies here: the first table has a version file, the
        // second never got one, and the intent is still PREPARING
        drop(a);
        drop(b);

        let intent = txn_intent_file(dir.path(), txn.sequence());
        assert!(intent.exists(), "the intent survives the crash");

        // Recovery reads the intent, then each log opens through it
        let recovery = recover_intents(dir.path()).expect("recover");
        assert_eq!(recovery.discarded, vec![txn.sequence()]);
        assert!(recovery.committed.is_empty());

        let status = IntentAware::new(&AllCommitted, dir.path());
        let a = TransactionLog::open(a_paths, &status).expect("reopen a");
        let b = TransactionLog::open(b_paths, &status).expect("reopen b");
        assert_eq!(live_rows(&a), 0, "the first table's write is discarded");
        assert_eq!(live_rows(&b), 0, "the second never happened");

        clear_recovered_intents(dir.path(), &recovery).expect("clear");
        assert!(!intent.exists());
    }

    #[test]
    fn test_a_crash_after_the_commit_point_keeps_every_participant() {
        let dir = tempfile::TempDir::new().expect("temp dir");
        let a = new_table(dir.path(), 25);
        let b = new_table(dir.path(), 26);
        let a_paths = a.paths().clone();
        let b_paths = b.paths().clone();

        let mut txn = CrossTableTxn::begin(dir.path(), 5_000).expect("begin");
        txn.prepare().expect("prepare");
        append_rows(&a, attempt(txn.txn_id()), 25, &rows(&[1, 2])).expect("a");
        append_rows(&b, attempt(txn.txn_id()), 26, &rows(&[3])).expect("b");
        // The commit point lands, then the process dies before publishing
        write_intent(
            &txn_intent_file(dir.path(), txn.sequence()),
            txn.sequence(),
            STATE_COMMITTED,
            5_000,
        )
        .expect("commit point");
        drop(a);
        drop(b);

        let recovery = recover_intents(dir.path()).expect("recover");
        assert_eq!(recovery.committed, vec![txn.sequence()]);

        let status = IntentAware::new(&AllCommitted, dir.path());
        let a = TransactionLog::open(a_paths, &status).expect("reopen a");
        let b = TransactionLog::open(b_paths, &status).expect("reopen b");
        assert_eq!(live_rows(&a), 2, "the commit point had landed");
        assert_eq!(live_rows(&b), 1);
        clear_recovered_intents(dir.path(), &recovery).expect("clear");
    }

    #[test]
    fn test_abort_removes_the_versions_before_the_intent() {
        let dir = tempfile::TempDir::new().expect("temp dir");
        let a = new_table(dir.path(), 27);
        let a_paths = a.paths().clone();

        let mut txn = CrossTableTxn::begin(dir.path(), 5_000).expect("begin");
        txn.prepare().expect("prepare");
        let a = std::sync::Arc::new(a);
        let out = append_rows(&a, attempt(txn.txn_id()), 27, &rows(&[1])).expect("a");
        txn.record(std::sync::Arc::clone(&a), out.version);
        assert!(a_paths.version_file(2).exists());

        txn.abort();
        assert!(
            !a_paths.version_file(2).exists(),
            "the version file goes before the intent does"
        );
        assert!(!txn_intent_file(dir.path(), txn.sequence()).exists());
        assert_eq!(live_rows(&a), 0);
    }

    #[test]
    fn test_intent_ids_never_collide_with_database_transaction_ids() {
        let dir = tempfile::TempDir::new().expect("temp dir");
        let mut txn = CrossTableTxn::begin(dir.path(), 1).expect("begin");
        assert!(txn.txn_id() & INTENT_TXN_FLAG != 0);
        assert_eq!(txn.txn_id() & !INTENT_TXN_FLAG, txn.sequence());
        txn.prepare().expect("prepare");

        // A plain database id routes to the inner oracle, an intent id to
        // the intent file
        struct NeverCommitted;
        impl CommitStatus for NeverCommitted {
            fn is_committed(&self, _: u64) -> bool {
                false
            }
        }
        let status = IntentAware::new(&NeverCommitted, dir.path());
        assert!(!status.is_committed(7), "a database id asks the database");
        assert!(!status.is_committed(txn.txn_id()), "still preparing");
        write_intent(
            &txn_intent_file(dir.path(), txn.sequence()),
            txn.sequence(),
            STATE_COMMITTED,
            1,
        )
        .expect("commit point");
        assert!(status.is_committed(txn.txn_id()));
        // A finished transaction leaves no intent, and its versions counted
        fs::remove_file(txn_intent_file(dir.path(), txn.sequence())).expect("remove");
        assert!(status.is_committed(txn.txn_id()));
    }
}
