//! The commit chain of a verifiable table, its anchors and what a walk
//! reports.
//!
//! These tests work on the chain itself rather than through SQL. What they
//! cover is the mechanism: that an entry links to the one before it, that a
//! chain reopens at the head it left, that an edited row, a removed row, a
//! removed commit and a truncated chain are each reported as what they are,
//! that a genesis set reads back the same whatever order it is stored in,
//! and that a chain rewritten from end to end still contradicts an anchor
//! taken over it.
//!
//! Run: cargo test -p zyron-lifecycle --test verify_test -- --nocapture

use std::collections::HashMap;

use zyron_common::Result;
use zyron_lifecycle::verify::anchor::{Anchor, AnchorStore, ExportedAnchor};
use zyron_lifecycle::verify::{
    self, ChainHash, ChainRegistry, CommitChain, CommitFields, CommitHash, CommitRows, RowHashes,
    RowMode, RowRequest, RowsHasher, SetHasher, VerifyFailure,
};

/// The scheme every chain in these tests links with
const ALGORITHM: u16 = verify::DEFAULT_CHAIN_ALGORITHM;

fn fields(txn_id: u64, rows_hash: ChainHash, row_count: u64, commit_ts: i64) -> CommitFields {
    CommitFields {
        txn_id,
        rows_hash,
        row_count,
        commit_ts,
        algorithm_id: ALGORITHM,
        genesis: false,
    }
}

/// A table whose rows are held in memory, so a test can edit one of them,
/// remove one, or remove a whole commit and see what a walk makes of it.
///
/// Rows are kept in the order they were written, which is the order the
/// heap stores them and the order a verification reads them back in. Every
/// row carries the stamp of the transaction that wrote it, the way a heap
/// row does, so a genesis set is the rows stamped below a fence
#[derive(Default)]
struct Rows {
    /// Every row in stored order, as (transaction, schema epoch, bytes)
    stored: Vec<(u64, u16, Vec<u8>)>,
    /// Where the genesis set's sorted runs would go
    spill: std::path::PathBuf,
}

impl Rows {
    fn in_dir(dir: &std::path::Path) -> Self {
        Self {
            stored: Vec::new(),
            spill: dir.join("spill"),
        }
    }

    fn write(&mut self, txn_id: u64, epoch: u16, rows: &[&[u8]]) -> (ChainHash, u64) {
        for row in rows {
            self.stored.push((txn_id, epoch, row.to_vec()));
        }
        self.rehash(txn_id)
    }

    /// Rehashes one commit's rows as they stand now, in stored order
    fn rehash(&self, txn_id: u64) -> (ChainHash, u64) {
        let mut hasher = RowsHasher::new();
        for (stamp, epoch, row) in &self.stored {
            if *stamp == txn_id {
                hasher.row(*epoch, row);
            }
        }
        let count = hasher.rows();
        (hasher.finish(), count)
    }

    /// The row at `index` among those one transaction wrote
    fn position(&self, txn_id: u64, index: usize) -> Option<usize> {
        self.stored
            .iter()
            .enumerate()
            .filter(|(_, (stamp, _, _))| *stamp == txn_id)
            .nth(index)
            .map(|(at, _)| at)
    }

    /// Changes one stored byte of one row, the way an edit outside SQL does
    fn edit(&mut self, txn_id: u64, row: usize) {
        if let Some(at) = self.position(txn_id, row)
            && let Some(first) = self.stored[at].2.first_mut()
        {
            *first = first.wrapping_add(1);
        }
    }

    /// Removes one row, the way a delete outside SQL does
    fn remove_row(&mut self, txn_id: u64, row: usize) {
        if let Some(at) = self.position(txn_id, row) {
            self.stored.remove(at);
        }
    }

    /// Removes every row one commit wrote
    fn remove_commit(&mut self, txn_id: u64) {
        self.stored.retain(|(stamp, _, _)| *stamp != txn_id);
    }
}

impl CommitRows for Rows {
    fn hash_commits(&self, request: &RowRequest) -> Result<RowHashes> {
        let mut out = RowHashes::default();
        for txn_id in &request.txn_ids {
            out.by_txn.insert(*txn_id, self.rehash(*txn_id));
        }
        if let Some(fence) = request.genesis_fence {
            let mut set = SetHasher::new(&self.spill);
            for (stamp, epoch, row) in &self.stored {
                if *stamp < fence {
                    set.row(*epoch, row)?;
                }
            }
            let rows = set.rows();
            out.genesis = Some((set.finish()?, rows));
        }
        Ok(out)
    }
}

fn never_cancelled() -> impl Fn() -> bool {
    || false
}

/// A chain of `commits` transactions, each writing `rows_per` rows, with the
/// rows they wrote beside it
fn build(dir: &std::path::Path, table_id: u32, commits: u64, rows_per: u64) -> (CommitChain, Rows) {
    let chain = CommitChain::open(dir, table_id).expect("opens");
    let mut rows = Rows::in_dir(dir);
    for commit in 1..=commits {
        let bodies: Vec<Vec<u8>> = (0..rows_per)
            .map(|row| format!("t{commit}-r{row}").into_bytes())
            .collect();
        let refs: Vec<&[u8]> = bodies.iter().map(|b| b.as_slice()).collect();
        let (rows_hash, count) = rows.write(commit, 1, &refs);
        chain
            .append(
                fields(commit, rows_hash, count, 10_000 + commit as i64),
                || 1_000 + commit,
            )
            .expect("appends");
        chain.publish(commit).expect("publishes");
    }
    chain.sync().expect("flushes");
    (chain, rows)
}

fn walk_all(chain: &CommitChain, rows: &Rows, anchors: &[Anchor]) -> verify::VerifyOutcome {
    let head = chain.head();
    verify::walk(
        chain,
        rows,
        0,
        head.commits.saturating_sub(1),
        RowMode::All,
        0,
        anchors,
        &never_cancelled(),
    )
    .expect("walks")
}

/// An untouched chain verifies, and the result states the mode it ran in and
/// how much it covered
#[test]
fn an_untouched_chain_is_intact_and_states_what_it_checked() {
    let dir = tempfile::TempDir::new().expect("temp dir");
    let (chain, rows) = build(dir.path(), 1, 5, 4);
    let outcome = walk_all(&chain, &rows, &[]);
    assert!(outcome.intact, "{:?}", outcome.failure);
    assert_eq!(outcome.commits_checked, 5);
    assert_eq!(outcome.rows_checked, 20);
    assert_eq!(outcome.commits_rehashed, 5);
    assert_eq!(outcome.mode, RowMode::All);
    assert!(outcome.summary().contains("every one of their 20 row(s)"));
}

/// A sampled pass walks the whole chain and reads back only the commits it
/// sampled, and says so
#[test]
fn a_sampled_pass_says_how_much_it_read() {
    let dir = tempfile::TempDir::new().expect("temp dir");
    let (chain, rows) = build(dir.path(), 2, 10, 3);
    let outcome = verify::walk(
        &chain,
        &rows,
        0,
        9,
        RowMode::Sampled,
        3,
        &[],
        &never_cancelled(),
    )
    .expect("walks");
    assert!(outcome.intact);
    assert_eq!(outcome.commits_checked, 10, "every entry's link is checked");
    assert_eq!(outcome.commits_rehashed, 3, "three commits were sampled");
    assert_eq!(outcome.rows_checked, 9);
    let summary = outcome.summary();
    assert!(summary.contains("3 of them sampled"), "{summary}");
    assert!(
        summary.contains("the rows of the rest not read"),
        "{summary}"
    );
}

/// Editing one row's stored bytes is reported as a row content mismatch,
/// naming the first commit that no longer matches
#[test]
fn an_edited_row_is_a_row_content_mismatch() {
    let dir = tempfile::TempDir::new().expect("temp dir");
    let (chain, mut rows) = build(dir.path(), 3, 6, 4);
    rows.edit(4, 2);
    let outcome = walk_all(&chain, &rows, &[]);
    assert!(!outcome.intact);
    match outcome.failure.expect("a failure") {
        VerifyFailure::RowContentMismatch {
            commit_version,
            sequence,
            ..
        } => {
            assert_eq!(
                commit_version, 1_004,
                "the fourth commit is the first bad one"
            );
            assert_eq!(sequence, 3);
        }
        other => panic!("expected a row content mismatch, got {other:?}"),
    }
}

/// Removing one row of a commit is reported as a row count mismatch on that
/// commit, which names what the entry recorded and what is there now
#[test]
fn a_removed_row_is_a_row_count_mismatch() {
    let dir = tempfile::TempDir::new().expect("temp dir");
    let (chain, mut rows) = build(dir.path(), 4, 5, 4);
    rows.remove_row(2, 1);
    let outcome = walk_all(&chain, &rows, &[]);
    assert!(!outcome.intact);
    match outcome.failure.expect("a failure") {
        VerifyFailure::RowCountMismatch {
            commit_version,
            expected,
            found,
            ..
        } => {
            assert_eq!(commit_version, 1_002);
            assert_eq!(expected, 4);
            assert_eq!(found, 3);
        }
        other => panic!("expected a row count mismatch, got {other:?}"),
    }
}

/// Removing every row a commit wrote is reported on that commit rather than
/// passing as a chain that simply covers fewer rows
#[test]
fn a_removed_commit_is_reported_on_that_commit() {
    let dir = tempfile::TempDir::new().expect("temp dir");
    let (chain, mut rows) = build(dir.path(), 5, 5, 3);
    rows.remove_commit(3);
    let outcome = walk_all(&chain, &rows, &[]);
    assert!(!outcome.intact);
    let failure = outcome.failure.expect("a failure");
    assert_eq!(failure.kind(), "row_count_mismatch");
    assert_eq!(failure.commit_version(), 1_003);
    assert!(failure.to_string().contains("0 row(s)"), "{failure}");
}

/// A chain whose last entries were removed is a shorter chain whose every
/// link still holds. Only an anchor naming a head it no longer reaches
/// contradicts it
#[test]
fn a_truncated_chain_contradicts_its_anchor() {
    let dir = tempfile::TempDir::new().expect("temp dir");
    let (chain, rows) = build(dir.path(), 6, 8, 2);
    let head = chain.head();
    let anchor = Anchor {
        table_id: 6,
        table_name: "ledger".to_string(),
        sequence: head.commits - 1,
        commit_version: head.head_version,
        head_hash: head.head_hash,
        taken_at: 90_000,
    };

    // Removing the last three entries leaves a chain that walks clean
    let shorter = truncate_to(chain.path(), 5);
    let without_anchor = verify::walk(
        &shorter,
        &rows,
        0,
        4,
        RowMode::All,
        0,
        &[],
        &never_cancelled(),
    )
    .expect("walks");
    assert!(
        without_anchor.intact,
        "a chain read on its own cannot see that it was cut short"
    );

    let against_anchor = verify::walk(
        &shorter,
        &rows,
        0,
        4,
        RowMode::All,
        0,
        std::slice::from_ref(&anchor),
        &never_cancelled(),
    )
    .expect("walks");
    assert!(!against_anchor.intact);
    match against_anchor.failure.expect("a failure") {
        VerifyFailure::AnchorContradiction {
            sequence,
            anchored_head,
            found_head,
            anchored_at,
            ..
        } => {
            assert_eq!(sequence, 7);
            assert_eq!(anchored_head, anchor.head_hash);
            assert_eq!(found_head, None, "the chain no longer reaches that commit");
            assert_eq!(anchored_at, 90_000);
        }
        other => panic!("expected an anchor contradiction, got {other:?}"),
    }
    assert_eq!(against_anchor.anchors_checked, 1);
}

/// The case the phase exists for. A chain rewritten from end to end after an
/// edit walks clean, and still contradicts the anchor taken over what it
/// used to be
#[test]
fn a_forged_chain_still_contradicts_an_exported_anchor() {
    let dir = tempfile::TempDir::new().expect("temp dir");
    let (chain, mut rows) = build(dir.path(), 7, 6, 3);
    let head = chain.head();
    let exported = ExportedAnchor {
        anchor: Anchor {
            table_id: 7,
            table_name: "ledger".to_string(),
            sequence: head.commits - 1,
            commit_version: head.head_version,
            head_hash: head.head_hash,
            taken_at: 55_000,
        },
        scheme: "Ed25519".to_string(),
        signature: vec![1u8; 64],
    };
    let held = exported.anchor.clone();
    drop(chain);

    // The edit, and then every hash after it recomputed, which is what a
    // forger with write access to the chain file can do
    rows.edit(2, 0);
    let forged_dir = tempfile::TempDir::new().expect("temp dir");
    let forged = CommitChain::open(forged_dir.path(), 7).expect("opens");
    for commit in 1..=6u64 {
        let (rows_hash, count) = rows.rehash(commit);
        forged
            .append(
                fields(commit, rows_hash, count, 10_000 + commit as i64),
                || 1_000 + commit,
            )
            .expect("appends");
        forged.publish(commit).expect("publishes");
    }
    forged.sync().expect("flushes");

    // Every link holds and every row rehashes, because the forger redid both
    let walked = verify::walk(
        &forged,
        &rows,
        0,
        5,
        RowMode::All,
        0,
        &[],
        &never_cancelled(),
    )
    .expect("walks");
    assert!(
        walked.intact,
        "a chain relinked over the edited rows is self-consistent"
    );

    // The anchor names a head the forged chain does not have
    let against = verify::walk(
        &forged,
        &rows,
        0,
        5,
        RowMode::All,
        0,
        std::slice::from_ref(&held),
        &never_cancelled(),
    )
    .expect("walks");
    assert!(!against.intact, "the anchor is what catches the forgery");
    let failure = against.failure.expect("a failure");
    assert_eq!(failure.kind(), "anchor_contradiction");
    assert!(!exported.agrees_with(forged.head().commits - 1, &forged.head().head_hash));

    // The same edit under a chain with no anchor over it is not caught,
    // which is the reason an anchor exists
    assert!(walked.intact);
}

/// A schema change between two commits leaves both verifiable, because the
/// epoch a row was written under is part of what its commit hashed
#[test]
fn a_schema_change_between_commits_leaves_both_verifiable() {
    let dir = tempfile::TempDir::new().expect("temp dir");
    let chain = CommitChain::open(dir.path(), 8).expect("opens");
    let mut rows = Rows::in_dir(dir.path());

    // One commit under the first epoch, then a column is added and the next
    // commit's rows are written under the second
    let before: Vec<&[u8]> = vec![b"a", b"b"];
    let (hash_before, count_before) = rows.write(1, 1, &before);
    chain
        .append(fields(1, hash_before, count_before, 100), || 10)
        .expect("appends");
    chain.publish(1).expect("publishes");
    let after: Vec<&[u8]> = vec![b"a\0x", b"b\0y"];
    let (hash_after, count_after) = rows.write(2, 2, &after);
    chain
        .append(fields(2, hash_after, count_after, 200), || 20)
        .expect("appends");
    chain.publish(2).expect("publishes");
    chain.sync().expect("flushes");

    let outcome = walk_all(&chain, &rows, &[]);
    assert!(outcome.intact, "{:?}", outcome.failure);
    assert_eq!(outcome.commits_checked, 2);
    assert_eq!(outcome.rows_checked, 4);
}

/// The rows a table held when it became verifiable are covered by a
/// genesis entry over them as a set. The set reads back the same whatever
/// order the rows are stored in, a full pass reads it, a sampled pass says
/// it did not, and an edit inside it is caught by the full pass
#[test]
fn a_genesis_set_covers_what_the_table_held_in_any_order() {
    let dir = tempfile::TempDir::new().expect("temp dir");
    let chain = CommitChain::open(dir.path(), 12).expect("opens");
    let mut rows = Rows::in_dir(dir.path());
    // Three transactions wrote before the table was chained, interleaved
    let early: Vec<(u64, &[u8])> = vec![(3, b"c1"), (5, b"e1"), (3, b"c2"), (4, b"d1"), (5, b"e2")];
    for (txn, row) in &early {
        rows.write(*txn, 1, &[row]);
    }
    let fence = 10u64;
    // The genesis set as the table would hash it, over sorted digests
    let mut set = SetHasher::new(dir.path().join("genesis"));
    for (_, row) in &early {
        set.row(1, row).expect("takes");
    }
    let count = set.rows();
    let genesis_hash = set.finish().expect("finishes");
    let genesis = chain
        .append(
            CommitFields {
                genesis: true,
                ..fields(fence, genesis_hash, count, 1)
            },
            || 1,
        )
        .expect("appends");
    chain.publish(fence).expect("publishes");
    assert!(genesis.genesis);
    // Commits from the fence on are chained one by one
    for txn in fence..fence + 4 {
        let body = format!("late-{txn}");
        let (hash, count) = rows.write(txn, 1, &[body.as_bytes()]);
        chain
            .append(fields(txn, hash, count, txn as i64), || txn)
            .expect("appends");
        chain.publish(txn).expect("publishes");
    }
    chain.sync().expect("flushes");

    let full = walk_all(&chain, &rows, &[]);
    assert!(full.intact, "{:?}", full.failure);
    assert_eq!(full.commits_checked, 5);
    assert_eq!(full.commits_rehashed, 5);
    assert_eq!(full.rows_checked, 9, "five in the set and four after it");
    assert!(!full.genesis_not_read);

    // The same rows stored in another order are the same set
    let mut shuffled = Rows::in_dir(dir.path());
    for (txn, _, row) in rows.stored.iter().rev() {
        shuffled.stored.push((*txn, 1, row.clone()));
    }
    let head = chain.head();
    let request = RowRequest {
        txn_ids: Vec::new(),
        genesis_fence: Some(fence),
    };
    assert_eq!(
        shuffled.hash_commits(&request).expect("hashes").genesis,
        Some((genesis_hash, count))
    );

    // A sampled pass checks the genesis entry's link and does not read the
    // set back, and says so
    let sampled = verify::walk(
        &chain,
        &rows,
        0,
        head.commits - 1,
        RowMode::Sampled,
        2,
        &[],
        &never_cancelled(),
    )
    .expect("walks");
    assert!(sampled.intact);
    assert!(sampled.genesis_not_read);
    assert_eq!(sampled.commits_checked, 5);
    assert_eq!(sampled.commits_rehashed, 1, "the genesis is passed over");
    let summary = sampled.summary();
    assert!(summary.contains("the genesis set not read"), "{summary}");

    // An edit inside the set is a content mismatch on the genesis entry
    rows.edit(4, 0);
    let edited = walk_all(&chain, &rows, &[]);
    assert!(!edited.intact);
    let failure = edited.failure.expect("a failure");
    assert_eq!(failure.kind(), "row_content_mismatch");
    assert_eq!(failure.sequence(), 0);
}

/// Sixteen writers hash their own rows and meet only at the link, and the
/// chain they produce walks clean
#[test]
fn sixteen_writers_produce_one_walkable_chain() {
    let dir = tempfile::TempDir::new().expect("temp dir");
    let chain = std::sync::Arc::new(CommitChain::open(dir.path(), 9).expect("opens"));
    let writers = 16u64;
    let per_writer = 25u64;

    let hashes: Vec<(u64, ChainHash, u64)> = std::thread::scope(|scope| {
        let handles: Vec<_> = (0..writers)
            .map(|writer| {
                scope.spawn(move || {
                    let mut mine = Vec::with_capacity(per_writer as usize);
                    for run in 0..per_writer {
                        let txn_id = writer * per_writer + run + 1;
                        let mut hasher = RowsHasher::new();
                        for row in 0..8u64 {
                            hasher.row(1, format!("w{writer}-t{txn_id}-r{row}").as_bytes());
                        }
                        let count = hasher.rows();
                        mine.push((txn_id, hasher.finish(), count));
                    }
                    mine
                })
            })
            .collect();
        handles
            .into_iter()
            .flat_map(|handle| handle.join().expect("the writer finished"))
            .collect()
    });

    // The link is the serial half, taken one at a time the way a commit
    // takes it
    for (txn_id, rows_hash, count) in &hashes {
        chain
            .append(fields(*txn_id, *rows_hash, *count, *txn_id as i64), || {
                *txn_id
            })
            .expect("appends");
        chain.publish(*txn_id).expect("publishes");
    }
    chain.sync().expect("flushes");

    let head = chain.head();
    assert_eq!(head.commits, writers * per_writer);
    let entries = chain.read_range(0, head.commits - 1).expect("reads");
    let mut prev = verify::NO_PREVIOUS;
    for entry in &entries {
        assert_eq!(entry.prev_hash, prev, "entry {} links", entry.sequence);
        assert_eq!(entry.compute_entry_hash(), entry.entry_hash);
        prev = entry.entry_hash;
    }
    assert_eq!(prev, head.head_hash);
    assert_eq!(head.bytes(), head.commits * verify::CHAIN_RECORD_LEN as u64);
}

/// A chain reopened over the same directory stands where it stood, and the
/// entries a restart found in the log are put back once each
#[test]
fn logged_entries_are_put_back_once() {
    let dir = tempfile::TempDir::new().expect("temp dir");
    let registry = ChainRegistry::open(dir.path()).expect("opens");
    let chain = registry.chain(10).expect("opens");
    let mut hasher = RowsHasher::new();
    hasher.row(1, b"only");
    let rows_hash = hasher.finish();
    let first = chain
        .append(fields(1, rows_hash, 1, 1), || 1)
        .expect("appends");
    chain.publish(1).expect("publishes");
    chain.sync().expect("flushes");

    // The entry the log holds for the commit after it
    let next = CommitHash::link(10, 1, first.entry_hash, 2, fields(2, rows_hash, 1, 2));
    let logged = verify::LoggedEntry::decode(10, 1, &next.encode()).expect("decodes");
    assert_eq!(
        verify::restore_logged_entries(&registry, &[logged]).expect("restores"),
        1
    );
    assert_eq!(chain.head().commits, 2);
    // Replaying the same log again changes nothing
    assert_eq!(
        verify::restore_logged_entries(&registry, &[logged]).expect("restores"),
        0
    );
    assert_eq!(chain.head().commits, 2);
}

/// An anchor store reopened states what it held, which is what lets an
/// anchor outlive the process that took it
#[test]
fn anchors_survive_a_restart() {
    let dir = tempfile::TempDir::new().expect("temp dir");
    {
        let store = AnchorStore::open(dir.path()).expect("opens");
        store
            .anchor(Anchor {
                table_id: 11,
                table_name: "ledger".to_string(),
                sequence: 3,
                commit_version: 99,
                head_hash: [4u8; 32],
                taken_at: 1234,
            })
            .expect("anchors");
    }
    let again = AnchorStore::open(dir.path()).expect("reopens");
    let held = again.for_table(11).expect("held");
    assert_eq!(held.sequence, 3);
    assert_eq!(held.head_hash, [4u8; 32]);
    assert_eq!(held.commit_version, 99);
}

/// The anchors of one pass reach the store in one write and every one of
/// them is there afterwards
#[test]
fn a_pass_anchors_every_table_at_once() {
    let dir = tempfile::TempDir::new().expect("temp dir");
    let store = AnchorStore::open(dir.path()).expect("opens");
    let anchors: Vec<Anchor> = (1..=5u32)
        .map(|table| Anchor {
            table_id: table,
            table_name: format!("t{table}"),
            sequence: table as u64,
            commit_version: 100 + table as u64,
            head_hash: [table as u8; 32],
            taken_at: 7,
        })
        .collect();
    store.anchor_all(anchors.clone()).expect("anchors");
    drop(store);
    let again = AnchorStore::open(dir.path()).expect("reopens");
    assert_eq!(again.all(), anchors);
    let held: HashMap<u32, Anchor> = again.all().into_iter().map(|a| (a.table_id, a)).collect();
    assert_eq!(held.len(), 5);
}

/// Cuts a chain file back to `commits` entries, the way a truncation
/// outside the database does, and opens what is left
fn truncate_to(path: &std::path::Path, commits: u64) -> CommitChain {
    let header = 20u64;
    let length = header + commits * verify::CHAIN_RECORD_LEN as u64;
    let file = std::fs::OpenOptions::new()
        .write(true)
        .open(path)
        .expect("opens the chain file");
    file.set_len(length).expect("cuts it back");
    drop(file);
    let table_id: u32 = path
        .file_stem()
        .and_then(|stem| stem.to_str())
        .and_then(|stem| stem.parse().ok())
        .expect("the chain file is named for its table");
    let dir = path
        .parent()
        .and_then(|verify| verify.parent())
        .expect("the chain sits under the data directory");
    CommitChain::open(dir, table_id).expect("reopens")
}
