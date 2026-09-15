//! Anchors of a commit chain's head.
//!
//! A chain walked on its own cannot see a truncation. Deleting the last few
//! entries and the rows they covered leaves a shorter chain whose every link
//! still holds. An anchor is what closes that: it records the head the chain
//! stood at, at a version, at an instant, and a chain that no longer reaches
//! that version, or that stands at a different head there, contradicts it.
//!
//! Anchors are taken on an interval and on demand, written into the audit
//! log as `ChainAnchored` and held here so a verification can be held
//! against them without reading the log back.
//!
//! An anchor held inside the cluster is only as good as the cluster. An
//! operator exports one, signed under the release signing scheme, and keeps
//! it outside: a chain rewritten from end to end still contradicts an
//! exported anchor, because whoever rewrote it could not produce the
//! signature over the head it used to name.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use parking_lot::Mutex;
use zyron_common::format::envelope;
use zyron_common::format::{FormatKind, FormatVersion};
use zyron_common::{Result, ZyronError};

use super::{
    ChainHash, hex, read_hash, read_text, read_u32, read_u64, write_replacing, write_text,
};

/// Version the anchor store and an exported anchor are written at
pub const ANCHOR_FORMAT_VERSION: FormatVersion = FormatVersion::V1;

/// How often the head of every verified table is anchored when nothing sets
/// the interval
pub const DEFAULT_ANCHOR_INTERVAL_SECS: u64 = 3600;

/// One recorded head of one table's chain.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Anchor {
    pub table_id: u32,
    pub table_name: String,
    /// The chain position the head stood at
    pub sequence: u64,
    /// The version the head's entry covered
    pub commit_version: u64,
    pub head_hash: ChainHash,
    /// Microseconds since the epoch
    pub taken_at: i64,
}

impl Anchor {
    /// The bytes a signature covers, which is every field an anchor states.
    ///
    /// Built the same way here and by whoever checks an exported anchor, so
    /// a signature is over one description of the head rather than over
    /// whatever the file happened to hold
    pub fn signed_bytes(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(96 + self.table_name.len());
        out.extend_from_slice(b"zyron-chain-anchor-v1");
        out.extend_from_slice(&self.table_id.to_le_bytes());
        out.extend_from_slice(self.table_name.as_bytes());
        out.extend_from_slice(&self.sequence.to_le_bytes());
        out.extend_from_slice(&self.commit_version.to_le_bytes());
        out.extend_from_slice(&self.head_hash);
        out.extend_from_slice(&self.taken_at.to_le_bytes());
        out
    }

    /// The line the audit record and the view state an anchor by.
    pub fn describe(&self) -> String {
        format!(
            "table {} at version {} stands at head {}",
            self.table_name,
            self.commit_version,
            hex(&self.head_hash)
        )
    }

    fn encode(&self, out: &mut Vec<u8>) {
        out.extend_from_slice(&self.table_id.to_le_bytes());
        write_text(out, &self.table_name);
        out.extend_from_slice(&self.sequence.to_le_bytes());
        out.extend_from_slice(&self.commit_version.to_le_bytes());
        out.extend_from_slice(&self.head_hash);
        out.extend_from_slice(&self.taken_at.to_le_bytes());
    }

    fn decode(data: &[u8], at: &mut usize) -> Result<Self> {
        Ok(Self {
            table_id: read_u32(data, at)?,
            table_name: read_text(data, at)?,
            sequence: read_u64(data, at)?,
            commit_version: read_u64(data, at)?,
            head_hash: read_hash(data, at)?,
            taken_at: read_u64(data, at)? as i64,
        })
    }
}

/// An anchor an operator holds outside the cluster.
///
/// The signature is over [`Anchor::signed_bytes`] under the release signing
/// scheme, so the artifact states both what the head was and that this
/// cluster is what said so
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExportedAnchor {
    pub anchor: Anchor,
    /// The scheme the signature was produced under
    pub scheme: String,
    pub signature: Vec<u8>,
}

impl ExportedAnchor {
    /// The artifact's bytes, which is what an operator keeps.
    pub fn encode(&self) -> Vec<u8> {
        let mut body = Vec::with_capacity(256);
        self.anchor.encode(&mut body);
        write_text(&mut body, &self.scheme);
        body.extend_from_slice(&(self.signature.len() as u32).to_le_bytes());
        body.extend_from_slice(&self.signature);
        envelope::encode(FormatKind::VerificationAnchor, ANCHOR_FORMAT_VERSION, &body)
    }

    /// Reads an artifact back.
    pub fn decode(bytes: &[u8]) -> Result<Self> {
        let parsed = envelope::decode_as(bytes, FormatKind::VerificationAnchor)?;
        let data = parsed.body;
        let mut at = 0usize;
        let anchor = Anchor::decode(data, &mut at)?;
        let scheme = read_text(data, &mut at)?;
        let len = read_u32(data, &mut at)? as usize;
        if data.len() < at + len {
            return Err(ZyronError::Internal(
                "an exported anchor ends inside its signature".to_string(),
            ));
        }
        let signature = data[at..at + len].to_vec();
        Ok(Self {
            anchor,
            scheme,
            signature,
        })
    }

    /// Whether the artifact describes the head a chain stands at, which is
    /// the half a reader can check without a key.
    ///
    /// The signature is checked separately, by whoever holds the verifying
    /// material. Both halves have to hold: an artifact that matches the
    /// chain but carries no valid signature was written by whoever rewrote
    /// the chain
    pub fn agrees_with(&self, sequence: u64, head_hash: &ChainHash) -> bool {
        self.anchor.sequence == sequence && self.anchor.head_hash == *head_hash
    }
}

/// Every anchor this node holds, the latest per table.
///
/// One anchor per table rather than a history: an anchor contradicted by a
/// shorter chain is contradicted whichever of them is compared, and the
/// latest is the one a truncation has to get past
#[derive(Debug)]
pub struct AnchorStore {
    path: PathBuf,
    anchors: Mutex<HashMap<u32, Anchor>>,
}

impl AnchorStore {
    /// Opens the store, putting back what the last process anchored.
    ///
    /// A store that cannot be read is an error rather than an empty store:
    /// starting with no anchors would let a chain that a held anchor
    /// contradicts verify clean
    pub fn open(dir: &Path) -> Result<Self> {
        let path = dir.join("verify").join("anchors.zvan");
        let anchors = if path.exists() {
            let bytes = std::fs::read(&path).map_err(|e| {
                ZyronError::Internal(format!(
                    "the anchor store at {} could not be read, {e}",
                    path.display()
                ))
            })?;
            let parsed = envelope::decode_as(&bytes, FormatKind::VerificationAnchor)?;
            let data = parsed.body;
            let mut at = 0usize;
            let count = read_u32(data, &mut at)? as usize;
            let mut held = HashMap::with_capacity(count);
            for _ in 0..count {
                let anchor = Anchor::decode(data, &mut at)?;
                held.insert(anchor.table_id, anchor);
            }
            held
        } else {
            HashMap::new()
        };
        Ok(Self {
            path,
            anchors: Mutex::new(anchors),
        })
    }

    /// Records an anchor and writes the store down.
    ///
    /// Written before the caller is told the head is anchored. An anchor
    /// reported as taken and not persisted would be gone at the next
    /// restart, and the truncation it was there to catch would pass
    pub fn anchor(&self, anchor: Anchor) -> Result<()> {
        self.anchor_all(std::iter::once(anchor))
    }

    /// Records every anchor of a pass and writes the store down once.
    ///
    /// The store is held under its lock while it is written, so two passes
    /// that anchor together both reach the file, and a pass over every
    /// verified table costs one write rather than one per table
    pub fn anchor_all(&self, anchors: impl IntoIterator<Item = Anchor>) -> Result<()> {
        let mut held = self.anchors.lock();
        let mut any = false;
        for anchor in anchors {
            held.insert(anchor.table_id, anchor);
            any = true;
        }
        if !any {
            return Ok(());
        }
        Self::persist(&held, &self.path)
    }

    /// The anchor over one table, if one was taken.
    pub fn for_table(&self, table_id: u32) -> Option<Anchor> {
        self.anchors.lock().get(&table_id).cloned()
    }

    /// Every anchor, ordered by table.
    pub fn all(&self) -> Vec<Anchor> {
        Self::ordered(&self.anchors.lock())
    }

    /// Drops the anchor over a table that is gone.
    pub fn forget(&self, table_id: u32) -> Result<()> {
        let mut held = self.anchors.lock();
        if held.remove(&table_id).is_none() {
            return Ok(());
        }
        Self::persist(&held, &self.path)
    }

    fn ordered(held: &HashMap<u32, Anchor>) -> Vec<Anchor> {
        let mut out: Vec<Anchor> = held.values().cloned().collect();
        out.sort_by_key(|anchor| anchor.table_id);
        out
    }

    fn persist(held: &HashMap<u32, Anchor>, path: &Path) -> Result<()> {
        let ordered = Self::ordered(held);
        let mut body = Vec::with_capacity(128 * ordered.len() + 4);
        body.extend_from_slice(&(ordered.len() as u32).to_le_bytes());
        for anchor in &ordered {
            anchor.encode(&mut body);
        }
        let bytes = envelope::encode(FormatKind::VerificationAnchor, ANCHOR_FORMAT_VERSION, &body);
        write_replacing(path, &bytes)
    }
}

/// Whether a head has gone unanchored longer than twice the interval, which
/// is what `chain_not_anchored` fires on.
///
/// A chain with no anchor at all and no commits is not overdue: there is
/// nothing to anchor. A chain that holds commits and has never been
/// anchored is overdue as soon as twice the interval has passed since its
/// head was written
pub fn anchor_overdue(
    head_commits: u64,
    head_ts: i64,
    anchored: Option<&Anchor>,
    interval_secs: u64,
    now_micros: i64,
) -> bool {
    if head_commits == 0 {
        return false;
    }
    let window = (interval_secs as i64)
        .saturating_mul(2)
        .saturating_mul(1_000_000);
    match anchored {
        Some(anchor) if anchor.sequence + 1 >= head_commits => false,
        Some(anchor) => now_micros.saturating_sub(anchor.taken_at) > window,
        None => now_micros.saturating_sub(head_ts) > window,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn anchor(table_id: u32, sequence: u64, taken_at: i64) -> Anchor {
        Anchor {
            table_id,
            table_name: format!("t{table_id}"),
            sequence,
            commit_version: 100 + sequence,
            head_hash: [sequence as u8; 32],
            taken_at,
        }
    }

    #[test]
    fn test_the_store_reopens_with_what_it_held() {
        let dir = tempfile::TempDir::new().expect("temp dir");
        let store = AnchorStore::open(dir.path()).expect("opens");
        assert!(store.all().is_empty());
        store.anchor(anchor(1, 4, 500)).expect("anchors");
        store.anchor(anchor(2, 9, 600)).expect("anchors");
        drop(store);

        let again = AnchorStore::open(dir.path()).expect("reopens");
        let held = again.all();
        assert_eq!(held.len(), 2);
        assert_eq!(held[0], anchor(1, 4, 500));
        assert_eq!(held[1], anchor(2, 9, 600));
    }

    #[test]
    fn test_a_later_anchor_replaces_the_one_before_it() {
        let dir = tempfile::TempDir::new().expect("temp dir");
        let store = AnchorStore::open(dir.path()).expect("opens");
        store.anchor(anchor(1, 4, 500)).expect("anchors");
        store.anchor(anchor(1, 7, 900)).expect("anchors");
        assert_eq!(store.for_table(1).map(|a| a.sequence), Some(7));
        assert_eq!(store.all().len(), 1);
    }

    #[test]
    fn test_forgetting_a_table_drops_its_anchor() {
        let dir = tempfile::TempDir::new().expect("temp dir");
        let store = AnchorStore::open(dir.path()).expect("opens");
        store.anchor(anchor(3, 1, 10)).expect("anchors");
        store.forget(3).expect("forgets");
        assert!(store.for_table(3).is_none());
        let again = AnchorStore::open(dir.path()).expect("reopens");
        assert!(again.all().is_empty());
    }

    #[test]
    fn test_an_exported_anchor_round_trips_and_states_the_head() {
        let exported = ExportedAnchor {
            anchor: anchor(5, 12, 777),
            scheme: "Ed25519".to_string(),
            signature: vec![9u8; 64],
        };
        let bytes = exported.encode();
        let back = ExportedAnchor::decode(&bytes).expect("decodes");
        assert_eq!(back, exported);
        assert!(back.agrees_with(12, &[12u8; 32]));
        assert!(!back.agrees_with(11, &[12u8; 32]));
        assert!(!back.agrees_with(12, &[0u8; 32]));
    }

    #[test]
    fn test_the_signed_bytes_change_with_every_field() {
        let base = anchor(1, 2, 3);
        let mut moved_head = base.clone();
        moved_head.head_hash = [1u8; 32];
        let mut moved_seq = base.clone();
        moved_seq.sequence = 3;
        let mut moved_version = base.clone();
        moved_version.commit_version = 999;
        let mut moved_time = base.clone();
        moved_time.taken_at = 4;
        for other in [moved_head, moved_seq, moved_version, moved_time] {
            assert_ne!(base.signed_bytes(), other.signed_bytes());
        }
    }

    #[test]
    fn test_a_head_already_anchored_is_not_overdue() {
        let held = anchor(1, 9, 1_000);
        assert!(!anchor_overdue(10, 1_000, Some(&held), 60, 10_000_000_000));
    }

    #[test]
    fn test_a_head_past_twice_the_interval_is_overdue() {
        let held = anchor(1, 4, 1_000);
        let interval = 60u64;
        let window = (interval as i64) * 2 * 1_000_000;
        assert!(!anchor_overdue(
            10,
            1_000,
            Some(&held),
            interval,
            1_000 + window
        ));
        assert!(anchor_overdue(
            10,
            1_000,
            Some(&held),
            interval,
            1_000 + window + 1
        ));
    }

    #[test]
    fn test_a_chain_never_anchored_is_overdue_from_its_head() {
        let interval = 60u64;
        let window = (interval as i64) * 2 * 1_000_000;
        assert!(!anchor_overdue(3, 500, None, interval, 500 + window));
        assert!(anchor_overdue(3, 500, None, interval, 500 + window + 1));
    }

    #[test]
    fn test_an_empty_chain_is_never_overdue() {
        assert!(!anchor_overdue(0, 0, None, 1, i64::MAX));
    }
}
