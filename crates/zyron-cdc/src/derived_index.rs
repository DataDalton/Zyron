//! The record index of a change data feed whose records are derived from
//! the table's own store.
//!
//! A heap table's feed counts the records it writes as it writes them and
//! keeps the count per version in its manifest. A lake table's feed writes
//! nothing, its transaction log is the change record, so the count of what
//! each commit yields is taken from the log once and kept here, the same
//! per version index with the same absolute counts, so a stream position
//! over either kind of table is the same number meaning the same place.
//!
//! The index is durable because retention reclaims the log's oldest
//! versions, and the changes of a reclaimed version can no longer be
//! counted from the log. A position counts from the version the feed began
//! at, which is below everything the log will eventually still hold, so
//! the counts of the reclaimed versions have to outlive them. The index
//! only ever grows at its end, a committed version never changes, and the
//! file is a log of that growth, one record holding the whole index, then
//! one record appended per extension, each framed and checksummed on its
//! own. Opening replays the records, an extension costs a write of what it
//! added, and once the appended records pass a threshold the next write
//! lays the file down whole again.
//!
//! Whether an update's removed side is a record is a setting of the feed,
//! and a heap feed applies it as it writes, so a record once written stays
//! written and the numbering of a version's records never moves. A derived
//! feed applies the setting as it reads, so the index keeps the setting in
//! force for every version it counted, and a read of a version derives it
//! under that same setting, which is what keeps a position that names a
//! record inside a version naming the same record after the setting changes.
//!
//! Beside each count the index keeps the transaction the version's records
//! are handed over with, so a read over several sources finds every
//! version a transaction committed to this one and ends where the
//! transaction is whole in all of them, the way a heap feed's transaction
//! spans decide where a bounded read ends. In memory only, it also keeps
//! the transactions whose commits it has counted or seen pending and whose
//! end no reader has observed yet. A read holds its window below the first
//! commit of a transaction its snapshot says is still open, the way a heap
//! feed holds back the rows of an unfinished transaction, so a transaction
//! that wrote a heap table and a lake table is handed over to both at once

use std::collections::HashMap;
use std::fs::{self, File, OpenOptions};
use std::io::{Read, Write};
use std::path::{Path, PathBuf};

use zyron_common::format::FormatKind;
use zyron_common::format::envelope;
use zyron_common::{Result, ZyronError};

use crate::change_feed::{ByteCursor, VersionCount, sync_parent_dir, version_index};
use crate::format::CHANGE_FEED_VERSION_INDEX_FORMAT_VERSION;

/// The file the index of one table's feed lives in
const INDEX_FILE: &str = "lake_index.zycdi";

/// Extension records appended after a whole record before the next write
/// lays the file down whole again. Opening replays at most this many
/// records past the whole one
const COMPACT_AFTER: usize = 64;

/// The record kinds the file holds, the first byte of each record's body
const RECORD_WHOLE: u8 = 0;
const RECORD_EXTENSION: u8 = 1;

/// Bytes of the per record header extension, which carries the body
/// length so the records of one file can be told apart
const RECORD_LENGTH_EXTENSION: usize = 4;

/// Bytes one version entry takes in a record, the version, its record
/// count, the records before it, its first instant and the transaction it
/// is handed over with, eight bytes each
const VERSION_ENTRY_LEN: usize = 40;

/// What one counted version yields
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CountedVersion {
    /// The change records the version's commit produces
    pub records: u64,
    /// The instant the version committed
    pub first_timestamp: i64,
    /// The transaction the commit ran under, zero for one that belongs to
    /// no transaction a reader could still hold open
    pub txn_id: u64,
    /// The transaction the version's records are handed over with, the
    /// same as `txn_id` when that is known, the commit's own transaction
    /// id for an intent whose owner is not recorded, and zero for a
    /// standalone commit, which is handed over on its own
    pub span_txn: u64,
}

/// What a derived source holds at one instant
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DerivedBoundary {
    /// The newest version readers can see, zero when none
    pub latest: u64,
    /// Records at or below it, counted from the feed's first version
    pub records: u64,
    /// The lowest version at or below `latest` committed by a transaction
    /// that has not ended, None when every counted commit's transaction
    /// has
    pub first_open: Option<u64>,
}

/// A per version record index over a derived feed, kept on disk beside
/// where a heap table's feed would keep its segments
pub struct DerivedRecordIndex {
    path: PathBuf,
    /// The version the feed began at, above which its records lie. An
    /// index on disk that began at another version counted a different
    /// feed, one turned on earlier, and is replaced rather than continued
    first_version: u64,
    /// The version the index is built through, so the versions above it
    /// are the only ones the store is asked about
    built_through: u64,
    /// Records in every version through `built_through`
    records: u64,
    /// One entry per version that yields records, ascending
    versions: Vec<VersionCount>,
    /// The transaction each entry of `versions` is handed over with, zero
    /// for a standalone commit
    span_txns: Vec<u64>,
    /// The lowest and highest version each transaction in `span_txns`
    /// committed at, rebuilt from the entries whenever they are read back
    spans: HashMap<u64, (u64, u64)>,
    /// The version each setting of the feed took effect at and whether an
    /// update's removed side is a record under it, ascending. The last one
    /// is the setting in force
    settings: Vec<(u64, bool)>,
    /// Extension records appended to the file since it was last written
    /// whole, None while the file has not been written at all
    appended: Option<usize>,
    /// The entries and settings added since the last write, which is what
    /// the next extension record carries
    unwritten_versions: usize,
    unwritten_settings: usize,
    /// Transactions with a counted or pending commit whose end no reader
    /// has observed, each with the lowest and highest version it committed
    in_flight: HashMap<u64, (u64, u64)>,
}

impl DerivedRecordIndex {
    /// Opens the index of a table's feed from disk, or starts one at
    /// `first_version` under `before_images` when none is there or the one
    /// there began at another version
    pub fn open(
        data_dir: &Path,
        table_id: u32,
        first_version: u64,
        before_images: bool,
    ) -> Result<Self> {
        Self::open_in(
            &data_dir.join("cdf").join(format!("{table_id:08}")),
            first_version,
            before_images,
        )
    }

    /// Opens the index kept in `dir`, which is a table's feed directory
    /// for the table's own feed and a branch's directory under it for the
    /// branch's, on the same terms as `open`
    pub fn open_in(dir: &Path, first_version: u64, before_images: bool) -> Result<Self> {
        let path = dir.join(INDEX_FILE);
        let fresh = Self {
            path: path.clone(),
            first_version,
            built_through: first_version,
            records: 0,
            versions: Vec::new(),
            span_txns: Vec::new(),
            spans: HashMap::new(),
            settings: vec![(first_version.saturating_add(1), before_images)],
            appended: None,
            unwritten_versions: 0,
            unwritten_settings: 1,
            in_flight: HashMap::new(),
        };
        if !path.exists() {
            return Ok(fresh);
        }
        let mut bytes = Vec::new();
        File::open(&path)?.read_to_end(&mut bytes)?;
        if bytes.is_empty() {
            return Ok(fresh);
        }
        let replayed = Self::replay(&path, &bytes)?;
        let Some((index, good_len)) = replayed else {
            return Ok(fresh);
        };
        if index.first_version != first_version {
            return Ok(fresh);
        }
        // A record the file ends in that never became whole is an
        // extension whose write was interrupted, so its versions were never
        // counted as far as the file says, and the next write starts where
        // the last whole record ended rather than after the torn bytes
        if good_len < bytes.len() {
            OpenOptions::new()
                .write(true)
                .open(&path)?
                .set_len(good_len as u64)?;
        }
        Ok(index)
    }

    /// Replays the records of an index file, answering with the index and
    /// the byte length of the records that decoded, None for a file whose
    /// first record does not decode or is not a whole index
    fn replay(path: &Path, bytes: &[u8]) -> Result<Option<(Self, usize)>> {
        let mut index: Option<Self> = None;
        let mut offset = 0usize;
        let mut good_len = 0usize;
        let mut appended = 0usize;
        while offset < bytes.len() {
            let Some((body, used)) = Self::next_record(path, &bytes[offset..], index.is_none())?
            else {
                break;
            };
            let mut cursor = ByteCursor::new(body);
            let kind = cursor.u8()?;
            match (kind, index.as_mut()) {
                (RECORD_WHOLE, _) => {
                    index = Some(Self::decode_whole(path, &mut cursor)?);
                    appended = 0;
                }
                (RECORD_EXTENSION, Some(index)) => {
                    index.apply_extension(path, &mut cursor)?;
                    appended += 1;
                }
                (RECORD_EXTENSION, None) => {
                    return Err(ZyronError::CdcDecoderError(format!(
                        "change feed version index {} begins with an extension record rather \
                         than a whole index",
                        path.display()
                    )));
                }
                (other, _) => {
                    return Err(ZyronError::CdcDecoderError(format!(
                        "change feed version index {} holds a record of kind {other}, which \
                         this binary does not read",
                        path.display()
                    )));
                }
            }
            offset += used;
            good_len = offset;
        }
        Ok(index.map(|mut index| {
            index.appended = Some(appended);
            (index, good_len)
        }))
    }

    /// The next record of the file, its body and the bytes it spans. None
    /// for a trailing record that does not decode, which is one whose
    /// write was interrupted. The first record has to decode, because a
    /// file that begins with bytes that are not an index is not one
    fn next_record<'a>(
        path: &Path,
        bytes: &'a [u8],
        first: bool,
    ) -> Result<Option<(&'a [u8], usize)>> {
        let refuse = |detail: String| {
            ZyronError::CdcDecoderError(format!(
                "change feed version index {} did not decode, {detail}",
                path.display()
            ))
        };
        let header = match envelope::decode_header(bytes) {
            Ok((header, extension)) => {
                if header.kind != FormatKind::ChangeFeedVersionIndex {
                    return Err(refuse(format!("a {} record is inside it", header.kind)));
                }
                if header.version != CHANGE_FEED_VERSION_INDEX_FORMAT_VERSION {
                    return Err(ZyronError::CdcDecoderError(format!(
                        "change feed version index is at format version {}, this binary \
                         writes and reads {}. Upgrade through a release that still reads {} \
                         to move it forward first",
                        header.version, CHANGE_FEED_VERSION_INDEX_FORMAT_VERSION, header.version
                    )));
                }
                if extension.len() != RECORD_LENGTH_EXTENSION {
                    return Err(refuse(format!(
                        "a record carries a {} byte header extension rather than the body \
                         length",
                        extension.len()
                    )));
                }
                let body_len =
                    u32::from_le_bytes([extension[0], extension[1], extension[2], extension[3]])
                        as usize;
                Some((header, body_len))
            }
            Err(e) if first => return Err(refuse(e.to_string())),
            Err(_) => None,
        };
        let Some((header, body_len)) = header else {
            return Ok(None);
        };
        let total = header.body_offset() + body_len + envelope::ENVELOPE_FOOTER_LEN;
        if total > bytes.len() {
            if first {
                return Err(refuse(format!(
                    "the first record declares {total} bytes and the file holds {}",
                    bytes.len()
                )));
            }
            return Ok(None);
        }
        match envelope::decode_as(&bytes[..total], FormatKind::ChangeFeedVersionIndex) {
            Ok(parsed) => Ok(Some((parsed.body, total))),
            Err(e) if first => Err(refuse(e.to_string())),
            Err(_) => Ok(None),
        }
    }

    /// Decodes a whole index record, everything after its kind byte
    fn decode_whole(path: &Path, cursor: &mut ByteCursor<'_>) -> Result<Self> {
        let first_version = cursor.u64()?;
        let built_through = cursor.u64()?;
        let records = cursor.u64()?;
        let (versions, span_txns) = Self::decode_versions(cursor)?;
        let settings = Self::decode_settings(cursor)?;
        if settings.is_empty() {
            return Err(ZyronError::CdcDecoderError(format!(
                "change feed version index {} records no setting",
                path.display()
            )));
        }
        let mut spans = HashMap::new();
        for (entry, txn) in versions.iter().zip(&span_txns) {
            Self::note_span(&mut spans, *txn, entry.version);
        }
        Ok(Self {
            path: path.to_path_buf(),
            first_version,
            built_through,
            records,
            versions,
            span_txns,
            spans,
            settings,
            appended: Some(0),
            unwritten_versions: 0,
            unwritten_settings: 0,
            in_flight: HashMap::new(),
        })
    }

    /// Widens the span of `txn` to reach `version`, or opens it there. A
    /// transaction of zero is a standalone commit and spans nothing
    fn note_span(spans: &mut HashMap<u64, (u64, u64)>, txn: u64, version: u64) {
        if txn == 0 {
            return;
        }
        spans
            .entry(txn)
            .and_modify(|(first, last)| {
                *first = (*first).min(version);
                *last = (*last).max(version);
            })
            .or_insert((version, version));
    }

    /// Applies an extension record, everything after its kind byte
    fn apply_extension(&mut self, path: &Path, cursor: &mut ByteCursor<'_>) -> Result<()> {
        let built_through = cursor.u64()?;
        let records = cursor.u64()?;
        let (versions, span_txns) = Self::decode_versions(cursor)?;
        let settings = Self::decode_settings(cursor)?;
        if built_through < self.built_through || records < self.records {
            return Err(ZyronError::CdcDecoderError(format!(
                "change feed version index {} has an extension record that moves the index \
                 backwards, from version {} to {} and {} records to {}",
                path.display(),
                self.built_through,
                built_through,
                self.records,
                records
            )));
        }
        if let (Some(last), Some(first)) = (self.versions.last(), versions.first())
            && first.version <= last.version
        {
            return Err(ZyronError::CdcDecoderError(format!(
                "change feed version index {} has an extension record whose first version {} \
                 is not above the version {} already counted",
                path.display(),
                first.version,
                last.version
            )));
        }
        self.built_through = built_through;
        self.records = records;
        for (entry, txn) in versions.iter().zip(&span_txns) {
            Self::note_span(&mut self.spans, *txn, entry.version);
        }
        self.versions.extend(versions);
        self.span_txns.extend(span_txns);
        for (from, on) in settings {
            match self.settings.last_mut() {
                Some(last) if last.0 == from => last.1 = on,
                _ => self.settings.push((from, on)),
            }
        }
        Ok(())
    }

    /// The entries of a record and, beside each, the transaction it is
    /// handed over with
    fn decode_versions(cursor: &mut ByteCursor<'_>) -> Result<(Vec<VersionCount>, Vec<u64>)> {
        let count = cursor.u32()? as usize;
        let mut versions = Vec::with_capacity(count);
        let mut span_txns = Vec::with_capacity(count);
        for _ in 0..count {
            versions.push(VersionCount {
                version: cursor.u64()?,
                records: cursor.u64()?,
                prior: cursor.u64()?,
                first_timestamp: cursor.i64()?,
            });
            span_txns.push(cursor.u64()?);
        }
        Ok((versions, span_txns))
    }

    fn decode_settings(cursor: &mut ByteCursor<'_>) -> Result<Vec<(u64, bool)>> {
        let count = cursor.u32()? as usize;
        let mut settings = Vec::with_capacity(count);
        for _ in 0..count {
            settings.push((cursor.u64()?, cursor.u8()? != 0));
        }
        Ok(settings)
    }

    /// The version the index is built through
    pub fn built_through(&self) -> u64 {
        self.built_through
    }

    /// Records in every version the index is built through
    pub fn records(&self) -> u64 {
        self.records
    }

    /// The entries, one per version that yields records, ascending
    pub fn versions(&self) -> &[VersionCount] {
        &self.versions
    }

    /// Whether an update's removed side is a record of `version`, under
    /// the setting in force when the version was, or will be, counted
    pub fn preimages_at(&self, version: u64) -> bool {
        let at = self.settings.partition_point(|(from, _)| *from <= version);
        match at.checked_sub(1) {
            Some(at) => self.settings[at].1,
            None => self.settings[0].1,
        }
    }

    /// The setting in force for the versions not yet counted
    pub fn before_images(&self) -> bool {
        self.settings.last().map(|(_, on)| *on).unwrap_or(true)
    }

    /// Extends the index through `through`, asking `count_at` what each
    /// version above `built_through` yields under the setting in force for
    /// it, None for a version that yields nothing. The index is made
    /// durable once anything was added, and the transaction of every
    /// counted version is noted as in flight until a reader observes its
    /// end.
    ///
    /// A version whose changes cannot be counted stops the extension there
    /// and is reported, so it is never recorded as yielding nothing and the
    /// next extension tries it again
    pub fn extend(
        &mut self,
        through: u64,
        mut count_at: impl FnMut(u64, bool) -> Result<Option<CountedVersion>>,
    ) -> Result<()> {
        let mut next = self.built_through.max(self.first_version).saturating_add(1);
        let mut added = false;
        let mut failed = None;
        while next <= through {
            let counted = match count_at(next, self.preimages_at(next)) {
                Ok(counted) => counted,
                Err(e) => {
                    failed = Some(e);
                    break;
                }
            };
            if let Some(counted) = counted {
                self.versions.push(VersionCount {
                    version: next,
                    records: counted.records,
                    prior: self.records,
                    first_timestamp: counted.first_timestamp,
                });
                self.span_txns.push(counted.span_txn);
                Self::note_span(&mut self.spans, counted.span_txn, next);
                self.records += counted.records;
                self.unwritten_versions += 1;
                self.note_in_flight(counted.txn_id, next);
            }
            self.built_through = next;
            added = true;
            next += 1;
        }
        if added {
            self.persist()?;
        }
        match failed {
            Some(e) => Err(e),
            None => Ok(()),
        }
    }

    /// Puts `before_images` in force from the version after `through`,
    /// counting everything through `through` under the setting it had
    /// first, so no version is ever counted under one setting and read
    /// under another. Nothing changes when the setting is already in force
    pub fn set_before_images(
        &mut self,
        through: u64,
        before_images: bool,
        count_at: impl FnMut(u64, bool) -> Result<Option<CountedVersion>>,
    ) -> Result<()> {
        if self.before_images() == before_images {
            return Ok(());
        }
        self.extend(through, count_at)?;
        let from = self.built_through.saturating_add(1);
        match self.settings.last_mut() {
            Some(last) if last.0 == from => {
                last.1 = before_images;
                self.unwritten_settings = self.unwritten_settings.max(1);
            }
            _ => {
                self.settings.push((from, before_images));
                self.unwritten_settings += 1;
            }
        }
        self.persist()
    }

    /// Records at or below `version`, counted from the feed's first version
    pub fn records_at_or_below(&self, version: u64) -> u64 {
        version_index::records_at_or_below(&self.versions, version).unwrap_or(0)
    }

    /// The transactions with a counted commit in `(from_exclusive,
    /// to_inclusive]`, each once, standalone commits left out
    pub fn txns_in(&self, from_exclusive: u64, to_inclusive: u64) -> Vec<u64> {
        let start = self
            .versions
            .partition_point(|entry| entry.version <= from_exclusive);
        let mut out = Vec::new();
        for (entry, txn) in self.versions[start..].iter().zip(&self.span_txns[start..]) {
            if entry.version > to_inclusive {
                break;
            }
            if *txn != 0 && out.last() != Some(txn) {
                out.push(*txn);
            }
        }
        out.sort_unstable();
        out.dedup();
        out
    }

    /// The lowest and highest version a transaction committed to this
    /// source at, None for one with no counted commit here
    pub fn span_of(&self, txn_id: u64) -> Option<(u64, u64)> {
        self.spans.get(&txn_id).copied()
    }

    /// Notes a commit of `txn_id` at `version` whose end no reader has
    /// observed. A transaction of zero belongs to no one a reader could
    /// hold open and is not noted
    pub fn note_in_flight(&mut self, txn_id: u64, version: u64) {
        if txn_id == 0 {
            return;
        }
        self.in_flight
            .entry(txn_id)
            .and_modify(|(lowest, highest)| {
                *lowest = (*lowest).min(version);
                *highest = (*highest).max(version);
            })
            .or_insert((version, version));
    }

    /// Forgets the transactions `ended` says are over whose every noted
    /// commit is at or below `head`, and answers with the lowest version
    /// the rest committed at, which is where a read that sees through
    /// `head` stops short
    pub fn retire_in_flight(&mut self, head: u64, ended: &dyn Fn(u64) -> bool) -> Option<u64> {
        self.in_flight
            .retain(|txn_id, (_, highest)| !(*highest <= head && ended(*txn_id)));
        self.in_flight.values().map(|(lowest, _)| *lowest).min()
    }

    /// Whether a transaction is noted as in flight
    pub fn holds_in_flight(&self, txn_id: u64) -> bool {
        self.in_flight.contains_key(&txn_id)
    }

    /// Writes what changed since the last write. An extension record is
    /// appended while the file is young, and the file is laid down whole
    /// through a temporary file and an atomic rename when it has never
    /// been written or has grown past the compaction threshold, so a stop
    /// at any instant leaves either the old index or the new one, and an
    /// interrupted append leaves a tail the next open discards
    fn persist(&mut self) -> Result<()> {
        match self.appended {
            Some(appended) if appended < COMPACT_AFTER => self.append_extension(),
            _ => self.write_whole(),
        }
    }

    fn append_extension(&mut self) -> Result<()> {
        let unwritten_versions = self.unwritten_versions.min(self.versions.len());
        let unwritten_settings = self.unwritten_settings.min(self.settings.len());
        let mut body = Vec::with_capacity(
            1 + 16 + 4 + unwritten_versions * VERSION_ENTRY_LEN + 4 + unwritten_settings * 9,
        );
        body.push(RECORD_EXTENSION);
        body.extend_from_slice(&self.built_through.to_le_bytes());
        body.extend_from_slice(&self.records.to_le_bytes());
        let from = self.versions.len() - unwritten_versions;
        Self::encode_versions(&mut body, &self.versions[from..], &self.span_txns[from..]);
        Self::encode_settings(
            &mut body,
            &self.settings[self.settings.len() - unwritten_settings..],
        );
        let bytes = Self::frame(&body);
        {
            let mut file = OpenOptions::new().append(true).open(&self.path)?;
            file.write_all(&bytes)?;
            file.sync_data()?;
        }
        self.appended = Some(self.appended.unwrap_or(0) + 1);
        self.unwritten_versions = 0;
        self.unwritten_settings = 0;
        Ok(())
    }

    fn write_whole(&mut self) -> Result<()> {
        let mut body = Vec::with_capacity(
            1 + 24 + 4 + self.versions.len() * VERSION_ENTRY_LEN + 4 + self.settings.len() * 9,
        );
        body.push(RECORD_WHOLE);
        body.extend_from_slice(&self.first_version.to_le_bytes());
        body.extend_from_slice(&self.built_through.to_le_bytes());
        body.extend_from_slice(&self.records.to_le_bytes());
        Self::encode_versions(&mut body, &self.versions, &self.span_txns);
        Self::encode_settings(&mut body, &self.settings);
        let bytes = Self::frame(&body);
        if let Some(parent) = self.path.parent() {
            fs::create_dir_all(parent)?;
        }
        let tmp = self.path.with_extension("zycdi.tmp");
        {
            let mut file = File::create(&tmp)?;
            file.write_all(&bytes)?;
            file.sync_all()?;
        }
        fs::rename(&tmp, &self.path)?;
        sync_parent_dir(&self.path)?;
        self.appended = Some(0);
        self.unwritten_versions = 0;
        self.unwritten_settings = 0;
        Ok(())
    }

    /// One record of the file, an envelope whose header extension carries
    /// the body length
    fn frame(body: &[u8]) -> Vec<u8> {
        let length = (body.len() as u32).to_le_bytes();
        envelope::encode_with(
            FormatKind::ChangeFeedVersionIndex,
            CHANGE_FEED_VERSION_INDEX_FORMAT_VERSION,
            0,
            &length,
            body,
        )
    }

    /// Writes entries with, beside each, the transaction it is handed over
    /// with, `VERSION_ENTRY_LEN` bytes per entry
    fn encode_versions(body: &mut Vec<u8>, versions: &[VersionCount], span_txns: &[u64]) {
        body.extend_from_slice(&(versions.len() as u32).to_le_bytes());
        for (entry, txn) in versions.iter().zip(span_txns) {
            body.extend_from_slice(&entry.version.to_le_bytes());
            body.extend_from_slice(&entry.records.to_le_bytes());
            body.extend_from_slice(&entry.prior.to_le_bytes());
            body.extend_from_slice(&entry.first_timestamp.to_le_bytes());
            body.extend_from_slice(&txn.to_le_bytes());
        }
    }

    fn encode_settings(body: &mut Vec<u8>, settings: &[(u64, bool)]) {
        body.extend_from_slice(&(settings.len() as u32).to_le_bytes());
        for (from, on) in settings {
            body.extend_from_slice(&from.to_le_bytes());
            body.push(u8::from(*on));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn counted(records: u64, first_timestamp: i64) -> CountedVersion {
        CountedVersion {
            records,
            first_timestamp,
            txn_id: 0,
            span_txn: 0,
        }
    }

    fn records_in(path: &Path) -> usize {
        let bytes = fs::read(path).expect("reads");
        let mut offset = 0;
        let mut records = 0;
        while offset < bytes.len() {
            let (header, extension) = envelope::decode_header(&bytes[offset..]).expect("header");
            let body_len =
                u32::from_le_bytes([extension[0], extension[1], extension[2], extension[3]])
                    as usize;
            offset += header.body_offset() + body_len + envelope::ENVELOPE_FOOTER_LEN;
            records += 1;
        }
        records
    }

    #[test]
    fn test_an_index_extends_persists_and_reopens_with_the_same_counts() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let mut index = DerivedRecordIndex::open(tmp.path(), 7, 1, true).expect("opens");
        assert_eq!(index.built_through(), 1);
        // Versions two and four yield records, three yields nothing
        index
            .extend(4, |version, before_images| {
                assert!(before_images);
                Ok(match version {
                    2 => Some(counted(3, 100)),
                    3 => None,
                    4 => Some(counted(5, 400)),
                    _ => unreachable!("only the versions above the first are asked about"),
                })
            })
            .expect("extends");
        assert_eq!(index.built_through(), 4);
        assert_eq!(index.records(), 8);
        assert_eq!(index.records_at_or_below(1), 0);
        assert_eq!(index.records_at_or_below(2), 3);
        assert_eq!(index.records_at_or_below(3), 3);
        assert_eq!(index.records_at_or_below(4), 8);

        let reopened = DerivedRecordIndex::open(tmp.path(), 7, 1, true).expect("reopens");
        assert_eq!(reopened.built_through(), 4);
        assert_eq!(reopened.records(), 8);
        assert_eq!(reopened.versions(), index.versions());
        assert!(reopened.preimages_at(2));

        // A feed that began at another version starts its index over
        let other = DerivedRecordIndex::open(tmp.path(), 7, 3, true).expect("opens");
        assert_eq!(other.built_through(), 3);
        assert_eq!(other.records(), 0);
    }

    #[test]
    fn test_a_version_that_cannot_be_counted_is_left_for_the_next_extension() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let mut index = DerivedRecordIndex::open(tmp.path(), 9, 0, true).expect("opens");
        let failed = index
            .extend(3, |version, _| match version {
                1 => Ok(Some(counted(2, 10))),
                _ => Err(ZyronError::Internal("unreadable".to_string())),
            })
            .expect_err("the second version fails");
        assert!(failed.to_string().contains("unreadable"));
        assert_eq!(index.built_through(), 1, "what was counted stays counted");
        assert_eq!(index.records(), 2);
        index
            .extend(3, |version, _| Ok(Some(counted(1, version as i64))))
            .expect("the retry counts the rest");
        assert_eq!(index.built_through(), 3);
        assert_eq!(index.records(), 4);
    }

    #[test]
    fn test_a_setting_change_counts_what_came_before_under_the_old_setting() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let mut index = DerivedRecordIndex::open(tmp.path(), 4, 0, true).expect("opens");
        // Versions one and two are committed under before images, then the
        // setting turns off. Each version is counted under the setting it
        // had, whichever version the index had reached when it changed
        index
            .set_before_images(2, false, |version, before_images| {
                assert!(
                    before_images,
                    "version {version} counts under the old setting"
                );
                Ok(Some(counted(2, version as i64)))
            })
            .expect("the setting changes");
        assert_eq!(index.built_through(), 2);
        assert!(index.preimages_at(1));
        assert!(index.preimages_at(2));
        assert!(!index.preimages_at(3));
        assert!(!index.before_images());
        index
            .extend(3, |_, before_images| {
                assert!(!before_images);
                Ok(Some(counted(1, 3)))
            })
            .expect("extends under the new setting");
        assert_eq!(index.records(), 5);

        let reopened = DerivedRecordIndex::open(tmp.path(), 4, 0, false).expect("reopens");
        assert!(reopened.preimages_at(2));
        assert!(!reopened.preimages_at(3));
        assert_eq!(reopened.records(), 5);
    }

    /// Each extension appends one record, the file is laid down whole once
    /// the appended records pass the threshold, and every state along the
    /// way reopens to the same index
    #[test]
    fn test_extensions_append_and_the_file_compacts_past_the_threshold() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let mut index = DerivedRecordIndex::open(tmp.path(), 11, 0, true).expect("opens");
        index
            .extend(1, |_, _| Ok(Some(counted(1, 1))))
            .expect("the first extension writes the file whole");
        let path = index.path.clone();
        assert_eq!(records_in(&path), 1, "the first write is one whole record");

        // The whole record plus one appended record per extension, up to
        // the threshold of appended records
        for version in 2..=COMPACT_AFTER as u64 + 1 {
            index
                .extend(version, |v, _| Ok(Some(counted(1, v as i64))))
                .expect("extends");
            assert_eq!(
                records_in(&path),
                version as usize,
                "each extension appends one record"
            );
            let reopened = DerivedRecordIndex::open(tmp.path(), 11, 0, true).expect("reopens");
            assert_eq!(reopened.versions(), index.versions());
            assert_eq!(reopened.records(), index.records());
        }
        let next = COMPACT_AFTER as u64 + 2;
        index
            .extend(next, |v, _| Ok(Some(counted(1, v as i64))))
            .expect("the extension past the threshold compacts");
        assert_eq!(records_in(&path), 1, "the file is one whole record again");
        let reopened = DerivedRecordIndex::open(tmp.path(), 11, 0, true).expect("reopens");
        assert_eq!(reopened.records(), next);
        assert_eq!(reopened.built_through(), next);
        assert_eq!(reopened.versions().len(), next as usize);
    }

    /// An append the process stopped inside leaves bytes the next open
    /// discards, and the versions it carried are counted again
    #[test]
    fn test_a_torn_tail_record_is_discarded_on_open() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let mut index = DerivedRecordIndex::open(tmp.path(), 12, 0, true).expect("opens");
        index
            .extend(2, |v, _| Ok(Some(counted(2, v as i64))))
            .expect("extends");
        index
            .extend(3, |v, _| Ok(Some(counted(2, v as i64))))
            .expect("extends");
        let path = index.path.clone();
        let whole = fs::read(&path).expect("reads");
        // The last record, cut short
        let cut = whole.len() - 5;
        fs::write(&path, &whole[..cut]).expect("writes the torn file");
        let reopened = DerivedRecordIndex::open(tmp.path(), 12, 0, true).expect("reopens");
        assert_eq!(
            reopened.built_through(),
            2,
            "the torn extension is not counted"
        );
        assert_eq!(reopened.records(), 4);
        assert!(
            fs::metadata(&path).expect("stat").len() < cut as u64,
            "the torn bytes are cut off"
        );
        let mut reopened = reopened;
        reopened
            .extend(3, |v, _| Ok(Some(counted(2, v as i64))))
            .expect("the version counts again");
        let again = DerivedRecordIndex::open(tmp.path(), 12, 0, true).expect("reopens");
        assert_eq!(again.records(), 6);
        assert_eq!(again.built_through(), 3);
    }

    /// A counted commit's transaction is held until a reader observes its
    /// end, and a read stops below the lowest commit of the ones still open
    #[test]
    fn test_in_flight_transactions_hold_a_read_below_their_first_commit() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let mut index = DerivedRecordIndex::open(tmp.path(), 13, 0, true).expect("opens");
        index
            .extend(4, |v, _| {
                let txn_id = match v {
                    1 | 2 => 40,
                    3 => 41,
                    _ => 42,
                };
                Ok(Some(CountedVersion {
                    records: 1,
                    first_timestamp: v as i64,
                    txn_id,
                    span_txn: txn_id,
                }))
            })
            .expect("extends");
        // Transaction 41 is over, 40 and 42 are not
        let first_open = index.retire_in_flight(4, &|txn| txn == 41);
        assert_eq!(first_open, Some(1));
        assert!(index.holds_in_flight(40));
        assert!(!index.holds_in_flight(41));
        assert!(index.holds_in_flight(42));
        // Transaction 40 ends, 42 commits again at a version readers cannot
        // see yet, so it stays held whatever the reader says of it
        index.note_in_flight(42, 6);
        let first_open = index.retire_in_flight(4, &|txn| txn == 40 || txn == 42);
        assert_eq!(first_open, Some(4));
        let first_open = index.retire_in_flight(6, &|_| true);
        assert_eq!(first_open, None);
    }

    /// The transaction each version is handed over with is kept beside its
    /// count, on disk through an append and a whole write, so the versions
    /// a transaction committed are found from the index alone
    #[test]
    fn test_the_transaction_of_each_version_persists_and_answers_spans() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let mut index = DerivedRecordIndex::open(tmp.path(), 14, 0, true).expect("opens");
        // Transaction 7 commits versions one and two, version three is a
        // standalone commit, transaction 8 commits four, and five yields
        // nothing
        let txn_of = |v: u64| match v {
            1 | 2 => 7,
            4 => 8,
            _ => 0,
        };
        index
            .extend(3, |v, _| {
                Ok(Some(CountedVersion {
                    records: 1,
                    first_timestamp: v as i64,
                    txn_id: txn_of(v),
                    span_txn: txn_of(v),
                }))
            })
            .expect("extends");
        index
            .extend(5, |v, _| {
                Ok((v == 4).then_some(CountedVersion {
                    records: 2,
                    first_timestamp: v as i64,
                    txn_id: txn_of(v),
                    span_txn: txn_of(v),
                }))
            })
            .expect("extends again, appending a record");
        assert_eq!(index.span_of(7), Some((1, 2)));
        assert_eq!(index.span_of(8), Some((4, 4)));
        assert_eq!(index.span_of(0), None);
        assert_eq!(index.txns_in(0, 5), vec![7, 8]);
        assert_eq!(index.txns_in(1, 3), vec![7]);
        assert_eq!(index.txns_in(2, 3), Vec::<u64>::new());
        assert_eq!(index.txns_in(3, 5), vec![8]);

        let reopened = DerivedRecordIndex::open(tmp.path(), 14, 0, true).expect("reopens");
        assert_eq!(reopened.span_of(7), Some((1, 2)));
        assert_eq!(reopened.span_of(8), Some((4, 4)));
        assert_eq!(reopened.txns_in(0, 5), vec![7, 8]);
        assert_eq!(reopened.versions(), index.versions());
    }
}
