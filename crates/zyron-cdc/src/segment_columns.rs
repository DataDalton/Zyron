//! The column-sliced form a sealed change segment takes.
//!
//! A segment is written as a run of framed records, each a header and the
//! row's tuple bytes, because that is what a write hands over and appending
//! it costs one copy. A read of a wide table's changes is the other way
//! round: it wants two columns of forty and it wants them for every record,
//! so a sealed segment stores its records column by column. The record
//! headers become one block per field, and the rows of each layout become
//! one block per column, each compressed on its own, so a read of two
//! columns decompresses two blocks and appends them whole rather than
//! walking every byte of every row.
//!
//! The rows a segment holds may have been written under more than one
//! layout, a schema epoch each and the feed's column subset or the whole
//! table, so the rows are grouped by layout and each record says which
//! group its row is in. The records stay in their written order, and a
//! reader that wants rows rather than columns rebuilds each row from its
//! group's blocks, byte for byte what the writer framed, so every reader of
//! frames reads a column-sliced segment without knowing it.
//!
//! A layout the feed was never told, or a row that does not walk exactly
//! under its layout, keeps the segment in its framed form. The columns are
//! an arrangement of the same bytes, never a second version of them.
//!
//! The body ends with a directory naming where every block stands, so a
//! reader opens the file, reads the directory off its tail, and fetches
//! the blocks it wants with positioned reads. A read of two columns of
//! forty moves a fortieth of the file through memory

use std::collections::BTreeMap;
use std::fs::File;

use zyron_catalog::PhysicalColumn;
use zyron_common::checksum::hot::hot_hash32;
use zyron_common::{Result, TypeId, ZyronError};

use crate::change_feed::CdfCodec;

/// The layouts a feed has seen its rows written under, by schema epoch and
/// whether the rows hold the feed's column subset, as the physical type of
/// each column in tuple order
pub type RowLayouts = BTreeMap<(u32, bool), Vec<TypeId>>;

/// The layout key a record's row was written under
pub fn layout_key(epoch: u32, projected: bool) -> (u32, bool) {
    (epoch, projected)
}

/// The physical types of a layout, which is what walking its rows needs
pub fn layout_types(layout: &[PhysicalColumn]) -> Vec<TypeId> {
    layout.iter().map(|column| column.physical_type).collect()
}

/// A record whose row is in no group, a truncate or a row of no bytes
const NO_GROUP: u16 = u16::MAX;

/// Bytes of a record's fixed header inside a frame, the fields before the
/// row bytes
const HEAD_LEN: usize = 38;

/// What one record's header carries, borrowed from its frame
struct RecordParts<'a> {
    change_type: u8,
    version: u64,
    timestamp: i64,
    txn_id: u64,
    ordinal: u64,
    epoch: u32,
    flags: u8,
    row: &'a [u8],
    key: &'a [u8],
}

/// Reads one record's parts out of its frame body
fn parts_of(record: &[u8]) -> Result<RecordParts<'_>> {
    if record.len() < HEAD_LEN + 4 {
        return Err(ZyronError::CdcDecoderError(
            "change record is shorter than its header".into(),
        ));
    }
    let u64_at = |at: usize| {
        let mut wide = [0u8; 8];
        wide.copy_from_slice(&record[at..at + 8]);
        u64::from_le_bytes(wide)
    };
    let u32_at = |at: usize| {
        let mut word = [0u8; 4];
        word.copy_from_slice(&record[at..at + 4]);
        u32::from_le_bytes(word)
    };
    let row_len = u32_at(38) as usize;
    let row_end = 42 + row_len;
    if row_end + 4 > record.len() {
        return Err(ZyronError::CdcDecoderError(
            "change record row bytes run past the record".into(),
        ));
    }
    let key_len = u32_at(row_end) as usize;
    let key_end = row_end + 4 + key_len;
    if key_end != record.len() {
        return Err(ZyronError::CdcDecoderError(
            "change record key bytes do not end the record".into(),
        ));
    }
    Ok(RecordParts {
        change_type: record[0],
        version: u64_at(1),
        timestamp: u64_at(9) as i64,
        txn_id: u64_at(17),
        ordinal: u64_at(25),
        epoch: u32_at(33),
        flags: record[37],
        row: &record[42..row_end],
        key: &record[row_end + 4..key_end],
    })
}

/// The record flag saying the row holds the feed's column subset
const FLAG_PROJECTED: u8 = 1 << 1;

// ---------------------------------------------------------------------------
// Blocks
// ---------------------------------------------------------------------------

/// A block stored as it was encoded, with no codec applied
const BLOCK_RAW: u8 = 0;

/// Bytes a block's prefix takes, codec, raw length, stored length, checksum
const BLOCK_PREFIX: usize = 1 + 4 + 4 + 4;

/// Appends one block, compressed with the codec when that makes it smaller
/// and stored raw otherwise, so an incompressible block costs no decode
fn write_block(out: &mut Vec<u8>, raw: &[u8], codec: CdfCodec) -> Result<()> {
    let compressed = match codec {
        CdfCodec::None => None,
        codec => {
            let packed = codec.compress(raw)?;
            (packed.len() < raw.len()).then_some(packed)
        }
    };
    let (stored, kind) = match &compressed {
        Some(packed) => (packed.as_slice(), codec as u8),
        None => (raw, BLOCK_RAW),
    };
    out.push(kind);
    out.extend_from_slice(&(raw.len() as u32).to_le_bytes());
    out.extend_from_slice(&(stored.len() as u32).to_le_bytes());
    out.extend_from_slice(&hot_hash32(stored).to_le_bytes());
    out.extend_from_slice(stored);
    Ok(())
}

/// Where one block stands in the body, so it is fetched and decoded when
/// asked for
#[derive(Debug, Clone, Copy)]
struct BlockRef {
    codec: u8,
    raw_len: usize,
    /// The stored bytes, prefix and checksum excluded, as offsets into the
    /// body
    start: usize,
    end: usize,
    checksum: u32,
}

impl BlockRef {
    fn write_into(&self, out: &mut Vec<u8>) {
        out.push(self.codec);
        out.extend_from_slice(&(self.raw_len as u32).to_le_bytes());
        out.extend_from_slice(&((self.end - self.start) as u32).to_le_bytes());
        out.extend_from_slice(&self.checksum.to_le_bytes());
        out.extend_from_slice(&(self.start as u64).to_le_bytes());
    }

    fn read_from(cursor: &mut DirectoryCursor<'_>) -> Result<Self> {
        let codec = cursor.u8()?;
        let raw_len = cursor.u32()? as usize;
        let stored_len = cursor.u32()? as usize;
        let checksum = cursor.u32()?;
        let start = cursor.u64()? as usize;
        Ok(Self {
            codec,
            raw_len,
            start,
            end: start + stored_len,
            checksum,
        })
    }
}

/// Appends one block and answers with where it stands
fn append_block(out: &mut Vec<u8>, raw: &[u8], codec: CdfCodec) -> Result<BlockRef> {
    let at = out.len();
    write_block(out, raw, codec)?;
    let codec_byte = out[at];
    let mut word = [0u8; 4];
    word.copy_from_slice(&out[at + 9..at + 13]);
    Ok(BlockRef {
        codec: codec_byte,
        raw_len: raw.len(),
        start: at + BLOCK_PREFIX,
        end: out.len(),
        checksum: u32::from_le_bytes(word),
    })
}

fn short_body() -> ZyronError {
    ZyronError::CdcDecoderError(
        "column-sliced change segment ends inside a block it declares".into(),
    )
}

/// Decodes one block's stored bytes, verifying the checksum first
fn decode_block(stored: &[u8], block: &BlockRef) -> Result<Vec<u8>> {
    if stored.len() != block.end - block.start {
        return Err(short_body());
    }
    if hot_hash32(stored) != block.checksum {
        return Err(ZyronError::CdcDecoderError(
            "column-sliced change segment block checksum does not match".into(),
        ));
    }
    if block.codec == BLOCK_RAW {
        return Ok(stored.to_vec());
    }
    let codec = CdfCodec::from_u8(block.codec)?;
    let raw = codec.decompress(stored, block.raw_len)?;
    if raw.len() != block.raw_len {
        return Err(ZyronError::CdcDecoderError(
            "column-sliced change segment block decoded to the wrong length".into(),
        ));
    }
    Ok(raw)
}

/// Reads the directory's fields in order
struct DirectoryCursor<'a> {
    bytes: &'a [u8],
    at: usize,
}

impl DirectoryCursor<'_> {
    fn take(&mut self, len: usize) -> Result<&[u8]> {
        if self.at + len > self.bytes.len() {
            return Err(ZyronError::CdcDecoderError(
                "column-sliced change segment directory ends inside a field".into(),
            ));
        }
        let out = &self.bytes[self.at..self.at + len];
        self.at += len;
        Ok(out)
    }

    fn u8(&mut self) -> Result<u8> {
        Ok(self.take(1)?[0])
    }

    fn u16(&mut self) -> Result<u16> {
        let b = self.take(2)?;
        Ok(u16::from_le_bytes([b[0], b[1]]))
    }

    fn u32(&mut self) -> Result<u32> {
        let b = self.take(4)?;
        Ok(u32::from_le_bytes([b[0], b[1], b[2], b[3]]))
    }

    fn u64(&mut self) -> Result<u64> {
        let b = self.take(8)?;
        Ok(u64::from_le_bytes([
            b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7],
        ]))
    }
}

/// How many header blocks a body carries, in the order the directory
/// lists them
const HEADER_BLOCKS: usize = 10;

fn bytes_to_u64s(bytes: &[u8]) -> Vec<u64> {
    bytes
        .chunks_exact(8)
        .map(|c| u64::from_le_bytes([c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7]]))
        .collect()
}

fn bytes_to_u32s(bytes: &[u8]) -> Vec<u32> {
    bytes
        .chunks_exact(4)
        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

fn bytes_to_u16s(bytes: &[u8]) -> Vec<u16> {
    bytes
        .chunks_exact(2)
        .map(|c| u16::from_le_bytes([c[0], c[1]]))
        .collect()
}

// ---------------------------------------------------------------------------
// Encoding
// ---------------------------------------------------------------------------

/// The columns of one layout group as they are gathered
struct GroupBuild {
    epoch: u32,
    projected: bool,
    types: Vec<TypeId>,
    rows: u32,
    /// One bit per row per column, set for a NULL cell
    nulls: Vec<Vec<u8>>,
    /// A fixed column's cells back to back, a variable one's bytes
    values: Vec<Vec<u8>>,
    /// A variable column's cell starts, one more than the rows
    offsets: Vec<Vec<u32>>,
}

impl GroupBuild {
    fn new(epoch: u32, projected: bool, types: Vec<TypeId>) -> Self {
        let columns = types.len();
        let offsets = types
            .iter()
            .map(|t| {
                if t.fixed_size().is_some() {
                    Vec::new()
                } else {
                    vec![0u32]
                }
            })
            .collect();
        Self {
            epoch,
            projected,
            types,
            rows: 0,
            nulls: vec![Vec::new(); columns],
            values: vec![Vec::new(); columns],
            offsets,
        }
    }

    /// Takes one row apart into the columns. Answers false when the row does
    /// not walk exactly under the layout, which leaves the segment framed
    fn take(&mut self, row: &[u8]) -> bool {
        let columns = self.types.len();
        let bitmap_len = columns.div_ceil(8);
        if row.len() < bitmap_len {
            return false;
        }
        let (bitmap, cells) = row.split_at(bitmap_len);
        let mut at = 0usize;
        let row_index = self.rows as usize;
        for (c, physical) in self.types.iter().enumerate() {
            let is_null = (bitmap[c / 8] >> (c % 8)) & 1 == 1;
            let width = match physical.fixed_size() {
                Some(size) => size,
                None => {
                    if at + 4 > cells.len() {
                        return false;
                    }
                    let len = u32::from_le_bytes([
                        cells[at],
                        cells[at + 1],
                        cells[at + 2],
                        cells[at + 3],
                    ]) as usize;
                    4 + len
                }
            };
            if at + width > cells.len() {
                return false;
            }
            if row_index % 8 == 0 {
                self.nulls[c].push(0);
            }
            if is_null {
                let last = self.nulls[c].len() - 1;
                self.nulls[c][last] |= 1 << (row_index % 8);
            }
            if physical.fixed_size().is_some() {
                self.values[c].extend_from_slice(&cells[at..at + width]);
            } else {
                self.values[c].extend_from_slice(&cells[at + 4..at + width]);
                let end = self.values[c].len() as u32;
                self.offsets[c].push(end);
            }
            at += width;
        }
        if at != cells.len() {
            return false;
        }
        self.rows += 1;
        true
    }
}

/// Encodes framed records as a column-sliced body, None when a record's
/// layout is not among `layouts` or a row does not walk under its layout.
///
/// `frames` is the segment's body as written, and the result replaces it
/// after the segment header. The records keep their order
pub fn encode(frames: &[u8], layouts: &RowLayouts, codec: CdfCodec) -> Result<Option<Vec<u8>>> {
    let mut change_types: Vec<u8> = Vec::new();
    let mut versions: Vec<u8> = Vec::new();
    let mut timestamps: Vec<u8> = Vec::new();
    let mut txn_ids: Vec<u8> = Vec::new();
    let mut ordinals: Vec<u8> = Vec::new();
    let mut epochs: Vec<u8> = Vec::new();
    let mut flags: Vec<u8> = Vec::new();
    let mut group_of: Vec<u8> = Vec::new();
    let mut key_offsets: Vec<u32> = vec![0];
    let mut key_bytes: Vec<u8> = Vec::new();
    let mut groups: Vec<GroupBuild> = Vec::new();
    let mut group_index: BTreeMap<(u32, bool), u16> = BTreeMap::new();
    let mut records = 0u32;

    let mut walked = true;
    crate::change_feed::walk_frames_exact(frames, |record| {
        let parts = parts_of(record)?;
        change_types.push(parts.change_type);
        versions.extend_from_slice(&parts.version.to_le_bytes());
        timestamps.extend_from_slice(&parts.timestamp.to_le_bytes());
        txn_ids.extend_from_slice(&parts.txn_id.to_le_bytes());
        ordinals.extend_from_slice(&parts.ordinal.to_le_bytes());
        epochs.extend_from_slice(&parts.epoch.to_le_bytes());
        flags.push(parts.flags);
        key_bytes.extend_from_slice(parts.key);
        key_offsets.push(key_bytes.len() as u32);
        records += 1;
        if parts.row.is_empty() {
            group_of.extend_from_slice(&NO_GROUP.to_le_bytes());
            return Ok(());
        }
        let key = layout_key(parts.epoch, parts.flags & FLAG_PROJECTED != 0);
        let group = match group_index.get(&key) {
            Some(g) => *g,
            None => {
                let Some(types) = layouts.get(&key) else {
                    walked = false;
                    return Ok(());
                };
                if groups.len() >= NO_GROUP as usize {
                    walked = false;
                    return Ok(());
                }
                let g = groups.len() as u16;
                groups.push(GroupBuild::new(key.0, key.1, types.clone()));
                group_index.insert(key, g);
                g
            }
        };
        if !groups[group as usize].take(parts.row) {
            walked = false;
            return Ok(());
        }
        group_of.extend_from_slice(&group.to_le_bytes());
        Ok(())
    })?;
    if !walked {
        return Ok(None);
    }

    let mut out = Vec::with_capacity(frames.len() / 2);
    let key_offset_bytes: Vec<u8> = key_offsets.iter().flat_map(|o| o.to_le_bytes()).collect();
    let header_blocks = [
        append_block(&mut out, &change_types, codec)?,
        append_block(&mut out, &versions, codec)?,
        append_block(&mut out, &timestamps, codec)?,
        append_block(&mut out, &txn_ids, codec)?,
        append_block(&mut out, &ordinals, codec)?,
        append_block(&mut out, &epochs, codec)?,
        append_block(&mut out, &flags, codec)?,
        append_block(&mut out, &group_of, codec)?,
        append_block(&mut out, &key_offset_bytes, codec)?,
        append_block(&mut out, &key_bytes, codec)?,
    ];
    let mut group_blocks: Vec<Vec<(BlockRef, BlockRef, Option<BlockRef>)>> =
        Vec::with_capacity(groups.len());
    for group in &groups {
        let mut blocks = Vec::with_capacity(group.types.len());
        for (c, physical) in group.types.iter().enumerate() {
            let nulls = append_block(&mut out, &group.nulls[c], codec)?;
            let values = append_block(&mut out, &group.values[c], codec)?;
            let offsets = if physical.fixed_size().is_none() {
                let bytes: Vec<u8> = group.offsets[c]
                    .iter()
                    .flat_map(|o| o.to_le_bytes())
                    .collect();
                Some(append_block(&mut out, &bytes, codec)?)
            } else {
                None
            };
            blocks.push((nulls, values, offsets));
        }
        group_blocks.push(blocks);
    }

    // The directory, then its length, so a reader finds it from the end
    let directory_start = out.len();
    out.extend_from_slice(&records.to_le_bytes());
    out.extend_from_slice(&(groups.len() as u32).to_le_bytes());
    for block in &header_blocks {
        block.write_into(&mut out);
    }
    for (group, blocks) in groups.iter().zip(&group_blocks) {
        out.extend_from_slice(&group.epoch.to_le_bytes());
        out.push(u8::from(group.projected));
        out.extend_from_slice(&group.rows.to_le_bytes());
        out.extend_from_slice(&(group.types.len() as u16).to_le_bytes());
        for physical in &group.types {
            out.push(*physical as u8);
        }
        for (nulls, values, offsets) in blocks {
            nulls.write_into(&mut out);
            values.write_into(&mut out);
            if let Some(offsets) = offsets {
                offsets.write_into(&mut out);
            }
        }
    }
    let directory_len = (out.len() - directory_start) as u32;
    out.extend_from_slice(&directory_len.to_le_bytes());
    Ok(Some(out))
}

// ---------------------------------------------------------------------------
// Decoding
// ---------------------------------------------------------------------------

/// The record headers a scan judges records by, one vector per field, in
/// record order
#[derive(Debug, Default, Clone)]
pub struct RecordHeads {
    pub change_types: Vec<u8>,
    pub versions: Vec<u64>,
    pub timestamps: Vec<i64>,
    pub txn_ids: Vec<u64>,
    pub ordinals: Vec<u64>,
    /// Which layout group each record's row is in, `u16::MAX` for a record
    /// with no row
    pub group_of: Vec<u16>,
}

impl RecordHeads {
    pub fn len(&self) -> usize {
        self.change_types.len()
    }

    pub fn is_empty(&self) -> bool {
        self.change_types.is_empty()
    }

    /// Whether a record's row is in a group at all
    pub fn has_row(&self, record: usize) -> bool {
        self.group_of[record] != NO_GROUP
    }
}

/// The rest of the record headers, which only a reader rebuilding frames
/// needs
struct RecordExtras {
    epochs: Vec<u32>,
    flags: Vec<u8>,
    key_offsets: Vec<u32>,
    key_bytes: Vec<u8>,
}

/// One layout group's shape and where its column blocks stand
#[derive(Debug, Clone)]
pub struct GroupShape {
    pub epoch: u32,
    pub projected: bool,
    pub rows: usize,
    pub types: Vec<TypeId>,
    /// Per column, the nulls block, the values block and, for a variable
    /// width column, the offsets block
    blocks: Vec<(BlockRef, BlockRef, Option<BlockRef>)>,
}

/// One column of one group, decoded
#[derive(Debug, Clone)]
pub enum ColumnBlock {
    /// Cells of one width back to back, a NULL cell's bytes zero
    Fixed {
        width: usize,
        nulls: Vec<u8>,
        values: Vec<u8>,
    },
    /// Cells laid end to end with their starts, one more than the rows
    Varlen {
        nulls: Vec<u8>,
        offsets: Vec<u32>,
        bytes: Vec<u8>,
    },
}

impl ColumnBlock {
    /// Whether the cell at `row` is NULL
    #[inline]
    pub fn is_null(&self, row: usize) -> bool {
        let nulls = match self {
            ColumnBlock::Fixed { nulls, .. } => nulls,
            ColumnBlock::Varlen { nulls, .. } => nulls,
        };
        nulls
            .get(row / 8)
            .is_some_and(|byte| (byte >> (row % 8)) & 1 == 1)
    }

    /// The cell's bytes at `row`, the width's worth for a fixed column and
    /// the cell's own for a variable one
    #[inline]
    pub fn cell(&self, row: usize) -> &[u8] {
        match self {
            ColumnBlock::Fixed { width, values, .. } => &values[row * width..(row + 1) * width],
            ColumnBlock::Varlen { offsets, bytes, .. } => {
                &bytes[offsets[row] as usize..offsets[row + 1] as usize]
            }
        }
    }
}

/// Where a body's bytes come from, the file they stand in read at the
/// offsets asked for, or the whole body held in memory
enum Backing {
    File { file: File, body_start: u64 },
    Memory(Vec<u8>),
}

impl Backing {
    /// The body's bytes from `start` for `len`
    fn read(&self, start: usize, len: usize) -> Result<std::borrow::Cow<'_, [u8]>> {
        match self {
            Backing::Memory(body) => {
                if start + len > body.len() {
                    return Err(short_body());
                }
                Ok(std::borrow::Cow::Borrowed(&body[start..start + len]))
            }
            Backing::File { file, body_start } => {
                let mut out = vec![0u8; len];
                read_file_at(file, *body_start + start as u64, &mut out)?;
                Ok(std::borrow::Cow::Owned(out))
            }
        }
    }
}

/// Fills `buf` from the file at `offset`, however many reads that takes
#[cfg(windows)]
pub(crate) fn read_file_at(file: &File, offset: u64, buf: &mut [u8]) -> Result<()> {
    use std::os::windows::fs::FileExt;
    let mut done = 0usize;
    while done < buf.len() {
        let read = file.seek_read(&mut buf[done..], offset + done as u64)?;
        if read == 0 {
            return Err(short_body());
        }
        done += read;
    }
    Ok(())
}

#[cfg(not(windows))]
pub(crate) fn read_file_at(file: &File, offset: u64, buf: &mut [u8]) -> Result<()> {
    use std::os::unix::fs::FileExt;
    file.read_exact_at(buf, offset).map_err(|e| {
        if e.kind() == std::io::ErrorKind::UnexpectedEof {
            short_body()
        } else {
            ZyronError::from(e)
        }
    })
}

/// A column-sliced segment body, its directory and the headers a scan
/// judges records by decoded, and its column blocks fetched and decoded as
/// they are asked for
pub struct ColumnarSegment {
    backing: Backing,
    body_len: usize,
    header_blocks: [BlockRef; HEADER_BLOCKS],
    heads: RecordHeads,
    groups: Vec<GroupShape>,
}

impl std::fmt::Debug for ColumnarSegment {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ColumnarSegment")
            .field("records", &self.heads.len())
            .field("groups", &self.groups.len())
            .finish()
    }
}

impl ColumnarSegment {
    /// Opens a body that stands in `file` from `body_start` for `body_len`
    /// bytes, reading its directory off the tail and its header blocks
    pub fn open(file: File, body_start: u64, body_len: usize) -> Result<Self> {
        Self::build(Backing::File { file, body_start }, body_len)
    }

    /// Parses a body held in memory
    pub fn parse(body: Vec<u8>) -> Result<Self> {
        let body_len = body.len();
        Self::build(Backing::Memory(body), body_len)
    }

    fn build(backing: Backing, body_len: usize) -> Result<Self> {
        if body_len < 4 {
            return Err(short_body());
        }
        let len_bytes = backing.read(body_len - 4, 4)?;
        let directory_len =
            u32::from_le_bytes([len_bytes[0], len_bytes[1], len_bytes[2], len_bytes[3]]) as usize;
        if directory_len + 4 > body_len {
            return Err(short_body());
        }
        let directory = backing.read(body_len - 4 - directory_len, directory_len)?;
        let mut cursor = DirectoryCursor {
            bytes: &directory,
            at: 0,
        };
        let records = cursor.u32()? as usize;
        let group_count = cursor.u32()? as usize;
        let mut header_blocks = [BlockRef {
            codec: BLOCK_RAW,
            raw_len: 0,
            start: 0,
            end: 0,
            checksum: 0,
        }; HEADER_BLOCKS];
        for block in header_blocks.iter_mut() {
            *block = BlockRef::read_from(&mut cursor)?;
        }
        let mut groups = Vec::with_capacity(group_count);
        for _ in 0..group_count {
            let epoch = cursor.u32()?;
            let projected = cursor.u8()? != 0;
            let rows = cursor.u32()? as usize;
            let columns = cursor.u16()? as usize;
            let mut types = Vec::with_capacity(columns);
            for code in cursor.take(columns)? {
                types.push(TypeId::from_u8(*code).ok_or_else(|| {
                    ZyronError::CdcDecoderError(format!(
                        "column-sliced change segment names an unknown physical type {code}"
                    ))
                })?);
            }
            let mut blocks = Vec::with_capacity(columns);
            for physical in &types {
                let nulls = BlockRef::read_from(&mut cursor)?;
                let values = BlockRef::read_from(&mut cursor)?;
                let offsets = if physical.fixed_size().is_none() {
                    Some(BlockRef::read_from(&mut cursor)?)
                } else {
                    None
                };
                blocks.push((nulls, values, offsets));
            }
            groups.push(GroupShape {
                epoch,
                projected,
                rows,
                types,
                blocks,
            });
        }
        drop(directory);
        for block in header_blocks.iter().chain(
            groups
                .iter()
                .flat_map(|g| g.blocks.iter())
                .flat_map(|(n, v, o)| [Some(n), Some(v), o.as_ref()])
                .flatten(),
        ) {
            if block.end > body_len || block.start > block.end {
                return Err(short_body());
            }
        }

        // The header blocks stand together at the front of the body, so
        // the ones a scan judges records by come in one read
        let heads_end = header_blocks[..8].iter().map(|b| b.end).max().unwrap_or(0);
        let front = backing.read(0, heads_end)?;
        let block_of = |at: usize| -> Result<Vec<u8>> {
            let block = &header_blocks[at];
            decode_block(&front[block.start..block.end], block)
        };
        let heads = RecordHeads {
            change_types: block_of(0)?,
            versions: bytes_to_u64s(&block_of(1)?),
            timestamps: bytes_to_u64s(&block_of(2)?)
                .into_iter()
                .map(|t| t as i64)
                .collect(),
            txn_ids: bytes_to_u64s(&block_of(3)?),
            ordinals: bytes_to_u64s(&block_of(4)?),
            group_of: bytes_to_u16s(&block_of(7)?),
        };
        let consistent = heads.change_types.len() == records
            && heads.versions.len() == records
            && heads.timestamps.len() == records
            && heads.txn_ids.len() == records
            && heads.ordinals.len() == records
            && heads.group_of.len() == records;
        if !consistent {
            return Err(ZyronError::CdcDecoderError(
                "column-sliced change segment header blocks disagree on the record count".into(),
            ));
        }
        let mut counted = vec![0usize; groups.len()];
        for group in &heads.group_of {
            if *group == NO_GROUP {
                continue;
            }
            let Some(count) = counted.get_mut(*group as usize) else {
                return Err(ZyronError::CdcDecoderError(
                    "column-sliced change segment names a group it does not hold".into(),
                ));
            };
            *count += 1;
        }
        if counted
            .iter()
            .zip(&groups)
            .any(|(count, group)| *count != group.rows)
        {
            return Err(ZyronError::CdcDecoderError(
                "column-sliced change segment records name more rows than its groups hold".into(),
            ));
        }
        Ok(Self {
            backing,
            body_len,
            header_blocks,
            heads,
            groups,
        })
    }

    pub fn heads(&self) -> &RecordHeads {
        &self.heads
    }

    pub fn groups(&self) -> &[GroupShape] {
        &self.groups
    }

    /// Fetches and decodes one block
    fn block(&self, block: &BlockRef) -> Result<Vec<u8>> {
        let stored = self.backing.read(block.start, block.end - block.start)?;
        decode_block(&stored, block)
    }

    /// Fetches one block and decodes it into `out`, which is exactly its
    /// raw length, so the bytes land where the reader keeps them
    fn block_into(&self, block: &BlockRef, out: &mut [u8]) -> Result<()> {
        if out.len() != block.raw_len {
            return Err(ZyronError::CdcDecoderError(
                "column-sliced change segment block was asked for at the wrong length".into(),
            ));
        }
        if block.codec == BLOCK_RAW {
            // A raw block's stored bytes are its values, read straight in
            match &self.backing {
                Backing::Memory(body) => {
                    if block.end > body.len() {
                        return Err(short_body());
                    }
                    out.copy_from_slice(&body[block.start..block.end]);
                }
                Backing::File { file, body_start } => {
                    read_file_at(file, *body_start + block.start as u64, out)?;
                }
            }
            if hot_hash32(out) != block.checksum {
                return Err(ZyronError::CdcDecoderError(
                    "column-sliced change segment block checksum does not match".into(),
                ));
            }
            return Ok(());
        }
        let stored = self.backing.read(block.start, block.end - block.start)?;
        if hot_hash32(&stored) != block.checksum {
            return Err(ZyronError::CdcDecoderError(
                "column-sliced change segment block checksum does not match".into(),
            ));
        }
        let codec = CdfCodec::from_u8(block.codec)?;
        let written = codec.decompress_into(&stored, out)?;
        if written != out.len() {
            return Err(ZyronError::CdcDecoderError(
                "column-sliced change segment block decoded to the wrong length".into(),
            ));
        }
        Ok(())
    }

    /// The null bits of one column of one group
    pub fn column_nulls(&self, group: usize, column: usize) -> Result<Vec<u8>> {
        let (nulls, _, _) = self.column_blocks(group, column)?;
        self.block(nulls)
    }

    /// Decodes the cells of a fixed width column straight into `out`, which
    /// is the column's width times its rows long
    pub fn column_values_into(&self, group: usize, column: usize, out: &mut [u8]) -> Result<()> {
        let (_, values, _) = self.column_blocks(group, column)?;
        let shape = &self.groups[group];
        let Some(width) = shape.types[column].fixed_size() else {
            return Err(ZyronError::CdcDecoderError(
                "column-sliced change segment variable column has no fixed cells to decode into"
                    .into(),
            ));
        };
        if out.len() != width * shape.rows {
            return Err(ZyronError::CdcDecoderError(
                "column-sliced change segment fixed column was asked for at the wrong length"
                    .into(),
            ));
        }
        self.block_into(values, out)
    }

    /// The blocks one column of one group is stored in
    fn column_blocks(
        &self,
        group: usize,
        column: usize,
    ) -> Result<&(BlockRef, BlockRef, Option<BlockRef>)> {
        let shape = self.groups.get(group).ok_or_else(|| {
            ZyronError::CdcDecoderError("column-sliced change segment group is out of range".into())
        })?;
        shape.blocks.get(column).ok_or_else(|| {
            ZyronError::CdcDecoderError(
                "column-sliced change segment column is out of range".into(),
            )
        })
    }

    /// The headers only a frame rebuild needs
    fn extras(&self) -> Result<RecordExtras> {
        let key_offsets = bytes_to_u32s(&self.block(&self.header_blocks[8])?);
        let key_bytes = self.block(&self.header_blocks[9])?;
        let records = self.heads.len();
        if key_offsets.len() != records + 1
            || key_offsets
                .last()
                .is_some_and(|end| *end as usize != key_bytes.len())
        {
            return Err(ZyronError::CdcDecoderError(
                "column-sliced change segment key offsets do not span the keys".into(),
            ));
        }
        let extras = RecordExtras {
            epochs: bytes_to_u32s(&self.block(&self.header_blocks[5])?),
            flags: self.block(&self.header_blocks[6])?,
            key_offsets,
            key_bytes,
        };
        if extras.epochs.len() != records || extras.flags.len() != records {
            return Err(ZyronError::CdcDecoderError(
                "column-sliced change segment header blocks disagree on the record count".into(),
            ));
        }
        Ok(extras)
    }

    /// The key bytes of one record
    fn key<'e>(extras: &'e RecordExtras, record: usize) -> &'e [u8] {
        let start = extras.key_offsets[record] as usize;
        let end = extras.key_offsets[record + 1] as usize;
        &extras.key_bytes[start..end]
    }

    /// Decodes one column of one group
    pub fn column(&self, group: usize, column: usize) -> Result<ColumnBlock> {
        let shape = self.groups.get(group).ok_or_else(|| {
            ZyronError::CdcDecoderError("column-sliced change segment group is out of range".into())
        })?;
        let (nulls, values, offsets) = shape.blocks.get(column).ok_or_else(|| {
            ZyronError::CdcDecoderError(
                "column-sliced change segment column is out of range".into(),
            )
        })?;
        let physical = shape.types[column];
        let nulls = self.block(nulls)?;
        let values = self.block(values)?;
        match (physical.fixed_size(), offsets) {
            (Some(width), _) => {
                if values.len() != width * shape.rows {
                    return Err(ZyronError::CdcDecoderError(
                        "column-sliced change segment fixed column holds the wrong byte count"
                            .into(),
                    ));
                }
                Ok(ColumnBlock::Fixed {
                    width,
                    nulls,
                    values,
                })
            }
            (None, Some(offsets)) => {
                let offsets = bytes_to_u32s(&self.block(offsets)?);
                // One more start than rows, ascending, ending where the
                // bytes end, so every cell a row names lies inside them
                if offsets.len() != shape.rows + 1
                    || offsets
                        .last()
                        .is_some_and(|end| *end as usize != values.len())
                    || offsets.windows(2).any(|pair| pair[0] > pair[1])
                {
                    return Err(ZyronError::CdcDecoderError(
                        "column-sliced change segment variable column offsets do not span it"
                            .into(),
                    ));
                }
                Ok(ColumnBlock::Varlen {
                    nulls,
                    offsets,
                    bytes: values,
                })
            }
            (None, None) => Err(ZyronError::CdcDecoderError(
                "column-sliced change segment variable column carries no offsets".into(),
            )),
        }
    }

    /// Rebuilds the segment's framed records, byte for byte what was
    /// written, for a reader that walks rows
    pub fn frames(&self) -> Result<Vec<u8>> {
        let extras = self.extras()?;
        let mut columns: Vec<Vec<ColumnBlock>> = Vec::with_capacity(self.groups.len());
        for (g, shape) in self.groups.iter().enumerate() {
            let mut decoded = Vec::with_capacity(shape.types.len());
            for c in 0..shape.types.len() {
                decoded.push(self.column(g, c)?);
            }
            columns.push(decoded);
        }
        let mut next_row = vec![0usize; self.groups.len()];
        let mut out = Vec::with_capacity(self.body_len * 2);
        let mut row = Vec::new();
        let heads = &self.heads;
        for record in 0..heads.len() {
            row.clear();
            let group = heads.group_of[record];
            if group != NO_GROUP {
                let g = group as usize;
                let shape = &self.groups[g];
                let r = next_row[g];
                next_row[g] += 1;
                let bitmap_len = shape.types.len().div_ceil(8);
                row.resize(bitmap_len, 0);
                for (c, block) in columns[g].iter().enumerate() {
                    if block.is_null(r) {
                        row[c / 8] |= 1 << (c % 8);
                    }
                    match block {
                        ColumnBlock::Fixed { .. } => row.extend_from_slice(block.cell(r)),
                        ColumnBlock::Varlen { .. } => {
                            let cell = block.cell(r);
                            row.extend_from_slice(&(cell.len() as u32).to_le_bytes());
                            row.extend_from_slice(cell);
                        }
                    }
                }
            }
            crate::change_feed::frame_parts(
                &mut out,
                heads.change_types[record],
                heads.versions[record],
                heads.timestamps[record],
                heads.txn_ids[record],
                heads.ordinals[record],
                extras.epochs[record],
                extras.flags[record],
                &row,
                Self::key(&extras, record),
            );
        }
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::change_feed::frame_parts;
    use std::io::Write;

    #[test]
    fn test_a_body_in_a_file_reads_the_same_as_one_in_memory() {
        let wide = layout(&[TypeId::Int64, TypeId::Text]);
        let mut layouts = RowLayouts::new();
        layouts.insert((1, false), wide.clone());
        let records: Vec<_> = (0..20u64)
            .map(|v| {
                (
                    1u8,
                    v,
                    1u32,
                    0u8,
                    row(&[Some(&(v as i64).to_le_bytes()), Some(b"cell")], &wide),
                    Vec::new(),
                )
            })
            .collect();
        let frames = frames_of(&records);
        let body = encode(&frames, &layouts, CdfCodec::Lz4)
            .expect("encodes")
            .expect("known");
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("body.bin");
        let lead = vec![7u8; 33];
        {
            let mut file = File::create(&path).expect("create");
            file.write_all(&lead).expect("lead");
            file.write_all(&body).expect("body");
            file.write_all(b"trailer").expect("trailer");
        }
        let in_file = ColumnarSegment::open(
            File::open(&path).expect("open"),
            lead.len() as u64,
            body.len(),
        )
        .expect("opens");
        let in_memory = ColumnarSegment::parse(body).expect("parses");
        assert_eq!(
            in_file.frames().expect("frames"),
            in_memory.frames().expect("frames")
        );
        assert_eq!(in_file.frames().expect("frames"), frames);
        let block = in_file.column(0, 1).expect("column");
        assert_eq!(block.cell(19), b"cell");

        // A fixed column decodes straight into a buffer of its own size,
        // from either backing, and its nulls come apart
        for segment in [&in_file, &in_memory] {
            let mut out = vec![0u8; 20 * 8];
            segment
                .column_values_into(0, 0, &mut out)
                .expect("decodes into");
            let values: Vec<i64> = out
                .chunks_exact(8)
                .map(|c| i64::from_le_bytes([c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7]]))
                .collect();
            assert_eq!(values, (0..20).collect::<Vec<i64>>());
            assert!(
                segment
                    .column_nulls(0, 0)
                    .expect("nulls")
                    .iter()
                    .all(|b| *b == 0)
            );
            let mut short = vec![0u8; 8];
            assert!(segment.column_values_into(0, 0, &mut short).is_err());
            assert!(
                segment.column_values_into(0, 1, &mut out).is_err(),
                "text has no fixed cells"
            );
        }
    }

    fn layout(types: &[TypeId]) -> Vec<TypeId> {
        types.to_vec()
    }

    /// A row of the given cells under a layout, NSM encoded
    fn row(cells: &[Option<&[u8]>], types: &[TypeId]) -> Vec<u8> {
        let mut out = vec![0u8; types.len().div_ceil(8)];
        for (i, (cell, physical)) in cells.iter().zip(types).enumerate() {
            match (cell, physical.fixed_size()) {
                (None, Some(width)) => {
                    out[i / 8] |= 1 << (i % 8);
                    out.extend(std::iter::repeat_n(0u8, width));
                }
                (None, None) => {
                    out[i / 8] |= 1 << (i % 8);
                    out.extend_from_slice(&0u32.to_le_bytes());
                }
                (Some(bytes), Some(_)) => out.extend_from_slice(bytes),
                (Some(bytes), None) => {
                    out.extend_from_slice(&(bytes.len() as u32).to_le_bytes());
                    out.extend_from_slice(bytes);
                }
            }
        }
        out
    }

    fn frames_of(records: &[(u8, u64, u32, u8, Vec<u8>, Vec<u8>)]) -> Vec<u8> {
        let mut out = Vec::new();
        for (i, (kind, version, epoch, flags, row, key)) in records.iter().enumerate() {
            frame_parts(
                &mut out,
                *kind,
                *version,
                *version as i64 * 1_000,
                7,
                i as u64,
                *epoch,
                *flags,
                row,
                key,
            );
        }
        out
    }

    #[test]
    fn test_a_sliced_segment_rebuilds_its_frames_byte_for_byte() {
        let wide = layout(&[TypeId::Int64, TypeId::Text, TypeId::Int32]);
        let narrow = layout(&[TypeId::Int64, TypeId::Int32]);
        let mut layouts = RowLayouts::new();
        layouts.insert((1, false), wide.clone());
        layouts.insert((1, true), narrow.clone());
        let records = vec![
            (
                1u8,
                10u64,
                1u32,
                0u8,
                row(
                    &[
                        Some(&7i64.to_le_bytes()),
                        Some(b"hello"),
                        Some(&3i32.to_le_bytes()),
                    ],
                    &wide,
                ),
                Vec::new(),
            ),
            (
                1,
                10,
                1,
                FLAG_PROJECTED,
                row(&[Some(&8i64.to_le_bytes()), None], &narrow),
                b"key".to_vec(),
            ),
            (
                4,
                11,
                1,
                1,
                row(&[None, None, Some(&9i32.to_le_bytes())], &wide),
                Vec::new(),
            ),
            // A truncate carries no row at all
            (5, 12, 1, 1, Vec::new(), Vec::new()),
        ];
        let frames = frames_of(&records);
        for codec in [CdfCodec::None, CdfCodec::Lz4, CdfCodec::Zstd] {
            let body = encode(&frames, &layouts, codec)
                .expect("encodes")
                .expect("every layout is known");
            let segment = ColumnarSegment::parse(body).expect("parses");
            assert_eq!(segment.heads().len(), 4);
            assert_eq!(segment.groups().len(), 2);
            assert_eq!(segment.frames().expect("rebuilds"), frames, "{codec:?}");
            // The wide group's text column reads its cells and its NULL
            let text = segment.column(0, 1).expect("column");
            assert_eq!(text.cell(0), b"hello");
            assert!(text.is_null(1));
            assert_eq!(segment.extras().expect("extras").key_bytes, b"key");
            assert!(!segment.heads().has_row(3));
        }
    }

    #[test]
    fn test_an_unknown_layout_keeps_the_segment_framed() {
        let wide = layout(&[TypeId::Int64]);
        let layouts = RowLayouts::new();
        let frames = frames_of(&[(
            1,
            1,
            1,
            0,
            row(&[Some(&1i64.to_le_bytes())], &wide),
            Vec::new(),
        )]);
        assert!(
            encode(&frames, &layouts, CdfCodec::Lz4)
                .expect("encodes")
                .is_none()
        );
    }

    #[test]
    fn test_a_row_that_does_not_walk_under_its_layout_keeps_the_segment_framed() {
        let mut layouts = RowLayouts::new();
        layouts.insert((1, false), layout(&[TypeId::Int64, TypeId::Int64]));
        // One column's bytes where the layout expects two
        let frames = frames_of(&[(
            1,
            1,
            1,
            0,
            row(&[Some(&1i64.to_le_bytes())], &[TypeId::Int64]),
            Vec::new(),
        )]);
        assert!(
            encode(&frames, &layouts, CdfCodec::Lz4)
                .expect("encodes")
                .is_none()
        );
    }

    #[test]
    fn test_a_block_that_does_not_shrink_is_stored_raw() {
        let mut layouts = RowLayouts::new();
        layouts.insert((1, false), layout(&[TypeId::Int64]));
        let mut state = 0x9E37_79B9_7F4A_7C15u64;
        let records: Vec<_> = (0..64u64)
            .map(|v| {
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                (
                    1u8,
                    v,
                    1u32,
                    0u8,
                    row(&[Some(&state.to_le_bytes())], &[TypeId::Int64]),
                    Vec::new(),
                )
            })
            .collect();
        let frames = frames_of(&records);
        let body = encode(&frames, &layouts, CdfCodec::Lz4)
            .expect("encodes")
            .expect("known");
        let segment = ColumnarSegment::parse(body).expect("parses");
        assert_eq!(segment.frames().expect("rebuilds"), frames);
        let values = segment.column(0, 0).expect("column");
        let ColumnBlock::Fixed { width, values, .. } = values else {
            panic!("a fixed column");
        };
        assert_eq!(width, 8);
        assert_eq!(values.len(), 64 * 8);
    }
}
