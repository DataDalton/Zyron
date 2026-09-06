//! Moves a columnar file from the page-padded layout to the aligned one.
//!
//! A 1.0 file pads its header to a page and every column segment to the
//! next page, so a file of a few short columns is mostly zeros. A 1.1 file
//! keeps the header in 512 bytes and starts each segment on a 64-byte
//! boundary. Nothing inside a segment changes and the segment index still
//! records every offset and size, so the move is a repack: each segment is
//! copied to its new offset with its own padding dropped, the index and
//! trailer are rebuilt over the new offsets, and the header is restamped at
//! the new version with the index where it now sits

use zyron_common::format::FormatKind;
use zyron_common::format::registry::{FormatFixture, FormatMigrator};

use crate::columnar::constants::{
    FILE_HEADER_SIZE, FOOTER_SIZE, SEGMENT_HEADER_SIZE, SEGMENT_INDEX_ENTRY_SIZE,
    ZONE_MAP_BATCH_SIZE, ZONE_MAP_ENTRY_SIZE, ZYR_FOOTER_SENTINEL, ZYR_FORMAT_VERSION,
    ZYR_FORMAT_VERSION_1_0,
};
use crate::columnar::file::{ZyrFileHeader, round_up_to_alignment};
use crate::columnar::segment::SegmentHeader;

/// A file the 0.11.0 writer produced, so the 1.0 reader and this step are
/// exercised against bytes that writer laid down rather than bytes the
/// current writer round-tripped
static FIXTURE_1_0: &[u8] = include_bytes!("../fixtures/v1_0.bin");

inventory::submit! {
    FormatFixture {
        kind: FormatKind::ZyrColumnar,
        version: ZYR_FORMAT_VERSION_1_0,
        bytes: FIXTURE_1_0,
        path: "crates/zyron-storage/src/columnar/fixtures/v1_0.bin",
    }
}

inventory::submit! {
    FormatMigrator {
        kind: FormatKind::ZyrColumnar,
        from: ZYR_FORMAT_VERSION_1_0,
        to: ZYR_FORMAT_VERSION,
        reversible: false,
        forward: zyr_1_0_to_1_1,
        backward: None,
        no_body_change: false,
        description: "column segments packed on a 64-byte alignment behind a 512-byte header, in place of a page for each",
    }
}

/// Repacks a 1.0 file into the 1.1 layout, the whole file in and the whole
/// file out.
///
/// The header, the index, the trailer and every segment header are
/// verified before a byte is copied, so a damaged file is refused rather
/// than repacked into a damaged one
pub fn zyr_1_0_to_1_1(file: &[u8]) -> Result<Vec<u8>, String> {
    if file.len() < FILE_HEADER_SIZE + FOOTER_SIZE {
        return Err(format!(
            "{} bytes is shorter than a columnar header and trailer",
            file.len()
        ));
    }
    let mut head = [0u8; FILE_HEADER_SIZE];
    head.copy_from_slice(&file[..FILE_HEADER_SIZE]);
    let mut header = ZyrFileHeader::from_bytes(&head).map_err(|e| e.to_string())?;
    if header.format_version != ZYR_FORMAT_VERSION_1_0 {
        return Err(format!(
            "at version {}, this step moves {} forward",
            header.format_version, ZYR_FORMAT_VERSION_1_0
        ));
    }

    let index_offset = header.segment_index_offset as usize;
    let index_size = header.segment_index_size as usize;
    if !index_size.is_multiple_of(SEGMENT_INDEX_ENTRY_SIZE) {
        return Err(format!(
            "segment index of {index_size} bytes is not a whole number of entries"
        ));
    }
    let trailer_start = index_offset
        .checked_add(index_size)
        .ok_or_else(|| "segment index runs past the file".to_string())?;
    if index_offset < FILE_HEADER_SIZE || trailer_start + FOOTER_SIZE > file.len() {
        return Err(format!(
            "segment index at {index_offset} of {index_size} bytes does not fit a {} byte file",
            file.len()
        ));
    }
    let index = &file[index_offset..trailer_start];
    let trailer = &file[trailer_start..trailer_start + FOOTER_SIZE];
    let trailer_offset = u64::from_le_bytes(
        trailer[0..8]
            .try_into()
            .map_err(|_| "trailer offset unreadable".to_string())?,
    );
    if trailer_offset as usize != index_offset {
        return Err(format!(
            "header places the segment index at {index_offset}, the trailer at {trailer_offset}"
        ));
    }
    if trailer[8..16] != ZYR_FOOTER_SENTINEL {
        return Err("trailer sentinel missing".to_string());
    }
    let stored = u32::from_le_bytes(
        trailer[16..20]
            .try_into()
            .map_err(|_| "trailer checksum unreadable".to_string())?,
    );
    let computed = zyron_common::hash32(index);
    if stored != computed {
        return Err(format!(
            "segment index checksum mismatch, stored {stored:#010x} computed {computed:#010x}"
        ));
    }

    let row_count = header.row_count as usize;
    let zones = row_count.div_ceil(ZONE_MAP_BATCH_SIZE as usize);
    let (entries, _) = index.as_chunks::<SEGMENT_INDEX_ENTRY_SIZE>();
    let mut out = vec![0u8; FILE_HEADER_SIZE];
    let mut repacked: Vec<(u32, u64, u64)> = Vec::with_capacity(entries.len());
    for entry in entries {
        let column_id = u32::from_le_bytes([entry[0], entry[1], entry[2], entry[3]]);
        let offset = u64::from_le_bytes(
            entry[4..12]
                .try_into()
                .map_err(|_| "segment offset unreadable".to_string())?,
        ) as usize;
        let size = u64::from_le_bytes(
            entry[12..20]
                .try_into()
                .map_err(|_| "segment size unreadable".to_string())?,
        ) as usize;
        let end = offset
            .checked_add(size)
            .ok_or_else(|| format!("segment for column {column_id} runs past the file"))?;
        if size < SEGMENT_HEADER_SIZE || end > index_offset {
            return Err(format!(
                "segment for column {column_id} at {offset} of {size} bytes does not fit"
            ));
        }
        let segment = &file[offset..end];
        let mut segment_head = [0u8; SEGMENT_HEADER_SIZE];
        segment_head.copy_from_slice(&segment[..SEGMENT_HEADER_SIZE]);
        let segment_header = SegmentHeader::from_bytes(&segment_head).map_err(|e| e.to_string())?;
        // The bytes a reader ever touches, the rest of the page is padding
        let null_len = if segment_header.null_count > 0 {
            row_count.div_ceil(8)
        } else {
            0
        };
        let raw_len = SEGMENT_HEADER_SIZE
            + segment_header.bloom_filter_size as usize
            + zones * ZONE_MAP_ENTRY_SIZE
            + null_len
            + segment_header.encoded_size as usize;
        if raw_len > segment.len() {
            return Err(format!(
                "segment for column {column_id} declares {raw_len} bytes in {} on disk",
                segment.len()
            ));
        }
        let new_offset = out.len() as u64;
        out.extend_from_slice(&segment[..raw_len]);
        out.resize(round_up_to_alignment(out.len()), 0);
        repacked.push((column_id, new_offset, out.len() as u64 - new_offset));
    }

    let new_index_offset = out.len() as u64;
    let mut index_hasher = zyron_common::Hasher::new();
    for (column_id, offset, size) in &repacked {
        let mut entry = [0u8; SEGMENT_INDEX_ENTRY_SIZE];
        entry[0..4].copy_from_slice(&column_id.to_le_bytes());
        entry[4..12].copy_from_slice(&offset.to_le_bytes());
        entry[12..20].copy_from_slice(&size.to_le_bytes());
        index_hasher.update(&entry);
        out.extend_from_slice(&entry);
    }
    out.extend_from_slice(&new_index_offset.to_le_bytes());
    out.extend_from_slice(&ZYR_FOOTER_SENTINEL);
    out.extend_from_slice(&index_hasher.finish32().to_le_bytes());

    // The header is the writer's own, restamped at the new version with the
    // index where it now sits
    header.format_version = ZYR_FORMAT_VERSION;
    header.segment_index_offset = new_index_offset;
    header.segment_index_size = (repacked.len() * SEGMENT_INDEX_ENTRY_SIZE) as u32;
    out[..FILE_HEADER_SIZE].copy_from_slice(&header.to_bytes());
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::columnar::file::ZyrFileReader;
    use zyron_common::format::envelope;

    /// Every column the fixture holds, decoded, so two files can be
    /// compared by what they say rather than by their bytes
    fn decoded(path: &std::path::Path) -> (u64, Vec<(Vec<u8>, Vec<u8>)>) {
        let reader = ZyrFileReader::open(path).expect("opens");
        let rows = reader.row_count() as usize;
        let id = reader.decode_column(0, rows, 8).expect("id column");
        let name = reader.decode_column(1, rows, 0).expect("name column");
        (reader.row_count(), vec![id, name])
    }

    fn written(dir: &std::path::Path, name: &str, bytes: &[u8]) -> std::path::PathBuf {
        let path = dir.join(name);
        std::fs::write(&path, bytes).expect("write");
        path
    }

    #[test]
    fn the_fixture_is_the_page_padded_layout_the_reader_still_opens() {
        let (kind, version) = envelope::peek(FIXTURE_1_0).expect("peeks");
        assert_eq!(kind, FormatKind::ZyrColumnar);
        assert_eq!(version, ZYR_FORMAT_VERSION_1_0);
        let dir = tempfile::tempdir().expect("tempdir");
        let path = written(dir.path(), "old.zyr", FIXTURE_1_0);
        let reader = ZyrFileReader::open(&path).expect("the 1.0 reader opens it");
        assert_eq!(reader.row_count(), 200);
        assert_eq!(reader.header().format_version, ZYR_FORMAT_VERSION_1_0);
        // The first segment sits a whole page in, which is the padding the
        // move removes
        assert_eq!(
            reader.header().segment_index_offset % zyron_common::page::PAGE_SIZE as u64,
            0
        );
    }

    #[test]
    fn a_padded_file_moves_forward_and_reads_the_same() {
        let moved = zyr_1_0_to_1_1(FIXTURE_1_0).expect("moves");
        let (_, version) = envelope::peek(&moved).expect("peeks");
        assert_eq!(version, ZYR_FORMAT_VERSION);
        assert!(
            moved.len() * 4 < FIXTURE_1_0.len(),
            "{} bytes moved from {}, the padding should be gone",
            moved.len(),
            FIXTURE_1_0.len()
        );
        let dir = tempfile::tempdir().expect("tempdir");
        let before = decoded(&written(dir.path(), "old.zyr", FIXTURE_1_0));
        let after = decoded(&written(dir.path(), "new.zyr", &moved));
        assert_eq!(after, before);
    }

    #[test]
    fn a_file_already_at_1_1_is_refused_by_the_step() {
        let moved = zyr_1_0_to_1_1(FIXTURE_1_0).expect("moves");
        let refused = zyr_1_0_to_1_1(&moved).expect_err("a 1.1 file is not 1.0");
        assert!(refused.contains("1.1"), "{refused}");
    }

    #[test]
    fn a_damaged_index_is_refused_rather_than_repacked() {
        let mut damaged = FIXTURE_1_0.to_vec();
        let index_offset = {
            let mut head = [0u8; FILE_HEADER_SIZE];
            head.copy_from_slice(&damaged[..FILE_HEADER_SIZE]);
            ZyrFileHeader::from_bytes(&head)
                .expect("header")
                .segment_index_offset as usize
        };
        damaged[index_offset + 5] ^= 0x5A;
        let refused = zyr_1_0_to_1_1(&damaged).expect_err("a damaged index is refused");
        assert!(refused.contains("checksum"), "{refused}");
    }
}
