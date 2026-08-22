//! Opening a .zyr file reads the header, then the index and footer together.
//!
//! The file header records where the segment index sits, so an open is a
//! file open and two reads. Getting there without the header field means
//! asking the filesystem how large the file is, reading the trailer at the
//! end to learn where the index starts, and only then reading the index:
//! five calls where three do. An open costs the same whatever the file
//! holds, so on a table of small files it is the scan.
//!
//! What has to hold is that the shorter path reads the same index the long
//! one did, and that a file whose two records of the index position
//! disagree is refused rather than read at the wrong offset.

use zyron_common::types::TypeId;
use zyron_storage::columnar::{
    BloomPolicy, ColumnSegment, FILE_HEADER_SIZE, FOOTER_SIZE, SEGMENT_INDEX_ENTRY_SIZE,
    SegmentOptions, SortOrder, ZYR_FORMAT_VERSION, ZyrFileHeader, ZyrFileReader, ZyrFileWriter,
};

/// Writes a file of `columns` Int64 columns, each holding `rows` values
fn write_file(path: &std::path::Path, columns: u32, rows: usize) {
    let header = ZyrFileHeader {
        format_version: ZYR_FORMAT_VERSION,
        column_count: columns,
        row_count: rows as u64,
        table_id: 7,
        xmin_range_lo: 0,
        xmin_range_hi: 0,
        xmax_range_lo: 0,
        xmax_range_hi: 0,
        primary_key_column_id: 0,
        sort_order: SortOrder::Asc,
        segment_index_offset: 0,
        segment_index_size: 0,
    };
    let mut writer = ZyrFileWriter::create(path, header).expect("create");
    for column in 0..columns {
        let owned: Vec<[u8; 8]> = (0..rows)
            .map(|i| ((i as i64) * (column as i64 + 1)).to_le_bytes())
            .collect();
        let values: Vec<Option<&[u8]>> = owned.iter().map(|v| Some(v.as_slice())).collect();
        let segment = ColumnSegment::build_with_options(
            column,
            TypeId::Int64,
            8,
            &values,
            SegmentOptions {
                bloom: BloomPolicy::Auto,
                exact_encoding: false,
                distinct_sketch: false,
            },
        )
        .expect("build");
        let zones: Vec<u8> = segment
            .zone_maps
            .iter()
            .flat_map(|z| z.to_bytes())
            .collect();
        let bloom = segment.bloom_filter.as_ref().map(|b| b.to_bytes());
        writer
            .write_segment(
                column,
                &segment.header.to_bytes(),
                bloom.as_deref(),
                &zones,
                &segment.null_bitmap,
                &segment.encoded_data,
            )
            .expect("write segment");
    }
    writer.finalize(false).expect("finalize");
}

/// The header names the index, the index is where it says, and the file
/// still decodes through it.
#[test]
fn the_header_records_where_the_segment_index_sits() {
    let tmp = tempfile::TempDir::new().expect("temp dir");

    for (columns, rows) in [(1u32, 1usize), (1, 4096), (3, 2048), (9, 100)] {
        let path = tmp.path().join(format!("c{columns}_r{rows}.zyr"));
        write_file(&path, columns, rows);

        let reader = ZyrFileReader::open(&path).expect("open");
        let header = reader.header();
        assert_eq!(
            header.segment_index_size as usize,
            columns as usize * SEGMENT_INDEX_ENTRY_SIZE,
            "the header has to size the index by the segments actually written"
        );
        assert!(
            header.segment_index_offset >= FILE_HEADER_SIZE as u64,
            "the index cannot start inside the header page"
        );
        assert_eq!(
            reader.segment_count(),
            columns as usize,
            "every segment written has to be reachable through the index"
        );

        // The recorded position agrees with the file on disk: the index and
        // the trailer are the last bytes of it
        let on_disk = std::fs::metadata(&path).expect("metadata").len();
        assert_eq!(
            header.segment_index_offset + header.segment_index_size as u64 + FOOTER_SIZE as u64,
            on_disk,
            "the index and footer are the tail of the file"
        );

        // And the columns still read back through that index
        for column in 0..columns {
            let (decoded, _) = reader.decode_column(column, rows, 8).expect("decode");
            assert_eq!(decoded.len(), rows * 8);
            let last =
                i64::from_le_bytes(decoded[decoded.len() - 8..].try_into().expect("last value"));
            assert_eq!(last, (rows as i64 - 1) * (column as i64 + 1));
        }
    }
}

/// The trailer keeps its own copy of the index position. Two records of one
/// fact are only worth carrying if they are compared, and a file where they
/// disagree has to be refused rather than read at whichever offset was
/// consulted first.
#[test]
fn a_header_that_disagrees_with_the_footer_is_refused() {
    let tmp = tempfile::TempDir::new().expect("temp dir");
    let path = tmp.path().join("tampered.zyr");
    write_file(&path, 2, 1024);

    let truthful = ZyrFileReader::open(&path).expect("open");
    let real_offset = truthful.header().segment_index_offset;
    let real_size = truthful.header().segment_index_size;
    drop(truthful);

    // Start the index one entry earlier and make it one entry longer, so it
    // still ends where the trailer begins. The read stays inside the file
    // and lands on the real trailer, which leaves the two records of the
    // index position as the only thing wrong with the file. Re-stamp the
    // header checksum so its own integrity check cannot be what catches it
    let mut bytes = std::fs::read(&path).expect("read");
    let entry = SEGMENT_INDEX_ENTRY_SIZE as u64;
    bytes[80..88].copy_from_slice(&(real_offset - entry).to_le_bytes());
    bytes[88..92].copy_from_slice(&(real_size + entry as u32).to_le_bytes());
    let checksum = {
        let mut h = zyron_common::Hasher::new();
        h.update(&bytes[0..12]);
        h.finish_phase();
        h.update(&bytes[16..128]);
        h.finish32()
    };
    bytes[12..16].copy_from_slice(&checksum.to_le_bytes());
    std::fs::write(&path, &bytes).expect("write");

    let message = match ZyrFileReader::open(&path) {
        Ok(_) => panic!("a file whose two index positions disagree must not open"),
        Err(e) => e.to_string(),
    };
    assert!(
        message.contains("disagrees between header and footer"),
        "the refusal has to name what disagreed, got: {message}"
    );
}

/// An index position pointing into the header page addresses bytes that are
/// not an index, so it is refused before anything is read at it.
#[test]
fn an_index_offset_inside_the_header_is_refused() {
    let tmp = tempfile::TempDir::new().expect("temp dir");
    let path = tmp.path().join("overlapping.zyr");
    write_file(&path, 1, 512);

    let mut bytes = std::fs::read(&path).expect("read");
    bytes[80..88].copy_from_slice(&64u64.to_le_bytes());
    let checksum = {
        let mut h = zyron_common::Hasher::new();
        h.update(&bytes[0..12]);
        h.finish_phase();
        h.update(&bytes[16..128]);
        h.finish32()
    };
    bytes[12..16].copy_from_slice(&checksum.to_le_bytes());
    std::fs::write(&path, &bytes).expect("write");

    let message = match ZyrFileReader::open(&path) {
        Ok(_) => panic!("an index inside the header page must not open"),
        Err(e) => e.to_string(),
    };
    assert!(
        message.contains("overlaps the file header"),
        "got: {message}"
    );
}
