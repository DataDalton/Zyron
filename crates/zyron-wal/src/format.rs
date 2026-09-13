//! The WAL segment format registration.
//!
//! A segment file carries the universal envelope in its header and a version
//! tag on every record, so one segment can hold records of more than one
//! version while a rolling upgrade is in flight. Segments migrate eagerly on
//! rotation: the writer emits the current version into every new segment, so
//! the old version leaves the directory as segments are recycled without a
//! sweep touching a live file

use zyron_common::format::registry::{DeprecationStatus, FormatRegistration, MigrationPolicy};
use zyron_common::format::version::{FormatVersion, VersionWindow};
use zyron_common::format::{FormatKind, RecordVersion};

/// The version new segment files are written at
pub const WAL_SEGMENT_FORMAT_VERSION: FormatVersion = FormatVersion::V1;

/// The version tag every record written today carries
pub const WAL_RECORD_VERSION: RecordVersion = RecordVersion::V1;

/// The same tag as the raw byte the record header holds, so the serialize
/// path writes it without a conversion
pub const WAL_RECORD_VERSION_BYTE: u8 = WAL_RECORD_VERSION.get();

inventory::submit! {
    FormatRegistration {
        kind: FormatKind::WalSegment,
        writer_current_version: WAL_SEGMENT_FORMAT_VERSION,
        reader_supported_versions: VersionWindow::single(WAL_SEGMENT_FORMAT_VERSION),
        migration_policy: MigrationPolicy::Lazy,
        migration_reversible: true,
        binary_version_gate: "0.11.0",
        deprecation_status: DeprecationStatus::Active,
        retirement_date: None,
        downgrade_write_supported: false,
        notes: "envelope header plus a per-record version tag, a new segment is written at the \
                current version and older segments are replayed then retired by checkpoint, \
                never rewritten in place. Record types are additive: a heap page change \
                carries the page id and the exact slots and bytes it wrote, and is replayed \
                onto the page image whose stamped LSN is below the record's, and a change \
                feed append carries the feed, the segment, the offset and the bytes, and is \
                laid back into the segment file before the feed reopens",
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::record::{LogRecord, LogRecordType, Lsn};
    use crate::segment::SegmentHeader;
    use bytes::Bytes;
    use zyron_common::format::envelope;

    #[test]
    fn test_segment_header_is_an_envelope() {
        let header = SegmentHeader::new(crate::segment::SegmentId(7), 1024, Lsn::new(7, 40));
        let bytes = header.to_bytes();
        let (kind, version) = envelope::peek(&bytes).expect("peeks");
        assert_eq!(kind, FormatKind::WalSegment);
        assert_eq!(version, WAL_SEGMENT_FORMAT_VERSION);
        let (parsed, extension) = envelope::decode_header(&bytes).expect("decodes");
        assert_eq!(parsed.header_length as usize, SegmentHeader::SIZE);
        assert_eq!(extension.len(), SegmentHeader::SIZE - 20);
    }

    #[test]
    fn test_segment_header_round_trips_and_validates() {
        let header = SegmentHeader::new(crate::segment::SegmentId(3), 4096, Lsn::new(3, 40));
        let bytes = header.to_bytes();
        let read = SegmentHeader::from_bytes(&bytes);
        assert_eq!(read.segment_id.0, 3);
        assert_eq!(read.segment_size, 4096);
        assert_eq!(read.first_lsn, Lsn::new(3, 40));
        assert_eq!(read.version, WAL_SEGMENT_FORMAT_VERSION);
        read.validate().expect("validates");
    }

    #[test]
    fn test_a_corrupted_header_byte_fails_validation() {
        let header = SegmentHeader::new(crate::segment::SegmentId(1), 4096, Lsn::new(1, 40));
        let bytes = header.to_bytes();
        for index in 0..SegmentHeader::SIZE {
            let mut corrupted = bytes;
            corrupted[index] ^= 0x01;
            let read = SegmentHeader::from_bytes(&corrupted);
            assert!(
                read.validate().is_err(),
                "flipping segment header byte {index} was not caught"
            );
        }
    }

    #[test]
    fn test_an_unknown_segment_version_names_the_upgrade_path() {
        let mut header = SegmentHeader::new(crate::segment::SegmentId(1), 4096, Lsn::new(1, 40));
        header.version = FormatVersion::new(9, 3);
        header.checksum = 0;
        let text = header.validate().expect_err("refuses").to_string();
        assert!(text.contains("9.3"), "{text}");
        assert!(text.contains("Upgrade through"), "{text}");
    }

    #[test]
    fn test_an_untagged_record_is_refused() {
        let mut record = LogRecord::new(
            Lsn::new(1, 40),
            Lsn::INVALID,
            1,
            LogRecordType::Insert,
            Bytes::new(),
        );
        record.record_version = 0;
        assert!(record.version().is_err());
        record.record_version = WAL_RECORD_VERSION_BYTE + 1;
        let text = record.version().expect_err("too new").to_string();
        assert!(text.contains("Upgrade through"), "{text}");
    }

    #[test]
    fn test_records_carry_a_version_tag() {
        let record = LogRecord::new(
            Lsn::new(1, 40),
            Lsn::INVALID,
            42,
            LogRecordType::Insert,
            Bytes::from_static(b"payload"),
        );
        assert_eq!(record.record_version, WAL_RECORD_VERSION_BYTE);
        let bytes = record.serialize();
        let read = LogRecord::deserialize(&bytes).expect("round trips");
        assert_eq!(read.version().expect("tagged"), WAL_RECORD_VERSION);
        assert_eq!(read.payload, record.payload);
    }
}
