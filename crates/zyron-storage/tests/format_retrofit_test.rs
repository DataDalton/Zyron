//! Per-format retrofit checks for the storage engine.
//!
//! Covers validation item 9 for the formats this crate owns: each one is
//! shown carrying its envelope or its stamp, at the version its registration
//! declares, with corruption caught. Every check names one format, so
//! removing the envelope from one of them fails that check alone

use zyron_common::format::{
    ALL_FORMAT_KINDS, FormatKind, FormatRegistration, FormatVersion, Framing, envelope,
    stamp::FormatStamp,
};
use zyron_common::page::{
    BTREE_PAGE_FORMAT_VERSION, FSM_PAGE_FORMAT_VERSION, HEAP_PAGE_FORMAT_VERSION, PAGE_SIZE,
    PageHeader, PageId, PageType, stamp_page_checksum, verify_page_checksum,
};

/// The registration this binary submitted for one kind
fn registration(kind: FormatKind) -> FormatRegistration {
    *inventory::iter::<FormatRegistration>
        .into_iter()
        .find(|r| r.kind == kind)
        .unwrap_or_else(|| panic!("`{kind}` submitted no registration"))
}

/// Every format this crate owns, with the module that writes it
const STORAGE_FORMATS: &[(FormatKind, &str)] = &[
    (FormatKind::HeapPage, "heap::page"),
    (FormatKind::BTreeInternal, "btree::page"),
    (FormatKind::BTreeLeaf, "btree::page"),
    (FormatKind::Fsm, "freespace"),
    (FormatKind::ZyrColumnar, "columnar::file"),
    (FormatKind::Checkpoint, "btree::checkpoint"),
    (FormatKind::BloomFilter, "columnar::bloom"),
    (FormatKind::MvccClog, "txn::status_map"),
];

/// Item 9. Every storage format registers exactly once, with a window that
/// holds the version it writes
#[test]
fn every_storage_format_is_registered_once() {
    for (kind, owner) in STORAGE_FORMATS {
        let count = inventory::iter::<FormatRegistration>
            .into_iter()
            .filter(|r| r.kind == *kind)
            .count();
        assert_eq!(
            count, 1,
            "`{kind}` from {owner} submitted {count} registrations"
        );
        let registration = registration(*kind);
        assert!(
            registration
                .reader_supported_versions
                .contains(registration.writer_current_version),
            "`{kind}` writes a version it cannot read"
        );
        assert!(!registration.binary_version_gate.is_empty());
        assert!(!registration.notes.is_empty());
    }
}

/// Item 9, heap pages. A heap page carries the ZHEP stamp at the registered
/// version, and the page checksum covers it
#[test]
fn a_heap_page_carries_its_stamp() {
    assert_page_stamp(
        PageType::Heap,
        FormatKind::HeapPage,
        HEAP_PAGE_FORMAT_VERSION,
    );
}

/// Item 9, B+tree internal pages
#[test]
fn a_btree_internal_page_carries_its_stamp() {
    assert_page_stamp(
        PageType::BTreeInternal,
        FormatKind::BTreeInternal,
        BTREE_PAGE_FORMAT_VERSION,
    );
}

/// Item 9, B+tree leaf pages
#[test]
fn a_btree_leaf_page_carries_its_stamp() {
    assert_page_stamp(
        PageType::BTreeLeaf,
        FormatKind::BTreeLeaf,
        BTREE_PAGE_FORMAT_VERSION,
    );
}

/// Item 9, free space map pages
#[test]
fn an_fsm_page_carries_its_stamp() {
    assert_page_stamp(
        PageType::FreeSpaceMap,
        FormatKind::Fsm,
        FSM_PAGE_FORMAT_VERSION,
    );
}

/// One page type's stamp, from construction through serialization and back,
/// with the checksum shown to cover it
fn assert_page_stamp(page_type: PageType, kind: FormatKind, version: FormatVersion) {
    let registration = registration(kind);
    assert_eq!(
        registration.writer_current_version, version,
        "`{kind}` stamps a version its registration does not declare"
    );

    let header = PageHeader::new(PageId::new(4, 9), page_type);
    let stamp = header
        .stamp()
        .unwrap_or_else(|| panic!("{kind} is unstamped"));
    assert_eq!(stamp.kind, kind);
    assert_eq!(stamp.version, version);
    assert_eq!(header.format_kind_version(), Some((kind, version)));

    // The stamp survives the header's own serialization
    let bytes = header.to_bytes();
    assert_eq!(&bytes[31..35], &kind.magic());
    let read = PageHeader::from_bytes(&bytes);
    assert_eq!(read.stamp(), Some(stamp));

    // And the page checksum covers every byte of it, so no separate
    // checksum is needed for a page-resident format
    let mut page = Box::new([0u8; PAGE_SIZE]);
    page[..PageHeader::SIZE].copy_from_slice(&bytes);
    stamp_page_checksum(&mut page);
    verify_page_checksum(&page, header.page_id).expect("verifies");
    for index in 31..40 {
        let mut corrupted = page.clone();
        corrupted[index] ^= 0x01;
        assert!(
            verify_page_checksum(&corrupted, header.page_id).is_err(),
            "{kind} stamp byte {index} is not covered by the page checksum"
        );
    }
}

/// Item 9, the columnar file. Its header is an envelope at the registered
/// version, and flipping any header byte is caught
#[test]
fn a_columnar_file_header_is_an_envelope() {
    use zyron_storage::columnar::{SortOrder, ZyrFileHeader};

    let registration = registration(FormatKind::ZyrColumnar);
    let header = ZyrFileHeader {
        format_version: registration.writer_current_version,
        column_count: 5,
        row_count: 4_096,
        table_id: 11,
        xmin_range_lo: 1,
        xmin_range_hi: 2,
        xmax_range_lo: 3,
        xmax_range_hi: 4,
        primary_key_column_id: 0,
        sort_order: SortOrder::None,
        segment_index_offset: 1_024,
        segment_index_size: 64,
    };
    let bytes = header.to_bytes();
    let (kind, version) = envelope::peek(&bytes).expect("peeks");
    assert_eq!(kind, FormatKind::ZyrColumnar);
    assert_eq!(version, registration.writer_current_version);

    let read = ZyrFileHeader::from_bytes(&bytes).expect("reads back");
    assert_eq!(read.column_count, 5);
    assert_eq!(read.row_count, 4_096);
    assert_eq!(read.table_id, 11);
    assert_eq!(read.segment_index_offset, 1_024);
    assert_eq!(read.segment_index_size, 64);

    for index in 0..128 {
        let mut corrupted = bytes;
        corrupted[index] ^= 0x01;
        assert!(
            ZyrFileHeader::from_bytes(&corrupted).is_err(),
            "columnar header byte {index} is not covered"
        );
    }
}

/// Item 9, the bloom filter. Its serialization opens with the ZBLM stamp and
/// a filter tagged with another version proves nothing absent
#[test]
fn a_bloom_filter_carries_its_stamp() {
    use zyron_storage::columnar::{BloomFilter, might_contain_serialized};

    let registration = registration(FormatKind::BloomFilter);
    let mut filter = BloomFilter::new(128);
    filter.insert(b"present");
    let serialized = filter.to_bytes();

    let stamp = FormatStamp::from_bytes(&serialized).expect("stamped");
    assert_eq!(stamp.kind, FormatKind::BloomFilter);
    assert_eq!(stamp.version, registration.writer_current_version);

    let read = BloomFilter::from_bytes(&serialized).expect("reads back");
    assert!(read.might_contain(b"present"));
    assert!(might_contain_serialized(&serialized, b"present"));

    // A filter stamped with a version this binary does not write is refused
    // on load and prunes nothing on probe, which is the conservative answer
    let mut bumped = serialized.clone();
    let ahead = FormatStamp::new(
        FormatKind::BloomFilter,
        FormatVersion::new(
            registration.writer_current_version.major,
            registration.writer_current_version.minor + 1,
        ),
    );
    bumped[..9].copy_from_slice(&ahead.to_bytes());
    assert!(BloomFilter::from_bytes(&bumped).is_err());
    assert!(
        might_contain_serialized(&bumped, b"never inserted"),
        "an unreadable filter prunes nothing"
    );
}

/// Item 9, the commit log. Its file is a whole envelope, and a file of
/// another format is refused rather than read
#[test]
fn the_commit_log_file_is_an_envelope() {
    use zyron_storage::txn::TxnStatusMap;

    let registration = registration(FormatKind::MvccClog);
    let dir = tempfile::tempdir().expect("tempdir");

    let map = TxnStatusMap::new();
    map.record_committed(7);
    map.record_aborted(9);
    map.persist(dir.path()).expect("persists");

    let path = dir.path().join(".zyclog");
    let bytes = std::fs::read(&path).expect("reads");
    let (kind, version) = envelope::peek(&bytes).expect("peeks");
    assert_eq!(kind, FormatKind::MvccClog);
    assert_eq!(version, registration.writer_current_version);
    envelope::decode_as(&bytes, FormatKind::MvccClog).expect("both checksums verify");

    let restored = TxnStatusMap::new();
    restored.load(dir.path()).expect("loads");
    assert!(restored.is_committed(7));
    assert!(restored.is_aborted(9));

    // A file of another format in the same place is refused, naming it
    std::fs::write(
        &path,
        envelope::encode(FormatKind::HeapPage, FormatVersion::V1, b"not a clog"),
    )
    .expect("writes");
    let err = TxnStatusMap::new().load(dir.path()).expect_err("refuses");
    assert!(err.to_string().contains("expected"), "{err}");
}

/// Item 9, the commit log again. A corrupted file is refused rather than
/// partly applied, because a partly applied commit log changes which
/// transactions look committed
#[test]
fn a_corrupted_commit_log_is_refused_rather_than_partly_applied() {
    use zyron_storage::txn::TxnStatusMap;

    let dir = tempfile::tempdir().expect("tempdir");
    let map = TxnStatusMap::new();
    map.record_committed(7);
    map.persist(dir.path()).expect("persists");

    let path = dir.path().join(".zyclog");
    let mut bytes = std::fs::read(&path).expect("reads");
    let last = bytes.len() - 1;
    bytes[last] ^= 0xFF;
    std::fs::write(&path, &bytes).expect("writes");

    let restored = TxnStatusMap::new();
    let err = restored.load(dir.path()).expect_err("refuses");
    assert!(err.to_string().contains("checksum"), "{err}");
    assert!(
        !restored.is_committed(7),
        "nothing from a refused file is applied"
    );
}

/// Item 9, the index checkpoint. Its header is an envelope at the version
/// the registration declares
#[test]
fn the_index_checkpoint_declares_its_version() {
    let registration = registration(FormatKind::Checkpoint);
    assert_eq!(
        registration.writer_current_version,
        zyron_storage::format::CHECKPOINT_FORMAT_VERSION
    );
    assert_eq!(
        registration.writer_current_version,
        FormatVersion::new(11, 0),
        "the checkpoint format carries the version it reached before this phase"
    );
    assert_eq!(registration.migration_policy.label(), "eager");
}

/// The four page formats are the ones that carry a stamp rather than an
/// envelope, the columnar file carries the envelope header and a trailer
/// of its own, and the other file formats carry the whole envelope
#[test]
fn the_framings_are_what_each_format_declares() {
    for (kind, owner) in STORAGE_FORMATS {
        let framing = kind.framing();
        match kind {
            FormatKind::HeapPage
            | FormatKind::BTreeInternal
            | FormatKind::BTreeLeaf
            | FormatKind::Fsm
            | FormatKind::BloomFilter => {
                assert_eq!(framing, Framing::Stamp, "{kind} from {owner}");
            }
            FormatKind::ZyrColumnar => {
                assert_eq!(framing, Framing::OwnTrailer, "{kind} from {owner}");
            }
            _ => assert_eq!(framing, Framing::Envelope, "{kind} from {owner}"),
        }
    }
}

/// Every kind this crate registers is one the enum knows about, so a
/// registration cannot name a format the allocation table has not reserved
#[test]
fn every_storage_kind_is_in_the_allocation_table() {
    for (kind, _) in STORAGE_FORMATS {
        assert!(ALL_FORMAT_KINDS.contains(kind), "{kind} is not allocated");
        assert_eq!(FormatKind::from_magic(kind.magic()), Some(*kind));
    }
}
