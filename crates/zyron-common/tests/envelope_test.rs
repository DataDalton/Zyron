//! The universal format envelope, checked end to end.
//!
//! Covers validation items 1, 2, 3, and 6 of the phase: round trip for every
//! format kind, corruption at any byte, magic uniqueness, and an unknown
//! version failing closed with an upgrade path in the message

use zyron_common::format::envelope::{
    self, ENVELOPE_FOOTER_LEN, ENVELOPE_HEADER_LEN, ENVELOPE_MIN_LEN, ENVELOPE_PEEK_LEN,
    EnvelopeError, flags,
};
use zyron_common::format::registry::{
    DeprecationStatus, FormatFixture, FormatMigrator, FormatRegistration, MigrationPolicy,
};
use zyron_common::format::stamp::{FORMAT_STAMP_LEN, FormatStamp, stamp_flags};
use zyron_common::format::version::VersionWindow;
use zyron_common::format::{
    ALL_FORMAT_KINDS, FormatKind, FormatRegistry, FormatVersion, Framing, MAGIC_ALLOCATIONS,
    RECORD_VERSION_MAX_INLINE, RECORD_VERSION_WIDE_ESCAPE, RecordVersion, WideRecordVersion,
    migration, text_envelope,
};

/// Item 1. Encode then decode preserves every field, for every format kind
#[test]
fn every_format_kind_round_trips_byte_identically() {
    for kind in ALL_FORMAT_KINDS {
        for version in [
            FormatVersion::V1,
            FormatVersion::new(1, 9),
            FormatVersion::new(11, 0),
            FormatVersion::new(u16::MAX, u16::MAX),
        ] {
            for body in [
                Vec::new(),
                b"one".to_vec(),
                (0u8..=255).cycle().take(4096).collect::<Vec<u8>>(),
            ] {
                let encoded = envelope::encode(*kind, version, &body);
                let decoded = envelope::decode(&encoded).unwrap_or_else(|e| {
                    panic!("{kind} at {version} with {} body bytes: {e}", body.len())
                });
                assert_eq!(decoded.header.kind, *kind);
                assert_eq!(decoded.header.version, version);
                assert_eq!(decoded.header.flags, 0);
                assert_eq!(decoded.header.header_length as usize, ENVELOPE_HEADER_LEN);
                assert!(decoded.extension.is_empty());
                assert_eq!(decoded.body, &body[..]);
                assert_eq!(encoded.len(), ENVELOPE_MIN_LEN + body.len());

                // Re-encoding the decoded parts reproduces the same bytes
                assert_eq!(
                    envelope::encode(decoded.header.kind, decoded.header.version, decoded.body),
                    encoded
                );
            }
        }
    }
}

/// Item 1, extension and flags. A per-format header extension survives the
/// round trip and is covered by the header checksum
#[test]
fn extensions_and_flags_round_trip() {
    let extension: Vec<u8> = (0u8..64).collect();
    let body = b"the body".to_vec();
    let encoded = envelope::encode_with(
        FormatKind::ZyrColumnar,
        FormatVersion::new(2, 3),
        flags::COMPRESSED | flags::ENCRYPTED | 0x00AB_CD00,
        &extension,
        &body,
    );
    let decoded = envelope::decode(&encoded).expect("decodes");
    assert!(decoded.header.is_compressed());
    assert!(decoded.header.is_encrypted());
    assert!(!decoded.header.is_downgrade_write());
    assert_eq!(decoded.extension, &extension[..]);
    assert_eq!(decoded.body, &body[..]);
    assert_eq!(
        decoded.header.header_length as usize,
        ENVELOPE_HEADER_LEN + extension.len()
    );
}

/// Item 2. Flipping any single byte of an envelope is caught
#[test]
fn corruption_at_any_byte_is_caught() {
    let extension: Vec<u8> = (0u8..16).collect();
    let body: Vec<u8> = (0u8..200).collect();
    let encoded = envelope::encode_with(
        FormatKind::WalSegment,
        FormatVersion::new(1, 2),
        flags::COMPRESSED,
        &extension,
        &body,
    );
    assert!(envelope::decode(&encoded).is_ok());
    for index in 0..encoded.len() {
        for bit in [0x01u8, 0x80] {
            let mut corrupted = encoded.clone();
            corrupted[index] ^= bit;
            assert!(
                envelope::decode(&corrupted).is_err(),
                "flipping bit {bit:#04x} of byte {index} was not caught"
            );
        }
    }
}

/// Item 2, truncation. Losing any suffix is caught
#[test]
fn truncation_is_caught() {
    let encoded = envelope::encode(FormatKind::LakeManifest, FormatVersion::V1, b"body bytes");
    for cut in 0..encoded.len() {
        assert!(
            envelope::decode(&encoded[..cut]).is_err(),
            "a {cut} byte prefix was accepted"
        );
    }
}

/// Item 3. No two formats share a magic, every magic maps back to its kind,
/// and every kind in the enum has an allocation
#[test]
fn magic_bytes_are_unique_and_total() {
    let mut seen = std::collections::HashSet::new();
    for row in MAGIC_ALLOCATIONS {
        assert!(
            seen.insert(row.magic),
            "magic {:?} is allocated twice",
            std::str::from_utf8(&row.magic)
        );
        assert_eq!(FormatKind::from_magic(row.magic), Some(row.kind));
        assert_eq!(row.magic[0], b'Z');
        assert!(!row.owner.is_empty());
        assert!(!row.doc.is_empty());
    }
    assert_eq!(MAGIC_ALLOCATIONS.len(), ALL_FORMAT_KINDS.len());
    assert_eq!(ALL_FORMAT_KINDS.len(), 29);
    for kind in ALL_FORMAT_KINDS {
        assert!(MAGIC_ALLOCATIONS.iter().any(|row| row.kind == *kind));
    }
}

/// A peek reads the kind and the version from eight bytes, whichever framing
/// the format uses, which is what makes an open cheap
#[test]
fn a_peek_identifies_every_kind_from_eight_bytes() {
    for kind in ALL_FORMAT_KINDS {
        let version = FormatVersion::new(3, 4);
        let encoded = envelope::encode(*kind, version, b"body");
        assert_eq!(
            envelope::peek(&encoded[..ENVELOPE_PEEK_LEN]).expect("peeks"),
            (*kind, version)
        );
        let stamp = FormatStamp::new(*kind, version).to_bytes();
        assert_eq!(
            envelope::peek(&stamp[..ENVELOPE_PEEK_LEN]).expect("peeks"),
            (*kind, version),
            "a stamp opens with the same eight bytes an envelope does"
        );
    }
}

/// Item 6. A version outside the reader window fails closed, naming the
/// version and the path forward
#[test]
fn an_unknown_version_fails_closed_with_an_upgrade_path() {
    let registry = one_bumped_registry(MigrationPolicy::Lazy, true);
    let bytes = envelope::encode(
        FormatKind::StatisticsFile,
        FormatVersion::new(99, 0),
        b"body",
    );
    let err = migration::open(&registry, &bytes).expect_err("refuses");
    let text = err.to_string();
    assert!(text.contains("99.0"), "{text}");
    assert!(text.contains("statistics_file"), "{text}");
    assert!(text.contains("Upgrade through"), "{text}");

    // A version ahead of the writer but inside the window is refused too,
    // which is a node meeting a file a newer peer wrote
    let ahead = envelope::encode(
        FormatKind::StatisticsFile,
        FormatVersion::new(1, 5),
        b"body",
    );
    assert!(migration::open(&registry, &ahead).is_err());
}

/// A file whose magic is not allocated is refused, naming the bytes found
#[test]
fn an_unknown_magic_fails_closed() {
    let mut bytes = envelope::encode(FormatKind::HeapPage, FormatVersion::V1, b"body");
    bytes[0..4].copy_from_slice(b"QQQQ");
    match envelope::decode(&bytes) {
        Err(EnvelopeError::UnknownMagic { magic }) => {
            assert_eq!(&magic, b"QQQQ");
            let text = EnvelopeError::UnknownMagic { magic }.to_string();
            assert!(text.contains("QQQQ"), "{text}");
            assert!(text.contains("no registered format"), "{text}");
        }
        other => panic!("expected UnknownMagic, got {other:?}"),
    }
}

/// A file of one kind read as another is refused before its body is touched
#[test]
fn a_foreign_file_is_refused_by_kind() {
    let bytes = envelope::encode(FormatKind::RaftLog, FormatVersion::V1, b"body");
    let err = envelope::decode_as(&bytes, FormatKind::WalSegment).expect_err("refuses");
    let text = err.to_string();
    assert!(text.contains("ZWAL"), "{text}");
    assert!(text.contains("ZRAF"), "{text}");
}

/// The stamp framing round trips for every format that uses it, and its
/// blank form is told apart from a corrupted one
#[test]
fn the_stamp_framing_round_trips() {
    for kind in ALL_FORMAT_KINDS
        .iter()
        .copied()
        .filter(|kind| kind.framing() == Framing::Stamp)
    {
        let stamp = FormatStamp {
            kind,
            version: FormatVersion::new(2, 7),
            flags: stamp_flags::COMPRESSED,
        };
        let bytes = stamp.to_bytes();
        assert_eq!(bytes.len(), FORMAT_STAMP_LEN);
        assert_eq!(bytes, stamp.to_bytes_const());
        let read = FormatStamp::from_bytes(&bytes).expect("reads back");
        assert_eq!(read, stamp);
        assert!(read.is_compressed());
        assert!(!FormatStamp::is_blank(&bytes));
    }
    assert!(FormatStamp::is_blank(&[0u8; FORMAT_STAMP_LEN]));
}

/// The text framing round trips for the two hand-editable files
#[test]
fn the_text_framing_round_trips() {
    for kind in ALL_FORMAT_KINDS
        .iter()
        .copied()
        .filter(|kind| kind.framing() == Framing::Text)
    {
        let version = FormatVersion::new(1, 3);
        let document = text_envelope::with_header(kind, version, "[server]\nport = 5433\n");
        assert_eq!(
            text_envelope::parse(&document).expect("parses"),
            (kind, version)
        );
        assert_eq!(
            text_envelope::parse_as(&document, kind, version).expect("matches"),
            version
        );
        assert!(document.contains("port = 5433"));
    }
}

/// Record version tags refuse both reserved values and order against each
/// other. Zero marks an unwritten slot and 255 escapes to a wide tag, so the
/// usable inline range is 1 through 254
#[test]
fn record_version_tags_reject_the_reserved_values() {
    assert!(RecordVersion::new(0).is_none());
    assert!(WideRecordVersion::new(0).is_none());
    assert!(RecordVersion::read(&[0]).is_err());
    assert!(RecordVersion::read(&[]).is_err());
    assert!(RecordVersion::new(RECORD_VERSION_WIDE_ESCAPE).is_none());
    assert!(RecordVersion::read(&[RECORD_VERSION_WIDE_ESCAPE]).is_err());
    for raw in 1u8..=RECORD_VERSION_MAX_INLINE {
        let tag = RecordVersion::new(raw).expect("usable");
        assert_eq!(RecordVersion::read(&[raw]).expect("reads"), tag);
    }
    let wide = WideRecordVersion::new(1_000).expect("nonzero");
    assert_eq!(
        WideRecordVersion::read(&wide.to_le_bytes()).expect("reads"),
        wide
    );
}

/// The envelope's fixed sizes are what every retrofit assumed
#[test]
fn the_envelope_dimensions_are_fixed() {
    assert_eq!(ENVELOPE_HEADER_LEN, 20);
    assert_eq!(ENVELOPE_FOOTER_LEN, 4);
    assert_eq!(ENVELOPE_MIN_LEN, 24);
    assert_eq!(ENVELOPE_PEEK_LEN, 8);
    assert_eq!(FORMAT_STAMP_LEN, 9);
}

fn append_marker(body: &[u8]) -> Result<Vec<u8>, String> {
    let mut out = body.to_vec();
    out.extend_from_slice(b"-v2");
    Ok(out)
}

fn strip_marker(body: &[u8]) -> Result<Vec<u8>, String> {
    match body.strip_suffix(b"-v2") {
        Some(rest) => Ok(rest.to_vec()),
        None => Err("no v2 marker".to_string()),
    }
}

/// A registry where the statistics format has two versions
fn one_bumped_registry(policy: MigrationPolicy, reversible: bool) -> FormatRegistry {
    let registrations: Vec<FormatRegistration> = ALL_FORMAT_KINDS
        .iter()
        .copied()
        .map(|kind| {
            let bumped = kind == FormatKind::StatisticsFile;
            FormatRegistration {
                kind,
                writer_current_version: if bumped {
                    FormatVersion::new(1, 1)
                } else {
                    FormatVersion::V1
                },
                reader_supported_versions: if bumped {
                    VersionWindow::new(FormatVersion::V1, FormatVersion::new(1, 1))
                } else {
                    VersionWindow::single(FormatVersion::V1)
                },
                migration_policy: if bumped {
                    policy
                } else {
                    MigrationPolicy::Lazy
                },
                migration_reversible: reversible,
                binary_version_gate: "0.11.0",
                deprecation_status: DeprecationStatus::Active,
                retirement_date: if bumped { Some("2027-06-01") } else { None },
                downgrade_write_supported: false,
                notes: "test",
            }
        })
        .collect();
    let migrators = [FormatMigrator {
        kind: FormatKind::StatisticsFile,
        from: FormatVersion::V1,
        to: FormatVersion::new(1, 1),
        reversible,
        forward: append_marker,
        backward: if reversible { Some(strip_marker) } else { None },
        no_body_change: false,
        description: "appends the v2 marker",
    }];
    let fixtures = [FormatFixture {
        kind: FormatKind::StatisticsFile,
        version: FormatVersion::V1,
        bytes: b"",
        path: "fixtures/v1.bin",
    }];
    FormatRegistry::from_parts(&registrations, &migrators, &fixtures).expect("loads")
}

/// Item 5. A v1 file dispatches to the v1 reader and a current one to the
/// current reader, and the dispatch says which it took
#[test]
fn reader_dispatch_picks_the_version_the_file_carries() {
    let registry = one_bumped_registry(MigrationPolicy::Lazy, true);

    let current = envelope::encode(
        FormatKind::StatisticsFile,
        FormatVersion::new(1, 1),
        b"body-v2",
    );
    let opened = migration::open(&registry, &current).expect("opens");
    assert_eq!(opened.path, migration::ReaderPath::Current);
    assert_eq!(
        opened.path.dispatched_version(FormatVersion::new(1, 1)),
        FormatVersion::new(1, 1)
    );
    assert_eq!(opened.body.as_ref(), b"body-v2");

    let old = envelope::encode(FormatKind::StatisticsFile, FormatVersion::V1, b"body");
    let opened = migration::open(&registry, &old).expect("opens");
    assert!(opened.path.needs_migration());
    assert_eq!(
        opened.path.dispatched_version(FormatVersion::new(1, 1)),
        FormatVersion::V1
    );
    assert_eq!(opened.body.as_ref(), b"body-v2");
}

/// Item 7. Opening an old file triggers the migration, and the policy
/// decides whether the migrated bytes are written back
#[test]
fn migration_runs_on_open_and_the_policy_decides_the_write_back() {
    for (policy, on_modify, on_read) in [
        (MigrationPolicy::Lazy, true, false),
        (MigrationPolicy::Eager, true, true),
        (MigrationPolicy::Coexist, false, false),
    ] {
        let registry = one_bumped_registry(policy, true);
        let old = envelope::encode(FormatKind::StatisticsFile, FormatVersion::V1, b"body");
        let opened = migration::open(&registry, &old).expect("opens");
        assert_eq!(
            opened.body.as_ref(),
            b"body-v2",
            "{policy} still migrates on read"
        );
        assert_eq!(
            opened.should_write_back(true),
            on_modify,
            "{policy} on modify"
        );
        assert_eq!(opened.should_write_back(false), on_read, "{policy} on read");

        // Whatever the policy, re-encoding produces a file at the current
        // version that opens with no migration
        let reencoded = opened.reencode();
        let reopened = migration::open(&registry, &reencoded).expect("opens");
        assert_eq!(reopened.path, migration::ReaderPath::Current);
        assert_eq!(reopened.body.as_ref(), b"body-v2");
    }
}

/// Item 8. A fixture parses, migrates to the current version, and the result
/// round trips through the writer
#[test]
fn a_fixture_migrates_forward_and_round_trips() {
    let registry = one_bumped_registry(MigrationPolicy::Eager, true);
    let entry = registry
        .get(FormatKind::StatisticsFile)
        .expect("registered");

    // The fixture as an older writer would have produced it
    let fixture = envelope::encode(
        FormatKind::StatisticsFile,
        FormatVersion::V1,
        b"fixture body",
    );
    let opened = migration::open(&registry, &fixture).expect("opens");
    assert_eq!(opened.version, FormatVersion::V1);
    assert_eq!(opened.body.as_ref(), b"fixture body-v2");

    let written = opened.reencode();
    let read_back = migration::open(&registry, &written).expect("opens");
    assert_eq!(read_back.version, FormatVersion::new(1, 1));
    assert_eq!(read_back.body.as_ref(), b"fixture body-v2");

    // And the migration is reversible, so the fixture can be reproduced
    let back = migration::migrate_body_backward(
        entry,
        FormatVersion::new(1, 1),
        FormatVersion::V1,
        &read_back.body,
    )
    .expect("undoes");
    assert_eq!(back.body, b"fixture body");
}

/// A one-way migration blocks the reverse and names the step
#[test]
fn a_one_way_migration_blocks_the_reverse() {
    let registry = one_bumped_registry(MigrationPolicy::Lazy, false);
    let entry = registry
        .get(FormatKind::StatisticsFile)
        .expect("registered");
    assert!(!entry.reversible_from(FormatVersion::V1));
    let err = migration::migrate_body_backward(
        entry,
        FormatVersion::new(1, 1),
        FormatVersion::V1,
        b"body-v2",
    )
    .expect_err("blocked");
    let text = err.to_string();
    assert!(text.contains("one way"), "{text}");
    assert!(text.contains("appends the v2 marker"), "{text}");
    assert!(text.contains("backup snapshot"), "{text}");
}
