//! Format kinds and the magic byte allocation registry.
//!
//! Every persistent thing Zyron writes has a format kind, and every format
//! kind owns exactly one 4-byte magic. The allocation table below is the
//! only place a magic is assigned, so two formats can never share one and a
//! file can always be identified from its first four bytes without knowing
//! which subsystem produced it

use std::fmt;

/// One persistent format Zyron writes.
///
/// The discriminant is not stored anywhere, the magic is. Reordering the
/// variants is safe, changing a magic is not
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum FormatKind {
    /// Write-ahead log segment file
    WalSegment,
    /// Heap data page
    HeapPage,
    /// B+tree internal page
    BTreeInternal,
    /// B+tree leaf page
    BTreeLeaf,
    /// Free space map page
    Fsm,
    /// Lake transaction log file
    LakeTransactionLog,
    /// Columnar `.zyr` file
    ZyrColumnar,
    /// Point-in-time snapshot file
    Snapshot,
    /// B+tree index checkpoint file
    Checkpoint,
    /// Serialized bloom filter
    BloomFilter,
    /// MVCC commit log
    MvccClog,
    /// ANALYZE statistics file
    StatisticsFile,
    /// Out-of-line extended value object
    Toast,
    /// Lake manifest file
    LakeManifest,
    /// Lake index artifact
    LakeIndex,
    /// Lake delete predicate record
    DeletePredicate,
    /// Raft log segment
    RaftLog,
    /// Raft snapshot transfer chunk stream
    SnapshotTransfer,
    /// Row replication apply log
    ReplicationApplyLog,
    /// Backup archive manifest
    BackupArchive,
    /// `zyron.toml` server configuration
    ZyronTomlConfig,
    /// Secret store persistence file
    SecretStorePersistence,
    /// Audit hash chain file
    AuditHashChain,
    /// Volume metadata file
    VolumeMetadata,
    /// Prompt registry storage file
    PromptRegistryStorage,
    /// Workflow definition on disk
    WorkflowDefinitionOnDisk,
    /// Streaming CDC checkpoint
    StreamingCdcCheckpoint,
    /// App image bundle
    AppImageBundle,
    /// Upgrade journal, the durable half of the upgrade board
    UpgradeJournal,
}

/// One row of the magic byte allocation registry
#[derive(Debug, Clone, Copy)]
pub struct MagicAllocation {
    pub kind: FormatKind,
    pub magic: [u8; 4],
    /// The subsystem that writes files of this kind
    pub owner: &'static str,
    /// One line naming what the format holds
    pub doc: &'static str,
}

/// The magic byte allocation registry.
///
/// A new format kind adds its row here and nowhere else. `MAGIC_ALLOCATIONS`
/// is checked for duplicate magics and for full coverage of the enum by the
/// tests at the bottom of this file and again by the release check
pub const MAGIC_ALLOCATIONS: &[MagicAllocation] = &[
    MagicAllocation {
        kind: FormatKind::WalSegment,
        magic: *b"ZWAL",
        owner: "zyron-wal",
        doc: "Write-ahead log segment holding versioned records",
    },
    MagicAllocation {
        kind: FormatKind::HeapPage,
        magic: *b"ZHEP",
        owner: "zyron-storage",
        doc: "Heap page holding tuples in a slotted layout",
    },
    MagicAllocation {
        kind: FormatKind::BTreeInternal,
        magic: *b"ZBPI",
        owner: "zyron-storage",
        doc: "B+tree internal page holding separator keys and child pointers",
    },
    MagicAllocation {
        kind: FormatKind::BTreeLeaf,
        magic: *b"ZBPL",
        owner: "zyron-storage",
        doc: "B+tree leaf page holding keys and row locators",
    },
    MagicAllocation {
        kind: FormatKind::Fsm,
        magic: *b"ZFSM",
        owner: "zyron-storage",
        doc: "Free space map page holding per-page free byte counts",
    },
    MagicAllocation {
        kind: FormatKind::LakeTransactionLog,
        magic: *b"ZLAK",
        owner: "zyron-lake",
        doc: "Lake transaction log entry stream per table version",
    },
    MagicAllocation {
        kind: FormatKind::ZyrColumnar,
        magic: *b"ZCOL",
        owner: "zyron-storage",
        doc: "Columnar .zyr file holding encoded column segments",
    },
    MagicAllocation {
        kind: FormatKind::Snapshot,
        magic: *b"ZSNP",
        owner: "zyron-cdc",
        doc: "Point-in-time snapshot capture of a table",
    },
    MagicAllocation {
        kind: FormatKind::Checkpoint,
        magic: *b"ZCPT",
        owner: "zyron-storage",
        doc: "B+tree index checkpoint holding a prefix-compressed key column and a locator column",
    },
    MagicAllocation {
        kind: FormatKind::BloomFilter,
        magic: *b"ZBLM",
        owner: "zyron-storage",
        doc: "Serialized bloom filter over a column segment or index term set",
    },
    MagicAllocation {
        kind: FormatKind::MvccClog,
        magic: *b"ZCLG",
        owner: "zyron-storage",
        doc: "MVCC commit log holding per-transaction status and commit LSN",
    },
    MagicAllocation {
        kind: FormatKind::StatisticsFile,
        magic: *b"ZSTS",
        owner: "zyron-catalog",
        doc: "ANALYZE output holding per-column histograms and cardinalities",
    },
    MagicAllocation {
        kind: FormatKind::Toast,
        magic: *b"ZTST",
        owner: "zyron-media",
        doc: "Out-of-line extended value object addressed by content hash",
    },
    MagicAllocation {
        kind: FormatKind::LakeManifest,
        magic: *b"ZLMI",
        owner: "zyron-lake",
        doc: "Lake manifest holding the file set, stats, and delete predicates",
    },
    MagicAllocation {
        kind: FormatKind::LakeIndex,
        magic: *b"ZLIX",
        owner: "zyron-lake",
        doc: "Lake index artifact holding per-file value to row mappings",
    },
    MagicAllocation {
        kind: FormatKind::DeletePredicate,
        magic: *b"ZDPR",
        owner: "zyron-lake",
        doc: "Predicate-based delete record readers filter live files through",
    },
    MagicAllocation {
        kind: FormatKind::RaftLog,
        magic: *b"ZRAF",
        owner: "zyron-raft",
        doc: "Raft log segment holding versioned consensus entries",
    },
    MagicAllocation {
        kind: FormatKind::SnapshotTransfer,
        magic: *b"ZSNT",
        owner: "zyron-raft",
        doc: "Raft snapshot transfer metadata and chunk stream",
    },
    MagicAllocation {
        kind: FormatKind::ReplicationApplyLog,
        magic: *b"ZRAL",
        owner: "zyron-executor",
        doc: "Row replication apply log holding versioned changeset entries",
    },
    MagicAllocation {
        kind: FormatKind::BackupArchive,
        magic: *b"ZBAK",
        owner: "zyron-server",
        doc: "Backup archive manifest holding the file set and checksums",
    },
    MagicAllocation {
        kind: FormatKind::ZyronTomlConfig,
        magic: *b"ZCFG",
        owner: "zyron-server",
        doc: "Server configuration file carrying a declared format version",
    },
    MagicAllocation {
        kind: FormatKind::SecretStorePersistence,
        magic: *b"ZSEC",
        owner: "zyron-auth",
        doc: "Secret store persistence holding wrapped credential material",
    },
    MagicAllocation {
        kind: FormatKind::AuditHashChain,
        magic: *b"ZAUD",
        owner: "zyron-server",
        doc: "Audit hash chain holding tamper-evident event records",
    },
    MagicAllocation {
        kind: FormatKind::VolumeMetadata,
        magic: *b"ZVOL",
        owner: "zyron-server",
        doc: "Volume metadata describing an arbitrary-file storage volume",
    },
    MagicAllocation {
        kind: FormatKind::PromptRegistryStorage,
        magic: *b"ZPMT",
        owner: "zyron-server",
        doc: "Prompt registry storage holding versioned prompt bodies",
    },
    MagicAllocation {
        kind: FormatKind::WorkflowDefinitionOnDisk,
        magic: *b"ZWFD",
        owner: "zyron-server",
        doc: "Workflow definition holding the task graph and its schedule",
    },
    MagicAllocation {
        kind: FormatKind::StreamingCdcCheckpoint,
        magic: *b"ZCDC",
        owner: "zyron-cdc",
        doc: "Streaming CDC checkpoint holding per-stream progress offsets",
    },
    MagicAllocation {
        kind: FormatKind::AppImageBundle,
        magic: *b"ZAIB",
        owner: "zyron-server",
        doc: "App image bundle holding layers and its signed attestation",
    },
    MagicAllocation {
        kind: FormatKind::UpgradeJournal,
        magic: *b"ZUPJ",
        owner: "zyron-server",
        doc: "Upgrade journal holding the settings, history, rewrite queue and any restart in progress",
    },
];

/// Every format kind, in allocation order
pub const ALL_FORMAT_KINDS: &[FormatKind] = &[
    FormatKind::WalSegment,
    FormatKind::HeapPage,
    FormatKind::BTreeInternal,
    FormatKind::BTreeLeaf,
    FormatKind::Fsm,
    FormatKind::LakeTransactionLog,
    FormatKind::ZyrColumnar,
    FormatKind::Snapshot,
    FormatKind::Checkpoint,
    FormatKind::BloomFilter,
    FormatKind::MvccClog,
    FormatKind::StatisticsFile,
    FormatKind::Toast,
    FormatKind::LakeManifest,
    FormatKind::LakeIndex,
    FormatKind::DeletePredicate,
    FormatKind::RaftLog,
    FormatKind::SnapshotTransfer,
    FormatKind::ReplicationApplyLog,
    FormatKind::BackupArchive,
    FormatKind::ZyronTomlConfig,
    FormatKind::SecretStorePersistence,
    FormatKind::AuditHashChain,
    FormatKind::VolumeMetadata,
    FormatKind::PromptRegistryStorage,
    FormatKind::WorkflowDefinitionOnDisk,
    FormatKind::StreamingCdcCheckpoint,
    FormatKind::AppImageBundle,
    FormatKind::UpgradeJournal,
];

impl FormatKind {
    /// The magic this kind writes into every file it produces
    #[inline]
    pub const fn magic(self) -> [u8; 4] {
        // A const linear walk rather than a table index, so the enum and the
        // allocation table cannot drift apart silently
        let mut i = 0;
        while i < MAGIC_ALLOCATIONS.len() {
            if MAGIC_ALLOCATIONS[i].kind as u8 == self as u8 {
                return MAGIC_ALLOCATIONS[i].magic;
            }
            i += 1;
        }
        // Unreachable while the coverage test passes, and a distinctive value
        // rather than a panic in a const fn
        *b"????"
    }

    /// The kind that owns a magic, or None when the bytes address no format
    #[inline]
    pub fn from_magic(magic: [u8; 4]) -> Option<FormatKind> {
        MAGIC_ALLOCATIONS
            .iter()
            .find(|row| row.magic == magic)
            .map(|row| row.kind)
    }

    /// The registry row for this kind
    pub fn allocation(self) -> &'static MagicAllocation {
        MAGIC_ALLOCATIONS
            .iter()
            .find(|row| row.kind == self)
            .unwrap_or(&MAGIC_ALLOCATIONS[0])
    }

    /// The subsystem that writes this format
    pub fn owner(self) -> &'static str {
        self.allocation().owner
    }

    /// One line naming what the format holds
    pub fn doc(self) -> &'static str {
        self.allocation().doc
    }

    /// The magic as text, which is what the catalog views and the CLI print
    pub fn magic_str(self) -> &'static str {
        let allocation = self.allocation();
        // Every allocated magic is four ASCII bytes, checked by the tests
        std::str::from_utf8(&allocation.magic).unwrap_or("????")
    }

    /// The catalog name of this kind, lowercase with underscores, which is
    /// what `SHOW FORMAT MIGRATIONS FOR FORMAT <kind>` accepts
    pub const fn catalog_name(self) -> &'static str {
        match self {
            FormatKind::WalSegment => "wal_segment",
            FormatKind::HeapPage => "heap_page",
            FormatKind::BTreeInternal => "btree_internal",
            FormatKind::BTreeLeaf => "btree_leaf",
            FormatKind::Fsm => "fsm",
            FormatKind::LakeTransactionLog => "lake_transaction_log",
            FormatKind::ZyrColumnar => "zyr_columnar",
            FormatKind::Snapshot => "snapshot",
            FormatKind::Checkpoint => "checkpoint",
            FormatKind::BloomFilter => "bloom_filter",
            FormatKind::MvccClog => "mvcc_clog",
            FormatKind::StatisticsFile => "statistics_file",
            FormatKind::Toast => "toast",
            FormatKind::LakeManifest => "lake_manifest",
            FormatKind::LakeIndex => "lake_index",
            FormatKind::DeletePredicate => "delete_predicate",
            FormatKind::RaftLog => "raft_log",
            FormatKind::SnapshotTransfer => "snapshot_transfer",
            FormatKind::ReplicationApplyLog => "replication_apply_log",
            FormatKind::BackupArchive => "backup_archive",
            FormatKind::ZyronTomlConfig => "zyron_toml_config",
            FormatKind::SecretStorePersistence => "secret_store_persistence",
            FormatKind::AuditHashChain => "audit_hash_chain",
            FormatKind::VolumeMetadata => "volume_metadata",
            FormatKind::PromptRegistryStorage => "prompt_registry_storage",
            FormatKind::WorkflowDefinitionOnDisk => "workflow_definition_on_disk",
            FormatKind::StreamingCdcCheckpoint => "streaming_cdc_checkpoint",
            FormatKind::AppImageBundle => "app_image_bundle",
            FormatKind::UpgradeJournal => "upgrade_journal",
        }
    }

    /// Resolves a catalog name back to its kind, case insensitively
    pub fn from_catalog_name(name: &str) -> Option<FormatKind> {
        ALL_FORMAT_KINDS
            .iter()
            .copied()
            .find(|kind| kind.catalog_name().eq_ignore_ascii_case(name))
    }

    /// True when this kind lives inside a fixed-size page rather than in a
    /// file of its own, which decides whether it carries the full envelope
    /// or the compact page stamp
    #[inline]
    pub const fn is_page_resident(self) -> bool {
        matches!(
            self,
            FormatKind::HeapPage
                | FormatKind::BTreeInternal
                | FormatKind::BTreeLeaf
                | FormatKind::Fsm
        )
    }

    /// How this kind carries its identity on disk.
    ///
    /// The one authoritative mapping. The documentation view, the CLI's file
    /// inspector, and the release check all read it, so a format cannot be
    /// described one way and read another
    pub const fn framing(self) -> Framing {
        match self {
            // Fixed-size records inside a container that checksums them
            FormatKind::HeapPage
            | FormatKind::BTreeInternal
            | FormatKind::BTreeLeaf
            | FormatKind::Fsm
            | FormatKind::BloomFilter
            | FormatKind::Toast => Framing::Stamp,
            // Hand-editable text, so the envelope is a declared section
            FormatKind::ZyronTomlConfig | FormatKind::BackupArchive => Framing::Text,
            // Records inside a stream, versioned per record
            FormatKind::AuditHashChain
            | FormatKind::ReplicationApplyLog
            | FormatKind::DeletePredicate
            | FormatKind::LakeTransactionLog => Framing::RecordTag,
            // An envelope header on a file whose trailer and checksums are
            // the format's own, so the substrate reads the header and moves
            // the file forward whole
            FormatKind::ZyrColumnar | FormatKind::LakeManifest => Framing::OwnTrailer,
            _ => Framing::Envelope,
        }
    }
}

/// How a format carries its identity on disk
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Framing {
    /// A 20-byte header and a 4-byte body checksum, in a file of its own
    Envelope,
    /// A 20-byte envelope header on a file that ends in a trailer of its
    /// own, checksummed by the format rather than by the envelope
    OwnTrailer,
    /// A 9-byte stamp inside a container that checksums it
    Stamp,
    /// A `[format]` section in a hand-editable text file
    Text,
    /// A version tag on each record inside a stream
    RecordTag,
}

impl Framing {
    pub const fn label(self) -> &'static str {
        match self {
            Framing::Envelope => "envelope",
            Framing::OwnTrailer => "own_trailer",
            Framing::Stamp => "stamp",
            Framing::Text => "text",
            Framing::RecordTag => "record_tag",
        }
    }

    /// Bytes the framing costs at the head of what it wraps
    pub const fn header_bytes(self) -> u32 {
        match self {
            Framing::Envelope | Framing::OwnTrailer => super::envelope::ENVELOPE_HEADER_LEN as u32,
            Framing::Stamp => super::stamp::FORMAT_STAMP_LEN as u32,
            Framing::Text => 0,
            Framing::RecordTag => 1,
        }
    }

    /// Whether a migrator for this framing takes and returns the whole
    /// file.
    ///
    /// The envelope framing hands a migrator the body alone and re-wraps
    /// the result, which a format with a trailer of its own cannot allow,
    /// because its checksums cover the header the envelope path would
    /// restamp
    pub const fn migrates_whole_file(self) -> bool {
        matches!(self, Framing::OwnTrailer)
    }

    /// Bytes the framing costs at the tail
    pub const fn footer_bytes(self) -> u32 {
        match self {
            Framing::Envelope => super::envelope::ENVELOPE_FOOTER_LEN as u32,
            _ => 0,
        }
    }

    /// The byte layout, as the documentation view prints it
    pub const fn layout(self) -> &'static str {
        match self {
            Framing::Envelope => {
                "[0..4) magic, [4..8) version, [8..12) header_length, [12..16) flags, \
                 [16..20) header_checksum, [20..header_length) extension, body, \
                 [len-4..len) body checksum"
            }
            Framing::OwnTrailer => {
                "[0..4) magic, [4..8) version, [8..12) header_length, [12..16) flags, \
                 [16..20) header_checksum, [20..header_length) extension, body and a \
                 trailer the format checksums itself"
            }
            Framing::Stamp => {
                "[0..4) magic, [4..6) major, [6..8) minor, [8] flags, body follows, \
                 integrity from the container"
            }
            Framing::Text => "[format] section with kind and version keys, body is TOML",
            Framing::RecordTag => {
                "[0] version tag, 1 through 254, the record lives inside its container. \
                 0 marks an unwritten slot and 255 escapes to a two byte version"
            }
        }
    }
}

impl fmt::Display for FormatKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.catalog_name())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;

    #[test]
    fn test_every_magic_is_unique() {
        let mut seen = HashSet::new();
        for row in MAGIC_ALLOCATIONS {
            assert!(
                seen.insert(row.magic),
                "magic {:?} is allocated twice, second holder is {:?}",
                std::str::from_utf8(&row.magic),
                row.kind
            );
        }
    }

    #[test]
    fn test_every_kind_has_exactly_one_allocation() {
        for kind in ALL_FORMAT_KINDS {
            let rows: Vec<_> = MAGIC_ALLOCATIONS
                .iter()
                .filter(|row| row.kind == *kind)
                .collect();
            assert_eq!(rows.len(), 1, "{kind:?} has {} allocations", rows.len());
        }
        assert_eq!(MAGIC_ALLOCATIONS.len(), ALL_FORMAT_KINDS.len());
    }

    #[test]
    fn test_magic_round_trips_to_kind() {
        for kind in ALL_FORMAT_KINDS {
            assert_eq!(FormatKind::from_magic(kind.magic()), Some(*kind));
        }
        assert_eq!(FormatKind::from_magic(*b"ZZZZ"), None);
    }

    #[test]
    fn test_every_magic_starts_with_z_and_is_ascii() {
        for row in MAGIC_ALLOCATIONS {
            assert_eq!(
                row.magic[0], b'Z',
                "{:?} magic does not start with Z",
                row.kind
            );
            assert!(
                row.magic.iter().all(|b| b.is_ascii_uppercase()),
                "{:?} magic is not uppercase ascii",
                row.kind
            );
        }
    }

    #[test]
    fn test_catalog_names_are_unique_and_round_trip() {
        let mut seen = HashSet::new();
        for kind in ALL_FORMAT_KINDS {
            let name = kind.catalog_name();
            assert!(seen.insert(name), "duplicate catalog name `{name}`");
            assert_eq!(FormatKind::from_catalog_name(name), Some(*kind));
            assert_eq!(
                FormatKind::from_catalog_name(&name.to_uppercase()),
                Some(*kind)
            );
        }
        assert_eq!(FormatKind::from_catalog_name("nothing_here"), None);
    }

    #[test]
    fn test_every_kind_declares_a_framing_and_its_costs() {
        for kind in ALL_FORMAT_KINDS {
            let framing = kind.framing();
            assert!(!framing.label().is_empty());
            assert!(!framing.layout().is_empty());
            if framing == Framing::Envelope {
                assert_eq!(framing.header_bytes(), 20);
                assert_eq!(framing.footer_bytes(), 4);
            }
            if kind.is_page_resident() {
                assert_eq!(framing, Framing::Stamp, "{kind} is page resident");
            }
        }
    }

    #[test]
    fn test_the_text_framings_are_the_two_hand_editable_files() {
        let text: Vec<_> = ALL_FORMAT_KINDS
            .iter()
            .copied()
            .filter(|kind| kind.framing() == Framing::Text)
            .collect();
        assert_eq!(
            text,
            vec![FormatKind::BackupArchive, FormatKind::ZyronTomlConfig]
        );
    }

    #[test]
    fn test_page_resident_kinds_are_the_four_page_formats() {
        let page_resident: Vec<_> = ALL_FORMAT_KINDS
            .iter()
            .copied()
            .filter(|k| k.is_page_resident())
            .collect();
        assert_eq!(
            page_resident,
            vec![
                FormatKind::HeapPage,
                FormatKind::BTreeInternal,
                FormatKind::BTreeLeaf,
                FormatKind::Fsm,
            ]
        );
    }
}
