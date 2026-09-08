//! Storage format registrations.
//!
//! The four page-resident formats carry their envelope in the page header
//! stamp rather than in a header of their own, so a page keeps every byte of
//! its payload capacity. They migrate lazily: a page moves to the current
//! version the next time it is written, which costs nothing on a read and
//! nothing on a page that is never touched again.
//!
//! The index checkpoint and the commit log carry the full envelope. The
//! checkpoint is rewritten wholesale on the next checkpoint and the commit
//! log on every persist, so each migrates with the next one. The columnar
//! file carries the envelope header and a trailer of its own, and an old
//! one is repacked by the upgrade sweep, whole, through its migration

use zyron_common::format::FormatKind;
use zyron_common::format::registry::{DeprecationStatus, FormatRegistration, MigrationPolicy};
use zyron_common::format::scan_migration::{CatalogScanMigration, FormatDocumentation};
use zyron_common::format::version::{FormatVersion, VersionWindow};
use zyron_common::page::{
    BTREE_PAGE_FORMAT_VERSION, FSM_PAGE_FORMAT_VERSION, HEAP_PAGE_FORMAT_VERSION,
    HEAP_PAGE_OLDEST_READABLE,
};

use crate::columnar::constants::{ZYR_FORMAT_VERSION, ZYR_READER_WINDOW};

/// The Zyron version these formats were last bumped in
const GATE: &str = "0.11.0";

/// The Zyron version the columnar file moved to 1.1 in
const ZYR_GATE: &str = "0.12.0";

/// The day the 1.0 columnar reader and its migration leave the tree. The
/// eager sweep has moved every file long before, this is the date the
/// release check holds the code to
const ZYR_1_0_RETIREMENT: &str = "2027-03-01";

/// Version the B+tree index checkpoint is written at
pub const CHECKPOINT_FORMAT_VERSION: FormatVersion = FormatVersion::new(11, 1);

/// The oldest checkpoint this binary reads. 11.0 stored a column of locator
/// payloads beside the keys, which 11.1 leaves out because each key already
/// ends in the suffix naming its row.
pub const CHECKPOINT_OLDEST_READABLE: FormatVersion = FormatVersion::new(11, 0);

/// The day the 11.0 checkpoint reader and its migration leave the tree.
///
/// A checkpoint is derived from the tree it describes, so an 11.0 file is
/// replaced the first time its index checkpoints again.
const CHECKPOINT_11_0_RETIREMENT: &str = "2027-03-01";

/// Version a serialized bloom filter is written at
pub const BLOOM_FILTER_FORMAT_VERSION: FormatVersion = FormatVersion::V1;

/// Version the MVCC commit log file is written at
pub const MVCC_CLOG_FORMAT_VERSION: FormatVersion = FormatVersion::V1;

/// The Zyron version the heap page moved to 1.1 in, when the slot's spare
/// two bytes became the schema epoch
const HEAP_EPOCH_GATE: &str = "0.15.0";

/// The day the 1.0 heap page reader and its migration leave the tree.
///
/// A 1.0 page moves to 1.1 the next time it is written, and a page nothing
/// ever writes again keeps reading through the layout recorded at upgrade, so
/// the reader stays until every directory has been through a full write cycle.
/// This is the date the release check holds the code to
const HEAP_1_0_RETIREMENT: &str = "2027-03-01";

inventory::submit! {
    FormatRegistration {
        kind: FormatKind::HeapPage,
        writer_current_version: HEAP_PAGE_FORMAT_VERSION,
        reader_supported_versions: VersionWindow::new(HEAP_PAGE_OLDEST_READABLE, HEAP_PAGE_FORMAT_VERSION),
        migration_policy: MigrationPolicy::Lazy,
        migration_reversible: false,
        binary_version_gate: HEAP_EPOCH_GATE,
        deprecation_status: DeprecationStatus::Active,
        retirement_date: Some(HEAP_1_0_RETIREMENT),
        downgrade_write_supported: false,
        notes: "slotted heap page, stamp in the page header, lazy on next page write",
    }
}

inventory::submit! {
    CatalogScanMigration {
        kind: FormatKind::HeapPage,
        from: HEAP_PAGE_OLDEST_READABLE,
        to: HEAP_PAGE_FORMAT_VERSION,
        introduced_in_binary_version: HEAP_EPOCH_GATE,
        one_way: true,
        scan_name: "heap_pre_stamp_columns",
        description: "records each heap table's column layout as the one every tuple stamped 0 decodes through",
    }
}

inventory::submit! {
    FormatDocumentation {
        kind: FormatKind::HeapPage,
        version: HEAP_PAGE_FORMAT_VERSION,
        text: "Heap tuple slots now record the schema epoch a row was written under. \
               Existing rows carry epoch 0 and read through the column layout recorded at upgrade. \
               ADD COLUMN, DROP COLUMN and compatible type changes complete without rewriting rows",
    }
}

inventory::submit! {
    FormatDocumentation {
        kind: FormatKind::Checkpoint,
        version: CHECKPOINT_FORMAT_VERSION,
        text: "An index checkpoint stores each key's value and the row it points at once rather \
               than twice. The seventeen byte suffix naming the row is rebuilt when the file \
               loads, from the seven byte locator kept behind each value. Existing checkpoints \
               are moved forward the first time their index checkpoints again",
    }
}

inventory::submit! {
    FormatRegistration {
        kind: FormatKind::BTreeInternal,
        writer_current_version: BTREE_PAGE_FORMAT_VERSION,
        reader_supported_versions: VersionWindow::single(BTREE_PAGE_FORMAT_VERSION),
        migration_policy: MigrationPolicy::Lazy,
        migration_reversible: true,
        binary_version_gate: GATE,
        deprecation_status: DeprecationStatus::Active,
        retirement_date: None,
        downgrade_write_supported: false,
        notes: "separator keys and child pointers, lazy on next page write",
    }
}

inventory::submit! {
    FormatRegistration {
        kind: FormatKind::BTreeLeaf,
        writer_current_version: BTREE_PAGE_FORMAT_VERSION,
        reader_supported_versions: VersionWindow::single(BTREE_PAGE_FORMAT_VERSION),
        migration_policy: MigrationPolicy::Lazy,
        migration_reversible: true,
        binary_version_gate: GATE,
        deprecation_status: DeprecationStatus::Active,
        retirement_date: None,
        downgrade_write_supported: false,
        notes: "keys and row locators, lazy on next page write",
    }
}

inventory::submit! {
    FormatRegistration {
        kind: FormatKind::Fsm,
        writer_current_version: FSM_PAGE_FORMAT_VERSION,
        reader_supported_versions: VersionWindow::single(FSM_PAGE_FORMAT_VERSION),
        migration_policy: MigrationPolicy::Lazy,
        migration_reversible: true,
        binary_version_gate: GATE,
        deprecation_status: DeprecationStatus::Active,
        retirement_date: None,
        downgrade_write_supported: false,
        notes: "per-page free byte counts, lazy on next page write",
    }
}

inventory::submit! {
    FormatRegistration {
        kind: FormatKind::ZyrColumnar,
        writer_current_version: ZYR_FORMAT_VERSION,
        reader_supported_versions: ZYR_READER_WINDOW,
        migration_policy: MigrationPolicy::Eager,
        migration_reversible: false,
        binary_version_gate: ZYR_GATE,
        deprecation_status: DeprecationStatus::Active,
        retirement_date: Some(ZYR_1_0_RETIREMENT),
        downgrade_write_supported: false,
        notes: "segments aligned to 64 bytes behind a 512-byte header, the page-padded 1.0 is repacked by the eager sweep",
    }
}

inventory::submit! {
    FormatRegistration {
        kind: FormatKind::Checkpoint,
        writer_current_version: CHECKPOINT_FORMAT_VERSION,
        reader_supported_versions: VersionWindow::new(
            CHECKPOINT_OLDEST_READABLE,
            CHECKPOINT_FORMAT_VERSION,
        ),
        migration_policy: MigrationPolicy::Eager,
        // The locator column 11.0 carried is derivable from the keys beside
        // it, so the step back down can rebuild it exactly
        migration_reversible: true,
        binary_version_gate: CHECKPOINT_KEY_NAMES_ROW_GATE,
        deprecation_status: DeprecationStatus::Active,
        retirement_date: Some(CHECKPOINT_11_0_RETIREMENT),
        downgrade_write_supported: false,
        notes: "prefix-compressed keys, each ending in the suffix naming its row, eager on the next checkpoint",
    }
}

/// The Zyron version the checkpoint dropped its locator column in.
const CHECKPOINT_KEY_NAMES_ROW_GATE: &str = "0.15.0";

inventory::submit! {
    FormatRegistration {
        kind: FormatKind::BloomFilter,
        writer_current_version: BLOOM_FILTER_FORMAT_VERSION,
        reader_supported_versions: VersionWindow::single(BLOOM_FILTER_FORMAT_VERSION),
        migration_policy: MigrationPolicy::Lazy,
        migration_reversible: true,
        binary_version_gate: GATE,
        deprecation_status: DeprecationStatus::Active,
        retirement_date: None,
        downgrade_write_supported: false,
        notes: "bit array and hash count, lazy on the next index rebuild",
    }
}

inventory::submit! {
    FormatRegistration {
        kind: FormatKind::MvccClog,
        writer_current_version: MVCC_CLOG_FORMAT_VERSION,
        reader_supported_versions: VersionWindow::single(MVCC_CLOG_FORMAT_VERSION),
        migration_policy: MigrationPolicy::Lazy,
        migration_reversible: true,
        binary_version_gate: GATE,
        deprecation_status: DeprecationStatus::Active,
        retirement_date: None,
        downgrade_write_supported: false,
        notes: "per-transaction status words and commit LSNs, lazy on next persist",
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use zyron_common::format::envelope;
    use zyron_common::page::{PageHeader, PageId, PageType};

    /// Every format this crate registers, and the page type that carries it
    /// where the format is page resident
    const STORAGE_KINDS: &[FormatKind] = &[
        FormatKind::HeapPage,
        FormatKind::BTreeInternal,
        FormatKind::BTreeLeaf,
        FormatKind::Fsm,
        FormatKind::ZyrColumnar,
        FormatKind::Checkpoint,
        FormatKind::BloomFilter,
        FormatKind::MvccClog,
    ];

    #[test]
    fn test_every_storage_format_registers_exactly_once() {
        for kind in STORAGE_KINDS {
            let count = inventory::iter::<FormatRegistration>
                .into_iter()
                .filter(|r| r.kind == *kind)
                .count();
            assert_eq!(count, 1, "{kind} submitted {count} registrations");
        }
    }

    #[test]
    fn test_page_formats_stamp_the_registered_version() {
        for (page_type, kind) in [
            (PageType::Heap, FormatKind::HeapPage),
            (PageType::BTreeInternal, FormatKind::BTreeInternal),
            (PageType::BTreeLeaf, FormatKind::BTreeLeaf),
            (PageType::FreeSpaceMap, FormatKind::Fsm),
        ] {
            let registration = inventory::iter::<FormatRegistration>
                .into_iter()
                .find(|r| r.kind == kind)
                .expect("registered");
            let header = PageHeader::new(PageId::new(1, 1), page_type);
            let stamp = header.stamp().expect("stamped");
            assert_eq!(stamp.kind, kind);
            assert_eq!(stamp.version, registration.writer_current_version);
        }
    }

    #[test]
    fn test_columnar_header_is_an_envelope() {
        let header = crate::columnar::ZyrFileHeader {
            format_version: ZYR_FORMAT_VERSION,
            column_count: 3,
            row_count: 100,
            table_id: 7,
            xmin_range_lo: 1,
            xmin_range_hi: 2,
            xmax_range_lo: 0,
            xmax_range_hi: 0,
            primary_key_column_id: 0,
            sort_order: crate::columnar::SortOrder::None,
            segment_index_offset: 0,
            segment_index_size: 0,
        };
        let bytes = header.to_bytes();
        let (kind, version) = envelope::peek(&bytes).expect("peeks");
        assert_eq!(kind, FormatKind::ZyrColumnar);
        assert_eq!(version, ZYR_FORMAT_VERSION);
        let read = crate::columnar::ZyrFileHeader::from_bytes(&bytes).expect("reads back");
        assert_eq!(read.column_count, 3);
        assert_eq!(read.row_count, 100);
        assert_eq!(read.table_id, 7);
    }

    #[test]
    fn test_a_corrupted_columnar_header_byte_is_caught() {
        let header = crate::columnar::ZyrFileHeader {
            format_version: ZYR_FORMAT_VERSION,
            column_count: 3,
            row_count: 100,
            table_id: 7,
            xmin_range_lo: 1,
            xmin_range_hi: 2,
            xmax_range_lo: 3,
            xmax_range_hi: 4,
            primary_key_column_id: 5,
            sort_order: crate::columnar::SortOrder::None,
            segment_index_offset: 6,
            segment_index_size: 7,
        };
        let bytes = header.to_bytes();
        for index in 0..128 {
            let mut corrupted = bytes;
            corrupted[index] ^= 0x01;
            assert!(
                crate::columnar::ZyrFileHeader::from_bytes(&corrupted).is_err(),
                "flipping columnar header byte {index} was not caught"
            );
        }
    }
}
