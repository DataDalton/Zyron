//! Moving a heap page from the version that had no schema epoch to the one
//! that reads it.
//!
//! A 1.0 page and a 1.1 page are the same bytes. The slot has always been 24
//! bytes wide with two spare at offset 6, and every 1.0 writer left them zero.
//! What 1.1 adds is a meaning for those two bytes: the epoch the row was
//! written under, where zero says the row predates stamping and reads through
//! the layout the table recorded at upgrade.
//!
//! So the migrator moves no byte. What completes the move is the catalog scan
//! registered beside it, which records that layout for every heap table. Until
//! that has run there is nothing for epoch 0 to mean, which is why the step is
//! one way: an older binary reading a 1.1 directory would see slots it thinks
//! are spare and rows written under layouts it has no record of.

use zyron_common::format::FormatKind;
use zyron_common::format::registry::{FormatFixture, FormatMigrator};
use zyron_common::page::{HEAP_PAGE_FORMAT_VERSION, HEAP_PAGE_OLDEST_READABLE};

/// A page the 0.14.0 writer produced, so the 1.0 reader and this step are
/// exercised against bytes that writer laid down rather than bytes the current
/// writer round-tripped.
static FIXTURE_1_0: &[u8] = include_bytes!("fixtures/v1_0.bin");

inventory::submit! {
    FormatFixture {
        kind: FormatKind::HeapPage,
        version: HEAP_PAGE_OLDEST_READABLE,
        bytes: FIXTURE_1_0,
        path: "crates/zyron-storage/src/heap/fixtures/v1_0.bin",
    }
}

inventory::submit! {
    FormatMigrator {
        kind: FormatKind::HeapPage,
        from: HEAP_PAGE_OLDEST_READABLE,
        to: HEAP_PAGE_FORMAT_VERSION,
        // The catalog scan that completes the move cannot be undone: once a
        // table's columns move on, the layout its unstamped rows were written
        // under is no longer derivable from anything
        reversible: false,
        forward: heap_page_1_0_to_1_1,
        backward: None,
        // The two spare slot bytes a 1.0 writer left zero are exactly the
        // epoch a 1.1 reader wants to see there
        no_body_change: true,
        description: "the two slot bytes at offset 6 now carry the schema epoch a row was written under",
    }
}

/// Reads a 1.0 heap page as 1.1.
///
/// The bytes are already right: a 1.0 slot's spare pair is zero, and zero is
/// the epoch that says "read through the recorded pre-stamp layout". The page
/// is handed back whole so the caller can restamp it at the current version.
pub fn heap_page_1_0_to_1_1(page: &[u8]) -> Result<Vec<u8>, String> {
    if page.len() < zyron_common::page::PageHeader::SIZE {
        return Err(format!(
            "{} bytes is shorter than a heap page header",
            page.len()
        ));
    }
    Ok(page.to_vec())
}

#[cfg(test)]
mod tests {
    use super::*;
    use zyron_common::format::FormatStamp;
    use zyron_common::page::{PAGE_SIZE, PageHeader, PageId, PageType, compute_page_checksum};

    use crate::heap::page::HeapPage;
    use crate::tuple::Tuple;

    /// Builds the bytes a 1.0 writer laid down: a heap page holding a few
    /// rows, every slot's spare pair zero, stamped 1.0.
    ///
    /// Run with `--ignored` to regenerate `fixtures/v1_0.bin` after changing
    /// what a heap page holds. The file is checked in because the registry
    /// exercises the 1.0 reader against bytes rather than against a
    /// round-trip through the current writer, and a fixture the current
    /// writer produced would prove nothing.
    #[test]
    #[ignore = "writes the checked-in fixture, run deliberately"]
    fn emit_heap_page_1_0_fixture() {
        let bytes = build_1_0_page();
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("src")
            .join("heap")
            .join("fixtures")
            .join("v1_0.bin");
        std::fs::create_dir_all(path.parent().expect("parent")).expect("fixtures dir");
        std::fs::write(&path, &bytes).expect("write fixture");
        println!("wrote {} bytes to {}", bytes.len(), path.display());
    }

    fn build_1_0_page() -> Vec<u8> {
        let page_id = PageId::new(7, 0);
        let mut page = HeapPage::new(page_id);
        for (i, body) in [b"alpha".as_slice(), b"bravo", b"charlie"]
            .into_iter()
            .enumerate()
        {
            // Epoch 0, which is what every 1.0 writer wrote
            page.insert_tuple(&Tuple::new(body.to_vec(), 100 + i as u64))
                .expect("insert");
        }
        let mut bytes = *page.as_bytes();

        // Restamp the header at 1.0 and recompute the checksum, so the file
        // is a page an older writer would have produced
        let mut header = PageHeader::from_bytes(&bytes[..PageHeader::SIZE]);
        header.set_stamp(FormatStamp::new(
            FormatKind::HeapPage,
            HEAP_PAGE_OLDEST_READABLE,
        ));
        bytes[..PageHeader::SIZE].copy_from_slice(&header.to_bytes());
        let checksum = compute_page_checksum(&bytes);
        bytes[26..30].copy_from_slice(&checksum.to_le_bytes());
        let _ = PageType::Heap;
        assert_eq!(bytes.len(), PAGE_SIZE);
        bytes.to_vec()
    }

    #[test]
    fn test_the_fixture_is_a_1_0_page_with_zero_epochs() {
        let header = PageHeader::from_bytes(&FIXTURE_1_0[..PageHeader::SIZE]);
        let stamp = header.stamp().expect("the fixture is stamped");
        assert_eq!(stamp.kind, FormatKind::HeapPage);
        assert_eq!(stamp.version, HEAP_PAGE_OLDEST_READABLE);

        let slot_count = HeapPage::heap_header_from_slice(FIXTURE_1_0).slot_count;
        assert!(slot_count > 0, "the fixture holds no rows");
        for slot in 0..slot_count {
            let view = HeapPage::get_tuple_view_from_slice(FIXTURE_1_0, crate::SlotId(slot))
                .expect("the fixture's slots decode");
            assert_eq!(
                view.header.schema_epoch, 0,
                "a 1.0 page cannot carry a schema epoch"
            );
        }
    }

    #[test]
    fn test_the_step_hands_the_page_back_unchanged() {
        let out = heap_page_1_0_to_1_1(FIXTURE_1_0).expect("the step runs");
        assert_eq!(out, FIXTURE_1_0, "the step moved a byte it should not have");
    }

    #[test]
    fn test_a_page_shorter_than_a_header_is_refused() {
        let err = heap_page_1_0_to_1_1(&[0u8; 8]).expect_err("a truncated page is refused");
        assert!(err.contains("shorter than"), "unexpected message: {err}");
    }
}
