//! The bulk B+tree loader, against the tree a per-key descent builds.
//!
//! A tree written level by level is only worth having if it is the same tree.
//! These compare an in-order traversal of a bulk-loaded tree against one built
//! by inserting the same keys one at a time, check that the merge with the keys
//! concurrent maintenance already placed lands each key once, and check that
//! the load survives the shapes a real build produces: keys of uneven length, a
//! stream long enough to need several interior levels, and an empty one.
//!
//! Run: cargo test -p zyron-storage --test btree_bulk_build_test -- --nocapture

use bytes::Bytes;
use zyron_common::RowLocator;
use zyron_common::page::PageId;
use zyron_storage::BTreeIndex;

fn locator(n: u64) -> RowLocator {
    RowLocator::Heap {
        page: PageId::new(0, n / 64),
        slot: (n % 64) as u16,
    }
}

/// Builds the key an index writes, the value big-endian so the byte order the
/// tree compares on is the numeric order, followed by the suffix naming the
/// row the key points at. The tree reads the row out of that suffix, so a key
/// without one is not a key any index holds.
fn key_for(n: u64) -> Bytes {
    let mut bytes = n.to_be_bytes().to_vec();
    locator(n).append_key_suffix(&mut bytes);
    Bytes::from(bytes)
}

/// A key whose value length varies, so leaves fill at different rates and the
/// level above them sees uneven page boundaries.
fn ragged_key(n: u64) -> Bytes {
    let mut bytes = n.to_be_bytes().to_vec();
    bytes.extend(std::iter::repeat_n(b'x', (n % 37) as usize));
    locator(n).append_key_suffix(&mut bytes);
    Bytes::from(bytes)
}

async fn empty_tree(dir: &std::path::Path, file_id: u32) -> BTreeIndex {
    BTreeIndex::create(file_id, dir.to_path_buf())
        .await
        .expect("tree")
}

#[tokio::test]
async fn test_bulk_build_and_sequential_insert_produce_the_same_traversal() {
    let tmp = tempfile::TempDir::new().expect("tmp");
    let bulk = empty_tree(tmp.path(), 1).await;
    let sequential = empty_tree(tmp.path(), 2).await;

    let count = 200_000u64;
    let items: Vec<(Bytes, RowLocator)> = (0..count).map(|n| (key_for(n), locator(n))).collect();

    let built = bulk
        .bulk_build_sorted(items.iter().cloned())
        .expect("bulk build");
    assert_eq!(built, count);

    for (key, loc) in &items {
        sequential.insert_sync(key, *loc).expect("insert");
    }

    let a = bulk.entries_in_key_order();
    let b = sequential.entries_in_key_order();
    assert_eq!(a.len(), count as usize, "bulk load lost entries");
    assert_eq!(b.len(), count as usize, "sequential insert lost entries");
    assert_eq!(a, b, "the two trees do not hold the same entries in order");
}

#[tokio::test]
async fn test_every_bulk_loaded_key_is_findable() {
    let tmp = tempfile::TempDir::new().expect("tmp");
    let tree = empty_tree(tmp.path(), 3).await;
    let count = 100_000u64;
    let items: Vec<(Bytes, RowLocator)> = (0..count).map(|n| (key_for(n), locator(n))).collect();
    tree.bulk_build_sorted(items).expect("bulk build");

    for n in (0..count).step_by(97) {
        assert_eq!(
            tree.search_sync(&key_for(n)),
            Some(locator(n)),
            "key {n} is not findable in the bulk-loaded tree"
        );
    }
    // A key that was never inserted must not be found, so the search is
    // answering from the tree rather than from the key
    assert_eq!(tree.search_sync(&key_for(count + 1)), None);
}

#[tokio::test]
async fn test_ragged_keys_load_and_read_back_in_order() {
    let tmp = tempfile::TempDir::new().expect("tmp");
    let bulk = empty_tree(tmp.path(), 4).await;
    let sequential = empty_tree(tmp.path(), 5).await;

    let count = 40_000u64;
    let items: Vec<(Bytes, RowLocator)> = (0..count).map(|n| (ragged_key(n), locator(n))).collect();
    bulk.bulk_build_sorted(items.iter().cloned())
        .expect("bulk build");
    for (key, loc) in &items {
        sequential.insert_sync(key, *loc).expect("insert");
    }

    assert_eq!(
        bulk.entries_in_key_order(),
        sequential.entries_in_key_order(),
        "uneven key lengths produced a different tree"
    );
}

#[tokio::test]
async fn test_the_merge_takes_a_key_in_both_streams_once() {
    let tmp = tempfile::TempDir::new().expect("tmp");
    let tree = empty_tree(tmp.path(), 6).await;

    // What index maintenance placed after the build published, every third key
    let maintained: Vec<u64> = (0..30_000u64).filter(|n| n % 3 == 0).collect();
    for n in &maintained {
        tree.insert_sync(&key_for(*n), locator(*n)).expect("insert");
    }

    // What the scan produced, every key including the ones already there
    let scanned: Vec<(Bytes, RowLocator)> =
        (0..30_000u64).map(|n| (key_for(n), locator(n))).collect();
    let built = tree.bulk_build_sorted(scanned).expect("bulk build");

    assert_eq!(built, 30_000, "a key present in both streams landed twice");
    let all = tree.entries_in_key_order();
    assert_eq!(all.len(), 30_000);
    let mut previous: Option<Bytes> = None;
    for (key, _) in &all {
        if let Some(prev) = &previous {
            assert!(prev < key, "the traversal is not in key order");
        }
        previous = Some(key.clone());
    }
}

#[tokio::test]
async fn test_a_key_written_after_the_merge_read_the_tree_still_lands() {
    let tmp = tempfile::TempDir::new().expect("tmp");
    let tree = std::sync::Arc::new(empty_tree(tmp.path(), 7).await);

    // The build's own set
    let scanned: Vec<(Bytes, RowLocator)> =
        (0..50_000u64).map(|n| (key_for(n), locator(n))).collect();

    // Maintenance that lands while the load is running. The swap re-reads the
    // old tree under its own lock, so a key placed here is not discarded
    let writer_tree = std::sync::Arc::clone(&tree);
    let writer = std::thread::spawn(move || {
        for n in 1_000_000u64..1_000_500 {
            let _guard = writer_tree.maintenance_guard();
            writer_tree
                .insert_sync(&key_for(n), locator(n))
                .expect("insert");
        }
    });

    tree.bulk_build_sorted(scanned).expect("bulk build");
    writer.join().expect("writer");

    for n in 1_000_000u64..1_000_500 {
        assert!(
            tree.search_sync(&key_for(n)).is_some(),
            "the key {n} a writer placed during the load was discarded by the swap"
        );
    }
    for n in (0..50_000u64).step_by(311) {
        assert!(
            tree.search_sync(&key_for(n)).is_some(),
            "the key {n} the scan produced is missing"
        );
    }
}

#[tokio::test]
async fn test_an_empty_stream_leaves_a_usable_tree() {
    let tmp = tempfile::TempDir::new().expect("tmp");
    let tree = empty_tree(tmp.path(), 8).await;
    let built = tree
        .bulk_build_sorted(Vec::<(Bytes, RowLocator)>::new())
        .expect("bulk build");
    assert_eq!(built, 0);
    assert!(tree.entries_in_key_order().is_empty());

    tree.insert_sync(&key_for(42), locator(42)).expect("insert");
    assert_eq!(tree.search_sync(&key_for(42)), Some(locator(42)));
}

#[tokio::test]
async fn test_keys_out_of_order_are_refused_rather_than_loaded() {
    let tmp = tempfile::TempDir::new().expect("tmp");
    let tree = empty_tree(tmp.path(), 9).await;
    let items = vec![
        (key_for(10), locator(10)),
        (key_for(20), locator(20)),
        (key_for(15), locator(15)),
    ];
    let err = tree
        .bulk_build_sorted(items)
        .expect_err("an unordered stream builds a tree that cannot find its own entries");
    assert!(
        err.to_string().contains("out of order"),
        "unexpected error: {err}"
    );
}

#[tokio::test]
async fn test_a_bulk_loaded_tree_takes_further_inserts() {
    let tmp = tempfile::TempDir::new().expect("tmp");
    let tree = empty_tree(tmp.path(), 10).await;
    let items: Vec<(Bytes, RowLocator)> = (0..20_000u64)
        .map(|n| (key_for(n * 2), locator(n * 2)))
        .collect();
    tree.bulk_build_sorted(items).expect("bulk build");

    // The odd keys go in through the ordinary descent, which has to work on
    // pages the loader wrote
    for n in 0..20_000u64 {
        tree.insert_sync(&key_for(n * 2 + 1), locator(n * 2 + 1))
            .expect("insert into a bulk-loaded tree");
    }
    assert_eq!(tree.entries_in_key_order().len(), 40_000);
    for n in (0..40_000u64).step_by(137) {
        assert!(
            tree.search_sync(&key_for(n)).is_some(),
            "key {n} is missing after inserting into a bulk-loaded tree"
        );
    }
}

#[tokio::test]
async fn test_deletes_work_against_a_bulk_loaded_tree() {
    let tmp = tempfile::TempDir::new().expect("tmp");
    let tree = empty_tree(tmp.path(), 11).await;
    let items: Vec<(Bytes, RowLocator)> =
        (0..10_000u64).map(|n| (key_for(n), locator(n))).collect();
    tree.bulk_build_sorted(items).expect("bulk build");

    for n in (0..10_000u64).step_by(2) {
        assert!(
            tree.delete_sync(&key_for(n)),
            "delete of key {n} found nothing"
        );
    }
    for n in 0..10_000u64 {
        let found = tree.search_sync(&key_for(n)).is_some();
        assert_eq!(
            found,
            n % 2 == 1,
            "key {n} has the wrong presence after deletes"
        );
    }
}
