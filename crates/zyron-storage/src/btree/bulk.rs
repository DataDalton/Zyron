//! Loading a whole B+tree from keys that are already in order.
//!
//! An index build produces its entries sorted, so descending the tree once per
//! key is wasted work: every descent lands to the right of the last one. This
//! writes each level once instead. Leaves fill left to right, and each interior
//! level is built from the list of pages the level below produced, so a key is
//! compared only against the page boundary it creates.
//!
//! The tree the build publishes is not empty while the load runs. Writers have
//! been maintaining it since publication, so their keys are merged into the
//! same ordered stream and a key that reaches both sides lands once. Whatever
//! arrives after the merge has read the tree is caught by a second pass under
//! the swap lock, which is the only window in the whole build where an index
//! maintenance call waits.

use bytes::Bytes;
use zyron_common::RowLocator;
use zyron_common::page::PageId;
use zyron_common::{Result, ZyronError};

use super::index::BTreeIndex;
use super::page::{BTreeInternalPage, BTreeLeafPage};
use super::types::{InternalEntry, LeafEntry, compare_keys};

/// What one finished level handed to the level above it: the page and the
/// first key it holds, which is the separator that routes to it.
struct LevelPage {
    page_num: u32,
    first_key: Bytes,
}

/// Writes leaf pages left to right, linking each to the next.
struct LeafWriter<'a> {
    tree: &'a BTreeIndex,
    current: BTreeLeafPage,
    current_num: u32,
    current_first_key: Option<Bytes>,
    current_count: usize,
    produced: Vec<LevelPage>,
    entries: u64,
}

impl<'a> LeafWriter<'a> {
    fn new(tree: &'a BTreeIndex) -> Self {
        let page_num = tree.pages_ref().allocate();
        let page = BTreeLeafPage::new(PageId::new(tree.file_id(), page_num as u64));
        Self {
            tree,
            current: page,
            current_num: page_num,
            current_first_key: None,
            current_count: 0,
            produced: Vec::new(),
            entries: 0,
        }
    }

    /// Appends one entry, opening a new leaf when the current one is full.
    ///
    /// A key that does not fit an empty page cannot be stored at all, which is
    /// reported rather than dropped: a build that silently skipped a key would
    /// leave every scan through the index short of the rows the table holds.
    fn push(&mut self, key: Bytes, locator: RowLocator) -> Result<()> {
        let entry = LeafEntry {
            key: key.clone(),
            locator,
        };
        let cost = entry.size_on_disk() + BTreeLeafPage::SLOT_SIZE;
        if !self.current.can_fit(cost) {
            if self.current_count == 0 {
                return Err(ZyronError::ExecutionError(format!(
                    "index key of {} bytes does not fit an empty B+tree leaf",
                    key.len()
                )));
            }
            self.seal_and_open()?;
        }
        self.current.insert(key.clone(), locator)?;
        if self.current_count == 0 {
            self.current_first_key = Some(key);
        }
        self.current_count += 1;
        self.entries += 1;
        Ok(())
    }

    /// Closes the current leaf, links it to the one that follows, and starts
    /// the next.
    fn seal_and_open(&mut self) -> Result<()> {
        let next_num = self.tree.pages_ref().allocate();
        self.current
            .set_next_leaf(Some(PageId::new(self.tree.file_id(), next_num as u64)));
        self.flush_current();
        self.current = BTreeLeafPage::new(PageId::new(self.tree.file_id(), next_num as u64));
        self.current_num = next_num;
        self.current_first_key = None;
        self.current_count = 0;
        Ok(())
    }

    fn flush_current(&mut self) {
        self.tree
            .pages_ref()
            .force_write(self.current_num, self.current.as_bytes());
        let first_key = self
            .current_first_key
            .clone()
            .unwrap_or_else(|| Bytes::from_static(&[]));
        self.produced.push(LevelPage {
            page_num: self.current_num,
            first_key,
        });
    }

    /// Writes the last leaf and returns the level, which is never empty: an
    /// empty stream still produces the one empty leaf that is the tree's root.
    fn finish(mut self) -> (Vec<LevelPage>, u64) {
        self.current.set_next_leaf(None);
        self.flush_current();
        (self.produced, self.entries)
    }
}

/// Builds one interior level from the pages below it.
///
/// The first child of a page is reached without a key, so its own first key is
/// what routes to the page as a whole and is handed up rather than stored.
fn build_interior_level(
    tree: &BTreeIndex,
    level: u16,
    children: &[LevelPage],
) -> Result<Vec<LevelPage>> {
    let mut produced: Vec<LevelPage> = Vec::new();
    let mut idx = 0usize;
    while idx < children.len() {
        let page_num = tree.pages_ref().allocate();
        let mut page = BTreeInternalPage::new(PageId::new(tree.file_id(), page_num as u64), level);
        page.set_leftmost_child(PageId::new(tree.file_id(), children[idx].page_num as u64));
        let page_first_key = children[idx].first_key.clone();
        idx += 1;

        let mut keys_on_page = 0usize;
        while idx < children.len() {
            let child = &children[idx];
            let entry = InternalEntry {
                key: child.first_key.clone(),
                child_page_id: PageId::new(tree.file_id(), child.page_num as u64),
            };
            if !page.can_fit(entry.size_on_disk()) {
                break;
            }
            page.insert(
                child.first_key.clone(),
                PageId::new(tree.file_id(), child.page_num as u64),
            )?;
            keys_on_page += 1;
            idx += 1;
        }
        // A page that took its leftmost child and nothing else still routes
        // correctly, and refusing to advance here would loop forever
        if keys_on_page == 0 && idx < children.len() && idx == 0 {
            return Err(ZyronError::ExecutionError(
                "a B+tree separator key does not fit an empty interior page".to_string(),
            ));
        }
        tree.pages_ref().force_write(page_num, page.as_bytes());
        produced.push(LevelPage {
            page_num,
            first_key: page_first_key,
        });
    }
    Ok(produced)
}

/// Merges two ordered streams, keeping one entry per key.
///
/// The tree's own entry wins a tie. It is the later write of the two: the scan
/// read the row under a snapshot taken before publication finished, while the
/// tree's entry was placed by a writer that ran after it.
fn merged<A, B>(scanned: A, existing: B) -> impl Iterator<Item = (Bytes, RowLocator)>
where
    A: Iterator<Item = (Bytes, RowLocator)>,
    B: Iterator<Item = (Bytes, RowLocator)>,
{
    let mut left = scanned.peekable();
    let mut right = existing.peekable();
    std::iter::from_fn(move || match (left.peek(), right.peek()) {
        (Some((lk, _)), Some((rk, _))) => match compare_keys(lk, rk) {
            std::cmp::Ordering::Less => left.next(),
            std::cmp::Ordering::Greater => right.next(),
            std::cmp::Ordering::Equal => {
                left.next();
                right.next()
            }
        },
        (Some(_), None) => left.next(),
        (None, Some(_)) => right.next(),
        (None, None) => None,
    })
}

impl BTreeIndex {
    /// Every entry the tree holds right now, in key order.
    ///
    /// Used by the bulk load to fold in what index maintenance has written
    /// since the build published its entry.
    pub fn entries_in_key_order(&self) -> Vec<(Bytes, RowLocator)> {
        let mut out = Vec::new();
        self.range_scan_for_each(None, None, |key, locator| {
            out.push((Bytes::copy_from_slice(key), locator));
            true
        });
        out
    }

    /// Replaces this tree with one built from `sorted` merged with what the
    /// tree already holds, writing each level in a single pass.
    ///
    /// `sorted` must be in ascending key order. The keys an index build
    /// produces already are, because they come out of an external sort, and an
    /// unordered stream would produce a tree whose search does not find its own
    /// entries, so the order is checked as the stream is consumed.
    ///
    /// Returns the number of entries the finished tree holds.
    pub fn bulk_build_sorted<I>(&self, sorted: I) -> Result<u64>
    where
        I: IntoIterator<Item = (Bytes, RowLocator)>,
    {
        let existing = self.entries_in_key_order();
        let mut writer = LeafWriter::new(self);
        let mut previous: Option<Bytes> = None;
        let scanned = sorted.into_iter();
        for (key, locator) in merged(scanned, existing.iter().cloned()) {
            if let Some(prev) = &previous
                && compare_keys(prev, &key).is_gt()
            {
                return Err(ZyronError::ExecutionError(
                    "bulk_build_sorted was handed keys out of order, which would build a tree \
                     whose search cannot find its own entries"
                        .to_string(),
                ));
            }
            previous = Some(key.clone());
            writer.push(key, locator)?;
        }
        let (mut level_pages, entries) = writer.finish();

        let mut height: u32 = 1;
        while level_pages.len() > 1 {
            let level = (height - 1) as u16;
            level_pages = build_interior_level(self, level, &level_pages)?;
            height += 1;
            if height as usize > Self::max_height() {
                return Err(ZyronError::ExecutionError(format!(
                    "bulk load produced a B+tree taller than the {} level limit",
                    Self::max_height()
                )));
            }
        }
        let new_root = level_pages[0].page_num;

        // Everything index maintenance wrote while the levels were being
        // written is still in the old tree. The swap and the catch-up run
        // together so a key placed between them cannot fall into the tree
        // being retired
        let _swap = self.lock_build_swap();
        let arrived = self.entries_in_key_order();
        self.install_root(new_root, height);
        let mut late = 0u64;
        for (key, locator) in arrived {
            if existing
                .binary_search_by(|(k, _)| compare_keys(k, &key))
                .is_ok()
            {
                continue;
            }
            self.insert_sync(&key, locator)?;
            late += 1;
        }
        Ok(entries + late)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use zyron_common::page::PageId;

    fn locator(n: u64) -> RowLocator {
        RowLocator::Heap {
            page: PageId::new(0, n),
            slot: (n % 64) as u16,
        }
    }

    /// Builds the key an index writes, the value big-endian so byte order is
    /// numeric order, followed by the suffix naming the row it points at.
    fn key_for(n: u64) -> Bytes {
        let mut bytes = n.to_be_bytes().to_vec();
        locator(n).append_key_suffix(&mut bytes);
        Bytes::from(bytes)
    }

    async fn empty_tree(dir: &std::path::Path, file_id: u32) -> BTreeIndex {
        BTreeIndex::create(file_id, dir.to_path_buf())
            .await
            .expect("tree")
    }

    #[tokio::test]
    async fn test_bulk_build_matches_sequential_insert() {
        let tmp = tempfile::TempDir::new().expect("tmp");
        let bulk = empty_tree(tmp.path(), 1).await;
        let sequential = empty_tree(tmp.path(), 2).await;

        let count = 50_000u64;
        let items: Vec<(Bytes, RowLocator)> =
            (0..count).map(|n| (key_for(n), locator(n))).collect();
        let built = bulk
            .bulk_build_sorted(items.iter().cloned())
            .expect("bulk build");
        assert_eq!(built, count);

        for (key, loc) in &items {
            sequential.insert_sync(key, *loc).expect("insert");
        }

        let a = bulk.entries_in_key_order();
        let b = sequential.entries_in_key_order();
        assert_eq!(a.len(), b.len(), "entry counts differ");
        assert_eq!(a, b, "in-order traversals differ");
    }

    #[tokio::test]
    async fn test_bulk_build_merges_concurrent_keys_without_duplicates() {
        let tmp = tempfile::TempDir::new().expect("tmp");
        let tree = empty_tree(tmp.path(), 3).await;

        // What maintenance placed after publication, odd keys only
        for n in (1..2_000u64).step_by(2) {
            tree.insert_sync(&key_for(n), locator(n)).expect("insert");
        }
        // What the scan produced, every key including the odd ones
        let scanned: Vec<(Bytes, RowLocator)> =
            (0..2_000u64).map(|n| (key_for(n), locator(n))).collect();
        let built = tree.bulk_build_sorted(scanned).expect("bulk build");
        assert_eq!(built, 2_000, "a key in both streams landed twice");

        let all = tree.entries_in_key_order();
        assert_eq!(all.len(), 2_000);
        for n in 0..2_000u64 {
            assert_eq!(
                tree.search_sync(&key_for(n)),
                Some(locator(n)),
                "key {n} is not in the built tree"
            );
        }
    }

    #[tokio::test]
    async fn test_bulk_build_refuses_keys_out_of_order() {
        let tmp = tempfile::TempDir::new().expect("tmp");
        let tree = empty_tree(tmp.path(), 4).await;
        let items = vec![
            (key_for(5), locator(5)),
            (key_for(3), locator(3)),
            (key_for(9), locator(9)),
        ];
        let err = tree.bulk_build_sorted(items).expect_err("out of order");
        assert!(
            err.to_string().contains("out of order"),
            "unexpected error: {err}"
        );
    }

    #[tokio::test]
    async fn test_bulk_build_of_an_empty_stream_leaves_a_searchable_tree() {
        let tmp = tempfile::TempDir::new().expect("tmp");
        let tree = empty_tree(tmp.path(), 5).await;
        let built = tree
            .bulk_build_sorted(Vec::<(Bytes, RowLocator)>::new())
            .expect("bulk build");
        assert_eq!(built, 0);
        assert_eq!(tree.entries_in_key_order().len(), 0);
        tree.insert_sync(&key_for(1), locator(1)).expect("insert");
        assert_eq!(tree.search_sync(&key_for(1)), Some(locator(1)));
    }
}
