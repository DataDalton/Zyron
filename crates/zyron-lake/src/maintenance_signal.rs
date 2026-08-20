// -----------------------------------------------------------------------------
// Which heads have changed since maintenance last looked.
//
// Background maintenance decides everything from a table's manifest, and a
// manifest changes only when a commit lands. A worker that enumerates every
// table on a timer therefore spends as much on a table nobody wrote as on
// one taking constant ingest, and the spend is not small: reconstructing a
// manifest and sweeping its file ranges for overlap costs O(files log files)
// per table per tick, forever, to conclude that nothing happened.
//
// A commit records its head here instead. A node whose tables are idle
// drains nothing and reads one atomic, and a table that just committed is
// looked at as soon as the worker's spacing floor allows rather than up to a
// full interval later.
//
// The record is bounded. Past `MAX_TRACKED_HEADS` distinct heads a mark
// raises the overflow flag rather than growing the map, and a worker that
// sees overflow enumerates every table it holds, which is exactly what it
// would have done on a timer. Memory is capped and no change is lost.
//
// Nothing here decides anything. It reports that a head moved; what that
// means for the table is the worker's judgement.
// -----------------------------------------------------------------------------

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Mutex, OnceLock};

/// Distinct heads tracked before marks start raising overflow instead.
///
/// A node hosting more actively written heads than this loses nothing: the
/// worker falls back to enumerating, so the guarantee is that memory here is
/// bounded whether or not anything ever drains it
pub const MAX_TRACKED_HEADS: usize = 4096;

/// One head that committed since the last drain.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DirtyHead {
    /// Registry key of the head, a table root for main and a branch
    /// directory for a branch
    pub key: PathBuf,
    /// Table the head belongs to
    pub table_id: u32,
    /// True for a table's main head, false for a branch head
    pub main_head: bool,
}

/// What one drain returned.
#[derive(Debug, Clone, Default)]
pub struct DirtyHeads {
    pub heads: Vec<DirtyHead>,
    /// True when marks were dropped because the map was full, so the reader
    /// has to enumerate rather than trust the list to be complete
    pub overflowed: bool,
}

/// Heads that committed, and the hook that tells a worker one did.
pub struct MaintenanceSignal {
    heads: Mutex<HashMap<PathBuf, (u32, bool)>>,
    /// Bumped by every mark, so a reader can tell whether anything arrived
    /// without taking the lock
    generation: AtomicU64,
    /// Raised when a mark found the map full
    overflowed: AtomicBool,
    /// Called after a mark lands. One worker per process installs it, and a
    /// process with no worker pays nothing for the absent call
    waker: OnceLock<Box<dyn Fn() + Send + Sync>>,
}

impl Default for MaintenanceSignal {
    fn default() -> Self {
        Self::new()
    }
}

impl MaintenanceSignal {
    pub fn new() -> Self {
        Self {
            heads: Mutex::new(HashMap::new()),
            generation: AtomicU64::new(0),
            overflowed: AtomicBool::new(false),
            waker: OnceLock::new(),
        }
    }

    /// How many marks have landed. A reader that has already drained at
    /// generation N skips the lock entirely while this still reads N
    #[inline]
    pub fn generation(&self) -> u64 {
        self.generation.load(Ordering::Acquire)
    }

    /// Records that one head committed.
    ///
    /// The generation moves after the insert, so a reader that samples the
    /// generation before draining can only ever be woken again for work it
    /// already took, never miss work it has not
    pub fn mark(&self, key: PathBuf, table_id: u32, main_head: bool) {
        {
            let mut heads = self.heads.lock().unwrap_or_else(|e| e.into_inner());
            let room = heads.len() < MAX_TRACKED_HEADS;
            match heads.get_mut(&key) {
                Some(slot) => *slot = (table_id, main_head),
                None if room => {
                    heads.insert(key, (table_id, main_head));
                }
                None => self.overflowed.store(true, Ordering::Release),
            }
        }
        self.generation.fetch_add(1, Ordering::Release);
        if let Some(waker) = self.waker.get() {
            waker();
        }
    }

    /// Takes every recorded head and clears the record.
    pub fn drain(&self) -> DirtyHeads {
        let taken = {
            let mut heads = self.heads.lock().unwrap_or_else(|e| e.into_inner());
            std::mem::take(&mut *heads)
        };
        DirtyHeads {
            heads: taken
                .into_iter()
                .map(|(key, (table_id, main_head))| DirtyHead {
                    key,
                    table_id,
                    main_head,
                })
                .collect(),
            overflowed: self.overflowed.swap(false, Ordering::AcqRel),
        }
    }

    /// Drops every head under one table root, used when the table goes away
    /// so a dropped table's marks do not hold a slot against the bound
    pub fn forget_under(&self, root: &Path) {
        let mut heads = self.heads.lock().unwrap_or_else(|e| e.into_inner());
        heads.retain(|key, _| !key.starts_with(root));
    }

    /// Installs the hook that wakes a worker when a head commits. Returns
    /// false when one is already installed, because a second worker reading
    /// the same record would take work the first is already doing
    pub fn set_waker(&self, waker: Box<dyn Fn() + Send + Sync>) -> bool {
        self.waker.set(waker).is_ok()
    }

    /// Whether one head is waiting to be looked at, without taking it
    pub fn is_marked(&self, key: &Path) -> bool {
        self.heads
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .contains_key(key)
    }

    /// Heads currently recorded, what a node reports about its own backlog
    pub fn tracked(&self) -> usize {
        self.heads.lock().unwrap_or_else(|e| e.into_inner()).len()
    }
}

// Process wide, like the log registry and the workload observer, because the
// record belongs to the node rather than to a session or a table
static SIGNAL: OnceLock<MaintenanceSignal> = OnceLock::new();

/// The node's maintenance signal.
pub fn maintenance_signal() -> &'static MaintenanceSignal {
    SIGNAL.get_or_init(MaintenanceSignal::new)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;
    use std::sync::atomic::AtomicUsize;

    #[test]
    fn test_a_drain_returns_each_head_once() {
        let signal = MaintenanceSignal::new();
        signal.mark(PathBuf::from("/lake/7"), 7, true);
        signal.mark(PathBuf::from("/lake/7"), 7, true);
        signal.mark(PathBuf::from("/lake/7/branches/dev"), 7, false);
        signal.mark(PathBuf::from("/lake/9"), 9, true);

        let mut drained = signal.drain();
        drained.heads.sort_by(|a, b| a.key.cmp(&b.key));
        assert_eq!(
            drained.heads.len(),
            3,
            "repeat marks coalesce onto one head"
        );
        assert!(!drained.overflowed);
        assert_eq!(drained.heads[0].table_id, 7);
        assert!(drained.heads[0].main_head);
        assert!(!drained.heads[1].main_head, "the branch head is not main");
        assert_eq!(drained.heads[2].table_id, 9);

        assert!(
            signal.drain().heads.is_empty(),
            "a drained record is empty until something else commits"
        );
    }

    /// The generation is what lets an idle worker skip the lock, so it has
    /// to move on every mark and stand still when nothing commits
    #[test]
    fn test_the_generation_moves_only_when_a_head_commits() {
        let signal = MaintenanceSignal::new();
        let start = signal.generation();
        assert_eq!(signal.generation(), start, "reading does not move it");
        signal.mark(PathBuf::from("/lake/1"), 1, true);
        assert_eq!(signal.generation(), start + 1);
        // A repeat mark of one head still moves it, because the reader has
        // to look again even when the head it must look at is unchanged
        signal.mark(PathBuf::from("/lake/1"), 1, true);
        assert_eq!(signal.generation(), start + 2);
        let _ = signal.drain();
        assert_eq!(signal.generation(), start + 2, "draining does not mark");
    }

    /// Memory is capped whether or not anything drains, and a reader is told
    /// the list stopped being complete rather than being left to trust it
    #[test]
    fn test_overflow_is_reported_rather_than_growing() {
        let signal = MaintenanceSignal::new();
        for i in 0..MAX_TRACKED_HEADS + 64 {
            signal.mark(PathBuf::from(format!("/lake/{}", i)), i as u32, true);
        }
        assert_eq!(signal.tracked(), MAX_TRACKED_HEADS);
        let drained = signal.drain();
        assert_eq!(drained.heads.len(), MAX_TRACKED_HEADS);
        assert!(drained.overflowed, "the reader has to know to enumerate");
        assert!(
            !signal.drain().overflowed,
            "the flag clears with the drain that reported it"
        );
    }

    #[test]
    fn test_dropping_a_table_forgets_its_heads() {
        let signal = MaintenanceSignal::new();
        signal.mark(PathBuf::from("/lake/4"), 4, true);
        signal.mark(PathBuf::from("/lake/4/branches/dev"), 4, false);
        signal.mark(PathBuf::from("/lake/40"), 40, true);
        signal.forget_under(Path::new("/lake/4"));
        let drained = signal.drain();
        assert_eq!(
            drained.heads.len(),
            1,
            "both heads under the dropped root go, the sibling table stays"
        );
        assert_eq!(drained.heads[0].table_id, 40);
    }

    #[test]
    fn test_one_waker_is_installed_and_called_per_mark() {
        let signal = MaintenanceSignal::new();
        let calls = Arc::new(AtomicUsize::new(0));
        let counter = Arc::clone(&calls);
        assert!(signal.set_waker(Box::new(move || {
            counter.fetch_add(1, Ordering::Relaxed);
        })));
        assert!(
            !signal.set_waker(Box::new(|| {})),
            "a second worker on one record would take the first's work"
        );
        signal.mark(PathBuf::from("/lake/2"), 2, true);
        signal.mark(PathBuf::from("/lake/3"), 3, true);
        assert_eq!(calls.load(Ordering::Relaxed), 2);
    }
}
