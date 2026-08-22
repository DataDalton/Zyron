//! Workload observation for Adaptive Clustering.
//!
//! Clustering decisions are only as good as the evidence behind them, and
//! evidence gathered on the query path has to cost nothing. Every counter
//! here is one `AtomicU64` carrying a 16-bit epoch tag in its high bits and
//! a 48-bit value in its low bits, so an observation is one load and one
//! compare-and-swap: no lock, no allocation, no sweeper thread.
//!
//! **Decay reclaims itself.** A counter whose tag is older than the epoch
//! being written is not swept by anything; the write that finds it simply
//! replaces it with its own delta. A term nobody has queried for eight
//! epochs therefore disappears without a background pass ever running.
//!
//! **Saturation is reported, not hidden.** A term whose neighbourhood is
//! full increments `dropped` rather than evicting someone else's counts.
//! An observer quietly losing observations would make the clustering
//! planner confident about a workload it never saw.
//!
//! Entry points take `Copy` arguments only and are called once per planned
//! scan and once per finished scan, never per row.

use std::sync::OnceLock;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

use crate::predicate::{CompareOp, LakePredicate};

/// Seconds one epoch covers.
pub const EPOCH_SECONDS: u64 = 300;
/// Epochs retained. Older observations are reclaimed by the next write into
/// their slot.
pub const EPOCHS: usize = 8;
/// Weight multiplier per epoch of age, so recent evidence outweighs old.
pub const DECAY: f64 = 0.8;

/// Counter slots. Open addressed, so a lookup is a hash and at most
/// `PROBE_LIMIT` cache-line reads.
const SLOTS: usize = 2048;
/// How far a probe walks before reporting the term as dropped.
const PROBE_LIMIT: usize = 8;

/// Bits the value occupies. The tag takes the rest.
const VALUE_BITS: u32 = 48;
const VALUE_MASK: u64 = (1u64 << VALUE_BITS) - 1;

/// The epoch a wall-clock second falls in.
#[inline]
pub fn epoch_of(unix_seconds: u64) -> u16 {
    ((unix_seconds / EPOCH_SECONDS) & 0xFFFF) as u16
}

/// The epoch now falls in, read from the wall clock.
#[inline]
pub fn current_epoch() -> u16 {
    epoch_of(
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0),
    )
}

/// Counters one column carries. The observer counts by a flat `u32` term,
/// so the column id is scaled and the low bits name what is counted
pub const TERMS_PER_COLUMN: u32 = 7;
/// Scans whose predicate compared this column for equality
pub const TERM_EQUALITY: u32 = 0;
/// Scans whose predicate compared this column for a range
pub const TERM_RANGE: u32 = 1;
/// Bytes those scans could have read
pub const TERM_BYTES_CONSIDERED: u32 = 2;
/// Bytes file statistics let those scans skip
pub const TERM_BYTES_SKIPPED: u32 = 3;
/// Rows those scans decoded
pub const TERM_ROWS_SCANNED: u32 = 4;
/// Rows those scans returned
pub const TERM_ROWS_MATCHED: u32 = 5;

/// Joins whose equality key reached this column.
///
/// Kept apart from `TERM_EQUALITY` because the two ask the layout for
/// different things. An equality filter carries a constant, so file bounds
/// on the column reject files outright. A join key carries no constant at
/// all: what it wants is for both sides to be ordered by it, so a file on
/// one side can be matched against the few files on the other whose ranges
/// overlap it. Both are reasons to order by the column, and folding them
/// into one counter would lose which one asked
pub const TERM_JOIN_KEY: u32 = 6;

/// The term id one column's counter lives under.
#[inline]
pub fn column_term(column_id: u32, slot: u32) -> u32 {
    column_id
        .saturating_mul(TERMS_PER_COLUMN)
        .saturating_add(slot)
}

/// The column and slot a term id names, the inverse of `column_term`.
#[inline]
pub fn term_column(term: u32) -> (u32, u32) {
    (term / TERMS_PER_COLUMN, term % TERMS_PER_COLUMN)
}

// The observer is process wide, like the lake log registry, because the
// evidence belongs to the node rather than to a session or a connection
static OBSERVER: OnceLock<WorkloadObserver> = OnceLock::new();

/// The node's workload observer.
///
/// Stays on in every deployment mode, including `db`, because Adaptive
/// Clustering governs the heap fold tier as well as lake layouts.
#[inline]
pub fn observer() -> &'static WorkloadObserver {
    OBSERVER.get_or_init(WorkloadObserver::new)
}

/// Records one join: the columns its equality keys reached, on both
/// tables.
///
/// Called once when a join is planned, never per row. Both sides are
/// credited because a join is only co-located when both are ordered by the
/// key: clustering one side and not the other leaves every file on the
/// clustered side matched against every file on the other, which is what
/// the layout was supposed to remove.
///
/// `keys` is resolved column pairs rather than the join expression. The
/// planner has already reduced the ON clause to equi-keys by the time a
/// join is costed, and this crate has no binder to reduce one itself, so
/// passing the expression would mean a second reducer to keep in step with
/// the first.
///
/// A pair naming a column on a table that is not a lake table is harmless:
/// the observation lands on a table id nothing reads evidence for
#[inline]
pub fn observe_join(left_table: u32, right_table: u32, keys: &[(u32, u32)], epoch: u16) {
    let observer = observer();
    for (left_column, right_column) in keys {
        observer.observe(
            left_table,
            column_term(*left_column, TERM_JOIN_KEY),
            epoch,
            1,
        );
        observer.observe(
            right_table,
            column_term(*right_column, TERM_JOIN_KEY),
            epoch,
            1,
        );
    }
}

/// Records one planned scan: which columns its predicate touched, and how
/// many bytes the file statistics let it skip.
///
/// Called once when a scan is planned and never per row. Lake pruning is
/// decided entirely from the manifest before any file is opened, so the
/// skip measurement is complete here.
///
/// A predicate that names a column more than once weighs it more than
/// once, which is what it means. Both byte counters move together, so
/// their ratio, the skip rate, is unaffected.
#[inline]
pub fn observe_scan(
    table_id: u32,
    predicate: &LakePredicate,
    bytes_considered: u64,
    bytes_skipped: u64,
    epoch: u16,
) {
    observe_terms(
        observer(),
        table_id,
        predicate,
        Counts {
            considered: bytes_considered,
            skipped: bytes_skipped,
            scanned: 0,
            matched: 0,
        },
        epoch,
    );
}

/// Records one finished scan: how many rows it decoded and how many it
/// returned.
///
/// Called once when the scan is exhausted, never per row, and it is the
/// only place selectivity can be known: how much a predicate skips is
/// decided from statistics before any file opens, but how much it
/// actually selects is only known once the rows have been read.
///
/// Both numbers are needed and they answer different questions. Skip rate
/// says how well the current layout serves the predicate. Selectivity
/// says how much of the table the predicate wants, which is what lets
/// replay place its probe where the real workload's constants sit rather
/// than guessing a constant and scoring a query nobody ran.
#[inline]
pub fn observe_scan_result(
    table_id: u32,
    predicate: &LakePredicate,
    rows_scanned: u64,
    rows_matched: u64,
    epoch: u16,
) {
    observe_terms(
        observer(),
        table_id,
        predicate,
        Counts {
            considered: 0,
            skipped: 0,
            scanned: rows_scanned,
            matched: rows_matched,
        },
        epoch,
    );
}

/// What one observation contributes to each of a column's counters
#[derive(Debug, Clone, Copy)]
struct Counts {
    considered: u64,
    skipped: u64,
    scanned: u64,
    matched: u64,
}

/// Same walk against a caller-supplied observer, so a test can drive the
/// evidence path without sharing the process-wide instance with every
/// other test in the binary
#[cfg(test)]
pub fn observe_for_test(
    observer: &WorkloadObserver,
    table_id: u32,
    predicate: &LakePredicate,
    bytes_considered: u64,
    bytes_skipped: u64,
    rows_scanned: u64,
    rows_matched: u64,
    epoch: u16,
) {
    observe_terms(
        observer,
        table_id,
        predicate,
        Counts {
            considered: bytes_considered,
            skipped: bytes_skipped,
            scanned: rows_scanned,
            matched: rows_matched,
        },
        epoch,
    );
}

fn observe_terms(
    observer: &WorkloadObserver,
    table_id: u32,
    predicate: &LakePredicate,
    counts: Counts,
    epoch: u16,
) {
    let leaf = |column_id: u32, slot: u32| {
        observer.observe(table_id, column_term(column_id, slot), epoch, 1);
        for (term, value) in [
            (TERM_BYTES_CONSIDERED, counts.considered),
            (TERM_BYTES_SKIPPED, counts.skipped),
            (TERM_ROWS_SCANNED, counts.scanned),
            (TERM_ROWS_MATCHED, counts.matched),
        ] {
            observer.observe(table_id, column_term(column_id, term), epoch, value);
        }
    };
    match predicate {
        LakePredicate::Compare { column_id, op, .. } => {
            let slot = match op {
                // Inequality selects nearly everything, so counting it as
                // an equality would ask for a layout that cannot help it
                CompareOp::Eq => TERM_EQUALITY,
                _ => TERM_RANGE,
            };
            leaf(*column_id, slot);
        }
        // Membership is an OR of equalities and prunes the same way
        LakePredicate::In { column_id, .. } => leaf(*column_id, TERM_EQUALITY),
        LakePredicate::IsNull { column_id } | LakePredicate::IsNotNull { column_id } => {
            leaf(*column_id, TERM_RANGE)
        }
        LakePredicate::And(list) | LakePredicate::Or(list) => {
            for inner in list {
                observe_terms(observer, table_id, inner, counts, epoch);
            }
        }
        LakePredicate::Not(inner) => observe_terms(observer, table_id, inner, counts, epoch),
    }
}

#[inline]
fn tag_of(counter: u64) -> u16 {
    (counter >> VALUE_BITS) as u16
}

#[inline]
fn value_of(counter: u64) -> u64 {
    counter & VALUE_MASK
}

#[inline]
fn pack(tag: u16, value: u64) -> u64 {
    ((tag as u64) << VALUE_BITS) | value.min(VALUE_MASK)
}

/// One term's counters, cache-line aligned so two hot terms never share a
/// line and false-share their way into a slowdown.
#[repr(align(64))]
struct Cell {
    /// Owning key, zero when the cell is free. Claimed once by CAS and
    /// never reassigned, so a reader that matches the key can trust every
    /// counter under it
    key: AtomicU64,
    epochs: [AtomicU64; EPOCHS],
}

impl Cell {
    const fn new() -> Self {
        #[allow(clippy::declare_interior_mutable_const)]
        const ZERO: AtomicU64 = AtomicU64::new(0);
        Self {
            key: AtomicU64::new(0),
            epochs: [ZERO; EPOCHS],
        }
    }
}

/// What the observer had to give up.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ObserverStats {
    /// Terms whose neighbourhood was full when they were observed
    pub dropped: u64,
    /// Cells currently claimed
    pub occupied: usize,
    pub capacity: usize,
}

/// Frequency and selectivity counting over a decaying window.
///
/// No model, no learning: this counts how often each term is queried and
/// how recently, which is all the clustering planner is allowed to use.
pub struct WorkloadObserver {
    cells: Box<[Cell]>,
    dropped: AtomicU64,
}

impl Default for WorkloadObserver {
    fn default() -> Self {
        Self::new()
    }
}

impl WorkloadObserver {
    pub fn new() -> Self {
        let mut cells = Vec::with_capacity(SLOTS);
        for _ in 0..SLOTS {
            cells.push(Cell::new());
        }
        Self {
            cells: cells.into_boxed_slice(),
            dropped: AtomicU64::new(0),
        }
    }

    /// Records `weight` observations of one term in one epoch.
    ///
    /// The term is whatever the caller counts by: a column id, a predicate
    /// class, a join pair hash. Nothing here interprets it.
    #[inline]
    pub fn observe(&self, table_id: u32, term: u32, epoch: u16, weight: u64) {
        if weight == 0 {
            return;
        }
        let key = Self::key_of(table_id, term);
        let Some(cell) = self.find_or_claim(key) else {
            self.dropped.fetch_add(1, Ordering::Relaxed);
            return;
        };
        let counter = &cell.epochs[epoch as usize % EPOCHS];
        let mut current = counter.load(Ordering::Relaxed);
        loop {
            // A counter left by an older epoch is reclaimed by this write,
            // which is the whole decay mechanism
            let next = if tag_of(current) == epoch {
                pack(epoch, value_of(current).saturating_add(weight))
            } else {
                pack(epoch, weight)
            };
            match counter.compare_exchange_weak(current, next, Ordering::Relaxed, Ordering::Relaxed)
            {
                Ok(_) => return,
                Err(seen) => current = seen,
            }
        }
    }

    /// The decayed weight of one term as of `now`.
    ///
    /// Counters older than the retained window contribute nothing, and a
    /// counter tagged ahead of `now` is a wrapped epoch from a long-idle
    /// term, which also contributes nothing rather than counting as future
    /// evidence.
    pub fn score(&self, table_id: u32, term: u32, now: u16) -> f64 {
        let key = Self::key_of(table_id, term);
        let Some(cell) = self.find(key) else {
            return 0.0;
        };
        let mut total = 0.0;
        for counter in &cell.epochs {
            let packed = counter.load(Ordering::Relaxed);
            let value = value_of(packed);
            if value == 0 {
                continue;
            }
            let Some(age) = now.checked_sub(tag_of(packed)) else {
                continue;
            };
            if (age as usize) < EPOCHS {
                total += value as f64 * DECAY.powi(age as i32);
            }
        }
        total
    }

    /// Every observed term of one table with its decayed weight, heaviest
    /// first. Allocates, so it belongs to the planner rather than the query
    /// path.
    pub fn terms_for(&self, table_id: u32, now: u16) -> Vec<(u32, f64)> {
        let mut out = Vec::new();
        for cell in self.cells.iter() {
            let key = cell.key.load(Ordering::Relaxed);
            if key == 0 {
                continue;
            }
            let (owner, term) = Self::unpack(key);
            if owner != table_id {
                continue;
            }
            let score = self.score(table_id, term, now);
            if score > 0.0 {
                out.push((term, score));
            }
        }
        out.sort_by(|a, b| b.1.total_cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
        out
    }

    /// Every table the observer still holds live evidence for, ascending.
    ///
    /// Reads move what a clustering proposal should be without committing
    /// anything, so a table nobody wrote can still want a different layout.
    /// This is how a maintenance pass finds those tables without asking
    /// each one in turn: one sweep of the counter array costs the
    /// observer's fixed size, whether the node hosts ten tables or ten
    /// thousand, and a table no query has touched inside the retained
    /// window never appears
    pub fn tables_with_evidence(&self, now: u16) -> Vec<u32> {
        let mut out: Vec<u32> = Vec::new();
        for cell in self.cells.iter() {
            let key = cell.key.load(Ordering::Relaxed);
            if key == 0 {
                continue;
            }
            let live = cell.epochs.iter().any(|counter| {
                let packed = counter.load(Ordering::Relaxed);
                if value_of(packed) == 0 {
                    return false;
                }
                now.checked_sub(tag_of(packed))
                    .is_some_and(|age| (age as usize) < EPOCHS)
            });
            if live {
                out.push(Self::unpack(key).0);
            }
        }
        out.sort_unstable();
        out.dedup();
        out
    }

    pub fn stats(&self) -> ObserverStats {
        ObserverStats {
            dropped: self.dropped.load(Ordering::Relaxed),
            occupied: self
                .cells
                .iter()
                .filter(|c| c.key.load(Ordering::Relaxed) != 0)
                .count(),
            capacity: self.cells.len(),
        }
    }

    /// Key zero means free, so the TABLE half is offset by one and every
    /// bit of the term survives untouched. The only table id that cannot be
    /// represented is `u32::MAX`, which the catalog's id allocator never
    /// issues.
    #[inline]
    fn key_of(table_id: u32, term: u32) -> u64 {
        ((table_id as u64 + 1) << 32) | term as u64
    }

    /// The table and term a claimed key names.
    #[inline]
    fn unpack(key: u64) -> (u32, u32) {
        (((key >> 32) - 1) as u32, key as u32)
    }

    #[inline]
    fn home(key: u64) -> usize {
        let mut state = key;
        (zyron_common::splitMix64(&mut state) as usize) % SLOTS
    }

    /// Finds the cell owning `key`, claiming a free one on the way.
    #[inline]
    fn find_or_claim(&self, key: u64) -> Option<&Cell> {
        let home = Self::home(key);
        for step in 0..PROBE_LIMIT {
            let cell = &self.cells[(home + step) % SLOTS];
            let seen = cell.key.load(Ordering::Relaxed);
            if seen == key {
                return Some(cell);
            }
            if seen == 0 {
                match cell
                    .key
                    .compare_exchange(0, key, Ordering::Relaxed, Ordering::Relaxed)
                {
                    Ok(_) => return Some(cell),
                    // Another thread claimed it first, it may be for this
                    // key or for another
                    Err(other) if other == key => return Some(cell),
                    Err(_) => continue,
                }
            }
        }
        None
    }

    #[inline]
    fn find(&self, key: u64) -> Option<&Cell> {
        let home = Self::home(key);
        for step in 0..PROBE_LIMIT {
            let cell = &self.cells[(home + step) % SLOTS];
            let seen = cell.key.load(Ordering::Relaxed);
            if seen == key {
                return Some(cell);
            }
            if seen == 0 {
                return None;
            }
        }
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_observations_accumulate_within_an_epoch() {
        let observer = WorkloadObserver::new();
        observer.observe(1, 7, 100, 3);
        observer.observe(1, 7, 100, 5);
        assert_eq!(observer.score(1, 7, 100), 8.0);
        // A different table with the same term is a different counter
        assert_eq!(observer.score(2, 7, 100), 0.0);
        // An unobserved term scores nothing rather than failing
        assert_eq!(observer.score(1, 8, 100), 0.0);
    }

    /// Maintenance asks this which tables reads have moved the answer for,
    /// so it has to name every table with live evidence and no table
    /// whose evidence has decayed out of the window
    #[test]
    fn test_evidence_names_the_tables_reads_touched_and_no_others() {
        let observer = WorkloadObserver::new();
        assert!(
            observer.tables_with_evidence(100).is_empty(),
            "an observer nothing has been read through names no table"
        );

        observer.observe(4, column_term(0, TERM_EQUALITY), 100, 1);
        observer.observe(4, column_term(1, TERM_RANGE), 100, 1);
        observer.observe(9, column_term(0, TERM_EQUALITY), 100, 1);
        assert_eq!(
            observer.tables_with_evidence(100),
            vec![4, 9],
            "each table once, however many of its terms were observed"
        );

        // Evidence inside the retained window still counts, evidence past
        // it is gone and the table stops being a reason to re-propose
        assert_eq!(
            observer.tables_with_evidence(100 + EPOCHS as u16 - 1),
            vec![4, 9]
        );
        assert!(
            observer
                .tables_with_evidence(100 + EPOCHS as u16)
                .is_empty(),
            "a table nothing has queried for a whole window is not asked about"
        );

        // A zero weight is not an observation, so it does not resurrect one
        observer.observe(9, column_term(2, TERM_EQUALITY), 100 + EPOCHS as u16, 0);
        assert!(
            observer
                .tables_with_evidence(100 + EPOCHS as u16)
                .is_empty()
        );
    }

    #[test]
    fn test_recent_epochs_outweigh_old_ones() {
        let observer = WorkloadObserver::new();
        observer.observe(1, 7, 100, 10);
        observer.observe(1, 7, 101, 10);
        // The older epoch is discounted once
        let expected = 10.0 * DECAY + 10.0;
        assert!((observer.score(1, 7, 101) - expected).abs() < 1e-9);
    }

    #[test]
    fn test_an_old_counter_is_reclaimed_by_the_write_that_finds_it() {
        let observer = WorkloadObserver::new();
        observer.observe(1, 7, 100, 1_000);
        assert_eq!(observer.score(1, 7, 100), 1_000.0);

        // Eight epochs later the same slot comes round again. No sweeper
        // ran, the write itself reclaimed it
        observer.observe(1, 7, 100 + EPOCHS as u16, 5);
        let now = 100 + EPOCHS as u16;
        assert_eq!(
            observer.score(1, 7, now),
            5.0,
            "the old value is gone, not decayed"
        );
    }

    #[test]
    fn test_evidence_ages_out_of_the_window_entirely() {
        let observer = WorkloadObserver::new();
        observer.observe(1, 7, 100, 10);
        // Still inside the window
        assert!(observer.score(1, 7, 100 + EPOCHS as u16 - 1) > 0.0);
        // Past it, and nothing has to run for that to be true
        assert_eq!(observer.score(1, 7, 100 + EPOCHS as u16), 0.0);
    }

    #[test]
    fn test_a_full_neighbourhood_is_reported_not_silently_lost() {
        let observer = WorkloadObserver::new();
        // Fill every slot, which guarantees some term's probe window is full
        for term in 0..(SLOTS as u32 * 2) {
            observer.observe(1, term, 100, 1);
        }
        let stats = observer.stats();
        assert_eq!(stats.capacity, SLOTS);
        // Open addressing with a bounded probe fills most of the table, not
        // all of it: a term whose eight-cell window is full is dropped
        // rather than evicting a neighbour
        assert!(
            stats.occupied > SLOTS / 2 && stats.occupied <= SLOTS,
            "claimed {} of {}",
            stats.occupied,
            SLOTS
        );
        assert!(
            stats.dropped > 0,
            "a term that could not be placed must be counted"
        );
        assert_eq!(
            stats.occupied as u64 + stats.dropped,
            SLOTS as u64 * 2,
            "every observation is either placed or counted, never lost"
        );
        // What was placed is still exact
        assert_eq!(observer.score(1, 0, 100), 1.0);
    }

    #[test]
    fn test_terms_are_ranked_by_decayed_weight() {
        let observer = WorkloadObserver::new();
        observer.observe(9, 1, 100, 5);
        observer.observe(9, 2, 100, 50);
        observer.observe(9, 3, 99, 50);
        observer.observe(8, 4, 100, 999);

        let ranked = observer.terms_for(9, 100);
        assert_eq!(ranked.len(), 3, "another table's terms are not mixed in");
        assert_eq!(ranked[0].0, 2, "the heaviest current term leads");
        assert_eq!(ranked[1].0, 3, "then the same weight one epoch older");
        assert_eq!(ranked[2].0, 1);
        assert!(ranked[1].1 < ranked[0].1);
    }

    #[test]
    fn test_a_zero_weight_observation_is_not_recorded() {
        let observer = WorkloadObserver::new();
        observer.observe(1, 7, 100, 0);
        assert_eq!(observer.score(1, 7, 100), 0.0);
        assert_eq!(observer.stats().occupied, 0, "no cell was claimed");
    }

    #[test]
    fn test_concurrent_observers_lose_no_counts() {
        let observer = std::sync::Arc::new(WorkloadObserver::new());
        let threads: Vec<_> = (0..8)
            .map(|_| {
                let observer = std::sync::Arc::clone(&observer);
                std::thread::spawn(move || {
                    for _ in 0..1_000 {
                        observer.observe(1, 7, 100, 1);
                    }
                })
            })
            .collect();
        for thread in threads {
            thread.join().expect("thread");
        }
        assert_eq!(
            observer.score(1, 7, 100),
            8_000.0,
            "compare and swap must not drop a count"
        );
    }

    #[test]
    fn test_epochs_come_from_the_clock_the_caller_reads() {
        assert_eq!(epoch_of(0), 0);
        assert_eq!(epoch_of(EPOCH_SECONDS - 1), 0);
        assert_eq!(epoch_of(EPOCH_SECONDS), 1);
        assert_eq!(epoch_of(EPOCH_SECONDS * 3 + 7), 3);
    }

    /// Concurrent observers of one term converge on a single cell.
    ///
    /// The cell array is open addressed with linear probing, the same shape
    /// that let the buffer pool's page table put one key in two slots. Here
    /// the claim publishes the key in one compare-exchange rather than
    /// reserving first, and a loser whose slot went to its own key keeps that
    /// cell instead of probing on, so one term stays one cell. This pins that
    /// down against a future two-step claim.
    #[test]
    fn test_concurrent_observers_of_one_term_share_one_cell() {
        use std::sync::Arc;

        for round in 0..100u32 {
            let observer = Arc::new(WorkloadObserver::new());
            let barrier = Arc::new(std::sync::Barrier::new(8));
            let mut handles = Vec::new();
            for _ in 0..8 {
                let observer = Arc::clone(&observer);
                let barrier = Arc::clone(&barrier);
                handles.push(std::thread::spawn(move || {
                    barrier.wait();
                    observer.observe(1, round, 0, 1);
                }));
            }
            for h in handles {
                h.join().expect("observer thread");
            }

            let stats = observer.stats();
            assert_eq!(
                stats.occupied, 1,
                "round {round}: one term must occupy one cell"
            );
            assert_eq!(stats.dropped, 0, "round {round}: nothing was dropped");
            let terms = observer.terms_for(1, 0);
            assert_eq!(terms.len(), 1, "round {round}: one term reported once");
            assert_eq!(terms[0].0, round);
        }
    }

    /// Distinct terms observed at once each get their own cell, so probing
    /// under contention neither merges them nor loses one.
    #[test]
    fn test_concurrent_observers_of_distinct_terms_keep_them_apart() {
        use std::sync::Arc;

        let observer = Arc::new(WorkloadObserver::new());
        let barrier = Arc::new(std::sync::Barrier::new(8));
        let mut handles = Vec::new();
        for t in 0..8u32 {
            let observer = Arc::clone(&observer);
            let barrier = Arc::clone(&barrier);
            handles.push(std::thread::spawn(move || {
                barrier.wait();
                for _ in 0..50 {
                    observer.observe(1, t, 0, 1);
                }
            }));
        }
        for h in handles {
            h.join().expect("observer thread");
        }

        let mut terms = observer.terms_for(1, 0);
        terms.sort_by_key(|(t, _)| *t);
        assert_eq!(terms.len(), 8, "every term kept its own cell");
        for (i, (term, score)) in terms.iter().enumerate() {
            assert_eq!(*term, i as u32);
            assert!(*score > 0.0, "term {term} recorded no weight");
        }
        assert_eq!(observer.stats().dropped, 0);
    }
}
