//! The working-set handover that makes scale-in cheap.
//!
//! Removing a node throws away its buffer pool. The data is safe, but the
//! traffic that was being served out of that pool moves to a survivor that has
//! never read those pages, and the mesh runs slower after the scale-in than
//! before it. That cost, not correctness, is what makes operators disable
//! scale-in and pay for capacity they are not using.
//!
//! What is measured here is the thing that actually matters: the buffer pool
//! hit rate the same workload sees on the survivor, against what it saw on the
//! node that left.
//!
//! Run: cargo test -p zyron-buffer --test hot_set_handoff_test

use std::sync::atomic::{AtomicU64, Ordering};

use zyron_buffer::{BufferPool, BufferPoolConfig};
use zyron_common::page::{PAGE_SIZE, PageId};
use zyron_pressure::hot_set::{HotSetManifest, PrefetchReport};

/// Distinct pages the workload can touch.
const TABLE_PAGES: u64 = 4_000;

/// Pages that carry most of the traffic, which is the set a handover exists
/// to move.
const HOT_PAGES: u64 = 1_000;

/// Frames each pool has. Smaller than the table, so residency is a decision
/// rather than an accident, and larger than the working set, so a survivor
/// can hold what it is handed.
const POOL_FRAMES: usize = 1_200;

/// Accesses used to warm a pool with real traffic.
const WARMUP_ACCESSES: u64 = 40_000;

/// Accesses in the measured window.
///
/// Short on purpose. A cold pool eventually warms itself, so a long enough
/// measurement shows no difference and would prove nothing. What a scale-in
/// actually costs is the window right after it, and that is the window this
/// measures: twice the working set, which is a few seconds of real traffic.
const MEASURED_ACCESSES: u64 = 2_000;

/// A page image, distinct per page so a mixed-up read would be visible.
fn page_bytes(page_id: PageId) -> Vec<u8> {
    let mut data = vec![0u8; PAGE_SIZE];
    data[..8].copy_from_slice(&page_id.as_u64().to_le_bytes());
    data
}

/// Stands in for the storage layer, counting what the pool had to go to disk
/// for.
#[derive(Default)]
struct Disk {
    reads: AtomicU64,
}

impl Disk {
    fn read(&self, page_id: PageId) -> Option<Vec<u8>> {
        self.reads.fetch_add(1, Ordering::Relaxed);
        Some(page_bytes(page_id))
    }

    fn reads(&self) -> u64 {
        self.reads.load(Ordering::Relaxed)
    }

    fn reset(&self) {
        self.reads.store(0, Ordering::Relaxed);
    }
}

/// A repeatable skewed access pattern: most traffic on a small set, the rest
/// spread over the table. Deterministic, so the pre-drain and post-drain runs
/// see exactly the same sequence and the only difference between them is what
/// was in the pool when they started.
fn access_sequence(count: u64, seed: u64) -> Vec<PageId> {
    let mut out = Vec::with_capacity(count as usize);
    let mut state = seed;
    for _ in 0..count {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        let page_num = if state % 100 < 90 {
            state % HOT_PAGES
        } else {
            HOT_PAGES + state % (TABLE_PAGES - HOT_PAGES)
        };
        out.push(PageId::new(0, page_num));
    }
    out
}

/// Runs the sequence against a pool and reports the fraction served without
/// touching the disk.
fn measure_hit_rate(pool: &BufferPool, disk: &Disk, sequence: &[PageId]) -> f64 {
    let mut hits = 0u64;
    for page_id in sequence {
        if let Some(frame) = pool.fetch_page(*page_id) {
            frame.unpin();
            hits += 1;
            continue;
        }
        let bytes = disk.read(*page_id).expect("page exists");
        match pool.load_page(*page_id, &bytes) {
            Ok(frame) => {
                frame.unpin();
            }
            Err(e) => panic!("load failed: {e}"),
        }
    }
    hits as f64 / sequence.len() as f64
}

fn pool() -> BufferPool {
    BufferPool::new(BufferPoolConfig {
        num_frames: POOL_FRAMES,
    })
}

/// The whole point: a survivor handed the departing node's manifest serves the
/// same traffic at the same hit rate, and one that is not handed it does not.
#[test]
fn a_survivor_handed_the_manifest_keeps_the_hit_rate() {
    let warmup = access_sequence(WARMUP_ACCESSES, 0x2545_F491_4F6C_DD1D);
    // A different draw from the same distribution, so the measured window is
    // traffic the pool has not been handed verbatim
    let measured = access_sequence(MEASURED_ACCESSES, 0x9E37_79B9_7F4A_7C15);

    // The node that is about to drain, warm from real traffic
    let departing = pool();
    let departing_disk = Disk::default();
    measure_hit_rate(&departing, &departing_disk, &warmup);
    let pre_drain = measure_hit_rate(&departing, &departing_disk, &measured);

    let manifest = HotSetManifest::build(
        1,
        0,
        departing.hot_pages(POOL_FRAMES),
        POOL_FRAMES as u32,
        Vec::new(),
        0,
    );
    assert!(
        !manifest.pages.is_empty(),
        "a warm pool produced no working set"
    );

    // The survivor, cold, handed the manifest
    let warmed = pool();
    let warmed_disk = Disk::default();
    let pages: Vec<PageId> = manifest.page_ids().collect();
    let report = warmed.prefetch(&pages, |p| warmed_disk.read(p));
    assert!(
        report.coverage() > 0.9,
        "prefetch covered only {:.2} of the manifest: {report:?}",
        report.coverage()
    );
    // The handover's own reads happened on the drain window's schedule, in
    // manifest order. What the serving window costs is counted separately,
    // because that is the number the traffic experiences
    let handover_reads = warmed_disk.reads();
    warmed_disk.reset();
    let post_drain = measure_hit_rate(&warmed, &warmed_disk, &measured);

    // The same survivor, cold, with the handover disabled
    let cold = pool();
    let cold_disk = Disk::default();
    let without_manifest = measure_hit_rate(&cold, &cold_disk, &measured);

    assert!(
        post_drain >= pre_drain * 0.9,
        "hit rate fell from {pre_drain:.3} to {post_drain:.3} despite the handover"
    );
    assert!(
        without_manifest < pre_drain * 0.6,
        "the control run kept {without_manifest:.3} against {pre_drain:.3}, so the \
         measurement is not sensitive to the handover and proves nothing"
    );
    // The claim the hit rate rests on: while serving, the warmed survivor goes
    // to storage a fraction as often
    assert!(
        warmed_disk.reads() * 4 < cold_disk.reads(),
        "the warmed survivor still read {} against the cold one's {}",
        warmed_disk.reads(),
        cold_disk.reads()
    );
    // And the handover itself is bounded by the manifest, so a drain cannot
    // turn into an unbounded read of the whole table
    assert!(
        handover_reads <= manifest.pages.len() as u64,
        "the handover read {handover_reads} for a manifest of {}",
        manifest.pages.len()
    );
}

/// Pages the clock has seen touched outrank pages that merely happen to be
/// resident, because the manifest is usually smaller than the pool.
#[test]
fn the_manifest_prefers_pages_the_clock_saw_touched() {
    let pool = pool();
    let disk = Disk::default();

    // Fill with pages nothing returns to
    for page_num in 0..400u64 {
        let page_id = PageId::new(0, page_num);
        let bytes = disk.read(page_id).expect("page");
        let frame = pool.load_page(page_id, &bytes).expect("load");
        frame.unpin();
    }
    // Then touch a small set repeatedly, which is what sets the clock's bits
    let repeated: Vec<PageId> = (0..50u64).map(|n| PageId::new(0, n)).collect();
    for _ in 0..5 {
        for page_id in &repeated {
            if let Some(frame) = pool.fetch_page(*page_id) {
                frame.unpin();
            }
        }
    }

    let top = pool.hot_pages(50);
    assert_eq!(top.len(), 50);
    let touched = top.iter().filter(|p| p.page_num < 50).count();
    assert!(
        touched >= 45,
        "only {touched} of the top 50 were pages the workload returned to"
    );
}

/// A prefetch must never evict. The survivor's own working set is definitely
/// in use, and the manifest is a guess about traffic that has not arrived.
#[test]
fn a_prefetch_declines_rather_than_evicting_the_survivor_own_pages() {
    let busy = pool();
    let disk = Disk::default();
    // Fill the pool completely with the survivor's own pages
    for page_num in 0..POOL_FRAMES as u64 {
        let page_id = PageId::new(7, page_num);
        let bytes = disk.read(page_id).expect("page");
        let frame = busy.load_page(page_id, &bytes).expect("load");
        frame.unpin();
    }
    let resident_before = busy.page_count();

    let incoming: Vec<PageId> = (0..500u64).map(|n| PageId::new(9, n)).collect();
    let report = busy.prefetch(&incoming, |p| disk.read(p));

    assert_eq!(report.requested, 500);
    assert_eq!(report.declined, 500, "{report:?}");
    assert_eq!(report.loaded, 0);
    assert_eq!(
        busy.page_count(),
        resident_before,
        "a prefetch changed what the survivor was holding"
    );
    for page_num in 0..POOL_FRAMES as u64 {
        assert!(
            busy.contains(PageId::new(7, page_num)),
            "the survivor lost page {page_num} to a prefetch"
        );
    }
}

/// A page the departing node listed and the survivor already has costs
/// nothing, and one that has since been dropped is reported rather than
/// silently counted as covered.
#[test]
fn the_report_separates_what_was_read_from_what_was_already_there() {
    let pool = pool();
    let disk = Disk::default();
    let resident = PageId::new(0, 1);
    let bytes = disk.read(resident).expect("page");
    let frame = pool.load_page(resident, &bytes).expect("load");
    frame.unpin();

    let gone = PageId::new(0, 2);
    let fresh = PageId::new(0, 3);
    let report = pool.prefetch(&[resident, gone, fresh], |p| {
        if p == gone { None } else { disk.read(p) }
    });

    assert_eq!(
        report,
        PrefetchReport {
            requested: 3,
            loaded: 1,
            already_resident: 1,
            declined: 0,
            failed: 1,
        }
    );
    assert!((report.coverage() - 2.0 / 3.0).abs() < 1e-9);
}
