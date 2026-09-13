//! Change feeds and change streams at the crate boundary.
//!
//! What these prove is the feed's configuration and the stream runtime built
//! over it. Before images off halve an update-heavy feed and make the
//! preimage unavailable, a column subset records only what it names, both
//! codecs round-trip, retention written as an interval and as days agree,
//! the pending counters answer without opening a change file, staleness is
//! found by the sweeper before a reader arrives, a reset outside retention is
//! refused, every alert fires on its condition, a byte cap purges oldest
//! first while the feed keeps accepting writes, and a sweep over ten
//! thousand streams stays inside its budget.
//!
//! Run: cargo test -p zyron-cdc --test change_stream_test -- --nocapture

use std::sync::Arc;

use tempfile::TempDir;
use zyron_catalog::{
    ChangeStreamEntry, ChangeStreamMode, ChangeStreamOrigin, ChangeStreamSource, DatabaseId,
    SchemaId, StreamPosition,
};
use zyron_cdc::change_stream::{self, ALERT_TEMPLATES, ChangeStreamRuntime, ResetTarget};
use zyron_cdc::{
    CdfCodec, CdfRegistry, ChangeRange, ChangeRecord, ChangeType, DerivedBoundary, FeedConfig,
    SourceRead,
};

fn record(table_id: u32, version: u64, kind: ChangeType, row: Vec<u8>) -> ChangeRecord {
    ChangeRecord {
        change_type: kind,
        commit_version: version,
        commit_timestamp: version as i64 * 1_000_000,
        table_id,
        txn_id: version,
        change_ordinal: 0,
        schema_version: 1,
        row_data: row,
        primary_key_data: Vec::new(),
        is_last_in_txn: true,
        projected: false,
    }
}

fn insert(table_id: u32, version: u64) -> ChangeRecord {
    record(table_id, version, ChangeType::Insert, vec![1; 64])
}

/// An update as the capture path writes it, the preimage then the
/// postimage, adjacent under one version
fn update(table_id: u32, version: u64) -> [ChangeRecord; 2] {
    [
        record(table_id, version, ChangeType::UpdatePreimage, vec![2; 64]),
        record(table_id, version, ChangeType::UpdatePostimage, vec![3; 64]),
    ]
}

fn stream(id: u32, table_id: u32, position: u64, consumed: u64) -> ChangeStreamEntry {
    ChangeStreamEntry {
        id,
        catalog_id: DatabaseId(1),
        schema_id: SchemaId(1),
        name: format!("s{id}"),
        source: ChangeStreamSource::Table(table_id),
        position: vec![StreamPosition {
            table_id,
            version: position,
            consumed,
        }],
        created_at: 0,
        created_from: ChangeStreamOrigin::Now,
        mode: ChangeStreamMode::Standard,
        predicate: None,
        columns: None,
        owner_id: 1,
        last_advanced_at: 0,
        last_advanced_by: 0,
        stale: false,
        stale_reason: String::new(),
        needs_attention: false,
        attention_reason: String::new(),
        initial_rows_pending: false,
        branch: None,
    }
}

fn registry() -> (Arc<CdfRegistry>, TempDir) {
    let tmp = TempDir::new().expect("temp dir");
    (Arc::new(CdfRegistry::new(tmp.path().to_path_buf())), tmp)
}

#[test]
fn before_images_off_halve_an_update_heavy_feed() {
    let (feeds, _tmp) = registry();
    // Uncompressed on both sides, so the bytes measure the images written
    // rather than how well two near-identical images compress together
    let with = feeds
        .enable_with_config(
            1,
            FeedConfig {
                codec: CdfCodec::None,
                ..FeedConfig::default()
            },
        )
        .expect("feed with before images");
    let without = feeds
        .enable_with_config(
            2,
            FeedConfig {
                before_image: false,
                codec: CdfCodec::None,
                ..FeedConfig::default()
            },
        )
        .expect("feed without before images");
    for version in 1..=500u64 {
        with.append_batch(&update(1, version)).expect("appends");
        without.append_batch(&update(2, version)).expect("appends");
    }
    with.seal_open_segment().expect("seals");
    without.seal_open_segment().expect("seals");
    assert_eq!(with.record_count(), 1000);
    assert_eq!(without.record_count(), 500, "one image per update");
    let ratio = without.file_size_bytes() as f64 / with.file_size_bytes() as f64;
    println!(
        "feed bytes with before images {} without {} ratio {ratio:.3}",
        with.file_size_bytes(),
        without.file_size_bytes()
    );
    assert!(
        ratio <= 0.55,
        "before images off keeps at most 55% of the bytes, kept {ratio:.3}"
    );

    // The preimage kind is never in the feed, which is what the planner
    // refuses a read of by name
    let preimages = without
        .read_range(&ChangeRange::everything().with_change_types(&[ChangeType::UpdatePreimage]))
        .expect("reads");
    assert!(preimages.is_empty());
}

#[test]
fn a_column_subset_records_only_what_it_names_and_the_identity() {
    let (feeds, _tmp) = registry();
    let feed = feeds
        .enable_with_config(
            3,
            FeedConfig {
                columns: Some(vec![0, 2]),
                ..FeedConfig::default()
            },
        )
        .expect("narrowed feed");
    assert_eq!(feed.config().columns, Some(vec![0, 2]));
    // A change touching none of the recorded columns still records the row,
    // so a consumer sees that it changed
    let mut identity = record(3, 1, ChangeType::UpdatePostimage, vec![7; 8]);
    identity.projected = true;
    feed.append_batch(&[identity]).expect("appends");
    let rows = feed.read_range(&ChangeRange::everything()).expect("reads");
    assert_eq!(rows.len(), 1);
    assert!(rows[0].projected, "the record says it carries the subset");
}

#[test]
fn both_codecs_round_trip_a_sealed_segment() {
    for codec in [CdfCodec::Lz4, CdfCodec::Zstd, CdfCodec::None] {
        let (feeds, _tmp) = registry();
        let feed = feeds
            .enable_with_config(
                4,
                FeedConfig {
                    codec,
                    ..FeedConfig::default()
                },
            )
            .expect("feed");
        let records: Vec<ChangeRecord> = (1..=300).map(|v| insert(4, v)).collect();
        feed.append_batch(&records).expect("appends");
        feed.seal_open_segment().expect("seals");
        let back = feed.read_range(&ChangeRange::everything()).expect("reads");
        assert_eq!(back.len(), 300, "{codec:?} read every record back");
        assert!(
            back.iter()
                .zip(records.iter())
                .all(|(a, b)| a.row_data == b.row_data && a.commit_version == b.commit_version),
            "{codec:?} preserved the rows"
        );
    }
}

#[test]
fn retention_as_an_interval_and_as_days_agree() {
    let from_days = FeedConfig::from_retention_days(3);
    assert_eq!(
        from_days.retention_micros,
        3 * zyron_cdc::change_feed::MICROS_PER_DAY
    );
    assert_eq!(from_days.retention_days(), 3);
    let (feeds, _tmp) = registry();
    let feed = feeds.enable_with_config(5, from_days).expect("feed");
    let mut config = feed.config();
    config.retention_micros = 36 * 60 * 60 * 1_000_000;
    feed.set_config(config).expect("stores the interval");
    assert_eq!(feed.config().retention_micros, 36 * 60 * 60 * 1_000_000);
    // A day count written afterward converts to the same unit
    assert_eq!(
        feed.retention_days(),
        1,
        "an interval below two days reads as one day"
    );
}

#[test]
fn pending_counters_answer_without_opening_a_change_file() {
    let (feeds, _tmp) = registry();
    let feed = feeds.enable_for_table(6, 7).expect("feed");
    let records: Vec<ChangeRecord> = (1..=5000).map(|v| insert(6, v)).collect();
    feed.append_batch(&records).expect("appends");
    feed.seal_open_segment().expect("seals");
    let runtime = ChangeStreamRuntime::new(Arc::clone(&feeds));
    let entry = stream(1, 6, 1200, 1200);

    let started = std::time::Instant::now();
    let pending = runtime.pending_rows(&entry);
    let versions = runtime.pending_versions(&entry);
    let lag = runtime.lag_seconds(&entry, 10_000 * 1_000_000);
    let took = started.elapsed();
    assert_eq!(pending, 3800);
    assert_eq!(versions, 3800);
    assert_eq!(
        lag,
        10_000 - 1201,
        "the age of the oldest unconsumed change"
    );
    println!("pending counters for 5000 records answered in {took:?}");
    assert!(
        took < std::time::Duration::from_millis(5),
        "answered from counters, took {took:?}"
    );
}

#[test]
fn staleness_is_found_by_the_sweeper_and_a_reset_recovers_it() {
    let (feeds, _tmp) = registry();
    let feed = feeds.enable_for_table(7, 7).expect("feed");
    for version in 1..=40u64 {
        feed.append_batch(&[insert(7, version)]).expect("appends");
        if version % 10 == 0 {
            feed.seal_open_segment().expect("seals");
        }
    }
    let runtime = ChangeStreamRuntime::new(Arc::clone(&feeds));
    let behind = Arc::new(stream(1, 7, 5, 5));
    let ahead = Arc::new(stream(2, 7, 35, 35));

    // Retention reclaims everything below version 21
    let removed = feed.purge_before_version(21).expect("purges");
    assert_eq!(removed, 20);
    assert!(feed.purge_floor() >= 20);

    let swept = runtime.sweep(&[Arc::clone(&behind), Arc::clone(&ahead)]);
    assert_eq!(
        swept.len(),
        1,
        "only the stream that lost changes is flagged"
    );
    assert!(swept[0].stale);
    assert_eq!(swept[0].stale_reason, "retention_passed");

    // A read names the position, the oldest change held and the reset
    let refused = runtime
        .plan_read(&behind, None)
        .expect_err("a stale stream cannot be read");
    let message = refused.to_string();
    assert!(message.contains("ChangeStreamStale"), "{message}");
    assert!(message.contains("is version 5"), "{message}");
    assert!(
        message.contains("oldest change still held is version 21"),
        "{message}"
    );
    assert!(message.contains("RESET TO VERSION 21"), "{message}");

    // A reset outside retention is refused, one inside recovers the stream
    let outside = change_stream::reset_to(&runtime, &behind, ResetTarget::Version(10), 1, 0)
        .expect_err("below retention");
    assert!(outside.to_string().contains("oldest"), "{outside}");
    let recovered = change_stream::reset_to(&runtime, &behind, ResetTarget::Earliest, 1, 0)
        .expect("resets to the oldest change held");
    assert!(!recovered.stale);
    assert!(runtime.staleness(&recovered).is_none());
    let plan = runtime.plan_read(&recovered, None).expect("reads again");
    assert_eq!(runtime.read(&plan).expect("reads").len(), 20);

    // The sweeper also clears a stream that was reset by hand
    let mut by_hand = ChangeStreamEntry::clone(&behind);
    by_hand.stale = true;
    by_hand.stale_reason = "retention_passed".into();
    by_hand.set_position(7, 30, 30);
    let cleared = runtime.sweep(&[Arc::new(by_hand)]);
    assert_eq!(cleared.len(), 1);
    assert!(!cleared[0].stale);
}

#[test]
fn an_append_only_stream_does_not_go_stale_when_updates_are_reclaimed() {
    let (feeds, _tmp) = registry();
    let feed = feeds.enable_for_table(8, 7).expect("feed");
    for version in 1..=10u64 {
        feed.append_batch(&update(8, version)).expect("appends");
    }
    feed.seal_open_segment().expect("seals");
    feed.purge_before_version(6).expect("purges");
    let runtime = ChangeStreamRuntime::new(Arc::clone(&feeds));
    let mut entry = stream(1, 8, 2, 4);
    entry.mode = ChangeStreamMode::AppendOnly;
    assert!(
        runtime.staleness(&entry).is_none(),
        "nothing it would yield was reclaimed"
    );
    let plan = runtime.plan_read(&entry, None).expect("plans");
    assert_eq!(plan.admitted_kinds(), Some(vec![ChangeType::Insert]));
}

#[test]
fn every_alert_template_fires_on_its_condition() {
    let names: Vec<&str> = ALERT_TEMPLATES.iter().map(|t| t.name).collect();
    for expected in [
        "cdc_stream_lag",
        "cdc_stream_stale",
        "cdc_stream_needs_attention",
        "cdf_retention_pressure",
        "cdc_apply_failed",
    ] {
        assert!(names.contains(&expected), "{expected} is declared");
    }

    let (feeds, _tmp) = registry();
    let feed = feeds.enable_for_table(9, 1).expect("feed");
    for version in 1..=30u64 {
        feed.append_batch(&[insert(9, version)]).expect("appends");
    }
    let runtime = ChangeStreamRuntime::new(Arc::clone(&feeds));
    let now = 100 * 1_000_000;

    // Lag by rows
    let behind = stream(1, 9, 0, 0);
    let status = runtime.status(&behind, now);
    let fired = change_stream::evaluate_alerts(&status, 10, 3600);
    assert!(
        fired.iter().any(|f| f.template == "cdc_stream_lag"),
        "{fired:?}"
    );

    // Lag by age with the row threshold out of reach
    let fired = change_stream::evaluate_alerts(&status, 1_000_000, 10);
    assert!(
        fired.iter().any(|f| f.template == "cdc_stream_lag"),
        "{fired:?}"
    );

    // Stale and needs attention each fire on their flag
    let mut stale = stream(2, 9, 0, 0);
    stale.stale = true;
    stale.stale_reason = "retention_passed".into();
    let status = runtime.status(&stale, now);
    let fired = change_stream::evaluate_alerts(&status, 1_000_000, 1_000_000);
    assert!(
        fired.iter().any(|f| f.template == "cdc_stream_stale"),
        "{fired:?}"
    );

    let mut attention = stream(3, 9, 30, 30);
    attention.needs_attention = true;
    attention.attention_reason = "column 'x' was dropped".into();
    let status = runtime.status(&attention, now);
    let fired = change_stream::evaluate_alerts(&status, 1_000_000, 1_000_000);
    assert!(
        fired
            .iter()
            .any(|f| f.template == "cdc_stream_needs_attention"),
        "{fired:?}"
    );

    // Retention pressure. One day of retention, the oldest unconsumed change
    // is at second one, and the margin covers the time left
    let day = zyron_cdc::change_feed::MICROS_PER_DAY;
    let near = change_stream::retention_pressure(&runtime, &behind, day, now);
    assert!(
        near.is_some(),
        "the oldest change expires inside the margin"
    );
    let far = change_stream::retention_pressure(&runtime, &behind, 60 * 1_000_000, now);
    assert!(far.is_none(), "a narrow margin does not fire yet");
}

/// A record of `txn` at `version`, for a feed whose transactions write at
/// more than one version
fn record_of(txn: u64, version: u64) -> ChangeRecord {
    let mut rec = insert(10, version);
    rec.txn_id = txn;
    rec
}

/// A bounded read ends on the version holding the record at its bound,
/// moved past every transaction that wrote inside the read and again beyond
/// it, and the spans that decide that outlive a seal and a reopen
#[test]
fn a_bounded_read_hands_over_whole_transactions() {
    let tmp = TempDir::new().expect("temp dir");
    let feed = zyron_cdc::ChangeDataFeed::open(tmp.path(), 10, 7).expect("opens");
    feed.enable();
    // Transaction 1 writes at versions 1 and 3, around transaction 2 at 2.
    // Transactions 4 and 5 write once each
    feed.append_batch(&[record_of(1, 1), record_of(1, 1)])
        .expect("appends");
    feed.append_batch(&[record_of(2, 2)]).expect("appends");
    feed.append_batch(&[record_of(1, 3)]).expect("appends");
    feed.append_batch(&[record_of(4, 4)]).expect("appends");
    feed.append_batch(&[record_of(5, 5)]).expect("appends");

    // One record's worth lands on version 1, and transaction 1 reaches to
    // version 3, so that is where the read ends
    assert_eq!(feed.bounded_cut(0, 0, 1), Some(3));
    // Three records' worth lands on version 2 and is moved the same way
    assert_eq!(feed.bounded_cut(0, 0, 3), Some(3));
    // Four records' worth lands on version 3 and nothing writes across it
    assert_eq!(feed.bounded_cut(0, 0, 4), Some(3));
    // From after version 3, one record's worth is version 4 alone
    assert_eq!(feed.bounded_cut(3, 4, 1), Some(4));
    // A bound the feed does not reach bounds nothing
    assert_eq!(feed.bounded_cut(3, 4, 2), None);
    assert_eq!(feed.bounded_cut(0, 0, 100), None);

    // The spans survive a seal, which writes them into the manifest, and a
    // reopen, which reads the open segment back
    feed.seal_open_segment().expect("seals");
    feed.append_batch(&[record_of(6, 6), record_of(4, 7)])
        .expect("appends after the seal");
    drop(feed);
    let reopened = zyron_cdc::ChangeDataFeed::open(tmp.path(), 10, 7).expect("reopens");
    let spans = reopened.txn_spans();
    let of = |txn: u64| spans.iter().find(|s| s.txn_id == txn).copied();
    assert_eq!(of(1).map(|s| (s.first, s.last)), Some((1, 3)));
    assert_eq!(of(4).map(|s| (s.first, s.last)), Some((4, 7)));
    assert_eq!(of(6).map(|s| (s.first, s.last)), Some((6, 6)));
    // After version 3, one record's worth is version 4, and transaction 4
    // reaches version 7, so the read takes versions 4 through 7
    assert_eq!(reopened.bounded_cut(3, 4, 1), Some(7));
}

#[test]
fn a_byte_cap_purges_oldest_first_and_writes_keep_landing() {
    let (feeds, _tmp) = registry();
    let feed = feeds.enable_for_table(10, 7).expect("feed");
    let mut sealed_bytes = Vec::new();
    for segment in 0..6u64 {
        let records: Vec<ChangeRecord> = (1..=200).map(|i| insert(10, segment * 200 + i)).collect();
        feed.append_batch(&records).expect("appends");
        feed.seal_open_segment().expect("seals");
        sealed_bytes.push(feed.file_size_bytes());
    }
    let total = feed.file_size_bytes();
    let cap = total / 2;
    let removed = feed.enforce_byte_cap(cap).expect("purges to the cap");
    assert!(removed > 0);
    assert!(feed.file_size_bytes() <= cap, "at or under the cap");
    let oldest = feed.oldest_version().expect("something remains");
    assert!(
        oldest > 200,
        "the oldest segments went first, oldest now {oldest}"
    );
    assert_eq!(
        feed.latest_version(),
        Some(1200),
        "the newest change is untouched"
    );

    // The table keeps accepting writes
    feed.append_batch(&[insert(10, 1201)])
        .expect("a write after the purge lands");
    assert_eq!(feed.latest_version(), Some(1201));

    // A stream below the purge is stale, one above it is not
    let runtime = ChangeStreamRuntime::new(Arc::clone(&feeds));
    assert!(runtime.staleness(&stream(1, 10, 100, 100)).is_some());
    assert!(runtime.staleness(&stream(2, 10, 1100, 1100)).is_none());

    // A stream whose unconsumed changes begin at the oldest one held is one
    // more cap purge from losing them, which is retention pressure
    let at_the_edge = stream(3, 10, oldest - 1, oldest - 1);
    let pressure = change_stream::retention_pressure(&runtime, &at_the_edge, 0, 0);
    assert_eq!(
        pressure.map(|p| p.template),
        Some("cdf_retention_pressure"),
        "the byte cap raises pressure on the stream reading from the oldest change"
    );
    let ahead = stream(4, 10, 1100, 1100);
    assert!(change_stream::retention_pressure(&runtime, &ahead, 0, 0).is_none());
}

#[test]
fn a_sweep_over_ten_thousand_streams_is_a_counter_compare() {
    let (feeds, _tmp) = registry();
    let feed = feeds.enable_for_table(11, 7).expect("feed");
    let records: Vec<ChangeRecord> = (1..=100).map(|v| insert(11, v)).collect();
    feed.append_batch(&records).expect("appends");
    feed.seal_open_segment().expect("seals");
    feed.purge_before_version(50).expect("purges");
    let runtime = ChangeStreamRuntime::new(Arc::clone(&feeds));
    let entries: Vec<Arc<ChangeStreamEntry>> = (0..10_000u32)
        .map(|i| Arc::new(stream(i, 11, (i % 100) as u64, (i % 100) as u64)))
        .collect();
    let started = std::time::Instant::now();
    let swept = runtime.sweep(&entries);
    let took = started.elapsed();
    println!(
        "sweep over {} streams took {took:?}, flagged {}",
        entries.len(),
        swept.len()
    );
    assert_eq!(
        swept.len(),
        4900,
        "every stream below the purge floor is flagged"
    );
    assert!(
        took < std::time::Duration::from_millis(200),
        "sweep took {took:?}"
    );
}

#[test]
fn a_position_replicates_as_a_count_and_re_addresses_on_another_member() {
    // Two members recorded the same changes at different versions of their
    // own, which is what a count re-addresses
    let (feeds_a, _tmp_a) = registry();
    let (feeds_b, _tmp_b) = registry();
    let a = feeds_a.enable_for_table(12, 7).expect("feed a");
    let b = feeds_b.enable_for_table(12, 7).expect("feed b");
    for i in 1..=30u64 {
        a.append_batch(&[insert(12, 100 + i)]).expect("appends");
        b.append_batch(&[insert(12, 500 + i)]).expect("appends");
    }
    let runtime_a = ChangeStreamRuntime::new(Arc::clone(&feeds_a));
    let runtime_b = ChangeStreamRuntime::new(Arc::clone(&feeds_b));
    let entry = stream(1, 12, 0, 0);
    let plan = runtime_a.plan_read(&entry, Some(112)).expect("plans");
    let advanced = change_stream::advanced(&entry, &plan, 1, 0);
    assert_eq!(advanced.consumed_of(12), 12, "twelve changes consumed");
    assert_eq!(advanced.position_of(12), 112);

    let mut on_b = ChangeStreamEntry::clone(&advanced);
    change_stream::localize_positions(&runtime_b, &mut on_b);
    assert_eq!(on_b.consumed_of(12), 12, "the count is the same number");
    assert_eq!(
        on_b.position_of(12),
        512,
        "the version is this member's own"
    );
    let plan_b = runtime_b.plan_read(&on_b, None).expect("plans on b");
    assert_eq!(
        runtime_b.read(&plan_b).expect("reads").len(),
        18,
        "the rest follows"
    );
}

/// A source whose changes are derived from a store of its own, standing
/// in for a lake table, with one record per commit and the transaction
/// each commit ran under
struct DerivedTable {
    /// Each commit's version and transaction, ascending by version, a
    /// transaction of zero for a standalone commit
    commits: Vec<(u64, u64)>,
}

impl DerivedTable {
    fn counted_through(&self, version: u64) -> u64 {
        self.commits.iter().filter(|(v, _)| *v <= version).count() as u64
    }
}

impl zyron_cdc::DerivedChangeSource for DerivedTable {
    fn latest_version(&self) -> u64 {
        self.commits.last().map(|(v, _)| *v).unwrap_or(0)
    }
    fn oldest_readable_version(&self) -> Option<u64> {
        None
    }
    fn pending_after(&self, position: u64) -> u64 {
        self.commits.len() as u64 - self.counted_through(position)
    }
    fn first_timestamp_after(&self, _position: u64) -> Option<i64> {
        None
    }
    fn version_at_timestamp(&self, _timestamp: i64) -> u64 {
        0
    }
    fn records_at_or_below(&self, version: u64) -> zyron_common::Result<u64> {
        Ok(self.counted_through(version))
    }
    fn records_before(&self, version: u64, ordinal: u64) -> zyron_common::Result<u64> {
        Ok(self.counted_through(version.saturating_sub(1)) + ordinal)
    }
    fn cursor_at_count(&self, count: u64) -> zyron_common::Result<Option<(u64, u64)>> {
        Ok(match count.checked_sub(1) {
            Some(at) => self.commits.get(at as usize).map(|(v, _)| (*v, 0)),
            None => None,
        })
    }
    fn version_at_count(&self, count: u64) -> u64 {
        match count.checked_sub(1) {
            Some(at) => self.commits.get(at as usize).map(|(v, _)| *v).unwrap_or(0),
            None => 0,
        }
    }
    fn bounded_cut(&self, consumed: u64, max_rows: u64) -> zyron_common::Result<Option<u64>> {
        let target = (consumed + max_rows) as usize;
        Ok(if target >= self.commits.len() {
            None
        } else {
            Some(self.commits[target - 1].0)
        })
    }
    fn preimages_at(&self, _version: u64) -> bool {
        true
    }
    fn boundary(&self, _ended: &dyn Fn(u64) -> bool) -> zyron_common::Result<DerivedBoundary> {
        Ok(DerivedBoundary {
            latest: self.latest_version(),
            records: self.commits.len() as u64,
            first_open: None,
        })
    }
    fn writers_above(
        &self,
        _head: u64,
        _aborted: &dyn Fn(u64) -> bool,
    ) -> zyron_common::Result<Vec<u64>> {
        Ok(Vec::new())
    }
    fn txns_in(&self, from_exclusive: u64, to_inclusive: u64) -> Vec<u64> {
        let mut out: Vec<u64> = self
            .commits
            .iter()
            .filter(|(v, txn)| *v > from_exclusive && *v <= to_inclusive && *txn != 0)
            .map(|(_, txn)| *txn)
            .collect();
        out.sort_unstable();
        out.dedup();
        out
    }
    fn span_of(&self, txn_id: u64) -> Option<(u64, u64)> {
        let mut span: Option<(u64, u64)> = None;
        for (v, txn) in &self.commits {
            if *txn == txn_id && txn_id != 0 {
                span = Some(match span {
                    Some((first, last)) => (first.min(*v), last.max(*v)),
                    None => (*v, *v),
                });
            }
        }
        span
    }
}

fn read_of(table_id: u32, from_exclusive: u64, to_inclusive: u64, limit: u64) -> SourceRead {
    SourceRead {
        table_id,
        branch: None,
        from_exclusive,
        to_inclusive,
        limit,
    }
}

/// The windows of a read over heap feeds and derived sources end where
/// every transaction inside any of them is whole in all of them. A
/// transaction with a commit above a source's limit is left for a later
/// read, and every window drops below its commits, and a window never
/// ends below the version its read starts after
#[test]
fn aligned_windows_hand_over_a_transaction_whole_across_every_source() {
    let (feeds, _tmp) = registry();
    // Heap feeds 10 and 11 share the change clock. Transaction 1 writes
    // feed 10 at version 1, transaction 2 writes both feeds at version 2,
    // transaction 3 writes feed 10 at version 3, transaction 5 writes
    // feed 11 at version 5
    let ten = feeds.enable_for_table(10, 7).expect("feed 10");
    let eleven = feeds.enable_for_table(11, 7).expect("feed 11");
    let on = |table: u32, txn: u64, version: u64| {
        let mut rec = insert(table, version);
        rec.txn_id = txn;
        rec
    };
    ten.append_batch(&[on(10, 1, 1), on(10, 2, 2), on(10, 3, 3)])
        .expect("appends to 10");
    eleven
        .append_batch(&[on(11, 2, 2), on(11, 5, 5)])
        .expect("appends to 11");
    // Derived source 20 commits its versions one through four, the second
    // under transaction 2, the third under 6 and the fourth under 3, the
    // first standalone. Derived source 21 commits under 6 and then 7
    feeds.register_derived(
        20,
        Arc::new(DerivedTable {
            commits: vec![(1, 0), (2, 2), (3, 6), (4, 3)],
        }),
    );
    feeds.register_derived(
        21,
        Arc::new(DerivedTable {
            commits: vec![(1, 6), (2, 7)],
        }),
    );

    // Nothing inside the windows reaches past them
    let aligned = feeds.align_windows(&[
        read_of(10, 0, 1, 10),
        read_of(11, 0, 1, 10),
        read_of(20, 0, 1, 10),
        read_of(21, 0, 0, 10),
    ]);
    assert_eq!(aligned, vec![1, 1, 1, 0]);

    // The heap windows reach transaction 2, which committed to source 20
    // at its version 2, so that window rises to it
    let aligned = feeds.align_windows(&[
        read_of(10, 0, 2, 10),
        read_of(11, 0, 2, 10),
        read_of(20, 0, 1, 10),
        read_of(21, 0, 0, 10),
    ]);
    assert_eq!(aligned, vec![2, 2, 2, 0]);

    // Source 20's window reaches transactions 6 and 3. Transaction 6
    // committed to source 21, whose window rises to it, and transaction 3
    // wrote heap feed 10 at version 3, so the heap windows rise together
    let aligned = feeds.align_windows(&[
        read_of(10, 0, 2, 10),
        read_of(11, 0, 2, 10),
        read_of(20, 0, 4, 10),
        read_of(21, 0, 0, 10),
    ]);
    assert_eq!(aligned, vec![3, 3, 4, 1]);

    // With the heap held at version 2, transaction 3 cannot be handed over
    // whole, so it waits and source 20's window drops below its commit,
    // while transactions 2 and 6 are still handed over whole
    let aligned = feeds.align_windows(&[
        read_of(10, 0, 2, 2),
        read_of(11, 0, 2, 2),
        read_of(20, 0, 4, 10),
        read_of(21, 0, 0, 10),
    ]);
    assert_eq!(aligned, vec![2, 2, 3, 1]);

    // A read that starts after everything a window could drop to ends
    // where it starts rather than below it
    let aligned = feeds.align_windows(&[read_of(10, 3, 3, 2), read_of(20, 3, 4, 10)]);
    assert_eq!(aligned, vec![3, 3]);
}
