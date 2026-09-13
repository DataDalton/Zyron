//! Tests the CDC stream driver. It reads change records past the change
//! stream's position, delivers them to the sink in batches, and answers with
//! the version the position moves to. Proves the sinks have a real driving
//! caller and that delivery is idempotent across passes.

use std::sync::Mutex;

use bytes::Bytes;
use zyron_cdc::cdc_stream::{
    CdcOutputStream, CdcSink, CdcSinkConfig, StreamRetryPolicy, TxnDecision, drive_stream_once,
};
use zyron_cdc::decoder::{DecodedChange, DecoderPlugin};
use zyron_cdc::{ChangeDataFeed, ChangeRecord, ChangeType};
use zyron_common::Result;

/// Test sink that records each delivered batch
struct CollectingSink {
    batches: Mutex<Vec<Vec<Vec<u8>>>>,
}

impl CollectingSink {
    fn new() -> Self {
        Self {
            batches: Mutex::new(Vec::new()),
        }
    }
}

impl CdcSink for CollectingSink {
    fn write_batch(&self, changes: &[Bytes]) -> Result<()> {
        self.batches
            .lock()
            .expect("lock")
            .push(changes.iter().map(|b| b.to_vec()).collect());
        Ok(())
    }
    fn flush(&self) -> Result<()> {
        Ok(())
    }
}

fn record(version: u64, payload: &str) -> ChangeRecord {
    ChangeRecord {
        change_type: ChangeType::Insert,
        commit_version: version,
        commit_timestamp: 1_000 + version as i64,
        table_id: 7,
        txn_id: version,
        change_ordinal: 0,
        schema_version: 1,
        row_data: payload.as_bytes().to_vec(),
        primary_key_data: Vec::new(),
        is_last_in_txn: true,
        projected: false,
    }
}

fn decode_record(rec: &ChangeRecord) -> Result<DecodedChange> {
    Ok(DecodedChange {
        table_name: "orders".into(),
        table_id: rec.table_id,
        operation: rec.change_type,
        old_values: None,
        new_values: Some(vec![(
            "payload".into(),
            String::from_utf8_lossy(&rec.row_data).into_owned(),
        )]),
        commit_lsn: rec.commit_version,
        commit_timestamp: rec.commit_timestamp,
        txn_id: rec.txn_id,
        is_last_in_txn: rec.is_last_in_txn,
        schema_version: rec.schema_version,
    })
}

fn make_stream() -> CdcOutputStream {
    CdcOutputStream {
        name: "s".into(),
        table_id: 7,
        change_stream: "__cdc_s".into(),
        sink: CdcSinkConfig::Webhook {
            url: "http://unused".into(),
            headers: vec![],
            batch_size: 10,
        },
        decoder_plugin: DecoderPlugin::ZyronCdc,
        filter: None,
        include_columns: None,
        batch_size: 2,
        batch_interval_ms: 100,
        active: true,
        retry_policy: StreamRetryPolicy::default(),
    }
}

fn open_feed(tmp: &tempfile::TempDir) -> ChangeDataFeed {
    std::fs::create_dir_all(tmp.path().join("cdf")).expect("cdf dir");
    let feed = ChangeDataFeed::open(&tmp.path().join("cdf"), 7, 7).expect("opens");
    feed.enable();
    feed
}

#[test]
fn driver_delivers_batches_and_reports_the_version_reached() {
    let tmp = tempfile::TempDir::new().expect("temp dir");
    let feed = open_feed(&tmp);
    feed.append_change(&record(1, "r1")).expect("appends");
    feed.append_change(&record(2, "r2")).expect("appends");
    feed.append_change(&record(3, "r3")).expect("appends");

    let stream = make_stream();
    let sink = CollectingSink::new();

    let pass = drive_stream_once(&stream, &feed, 0, &sink, decode_record, &|_| {
        TxnDecision::Committed
    })
    .expect("drives");
    assert_eq!(pass.delivered, 3, "all three records delivered");
    assert_eq!(
        pass.complete_version, 3,
        "the position moves to the last version"
    );

    // batch_size = 2, so the three records arrive as batches of 2 then 1.
    let batches = sink.batches.lock().expect("lock");
    assert_eq!(batches.len(), 2, "two batches: {:?}", batches.len());
    assert_eq!(batches[0].len(), 2);
    assert_eq!(batches[1].len(), 1);
    drop(batches);

    // A second pass from the version reached delivers nothing
    let again = drive_stream_once(
        &stream,
        &feed,
        pass.complete_version,
        &sink,
        decode_record,
        &|_| TxnDecision::Committed,
    )
    .expect("drives again");
    assert_eq!(
        again.delivered, 0,
        "no redelivery of already-confirmed records"
    );
    assert_eq!(again.complete_version, 3, "the position stays");
}

#[test]
fn driver_resumes_from_the_position() {
    let tmp = tempfile::TempDir::new().expect("temp dir");
    let feed = open_feed(&tmp);
    feed.append_change(&record(1, "r1")).expect("appends");

    let stream = make_stream();
    let sink = CollectingSink::new();

    let first = drive_stream_once(&stream, &feed, 0, &sink, decode_record, &|_| {
        TxnDecision::Committed
    })
    .expect("drives");
    assert_eq!(first.delivered, 1);

    // New records appended after the first pass are picked up on the next one.
    feed.append_change(&record(2, "r2")).expect("appends");
    feed.append_change(&record(3, "r3")).expect("appends");
    let second = drive_stream_once(
        &stream,
        &feed,
        first.complete_version,
        &sink,
        decode_record,
        &|_| TxnDecision::Committed,
    )
    .expect("drives");
    assert_eq!(second.delivered, 2, "only the new records are delivered");
    assert_eq!(second.complete_version, 3);
}

// Change records land in the feed at execution time, before their
// transaction decides. A rolled back transaction's records must never reach
// the sink, and the position still moves past them so they never redeliver
#[test]
fn driver_skips_aborted_transactions() {
    let tmp = tempfile::TempDir::new().expect("temp dir");
    let feed = open_feed(&tmp);
    feed.append_change(&record(1, "r1")).expect("appends");
    feed.append_change(&record(2, "rolled-back"))
        .expect("appends");
    feed.append_change(&record(3, "r3")).expect("appends");

    let stream = make_stream();
    let sink = CollectingSink::new();

    let pass = drive_stream_once(&stream, &feed, 0, &sink, decode_record, &|txn| {
        if txn == 2 {
            TxnDecision::Aborted
        } else {
            TxnDecision::Committed
        }
    })
    .expect("drives");

    assert_eq!(
        pass.delivered, 2,
        "the aborted transaction's record is skipped"
    );
    assert_eq!(
        pass.complete_version, 3,
        "the position moves past the aborted record"
    );
    let batches = sink.batches.lock().expect("lock");
    let all: Vec<&[u8]> = batches.iter().flatten().map(|b| b.as_slice()).collect();
    assert!(
        all.iter()
            .all(|b| !b.windows(11).any(|w| w == b"rolled-back"))
    );
}

// An undecided transaction holds the pass before its version begins, so a
// change that may yet roll back never reaches the sink and the position
// never passes it
#[test]
fn driver_stops_before_an_undecided_transaction() {
    let tmp = tempfile::TempDir::new().expect("temp dir");
    let feed = open_feed(&tmp);
    feed.append_change(&record(1, "r1")).expect("appends");
    feed.append_change(&record(2, "open")).expect("appends");
    feed.append_change(&record(3, "r3")).expect("appends");

    let stream = make_stream();
    let sink = CollectingSink::new();
    let pass = drive_stream_once(&stream, &feed, 0, &sink, decode_record, &|txn| {
        if txn == 2 {
            TxnDecision::InFlight
        } else {
            TxnDecision::Committed
        }
    })
    .expect("drives");
    assert_eq!(pass.delivered, 1);
    assert_eq!(
        pass.complete_version, 1,
        "the position stops below the open transaction"
    );

    // Once it commits, the next pass takes it and what followed
    let rest = drive_stream_once(
        &stream,
        &feed,
        pass.complete_version,
        &sink,
        decode_record,
        &|_| TxnDecision::Committed,
    )
    .expect("drives");
    assert_eq!(rest.delivered, 2);
    assert_eq!(rest.complete_version, 3);
}
