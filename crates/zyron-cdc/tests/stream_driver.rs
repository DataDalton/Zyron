//! Tests the CDC stream driver: it reads change records past the slot's
//! confirmed version, delivers them to the sink in batches, and advances both
//! the slot and the sink checkpoint. Proves the sinks have a real driving
//! caller and that delivery is idempotent across passes.

use std::sync::Mutex;
use std::sync::atomic::{AtomicU64, Ordering};

use bytes::Bytes;
use zyron_cdc::cdc_stream::{
    CdcOutputStream, CdcSink, CdcSinkConfig, SinkCheckpoint, StreamRetryPolicy, TxnDecision,
    drive_stream_once,
};
use zyron_cdc::decoder::{DecodedChange, DecoderPlugin};
use zyron_cdc::{ChangeDataFeed, ChangeRecord, ChangeType, SlotLagConfig, SlotManager};
use zyron_common::Result;

/// Test sink that records each delivered batch and the confirmed LSN.
struct CollectingSink {
    batches: Mutex<Vec<Vec<Vec<u8>>>>,
    lsn: AtomicU64,
}

impl CollectingSink {
    fn new() -> Self {
        Self {
            batches: Mutex::new(Vec::new()),
            lsn: AtomicU64::new(0),
        }
    }
}

impl CdcSink for CollectingSink {
    fn write_batch(&self, changes: &[Bytes]) -> Result<()> {
        self.batches
            .lock()
            .unwrap()
            .push(changes.iter().map(|b| b.to_vec()).collect());
        Ok(())
    }
    fn flush(&self) -> Result<()> {
        Ok(())
    }
    fn set_confirmed_lsn(&self, lsn: u64) {
        self.lsn.store(lsn, Ordering::Relaxed);
    }
    fn checkpoint(&self) -> Result<SinkCheckpoint> {
        Ok(SinkCheckpoint {
            stream_name: "s".into(),
            last_confirmed_lsn: self.lsn.load(Ordering::Relaxed),
            sink_specific_offset: None,
            last_flush_timestamp: 0,
        })
    }
}

fn record(version: u64, payload: &str) -> ChangeRecord {
    ChangeRecord {
        change_type: ChangeType::Insert,
        commit_version: version,
        commit_timestamp: 1_000 + version as i64,
        table_id: 7,
        txn_id: version,
        schema_version: 1,
        row_data: payload.as_bytes().to_vec(),
        primary_key_data: Vec::new(),
        is_last_in_txn: true,
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
        slot_name: "s_slot".into(),
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

#[test]
fn driver_delivers_batches_and_advances_slot() {
    let tmp = tempfile::TempDir::new().unwrap();
    std::fs::create_dir_all(tmp.path().join("cdf")).unwrap();
    std::fs::create_dir_all(tmp.path().join("slots")).unwrap();
    let mut feed = ChangeDataFeed::open(&tmp.path().join("cdf"), 7, 7).unwrap();
    feed.enable();
    feed.append_change(&record(1, "r1")).unwrap();
    feed.append_change(&record(2, "r2")).unwrap();
    feed.append_change(&record(3, "r3")).unwrap();

    let slot_mgr = SlotManager::open(&tmp.path().join("slots"), SlotLagConfig::default()).unwrap();
    slot_mgr
        .create_slot("s_slot", DecoderPlugin::ZyronCdc, Some(vec![7]))
        .unwrap();

    let stream = make_stream();
    let sink = CollectingSink::new();

    let delivered = drive_stream_once(&stream, &feed, &slot_mgr, &sink, decode_record, &|_| {
        TxnDecision::Committed
    })
    .unwrap();
    assert_eq!(delivered, 3, "all three records delivered");

    // batch_size = 2, so the three records arrive as batches of 2 then 1.
    let batches = sink.batches.lock().unwrap();
    assert_eq!(batches.len(), 2, "two batches: {:?}", batches.len());
    assert_eq!(batches[0].len(), 2);
    assert_eq!(batches[1].len(), 1);
    drop(batches);

    // Slot and sink checkpoint advanced to the last version.
    assert_eq!(slot_mgr.get_slot("s_slot").unwrap().confirmed_lsn, 3);
    assert_eq!(sink.checkpoint().unwrap().last_confirmed_lsn, 3);

    // A second pass with no new records delivers nothing.
    let again = drive_stream_once(&stream, &feed, &slot_mgr, &sink, decode_record, &|_| {
        TxnDecision::Committed
    })
    .unwrap();
    assert_eq!(again, 0, "no redelivery of already-confirmed records");
}

#[test]
fn driver_resumes_from_confirmed_version() {
    let tmp = tempfile::TempDir::new().unwrap();
    std::fs::create_dir_all(tmp.path().join("cdf")).unwrap();
    std::fs::create_dir_all(tmp.path().join("slots")).unwrap();
    let mut feed = ChangeDataFeed::open(&tmp.path().join("cdf"), 7, 7).unwrap();
    feed.enable();
    feed.append_change(&record(1, "r1")).unwrap();

    let slot_mgr = SlotManager::open(&tmp.path().join("slots"), SlotLagConfig::default()).unwrap();
    slot_mgr
        .create_slot("s_slot", DecoderPlugin::ZyronCdc, Some(vec![7]))
        .unwrap();

    let stream = make_stream();
    let sink = CollectingSink::new();

    assert_eq!(
        drive_stream_once(&stream, &feed, &slot_mgr, &sink, decode_record, &|_| {
            TxnDecision::Committed
        },)
        .unwrap(),
        1
    );

    // New records appended after the first pass are picked up on the next one.
    feed.append_change(&record(2, "r2")).unwrap();
    feed.append_change(&record(3, "r3")).unwrap();
    assert_eq!(
        drive_stream_once(&stream, &feed, &slot_mgr, &sink, decode_record, &|_| {
            TxnDecision::Committed
        },)
        .unwrap(),
        2,
        "only the new records are delivered"
    );
    assert_eq!(slot_mgr.get_slot("s_slot").unwrap().confirmed_lsn, 3);
}

// Change records land in the feed at execution time, before their
// transaction decides. A rolled back transaction's records must never reach
// the sink, and the slot still advances past them so they never redeliver
#[test]
fn driver_skips_aborted_transactions() {
    let tmp = tempfile::TempDir::new().unwrap();
    std::fs::create_dir_all(tmp.path().join("cdf")).unwrap();
    std::fs::create_dir_all(tmp.path().join("slots")).unwrap();
    let mut feed = ChangeDataFeed::open(&tmp.path().join("cdf"), 7, 7).unwrap();
    feed.enable();
    feed.append_change(&record(1, "r1")).unwrap();
    feed.append_change(&record(2, "rolled-back")).unwrap();
    feed.append_change(&record(3, "r3")).unwrap();

    let slot_mgr = SlotManager::open(&tmp.path().join("slots"), SlotLagConfig::default()).unwrap();
    slot_mgr
        .create_slot("s_slot", DecoderPlugin::ZyronCdc, Some(vec![7]))
        .unwrap();

    let stream = make_stream();
    let sink = CollectingSink::new();

    let delivered = drive_stream_once(&stream, &feed, &slot_mgr, &sink, decode_record, &|txn| {
        if txn == 2 {
            TxnDecision::Aborted
        } else {
            TxnDecision::Committed
        }
    })
    .unwrap();

    assert_eq!(delivered, 2, "the aborted transaction's record is skipped");
    assert_eq!(
        slot_mgr.get_slot("s_slot").unwrap().confirmed_lsn,
        3,
        "the slot advances past the aborted record"
    );
    let batches = sink.batches.lock().unwrap();
    let all: Vec<&[u8]> = batches.iter().flatten().map(|b| b.as_slice()).collect();
    assert_eq!(all.len(), 2);
    for payload in &all {
        let text = String::from_utf8_lossy(payload);
        assert!(
            !text.contains("rolled-back"),
            "a rolled back change must never reach the sink"
        );
    }
}

// An undecided transaction holds delivery: nothing at or past its version
// moves, the slot stops at the last fully decided version, and the next
// pass delivers the rest once the transaction commits
#[test]
fn driver_holds_at_in_flight_transaction() {
    let tmp = tempfile::TempDir::new().unwrap();
    std::fs::create_dir_all(tmp.path().join("cdf")).unwrap();
    std::fs::create_dir_all(tmp.path().join("slots")).unwrap();
    let mut feed = ChangeDataFeed::open(&tmp.path().join("cdf"), 7, 7).unwrap();
    feed.enable();
    feed.append_change(&record(1, "r1")).unwrap();
    feed.append_change(&record(2, "pending")).unwrap();
    feed.append_change(&record(3, "r3")).unwrap();

    let slot_mgr = SlotManager::open(&tmp.path().join("slots"), SlotLagConfig::default()).unwrap();
    slot_mgr
        .create_slot("s_slot", DecoderPlugin::ZyronCdc, Some(vec![7]))
        .unwrap();

    let stream = make_stream();
    let sink = CollectingSink::new();

    let delivered = drive_stream_once(&stream, &feed, &slot_mgr, &sink, decode_record, &|txn| {
        if txn == 2 {
            TxnDecision::InFlight
        } else {
            TxnDecision::Committed
        }
    })
    .unwrap();
    assert_eq!(delivered, 1, "delivery holds at the undecided transaction");
    assert_eq!(
        slot_mgr.get_slot("s_slot").unwrap().confirmed_lsn,
        1,
        "the slot never advances past an undecided change"
    );

    // The transaction commits, the next pass delivers the held records
    let delivered = drive_stream_once(&stream, &feed, &slot_mgr, &sink, decode_record, &|_| {
        TxnDecision::Committed
    })
    .unwrap();
    assert_eq!(delivered, 2);
    assert_eq!(slot_mgr.get_slot("s_slot").unwrap().confirmed_lsn, 3);
}

// All records of one commit version move together: a batch boundary in the
// middle of a multi-row statement must not advance the slot past records
// that have not been handed to the sink, or a crash between flushes loses
// the statement's tail
#[test]
fn driver_advances_slot_only_at_version_boundaries() {
    let tmp = tempfile::TempDir::new().unwrap();
    std::fs::create_dir_all(tmp.path().join("cdf")).unwrap();
    std::fs::create_dir_all(tmp.path().join("slots")).unwrap();
    let mut feed = ChangeDataFeed::open(&tmp.path().join("cdf"), 7, 7).unwrap();
    feed.enable();
    // One statement wrote three rows, all sharing commit version 5, then a
    // later transaction is still undecided
    feed.append_change(&record(5, "r5a")).unwrap();
    feed.append_change(&record(5, "r5b")).unwrap();
    feed.append_change(&record(5, "r5c")).unwrap();
    feed.append_change(&record(6, "pending")).unwrap();

    let slot_mgr = SlotManager::open(&tmp.path().join("slots"), SlotLagConfig::default()).unwrap();
    slot_mgr
        .create_slot("s_slot", DecoderPlugin::ZyronCdc, Some(vec![7]))
        .unwrap();

    let stream = make_stream();
    let sink = CollectingSink::new();

    // batch_size is 2, so the three-row version crosses a batch boundary
    let delivered = drive_stream_once(&stream, &feed, &slot_mgr, &sink, decode_record, &|txn| {
        if txn == 6 {
            TxnDecision::InFlight
        } else {
            TxnDecision::Committed
        }
    })
    .unwrap();

    assert_eq!(delivered, 3, "the whole statement is delivered");
    assert_eq!(
        slot_mgr.get_slot("s_slot").unwrap().confirmed_lsn,
        5,
        "the slot lands on the completed version, not inside or past it"
    );
    let batches = sink.batches.lock().unwrap();
    let total: usize = batches.iter().map(|b| b.len()).sum();
    assert_eq!(total, 3, "every row of the statement reached the sink");
}
