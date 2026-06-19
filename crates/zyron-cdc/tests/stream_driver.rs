//! Tests the CDC stream driver: it reads change records past the slot's
//! confirmed version, delivers them to the sink in batches, and advances both
//! the slot and the sink checkpoint. Proves the sinks have a real driving
//! caller and that delivery is idempotent across passes.

use std::sync::Mutex;
use std::sync::atomic::{AtomicU64, Ordering};

use bytes::Bytes;
use zyron_cdc::cdc_stream::{
    CdcOutputStream, CdcSink, CdcSinkConfig, SinkCheckpoint, StreamRetryPolicy, drive_stream_once,
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
        txn_id: version as u32,
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

    let delivered = drive_stream_once(&stream, &feed, &slot_mgr, &sink, decode_record).unwrap();
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
    let again = drive_stream_once(&stream, &feed, &slot_mgr, &sink, decode_record).unwrap();
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
        drive_stream_once(&stream, &feed, &slot_mgr, &sink, decode_record).unwrap(),
        1
    );

    // New records appended after the first pass are picked up on the next one.
    feed.append_change(&record(2, "r2")).unwrap();
    feed.append_change(&record(3, "r3")).unwrap();
    assert_eq!(
        drive_stream_once(&stream, &feed, &slot_mgr, &sink, decode_record).unwrap(),
        2,
        "only the new records are delivered"
    );
    assert_eq!(slot_mgr.get_slot("s_slot").unwrap().confirmed_lsn, 3);
}
