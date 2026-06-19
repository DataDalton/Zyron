//! Inbound source IO for CDC ingestion: a Kafka consumer built on the pure-Rust
//! rskafka client, and an S3 object reader that splits newline-delimited record
//! files. Both run their async network calls on the shared sink IO runtime so
//! they can be driven from the synchronous ingest worker.

use std::ops::Range;

use zyron_common::{Result, ZyronError};

use crate::sink_io::{block_on_io, s3_get, s3_list_objects_v2};

/// Where a Kafka consumer begins when it has no stored checkpoint.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KafkaStart {
    Earliest,
    Latest,
}

impl KafkaStart {
    /// Parses a start-offset option: "latest" begins at the partition tail,
    /// anything else (including "earliest" and the default) begins at the head.
    pub fn from_option(opt: Option<&str>) -> Self {
        match opt.map(|s| s.to_ascii_lowercase()).as_deref() {
            Some("latest") | Some("end") => KafkaStart::Latest,
            _ => KafkaStart::Earliest,
        }
    }
}

/// Resolves the starting offset for a Kafka partition when no checkpoint exists.
pub fn kafka_start_offset(brokers: &str, topic: &str, start: KafkaStart) -> Result<i64> {
    let broker_list = parse_brokers(brokers)?;
    let topic = topic.to_string();
    block_on_io(async move {
        use rskafka::client::ClientBuilder;
        use rskafka::client::partition::{OffsetAt, UnknownTopicHandling};

        let client = ClientBuilder::new(broker_list)
            .build()
            .await
            .map_err(|e| ZyronError::CdcIngestError(format!("Kafka connect failed: {e}")))?;
        let partition = client
            .partition_client(topic.clone(), 0, UnknownTopicHandling::Retry)
            .await
            .map_err(|e| {
                ZyronError::CdcIngestError(format!("Kafka partition client failed: {e}"))
            })?;
        let at = match start {
            KafkaStart::Earliest => OffsetAt::Earliest,
            KafkaStart::Latest => OffsetAt::Latest,
        };
        partition
            .get_offset(at)
            .await
            .map_err(|e| ZyronError::CdcIngestError(format!("Kafka get_offset failed: {e}")))
    })
}

/// Fetches up to `max_bytes` worth of records from partition 0 starting at
/// `offset`. Returns the record value payloads in offset order and the next
/// offset to fetch (the offset after the last record read, or `offset` when the
/// partition had nothing new).
pub fn kafka_consume(
    brokers: &str,
    topic: &str,
    offset: i64,
    max_bytes: i32,
) -> Result<(Vec<Vec<u8>>, i64)> {
    let broker_list = parse_brokers(brokers)?;
    let topic = topic.to_string();
    block_on_io(async move {
        use rskafka::client::ClientBuilder;
        use rskafka::client::partition::UnknownTopicHandling;

        let client = ClientBuilder::new(broker_list)
            .build()
            .await
            .map_err(|e| ZyronError::CdcIngestError(format!("Kafka connect failed: {e}")))?;
        let partition = client
            .partition_client(topic.clone(), 0, UnknownTopicHandling::Retry)
            .await
            .map_err(|e| {
                ZyronError::CdcIngestError(format!("Kafka partition client failed: {e}"))
            })?;

        // bytes range: at least one record, up to max_bytes. max_wait_ms is
        // short so an idle partition returns promptly rather than long-polling.
        let range: Range<i32> = 1..max_bytes.max(1);
        let (records, _high_watermark) = partition
            .fetch_records(offset, range, 500)
            .await
            .map_err(|e| ZyronError::CdcIngestError(format!("Kafka fetch failed: {e}")))?;

        let mut payloads = Vec::with_capacity(records.len());
        let mut next = offset;
        for ro in records {
            next = ro.offset + 1;
            if let Some(value) = ro.record.value {
                payloads.push(value);
            }
        }
        Ok((payloads, next))
    })
}

/// Reads new S3 objects under `prefix` whose keys sort after `start_after`,
/// returning each as (key, records) where records are the object's
/// newline-delimited payloads. Capped at `max_objects` objects per call.
pub fn s3_consume_objects(
    bucket: &str,
    region: &str,
    prefix: &str,
    start_after: &str,
    max_objects: u32,
) -> Result<Vec<(String, Vec<Vec<u8>>)>> {
    let keys = s3_list_objects_v2(bucket, region, prefix, start_after, max_objects)?;
    let mut out = Vec::with_capacity(keys.len());
    for key in keys {
        let body = s3_get(bucket, region, &key)?;
        out.push((key, split_ndjson(&body)));
    }
    Ok(out)
}

/// Splits a record file into one payload per non-empty line. A file holding a
/// single JSON document (no trailing newline) yields one record.
fn split_ndjson(body: &[u8]) -> Vec<Vec<u8>> {
    body.split(|&b| b == b'\n')
        .map(|line| {
            // Trim a trailing CR so CRLF files split cleanly.
            if line.last() == Some(&b'\r') {
                &line[..line.len() - 1]
            } else {
                line
            }
        })
        .filter(|line| !line.iter().all(|b| b.is_ascii_whitespace()))
        .map(|line| line.to_vec())
        .collect()
}

/// Parses and validates a comma-separated broker list.
fn parse_brokers(brokers: &str) -> Result<Vec<String>> {
    let list: Vec<String> = brokers
        .split(',')
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .collect();
    if list.is_empty() {
        return Err(ZyronError::CdcIngestError(
            "Kafka brokers list is empty".into(),
        ));
    }
    Ok(list)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn split_ndjson_drops_blank_lines() {
        let body = b"{\"a\":1}\n\n{\"b\":2}\n";
        let records = split_ndjson(body);
        assert_eq!(records.len(), 2);
        assert_eq!(records[0], b"{\"a\":1}");
        assert_eq!(records[1], b"{\"b\":2}");
    }

    #[test]
    fn split_ndjson_handles_crlf_and_single_doc() {
        assert_eq!(split_ndjson(b"{\"a\":1}\r\n").len(), 1);
        assert_eq!(split_ndjson(b"{\"a\":1}"), vec![b"{\"a\":1}".to_vec()]);
    }

    #[test]
    fn kafka_start_from_option() {
        assert_eq!(KafkaStart::from_option(Some("latest")), KafkaStart::Latest);
        assert_eq!(
            KafkaStart::from_option(Some("earliest")),
            KafkaStart::Earliest
        );
        assert_eq!(KafkaStart::from_option(None), KafkaStart::Earliest);
    }

    #[test]
    fn parse_brokers_rejects_empty() {
        assert!(parse_brokers("").is_err());
        assert_eq!(parse_brokers("a:9092, b:9092").unwrap().len(), 2);
    }
}
