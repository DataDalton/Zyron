//! Logical decoders that transform WAL records into structured change events.
//!
//! Four decoder formats are supported:
//! - ZyronCdc: native binary format (fastest throughput)
//! - Debezium: Debezium-compatible JSON envelope
//! - Wal2Json: PostgreSQL wal2json compatible format
//! - Avro: Apache Avro schema-registered format

use bytes::Bytes;
use serde::{Deserialize, Serialize};
use zyron_common::{Result, ZyronError};

use crate::change_feed::ChangeType;

// ---------------------------------------------------------------------------
// DecoderPlugin
// ---------------------------------------------------------------------------

/// Decoder plugin identifier for replication slots.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DecoderPlugin {
    ZyronCdc,
    Debezium,
    Wal2Json,
    Avro,
}

impl DecoderPlugin {
    pub fn from_str(s: &str) -> Result<Self> {
        if s.eq_ignore_ascii_case("zyron_cdc") || s.eq_ignore_ascii_case("zyroncdc") {
            Ok(Self::ZyronCdc)
        } else if s.eq_ignore_ascii_case("debezium") {
            Ok(Self::Debezium)
        } else if s.eq_ignore_ascii_case("wal2json") {
            Ok(Self::Wal2Json)
        } else if s.eq_ignore_ascii_case("avro") {
            Ok(Self::Avro)
        } else {
            Err(ZyronError::CdcDecoderError(format!(
                "unknown decoder plugin: {s}"
            )))
        }
    }

    pub fn as_str(&self) -> &'static str {
        match self {
            Self::ZyronCdc => "zyron_cdc",
            Self::Debezium => "debezium",
            Self::Wal2Json => "wal2json",
            Self::Avro => "avro",
        }
    }
}

// ---------------------------------------------------------------------------
// DecodedChange
// ---------------------------------------------------------------------------

/// A decoded change event with full context.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DecodedChange {
    pub table_name: String,
    pub table_id: u32,
    pub operation: ChangeType,
    pub old_values: Option<Vec<(String, String)>>,
    pub new_values: Option<Vec<(String, String)>>,
    pub commit_lsn: u64,
    pub commit_timestamp: i64,
    pub txn_id: u32,
    pub is_last_in_txn: bool,
    pub schema_version: u32,
}

// ---------------------------------------------------------------------------
// LogicalDecoder trait
// ---------------------------------------------------------------------------

/// Trait for logical decoders that transform WAL records into structured changes.
pub trait LogicalDecoder: Send + Sync {
    /// Serializes a DecodedChange into the output format.
    fn serialize(&self, change: &DecodedChange) -> Result<Bytes>;

    /// Deserializes bytes back into a DecodedChange.
    fn deserialize(&self, data: &[u8]) -> Result<DecodedChange>;

    /// Returns the plugin identifier.
    fn plugin(&self) -> DecoderPlugin;
}

// ---------------------------------------------------------------------------
// ZyronCdcDecoder
// ---------------------------------------------------------------------------

/// Native binary decoder using packed binary format for ZyronDB-to-ZyronDB replication.
pub struct ZyronCdcDecoder;

fn write_kv_pairs(buf: &mut Vec<u8>, pairs: &[(String, String)]) {
    buf.extend_from_slice(&(pairs.len() as u16).to_le_bytes());
    for (k, v) in pairs {
        buf.extend_from_slice(&(k.len() as u16).to_le_bytes());
        buf.extend_from_slice(k.as_bytes());
        buf.extend_from_slice(&(v.len() as u16).to_le_bytes());
        buf.extend_from_slice(v.as_bytes());
    }
}

fn read_kv_pairs(data: &[u8], off: &mut usize) -> Result<Vec<(String, String)>> {
    if *off + 2 > data.len() {
        return Err(ZyronError::CdcDecoderError(
            "truncated kv pair count".into(),
        ));
    }
    let count = u16::from_le_bytes(data[*off..*off + 2].try_into().unwrap()) as usize;
    *off += 2;
    let mut pairs = Vec::with_capacity(count);
    for _ in 0..count {
        if *off + 2 > data.len() {
            return Err(ZyronError::CdcDecoderError("truncated key length".into()));
        }
        let klen = u16::from_le_bytes(data[*off..*off + 2].try_into().unwrap()) as usize;
        *off += 2;
        if *off + klen > data.len() {
            return Err(ZyronError::CdcDecoderError("truncated key data".into()));
        }
        let key = String::from_utf8_lossy(&data[*off..*off + klen]).into_owned();
        *off += klen;

        if *off + 2 > data.len() {
            return Err(ZyronError::CdcDecoderError("truncated val length".into()));
        }
        let vlen = u16::from_le_bytes(data[*off..*off + 2].try_into().unwrap()) as usize;
        *off += 2;
        if *off + vlen > data.len() {
            return Err(ZyronError::CdcDecoderError("truncated val data".into()));
        }
        let val = String::from_utf8_lossy(&data[*off..*off + vlen]).into_owned();
        *off += vlen;

        pairs.push((key, val));
    }
    Ok(pairs)
}

impl LogicalDecoder for ZyronCdcDecoder {
    fn serialize(&self, change: &DecodedChange) -> Result<Bytes> {
        let mut buf = Vec::with_capacity(128);

        // table_name
        buf.extend_from_slice(&(change.table_name.len() as u16).to_le_bytes());
        buf.extend_from_slice(change.table_name.as_bytes());
        // table_id
        buf.extend_from_slice(&change.table_id.to_le_bytes());
        // operation
        buf.push(change.operation as u8);
        // commit_lsn
        buf.extend_from_slice(&change.commit_lsn.to_le_bytes());
        // commit_timestamp
        buf.extend_from_slice(&change.commit_timestamp.to_le_bytes());
        // txn_id
        buf.extend_from_slice(&change.txn_id.to_le_bytes());
        // is_last_in_txn
        buf.push(change.is_last_in_txn as u8);
        // schema_version
        buf.extend_from_slice(&change.schema_version.to_le_bytes());

        // old_values
        match &change.old_values {
            Some(pairs) => {
                buf.push(1);
                write_kv_pairs(&mut buf, pairs);
            }
            None => buf.push(0),
        }

        // new_values
        match &change.new_values {
            Some(pairs) => {
                buf.push(1);
                write_kv_pairs(&mut buf, pairs);
            }
            None => buf.push(0),
        }

        Ok(Bytes::from(buf))
    }

    fn deserialize(&self, data: &[u8]) -> Result<DecodedChange> {
        let mut off = 0usize;
        let err = |msg: &str| ZyronError::CdcDecoderError(format!("zyron_cdc deserialize: {msg}"));

        // table_name
        if off + 2 > data.len() {
            return Err(err("truncated table_name length"));
        }
        let tlen = u16::from_le_bytes(data[off..off + 2].try_into().unwrap()) as usize;
        off += 2;
        if off + tlen > data.len() {
            return Err(err("truncated table_name"));
        }
        let table_name = String::from_utf8_lossy(&data[off..off + tlen]).into_owned();
        off += tlen;

        // Fixed fields: table_id(4) + op(1) + lsn(8) + ts(8) + txn(4) + last(1) + ver(4) = 30
        if off + 30 > data.len() {
            return Err(err("truncated fixed fields"));
        }

        let table_id = u32::from_le_bytes(data[off..off + 4].try_into().unwrap());
        off += 4;

        let op_byte = data[off];
        off += 1;
        let operation = ChangeType::from_u8(op_byte)
            .map_err(|_| err(&format!("invalid operation byte: {op_byte}")))?;

        let commit_lsn = u64::from_le_bytes(data[off..off + 8].try_into().unwrap());
        off += 8;

        let commit_timestamp = i64::from_le_bytes(data[off..off + 8].try_into().unwrap());
        off += 8;

        let txn_id = u32::from_le_bytes(data[off..off + 4].try_into().unwrap());
        off += 4;

        let is_last_in_txn = data[off] != 0;
        off += 1;

        let schema_version = u32::from_le_bytes(data[off..off + 4].try_into().unwrap());
        off += 4;

        // old_values
        if off >= data.len() {
            return Err(err("truncated old_values flag"));
        }
        let has_old = data[off];
        off += 1;
        let old_values = if has_old == 1 {
            Some(read_kv_pairs(data, &mut off)?)
        } else {
            None
        };

        // new_values
        if off >= data.len() {
            return Err(err("truncated new_values flag"));
        }
        let has_new = data[off];
        off += 1;
        let new_values = if has_new == 1 {
            Some(read_kv_pairs(data, &mut off)?)
        } else {
            None
        };

        Ok(DecodedChange {
            table_name,
            table_id,
            operation,
            old_values,
            new_values,
            commit_lsn,
            commit_timestamp,
            txn_id,
            is_last_in_txn,
            schema_version,
        })
    }

    fn plugin(&self) -> DecoderPlugin {
        DecoderPlugin::ZyronCdc
    }
}

// ---------------------------------------------------------------------------
// DebeziumDecoder
// ---------------------------------------------------------------------------

/// Debezium-compatible JSON envelope decoder.
/// Output format: {"source":{...}, "before":{...}, "after":{...}, "op":"c/u/d",
/// "ts_ms":..., "transaction":{"id":"...", "total_order":N}}
pub struct DebeziumDecoder {
    pub server_name: String,
    pub database_name: String,
}

impl DebeziumDecoder {
    pub fn new(server_name: String, database_name: String) -> Self {
        Self {
            server_name,
            database_name,
        }
    }

    fn op_code(change_type: &ChangeType) -> &'static str {
        match change_type {
            ChangeType::Insert => "c",
            ChangeType::UpdatePreimage | ChangeType::UpdatePostimage => "u",
            ChangeType::Delete => "d",
            ChangeType::SchemaChange => "s",
            ChangeType::Truncate => "t",
        }
    }

    fn build_envelope(&self, change: &DecodedChange) -> serde_json::Value {
        let before = change.old_values.as_ref().map(|vals| {
            let mut map = serde_json::Map::new();
            for (k, v) in vals {
                map.insert(k.clone(), serde_json::Value::String(v.clone()));
            }
            serde_json::Value::Object(map)
        });

        let after = change.new_values.as_ref().map(|vals| {
            let mut map = serde_json::Map::new();
            for (k, v) in vals {
                map.insert(k.clone(), serde_json::Value::String(v.clone()));
            }
            serde_json::Value::Object(map)
        });

        serde_json::json!({
            "source": {
                "connector": "zyrondb",
                "name": self.server_name,
                "db": self.database_name,
                "table": change.table_name,
                "lsn": change.commit_lsn,
                "schema_version": change.schema_version,
            },
            "before": before,
            "after": after,
            "op": Self::op_code(&change.operation),
            "ts_ms": change.commit_timestamp / 1000, // micros to millis
            "transaction": {
                "id": change.txn_id.to_string(),
                "total_order": change.commit_lsn,
                "data_collection_order": 1,
            }
        })
    }
}

impl LogicalDecoder for DebeziumDecoder {
    fn serialize(&self, change: &DecodedChange) -> Result<Bytes> {
        let envelope = self.build_envelope(change);
        let data = serde_json::to_vec(&envelope)
            .map_err(|e| ZyronError::CdcDecoderError(format!("debezium serialize failed: {e}")))?;
        Ok(Bytes::from(data))
    }

    fn deserialize(&self, data: &[u8]) -> Result<DecodedChange> {
        let envelope: serde_json::Value = serde_json::from_slice(data).map_err(|e| {
            ZyronError::CdcDecoderError(format!("debezium deserialize failed: {e}"))
        })?;

        let op = envelope["op"]
            .as_str()
            .ok_or_else(|| ZyronError::CdcDecoderError("missing op field".into()))?;

        let operation = match op {
            "c" => ChangeType::Insert,
            "u" => ChangeType::UpdatePostimage,
            "d" => ChangeType::Delete,
            "s" => ChangeType::SchemaChange,
            "t" => ChangeType::Truncate,
            _ => {
                return Err(ZyronError::CdcDecoderError(format!(
                    "unknown debezium op: {op}"
                )));
            }
        };

        let table_name = envelope["source"]["table"]
            .as_str()
            .unwrap_or("")
            .to_string();

        let commit_lsn = envelope["source"]["lsn"].as_u64().unwrap_or(0);
        let ts_ms = envelope["ts_ms"].as_i64().unwrap_or(0);
        let txn_id_str = envelope["transaction"]["id"].as_str().unwrap_or("0");
        let txn_id: u32 = txn_id_str.parse().unwrap_or(0);
        let schema_version = envelope["source"]["schema_version"].as_u64().unwrap_or(0) as u32;

        fn extract_kv(val: &serde_json::Value) -> Option<Vec<(String, String)>> {
            val.as_object().map(|obj| {
                obj.iter()
                    .map(|(k, v)| (k.clone(), v.as_str().unwrap_or("").to_string()))
                    .collect()
            })
        }

        Ok(DecodedChange {
            table_name,
            table_id: 0,
            operation,
            old_values: extract_kv(&envelope["before"]),
            new_values: extract_kv(&envelope["after"]),
            commit_lsn,
            commit_timestamp: ts_ms * 1000, // millis to micros
            txn_id,
            is_last_in_txn: false,
            schema_version,
        })
    }

    fn plugin(&self) -> DecoderPlugin {
        DecoderPlugin::Debezium
    }
}

// ---------------------------------------------------------------------------
// Wal2JsonDecoder
// ---------------------------------------------------------------------------

/// PostgreSQL wal2json compatible format decoder.
pub struct Wal2JsonDecoder;

impl LogicalDecoder for Wal2JsonDecoder {
    fn serialize(&self, change: &DecodedChange) -> Result<Bytes> {
        let kind = match change.operation {
            ChangeType::Insert => "insert",
            ChangeType::UpdatePreimage | ChangeType::UpdatePostimage => "update",
            ChangeType::Delete => "delete",
            ChangeType::SchemaChange => "message",
            ChangeType::Truncate => "truncate",
        };

        let mut columnvalues = Vec::new();
        let mut columnnames = Vec::new();
        let mut columntypes = Vec::new();

        if let Some(ref vals) = change.new_values {
            for (name, val) in vals {
                columnnames.push(serde_json::Value::String(name.clone()));
                columntypes.push(serde_json::Value::String("text".into()));
                columnvalues.push(serde_json::Value::String(val.clone()));
            }
        }

        let mut old_keys = serde_json::Map::new();
        if let Some(ref vals) = change.old_values {
            let mut key_names = Vec::new();
            let mut key_values = Vec::new();
            for (name, val) in vals {
                key_names.push(serde_json::Value::String(name.clone()));
                key_values.push(serde_json::Value::String(val.clone()));
            }
            old_keys.insert("keynames".into(), serde_json::Value::Array(key_names));
            old_keys.insert("keyvalues".into(), serde_json::Value::Array(key_values));
        }

        let entry = serde_json::json!({
            "kind": kind,
            "schema": "public",
            "table": change.table_name,
            "columnnames": columnnames,
            "columntypes": columntypes,
            "columnvalues": columnvalues,
            "oldkeys": old_keys,
        });

        let wrapper = serde_json::json!({
            "change": [entry],
            "xid": change.txn_id,
            "lsn": change.commit_lsn,
            "timestamp": change.commit_timestamp,
        });

        let data = serde_json::to_vec(&wrapper)
            .map_err(|e| ZyronError::CdcDecoderError(format!("wal2json serialize failed: {e}")))?;
        Ok(Bytes::from(data))
    }

    fn deserialize(&self, data: &[u8]) -> Result<DecodedChange> {
        let wrapper: serde_json::Value = serde_json::from_slice(data).map_err(|e| {
            ZyronError::CdcDecoderError(format!("wal2json deserialize failed: {e}"))
        })?;

        let changes = wrapper["change"]
            .as_array()
            .ok_or_else(|| ZyronError::CdcDecoderError("missing change array".into()))?;

        if changes.is_empty() {
            return Err(ZyronError::CdcDecoderError("empty change array".into()));
        }

        let entry = &changes[0];
        let kind = entry["kind"]
            .as_str()
            .ok_or_else(|| ZyronError::CdcDecoderError("missing kind field".into()))?;

        let operation = match kind {
            "insert" => ChangeType::Insert,
            "update" => ChangeType::UpdatePostimage,
            "delete" => ChangeType::Delete,
            "message" => ChangeType::SchemaChange,
            "truncate" => ChangeType::Truncate,
            _ => {
                return Err(ZyronError::CdcDecoderError(format!(
                    "unknown wal2json kind: {kind}"
                )));
            }
        };

        let table_name = entry["table"].as_str().unwrap_or("").to_string();

        let new_values = entry["columnnames"].as_array().and_then(|names| {
            entry["columnvalues"].as_array().map(|vals| {
                names
                    .iter()
                    .zip(vals)
                    .map(|(n, v)| {
                        (
                            n.as_str().unwrap_or("").to_string(),
                            v.as_str().unwrap_or("").to_string(),
                        )
                    })
                    .collect::<Vec<_>>()
            })
        });

        let old_values = entry["oldkeys"]["keynames"].as_array().and_then(|names| {
            entry["oldkeys"]["keyvalues"].as_array().map(|vals| {
                names
                    .iter()
                    .zip(vals)
                    .map(|(n, v)| {
                        (
                            n.as_str().unwrap_or("").to_string(),
                            v.as_str().unwrap_or("").to_string(),
                        )
                    })
                    .collect::<Vec<_>>()
            })
        });

        Ok(DecodedChange {
            table_name,
            table_id: 0,
            operation,
            old_values,
            new_values,
            commit_lsn: wrapper["lsn"].as_u64().unwrap_or(0),
            commit_timestamp: wrapper["timestamp"].as_i64().unwrap_or(0),
            txn_id: wrapper["xid"].as_u64().unwrap_or(0) as u32,
            is_last_in_txn: false,
            schema_version: 0,
        })
    }

    fn plugin(&self) -> DecoderPlugin {
        DecoderPlugin::Wal2Json
    }
}

// ---------------------------------------------------------------------------
// AvroDecoder
// ---------------------------------------------------------------------------

/// Apache Avro format decoder. Serializes changes as JSON with an Avro-style
/// schema envelope. Full Avro binary encoding requires a schema registry,
/// which is deferred to the sink layer.
pub struct AvroDecoder;

impl LogicalDecoder for AvroDecoder {
    fn serialize(&self, change: &DecodedChange) -> Result<Bytes> {
        // Avro logical format: JSON with schema metadata.
        // Full binary Avro encoding is done by the sink (Kafka/S3) using
        // a schema registry. Here we produce the logical representation.
        let data = serde_json::to_vec(change)
            .map_err(|e| ZyronError::CdcDecoderError(format!("avro serialize failed: {e}")))?;
        Ok(Bytes::from(data))
    }

    fn deserialize(&self, data: &[u8]) -> Result<DecodedChange> {
        serde_json::from_slice(data)
            .map_err(|e| ZyronError::CdcDecoderError(format!("avro deserialize failed: {e}")))
    }

    fn plugin(&self) -> DecoderPlugin {
        DecoderPlugin::Avro
    }
}

// ---------------------------------------------------------------------------
// Factory
// ---------------------------------------------------------------------------

/// Creates a decoder instance for the given plugin.
pub fn create_decoder(plugin: DecoderPlugin) -> Box<dyn LogicalDecoder> {
    match plugin {
        DecoderPlugin::ZyronCdc => Box::new(ZyronCdcDecoder),
        DecoderPlugin::Debezium => {
            Box::new(DebeziumDecoder::new("zyrondb".into(), "default".into()))
        }
        DecoderPlugin::Wal2Json => Box::new(Wal2JsonDecoder),
        DecoderPlugin::Avro => Box::new(AvroDecoder),
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_change() -> DecodedChange {
        DecodedChange {
            table_name: "users".into(),
            table_id: 42,
            operation: ChangeType::Insert,
            old_values: None,
            new_values: Some(vec![
                ("id".into(), "1".into()),
                ("name".into(), "Alice".into()),
            ]),
            commit_lsn: 12345,
            commit_timestamp: 1700000000_000_000,
            txn_id: 100,
            is_last_in_txn: true,
            schema_version: 1,
        }
    }

    fn sample_update() -> DecodedChange {
        DecodedChange {
            table_name: "users".into(),
            table_id: 42,
            operation: ChangeType::UpdatePostimage,
            old_values: Some(vec![
                ("id".into(), "1".into()),
                ("name".into(), "Alice".into()),
            ]),
            new_values: Some(vec![
                ("id".into(), "1".into()),
                ("name".into(), "Bob".into()),
            ]),
            commit_lsn: 12346,
            commit_timestamp: 1700000001_000_000,
            txn_id: 101,
            is_last_in_txn: true,
            schema_version: 1,
        }
    }

    #[test]
    fn test_decoder_plugin_from_str() {
        assert_eq!(
            DecoderPlugin::from_str("zyron_cdc").unwrap(),
            DecoderPlugin::ZyronCdc
        );
        assert_eq!(
            DecoderPlugin::from_str("debezium").unwrap(),
            DecoderPlugin::Debezium
        );
        assert_eq!(
            DecoderPlugin::from_str("wal2json").unwrap(),
            DecoderPlugin::Wal2Json
        );
        assert_eq!(
            DecoderPlugin::from_str("avro").unwrap(),
            DecoderPlugin::Avro
        );
        assert!(DecoderPlugin::from_str("unknown").is_err());
    }

    #[test]
    fn test_zyron_cdc_roundtrip() {
        let decoder = ZyronCdcDecoder;
        let change = sample_change();
        let bytes = decoder.serialize(&change).unwrap();
        let decoded = decoder.deserialize(&bytes).unwrap();
        assert_eq!(decoded.table_name, "users");
        assert_eq!(decoded.operation, ChangeType::Insert);
        assert_eq!(decoded.commit_lsn, 12345);
        assert_eq!(decoded.txn_id, 100);
        assert!(decoded.new_values.is_some());
        assert!(decoded.old_values.is_none());
    }

    #[test]
    fn test_debezium_roundtrip() {
        let decoder = DebeziumDecoder::new("test_server".into(), "test_db".into());
        let change = sample_update();
        let bytes = decoder.serialize(&change).unwrap();

        // Verify JSON structure.
        let json: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(json["op"], "u");
        assert_eq!(json["source"]["connector"], "zyrondb");
        assert_eq!(json["source"]["name"], "test_server");
        assert_eq!(json["source"]["table"], "users");
        assert!(json["before"].is_object());
        assert!(json["after"].is_object());
        assert_eq!(json["transaction"]["id"], "101");

        // Roundtrip.
        let decoded = decoder.deserialize(&bytes).unwrap();
        assert_eq!(decoded.table_name, "users");
        assert!(decoded.old_values.is_some());
        assert!(decoded.new_values.is_some());
    }

    #[test]
    fn test_debezium_insert_has_null_before() {
        let decoder = DebeziumDecoder::new("srv".into(), "db".into());
        let change = sample_change();
        let bytes = decoder.serialize(&change).unwrap();
        let json: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(json["op"], "c");
        assert!(json["before"].is_null());
        assert!(json["after"].is_object());
    }

    #[test]
    fn test_wal2json_roundtrip() {
        let decoder = Wal2JsonDecoder;
        let change = sample_change();
        let bytes = decoder.serialize(&change).unwrap();

        let json: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        assert!(json["change"].is_array());
        assert_eq!(json["change"][0]["kind"], "insert");
        assert_eq!(json["change"][0]["table"], "users");
        assert_eq!(json["xid"], 100);

        let decoded = decoder.deserialize(&bytes).unwrap();
        assert_eq!(decoded.table_name, "users");
        assert_eq!(decoded.operation, ChangeType::Insert);
    }

    #[test]
    fn test_avro_roundtrip() {
        let decoder = AvroDecoder;
        let change = sample_change();
        let bytes = decoder.serialize(&change).unwrap();
        let decoded = decoder.deserialize(&bytes).unwrap();
        assert_eq!(decoded.table_name, "users");
        assert_eq!(decoded.commit_lsn, 12345);
    }

    #[test]
    fn test_create_decoder_factory() {
        let d = create_decoder(DecoderPlugin::ZyronCdc);
        assert_eq!(d.plugin(), DecoderPlugin::ZyronCdc);

        let d = create_decoder(DecoderPlugin::Debezium);
        assert_eq!(d.plugin(), DecoderPlugin::Debezium);

        let d = create_decoder(DecoderPlugin::Wal2Json);
        assert_eq!(d.plugin(), DecoderPlugin::Wal2Json);

        let d = create_decoder(DecoderPlugin::Avro);
        assert_eq!(d.plugin(), DecoderPlugin::Avro);
    }

    #[test]
    fn test_all_change_types_serialize() {
        let decoder = ZyronCdcDecoder;
        for ct in [
            ChangeType::Insert,
            ChangeType::UpdatePreimage,
            ChangeType::UpdatePostimage,
            ChangeType::Delete,
            ChangeType::SchemaChange,
            ChangeType::Truncate,
        ] {
            let mut change = sample_change();
            change.operation = ct;
            let bytes = decoder.serialize(&change).unwrap();
            let decoded = decoder.deserialize(&bytes).unwrap();
            assert_eq!(decoded.operation, ct);
        }
    }
}
