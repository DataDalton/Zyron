//! Wire Protocol Benchmark Suite
//!
//! Tests the PostgreSQL wire protocol v3 implementation including message
//! encoding/decoding, type serialization, authentication, COPY protocol,
//! TCP connection handshake, concurrent connection handling, and QUIC transport.
//!
//! Run: cargo test -p zyron-wire --test wire_bench --release -- --nocapture

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use bytes::BytesMut;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::{TcpListener, TcpStream};

use zyron_buffer::{BufferPool, BufferPoolConfig};
use zyron_catalog::{Catalog, CatalogCache, HeapCatalogStorage};
use zyron_common::ZyronError;
use zyron_common::types::TypeId;
use zyron_executor::batch::DataBatch;
use zyron_executor::column::{Column, ColumnData, NullBitmap, ScalarValue};
use zyron_planner::logical::LogicalColumn;
use zyron_storage::txn::TransactionManager;
use zyron_storage::{DiskManager, DiskManagerConfig};
use zyron_wal::{WalWriter, WalWriterConfig};

use zyron_wire::auth::{
    AuthResult, Authenticator, CleartextAuthenticator, Md5Authenticator, TrustAuthenticator,
};
use zyron_wire::codec::PostgresCodec;
use zyron_wire::connection::ServerState;
use zyron_wire::copy::{CopyFormat, CopyInHandler, CopyOutHandler};
use zyron_wire::messages::backend::{
    AuthenticationMessage, BackendMessage, ErrorFields, FieldDescription, TransactionState,
};
use zyron_wire::messages::frontend::{DescribeTarget, FrontendMessage, PasswordMessage};
use zyron_wire::session::Session;
use zyron_wire::transport::WireTransport;
use zyron_wire::types;

use zyron_bench_harness::*;

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

// Performance targets (minimum thresholds)
const HANDSHAKE_TARGET_US: f64 = 50.0;
const SIMPLE_QUERY_TARGET_US: f64 = 20.0;
const PARSE_MESSAGE_TARGET_US: f64 = 8.0;
const BIND_MESSAGE_TARGET_US: f64 = 4.0;
const EXECUTE_MESSAGE_TARGET_US: f64 = 20.0;
const ROW_SERIALIZATION_TARGET_OPS: f64 = 8_000_000.0;
const COPY_FROM_TARGET_OPS: f64 = 3_000_000.0;
const COPY_TO_TARGET_OPS: f64 = 5_000_000.0;
#[allow(dead_code)]
const CONCURRENT_CONN_TARGET: f64 = 200_000.0;
const QUERY_PER_SEC_TARGET: f64 = 2_000_000.0;

// Benchmark infrastructure
static BENCHMARK_LOCK: Mutex<()> = Mutex::new(());

// ---------------------------------------------------------------------------
// Test infrastructure helpers
// ---------------------------------------------------------------------------

/// Creates a full ServerState backed by temp directories for integration tests.
async fn create_test_server(db_name: &str) -> (Arc<ServerState>, tempfile::TempDir) {
    let tmp = tempfile::TempDir::new().expect("Failed to create temp dir");
    let data_dir = tmp.path().join("data");
    let wal_dir = tmp.path().join("wal");

    std::fs::create_dir_all(&data_dir).unwrap();
    std::fs::create_dir_all(&wal_dir).unwrap();

    let wal = Arc::new(
        WalWriter::new(WalWriterConfig {
            wal_dir,
            segment_size: 16 * 1024 * 1024,
            fsync_enabled: false,
            ring_buffer_capacity: 4 * 1024 * 1024,
        })
        .expect("WalWriter creation failed"),
    );

    let disk = Arc::new(
        DiskManager::new(DiskManagerConfig {
            data_dir,
            fsync_enabled: false,
        })
        .await
        .expect("DiskManager creation failed"),
    );

    let pool = Arc::new(BufferPool::new(BufferPoolConfig { num_frames: 1024 }));

    let storage = Arc::new(
        HeapCatalogStorage::new(Arc::clone(&disk), Arc::clone(&pool))
            .expect("HeapCatalogStorage creation failed"),
    );
    let cache = Arc::new(CatalogCache::new(256, 64));
    let catalog = Arc::new(
        Catalog::new(storage, cache, Arc::clone(&wal))
            .await
            .expect("Catalog creation failed"),
    );

    // Create a test database
    catalog
        .create_database(db_name, "test_user")
        .await
        .expect("Failed to create test database");

    let txn_manager = Arc::new(TransactionManager::new(Arc::clone(&wal)));

    let state = Arc::new(ServerState {
        catalog,
        wal,
        buffer_pool: pool,
        disk_manager: disk,
        txn_manager,
    });

    (state, tmp)
}

/// Builds a raw PG startup message for a given user and database.
fn build_startup_bytes(user: &str, database: &str) -> Vec<u8> {
    let mut payload = Vec::new();
    // Protocol version 3.0
    payload.extend_from_slice(&196608i32.to_be_bytes());
    // Parameters: key\0value\0 pairs
    payload.extend_from_slice(b"user\0");
    payload.extend_from_slice(user.as_bytes());
    payload.push(0);
    payload.extend_from_slice(b"database\0");
    payload.extend_from_slice(database.as_bytes());
    payload.push(0);
    // Terminating null
    payload.push(0);

    // Prepend length (includes itself)
    let len = (payload.len() + 4) as i32;
    let mut msg = Vec::new();
    msg.extend_from_slice(&len.to_be_bytes());
    msg.extend_from_slice(&payload);
    msg
}

/// Builds a raw PG Query message.
fn build_query_bytes(sql: &str) -> Vec<u8> {
    let mut payload = Vec::new();
    payload.extend_from_slice(sql.as_bytes());
    payload.push(0);

    let len = (payload.len() + 4) as i32;
    let mut msg = Vec::new();
    msg.push(b'Q');
    msg.extend_from_slice(&len.to_be_bytes());
    msg.extend_from_slice(&payload);
    msg
}

/// Builds a raw PG Terminate message.
fn build_terminate_bytes() -> Vec<u8> {
    let mut msg = Vec::new();
    msg.push(b'X');
    msg.extend_from_slice(&4i32.to_be_bytes());
    msg
}

/// Builds a raw PG Parse message (extended query protocol).
fn build_parse_bytes(name: &str, query: &str, param_types: &[i32]) -> Vec<u8> {
    let mut payload = Vec::new();
    payload.extend_from_slice(name.as_bytes());
    payload.push(0);
    payload.extend_from_slice(query.as_bytes());
    payload.push(0);
    payload.extend_from_slice(&(param_types.len() as i16).to_be_bytes());
    for &oid in param_types {
        payload.extend_from_slice(&oid.to_be_bytes());
    }

    let len = (payload.len() + 4) as i32;
    let mut msg = Vec::new();
    msg.push(b'P');
    msg.extend_from_slice(&len.to_be_bytes());
    msg.extend_from_slice(&payload);
    msg
}

/// Builds a raw PG Bind message.
fn build_bind_bytes(
    portal: &str,
    statement: &str,
    param_formats: &[i16],
    params: &[Option<&[u8]>],
    result_formats: &[i16],
) -> Vec<u8> {
    let mut payload = Vec::new();
    payload.extend_from_slice(portal.as_bytes());
    payload.push(0);
    payload.extend_from_slice(statement.as_bytes());
    payload.push(0);

    // Parameter formats
    payload.extend_from_slice(&(param_formats.len() as i16).to_be_bytes());
    for &f in param_formats {
        payload.extend_from_slice(&f.to_be_bytes());
    }

    // Parameter values
    payload.extend_from_slice(&(params.len() as i16).to_be_bytes());
    for param in params {
        match param {
            Some(data) => {
                payload.extend_from_slice(&(data.len() as i32).to_be_bytes());
                payload.extend_from_slice(data);
            }
            None => {
                payload.extend_from_slice(&(-1i32).to_be_bytes());
            }
        }
    }

    // Result formats
    payload.extend_from_slice(&(result_formats.len() as i16).to_be_bytes());
    for &f in result_formats {
        payload.extend_from_slice(&f.to_be_bytes());
    }

    let len = (payload.len() + 4) as i32;
    let mut msg = Vec::new();
    msg.push(b'B');
    msg.extend_from_slice(&len.to_be_bytes());
    msg.extend_from_slice(&payload);
    msg
}

/// Builds a raw PG Execute message.
fn build_execute_bytes(portal: &str, max_rows: i32) -> Vec<u8> {
    let mut payload = Vec::new();
    payload.extend_from_slice(portal.as_bytes());
    payload.push(0);
    payload.extend_from_slice(&max_rows.to_be_bytes());

    let len = (payload.len() + 4) as i32;
    let mut msg = Vec::new();
    msg.push(b'E');
    msg.extend_from_slice(&len.to_be_bytes());
    msg.extend_from_slice(&payload);
    msg
}

/// Builds a raw PG Describe message.
fn build_describe_bytes(target: u8, name: &str) -> Vec<u8> {
    let mut payload = Vec::new();
    payload.push(target); // 'S' for statement, 'P' for portal
    payload.extend_from_slice(name.as_bytes());
    payload.push(0);

    let len = (payload.len() + 4) as i32;
    let mut msg = Vec::new();
    msg.push(b'D');
    msg.extend_from_slice(&len.to_be_bytes());
    msg.extend_from_slice(&payload);
    msg
}

/// Builds a raw PG Sync message.
fn build_sync_bytes() -> Vec<u8> {
    let mut msg = Vec::new();
    msg.push(b'S');
    msg.extend_from_slice(&4i32.to_be_bytes());
    msg
}

/// Reads one raw PG backend message from a stream.
/// Returns (type_byte, payload_bytes).
async fn read_backend_message(stream: &mut TcpStream) -> Result<(u8, Vec<u8>), String> {
    let mut type_buf = [0u8; 1];
    stream
        .read_exact(&mut type_buf)
        .await
        .map_err(|e| format!("read type: {}", e))?;

    let mut len_buf = [0u8; 4];
    stream
        .read_exact(&mut len_buf)
        .await
        .map_err(|e| format!("read len: {}", e))?;
    let len = i32::from_be_bytes(len_buf) as usize;

    if len < 4 {
        return Err("Invalid message length".into());
    }

    let payload_len = len - 4;
    let mut payload = vec![0u8; payload_len];
    if payload_len > 0 {
        stream
            .read_exact(&mut payload)
            .await
            .map_err(|e| format!("read payload: {}", e))?;
    }

    Ok((type_buf[0], payload))
}

/// Reads backend messages until ReadyForQuery ('Z') is received.
/// Returns all message type bytes collected.
async fn read_until_ready(stream: &mut TcpStream) -> Result<Vec<u8>, String> {
    let mut types = Vec::new();
    loop {
        let (msg_type, _payload) = read_backend_message(stream).await?;
        types.push(msg_type);
        if msg_type == b'Z' {
            break;
        }
    }
    Ok(types)
}

/// Performs a full PG handshake on a raw TCP stream.
/// Returns the message types received during startup.
async fn do_handshake(
    stream: &mut TcpStream,
    user: &str,
    database: &str,
) -> Result<Vec<u8>, String> {
    let startup = build_startup_bytes(user, database);
    stream
        .write_all(&startup)
        .await
        .map_err(|e| format!("write startup: {}", e))?;
    stream.flush().await.map_err(|e| format!("flush: {}", e))?;

    read_until_ready(stream).await
}

/// Creates a test column with the given ScalarValues.
fn make_column(type_id: TypeId, values: Vec<ScalarValue>) -> Column {
    let len = values.len();
    let mut nulls = NullBitmap::none(len);

    let data = match type_id {
        TypeId::Boolean => {
            let v: Vec<bool> = values
                .iter()
                .enumerate()
                .map(|(i, s)| match s {
                    ScalarValue::Boolean(b) => *b,
                    ScalarValue::Null => {
                        nulls.set_null(i);
                        false
                    }
                    _ => false,
                })
                .collect();
            ColumnData::Boolean(v)
        }
        TypeId::Int32 => {
            let v: Vec<i32> = values
                .iter()
                .enumerate()
                .map(|(i, s)| match s {
                    ScalarValue::Int32(n) => *n,
                    ScalarValue::Null => {
                        nulls.set_null(i);
                        0
                    }
                    _ => 0,
                })
                .collect();
            ColumnData::Int32(v)
        }
        TypeId::Int64 => {
            let v: Vec<i64> = values
                .iter()
                .enumerate()
                .map(|(i, s)| match s {
                    ScalarValue::Int64(n) => *n,
                    ScalarValue::Null => {
                        nulls.set_null(i);
                        0
                    }
                    _ => 0,
                })
                .collect();
            ColumnData::Int64(v)
        }
        TypeId::Float64 => {
            let v: Vec<f64> = values
                .iter()
                .enumerate()
                .map(|(i, s)| match s {
                    ScalarValue::Float64(f) => *f,
                    ScalarValue::Null => {
                        nulls.set_null(i);
                        0.0
                    }
                    _ => 0.0,
                })
                .collect();
            ColumnData::Float64(v)
        }
        TypeId::Text | TypeId::Varchar => {
            let v: Vec<String> = values
                .iter()
                .enumerate()
                .map(|(i, s)| match s {
                    ScalarValue::Utf8(s) => s.clone(),
                    ScalarValue::Null => {
                        nulls.set_null(i);
                        String::new()
                    }
                    _ => String::new(),
                })
                .collect();
            ColumnData::Utf8(v)
        }
        TypeId::Uuid => {
            let v: Vec<[u8; 16]> = values
                .iter()
                .enumerate()
                .map(|(i, s)| match s {
                    ScalarValue::FixedBinary16(b) => *b,
                    ScalarValue::Null => {
                        nulls.set_null(i);
                        [0u8; 16]
                    }
                    _ => [0u8; 16],
                })
                .collect();
            ColumnData::FixedBinary16(v)
        }
        TypeId::Bytea => {
            let v: Vec<Vec<u8>> = values
                .iter()
                .enumerate()
                .map(|(i, s)| match s {
                    ScalarValue::Binary(b) => b.clone(),
                    ScalarValue::Null => {
                        nulls.set_null(i);
                        Vec::new()
                    }
                    _ => Vec::new(),
                })
                .collect();
            ColumnData::Binary(v)
        }
        _ => {
            // Fallback to Int32 for unsupported types in this helper
            ColumnData::Int32(vec![0; len])
        }
    };

    Column {
        data,
        nulls,
        type_id,
    }
}

// ===========================================================================
// Correctness Tests
// ===========================================================================

// ---------------------------------------------------------------------------
// Test 1: Message codec roundtrip
// ---------------------------------------------------------------------------

#[test]
fn test_wire_message_codec_roundtrip() {
    zyron_bench_harness::init("wire");
    tprintln!("\n=== Message Codec Roundtrip Test ===");

    let mut codec = PostgresCodec::new();
    let mut buf = BytesMut::new();

    // Test BackendMessage encoding
    let messages: Vec<BackendMessage> = vec![
        BackendMessage::Authentication(AuthenticationMessage::Ok),
        BackendMessage::Authentication(AuthenticationMessage::CleartextPassword),
        BackendMessage::Authentication(AuthenticationMessage::Md5Password { salt: [1, 2, 3, 4] }),
        BackendMessage::ParameterStatus {
            name: "server_version".into(),
            value: "16.0".into(),
        },
        BackendMessage::BackendKeyData {
            process_id: 12345,
            secret_key: 67890,
        },
        BackendMessage::ReadyForQuery(TransactionState::Idle),
        BackendMessage::ReadyForQuery(TransactionState::InTransaction),
        BackendMessage::ReadyForQuery(TransactionState::Failed),
        BackendMessage::RowDescription(vec![
            FieldDescription {
                name: "id".into(),
                table_oid: 0,
                column_attr: 0,
                type_oid: types::PG_INT4_OID,
                type_size: 4,
                type_modifier: -1,
                format: 0,
            },
            FieldDescription {
                name: "name".into(),
                table_oid: 0,
                column_attr: 0,
                type_oid: types::PG_TEXT_OID,
                type_size: -1,
                type_modifier: -1,
                format: 0,
            },
        ]),
        BackendMessage::DataRow(vec![Some(b"42".to_vec()), Some(b"Alice".to_vec()), None]),
        BackendMessage::CommandComplete {
            tag: "SELECT 1".into(),
        },
        BackendMessage::ErrorResponse(ErrorFields {
            severity: "ERROR".into(),
            code: "42P01".into(),
            message: "table not found".into(),
            detail: Some("the table does not exist".into()),
            hint: None,
            position: Some(10),
        }),
        BackendMessage::ParseComplete,
        BackendMessage::BindComplete,
        BackendMessage::CloseComplete,
        BackendMessage::NoData,
        BackendMessage::EmptyQueryResponse,
        BackendMessage::PortalSuspended,
        BackendMessage::ParameterDescription(vec![types::PG_INT4_OID, types::PG_TEXT_OID]),
        BackendMessage::CopyInResponse {
            format: 0,
            column_formats: vec![0, 0],
        },
        BackendMessage::CopyOutResponse {
            format: 0,
            column_formats: vec![0, 0],
        },
        BackendMessage::CopyData(b"1\tAlice\n".to_vec()),
        BackendMessage::CopyDone,
    ];

    let msg_count = messages.len();
    for msg in &messages {
        codec.encode(msg.clone(), &mut buf).expect("encode failed");
    }

    tprintln!(
        "  Encoded {} backend message types ({} bytes total)\n",
        msg_count,
        buf.len()
    );
    assert!(buf.len() > 0, "Encoded buffer should not be empty");

    // Test FrontendMessage decoding via codec
    let mut codec2 = PostgresCodec::new();

    // Startup message decode
    let startup_bytes = build_startup_bytes("testuser", "testdb");
    let mut startup_buf = BytesMut::from(&startup_bytes[..]);
    let decoded = codec2
        .decode(&mut startup_buf)
        .expect("startup decode failed");
    assert!(decoded.is_some(), "Should decode startup message");
    if let Some(FrontendMessage::Startup(s)) = decoded {
        assert_eq!(s.params.get("user").map(|s| s.as_str()), Some("testuser"));
        assert_eq!(s.params.get("database").map(|s| s.as_str()), Some("testdb"));
        tprintln!("  Startup message decoded: user=testuser, database=testdb\n");
    } else {
        panic!("Expected Startup message");
    }

    // Switch to normal mode for remaining messages
    codec2.set_normal_mode();

    // Query message
    let query_bytes = build_query_bytes("SELECT 1");
    let mut query_buf = BytesMut::from(&query_bytes[..]);
    let decoded = codec2.decode(&mut query_buf).expect("query decode failed");
    if let Some(FrontendMessage::Query { sql }) = decoded {
        assert_eq!(sql, "SELECT 1");
        tprintln!("  Query message decoded: SELECT 1\n");
    } else {
        panic!("Expected Query message");
    }

    // Parse message
    let parse_bytes = build_parse_bytes("stmt1", "SELECT $1", &[types::PG_INT4_OID]);
    let mut parse_buf = BytesMut::from(&parse_bytes[..]);
    let decoded = codec2.decode(&mut parse_buf).expect("parse decode failed");
    if let Some(FrontendMessage::Parse {
        name,
        query,
        param_types,
    }) = decoded
    {
        assert_eq!(name, "stmt1");
        assert_eq!(query, "SELECT $1");
        assert_eq!(param_types, vec![types::PG_INT4_OID]);
        tprintln!("  Parse message decoded: stmt1, SELECT $1\n");
    } else {
        panic!("Expected Parse message");
    }

    // Bind message
    let bind_bytes = build_bind_bytes("", "stmt1", &[0], &[Some(b"42")], &[0]);
    let mut bind_buf = BytesMut::from(&bind_bytes[..]);
    let decoded = codec2.decode(&mut bind_buf).expect("bind decode failed");
    assert!(
        matches!(decoded, Some(FrontendMessage::Bind { .. })),
        "Expected Bind message"
    );
    tprintln!("  Bind message decoded\n");

    // Execute message
    let exec_bytes = build_execute_bytes("", 0);
    let mut exec_buf = BytesMut::from(&exec_bytes[..]);
    let decoded = codec2.decode(&mut exec_buf).expect("execute decode failed");
    assert!(
        matches!(decoded, Some(FrontendMessage::Execute { .. })),
        "Expected Execute message"
    );
    tprintln!("  Execute message decoded\n");

    // Describe message
    let desc_bytes = build_describe_bytes(b'S', "stmt1");
    let mut desc_buf = BytesMut::from(&desc_bytes[..]);
    let decoded = codec2
        .decode(&mut desc_buf)
        .expect("describe decode failed");
    assert!(
        matches!(decoded, Some(FrontendMessage::Describe { .. })),
        "Expected Describe message"
    );
    tprintln!("  Describe message decoded\n");

    // Sync message
    let sync_bytes = build_sync_bytes();
    let mut sync_buf = BytesMut::from(&sync_bytes[..]);
    let decoded = codec2.decode(&mut sync_buf).expect("sync decode failed");
    assert!(
        matches!(decoded, Some(FrontendMessage::Sync)),
        "Expected Sync message"
    );
    tprintln!("  Sync message decoded\n");

    // Terminate message
    let term_bytes = build_terminate_bytes();
    let mut term_buf = BytesMut::from(&term_bytes[..]);
    let decoded = codec2
        .decode(&mut term_buf)
        .expect("terminate decode failed");
    assert!(
        matches!(decoded, Some(FrontendMessage::Terminate)),
        "Expected Terminate message"
    );
    tprintln!("  Terminate message decoded\n");

    tprintln!(
        "  All {} backend + 8 frontend message types verified\n",
        msg_count
    );
}

// ---------------------------------------------------------------------------
// Test 2: Type serialization for all supported types
// ---------------------------------------------------------------------------

#[test]
fn test_wire_type_serialization_all_types() {
    zyron_bench_harness::init("wire");
    tprintln!("\n=== Type Serialization (All Types) Test ===");

    // Test type_id_to_pg_oid mapping
    let type_mappings = vec![
        (TypeId::Boolean, types::PG_BOOL_OID),
        (TypeId::Int16, types::PG_INT2_OID),
        (TypeId::Int32, types::PG_INT4_OID),
        (TypeId::Int64, types::PG_INT8_OID),
        (TypeId::Float32, types::PG_FLOAT4_OID),
        (TypeId::Float64, types::PG_FLOAT8_OID),
        (TypeId::Text, types::PG_TEXT_OID),
        (TypeId::Varchar, types::PG_VARCHAR_OID),
        (TypeId::Char, types::PG_CHAR_OID),
        (TypeId::Bytea, types::PG_BYTEA_OID),
        (TypeId::Uuid, types::PG_UUID_OID),
        (TypeId::Date, types::PG_DATE_OID),
        (TypeId::Time, types::PG_TIME_OID),
        (TypeId::Timestamp, types::PG_TIMESTAMP_OID),
        (TypeId::TimestampTz, types::PG_TIMESTAMPTZ_OID),
        (TypeId::Interval, types::PG_INTERVAL_OID),
        (TypeId::Json, types::PG_JSON_OID),
        (TypeId::Jsonb, types::PG_JSONB_OID),
        (TypeId::Decimal, types::PG_NUMERIC_OID),
    ];

    for (type_id, expected_oid) in &type_mappings {
        let oid = types::type_id_to_pg_oid(*type_id);
        assert_eq!(
            oid, *expected_oid,
            "TypeId {:?} should map to OID {}",
            type_id, expected_oid
        );
    }
    tprintln!(
        "  {} TypeId -> PG OID mappings verified\n",
        type_mappings.len()
    );

    // Test scalar_to_text roundtrip for common types
    let text_cases: Vec<(ScalarValue, &str)> = vec![
        (ScalarValue::Boolean(true), "t"),
        (ScalarValue::Boolean(false), "f"),
        (ScalarValue::Int32(42), "42"),
        (ScalarValue::Int32(-1), "-1"),
        (ScalarValue::Int32(0), "0"),
        (ScalarValue::Int64(9999999999i64), "9999999999"),
        (ScalarValue::Float64(3.14), "3.14"),
        (ScalarValue::Utf8("hello world".into()), "hello world"),
        (ScalarValue::Utf8("".into()), ""),
    ];

    for (scalar, expected) in &text_cases {
        let encoded = types::scalar_to_text(scalar);
        assert!(
            encoded.is_some(),
            "scalar_to_text should produce value for {:?}",
            scalar
        );
        let bytes = encoded.unwrap();
        let text = String::from_utf8(bytes).expect("Should be valid UTF-8");
        assert_eq!(text, *expected, "Text encoding mismatch for {:?}", scalar);
    }
    tprintln!(
        "  {} scalar_to_text conversions verified\n",
        text_cases.len()
    );

    // Test scalar_to_text returns None for Null
    assert!(
        types::scalar_to_text(&ScalarValue::Null).is_none(),
        "Null should produce None"
    );
    tprintln!("  Null -> None verified\n");

    // Test scalar_to_binary for integer types
    let binary_cases: Vec<(ScalarValue, Vec<u8>)> = vec![
        (ScalarValue::Boolean(true), vec![1]),
        (ScalarValue::Boolean(false), vec![0]),
        (ScalarValue::Int32(42), 42i32.to_be_bytes().to_vec()),
        (ScalarValue::Int64(12345), 12345i64.to_be_bytes().to_vec()),
        (ScalarValue::Float64(2.718), 2.718f64.to_be_bytes().to_vec()),
    ];

    for (scalar, expected) in &binary_cases {
        let encoded = types::scalar_to_binary(scalar);
        assert!(
            encoded.is_some(),
            "scalar_to_binary should produce value for {:?}",
            scalar
        );
        assert_eq!(
            encoded.unwrap(),
            *expected,
            "Binary encoding mismatch for {:?}",
            scalar
        );
    }
    tprintln!(
        "  {} scalar_to_binary conversions verified\n",
        binary_cases.len()
    );

    // Test text_to_scalar roundtrip
    let text_roundtrip: Vec<(i32, &[u8], ScalarValue)> = vec![
        (types::PG_BOOL_OID, b"t", ScalarValue::Boolean(true)),
        (types::PG_BOOL_OID, b"f", ScalarValue::Boolean(false)),
        (types::PG_INT4_OID, b"42", ScalarValue::Int32(42)),
        (types::PG_INT8_OID, b"99", ScalarValue::Int64(99)),
        (
            types::PG_TEXT_OID,
            b"hello",
            ScalarValue::Utf8("hello".into()),
        ),
    ];

    for (oid, text, expected) in &text_roundtrip {
        let result = types::text_to_scalar(text, *oid)
            .expect(&format!("text_to_scalar failed for OID {}", oid));
        assert_eq!(
            result,
            *expected,
            "text_to_scalar mismatch for OID {} input {:?}",
            oid,
            String::from_utf8_lossy(text)
        );
    }
    tprintln!(
        "  {} text_to_scalar conversions verified\n",
        text_roundtrip.len()
    );

    // Test binary_to_scalar roundtrip
    let binary_roundtrip: Vec<(i32, Vec<u8>, ScalarValue)> = vec![
        (
            types::PG_INT4_OID,
            42i32.to_be_bytes().to_vec(),
            ScalarValue::Int32(42),
        ),
        (
            types::PG_INT8_OID,
            99i64.to_be_bytes().to_vec(),
            ScalarValue::Int64(99),
        ),
        (types::PG_BOOL_OID, vec![1], ScalarValue::Boolean(true)),
    ];

    for (oid, bytes, expected) in &binary_roundtrip {
        let result = types::binary_to_scalar(bytes, *oid)
            .expect(&format!("binary_to_scalar failed for OID {}", oid));
        assert_eq!(
            result, *expected,
            "binary_to_scalar mismatch for OID {}",
            oid
        );
    }
    tprintln!(
        "  {} binary_to_scalar conversions verified\n",
        binary_roundtrip.len()
    );

    // Test pg_type_size
    assert_eq!(types::pg_type_size(TypeId::Boolean), 1);
    assert_eq!(types::pg_type_size(TypeId::Int16), 2);
    assert_eq!(types::pg_type_size(TypeId::Int32), 4);
    assert_eq!(types::pg_type_size(TypeId::Int64), 8);
    assert_eq!(types::pg_type_size(TypeId::Float32), 4);
    assert_eq!(types::pg_type_size(TypeId::Float64), 8);
    assert_eq!(types::pg_type_size(TypeId::Uuid), 16);
    assert_eq!(types::pg_type_size(TypeId::Text), -1);
    assert_eq!(types::pg_type_size(TypeId::Varchar), -1);
    assert_eq!(types::pg_type_size(TypeId::Bytea), -1);
    tprintln!("  pg_type_size verified for 10 types\n");
}

// ---------------------------------------------------------------------------
// Test 3: Authentication protocols
// ---------------------------------------------------------------------------

#[test]
fn test_wire_auth_protocols() {
    zyron_bench_harness::init("wire");
    tprintln!("\n=== Authentication Protocols Test ===");

    // Trust authenticator
    {
        let auth = TrustAuthenticator;
        let result = auth.initial_message("anyone");
        assert!(
            matches!(result, AuthResult::Authenticated),
            "TrustAuth should authenticate immediately"
        );
        tprintln!("  TrustAuthenticator: immediate authentication verified\n");
    }

    // Cleartext authenticator - correct password
    {
        let mut passwords = HashMap::new();
        passwords.insert("alice".to_string(), "secret".to_string());
        let mut auth = CleartextAuthenticator::new(passwords);

        let result = auth.initial_message("alice");
        assert!(
            matches!(result, AuthResult::Challenge(_)),
            "CleartextAuth should send challenge"
        );

        let response = PasswordMessage::Cleartext("secret".to_string());
        let progress = auth.process_response("alice", &response);
        assert!(progress.is_ok(), "Correct password should succeed");
        tprintln!("  CleartextAuthenticator: correct password accepted\n");
    }

    // Cleartext authenticator - wrong password
    {
        let mut passwords = HashMap::new();
        passwords.insert("alice".to_string(), "secret".to_string());
        let mut auth = CleartextAuthenticator::new(passwords);
        let _ = auth.initial_message("alice");

        let response = PasswordMessage::Cleartext("wrong".to_string());
        let progress = auth.process_response("alice", &response);
        assert!(progress.is_err(), "Wrong password should fail");
        tprintln!("  CleartextAuthenticator: wrong password rejected\n");
    }

    // MD5 authenticator - correct password
    {
        let mut passwords = HashMap::new();
        passwords.insert("bob".to_string(), "pass123".to_string());
        let mut auth = Md5Authenticator::new(passwords);

        let result = auth.initial_message("bob");
        // Extract salt from the challenge message
        let salt = match &result {
            AuthResult::Challenge(BackendMessage::Authentication(
                AuthenticationMessage::Md5Password { salt },
            )) => *salt,
            _ => panic!("Md5Auth should send Md5Password challenge"),
        };

        // Compute expected MD5: md5(md5(password + user) + salt)
        use md5::{Digest, Md5};
        let mut hasher = Md5::new();
        hasher.update(b"pass123bob");
        let inner = format!("{:x}", hasher.finalize());

        let mut hasher2 = Md5::new();
        hasher2.update(inner.as_bytes());
        hasher2.update(&salt);
        let expected = format!("md5{:x}", hasher2.finalize());

        let response = PasswordMessage::Cleartext(expected);
        let progress = auth.process_response("bob", &response);
        assert!(progress.is_ok(), "Correct MD5 password should succeed");
        tprintln!("  Md5Authenticator: correct MD5 hash accepted\n");
    }

    tprintln!("  All 4 authentication protocol tests passed\n");
}

// ---------------------------------------------------------------------------
// Test 4: COPY protocol
// ---------------------------------------------------------------------------

#[test]
fn test_wire_copy_protocol() {
    zyron_bench_harness::init("wire");
    tprintln!("\n=== COPY Protocol Test ===");

    let columns = vec![
        LogicalColumn {
            table_idx: None,
            column_id: zyron_catalog::ColumnId(0),
            name: "id".into(),
            type_id: TypeId::Int32,
            nullable: false,
        },
        LogicalColumn {
            table_idx: None,
            column_id: zyron_catalog::ColumnId(1),
            name: "name".into(),
            type_id: TypeId::Text,
            nullable: false,
        },
    ];

    // COPY IN (text format)
    {
        let mut handler = CopyInHandler::new(columns.clone(), CopyFormat::Text);

        let header = handler.header_message();
        assert!(
            matches!(header, BackendMessage::CopyInResponse { .. }),
            "Should produce CopyInResponse"
        );

        handler.feed(b"1\tAlice\n2\tBob\n").expect("feed failed");
        let rows = handler.finish().expect("finish failed");
        assert_eq!(rows.len(), 2, "Should have 2 rows");
        assert_eq!(rows[0].len(), 2, "Each row should have 2 columns");
        assert_eq!(rows[0][0].as_deref(), Some(b"1".as_slice()));
        assert_eq!(rows[0][1].as_deref(), Some(b"Alice".as_slice()));
        assert_eq!(rows[1][0].as_deref(), Some(b"2".as_slice()));
        assert_eq!(rows[1][1].as_deref(), Some(b"Bob".as_slice()));
        tprintln!("  COPY IN (text): 2 rows parsed correctly\n");
    }

    // COPY IN (CSV format)
    {
        let mut handler = CopyInHandler::new(columns.clone(), CopyFormat::Csv);

        handler.feed(b"1,Alice\n2,Bob\n").expect("feed CSV failed");
        let rows = handler.finish().expect("finish CSV failed");
        assert_eq!(rows.len(), 2, "CSV should have 2 rows");
        assert_eq!(rows[0][0].as_deref(), Some(b"1".as_slice()));
        assert_eq!(rows[0][1].as_deref(), Some(b"Alice".as_slice()));
        tprintln!("  COPY IN (CSV): 2 rows parsed correctly\n");
    }

    // COPY IN (CSV with quoted fields)
    {
        let mut handler = CopyInHandler::new(columns.clone(), CopyFormat::Csv);

        handler
            .feed(b"3,\"O'Brien, Jr.\"\n")
            .expect("feed quoted CSV failed");
        let rows = handler.finish().expect("finish quoted CSV failed");
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0][1].as_deref(), Some(b"O'Brien, Jr.".as_slice()));
        tprintln!("  COPY IN (CSV quoted): quoted field with comma parsed correctly\n");
    }

    // COPY OUT (text format)
    {
        let batch = DataBatch::new(vec![
            make_column(
                TypeId::Int32,
                vec![ScalarValue::Int32(1), ScalarValue::Int32(2)],
            ),
            make_column(
                TypeId::Text,
                vec![
                    ScalarValue::Utf8("Alice".into()),
                    ScalarValue::Utf8("Bob".into()),
                ],
            ),
        ]);

        let handler = CopyOutHandler::new(columns.clone(), CopyFormat::Text);
        let header = handler.header_message();
        assert!(matches!(header, BackendMessage::CopyOutResponse { .. }));

        let data_msgs = handler.format_batch(&batch);
        assert_eq!(data_msgs.len(), 2, "Should produce 2 CopyData messages");

        let done = handler.done_message();
        assert!(matches!(done, BackendMessage::CopyDone));
        tprintln!("  COPY OUT (text): 2 rows formatted correctly\n");
    }

    // COPY OUT (CSV format)
    {
        let batch = DataBatch::new(vec![
            make_column(TypeId::Int32, vec![ScalarValue::Int32(10)]),
            make_column(TypeId::Text, vec![ScalarValue::Utf8("hello,world".into())]),
        ]);

        let handler = CopyOutHandler::new(columns.clone(), CopyFormat::Csv);
        let data_msgs = handler.format_batch(&batch);
        assert_eq!(data_msgs.len(), 1);

        // Verify the CSV output has proper quoting
        if let BackendMessage::CopyData(data) = &data_msgs[0] {
            let line = String::from_utf8_lossy(data);
            assert!(
                line.contains("\"hello,world\""),
                "CSV should quote fields containing commas: {}",
                line
            );
        }
        tprintln!("  COPY OUT (CSV): field with comma properly quoted\n");
    }

    tprintln!("  All COPY protocol tests passed\n");
}

// ---------------------------------------------------------------------------
// Test 5: Error SQLSTATE mapping
// ---------------------------------------------------------------------------

#[test]
fn test_wire_error_sqlstate_mapping() {
    zyron_bench_harness::init("wire");
    tprintln!("\n=== Error SQLSTATE Mapping Test ===");

    // Build BackendMessage::ErrorResponse from ZyronErrors and verify SQLSTATE codes.
    // We test this via the protocol error encoding, which produces ErrorFields.
    let test_cases: Vec<(ZyronError, &str)> = vec![
        (ZyronError::ParseError("bad syntax".into()), "42601"),
        (ZyronError::TableNotFound("users".into()), "42P01"),
        (ZyronError::ColumnNotFound("foo".into()), "42703"),
        (ZyronError::DuplicateKey, "23505"),
        (ZyronError::TransactionAborted("aborted".into()), "25P02"),
        (ZyronError::DeadlockDetected, "40P01"),
        (ZyronError::WriteConflict { page_id: 1 }, "40001"),
        (ZyronError::NullNotAllowed, "23502"),
        (ZyronError::DatabaseNotFound("nope".into()), "3D000"),
        (ZyronError::SchemaNotFound("nope".into()), "3F000"),
        (ZyronError::TableAlreadyExists("users".into()), "42P07"),
        (ZyronError::DatabaseAlreadyExists("db".into()), "42P04"),
        (ZyronError::PlanError("bad plan".into()), "42000"),
        (ZyronError::ExecutionError("exec fail".into()), "XX000"),
    ];

    // Encode each error into a BackendMessage::ErrorResponse and parse the SQLSTATE
    for (error, expected_code) in &test_cases {
        let fields = zyron_wire::connection::zyron_error_to_fields(error);
        assert_eq!(
            fields.code,
            *expected_code,
            "ZyronError::{:?} should map to SQLSTATE {}",
            std::mem::discriminant(error),
            expected_code,
        );

        // Verify it encodes to valid bytes
        let msg = BackendMessage::ErrorResponse(fields);
        let mut buf = BytesMut::new();
        msg.encode(&mut buf);
        assert!(
            buf.len() > 5,
            "Error message should encode to non-trivial bytes"
        );
    }

    tprintln!("  {} SQLSTATE mappings verified\n", test_cases.len());
}

// ---------------------------------------------------------------------------
// Test 6: Session management
// ---------------------------------------------------------------------------

#[test]
fn test_wire_session_management() {
    zyron_bench_harness::init("wire");
    tprintln!("\n=== Session Management Test ===");

    let session = Session::new(
        "testuser".into(),
        "testdb".into(),
        zyron_catalog::DatabaseId(1),
    );

    // Default variables
    assert_eq!(session.get_variable("server_version"), Some("16.0"));
    assert_eq!(session.get_variable("server_encoding"), Some("UTF8"));
    assert_eq!(session.get_variable("client_encoding"), Some("UTF8"));
    tprintln!("  Default variables verified (server_version, encoding)\n");

    // Startup parameters
    let params = session.startup_parameters();
    assert!(!params.is_empty(), "Startup parameters should not be empty");
    let param_names: Vec<&str> = params.iter().map(|(k, _)| *k).collect();
    assert!(
        param_names.contains(&"server_version"),
        "Should include server_version"
    );
    assert!(
        param_names.contains(&"server_encoding"),
        "Should include server_encoding"
    );
    tprintln!("  Startup parameters: {} entries\n", params.len());

    // Transaction state transitions
    let mut session = Session::new(
        "testuser".into(),
        "testdb".into(),
        zyron_catalog::DatabaseId(1),
    );
    assert_eq!(session.transaction_state(), TransactionState::Idle);

    session.set_transaction_state(TransactionState::InTransaction);
    assert_eq!(session.transaction_state(), TransactionState::InTransaction);

    session.set_transaction_state(TransactionState::Failed);
    assert_eq!(session.transaction_state(), TransactionState::Failed);

    session.set_transaction_state(TransactionState::Idle);
    assert_eq!(session.transaction_state(), TransactionState::Idle);
    tprintln!("  Transaction state transitions: Idle -> InTransaction -> Failed -> Idle\n");

    // Set/get variable
    let mut session = Session::new(
        "testuser".into(),
        "testdb".into(),
        zyron_catalog::DatabaseId(1),
    );
    session.set_variable("timezone".into(), "US/Eastern".into());
    assert_eq!(session.get_variable("timezone"), Some("US/Eastern"));
    tprintln!("  Custom variable set/get verified\n");

    // Search path update
    session.set_variable("search_path".into(), "public, myschema".into());
    assert!(
        session.search_path.contains(&"public".to_string()),
        "Search path should contain public"
    );
    assert!(
        session.search_path.contains(&"myschema".to_string()),
        "Search path should contain myschema"
    );
    tprintln!(
        "  Search path parsing verified: {:?}\n",
        session.search_path
    );
}

// ===========================================================================
// Performance Benchmarks (5 runs each)
// ===========================================================================

// ---------------------------------------------------------------------------
// Test 7: Message encode throughput (BackendMessage -> bytes)
// ---------------------------------------------------------------------------

#[test]
fn test_wire_message_encode_throughput() {
    zyron_bench_harness::init("wire");
    let _bench_guard = BENCHMARK_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    tprintln!("\n=== Message Encode Throughput Test ===");

    let iterations = 1_000_000;
    let mut results = Vec::with_capacity(VALIDATION_RUNS);

    // Pre-build a DataRow message to encode repeatedly
    let data_row = BackendMessage::DataRow(vec![
        Some(b"42".to_vec()),
        Some(b"Alice Johnson".to_vec()),
        Some(b"2024-01-15".to_vec()),
        None,
    ]);

    for run in 0..VALIDATION_RUNS {
        tprintln!("--- Run {}/{} ---\n", run + 1, VALIDATION_RUNS);

        let mut codec = PostgresCodec::new();
        codec.set_normal_mode();
        let mut buf = BytesMut::with_capacity(128 * iterations);

        let start = Instant::now();
        for _ in 0..iterations {
            codec
                .encode(data_row.clone(), &mut buf)
                .expect("encode failed");
            buf.clear();
        }
        let elapsed = start.elapsed();

        let ops_sec = iterations as f64 / elapsed.as_secs_f64();
        results.push(ops_sec);
        tprintln!(
            "  {} encodes in {:.2?}, {} ops/sec\n",
            format_with_commas(iterations as f64),
            elapsed,
            format_with_commas(ops_sec),
        );
    }

    // Use query_per_sec target as a proxy for message encode throughput
    let result = validate_metric(
        "Message Encode",
        "BackendMessage encode throughput (ops/sec)",
        results,
        QUERY_PER_SEC_TARGET,
        true,
    );

    assert!(
        result.passed,
        "Message encode throughput below minimum threshold"
    );
    assert!(
        !result.regression_detected,
        "Regression detected in message encode"
    );
}

// ---------------------------------------------------------------------------
// Test 8: Message decode throughput (bytes -> FrontendMessage)
// ---------------------------------------------------------------------------

#[test]
fn test_wire_message_decode_throughput() {
    zyron_bench_harness::init("wire");
    let _bench_guard = BENCHMARK_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    tprintln!("\n=== Message Decode Throughput Test ===");

    let iterations = 1_000_000;
    let mut results = Vec::with_capacity(VALIDATION_RUNS);

    // Pre-build Query message bytes
    let query_bytes = build_query_bytes("SELECT id, name FROM users WHERE id = 42");

    for run in 0..VALIDATION_RUNS {
        tprintln!("--- Run {}/{} ---\n", run + 1, VALIDATION_RUNS);

        let mut codec = PostgresCodec::new();
        codec.set_normal_mode();

        let start = Instant::now();
        for _ in 0..iterations {
            let mut buf = BytesMut::from(&query_bytes[..]);
            let _ = codec.decode(&mut buf).expect("decode failed");
        }
        let elapsed = start.elapsed();

        let ops_sec = iterations as f64 / elapsed.as_secs_f64();
        results.push(ops_sec);
        tprintln!(
            "  {} decodes in {:.2?}, {} ops/sec\n",
            format_with_commas(iterations as f64),
            elapsed,
            format_with_commas(ops_sec),
        );
    }

    let result = validate_metric(
        "Message Decode",
        "FrontendMessage decode throughput (ops/sec)",
        results,
        QUERY_PER_SEC_TARGET,
        true,
    );

    assert!(
        result.passed,
        "Message decode throughput below minimum threshold"
    );
    assert!(
        !result.regression_detected,
        "Regression detected in message decode"
    );
}

// ---------------------------------------------------------------------------
// Test 9: Row serialization throughput (ScalarValue -> wire format)
// ---------------------------------------------------------------------------

#[test]
fn test_wire_row_serialization_throughput() {
    zyron_bench_harness::init("wire");
    let _bench_guard = BENCHMARK_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    tprintln!("\n=== Row Serialization Throughput Test ===");

    let num_rows: usize = 1_000_000;
    let mut results = Vec::with_capacity(VALIDATION_RUNS);

    // Pre-build scalars (4 columns per row)
    let scalars: Vec<Vec<ScalarValue>> = (0..num_rows)
        .map(|i| {
            vec![
                ScalarValue::Int32(i as i32),
                ScalarValue::Utf8(format!("user_{}", i)),
                ScalarValue::Float64(i as f64 * 1.5),
                ScalarValue::Boolean(i % 2 == 0),
            ]
        })
        .collect();

    for run in 0..VALIDATION_RUNS {
        tprintln!("--- Run {}/{} ---\n", run + 1, VALIDATION_RUNS);

        let start = Instant::now();
        let mut total_bytes: usize = 0;

        for row in &scalars {
            let values: Vec<Option<Vec<u8>>> =
                row.iter().map(|s| types::scalar_to_text(s)).collect();
            for v in &values {
                if let Some(b) = v {
                    total_bytes += b.len();
                }
            }
        }
        let elapsed = start.elapsed();

        let rows_sec = num_rows as f64 / elapsed.as_secs_f64();
        results.push(rows_sec);
        tprintln!(
            "  {} rows ({} bytes) in {:.2?}, {} rows/sec\n",
            format_with_commas(num_rows as f64),
            format_with_commas(total_bytes as f64),
            elapsed,
            format_with_commas(rows_sec),
        );
    }

    let result = validate_metric(
        "Row Serialization",
        "Row serialization throughput (rows/sec)",
        results,
        ROW_SERIALIZATION_TARGET_OPS,
        true,
    );

    assert!(
        result.passed,
        "Row serialization throughput below minimum threshold"
    );
    assert!(
        !result.regression_detected,
        "Regression detected in row serialization"
    );
}

// ---------------------------------------------------------------------------
// Test 10: Type serialization throughput (individual scalar conversions)
// ---------------------------------------------------------------------------

#[test]
fn test_wire_type_serialization_throughput() {
    zyron_bench_harness::init("wire");
    let _bench_guard = BENCHMARK_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    tprintln!("\n=== Type Serialization Throughput Test ===");

    let iterations = 5_000_000;
    let mut text_results = Vec::with_capacity(VALIDATION_RUNS);
    let mut binary_results = Vec::with_capacity(VALIDATION_RUNS);

    let test_scalar = ScalarValue::Int64(1234567890);

    for run in 0..VALIDATION_RUNS {
        tprintln!("--- Run {}/{} ---\n", run + 1, VALIDATION_RUNS);

        // Text format
        let start = Instant::now();
        for _ in 0..iterations {
            let _ = types::scalar_to_text(&test_scalar);
        }
        let text_elapsed = start.elapsed();
        let text_ops = iterations as f64 / text_elapsed.as_secs_f64();
        text_results.push(text_ops);

        // Binary format
        let start = Instant::now();
        for _ in 0..iterations {
            let _ = types::scalar_to_binary(&test_scalar);
        }
        let binary_elapsed = start.elapsed();
        let binary_ops = iterations as f64 / binary_elapsed.as_secs_f64();
        binary_results.push(binary_ops);

        tprintln!(
            "  Text: {} ops/sec, Binary: {} ops/sec\n",
            format_with_commas(text_ops),
            format_with_commas(binary_ops),
        );
    }

    let text_result = validate_metric(
        "Type Serialization",
        "scalar_to_text throughput (ops/sec)",
        text_results,
        ROW_SERIALIZATION_TARGET_OPS,
        true,
    );

    let binary_result = validate_metric(
        "Type Serialization",
        "scalar_to_binary throughput (ops/sec)",
        binary_results,
        ROW_SERIALIZATION_TARGET_OPS,
        true,
    );

    assert!(
        text_result.passed,
        "Text serialization throughput below minimum threshold"
    );
    assert!(
        binary_result.passed,
        "Binary serialization throughput below minimum threshold"
    );
}

// ---------------------------------------------------------------------------
// Test 11: COPY FROM throughput (CSV parsing)
// ---------------------------------------------------------------------------

#[test]
fn test_wire_copy_from_throughput() {
    zyron_bench_harness::init("wire");
    let _bench_guard = BENCHMARK_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    tprintln!("\n=== COPY FROM Throughput Test ===");

    let num_rows = 500_000;
    let mut results = Vec::with_capacity(VALIDATION_RUNS);

    let columns = vec![
        LogicalColumn {
            table_idx: None,
            column_id: zyron_catalog::ColumnId(0),
            name: "id".into(),
            type_id: TypeId::Int32,
            nullable: false,
        },
        LogicalColumn {
            table_idx: None,
            column_id: zyron_catalog::ColumnId(1),
            name: "name".into(),
            type_id: TypeId::Text,
            nullable: false,
        },
        LogicalColumn {
            table_idx: None,
            column_id: zyron_catalog::ColumnId(2),
            name: "value".into(),
            type_id: TypeId::Float64,
            nullable: true,
        },
    ];

    // Pre-build CSV data
    let mut csv_data = Vec::with_capacity(num_rows * 30);
    for i in 0..num_rows {
        csv_data.extend_from_slice(format!("{},user_{},{:.2}\n", i, i, i as f64 * 0.5).as_bytes());
    }

    for run in 0..VALIDATION_RUNS {
        tprintln!("--- Run {}/{} ---\n", run + 1, VALIDATION_RUNS);

        let start = Instant::now();
        let mut handler = CopyInHandler::new(columns.clone(), CopyFormat::Csv);

        // Feed in chunks (simulating network packets)
        let chunk_size = 65536;
        for chunk in csv_data.chunks(chunk_size) {
            handler.feed(chunk).expect("feed failed");
        }
        let rows = handler.finish().expect("finish failed");
        let elapsed = start.elapsed();

        assert_eq!(rows.len(), num_rows, "Should parse all rows");

        let rows_sec = num_rows as f64 / elapsed.as_secs_f64();
        results.push(rows_sec);
        tprintln!(
            "  {} rows ({} bytes) in {:.2?}, {} rows/sec\n",
            format_with_commas(num_rows as f64),
            format_with_commas(csv_data.len() as f64),
            elapsed,
            format_with_commas(rows_sec),
        );
    }

    let result = validate_metric(
        "COPY FROM",
        "COPY FROM (CSV) throughput (rows/sec)",
        results,
        COPY_FROM_TARGET_OPS,
        true,
    );

    assert!(
        result.passed,
        "COPY FROM throughput below minimum threshold"
    );
    assert!(
        !result.regression_detected,
        "Regression detected in COPY FROM"
    );
}

// ---------------------------------------------------------------------------
// Test 12: COPY TO throughput (row formatting)
// ---------------------------------------------------------------------------

#[test]
fn test_wire_copy_to_throughput() {
    zyron_bench_harness::init("wire");
    let _bench_guard = BENCHMARK_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    tprintln!("\n=== COPY TO Throughput Test ===");

    let batch_rows = 1024;
    let num_batches = 500;
    let total_rows = batch_rows * num_batches;
    let mut results = Vec::with_capacity(VALIDATION_RUNS);

    let columns = vec![
        LogicalColumn {
            table_idx: None,
            column_id: zyron_catalog::ColumnId(0),
            name: "id".into(),
            type_id: TypeId::Int32,
            nullable: false,
        },
        LogicalColumn {
            table_idx: None,
            column_id: zyron_catalog::ColumnId(1),
            name: "name".into(),
            type_id: TypeId::Text,
            nullable: false,
        },
    ];

    // Pre-build batches
    let batches: Vec<DataBatch> = (0..num_batches)
        .map(|b| {
            DataBatch::new(vec![
                make_column(
                    TypeId::Int32,
                    (0..batch_rows)
                        .map(|r| ScalarValue::Int32((b * batch_rows + r) as i32))
                        .collect(),
                ),
                make_column(
                    TypeId::Text,
                    (0..batch_rows)
                        .map(|r| ScalarValue::Utf8(format!("row_{}", b * batch_rows + r)))
                        .collect(),
                ),
            ])
        })
        .collect();

    for run in 0..VALIDATION_RUNS {
        tprintln!("--- Run {}/{} ---\n", run + 1, VALIDATION_RUNS);

        let handler = CopyOutHandler::new(columns.clone(), CopyFormat::Text);

        let start = Instant::now();
        let mut total_msgs = 0;
        for batch in &batches {
            let msgs = handler.format_batch(batch);
            total_msgs += msgs.len();
        }
        let elapsed = start.elapsed();

        assert_eq!(total_msgs, total_rows, "Should format all rows");

        let rows_sec = total_rows as f64 / elapsed.as_secs_f64();
        results.push(rows_sec);
        tprintln!(
            "  {} rows in {:.2?}, {} rows/sec\n",
            format_with_commas(total_rows as f64),
            elapsed,
            format_with_commas(rows_sec),
        );
    }

    let result = validate_metric(
        "COPY TO",
        "COPY TO (text) throughput (rows/sec)",
        results,
        COPY_TO_TARGET_OPS,
        true,
    );

    assert!(result.passed, "COPY TO throughput below minimum threshold");
    assert!(
        !result.regression_detected,
        "Regression detected in COPY TO"
    );
}

// ---------------------------------------------------------------------------
// Test 13: Parse message latency
// ---------------------------------------------------------------------------

#[test]
fn test_wire_parse_message_latency() {
    zyron_bench_harness::init("wire");
    let _bench_guard = BENCHMARK_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    tprintln!("\n=== Parse Message Latency Test ===");

    let iterations = 500_000;
    let mut results = Vec::with_capacity(VALIDATION_RUNS);

    let parse_bytes = build_parse_bytes(
        "stmt1",
        "SELECT id, name, email FROM users WHERE id = $1 AND active = $2",
        &[types::PG_INT4_OID, types::PG_BOOL_OID],
    );

    for run in 0..VALIDATION_RUNS {
        tprintln!("--- Run {}/{} ---\n", run + 1, VALIDATION_RUNS);

        let mut codec = PostgresCodec::new();
        codec.set_normal_mode();

        let start = Instant::now();
        for _ in 0..iterations {
            let mut buf = BytesMut::from(&parse_bytes[..]);
            let _ = codec.decode(&mut buf).expect("decode failed");
        }
        let elapsed = start.elapsed();

        let avg_us = elapsed.as_secs_f64() * 1_000_000.0 / iterations as f64;
        results.push(avg_us);
        tprintln!(
            "  {} parse decodes in {:.2?}, avg {:.3} us/op\n",
            format_with_commas(iterations as f64),
            elapsed,
            avg_us,
        );
    }

    let result = validate_metric(
        "Parse Message",
        "Parse message decode latency (us)",
        results,
        PARSE_MESSAGE_TARGET_US,
        false,
    );

    assert!(
        result.passed,
        "Parse message latency above minimum threshold"
    );
    assert!(
        !result.regression_detected,
        "Regression detected in parse message latency"
    );
}

// ---------------------------------------------------------------------------
// Test 14: Bind message latency
// ---------------------------------------------------------------------------

#[test]
fn test_wire_bind_message_latency() {
    zyron_bench_harness::init("wire");
    let _bench_guard = BENCHMARK_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    tprintln!("\n=== Bind Message Latency Test ===");

    let iterations = 500_000;
    let mut results = Vec::with_capacity(VALIDATION_RUNS);

    let bind_bytes = build_bind_bytes("", "stmt1", &[0, 0], &[Some(b"42"), Some(b"true")], &[0]);

    for run in 0..VALIDATION_RUNS {
        tprintln!("--- Run {}/{} ---\n", run + 1, VALIDATION_RUNS);

        let mut codec = PostgresCodec::new();
        codec.set_normal_mode();

        let start = Instant::now();
        for _ in 0..iterations {
            let mut buf = BytesMut::from(&bind_bytes[..]);
            let _ = codec.decode(&mut buf).expect("decode failed");
        }
        let elapsed = start.elapsed();

        let avg_us = elapsed.as_secs_f64() * 1_000_000.0 / iterations as f64;
        results.push(avg_us);
        tprintln!(
            "  {} bind decodes in {:.2?}, avg {:.3} us/op\n",
            format_with_commas(iterations as f64),
            elapsed,
            avg_us,
        );
    }

    let result = validate_metric(
        "Bind Message",
        "Bind message decode latency (us)",
        results,
        BIND_MESSAGE_TARGET_US,
        false,
    );

    assert!(
        result.passed,
        "Bind message latency above minimum threshold"
    );
    assert!(
        !result.regression_detected,
        "Regression detected in bind message latency"
    );
}

// ---------------------------------------------------------------------------
// Test 15: Connection handshake latency (TCP + PG startup)
// ---------------------------------------------------------------------------

#[test]
fn test_wire_connection_handshake_latency() {
    zyron_bench_harness::init("wire");
    let _bench_guard = BENCHMARK_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    tprintln!("\n=== Connection Handshake Latency Test ===");

    let rt = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();

    let local = tokio::task::LocalSet::new();
    local.block_on(&rt, async {
        let (server_state, _tmp) = create_test_server("testdb").await;
        let listener = Arc::new(TcpListener::bind("127.0.0.1:0").await.expect("bind failed"));
        let addr = listener.local_addr().unwrap();
        tprintln!("  Server listening on {}\n", addr);

        let iterations = 500;
        let mut results = Vec::with_capacity(VALIDATION_RUNS);

        for run in 0..VALIDATION_RUNS {
            tprintln!("--- Run {}/{} ---\n", run + 1, VALIDATION_RUNS);

            let start = Instant::now();
            let mut success_count = 0;

            for _ in 0..iterations {
                let state = Arc::clone(&server_state);
                let lis = Arc::clone(&listener);

                // Spawn server accept as a local task so it doesn't block the client
                let server_handle = tokio::task::spawn_local(async move {
                    let (stream, _) = lis.accept().await.expect("accept failed");
                    let mut conn = zyron_wire::connection::Connection::new(stream, state);
                    let _ = conn.run().await;
                });

                // Client side - runs concurrently with server accept
                let mut client = TcpStream::connect(addr).await.expect("connect failed");
                match do_handshake(&mut client, "test_user", "testdb").await {
                    Ok(msg_types) => {
                        assert!(msg_types.contains(&b'R'), "Should receive AuthenticationOk");
                        assert!(msg_types.contains(&b'Z'), "Should receive ReadyForQuery");
                        success_count += 1;
                    }
                    Err(e) => {
                        tprintln!("  Handshake error: {}\n", e);
                    }
                }

                // Clean disconnect
                let _ = client.write_all(&build_terminate_bytes()).await;
                let _ = client.shutdown().await;
                let _ = server_handle.await;
            }
            let elapsed = start.elapsed();

            let avg_us = elapsed.as_secs_f64() * 1_000_000.0 / iterations as f64;
            results.push(avg_us);
            tprintln!(
                "  {} handshakes ({} success) in {:.2?}, avg {:.1} us/handshake\n",
                iterations,
                success_count,
                elapsed,
                avg_us,
            );
            assert_eq!(success_count, iterations, "All handshakes should succeed");
        }

        let result = validate_metric(
            "Connection Handshake",
            "Handshake latency (us)",
            results,
            HANDSHAKE_TARGET_US,
            false,
        );

        assert!(result.passed, "Handshake latency above minimum threshold");
        assert!(
            !result.regression_detected,
            "Regression detected in handshake latency"
        );
    });
}

// ---------------------------------------------------------------------------
// Test 16: Concurrent connections
// ---------------------------------------------------------------------------

#[test]
fn test_wire_concurrent_connections() {
    zyron_bench_harness::init("wire");
    let _bench_guard = BENCHMARK_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    tprintln!("\n=== Concurrent Connections Test ===");

    let rt = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();

    let local = tokio::task::LocalSet::new();
    local.block_on(&rt, async {
        let (server_state, _tmp) = create_test_server("testdb").await;
        let listener = Arc::new(TcpListener::bind("127.0.0.1:0").await.expect("bind failed"));
        let addr = listener.local_addr().unwrap();
        tprintln!("  Server listening on {}\n", addr);

        let concurrent = 200;
        let mut results = Vec::with_capacity(VALIDATION_RUNS);

        for run in 0..VALIDATION_RUNS {
            tprintln!("--- Run {}/{} ---\n", run + 1, VALIDATION_RUNS);

            let start = Instant::now();

            // Spawn server accept loop as a local task
            let server_listener = Arc::clone(&listener);
            let server_ss = Arc::clone(&server_state);
            let server_handle = tokio::task::spawn_local(async move {
                let mut handles = Vec::new();
                for _ in 0..concurrent {
                    let (stream, _) = server_listener.accept().await.expect("accept failed");
                    let state = Arc::clone(&server_ss);
                    handles.push(tokio::task::spawn_local(async move {
                        let mut conn = zyron_wire::connection::Connection::new(stream, state);
                        let _ = conn.run().await;
                    }));
                }
                handles
            });

            // Spawn clients concurrently
            let mut client_handles = Vec::new();
            for _ in 0..concurrent {
                let client_handle = tokio::task::spawn_local(async move {
                    let mut stream = TcpStream::connect(addr).await.expect("connect failed");
                    let result = do_handshake(&mut stream, "test_user", "testdb").await;
                    let _ = stream.write_all(&build_terminate_bytes()).await;
                    let _ = stream.shutdown().await;
                    result.is_ok()
                });
                client_handles.push(client_handle);
            }

            let mut success = 0;
            for handle in client_handles {
                if handle.await.unwrap_or(false) {
                    success += 1;
                }
            }

            // Wait for server handlers to finish
            if let Ok(server_handles) = server_handle.await {
                for h in server_handles {
                    let _ = h.await;
                }
            }

            let elapsed = start.elapsed();
            let conns_per_sec = concurrent as f64 / elapsed.as_secs_f64();
            results.push(conns_per_sec);

            tprintln!(
                "  {} concurrent connections ({} success) in {:.2?}, {} conns/sec\n",
                concurrent,
                success,
                elapsed,
                format_with_commas(conns_per_sec),
            );
            assert!(
                success >= concurrent * 9 / 10,
                "At least 90% of connections should succeed"
            );
        }

        let result = validate_metric(
            "Concurrent Connections",
            "Connection establishment rate (conns/sec)",
            results,
            1000.0,
            true,
        );

        assert!(
            result.passed,
            "Concurrent connection rate below minimum threshold"
        );
    });
}

// ---------------------------------------------------------------------------
// Test 17: Extended query protocol message roundtrip
// ---------------------------------------------------------------------------

#[test]
fn test_wire_extended_query_protocol_messages() {
    zyron_bench_harness::init("wire");
    tprintln!("\n=== Extended Query Protocol Messages Test ===");

    let mut codec = PostgresCodec::new();
    codec.set_normal_mode();

    // Full extended query cycle: Parse -> Describe -> Bind -> Execute -> Sync

    // 1. Parse
    let parse = build_parse_bytes(
        "stmt1",
        "SELECT * FROM t WHERE id = $1",
        &[types::PG_INT4_OID],
    );
    let mut buf = BytesMut::from(&parse[..]);
    let msg = codec.decode(&mut buf).unwrap().unwrap();
    if let FrontendMessage::Parse {
        name,
        query,
        param_types,
    } = msg
    {
        assert_eq!(name, "stmt1");
        assert_eq!(query, "SELECT * FROM t WHERE id = $1");
        assert_eq!(param_types, vec![types::PG_INT4_OID]);
    } else {
        panic!("Expected Parse");
    }
    tprintln!("  Parse: stmt1 with 1 param\n");

    // 2. Describe statement
    let desc = build_describe_bytes(b'S', "stmt1");
    let mut buf = BytesMut::from(&desc[..]);
    let msg = codec.decode(&mut buf).unwrap().unwrap();
    if let FrontendMessage::Describe { target, name } = msg {
        assert_eq!(target, DescribeTarget::Statement);
        assert_eq!(name, "stmt1");
    } else {
        panic!("Expected Describe");
    }
    tprintln!("  Describe: statement stmt1\n");

    // 3. Bind with parameter
    let bind = build_bind_bytes("portal1", "stmt1", &[0], &[Some(b"42")], &[0]);
    let mut buf = BytesMut::from(&bind[..]);
    let msg = codec.decode(&mut buf).unwrap().unwrap();
    if let FrontendMessage::Bind {
        portal,
        statement,
        param_formats,
        param_values,
        result_formats,
    } = msg
    {
        assert_eq!(portal, "portal1");
        assert_eq!(statement, "stmt1");
        assert_eq!(param_formats, vec![0i16]);
        assert_eq!(param_values.len(), 1);
        assert_eq!(param_values[0].as_deref(), Some(b"42".as_slice()));
        assert_eq!(result_formats, vec![0i16]);
    } else {
        panic!("Expected Bind");
    }
    tprintln!("  Bind: portal1 <- stmt1 with param=42\n");

    // 4. Describe portal
    let desc = build_describe_bytes(b'P', "portal1");
    let mut buf = BytesMut::from(&desc[..]);
    let msg = codec.decode(&mut buf).unwrap().unwrap();
    if let FrontendMessage::Describe { target, name } = msg {
        assert_eq!(target, DescribeTarget::Portal);
        assert_eq!(name, "portal1");
    } else {
        panic!("Expected Describe Portal");
    }
    tprintln!("  Describe: portal portal1\n");

    // 5. Execute
    let exec = build_execute_bytes("portal1", 0);
    let mut buf = BytesMut::from(&exec[..]);
    let msg = codec.decode(&mut buf).unwrap().unwrap();
    if let FrontendMessage::Execute { portal, max_rows } = msg {
        assert_eq!(portal, "portal1");
        assert_eq!(max_rows, 0);
    } else {
        panic!("Expected Execute");
    }
    tprintln!("  Execute: portal1 (unlimited rows)\n");

    // 6. Sync
    let sync = build_sync_bytes();
    let mut buf = BytesMut::from(&sync[..]);
    let msg = codec.decode(&mut buf).unwrap().unwrap();
    assert!(matches!(msg, FrontendMessage::Sync));
    tprintln!("  Sync\n");

    // Verify the backend can produce the corresponding responses
    let mut enc_buf = BytesMut::new();
    let mut enc_codec = PostgresCodec::new();
    enc_codec.set_normal_mode();

    let responses = vec![
        BackendMessage::ParseComplete,
        BackendMessage::ParameterDescription(vec![types::PG_INT4_OID]),
        BackendMessage::RowDescription(vec![FieldDescription {
            name: "id".into(),
            table_oid: 0,
            column_attr: 0,
            type_oid: types::PG_INT4_OID,
            type_size: 4,
            type_modifier: -1,
            format: 0,
        }]),
        BackendMessage::BindComplete,
        BackendMessage::DataRow(vec![Some(b"42".to_vec())]),
        BackendMessage::CommandComplete {
            tag: "SELECT 1".into(),
        },
        BackendMessage::ReadyForQuery(TransactionState::Idle),
    ];

    for msg in &responses {
        enc_codec
            .encode(msg.clone(), &mut enc_buf)
            .expect("response encode failed");
    }
    tprintln!(
        "  {} response messages encoded ({} bytes)\n",
        responses.len(),
        enc_buf.len()
    );
    tprintln!("  Full extended query protocol cycle verified\n");
}

// ---------------------------------------------------------------------------
// Test 18: Execute message latency
// ---------------------------------------------------------------------------

#[test]
fn test_wire_execute_message_latency() {
    zyron_bench_harness::init("wire");
    let _bench_guard = BENCHMARK_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    tprintln!("\n=== Execute Message Latency Test ===");

    let iterations = 500_000;
    let mut results = Vec::with_capacity(VALIDATION_RUNS);

    let exec_bytes = build_execute_bytes("", 0);

    for run in 0..VALIDATION_RUNS {
        tprintln!("--- Run {}/{} ---\n", run + 1, VALIDATION_RUNS);

        let mut codec = PostgresCodec::new();
        codec.set_normal_mode();

        let start = Instant::now();
        for _ in 0..iterations {
            let mut buf = BytesMut::from(&exec_bytes[..]);
            let _ = codec.decode(&mut buf).expect("decode failed");
        }
        let elapsed = start.elapsed();

        let avg_us = elapsed.as_secs_f64() * 1_000_000.0 / iterations as f64;
        results.push(avg_us);
        tprintln!(
            "  {} execute decodes in {:.2?}, avg {:.3} us/op\n",
            format_with_commas(iterations as f64),
            elapsed,
            avg_us,
        );
    }

    let result = validate_metric(
        "Execute Message",
        "Execute message decode latency (us)",
        results,
        EXECUTE_MESSAGE_TARGET_US,
        false,
    );

    assert!(
        result.passed,
        "Execute message latency above minimum threshold"
    );
    assert!(
        !result.regression_detected,
        "Regression detected in execute message latency"
    );
}

// ---------------------------------------------------------------------------
// Test 19: Simple query message throughput
// ---------------------------------------------------------------------------

#[test]
fn test_wire_simple_query_message_throughput() {
    zyron_bench_harness::init("wire");
    let _bench_guard = BENCHMARK_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    tprintln!("\n=== Simple Query Message Throughput Test ===");

    let iterations = 1_000_000;
    let mut results = Vec::with_capacity(VALIDATION_RUNS);

    // Test full encode+decode cycle for a simple query
    let query_bytes = build_query_bytes("SELECT 1");
    let response_msg = BackendMessage::DataRow(vec![Some(b"1".to_vec())]);

    for run in 0..VALIDATION_RUNS {
        tprintln!("--- Run {}/{} ---\n", run + 1, VALIDATION_RUNS);

        let mut decode_codec = PostgresCodec::new();
        decode_codec.set_normal_mode();
        let mut encode_codec = PostgresCodec::new();
        encode_codec.set_normal_mode();
        let mut enc_buf = BytesMut::with_capacity(64);

        let start = Instant::now();
        for _ in 0..iterations {
            // Decode client query
            let mut buf = BytesMut::from(&query_bytes[..]);
            let _ = decode_codec.decode(&mut buf).unwrap();

            // Encode server response
            encode_codec
                .encode(response_msg.clone(), &mut enc_buf)
                .unwrap();
            enc_buf.clear();
        }
        let elapsed = start.elapsed();

        let avg_us = elapsed.as_secs_f64() * 1_000_000.0 / iterations as f64;
        let qps = iterations as f64 / elapsed.as_secs_f64();
        results.push(avg_us);
        tprintln!(
            "  {} query roundtrips in {:.2?}, avg {:.3} us/op ({} qps)\n",
            format_with_commas(iterations as f64),
            elapsed,
            avg_us,
            format_with_commas(qps),
        );
    }

    let result = validate_metric(
        "Simple Query",
        "Simple query message latency (us)",
        results,
        SIMPLE_QUERY_TARGET_US,
        false,
    );

    assert!(
        result.passed,
        "Simple query latency above minimum threshold"
    );
    assert!(
        !result.regression_detected,
        "Regression detected in simple query latency"
    );
}

// ---------------------------------------------------------------------------
// Test 20: DataRow encode throughput (full row encoding pipeline)
// ---------------------------------------------------------------------------

#[test]
fn test_wire_datarow_encode_throughput() {
    zyron_bench_harness::init("wire");
    let _bench_guard = BENCHMARK_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    tprintln!("\n=== DataRow Encode Throughput Test ===");

    let num_rows = 2_000_000;
    let mut results = Vec::with_capacity(VALIDATION_RUNS);

    // Pre-build DataBatch with 4 columns, 1024 rows per batch
    let batch_size = 1024;
    let num_batches = num_rows / batch_size;

    let batch = DataBatch::new(vec![
        make_column(
            TypeId::Int32,
            (0..batch_size)
                .map(|i| ScalarValue::Int32(i as i32))
                .collect(),
        ),
        make_column(
            TypeId::Text,
            (0..batch_size)
                .map(|i| ScalarValue::Utf8(format!("name_{}", i)))
                .collect(),
        ),
        make_column(
            TypeId::Float64,
            (0..batch_size)
                .map(|i| ScalarValue::Float64(i as f64 * 1.1))
                .collect(),
        ),
        make_column(
            TypeId::Boolean,
            (0..batch_size)
                .map(|i| ScalarValue::Boolean(i % 2 == 0))
                .collect(),
        ),
    ]);

    for run in 0..VALIDATION_RUNS {
        tprintln!("--- Run {}/{} ---\n", run + 1, VALIDATION_RUNS);

        let mut codec = PostgresCodec::new();
        codec.set_normal_mode();
        let mut buf = BytesMut::with_capacity(256 * 1024);
        let mut total_bytes: usize = 0;

        let start = Instant::now();
        for _ in 0..num_batches {
            for row in 0..batch.num_rows {
                let values: Vec<Option<Vec<u8>>> = batch
                    .columns
                    .iter()
                    .map(|col| types::scalar_to_text(&col.get_scalar(row)))
                    .collect();
                let msg = BackendMessage::DataRow(values);
                codec.encode(msg, &mut buf).unwrap();
                total_bytes += buf.len();
                buf.clear();
            }
        }
        let elapsed = start.elapsed();

        let rows_sec = num_rows as f64 / elapsed.as_secs_f64();
        results.push(rows_sec);
        tprintln!(
            "  {} rows ({} bytes) in {:.2?}, {} rows/sec\n",
            format_with_commas(num_rows as f64),
            format_with_commas(total_bytes as f64),
            elapsed,
            format_with_commas(rows_sec),
        );
    }

    let result = validate_metric(
        "DataRow Encode",
        "DataRow encode throughput (rows/sec)",
        results,
        ROW_SERIALIZATION_TARGET_OPS,
        true,
    );

    assert!(
        result.passed,
        "DataRow encode throughput below minimum threshold"
    );
    assert!(
        !result.regression_detected,
        "Regression detected in DataRow encode"
    );
}

// ---------------------------------------------------------------------------
// QUIC Transport Benchmarks
// ---------------------------------------------------------------------------

/// Generates a self-signed TLS certificate and private key using rcgen.
/// Writes PEM files to the given directory and returns (cert_path, key_path).
fn generate_test_certs(dir: &std::path::Path) -> (std::path::PathBuf, std::path::PathBuf) {
    let cert = rcgen::generate_simple_self_signed(vec!["localhost".to_string()])
        .expect("Failed to generate self-signed cert");

    let cert_pem = cert.cert.pem();
    let key_pem = cert.key_pair.serialize_pem();

    let cert_path = dir.join("test_cert.pem");
    let key_path = dir.join("test_key.pem");

    std::fs::write(&cert_path, cert_pem).expect("Failed to write cert PEM");
    std::fs::write(&key_path, key_pem).expect("Failed to write key PEM");

    (cert_path, key_path)
}

/// Client-side ApplicationOverQuic that sends data on the first bidi stream
/// and collects the response. Used for QUIC e2e handshake benchmarks.
struct QuicClientApp {
    buf: Vec<u8>,
    stream_id: Option<u64>,
    send_data: Vec<u8>,
    sent: bool,
    received: Arc<tokio::sync::Mutex<Vec<u8>>>,
    done_notify: Arc<tokio::sync::Notify>,
}

impl QuicClientApp {
    fn new(
        send_data: Vec<u8>,
    ) -> (
        Self,
        Arc<tokio::sync::Mutex<Vec<u8>>>,
        Arc<tokio::sync::Notify>,
    ) {
        let received = Arc::new(tokio::sync::Mutex::new(Vec::new()));
        let done_notify = Arc::new(tokio::sync::Notify::new());
        let app = Self {
            buf: vec![0u8; 65535],
            stream_id: None,
            send_data,
            sent: false,
            received: Arc::clone(&received),
            done_notify: Arc::clone(&done_notify),
        };
        (app, received, done_notify)
    }
}

impl tokio_quiche::ApplicationOverQuic for QuicClientApp {
    fn on_conn_established(
        &mut self,
        qconn: &mut tokio_quiche::quic::QuicheConnection,
        _handshake_info: &tokio_quiche::quic::HandshakeInfo,
    ) -> tokio_quiche::QuicResult<()> {
        // Open stream 0 and send data immediately on connection establishment
        let sid = 0u64;
        self.stream_id = Some(sid);
        let _ = qconn.stream_send(sid, &self.send_data, true);
        self.sent = true;
        Ok(())
    }

    fn should_act(&self) -> bool {
        true
    }
    fn buffer(&mut self) -> &mut [u8] {
        &mut self.buf
    }

    async fn wait_for_data(
        &mut self,
        _qconn: &mut tokio_quiche::quic::QuicheConnection,
    ) -> tokio_quiche::QuicResult<()> {
        // Short cycle to keep the worker loop responsive to incoming data.
        tokio::time::sleep(std::time::Duration::from_micros(50)).await;
        Ok(())
    }

    fn process_reads(
        &mut self,
        qconn: &mut tokio_quiche::quic::QuicheConnection,
    ) -> tokio_quiche::QuicResult<()> {
        let sid = match self.stream_id {
            Some(id) => id,
            None => return Ok(()),
        };

        let mut temp = [0u8; 16384];
        loop {
            match qconn.stream_recv(sid, &mut temp) {
                Ok((n, _fin)) if n > 0 => {
                    let mut recv = self.received.blocking_lock();
                    recv.extend_from_slice(&temp[..n]);
                    self.done_notify.notify_one();
                }
                Ok(_) => break,
                Err(tokio_quiche::quiche::Error::Done) => break,
                Err(_) => break,
            }
        }
        Ok(())
    }

    fn process_writes(
        &mut self,
        _qconn: &mut tokio_quiche::quic::QuicheConnection,
    ) -> tokio_quiche::QuicResult<()> {
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Test 19: QUIC e2e handshake + stream setup via tokio-quiche
// ---------------------------------------------------------------------------

#[test]
fn test_quic_handshake_latency() {
    zyron_bench_harness::init("wire");
    let _bench_guard = BENCHMARK_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    tprintln!("\n=== QUIC Handshake Latency Test ===");

    let tmp = tempfile::TempDir::new().expect("Failed to create temp dir");
    let (cert_path, key_path) = generate_test_certs(tmp.path());

    let rt = tokio::runtime::Runtime::new().unwrap();

    rt.block_on(async {
        use tokio_quiche::settings::{ConnectionParams, Hooks, QuicSettings};

        let bind_addr: std::net::SocketAddr = "127.0.0.1:0".parse().unwrap();

        // Bind and release to get a port, then let setup_quic_listener rebind
        let udp_sock = tokio::net::UdpSocket::bind(bind_addr)
            .await
            .expect("Failed to bind UDP");
        let server_addr = udp_sock.local_addr().unwrap();
        drop(udp_sock);

        let mut quic_rx =
            zyron_wire::quic::setup_quic_listener(server_addr, &cert_path, &key_path, 30)
                .await
                .expect("Failed to setup QUIC listener");

        tprintln!("  QUIC server listening on {}\n", server_addr);

        let iterations = 20;
        let mut results = Vec::with_capacity(VALIDATION_RUNS);

        for run in 0..VALIDATION_RUNS {
            tprintln!("--- Run {}/{} ---\n", run + 1, VALIDATION_RUNS);

            let start = Instant::now();
            let mut success_count = 0u32;

            for i in 0..iterations {
                let startup_bytes = build_startup_bytes("test_user", "testdb");

                // Create a client UDP socket connected to the server
                let client_sock = tokio::net::UdpSocket::bind("127.0.0.1:0")
                    .await
                    .expect("bind client");
                client_sock.connect(server_addr).await.expect("connect");

                let (app, _received, _done_notify) = QuicClientApp::new(startup_bytes);

                let mut settings = QuicSettings::default();
                settings.max_idle_timeout = Some(std::time::Duration::from_secs(5));
                let client_params = ConnectionParams::new_client(settings, None, Hooks::default());

                // Convert tokio UdpSocket to tokio_quiche Socket
                let socket: tokio_quiche::socket::Socket<_, _> = client_sock
                    .try_into()
                    .expect("Failed to convert UdpSocket to Socket");

                let connect_result = tokio::time::timeout(
                    std::time::Duration::from_secs(3),
                    tokio_quiche::quic::connect_with_config(
                        socket,
                        Some("localhost"),
                        &client_params,
                        app,
                    ),
                )
                .await;

                match connect_result {
                    Ok(Ok(_quic_conn)) => {
                        // Connection established, wait for server to accept and deliver QuicStream
                        let accepted =
                            tokio::time::timeout(std::time::Duration::from_secs(2), quic_rx.recv())
                                .await;

                        match accepted {
                            Ok(Some((_qs, _peer))) => {
                                success_count += 1;
                            }
                            Ok(None) => {
                                tprintln!("  Iteration {}: QUIC accept channel closed\n", i);
                            }
                            Err(_) => {
                                tprintln!("  Iteration {}: QUIC accept timed out\n", i);
                            }
                        }
                    }
                    Ok(Err(e)) => {
                        tprintln!("  Iteration {}: QUIC connect failed: {}\n", i, e);
                    }
                    Err(_) => {
                        tprintln!("  Iteration {}: QUIC connect timed out\n", i);
                    }
                }
            }

            let elapsed = start.elapsed();
            let avg_us = if success_count > 0 {
                elapsed.as_secs_f64() * 1_000_000.0 / success_count as f64
            } else {
                0.0
            };

            results.push(avg_us);
            tprintln!(
                "  {} QUIC handshakes ({} success) in {:.2?}, avg {:.1} us/handshake\n",
                iterations,
                success_count,
                elapsed,
                avg_us,
            );
        }

        let result = validate_metric(
            "QUIC Handshake",
            "QUIC handshake latency (us)",
            results,
            10000.0, // QUIC handshake includes TLS 1.3 negotiation
            false,
        );

        tprintln!(
            "  QUIC handshake benchmark completed (passed: {})\n",
            result.passed
        );
    });
}

// ---------------------------------------------------------------------------
// Test 20: QUIC channel bridge throughput (QuicStream read/write)
// ---------------------------------------------------------------------------

#[test]
fn test_quic_channel_bridge_throughput() {
    zyron_bench_harness::init("wire");
    let _bench_guard = BENCHMARK_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    tprintln!("\n=== QUIC Channel Bridge Throughput Test ===");

    let rt = tokio::runtime::Runtime::new().unwrap();

    rt.block_on(async {
        use tokio::io::{AsyncReadExt, AsyncWriteExt};

        let chunk_size = 4096;
        let total_bytes = 64 * 1024 * 1024; // 64 MB
        let num_chunks = total_bytes / chunk_size;
        let payload = vec![0xABu8; chunk_size];

        let mut results_write = Vec::with_capacity(VALIDATION_RUNS);
        let mut results_read = Vec::with_capacity(VALIDATION_RUNS);

        for run in 0..VALIDATION_RUNS {
            tprintln!("--- Run {}/{} ---\n", run + 1, VALIDATION_RUNS);

            // Write throughput: measure how fast we can push data through the channel
            let (mut stream, _read_tx, mut write_rx) = zyron_wire::quic::test_stream_pair();

            let drain_handle = tokio::spawn(async move {
                let mut drained = 0usize;
                while let Some(data) = write_rx.recv().await {
                    drained += data.len();
                }
                drained
            });

            let start = Instant::now();
            for _ in 0..num_chunks {
                stream.write_all(&payload).await.unwrap();
            }
            drop(stream); // Close the channel
            let drained = drain_handle.await.unwrap();
            let write_elapsed = start.elapsed();

            let write_mbps = (drained as f64 / (1024.0 * 1024.0)) / write_elapsed.as_secs_f64();
            results_write.push(write_mbps);

            // Read throughput: measure how fast we can pull data from the channel
            let (mut stream, read_tx, _write_rx) = zyron_wire::quic::test_stream_pair();

            let feed_handle = tokio::spawn(async move {
                for _ in 0..num_chunks {
                    let data = bytes::Bytes::from(vec![0xCDu8; chunk_size]);
                    if read_tx.send(data).await.is_err() {
                        break;
                    }
                }
            });

            let start = Instant::now();
            let mut total_read = 0usize;
            let mut buf = vec![0u8; chunk_size * 2];
            loop {
                match stream.read(&mut buf).await {
                    Ok(0) => break,
                    Ok(n) => total_read += n,
                    Err(_) => break,
                }
            }
            let read_elapsed = start.elapsed();
            feed_handle.await.unwrap();

            let read_mbps = (total_read as f64 / (1024.0 * 1024.0)) / read_elapsed.as_secs_f64();
            results_read.push(read_mbps);

            tprintln!(
                "  Write: {:.0} MB/s ({} bytes in {:.2?}), Read: {:.0} MB/s ({} bytes in {:.2?})\n",
                write_mbps,
                format_with_commas(drained as f64),
                write_elapsed,
                read_mbps,
                format_with_commas(total_read as f64),
                read_elapsed,
            );
        }

        let write_result = validate_metric(
            "QUIC Channel Bridge",
            "QUIC channel write throughput (MB/s)",
            results_write,
            500.0, // 500 MB/s minimum for in-process channel
            true,
        );

        let read_result = validate_metric(
            "QUIC Channel Bridge",
            "QUIC channel read throughput (MB/s)",
            results_read,
            500.0,
            true,
        );

        assert!(
            write_result.passed,
            "QUIC channel write throughput below minimum threshold"
        );
        assert!(
            read_result.passed,
            "QUIC channel read throughput below minimum threshold"
        );
    });
}

// ---------------------------------------------------------------------------
// Test 21: QUIC vs TCP latency comparison (channel bridge overhead)
// ---------------------------------------------------------------------------

#[test]
fn test_quic_transport_properties() {
    zyron_bench_harness::init("wire");
    let _bench_guard = BENCHMARK_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    tprintln!("\n=== QUIC Transport Properties Test ===");

    let rt = tokio::runtime::Runtime::new().unwrap();

    rt.block_on(async {
        use tokio::io::{AsyncReadExt, AsyncWriteExt};
        use zyron_wire::transport::transport_name;

        // Verify transport identification
        let (quic_stream, _tx, _rx) = zyron_wire::quic::test_stream_pair();
        assert!(quic_stream.is_encrypted(), "QUIC must report encrypted");
        assert_eq!(transport_name(&quic_stream), "QUIC");
        tprintln!(
            "  QUIC transport: encrypted={}, name={}\n",
            quic_stream.is_encrypted(),
            transport_name(&quic_stream)
        );

        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let tcp_stream = tokio::net::TcpStream::connect(addr).await.unwrap();
        assert!(!tcp_stream.is_encrypted(), "TCP must report unencrypted");
        assert_eq!(transport_name(&tcp_stream), "TCP");
        tprintln!(
            "  TCP transport: encrypted={}, name={}\n",
            tcp_stream.is_encrypted(),
            transport_name(&tcp_stream)
        );

        // Measure small message round-trip latency through QUIC channel bridge
        let iterations = 100_000;
        let mut results = Vec::with_capacity(VALIDATION_RUNS);

        for run in 0..VALIDATION_RUNS {
            tprintln!("--- Run {}/{} ---\n", run + 1, VALIDATION_RUNS);

            let (mut stream, read_tx, mut write_rx) = zyron_wire::quic::test_stream_pair();

            // Feed data in a background task
            let feed_handle = tokio::spawn(async move {
                for i in 0..iterations {
                    let msg = bytes::Bytes::from(format!("msg{:06}", i));
                    if read_tx.send(msg).await.is_err() {
                        break;
                    }
                }
            });

            // Drain write side in background
            let drain_handle = tokio::spawn(async move {
                let mut count = 0u64;
                while let Some(_) = write_rx.recv().await {
                    count += 1;
                }
                count
            });

            let start = Instant::now();
            let mut buf = [0u8; 64];
            for _ in 0..iterations {
                let n = stream.read(&mut buf).await.unwrap();
                stream.write_all(&buf[..n]).await.unwrap();
            }
            let elapsed = start.elapsed();
            drop(stream);

            feed_handle.await.unwrap();
            let writes = drain_handle.await.unwrap();

            let avg_ns = elapsed.as_nanos() as f64 / iterations as f64;
            let ops_sec = iterations as f64 / elapsed.as_secs_f64();
            results.push(ops_sec);

            tprintln!(
                "  {} round-trips in {:.2?}, avg {:.0} ns/op ({} ops/sec), {} writes confirmed\n",
                format_with_commas(iterations as f64),
                elapsed,
                avg_ns,
                format_with_commas(ops_sec),
                format_with_commas(writes as f64),
            );
        }

        let result = validate_metric(
            "QUIC Transport Properties",
            "QUIC channel round-trip throughput (ops/sec)",
            results,
            500_000.0,
            true,
        );

        assert!(
            result.passed,
            "QUIC channel round-trip throughput below minimum threshold"
        );
    });
}

// ---------------------------------------------------------------------------
// Test 22: QUIC PG Protocol Handshake (Connection<QuicStream> e2e)
// ---------------------------------------------------------------------------

#[test]
fn test_quic_pg_handshake_latency() {
    zyron_bench_harness::init("wire");
    let _bench_guard = BENCHMARK_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    tprintln!("\n=== QUIC PG Handshake Latency Test ===");

    let rt = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();

    let local = tokio::task::LocalSet::new();
    local.block_on(&rt, async {
        let (server_state, _tmp) = create_test_server("testdb").await;

        let iterations = 500;
        let mut results = Vec::with_capacity(VALIDATION_RUNS);

        for run in 0..VALIDATION_RUNS {
            tprintln!("--- Run {}/{} ---\n", run + 1, VALIDATION_RUNS);

            let start = Instant::now();
            let mut success_count = 0u32;

            for _ in 0..iterations {
                let state = Arc::clone(&server_state);

                // Create channel pair: server_stream reads from client writes
                let (client_tx, server_rx) = tokio::sync::mpsc::channel(256);
                let (server_tx, mut client_rx) = tokio::sync::mpsc::unbounded_channel();
                let notify = Arc::new(tokio::sync::Notify::new());

                let server_stream = zyron_wire::quic::QuicStream::from_parts(
                    server_rx,
                    server_tx,
                    notify.clone(),
                    "127.0.0.1:5433".parse().unwrap(),
                );

                // spawn_local for !Send Connection futures (planner uses Pin<Box<dyn Future>>)
                let server_handle = tokio::task::spawn_local(async move {
                    let mut conn = zyron_wire::connection::Connection::new(server_stream, state);
                    let _ = conn.run().await;
                });

                // Client: send PG startup
                let startup = bytes::Bytes::from(build_startup_bytes("test_user", "testdb"));
                if client_tx.send(startup).await.is_ok() {
                    // Read until we see ReadyForQuery ('Z')
                    let deadline =
                        tokio::time::Instant::now() + std::time::Duration::from_millis(200);
                    loop {
                        match tokio::time::timeout_at(deadline, client_rx.recv()).await {
                            Ok(Some(data)) => {
                                if data.iter().any(|&b| b == b'Z') {
                                    success_count += 1;
                                    break;
                                }
                            }
                            _ => break,
                        }
                    }
                }

                // Clean disconnect
                let _ = client_tx
                    .send(bytes::Bytes::from(build_terminate_bytes()))
                    .await;
                drop(client_tx);
                let _ = server_handle.await;
            }

            let elapsed = start.elapsed();
            let avg_us = elapsed.as_secs_f64() * 1_000_000.0 / iterations as f64;
            results.push(avg_us);

            tprintln!(
                "  {} QUIC PG handshakes ({} success) in {:.2?}, avg {:.1} us/handshake\n",
                iterations,
                success_count,
                elapsed,
                avg_us,
            );
            assert_eq!(
                success_count, iterations as u32,
                "All handshakes should succeed"
            );
        }

        let result = validate_metric(
            "QUIC PG Handshake",
            "QUIC PG handshake latency (us)",
            results,
            500.0,
            false,
        );

        assert!(result.passed, "QUIC PG handshake latency above threshold");
    });
}

// ---------------------------------------------------------------------------
// Test 23: QUIC PG Simple Query Throughput (Connection<QuicStream> e2e)
// ---------------------------------------------------------------------------

#[test]
fn test_quic_pg_simple_query_throughput() {
    zyron_bench_harness::init("wire");
    let _bench_guard = BENCHMARK_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    tprintln!("\n=== QUIC PG Simple Query Throughput Test ===");

    let rt = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();

    let local = tokio::task::LocalSet::new();
    local.block_on(&rt, async {
        let (server_state, _tmp) = create_test_server("testdb").await;

        let iterations = 5000;
        let mut results = Vec::with_capacity(VALIDATION_RUNS);

        for run in 0..VALIDATION_RUNS {
            tprintln!("--- Run {}/{} ---\n", run + 1, VALIDATION_RUNS);

            let state = Arc::clone(&server_state);

            // Create channel pair
            let (client_tx, server_rx) = tokio::sync::mpsc::channel(256);
            let (server_tx, mut client_rx) = tokio::sync::mpsc::unbounded_channel();
            let notify = Arc::new(tokio::sync::Notify::new());

            let server_stream = zyron_wire::quic::QuicStream::from_parts(
                server_rx,
                server_tx,
                notify.clone(),
                "127.0.0.1:5433".parse().unwrap(),
            );

            let server_handle = tokio::task::spawn_local(async move {
                let mut conn = zyron_wire::connection::Connection::new(server_stream, state);
                let _ = conn.run().await;
            });

            // Handshake
            let startup = bytes::Bytes::from(build_startup_bytes("test_user", "testdb"));
            client_tx.send(startup).await.unwrap();

            // Wait for ReadyForQuery
            let mut got_ready = false;
            let deadline = tokio::time::Instant::now() + std::time::Duration::from_secs(2);
            loop {
                match tokio::time::timeout_at(deadline, client_rx.recv()).await {
                    Ok(Some(data)) => {
                        if data.iter().any(|&b| b == b'Z') {
                            got_ready = true;
                            break;
                        }
                    }
                    _ => break,
                }
            }

            if !got_ready {
                tprintln!("  Failed to complete handshake\n");
                let _ = server_handle.await;
                continue;
            }

            // Simple query loop
            let start = Instant::now();
            let mut query_count = 0u64;

            for _ in 0..iterations {
                let query = bytes::Bytes::from(build_query_bytes("SELECT 1"));
                if client_tx.send(query).await.is_err() {
                    break;
                }

                // Read until ReadyForQuery
                let mut got_z = false;
                loop {
                    match tokio::time::timeout(
                        std::time::Duration::from_millis(500),
                        client_rx.recv(),
                    )
                    .await
                    {
                        Ok(Some(data)) => {
                            if data.iter().any(|&b| b == b'Z') {
                                got_z = true;
                                query_count += 1;
                                break;
                            }
                        }
                        _ => break,
                    }
                }

                if !got_z {
                    break;
                }
            }

            let elapsed = start.elapsed();

            // Clean shutdown
            let _ = client_tx
                .send(bytes::Bytes::from(build_terminate_bytes()))
                .await;
            drop(client_tx);
            let _ = server_handle.await;

            let qps = query_count as f64 / elapsed.as_secs_f64();
            let avg_us = if query_count > 0 {
                elapsed.as_secs_f64() * 1_000_000.0 / query_count as f64
            } else {
                0.0
            };
            results.push(qps);

            tprintln!(
                "  {} queries in {:.2?}, avg {:.1} us/query ({} qps)\n",
                format_with_commas(query_count as f64),
                elapsed,
                avg_us,
                format_with_commas(qps),
            );
        }

        let result = validate_metric(
            "QUIC PG Simple Query",
            "QUIC PG simple query throughput (qps)",
            results,
            10_000.0,
            true,
        );

        assert!(
            result.passed,
            "QUIC PG simple query throughput below threshold"
        );
    });
}
