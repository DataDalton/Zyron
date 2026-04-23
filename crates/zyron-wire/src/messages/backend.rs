//! Backend (server-to-client) message definitions and encoding.
//!
//! Handles all messages the server sends to PostgreSQL clients, including
//! authentication responses, query results, error/notice reporting, and
//! the extended query protocol acknowledgments.

use bytes::{Buf, BufMut, BytesMut};

use super::ProtocolError;

// ----------------------------------------------------------------------------
// Zyron subscription extension message type codes
// ----------------------------------------------------------------------------

/// Change batch byte (X). Carries one or more row deltas to the consumer.
pub const CHANGE_BATCH_MSG_TYPE: u8 = b'X';

/// Subscription status byte (Q). Periodic heartbeat from producer.
pub const SUBSCRIPTION_STATUS_MSG_TYPE: u8 = b'Q';

/// Schema update byte (0x76, lowercase v). Distinct from ParameterStatus
/// which also uses uppercase S, so the lowercase byte avoids any collision
/// in shared inspection code paths.
pub const SCHEMA_UPDATE_MSG_TYPE: u8 = 0x76;

/// Subscribe acknowledgement byte (K). Server's response to a Y request.
/// PG assigns K to BackendKeyData, which is only valid during startup, so
/// post-startup the byte is unambiguous inside subscription mode.
pub const SUBSCRIBE_OK_MSG_TYPE: u8 = b'K';

/// Messages sent from server to client.
#[derive(Debug, Clone)]
pub enum BackendMessage {
    /// Authentication request or success.
    Authentication(AuthenticationMessage),
    /// Server parameter notification (server_version, client_encoding, etc.).
    ParameterStatus { name: String, value: String },
    /// Backend key data for cancel requests.
    BackendKeyData { process_id: i32, secret_key: i32 },
    /// Server is ready for a new query cycle.
    ReadyForQuery(TransactionState),
    /// Column metadata for query results.
    RowDescription(Vec<FieldDescription>),
    /// A single data row with column values.
    DataRow(Vec<Option<Vec<u8>>>),
    /// Command completion tag (e.g. "SELECT 5", "INSERT 0 1").
    CommandComplete { tag: String },
    /// Error response with structured fields.
    ErrorResponse(ErrorFields),
    /// Notice (warning/info) with structured fields.
    NoticeResponse(ErrorFields),
    /// Parse step completed successfully.
    ParseComplete,
    /// Bind step completed successfully.
    BindComplete,
    /// Close step completed successfully.
    CloseComplete,
    /// Portal suspended due to row limit in Execute.
    PortalSuspended,
    /// No data available (Describe on a non-SELECT prepared statement).
    NoData,
    /// Parameter type OIDs for a prepared statement.
    ParameterDescription(Vec<i32>),
    /// Empty query string was received.
    EmptyQueryResponse,
    /// Server ready to receive COPY data from client.
    CopyInResponse {
        format: i8,
        column_formats: Vec<i16>,
    },
    /// Server sending COPY data to client.
    CopyOutResponse {
        format: i8,
        column_formats: Vec<i16>,
    },
    /// COPY data chunk from server.
    CopyData(Vec<u8>),
    /// COPY operation complete from server.
    CopyDone,
    /// X: Change batch delivery with one or more row deltas.
    ChangeBatch(ChangeBatchMessage),
    /// Q: Subscription status heartbeat from producer.
    SubscriptionStatus(SubscriptionStatusMessage),
    /// v: Schema fingerprint change notification.
    SchemaUpdate(SchemaUpdateMessage),
    /// K: Subscribe acknowledgement carrying initial schema and resume LSN.
    SubscribeOk(SubscribeOkMessage),
}

/// One change row inside a ChangeBatchMessage.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RowDelta {
    /// 0 = Insert, 1 = Delete, 2 = UpdateBefore, 3 = UpdateAfter.
    pub change_type: u8,
    /// Source table OID.
    pub table_id: u32,
    /// LSN at which this row was produced.
    pub lsn: u64,
    /// Row payload encoded per the publication's NSM layout.
    pub row_bytes: Vec<u8>,
    /// Optional primary key bytes for UPSERT dedup on the consumer side.
    pub primary_key_bytes: Vec<u8>,
}

/// X body. A contiguous batch of changes between two LSNs.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ChangeBatchMessage {
    /// Smallest LSN in the batch.
    pub start_lsn: u64,
    /// Largest LSN in the batch.
    pub end_lsn: u64,
    /// Number of row deltas.
    pub row_count: u32,
    pub rows: Vec<RowDelta>,
    /// Producer's commit time in microseconds since Unix epoch.
    pub commit_timestamp_us: i64,
}

/// Q body. Heartbeat with producer's current committed LSN and wall clock.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SubscriptionStatusMessage {
    pub committed_lsn: u64,
    pub producer_now_us: i64,
}

/// One column definition inside a schema notification.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PublishedColumn {
    pub name: String,
    /// Compact TypeId value as a single byte for on-the-wire size.
    pub type_id: u8,
    pub nullable: bool,
    pub ordinal: u16,
}

/// v body. Fires when the publication's schema fingerprint changes.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SchemaUpdateMessage {
    pub publication: String,
    pub new_fingerprint: [u8; 32],
    pub columns: Vec<PublishedColumn>,
}

/// K body. Initial response to Y with the active schema and resume LSN.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SubscribeOkMessage {
    pub schema_fingerprint: [u8; 32],
    pub columns: Vec<PublishedColumn>,
    pub resumed_at_lsn: u64,
    pub features: u32,
}

/// Authentication message variants.
#[derive(Debug, Clone)]
pub enum AuthenticationMessage {
    /// Authentication succeeded.
    Ok,
    /// Server requests cleartext password.
    CleartextPassword,
    /// Server requests MD5-hashed password with the given 4-byte salt.
    Md5Password { salt: [u8; 4] },
    /// Server lists supported SASL mechanisms.
    SaslMechanisms(Vec<String>),
    /// Server SASL challenge data.
    SaslContinue(Vec<u8>),
    /// Server SASL final verification data.
    SaslFinal(Vec<u8>),
}

/// Transaction status indicator for ReadyForQuery messages.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TransactionState {
    /// Not in a transaction block.
    Idle,
    /// Inside a transaction block.
    InTransaction,
    /// Inside a failed transaction block (queries rejected until ROLLBACK).
    Failed,
}

impl TransactionState {
    /// Returns the single-byte wire representation.
    fn as_byte(self) -> u8 {
        match self {
            TransactionState::Idle => b'I',
            TransactionState::InTransaction => b'T',
            TransactionState::Failed => b'E',
        }
    }
}

/// Column metadata for RowDescription messages.
#[derive(Debug, Clone)]
pub struct FieldDescription {
    /// Column name.
    pub name: String,
    /// Table OID (0 if not a table column).
    pub table_oid: i32,
    /// Column attribute number (0 if not a table column).
    pub column_attr: i16,
    /// PostgreSQL type OID.
    pub type_oid: i32,
    /// Type size in bytes (-1 for variable-length).
    pub type_size: i16,
    /// Type-specific modifier (-1 if not applicable).
    pub type_modifier: i32,
    /// Format code: 0 = text, 1 = binary.
    pub format: i16,
}

/// Structured error/notice fields matching PostgreSQL's ErrorResponse format.
#[derive(Debug, Clone)]
pub struct ErrorFields {
    /// Severity level (ERROR, FATAL, PANIC, WARNING, NOTICE, DEBUG, INFO, LOG).
    pub severity: String,
    /// SQLSTATE 5-character error code.
    pub code: String,
    /// Primary human-readable error message.
    pub message: String,
    /// Optional detailed error description.
    pub detail: Option<String>,
    /// Optional suggestion for fixing the error.
    pub hint: Option<String>,
    /// Optional character position in the query where the error occurred.
    pub position: Option<i32>,
}

impl BackendMessage {
    /// Encodes this message into the output buffer.
    /// Format: type byte (1) + length (4, includes itself) + payload.
    pub fn encode(&self, buf: &mut BytesMut) {
        match self {
            BackendMessage::Authentication(auth) => {
                buf.put_u8(b'R');
                let len_pos = buf.len();
                buf.put_i32(0); // placeholder

                match auth {
                    AuthenticationMessage::Ok => {
                        buf.put_i32(0);
                    }
                    AuthenticationMessage::CleartextPassword => {
                        buf.put_i32(3);
                    }
                    AuthenticationMessage::Md5Password { salt } => {
                        buf.put_i32(5);
                        buf.put_slice(salt);
                    }
                    AuthenticationMessage::SaslMechanisms(mechanisms) => {
                        buf.put_i32(10);
                        for mechanism in mechanisms {
                            put_cstring(buf, mechanism);
                        }
                        buf.put_u8(0); // terminator
                    }
                    AuthenticationMessage::SaslContinue(data) => {
                        buf.put_i32(11);
                        buf.put_slice(data);
                    }
                    AuthenticationMessage::SaslFinal(data) => {
                        buf.put_i32(12);
                        buf.put_slice(data);
                    }
                }

                patch_length(buf, len_pos);
            }

            BackendMessage::ParameterStatus { name, value } => {
                buf.put_u8(b'S');
                let len_pos = buf.len();
                buf.put_i32(0);
                put_cstring(buf, name);
                put_cstring(buf, value);
                patch_length(buf, len_pos);
            }

            BackendMessage::BackendKeyData {
                process_id,
                secret_key,
            } => {
                buf.put_u8(b'K');
                buf.put_i32(12); // fixed length: 4 (self) + 4 + 4
                buf.put_i32(*process_id);
                buf.put_i32(*secret_key);
            }

            BackendMessage::ReadyForQuery(state) => {
                buf.put_u8(b'Z');
                buf.put_i32(5); // fixed length: 4 (self) + 1
                buf.put_u8(state.as_byte());
            }

            BackendMessage::RowDescription(fields) => {
                buf.put_u8(b'T');
                let len_pos = buf.len();
                buf.put_i32(0);
                buf.put_i16(fields.len() as i16);
                for field in fields {
                    put_cstring(buf, &field.name);
                    buf.put_i32(field.table_oid);
                    buf.put_i16(field.column_attr);
                    buf.put_i32(field.type_oid);
                    buf.put_i16(field.type_size);
                    buf.put_i32(field.type_modifier);
                    buf.put_i16(field.format);
                }
                patch_length(buf, len_pos);
            }

            BackendMessage::DataRow(values) => {
                buf.put_u8(b'D');
                let len_pos = buf.len();
                buf.put_i32(0);
                buf.put_i16(values.len() as i16);
                for value in values {
                    match value {
                        None => buf.put_i32(-1),
                        Some(data) => {
                            buf.put_i32(data.len() as i32);
                            buf.put_slice(data);
                        }
                    }
                }
                patch_length(buf, len_pos);
            }

            BackendMessage::CommandComplete { tag } => {
                buf.put_u8(b'C');
                let len_pos = buf.len();
                buf.put_i32(0);
                put_cstring(buf, tag);
                patch_length(buf, len_pos);
            }

            BackendMessage::ErrorResponse(fields) => {
                encode_error_notice(buf, b'E', fields);
            }

            BackendMessage::NoticeResponse(fields) => {
                encode_error_notice(buf, b'N', fields);
            }

            BackendMessage::ParseComplete => {
                buf.put_u8(b'1');
                buf.put_i32(4);
            }

            BackendMessage::BindComplete => {
                buf.put_u8(b'2');
                buf.put_i32(4);
            }

            BackendMessage::CloseComplete => {
                buf.put_u8(b'3');
                buf.put_i32(4);
            }

            BackendMessage::PortalSuspended => {
                buf.put_u8(b's');
                buf.put_i32(4);
            }

            BackendMessage::NoData => {
                buf.put_u8(b'n');
                buf.put_i32(4);
            }

            BackendMessage::ParameterDescription(types) => {
                buf.put_u8(b't');
                let len_pos = buf.len();
                buf.put_i32(0);
                buf.put_i16(types.len() as i16);
                for oid in types {
                    buf.put_i32(*oid);
                }
                patch_length(buf, len_pos);
            }

            BackendMessage::EmptyQueryResponse => {
                buf.put_u8(b'I');
                buf.put_i32(4);
            }

            BackendMessage::CopyInResponse {
                format,
                column_formats,
            } => {
                buf.put_u8(b'G');
                let len_pos = buf.len();
                buf.put_i32(0);
                buf.put_i8(*format);
                buf.put_i16(column_formats.len() as i16);
                for f in column_formats {
                    buf.put_i16(*f);
                }
                patch_length(buf, len_pos);
            }

            BackendMessage::CopyOutResponse {
                format,
                column_formats,
            } => {
                buf.put_u8(b'H');
                let len_pos = buf.len();
                buf.put_i32(0);
                buf.put_i8(*format);
                buf.put_i16(column_formats.len() as i16);
                for f in column_formats {
                    buf.put_i16(*f);
                }
                patch_length(buf, len_pos);
            }

            BackendMessage::CopyData(data) => {
                buf.put_u8(b'd');
                buf.put_i32(4 + data.len() as i32);
                buf.put_slice(data);
            }

            BackendMessage::CopyDone => {
                buf.put_u8(b'c');
                buf.put_i32(4);
            }

            BackendMessage::ChangeBatch(msg) => msg.encode(buf),
            BackendMessage::SubscriptionStatus(msg) => msg.encode(buf),
            BackendMessage::SchemaUpdate(msg) => msg.encode(buf),
            BackendMessage::SubscribeOk(msg) => msg.encode(buf),
        }
    }
}

impl ChangeBatchMessage {
    /// Encodes a full X message including type byte and length prefix.
    pub fn encode(&self, buf: &mut BytesMut) {
        buf.put_u8(CHANGE_BATCH_MSG_TYPE);
        let len_pos = buf.len();
        buf.put_i32(0);
        buf.put_u64(self.start_lsn);
        buf.put_u64(self.end_lsn);
        buf.put_u32(self.row_count);
        buf.put_i64(self.commit_timestamp_us);
        for row in &self.rows {
            buf.put_u8(row.change_type);
            buf.put_u32(row.table_id);
            buf.put_u64(row.lsn);
            buf.put_u32(row.row_bytes.len() as u32);
            buf.put_slice(&row.row_bytes);
            buf.put_u32(row.primary_key_bytes.len() as u32);
            buf.put_slice(&row.primary_key_bytes);
        }
        patch_length(buf, len_pos);
    }

    /// Decodes the body after type byte and length have been stripped.
    pub fn decode(payload: &mut BytesMut) -> Result<Self, ProtocolError> {
        ensure_remaining(payload, 8 + 8 + 4 + 8)?;
        let start_lsn = payload.get_u64();
        let end_lsn = payload.get_u64();
        let row_count = payload.get_u32();
        let commit_timestamp_us = payload.get_i64();
        let mut rows = Vec::with_capacity(row_count as usize);
        for _ in 0..row_count {
            ensure_remaining(payload, 1 + 4 + 8 + 4)?;
            let change_type = payload.get_u8();
            let table_id = payload.get_u32();
            let lsn = payload.get_u64();
            let row_len = payload.get_u32() as usize;
            ensure_remaining(payload, row_len + 4)?;
            let row_bytes = payload.split_to(row_len).to_vec();
            let pk_len = payload.get_u32() as usize;
            ensure_remaining(payload, pk_len)?;
            let primary_key_bytes = payload.split_to(pk_len).to_vec();
            rows.push(RowDelta {
                change_type,
                table_id,
                lsn,
                row_bytes,
                primary_key_bytes,
            });
        }
        Ok(ChangeBatchMessage {
            start_lsn,
            end_lsn,
            row_count,
            rows,
            commit_timestamp_us,
        })
    }
}

impl SubscriptionStatusMessage {
    /// Encodes a full Q message. Fixed 20-byte payload.
    pub fn encode(&self, buf: &mut BytesMut) {
        buf.put_u8(SUBSCRIPTION_STATUS_MSG_TYPE);
        buf.put_i32(20);
        buf.put_u64(self.committed_lsn);
        buf.put_i64(self.producer_now_us);
    }

    /// Decodes the body after type byte and length have been stripped.
    pub fn decode(payload: &mut BytesMut) -> Result<Self, ProtocolError> {
        ensure_remaining(payload, 16)?;
        let committed_lsn = payload.get_u64();
        let producer_now_us = payload.get_i64();
        Ok(SubscriptionStatusMessage {
            committed_lsn,
            producer_now_us,
        })
    }
}

impl SchemaUpdateMessage {
    /// Encodes a full v message including type byte and length prefix.
    pub fn encode(&self, buf: &mut BytesMut) {
        buf.put_u8(SCHEMA_UPDATE_MSG_TYPE);
        let len_pos = buf.len();
        buf.put_i32(0);
        put_cstring(buf, &self.publication);
        buf.put_slice(&self.new_fingerprint);
        buf.put_u32(self.columns.len() as u32);
        for col in &self.columns {
            encode_column(buf, col);
        }
        patch_length(buf, len_pos);
    }

    /// Decodes the body after type byte and length have been stripped.
    pub fn decode(payload: &mut BytesMut) -> Result<Self, ProtocolError> {
        let publication = read_cstring(payload)?;
        ensure_remaining(payload, 32 + 4)?;
        let mut new_fingerprint = [0u8; 32];
        payload.copy_to_slice(&mut new_fingerprint);
        let col_count = payload.get_u32();
        let mut columns = Vec::with_capacity(col_count as usize);
        for _ in 0..col_count {
            columns.push(decode_column(payload)?);
        }
        Ok(SchemaUpdateMessage {
            publication,
            new_fingerprint,
            columns,
        })
    }
}

impl SubscribeOkMessage {
    /// Encodes a full K message including type byte and length prefix.
    pub fn encode(&self, buf: &mut BytesMut) {
        buf.put_u8(SUBSCRIBE_OK_MSG_TYPE);
        let len_pos = buf.len();
        buf.put_i32(0);
        buf.put_slice(&self.schema_fingerprint);
        buf.put_u32(self.columns.len() as u32);
        for col in &self.columns {
            encode_column(buf, col);
        }
        buf.put_u64(self.resumed_at_lsn);
        buf.put_u32(self.features);
        patch_length(buf, len_pos);
    }

    /// Decodes the body after type byte and length have been stripped.
    pub fn decode(payload: &mut BytesMut) -> Result<Self, ProtocolError> {
        ensure_remaining(payload, 32 + 4)?;
        let mut schema_fingerprint = [0u8; 32];
        payload.copy_to_slice(&mut schema_fingerprint);
        let col_count = payload.get_u32();
        let mut columns = Vec::with_capacity(col_count as usize);
        for _ in 0..col_count {
            columns.push(decode_column(payload)?);
        }
        ensure_remaining(payload, 12)?;
        let resumed_at_lsn = payload.get_u64();
        let features = payload.get_u32();
        Ok(SubscribeOkMessage {
            schema_fingerprint,
            columns,
            resumed_at_lsn,
            features,
        })
    }
}

fn encode_column(buf: &mut BytesMut, col: &PublishedColumn) {
    put_cstring(buf, &col.name);
    buf.put_u8(col.type_id);
    buf.put_u8(u8::from(col.nullable));
    buf.put_u16(col.ordinal);
}

fn decode_column(payload: &mut BytesMut) -> Result<PublishedColumn, ProtocolError> {
    let name = read_cstring(payload)?;
    ensure_remaining(payload, 1 + 1 + 2)?;
    let type_id = payload.get_u8();
    let nullable = payload.get_u8() != 0;
    let ordinal = payload.get_u16();
    Ok(PublishedColumn {
        name,
        type_id,
        nullable,
        ordinal,
    })
}

fn ensure_remaining(buf: &BytesMut, needed: usize) -> Result<(), ProtocolError> {
    if buf.remaining() < needed {
        return Err(ProtocolError::Malformed(format!(
            "truncated backend message, need {} more bytes",
            needed
        )));
    }
    Ok(())
}

fn read_cstring(buf: &mut BytesMut) -> Result<String, ProtocolError> {
    let bytes = buf.as_ref();
    let null_pos = memchr::memchr(0, bytes)
        .ok_or_else(|| ProtocolError::Malformed("missing null terminator".into()))?;
    let s = std::str::from_utf8(&bytes[..null_pos])
        .map_err(|e| ProtocolError::Malformed(format!("invalid UTF-8: {}", e)))?
        .to_string();
    buf.advance(null_pos + 1);
    Ok(s)
}

impl BackendMessage {
    /// Decodes a backend message restricted to the subscription-mode
    /// protocol. Accepts X, Q, v, K, and ErrorResponse only.
    pub fn decode_subscription(
        msg_type: u8,
        payload: &mut BytesMut,
    ) -> Result<Self, ProtocolError> {
        match msg_type {
            CHANGE_BATCH_MSG_TYPE => Ok(BackendMessage::ChangeBatch(ChangeBatchMessage::decode(
                payload,
            )?)),
            SUBSCRIPTION_STATUS_MSG_TYPE => Ok(BackendMessage::SubscriptionStatus(
                SubscriptionStatusMessage::decode(payload)?,
            )),
            SCHEMA_UPDATE_MSG_TYPE => Ok(BackendMessage::SchemaUpdate(
                SchemaUpdateMessage::decode(payload)?,
            )),
            SUBSCRIBE_OK_MSG_TYPE => Ok(BackendMessage::SubscribeOk(SubscribeOkMessage::decode(
                payload,
            )?)),
            b'E' => {
                let fields = decode_error_response(payload)?;
                Ok(BackendMessage::ErrorResponse(fields))
            }
            _ => Err(ProtocolError::InvalidMessageType(msg_type)),
        }
    }
}

/// Parses an ErrorResponse body. Used during subscription-mode dispatch so
/// the consumer can surface producer-side failures as typed errors.
fn decode_error_response(payload: &mut BytesMut) -> Result<ErrorFields, ProtocolError> {
    let mut severity = String::new();
    let mut code = String::new();
    let mut message = String::new();
    let mut detail = None;
    let mut hint = None;
    let mut position = None;
    loop {
        ensure_remaining(payload, 1)?;
        let tag = payload.get_u8();
        if tag == 0 {
            break;
        }
        let value = read_cstring(payload)?;
        match tag {
            b'S' => severity = value,
            b'V' => {
                if severity.is_empty() {
                    severity = value;
                }
            }
            b'C' => code = value,
            b'M' => message = value,
            b'D' => detail = Some(value),
            b'H' => hint = Some(value),
            b'P' => {
                if let Ok(n) = value.parse::<i32>() {
                    position = Some(n);
                }
            }
            _ => {}
        }
    }
    Ok(ErrorFields {
        severity,
        code,
        message,
        detail,
        hint,
        position,
    })
}

/// Encodes an ErrorResponse or NoticeResponse message.
fn encode_error_notice(buf: &mut BytesMut, msg_type: u8, fields: &ErrorFields) {
    buf.put_u8(msg_type);
    let len_pos = buf.len();
    buf.put_i32(0);

    // Severity (S)
    buf.put_u8(b'S');
    put_cstring(buf, &fields.severity);

    // Severity non-localized (V) - same as S for compatibility
    buf.put_u8(b'V');
    put_cstring(buf, &fields.severity);

    // SQLSTATE code (C)
    buf.put_u8(b'C');
    put_cstring(buf, &fields.code);

    // Message (M)
    buf.put_u8(b'M');
    put_cstring(buf, &fields.message);

    // Detail (D)
    if let Some(detail) = &fields.detail {
        buf.put_u8(b'D');
        put_cstring(buf, detail);
    }

    // Hint (H)
    if let Some(hint) = &fields.hint {
        buf.put_u8(b'H');
        put_cstring(buf, hint);
    }

    // Position (P)
    if let Some(position) = fields.position {
        buf.put_u8(b'P');
        put_cstring(buf, &position.to_string());
    }

    // Terminator
    buf.put_u8(0);

    patch_length(buf, len_pos);
}

/// Writes a null-terminated string to the buffer.
fn put_cstring(buf: &mut BytesMut, s: &str) {
    buf.put_slice(s.as_bytes());
    buf.put_u8(0);
}

/// Patches the 4-byte length field at `len_pos` with the actual length.
/// The length includes itself (4 bytes) but not the type byte.
fn patch_length(buf: &mut BytesMut, len_pos: usize) {
    let total = (buf.len() - len_pos) as i32;
    buf[len_pos..len_pos + 4].copy_from_slice(&total.to_be_bytes());
}

#[cfg(test)]
mod tests {
    use super::*;
    use bytes::Buf;

    fn decode_header(buf: &mut BytesMut) -> (u8, i32) {
        let msg_type = buf.get_u8();
        let length = buf.get_i32();
        (msg_type, length)
    }

    #[test]
    fn test_encode_authentication_ok() {
        let mut buf = BytesMut::new();
        BackendMessage::Authentication(AuthenticationMessage::Ok).encode(&mut buf);

        let (t, len) = decode_header(&mut buf);
        assert_eq!(t, b'R');
        assert_eq!(len, 8); // 4 (self) + 4 (auth type)
        assert_eq!(buf.get_i32(), 0); // AuthenticationOk
    }

    #[test]
    fn test_encode_authentication_cleartext() {
        let mut buf = BytesMut::new();
        BackendMessage::Authentication(AuthenticationMessage::CleartextPassword).encode(&mut buf);

        let (t, _) = decode_header(&mut buf);
        assert_eq!(t, b'R');
        assert_eq!(buf.get_i32(), 3); // CleartextPassword
    }

    #[test]
    fn test_encode_authentication_md5() {
        let mut buf = BytesMut::new();
        let salt = [0x01, 0x02, 0x03, 0x04];
        BackendMessage::Authentication(AuthenticationMessage::Md5Password { salt })
            .encode(&mut buf);

        let (t, len) = decode_header(&mut buf);
        assert_eq!(t, b'R');
        assert_eq!(len, 12); // 4 (self) + 4 (auth type) + 4 (salt)
        assert_eq!(buf.get_i32(), 5); // MD5
        let mut salt_buf = [0u8; 4];
        buf.copy_to_slice(&mut salt_buf);
        assert_eq!(salt_buf, salt);
    }

    #[test]
    fn test_encode_parameter_status() {
        let mut buf = BytesMut::new();
        BackendMessage::ParameterStatus {
            name: "server_version".into(),
            value: "16.0".into(),
        }
        .encode(&mut buf);

        let (t, _) = decode_header(&mut buf);
        assert_eq!(t, b'S');
    }

    #[test]
    fn test_encode_ready_for_query() {
        for (state, expected_byte) in [
            (TransactionState::Idle, b'I'),
            (TransactionState::InTransaction, b'T'),
            (TransactionState::Failed, b'E'),
        ] {
            let mut buf = BytesMut::new();
            BackendMessage::ReadyForQuery(state).encode(&mut buf);

            let (t, len) = decode_header(&mut buf);
            assert_eq!(t, b'Z');
            assert_eq!(len, 5);
            assert_eq!(buf.get_u8(), expected_byte);
        }
    }

    #[test]
    fn test_encode_row_description() {
        let mut buf = BytesMut::new();
        BackendMessage::RowDescription(vec![FieldDescription {
            name: "id".into(),
            table_oid: 0,
            column_attr: 0,
            type_oid: 23,
            type_size: 4,
            type_modifier: -1,
            format: 0,
        }])
        .encode(&mut buf);

        let (t, _) = decode_header(&mut buf);
        assert_eq!(t, b'T');
        assert_eq!(buf.get_i16(), 1); // 1 field
    }

    #[test]
    fn test_encode_data_row() {
        let mut buf = BytesMut::new();
        BackendMessage::DataRow(vec![Some(b"hello".to_vec()), None, Some(b"42".to_vec())])
            .encode(&mut buf);

        let (t, _) = decode_header(&mut buf);
        assert_eq!(t, b'D');
        assert_eq!(buf.get_i16(), 3); // 3 columns
        // Column 1: "hello"
        assert_eq!(buf.get_i32(), 5);
        let mut col1 = vec![0u8; 5];
        buf.copy_to_slice(&mut col1);
        assert_eq!(&col1, b"hello");
        // Column 2: NULL
        assert_eq!(buf.get_i32(), -1);
        // Column 3: "42"
        assert_eq!(buf.get_i32(), 2);
        let mut col3 = vec![0u8; 2];
        buf.copy_to_slice(&mut col3);
        assert_eq!(&col3, b"42");
    }

    #[test]
    fn test_encode_command_complete() {
        let mut buf = BytesMut::new();
        BackendMessage::CommandComplete {
            tag: "SELECT 5".into(),
        }
        .encode(&mut buf);

        let (t, _) = decode_header(&mut buf);
        assert_eq!(t, b'C');
    }

    #[test]
    fn test_encode_error_response() {
        let mut buf = BytesMut::new();
        BackendMessage::ErrorResponse(ErrorFields {
            severity: "ERROR".into(),
            code: "42601".into(),
            message: "syntax error".into(),
            detail: Some("near token X".into()),
            hint: None,
            position: Some(15),
        })
        .encode(&mut buf);

        let (t, _) = decode_header(&mut buf);
        assert_eq!(t, b'E');
        // Verify the buffer contains the expected field codes
        let data = buf.to_vec();
        assert!(data.contains(&b'S')); // severity
        assert!(data.contains(&b'C')); // code
        assert!(data.contains(&b'M')); // message
        assert!(data.contains(&b'D')); // detail
        assert!(data.contains(&b'P')); // position
        assert_eq!(*data.last().unwrap(), 0); // terminator
    }

    #[test]
    fn test_encode_notice_response() {
        let mut buf = BytesMut::new();
        BackendMessage::NoticeResponse(ErrorFields {
            severity: "WARNING".into(),
            code: "01000".into(),
            message: "something happened".into(),
            detail: None,
            hint: None,
            position: None,
        })
        .encode(&mut buf);

        let (t, _) = decode_header(&mut buf);
        assert_eq!(t, b'N');
    }

    #[test]
    fn test_encode_parse_complete() {
        let mut buf = BytesMut::new();
        BackendMessage::ParseComplete.encode(&mut buf);
        let (t, len) = decode_header(&mut buf);
        assert_eq!(t, b'1');
        assert_eq!(len, 4);
    }

    #[test]
    fn test_encode_bind_complete() {
        let mut buf = BytesMut::new();
        BackendMessage::BindComplete.encode(&mut buf);
        let (t, len) = decode_header(&mut buf);
        assert_eq!(t, b'2');
        assert_eq!(len, 4);
    }

    #[test]
    fn test_encode_close_complete() {
        let mut buf = BytesMut::new();
        BackendMessage::CloseComplete.encode(&mut buf);
        let (t, len) = decode_header(&mut buf);
        assert_eq!(t, b'3');
        assert_eq!(len, 4);
    }

    #[test]
    fn test_encode_no_data() {
        let mut buf = BytesMut::new();
        BackendMessage::NoData.encode(&mut buf);
        let (t, len) = decode_header(&mut buf);
        assert_eq!(t, b'n');
        assert_eq!(len, 4);
    }

    #[test]
    fn test_encode_empty_query_response() {
        let mut buf = BytesMut::new();
        BackendMessage::EmptyQueryResponse.encode(&mut buf);
        let (t, len) = decode_header(&mut buf);
        assert_eq!(t, b'I');
        assert_eq!(len, 4);
    }

    #[test]
    fn test_encode_portal_suspended() {
        let mut buf = BytesMut::new();
        BackendMessage::PortalSuspended.encode(&mut buf);
        let (t, len) = decode_header(&mut buf);
        assert_eq!(t, b's');
        assert_eq!(len, 4);
    }

    #[test]
    fn test_encode_parameter_description() {
        let mut buf = BytesMut::new();
        BackendMessage::ParameterDescription(vec![23, 25]).encode(&mut buf);
        let (t, _) = decode_header(&mut buf);
        assert_eq!(t, b't');
        assert_eq!(buf.get_i16(), 2);
        assert_eq!(buf.get_i32(), 23);
        assert_eq!(buf.get_i32(), 25);
    }

    #[test]
    fn test_encode_backend_key_data() {
        let mut buf = BytesMut::new();
        BackendMessage::BackendKeyData {
            process_id: 1234,
            secret_key: 5678,
        }
        .encode(&mut buf);

        let (t, len) = decode_header(&mut buf);
        assert_eq!(t, b'K');
        assert_eq!(len, 12);
        assert_eq!(buf.get_i32(), 1234);
        assert_eq!(buf.get_i32(), 5678);
    }

    #[test]
    fn test_encode_copy_in_response() {
        let mut buf = BytesMut::new();
        BackendMessage::CopyInResponse {
            format: 0,
            column_formats: vec![0, 0],
        }
        .encode(&mut buf);

        let (t, _) = decode_header(&mut buf);
        assert_eq!(t, b'G');
    }

    #[test]
    fn test_encode_copy_out_response() {
        let mut buf = BytesMut::new();
        BackendMessage::CopyOutResponse {
            format: 0,
            column_formats: vec![0],
        }
        .encode(&mut buf);

        let (t, _) = decode_header(&mut buf);
        assert_eq!(t, b'H');
    }

    #[test]
    fn test_encode_copy_data() {
        let mut buf = BytesMut::new();
        BackendMessage::CopyData(b"row\tdata\n".to_vec()).encode(&mut buf);

        let (t, len) = decode_header(&mut buf);
        assert_eq!(t, b'd');
        assert_eq!(len, 4 + 9);
    }

    #[test]
    fn test_encode_copy_done() {
        let mut buf = BytesMut::new();
        BackendMessage::CopyDone.encode(&mut buf);
        let (t, len) = decode_header(&mut buf);
        assert_eq!(t, b'c');
        assert_eq!(len, 4);
    }

    #[test]
    fn test_encode_sasl_mechanisms() {
        let mut buf = BytesMut::new();
        BackendMessage::Authentication(AuthenticationMessage::SaslMechanisms(vec![
            "SCRAM-SHA-256".into(),
        ]))
        .encode(&mut buf);

        let (t, _) = decode_header(&mut buf);
        assert_eq!(t, b'R');
        assert_eq!(buf.get_i32(), 10); // SASL
    }
}
