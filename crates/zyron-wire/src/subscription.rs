//! Wire-level push subscription state machine, flow-control bookkeeping,
//! producer emission loop, and consumer receive loop.
//!
//! The protocol runs on top of the shared PG wire framing once a connection
//! has transitioned into subscription mode by sending Y. In that mode the
//! producer may send X, Q, v, K, and ErrorResponse; the consumer may send
//! W, A, and j. The codec itself is symmetric with the pre-subscribe phase
//! except that dispatch uses the narrower `decode_subscription` helpers.

use std::net::SocketAddr;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicI64, AtomicU64, Ordering};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use bytes::{Buf, BytesMut};
use tokio::io::{AsyncRead, AsyncReadExt, AsyncWrite, AsyncWriteExt};

use crate::messages::ProtocolError;
use crate::messages::backend::{
    BackendMessage, ChangeBatchMessage, SchemaUpdateMessage, SubscribeOkMessage,
    SubscriptionStatusMessage,
};
use crate::messages::frontend::{
    EndSubscriptionMessage, FlowControlMessage, FrontendMessage, SubscribeMessage,
    SubscriptionAckMessage,
};

// ----------------------------------------------------------------------------
// Subscription state machine
// ----------------------------------------------------------------------------

/// Lifecycle states for a push subscription from the perspective of either
/// side. Transitions are driven by the exchange of Y, K, X, v, j, and
/// ErrorResponse messages.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SubscriptionState {
    /// No subscription active on the connection.
    Idle,
    /// Consumer has sent Y and is awaiting K.
    Subscribing,
    /// Receiving X batches after a successful K.
    Active,
    /// Producer sent v. Consumer is applying the new schema.
    SchemaUpdating,
    /// Consumer has sent j and is awaiting acknowledgement.
    Ending,
    /// Subscription terminated gracefully.
    Ended,
    /// Subscription terminated with an error.
    Failed,
}

impl SubscriptionState {
    /// Returns true if the state represents an open subscription that can
    /// still produce or consume data.
    pub fn is_open(self) -> bool {
        matches!(
            self,
            SubscriptionState::Subscribing
                | SubscriptionState::Active
                | SubscriptionState::SchemaUpdating
                | SubscriptionState::Ending
        )
    }
}

// ----------------------------------------------------------------------------
// Producer-side context
// ----------------------------------------------------------------------------

/// Per-consumer context tracked by the producer. All counters use relaxed
/// atomics because the producer emission loop owns the stream and updates
/// counters from a single task, while external monitoring threads only read.
#[derive(Debug)]
pub struct SubscriptionServerContext {
    pub subscription_id: u32,
    pub publication_id: u32,
    pub consumer_id: String,
    pub last_acked_lsn: AtomicU64,
    pub last_pushed_lsn: AtomicU64,
    pub credit_remaining_bytes: AtomicI64,
    pub buffered_bytes: AtomicU64,
    pub schema_fingerprint: [u8; 32],
    pub watermark_high: u64,
    pub watermark_low: u64,
    pub peer_addr: SocketAddr,
    pub role_id: u32,
    pub started_at: Instant,
    pub rows_delivered: AtomicU64,
    pub bytes_delivered: AtomicU64,
}

impl SubscriptionServerContext {
    /// Creates a context with the provided starting LSN and watermarks.
    /// `watermark_low` must be strictly less than `watermark_high` or the
    /// backpressure release check can never fire.
    pub fn new(
        subscription_id: u32,
        publication_id: u32,
        consumer_id: String,
        schema_fingerprint: [u8; 32],
        peer_addr: SocketAddr,
        role_id: u32,
        initial_credit: u32,
        resume_lsn: u64,
        watermark_high: u64,
        watermark_low: u64,
    ) -> Self {
        Self {
            subscription_id,
            publication_id,
            consumer_id,
            last_acked_lsn: AtomicU64::new(resume_lsn),
            last_pushed_lsn: AtomicU64::new(resume_lsn),
            credit_remaining_bytes: AtomicI64::new(initial_credit as i64),
            buffered_bytes: AtomicU64::new(0),
            schema_fingerprint,
            watermark_high,
            watermark_low,
            peer_addr,
            role_id,
            started_at: Instant::now(),
            rows_delivered: AtomicU64::new(0),
            bytes_delivered: AtomicU64::new(0),
        }
    }

    /// Returns true if the producer currently has credit and buffer capacity
    /// available to send more bytes.
    pub fn can_send(&self) -> bool {
        self.credit_remaining_bytes.load(Ordering::Acquire) > 0
            && self.buffered_bytes.load(Ordering::Acquire) < self.watermark_high
    }

    /// Applies a W grant. Adds bytes to credit and returns the new total.
    pub fn grant_credit(&self, bytes: u32) -> i64 {
        self.credit_remaining_bytes
            .fetch_add(bytes as i64, Ordering::AcqRel)
            + bytes as i64
    }

    /// Records an A ack for the given LSN. Subtracts the range byte count
    /// from buffered_bytes and advances last_acked_lsn. The caller supplies
    /// the byte count because only the producer knows the batch sizes
    /// originally emitted between the old ack point and the new one.
    pub fn apply_ack(&self, acked_lsn: u64, bytes_released: u64) {
        let prev = self.last_acked_lsn.load(Ordering::Acquire);
        if acked_lsn > prev {
            self.last_acked_lsn.store(acked_lsn, Ordering::Release);
        }
        if bytes_released > 0 {
            let cur = self.buffered_bytes.load(Ordering::Acquire);
            let new = cur.saturating_sub(bytes_released);
            self.buffered_bytes.store(new, Ordering::Release);
        }
    }

    /// Records an emitted batch. Decrements credit and tracks buffered bytes.
    pub fn record_push(&self, end_lsn: u64, encoded_len: u64, row_count: u64) {
        self.last_pushed_lsn.store(end_lsn, Ordering::Release);
        self.credit_remaining_bytes
            .fetch_sub(encoded_len as i64, Ordering::AcqRel);
        self.buffered_bytes.fetch_add(encoded_len, Ordering::AcqRel);
        self.rows_delivered.fetch_add(row_count, Ordering::Relaxed);
        self.bytes_delivered
            .fetch_add(encoded_len, Ordering::Relaxed);
    }
}

/// Lookup table for active subscriptions on a producer process.
#[derive(Debug, Default)]
pub struct PubSubServerState {
    subscriptions:
        parking_lot::RwLock<std::collections::HashMap<u32, Arc<SubscriptionServerContext>>>,
}

impl PubSubServerState {
    pub fn new() -> Self {
        Self {
            subscriptions: parking_lot::RwLock::new(std::collections::HashMap::new()),
        }
    }

    /// Registers a new subscription context.
    pub fn insert(&self, ctx: Arc<SubscriptionServerContext>) {
        let id = ctx.subscription_id;
        self.subscriptions.write().insert(id, ctx);
    }

    /// Fetches a subscription context by id.
    pub fn get(&self, id: u32) -> Option<Arc<SubscriptionServerContext>> {
        self.subscriptions.read().get(&id).cloned()
    }

    /// Removes a subscription context. Returns the removed entry if it was
    /// present.
    pub fn remove(&self, id: u32) -> Option<Arc<SubscriptionServerContext>> {
        self.subscriptions.write().remove(&id)
    }

    /// Returns the current active subscription count.
    pub fn len(&self) -> usize {
        self.subscriptions.read().len()
    }

    /// Returns true if there are no active subscriptions.
    pub fn is_empty(&self) -> bool {
        self.subscriptions.read().is_empty()
    }
}

// ----------------------------------------------------------------------------
// Change source trait for the producer loop
// ----------------------------------------------------------------------------

/// Abstraction over the upstream change data feed. The producer loop calls
/// `next_batch` with a byte budget and a row budget. The implementation
/// returns an already-encoded ChangeBatchMessage or None when no data is
/// available within the poll deadline.
#[async_trait::async_trait]
pub trait ChangeSource: Send + Sync {
    /// Fetches the next batch. `after_lsn` is exclusive; the returned batch
    /// must start at `after_lsn + 1` or later. `max_bytes` and `max_rows`
    /// cap the returned batch size. A None return indicates the source has
    /// no new data within the poll window.
    async fn next_batch(
        &self,
        after_lsn: u64,
        max_bytes: u32,
        max_rows: u32,
    ) -> Result<Option<ChangeBatchMessage>, ProtocolError>;

    /// Returns the producer's current committed LSN for heartbeat reports.
    async fn committed_lsn(&self) -> u64;
}

// ----------------------------------------------------------------------------
// Shared framing helpers
// ----------------------------------------------------------------------------

/// Writes a BytesMut payload to the stream. Used by both producer and
/// consumer loops to flush encoded messages.
async fn write_all<S: AsyncWrite + Unpin>(
    stream: &mut S,
    data: &[u8],
) -> Result<(), ProtocolError> {
    stream.write_all(data).await.map_err(ProtocolError::Io)?;
    stream.flush().await.map_err(ProtocolError::Io)?;
    Ok(())
}

/// Reads a single subscription-mode frame from the stream. Returns the raw
/// type byte and the payload bytes with the length prefix stripped. The
/// caller dispatches the payload to the appropriate decode routine.
async fn read_frame<S: AsyncRead + Unpin>(
    stream: &mut S,
    buffer: &mut BytesMut,
) -> Result<(u8, BytesMut), ProtocolError> {
    while buffer.len() < 5 {
        let mut chunk = [0u8; 4096];
        let n = stream.read(&mut chunk).await.map_err(ProtocolError::Io)?;
        if n == 0 {
            return Err(ProtocolError::ConnectionClosed);
        }
        buffer.extend_from_slice(&chunk[..n]);
    }
    let msg_type = buffer[0];
    let len = i32::from_be_bytes([buffer[1], buffer[2], buffer[3], buffer[4]]) as usize;
    if len < 4 {
        return Err(ProtocolError::Malformed("frame length too small".into()));
    }
    let total = 1 + len;
    while buffer.len() < total {
        let mut chunk = [0u8; 4096];
        let n = stream.read(&mut chunk).await.map_err(ProtocolError::Io)?;
        if n == 0 {
            return Err(ProtocolError::ConnectionClosed);
        }
        buffer.extend_from_slice(&chunk[..n]);
    }
    let mut frame = buffer.split_to(total);
    // Drop type byte plus 4-byte length prefix.
    frame.advance(5);
    Ok((msg_type, frame))
}

// ----------------------------------------------------------------------------
// Producer emission loop
// ----------------------------------------------------------------------------

/// Runtime configuration for the producer side.
#[derive(Debug, Clone)]
pub struct ProducerConfig {
    /// Target batch size hint, bounded by whatever Y requested.
    pub batch_size_hint: u32,
    /// Heartbeat cadence. Q is emitted when at least this long elapses.
    pub heartbeat_interval: Duration,
    /// How long the producer spin-waits for credit or buffer drain before
    /// re-checking shutdown. Kept small so shutdown is responsive.
    pub backpressure_poll: Duration,
    /// Poll interval passed into the change source when no data is ready.
    pub source_poll: Duration,
}

impl Default for ProducerConfig {
    fn default() -> Self {
        Self {
            batch_size_hint: 512,
            heartbeat_interval: Duration::from_secs(10),
            backpressure_poll: Duration::from_millis(5),
            source_poll: Duration::from_millis(10),
        }
    }
}

/// Drives a subscribed consumer by polling the change source and writing X
/// frames. The loop also writes Q heartbeats on the configured interval.
/// W and A incoming messages are handled via the returned
/// `ProducerInbound` helpers; the caller is expected to run a separate task
/// that reads the inbound stream and feeds decoded messages back in.
pub async fn drive_subscription<S>(
    stream: &mut S,
    ctx: &SubscriptionServerContext,
    source: &dyn ChangeSource,
    cfg: &ProducerConfig,
    shutdown: Arc<AtomicBool>,
) -> Result<(), ProtocolError>
where
    S: AsyncWrite + Unpin,
{
    let mut last_heartbeat = Instant::now();
    let mut encoded = BytesMut::with_capacity(8192);

    while !shutdown.load(Ordering::Acquire) {
        if !ctx.can_send() {
            tokio::time::sleep(cfg.backpressure_poll).await;
            if last_heartbeat.elapsed() >= cfg.heartbeat_interval {
                send_heartbeat(stream, source, &mut encoded).await?;
                last_heartbeat = Instant::now();
            }
            continue;
        }

        let credit = ctx.credit_remaining_bytes.load(Ordering::Acquire);
        let cap_bytes = credit.clamp(0, u32::MAX as i64) as u32;
        let after = ctx.last_pushed_lsn.load(Ordering::Acquire);

        match source
            .next_batch(after, cap_bytes, cfg.batch_size_hint)
            .await?
        {
            Some(batch) => {
                encoded.clear();
                batch.encode(&mut encoded);
                let encoded_len = encoded.len() as u64;
                let row_count = batch.row_count as u64;
                let end_lsn = batch.end_lsn;
                write_all(stream, &encoded).await?;
                ctx.record_push(end_lsn, encoded_len, row_count);
            }
            None => {
                tokio::time::sleep(cfg.source_poll).await;
            }
        }

        if last_heartbeat.elapsed() >= cfg.heartbeat_interval {
            send_heartbeat(stream, source, &mut encoded).await?;
            last_heartbeat = Instant::now();
        }
    }

    Ok(())
}

async fn send_heartbeat<S: AsyncWrite + Unpin>(
    stream: &mut S,
    source: &dyn ChangeSource,
    scratch: &mut BytesMut,
) -> Result<(), ProtocolError> {
    let committed_lsn = source.committed_lsn().await;
    let now_us = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_micros() as i64)
        .unwrap_or(0);
    let msg = SubscriptionStatusMessage {
        committed_lsn,
        producer_now_us: now_us,
    };
    scratch.clear();
    msg.encode(scratch);
    write_all(stream, scratch).await
}

// ----------------------------------------------------------------------------
// Producer inbound handling
// ----------------------------------------------------------------------------

/// Reads the next inbound message on the producer side (W, A, or j) and
/// applies its effect to the supplied context. Returns true when the
/// consumer has requested graceful end via j.
pub async fn handle_inbound_once<S: AsyncRead + Unpin>(
    stream: &mut S,
    buffer: &mut BytesMut,
    ctx: &SubscriptionServerContext,
    bytes_between_acks: &(dyn Fn(u64, u64) -> u64 + Send + Sync),
) -> Result<ProducerInboundOutcome, ProtocolError> {
    let (msg_type, mut payload) = read_frame(stream, buffer).await?;
    match FrontendMessage::decode_subscription(msg_type, &mut payload)? {
        FrontendMessage::FlowControl(FlowControlMessage { credit_bytes }) => {
            ctx.grant_credit(credit_bytes);
            Ok(ProducerInboundOutcome::Credit(credit_bytes))
        }
        FrontendMessage::SubscriptionAck(SubscriptionAckMessage { acked_lsn }) => {
            let prev = ctx.last_acked_lsn.load(Ordering::Acquire);
            let released = if acked_lsn > prev {
                bytes_between_acks(prev, acked_lsn)
            } else {
                0
            };
            ctx.apply_ack(acked_lsn, released);
            Ok(ProducerInboundOutcome::Ack {
                acked_lsn,
                bytes_released: released,
            })
        }
        FrontendMessage::EndSubscription(EndSubscriptionMessage { final_lsn }) => {
            Ok(ProducerInboundOutcome::End { final_lsn })
        }
        other => Err(ProtocolError::Malformed(format!(
            "unexpected frontend message during subscription: {:?}",
            other
        ))),
    }
}

/// Result of a single inbound producer-side handler call.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProducerInboundOutcome {
    Credit(u32),
    Ack { acked_lsn: u64, bytes_released: u64 },
    End { final_lsn: u64 },
}

// ----------------------------------------------------------------------------
// Consumer-side helpers
// ----------------------------------------------------------------------------

/// Runtime configuration for the consumer side.
#[derive(Debug, Clone)]
pub struct ConsumerConfig {
    pub initial_credit: u32,
    pub credit_refill_threshold: u32,
    pub credit_refill_grant: u32,
    pub consumer_id: String,
    pub publication: String,
    pub from_lsn: u64,
    pub schema_fingerprint_pin: Option<[u8; 32]>,
    pub features: u32,
    pub batch_size_hint: u32,
    pub end_wait: Duration,
}

impl Default for ConsumerConfig {
    fn default() -> Self {
        Self {
            initial_credit: 1 << 20,
            credit_refill_threshold: 256 * 1024,
            credit_refill_grant: 1 << 20,
            consumer_id: String::new(),
            publication: String::new(),
            from_lsn: 0,
            schema_fingerprint_pin: None,
            features: 0,
            batch_size_hint: 512,
            end_wait: Duration::from_secs(5),
        }
    }
}

/// Consumer handle wrapping an established connection after subscribe.
pub struct SubscriptionHandle<S> {
    stream: S,
    buffer: BytesMut,
    state: SubscriptionState,
    schema_fingerprint: [u8; 32],
    columns: Vec<crate::messages::backend::PublishedColumn>,
    resumed_at_lsn: u64,
    features: u32,
}

impl<S: AsyncRead + AsyncWrite + Unpin> SubscriptionHandle<S> {
    /// Sends Y on the given stream and waits for K. On success the returned
    /// handle owns the stream and is in the Active state.
    pub async fn start(mut stream: S, cfg: &ConsumerConfig) -> Result<Self, ProtocolError> {
        let mut buf = BytesMut::with_capacity(256);
        let req = SubscribeMessage {
            publication: cfg.publication.clone(),
            from_lsn: cfg.from_lsn,
            initial_credit: cfg.initial_credit,
            consumer_id: cfg.consumer_id.clone(),
            schema_fingerprint_pin: cfg.schema_fingerprint_pin,
            features: cfg.features,
            batch_size_hint: cfg.batch_size_hint,
        };
        req.encode(&mut buf);
        write_all(&mut stream, &buf).await?;
        buf.clear();

        let mut read_buf = BytesMut::with_capacity(4096);
        let (msg_type, mut payload) = read_frame(&mut stream, &mut read_buf).await?;
        let msg = BackendMessage::decode_subscription(msg_type, &mut payload)?;
        match msg {
            BackendMessage::SubscribeOk(SubscribeOkMessage {
                schema_fingerprint,
                columns,
                resumed_at_lsn,
                features,
            }) => Ok(Self {
                stream,
                buffer: read_buf,
                state: SubscriptionState::Active,
                schema_fingerprint,
                columns,
                resumed_at_lsn,
                features,
            }),
            BackendMessage::ErrorResponse(fields) => Err(ProtocolError::AuthFailed(fields.message)),
            _ => Err(ProtocolError::Malformed(
                "expected SubscribeOk or ErrorResponse".into(),
            )),
        }
    }

    /// Returns the schema fingerprint reported at subscribe time.
    pub fn schema_fingerprint(&self) -> [u8; 32] {
        self.schema_fingerprint
    }

    /// Returns the column list reported at subscribe time.
    pub fn columns(&self) -> &[crate::messages::backend::PublishedColumn] {
        &self.columns
    }

    /// Returns the LSN the producer actually resumed from.
    pub fn resumed_at_lsn(&self) -> u64 {
        self.resumed_at_lsn
    }

    /// Returns server-advertised feature bits from the SubscribeOk reply.
    pub fn features(&self) -> u32 {
        self.features
    }

    /// Returns the current state.
    pub fn state(&self) -> SubscriptionState {
        self.state
    }

    /// Receives the next server-to-client event. Returns None when the
    /// connection ends cleanly.
    pub async fn recv(&mut self) -> Result<Option<ConsumerEvent>, ProtocolError> {
        if !self.state.is_open() {
            return Ok(None);
        }
        let (msg_type, mut payload) = match read_frame(&mut self.stream, &mut self.buffer).await {
            Ok(f) => f,
            Err(ProtocolError::ConnectionClosed) => {
                self.state = SubscriptionState::Ended;
                return Ok(None);
            }
            Err(e) => {
                self.state = SubscriptionState::Failed;
                return Err(e);
            }
        };
        let msg = BackendMessage::decode_subscription(msg_type, &mut payload)?;
        match msg {
            BackendMessage::ChangeBatch(batch) => Ok(Some(ConsumerEvent::Batch(batch))),
            BackendMessage::SubscriptionStatus(status) => Ok(Some(ConsumerEvent::Status(status))),
            BackendMessage::SchemaUpdate(update) => {
                self.state = SubscriptionState::SchemaUpdating;
                self.schema_fingerprint = update.new_fingerprint;
                self.columns = update.columns.clone();
                Ok(Some(ConsumerEvent::Schema(update)))
            }
            BackendMessage::ErrorResponse(fields) => {
                self.state = SubscriptionState::Failed;
                Err(ProtocolError::AuthFailed(fields.message))
            }
            _ => Err(ProtocolError::Malformed(
                "unexpected backend message during subscription".into(),
            )),
        }
    }

    /// Resumes receiving after a schema update has been applied.
    pub fn resume_after_schema(&mut self) {
        if self.state == SubscriptionState::SchemaUpdating {
            self.state = SubscriptionState::Active;
        }
    }

    /// Sends A to advance the ack LSN.
    pub async fn ack(&mut self, lsn: u64) -> Result<(), ProtocolError> {
        let mut buf = BytesMut::with_capacity(16);
        SubscriptionAckMessage { acked_lsn: lsn }.encode(&mut buf);
        write_all(&mut self.stream, &buf).await
    }

    /// Sends W to grant additional credit.
    pub async fn grant_credit(&mut self, bytes: u32) -> Result<(), ProtocolError> {
        let mut buf = BytesMut::with_capacity(12);
        FlowControlMessage {
            credit_bytes: bytes,
        }
        .encode(&mut buf);
        write_all(&mut self.stream, &buf).await
    }

    /// Sends j and transitions the state machine to Ending.
    pub async fn end(&mut self, final_lsn: u64) -> Result<(), ProtocolError> {
        let mut buf = BytesMut::with_capacity(16);
        EndSubscriptionMessage { final_lsn }.encode(&mut buf);
        write_all(&mut self.stream, &buf).await?;
        self.state = SubscriptionState::Ending;
        Ok(())
    }

    /// Consumes the handle and returns the underlying stream.
    pub fn into_inner(self) -> S {
        self.stream
    }
}

/// Events delivered by the consumer receive loop.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ConsumerEvent {
    Batch(ChangeBatchMessage),
    Status(SubscriptionStatusMessage),
    Schema(SchemaUpdateMessage),
}

/// Runs the consumer side against an already-subscribed handle. Each X
/// batch is passed to `on_batch`. The loop issues W credit refills when
/// processed bytes since the last W meet `credit_refill_threshold`.
pub async fn run_subscription_consumer<S, F>(
    handle: &mut SubscriptionHandle<S>,
    cfg: &ConsumerConfig,
    mut on_batch: F,
    shutdown: Arc<AtomicBool>,
) -> Result<u64, ProtocolError>
where
    S: AsyncRead + AsyncWrite + Unpin,
    F: FnMut(&ChangeBatchMessage) -> Result<(), ProtocolError>,
{
    let mut bytes_since_refill: u32 = 0;
    let mut last_lsn = handle.resumed_at_lsn;

    while handle.state.is_open() && !shutdown.load(Ordering::Acquire) {
        match handle.recv().await? {
            Some(ConsumerEvent::Batch(batch)) => {
                on_batch(&batch)?;
                last_lsn = batch.end_lsn;
                handle.ack(batch.end_lsn).await?;
                let approx = approximate_batch_bytes(&batch);
                bytes_since_refill = bytes_since_refill.saturating_add(approx);
                if bytes_since_refill >= cfg.credit_refill_threshold {
                    handle.grant_credit(cfg.credit_refill_grant).await?;
                    bytes_since_refill = 0;
                }
            }
            Some(ConsumerEvent::Status(_)) => {
                // Observability only. No action required here.
            }
            Some(ConsumerEvent::Schema(_)) => {
                handle.resume_after_schema();
            }
            None => break,
        }
    }

    if handle.state == SubscriptionState::Active {
        handle.end(last_lsn).await?;
    }

    Ok(last_lsn)
}

fn approximate_batch_bytes(batch: &ChangeBatchMessage) -> u32 {
    let mut total: u32 = 5 + 8 + 8 + 4 + 8;
    for row in &batch.rows {
        total = total.saturating_add(1 + 4 + 8 + 4 + row.row_bytes.len() as u32);
        total = total.saturating_add(4 + row.primary_key_bytes.len() as u32);
    }
    total
}

// ----------------------------------------------------------------------------
// Producer helpers for encoding K, v, and ErrorResponse
// ----------------------------------------------------------------------------

/// Sends the initial K reply to a Y request.
pub async fn send_subscribe_ok<S: AsyncWrite + Unpin>(
    stream: &mut S,
    msg: &SubscribeOkMessage,
) -> Result<(), ProtocolError> {
    let mut buf = BytesMut::with_capacity(256);
    msg.encode(&mut buf);
    write_all(stream, &buf).await
}

/// Sends a v schema update.
pub async fn send_schema_update<S: AsyncWrite + Unpin>(
    stream: &mut S,
    msg: &SchemaUpdateMessage,
) -> Result<(), ProtocolError> {
    let mut buf = BytesMut::with_capacity(256);
    msg.encode(&mut buf);
    write_all(stream, &buf).await
}

// ----------------------------------------------------------------------------
// Tests
// ----------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::messages::backend::{PublishedColumn, RowDelta};
    use async_trait::async_trait;
    use std::sync::Mutex;
    use std::sync::atomic::AtomicU32;
    use tokio::io::duplex;

    fn fingerprint(tag: u8) -> [u8; 32] {
        [tag; 32]
    }

    fn sample_rows(start: u64, count: u32) -> (u64, u64, Vec<RowDelta>) {
        let mut rows = Vec::with_capacity(count as usize);
        for i in 0..count {
            rows.push(RowDelta {
                change_type: 0,
                table_id: 42,
                lsn: start + i as u64,
                row_bytes: vec![0xAA, 0xBB, i as u8],
                primary_key_bytes: vec![i as u8],
            });
        }
        (start, start + count.saturating_sub(1) as u64, rows)
    }

    #[test]
    fn test_subscribe_message_roundtrip() {
        let msg = SubscribeMessage {
            publication: "orders".into(),
            from_lsn: 42,
            initial_credit: 1024,
            consumer_id: "consumer-a".into(),
            schema_fingerprint_pin: Some(fingerprint(1)),
            features: 0b1011,
            batch_size_hint: 256,
        };
        let mut buf = BytesMut::new();
        msg.encode(&mut buf);

        let msg_type = buf[0];
        let len = i32::from_be_bytes([buf[1], buf[2], buf[3], buf[4]]) as usize;
        assert_eq!(msg_type, crate::messages::frontend::SUBSCRIBE_MSG_TYPE);
        assert_eq!(buf.len(), 1 + len);

        let mut payload = buf.split_off(5);
        buf.clear();
        let decoded = SubscribeMessage::decode(&mut payload).unwrap();
        assert_eq!(decoded, msg);
    }

    #[test]
    fn test_subscribe_message_without_fingerprint() {
        let msg = SubscribeMessage {
            publication: "p".into(),
            from_lsn: 0,
            initial_credit: 1,
            consumer_id: "c".into(),
            schema_fingerprint_pin: None,
            features: 0,
            batch_size_hint: 1,
        };
        let mut buf = BytesMut::new();
        msg.encode(&mut buf);
        let mut payload = buf.split_off(5);
        let decoded = SubscribeMessage::decode(&mut payload).unwrap();
        assert!(decoded.schema_fingerprint_pin.is_none());
    }

    #[test]
    fn test_flow_control_roundtrip() {
        let msg = FlowControlMessage {
            credit_bytes: 65536,
        };
        let mut buf = BytesMut::new();
        msg.encode(&mut buf);
        assert_eq!(buf[0], crate::messages::frontend::FLOW_CONTROL_MSG_TYPE);
        let mut payload = buf.split_off(5);
        let decoded = FlowControlMessage::decode(&mut payload).unwrap();
        assert_eq!(decoded, msg);
    }

    #[test]
    fn test_subscription_ack_roundtrip() {
        let msg = SubscriptionAckMessage {
            acked_lsn: 9_999_999_999,
        };
        let mut buf = BytesMut::new();
        msg.encode(&mut buf);
        let mut payload = buf.split_off(5);
        let decoded = SubscriptionAckMessage::decode(&mut payload).unwrap();
        assert_eq!(decoded, msg);
    }

    #[test]
    fn test_end_subscription_roundtrip() {
        let msg = EndSubscriptionMessage { final_lsn: 777 };
        let mut buf = BytesMut::new();
        msg.encode(&mut buf);
        assert_eq!(buf[0], crate::messages::frontend::END_SUBSCRIPTION_MSG_TYPE);
        let mut payload = buf.split_off(5);
        let decoded = EndSubscriptionMessage::decode(&mut payload).unwrap();
        assert_eq!(decoded, msg);
    }

    #[test]
    fn test_change_batch_roundtrip() {
        let (start, end, rows) = sample_rows(100, 4);
        let msg = ChangeBatchMessage {
            start_lsn: start,
            end_lsn: end,
            row_count: rows.len() as u32,
            rows,
            commit_timestamp_us: 1_700_000_000_000_000,
        };
        let mut buf = BytesMut::new();
        msg.encode(&mut buf);
        assert_eq!(buf[0], crate::messages::backend::CHANGE_BATCH_MSG_TYPE);
        let mut payload = buf.split_off(5);
        let decoded = ChangeBatchMessage::decode(&mut payload).unwrap();
        assert_eq!(decoded, msg);
    }

    #[test]
    fn test_subscription_status_roundtrip() {
        let msg = SubscriptionStatusMessage {
            committed_lsn: 42,
            producer_now_us: 1_700_000_000,
        };
        let mut buf = BytesMut::new();
        msg.encode(&mut buf);
        let mut payload = buf.split_off(5);
        let decoded = SubscriptionStatusMessage::decode(&mut payload).unwrap();
        assert_eq!(decoded, msg);
    }

    #[test]
    fn test_schema_update_roundtrip() {
        let msg = SchemaUpdateMessage {
            publication: "orders".into(),
            new_fingerprint: fingerprint(7),
            columns: vec![
                PublishedColumn {
                    name: "id".into(),
                    type_id: 3,
                    nullable: false,
                    ordinal: 0,
                },
                PublishedColumn {
                    name: "note".into(),
                    type_id: 9,
                    nullable: true,
                    ordinal: 1,
                },
            ],
        };
        let mut buf = BytesMut::new();
        msg.encode(&mut buf);
        let mut payload = buf.split_off(5);
        let decoded = SchemaUpdateMessage::decode(&mut payload).unwrap();
        assert_eq!(decoded, msg);
    }

    #[test]
    fn test_subscribe_ok_roundtrip() {
        let msg = SubscribeOkMessage {
            schema_fingerprint: fingerprint(11),
            columns: vec![PublishedColumn {
                name: "col".into(),
                type_id: 1,
                nullable: false,
                ordinal: 0,
            }],
            resumed_at_lsn: 5,
            features: 0xAA,
        };
        let mut buf = BytesMut::new();
        msg.encode(&mut buf);
        let mut payload = buf.split_off(5);
        let decoded = SubscribeOkMessage::decode(&mut payload).unwrap();
        assert_eq!(decoded, msg);
    }

    #[test]
    fn test_state_transitions() {
        assert!(SubscriptionState::Subscribing.is_open());
        assert!(SubscriptionState::Active.is_open());
        assert!(SubscriptionState::SchemaUpdating.is_open());
        assert!(SubscriptionState::Ending.is_open());
        assert!(!SubscriptionState::Idle.is_open());
        assert!(!SubscriptionState::Ended.is_open());
        assert!(!SubscriptionState::Failed.is_open());
    }

    fn sample_ctx(watermark_high: u64, watermark_low: u64) -> SubscriptionServerContext {
        SubscriptionServerContext::new(
            1,
            2,
            "consumer-a".into(),
            fingerprint(0),
            "127.0.0.1:0".parse().unwrap(),
            9,
            4096,
            0,
            watermark_high,
            watermark_low,
        )
    }

    #[test]
    fn test_credit_accounting_grant_and_consume() {
        let ctx = sample_ctx(1 << 20, 1 << 18);
        assert_eq!(ctx.credit_remaining_bytes.load(Ordering::Acquire), 4096);
        ctx.record_push(10, 1024, 5);
        assert_eq!(ctx.credit_remaining_bytes.load(Ordering::Acquire), 3072);
        assert_eq!(ctx.buffered_bytes.load(Ordering::Acquire), 1024);
        ctx.grant_credit(2048);
        assert_eq!(ctx.credit_remaining_bytes.load(Ordering::Acquire), 5120);
    }

    #[test]
    fn test_backpressure_blocks_when_buffered_exceeds_high_watermark() {
        let ctx = sample_ctx(1024, 512);
        // Still under watermark.
        ctx.record_push(1, 512, 1);
        assert!(ctx.can_send());
        // Push crosses high watermark.
        ctx.record_push(2, 1024, 1);
        assert!(!ctx.can_send());
    }

    #[test]
    fn test_ack_releases_buffered_bytes() {
        let ctx = sample_ctx(1 << 20, 1 << 18);
        ctx.record_push(10, 2048, 4);
        ctx.apply_ack(10, 2048);
        assert_eq!(ctx.buffered_bytes.load(Ordering::Acquire), 0);
        assert_eq!(ctx.last_acked_lsn.load(Ordering::Acquire), 10);
    }

    #[test]
    fn test_pubsub_state_insert_and_remove() {
        let state = PubSubServerState::new();
        let ctx = Arc::new(sample_ctx(1 << 20, 1 << 18));
        state.insert(ctx.clone());
        assert_eq!(state.len(), 1);
        let fetched = state.get(1).unwrap();
        assert_eq!(fetched.subscription_id, 1);
        let removed = state.remove(1).unwrap();
        assert_eq!(removed.subscription_id, 1);
        assert!(state.is_empty());
    }

    /// Change source backed by a preloaded queue for deterministic tests.
    struct MockSource {
        batches: Mutex<std::collections::VecDeque<ChangeBatchMessage>>,
        committed: AtomicU64,
        poll_count: AtomicU32,
    }

    impl MockSource {
        fn new(batches: Vec<ChangeBatchMessage>, committed: u64) -> Self {
            Self {
                batches: Mutex::new(batches.into_iter().collect()),
                committed: AtomicU64::new(committed),
                poll_count: AtomicU32::new(0),
            }
        }
    }

    #[async_trait]
    impl ChangeSource for MockSource {
        async fn next_batch(
            &self,
            _after: u64,
            _max_bytes: u32,
            _max_rows: u32,
        ) -> Result<Option<ChangeBatchMessage>, ProtocolError> {
            self.poll_count.fetch_add(1, Ordering::Relaxed);
            Ok(self.batches.lock().unwrap().pop_front())
        }

        async fn committed_lsn(&self) -> u64 {
            self.committed.load(Ordering::Acquire)
        }
    }

    #[tokio::test]
    async fn test_producer_loop_drains_change_source_and_stops_on_shutdown() {
        let (client, mut server) = duplex(8192);
        let (_, _, rows) = sample_rows(1, 3);
        let batch = ChangeBatchMessage {
            start_lsn: 1,
            end_lsn: 3,
            row_count: rows.len() as u32,
            rows,
            commit_timestamp_us: 0,
        };
        let source = MockSource::new(vec![batch.clone()], 3);
        let ctx = Arc::new(sample_ctx(1 << 20, 1 << 18));
        let shutdown = Arc::new(AtomicBool::new(false));
        let cfg = ProducerConfig {
            batch_size_hint: 16,
            heartbeat_interval: Duration::from_secs(3600),
            backpressure_poll: Duration::from_millis(1),
            source_poll: Duration::from_millis(1),
        };

        let shutdown_c = shutdown.clone();
        let ctx_c = ctx.clone();
        let mut writer = server;
        let producer = tokio::spawn(async move {
            drive_subscription(&mut writer, &ctx_c, &source, &cfg, shutdown_c).await
        });

        // Read exactly one X frame from the client side, then shut down.
        let mut read_buf = BytesMut::with_capacity(4096);
        let mut reader = client;
        let (msg_type, mut payload) = read_frame(&mut reader, &mut read_buf).await.unwrap();
        assert_eq!(msg_type, crate::messages::backend::CHANGE_BATCH_MSG_TYPE);
        let decoded = ChangeBatchMessage::decode(&mut payload).unwrap();
        assert_eq!(decoded, batch);

        shutdown.store(true, Ordering::Release);
        let _ = producer.await.unwrap();
        assert_eq!(ctx.last_pushed_lsn.load(Ordering::Acquire), 3);
    }

    #[tokio::test]
    async fn test_consumer_handshake_and_receive_batch() {
        let (client, mut server) = duplex(8192);

        // Producer-side fixture: read Y, reply K, then push a batch.
        let batch = {
            let (_, _, rows) = sample_rows(1, 2);
            ChangeBatchMessage {
                start_lsn: 1,
                end_lsn: 2,
                row_count: 2,
                rows,
                commit_timestamp_us: 0,
            }
        };
        let batch_c = batch.clone();
        let producer = tokio::spawn(async move {
            let mut rbuf = BytesMut::with_capacity(512);
            let (t, mut p) = read_frame(&mut server, &mut rbuf).await.unwrap();
            assert_eq!(t, crate::messages::frontend::SUBSCRIBE_MSG_TYPE);
            let _ = SubscribeMessage::decode(&mut p).unwrap();
            let ok = SubscribeOkMessage {
                schema_fingerprint: fingerprint(2),
                columns: vec![],
                resumed_at_lsn: 0,
                features: 0,
            };
            send_subscribe_ok(&mut server, &ok).await.unwrap();
            let mut out = BytesMut::new();
            batch_c.encode(&mut out);
            server.write_all(&out).await.unwrap();
            server.flush().await.unwrap();

            // Read ack and end from the consumer.
            let (t2, mut p2) = read_frame(&mut server, &mut rbuf).await.unwrap();
            assert_eq!(t2, crate::messages::frontend::SUBSCRIPTION_ACK_MSG_TYPE);
            let ack = SubscriptionAckMessage::decode(&mut p2).unwrap();
            assert_eq!(ack.acked_lsn, 2);

            let (t3, mut p3) = read_frame(&mut server, &mut rbuf).await.unwrap();
            assert_eq!(t3, crate::messages::frontend::END_SUBSCRIPTION_MSG_TYPE);
            let end = EndSubscriptionMessage::decode(&mut p3).unwrap();
            assert_eq!(end.final_lsn, 2);
        });

        let cfg = ConsumerConfig {
            initial_credit: 1 << 20,
            credit_refill_threshold: 256,
            credit_refill_grant: 1 << 20,
            consumer_id: "c".into(),
            publication: "p".into(),
            from_lsn: 0,
            schema_fingerprint_pin: None,
            features: 0,
            batch_size_hint: 1,
            end_wait: Duration::from_secs(1),
        };

        let mut handle = SubscriptionHandle::start(client, &cfg).await.unwrap();
        assert_eq!(handle.state(), SubscriptionState::Active);
        assert_eq!(handle.schema_fingerprint(), fingerprint(2));

        let shutdown = Arc::new(AtomicBool::new(false));
        let shutdown_c = shutdown.clone();
        let received = Arc::new(AtomicU64::new(0));
        let received_c = received.clone();
        let final_lsn = run_subscription_consumer(
            &mut handle,
            &cfg,
            move |b| {
                received_c.fetch_add(b.row_count as u64, Ordering::Release);
                shutdown_c.store(true, Ordering::Release);
                Ok(())
            },
            shutdown,
        )
        .await;

        assert!(final_lsn.is_ok());
        assert_eq!(final_lsn.unwrap(), 2);
        assert_eq!(received.load(Ordering::Acquire), 2);
        producer.await.unwrap();
    }

    #[tokio::test]
    async fn test_producer_inbound_credit_grant() {
        let (mut client, mut server) = duplex(1024);
        let ctx = Arc::new(sample_ctx(1 << 20, 1 << 18));
        let ctx_c = ctx.clone();
        let server_task = tokio::spawn(async move {
            let mut buf = BytesMut::with_capacity(256);
            let outcome = handle_inbound_once(&mut server, &mut buf, &ctx_c, &|_, _| 0)
                .await
                .unwrap();
            outcome
        });

        let mut out = BytesMut::new();
        FlowControlMessage { credit_bytes: 2048 }.encode(&mut out);
        client.write_all(&out).await.unwrap();
        client.flush().await.unwrap();

        let outcome = server_task.await.unwrap();
        assert_eq!(outcome, ProducerInboundOutcome::Credit(2048));
        assert_eq!(
            ctx.credit_remaining_bytes.load(Ordering::Acquire),
            4096 + 2048
        );
    }

    #[tokio::test]
    async fn test_producer_inbound_end_subscription() {
        let (mut client, mut server) = duplex(1024);
        let ctx = Arc::new(sample_ctx(1 << 20, 1 << 18));
        let ctx_c = ctx.clone();
        let server_task = tokio::spawn(async move {
            let mut buf = BytesMut::with_capacity(256);
            handle_inbound_once(&mut server, &mut buf, &ctx_c, &|_, _| 0)
                .await
                .unwrap()
        });

        let mut out = BytesMut::new();
        EndSubscriptionMessage { final_lsn: 42 }.encode(&mut out);
        client.write_all(&out).await.unwrap();
        client.flush().await.unwrap();

        let outcome = server_task.await.unwrap();
        assert_eq!(outcome, ProducerInboundOutcome::End { final_lsn: 42 });
    }

    #[tokio::test]
    async fn test_producer_inbound_ack_releases_buffered() {
        let (mut client, mut server) = duplex(1024);
        let ctx = Arc::new(sample_ctx(1 << 20, 1 << 18));
        ctx.record_push(10, 4096, 4);
        let ctx_c = ctx.clone();
        let server_task = tokio::spawn(async move {
            let mut buf = BytesMut::with_capacity(256);
            handle_inbound_once(&mut server, &mut buf, &ctx_c, &|_prev, _new| 4096)
                .await
                .unwrap()
        });

        let mut out = BytesMut::new();
        SubscriptionAckMessage { acked_lsn: 10 }.encode(&mut out);
        client.write_all(&out).await.unwrap();
        client.flush().await.unwrap();

        let outcome = server_task.await.unwrap();
        assert_eq!(
            outcome,
            ProducerInboundOutcome::Ack {
                acked_lsn: 10,
                bytes_released: 4096
            }
        );
        assert_eq!(ctx.buffered_bytes.load(Ordering::Acquire), 0);
        assert_eq!(ctx.last_acked_lsn.load(Ordering::Acquire), 10);
    }

    #[tokio::test]
    async fn test_schema_update_transitions_state() {
        let (client, mut server) = duplex(4096);

        let producer = tokio::spawn(async move {
            let mut rbuf = BytesMut::with_capacity(512);
            let (_t, mut p) = read_frame(&mut server, &mut rbuf).await.unwrap();
            let _ = SubscribeMessage::decode(&mut p).unwrap();
            let ok = SubscribeOkMessage {
                schema_fingerprint: fingerprint(1),
                columns: vec![],
                resumed_at_lsn: 0,
                features: 0,
            };
            send_subscribe_ok(&mut server, &ok).await.unwrap();
            let update = SchemaUpdateMessage {
                publication: "p".into(),
                new_fingerprint: fingerprint(9),
                columns: vec![],
            };
            send_schema_update(&mut server, &update).await.unwrap();
        });

        let cfg = ConsumerConfig {
            initial_credit: 16,
            credit_refill_threshold: 1,
            credit_refill_grant: 16,
            consumer_id: "c".into(),
            publication: "p".into(),
            from_lsn: 0,
            schema_fingerprint_pin: None,
            features: 0,
            batch_size_hint: 1,
            end_wait: Duration::from_secs(1),
        };
        let mut handle = SubscriptionHandle::start(client, &cfg).await.unwrap();
        assert_eq!(handle.schema_fingerprint(), fingerprint(1));
        let evt = handle.recv().await.unwrap().unwrap();
        match evt {
            ConsumerEvent::Schema(_) => {}
            _ => panic!("expected schema event"),
        }
        assert_eq!(handle.state(), SubscriptionState::SchemaUpdating);
        assert_eq!(handle.schema_fingerprint(), fingerprint(9));
        handle.resume_after_schema();
        assert_eq!(handle.state(), SubscriptionState::Active);
        producer.await.unwrap();
    }

    #[tokio::test]
    async fn test_error_response_fails_subscribe() {
        let (client, mut server) = duplex(1024);
        let producer = tokio::spawn(async move {
            let mut rbuf = BytesMut::with_capacity(256);
            let (_t, mut p) = read_frame(&mut server, &mut rbuf).await.unwrap();
            let _ = SubscribeMessage::decode(&mut p).unwrap();
            let fields = crate::messages::backend::ErrorFields {
                severity: "ERROR".into(),
                code: "42501".into(),
                message: "no subscribe privilege".into(),
                detail: None,
                hint: None,
                position: None,
            };
            let mut out = BytesMut::new();
            BackendMessage::ErrorResponse(fields).encode(&mut out);
            server.write_all(&out).await.unwrap();
            server.flush().await.unwrap();
        });
        let cfg = ConsumerConfig {
            publication: "p".into(),
            consumer_id: "c".into(),
            ..ConsumerConfig::default()
        };
        let res = SubscriptionHandle::start(client, &cfg).await;
        assert!(matches!(res, Err(ProtocolError::AuthFailed(_))));
        producer.await.unwrap();
    }
}
