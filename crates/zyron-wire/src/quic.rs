//! QUIC transport implementation for the PostgreSQL wire protocol.
//!
//! Bridges tokio-quiche's callback-based ApplicationOverQuic model to
//! the AsyncRead + AsyncWrite interface expected by Connection<T>.
//! Each QUIC connection handles one bidirectional stream (the PG session).
//! Data flows through channels between the quiche worker loop and the
//! QuicStream wrapper.

use std::io;
use std::net::SocketAddr;
use std::pin::Pin;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::task::{Context, Poll};

use bytes::{Bytes, BytesMut};
use tokio::io::{AsyncRead, AsyncWrite, ReadBuf};
use tokio::sync::Notify;
use tokio::sync::mpsc;

use tokio_quiche::quic::{HandshakeInfo, QuicheConnection};
use tokio_quiche::quiche;
use tokio_quiche::{ApplicationOverQuic, QuicResult};

use crate::transport::WireTransport;

/// Channel capacity for read-side bridging (QUIC -> PG wire protocol).
/// 256 slots at up to 65KB per chunk = ~16MB of buffered data before
/// backpressure. Larger than typical PG message bursts.
const READ_CHANNEL_SIZE: usize = 256;

/// Minimum bytes pending before notifying the quiche worker on writes.
/// Batches small writes (parameter status, auth messages) into fewer wakeups.
const WRITE_NOTIFY_THRESHOLD: usize = 1024;

/// Bidirectional byte stream over a QUIC connection.
///
/// Wraps mpsc channels that bridge to the quiche worker loop.
/// The read side receives data from QUIC via a bounded channel.
/// The write side sends data to QUIC via an unbounded channel
/// (backpressure is handled by QUIC's own flow control).
pub struct QuicStream {
    read_rx: mpsc::Receiver<Bytes>,
    write_tx: mpsc::UnboundedSender<Bytes>,
    /// Wakes the quiche worker when write data is available.
    write_notify: Arc<Notify>,
    read_buf: BytesMut,
    peer_addr: SocketAddr,
    /// Tracks pending write bytes since last notify to batch wakeups.
    pending_write_bytes: usize,
    /// Signals the quiche worker that this stream is shutting down.
    closed: Arc<AtomicBool>,
}

impl QuicStream {
    pub fn peer_addr(&self) -> SocketAddr {
        self.peer_addr
    }

    /// Creates a QuicStream from raw channel parts. Used for testing
    /// protocol operations over the channel bridge without a real QUIC connection.
    #[doc(hidden)]
    pub fn from_parts(
        read_rx: mpsc::Receiver<Bytes>,
        write_tx: mpsc::UnboundedSender<Bytes>,
        write_notify: Arc<Notify>,
        peer_addr: SocketAddr,
    ) -> Self {
        Self {
            read_rx,
            write_tx,
            write_notify,
            read_buf: BytesMut::new(),
            peer_addr,
            pending_write_bytes: 0,
            closed: Arc::new(AtomicBool::new(false)),
        }
    }
}

impl WireTransport for QuicStream {
    fn is_encrypted(&self) -> bool {
        true
    }
}

impl AsyncRead for QuicStream {
    fn poll_read(
        mut self: Pin<&mut Self>,
        cx: &mut Context<'_>,
        buf: &mut ReadBuf<'_>,
    ) -> Poll<io::Result<()>> {
        // Drain buffered data first
        if !self.read_buf.is_empty() {
            let n = std::cmp::min(self.read_buf.len(), buf.remaining());
            buf.put_slice(&self.read_buf.split_to(n));
            return Poll::Ready(Ok(()));
        }

        match self.read_rx.poll_recv(cx) {
            Poll::Ready(Some(data)) => {
                let n = std::cmp::min(data.len(), buf.remaining());
                buf.put_slice(&data[..n]);
                if n < data.len() {
                    self.read_buf.extend_from_slice(&data[n..]);
                }
                Poll::Ready(Ok(()))
            }
            Poll::Ready(None) => Poll::Ready(Ok(())),
            Poll::Pending => Poll::Pending,
        }
    }
}

impl AsyncWrite for QuicStream {
    fn poll_write(
        mut self: Pin<&mut Self>,
        _cx: &mut Context<'_>,
        buf: &[u8],
    ) -> Poll<io::Result<usize>> {
        let len = buf.len();
        match self.write_tx.send(Bytes::copy_from_slice(buf)) {
            Ok(()) => {
                self.pending_write_bytes += len;
                // Batch notifications: only wake the quiche worker when enough
                // data has accumulated or a large write comes in.
                if self.pending_write_bytes >= WRITE_NOTIFY_THRESHOLD {
                    self.write_notify.notify_one();
                    self.pending_write_bytes = 0;
                }
                Poll::Ready(Ok(len))
            }
            Err(_) => Poll::Ready(Err(io::Error::new(
                io::ErrorKind::BrokenPipe,
                "QUIC connection closed",
            ))),
        }
    }

    fn poll_flush(mut self: Pin<&mut Self>, _cx: &mut Context<'_>) -> Poll<io::Result<()>> {
        // Flush any pending write notification so the quiche worker picks up
        // all buffered data immediately.
        if self.pending_write_bytes > 0 {
            self.write_notify.notify_one();
            self.pending_write_bytes = 0;
        }
        Poll::Ready(Ok(()))
    }

    fn poll_shutdown(self: Pin<&mut Self>, _cx: &mut Context<'_>) -> Poll<io::Result<()>> {
        // Signal the quiche worker that this stream is closing.
        self.closed.store(true, Ordering::Release);
        self.write_notify.notify_one();
        Poll::Ready(Ok(()))
    }
}

impl Unpin for QuicStream {}

impl Drop for QuicStream {
    fn drop(&mut self) {
        // Ensure the quiche worker is notified on drop even if shutdown was
        // not called explicitly (connection abort, panic).
        self.closed.store(true, Ordering::Release);
        self.write_notify.notify_one();
    }
}

/// Application that bridges a single QUIC bidirectional stream to the PG wire protocol.
///
/// Runs inside the tokio-quiche worker loop. Reads from QUIC streams and pushes
/// data through channels to the QuicStream. Receives write data from the QuicStream
/// and sends it to the QUIC stream.
pub(crate) struct WireProtocolApp {
    buf: Vec<u8>,
    /// Stream ID of the PG session (first client-initiated bidi stream).
    stream_id: Option<u64>,
    /// Sends received data to the QuicStream's read side.
    read_tx: mpsc::Sender<Bytes>,
    /// Receives write data from the QuicStream.
    write_rx: mpsc::UnboundedReceiver<Bytes>,
    /// Shared notify for write wakeups.
    write_notify: Arc<Notify>,
    /// Shared closed flag set by QuicStream on shutdown/drop.
    closed: Arc<AtomicBool>,
    /// Buffered writes that couldn't be sent due to QUIC flow control.
    /// Drained first on the next process_writes cycle to prevent data loss.
    pending_write_buf: std::collections::VecDeque<Bytes>,
    /// Staging buffer for accumulating multiple channel messages into one
    /// stream_send call. Reduces per-message quiche overhead.
    write_staging: BytesMut,
}

impl WireProtocolApp {
    /// Creates a new WireProtocolApp and its associated QuicStream sender.
    ///
    /// The `stream_ready_tx` channel delivers the QuicStream back to the accept
    /// loop once the QUIC handshake completes and a bidi stream is available.
    pub fn new(stream_ready_tx: mpsc::Sender<QuicStream>, peer_addr: SocketAddr) -> Self {
        let (read_tx, read_rx) = mpsc::channel(READ_CHANNEL_SIZE);
        let (write_tx, write_rx) = mpsc::unbounded_channel();
        let write_notify = Arc::new(Notify::new());
        let closed = Arc::new(AtomicBool::new(false));

        let quic_stream = QuicStream {
            read_rx,
            write_tx,
            write_notify: Arc::clone(&write_notify),
            read_buf: BytesMut::new(),
            peer_addr,
            pending_write_bytes: 0,
            closed: Arc::clone(&closed),
        };

        // Send the QuicStream immediately. The connection handler will start
        // reading from it once the QUIC stream actually has data.
        let _ = stream_ready_tx.try_send(quic_stream);

        Self {
            buf: vec![0u8; 65535],
            stream_id: None,
            read_tx,
            write_rx,
            write_notify,
            closed,
            pending_write_buf: std::collections::VecDeque::new(),
            write_staging: BytesMut::with_capacity(65536),
        }
    }
}

impl ApplicationOverQuic for WireProtocolApp {
    fn on_conn_established(
        &mut self,
        _qconn: &mut QuicheConnection,
        _handshake_info: &HandshakeInfo,
    ) -> QuicResult<()> {
        Ok(())
    }

    fn should_act(&self) -> bool {
        // Stop the worker loop if the PG handler has disconnected.
        !self.closed.load(Ordering::Acquire)
    }

    fn buffer(&mut self) -> &mut [u8] {
        &mut self.buf
    }

    async fn wait_for_data(&mut self, _qconn: &mut QuicheConnection) -> QuicResult<()> {
        // Wait for either:
        // 1. The PG handler sending response data through the write channel.
        // 2. A short timeout so the worker loop cycles to process incoming
        //    QUIC packets (reads) that arrived while we were waiting.
        tokio::select! {
            _ = self.write_notify.notified() => {}
            _ = tokio::time::sleep(std::time::Duration::from_micros(50)) => {}
        }
        Ok(())
    }

    fn process_reads(&mut self, qconn: &mut QuicheConnection) -> QuicResult<()> {
        while let Some(sid) = qconn.stream_readable_next() {
            // Only handle client-initiated bidirectional streams (id % 4 == 0).
            if sid % 4 != 0 {
                continue;
            }

            // Latch onto the first bidi stream as the PG session stream.
            if self.stream_id.is_none() {
                self.stream_id = Some(sid);
            }

            // Ignore streams that aren't our PG session.
            if self.stream_id != Some(sid) {
                continue;
            }

            loop {
                match qconn.stream_recv(sid, &mut self.buf) {
                    Ok((n, _fin)) if n > 0 => {
                        let data = Bytes::copy_from_slice(&self.buf[..n]);
                        match self.read_tx.try_send(data) {
                            Ok(()) => {}
                            Err(mpsc::error::TrySendError::Full(_)) => {
                                // Channel full. Stop reading so quiche retains
                                // the remaining data in its buffer. It will be
                                // readable again on the next cycle.
                                return Ok(());
                            }
                            Err(mpsc::error::TrySendError::Closed(_)) => {
                                // PG handler disconnected. Close the QUIC stream.
                                let _ = qconn.stream_send(sid, b"", true);
                                self.stream_id = None;
                                return Ok(());
                            }
                        }
                    }
                    Ok(_) => break,
                    Err(quiche::Error::Done) => break,
                    Err(e) => {
                        tracing::warn!("QUIC stream {} read error: {}", sid, e);
                        self.stream_id = None;
                        return Ok(());
                    }
                }
            }
        }

        Ok(())
    }

    fn process_writes(&mut self, qconn: &mut QuicheConnection) -> QuicResult<()> {
        let sid = match self.stream_id {
            Some(id) => id,
            None => return Ok(()),
        };

        // Check if the PG handler has shut down.
        if self.closed.load(Ordering::Acquire) {
            // Drain pending buffer and channel, then send FIN.
            for data in self.pending_write_buf.drain(..) {
                let _ = qconn.stream_send(sid, &data, false);
            }
            while let Ok(data) = self.write_rx.try_recv() {
                let _ = qconn.stream_send(sid, &data, false);
            }
            let _ = qconn.stream_send(sid, b"", true);
            self.stream_id = None;
            return Ok(());
        }

        // Drain pending writes from previous flow-control backpressure first.
        while let Some(data) = self.pending_write_buf.pop_front() {
            match qconn.stream_send(sid, &data, false) {
                Ok(_) => {}
                Err(quiche::Error::Done) => {
                    self.pending_write_buf.push_front(data);
                    return Ok(());
                }
                Err(e) => {
                    tracing::warn!("QUIC stream {} write error: {}", sid, e);
                    self.stream_id = None;
                    return Ok(());
                }
            }
        }

        // Accumulate up to 64 channel messages into the staging buffer,
        // then send once. Reduces per-message quiche overhead from N
        // stream_send calls to 1 per cycle.
        self.write_staging.clear();
        let mut writes = 0;
        while writes < 64 && self.write_staging.len() < 131072 {
            match self.write_rx.try_recv() {
                Ok(data) => {
                    self.write_staging.extend_from_slice(&data);
                    writes += 1;
                }
                Err(_) => break,
            }
        }

        if !self.write_staging.is_empty() {
            match qconn.stream_send(sid, &self.write_staging, false) {
                Ok(_) => {}
                Err(quiche::Error::Done) => {
                    self.pending_write_buf
                        .push_back(self.write_staging.split().freeze());
                }
                Err(e) => {
                    tracing::warn!("QUIC stream {} write error: {}", sid, e);
                    self.stream_id = None;
                    return Ok(());
                }
            }
        }

        Ok(())
    }
}

/// Creates a connected QuicStream pair for testing.
/// Returns (stream, read_tx, write_rx) where read_tx feeds data into the stream's
/// read side and write_rx receives data written by the stream.
#[doc(hidden)]
pub fn test_stream_pair() -> (
    QuicStream,
    mpsc::Sender<Bytes>,
    mpsc::UnboundedReceiver<Bytes>,
) {
    let (read_tx, read_rx) = mpsc::channel(READ_CHANNEL_SIZE);
    let (write_tx, write_rx) = mpsc::unbounded_channel();
    let write_notify = Arc::new(Notify::new());

    let stream = QuicStream {
        read_rx,
        write_tx,
        write_notify,
        read_buf: BytesMut::new(),
        peer_addr: "127.0.0.1:5433".parse().unwrap(),
        pending_write_bytes: 0,
        closed: Arc::new(AtomicBool::new(false)),
    };

    (stream, read_tx, write_rx)
}

/// Sets up a QUIC listener on the given UDP address using the provided TLS config.
///
/// Returns a stream of QuicStream instances, one per accepted QUIC connection.
/// Each QuicStream corresponds to the first client-initiated bidirectional stream.
pub async fn setup_quic_listener(
    bind_addr: SocketAddr,
    tls_cert_path: &std::path::Path,
    tls_key_path: &std::path::Path,
    idle_timeout_secs: u32,
) -> io::Result<mpsc::Receiver<(QuicStream, SocketAddr)>> {
    use futures::stream::StreamExt;
    use tokio_quiche::metrics::DefaultMetrics;
    use tokio_quiche::settings::{
        CertificateKind, ConnectionParams, Hooks, QuicSettings, TlsCertificatePaths,
    };

    let socket = tokio::net::UdpSocket::bind(bind_addr).await?;

    // Increase kernel UDP buffer sizes to prevent packet drops at high throughput.
    let sock_ref = socket2::SockRef::from(&socket);
    let _ = sock_ref.set_recv_buffer_size(1048576); // 1 MB
    let _ = sock_ref.set_send_buffer_size(1048576);

    let cert_str = tls_cert_path
        .to_str()
        .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidInput, "Invalid cert path"))?;
    let key_str = tls_key_path
        .to_str()
        .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidInput, "Invalid key path"))?;

    let tls_cert = TlsCertificatePaths {
        cert: cert_str,
        private_key: key_str,
        kind: CertificateKind::X509,
    };

    let mut settings = QuicSettings::default();
    settings.max_idle_timeout = Some(std::time::Duration::from_secs(idle_timeout_secs as u64));

    let params = ConnectionParams::new_server(settings, tls_cert, Hooks::default());

    let mut listeners = tokio_quiche::listen([socket], params, DefaultMetrics)
        .map_err(|e| io::Error::new(io::ErrorKind::Other, format!("QUIC listen failed: {}", e)))?;

    let mut accept_stream = listeners.remove(0);

    // Channel for delivering accepted QuicStreams to the server accept loop.
    let (conn_tx, conn_rx) = mpsc::channel::<(QuicStream, SocketAddr)>(128);

    tokio::spawn(async move {
        while let Some(result) = accept_stream.next().await {
            match result {
                Ok(initial_conn) => {
                    let peer_addr = initial_conn.peer_addr();

                    let (stream_tx, mut stream_rx) = mpsc::channel::<QuicStream>(1);
                    let app = WireProtocolApp::new(stream_tx, peer_addr);

                    // Start the QUIC worker task for this connection.
                    initial_conn.start(app);

                    // Deliver the QuicStream to the accept loop.
                    // The stream was already sent synchronously in WireProtocolApp::new
                    // via try_send, so recv returns immediately.
                    let conn_tx = conn_tx.clone();
                    tokio::spawn(async move {
                        if let Some(quic_stream) = stream_rx.recv().await {
                            let _ = conn_tx.send((quic_stream, peer_addr)).await;
                        }
                    });
                }
                Err(e) => {
                    tracing::warn!("QUIC accept error: {}", e);
                }
            }
        }
    });

    Ok(conn_rx)
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokio::io::{AsyncReadExt, AsyncWriteExt};

    #[tokio::test]
    async fn test_quic_stream_is_encrypted() {
        let (stream, _read_tx, _write_rx) = test_stream_pair();
        assert!(stream.is_encrypted());
    }

    #[tokio::test]
    async fn test_quic_stream_peer_addr() {
        let (stream, _read_tx, _write_rx) = test_stream_pair();
        assert_eq!(stream.peer_addr().port(), 5433);
    }

    #[tokio::test]
    async fn test_quic_stream_read_single_chunk() {
        let (mut stream, read_tx, _write_rx) = test_stream_pair();

        read_tx.send(Bytes::from("hello")).await.unwrap();

        let mut buf = [0u8; 16];
        let n = stream.read(&mut buf).await.unwrap();
        assert_eq!(n, 5);
        assert_eq!(&buf[..n], b"hello");
    }

    #[tokio::test]
    async fn test_quic_stream_read_multiple_chunks() {
        let (mut stream, read_tx, _write_rx) = test_stream_pair();

        read_tx.send(Bytes::from("abc")).await.unwrap();
        read_tx.send(Bytes::from("def")).await.unwrap();

        let mut buf = [0u8; 16];
        let n = stream.read(&mut buf).await.unwrap();
        assert_eq!(&buf[..n], b"abc");

        let n = stream.read(&mut buf).await.unwrap();
        assert_eq!(&buf[..n], b"def");
    }

    #[tokio::test]
    async fn test_quic_stream_read_partial_buffer() {
        let (mut stream, read_tx, _write_rx) = test_stream_pair();

        // Send 10 bytes but read into a 4-byte buffer
        read_tx.send(Bytes::from("0123456789")).await.unwrap();

        let mut buf = [0u8; 4];
        let n = stream.read(&mut buf).await.unwrap();
        assert_eq!(n, 4);
        assert_eq!(&buf[..n], b"0123");

        // Remaining 6 bytes should be buffered internally
        let mut buf2 = [0u8; 16];
        let n = stream.read(&mut buf2).await.unwrap();
        assert_eq!(n, 6);
        assert_eq!(&buf2[..n], b"456789");
    }

    #[tokio::test]
    async fn test_quic_stream_read_eof_on_channel_close() {
        let (mut stream, read_tx, _write_rx) = test_stream_pair();

        drop(read_tx);

        let mut buf = [0u8; 16];
        let n = stream.read(&mut buf).await.unwrap();
        assert_eq!(n, 0); // EOF
    }

    #[tokio::test]
    async fn test_quic_stream_write() {
        let (mut stream, _read_tx, mut write_rx) = test_stream_pair();

        let n = stream.write(b"hello world").await.unwrap();
        assert_eq!(n, 11);

        // Flush to ensure the notify fires
        stream.flush().await.unwrap();

        let data = write_rx.recv().await.unwrap();
        assert_eq!(&data[..], b"hello world");
    }

    #[tokio::test]
    async fn test_quic_stream_write_multiple() {
        let (mut stream, _read_tx, mut write_rx) = test_stream_pair();

        stream.write_all(b"first").await.unwrap();
        stream.write_all(b"second").await.unwrap();

        let d1 = write_rx.recv().await.unwrap();
        let d2 = write_rx.recv().await.unwrap();
        assert_eq!(&d1[..], b"first");
        assert_eq!(&d2[..], b"second");
    }

    #[tokio::test]
    async fn test_quic_stream_write_after_receiver_drop() {
        let (mut stream, _read_tx, write_rx) = test_stream_pair();

        drop(write_rx);

        let result = stream.write(b"data").await;
        assert!(result.is_err());
        assert_eq!(result.unwrap_err().kind(), io::ErrorKind::BrokenPipe);
    }

    #[tokio::test]
    async fn test_quic_stream_flush_clears_pending() {
        let (mut stream, _read_tx, _write_rx) = test_stream_pair();

        // Write less than threshold
        stream.write(b"small").await.unwrap();
        assert!(stream.pending_write_bytes > 0);

        stream.flush().await.unwrap();
        assert_eq!(stream.pending_write_bytes, 0);
    }

    #[tokio::test]
    async fn test_quic_stream_shutdown_sets_closed() {
        let (mut stream, _read_tx, _write_rx) = test_stream_pair();
        let closed = Arc::clone(&stream.closed);

        assert!(!closed.load(Ordering::Acquire));
        stream.shutdown().await.unwrap();
        assert!(closed.load(Ordering::Acquire));
    }

    #[tokio::test]
    async fn test_quic_stream_drop_sets_closed() {
        let (stream, _read_tx, _write_rx) = test_stream_pair();
        let closed = Arc::clone(&stream.closed);

        assert!(!closed.load(Ordering::Acquire));
        drop(stream);
        assert!(closed.load(Ordering::Acquire));
    }

    #[tokio::test]
    async fn test_quic_stream_bidirectional() {
        let (mut stream, read_tx, mut write_rx) = test_stream_pair();

        // Simulate: QUIC side sends data, PG handler reads it
        read_tx.send(Bytes::from("SELECT 1")).await.unwrap();
        let mut buf = [0u8; 64];
        let n = stream.read(&mut buf).await.unwrap();
        assert_eq!(&buf[..n], b"SELECT 1");

        // Simulate: PG handler sends response, QUIC side receives it
        stream.write_all(b"T\x00\x00\x00\x04").await.unwrap();
        stream.flush().await.unwrap();
        let response = write_rx.recv().await.unwrap();
        assert_eq!(&response[..], b"T\x00\x00\x00\x04");
    }

    #[tokio::test]
    async fn test_quic_stream_large_payload() {
        let (mut stream, read_tx, mut write_rx) = test_stream_pair();

        // 64KB payload (larger than a single QUIC packet)
        let payload = vec![0xABu8; 65536];
        read_tx.send(Bytes::from(payload.clone())).await.unwrap();

        let mut result = vec![0u8; 65536];
        stream.read_exact(&mut result).await.unwrap();
        assert_eq!(result, payload);

        // Write 64KB back (exceeds threshold, auto-notifies)
        stream.write_all(&payload).await.unwrap();
        let received = write_rx.recv().await.unwrap();
        assert_eq!(received.len(), 65536);
    }

    #[tokio::test]
    async fn test_wire_protocol_app_creation() {
        let (tx, mut rx) = mpsc::channel::<QuicStream>(1);
        let addr: SocketAddr = "10.0.0.1:5433".parse().unwrap();

        let _app = WireProtocolApp::new(tx, addr);

        // The QuicStream should be delivered immediately via try_send
        let stream = rx.recv().await.unwrap();
        assert_eq!(stream.peer_addr(), addr);
        assert!(stream.is_encrypted());
    }

    #[tokio::test]
    async fn test_wire_protocol_app_should_act_respects_closed() {
        let (tx, _rx) = mpsc::channel::<QuicStream>(1);
        let addr: SocketAddr = "10.0.0.1:5433".parse().unwrap();

        let app = WireProtocolApp::new(tx, addr);
        assert!(app.should_act());

        // Simulate PG handler shutting down
        app.closed.store(true, Ordering::Release);
        assert!(!app.should_act());
    }

    #[tokio::test]
    async fn test_wire_protocol_app_buffer_size() {
        let (tx, _rx) = mpsc::channel::<QuicStream>(1);
        let addr: SocketAddr = "10.0.0.1:5433".parse().unwrap();

        let mut app = WireProtocolApp::new(tx, addr);
        assert_eq!(app.buffer().len(), 65535);
    }

    #[tokio::test]
    async fn test_write_notify_batching() {
        let (mut stream, _read_tx, _write_rx) = test_stream_pair();

        // Small writes below threshold should not trigger notify
        stream.write(b"a").await.unwrap();
        assert_eq!(stream.pending_write_bytes, 1);

        // Write enough to exceed threshold
        let big = vec![0u8; WRITE_NOTIFY_THRESHOLD];
        stream.write(&big).await.unwrap();
        // After exceeding threshold, pending should reset
        assert_eq!(stream.pending_write_bytes, 0);
    }
}
