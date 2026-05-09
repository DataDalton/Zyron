//! Quota gossip background worker.
//!
//! Periodically snapshots the local QuotaRegistry and emits the entries to a
//! configurable transport, so peer ZyronDB instances can converge their
//! per-key quota usage via the monotone-max merge in
//! `zyron_types::scheduling::QuotaRegistry::merge_remote`.
//!
//! Transport is pluggable through the QuotaGossipTransport trait. The default
//! NoopTransport drops outbound frames, useful when no peers are configured
//! or in single-node deployments. ChannelTransport is provided for tests and
//! in-process multi-replica setups.

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, OnceLock};
use std::thread::{self, JoinHandle};
use std::time::Duration;

use tracing::{debug, info};

use zyron_types::scheduling::QuotaRegistry;

/// Configuration for the quota gossip worker.
#[derive(Debug, Clone)]
pub struct QuotaGossipConfig {
    /// Interval between gossip cycles (default 5s)
    pub interval_secs: u64,
    /// Local replica id, included in outbound frames so receivers can ignore
    /// echo from their own broadcast
    pub replica_id: u64,
    /// Skip emitting if the registry is empty (default true)
    pub skip_empty: bool,
}

impl Default for QuotaGossipConfig {
    fn default() -> Self {
        Self {
            interval_secs: 5,
            replica_id: 0,
            skip_empty: true,
        }
    }
}

/// One gossip frame as it travels between replicas.
#[derive(Debug, Clone)]
pub struct QuotaGossipFrame {
    pub source_replica_id: u64,
    pub generation: u64,
    pub entries: Vec<(String, u64)>,
}

/// Pluggable transport used by the worker to ship outbound frames and receive
/// inbound ones. Implementors are responsible for the wire encoding and the
/// actual network IO. Both methods must be cheap and non-blocking, the worker
/// loop holds no other locks while calling them.
pub trait QuotaGossipTransport: Send + Sync {
    /// Send a frame to all peers.
    fn broadcast(&self, frame: QuotaGossipFrame);
    /// Drain inbound frames received since the last call. Returns empty Vec
    /// when no traffic has arrived.
    fn drain_inbound(&self) -> Vec<QuotaGossipFrame>;
}

/// Default transport that drops everything. Used when no peers are configured
pub struct NoopTransport;

impl QuotaGossipTransport for NoopTransport {
    fn broadcast(&self, _frame: QuotaGossipFrame) {}
    fn drain_inbound(&self) -> Vec<QuotaGossipFrame> {
        Vec::new()
    }
}

// ---------------------------------------------------------------------------
// HTTP peer transport (hand-rolled HTTP/1.1, no external HTTP dep)
// ---------------------------------------------------------------------------

/// Configuration for an HTTP-based gossip transport
#[derive(Debug, Clone)]
pub struct HttpTransportConfig {
    /// Local bind address for the inbound listener (e.g. "0.0.0.0:9090")
    pub bind_addr: String,
    /// Peer URLs to broadcast to (e.g. ["http://10.0.0.2:9090", ...])
    pub peers: Vec<String>,
    /// Connect/read timeout per request
    pub timeout_ms: u64,
}

impl Default for HttpTransportConfig {
    fn default() -> Self {
        Self {
            bind_addr: "127.0.0.1:0".into(),
            peers: Vec::new(),
            timeout_ms: 2000,
        }
    }
}

/// Inbound frame queue shared between the HTTP listener and the worker
struct HttpInboundQueue {
    inner: parking_lot::Mutex<std::collections::VecDeque<QuotaGossipFrame>>,
}

impl HttpInboundQueue {
    fn new() -> Self {
        Self {
            inner: parking_lot::Mutex::new(std::collections::VecDeque::new()),
        }
    }
    fn push(&self, frame: QuotaGossipFrame) {
        self.inner.lock().push_back(frame);
    }
    fn drain(&self) -> Vec<QuotaGossipFrame> {
        let mut g = self.inner.lock();
        g.drain(..).collect()
    }
}

/// HTTP/1.1 transport: POSTs JSON frames to each peer, accepts inbound POSTs
/// on a configured bind address. Hand-rolled to avoid pulling in a full HTTP
/// stack just for gossip
pub struct HttpQuotaGossipTransport {
    peers: Vec<String>,
    timeout: Duration,
    inbound: Arc<HttpInboundQueue>,
    /// Listener thread handle so we can shut it down cleanly. Stored as Option
    /// because Drop takes ownership and join() consumes the JoinHandle
    listener: parking_lot::Mutex<Option<std::thread::JoinHandle<()>>>,
    listener_shutdown: Arc<AtomicBool>,
    /// Bound address (after binding "0.0.0.0:0" we discover the real port)
    pub bound_addr: std::net::SocketAddr,
}

impl HttpQuotaGossipTransport {
    /// Starts a transport with a bound TCP listener and the given peer list.
    /// The listener accepts POST requests at any path with a JSON body shaped
    /// like `{"source_replica_id":N, "generation":G, "entries":[["key",N], ...]}`
    pub fn start(config: HttpTransportConfig) -> Result<Self, std::io::Error> {
        use std::net::TcpListener;
        let listener = TcpListener::bind(&config.bind_addr)?;
        listener.set_nonblocking(true)?;
        let bound_addr = listener.local_addr()?;

        let inbound = Arc::new(HttpInboundQueue::new());
        let shutdown = Arc::new(AtomicBool::new(false));

        let inbound_for_thread = Arc::clone(&inbound);
        let shutdown_for_thread = Arc::clone(&shutdown);
        let timeout = Duration::from_millis(config.timeout_ms);

        let handle = std::thread::Builder::new()
            .name("zyron-quota-gossip-http".into())
            .spawn(move || {
                Self::listener_loop(listener, inbound_for_thread, shutdown_for_thread, timeout);
            })
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))?;

        Ok(Self {
            peers: config.peers,
            timeout,
            inbound,
            listener: parking_lot::Mutex::new(Some(handle)),
            listener_shutdown: shutdown,
            bound_addr,
        })
    }

    fn listener_loop(
        listener: std::net::TcpListener,
        inbound: Arc<HttpInboundQueue>,
        shutdown: Arc<AtomicBool>,
        timeout: Duration,
    ) {
        loop {
            if shutdown.load(Ordering::Acquire) {
                return;
            }
            match listener.accept() {
                Ok((mut stream, _peer)) => {
                    let _ = stream.set_read_timeout(Some(timeout));
                    let _ = stream.set_write_timeout(Some(timeout));
                    if let Some(frame) = read_http_post_json::<QuotaGossipFrameWire>(&mut stream) {
                        inbound.push(frame.into());
                        let _ = write_http_response(&mut stream, 200, "OK", b"");
                    } else {
                        let _ = write_http_response(&mut stream, 400, "Bad Request", b"");
                    }
                }
                Err(ref e) if e.kind() == std::io::ErrorKind::WouldBlock => {
                    std::thread::sleep(Duration::from_millis(50));
                }
                Err(_) => return,
            }
        }
    }

    /// Peek the inbound queue length without draining
    pub fn inbound_len(&self) -> usize {
        self.inbound.inner.lock().len()
    }

    /// Stops the listener thread and waits for it to exit
    pub fn shutdown(&self) {
        self.listener_shutdown.store(true, Ordering::Release);
        if let Some(handle) = self.listener.lock().take() {
            // The accept loop polls shutdown every ~50ms; join with bounded patience
            let _ = handle.join();
        }
    }
}

impl QuotaGossipTransport for HttpQuotaGossipTransport {
    fn broadcast(&self, frame: QuotaGossipFrame) {
        // Serialize once and reuse for every peer rather than re-encoding
        // the same body N times for an N-peer fanout
        let body = frame_to_json(&frame);
        for peer in &self.peers {
            if let Err(e) = http_post(peer, &body, self.timeout) {
                tracing::debug!("quota gossip POST to {} failed: {}", peer, e);
            }
        }
    }
    fn drain_inbound(&self) -> Vec<QuotaGossipFrame> {
        self.inbound.drain()
    }
}

impl Drop for HttpQuotaGossipTransport {
    fn drop(&mut self) {
        self.shutdown();
    }
}

/// Wire form (matches the JSON shape exactly)
#[derive(Debug)]
struct QuotaGossipFrameWire {
    source_replica_id: u64,
    generation: u64,
    entries: Vec<(String, u64)>,
}

impl From<QuotaGossipFrameWire> for QuotaGossipFrame {
    fn from(w: QuotaGossipFrameWire) -> Self {
        Self {
            source_replica_id: w.source_replica_id,
            generation: w.generation,
            entries: w.entries,
        }
    }
}

fn frame_to_json(f: &QuotaGossipFrame) -> String {
    let mut s = String::with_capacity(64 + 32 * f.entries.len());
    s.push('{');
    s.push_str(&format!(r#""source_replica_id":{},"#, f.source_replica_id));
    s.push_str(&format!(r#""generation":{},"#, f.generation));
    s.push_str(r#""entries":["#);
    for (i, (k, v)) in f.entries.iter().enumerate() {
        if i > 0 {
            s.push(',');
        }
        s.push('[');
        s.push('"');
        for c in k.chars() {
            match c {
                '"' => s.push_str("\\\""),
                '\\' => s.push_str("\\\\"),
                '\n' => s.push_str("\\n"),
                '\r' => s.push_str("\\r"),
                '\t' => s.push_str("\\t"),
                c if (c as u32) < 0x20 => s.push_str(&format!("\\u{:04x}", c as u32)),
                c => s.push(c),
            }
        }
        s.push('"');
        s.push(',');
        s.push_str(&v.to_string());
        s.push(']');
    }
    s.push_str("]}");
    s
}

fn http_post(url: &str, body: &str, timeout: Duration) -> std::io::Result<()> {
    use std::io::{Read, Write};
    let (host, port, path) = parse_http_url(url)?;
    let addr = format!("{}:{}", host, port);
    let mut stream = std::net::TcpStream::connect_timeout(
        &addr
            .to_socket_addrs()?
            .next()
            .ok_or_else(|| std::io::Error::new(std::io::ErrorKind::AddrNotAvailable, "no addrs"))?,
        timeout,
    )?;
    stream.set_read_timeout(Some(timeout))?;
    stream.set_write_timeout(Some(timeout))?;
    // The Host header for IPv6 literals must wrap the address in brackets per
    // RFC 7230, otherwise the colon would be ambiguous with the port separator
    let host_header = if host.contains(':') {
        format!("[{}]:{}", host, port)
    } else {
        format!("{}:{}", host, port)
    };
    let req = format!(
        "POST {} HTTP/1.1\r\nHost: {}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
        path,
        host_header,
        body.len(),
        body
    );
    stream.write_all(req.as_bytes())?;
    let mut sink = Vec::new();
    let _ = stream.read_to_end(&mut sink);
    Ok(())
}

use std::net::ToSocketAddrs;

/// Parses a `http://host[:port]/path` URL into its components. Accepts both
/// IPv4 / hostname forms (`http://10.0.0.1:9090/x`) and IPv6 bracket forms
/// (`http://[::1]:9090/x`). Defaults to port 80 when omitted. Path defaults
/// to `/` when omitted. Returns Err with a specific reason on failure so the
/// caller can surface it to the operator instead of silently swallowing
fn parse_http_url(url: &str) -> std::io::Result<(&str, u16, &str)> {
    use std::io::{Error, ErrorKind};
    // The transport speaks plain HTTP. https:// is rejected explicitly so an
    // operator misconfiguring TLS doesn't get silent plaintext gossip
    if let Some(_rest) = url.strip_prefix("https://") {
        return Err(Error::new(
            ErrorKind::Unsupported,
            "https:// is not supported by the quota gossip transport, use http:// or wrap with a TLS proxy",
        ));
    }
    let rest = url.strip_prefix("http://").ok_or_else(|| {
        Error::new(
            ErrorKind::InvalidInput,
            format!("URL must start with http://, got {:?}", url),
        )
    })?;

    // Split host[:port] from /path
    let (host_port, path) = match rest.find('/') {
        Some(i) => (&rest[..i], &rest[i..]),
        None => (rest, "/"),
    };
    if host_port.is_empty() {
        return Err(Error::new(ErrorKind::InvalidInput, "URL has empty host"));
    }

    // IPv6 literal forms: [host]:port  or  [host]
    let (host, port) = if let Some(rest_after_lb) = host_port.strip_prefix('[') {
        let close = rest_after_lb.find(']').ok_or_else(|| {
            Error::new(
                ErrorKind::InvalidInput,
                format!("IPv6 literal missing closing ']' in {:?}", host_port),
            )
        })?;
        let host = &rest_after_lb[..close];
        let after = &rest_after_lb[close + 1..];
        let port = if let Some(p) = after.strip_prefix(':') {
            p.parse::<u16>().map_err(|e| {
                Error::new(
                    ErrorKind::InvalidInput,
                    format!("invalid port {:?} in URL: {}", p, e),
                )
            })?
        } else if after.is_empty() {
            80
        } else {
            return Err(Error::new(
                ErrorKind::InvalidInput,
                format!(
                    "expected ':port' or end after IPv6 literal, got {:?}",
                    after
                ),
            ));
        };
        (host, port)
    } else {
        // IPv4 / hostname forms: host or host:port. Multiple colons in the
        // host portion mean an unbracketed IPv6 literal, which is ambiguous
        // with the port separator. Require brackets per RFC 3986
        let colon_count = host_port.bytes().filter(|&b| b == b':').count();
        if colon_count > 1 {
            return Err(Error::new(
                ErrorKind::InvalidInput,
                format!(
                    "ambiguous host {:?}, IPv6 literals must be wrapped in [brackets]",
                    host_port
                ),
            ));
        }
        match host_port.rfind(':') {
            Some(i) => {
                let port_str = &host_port[i + 1..];
                let port = port_str.parse::<u16>().map_err(|e| {
                    Error::new(
                        ErrorKind::InvalidInput,
                        format!("invalid port {:?} in URL: {}", port_str, e),
                    )
                })?;
                (&host_port[..i], port)
            }
            None => (host_port, 80u16),
        }
    };
    Ok((host, port, path))
}

// Cap the inbound HTTP request size to bound memory in the face of a peer
// (or attacker) advertising a large Content-Length and slow-dripping bytes.
// 1 MiB is well above any plausible gossip payload (current QuotaRegistry
// snapshots are at most a few KB) but well below what would matter for OOM
const MAX_HTTP_HEADER_BYTES: usize = 16 * 1024;
const MAX_HTTP_BODY_BYTES: usize = 1 * 1024 * 1024;

fn read_http_post_json<T>(stream: &mut std::net::TcpStream) -> Option<T>
where
    T: WireDecode,
{
    use std::io::Read;
    let mut buf = Vec::with_capacity(4096);
    let mut tmp = [0u8; 4096];
    let mut header_end = None;
    while header_end.is_none() {
        let n = stream.read(&mut tmp).ok()?;
        if n == 0 {
            return None;
        }
        buf.extend_from_slice(&tmp[..n]);
        if buf.len() > MAX_HTTP_HEADER_BYTES {
            return None;
        }
        if let Some(p) = find_subseq(&buf, b"\r\n\r\n") {
            header_end = Some(p);
        }
    }
    let header_end = header_end?;
    let headers = std::str::from_utf8(&buf[..header_end]).ok()?;
    let mut content_length: Option<usize> = None;
    for line in headers.split("\r\n") {
        let lower = line.to_ascii_lowercase();
        if let Some(rest) = lower.strip_prefix("content-length:") {
            let v: usize = rest.trim().parse().ok()?;
            content_length = Some(v);
        }
    }
    // Reject any request without an explicit Content-Length. Chunked transfer
    // encoding is not supported by this transport, falling through to length-0
    // body would silently parse a partial JSON object as a blank frame
    let content_length = content_length?;
    if content_length > MAX_HTTP_BODY_BYTES {
        return None;
    }
    let body_start = header_end + 4;
    let needed = body_start + content_length;
    while buf.len() < needed {
        let n = stream.read(&mut tmp).ok()?;
        if n == 0 {
            // Connection closed before the full body arrived. Refuse the
            // request rather than silently truncating to whatever bytes we got
            return None;
        }
        buf.extend_from_slice(&tmp[..n]);
        if buf.len() > body_start + MAX_HTTP_BODY_BYTES {
            return None;
        }
    }
    let body = &buf[body_start..needed];
    T::decode(std::str::from_utf8(body).ok()?)
}

fn find_subseq(haystack: &[u8], needle: &[u8]) -> Option<usize> {
    haystack.windows(needle.len()).position(|w| w == needle)
}

fn write_http_response(
    stream: &mut std::net::TcpStream,
    status: u16,
    status_text: &str,
    body: &[u8],
) -> std::io::Result<()> {
    use std::io::Write;
    let resp = format!(
        "HTTP/1.1 {} {}\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
        status,
        status_text,
        body.len()
    );
    stream.write_all(resp.as_bytes())?;
    stream.write_all(body)?;
    Ok(())
}

trait WireDecode: Sized {
    fn decode(s: &str) -> Option<Self>;
}

impl WireDecode for QuotaGossipFrameWire {
    fn decode(s: &str) -> Option<Self> {
        // Minimal JSON object parser tailored to QuotaGossipFrame's shape.
        // Avoids pulling serde into this crate just for one inbound type
        let v: serde_json::Value = serde_json::from_str(s).ok()?;
        let obj = v.as_object()?;
        let source_replica_id = obj.get("source_replica_id")?.as_u64()?;
        let generation = obj.get("generation")?.as_u64()?;
        let entries_v = obj.get("entries")?.as_array()?;
        let mut entries = Vec::with_capacity(entries_v.len());
        for e in entries_v {
            let pair = e.as_array()?;
            if pair.len() != 2 {
                return None;
            }
            let key = pair[0].as_str()?.to_string();
            let val = pair[1].as_u64()?;
            entries.push((key, val));
        }
        Some(QuotaGossipFrameWire {
            source_replica_id,
            generation,
            entries,
        })
    }
}

/// In-process transport that fans out via crossbeam channels. Useful for
/// integration tests where multiple QuotaGossip instances share a hub
pub struct ChannelTransport {
    sender: crossbeam::channel::Sender<QuotaGossipFrame>,
    receiver: parking_lot::Mutex<crossbeam::channel::Receiver<QuotaGossipFrame>>,
}

impl ChannelTransport {
    /// Creates a new transport bound to the given outbound and inbound channels
    pub fn new(
        sender: crossbeam::channel::Sender<QuotaGossipFrame>,
        receiver: crossbeam::channel::Receiver<QuotaGossipFrame>,
    ) -> Self {
        Self {
            sender,
            receiver: parking_lot::Mutex::new(receiver),
        }
    }
}

impl QuotaGossipTransport for ChannelTransport {
    fn broadcast(&self, frame: QuotaGossipFrame) {
        let _ = self.sender.send(frame);
    }
    fn drain_inbound(&self) -> Vec<QuotaGossipFrame> {
        let rx = self.receiver.lock();
        let mut out = Vec::new();
        while let Ok(f) = rx.try_recv() {
            out.push(f);
        }
        out
    }
}

/// Background worker handle.
pub struct QuotaGossipWorker {
    shutdown: Arc<AtomicBool>,
    waker: Arc<OnceLock<thread::Thread>>,
    thread: Option<JoinHandle<()>>,
    pub stats: Arc<QuotaGossipStats>,
}

#[derive(Debug, Default)]
pub struct QuotaGossipStats {
    pub cycles: AtomicU64,
    pub frames_sent: AtomicU64,
    pub frames_received: AtomicU64,
    pub keys_merged: AtomicU64,
}

impl QuotaGossipWorker {
    /// Starts the worker thread.
    pub fn start(
        registry: Arc<QuotaRegistry>,
        transport: Arc<dyn QuotaGossipTransport>,
        config: QuotaGossipConfig,
    ) -> Self {
        let shutdown = Arc::new(AtomicBool::new(false));
        let waker = Arc::new(OnceLock::new());
        let stats = Arc::new(QuotaGossipStats::default());

        let thread_shutdown = Arc::clone(&shutdown);
        let thread_waker = Arc::clone(&waker);
        let thread_stats = Arc::clone(&stats);

        let handle = thread::Builder::new()
            .name("zyron-quota-gossip".into())
            .spawn(move || {
                let _ = thread_waker.set(thread::current());
                Self::worker_loop(
                    &registry,
                    transport.as_ref(),
                    &config,
                    &thread_shutdown,
                    &thread_stats,
                );
            })
            .expect("failed to spawn quota gossip thread");

        Self {
            shutdown,
            waker,
            thread: Some(handle),
            stats,
        }
    }

    fn worker_loop(
        registry: &QuotaRegistry,
        transport: &dyn QuotaGossipTransport,
        config: &QuotaGossipConfig,
        shutdown: &AtomicBool,
        stats: &QuotaGossipStats,
    ) {
        let interval = Duration::from_secs(config.interval_secs);
        let mut generation: u64 = 0;
        loop {
            thread::park_timeout(interval);
            if shutdown.load(Ordering::Acquire) {
                return;
            }
            stats.cycles.fetch_add(1, Ordering::Relaxed);

            // Drain inbound first so our own broadcast does not include echo
            let inbound = transport.drain_inbound();
            for frame in inbound {
                if frame.source_replica_id == config.replica_id {
                    continue;
                }
                stats.frames_received.fetch_add(1, Ordering::Relaxed);
                stats
                    .keys_merged
                    .fetch_add(frame.entries.len() as u64, Ordering::Relaxed);
                let pairs: Vec<(String, u64)> = frame.entries;
                let refs: Vec<(String, u64)> = pairs;
                registry.merge_remote(&refs);
            }

            let entries = registry.snapshot();
            if config.skip_empty && entries.is_empty() {
                debug!("quota gossip: registry empty, skipping broadcast");
                continue;
            }
            generation = generation.wrapping_add(1);
            let frame = QuotaGossipFrame {
                source_replica_id: config.replica_id,
                generation,
                entries,
            };
            transport.broadcast(frame);
            stats.frames_sent.fetch_add(1, Ordering::Relaxed);
        }
    }

    /// Wakes the worker out of its sleep so the next cycle runs immediately.
    /// No-op if the worker thread has not yet registered itself
    pub fn wake(&self) {
        if let Some(t) = self.waker.get() {
            t.unpark();
        }
    }

    /// Stops the worker thread and joins it.
    pub fn shutdown(&mut self) {
        self.shutdown.store(true, Ordering::Release);
        self.wake();
        if let Some(handle) = self.thread.take() {
            let _ = handle.join();
        }
        info!("quota gossip worker stopped");
    }
}

impl Drop for QuotaGossipWorker {
    fn drop(&mut self) {
        if self.thread.is_some() {
            self.shutdown();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Instant;

    fn wait_for<F: Fn() -> bool>(check: F, timeout: Duration) -> bool {
        let deadline = Instant::now() + timeout;
        while Instant::now() < deadline {
            if check() {
                return true;
            }
            thread::sleep(Duration::from_millis(20));
        }
        false
    }

    #[test]
    fn gossip_broadcasts_local_quota_state() {
        let registry = Arc::new(QuotaRegistry::new());
        let _ = registry.increment("key1", 5, 1000);
        let _ = registry.increment("key2", 7, 1000);

        let (tx, rx) = crossbeam::channel::unbounded();
        // Use an unrelated rx for the worker's drain_inbound (no inbound traffic)
        let (_tx_in, rx_in) = crossbeam::channel::unbounded();
        let transport = Arc::new(ChannelTransport::new(tx, rx_in));
        let mut worker = QuotaGossipWorker::start(
            registry.clone(),
            transport.clone(),
            QuotaGossipConfig {
                interval_secs: 1,
                replica_id: 1,
                skip_empty: true,
            },
        );

        // Give the worker a tick to broadcast
        worker.wake();
        let received = wait_for(
            || rx.try_iter().next().is_some() || rx.len() > 0,
            Duration::from_secs(3),
        );
        worker.shutdown();
        assert!(received, "expected outbound gossip frame within 3s");
    }

    #[test]
    fn http_transport_handles_special_chars_in_keys() {
        // Keys with quotes, backslashes, newlines, and unicode must round-trip
        let b = HttpQuotaGossipTransport::start(HttpTransportConfig {
            bind_addr: "127.0.0.1:0".into(),
            peers: Vec::new(),
            timeout_ms: 2000,
        })
        .expect("start node B");
        let b_addr = b.bound_addr;
        let a = HttpQuotaGossipTransport::start(HttpTransportConfig {
            bind_addr: "127.0.0.1:0".into(),
            peers: vec![format!("http://{}:{}", b_addr.ip(), b_addr.port())],
            timeout_ms: 2000,
        })
        .expect("start node A");

        let tricky_keys = vec![
            ("with\"quote".to_string(), 1u64),
            ("with\\backslash".to_string(), 2u64),
            ("with\nnewline".to_string(), 3u64),
            ("café".to_string(), 4u64),
            ("control\x01char".to_string(), 5u64),
        ];
        let frame = QuotaGossipFrame {
            source_replica_id: 99,
            generation: 1,
            entries: tricky_keys.clone(),
        };
        a.broadcast(frame);

        let received = wait_for(|| b.inbound_len() > 0, Duration::from_secs(2));
        let drained = b.drain_inbound();
        a.shutdown();
        b.shutdown();
        assert!(
            received,
            "node B never received gossip frame with special chars"
        );
        assert_eq!(drained.len(), 1);
        let mut got = drained[0].entries.clone();
        got.sort();
        let mut expected = tricky_keys;
        expected.sort();
        assert_eq!(got, expected, "special-char keys did not round-trip");
    }

    #[test]
    fn parse_http_url_handles_ipv4_with_port() {
        let (h, p, path) = super::parse_http_url("http://10.0.0.1:9090/x").unwrap();
        assert_eq!(h, "10.0.0.1");
        assert_eq!(p, 9090);
        assert_eq!(path, "/x");
    }

    #[test]
    fn parse_http_url_defaults_port_to_80() {
        let (h, p, _) = super::parse_http_url("http://example.com/").unwrap();
        assert_eq!(h, "example.com");
        assert_eq!(p, 80);
    }

    #[test]
    fn parse_http_url_handles_ipv6_brackets() {
        let (h, p, path) = super::parse_http_url("http://[::1]:9090/x").unwrap();
        assert_eq!(h, "::1");
        assert_eq!(p, 9090);
        assert_eq!(path, "/x");
        // Without explicit port the IPv6 form defaults to 80 too
        let (h, p, path) = super::parse_http_url("http://[fe80::1]/").unwrap();
        assert_eq!(h, "fe80::1");
        assert_eq!(p, 80);
        assert_eq!(path, "/");
    }

    #[test]
    fn parse_http_url_rejects_https() {
        let err = super::parse_http_url("https://example.com/").unwrap_err();
        assert_eq!(err.kind(), std::io::ErrorKind::Unsupported);
    }

    #[test]
    fn parse_http_url_rejects_invalid_port() {
        let err = super::parse_http_url("http://example.com:notaport/").unwrap_err();
        assert_eq!(err.kind(), std::io::ErrorKind::InvalidInput);
        let err = super::parse_http_url("http://example.com:99999/").unwrap_err();
        assert_eq!(err.kind(), std::io::ErrorKind::InvalidInput);
    }

    #[test]
    fn parse_http_url_rejects_empty_host() {
        let err = super::parse_http_url("http:///path").unwrap_err();
        assert_eq!(err.kind(), std::io::ErrorKind::InvalidInput);
    }

    #[test]
    fn parse_http_url_rejects_unbracketed_ipv6() {
        // Without brackets the rfind(':') splits at the wrong colon
        let err = super::parse_http_url("http://::1:9090/x").unwrap_err();
        assert_eq!(err.kind(), std::io::ErrorKind::InvalidInput);
    }

    #[test]
    fn http_transport_dead_peer_does_not_crash_broadcast() {
        // Configure A to broadcast to a peer that is not listening; broadcast
        // should swallow the error and continue (so a single dead peer never
        // crashes the whole gossip cycle)
        let a = HttpQuotaGossipTransport::start(HttpTransportConfig {
            bind_addr: "127.0.0.1:0".into(),
            // 127.0.0.2:1 is reserved/unreachable — connect should fail fast
            peers: vec!["http://127.0.0.1:1".into()],
            timeout_ms: 500,
        })
        .expect("start node A");
        a.broadcast(QuotaGossipFrame {
            source_replica_id: 1,
            generation: 1,
            entries: vec![("x".into(), 1)],
        });
        a.shutdown();
    }

    #[test]
    fn http_transport_round_trips_a_frame_between_two_nodes() {
        // Bind two HTTP transports to localhost ephemeral ports; have node A
        // POST a frame to node B. Node B's drain_inbound should then return
        // the frame intact
        let b = HttpQuotaGossipTransport::start(HttpTransportConfig {
            bind_addr: "127.0.0.1:0".into(),
            peers: Vec::new(),
            timeout_ms: 2000,
        })
        .expect("start node B");
        let b_addr = b.bound_addr;
        let a = HttpQuotaGossipTransport::start(HttpTransportConfig {
            bind_addr: "127.0.0.1:0".into(),
            peers: vec![format!("http://{}:{}", b_addr.ip(), b_addr.port())],
            timeout_ms: 2000,
        })
        .expect("start node A");

        let frame = QuotaGossipFrame {
            source_replica_id: 11,
            generation: 4,
            entries: vec![("apikey:42".to_string(), 17), ("queries".into(), 99)],
        };
        a.broadcast(frame.clone());

        // Listener thread polls every 50ms; give it up to 2s to land
        let received = wait_for(|| b.inbound_len() > 0, Duration::from_secs(2));
        let drained = b.drain_inbound();
        a.shutdown();
        b.shutdown();
        assert!(received, "node B never received gossip frame");
        assert_eq!(drained.len(), 1, "expected exactly one frame");
        let r = &drained[0];
        assert_eq!(r.source_replica_id, 11);
        assert_eq!(r.generation, 4);
        assert_eq!(r.entries.len(), 2);
        assert!(r.entries.contains(&("apikey:42".to_string(), 17)));
        assert!(r.entries.contains(&("queries".to_string(), 99)));
    }

    #[test]
    fn gossip_merge_remote_advances_local_state() {
        let registry = Arc::new(QuotaRegistry::new());
        let _ = registry.increment("k", 1, 1000);

        // Build a transport whose inbound channel we control
        let (_tx_out, _rx_out) = crossbeam::channel::unbounded::<QuotaGossipFrame>();
        let (tx_in, rx_in) = crossbeam::channel::unbounded();
        let (junk_tx, _junk_rx) = crossbeam::channel::unbounded::<QuotaGossipFrame>();
        let transport = Arc::new(ChannelTransport::new(junk_tx, rx_in));

        let mut worker = QuotaGossipWorker::start(
            registry.clone(),
            transport.clone(),
            QuotaGossipConfig {
                interval_secs: 1,
                replica_id: 7,
                skip_empty: false,
            },
        );

        // Push an inbound gossip frame from replica 9 raising k to 50
        tx_in
            .send(QuotaGossipFrame {
                source_replica_id: 9,
                generation: 1,
                entries: vec![("k".to_string(), 50)],
            })
            .unwrap();

        worker.wake();
        let merged = wait_for(
            || {
                registry
                    .snapshot()
                    .iter()
                    .any(|(k, v)| k == "k" && *v >= 50)
            },
            Duration::from_secs(3),
        );
        worker.shutdown();
        assert!(merged, "expected QuotaRegistry to converge to remote max");
    }
}
