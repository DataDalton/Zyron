//! Transport layer abstraction for the PostgreSQL wire protocol.
//!
//! Provides a trait over byte streams so the PG wire protocol codec
//! works identically over TCP and QUIC transports. TCP is always
//! available. QUIC support requires the "quic" feature flag.

use std::time::Duration;
use tokio::io::{AsyncRead, AsyncWrite};
use tokio::net::TcpStream;
use tracing::debug;

/// Abstraction over TCP and QUIC byte streams.
///
/// The PG wire protocol codec is identical regardless of transport.
/// Transport-specific behavior (encryption, keepalive, socket tuning)
/// is handled by each implementation.
pub trait WireTransport: AsyncRead + AsyncWrite + Unpin + Send {
    /// Returns true if the transport provides built-in encryption.
    /// QUIC has mandatory TLS 1.3, so QUIC streams return true.
    /// TCP streams return false (TLS is negotiated separately).
    fn is_encrypted(&self) -> bool;

    /// Configures transport-specific options immediately at connection creation.
    /// TCP: disables Nagle's algorithm for low-latency message exchange.
    /// QUIC: no-op.
    fn configure_immediate(&self) {}

    /// Configures transport-specific options after the startup handshake.
    /// TCP: sets keepalive and OS-specific socket tuning.
    /// QUIC: no-op (handled by the QUIC connection layer).
    fn configure_post_handshake(&self) {}
}

/// Returns the transport name for logging and diagnostics.
pub fn transport_name<T: WireTransport>(stream: &T) -> &'static str {
    if stream.is_encrypted() { "QUIC" } else { "TCP" }
}

/// Metadata reported by a transport about its TLS session, if any.
pub trait TlsTransportInfo {
    /// Returns the negotiated cipher suite name in a stable form.
    fn negotiated_cipher(&self) -> Option<&'static str>;
    /// Returns the peer's leaf certificate as DER bytes.
    fn peer_cert_der(&self) -> Option<Vec<u8>>;
    /// Returns the SHA-256 fingerprint of the peer's leaf certificate.
    fn peer_cert_fingerprint_sha256(&self) -> Option<[u8; 32]>;
}

impl<S: AsyncRead + AsyncWrite + Unpin + Send> WireTransport
    for tokio_rustls::server::TlsStream<S>
{
    fn is_encrypted(&self) -> bool {
        true
    }
}

impl<S: AsyncRead + AsyncWrite + Unpin + Send> TlsTransportInfo
    for tokio_rustls::server::TlsStream<S>
{
    fn negotiated_cipher(&self) -> Option<&'static str> {
        let (_, conn) = self.get_ref();
        conn.negotiated_cipher_suite()
            .map(|c| c.suite().as_str().unwrap_or("unknown"))
    }

    fn peer_cert_der(&self) -> Option<Vec<u8>> {
        let (_, conn) = self.get_ref();
        conn.peer_certificates()
            .and_then(|c| c.first().map(|c| c.as_ref().to_vec()))
    }

    fn peer_cert_fingerprint_sha256(&self) -> Option<[u8; 32]> {
        self.peer_cert_der()
            .map(|der| crate::tls::sha256_fingerprint(&der))
    }
}

impl<S: AsyncRead + AsyncWrite + Unpin + Send> WireTransport
    for tokio_rustls::client::TlsStream<S>
{
    fn is_encrypted(&self) -> bool {
        true
    }
}

impl<S: AsyncRead + AsyncWrite + Unpin + Send> TlsTransportInfo
    for tokio_rustls::client::TlsStream<S>
{
    fn negotiated_cipher(&self) -> Option<&'static str> {
        let (_, conn) = self.get_ref();
        conn.negotiated_cipher_suite()
            .map(|c| c.suite().as_str().unwrap_or("unknown"))
    }

    fn peer_cert_der(&self) -> Option<Vec<u8>> {
        let (_, conn) = self.get_ref();
        conn.peer_certificates()
            .and_then(|c| c.first().map(|c| c.as_ref().to_vec()))
    }

    fn peer_cert_fingerprint_sha256(&self) -> Option<[u8; 32]> {
        self.peer_cert_der()
            .map(|der| crate::tls::sha256_fingerprint(&der))
    }
}

impl WireTransport for TcpStream {
    fn is_encrypted(&self) -> bool {
        false
    }

    fn configure_immediate(&self) {
        // Disable Nagle's algorithm for low-latency message exchange.
        let _ = self.set_nodelay(true);
    }

    fn configure_post_handshake(&self) {
        // Socket tuning is best effort. A kernel that rejects an option leaves
        // the connection usable with weaker latency or dead-peer behavior, so
        // failures are logged rather than propagated. They are not discarded
        // silently: without TCP_USER_TIMEOUT, dead-peer detection falls back to
        // keepalive alone, which is worth being able to see in a log.
        let sock_ref = socket2::SockRef::from(self);

        // TCP keepalive (60s idle, 10s probe interval)
        let keepalive = socket2::TcpKeepalive::new()
            .with_time(Duration::from_secs(60))
            .with_interval(Duration::from_secs(10));
        if let Err(e) = sock_ref.set_tcp_keepalive(&keepalive) {
            debug!("tcp keepalive not applied: {}", e);
        }

        // Linux-specific TCP tuning for lower latency and faster dead-connection detection.
        #[cfg(target_os = "linux")]
        {
            // Declare the connection dead after 30s of unacknowledged data
            // instead of waiting out the default retransmit schedule.
            if let Err(e) = sock_ref.set_tcp_user_timeout(Some(Duration::from_secs(30))) {
                debug!("tcp user timeout not applied: {}", e);
            }

            // Report the socket writable only when unsent data is below 16KB.
            if let Err(e) = sock_ref.set_tcp_notsent_lowat(16384) {
                debug!("tcp notsent lowat not applied: {}", e);
            }

            // Busy-poll the receive path for up to 50us before sleeping. Needs
            // net.core.busy_poll enabled, and applies to blocking receives, so
            // its effect under the runtime's epoll-driven non-blocking sockets
            // is limited.
            if let Err(e) = sock_ref.set_busy_poll(50) {
                debug!("tcp busy poll not applied: {}", e);
            }

            // Switch to quickack mode so the first post-handshake exchange is
            // not held for the delayed-ack timer. The kernel treats this as a
            // one-shot, reverting to delayed acks as the connection proceeds,
            // so holding quickack on would require re-arming after each read.
            if let Err(e) = sock_ref.set_tcp_quickack(true) {
                debug!("tcp quickack not applied: {}", e);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokio::net::TcpListener;

    #[tokio::test]
    async fn test_tcp_is_not_encrypted() {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();

        let client = TcpStream::connect(addr).await.unwrap();
        assert!(!client.is_encrypted());
    }

    #[tokio::test]
    async fn test_tcp_configure_immediate_sets_nodelay() {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();

        let client = TcpStream::connect(addr).await.unwrap();
        client.configure_immediate();
        assert!(client.nodelay().unwrap());
    }

    #[tokio::test]
    async fn test_tcp_configure_post_handshake_no_panic() {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();

        let client = TcpStream::connect(addr).await.unwrap();
        // Should not panic on any platform.
        client.configure_post_handshake();
    }

    #[tokio::test]
    async fn test_transport_name_tcp() {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();

        let client = TcpStream::connect(addr).await.unwrap();
        assert_eq!(transport_name(&client), "TCP");
    }

    #[tokio::test]
    async fn test_transport_name_quic() {
        let (stream, _tx, _rx) = crate::quic::test_stream_pair();
        assert_eq!(transport_name(&stream), "QUIC");
    }
}
