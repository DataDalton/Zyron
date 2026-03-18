//! Transport layer abstraction for the PostgreSQL wire protocol.
//!
//! Provides a trait over byte streams so the PG wire protocol codec
//! works identically over TCP and QUIC transports. TCP is always
//! available. QUIC support requires the "quic" feature flag.

use std::time::Duration;
use tokio::io::{AsyncRead, AsyncWrite};
use tokio::net::TcpStream;

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

impl WireTransport for TcpStream {
    fn is_encrypted(&self) -> bool {
        false
    }

    fn configure_immediate(&self) {
        // Disable Nagle's algorithm for low-latency message exchange.
        let _ = self.set_nodelay(true);
    }

    fn configure_post_handshake(&self) {
        // TCP keepalive (60s idle, 10s probe interval)
        let keepalive = socket2::TcpKeepalive::new()
            .with_time(Duration::from_secs(60))
            .with_interval(Duration::from_secs(10));
        let sock_ref = socket2::SockRef::from(self);
        let _ = sock_ref.set_tcp_keepalive(&keepalive);

        // Linux-specific TCP tuning for lower latency and faster dead-connection detection.
        #[cfg(target_os = "linux")]
        {
            use std::os::unix::io::AsRawFd;
            let fd = sock_ref.as_raw_fd();

            // Limit time for unacknowledged data before declaring the connection dead (30s).
            let timeout_ms: libc::c_int = 30000;
            unsafe {
                libc::setsockopt(
                    fd,
                    libc::IPPROTO_TCP,
                    libc::TCP_USER_TIMEOUT,
                    &timeout_ms as *const _ as *const libc::c_void,
                    std::mem::size_of::<libc::c_int>() as libc::socklen_t,
                );
            }

            // Report socket as writable only when unsent data is below 16KB.
            let _ = sock_ref.set_tcp_notsent_lowat(16384);

            // Busy-poll for up to 50us to reduce network latency.
            let _ = sock_ref.set_busy_poll(50);

            // Batch TCP ACKs to reduce per-packet overhead.
            let _ = sock_ref.set_tcp_ack_frequency(2);

            // Disable delayed ACKs for faster initial handshake response.
            let quickack: libc::c_int = 1;
            unsafe {
                libc::setsockopt(
                    fd,
                    libc::IPPROTO_TCP,
                    libc::TCP_QUICKACK,
                    &quickack as *const _ as *const libc::c_void,
                    std::mem::size_of::<libc::c_int>() as libc::socklen_t,
                );
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
