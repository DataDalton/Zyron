//! TLS 1.3 transport for the PostgreSQL wire protocol.
//!
//! Provides server-side and client-side TLS over an arbitrary byte stream.
//! Supports optional mTLS on the server, custom CA roots and client certs on
//! the client, and SHA-256 fingerprint pinning for zero-trust pinning of a
//! known peer certificate.
//!
//! TLS client requires explicit trust configuration, either a CA certificate
//! (ca_cert_pem) for chain verification or a SHA-256 server cert fingerprint
//! (fingerprint_pin) for pinned verification. There is no implicit system or
//! Mozilla trust anchor.

use std::io;
use std::path::PathBuf;
use std::sync::Arc;

use sha2::{Digest, Sha256};
use tokio::io::{AsyncRead, AsyncWrite};

use std::sync::Once;

use rustls::client::danger::{HandshakeSignatureValid, ServerCertVerified, ServerCertVerifier};
use rustls::pki_types::{CertificateDer, PrivateKeyDer, ServerName, UnixTime};
use rustls::server::WebPkiClientVerifier;
use rustls::{ClientConfig, DigitallySignedStruct, RootCertStore, ServerConfig, SignatureScheme};

// ----------------------------------------------------------------------------
// Public configuration types
// ----------------------------------------------------------------------------

/// TLS version floor for a server or client.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TlsVersion {
    Tls12,
    Tls13,
}

/// Per-listener TLS configuration for the server.
#[derive(Debug, Clone)]
pub struct TlsConfig {
    pub cert_pem_path: Option<PathBuf>,
    pub key_pem_path: Option<PathBuf>,
    pub client_ca_pem_path: Option<PathBuf>,
    pub require_client_cert: bool,
    pub min_version: TlsVersion,
}

impl Default for TlsConfig {
    fn default() -> Self {
        Self {
            cert_pem_path: None,
            key_pem_path: None,
            client_ca_pem_path: None,
            require_client_cert: false,
            min_version: TlsVersion::Tls13,
        }
    }
}

/// How the server treats SSL negotiation requests from clients.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TlsMode {
    /// Reply with `N` to SslRequest. Plaintext sessions only.
    Disabled,
    /// Reply with `S` and perform a TLS handshake if the client asks.
    Optional,
    /// Reject plaintext startup. Require the TLS upgrade path.
    Required,
}

/// Errors produced by the TLS setup and handshake layer.
#[derive(Debug, thiserror::Error)]
pub enum TlsError {
    #[error("IO error: {0}")]
    Io(#[from] io::Error),
    #[error("TLS handshake failed: {0}")]
    Handshake(String),
    #[error("TLS configuration error: {0}")]
    Config(String),
    #[error("Certificate pin mismatch: expected {expected}, got {actual}")]
    PinMismatch { expected: String, actual: String },
}

pub type Result<T> = std::result::Result<T, TlsError>;

/// Installs the ring-based rustls CryptoProvider once per process. Rustls
/// 0.23 requires an explicit provider when multiple feature-backed providers
/// are possible. The installation is idempotent and thread-safe.
pub fn install_default_crypto_provider() {
    static ONCE: Once = Once::new();
    ONCE.call_once(|| {
        let _ = rustls::crypto::ring::default_provider().install_default();
    });
}

// ----------------------------------------------------------------------------
// Server acceptor
// ----------------------------------------------------------------------------

/// Accepts incoming TLS connections using a preloaded rustls ServerConfig.
#[derive(Clone)]
pub struct ServerTlsAcceptor {
    acceptor: tokio_rustls::TlsAcceptor,
}

impl ServerTlsAcceptor {
    /// Builds an acceptor from on-disk PEM certificate and key files.
    pub fn from_config(config: &TlsConfig) -> Result<Self> {
        install_default_crypto_provider();
        let cert_path = config
            .cert_pem_path
            .as_ref()
            .ok_or_else(|| TlsError::Config("cert_pem_path is required".into()))?;
        let key_path = config
            .key_pem_path
            .as_ref()
            .ok_or_else(|| TlsError::Config("key_pem_path is required".into()))?;

        let certs = load_certs(cert_path)?;
        let key = load_private_key(key_path)?;

        let builder = match config.min_version {
            TlsVersion::Tls13 => {
                ServerConfig::builder_with_protocol_versions(&[&rustls::version::TLS13])
            }
            TlsVersion::Tls12 => ServerConfig::builder(),
        };

        let builder = if let Some(ca_path) = &config.client_ca_pem_path {
            let roots = load_root_store(ca_path)?;
            let verifier = if config.require_client_cert {
                WebPkiClientVerifier::builder(Arc::new(roots))
                    .build()
                    .map_err(|e| TlsError::Config(format!("client verifier: {}", e)))?
            } else {
                WebPkiClientVerifier::builder(Arc::new(roots))
                    .allow_unauthenticated()
                    .build()
                    .map_err(|e| TlsError::Config(format!("client verifier: {}", e)))?
            };
            builder.with_client_cert_verifier(verifier)
        } else {
            builder.with_no_client_auth()
        };

        let server_config = builder
            .with_single_cert(certs, key)
            .map_err(|e| TlsError::Config(format!("server cert/key: {}", e)))?;

        Ok(Self {
            acceptor: tokio_rustls::TlsAcceptor::from(Arc::new(server_config)),
        })
    }

    /// Wraps a plain stream in a TLS session by performing the handshake.
    pub async fn accept<S>(&self, stream: S) -> Result<tokio_rustls::server::TlsStream<S>>
    where
        S: AsyncRead + AsyncWrite + Unpin,
    {
        self.acceptor
            .accept(stream)
            .await
            .map_err(|e| TlsError::Handshake(e.to_string()))
    }
}

// ----------------------------------------------------------------------------
// Client connector
// ----------------------------------------------------------------------------

/// Connects to a TLS peer with optional custom CAs, client certs, and
/// fingerprint pinning.
#[derive(Clone)]
pub struct ClientTlsConnector {
    connector: tokio_rustls::TlsConnector,
    server_name: ServerName<'static>,
}

impl ClientTlsConnector {
    /// Creates a client connector. TLS client requires explicit trust
    /// configuration, either a CA certificate (`ca_cert_pem`) for chain
    /// verification or a SHA-256 server cert fingerprint (`fingerprint_pin`)
    /// for pinned verification. Both may be provided, in which case the chain
    /// is verified first and the leaf fingerprint is matched against the pin.
    /// If `client_cert_and_key` is supplied the connector performs mTLS.
    pub fn new(
        server_name: &str,
        ca_cert_pem: Option<&[u8]>,
        client_cert_and_key: Option<(&[u8], &[u8])>,
        fingerprint_pin: Option<[u8; 32]>,
    ) -> Result<Self> {
        install_default_crypto_provider();
        if ca_cert_pem.is_none() && fingerprint_pin.is_none() {
            return Err(TlsError::Config(
                "TLS client requires either ca_cert_pem or fingerprint_pin to be provided".into(),
            ));
        }

        let mut roots = RootCertStore::empty();
        if let Some(ca_bytes) = ca_cert_pem {
            for cert in parse_certs(ca_bytes)? {
                roots
                    .add(cert)
                    .map_err(|e| TlsError::Config(format!("ca add: {}", e)))?;
            }
        }

        let builder = ClientConfig::builder_with_protocol_versions(&[&rustls::version::TLS13]);

        let client_config = if let Some(pin) = fingerprint_pin {
            // Two paths: with a CA, chain through webpki and then match the
            // leaf fingerprint. Without a CA, skip chain verification and
            // rely on the fingerprint alone.
            let verifier: Arc<dyn ServerCertVerifier> = if ca_cert_pem.is_some() {
                let inner = rustls::client::WebPkiServerVerifier::builder(Arc::new(roots.clone()))
                    .build()
                    .map_err(|e| TlsError::Config(format!("server verifier: {}", e)))?;
                Arc::new(PinnedServerVerifier {
                    inner: Some(inner),
                    pin,
                })
            } else {
                Arc::new(PinnedServerVerifier { inner: None, pin })
            };
            let builder = builder
                .dangerous()
                .with_custom_certificate_verifier(verifier);
            match client_cert_and_key {
                Some((cert_pem, key_pem)) => {
                    let certs = parse_certs(cert_pem)?;
                    let key = parse_private_key(key_pem)?;
                    builder
                        .with_client_auth_cert(certs, key)
                        .map_err(|e| TlsError::Config(format!("client cert: {}", e)))?
                }
                None => builder.with_no_client_auth(),
            }
        } else {
            let builder = builder.with_root_certificates(roots);
            match client_cert_and_key {
                Some((cert_pem, key_pem)) => {
                    let certs = parse_certs(cert_pem)?;
                    let key = parse_private_key(key_pem)?;
                    builder
                        .with_client_auth_cert(certs, key)
                        .map_err(|e| TlsError::Config(format!("client cert: {}", e)))?
                }
                None => builder.with_no_client_auth(),
            }
        };

        let server_name = ServerName::try_from(server_name.to_string())
            .map_err(|e| TlsError::Config(format!("server name: {}", e)))?;

        Ok(Self {
            connector: tokio_rustls::TlsConnector::from(Arc::new(client_config)),
            server_name,
        })
    }

    /// Performs the client handshake over the given stream.
    pub async fn connect<S>(&self, stream: S) -> Result<tokio_rustls::client::TlsStream<S>>
    where
        S: AsyncRead + AsyncWrite + Unpin,
    {
        self.connector
            .connect(self.server_name.clone(), stream)
            .await
            .map_err(|e| TlsError::Handshake(e.to_string()))
    }

    /// Returns the server name used for SNI and hostname verification.
    pub fn server_name(&self) -> &ServerName<'static> {
        &self.server_name
    }
}

// ----------------------------------------------------------------------------
// Fingerprint pinning verifier
// ----------------------------------------------------------------------------

/// Verifies the leaf certificate's SHA-256 fingerprint against a preshared
/// value. If `inner` is set, the chain is verified with webpki first, then
/// the fingerprint is matched. If `inner` is None, the fingerprint is the
/// sole trust anchor and signatures are accepted at the default rustls
/// crypto provider's supported schemes.
#[derive(Debug)]
struct PinnedServerVerifier {
    inner: Option<Arc<rustls::client::WebPkiServerVerifier>>,
    pin: [u8; 32],
}

impl ServerCertVerifier for PinnedServerVerifier {
    fn verify_server_cert(
        &self,
        end_entity: &CertificateDer<'_>,
        intermediates: &[CertificateDer<'_>],
        server_name: &ServerName<'_>,
        ocsp: &[u8],
        now: UnixTime,
    ) -> std::result::Result<ServerCertVerified, rustls::Error> {
        if let Some(inner) = &self.inner {
            inner.verify_server_cert(end_entity, intermediates, server_name, ocsp, now)?;
        }

        let mut hasher = Sha256::new();
        hasher.update(end_entity.as_ref());
        let fp: [u8; 32] = hasher.finalize().into();
        if fp != self.pin {
            return Err(rustls::Error::General(format!(
                "certificate fingerprint mismatch: got {}",
                hex_encode(&fp)
            )));
        }
        Ok(ServerCertVerified::assertion())
    }

    fn verify_tls12_signature(
        &self,
        message: &[u8],
        cert: &CertificateDer<'_>,
        dss: &DigitallySignedStruct,
    ) -> std::result::Result<HandshakeSignatureValid, rustls::Error> {
        match &self.inner {
            Some(inner) => inner.verify_tls12_signature(message, cert, dss),
            None => default_verify_tls12_signature(message, cert, dss),
        }
    }

    fn verify_tls13_signature(
        &self,
        message: &[u8],
        cert: &CertificateDer<'_>,
        dss: &DigitallySignedStruct,
    ) -> std::result::Result<HandshakeSignatureValid, rustls::Error> {
        match &self.inner {
            Some(inner) => inner.verify_tls13_signature(message, cert, dss),
            None => default_verify_tls13_signature(message, cert, dss),
        }
    }

    fn supported_verify_schemes(&self) -> Vec<SignatureScheme> {
        match &self.inner {
            Some(inner) => inner.supported_verify_schemes(),
            None => rustls::crypto::ring::default_provider()
                .signature_verification_algorithms
                .supported_schemes(),
        }
    }
}

// -----------------------------------------------------------------------------
// Fingerprint-only signature verification
// -----------------------------------------------------------------------------
//
// When the client has no CA chain and is trusting solely by SHA-256 pin of
// the leaf cert, rustls still requires valid handshake signatures from the
// leaf public key. These helpers use the ring crypto provider's registered
// signature algorithms to verify the handshake messages.

fn default_verify_tls12_signature(
    message: &[u8],
    cert: &CertificateDer<'_>,
    dss: &DigitallySignedStruct,
) -> std::result::Result<HandshakeSignatureValid, rustls::Error> {
    rustls::crypto::verify_tls12_signature(
        message,
        cert,
        dss,
        &rustls::crypto::ring::default_provider().signature_verification_algorithms,
    )
}

fn default_verify_tls13_signature(
    message: &[u8],
    cert: &CertificateDer<'_>,
    dss: &DigitallySignedStruct,
) -> std::result::Result<HandshakeSignatureValid, rustls::Error> {
    rustls::crypto::verify_tls13_signature(
        message,
        cert,
        dss,
        &rustls::crypto::ring::default_provider().signature_verification_algorithms,
    )
}

// ----------------------------------------------------------------------------
// PEM helpers
// ----------------------------------------------------------------------------

fn load_certs(path: &PathBuf) -> Result<Vec<CertificateDer<'static>>> {
    let bytes = std::fs::read(path)?;
    parse_certs(&bytes)
}

fn parse_certs(bytes: &[u8]) -> Result<Vec<CertificateDer<'static>>> {
    crate::pem::parse_certificates(bytes).map_err(|e| TlsError::Config(format!("cert pem: {}", e)))
}

fn load_private_key(path: &PathBuf) -> Result<PrivateKeyDer<'static>> {
    let bytes = std::fs::read(path)?;
    parse_private_key(&bytes)
}

fn parse_private_key(bytes: &[u8]) -> Result<PrivateKeyDer<'static>> {
    crate::pem::parse_private_key(bytes).map_err(|e| TlsError::Config(format!("key pem: {}", e)))
}

fn load_root_store(path: &PathBuf) -> Result<RootCertStore> {
    let certs = load_certs(path)?;
    let mut store = RootCertStore::empty();
    for cert in certs {
        store
            .add(cert)
            .map_err(|e| TlsError::Config(format!("root store: {}", e)))?;
    }
    Ok(store)
}

fn hex_encode(bytes: &[u8]) -> String {
    let mut out = String::with_capacity(bytes.len() * 2);
    for b in bytes {
        out.push_str(&format!("{:02x}", b));
    }
    out
}

/// Computes the SHA-256 fingerprint of a DER-encoded certificate.
pub fn sha256_fingerprint(der: &[u8]) -> [u8; 32] {
    let mut hasher = Sha256::new();
    hasher.update(der);
    hasher.finalize().into()
}

// ----------------------------------------------------------------------------
// Tests
// ----------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use rcgen::{CertifiedKey, generate_simple_self_signed};

    fn make_self_signed() -> (Vec<u8>, Vec<u8>, [u8; 32]) {
        let CertifiedKey { cert, key_pair, .. } =
            generate_simple_self_signed(vec!["localhost".to_string()]).unwrap();
        let cert_pem = cert.pem().into_bytes();
        let key_pem = key_pair.serialize_pem().into_bytes();
        let fp = sha256_fingerprint(cert.der().as_ref());
        (cert_pem, key_pem, fp)
    }

    #[test]
    fn test_parse_certs_roundtrip() {
        let (cert_pem, _key, _fp) = make_self_signed();
        let certs = parse_certs(&cert_pem).unwrap();
        assert_eq!(certs.len(), 1);
    }

    #[test]
    fn test_parse_private_key_ok() {
        let (_cert, key_pem, _fp) = make_self_signed();
        let _ = parse_private_key(&key_pem).unwrap();
    }

    #[test]
    fn test_parse_certs_empty_errors() {
        assert!(parse_certs(b"not a pem").is_err());
    }

    #[test]
    fn test_sha256_fingerprint_stable() {
        let fp1 = sha256_fingerprint(b"hello");
        let fp2 = sha256_fingerprint(b"hello");
        assert_eq!(fp1, fp2);
        assert_ne!(fp1, sha256_fingerprint(b"world"));
    }

    #[test]
    fn test_hex_encode_format() {
        assert_eq!(hex_encode(&[0x00, 0xff, 0xab]), "00ffab");
    }

    #[tokio::test]
    async fn test_tls_roundtrip_self_signed() {
        let (cert_pem, key_pem, fp) = make_self_signed();

        // Write PEMs to a temp directory for the server path.
        let dir = tempfile::tempdir().unwrap();
        let cert_path = dir.path().join("cert.pem");
        let key_path = dir.path().join("key.pem");
        std::fs::write(&cert_path, &cert_pem).unwrap();
        std::fs::write(&key_path, &key_pem).unwrap();

        let server_cfg = TlsConfig {
            cert_pem_path: Some(cert_path),
            key_pem_path: Some(key_path),
            client_ca_pem_path: None,
            require_client_cert: false,
            min_version: TlsVersion::Tls13,
        };
        let acceptor = ServerTlsAcceptor::from_config(&server_cfg).unwrap();
        let client = ClientTlsConnector::new("localhost", Some(&cert_pem), None, Some(fp)).unwrap();

        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();

        let server_task = tokio::spawn(async move {
            let (stream, _) = listener.accept().await.unwrap();
            let mut tls = acceptor.accept(stream).await.unwrap();
            use tokio::io::{AsyncReadExt, AsyncWriteExt};
            let mut buf = [0u8; 4];
            tls.read_exact(&mut buf).await.unwrap();
            tls.write_all(b"pong").await.unwrap();
            tls.flush().await.unwrap();
            buf
        });

        let stream = tokio::net::TcpStream::connect(addr).await.unwrap();
        let mut tls = client.connect(stream).await.unwrap();
        use tokio::io::{AsyncReadExt, AsyncWriteExt};
        tls.write_all(b"ping").await.unwrap();
        tls.flush().await.unwrap();
        let mut resp = [0u8; 4];
        tls.read_exact(&mut resp).await.unwrap();
        assert_eq!(&resp, b"pong");
        let got = server_task.await.unwrap();
        assert_eq!(&got, b"ping");
    }

    #[tokio::test]
    async fn test_tls_fingerprint_mismatch_rejected() {
        let (cert_pem, key_pem, _fp) = make_self_signed();
        let dir = tempfile::tempdir().unwrap();
        let cert_path = dir.path().join("cert.pem");
        let key_path = dir.path().join("key.pem");
        std::fs::write(&cert_path, &cert_pem).unwrap();
        std::fs::write(&key_path, &key_pem).unwrap();

        let acceptor = ServerTlsAcceptor::from_config(&TlsConfig {
            cert_pem_path: Some(cert_path),
            key_pem_path: Some(key_path),
            client_ca_pem_path: None,
            require_client_cert: false,
            min_version: TlsVersion::Tls13,
        })
        .unwrap();

        let bad_pin = [0u8; 32];
        let client =
            ClientTlsConnector::new("localhost", Some(&cert_pem), None, Some(bad_pin)).unwrap();

        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();

        tokio::spawn(async move {
            if let Ok((stream, _)) = listener.accept().await {
                let _ = acceptor.accept(stream).await;
            }
        });

        let stream = tokio::net::TcpStream::connect(addr).await.unwrap();
        let result = client.connect(stream).await;
        assert!(result.is_err());
    }

    #[test]
    fn test_missing_cert_path_errors() {
        match ServerTlsAcceptor::from_config(&TlsConfig::default()) {
            Err(TlsError::Config(_)) => {}
            Err(other) => panic!("expected Config error, got {:?}", other),
            Ok(_) => panic!("expected error"),
        }
    }

    #[test]
    fn client_tls_rejects_without_trust_config() {
        match ClientTlsConnector::new("localhost", None, None, None) {
            Err(TlsError::Config(msg)) => {
                assert!(msg.contains("ca_cert_pem"));
                assert!(msg.contains("fingerprint_pin"));
            }
            Err(other) => panic!("expected Config error, got {:?}", other),
            Ok(_) => panic!("expected error"),
        }
    }

    #[tokio::test]
    async fn client_tls_ok_with_ca_cert() {
        let (cert_pem, key_pem, _fp) = make_self_signed();
        let dir = tempfile::tempdir().unwrap();
        let cert_path = dir.path().join("cert.pem");
        let key_path = dir.path().join("key.pem");
        std::fs::write(&cert_path, &cert_pem).unwrap();
        std::fs::write(&key_path, &key_pem).unwrap();

        let acceptor = ServerTlsAcceptor::from_config(&TlsConfig {
            cert_pem_path: Some(cert_path),
            key_pem_path: Some(key_path),
            client_ca_pem_path: None,
            require_client_cert: false,
            min_version: TlsVersion::Tls13,
        })
        .unwrap();
        let client = ClientTlsConnector::new("localhost", Some(&cert_pem), None, None).unwrap();

        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();

        let server_task = tokio::spawn(async move {
            let (stream, _) = listener.accept().await.unwrap();
            let mut tls = acceptor.accept(stream).await.unwrap();
            use tokio::io::{AsyncReadExt, AsyncWriteExt};
            let mut buf = [0u8; 2];
            tls.read_exact(&mut buf).await.unwrap();
            tls.write_all(b"ok").await.unwrap();
            tls.flush().await.unwrap();
        });

        let stream = tokio::net::TcpStream::connect(addr).await.unwrap();
        let mut tls = client.connect(stream).await.unwrap();
        use tokio::io::{AsyncReadExt, AsyncWriteExt};
        tls.write_all(b"hi").await.unwrap();
        tls.flush().await.unwrap();
        let mut resp = [0u8; 2];
        tls.read_exact(&mut resp).await.unwrap();
        assert_eq!(&resp, b"ok");
        server_task.await.unwrap();
    }

    #[tokio::test]
    async fn client_tls_ok_with_fingerprint_pin() {
        let (cert_pem, key_pem, fp) = make_self_signed();
        let dir = tempfile::tempdir().unwrap();
        let cert_path = dir.path().join("cert.pem");
        let key_path = dir.path().join("key.pem");
        std::fs::write(&cert_path, &cert_pem).unwrap();
        std::fs::write(&key_path, &key_pem).unwrap();

        let acceptor = ServerTlsAcceptor::from_config(&TlsConfig {
            cert_pem_path: Some(cert_path),
            key_pem_path: Some(key_path),
            client_ca_pem_path: None,
            require_client_cert: false,
            min_version: TlsVersion::Tls13,
        })
        .unwrap();
        // Fingerprint-only trust, no CA.
        let client = ClientTlsConnector::new("localhost", None, None, Some(fp)).unwrap();

        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();

        let server_task = tokio::spawn(async move {
            let (stream, _) = listener.accept().await.unwrap();
            let mut tls = acceptor.accept(stream).await.unwrap();
            use tokio::io::{AsyncReadExt, AsyncWriteExt};
            let mut buf = [0u8; 3];
            tls.read_exact(&mut buf).await.unwrap();
            tls.write_all(b"pin").await.unwrap();
            tls.flush().await.unwrap();
        });

        let stream = tokio::net::TcpStream::connect(addr).await.unwrap();
        let mut tls = client.connect(stream).await.unwrap();
        use tokio::io::{AsyncReadExt, AsyncWriteExt};
        tls.write_all(b"hey").await.unwrap();
        tls.flush().await.unwrap();
        let mut resp = [0u8; 3];
        tls.read_exact(&mut resp).await.unwrap();
        assert_eq!(&resp, b"pin");
        server_task.await.unwrap();
    }
}
