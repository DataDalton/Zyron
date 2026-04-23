// -----------------------------------------------------------------------------
// PEM parsing
// -----------------------------------------------------------------------------
//
// Minimal PEM (RFC 7468) parser. Handles BEGIN/END markers, whitespace
// between, base64-decodes the body. Supports the three private-key shapes
// rustls accepts: PKCS#8, PKCS#1 (RSA), and SEC1 (EC).

use base64::Engine;
use rustls::pki_types::{
    CertificateDer, PrivateKeyDer, PrivatePkcs1KeyDer, PrivatePkcs8KeyDer, PrivateSec1KeyDer,
};
use zyron_common::{Result, ZyronError};

pub fn parse_certificates(pem: &[u8]) -> Result<Vec<CertificateDer<'static>>> {
    let text = std::str::from_utf8(pem)
        .map_err(|_| ZyronError::Internal("pem input is not valid utf-8".to_string()))?;
    let mut out = Vec::new();
    let mut cursor = text;
    while let Some((der, rest)) = next_block(cursor, "CERTIFICATE")? {
        out.push(CertificateDer::from(der));
        cursor = rest;
    }
    if out.is_empty() {
        return Err(ZyronError::Internal(
            "no CERTIFICATE blocks found in pem input".to_string(),
        ));
    }
    Ok(out)
}

pub fn parse_private_key(pem: &[u8]) -> Result<PrivateKeyDer<'static>> {
    let text = std::str::from_utf8(pem)
        .map_err(|_| ZyronError::Internal("pem input is not valid utf-8".to_string()))?;
    // Try each kind in turn. Return the first match.
    if let Some((der, _)) = next_block(text, "PRIVATE KEY")? {
        return Ok(PrivateKeyDer::Pkcs8(PrivatePkcs8KeyDer::from(der)));
    }
    if let Some((der, _)) = next_block(text, "RSA PRIVATE KEY")? {
        return Ok(PrivateKeyDer::Pkcs1(PrivatePkcs1KeyDer::from(der)));
    }
    if let Some((der, _)) = next_block(text, "EC PRIVATE KEY")? {
        return Ok(PrivateKeyDer::Sec1(PrivateSec1KeyDer::from(der)));
    }
    Err(ZyronError::Internal(
        "no PRIVATE KEY block found in pem input".to_string(),
    ))
}

/// Pulls the next block of the named label from the input, returning the
/// DER bytes plus the remaining input after the end marker. Returns None
/// when no more matching blocks remain.
fn next_block<'a>(text: &'a str, label: &str) -> Result<Option<(Vec<u8>, &'a str)>> {
    let begin_marker = format!("-----BEGIN {}-----", label);
    let end_marker = format!("-----END {}-----", label);
    let Some(begin_at) = text.find(&begin_marker) else {
        return Ok(None);
    };
    let body_start = begin_at + begin_marker.len();
    let Some(end_rel) = text[body_start..].find(&end_marker) else {
        return Err(ZyronError::Internal(format!(
            "pem block is missing closing {end_marker}"
        )));
    };
    let body = &text[body_start..body_start + end_rel];
    let after = &text[body_start + end_rel + end_marker.len()..];
    // Strip whitespace.
    let compact: String = body.chars().filter(|c| !c.is_whitespace()).collect();
    let der = base64::engine::general_purpose::STANDARD
        .decode(&compact)
        .map_err(|e| ZyronError::Internal(format!("pem base64 decode failed: {e}")))?;
    Ok(Some((der, after)))
}

// -----------------------------------------------------------------------------
// Tests
// -----------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    const TEST_CERT_PEM: &str = "-----BEGIN CERTIFICATE-----\nMIIBkTCCATegAwIBAgIUdGVzdDEwMjMgZm9yIHp5cm9uREIwCgYIKoZIzj0EAwIw\nEjEQMA4GA1UEAxMHdGVzdC1jYTAeFw0yNDAxMDEwMDAwMDBaFw0zNDAxMDEwMDAw\nMDBaMBIxEDAOBgNVBAMTB3Rlc3QtY2EwWTATBgcqhkjOPQIBBggqhkjOPQMBBwNC\nAAQz2XcYqdy9RVfH4JZTPRcGZt+vAJqmjJOWzlj8sYVaaMR88xbH4XBT5HCZ4hJT\ntoPQf+gLw8gxmZ8wOj8nLLGXo1MwUTAdBgNVHQ4EFgQUIm+a/2Gp3V/u7Nxa9JkP\nSNGXwIkwHwYDVR0jBBgwFoAUIm+a/2Gp3V/u7Nxa9JkPSNGXwIkwDwYDVR0TAQH/\nBAUwAwEB/zAKBggqhkjOPQQDAgNIADBFAiEA1ks1xBAgQWvOR2UZZE0QhE2uqHLR\nALHddXdqjXY6FYUCIH/QI6Al6BRRRQYx5JVgP+OfXsxwTQbVwh08sFvcI5lz\n-----END CERTIFICATE-----\n";

    #[test]
    fn parse_single_cert_block() {
        let certs = parse_certificates(TEST_CERT_PEM.as_bytes()).unwrap();
        assert_eq!(certs.len(), 1);
    }

    #[test]
    fn parse_empty_input_errors() {
        assert!(parse_certificates(b"").is_err());
    }

    #[test]
    fn parse_malformed_input_errors() {
        assert!(parse_certificates(b"-----BEGIN CERTIFICATE-----\ngarbage").is_err());
    }

    #[test]
    fn rcgen_roundtrip_cert() {
        let cert = rcgen::generate_simple_self_signed(vec!["test.local".to_string()]).unwrap();
        let pem = cert.cert.pem();
        let parsed = parse_certificates(pem.as_bytes()).unwrap();
        assert_eq!(parsed.len(), 1);
        assert_eq!(parsed[0].as_ref(), cert.cert.der().as_ref());
    }

    #[test]
    fn rcgen_roundtrip_pkcs8_key() {
        let cert = rcgen::generate_simple_self_signed(vec!["test.local".to_string()]).unwrap();
        let pem = cert.key_pair.serialize_pem();
        let parsed = parse_private_key(pem.as_bytes()).unwrap();
        // ECDSA keys come out as PKCS#8 by default from rcgen.
        assert!(matches!(parsed, PrivateKeyDer::Pkcs8(_)));
    }
}
