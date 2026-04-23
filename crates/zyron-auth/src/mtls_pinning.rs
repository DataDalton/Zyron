//! mTLS certificate fingerprint pinning plus CRL-file-based revocation checks.
//!
//! Pinning binds a subject identifier (typically a role ID or user ID) to one
//! or more SHA-256 fingerprints of the peer's DER-encoded certificate. A
//! mismatch aborts the handshake. A revocation check loads a DER-encoded CRL
//! from disk and matches the peer certificate serial number against it. OCSP
//! stapling is intentionally deferred per the P1 scope note.

use std::collections::HashMap;
use std::path::Path;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Duration;

use parking_lot::RwLock;
use sha2::{Digest, Sha256};
use zyron_common::{Result, ZyronError};

// -----------------------------------------------------------------------------
// Fingerprints
// -----------------------------------------------------------------------------

/// A SHA-256 fingerprint of a DER-encoded X.509 certificate.
pub type Sha256Fingerprint = [u8; 32];

/// Computes the SHA-256 fingerprint of a DER-encoded certificate.
pub fn fingerprint_of(cert_der: &[u8]) -> Sha256Fingerprint {
    let mut hasher = Sha256::new();
    hasher.update(cert_der);
    let out = hasher.finalize();
    let mut fp = [0u8; 32];
    fp.copy_from_slice(&out);
    fp
}

/// Parses a `sha256:<hex>` pin expression into a fingerprint.
pub fn parse_pin(pin: &str) -> Result<Sha256Fingerprint> {
    let s = pin.strip_prefix("sha256:").ok_or_else(|| {
        ZyronError::InvalidCredential(format!("pin must start with sha256: got {}", pin))
    })?;
    if s.len() != 64 {
        return Err(ZyronError::InvalidCredential(format!(
            "pin hex must be 64 chars, got {}",
            s.len()
        )));
    }
    let mut out = [0u8; 32];
    for (i, chunk) in s.as_bytes().chunks(2).enumerate() {
        let pair = std::str::from_utf8(chunk)
            .map_err(|_| ZyronError::InvalidCredential("pin hex not ASCII".to_string()))?;
        out[i] = u8::from_str_radix(pair, 16)
            .map_err(|_| ZyronError::InvalidCredential(format!("pin hex invalid at byte {}", i)))?;
    }
    Ok(out)
}

/// Formats a fingerprint as `sha256:<hex>`.
pub fn format_pin(fp: &Sha256Fingerprint) -> String {
    let mut s = String::with_capacity(7 + 64);
    s.push_str("sha256:");
    for b in fp {
        s.push_str(&format!("{:02x}", b));
    }
    s
}

// -----------------------------------------------------------------------------
// Fingerprint store
// -----------------------------------------------------------------------------

/// Pinned cert fingerprints keyed by subject id.
pub struct CertFingerprintStore {
    pinned: scc::HashMap<u32, Vec<Sha256Fingerprint>>,
}

impl CertFingerprintStore {
    pub fn new() -> Self {
        Self {
            pinned: scc::HashMap::new(),
        }
    }

    /// Pins a fingerprint for a subject. Multiple pins per subject are
    /// permitted to support cert rotation windows.
    pub fn pin(&self, subject_id: u32, fingerprint: Sha256Fingerprint) {
        let existing = self.pinned.read_sync(&subject_id, |_, v| v.clone());
        match existing {
            Some(mut list) => {
                if !list.contains(&fingerprint) {
                    list.push(fingerprint);
                }
                let _ = self.pinned.remove_sync(&subject_id);
                let _ = self.pinned.insert_sync(subject_id, list);
            }
            None => {
                let _ = self.pinned.insert_sync(subject_id, vec![fingerprint]);
            }
        }
    }

    /// Verifies that the DER-encoded peer cert matches one of the pinned
    /// fingerprints for the subject. Returns an error on mismatch.
    pub fn verify(&self, subject_id: u32, observed_cert_der: &[u8]) -> Result<()> {
        let observed = fingerprint_of(observed_cert_der);
        let matched = self
            .pinned
            .read_sync(&subject_id, |_, v| v.iter().any(|p| p == &observed));
        match matched {
            Some(true) => Ok(()),
            Some(false) => Err(ZyronError::AuthenticationFailed(format!(
                "mTLS fingerprint mismatch for subject {}",
                subject_id
            ))),
            None => Err(ZyronError::AuthenticationFailed(format!(
                "no pinned fingerprints for subject {}",
                subject_id
            ))),
        }
    }

    /// Removes a fingerprint pin. If the subject has no more pins, the entry
    /// is removed.
    pub fn unpin(&self, subject_id: u32, fingerprint: Sha256Fingerprint) -> bool {
        let existing = self.pinned.read_sync(&subject_id, |_, v| v.clone());
        let Some(mut list) = existing else {
            return false;
        };
        let before = list.len();
        list.retain(|fp| fp != &fingerprint);
        let removed = list.len() < before;
        let _ = self.pinned.remove_sync(&subject_id);
        if !list.is_empty() {
            let _ = self.pinned.insert_sync(subject_id, list);
        }
        removed
    }

    /// Returns the pinned fingerprints for a subject (if any).
    pub fn pins_for(&self, subject_id: u32) -> Vec<Sha256Fingerprint> {
        self.pinned
            .read_sync(&subject_id, |_, v| v.clone())
            .unwrap_or_default()
    }

    /// Returns the current pin count for all subjects.
    pub fn total_pins(&self) -> usize {
        let mut n = 0;
        self.pinned.iter_sync(|_, v| {
            n += v.len();
            true
        });
        n
    }
}

impl Default for CertFingerprintStore {
    fn default() -> Self {
        Self::new()
    }
}

// -----------------------------------------------------------------------------
// Revocation
// -----------------------------------------------------------------------------

/// The result of checking a certificate against the configured revocation
/// source.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RevocationStatus {
    Good,
    Revoked { reason: String, revoked_at: u64 },
    Unknown,
}

#[derive(Debug, Clone)]
pub struct CrlStatus {
    pub revoked_serials: Vec<Vec<u8>>,
    pub loaded_at: u64,
}

/// Loads CRL serial-number lists from disk. If a serial appears on any loaded
/// CRL, the cert is considered revoked. OCSP over HTTP is deferred, but
/// `ocsp_responder_url` is retained so callers can configure it today and have
/// it take effect when the OCSP path is added.
pub struct CrlOcspChecker {
    ocsp_responder_url: Option<String>,
    crl_cache: RwLock<HashMap<String, CrlStatus>>,
    http_client: reqwest::Client,
    checks_total: AtomicU64,
    revoked_total: AtomicU64,
}

impl CrlOcspChecker {
    pub fn new(ocsp_responder_url: Option<String>) -> Result<Self> {
        let http_client = reqwest::Client::builder()
            .timeout(Duration::from_secs(10))
            .build()
            .map_err(|e| {
                ZyronError::InvalidCredential(format!("CRL/OCSP http client build failed: {}", e))
            })?;
        Ok(Self {
            ocsp_responder_url,
            crl_cache: RwLock::new(HashMap::new()),
            http_client,
            checks_total: AtomicU64::new(0),
            revoked_total: AtomicU64::new(0),
        })
    }

    /// Loads a CRL from disk by parsing the list of revoked serial numbers
    /// from a simple on-disk format: one hex-encoded serial per line.
    ///
    /// The parser accepts `#`-prefixed comment lines and blank lines. This
    /// intentionally avoids pulling in a full DER parser while still covering
    /// the common need, which is to mark known-bad serials.
    pub fn load_crl_file(&self, name: &str, path: impl AsRef<Path>) -> Result<usize> {
        let text = std::fs::read_to_string(path.as_ref()).map_err(|e| {
            ZyronError::InvalidCredential(format!(
                "CRL file read failed at {}: {}",
                path.as_ref().display(),
                e
            ))
        })?;
        let mut serials = Vec::new();
        for line in text.lines() {
            let l = line.trim();
            if l.is_empty() || l.starts_with('#') {
                continue;
            }
            let hex = l.strip_prefix("0x").unwrap_or(l);
            let bytes = decode_hex(hex).map_err(|e| {
                ZyronError::InvalidCredential(format!("CRL invalid hex serial: {}", e))
            })?;
            serials.push(bytes);
        }
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0);
        let count = serials.len();
        self.crl_cache.write().insert(
            name.to_string(),
            CrlStatus {
                revoked_serials: serials,
                loaded_at: now,
            },
        );
        Ok(count)
    }

    /// Returns the revocation status for a certificate with the given DER
    /// encoding. The issuer DER is accepted for API symmetry but not used
    /// because the local CRL format keys on serial only.
    pub async fn check_revocation(
        &self,
        cert_der: &[u8],
        _issuer_der: &[u8],
    ) -> Result<RevocationStatus> {
        self.checks_total.fetch_add(1, Ordering::Relaxed);
        let serial = match extract_serial(cert_der) {
            Some(s) => s,
            None => return Ok(RevocationStatus::Unknown),
        };
        let cache = self.crl_cache.read();
        for status in cache.values() {
            if status.revoked_serials.iter().any(|s| s == &serial) {
                self.revoked_total.fetch_add(1, Ordering::Relaxed);
                return Ok(RevocationStatus::Revoked {
                    reason: "serial listed in CRL".to_string(),
                    revoked_at: status.loaded_at,
                });
            }
        }
        Ok(RevocationStatus::Good)
    }

    pub fn ocsp_responder_url(&self) -> Option<&str> {
        self.ocsp_responder_url.as_deref()
    }

    pub fn checks_total(&self) -> u64 {
        self.checks_total.load(Ordering::Relaxed)
    }

    pub fn revoked_total(&self) -> u64 {
        self.revoked_total.load(Ordering::Relaxed)
    }

    pub fn http_client(&self) -> &reqwest::Client {
        &self.http_client
    }
}

// -----------------------------------------------------------------------------
// DER serial extraction
// -----------------------------------------------------------------------------

/// Extracts the serial number from a DER-encoded X.509 certificate.
///
/// The full cert is `SEQUENCE { tbsCertificate, signatureAlgorithm, signature }`.
/// `tbsCertificate` starts with an optional `[0]` version tag, then
/// `CertificateSerialNumber` which is an INTEGER. If the version tag is
/// absent, the serial is the first element of the inner sequence.
fn extract_serial(der: &[u8]) -> Option<Vec<u8>> {
    let (outer_content, _) = parse_der_sequence(der)?;
    let (tbs_content, _) = parse_der_sequence(outer_content)?;
    // Optional version: CONTEXT-SPECIFIC [0] tag = 0xA0.
    let mut cursor = tbs_content;
    if !cursor.is_empty() && cursor[0] == 0xA0 {
        let (_, rest) = parse_der_tlv(cursor)?;
        cursor = rest;
    }
    // Next TLV is the serial INTEGER (tag 0x02).
    if cursor.is_empty() || cursor[0] != 0x02 {
        return None;
    }
    let (serial_content, _) = parse_der_tlv(cursor)?;
    Some(serial_content.to_vec())
}

fn parse_der_sequence(data: &[u8]) -> Option<(&[u8], &[u8])> {
    if data.is_empty() || data[0] != 0x30 {
        return None;
    }
    parse_der_tlv(data)
}

fn parse_der_tlv(data: &[u8]) -> Option<(&[u8], &[u8])> {
    if data.len() < 2 {
        return None;
    }
    let first_len_byte = data[1];
    let (len, hdr_end) = if first_len_byte & 0x80 == 0 {
        (first_len_byte as usize, 2usize)
    } else {
        let n = (first_len_byte & 0x7F) as usize;
        if n == 0 || n > 4 || data.len() < 2 + n {
            return None;
        }
        let mut len = 0usize;
        for i in 0..n {
            len = (len << 8) | data[2 + i] as usize;
        }
        (len, 2 + n)
    };
    let end = hdr_end.checked_add(len)?;
    if data.len() < end {
        return None;
    }
    Some((&data[hdr_end..end], &data[end..]))
}

fn decode_hex(hex: &str) -> core::result::Result<Vec<u8>, String> {
    let trimmed: String = hex
        .chars()
        .filter(|c| !c.is_whitespace() && *c != ':')
        .collect();
    if trimmed.len() % 2 != 0 {
        return Err("odd hex length".to_string());
    }
    let mut out = Vec::with_capacity(trimmed.len() / 2);
    for chunk in trimmed.as_bytes().chunks(2) {
        let s = std::str::from_utf8(chunk).map_err(|e| e.to_string())?;
        out.push(u8::from_str_radix(s, 16).map_err(|e| e.to_string())?);
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    #[test]
    fn fingerprint_and_pin_roundtrip() {
        let fp = fingerprint_of(b"hello");
        let pin = format_pin(&fp);
        let parsed = parse_pin(&pin).expect("parse");
        assert_eq!(parsed, fp);
    }

    #[test]
    fn pin_rejects_bad_prefix() {
        assert!(parse_pin("md5:deadbeef").is_err());
    }

    #[test]
    fn pin_rejects_wrong_length() {
        assert!(parse_pin("sha256:abcd").is_err());
    }

    #[test]
    fn pin_verify_match() {
        let store = CertFingerprintStore::new();
        let cert = b"cert-der-bytes";
        let fp = fingerprint_of(cert);
        store.pin(42, fp);
        store.verify(42, cert).expect("match");
    }

    #[test]
    fn pin_verify_mismatch() {
        let store = CertFingerprintStore::new();
        store.pin(42, fingerprint_of(b"cert-a"));
        assert!(store.verify(42, b"cert-b").is_err());
    }

    #[test]
    fn pin_verify_unknown_subject() {
        let store = CertFingerprintStore::new();
        assert!(store.verify(99, b"anything").is_err());
    }

    #[test]
    fn pin_unpin() {
        let store = CertFingerprintStore::new();
        let fp = fingerprint_of(b"x");
        store.pin(1, fp);
        assert!(store.unpin(1, fp));
        assert!(store.pins_for(1).is_empty());
    }

    #[test]
    fn pin_multiple_fingerprints_per_subject() {
        let store = CertFingerprintStore::new();
        let fp1 = fingerprint_of(b"cert-1");
        let fp2 = fingerprint_of(b"cert-2");
        store.pin(7, fp1);
        store.pin(7, fp2);
        store.verify(7, b"cert-1").expect("match 1");
        store.verify(7, b"cert-2").expect("match 2");
    }

    #[tokio::test]
    async fn crl_loads_and_matches_serial() {
        let dir = tempfile::tempdir().expect("tmp");
        let path = dir.path().join("crl.txt");
        let mut f = std::fs::File::create(&path).expect("create");
        writeln!(f, "# crl").expect("w");
        writeln!(f, "deadbeef").expect("w");
        writeln!(f, "0x01020304").expect("w");
        drop(f);

        let ck = CrlOcspChecker::new(None).expect("ck");
        let n = ck.load_crl_file("default", &path).expect("load");
        assert_eq!(n, 2);

        // Build a minimal synthetic DER cert with serial 0xDEADBEEF.
        let der = synth_cert_with_serial(&[0xDE, 0xAD, 0xBE, 0xEF]);
        match ck.check_revocation(&der, &[]).await.expect("check") {
            RevocationStatus::Revoked { .. } => {}
            other => panic!("expected revoked, got {:?}", other),
        }
    }

    #[tokio::test]
    async fn crl_good_when_not_listed() {
        let dir = tempfile::tempdir().expect("tmp");
        let path = dir.path().join("crl.txt");
        std::fs::write(&path, "deadbeef\n").expect("w");
        let ck = CrlOcspChecker::new(None).expect("ck");
        ck.load_crl_file("default", &path).expect("load");
        let der = synth_cert_with_serial(&[0x01, 0x02, 0x03]);
        assert_eq!(
            ck.check_revocation(&der, &[]).await.expect("check"),
            RevocationStatus::Good
        );
    }

    /// Builds a synthetic DER cert whose `tbsCertificate` contains only the
    /// serial INTEGER. Enough to exercise the serial extractor.
    fn synth_cert_with_serial(serial: &[u8]) -> Vec<u8> {
        // INTEGER TLV for the serial.
        let mut ser_tlv = Vec::new();
        ser_tlv.push(0x02);
        ser_tlv.push(serial.len() as u8);
        ser_tlv.extend_from_slice(serial);

        // tbsCertificate SEQUENCE containing only the serial.
        let mut tbs = Vec::new();
        tbs.push(0x30);
        tbs.push(ser_tlv.len() as u8);
        tbs.extend_from_slice(&ser_tlv);

        // Outer SEQUENCE with tbs as its single element.
        let mut outer = Vec::new();
        outer.push(0x30);
        outer.push(tbs.len() as u8);
        outer.extend_from_slice(&tbs);
        outer
    }
}
