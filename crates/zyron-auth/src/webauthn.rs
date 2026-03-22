//! FIDO2/WebAuthn credential types, COSE key parsing, and signature verification.
//!
//! Provides the core types for hardware security key authentication (YubiKey,
//! Google Titan). Handles COSE public key parsing from CBOR, P-256 ECDSA and
//! Ed25519 signature verification, authenticator data parsing, and client data
//! JSON validation.

use crate::cbor;
use crate::role::UserId;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use zyron_common::{Result, ZyronError};

// ---------------------------------------------------------------------------
// COSE algorithm and key types
// ---------------------------------------------------------------------------

/// COSE algorithm identifiers used by WebAuthn authenticators.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[repr(i32)]
pub enum CoseAlgorithm {
    /// ECDSA with SHA-256 over P-256 curve.
    Es256 = -7,
    /// EdDSA (Ed25519).
    EdDsa = -8,
}

impl CoseAlgorithm {
    fn from_i64(v: i64) -> Result<Self> {
        match v {
            -7 => Ok(CoseAlgorithm::Es256),
            -8 => Ok(CoseAlgorithm::EdDsa),
            _ => Err(ZyronError::InvalidCredential(format!(
                "Unsupported COSE algorithm: {}",
                v
            ))),
        }
    }
}

/// Transport methods the authenticator supports.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[repr(u8)]
pub enum CredentialTransport {
    Usb = 0,
    Nfc = 1,
    Ble = 2,
    Internal = 3,
}

impl CredentialTransport {
    fn from_u8(v: u8) -> Result<Self> {
        match v {
            0 => Ok(CredentialTransport::Usb),
            1 => Ok(CredentialTransport::Nfc),
            2 => Ok(CredentialTransport::Ble),
            3 => Ok(CredentialTransport::Internal),
            _ => Err(ZyronError::DecodingFailed(format!(
                "Unknown CredentialTransport: {}",
                v
            ))),
        }
    }
}

/// A parsed COSE public key. Supports P-256 (ES256) and Ed25519 (EdDSA).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CosePublicKey {
    P256 { x: [u8; 32], y: [u8; 32] },
    Ed25519 { public_key: [u8; 32] },
}

// ---------------------------------------------------------------------------
// WebAuthn credential
// ---------------------------------------------------------------------------

/// A stored WebAuthn credential for a user.
#[derive(Debug, Clone)]
pub struct WebAuthnCredential {
    pub credential_id: Vec<u8>,
    pub user_id: UserId,
    pub public_key: CosePublicKey,
    pub sign_count: u32,
    pub transports: Vec<CredentialTransport>,
    pub created_at: u64,
    pub last_used_at: u64,
    pub friendly_name: String,
}

impl WebAuthnCredential {
    /// Serializes to bytes for heap storage.
    /// Layout: cred_id_len(4) + cred_id(N) + user_id(4) + key_type(1) + key_data(64 or 32)
    ///       + sign_count(4) + transport_count(4) + transports(N)
    ///       + created_at(8) + last_used_at(8) + name_len(4) + name(N)
    pub fn to_bytes(&self) -> Vec<u8> {
        let name_bytes = self.friendly_name.as_bytes();
        let mut buf = Vec::with_capacity(128 + self.credential_id.len() + name_bytes.len());

        // credential_id
        buf.extend_from_slice(&(self.credential_id.len() as u32).to_le_bytes());
        buf.extend_from_slice(&self.credential_id);

        // user_id
        buf.extend_from_slice(&self.user_id.0.to_le_bytes());

        // public_key
        match &self.public_key {
            CosePublicKey::P256 { x, y } => {
                buf.push(0); // P256 tag
                buf.extend_from_slice(x);
                buf.extend_from_slice(y);
            }
            CosePublicKey::Ed25519 { public_key } => {
                buf.push(1); // Ed25519 tag
                buf.extend_from_slice(public_key);
            }
        }

        // sign_count
        buf.extend_from_slice(&self.sign_count.to_le_bytes());

        // transports
        buf.extend_from_slice(&(self.transports.len() as u32).to_le_bytes());
        for t in &self.transports {
            buf.push(*t as u8);
        }

        // timestamps
        buf.extend_from_slice(&self.created_at.to_le_bytes());
        buf.extend_from_slice(&self.last_used_at.to_le_bytes());

        // friendly_name
        buf.extend_from_slice(&(name_bytes.len() as u32).to_le_bytes());
        buf.extend_from_slice(name_bytes);

        buf
    }

    /// Deserializes from bytes.
    pub fn from_bytes(data: &[u8]) -> Result<Self> {
        if data.len() < 13 {
            return Err(ZyronError::DecodingFailed(
                "WebAuthnCredential data too short".to_string(),
            ));
        }
        let mut pos = 0;

        // credential_id
        let cred_id_len =
            u32::from_le_bytes([data[pos], data[pos + 1], data[pos + 2], data[pos + 3]]) as usize;
        pos += 4;
        if data.len() < pos + cred_id_len {
            return Err(ZyronError::DecodingFailed(
                "WebAuthnCredential cred_id truncated".to_string(),
            ));
        }
        let credential_id = data[pos..pos + cred_id_len].to_vec();
        pos += cred_id_len;

        // user_id
        if data.len() < pos + 4 {
            return Err(ZyronError::DecodingFailed(
                "WebAuthnCredential user_id truncated".to_string(),
            ));
        }
        let user_id = UserId(u32::from_le_bytes([
            data[pos],
            data[pos + 1],
            data[pos + 2],
            data[pos + 3],
        ]));
        pos += 4;

        // public_key
        if data.len() < pos + 1 {
            return Err(ZyronError::DecodingFailed(
                "WebAuthnCredential key type missing".to_string(),
            ));
        }
        let key_tag = data[pos];
        pos += 1;
        let public_key = match key_tag {
            0 => {
                // P256
                if data.len() < pos + 64 {
                    return Err(ZyronError::DecodingFailed(
                        "WebAuthnCredential P256 key truncated".to_string(),
                    ));
                }
                let mut x = [0u8; 32];
                let mut y = [0u8; 32];
                x.copy_from_slice(&data[pos..pos + 32]);
                y.copy_from_slice(&data[pos + 32..pos + 64]);
                pos += 64;
                CosePublicKey::P256 { x, y }
            }
            1 => {
                // Ed25519
                if data.len() < pos + 32 {
                    return Err(ZyronError::DecodingFailed(
                        "WebAuthnCredential Ed25519 key truncated".to_string(),
                    ));
                }
                let mut public_key = [0u8; 32];
                public_key.copy_from_slice(&data[pos..pos + 32]);
                pos += 32;
                CosePublicKey::Ed25519 { public_key }
            }
            _ => {
                return Err(ZyronError::DecodingFailed(format!(
                    "Unknown key type tag: {}",
                    key_tag
                )));
            }
        };

        // sign_count
        if data.len() < pos + 4 {
            return Err(ZyronError::DecodingFailed(
                "WebAuthnCredential sign_count truncated".to_string(),
            ));
        }
        let sign_count =
            u32::from_le_bytes([data[pos], data[pos + 1], data[pos + 2], data[pos + 3]]);
        pos += 4;

        // transports
        if data.len() < pos + 4 {
            return Err(ZyronError::DecodingFailed(
                "WebAuthnCredential transports count truncated".to_string(),
            ));
        }
        let transport_count =
            u32::from_le_bytes([data[pos], data[pos + 1], data[pos + 2], data[pos + 3]]) as usize;
        pos += 4;
        let mut transports = Vec::with_capacity(transport_count);
        for _ in 0..transport_count {
            if data.len() < pos + 1 {
                return Err(ZyronError::DecodingFailed(
                    "WebAuthnCredential transport truncated".to_string(),
                ));
            }
            transports.push(CredentialTransport::from_u8(data[pos])?);
            pos += 1;
        }

        // timestamps
        if data.len() < pos + 16 {
            return Err(ZyronError::DecodingFailed(
                "WebAuthnCredential timestamps truncated".to_string(),
            ));
        }
        let created_at = u64::from_le_bytes([
            data[pos],
            data[pos + 1],
            data[pos + 2],
            data[pos + 3],
            data[pos + 4],
            data[pos + 5],
            data[pos + 6],
            data[pos + 7],
        ]);
        pos += 8;
        let last_used_at = u64::from_le_bytes([
            data[pos],
            data[pos + 1],
            data[pos + 2],
            data[pos + 3],
            data[pos + 4],
            data[pos + 5],
            data[pos + 6],
            data[pos + 7],
        ]);
        pos += 8;

        // friendly_name
        if data.len() < pos + 4 {
            return Err(ZyronError::DecodingFailed(
                "WebAuthnCredential name length truncated".to_string(),
            ));
        }
        let name_len =
            u32::from_le_bytes([data[pos], data[pos + 1], data[pos + 2], data[pos + 3]]) as usize;
        pos += 4;
        if data.len() < pos + name_len {
            return Err(ZyronError::DecodingFailed(
                "WebAuthnCredential name truncated".to_string(),
            ));
        }
        let friendly_name = std::str::from_utf8(&data[pos..pos + name_len])
            .map_err(|_| {
                ZyronError::DecodingFailed("WebAuthnCredential name invalid UTF-8".to_string())
            })?
            .to_string();

        Ok(Self {
            credential_id,
            user_id,
            public_key,
            sign_count,
            transports,
            created_at,
            last_used_at,
            friendly_name,
        })
    }
}

// ---------------------------------------------------------------------------
// Relying party configuration
// ---------------------------------------------------------------------------

/// Configuration for the WebAuthn relying party (the database server).
#[derive(Debug, Clone)]
pub struct RelyingPartyConfig {
    pub rp_id: String,
    pub rp_name: String,
    pub origin: String,
    pub challenge_timeout_secs: u64,
}

impl Default for RelyingPartyConfig {
    fn default() -> Self {
        Self {
            rp_id: "localhost".to_string(),
            rp_name: "ZyronDB".to_string(),
            origin: "https://localhost".to_string(),
            challenge_timeout_secs: 60,
        }
    }
}

// ---------------------------------------------------------------------------
// COSE key parsing
// ---------------------------------------------------------------------------

/// Parses a COSE public key from CBOR bytes.
/// COSE EC2 key map: {1: 2 (kty=EC2), 3: -7 (alg=ES256), -1: 1 (crv=P-256), -2: x, -3: y}
/// COSE OKP key map: {1: 1 (kty=OKP), 3: -8 (alg=EdDSA), -1: 6 (crv=Ed25519), -2: x}
pub fn parse_cose_public_key(cbor_bytes: &[u8]) -> Result<CosePublicKey> {
    let (value, _) = cbor::decode(cbor_bytes)?;

    let kty = value
        .map_get_int(1)
        .and_then(|v| v.as_unsigned())
        .ok_or_else(|| ZyronError::InvalidCredential("COSE key missing kty (1)".to_string()))?;

    let alg = value
        .map_get_int(3)
        .and_then(|v| v.as_int())
        .ok_or_else(|| ZyronError::InvalidCredential("COSE key missing alg (3)".to_string()))?;

    let algorithm = CoseAlgorithm::from_i64(alg)?;

    match (kty, algorithm) {
        (2, CoseAlgorithm::Es256) => {
            // EC2 P-256: verify curve parameter (-1) is 1 (P-256)
            let crv = value
                .map_get_int(-1)
                .and_then(|v| v.as_unsigned())
                .unwrap_or(0);
            if crv != 1 {
                return Err(ZyronError::InvalidCredential(format!(
                    "COSE EC2 key curve must be 1 (P-256), got {}",
                    crv
                )));
            }
            let x_bytes = value
                .map_get_int(-2)
                .and_then(|v| v.as_bytes())
                .ok_or_else(|| {
                    ZyronError::InvalidCredential("COSE EC2 key missing x (-2)".to_string())
                })?;
            let y_bytes = value
                .map_get_int(-3)
                .and_then(|v| v.as_bytes())
                .ok_or_else(|| {
                    ZyronError::InvalidCredential("COSE EC2 key missing y (-3)".to_string())
                })?;

            if x_bytes.len() != 32 || y_bytes.len() != 32 {
                return Err(ZyronError::InvalidCredential(format!(
                    "COSE EC2 key coordinates must be 32 bytes, got x={} y={}",
                    x_bytes.len(),
                    y_bytes.len()
                )));
            }

            let mut x = [0u8; 32];
            let mut y = [0u8; 32];
            x.copy_from_slice(x_bytes);
            y.copy_from_slice(y_bytes);

            Ok(CosePublicKey::P256 { x, y })
        }
        (1, CoseAlgorithm::EdDsa) => {
            // OKP Ed25519: verify curve parameter (-1) is 6 (Ed25519)
            let crv = value
                .map_get_int(-1)
                .and_then(|v| v.as_unsigned())
                .unwrap_or(0);
            if crv != 6 {
                return Err(ZyronError::InvalidCredential(format!(
                    "COSE OKP key curve must be 6 (Ed25519), got {}",
                    crv
                )));
            }
            let pk_bytes = value
                .map_get_int(-2)
                .and_then(|v| v.as_bytes())
                .ok_or_else(|| {
                    ZyronError::InvalidCredential("COSE OKP key missing x (-2)".to_string())
                })?;

            if pk_bytes.len() != 32 {
                return Err(ZyronError::InvalidCredential(format!(
                    "COSE Ed25519 key must be 32 bytes, got {}",
                    pk_bytes.len()
                )));
            }

            let mut public_key = [0u8; 32];
            public_key.copy_from_slice(pk_bytes);

            Ok(CosePublicKey::Ed25519 { public_key })
        }
        _ => Err(ZyronError::InvalidCredential(format!(
            "Unsupported COSE key type/algorithm combination: kty={}, alg={}",
            kty, alg
        ))),
    }
}

// ---------------------------------------------------------------------------
// Signature verification
// ---------------------------------------------------------------------------

/// Verifies a signature against a COSE public key.
/// For P-256: ECDSA signature in DER or raw r||s format (64 bytes).
/// For Ed25519: 64-byte Ed25519 signature.
pub fn verify_signature(
    public_key: &CosePublicKey,
    message: &[u8],
    signature: &[u8],
) -> Result<bool> {
    match public_key {
        CosePublicKey::P256 { x, y } => verify_p256(x, y, message, signature),
        CosePublicKey::Ed25519 { public_key } => verify_ed25519(public_key, message, signature),
    }
}

/// Verifies a P-256 ECDSA signature.
fn verify_p256(x: &[u8; 32], y: &[u8; 32], message: &[u8], signature: &[u8]) -> Result<bool> {
    use p256::EncodedPoint;
    use p256::ecdsa::signature::Verifier;
    use p256::ecdsa::{Signature, VerifyingKey};

    // Construct the uncompressed public key point (0x04 || x || y)
    let point = EncodedPoint::from_affine_coordinates(x.into(), y.into(), false);
    let verifying_key = VerifyingKey::from_encoded_point(&point)
        .map_err(|e| ZyronError::InvalidCredential(format!("Invalid P-256 public key: {}", e)))?;

    // WebAuthn signatures may be DER-encoded or raw r||s (64 bytes)
    let sig = if signature.len() == 64 {
        Signature::from_slice(signature)
    } else {
        Signature::from_der(signature)
    }
    .map_err(|e| ZyronError::InvalidCredential(format!("Invalid ECDSA signature format: {}", e)))?;

    match verifying_key.verify(message, &sig) {
        Ok(()) => Ok(true),
        Err(_) => Ok(false),
    }
}

/// Verifies an Ed25519 signature.
fn verify_ed25519(public_key_bytes: &[u8; 32], message: &[u8], signature: &[u8]) -> Result<bool> {
    use ed25519_dalek::{Signature, Verifier, VerifyingKey};

    let verifying_key = VerifyingKey::from_bytes(public_key_bytes)
        .map_err(|e| ZyronError::InvalidCredential(format!("Invalid Ed25519 public key: {}", e)))?;

    if signature.len() != 64 {
        return Err(ZyronError::InvalidCredential(format!(
            "Ed25519 signature must be 64 bytes, got {}",
            signature.len()
        )));
    }

    let mut sig_bytes = [0u8; 64];
    sig_bytes.copy_from_slice(signature);
    let sig = Signature::from_bytes(&sig_bytes);

    match verifying_key.verify(message, &sig) {
        Ok(()) => Ok(true),
        Err(_) => Ok(false),
    }
}

// ---------------------------------------------------------------------------
// Authenticator data parsing
// ---------------------------------------------------------------------------

/// Parsed authenticator data from a WebAuthn assertion or attestation.
#[derive(Debug)]
pub struct AuthenticatorData {
    /// SHA-256 hash of the relying party ID.
    pub rp_id_hash: [u8; 32],
    /// Flags byte: bit 0 = user present, bit 2 = user verified, bit 6 = attested credential data.
    pub flags: u8,
    /// Signature counter for clone detection.
    pub sign_count: u32,
    /// Present only during registration (when flags bit 6 is set).
    pub attested_credential: Option<AttestedCredentialData>,
}

impl AuthenticatorData {
    /// Returns true if the user presence flag is set.
    pub fn user_present(&self) -> bool {
        self.flags & 0x01 != 0
    }

    /// Returns true if the user verification flag is set.
    pub fn user_verified(&self) -> bool {
        self.flags & 0x04 != 0
    }

    /// Returns true if attested credential data is present.
    pub fn has_attested_credential(&self) -> bool {
        self.flags & 0x40 != 0
    }
}

/// Attested credential data included in authenticator data during registration.
#[derive(Debug)]
pub struct AttestedCredentialData {
    pub aaguid: [u8; 16],
    pub credential_id: Vec<u8>,
    pub public_key_cbor: Vec<u8>,
}

/// Parses authenticator data bytes.
/// Format: rpIdHash(32) || flags(1) || signCount(4) || [attestedCredentialData] || [extensions]
pub fn parse_authenticator_data(data: &[u8]) -> Result<AuthenticatorData> {
    if data.len() < 37 {
        return Err(ZyronError::InvalidCredential(
            "Authenticator data too short (minimum 37 bytes)".to_string(),
        ));
    }

    let mut rp_id_hash = [0u8; 32];
    rp_id_hash.copy_from_slice(&data[0..32]);

    let flags = data[32];
    let sign_count = u32::from_be_bytes([data[33], data[34], data[35], data[36]]);

    let attested_credential = if flags & 0x40 != 0 {
        // Attested credential data present
        if data.len() < 55 {
            return Err(ZyronError::InvalidCredential(
                "Authenticator data too short for attested credential".to_string(),
            ));
        }

        let mut aaguid = [0u8; 16];
        aaguid.copy_from_slice(&data[37..53]);

        let cred_id_len = u16::from_be_bytes([data[53], data[54]]) as usize;
        if data.len() < 55 + cred_id_len {
            return Err(ZyronError::InvalidCredential(
                "Authenticator data credential ID truncated".to_string(),
            ));
        }
        let credential_id = data[55..55 + cred_id_len].to_vec();

        // The remaining bytes after credential ID are the CBOR-encoded public key
        let pk_start = 55 + cred_id_len;
        let public_key_cbor = data[pk_start..].to_vec();

        Some(AttestedCredentialData {
            aaguid,
            credential_id,
            public_key_cbor,
        })
    } else {
        None
    };

    Ok(AuthenticatorData {
        rp_id_hash,
        flags,
        sign_count,
        attested_credential,
    })
}

// ---------------------------------------------------------------------------
// Client data JSON parsing
// ---------------------------------------------------------------------------

/// Parsed client data from a WebAuthn assertion or attestation.
#[derive(Debug)]
pub struct ClientData {
    /// Base64url-encoded challenge.
    pub challenge: String,
    /// Expected origin (e.g., "https://db.example.com").
    pub origin: String,
    /// "webauthn.get" for authentication, "webauthn.create" for registration.
    pub data_type: String,
}

/// Parses the clientDataJSON bytes into a ClientData struct.
/// Uses minimal hand-written JSON extraction (no serde_json dependency).
pub fn parse_client_data(json: &[u8]) -> Result<ClientData> {
    let json_str = std::str::from_utf8(json).map_err(|_| {
        ZyronError::InvalidCredential("clientDataJSON is not valid UTF-8".to_string())
    })?;

    let challenge = extract_json_string(json_str, "challenge").ok_or_else(|| {
        ZyronError::InvalidCredential("clientDataJSON missing 'challenge' field".to_string())
    })?;

    let origin = extract_json_string(json_str, "origin").ok_or_else(|| {
        ZyronError::InvalidCredential("clientDataJSON missing 'origin' field".to_string())
    })?;

    let data_type = extract_json_string(json_str, "type").ok_or_else(|| {
        ZyronError::InvalidCredential("clientDataJSON missing 'type' field".to_string())
    })?;

    Ok(ClientData {
        challenge,
        origin,
        data_type,
    })
}

/// Extracts a string value from a JSON object by key name.
/// Handles escaped quotes within values. Returns None if not found.
fn extract_json_string(json: &str, key: &str) -> Option<String> {
    let search = format!("\"{}\"", key);
    // Find the key as a top-level JSON key (preceded by '{' or ',')
    // to avoid matching key names that appear inside string values.
    let mut search_from = 0;
    let key_pos = loop {
        let pos = json[search_from..].find(&search)?;
        let abs_pos = search_from + pos;
        let before = json[..abs_pos].trim_end();
        if before.ends_with('{') || before.ends_with(',') {
            break abs_pos;
        }
        search_from = abs_pos + 1;
        if search_from >= json.len() {
            return None;
        }
    };
    let after_key = &json[key_pos + search.len()..];

    // Skip whitespace and colon
    let after_colon = after_key.trim_start();
    let after_colon = after_colon.strip_prefix(':')?;
    let after_colon = after_colon.trim_start();

    // Expect opening quote
    let after_colon = after_colon.strip_prefix('"')?;

    // Find closing quote. A quote is only a real terminator if preceded
    // by an even number of backslashes (0, 2, 4, ...).
    let mut end = 0;
    let bytes = after_colon.as_bytes();
    while end < bytes.len() {
        if bytes[end] == b'"' {
            let mut backslash_count = 0;
            let mut pos = end;
            while pos > 0 && bytes[pos - 1] == b'\\' {
                backslash_count += 1;
                pos -= 1;
            }
            if backslash_count % 2 == 0 {
                break;
            }
        }
        end += 1;
    }

    Some(after_colon[..end].to_string())
}

// ---------------------------------------------------------------------------
// Attestation parsing (registration)
// ---------------------------------------------------------------------------

/// Result of parsing an attestation object during registration.
#[derive(Debug)]
pub struct AttestationResult {
    pub credential_id: Vec<u8>,
    pub public_key: CosePublicKey,
    pub sign_count: u32,
}

/// Parses a CBOR-encoded attestation object from a WebAuthn registration response.
/// Extracts the credential ID and public key from the authData field.
pub fn parse_attestation_object(data: &[u8]) -> Result<AttestationResult> {
    let (value, _) = cbor::decode(data)?;

    let auth_data_bytes = value
        .map_get_int(2) // "authData" is sometimes keyed as text, sometimes int. Try text key.
        .or_else(|| {
            // Try text key "authData"
            if let Some(entries) = value.as_map() {
                for (k, v) in entries {
                    if k.as_text() == Some("authData") {
                        return Some(v);
                    }
                }
            }
            None
        })
        .and_then(|v| v.as_bytes())
        .ok_or_else(|| {
            ZyronError::InvalidCredential("Attestation object missing authData".to_string())
        })?;

    let auth_data = parse_authenticator_data(auth_data_bytes)?;

    let attested = auth_data.attested_credential.ok_or_else(|| {
        ZyronError::InvalidCredential(
            "Attestation authData does not contain attested credential data".to_string(),
        )
    })?;

    let public_key = parse_cose_public_key(&attested.public_key_cbor)?;

    Ok(AttestationResult {
        credential_id: attested.credential_id,
        public_key,
        sign_count: auth_data.sign_count,
    })
}

// ---------------------------------------------------------------------------
// Assertion verification (authentication)
// ---------------------------------------------------------------------------

/// Verifies a WebAuthn assertion (authentication response).
/// Returns the new sign count on success.
pub fn verify_assertion(
    credential: &WebAuthnCredential,
    authenticator_data: &[u8],
    client_data_json: &[u8],
    signature: &[u8],
    rp_config: &RelyingPartyConfig,
    challenge: &[u8; 32],
) -> Result<u32> {
    // 1. Parse authenticator data
    let auth_data = parse_authenticator_data(authenticator_data)?;

    // 2. Verify rpIdHash
    let expected_rp_hash = Sha256::digest(rp_config.rp_id.as_bytes());
    if auth_data.rp_id_hash != expected_rp_hash.as_slice() {
        return Err(ZyronError::AuthenticationFailed(
            "RP ID hash mismatch in authenticator data".to_string(),
        ));
    }

    // 3. Verify user presence
    if !auth_data.user_present() {
        return Err(ZyronError::AuthenticationFailed(
            "User presence flag not set in authenticator data".to_string(),
        ));
    }

    // 4. Parse client data JSON
    let client_data = parse_client_data(client_data_json)?;

    // 5. Verify type
    if client_data.data_type != "webauthn.get" {
        return Err(ZyronError::AuthenticationFailed(format!(
            "Expected clientDataJSON type 'webauthn.get', got '{}'",
            client_data.data_type
        )));
    }

    // 6. Verify challenge
    let decoded_challenge = base64url_decode(&client_data.challenge).map_err(|e| {
        ZyronError::AuthenticationFailed(format!("Invalid base64url challenge: {}", e))
    })?;
    if decoded_challenge.as_slice() != challenge {
        return Err(ZyronError::AuthenticationFailed(
            "Challenge mismatch in clientDataJSON".to_string(),
        ));
    }

    // 7. Verify origin
    if client_data.origin != rp_config.origin {
        return Err(ZyronError::AuthenticationFailed(format!(
            "Origin mismatch: expected '{}', got '{}'",
            rp_config.origin, client_data.origin
        )));
    }

    // 8. Compute verification message: authenticatorData || SHA-256(clientDataJSON)
    let client_data_hash = Sha256::digest(client_data_json);
    let mut verification_message = Vec::with_capacity(authenticator_data.len() + 32);
    verification_message.extend_from_slice(authenticator_data);
    verification_message.extend_from_slice(&client_data_hash);

    // 9. Verify signature
    let valid = verify_signature(&credential.public_key, &verification_message, signature)?;
    if !valid {
        return Err(ZyronError::AuthenticationFailed(
            "WebAuthn signature verification failed".to_string(),
        ));
    }

    // 10. Check sign count (clone detection)
    if auth_data.sign_count != 0 && auth_data.sign_count <= credential.sign_count {
        return Err(ZyronError::AuthenticationFailed(
            "Sign count did not increase, possible credential clone detected".to_string(),
        ));
    }

    Ok(auth_data.sign_count)
}

// ---------------------------------------------------------------------------
// Base64url encoding/decoding
// ---------------------------------------------------------------------------

/// Decodes a base64url-encoded string (no padding).
pub fn base64url_decode(input: &str) -> std::result::Result<Vec<u8>, String> {
    // Convert base64url to standard base64
    let mut standard = input.replace('-', "+").replace('_', "/");
    // Add padding
    match standard.len() % 4 {
        2 => standard.push_str("=="),
        3 => standard.push('='),
        _ => {}
    }

    base64::Engine::decode(&base64::engine::general_purpose::STANDARD, &standard)
        .map_err(|e| format!("base64url decode error: {}", e))
}

/// Encodes bytes as a base64url string (no padding).
pub fn base64url_encode(input: &[u8]) -> String {
    use base64::Engine;
    let standard = base64::engine::general_purpose::STANDARD_NO_PAD.encode(input);
    standard.replace('+', "-").replace('/', "_")
}

// ---------------------------------------------------------------------------
// Challenge generation
// ---------------------------------------------------------------------------

/// Generates a cryptographically random 32-byte challenge.
pub fn generate_challenge() -> [u8; 32] {
    use rand::Rng;
    let mut challenge = [0u8; 32];
    rand::rng().fill_bytes(&mut challenge);
    challenge
}

#[cfg(test)]
mod tests {
    use super::*;

    // -- COSE key parsing --

    fn build_p256_cose_cbor(x: &[u8; 32], y: &[u8; 32]) -> Vec<u8> {
        let mut data = Vec::new();
        data.push(0xa5); // map of 5 items
        data.push(0x01);
        data.push(0x02); // 1: 2 (kty: EC2)
        data.push(0x03);
        data.push(0x26); // 3: -7 (alg: ES256)
        data.push(0x20);
        data.push(0x01); // -1: 1 (crv: P-256)
        data.push(0x21);
        data.push(0x58);
        data.push(32); // -2: bstr(32)
        data.extend_from_slice(x);
        data.push(0x22);
        data.push(0x58);
        data.push(32); // -3: bstr(32)
        data.extend_from_slice(y);
        data
    }

    fn build_ed25519_cose_cbor(pk: &[u8; 32]) -> Vec<u8> {
        let mut data = Vec::new();
        data.push(0xa4); // map of 4 items
        data.push(0x01);
        data.push(0x01); // 1: 1 (kty: OKP)
        data.push(0x03);
        data.push(0x27); // 3: -8 (alg: EdDSA)
        data.push(0x20);
        data.push(0x06); // -1: 6 (crv: Ed25519)
        data.push(0x21);
        data.push(0x58);
        data.push(32); // -2: bstr(32)
        data.extend_from_slice(pk);
        data
    }

    #[test]
    fn test_parse_cose_p256() {
        let x = [0xAA; 32];
        let y = [0xBB; 32];
        let cbor = build_p256_cose_cbor(&x, &y);
        let key = parse_cose_public_key(&cbor).expect("parse");
        match key {
            CosePublicKey::P256 { x: px, y: py } => {
                assert_eq!(px, x);
                assert_eq!(py, y);
            }
            _ => panic!("expected P256"),
        }
    }

    #[test]
    fn test_parse_cose_ed25519() {
        let pk = [0xCC; 32];
        let cbor = build_ed25519_cose_cbor(&pk);
        let key = parse_cose_public_key(&cbor).expect("parse");
        match key {
            CosePublicKey::Ed25519 { public_key } => {
                assert_eq!(public_key, pk);
            }
            _ => panic!("expected Ed25519"),
        }
    }

    #[test]
    fn test_parse_cose_missing_kty() {
        // Map without key 1
        let cbor = [0xa1, 0x03, 0x26]; // {3: -7}
        assert!(parse_cose_public_key(&cbor).is_err());
    }

    // -- P-256 signature verification --

    #[test]
    fn test_p256_sign_and_verify() {
        use p256::ecdsa::{SigningKey, signature::Signer};

        let signing_key = SigningKey::random(&mut p256::elliptic_curve::rand_core::OsRng);
        let verifying_key = signing_key.verifying_key();
        let point = verifying_key.to_encoded_point(false);

        let mut x = [0u8; 32];
        let mut y = [0u8; 32];
        x.copy_from_slice(point.x().unwrap());
        y.copy_from_slice(point.y().unwrap());

        let message = b"test message for P-256 ECDSA verification";
        let sig: p256::ecdsa::Signature = signing_key.sign(message);

        let result = verify_p256(&x, &y, message, &sig.to_der().to_bytes()).expect("verify");
        assert!(result);
    }

    #[test]
    fn test_p256_verify_wrong_message() {
        use p256::ecdsa::{SigningKey, signature::Signer};

        let signing_key = SigningKey::random(&mut p256::elliptic_curve::rand_core::OsRng);
        let verifying_key = signing_key.verifying_key();
        let point = verifying_key.to_encoded_point(false);

        let mut x = [0u8; 32];
        let mut y = [0u8; 32];
        x.copy_from_slice(point.x().unwrap());
        y.copy_from_slice(point.y().unwrap());

        let sig: p256::ecdsa::Signature = signing_key.sign(b"original message");
        let result =
            verify_p256(&x, &y, b"different message", &sig.to_der().to_bytes()).expect("verify");
        assert!(!result);
    }

    #[test]
    fn test_p256_raw_signature_format() {
        use p256::ecdsa::{SigningKey, signature::Signer};

        let signing_key = SigningKey::random(&mut p256::elliptic_curve::rand_core::OsRng);
        let verifying_key = signing_key.verifying_key();
        let point = verifying_key.to_encoded_point(false);

        let mut x = [0u8; 32];
        let mut y = [0u8; 32];
        x.copy_from_slice(point.x().unwrap());
        y.copy_from_slice(point.y().unwrap());

        let message = b"raw signature test";
        let sig: p256::ecdsa::Signature = signing_key.sign(message);

        // Use raw r||s format (64 bytes)
        let raw_sig = sig.to_bytes();
        assert_eq!(raw_sig.len(), 64);
        let result = verify_p256(&x, &y, message, &raw_sig).expect("verify");
        assert!(result);
    }

    // -- Ed25519 signature verification --

    #[test]
    fn test_ed25519_sign_and_verify() {
        use ed25519_dalek::{Signer, SigningKey};

        let signing_key = SigningKey::generate(&mut p256::elliptic_curve::rand_core::OsRng);
        let verifying_key = signing_key.verifying_key();

        let message = b"test message for Ed25519 verification";
        let sig = signing_key.sign(message);

        let result =
            verify_ed25519(&verifying_key.to_bytes(), message, &sig.to_bytes()).expect("verify");
        assert!(result);
    }

    #[test]
    fn test_ed25519_verify_wrong_message() {
        use ed25519_dalek::{Signer, SigningKey};

        let signing_key = SigningKey::generate(&mut p256::elliptic_curve::rand_core::OsRng);
        let verifying_key = signing_key.verifying_key();

        let sig = signing_key.sign(b"original");
        let result = verify_ed25519(&verifying_key.to_bytes(), b"different", &sig.to_bytes())
            .expect("verify");
        assert!(!result);
    }

    // -- Authenticator data parsing --

    #[test]
    fn test_parse_authenticator_data_minimal() {
        let mut data = vec![0u8; 37];
        data[0..32].copy_from_slice(&[0xAA; 32]); // rpIdHash
        data[32] = 0x01; // flags: user present
        data[33..37].copy_from_slice(&42u32.to_be_bytes()); // signCount

        let auth_data = parse_authenticator_data(&data).expect("parse");
        assert_eq!(auth_data.rp_id_hash, [0xAA; 32]);
        assert!(auth_data.user_present());
        assert!(!auth_data.user_verified());
        assert_eq!(auth_data.sign_count, 42);
        assert!(auth_data.attested_credential.is_none());
    }

    #[test]
    fn test_parse_authenticator_data_too_short() {
        assert!(parse_authenticator_data(&[0u8; 10]).is_err());
    }

    // -- Client data JSON parsing --

    #[test]
    fn test_parse_client_data() {
        let json = br#"{"type":"webauthn.get","challenge":"dGVzdA","origin":"https://example.com","crossOrigin":false}"#;
        let data = parse_client_data(json).expect("parse");
        assert_eq!(data.data_type, "webauthn.get");
        assert_eq!(data.challenge, "dGVzdA");
        assert_eq!(data.origin, "https://example.com");
    }

    #[test]
    fn test_parse_client_data_missing_challenge() {
        let json = br#"{"type":"webauthn.get","origin":"https://example.com"}"#;
        assert!(parse_client_data(json).is_err());
    }

    // -- WebAuthnCredential serialization --

    #[test]
    fn test_credential_roundtrip_p256() {
        let cred = WebAuthnCredential {
            credential_id: vec![1, 2, 3, 4, 5],
            user_id: UserId(42),
            public_key: CosePublicKey::P256 {
                x: [0xAA; 32],
                y: [0xBB; 32],
            },
            sign_count: 10,
            transports: vec![CredentialTransport::Usb, CredentialTransport::Nfc],
            created_at: 1700000000,
            last_used_at: 1700001000,
            friendly_name: "YubiKey 5".to_string(),
        };
        let bytes = cred.to_bytes();
        let restored = WebAuthnCredential::from_bytes(&bytes).expect("decode");
        assert_eq!(restored.credential_id, vec![1, 2, 3, 4, 5]);
        assert_eq!(restored.user_id, UserId(42));
        assert_eq!(
            restored.public_key,
            CosePublicKey::P256 {
                x: [0xAA; 32],
                y: [0xBB; 32]
            }
        );
        assert_eq!(restored.sign_count, 10);
        assert_eq!(restored.transports.len(), 2);
        assert_eq!(restored.friendly_name, "YubiKey 5");
    }

    #[test]
    fn test_credential_roundtrip_ed25519() {
        let cred = WebAuthnCredential {
            credential_id: vec![0xFF; 64],
            user_id: UserId(1),
            public_key: CosePublicKey::Ed25519 {
                public_key: [0xCC; 32],
            },
            sign_count: 0,
            transports: vec![CredentialTransport::Internal],
            created_at: 0,
            last_used_at: 0,
            friendly_name: "Titan".to_string(),
        };
        let bytes = cred.to_bytes();
        let restored = WebAuthnCredential::from_bytes(&bytes).expect("decode");
        assert_eq!(restored.credential_id.len(), 64);
        assert_eq!(
            restored.public_key,
            CosePublicKey::Ed25519 {
                public_key: [0xCC; 32]
            }
        );
        assert_eq!(restored.friendly_name, "Titan");
    }

    #[test]
    fn test_credential_from_bytes_too_short() {
        assert!(WebAuthnCredential::from_bytes(&[0u8; 5]).is_err());
    }

    // -- Base64url --

    #[test]
    fn test_base64url_roundtrip() {
        let data = [0x01, 0x02, 0x03, 0xFF, 0xFE];
        let encoded = base64url_encode(&data);
        assert!(!encoded.contains('+'));
        assert!(!encoded.contains('/'));
        assert!(!encoded.contains('='));
        let decoded = base64url_decode(&encoded).expect("decode");
        assert_eq!(decoded, data);
    }

    #[test]
    fn test_base64url_decode_invalid() {
        assert!(base64url_decode("!!!invalid!!!").is_err());
    }

    // -- Challenge generation --

    #[test]
    fn test_generate_challenge_unique() {
        let c1 = generate_challenge();
        let c2 = generate_challenge();
        assert_ne!(c1, c2);
        assert_eq!(c1.len(), 32);
    }

    // -- verify_assertion integration test --

    #[test]
    fn test_verify_assertion_rp_id_mismatch() {
        use sha2::Digest;

        let rp_config = RelyingPartyConfig {
            rp_id: "example.com".to_string(),
            rp_name: "Test".to_string(),
            origin: "https://example.com".to_string(),
            challenge_timeout_secs: 60,
        };

        // Build authenticator data with wrong rpIdHash
        let wrong_hash = Sha256::digest(b"wrong.com");
        let mut auth_data = vec![0u8; 37];
        auth_data[0..32].copy_from_slice(&wrong_hash);
        auth_data[32] = 0x01; // user present
        auth_data[33..37].copy_from_slice(&1u32.to_be_bytes());

        let cred = WebAuthnCredential {
            credential_id: vec![1],
            user_id: UserId(1),
            public_key: CosePublicKey::P256 {
                x: [0; 32],
                y: [0; 32],
            },
            sign_count: 0,
            transports: vec![],
            created_at: 0,
            last_used_at: 0,
            friendly_name: "test".to_string(),
        };

        let challenge = [0u8; 32];
        let client_json = br#"{"type":"webauthn.get","challenge":"AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA","origin":"https://example.com"}"#;

        let result = verify_assertion(
            &cred,
            &auth_data,
            client_json,
            &[0; 64],
            &rp_config,
            &challenge,
        );
        assert!(result.is_err());
        assert!(
            result
                .unwrap_err()
                .to_string()
                .contains("RP ID hash mismatch")
        );
    }

    // -- extract_json_string --

    #[test]
    fn test_extract_json_string_basic() {
        let json = r#"{"key": "value", "other": "data"}"#;
        assert_eq!(extract_json_string(json, "key"), Some("value".to_string()));
        assert_eq!(extract_json_string(json, "other"), Some("data".to_string()));
        assert_eq!(extract_json_string(json, "missing"), None);
    }
}
