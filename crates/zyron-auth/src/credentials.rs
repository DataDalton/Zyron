//! Credential types for password hashing, API keys, JWTs, and TOTP.
//!
//! Each credential type implements its own verification logic. Password
//! credentials use Balloon Hashing (SHA-256), API keys use SHA-256 with
//! constant-time comparison, JWTs use HMAC-SHA256/384/512, and TOTP
//! uses HMAC-SHA1 per RFC 6238.

use hmac::{Hmac, Mac};
use sha2::{Digest, Sha256};
use zyron_common::{Result, ZyronError};

use crate::balloon::{self, BalloonParams};
use crate::hmac_sha2::{HmacSha256Key, HmacSha384Key, HmacSha512Key};

// ---------------------------------------------------------------------------
// PasswordCredential
// ---------------------------------------------------------------------------

/// Stores a Balloon-hashed password in PHC string format.
pub struct PasswordCredential {
    hash: String,
}

impl PasswordCredential {
    /// Hashes the plaintext password using Balloon Hashing with default parameters.
    pub fn from_plaintext(password: &str) -> Result<Self> {
        let hash = balloon::balloon_hash_encoded(password)?;
        Ok(Self { hash })
    }

    /// Hashes the plaintext password using Balloon Hashing with test parameters (fast).
    pub fn from_plaintext_with_params(password: &str, params: &BalloonParams) -> Result<Self> {
        let hash = balloon::balloon_hash_encoded_with_params(password, params)?;
        Ok(Self { hash })
    }

    /// Verifies a plaintext password against the stored hash.
    pub fn verify(&self, password: &str) -> Result<bool> {
        balloon::balloon_verify(password, &self.hash)
    }

    /// Returns the stored hash string for persistence.
    pub fn as_stored(&self) -> &str {
        &self.hash
    }

    /// Reconstructs from a previously stored hash string.
    pub fn from_stored(s: String) -> Self {
        Self { hash: s }
    }
}

// ---------------------------------------------------------------------------
// SCRAM-SHA-256 verifier derivation
// ---------------------------------------------------------------------------

/// Iteration count for the SCRAM-SHA-256 PBKDF2 derivation. Matches the
/// PostgreSQL default and the RFC 7677 minimum recommendation.
pub const SCRAM_SHA_256_ITERATIONS: u32 = 4096;

/// Derives a PostgreSQL-format SCRAM-SHA-256 secret from a plaintext password
/// using a fresh random 16-byte salt.
///
/// Format: SCRAM-SHA-256$<iterations>:<base64 salt>$<base64 StoredKey>:<base64 ServerKey>
pub fn scram_sha256_secret(password: &str) -> String {
    use rand::Rng;
    let mut salt = [0u8; 16];
    rand::rng().fill_bytes(&mut salt);
    scram_sha256_secret_with_salt(password, &salt, SCRAM_SHA_256_ITERATIONS)
}

/// Derives a SCRAM-SHA-256 secret with an explicit salt and iteration count.
pub fn scram_sha256_secret_with_salt(password: &str, salt: &[u8], iterations: u32) -> String {
    let salted_password = pbkdf2_sha256(password.as_bytes(), salt, iterations);
    let client_key = hmac_sha256(&salted_password, b"Client Key");
    let stored_key = sha256_hash(&client_key);
    let server_key = hmac_sha256(&salted_password, b"Server Key");

    use base64::Engine;
    let std_b64 = base64::engine::general_purpose::STANDARD;
    format!(
        "SCRAM-SHA-256${}:{}${}:{}",
        iterations,
        std_b64.encode(salt),
        std_b64.encode(stored_key),
        std_b64.encode(server_key),
    )
}

/// Parsed components of a PostgreSQL SCRAM-SHA-256 secret.
pub struct ScramSecret {
    pub iterations: u32,
    pub salt: Vec<u8>,
    pub stored_key: [u8; 32],
    pub server_key: [u8; 32],
}

/// Parses a stored SCRAM-SHA-256 secret string into its components.
pub fn parse_scram_secret(secret: &str) -> Result<ScramSecret> {
    let body = secret.strip_prefix("SCRAM-SHA-256$").ok_or_else(|| {
        ZyronError::InvalidCredential("SCRAM secret missing SCRAM-SHA-256 prefix".to_string())
    })?;
    let (iter_salt, keys) = body.split_once('$').ok_or_else(|| {
        ZyronError::InvalidCredential("SCRAM secret missing key section".to_string())
    })?;
    let (iter_str, salt_b64) = iter_salt
        .split_once(':')
        .ok_or_else(|| ZyronError::InvalidCredential("SCRAM secret missing salt".to_string()))?;
    let (stored_b64, server_b64) = keys.split_once(':').ok_or_else(|| {
        ZyronError::InvalidCredential("SCRAM secret missing server key".to_string())
    })?;

    let iterations = iter_str.parse::<u32>().map_err(|_| {
        ZyronError::InvalidCredential("SCRAM secret invalid iteration count".to_string())
    })?;

    use base64::Engine;
    let std_b64 = base64::engine::general_purpose::STANDARD;
    let salt = std_b64.decode(salt_b64).map_err(|_| {
        ZyronError::InvalidCredential("SCRAM secret invalid salt base64".to_string())
    })?;
    let stored_vec = std_b64.decode(stored_b64).map_err(|_| {
        ZyronError::InvalidCredential("SCRAM secret invalid stored key base64".to_string())
    })?;
    let server_vec = std_b64.decode(server_b64).map_err(|_| {
        ZyronError::InvalidCredential("SCRAM secret invalid server key base64".to_string())
    })?;
    if stored_vec.len() != 32 || server_vec.len() != 32 {
        return Err(ZyronError::InvalidCredential(
            "SCRAM secret keys must be 32 bytes".to_string(),
        ));
    }
    let mut stored_key = [0u8; 32];
    stored_key.copy_from_slice(&stored_vec);
    let mut server_key = [0u8; 32];
    server_key.copy_from_slice(&server_vec);

    Ok(ScramSecret {
        iterations,
        salt,
        stored_key,
        server_key,
    })
}

/// Derives the MD5 credential value md5(password + username) as 32 lowercase
/// hex chars. The wire MD5 flow validates md5(this + salt) against the client
/// response, so the stored value omits the per-connection salt.
pub fn md5_password_credential(user: &str, password: &str) -> String {
    use md5::{Digest, Md5};
    let mut hasher = Md5::new();
    hasher.update(password.as_bytes());
    hasher.update(user.as_bytes());
    format!("{:x}", hasher.finalize())
}

/// PBKDF2-HMAC-SHA-256, the SCRAM salted password derivation.
///
/// The one implementation in the build. A server verifying SCRAM and a client
/// answering it have to derive the identical key or every Zyron-to-Zyron
/// connection fails authentication, and three separate copies of a primitive
/// that must agree byte for byte is a drift waiting to happen.
pub fn pbkdf2_sha256(password: &[u8], salt: &[u8], iterations: u32) -> [u8; 32] {
    let mut result = [0u8; 32];
    pbkdf2::pbkdf2_hmac::<Sha256>(password, salt, iterations, &mut result);
    result
}

/// HMAC-SHA-256 returning a fixed 32-byte array.
fn hmac_sha256(key: &[u8], data: &[u8]) -> [u8; 32] {
    crate::hmac_sha2::hmac_sha256(key, data)
}

// ---------------------------------------------------------------------------
// ApiKeyCredential
// ---------------------------------------------------------------------------

/// Stores a hashed API key with a readable prefix for identification.
/// The full key is never stored, only its SHA-256 hash.
pub struct ApiKeyCredential {
    prefix: String,
    key_hash: [u8; 32],
}

impl ApiKeyCredential {
    /// Generates a new API key. Returns the credential and the full plaintext key.
    /// The key format is "zyron_" followed by 32 random base64url characters.
    pub fn generate() -> (Self, String) {
        use rand::Rng;
        let mut rng = rand::rng();

        // Generate 24 random bytes, which encode to 32 base64url chars.
        let mut raw = [0u8; 24];
        rng.fill_bytes(&mut raw);
        let suffix = base64url_encode(&raw);

        let full_key = format!("zyron_{}", suffix);
        let prefix = format!("zyron_{}", &suffix[..8]);

        let key_hash = sha256_hash(full_key.as_bytes());

        let cred = Self { prefix, key_hash };
        (cred, full_key)
    }

    /// Verifies a presented key against the stored hash using constant-time comparison.
    pub fn verify(&self, presented_key: &str) -> bool {
        let hash = sha256_hash(presented_key.as_bytes());
        balloon::constant_time_eq(&hash, &self.key_hash)
    }

    /// Reconstructs from stored prefix and hash.
    pub fn from_stored(prefix: String, key_hash: [u8; 32]) -> Self {
        Self { prefix, key_hash }
    }

    /// Returns the human-readable prefix for display.
    pub fn prefix(&self) -> &str {
        &self.prefix
    }

    /// Returns the stored SHA-256 hash of the full key.
    pub fn key_hash(&self) -> &[u8; 32] {
        &self.key_hash
    }
}

// ---------------------------------------------------------------------------
// JWT types
// ---------------------------------------------------------------------------

/// HMAC signing algorithm for JWT tokens.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum JwtAlgorithm {
    Hs256,
    Hs384,
    Hs512,
}

impl JwtAlgorithm {
    /// Returns the algorithm name as it appears in the JWT header.
    fn as_str(&self) -> &'static str {
        match self {
            JwtAlgorithm::Hs256 => "HS256",
            JwtAlgorithm::Hs384 => "HS384",
            JwtAlgorithm::Hs512 => "HS512",
        }
    }

    /// Parses from the JWT header algorithm string.
    #[allow(dead_code)]
    pub(crate) fn from_str(s: &str) -> Result<Self> {
        match s {
            "HS256" => Ok(JwtAlgorithm::Hs256),
            "HS384" => Ok(JwtAlgorithm::Hs384),
            "HS512" => Ok(JwtAlgorithm::Hs512),
            _ => Err(ZyronError::InvalidCredential(format!(
                "Unsupported JWT algorithm: {}",
                s
            ))),
        }
    }

    /// Minimum key length in bytes for each algorithm.
    fn min_key_len(&self) -> usize {
        match self {
            JwtAlgorithm::Hs256 => 32,
            JwtAlgorithm::Hs384 => 48,
            JwtAlgorithm::Hs512 => 64,
        }
    }

    /// HMAC output length in bytes for this algorithm
    fn signature_len(&self) -> usize {
        match self {
            JwtAlgorithm::Hs256 => 32,
            JwtAlgorithm::Hs384 => 48,
            JwtAlgorithm::Hs512 => 64,
        }
    }
}

/// JWT header (alg + typ fields).
#[derive(Debug, Clone)]
pub struct JwtHeader {
    pub alg: String,
    pub typ: String,
}

/// JWT claims payload with standard and custom fields.
#[derive(Debug, Clone)]
pub struct JwtClaims {
    pub sub: String,
    pub iss: Option<String>,
    pub exp: u64,
    pub iat: u64,
    pub roles: Vec<String>,
    pub custom: std::collections::HashMap<String, String>,
}

/// Signing key for one algorithm, holding the padded HMAC blocks so a
/// signature costs two hash passes and no key derivation
enum JwtMacKey {
    Hs256(HmacSha256Key),
    Hs384(HmacSha384Key),
    Hs512(HmacSha512Key),
}

/// JWT credential that can encode and decode tokens using HMAC signing
/// Pre-computes the HMAC key schedule on construction to avoid per-call overhead
pub struct JwtCredential {
    algorithm: JwtAlgorithm,
    issuer: Option<String>,
    max_age_secs: u64,
    /// Padded key blocks for the declared algorithm, so the variant present
    /// always matches `algorithm` and signing cannot fail on a missing key
    mac_key: JwtMacKey,
    /// Pre-computed base64url-encoded header for this algorithm, the header
    /// JSON is constant per algorithm so encoding it once at construction
    /// saves one format!, one base64 alloc, and one String alloc per encode
    header_b64: String,
}

/// Stack-allocated signature buffer sized for the largest HMAC output (HS512 = 64 bytes)
/// Avoids one heap allocation per sign() call on the encode/decode hot path
struct SignatureBuf {
    bytes: [u8; 64],
    len: usize,
}

impl SignatureBuf {
    fn as_slice(&self) -> &[u8] {
        &self.bytes[..self.len]
    }
}

impl JwtCredential {
    /// Creates a new JWT credential. The secret must be at least 32 bytes for HS256,
    /// 48 for HS384, or 64 for HS512.
    pub fn new(secret: Vec<u8>, algorithm: JwtAlgorithm) -> Result<Self> {
        let min = algorithm.min_key_len();
        if secret.len() < min {
            return Err(ZyronError::InvalidCredential(format!(
                "JWT secret too short for {}: need {} bytes, got {}",
                algorithm.as_str(),
                min,
                secret.len()
            )));
        }
        let mac_key = match algorithm {
            JwtAlgorithm::Hs256 => JwtMacKey::Hs256(HmacSha256Key::new(&secret)),
            JwtAlgorithm::Hs384 => JwtMacKey::Hs384(HmacSha384Key::new(&secret)),
            JwtAlgorithm::Hs512 => JwtMacKey::Hs512(HmacSha512Key::new(&secret)),
        };
        let header_json = format!("{{\"alg\":\"{}\",\"typ\":\"JWT\"}}", algorithm.as_str());
        let header_b64 = base64url_encode(header_json.as_bytes());

        Ok(Self {
            algorithm,
            issuer: None,
            max_age_secs: 3600,
            mac_key,
            header_b64,
        })
    }

    /// Sets the issuer claim for tokens created by this credential.
    pub fn with_issuer(mut self, issuer: String) -> Self {
        self.issuer = Some(issuer);
        self
    }

    /// Sets the maximum token age in seconds.
    pub fn with_max_age(mut self, secs: u64) -> Self {
        self.max_age_secs = secs;
        self
    }

    /// Encodes the claims into a signed JWT string (header.payload.signature)
    /// Builds the token in a single output buffer with capacity reservation,
    /// avoiding intermediate String/Vec allocations for header, payload b64,
    /// signing input, signature, and sig b64
    pub fn encode(&self, claims: &JwtClaims) -> Result<String> {
        use base64::Engine;
        let payload_json = claims_to_json(claims);
        let sig_b64_max = self.algorithm.signature_len() * 4 / 3 + 4;
        let payload_b64_len = (payload_json.len() * 4 + 2) / 3;
        let cap = self.header_b64.len() + 1 + payload_b64_len + 1 + sig_b64_max;

        let mut token = String::with_capacity(cap);
        token.push_str(&self.header_b64);
        token.push('.');
        base64::engine::general_purpose::URL_SAFE_NO_PAD
            .encode_string(payload_json.as_bytes(), &mut token);

        let signature = self.sign(token.as_bytes());
        token.push('.');
        base64::engine::general_purpose::URL_SAFE_NO_PAD
            .encode_string(signature.as_slice(), &mut token);

        Ok(token)
    }

    /// Decodes and verifies a JWT token. Checks signature, expiration, and issuer.
    pub fn decode(&self, token: &str) -> Result<JwtClaims> {
        // Split into header.payload.signature using byte offsets instead of
        // collecting into a Vec. The signing input is header.payload (the
        // original bytes up to the last dot), avoiding a format! allocation.
        let last_dot = token.rfind('.').ok_or_else(|| {
            ZyronError::InvalidCredential("JWT must have three dot-separated parts".to_string())
        })?;
        let first_dot = token[..last_dot].find('.').ok_or_else(|| {
            ZyronError::InvalidCredential("JWT must have three dot-separated parts".to_string())
        })?;

        let signing_input = &token[..last_dot]; // header.payload (no allocation)
        let sig_b64 = &token[last_dot + 1..];
        let payload_b64 = &token[first_dot + 1..last_dot];

        let presented_sig = base64url_decode(sig_b64)?;
        let expected_sig = self.sign(signing_input.as_bytes());

        if !balloon::constant_time_eq(&presented_sig, expected_sig.as_slice()) {
            return Err(ZyronError::InvalidCredential(
                "JWT signature verification failed".to_string(),
            ));
        }

        let payload_bytes = base64url_decode(payload_b64)?;
        let payload_str = std::str::from_utf8(&payload_bytes).map_err(|_| {
            ZyronError::InvalidCredential("JWT payload is not valid UTF-8".to_string())
        })?;
        let claims = json_to_claims(payload_str)?;

        if claims.exp == 0 {
            return Err(ZyronError::InvalidCredential(
                "JWT missing exp claim".to_string(),
            ));
        }

        // Check expiration against current system time.
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0);
        if now > claims.exp {
            return Err(ZyronError::InvalidCredential("JWT has expired".to_string()));
        }

        // Check issuer if configured.
        if let Some(ref expected_iss) = self.issuer {
            match &claims.iss {
                Some(iss) if iss == expected_iss => {}
                _ => {
                    return Err(ZyronError::InvalidCredential(
                        "JWT issuer mismatch".to_string(),
                    ));
                }
            }
        }

        Ok(claims)
    }

    /// Decodes a JWT without verifying the signature. Returns header and claims.
    pub fn decode_unverified(token: &str) -> Result<(JwtHeader, JwtClaims)> {
        let parts: Vec<&str> = token.split('.').collect();
        if parts.len() != 3 {
            return Err(ZyronError::InvalidCredential(
                "JWT must have three dot-separated parts".to_string(),
            ));
        }

        let header_bytes = base64url_decode(parts[0])?;
        let header_str = std::str::from_utf8(&header_bytes).map_err(|_| {
            ZyronError::InvalidCredential("JWT header is not valid UTF-8".to_string())
        })?;
        let header = json_to_header(header_str)?;

        let payload_bytes = base64url_decode(parts[1])?;
        let payload_str = std::str::from_utf8(&payload_bytes).map_err(|_| {
            ZyronError::InvalidCredential("JWT payload is not valid UTF-8".to_string())
        })?;
        let claims = json_to_claims(payload_str)?;

        Ok((header, claims))
    }

    /// Computes the HMAC signature into a stack-allocated SignatureBuf from
    /// the padded key blocks, which costs two hash passes and no allocation
    #[inline(always)]
    fn sign(&self, input: &[u8]) -> SignatureBuf {
        let mut buf = SignatureBuf {
            bytes: [0u8; 64],
            len: 0,
        };
        match &self.mac_key {
            JwtMacKey::Hs256(key) => {
                buf.bytes[..32].copy_from_slice(&key.sign(input));
                buf.len = 32;
            }
            JwtMacKey::Hs384(key) => {
                buf.bytes[..48].copy_from_slice(&key.sign(input));
                buf.len = 48;
            }
            JwtMacKey::Hs512(key) => {
                buf.bytes[..64].copy_from_slice(&key.sign(input));
                buf.len = 64;
            }
        }
        buf
    }

    // -----------------------------------------------------------------------------
    // Audience and scope validation
    // -----------------------------------------------------------------------------

    /// Decodes the token and confirms the `aud` claim matches `required_aud`.
    /// The `aud` claim may be a single string or a JSON array of strings.
    pub fn verify_with_audience(&self, token: &str, required_aud: &str) -> Result<JwtClaims> {
        let claims = self.decode(token)?;
        check_audience(&claims, required_aud)?;
        Ok(claims)
    }

    /// Decodes the token and confirms the `scope` claim contains
    /// `required_scope`. The `scope` claim may be a space-separated string or
    /// a JSON array of strings.
    pub fn verify_with_scope(&self, token: &str, required_scope: &str) -> Result<JwtClaims> {
        let claims = self.decode(token)?;
        check_scopes(&claims, &[required_scope])?;
        Ok(claims)
    }

    /// Decodes the token and applies the configured subset of audience,
    /// scope, and issuer checks.
    pub fn verify_full(
        &self,
        token: &str,
        required_aud: Option<&str>,
        required_scopes: &[&str],
        required_iss: Option<&str>,
    ) -> Result<JwtClaims> {
        let claims = self.decode(token)?;
        if let Some(aud) = required_aud {
            check_audience(&claims, aud)?;
        }
        if !required_scopes.is_empty() {
            check_scopes(&claims, required_scopes)?;
        }
        if let Some(iss) = required_iss {
            match &claims.iss {
                Some(got) if got == iss => {}
                _ => {
                    return Err(ZyronError::InvalidCredential(
                        "JWT issuer does not match required value".to_string(),
                    ));
                }
            }
        }
        Ok(claims)
    }
}

// -----------------------------------------------------------------------------
// Claim matchers shared by verify_with_audience / verify_with_scope / verify_full
// -----------------------------------------------------------------------------

/// Checks that the claims include the required audience. Accepts either a
/// single string `aud` claim or a JSON array encoded as the custom field.
fn check_audience(claims: &JwtClaims, required: &str) -> Result<()> {
    if let Some(aud) = claims.custom.get("aud") {
        if aud == required {
            return Ok(());
        }
        // Array form. The JWT parser stores a raw array as the verbatim text
        // `["a","b"]` in a custom field, so split by `"` and scan tokens.
        if aud.starts_with('[') && aud.ends_with(']') {
            let inner = &aud[1..aud.len() - 1];
            for part in inner.split(',') {
                let p = part.trim().trim_matches('"');
                if p == required {
                    return Ok(());
                }
            }
        }
        return Err(ZyronError::InvalidCredential(format!(
            "JWT audience does not include {}",
            required
        )));
    }
    Err(ZyronError::InvalidCredential(
        "JWT missing aud claim".to_string(),
    ))
}

/// Checks that the claims include every requested scope. The `scope` claim is
/// space-separated per RFC 8693. Array form is accepted as a tolerant
/// fallback.
fn check_scopes(claims: &JwtClaims, required: &[&str]) -> Result<()> {
    let scope = claims
        .custom
        .get("scope")
        .or_else(|| claims.custom.get("scp"))
        .ok_or_else(|| ZyronError::InvalidCredential("JWT missing scope claim".to_string()))?;
    let mut have: Vec<&str> = Vec::new();
    if scope.starts_with('[') && scope.ends_with(']') {
        let inner = &scope[1..scope.len() - 1];
        for part in inner.split(',') {
            have.push(part.trim().trim_matches('"'));
        }
    } else {
        for part in scope.split_whitespace() {
            have.push(part);
        }
    }
    for r in required {
        if !have.iter().any(|h| h == r) {
            return Err(ZyronError::InvalidCredential(format!(
                "JWT scope missing required {}",
                r
            )));
        }
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// TOTP (RFC 6238)
// ---------------------------------------------------------------------------

/// Time-based One-Time Password credential using HMAC-SHA1.
/// Implements RFC 6238 with configurable digits and period.
/// Pre-computes the HMAC key schedule so repeated generate/verify
/// calls skip the key pad derivation (XOR of key with ipad/opad).
pub struct TotpCredential {
    secret: Vec<u8>,
    digits: u32,
    period: u64,
    /// Pre-computed HMAC-SHA1 instance cloned per operation. Avoids
    /// re-deriving the inner/outer key pads on every generate_code call.
    hmac_template: Hmac<sha1::Sha1>,
}

impl TotpCredential {
    /// Generates a new TOTP credential with a random 20-byte secret.
    pub fn generate() -> Self {
        use rand::Rng;
        let mut secret = vec![0u8; 20];
        rand::rng().fill_bytes(&mut secret);
        let hmac_template =
            Hmac::<sha1::Sha1>::new_from_slice(&secret).expect("HMAC-SHA1 accepts any key length");
        Self {
            secret,
            digits: 6,
            period: 30,
            hmac_template,
        }
    }

    /// Creates a TOTP credential from an existing secret.
    pub fn from_secret(secret: Vec<u8>) -> Self {
        let hmac_template =
            Hmac::<sha1::Sha1>::new_from_slice(&secret).expect("HMAC-SHA1 accepts any key length");
        Self {
            secret,
            digits: 6,
            period: 30,
            hmac_template,
        }
    }

    /// Returns the raw TOTP integer for the given timestamp, avoiding String allocation.
    /// Clones the pre-computed HMAC template instead of re-deriving key pads.
    fn generate_code_raw(&self, timestamp: u64) -> u32 {
        let counter = timestamp / self.period;
        let counter_bytes = counter.to_be_bytes();

        let mut mac = self.hmac_template.clone();
        mac.update(&counter_bytes);
        let result = mac.finalize().into_bytes();

        // Dynamic truncation per RFC 4226 section 5.4.
        let offset = (result[19] & 0x0f) as usize;
        let binary = ((result[offset] as u32 & 0x7f) << 24)
            | ((result[offset + 1] as u32) << 16)
            | ((result[offset + 2] as u32) << 8)
            | (result[offset + 3] as u32);

        let modulus = 10u32.pow(self.digits);
        binary % modulus
    }

    /// Generates the TOTP code for the given unix timestamp.
    /// counter = timestamp / period, then HMAC-SHA1 with dynamic truncation.
    pub fn generate_code(&self, timestamp: u64) -> String {
        let code = self.generate_code_raw(timestamp);
        format!("{:0>width$}", code, width = self.digits as usize)
    }

    /// Verifies a TOTP code, checking the current period and one period before and after
    /// to account for clock drift. Uses integer comparison to avoid String allocation.
    pub fn verify(&self, code: &str, timestamp: u64) -> bool {
        let parsed = match code.parse::<u32>() {
            Ok(v) => v,
            Err(_) => return false,
        };
        let modulus = 10u32.pow(self.digits);
        if parsed >= modulus {
            return false;
        }

        // Check current window first (most likely match), then +/- 1 period.
        if self.generate_code_raw(timestamp) == parsed {
            return true;
        }
        if self.generate_code_raw(timestamp.saturating_sub(self.period)) == parsed {
            return true;
        }
        if self.generate_code_raw(timestamp.saturating_add(self.period)) == parsed {
            return true;
        }
        false
    }

    /// Returns the secret encoded in base32 (RFC 4648) for QR code generation.
    pub fn secret_base32(&self) -> String {
        base32_encode(&self.secret)
    }

    /// Returns the raw secret bytes.
    pub fn secret(&self) -> &[u8] {
        &self.secret
    }
}

// ---------------------------------------------------------------------------
// Helper functions
// ---------------------------------------------------------------------------

/// SHA-256 hash of a byte slice.
fn sha256_hash(data: &[u8]) -> [u8; 32] {
    let mut hasher = Sha256::new();
    hasher.update(data);
    let result = hasher.finalize();
    let mut out = [0u8; 32];
    out.copy_from_slice(&result);
    out
}

/// Base64url encoding without padding (RFC 4648 section 5).
fn base64url_encode(data: &[u8]) -> String {
    use base64::Engine;
    base64::engine::general_purpose::URL_SAFE_NO_PAD.encode(data)
}

/// Base64url decoding without padding.
fn base64url_decode(s: &str) -> Result<Vec<u8>> {
    use base64::Engine;
    base64::engine::general_purpose::URL_SAFE_NO_PAD
        .decode(s)
        .map_err(|_| ZyronError::InvalidCredential("Invalid base64url encoding".to_string()))
}

/// Base32 encoding (RFC 4648) without padding.
fn base32_encode(data: &[u8]) -> String {
    const ALPHABET: &[u8; 32] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZ234567";
    let mut result = String::with_capacity((data.len() * 8 + 4) / 5);
    let mut buffer: u64 = 0;
    let mut bits_in_buffer = 0;

    for &byte in data {
        buffer = (buffer << 8) | byte as u64;
        bits_in_buffer += 8;
        while bits_in_buffer >= 5 {
            bits_in_buffer -= 5;
            let index = ((buffer >> bits_in_buffer) & 0x1f) as usize;
            result.push(ALPHABET[index] as char);
        }
    }
    // Flush remaining bits (left-padded with zeros).
    if bits_in_buffer > 0 {
        let index = ((buffer << (5 - bits_in_buffer)) & 0x1f) as usize;
        result.push(ALPHABET[index] as char);
    }

    result
}

/// Serializes JWT claims to a JSON string without serde_json
/// Uses fmt::Write for u64 fields to avoid intermediate String allocation
/// from .to_string()
fn claims_to_json(claims: &JwtClaims) -> String {
    use std::fmt::Write;
    let mut s = String::with_capacity(256);
    s.push('{');

    s.push_str("\"sub\":\"");
    json_escape_into(&mut s, &claims.sub);
    s.push('"');

    if let Some(ref iss) = claims.iss {
        s.push_str(",\"iss\":\"");
        json_escape_into(&mut s, iss);
        s.push('"');
    }

    s.push_str(",\"exp\":");
    let _ = write!(&mut s, "{}", claims.exp);

    s.push_str(",\"iat\":");
    let _ = write!(&mut s, "{}", claims.iat);

    if !claims.roles.is_empty() {
        s.push_str(",\"roles\":[");
        for (i, role) in claims.roles.iter().enumerate() {
            if i > 0 {
                s.push(',');
            }
            s.push('"');
            json_escape_into(&mut s, role);
            s.push('"');
        }
        s.push(']');
    }

    for (key, value) in &claims.custom {
        s.push(',');
        s.push('"');
        json_escape_into(&mut s, key);
        s.push_str("\":\"");
        json_escape_into(&mut s, value);
        s.push('"');
    }

    s.push('}');
    s
}

/// Unescapes a JSON string value (handles \\, \", \n, \r, \t).
fn json_unescape(s: &str) -> String {
    let mut result = String::with_capacity(s.len());
    let mut chars = s.chars();
    while let Some(ch) = chars.next() {
        if ch == '\\' {
            match chars.next() {
                Some('"') => result.push('"'),
                Some('\\') => result.push('\\'),
                Some('n') => result.push('\n'),
                Some('r') => result.push('\r'),
                Some('t') => result.push('\t'),
                Some('/') => result.push('/'),
                Some(other) => {
                    result.push('\\');
                    result.push(other);
                }
                None => result.push('\\'),
            }
        } else {
            result.push(ch);
        }
    }
    result
}

/// Escapes a string for JSON output (handles backslash, quote, control chars).
fn json_escape_into(buf: &mut String, s: &str) {
    for ch in s.chars() {
        match ch {
            '"' => buf.push_str("\\\""),
            '\\' => buf.push_str("\\\\"),
            '\n' => buf.push_str("\\n"),
            '\r' => buf.push_str("\\r"),
            '\t' => buf.push_str("\\t"),
            c if (c as u32) < 0x20 => {
                buf.push_str(&format!("\\u{:04x}", c as u32));
            }
            c => buf.push(c),
        }
    }
}

/// Parses a minimal JSON object into JwtHeader.
fn json_to_header(json: &str) -> Result<JwtHeader> {
    let alg = extract_json_string(json, "alg")?
        .ok_or_else(|| ZyronError::InvalidCredential("JWT header missing alg".to_string()))?;
    let typ = extract_json_string(json, "typ")?.unwrap_or_else(|| "JWT".to_string());
    Ok(JwtHeader { alg, typ })
}

/// Parses a minimal JSON object into JwtClaims using a single pass.
/// Scans the JSON once to extract all key-value pairs instead of running
/// separate find() calls per field.
fn json_to_claims(json: &str) -> Result<JwtClaims> {
    // Fast path: single-pass extraction for the common case where claims
    // contain only standard fields with simple string/number values.
    let mut sub: Option<String> = None;
    let mut iss: Option<String> = None;
    let mut exp: Option<u64> = None;
    let mut iat: u64 = 0;
    let mut roles: Vec<String> = Vec::new();
    let mut custom = std::collections::HashMap::new();

    let trimmed = json.trim();
    if !trimmed.starts_with('{') || !trimmed.ends_with('}') {
        return Err(ZyronError::InvalidCredential(
            "JWT claims must be a JSON object".to_string(),
        ));
    }
    let inner = &trimmed[1..trimmed.len() - 1];
    let bytes = inner.as_bytes();
    let len = bytes.len();
    let mut pos = 0;

    while pos < len {
        // Skip whitespace and commas
        while pos < len
            && (bytes[pos] == b' '
                || bytes[pos] == b','
                || bytes[pos] == b'\n'
                || bytes[pos] == b'\r'
                || bytes[pos] == b'\t')
        {
            pos += 1;
        }
        if pos >= len {
            break;
        }

        // Parse key
        if bytes[pos] != b'"' {
            pos += 1;
            continue;
        }
        pos += 1;
        let key_start = pos;
        while pos < len && bytes[pos] != b'"' {
            pos += 1;
        }
        let key = &inner[key_start..pos];
        pos += 1; // closing quote

        // Skip colon and whitespace
        while pos < len && (bytes[pos] == b':' || bytes[pos] == b' ') {
            pos += 1;
        }
        if pos >= len {
            break;
        }

        // Parse value based on first character
        if bytes[pos] == b'"' {
            // String value
            pos += 1;
            let val_start = pos;
            let mut has_escape = false;
            while pos < len && bytes[pos] != b'"' {
                if bytes[pos] == b'\\' {
                    has_escape = true;
                    pos += 1;
                }
                pos += 1;
            }
            let raw_val = &inner[val_start..pos];
            pos += 1; // closing quote

            let val = if has_escape {
                json_unescape(raw_val)
            } else {
                raw_val.to_string()
            };

            match key {
                "sub" => sub = Some(val),
                "iss" => iss = Some(val),
                _ => {
                    custom.insert(key.to_string(), val);
                }
            }
        } else if bytes[pos] == b'[' {
            // Array value. `roles` decomposes into the typed Vec. Other
            // arrays (for example `aud`, `scp`) land in custom as the raw
            // JSON text so claim validators can inspect them.
            let arr_start = pos;
            pos += 1;
            if key == "roles" {
                while pos < len && bytes[pos] != b']' {
                    while pos < len && bytes[pos] != b'"' && bytes[pos] != b']' {
                        pos += 1;
                    }
                    if pos >= len || bytes[pos] == b']' {
                        break;
                    }
                    pos += 1; // opening quote
                    let val_start = pos;
                    while pos < len && bytes[pos] != b'"' {
                        if bytes[pos] == b'\\' {
                            pos += 1;
                        }
                        pos += 1;
                    }
                    roles.push(inner[val_start..pos].to_string());
                    pos += 1; // closing quote
                }
            }
            // Skip to end of array
            while pos < len && bytes[pos] != b']' {
                pos += 1;
            }
            if pos < len {
                pos += 1;
            }
            if key != "roles" && key != "sub" && key != "iss" && key != "exp" && key != "iat" {
                custom.insert(key.to_string(), inner[arr_start..pos].to_string());
            }
        } else {
            // Number or other literal
            let val_start = pos;
            while pos < len && bytes[pos] != b',' && bytes[pos] != b'}' && bytes[pos] != b' ' {
                pos += 1;
            }
            let num_str = inner[val_start..pos].trim();
            if let Ok(n) = num_str.parse::<u64>() {
                match key {
                    "exp" => exp = Some(n),
                    "iat" => iat = n,
                    _ => {}
                }
            }
        }
    }

    let sub =
        sub.ok_or_else(|| ZyronError::InvalidCredential("JWT claims missing sub".to_string()))?;
    let exp =
        exp.ok_or_else(|| ZyronError::InvalidCredential("JWT claims missing exp".to_string()))?;

    Ok(JwtClaims {
        sub,
        iss,
        exp,
        iat,
        roles,
        custom,
    })
}

/// Extracts a string value for a given key from a JSON object.
fn extract_json_string(json: &str, key: &str) -> Result<Option<String>> {
    let search = format!("\"{}\":\"", key);
    // Find the key as a top-level JSON key (preceded by '{' or ',')
    // to avoid matching key names inside string values.
    let start = {
        let mut search_from = 0;
        loop {
            let pos = match json[search_from..].find(&search) {
                Some(p) => search_from + p,
                None => return Ok(None),
            };
            let before = json[..pos].trim_end();
            if before.ends_with('{') || before.ends_with(',') {
                break pos + search.len();
            }
            search_from = pos + 1;
            if search_from >= json.len() {
                return Ok(None);
            }
        }
    };

    let mut result = String::new();
    let mut chars = json[start..].chars();
    let mut escaped = false;

    loop {
        let ch = chars.next().ok_or_else(|| {
            ZyronError::InvalidCredential(format!("Unterminated string for key \"{}\"", key))
        })?;
        if escaped {
            match ch {
                '"' => result.push('"'),
                '\\' => result.push('\\'),
                '/' => result.push('/'),
                'n' => result.push('\n'),
                'r' => result.push('\r'),
                't' => result.push('\t'),
                'b' => result.push('\u{0008}'),
                'f' => result.push('\u{000C}'),
                'u' => {
                    // Parse \uXXXX unicode escape (4 hex digits).
                    let mut hex = String::with_capacity(4);
                    for _ in 0..4 {
                        let h = chars.next().ok_or_else(|| {
                            ZyronError::InvalidCredential(format!(
                                "Truncated \\u escape in key \"{}\"",
                                key
                            ))
                        })?;
                        hex.push(h);
                    }
                    let code_point = u32::from_str_radix(&hex, 16).map_err(|_| {
                        ZyronError::InvalidCredential(format!(
                            "Invalid \\u escape in key \"{}\"",
                            key
                        ))
                    })?;
                    let c = char::from_u32(code_point).ok_or_else(|| {
                        ZyronError::InvalidCredential(format!(
                            "Invalid unicode code point in key \"{}\"",
                            key
                        ))
                    })?;
                    result.push(c);
                }
                _ => {
                    result.push('\\');
                    result.push(ch);
                }
            }
            escaped = false;
        } else if ch == '\\' {
            escaped = true;
        } else if ch == '"' {
            break;
        } else {
            result.push(ch);
        }
    }

    Ok(Some(result))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_password_roundtrip() {
        let params = BalloonParams::test();
        let cred = PasswordCredential::from_plaintext_with_params("secret123", &params)
            .expect("hash failed");
        assert!(cred.verify("secret123").expect("verify failed"));
        assert!(!cred.verify("wrong").expect("verify failed"));
    }

    #[test]
    fn test_password_from_stored() {
        let params = BalloonParams::test();
        let cred =
            PasswordCredential::from_plaintext_with_params("mypass", &params).expect("hash failed");
        let stored = cred.as_stored().to_string();
        let restored = PasswordCredential::from_stored(stored);
        assert!(restored.verify("mypass").expect("verify failed"));
    }

    #[test]
    fn test_scram_secret_format_and_parse() {
        let secret = scram_sha256_secret("hunter2");
        assert!(secret.starts_with("SCRAM-SHA-256$4096:"));
        let parsed = parse_scram_secret(&secret).expect("parse");
        assert_eq!(parsed.iterations, 4096);
        assert_eq!(parsed.salt.len(), 16);
        assert_eq!(parsed.stored_key.len(), 32);
        assert_eq!(parsed.server_key.len(), 32);
    }

    #[test]
    fn test_scram_secret_deterministic_with_salt() {
        let salt = [0x11u8; 16];
        let a = scram_sha256_secret_with_salt("pw", &salt, 4096);
        let b = scram_sha256_secret_with_salt("pw", &salt, 4096);
        assert_eq!(a, b);
        let c = scram_sha256_secret_with_salt("other", &salt, 4096);
        assert_ne!(a, c);
    }

    #[test]
    fn test_parse_scram_secret_rejects_garbage() {
        assert!(parse_scram_secret("not-a-secret").is_err());
        assert!(parse_scram_secret("SCRAM-SHA-256$4096:").is_err());
    }

    #[test]
    fn test_md5_credential_known_vector() {
        // md5("passuser") = the inner PostgreSQL value for password "pass",
        // user "user".
        let inner = md5_password_credential("user", "pass");
        assert_eq!(inner.len(), 32);
        assert!(inner.chars().all(|c| c.is_ascii_hexdigit()));
        // Deterministic and dependent on both inputs.
        assert_eq!(inner, md5_password_credential("user", "pass"));
        assert_ne!(inner, md5_password_credential("user", "other"));
        assert_ne!(inner, md5_password_credential("other", "pass"));
    }

    #[test]
    fn test_api_key_generate_and_verify() {
        let (cred, full_key) = ApiKeyCredential::generate();
        assert!(full_key.starts_with("zyron_"));
        assert!(cred.prefix().starts_with("zyron_"));
        assert!(cred.verify(&full_key));
        assert!(!cred.verify("zyron_wrongkey12345678901234567890"));
    }

    #[test]
    fn test_api_key_from_stored() {
        let (cred, full_key) = ApiKeyCredential::generate();
        let restored = ApiKeyCredential::from_stored(cred.prefix().to_string(), *cred.key_hash());
        assert!(restored.verify(&full_key));
    }

    #[test]
    fn test_jwt_secret_too_short() {
        let short_secret = vec![0u8; 16];
        let result = JwtCredential::new(short_secret, JwtAlgorithm::Hs256);
        assert!(result.is_err());
    }

    #[test]
    fn test_jwt_encode_decode_hs256() {
        let secret = vec![0xab; 32];
        let cred = JwtCredential::new(secret, JwtAlgorithm::Hs256).expect("create failed");

        let claims = JwtClaims {
            sub: "user42".to_string(),
            iss: None,
            exp: 9999999999,
            iat: 1000000000,
            roles: vec!["admin".to_string(), "reader".to_string()],
            custom: std::collections::HashMap::new(),
        };

        let token = cred.encode(&claims).expect("encode failed");
        let decoded = cred.decode(&token).expect("decode failed");

        assert_eq!(decoded.sub, "user42");
        assert_eq!(decoded.exp, 9999999999);
        assert_eq!(decoded.iat, 1000000000);
        assert_eq!(decoded.roles, vec!["admin", "reader"]);
    }

    #[test]
    fn test_jwt_encode_decode_hs384() {
        let secret = vec![0xcd; 48];
        let cred = JwtCredential::new(secret, JwtAlgorithm::Hs384).expect("create failed");

        let claims = JwtClaims {
            sub: "user99".to_string(),
            iss: Some("zyron".to_string()),
            exp: 9999999999,
            iat: 1000000000,
            roles: Vec::new(),
            custom: std::collections::HashMap::new(),
        };

        let token = cred.encode(&claims).expect("encode failed");
        let cred_with_iss = JwtCredential::new(vec![0xcd; 48], JwtAlgorithm::Hs384)
            .expect("create failed")
            .with_issuer("zyron".to_string());
        let decoded = cred_with_iss.decode(&token).expect("decode failed");
        assert_eq!(decoded.sub, "user99");
        assert_eq!(decoded.iss, Some("zyron".to_string()));
    }

    #[test]
    fn test_jwt_encode_decode_hs512() {
        let secret = vec![0xef; 64];
        let cred = JwtCredential::new(secret, JwtAlgorithm::Hs512).expect("create failed");

        let claims = JwtClaims {
            sub: "svc_account".to_string(),
            iss: None,
            exp: 9999999999,
            iat: 1000000000,
            roles: vec!["service".to_string()],
            custom: std::collections::HashMap::new(),
        };

        let token = cred.encode(&claims).expect("encode failed");
        let decoded = cred.decode(&token).expect("decode failed");
        assert_eq!(decoded.sub, "svc_account");
    }

    #[test]
    fn test_jwt_issuer_mismatch() {
        let secret = vec![0xab; 32];
        let cred = JwtCredential::new(secret, JwtAlgorithm::Hs256)
            .expect("create failed")
            .with_issuer("expected_issuer".to_string());

        let claims = JwtClaims {
            sub: "user1".to_string(),
            iss: Some("wrong_issuer".to_string()),
            exp: 9999999999,
            iat: 1000000000,
            roles: Vec::new(),
            custom: std::collections::HashMap::new(),
        };

        // Encode with a different credential that has no issuer check.
        let encoder =
            JwtCredential::new(vec![0xab; 32], JwtAlgorithm::Hs256).expect("create failed");
        let token = encoder.encode(&claims).expect("encode failed");
        let result = cred.decode(&token);
        assert!(result.is_err());
    }

    #[test]
    fn test_jwt_tampered_signature() {
        let secret = vec![0xab; 32];
        let cred = JwtCredential::new(secret, JwtAlgorithm::Hs256).expect("create failed");

        let claims = JwtClaims {
            sub: "user1".to_string(),
            iss: None,
            exp: 9999999999,
            iat: 1000000000,
            roles: Vec::new(),
            custom: std::collections::HashMap::new(),
        };

        let mut token = cred.encode(&claims).expect("encode failed");
        // Tamper with the last character of the signature.
        let last = token.pop();
        match last {
            Some('A') => token.push('B'),
            _ => token.push('A'),
        }
        assert!(cred.decode(&token).is_err());
    }

    #[test]
    fn test_jwt_decode_unverified() {
        let secret = vec![0xab; 32];
        let cred = JwtCredential::new(secret, JwtAlgorithm::Hs256).expect("create failed");

        let claims = JwtClaims {
            sub: "peek_user".to_string(),
            iss: None,
            exp: 9999999999,
            iat: 1000000000,
            roles: vec!["viewer".to_string()],
            custom: std::collections::HashMap::new(),
        };

        let token = cred.encode(&claims).expect("encode failed");
        let (header, decoded_claims) =
            JwtCredential::decode_unverified(&token).expect("decode failed");
        assert_eq!(header.alg, "HS256");
        assert_eq!(header.typ, "JWT");
        assert_eq!(decoded_claims.sub, "peek_user");
    }

    #[test]
    fn test_jwt_custom_fields() {
        let secret = vec![0xab; 32];
        let cred = JwtCredential::new(secret, JwtAlgorithm::Hs256).expect("create failed");

        let mut custom = std::collections::HashMap::new();
        custom.insert("tenant".to_string(), "acme".to_string());
        custom.insert("env".to_string(), "prod".to_string());

        let claims = JwtClaims {
            sub: "user1".to_string(),
            iss: None,
            exp: 9999999999,
            iat: 1000000000,
            roles: Vec::new(),
            custom,
        };

        let token = cred.encode(&claims).expect("encode failed");
        let decoded = cred.decode(&token).expect("decode failed");
        assert_eq!(decoded.custom.get("tenant"), Some(&"acme".to_string()));
        assert_eq!(decoded.custom.get("env"), Some(&"prod".to_string()));
    }

    #[test]
    fn test_jwt_missing_exp() {
        // Manually construct a token with exp=0 (treated as missing).
        let secret = vec![0xab; 32];
        let cred = JwtCredential::new(secret, JwtAlgorithm::Hs256).expect("create failed");

        let claims = JwtClaims {
            sub: "user1".to_string(),
            iss: None,
            exp: 0,
            iat: 1000000000,
            roles: Vec::new(),
            custom: std::collections::HashMap::new(),
        };

        let token = cred.encode(&claims).expect("encode failed");
        assert!(cred.decode(&token).is_err());
    }

    #[test]
    fn test_jwt_invalid_format() {
        let secret = vec![0xab; 32];
        let cred = JwtCredential::new(secret, JwtAlgorithm::Hs256).expect("create failed");
        assert!(cred.decode("not.a.valid.jwt.token").is_err());
        assert!(cred.decode("nodotsatall").is_err());
    }

    #[test]
    fn test_totp_generate_code_known_vector() {
        // Known test vector: secret = "12345678901234567890" (ASCII),
        // timestamp 59, period 30 -> counter 1.
        // RFC 6238 test vector for SHA1: counter=1 -> code 287082.
        let secret = b"12345678901234567890".to_vec();
        let totp = TotpCredential::from_secret(secret);
        let code = totp.generate_code(59);
        assert_eq!(code, "287082");
    }

    #[test]
    fn test_totp_verify_current_window() {
        let totp = TotpCredential::generate();
        let timestamp = 1000000000u64;
        let code = totp.generate_code(timestamp);
        assert!(totp.verify(&code, timestamp));
    }

    #[test]
    fn test_totp_verify_adjacent_window() {
        let totp = TotpCredential::generate();
        let timestamp = 1000000000u64;
        // Generate code for current period, verify with a timestamp one period later.
        let code = totp.generate_code(timestamp);
        assert!(totp.verify(&code, timestamp + 30));
    }

    #[test]
    fn test_totp_verify_wrong_code() {
        let totp = TotpCredential::generate();
        let timestamp = 1000000000u64;
        assert!(!totp.verify("000000", timestamp));
    }

    #[test]
    fn test_totp_secret_base32() {
        let secret = vec![0x48, 0x65, 0x6c, 0x6c, 0x6f]; // "Hello"
        let totp = TotpCredential::from_secret(secret);
        let b32 = totp.secret_base32();
        assert_eq!(b32, "JBSWY3DP"); // Known base32 encoding of "Hello" (with trailing bits)
    }

    #[test]
    fn test_totp_from_secret_roundtrip() {
        let original = TotpCredential::generate();
        let secret_copy = original.secret().to_vec();
        let restored = TotpCredential::from_secret(secret_copy);
        let ts = 1000000000u64;
        assert_eq!(original.generate_code(ts), restored.generate_code(ts));
    }

    #[test]
    fn test_base32_encode_empty() {
        assert_eq!(base32_encode(&[]), "");
    }

    #[test]
    fn test_base32_encode_single_byte() {
        // 0x00 -> 00000 000 -> "AA" (first 5 bits = 0 -> A, remaining 3 bits padded -> 0 -> A)
        assert_eq!(base32_encode(&[0x00]), "AA");
    }

    #[test]
    fn test_base64url_roundtrip() {
        let data = b"Hello, Zyron!";
        let encoded = base64url_encode(data);
        let decoded = base64url_decode(&encoded).expect("decode failed");
        assert_eq!(decoded, data);
    }

    #[test]
    fn test_json_escape_special_chars() {
        let claims = JwtClaims {
            sub: "user\"with\\quotes".to_string(),
            iss: None,
            exp: 100,
            iat: 50,
            roles: Vec::new(),
            custom: std::collections::HashMap::new(),
        };
        let json = claims_to_json(&claims);
        assert!(json.contains("user\\\"with\\\\quotes"));
    }

    #[test]
    fn test_claims_json_roundtrip() {
        let mut custom = std::collections::HashMap::new();
        custom.insert("key1".to_string(), "val1".to_string());

        let claims = JwtClaims {
            sub: "testuser".to_string(),
            iss: Some("zyron".to_string()),
            exp: 9999999999,
            iat: 1000000000,
            roles: vec!["admin".to_string()],
            custom,
        };

        let json = claims_to_json(&claims);
        let parsed = json_to_claims(&json).expect("parse failed");
        assert_eq!(parsed.sub, claims.sub);
        assert_eq!(parsed.iss, claims.iss);
        assert_eq!(parsed.exp, claims.exp);
        assert_eq!(parsed.iat, claims.iat);
        assert_eq!(parsed.roles, claims.roles);
        assert_eq!(parsed.custom.get("key1"), Some(&"val1".to_string()));
    }

    #[test]
    fn test_sha256_hash_deterministic() {
        let h1 = sha256_hash(b"test data");
        let h2 = sha256_hash(b"test data");
        assert_eq!(h1, h2);
        let h3 = sha256_hash(b"different");
        assert_ne!(h1, h3);
    }

    fn make_claims(
        custom: std::collections::HashMap<String, String>,
        issuer: Option<String>,
    ) -> JwtClaims {
        JwtClaims {
            sub: "svc".to_string(),
            iss: issuer,
            exp: 9999999999,
            iat: 0,
            roles: Vec::new(),
            custom,
        }
    }

    fn issue_token(claims: &JwtClaims) -> (JwtCredential, String) {
        let cred = JwtCredential::new(vec![0xa1; 32], JwtAlgorithm::Hs256).expect("cred");
        let token = cred.encode(claims).expect("encode");
        (cred, token)
    }

    #[test]
    fn verify_audience_single_string_match() {
        let mut custom = std::collections::HashMap::new();
        custom.insert("aud".to_string(), "zyron-prod".to_string());
        let (cred, tok) = issue_token(&make_claims(custom, None));
        let c = cred
            .verify_with_audience(&tok, "zyron-prod")
            .expect("verify");
        assert_eq!(c.sub, "svc");
    }

    #[test]
    fn verify_audience_string_mismatch() {
        let mut custom = std::collections::HashMap::new();
        custom.insert("aud".to_string(), "zyron-dev".to_string());
        let (cred, tok) = issue_token(&make_claims(custom, None));
        assert!(cred.verify_with_audience(&tok, "zyron-prod").is_err());
    }

    #[test]
    fn verify_audience_array_match() {
        // Construct JWT with array audience by hand-crafting payload JSON.
        let cred = JwtCredential::new(vec![0xa2; 32], JwtAlgorithm::Hs256).expect("cred");
        let payload = r#"{"sub":"svc","exp":9999999999,"aud":["a","zyron-prod","b"]}"#;
        let header_b64 = base64url_encode(b"{\"alg\":\"HS256\",\"typ\":\"JWT\"}");
        let payload_b64 = base64url_encode(payload.as_bytes());
        let signing_input = format!("{}.{}", header_b64, payload_b64);
        let sig = cred.sign(signing_input.as_bytes());
        let sig_b64 = base64url_encode(sig.as_slice());
        let token = format!("{}.{}", signing_input, sig_b64);

        let c = cred
            .verify_with_audience(&token, "zyron-prod")
            .expect("verify");
        assert_eq!(c.sub, "svc");
    }

    #[test]
    fn verify_scope_single_present() {
        let mut custom = std::collections::HashMap::new();
        custom.insert(
            "scope".to_string(),
            "read:publications write:subs".to_string(),
        );
        let (cred, tok) = issue_token(&make_claims(custom, None));
        cred.verify_with_scope(&tok, "read:publications")
            .expect("verify");
    }

    #[test]
    fn verify_scope_missing() {
        let mut custom = std::collections::HashMap::new();
        custom.insert("scope".to_string(), "read:publications".to_string());
        let (cred, tok) = issue_token(&make_claims(custom, None));
        assert!(
            cred.verify_with_scope(&tok, "publication.subscribe:alpha")
                .is_err()
        );
    }

    #[test]
    fn verify_full_all_checks() {
        let mut custom = std::collections::HashMap::new();
        custom.insert("aud".to_string(), "zyron".to_string());
        custom.insert(
            "scope".to_string(),
            "read:publications publication.subscribe:alpha".to_string(),
        );
        let (cred, tok) = issue_token(&make_claims(custom, Some("issuer-x".to_string())));
        let c = cred
            .verify_full(
                &tok,
                Some("zyron"),
                &["publication.subscribe:alpha"],
                Some("issuer-x"),
            )
            .expect("verify");
        assert_eq!(c.sub, "svc");
    }

    #[test]
    fn verify_full_missing_aud_errors() {
        let (cred, tok) = issue_token(&make_claims(std::collections::HashMap::new(), None));
        assert!(cred.verify_full(&tok, Some("zyron"), &[], None).is_err());
    }
}
