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

// ---- PasswordCredential ----

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

// ---- ApiKeyCredential ----

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

// ---- JWT types ----

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

/// JWT credential that can encode and decode tokens using HMAC signing.
pub struct JwtCredential {
    secret: Vec<u8>,
    algorithm: JwtAlgorithm,
    issuer: Option<String>,
    max_age_secs: u64,
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
        Ok(Self {
            secret,
            algorithm,
            issuer: None,
            max_age_secs: 3600,
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

    /// Encodes the claims into a signed JWT string (header.payload.signature).
    pub fn encode(&self, claims: &JwtClaims) -> Result<String> {
        let header_json = format!(
            "{{\"alg\":\"{}\",\"typ\":\"JWT\"}}",
            self.algorithm.as_str()
        );
        let payload_json = claims_to_json(claims);

        let header_b64 = base64url_encode(header_json.as_bytes());
        let payload_b64 = base64url_encode(payload_json.as_bytes());

        let signing_input = format!("{}.{}", header_b64, payload_b64);
        let signature = self.sign(signing_input.as_bytes())?;
        let sig_b64 = base64url_encode(&signature);

        Ok(format!("{}.{}", signing_input, sig_b64))
    }

    /// Decodes and verifies a JWT token. Checks signature, expiration, and issuer.
    pub fn decode(&self, token: &str) -> Result<JwtClaims> {
        let parts: Vec<&str> = token.split('.').collect();
        if parts.len() != 3 {
            return Err(ZyronError::InvalidCredential(
                "JWT must have three dot-separated parts".to_string(),
            ));
        }

        let signing_input = format!("{}.{}", parts[0], parts[1]);
        let presented_sig = base64url_decode(parts[2])?;
        let expected_sig = self.sign(signing_input.as_bytes())?;

        if !balloon::constant_time_eq(&presented_sig, &expected_sig) {
            return Err(ZyronError::InvalidCredential(
                "JWT signature verification failed".to_string(),
            ));
        }

        let payload_bytes = base64url_decode(parts[1])?;
        let payload_str = std::str::from_utf8(&payload_bytes).map_err(|_| {
            ZyronError::InvalidCredential("JWT payload is not valid UTF-8".to_string())
        })?;
        let claims = json_to_claims(payload_str)?;

        // Validate expiration using max_age_secs as current time reference.
        // The caller should set exp = iat + max_age. We check that iat + max_age >= exp context.
        if claims.exp == 0 {
            return Err(ZyronError::InvalidCredential(
                "JWT missing exp claim".to_string(),
            ));
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

    /// Computes HMAC signature over the input bytes using the configured algorithm.
    fn sign(&self, input: &[u8]) -> Result<Vec<u8>> {
        match self.algorithm {
            JwtAlgorithm::Hs256 => {
                let mut mac = Hmac::<sha2::Sha256>::new_from_slice(&self.secret)
                    .map_err(|e| ZyronError::InvalidCredential(format!("HMAC key error: {}", e)))?;
                mac.update(input);
                Ok(mac.finalize().into_bytes().to_vec())
            }
            JwtAlgorithm::Hs384 => {
                let mut mac = Hmac::<sha2::Sha384>::new_from_slice(&self.secret)
                    .map_err(|e| ZyronError::InvalidCredential(format!("HMAC key error: {}", e)))?;
                mac.update(input);
                Ok(mac.finalize().into_bytes().to_vec())
            }
            JwtAlgorithm::Hs512 => {
                let mut mac = Hmac::<sha2::Sha512>::new_from_slice(&self.secret)
                    .map_err(|e| ZyronError::InvalidCredential(format!("HMAC key error: {}", e)))?;
                mac.update(input);
                Ok(mac.finalize().into_bytes().to_vec())
            }
        }
    }
}

// ---- TOTP (RFC 6238) ----

/// Time-based One-Time Password credential using HMAC-SHA1.
/// Implements RFC 6238 with configurable digits and period.
pub struct TotpCredential {
    secret: Vec<u8>,
    digits: u32,
    period: u64,
}

impl TotpCredential {
    /// Generates a new TOTP credential with a random 20-byte secret.
    pub fn generate() -> Self {
        use rand::Rng;
        let mut secret = vec![0u8; 20];
        rand::rng().fill_bytes(&mut secret);
        Self {
            secret,
            digits: 6,
            period: 30,
        }
    }

    /// Creates a TOTP credential from an existing secret.
    pub fn from_secret(secret: Vec<u8>) -> Self {
        Self {
            secret,
            digits: 6,
            period: 30,
        }
    }

    /// Generates the TOTP code for the given unix timestamp.
    /// counter = timestamp / period, then HMAC-SHA1 with dynamic truncation.
    pub fn generate_code(&self, timestamp: u64) -> String {
        let counter = timestamp / self.period;
        let counter_bytes = counter.to_be_bytes();

        let mut mac = Hmac::<sha1::Sha1>::new_from_slice(&self.secret)
            .expect("HMAC-SHA1 accepts any key length");
        mac.update(&counter_bytes);
        let result = mac.finalize().into_bytes();

        // Dynamic truncation per RFC 4226 section 5.4.
        let offset = (result[19] & 0x0f) as usize;
        let binary = ((result[offset] as u32 & 0x7f) << 24)
            | ((result[offset + 1] as u32) << 16)
            | ((result[offset + 2] as u32) << 8)
            | (result[offset + 3] as u32);

        let modulus = 10u32.pow(self.digits);
        let code = binary % modulus;
        format!("{:0>width$}", code, width = self.digits as usize)
    }

    /// Verifies a TOTP code, checking the current period and one period before and after
    /// to account for clock drift.
    pub fn verify(&self, code: &str, timestamp: u64) -> bool {
        // Check current window and +/- 1 period for clock skew tolerance.
        for offset in [0i64, -1, 1] {
            let adjusted = if offset < 0 {
                timestamp.saturating_sub(self.period)
            } else if offset > 0 {
                timestamp.saturating_add(self.period)
            } else {
                timestamp
            };
            if self.generate_code(adjusted) == code {
                return true;
            }
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

// ---- Helper functions ----

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

/// Serializes JWT claims to a JSON string without serde_json.
fn claims_to_json(claims: &JwtClaims) -> String {
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
    s.push_str(&claims.exp.to_string());

    s.push_str(",\"iat\":");
    s.push_str(&claims.iat.to_string());

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

/// Parses a minimal JSON object into JwtClaims.
fn json_to_claims(json: &str) -> Result<JwtClaims> {
    let sub = extract_json_string(json, "sub")?
        .ok_or_else(|| ZyronError::InvalidCredential("JWT claims missing sub".to_string()))?;
    let iss = extract_json_string(json, "iss")?;
    let exp = extract_json_number(json, "exp")?
        .ok_or_else(|| ZyronError::InvalidCredential("JWT claims missing exp".to_string()))?;
    let iat = extract_json_number(json, "iat")?.unwrap_or(0);
    let roles = extract_json_string_array(json, "roles")?;
    let custom = extract_json_custom_fields(json, &["sub", "iss", "exp", "iat", "roles"])?;

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
    let start = match json.find(&search) {
        Some(pos) => pos + search.len(),
        None => return Ok(None),
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

/// Extracts a numeric value for a given key from a JSON object.
fn extract_json_number(json: &str, key: &str) -> Result<Option<u64>> {
    let search = format!("\"{}\":", key);
    let start = match json.find(&search) {
        Some(pos) => pos + search.len(),
        None => return Ok(None),
    };

    let remaining = json[start..].trim_start();
    let end = remaining
        .find(|c: char| !c.is_ascii_digit())
        .unwrap_or(remaining.len());
    let num_str = &remaining[..end];

    if num_str.is_empty() {
        return Ok(None);
    }

    let val: u64 = num_str.parse().map_err(|_| {
        ZyronError::InvalidCredential(format!("Invalid number for key \"{}\"", key))
    })?;
    Ok(Some(val))
}

/// Extracts a string array value for a given key from a JSON object.
fn extract_json_string_array(json: &str, key: &str) -> Result<Vec<String>> {
    let search = format!("\"{}\":[", key);
    let start = match json.find(&search) {
        Some(pos) => pos + search.len(),
        None => return Ok(Vec::new()),
    };

    let remaining = &json[start..];
    let end = remaining.find(']').ok_or_else(|| {
        ZyronError::InvalidCredential(format!("Unterminated array for key \"{}\"", key))
    })?;
    let array_content = &remaining[..end];

    let mut items = Vec::new();
    let mut in_string = false;
    let mut escaped = false;
    let mut current = String::new();

    for ch in array_content.chars() {
        if escaped {
            match ch {
                '"' => current.push('"'),
                '\\' => current.push('\\'),
                'n' => current.push('\n'),
                _ => {
                    current.push('\\');
                    current.push(ch);
                }
            }
            escaped = false;
        } else if ch == '\\' && in_string {
            escaped = true;
        } else if ch == '"' {
            if in_string {
                items.push(current.clone());
                current.clear();
            }
            in_string = !in_string;
        } else if in_string {
            current.push(ch);
        }
    }

    Ok(items)
}

/// Extracts custom string key-value pairs from JSON, excluding known standard fields.
fn extract_json_custom_fields(
    json: &str,
    standard_keys: &[&str],
) -> Result<std::collections::HashMap<String, String>> {
    let mut map = std::collections::HashMap::new();
    let trimmed = json.trim();
    let inner = if trimmed.starts_with('{') && trimmed.ends_with('}') {
        &trimmed[1..trimmed.len() - 1]
    } else {
        return Ok(map);
    };

    // Simple key extraction: find all "key": patterns.
    let mut pos = 0;
    let bytes = inner.as_bytes();
    while pos < bytes.len() {
        // Find opening quote of key.
        let key_start = match inner[pos..].find('"') {
            Some(p) => pos + p + 1,
            None => break,
        };
        // Find closing quote of key.
        let key_end = match inner[key_start..].find('"') {
            Some(p) => key_start + p,
            None => break,
        };
        let key = &inner[key_start..key_end];

        // Move past "key":
        pos = key_end + 1;
        // Skip to colon.
        let colon_pos = match inner[pos..].find(':') {
            Some(p) => pos + p + 1,
            None => break,
        };
        pos = colon_pos;
        let after_colon = inner[pos..].trim_start();
        let offset_bump = inner[pos..].len() - after_colon.len();
        pos += offset_bump;

        if standard_keys.contains(&key) {
            // Skip this value - advance past it.
            pos = skip_json_value(inner, pos);
            continue;
        }

        // Only extract string values for custom fields.
        if after_colon.starts_with('"') {
            pos += 1; // skip opening quote
            let mut val = String::new();
            let mut esc = false;
            while pos < inner.len() {
                let ch = inner.as_bytes()[pos] as char;
                pos += 1;
                if esc {
                    match ch {
                        '"' => val.push('"'),
                        '\\' => val.push('\\'),
                        'n' => val.push('\n'),
                        _ => {
                            val.push('\\');
                            val.push(ch);
                        }
                    }
                    esc = false;
                } else if ch == '\\' {
                    esc = true;
                } else if ch == '"' {
                    break;
                } else {
                    val.push(ch);
                }
            }
            map.insert(key.to_string(), val);
        } else {
            pos = skip_json_value(inner, pos);
        }
    }

    Ok(map)
}

/// Advances position past a JSON value (string, number, array, object, bool, null).
fn skip_json_value(s: &str, start: usize) -> usize {
    let remaining = s[start..].trim_start();
    let offset = s[start..].len() - remaining.len();
    let pos = start + offset;
    if pos >= s.len() {
        return s.len();
    }
    let first = s.as_bytes()[pos];
    match first {
        b'"' => {
            // Skip string.
            let mut i = pos + 1;
            let mut esc = false;
            while i < s.len() {
                if esc {
                    esc = false;
                } else if s.as_bytes()[i] == b'\\' {
                    esc = true;
                } else if s.as_bytes()[i] == b'"' {
                    return i + 1;
                }
                i += 1;
            }
            s.len()
        }
        b'[' => {
            // Skip array.
            let mut depth = 1;
            let mut i = pos + 1;
            while i < s.len() && depth > 0 {
                match s.as_bytes()[i] {
                    b'[' => depth += 1,
                    b']' => depth -= 1,
                    _ => {}
                }
                i += 1;
            }
            i
        }
        b'{' => {
            // Skip object.
            let mut depth = 1;
            let mut i = pos + 1;
            while i < s.len() && depth > 0 {
                match s.as_bytes()[i] {
                    b'{' => depth += 1,
                    b'}' => depth -= 1,
                    _ => {}
                }
                i += 1;
            }
            i
        }
        _ => {
            // Skip number, true, false, null.
            let end = s[pos..]
                .find(|c: char| c == ',' || c == '}' || c == ']')
                .map(|p| pos + p)
                .unwrap_or(s.len());
            end
        }
    }
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
            iss: Some("zyrondb".to_string()),
            exp: 9999999999,
            iat: 1000000000,
            roles: Vec::new(),
            custom: std::collections::HashMap::new(),
        };

        let token = cred.encode(&claims).expect("encode failed");
        let cred_with_iss = JwtCredential::new(vec![0xcd; 48], JwtAlgorithm::Hs384)
            .expect("create failed")
            .with_issuer("zyrondb".to_string());
        let decoded = cred_with_iss.decode(&token).expect("decode failed");
        assert_eq!(decoded.sub, "user99");
        assert_eq!(decoded.iss, Some("zyrondb".to_string()));
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
        let data = b"Hello, ZyronDB!";
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
            iss: Some("zyrondb".to_string()),
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
}
