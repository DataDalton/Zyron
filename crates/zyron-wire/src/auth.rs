//! Authentication handlers for the PostgreSQL wire protocol.
//!
//! Provides pluggable authentication via the Authenticator trait with
//! implementations for trust (no password), cleartext, MD5, and SCRAM-SHA-256.

use std::collections::HashMap;

use crate::messages::backend::{AuthenticationMessage, BackendMessage};
use crate::messages::frontend::PasswordMessage;

/// Result of processing an authentication response.
pub enum AuthProgress {
    /// Authentication succeeded.
    Authenticated,
    /// Send this message to the client and wait for another response.
    Continue(BackendMessage),
}

/// Authentication error.
#[derive(Debug, thiserror::Error)]
pub enum AuthError {
    #[error("Authentication failed for user \"{0}\"")]
    Failed(String),
    #[error("Unexpected message during authentication")]
    UnexpectedMessage,
    #[error("SASL protocol error: {0}")]
    SaslError(String),
}

/// Pluggable authentication handler. Each implementation handles one auth flow.
pub trait Authenticator: Send + Sync {
    /// Returns the initial authentication message to send to the client.
    /// For TrustAuthenticator, this returns AuthenticationOk directly.
    fn initial_message(&self, user: &str) -> AuthResult;

    /// Processes a client password/SASL response.
    fn process_response(
        &mut self,
        user: &str,
        response: &PasswordMessage,
    ) -> Result<AuthProgress, AuthError>;
}

/// Result of the initial authentication step.
pub enum AuthResult {
    /// Authentication complete, no password needed.
    Authenticated,
    /// Send this message to the client to request credentials.
    Challenge(BackendMessage),
}

// ---------------------------------------------------------------------------
// Trust authenticator (no password)
// ---------------------------------------------------------------------------

/// Trust authentication: accepts all connections without a password.
/// Suitable for local development and testing.
pub struct TrustAuthenticator;

impl Authenticator for TrustAuthenticator {
    fn initial_message(&self, _user: &str) -> AuthResult {
        AuthResult::Authenticated
    }

    fn process_response(
        &mut self,
        _user: &str,
        _response: &PasswordMessage,
    ) -> Result<AuthProgress, AuthError> {
        Ok(AuthProgress::Authenticated)
    }
}

// ---------------------------------------------------------------------------
// Cleartext password authenticator
// ---------------------------------------------------------------------------

/// Cleartext password authentication. Sends the password in plain text.
/// Only appropriate for testing or when the connection is encrypted.
pub struct CleartextAuthenticator {
    passwords: HashMap<String, String>,
}

impl CleartextAuthenticator {
    pub fn new(passwords: HashMap<String, String>) -> Self {
        Self { passwords }
    }
}

impl Authenticator for CleartextAuthenticator {
    fn initial_message(&self, _user: &str) -> AuthResult {
        AuthResult::Challenge(BackendMessage::Authentication(
            AuthenticationMessage::CleartextPassword,
        ))
    }

    fn process_response(
        &mut self,
        user: &str,
        response: &PasswordMessage,
    ) -> Result<AuthProgress, AuthError> {
        let password = match response {
            PasswordMessage::Cleartext(pw) => pw,
            _ => return Err(AuthError::UnexpectedMessage),
        };

        match self.passwords.get(user) {
            Some(expected) if expected == password => Ok(AuthProgress::Authenticated),
            _ => Err(AuthError::Failed(user.to_string())),
        }
    }
}

// ---------------------------------------------------------------------------
// MD5 password authenticator
// ---------------------------------------------------------------------------

/// MD5 password authentication.
/// Protocol: client sends md5(md5(password + user) + salt) as "md5" + 32 hex chars.
pub struct Md5Authenticator {
    passwords: HashMap<String, String>,
    salt: [u8; 4],
}

impl Md5Authenticator {
    pub fn new(passwords: HashMap<String, String>) -> Self {
        let mut salt = [0u8; 4];
        use rand::RngCore;
        rand::rng().fill_bytes(&mut salt);
        Self { passwords, salt }
    }

    /// Creates an authenticator with a fixed salt (for testing).
    #[cfg(test)]
    pub fn with_salt(passwords: HashMap<String, String>, salt: [u8; 4]) -> Self {
        Self { passwords, salt }
    }
}

impl Authenticator for Md5Authenticator {
    fn initial_message(&self, _user: &str) -> AuthResult {
        AuthResult::Challenge(BackendMessage::Authentication(
            AuthenticationMessage::Md5Password { salt: self.salt },
        ))
    }

    fn process_response(
        &mut self,
        user: &str,
        response: &PasswordMessage,
    ) -> Result<AuthProgress, AuthError> {
        let received = match response {
            PasswordMessage::Cleartext(pw) => pw.as_bytes(),
            PasswordMessage::Md5(data) => data.as_slice(),
            _ => return Err(AuthError::UnexpectedMessage),
        };

        let expected_password = match self.passwords.get(user) {
            Some(pw) => pw,
            None => return Err(AuthError::Failed(user.to_string())),
        };

        let expected = compute_md5_password(user, expected_password, &self.salt);

        if constant_time_eq(received, expected.as_bytes()) {
            Ok(AuthProgress::Authenticated)
        } else {
            Err(AuthError::Failed(user.to_string()))
        }
    }
}

/// Computes the MD5 password hash per PostgreSQL spec:
/// "md5" + md5(md5(password + username) + salt)
fn compute_md5_password(user: &str, password: &str, salt: &[u8; 4]) -> String {
    use md5::{Digest, Md5};

    // Step 1: md5(password + username)
    let mut hasher = Md5::new();
    hasher.update(password.as_bytes());
    hasher.update(user.as_bytes());
    let inner = hasher.finalize();
    let inner_hex = format!("{:x}", inner);

    // Step 2: md5(inner_hex + salt)
    let mut hasher = Md5::new();
    hasher.update(inner_hex.as_bytes());
    hasher.update(salt);
    let outer = hasher.finalize();

    format!("md5{:x}", outer)
}

// ---------------------------------------------------------------------------
// SCRAM-SHA-256 authenticator
// ---------------------------------------------------------------------------

/// SCRAM-SHA-256 authentication (RFC 5802 / 7677).
/// Multi-round protocol: server sends mechanisms, client sends initial response,
/// server sends challenge, client sends proof, server sends final verification.
pub struct ScramAuthenticator {
    /// Stored passwords for lookup.
    passwords: HashMap<String, String>,
    /// Current SCRAM state machine.
    state: ScramState,
}

enum ScramState {
    WaitingForInitial,
    WaitingForProof {
        server_nonce: String,
        _salt: Vec<u8>,
        _iterations: u32,
        client_first_bare: String,
        server_first: String,
        stored_key: [u8; 32],
        server_key: [u8; 32],
    },
}

impl ScramAuthenticator {
    pub fn new(passwords: HashMap<String, String>) -> Self {
        Self {
            passwords,
            state: ScramState::WaitingForInitial,
        }
    }
}

impl Authenticator for ScramAuthenticator {
    fn initial_message(&self, _user: &str) -> AuthResult {
        AuthResult::Challenge(BackendMessage::Authentication(
            AuthenticationMessage::SaslMechanisms(vec!["SCRAM-SHA-256".into()]),
        ))
    }

    fn process_response(
        &mut self,
        user: &str,
        response: &PasswordMessage,
    ) -> Result<AuthProgress, AuthError> {
        match &self.state {
            ScramState::WaitingForInitial => {
                let data = match response {
                    PasswordMessage::SaslInitial { data, .. } => data,
                    _ => return Err(AuthError::UnexpectedMessage),
                };

                let client_first = String::from_utf8(data.clone())
                    .map_err(|_| AuthError::SaslError("Invalid UTF-8 in SASL initial".into()))?;

                // Parse client-first-message: "n,,n=user,r=client-nonce"
                let client_first_bare = client_first
                    .strip_prefix("n,,")
                    .ok_or_else(|| AuthError::SaslError("Invalid client-first format".into()))?
                    .to_string();

                let client_nonce = extract_field(&client_first_bare, "r=")
                    .ok_or_else(|| AuthError::SaslError("Missing client nonce".into()))?;

                let password = self
                    .passwords
                    .get(user)
                    .ok_or_else(|| AuthError::Failed(user.to_string()))?;

                // Generate server nonce and salt
                let server_nonce = format!("{}{}", client_nonce, generate_nonce());
                let salt = generate_salt();
                let iterations = 4096u32;

                // Derive keys from password
                let salted_password = pbkdf2_sha256(password.as_bytes(), &salt, iterations);
                let client_key = hmac_sha256(&salted_password, b"Client Key");
                let stored_key = sha256(&client_key);
                let server_key = hmac_sha256(&salted_password, b"Server Key");

                let salt_b64 =
                    base64::Engine::encode(&base64::engine::general_purpose::STANDARD, &salt);
                let server_first = format!("r={},s={},i={}", server_nonce, salt_b64, iterations);

                self.state = ScramState::WaitingForProof {
                    server_nonce,
                    _salt: salt,
                    _iterations: iterations,
                    client_first_bare,
                    server_first: server_first.clone(),
                    stored_key,
                    server_key,
                };

                Ok(AuthProgress::Continue(BackendMessage::Authentication(
                    AuthenticationMessage::SaslContinue(server_first.into_bytes()),
                )))
            }

            ScramState::WaitingForProof {
                server_nonce,
                client_first_bare,
                server_first,
                stored_key,
                server_key,
                ..
            } => {
                let data = match response {
                    PasswordMessage::SaslResponse(data) => data,
                    _ => return Err(AuthError::UnexpectedMessage),
                };

                let client_final = String::from_utf8(data.clone())
                    .map_err(|_| AuthError::SaslError("Invalid UTF-8 in SASL response".into()))?;

                // Parse client-final-message: "c=biws,r=nonce,p=proof"
                let proof_b64 = extract_field(&client_final, "p=")
                    .ok_or_else(|| AuthError::SaslError("Missing proof".into()))?;

                let received_nonce = extract_field(&client_final, "r=")
                    .ok_or_else(|| AuthError::SaslError("Missing nonce".into()))?;

                if received_nonce != *server_nonce {
                    return Err(AuthError::SaslError("Nonce mismatch".into()));
                }

                // Compute auth message
                let client_final_without_proof = client_final
                    .rsplitn(2, ",p=")
                    .last()
                    .unwrap_or(&client_final);
                let auth_message = format!(
                    "{},{},{}",
                    client_first_bare, server_first, client_final_without_proof
                );

                // Verify client proof
                let client_signature = hmac_sha256(stored_key, auth_message.as_bytes());
                let proof_bytes =
                    base64::Engine::decode(&base64::engine::general_purpose::STANDARD, &proof_b64)
                        .map_err(|_| AuthError::SaslError("Invalid base64 in proof".into()))?;

                // Proof must be exactly 32 bytes (SHA-256 output length).
                if proof_bytes.len() != 32 {
                    return Err(AuthError::SaslError("Invalid proof length".into()));
                }

                // client_key = proof XOR client_signature
                let mut recovered_client_key = [0u8; 32];
                for i in 0..32 {
                    recovered_client_key[i] = proof_bytes[i] ^ client_signature[i];
                }

                let recovered_stored_key = sha256(&recovered_client_key);
                if !constant_time_eq(&recovered_stored_key, stored_key) {
                    return Err(AuthError::Failed(user.to_string()));
                }

                // Compute server signature for verification
                let server_signature = hmac_sha256(server_key, auth_message.as_bytes());
                let server_sig_b64 = base64::Engine::encode(
                    &base64::engine::general_purpose::STANDARD,
                    &server_signature,
                );
                let server_final = format!("v={}", server_sig_b64);

                Ok(AuthProgress::Continue(BackendMessage::Authentication(
                    AuthenticationMessage::SaslFinal(server_final.into_bytes()),
                )))
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Crypto helpers
// ---------------------------------------------------------------------------

/// PBKDF2-SHA-256 key derivation.
fn pbkdf2_sha256(password: &[u8], salt: &[u8], iterations: u32) -> [u8; 32] {
    use hmac::Mac;

    let mut result = [0u8; 32];
    // PBKDF2 with HMAC-SHA256, single block (dk_len <= hash_len)
    // U1 = HMAC(password, salt || INT(1))
    let mut mac =
        hmac::Hmac::<sha2::Sha256>::new_from_slice(password).expect("HMAC key length is valid");
    mac.update(salt);
    mac.update(&1u32.to_be_bytes());
    let u1 = mac.finalize().into_bytes();

    result.copy_from_slice(&u1);
    let mut prev = u1;

    for _ in 1..iterations {
        let mut mac =
            hmac::Hmac::<sha2::Sha256>::new_from_slice(password).expect("HMAC key length is valid");
        mac.update(&prev);
        let ui = mac.finalize().into_bytes();
        for j in 0..32 {
            result[j] ^= ui[j];
        }
        prev = ui;
    }

    result
}

/// HMAC-SHA-256.
fn hmac_sha256(key: &[u8], data: &[u8]) -> [u8; 32] {
    use hmac::Mac;

    let mut mac =
        hmac::Hmac::<sha2::Sha256>::new_from_slice(key).expect("HMAC key length is valid");
    mac.update(data);
    let result = mac.finalize().into_bytes();
    let mut out = [0u8; 32];
    out.copy_from_slice(&result);
    out
}

/// SHA-256 hash.
fn sha256(data: &[u8]) -> [u8; 32] {
    use sha2::Digest;

    let mut hasher = sha2::Sha256::new();
    hasher.update(data);
    let result = hasher.finalize();
    let mut out = [0u8; 32];
    out.copy_from_slice(&result);
    out
}

/// Constant-time byte comparison to prevent timing side-channel attacks.
/// Returns true if both slices have the same length and identical contents.
fn constant_time_eq(a: &[u8], b: &[u8]) -> bool {
    if a.len() != b.len() {
        return false;
    }
    a.iter().zip(b).fold(0u8, |acc, (x, y)| acc | (x ^ y)) == 0
}

/// Generates a random 18-byte nonce encoded as base64.
fn generate_nonce() -> String {
    use rand::RngCore;
    let mut bytes = [0u8; 18];
    rand::rng().fill_bytes(&mut bytes);
    base64::Engine::encode(&base64::engine::general_purpose::STANDARD, &bytes)
}

/// Generates a random 16-byte salt.
fn generate_salt() -> Vec<u8> {
    use rand::RngCore;
    let mut bytes = [0u8; 16];
    rand::rng().fill_bytes(&mut bytes);
    bytes.to_vec()
}

/// Extracts a field value from a SCRAM message (e.g. "r=nonce" -> "nonce").
fn extract_field<'a>(message: &'a str, prefix: &str) -> Option<String> {
    for part in message.split(',') {
        if part.starts_with(prefix) {
            return Some(part[prefix.len()..].to_string());
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_trust_authenticator() {
        let mut auth = TrustAuthenticator;
        match auth.initial_message("anyone") {
            AuthResult::Authenticated => {}
            _ => panic!("Trust should authenticate immediately"),
        }

        let result = auth
            .process_response("anyone", &PasswordMessage::Cleartext("ignored".into()))
            .unwrap();
        assert!(matches!(result, AuthProgress::Authenticated));
    }

    #[test]
    fn test_cleartext_authenticator_success() {
        let mut passwords = HashMap::new();
        passwords.insert("alice".into(), "secret123".into());
        let mut auth = CleartextAuthenticator::new(passwords);

        match auth.initial_message("alice") {
            AuthResult::Challenge(_) => {}
            _ => panic!("Should send challenge"),
        }

        let result = auth
            .process_response("alice", &PasswordMessage::Cleartext("secret123".into()))
            .unwrap();
        assert!(matches!(result, AuthProgress::Authenticated));
    }

    #[test]
    fn test_cleartext_authenticator_wrong_password() {
        let mut passwords = HashMap::new();
        passwords.insert("alice".into(), "secret123".into());
        let mut auth = CleartextAuthenticator::new(passwords);

        let result = auth.process_response("alice", &PasswordMessage::Cleartext("wrong".into()));
        assert!(result.is_err());
    }

    #[test]
    fn test_cleartext_authenticator_unknown_user() {
        let mut auth = CleartextAuthenticator::new(HashMap::new());

        let result = auth.process_response("nobody", &PasswordMessage::Cleartext("any".into()));
        assert!(result.is_err());
    }

    #[test]
    fn test_md5_authenticator() {
        let mut passwords = HashMap::new();
        passwords.insert("alice".into(), "secret123".into());
        let salt = [0x01, 0x02, 0x03, 0x04];
        let mut auth = Md5Authenticator::with_salt(passwords.clone(), salt);

        match auth.initial_message("alice") {
            AuthResult::Challenge(msg) => match msg {
                BackendMessage::Authentication(AuthenticationMessage::Md5Password { salt: s }) => {
                    assert_eq!(s, salt);
                }
                _ => panic!("Expected MD5 challenge"),
            },
            _ => panic!("Should send challenge"),
        }

        // Compute expected hash
        let expected = compute_md5_password("alice", "secret123", &salt);
        let result = auth
            .process_response("alice", &PasswordMessage::Cleartext(expected))
            .unwrap();
        assert!(matches!(result, AuthProgress::Authenticated));
    }

    #[test]
    fn test_md5_wrong_password() {
        let mut passwords = HashMap::new();
        passwords.insert("alice".into(), "secret123".into());
        let salt = [0x01, 0x02, 0x03, 0x04];
        let mut auth = Md5Authenticator::with_salt(passwords, salt);

        let result = auth.process_response(
            "alice",
            &PasswordMessage::Cleartext("md5ffffffffffffffffffffffffffffffff".into()),
        );
        assert!(result.is_err());
    }

    #[test]
    fn test_compute_md5_password() {
        // Verify against known PostgreSQL md5 password behavior
        let hash = compute_md5_password("user", "pass", &[0, 0, 0, 0]);
        assert!(hash.starts_with("md5"));
        assert_eq!(hash.len(), 35); // "md5" + 32 hex chars
    }

    #[test]
    fn test_extract_field() {
        assert_eq!(
            extract_field("n=user,r=abc123", "r="),
            Some("abc123".into())
        );
        assert_eq!(extract_field("n=user,r=abc123", "n="), Some("user".into()));
        assert_eq!(extract_field("n=user,r=abc123", "s="), None);
    }

    #[test]
    fn test_hmac_sha256() {
        let key = b"key";
        let data = b"data";
        let result = hmac_sha256(key, data);
        assert_eq!(result.len(), 32);
        // Verify deterministic
        assert_eq!(result, hmac_sha256(key, data));
    }

    #[test]
    fn test_sha256() {
        let result = sha256(b"hello");
        assert_eq!(result.len(), 32);
        // Known SHA-256 of "hello"
        assert_eq!(result[0], 0x2c);
        assert_eq!(result[1], 0xf2);
    }

    #[test]
    fn test_pbkdf2_sha256_deterministic() {
        let result1 = pbkdf2_sha256(b"password", b"salt", 1);
        let result2 = pbkdf2_sha256(b"password", b"salt", 1);
        assert_eq!(result1, result2);
    }

    #[test]
    fn test_generate_nonce_uniqueness() {
        let n1 = generate_nonce();
        let n2 = generate_nonce();
        assert_ne!(n1, n2);
        assert!(!n1.is_empty());
    }

    #[test]
    fn test_generate_salt_length() {
        let salt = generate_salt();
        assert_eq!(salt.len(), 16);
    }
}
