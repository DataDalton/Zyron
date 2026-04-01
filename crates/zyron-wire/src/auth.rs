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
            Some(expected) if constant_time_eq(expected.as_bytes(), password.as_bytes()) => {
                Ok(AuthProgress::Authenticated)
            }
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
        use rand::Rng;
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

                let client_first = std::str::from_utf8(&data)
                    .map_err(|_| AuthError::SaslError("Invalid UTF-8 in SASL initial".into()))?
                    .to_string();

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

                let client_final = std::str::from_utf8(&data)
                    .map_err(|_| AuthError::SaslError("Invalid UTF-8 in SASL response".into()))?
                    .to_string();

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
    use rand::Rng;
    let mut bytes = [0u8; 18];
    rand::rng().fill_bytes(&mut bytes);
    base64::Engine::encode(&base64::engine::general_purpose::STANDARD, &bytes)
}

/// Generates a random 16-byte salt.
fn generate_salt() -> Vec<u8> {
    use rand::Rng;
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

// ---------------------------------------------------------------------------
// WebAuthn (FIDO2) authenticator
// ---------------------------------------------------------------------------

/// WebAuthn authentication state machine.
/// Uses the existing SASL message flow (SaslMechanisms, SaslInitial, SaslContinue,
/// SaslResponse, SaslFinal) with mechanism name "WEBAUTHN".
///
/// Flow:
/// 1. Server sends SaslMechanisms(["WEBAUTHN"])
/// 2. Client sends SaslInitial { mechanism: "WEBAUTHN", data: empty }
/// 3. Server generates challenge, sends SaslContinue with JSON options
/// 4. Client interacts with hardware key, sends SaslResponse with JSON assertion
/// 5. Server verifies signature, sends SaslFinal
pub struct WebAuthnAuthenticator {
    state: WebAuthnState,
    security_manager: std::sync::Arc<zyron_auth::SecurityManager>,
    rp_config: std::sync::Arc<zyron_auth::RelyingPartyConfig>,
    /// Resolves username to UserId for credential lookup.
    user_lookup: std::sync::Arc<dyn Fn(&str) -> Option<zyron_auth::role::UserId> + Send + Sync>,
}

enum WebAuthnState {
    WaitingForInitial,
    WaitingForAssertion {
        challenge: [u8; 32],
        issued_at: u64,
        allowed_credential_ids: Vec<Vec<u8>>,
        user_id: zyron_auth::role::UserId,
    },
}

impl WebAuthnAuthenticator {
    pub fn new(
        security_manager: std::sync::Arc<zyron_auth::SecurityManager>,
        rp_config: std::sync::Arc<zyron_auth::RelyingPartyConfig>,
        user_lookup: std::sync::Arc<dyn Fn(&str) -> Option<zyron_auth::role::UserId> + Send + Sync>,
    ) -> Self {
        Self {
            state: WebAuthnState::WaitingForInitial,
            security_manager,
            rp_config,
            user_lookup,
        }
    }
}

impl Authenticator for WebAuthnAuthenticator {
    fn initial_message(&self, _user: &str) -> AuthResult {
        AuthResult::Challenge(BackendMessage::Authentication(
            AuthenticationMessage::SaslMechanisms(vec!["WEBAUTHN".into()]),
        ))
    }

    fn process_response(
        &mut self,
        user: &str,
        response: &PasswordMessage,
    ) -> Result<AuthProgress, AuthError> {
        match &self.state {
            WebAuthnState::WaitingForInitial => {
                let _data = match response {
                    PasswordMessage::SaslInitial { mechanism, data } => {
                        if mechanism != "WEBAUTHN" {
                            return Err(AuthError::SaslError(format!(
                                "Expected mechanism WEBAUTHN, got {}",
                                mechanism
                            )));
                        }
                        data
                    }
                    _ => return Err(AuthError::UnexpectedMessage),
                };

                // Look up user and their credentials
                let user_id =
                    (self.user_lookup)(user).ok_or_else(|| AuthError::Failed(user.to_string()))?;

                let credentials = self
                    .security_manager
                    .webauthn_store
                    .credentials_for_user(user_id);
                if credentials.is_empty() {
                    return Err(AuthError::Failed(format!(
                        "No WebAuthn credentials registered for user \"{}\"",
                        user
                    )));
                }

                // Generate challenge
                let challenge = zyron_auth::webauthn::generate_challenge();
                let challenge_b64 = zyron_auth::webauthn::base64url_encode(&challenge);

                let issued_at = std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap_or_default()
                    .as_secs();

                // Build allowCredentials list
                let allowed_credential_ids: Vec<Vec<u8>> = credentials
                    .iter()
                    .map(|c| c.credential_id.clone())
                    .collect();

                let allow_creds_json: Vec<String> = credentials
                    .iter()
                    .map(|c| {
                        let id_b64 = zyron_auth::webauthn::base64url_encode(&c.credential_id);
                        format!("{{\"type\":\"public-key\",\"id\":\"{}\"}}", id_b64)
                    })
                    .collect();

                let options_json = format!(
                    "{{\"challenge\":\"{}\",\"rpId\":\"{}\",\"allowCredentials\":[{}],\"timeout\":{}}}",
                    challenge_b64,
                    self.rp_config.rp_id,
                    allow_creds_json.join(","),
                    self.rp_config.challenge_timeout_secs * 1000,
                );

                self.state = WebAuthnState::WaitingForAssertion {
                    challenge,
                    issued_at,
                    allowed_credential_ids,
                    user_id,
                };

                Ok(AuthProgress::Continue(BackendMessage::Authentication(
                    AuthenticationMessage::SaslContinue(options_json.into_bytes()),
                )))
            }

            WebAuthnState::WaitingForAssertion {
                challenge,
                issued_at,
                allowed_credential_ids,
                user_id,
            } => {
                let data = match response {
                    PasswordMessage::SaslResponse(data) => data,
                    _ => return Err(AuthError::UnexpectedMessage),
                };

                // Check challenge timeout
                let now = std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap_or_default()
                    .as_secs();
                if now - issued_at > self.rp_config.challenge_timeout_secs {
                    return Err(AuthError::SaslError("WebAuthn challenge expired".into()));
                }

                // Parse the assertion JSON from client
                let json_str = std::str::from_utf8(data)
                    .map_err(|_| AuthError::SaslError("Invalid UTF-8 in assertion".into()))?;

                let authenticator_data_b64 = extract_json_field(json_str, "authenticatorData")
                    .ok_or_else(|| AuthError::SaslError("Missing authenticatorData".into()))?;
                let client_data_json_b64 = extract_json_field(json_str, "clientDataJSON")
                    .ok_or_else(|| AuthError::SaslError("Missing clientDataJSON".into()))?;
                let signature_b64 = extract_json_field(json_str, "signature")
                    .ok_or_else(|| AuthError::SaslError("Missing signature".into()))?;
                let credential_id_b64 = extract_json_field(json_str, "credentialId")
                    .ok_or_else(|| AuthError::SaslError("Missing credentialId".into()))?;

                // Decode base64url fields
                let authenticator_data = zyron_auth::webauthn::base64url_decode(
                    &authenticator_data_b64,
                )
                .map_err(|e| AuthError::SaslError(format!("Invalid authenticatorData: {}", e)))?;
                let client_data_json =
                    zyron_auth::webauthn::base64url_decode(&client_data_json_b64).map_err(|e| {
                        AuthError::SaslError(format!("Invalid clientDataJSON: {}", e))
                    })?;
                let signature = zyron_auth::webauthn::base64url_decode(&signature_b64)
                    .map_err(|e| AuthError::SaslError(format!("Invalid signature: {}", e)))?;
                let credential_id = zyron_auth::webauthn::base64url_decode(&credential_id_b64)
                    .map_err(|e| AuthError::SaslError(format!("Invalid credentialId: {}", e)))?;

                // Verify credential ID is in the allowed list
                if !allowed_credential_ids.iter().any(|id| id == &credential_id) {
                    return Err(AuthError::SaslError(
                        "Credential ID not in allowed list".into(),
                    ));
                }

                // Find the stored credential
                let credential = self
                    .security_manager
                    .webauthn_store
                    .find_credential(&credential_id)
                    .ok_or_else(|| AuthError::SaslError("Credential not found in store".into()))?;

                // Verify credential belongs to this user
                if credential.user_id != *user_id {
                    return Err(AuthError::Failed(user.to_string()));
                }

                // Verify the assertion using the webauthn module
                let new_sign_count = zyron_auth::webauthn::verify_assertion(
                    &credential,
                    &authenticator_data,
                    &client_data_json,
                    &signature,
                    &self.rp_config,
                    challenge,
                )
                .map_err(|e| {
                    AuthError::SaslError(format!("WebAuthn verification failed: {}", e))
                })?;

                // Update sign count in store
                self.security_manager
                    .webauthn_store
                    .update_sign_count(&credential_id, new_sign_count);

                // Send SaslFinal (empty success payload) then return Authenticated
                // The connection loop expects Continue with SaslFinal, then Authenticated
                // on the next iteration. Since SaslFinal is the terminal SASL message,
                // we return Authenticated directly here. The caller sends AuthenticationOk.
                Ok(AuthProgress::Authenticated)
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Composed authenticator for MFA (password + WebAuthn)
// ---------------------------------------------------------------------------

/// Chains two authentication methods: password first, then WebAuthn.
/// Used for PasswordAndFido2 auth method.
pub struct ComposedAuthenticator {
    password_auth: Box<dyn Authenticator>,
    webauthn_auth: WebAuthnAuthenticator,
    phase: ComposedPhase,
}

enum ComposedPhase {
    Password,
    WebAuthn,
}

impl ComposedAuthenticator {
    pub fn new(
        password_auth: Box<dyn Authenticator>,
        webauthn_auth: WebAuthnAuthenticator,
    ) -> Self {
        Self {
            password_auth,
            webauthn_auth,
            phase: ComposedPhase::Password,
        }
    }
}

impl Authenticator for ComposedAuthenticator {
    fn initial_message(&self, user: &str) -> AuthResult {
        // Start with password authentication
        self.password_auth.initial_message(user)
    }

    fn process_response(
        &mut self,
        user: &str,
        response: &PasswordMessage,
    ) -> Result<AuthProgress, AuthError> {
        match self.phase {
            ComposedPhase::Password => {
                match self.password_auth.process_response(user, response)? {
                    AuthProgress::Authenticated => {
                        // Password phase complete, transition to WebAuthn
                        self.phase = ComposedPhase::WebAuthn;
                        // Send WebAuthn SASL mechanisms as the next challenge
                        match self.webauthn_auth.initial_message(user) {
                            AuthResult::Challenge(msg) => Ok(AuthProgress::Continue(msg)),
                            AuthResult::Authenticated => Ok(AuthProgress::Authenticated),
                        }
                    }
                    AuthProgress::Continue(msg) => Ok(AuthProgress::Continue(msg)),
                }
            }
            ComposedPhase::WebAuthn => self.webauthn_auth.process_response(user, response),
        }
    }
}

// ---------------------------------------------------------------------------
// TOTP authenticator
// ---------------------------------------------------------------------------

/// TOTP authentication: requests a 6-digit one-time code from the client
/// and verifies it against the user's stored TOTP secret via SecurityManager.
pub struct TotpAuthenticator {
    security_manager: std::sync::Arc<zyron_auth::SecurityManager>,
}

impl TotpAuthenticator {
    pub fn new(security_manager: std::sync::Arc<zyron_auth::SecurityManager>) -> Self {
        Self { security_manager }
    }
}

impl Authenticator for TotpAuthenticator {
    fn initial_message(&self, _user: &str) -> AuthResult {
        // Request the TOTP code via cleartext password message.
        // The client sends the 6-digit code as the "password".
        AuthResult::Challenge(BackendMessage::Authentication(
            AuthenticationMessage::CleartextPassword,
        ))
    }

    fn process_response(
        &mut self,
        user: &str,
        response: &PasswordMessage,
    ) -> Result<AuthProgress, AuthError> {
        let code = match response {
            PasswordMessage::Cleartext(pw) => pw,
            _ => return Err(AuthError::UnexpectedMessage),
        };

        let totp_secret = self
            .security_manager
            .totp_secret_cache
            .get(&user.to_string())
            .ok_or_else(|| {
                AuthError::Failed(format!("No TOTP secret configured for user \"{}\"", user))
            })?;

        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();

        let totp = zyron_auth::TotpCredential::from_secret(totp_secret);
        if totp.verify(code, now) {
            Ok(AuthProgress::Authenticated)
        } else {
            Err(AuthError::Failed(user.to_string()))
        }
    }
}

// ---------------------------------------------------------------------------
// Password + TOTP composed authenticator
// ---------------------------------------------------------------------------

/// Chains password authentication (SCRAM-SHA-256) followed by TOTP verification.
/// Password phase completes first, then the server requests the TOTP code.
pub struct PasswordTotpAuthenticator {
    password_auth: Box<dyn Authenticator>,
    totp_auth: TotpAuthenticator,
    phase: PasswordTotpPhase,
}

enum PasswordTotpPhase {
    Password,
    Totp,
}

impl PasswordTotpAuthenticator {
    pub fn new(password_auth: Box<dyn Authenticator>, totp_auth: TotpAuthenticator) -> Self {
        Self {
            password_auth,
            totp_auth,
            phase: PasswordTotpPhase::Password,
        }
    }
}

impl Authenticator for PasswordTotpAuthenticator {
    fn initial_message(&self, user: &str) -> AuthResult {
        self.password_auth.initial_message(user)
    }

    fn process_response(
        &mut self,
        user: &str,
        response: &PasswordMessage,
    ) -> Result<AuthProgress, AuthError> {
        match self.phase {
            PasswordTotpPhase::Password => {
                match self.password_auth.process_response(user, response)? {
                    AuthProgress::Authenticated => {
                        // Password phase complete, transition to TOTP
                        self.phase = PasswordTotpPhase::Totp;
                        match self.totp_auth.initial_message(user) {
                            AuthResult::Challenge(msg) => Ok(AuthProgress::Continue(msg)),
                            AuthResult::Authenticated => Ok(AuthProgress::Authenticated),
                        }
                    }
                    AuthProgress::Continue(msg) => Ok(AuthProgress::Continue(msg)),
                }
            }
            PasswordTotpPhase::Totp => self.totp_auth.process_response(user, response),
        }
    }
}

// ---------------------------------------------------------------------------
// API Key authenticator
// ---------------------------------------------------------------------------

/// API Key authentication: the client sends the full API key as the "password".
/// The server hashes it and compares against the stored SHA-256 hash.
pub struct ApiKeyAuthenticator {
    security_manager: std::sync::Arc<zyron_auth::SecurityManager>,
}

impl ApiKeyAuthenticator {
    pub fn new(security_manager: std::sync::Arc<zyron_auth::SecurityManager>) -> Self {
        Self { security_manager }
    }
}

impl Authenticator for ApiKeyAuthenticator {
    fn initial_message(&self, _user: &str) -> AuthResult {
        // Request the API key via cleartext password message.
        AuthResult::Challenge(BackendMessage::Authentication(
            AuthenticationMessage::CleartextPassword,
        ))
    }

    fn process_response(
        &mut self,
        user: &str,
        response: &PasswordMessage,
    ) -> Result<AuthProgress, AuthError> {
        let presented_key = match response {
            PasswordMessage::Cleartext(pw) => pw,
            _ => return Err(AuthError::UnexpectedMessage),
        };

        let (prefix, hash) = self
            .security_manager
            .api_key_cache
            .get(&user.to_string())
            .ok_or_else(|| {
                AuthError::Failed(format!("No API key configured for user \"{}\"", user))
            })?;

        // Reconstruct the credential and verify. The hash is stored as Vec<u8>
        // but ApiKeyCredential::from_stored expects [u8; 32].
        if hash.len() != 32 {
            return Err(AuthError::Failed(user.to_string()));
        }
        let mut key_hash = [0u8; 32];
        key_hash.copy_from_slice(&hash);

        let credential = zyron_auth::ApiKeyCredential::from_stored(prefix, key_hash);
        if credential.verify(presented_key) {
            Ok(AuthProgress::Authenticated)
        } else {
            Err(AuthError::Failed(user.to_string()))
        }
    }
}

// ---------------------------------------------------------------------------
// JWT authenticator
// ---------------------------------------------------------------------------

/// JWT authentication: the client sends a signed JWT token as the "password".
/// The server verifies the signature, checks expiration, and validates that
/// the token subject matches the connecting username.
pub struct JwtAuthenticator {
    security_manager: std::sync::Arc<zyron_auth::SecurityManager>,
}

impl JwtAuthenticator {
    pub fn new(security_manager: std::sync::Arc<zyron_auth::SecurityManager>) -> Self {
        Self { security_manager }
    }
}

impl Authenticator for JwtAuthenticator {
    fn initial_message(&self, _user: &str) -> AuthResult {
        // Request the JWT token via cleartext password message.
        AuthResult::Challenge(BackendMessage::Authentication(
            AuthenticationMessage::CleartextPassword,
        ))
    }

    fn process_response(
        &mut self,
        user: &str,
        response: &PasswordMessage,
    ) -> Result<AuthProgress, AuthError> {
        let token = match response {
            PasswordMessage::Cleartext(pw) => pw,
            _ => return Err(AuthError::UnexpectedMessage),
        };

        let secret = self.security_manager.jwt_secret.as_ref().ok_or_else(|| {
            AuthError::Failed("JWT authentication not configured: no jwt_secret set".to_string())
        })?;

        let credential =
            zyron_auth::JwtCredential::new(secret.clone(), self.security_manager.jwt_algorithm)
                .map_err(|e| AuthError::Failed(format!("JWT configuration error: {}", e)))?;

        // Apply issuer validation if configured.
        let credential = match &self.security_manager.jwt_issuer {
            Some(issuer) => credential.with_issuer(issuer.clone()),
            None => credential,
        };

        let claims = credential
            .decode(token)
            .map_err(|e| AuthError::Failed(format!("JWT verification failed: {}", e)))?;

        // Validate that the token subject matches the connecting username.
        if claims.sub != user {
            return Err(AuthError::Failed(format!(
                "JWT subject \"{}\" does not match connecting user \"{}\"",
                claims.sub, user
            )));
        }

        Ok(AuthProgress::Authenticated)
    }
}

/// Extracts a string value from a JSON object for WebAuthn assertion parsing.
fn extract_json_field(json: &str, key: &str) -> Option<String> {
    let search = format!("\"{}\"", key);
    // Find the key as a top-level JSON key (preceded by '{' or ',')
    // to avoid matching key names inside string values.
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
    let after_colon = after_key
        .trim_start()
        .strip_prefix(':')?
        .trim_start()
        .strip_prefix('"')?;
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
