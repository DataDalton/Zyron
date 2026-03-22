//! Webhook signature verification helpers.
//!
//! SQL-level functions for verifying webhook payload authenticity from
//! external services (Stripe, GitHub, Slack). Uses HMAC-SHA256 with
//! constant-time comparison to prevent timing attacks.

use hmac::{Hmac, Mac};
use sha2::Sha256;
use zyron_common::{Result, ZyronError};

type HmacSha256 = Hmac<Sha256>;

/// Supported HMAC algorithms for generic webhook verification.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HmacAlgorithm {
    Sha256,
}

/// Computes HMAC and compares against the expected signature using constant-time comparison.
/// Returns true if the signatures match.
pub fn verify_hmac(
    payload: &[u8],
    expected_hex: &str,
    secret: &[u8],
    algorithm: HmacAlgorithm,
) -> Result<bool> {
    let expected_bytes = hex_decode(expected_hex).map_err(|e| {
        ZyronError::WebhookVerificationFailed(format!("Invalid hex signature: {}", e))
    })?;

    match algorithm {
        HmacAlgorithm::Sha256 => {
            let mut mac = HmacSha256::new_from_slice(secret).map_err(|e| {
                ZyronError::WebhookVerificationFailed(format!("HMAC init failed: {}", e))
            })?;
            mac.update(payload);
            match mac.verify_slice(&expected_bytes) {
                Ok(()) => Ok(true),
                Err(_) => Ok(false),
            }
        }
    }
}

/// Maximum age for webhook timestamps before rejection (5 minutes).
/// Prevents replay attacks with captured payloads.
const WEBHOOK_MAX_AGE_SECS: u64 = 300;

/// Verifies a Stripe webhook signature with timestamp freshness check.
/// Stripe format: sig_header = "t=timestamp,v1=hex_signature"
/// Signed message = "timestamp.payload"
/// Rejects payloads older than 5 minutes to prevent replay attacks.
pub fn verify_stripe_webhook(payload: &[u8], sig_header: &str, secret: &str) -> Result<bool> {
    let mut timestamp = None;
    let mut signature = None;

    for part in sig_header.split(',') {
        let part = part.trim();
        if let Some(t) = part.strip_prefix("t=") {
            timestamp = Some(t);
        } else if let Some(v) = part.strip_prefix("v1=") {
            signature = Some(v);
        }
    }

    let timestamp = timestamp.ok_or_else(|| {
        ZyronError::WebhookVerificationFailed(
            "Missing timestamp in Stripe signature header".to_string(),
        )
    })?;
    let signature = signature.ok_or_else(|| {
        ZyronError::WebhookVerificationFailed(
            "Missing v1 signature in Stripe signature header".to_string(),
        )
    })?;

    // Verify timestamp freshness to prevent replay attacks
    check_timestamp_freshness(timestamp)?;

    // Signed payload = "timestamp.body"
    let payload_str = std::str::from_utf8(payload).map_err(|_| {
        ZyronError::WebhookVerificationFailed("Payload is not valid UTF-8".to_string())
    })?;
    let signed_payload = format!("{}.{}", timestamp, payload_str);

    verify_hmac(
        signed_payload.as_bytes(),
        signature,
        secret.as_bytes(),
        HmacAlgorithm::Sha256,
    )
}

/// Verifies a GitHub webhook signature.
/// GitHub format: sig_header = "sha256=hex_signature"
/// HMAC-SHA256 of the raw payload.
pub fn verify_github_webhook(payload: &[u8], sig_header: &str, secret: &str) -> Result<bool> {
    let hex_sig = sig_header.strip_prefix("sha256=").ok_or_else(|| {
        ZyronError::WebhookVerificationFailed(
            "GitHub signature header must start with 'sha256='".to_string(),
        )
    })?;

    verify_hmac(payload, hex_sig, secret.as_bytes(), HmacAlgorithm::Sha256)
}

/// Verifies a Slack webhook signature with timestamp freshness check.
/// Slack format: sig_header = "v0=hex_signature"
/// Signed message = "v0:timestamp:payload"
/// Rejects payloads older than 5 minutes to prevent replay attacks.
pub fn verify_slack_webhook(
    payload: &[u8],
    timestamp: &str,
    sig_header: &str,
    secret: &str,
) -> Result<bool> {
    // Verify timestamp freshness
    check_timestamp_freshness(timestamp)?;

    let hex_sig = sig_header.strip_prefix("v0=").ok_or_else(|| {
        ZyronError::WebhookVerificationFailed(
            "Slack signature header must start with 'v0='".to_string(),
        )
    })?;

    let payload_str = std::str::from_utf8(payload).map_err(|_| {
        ZyronError::WebhookVerificationFailed("Payload is not valid UTF-8".to_string())
    })?;
    let signed_payload = format!("v0:{}:{}", timestamp, payload_str);

    verify_hmac(
        signed_payload.as_bytes(),
        hex_sig,
        secret.as_bytes(),
        HmacAlgorithm::Sha256,
    )
}

/// Validates that a webhook timestamp is within the acceptable freshness window.
/// Rejects timestamps older than WEBHOOK_MAX_AGE_SECS (5 minutes) or timestamps
/// that are too far in the future (clock skew tolerance of 60 seconds).
fn check_timestamp_freshness(timestamp_str: &str) -> Result<()> {
    let ts: u64 = timestamp_str.parse().map_err(|_| {
        ZyronError::WebhookVerificationFailed(format!("Invalid timestamp: {}", timestamp_str))
    })?;

    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();

    if now > ts && now - ts > WEBHOOK_MAX_AGE_SECS {
        return Err(ZyronError::WebhookVerificationFailed(format!(
            "Webhook timestamp too old: {} seconds ago (max {})",
            now - ts,
            WEBHOOK_MAX_AGE_SECS
        )));
    }

    // Allow up to 60 seconds of clock skew into the future
    if ts > now && ts - now > 60 {
        return Err(ZyronError::WebhookVerificationFailed(
            "Webhook timestamp is too far in the future".to_string(),
        ));
    }

    Ok(())
}

/// Decodes a hex string to bytes.
fn hex_decode(hex: &str) -> std::result::Result<Vec<u8>, String> {
    if hex.len() % 2 != 0 {
        return Err("Hex string has odd length".to_string());
    }
    let mut bytes = Vec::with_capacity(hex.len() / 2);
    for i in (0..hex.len()).step_by(2) {
        let byte = u8::from_str_radix(&hex[i..i + 2], 16)
            .map_err(|_| format!("Invalid hex at position {}", i))?;
        bytes.push(byte);
    }
    Ok(bytes)
}

/// Hex encoding lookup table.
const HEX_CHARS: &[u8; 16] = b"0123456789abcdef";

/// Encodes bytes as a lowercase hex string using a lookup table.
fn hex_encode(bytes: &[u8]) -> String {
    let mut hex = String::with_capacity(bytes.len() * 2);
    for &b in bytes {
        hex.push(HEX_CHARS[(b >> 4) as usize] as char);
        hex.push(HEX_CHARS[(b & 0x0f) as usize] as char);
    }
    hex
}

/// Computes an HMAC-SHA256 and returns the hex-encoded signature.
/// Useful for generating test signatures.
pub fn compute_hmac_sha256(payload: &[u8], secret: &[u8]) -> Result<String> {
    let mut mac = HmacSha256::new_from_slice(secret)
        .map_err(|e| ZyronError::WebhookVerificationFailed(format!("HMAC init failed: {}", e)))?;
    mac.update(payload);
    let result = mac.finalize();
    Ok(hex_encode(&result.into_bytes()))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn current_timestamp() -> String {
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs()
            .to_string()
    }

    #[test]
    fn test_verify_hmac_valid() {
        let secret = b"test_secret";
        let payload = b"test payload";
        let sig = compute_hmac_sha256(payload, secret).expect("compute");
        assert!(verify_hmac(payload, &sig, secret, HmacAlgorithm::Sha256).expect("verify"));
    }

    #[test]
    fn test_verify_hmac_invalid() {
        let secret = b"test_secret";
        let payload = b"test payload";
        let bad_sig = "00".repeat(32);
        assert!(!verify_hmac(payload, &bad_sig, secret, HmacAlgorithm::Sha256).expect("verify"));
    }

    #[test]
    fn test_verify_hmac_bad_hex() {
        assert!(verify_hmac(b"", "zzzz", b"secret", HmacAlgorithm::Sha256).is_err());
    }

    #[test]
    fn test_verify_stripe_valid() {
        let secret = "whsec_test_secret";
        let payload = b"{\"type\":\"payment_intent.succeeded\"}";
        let timestamp = current_timestamp();
        let signed = format!("{}.{}", timestamp, std::str::from_utf8(payload).unwrap());
        let sig = compute_hmac_sha256(signed.as_bytes(), secret.as_bytes()).expect("compute");
        let header = format!("t={},v1={}", timestamp, sig);
        assert!(verify_stripe_webhook(payload, &header, secret).expect("verify"));
    }

    #[test]
    fn test_verify_stripe_missing_timestamp() {
        assert!(verify_stripe_webhook(b"body", "v1=abc", "secret").is_err());
    }

    #[test]
    fn test_verify_stripe_missing_v1() {
        assert!(verify_stripe_webhook(b"body", "t=123", "secret").is_err());
    }

    #[test]
    fn test_verify_stripe_invalid_sig() {
        let ts = current_timestamp();
        let header = format!("t={},v1={}", ts, "00".repeat(32));
        assert!(!verify_stripe_webhook(b"body", &header, "secret").expect("verify"));
    }

    #[test]
    fn test_verify_stripe_stale_timestamp() {
        let secret = "whsec_test";
        let payload = b"body";
        let old_ts = "1614556828"; // Far in the past
        let signed = format!("{}.{}", old_ts, std::str::from_utf8(payload).unwrap());
        let sig = compute_hmac_sha256(signed.as_bytes(), secret.as_bytes()).expect("compute");
        let header = format!("t={},v1={}", old_ts, sig);
        assert!(verify_stripe_webhook(payload, &header, secret).is_err());
    }

    #[test]
    fn test_verify_github_valid() {
        let secret = "github_secret";
        let payload = b"{\"action\":\"push\"}";
        let sig = compute_hmac_sha256(payload, secret.as_bytes()).expect("compute");
        let header = format!("sha256={}", sig);
        assert!(verify_github_webhook(payload, &header, secret).expect("verify"));
    }

    #[test]
    fn test_verify_github_bad_prefix() {
        assert!(verify_github_webhook(b"body", "sha1=abc", "secret").is_err());
    }

    #[test]
    fn test_verify_github_invalid_sig() {
        let header = format!("sha256={}", "00".repeat(32));
        assert!(!verify_github_webhook(b"body", &header, "secret").expect("verify"));
    }

    #[test]
    fn test_verify_slack_valid() {
        let secret = "slack_secret";
        let payload = b"token=abc&team_id=T123";
        let timestamp = &current_timestamp();
        let signed = format!("v0:{}:{}", timestamp, std::str::from_utf8(payload).unwrap());
        let sig = compute_hmac_sha256(signed.as_bytes(), secret.as_bytes()).expect("compute");
        let header = format!("v0={}", sig);
        assert!(verify_slack_webhook(payload, timestamp, &header, secret).expect("verify"));
    }

    #[test]
    fn test_verify_slack_bad_prefix() {
        assert!(verify_slack_webhook(b"body", "123", "sha256=abc", "secret").is_err());
    }

    #[test]
    fn test_hex_roundtrip() {
        let bytes = vec![0x01, 0xAB, 0xCD, 0xEF, 0x00, 0xFF];
        let hex = hex_encode(&bytes);
        let decoded = hex_decode(&hex).expect("decode");
        assert_eq!(decoded, bytes);
    }
}
