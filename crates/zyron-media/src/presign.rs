//! Presigned media handles
//!
//! A handle is the resource path plus query parameters for expiry, method
//! and an hmac sha256 signature over resource, method and expiry. Verify
//! recomputes the mac with a constant time comparison before trusting the
//! expiry field

use hmac::{Hmac, Mac};
use sha2::Sha256;

use crate::error::{MediaError, MediaResult};

type HmacSha256 = Hmac<Sha256>;

/// A verified handle with its parsed fields
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VerifiedHandle {
    pub resource: String,
    pub method: String,
    pub expires_at_unix: i64,
}

/// Signs a resource and method until the expiry timestamp
pub fn sign(
    secret: &[u8],
    resource: &str,
    method: &str,
    expires_at_unix: i64,
) -> MediaResult<String> {
    if resource.contains('?') || resource.contains('&') {
        return Err(MediaError::InvalidArgument(
            "resource must not contain query delimiters".to_string(),
        ));
    }
    if method.is_empty() || !method.chars().all(|c| c.is_ascii_alphanumeric()) {
        return Err(MediaError::InvalidArgument(format!(
            "invalid handle method {method}"
        )));
    }
    let mac_hex = compute_mac(secret, resource, method, expires_at_unix)?;
    Ok(format!(
        "{resource}?zx={expires_at_unix}&zm={method}&zs={mac_hex}"
    ))
}

/// Verifies a signed handle against the secret and the current time
pub fn verify(secret: &[u8], signed: &str, now_unix: i64) -> MediaResult<VerifiedHandle> {
    let (resource, query) = signed
        .split_once('?')
        .ok_or_else(|| MediaError::InvalidHandle("handle has no query section".to_string()))?;

    let mut expiry: Option<i64> = None;
    let mut method: Option<&str> = None;
    let mut signature: Option<&str> = None;
    for pair in query.split('&') {
        let (key, value) = pair.split_once('=').ok_or_else(|| {
            MediaError::InvalidHandle(format!("malformed query parameter {pair}"))
        })?;
        match key {
            "zx" => {
                expiry = Some(value.parse::<i64>().map_err(|_| {
                    MediaError::InvalidHandle(format!("expiry {value} is not an integer"))
                })?);
            }
            "zm" => method = Some(value),
            "zs" => signature = Some(value),
            other => {
                return Err(MediaError::InvalidHandle(format!(
                    "unexpected query parameter {other}"
                )));
            }
        }
    }
    let expires_at_unix =
        expiry.ok_or_else(|| MediaError::InvalidHandle("handle is missing zx".to_string()))?;
    let method =
        method.ok_or_else(|| MediaError::InvalidHandle("handle is missing zm".to_string()))?;
    let signature =
        signature.ok_or_else(|| MediaError::InvalidHandle("handle is missing zs".to_string()))?;

    let signature_bytes = hex::decode(signature)
        .map_err(|_| MediaError::InvalidHandle("signature is not hex".to_string()))?;

    let mut mac = HmacSha256::new_from_slice(secret)
        .map_err(|_| MediaError::InvalidHandle("secret key is unusable".to_string()))?;
    mac.update(mac_payload(resource, method, expires_at_unix).as_bytes());
    mac.verify_slice(&signature_bytes)
        .map_err(|_| MediaError::InvalidHandle("signature does not match".to_string()))?;

    if now_unix > expires_at_unix {
        return Err(MediaError::HandleExpired {
            expires_at_unix,
            now_unix,
        });
    }

    Ok(VerifiedHandle {
        resource: resource.to_string(),
        method: method.to_string(),
        expires_at_unix,
    })
}

fn mac_payload(resource: &str, method: &str, expires_at_unix: i64) -> String {
    format!("{resource}|{method}|{expires_at_unix}")
}

fn compute_mac(
    secret: &[u8],
    resource: &str,
    method: &str,
    expires_at_unix: i64,
) -> MediaResult<String> {
    let mut mac = HmacSha256::new_from_slice(secret)
        .map_err(|_| MediaError::InvalidHandle("secret key is unusable".to_string()))?;
    mac.update(mac_payload(resource, method, expires_at_unix).as_bytes());
    Ok(hex::encode(mac.finalize().into_bytes()))
}

#[cfg(test)]
mod tests {
    use super::*;

    const SECRET: &[u8] = b"a signing secret for tests";

    #[test]
    fn sign_verify_round_trip() {
        let signed = sign(SECRET, "/media/abc123", "GET", 2_000_000_000).expect("sign");
        let handle = verify(SECRET, &signed, 1_900_000_000).expect("verify");
        assert_eq!(handle.resource, "/media/abc123");
        assert_eq!(handle.method, "GET");
        assert_eq!(handle.expires_at_unix, 2_000_000_000);
    }

    #[test]
    fn expired_handle_rejected() {
        let signed = sign(SECRET, "/media/abc123", "GET", 1_000).expect("sign");
        match verify(SECRET, &signed, 2_000) {
            Err(MediaError::HandleExpired { .. }) => {}
            other => panic!("expected expiry rejection, got {other:?}"),
        }
    }

    #[test]
    fn tampered_handle_rejected() {
        let signed = sign(SECRET, "/media/abc123", "GET", 2_000_000_000).expect("sign");
        let tampered = signed.replace("/media/abc123", "/media/zzz999");
        assert!(verify(SECRET, &tampered, 1_000).is_err());
        let method_swap = signed.replace("zm=GET", "zm=PUT");
        assert!(verify(SECRET, &method_swap, 1_000).is_err());
        let expiry_swap = signed.replace("zx=2000000000", "zx=3000000000");
        assert!(verify(SECRET, &expiry_swap, 1_000).is_err());
        let wrong_secret = verify(b"another secret", &signed, 1_000);
        assert!(wrong_secret.is_err());
    }
}
