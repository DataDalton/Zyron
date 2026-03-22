//! SQL-level password hashing and key generation functions.
//!
//! Wraps the existing Balloon KDF for password hashing and provides key
//! generation using the rand crate. These are the core implementations
//! that SQL function wrappers in the executor will delegate to.

use crate::balloon::{
    BalloonParams, balloon_hash_encoded, balloon_hash_encoded_with_params, balloon_verify,
};
use sha2::{Digest, Sha256};
use zyron_common::Result;

/// Hashes a password using Balloon KDF with default parameters.
/// Returns the hash in PHC format:
/// $balloon-aes$v=1$s=N,t=N,d=N$salt$hash
pub fn password_hash(password: &str) -> Result<String> {
    balloon_hash_encoded(password)
}

/// Hashes a password with custom Balloon KDF parameters.
pub fn password_hash_with_params(password: &str, params: &BalloonParams) -> Result<String> {
    balloon_hash_encoded_with_params(password, params)
}

/// Verifies a password against a Balloon KDF hash in PHC format.
pub fn password_verify(password: &str, hash: &str) -> Result<bool> {
    balloon_verify(password, hash)
}

/// Generates a random symmetric key of the given bit length.
/// Supported lengths: 128, 256. Returns the raw key bytes.
pub fn generate_symmetric_key(bits: u16) -> Result<Vec<u8>> {
    let byte_len = match bits {
        128 => 16,
        256 => 32,
        _ => {
            return Err(zyron_common::ZyronError::InvalidParameter {
                name: "bits".to_string(),
                value: bits.to_string(),
            });
        }
    };
    use rand::Rng;
    let mut key = vec![0u8; byte_len];
    rand::rng().fill_bytes(&mut key);
    Ok(key)
}

/// Generates an HMAC-based signing keypair.
/// The "private key" is 32 random bytes. The "public key" is SHA-256(private_key).
/// Sign with HMAC-SHA256(private_key, message), verify by recomputing.
/// Not traditional asymmetric crypto. For application-level signing only.
pub fn generate_signing_keypair() -> Result<(Vec<u8>, Vec<u8>)> {
    use rand::Rng;
    let mut private_key = vec![0u8; 32];
    rand::rng().fill_bytes(&mut private_key);

    let mut hasher = Sha256::new();
    hasher.update(&private_key);
    let public_key = hasher.finalize().to_vec();

    Ok((private_key, public_key))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_password_hash_and_verify() {
        let hash = password_hash("test_password_123").expect("hash");
        assert!(hash.starts_with("$balloon-aes$"));
        assert!(password_verify("test_password_123", &hash).expect("verify"));
        assert!(!password_verify("wrong_password", &hash).expect("verify"));
    }

    #[test]
    fn test_password_hash_with_custom_params() {
        let params = BalloonParams {
            space_cost: 1024,
            time_cost: 1,
            delta: 3,
        };
        let hash = password_hash_with_params("custom", &params).expect("hash");
        assert!(password_verify("custom", &hash).expect("verify"));
    }

    #[test]
    fn test_generate_symmetric_key_128() {
        let key = generate_symmetric_key(128).expect("gen");
        assert_eq!(key.len(), 16);
    }

    #[test]
    fn test_generate_symmetric_key_256() {
        let key = generate_symmetric_key(256).expect("gen");
        assert_eq!(key.len(), 32);
    }

    #[test]
    fn test_generate_symmetric_key_invalid_bits() {
        assert!(generate_symmetric_key(64).is_err());
    }

    #[test]
    fn test_generate_symmetric_key_unique() {
        let k1 = generate_symmetric_key(256).expect("gen");
        let k2 = generate_symmetric_key(256).expect("gen");
        assert_ne!(k1, k2);
    }

    #[test]
    fn test_generate_signing_keypair() {
        let (private_key, public_key) = generate_signing_keypair().expect("gen");
        assert_eq!(private_key.len(), 32);
        assert_eq!(public_key.len(), 32);
    }

    #[test]
    fn test_generate_signing_keypair_unique() {
        let (pk1, _) = generate_signing_keypair().expect("gen");
        let (pk2, _) = generate_signing_keypair().expect("gen");
        assert_ne!(pk1, pk2);
    }

    #[test]
    fn test_generate_signing_keypair_public_is_hash_of_private() {
        let (private_key, public_key) = generate_signing_keypair().expect("gen");
        let mut hasher = Sha256::new();
        hasher.update(&private_key);
        let expected_public = hasher.finalize().to_vec();
        assert_eq!(public_key, expected_public);
    }
}
