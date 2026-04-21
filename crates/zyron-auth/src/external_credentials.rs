// -----------------------------------------------------------------------------
// External credential sealing
// -----------------------------------------------------------------------------
//
// Encrypts and decrypts credential blobs for external sources and sinks
// (S3, GCS, Azure, etc) using the shared KeyStore infrastructure. Callers
// never store plaintext credentials in the catalog. The catalog entry
// carries a key handle and an AES-GCM ciphertext. Decryption happens only
// at job spawn time through a SecurityManager-authenticated caller.
//
// Credentials are modeled as a flat HashMap<String, String> so backend
// adapters can define their own keys (aws_access_key_id, sas_token, etc)
// without the auth layer knowing about each backend. The full map is
// JSON-encoded before encryption and JSON-decoded after decryption.

use std::collections::HashMap;

use zyron_common::{Result, ZyronError};

use crate::encryption::{EncryptionAlgorithm, KeyStore, decrypt_value, encrypt_value};

// -----------------------------------------------------------------------------
// Sealed credential blob
// -----------------------------------------------------------------------------

/// A credential blob paired with the KeyStore handle used to encrypt it.
/// Persisted alongside the external source/sink entry in the catalog.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SealedCredentials {
    pub key_id: u32,
    pub ciphertext: Vec<u8>,
}

// -----------------------------------------------------------------------------
// Seal and open
// -----------------------------------------------------------------------------

/// Encrypts a credential map using a freshly allocated AES-256-GCM key from
/// the given key store. Returns the sealed bytes plus the key handle that
/// must be stored with the catalog entry.
pub fn seal_credentials(
    credentials: &HashMap<String, String>,
    key_store: &dyn KeyStore,
) -> Result<SealedCredentials> {
    let plaintext = serde_json::to_vec(credentials)
        .map_err(|e| ZyronError::Internal(format!("failed to serialize credentials: {e}")))?;
    let key_id = key_store.create_key(EncryptionAlgorithm::Aes256Gcm)?;
    let key_material = key_store.get_key(key_id)?;
    let ciphertext = encrypt_value(
        &plaintext,
        &key_material,
        EncryptionAlgorithm::Aes256Gcm,
        b"external-credentials",
    )?;
    Ok(SealedCredentials { key_id, ciphertext })
}

/// Decrypts a previously sealed credential blob. Returns the plaintext
/// HashMap. Callers must not log or echo the returned map.
pub fn open_credentials(
    sealed: &SealedCredentials,
    key_store: &dyn KeyStore,
) -> Result<HashMap<String, String>> {
    let key_material = key_store.get_key(sealed.key_id)?;
    let plaintext = decrypt_value(
        &sealed.ciphertext,
        &key_material,
        EncryptionAlgorithm::Aes256Gcm,
        b"external-credentials",
    )?;
    let parsed: HashMap<String, String> = serde_json::from_slice(&plaintext)
        .map_err(|e| ZyronError::Internal(format!("failed to parse credentials: {e}")))?;
    Ok(parsed)
}

/// Re-encrypts credentials under a fresh key. Used by credential rotation.
pub fn rotate_credentials(
    old: &SealedCredentials,
    key_store: &dyn KeyStore,
) -> Result<SealedCredentials> {
    let plaintext = open_credentials(old, key_store)?;
    seal_credentials(&plaintext, key_store)
}

// -----------------------------------------------------------------------------
// Tests
// -----------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::encryption::LocalKeyStore;

    fn sample_creds() -> HashMap<String, String> {
        let mut m = HashMap::new();
        m.insert("aws_access_key_id".to_string(), "AKIAEXAMPLE".to_string());
        m.insert(
            "aws_secret_access_key".to_string(),
            "wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY".to_string(),
        );
        m
    }

    #[test]
    fn roundtrip_sealed_credentials() {
        let ks = LocalKeyStore::new([7u8; 32]);
        let creds = sample_creds();
        let sealed = seal_credentials(&creds, &ks).expect("seal");
        assert!(!sealed.ciphertext.is_empty());
        assert!(sealed.key_id > 0);
        let opened = open_credentials(&sealed, &ks).expect("open");
        assert_eq!(opened, creds);
    }

    #[test]
    fn rotate_produces_new_key_id() {
        let ks = LocalKeyStore::new([9u8; 32]);
        let creds = sample_creds();
        let sealed = seal_credentials(&creds, &ks).unwrap();
        let rotated = rotate_credentials(&sealed, &ks).unwrap();
        assert_ne!(sealed.key_id, rotated.key_id);
        let opened = open_credentials(&rotated, &ks).unwrap();
        assert_eq!(opened, creds);
    }

    #[test]
    fn open_with_wrong_key_store_fails() {
        let ks_a = LocalKeyStore::new([1u8; 32]);
        let ks_b = LocalKeyStore::new([2u8; 32]);
        let creds = sample_creds();
        let sealed = seal_credentials(&creds, &ks_a).unwrap();
        // Different master key so the wrapped DEK cannot be unwrapped.
        assert!(open_credentials(&sealed, &ks_b).is_err());
    }
}
