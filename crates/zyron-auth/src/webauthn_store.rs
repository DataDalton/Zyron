//! In-memory WebAuthn credential store backed by heap storage.
//!
//! Stores WebAuthn credentials indexed by UserId for fast lookup during
//! authentication. Supports multiple credentials per user (multiple hardware keys).

use crate::rcu::RcuMap;
use crate::role::UserId;
use crate::webauthn::WebAuthnCredential;
use zyron_common::{Result, ZyronError};

/// Concurrent in-memory store for WebAuthn credentials, keyed by UserId.
/// Maintains a secondary index from credential_id to UserId for O(1) lookup.
pub struct WebAuthnCredentialStore {
    credentials: RcuMap<UserId, Vec<WebAuthnCredential>>,
    /// Secondary index: credential_id bytes -> owning UserId
    cred_index: RcuMap<Vec<u8>, UserId>,
}

impl WebAuthnCredentialStore {
    pub fn new() -> Self {
        Self {
            credentials: RcuMap::empty_map(),
            cred_index: RcuMap::empty_map(),
        }
    }

    /// Adds a credential for a user. Rejects duplicate credential IDs.
    pub fn add_credential(&self, cred: WebAuthnCredential) -> Result<()> {
        let user_id = cred.user_id;
        let cred_id = cred.credential_id.clone();
        // Check for duplicate first
        let snap = self.credentials.load();
        if let Some(existing) = snap.get(&user_id) {
            if existing.iter().any(|c| c.credential_id == cred_id) {
                return Err(ZyronError::PolicyAlreadyExists(
                    "WebAuthn credential ID already registered".to_string(),
                ));
            }
        }
        drop(snap);
        let cred_id_for_index = cred.credential_id.clone();
        self.credentials.update(|m| {
            m.entry(user_id).or_default().push(cred);
        });
        self.cred_index.update(|m| {
            m.insert(cred_id_for_index, user_id);
        });
        Ok(())
    }

    /// Removes a credential by user_id and credential_id. Returns true if removed.
    pub fn remove_credential(&self, user_id: UserId, credential_id: &[u8]) -> bool {
        let snap = self.credentials.load();
        let has_match = snap
            .get(&user_id)
            .map(|v| v.iter().any(|c| c.credential_id == credential_id))
            .unwrap_or(false);
        if !has_match {
            return false;
        }
        drop(snap);
        let cred_id_owned = credential_id.to_vec();
        self.credentials.update(|m| {
            if let Some(v) = m.get_mut(&user_id) {
                v.retain(|c| c.credential_id != cred_id_owned);
            }
        });
        self.cred_index.remove(&cred_id_owned);
        true
    }

    /// Returns all credentials for a user.
    pub fn credentials_for_user(&self, user_id: UserId) -> Vec<WebAuthnCredential> {
        self.credentials.get(&user_id).unwrap_or_default()
    }

    /// Finds a credential by credential_id using the secondary index (O(1) lookup).
    pub fn find_credential(&self, credential_id: &[u8]) -> Option<WebAuthnCredential> {
        let user_id = self.cred_index.get(&credential_id.to_vec())?;
        let creds = self.credentials.get(&user_id)?;
        creds
            .iter()
            .find(|c| c.credential_id == credential_id)
            .cloned()
    }

    /// Updates the sign count for a credential. Returns true if found and updated.
    pub fn update_sign_count(&self, credential_id: &[u8], new_count: u32) -> bool {
        // O(1) lookup via secondary index
        let user_id = match self.cred_index.get(&credential_id.to_vec()) {
            Some(id) => id,
            None => return false,
        };

        // Mutate via update
        let cred_id_owned = credential_id.to_vec();
        let mut found = false;
        self.credentials.update(|m| {
            if let Some(v) = m.get_mut(&user_id) {
                for cred in v.iter_mut() {
                    if cred.credential_id == cred_id_owned {
                        cred.sign_count = new_count;
                        cred.last_used_at = std::time::SystemTime::now()
                            .duration_since(std::time::UNIX_EPOCH)
                            .unwrap_or_default()
                            .as_secs();
                        found = true;
                        break;
                    }
                }
            }
        });
        found
    }

    /// Bulk-loads credentials from storage.
    pub fn load(&self, credentials: Vec<WebAuthnCredential>) {
        for cred in credentials {
            let _ = self.add_credential(cred);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::webauthn::{CosePublicKey, CredentialTransport};

    fn make_cred(user_id: u32, cred_id: &[u8], name: &str) -> WebAuthnCredential {
        WebAuthnCredential {
            credential_id: cred_id.to_vec(),
            user_id: UserId(user_id),
            public_key: CosePublicKey::P256 {
                x: [0xAA; 32],
                y: [0xBB; 32],
            },
            sign_count: 0,
            transports: vec![CredentialTransport::Usb],
            created_at: 1700000000,
            last_used_at: 1700000000,
            friendly_name: name.to_string(),
        }
    }

    #[test]
    fn test_add_and_get_credentials() {
        let store = WebAuthnCredentialStore::new();
        store
            .add_credential(make_cred(1, &[1, 2, 3], "Key A"))
            .expect("add");
        store
            .add_credential(make_cred(1, &[4, 5, 6], "Key B"))
            .expect("add");

        let creds = store.credentials_for_user(UserId(1));
        assert_eq!(creds.len(), 2);
    }

    #[test]
    fn test_reject_duplicate_credential_id() {
        let store = WebAuthnCredentialStore::new();
        store
            .add_credential(make_cred(1, &[1, 2, 3], "Key A"))
            .expect("add");
        assert!(
            store
                .add_credential(make_cred(1, &[1, 2, 3], "Key B"))
                .is_err()
        );
    }

    #[test]
    fn test_remove_credential() {
        let store = WebAuthnCredentialStore::new();
        store
            .add_credential(make_cred(1, &[1, 2, 3], "Key A"))
            .expect("add");
        assert!(store.remove_credential(UserId(1), &[1, 2, 3]));
        assert!(store.credentials_for_user(UserId(1)).is_empty());
    }

    #[test]
    fn test_remove_nonexistent() {
        let store = WebAuthnCredentialStore::new();
        assert!(!store.remove_credential(UserId(1), &[9, 9, 9]));
    }

    #[test]
    fn test_find_credential() {
        let store = WebAuthnCredentialStore::new();
        store
            .add_credential(make_cred(1, &[1, 2, 3], "Key A"))
            .expect("add");
        store
            .add_credential(make_cred(2, &[4, 5, 6], "Key B"))
            .expect("add");

        let found = store.find_credential(&[4, 5, 6]);
        assert!(found.is_some());
        assert_eq!(found.unwrap().friendly_name, "Key B");

        assert!(store.find_credential(&[9, 9, 9]).is_none());
    }

    #[test]
    fn test_update_sign_count() {
        let store = WebAuthnCredentialStore::new();
        store
            .add_credential(make_cred(1, &[1, 2, 3], "Key A"))
            .expect("add");
        assert!(store.update_sign_count(&[1, 2, 3], 42));

        let cred = store.find_credential(&[1, 2, 3]).unwrap();
        assert_eq!(cred.sign_count, 42);
    }

    #[test]
    fn test_update_sign_count_not_found() {
        let store = WebAuthnCredentialStore::new();
        assert!(!store.update_sign_count(&[9, 9, 9], 1));
    }

    #[test]
    fn test_empty_user() {
        let store = WebAuthnCredentialStore::new();
        assert!(store.credentials_for_user(UserId(999)).is_empty());
    }

    #[test]
    fn test_load_bulk() {
        let store = WebAuthnCredentialStore::new();
        let creds = vec![
            make_cred(1, &[1], "A"),
            make_cred(1, &[2], "B"),
            make_cred(2, &[3], "C"),
        ];
        store.load(creds);
        assert_eq!(store.credentials_for_user(UserId(1)).len(), 2);
        assert_eq!(store.credentials_for_user(UserId(2)).len(), 1);
    }
}
