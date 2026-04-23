//! Identity-to-role mapping store for Kubernetes service accounts, JWT
//! subjects, and mTLS client certificates.
//!
//! This store answers "given an external identity, which Zyron role should the
//! resulting session carry?" Each mapping kind has its own lookup path so the
//! wire handshake can resolve a role in a single read after authentication.

use std::sync::Arc;

use parking_lot::RwLock;
use serde::{Deserialize, Serialize};

use crate::mtls_pinning::Sha256Fingerprint;
use crate::role::RoleId;

// -----------------------------------------------------------------------------
// Mapping kinds
// -----------------------------------------------------------------------------

/// Which security map an entry belongs to. Serialized with entries so a single
/// catalog table can persist every kind.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SecurityMapKind {
    K8sSa = 0,
    Jwt = 1,
    MtlsSubject = 2,
    MtlsFingerprint = 3,
}

impl SecurityMapKind {
    pub fn from_u8(v: u8) -> Option<Self> {
        match v {
            0 => Some(Self::K8sSa),
            1 => Some(Self::Jwt),
            2 => Some(Self::MtlsSubject),
            3 => Some(Self::MtlsFingerprint),
            _ => None,
        }
    }
}

// -----------------------------------------------------------------------------
// Entry shape for persistence
// -----------------------------------------------------------------------------

/// Persistable security-map row. The `key` is kind-specific:
///   - K8sSa: "namespace/name"
///   - Jwt: "issuer|subject"
///   - MtlsSubject: "CN=..."
///   - MtlsFingerprint: "sha256:<hex>"
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SecurityMapEntry {
    pub kind: SecurityMapKind,
    pub key: String,
    pub role: RoleId,
}

impl SecurityMapEntry {
    /// Serializes the entry to bytes.
    pub fn to_bytes(&self) -> Vec<u8> {
        let key_bytes = self.key.as_bytes();
        let mut buf = Vec::with_capacity(1 + 4 + key_bytes.len() + 4);
        buf.push(self.kind as u8);
        buf.extend_from_slice(&(key_bytes.len() as u32).to_le_bytes());
        buf.extend_from_slice(key_bytes);
        buf.extend_from_slice(&self.role.0.to_le_bytes());
        buf
    }

    /// Parses the entry from bytes.
    pub fn from_bytes(data: &[u8]) -> Option<Self> {
        if data.len() < 1 + 4 + 4 {
            return None;
        }
        let kind = SecurityMapKind::from_u8(data[0])?;
        let key_len = u32::from_le_bytes([data[1], data[2], data[3], data[4]]) as usize;
        if data.len() < 5 + key_len + 4 {
            return None;
        }
        let key = std::str::from_utf8(&data[5..5 + key_len]).ok()?.to_string();
        let role_bytes = &data[5 + key_len..5 + key_len + 4];
        let role = RoleId(u32::from_le_bytes([
            role_bytes[0],
            role_bytes[1],
            role_bytes[2],
            role_bytes[3],
        ]));
        Some(SecurityMapEntry { kind, key, role })
    }
}

// -----------------------------------------------------------------------------
// Store
// -----------------------------------------------------------------------------

/// Thread-safe identity-to-role mapping store.
pub struct SecurityMapStore {
    k8s_sa_map: scc::HashMap<String, RoleId>,
    jwt_map: scc::HashMap<(String, String), RoleId>,
    mtls_subject_map: scc::HashMap<String, RoleId>,
    mtls_fingerprint_map: scc::HashMap<Sha256Fingerprint, RoleId>,
    snapshot_cache: RwLock<Arc<Vec<SecurityMapEntry>>>,
}

impl SecurityMapStore {
    pub fn new() -> Self {
        Self {
            k8s_sa_map: scc::HashMap::new(),
            jwt_map: scc::HashMap::new(),
            mtls_subject_map: scc::HashMap::new(),
            mtls_fingerprint_map: scc::HashMap::new(),
            snapshot_cache: RwLock::new(Arc::new(Vec::new())),
        }
    }

    /// Maps a Kubernetes service account `"namespace/name"` to a Zyron role.
    pub fn map_k8s_sa(&self, sa_qualified: &str, role: RoleId) {
        let _ = self.k8s_sa_map.remove_sync(&sa_qualified.to_string());
        let _ = self.k8s_sa_map.insert_sync(sa_qualified.to_string(), role);
    }

    /// Resolves a mapped Zyron role for a K8s service account by namespace and
    /// name.
    pub fn resolve_k8s_sa(&self, ns: &str, name: &str) -> Option<RoleId> {
        let key = format!("{}/{}", ns, name);
        self.k8s_sa_map.read_sync(&key, |_, v| *v)
    }

    /// Maps a JWT issuer and subject pair to a role.
    pub fn map_jwt(&self, issuer: &str, subject: &str, role: RoleId) {
        let key = (issuer.to_string(), subject.to_string());
        let _ = self.jwt_map.remove_sync(&key);
        let _ = self.jwt_map.insert_sync(key, role);
    }

    pub fn resolve_jwt(&self, issuer: &str, subject: &str) -> Option<RoleId> {
        let key = (issuer.to_string(), subject.to_string());
        self.jwt_map.read_sync(&key, |_, v| *v)
    }

    /// Maps an mTLS distinguished name (for example `CN=svc-a`) to a role.
    pub fn map_mtls_subject(&self, subject: &str, role: RoleId) {
        let _ = self.mtls_subject_map.remove_sync(&subject.to_string());
        let _ = self.mtls_subject_map.insert_sync(subject.to_string(), role);
    }

    pub fn resolve_mtls_subject(&self, subject: &str) -> Option<RoleId> {
        self.mtls_subject_map
            .read_sync(&subject.to_string(), |_, v| *v)
    }

    /// Maps an mTLS certificate fingerprint to a role.
    pub fn map_mtls_fingerprint(&self, fp: Sha256Fingerprint, role: RoleId) {
        let _ = self.mtls_fingerprint_map.remove_sync(&fp);
        let _ = self.mtls_fingerprint_map.insert_sync(fp, role);
    }

    pub fn resolve_mtls_fingerprint(&self, fp: &Sha256Fingerprint) -> Option<RoleId> {
        self.mtls_fingerprint_map.read_sync(fp, |_, v| *v)
    }

    /// Removes a mapping. For JWT keys use `"issuer|subject"`. For fingerprint
    /// keys use the `sha256:<hex>` form.
    pub fn unmap(&self, kind: SecurityMapKind, key: &str) -> bool {
        match kind {
            SecurityMapKind::K8sSa => self.k8s_sa_map.remove_sync(&key.to_string()).is_some(),
            SecurityMapKind::Jwt => {
                let mut parts = key.splitn(2, '|');
                let iss = parts.next().unwrap_or("").to_string();
                let sub = parts.next().unwrap_or("").to_string();
                self.jwt_map.remove_sync(&(iss, sub)).is_some()
            }
            SecurityMapKind::MtlsSubject => self
                .mtls_subject_map
                .remove_sync(&key.to_string())
                .is_some(),
            SecurityMapKind::MtlsFingerprint => match crate::mtls_pinning::parse_pin(key) {
                Ok(fp) => self.mtls_fingerprint_map.remove_sync(&fp).is_some(),
                Err(_) => false,
            },
        }
    }

    /// Returns a snapshot of every mapping currently registered. Used by the
    /// catalog persister to write the full state back to disk.
    pub fn snapshot(&self) -> Vec<SecurityMapEntry> {
        let mut out = Vec::new();
        self.k8s_sa_map.iter_sync(|k, v| {
            out.push(SecurityMapEntry {
                kind: SecurityMapKind::K8sSa,
                key: k.clone(),
                role: *v,
            });
            true
        });
        self.jwt_map.iter_sync(|(iss, sub), v| {
            out.push(SecurityMapEntry {
                kind: SecurityMapKind::Jwt,
                key: format!("{}|{}", iss, sub),
                role: *v,
            });
            true
        });
        self.mtls_subject_map.iter_sync(|k, v| {
            out.push(SecurityMapEntry {
                kind: SecurityMapKind::MtlsSubject,
                key: k.clone(),
                role: *v,
            });
            true
        });
        self.mtls_fingerprint_map.iter_sync(|fp, v| {
            out.push(SecurityMapEntry {
                kind: SecurityMapKind::MtlsFingerprint,
                key: crate::mtls_pinning::format_pin(fp),
                role: *v,
            });
            true
        });
        *self.snapshot_cache.write() = Arc::new(out.clone());
        out
    }

    /// Replaces every mapping with the supplied entries.
    pub fn load(&self, entries: Vec<SecurityMapEntry>) {
        self.k8s_sa_map.clear_sync();
        self.jwt_map.clear_sync();
        self.mtls_subject_map.clear_sync();
        self.mtls_fingerprint_map.clear_sync();
        for entry in entries {
            match entry.kind {
                SecurityMapKind::K8sSa => {
                    let _ = self.k8s_sa_map.insert_sync(entry.key, entry.role);
                }
                SecurityMapKind::Jwt => {
                    let mut parts = entry.key.splitn(2, '|');
                    let iss = parts.next().unwrap_or("").to_string();
                    let sub = parts.next().unwrap_or("").to_string();
                    let _ = self.jwt_map.insert_sync((iss, sub), entry.role);
                }
                SecurityMapKind::MtlsSubject => {
                    let _ = self.mtls_subject_map.insert_sync(entry.key, entry.role);
                }
                SecurityMapKind::MtlsFingerprint => {
                    if let Ok(fp) = crate::mtls_pinning::parse_pin(&entry.key) {
                        let _ = self.mtls_fingerprint_map.insert_sync(fp, entry.role);
                    }
                }
            }
        }
    }
}

impl Default for SecurityMapStore {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn k8s_sa_map_and_resolve() {
        let store = SecurityMapStore::new();
        store.map_k8s_sa("prod/zyron-reader", RoleId(10));
        assert_eq!(
            store.resolve_k8s_sa("prod", "zyron-reader"),
            Some(RoleId(10))
        );
        assert_eq!(store.resolve_k8s_sa("prod", "other"), None);
    }

    #[test]
    fn jwt_map_and_resolve() {
        let store = SecurityMapStore::new();
        store.map_jwt("https://issuer", "user123", RoleId(5));
        assert_eq!(
            store.resolve_jwt("https://issuer", "user123"),
            Some(RoleId(5))
        );
        assert_eq!(store.resolve_jwt("other", "user123"), None);
    }

    #[test]
    fn mtls_subject_map_and_resolve() {
        let store = SecurityMapStore::new();
        store.map_mtls_subject("CN=partner-a", RoleId(1));
        assert_eq!(store.resolve_mtls_subject("CN=partner-a"), Some(RoleId(1)));
        assert!(store.resolve_mtls_subject("CN=partner-b").is_none());
    }

    #[test]
    fn mtls_fingerprint_map_and_resolve() {
        let store = SecurityMapStore::new();
        let fp = [0x55u8; 32];
        store.map_mtls_fingerprint(fp, RoleId(7));
        assert_eq!(store.resolve_mtls_fingerprint(&fp), Some(RoleId(7)));
        assert!(store.resolve_mtls_fingerprint(&[0u8; 32]).is_none());
    }

    #[test]
    fn unmap_removes_entries() {
        let store = SecurityMapStore::new();
        store.map_k8s_sa("prod/a", RoleId(1));
        assert!(store.unmap(SecurityMapKind::K8sSa, "prod/a"));
        assert!(store.resolve_k8s_sa("prod", "a").is_none());
    }

    #[test]
    fn snapshot_load_roundtrip() {
        let store = SecurityMapStore::new();
        store.map_k8s_sa("ns/x", RoleId(1));
        store.map_jwt("issuer", "sub", RoleId(2));
        store.map_mtls_subject("CN=a", RoleId(3));
        store.map_mtls_fingerprint([0x11; 32], RoleId(4));
        let snap = store.snapshot();
        assert_eq!(snap.len(), 4);

        let store2 = SecurityMapStore::new();
        store2.load(snap);
        assert_eq!(store2.resolve_k8s_sa("ns", "x"), Some(RoleId(1)));
        assert_eq!(store2.resolve_jwt("issuer", "sub"), Some(RoleId(2)));
        assert_eq!(store2.resolve_mtls_subject("CN=a"), Some(RoleId(3)));
        assert_eq!(
            store2.resolve_mtls_fingerprint(&[0x11; 32]),
            Some(RoleId(4))
        );
    }

    #[test]
    fn entry_bytes_roundtrip() {
        let e = SecurityMapEntry {
            kind: SecurityMapKind::Jwt,
            key: "issuer|sub".to_string(),
            role: RoleId(99),
        };
        let bytes = e.to_bytes();
        let back = SecurityMapEntry::from_bytes(&bytes).expect("decode");
        assert_eq!(back.kind, SecurityMapKind::Jwt);
        assert_eq!(back.key, "issuer|sub");
        assert_eq!(back.role, RoleId(99));
    }
}
