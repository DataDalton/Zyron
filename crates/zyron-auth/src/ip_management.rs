//! Global IP block and trust management.
//!
//! Provides shared IP blocking and trusted IP lists used by brute force defense,
//! compliance, and other subsystems. Controlled by ManageAuthRules privilege.

use crate::role::UserId;
use zyron_catalog::encoding::{
    read_string, read_u8, read_u32, read_u64, write_string, write_u8, write_u32, write_u64,
};
use zyron_common::{Result, ZyronError};

/// Source of an IP block action.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum IpBlockSource {
    Admin(UserId),
    BruteForce,
    System(String),
}

impl IpBlockSource {
    /// Serializes the source to bytes. Tag 0 = Admin(u32), tag 1 = BruteForce, tag 2 = System(string).
    pub fn to_bytes(&self, buf: &mut Vec<u8>) {
        match self {
            IpBlockSource::Admin(uid) => {
                write_u8(buf, 0);
                write_u32(buf, uid.0);
            }
            IpBlockSource::BruteForce => {
                write_u8(buf, 1);
            }
            IpBlockSource::System(desc) => {
                write_u8(buf, 2);
                write_string(buf, desc);
            }
        }
    }

    /// Deserializes the source from bytes at the given offset.
    pub fn from_bytes(data: &[u8], offset: &mut usize) -> Result<Self> {
        let tag = read_u8(data, offset)?;
        match tag {
            0 => {
                let uid = read_u32(data, offset)?;
                Ok(IpBlockSource::Admin(UserId(uid)))
            }
            1 => Ok(IpBlockSource::BruteForce),
            2 => {
                let desc = read_string(data, offset)?;
                Ok(IpBlockSource::System(desc))
            }
            _ => Err(ZyronError::DecodingFailed(format!(
                "Invalid IpBlockSource tag: {}",
                tag
            ))),
        }
    }
}

/// A blocked IP or CIDR range with metadata.
#[derive(Debug, Clone)]
pub struct IpBlockEntry {
    pub ip_or_cidr: String,
    pub blocked_at: u64,
    pub expires_at: u64, // 0 = permanent
    pub reason: String,
    pub source: IpBlockSource,
}

impl IpBlockEntry {
    /// Serializes the entry to a byte vector.
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut buf = Vec::with_capacity(128);
        write_string(&mut buf, &self.ip_or_cidr);
        write_u64(&mut buf, self.blocked_at);
        write_u64(&mut buf, self.expires_at);
        write_string(&mut buf, &self.reason);
        self.source.to_bytes(&mut buf);
        buf
    }

    /// Deserializes from bytes.
    pub fn from_bytes(data: &[u8]) -> Result<Self> {
        let mut offset = 0;
        let ip_or_cidr = read_string(data, &mut offset)?;
        let blocked_at = read_u64(data, &mut offset)?;
        let expires_at = read_u64(data, &mut offset)?;
        let reason = read_string(data, &mut offset)?;
        let source = IpBlockSource::from_bytes(data, &mut offset)?;
        Ok(Self {
            ip_or_cidr,
            blocked_at,
            expires_at,
            reason,
            source,
        })
    }
}

/// A trusted IP or CIDR range that bypasses IP-based checks.
#[derive(Debug, Clone)]
pub struct TrustedIpEntry {
    pub ip_or_cidr: String,
    pub added_by: UserId,
    pub added_at: u64,
    pub reason: String,
}

impl TrustedIpEntry {
    /// Serializes the entry to a byte vector.
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut buf = Vec::with_capacity(64);
        write_string(&mut buf, &self.ip_or_cidr);
        write_u32(&mut buf, self.added_by.0);
        write_u64(&mut buf, self.added_at);
        write_string(&mut buf, &self.reason);
        buf
    }

    /// Deserializes from bytes.
    pub fn from_bytes(data: &[u8]) -> Result<Self> {
        let mut offset = 0;
        let ip_or_cidr = read_string(data, &mut offset)?;
        let added_by = UserId(read_u32(data, &mut offset)?);
        let added_at = read_u64(data, &mut offset)?;
        let reason = read_string(data, &mut offset)?;
        Ok(Self {
            ip_or_cidr,
            added_by,
            added_at,
            reason,
        })
    }
}

/// Returns the current time in seconds since the Unix epoch.
fn now_secs() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}

/// Parses an IPv4 dotted-quad string into a 32-bit integer.
fn parse_ipv4(s: &str) -> Option<u32> {
    let parts: Vec<&str> = s.split('.').collect();
    if parts.len() != 4 {
        return None;
    }
    let a = parts[0].parse::<u8>().ok()? as u32;
    let b = parts[1].parse::<u8>().ok()? as u32;
    let c = parts[2].parse::<u8>().ok()? as u32;
    let d = parts[3].parse::<u8>().ok()? as u32;
    Some((a << 24) | (b << 16) | (c << 8) | d)
}

/// Checks if an IP address matches a CIDR range or exact address.
fn matches_cidr(cidr: &str, ip: &str) -> bool {
    let ip_addr = match parse_ipv4(ip) {
        Some(a) => a,
        None => return false,
    };
    if let Some(slash) = cidr.find('/') {
        let net = match parse_ipv4(&cidr[..slash]) {
            Some(a) => a,
            None => return false,
        };
        let bits: u32 = match cidr[slash + 1..].parse() {
            Ok(b) if b <= 32 => b,
            _ => return false,
        };
        let mask = if bits == 0 { 0 } else { !0u32 << (32 - bits) };
        (ip_addr & mask) == (net & mask)
    } else {
        parse_ipv4(cidr) == Some(ip_addr)
    }
}

/// Thread-safe IP blocking and trust management.
pub struct IpManager {
    blocks: scc::HashMap<String, IpBlockEntry>,
    trusted: parking_lot::RwLock<Vec<TrustedIpEntry>>,
}

impl IpManager {
    /// Creates an empty IpManager.
    pub fn new() -> Self {
        Self {
            blocks: scc::HashMap::new(),
            trusted: parking_lot::RwLock::new(Vec::new()),
        }
    }

    /// Bulk loads blocked IP entries from storage.
    pub fn load_blocks(&self, entries: Vec<IpBlockEntry>) {
        for entry in entries {
            let _ = self.blocks.insert_sync(entry.ip_or_cidr.clone(), entry);
        }
    }

    /// Bulk loads trusted IP entries from storage.
    pub fn load_trusted(&self, entries: Vec<TrustedIpEntry>) {
        let mut trusted = self.trusted.write();
        *trusted = entries;
    }

    /// Checks if an IP is blocked. Returns the block entry if found.
    /// Removes expired entries lazily during lookup.
    pub fn is_blocked(&self, ip: &str) -> Option<IpBlockEntry> {
        let now = now_secs();
        let mut expired_keys = Vec::new();
        let mut result = None;

        self.blocks.iter_sync(|key, entry| {
            if entry.expires_at > 0 && entry.expires_at <= now {
                expired_keys.push(key.clone());
                return true;
            }
            if result.is_none() && matches_cidr(key, ip) {
                result = Some(entry.clone());
            }
            true
        });

        for key in expired_keys {
            let _ = self.blocks.remove_sync(&key);
        }

        result
    }

    /// Checks if an IP matches any trusted CIDR range.
    pub fn is_trusted(&self, ip: &str) -> bool {
        let trusted = self.trusted.read();
        for entry in trusted.iter() {
            if matches_cidr(&entry.ip_or_cidr, ip) {
                return true;
            }
        }
        false
    }

    /// Blocks an IP or CIDR range.
    pub fn block_ip(
        &self,
        ip_or_cidr: String,
        expires_at: u64,
        reason: String,
        source: IpBlockSource,
    ) {
        let entry = IpBlockEntry {
            ip_or_cidr: ip_or_cidr.clone(),
            blocked_at: now_secs(),
            expires_at,
            reason,
            source,
        };
        let _ = self.blocks.insert_sync(ip_or_cidr, entry);
    }

    /// Removes a block for the given IP or CIDR. Returns true if found and removed.
    pub fn unblock_ip(&self, ip_or_cidr: &str) -> bool {
        self.blocks.remove_sync(ip_or_cidr).is_some()
    }

    /// Removes all blocks and returns the count removed.
    pub fn unblock_all(&self) -> u32 {
        let mut count = 0u32;
        let mut keys = Vec::new();
        self.blocks.iter_sync(|key, _| {
            keys.push(key.clone());
            count += 1;
            true
        });
        for key in keys {
            let _ = self.blocks.remove_sync(&key);
        }
        count
    }

    /// Adds a trusted IP entry.
    pub fn add_trusted(&self, entry: TrustedIpEntry) {
        let mut trusted = self.trusted.write();
        trusted.push(entry);
    }

    /// Removes a trusted IP entry by its IP or CIDR string. Returns true if found.
    pub fn remove_trusted(&self, ip_or_cidr: &str) -> bool {
        let mut trusted = self.trusted.write();
        let len_before = trusted.len();
        trusted.retain(|e| e.ip_or_cidr != ip_or_cidr);
        trusted.len() < len_before
    }

    /// Returns a snapshot of all active (non-expired) blocks.
    pub fn active_blocks(&self) -> Vec<IpBlockEntry> {
        let now = now_secs();
        let mut result = Vec::new();
        self.blocks.iter_sync(|_, entry| {
            if entry.expires_at == 0 || entry.expires_at > now {
                result.push(entry.clone());
            }
            true
        });
        result
    }

    /// Returns a snapshot of all trusted IP entries.
    pub fn trusted_list(&self) -> Vec<TrustedIpEntry> {
        self.trusted.read().clone()
    }

    /// Removes all expired blocks.
    pub fn cleanup_expired(&self) {
        let now = now_secs();
        let mut expired = Vec::new();
        self.blocks.iter_sync(|key, entry| {
            if entry.expires_at > 0 && entry.expires_at <= now {
                expired.push(key.clone());
            }
            true
        });
        for key in expired {
            let _ = self.blocks.remove_sync(&key);
        }
    }

    /// Returns all block entries for persistence.
    pub fn export_blocks(&self) -> Vec<IpBlockEntry> {
        let mut result = Vec::new();
        self.blocks.iter_sync(|_, entry| {
            result.push(entry.clone());
            true
        });
        result
    }

    /// Returns all trusted entries for persistence.
    pub fn export_trusted(&self) -> Vec<TrustedIpEntry> {
        self.trusted.read().clone()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_block_and_is_blocked() {
        let mgr = IpManager::new();
        mgr.block_ip(
            "10.0.0.1".to_string(),
            0,
            "test block".to_string(),
            IpBlockSource::Admin(UserId(1)),
        );
        let entry = mgr.is_blocked("10.0.0.1");
        assert!(entry.is_some());
        let entry = entry.unwrap();
        assert_eq!(entry.ip_or_cidr, "10.0.0.1");
        assert_eq!(entry.reason, "test block");
    }

    #[test]
    fn test_block_expired_removed() {
        let mgr = IpManager::new();
        // Block with an already-expired timestamp
        mgr.block_ip(
            "10.0.0.2".to_string(),
            1, // expired (epoch + 1 second)
            "expired block".to_string(),
            IpBlockSource::BruteForce,
        );
        let entry = mgr.is_blocked("10.0.0.2");
        assert!(entry.is_none(), "Expired block should be removed lazily");
    }

    #[test]
    fn test_permanent_block_never_expires() {
        let mgr = IpManager::new();
        mgr.block_ip(
            "10.0.0.3".to_string(),
            0, // permanent
            "permanent".to_string(),
            IpBlockSource::System("compliance".to_string()),
        );
        let entry = mgr.is_blocked("10.0.0.3");
        assert!(entry.is_some());
        assert_eq!(entry.unwrap().expires_at, 0);
    }

    #[test]
    fn test_unblock() {
        let mgr = IpManager::new();
        mgr.block_ip(
            "10.0.0.4".to_string(),
            0,
            "test".to_string(),
            IpBlockSource::BruteForce,
        );
        assert!(mgr.is_blocked("10.0.0.4").is_some());
        let removed = mgr.unblock_ip("10.0.0.4");
        assert!(removed);
        assert!(mgr.is_blocked("10.0.0.4").is_none());
    }

    #[test]
    fn test_unblock_all() {
        let mgr = IpManager::new();
        mgr.block_ip(
            "10.0.0.5".to_string(),
            0,
            "a".to_string(),
            IpBlockSource::BruteForce,
        );
        mgr.block_ip(
            "10.0.0.6".to_string(),
            0,
            "b".to_string(),
            IpBlockSource::BruteForce,
        );
        mgr.block_ip(
            "10.0.0.7".to_string(),
            0,
            "c".to_string(),
            IpBlockSource::BruteForce,
        );
        let count = mgr.unblock_all();
        assert_eq!(count, 3);
        assert!(mgr.is_blocked("10.0.0.5").is_none());
        assert!(mgr.is_blocked("10.0.0.6").is_none());
    }

    #[test]
    fn test_trusted_ip_exact() {
        let mgr = IpManager::new();
        mgr.add_trusted(TrustedIpEntry {
            ip_or_cidr: "192.168.1.100".to_string(),
            added_by: UserId(1),
            added_at: 1000,
            reason: "office".to_string(),
        });
        assert!(mgr.is_trusted("192.168.1.100"));
        assert!(!mgr.is_trusted("192.168.1.101"));
    }

    #[test]
    fn test_trusted_ip_cidr() {
        let mgr = IpManager::new();
        mgr.add_trusted(TrustedIpEntry {
            ip_or_cidr: "10.0.0.0/24".to_string(),
            added_by: UserId(1),
            added_at: 1000,
            reason: "internal".to_string(),
        });
        assert!(mgr.is_trusted("10.0.0.1"));
        assert!(mgr.is_trusted("10.0.0.255"));
        assert!(!mgr.is_trusted("10.0.1.1"));
    }

    #[test]
    fn test_trusted_ip_not_matched() {
        let mgr = IpManager::new();
        mgr.add_trusted(TrustedIpEntry {
            ip_or_cidr: "172.16.0.0/16".to_string(),
            added_by: UserId(1),
            added_at: 1000,
            reason: "vpn".to_string(),
        });
        assert!(!mgr.is_trusted("192.168.1.1"));
    }

    #[test]
    fn test_block_cidr_range() {
        let mgr = IpManager::new();
        mgr.block_ip(
            "10.0.0.0/24".to_string(),
            0,
            "range block".to_string(),
            IpBlockSource::Admin(UserId(1)),
        );
        assert!(mgr.is_blocked("10.0.0.50").is_some());
        assert!(mgr.is_blocked("10.0.0.200").is_some());
        assert!(mgr.is_blocked("10.0.1.1").is_none());
    }

    #[test]
    fn test_cleanup_expired() {
        let mgr = IpManager::new();
        mgr.block_ip(
            "10.0.0.10".to_string(),
            1,
            "old".to_string(),
            IpBlockSource::BruteForce,
        );
        mgr.block_ip(
            "10.0.0.11".to_string(),
            0,
            "permanent".to_string(),
            IpBlockSource::BruteForce,
        );
        mgr.cleanup_expired();
        // Expired one should be gone
        assert!(mgr.is_blocked("10.0.0.10").is_none());
        // Permanent one should remain
        assert!(mgr.is_blocked("10.0.0.11").is_some());
    }

    #[test]
    fn test_ip_block_entry_roundtrip() {
        let entry = IpBlockEntry {
            ip_or_cidr: "10.0.0.0/8".to_string(),
            blocked_at: 1700000000,
            expires_at: 1700003600,
            reason: "suspicious activity".to_string(),
            source: IpBlockSource::Admin(UserId(42)),
        };
        let bytes = entry.to_bytes();
        let decoded = IpBlockEntry::from_bytes(&bytes).unwrap();
        assert_eq!(decoded.ip_or_cidr, entry.ip_or_cidr);
        assert_eq!(decoded.blocked_at, entry.blocked_at);
        assert_eq!(decoded.expires_at, entry.expires_at);
        assert_eq!(decoded.reason, entry.reason);
        assert_eq!(decoded.source, IpBlockSource::Admin(UserId(42)));
    }

    #[test]
    fn test_trusted_ip_entry_roundtrip() {
        let entry = TrustedIpEntry {
            ip_or_cidr: "192.168.0.0/16".to_string(),
            added_by: UserId(7),
            added_at: 1700000000,
            reason: "corporate network".to_string(),
        };
        let bytes = entry.to_bytes();
        let decoded = TrustedIpEntry::from_bytes(&bytes).unwrap();
        assert_eq!(decoded.ip_or_cidr, entry.ip_or_cidr);
        assert_eq!(decoded.added_by, UserId(7));
        assert_eq!(decoded.added_at, entry.added_at);
        assert_eq!(decoded.reason, entry.reason);
    }

    #[test]
    fn test_ip_block_source_roundtrip() {
        // Admin variant
        let mut buf = Vec::new();
        IpBlockSource::Admin(UserId(99)).to_bytes(&mut buf);
        let mut offset = 0;
        let decoded = IpBlockSource::from_bytes(&buf, &mut offset).unwrap();
        assert_eq!(decoded, IpBlockSource::Admin(UserId(99)));

        // BruteForce variant
        let mut buf = Vec::new();
        IpBlockSource::BruteForce.to_bytes(&mut buf);
        let mut offset = 0;
        let decoded = IpBlockSource::from_bytes(&buf, &mut offset).unwrap();
        assert_eq!(decoded, IpBlockSource::BruteForce);

        // System variant
        let mut buf = Vec::new();
        IpBlockSource::System("firewall rule".to_string()).to_bytes(&mut buf);
        let mut offset = 0;
        let decoded = IpBlockSource::from_bytes(&buf, &mut offset).unwrap();
        assert_eq!(decoded, IpBlockSource::System("firewall rule".to_string()));
    }
}
