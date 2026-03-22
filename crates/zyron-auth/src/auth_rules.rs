//! Authentication method resolution (pg_hba.conf-style rules).
//!
//! AuthResolver evaluates a sorted list of rules to determine which
//! authentication method applies to a connection. Rules are matched
//! by connection type, database, user, and source IP in priority order.
//! The first matching rule wins. If no rule matches, Trust is returned.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use zyron_common::{Result, ZyronError};

/// Authentication methods supported by the server.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[repr(u8)]
pub enum AuthMethod {
    Trust = 0,
    Reject = 1,
    Password = 2,
    Md5 = 3,
    ScramSha256 = 4,
    BalloonSha256 = 5,
    Certificate = 6,
    Jwt = 7,
    ApiKey = 8,
    PasswordAndTotp = 9,
    Fido2 = 10,
    PasswordAndFido2 = 11,
}

impl AuthMethod {
    /// Converts a u8 to an AuthMethod.
    pub fn from_u8(v: u8) -> Result<Self> {
        match v {
            0 => Ok(AuthMethod::Trust),
            1 => Ok(AuthMethod::Reject),
            2 => Ok(AuthMethod::Password),
            3 => Ok(AuthMethod::Md5),
            4 => Ok(AuthMethod::ScramSha256),
            5 => Ok(AuthMethod::BalloonSha256),
            6 => Ok(AuthMethod::Certificate),
            7 => Ok(AuthMethod::Jwt),
            8 => Ok(AuthMethod::ApiKey),
            9 => Ok(AuthMethod::PasswordAndTotp),
            10 => Ok(AuthMethod::Fido2),
            11 => Ok(AuthMethod::PasswordAndFido2),
            _ => Err(ZyronError::Internal(format!(
                "Invalid AuthMethod value: {}",
                v
            ))),
        }
    }
}

/// The type of connection being made.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ConnectionType {
    Local,
    Host,
    HostSsl,
    HostNoSsl,
    HostQuic,
    All,
}

impl ConnectionType {
    fn to_u8(self) -> u8 {
        match self {
            ConnectionType::Local => 0,
            ConnectionType::Host => 1,
            ConnectionType::HostSsl => 2,
            ConnectionType::HostNoSsl => 3,
            ConnectionType::HostQuic => 4,
            ConnectionType::All => 5,
        }
    }

    fn from_u8(v: u8) -> Result<Self> {
        match v {
            0 => Ok(ConnectionType::Local),
            1 => Ok(ConnectionType::Host),
            2 => Ok(ConnectionType::HostSsl),
            3 => Ok(ConnectionType::HostNoSsl),
            4 => Ok(ConnectionType::HostQuic),
            5 => Ok(ConnectionType::All),
            _ => Err(ZyronError::Internal(format!(
                "Invalid ConnectionType value: {}",
                v
            ))),
        }
    }
}

/// A single authentication rule evaluated during connection authentication.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AuthRule {
    /// Lower priority values are evaluated first.
    pub priority: u16,
    pub connection_type: ConnectionType,
    /// Pattern for database matching. Supports "*" (all), exact, or comma-separated list.
    pub database_pattern: String,
    /// Pattern for user matching. Supports "*" (all), exact, or comma-separated list.
    pub user_pattern: String,
    /// Optional CIDR for source IP matching. IPv4 only (e.g., "10.0.0.0/8").
    pub source_cidr: Option<String>,
    pub method: AuthMethod,
    /// Additional method-specific options (e.g., jwt_issuer, certificate_cn).
    pub options: HashMap<String, String>,
}

impl AuthRule {
    /// Serializes the rule to bytes.
    /// Layout: priority(2) + conn_type(1) + method(1) + has_cidr(1)
    ///       + db_len(2) + db(N) + user_len(2) + user(N)
    ///       + [cidr_len(2) + cidr(N)] + options_count(2) + [key_len(2) + key + val_len(2) + val]*
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut buf = Vec::with_capacity(64);
        buf.extend_from_slice(&self.priority.to_le_bytes());
        buf.push(self.connection_type.to_u8());
        buf.push(self.method as u8);

        let db_bytes = self.database_pattern.as_bytes();
        buf.extend_from_slice(&(db_bytes.len() as u16).to_le_bytes());
        buf.extend_from_slice(db_bytes);

        let user_bytes = self.user_pattern.as_bytes();
        buf.extend_from_slice(&(user_bytes.len() as u16).to_le_bytes());
        buf.extend_from_slice(user_bytes);

        match &self.source_cidr {
            Some(cidr) => {
                buf.push(1);
                let cidr_bytes = cidr.as_bytes();
                buf.extend_from_slice(&(cidr_bytes.len() as u16).to_le_bytes());
                buf.extend_from_slice(cidr_bytes);
            }
            None => {
                buf.push(0);
            }
        }

        buf.extend_from_slice(&(self.options.len() as u16).to_le_bytes());
        for (key, val) in &self.options {
            let kb = key.as_bytes();
            let vb = val.as_bytes();
            buf.extend_from_slice(&(kb.len() as u16).to_le_bytes());
            buf.extend_from_slice(kb);
            buf.extend_from_slice(&(vb.len() as u16).to_le_bytes());
            buf.extend_from_slice(vb);
        }

        buf
    }

    /// Deserializes a rule from bytes.
    pub fn from_bytes(data: &[u8]) -> Result<Self> {
        if data.len() < 8 {
            return Err(ZyronError::DecodingFailed(
                "AuthRule data too short".to_string(),
            ));
        }
        let mut pos = 0;

        let priority = u16::from_le_bytes([data[pos], data[pos + 1]]);
        pos += 2;
        let connection_type = ConnectionType::from_u8(data[pos])?;
        pos += 1;
        let method = AuthMethod::from_u8(data[pos])?;
        pos += 1;

        let db_len = read_u16(data, &mut pos)?;
        let database_pattern = read_str(data, &mut pos, db_len)?;

        let user_len = read_u16(data, &mut pos)?;
        let user_pattern = read_str(data, &mut pos, user_len)?;

        if pos >= data.len() {
            return Err(ZyronError::DecodingFailed(
                "AuthRule truncated at cidr flag".to_string(),
            ));
        }
        let has_cidr = data[pos];
        pos += 1;
        let source_cidr = if has_cidr == 1 {
            let cidr_len = read_u16(data, &mut pos)?;
            Some(read_str(data, &mut pos, cidr_len)?)
        } else {
            None
        };

        let opt_count = read_u16(data, &mut pos)?;
        let mut options = HashMap::new();
        for _ in 0..opt_count {
            let kl = read_u16(data, &mut pos)?;
            let key = read_str(data, &mut pos, kl)?;
            let vl = read_u16(data, &mut pos)?;
            let val = read_str(data, &mut pos, vl)?;
            options.insert(key, val);
        }

        Ok(Self {
            priority,
            connection_type,
            database_pattern,
            user_pattern,
            source_cidr,
            method,
            options,
        })
    }
}

/// Helper to read a u16 from a byte slice at the given offset.
fn read_u16(data: &[u8], pos: &mut usize) -> Result<usize> {
    if *pos + 2 > data.len() {
        return Err(ZyronError::DecodingFailed(
            "AuthRule truncated reading u16".to_string(),
        ));
    }
    let val = u16::from_le_bytes([data[*pos], data[*pos + 1]]) as usize;
    *pos += 2;
    Ok(val)
}

/// Helper to read a UTF-8 string of the given length from a byte slice.
fn read_str(data: &[u8], pos: &mut usize, len: usize) -> Result<String> {
    if *pos + len > data.len() {
        return Err(ZyronError::DecodingFailed(
            "AuthRule truncated reading string".to_string(),
        ));
    }
    let s = std::str::from_utf8(&data[*pos..*pos + len])
        .map_err(|_| ZyronError::DecodingFailed("Invalid UTF-8 in AuthRule".to_string()))?
        .to_string();
    *pos += len;
    Ok(s)
}

/// Resolves authentication methods from a priority-sorted list of rules.
pub struct AuthResolver {
    rules: Vec<AuthRule>,
}

impl AuthResolver {
    /// Creates a new resolver, sorting rules by priority (ascending).
    pub fn new(mut rules: Vec<AuthRule>) -> Self {
        rules.sort_by_key(|r| r.priority);
        Self { rules }
    }

    /// Returns the authentication method for the first matching rule.
    /// If no rule matches, returns Trust.
    pub fn resolve(
        &self,
        conn_type: ConnectionType,
        database: &str,
        user: &str,
        source_ip: Option<&str>,
    ) -> AuthMethod {
        for rule in &self.rules {
            if matches_rule(rule, conn_type, database, user, source_ip) {
                return rule.method;
            }
        }
        AuthMethod::Trust
    }
}

/// Checks if a rule matches the given connection parameters.
fn matches_rule(
    rule: &AuthRule,
    conn_type: ConnectionType,
    database: &str,
    user: &str,
    source_ip: Option<&str>,
) -> bool {
    if !matches_connection_type(rule.connection_type, conn_type) {
        return false;
    }
    if !matches_pattern(&rule.database_pattern, database) {
        return false;
    }
    if !matches_pattern(&rule.user_pattern, user) {
        return false;
    }
    if let Some(ref cidr) = rule.source_cidr {
        match source_ip {
            Some(ip) => {
                if !matches_cidr(cidr, ip) {
                    return false;
                }
            }
            None => return false,
        }
    }
    true
}

/// Checks if the rule's connection type matches the actual connection type.
/// ConnectionType::All matches everything. Otherwise requires exact match.
fn matches_connection_type(rule_type: ConnectionType, actual: ConnectionType) -> bool {
    rule_type == ConnectionType::All || rule_type == actual
}

/// Checks if a pattern matches a value.
/// Patterns: "*" matches everything, otherwise exact match or comma-separated list.
fn matches_pattern(pattern: &str, value: &str) -> bool {
    if pattern == "*" {
        return true;
    }
    // Check comma-separated list
    if pattern.contains(',') {
        for part in pattern.split(',') {
            if part.trim() == value {
                return true;
            }
        }
        return false;
    }
    pattern == value
}

/// Checks if an IP address falls within a CIDR range. IPv4 only.
/// Returns false on any parse error.
fn matches_cidr(cidr: &str, ip: &str) -> bool {
    let (net_str, mask_str) = match cidr.split_once('/') {
        Some(parts) => parts,
        None => return cidr == ip,
    };

    let net_addr = match parse_ipv4(net_str) {
        Some(addr) => addr,
        None => return false,
    };
    let ip_addr = match parse_ipv4(ip) {
        Some(addr) => addr,
        None => return false,
    };
    let mask_bits: u32 = match mask_str.parse() {
        Ok(m) if m <= 32 => m,
        _ => return false,
    };

    if mask_bits == 0 {
        return true;
    }

    let mask = if mask_bits == 32 {
        u32::MAX
    } else {
        u32::MAX << (32 - mask_bits)
    };

    (net_addr & mask) == (ip_addr & mask)
}

/// Parses an IPv4 address string "a.b.c.d" into a u32.
fn parse_ipv4(s: &str) -> Option<u32> {
    let parts: Vec<&str> = s.split('.').collect();
    if parts.len() != 4 {
        return None;
    }
    let a: u8 = parts[0].parse().ok()?;
    let b: u8 = parts[1].parse().ok()?;
    let c: u8 = parts[2].parse().ok()?;
    let d: u8 = parts[3].parse().ok()?;
    Some(u32::from_be_bytes([a, b, c, d]))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_rule(
        priority: u16,
        conn: ConnectionType,
        db: &str,
        user: &str,
        cidr: Option<&str>,
        method: AuthMethod,
    ) -> AuthRule {
        AuthRule {
            priority,
            connection_type: conn,
            database_pattern: db.to_string(),
            user_pattern: user.to_string(),
            source_cidr: cidr.map(|s| s.to_string()),
            method,
            options: HashMap::new(),
        }
    }

    #[test]
    fn test_exact_match() {
        let resolver = AuthResolver::new(vec![make_rule(
            1,
            ConnectionType::Host,
            "mydb",
            "alice",
            None,
            AuthMethod::ScramSha256,
        )]);
        assert_eq!(
            resolver.resolve(ConnectionType::Host, "mydb", "alice", None),
            AuthMethod::ScramSha256,
        );
    }

    #[test]
    fn test_wildcard_match() {
        let resolver = AuthResolver::new(vec![make_rule(
            1,
            ConnectionType::All,
            "*",
            "*",
            None,
            AuthMethod::Md5,
        )]);
        assert_eq!(
            resolver.resolve(ConnectionType::HostSsl, "anydb", "anyuser", None),
            AuthMethod::Md5,
        );
    }

    #[test]
    fn test_no_match_returns_trust() {
        let resolver = AuthResolver::new(vec![make_rule(
            1,
            ConnectionType::Local,
            "mydb",
            "alice",
            None,
            AuthMethod::Reject,
        )]);
        // Connection type does not match.
        assert_eq!(
            resolver.resolve(ConnectionType::Host, "mydb", "alice", None),
            AuthMethod::Trust,
        );
    }

    #[test]
    fn test_priority_ordering() {
        let resolver = AuthResolver::new(vec![
            make_rule(
                10,
                ConnectionType::All,
                "*",
                "*",
                None,
                AuthMethod::Password,
            ),
            make_rule(
                1,
                ConnectionType::All,
                "*",
                "*",
                None,
                AuthMethod::ScramSha256,
            ),
        ]);
        // Priority 1 should match first even though it was added second.
        assert_eq!(
            resolver.resolve(ConnectionType::Host, "db", "user", None),
            AuthMethod::ScramSha256,
        );
    }

    #[test]
    fn test_cidr_match() {
        let resolver = AuthResolver::new(vec![make_rule(
            1,
            ConnectionType::Host,
            "*",
            "*",
            Some("192.168.1.0/24"),
            AuthMethod::BalloonSha256,
        )]);
        assert_eq!(
            resolver.resolve(ConnectionType::Host, "db", "user", Some("192.168.1.50")),
            AuthMethod::BalloonSha256,
        );
        // IP outside the subnet.
        assert_eq!(
            resolver.resolve(ConnectionType::Host, "db", "user", Some("10.0.0.1")),
            AuthMethod::Trust,
        );
    }

    #[test]
    fn test_cidr_match_exact_host() {
        let resolver = AuthResolver::new(vec![make_rule(
            1,
            ConnectionType::Host,
            "*",
            "*",
            Some("10.0.0.5/32"),
            AuthMethod::Certificate,
        )]);
        assert_eq!(
            resolver.resolve(ConnectionType::Host, "db", "user", Some("10.0.0.5")),
            AuthMethod::Certificate,
        );
        assert_eq!(
            resolver.resolve(ConnectionType::Host, "db", "user", Some("10.0.0.6")),
            AuthMethod::Trust,
        );
    }

    #[test]
    fn test_cidr_requires_ip() {
        let resolver = AuthResolver::new(vec![make_rule(
            1,
            ConnectionType::Host,
            "*",
            "*",
            Some("10.0.0.0/8"),
            AuthMethod::Reject,
        )]);
        // No source IP provided, rule with CIDR should not match.
        assert_eq!(
            resolver.resolve(ConnectionType::Host, "db", "user", None),
            AuthMethod::Trust,
        );
    }

    #[test]
    fn test_comma_separated_pattern() {
        let resolver = AuthResolver::new(vec![make_rule(
            1,
            ConnectionType::All,
            "db1,db2,db3",
            "*",
            None,
            AuthMethod::Jwt,
        )]);
        assert_eq!(
            resolver.resolve(ConnectionType::Host, "db2", "user", None),
            AuthMethod::Jwt,
        );
        assert_eq!(
            resolver.resolve(ConnectionType::Host, "db4", "user", None),
            AuthMethod::Trust,
        );
    }

    #[test]
    fn test_matches_cidr_invalid_ip() {
        assert!(!matches_cidr("10.0.0.0/8", "not_an_ip"));
    }

    #[test]
    fn test_matches_cidr_invalid_cidr() {
        assert!(!matches_cidr("not_cidr/8", "10.0.0.1"));
    }

    #[test]
    fn test_matches_cidr_no_mask() {
        // CIDR without slash treated as exact match.
        assert!(matches_cidr("10.0.0.1", "10.0.0.1"));
        assert!(!matches_cidr("10.0.0.1", "10.0.0.2"));
    }

    #[test]
    fn test_matches_cidr_zero_mask() {
        // /0 matches all IPs.
        assert!(matches_cidr("0.0.0.0/0", "192.168.1.1"));
    }

    #[test]
    fn test_parse_ipv4_valid() {
        assert_eq!(parse_ipv4("10.0.0.1"), Some(0x0A000001));
        assert_eq!(parse_ipv4("255.255.255.255"), Some(0xFFFFFFFF));
        assert_eq!(parse_ipv4("0.0.0.0"), Some(0));
    }

    #[test]
    fn test_parse_ipv4_invalid() {
        assert!(parse_ipv4("256.0.0.1").is_none());
        assert!(parse_ipv4("10.0.0").is_none());
        assert!(parse_ipv4("not.an.ip.addr").is_none());
    }

    #[test]
    fn test_auth_rule_to_bytes_from_bytes() {
        let mut opts = HashMap::new();
        opts.insert("issuer".to_string(), "https://auth.example.com".to_string());
        let rule = AuthRule {
            priority: 5,
            connection_type: ConnectionType::HostSsl,
            database_pattern: "production".to_string(),
            user_pattern: "admin,dba".to_string(),
            source_cidr: Some("10.0.0.0/8".to_string()),
            method: AuthMethod::Jwt,
            options: opts,
        };
        let bytes = rule.to_bytes();
        let restored = AuthRule::from_bytes(&bytes).expect("decode should succeed");
        assert_eq!(restored.priority, 5);
        assert_eq!(restored.connection_type, ConnectionType::HostSsl);
        assert_eq!(restored.database_pattern, "production");
        assert_eq!(restored.user_pattern, "admin,dba");
        assert_eq!(restored.source_cidr.as_deref(), Some("10.0.0.0/8"));
        assert_eq!(restored.method, AuthMethod::Jwt);
        assert_eq!(
            restored.options.get("issuer").map(|s| s.as_str()),
            Some("https://auth.example.com")
        );
    }

    #[test]
    fn test_auth_rule_to_bytes_from_bytes_no_cidr_no_opts() {
        let rule = make_rule(
            1,
            ConnectionType::Local,
            "db",
            "user",
            None,
            AuthMethod::Trust,
        );
        let bytes = rule.to_bytes();
        let restored = AuthRule::from_bytes(&bytes).expect("decode should succeed");
        assert_eq!(restored.priority, 1);
        assert_eq!(restored.connection_type, ConnectionType::Local);
        assert!(restored.source_cidr.is_none());
        assert!(restored.options.is_empty());
    }

    #[test]
    fn test_auth_rule_from_bytes_too_short() {
        assert!(AuthRule::from_bytes(&[0u8; 3]).is_err());
    }

    #[test]
    fn test_auth_method_from_u8_all_variants() {
        for i in 0..=11u8 {
            assert!(AuthMethod::from_u8(i).is_ok());
        }
        assert!(AuthMethod::from_u8(12).is_err());
    }

    #[test]
    fn test_connection_type_roundtrip() {
        let types = [
            ConnectionType::Local,
            ConnectionType::Host,
            ConnectionType::HostSsl,
            ConnectionType::HostNoSsl,
            ConnectionType::HostQuic,
            ConnectionType::All,
        ];
        for ct in types {
            let restored = ConnectionType::from_u8(ct.to_u8()).expect("roundtrip");
            assert_eq!(restored, ct);
        }
    }

    #[test]
    fn test_first_matching_rule_wins() {
        let resolver = AuthResolver::new(vec![
            make_rule(
                1,
                ConnectionType::Host,
                "mydb",
                "alice",
                None,
                AuthMethod::ScramSha256,
            ),
            make_rule(2, ConnectionType::All, "*", "*", None, AuthMethod::Password),
        ]);
        // alice on mydb should match rule 1, not the wildcard rule 2.
        assert_eq!(
            resolver.resolve(ConnectionType::Host, "mydb", "alice", None),
            AuthMethod::ScramSha256,
        );
        // Other users should match rule 2.
        assert_eq!(
            resolver.resolve(ConnectionType::Host, "mydb", "bob", None),
            AuthMethod::Password,
        );
    }
}
