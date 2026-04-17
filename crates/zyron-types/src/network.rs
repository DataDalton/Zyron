//! Network address types: INET, CIDR, MACADDR.
//!
//! INET/CIDR storage (18 bytes):
//!   byte 0: address family (4 for IPv4, 6 for IPv6)
//!   bytes 1-16: address (IPv4 is zero-padded, IPv6 uses all 16 bytes)
//!   byte 17: prefix length (0-32 for IPv4, 0-128 for IPv6)
//!
//! MACADDR storage (6 bytes): raw address bytes.

use zyron_common::{Result, ZyronError};

// ---------------------------------------------------------------------------
// INET parsing and formatting
// ---------------------------------------------------------------------------

/// Parses an IP address string (with optional CIDR prefix) into an 18-byte representation.
/// Accepts: "192.168.1.1", "192.168.1.0/24", "::1", "fe80::1/64".
pub fn inet_parse(text: &str) -> Result<[u8; 18]> {
    let trimmed = text.trim();
    let (addr_str, prefix_str) = match trimmed.rfind('/') {
        Some(idx) => (&trimmed[..idx], Some(&trimmed[idx + 1..])),
        None => (trimmed, None),
    };

    let mut result = [0u8; 18];

    if addr_str.contains(':') {
        // IPv6
        let ipv6 = parse_ipv6(addr_str)?;
        result[0] = 6;
        result[1..17].copy_from_slice(&ipv6);
        result[17] = match prefix_str {
            Some(p) => {
                let prefix = p
                    .parse::<u8>()
                    .map_err(|e| ZyronError::ExecutionError(format!("Invalid prefix: {}", e)))?;
                if prefix > 128 {
                    return Err(ZyronError::ExecutionError(
                        "IPv6 prefix must be 0-128".into(),
                    ));
                }
                prefix
            }
            None => 128,
        };
    } else {
        // IPv4
        let ipv4 = parse_ipv4(addr_str)?;
        result[0] = 4;
        // Zero-pad bytes 1-12, put IPv4 in bytes 13-16
        result[13..17].copy_from_slice(&ipv4);
        result[17] = match prefix_str {
            Some(p) => {
                let prefix = p
                    .parse::<u8>()
                    .map_err(|e| ZyronError::ExecutionError(format!("Invalid prefix: {}", e)))?;
                if prefix > 32 {
                    return Err(ZyronError::ExecutionError(
                        "IPv4 prefix must be 0-32".into(),
                    ));
                }
                prefix
            }
            None => 32,
        };
    }

    Ok(result)
}

fn parse_ipv4(text: &str) -> Result<[u8; 4]> {
    let parts: Vec<&str> = text.split('.').collect();
    if parts.len() != 4 {
        return Err(ZyronError::ExecutionError(format!(
            "Invalid IPv4 address: {}",
            text
        )));
    }
    let mut bytes = [0u8; 4];
    for (i, part) in parts.iter().enumerate() {
        bytes[i] = part
            .parse::<u8>()
            .map_err(|_| ZyronError::ExecutionError(format!("Invalid IPv4 octet: {}", part)))?;
    }
    Ok(bytes)
}

fn parse_ipv6(text: &str) -> Result<[u8; 16]> {
    let mut result = [0u8; 16];

    // Handle "::" abbreviation
    let (left_str, right_str) = match text.find("::") {
        Some(idx) => (&text[..idx], &text[idx + 2..]),
        None => (text, ""),
    };

    let left_groups: Vec<&str> = if left_str.is_empty() {
        Vec::new()
    } else {
        left_str.split(':').collect()
    };
    let right_groups: Vec<&str> = if right_str.is_empty() {
        Vec::new()
    } else {
        right_str.split(':').collect()
    };

    // No "::" means all 8 groups should be present
    if !text.contains("::") && left_groups.len() != 8 {
        return Err(ZyronError::ExecutionError(format!(
            "Invalid IPv6 address: {}",
            text
        )));
    }

    if left_groups.len() + right_groups.len() > 8 {
        return Err(ZyronError::ExecutionError(format!(
            "Invalid IPv6 address (too many groups): {}",
            text
        )));
    }

    let total = left_groups.len() + right_groups.len();
    let zeros = 8 - total;

    // Parse left groups
    for (i, group) in left_groups.iter().enumerate() {
        let val = u16::from_str_radix(group, 16)
            .map_err(|_| ZyronError::ExecutionError(format!("Invalid IPv6 group: {}", group)))?;
        result[i * 2] = (val >> 8) as u8;
        result[i * 2 + 1] = (val & 0xFF) as u8;
    }

    // Right groups go at the end
    for (i, group) in right_groups.iter().enumerate() {
        let val = u16::from_str_radix(group, 16)
            .map_err(|_| ZyronError::ExecutionError(format!("Invalid IPv6 group: {}", group)))?;
        let pos = left_groups.len() + zeros + i;
        result[pos * 2] = (val >> 8) as u8;
        result[pos * 2 + 1] = (val & 0xFF) as u8;
    }

    Ok(result)
}

/// Formats an 18-byte INET representation as a string.
pub fn inet_format(addr: &[u8; 18]) -> String {
    let family = addr[0];
    let prefix = addr[17];

    if family == 4 {
        let ipv4 = &addr[13..17];
        if prefix == 32 {
            format!("{}.{}.{}.{}", ipv4[0], ipv4[1], ipv4[2], ipv4[3])
        } else {
            format!("{}.{}.{}.{}/{}", ipv4[0], ipv4[1], ipv4[2], ipv4[3], prefix)
        }
    } else if family == 6 {
        let s = format_ipv6(&addr[1..17]);
        if prefix == 128 {
            s
        } else {
            format!("{}/{}", s, prefix)
        }
    } else {
        String::from("invalid")
    }
}

fn format_ipv6(bytes: &[u8]) -> String {
    let mut groups = [0u16; 8];
    for i in 0..8 {
        groups[i] = ((bytes[i * 2] as u16) << 8) | (bytes[i * 2 + 1] as u16);
    }

    // Find longest run of zeros for "::" abbreviation
    let mut best_start = 0;
    let mut best_len = 0;
    let mut current_start = 0;
    let mut current_len = 0;

    for i in 0..8 {
        if groups[i] == 0 {
            if current_len == 0 {
                current_start = i;
            }
            current_len += 1;
            if current_len > best_len {
                best_start = current_start;
                best_len = current_len;
            }
        } else {
            current_len = 0;
        }
    }

    // Only abbreviate runs of 2+ zeros
    if best_len < 2 {
        best_len = 0;
    }

    let mut result = String::new();
    let mut i = 0;
    while i < 8 {
        if best_len > 0 && i == best_start {
            result.push_str("::");
            i += best_len;
            // Skip trailing colon for boundary cases
            if i >= 8 {
                break;
            }
            result.push_str(&format!("{:x}", groups[i]));
            i += 1;
        } else {
            if i > 0 && !result.ends_with(':') {
                result.push(':');
            }
            result.push_str(&format!("{:x}", groups[i]));
            i += 1;
        }
    }

    if result.is_empty() {
        "::".to_string()
    } else {
        result
    }
}

/// Returns the address family (4 or 6).
pub fn inet_family(addr: &[u8; 18]) -> u8 {
    addr[0]
}

/// Returns the prefix length.
pub fn inet_prefix(addr: &[u8; 18]) -> u8 {
    addr[17]
}

/// Checks if `addr` falls within the `network` (CIDR containment).
pub fn inet_contains(network: &[u8; 18], addr: &[u8; 18]) -> bool {
    if network[0] != addr[0] {
        return false;
    }
    let prefix = network[17];
    let addr_bytes = if network[0] == 4 {
        &network[13..17]
    } else {
        &network[1..17]
    };
    let test_bytes = if addr[0] == 4 {
        &addr[13..17]
    } else {
        &addr[1..17]
    };

    let full_bytes = (prefix / 8) as usize;
    let remainder_bits = prefix % 8;

    // Compare full bytes
    if addr_bytes[..full_bytes] != test_bytes[..full_bytes] {
        return false;
    }

    // Compare partial byte
    if remainder_bits > 0 && full_bytes < addr_bytes.len() {
        let mask = (0xFFu8 << (8 - remainder_bits)) & 0xFF;
        if (addr_bytes[full_bytes] & mask) != (test_bytes[full_bytes] & mask) {
            return false;
        }
    }

    true
}

/// Returns the network address (masks host bits to zero).
pub fn inet_network(addr: &[u8; 18]) -> [u8; 18] {
    let mut result = *addr;
    let prefix = addr[17];
    let addr_bytes = if addr[0] == 4 {
        &mut result[13..17]
    } else {
        &mut result[1..17]
    };

    let full_bytes = (prefix / 8) as usize;
    let remainder_bits = prefix % 8;

    // Zero bytes beyond the prefix
    if remainder_bits > 0 && full_bytes < addr_bytes.len() {
        let mask = (0xFFu8 << (8 - remainder_bits)) & 0xFF;
        addr_bytes[full_bytes] &= mask;
        for b in addr_bytes[full_bytes + 1..].iter_mut() {
            *b = 0;
        }
    } else {
        for b in addr_bytes[full_bytes..].iter_mut() {
            *b = 0;
        }
    }

    result
}

/// Returns the broadcast address for IPv4. For IPv6, returns the last address in the prefix.
pub fn inet_broadcast(addr: &[u8; 18]) -> [u8; 18] {
    let mut result = *addr;
    let prefix = addr[17];
    let addr_bytes = if addr[0] == 4 {
        &mut result[13..17]
    } else {
        &mut result[1..17]
    };

    let full_bytes = (prefix / 8) as usize;
    let remainder_bits = prefix % 8;

    // Set host bits to 1
    if remainder_bits > 0 && full_bytes < addr_bytes.len() {
        let mask = (0xFFu8 >> remainder_bits) & 0xFF;
        addr_bytes[full_bytes] |= mask;
        for b in addr_bytes[full_bytes + 1..].iter_mut() {
            *b = 0xFF;
        }
    } else {
        for b in addr_bytes[full_bytes..].iter_mut() {
            *b = 0xFF;
        }
    }

    result
}

/// Returns the netmask for the given prefix length.
pub fn inet_netmask(addr: &[u8; 18]) -> [u8; 18] {
    let mut result = [0u8; 18];
    result[0] = addr[0];
    result[17] = addr[17];
    let prefix = addr[17] as usize;
    let bytes_range = if addr[0] == 4 { 13..17 } else { 1..17 };

    let target = if addr[0] == 4 {
        &mut result[13..17]
    } else {
        &mut result[1..17]
    };
    let max_bits = (bytes_range.len()) * 8;
    let full_bytes = prefix.min(max_bits) / 8;
    let remainder = prefix.min(max_bits) % 8;

    for i in 0..full_bytes {
        target[i] = 0xFF;
    }
    if remainder > 0 && full_bytes < target.len() {
        target[full_bytes] = (0xFFu8 << (8 - remainder)) & 0xFF;
    }

    result
}

/// Returns the host part of an address (zeros out network bits).
pub fn inet_host(addr: &[u8; 18]) -> [u8; 18] {
    let mut result = *addr;
    let prefix = addr[17];
    let addr_bytes = if addr[0] == 4 {
        &mut result[13..17]
    } else {
        &mut result[1..17]
    };

    let full_bytes = (prefix / 8) as usize;
    let remainder_bits = prefix % 8;

    // Zero bytes within the prefix
    for b in addr_bytes[..full_bytes].iter_mut() {
        *b = 0;
    }
    if remainder_bits > 0 && full_bytes < addr_bytes.len() {
        let mask = (0xFFu8 >> remainder_bits) & 0xFF;
        addr_bytes[full_bytes] &= mask;
    }

    result
}

/// Returns true if the address is in a private (RFC 1918 for IPv4, RFC 4193 for IPv6) range.
pub fn inet_is_private(addr: &[u8; 18]) -> bool {
    if addr[0] == 4 {
        let ip = &addr[13..17];
        // 10.0.0.0/8
        if ip[0] == 10 {
            return true;
        }
        // 172.16.0.0/12
        if ip[0] == 172 && (ip[1] & 0xF0) == 0x10 {
            return true;
        }
        // 192.168.0.0/16
        if ip[0] == 192 && ip[1] == 168 {
            return true;
        }
        false
    } else if addr[0] == 6 {
        // fc00::/7 (Unique Local Address)
        (addr[1] & 0xFE) == 0xFC
    } else {
        false
    }
}

/// Returns true if the address is a loopback address.
pub fn inet_is_loopback(addr: &[u8; 18]) -> bool {
    if addr[0] == 4 {
        addr[13] == 127
    } else if addr[0] == 6 {
        // ::1
        addr[1..16].iter().all(|&b| b == 0) && addr[16] == 1
    } else {
        false
    }
}

// ---------------------------------------------------------------------------
// MAC address
// ---------------------------------------------------------------------------

/// Parses a MAC address from various formats: "AA:BB:CC:DD:EE:FF", "AA-BB-CC-DD-EE-FF", "AABB.CCDD.EEFF".
pub fn macaddr_parse(text: &str) -> Result<[u8; 6]> {
    // Strip separators
    let hex: String = text.chars().filter(|c| c.is_ascii_hexdigit()).collect();

    if hex.len() != 12 {
        return Err(ZyronError::ExecutionError(format!(
            "Invalid MAC address: {}",
            text
        )));
    }

    let mut result = [0u8; 6];
    for i in 0..6 {
        result[i] = u8::from_str_radix(&hex[i * 2..i * 2 + 2], 16)
            .map_err(|_| ZyronError::ExecutionError(format!("Invalid MAC: {}", text)))?;
    }
    Ok(result)
}

/// Formats a MAC address with colons: "AA:BB:CC:DD:EE:FF".
pub fn macaddr_format(addr: &[u8; 6]) -> String {
    format!(
        "{:02X}:{:02X}:{:02X}:{:02X}:{:02X}:{:02X}",
        addr[0], addr[1], addr[2], addr[3], addr[4], addr[5]
    )
}

/// Returns the OUI (first 3 bytes) of a MAC address.
pub fn macaddr_oui(addr: &[u8; 6]) -> [u8; 3] {
    [addr[0], addr[1], addr[2]]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ipv4_parse() {
        let addr = inet_parse("192.168.1.1").unwrap();
        assert_eq!(addr[0], 4);
        assert_eq!(&addr[13..17], &[192, 168, 1, 1]);
        assert_eq!(addr[17], 32);
    }

    #[test]
    fn test_ipv4_with_prefix() {
        let addr = inet_parse("10.0.0.0/8").unwrap();
        assert_eq!(addr[17], 8);
    }

    #[test]
    fn test_ipv4_invalid() {
        assert!(inet_parse("999.1.1.1").is_err());
        assert!(inet_parse("1.2.3").is_err());
        assert!(inet_parse("1.2.3.4/33").is_err());
    }

    #[test]
    fn test_ipv4_format() {
        let addr = inet_parse("192.168.1.1").unwrap();
        assert_eq!(inet_format(&addr), "192.168.1.1");
    }

    #[test]
    fn test_ipv4_format_with_prefix() {
        let addr = inet_parse("10.0.0.0/8").unwrap();
        assert_eq!(inet_format(&addr), "10.0.0.0/8");
    }

    #[test]
    fn test_ipv6_parse_full() {
        let addr = inet_parse("2001:db8:85a3:0:0:8a2e:370:7334").unwrap();
        assert_eq!(addr[0], 6);
        assert_eq!(addr[17], 128);
    }

    #[test]
    fn test_ipv6_parse_abbreviated() {
        let addr = inet_parse("::1").unwrap();
        assert_eq!(addr[0], 6);
        assert_eq!(addr[16], 1);
        for i in 1..16 {
            assert_eq!(addr[i], 0);
        }
    }

    #[test]
    fn test_ipv6_with_prefix() {
        let addr = inet_parse("fe80::/64").unwrap();
        assert_eq!(addr[17], 64);
    }

    #[test]
    fn test_ipv6_format() {
        let addr = inet_parse("::1").unwrap();
        let formatted = inet_format(&addr);
        assert_eq!(formatted, "::1");
    }

    #[test]
    fn test_inet_family() {
        let v4 = inet_parse("1.2.3.4").unwrap();
        let v6 = inet_parse("::1").unwrap();
        assert_eq!(inet_family(&v4), 4);
        assert_eq!(inet_family(&v6), 6);
    }

    #[test]
    fn test_inet_contains_v4() {
        let network = inet_parse("192.168.0.0/16").unwrap();
        let addr = inet_parse("192.168.1.100").unwrap();
        assert!(inet_contains(&network, &addr));
    }

    #[test]
    fn test_inet_not_contains() {
        let network = inet_parse("192.168.0.0/16").unwrap();
        let addr = inet_parse("10.0.0.1").unwrap();
        assert!(!inet_contains(&network, &addr));
    }

    #[test]
    fn test_inet_contains_exact() {
        let network = inet_parse("192.168.1.1/32").unwrap();
        let addr = inet_parse("192.168.1.1").unwrap();
        assert!(inet_contains(&network, &addr));
    }

    #[test]
    fn test_inet_contains_cross_family() {
        let v4 = inet_parse("192.168.0.0/16").unwrap();
        let v6 = inet_parse("::1").unwrap();
        assert!(!inet_contains(&v4, &v6));
    }

    #[test]
    fn test_inet_network() {
        let addr = inet_parse("192.168.1.100/16").unwrap();
        let network = inet_network(&addr);
        assert_eq!(&network[13..17], &[192, 168, 0, 0]);
    }

    #[test]
    fn test_inet_broadcast() {
        let addr = inet_parse("192.168.0.0/24").unwrap();
        let bc = inet_broadcast(&addr);
        assert_eq!(&bc[13..17], &[192, 168, 0, 255]);
    }

    #[test]
    fn test_inet_netmask() {
        let addr = inet_parse("10.0.0.0/8").unwrap();
        let mask = inet_netmask(&addr);
        assert_eq!(&mask[13..17], &[255, 0, 0, 0]);
    }

    #[test]
    fn test_inet_is_private_v4() {
        assert!(inet_is_private(&inet_parse("10.0.0.1").unwrap()));
        assert!(inet_is_private(&inet_parse("172.16.0.1").unwrap()));
        assert!(inet_is_private(&inet_parse("192.168.1.1").unwrap()));
        assert!(!inet_is_private(&inet_parse("8.8.8.8").unwrap()));
    }

    #[test]
    fn test_inet_is_loopback() {
        assert!(inet_is_loopback(&inet_parse("127.0.0.1").unwrap()));
        assert!(inet_is_loopback(&inet_parse("::1").unwrap()));
        assert!(!inet_is_loopback(&inet_parse("192.168.1.1").unwrap()));
    }

    #[test]
    fn test_macaddr_parse_colon() {
        let addr = macaddr_parse("AA:BB:CC:DD:EE:FF").unwrap();
        assert_eq!(addr, [0xAA, 0xBB, 0xCC, 0xDD, 0xEE, 0xFF]);
    }

    #[test]
    fn test_macaddr_parse_dash() {
        let addr = macaddr_parse("AA-BB-CC-DD-EE-FF").unwrap();
        assert_eq!(addr, [0xAA, 0xBB, 0xCC, 0xDD, 0xEE, 0xFF]);
    }

    #[test]
    fn test_macaddr_parse_cisco() {
        let addr = macaddr_parse("aabb.ccdd.eeff").unwrap();
        assert_eq!(addr, [0xAA, 0xBB, 0xCC, 0xDD, 0xEE, 0xFF]);
    }

    #[test]
    fn test_macaddr_parse_invalid() {
        assert!(macaddr_parse("AA:BB:CC").is_err());
        assert!(macaddr_parse("ZZ:BB:CC:DD:EE:FF").is_err());
    }

    #[test]
    fn test_macaddr_format() {
        let addr = [0xAA, 0xBB, 0xCC, 0xDD, 0xEE, 0xFF];
        assert_eq!(macaddr_format(&addr), "AA:BB:CC:DD:EE:FF");
    }

    #[test]
    fn test_macaddr_oui() {
        let addr = [0xAA, 0xBB, 0xCC, 0xDD, 0xEE, 0xFF];
        assert_eq!(macaddr_oui(&addr), [0xAA, 0xBB, 0xCC]);
    }

    #[test]
    fn test_ipv4_roundtrip() {
        let text = "172.16.254.1";
        let addr = inet_parse(text).unwrap();
        assert_eq!(inet_format(&addr), text);
    }
}
