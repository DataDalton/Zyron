//! PII anonymization functions, the `zyron_sys.security.masking_*` family.
//!
//! Every function is deterministic so the same input always masks to the
//! same output, which keeps masked columns joinable, and non reversible
//! where hashing is involved.

use std::net::{IpAddr, Ipv4Addr, Ipv6Addr};

use sha2::{Digest, Sha256};
use zyron_common::{Result, ZyronError};

fn short_hash(input: &str, hex_chars: usize) -> String {
    let digest = Sha256::digest(input.as_bytes());
    let mut out = String::with_capacity(hex_chars);
    for byte in digest.iter() {
        if out.len() >= hex_chars {
            break;
        }
        out.push_str(&format!("{byte:02x}"));
    }
    out.truncate(hex_chars);
    out
}

/// Zeros the host portion of an IP address, keeping the network prefix.
/// IPv4 defaults pair with keep_prefix_bits 24, IPv6 accepts 0..=128
pub fn masking_ip(ip: &str, keep_prefix_bits: u32) -> Result<String> {
    let parsed: IpAddr = ip
        .trim()
        .parse()
        .map_err(|_| ZyronError::InvalidParameter {
            name: "ip".to_string(),
            value: ip.to_string(),
        })?;
    match parsed {
        IpAddr::V4(v4) => {
            if keep_prefix_bits > 32 {
                return Err(ZyronError::InvalidParameter {
                    name: "keep_prefix_bits".to_string(),
                    value: keep_prefix_bits.to_string(),
                });
            }
            let raw = u32::from(v4);
            let mask = if keep_prefix_bits == 0 {
                0
            } else {
                u32::MAX << (32 - keep_prefix_bits)
            };
            Ok(Ipv4Addr::from(raw & mask).to_string())
        }
        IpAddr::V6(v6) => {
            if keep_prefix_bits > 128 {
                return Err(ZyronError::InvalidParameter {
                    name: "keep_prefix_bits".to_string(),
                    value: keep_prefix_bits.to_string(),
                });
            }
            let raw = u128::from(v6);
            let mask = if keep_prefix_bits == 0 {
                0
            } else {
                u128::MAX << (128 - keep_prefix_bits)
            };
            Ok(Ipv6Addr::from(raw & mask).to_string())
        }
    }
}

/// Hashes the local part of an address and keeps the domain, so the same
/// sender masks identically without exposing who it was
pub fn masking_email(email: &str) -> Result<String> {
    let trimmed = email.trim();
    let Some((local, domain)) = trimmed.rsplit_once('@') else {
        return Err(ZyronError::InvalidParameter {
            name: "email".to_string(),
            value: email.to_string(),
        });
    };
    if local.is_empty() || domain.is_empty() {
        return Err(ZyronError::InvalidParameter {
            name: "email".to_string(),
            value: email.to_string(),
        });
    }
    Ok(format!("{}@{domain}", short_hash(local, 12)))
}

/// Replaces every digit with an asterisk, optionally keeping a leading
/// country code of up to three digits after a plus sign. Separators and
/// spacing survive so the shape of the number stays recognizable
pub fn masking_phone(phone: &str, keep_country_code: bool) -> Result<String> {
    if !phone.chars().any(|c| c.is_ascii_digit()) {
        return Err(ZyronError::InvalidParameter {
            name: "phone".to_string(),
            value: phone.to_string(),
        });
    }
    let trimmed = phone.trim();
    let mut out = String::with_capacity(trimmed.len());
    let mut country_digits_left = if keep_country_code && trimmed.starts_with('+') {
        3
    } else {
        0
    };
    let mut in_country_code = country_digits_left > 0;
    for c in trimmed.chars() {
        if c.is_ascii_digit() {
            if in_country_code && country_digits_left > 0 {
                out.push(c);
                country_digits_left -= 1;
            } else {
                out.push('*');
            }
        } else {
            // The first separator ends the country code run
            if in_country_code && c != '+' {
                in_country_code = false;
            }
            out.push(c);
        }
    }
    Ok(out)
}

/// Keeps the last four digits and replaces the rest with a short hash of
/// the full number, so equal inputs stay equal without being recoverable
pub fn masking_ssn(ssn: &str) -> Result<String> {
    let digits: Vec<char> = ssn.chars().filter(|c| c.is_ascii_digit()).collect();
    if digits.len() < 4 {
        return Err(ZyronError::InvalidParameter {
            name: "ssn".to_string(),
            value: ssn.to_string(),
        });
    }
    let last4: String = digits[digits.len() - 4..].iter().collect();
    Ok(format!("{}-{last4}", short_hash(ssn.trim(), 6)))
}

/// Reduces a personal name to its initials
pub fn masking_name(name: &str) -> Result<String> {
    let mut initials = String::new();
    for word in name.split_whitespace() {
        if let Some(first) = word.chars().find(|c| c.is_alphabetic()) {
            for upper in first.to_uppercase() {
                initials.push(upper);
            }
            initials.push('.');
        }
    }
    if initials.is_empty() {
        return Err(ZyronError::InvalidParameter {
            name: "name".to_string(),
            value: name.to_string(),
        });
    }
    Ok(initials)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_masking_ip_v4_prefix() {
        assert_eq!(
            masking_ip("192.168.1.100", 24).expect("mask"),
            "192.168.1.0"
        );
        assert_eq!(
            masking_ip("192.168.1.100", 16).expect("mask"),
            "192.168.0.0"
        );
        assert_eq!(masking_ip("10.1.2.3", 0).expect("mask"), "0.0.0.0");
        assert!(masking_ip("192.168.1.100", 33).is_err());
        assert!(masking_ip("not an ip", 24).is_err());
    }

    #[test]
    fn test_masking_ip_v6() {
        assert_eq!(
            masking_ip("2001:db8:1:2:3:4:5:6", 32).expect("mask"),
            "2001:db8::"
        );
    }

    #[test]
    fn test_masking_email_keeps_domain_and_determinism() {
        let a = masking_email("alice@example.com").expect("mask");
        let b = masking_email("alice@example.com").expect("mask");
        assert_eq!(a, b);
        assert!(a.ends_with("@example.com"));
        assert!(!a.starts_with("alice@"));
        assert!(masking_email("not-an-email").is_err());
    }

    #[test]
    fn test_masking_phone() {
        assert_eq!(
            masking_phone("+1-555-867-5309", true).expect("mask"),
            "+1-***-***-****"
        );
        assert_eq!(
            masking_phone("+1-555-867-5309", false).expect("mask"),
            "+*-***-***-****"
        );
        assert_eq!(
            masking_phone("5558675309", true).expect("mask"),
            "**********"
        );
        assert!(masking_phone("no digits", true).is_err());
    }

    #[test]
    fn test_masking_ssn_keeps_last_four() {
        let masked = masking_ssn("123-45-6789").expect("mask");
        assert!(masked.ends_with("-6789"));
        assert!(!masked.contains("123"));
        assert_eq!(masked, masking_ssn("123-45-6789").expect("mask"));
        assert!(masking_ssn("12").is_err());
    }

    #[test]
    fn test_masking_name_initials() {
        assert_eq!(masking_name("John Smith").expect("mask"), "J.S.");
        assert_eq!(masking_name("ada").expect("mask"), "A.");
        assert!(masking_name("123").is_err());
    }
}
