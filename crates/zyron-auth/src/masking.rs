//! Column-level data masking at the privilege layer.
//!
//! Masking rules transform sensitive column values before returning them
//! to users who lack full access. Each rule specifies a mask function
//! (email, phone, SSN, credit card, partial, hash, null, redact, or custom)
//! and can be scoped to a specific role or applied globally.

use serde::{Deserialize, Serialize};
use zyron_common::{Result, ZyronError};

use crate::role::RoleId;

/// Identifies the masking transformation to apply to column values.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MaskFunction {
    /// Masks the local part of an email address, preserving first char and domain.
    Email,
    /// Shows the first 6 characters and masks the rest.
    Phone,
    /// Masks all but the last 4 characters of a Social Security Number.
    Ssn,
    /// Masks all but the last 4 digits of a credit card number.
    CreditCard,
    /// Shows the first N characters, masks the rest.
    Partial(u8),
    /// Replaces the value with its SHA-256 hex digest (deterministic for JOINs).
    Hash,
    /// Replaces the value with an empty string.
    Null,
    /// Replaces the value with "[REDACTED]".
    Redact,
    /// A named custom masking function (resolved at runtime).
    Custom(String),
    /// Adds random noise to numeric values within the given factor (0.0 to 1.0).
    /// For example, factor 0.1 adds up to 10% noise.
    NoiseMask { factor: f64 },
    /// Replaces numeric values with a range bucket based on boundaries.
    /// For example, boundaries [50000.0, 100000.0, 200000.0] maps 75000 to "50000-100000".
    BucketMask { boundaries: Vec<f64> },
    /// Shows the first N and last M characters, masking the middle with mask_char.
    PartialMask {
        show_first: u8,
        show_last: u8,
        mask_char: u8,
    },
}

impl PartialEq for MaskFunction {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (MaskFunction::Email, MaskFunction::Email)
            | (MaskFunction::Phone, MaskFunction::Phone)
            | (MaskFunction::Ssn, MaskFunction::Ssn)
            | (MaskFunction::CreditCard, MaskFunction::CreditCard)
            | (MaskFunction::Hash, MaskFunction::Hash)
            | (MaskFunction::Null, MaskFunction::Null)
            | (MaskFunction::Redact, MaskFunction::Redact) => true,
            (MaskFunction::Partial(a), MaskFunction::Partial(b)) => a == b,
            (MaskFunction::Custom(a), MaskFunction::Custom(b)) => a == b,
            (MaskFunction::NoiseMask { factor: a }, MaskFunction::NoiseMask { factor: b }) => {
                a.to_bits() == b.to_bits()
            }
            (
                MaskFunction::BucketMask { boundaries: a },
                MaskFunction::BucketMask { boundaries: b },
            ) => {
                a.len() == b.len()
                    && a.iter()
                        .zip(b.iter())
                        .all(|(x, y)| x.to_bits() == y.to_bits())
            }
            (
                MaskFunction::PartialMask {
                    show_first: sf1,
                    show_last: sl1,
                    mask_char: mc1,
                },
                MaskFunction::PartialMask {
                    show_first: sf2,
                    show_last: sl2,
                    mask_char: mc2,
                },
            ) => sf1 == sf2 && sl1 == sl2 && mc1 == mc2,
            _ => false,
        }
    }
}

impl Eq for MaskFunction {}

impl MaskFunction {
    // Tag bytes for binary serialization of each variant.
    const TAG_EMAIL: u8 = 0;
    const TAG_PHONE: u8 = 1;
    const TAG_SSN: u8 = 2;
    const TAG_CREDIT_CARD: u8 = 3;
    const TAG_PARTIAL: u8 = 4;
    const TAG_HASH: u8 = 5;
    const TAG_NULL: u8 = 6;
    const TAG_REDACT: u8 = 7;
    const TAG_CUSTOM: u8 = 8;
    const TAG_NOISE_MASK: u8 = 9;
    const TAG_BUCKET_MASK: u8 = 10;
    const TAG_PARTIAL_MASK: u8 = 11;

    /// Serializes the mask function to bytes.
    /// Format: 1-byte tag, followed by variant-specific data.
    pub fn to_bytes(&self) -> Vec<u8> {
        match self {
            MaskFunction::Email => vec![Self::TAG_EMAIL],
            MaskFunction::Phone => vec![Self::TAG_PHONE],
            MaskFunction::Ssn => vec![Self::TAG_SSN],
            MaskFunction::CreditCard => vec![Self::TAG_CREDIT_CARD],
            MaskFunction::Partial(n) => vec![Self::TAG_PARTIAL, *n],
            MaskFunction::Hash => vec![Self::TAG_HASH],
            MaskFunction::Null => vec![Self::TAG_NULL],
            MaskFunction::Redact => vec![Self::TAG_REDACT],
            MaskFunction::Custom(name) => {
                let name_bytes = name.as_bytes();
                let mut buf = Vec::with_capacity(1 + 2 + name_bytes.len());
                buf.push(Self::TAG_CUSTOM);
                buf.extend_from_slice(&(name_bytes.len() as u16).to_le_bytes());
                buf.extend_from_slice(name_bytes);
                buf
            }
            MaskFunction::NoiseMask { factor } => {
                let mut buf = Vec::with_capacity(9);
                buf.push(Self::TAG_NOISE_MASK);
                buf.extend_from_slice(&factor.to_le_bytes());
                buf
            }
            MaskFunction::BucketMask { boundaries } => {
                let mut buf = Vec::with_capacity(1 + 4 + boundaries.len() * 8);
                buf.push(Self::TAG_BUCKET_MASK);
                buf.extend_from_slice(&(boundaries.len() as u32).to_le_bytes());
                for b in boundaries {
                    buf.extend_from_slice(&b.to_le_bytes());
                }
                buf
            }
            MaskFunction::PartialMask {
                show_first,
                show_last,
                mask_char,
            } => {
                vec![Self::TAG_PARTIAL_MASK, *show_first, *show_last, *mask_char]
            }
        }
    }

    /// Deserializes a mask function from a byte slice. Returns the parsed
    /// function and the number of bytes consumed.
    pub fn from_bytes(data: &[u8]) -> Result<(Self, usize)> {
        if data.is_empty() {
            return Err(ZyronError::DecodingFailed(
                "MaskFunction requires at least 1 byte".to_string(),
            ));
        }
        match data[0] {
            Self::TAG_EMAIL => Ok((MaskFunction::Email, 1)),
            Self::TAG_PHONE => Ok((MaskFunction::Phone, 1)),
            Self::TAG_SSN => Ok((MaskFunction::Ssn, 1)),
            Self::TAG_CREDIT_CARD => Ok((MaskFunction::CreditCard, 1)),
            Self::TAG_PARTIAL => {
                if data.len() < 2 {
                    return Err(ZyronError::DecodingFailed(
                        "MaskFunction::Partial requires 2 bytes".to_string(),
                    ));
                }
                Ok((MaskFunction::Partial(data[1]), 2))
            }
            Self::TAG_HASH => Ok((MaskFunction::Hash, 1)),
            Self::TAG_NULL => Ok((MaskFunction::Null, 1)),
            Self::TAG_REDACT => Ok((MaskFunction::Redact, 1)),
            Self::TAG_CUSTOM => {
                if data.len() < 3 {
                    return Err(ZyronError::DecodingFailed(
                        "MaskFunction::Custom requires at least 3 bytes".to_string(),
                    ));
                }
                let name_len = u16::from_le_bytes([data[1], data[2]]) as usize;
                if data.len() < 3 + name_len {
                    return Err(ZyronError::DecodingFailed(format!(
                        "MaskFunction::Custom name requires {} bytes, got {}",
                        name_len,
                        data.len() - 3
                    )));
                }
                let name = std::str::from_utf8(&data[3..3 + name_len])
                    .map_err(|_| {
                        ZyronError::DecodingFailed(
                            "MaskFunction::Custom name is not valid UTF-8".to_string(),
                        )
                    })?
                    .to_string();
                Ok((MaskFunction::Custom(name), 3 + name_len))
            }
            Self::TAG_NOISE_MASK => {
                if data.len() < 9 {
                    return Err(ZyronError::DecodingFailed(
                        "MaskFunction::NoiseMask requires 9 bytes".to_string(),
                    ));
                }
                let factor = f64::from_le_bytes([
                    data[1], data[2], data[3], data[4], data[5], data[6], data[7], data[8],
                ]);
                Ok((MaskFunction::NoiseMask { factor }, 9))
            }
            Self::TAG_BUCKET_MASK => {
                if data.len() < 5 {
                    return Err(ZyronError::DecodingFailed(
                        "MaskFunction::BucketMask requires at least 5 bytes".to_string(),
                    ));
                }
                let count = u32::from_le_bytes([data[1], data[2], data[3], data[4]]) as usize;
                let needed = 5 + count * 8;
                if data.len() < needed {
                    return Err(ZyronError::DecodingFailed(format!(
                        "MaskFunction::BucketMask requires {} bytes, got {}",
                        needed,
                        data.len()
                    )));
                }
                let mut boundaries = Vec::with_capacity(count);
                let mut pos = 5;
                for _ in 0..count {
                    let val = f64::from_le_bytes([
                        data[pos],
                        data[pos + 1],
                        data[pos + 2],
                        data[pos + 3],
                        data[pos + 4],
                        data[pos + 5],
                        data[pos + 6],
                        data[pos + 7],
                    ]);
                    boundaries.push(val);
                    pos += 8;
                }
                Ok((MaskFunction::BucketMask { boundaries }, needed))
            }
            Self::TAG_PARTIAL_MASK => {
                if data.len() < 4 {
                    return Err(ZyronError::DecodingFailed(
                        "MaskFunction::PartialMask requires 4 bytes".to_string(),
                    ));
                }
                Ok((
                    MaskFunction::PartialMask {
                        show_first: data[1],
                        show_last: data[2],
                        mask_char: data[3],
                    },
                    4,
                ))
            }
            tag => Err(ZyronError::DecodingFailed(format!(
                "Unknown MaskFunction tag: {}",
                tag
            ))),
        }
    }
}

/// A masking rule binding a mask function to a specific column,
/// optionally scoped to a particular role.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MaskingRule {
    pub table_id: u32,
    pub column_id: u16,
    /// If None, the rule applies to all roles without explicit access.
    pub role_id: Option<RoleId>,
    pub function: MaskFunction,
}

impl MaskingRule {
    /// Serializes this masking rule to bytes.
    /// Layout: table_id (4 LE) + column_id (2 LE) + has_role (1) + [role_id (4 LE)] + function bytes.
    pub fn to_bytes(&self) -> Vec<u8> {
        let func_bytes = self.function.to_bytes();
        let role_size = if self.role_id.is_some() { 4 } else { 0 };
        let mut buf = Vec::with_capacity(4 + 2 + 1 + role_size + func_bytes.len());
        buf.extend_from_slice(&self.table_id.to_le_bytes());
        buf.extend_from_slice(&self.column_id.to_le_bytes());
        match self.role_id {
            Some(rid) => {
                buf.push(1);
                buf.extend_from_slice(&rid.0.to_le_bytes());
            }
            None => {
                buf.push(0);
            }
        }
        buf.extend_from_slice(&func_bytes);
        buf
    }

    /// Deserializes a masking rule from a byte slice.
    pub fn from_bytes(data: &[u8]) -> Result<Self> {
        if data.len() < 7 {
            return Err(ZyronError::DecodingFailed(
                "MaskingRule requires at least 7 bytes".to_string(),
            ));
        }
        let table_id = u32::from_le_bytes([data[0], data[1], data[2], data[3]]);
        let column_id = u16::from_le_bytes([data[4], data[5]]);
        let has_role = data[6];
        let (role_id, offset) = if has_role == 1 {
            if data.len() < 11 {
                return Err(ZyronError::DecodingFailed(
                    "MaskingRule with role requires at least 11 bytes".to_string(),
                ));
            }
            let rid = u32::from_le_bytes([data[7], data[8], data[9], data[10]]);
            (Some(RoleId(rid)), 11)
        } else {
            (None, 7)
        };
        let (function, _) = MaskFunction::from_bytes(&data[offset..])?;
        Ok(Self {
            table_id,
            column_id,
            role_id,
            function,
        })
    }
}

/// Masks an email address by showing the first character of the local part,
/// replacing the rest with asterisks, and preserving the domain.
/// Example: "john@example.com" -> "j***@example.com"
pub fn apply_email_mask(value: &str) -> String {
    match value.find('@') {
        Some(at_pos) if at_pos > 0 => {
            let mut chars = value[..at_pos].chars();
            let first_char = chars.next().unwrap_or('*');
            let local_rest_count = chars.count();
            let domain = &value[at_pos..];
            let mut result = String::with_capacity(value.len());
            result.push(first_char);
            for _ in 0..local_rest_count {
                result.push('*');
            }
            result.push_str(domain);
            result
        }
        _ => {
            // No @ or empty local part, mask entire value.
            "*".repeat(value.chars().count())
        }
    }
}

/// Masks a phone number by showing the first 6 characters and replacing
/// the rest with asterisks. Non-ASCII input is returned unchanged.
pub fn apply_phone_mask(value: &str) -> String {
    if !value.is_ascii() {
        return value.to_string();
    }
    let bytes = value.as_bytes();
    let len = bytes.len();
    if len <= 6 {
        return value.to_string();
    }
    let mut result = String::with_capacity(len);
    unsafe {
        let out = result.as_mut_vec();
        out.set_len(len);
        let ptr = out.as_mut_ptr();
        for i in 0..len {
            *ptr.add(i) = if i < 6 { bytes[i] } else { b'*' };
        }
    }
    result
}

/// Masks a Social Security Number by hiding everything except the last 4 characters.
/// Example: "123-45-6789" -> "***-**-6789". Non-ASCII input is returned unchanged.
pub fn apply_ssn_mask(value: &str) -> String {
    if !value.is_ascii() {
        return value.to_string();
    }
    let bytes = value.as_bytes();
    let len = bytes.len();
    if len <= 4 {
        return value.to_string();
    }
    let mask_count = len - 4;
    let mut result = String::with_capacity(len);
    unsafe {
        let out = result.as_mut_vec();
        out.set_len(len);
        let ptr = out.as_mut_ptr();
        for i in 0..len {
            *ptr.add(i) = if i < mask_count {
                if bytes[i] == b'-' { b'-' } else { b'*' }
            } else {
                bytes[i]
            };
        }
    }
    result
}

/// Masks a credit card number by hiding all but the last 4 digits.
/// Non-digit characters (spaces, dashes) are preserved. Non-ASCII input
/// is returned unchanged.
pub fn apply_credit_card_mask(value: &str) -> String {
    if !value.is_ascii() {
        return value.to_string();
    }
    let bytes = value.as_bytes();
    let len = bytes.len();
    let digit_count = bytes.iter().filter(|&&b| b.is_ascii_digit()).count();
    if digit_count <= 4 {
        return value.to_string();
    }
    let digits_to_mask = digit_count - 4;
    let mut masked_count = 0;
    let mut result = String::with_capacity(len);
    unsafe {
        let out = result.as_mut_vec();
        out.set_len(len);
        let ptr = out.as_mut_ptr();
        for i in 0..len {
            *ptr.add(i) = if bytes[i].is_ascii_digit() {
                if masked_count < digits_to_mask {
                    masked_count += 1;
                    b'*'
                } else {
                    bytes[i]
                }
            } else {
                bytes[i]
            };
        }
    }
    result
}

/// Shows the first N characters and masks the rest with asterisks. Non-ASCII
/// input is returned unchanged.
pub fn apply_partial_mask(value: &str, show_chars: u8) -> String {
    if !value.is_ascii() {
        return value.to_string();
    }
    let bytes = value.as_bytes();
    let len = bytes.len();
    let show = show_chars as usize;
    if len <= show {
        return value.to_string();
    }
    let mut result = String::with_capacity(len);
    unsafe {
        let out = result.as_mut_vec();
        out.set_len(len);
        let ptr = out.as_mut_ptr();
        for i in 0..len {
            *ptr.add(i) = if i < show { bytes[i] } else { b'*' };
        }
    }
    result
}

/// Hex encoding lookup table. Avoids per-byte format! allocations.
const HEX_CHARS: &[u8; 16] = b"0123456789abcdef";

/// Replaces the value with its AES-based 256-bit hash hex digest.
/// Uses VAES (x86_64) or AESE (aarch64) for hardware-accelerated hashing.
/// Deterministic output allows masked columns to be used in JOINs.
/// Writes into a caller-provided buffer. Zero allocation when buffer
/// capacity >= 64 bytes.
pub fn apply_hash_mask(value: &str, buf: &mut String) {
    let hash = crate::encryption::aes_hash_256(value.as_bytes());
    buf.clear();
    buf.reserve(64);
    unsafe {
        let out = buf.as_mut_vec();
        out.set_len(64);
        let ptr = out.as_mut_ptr();
        for (i, &b) in hash.iter().enumerate() {
            *ptr.add(i * 2) = HEX_CHARS[(b >> 4) as usize];
            *ptr.add(i * 2 + 1) = HEX_CHARS[(b & 0x0f) as usize];
        }
    }
}

/// Adds deterministic noise to a numeric value within the given factor.
/// The noise is derived from a SHA-256 hash of the input value, producing
/// consistent output for the same input (prevents statistical recovery
/// through repeated queries). Non-numeric values return "0".
pub fn apply_noise_mask(value: &str, factor: f64) -> String {
    use sha2::{Digest, Sha256};

    let num: f64 = value.parse().unwrap_or(0.0);
    // Derive a deterministic noise value from the input using SHA-256.
    // The hash output is mapped to [-1.0, 1.0) range.
    let mut hasher = Sha256::new();
    hasher.update(value.as_bytes());
    hasher.update(b"noise_mask_seed");
    let hash = hasher.finalize();
    let hash_u64 = u64::from_le_bytes([
        hash[0], hash[1], hash[2], hash[3], hash[4], hash[5], hash[6], hash[7],
    ]);
    let normalized = (hash_u64 as f64 / u64::MAX as f64) * 2.0 - 1.0;
    let noise = normalized * factor * num;
    format!("{}", num + noise)
}

/// Replaces a numeric value with its corresponding range bucket.
/// Boundaries must be sorted. Values below the first boundary return
/// "< first". Values above the last return ">= last".
pub fn apply_bucket_mask(value: &str, boundaries: &[f64]) -> String {
    let num: f64 = value.parse().unwrap_or(0.0);
    if boundaries.is_empty() {
        return value.to_string();
    }
    if num < boundaries[0] {
        return format!("<{}", boundaries[0]);
    }
    for i in 0..boundaries.len() - 1 {
        if num >= boundaries[i] && num < boundaries[i + 1] {
            return format!("{}-{}", boundaries[i], boundaries[i + 1]);
        }
    }
    format!(">={}", boundaries[boundaries.len() - 1])
}

/// Shows the first N and last M characters, masking the middle with mask_char.
/// If the value is shorter than show_first + show_last, returns it unchanged.
/// Non-ASCII input is returned unchanged.
pub fn apply_partial_mask_extended(
    value: &str,
    show_first: u8,
    show_last: u8,
    mask_char: u8,
) -> String {
    if !value.is_ascii() {
        return value.to_string();
    }
    let bytes = value.as_bytes();
    let len = bytes.len();
    let first = show_first as usize;
    let last = show_last as usize;
    if len <= first + last {
        return value.to_string();
    }
    let mut result = String::with_capacity(len);
    unsafe {
        let out = result.as_mut_vec();
        out.set_len(len);
        let ptr = out.as_mut_ptr();
        for i in 0..len {
            *ptr.add(i) = if i < first || i >= len - last {
                bytes[i]
            } else {
                mask_char
            };
        }
    }
    result
}

/// Dispatches to the appropriate masking function based on the rule.
/// Writes the masked value into buf. Returns false for Null mask
/// (represents SQL NULL in the result row).
pub fn apply_mask(value: &str, function: &MaskFunction, buf: &mut String) -> bool {
    match function {
        MaskFunction::Hash => {
            apply_hash_mask(value, buf);
            true
        }
        MaskFunction::Null => false,
        other => {
            buf.clear();
            let masked = match other {
                MaskFunction::Email => apply_email_mask(value),
                MaskFunction::Phone => apply_phone_mask(value),
                MaskFunction::Ssn => apply_ssn_mask(value),
                MaskFunction::CreditCard => apply_credit_card_mask(value),
                MaskFunction::Partial(n) => apply_partial_mask(value, *n),
                MaskFunction::Redact => "[REDACTED]".to_string(),
                MaskFunction::Custom(_) => value.to_string(),
                MaskFunction::NoiseMask { factor } => apply_noise_mask(value, *factor),
                MaskFunction::BucketMask { boundaries } => apply_bucket_mask(value, boundaries),
                MaskFunction::PartialMask {
                    show_first,
                    show_last,
                    mask_char,
                } => apply_partial_mask_extended(value, *show_first, *show_last, *mask_char),
                MaskFunction::Hash | MaskFunction::Null => unreachable!(),
            };
            buf.push_str(&masked);
            true
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // -- MaskFunction serialization tests --

    #[test]
    fn test_mask_function_roundtrip_email() {
        let bytes = MaskFunction::Email.to_bytes();
        let (func, consumed) = MaskFunction::from_bytes(&bytes).expect("decode");
        assert_eq!(func, MaskFunction::Email);
        assert_eq!(consumed, 1);
    }

    #[test]
    fn test_mask_function_roundtrip_partial() {
        let bytes = MaskFunction::Partial(5).to_bytes();
        let (func, consumed) = MaskFunction::from_bytes(&bytes).expect("decode");
        assert_eq!(func, MaskFunction::Partial(5));
        assert_eq!(consumed, 2);
    }

    #[test]
    fn test_mask_function_roundtrip_custom() {
        let bytes = MaskFunction::Custom("my_mask_fn".to_string()).to_bytes();
        let (func, consumed) = MaskFunction::from_bytes(&bytes).expect("decode");
        assert_eq!(func, MaskFunction::Custom("my_mask_fn".to_string()));
        assert_eq!(consumed, 3 + 10);
    }

    #[test]
    fn test_mask_function_from_bytes_empty() {
        assert!(MaskFunction::from_bytes(&[]).is_err());
    }

    #[test]
    fn test_mask_function_from_bytes_unknown_tag() {
        assert!(MaskFunction::from_bytes(&[255]).is_err());
    }

    // -- MaskingRule serialization tests --

    #[test]
    fn test_masking_rule_roundtrip_no_role() {
        let rule = MaskingRule {
            table_id: 42,
            column_id: 3,
            role_id: None,
            function: MaskFunction::Ssn,
        };
        let bytes = rule.to_bytes();
        let decoded = MaskingRule::from_bytes(&bytes).expect("decode");
        assert_eq!(decoded.table_id, 42);
        assert_eq!(decoded.column_id, 3);
        assert_eq!(decoded.role_id, None);
        assert_eq!(decoded.function, MaskFunction::Ssn);
    }

    #[test]
    fn test_masking_rule_roundtrip_with_role() {
        let rule = MaskingRule {
            table_id: 100,
            column_id: 7,
            role_id: Some(RoleId(55)),
            function: MaskFunction::Hash,
        };
        let bytes = rule.to_bytes();
        let decoded = MaskingRule::from_bytes(&bytes).expect("decode");
        assert_eq!(decoded.table_id, 100);
        assert_eq!(decoded.column_id, 7);
        assert_eq!(decoded.role_id, Some(RoleId(55)));
        assert_eq!(decoded.function, MaskFunction::Hash);
    }

    #[test]
    fn test_masking_rule_from_bytes_too_short() {
        assert!(MaskingRule::from_bytes(&[0; 3]).is_err());
    }

    // -- Email masking tests --

    #[test]
    fn test_email_mask_standard() {
        assert_eq!(apply_email_mask("john@example.com"), "j***@example.com");
    }

    #[test]
    fn test_email_mask_single_char_local() {
        assert_eq!(apply_email_mask("a@test.com"), "a@test.com");
    }

    #[test]
    fn test_email_mask_long_local() {
        assert_eq!(
            apply_email_mask("longname@domain.org"),
            "l*******@domain.org"
        );
    }

    #[test]
    fn test_email_mask_no_at_sign() {
        assert_eq!(apply_email_mask("noemail"), "*******");
    }

    // -- Phone masking tests --

    #[test]
    fn test_phone_mask_standard() {
        assert_eq!(apply_phone_mask("+1-555-123-4567"), "+1-555*********");
    }

    #[test]
    fn test_phone_mask_short() {
        assert_eq!(apply_phone_mask("12345"), "12345");
    }

    #[test]
    fn test_phone_mask_exactly_six() {
        assert_eq!(apply_phone_mask("123456"), "123456");
    }

    // -- SSN masking tests --

    #[test]
    fn test_ssn_mask_standard() {
        assert_eq!(apply_ssn_mask("123-45-6789"), "***-**-6789");
    }

    #[test]
    fn test_ssn_mask_no_dashes() {
        assert_eq!(apply_ssn_mask("123456789"), "*****6789");
    }

    #[test]
    fn test_ssn_mask_short() {
        assert_eq!(apply_ssn_mask("1234"), "1234");
    }

    // -- Credit card masking tests --

    #[test]
    fn test_credit_card_mask_standard() {
        assert_eq!(
            apply_credit_card_mask("4111-1111-1111-1111"),
            "****-****-****-1111"
        );
    }

    #[test]
    fn test_credit_card_mask_no_separators() {
        assert_eq!(
            apply_credit_card_mask("4111111111111111"),
            "************1111"
        );
    }

    #[test]
    fn test_credit_card_mask_short() {
        assert_eq!(apply_credit_card_mask("1234"), "1234");
    }

    #[test]
    fn test_credit_card_mask_with_spaces() {
        assert_eq!(
            apply_credit_card_mask("4111 1111 1111 1111"),
            "**** **** **** 1111"
        );
    }

    // -- Partial masking tests --

    #[test]
    fn test_partial_mask_normal() {
        assert_eq!(apply_partial_mask("secret_data", 3), "sec********");
    }

    #[test]
    fn test_partial_mask_show_zero() {
        assert_eq!(apply_partial_mask("hello", 0), "*****");
    }

    #[test]
    fn test_partial_mask_show_all() {
        assert_eq!(apply_partial_mask("hi", 5), "hi");
    }

    // -- Hash masking tests --

    #[test]
    fn test_hash_mask_deterministic() {
        let mut h1 = String::new();
        let mut h2 = String::new();
        apply_hash_mask("test_value", &mut h1);
        apply_hash_mask("test_value", &mut h2);
        assert_eq!(h1, h2);
        assert_eq!(h1.len(), 64);
    }

    #[test]
    fn test_hash_mask_different_inputs() {
        let mut h1 = String::new();
        let mut h2 = String::new();
        apply_hash_mask("value_a", &mut h1);
        apply_hash_mask("value_b", &mut h2);
        assert_ne!(h1, h2);
    }

    // -- apply_mask dispatch tests --

    #[test]
    fn test_apply_mask_null() {
        let mut buf = String::new();
        assert!(!apply_mask("anything", &MaskFunction::Null, &mut buf));
    }

    #[test]
    fn test_apply_mask_redact() {
        let mut buf = String::new();
        assert!(apply_mask("anything", &MaskFunction::Redact, &mut buf));
        assert_eq!(buf, "[REDACTED]");
    }

    #[test]
    fn test_apply_mask_custom_passthrough() {
        let mut buf = String::new();
        assert!(apply_mask(
            "data",
            &MaskFunction::Custom("my_fn".to_string()),
            &mut buf
        ));
        assert_eq!(buf, "data");
    }

    #[test]
    fn test_apply_mask_dispatches_email() {
        let mut buf = String::new();
        assert!(apply_mask(
            "john@example.com",
            &MaskFunction::Email,
            &mut buf
        ));
        assert_eq!(buf, "j***@example.com");
    }

    #[test]
    fn test_apply_mask_dispatches_hash() {
        let mut buf = String::new();
        assert!(apply_mask("test", &MaskFunction::Hash, &mut buf));
        assert_eq!(buf.len(), 64);
    }

    // -- MaskingRule with all function variants --

    #[test]
    fn test_masking_rule_roundtrip_all_variants() {
        let variants = vec![
            MaskFunction::Email,
            MaskFunction::Phone,
            MaskFunction::Ssn,
            MaskFunction::CreditCard,
            MaskFunction::Partial(10),
            MaskFunction::Hash,
            MaskFunction::Null,
            MaskFunction::Redact,
            MaskFunction::Custom("test_fn".to_string()),
            MaskFunction::NoiseMask { factor: 0.15 },
            MaskFunction::BucketMask {
                boundaries: vec![50000.0, 100000.0, 200000.0],
            },
            MaskFunction::PartialMask {
                show_first: 3,
                show_last: 4,
                mask_char: b'X',
            },
        ];
        for (i, func) in variants.into_iter().enumerate() {
            let rule = MaskingRule {
                table_id: i as u32,
                column_id: 0,
                role_id: None,
                function: func.clone(),
            };
            let bytes = rule.to_bytes();
            let decoded = MaskingRule::from_bytes(&bytes).expect("decode");
            assert_eq!(decoded.function, func);
        }
    }

    // -- NoiseMask tests --

    #[test]
    fn test_noise_mask_roundtrip() {
        let func = MaskFunction::NoiseMask { factor: 0.25 };
        let bytes = func.to_bytes();
        let (decoded, consumed) = MaskFunction::from_bytes(&bytes).expect("decode");
        assert_eq!(decoded, func);
        assert_eq!(consumed, 9);
    }

    #[test]
    fn test_noise_mask_apply() {
        let result = apply_noise_mask("100.0", 0.1);
        let val: f64 = result.parse().expect("should be numeric");
        assert!(val >= 90.0 && val <= 110.0, "noise out of range: {}", val);
    }

    #[test]
    fn test_noise_mask_non_numeric() {
        let result = apply_noise_mask("not_a_number", 0.1);
        let val: f64 = result.parse().expect("should be numeric");
        assert!(val.abs() < 0.001);
    }

    // -- BucketMask tests --

    #[test]
    fn test_bucket_mask_roundtrip() {
        let func = MaskFunction::BucketMask {
            boundaries: vec![100.0, 200.0, 300.0],
        };
        let bytes = func.to_bytes();
        let (decoded, consumed) = MaskFunction::from_bytes(&bytes).expect("decode");
        assert_eq!(decoded, func);
        assert_eq!(consumed, 5 + 3 * 8);
    }

    #[test]
    fn test_bucket_mask_below_first() {
        assert_eq!(apply_bucket_mask("25000", &[50000.0, 100000.0]), "<50000");
    }

    #[test]
    fn test_bucket_mask_in_range() {
        assert_eq!(
            apply_bucket_mask("75000", &[50000.0, 100000.0, 200000.0]),
            "50000-100000"
        );
    }

    #[test]
    fn test_bucket_mask_above_last() {
        assert_eq!(
            apply_bucket_mask("300000", &[50000.0, 100000.0, 200000.0]),
            ">=200000"
        );
    }

    #[test]
    fn test_bucket_mask_empty_boundaries() {
        assert_eq!(apply_bucket_mask("100", &[]), "100");
    }

    // -- PartialMask tests --

    #[test]
    fn test_partial_mask_extended_roundtrip() {
        let func = MaskFunction::PartialMask {
            show_first: 3,
            show_last: 4,
            mask_char: b'X',
        };
        let bytes = func.to_bytes();
        let (decoded, consumed) = MaskFunction::from_bytes(&bytes).expect("decode");
        assert_eq!(decoded, func);
        assert_eq!(consumed, 4);
    }

    #[test]
    fn test_partial_mask_extended_apply() {
        assert_eq!(
            apply_partial_mask_extended("123-45-6789", 0, 4, b'*'),
            "*******6789"
        );
    }

    #[test]
    fn test_partial_mask_extended_both_ends() {
        assert_eq!(
            apply_partial_mask_extended("1234567890", 3, 2, b'X'),
            "123XXXXX90"
        );
    }

    #[test]
    fn test_partial_mask_extended_short_value() {
        assert_eq!(apply_partial_mask_extended("hi", 3, 4, b'*'), "hi");
    }

    #[test]
    fn test_apply_mask_dispatches_noise() {
        let mut buf = String::new();
        assert!(apply_mask(
            "100.0",
            &MaskFunction::NoiseMask { factor: 0.1 },
            &mut buf
        ));
        let val: f64 = buf.parse().expect("numeric");
        assert!(val >= 90.0 && val <= 110.0);
    }

    #[test]
    fn test_apply_mask_dispatches_bucket() {
        let mut buf = String::new();
        assert!(apply_mask(
            "75000",
            &MaskFunction::BucketMask {
                boundaries: vec![50000.0, 100000.0],
            },
            &mut buf,
        ));
        assert_eq!(buf, "50000-100000");
    }

    #[test]
    fn test_apply_mask_dispatches_partial_mask() {
        let mut buf = String::new();
        assert!(apply_mask(
            "1234567890",
            &MaskFunction::PartialMask {
                show_first: 2,
                show_last: 3,
                mask_char: b'#',
            },
            &mut buf,
        ));
        assert_eq!(buf, "12#####890");
    }
}
