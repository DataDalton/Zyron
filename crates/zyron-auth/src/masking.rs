//! Column-level data masking at the privilege layer.
//!
//! Masking rules transform sensitive column values before returning them
//! to users who lack full access. Each rule specifies a mask function
//! (email, phone, SSN, credit card, partial, hash, null, redact, or custom)
//! and can be scoped to a specific role or applied globally.

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use zyron_common::{Result, ZyronError};

use crate::role::RoleId;

/// Identifies the masking transformation to apply to column values.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
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
}

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

    /// Serializes the mask function to bytes.
    /// Format: 1-byte tag, followed by variant-specific data.
    fn to_bytes(&self) -> Vec<u8> {
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
        }
    }

    /// Deserializes a mask function from a byte slice. Returns the parsed
    /// function and the number of bytes consumed.
    fn from_bytes(data: &[u8]) -> Result<(Self, usize)> {
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
            let first_char = &value[..1];
            let domain = &value[at_pos..];
            let mask_len = at_pos - 1;
            let mut result = String::with_capacity(value.len());
            result.push_str(first_char);
            for _ in 0..mask_len {
                result.push('*');
            }
            result.push_str(domain);
            result
        }
        _ => {
            // No @ or empty local part, mask entire value.
            "*".repeat(value.len())
        }
    }
}

/// Masks a phone number by showing the first 6 characters and replacing
/// the rest with asterisks.
pub fn apply_phone_mask(value: &str) -> String {
    if value.len() <= 6 {
        return value.to_string();
    }
    let visible = &value[..6];
    let mask_len = value.len() - 6;
    let mut result = String::with_capacity(value.len());
    result.push_str(visible);
    for _ in 0..mask_len {
        result.push('*');
    }
    result
}

/// Masks a Social Security Number by hiding everything except the last 4 characters.
/// Example: "123-45-6789" -> "***-**-6789"
pub fn apply_ssn_mask(value: &str) -> String {
    if value.len() <= 4 {
        return value.to_string();
    }
    let mask_len = value.len() - 4;
    let last_four = &value[mask_len..];
    let mut result = String::with_capacity(value.len());
    for ch in value[..mask_len].chars() {
        if ch == '-' {
            result.push('-');
        } else {
            result.push('*');
        }
    }
    result.push_str(last_four);
    result
}

/// Masks a credit card number by hiding all but the last 4 digits.
/// Non-digit characters (spaces, dashes) are preserved.
pub fn apply_credit_card_mask(value: &str) -> String {
    let digit_count = value.chars().filter(|c| c.is_ascii_digit()).count();
    if digit_count <= 4 {
        return value.to_string();
    }
    let digits_to_mask = digit_count - 4;
    let mut masked_count = 0;
    let mut result = String::with_capacity(value.len());
    for ch in value.chars() {
        if ch.is_ascii_digit() {
            if masked_count < digits_to_mask {
                result.push('*');
                masked_count += 1;
            } else {
                result.push(ch);
            }
        } else {
            result.push(ch);
        }
    }
    result
}

/// Shows the first N characters and masks the rest with asterisks.
pub fn apply_partial_mask(value: &str, show_chars: u8) -> String {
    let show = show_chars as usize;
    if value.len() <= show {
        return value.to_string();
    }
    let visible = &value[..show];
    let mask_len = value.len() - show;
    let mut result = String::with_capacity(value.len());
    result.push_str(visible);
    for _ in 0..mask_len {
        result.push('*');
    }
    result
}

/// Replaces the value with its SHA-256 hex digest.
/// Deterministic output allows masked columns to be used in JOINs.
pub fn apply_hash_mask(value: &str) -> String {
    let mut hasher = Sha256::new();
    hasher.update(value.as_bytes());
    let hash = hasher.finalize();
    let mut hex = String::with_capacity(64);
    for byte in hash.iter() {
        hex.push_str(&format!("{:02x}", byte));
    }
    hex
}

/// Dispatches to the appropriate masking function based on the rule.
/// Returns None for Null mask (represents SQL NULL in the result row).
pub fn apply_mask(value: &str, function: &MaskFunction) -> Option<String> {
    match function {
        MaskFunction::Email => Some(apply_email_mask(value)),
        MaskFunction::Phone => Some(apply_phone_mask(value)),
        MaskFunction::Ssn => Some(apply_ssn_mask(value)),
        MaskFunction::CreditCard => Some(apply_credit_card_mask(value)),
        MaskFunction::Partial(n) => Some(apply_partial_mask(value, *n)),
        MaskFunction::Hash => Some(apply_hash_mask(value)),
        MaskFunction::Null => None,
        MaskFunction::Redact => Some("[REDACTED]".to_string()),
        MaskFunction::Custom(_) => Some(value.to_string()),
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
        let h1 = apply_hash_mask("test_value");
        let h2 = apply_hash_mask("test_value");
        assert_eq!(h1, h2);
        assert_eq!(h1.len(), 64);
    }

    #[test]
    fn test_hash_mask_different_inputs() {
        let h1 = apply_hash_mask("value_a");
        let h2 = apply_hash_mask("value_b");
        assert_ne!(h1, h2);
    }

    // -- apply_mask dispatch tests --

    #[test]
    fn test_apply_mask_null() {
        assert_eq!(apply_mask("anything", &MaskFunction::Null), None);
    }

    #[test]
    fn test_apply_mask_redact() {
        assert_eq!(
            apply_mask("anything", &MaskFunction::Redact),
            Some("[REDACTED]".to_string())
        );
    }

    #[test]
    fn test_apply_mask_custom_passthrough() {
        assert_eq!(
            apply_mask("data", &MaskFunction::Custom("my_fn".to_string())),
            Some("data".to_string())
        );
    }

    #[test]
    fn test_apply_mask_dispatches_email() {
        assert_eq!(
            apply_mask("john@example.com", &MaskFunction::Email),
            Some("j***@example.com".to_string())
        );
    }

    #[test]
    fn test_apply_mask_dispatches_hash() {
        let result = apply_mask("test", &MaskFunction::Hash).expect("hash mask returns Some");
        assert_eq!(result.len(), 64);
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
}
