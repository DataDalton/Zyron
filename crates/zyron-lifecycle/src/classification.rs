//! Column data classification and PII auto-detection.
//!
//! Wraps the Phase 8 lock-free `ClassificationStore`. `SET CLASSIFICATION`
//! drives `apply_column_classification`. Auto-detection uses deterministic
//! rule-based pattern checks (no regex engine, no ML): email shape, digit-run
//! lengths for SSN and credit card, phone shapes.

use std::sync::Arc;

use zyron_auth::{ClassificationLevel, ClassificationStore};
use zyron_common::{Result, ZyronError};

/// Detected PII category for a column sample.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PiiKind {
    Email,
    Ssn,
    CreditCard,
    Phone,
}

impl PiiKind {
    /// The classification level a detected PII column should receive.
    pub fn suggested_level(&self) -> ClassificationLevel {
        match self {
            PiiKind::Email | PiiKind::Phone => ClassificationLevel::Confidential,
            PiiKind::Ssn | PiiKind::CreditCard => ClassificationLevel::Restricted,
        }
    }
}

/// Service wrapping the shared classification store.
pub struct ClassificationService {
    store: ClassificationStore,
}

impl ClassificationService {
    pub fn new() -> Self {
        Self {
            store: ClassificationStore::new(),
        }
    }

    pub fn store(&self) -> &ClassificationStore {
        &self.store
    }

    /// Parses a level string ('public'|'internal'|'confidential'|'restricted')
    /// case-insensitively.
    pub fn parse_level(level: &str) -> Result<ClassificationLevel> {
        match level.to_ascii_lowercase().as_str() {
            "public" => Ok(ClassificationLevel::Public),
            "internal" => Ok(ClassificationLevel::Internal),
            "confidential" => Ok(ClassificationLevel::Confidential),
            "restricted" => Ok(ClassificationLevel::Restricted),
            other => Err(ZyronError::Internal(format!(
                "unknown classification level: {other}"
            ))),
        }
    }

    /// Applies a classification level to a column. Confidential/Restricted
    /// columns become candidates for masking by the project operator.
    pub fn apply_column_classification(
        &self,
        table_id: u32,
        column_id: u16,
        level: &str,
    ) -> Result<ClassificationLevel> {
        let parsed = Self::parse_level(level)?;
        self.store.set_classification(table_id, column_id, parsed);
        Ok(parsed)
    }
}

impl Default for ClassificationService {
    fn default() -> Self {
        Self::new()
    }
}

/// Strips common separators used in formatted identifiers.
fn digits_only(s: &str) -> String {
    s.chars().filter(|c| c.is_ascii_digit()).collect()
}

/// Deterministic single-value PII classification. Rule-based only.
pub fn classify_value(value: &str) -> Option<PiiKind> {
    let v = value.trim();
    if v.is_empty() {
        return None;
    }
    if is_email(v) {
        return Some(PiiKind::Email);
    }
    let digits = digits_only(v);
    // Credit card: 13-19 digits passing the Luhn check.
    if (13..=19).contains(&digits.len()) && luhn_valid(&digits) {
        return Some(PiiKind::CreditCard);
    }
    // US SSN: exactly 9 digits, typically grouped 3-2-4.
    if digits.len() == 9 && looks_like_ssn(v) {
        return Some(PiiKind::Ssn);
    }
    // Phone: 10-15 digits with phone-shaped punctuation.
    if (10..=15).contains(&digits.len()) && looks_like_phone(v) {
        return Some(PiiKind::Phone);
    }
    None
}

/// Classifies a column from a sample of its values. Returns the highest-
/// confidence PII kind when a strict majority of non-empty samples agree.
pub fn auto_classify_column(samples: &[String]) -> Option<PiiKind> {
    let mut email = 0usize;
    let mut ssn = 0usize;
    let mut card = 0usize;
    let mut phone = 0usize;
    let mut non_empty = 0usize;
    for s in samples {
        if s.trim().is_empty() {
            continue;
        }
        non_empty += 1;
        match classify_value(s) {
            Some(PiiKind::Email) => email += 1,
            Some(PiiKind::Ssn) => ssn += 1,
            Some(PiiKind::CreditCard) => card += 1,
            Some(PiiKind::Phone) => phone += 1,
            None => {}
        }
    }
    if non_empty == 0 {
        return None;
    }
    let threshold = non_empty / 2 + 1;
    let mut best: Option<(PiiKind, usize)> = None;
    for (kind, count) in [
        (PiiKind::Email, email),
        (PiiKind::Ssn, ssn),
        (PiiKind::CreditCard, card),
        (PiiKind::Phone, phone),
    ] {
        if count >= threshold && best.map(|(_, b)| count > b).unwrap_or(true) {
            best = Some((kind, count));
        }
    }
    best.map(|(k, _)| k)
}

fn is_email(s: &str) -> bool {
    let at = match s.find('@') {
        Some(i) => i,
        None => return false,
    };
    if at == 0 || at == s.len() - 1 {
        return false;
    }
    let (local, domain) = s.split_at(at);
    let domain = &domain[1..];
    !local.is_empty()
        && domain.contains('.')
        && !domain.starts_with('.')
        && !domain.ends_with('.')
        && !s.contains(' ')
}

fn looks_like_ssn(s: &str) -> bool {
    // Accept "123-45-6789", "123 45 6789", or 9 contiguous digits.
    let cleaned: String = s
        .chars()
        .filter(|c| c.is_ascii_digit() || *c == '-' || *c == ' ')
        .collect();
    let parts: Vec<&str> = cleaned.split(|c| c == '-' || c == ' ').collect();
    if parts.len() == 3 {
        return parts[0].len() == 3 && parts[1].len() == 2 && parts[2].len() == 4;
    }
    digits_only(s).len() == 9 && s.chars().all(|c| c.is_ascii_digit())
}

fn looks_like_phone(s: &str) -> bool {
    s.chars()
        .all(|c| c.is_ascii_digit() || "+-() .".contains(c))
        && (s.contains('-') || s.contains('(') || s.contains('+') || s.contains(' '))
}

/// Luhn checksum used for credit card validation.
fn luhn_valid(digits: &str) -> bool {
    let mut sum = 0u32;
    let mut alt = false;
    for c in digits.chars().rev() {
        let mut d = match c.to_digit(10) {
            Some(d) => d,
            None => return false,
        };
        if alt {
            d *= 2;
            if d > 9 {
                d -= 9;
            }
        }
        sum += d;
        alt = !alt;
    }
    sum % 10 == 0
}

/// Builds a shared classification service handle.
pub fn shared() -> Arc<ClassificationService> {
    Arc::new(ClassificationService::new())
}
