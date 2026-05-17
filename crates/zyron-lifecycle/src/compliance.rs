//! Regulatory retention validation (GDPR / HIPAA / SOX) and compliance
//! event typing.

use zyron_common::{Result, ZyronError};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Regulation {
    Gdpr,
    Hipaa,
    Sox,
}

/// A min/max retention bound for a data category under a regulation. Periods
/// are in seconds; `max_retention == 0` means unbounded.
#[derive(Debug, Clone)]
pub struct RetentionRequirement {
    pub regulation: Regulation,
    pub data_category: String,
    pub min_retention: i64,
    pub max_retention: i64,
}

impl RetentionRequirement {
    /// SOX requires financial records be kept at least 7 years.
    pub fn sox_financial() -> Self {
        Self {
            regulation: Regulation::Sox,
            data_category: "financial".to_string(),
            min_retention: 7 * 365 * 24 * 3600,
            max_retention: 0,
        }
    }

    /// GDPR: PII should not be kept longer than necessary; cap at 30 days for
    /// the default "pii" category unless a lawful basis extends it.
    pub fn gdpr_pii() -> Self {
        Self {
            regulation: Regulation::Gdpr,
            data_category: "pii".to_string(),
            min_retention: 0,
            max_retention: 30 * 24 * 3600,
        }
    }
}

/// Validates a proposed retention period (seconds) for a data category
/// against the configured requirements. Returns RetentionViolation when the
/// period is below a floor or above a ceiling.
pub fn validate_retention(
    data_category: &str,
    proposed_seconds: i64,
    reqs: &[RetentionRequirement],
) -> Result<()> {
    for r in reqs {
        if !r.data_category.eq_ignore_ascii_case(data_category) {
            continue;
        }
        if r.min_retention > 0 && proposed_seconds < r.min_retention {
            return Err(ZyronError::RetentionViolation(format!(
                "{:?}: retention {}s is below the {}s minimum for '{}'",
                r.regulation, proposed_seconds, r.min_retention, data_category
            )));
        }
        if r.max_retention > 0 && proposed_seconds > r.max_retention {
            return Err(ZyronError::RetentionViolation(format!(
                "{:?}: retention {}s exceeds the {}s maximum for '{}'",
                r.regulation, proposed_seconds, r.max_retention, data_category
            )));
        }
    }
    Ok(())
}

/// Compliance event type codes (mirrors ComplianceLogEntry.event_type).
pub mod event {
    pub const TTL: u8 = 0;
    pub const ARCHIVE: u8 = 1;
    pub const RESTORE: u8 = 2;
    pub const LEGAL_HOLD: u8 = 3;
    pub const FORGET_USER: u8 = 4;
    pub const EXPORT_USER: u8 = 5;
    pub const CLASSIFICATION: u8 = 6;
    pub const TIER_MOVE: u8 = 7;
    pub const RETENTION_LOCK: u8 = 8;
    pub const CRYPTO_SHRED: u8 = 9;
    pub const PURGE: u8 = 10;
    pub const UNDROP: u8 = 11;
}
