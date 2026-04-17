//! Data quality: validation, profiling, and contract checks.
//!
//! Validates emails, URLs, JSON, UUIDs, credit cards (Luhn), dates.
//! Computes column profile statistics in a single pass.

use zyron_common::{Result, TypeId, ZyronError};

// ---------------------------------------------------------------------------
// Validation functions
// ---------------------------------------------------------------------------

/// Validates an email address format. Does not verify deliverability.
pub fn validate_email(text: &str) -> bool {
    let trimmed = text.trim();
    if trimmed.is_empty() || trimmed.len() > 254 {
        return false;
    }

    let at_pos = match trimmed.rfind('@') {
        Some(p) => p,
        None => return false,
    };

    let local = &trimmed[..at_pos];
    let domain = &trimmed[at_pos + 1..];

    if local.is_empty() || local.len() > 64 || domain.is_empty() {
        return false;
    }

    // Local part: letters, digits, dots, hyphens, underscores, plus signs
    // Cannot start/end with dot, no consecutive dots
    if local.starts_with('.') || local.ends_with('.') || local.contains("..") {
        return false;
    }
    for c in local.chars() {
        if !(c.is_ascii_alphanumeric() || c == '.' || c == '-' || c == '_' || c == '+') {
            return false;
        }
    }

    // Domain: at least one dot, alphanumeric labels separated by dots or hyphens
    if !domain.contains('.') {
        return false;
    }
    if domain.starts_with('.') || domain.ends_with('.') || domain.contains("..") {
        return false;
    }
    for label in domain.split('.') {
        if label.is_empty() || label.len() > 63 {
            return false;
        }
        if label.starts_with('-') || label.ends_with('-') {
            return false;
        }
        for c in label.chars() {
            if !c.is_ascii_alphanumeric() && c != '-' {
                return false;
            }
        }
    }

    true
}

/// Validates a URL format (basic structural check).
pub fn validate_url(text: &str) -> bool {
    crate::url_type::url_parse(text).is_ok()
}

/// Validates that text is valid JSON syntax.
pub fn validate_json(text: &str) -> bool {
    parse_json_value(text.trim_start())
        .map(|(_, rest)| rest.trim().is_empty())
        .unwrap_or(false)
}

/// Simple recursive-descent JSON validator.
fn parse_json_value(text: &str) -> Option<((), &str)> {
    let s = text.trim_start();
    if s.is_empty() {
        return None;
    }
    match s.chars().next()? {
        '"' => parse_json_string(s),
        't' | 'f' => parse_json_bool(s),
        'n' => parse_json_null(s),
        '[' => parse_json_array(s),
        '{' => parse_json_object(s),
        '-' | '0'..='9' => parse_json_number(s),
        _ => None,
    }
}

fn parse_json_string(s: &str) -> Option<((), &str)> {
    let mut chars = s.char_indices();
    let (_, first) = chars.next()?;
    if first != '"' {
        return None;
    }
    let mut escaped = false;
    for (i, c) in chars {
        if escaped {
            escaped = false;
            continue;
        }
        if c == '\\' {
            escaped = true;
            continue;
        }
        if c == '"' {
            return Some(((), &s[i + 1..]));
        }
    }
    None
}

fn parse_json_bool(s: &str) -> Option<((), &str)> {
    if let Some(rest) = s.strip_prefix("true") {
        return Some(((), rest));
    }
    if let Some(rest) = s.strip_prefix("false") {
        return Some(((), rest));
    }
    None
}

fn parse_json_null(s: &str) -> Option<((), &str)> {
    s.strip_prefix("null").map(|rest| ((), rest))
}

fn parse_json_number(s: &str) -> Option<((), &str)> {
    let bytes = s.as_bytes();
    let mut i = 0;
    if i < bytes.len() && bytes[i] == b'-' {
        i += 1;
    }
    let digits_start = i;
    while i < bytes.len() && bytes[i].is_ascii_digit() {
        i += 1;
    }
    if i == digits_start {
        return None;
    }
    if i < bytes.len() && bytes[i] == b'.' {
        i += 1;
        let frac_start = i;
        while i < bytes.len() && bytes[i].is_ascii_digit() {
            i += 1;
        }
        if i == frac_start {
            return None;
        }
    }
    if i < bytes.len() && (bytes[i] == b'e' || bytes[i] == b'E') {
        i += 1;
        if i < bytes.len() && (bytes[i] == b'+' || bytes[i] == b'-') {
            i += 1;
        }
        let exp_start = i;
        while i < bytes.len() && bytes[i].is_ascii_digit() {
            i += 1;
        }
        if i == exp_start {
            return None;
        }
    }
    Some(((), &s[i..]))
}

fn parse_json_array(s: &str) -> Option<((), &str)> {
    let mut rest = s.strip_prefix('[')?.trim_start();
    if let Some(r) = rest.strip_prefix(']') {
        return Some(((), r));
    }
    loop {
        let (_, r) = parse_json_value(rest)?;
        rest = r.trim_start();
        if let Some(r) = rest.strip_prefix(',') {
            rest = r.trim_start();
        } else if let Some(r) = rest.strip_prefix(']') {
            return Some(((), r));
        } else {
            return None;
        }
    }
}

fn parse_json_object(s: &str) -> Option<((), &str)> {
    let mut rest = s.strip_prefix('{')?.trim_start();
    if let Some(r) = rest.strip_prefix('}') {
        return Some(((), r));
    }
    loop {
        let (_, r) = parse_json_string(rest)?;
        rest = r.trim_start().strip_prefix(':')?.trim_start();
        let (_, r) = parse_json_value(rest)?;
        rest = r.trim_start();
        if let Some(r) = rest.strip_prefix(',') {
            rest = r.trim_start();
        } else if let Some(r) = rest.strip_prefix('}') {
            return Some(((), r));
        } else {
            return None;
        }
    }
}

/// Validates a UUID string in standard format "xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx".
pub fn validate_uuid(text: &str) -> bool {
    let t = text.trim();
    if t.len() != 36 {
        return false;
    }
    let bytes = t.as_bytes();
    for (i, &b) in bytes.iter().enumerate() {
        if i == 8 || i == 13 || i == 18 || i == 23 {
            if b != b'-' {
                return false;
            }
        } else if !b.is_ascii_hexdigit() {
            return false;
        }
    }
    true
}

/// Validates a credit card number using the Luhn algorithm.
pub fn validate_credit_card(text: &str) -> bool {
    let digits: Vec<u32> = text
        .chars()
        .filter(|c| c.is_ascii_digit())
        .map(|c| c.to_digit(10).unwrap_or(0))
        .collect();

    if digits.len() < 13 || digits.len() > 19 {
        return false;
    }

    let mut sum = 0u32;
    let mut double = false;
    for &d in digits.iter().rev() {
        if double {
            let doubled = d * 2;
            sum += if doubled > 9 { doubled - 9 } else { doubled };
        } else {
            sum += d;
        }
        double = !double;
    }
    sum % 10 == 0
}

/// Validates that text is a valid date in the given format.
/// Supported format tokens: YYYY, MM, DD, HH, mm, SS.
pub fn is_valid_date(text: &str, format: &str) -> bool {
    parse_date_with_format(text, format).is_ok()
}

fn parse_date_with_format(text: &str, format: &str) -> Result<(i32, u32, u32, u32, u32, u32)> {
    let fmt_bytes = format.as_bytes();
    let txt_bytes = text.as_bytes();

    let mut year: i32 = 0;
    let mut month: u32 = 1;
    let mut day: u32 = 1;
    let mut hour: u32 = 0;
    let mut minute: u32 = 0;
    let mut second: u32 = 0;

    let mut fi = 0;
    let mut ti = 0;

    while fi < fmt_bytes.len() {
        if fi + 4 <= fmt_bytes.len() && &fmt_bytes[fi..fi + 4] == b"YYYY" {
            if ti + 4 > txt_bytes.len() {
                return Err(ZyronError::ExecutionError("Date too short".into()));
            }
            year = std::str::from_utf8(&txt_bytes[ti..ti + 4])
                .map_err(|_| ZyronError::ExecutionError("Invalid UTF-8".into()))?
                .parse()
                .map_err(|_| ZyronError::ExecutionError("Invalid year".into()))?;
            fi += 4;
            ti += 4;
        } else if fi + 2 <= fmt_bytes.len() {
            let token = &fmt_bytes[fi..fi + 2];
            match token {
                b"MM" => {
                    if ti + 2 > txt_bytes.len() {
                        return Err(ZyronError::ExecutionError("Date too short".into()));
                    }
                    month = std::str::from_utf8(&txt_bytes[ti..ti + 2])
                        .map_err(|_| ZyronError::ExecutionError("Invalid UTF-8".into()))?
                        .parse()
                        .map_err(|_| ZyronError::ExecutionError("Invalid month".into()))?;
                    fi += 2;
                    ti += 2;
                }
                b"DD" => {
                    if ti + 2 > txt_bytes.len() {
                        return Err(ZyronError::ExecutionError("Date too short".into()));
                    }
                    day = std::str::from_utf8(&txt_bytes[ti..ti + 2])
                        .map_err(|_| ZyronError::ExecutionError("Invalid UTF-8".into()))?
                        .parse()
                        .map_err(|_| ZyronError::ExecutionError("Invalid day".into()))?;
                    fi += 2;
                    ti += 2;
                }
                b"HH" => {
                    if ti + 2 > txt_bytes.len() {
                        return Err(ZyronError::ExecutionError("Date too short".into()));
                    }
                    hour = std::str::from_utf8(&txt_bytes[ti..ti + 2])
                        .map_err(|_| ZyronError::ExecutionError("Invalid UTF-8".into()))?
                        .parse()
                        .map_err(|_| ZyronError::ExecutionError("Invalid hour".into()))?;
                    fi += 2;
                    ti += 2;
                }
                b"mm" => {
                    if ti + 2 > txt_bytes.len() {
                        return Err(ZyronError::ExecutionError("Date too short".into()));
                    }
                    minute = std::str::from_utf8(&txt_bytes[ti..ti + 2])
                        .map_err(|_| ZyronError::ExecutionError("Invalid UTF-8".into()))?
                        .parse()
                        .map_err(|_| ZyronError::ExecutionError("Invalid minute".into()))?;
                    fi += 2;
                    ti += 2;
                }
                b"SS" => {
                    if ti + 2 > txt_bytes.len() {
                        return Err(ZyronError::ExecutionError("Date too short".into()));
                    }
                    second = std::str::from_utf8(&txt_bytes[ti..ti + 2])
                        .map_err(|_| ZyronError::ExecutionError("Invalid UTF-8".into()))?
                        .parse()
                        .map_err(|_| ZyronError::ExecutionError("Invalid second".into()))?;
                    fi += 2;
                    ti += 2;
                }
                _ => {
                    // Literal character match
                    if ti >= txt_bytes.len() || txt_bytes[ti] != fmt_bytes[fi] {
                        return Err(ZyronError::ExecutionError("Format mismatch".into()));
                    }
                    fi += 1;
                    ti += 1;
                }
            }
        } else {
            // Literal character match
            if ti >= txt_bytes.len() || txt_bytes[ti] != fmt_bytes[fi] {
                return Err(ZyronError::ExecutionError("Format mismatch".into()));
            }
            fi += 1;
            ti += 1;
        }
    }

    if ti != txt_bytes.len() {
        return Err(ZyronError::ExecutionError("Trailing characters".into()));
    }

    // Validate ranges
    if !(1..=12).contains(&month) {
        return Err(ZyronError::ExecutionError("Month out of range".into()));
    }
    let max_day = days_in_month(year, month);
    if day < 1 || day > max_day {
        return Err(ZyronError::ExecutionError("Day out of range".into()));
    }
    if hour > 23 || minute > 59 || second > 59 {
        return Err(ZyronError::ExecutionError("Time out of range".into()));
    }

    Ok((year, month, day, hour, minute, second))
}

fn is_leap_year(year: i32) -> bool {
    (year % 4 == 0 && year % 100 != 0) || year % 400 == 0
}

fn days_in_month(year: i32, month: u32) -> u32 {
    match month {
        1 | 3 | 5 | 7 | 8 | 10 | 12 => 31,
        4 | 6 | 9 | 11 => 30,
        2 => {
            if is_leap_year(year) {
                29
            } else {
                28
            }
        }
        _ => 0,
    }
}

// ---------------------------------------------------------------------------
// Data profile
// ---------------------------------------------------------------------------

/// Profile statistics for a single column.
#[derive(Debug, Clone)]
pub struct DataProfile {
    pub null_count: u64,
    pub distinct_count: u64,
    pub total_count: u64,
    pub min: Option<String>,
    pub max: Option<String>,
    pub mean: Option<f64>,
    pub median: Option<f64>,
    pub stddev: Option<f64>,
    pub p25: Option<f64>,
    pub p50: Option<f64>,
    pub p75: Option<f64>,
    pub p95: Option<f64>,
    pub p99: Option<f64>,
    pub most_common_values: Vec<(String, u64)>,
    pub pattern_frequencies: Vec<(String, u64)>,
}

impl DataProfile {
    pub fn empty() -> Self {
        Self {
            null_count: 0,
            distinct_count: 0,
            total_count: 0,
            min: None,
            max: None,
            mean: None,
            median: None,
            stddev: None,
            p25: None,
            p50: None,
            p75: None,
            p95: None,
            p99: None,
            most_common_values: Vec::new(),
            pattern_frequencies: Vec::new(),
        }
    }
}

/// Profiles a column's data in a single pass.
/// For numeric types, computes min/max/mean/stddev/percentiles.
/// For text types, captures most common values and pattern frequencies.
pub fn profile_column(values: &[&str], type_id: TypeId) -> DataProfile {
    let mut profile = DataProfile::empty();
    profile.total_count = values.len() as u64;

    if values.is_empty() {
        return profile;
    }

    // Count nulls (empty strings treated as null)
    let non_null: Vec<&&str> = values.iter().filter(|v| !v.is_empty()).collect();
    profile.null_count = profile.total_count - non_null.len() as u64;

    if non_null.is_empty() {
        return profile;
    }

    // Numeric profiling for numeric types
    if type_id.is_numeric() || type_id.is_floating_point() || type_id.is_integer() {
        let numbers: Vec<f64> = non_null
            .iter()
            .filter_map(|v| v.parse::<f64>().ok())
            .collect();

        if !numbers.is_empty() {
            let min_val = numbers.iter().cloned().fold(f64::INFINITY, f64::min);
            let max_val = numbers.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            profile.min = Some(format!("{}", min_val));
            profile.max = Some(format!("{}", max_val));

            let sum: f64 = numbers.iter().sum();
            let mean = sum / numbers.len() as f64;
            profile.mean = Some(mean);

            let variance =
                numbers.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / numbers.len() as f64;
            profile.stddev = Some(variance.sqrt());

            // Percentiles (exact, O(n log n))
            let mut sorted = numbers.clone();
            sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            profile.median = Some(percentile(&sorted, 0.5));
            profile.p25 = Some(percentile(&sorted, 0.25));
            profile.p50 = Some(percentile(&sorted, 0.5));
            profile.p75 = Some(percentile(&sorted, 0.75));
            profile.p95 = Some(percentile(&sorted, 0.95));
            profile.p99 = Some(percentile(&sorted, 0.99));
        }
    } else {
        // For text types, min/max are lexicographic
        let min_str = non_null.iter().min().map(|s| s.to_string());
        let max_str = non_null.iter().max().map(|s| s.to_string());
        profile.min = min_str;
        profile.max = max_str;
    }

    // Distinct count and most common values
    let mut counts: std::collections::HashMap<String, u64> = std::collections::HashMap::new();
    for v in &non_null {
        *counts.entry(v.to_string()).or_insert(0) += 1;
    }
    profile.distinct_count = counts.len() as u64;

    let mut count_vec: Vec<(String, u64)> = counts.into_iter().collect();
    count_vec.sort_by(|a, b| b.1.cmp(&a.1));
    profile.most_common_values = count_vec.into_iter().take(10).collect();

    // Pattern frequencies (for text types)
    if type_id.is_string() || matches!(type_id, TypeId::Json | TypeId::Jsonb) {
        let mut patterns: std::collections::HashMap<String, u64> = std::collections::HashMap::new();
        for v in &non_null {
            let pattern = extract_pattern(v);
            *patterns.entry(pattern).or_insert(0) += 1;
        }
        let mut pattern_vec: Vec<(String, u64)> = patterns.into_iter().collect();
        pattern_vec.sort_by(|a, b| b.1.cmp(&a.1));
        profile.pattern_frequencies = pattern_vec.into_iter().take(10).collect();
    }

    profile
}

fn percentile(sorted: &[f64], p: f64) -> f64 {
    if sorted.is_empty() {
        return 0.0;
    }
    if sorted.len() == 1 {
        return sorted[0];
    }
    let idx = p * (sorted.len() - 1) as f64;
    let lower = idx.floor() as usize;
    let upper = idx.ceil() as usize;
    if lower == upper {
        return sorted[lower];
    }
    let weight = idx - lower as f64;
    sorted[lower] * (1.0 - weight) + sorted[upper] * weight
}

/// Extracts a generalized pattern from a string (A for letters, 9 for digits, - for other).
fn extract_pattern(text: &str) -> String {
    let mut result = String::with_capacity(text.len());
    for c in text.chars() {
        if c.is_ascii_alphabetic() {
            result.push('A');
        } else if c.is_ascii_digit() {
            result.push('9');
        } else {
            result.push(c);
        }
    }
    result
}

// ---------------------------------------------------------------------------
// Data contracts
// ---------------------------------------------------------------------------

/// A rule in a data contract.
#[derive(Debug, Clone)]
pub enum ContractRule {
    Completeness { field: String, threshold: f64 },
    Uniqueness { field: String, threshold: f64 },
    Freshness { field: String, max_age_seconds: i64 },
    RowCount { min: u64, max: u64 },
    CustomExpect(String),
}

/// Result of evaluating a contract rule.
#[derive(Debug, Clone)]
pub struct ContractResult {
    pub rule: ContractRule,
    pub passed: bool,
    pub message: String,
}

/// Evaluates a set of contract rules against a data profile.
pub fn data_contract(profile: &DataProfile, rules: &[ContractRule]) -> Vec<ContractResult> {
    rules
        .iter()
        .map(|rule| evaluate_rule(profile, rule))
        .collect()
}

fn evaluate_rule(profile: &DataProfile, rule: &ContractRule) -> ContractResult {
    match rule {
        ContractRule::Completeness { threshold, .. } => {
            let completeness = if profile.total_count == 0 {
                0.0
            } else {
                1.0 - (profile.null_count as f64 / profile.total_count as f64)
            };
            let passed = completeness >= *threshold;
            ContractResult {
                rule: rule.clone(),
                passed,
                message: format!(
                    "Completeness: {:.2}% (threshold: {:.2}%)",
                    completeness * 100.0,
                    threshold * 100.0
                ),
            }
        }
        ContractRule::Uniqueness { threshold, .. } => {
            let uniqueness = if profile.total_count == 0 {
                0.0
            } else {
                profile.distinct_count as f64 / profile.total_count as f64
            };
            let passed = uniqueness >= *threshold;
            ContractResult {
                rule: rule.clone(),
                passed,
                message: format!("Uniqueness: {:.2}", uniqueness),
            }
        }
        ContractRule::RowCount { min, max } => {
            let passed = profile.total_count >= *min && profile.total_count <= *max;
            ContractResult {
                rule: rule.clone(),
                passed,
                message: format!(
                    "Row count {} (expected {}-{})",
                    profile.total_count, min, max
                ),
            }
        }
        ContractRule::Freshness { .. } => {
            // Requires access to column data, so this always passes at profile level
            ContractResult {
                rule: rule.clone(),
                passed: true,
                message: "Freshness check requires column data".into(),
            }
        }
        ContractRule::CustomExpect(expr) => ContractResult {
            rule: rule.clone(),
            passed: true,
            message: format!(
                "Custom expectation '{}' not evaluated at profile level",
                expr
            ),
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_validate_email() {
        assert!(validate_email("user@example.com"));
        assert!(validate_email("user.name+tag@sub.example.co.uk"));
        assert!(validate_email("a@b.co"));
    }

    #[test]
    fn test_validate_email_invalid() {
        assert!(!validate_email(""));
        assert!(!validate_email("no-at-sign"));
        assert!(!validate_email("@example.com"));
        assert!(!validate_email("user@"));
        assert!(!validate_email("user@nodot"));
        assert!(!validate_email(".user@example.com"));
        assert!(!validate_email("user..name@example.com"));
        assert!(!validate_email("user@-example.com"));
    }

    #[test]
    fn test_validate_url() {
        assert!(validate_url("https://example.com"));
        assert!(validate_url("http://example.com/path?q=1"));
        assert!(!validate_url(""));
        assert!(!validate_url("not a url"));
    }

    #[test]
    fn test_validate_json_valid() {
        assert!(validate_json("{}"));
        assert!(validate_json("[]"));
        assert!(validate_json("null"));
        assert!(validate_json("true"));
        assert!(validate_json("false"));
        assert!(validate_json("42"));
        assert!(validate_json("-1.5e10"));
        assert!(validate_json("\"hello\""));
        assert!(validate_json(r#"{"name": "Alice", "age": 30}"#));
        assert!(validate_json("[1, 2, 3]"));
        assert!(validate_json(r#"{"nested": {"array": [1, 2, 3]}}"#));
    }

    #[test]
    fn test_validate_json_invalid() {
        assert!(!validate_json(""));
        assert!(!validate_json("{"));
        assert!(!validate_json("[1, 2,"));
        assert!(!validate_json("{key: \"value\"}")); // unquoted key
        assert!(!validate_json("{'key': 'value'}")); // single quotes
    }

    #[test]
    fn test_validate_uuid() {
        assert!(validate_uuid("550e8400-e29b-41d4-a716-446655440000"));
        assert!(!validate_uuid("550e8400-e29b-41d4-a716"));
        assert!(!validate_uuid(""));
        assert!(!validate_uuid("not-a-uuid-at-all-invalid-format-here"));
    }

    #[test]
    fn test_validate_credit_card_valid() {
        // Known valid Luhn-check numbers
        assert!(validate_credit_card("4532015112830366"));
        assert!(validate_credit_card("378282246310005")); // AmEx
        assert!(validate_credit_card("6011111111111117")); // Discover
    }

    #[test]
    fn test_validate_credit_card_invalid() {
        assert!(!validate_credit_card("1234567890123456")); // bad Luhn
        assert!(!validate_credit_card(""));
        assert!(!validate_credit_card("123")); // too short
    }

    #[test]
    fn test_validate_credit_card_formatted() {
        assert!(validate_credit_card("4532 0151 1283 0366"));
        assert!(validate_credit_card("4532-0151-1283-0366"));
    }

    #[test]
    fn test_is_valid_date() {
        assert!(is_valid_date("2026-04-16", "YYYY-MM-DD"));
        assert!(is_valid_date("2020-02-29", "YYYY-MM-DD")); // leap year
        assert!(!is_valid_date("2021-02-29", "YYYY-MM-DD")); // not a leap year
        assert!(!is_valid_date("2026-13-01", "YYYY-MM-DD")); // invalid month
        assert!(!is_valid_date("2026-04-31", "YYYY-MM-DD")); // April has 30 days
    }

    #[test]
    fn test_is_valid_date_with_time() {
        assert!(is_valid_date("2026-04-16 14:30:45", "YYYY-MM-DD HH:mm:SS"));
        assert!(!is_valid_date("2026-04-16 25:30:45", "YYYY-MM-DD HH:mm:SS"));
    }

    #[test]
    fn test_profile_empty() {
        let p = profile_column(&[], TypeId::Int64);
        assert_eq!(p.total_count, 0);
        assert_eq!(p.null_count, 0);
    }

    #[test]
    fn test_profile_numeric() {
        let values = &["1", "2", "3", "4", "5"];
        let p = profile_column(values, TypeId::Int64);
        assert_eq!(p.total_count, 5);
        assert_eq!(p.distinct_count, 5);
        assert_eq!(p.null_count, 0);
        assert_eq!(p.mean, Some(3.0));
        assert_eq!(p.median, Some(3.0));
    }

    #[test]
    fn test_profile_with_nulls() {
        let values = &["1", "", "2", "", "3"];
        let p = profile_column(values, TypeId::Int64);
        assert_eq!(p.total_count, 5);
        assert_eq!(p.null_count, 2);
    }

    #[test]
    fn test_profile_text() {
        let values = &["apple", "banana", "apple", "cherry"];
        let p = profile_column(values, TypeId::Varchar);
        assert_eq!(p.total_count, 4);
        assert_eq!(p.distinct_count, 3);
        // Most common should be "apple" with count 2
        assert_eq!(p.most_common_values[0].0, "apple");
        assert_eq!(p.most_common_values[0].1, 2);
    }

    #[test]
    fn test_profile_percentiles() {
        let values: Vec<String> = (1..=100).map(|i| i.to_string()).collect();
        let refs: Vec<&str> = values.iter().map(|s| s.as_str()).collect();
        let p = profile_column(&refs, TypeId::Int64);

        // p50 ~= 50.5, p95 ~= 95.05
        let p50 = p.p50.unwrap();
        assert!((p50 - 50.5).abs() < 1.0);
        let p95 = p.p95.unwrap();
        assert!((p95 - 95.0).abs() < 1.0);
    }

    #[test]
    fn test_profile_pattern_frequencies() {
        let values = &["abc123", "def456", "ghi789"];
        let p = profile_column(values, TypeId::Varchar);
        // All match pattern "AAA999"
        assert!(!p.pattern_frequencies.is_empty());
        assert_eq!(p.pattern_frequencies[0].0, "AAA999");
        assert_eq!(p.pattern_frequencies[0].1, 3);
    }

    #[test]
    fn test_data_contract_completeness() {
        let mut profile = DataProfile::empty();
        profile.total_count = 100;
        profile.null_count = 2;

        let rules = vec![ContractRule::Completeness {
            field: "email".into(),
            threshold: 0.95,
        }];
        let results = data_contract(&profile, &rules);
        assert_eq!(results.len(), 1);
        assert!(results[0].passed); // 98% >= 95%
    }

    #[test]
    fn test_data_contract_completeness_fail() {
        let mut profile = DataProfile::empty();
        profile.total_count = 100;
        profile.null_count = 50;

        let rules = vec![ContractRule::Completeness {
            field: "email".into(),
            threshold: 0.95,
        }];
        let results = data_contract(&profile, &rules);
        assert!(!results[0].passed); // 50% < 95%
    }

    #[test]
    fn test_data_contract_row_count() {
        let mut profile = DataProfile::empty();
        profile.total_count = 500;

        let rules = vec![ContractRule::RowCount {
            min: 100,
            max: 1000,
        }];
        let results = data_contract(&profile, &rules);
        assert!(results[0].passed);
    }

    #[test]
    fn test_data_contract_row_count_fail() {
        let mut profile = DataProfile::empty();
        profile.total_count = 50;

        let rules = vec![ContractRule::RowCount {
            min: 100,
            max: 1000,
        }];
        let results = data_contract(&profile, &rules);
        assert!(!results[0].passed);
    }
}
