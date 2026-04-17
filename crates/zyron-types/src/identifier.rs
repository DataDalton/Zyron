//! Identifier validation functions.
//!
//! ISBN-10/13, IBAN, EAN-8/EAN-13/UPC-A, VIN, ISSN, SWIFT/BIC, SSN format.
//! All use check-digit algorithms (modular arithmetic).

use zyron_common::{Result, ZyronError};

// ---------------------------------------------------------------------------
// ISBN (International Standard Book Number)
// ---------------------------------------------------------------------------

/// Validates an ISBN-10 or ISBN-13 string. Hyphens and spaces are stripped.
pub fn validate_isbn(text: &str) -> bool {
    let clean: String = text.chars().filter(|c| c.is_ascii_alphanumeric()).collect();
    match clean.len() {
        10 => validate_isbn10(&clean),
        13 => validate_isbn13(&clean),
        _ => false,
    }
}

fn validate_isbn10(clean: &str) -> bool {
    let bytes = clean.as_bytes();
    let mut sum = 0u32;
    for (i, &b) in bytes.iter().enumerate() {
        let val = if i == 9 && (b == b'X' || b == b'x') {
            10
        } else if b.is_ascii_digit() {
            (b - b'0') as u32
        } else {
            return false;
        };
        sum += val * (10 - i as u32);
    }
    sum % 11 == 0
}

fn validate_isbn13(clean: &str) -> bool {
    if !clean.bytes().all(|b| b.is_ascii_digit()) {
        return false;
    }
    let bytes = clean.as_bytes();
    let mut sum = 0u32;
    for (i, &b) in bytes.iter().enumerate() {
        let val = (b - b'0') as u32;
        if i % 2 == 0 {
            sum += val;
        } else {
            sum += val * 3;
        }
    }
    sum % 10 == 0
}

/// Formats an ISBN string into ISBN-10 or ISBN-13 format with hyphens.
/// Returns the cleaned ISBN if formatting details are not available.
pub fn isbn_format(text: &str, version: u8) -> Result<String> {
    let clean: String = text.chars().filter(|c| c.is_ascii_alphanumeric()).collect();
    match version {
        10 => {
            if clean.len() != 10 || !validate_isbn10(&clean) {
                return Err(ZyronError::ExecutionError("Invalid ISBN-10".into()));
            }
            Ok(clean)
        }
        13 => {
            if clean.len() == 13 && validate_isbn13(&clean) {
                Ok(format!(
                    "{}-{}-{}-{}-{}",
                    &clean[0..3],
                    &clean[3..4],
                    &clean[4..9],
                    &clean[9..12],
                    &clean[12..13]
                ))
            } else {
                Err(ZyronError::ExecutionError("Invalid ISBN-13".into()))
            }
        }
        _ => Err(ZyronError::ExecutionError(
            "ISBN version must be 10 or 13".into(),
        )),
    }
}

/// Converts an ISBN-10 to ISBN-13 by prepending "978" and recalculating the check digit.
pub fn isbn_to_13(text: &str) -> Result<String> {
    let clean: String = text.chars().filter(|c| c.is_ascii_alphanumeric()).collect();
    if clean.len() != 10 || !validate_isbn10(&clean) {
        return Err(ZyronError::ExecutionError("Invalid ISBN-10".into()));
    }

    let base = format!("978{}", &clean[..9]);
    let bytes = base.as_bytes();
    let mut sum = 0u32;
    for (i, &b) in bytes.iter().enumerate() {
        let val = (b - b'0') as u32;
        if i % 2 == 0 {
            sum += val;
        } else {
            sum += val * 3;
        }
    }
    let check = (10 - (sum % 10)) % 10;
    Ok(format!("{}{}", base, check))
}

// ---------------------------------------------------------------------------
// IBAN (International Bank Account Number)
// ---------------------------------------------------------------------------

/// Validates an IBAN using the mod-97 check algorithm (ISO 13616).
pub fn validate_iban(text: &str) -> bool {
    let clean: String = text
        .chars()
        .filter(|c| !c.is_whitespace())
        .map(|c| c.to_ascii_uppercase())
        .collect();

    if clean.len() < 5 || clean.len() > 34 {
        return false;
    }

    // First two chars must be letters (country code)
    if !clean[..2].chars().all(|c| c.is_ascii_uppercase()) {
        return false;
    }
    // Next two must be digits (check digits)
    if !clean[2..4].chars().all(|c| c.is_ascii_digit()) {
        return false;
    }
    // Remaining must be alphanumeric
    if !clean[4..].chars().all(|c| c.is_ascii_alphanumeric()) {
        return false;
    }

    // Rearrange: move first 4 chars to end
    let rearranged = format!("{}{}", &clean[4..], &clean[..4]);

    // Convert letters to numbers (A=10, B=11, ..., Z=35) and compute mod 97
    let mut remainder = 0u64;
    for c in rearranged.chars() {
        let val = if c.is_ascii_digit() {
            (c as u64) - ('0' as u64)
        } else {
            (c as u64) - ('A' as u64) + 10
        };
        if val >= 10 {
            remainder = (remainder * 100 + val) % 97;
        } else {
            remainder = (remainder * 10 + val) % 97;
        }
    }

    remainder == 1
}

/// Extracts the country code from an IBAN (first two letters).
pub fn iban_country(text: &str) -> Result<String> {
    let clean: String = text.chars().filter(|c| !c.is_whitespace()).collect();
    if clean.len() < 2 || !clean[..2].chars().all(|c| c.is_ascii_alphabetic()) {
        return Err(ZyronError::ExecutionError("Invalid IBAN format".into()));
    }
    Ok(clean[..2].to_uppercase())
}

/// Extracts the BBAN (Basic Bank Account Number) from an IBAN.
/// The BBAN is the country-specific part after the country code and check digits.
pub fn iban_bban(text: &str) -> Result<String> {
    let clean: String = text.chars().filter(|c| !c.is_whitespace()).collect();
    if clean.len() < 5 {
        return Err(ZyronError::ExecutionError("Invalid IBAN format".into()));
    }
    Ok(clean[4..].to_string())
}

// ---------------------------------------------------------------------------
// EAN (European Article Number) / UPC
// ---------------------------------------------------------------------------

/// Validates EAN-8, EAN-13, or UPC-A barcodes using the check digit algorithm.
pub fn validate_ean(text: &str) -> bool {
    let clean: String = text.chars().filter(|c| c.is_ascii_digit()).collect();
    match clean.len() {
        8 => validate_ean_checksum(&clean, true),
        12 => validate_ean_checksum(&clean, false), // UPC-A
        13 => validate_ean_checksum(&clean, true),
        _ => false,
    }
}

fn validate_ean_checksum(clean: &str, ean_mode: bool) -> bool {
    let bytes = clean.as_bytes();
    let len = bytes.len();
    let mut sum = 0u32;

    for (i, &b) in bytes.iter().enumerate() {
        if !b.is_ascii_digit() {
            return false;
        }
        let val = (b - b'0') as u32;
        // EAN: odd positions (0-indexed even) weight 1, even positions weight 3
        // UPC-A (12 digits): odd positions weight 3, even positions weight 1
        let weight = if ean_mode {
            if i % 2 == 0 { 1 } else { 3 }
        } else {
            if i % 2 == 0 { 3 } else { 1 }
        };
        if i < len - 1 {
            sum += val * weight;
        } else {
            // Check digit
            let expected = (10 - (sum % 10)) % 10;
            return val == expected;
        }
    }
    false
}

// ---------------------------------------------------------------------------
// VIN (Vehicle Identification Number)
// ---------------------------------------------------------------------------

/// Validates a 17-character Vehicle Identification Number.
/// Uses the North American (FMVSS 115) check digit algorithm.
pub fn validate_vin(text: &str) -> bool {
    let clean: String = text
        .chars()
        .filter(|c| !c.is_whitespace())
        .map(|c| c.to_ascii_uppercase())
        .collect();

    if clean.len() != 17 {
        return false;
    }

    // VIN cannot contain I, O, Q
    if clean.chars().any(|c| c == 'I' || c == 'O' || c == 'Q') {
        return false;
    }

    // All characters must be alphanumeric
    if !clean.chars().all(|c| c.is_ascii_alphanumeric()) {
        return false;
    }

    // Transliteration values
    let transliterate = |c: char| -> Option<u32> {
        match c {
            'A' | 'J' => Some(1),
            'B' | 'K' | 'S' => Some(2),
            'C' | 'L' | 'T' => Some(3),
            'D' | 'M' | 'U' => Some(4),
            'E' | 'N' | 'V' => Some(5),
            'F' | 'W' => Some(6),
            'G' | 'P' | 'X' => Some(7),
            'H' | 'Y' => Some(8),
            'R' | 'Z' => Some(9),
            '0'..='9' => Some((c as u32) - ('0' as u32)),
            _ => None,
        }
    };

    // Position weights
    let weights: [u32; 17] = [8, 7, 6, 5, 4, 3, 2, 10, 0, 9, 8, 7, 6, 5, 4, 3, 2];

    let mut sum = 0u32;
    for (i, c) in clean.chars().enumerate() {
        let val = match transliterate(c) {
            Some(v) => v,
            None => return false,
        };
        sum += val * weights[i];
    }

    let check_digit = sum % 11;
    let expected = if check_digit == 10 {
        'X'
    } else {
        (b'0' + check_digit as u8) as char
    };
    let actual = clean.chars().nth(8).unwrap_or('\0');

    expected == actual
}

/// Extracts the model year from a VIN (position 10).
/// Returns the approximate year range. Since VIN year codes repeat every 30 years,
/// this returns the most recent valid year.
pub fn vin_year(text: &str) -> Result<u16> {
    let clean: String = text
        .chars()
        .filter(|c| !c.is_whitespace())
        .map(|c| c.to_ascii_uppercase())
        .collect();

    if clean.len() != 17 {
        return Err(ZyronError::ExecutionError(
            "VIN must be 17 characters".into(),
        ));
    }

    let year_char = clean.chars().nth(9).unwrap_or('\0');
    let base_year = match year_char {
        'A' => 2010,
        'B' => 2011,
        'C' => 2012,
        'D' => 2013,
        'E' => 2014,
        'F' => 2015,
        'G' => 2016,
        'H' => 2017,
        'J' => 2018,
        'K' => 2019,
        'L' => 2020,
        'M' => 2021,
        'N' => 2022,
        'P' => 2023,
        'R' => 2024,
        'S' => 2025,
        'T' => 2026,
        'V' => 2027,
        'W' => 2028,
        'X' => 2029,
        'Y' => 2030,
        '1' => 2001,
        '2' => 2002,
        '3' => 2003,
        '4' => 2004,
        '5' => 2005,
        '6' => 2006,
        '7' => 2007,
        '8' => 2008,
        '9' => 2009,
        _ => {
            return Err(ZyronError::ExecutionError(format!(
                "Invalid VIN year character: {}",
                year_char
            )));
        }
    };
    Ok(base_year)
}

/// Extracts the country/region from a VIN using the WMI (World Manufacturer Identifier).
/// Returns a region name based on the first character.
pub fn vin_country(text: &str) -> Result<String> {
    let clean: String = text
        .chars()
        .filter(|c| !c.is_whitespace())
        .map(|c| c.to_ascii_uppercase())
        .collect();

    if clean.is_empty() {
        return Err(ZyronError::ExecutionError("Empty VIN".into()));
    }

    let region = match clean.chars().next().unwrap_or('\0') {
        '1' | '4' | '5' => "United States",
        '2' => "Canada",
        '3' => "Mexico",
        '6' | '7' => "Oceania",
        '8' => "South America",
        '9' => "South America/Oceania",
        'J' => "Japan",
        'K' => "South Korea",
        'L' => "China",
        'M' => "India/Indonesia/Thailand",
        'N' => "Iran/Pakistan/Turkey",
        'S' => "United Kingdom",
        'T' => "Switzerland/Czech Republic/Hungary",
        'V' => "France/Spain",
        'W' => "Germany",
        'X' => "Russia/Netherlands",
        'Y' => "Sweden/Finland/Norway",
        'Z' => "Italy",
        c => {
            return Err(ZyronError::ExecutionError(format!(
                "Unknown VIN region for character: {}",
                c
            )));
        }
    };
    Ok(region.to_string())
}

/// Extracts the manufacturer from a VIN using the WMI (first 3 characters).
/// Returns the WMI code (actual manufacturer lookup requires a database).
pub fn vin_manufacturer(text: &str) -> Result<String> {
    let clean: String = text
        .chars()
        .filter(|c| !c.is_whitespace())
        .map(|c| c.to_ascii_uppercase())
        .collect();

    if clean.len() < 3 {
        return Err(ZyronError::ExecutionError(
            "VIN must be at least 3 characters for WMI".into(),
        ));
    }
    Ok(clean[..3].to_string())
}

// ---------------------------------------------------------------------------
// ISSN (International Standard Serial Number)
// ---------------------------------------------------------------------------

/// Validates an 8-digit ISSN with check digit.
pub fn validate_issn(text: &str) -> bool {
    let clean: String = text.chars().filter(|c| c.is_ascii_alphanumeric()).collect();

    if clean.len() != 8 {
        return false;
    }

    let bytes = clean.as_bytes();
    let mut sum = 0u32;
    for (i, &b) in bytes[..7].iter().enumerate() {
        if !b.is_ascii_digit() {
            return false;
        }
        sum += (b - b'0') as u32 * (8 - i as u32);
    }

    let remainder = sum % 11;
    let expected_check = if remainder == 0 { 0 } else { 11 - remainder };

    let actual = bytes[7];
    if expected_check == 10 {
        actual == b'X' || actual == b'x'
    } else {
        actual.is_ascii_digit() && (actual - b'0') as u32 == expected_check
    }
}

// ---------------------------------------------------------------------------
// SWIFT/BIC (Bank Identifier Code)
// ---------------------------------------------------------------------------

/// Validates a SWIFT/BIC code (8 or 11 characters).
/// Format: AAAA BB CC (DDD) where:
/// - AAAA: bank code (letters)
/// - BB: country code (letters)
/// - CC: location code (alphanumeric)
/// - DDD: optional branch code (alphanumeric)
pub fn validate_swift(text: &str) -> bool {
    let clean: String = text
        .chars()
        .filter(|c| !c.is_whitespace())
        .map(|c| c.to_ascii_uppercase())
        .collect();

    if clean.len() != 8 && clean.len() != 11 {
        return false;
    }

    // First 4: bank code (letters only)
    if !clean[..4].chars().all(|c| c.is_ascii_uppercase()) {
        return false;
    }
    // Next 2: country code (letters only)
    if !clean[4..6].chars().all(|c| c.is_ascii_uppercase()) {
        return false;
    }
    // Next 2: location code (alphanumeric)
    if !clean[6..8].chars().all(|c| c.is_ascii_alphanumeric()) {
        return false;
    }
    // Optional 3: branch code (alphanumeric)
    if clean.len() == 11 && !clean[8..11].chars().all(|c| c.is_ascii_alphanumeric()) {
        return false;
    }
    true
}

// ---------------------------------------------------------------------------
// SSN (US Social Security Number) - format validation only
// ---------------------------------------------------------------------------

/// Validates US Social Security Number format (XXX-XX-XXXX).
/// Format validation only, no PII inference. Rejects known invalid ranges:
/// area numbers 000, 666, or 900-999.
pub fn validate_ssn(text: &str) -> bool {
    let clean: String = text.chars().filter(|c| c.is_ascii_digit()).collect();

    if clean.len() != 9 {
        return false;
    }

    let area: u32 = clean[..3].parse().unwrap_or(0);
    let group: u32 = clean[3..5].parse().unwrap_or(0);
    let serial: u32 = clean[5..9].parse().unwrap_or(0);

    // Invalid area numbers
    if area == 0 || area == 666 || area >= 900 {
        return false;
    }
    // Group and serial cannot be zero
    if group == 0 || serial == 0 {
        return false;
    }
    true
}

#[cfg(test)]
mod tests {
    use super::*;

    // ISBN
    #[test]
    fn test_isbn10_valid() {
        assert!(validate_isbn("0-306-40615-2"));
        assert!(validate_isbn("0306406152"));
    }

    #[test]
    fn test_isbn10_with_x() {
        assert!(validate_isbn("0-8044-2957-X"));
    }

    #[test]
    fn test_isbn13_valid() {
        assert!(validate_isbn("978-0-306-40615-7"));
        assert!(validate_isbn("9780306406157"));
    }

    #[test]
    fn test_isbn_invalid() {
        assert!(!validate_isbn("0-306-40615-3")); // bad check digit
        assert!(!validate_isbn("123"));
        assert!(!validate_isbn(""));
    }

    #[test]
    fn test_isbn_to_13() {
        let isbn13 = isbn_to_13("0-306-40615-2").unwrap();
        assert_eq!(isbn13, "9780306406157");
    }

    #[test]
    fn test_isbn_to_13_invalid() {
        assert!(isbn_to_13("invalid").is_err());
    }

    #[test]
    fn test_isbn_format_13() {
        let formatted = isbn_format("9780306406157", 13).unwrap();
        assert!(formatted.contains('-'));
    }

    #[test]
    fn test_isbn_format_invalid_version() {
        assert!(isbn_format("9780306406157", 99).is_err());
    }

    // IBAN
    #[test]
    fn test_iban_valid() {
        assert!(validate_iban("DE89 3704 0044 0532 0130 00"));
        assert!(validate_iban("GB29 NWBK 6016 1331 9268 19"));
        assert!(validate_iban("FR76 3000 6000 0112 3456 7890 189"));
    }

    #[test]
    fn test_iban_invalid() {
        assert!(!validate_iban("DE89370400440532013001")); // bad check
        assert!(!validate_iban("XX00"));
        assert!(!validate_iban(""));
    }

    #[test]
    fn test_iban_country() {
        assert_eq!(iban_country("DE89370400440532013000").unwrap(), "DE");
        assert_eq!(iban_country("GB29NWBK60161331926819").unwrap(), "GB");
    }

    #[test]
    fn test_iban_bban() {
        let bban = iban_bban("DE89370400440532013000").unwrap();
        assert_eq!(bban, "370400440532013000");
    }

    // EAN
    #[test]
    fn test_ean13_valid() {
        assert!(validate_ean("4006381333931")); // EAN-13
    }

    #[test]
    fn test_ean8_valid() {
        assert!(validate_ean("96385074")); // EAN-8
    }

    #[test]
    fn test_ean_invalid() {
        assert!(!validate_ean("4006381333932")); // bad check digit
        assert!(!validate_ean("123"));
    }

    #[test]
    fn test_upc_valid() {
        assert!(validate_ean("036000291452")); // UPC-A (12 digits)
    }

    // VIN
    #[test]
    fn test_vin_valid() {
        // 11111111111111111 is a known valid test VIN
        assert!(validate_vin("11111111111111111"));
    }

    #[test]
    fn test_vin_invalid_length() {
        assert!(!validate_vin("1234"));
        assert!(!validate_vin(""));
    }

    #[test]
    fn test_vin_invalid_chars() {
        // VIN cannot contain I, O, Q
        assert!(!validate_vin("1I111111111111111"));
    }

    #[test]
    fn test_vin_year() {
        // Position 10 (0-indexed 9) determines year
        let vin = "11111111111111111";
        let year = vin_year(vin).unwrap();
        assert!(year >= 2001);
    }

    #[test]
    fn test_vin_country() {
        let country = vin_country("1HGBH41JXMN109186").unwrap();
        assert_eq!(country, "United States");
    }

    #[test]
    fn test_vin_country_japan() {
        let country = vin_country("JHM").unwrap();
        assert_eq!(country, "Japan");
    }

    #[test]
    fn test_vin_manufacturer() {
        let wmi = vin_manufacturer("1HGBH41JXMN109186").unwrap();
        assert_eq!(wmi, "1HG");
    }

    // ISSN
    #[test]
    fn test_issn_valid() {
        assert!(validate_issn("0378-5955"));
        assert!(validate_issn("03785955"));
    }

    #[test]
    fn test_issn_with_x() {
        assert!(validate_issn("0317-8471"));
    }

    #[test]
    fn test_issn_invalid() {
        assert!(!validate_issn("0378-5956")); // bad check digit
        assert!(!validate_issn("12345"));
    }

    // SWIFT
    #[test]
    fn test_swift_valid_8() {
        assert!(validate_swift("DEUTDEFF"));
    }

    #[test]
    fn test_swift_valid_11() {
        assert!(validate_swift("DEUTDEFF500"));
    }

    #[test]
    fn test_swift_invalid() {
        assert!(!validate_swift("DEUT")); // too short
        assert!(!validate_swift("1234DEFF")); // bank code not letters
        assert!(!validate_swift("DEUT12FF")); // country code not letters
    }

    // SSN
    #[test]
    fn test_ssn_valid() {
        assert!(validate_ssn("123-45-6789"));
        assert!(validate_ssn("123456789"));
    }

    #[test]
    fn test_ssn_invalid_area() {
        assert!(!validate_ssn("000-12-3456")); // area 000
        assert!(!validate_ssn("666-12-3456")); // area 666
        assert!(!validate_ssn("900-12-3456")); // area >= 900
    }

    #[test]
    fn test_ssn_invalid_group_serial() {
        assert!(!validate_ssn("123-00-1234")); // group 00
        assert!(!validate_ssn("123-12-0000")); // serial 0000
    }

    #[test]
    fn test_ssn_invalid_format() {
        assert!(!validate_ssn("12345")); // too short
        assert!(!validate_ssn("")); // empty
    }
}
