//! Composite interval type with calendar-aware arithmetic.
//!
//! Intervals are stored as (months, days, nanoseconds) to support both
//! fixed-duration units (nanoseconds, microseconds, seconds, hours, days)
//! and calendar units (months, quarters, years, decades, centuries, millennia).
//!
//! Calendar arithmetic clamps month additions to the last valid day of the
//! target month, matching SQL standard behavior (Jan 31 + 1 month = Feb 28 or
//! Feb 29 depending on leap year).
//!
//! On-disk / wire format: little-endian [months: i32][days: i32][nanoseconds: i64].

use crate::error::{Result, ZyronError};
use serde::{Deserialize, Serialize};

// ---------------------------------------------------------------------------
// Core type
// ---------------------------------------------------------------------------

/// A SQL interval with calendar-aware components.
///
/// Composite representation:
/// - `months`: number of calendar months (positive or negative)
/// - `days`: number of calendar days (positive or negative)
/// - `nanoseconds`: sub-day duration in nanoseconds (positive or negative)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[repr(C)]
pub struct Interval {
    pub months: i32,
    pub days: i32,
    pub nanoseconds: i64,
}

impl Interval {
    pub const ZERO: Interval = Interval {
        months: 0,
        days: 0,
        nanoseconds: 0,
    };

    /// Constructs a new Interval.
    pub fn new(months: i32, days: i32, nanoseconds: i64) -> Self {
        Self {
            months,
            days,
            nanoseconds,
        }
    }

    /// Constructs an interval from months only.
    pub fn from_months(months: i32) -> Self {
        Self::new(months, 0, 0)
    }

    /// Constructs an interval from days only.
    pub fn from_days(days: i32) -> Self {
        Self::new(0, days, 0)
    }

    /// Constructs an interval from nanoseconds only.
    pub fn from_nanoseconds(nanoseconds: i64) -> Self {
        Self::new(0, 0, nanoseconds)
    }

    /// Constructs an interval from microseconds only.
    pub fn from_microseconds(micros: i64) -> Self {
        Self::new(0, 0, micros.saturating_mul(1_000))
    }

    /// Negates all components.
    pub fn negate(self) -> Self {
        Self::new(
            self.months.wrapping_neg(),
            self.days.wrapping_neg(),
            self.nanoseconds.wrapping_neg(),
        )
    }

    /// Field-wise addition (signed).
    pub fn add(self, other: Interval) -> Self {
        Self::new(
            self.months.wrapping_add(other.months),
            self.days.wrapping_add(other.days),
            self.nanoseconds.wrapping_add(other.nanoseconds),
        )
    }

    /// Field-wise subtraction.
    pub fn subtract(self, other: Interval) -> Self {
        self.add(other.negate())
    }

    /// Multiplies all components by a scalar integer. Saturates on overflow.
    pub fn multiply_by(self, factor: i64) -> Self {
        let months = (self.months as i64)
            .saturating_mul(factor)
            .clamp(i32::MIN as i64, i32::MAX as i64) as i32;
        let days = (self.days as i64)
            .saturating_mul(factor)
            .clamp(i32::MIN as i64, i32::MAX as i64) as i32;
        let nanos = self.nanoseconds.saturating_mul(factor);
        Self::new(months, days, nanos)
    }

    /// Returns true if all components are zero.
    pub fn is_zero(&self) -> bool {
        self.months == 0 && self.days == 0 && self.nanoseconds == 0
    }

    // -----------------------------------------------------------------------
    // Serialization
    // -----------------------------------------------------------------------

    /// Serializes to 16 bytes (little-endian).
    /// Layout: [months: i32 LE][days: i32 LE][nanoseconds: i64 LE].
    pub fn to_le_bytes(&self) -> [u8; 16] {
        let mut buf = [0u8; 16];
        buf[0..4].copy_from_slice(&self.months.to_le_bytes());
        buf[4..8].copy_from_slice(&self.days.to_le_bytes());
        buf[8..16].copy_from_slice(&self.nanoseconds.to_le_bytes());
        buf
    }

    /// Deserializes from 16 little-endian bytes.
    pub fn from_le_bytes(bytes: &[u8; 16]) -> Self {
        let months = i32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]);
        let days = i32::from_le_bytes([bytes[4], bytes[5], bytes[6], bytes[7]]);
        let nanoseconds = i64::from_le_bytes([
            bytes[8], bytes[9], bytes[10], bytes[11], bytes[12], bytes[13], bytes[14], bytes[15],
        ]);
        Self::new(months, days, nanoseconds)
    }

    // -----------------------------------------------------------------------
    // Calendar arithmetic
    // -----------------------------------------------------------------------

    /// Conservative fixed-duration approximation in nanoseconds.
    /// Month = 30 days, year = 365.25 days.
    /// Used for total ordering of intervals with different composite shapes.
    /// Returned as i128 to avoid overflow for large intervals.
    pub fn approximate_nanoseconds(&self) -> i128 {
        const NANOS_PER_DAY: i128 = 86_400_000_000_000;
        const NANOS_PER_MONTH: i128 = 30 * NANOS_PER_DAY;
        (self.months as i128) * NANOS_PER_MONTH
            + (self.days as i128) * NANOS_PER_DAY
            + (self.nanoseconds as i128)
    }

    /// Applies the interval to a timestamp (microseconds since Unix epoch),
    /// returning the resulting timestamp in microseconds.
    ///
    /// Calendar-aware: adding 1 month to Jan 31 produces Feb 28 (or Feb 29
    /// in a leap year). Adding 1 year to Feb 29 produces Feb 28.
    pub fn add_to_timestamp_micros(&self, ts_micros: i64) -> i64 {
        let (mut year, mut month, mut day, time_of_day_micros) =
            decompose_timestamp_micros(ts_micros);

        // Apply months with clamping
        if self.months != 0 {
            let total_months = year as i64 * 12 + (month as i64 - 1) + self.months as i64;
            year = total_months.div_euclid(12) as i32;
            month = (total_months.rem_euclid(12) + 1) as u32;
            // Clamp day to last valid day of the target month
            let max_day = days_in_month(year, month);
            if day > max_day {
                day = max_day;
            }
        }

        // Apply days
        if self.days != 0 {
            let (ny, nm, nd) = add_days_to_date(year, month, day, self.days as i64);
            year = ny;
            month = nm;
            day = nd;
        }

        // Re-compose to microseconds, add sub-day nanoseconds (truncated to us)
        let base_micros = compose_timestamp_micros(year, month, day, 0);
        let interval_micros = self.nanoseconds / 1_000;
        base_micros
            .saturating_add(time_of_day_micros)
            .saturating_add(interval_micros)
    }

    /// Subtracts the interval from a timestamp.
    pub fn subtract_from_timestamp_micros(&self, ts_micros: i64) -> i64 {
        self.negate().add_to_timestamp_micros(ts_micros)
    }
}

impl Default for Interval {
    fn default() -> Self {
        Self::ZERO
    }
}

// ---------------------------------------------------------------------------
// Ordering via approximate nanoseconds
// ---------------------------------------------------------------------------

impl PartialOrd for Interval {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Interval {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.approximate_nanoseconds()
            .cmp(&other.approximate_nanoseconds())
    }
}

// ---------------------------------------------------------------------------
// Calendar primitives (shared with business_time)
// ---------------------------------------------------------------------------

/// Returns true if the given Gregorian year is a leap year.
pub fn is_leap(year: i32) -> bool {
    (year % 4 == 0 && year % 100 != 0) || year % 400 == 0
}

/// Returns the number of days in the given Gregorian month (1-12).
pub fn days_in_month(year: i32, month: u32) -> u32 {
    match month {
        1 | 3 | 5 | 7 | 8 | 10 | 12 => 31,
        4 | 6 | 9 | 11 => 30,
        2 => {
            if is_leap(year) {
                29
            } else {
                28
            }
        }
        _ => 0,
    }
}

/// Decomposes a Unix-epoch microsecond timestamp into
/// (year, month, day, time-of-day-microseconds).
fn decompose_timestamp_micros(ts_micros: i64) -> (i32, u32, u32, i64) {
    let seconds = ts_micros.div_euclid(1_000_000);
    let micros_in_second = ts_micros.rem_euclid(1_000_000);

    let days_since_epoch = seconds.div_euclid(86_400);
    let time_of_day_seconds = seconds.rem_euclid(86_400);
    let time_of_day_micros = time_of_day_seconds * 1_000_000 + micros_in_second;

    let (year, month, day) = ymd_from_days(days_since_epoch as i32);
    (year, month, day, time_of_day_micros)
}

/// Composes (year, month, day, time-of-day-micros) into a Unix-epoch microsecond timestamp.
fn compose_timestamp_micros(year: i32, month: u32, day: u32, time_of_day_micros: i64) -> i64 {
    let days = days_from_ymd(year, month, day) as i64;
    days.saturating_mul(86_400_000_000)
        .saturating_add(time_of_day_micros)
}

/// Returns days since 1970-01-01 for a given Gregorian date.
pub fn days_from_ymd(year: i32, month: u32, day: u32) -> i32 {
    // Days from 0000-03-01 is the "shifted" epoch where leap years are easier to count.
    // Simpler: iterate years between 1970 and target.
    let mut total: i32 = 0;
    if year >= 1970 {
        for y in 1970..year {
            total += if is_leap(y) { 366 } else { 365 };
        }
    } else {
        for y in year..1970 {
            total -= if is_leap(y) { 366 } else { 365 };
        }
    }
    for m in 1..month {
        total += days_in_month(year, m) as i32;
    }
    total + (day as i32 - 1)
}

/// Converts days since 1970-01-01 to (year, month, day).
pub fn ymd_from_days(days: i32) -> (i32, u32, u32) {
    let mut year: i32 = 1970;
    let mut remaining = days;

    if remaining >= 0 {
        loop {
            let year_days = if is_leap(year) { 366 } else { 365 };
            if remaining < year_days {
                break;
            }
            remaining -= year_days;
            year += 1;
        }
    } else {
        while remaining < 0 {
            year -= 1;
            let year_days = if is_leap(year) { 366 } else { 365 };
            remaining += year_days;
        }
    }

    let mut month: u32 = 1;
    while month <= 12 {
        let md = days_in_month(year, month) as i32;
        if remaining < md {
            break;
        }
        remaining -= md;
        month += 1;
    }

    let day = (remaining + 1) as u32;
    (year, month, day)
}

/// Adds a signed number of days to (year, month, day), returning the new date.
fn add_days_to_date(year: i32, month: u32, day: u32, delta: i64) -> (i32, u32, u32) {
    let base_days = days_from_ymd(year, month, day);
    let new_days = (base_days as i64).saturating_add(delta) as i32;
    ymd_from_days(new_days)
}

// ---------------------------------------------------------------------------
// String parser
// ---------------------------------------------------------------------------

/// Parses an interval string into an Interval value.
///
/// Supported forms:
/// - Single component: `"1 hour"`, `"30 minutes"`, `"-2 days"`
/// - Compound: `"1 day 2 hours"`, `"1 year 3 months 4 days"`, `"2 hours and 30 minutes"`
/// - PostgreSQL colon form: `"02:30:00"`, `"02:30:00.5"`, `"1 day 02:30:00"`,
///   `"1 year 2 months 3 days 04:05:06.789"`, `"-02:30:00"`
///
/// Units (case-insensitive, optional trailing 's'):
/// - Sub-second: picosecond (ps), nanosecond (ns), microsecond (us / μs), millisecond (ms)
/// - Time: second (s / sec), minute (min / m), hour (h / hr)
/// - Day-based: day (d), week (w)
/// - Calendar: month (mon), quarter, year (y / yr), decade, century, millennium
pub fn parse_interval_string(s: &str) -> Result<Interval> {
    let trimmed = s.trim();
    if trimmed.is_empty() {
        return Err(ZyronError::ExecutionError("Empty interval string".into()));
    }

    // Normalize whitespace and "and" separators
    let normalized = trimmed.replace(',', " ").replace(" and ", " ");
    let tokens: Vec<&str> = normalized.split_whitespace().collect();
    if tokens.is_empty() {
        return Err(ZyronError::ExecutionError(
            "Interval string has no tokens".into(),
        ));
    }

    let mut acc = Interval::ZERO;
    let mut i = 0;
    while i < tokens.len() {
        let token = tokens[i];

        // Check for colon form (HH:MM:SS[.fff])
        if is_colon_token(token) {
            let colon_interval = parse_colon_form(token)?;
            acc = acc.add(colon_interval);
            i += 1;
            continue;
        }

        // Must be a number followed by a unit
        let count_str = token;
        let count: f64 = count_str
            .parse()
            .map_err(|_| ZyronError::ExecutionError(format!("Invalid number: {}", count_str)))?;

        let unit_token = tokens.get(i + 1).copied().ok_or_else(|| {
            ZyronError::ExecutionError(format!("Interval token '{}' missing unit", count_str))
        })?;

        let unit_interval = interval_from_unit(count, unit_token)?;
        acc = acc.add(unit_interval);
        i += 2;
    }

    Ok(acc)
}

/// Parses "HH:MM:SS" or "HH:MM:SS.fff..." form.
/// A leading `-` negates the entire time component.
fn parse_colon_form(token: &str) -> Result<Interval> {
    let (sign, body) = if let Some(stripped) = token.strip_prefix('-') {
        (-1i64, stripped)
    } else {
        (1i64, token)
    };

    let parts: Vec<&str> = body.split(':').collect();
    if parts.len() != 3 {
        return Err(ZyronError::ExecutionError(format!(
            "Invalid colon form '{}' (expected HH:MM:SS[.frac])",
            token
        )));
    }

    let hours: i64 = parts[0]
        .parse()
        .map_err(|_| ZyronError::ExecutionError(format!("Invalid hours in '{}'", token)))?;
    let minutes: i64 = parts[1]
        .parse()
        .map_err(|_| ZyronError::ExecutionError(format!("Invalid minutes in '{}'", token)))?;

    if !(0..60).contains(&minutes) {
        return Err(ZyronError::ExecutionError(format!(
            "Minutes must be 0-59 in '{}'",
            token
        )));
    }

    let (seconds, frac_nanos) = parse_seconds_with_fraction(parts[2], token)?;
    if !(0..60).contains(&seconds) {
        return Err(ZyronError::ExecutionError(format!(
            "Seconds must be 0-59 in '{}'",
            token
        )));
    }

    const NS_PER_HOUR: i64 = 3_600_000_000_000;
    const NS_PER_MINUTE: i64 = 60_000_000_000;
    const NS_PER_SECOND: i64 = 1_000_000_000;

    let total_nanos = hours
        .saturating_mul(NS_PER_HOUR)
        .saturating_add(minutes.saturating_mul(NS_PER_MINUTE))
        .saturating_add(seconds.saturating_mul(NS_PER_SECOND))
        .saturating_add(frac_nanos)
        .saturating_mul(sign);

    Ok(Interval::from_nanoseconds(total_nanos))
}

/// Parses a "SS" or "SS.fff..." string, returning (whole seconds, fractional nanoseconds).
fn parse_seconds_with_fraction(s: &str, full_token: &str) -> Result<(i64, i64)> {
    if let Some(dot_idx) = s.find('.') {
        let whole_part = &s[..dot_idx];
        let frac_part = &s[dot_idx + 1..];
        let whole: i64 = whole_part.parse().map_err(|_| {
            ZyronError::ExecutionError(format!("Invalid seconds in '{}'", full_token))
        })?;
        // Pad or truncate fractional part to 9 digits (nanoseconds)
        let mut frac_str = String::with_capacity(9);
        frac_str.push_str(frac_part);
        while frac_str.len() < 9 {
            frac_str.push('0');
        }
        let frac: i64 = frac_str[..9].parse().map_err(|_| {
            ZyronError::ExecutionError(format!("Invalid fraction in '{}'", full_token))
        })?;
        Ok((whole, frac))
    } else {
        let whole: i64 = s.parse().map_err(|_| {
            ZyronError::ExecutionError(format!("Invalid seconds in '{}'", full_token))
        })?;
        Ok((whole, 0))
    }
}

fn is_colon_token(token: &str) -> bool {
    let body = token.strip_prefix('-').unwrap_or(token);
    let parts: Vec<&str> = body.split(':').collect();
    if parts.len() != 3 {
        return false;
    }
    // Only consider it a colon token if parts 0 and 1 are pure integers
    parts[0].chars().all(|c| c.is_ascii_digit())
        && parts[1].chars().all(|c| c.is_ascii_digit())
        && parts[2].chars().all(|c| c.is_ascii_digit() || c == '.')
}

/// Interprets a (count, unit) pair as an Interval.
/// Count may be fractional for units smaller than month; fractional months/years/etc
/// are rejected because month/year durations are calendar-dependent.
fn interval_from_unit(count: f64, unit: &str) -> Result<Interval> {
    let unit_lower = unit.to_lowercase();

    // Abbreviations are matched as-is (not stripped of trailing 's' since 'ms'/'us'/'ps' are themselves
    // the canonical forms). Full words get an optional trailing 's' stripped.
    let normalized: &str = match unit_lower.as_str() {
        "ps" | "ns" | "us" | "μs" | "ms" | "s" | "m" | "h" | "d" | "w" | "hr" | "yr" | "y"
        | "min" | "sec" | "mon" => unit_lower.as_str(),
        other => other.trim_end_matches('s'),
    };

    match normalized {
        // Sub-second
        "picosecond" | "ps" => {
            let nanos = (count / 1000.0).trunc() as i64;
            Ok(Interval::from_nanoseconds(nanos))
        }
        "nanosecond" | "nano" | "ns" => Ok(Interval::from_nanoseconds(count as i64)),
        "microsecond" | "micro" | "us" | "μs" => {
            Ok(Interval::from_nanoseconds((count * 1_000.0) as i64))
        }
        "millisecond" | "milli" | "ms" => {
            Ok(Interval::from_nanoseconds((count * 1_000_000.0) as i64))
        }

        // Second and above (fractional allowed, converted to nanoseconds)
        "second" | "sec" | "s" => Ok(Interval::from_nanoseconds((count * 1e9) as i64)),
        "minute" | "min" => Ok(Interval::from_nanoseconds((count * 60e9) as i64)),
        // Accept 'm' for minutes; 'month' is abbreviated 'mon'
        "m" => Ok(Interval::from_nanoseconds((count * 60e9) as i64)),
        "hour" | "hr" | "h" => Ok(Interval::from_nanoseconds((count * 3600e9) as i64)),

        // Day-based
        "day" | "d" => {
            if count.fract() == 0.0 {
                Ok(Interval::from_days(count as i32))
            } else {
                let whole_days = count.trunc() as i32;
                let frac_nanos = (count.fract() * 86_400e9) as i64;
                Ok(Interval::new(0, whole_days, frac_nanos))
            }
        }
        "week" | "w" => {
            let as_days = count * 7.0;
            if as_days.fract() == 0.0 {
                Ok(Interval::from_days(as_days as i32))
            } else {
                let whole_days = as_days.trunc() as i32;
                let frac_nanos = (as_days.fract() * 86_400e9) as i64;
                Ok(Interval::new(0, whole_days, frac_nanos))
            }
        }

        // Calendar units - must be integer
        "month" | "mon" => integer_months(count, unit).map(Interval::from_months),
        "quarter" => {
            integer_months(count, unit).map(|n| Interval::from_months(n.saturating_mul(3)))
        }
        "year" | "yr" | "y" => {
            integer_months(count, unit).map(|n| Interval::from_months(n.saturating_mul(12)))
        }
        "decade" => {
            integer_months(count, unit).map(|n| Interval::from_months(n.saturating_mul(120)))
        }
        "century" | "centurie" => {
            integer_months(count, unit).map(|n| Interval::from_months(n.saturating_mul(1200)))
        }
        "millennium" | "millennia" | "millenium" => {
            integer_months(count, unit).map(|n| Interval::from_months(n.saturating_mul(12000)))
        }

        _ => Err(ZyronError::ExecutionError(format!(
            "Unknown interval unit: {}",
            unit
        ))),
    }
}

fn integer_months(count: f64, unit: &str) -> Result<i32> {
    if count.fract() != 0.0 {
        return Err(ZyronError::ExecutionError(format!(
            "Fractional values not allowed for calendar unit '{}' (ambiguous duration)",
            unit
        )));
    }
    let as_i64 = count as i64;
    if as_i64 < i32::MIN as i64 || as_i64 > i32::MAX as i64 {
        return Err(ZyronError::ExecutionError(format!(
            "Interval count '{}' overflows i32 months",
            count
        )));
    }
    Ok(as_i64 as i32)
}

// ---------------------------------------------------------------------------
// Display (PostgreSQL-style)
// ---------------------------------------------------------------------------

impl std::fmt::Display for Interval {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut parts: Vec<String> = Vec::new();

        // Years and months
        let abs_months = self.months.unsigned_abs();
        let years = abs_months / 12;
        let months = abs_months % 12;
        let months_sign = if self.months < 0 { "-" } else { "" };
        if years > 0 {
            parts.push(format!(
                "{}{} year{}",
                months_sign,
                years,
                if years == 1 { "" } else { "s" }
            ));
        }
        if months > 0 {
            parts.push(format!(
                "{}{} mon{}",
                months_sign,
                months,
                if months == 1 { "" } else { "s" }
            ));
        }

        if self.days != 0 {
            parts.push(format!(
                "{} day{}",
                self.days,
                if self.days.abs() == 1 { "" } else { "s" }
            ));
        }

        if self.nanoseconds != 0 {
            let abs_nanos = self.nanoseconds.unsigned_abs();
            let sign = if self.nanoseconds < 0 { "-" } else { "" };
            let total_ns = abs_nanos;
            let h = total_ns / 3_600_000_000_000;
            let rem = total_ns % 3_600_000_000_000;
            let m = rem / 60_000_000_000;
            let rem = rem % 60_000_000_000;
            let s = rem / 1_000_000_000;
            let frac = rem % 1_000_000_000;

            if frac > 0 {
                // Trim trailing zeros on fraction
                let frac_str = format!("{:09}", frac);
                let trimmed_frac = frac_str.trim_end_matches('0');
                parts.push(format!(
                    "{}{:02}:{:02}:{:02}.{}",
                    sign, h, m, s, trimmed_frac
                ));
            } else {
                parts.push(format!("{}{:02}:{:02}:{:02}", sign, h, m, s));
            }
        }

        if parts.is_empty() {
            f.write_str("00:00:00")
        } else {
            f.write_str(&parts.join(" "))
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // Basic construction
    #[test]
    fn test_zero() {
        assert_eq!(Interval::ZERO.months, 0);
        assert_eq!(Interval::ZERO.days, 0);
        assert_eq!(Interval::ZERO.nanoseconds, 0);
    }

    #[test]
    fn test_new() {
        let i = Interval::new(1, 2, 3);
        assert_eq!(i.months, 1);
        assert_eq!(i.days, 2);
        assert_eq!(i.nanoseconds, 3);
    }

    // Serialization roundtrip
    #[test]
    fn test_serialization_roundtrip() {
        let cases = [
            Interval::new(0, 0, 0),
            Interval::new(12, 30, 86_400_000_000_000),
            Interval::new(-1, -2, -3),
            Interval::new(i32::MAX, i32::MAX, i64::MAX),
            Interval::new(i32::MIN, i32::MIN, i64::MIN),
        ];
        for original in cases {
            let bytes = original.to_le_bytes();
            assert_eq!(bytes.len(), 16);
            let restored = Interval::from_le_bytes(&bytes);
            assert_eq!(original, restored);
        }
    }

    // Parser - simple single-component
    #[test]
    fn test_parse_hours() {
        let i = parse_interval_string("1 hour").unwrap();
        assert_eq!(i, Interval::from_nanoseconds(3_600_000_000_000));
    }

    #[test]
    fn test_parse_hours_plural() {
        let i = parse_interval_string("3 hours").unwrap();
        assert_eq!(i, Interval::from_nanoseconds(3 * 3_600_000_000_000));
    }

    #[test]
    fn test_parse_minutes() {
        let i = parse_interval_string("30 minutes").unwrap();
        assert_eq!(i, Interval::from_nanoseconds(30 * 60_000_000_000));
    }

    #[test]
    fn test_parse_seconds() {
        let i = parse_interval_string("10 seconds").unwrap();
        assert_eq!(i, Interval::from_nanoseconds(10 * 1_000_000_000));
    }

    #[test]
    fn test_parse_milliseconds() {
        let i = parse_interval_string("500 ms").unwrap();
        assert_eq!(i, Interval::from_nanoseconds(500_000_000));
    }

    #[test]
    fn test_parse_microseconds() {
        let i = parse_interval_string("100 us").unwrap();
        assert_eq!(i, Interval::from_nanoseconds(100_000));
    }

    #[test]
    fn test_parse_nanoseconds() {
        let i = parse_interval_string("42 nanoseconds").unwrap();
        assert_eq!(i, Interval::from_nanoseconds(42));
    }

    #[test]
    fn test_parse_picoseconds() {
        // 2000 picoseconds = 2 nanoseconds
        let i = parse_interval_string("2000 ps").unwrap();
        assert_eq!(i, Interval::from_nanoseconds(2));
    }

    #[test]
    fn test_parse_days() {
        let i = parse_interval_string("5 days").unwrap();
        assert_eq!(i, Interval::from_days(5));
    }

    #[test]
    fn test_parse_weeks() {
        let i = parse_interval_string("2 weeks").unwrap();
        assert_eq!(i, Interval::from_days(14));
    }

    #[test]
    fn test_parse_months() {
        let i = parse_interval_string("3 months").unwrap();
        assert_eq!(i, Interval::from_months(3));
    }

    #[test]
    fn test_parse_quarters() {
        let i = parse_interval_string("2 quarters").unwrap();
        assert_eq!(i, Interval::from_months(6));
    }

    #[test]
    fn test_parse_years() {
        let i = parse_interval_string("1 year").unwrap();
        assert_eq!(i, Interval::from_months(12));
    }

    #[test]
    fn test_parse_decades() {
        let i = parse_interval_string("1 decade").unwrap();
        assert_eq!(i, Interval::from_months(120));
    }

    #[test]
    fn test_parse_centuries() {
        let i = parse_interval_string("2 centuries").unwrap();
        assert_eq!(i, Interval::from_months(2400));
    }

    #[test]
    fn test_parse_millennium() {
        let i = parse_interval_string("1 millennium").unwrap();
        assert_eq!(i, Interval::from_months(12000));
    }

    // Compound
    #[test]
    fn test_parse_compound_days_hours() {
        let i = parse_interval_string("1 day 2 hours").unwrap();
        assert_eq!(i, Interval::new(0, 1, 2 * 3_600_000_000_000));
    }

    #[test]
    fn test_parse_compound_year_month_day() {
        let i = parse_interval_string("1 year 3 months 4 days").unwrap();
        assert_eq!(i, Interval::new(15, 4, 0));
    }

    #[test]
    fn test_parse_with_and() {
        let a = parse_interval_string("2 hours and 30 minutes").unwrap();
        let b = parse_interval_string("2 hours 30 minutes").unwrap();
        assert_eq!(a, b);
    }

    // Negative
    #[test]
    fn test_parse_negative_days() {
        let i = parse_interval_string("-2 days").unwrap();
        assert_eq!(i, Interval::from_days(-2));
    }

    // Colon form
    #[test]
    fn test_parse_colon_hh_mm_ss() {
        let i = parse_interval_string("02:30:00").unwrap();
        assert_eq!(
            i,
            Interval::from_nanoseconds(2 * 3_600_000_000_000 + 30 * 60_000_000_000)
        );
    }

    #[test]
    fn test_parse_colon_with_fraction() {
        let i = parse_interval_string("00:00:00.5").unwrap();
        assert_eq!(i, Interval::from_nanoseconds(500_000_000));
    }

    #[test]
    fn test_parse_colon_with_nanoseconds() {
        let i = parse_interval_string("00:00:00.000000500").unwrap();
        assert_eq!(i, Interval::from_nanoseconds(500));
    }

    #[test]
    fn test_parse_colon_negative() {
        let i = parse_interval_string("-02:30:00").unwrap();
        assert_eq!(
            i,
            Interval::from_nanoseconds(-(2 * 3_600_000_000_000 + 30 * 60_000_000_000))
        );
    }

    #[test]
    fn test_parse_compound_with_colon() {
        let i = parse_interval_string("1 day 02:30:00").unwrap();
        assert_eq!(
            i,
            Interval::new(0, 1, 2 * 3_600_000_000_000 + 30 * 60_000_000_000)
        );
    }

    #[test]
    fn test_parse_full_compound() {
        let i = parse_interval_string("1 year 2 months 3 days 04:05:06.789").unwrap();
        let expected_nanos =
            4 * 3_600_000_000_000 + 5 * 60_000_000_000 + 6 * 1_000_000_000 + 789_000_000;
        assert_eq!(i, Interval::new(14, 3, expected_nanos));
    }

    // Fractional calendar units should error
    #[test]
    fn test_parse_fractional_month_error() {
        assert!(parse_interval_string("1.5 months").is_err());
    }

    #[test]
    fn test_parse_fractional_year_error() {
        assert!(parse_interval_string("2.5 years").is_err());
    }

    // Invalid inputs
    #[test]
    fn test_parse_empty_error() {
        assert!(parse_interval_string("").is_err());
        assert!(parse_interval_string("   ").is_err());
    }

    #[test]
    fn test_parse_bare_number_error() {
        assert!(parse_interval_string("5").is_err());
    }

    #[test]
    fn test_parse_unknown_unit_error() {
        assert!(parse_interval_string("5 foobars").is_err());
    }

    // Calendar arithmetic
    #[test]
    fn test_is_leap() {
        assert!(is_leap(2000));
        assert!(is_leap(2024));
        assert!(!is_leap(1900));
        assert!(!is_leap(2023));
        assert!(is_leap(2400));
    }

    #[test]
    fn test_days_in_month() {
        assert_eq!(days_in_month(2024, 1), 31);
        assert_eq!(days_in_month(2024, 2), 29); // leap
        assert_eq!(days_in_month(2023, 2), 28);
        assert_eq!(days_in_month(2024, 4), 30);
        assert_eq!(days_in_month(2024, 12), 31);
    }

    #[test]
    fn test_ymd_days_roundtrip() {
        let cases = [
            (1970, 1, 1),
            (2024, 2, 29),
            (2000, 12, 31),
            (1999, 6, 15),
            (1900, 3, 1),
        ];
        for (y, m, d) in cases {
            let days = days_from_ymd(y, m, d);
            let (y2, m2, d2) = ymd_from_days(days);
            assert_eq!((y, m, d), (y2, m2, d2), "roundtrip failed for {y}-{m}-{d}");
        }
    }

    #[test]
    fn test_add_months_simple() {
        // 2024-01-15 + 1 month = 2024-02-15
        let ts = days_from_ymd(2024, 1, 15) as i64 * 86_400_000_000;
        let result = Interval::from_months(1).add_to_timestamp_micros(ts);
        let (y, m, d, _) = decompose_timestamp_micros(result);
        assert_eq!((y, m, d), (2024, 2, 15));
    }

    #[test]
    fn test_add_month_clamps_jan31_to_feb29() {
        // 2024-01-31 + 1 month = 2024-02-29 (leap year clamping)
        let ts = days_from_ymd(2024, 1, 31) as i64 * 86_400_000_000;
        let result = Interval::from_months(1).add_to_timestamp_micros(ts);
        let (y, m, d, _) = decompose_timestamp_micros(result);
        assert_eq!((y, m, d), (2024, 2, 29));
    }

    #[test]
    fn test_add_month_clamps_jan31_to_feb28_non_leap() {
        // 2023-01-31 + 1 month = 2023-02-28 (non-leap)
        let ts = days_from_ymd(2023, 1, 31) as i64 * 86_400_000_000;
        let result = Interval::from_months(1).add_to_timestamp_micros(ts);
        let (y, m, d, _) = decompose_timestamp_micros(result);
        assert_eq!((y, m, d), (2023, 2, 28));
    }

    #[test]
    fn test_add_year_clamps_leap_day() {
        // 2024-02-29 + 1 year = 2025-02-28
        let ts = days_from_ymd(2024, 2, 29) as i64 * 86_400_000_000;
        let result = Interval::from_months(12).add_to_timestamp_micros(ts);
        let (y, m, d, _) = decompose_timestamp_micros(result);
        assert_eq!((y, m, d), (2025, 2, 28));
    }

    #[test]
    fn test_add_days_across_month_boundary() {
        // 2024-01-31 + 1 day = 2024-02-01
        let ts = days_from_ymd(2024, 1, 31) as i64 * 86_400_000_000;
        let result = Interval::from_days(1).add_to_timestamp_micros(ts);
        let (y, m, d, _) = decompose_timestamp_micros(result);
        assert_eq!((y, m, d), (2024, 2, 1));
    }

    #[test]
    fn test_add_large_number_of_months() {
        // 2024-01-15 + 25 months = 2026-02-15
        let ts = days_from_ymd(2024, 1, 15) as i64 * 86_400_000_000;
        let result = Interval::from_months(25).add_to_timestamp_micros(ts);
        let (y, m, d, _) = decompose_timestamp_micros(result);
        assert_eq!((y, m, d), (2026, 2, 15));
    }

    #[test]
    fn test_subtract_months() {
        // 2024-03-15 - 2 months = 2024-01-15
        let ts = days_from_ymd(2024, 3, 15) as i64 * 86_400_000_000;
        let result = Interval::from_months(2).subtract_from_timestamp_micros(ts);
        let (y, m, d, _) = decompose_timestamp_micros(result);
        assert_eq!((y, m, d), (2024, 1, 15));
    }

    #[test]
    fn test_negate() {
        let i = Interval::new(1, 2, 3);
        let neg = i.negate();
        assert_eq!(neg, Interval::new(-1, -2, -3));
    }

    #[test]
    fn test_field_add() {
        let a = Interval::new(1, 2, 3);
        let b = Interval::new(4, 5, 6);
        assert_eq!(a.add(b), Interval::new(5, 7, 9));
    }

    #[test]
    fn test_multiply_by() {
        let i = Interval::new(1, 2, 3);
        let m = i.multiply_by(5);
        assert_eq!(m, Interval::new(5, 10, 15));
    }

    // Ordering
    #[test]
    fn test_ordering_simple() {
        let hour = Interval::from_nanoseconds(3_600_000_000_000);
        let day = Interval::from_days(1);
        assert!(hour < day);
    }

    #[test]
    fn test_ordering_approx_month_vs_days() {
        // 1 month approximated as 30 days
        let month = Interval::from_months(1);
        let thirty_days = Interval::from_days(30);
        assert_eq!(month.cmp(&thirty_days), std::cmp::Ordering::Equal);
    }

    #[test]
    fn test_equality_strict_not_approximate() {
        let month = Interval::from_months(1);
        let thirty_days = Interval::from_days(30);
        // Strict equality: months != days even if they approximate the same
        assert_ne!(month, thirty_days);
    }

    // Display
    #[test]
    fn test_display_zero() {
        let s = Interval::ZERO.to_string();
        assert_eq!(s, "00:00:00");
    }

    #[test]
    fn test_display_with_components() {
        let i = Interval::new(14, 3, 4 * 3_600_000_000_000);
        let s = i.to_string();
        assert!(s.contains("1 year"));
        assert!(s.contains("2 mons"));
        assert!(s.contains("3 days"));
        assert!(s.contains("04:00:00"));
    }

    #[test]
    fn test_display_fractional_seconds() {
        let i = Interval::from_nanoseconds(500_000_000); // 0.5 second
        let s = i.to_string();
        assert!(s.contains(".5"));
    }

    // ----- End-to-end correctness scenarios from the plan -----

    fn ts_micros(y: i32, m: u32, d: u32) -> i64 {
        (days_from_ymd(y, m, d) as i64) * 86_400_000_000
    }

    #[test]
    fn test_plan_leap_year_plus_year_from_feb29() {
        // INTERVAL '1 year' applied to 2024-02-29 should produce 2025-02-28
        let ts = ts_micros(2024, 2, 29);
        let result = Interval::from_months(12).add_to_timestamp_micros(ts);
        let (y, m, d) = ymd_from_days((result / 86_400_000_000) as i32);
        assert_eq!((y, m, d), (2025, 2, 28));
    }

    #[test]
    fn test_plan_month_from_jan31_leap_year() {
        // INTERVAL '1 month' applied to 2024-01-31 should produce 2024-02-29 (leap year)
        let ts = ts_micros(2024, 1, 31);
        let result = Interval::from_months(1).add_to_timestamp_micros(ts);
        let (y, m, d) = ymd_from_days((result / 86_400_000_000) as i32);
        assert_eq!((y, m, d), (2024, 2, 29));
    }

    #[test]
    fn test_plan_month_from_jan31_non_leap() {
        // INTERVAL '1 month' applied to 2023-01-31 should produce 2023-02-28
        let ts = ts_micros(2023, 1, 31);
        let result = Interval::from_months(1).add_to_timestamp_micros(ts);
        let (y, m, d) = ymd_from_days((result / 86_400_000_000) as i32);
        assert_eq!((y, m, d), (2023, 2, 28));
    }

    #[test]
    fn test_plan_compound_parse_full() {
        // INTERVAL '1 year 3 months 2 days 4 hours' exact component check
        let i = parse_interval_string("1 year 3 months 2 days 4 hours").unwrap();
        assert_eq!(i.months, 15);
        assert_eq!(i.days, 2);
        assert_eq!(i.nanoseconds, 4 * 3_600_000_000_000);
    }

    #[test]
    fn test_plan_pg_format_roundtrip() {
        // An interval produced by Display should be re-parseable.
        let original = Interval::new(14, 3, 4 * 3_600_000_000_000);
        let text = original.to_string();
        // Display produces "1 year 2 mons 3 days 04:00:00" - our parser handles this exactly.
        let parsed = parse_interval_string(&text).unwrap();
        assert_eq!(parsed, original);
    }

    #[test]
    fn test_plan_serialization_roundtrip_with_neg_components() {
        let original = Interval::new(-5, -12, -3_600_000_000_000);
        let bytes = original.to_le_bytes();
        assert_eq!(bytes.len(), 16);
        let restored = Interval::from_le_bytes(&bytes);
        assert_eq!(original, restored);
    }
}
