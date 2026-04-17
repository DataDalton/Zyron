//! Cron expression parsing and evaluation.
//!
//! Supports standard 5-field cron (minute hour day month weekday)
//! plus @yearly/@monthly/@weekly/@daily/@hourly shortcuts.
//! Uses bitsets for each field for O(1) matching.

use zyron_common::{Result, ZyronError};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CronExpr {
    /// Bitset for minutes (0-59), bit i set = minute i matches.
    pub minute: u64,
    /// Bitset for hours (0-23).
    pub hour: u32,
    /// Bitset for day of month (1-31). Bit 0 is unused.
    pub day_of_month: u32,
    /// Bitset for month (1-12). Bit 0 is unused.
    pub month: u16,
    /// Bitset for day of week (0-6, Sunday=0).
    pub day_of_week: u8,
}

/// Parses a cron expression into an internal representation.
/// Supports: "* * * * *", shortcuts ("@yearly", "@monthly", "@weekly", "@daily", "@hourly"),
/// ranges ("1-5"), step values ("*/5"), lists ("1,3,5"), combinations.
/// Day-of-week accepts numeric (0-6, Sunday=0) or 3-letter abbreviations (SUN-SAT).
/// Month accepts numeric (1-12) or 3-letter abbreviations (JAN-DEC).
pub fn cron_parse(expr: &str) -> Result<CronExpr> {
    let trimmed = expr.trim();

    // Handle shortcuts
    let expanded = match trimmed {
        "@yearly" | "@annually" => "0 0 1 1 *",
        "@monthly" => "0 0 1 * *",
        "@weekly" => "0 0 * * 0",
        "@daily" | "@midnight" => "0 0 * * *",
        "@hourly" => "0 * * * *",
        s => s,
    };

    let fields: Vec<&str> = expanded.split_whitespace().collect();
    if fields.len() != 5 {
        return Err(ZyronError::ExecutionError(format!(
            "Cron expression must have 5 fields, got {}",
            fields.len()
        )));
    }

    let minute = parse_field(fields[0], 0, 59, None)?;
    let hour = parse_field(fields[1], 0, 23, None)? as u32;
    let day_of_month = parse_field(fields[2], 1, 31, None)? as u32;
    let month = parse_field(fields[3], 1, 12, Some(&MONTH_NAMES))? as u16;
    let day_of_week = parse_field(fields[4], 0, 6, Some(&DOW_NAMES))? as u8;

    Ok(CronExpr {
        minute,
        hour,
        day_of_month,
        month,
        day_of_week,
    })
}

const MONTH_NAMES: [(&str, u32); 12] = [
    ("JAN", 1),
    ("FEB", 2),
    ("MAR", 3),
    ("APR", 4),
    ("MAY", 5),
    ("JUN", 6),
    ("JUL", 7),
    ("AUG", 8),
    ("SEP", 9),
    ("OCT", 10),
    ("NOV", 11),
    ("DEC", 12),
];

const DOW_NAMES: [(&str, u32); 8] = [
    ("SUN", 0),
    ("MON", 1),
    ("TUE", 2),
    ("WED", 3),
    ("THU", 4),
    ("FRI", 5),
    ("SAT", 6),
    ("SUN2", 7), // Cron allows 7 for Sunday
];

fn parse_field(field: &str, min: u32, max: u32, names: Option<&[(&str, u32)]>) -> Result<u64> {
    let mut result: u64 = 0;

    for segment in field.split(',') {
        let segment = segment.trim();
        if segment.is_empty() {
            continue;
        }

        // Check for step: "*/5" or "1-20/5"
        let (range_part, step) = match segment.find('/') {
            Some(idx) => {
                let step_str = &segment[idx + 1..];
                let step_val = step_str.parse::<u32>().map_err(|_| {
                    ZyronError::ExecutionError(format!("Invalid step: {}", step_str))
                })?;
                if step_val == 0 {
                    return Err(ZyronError::ExecutionError("Step cannot be zero".into()));
                }
                (&segment[..idx], step_val)
            }
            None => (segment, 1),
        };

        let (start, end) = if range_part == "*" {
            (min, max)
        } else if let Some(dash) = range_part.find('-') {
            let s = parse_value(&range_part[..dash], names)?;
            let e = parse_value(&range_part[dash + 1..], names)?;
            (s, e)
        } else {
            let v = parse_value(range_part, names)?;
            // If step > 1 and no range, treat as "v-max"
            if step > 1 { (v, max) } else { (v, v) }
        };

        // Accept 7 as Sunday for day-of-week (max == 6)
        let is_dow_field = max == 6;
        let (check_start, check_end) = (start, end);

        if check_start < min
            || (check_end > max && !(is_dow_field && check_end == 7))
            || check_start > check_end
        {
            return Err(ZyronError::ExecutionError(format!(
                "Value out of range: {} (expected {}..={})",
                segment, min, max
            )));
        }

        let mut i = start;
        while i <= end {
            // Map 7 -> 0 for Sunday in day-of-week field
            let bit = if is_dow_field && i == 7 { 0 } else { i };
            result |= 1u64 << bit;
            i += step;
        }
    }

    if result == 0 {
        return Err(ZyronError::ExecutionError(format!(
            "Empty field: {}",
            field
        )));
    }

    Ok(result)
}

fn parse_value(s: &str, names: Option<&[(&str, u32)]>) -> Result<u32> {
    let trimmed = s.trim();
    if let Ok(v) = trimmed.parse::<u32>() {
        return Ok(v);
    }
    if let Some(table) = names {
        let upper = trimmed.to_uppercase();
        for (name, val) in table {
            if *name == upper.as_str() {
                return Ok(*val);
            }
        }
    }
    Err(ZyronError::ExecutionError(format!("Invalid value: {}", s)))
}

/// Returns the next time the cron expression matches after the given time (epoch microseconds).
pub fn cron_next(expr: &CronExpr, after_micros: i64) -> Result<i64> {
    // Add 1 minute to avoid matching the same time
    let mut dt = DateTime::from_micros(after_micros)
        .add_minutes(1)
        .zero_seconds();

    // Search up to 4 years ahead (worst case: Feb 29 only, leap year alignment)
    for _ in 0..(4 * 366 * 24 * 60) {
        if cron_matches_datetime(expr, &dt) {
            return Ok(dt.to_micros());
        }
        dt = dt.add_minutes(1);
    }

    Err(ZyronError::ExecutionError(
        "Could not find next cron match within 4 years".into(),
    ))
}

/// Returns the previous time the cron expression matched before the given time.
pub fn cron_prev(expr: &CronExpr, before_micros: i64) -> Result<i64> {
    let mut dt = DateTime::from_micros(before_micros)
        .sub_minutes(1)
        .zero_seconds();

    for _ in 0..(4 * 366 * 24 * 60) {
        if cron_matches_datetime(expr, &dt) {
            return Ok(dt.to_micros());
        }
        dt = dt.sub_minutes(1);
    }

    Err(ZyronError::ExecutionError(
        "Could not find previous cron match within 4 years".into(),
    ))
}

/// Returns true if the cron expression matches the given timestamp.
pub fn cron_matches(expr: &CronExpr, timestamp_micros: i64) -> bool {
    let dt = DateTime::from_micros(timestamp_micros);
    cron_matches_datetime(expr, &dt)
}

fn cron_matches_datetime(expr: &CronExpr, dt: &DateTime) -> bool {
    let minute_match = (expr.minute >> dt.minute) & 1 == 1;
    let hour_match = (expr.hour >> dt.hour) & 1 == 1;
    let dom_match = (expr.day_of_month >> dt.day) & 1 == 1;
    let month_match = (expr.month >> dt.month) & 1 == 1;
    let dow_match = (expr.day_of_week >> dt.day_of_week()) & 1 == 1;

    // Cron day-of-month and day-of-week use OR logic when both are restricted
    let day_match = if is_dom_restricted(expr) && is_dow_restricted(expr) {
        dom_match || dow_match
    } else {
        dom_match && dow_match
    };

    minute_match && hour_match && month_match && day_match
}

fn is_dom_restricted(expr: &CronExpr) -> bool {
    // All bits 1-31 set means unrestricted
    expr.day_of_month != 0xFFFFFFFE
}

fn is_dow_restricted(expr: &CronExpr) -> bool {
    // All bits 0-6 set means unrestricted
    expr.day_of_week != 0x7F
}

/// Returns all cron match times between start and end (inclusive of start, exclusive of end).
pub fn cron_between(expr: &CronExpr, start_micros: i64, end_micros: i64) -> Result<Vec<i64>> {
    let mut results = Vec::new();
    let mut current = start_micros - 60_000_000; // Back up one minute

    loop {
        let next = cron_next(expr, current)?;
        if next >= end_micros {
            break;
        }
        results.push(next);
        current = next;

        // Safety limit
        if results.len() > 1_000_000 {
            return Err(ZyronError::ExecutionError(
                "Too many cron matches in range".into(),
            ));
        }
    }

    Ok(results)
}

/// Returns a human-readable description of the cron expression.
pub fn cron_human_readable(expr: &CronExpr) -> String {
    let minute = bitset_to_description(expr.minute, 0, 59);
    let hour = bitset_to_description(expr.hour as u64, 0, 23);
    let dom = bitset_to_description(expr.day_of_month as u64, 1, 31);
    let month = bitset_to_description(expr.month as u64, 1, 12);
    let dow = bitset_to_description(expr.day_of_week as u64, 0, 6);

    format!(
        "Minute: {}, Hour: {}, Day: {}, Month: {}, Weekday: {}",
        minute, hour, dom, month, dow
    )
}

fn bitset_to_description(bits: u64, min: u32, max: u32) -> String {
    let count = (min..=max).filter(|&i| (bits >> i) & 1 == 1).count();
    if count == (max - min + 1) as usize {
        return "*".to_string();
    }
    let values: Vec<String> = (min..=max)
        .filter(|&i| (bits >> i) & 1 == 1)
        .map(|i| i.to_string())
        .collect();
    values.join(",")
}

// ---------------------------------------------------------------------------
// Minimal DateTime for cron iteration (no external dep)
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy)]
struct DateTime {
    year: i32,
    month: u32,
    day: u32,
    hour: u32,
    minute: u32,
    second: u32,
    micros: i64,
}

impl DateTime {
    fn from_micros(micros: i64) -> Self {
        let seconds = micros.div_euclid(1_000_000);
        let sub_micros = micros.rem_euclid(1_000_000);

        // Split into date and time of day
        let days = seconds.div_euclid(86400);
        let time_secs = seconds.rem_euclid(86400) as u32;

        let hour = time_secs / 3600;
        let minute = (time_secs % 3600) / 60;
        let second = time_secs % 60;

        let (year, month, day) = days_to_ymd(days as i64);

        Self {
            year,
            month,
            day,
            hour,
            minute,
            second,
            micros: sub_micros,
        }
    }

    fn to_micros(&self) -> i64 {
        let days = ymd_to_days(self.year, self.month, self.day);
        let seconds = days * 86400
            + (self.hour as i64) * 3600
            + (self.minute as i64) * 60
            + (self.second as i64);
        seconds * 1_000_000 + self.micros
    }

    fn zero_seconds(mut self) -> Self {
        self.second = 0;
        self.micros = 0;
        self
    }

    fn add_minutes(mut self, n: i64) -> Self {
        let total_min = self.minute as i64 + n;
        let extra_hours = total_min.div_euclid(60);
        self.minute = total_min.rem_euclid(60) as u32;

        let total_hours = self.hour as i64 + extra_hours;
        let extra_days = total_hours.div_euclid(24);
        self.hour = total_hours.rem_euclid(24) as u32;

        if extra_days != 0 {
            let days = ymd_to_days(self.year, self.month, self.day) + extra_days;
            let (y, m, d) = days_to_ymd(days);
            self.year = y;
            self.month = m;
            self.day = d;
        }
        self
    }

    fn sub_minutes(self, n: i64) -> Self {
        self.add_minutes(-n)
    }

    fn day_of_week(&self) -> u32 {
        // Zeller's congruence or simpler: compute from days since epoch
        let days = ymd_to_days(self.year, self.month, self.day);
        // 1970-01-01 was a Thursday (day 4)
        let dow = ((days % 7) + 7 + 4) % 7;
        dow as u32
    }
}

fn is_leap(year: i32) -> bool {
    (year % 4 == 0 && year % 100 != 0) || year % 400 == 0
}

fn days_in_month(year: i32, month: u32) -> u32 {
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

fn ymd_to_days(year: i32, month: u32, day: u32) -> i64 {
    // Days since 1970-01-01
    let mut total: i64 = 0;
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
        total += days_in_month(year, m) as i64;
    }
    total += (day - 1) as i64;
    total
}

fn days_to_ymd(days: i64) -> (i32, u32, u32) {
    let mut year: i32 = 1970;
    let mut remaining = days;

    if remaining >= 0 {
        loop {
            let year_days = if is_leap(year) { 366 } else { 365 } as i64;
            if remaining < year_days {
                break;
            }
            remaining -= year_days;
            year += 1;
        }
    } else {
        while remaining < 0 {
            year -= 1;
            let year_days = if is_leap(year) { 366 } else { 365 } as i64;
            remaining += year_days;
        }
    }

    let mut month = 1u32;
    while month <= 12 {
        let md = days_in_month(year, month) as i64;
        if remaining < md {
            break;
        }
        remaining -= md;
        month += 1;
    }

    let day = (remaining + 1) as u32;
    (year, month, day)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_basic() {
        let expr = cron_parse("0 9 * * MON").unwrap();
        // 0th minute, 9th hour, Monday
        assert_eq!(expr.minute & 1, 1);
        assert_eq!((expr.hour >> 9) & 1, 1);
        assert_eq!((expr.day_of_week >> 1) & 1, 1);
    }

    #[test]
    fn test_parse_shortcuts() {
        assert!(cron_parse("@yearly").is_ok());
        assert!(cron_parse("@monthly").is_ok());
        assert!(cron_parse("@weekly").is_ok());
        assert!(cron_parse("@daily").is_ok());
        assert!(cron_parse("@hourly").is_ok());
    }

    #[test]
    fn test_parse_range() {
        let expr = cron_parse("0 9-17 * * *").unwrap();
        for h in 9..=17 {
            assert_eq!((expr.hour >> h) & 1, 1);
        }
        assert_eq!((expr.hour >> 8) & 1, 0);
        assert_eq!((expr.hour >> 18) & 1, 0);
    }

    #[test]
    fn test_parse_step() {
        let expr = cron_parse("*/15 * * * *").unwrap();
        // Minutes 0, 15, 30, 45
        assert_eq!((expr.minute >> 0) & 1, 1);
        assert_eq!((expr.minute >> 15) & 1, 1);
        assert_eq!((expr.minute >> 30) & 1, 1);
        assert_eq!((expr.minute >> 45) & 1, 1);
        assert_eq!((expr.minute >> 1) & 1, 0);
    }

    #[test]
    fn test_parse_list() {
        let expr = cron_parse("0 8,12,16 * * *").unwrap();
        assert_eq!((expr.hour >> 8) & 1, 1);
        assert_eq!((expr.hour >> 12) & 1, 1);
        assert_eq!((expr.hour >> 16) & 1, 1);
        assert_eq!((expr.hour >> 9) & 1, 0);
    }

    #[test]
    fn test_parse_named_month() {
        let expr = cron_parse("0 0 1 JAN *").unwrap();
        assert_eq!((expr.month >> 1) & 1, 1);
    }

    #[test]
    fn test_parse_named_weekday() {
        let expr = cron_parse("0 9 * * MON-FRI").unwrap();
        for d in 1..=5 {
            assert_eq!((expr.day_of_week >> d) & 1, 1);
        }
    }

    #[test]
    fn test_parse_invalid() {
        assert!(cron_parse("invalid").is_err());
        assert!(cron_parse("1 2 3").is_err()); // too few fields
        assert!(cron_parse("60 * * * *").is_err()); // minute out of range
        assert!(cron_parse("* * * 13 *").is_err()); // month out of range
    }

    #[test]
    fn test_matches_everything() {
        let expr = cron_parse("* * * * *").unwrap();
        assert!(cron_matches(&expr, 0));
        assert!(cron_matches(&expr, 1_700_000_000_000_000));
    }

    #[test]
    fn test_next_simple() {
        let expr = cron_parse("0 0 * * *").unwrap(); // Midnight
        // 2024-01-01 00:00:00 UTC
        let start = 1_704_067_200_000_000i64;
        let next = cron_next(&expr, start).unwrap();
        // Next midnight is 24 hours later
        assert_eq!(next, start + 86400 * 1_000_000);
    }

    #[test]
    fn test_prev() {
        let expr = cron_parse("0 0 * * *").unwrap();
        // 2024-01-02 12:00:00 UTC
        let now = 1_704_196_800_000_000i64;
        let prev = cron_prev(&expr, now).unwrap();
        // Most recent midnight was earlier today
        assert!(prev < now);
        assert!(now - prev < 86400 * 1_000_000);
    }

    #[test]
    fn test_next_prev_roundtrip() {
        let expr = cron_parse("*/15 * * * *").unwrap();
        let now = 1_704_067_200_000_000i64;
        let next = cron_next(&expr, now).unwrap();
        let back = cron_prev(&expr, next + 60_000_000).unwrap();
        assert_eq!(back, next);
    }

    #[test]
    fn test_between() {
        let expr = cron_parse("0 * * * *").unwrap(); // top of every hour
        let start = 1_704_067_200_000_000i64; // 2024-01-01 00:00:00
        let end = start + 86400 * 1_000_000; // 24 hours later
        let matches = cron_between(&expr, start, end).unwrap();
        // 24 hourly matches in a day
        assert_eq!(matches.len(), 24);
    }

    #[test]
    fn test_human_readable() {
        let expr = cron_parse("0 9 * * MON").unwrap();
        let desc = cron_human_readable(&expr);
        assert!(desc.contains("Minute"));
        assert!(desc.contains("Hour"));
    }

    #[test]
    fn test_sunday_7() {
        // Cron allows 7 for Sunday (same as 0)
        let expr = cron_parse("0 0 * * 7").unwrap();
        assert_eq!((expr.day_of_week >> 0) & 1, 1);
    }
}
