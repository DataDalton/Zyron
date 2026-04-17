//! Business time and fiscal calendar functions.
//!
//! Fiscal quarter/year calculations, business day arithmetic with holidays,
//! natural language date parsing ("next Tuesday", "3 days ago").
//!
//! Dates are represented as i32 days since 1970-01-01 (Unix epoch).

use zyron_common::{Result, ZyronError};

// ---------------------------------------------------------------------------
// Date arithmetic primitives
// ---------------------------------------------------------------------------

fn is_leap(year: i32) -> bool {
    (year % 4 == 0 && year % 100 != 0) || year % 400 == 0
}

fn days_in_month(year: i32, month: u32) -> i32 {
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

fn days_from_ymd(year: i32, month: u32, day: u32) -> i32 {
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
        total += days_in_month(year, m);
    }
    total + (day as i32 - 1)
}

fn ymd_from_days(days: i32) -> (i32, u32, u32) {
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
        let md = days_in_month(year, month);
        if remaining < md {
            break;
        }
        remaining -= md;
        month += 1;
    }

    let day = (remaining + 1) as u32;
    (year, month, day)
}

/// Returns the day of week (0 = Sunday, 6 = Saturday) for a given day number.
pub fn day_of_week(days: i32) -> u32 {
    // 1970-01-01 was a Thursday (day 4)
    (((days % 7) + 7 + 4) % 7) as u32
}

// ---------------------------------------------------------------------------
// Fiscal calendar
// ---------------------------------------------------------------------------

/// Returns the fiscal quarter (1-4) for a given date.
/// fy_start_month is the month the fiscal year begins (1-12).
pub fn fiscal_quarter(date_days: i32, fy_start_month: u8) -> u8 {
    let (_, month, _) = ymd_from_days(date_days);
    let start = fy_start_month as u32;
    // Calculate months since fiscal year start
    let months_since_start = (month + 12 - start) % 12;
    (months_since_start / 3 + 1) as u8
}

/// Returns the fiscal year for a given date.
pub fn fiscal_year(date_days: i32, fy_start_month: u8) -> i32 {
    let (year, month, _) = ymd_from_days(date_days);
    if month >= fy_start_month as u32 {
        year
    } else {
        year - 1
    }
}

/// Returns the week number within the fiscal year (1-53).
pub fn week_of_fiscal_year(date_days: i32, fy_start_month: u8) -> u8 {
    let fy = fiscal_year(date_days, fy_start_month);
    let fy_start = days_from_ymd(fy, fy_start_month as u32, 1);
    let days_since_fy_start = date_days - fy_start;
    (days_since_fy_start / 7 + 1) as u8
}

// ---------------------------------------------------------------------------
// Business days
// ---------------------------------------------------------------------------

/// Returns true if the given day is a business day (not weekend, not holiday).
pub fn is_business_day(date_days: i32, holidays: &[i32]) -> bool {
    let dow = day_of_week(date_days);
    if dow == 0 || dow == 6 {
        return false;
    }
    !holidays.contains(&date_days)
}

/// Returns the next business day on or after the given date.
pub fn next_business_day(date_days: i32, holidays: &[i32]) -> i32 {
    let mut current = date_days + 1;
    while !is_business_day(current, holidays) {
        current += 1;
    }
    current
}

/// Adds N business days to a date, skipping weekends and holidays.
/// N can be negative to subtract business days.
pub fn add_business_days(date_days: i32, n: i32, holidays: &[i32]) -> i32 {
    if n == 0 {
        return date_days;
    }

    let step = if n > 0 { 1 } else { -1 };
    let mut remaining = n.abs();
    let mut current = date_days;

    while remaining > 0 {
        current += step;
        if is_business_day(current, holidays) {
            remaining -= 1;
        }
    }

    current
}

/// Counts the number of business days between start and end (inclusive of both).
pub fn business_days_between(start: i32, end: i32, holidays: &[i32]) -> i32 {
    if start > end {
        return -business_days_between(end, start, holidays);
    }
    let mut count = 0;
    let mut current = start;
    while current <= end {
        if is_business_day(current, holidays) {
            count += 1;
        }
        current += 1;
    }
    count
}

// ---------------------------------------------------------------------------
// Natural language date parsing
// ---------------------------------------------------------------------------

/// Parses natural-language date strings relative to a reference date.
/// Supported: "today", "yesterday", "tomorrow", "N days/weeks/months/years ago",
/// "N days/weeks/months/years from now", "next Monday", "last Friday",
/// "beginning of month", "end of month", "beginning of year", "end of year".
pub fn parse_natural_date(text: &str, reference_date: i32) -> Result<i32> {
    let lower = text.trim().to_lowercase();

    match lower.as_str() {
        "today" | "now" => return Ok(reference_date),
        "yesterday" => return Ok(reference_date - 1),
        "tomorrow" => return Ok(reference_date + 1),
        "beginning of month" | "start of month" => {
            let (y, m, _) = ymd_from_days(reference_date);
            return Ok(days_from_ymd(y, m, 1));
        }
        "end of month" => {
            let (y, m, _) = ymd_from_days(reference_date);
            let last_day = days_in_month(y, m) as u32;
            return Ok(days_from_ymd(y, m, last_day));
        }
        "beginning of year" | "start of year" => {
            let (y, _, _) = ymd_from_days(reference_date);
            return Ok(days_from_ymd(y, 1, 1));
        }
        "end of year" => {
            let (y, _, _) = ymd_from_days(reference_date);
            return Ok(days_from_ymd(y, 12, 31));
        }
        _ => {}
    }

    // "next DAY" or "last DAY"
    if let Some(rest) = lower.strip_prefix("next ") {
        if let Some(target_dow) = parse_day_name(rest) {
            let current_dow = day_of_week(reference_date);
            let mut days_ahead = (target_dow as i32 - current_dow as i32 + 7) % 7;
            if days_ahead == 0 {
                days_ahead = 7;
            }
            return Ok(reference_date + days_ahead);
        }
    }
    if let Some(rest) = lower.strip_prefix("last ") {
        if let Some(target_dow) = parse_day_name(rest) {
            let current_dow = day_of_week(reference_date);
            let mut days_back = (current_dow as i32 - target_dow as i32 + 7) % 7;
            if days_back == 0 {
                days_back = 7;
            }
            return Ok(reference_date - days_back);
        }
    }

    // "N UNIT ago" or "N UNIT from now" or "in N UNIT"
    let parts: Vec<&str> = lower.split_whitespace().collect();
    if parts.len() >= 3 {
        let (count_str, unit_str, direction) = if parts.last() == Some(&"ago") {
            (parts[0], parts[1], -1)
        } else if parts.last() == Some(&"now") && parts.get(parts.len() - 2) == Some(&"from") {
            (parts[0], parts[1], 1)
        } else if parts[0] == "in" {
            (parts[1], parts[2], 1)
        } else {
            return Err(ZyronError::ExecutionError(format!(
                "Cannot parse natural date: {}",
                text
            )));
        };

        let count: i32 = count_str.parse().map_err(|_| {
            ZyronError::ExecutionError(format!("Invalid count: {}", count_str))
        })?;
        let count = count * direction;

        let unit_singular = unit_str.trim_end_matches('s');
        match unit_singular {
            "day" => return Ok(reference_date + count),
            "week" => return Ok(reference_date + count * 7),
            "month" => {
                let (y, m, d) = ymd_from_days(reference_date);
                let total_months = y * 12 + (m as i32 - 1) + count;
                let new_year = total_months.div_euclid(12);
                let new_month = total_months.rem_euclid(12) as u32 + 1;
                let max_day = days_in_month(new_year, new_month) as u32;
                let actual_day = d.min(max_day);
                return Ok(days_from_ymd(new_year, new_month, actual_day));
            }
            "year" => {
                let (y, m, d) = ymd_from_days(reference_date);
                let new_year = y + count;
                let max_day = days_in_month(new_year, m) as u32;
                let actual_day = d.min(max_day);
                return Ok(days_from_ymd(new_year, m, actual_day));
            }
            _ => {}
        }
    }

    Err(ZyronError::ExecutionError(format!(
        "Cannot parse natural date: {}",
        text
    )))
}

fn parse_day_name(text: &str) -> Option<u32> {
    match text.trim() {
        "sunday" | "sun" => Some(0),
        "monday" | "mon" => Some(1),
        "tuesday" | "tue" | "tues" => Some(2),
        "wednesday" | "wed" => Some(3),
        "thursday" | "thu" | "thurs" => Some(4),
        "friday" | "fri" => Some(5),
        "saturday" | "sat" => Some(6),
        _ => None,
    }
}

/// Parses a natural-language duration into an Interval.
/// Delegates to `zyron_common::parse_interval_string` which handles the full
/// range of units (picosecond through millennium) plus the PostgreSQL colon form.
/// Supported: "2 hours and 30 minutes", "1 week", "5 days", "1 year 3 months".
pub fn parse_natural_duration(text: &str) -> Result<zyron_common::Interval> {
    zyron_common::parse_interval_string(text)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ymd(y: i32, m: u32, d: u32) -> i32 {
        days_from_ymd(y, m, d)
    }

    #[test]
    fn test_day_of_week() {
        // 2024-01-01 was a Monday
        assert_eq!(day_of_week(ymd(2024, 1, 1)), 1);
        // 2024-01-07 was a Sunday
        assert_eq!(day_of_week(ymd(2024, 1, 7)), 0);
        // 2024-01-06 was a Saturday
        assert_eq!(day_of_week(ymd(2024, 1, 6)), 6);
    }

    #[test]
    fn test_fiscal_quarter_calendar_year() {
        // FY starts in January
        assert_eq!(fiscal_quarter(ymd(2024, 1, 15), 1), 1);
        assert_eq!(fiscal_quarter(ymd(2024, 4, 15), 1), 2);
        assert_eq!(fiscal_quarter(ymd(2024, 7, 15), 1), 3);
        assert_eq!(fiscal_quarter(ymd(2024, 10, 15), 1), 4);
    }

    #[test]
    fn test_fiscal_quarter_july_start() {
        // FY starts in July (e.g., US federal government)
        assert_eq!(fiscal_quarter(ymd(2024, 7, 15), 7), 1);
        assert_eq!(fiscal_quarter(ymd(2024, 10, 15), 7), 2);
        assert_eq!(fiscal_quarter(ymd(2025, 1, 15), 7), 3);
        assert_eq!(fiscal_quarter(ymd(2025, 4, 15), 7), 4);
    }

    #[test]
    fn test_fiscal_year() {
        // FY starts in April (UK-style)
        assert_eq!(fiscal_year(ymd(2024, 3, 15), 4), 2023);
        assert_eq!(fiscal_year(ymd(2024, 4, 15), 4), 2024);
    }

    #[test]
    fn test_week_of_fiscal_year() {
        let week = week_of_fiscal_year(ymd(2024, 1, 1), 1);
        assert_eq!(week, 1);
        let week = week_of_fiscal_year(ymd(2024, 1, 8), 1);
        assert_eq!(week, 2);
    }

    #[test]
    fn test_is_business_day() {
        // 2024-01-01 Monday
        assert!(is_business_day(ymd(2024, 1, 1), &[]));
        // 2024-01-06 Saturday
        assert!(!is_business_day(ymd(2024, 1, 6), &[]));
        // 2024-01-07 Sunday
        assert!(!is_business_day(ymd(2024, 1, 7), &[]));
    }

    #[test]
    fn test_is_business_day_with_holiday() {
        let holidays = vec![ymd(2024, 1, 1)];
        assert!(!is_business_day(ymd(2024, 1, 1), &holidays));
    }

    #[test]
    fn test_next_business_day() {
        // Friday -> next is Monday
        let friday = ymd(2024, 1, 5);
        let next = next_business_day(friday, &[]);
        assert_eq!(next, ymd(2024, 1, 8));
    }

    #[test]
    fn test_add_business_days() {
        // Monday + 5 business days = next Monday
        let monday = ymd(2024, 1, 1);
        let result = add_business_days(monday, 5, &[]);
        assert_eq!(result, ymd(2024, 1, 8));
    }

    #[test]
    fn test_add_business_days_weekend_skip() {
        // Friday + 1 = Monday (skip weekend)
        let friday = ymd(2024, 1, 5);
        let result = add_business_days(friday, 1, &[]);
        assert_eq!(result, ymd(2024, 1, 8));
    }

    #[test]
    fn test_add_business_days_negative() {
        let monday = ymd(2024, 1, 8);
        let result = add_business_days(monday, -1, &[]);
        assert_eq!(result, ymd(2024, 1, 5)); // Previous Friday
    }

    #[test]
    fn test_business_days_between() {
        // Week: Mon-Fri = 5 business days
        let count = business_days_between(ymd(2024, 1, 1), ymd(2024, 1, 7), &[]);
        assert_eq!(count, 5);
    }

    #[test]
    fn test_business_days_between_with_holiday() {
        let holidays = vec![ymd(2024, 1, 1)];
        let count = business_days_between(ymd(2024, 1, 1), ymd(2024, 1, 7), &holidays);
        assert_eq!(count, 4);
    }

    #[test]
    fn test_parse_natural_today() {
        let today = ymd(2024, 4, 15);
        assert_eq!(parse_natural_date("today", today).unwrap(), today);
        assert_eq!(parse_natural_date("now", today).unwrap(), today);
    }

    #[test]
    fn test_parse_natural_yesterday() {
        let today = ymd(2024, 4, 15);
        assert_eq!(parse_natural_date("yesterday", today).unwrap(), today - 1);
    }

    #[test]
    fn test_parse_natural_tomorrow() {
        let today = ymd(2024, 4, 15);
        assert_eq!(parse_natural_date("tomorrow", today).unwrap(), today + 1);
    }

    #[test]
    fn test_parse_natural_days_ago() {
        let today = ymd(2024, 4, 15);
        let result = parse_natural_date("3 days ago", today).unwrap();
        assert_eq!(result, today - 3);
    }

    #[test]
    fn test_parse_natural_days_from_now() {
        let today = ymd(2024, 4, 15);
        let result = parse_natural_date("5 days from now", today).unwrap();
        assert_eq!(result, today + 5);
    }

    #[test]
    fn test_parse_natural_in_days() {
        let today = ymd(2024, 4, 15);
        let result = parse_natural_date("in 7 days", today).unwrap();
        assert_eq!(result, today + 7);
    }

    #[test]
    fn test_parse_natural_weeks() {
        let today = ymd(2024, 4, 15);
        let result = parse_natural_date("2 weeks ago", today).unwrap();
        assert_eq!(result, today - 14);
    }

    #[test]
    fn test_parse_natural_months() {
        let today = ymd(2024, 4, 15);
        let result = parse_natural_date("2 months ago", today).unwrap();
        let (y, m, d) = ymd_from_days(result);
        assert_eq!((y, m, d), (2024, 2, 15));
    }

    #[test]
    fn test_parse_natural_next_monday() {
        // Start from Monday 2024-01-01
        let monday = ymd(2024, 1, 1);
        let result = parse_natural_date("next monday", monday).unwrap();
        assert_eq!(result, monday + 7);
    }

    #[test]
    fn test_parse_natural_next_friday() {
        let monday = ymd(2024, 1, 1);
        let result = parse_natural_date("next friday", monday).unwrap();
        assert_eq!(result, ymd(2024, 1, 5));
    }

    #[test]
    fn test_parse_natural_last_friday() {
        let tuesday = ymd(2024, 1, 2);
        let result = parse_natural_date("last friday", tuesday).unwrap();
        assert_eq!(result, ymd(2023, 12, 29));
    }

    #[test]
    fn test_parse_natural_beginning_of_month() {
        let mid = ymd(2024, 4, 15);
        let result = parse_natural_date("beginning of month", mid).unwrap();
        assert_eq!(result, ymd(2024, 4, 1));
    }

    #[test]
    fn test_parse_natural_end_of_month() {
        let mid = ymd(2024, 4, 15);
        let result = parse_natural_date("end of month", mid).unwrap();
        assert_eq!(result, ymd(2024, 4, 30));
    }

    #[test]
    fn test_parse_natural_duration() {
        let iv = parse_natural_duration("1 hour").unwrap();
        assert_eq!(iv.nanoseconds, 3_600_000_000_000);
        assert_eq!(iv.days, 0);
        assert_eq!(iv.months, 0);
    }

    #[test]
    fn test_parse_natural_duration_compound() {
        let iv = parse_natural_duration("2 hours and 30 minutes").unwrap();
        let expected = (2 * 3600 + 30 * 60) * 1_000_000_000;
        assert_eq!(iv.nanoseconds, expected);
    }

    #[test]
    fn test_parse_natural_duration_weeks() {
        let iv = parse_natural_duration("1 week").unwrap();
        assert_eq!(iv.days, 7);
        assert_eq!(iv.nanoseconds, 0);
    }

    #[test]
    fn test_parse_natural_duration_calendar() {
        // Calendar units now supported through the shared parser.
        let iv = parse_natural_duration("1 year 3 months").unwrap();
        assert_eq!(iv.months, 15);
    }
}
