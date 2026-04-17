//! Number, currency, byte, duration, percentage, and ordinal formatting.
//! Unit conversion between compatible measurement systems.

use zyron_common::{Result, ZyronError};

/// Formats a number with thousands separators.
/// locale: "en" for comma separator and period decimal,
/// "de" for period separator and comma decimal.
pub fn format_number(value: f64, locale: &str) -> String {
    let (thousands_sep, decimal_sep) = separator_chars(locale);
    let is_negative = value < 0.0;
    let abs_val = value.abs();

    let integer_part = abs_val.trunc() as u64;
    let fractional = abs_val - abs_val.trunc();

    let int_str = format_integer_with_sep(integer_part, thousands_sep);

    let result = if fractional.abs() < 1e-10 {
        int_str
    } else {
        // Use ryu for precise float formatting, then extract fractional digits
        let full = format!("{}", abs_val);
        let frac_digits = if let Some(dot_pos) = full.find('.') {
            let raw = &full[dot_pos + 1..];
            raw.trim_end_matches('0')
        } else {
            ""
        };
        if frac_digits.is_empty() {
            int_str
        } else {
            format!("{}{}{}", int_str, decimal_sep, frac_digits)
        }
    };

    if is_negative {
        format!("-{}", result)
    } else {
        result
    }
}

fn separator_chars(locale: &str) -> (char, char) {
    match locale {
        "de" | "de-DE" | "fr" | "fr-FR" | "es" | "es-ES" | "pt" | "pt-BR" => ('.', ','),
        _ => (',', '.'),
    }
}

fn format_integer_with_sep(val: u64, sep: char) -> String {
    let s = val.to_string();
    if s.len() <= 3 {
        return s;
    }
    let mut result = String::with_capacity(s.len() + s.len() / 3);
    for (i, c) in s.chars().enumerate() {
        if i > 0 && (s.len() - i) % 3 == 0 {
            result.push(sep);
        }
        result.push(c);
    }
    result
}

/// Formats a value as currency with the given currency code and locale.
pub fn format_currency(value: f64, currency: &str, locale: &str) -> String {
    let (symbol, decimals, symbol_before) = currency_info(currency);
    let (thousands_sep, decimal_sep) = separator_chars(locale);

    let is_negative = value < 0.0;
    let abs_val = value.abs();

    let factor = 10f64.powi(decimals as i32);
    let rounded = (abs_val * factor).round() / factor;

    let integer_part = rounded.trunc() as u64;
    let fractional = ((rounded - rounded.trunc()) * factor).round() as u64;

    let int_str = format_integer_with_sep(integer_part, thousands_sep);
    let amount = if decimals > 0 {
        format!(
            "{}{}{}",
            int_str,
            decimal_sep,
            format!("{:0>width$}", fractional, width = decimals as usize)
        )
    } else {
        int_str
    };

    let formatted = if symbol_before {
        format!("{}{}", symbol, amount)
    } else {
        format!("{} {}", amount, symbol)
    };

    if is_negative {
        format!("-{}", formatted)
    } else {
        formatted
    }
}

fn currency_info(code: &str) -> (&str, u8, bool) {
    // Returns (symbol, decimal_places, symbol_before_amount)
    match code.to_uppercase().as_str() {
        "USD" => ("$", 2, true),
        "EUR" => ("\u{20AC}", 2, true),
        "GBP" => ("\u{00A3}", 2, true),
        "JPY" => ("\u{00A5}", 0, true),
        "CNY" | "RMB" => ("\u{00A5}", 2, true),
        "KRW" => ("\u{20A9}", 0, true),
        "INR" => ("\u{20B9}", 2, true),
        "BRL" => ("R$", 2, true),
        "CAD" => ("CA$", 2, true),
        "AUD" => ("A$", 2, true),
        "CHF" => ("CHF", 2, false),
        "SEK" | "NOK" | "DKK" => ("kr", 2, false),
        "PLN" => ("z\u{0142}", 2, false),
        "TRY" => ("\u{20BA}", 2, true),
        "RUB" => ("\u{20BD}", 2, false),
        "MXN" => ("MX$", 2, true),
        "BHD" | "KWD" | "OMR" => (code, 3, true),
        _ => (code, 2, true),
    }
}

/// Formats a byte count as a human-readable string (e.g., "1.5 KB").
/// Uses decimal (SI) prefixes: KB = 1000, MB = 1000000, etc.
pub fn format_bytes(bytes: u64) -> String {
    const UNITS: &[&str] = &["B", "KB", "MB", "GB", "TB", "PB", "EB"];

    if bytes == 0 {
        return "0 B".to_string();
    }

    let mut value = bytes as f64;
    let mut unit_idx = 0;

    while value >= 1000.0 && unit_idx < UNITS.len() - 1 {
        value /= 1000.0;
        unit_idx += 1;
    }

    if unit_idx == 0 {
        format!("{} B", bytes)
    } else if value >= 100.0 {
        format!("{:.0} {}", value, UNITS[unit_idx])
    } else if value >= 10.0 {
        format!("{:.1} {}", value, UNITS[unit_idx])
    } else {
        // Trim trailing zeros
        let s = format!("{:.2}", value);
        let trimmed = s.trim_end_matches('0').trim_end_matches('.');
        format!("{} {}", trimmed, UNITS[unit_idx])
    }
}

/// Formats a duration in seconds as a human-readable string (e.g., "1h 2m 3s").
pub fn format_duration(seconds: f64) -> String {
    if seconds < 0.0 {
        return format!("-{}", format_duration(-seconds));
    }

    let total_secs = seconds.round() as u64;

    if total_secs == 0 {
        return "0s".to_string();
    }

    let days = total_secs / 86400;
    let hours = (total_secs % 86400) / 3600;
    let mins = (total_secs % 3600) / 60;
    let secs = total_secs % 60;

    let mut parts = Vec::new();
    if days > 0 {
        parts.push(format!("{}d", days));
    }
    if hours > 0 {
        parts.push(format!("{}h", hours));
    }
    if mins > 0 {
        parts.push(format!("{}m", mins));
    }
    if secs > 0 {
        parts.push(format!("{}s", secs));
    }

    parts.join(" ")
}

/// Formats a fraction as a percentage string (e.g., 0.1523 -> "15.23%").
pub fn format_percentage(value: f64, decimals: u32) -> String {
    let pct = value * 100.0;
    format!("{:.prec$}%", pct, prec = decimals as usize)
}

/// Formats a number as an ordinal string (e.g., 1 -> "1st", 2 -> "2nd").
pub fn format_ordinal(n: i64) -> String {
    let abs_n = n.unsigned_abs();
    let suffix = match (abs_n % 100, abs_n % 10) {
        (11..=13, _) => "th",
        (_, 1) => "st",
        (_, 2) => "nd",
        (_, 3) => "rd",
        _ => "th",
    };
    format!("{}{}", n, suffix)
}

/// Parses a locale-formatted number string back to f64.
/// Handles thousands separators and locale-specific decimal separators.
pub fn parse_number(text: &str, locale: &str) -> Result<f64> {
    let (thousands_sep, decimal_sep) = separator_chars(locale);
    let cleaned: String = text
        .chars()
        .filter(|&c| c != thousands_sep && !c.is_whitespace())
        .map(|c| if c == decimal_sep { '.' } else { c })
        .collect();

    // Remove currency symbols and whitespace
    let numeric: String = cleaned
        .trim_start_matches(|c: char| !c.is_ascii_digit() && c != '-' && c != '.')
        .trim_end_matches(|c: char| !c.is_ascii_digit())
        .to_string();

    numeric.parse::<f64>().map_err(|e| {
        ZyronError::ExecutionError(format!("Cannot parse '{}' as number: {}", text, e))
    })
}

/// Converts a value between compatible units.
/// Supported categories: length, mass, temperature, speed, volume, area, time, data.
pub fn convert_units(value: f64, from: &str, to: &str) -> Result<f64> {
    let from_lower = from.to_lowercase();
    let to_lower = to.to_lowercase();

    // Temperature special handling (not linear conversion)
    if is_temp_unit(&from_lower) && is_temp_unit(&to_lower) {
        return convert_temperature(value, &from_lower, &to_lower);
    }

    let from_factor = unit_to_si(&from_lower)?;
    let to_factor = unit_to_si(&to_lower)?;

    // Check that units are in the same dimension
    let from_dim = unit_dimension(&from_lower);
    let to_dim = unit_dimension(&to_lower);
    if from_dim != to_dim {
        return Err(ZyronError::ExecutionError(format!(
            "Cannot convert between {} ({}) and {} ({})",
            from, from_dim, to, to_dim
        )));
    }

    Ok(value * from_factor / to_factor)
}

fn is_temp_unit(unit: &str) -> bool {
    matches!(unit, "c" | "celsius" | "f" | "fahrenheit" | "k" | "kelvin")
}

fn convert_temperature(value: f64, from: &str, to: &str) -> Result<f64> {
    // Convert to Kelvin first
    let kelvin = match from {
        "c" | "celsius" => value + 273.15,
        "f" | "fahrenheit" => (value - 32.0) * 5.0 / 9.0 + 273.15,
        "k" | "kelvin" => value,
        _ => {
            return Err(ZyronError::ExecutionError(format!(
                "Unknown temperature unit: {}",
                from
            )));
        }
    };
    // Convert from Kelvin to target
    match to {
        "c" | "celsius" => Ok(kelvin - 273.15),
        "f" | "fahrenheit" => Ok((kelvin - 273.15) * 9.0 / 5.0 + 32.0),
        "k" | "kelvin" => Ok(kelvin),
        _ => Err(ZyronError::ExecutionError(format!(
            "Unknown temperature unit: {}",
            to
        ))),
    }
}

fn unit_dimension(unit: &str) -> &'static str {
    match unit {
        "m" | "meter" | "meters" | "km" | "kilometer" | "kilometers" | "cm" | "centimeter"
        | "centimeters" | "mm" | "millimeter" | "millimeters" | "mi" | "mile" | "miles" | "yd"
        | "yard" | "yards" | "ft" | "foot" | "feet" | "in" | "inch" | "inches" | "nm"
        | "nautical_mile" | "nautical_miles" => "length",

        "kg" | "kilogram" | "kilograms" | "g" | "gram" | "grams" | "mg" | "milligram"
        | "milligrams" | "lb" | "lbs" | "pound" | "pounds" | "oz" | "ounce" | "ounces" | "ton"
        | "tons" | "tonne" | "tonnes" => "mass",

        "l" | "liter" | "liters" | "ml" | "milliliter" | "milliliters" | "gal" | "gallon"
        | "gallons" | "qt" | "quart" | "quarts" | "pt" | "pint" | "pints" | "cup" | "cups"
        | "fl_oz" | "fluid_ounce" | "fluid_ounces" => "volume",

        "m2" | "sq_m" | "hectare" | "hectares" | "ha" | "km2" | "sq_km" | "acre" | "acres"
        | "sq_ft" | "sq_mi" => "area",

        "m/s" | "km/h" | "kph" | "mph" | "knot" | "knots" | "ft/s" => "speed",

        "s" | "second" | "seconds" | "ms" | "millisecond" | "milliseconds" | "us"
        | "microsecond" | "microseconds" | "min" | "minute" | "minutes" | "h" | "hour"
        | "hours" | "d" | "day" | "days" => "time",

        "b" | "byte" | "bytes" | "kb" | "kilobyte" | "kilobytes" | "mb" | "megabyte"
        | "megabytes" | "gb" | "gigabyte" | "gigabytes" | "tb" | "terabyte" | "terabytes"
        | "pb" | "petabyte" | "petabytes" | "kib" | "mib" | "gib" | "tib" => "data",

        _ => "unknown",
    }
}

/// Returns the conversion factor to the SI base unit for the given unit.
fn unit_to_si(unit: &str) -> Result<f64> {
    let factor = match unit {
        // Length (base: meters)
        "m" | "meter" | "meters" => 1.0,
        "km" | "kilometer" | "kilometers" => 1000.0,
        "cm" | "centimeter" | "centimeters" => 0.01,
        "mm" | "millimeter" | "millimeters" => 0.001,
        "mi" | "mile" | "miles" => 1609.344,
        "yd" | "yard" | "yards" => 0.9144,
        "ft" | "foot" | "feet" => 0.3048,
        "in" | "inch" | "inches" => 0.0254,
        "nm" | "nautical_mile" | "nautical_miles" => 1852.0,

        // Mass (base: kilograms)
        "kg" | "kilogram" | "kilograms" => 1.0,
        "g" | "gram" | "grams" => 0.001,
        "mg" | "milligram" | "milligrams" => 0.000001,
        "lb" | "lbs" | "pound" | "pounds" => 0.45359237,
        "oz" | "ounce" | "ounces" => 0.028349523125,
        "ton" | "tons" => 907.18474,  // US short ton
        "tonne" | "tonnes" => 1000.0, // metric tonne

        // Volume (base: liters)
        "l" | "liter" | "liters" => 1.0,
        "ml" | "milliliter" | "milliliters" => 0.001,
        "gal" | "gallon" | "gallons" => 3.785411784,
        "qt" | "quart" | "quarts" => 0.946352946,
        "pt" | "pint" | "pints" => 0.473176473,
        "cup" | "cups" => 0.2365882365,
        "fl_oz" | "fluid_ounce" | "fluid_ounces" => 0.0295735296,

        // Area (base: square meters)
        "m2" | "sq_m" => 1.0,
        "km2" | "sq_km" => 1_000_000.0,
        "hectare" | "hectares" | "ha" => 10_000.0,
        "acre" | "acres" => 4046.8564224,
        "sq_ft" => 0.09290304,
        "sq_mi" => 2_589_988.11,

        // Speed (base: m/s)
        "m/s" => 1.0,
        "km/h" | "kph" => 1.0 / 3.6,
        "mph" => 0.44704,
        "knot" | "knots" => 0.514444,
        "ft/s" => 0.3048,

        // Time (base: seconds)
        "s" | "second" | "seconds" => 1.0,
        "ms" | "millisecond" | "milliseconds" => 0.001,
        "us" | "microsecond" | "microseconds" => 0.000001,
        "min" | "minute" | "minutes" => 60.0,
        "h" | "hour" | "hours" => 3600.0,
        "d" | "day" | "days" => 86400.0,

        // Data (base: bytes)
        "b" | "byte" | "bytes" => 1.0,
        "kb" | "kilobyte" | "kilobytes" => 1000.0,
        "mb" | "megabyte" | "megabytes" => 1_000_000.0,
        "gb" | "gigabyte" | "gigabytes" => 1_000_000_000.0,
        "tb" | "terabyte" | "terabytes" => 1_000_000_000_000.0,
        "pb" | "petabyte" | "petabytes" => 1_000_000_000_000_000.0,
        "kib" => 1024.0,
        "mib" => 1_048_576.0,
        "gib" => 1_073_741_824.0,
        "tib" => 1_099_511_627_776.0,

        _ => {
            return Err(ZyronError::ExecutionError(format!(
                "Unknown unit: {}",
                unit
            )));
        }
    };
    Ok(factor)
}

#[cfg(test)]
mod tests {
    use super::*;

    // format_number
    #[test]
    fn test_format_number_basic() {
        assert_eq!(format_number(1234567.89, "en"), "1,234,567.89");
    }

    #[test]
    fn test_format_number_german() {
        assert_eq!(format_number(1234567.89, "de"), "1.234.567,89");
    }

    #[test]
    fn test_format_number_integer() {
        assert_eq!(format_number(42.0, "en"), "42");
    }

    #[test]
    fn test_format_number_negative() {
        assert_eq!(format_number(-1234.0, "en"), "-1,234");
    }

    #[test]
    fn test_format_number_small() {
        assert_eq!(format_number(0.5, "en"), "0.5");
    }

    // format_currency
    #[test]
    fn test_format_currency_usd() {
        assert_eq!(format_currency(1234.56, "USD", "en"), "$1,234.56");
    }

    #[test]
    fn test_format_currency_eur() {
        let result = format_currency(1234.56, "EUR", "en");
        assert!(result.contains("1,234.56"));
    }

    #[test]
    fn test_format_currency_jpy() {
        let result = format_currency(1234.0, "JPY", "en");
        assert!(result.contains("1,234"));
        assert!(!result.contains('.'));
    }

    #[test]
    fn test_format_currency_negative() {
        let result = format_currency(-19.99, "USD", "en");
        assert!(result.starts_with('-'));
        assert!(result.contains("19.99"));
    }

    // format_bytes
    #[test]
    fn test_format_bytes() {
        assert_eq!(format_bytes(0), "0 B");
        assert_eq!(format_bytes(500), "500 B");
        assert_eq!(format_bytes(1000), "1 KB");
        assert_eq!(format_bytes(1500), "1.5 KB");
        assert_eq!(format_bytes(1536), "1.54 KB");
        assert_eq!(format_bytes(1_000_000), "1 MB");
        assert_eq!(format_bytes(1_500_000_000), "1.5 GB");
    }

    // format_duration
    #[test]
    fn test_format_duration() {
        assert_eq!(format_duration(0.0), "0s");
        assert_eq!(format_duration(30.0), "30s");
        assert_eq!(format_duration(90.0), "1m 30s");
        assert_eq!(format_duration(3661.0), "1h 1m 1s");
        assert_eq!(format_duration(86400.0), "1d");
        assert_eq!(format_duration(90061.0), "1d 1h 1m 1s");
    }

    #[test]
    fn test_format_duration_negative() {
        assert_eq!(format_duration(-60.0), "-1m");
    }

    // format_percentage
    #[test]
    fn test_format_percentage() {
        assert_eq!(format_percentage(0.1523, 2), "15.23%");
        assert_eq!(format_percentage(1.0, 0), "100%");
        assert_eq!(format_percentage(0.0, 1), "0.0%");
    }

    // format_ordinal
    #[test]
    fn test_format_ordinal() {
        assert_eq!(format_ordinal(1), "1st");
        assert_eq!(format_ordinal(2), "2nd");
        assert_eq!(format_ordinal(3), "3rd");
        assert_eq!(format_ordinal(4), "4th");
        assert_eq!(format_ordinal(11), "11th");
        assert_eq!(format_ordinal(12), "12th");
        assert_eq!(format_ordinal(13), "13th");
        assert_eq!(format_ordinal(21), "21st");
        assert_eq!(format_ordinal(22), "22nd");
        assert_eq!(format_ordinal(23), "23rd");
        assert_eq!(format_ordinal(101), "101st");
        assert_eq!(format_ordinal(111), "111th");
    }

    // parse_number
    #[test]
    fn test_parse_number_en() {
        assert!((parse_number("1,234.56", "en").unwrap() - 1234.56).abs() < 1e-10);
    }

    #[test]
    fn test_parse_number_de() {
        assert!((parse_number("1.234,56", "de").unwrap() - 1234.56).abs() < 1e-10);
    }

    #[test]
    fn test_parse_number_invalid() {
        assert!(parse_number("not_a_number", "en").is_err());
    }

    // convert_units
    #[test]
    fn test_convert_km_to_miles() {
        let result = convert_units(1.0, "km", "mi").unwrap();
        assert!((result - 0.621371).abs() < 0.001);
    }

    #[test]
    fn test_convert_lb_to_kg() {
        let result = convert_units(1.0, "lb", "kg").unwrap();
        assert!((result - 0.45359237).abs() < 0.001);
    }

    #[test]
    fn test_convert_celsius_to_fahrenheit() {
        let result = convert_units(100.0, "celsius", "fahrenheit").unwrap();
        assert!((result - 212.0).abs() < 0.01);
    }

    #[test]
    fn test_convert_celsius_to_kelvin() {
        let result = convert_units(0.0, "celsius", "kelvin").unwrap();
        assert!((result - 273.15).abs() < 0.01);
    }

    #[test]
    fn test_convert_fahrenheit_to_celsius() {
        let result = convert_units(32.0, "fahrenheit", "celsius").unwrap();
        assert!(result.abs() < 0.01);
    }

    #[test]
    fn test_convert_incompatible() {
        assert!(convert_units(1.0, "km", "kg").is_err());
    }

    #[test]
    fn test_convert_unknown_unit() {
        assert!(convert_units(1.0, "zorps", "km").is_err());
    }

    #[test]
    fn test_convert_same_unit() {
        let result = convert_units(42.0, "km", "km").unwrap();
        assert!((result - 42.0).abs() < 1e-10);
    }

    #[test]
    fn test_convert_data_units() {
        let result = convert_units(1.0, "gb", "mb").unwrap();
        assert!((result - 1000.0).abs() < 0.01);
    }

    #[test]
    fn test_convert_binary_data_units() {
        let result = convert_units(1.0, "gib", "mib").unwrap();
        assert!((result - 1024.0).abs() < 0.01);
    }

    #[test]
    fn test_convert_gallons_to_liters() {
        let result = convert_units(1.0, "gallon", "liter").unwrap();
        assert!((result - 3.785411784).abs() < 0.001);
    }

    #[test]
    fn test_convert_hours_to_seconds() {
        let result = convert_units(1.0, "hour", "seconds").unwrap();
        assert!((result - 3600.0).abs() < 0.01);
    }
}
