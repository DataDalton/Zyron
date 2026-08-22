//! Fixed-point DECIMAL / NUMERIC values.
//!
//! A `DECIMAL(p, s)` is stored as an `i128` holding the value multiplied by
//! `10^s`, so `10.50` at scale 2 is `1050`. Exact by construction: the whole
//! point of the type is that a value entered as `0.10` reads back as `0.10`
//! rather than as the nearest binary fraction, so no conversion here routes
//! through a float unless the caller started with one.
//!
//! Scale travels with the column rather than the value, which is what lets
//! the stored form be a plain integer and lets addition and comparison of
//! two columns of the same scale run as integer operations.

use crate::{Result, ZyronError};

/// Largest scale a column may declare. `10^38` overflows an i128, so a scale
/// beyond this leaves no digits for the integer part.
pub const MAX_DECIMAL_SCALE: u8 = 38;

/// `10^scale`, for rescaling between two declared scales.
///
/// Returns an error rather than saturating, because a saturated multiplier
/// would silently move the decimal point.
pub fn scale_factor(scale: u8) -> Result<i128> {
    if scale > MAX_DECIMAL_SCALE {
        return Err(ZyronError::ExecutionError(format!(
            "decimal scale {scale} exceeds the maximum of {MAX_DECIMAL_SCALE}"
        )));
    }
    10i128
        .checked_pow(scale as u32)
        .ok_or_else(|| ZyronError::ExecutionError(format!("decimal scale {scale} overflows")))
}

/// Moves a scaled value from one scale to another.
///
/// Scaling up is exact. Scaling down rounds half away from zero, which is
/// what SQL specifies for an assignment that loses digits, and is checked so
/// a value too large for the target reports rather than wraps.
pub fn rescale(value: i128, from: u8, to: u8) -> Result<i128> {
    if from == to {
        return Ok(value);
    }
    if to > from {
        let factor = scale_factor(to - from)?;
        return value.checked_mul(factor).ok_or_else(|| {
            ZyronError::ExecutionError(format!(
                "decimal value overflows when rescaled from {from} to {to}"
            ))
        });
    }
    let factor = scale_factor(from - to)?;
    let quotient = value / factor;
    let remainder = value % factor;
    // Half away from zero: a remainder at or past half a unit moves the
    // quotient one step further from zero
    let bump = if remainder.abs() * 2 >= factor {
        if value < 0 { -1 } else { 1 }
    } else {
        0
    };
    Ok(quotient + bump)
}

/// Parses a decimal literal into a value scaled by `scale`.
///
/// Accepts an optional sign, digits, an optional fractional part and an
/// optional exponent. Digits below the target scale round half away from
/// zero, the same rule `rescale` applies, so a literal and a stored value
/// that mean the same number compare equal.
///
/// Parsing is done on the digits themselves rather than through a float, so
/// a value like `0.1` is exact instead of the nearest binary approximation.
pub fn parse_decimal(text: &str, scale: u8) -> Result<i128> {
    let s = text.trim();
    if s.is_empty() {
        return Err(ZyronError::ExecutionError(
            "cannot read an empty string as a decimal".to_string(),
        ));
    }
    let bad = || ZyronError::ExecutionError(format!("cannot read '{text}' as a decimal"));

    let (negative, rest) = match s.as_bytes()[0] {
        b'-' => (true, &s[1..]),
        b'+' => (false, &s[1..]),
        _ => (false, s),
    };

    // Split off an exponent, so 1.5e3 and 1500 agree
    let (mantissa, exponent) = match rest.find(['e', 'E']) {
        Some(at) => {
            let exp: i32 = rest[at + 1..].parse().map_err(|_| bad())?;
            (&rest[..at], exp)
        }
        None => (rest, 0),
    };

    let (int_part, frac_part) = match mantissa.find('.') {
        Some(at) => (&mantissa[..at], &mantissa[at + 1..]),
        None => (mantissa, ""),
    };
    if int_part.is_empty() && frac_part.is_empty() {
        return Err(bad());
    }
    if !int_part.bytes().all(|b| b.is_ascii_digit())
        || !frac_part.bytes().all(|b| b.is_ascii_digit())
    {
        return Err(bad());
    }

    // Read every digit as an integer, remembering where the point sat. The
    // exponent shifts the point rather than multiplying, so no precision is
    // lost before the final rescale
    let mut digits: i128 = 0;
    for b in int_part.bytes().chain(frac_part.bytes()) {
        digits = digits
            .checked_mul(10)
            .and_then(|d| d.checked_add((b - b'0') as i128))
            .ok_or_else(|| {
                ZyronError::ExecutionError(format!("decimal literal '{text}' has too many digits"))
            })?;
    }
    let literal_scale = frac_part.len() as i32 - exponent;

    let value = if literal_scale == scale as i32 {
        digits
    } else if literal_scale < scale as i32 {
        let up = (scale as i32 - literal_scale) as u32;
        if up > MAX_DECIMAL_SCALE as u32 {
            return Err(ZyronError::ExecutionError(format!(
                "decimal literal '{text}' cannot be represented at scale {scale}"
            )));
        }
        digits.checked_mul(scale_factor(up as u8)?).ok_or_else(|| {
            ZyronError::ExecutionError(format!(
                "decimal literal '{text}' overflows at scale {scale}"
            ))
        })?
    } else {
        let down = literal_scale - scale as i32;
        if down > MAX_DECIMAL_SCALE as i32 {
            // Every digit is below the target scale, so the value rounds to
            // zero rather than reporting an overflow that did not happen
            0
        } else {
            rescale(digits, (literal_scale.max(0)) as u8, scale)?
        }
    };
    Ok(if negative { -value } else { value })
}

/// Renders a scaled value as SQL text, with exactly `scale` digits after the
/// point, and no point at all when the scale is zero.
pub fn format_decimal(value: i128, scale: u8) -> String {
    if scale == 0 {
        return value.to_string();
    }
    let factor = match scale_factor(scale) {
        Ok(f) => f,
        // A scale past the maximum cannot be rendered as a fixed point, so
        // the unscaled integer is the honest answer rather than a wrong one
        Err(_) => return value.to_string(),
    };
    let negative = value < 0;
    let magnitude = value.unsigned_abs();
    let unit = factor.unsigned_abs();
    let whole = magnitude / unit;
    let frac = magnitude % unit;
    format!(
        "{}{}.{:0width$}",
        if negative { "-" } else { "" },
        whole,
        frac,
        width = scale as usize
    )
}

/// Rejects a value whose digit count exceeds the column's declared
/// precision, so `DECIMAL(5,2)` refuses `1234.56` rather than storing a
/// number the declaration says cannot exist.
pub fn check_precision(value: i128, precision: Option<u8>, scale: u8) -> Result<()> {
    let Some(precision) = precision.filter(|p| *p > 0) else {
        return Ok(());
    };
    if precision > MAX_DECIMAL_SCALE {
        return Ok(());
    }
    let limit = scale_factor(precision)?;
    if value.unsigned_abs() >= limit.unsigned_abs() {
        return Err(ZyronError::CheckViolation(format!(
            "value {} exceeds the {} digits DECIMAL({}, {}) declares",
            format_decimal(value, scale),
            precision,
            precision,
            scale
        )));
    }
    Ok(())
}

/// Converts an f64 to a scaled decimal.
///
/// Used only when a value genuinely arrived as a float, since the conversion
/// cannot be exact. Rounds half away from zero at the target scale.
pub fn decimal_from_f64(value: f64, scale: u8) -> Result<i128> {
    if !value.is_finite() {
        return Err(ZyronError::ExecutionError(format!(
            "cannot read {value} as a decimal"
        )));
    }
    let factor = scale_factor(scale)? as f64;
    let scaled = value * factor;
    if scaled.abs() >= i128::MAX as f64 {
        return Err(ZyronError::ExecutionError(format!(
            "value {value} overflows a decimal at scale {scale}"
        )));
    }
    Ok(if scaled < 0.0 {
        (scaled - 0.5).ceil() as i128
    } else {
        (scaled + 0.5).floor() as i128
    })
}

/// Converts a scaled decimal to f64, for the paths that genuinely need a
/// float such as a statistical aggregate.
pub fn decimal_to_f64(value: i128, scale: u8) -> f64 {
    match scale_factor(scale) {
        Ok(factor) => value as f64 / factor as f64,
        Err(_) => value as f64,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_a_literal_round_trips_through_its_scale() {
        for (text, scale, scaled, rendered) in [
            ("10.50", 2u8, 1050i128, "10.50"),
            ("0.01", 2, 1, "0.01"),
            ("-3.25", 2, -325, "-3.25"),
            ("7", 0, 7, "7"),
            ("7", 2, 700, "7.00"),
            ("1.234", 3, 1234, "1.234"),
            ("0", 2, 0, "0.00"),
            ("-0.5", 2, -50, "-0.50"),
            ("1000", 2, 100_000, "1000.00"),
        ] {
            let v = parse_decimal(text, scale).expect(text);
            assert_eq!(v, scaled, "parsing {text} at scale {scale}");
            assert_eq!(format_decimal(v, scale), rendered);
        }
    }

    /// A decimal exists so `0.1` is `0.1`. Going through an f64 would make it
    /// 0.1000000000000000055511151231257827, so the digits are read directly.
    #[test]
    fn test_a_tenth_is_exact() {
        assert_eq!(
            parse_decimal("0.1", 20).unwrap(),
            10_000_000_000_000_000_000
        );
        let mut sum = 0i128;
        for _ in 0..10 {
            sum += parse_decimal("0.1", 2).unwrap();
        }
        assert_eq!(
            format_decimal(sum, 2),
            "1.00",
            "ten tenths make exactly one"
        );
    }

    #[test]
    fn test_an_exponent_shifts_the_point() {
        assert_eq!(parse_decimal("1.5e3", 2).unwrap(), 150_000);
        assert_eq!(
            format_decimal(parse_decimal("1.5e3", 2).unwrap(), 2),
            "1500.00"
        );
        assert_eq!(parse_decimal("15e-2", 4).unwrap(), 1500);
        assert_eq!(
            format_decimal(parse_decimal("15e-2", 4).unwrap(), 4),
            "0.1500"
        );
    }

    #[test]
    fn test_digits_below_the_scale_round_half_away_from_zero() {
        assert_eq!(parse_decimal("1.005", 2).unwrap(), 101);
        assert_eq!(parse_decimal("1.004", 2).unwrap(), 100);
        assert_eq!(parse_decimal("-1.005", 2).unwrap(), -101);
        assert_eq!(parse_decimal("-1.004", 2).unwrap(), -100);
        assert_eq!(parse_decimal("2.5", 0).unwrap(), 3);
        assert_eq!(parse_decimal("-2.5", 0).unwrap(), -3);
    }

    #[test]
    fn test_rescaling_moves_the_point_both_ways() {
        assert_eq!(rescale(1050, 2, 4).unwrap(), 105_000);
        assert_eq!(rescale(105_000, 4, 2).unwrap(), 1050);
        assert_eq!(rescale(1055, 3, 2).unwrap(), 106, "half rounds away");
        assert_eq!(rescale(-1055, 3, 2).unwrap(), -106);
        assert_eq!(rescale(7, 0, 0).unwrap(), 7);
    }

    #[test]
    fn test_precision_refuses_a_value_wider_than_the_declaration() {
        // DECIMAL(5,2) holds up to 999.99
        assert!(check_precision(99_999, Some(5), 2).is_ok());
        assert!(check_precision(100_000, Some(5), 2).is_err());
        assert!(check_precision(-100_000, Some(5), 2).is_err());
        // No declared precision places no bound
        assert!(check_precision(i128::MAX / 2, None, 2).is_ok());
    }

    #[test]
    fn test_text_that_is_not_a_number_is_refused() {
        for bad in ["", "  ", "abc", "1.2.3", "1e", "--1", "1x", "."] {
            assert!(
                parse_decimal(bad, 2).is_err(),
                "'{bad}' should not read as a decimal"
            );
        }
    }

    #[test]
    fn test_a_float_converts_with_rounding_at_the_target_scale() {
        assert_eq!(decimal_from_f64(10.5, 2).unwrap(), 1050);
        assert_eq!(decimal_from_f64(-10.5, 2).unwrap(), -1050);
        assert_eq!(decimal_from_f64(0.125, 2).unwrap(), 13);
        assert!(decimal_from_f64(f64::NAN, 2).is_err());
        assert!(decimal_from_f64(f64::INFINITY, 2).is_err());
    }

    #[test]
    fn test_conversion_back_to_a_float() {
        assert!((decimal_to_f64(1050, 2) - 10.5).abs() < 1e-12);
        assert!((decimal_to_f64(-325, 2) + 3.25).abs() < 1e-12);
        assert_eq!(decimal_to_f64(7, 0), 7.0);
    }

    #[test]
    fn test_a_scale_past_the_maximum_is_refused_rather_than_saturating() {
        assert!(scale_factor(MAX_DECIMAL_SCALE + 1).is_err());
        assert!(scale_factor(MAX_DECIMAL_SCALE).is_ok());
    }
}
