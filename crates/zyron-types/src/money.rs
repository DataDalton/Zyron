//! Currency-aware money type with fixed-point arithmetic.
//!
//! Storage: i64 minor units + u16 ISO 4217 currency code.
//! Operations prevent mixed-currency arithmetic bugs by returning errors
//! on currency mismatch (except explicit conversion).

use zyron_common::{Result, ZyronError};

/// Currency information: name, symbol, decimal places, ISO numeric code.
#[derive(Debug, Clone, Copy)]
pub struct CurrencyInfo {
    pub code: &'static str,
    pub symbol: &'static str,
    pub decimals: u8,
    pub numeric: u16,
}

/// Looks up currency information by ISO 4217 alpha code.
pub fn currency_lookup(code: &str) -> Option<CurrencyInfo> {
    let upper = code.to_ascii_uppercase();
    CURRENCIES
        .iter()
        .find(|c| c.code == upper.as_str())
        .copied()
}

/// Looks up currency information by ISO 4217 numeric code.
pub fn currency_by_numeric(num: u16) -> Option<CurrencyInfo> {
    CURRENCIES.iter().find(|c| c.numeric == num).copied()
}

/// Creates a money value from a decimal amount and a currency code.
/// Returns (minor_units, numeric_code).
pub fn money_create(amount: f64, currency: &str) -> Result<(i64, u16)> {
    let info = currency_lookup(currency)
        .ok_or_else(|| ZyronError::ExecutionError(format!("Unknown currency: {}", currency)))?;
    let factor = 10i64.pow(info.decimals as u32);
    let minor_units = (amount * factor as f64).round() as i64;
    Ok((minor_units, info.numeric))
}

/// Formats a money value with its currency symbol.
pub fn money_format(minor_units: i64, currency: u16) -> String {
    let info = match currency_by_numeric(currency) {
        Some(i) => i,
        None => return format!("{} ({})", minor_units, currency),
    };

    let is_negative = minor_units < 0;
    let abs_units = minor_units.unsigned_abs();
    let factor = 10u64.pow(info.decimals as u32);

    let major = abs_units / factor;
    let minor = abs_units % factor;

    // Format major part with thousand separators
    let major_str = {
        let s = major.to_string();
        let mut result = String::with_capacity(s.len() + s.len() / 3);
        for (i, c) in s.chars().enumerate() {
            if i > 0 && (s.len() - i) % 3 == 0 {
                result.push(',');
            }
            result.push(c);
        }
        result
    };

    let amount = if info.decimals > 0 {
        format!(
            "{}.{:0>width$}",
            major_str,
            minor,
            width = info.decimals as usize
        )
    } else {
        major_str
    };

    let sign = if is_negative { "-" } else { "" };
    format!("{}{}{}", sign, info.symbol, amount)
}

/// Adds two money values. Requires same currency.
pub fn money_add(a_val: i64, a_cur: u16, b_val: i64, b_cur: u16) -> Result<(i64, u16)> {
    if a_cur != b_cur {
        return Err(ZyronError::ExecutionError(format!(
            "Cannot add different currencies: {} and {}",
            a_cur, b_cur
        )));
    }
    let sum = a_val
        .checked_add(b_val)
        .ok_or_else(|| ZyronError::ExecutionError("Money addition overflow".into()))?;
    Ok((sum, a_cur))
}

/// Subtracts two money values. Requires same currency.
pub fn money_subtract(a_val: i64, a_cur: u16, b_val: i64, b_cur: u16) -> Result<(i64, u16)> {
    if a_cur != b_cur {
        return Err(ZyronError::ExecutionError(format!(
            "Cannot subtract different currencies: {} and {}",
            a_cur, b_cur
        )));
    }
    let diff = a_val
        .checked_sub(b_val)
        .ok_or_else(|| ZyronError::ExecutionError("Money subtraction underflow".into()))?;
    Ok((diff, a_cur))
}

/// Multiplies a money value by a scalar factor.
pub fn money_multiply(val: i64, cur: u16, factor: f64) -> (i64, u16) {
    let result = (val as f64 * factor).round() as i64;
    (result, cur)
}

/// Converts a money value from one currency to another with an explicit exchange rate.
pub fn money_convert(val: i64, from_cur: u16, to_cur: u16, rate: f64) -> Result<(i64, u16)> {
    let from_info = currency_by_numeric(from_cur).ok_or_else(|| {
        ZyronError::ExecutionError(format!("Unknown source currency: {}", from_cur))
    })?;
    let to_info = currency_by_numeric(to_cur).ok_or_else(|| {
        ZyronError::ExecutionError(format!("Unknown target currency: {}", to_cur))
    })?;

    // Convert minor units -> decimal amount -> apply rate -> target minor units
    let from_factor = 10f64.powi(from_info.decimals as i32);
    let to_factor = 10f64.powi(to_info.decimals as i32);

    let decimal_amount = (val as f64) / from_factor;
    let converted = decimal_amount * rate;
    let target_units = (converted * to_factor).round() as i64;

    Ok((target_units, to_cur))
}

/// Returns the alpha currency code for a numeric code.
pub fn money_currency_code(cur: u16) -> &'static str {
    currency_by_numeric(cur).map(|c| c.code).unwrap_or("???")
}

/// Returns the currency symbol.
pub fn money_currency_symbol(cur: u16) -> &'static str {
    currency_by_numeric(cur).map(|c| c.symbol).unwrap_or("?")
}

/// Returns the number of decimal places used by the currency.
pub fn money_minor_digits(cur: u16) -> u8 {
    currency_by_numeric(cur).map(|c| c.decimals).unwrap_or(2)
}

// ISO 4217 currency table (subset of commonly-used currencies)
const CURRENCIES: &[CurrencyInfo] = &[
    CurrencyInfo {
        code: "USD",
        symbol: "$",
        decimals: 2,
        numeric: 840,
    },
    CurrencyInfo {
        code: "EUR",
        symbol: "\u{20AC}",
        decimals: 2,
        numeric: 978,
    },
    CurrencyInfo {
        code: "GBP",
        symbol: "\u{00A3}",
        decimals: 2,
        numeric: 826,
    },
    CurrencyInfo {
        code: "JPY",
        symbol: "\u{00A5}",
        decimals: 0,
        numeric: 392,
    },
    CurrencyInfo {
        code: "CNY",
        symbol: "\u{00A5}",
        decimals: 2,
        numeric: 156,
    },
    CurrencyInfo {
        code: "KRW",
        symbol: "\u{20A9}",
        decimals: 0,
        numeric: 410,
    },
    CurrencyInfo {
        code: "INR",
        symbol: "\u{20B9}",
        decimals: 2,
        numeric: 356,
    },
    CurrencyInfo {
        code: "CAD",
        symbol: "CA$",
        decimals: 2,
        numeric: 124,
    },
    CurrencyInfo {
        code: "AUD",
        symbol: "A$",
        decimals: 2,
        numeric: 36,
    },
    CurrencyInfo {
        code: "CHF",
        symbol: "CHF",
        decimals: 2,
        numeric: 756,
    },
    CurrencyInfo {
        code: "NZD",
        symbol: "NZ$",
        decimals: 2,
        numeric: 554,
    },
    CurrencyInfo {
        code: "SEK",
        symbol: "kr",
        decimals: 2,
        numeric: 752,
    },
    CurrencyInfo {
        code: "NOK",
        symbol: "kr",
        decimals: 2,
        numeric: 578,
    },
    CurrencyInfo {
        code: "DKK",
        symbol: "kr",
        decimals: 2,
        numeric: 208,
    },
    CurrencyInfo {
        code: "PLN",
        symbol: "z\u{0142}",
        decimals: 2,
        numeric: 985,
    },
    CurrencyInfo {
        code: "BRL",
        symbol: "R$",
        decimals: 2,
        numeric: 986,
    },
    CurrencyInfo {
        code: "MXN",
        symbol: "MX$",
        decimals: 2,
        numeric: 484,
    },
    CurrencyInfo {
        code: "RUB",
        symbol: "\u{20BD}",
        decimals: 2,
        numeric: 643,
    },
    CurrencyInfo {
        code: "TRY",
        symbol: "\u{20BA}",
        decimals: 2,
        numeric: 949,
    },
    CurrencyInfo {
        code: "ZAR",
        symbol: "R",
        decimals: 2,
        numeric: 710,
    },
    CurrencyInfo {
        code: "SGD",
        symbol: "S$",
        decimals: 2,
        numeric: 702,
    },
    CurrencyInfo {
        code: "HKD",
        symbol: "HK$",
        decimals: 2,
        numeric: 344,
    },
    CurrencyInfo {
        code: "THB",
        symbol: "\u{0E3F}",
        decimals: 2,
        numeric: 764,
    },
    CurrencyInfo {
        code: "MYR",
        symbol: "RM",
        decimals: 2,
        numeric: 458,
    },
    CurrencyInfo {
        code: "IDR",
        symbol: "Rp",
        decimals: 2,
        numeric: 360,
    },
    CurrencyInfo {
        code: "PHP",
        symbol: "\u{20B1}",
        decimals: 2,
        numeric: 608,
    },
    CurrencyInfo {
        code: "VND",
        symbol: "\u{20AB}",
        decimals: 0,
        numeric: 704,
    },
    CurrencyInfo {
        code: "ILS",
        symbol: "\u{20AA}",
        decimals: 2,
        numeric: 376,
    },
    CurrencyInfo {
        code: "AED",
        symbol: "AED",
        decimals: 2,
        numeric: 784,
    },
    CurrencyInfo {
        code: "SAR",
        symbol: "SAR",
        decimals: 2,
        numeric: 682,
    },
    CurrencyInfo {
        code: "EGP",
        symbol: "\u{00A3}",
        decimals: 2,
        numeric: 818,
    },
    CurrencyInfo {
        code: "NGN",
        symbol: "\u{20A6}",
        decimals: 2,
        numeric: 566,
    },
    CurrencyInfo {
        code: "ARS",
        symbol: "AR$",
        decimals: 2,
        numeric: 32,
    },
    CurrencyInfo {
        code: "CLP",
        symbol: "CL$",
        decimals: 0,
        numeric: 152,
    },
    CurrencyInfo {
        code: "COP",
        symbol: "CO$",
        decimals: 2,
        numeric: 170,
    },
    CurrencyInfo {
        code: "BHD",
        symbol: "BHD",
        decimals: 3,
        numeric: 48,
    },
    CurrencyInfo {
        code: "KWD",
        symbol: "KWD",
        decimals: 3,
        numeric: 414,
    },
    CurrencyInfo {
        code: "OMR",
        symbol: "OMR",
        decimals: 3,
        numeric: 512,
    },
    CurrencyInfo {
        code: "JOD",
        symbol: "JOD",
        decimals: 3,
        numeric: 400,
    },
    CurrencyInfo {
        code: "TND",
        symbol: "TND",
        decimals: 3,
        numeric: 788,
    },
    CurrencyInfo {
        code: "CZK",
        symbol: "K\u{010D}",
        decimals: 2,
        numeric: 203,
    },
    CurrencyInfo {
        code: "HUF",
        symbol: "Ft",
        decimals: 2,
        numeric: 348,
    },
    CurrencyInfo {
        code: "RON",
        symbol: "lei",
        decimals: 2,
        numeric: 946,
    },
    CurrencyInfo {
        code: "BGN",
        symbol: "\u{043B}\u{0432}",
        decimals: 2,
        numeric: 975,
    },
    CurrencyInfo {
        code: "HRK",
        symbol: "kn",
        decimals: 2,
        numeric: 191,
    },
    CurrencyInfo {
        code: "ISK",
        symbol: "kr",
        decimals: 0,
        numeric: 352,
    },
    CurrencyInfo {
        code: "UAH",
        symbol: "\u{20B4}",
        decimals: 2,
        numeric: 980,
    },
    CurrencyInfo {
        code: "PKR",
        symbol: "Rs",
        decimals: 2,
        numeric: 586,
    },
    CurrencyInfo {
        code: "BDT",
        symbol: "\u{09F3}",
        decimals: 2,
        numeric: 50,
    },
    CurrencyInfo {
        code: "LKR",
        symbol: "Rs",
        decimals: 2,
        numeric: 144,
    },
    CurrencyInfo {
        code: "XAU",
        symbol: "XAU",
        decimals: 3,
        numeric: 959,
    },
    CurrencyInfo {
        code: "XAG",
        symbol: "XAG",
        decimals: 3,
        numeric: 961,
    },
];

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_currency_lookup() {
        let usd = currency_lookup("USD").unwrap();
        assert_eq!(usd.numeric, 840);
        assert_eq!(usd.decimals, 2);
        assert_eq!(usd.symbol, "$");
    }

    #[test]
    fn test_currency_lookup_case() {
        assert!(currency_lookup("usd").is_some());
        assert!(currency_lookup("USD").is_some());
        assert!(currency_lookup("Usd").is_some());
    }

    #[test]
    fn test_currency_lookup_unknown() {
        assert!(currency_lookup("XYZ").is_none());
    }

    #[test]
    fn test_currency_by_numeric() {
        let usd = currency_by_numeric(840).unwrap();
        assert_eq!(usd.code, "USD");
    }

    #[test]
    fn test_money_create_usd() {
        let (val, cur) = money_create(19.99, "USD").unwrap();
        assert_eq!(val, 1999);
        assert_eq!(cur, 840);
    }

    #[test]
    fn test_money_create_jpy() {
        let (val, cur) = money_create(1000.0, "JPY").unwrap();
        assert_eq!(val, 1000); // JPY has 0 decimals
        assert_eq!(cur, 392);
    }

    #[test]
    fn test_money_create_bhd() {
        let (val, cur) = money_create(1.234, "BHD").unwrap();
        assert_eq!(val, 1234); // BHD has 3 decimals
        assert_eq!(cur, 48);
    }

    #[test]
    fn test_money_create_invalid_currency() {
        assert!(money_create(10.0, "XYZ").is_err());
    }

    #[test]
    fn test_money_format_usd() {
        let formatted = money_format(1999, 840);
        assert_eq!(formatted, "$19.99");
    }

    #[test]
    fn test_money_format_large() {
        let formatted = money_format(1234567890, 840);
        assert_eq!(formatted, "$12,345,678.90");
    }

    #[test]
    fn test_money_format_jpy() {
        let formatted = money_format(1000, 392);
        assert!(formatted.contains("1,000"));
        assert!(!formatted.contains('.'));
    }

    #[test]
    fn test_money_format_negative() {
        let formatted = money_format(-1999, 840);
        assert!(formatted.starts_with('-'));
    }

    #[test]
    fn test_money_add_same_currency() {
        let (val, cur) = money_add(1999, 840, 500, 840).unwrap();
        assert_eq!(val, 2499);
        assert_eq!(cur, 840);
    }

    #[test]
    fn test_money_add_different_currencies() {
        assert!(money_add(100, 840, 100, 978).is_err());
    }

    #[test]
    fn test_money_subtract() {
        let (val, _) = money_subtract(2000, 840, 500, 840).unwrap();
        assert_eq!(val, 1500);
    }

    #[test]
    fn test_money_subtract_different_currencies() {
        assert!(money_subtract(100, 840, 100, 978).is_err());
    }

    #[test]
    fn test_money_multiply() {
        let (val, cur) = money_multiply(1000, 840, 2.5);
        assert_eq!(val, 2500);
        assert_eq!(cur, 840);
    }

    #[test]
    fn test_money_convert() {
        // 100 USD -> EUR at rate 0.85 = 85 EUR
        let (val, cur) = money_convert(10000, 840, 978, 0.85).unwrap();
        assert_eq!(val, 8500);
        assert_eq!(cur, 978);
    }

    #[test]
    fn test_money_convert_different_decimals() {
        // 10.00 USD (1000 minor) -> JPY at rate 150 = 1500 JPY (JPY has 0 decimals)
        let (val, cur) = money_convert(1000, 840, 392, 150.0).unwrap();
        assert_eq!(val, 1500);
        assert_eq!(cur, 392);
    }

    #[test]
    fn test_currency_code() {
        assert_eq!(money_currency_code(840), "USD");
        assert_eq!(money_currency_code(978), "EUR");
        assert_eq!(money_currency_code(392), "JPY");
    }

    #[test]
    fn test_currency_symbol() {
        assert_eq!(money_currency_symbol(840), "$");
    }

    #[test]
    fn test_minor_digits() {
        assert_eq!(money_minor_digits(840), 2);
        assert_eq!(money_minor_digits(392), 0); // JPY
        assert_eq!(money_minor_digits(48), 3); // BHD
    }

    #[test]
    fn test_money_add_overflow() {
        assert!(money_add(i64::MAX, 840, 1, 840).is_err());
    }

    #[test]
    fn test_money_format_zero() {
        assert_eq!(money_format(0, 840), "$0.00");
    }

    #[test]
    fn test_money_roundtrip() {
        let (val, cur) = money_create(99.99, "USD").unwrap();
        let formatted = money_format(val, cur);
        assert_eq!(formatted, "$99.99");
    }
}
