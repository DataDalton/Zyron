//! Currency-aware money type with fixed-point arithmetic.
//!
//! Storage: i64 minor units + u16 ISO 4217 currency code.
//! Operations prevent mixed-currency arithmetic bugs by returning errors
//! on currency mismatch (except explicit conversion).

use std::collections::BTreeMap;
use std::sync::OnceLock;

use zyron_common::{Result, ZyronError};

/// Currency information: name, symbol, decimal places, ISO numeric code.
#[derive(Debug, Clone, Copy)]
pub struct CurrencyInfo {
    pub code: &'static str,
    pub symbol: &'static str,
    pub decimals: u8,
    pub numeric: u16,
}

/// Rounds an f64 amount to i64 minor units and rejects values that fall
/// outside the i64 range. A direct cast saturates silently so the bounds are
/// checked against the exact i64 limits before converting.
fn round_to_i64(value: f64) -> Result<i64> {
    let rounded = value.round();
    if !rounded.is_finite() || rounded < i64::MIN as f64 || rounded > i64::MAX as f64 {
        return Err(ZyronError::ExecutionError(format!(
            "Money value out of range: {}",
            value
        )));
    }
    Ok(rounded as i64)
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
    let minor_units = round_to_i64(amount * factor as f64)?;
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
pub fn money_multiply(val: i64, cur: u16, factor: f64) -> Result<(i64, u16)> {
    let result = round_to_i64(val as f64 * factor)?;
    Ok((result, cur))
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
    let target_units = round_to_i64(converted * to_factor)?;

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

/// Rounds a money value to the given number of decimal places using half
/// away from zero. Places at or above the currency's minor digits leave the
/// value unchanged, negative places round into the major units
pub fn money_round(val: i64, cur: u16, decimal_places: i32) -> Result<(i64, u16)> {
    let info = currency_by_numeric(cur)
        .ok_or_else(|| ZyronError::ExecutionError(format!("Unknown currency: {}", cur)))?;
    if !(-18..=18).contains(&decimal_places) {
        return Err(ZyronError::InvalidParameter {
            name: "decimal_places".to_string(),
            value: decimal_places.to_string(),
        });
    }
    let decimals = info.decimals as i32;
    if decimal_places >= decimals {
        return Ok((val, cur));
    }
    let drop = (decimals - decimal_places) as u32;
    let unit = 10i128.pow(drop);
    let v = val as i128;
    let rem = v % unit;
    let base = v - rem;
    let rounded = if rem.abs() * 2 >= unit {
        if v >= 0 { base + unit } else { base - unit }
    } else {
        base
    };
    i64::try_from(rounded)
        .map(|r| (r, cur))
        .map_err(|_| ZyronError::ExecutionError(format!("Money value out of range: {}", rounded)))
}

// ---------------------------------------------------------------------------
// Locale aware money parsing
// ---------------------------------------------------------------------------

/// Number formatting rules and fallback currency for a parse locale
struct LocaleFormat {
    thousands: &'static [char],
    decimal: char,
    default_currency: &'static str,
}

fn locale_format(locale: &str) -> Result<LocaleFormat> {
    match locale.to_ascii_lowercase().as_str() {
        "en_us" => Ok(LocaleFormat {
            thousands: &[','],
            decimal: '.',
            default_currency: "USD",
        }),
        "en_gb" => Ok(LocaleFormat {
            thousands: &[','],
            decimal: '.',
            default_currency: "GBP",
        }),
        "de_de" => Ok(LocaleFormat {
            thousands: &['.'],
            decimal: ',',
            default_currency: "EUR",
        }),
        "fr_fr" => Ok(LocaleFormat {
            thousands: &[' ', '\u{00A0}', '\u{202F}'],
            decimal: ',',
            default_currency: "EUR",
        }),
        "ja_jp" => Ok(LocaleFormat {
            thousands: &[','],
            decimal: '.',
            default_currency: "JPY",
        }),
        _ => Err(ZyronError::InvalidParameter {
            name: "locale".to_string(),
            value: locale.to_string(),
        }),
    }
}

/// Strips a leading or trailing ISO alpha code and returns the currency it
/// names. The code must be exactly three letters standing apart from the
/// number
fn strip_iso_code<'a>(rest: &'a str) -> Option<(CurrencyInfo, &'a str)> {
    let leading: String = rest
        .chars()
        .take_while(|c| c.is_ascii_alphabetic())
        .collect();
    if leading.len() == 3 {
        if let Some(info) = currency_lookup(&leading) {
            return Some((info, rest[leading.len()..].trim_start()));
        }
    }
    let trailing: String = rest
        .chars()
        .rev()
        .take_while(|c| c.is_ascii_alphabetic())
        .collect::<Vec<_>>()
        .into_iter()
        .rev()
        .collect();
    if trailing.len() == 3 {
        if let Some(info) = currency_lookup(&trailing) {
            return Some((info, rest[..rest.len() - trailing.len()].trim_end()));
        }
    }
    None
}

/// Strips a leading or trailing currency symbol. Ambiguous symbols resolve
/// to the locale's default currency when its symbol matches, then to the
/// longest matching symbol in table order
fn strip_symbol<'a>(rest: &'a str, default_currency: &str) -> Option<(CurrencyInfo, &'a str)> {
    let mut best: Option<(bool, usize, CurrencyInfo, &'a str)> = None;
    for info in CURRENCIES {
        let candidates = [
            rest.strip_prefix(info.symbol).map(|r| r.trim_start()),
            rest.strip_suffix(info.symbol).map(|r| r.trim_end()),
        ];
        for remainder in candidates.into_iter().flatten() {
            let is_default = info.code == default_currency;
            let len = info.symbol.len();
            let better = match &best {
                None => true,
                Some((bd, bl, _, _)) => (is_default, len) > (*bd, *bl),
            };
            if better {
                best = Some((is_default, len, *info, remainder));
            }
        }
    }
    best.map(|(_, _, info, remainder)| (info, remainder))
}

/// Parses a localized money string like "$1,234.56", "1.234,56 EUR", or
/// "\u{00A5}1,000" into (minor_units, numeric_code). The locale sets the
/// separator characters and the currency assumed when the text names none
pub fn parse_money(text: &str, locale: &str) -> Result<(i64, u16)> {
    let fmt = locale_format(locale)?;
    let invalid = || ZyronError::InvalidParameter {
        name: "text".to_string(),
        value: text.to_string(),
    };

    let mut rest = text.trim();
    if rest.is_empty() {
        return Err(invalid());
    }

    let mut negative = false;
    if let Some(r) = rest.strip_prefix('-') {
        negative = true;
        rest = r.trim_start();
    } else if let Some(r) = rest.strip_prefix('+') {
        rest = r.trim_start();
    }

    let detected = strip_iso_code(rest).or_else(|| strip_symbol(rest, fmt.default_currency));
    let info = match detected {
        Some((info, remainder)) => {
            rest = remainder;
            info
        }
        None => currency_lookup(fmt.default_currency).ok_or_else(invalid)?,
    };

    // the sign may also sit between the currency marker and the digits
    if !negative {
        if let Some(r) = rest.strip_prefix('-') {
            negative = true;
            rest = r.trim_start();
        }
    }

    // split into integer digits and fraction digits per the locale rules
    let mut int_digits = String::new();
    let mut frac_digits = String::new();
    let mut seen_decimal = false;
    for c in rest.chars() {
        if c.is_ascii_digit() {
            if seen_decimal {
                frac_digits.push(c);
            } else {
                int_digits.push(c);
            }
        } else if c == fmt.decimal && !seen_decimal {
            seen_decimal = true;
        } else if !seen_decimal && fmt.thousands.contains(&c) {
            // grouping separators are dropped, they carry no value
        } else {
            return Err(invalid());
        }
    }
    if int_digits.is_empty() && frac_digits.is_empty() {
        return Err(invalid());
    }

    let d = info.decimals as u32;
    let scale = 10u128.pow(d);
    let int_val: u128 = if int_digits.is_empty() {
        0
    } else {
        int_digits.parse().map_err(|_| invalid())?
    };
    let mut units = int_val
        .checked_mul(scale)
        .ok_or_else(|| ZyronError::ExecutionError(format!("Money value out of range: {}", text)))?;
    let range_err = || ZyronError::ExecutionError(format!("Money value out of range: {}", text));
    if frac_digits.len() as u32 <= d {
        let frac_val: u128 = if frac_digits.is_empty() {
            0
        } else {
            frac_digits.parse().map_err(|_| invalid())?
        };
        units = units
            .checked_add(frac_val * 10u128.pow(d - frac_digits.len() as u32))
            .ok_or_else(range_err)?;
    } else {
        let keep = &frac_digits[..d as usize];
        let keep_val: u128 = if keep.is_empty() {
            0
        } else {
            keep.parse().map_err(|_| invalid())?
        };
        units = units.checked_add(keep_val).ok_or_else(range_err)?;
        // half away from zero on the first dropped digit
        let next = frac_digits.as_bytes()[d as usize] - b'0';
        if next >= 5 {
            units = units.checked_add(1).ok_or_else(range_err)?;
        }
    }

    let magnitude = i64::try_from(units)
        .map_err(|_| ZyronError::ExecutionError(format!("Money value out of range: {}", text)))?;
    let value = if negative { -magnitude } else { magnitude };
    Ok((value, info.numeric))
}

// ---------------------------------------------------------------------------
// Currency rate store
// ---------------------------------------------------------------------------

/// One exchange rate observation, dated in days since 1970-01-01
#[derive(Debug, Clone)]
pub struct CurrencyRate {
    pub from: String,
    pub to: String,
    pub rate_date_days: i32,
    pub rate: f64,
}

/// Process global exchange rate table with lock free reads. Keys are
/// (from, to) alpha code pairs, each holding rates ordered by date
pub struct CurrencyRateStore {
    inner: scc::HashMap<(String, String), BTreeMap<i32, f64>>,
}

impl Default for CurrencyRateStore {
    fn default() -> Self {
        Self::new()
    }
}

impl CurrencyRateStore {
    pub fn new() -> Self {
        Self {
            inner: scc::HashMap::new(),
        }
    }

    /// Replaces the entire rate table with a new snapshot
    pub fn replace_all(&self, rates: Vec<CurrencyRate>) {
        let mut grouped: std::collections::HashMap<(String, String), BTreeMap<i32, f64>> =
            std::collections::HashMap::new();
        for rate in rates {
            grouped
                .entry((rate.from.to_ascii_uppercase(), rate.to.to_ascii_uppercase()))
                .or_default()
                .insert(rate.rate_date_days, rate.rate);
        }
        self.inner.clear_sync();
        for (key, dates) in grouped {
            let _ = self.inner.insert_sync(key, dates);
        }
    }

    pub fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }

    /// Latest rate at or before the given date for the stored direction,
    /// or the latest overall when no date is given
    fn direct_rate(&self, from: &str, to: &str, date_days: Option<i32>) -> Option<f64> {
        self.inner
            .read_sync(
                &(from.to_string(), to.to_string()),
                |_, dates| match date_days {
                    Some(d) => dates.range(..=d).next_back().map(|(_, &r)| r),
                    None => dates.iter().next_back().map(|(_, &r)| r),
                },
            )
            .flatten()
    }

    /// Resolves a rate for the pair. Identity pairs are 1.0, a pair stored
    /// only in the reverse direction falls back to the reciprocal
    pub fn rate_at(&self, from: &str, to: &str, date_days: Option<i32>) -> Option<f64> {
        let from_u = from.to_ascii_uppercase();
        let to_u = to.to_ascii_uppercase();
        if from_u == to_u {
            return Some(1.0);
        }
        if let Some(rate) = self.direct_rate(&from_u, &to_u, date_days) {
            return Some(rate);
        }
        self.direct_rate(&to_u, &from_u, date_days)
            .filter(|r| r.is_finite() && *r != 0.0)
            .map(|r| 1.0 / r)
    }
}

/// Global rate store fed by the currency rate catalog
pub fn currency_rate_store() -> &'static CurrencyRateStore {
    static STORE: OnceLock<CurrencyRateStore> = OnceLock::new();
    STORE.get_or_init(CurrencyRateStore::new)
}

/// Converts a money value between currencies using the global rate store.
/// from_ccy must name the money's own currency. A dated request uses the
/// latest rate at or before that date, falling back to the latest rate on
/// record when none is that old
pub fn convert_currency(
    val: i64,
    cur: u16,
    from_ccy: &str,
    to_ccy: &str,
    rate_date_days: Option<i32>,
) -> Result<(i64, u16)> {
    let from_info = currency_lookup(from_ccy).ok_or_else(|| ZyronError::InvalidParameter {
        name: "from_currency".to_string(),
        value: from_ccy.to_string(),
    })?;
    let to_info = currency_lookup(to_ccy).ok_or_else(|| ZyronError::InvalidParameter {
        name: "to_currency".to_string(),
        value: to_ccy.to_string(),
    })?;
    if from_info.numeric != cur {
        return Err(ZyronError::ExecutionError(format!(
            "convert_currency: source currency {} does not match money currency {}",
            from_info.code,
            money_currency_code(cur)
        )));
    }
    if from_info.numeric == to_info.numeric {
        return Ok((val, cur));
    }
    let store = currency_rate_store();
    let rate = store
        .rate_at(from_info.code, to_info.code, rate_date_days)
        .or_else(|| rate_date_days.and_then(|_| store.rate_at(from_info.code, to_info.code, None)))
        .ok_or_else(|| {
            if store.is_empty() {
                ZyronError::InvalidParameter {
                    name: "currency_rates".to_string(),
                    value: "rate store is empty, populate zyron_sys.cost.currency_rates"
                        .to_string(),
                }
            } else {
                ZyronError::InvalidParameter {
                    name: "currency_pair".to_string(),
                    value: format!("no rate for {}/{}", from_info.code, to_info.code),
                }
            }
        })?;
    money_convert(val, cur, to_info.numeric, rate)
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
        let (val, cur) = money_multiply(1000, 840, 2.5).unwrap();
        assert_eq!(val, 2500);
        assert_eq!(cur, 840);
    }

    #[test]
    fn test_money_multiply_overflow() {
        assert!(money_multiply(i64::MAX, 840, 2.0).is_err());
    }

    #[test]
    fn test_money_create_overflow() {
        assert!(money_create(1e30, "USD").is_err());
    }

    #[test]
    fn test_money_convert_overflow() {
        assert!(money_convert(i64::MAX, 840, 978, 1e10).is_err());
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

    // money_round
    #[test]
    fn test_money_round_noop_at_or_above_minor_digits() {
        assert_eq!(money_round(1999, 840, 2).unwrap(), (1999, 840));
        assert_eq!(money_round(1999, 840, 5).unwrap(), (1999, 840));
    }

    #[test]
    fn test_money_round_half_away_from_zero() {
        // $19.50 to zero places rounds away to $20.00
        assert_eq!(money_round(1950, 840, 0).unwrap(), (2000, 840));
        // -$19.50 rounds away to -$20.00
        assert_eq!(money_round(-1950, 840, 0).unwrap(), (-2000, 840));
        // $19.49 rounds down to $19.00
        assert_eq!(money_round(1949, 840, 0).unwrap(), (1900, 840));
        // $19.95 to one place rounds to $20.00
        assert_eq!(money_round(1995, 840, 1).unwrap(), (2000, 840));
        // $19.94 to one place rounds to $19.90
        assert_eq!(money_round(1994, 840, 1).unwrap(), (1990, 840));
    }

    #[test]
    fn test_money_round_three_decimal_currency() {
        // BHD 12.345 to two places rounds the half digit away from zero
        assert_eq!(money_round(12345, 48, 2).unwrap(), (12350, 48));
        assert_eq!(money_round(12344, 48, 2).unwrap(), (12340, 48));
    }

    #[test]
    fn test_money_round_negative_places() {
        // $123.00 to -1 places rounds to $120.00
        assert_eq!(money_round(12300, 840, -1).unwrap(), (12000, 840));
        // $125.00 to -1 places rounds away to $130.00
        assert_eq!(money_round(12500, 840, -1).unwrap(), (13000, 840));
    }

    #[test]
    fn test_money_round_places_out_of_range() {
        assert!(money_round(100, 840, 40).is_err());
        assert!(money_round(100, 840, -40).is_err());
    }

    // parse_money
    #[test]
    fn test_parse_money_en_us() {
        assert_eq!(parse_money("$1,234.56", "en_US").unwrap(), (123456, 840));
        assert_eq!(parse_money("1234.56", "en_US").unwrap(), (123456, 840));
        assert_eq!(parse_money("-$5.00", "en_US").unwrap(), (-500, 840));
        assert_eq!(parse_money("USD 12.34", "en_US").unwrap(), (1234, 840));
    }

    #[test]
    fn test_parse_money_en_gb() {
        assert_eq!(
            parse_money("\u{00A3}1,234.56", "en_GB").unwrap(),
            (123456, 826)
        );
        assert_eq!(parse_money("99.99", "en_GB").unwrap(), (9999, 826));
    }

    #[test]
    fn test_parse_money_de_de() {
        // German grouping uses dots and a comma decimal
        assert_eq!(
            parse_money("1.234,56 \u{20AC}", "de_DE").unwrap(),
            (123456, 978)
        );
        assert_eq!(parse_money("\u{20AC}99,50", "de_DE").unwrap(), (9950, 978));
    }

    #[test]
    fn test_parse_money_fr_fr() {
        assert_eq!(
            parse_money("1 234,56 \u{20AC}", "fr_FR").unwrap(),
            (123456, 978)
        );
        assert_eq!(
            parse_money("1\u{202F}234,56", "fr_FR").unwrap(),
            (123456, 978)
        );
    }

    #[test]
    fn test_parse_money_ja_jp() {
        // JPY has zero minor digits
        assert_eq!(parse_money("\u{00A5}1,000", "ja_JP").unwrap(), (1000, 392));
        assert_eq!(parse_money("1000", "ja_JP").unwrap(), (1000, 392));
        assert_eq!(parse_money("JPY 500", "en_US").unwrap(), (500, 392));
    }

    #[test]
    fn test_parse_money_iso_code_overrides_locale_default() {
        assert_eq!(parse_money("EUR 10.00", "en_US").unwrap(), (1000, 978));
    }

    #[test]
    fn test_parse_money_rounds_excess_fraction() {
        assert_eq!(parse_money("$1.005", "en_US").unwrap(), (101, 840));
        assert_eq!(parse_money("$1.004", "en_US").unwrap(), (100, 840));
    }

    #[test]
    fn test_parse_money_unknown_locale() {
        assert!(matches!(
            parse_money("$1.00", "xx_XX"),
            Err(ZyronError::InvalidParameter { .. })
        ));
    }

    #[test]
    fn test_parse_money_unparseable() {
        assert!(parse_money("hello", "en_US").is_err());
        assert!(parse_money("", "en_US").is_err());
        assert!(parse_money("$1.2.3", "en_US").is_err());
    }

    // convert_currency, serialized because the store is process global
    static STORE_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

    fn store_guard() -> std::sync::MutexGuard<'static, ()> {
        STORE_LOCK.lock().unwrap_or_else(|p| p.into_inner())
    }

    #[test]
    fn test_convert_currency_identity() {
        let _guard = store_guard();
        currency_rate_store().replace_all(Vec::new());
        // identity pairs need no stored rate
        assert_eq!(
            convert_currency(1000, 840, "USD", "USD", None).unwrap(),
            (1000, 840)
        );
    }

    #[test]
    fn test_convert_currency_empty_store_errors() {
        let _guard = store_guard();
        currency_rate_store().replace_all(Vec::new());
        let err = convert_currency(1000, 840, "USD", "EUR", None);
        match err {
            Err(ZyronError::InvalidParameter { value, .. }) => {
                assert!(value.contains("zyron_sys.cost.currency_rates"));
            }
            other => panic!("expected InvalidParameter, got {:?}", other),
        }
    }

    #[test]
    fn test_convert_currency_direct_and_inverse() {
        let _guard = store_guard();
        currency_rate_store().replace_all(vec![CurrencyRate {
            from: "USD".to_string(),
            to: "EUR".to_string(),
            rate_date_days: 20000,
            rate: 0.8,
        }]);
        // direct pair
        assert_eq!(
            convert_currency(10000, 840, "USD", "EUR", None).unwrap(),
            (8000, 978)
        );
        // reverse pair falls back to the reciprocal
        assert_eq!(
            convert_currency(8000, 978, "EUR", "USD", None).unwrap(),
            (10000, 840)
        );
    }

    #[test]
    fn test_convert_currency_date_selection_and_fallback() {
        let _guard = store_guard();
        currency_rate_store().replace_all(vec![
            CurrencyRate {
                from: "USD".to_string(),
                to: "EUR".to_string(),
                rate_date_days: 20000,
                rate: 0.8,
            },
            CurrencyRate {
                from: "USD".to_string(),
                to: "EUR".to_string(),
                rate_date_days: 20100,
                rate: 0.9,
            },
        ]);
        // at or before 20050 selects the 20000 rate
        assert_eq!(
            convert_currency(10000, 840, "USD", "EUR", Some(20050)).unwrap(),
            (8000, 978)
        );
        // after both dates selects the 20100 rate
        assert_eq!(
            convert_currency(10000, 840, "USD", "EUR", Some(30000)).unwrap(),
            (9000, 978)
        );
        // before every dated rate falls back to the latest on record
        assert_eq!(
            convert_currency(10000, 840, "USD", "EUR", Some(10000)).unwrap(),
            (9000, 978)
        );
        // no date means latest overall
        assert_eq!(
            convert_currency(10000, 840, "USD", "EUR", None).unwrap(),
            (9000, 978)
        );
    }

    #[test]
    fn test_convert_currency_missing_pair_errors() {
        let _guard = store_guard();
        currency_rate_store().replace_all(vec![CurrencyRate {
            from: "USD".to_string(),
            to: "EUR".to_string(),
            rate_date_days: 20000,
            rate: 0.8,
        }]);
        assert!(convert_currency(1000, 840, "USD", "JPY", None).is_err());
    }

    #[test]
    fn test_convert_currency_source_mismatch_errors() {
        let _guard = store_guard();
        // money carries USD but from_ccy claims EUR
        assert!(convert_currency(1000, 840, "EUR", "GBP", None).is_err());
    }
}
