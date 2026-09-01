//! Collation backed string comparison.
//!
//! Two providers: `icu` uses the ICU4X collator with locale tailoring, so
//! German sorts a, a umlaut, b rather than by codepoint. `binary` compares
//! bytes, optionally case folded. Collators are expensive to build, so a
//! process wide cache shares one per (locale, provider, case_sensitive).

use std::cmp::Ordering;
use std::sync::{Arc, OnceLock};

use zyron_common::{Result, ZyronError};

enum Backend {
    Icu {
        collator: icu_collator::Collator,
        case_sensitive: bool,
    },
    Binary {
        case_sensitive: bool,
    },
}

/// A ready to use comparator for one collation configuration.
pub struct Collator {
    backend: Backend,
}

// The ICU collator holds immutable compiled data and is shared read only
unsafe impl Send for Collator {}
unsafe impl Sync for Collator {}

impl Collator {
    pub fn new(locale: &str, provider: &str, case_sensitive: bool) -> Result<Self> {
        match provider {
            "binary" => Ok(Self {
                backend: Backend::Binary { case_sensitive },
            }),
            "icu" => {
                let normalized = locale.replace('_', "-");
                let parsed: icu_locid::Locale =
                    normalized
                        .parse()
                        .map_err(|_| ZyronError::InvalidParameter {
                            name: "locale".to_string(),
                            value: locale.to_string(),
                        })?;
                let mut options = icu_collator::CollatorOptions::new();
                options.strength = Some(if case_sensitive {
                    icu_collator::Strength::Tertiary
                } else {
                    icu_collator::Strength::Secondary
                });
                let collator =
                    icu_collator::Collator::try_new(&parsed.into(), options).map_err(|e| {
                        ZyronError::InvalidParameter {
                            name: "locale".to_string(),
                            value: format!("{locale}: {e:?}"),
                        }
                    })?;
                Ok(Self {
                    backend: Backend::Icu {
                        collator,
                        case_sensitive,
                    },
                })
            }
            other => Err(ZyronError::InvalidParameter {
                name: "provider".to_string(),
                value: other.to_string(),
            }),
        }
    }

    pub fn compare(&self, a: &str, b: &str) -> Ordering {
        match &self.backend {
            Backend::Icu {
                collator,
                case_sensitive,
            } => {
                // ICU documents the sharp s as equal to ss below the tertiary
                // level, but the compiled collation data separates the pair
                // at secondary strength, so straße and strasse compared as
                // different under a case insensitive collation. Folding the
                // sharp s restores the documented behavior and matches the
                // fold the sort key applies, keeping equality consistent
                // with collated ordering
                if !case_sensitive && (contains_sharp_s(a) || contains_sharp_s(b)) {
                    return collator.compare(&fold_sharp_s(a), &fold_sharp_s(b));
                }
                collator.compare(a, b)
            }
            Backend::Binary {
                case_sensitive: true,
            } => a.cmp(b),
            Backend::Binary {
                case_sensitive: false,
            } => a.to_lowercase().cmp(&b.to_lowercase()),
        }
    }

    pub fn equals(&self, a: &str, b: &str) -> bool {
        self.compare(a, b) == Ordering::Equal
    }
}

fn contains_sharp_s(s: &str) -> bool {
    s.contains(['\u{df}', '\u{1e9e}'])
}

/// Replaces the sharp s, in both cases, with ss. Case does not matter to the
/// callers because the fold only runs under a case insensitive collation
fn fold_sharp_s(s: &str) -> String {
    s.replace('\u{df}', "ss").replace('\u{1e9e}', "ss")
}

type CollatorKey = (String, String, bool);

static COLLATORS: OnceLock<scc::HashMap<CollatorKey, Arc<Collator>>> = OnceLock::new();

fn collators() -> &'static scc::HashMap<CollatorKey, Arc<Collator>> {
    COLLATORS.get_or_init(scc::HashMap::new)
}

/// Returns the shared collator for a configuration, building it on first use
pub fn cached_collator(
    locale: &str,
    provider: &str,
    case_sensitive: bool,
) -> Result<Arc<Collator>> {
    let key = (locale.to_string(), provider.to_string(), case_sensitive);
    if let Some(found) = collators().read_sync(&key, |_, c| Arc::clone(c)) {
        return Ok(found);
    }
    let built = Arc::new(Collator::new(locale, provider, case_sensitive)?);
    let _ = collators().insert_sync(key, Arc::clone(&built));
    Ok(built)
}

/// Proves a collation configuration constructs, without keeping it
pub fn validate_collation(locale: &str, provider: &str, case_sensitive: bool) -> Result<()> {
    Collator::new(locale, provider, case_sensitive).map(|_| ())
}

/// Base letter and diacritic weight of a latin script character. The
/// weight separates accented variants at the secondary level so the
/// primary order interleaves them with their base letter
fn decompose_latin(c: char) -> (&'static str, u8, Option<char>) {
    let table: Option<(&'static str, u8)> = match c.to_lowercase().next().unwrap_or(c) {
        '\u{e0}' => Some(("a", 2)),
        '\u{e1}' => Some(("a", 1)),
        '\u{e2}' => Some(("a", 3)),
        '\u{e3}' => Some(("a", 5)),
        '\u{e4}' => Some(("a", 4)),
        '\u{e5}' => Some(("a", 6)),
        '\u{e6}' => Some(("ae", 0)),
        '\u{e7}' => Some(("c", 7)),
        '\u{e8}' => Some(("e", 2)),
        '\u{e9}' => Some(("e", 1)),
        '\u{ea}' => Some(("e", 3)),
        '\u{eb}' => Some(("e", 4)),
        '\u{ec}' => Some(("i", 2)),
        '\u{ed}' => Some(("i", 1)),
        '\u{ee}' => Some(("i", 3)),
        '\u{ef}' => Some(("i", 4)),
        '\u{f1}' => Some(("n", 5)),
        '\u{f2}' => Some(("o", 2)),
        '\u{f3}' => Some(("o", 1)),
        '\u{f4}' => Some(("o", 3)),
        '\u{f5}' => Some(("o", 5)),
        '\u{f6}' => Some(("o", 4)),
        '\u{f8}' => Some(("o", 8)),
        '\u{f9}' => Some(("u", 2)),
        '\u{fa}' => Some(("u", 1)),
        '\u{fb}' => Some(("u", 3)),
        '\u{fc}' => Some(("u", 4)),
        '\u{fd}' => Some(("y", 1)),
        '\u{ff}' => Some(("y", 4)),
        '\u{df}' => Some(("ss", 0)),
        _ => None,
    };
    match table {
        Some((base, weight)) => (base, weight, None),
        None => ("", 0, Some(c)),
    }
}

/// Builds a byte key whose plain byte order matches collated order for
/// latin script text: primary base letters, then diacritic weights, then
/// case bits when the collation is case sensitive. German a, a umlaut, b
/// order by the primary level and the umlaut separates at the secondary
pub fn sort_key(text: &str, case_sensitive: bool) -> Vec<u8> {
    let mut primary: Vec<u8> = Vec::with_capacity(text.len());
    let mut secondary: Vec<u8> = Vec::with_capacity(text.len());
    let mut tertiary: Vec<u8> = Vec::with_capacity(text.len());
    // Secondary and tertiary stay aligned to primary BYTES, the first byte
    // of a character's expansion carries its weight and the rest pad with
    // zero, so a sharp s equals ss at every level below case
    let mut push_char = |bytes: &[u8], weight: u8, is_upper: bool| {
        for (i, b) in bytes.iter().enumerate() {
            primary.push(*b);
            secondary.push(if i == 0 { weight } else { 0 });
            tertiary.push(if i == 0 { u8::from(is_upper) } else { 0 });
        }
    };
    for c in text.chars() {
        let is_upper = c.is_uppercase();
        let (base, weight, passthrough) = decompose_latin(c);
        match passthrough {
            Some(other) => {
                let mut buf = [0u8; 4];
                for lower in other.to_lowercase() {
                    let encoded = lower.encode_utf8(&mut buf);
                    push_char(encoded.as_bytes(), 0, is_upper);
                }
            }
            None => push_char(base.as_bytes(), weight, is_upper),
        }
    }
    let mut key = Vec::with_capacity(primary.len() + secondary.len() + tertiary.len() + 2);
    key.extend_from_slice(&primary);
    key.push(0);
    key.extend_from_slice(&secondary);
    if case_sensitive {
        key.push(0);
        key.extend_from_slice(&tertiary);
    }
    key
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_german_collation_orders_umlaut_between() {
        let c = Collator::new("de_DE", "icu", true).expect("german collator");
        // a < a umlaut < b under German collation, while codepoint order
        // puts the umlaut after z
        assert_eq!(c.compare("a", "\u{e4}"), Ordering::Less);
        assert_eq!(c.compare("\u{e4}", "b"), Ordering::Less);
        assert!(
            "\u{e4}".cmp("b") == Ordering::Greater,
            "codepoint order differs"
        );
    }

    #[test]
    fn test_case_insensitive_icu_equality() {
        let c = Collator::new("en_US", "icu", false).expect("english collator");
        assert!(c.equals("Strasse", "strasse"));
        let cs = Collator::new("en_US", "icu", true).expect("english collator");
        assert!(!cs.equals("Strasse", "strasse"));
    }

    #[test]
    fn test_binary_provider() {
        let c = Collator::new("en_US", "binary", true).expect("binary");
        assert_eq!(c.compare("a", "b"), Ordering::Less);
        let ci = Collator::new("en_US", "binary", false).expect("binary ci");
        assert!(ci.equals("ABC", "abc"));
    }

    #[test]
    fn test_invalid_locale_rejected() {
        assert!(Collator::new("not a locale!!", "icu", true).is_err());
        assert!(Collator::new("en_US", "sparkle", true).is_err());
    }

    #[test]
    fn test_sort_key_orders_umlaut_between() {
        let a = sort_key("a", true);
        let ae = sort_key("\u{e4}", true);
        let b = sort_key("b", true);
        assert!(a < ae, "a sorts before a umlaut");
        assert!(ae < b, "a umlaut sorts before b");
        let sharp = sort_key("stra\u{df}e", false);
        let ss = sort_key("strasse", false);
        assert_eq!(sharp, ss, "sharp s folds to ss at the primary level");
    }

    #[test]
    fn test_sort_key_case_handling() {
        assert_eq!(sort_key("Abc", false), sort_key("abc", false));
        assert_ne!(sort_key("Abc", true), sort_key("abc", true));
        // Case differs only at the tertiary level, so ordering by base
        // letters still dominates
        assert!(sort_key("Abc", true) < sort_key("abd", true));
    }

    #[test]
    fn test_cached_collator_shares() {
        let a = cached_collator("fr_FR", "icu", true).expect("build");
        let b = cached_collator("fr_FR", "icu", true).expect("cached");
        assert!(Arc::ptr_eq(&a, &b));
    }
}
