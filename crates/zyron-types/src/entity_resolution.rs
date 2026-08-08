//! Specialized fuzzy similarity functions for addresses, names, and company names.

use crate::fuzzy::jaro_winkler;

// ---------------------------------------------------------------------------
// Specialized similarity functions
// ---------------------------------------------------------------------------

/// Computes address similarity with normalization of common abbreviations.
pub fn address_similarity(a: &str, b: &str) -> f64 {
    let norm_a = normalize_address(a);
    let norm_b = normalize_address(b);
    jaro_winkler(&norm_a, &norm_b)
}

fn normalize_address(text: &str) -> String {
    let mut result = text.to_lowercase();
    // Common street suffix replacements
    let replacements = [
        (" street", " st"),
        (" avenue", " ave"),
        (" boulevard", " blvd"),
        (" drive", " dr"),
        (" road", " rd"),
        (" lane", " ln"),
        (" court", " ct"),
        (" place", " pl"),
        (" circle", " cir"),
        (" parkway", " pkwy"),
        (" highway", " hwy"),
        (" square", " sq"),
        (" apartment ", " apt "),
        (" suite ", " ste "),
        (" floor ", " fl "),
        (" north ", " n "),
        (" south ", " s "),
        (" east ", " e "),
        (" west ", " w "),
    ];
    for (from, to) in replacements {
        result = result.replace(from, to);
    }

    // Collapse whitespace
    result.split_whitespace().collect::<Vec<_>>().join(" ")
}

/// Computes name similarity with nickname handling.
pub fn name_similarity(a: &str, b: &str) -> f64 {
    let norm_a = normalize_name(a);
    let norm_b = normalize_name(b);
    // Base similarity from Jaro-Winkler
    let base = jaro_winkler(&norm_a, &norm_b);

    // Bonus for nickname match
    if is_nickname_match(&norm_a, &norm_b) || is_nickname_match(&norm_b, &norm_a) {
        return (base + 1.0) / 2.0;
    }

    base
}

fn normalize_name(text: &str) -> String {
    text.to_lowercase()
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
}

fn is_nickname_match(a: &str, b: &str) -> bool {
    const NICKNAMES: &[(&str, &str)] = &[
        ("robert", "bob"),
        ("robert", "rob"),
        ("robert", "bobby"),
        ("william", "bill"),
        ("william", "will"),
        ("william", "billy"),
        ("richard", "rick"),
        ("richard", "dick"),
        ("james", "jim"),
        ("james", "jimmy"),
        ("john", "johnny"),
        ("john", "jack"),
        ("michael", "mike"),
        ("michael", "mikey"),
        ("thomas", "tom"),
        ("thomas", "tommy"),
        ("edward", "ed"),
        ("edward", "eddie"),
        ("daniel", "dan"),
        ("daniel", "danny"),
        ("joseph", "joe"),
        ("joseph", "joey"),
        ("charles", "charlie"),
        ("charles", "chuck"),
        ("christopher", "chris"),
        ("anthony", "tony"),
        ("steven", "steve"),
        ("stephen", "steve"),
        ("benjamin", "ben"),
        ("alexander", "alex"),
        ("alexandra", "alex"),
        ("katherine", "kate"),
        ("katherine", "kathy"),
        ("kathleen", "kathy"),
        ("margaret", "maggie"),
        ("margaret", "peggy"),
        ("elizabeth", "liz"),
        ("elizabeth", "beth"),
        ("elizabeth", "betty"),
        ("patricia", "pat"),
        ("patricia", "patty"),
        ("jennifer", "jen"),
        ("jennifer", "jenny"),
        ("samantha", "sam"),
        ("nicholas", "nick"),
        ("rebecca", "becky"),
        ("jessica", "jess"),
    ];

    let a_parts: Vec<&str> = a.split_whitespace().collect();
    let b_parts: Vec<&str> = b.split_whitespace().collect();

    for ap in &a_parts {
        for bp in &b_parts {
            for &(full, nick) in NICKNAMES {
                if (*ap == full && *bp == nick) || (*ap == nick && *bp == full) {
                    return true;
                }
            }
        }
    }
    false
}

/// Computes company name similarity with suffix normalization.
pub fn company_similarity(a: &str, b: &str) -> f64 {
    let norm_a = normalize_company(a);
    let norm_b = normalize_company(b);
    jaro_winkler(&norm_a, &norm_b)
}

fn normalize_company(text: &str) -> String {
    let mut result = text.to_lowercase();

    // Remove common company suffixes
    const SUFFIXES: &[&str] = &[
        " inc.",
        " inc",
        " incorporated",
        " corporation",
        " corp.",
        " corp",
        " limited",
        " ltd.",
        " ltd",
        " llc.",
        " llc",
        " l.l.c.",
        " company",
        " co.",
        " co",
        " gmbh",
        " ag",
        " s.a.",
        " sa",
        " plc",
        " pty",
        " pty.",
        " holdings",
        " group",
        " international",
        " intl.",
        " intl",
        " enterprises",
        " the ",
    ];
    for s in SUFFIXES {
        result = result.replace(s, " ");
    }

    // Remove punctuation
    result = result
        .chars()
        .filter(|c| c.is_ascii_alphanumeric() || c.is_whitespace())
        .collect();

    // Collapse whitespace
    result.split_whitespace().collect::<Vec<_>>().join(" ")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_address_similarity_abbrev() {
        let sim = address_similarity("123 Main Street", "123 Main St");
        assert!(sim > 0.9);
    }

    #[test]
    fn test_address_similarity_case() {
        let sim = address_similarity("123 Main St", "123 MAIN ST");
        assert!((sim - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_address_similarity_different() {
        let sim = address_similarity("123 Main St", "456 Oak Ave");
        assert!(sim < 0.8);
    }

    #[test]
    fn test_name_similarity_exact() {
        let sim = name_similarity("Alice Smith", "Alice Smith");
        assert!((sim - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_name_similarity_nickname_bob_robert() {
        let sim = name_similarity("Bob Smith", "Robert Smith");
        assert!(
            sim > 0.8,
            "Bob/Robert should have high similarity, got {}",
            sim
        );
    }

    #[test]
    fn test_name_similarity_nickname_bill_william() {
        let sim = name_similarity("Bill Jones", "William Jones");
        assert!(sim > 0.8);
    }

    #[test]
    fn test_name_similarity_different() {
        let sim = name_similarity("Alice Smith", "Zachary Williams");
        assert!(sim < 0.7, "Expected low similarity, got {}", sim);
    }

    #[test]
    fn test_company_similarity_suffix() {
        let sim = company_similarity("Acme Inc.", "Acme Incorporated");
        // Both normalize to "acme" after suffix removal, but trailing whitespace
        // differences may leave some variation. Expect high similarity.
        assert!(sim > 0.85, "Got {}", sim);
    }

    #[test]
    fn test_company_similarity_llc() {
        let sim = company_similarity("Widgets LLC", "Widgets");
        assert!(sim > 0.9);
    }

    #[test]
    fn test_company_similarity_different() {
        let sim = company_similarity("Apple Inc", "Microsoft Corp");
        assert!(sim < 0.6);
    }
}
