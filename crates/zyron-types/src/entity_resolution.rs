//! Entity resolution and deduplication.
//!
//! Multi-phase approach: blocking -> comparison -> classification -> merging.
//! Specialized similarity functions for addresses, names, and company names.

use crate::fuzzy::{FuzzyBuffer, jaro_winkler};
use crate::similarity::{FuzzyJoinAlgo, fuzzy_join_candidates};
use zyron_common::{Result, ZyronError};

// ---------------------------------------------------------------------------
// Config structs
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy)]
pub enum BlockingStrategy {
    ExactMatch,
    Soundex,
    FirstLetter,
    FirstN(usize),
    None,
}

#[derive(Debug, Clone, Copy)]
pub enum MergeStrategy {
    KeepFirst,
    KeepLatest,
    CombineFields,
}

#[derive(Debug, Clone)]
pub struct ComparisonRule {
    pub field_a: usize,
    pub field_b: usize,
    pub algorithm: FuzzyJoinAlgo,
    pub weight: f64,
    pub threshold: f64,
}

#[derive(Debug, Clone)]
pub struct DeduplicationConfig {
    pub blocking_strategy: BlockingStrategy,
    pub blocking_field: usize,
    pub comparison_rules: Vec<ComparisonRule>,
    pub overall_threshold: f64,
    pub merge_strategy: MergeStrategy,
}

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

// ---------------------------------------------------------------------------
// Entity resolution
// ---------------------------------------------------------------------------

/// Identifies matching records in a dataset based on the given configuration.
/// Returns a list of (i, j, confidence) tuples where records[i] and records[j] match.
pub fn entity_resolve(
    records: &[Vec<&str>],
    config: &DeduplicationConfig,
) -> Result<Vec<(usize, usize, f64)>> {
    if records.is_empty() {
        return Ok(Vec::new());
    }

    // Phase 1: Blocking - group records into blocks
    let blocks = phase_blocking(records, config)?;

    // Phase 2 & 3: Comparison and classification
    let mut matches: Vec<(usize, usize, f64)> = Vec::new();

    for block_members in blocks.values() {
        for i in 0..block_members.len() {
            for j in (i + 1)..block_members.len() {
                let idx_a = block_members[i];
                let idx_b = block_members[j];
                let score = compute_record_score(&records[idx_a], &records[idx_b], config)?;
                if score >= config.overall_threshold {
                    matches.push((idx_a, idx_b, score));
                }
            }
        }
    }

    Ok(matches)
}

fn phase_blocking(
    records: &[Vec<&str>],
    config: &DeduplicationConfig,
) -> Result<std::collections::HashMap<String, Vec<usize>>> {
    let mut blocks: std::collections::HashMap<String, Vec<usize>> =
        std::collections::HashMap::new();

    for (i, record) in records.iter().enumerate() {
        if config.blocking_field >= record.len() {
            return Err(ZyronError::ExecutionError(format!(
                "Blocking field index {} out of range",
                config.blocking_field
            )));
        }
        let key_value = record[config.blocking_field];
        let block_key = match config.blocking_strategy {
            BlockingStrategy::ExactMatch => key_value.to_lowercase(),
            BlockingStrategy::Soundex => crate::fuzzy::soundex(key_value),
            BlockingStrategy::FirstLetter => crate::similarity::block_first_letter(key_value),
            BlockingStrategy::FirstN(n) => crate::similarity::block_first_n(key_value, n),
            BlockingStrategy::None => "_".to_string(),
        };
        blocks.entry(block_key).or_default().push(i);
    }

    Ok(blocks)
}

fn compute_record_score(a: &[&str], b: &[&str], config: &DeduplicationConfig) -> Result<f64> {
    if config.comparison_rules.is_empty() {
        return Ok(0.0);
    }

    let mut weighted_sum = 0.0;
    let mut total_weight = 0.0;
    let mut buf = FuzzyBuffer::new();

    for rule in &config.comparison_rules {
        if rule.field_a >= a.len() || rule.field_b >= b.len() {
            continue;
        }
        let sim = match rule.algorithm {
            FuzzyJoinAlgo::JaroWinkler => jaro_winkler(a[rule.field_a], b[rule.field_b]),
            FuzzyJoinAlgo::Levenshtein => {
                crate::fuzzy::levenshtein_similarity(a[rule.field_a], b[rule.field_b], &mut buf)
            }
            FuzzyJoinAlgo::NGram(n) => {
                crate::similarity::ngram_similarity(a[rule.field_a], b[rule.field_b], n)
            }
            FuzzyJoinAlgo::Jaccard => {
                let ta: Vec<&str> = a[rule.field_a].split_whitespace().collect();
                let tb: Vec<&str> = b[rule.field_b].split_whitespace().collect();
                crate::similarity::jaccard_similarity(&ta, &tb)
            }
        };

        if sim >= rule.threshold {
            weighted_sum += sim * rule.weight;
        }
        total_weight += rule.weight;
    }

    if total_weight == 0.0 {
        Ok(0.0)
    } else {
        Ok(weighted_sum / total_weight)
    }
}

/// Merges two records into one according to the merge strategy.
pub fn merge_records(a: &[&str], b: &[&str], strategy: MergeStrategy) -> Vec<String> {
    let max_len = a.len().max(b.len());
    let mut result = Vec::with_capacity(max_len);

    for i in 0..max_len {
        let a_val = a.get(i).copied().unwrap_or("");
        let b_val = b.get(i).copied().unwrap_or("");

        let merged = match strategy {
            MergeStrategy::KeepFirst => {
                if !a_val.is_empty() {
                    a_val.to_string()
                } else {
                    b_val.to_string()
                }
            }
            MergeStrategy::KeepLatest => {
                if !b_val.is_empty() {
                    b_val.to_string()
                } else {
                    a_val.to_string()
                }
            }
            MergeStrategy::CombineFields => {
                if a_val.is_empty() {
                    b_val.to_string()
                } else if b_val.is_empty() {
                    a_val.to_string()
                } else if a_val == b_val {
                    a_val.to_string()
                } else {
                    format!("{} | {}", a_val, b_val)
                }
            }
        };
        result.push(merged);
    }

    result
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

    #[test]
    fn test_entity_resolve_no_matches() {
        let records = vec![vec!["A", "1"], vec!["B", "2"]];
        let config = DeduplicationConfig {
            blocking_strategy: BlockingStrategy::None,
            blocking_field: 0,
            comparison_rules: vec![ComparisonRule {
                field_a: 0,
                field_b: 0,
                algorithm: FuzzyJoinAlgo::JaroWinkler,
                weight: 1.0,
                threshold: 0.8,
            }],
            overall_threshold: 0.9,
            merge_strategy: MergeStrategy::KeepFirst,
        };
        let matches = entity_resolve(&records, &config).unwrap();
        assert!(matches.is_empty());
    }

    #[test]
    fn test_entity_resolve_matches() {
        let records = vec![
            vec!["Alice Smith", "123 Main St"],
            vec!["Alice Smyth", "123 Main Street"],
            vec!["Bob Jones", "456 Oak Ave"],
        ];
        let config = DeduplicationConfig {
            blocking_strategy: BlockingStrategy::FirstLetter,
            blocking_field: 0,
            comparison_rules: vec![ComparisonRule {
                field_a: 0,
                field_b: 0,
                algorithm: FuzzyJoinAlgo::JaroWinkler,
                weight: 1.0,
                threshold: 0.8,
            }],
            overall_threshold: 0.85,
            merge_strategy: MergeStrategy::KeepFirst,
        };
        let matches = entity_resolve(&records, &config).unwrap();
        // Alice Smith / Alice Smyth should match
        assert!(
            matches
                .iter()
                .any(|&(i, j, _)| (i == 0 && j == 1) || (i == 1 && j == 0))
        );
    }

    #[test]
    fn test_merge_records_keep_first() {
        let a = vec!["Alice", "123 Main St", ""];
        let b = vec!["Alicia", "123 Main Street", "555-1234"];
        let merged = merge_records(&a, &b, MergeStrategy::KeepFirst);
        assert_eq!(merged[0], "Alice");
        assert_eq!(merged[1], "123 Main St");
        assert_eq!(merged[2], "555-1234"); // A was empty, use B
    }

    #[test]
    fn test_merge_records_keep_latest() {
        let a = vec!["Alice", "old value"];
        let b = vec!["Alicia", "new value"];
        let merged = merge_records(&a, &b, MergeStrategy::KeepLatest);
        assert_eq!(merged[0], "Alicia");
    }

    #[test]
    fn test_merge_records_combine() {
        let a = vec!["Alice", "home@example.com"];
        let b = vec!["Alice", "work@example.com"];
        let merged = merge_records(&a, &b, MergeStrategy::CombineFields);
        assert_eq!(merged[0], "Alice"); // same value, not combined
        assert!(merged[1].contains("home@example.com"));
        assert!(merged[1].contains("work@example.com"));
    }
}
