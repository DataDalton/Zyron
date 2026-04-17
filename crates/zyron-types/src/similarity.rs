//! Set and vector similarity functions.
//!
//! Jaccard, Sørensen-Dice, cosine, n-gram similarity, q-gram distance,
//! overlap coefficient. Supports blocked fuzzy-join candidate generation.

use std::collections::HashSet;
use zyron_common::{Result, ZyronError};

// ---------------------------------------------------------------------------
// Token-based similarity
// ---------------------------------------------------------------------------

/// Jaccard similarity: |A intersect B| / |A union B|.
pub fn jaccard_similarity(a: &[&str], b: &[&str]) -> f64 {
    if a.is_empty() && b.is_empty() {
        return 1.0;
    }
    let set_a: HashSet<&&str> = a.iter().collect();
    let set_b: HashSet<&&str> = b.iter().collect();
    let intersection = set_a.intersection(&set_b).count();
    let union = set_a.union(&set_b).count();
    if union == 0 {
        return 0.0;
    }
    intersection as f64 / union as f64
}

/// Sørensen-Dice coefficient: 2*|A intersect B| / (|A| + |B|).
pub fn sorensen_dice(a: &[&str], b: &[&str]) -> f64 {
    if a.is_empty() && b.is_empty() {
        return 1.0;
    }
    let set_a: HashSet<&&str> = a.iter().collect();
    let set_b: HashSet<&&str> = b.iter().collect();
    let intersection = set_a.intersection(&set_b).count();
    let total = set_a.len() + set_b.len();
    if total == 0 {
        return 0.0;
    }
    (2.0 * intersection as f64) / total as f64
}

/// Overlap coefficient: |A intersect B| / min(|A|, |B|).
pub fn overlap_coefficient(a: &[&str], b: &[&str]) -> f64 {
    if a.is_empty() || b.is_empty() {
        return 0.0;
    }
    let set_a: HashSet<&&str> = a.iter().collect();
    let set_b: HashSet<&&str> = b.iter().collect();
    let intersection = set_a.intersection(&set_b).count();
    let smaller = set_a.len().min(set_b.len());
    intersection as f64 / smaller as f64
}

// ---------------------------------------------------------------------------
// Vector similarity
// ---------------------------------------------------------------------------

/// Cosine similarity between two equal-length numeric vectors.
pub fn cosine_similarity(a: &[f64], b: &[f64]) -> Result<f64> {
    if a.len() != b.len() {
        return Err(ZyronError::ExecutionError(format!(
            "Vector length mismatch: {} vs {}",
            a.len(),
            b.len()
        )));
    }
    if a.is_empty() {
        return Ok(0.0);
    }

    let mut dot = 0.0;
    let mut norm_a = 0.0;
    let mut norm_b = 0.0;
    for i in 0..a.len() {
        dot += a[i] * b[i];
        norm_a += a[i] * a[i];
        norm_b += b[i] * b[i];
    }

    if norm_a == 0.0 || norm_b == 0.0 {
        return Ok(0.0);
    }
    Ok(dot / (norm_a.sqrt() * norm_b.sqrt()))
}

// ---------------------------------------------------------------------------
// N-gram / Q-gram
// ---------------------------------------------------------------------------

/// Extracts character n-grams from a string.
pub fn ngrams(text: &str, n: usize) -> Vec<String> {
    if n == 0 {
        return Vec::new();
    }
    let chars: Vec<char> = text.chars().collect();
    if chars.len() < n {
        return Vec::new();
    }
    (0..=chars.len() - n)
        .map(|i| chars[i..i + n].iter().collect())
        .collect()
}

/// Character n-gram similarity using Jaccard over n-gram sets.
pub fn ngram_similarity(a: &str, b: &str, n: usize) -> f64 {
    let ngrams_a = ngrams(a, n);
    let ngrams_b = ngrams(b, n);
    if ngrams_a.is_empty() && ngrams_b.is_empty() {
        return 1.0;
    }
    let refs_a: Vec<&str> = ngrams_a.iter().map(|s| s.as_str()).collect();
    let refs_b: Vec<&str> = ngrams_b.iter().map(|s| s.as_str()).collect();
    jaccard_similarity(&refs_a, &refs_b)
}

/// Q-gram distance: sum of absolute differences in q-gram frequency profiles.
pub fn qgram_distance(a: &str, b: &str, q: usize) -> usize {
    let ngrams_a = ngrams(a, q);
    let ngrams_b = ngrams(b, q);

    let mut freq_a: std::collections::HashMap<String, i32> = std::collections::HashMap::new();
    let mut freq_b: std::collections::HashMap<String, i32> = std::collections::HashMap::new();

    for ng in &ngrams_a {
        *freq_a.entry(ng.clone()).or_insert(0) += 1;
    }
    for ng in &ngrams_b {
        *freq_b.entry(ng.clone()).or_insert(0) += 1;
    }

    let mut distance = 0;
    let all_keys: HashSet<&String> = freq_a.keys().chain(freq_b.keys()).collect();
    for k in all_keys {
        let a_count = freq_a.get(k).copied().unwrap_or(0);
        let b_count = freq_b.get(k).copied().unwrap_or(0);
        distance += (a_count - b_count).unsigned_abs() as usize;
    }
    distance
}

// ---------------------------------------------------------------------------
// Fuzzy join with blocking
// ---------------------------------------------------------------------------

/// Fuzzy join candidate algorithm specifier.
#[derive(Debug, Clone, Copy)]
pub enum FuzzyJoinAlgo {
    JaroWinkler,
    Levenshtein,
    NGram(usize),
    Jaccard,
}

/// Finds pairs of indices (i, j) where the fuzzy similarity between
/// left_keys[i] and right_keys[j] exceeds the threshold.
///
/// If a blocking function is provided, only pairs sharing the same block key
/// are compared, reducing complexity from O(n*m) to O(n*k).
pub fn fuzzy_join_candidates(
    left_keys: &[&str],
    right_keys: &[&str],
    algorithm: FuzzyJoinAlgo,
    threshold: f64,
    blocking_fn: Option<fn(&str) -> String>,
) -> Vec<(usize, usize, f64)> {
    let mut results = Vec::new();

    if let Some(block_fn) = blocking_fn {
        // Group right_keys by block key
        let mut right_blocks: std::collections::HashMap<String, Vec<usize>> =
            std::collections::HashMap::new();
        for (j, k) in right_keys.iter().enumerate() {
            right_blocks.entry(block_fn(k)).or_default().push(j);
        }

        // For each left key, only compare against right keys in the same block
        for (i, left_key) in left_keys.iter().enumerate() {
            let block = block_fn(left_key);
            if let Some(candidates) = right_blocks.get(&block) {
                for &j in candidates {
                    let sim = compute_similarity(left_key, right_keys[j], algorithm);
                    if sim >= threshold {
                        results.push((i, j, sim));
                    }
                }
            }
        }
    } else {
        // Full O(n*m) comparison
        for (i, left_key) in left_keys.iter().enumerate() {
            for (j, right_key) in right_keys.iter().enumerate() {
                let sim = compute_similarity(left_key, right_key, algorithm);
                if sim >= threshold {
                    results.push((i, j, sim));
                }
            }
        }
    }

    results
}

fn compute_similarity(a: &str, b: &str, algo: FuzzyJoinAlgo) -> f64 {
    match algo {
        FuzzyJoinAlgo::JaroWinkler => crate::fuzzy::jaro_winkler(a, b),
        FuzzyJoinAlgo::Levenshtein => {
            let mut buf = crate::fuzzy::FuzzyBuffer::new();
            crate::fuzzy::levenshtein_similarity(a, b, &mut buf)
        }
        FuzzyJoinAlgo::NGram(n) => ngram_similarity(a, b, n),
        FuzzyJoinAlgo::Jaccard => {
            let tokens_a: Vec<&str> = a.split_whitespace().collect();
            let tokens_b: Vec<&str> = b.split_whitespace().collect();
            jaccard_similarity(&tokens_a, &tokens_b)
        }
    }
}

/// Block key: first N characters (case-insensitive).
pub fn block_first_n(text: &str, n: usize) -> String {
    text.chars()
        .take(n)
        .map(|c| c.to_ascii_lowercase())
        .collect()
}

/// Block key: Soundex code.
pub fn block_soundex(text: &str) -> String {
    crate::fuzzy::soundex(text)
}

/// Block key: first letter.
pub fn block_first_letter(text: &str) -> String {
    text.chars()
        .find(|c| c.is_ascii_alphabetic())
        .map(|c| c.to_ascii_uppercase().to_string())
        .unwrap_or_default()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_jaccard_identical() {
        let a = vec!["a", "b", "c"];
        let b = vec!["a", "b", "c"];
        assert_eq!(jaccard_similarity(&a, &b), 1.0);
    }

    #[test]
    fn test_jaccard_empty() {
        let a: Vec<&str> = Vec::new();
        let b: Vec<&str> = Vec::new();
        assert_eq!(jaccard_similarity(&a, &b), 1.0);
    }

    #[test]
    fn test_jaccard_disjoint() {
        let a = vec!["a", "b"];
        let b = vec!["c", "d"];
        assert_eq!(jaccard_similarity(&a, &b), 0.0);
    }

    #[test]
    fn test_jaccard_partial() {
        let a = vec!["a", "b", "c"];
        let b = vec!["b", "c", "d"];
        // |A int B| = 2, |A union B| = 4
        assert!((jaccard_similarity(&a, &b) - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_sorensen_dice() {
        let a = vec!["a", "b", "c"];
        let b = vec!["b", "c", "d"];
        // 2 * 2 / (3 + 3) = 2/3
        let d = sorensen_dice(&a, &b);
        assert!((d - 2.0 / 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_overlap_coefficient() {
        let a = vec!["a", "b", "c"];
        let b = vec!["a", "b", "c", "d", "e"];
        // |A int B| / min(|A|, |B|) = 3 / 3 = 1.0
        assert_eq!(overlap_coefficient(&a, &b), 1.0);
    }

    #[test]
    fn test_cosine_similarity_identical() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![1.0, 2.0, 3.0];
        let sim = cosine_similarity(&a, &b).unwrap();
        assert!((sim - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_cosine_similarity_orthogonal() {
        let a = vec![1.0, 0.0];
        let b = vec![0.0, 1.0];
        let sim = cosine_similarity(&a, &b).unwrap();
        assert!(sim.abs() < 1e-10);
    }

    #[test]
    fn test_cosine_similarity_opposite() {
        let a = vec![1.0, 2.0];
        let b = vec![-1.0, -2.0];
        let sim = cosine_similarity(&a, &b).unwrap();
        assert!((sim - (-1.0)).abs() < 1e-10);
    }

    #[test]
    fn test_cosine_zero_vector() {
        let a = vec![0.0, 0.0];
        let b = vec![1.0, 1.0];
        assert_eq!(cosine_similarity(&a, &b).unwrap(), 0.0);
    }

    #[test]
    fn test_cosine_mismatched_length() {
        let a = vec![1.0];
        let b = vec![1.0, 2.0];
        assert!(cosine_similarity(&a, &b).is_err());
    }

    #[test]
    fn test_ngrams() {
        let g = ngrams("hello", 2);
        assert_eq!(g, vec!["he", "el", "ll", "lo"]);
    }

    #[test]
    fn test_ngrams_too_short() {
        let g = ngrams("hi", 3);
        assert!(g.is_empty());
    }

    #[test]
    fn test_ngram_similarity_identical() {
        assert_eq!(ngram_similarity("hello", "hello", 2), 1.0);
    }

    #[test]
    fn test_ngram_similarity_disjoint() {
        // No shared bigrams
        assert_eq!(ngram_similarity("abc", "xyz", 2), 0.0);
    }

    #[test]
    fn test_qgram_distance_identical() {
        assert_eq!(qgram_distance("hello", "hello", 2), 0);
    }

    #[test]
    fn test_qgram_distance_different() {
        let d = qgram_distance("hello", "world", 2);
        assert!(d > 0);
    }

    #[test]
    fn test_fuzzy_join_no_blocking() {
        let left = vec!["Alice", "Bob"];
        let right = vec!["Alicia", "Robert"];
        let matches = fuzzy_join_candidates(&left, &right, FuzzyJoinAlgo::JaroWinkler, 0.8, None);
        // Alice ~ Alicia should match
        assert!(matches.iter().any(|&(i, j, _)| i == 0 && j == 0));
    }

    #[test]
    fn test_fuzzy_join_with_blocking() {
        let left = vec!["Alice", "Bob"];
        let right = vec!["Alicia", "Robert"];
        let matches = fuzzy_join_candidates(
            &left,
            &right,
            FuzzyJoinAlgo::JaroWinkler,
            0.8,
            Some(block_first_letter),
        );
        // Only A-A and B-B blocks considered
        for (i, j, _) in matches {
            let l = left[i];
            let r = right[j];
            assert_eq!(
                l.chars().next().unwrap().to_ascii_uppercase(),
                r.chars().next().unwrap().to_ascii_uppercase()
            );
        }
    }

    #[test]
    fn test_block_first_n() {
        assert_eq!(block_first_n("Hello", 2), "he");
    }

    #[test]
    fn test_block_first_letter() {
        assert_eq!(block_first_letter("alice"), "A");
        assert_eq!(block_first_letter("Bob"), "B");
    }

    #[test]
    fn test_block_soundex() {
        let s = block_soundex("Robert");
        assert!(!s.is_empty());
    }
}
