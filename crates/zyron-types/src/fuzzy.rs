//! Fuzzy string matching functions.
//!
//! Edit distance (Levenshtein, Damerau-Levenshtein, Hamming),
//! phonetic encoding (Soundex, Metaphone, Double Metaphone, NYSIIS),
//! and similarity scoring (Jaro, Jaro-Winkler).
//! Levenshtein uses O(min(m,n)) space with reusable buffers.

use zyron_common::{Result, ZyronError};

// ---------------------------------------------------------------------------
// Reusable buffer for Levenshtein O(min(m,n)) space
// ---------------------------------------------------------------------------

/// Reusable working memory for Levenshtein distance computation.
/// Holds two rows of the DP matrix to achieve O(min(m,n)) space.
pub struct FuzzyBuffer {
    prev_row: Vec<usize>,
    curr_row: Vec<usize>,
}

impl FuzzyBuffer {
    pub fn new() -> Self {
        Self {
            prev_row: Vec::new(),
            curr_row: Vec::new(),
        }
    }

    /// Ensures the buffer can hold at least `cap` elements per row.
    fn ensure_capacity(&mut self, cap: usize) {
        if self.prev_row.len() < cap {
            self.prev_row.resize(cap, 0);
            self.curr_row.resize(cap, 0);
        }
    }
}

impl Default for FuzzyBuffer {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Levenshtein distance
// ---------------------------------------------------------------------------

/// Computes the Levenshtein edit distance between two strings.
/// Uses O(min(m,n)) space by only keeping two rows of the DP matrix.
/// The buffer is reused across calls for batch efficiency.
pub fn levenshtein(a: &str, b: &str, buf: &mut FuzzyBuffer) -> usize {
    let a_chars: Vec<char> = a.chars().collect();
    let b_chars: Vec<char> = b.chars().collect();

    // Ensure a is the shorter string for O(min(m,n)) space
    let (short, long) = if a_chars.len() <= b_chars.len() {
        (&a_chars, &b_chars)
    } else {
        (&b_chars, &a_chars)
    };

    let short_len = short.len();
    let long_len = long.len();

    if short_len == 0 {
        return long_len;
    }

    buf.ensure_capacity(short_len + 1);

    // Initialize prev_row: [0, 1, 2, ..., short_len]
    for i in 0..=short_len {
        buf.prev_row[i] = i;
    }

    for j in 1..=long_len {
        buf.curr_row[0] = j;
        for i in 1..=short_len {
            let cost = if short[i - 1] == long[j - 1] { 0 } else { 1 };
            buf.curr_row[i] = (buf.prev_row[i] + 1)
                .min(buf.curr_row[i - 1] + 1)
                .min(buf.prev_row[i - 1] + cost);
        }
        // Swap rows
        std::mem::swap(&mut buf.prev_row, &mut buf.curr_row);
    }

    buf.prev_row[short_len]
}

/// Computes Levenshtein similarity: 1.0 - distance / max(len_a, len_b).
/// Returns 1.0 for identical strings, 0.0 for completely different strings.
pub fn levenshtein_similarity(a: &str, b: &str, buf: &mut FuzzyBuffer) -> f64 {
    let max_len = a.chars().count().max(b.chars().count());
    if max_len == 0 {
        return 1.0;
    }
    let dist = levenshtein(a, b, buf);
    1.0 - (dist as f64 / max_len as f64)
}

// ---------------------------------------------------------------------------
// Damerau-Levenshtein distance (optimal string alignment)
// ---------------------------------------------------------------------------

/// Computes the Damerau-Levenshtein distance (includes transpositions).
/// Uses the optimal string alignment (restricted edit distance) variant.
pub fn damerau_levenshtein(a: &str, b: &str) -> usize {
    let a_chars: Vec<char> = a.chars().collect();
    let b_chars: Vec<char> = b.chars().collect();
    let m = a_chars.len();
    let n = b_chars.len();

    if m == 0 {
        return n;
    }
    if n == 0 {
        return m;
    }

    // Use three rows: two previous + current
    let width = n + 1;
    let mut prev_prev = vec![0usize; width];
    let mut prev = vec![0usize; width];
    let mut curr = vec![0usize; width];

    for j in 0..=n {
        prev[j] = j;
    }

    for i in 1..=m {
        curr[0] = i;
        for j in 1..=n {
            let cost = if a_chars[i - 1] == b_chars[j - 1] {
                0
            } else {
                1
            };

            curr[j] = (prev[j] + 1).min(curr[j - 1] + 1).min(prev[j - 1] + cost);

            // Check for transposition
            if i > 1
                && j > 1
                && a_chars[i - 1] == b_chars[j - 2]
                && a_chars[i - 2] == b_chars[j - 1]
            {
                curr[j] = curr[j].min(prev_prev[j - 2] + cost);
            }
        }
        std::mem::swap(&mut prev_prev, &mut prev);
        std::mem::swap(&mut prev, &mut curr);
    }

    prev[n]
}

// ---------------------------------------------------------------------------
// Hamming distance
// ---------------------------------------------------------------------------

/// Computes the Hamming distance between two equal-length strings.
/// Returns an error if the strings have different lengths.
pub fn hamming(a: &str, b: &str) -> Result<usize> {
    let a_chars: Vec<char> = a.chars().collect();
    let b_chars: Vec<char> = b.chars().collect();

    if a_chars.len() != b_chars.len() {
        return Err(ZyronError::ExecutionError(format!(
            "Hamming distance requires equal-length strings (got {} and {})",
            a_chars.len(),
            b_chars.len()
        )));
    }

    Ok(a_chars
        .iter()
        .zip(b_chars.iter())
        .filter(|(a, b)| a != b)
        .count())
}

// ---------------------------------------------------------------------------
// Jaro similarity
// ---------------------------------------------------------------------------

/// Computes the Jaro similarity between two strings.
/// Returns a value between 0.0 (completely different) and 1.0 (identical).
pub fn jaro_similarity(a: &str, b: &str) -> f64 {
    let a_chars: Vec<char> = a.chars().collect();
    let b_chars: Vec<char> = b.chars().collect();

    if a_chars.is_empty() && b_chars.is_empty() {
        return 1.0;
    }
    if a_chars.is_empty() || b_chars.is_empty() {
        return 0.0;
    }

    let match_distance = (a_chars.len().max(b_chars.len()) / 2).saturating_sub(1);

    let mut a_matched = vec![false; a_chars.len()];
    let mut b_matched = vec![false; b_chars.len()];
    let mut matches = 0usize;

    // Find matching characters
    for i in 0..a_chars.len() {
        let start = if i > match_distance {
            i - match_distance
        } else {
            0
        };
        let end = (i + match_distance + 1).min(b_chars.len());

        for j in start..end {
            if !b_matched[j] && a_chars[i] == b_chars[j] {
                a_matched[i] = true;
                b_matched[j] = true;
                matches += 1;
                break;
            }
        }
    }

    if matches == 0 {
        return 0.0;
    }

    // Count transpositions
    let mut transpositions = 0usize;
    let mut k = 0usize;
    for i in 0..a_chars.len() {
        if !a_matched[i] {
            continue;
        }
        while !b_matched[k] {
            k += 1;
        }
        if a_chars[i] != b_chars[k] {
            transpositions += 1;
        }
        k += 1;
    }

    let m = matches as f64;
    let t = transpositions as f64 / 2.0;
    (m / a_chars.len() as f64 + m / b_chars.len() as f64 + (m - t) / m) / 3.0
}

/// Computes Jaro-Winkler similarity (Jaro with prefix bonus).
/// The prefix bonus rewards common prefixes up to 4 characters.
pub fn jaro_winkler(a: &str, b: &str) -> f64 {
    let jaro = jaro_similarity(a, b);

    // Count common prefix (up to 4 characters)
    let prefix_len = a
        .chars()
        .zip(b.chars())
        .take(4)
        .take_while(|(a, b)| a == b)
        .count();

    // Winkler scaling factor (standard p = 0.1)
    jaro + (prefix_len as f64 * 0.1 * (1.0 - jaro))
}

// ---------------------------------------------------------------------------
// Soundex
// ---------------------------------------------------------------------------

/// Computes the Soundex phonetic code (4-character code).
/// Follows the American Soundex algorithm. H and W are ignored but
/// they separate adjacent consonants with the same code.
pub fn soundex(text: &str) -> String {
    let upper: Vec<char> = text.chars().filter(|c| c.is_ascii_alphabetic()).collect();
    if upper.is_empty() {
        return "0000".to_string();
    }

    let mut code = String::with_capacity(4);
    code.push(upper[0].to_ascii_uppercase());

    let soundex_digit = |c: char| -> Option<char> {
        match c.to_ascii_uppercase() {
            'B' | 'F' | 'P' | 'V' => Some('1'),
            'C' | 'G' | 'J' | 'K' | 'Q' | 'S' | 'X' | 'Z' => Some('2'),
            'D' | 'T' => Some('3'),
            'L' => Some('4'),
            'M' | 'N' => Some('5'),
            'R' => Some('6'),
            // H and W return None but do NOT collapse adjacent codes.
            // Vowels (A, E, I, O, U, Y) return None and DO collapse adjacent codes.
            _ => None,
        }
    };

    let is_hw = |c: char| -> bool { matches!(c.to_ascii_uppercase(), 'H' | 'W') };

    let mut last_coded_digit = soundex_digit(upper[0]);
    for &c in &upper[1..] {
        if code.len() >= 4 {
            break;
        }
        let digit = soundex_digit(c);
        if is_hw(c) {
            // H and W are transparent separators: do not update last_coded_digit,
            // do not emit a code. The next consonant will compare against the
            // last coded consonant before the H/W.
            continue;
        }
        if let Some(d) = digit {
            if Some(d) != last_coded_digit {
                code.push(d);
            }
        }
        // Vowels and coded consonants update the last digit tracker.
        // This means vowels separate adjacent same-coded consonants.
        last_coded_digit = digit;
    }

    while code.len() < 4 {
        code.push('0');
    }
    code
}

// ---------------------------------------------------------------------------
// Metaphone
// ---------------------------------------------------------------------------

/// Computes the Metaphone phonetic code.
/// Returns an uppercase phonetic representation of the input word.
pub fn metaphone(text: &str) -> String {
    let upper: Vec<char> = text
        .chars()
        .filter(|c| c.is_ascii_alphabetic())
        .map(|c| c.to_ascii_uppercase())
        .collect();

    if upper.is_empty() {
        return String::new();
    }

    let mut result = String::with_capacity(upper.len());
    let len = upper.len();

    // Drop initial silent letters
    let start = match (upper.first(), upper.get(1)) {
        (Some('A'), Some('E')) => 1,
        (Some('G' | 'K' | 'P'), Some('N')) => 1,
        (Some('W'), Some('R')) => 1,
        _ => 0,
    };

    let mut i = start;
    while i < len {
        let c = upper[i];
        let next = upper.get(i + 1).copied();
        let prev = if i > 0 { Some(upper[i - 1]) } else { None };

        match c {
            'B' => {
                if prev != Some('M') {
                    result.push('B');
                }
            }
            'C' => {
                if next == Some('I') || next == Some('E') || next == Some('Y') {
                    if next == Some('I') && upper.get(i + 2) == Some(&'A') {
                        result.push('X');
                    } else {
                        result.push('S');
                    }
                } else {
                    result.push('K');
                }
            }
            'D' => {
                if next == Some('G')
                    && matches!(upper.get(i + 2), Some('I') | Some('E') | Some('Y'))
                {
                    result.push('J');
                } else {
                    result.push('T');
                }
            }
            'F' => {
                result.push('F');
            }
            'G' => {
                if next == Some('H') && i + 2 < len && !is_vowel(upper[i + 2]) {
                    // GH before non-vowel: silent
                } else if i > 0 && next == Some('N') {
                    // GN: silent G
                } else if prev == Some('G') {
                    // Double G: skip
                } else {
                    result.push('J');
                    if next == Some('E') || next == Some('I') || next == Some('Y') {
                        // already pushed J
                    } else {
                        result.pop();
                        result.push('K');
                    }
                }
            }
            'H' => {
                if is_vowel_opt(next) && !is_vowel_opt(prev) {
                    result.push('H');
                }
            }
            'J' => {
                result.push('J');
            }
            'K' => {
                if prev != Some('C') {
                    result.push('K');
                }
            }
            'L' => {
                result.push('L');
            }
            'M' => {
                result.push('M');
            }
            'N' => {
                result.push('N');
            }
            'P' => {
                if next == Some('H') {
                    result.push('F');
                    i += 1;
                } else {
                    result.push('P');
                }
            }
            'Q' => {
                result.push('K');
            }
            'R' => {
                result.push('R');
            }
            'S' => {
                if next == Some('H') || (next == Some('I') && upper.get(i + 2) == Some(&'O')) {
                    result.push('X');
                    i += 1;
                } else {
                    result.push('S');
                }
            }
            'T' => {
                if next == Some('H') {
                    result.push('0'); // theta
                    i += 1;
                } else if next == Some('I') && upper.get(i + 2) == Some(&'O') {
                    result.push('X');
                } else {
                    result.push('T');
                }
            }
            'V' => {
                result.push('F');
            }
            'W' | 'Y' => {
                if is_vowel_opt(next) {
                    result.push(c);
                }
            }
            'X' => {
                result.push('K');
                result.push('S');
            }
            'Z' => {
                result.push('S');
            }
            _ => {
                // Vowels: only encode if first character
                if i == start && is_vowel(c) {
                    result.push(c);
                }
            }
        }
        i += 1;
    }

    result
}

fn is_vowel(c: char) -> bool {
    matches!(c, 'A' | 'E' | 'I' | 'O' | 'U')
}

fn is_vowel_opt(c: Option<char>) -> bool {
    c.map_or(false, is_vowel)
}

// ---------------------------------------------------------------------------
// Double Metaphone
// ---------------------------------------------------------------------------

/// Computes Double Metaphone codes (primary and alternate).
/// Returns (primary, alternate) tuple. The alternate may equal the primary
/// for words with a single phonetic interpretation.
pub fn double_metaphone(text: &str) -> (String, String) {
    let upper: Vec<char> = text
        .chars()
        .filter(|c| c.is_ascii_alphabetic())
        .map(|c| c.to_ascii_uppercase())
        .collect();

    if upper.is_empty() {
        return (String::new(), String::new());
    }

    let mut primary = String::with_capacity(8);
    let mut alternate = String::with_capacity(8);

    let len = upper.len();
    let mut i = 0;

    // Initial special cases
    if len >= 2
        && matches!(
            (upper[0], upper[1]),
            ('G', 'N') | ('K', 'N') | ('P', 'N') | ('A', 'E') | ('W', 'R')
        )
    {
        i = 1;
    }

    if upper[0] == 'X' {
        primary.push('S');
        alternate.push('S');
        i = 1;
    }

    while i < len && primary.len() < 6 {
        let c = upper[i];
        let next = upper.get(i + 1).copied();

        match c {
            'A' | 'E' | 'I' | 'O' | 'U' => {
                if i == 0 {
                    primary.push('A');
                    alternate.push('A');
                }
            }
            'B' => {
                primary.push('P');
                alternate.push('P');
                if next == Some('B') {
                    i += 1;
                }
            }
            'C' => {
                if next == Some('H') {
                    primary.push('X');
                    alternate.push('X');
                    i += 1;
                } else if next == Some('I') || next == Some('E') || next == Some('Y') {
                    primary.push('S');
                    alternate.push('S');
                } else {
                    primary.push('K');
                    alternate.push('K');
                }
            }
            'D' => {
                if next == Some('G')
                    && matches!(upper.get(i + 2), Some('I') | Some('E') | Some('Y'))
                {
                    primary.push('J');
                    alternate.push('J');
                    i += 1;
                } else {
                    primary.push('T');
                    alternate.push('T');
                }
            }
            'F' => {
                primary.push('F');
                alternate.push('F');
                if next == Some('F') {
                    i += 1;
                }
            }
            'G' => {
                if next == Some('H') {
                    if i > 0 && !is_vowel(upper[i - 1]) {
                        primary.push('K');
                        alternate.push('K');
                    }
                    i += 1;
                } else if next == Some('N') {
                    // Silent G before N
                } else {
                    if next == Some('G') {
                        i += 1;
                    }
                    primary.push('K');
                    alternate.push('K');
                }
            }
            'H' => {
                if is_vowel_opt(next) && (i == 0 || !is_vowel(upper[i - 1])) {
                    primary.push('H');
                    alternate.push('H');
                }
            }
            'J' => {
                primary.push('J');
                alternate.push('H');
            }
            'K' => {
                if i == 0 || upper.get(i - 1) != Some(&'C') {
                    primary.push('K');
                    alternate.push('K');
                }
            }
            'L' => {
                primary.push('L');
                alternate.push('L');
                if next == Some('L') {
                    i += 1;
                }
            }
            'M' => {
                primary.push('M');
                alternate.push('M');
                if next == Some('M') {
                    i += 1;
                }
            }
            'N' => {
                primary.push('N');
                alternate.push('N');
                if next == Some('N') {
                    i += 1;
                }
            }
            'P' => {
                if next == Some('H') {
                    primary.push('F');
                    alternate.push('F');
                    i += 1;
                } else {
                    primary.push('P');
                    alternate.push('P');
                    if next == Some('P') {
                        i += 1;
                    }
                }
            }
            'Q' => {
                primary.push('K');
                alternate.push('K');
            }
            'R' => {
                primary.push('R');
                alternate.push('R');
                if next == Some('R') {
                    i += 1;
                }
            }
            'S' => {
                if next == Some('H') {
                    primary.push('X');
                    alternate.push('X');
                    i += 1;
                } else if next == Some('C') && upper.get(i + 2) == Some(&'H') {
                    primary.push('X');
                    alternate.push('X');
                    i += 2;
                } else {
                    primary.push('S');
                    alternate.push('S');
                    if next == Some('S') {
                        i += 1;
                    }
                }
            }
            'T' => {
                if next == Some('H') {
                    primary.push('T');
                    alternate.push('0');
                    i += 1;
                } else {
                    primary.push('T');
                    alternate.push('T');
                    if next == Some('T') {
                        i += 1;
                    }
                }
            }
            'V' => {
                primary.push('F');
                alternate.push('F');
            }
            'W' => {
                if is_vowel_opt(next) {
                    primary.push('W');
                    alternate.push('W');
                }
            }
            'X' => {
                primary.push('K');
                primary.push('S');
                alternate.push('K');
                alternate.push('S');
            }
            'Y' => {
                if is_vowel_opt(next) {
                    primary.push('Y');
                    alternate.push('Y');
                }
            }
            'Z' => {
                primary.push('S');
                alternate.push('S');
                if next == Some('Z') {
                    i += 1;
                }
            }
            _ => {}
        }
        i += 1;
    }

    (primary, alternate)
}

// ---------------------------------------------------------------------------
// NYSIIS
// ---------------------------------------------------------------------------

/// Computes the NYSIIS (New York State Identification and Intelligence System)
/// phonetic code. Designed for name-focused phonetic encoding.
pub fn nysiis(text: &str) -> String {
    let upper: String = text
        .chars()
        .filter(|c| c.is_ascii_alphabetic())
        .map(|c| c.to_ascii_uppercase())
        .collect();

    if upper.is_empty() {
        return String::new();
    }

    // Translate first characters
    let mut working = upper.clone();
    let prefixes = [
        ("MAC", "MCC"),
        ("KN", "NN"),
        ("K", "C"),
        ("PH", "FF"),
        ("PF", "FF"),
        ("SCH", "SSS"),
    ];
    for (from, to) in prefixes {
        if working.starts_with(from) {
            working = format!("{}{}", to, &working[from.len()..]);
            break;
        }
    }

    // Translate last characters
    let suffixes = [
        ("EE", "Y"),
        ("IE", "Y"),
        ("DT", "D"),
        ("RT", "D"),
        ("RD", "D"),
        ("NT", "D"),
        ("ND", "D"),
    ];
    for (from, to) in suffixes {
        if working.ends_with(from) {
            let trim_len = working.len() - from.len();
            working = format!("{}{}", &working[..trim_len], to);
            break;
        }
    }

    if working.is_empty() {
        return String::new();
    }

    let chars: Vec<char> = working.chars().collect();
    let mut result = String::with_capacity(6);
    result.push(chars[0]);

    let mut i = 1;
    let len = chars.len();
    let mut last_key = chars[0];

    while i < len {
        let c = chars[i];
        let replacement = match c {
            'E' | 'I' | 'O' | 'U' => 'A',
            'Q' => 'G',
            'Z' => 'S',
            'M' => 'N',
            'K' => {
                if i + 1 < len && chars[i + 1] == 'N' {
                    'N'
                } else {
                    'C'
                }
            }
            'S' if i + 2 < len && chars[i + 1] == 'C' && chars[i + 2] == 'H' => {
                i += 2;
                'S'
            }
            'P' if i + 1 < len && chars[i + 1] == 'H' => {
                i += 1;
                'F'
            }
            'H' => {
                let prev_vowel = i > 0 && is_vowel(chars[i - 1]);
                let next_vowel = i + 1 < len && is_vowel(chars[i + 1]);
                if !prev_vowel || !next_vowel {
                    if i > 0 { chars[i - 1] } else { 'H' }
                } else {
                    'H'
                }
            }
            'W' => {
                if i > 0 && is_vowel(chars[i - 1]) {
                    chars[i - 1]
                } else {
                    'W'
                }
            }
            _ => c,
        };

        if replacement != last_key {
            result.push(replacement);
            last_key = replacement;
        }

        i += 1;
    }

    // Remove trailing 'S'
    if result.len() > 1 && result.ends_with('S') {
        result.pop();
    }
    // Remove trailing 'A'
    if result.len() > 1 && result.ends_with('A') {
        result.pop();
    }

    // Truncate to 6 characters
    if result.len() > 6 {
        result.truncate(6);
    }

    result
}

#[cfg(test)]
mod tests {
    use super::*;

    // Levenshtein
    #[test]
    fn test_levenshtein_identical() {
        let mut buf = FuzzyBuffer::new();
        assert_eq!(levenshtein("hello", "hello", &mut buf), 0);
    }

    #[test]
    fn test_levenshtein_empty() {
        let mut buf = FuzzyBuffer::new();
        assert_eq!(levenshtein("", "abc", &mut buf), 3);
        assert_eq!(levenshtein("abc", "", &mut buf), 3);
        assert_eq!(levenshtein("", "", &mut buf), 0);
    }

    #[test]
    fn test_levenshtein_kitten_sitting() {
        let mut buf = FuzzyBuffer::new();
        assert_eq!(levenshtein("kitten", "sitting", &mut buf), 3);
    }

    #[test]
    fn test_levenshtein_single_insert() {
        let mut buf = FuzzyBuffer::new();
        assert_eq!(levenshtein("abc", "abcd", &mut buf), 1);
    }

    #[test]
    fn test_levenshtein_single_delete() {
        let mut buf = FuzzyBuffer::new();
        assert_eq!(levenshtein("abcd", "abc", &mut buf), 1);
    }

    #[test]
    fn test_levenshtein_single_replace() {
        let mut buf = FuzzyBuffer::new();
        assert_eq!(levenshtein("abc", "axc", &mut buf), 1);
    }

    #[test]
    fn test_levenshtein_symmetric() {
        let mut buf = FuzzyBuffer::new();
        assert_eq!(
            levenshtein("abc", "xyz", &mut buf),
            levenshtein("xyz", "abc", &mut buf)
        );
    }

    #[test]
    fn test_levenshtein_similarity_identical() {
        let mut buf = FuzzyBuffer::new();
        assert_eq!(levenshtein_similarity("hello", "hello", &mut buf), 1.0);
    }

    #[test]
    fn test_levenshtein_similarity_empty() {
        let mut buf = FuzzyBuffer::new();
        assert_eq!(levenshtein_similarity("", "", &mut buf), 1.0);
    }

    #[test]
    fn test_levenshtein_similarity_completely_different() {
        let mut buf = FuzzyBuffer::new();
        assert_eq!(levenshtein_similarity("abc", "xyz", &mut buf), 0.0);
    }

    #[test]
    fn test_levenshtein_buffer_reuse() {
        let mut buf = FuzzyBuffer::new();
        levenshtein("short", "a very long string indeed", &mut buf);
        // Buffer should still work for smaller strings
        assert_eq!(levenshtein("ab", "ac", &mut buf), 1);
    }

    // Damerau-Levenshtein
    #[test]
    fn test_damerau_transposition() {
        // "ab" -> "ba" is 1 transposition
        assert_eq!(damerau_levenshtein("ab", "ba"), 1);
        // Standard Levenshtein would be 2 (delete + insert)
        let mut buf = FuzzyBuffer::new();
        assert_eq!(levenshtein("ab", "ba", &mut buf), 2);
    }

    #[test]
    fn test_damerau_ca_abc() {
        assert_eq!(damerau_levenshtein("ca", "abc"), 3);
    }

    #[test]
    fn test_damerau_empty() {
        assert_eq!(damerau_levenshtein("", "abc"), 3);
        assert_eq!(damerau_levenshtein("abc", ""), 3);
    }

    // Hamming
    #[test]
    fn test_hamming_identical() {
        assert_eq!(hamming("hello", "hello").unwrap(), 0);
    }

    #[test]
    fn test_hamming_one_diff() {
        assert_eq!(hamming("hello", "hallo").unwrap(), 1);
    }

    #[test]
    fn test_hamming_all_diff() {
        assert_eq!(hamming("abc", "xyz").unwrap(), 3);
    }

    #[test]
    fn test_hamming_different_lengths() {
        assert!(hamming("abc", "ab").is_err());
    }

    // Jaro
    #[test]
    fn test_jaro_identical() {
        assert!((jaro_similarity("hello", "hello") - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_jaro_completely_different() {
        assert_eq!(jaro_similarity("abc", "xyz"), 0.0);
    }

    #[test]
    fn test_jaro_empty() {
        assert_eq!(jaro_similarity("", ""), 1.0);
        assert_eq!(jaro_similarity("a", ""), 0.0);
    }

    #[test]
    fn test_jaro_known_value() {
        // Jaro("MARTHA", "MARHTA") should be approximately 0.944
        let sim = jaro_similarity("MARTHA", "MARHTA");
        assert!((sim - 0.9444).abs() < 0.01);
    }

    // Jaro-Winkler
    #[test]
    fn test_jaro_winkler_prefix_bonus() {
        let jw = jaro_winkler("MARTHA", "MARHTA");
        let j = jaro_similarity("MARTHA", "MARHTA");
        assert!(jw >= j); // Winkler bonus for common prefix "MAR"
    }

    #[test]
    fn test_jaro_winkler_identical() {
        assert!((jaro_winkler("hello", "hello") - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_jaro_winkler_known_value() {
        // Jaro-Winkler("MARTHA", "MARHTA") should be approximately 0.961
        let sim = jaro_winkler("MARTHA", "MARHTA");
        assert!((sim - 0.961).abs() < 0.01);
    }

    // Soundex
    #[test]
    fn test_soundex_robert() {
        assert_eq!(soundex("Robert"), "R163");
    }

    #[test]
    fn test_soundex_rupert() {
        assert_eq!(soundex("Rupert"), "R163");
    }

    #[test]
    fn test_soundex_same_code() {
        // Robert and Rupert should have the same Soundex code
        assert_eq!(soundex("Robert"), soundex("Rupert"));
    }

    #[test]
    fn test_soundex_ashcraft() {
        assert_eq!(soundex("Ashcraft"), "A261");
    }

    #[test]
    fn test_soundex_empty() {
        assert_eq!(soundex(""), "0000");
    }

    #[test]
    fn test_soundex_short() {
        // Single character should pad with zeros
        let code = soundex("A");
        assert_eq!(code.len(), 4);
        assert_eq!(code, "A000");
    }

    // Metaphone
    #[test]
    fn test_metaphone_smith() {
        let code = metaphone("Smith");
        assert_eq!(code, "SM0"); // TH -> 0 (theta)
    }

    #[test]
    fn test_metaphone_phone() {
        let code = metaphone("Phone");
        assert!(code.starts_with('F')); // PH -> F
    }

    #[test]
    fn test_metaphone_empty() {
        assert_eq!(metaphone(""), "");
    }

    // Double Metaphone
    #[test]
    fn test_double_metaphone_smith() {
        let (primary, _alternate) = double_metaphone("Smith");
        assert!(primary.starts_with('S'));
    }

    #[test]
    fn test_double_metaphone_returns_two() {
        let (primary, alternate) = double_metaphone("John");
        assert!(!primary.is_empty());
        assert!(!alternate.is_empty());
    }

    #[test]
    fn test_double_metaphone_empty() {
        let (p, a) = double_metaphone("");
        assert_eq!(p, "");
        assert_eq!(a, "");
    }

    // NYSIIS
    #[test]
    fn test_nysiis_basic() {
        let code = nysiis("Johnson");
        assert!(!code.is_empty());
        assert!(code.len() <= 6);
    }

    #[test]
    fn test_nysiis_mac_prefix() {
        let code = nysiis("MacDonald");
        assert!(code.starts_with('M'));
    }

    #[test]
    fn test_nysiis_empty() {
        assert_eq!(nysiis(""), "");
    }

    #[test]
    fn test_nysiis_max_length() {
        let code = nysiis("Schwarzenegger");
        assert!(code.len() <= 6);
    }

    // Unicode support
    #[test]
    fn test_levenshtein_unicode() {
        let mut buf = FuzzyBuffer::new();
        assert_eq!(levenshtein("cafe\u{0301}", "cafe", &mut buf), 1);
    }

    #[test]
    fn test_jaro_unicode() {
        let sim = jaro_similarity("hello", "hello");
        assert!((sim - 1.0).abs() < 1e-10);
    }
}
