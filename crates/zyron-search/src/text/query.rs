//! FTS query parser with Lucene-like syntax.
//!
//! Parses search query strings into an FtsQuery tree that the inverted
//! index can evaluate. Supports term, phrase, boolean, fuzzy, prefix,
//! proximity, and wildcard queries.

use zyron_common::{Result, ZyronError};

/// Parsed full-text search query representation.
#[derive(Debug, Clone, PartialEq)]
pub enum FtsQuery {
    /// Single term lookup.
    Term(String),
    /// Exact phrase match (terms must appear in order, adjacent).
    Phrase(Vec<String>),
    /// Boolean combination of sub-queries.
    Boolean {
        must: Vec<FtsQuery>,
        should: Vec<FtsQuery>,
        must_not: Vec<FtsQuery>,
    },
    /// Fuzzy term match allowing edit distance.
    Fuzzy { term: String, max_edits: u8 },
    /// Prefix match (e.g., "data*" matches "database", "datastore").
    Prefix(String),
    /// Proximity query: terms must appear within a distance of each other.
    Proximity { terms: Vec<String>, distance: u32 },
    /// Wildcard pattern (supports * and ?).
    Wildcard(String),
}

/// Parses Lucene-like query strings into FtsQuery trees.
///
/// Syntax:
///   term          -> should (default)
///   +term         -> must
///   -term         -> must_not
///   "exact phrase" -> Phrase
///   term~2        -> Fuzzy(max_edits=2)
///   prefix*       -> Prefix
///   "t1 t2"~5    -> Proximity(distance=5)
///   te?t         -> Wildcard
pub struct FtsQueryParser;

impl FtsQueryParser {
    /// Parses a query string into an FtsQuery tree.
    pub fn parse(query: &str) -> Result<FtsQuery> {
        let query = query.trim();
        if query.is_empty() {
            return Err(ZyronError::FtsQueryError("empty query".to_string()));
        }

        let mut must = Vec::new();
        let mut should = Vec::new();
        let mut must_not = Vec::new();

        let mut chars = query.chars().peekable();

        while chars.peek().is_some() {
            // Skip whitespace
            while chars.peek().map_or(false, |c| c.is_whitespace()) {
                chars.next();
            }

            if chars.peek().is_none() {
                break;
            }

            // Determine modifier
            let modifier = match chars.peek() {
                Some('+') => {
                    chars.next();
                    Modifier::Must
                }
                Some('-') => {
                    chars.next();
                    Modifier::MustNot
                }
                _ => Modifier::Should,
            };

            // Parse the next query element
            let sub_query = if chars.peek() == Some(&'"') {
                Self::parse_quoted(&mut chars)?
            } else {
                Self::parse_term(&mut chars)?
            };

            match modifier {
                Modifier::Must => must.push(sub_query),
                Modifier::MustNot => must_not.push(sub_query),
                Modifier::Should => should.push(sub_query),
            }
        }

        // If we have a single should term with no booleans, return it directly
        if must.is_empty() && must_not.is_empty() && should.len() == 1 {
            return Ok(should.into_iter().next().unwrap());
        }

        // If we have only should terms, wrap as boolean
        Ok(FtsQuery::Boolean {
            must,
            should,
            must_not,
        })
    }

    fn parse_quoted(chars: &mut std::iter::Peekable<std::str::Chars<'_>>) -> Result<FtsQuery> {
        // Consume opening quote
        chars.next();

        let mut phrase_text = String::new();
        loop {
            match chars.next() {
                Some('"') => break,
                Some(c) => phrase_text.push(c),
                None => {
                    return Err(ZyronError::FtsQueryError(
                        "unterminated quoted phrase".to_string(),
                    ));
                }
            }
        }

        let words: Vec<String> = phrase_text
            .split_whitespace()
            .map(|s| s.to_lowercase())
            .collect();

        if words.is_empty() {
            return Err(ZyronError::FtsQueryError("empty phrase".to_string()));
        }

        // Check for proximity modifier: "word1 word2"~N
        if chars.peek() == Some(&'~') {
            chars.next();
            let distance = Self::parse_number(chars)?;
            return Ok(FtsQuery::Proximity {
                terms: words,
                distance,
            });
        }

        Ok(FtsQuery::Phrase(words))
    }

    fn parse_term(chars: &mut std::iter::Peekable<std::str::Chars<'_>>) -> Result<FtsQuery> {
        let mut term = String::new();

        while let Some(&c) = chars.peek() {
            if c.is_whitespace() || c == '"' {
                break;
            }
            if c == '+' || c == '-' {
                // Only break if the term is non-empty (modifier for next token)
                if !term.is_empty() {
                    break;
                }
            }
            term.push(c);
            chars.next();
        }

        if term.is_empty() {
            return Err(ZyronError::FtsQueryError("empty term".to_string()));
        }

        // Check for wildcard (contains * or ?)
        if term.contains('?') {
            return Ok(FtsQuery::Wildcard(term.to_lowercase()));
        }

        // Check for prefix: term*
        if let Some(prefix) = term.strip_suffix('*') {
            if !prefix.is_empty() {
                return Ok(FtsQuery::Prefix(prefix.to_lowercase()));
            }
        }

        // Check for fuzzy: term~N
        if term.contains('~') {
            let parts: Vec<&str> = term.splitn(2, '~').collect();
            if parts.len() == 2 {
                let base = parts[0].to_lowercase();
                let edits = parts[1].parse::<u8>().unwrap_or(1);
                return Ok(FtsQuery::Fuzzy {
                    term: base,
                    max_edits: edits,
                });
            }
        }

        Ok(FtsQuery::Term(term.to_lowercase()))
    }

    fn parse_number(chars: &mut std::iter::Peekable<std::str::Chars<'_>>) -> Result<u32> {
        let mut num_str = String::new();
        while let Some(&c) = chars.peek() {
            if c.is_ascii_digit() {
                num_str.push(c);
                chars.next();
            } else {
                break;
            }
        }

        if num_str.is_empty() {
            return Ok(1); // default distance/edits
        }

        num_str
            .parse::<u32>()
            .map_err(|_| ZyronError::FtsQueryError(format!("invalid number: {num_str}")))
    }
}

enum Modifier {
    Must,
    MustNot,
    Should,
}

/// Computes the Levenshtein edit distance between two strings.
/// Used for fuzzy matching in the inverted index search.
pub fn edit_distance(a: &str, b: &str) -> u32 {
    let a_chars: Vec<char> = a.chars().collect();
    let b_chars: Vec<char> = b.chars().collect();
    let m = a_chars.len();
    let n = b_chars.len();

    if m == 0 {
        return n as u32;
    }
    if n == 0 {
        return m as u32;
    }

    // Use two rows for space efficiency
    let mut prev = vec![0u32; n + 1];
    let mut curr = vec![0u32; n + 1];

    for j in 0..=n {
        prev[j] = j as u32;
    }

    for i in 1..=m {
        curr[0] = i as u32;
        for j in 1..=n {
            let cost = if a_chars[i - 1] == b_chars[j - 1] {
                0
            } else {
                1
            };
            curr[j] = (prev[j] + 1).min(curr[j - 1] + 1).min(prev[j - 1] + cost);
        }
        std::mem::swap(&mut prev, &mut curr);
    }

    prev[n]
}

/// Checks if a term matches a wildcard pattern (* = any sequence, ? = any char).
pub fn wildcard_match(pattern: &str, text: &str) -> bool {
    let p: Vec<char> = pattern.chars().collect();
    let t: Vec<char> = text.chars().collect();
    let mut dp = vec![vec![false; t.len() + 1]; p.len() + 1];
    dp[0][0] = true;

    // Handle leading *
    for i in 1..=p.len() {
        if p[i - 1] == '*' {
            dp[i][0] = dp[i - 1][0];
        }
    }

    for i in 1..=p.len() {
        for j in 1..=t.len() {
            if p[i - 1] == '*' {
                dp[i][j] = dp[i - 1][j] || dp[i][j - 1];
            } else if p[i - 1] == '?' || p[i - 1] == t[j - 1] {
                dp[i][j] = dp[i - 1][j - 1];
            }
        }
    }

    dp[p.len()][t.len()]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_single_term() {
        let q = FtsQueryParser::parse("hello").unwrap();
        assert_eq!(q, FtsQuery::Term("hello".to_string()));
    }

    #[test]
    fn test_parse_multiple_terms() {
        let q = FtsQueryParser::parse("hello world").unwrap();
        match q {
            FtsQuery::Boolean {
                must,
                should,
                must_not,
            } => {
                assert!(must.is_empty());
                assert!(must_not.is_empty());
                assert_eq!(should.len(), 2);
            }
            _ => panic!("expected boolean query"),
        }
    }

    #[test]
    fn test_parse_boolean_must() {
        let q = FtsQueryParser::parse("+database +performance").unwrap();
        match q {
            FtsQuery::Boolean {
                must,
                should,
                must_not,
            } => {
                assert_eq!(must.len(), 2);
                assert!(should.is_empty());
                assert!(must_not.is_empty());
            }
            _ => panic!("expected boolean query"),
        }
    }

    #[test]
    fn test_parse_boolean_must_not() {
        let q = FtsQueryParser::parse("+postgresql -mysql").unwrap();
        match q {
            FtsQuery::Boolean {
                must,
                should,
                must_not,
            } => {
                assert_eq!(must.len(), 1);
                assert_eq!(must_not.len(), 1);
            }
            _ => panic!("expected boolean query"),
        }
    }

    #[test]
    fn test_parse_phrase() {
        let q = FtsQueryParser::parse("\"exact phrase\"").unwrap();
        assert_eq!(
            q,
            FtsQuery::Phrase(vec!["exact".to_string(), "phrase".to_string()])
        );
    }

    #[test]
    fn test_parse_proximity() {
        let q = FtsQueryParser::parse("\"query optimization\"~5").unwrap();
        match q {
            FtsQuery::Proximity { terms, distance } => {
                assert_eq!(terms, vec!["query", "optimization"]);
                assert_eq!(distance, 5);
            }
            _ => panic!("expected proximity query"),
        }
    }

    #[test]
    fn test_parse_fuzzy() {
        let q = FtsQueryParser::parse("databse~2").unwrap();
        assert_eq!(
            q,
            FtsQuery::Fuzzy {
                term: "databse".to_string(),
                max_edits: 2
            }
        );
    }

    #[test]
    fn test_parse_prefix() {
        let q = FtsQueryParser::parse("data*").unwrap();
        assert_eq!(q, FtsQuery::Prefix("data".to_string()));
    }

    #[test]
    fn test_parse_wildcard() {
        let q = FtsQueryParser::parse("te?t").unwrap();
        assert_eq!(q, FtsQuery::Wildcard("te?t".to_string()));
    }

    #[test]
    fn test_parse_empty_error() {
        assert!(FtsQueryParser::parse("").is_err());
        assert!(FtsQueryParser::parse("   ").is_err());
    }

    #[test]
    fn test_parse_unterminated_quote() {
        assert!(FtsQueryParser::parse("\"unterminated").is_err());
    }

    #[test]
    fn test_edit_distance() {
        assert_eq!(edit_distance("kitten", "sitting"), 3);
        assert_eq!(edit_distance("", "abc"), 3);
        assert_eq!(edit_distance("abc", "abc"), 0);
        assert_eq!(edit_distance("database", "databse"), 1);
    }

    #[test]
    fn test_wildcard_match_star() {
        assert!(wildcard_match("data*", "database"));
        assert!(wildcard_match("data*", "datastore"));
        assert!(!wildcard_match("data*", "metadata"));
        assert!(wildcard_match("*base", "database"));
    }

    #[test]
    fn test_wildcard_match_question() {
        assert!(wildcard_match("te?t", "test"));
        assert!(wildcard_match("te?t", "text"));
        assert!(!wildcard_match("te?t", "tet"));
    }

    #[test]
    fn test_wildcard_match_combined() {
        assert!(wildcard_match("d?ta*", "database"));
        assert!(wildcard_match("d?ta*", "datastore"));
        assert!(!wildcard_match("d?ta*", "deatabase"));
    }

    #[test]
    fn test_parse_mixed_boolean() {
        let q = FtsQueryParser::parse("+postgresql performance -mysql").unwrap();
        match q {
            FtsQuery::Boolean {
                must,
                should,
                must_not,
            } => {
                assert_eq!(must.len(), 1);
                assert_eq!(should.len(), 1);
                assert_eq!(must_not.len(), 1);
            }
            _ => panic!("expected boolean query"),
        }
    }
}
