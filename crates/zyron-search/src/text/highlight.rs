//! Result highlighting for full-text search.
//!
//! Wraps matching terms in configurable tags (e.g., <b>term</b>) and
//! selects the best text fragments containing query terms.

use std::collections::HashSet;

use super::analyzer::Analyzer;
use super::query::FtsQuery;

/// Configuration for highlighting matched terms in search results.
pub struct HighlightConfig {
    /// Opening tag to wrap matched terms. Default: "<b>".
    pub pre_tag: String,
    /// Closing tag to wrap matched terms. Default: "</b>".
    pub post_tag: String,
    /// Maximum characters per fragment. Default: 150.
    pub fragment_size: usize,
    /// Maximum number of fragments to return. Default: 3.
    pub num_fragments: usize,
}

impl Default for HighlightConfig {
    fn default() -> Self {
        Self {
            pre_tag: "<b>".to_string(),
            post_tag: "</b>".to_string(),
            fragment_size: 150,
            num_fragments: 3,
        }
    }
}

/// Highlights matching terms in the text by wrapping them in tags.
/// Returns the top-N fragments containing the most query term matches.
pub fn highlight(
    text: &str,
    query: &FtsQuery,
    analyzer: &dyn Analyzer,
    config: &HighlightConfig,
) -> String {
    let query_terms = extract_query_terms(query);
    if query_terms.is_empty() {
        // No terms to highlight, return a truncated fragment
        let end = text.len().min(config.fragment_size);
        return safe_truncate(text, end).to_string();
    }

    // Analyze query terms to get their stemmed/lowercased forms
    let analyzed_terms: HashSet<String> = query_terms
        .iter()
        .flat_map(|t| analyzer.analyze(t).into_iter().map(|tok| tok.term))
        .collect();

    // Split text into fragments (sentence-like chunks)
    let fragments = split_into_fragments(text, config.fragment_size);

    // Score each fragment by number of matching terms
    let mut scored_fragments: Vec<(usize, &str, usize)> = fragments
        .iter()
        .enumerate()
        .map(|(i, frag)| {
            let tokens = analyzer.analyze(frag);
            let match_count = tokens
                .iter()
                .filter(|t| analyzed_terms.contains(&t.term))
                .count();
            (i, *frag, match_count)
        })
        .collect();

    // Sort by match count descending, take top N
    scored_fragments.sort_by(|a, b| b.2.cmp(&a.2));
    scored_fragments.truncate(config.num_fragments);

    // Re-sort by original position for coherent output
    scored_fragments.sort_by_key(|f| f.0);

    // Highlight matching tokens in each fragment
    let highlighted: Vec<String> = scored_fragments
        .iter()
        .map(|(_, frag, _)| highlight_fragment(frag, &analyzed_terms, analyzer, config))
        .collect();

    highlighted.join(" ... ")
}

/// Extracts all query terms (recursively) from an FtsQuery.
fn extract_query_terms(query: &FtsQuery) -> Vec<String> {
    match query {
        FtsQuery::Term(t) => vec![t.clone()],
        FtsQuery::Phrase(words) => words.clone(),
        FtsQuery::Boolean {
            must,
            should,
            must_not: _,
        } => {
            let mut terms = Vec::new();
            for q in must {
                terms.extend(extract_query_terms(q));
            }
            for q in should {
                terms.extend(extract_query_terms(q));
            }
            terms
        }
        FtsQuery::Fuzzy { term, .. } => vec![term.clone()],
        FtsQuery::Prefix(p) => vec![p.clone()],
        FtsQuery::Proximity { terms, .. } => terms.clone(),
        FtsQuery::Wildcard(w) => vec![w.clone()],
    }
}

/// Splits text into fragments of approximately the given size,
/// breaking on sentence boundaries when possible.
fn split_into_fragments(text: &str, max_size: usize) -> Vec<&str> {
    if text.len() <= max_size {
        return vec![text];
    }

    let mut fragments = Vec::new();
    let mut start = 0;

    while start < text.len() {
        let end = (start + max_size).min(text.len());

        // Try to break at a sentence boundary
        let chunk = &text[start..end];
        let break_pos = chunk
            .rfind(". ")
            .or_else(|| chunk.rfind("! "))
            .or_else(|| chunk.rfind("? "))
            .or_else(|| chunk.rfind(' '));

        let actual_end = match break_pos {
            Some(pos) if pos > max_size / 4 => start + pos + 1,
            _ => end,
        };

        let frag = safe_truncate(text, actual_end);
        let frag = &frag[start..];
        if !frag.trim().is_empty() {
            fragments.push(frag);
        }
        start = actual_end;
    }

    if fragments.is_empty() && !text.is_empty() {
        fragments.push(safe_truncate(text, max_size));
    }

    fragments
}

/// Highlights matching terms in a single fragment.
fn highlight_fragment(
    fragment: &str,
    query_terms: &HashSet<String>,
    analyzer: &dyn Analyzer,
    config: &HighlightConfig,
) -> String {
    let tokens = analyzer.analyze(fragment);
    if tokens.is_empty() {
        return fragment.to_string();
    }

    // Build a set of byte ranges that need highlighting
    let mut highlight_ranges: Vec<(usize, usize)> = Vec::new();
    for token in &tokens {
        if query_terms.contains(&token.term) {
            highlight_ranges.push((token.start_offset as usize, token.end_offset as usize));
        }
    }

    if highlight_ranges.is_empty() {
        return fragment.to_string();
    }

    // Sort ranges by start offset
    highlight_ranges.sort_by_key(|r| r.0);

    // Build highlighted string
    let mut result = String::new();
    let mut cursor = 0;

    for (start, end) in &highlight_ranges {
        let start = *start;
        let end = (*end).min(fragment.len());
        if start > cursor {
            result.push_str(&fragment[cursor..start]);
        }
        result.push_str(&config.pre_tag);
        result.push_str(&fragment[start..end]);
        result.push_str(&config.post_tag);
        cursor = end;
    }

    if cursor < fragment.len() {
        result.push_str(&fragment[cursor..]);
    }

    result
}

/// Truncates a string to at most `max_bytes`, avoiding splitting UTF-8.
fn safe_truncate(s: &str, max_bytes: usize) -> &str {
    if max_bytes >= s.len() {
        return s;
    }
    let mut end = max_bytes;
    while end > 0 && !s.is_char_boundary(end) {
        end -= 1;
    }
    &s[..end]
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::text::analyzer::SimpleAnalyzer;

    #[test]
    fn test_highlight_basic() {
        let analyzer = SimpleAnalyzer;
        let config = HighlightConfig::default();
        let query = FtsQuery::Term("database".to_string());

        let result = highlight(
            "A database is a collection of data. The database stores records.",
            &query,
            &analyzer,
            &config,
        );

        assert!(result.contains("<b>database</b>"));
    }

    #[test]
    fn test_highlight_no_match() {
        let analyzer = SimpleAnalyzer;
        let config = HighlightConfig::default();
        let query = FtsQuery::Term("xyz".to_string());

        let result = highlight("hello world", &query, &analyzer, &config);
        assert!(!result.contains("<b>"));
    }

    #[test]
    fn test_highlight_custom_tags() {
        let analyzer = SimpleAnalyzer;
        let config = HighlightConfig {
            pre_tag: "<mark>".to_string(),
            post_tag: "</mark>".to_string(),
            ..Default::default()
        };
        let query = FtsQuery::Term("hello".to_string());

        let result = highlight("hello world", &query, &analyzer, &config);
        assert!(result.contains("<mark>hello</mark>"));
    }

    #[test]
    fn test_highlight_phrase() {
        let analyzer = SimpleAnalyzer;
        let config = HighlightConfig::default();
        let query = FtsQuery::Phrase(vec!["quick".to_string(), "brown".to_string()]);

        let result = highlight("the quick brown fox jumps", &query, &analyzer, &config);

        assert!(result.contains("<b>quick</b>"));
        assert!(result.contains("<b>brown</b>"));
    }

    #[test]
    fn test_highlight_empty_text() {
        let analyzer = SimpleAnalyzer;
        let config = HighlightConfig::default();
        let query = FtsQuery::Term("test".to_string());

        let result = highlight("", &query, &analyzer, &config);
        assert_eq!(result, "");
    }

    #[test]
    fn test_extract_query_terms_boolean() {
        let query = FtsQuery::Boolean {
            must: vec![FtsQuery::Term("a".to_string())],
            should: vec![FtsQuery::Term("b".to_string())],
            must_not: vec![FtsQuery::Term("c".to_string())],
        };
        let terms = extract_query_terms(&query);
        assert!(terms.contains(&"a".to_string()));
        assert!(terms.contains(&"b".to_string()));
        // must_not terms are not extracted for highlighting
        assert!(!terms.contains(&"c".to_string()));
    }

    #[test]
    fn test_fragment_splitting() {
        let text = "First sentence. Second sentence. Third sentence. Fourth sentence.";
        let fragments = split_into_fragments(text, 30);
        assert!(fragments.len() >= 2);
    }

    #[test]
    fn test_safe_truncate() {
        let s = "hello world";
        assert_eq!(safe_truncate(s, 5), "hello");
        assert_eq!(safe_truncate(s, 100), "hello world");

        // Multi-byte: ensure no mid-char split
        let s = "caf\u{00E9}";
        let truncated = safe_truncate(s, 4);
        assert!(truncated.len() <= 4);
    }
}
