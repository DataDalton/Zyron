//! Text analysis pipeline for full-text search.
//!
//! Provides tokenizers that break text into tokens, filters that transform
//! token streams (lowercase, stopword removal, stemming), and composed
//! analyzers that chain a tokenizer with filters. Includes a hand-rolled
//! Porter stemmer for English and language detection via trigram frequencies.

use std::collections::{HashMap, HashSet};
use zyron_common::Result;

/// A single token produced by text analysis.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Token {
    pub term: String,
    pub position: u32,
    pub start_offset: u32,
    pub end_offset: u32,
}

/// Breaks raw text into a stream of tokens.
pub trait Tokenizer: Send + Sync {
    fn tokenize(&self, text: &str) -> Vec<Token>;
}

/// Transforms a token stream (e.g., lowercase, remove stopwords, stem).
pub trait TokenFilter: Send + Sync {
    fn filter(&self, tokens: Vec<Token>) -> Vec<Token>;
}

/// Composed analysis pipeline: tokenizer + filters.
pub trait Analyzer: Send + Sync {
    fn analyze(&self, text: &str) -> Vec<Token>;

    /// Zero-allocation analysis into a reusable buffer. All processed terms
    /// are concatenated into `buf.terms`, with `buf.tokens` recording
    /// (offset, length, position) ranges. After warmup, no heap allocations
    /// occur per document. The default implementation falls back to `analyze()`.
    fn analyze_into(&self, text: &str, buf: &mut AnalysisBuffer) {
        buf.clear();
        for token in self.analyze(text) {
            buf.push_token(&token.term, token.position);
        }
    }

    fn name(&self) -> &str;
}

// ---------------------------------------------------------------------------
// AnalysisBuffer: reusable zero-allocation token output
// ---------------------------------------------------------------------------

/// Reusable buffer for analyzer output. All terms are concatenated into a
/// single String. Tokens record (byte_offset, byte_length, position).
/// After initial warmup allocation, reuse across documents is allocation-free.
pub struct AnalysisBuffer {
    /// Concatenated processed terms.
    pub terms: String,
    /// (byte_offset_in_terms, byte_length, position) per token.
    pub tokens: Vec<(u32, u16, u32)>,
}

impl AnalysisBuffer {
    pub fn new() -> Self {
        Self {
            terms: String::with_capacity(512),
            tokens: Vec::with_capacity(64),
        }
    }

    pub fn clear(&mut self) {
        self.terms.clear();
        self.tokens.clear();
    }

    #[inline]
    pub fn push_token(&mut self, term: &str, position: u32) {
        let offset = self.terms.len() as u32;
        self.terms.push_str(term);
        self.tokens.push((offset, term.len() as u16, position));
    }

    #[inline]
    pub fn term_at(&self, idx: usize) -> &str {
        let (off, len, _) = self.tokens[idx];
        &self.terms[off as usize..(off as usize + len as usize)]
    }

    #[inline]
    pub fn position_at(&self, idx: usize) -> u32 {
        self.tokens[idx].2
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.tokens.len()
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.tokens.is_empty()
    }
}

impl Default for AnalysisBuffer {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Tokenizers
// ---------------------------------------------------------------------------

/// Splits text on whitespace boundaries only. Case-sensitive.
pub struct WhitespaceTokenizer;

impl Tokenizer for WhitespaceTokenizer {
    fn tokenize(&self, text: &str) -> Vec<Token> {
        let mut tokens = Vec::with_capacity(text.len() / 5 + 1);
        let mut position = 0u32;
        let mut chars = text.char_indices().peekable();

        while let Some(&(start, ch)) = chars.peek() {
            if ch.is_whitespace() {
                chars.next();
                continue;
            }
            let token_start = start as u32;
            let mut end = start;
            while let Some(&(i, c)) = chars.peek() {
                if c.is_whitespace() {
                    break;
                }
                end = i + c.len_utf8();
                chars.next();
            }
            let term = &text[start..end];
            if !term.is_empty() {
                tokens.push(Token {
                    term: term.to_string(),
                    position,
                    start_offset: token_start,
                    end_offset: end as u32,
                });
                position += 1;
            }
        }
        tokens
    }
}

/// Splits text on whitespace and punctuation, producing word tokens.
/// Handles Unicode letters and digits as token characters.
pub struct StandardTokenizer;

impl StandardTokenizer {
    fn is_token_char(c: char) -> bool {
        c.is_alphanumeric() || c == '_'
    }
}

impl Tokenizer for StandardTokenizer {
    fn tokenize(&self, text: &str) -> Vec<Token> {
        let mut tokens = Vec::with_capacity(text.len() / 5 + 1);
        let mut position = 0u32;
        let mut chars = text.char_indices().peekable();

        while let Some(&(start, ch)) = chars.peek() {
            if !Self::is_token_char(ch) {
                chars.next();
                continue;
            }
            let token_start = start as u32;
            let mut end = start;
            while let Some(&(i, c)) = chars.peek() {
                if !Self::is_token_char(c) {
                    break;
                }
                end = i + c.len_utf8();
                chars.next();
            }
            let term = &text[start..end];
            if !term.is_empty() {
                tokens.push(Token {
                    term: term.to_string(),
                    position,
                    start_offset: token_start,
                    end_offset: end as u32,
                });
                position += 1;
            }
        }
        tokens
    }
}

/// Produces overlapping 2-character windows for CJK text.
/// Non-CJK characters are tokenized as standard words.
pub struct CharBigramTokenizer;

impl CharBigramTokenizer {
    fn is_cjk(c: char) -> bool {
        matches!(c,
            '\u{4E00}'..='\u{9FFF}' |   // CJK Unified Ideographs
            '\u{3400}'..='\u{4DBF}' |   // CJK Extension A
            '\u{3040}'..='\u{309F}' |   // Hiragana
            '\u{30A0}'..='\u{30FF}' |   // Katakana
            '\u{F900}'..='\u{FAFF}' |   // CJK Compatibility Ideographs
            '\u{AC00}'..='\u{D7AF}'     // Hangul Syllables
        )
    }
}

impl Tokenizer for CharBigramTokenizer {
    fn tokenize(&self, text: &str) -> Vec<Token> {
        let mut tokens = Vec::new();
        let mut position = 0u32;
        let chars: Vec<(usize, char)> = text.char_indices().collect();
        let mut i = 0;

        while i < chars.len() {
            let (start, ch) = chars[i];

            // CJK character: emit bigrams
            if Self::is_cjk(ch) {
                if i + 1 < chars.len() && Self::is_cjk(chars[i + 1].1) {
                    let end = chars[i + 1].0 + chars[i + 1].1.len_utf8();
                    let term = &text[start..end];
                    tokens.push(Token {
                        term: term.to_string(),
                        position,
                        start_offset: start as u32,
                        end_offset: end as u32,
                    });
                    position += 1;
                } else {
                    // Single CJK char at end, emit as unigram
                    let end = start + ch.len_utf8();
                    tokens.push(Token {
                        term: text[start..end].to_string(),
                        position,
                        start_offset: start as u32,
                        end_offset: end as u32,
                    });
                    position += 1;
                }
                i += 1;
                continue;
            }

            // Non-CJK: collect alphanumeric word
            if ch.is_alphanumeric() || ch == '_' {
                let mut end = start + ch.len_utf8();
                let mut j = i + 1;
                while j < chars.len() {
                    let (idx, c) = chars[j];
                    if !c.is_alphanumeric() && c != '_' {
                        break;
                    }
                    end = idx + c.len_utf8();
                    j += 1;
                }
                let term = &text[start..end];
                if !term.is_empty() {
                    tokens.push(Token {
                        term: term.to_string(),
                        position,
                        start_offset: start as u32,
                        end_offset: end as u32,
                    });
                    position += 1;
                }
                i = j;
                continue;
            }

            // Skip whitespace and punctuation
            i += 1;
        }
        tokens
    }
}

// ---------------------------------------------------------------------------
// Token Filters
// ---------------------------------------------------------------------------

/// Lowercases all token terms (Unicode-aware).
pub struct LowercaseFilter;

impl TokenFilter for LowercaseFilter {
    fn filter(&self, tokens: Vec<Token>) -> Vec<Token> {
        tokens
            .into_iter()
            .map(|mut t| {
                if t.term.is_ascii() {
                    t.term.make_ascii_lowercase();
                } else {
                    t.term = t.term.to_lowercase();
                }
                t
            })
            .collect()
    }
}

/// Removes tokens whose terms appear in the stopword set.
pub struct StopwordFilter {
    stopwords: &'static HashSet<&'static str>,
}

static ENGLISH_STOPWORD_SET: std::sync::LazyLock<HashSet<&'static str>> =
    std::sync::LazyLock::new(|| ENGLISH_STOPWORDS.iter().copied().collect());
static SPANISH_STOPWORD_SET: std::sync::LazyLock<HashSet<&'static str>> =
    std::sync::LazyLock::new(|| SPANISH_STOPWORDS.iter().copied().collect());
static GERMAN_STOPWORD_SET: std::sync::LazyLock<HashSet<&'static str>> =
    std::sync::LazyLock::new(|| GERMAN_STOPWORDS.iter().copied().collect());
static FRENCH_STOPWORD_SET: std::sync::LazyLock<HashSet<&'static str>> =
    std::sync::LazyLock::new(|| FRENCH_STOPWORDS.iter().copied().collect());
static EMPTY_STOPWORD_SET: std::sync::LazyLock<HashSet<&'static str>> =
    std::sync::LazyLock::new(HashSet::new);

impl StopwordFilter {
    pub fn new(language: Language) -> Self {
        let stopwords = match language {
            Language::English => &*ENGLISH_STOPWORD_SET,
            Language::Spanish => &*SPANISH_STOPWORD_SET,
            Language::German => &*GERMAN_STOPWORD_SET,
            Language::French => &*FRENCH_STOPWORD_SET,
            _ => &*EMPTY_STOPWORD_SET,
        };
        Self { stopwords }
    }

    pub fn from_static_set(set: &'static HashSet<&'static str>) -> Self {
        Self { stopwords: set }
    }
}

impl TokenFilter for StopwordFilter {
    fn filter(&self, tokens: Vec<Token>) -> Vec<Token> {
        tokens
            .into_iter()
            .filter(|t| !self.stopwords.contains(t.term.as_str()))
            .collect()
    }
}

/// Porter stemmer for English. Hand-rolled implementation of the 5-step
/// algorithm from Martin Porter's 1980 paper. No external dependencies.
pub struct PorterStemmerFilter;

impl TokenFilter for PorterStemmerFilter {
    fn filter(&self, tokens: Vec<Token>) -> Vec<Token> {
        tokens
            .into_iter()
            .map(|mut t| {
                t.term = porter_stem(&t.term);
                t
            })
            .collect()
    }
}

/// Generates n-grams of configurable length from each token.
pub struct NGramFilter {
    pub min_gram: usize,
    pub max_gram: usize,
}

impl NGramFilter {
    pub fn new(min_gram: usize, max_gram: usize) -> Self {
        Self { min_gram, max_gram }
    }
}

impl TokenFilter for NGramFilter {
    fn filter(&self, tokens: Vec<Token>) -> Vec<Token> {
        let mut result = Vec::new();
        for token in &tokens {
            let chars: Vec<char> = token.term.chars().collect();
            for n in self.min_gram..=self.max_gram {
                if n > chars.len() {
                    break;
                }
                for start in 0..=chars.len() - n {
                    let ngram: String = chars[start..start + n].iter().collect();
                    result.push(Token {
                        term: ngram,
                        position: token.position,
                        start_offset: token.start_offset,
                        end_offset: token.end_offset,
                    });
                }
            }
        }
        result
    }
}

/// Generates prefix n-grams (edge n-grams) from each token.
/// Useful for autocomplete indexing.
pub struct EdgeNGramFilter {
    pub min_gram: usize,
    pub max_gram: usize,
}

impl EdgeNGramFilter {
    pub fn new(min_gram: usize, max_gram: usize) -> Self {
        Self { min_gram, max_gram }
    }
}

impl TokenFilter for EdgeNGramFilter {
    fn filter(&self, tokens: Vec<Token>) -> Vec<Token> {
        let mut result = Vec::new();
        for token in &tokens {
            let chars: Vec<char> = token.term.chars().collect();
            for n in self.min_gram..=self.max_gram.min(chars.len()) {
                let edge: String = chars[..n].iter().collect();
                result.push(Token {
                    term: edge,
                    position: token.position,
                    start_offset: token.start_offset,
                    end_offset: token.end_offset,
                });
            }
        }
        result
    }
}

/// Expands tokens using a synonym mapping. Each token that matches a
/// synonym group emits additional tokens for all synonyms in the group.
pub struct SynonymFilter {
    expansions: HashMap<String, Vec<String>>,
}

impl SynonymFilter {
    pub fn new(expansions: HashMap<String, Vec<String>>) -> Self {
        Self { expansions }
    }
}

impl TokenFilter for SynonymFilter {
    fn filter(&self, tokens: Vec<Token>) -> Vec<Token> {
        let mut result = Vec::new();
        for token in tokens {
            if let Some(synonyms) = self.expansions.get(&token.term) {
                for syn in synonyms {
                    result.push(Token {
                        term: syn.clone(),
                        position: token.position,
                        start_offset: token.start_offset,
                        end_offset: token.end_offset,
                    });
                }
            }
            result.push(token);
        }
        result
    }
}

// ---------------------------------------------------------------------------
// Analyzers
// ---------------------------------------------------------------------------

/// Standard analyzer: StandardTokenizer + Lowercase + Stopword(English) + PorterStemmer.
pub struct StandardAnalyzer;

impl Analyzer for StandardAnalyzer {
    fn analyze(&self, text: &str) -> Vec<Token> {
        let tokenizer = StandardTokenizer;
        let tokens = tokenizer.tokenize(text);
        let tokens = LowercaseFilter.filter(tokens);
        let tokens = StopwordFilter::new(Language::English).filter(tokens);
        PorterStemmerFilter.filter(tokens)
    }

    fn analyze_into(&self, text: &str, buf: &mut AnalysisBuffer) {
        buf.clear();
        let stopwords = &*ENGLISH_STOPWORD_SET;
        let mut position = 0u32;
        let bytes = text.as_bytes();
        let mut i = 0;
        // Thread-local scratch buffer eliminates per-call heap allocation.
        // After the first call, capacity is reused across all subsequent calls.
        thread_local! { static SCRATCH: std::cell::RefCell<String> = const { std::cell::RefCell::new(String::new()) }; }
        SCRATCH.with_borrow_mut(|scratch| {
            while i < bytes.len() {
                if !StandardTokenizer::is_token_char(bytes[i] as char) {
                    i += 1;
                    continue;
                }
                let start = i;
                while i < bytes.len() && StandardTokenizer::is_token_char(bytes[i] as char) {
                    i += 1;
                }
                // Lowercase into scratch buffer (zero alloc after warmup)
                scratch.clear();
                for &b in &bytes[start..i] {
                    scratch.push(b.to_ascii_lowercase() as char);
                }
                // Stopword filter. English stopwords are at most 10 chars,
                // so skip the hash lookup for longer tokens.
                if scratch.len() <= 10 && stopwords.contains(scratch.as_str()) {
                    continue;
                }
                // Stem in-place into scratch buffer
                porter_stem_inplace(scratch);
                buf.push_token(scratch, position);
                position += 1;
            }
        }); // end SCRATCH.with_borrow_mut
    }

    fn name(&self) -> &str {
        "standard"
    }
}

/// Simple analyzer: StandardTokenizer + Lowercase. No stemming or stopwords.
pub struct SimpleAnalyzer;

impl Analyzer for SimpleAnalyzer {
    fn analyze(&self, text: &str) -> Vec<Token> {
        let tokenizer = StandardTokenizer;
        let tokens = tokenizer.tokenize(text);
        LowercaseFilter.filter(tokens)
    }

    /// Zero-allocation simple analysis: tokenize + lowercase directly into buffer.
    /// No per-token String allocation after warmup.
    fn analyze_into(&self, text: &str, buf: &mut AnalysisBuffer) {
        buf.clear();
        let mut position = 0u32;
        let bytes = text.as_bytes();
        let mut i = 0;
        while i < bytes.len() {
            if !StandardTokenizer::is_token_char(bytes[i] as char) {
                i += 1;
                continue;
            }
            let start = i;
            while i < bytes.len() && StandardTokenizer::is_token_char(bytes[i] as char) {
                i += 1;
            }
            // Write lowercased term directly into buffer
            let offset = buf.terms.len() as u32;
            for &b in &bytes[start..i] {
                buf.terms.push(b.to_ascii_lowercase() as char);
            }
            let len = (buf.terms.len() as u32 - offset) as u16;
            buf.tokens.push((offset, len, position));
            position += 1;
        }
    }

    fn name(&self) -> &str {
        "simple"
    }
}

/// Whitespace analyzer: WhitespaceTokenizer only. Case-sensitive.
pub struct WhitespaceAnalyzer;

impl Analyzer for WhitespaceAnalyzer {
    fn analyze(&self, text: &str) -> Vec<Token> {
        WhitespaceTokenizer.tokenize(text)
    }

    fn name(&self) -> &str {
        "whitespace"
    }
}

/// User-configured analysis pipeline with arbitrary tokenizer + filter chain.
pub struct CustomAnalyzer {
    analyzer_name: String,
    tokenizer: Box<dyn Tokenizer>,
    filters: Vec<Box<dyn TokenFilter>>,
}

impl CustomAnalyzer {
    pub fn new(
        name: String,
        tokenizer: Box<dyn Tokenizer>,
        filters: Vec<Box<dyn TokenFilter>>,
    ) -> Self {
        Self {
            analyzer_name: name,
            tokenizer,
            filters,
        }
    }
}

impl Analyzer for CustomAnalyzer {
    fn analyze(&self, text: &str) -> Vec<Token> {
        let mut tokens = self.tokenizer.tokenize(text);
        for filter in &self.filters {
            tokens = filter.filter(tokens);
        }
        tokens
    }

    fn name(&self) -> &str {
        &self.analyzer_name
    }
}

// ---------------------------------------------------------------------------
// Language Support
// ---------------------------------------------------------------------------

/// Supported languages for text analysis.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Language {
    English,
    Spanish,
    German,
    French,
    ChineseSimplified,
    Japanese,
}

impl Language {
    /// Parses a language name string into a Language enum.
    pub fn from_str(s: &str) -> Result<Self> {
        match s.to_lowercase().as_str() {
            "english" | "en" => Ok(Language::English),
            "spanish" | "es" => Ok(Language::Spanish),
            "german" | "de" => Ok(Language::German),
            "french" | "fr" => Ok(Language::French),
            "chinese" | "zh" => Ok(Language::ChineseSimplified),
            "japanese" | "ja" => Ok(Language::Japanese),
            other => Err(zyron_common::ZyronError::FtsUnsupportedLanguage(
                other.to_string(),
            )),
        }
    }
}

/// Creates an analyzer configured for the given language.
pub fn analyzer_for_language(lang: Language) -> Box<dyn Analyzer> {
    match lang {
        Language::English => Box::new(StandardAnalyzer),
        Language::Spanish => Box::new(LanguageAnalyzer {
            language: Language::Spanish,
        }),
        Language::German => Box::new(LanguageAnalyzer {
            language: Language::German,
        }),
        Language::French => Box::new(LanguageAnalyzer {
            language: Language::French,
        }),
        Language::ChineseSimplified => Box::new(CjkAnalyzer),
        Language::Japanese => Box::new(CjkAnalyzer),
    }
}

/// Analyzer for European languages: StandardTokenizer + Lowercase + language stopwords.
struct LanguageAnalyzer {
    language: Language,
}

impl Analyzer for LanguageAnalyzer {
    fn analyze(&self, text: &str) -> Vec<Token> {
        let tokenizer = StandardTokenizer;
        let tokens = tokenizer.tokenize(text);
        let tokens = LowercaseFilter.filter(tokens);
        StopwordFilter::new(self.language).filter(tokens)
    }

    fn name(&self) -> &str {
        match self.language {
            Language::Spanish => "spanish",
            Language::German => "german",
            Language::French => "french",
            _ => "language",
        }
    }
}

/// CJK analyzer: CharBigramTokenizer + Lowercase.
struct CjkAnalyzer;

impl Analyzer for CjkAnalyzer {
    fn analyze(&self, text: &str) -> Vec<Token> {
        let tokenizer = CharBigramTokenizer;
        let tokens = tokenizer.tokenize(text);
        LowercaseFilter.filter(tokens)
    }

    fn name(&self) -> &str {
        "cjk"
    }
}

// ---------------------------------------------------------------------------
// Language Detection
// ---------------------------------------------------------------------------

/// Detects the language of a text sample using trigram frequency analysis.
pub struct LanguageDetector;

impl LanguageDetector {
    /// Detects the most likely language from the given text.
    /// Returns English as default for short or ambiguous text.
    pub fn detect(text: &str) -> Language {
        if text.len() < 20 {
            return Language::English;
        }

        // Check for CJK characters first
        let cjk_count = text
            .chars()
            .filter(|c| CharBigramTokenizer::is_cjk(*c))
            .count();
        let total_chars = text.chars().count();
        if total_chars > 0 && cjk_count * 100 / total_chars > 30 {
            // Distinguish Chinese from Japanese by checking for Hiragana/Katakana
            let has_kana = text
                .chars()
                .any(|c| matches!(c, '\u{3040}'..='\u{309F}' | '\u{30A0}'..='\u{30FF}'));
            if has_kana {
                return Language::Japanese;
            }
            return Language::ChineseSimplified;
        }

        let trigrams = extract_trigrams(text);
        if trigrams.is_empty() {
            return Language::English;
        }

        let languages = [
            (Language::English, &ENGLISH_TRIGRAMS[..]),
            (Language::Spanish, &SPANISH_TRIGRAMS[..]),
            (Language::German, &GERMAN_TRIGRAMS[..]),
            (Language::French, &FRENCH_TRIGRAMS[..]),
        ];

        let mut best_lang = Language::English;
        let mut best_score = f64::MAX;

        for (lang, reference) in &languages {
            let ref_set: HashSet<&str> = reference.iter().copied().collect();
            let mut distance = 0.0f64;
            for (tri, _freq) in &trigrams {
                if !ref_set.contains(tri.as_str()) {
                    distance += 1.0;
                }
            }
            // Normalize by number of trigrams
            let normalized = distance / trigrams.len() as f64;
            if normalized < best_score {
                best_score = normalized;
                best_lang = *lang;
            }
        }

        best_lang
    }
}

fn extract_trigrams(text: &str) -> Vec<(String, u32)> {
    let lower = text.to_lowercase();
    let chars: Vec<char> = lower.chars().collect();
    if chars.len() < 3 {
        return Vec::new();
    }

    let mut freq: HashMap<String, u32> = HashMap::new();
    for window in chars.windows(3) {
        let trigram: String = window.iter().collect();
        *freq.entry(trigram).or_insert(0) += 1;
    }

    let mut sorted: Vec<(String, u32)> = freq.into_iter().collect();
    sorted.sort_by(|a, b| b.1.cmp(&a.1));
    sorted.truncate(300);
    sorted
}

/// Creates an analyzer from a name string (e.g., from SQL WITH options).
pub fn analyzer_from_name(name: &str) -> Result<Box<dyn Analyzer>> {
    match name.to_lowercase().as_str() {
        "standard" => Ok(Box::new(StandardAnalyzer)),
        "simple" => Ok(Box::new(SimpleAnalyzer)),
        "whitespace" => Ok(Box::new(WhitespaceAnalyzer)),
        "cjk" => Ok(Box::new(CjkAnalyzer)),
        other => Err(zyron_common::ZyronError::FtsAnalyzerError(format!(
            "unknown analyzer: {other}"
        ))),
    }
}

// ---------------------------------------------------------------------------
// Porter Stemmer
// ---------------------------------------------------------------------------

/// Porter stemmer implementation. Applies the 5-step suffix stripping
/// algorithm to produce word stems for English text.
pub fn porter_stem(word: &str) -> String {
    if word.len() <= 2 {
        return word.to_string();
    }

    let mut stem = word.to_string();

    // Step 1a: plural forms
    stem = step_1a(&stem);
    // Step 1b: -eed, -ed, -ing
    stem = step_1b(&stem);
    // Step 1c: y -> i
    stem = step_1c(&stem);
    // Step 2: double suffixes
    stem = step_2(&stem);
    // Step 3: -icate, -ful, etc.
    stem = step_3(&stem);
    // Step 4: -al, -ance, etc.
    stem = step_4(&stem);
    // Step 5: final cleanup
    stem = step_5(&stem);

    stem
}

/// In-place Porter stemmer. Modifies the buffer directly instead of allocating
/// new Strings per step. After warmup, zero heap allocations.
pub fn porter_stem_inplace(buf: &mut String) {
    if buf.len() <= 2 {
        return;
    }
    ip_step_1a(buf);
    ip_step_1b(buf);
    ip_step_1c(buf);
    ip_step_2(buf);
    ip_step_3(buf);
    ip_step_4(buf);
    ip_step_5(buf);
}

fn ip_replace_suffix(buf: &mut String, suffix: &str, replacement: &str) {
    let new_len = buf.len() - suffix.len();
    buf.truncate(new_len);
    buf.push_str(replacement);
}

fn ip_step_1a(buf: &mut String) {
    if buf.ends_with("sses") {
        ip_replace_suffix(buf, "sses", "ss");
    } else if buf.ends_with("ies") {
        ip_replace_suffix(buf, "ies", "i");
    } else if buf.ends_with("ss") { /* keep */
    } else if buf.ends_with('s') && buf.len() > 1 {
        buf.pop();
    }
}

fn ip_step_1b(buf: &mut String) {
    if buf.ends_with("eed") {
        let stem_len = buf.len() - 3;
        if measure(&buf[..stem_len]) > 0 {
            ip_replace_suffix(buf, "eed", "ee");
        }
        return;
    }

    let (suffix_len, has_vowel) = if buf.ends_with("ed") {
        let stem = &buf[..buf.len() - 2];
        (2, contains_vowel(stem))
    } else if buf.ends_with("ing") {
        let stem = &buf[..buf.len() - 3];
        (3, contains_vowel(stem))
    } else {
        return;
    };

    if !has_vowel {
        return;
    }
    buf.truncate(buf.len() - suffix_len);

    if buf.ends_with("at") || buf.ends_with("bl") || buf.ends_with("iz") {
        buf.push('e');
    } else if ends_double_consonant(buf) {
        let last = buf.as_bytes()[buf.len() - 1];
        if last != b'l' && last != b's' && last != b'z' {
            buf.pop();
        }
    } else if measure(buf) == 1 && ends_cvc(buf) {
        buf.push('e');
    }
}

fn ip_step_1c(buf: &mut String) {
    if buf.ends_with('y') && buf.len() > 1 && contains_vowel(&buf[..buf.len() - 1]) {
        buf.pop();
        buf.push('i');
    }
}

fn ip_step_2(buf: &mut String) {
    let replacements: &[(&str, &str)] = &[
        ("ational", "ate"),
        ("tional", "tion"),
        ("enci", "ence"),
        ("anci", "ance"),
        ("izer", "ize"),
        ("abli", "able"),
        ("alli", "al"),
        ("entli", "ent"),
        ("eli", "e"),
        ("ousli", "ous"),
        ("ization", "ize"),
        ("ation", "ate"),
        ("ator", "ate"),
        ("alism", "al"),
        ("iveness", "ive"),
        ("fulness", "ful"),
        ("ousness", "ous"),
        ("aliti", "al"),
        ("iviti", "ive"),
        ("biliti", "ble"),
    ];
    for &(suffix, replacement) in replacements {
        if buf.ends_with(suffix) {
            let stem_len = buf.len() - suffix.len();
            if measure(&buf[..stem_len]) > 0 {
                ip_replace_suffix(buf, suffix, replacement);
            }
            return;
        }
    }
}

fn ip_step_3(buf: &mut String) {
    let replacements: &[(&str, &str)] = &[
        ("icate", "ic"),
        ("ative", ""),
        ("alize", "al"),
        ("iciti", "ic"),
        ("ical", "ic"),
        ("ful", ""),
        ("ness", ""),
    ];
    for &(suffix, replacement) in replacements {
        if buf.ends_with(suffix) {
            let stem_len = buf.len() - suffix.len();
            if measure(&buf[..stem_len]) > 0 {
                ip_replace_suffix(buf, suffix, replacement);
            }
            return;
        }
    }
}

fn ip_step_4(buf: &mut String) {
    let suffixes: &[&str] = &[
        "al", "ance", "ence", "er", "ic", "able", "ible", "ant", "ement", "ment", "ent", "ion",
        "ou", "ism", "ate", "iti", "ous", "ive", "ize",
    ];
    for suffix in suffixes {
        if buf.ends_with(suffix) {
            let stem_len = buf.len() - suffix.len();
            if *suffix == "ion" {
                if measure(&buf[..stem_len]) > 1 {
                    let last_byte = buf
                        .as_bytes()
                        .get(stem_len.wrapping_sub(1))
                        .copied()
                        .unwrap_or(0);
                    if last_byte == b's' || last_byte == b't' {
                        buf.truncate(stem_len);
                    }
                }
            } else if measure(&buf[..stem_len]) > 1 {
                buf.truncate(stem_len);
            }
            return;
        }
    }
}

fn ip_step_5(buf: &mut String) {
    // 5a: remove trailing -e
    if buf.ends_with('e') {
        let stem = &buf[..buf.len() - 1];
        let m = measure(stem);
        if m > 1 || (m == 1 && !ends_cvc(stem)) {
            buf.pop();
        }
    }
    // 5b: -ll -> -l if measure > 1
    if buf.ends_with("ll") && measure(&buf[..buf.len() - 1]) > 1 {
        buf.pop();
    }
}

/// Counts the "measure" of a word: the number of vowel-consonant sequences
/// in the stem. Used throughout the Porter algorithm to decide whether a
/// suffix can be removed. Operates on bytes directly to avoid Vec<char>
/// allocation (Porter stemming is ASCII-only).
fn measure(s: &str) -> usize {
    let b = s.as_bytes();
    if b.is_empty() {
        return 0;
    }

    let mut m = 0;
    let mut i = 0;

    // Skip initial consonants
    while i < b.len() && !is_vowel_byte(b, i) {
        i += 1;
    }

    loop {
        while i < b.len() && is_vowel_byte(b, i) {
            i += 1;
        }
        if i >= b.len() {
            return m;
        }
        while i < b.len() && !is_vowel_byte(b, i) {
            i += 1;
        }
        m += 1;
    }
}

#[inline(always)]
fn is_vowel_byte(b: &[u8], i: usize) -> bool {
    match b[i] {
        b'a' | b'e' | b'i' | b'o' | b'u' => true,
        b'y' => i > 0 && !matches!(b[i - 1], b'a' | b'e' | b'i' | b'o' | b'u'),
        _ => false,
    }
}

#[inline(always)]
fn contains_vowel(s: &str) -> bool {
    let b = s.as_bytes();
    (0..b.len()).any(|i| is_vowel_byte(b, i))
}

#[inline(always)]
fn ends_double_consonant(s: &str) -> bool {
    let b = s.as_bytes();
    if b.len() < 2 {
        return false;
    }
    let last = b.len() - 1;
    b[last] == b[last - 1] && !is_vowel_byte(b, last)
}

#[inline(always)]
fn ends_cvc(s: &str) -> bool {
    let b = s.as_bytes();
    if b.len() < 3 {
        return false;
    }
    let last = b.len() - 1;
    !is_vowel_byte(b, last)
        && is_vowel_byte(b, last - 1)
        && !is_vowel_byte(b, last - 2)
        && !matches!(b[last], b'w' | b'x' | b'y')
}

fn step_1a(word: &str) -> String {
    if let Some(stem) = word.strip_suffix("sses") {
        return format!("{stem}ss");
    }
    if let Some(stem) = word.strip_suffix("ies") {
        return format!("{stem}i");
    }
    if word.ends_with("ss") {
        return word.to_string();
    }
    if let Some(stem) = word.strip_suffix('s') {
        if !stem.is_empty() {
            return stem.to_string();
        }
    }
    word.to_string()
}

fn step_1b(word: &str) -> String {
    if let Some(stem) = word.strip_suffix("eed") {
        if measure(stem) > 0 {
            return format!("{stem}ee");
        }
        return word.to_string();
    }

    let (trimmed, found) = if let Some(stem) = word.strip_suffix("ed") {
        (stem, contains_vowel(stem))
    } else if let Some(stem) = word.strip_suffix("ing") {
        (stem, contains_vowel(stem))
    } else {
        return word.to_string();
    };

    if !found {
        return word.to_string();
    }

    let stem = trimmed.to_string();

    if stem.ends_with("at") || stem.ends_with("bl") || stem.ends_with("iz") {
        return format!("{stem}e");
    }

    if ends_double_consonant(&stem) {
        let last = stem.chars().last().unwrap_or(' ');
        if last != 'l' && last != 's' && last != 'z' {
            return stem[..stem.len() - last.len_utf8()].to_string();
        }
    }

    if measure(&stem) == 1 && ends_cvc(&stem) {
        return format!("{stem}e");
    }

    stem
}

fn step_1c(word: &str) -> String {
    if let Some(stem) = word.strip_suffix('y') {
        if contains_vowel(stem) && !stem.is_empty() {
            return format!("{stem}i");
        }
    }
    word.to_string()
}

fn step_2(word: &str) -> String {
    let replacements: &[(&str, &str)] = &[
        ("ational", "ate"),
        ("tional", "tion"),
        ("enci", "ence"),
        ("anci", "ance"),
        ("izer", "ize"),
        ("abli", "able"),
        ("alli", "al"),
        ("entli", "ent"),
        ("eli", "e"),
        ("ousli", "ous"),
        ("ization", "ize"),
        ("ation", "ate"),
        ("ator", "ate"),
        ("alism", "al"),
        ("iveness", "ive"),
        ("fulness", "ful"),
        ("ousness", "ous"),
        ("aliti", "al"),
        ("iviti", "ive"),
        ("biliti", "ble"),
    ];

    for (suffix, replacement) in replacements {
        if let Some(stem) = word.strip_suffix(suffix) {
            if measure(stem) > 0 {
                return format!("{stem}{replacement}");
            }
            return word.to_string();
        }
    }
    word.to_string()
}

fn step_3(word: &str) -> String {
    let replacements: &[(&str, &str)] = &[
        ("icate", "ic"),
        ("ative", ""),
        ("alize", "al"),
        ("iciti", "ic"),
        ("ical", "ic"),
        ("ful", ""),
        ("ness", ""),
    ];

    for (suffix, replacement) in replacements {
        if let Some(stem) = word.strip_suffix(suffix) {
            if measure(stem) > 0 {
                return format!("{stem}{replacement}");
            }
            return word.to_string();
        }
    }
    word.to_string()
}

fn step_4(word: &str) -> String {
    let suffixes: &[&str] = &[
        "al", "ance", "ence", "er", "ic", "able", "ible", "ant", "ement", "ment", "ent", "ion",
        "ou", "ism", "ate", "iti", "ous", "ive", "ize",
    ];

    for suffix in suffixes {
        if let Some(stem) = word.strip_suffix(suffix) {
            if *suffix == "ion" {
                // Special: -ion requires stem to end in s or t
                if measure(stem) > 1 && (stem.ends_with('s') || stem.ends_with('t')) {
                    return stem.to_string();
                }
            } else if measure(stem) > 1 {
                return stem.to_string();
            }
            return word.to_string();
        }
    }
    word.to_string()
}

fn step_5(word: &str) -> String {
    let mut result = word.to_string();

    // Step 5a: remove trailing -e
    if let Some(stem) = result.strip_suffix('e') {
        let m = measure(stem);
        if m > 1 || (m == 1 && !ends_cvc(stem)) {
            result = stem.to_string();
        }
    }

    // Step 5b: -ll -> -l if measure > 1
    if result.ends_with("ll") && measure(&result[..result.len() - 1]) > 1 {
        result.pop();
    }

    result
}

// ---------------------------------------------------------------------------
// Stopword Lists
// ---------------------------------------------------------------------------

const ENGLISH_STOPWORDS: &[&str] = &[
    "a",
    "about",
    "above",
    "after",
    "again",
    "against",
    "all",
    "am",
    "an",
    "and",
    "any",
    "are",
    "aren't",
    "as",
    "at",
    "be",
    "because",
    "been",
    "before",
    "being",
    "below",
    "between",
    "both",
    "but",
    "by",
    "can't",
    "cannot",
    "could",
    "couldn't",
    "did",
    "didn't",
    "do",
    "does",
    "doesn't",
    "doing",
    "don't",
    "down",
    "during",
    "each",
    "few",
    "for",
    "from",
    "further",
    "get",
    "got",
    "had",
    "hadn't",
    "has",
    "hasn't",
    "have",
    "haven't",
    "having",
    "he",
    "he'd",
    "he'll",
    "he's",
    "her",
    "here",
    "here's",
    "hers",
    "herself",
    "him",
    "himself",
    "his",
    "how",
    "how's",
    "i",
    "i'd",
    "i'll",
    "i'm",
    "i've",
    "if",
    "in",
    "into",
    "is",
    "isn't",
    "it",
    "it's",
    "its",
    "itself",
    "just",
    "let's",
    "me",
    "might",
    "more",
    "most",
    "mustn't",
    "my",
    "myself",
    "no",
    "nor",
    "not",
    "now",
    "of",
    "off",
    "on",
    "once",
    "only",
    "or",
    "other",
    "ought",
    "our",
    "ours",
    "ourselves",
    "out",
    "over",
    "own",
    "same",
    "shan't",
    "she",
    "she'd",
    "she'll",
    "she's",
    "should",
    "shouldn't",
    "so",
    "some",
    "such",
    "than",
    "that",
    "that's",
    "the",
    "their",
    "theirs",
    "them",
    "themselves",
    "then",
    "there",
    "there's",
    "these",
    "they",
    "they'd",
    "they'll",
    "they're",
    "they've",
    "this",
    "those",
    "through",
    "to",
    "too",
    "under",
    "until",
    "up",
    "us",
    "very",
    "was",
    "wasn't",
    "we",
    "we'd",
    "we'll",
    "we're",
    "we've",
    "were",
    "weren't",
    "what",
    "what's",
    "when",
    "when's",
    "where",
    "where's",
    "which",
    "while",
    "who",
    "who's",
    "whom",
    "why",
    "why's",
    "will",
    "with",
    "won't",
    "would",
    "wouldn't",
    "you",
    "you'd",
    "you'll",
    "you're",
    "you've",
    "your",
    "yours",
    "yourself",
    "yourselves",
];

const SPANISH_STOPWORDS: &[&str] = &[
    "a", "al", "algo", "algunas", "alguno", "algunos", "ante", "antes", "como", "con", "contra",
    "cual", "cuando", "de", "del", "desde", "donde", "durante", "e", "el", "ella", "ellas",
    "ellos", "en", "entre", "era", "esa", "esas", "ese", "eso", "esos", "esta", "estaba", "estado",
    "estar", "estas", "este", "esto", "estos", "fue", "ha", "hace", "hasta", "hay", "la", "las",
    "le", "les", "lo", "los", "me", "mi", "muy", "nada", "ni", "no", "nos", "nosotros", "nuestro",
    "o", "otra", "otras", "otro", "otros", "para", "pero", "por", "que", "se", "ser", "si", "sin",
    "sobre", "son", "su", "sus", "te", "ti", "tiene", "todo", "todos", "tu", "tus", "un", "una",
    "uno", "unos", "usted", "ustedes", "y", "ya", "yo",
];

const GERMAN_STOPWORDS: &[&str] = &[
    "aber",
    "alle",
    "allem",
    "allen",
    "aller",
    "allerdings",
    "alles",
    "also",
    "am",
    "an",
    "andere",
    "anderem",
    "anderen",
    "anderer",
    "anderes",
    "anderm",
    "andern",
    "anders",
    "auch",
    "auf",
    "aus",
    "bei",
    "beim",
    "bereits",
    "bin",
    "bis",
    "bist",
    "da",
    "damit",
    "dann",
    "das",
    "dass",
    "dazu",
    "dein",
    "deine",
    "deinem",
    "deinen",
    "deiner",
    "dem",
    "den",
    "denn",
    "der",
    "des",
    "die",
    "dies",
    "diese",
    "dieselbe",
    "diesem",
    "diesen",
    "dieser",
    "dieses",
    "dort",
    "du",
    "durch",
    "ein",
    "eine",
    "einem",
    "einen",
    "einer",
    "einige",
    "einigem",
    "einigen",
    "einiger",
    "einiges",
    "er",
    "es",
    "etwas",
    "euch",
    "euer",
    "eure",
    "eurem",
    "euren",
    "eurer",
    "fur",
    "gegen",
    "habe",
    "haben",
    "hat",
    "hatte",
    "hier",
    "hin",
    "ich",
    "ihm",
    "ihn",
    "ihnen",
    "ihr",
    "ihre",
    "ihrem",
    "ihren",
    "ihrer",
    "im",
    "in",
    "indem",
    "infolgedessen",
    "ins",
    "ist",
    "jede",
    "jedem",
    "jeden",
    "jeder",
    "jedes",
    "jene",
    "jenem",
    "jenen",
    "jener",
    "jenes",
    "jetzt",
    "kann",
    "kein",
    "keine",
    "keinem",
    "keinen",
    "keiner",
    "man",
    "manche",
    "manchem",
    "manchen",
    "mancher",
    "manches",
    "mein",
    "meine",
    "meinem",
    "meinen",
    "meiner",
    "mich",
    "mir",
    "mit",
    "muss",
    "nach",
    "nicht",
    "nichts",
    "noch",
    "nun",
    "nur",
    "ob",
    "oder",
    "ohne",
    "sehr",
    "sein",
    "seine",
    "seinem",
    "seinen",
    "seiner",
    "sich",
    "sie",
    "sind",
    "so",
    "solche",
    "solchem",
    "solchen",
    "solcher",
    "soll",
    "sollte",
    "sondern",
    "sonst",
    "um",
    "und",
    "uns",
    "unser",
    "unsere",
    "unserem",
    "unseren",
    "unserer",
    "unter",
    "viel",
    "vom",
    "von",
    "vor",
    "wahrend",
    "war",
    "warum",
    "was",
    "weil",
    "welche",
    "welchem",
    "welchen",
    "welcher",
    "wenn",
    "wer",
    "werde",
    "werden",
    "wie",
    "wieder",
    "will",
    "wir",
    "wird",
    "wo",
    "wollen",
    "worden",
    "wurde",
    "zu",
    "zum",
    "zur",
    "zwar",
    "zwischen",
];

const FRENCH_STOPWORDS: &[&str] = &[
    "a", "ai", "au", "aux", "avec", "c", "ce", "ces", "dans", "de", "des", "du", "elle", "elles",
    "en", "est", "et", "eu", "eux", "fait", "il", "ils", "j", "je", "l", "la", "le", "les", "leur",
    "leurs", "lui", "m", "ma", "mais", "me", "mes", "moi", "mon", "n", "ne", "ni", "nos", "notre",
    "nous", "on", "ont", "ou", "par", "pas", "plus", "pour", "qu", "que", "qui", "s", "sa", "se",
    "ses", "si", "son", "sur", "t", "ta", "te", "tes", "toi", "ton", "tu", "un", "une", "vos",
    "votre", "vous", "y",
];

// ---------------------------------------------------------------------------
// Language Detection Trigrams (top 50 per language for compact detection)
// ---------------------------------------------------------------------------

const ENGLISH_TRIGRAMS: &[&str] = &[
    "the", "he ", "th ", "in ", "nd ", "an ", "er ", "ed ", "hat", "on ", "re ", "is ", "ng ",
    "es ", "or ", "of ", "at ", "en ", "to ", "it ", "al ", "nt ", "st ", "as ", "ar ", "ou ",
    "ll ", "le ", "se ", "ha ", "te ", "ion", "ing", "tio", "ent", "ati", "for", "his", "ter",
    "her", "tha", "was", "not", "ith", "are", "ver", "all", "com", "con", "pro",
];

const SPANISH_TRIGRAMS: &[&str] = &[
    "de ", "os ", "la ", "en ", "el ", "es ", "ue ", "as ", "on ", "que", " de", "ion", " la",
    "do ", "re ", "er ", "ar ", "aci", "al ", "ta ", "nte", "con", " el", "an ", " en", "te ",
    "ad ", "or ", "ra ", "da ", " co", "ci ", "ien", "se ", "lo ", "par", "las", "un ", "com",
    "est", "ent", "ida", "sta", " lo", "res", "por", "tra", "nos", "ado", "una",
];

const GERMAN_TRIGRAMS: &[&str] = &[
    "en ", "er ", "der", "die", "ein", "ich", "nd ", "ch ", "sch", "den", "in ", "te ", "ie ",
    "ge ", "und", " de", "ung", "eit", "ine", "nen", "ist", "che", "ber", "gen", "ver", " di",
    "auf", " ei", "ste", "das", " da", "uch", "hen", "an ", "ese", "ier", "ach", "lic", " ge",
    "ent", "ren", "tte", "nde", "and", "ter", "tig", "mit", "aus", " au", " un",
];

const FRENCH_TRIGRAMS: &[&str] = &[
    "es ", "de ", "le ", "ent", "ion", "les", " de", "re ", "on ", "nt ", "que", "la ", " le",
    "ne ", "en ", "ns ", " la", "tio", "ati", " co", "te ", "is ", "it ", "par", "ou ", " pa",
    "men", "eur", "ai ", "us ", " qu", "ur ", " en", "ons", "est", "ait", "con", " un", "se ",
    " pr", "eme", "er ", "ier", "ant", "ce ", "des", "une", "res", "our", "dan",
];

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_whitespace_tokenizer() {
        let tok = WhitespaceTokenizer;
        let tokens = tok.tokenize("hello world  foo");
        assert_eq!(tokens.len(), 3);
        assert_eq!(tokens[0].term, "hello");
        assert_eq!(tokens[1].term, "world");
        assert_eq!(tokens[2].term, "foo");
        assert_eq!(tokens[0].position, 0);
        assert_eq!(tokens[1].position, 1);
    }

    #[test]
    fn test_standard_tokenizer() {
        let tok = StandardTokenizer;
        let tokens = tok.tokenize("Hello, world! foo-bar");
        assert_eq!(tokens.len(), 4);
        assert_eq!(tokens[0].term, "Hello");
        assert_eq!(tokens[1].term, "world");
        assert_eq!(tokens[2].term, "foo");
        assert_eq!(tokens[3].term, "bar");
    }

    #[test]
    fn test_standard_tokenizer_unicode() {
        let tok = StandardTokenizer;
        let tokens = tok.tokenize("cafe\u{0301} na\u{00EF}ve");
        assert_eq!(tokens.len(), 2);
    }

    #[test]
    fn test_char_bigram_tokenizer_cjk() {
        let tok = CharBigramTokenizer;
        let tokens = tok.tokenize("\u{4F60}\u{597D}\u{4E16}\u{754C}");
        // 4 CJK chars -> 3 bigrams + 1 trailing unigram
        assert_eq!(tokens.len(), 4);
        assert_eq!(tokens[0].term, "\u{4F60}\u{597D}");
        assert_eq!(tokens[1].term, "\u{597D}\u{4E16}");
        assert_eq!(tokens[2].term, "\u{4E16}\u{754C}");
        assert_eq!(tokens[3].term, "\u{754C}");
    }

    #[test]
    fn test_char_bigram_mixed() {
        let tok = CharBigramTokenizer;
        let tokens = tok.tokenize("hello \u{4F60}\u{597D}");
        // "hello" + bigram + trailing unigram
        assert_eq!(tokens.len(), 3);
        assert_eq!(tokens[0].term, "hello");
        assert_eq!(tokens[1].term, "\u{4F60}\u{597D}");
        assert_eq!(tokens[2].term, "\u{597D}");
    }

    #[test]
    fn test_lowercase_filter() {
        let tokens = vec![Token {
            term: "Hello".to_string(),
            position: 0,
            start_offset: 0,
            end_offset: 5,
        }];
        let result = LowercaseFilter.filter(tokens);
        assert_eq!(result[0].term, "hello");
    }

    #[test]
    fn test_stopword_filter() {
        let filter = StopwordFilter::new(Language::English);
        let tokens = vec![
            Token {
                term: "the".to_string(),
                position: 0,
                start_offset: 0,
                end_offset: 3,
            },
            Token {
                term: "quick".to_string(),
                position: 1,
                start_offset: 4,
                end_offset: 9,
            },
            Token {
                term: "fox".to_string(),
                position: 2,
                start_offset: 10,
                end_offset: 13,
            },
        ];
        let result = filter.filter(tokens);
        assert_eq!(result.len(), 2);
        assert_eq!(result[0].term, "quick");
        assert_eq!(result[1].term, "fox");
    }

    #[test]
    fn test_porter_stemmer_basic() {
        assert_eq!(porter_stem("caresses"), "caress");
        assert_eq!(porter_stem("ponies"), "poni");
        assert_eq!(porter_stem("cats"), "cat");
        assert_eq!(porter_stem("running"), "run");
        assert_eq!(porter_stem("agreed"), "agre");
    }

    #[test]
    fn test_porter_stemmer_step2() {
        assert_eq!(porter_stem("relational"), "relat");
        assert_eq!(porter_stem("conditional"), "condit");
        assert_eq!(porter_stem("digitizer"), "digit");
    }

    #[test]
    fn test_porter_stemmer_short_words() {
        assert_eq!(porter_stem("a"), "a");
        assert_eq!(porter_stem("an"), "an");
        assert_eq!(porter_stem("is"), "is");
    }

    #[test]
    fn test_ngram_filter() {
        let filter = NGramFilter::new(2, 3);
        let tokens = vec![Token {
            term: "hello".to_string(),
            position: 0,
            start_offset: 0,
            end_offset: 5,
        }];
        let result = filter.filter(tokens);
        // 2-grams: he, el, ll, lo = 4
        // 3-grams: hel, ell, llo = 3
        assert_eq!(result.len(), 7);
    }

    #[test]
    fn test_edge_ngram_filter() {
        let filter = EdgeNGramFilter::new(1, 4);
        let tokens = vec![Token {
            term: "hello".to_string(),
            position: 0,
            start_offset: 0,
            end_offset: 5,
        }];
        let result = filter.filter(tokens);
        assert_eq!(result.len(), 4);
        assert_eq!(result[0].term, "h");
        assert_eq!(result[1].term, "he");
        assert_eq!(result[2].term, "hel");
        assert_eq!(result[3].term, "hell");
    }

    #[test]
    fn test_standard_analyzer() {
        let analyzer = StandardAnalyzer;
        let tokens = analyzer.analyze("The quick brown foxes are jumping");
        // "the" and "are" removed by stopwords
        // remaining: "quick", "brown", "fox" (stemmed), "jump" (stemmed)
        let terms: Vec<&str> = tokens.iter().map(|t| t.term.as_str()).collect();
        assert!(!terms.contains(&"the"));
        assert!(!terms.contains(&"are"));
        assert!(terms.contains(&"quick"));
        assert!(terms.contains(&"brown"));
    }

    #[test]
    fn test_simple_analyzer() {
        let analyzer = SimpleAnalyzer;
        let tokens = analyzer.analyze("The Quick FOX");
        let terms: Vec<&str> = tokens.iter().map(|t| t.term.as_str()).collect();
        assert_eq!(terms, vec!["the", "quick", "fox"]);
    }

    #[test]
    fn test_whitespace_analyzer() {
        let analyzer = WhitespaceAnalyzer;
        let tokens = analyzer.analyze("The Quick FOX");
        let terms: Vec<&str> = tokens.iter().map(|t| t.term.as_str()).collect();
        assert_eq!(terms, vec!["The", "Quick", "FOX"]);
    }

    #[test]
    fn test_language_from_str() {
        assert_eq!(Language::from_str("english").unwrap(), Language::English);
        assert_eq!(Language::from_str("en").unwrap(), Language::English);
        assert_eq!(Language::from_str("spanish").unwrap(), Language::Spanish);
        assert!(Language::from_str("klingon").is_err());
    }

    #[test]
    fn test_language_detector_english() {
        let text = "The quick brown fox jumps over the lazy dog and then runs away quickly";
        let lang = LanguageDetector::detect(text);
        assert_eq!(lang, Language::English);
    }

    #[test]
    fn test_language_detector_short_text() {
        let text = "Hi";
        let lang = LanguageDetector::detect(text);
        assert_eq!(lang, Language::English); // default for short text
    }

    #[test]
    fn test_analyzer_from_name() {
        assert!(analyzer_from_name("standard").is_ok());
        assert!(analyzer_from_name("simple").is_ok());
        assert!(analyzer_from_name("whitespace").is_ok());
        assert!(analyzer_from_name("unknown").is_err());
    }

    #[test]
    fn test_synonym_filter() {
        let mut expansions = HashMap::new();
        expansions.insert(
            "db".to_string(),
            vec!["database".to_string(), "datastore".to_string()],
        );
        let filter = SynonymFilter::new(expansions);

        let tokens = vec![Token {
            term: "db".to_string(),
            position: 0,
            start_offset: 0,
            end_offset: 2,
        }];
        let result = filter.filter(tokens);
        assert_eq!(result.len(), 3);
        let terms: Vec<&str> = result.iter().map(|t| t.term.as_str()).collect();
        assert!(terms.contains(&"database"));
        assert!(terms.contains(&"datastore"));
        assert!(terms.contains(&"db"));
    }

    #[test]
    fn test_custom_analyzer() {
        let analyzer = CustomAnalyzer::new(
            "test".to_string(),
            Box::new(StandardTokenizer),
            vec![Box::new(LowercaseFilter)],
        );
        let tokens = analyzer.analyze("Hello WORLD");
        assert_eq!(tokens[0].term, "hello");
        assert_eq!(tokens[1].term, "world");
        assert_eq!(analyzer.name(), "test");
    }

    #[test]
    fn test_porter_stem_measure() {
        assert_eq!(measure("tr"), 0);
        assert_eq!(measure("ee"), 0);
        assert_eq!(measure("tree"), 0);
        assert_eq!(measure("trouble"), 1);
        assert_eq!(measure("oat"), 1);
        assert_eq!(measure("troubles"), 2);
    }

    #[test]
    fn test_empty_input() {
        let tok = StandardTokenizer;
        assert!(tok.tokenize("").is_empty());
        assert!(tok.tokenize("   ").is_empty());

        let analyzer = StandardAnalyzer;
        assert!(analyzer.analyze("").is_empty());
    }
}
