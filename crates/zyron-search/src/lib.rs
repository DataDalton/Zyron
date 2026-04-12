//! Full-text search engine for ZyronDB.
//!
//! Provides text analysis, inverted indexing, BM25 relevance scoring,
//! query parsing, result highlighting, autocomplete, and synonym expansion.
//! All text processing is hand-rolled with no external NLP dependencies.

pub mod graph;
pub mod text;
pub mod vector;

pub use text::analyzer::{
    AnalysisBuffer, Analyzer, CustomAnalyzer, SimpleAnalyzer, StandardAnalyzer, Token, TokenFilter,
    Tokenizer, WhitespaceAnalyzer,
};
pub use text::autocomplete::PrefixIndex;
pub use text::highlight::{HighlightConfig, highlight};
pub use text::inverted_index::{
    DocId, InvertedIndex, Posting, PostingsList, TermInfo, decode_doc_id, encode_doc_id,
};
pub use text::manager::FtsManager;
pub use text::query::{FtsQuery, FtsQueryParser};
pub use text::scoring::{Bm25Scorer, FieldBoost, RelevanceScorer};
pub use text::synonym::SynonymSet;
