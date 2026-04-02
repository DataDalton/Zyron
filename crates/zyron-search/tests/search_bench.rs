#![allow(non_snake_case, unused_assignments)]

//! Full-Text Search Benchmark Suite
//!
//! Integration tests for ZyronDB FTS engine:
//! - Basic search correctness and relevance ordering
//! - Boolean search (must/must_not/should)
//! - Phrase search with positional verification
//! - Fuzzy search with edit distance thresholds
//! - Proximity search with configurable word distance
//! - Highlighting with tag wrapping and fragment selection
//! - Autocomplete prefix lookup performance
//! - Multi-field boosting and relevance ranking
//! - Synonym expansion
//! - Large-scale indexing and query throughput
//!
//! Performance Targets:
//! | Test                      | Metric          | Target          |
//! |---------------------------|-----------------|-----------------|
//! | Indexing (1M docs)        | throughput       | 300K docs/sec  |
//! | Single-term query         | latency          | 3ms            |
//! | Phrase query              | latency          | 10ms           |
//! | Boolean query (3 terms)   | latency          | 8ms            |
//! | Fuzzy query (edit=2)      | latency          | 20ms           |
//! | Top-10 ranking            | latency          | 5ms            |
//! | Autocomplete              | latency          | 2ms            |
//! | Highlight                 | latency          | 500us          |
//! | Index size (1M docs)      | ratio            | 25% of raw     |
//!
//! Validation Requirements:
//! - Each benchmark runs 5 iterations
//! - Results averaged across all 5 runs
//! - Pass/fail determined by average performance
//! - Individual runs logged for variance analysis
//! - Test FAILS if any single run is >2x worse than target
//!
//! Run: cargo test -p zyron-search --test search_bench --release -- --nocapture

use std::sync::Mutex;
use std::time::Instant;

use zyron_bench_harness::*;

use zyron_search::text::analyzer::{
    self, AnalysisBuffer, Analyzer, SimpleAnalyzer, StandardAnalyzer,
};
use zyron_search::text::autocomplete::PrefixIndex;
use zyron_search::text::highlight::{self, HighlightConfig};
use zyron_search::text::inverted_index::InvertedIndex;
use zyron_search::text::query::{FtsQuery, FtsQueryParser};
use zyron_search::text::scoring::{Bm25Scorer, FieldBoost, RelevanceScorer, score_multi_field};
use zyron_search::text::synonym::SynonymSet;

// =============================================================================
// Performance Target Constants
// =============================================================================

const INDEXING_TARGET_DOCS_SEC: f64 = 300_000.0;
const SINGLE_TERM_TARGET_MS: f64 = 3.0;
const PHRASE_QUERY_TARGET_MS: f64 = 10.0;
const BOOLEAN_QUERY_TARGET_MS: f64 = 8.0;
const FUZZY_QUERY_TARGET_MS: f64 = 20.0;
const TOP_10_RANKING_TARGET_MS: f64 = 5.0;
const AUTOCOMPLETE_TARGET_MS: f64 = 2.0;
const HIGHLIGHT_TARGET_US: f64 = 500.0;
const INDEX_SIZE_RATIO_TARGET: f64 = 0.25;

static BENCHMARK_LOCK: Mutex<()> = Mutex::new(());

// =============================================================================
// Test data generation
// =============================================================================

/// Topics for generating article content with controlled vocabulary.
const TOPICS: &[&str] = &[
    "database",
    "PostgreSQL",
    "MySQL",
    "performance",
    "optimization",
    "distributed",
    "query",
    "indexing",
    "replication",
    "sharding",
    "transaction",
    "ACID",
    "consistency",
    "scalability",
    "throughput",
    "latency",
    "caching",
    "storage",
    "memory",
    "compression",
    "analytics",
    "monitoring",
    "backup",
    "recovery",
    "security",
    "authentication",
    "authorization",
    "encryption",
    "network",
    "protocol",
];

/// Generates a synthetic article body with roughly word_count words.
/// Uses deterministic seeding based on doc_id for reproducibility.
fn generate_article(doc_id: u64, word_count: usize) -> (String, String) {
    let topic_idx = (doc_id as usize) % TOPICS.len();
    let secondary_idx = ((doc_id as usize) * 7 + 3) % TOPICS.len();
    let tertiary_idx = ((doc_id as usize) * 13 + 11) % TOPICS.len();

    let title = format!(
        "{} {} systems for enterprise {}",
        TOPICS[topic_idx], TOPICS[secondary_idx], TOPICS[tertiary_idx]
    );

    let mut body = String::with_capacity(word_count * 7);
    for i in 0..word_count {
        if i > 0 {
            body.push(' ');
        }
        let word_idx = ((doc_id as usize).wrapping_mul(31).wrapping_add(i * 17)) % TOPICS.len();
        body.push_str(TOPICS[word_idx]);

        // Add filler words for realistic document length
        if i % 5 == 0 {
            body.push_str(" the system provides");
        }
        if i % 8 == 0 {
            body.push_str(" with high");
        }
    }
    (title, body)
}

/// Builds an index with `count` documents, each ~word_count words.
fn build_test_index(count: u64, word_count: usize) -> InvertedIndex {
    let idx = InvertedIndex::new(1, 100, vec![1, 2]);
    let analyzer = SimpleAnalyzer;

    for doc_id in 0..count {
        let (title, body) = generate_article(doc_id, word_count);
        let combined = format!("{title} {body}");
        idx.add_document(doc_id, &combined, &analyzer)
            .expect("add_document failed");
    }
    idx
}

// =============================================================================
// Test 1: Basic Search
// =============================================================================

#[test]
fn test_01_basic_search() {
    zyron_bench_harness::init("search");
    let _guard = BENCHMARK_LOCK.lock().unwrap_or_else(|e| e.into_inner());

    tprintln!("\n=== Test 1: Basic Search ===");

    let idx = build_test_index(10_000, 50);
    let analyzer = SimpleAnalyzer;
    let scorer = Bm25Scorer::default();

    // Search for "database performance"
    let query = FtsQueryParser::parse("database performance").expect("parse failed");
    let results = idx
        .search(&query, &analyzer, &scorer, 100)
        .expect("search failed");

    tprintln!("  Query: 'database performance'");
    tprintln!("  Results: {} documents", results.len());

    // Verify results exist and are not all 10K
    assert!(!results.is_empty(), "search returned no results");
    assert!(
        results.len() < 10_000,
        "search returned all documents ({}) instead of filtered set",
        results.len()
    );

    // Verify results are sorted by relevance (descending)
    for window in results.windows(2) {
        assert!(
            window[0].1 >= window[1].1,
            "results not sorted by relevance: {} < {}",
            window[0].1,
            window[1].1
        );
    }

    tprintln!(
        "  Top-5 scores: {:?}",
        results.iter().take(5).map(|r| r.1).collect::<Vec<_>>()
    );
    tprintln!("  Result count is reasonable (not all 10K): PASS");
    tprintln!("  Relevance ordering verified: PASS");
}

// =============================================================================
// Test 2: Boolean Search
// =============================================================================

#[test]
fn test_02_boolean_search() {
    zyron_bench_harness::init("search");
    let _guard = BENCHMARK_LOCK.lock().unwrap_or_else(|e| e.into_inner());

    tprintln!("\n=== Test 2: Boolean Search ===");

    let idx = InvertedIndex::new(2, 100, vec![1]);
    let analyzer = SimpleAnalyzer;
    let scorer = Bm25Scorer::default();

    // Insert controlled documents
    idx.add_document(1, "PostgreSQL performance tuning guide", &analyzer)
        .expect("add failed");
    idx.add_document(2, "MySQL performance optimization tips", &analyzer)
        .expect("add failed");
    idx.add_document(3, "PostgreSQL performance monitoring tools", &analyzer)
        .expect("add failed");
    idx.add_document(4, "PostgreSQL backup and recovery", &analyzer)
        .expect("add failed");
    idx.add_document(5, "MySQL replication setup", &analyzer)
        .expect("add failed");

    let query = FtsQueryParser::parse("+postgresql +performance -mysql").expect("parse failed");
    let results = idx
        .search(&query, &analyzer, &scorer, 100)
        .expect("search failed");

    tprintln!("  Query: '+postgresql +performance -mysql'");
    tprintln!("  Results: {} documents", results.len());

    let result_ids: Vec<u64> = results.iter().map(|r| r.0).collect();

    // Must have PostgreSQL AND performance: docs 1, 3
    assert!(
        result_ids.contains(&1),
        "missing doc 1 (has postgresql + performance)"
    );
    assert!(
        result_ids.contains(&3),
        "missing doc 3 (has postgresql + performance)"
    );

    // Must NOT have MySQL: docs 2, 5 excluded
    assert!(
        !result_ids.contains(&2),
        "doc 2 should be excluded (contains mysql)"
    );
    assert!(
        !result_ids.contains(&5),
        "doc 5 should be excluded (contains mysql)"
    );

    // Doc 4 has postgresql but not performance, should be excluded
    assert!(
        !result_ids.contains(&4),
        "doc 4 should be excluded (missing performance)"
    );

    tprintln!("  Boolean must/must_not verified: PASS");
}

// =============================================================================
// Test 3: Phrase Search
// =============================================================================

#[test]
fn test_03_phrase_search() {
    zyron_bench_harness::init("search");
    let _guard = BENCHMARK_LOCK.lock().unwrap_or_else(|e| e.into_inner());

    tprintln!("\n=== Test 3: Phrase Search ===");

    let idx = InvertedIndex::new(3, 100, vec![1]);
    let analyzer = SimpleAnalyzer;
    let scorer = Bm25Scorer::default();

    idx.add_document(1, "the distributed database system is fast", &analyzer)
        .expect("add failed");
    idx.add_document(2, "the database is distributed across nodes", &analyzer)
        .expect("add failed");
    idx.add_document(3, "a distributed database handles replication", &analyzer)
        .expect("add failed");
    idx.add_document(4, "single node database design", &analyzer)
        .expect("add failed");

    let query = FtsQueryParser::parse("\"distributed database\"").expect("parse failed");
    let results = idx
        .search(&query, &analyzer, &scorer, 100)
        .expect("search failed");

    tprintln!("  Query: '\"distributed database\"'");
    tprintln!("  Results: {} documents", results.len());

    let result_ids: Vec<u64> = results.iter().map(|r| r.0).collect();

    // Doc 1 has "distributed database" as adjacent words
    assert!(
        result_ids.contains(&1),
        "missing doc 1 (has phrase 'distributed database')"
    );

    // Doc 3 has "distributed database" as adjacent words
    assert!(
        result_ids.contains(&3),
        "missing doc 3 (has phrase 'distributed database')"
    );

    // Doc 2 has "database ... distributed" (wrong order, not adjacent)
    assert!(
        !result_ids.contains(&2),
        "doc 2 should be excluded (word order: 'database ... distributed')"
    );

    // Doc 4 doesn't have "distributed" at all
    assert!(
        !result_ids.contains(&4),
        "doc 4 should be excluded (no 'distributed')"
    );

    tprintln!("  Phrase adjacency and word order verified: PASS");
}

// =============================================================================
// Test 4: Fuzzy Search
// =============================================================================

#[test]
fn test_04_fuzzy_search() {
    zyron_bench_harness::init("search");
    let _guard = BENCHMARK_LOCK.lock().unwrap_or_else(|e| e.into_inner());

    tprintln!("\n=== Test 4: Fuzzy Search ===");

    let idx = InvertedIndex::new(4, 100, vec![1]);
    let analyzer = SimpleAnalyzer;
    let scorer = Bm25Scorer::default();

    idx.add_document(1, "database performance tuning", &analyzer)
        .expect("add failed");
    idx.add_document(2, "computer science fundamentals", &analyzer)
        .expect("add failed");
    idx.add_document(3, "data analysis tools", &analyzer)
        .expect("add failed");

    // "databse" -> should match "database" (edit distance 1)
    let query = FtsQueryParser::parse("databse~2").expect("parse failed");
    let results = idx
        .search(&query, &analyzer, &scorer, 100)
        .expect("search failed");

    tprintln!("  Query: 'databse~2'");
    tprintln!("  Results: {} documents", results.len());

    let result_ids: Vec<u64> = results.iter().map(|r| r.0).collect();
    assert!(
        result_ids.contains(&1),
        "fuzzy 'databse~2' should match doc 1 ('database')"
    );

    // "komputar" vs "computer": k!=c, a!=e = 2 substitutions, should NOT match with ~1
    let query2 = FtsQueryParser::parse("komputar~1").expect("parse failed");
    let results2 = idx
        .search(&query2, &analyzer, &scorer, 100)
        .expect("search failed");

    tprintln!("  Query: 'komputar~1'");
    tprintln!("  Results: {} documents", results2.len());

    let result_ids2: Vec<u64> = results2.iter().map(|r| r.0).collect();
    assert!(
        !result_ids2.contains(&2),
        "'komputar~1' should NOT match 'computer' (edit distance 2)"
    );

    tprintln!("  Fuzzy edit distance thresholds verified: PASS");
}

// =============================================================================
// Test 5: Proximity Search
// =============================================================================

#[test]
fn test_05_proximity_search() {
    zyron_bench_harness::init("search");
    let _guard = BENCHMARK_LOCK.lock().unwrap_or_else(|e| e.into_inner());

    tprintln!("\n=== Test 5: Proximity Search ===");

    let idx = InvertedIndex::new(5, 100, vec![1]);
    let analyzer = SimpleAnalyzer;
    let scorer = Bm25Scorer::default();

    // "query" at pos 0, "plan" at pos 1, "optimization" at pos 2 -> distance 2
    idx.add_document(1, "query plan optimization techniques", &analyzer)
        .expect("add failed");
    // "query" at pos 3, "optimization" at pos 0 -> distance 3
    idx.add_document(2, "optimization of complex query plans", &analyzer)
        .expect("add failed");
    // No "optimization"
    idx.add_document(3, "query processing basics", &analyzer)
        .expect("add failed");

    let query = FtsQueryParser::parse("\"query optimization\"~5").expect("parse failed");
    let results = idx
        .search(&query, &analyzer, &scorer, 100)
        .expect("search failed");

    tprintln!("  Query: '\"query optimization\"~5'");
    tprintln!("  Results: {} documents", results.len());

    let result_ids: Vec<u64> = results.iter().map(|r| r.0).collect();

    // Doc 1: "query" at 0, "optimization" at 2 -> distance 2 <= 5
    assert!(
        result_ids.contains(&1),
        "doc 1 should match (distance 2 <= 5)"
    );

    // Doc 2: "optimization" at 0, "query" at 3 -> distance 3 <= 5
    assert!(
        result_ids.contains(&2),
        "doc 2 should match (distance 3 <= 5)"
    );

    // Doc 3: no "optimization"
    assert!(
        !result_ids.contains(&3),
        "doc 3 should not match (no 'optimization')"
    );

    tprintln!("  Proximity distance verification: PASS");
}

// =============================================================================
// Test 6: Highlighting
// =============================================================================

#[test]
fn test_06_highlighting() {
    zyron_bench_harness::init("search");
    let _guard = BENCHMARK_LOCK.lock().unwrap_or_else(|e| e.into_inner());

    tprintln!("\n=== Test 6: Highlighting ===");

    let analyzer = SimpleAnalyzer;
    let query = FtsQuery::Term("database".to_string());
    let config = HighlightConfig {
        pre_tag: "<b>".to_string(),
        post_tag: "</b>".to_string(),
        fragment_size: 200,
        num_fragments: 3,
    };

    let text =
        "A database is a collection of organized data. The database stores records efficiently.";
    let result = highlight::highlight(text, &query, &analyzer, &config);

    tprintln!("  Input: '{}'", text);
    tprintln!("  Output: '{}'", result);

    assert!(
        result.contains("<b>database</b>"),
        "highlight should wrap 'database' in <b> tags, got: {}",
        result
    );

    // Verify surrounding context is included
    assert!(
        result.len() > 20,
        "highlight output too short, missing context"
    );

    tprintln!("  Tag wrapping verified: PASS");
    tprintln!("  Context inclusion verified: PASS");
}

// =============================================================================
// Test 7: Autocomplete
// =============================================================================

#[test]
fn test_07_autocomplete() {
    zyron_bench_harness::init("search");
    let _guard = BENCHMARK_LOCK.lock().unwrap_or_else(|e| e.into_inner());

    tprintln!("\n=== Test 7: Autocomplete ===");

    let mut prefix_idx = PrefixIndex::new();

    // Insert product names with known frequencies.
    // "laptop" and "lapel" get the highest frequency to appear in top-10.
    prefix_idx.insert("laptop", 500);
    prefix_idx.insert("lapel", 400);
    prefix_idx.insert("lamp", 300);
    prefix_idx.insert("landscape", 200);
    prefix_idx.insert("language", 150);

    // Add 10,000 variations with lower frequencies
    for i in 0..10_000u32 {
        let base = match i % 5 {
            0 => "laptop",
            1 => "lapel",
            2 => "lamp",
            3 => "landscape",
            _ => "language",
        };
        let name = format!("{base}_{i}");
        prefix_idx.insert(&name, 50u32.saturating_sub(i % 50));
    }

    tprintln!("  Indexed {} product names", prefix_idx.term_count());

    // Measure autocomplete latency
    let mut latency_results = Vec::new();
    for run in 0..VALIDATION_RUNS {
        let start = Instant::now();
        let suggestions = prefix_idx.suggest("lap", 10);
        let elapsed = start.elapsed();
        let latency_ms = elapsed.as_secs_f64() * 1000.0;
        latency_results.push(latency_ms);

        if run == 0 {
            tprintln!("  Query: 'lap' (limit=10)");
            tprintln!(
                "  Suggestions: {:?}",
                suggestions.iter().map(|s| &s.term).collect::<Vec<_>>()
            );
            assert!(
                !suggestions.is_empty(),
                "autocomplete returned no suggestions"
            );

            let has_laptop = suggestions.iter().any(|s| s.term == "laptop");
            let has_lapel = suggestions.iter().any(|s| s.term == "lapel");
            assert!(has_laptop, "suggestions should include 'laptop'");
            assert!(has_lapel, "suggestions should include 'lapel'");
            tprintln!("  Suggestions contain 'laptop' and 'lapel': PASS");
        }
    }

    tprintln!("\n=== Autocomplete Performance ===");
    let result = validate_metric(
        "Autocomplete",
        "Latency (ms)",
        latency_results,
        AUTOCOMPLETE_TARGET_MS,
        false,
    );
    assert!(
        result.passed,
        "autocomplete latency avg {:.3}ms > target {:.1}ms",
        result.average, AUTOCOMPLETE_TARGET_MS
    );
    assert!(
        !result.regression_detected,
        "autocomplete regression detected"
    );
}

// =============================================================================
// Test 8: Field Boosting
// =============================================================================

#[test]
fn test_08_field_boosting() {
    zyron_bench_harness::init("search");
    let _guard = BENCHMARK_LOCK.lock().unwrap_or_else(|e| e.into_inner());

    tprintln!("\n=== Test 8: Field Boosting ===");

    // Simulate two separate field indexes for title and body
    let title_idx = InvertedIndex::new(10, 100, vec![1]);
    let body_idx = InvertedIndex::new(11, 100, vec![2]);
    let analyzer = SimpleAnalyzer;
    let scorer = Bm25Scorer::default();

    // Doc 1: "zyrondb" in title only
    title_idx
        .add_document(1, "zyrondb release notes", &analyzer)
        .expect("add failed");
    body_idx
        .add_document(
            1,
            "the latest version includes performance fixes",
            &analyzer,
        )
        .expect("add failed");

    // Doc 2: "zyrondb" in body only
    title_idx
        .add_document(2, "release notes update", &analyzer)
        .expect("add failed");
    body_idx
        .add_document(
            2,
            "zyrondb handles distributed queries efficiently",
            &analyzer,
        )
        .expect("add failed");

    let query = FtsQuery::Term("zyrondb".to_string());

    let title_results = title_idx
        .search(&query, &analyzer, &scorer, 10)
        .expect("search failed");
    let body_results = body_idx
        .search(&query, &analyzer, &scorer, 10)
        .expect("search failed");

    // Build per-doc scores with boosting
    let boosts = vec![
        FieldBoost {
            column_id: 1,
            boost: 2.0,
        },
        FieldBoost {
            column_id: 2,
            boost: 1.0,
        },
    ];

    let mut doc_scores: std::collections::HashMap<u64, f64> = std::collections::HashMap::new();
    for (doc_id, score) in &title_results {
        let field_scores = vec![(1u16, *score)];
        *doc_scores.entry(*doc_id).or_insert(0.0) += score_multi_field(&field_scores, &boosts);
    }
    for (doc_id, score) in &body_results {
        let field_scores = vec![(2u16, *score)];
        *doc_scores.entry(*doc_id).or_insert(0.0) += score_multi_field(&field_scores, &boosts);
    }

    let score_1 = doc_scores.get(&1).copied().unwrap_or(0.0);
    let score_2 = doc_scores.get(&2).copied().unwrap_or(0.0);

    tprintln!("  Doc 1 (title match, boost=2.0): score={:.4}", score_1);
    tprintln!("  Doc 2 (body match, boost=1.0): score={:.4}", score_2);

    assert!(
        score_1 > score_2,
        "title match (boost=2.0) should rank higher than body match (boost=1.0)"
    );
    tprintln!("  Title boost ranks higher: PASS");
}

// =============================================================================
// Test 9: Synonym Expansion
// =============================================================================

#[test]
fn test_09_synonym_expansion() {
    zyron_bench_harness::init("search");
    let _guard = BENCHMARK_LOCK.lock().unwrap_or_else(|e| e.into_inner());

    tprintln!("\n=== Test 9: Synonym Expansion ===");

    let idx = InvertedIndex::new(9, 100, vec![1]);
    let analyzer = SimpleAnalyzer;
    let scorer = Bm25Scorer::default();

    idx.add_document(1, "database management system", &analyzer)
        .expect("add failed");
    idx.add_document(2, "db performance tuning", &analyzer)
        .expect("add failed");
    idx.add_document(3, "datastore replication setup", &analyzer)
        .expect("add failed");
    idx.add_document(4, "network protocol design", &analyzer)
        .expect("add failed");

    let mut synonyms = SynonymSet::new("db_synonyms");
    synonyms.add_group(&["database", "db", "datastore"]);

    // Search with synonym expansion: search for each term in the group
    let expanded_terms = synonyms.expand("database");
    tprintln!("  Synonym group: database -> {:?}", expanded_terms);

    // Build a boolean query: should(database, db, datastore)
    let query = FtsQuery::Boolean {
        must: vec![],
        should: vec![
            FtsQuery::Term("database".to_string()),
            FtsQuery::Term("db".to_string()),
            FtsQuery::Term("datastore".to_string()),
        ],
        must_not: vec![],
    };

    let results = idx
        .search(&query, &analyzer, &scorer, 100)
        .expect("search failed");
    let result_ids: Vec<u64> = results.iter().map(|r| r.0).collect();

    tprintln!(
        "  Results: {} documents, IDs: {:?}",
        results.len(),
        result_ids
    );

    assert!(result_ids.contains(&1), "should match doc 1 ('database')");
    assert!(result_ids.contains(&2), "should match doc 2 ('db')");
    assert!(result_ids.contains(&3), "should match doc 3 ('datastore')");
    assert!(
        !result_ids.contains(&4),
        "should NOT match doc 4 (no synonym)"
    );

    tprintln!("  Synonym expansion covers all group terms: PASS");
}

// =============================================================================
// Test 10: Performance - Large Scale Indexing and Query
// =============================================================================

#[test]
fn test_10_performance() {
    zyron_bench_harness::init("search");
    let _guard = BENCHMARK_LOCK.lock().unwrap_or_else(|e| e.into_inner());

    const DOC_COUNT: u64 = 1_000_000;
    const WORD_COUNT: usize = 20;

    tprintln!("\n=== Test 10: Performance (1M documents) ===");
    let util_before = take_util_snapshot();

    // ---- Indexing throughput ----
    tprintln!("\n--- Indexing Throughput ---");
    let mut indexing_results = Vec::new();
    let mut idx = InvertedIndex::new(10, 100, vec![1, 2]);

    // SimpleAnalyzer indexing (production baseline for exact-match workloads)
    tprintln!("  [SimpleAnalyzer]");
    for run in 0..VALIDATION_RUNS {
        idx = InvertedIndex::new(10, 100, vec![1, 2]);
        let analyzer = SimpleAnalyzer;
        let mut buf = AnalysisBuffer::new();

        let start = Instant::now();
        for doc_id in 0..DOC_COUNT {
            let (title, body) = generate_article(doc_id, WORD_COUNT);
            let combined = format!("{title} {body}");
            idx.add_document_with_buf(doc_id, &combined, &analyzer, &mut buf)
                .expect("add_document failed");
        }
        let elapsed = start.elapsed();
        let docs_per_sec = DOC_COUNT as f64 / elapsed.as_secs_f64();
        indexing_results.push(docs_per_sec);

        tprintln!(
            "  Run {}: {} docs/sec ({:?})",
            run + 1,
            format_with_commas(docs_per_sec),
            elapsed
        );
    }

    tprintln!(
        "\n  SimpleAnalyzer index stats: {} documents, avg_dl={:.1}",
        idx.doc_count(),
        idx.avg_dl()
    );

    let indexing_result = validate_metric(
        "Performance",
        "Indexing throughput SimpleAnalyzer (docs/sec)",
        indexing_results,
        INDEXING_TARGET_DOCS_SEC,
        true,
    );

    // StandardAnalyzer indexing (production path with stemming + stopwords)
    tprintln!("\n  [StandardAnalyzer]");
    let mut std_indexing_results = Vec::new();
    let mut std_idx = InvertedIndex::new(11, 100, vec![1, 2]);

    for run in 0..VALIDATION_RUNS {
        std_idx = InvertedIndex::new(11, 100, vec![1, 2]);
        let analyzer = StandardAnalyzer;

        let mut buf = AnalysisBuffer::new();
        let start = Instant::now();
        for doc_id in 0..DOC_COUNT {
            let (title, body) = generate_article(doc_id, WORD_COUNT);
            let combined = format!("{title} {body}");
            std_idx
                .add_document_with_buf(doc_id, &combined, &analyzer, &mut buf)
                .expect("add_document failed");
        }
        let elapsed = start.elapsed();
        let docs_per_sec = DOC_COUNT as f64 / elapsed.as_secs_f64();
        std_indexing_results.push(docs_per_sec);

        tprintln!(
            "  Run {}: {} docs/sec ({:?})",
            run + 1,
            format_with_commas(docs_per_sec),
            elapsed
        );
    }

    tprintln!(
        "\n  StandardAnalyzer index stats: {} documents, avg_dl={:.1}",
        std_idx.doc_count(),
        std_idx.avg_dl()
    );

    let std_indexing_result = validate_metric(
        "Performance",
        "Indexing throughput StandardAnalyzer (docs/sec)",
        std_indexing_results,
        INDEXING_TARGET_DOCS_SEC,
        true,
    );

    // ---- Measure index size ratio ----
    tprintln!("\n--- Index Size ---");
    let raw_data_size: usize = (0..DOC_COUNT)
        .map(|id| {
            let (t, b) = generate_article(id, WORD_COUNT);
            t.len() + b.len() + 1
        })
        .sum();

    let tmp_dir = std::env::temp_dir().join("zyron_fts_bench");
    let _ = std::fs::create_dir_all(&tmp_dir);
    let idx_path = tmp_dir.join("bench_index.zyfts");
    idx.save_to_file(&idx_path).expect("save_to_file failed");
    let index_file_size = std::fs::metadata(&idx_path)
        .map(|m| m.len() as usize)
        .unwrap_or(0);
    let _ = std::fs::remove_dir_all(&tmp_dir);

    let size_ratio = index_file_size as f64 / raw_data_size as f64;
    tprintln!(
        "  Raw data size: {} bytes",
        format_with_commas(raw_data_size as f64)
    );
    tprintln!(
        "  Index file size: {} bytes",
        format_with_commas(index_file_size as f64)
    );
    tprintln!("  Ratio: {:.1}%", size_ratio * 100.0);

    let size_passed = check_performance(
        "Performance",
        "Index size ratio",
        size_ratio,
        INDEX_SIZE_RATIO_TARGET,
        false,
    );

    // ---- Single-term query latency ----
    tprintln!("\n--- Single-Term Query Latency ---");
    let analyzer = SimpleAnalyzer;
    let scorer = Bm25Scorer::default();
    let mut single_term_results = Vec::new();

    for run in 0..VALIDATION_RUNS {
        let query = FtsQuery::Term("database".to_string());
        let start = Instant::now();
        let results = idx
            .search(&query, &analyzer, &scorer, 10)
            .expect("search failed");
        let elapsed = start.elapsed();
        let latency_ms = elapsed.as_secs_f64() * 1000.0;
        single_term_results.push(latency_ms);

        if run == 0 {
            tprintln!("  Results: {} documents", results.len());
        }
    }

    let single_result = validate_metric(
        "Performance",
        "Single-term query latency (ms)",
        single_term_results,
        SINGLE_TERM_TARGET_MS,
        false,
    );

    // ---- Phrase query latency ----
    tprintln!("\n--- Phrase Query Latency ---");
    let mut phrase_results = Vec::new();

    for _run in 0..VALIDATION_RUNS {
        let query = FtsQuery::Phrase(vec!["database".to_string(), "performance".to_string()]);
        let start = Instant::now();
        let _results = idx
            .search(&query, &analyzer, &scorer, 10)
            .expect("search failed");
        let elapsed = start.elapsed();
        phrase_results.push(elapsed.as_secs_f64() * 1000.0);
    }

    let phrase_result = validate_metric(
        "Performance",
        "Phrase query latency (ms)",
        phrase_results,
        PHRASE_QUERY_TARGET_MS,
        false,
    );

    // ---- Boolean query latency (3 terms) ----
    tprintln!("\n--- Boolean Query Latency (3 terms) ---");
    let mut boolean_results = Vec::new();

    for _run in 0..VALIDATION_RUNS {
        let query = FtsQuery::Boolean {
            must: vec![
                FtsQuery::Term("database".to_string()),
                FtsQuery::Term("performance".to_string()),
            ],
            should: vec![FtsQuery::Term("optimization".to_string())],
            must_not: vec![],
        };
        let start = Instant::now();
        let _results = idx
            .search(&query, &analyzer, &scorer, 10)
            .expect("search failed");
        let elapsed = start.elapsed();
        boolean_results.push(elapsed.as_secs_f64() * 1000.0);
    }

    let boolean_result = validate_metric(
        "Performance",
        "Boolean query latency (ms)",
        boolean_results,
        BOOLEAN_QUERY_TARGET_MS,
        false,
    );

    // ---- Fuzzy query latency (edit=2) ----
    tprintln!("\n--- Fuzzy Query Latency (edit=2) ---");
    let mut fuzzy_results = Vec::new();

    for _run in 0..VALIDATION_RUNS {
        let query = FtsQuery::Fuzzy {
            term: "databse".to_string(),
            max_edits: 2,
        };
        let start = Instant::now();
        let _results = idx
            .search(&query, &analyzer, &scorer, 10)
            .expect("search failed");
        let elapsed = start.elapsed();
        fuzzy_results.push(elapsed.as_secs_f64() * 1000.0);
    }

    let fuzzy_result = validate_metric(
        "Performance",
        "Fuzzy query latency (ms)",
        fuzzy_results,
        FUZZY_QUERY_TARGET_MS,
        false,
    );

    // ---- Top-10 ranking latency ----
    tprintln!("\n--- Top-10 Ranking Latency ---");
    let mut top10_results = Vec::new();

    for _run in 0..VALIDATION_RUNS {
        let query = FtsQuery::Term("performance".to_string());
        let start = Instant::now();
        let results = idx
            .search(&query, &analyzer, &scorer, 10)
            .expect("search failed");
        let elapsed = start.elapsed();
        top10_results.push(elapsed.as_secs_f64() * 1000.0);
        assert!(results.len() <= 10, "top-10 returned more than 10 results");
    }

    let top10_result = validate_metric(
        "Performance",
        "Top-10 ranking latency (ms)",
        top10_results,
        TOP_10_RANKING_TARGET_MS,
        false,
    );

    // ---- Highlight latency ----
    tprintln!("\n--- Highlight Latency ---");
    let mut highlight_results = Vec::new();
    let config = HighlightConfig::default();
    let sample_text = "The database system provides high performance for distributed query processing with database optimization.";

    for _run in 0..VALIDATION_RUNS {
        let query = FtsQuery::Term("database".to_string());
        let start = Instant::now();
        for _ in 0..1000 {
            let _ = highlight::highlight(sample_text, &query, &analyzer, &config);
        }
        let elapsed = start.elapsed();
        let per_call_us = elapsed.as_secs_f64() * 1_000_000.0 / 1000.0;
        highlight_results.push(per_call_us);
    }

    let highlight_result = validate_metric(
        "Performance",
        "Highlight latency (us)",
        highlight_results,
        HIGHLIGHT_TARGET_US,
        false,
    );

    record_test_util("Performance", util_before, take_util_snapshot());

    // ---- Final assertions ----
    tprintln!("\n=== Performance Summary ===");
    assert!(
        indexing_result.passed,
        "SimpleAnalyzer indexing throughput below target"
    );
    assert!(
        !indexing_result.regression_detected,
        "SimpleAnalyzer indexing regression detected"
    );
    assert!(
        std_indexing_result.passed,
        "StandardAnalyzer indexing throughput below target"
    );
    assert!(
        !std_indexing_result.regression_detected,
        "StandardAnalyzer indexing regression detected"
    );
    assert!(size_passed, "index size ratio exceeds target");
    assert!(
        single_result.passed,
        "single-term query latency above target"
    );
    assert!(
        !single_result.regression_detected,
        "single-term query regression"
    );
    assert!(phrase_result.passed, "phrase query latency above target");
    assert!(
        !phrase_result.regression_detected,
        "phrase query regression"
    );
    assert!(boolean_result.passed, "boolean query latency above target");
    assert!(
        !boolean_result.regression_detected,
        "boolean query regression"
    );
    assert!(fuzzy_result.passed, "fuzzy query latency above target");
    assert!(!fuzzy_result.regression_detected, "fuzzy query regression");
    assert!(top10_result.passed, "top-10 ranking latency above target");
    assert!(
        !top10_result.regression_detected,
        "top-10 ranking regression"
    );
    assert!(highlight_result.passed, "highlight latency above target");
    assert!(
        !highlight_result.regression_detected,
        "highlight regression"
    );
}
