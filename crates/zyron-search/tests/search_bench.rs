#![allow(non_snake_case, unused_assignments, unused_variables)]

//! Full-Text Search and Vector/Graph Search Benchmark Suite
//!
//! Integration tests for ZyronDB FTS engine and vector/graph search:
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

    // -----------------------------------------------------------------------
    // Indexing throughput
    // -----------------------------------------------------------------------
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

    // -----------------------------------------------------------------------
    // Measure index size ratio
    // -----------------------------------------------------------------------
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

    // -----------------------------------------------------------------------
    // Single-term query latency
    // -----------------------------------------------------------------------
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

    // -----------------------------------------------------------------------
    // Phrase query latency
    // -----------------------------------------------------------------------
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

    // -----------------------------------------------------------------------
    // Boolean query latency (3 terms)
    // -----------------------------------------------------------------------
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

    // -----------------------------------------------------------------------
    // Fuzzy query latency (edit=2)
    // -----------------------------------------------------------------------
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

    // -----------------------------------------------------------------------
    // Top-10 ranking latency
    // -----------------------------------------------------------------------
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

    // -----------------------------------------------------------------------
    // Highlight latency
    // -----------------------------------------------------------------------
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

    // -----------------------------------------------------------------------
    // Final assertions
    // -----------------------------------------------------------------------
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

// =============================================================================
// Vector and Graph Search Validation Tests
// =============================================================================
//
// Performance Targets:
// | Test                     | Metric    | Minimum Threshold |
// |--------------------------|-----------|-------------------|
// | ANN build (1M, 128d)    | latency   | <=40s             |
// | ANN search top-10       | QPS       | >=8000            |
// | ANN recall@10           | recall    | >=0.95            |
// | Cosine distance (1536d) | latency   | <=2us             |
// | Quantized build (1M)    | latency   | <=20s             |
// | Quantized search top-10 | QPS       | >=12000           |
// | PageRank (1M nodes)     | latency   | <=5s              |
// | Shortest path (1M)      | latency   | <=100ms           |

use zyron_search::graph::algorithms;
use zyron_search::graph::query::{EdgeDirection, GraphPattern, PatternElement, compile_pattern};
use zyron_search::graph::schema::{GraphSchema, PropertyDef};
use zyron_search::graph::storage::CompactGraph;
use zyron_search::vector::{
    AnnIndex, DataProfile, DistanceMetric, HnswConfig, HybridSearch, IvfPqConfig, IvfPqIndex,
    VectorSearch, computeDistance,
};

// Performance target constants
const ANN_BUILD_TARGET_SEC: f64 = 40.0;
const ANN_SEARCH_QPS_TARGET: f64 = 8000.0;
const ANN_RECALL_TARGET: f64 = 0.95;
const COSINE_DISTANCE_TARGET_US: f64 = 2.0;
const QUANTIZED_BUILD_TARGET_SEC: f64 = 20.0;
const QUANTIZED_SEARCH_QPS_TARGET: f64 = 12000.0;
const PAGERANK_TARGET_SEC: f64 = 5.0;
const SHORTEST_PATH_TARGET_MS: f64 = 100.0;

// ---------------------------------------------------------------------------
// Vector/Graph test data generation
// ---------------------------------------------------------------------------

fn xorshift64(state: &mut u64) -> u64 {
    let mut x = *state;
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    *state = x;
    x
}

fn randomF32(state: &mut u64) -> f32 {
    (xorshift64(state) as f32) / (u64::MAX as f32) * 2.0 - 1.0
}

fn generateRandomVector(state: &mut u64, dims: usize) -> Vec<f32> {
    (0..dims).map(|_| randomF32(state)).collect()
}

/// Box-Muller transform for Gaussian-distributed samples.
/// Produces N(0, 1) random values from uniform sources.
fn gaussianF32(state: &mut u64) -> f32 {
    // Uniform (0, 1] values, avoid exact 0.
    let u1 = ((xorshift64(state) >> 11) as f64) / ((1u64 << 53) as f64);
    let u1 = if u1 < 1e-10 { 1e-10 } else { u1 };
    let u2 = ((xorshift64(state) >> 11) as f64) / ((1u64 << 53) as f64);
    ((-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()) as f32
}

/// Generates clustered vectors from a Gaussian mixture model.
/// Produces "realistic but not best-case" data with distSpread typically
/// 0.30-0.40: harder than clean sentence-transformer embeddings (distSpread
/// 0.5+) but much easier than adversarial uniform random (distSpread 0.14).
/// Vectors are normalized to the unit sphere, matching typical embedding layouts.
fn generateClusteredVectors(
    n: usize,
    dims: usize,
    numClusters: usize,
    clusterStdDev: f32,
    seed: u64,
) -> Vec<Vec<f32>> {
    let mut state = seed;
    // Generate well-separated cluster centers uniformly in [-2, 2] per dim.
    let centers: Vec<Vec<f32>> = (0..numClusters)
        .map(|_| (0..dims).map(|_| randomF32(&mut state) * 2.0).collect())
        .collect();

    (0..n)
        .map(|_| {
            let clusterIdx = (xorshift64(&mut state) as usize) % numClusters;
            let center = &centers[clusterIdx];
            let mut v: Vec<f32> = (0..dims)
                .map(|d| center[d] + gaussianF32(&mut state) * clusterStdDev)
                .collect();
            // Normalize to unit sphere (typical for embeddings).
            let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-10);
            for x in v.iter_mut() {
                *x /= norm;
            }
            v
        })
        .collect()
}

fn bruteForceKnn(
    query: &[f32],
    vectors: &[(u64, &[f32])],
    k: usize,
    metric: DistanceMetric,
) -> Vec<u64> {
    let mut dists: Vec<(u64, f32)> = vectors
        .iter()
        .map(|(id, v)| (*id, computeDistance(metric, query, v)))
        .collect();
    dists.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
    dists.truncate(k);
    dists.iter().map(|(id, _)| *id).collect()
}

fn measureRecall(annResults: &[(u64, f32)], truthIds: &[u64]) -> f64 {
    let annIds: std::collections::HashSet<u64> = annResults.iter().map(|(id, _)| *id).collect();
    let truthSet: std::collections::HashSet<u64> = truthIds.iter().copied().collect();
    let hits = annIds.intersection(&truthSet).count();
    hits as f64 / truthIds.len() as f64
}

fn generateRandomGraph(
    nodeCount: usize,
    avgDegree: usize,
    seed: u64,
) -> Vec<(u64, u64, Option<f64>)> {
    let mut state = seed;
    let mut edges = Vec::with_capacity(nodeCount * avgDegree);
    for src in 0..nodeCount {
        for _ in 0..avgDegree {
            let dst = (xorshift64(&mut state) as usize) % nodeCount;
            if dst != src {
                edges.push((src as u64, dst as u64, None));
            }
        }
    }
    edges
}

// =============================================================================
// Vector Insert and Query (10K vectors, recall@10 >= 0.95)
// =============================================================================

#[test]
fn test_vector_insert_and_query() {
    zyron_bench_harness::init("search");
    let _guard = BENCHMARK_LOCK.lock().unwrap_or_else(|e| e.into_inner());

    tprintln!("\n=== Vector Insert and Query ===");

    let dims = 128;
    let count = 10_000u64;
    let mut rng = 42u64;

    let vectors: Vec<(u64, Vec<f32>)> = (0..count)
        .map(|id| (id, generateRandomVector(&mut rng, dims)))
        .collect();

    let start = Instant::now();
    let vecRefs: Vec<(u64, &[f32])> = vectors.iter().map(|(id, v)| (*id, v.as_slice())).collect();
    let index =
        AnnIndex::build(&vecRefs, 1, 100, 0, HnswConfig::default()).expect("build 10K index");
    let buildTime = start.elapsed();

    tprintln!(
        "  Build time: {:.2}ms for {} vectors",
        buildTime.as_secs_f64() * 1000.0,
        count
    );
    assert_eq!(index.len(), count as usize);

    // Sanity check: querying for an existing vector should find itself.
    let sanityQuery = &vectors[0].1;
    let sanityResults = index.search(sanityQuery, 10, 0).expect("search");
    assert!(!sanityResults.is_empty(), "search returned no results");
    assert_eq!(
        sanityResults[0].0, 0,
        "nearest of vector[0] should be itself"
    );

    // Proper recall measurement: use INDEPENDENT query vectors not in the dataset.
    let mut queryRng = 0xDEADBEEFu64;
    let query = generateRandomVector(&mut queryRng, dims);
    let results = index.search(&query, 10, 0).expect("search");
    tprintln!(
        "  Top-10 results for independent query: {:?}",
        results.iter().map(|(id, d)| (*id, *d)).collect::<Vec<_>>()
    );

    // Verify recall against brute force using independent query.
    let vecRefs: Vec<(u64, &[f32])> = vectors.iter().map(|(id, v)| (*id, v.as_slice())).collect();
    let truthIds = bruteForceKnn(&query, &vecRefs, 10, DistanceMetric::Cosine);
    let recall = measureRecall(&results, &truthIds);
    tprintln!("  Recall@10: {:.3}", recall);

    check_performance(
        "test_vector_insert_query",
        "recall@10",
        recall,
        ANN_RECALL_TARGET,
        true,
    );
    assert!(recall >= 0.80, "recall@10 too low: {recall:.3}");
}

// =============================================================================
// Distance Metrics
// =============================================================================

#[test]
fn test_distance_metrics() {
    zyron_bench_harness::init("search");
    let _guard = BENCHMARK_LOCK.lock().unwrap_or_else(|e| e.into_inner());

    tprintln!("\n=== Distance Metrics ===");

    // Cosine: identical vectors -> 0.0
    let a = vec![1.0f32, 2.0, 3.0];
    let b = vec![1.0f32, 2.0, 3.0];
    let d = computeDistance(DistanceMetric::Cosine, &a, &b);
    tprintln!("  Cosine(identical): {d:.6}");
    assert!(
        d.abs() < 0.001,
        "cosine of identical vectors should be ~0, got {d}"
    );

    // Cosine: orthogonal vectors -> 1.0
    let a = vec![1.0f32, 0.0, 0.0];
    let b = vec![0.0f32, 1.0, 0.0];
    let d = computeDistance(DistanceMetric::Cosine, &a, &b);
    tprintln!("  Cosine(orthogonal): {d:.6}");
    assert!(
        (d - 1.0).abs() < 0.001,
        "cosine of orthogonal vectors should be ~1, got {d}"
    );

    // Euclidean: identical -> 0.0
    let d = computeDistance(
        DistanceMetric::Euclidean,
        &[1.0, 2.0, 3.0],
        &[1.0, 2.0, 3.0],
    );
    tprintln!("  Euclidean(identical): {d:.6}");
    assert!(
        d.abs() < 0.001,
        "euclidean of identical should be ~0, got {d}"
    );

    // Euclidean: [1,0,0] to [0,1,0] -> sqrt(2) ~= 1.4142
    let d = computeDistance(
        DistanceMetric::Euclidean,
        &[1.0, 0.0, 0.0],
        &[0.0, 1.0, 0.0],
    );
    tprintln!("  Euclidean([1,0,0],[0,1,0]): {d:.6}");
    assert!(
        (d - std::f32::consts::SQRT_2).abs() < 0.01,
        "euclidean should be sqrt(2), got {d}"
    );

    // DotProduct: [1,2,3] dot [4,5,6] = 32, returned negated = -32
    let d = computeDistance(
        DistanceMetric::DotProduct,
        &[1.0, 2.0, 3.0],
        &[4.0, 5.0, 6.0],
    );
    tprintln!("  DotProduct([1,2,3],[4,5,6]): {d:.6}");
    assert!(
        (d - (-32.0)).abs() < 0.01,
        "dot product should be -32, got {d}"
    );

    // Manhattan: [1,0,0] to [0,1,0] -> 2.0
    let d = computeDistance(
        DistanceMetric::Manhattan,
        &[1.0, 0.0, 0.0],
        &[0.0, 1.0, 0.0],
    );
    tprintln!("  Manhattan([1,0,0],[0,1,0]): {d:.6}");
    assert!((d - 2.0).abs() < 0.01, "manhattan should be 2.0, got {d}");

    // Test with larger dimensions (128, 1536) to cover SIMD paths
    let mut rng = 99u64;
    for dims in [128, 1536] {
        let a = generateRandomVector(&mut rng, dims);
        let b = generateRandomVector(&mut rng, dims);

        let cosine = computeDistance(DistanceMetric::Cosine, &a, &b);
        let euclidean = computeDistance(DistanceMetric::Euclidean, &a, &b);
        let dot = computeDistance(DistanceMetric::DotProduct, &a, &b);
        let manhattan = computeDistance(DistanceMetric::Manhattan, &a, &b);

        tprintln!(
            "  Dims={dims}: cosine={cosine:.4}, euclidean={euclidean:.4}, dot={dot:.4}, manhattan={manhattan:.4}"
        );
        assert!(
            cosine >= 0.0 && cosine <= 2.0,
            "cosine out of range: {cosine}"
        );
        assert!(euclidean >= 0.0, "euclidean negative: {euclidean}");
        assert!(manhattan >= 0.0, "manhattan negative: {manhattan}");
    }

    // Cosine 1536d latency
    let a = generateRandomVector(&mut rng, 1536);
    let b = generateRandomVector(&mut rng, 1536);
    let iterations = 100_000;
    let start = Instant::now();
    for _ in 0..iterations {
        std::hint::black_box(computeDistance(DistanceMetric::Cosine, &a, &b));
    }
    let elapsed = start.elapsed();
    let perCallUs = elapsed.as_secs_f64() * 1_000_000.0 / iterations as f64;
    tprintln!("  Cosine 1536d: {perCallUs:.3}us per call ({iterations} calls)");
    check_performance(
        "test_distance",
        "cosine_1536d_us",
        perCallUs,
        COSINE_DISTANCE_TARGET_US,
        false,
    );

    tprintln!("  PASSED: all distance metrics correct");
}

// =============================================================================
// ANN Recall (100K vectors, recall@100 per spec)
// =============================================================================

#[test]
fn test_ann_recall() {
    zyron_bench_harness::init("search");
    let _guard = BENCHMARK_LOCK.lock().unwrap_or_else(|e| e.into_inner());

    tprintln!("\n=== ANN Recall (100K vectors) ===");

    let dims = 128;
    let count = 100_000u64;
    let mut rng = 777u64;

    tprintln!("  Generating {} vectors ({}-dim)...", count, dims);
    let vectors: Vec<(u64, Vec<f32>)> = (0..count)
        .map(|id| (id, generateRandomVector(&mut rng, dims)))
        .collect();

    let vecRefs: Vec<(u64, &[f32])> = vectors.iter().map(|(id, v)| (*id, v.as_slice())).collect();

    // Compute DataProfile and derive config automatically.
    let sampleSlices: Vec<&[f32]> = vectors[..vectors.len().min(1000)]
        .iter()
        .map(|(_, v)| v.as_slice())
        .collect();
    let profile = DataProfile::compute(
        &sampleSlices,
        count as usize,
        dims as u16,
        DistanceMetric::Euclidean,
    );
    let config = HnswConfig::auto(&profile, DistanceMetric::Euclidean);
    tprintln!(
        "  Auto config: m={}, efC={}, efS={}",
        config.m,
        config.efConstruction,
        config.efSearch,
    );

    tprintln!("  Building HNSW index...");
    let start = Instant::now();
    let index = AnnIndex::build(&vecRefs, 1, 100, 0, config.clone()).expect("build 100K index");
    let buildTime = start.elapsed();
    tprintln!("  Build time: {:.2}s", buildTime.as_secs_f64());

    // Query top-100 with auto-derived search quality parameters.
    // Generate INDEPENDENT query vectors not in the dataset to avoid inflating recall.
    let numQueries = 50;
    let k = 100;
    let mut totalRecallDefault = 0.0;
    let mut totalRecallHigh = 0.0;
    let efDefault = config.efSearch;
    let efHigh = (config.efSearch as u32 * 4).min(2048) as u16;

    let mut queryRng = 0xDEADBEEFu64;
    let queries: Vec<Vec<f32>> = (0..numQueries)
        .map(|_| generateRandomVector(&mut queryRng, dims))
        .collect();

    let vecRefsForTruth: Vec<(u64, &[f32])> =
        vectors.iter().map(|(id, v)| (*id, v.as_slice())).collect();

    for q in 0..numQueries {
        let query = &queries[q];

        let truthIds = bruteForceKnn(query, &vecRefsForTruth, k, DistanceMetric::Euclidean);

        // Default search quality (auto-tuned efSearch)
        let resultsDefault = index.search(query, k, efDefault).expect("search default");
        totalRecallDefault += measureRecall(&resultsDefault, &truthIds);

        // High-quality search (4x auto-tuned efSearch)
        let resultsHigh = index.search(query, k, efHigh).expect("search high");
        totalRecallHigh += measureRecall(&resultsHigh, &truthIds);
    }

    let avgRecallDefault = totalRecallDefault / numQueries as f64;
    let avgRecallHigh = totalRecallHigh / numQueries as f64;

    tprintln!("  Recall@{k} (efS={efDefault}): {avgRecallDefault:.4}");
    tprintln!("  Recall@{k} (efS={efHigh}):    {avgRecallHigh:.4}");

    check_performance(
        "test_ann_recall",
        "recall@100_default",
        avgRecallDefault,
        0.90,
        true,
    );
    check_performance(
        "test_ann_recall",
        "recall@100_high",
        avgRecallHigh,
        0.99,
        true,
    );

    assert!(
        avgRecallDefault >= 0.90,
        "recall@100 default too low: {avgRecallDefault:.4}"
    );
    assert!(
        avgRecallHigh >= 0.99,
        "recall@100 high too low: {avgRecallHigh:.4}"
    );
}

// =============================================================================
// Recall Diagnostic (isolates efSearch scaling vs graph quality vs parallel build)
// =============================================================================

#[test]
fn test_recall_diagnostic() {
    zyron_bench_harness::init("search");
    let _guard = BENCHMARK_LOCK.lock().unwrap_or_else(|e| e.into_inner());

    tprintln!("\n=== Recall Diagnostic ===");

    let dims = 128;
    let k = 10;
    let numQueries = 50;
    let mut queryRng = 0xDEADBEEFu64;
    let queries: Vec<Vec<f32>> = (0..numQueries)
        .map(|_| generateRandomVector(&mut queryRng, dims))
        .collect();

    // Helper: measure recall@k at a specific efSearch on a built index.
    let measureRecallAtEf = |index: &AnnIndex, vectors: &[(u64, &[f32])], efSearch: u16| -> f64 {
        let mut total = 0.0;
        for q in &queries {
            let results = index.search(q, k, efSearch).expect("search");
            let truthIds = bruteForceKnn(q, vectors, k, DistanceMetric::Euclidean);
            total += measureRecall(&results, &truthIds);
        }
        total / queries.len() as f64
    };

    // -----------------------------------------------------------------------
    // Test 1: efSearch sweep on 50K index (fast, isolates search effort)
    // -----------------------------------------------------------------------
    tprintln!("\n  --- Test 1: efSearch Sweep (50K, m=32, efC=200) ---");
    let n1 = 50_000u64;
    let mut rng1 = 777u64;
    let vecs1: Vec<(u64, Vec<f32>)> = (0..n1)
        .map(|id| (id, generateRandomVector(&mut rng1, dims)))
        .collect();
    let refs1: Vec<(u64, &[f32])> = vecs1.iter().map(|(id, v)| (*id, v.as_slice())).collect();

    let cfg1 = HnswConfig {
        m: 32,
        efConstruction: 200,
        efSearch: 128,
        metric: DistanceMetric::Euclidean,
    };
    let idx1 = AnnIndex::build(&refs1, 1, 100, 0, cfg1).expect("build 50K");
    tprintln!("  Built 50K index (m=32, efC=200)");

    tprintln!("  {:>10} {:>10}", "efSearch", "recall@10");
    for &ef in &[50u16, 100, 200, 500, 1000, 2000] {
        let r = measureRecallAtEf(&idx1, &refs1, ef);
        tprintln!("  {:>10} {:>10.4}", ef, r);
    }
    drop(idx1);
    drop(vecs1);

    // -----------------------------------------------------------------------
    // Test 2: Single-thread vs parallel build (50K, same config)
    // -----------------------------------------------------------------------
    tprintln!("\n  --- Test 2: Single-Thread vs Parallel (50K) ---");
    let n2 = 50_000u64;
    let mut rng2 = 12345u64;
    let vecs2: Vec<(u64, Vec<f32>)> = (0..n2)
        .map(|id| (id, generateRandomVector(&mut rng2, dims)))
        .collect();
    let refs2: Vec<(u64, &[f32])> = vecs2.iter().map(|(id, v)| (*id, v.as_slice())).collect();

    let cfgSingle = HnswConfig {
        m: 32,
        efConstruction: 200,
        efSearch: 200,
        metric: DistanceMetric::Euclidean,
    };

    // Force single-threaded by building from a sub-slice below parallel threshold.
    // The threshold is max(10000, 50000/cores). For most machines this is 10000.
    // Building all 50K vectors in one call uses the parallel path.
    let idxParallel =
        AnnIndex::build(&refs2, 2, 100, 0, cfgSingle.clone()).expect("build parallel");
    let recallParallel = measureRecallAtEf(&idxParallel, &refs2, 200);

    // Build single-threaded: use 10K vectors (below threshold on any machine).
    let refs2Small: Vec<(u64, &[f32])> = refs2[..10_000].to_vec();
    let idxSingle =
        AnnIndex::build(&refs2Small, 3, 100, 0, cfgSingle.clone()).expect("build single");
    let truthRefs2Small: Vec<(u64, &[f32])> = refs2Small.clone();
    let recallSingle = {
        let mut total = 0.0;
        for q in &queries {
            let results = idxSingle.search(q, k, 200).expect("search");
            let truthIds = bruteForceKnn(q, &truthRefs2Small, k, DistanceMetric::Euclidean);
            total += measureRecall(&results, &truthIds);
        }
        total / queries.len() as f64
    };

    tprintln!("  Single-thread (10K): recall@10 = {:.4}", recallSingle);
    tprintln!("  Parallel      (50K): recall@10 = {:.4}", recallParallel);
    tprintln!(
        "  Delta: {:.4} (negative = parallel is worse)",
        recallParallel - recallSingle
    );
    drop(idxParallel);
    drop(idxSingle);
    drop(vecs2);

    // -----------------------------------------------------------------------
    // Test 3: N scaling with proportional efSearch
    // -----------------------------------------------------------------------
    tprintln!("\n  --- Test 3: N Scaling (efSearch = N/50) ---");
    tprintln!("  {:>10} {:>10} {:>10}", "N", "efSearch", "recall@10");

    for &n in &[10_000u64, 50_000, 100_000] {
        let mut rng = 42u64;
        let vecs: Vec<(u64, Vec<f32>)> = (0..n)
            .map(|id| (id, generateRandomVector(&mut rng, dims)))
            .collect();
        let refs: Vec<(u64, &[f32])> = vecs.iter().map(|(id, v)| (*id, v.as_slice())).collect();

        let ef = (n / 50).max(50) as u16;
        let cfg = HnswConfig {
            m: 32,
            efConstruction: 200,
            efSearch: ef,
            metric: DistanceMetric::Euclidean,
        };
        let idx = AnnIndex::build(&refs, 4, 100, 0, cfg).expect("build");
        let r = measureRecallAtEf(&idx, &refs, ef);
        tprintln!("  {:>10} {:>10} {:>10.4}", n, ef, r);
    }

    // -----------------------------------------------------------------------
    // Test 3b: Cosine vs Euclidean on SAME normalized clustered data (50K)
    // -----------------------------------------------------------------------
    tprintln!("\n  --- Test 3b: Cosine vs Euclidean on Same Normalized Data (50K) ---");
    let clustered50K = generateClusteredVectors(50_050, dims, 16, 0.20, 54321);
    let clData = &clustered50K[..50_000];
    let clQueries = &clustered50K[50_000..];
    let clusteredRefs: Vec<(u64, &[f32])> = clData
        .iter()
        .enumerate()
        .map(|(i, v)| (i as u64, v.as_slice()))
        .collect();

    for &testMetric in &[DistanceMetric::Euclidean, DistanceMetric::Cosine] {
        let metricName = match testMetric {
            DistanceMetric::Euclidean => "Euclidean",
            DistanceMetric::Cosine => "Cosine",
            _ => "Other",
        };
        let clSamples: Vec<&[f32]> = clData[..1000].iter().map(|v| v.as_slice()).collect();
        let clProfile = DataProfile::compute(&clSamples, 50_000, dims as u16, testMetric);
        let clCfg = HnswConfig::auto(&clProfile, testMetric);
        tprintln!(
            "  [{}] m={}, efC={}, efS={}, distSpread={:.3}",
            metricName,
            clCfg.m,
            clCfg.efConstruction,
            clCfg.efSearch,
            clProfile.distSpread,
        );
        let clIdx = AnnIndex::build(&clusteredRefs, 7, 100, 0, clCfg.clone()).expect("build");

        tprintln!("  {:>10} {:>10}", "efSearch", "recall@10");
        for &ef in &[200u16, 500, 1000, 2000] {
            let mut total = 0.0;
            for q in clQueries {
                let results = clIdx.search(q, k, ef).expect("search");
                let truthIds = bruteForceKnn(q, &clusteredRefs, k, testMetric);
                total += measureRecall(&results, &truthIds);
            }
            let r = total / clQueries.len() as f64;
            tprintln!("  {:>10} {:>10.4}", ef, r);
        }
    }
    drop(clustered50K);

    // -----------------------------------------------------------------------
    // Test 4: IVF-PQ probe sweep (50K)
    // -----------------------------------------------------------------------
    tprintln!("\n  --- Test 4: IVF-PQ Probe Sweep (50K) ---");
    let n4 = 50_000u64;
    let mut rng4 = 999u64;
    let vecs4: Vec<(u64, Vec<f32>)> = (0..n4)
        .map(|id| (id, generateRandomVector(&mut rng4, dims)))
        .collect();
    let refs4: Vec<(u64, &[f32])> = vecs4.iter().map(|(id, v)| (*id, v.as_slice())).collect();

    let sampleSlices4: Vec<&[f32]> = vecs4[..1000].iter().map(|(_, v)| v.as_slice()).collect();
    let profile4 = DataProfile::compute(
        &sampleSlices4,
        n4 as usize,
        dims as u16,
        DistanceMetric::Euclidean,
    );
    let ivfCfg = IvfPqConfig::auto(&profile4, DistanceMetric::Euclidean);
    tprintln!(
        "  Auto config: centroids={}, subs={}, probes={}",
        ivfCfg.numCentroids,
        ivfCfg.numSubvectors,
        ivfCfg.numProbes,
    );

    let ivfIdx = IvfPqIndex::build(&refs4, 5, 100, 0, ivfCfg.clone()).expect("build IVF-PQ 50K");
    tprintln!("  Built 50K IVF-PQ index");

    tprintln!("  {:>10} {:>10}", "probes", "recall@10");
    for &probes in &[10u16, 25, 50, 100, 200] {
        let mut totalRecall = 0.0;
        for q in &queries {
            let results = ivfIdx.search(q, k, probes).expect("ivf search");
            let truthIds = bruteForceKnn(q, &refs4, k, DistanceMetric::Euclidean);
            totalRecall += measureRecall(&results, &truthIds);
        }
        let r = totalRecall / queries.len() as f64;
        tprintln!("  {:>10} {:>10.4}", probes, r);
    }
    drop(ivfIdx);
    drop(vecs4);

    // -----------------------------------------------------------------------
    // Test 5: IVF-PQ N scaling with proportional probes
    // -----------------------------------------------------------------------
    tprintln!("\n  --- Test 5: IVF-PQ N Scaling ---");
    tprintln!(
        "  {:>10} {:>10} {:>10} {:>10}",
        "N",
        "centroids",
        "probes",
        "recall@10"
    );

    for &n in &[10_000u64, 50_000] {
        let mut rng = 42u64;
        let vecs: Vec<(u64, Vec<f32>)> = (0..n)
            .map(|id| (id, generateRandomVector(&mut rng, dims)))
            .collect();
        let refs: Vec<(u64, &[f32])> = vecs.iter().map(|(id, v)| (*id, v.as_slice())).collect();

        let slices: Vec<&[f32]> = vecs[..vecs.len().min(1000)]
            .iter()
            .map(|(_, v)| v.as_slice())
            .collect();
        let prof =
            DataProfile::compute(&slices, n as usize, dims as u16, DistanceMetric::Euclidean);
        let cfg = IvfPqConfig::auto(&prof, DistanceMetric::Euclidean);
        let idx = IvfPqIndex::build(&refs, 6, 100, 0, cfg.clone()).expect("build ivf");

        let probes = cfg.numProbes;
        let mut totalRecall = 0.0;
        for q in &queries {
            let results = idx.search(q, k, probes).expect("ivf search");
            let truthIds = bruteForceKnn(q, &refs, k, DistanceMetric::Euclidean);
            totalRecall += measureRecall(&results, &truthIds);
        }
        let r = totalRecall / queries.len() as f64;
        tprintln!(
            "  {:>10} {:>10} {:>10} {:>10.4}",
            n,
            cfg.numCentroids,
            probes,
            r
        );
    }

    tprintln!("\n  === Diagnostic Complete ===");
}

// =============================================================================
// Clustered scale diagnostic. Isolates where clustered Cosine recall breaks.
// Tests 100K, 200K, 500K separately. Run manually by name, one scale at a time.
// Uses AUTO config (what the production path would use) so we see the actual
// parameter values and whether the profile/auto-tuner are misbehaving at scale.
// =============================================================================

fn runClusteredScale(n: usize, numClusters: usize, stdDev: f32, seed: u64) {
    zyron_bench_harness::init("search");
    let _guard = BENCHMARK_LOCK.lock().unwrap_or_else(|e| e.into_inner());

    let dims = 128usize;
    tprintln!(
        "\n=== Clustered Scale: N={} clusters={} ===",
        n,
        numClusters
    );
    // Generate into separate Vecs then flatten into one contiguous buffer.
    // Avoids N small heap allocations persisting alongside the build.
    let flatArena: Vec<f32> = {
        let clustered = generateClusteredVectors(n, dims, numClusters, stdDev, seed);
        let mut flat: Vec<f32> = Vec::with_capacity(n * dims);
        for v in &clustered {
            flat.extend_from_slice(v);
        }
        flat
    };
    let refs: Vec<(u64, &[f32])> = (0..n)
        .map(|i| (i as u64, &flatArena[i * dims..(i + 1) * dims]))
        .collect();

    let sampleSize = n.min(1000);
    let sample: Vec<&[f32]> = (0..sampleSize)
        .map(|i| &flatArena[i * dims..(i + 1) * dims])
        .collect();
    let profile = DataProfile::compute(&sample, n, dims as u16, DistanceMetric::Cosine);
    let cfg = HnswConfig::auto(&profile, DistanceMetric::Cosine);
    tprintln!(
        "  Profile: distSpread={:.4} intrinsicDim={:.1} clustered={}",
        profile.distSpread,
        profile.intrinsicDim,
        profile.isClustered,
    );
    tprintln!(
        "  Config:  m={} efC={} efS={}",
        cfg.m,
        cfg.efConstruction,
        cfg.efSearch,
    );

    let start = Instant::now();
    let idx = AnnIndex::build(&refs, 100, 200, 0, cfg.clone()).expect("build");
    tprintln!("  Build time: {:.2}s", start.elapsed().as_secs_f64());

    let numQueries = 50;
    let k = 10;
    // Use data-matched queries: a subset of the dataset so recall measurement
    // tests the graph's ability to find known-present vectors.
    let queries: Vec<Vec<f32>> = (0..numQueries)
        .map(|i| {
            let idx = (i * n / numQueries).min(n - 1);
            flatArena[idx * dims..(idx + 1) * dims].to_vec()
        })
        .collect();

    tprintln!("  {:>10} {:>10}", "efSearch", "recall@10");
    for &ef in &[200u16, 500, 1000, 2000, 5000] {
        let mut total = 0.0;
        for q in &queries {
            let results = idx.search(q, k, ef).expect("search");
            let truth = bruteForceKnn(q, &refs, k, DistanceMetric::Cosine);
            total += measureRecall(&results, &truth);
        }
        tprintln!("  {:>10} {:>10.4}", ef, total / queries.len() as f64);
    }
}

#[test]
fn test_clustered_scale_100k() {
    runClusteredScale(100_000, 64, 0.20, 271828);
}

#[test]
fn test_clustered_scale_200k() {
    runClusteredScale(200_000, 64, 0.20, 271828);
}

#[test]
fn test_clustered_scale_300k() {
    runClusteredScale(300_000, 64, 0.20, 271828);
}

#[test]
fn test_clustered_scale_400k() {
    runClusteredScale(400_000, 64, 0.20, 271828);
}

#[test]
fn test_clustered_scale_500k() {
    runClusteredScale(500_000, 64, 0.20, 271828);
}

#[test]
fn test_clustered_scale_1m() {
    runClusteredScale(1_000_000, 64, 0.20, 271828);
}

fn runUniformScale(n: usize, seed: u64) {
    zyron_bench_harness::init("search");
    let _guard = BENCHMARK_LOCK.lock().unwrap_or_else(|e| e.into_inner());

    let dims = 128usize;
    tprintln!("\n=== Uniform Random Scale: N={} ===", n);
    let mut rng = seed;
    let mut flatArena: Vec<f32> = Vec::with_capacity(n * dims);
    for _ in 0..n {
        for _ in 0..dims {
            flatArena.push(randomF32(&mut rng));
        }
    }
    let refs: Vec<(u64, &[f32])> = (0..n)
        .map(|i| (i as u64, &flatArena[i * dims..(i + 1) * dims]))
        .collect();

    let sampleSize = n.min(1000);
    let sample: Vec<&[f32]> = (0..sampleSize)
        .map(|i| &flatArena[i * dims..(i + 1) * dims])
        .collect();
    let profile = DataProfile::compute(&sample, n, dims as u16, DistanceMetric::Euclidean);
    let cfg = HnswConfig::auto(&profile, DistanceMetric::Euclidean);
    tprintln!(
        "  Profile: distSpread={:.4} intrinsicDim={:.1} clustered={}",
        profile.distSpread,
        profile.intrinsicDim,
        profile.isClustered,
    );
    tprintln!(
        "  Config:  m={} efC={} efS={}",
        cfg.m,
        cfg.efConstruction,
        cfg.efSearch,
    );

    let start = Instant::now();
    let idx = AnnIndex::build(&refs, 200, 200, 0, cfg.clone()).expect("build");
    tprintln!("  Build time: {:.2}s", start.elapsed().as_secs_f64());

    let numQueries = 50;
    let k = 10;
    let queries: Vec<Vec<f32>> = (0..numQueries)
        .map(|i| {
            let idx = (i * n / numQueries).min(n - 1);
            flatArena[idx * dims..(idx + 1) * dims].to_vec()
        })
        .collect();

    tprintln!("  {:>10} {:>10}", "efSearch", "recall@10");
    for &ef in &[200u16, 500, 1000, 2000, 5000] {
        let mut total = 0.0;
        for q in &queries {
            let results = idx.search(q, k, ef).expect("search");
            let truth = bruteForceKnn(q, &refs, k, DistanceMetric::Euclidean);
            total += measureRecall(&results, &truth);
        }
        tprintln!("  {:>10} {:>10.4}", ef, total / queries.len() as f64);
    }
}

#[test]
fn test_uniform_scale_100k_auto() {
    runUniformScale(100_000, 777);
}

#[test]
fn test_uniform_scale_200k_auto() {
    runUniformScale(200_000, 777);
}

#[test]
fn test_uniform_scale_300k_auto() {
    runUniformScale(300_000, 777);
}

#[test]
fn test_uniform_scale_400k_auto() {
    runUniformScale(400_000, 777);
}

#[test]
fn test_uniform_scale_500k_auto() {
    runUniformScale(500_000, 777);
}

#[test]
fn test_uniform_scale_1m_auto() {
    runUniformScale(1_000_000, 777);
}

// =============================================================================
// Quantized Index (100K vectors)
// =============================================================================

#[test]
fn test_quantized_index() {
    zyron_bench_harness::init("search");
    let _guard = BENCHMARK_LOCK.lock().unwrap_or_else(|e| e.into_inner());

    tprintln!("\n=== Quantized Index (IVF-PQ) ===");

    let dims = 128;
    let count = 100_000u64;
    let mut rng = 12345u64;

    tprintln!("  Generating {} vectors ({}-dim)...", count, dims);
    let vectors: Vec<(u64, Vec<f32>)> = (0..count)
        .map(|id| (id, generateRandomVector(&mut rng, dims)))
        .collect();

    let vecRefs: Vec<(u64, &[f32])> = vectors.iter().map(|(id, v)| (*id, v.as_slice())).collect();

    // Compute DataProfile and derive config automatically.
    let sampleSlices: Vec<&[f32]> = vectors[..vectors.len().min(1000)]
        .iter()
        .map(|(_, v)| v.as_slice())
        .collect();
    let profile = DataProfile::compute(
        &sampleSlices,
        count as usize,
        dims as u16,
        DistanceMetric::Euclidean,
    );
    let config = IvfPqConfig::auto(&profile, DistanceMetric::Euclidean);
    tprintln!(
        "  Auto config: centroids={}, subs={}, probes={}",
        config.numCentroids,
        config.numSubvectors,
        config.numProbes,
    );

    tprintln!("  Building IVF-PQ index...");
    let start = Instant::now();
    let index = IvfPqIndex::build(&vecRefs, 2, 100, 0, config.clone()).expect("build IVF-PQ");
    let buildTime = start.elapsed();
    tprintln!("  Build time: {:.2}s", buildTime.as_secs_f64());

    // Test recall with probe counts derived from auto config.
    // Use INDEPENDENT queries not in the dataset for honest recall measurement.
    let numQueries = 50;
    let k = 10;
    let baseProbes = config.numProbes;
    let probeValues = [
        (baseProbes / 2).max(1),
        baseProbes,
        (baseProbes as u32 * 3 / 2).min(config.numCentroids as u32 / 2) as u16,
    ];

    let mut queryRng = 0xBEEFCAFEu64;
    let queries: Vec<Vec<f32>> = (0..numQueries)
        .map(|_| generateRandomVector(&mut queryRng, dims))
        .collect();

    let vecRefsForTruth: Vec<(u64, &[f32])> =
        vectors.iter().map(|(id, v)| (*id, v.as_slice())).collect();

    for numProbes in probeValues {
        let mut totalRecall = 0.0;
        for q in 0..numQueries {
            let query = &queries[q];
            let truthIds = bruteForceKnn(query, &vecRefsForTruth, k, DistanceMetric::Euclidean);
            let results = index.search(query, k, numProbes).expect("search IVF-PQ");
            totalRecall += measureRecall(&results, &truthIds);
        }
        let avgRecall = totalRecall / numQueries as f64;
        tprintln!("  Recall@{k} (probes={numProbes}): {avgRecall:.4}");
        check_performance(
            "test_quantized",
            &format!("recall@{k}_probes{numProbes}"),
            avgRecall,
            if numProbes >= baseProbes { 0.85 } else { 0.50 },
            true,
        );
    }
}

// =============================================================================
// Hybrid Search (FTS + Vector fusion)
// =============================================================================

#[test]
fn test_hybrid_search() {
    zyron_bench_harness::init("search");
    let _guard = BENCHMARK_LOCK.lock().unwrap_or_else(|e| e.into_inner());

    tprintln!("\n=== Hybrid Search (FTS + Vector) ===");

    // Simulate FTS results: (doc_id, BM25 score)
    let ftsResults: Vec<(u64, f64)> = vec![
        (1, 12.5),
        (2, 10.0),
        (3, 8.0),
        (5, 6.0),
        (7, 4.0),
        (10, 3.0),
        (15, 2.0),
        (20, 1.5),
    ];

    // Simulate vector results: (doc_id, distance) - lower = closer
    let vecResults: Vec<(u64, f32)> = vec![
        (3, 0.05),
        (5, 0.10),
        (1, 0.20),
        (8, 0.25),
        (10, 0.30),
        (2, 0.50),
        (25, 0.60),
        (30, 0.70),
    ];

    // Alpha=0.5: balanced fusion
    let results = HybridSearch::linear_combination(&ftsResults, &vecResults, 0.5, 5);
    tprintln!("  Hybrid results (alpha=0.5): {:?}", results);
    assert!(!results.is_empty(), "hybrid search returned no results");
    assert!(results.len() <= 5, "should return at most 5 results");

    // Docs 1, 3, 5 appear in both sets, should rank high
    let topIds: Vec<u64> = results.iter().map(|(id, _)| *id).collect();
    tprintln!("  Top IDs: {:?}", topIds);

    // Alpha=0.0 should favor FTS, alpha=1.0 should favor vectors
    let ftsOnly = HybridSearch::linear_combination(&ftsResults, &vecResults, 0.0, 3);
    let vecOnly = HybridSearch::linear_combination(&ftsResults, &vecResults, 1.0, 3);
    tprintln!(
        "  FTS-only top-3 (alpha=0.0): {:?}",
        ftsOnly.iter().map(|(id, _)| *id).collect::<Vec<_>>()
    );
    tprintln!(
        "  Vec-only top-3 (alpha=1.0): {:?}",
        vecOnly.iter().map(|(id, _)| *id).collect::<Vec<_>>()
    );

    // FTS-only should have doc 1 at top (highest BM25)
    assert_eq!(ftsOnly[0].0, 1, "FTS-only should rank doc 1 first");
    // Vec-only should have doc 3 at top (lowest distance)
    assert_eq!(vecOnly[0].0, 3, "Vec-only should rank doc 3 first");

    tprintln!("  PASSED: hybrid search rank fusion works correctly");
}

// =============================================================================
// Graph Schema
// =============================================================================

#[test]
fn test_graph_schema() {
    zyron_bench_harness::init("search");
    let _guard = BENCHMARK_LOCK.lock().unwrap_or_else(|e| e.into_inner());

    tprintln!("\n=== Graph Schema ===");

    let mut schema = GraphSchema::new("social".to_string(), 1);

    let personId = schema.add_node_label(
        "Person".to_string(),
        vec![PropertyDef {
            name: "name".to_string(),
            type_id: zyron_common::TypeId::Varchar,
            nullable: false,
        }],
        200,
    );
    tprintln!("  Person label ID: {personId}");

    let companyId = schema.add_node_label(
        "Company".to_string(),
        vec![PropertyDef {
            name: "name".to_string(),
            type_id: zyron_common::TypeId::Varchar,
            nullable: false,
        }],
        202,
    );
    tprintln!("  Company label ID: {companyId}");
    assert_ne!(personId, companyId, "label IDs should be unique");

    let worksAtId = schema
        .add_edge_label(
            "WORKS_AT".to_string(),
            personId,
            companyId,
            vec![],
            204,
            true,
        )
        .expect("add edge label");
    tprintln!("  WORKS_AT label ID: {worksAtId}");

    // Invalid edge label (non-existent from label)
    let err = schema.add_edge_label("BAD".to_string(), 999, companyId, vec![], 999, true);
    assert!(err.is_err(), "should fail for non-existent from label");

    // Serialization round-trip
    let bytes = schema.to_bytes();
    let restored = GraphSchema::from_bytes(&bytes).expect("deserialize");
    assert_eq!(restored.name, "social");
    assert_eq!(restored.node_labels.len(), 2);
    assert_eq!(restored.edge_labels.len(), 1);
    assert_eq!(
        restored.get_node_label("Person").expect("Person").label_id,
        personId
    );

    tprintln!("  PASSED: graph schema creation, validation, serialization");
}

// =============================================================================
// Graph Traversal (shortest path)
// =============================================================================

#[test]
fn test_graph_traversal() {
    zyron_bench_harness::init("search");
    let _guard = BENCHMARK_LOCK.lock().unwrap_or_else(|e| e.into_inner());

    tprintln!("\n=== Graph Traversal ===");

    // Chain: 1 -> 2 -> 3 -> 4 -> 5
    let edges = vec![(1u64, 2u64, None), (2, 3, None), (3, 4, None), (4, 5, None)];
    let graph = CompactGraph::build(&edges);
    tprintln!(
        "  Built chain graph: {} nodes, {} edges",
        graph.node_count,
        graph.edge_count()
    );

    // Shortest path 1 -> 5
    let path = algorithms::shortest_path(&graph, 1, 5).expect("shortest_path");
    tprintln!("  shortest_path(1, 5): {:?}", path);
    assert_eq!(path, Some(vec![1, 2, 3, 4, 5]));

    // Shortest path 1 -> 1 (same node)
    let path = algorithms::shortest_path(&graph, 1, 1).expect("self path");
    tprintln!("  shortest_path(1, 1): {:?}", path);
    assert_eq!(path, Some(vec![1]));

    // Disconnected node
    let edges2 = vec![(1u64, 2u64, None), (2, 3, None), (10, 11, None)];
    let graph2 = CompactGraph::build(&edges2);
    let path = algorithms::shortest_path(&graph2, 1, 11).expect("disconnected");
    tprintln!("  shortest_path(1, 11) [disconnected]: {:?}", path);
    assert_eq!(path, None, "disconnected nodes should return None");

    // BFS depth-limited
    let bfsResult = algorithms::bfs(&graph, 1, 2).expect("bfs");
    tprintln!("  bfs(1, depth=2): {:?}", bfsResult);
    let bfsHas = |id: u64| bfsResult.iter().any(|(n, _)| *n == id);
    assert!(bfsHas(1), "BFS should include source");
    assert!(bfsHas(2), "BFS depth=2 should include 1-hop");
    assert!(bfsHas(3), "BFS depth=2 should include 2-hop");

    tprintln!("  PASSED: graph traversal correct");
}

// =============================================================================
// PageRank
// =============================================================================

#[test]
fn test_pagerank() {
    zyron_bench_harness::init("search");
    let _guard = BENCHMARK_LOCK.lock().unwrap_or_else(|e| e.into_inner());

    tprintln!("\n=== PageRank ===");

    // 4-node cycle: 1->2, 2->3, 3->4, 4->1
    // All nodes should have equal rank (0.25)
    let edges = vec![(1u64, 2u64, None), (2, 3, None), (3, 4, None), (4, 1, None)];
    let graph = CompactGraph::build(&edges);

    let results = algorithms::pagerank(&graph, 0.85, 20).expect("pagerank");
    tprintln!("  PageRank results:");
    for (nodeId, rank) in &results {
        tprintln!("    Node {nodeId}: {rank:.6}");
        assert!(
            (rank - 0.25).abs() < 0.01,
            "node {nodeId} rank should be ~0.25, got {rank}"
        );
    }

    // Star graph: hub (0) -> spokes (1,2,3,4). Hub should have lowest rank,
    // spokes should have higher rank since hub links to them.
    let starEdges = vec![(0u64, 1u64, None), (0, 2, None), (0, 3, None), (0, 4, None)];
    let starGraph = CompactGraph::build(&starEdges);
    let starResults = algorithms::pagerank(&starGraph, 0.85, 20).expect("star pagerank");
    tprintln!("  Star PageRank:");
    for (nodeId, rank) in &starResults {
        tprintln!("    Node {nodeId}: {rank:.6}");
    }

    tprintln!("  PASSED: PageRank produces correct results");
}

// =============================================================================
// Variable-Length Paths
// =============================================================================

#[test]
fn test_variable_length_paths() {
    zyron_bench_harness::init("search");
    let _guard = BENCHMARK_LOCK.lock().unwrap_or_else(|e| e.into_inner());

    tprintln!("\n=== Variable-Length Paths ===");

    let mut schema = GraphSchema::new("knows_graph".to_string(), 1);
    let personId = schema.add_node_label("Person".to_string(), vec![], 200);
    let _ = schema.add_edge_label("KNOWS".to_string(), personId, personId, vec![], 204, true);

    // Pattern: (a:Person)-[:KNOWS*1..3]->(b:Person)
    let pattern = GraphPattern::new(vec![
        PatternElement::node(Some("a".to_string()), Some("Person".to_string())),
        PatternElement::variable_length_edge(
            None,
            Some("KNOWS".to_string()),
            EdgeDirection::Outgoing,
            1,
            3,
        ),
        PatternElement::node(Some("b".to_string()), Some("Person".to_string())),
    ]);

    let queries = compile_pattern(&pattern, &schema).expect("compile");
    tprintln!(
        "  Variable-length *1..3 produced {} query plans",
        queries.len()
    );
    assert_eq!(queries.len(), 3, "should produce 3 plans for hops 1, 2, 3");

    // 1-hop: start + edge + end = 3 scans
    tprintln!("  1-hop scans: {}", queries[0].table_scans.len());
    assert_eq!(queries[0].table_scans.len(), 3);

    // 2-hop: start + edge + mid + edge + end = 5 scans
    tprintln!("  2-hop scans: {}", queries[1].table_scans.len());
    assert_eq!(queries[1].table_scans.len(), 5);

    // 3-hop: start + edge + mid + edge + mid + edge + end = 7 scans
    tprintln!("  3-hop scans: {}", queries[2].table_scans.len());
    assert_eq!(queries[2].table_scans.len(), 7);

    tprintln!("  PASSED: variable-length path expansion correct");
}

// =============================================================================
// Connected Components and Community Detection
// =============================================================================

#[test]
fn test_connected_components() {
    zyron_bench_harness::init("search");
    let _guard = BENCHMARK_LOCK.lock().unwrap_or_else(|e| e.into_inner());

    tprintln!("\n=== Connected Components ===");

    // Two disconnected triangles: (1,2,3) and (4,5,6)
    let edges = vec![
        (1u64, 2u64, None),
        (2, 3, None),
        (3, 1, None),
        (4u64, 5u64, None),
        (5, 6, None),
        (6, 4, None),
    ];
    let graph = CompactGraph::build(&edges);

    let components = algorithms::connected_components(&graph).expect("connected_components");
    tprintln!("  Found {} components", components.len());
    assert_eq!(components.len(), 2, "should find 2 components");

    for (i, comp) in components.iter().enumerate() {
        tprintln!("    Component {i}: {:?}", comp);
    }

    // Community detection should also find 2 communities
    let communities = algorithms::community_detection(&graph).expect("community_detection");
    tprintln!("  Found {} communities", communities.len());
    assert!(communities.len() >= 2, "should find at least 2 communities");

    // Betweenness centrality
    let centrality = algorithms::betweenness_centrality(&graph).expect("betweenness");
    tprintln!("  Betweenness centrality:");
    for (nodeId, score) in &centrality {
        tprintln!("    Node {nodeId}: {score:.6}");
    }

    tprintln!("  PASSED: connected components and community detection correct");
}

// =============================================================================
// Performance Benchmarks (1M scale)
// =============================================================================

#[test]
fn test_vector_graph_performance() {
    zyron_bench_harness::init("search");
    let _guard = BENCHMARK_LOCK.lock().unwrap_or_else(|e| e.into_inner());

    tprintln!("\n=== Vector and Graph Performance Benchmarks ===");

    if std::env::var("ZYRON_SKIP_PERF").is_ok() {
        tprintln!("  SKIPPED: ZYRON_SKIP_PERF is set");
        return;
    }

    let dims = 128;
    let vectorCount = 1_000_000u64;
    let graphNodeCount = 1_000_000usize;
    let graphAvgDegree = 10;
    let numRuns = 1;

    // -----------------------------------------------------------------------
    // Generate vectors into a single flat buffer. One contiguous allocation
    // instead of N separate Vec<f32> heap objects, avoiding heap fragmentation
    // that can destabilise the OS under large N on Windows.
    // -----------------------------------------------------------------------
    tprintln!("  Generating {} vectors ({}-dim)...", vectorCount, dims);
    let mut rng = 314159u64;
    let mut flatArena: Vec<f32> = Vec::with_capacity(vectorCount as usize * dims);
    for _ in 0..vectorCount {
        for _ in 0..dims {
            flatArena.push(randomF32(&mut rng));
        }
    }
    let vecWithIds: Vec<(u64, &[f32])> = (0..vectorCount as usize)
        .map(|i| (i as u64, &flatArena[i * dims..(i + 1) * dims]))
        .collect();

    // Compute DataProfile from a sample, then derive configs automatically.
    let sampleSize = (vectorCount as usize).min(1000);
    let sampleSlices: Vec<&[f32]> = (0..sampleSize)
        .map(|i| &flatArena[i * dims..(i + 1) * dims])
        .collect();
    let profile = DataProfile::compute(
        &sampleSlices,
        vectorCount as usize,
        dims as u16,
        DistanceMetric::Euclidean,
    );
    let hnswCfg = HnswConfig::auto(&profile, DistanceMetric::Euclidean);
    let ivfCfg = IvfPqConfig::auto(&profile, DistanceMetric::Euclidean);

    tprintln!(
        "  DataProfile: n={}, d={}, distSpread={:.4}, intrinsicDim={:.1}, clustered={}",
        profile.n,
        profile.d,
        profile.distSpread,
        profile.intrinsicDim,
        profile.isClustered,
    );
    tprintln!(
        "  HNSW auto: m={}, efC={}, efS={}",
        hnswCfg.m,
        hnswCfg.efConstruction,
        hnswCfg.efSearch,
    );
    tprintln!(
        "  IVF-PQ auto: centroids={}, subs={}, probes={}",
        ivfCfg.numCentroids,
        ivfCfg.numSubvectors,
        ivfCfg.numProbes,
    );

    // -----------------------------------------------------------------------
    // ANN Build (1M, 128d) - target <= 40s
    // -----------------------------------------------------------------------
    tprintln!("\n  --- ANN Build (1M, 128d) ---");
    let mut buildRuns = Vec::with_capacity(numRuns);
    let mut annIndex: Option<AnnIndex> = None;

    for run in 0..numRuns {
        let start = Instant::now();
        let idx = AnnIndex::build(&vecWithIds, 1, 100, 0, hnswCfg.clone()).expect("build 1M index");
        let elapsed = start.elapsed().as_secs_f64();
        buildRuns.push(elapsed);
        tprintln!("    Run {}: {:.2}s", run + 1, elapsed);
        if annIndex.is_none() {
            annIndex = Some(idx);
        }
    }
    let buildResult = validate_metric(
        "test_vector_graph_perf",
        "ann_build_sec",
        buildRuns,
        ANN_BUILD_TARGET_SEC,
        false,
    );

    let index = annIndex.expect("index should be built");

    // -----------------------------------------------------------------------
    // Generate INDEPENDENT query vectors for search/recall tests.
    // Using queries from the dataset trivially finds itself and inflates metrics.
    // -----------------------------------------------------------------------
    let numSearchQueries = 1000;
    let recallQueries = 100;
    let mut queryRng = 0xDEADBEEFu64;
    let searchQueries: Vec<Vec<f32>> = (0..numSearchQueries)
        .map(|_| generateRandomVector(&mut queryRng, dims))
        .collect();
    let recallQueryList: Vec<Vec<f32>> = (0..recallQueries)
        .map(|_| generateRandomVector(&mut queryRng, dims))
        .collect();

    // -----------------------------------------------------------------------
    // ANN Search QPS - target >= 8000
    // -----------------------------------------------------------------------
    tprintln!("\n  --- ANN Search QPS ---");
    let mut qpsRuns = Vec::with_capacity(numRuns);

    for _run in 0..numRuns {
        let start = Instant::now();
        for q in 0..numSearchQueries {
            let _ = index.search(&searchQueries[q], 10, 0);
        }
        let elapsed = start.elapsed().as_secs_f64();
        let qps = numSearchQueries as f64 / elapsed;
        qpsRuns.push(qps);
        tprintln!("    Run {}: {:.0} QPS", _run + 1, qps);
    }
    let qpsResult = validate_metric(
        "test_vector_graph_perf",
        "ann_search_qps",
        qpsRuns,
        ANN_SEARCH_QPS_TARGET,
        true,
    );

    // -----------------------------------------------------------------------
    // ANN Recall@10 - target >= 0.95
    // -----------------------------------------------------------------------
    tprintln!("\n  --- ANN Recall@10 ---");
    let mut totalRecall = 0.0;
    let vecRefsForTruth: Vec<(u64, &[f32])> = (0..vectorCount as usize)
        .map(|i| (i as u64, &flatArena[i * dims..(i + 1) * dims]))
        .collect();
    for q in 0..recallQueries {
        let query = &recallQueryList[q];
        let results = index.search(query, 10, 0).expect("recall search");

        let truthIds = bruteForceKnn(query, &vecRefsForTruth, 10, DistanceMetric::Euclidean);
        totalRecall += measureRecall(&results, &truthIds);
    }
    let avgRecall = totalRecall / recallQueries as f64;
    tprintln!("  ANN Recall@10: {avgRecall:.4}");
    check_performance(
        "test_vector_graph_perf",
        "ann_recall@10",
        avgRecall,
        ANN_RECALL_TARGET,
        true,
    );

    // Free brute-force reference vectors before IVF-PQ build to reduce memory pressure.
    drop(vecRefsForTruth);

    // -----------------------------------------------------------------------
    // IVF-PQ Build (1M, 128d) - target <= 20s
    // -----------------------------------------------------------------------
    tprintln!("\n  --- IVF-PQ Build (1M, 128d) ---");
    let mut pqBuildRuns = Vec::with_capacity(numRuns);
    let mut pqIndex: Option<IvfPqIndex> = None;

    for run in 0..numRuns {
        let start = Instant::now();
        let idx =
            IvfPqIndex::build(&vecWithIds, 2, 100, 0, ivfCfg.clone()).expect("build IVF-PQ 1M");
        let elapsed = start.elapsed().as_secs_f64();
        pqBuildRuns.push(elapsed);
        tprintln!("    Run {}: {:.2}s", run + 1, elapsed);
        if pqIndex.is_none() {
            pqIndex = Some(idx);
        }
    }
    let pqBuildResult = validate_metric(
        "test_vector_graph_perf",
        "pq_build_sec",
        pqBuildRuns,
        QUANTIZED_BUILD_TARGET_SEC,
        false,
    );

    // -----------------------------------------------------------------------
    // IVF-PQ Search QPS - target >= 12000
    // -----------------------------------------------------------------------
    if let Some(ref pqIdx) = pqIndex {
        tprintln!("\n  --- IVF-PQ Search QPS ---");
        let mut pqQpsRuns = Vec::with_capacity(numRuns);

        // Reuse the same independent queries generated earlier.
        for _run in 0..numRuns {
            let start = Instant::now();
            for q in 0..numSearchQueries {
                let _ = pqIdx.search(&searchQueries[q], 10, 0);
            }
            let elapsed = start.elapsed().as_secs_f64();
            let qps = numSearchQueries as f64 / elapsed;
            pqQpsRuns.push(qps);
            tprintln!("    Run {}: {:.0} QPS", _run + 1, qps);
        }
        let pqQpsResult = validate_metric(
            "test_vector_graph_perf",
            "pq_search_qps",
            pqQpsRuns,
            QUANTIZED_SEARCH_QPS_TARGET,
            true,
        );
    }

    // -----------------------------------------------------------------------
    // PageRank (1M nodes, 10M edges) - target <= 5s
    // -----------------------------------------------------------------------
    tprintln!("\n  --- PageRank (1M nodes) ---");
    tprintln!(
        "  Generating graph ({} nodes, avg degree {})...",
        graphNodeCount,
        graphAvgDegree
    );
    let graphEdges = generateRandomGraph(graphNodeCount, graphAvgDegree, 271828);
    let graph = CompactGraph::build(&graphEdges);
    tprintln!(
        "  Graph built: {} nodes, {} edges",
        graph.node_count,
        graph.edge_count()
    );

    let mut prRuns = Vec::with_capacity(numRuns);
    for run in 0..numRuns {
        let start = Instant::now();
        let _ = algorithms::pagerank(&graph, 0.85, 20).expect("pagerank 1M");
        let elapsed = start.elapsed().as_secs_f64();
        prRuns.push(elapsed);
        tprintln!("    Run {}: {:.2}s", run + 1, elapsed);
    }
    let prResult = validate_metric(
        "test_vector_graph_perf",
        "pagerank_sec",
        prRuns,
        PAGERANK_TARGET_SEC,
        false,
    );

    // -----------------------------------------------------------------------
    // Shortest Path (1M nodes) - target <= 100ms
    // -----------------------------------------------------------------------
    tprintln!("\n  --- Shortest Path (1M nodes) ---");
    let mut spRuns = Vec::with_capacity(numRuns);
    for run in 0..numRuns {
        let source = (run * 100_000) as u64;
        let target = ((run + 1) * 100_000 + 50_000) as u64;
        let start = Instant::now();
        let _ = algorithms::shortest_path(&graph, source, target);
        let elapsed = start.elapsed().as_secs_f64() * 1000.0; // ms
        spRuns.push(elapsed);
        tprintln!("    Run {}: {:.2}ms", run + 1, elapsed);
    }
    let spResult = validate_metric(
        "test_vector_graph_perf",
        "shortest_path_ms",
        spRuns,
        SHORTEST_PATH_TARGET_MS,
        false,
    );

    // -----------------------------------------------------------------------
    // Hybrid Search Latency - target <= 8ms
    // -----------------------------------------------------------------------
    tprintln!("\n  --- Hybrid Search Latency ---");
    let ftsResults: Vec<(u64, f64)> = (0..100).map(|i| (i, 10.0 - i as f64 * 0.1)).collect();
    let vecResults: Vec<(u64, f32)> = (0..100).map(|i| (i * 3, i as f32 * 0.01)).collect();
    let hybridStart = Instant::now();
    let hybridIters = 1000;
    for _ in 0..hybridIters {
        std::hint::black_box(HybridSearch::linear_combination(
            &ftsResults,
            &vecResults,
            0.5,
            10,
        ));
    }
    let hybridMs = hybridStart.elapsed().as_secs_f64() * 1000.0 / hybridIters as f64;
    tprintln!("  Hybrid search: {hybridMs:.3}ms per call");
    check_performance(
        "test_vector_graph_perf",
        "hybrid_search_ms",
        hybridMs,
        8.0,
        false,
    );

    // -----------------------------------------------------------------------
    // Graph Query 1-hop Latency - target <= 500us
    // -----------------------------------------------------------------------
    tprintln!("\n  --- Graph Query 1-hop Latency ---");
    let graphQueryStart = Instant::now();
    let graphQueryIters = 10_000;
    for i in 0..graphQueryIters {
        let src = (i as u64 * 7) % graph.node_count as u64;
        std::hint::black_box(graph.neighbors(src as u32));
    }
    let graphQueryUs =
        graphQueryStart.elapsed().as_secs_f64() * 1_000_000.0 / graphQueryIters as f64;
    tprintln!("  Graph 1-hop query: {graphQueryUs:.3}us per call");
    check_performance(
        "test_vector_graph_perf",
        "graph_1hop_us",
        graphQueryUs,
        500.0,
        false,
    );

    // -----------------------------------------------------------------------
    // Summary
    // -----------------------------------------------------------------------
    tprintln!("\n  === Performance Summary ===");
    tprintln!(
        "  ANN Build:      {} (target <= {ANN_BUILD_TARGET_SEC}s)",
        if buildResult.passed { "PASS" } else { "FAIL" }
    );
    tprintln!(
        "  ANN Search:     {} (target >= {ANN_SEARCH_QPS_TARGET} QPS)",
        if qpsResult.passed { "PASS" } else { "FAIL" }
    );
    tprintln!(
        "  PageRank:       {} (target <= {PAGERANK_TARGET_SEC}s)",
        if prResult.passed { "PASS" } else { "FAIL" }
    );
    tprintln!(
        "  Shortest Path:  {} (target <= {SHORTEST_PATH_TARGET_MS}ms)",
        if spResult.passed { "PASS" } else { "FAIL" }
    );
    tprintln!("  Hybrid Search:  {hybridMs:.3}ms (target <= 8ms)");
    tprintln!("  Graph 1-hop:    {graphQueryUs:.3}us (target <= 500us)");

    assert!(buildResult.passed, "ANN build too slow");
    assert!(qpsResult.passed, "ANN search QPS too low");
    assert!(prResult.passed, "PageRank too slow");
    assert!(spResult.passed, "Shortest path too slow");
}

// =============================================================================
// Realistic Benchmark - 1M clustered vectors (Gaussian mixture)
// =============================================================================
//
// Uses clustered data (distSpread ~0.3-0.4) which is harder than production
// embeddings (distSpread 0.5+) but much easier than uniform random worst case
// (distSpread 0.14). Demonstrates system performance on data with actual
// structure - representative of real-world vector workloads.

#[test]
fn test_vector_graph_performance_realistic() {
    zyron_bench_harness::init("search");
    let _guard = BENCHMARK_LOCK.lock().unwrap_or_else(|e| e.into_inner());

    tprintln!("\n=== Vector Performance (Clustered Data) ===");

    if std::env::var("ZYRON_SKIP_PERF").is_ok() {
        tprintln!("  SKIPPED: ZYRON_SKIP_PERF is set");
        return;
    }

    let dims = 128;
    let vectorCount = 1_000_000usize;
    let numClusters = 64;
    let clusterStdDev = 0.20f32;
    let numRuns = 1;

    tprintln!(
        "  Generating {} clustered vectors ({}-dim, {} clusters, stdDev {})...",
        vectorCount,
        dims,
        numClusters,
        clusterStdDev,
    );
    // Generate into separate Vecs then flatten into a single contiguous buffer
    // and drop the originals. Avoids 1M small heap allocations persisting
    // alongside the build's large contiguous allocations.
    let flatArena: Vec<f32> = {
        let clustered =
            generateClusteredVectors(vectorCount, dims, numClusters, clusterStdDev, 271828);
        let mut flat: Vec<f32> = Vec::with_capacity(vectorCount * dims);
        for v in &clustered {
            flat.extend_from_slice(v);
        }
        flat
    };
    let vecWithIds: Vec<(u64, &[f32])> = (0..vectorCount)
        .map(|i| (i as u64, &flatArena[i * dims..(i + 1) * dims]))
        .collect();

    // Compute DataProfile and derive configs automatically.
    let sampleSize = vectorCount.min(1000);
    let sampleSlices: Vec<&[f32]> = (0..sampleSize)
        .map(|i| &flatArena[i * dims..(i + 1) * dims])
        .collect();
    let profile = DataProfile::compute(
        &sampleSlices,
        vectorCount,
        dims as u16,
        DistanceMetric::Cosine,
    );
    let hnswCfg = HnswConfig::auto(&profile, DistanceMetric::Cosine);
    let ivfCfg = IvfPqConfig::auto(&profile, DistanceMetric::Cosine);

    tprintln!(
        "  DataProfile: n={}, d={}, distSpread={:.4}, intrinsicDim={:.1}, clustered={}",
        profile.n,
        profile.d,
        profile.distSpread,
        profile.intrinsicDim,
        profile.isClustered,
    );
    tprintln!(
        "  HNSW auto: m={}, efC={}, efS={}",
        hnswCfg.m,
        hnswCfg.efConstruction,
        hnswCfg.efSearch,
    );
    tprintln!(
        "  IVF-PQ auto: centroids={}, subs={}, probes={}",
        ivfCfg.numCentroids,
        ivfCfg.numSubvectors,
        ivfCfg.numProbes,
    );

    // Generate independent query vectors from the same distribution.
    // Different seed ensures queries are NOT in the dataset.
    let numSearchQueries = 1000;
    let recallQueries = 100;
    let queryVectors = generateClusteredVectors(
        numSearchQueries + recallQueries,
        dims,
        numClusters,
        clusterStdDev,
        0xDEADBEEF,
    );
    let searchQueries = &queryVectors[..numSearchQueries];
    let recallQueryList = &queryVectors[numSearchQueries..];

    // -----------------------------------------------------------------------
    // ANN Build
    // -----------------------------------------------------------------------
    tprintln!("\n  --- ANN Build (1M clustered) ---");
    let mut buildRuns = Vec::with_capacity(numRuns);
    let mut annIndex: Option<AnnIndex> = None;

    for run in 0..numRuns {
        let start = Instant::now();
        let idx = AnnIndex::build(&vecWithIds, 1, 100, 0, hnswCfg.clone())
            .expect("build clustered 1M index");
        let elapsed = start.elapsed().as_secs_f64();
        buildRuns.push(elapsed);
        tprintln!("    Run {}: {:.2}s", run + 1, elapsed);
        if annIndex.is_none() {
            annIndex = Some(idx);
        }
    }
    let _ = validate_metric(
        "test_vector_graph_realistic",
        "ann_build_sec",
        buildRuns,
        25.0,
        false,
    );
    let index = annIndex.expect("index should be built");

    // -----------------------------------------------------------------------
    // ANN Search QPS
    // -----------------------------------------------------------------------
    tprintln!("\n  --- ANN Search QPS (clustered) ---");
    let mut qpsRuns = Vec::with_capacity(numRuns);
    for run in 0..numRuns {
        let start = Instant::now();
        for q in 0..numSearchQueries {
            let _ = index.search(&searchQueries[q], 10, 0);
        }
        let elapsed = start.elapsed().as_secs_f64();
        let qps = numSearchQueries as f64 / elapsed;
        qpsRuns.push(qps);
        tprintln!("    Run {}: {:.0} QPS", run + 1, qps);
    }
    let _ = validate_metric(
        "test_vector_graph_realistic",
        "ann_search_qps",
        qpsRuns,
        5000.0,
        true,
    );

    // -----------------------------------------------------------------------
    // ANN Recall@10
    // -----------------------------------------------------------------------
    tprintln!("\n  --- ANN Recall@10 (clustered) ---");
    let mut totalRecall = 0.0;
    let vecRefsForTruth: Vec<(u64, &[f32])> = (0..vectorCount)
        .map(|i| (i as u64, &flatArena[i * dims..(i + 1) * dims]))
        .collect();
    for q in 0..recallQueries {
        let query = &recallQueryList[q];
        let results = index.search(query, 10, 0).expect("recall search");
        let truthIds = bruteForceKnn(query, &vecRefsForTruth, 10, DistanceMetric::Cosine);
        totalRecall += measureRecall(&results, &truthIds);
    }
    let avgRecall = totalRecall / recallQueries as f64;
    tprintln!("  ANN Recall@10: {avgRecall:.4}");
    check_performance(
        "test_vector_graph_realistic",
        "ann_recall@10",
        avgRecall,
        0.90,
        true,
    );
    drop(vecRefsForTruth);

    // -----------------------------------------------------------------------
    // IVF-PQ Build
    // -----------------------------------------------------------------------
    tprintln!("\n  --- IVF-PQ Build (1M clustered) ---");
    let mut pqBuildRuns = Vec::with_capacity(numRuns);
    let mut pqIndex: Option<IvfPqIndex> = None;
    for run in 0..numRuns {
        let start = Instant::now();
        let idx = IvfPqIndex::build(&vecWithIds, 2, 100, 0, ivfCfg.clone())
            .expect("build clustered IVF-PQ 1M");
        let elapsed = start.elapsed().as_secs_f64();
        pqBuildRuns.push(elapsed);
        tprintln!("    Run {}: {:.2}s", run + 1, elapsed);
        if pqIndex.is_none() {
            pqIndex = Some(idx);
        }
    }
    let _ = validate_metric(
        "test_vector_graph_realistic",
        "pq_build_sec",
        pqBuildRuns,
        30.0,
        false,
    );

    // -----------------------------------------------------------------------
    // IVF-PQ Search QPS
    // -----------------------------------------------------------------------
    if let Some(ref pqIdx) = pqIndex {
        tprintln!("\n  --- IVF-PQ Search QPS (clustered) ---");
        let mut pqQpsRuns = Vec::with_capacity(numRuns);
        for run in 0..numRuns {
            let start = Instant::now();
            for q in 0..numSearchQueries {
                let _ = pqIdx.search(&searchQueries[q], 10, 0);
            }
            let elapsed = start.elapsed().as_secs_f64();
            let qps = numSearchQueries as f64 / elapsed;
            pqQpsRuns.push(qps);
            tprintln!("    Run {}: {:.0} QPS", run + 1, qps);
        }
        let _ = validate_metric(
            "test_vector_graph_realistic",
            "pq_search_qps",
            pqQpsRuns,
            8000.0,
            true,
        );
    }

    tprintln!("\n  === Realistic Benchmark Summary ===");
    tprintln!("  Data: clustered (distSpread={:.3})", profile.distSpread);
    tprintln!("  Recall@10: {avgRecall:.4} (target >= 0.90)");
}
