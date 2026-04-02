//! Relevance scoring for full-text search.
//!
//! Provides the BM25 scoring algorithm (default) and a trait for pluggable
//! scoring functions. BM25 accounts for term frequency, inverse document
//! frequency, and document length normalization.

/// Computes a relevance score for a single term in a single document.
pub trait RelevanceScorer: Send + Sync {
    /// Scores a term occurrence.
    ///
    /// - `term_freq`: number of times the term appears in this document
    /// - `doc_freq`: number of documents containing this term
    /// - `doc_length`: total token count of this document
    /// - `avg_doc_length`: average document length across the index
    /// - `total_docs`: total number of documents in the index
    fn score(
        &self,
        term_freq: u16,
        doc_freq: u32,
        doc_length: u32,
        avg_doc_length: f64,
        total_docs: u64,
    ) -> f64;

    fn name(&self) -> &str;
}

/// Okapi BM25 scorer. The default scoring function for full-text search.
///
/// Formula: IDF * (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * dl / avgdl))
/// where IDF = ln((N - df + 0.5) / (df + 0.5) + 1)
pub struct Bm25Scorer {
    /// Term frequency saturation parameter. Higher values give more weight
    /// to repeated terms. Default: 1.2.
    pub k1: f64,
    /// Document length normalization parameter. 0.0 disables length
    /// normalization, 1.0 applies full normalization. Default: 0.75.
    pub b: f64,
}

impl Default for Bm25Scorer {
    fn default() -> Self {
        Self { k1: 1.2, b: 0.75 }
    }
}

impl Bm25Scorer {
    pub fn new(k1: f64, b: f64) -> Self {
        Self { k1, b }
    }
}

impl RelevanceScorer for Bm25Scorer {
    fn score(
        &self,
        term_freq: u16,
        doc_freq: u32,
        doc_length: u32,
        avg_doc_length: f64,
        total_docs: u64,
    ) -> f64 {
        if total_docs == 0 || term_freq == 0 {
            return 0.0;
        }

        let tf = term_freq as f64;
        let df = (doc_freq as f64).min(total_docs as f64);
        let dl = doc_length as f64;
        let n = total_docs as f64;

        // Inverse document frequency
        let idf = ((n - df + 0.5) / (df + 0.5) + 1.0).ln();

        // Term frequency normalization
        let avgdl = if avg_doc_length > 0.0 {
            avg_doc_length
        } else {
            1.0
        };
        let tf_norm =
            (tf * (self.k1 + 1.0)) / (tf + self.k1 * (1.0 - self.b + self.b * dl / avgdl));

        idf * tf_norm
    }

    fn name(&self) -> &str {
        "bm25"
    }
}

/// Per-field boost weight for multi-field scoring.
#[derive(Debug, Clone)]
pub struct FieldBoost {
    pub column_id: u16,
    pub boost: f64,
}

/// Combines per-field scores with boost weights.
/// Each field score is multiplied by its boost factor and summed.
pub fn score_multi_field(per_field_scores: &[(u16, f64)], boosts: &[FieldBoost]) -> f64 {
    let mut total = 0.0;
    for (col_id, score) in per_field_scores {
        let boost = boosts
            .iter()
            .find(|b| b.column_id == *col_id)
            .map(|b| b.boost)
            .unwrap_or(1.0);
        total += score * boost;
    }
    total
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bm25_basic_score() {
        let scorer = Bm25Scorer::default();
        let score = scorer.score(3, 10, 100, 120.0, 1000);
        assert!(score > 0.0, "BM25 score should be positive");
    }

    #[test]
    fn test_bm25_higher_tf_higher_score() {
        let scorer = Bm25Scorer::default();
        let low_tf = scorer.score(1, 10, 100, 100.0, 1000);
        let high_tf = scorer.score(5, 10, 100, 100.0, 1000);
        assert!(high_tf > low_tf, "higher tf should produce higher score");
    }

    #[test]
    fn test_bm25_rarer_term_higher_score() {
        let scorer = Bm25Scorer::default();
        let common = scorer.score(3, 500, 100, 100.0, 1000);
        let rare = scorer.score(3, 5, 100, 100.0, 1000);
        assert!(rare > common, "rarer term (lower df) should score higher");
    }

    #[test]
    fn test_bm25_shorter_doc_higher_score() {
        let scorer = Bm25Scorer::default();
        let long_doc = scorer.score(3, 10, 500, 100.0, 1000);
        let short_doc = scorer.score(3, 10, 50, 100.0, 1000);
        assert!(
            short_doc > long_doc,
            "shorter doc with same tf should score higher"
        );
    }

    #[test]
    fn test_bm25_zero_avgdl_no_panic() {
        let scorer = Bm25Scorer::default();
        let score = scorer.score(1, 1, 10, 0.0, 100);
        assert!(score.is_finite());
    }

    #[test]
    fn test_bm25_custom_params() {
        let scorer = Bm25Scorer::new(2.0, 0.5);
        let score = scorer.score(2, 10, 100, 100.0, 1000);
        assert!(score > 0.0);
    }

    #[test]
    fn test_field_boost() {
        let scores = vec![(1u16, 2.0), (2u16, 3.0)];
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
        let total = score_multi_field(&scores, &boosts);
        // 2.0 * 2.0 + 3.0 * 1.0 = 7.0
        assert!((total - 7.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_field_boost_default() {
        let scores = vec![(1u16, 5.0)];
        let boosts: Vec<FieldBoost> = vec![];
        let total = score_multi_field(&scores, &boosts);
        // No boost found, default 1.0
        assert!((total - 5.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_bm25_tf_saturation() {
        let scorer = Bm25Scorer::default();
        let tf_10 = scorer.score(10, 10, 100, 100.0, 1000);
        let tf_100 = scorer.score(100, 10, 100, 100.0, 1000);
        // Score growth should saturate: tf_100 should not be 10x tf_10
        assert!(tf_100 < tf_10 * 3.0, "BM25 should saturate term frequency");
    }
}
