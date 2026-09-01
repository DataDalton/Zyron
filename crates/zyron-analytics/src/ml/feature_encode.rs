#![allow(non_snake_case)]
// Feature encoding transforms for categorical and text columns
// Target encoding, TF-IDF, label encoding, robust scaling, one hot
// expansion in long form, and simple text statistics

use std::collections::HashMap;
use zyron_common::error::{Result, ZyronError};

/// Target encoding with global mean smoothing
/// encoded = (count * categoryMean + smoothing * globalMean) / (count + smoothing)
/// Returns the per row encoded values and the sorted (category, encoded) map
pub fn targetEncode(
    categories: &[String],
    targets: &[f64],
    smoothing: f64,
) -> Result<(Vec<f64>, Vec<(String, f64)>)> {
    let n = categories.len();
    if n == 0 || targets.len() != n {
        return Err(ZyronError::InvalidParameter {
            name: "categories".to_string(),
            value: "categories and targets must be nonempty and equal length".to_string(),
        });
    }
    if !(smoothing >= 0.0 && smoothing.is_finite()) {
        return Err(ZyronError::InvalidParameter {
            name: "smoothing".to_string(),
            value: format!("must be a nonnegative finite number, got {}", smoothing),
        });
    }
    if targets.iter().any(|v| !v.is_finite()) {
        return Err(ZyronError::InvalidParameter {
            name: "targets".to_string(),
            value: "targets contain non finite values".to_string(),
        });
    }
    let global_mean = targets.iter().sum::<f64>() / n as f64;
    let mut sums: HashMap<&str, (f64, u64)> = HashMap::new();
    for i in 0..n {
        let entry = sums.entry(categories[i].as_str()).or_insert((0.0, 0));
        entry.0 += targets[i];
        entry.1 += 1;
    }
    let mut mapping: Vec<(String, f64)> = sums
        .iter()
        .map(|(cat, (sum, count))| {
            let cf = *count as f64;
            let cat_mean = sum / cf;
            let encoded = (cf * cat_mean + smoothing * global_mean) / (cf + smoothing);
            (cat.to_string(), encoded)
        })
        .collect();
    mapping.sort_by(|a, b| a.0.cmp(&b.0));
    let lookup: HashMap<&str, f64> = mapping.iter().map(|(c, e)| (c.as_str(), *e)).collect();
    let encoded: Vec<f64> = categories
        .iter()
        .map(|c| lookup.get(c.as_str()).copied().unwrap_or(global_mean))
        .collect();
    Ok((encoded, mapping))
}

fn tokenize(doc: &str) -> Vec<String> {
    doc.split_whitespace()
        .map(|t| t.to_lowercase())
        .filter(|t| !t.is_empty())
        .collect()
}

/// TF-IDF with smoothed logarithmic inverse document frequency
/// idf = ln((1 + docs) / (1 + docFrequency)) + 1, tf is the term share of
/// the document. Tokenization is lowercase whitespace splitting. Each
/// document yields its terms sorted lexicographically
pub fn tfidf(docs: &[String]) -> Vec<Vec<(String, f64)>> {
    let n = docs.len();
    let tokenized: Vec<Vec<String>> = docs.iter().map(|d| tokenize(d)).collect();
    let mut doc_freq: HashMap<&str, u64> = HashMap::new();
    for tokens in &tokenized {
        let mut seen: HashMap<&str, ()> = HashMap::new();
        for t in tokens {
            if seen.insert(t.as_str(), ()).is_none() {
                *doc_freq.entry(t.as_str()).or_insert(0) += 1;
            }
        }
    }
    let mut out = Vec::with_capacity(n);
    for tokens in &tokenized {
        let total = tokens.len() as f64;
        let mut counts: HashMap<&str, u64> = HashMap::new();
        for t in tokens {
            *counts.entry(t.as_str()).or_insert(0) += 1;
        }
        let mut scored: Vec<(String, f64)> = counts
            .iter()
            .map(|(term, count)| {
                let tf = *count as f64 / total;
                let df = doc_freq.get(term).copied().unwrap_or(0) as f64;
                let idf = ((1.0 + n as f64) / (1.0 + df)).ln() + 1.0;
                (term.to_string(), tf * idf)
            })
            .collect();
        scored.sort_by(|a, b| a.0.cmp(&b.0));
        out.push(scored);
    }
    out
}

/// Integer codes from the sorted distinct category list
/// Returns per row codes and the mapping where code i names mapping[i]
pub fn labelEncode(categories: &[String]) -> (Vec<i64>, Vec<String>) {
    let mut mapping: Vec<String> = categories.to_vec();
    mapping.sort();
    mapping.dedup();
    let lookup: HashMap<&str, i64> = mapping
        .iter()
        .enumerate()
        .map(|(i, c)| (c.as_str(), i as i64))
        .collect();
    let codes = categories
        .iter()
        .map(|c| lookup.get(c.as_str()).copied().unwrap_or(-1))
        .collect();
    (codes, mapping)
}

/// Robust scaling by median and interquartile range
/// Returns (scaled values, median, iqr). A zero IQR is an error because
/// the scale would divide by zero
pub fn robustScale(values: &[f64]) -> Result<(Vec<f64>, f64, f64)> {
    let finite: Vec<f64> = values.iter().copied().filter(|v| v.is_finite()).collect();
    if finite.is_empty() {
        return Err(ZyronError::InvalidParameter {
            name: "values".to_string(),
            value: "no finite values".to_string(),
        });
    }
    let mut sorted = finite;
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let quantile = |q: f64| -> f64 {
        let pos = q * (sorted.len() as f64 - 1.0);
        let lo = pos.floor() as usize;
        let hi = (pos.ceil() as usize).min(sorted.len() - 1);
        let frac = pos - lo as f64;
        sorted[lo] + (sorted[hi] - sorted[lo]) * frac
    };
    let median = quantile(0.5);
    let iqr = quantile(0.75) - quantile(0.25);
    if iqr == 0.0 {
        return Err(ZyronError::InvalidParameter {
            name: "values".to_string(),
            value: "interquartile range is zero, robust scaling is undefined".to_string(),
        });
    }
    let scaled = values.iter().map(|v| (v - median) / iqr).collect();
    Ok((scaled, median, iqr))
}

/// One hot expansion in long form
/// Emits (rowIndex, category, indicator) for every row and every distinct
/// category, sorted category order within a row
pub fn oneHotEncode(categories: &[String]) -> Vec<(usize, String, i64)> {
    let mut levels: Vec<String> = categories.to_vec();
    levels.sort();
    levels.dedup();
    let mut out = Vec::with_capacity(categories.len() * levels.len());
    for (i, cat) in categories.iter().enumerate() {
        for level in &levels {
            out.push((i, level.clone(), if cat == level { 1 } else { 0 }));
        }
    }
    out
}

#[derive(Debug, Clone)]
pub struct TextFeatureRow {
    // total character count
    pub length: i64,
    pub wordCount: i64,
    // non whitespace character count
    pub charCount: i64,
    pub avgWordLength: f64,
    // maximal runs of sentence terminators . ! ?
    pub sentenceCount: i64,
    // ASCII punctuation share of all characters
    pub punctRatio: f64,
}

/// Simple per text statistics for feature engineering
pub fn textFeatures(texts: &[String]) -> Vec<TextFeatureRow> {
    texts
        .iter()
        .map(|t| {
            let length = t.chars().count() as i64;
            let mut word_count = 0i64;
            let mut word_chars = 0i64;
            for w in t.split_whitespace() {
                word_count += 1;
                word_chars += w.chars().count() as i64;
            }
            let mut punct = 0i64;
            let mut sentences = 0i64;
            let mut in_terminator_run = false;
            let mut char_count = 0i64;
            for c in t.chars() {
                if !c.is_whitespace() {
                    char_count += 1;
                }
                if c.is_ascii_punctuation() {
                    punct += 1;
                }
                if matches!(c, '.' | '!' | '?') {
                    if !in_terminator_run {
                        sentences += 1;
                        in_terminator_run = true;
                    }
                } else {
                    in_terminator_run = false;
                }
            }
            TextFeatureRow {
                length,
                wordCount: word_count,
                charCount: char_count,
                avgWordLength: if word_count > 0 {
                    word_chars as f64 / word_count as f64
                } else {
                    0.0
                },
                sentenceCount: sentences,
                punctRatio: if length > 0 {
                    punct as f64 / length as f64
                } else {
                    0.0
                },
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn strs(v: &[&str]) -> Vec<String> {
        v.iter().map(|s| s.to_string()).collect()
    }

    #[test]
    fn targetEncodeSmoothingBounds() {
        let cats = strs(&["a", "a", "a", "b"]);
        let targets = [1.0, 1.0, 1.0, 0.0];
        let global = 0.75;
        // no smoothing reproduces the raw category means
        let (raw, _) = targetEncode(&cats, &targets, 0.0).expect("raw");
        assert!((raw[0] - 1.0).abs() < 1e-12);
        assert!((raw[3] - 0.0).abs() < 1e-12);
        // moderate smoothing pulls each category toward the global mean
        let (smooth, mapping) = targetEncode(&cats, &targets, 5.0).expect("smooth");
        assert!(smooth[0] < 1.0 && smooth[0] > global);
        assert!(smooth[3] > 0.0 && smooth[3] < global);
        assert_eq!(mapping.len(), 2);
        // extreme smoothing converges on the global mean
        let (heavy, _) = targetEncode(&cats, &targets, 1e9).expect("heavy");
        assert!((heavy[0] - global).abs() < 1e-6);
        assert!((heavy[3] - global).abs() < 1e-6);
    }

    #[test]
    fn tfidfScoresRareTermsHigher() {
        let docs = strs(&["apple banana", "apple cherry", "apple durian"]);
        let out = tfidf(&docs);
        assert_eq!(out.len(), 3);
        let first = &out[0];
        let apple = first.iter().find(|(t, _)| t == "apple").expect("apple");
        let banana = first.iter().find(|(t, _)| t == "banana").expect("banana");
        assert!(
            banana.1 > apple.1,
            "banana {} vs apple {}",
            banana.1,
            apple.1
        );
    }

    #[test]
    fn labelEncodeIsSortedAndStable() {
        let cats = strs(&["b", "a", "c", "a"]);
        let (codes, mapping) = labelEncode(&cats);
        assert_eq!(mapping, strs(&["a", "b", "c"]));
        assert_eq!(codes, vec![1, 0, 2, 0]);
    }

    #[test]
    fn robustScaleZeroIqrErrors() {
        let constant = vec![5.0; 12];
        assert!(robustScale(&constant).is_err());
        let spread = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let (scaled, median, iqr) = robustScale(&spread).expect("scale");
        assert!((median - 3.0).abs() < 1e-12);
        assert!((iqr - 2.0).abs() < 1e-12);
        assert!((scaled[2] - 0.0).abs() < 1e-12);
    }

    #[test]
    fn oneHotLongForm() {
        let cats = strs(&["x", "y"]);
        let out = oneHotEncode(&cats);
        assert_eq!(out.len(), 4);
        assert_eq!(out[0], (0, "x".to_string(), 1));
        assert_eq!(out[1], (0, "y".to_string(), 0));
        assert_eq!(out[3], (1, "y".to_string(), 1));
    }

    #[test]
    fn textFeaturesCountsSentences() {
        let texts = strs(&["Hello world. How are you? Fine!!", ""]);
        let out = textFeatures(&texts);
        assert_eq!(out[0].sentenceCount, 3);
        assert_eq!(out[0].wordCount, 6);
        assert!(out[0].punctRatio > 0.0);
        assert_eq!(out[1].length, 0);
        assert_eq!(out[1].avgWordLength, 0.0);
    }
}
