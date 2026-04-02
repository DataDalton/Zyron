//! Synonym expansion for full-text search.
//!
//! Manages synonym groups that map terms to their equivalents. Used by
//! the SynonymFilter in the analyzer pipeline and for query-time expansion
//! via EXPAND SYNONYMS syntax.

use std::collections::HashMap;

/// A named set of synonym groups. Each group contains bidirectional
/// synonym mappings (every term expands to every other term in the group).
#[derive(Debug, Clone)]
pub struct SynonymSet {
    pub name: String,
    /// Maps each term to its synonym expansions.
    expansions: HashMap<String, Vec<String>>,
}

impl SynonymSet {
    /// Creates a new empty synonym set with the given name.
    pub fn new(name: &str) -> Self {
        Self {
            name: name.to_string(),
            expansions: HashMap::new(),
        }
    }

    /// Adds a bidirectional synonym group. Every term in the group
    /// will expand to every other term in the group.
    pub fn add_group(&mut self, terms: &[&str]) {
        let lowered: Vec<String> = terms.iter().map(|t| t.to_lowercase()).collect();
        for (i, lower) in lowered.iter().enumerate() {
            let synonyms: Vec<String> = lowered
                .iter()
                .enumerate()
                .filter(|&(j, _)| j != i)
                .map(|(_, s)| s.clone())
                .collect();
            self.expansions
                .entry(lower.clone())
                .or_insert_with(Vec::new)
                .extend(synonyms);
        }
        // Deduplicate
        for synonyms in self.expansions.values_mut() {
            synonyms.sort();
            synonyms.dedup();
        }
    }

    /// Adds a one-directional synonym mapping. The source term expands
    /// to the targets, but not vice versa.
    pub fn add_mapping(&mut self, source: &str, targets: &[&str]) {
        let lower = source.to_lowercase();
        let target_list: Vec<String> = targets.iter().map(|t| t.to_lowercase()).collect();
        self.expansions
            .entry(lower)
            .or_insert_with(Vec::new)
            .extend(target_list);
        // Deduplicate
        if let Some(synonyms) = self.expansions.get_mut(&source.to_lowercase()) {
            synonyms.sort();
            synonyms.dedup();
        }
    }

    /// Returns the synonym expansions for a term, or an empty slice if none.
    pub fn expand(&self, term: &str) -> Vec<String> {
        self.expansions
            .get(&term.to_lowercase())
            .cloned()
            .unwrap_or_default()
    }

    /// Returns true if the term has synonyms in this set.
    pub fn has_synonyms(&self, term: &str) -> bool {
        self.expansions.contains_key(&term.to_lowercase())
    }

    /// Returns all term -> synonyms mappings as a reference to the internal map.
    /// Useful for building a SynonymFilter.
    pub fn all_expansions(&self) -> &HashMap<String, Vec<String>> {
        &self.expansions
    }

    /// Returns the total number of terms with synonym mappings.
    pub fn term_count(&self) -> usize {
        self.expansions.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_add_group_bidirectional() {
        let mut set = SynonymSet::new("test");
        set.add_group(&["database", "db", "datastore"]);

        let expanded = set.expand("database");
        assert!(expanded.contains(&"db".to_string()));
        assert!(expanded.contains(&"datastore".to_string()));

        let expanded = set.expand("db");
        assert!(expanded.contains(&"database".to_string()));
        assert!(expanded.contains(&"datastore".to_string()));
    }

    #[test]
    fn test_add_mapping_directional() {
        let mut set = SynonymSet::new("test");
        set.add_mapping("automobile", &["car", "vehicle"]);

        let expanded = set.expand("automobile");
        assert!(expanded.contains(&"car".to_string()));
        assert!(expanded.contains(&"vehicle".to_string()));

        // Reverse should not exist
        assert!(set.expand("car").is_empty());
    }

    #[test]
    fn test_expand_no_match() {
        let set = SynonymSet::new("test");
        assert!(set.expand("unknown").is_empty());
    }

    #[test]
    fn test_case_insensitive() {
        let mut set = SynonymSet::new("test");
        set.add_group(&["Database", "DB"]);

        assert!(!set.expand("database").is_empty());
        assert!(!set.expand("DATABASE").is_empty());
    }

    #[test]
    fn test_has_synonyms() {
        let mut set = SynonymSet::new("test");
        set.add_group(&["fast", "quick", "rapid"]);

        assert!(set.has_synonyms("fast"));
        assert!(set.has_synonyms("quick"));
        assert!(!set.has_synonyms("slow"));
    }

    #[test]
    fn test_term_count() {
        let mut set = SynonymSet::new("test");
        assert_eq!(set.term_count(), 0);

        set.add_group(&["a", "b", "c"]);
        assert_eq!(set.term_count(), 3);
    }

    #[test]
    fn test_deduplication() {
        let mut set = SynonymSet::new("test");
        set.add_group(&["a", "b"]);
        set.add_group(&["a", "b", "c"]);

        let expanded = set.expand("a");
        // "b" should appear only once
        let b_count = expanded.iter().filter(|s| s.as_str() == "b").count();
        assert_eq!(b_count, 1);
    }

    #[test]
    fn test_empty_group() {
        let mut set = SynonymSet::new("test");
        set.add_group(&[]);
        assert_eq!(set.term_count(), 0);
    }

    #[test]
    fn test_single_term_group() {
        let mut set = SynonymSet::new("test");
        set.add_group(&["alone"]);
        // A single term has no synonyms (it maps to empty)
        assert!(set.expand("alone").is_empty());
    }
}
