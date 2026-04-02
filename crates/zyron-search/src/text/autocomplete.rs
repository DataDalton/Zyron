//! Prefix-based autocomplete using a trie data structure.
//!
//! Provides O(prefix_length) lookup for term suggestions, ordered by
//! frequency. Built from inverted index term dictionaries.

use std::collections::HashMap;

/// Trie-based prefix index for autocomplete lookups.
pub struct PrefixIndex {
    root: TrieNode,
}

/// A suggestion returned by autocomplete.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Suggestion {
    pub term: String,
    pub frequency: u32,
}

struct TrieNode {
    children: HashMap<u8, Box<TrieNode>>,
    is_terminal: bool,
    frequency: u32,
}

impl TrieNode {
    fn new() -> Self {
        Self {
            children: HashMap::new(),
            is_terminal: false,
            frequency: 0,
        }
    }
}

impl PrefixIndex {
    /// Creates a new empty prefix index.
    pub fn new() -> Self {
        Self {
            root: TrieNode::new(),
        }
    }

    /// Inserts a term with the given frequency into the trie.
    pub fn insert(&mut self, term: &str, frequency: u32) {
        let mut node = &mut self.root;
        for byte in term.as_bytes() {
            node = node
                .children
                .entry(*byte)
                .or_insert_with(|| Box::new(TrieNode::new()));
        }
        node.is_terminal = true;
        node.frequency = node.frequency.saturating_add(frequency);
    }

    /// Returns up to `limit` suggestions matching the given prefix,
    /// sorted by frequency descending.
    pub fn suggest(&self, prefix: &str, limit: usize) -> Vec<Suggestion> {
        // Navigate to the prefix node
        let mut node = &self.root;
        for byte in prefix.as_bytes() {
            match node.children.get(byte) {
                Some(child) => node = child,
                None => return Vec::new(),
            }
        }

        // Collect all completions from this node
        let mut results = Vec::new();
        let mut current_term = prefix.to_string();
        Self::collect_terms(node, &mut current_term, &mut results);

        // Sort by frequency descending
        results.sort_by(|a, b| b.frequency.cmp(&a.frequency));
        results.truncate(limit);
        results
    }

    /// Returns the total number of terms in the index.
    pub fn term_count(&self) -> usize {
        Self::count_terminals(&self.root)
    }

    fn collect_terms(node: &TrieNode, current: &mut String, results: &mut Vec<Suggestion>) {
        if node.is_terminal {
            results.push(Suggestion {
                term: current.clone(),
                frequency: node.frequency,
            });
        }
        // Sort children for deterministic output
        let mut keys: Vec<u8> = node.children.keys().copied().collect();
        keys.sort();
        for key in keys {
            if let Some(child) = node.children.get(&key) {
                current.push(key as char);
                Self::collect_terms(child, current, results);
                current.pop();
            }
        }
    }

    fn count_terminals(node: &TrieNode) -> usize {
        let mut count = if node.is_terminal { 1 } else { 0 };
        for child in node.children.values() {
            count += Self::count_terminals(child);
        }
        count
    }
}

impl Default for PrefixIndex {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_insert_and_suggest() {
        let mut idx = PrefixIndex::new();
        idx.insert("database", 100);
        idx.insert("datastore", 50);
        idx.insert("data", 200);
        idx.insert("delta", 10);

        let results = idx.suggest("data", 10);
        assert_eq!(results.len(), 3);
        // Sorted by frequency: data(200), database(100), datastore(50)
        assert_eq!(results[0].term, "data");
        assert_eq!(results[0].frequency, 200);
        assert_eq!(results[1].term, "database");
        assert_eq!(results[2].term, "datastore");
    }

    #[test]
    fn test_suggest_no_match() {
        let mut idx = PrefixIndex::new();
        idx.insert("hello", 10);

        let results = idx.suggest("xyz", 10);
        assert!(results.is_empty());
    }

    #[test]
    fn test_suggest_limit() {
        let mut idx = PrefixIndex::new();
        idx.insert("aa", 1);
        idx.insert("ab", 2);
        idx.insert("ac", 3);
        idx.insert("ad", 4);

        let results = idx.suggest("a", 2);
        assert_eq!(results.len(), 2);
        // Top 2 by frequency: ad(4), ac(3)
        assert_eq!(results[0].term, "ad");
        assert_eq!(results[1].term, "ac");
    }

    #[test]
    fn test_empty_prefix() {
        let mut idx = PrefixIndex::new();
        idx.insert("hello", 10);
        idx.insert("world", 20);

        let results = idx.suggest("", 10);
        assert_eq!(results.len(), 2);
    }

    #[test]
    fn test_exact_match() {
        let mut idx = PrefixIndex::new();
        idx.insert("test", 5);

        let results = idx.suggest("test", 10);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].term, "test");
    }

    #[test]
    fn test_term_count() {
        let mut idx = PrefixIndex::new();
        assert_eq!(idx.term_count(), 0);

        idx.insert("a", 1);
        idx.insert("ab", 1);
        idx.insert("abc", 1);
        assert_eq!(idx.term_count(), 3);
    }

    #[test]
    fn test_frequency_accumulation() {
        let mut idx = PrefixIndex::new();
        idx.insert("test", 10);
        idx.insert("test", 5);

        let results = idx.suggest("test", 10);
        assert_eq!(results[0].frequency, 15);
    }

    #[test]
    fn test_empty_index() {
        let idx = PrefixIndex::new();
        let results = idx.suggest("anything", 10);
        assert!(results.is_empty());
        assert_eq!(idx.term_count(), 0);
    }
}
