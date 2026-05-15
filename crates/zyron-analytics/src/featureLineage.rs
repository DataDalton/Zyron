#![allow(non_snake_case)]
// Feature lineage, tracks the source tables and columns each feature
// reads from, plus the dependency graph between features within a group
// Lineage is computed at registration time by inspecting the feature's
// source SQL through a lightweight table/column extractor

use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LineageEntry {
    pub featureName: String,
    pub sourceTables: Vec<String>,
    pub sourceColumns: Vec<(String, String)>,
    pub transformChain: Vec<String>,
    pub dependencies: Vec<String>,
    pub lastComputedMs: i64,
}

impl LineageEntry {
    pub fn new(featureName: String) -> Self {
        Self {
            featureName,
            sourceTables: Vec::new(),
            sourceColumns: Vec::new(),
            transformChain: Vec::new(),
            dependencies: Vec::new(),
            lastComputedMs: 0,
        }
    }
}

/// Lineage metadata indexed by qualified feature name (group.feature)
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct FeatureLineageRegistry {
    pub entries: HashMap<String, LineageEntry>,
    pub tableMTimeMs: HashMap<String, i64>,
}

impl FeatureLineageRegistry {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn register(&mut self, qualifiedName: String, entry: LineageEntry) {
        self.entries.insert(qualifiedName, entry);
    }

    pub fn get(&self, qualifiedName: &str) -> Option<&LineageEntry> {
        self.entries.get(qualifiedName)
    }

    pub fn touchTable(&mut self, table: &str, atMs: i64) {
        let cur = self.tableMTimeMs.entry(table.to_string()).or_insert(0);
        if atMs > *cur {
            *cur = atMs;
        }
    }

    /// A feature is stale if any of its source tables changed after the
    /// lineage entry's lastComputedMs
    pub fn isStale(&self, qualifiedName: &str) -> bool {
        let entry = match self.get(qualifiedName) {
            Some(e) => e,
            None => return true,
        };
        for table in &entry.sourceTables {
            if let Some(t) = self.tableMTimeMs.get(table) {
                if *t > entry.lastComputedMs {
                    return true;
                }
            }
        }
        false
    }

    /// Returns features in topological order so refreshes process
    /// dependencies before dependents
    pub fn topologicalOrder(&self) -> Vec<String> {
        let mut visited: HashSet<String> = HashSet::new();
        let mut order: Vec<String> = Vec::new();
        for name in self.entries.keys() {
            self.visit(name, &mut visited, &mut order);
        }
        order
    }

    fn visit(&self, name: &str, visited: &mut HashSet<String>, order: &mut Vec<String>) {
        if visited.contains(name) {
            return;
        }
        visited.insert(name.to_string());
        if let Some(entry) = self.entries.get(name) {
            for dep in &entry.dependencies {
                self.visit(dep, visited, order);
            }
        }
        order.push(name.to_string());
    }
}

/// Light SQL token-level extractor for table and column references
/// Looks for FROM <ident>, JOIN <ident>, and SELECT <ident> patterns
/// Conservative, returns a superset of true references which is fine for
/// staleness checks
pub fn extractTablesAndColumns(sql: &str) -> (Vec<String>, Vec<(String, String)>) {
    let lower = sql.to_ascii_lowercase();
    let mut tokens: Vec<String> = Vec::new();
    let mut current = String::new();
    for c in lower.chars() {
        if c.is_alphanumeric() || c == '_' || c == '.' {
            current.push(c);
        } else if !current.is_empty() {
            tokens.push(std::mem::take(&mut current));
        }
    }
    if !current.is_empty() {
        tokens.push(current);
    }
    let mut tables: Vec<String> = Vec::new();
    let mut tableSet: HashSet<String> = HashSet::new();
    let mut columns: Vec<(String, String)> = Vec::new();
    let mut i = 0usize;
    while i + 1 < tokens.len() {
        let t = tokens[i].as_str();
        if t == "from" || t == "join" || t == "into" || t == "update" {
            let next = &tokens[i + 1];
            if !isKeyword(next) {
                let tableName = next.split('.').last().unwrap_or(next).to_string();
                if !tableSet.contains(&tableName) {
                    tableSet.insert(tableName.clone());
                    tables.push(tableName);
                }
            }
        }
        i += 1;
    }
    // Collect dotted column references like t.col
    for t in &tokens {
        if let Some(dot) = t.find('.') {
            let table = &t[..dot];
            let column = &t[dot + 1..];
            if !table.is_empty()
                && !column.is_empty()
                && !column.chars().all(|c| c.is_ascii_digit())
            {
                columns.push((table.to_string(), column.to_string()));
            }
        }
    }
    (tables, columns)
}

fn isKeyword(tok: &str) -> bool {
    matches!(
        tok,
        "select"
            | "from"
            | "where"
            | "group"
            | "by"
            | "order"
            | "having"
            | "join"
            | "inner"
            | "left"
            | "right"
            | "full"
            | "outer"
            | "cross"
            | "on"
            | "as"
            | "and"
            | "or"
            | "not"
            | "in"
            | "is"
            | "null"
            | "case"
            | "when"
            | "then"
            | "else"
            | "end"
            | "limit"
            | "offset"
            | "with"
            | "union"
            | "all"
            | "distinct"
            | "into"
            | "values"
            | "update"
            | "set"
            | "delete"
            | "between"
            | "like"
            | "exists"
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn extractsTablesFromSelect() {
        let sql = "SELECT user_id, SUM(amount) FROM orders WHERE user_id = entity.user_id GROUP BY user_id";
        let (tables, _cols) = extractTablesAndColumns(sql);
        assert!(tables.contains(&"orders".to_string()), "{:?}", tables);
    }

    #[test]
    fn extractsJoinedTables() {
        let sql = "SELECT a.x FROM orders a JOIN users b ON a.user_id = b.id";
        let (tables, _cols) = extractTablesAndColumns(sql);
        assert!(tables.contains(&"orders".to_string()));
        assert!(tables.contains(&"users".to_string()));
    }

    #[test]
    fn stalenessAfterTableChange() {
        let mut reg = FeatureLineageRegistry::new();
        let mut e = LineageEntry::new("user_features.total_purchases".into());
        e.sourceTables.push("orders".into());
        e.lastComputedMs = 100;
        reg.register("user_features.total_purchases".into(), e);
        assert!(!reg.isStale("user_features.total_purchases"));
        reg.touchTable("orders", 200);
        assert!(reg.isStale("user_features.total_purchases"));
    }

    #[test]
    fn topologicalRespectsDeps() {
        let mut reg = FeatureLineageRegistry::new();
        let mut a = LineageEntry::new("g.a".into());
        let mut b = LineageEntry::new("g.b".into());
        b.dependencies.push("g.a".into());
        let mut c = LineageEntry::new("g.c".into());
        c.dependencies.push("g.b".into());
        reg.register("g.c".into(), c);
        reg.register("g.b".into(), b);
        reg.register("g.a".into(), a);
        let order = reg.topologicalOrder();
        let posA = order.iter().position(|x| x == "g.a").unwrap();
        let posB = order.iter().position(|x| x == "g.b").unwrap();
        let posC = order.iter().position(|x| x == "g.c").unwrap();
        assert!(posA < posB);
        assert!(posB < posC);
    }
}
