//! Tree and hierarchy operations.
//!
//! Closure table, materialized path, and nested set representations.
//! Closure table enables O(1) ancestry queries at the cost of write amplification.

use zyron_common::{Result, ZyronError};

// ---------------------------------------------------------------------------
// Closure table
// ---------------------------------------------------------------------------

/// Computes the new closure table rows to insert when adding a child under a parent.
/// Given the existing (ancestor_id, descendant_id, depth) rows, returns new rows:
/// for each ancestor of parent_id (including parent itself), add a row pointing to child_id.
/// Plus a self-reference for child_id.
pub fn closure_table_insert(
    existing_closure: &[(i64, i64, i32)],
    parent_id: i64,
    child_id: i64,
) -> Vec<(i64, i64, i32)> {
    let mut new_rows = Vec::new();
    // Self-reference for the new child
    new_rows.push((child_id, child_id, 0));
    // For each ancestor of parent_id (including parent itself), add link to child
    for &(ancestor, descendant, depth) in existing_closure {
        if descendant == parent_id {
            new_rows.push((ancestor, child_id, depth + 1));
        }
    }
    new_rows
}

/// Returns all ancestors of the given node (including itself).
pub fn closure_table_ancestors(closure: &[(i64, i64, i32)], node_id: i64) -> Vec<i64> {
    let mut result: Vec<(i64, i32)> = closure
        .iter()
        .filter(|&&(_, desc, _)| desc == node_id)
        .map(|&(anc, _, depth)| (anc, depth))
        .collect();
    // Sort by depth descending (root first)
    result.sort_by(|a, b| b.1.cmp(&a.1));
    result.into_iter().map(|(anc, _)| anc).collect()
}

/// Returns all descendants of the given node (including itself).
pub fn closure_table_descendants(closure: &[(i64, i64, i32)], node_id: i64) -> Vec<i64> {
    let mut result: Vec<(i64, i32)> = closure
        .iter()
        .filter(|&&(anc, _, _)| anc == node_id)
        .map(|&(_, desc, depth)| (desc, depth))
        .collect();
    result.sort_by_key(|&(_, depth)| depth);
    result.into_iter().map(|(desc, _)| desc).collect()
}

/// Returns the depth of the node from the root (0 if root, 1 if child of root, ...).
pub fn closure_table_depth(closure: &[(i64, i64, i32)], node_id: i64) -> i32 {
    // Depth = max depth from any ancestor
    closure
        .iter()
        .filter(|&&(_, desc, _)| desc == node_id)
        .map(|&(_, _, depth)| depth)
        .max()
        .unwrap_or(0)
}

// ---------------------------------------------------------------------------
// Materialized path
// ---------------------------------------------------------------------------

/// Builds a materialized path from path segments: "/root/parent/child".
pub fn materialized_path(segments: &[&str]) -> String {
    if segments.is_empty() {
        return "/".to_string();
    }
    let mut result = String::new();
    for seg in segments {
        result.push('/');
        result.push_str(seg);
    }
    result
}

/// Returns all ancestor paths of the given path (from root to parent).
/// "/a/b/c" -> ["/", "/a", "/a/b"]
pub fn path_ancestors(path: &str) -> Vec<String> {
    let trimmed = path.strip_prefix('/').unwrap_or(path);
    if trimmed.is_empty() {
        return vec![];
    }
    let segments: Vec<&str> = trimmed.split('/').filter(|s| !s.is_empty()).collect();

    let mut result = vec!["/".to_string()];
    let mut current = String::new();
    for seg in &segments[..segments.len().saturating_sub(1)] {
        current.push('/');
        current.push_str(seg);
        result.push(current.clone());
    }
    result
}

/// Returns the depth of a path (number of segments).
pub fn path_depth(path: &str) -> i32 {
    let trimmed = path.strip_prefix('/').unwrap_or(path);
    if trimmed.is_empty() {
        return 0;
    }
    trimmed.split('/').filter(|s| !s.is_empty()).count() as i32
}

/// Returns true if ancestor_path is an ancestor of descendant_path.
pub fn is_ancestor(ancestor_path: &str, descendant_path: &str) -> bool {
    if ancestor_path == descendant_path {
        return false;
    }
    if ancestor_path == "/" {
        return descendant_path.starts_with('/') && descendant_path != "/";
    }
    // descendant_path must start with ancestor_path followed by '/'
    let prefix = if ancestor_path.ends_with('/') {
        ancestor_path.to_string()
    } else {
        format!("{}/", ancestor_path)
    };
    descendant_path.starts_with(&prefix)
}

// ---------------------------------------------------------------------------
// Nested set model
// ---------------------------------------------------------------------------

/// Inserts a new node under the given parent, updating lft/rgt values of existing nodes.
/// nodes: mutable slice of (node_id, lft, rgt).
pub fn nested_set_insert(
    nodes: &mut Vec<(i64, i32, i32)>,
    parent_id: i64,
    node_id: i64,
) -> Result<()> {
    // Find parent's rgt value
    let parent_rgt = nodes
        .iter()
        .find(|&&(id, _, _)| id == parent_id)
        .map(|&(_, _, rgt)| rgt)
        .ok_or_else(|| ZyronError::ExecutionError(format!("Parent {} not found", parent_id)))?;

    // Shift all lft/rgt values >= parent's rgt by +2
    for (_, lft, rgt) in nodes.iter_mut() {
        if *lft >= parent_rgt {
            *lft += 2;
        }
        if *rgt >= parent_rgt {
            *rgt += 2;
        }
    }

    // Insert new node
    nodes.push((node_id, parent_rgt, parent_rgt + 1));
    Ok(())
}

/// Returns all descendants (including the node itself) using lft/rgt ranges.
pub fn nested_set_subtree(nodes: &[(i64, i32, i32)], node_id: i64) -> Vec<i64> {
    let target = nodes.iter().find(|&&(id, _, _)| id == node_id);
    let (_, tlft, trgt) = match target {
        Some(&t) => t,
        None => return Vec::new(),
    };

    nodes
        .iter()
        .filter(|&&(_, lft, rgt)| lft >= tlft && rgt <= trgt)
        .map(|&(id, _, _)| id)
        .collect()
}

/// Rebuilds lft/rgt values from a list of (node_id, parent_id) relations.
/// parent_id of None means root node.
pub fn nested_set_rebuild(parent_child: &[(i64, Option<i64>)]) -> Vec<(i64, i32, i32)> {
    let mut result = Vec::new();
    let mut counter = 1i32;

    // Find roots
    let roots: Vec<i64> = parent_child
        .iter()
        .filter(|(_, parent)| parent.is_none())
        .map(|(id, _)| *id)
        .collect();

    for root in roots {
        traverse(root, parent_child, &mut counter, &mut result);
    }

    result
}

fn traverse(
    node_id: i64,
    parent_child: &[(i64, Option<i64>)],
    counter: &mut i32,
    result: &mut Vec<(i64, i32, i32)>,
) {
    let lft = *counter;
    *counter += 1;

    let children: Vec<i64> = parent_child
        .iter()
        .filter(|(_, parent)| *parent == Some(node_id))
        .map(|(id, _)| *id)
        .collect();

    for child in children {
        traverse(child, parent_child, counter, result);
    }

    let rgt = *counter;
    *counter += 1;

    result.push((node_id, lft, rgt));
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_closure_table_insert_root() {
        // Insert root node (1)
        let new_rows = closure_table_insert(&[], 0, 1);
        assert_eq!(new_rows, vec![(1, 1, 0)]);
    }

    #[test]
    fn test_closure_table_insert_child() {
        // Existing: (1, 1, 0)
        let existing = vec![(1, 1, 0)];
        let new_rows = closure_table_insert(&existing, 1, 2);
        assert!(new_rows.contains(&(2, 2, 0))); // self-ref
        assert!(new_rows.contains(&(1, 2, 1))); // 1 is ancestor of 2
    }

    #[test]
    fn test_closure_table_ancestors() {
        let closure = vec![
            (1, 1, 0),
            (1, 2, 1),
            (2, 2, 0),
            (1, 3, 2),
            (2, 3, 1),
            (3, 3, 0),
        ];
        let ancestors = closure_table_ancestors(&closure, 3);
        // Ancestors of 3: 1 (depth 2), 2 (depth 1), 3 (depth 0)
        assert_eq!(ancestors, vec![1, 2, 3]);
    }

    #[test]
    fn test_closure_table_descendants() {
        let closure = vec![
            (1, 1, 0),
            (1, 2, 1),
            (1, 3, 2),
            (2, 2, 0),
            (2, 3, 1),
            (3, 3, 0),
        ];
        let descendants = closure_table_descendants(&closure, 1);
        assert_eq!(descendants, vec![1, 2, 3]);
    }

    #[test]
    fn test_closure_table_depth() {
        let closure = vec![(1, 1, 0), (1, 2, 1), (2, 2, 0)];
        assert_eq!(closure_table_depth(&closure, 1), 0);
        assert_eq!(closure_table_depth(&closure, 2), 1);
    }

    #[test]
    fn test_materialized_path() {
        let p = materialized_path(&["root", "parent", "child"]);
        assert_eq!(p, "/root/parent/child");
    }

    #[test]
    fn test_materialized_path_empty() {
        assert_eq!(materialized_path(&[]), "/");
    }

    #[test]
    fn test_path_ancestors() {
        let ancestors = path_ancestors("/a/b/c");
        assert_eq!(ancestors, vec!["/", "/a", "/a/b"]);
    }

    #[test]
    fn test_path_ancestors_root_child() {
        let ancestors = path_ancestors("/a");
        assert_eq!(ancestors, vec!["/"]);
    }

    #[test]
    fn test_path_ancestors_root() {
        let ancestors = path_ancestors("/");
        assert_eq!(ancestors, Vec::<String>::new());
    }

    #[test]
    fn test_path_depth() {
        assert_eq!(path_depth("/"), 0);
        assert_eq!(path_depth("/a"), 1);
        assert_eq!(path_depth("/a/b/c"), 3);
    }

    #[test]
    fn test_is_ancestor() {
        assert!(is_ancestor("/a", "/a/b"));
        assert!(is_ancestor("/a", "/a/b/c"));
        assert!(is_ancestor("/", "/a"));
        assert!(!is_ancestor("/a", "/a")); // not self
        assert!(!is_ancestor("/a", "/b"));
        assert!(!is_ancestor("/a/b", "/a"));
    }

    #[test]
    fn test_nested_set_insert_first() {
        let mut nodes: Vec<(i64, i32, i32)> = vec![(1, 1, 2)]; // just root
        nested_set_insert(&mut nodes, 1, 2).unwrap();
        // Root was (1, 1, 2). Child (2) gets (2, 2, 3). Root becomes (1, 1, 4).
        assert_eq!(nodes.len(), 2);
        assert!(nodes.contains(&(1, 1, 4)));
        assert!(nodes.contains(&(2, 2, 3)));
    }

    #[test]
    fn test_nested_set_subtree() {
        let nodes = vec![
            (1, 1, 6), // root
            (2, 2, 3), // child of root
            (3, 4, 5), // another child of root
        ];
        let subtree = nested_set_subtree(&nodes, 1);
        assert_eq!(subtree.len(), 3);
    }

    #[test]
    fn test_nested_set_subtree_leaf() {
        let nodes = vec![(1, 1, 4), (2, 2, 3)];
        let subtree = nested_set_subtree(&nodes, 2);
        assert_eq!(subtree, vec![2]);
    }

    #[test]
    fn test_nested_set_rebuild() {
        // Hierarchy: 1 is root, 2 is child of 1, 3 is child of 1
        let relations = vec![(1, None), (2, Some(1)), (3, Some(1))];
        let result = nested_set_rebuild(&relations);
        assert_eq!(result.len(), 3);
        // Root has lft=1, rgt=6 (spans entire tree)
        let root = result.iter().find(|(id, _, _)| *id == 1).unwrap();
        assert_eq!(root.1, 1);
        assert_eq!(root.2, 6);
    }

    #[test]
    fn test_closure_insert_nonexistent_parent() {
        let existing = vec![(1, 1, 0)];
        // Insert with parent 99 (doesn't exist) - should just add self-reference
        let new_rows = closure_table_insert(&existing, 99, 2);
        assert_eq!(new_rows, vec![(2, 2, 0)]);
    }

    #[test]
    fn test_nested_set_insert_nonexistent_parent() {
        let mut nodes: Vec<(i64, i32, i32)> = vec![(1, 1, 2)];
        assert!(nested_set_insert(&mut nodes, 99, 2).is_err());
    }
}
