//! Cross-object dependency graph for CASCADE/RESTRICT drop behavior.
//!
//! Tracks directed edges between database objects (functions, triggers,
//! views, pipelines, etc.) so that DROP operations can determine which
//! dependent objects must also be dropped (CASCADE) or block the drop
//! (RESTRICT).

use zyron_common::{Result, ZyronError};

/// The kind of database object participating in a dependency relationship.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DependencyKind {
    Function,
    Trigger,
    View,
    MaterializedView,
    Pipeline,
    Procedure,
    Aggregate,
    EventHandler,
    Table,
}

impl std::fmt::Display for DependencyKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let label = match self {
            DependencyKind::Function => "Function",
            DependencyKind::Trigger => "Trigger",
            DependencyKind::View => "View",
            DependencyKind::MaterializedView => "MaterializedView",
            DependencyKind::Pipeline => "Pipeline",
            DependencyKind::Procedure => "Procedure",
            DependencyKind::Aggregate => "Aggregate",
            DependencyKind::EventHandler => "EventHandler",
            DependencyKind::Table => "Table",
        };
        write!(f, "{}", label)
    }
}

/// A directed edge from a source object to a target object it depends on.
/// For example, a trigger (source) depends on a function (target) that
/// it executes. Dropping the function requires dropping the trigger first.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DependencyEdge {
    pub sourceKind: DependencyKind,
    pub sourceName: String,
    pub targetKind: DependencyKind,
    pub targetName: String,
}

/// Builds the lookup key for the dependency maps.
fn makeKey(kind: DependencyKind, name: &str) -> String {
    format!("{}:{}", kind, name)
}

/// In-memory dependency graph tracking relationships between database
/// objects. Used to enforce RESTRICT semantics and compute CASCADE
/// drop ordering.
pub struct DependencyTracker {
    /// Maps each object to the list of edges where it is the source
    /// (the object that depends on something else).
    outgoing: scc::HashMap<String, Vec<DependencyEdge>>,
    /// Maps each object to the list of edges where it is the target
    /// (the object that is depended upon).
    incoming: scc::HashMap<String, Vec<DependencyEdge>>,
}

impl DependencyTracker {
    /// Creates an empty dependency tracker.
    pub fn new() -> Self {
        Self {
            outgoing: scc::HashMap::new(),
            incoming: scc::HashMap::new(),
        }
    }

    /// Records a dependency: source depends on target. For example,
    /// trigger "audit_insert" depends on function "log_change".
    pub fn addDependency(
        &self,
        sourceKind: DependencyKind,
        sourceName: &str,
        targetKind: DependencyKind,
        targetName: &str,
    ) -> Result<()> {
        let edge = DependencyEdge {
            sourceKind,
            sourceName: sourceName.to_string(),
            targetKind,
            targetName: targetName.to_string(),
        };

        let sourceKey = makeKey(sourceKind, sourceName);
        let targetKey = makeKey(targetKind, targetName);

        // Add to outgoing edges for the source object.
        let edgeClone = edge.clone();
        let outEntry = self.outgoing.entry_sync(sourceKey);
        match outEntry {
            scc::hash_map::Entry::Occupied(mut occ) => {
                occ.get_mut().push(edgeClone);
            }
            scc::hash_map::Entry::Vacant(vac) => {
                vac.insert_entry(vec![edgeClone]);
            }
        }

        // Add to incoming edges for the target object.
        let inEntry = self.incoming.entry_sync(targetKey);
        match inEntry {
            scc::hash_map::Entry::Occupied(mut occ) => {
                occ.get_mut().push(edge);
            }
            scc::hash_map::Entry::Vacant(vac) => {
                vac.insert_entry(vec![edge]);
            }
        }

        Ok(())
    }

    /// Removes all dependency edges involving the given object,
    /// both as a source and as a target.
    pub fn removeDependenciesFor(&self, kind: DependencyKind, name: &str) -> Result<()> {
        let key = makeKey(kind, name);

        // Collect outgoing edges so we can clean up the corresponding
        // incoming entries in other objects.
        let outgoingEdges: Vec<DependencyEdge> = self
            .outgoing
            .read_sync(&key, |_k, v| v.clone())
            .unwrap_or_default();

        // For each outgoing edge, remove the matching incoming entry
        // from the target.
        for edge in &outgoingEdges {
            let targetKey = makeKey(edge.targetKind, &edge.targetName);
            let entry = self.incoming.entry_sync(targetKey);
            if let scc::hash_map::Entry::Occupied(mut occ) = entry {
                occ.get_mut()
                    .retain(|e| !(e.sourceKind == kind && e.sourceName == name));
            }
        }

        // Collect incoming edges so we can clean up the corresponding
        // outgoing entries in other objects.
        let incomingEdges: Vec<DependencyEdge> = self
            .incoming
            .read_sync(&key, |_k, v| v.clone())
            .unwrap_or_default();

        // For each incoming edge, remove the matching outgoing entry
        // from the source.
        for edge in &incomingEdges {
            let sourceKey = makeKey(edge.sourceKind, &edge.sourceName);
            let entry = self.outgoing.entry_sync(sourceKey);
            if let scc::hash_map::Entry::Occupied(mut occ) = entry {
                occ.get_mut()
                    .retain(|e| !(e.targetKind == kind && e.targetName == name));
            }
        }

        // Remove the object's own entries.
        let _ = self.outgoing.remove_sync(&key);
        let _ = self.incoming.remove_sync(&key);

        Ok(())
    }

    /// Checks whether the given object can be dropped.
    ///
    /// If cascade is false and dependents exist, returns a
    /// DependencyViolation error listing the dependent objects.
    ///
    /// If cascade is true, returns the full list of dependency edges
    /// from objects that depend on this one (directly or transitively),
    /// collected in reverse topological order for safe drop sequencing.
    pub fn checkDropAllowed(
        &self,
        kind: DependencyKind,
        name: &str,
        cascade: bool,
    ) -> Result<Vec<DependencyEdge>> {
        let directDependents = self.getDependents(kind, name);

        if directDependents.is_empty() {
            return Ok(Vec::new());
        }

        if !cascade {
            let dependentNames: Vec<String> = directDependents
                .iter()
                .map(|e| format!("{} {}", e.sourceKind, e.sourceName))
                .collect();
            return Err(ZyronError::DependencyViolation {
                object: format!("{} {}", kind, name),
                dependents: dependentNames.join(", "),
            });
        }

        // Cascade: collect all transitive dependents via BFS.
        let mut allEdges = Vec::new();
        let mut visited = hashbrown::HashSet::new();
        let mut queue = std::collections::VecDeque::new();

        // Seed the queue with direct dependents.
        for edge in &directDependents {
            let depKey = makeKey(edge.sourceKind, &edge.sourceName);
            if visited.insert(depKey.clone()) {
                queue.push_back((edge.sourceKind, edge.sourceName.clone()));
                allEdges.push(edge.clone());
            }
        }

        // BFS to collect transitive dependents.
        while let Some((depKind, depName)) = queue.pop_front() {
            let transitiveDeps = self.getDependents(depKind, &depName);
            for edge in transitiveDeps {
                let depKey = makeKey(edge.sourceKind, &edge.sourceName);
                if visited.insert(depKey.clone()) {
                    queue.push_back((edge.sourceKind, edge.sourceName.clone()));
                    allEdges.push(edge);
                }
            }
        }

        // Reverse so that leaf dependents come first (safe drop order).
        allEdges.reverse();
        Ok(allEdges)
    }

    /// Returns edges from objects that directly depend on the given
    /// object. These are the incoming edges for the target.
    pub fn getDependents(&self, kind: DependencyKind, name: &str) -> Vec<DependencyEdge> {
        let key = makeKey(kind, name);
        self.incoming
            .read_sync(&key, |_k, v| v.clone())
            .unwrap_or_default()
    }

    /// Returns edges to objects that the given object depends on.
    /// These are the outgoing edges from the source.
    pub fn getDependencies(&self, kind: DependencyKind, name: &str) -> Vec<DependencyEdge> {
        let key = makeKey(kind, name);
        self.outgoing
            .read_sync(&key, |_k, v| v.clone())
            .unwrap_or_default()
    }
}

impl Default for DependencyTracker {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_add_and_get_dependencies() {
        let tracker = DependencyTracker::new();
        tracker
            .addDependency(
                DependencyKind::Trigger,
                "audit_insert",
                DependencyKind::Function,
                "log_change",
            )
            .expect("add dependency");

        let deps = tracker.getDependencies(DependencyKind::Trigger, "audit_insert");
        assert_eq!(deps.len(), 1);
        assert_eq!(deps[0].targetKind, DependencyKind::Function);
        assert_eq!(deps[0].targetName, "log_change");
    }

    #[test]
    fn test_get_dependents() {
        let tracker = DependencyTracker::new();
        tracker
            .addDependency(
                DependencyKind::Trigger,
                "audit_insert",
                DependencyKind::Function,
                "log_change",
            )
            .expect("add dependency");

        let dependents = tracker.getDependents(DependencyKind::Function, "log_change");
        assert_eq!(dependents.len(), 1);
        assert_eq!(dependents[0].sourceKind, DependencyKind::Trigger);
        assert_eq!(dependents[0].sourceName, "audit_insert");
    }

    #[test]
    fn test_check_drop_restrict_blocks() {
        let tracker = DependencyTracker::new();
        tracker
            .addDependency(
                DependencyKind::Trigger,
                "trg1",
                DependencyKind::Function,
                "fn1",
            )
            .expect("add");

        let result = tracker.checkDropAllowed(DependencyKind::Function, "fn1", false);
        assert!(result.is_err());
        let err = result.expect_err("should fail");
        assert!(matches!(err, ZyronError::DependencyViolation { .. }));
    }

    #[test]
    fn test_check_drop_restrict_allows_no_dependents() {
        let tracker = DependencyTracker::new();
        let result = tracker.checkDropAllowed(DependencyKind::Function, "standalone", false);
        assert!(result.is_ok());
        assert!(result.expect("should be ok").is_empty());
    }

    #[test]
    fn test_check_drop_cascade_returns_edges() {
        let tracker = DependencyTracker::new();
        tracker
            .addDependency(
                DependencyKind::Trigger,
                "trg1",
                DependencyKind::Function,
                "fn1",
            )
            .expect("add");
        tracker
            .addDependency(
                DependencyKind::Pipeline,
                "pipe1",
                DependencyKind::Function,
                "fn1",
            )
            .expect("add");

        let edges = tracker
            .checkDropAllowed(DependencyKind::Function, "fn1", true)
            .expect("cascade");
        assert_eq!(edges.len(), 2);
    }

    #[test]
    fn test_transitive_cascade() {
        let tracker = DependencyTracker::new();
        // mv1 depends on table1
        tracker
            .addDependency(
                DependencyKind::MaterializedView,
                "mv1",
                DependencyKind::Table,
                "table1",
            )
            .expect("add");
        // pipeline1 depends on mv1
        tracker
            .addDependency(
                DependencyKind::Pipeline,
                "pipeline1",
                DependencyKind::MaterializedView,
                "mv1",
            )
            .expect("add");

        let edges = tracker
            .checkDropAllowed(DependencyKind::Table, "table1", true)
            .expect("cascade");
        // Should include both mv1 (direct) and pipeline1 (transitive).
        assert_eq!(edges.len(), 2);

        // The first edge in the reversed list should be the leaf
        // (pipeline1) for safe drop ordering.
        assert_eq!(edges[0].sourceName, "pipeline1");
        assert_eq!(edges[1].sourceName, "mv1");
    }

    #[test]
    fn test_remove_dependencies_for() {
        let tracker = DependencyTracker::new();
        tracker
            .addDependency(
                DependencyKind::Trigger,
                "trg1",
                DependencyKind::Function,
                "fn1",
            )
            .expect("add");
        tracker
            .addDependency(
                DependencyKind::Trigger,
                "trg1",
                DependencyKind::Table,
                "users",
            )
            .expect("add");

        tracker
            .removeDependenciesFor(DependencyKind::Trigger, "trg1")
            .expect("remove");

        // trg1 should have no outgoing edges.
        let deps = tracker.getDependencies(DependencyKind::Trigger, "trg1");
        assert!(deps.is_empty());

        // fn1 should have no incoming edges from trg1.
        let dependents = tracker.getDependents(DependencyKind::Function, "fn1");
        assert!(dependents.is_empty());

        // users should have no incoming edges from trg1.
        let dependents = tracker.getDependents(DependencyKind::Table, "users");
        assert!(dependents.is_empty());
    }

    #[test]
    fn test_multiple_sources_same_target() {
        let tracker = DependencyTracker::new();
        tracker
            .addDependency(
                DependencyKind::Trigger,
                "trg1",
                DependencyKind::Function,
                "shared_fn",
            )
            .expect("add");
        tracker
            .addDependency(
                DependencyKind::Trigger,
                "trg2",
                DependencyKind::Function,
                "shared_fn",
            )
            .expect("add");

        let dependents = tracker.getDependents(DependencyKind::Function, "shared_fn");
        assert_eq!(dependents.len(), 2);
    }

    #[test]
    fn test_remove_preserves_other_objects() {
        let tracker = DependencyTracker::new();
        tracker
            .addDependency(
                DependencyKind::Trigger,
                "trg1",
                DependencyKind::Function,
                "fn1",
            )
            .expect("add");
        tracker
            .addDependency(
                DependencyKind::Trigger,
                "trg2",
                DependencyKind::Function,
                "fn1",
            )
            .expect("add");

        tracker
            .removeDependenciesFor(DependencyKind::Trigger, "trg1")
            .expect("remove");

        // trg2's dependency on fn1 should remain.
        let dependents = tracker.getDependents(DependencyKind::Function, "fn1");
        assert_eq!(dependents.len(), 1);
        assert_eq!(dependents[0].sourceName, "trg2");
    }

    #[test]
    fn test_dependency_kind_display() {
        assert_eq!(DependencyKind::Function.to_string(), "Function");
        assert_eq!(
            DependencyKind::MaterializedView.to_string(),
            "MaterializedView"
        );
        assert_eq!(DependencyKind::Table.to_string(), "Table");
    }

    #[test]
    fn test_empty_tracker_queries() {
        let tracker = DependencyTracker::new();
        assert!(
            tracker
                .getDependents(DependencyKind::Function, "nothing")
                .is_empty()
        );
        assert!(
            tracker
                .getDependencies(DependencyKind::Table, "nothing")
                .is_empty()
        );
    }
}
