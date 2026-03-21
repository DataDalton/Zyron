//! Wait-for graph based deadlock detection.
//!
//! Tracks which transactions are waiting on which other transactions.
//! When a cycle is detected in the wait-for graph, the youngest transaction
//! in the cycle is selected as the deadlock victim and aborted.
//!
//! Concurrency note: add_edge performs cycle detection and insertion
//! non-atomically. This is a deliberate trade-off: false negatives (missed
//! cycles) are acceptable because the next lock acquisition attempt will
//! re-check. The alternative (a global lock) would serialize all lock
//! acquisitions and defeat the purpose of concurrent transaction processing.

use std::collections::HashSet;

/// Maximum DFS traversal depth to prevent infinite loops on corrupted state.
const MAX_CYCLE_DEPTH: usize = 1000;

/// Wait-for graph for deadlock detection.
/// Each edge represents "txn A is waiting for txn B to release a lock".
pub struct WaitForGraph {
    /// Edges: waiter_txn_id -> holder_txn_id
    edges: scc::HashMap<u64, u64>,
}

impl WaitForGraph {
    /// Creates a new empty wait-for graph.
    pub fn new() -> Self {
        Self {
            edges: scc::HashMap::new(),
        }
    }

    /// Adds a wait-for edge: `waiter` is waiting for `holder` to release a lock.
    /// Returns Some(victim_txn_id) if adding this edge creates a cycle (deadlock).
    /// The victim is the youngest (highest txn_id) transaction in the cycle.
    /// Returns None if no cycle is created.
    pub fn add_edge(&self, waiter: u64, holder: u64) -> Option<u64> {
        if self.would_create_cycle(waiter, holder) {
            let cycle = self.collect_cycle(waiter, holder);
            return cycle.into_iter().max();
        }

        let _ = self.edges.insert_sync(waiter, holder);
        None
    }

    /// Removes an edge when a transaction commits, aborts, or acquires its lock.
    pub fn remove_edge(&self, waiter: u64) {
        let _ = self.edges.remove_sync(&waiter);
    }

    /// Removes all edges involving a transaction (as waiter or holder).
    /// Called when a transaction commits or aborts.
    pub fn remove_transaction(&self, txn_id: u64) {
        let _ = self.edges.remove_sync(&txn_id);

        let mut waiters_to_remove = Vec::new();
        self.edges.iter_sync(|&waiter, &holder| {
            if holder == txn_id {
                waiters_to_remove.push(waiter);
            }
            true
        });
        for waiter in waiters_to_remove {
            let _ = self.edges.remove_sync(&waiter);
        }
    }

    /// Returns the number of edges in the wait-for graph.
    pub fn edge_count(&self) -> usize {
        self.edges.len()
    }

    /// Checks if adding an edge from `waiter` to `holder` would create a cycle.
    /// Uses HashSet for O(1) visited checks and a depth limit for safety.
    fn would_create_cycle(&self, waiter: u64, holder: u64) -> bool {
        if waiter == holder {
            return true;
        }

        let mut current = holder;
        let mut visited = HashSet::with_capacity(8);
        visited.insert(waiter);

        for _ in 0..MAX_CYCLE_DEPTH {
            if visited.contains(&current) {
                return true;
            }
            visited.insert(current);

            match self.edges.read_sync(&current, |_, &next| next) {
                Some(next) => {
                    if next == waiter {
                        return true;
                    }
                    current = next;
                }
                None => return false,
            }
        }

        // Depth limit reached. Treat as potential cycle to be safe.
        true
    }

    /// Collects all transaction IDs in the cycle that would be created
    /// by adding waiter -> holder.
    fn collect_cycle(&self, waiter: u64, holder: u64) -> Vec<u64> {
        let mut cycle = vec![waiter, holder];
        let mut current = holder;

        for _ in 0..MAX_CYCLE_DEPTH {
            match self.edges.read_sync(&current, |_, &next| next) {
                Some(next) => {
                    if next == waiter {
                        break;
                    }
                    cycle.push(next);
                    current = next;
                }
                None => break,
            }
        }

        cycle
    }
}

impl Default for WaitForGraph {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_no_cycle() {
        let graph = WaitForGraph::new();
        assert!(graph.add_edge(1, 2).is_none());
        assert_eq!(graph.edge_count(), 1);
    }

    #[test]
    fn test_simple_cycle() {
        let graph = WaitForGraph::new();
        assert!(graph.add_edge(1, 2).is_none());
        let victim = graph.add_edge(2, 1);
        assert_eq!(victim, Some(2));
    }

    #[test]
    fn test_three_way_cycle() {
        let graph = WaitForGraph::new();
        assert!(graph.add_edge(1, 2).is_none());
        assert!(graph.add_edge(2, 3).is_none());
        let victim = graph.add_edge(3, 1);
        assert_eq!(victim, Some(3));
    }

    #[test]
    fn test_self_cycle() {
        let graph = WaitForGraph::new();
        let victim = graph.add_edge(1, 1);
        assert_eq!(victim, Some(1));
    }

    #[test]
    fn test_remove_edge() {
        let graph = WaitForGraph::new();
        graph.add_edge(1, 2);
        assert_eq!(graph.edge_count(), 1);
        graph.remove_edge(1);
        assert_eq!(graph.edge_count(), 0);
    }

    #[test]
    fn test_remove_transaction() {
        let graph = WaitForGraph::new();
        graph.add_edge(1, 2);
        graph.add_edge(3, 2);
        assert_eq!(graph.edge_count(), 2);
        graph.remove_transaction(2);
        assert_eq!(graph.edge_count(), 0);
    }

    #[test]
    fn test_chain_no_cycle() {
        let graph = WaitForGraph::new();
        graph.add_edge(1, 2);
        graph.add_edge(2, 3);
        assert!(graph.add_edge(3, 4).is_none());
        assert_eq!(graph.edge_count(), 3);
    }

    #[test]
    fn test_remove_breaks_cycle_potential() {
        let graph = WaitForGraph::new();
        graph.add_edge(1, 2);
        graph.add_edge(2, 3);
        graph.remove_edge(1);
        assert!(graph.add_edge(3, 1).is_none());
    }
}
