//! Graph search module for ZyronDB's property graph model.
//!
//! Provides graph schema definitions, CSR storage for cache-friendly traversal,
//! SIMD-accelerated graph algorithms (PageRank, BFS, shortest path, connected
//! components, community detection, betweenness centrality), Cypher-like pattern
//! query compilation, and schema lifecycle management.

pub mod algorithms;
pub mod manager;
pub mod query;
pub mod schema;
pub mod storage;

pub use manager::GraphManager;
pub use query::{
    CompiledGraphQuery, EdgeDirection, GraphPattern, GraphReturnItem, JoinCondition,
    PatternElement, TableScanRef, compile_pattern,
};
pub use schema::{EdgeId, EdgeLabel, GraphSchema, LabelId, NodeId, NodeLabel, PropertyDef};
pub use storage::{CompactGraph, CsrCache};
