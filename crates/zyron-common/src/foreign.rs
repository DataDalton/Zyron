//! What one node asks another for when it reads a foreign table.
//!
//! The request lives here rather than beside either end of it. The planner
//! decides what to ask, the executor runs the ask, and the wire layer turns
//! it into a statement, and those three crates form a chain in that order,
//! so the shared shape has to sit under all of them.
//!
//! It is deliberately a description of a read and not a plan fragment: the
//! peer is another database, not another copy of this one, so what crosses
//! the boundary is a projection, a filter and a row cap, and nothing that
//! assumes the far side runs the same build.

use crate::types::TypeId;

/// One remote read, as the peer will be asked for it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ForeignRequest {
    /// Peer name, resolved to an address by whoever holds the registry
    pub peer: String,
    /// Table name on the peer
    pub table: String,
    /// Columns to fetch, in output order. Never empty: a scan that asked
    /// for nothing would still pay a round trip
    pub columns: Vec<String>,
    /// The type each column is expected to come back as, parallel to
    /// `columns`. The reader decodes with these rather than trusting what
    /// the peer says, so a column that changed type there surfaces as a
    /// decode failure instead of a silently wrong value
    pub column_types: Vec<TypeId>,
    /// Predicate rendered for the remote, None when none could be pushed
    pub predicate: Option<String>,
    /// Row cap when the query has one, so a LIMIT does not fetch a table
    pub limit: Option<usize>,
}

impl ForeignRequest {
    /// How much of the remote table this asks for, as a fraction, used to
    /// cost the scan before anything is fetched. A pushed predicate is
    /// worth more against a peer that can skip files with it than against
    /// one that has to walk an index, which is why the caller pairs this
    /// with the peer's mode rather than using it alone
    pub fn is_filtered(&self) -> bool {
        self.predicate.is_some()
    }
}
