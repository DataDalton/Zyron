//! Turning a transaction's effects into something another node can replay.
//!
//! Capture sits in the DML operators, so anything that writes rows is caught
//! once, wherever it came from: a statement, a procedure body, a trigger, a
//! `MERGE`, a `COPY`. Apply sits behind the consensus applier and puts the
//! same rows back through the storage paths that produced them, with the
//! decisions the leader already made left alone.

pub mod apply;
pub mod capture;
pub mod changeset;

pub use capture::{
    IdentityImages, capture_delete, capture_insert, capture_lake_version, capture_sequence,
    capture_update, changeset, identity_images, identity_of,
};
pub use changeset::{
    ChangesetChunk, ChangesetHeader, ChangesetOp, ChangesetReader, ChangesetSink, FLAG_ABORT,
    FLAG_BARRIER, FLAG_LAST, Origin, ReplicaIdentity, RowImage, StatementContext, TxnChangeset,
};
