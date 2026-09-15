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
    COMMIT_CHAINS_INTRODUCED_IN, ChangesetChunk, ChangesetHeader, ChangesetOp, ChangesetReader,
    ChangesetSink, FEED_IMAGES_INTRODUCED_IN, FLAG_ABORT, FLAG_BARRIER, FLAG_LAST,
    LAKE_FILES_INTRODUCED_IN, Origin, ReplicaIdentity, RowImage, SCHEDULE_RUNS_INTRODUCED_IN,
    StatementContext, TxnChangeset,
};
