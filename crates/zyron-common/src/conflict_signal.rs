//! Where a transaction conflict is reported, for whatever is counting them.
//!
//! Producing a `TransactionConflict` is not only an error, it is the event the
//! node's conflict signal is made of. Conflict is the one kind of saturation
//! that capacity makes worse, so a controller that cannot see the rate will
//! answer it by adding concurrency and make it worse still. That is why the
//! error constructor records it, and why a guard test keeps the struct-literal
//! form out of the tree: there is one way to produce one, and it counts.
//!
//! The counting lives in `zyron-pressure`, which sits above this crate. So
//! this is the seam: a slot something installs itself into, and a call that
//! does nothing when nothing has. A node whose controller has never been
//! touched records nothing, and nothing is reading it either.

use std::sync::OnceLock;

/// Something that counts conflict aborts.
pub trait ConflictSink: Send + Sync {
    fn record_conflict_abort(&self);
}

static SINK: OnceLock<&'static dyn ConflictSink> = OnceLock::new();

/// Installs the counter. Called once, by the crate that owns the signal.
///
/// A second call is ignored rather than replacing the first, because two
/// counters would mean the rate depends on which one a caller reached.
pub fn install_conflict_sink(sink: &'static dyn ConflictSink) {
    let _ = SINK.set(sink);
}

/// Reports one conflict abort, if anything is listening.
#[inline]
pub fn record_conflict_abort() {
    if let Some(sink) = SINK.get() {
        sink.record_conflict_abort();
    }
}

/// Whether a counter has been installed.
///
/// Read by the test that proves the seam is connected in a running node,
/// which would otherwise pass on a node where nothing counts anything.
pub fn has_conflict_sink() -> bool {
    SINK.get().is_some()
}
