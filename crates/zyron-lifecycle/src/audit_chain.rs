//! In-memory tamper-evident audit chain helper. The durable chain lives in
//! the catalog `compliance_log` system table (see Catalog::append_compliance_log
//! / verify_compliance_chain); this type tracks the latest hash so callers can
//! build the next entry without an extra catalog read on the hot path, and
//! verifies a loaded chain.

use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};

use zyron_catalog::schema::ComplianceLogEntry;

pub struct AuditChain {
    last_hash: AtomicU32,
    last_id: AtomicU64,
}

impl AuditChain {
    pub fn new() -> Self {
        Self {
            last_hash: AtomicU32::new(0),
            last_id: AtomicU64::new(0),
        }
    }

    /// Seeds the in-memory tip from the persisted log (called on startup).
    pub fn seed(&self, log: &[ComplianceLogEntry]) {
        if let Some(last) = log.last() {
            self.last_hash.store(last.entry_hash, Ordering::Release);
            self.last_id.store(last.event_id, Ordering::Release);
        }
    }

    /// Builds the next chained entry, advancing the in-memory tip. The caller
    /// persists the returned entry via the catalog.
    pub fn next_entry(
        &self,
        event_type: u8,
        subject: String,
        table_id: u32,
        ts: i64,
        detail: String,
    ) -> ComplianceLogEntry {
        let prev_hash = self.last_hash.load(Ordering::Acquire);
        let event_id = self.last_id.fetch_add(1, Ordering::AcqRel) + 1;
        let mut e = ComplianceLogEntry {
            event_id,
            event_type,
            subject,
            table_id,
            ts,
            detail,
            prev_hash,
            entry_hash: 0,
        };
        e.entry_hash = e.compute_hash();
        self.last_hash.store(e.entry_hash, Ordering::Release);
        e
    }

    /// Verifies a loaded chain. Returns (verified_count, intact).
    pub fn verify(log: &[ComplianceLogEntry]) -> (usize, bool) {
        let mut prev = 0u32;
        let mut verified = 0usize;
        for e in log {
            if e.prev_hash != prev || e.entry_hash != e.compute_hash() {
                return (verified, false);
            }
            prev = e.entry_hash;
            verified += 1;
        }
        (verified, true)
    }
}

impl Default for AuditChain {
    fn default() -> Self {
        Self::new()
    }
}
