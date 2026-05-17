//! Lock-free legal-hold registry.
//!
//! Reads on the DML hot path go through an `RcuMap<table_id, Vec<CompiledHold>>`
//! plus an atomic generation counter. The registry is rebuilt from the catalog
//! on create/drop/release. Predicate evaluation against a decoded row is
//! delegated to an injected evaluator so this crate does not depend on the
//! executor.

use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use zyron_auth::rcu::RcuMap;
use zyron_catalog::schema::LegalHoldEntry;
use zyron_common::Result;

/// A hold compiled for fast matching. An empty predicate means the whole
/// table is held.
#[derive(Debug, Clone)]
pub struct CompiledHold {
    pub name: String,
    pub table_id: u32,
    pub predicate_sql: String,
}

impl CompiledHold {
    pub fn whole_table(&self) -> bool {
        self.predicate_sql.trim().is_empty()
    }
}

/// Evaluates a hold predicate against a decoded row. Implemented by the
/// executor bridge; injected to avoid a zyron-executor dependency cycle.
pub trait HoldPredicateEvaluator: Send + Sync {
    /// Returns true when `predicate_sql` matches the row identified by
    /// `table_id` + raw tuple bytes.
    fn matches(&self, table_id: u32, predicate_sql: &str, tuple: &[u8]) -> Result<bool>;
}

/// Lock-free registry of active legal holds.
pub struct LegalHoldRegistry {
    by_table: RcuMap<u32, Vec<CompiledHold>>,
    generation: AtomicU64,
}

impl LegalHoldRegistry {
    pub fn new() -> Self {
        Self {
            by_table: RcuMap::empty_map(),
            generation: AtomicU64::new(0),
        }
    }

    /// Current registry generation. Callers that must not race a concurrent
    /// reload (e.g. the purge worker) read this before and after their work.
    pub fn generation(&self) -> u64 {
        self.generation.load(Ordering::Acquire)
    }

    /// Rebuilds the registry from the catalog's active legal holds.
    pub fn reload(&self, holds: &[LegalHoldEntry]) {
        // Collect distinct tables then rebuild each table's vector.
        let mut tables: Vec<u32> = holds.iter().map(|h| h.table_id).collect();
        tables.sort_unstable();
        tables.dedup();
        // Drop tables that no longer have any hold.
        // RcuMap has no iter; track via the rebuilt set and overwrite/remove.
        for t in &tables {
            let v: Vec<CompiledHold> = holds
                .iter()
                .filter(|h| h.table_id == *t && h.is_active())
                .map(|h| CompiledHold {
                    name: h.name.clone(),
                    table_id: h.table_id,
                    predicate_sql: h.predicate_sql.clone(),
                })
                .collect();
            if v.is_empty() {
                self.by_table.remove(t);
            } else {
                self.by_table.insert(*t, v);
            }
        }
        self.generation.fetch_add(1, Ordering::AcqRel);
    }

    /// True when the table has at least one active hold.
    pub fn table_has_hold(&self, table_id: u32) -> bool {
        self.by_table
            .get(&table_id)
            .map(|v| !v.is_empty())
            .unwrap_or(false)
    }

    /// True when the given row is protected by any active hold on its table.
    /// A whole-table hold protects every row. Predicate holds delegate to the
    /// evaluator. On evaluator error the row is treated as protected
    /// (fail closed).
    pub fn row_protected(
        &self,
        table_id: u32,
        tuple: &[u8],
        evaluator: &dyn HoldPredicateEvaluator,
    ) -> bool {
        let holds = match self.by_table.get(&table_id) {
            Some(h) => h,
            None => return false,
        };
        for h in &holds {
            if h.whole_table() {
                return true;
            }
            match evaluator.matches(table_id, &h.predicate_sql, tuple) {
                Ok(true) => return true,
                Ok(false) => {}
                Err(_) => return true,
            }
        }
        false
    }
}

impl Default for LegalHoldRegistry {
    fn default() -> Self {
        Self::new()
    }
}

/// Shared registry handle.
pub fn shared() -> Arc<LegalHoldRegistry> {
    Arc::new(LegalHoldRegistry::new())
}
