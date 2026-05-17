//! DML enforcement hooks: composes ordered BEFORE hooks and enforces legal
//! holds, retention locks, and WORM/immutable tables on DELETE/UPDATE. Lives
//! in zyron-wire so both the server's per-connection executor and the
//! lifecycle DDL dispatch (FORGET USER, retention) share one implementation.

use std::sync::Arc;

use zyron_common::Result;
use zyron_executor::context::DmlHook;

/// Composes multiple DML hooks in order. The first hook to return `Ok(false)`
/// (cancel) or `Err` short-circuits; an `Err` propagates immediately so a
/// legal-hold / WORM violation is a hard error, not a silent cancel.
pub struct CompositeDmlHook {
    hooks: Vec<Arc<dyn DmlHook>>,
}

impl CompositeDmlHook {
    pub fn new(hooks: Vec<Arc<dyn DmlHook>>) -> Self {
        Self { hooks }
    }
}

impl DmlHook for CompositeDmlHook {
    fn before_insert(&self, table_id: u32, tuples: &[&[u8]], txn_id: u32) -> Result<bool> {
        for h in &self.hooks {
            if !h.before_insert(table_id, tuples, txn_id)? {
                return Ok(false);
            }
        }
        Ok(true)
    }

    fn before_delete(&self, table_id: u32, old_data: &[&[u8]], txn_id: u32) -> Result<bool> {
        for h in &self.hooks {
            if !h.before_delete(table_id, old_data, txn_id)? {
                return Ok(false);
            }
        }
        Ok(true)
    }

    fn before_update(
        &self,
        table_id: u32,
        old_data: &[&[u8]],
        new_data: &[&[u8]],
        txn_id: u32,
    ) -> Result<bool> {
        for h in &self.hooks {
            if !h.before_update(table_id, old_data, new_data, txn_id)? {
                return Ok(false);
            }
        }
        Ok(true)
    }
}

/// Enforces legal holds, retention locks, and WORM/immutable tables before
/// any DELETE/UPDATE mutation. A protected row produces a hard
/// `LegalHoldViolation` / `RetentionViolation` error.
///
/// When no row-predicate evaluator is wired, predicate holds fail closed:
/// any active hold on the table blocks the mutation (compliance-safe).
pub struct LegalHoldDmlHook {
    registry: Arc<zyron_lifecycle::legal_hold::LegalHoldRegistry>,
    catalog: Arc<zyron_catalog::Catalog>,
    evaluator: Option<Arc<dyn zyron_lifecycle::legal_hold::HoldPredicateEvaluator>>,
}

impl LegalHoldDmlHook {
    pub fn new(
        registry: Arc<zyron_lifecycle::legal_hold::LegalHoldRegistry>,
        catalog: Arc<zyron_catalog::Catalog>,
    ) -> Self {
        Self {
            registry,
            catalog,
            evaluator: None,
        }
    }

    pub fn with_evaluator(
        mut self,
        evaluator: Arc<dyn zyron_lifecycle::legal_hold::HoldPredicateEvaluator>,
    ) -> Self {
        self.evaluator = Some(evaluator);
        self
    }

    fn worm_check(&self, table_id: u32) -> Result<()> {
        if let Ok(entry) = self
            .catalog
            .get_table_by_id(zyron_catalog::TableId(table_id))
        {
            if zyron_lifecycle::worm::write_locked(&entry) {
                return Err(zyron_common::ZyronError::RetentionViolation(
                    zyron_lifecycle::worm::lock_reason(&entry),
                ));
            }
        }
        Ok(())
    }

    fn hold_check(&self, table_id: u32, rows: &[&[u8]]) -> Result<()> {
        if !self.registry.table_has_hold(table_id) {
            return Ok(());
        }
        match &self.evaluator {
            Some(ev) => {
                for r in rows {
                    if self.registry.row_protected(table_id, r, ev.as_ref()) {
                        return Err(zyron_common::ZyronError::LegalHoldViolation(format!(
                            "rows in table {table_id} are under an active legal hold"
                        )));
                    }
                }
                Ok(())
            }
            None => Err(zyron_common::ZyronError::LegalHoldViolation(format!(
                "table {table_id} is under an active legal hold"
            ))),
        }
    }
}

impl DmlHook for LegalHoldDmlHook {
    fn before_insert(&self, _table_id: u32, _tuples: &[&[u8]], _txn_id: u32) -> Result<bool> {
        Ok(true)
    }

    fn before_delete(&self, table_id: u32, old_data: &[&[u8]], _txn_id: u32) -> Result<bool> {
        self.worm_check(table_id)?;
        self.hold_check(table_id, old_data)?;
        Ok(true)
    }

    fn before_update(
        &self,
        table_id: u32,
        old_data: &[&[u8]],
        _new_data: &[&[u8]],
        _txn_id: u32,
    ) -> Result<bool> {
        self.worm_check(table_id)?;
        self.hold_check(table_id, old_data)?;
        Ok(true)
    }
}
