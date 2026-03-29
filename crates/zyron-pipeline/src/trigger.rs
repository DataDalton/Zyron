//! DML trigger management and execution for ZyronDB.
//!
//! Provides a trigger registry that stores trigger definitions per table,
//! sorted by priority. The TriggerManager is a registry, not an executor.
//! Callers (the executor layer) retrieve matching triggers and execute them.

use crate::ids::TriggerId;
use std::sync::Arc;
use zyron_common::{Result, ZyronError};

/// Maximum allowed trigger recursion depth. Prevents infinite cascades
/// when triggers fire other triggers.
pub const MAX_TRIGGER_DEPTH: u32 = 16;

/// When a trigger fires relative to the triggering operation.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
#[repr(u8)]
pub enum TriggerTiming {
    Before = 0,
    After = 1,
    InsteadOf = 2,
}

/// The DML event that causes a trigger to fire.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
#[repr(u8)]
pub enum TriggerEvent {
    Insert = 0,
    Update = 1,
    Delete = 2,
    Truncate = 3,
}

/// Whether the trigger fires once per row or once per statement.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
#[repr(u8)]
pub enum TriggerLevel {
    Row = 0,
    Statement = 1,
}

/// Holds old and new row collections for statement-level triggers.
/// Statement-level triggers can access all affected rows through
/// the aliased transition table names.
#[derive(Clone, Debug, Default)]
pub struct TransitionTables {
    pub oldTableAlias: Option<String>,
    pub newTableAlias: Option<String>,
    pub oldRows: Vec<Vec<u8>>,
    pub newRows: Vec<Vec<u8>>,
}

/// A registered trigger definition bound to a specific table.
#[derive(Clone, Debug)]
pub struct Trigger {
    pub id: TriggerId,
    pub name: String,
    pub tableId: u32,
    pub timing: TriggerTiming,
    pub events: Vec<TriggerEvent>,
    pub level: TriggerLevel,
    pub whenCondition: Option<String>,
    pub functionName: String,
    pub args: Vec<String>,
    pub enabled: bool,
    pub priority: u32,
    /// Transition table aliases: (old_table_alias, new_table_alias).
    pub transitionTables: Option<(Option<String>, Option<String>)>,
}

/// Per-row or per-statement context passed to trigger execution.
/// Contains the old/new row data and metadata about the operation.
#[derive(Clone, Debug)]
pub struct TriggerContext {
    pub oldRow: Option<Vec<u8>>,
    pub newRow: Option<Vec<u8>>,
    pub operation: TriggerEvent,
    pub tableName: String,
    pub triggerName: String,
    pub tableId: u32,
    pub txnId: u32,
    pub transitionTables: Option<TransitionTables>,
}

/// In-memory trigger registry keyed by table_id.
/// Each table maps to a list of triggers sorted by ascending priority.
pub struct TriggerManager {
    triggers: scc::HashMap<u32, Vec<Arc<Trigger>>>,
}

impl TriggerManager {
    /// Creates an empty trigger registry.
    pub fn new() -> Self {
        Self {
            triggers: scc::HashMap::new(),
        }
    }

    /// Registers a trigger definition. The trigger is inserted into the
    /// list for its table_id and the list is kept sorted by priority.
    /// Returns an error if a trigger with the same name already exists
    /// on the same table.
    pub fn registerTrigger(&self, trigger: Trigger) -> Result<()> {
        let tableId = trigger.tableId;
        let arc = Arc::new(trigger);

        // Single entry_sync call: check for duplicate and insert.
        let entry = self.triggers.entry_sync(tableId);
        match entry {
            scc::hash_map::Entry::Occupied(mut occ) => {
                let list = occ.get_mut();
                if list.iter().any(|t| t.name == arc.name) {
                    return Err(ZyronError::TriggerAlreadyExists(arc.name.clone()));
                }
                list.push(arc);
                list.sort_by_key(|t| t.priority);
            }
            scc::hash_map::Entry::Vacant(vac) => {
                vac.insert_entry(vec![arc]);
            }
        }
        Ok(())
    }

    /// Removes a trigger by name from the given table.
    /// Returns an error if the trigger is not found.
    pub fn dropTrigger(&self, name: &str, tableId: u32) -> Result<()> {
        let mut found = false;
        let entry = self.triggers.entry_sync(tableId);
        match entry {
            scc::hash_map::Entry::Occupied(mut occ) => {
                let list = occ.get_mut();
                let before = list.len();
                list.retain(|t| t.name != name);
                found = list.len() < before;
            }
            scc::hash_map::Entry::Vacant(_) => {}
        }
        if !found {
            return Err(ZyronError::TriggerNotFound(name.to_string()));
        }
        Ok(())
    }

    /// Enables a trigger by name on the given table.
    /// Returns an error if the trigger is not found.
    pub fn enableTrigger(&self, name: &str, tableId: u32) -> Result<()> {
        self.setEnabled(name, tableId, true)
    }

    /// Disables a trigger by name on the given table.
    /// Returns an error if the trigger is not found.
    pub fn disableTrigger(&self, name: &str, tableId: u32) -> Result<()> {
        self.setEnabled(name, tableId, false)
    }

    /// Sets the enabled flag for a trigger. Shared implementation
    /// for enableTrigger and disableTrigger.
    fn setEnabled(&self, name: &str, tableId: u32, enabled: bool) -> Result<()> {
        let mut found = false;
        let entry = self.triggers.entry_sync(tableId);
        match entry {
            scc::hash_map::Entry::Occupied(mut occ) => {
                let list = occ.get_mut();
                for i in 0..list.len() {
                    if list[i].name == name {
                        let mut updated = (*list[i]).clone();
                        updated.enabled = enabled;
                        list[i] = Arc::new(updated);
                        found = true;
                        break;
                    }
                }
            }
            scc::hash_map::Entry::Vacant(_) => {}
        }
        if !found {
            return Err(ZyronError::TriggerNotFound(name.to_string()));
        }
        Ok(())
    }

    /// Fast path check for whether any enabled triggers exist matching
    /// the given table, timing, and event combination.
    pub fn hasTriggers(&self, tableId: u32, timing: TriggerTiming, event: TriggerEvent) -> bool {
        let mut found = false;
        self.triggers.read_sync(&tableId, |_k, v| {
            for t in v.iter() {
                if t.enabled && t.timing == timing && t.events.contains(&event) {
                    found = true;
                    return;
                }
            }
        });
        found
    }

    /// Returns all enabled triggers matching the given table, timing, and event,
    /// sorted by ascending priority.
    pub fn getMatchingTriggers(
        &self,
        tableId: u32,
        timing: TriggerTiming,
        event: TriggerEvent,
    ) -> Vec<Arc<Trigger>> {
        // The internal list is pre-sorted by priority from registerTrigger.
        // Filtering preserves relative order, so no re-sort needed.
        self.triggers
            .read_sync(&tableId, |_k, v| {
                let mut result = Vec::with_capacity(v.len());
                for t in v.iter() {
                    if t.enabled && t.timing == timing && t.events.contains(&event) {
                        result.push(Arc::clone(t));
                    }
                }
                result
            })
            .unwrap_or_default()
    }

    /// Returns the total number of triggers registered across all tables.
    pub fn triggerCount(&self) -> usize {
        let mut count = 0;
        self.triggers.iter_sync(|_k, v| {
            count += v.len();
            true
        });
        count
    }
}

impl Default for TriggerManager {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn makeTrigger(
        name: &str,
        tableId: u32,
        timing: TriggerTiming,
        event: TriggerEvent,
        priority: u32,
    ) -> Trigger {
        Trigger {
            id: TriggerId(1),
            name: name.to_string(),
            tableId,
            timing,
            events: vec![event],
            level: TriggerLevel::Row,
            whenCondition: None,
            functionName: "test_func".to_string(),
            args: Vec::new(),
            enabled: true,
            priority,
            transitionTables: None,
        }
    }

    #[test]
    fn test_register_and_get_matching() {
        let mgr = TriggerManager::new();
        let t1 = makeTrigger("t1", 10, TriggerTiming::Before, TriggerEvent::Insert, 100);
        let t2 = makeTrigger("t2", 10, TriggerTiming::Before, TriggerEvent::Insert, 50);
        let t3 = makeTrigger("t3", 10, TriggerTiming::After, TriggerEvent::Insert, 200);

        mgr.registerTrigger(t1).expect("register t1");
        mgr.registerTrigger(t2).expect("register t2");
        mgr.registerTrigger(t3).expect("register t3");

        let before = mgr.getMatchingTriggers(10, TriggerTiming::Before, TriggerEvent::Insert);
        assert_eq!(before.len(), 2);
        assert_eq!(before[0].name, "t2");
        assert_eq!(before[1].name, "t1");

        let after = mgr.getMatchingTriggers(10, TriggerTiming::After, TriggerEvent::Insert);
        assert_eq!(after.len(), 1);
        assert_eq!(after[0].name, "t3");
    }

    #[test]
    fn test_duplicate_name_rejected() {
        let mgr = TriggerManager::new();
        let t1 = makeTrigger("dup", 10, TriggerTiming::Before, TriggerEvent::Insert, 100);
        let t2 = makeTrigger("dup", 10, TriggerTiming::After, TriggerEvent::Delete, 200);

        mgr.registerTrigger(t1).expect("first register");
        let err = mgr.registerTrigger(t2).unwrap_err();
        assert!(matches!(err, ZyronError::TriggerAlreadyExists(_)));
    }

    #[test]
    fn test_drop_trigger() {
        let mgr = TriggerManager::new();
        let t1 = makeTrigger("t1", 10, TriggerTiming::Before, TriggerEvent::Insert, 100);
        mgr.registerTrigger(t1).expect("register");

        mgr.dropTrigger("t1", 10).expect("drop");
        assert!(!mgr.hasTriggers(10, TriggerTiming::Before, TriggerEvent::Insert));
    }

    #[test]
    fn test_drop_nonexistent_returns_error() {
        let mgr = TriggerManager::new();
        let err = mgr.dropTrigger("nope", 10).unwrap_err();
        assert!(matches!(err, ZyronError::TriggerNotFound(_)));
    }

    #[test]
    fn test_enable_disable() {
        let mgr = TriggerManager::new();
        let t1 = makeTrigger("t1", 10, TriggerTiming::Before, TriggerEvent::Insert, 100);
        mgr.registerTrigger(t1).expect("register");

        mgr.disableTrigger("t1", 10).expect("disable");
        assert!(!mgr.hasTriggers(10, TriggerTiming::Before, TriggerEvent::Insert));

        mgr.enableTrigger("t1", 10).expect("enable");
        assert!(mgr.hasTriggers(10, TriggerTiming::Before, TriggerEvent::Insert));
    }

    #[test]
    fn test_enable_nonexistent_returns_error() {
        let mgr = TriggerManager::new();
        let err = mgr.enableTrigger("nope", 99).unwrap_err();
        assert!(matches!(err, ZyronError::TriggerNotFound(_)));
    }

    #[test]
    fn test_has_triggers_empty() {
        let mgr = TriggerManager::new();
        assert!(!mgr.hasTriggers(10, TriggerTiming::Before, TriggerEvent::Insert));
    }

    #[test]
    fn test_has_triggers_wrong_event() {
        let mgr = TriggerManager::new();
        let t1 = makeTrigger("t1", 10, TriggerTiming::Before, TriggerEvent::Insert, 100);
        mgr.registerTrigger(t1).expect("register");

        assert!(!mgr.hasTriggers(10, TriggerTiming::Before, TriggerEvent::Delete));
        assert!(!mgr.hasTriggers(10, TriggerTiming::After, TriggerEvent::Insert));
    }

    #[test]
    fn test_multiple_events_per_trigger() {
        let mgr = TriggerManager::new();
        let t = Trigger {
            id: TriggerId(1),
            name: "multi".to_string(),
            tableId: 10,
            timing: TriggerTiming::After,
            events: vec![
                TriggerEvent::Insert,
                TriggerEvent::Update,
                TriggerEvent::Delete,
            ],
            level: TriggerLevel::Row,
            whenCondition: None,
            functionName: "audit_func".to_string(),
            args: Vec::new(),
            enabled: true,
            priority: 100,
            transitionTables: None,
        };
        mgr.registerTrigger(t).expect("register");

        assert!(mgr.hasTriggers(10, TriggerTiming::After, TriggerEvent::Insert));
        assert!(mgr.hasTriggers(10, TriggerTiming::After, TriggerEvent::Update));
        assert!(mgr.hasTriggers(10, TriggerTiming::After, TriggerEvent::Delete));
        assert!(!mgr.hasTriggers(10, TriggerTiming::After, TriggerEvent::Truncate));
    }

    #[test]
    fn test_trigger_count() {
        let mgr = TriggerManager::new();
        assert_eq!(mgr.triggerCount(), 0);

        mgr.registerTrigger(makeTrigger(
            "a",
            10,
            TriggerTiming::Before,
            TriggerEvent::Insert,
            1,
        ))
        .expect("register a");
        mgr.registerTrigger(makeTrigger(
            "b",
            20,
            TriggerTiming::After,
            TriggerEvent::Delete,
            1,
        ))
        .expect("register b");
        assert_eq!(mgr.triggerCount(), 2);
    }

    #[test]
    fn test_different_tables_independent() {
        let mgr = TriggerManager::new();
        let t1 = makeTrigger("t1", 10, TriggerTiming::Before, TriggerEvent::Insert, 100);
        let t2 = makeTrigger("t1", 20, TriggerTiming::Before, TriggerEvent::Insert, 100);

        mgr.registerTrigger(t1).expect("register on table 10");
        mgr.registerTrigger(t2).expect("register on table 20");

        mgr.dropTrigger("t1", 10).expect("drop from table 10");
        assert!(mgr.hasTriggers(20, TriggerTiming::Before, TriggerEvent::Insert));
    }

    #[test]
    fn test_statement_level_trigger() {
        let mgr = TriggerManager::new();
        let t = Trigger {
            id: TriggerId(5),
            name: "stmt_trig".to_string(),
            tableId: 30,
            timing: TriggerTiming::After,
            events: vec![TriggerEvent::Insert],
            level: TriggerLevel::Statement,
            whenCondition: Some("true".to_string()),
            functionName: "batch_audit".to_string(),
            args: vec!["arg1".to_string()],
            enabled: true,
            priority: 500,
            transitionTables: Some((Some("old_data".to_string()), Some("new_data".to_string()))),
        };
        mgr.registerTrigger(t).expect("register");

        let matched = mgr.getMatchingTriggers(30, TriggerTiming::After, TriggerEvent::Insert);
        assert_eq!(matched.len(), 1);
        assert_eq!(matched[0].level, TriggerLevel::Statement);
        assert!(matched[0].transitionTables.is_some());
    }

    #[test]
    fn test_disabled_triggers_excluded_from_matching() {
        let mgr = TriggerManager::new();
        let mut t = makeTrigger("t1", 10, TriggerTiming::Before, TriggerEvent::Insert, 100);
        t.enabled = false;
        mgr.registerTrigger(t).expect("register");

        let matched = mgr.getMatchingTriggers(10, TriggerTiming::Before, TriggerEvent::Insert);
        assert!(matched.is_empty());
    }

    #[test]
    fn test_priority_ordering() {
        let mgr = TriggerManager::new();
        mgr.registerTrigger(makeTrigger(
            "high",
            10,
            TriggerTiming::Before,
            TriggerEvent::Insert,
            1000,
        ))
        .expect("register high");
        mgr.registerTrigger(makeTrigger(
            "low",
            10,
            TriggerTiming::Before,
            TriggerEvent::Insert,
            10,
        ))
        .expect("register low");
        mgr.registerTrigger(makeTrigger(
            "mid",
            10,
            TriggerTiming::Before,
            TriggerEvent::Insert,
            500,
        ))
        .expect("register mid");

        let matched = mgr.getMatchingTriggers(10, TriggerTiming::Before, TriggerEvent::Insert);
        assert_eq!(matched.len(), 3);
        assert_eq!(matched[0].name, "low");
        assert_eq!(matched[1].name, "mid");
        assert_eq!(matched[2].name, "high");
    }
}
