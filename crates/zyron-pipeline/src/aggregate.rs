//! User-defined aggregate framework for ZyronDB.
//!
//! Provides a registry of user-defined aggregates (UDAs). Each aggregate
//! is defined by a state transition function (sfunc), an optional final
//! function (finalfunc), an optional combine function (combinefunc) for
//! parallel aggregation, and an initial condition value.

use crate::ids::AggregateId;
use std::sync::Arc;
use zyron_common::{Result, TypeId, ZyronError};

/// A user-defined aggregate definition. Composed of named functions
/// that manage the aggregate state through transition, finalization,
/// and optional parallel combination.
#[derive(Clone, Debug)]
pub struct UserDefinedAggregate {
    pub id: AggregateId,
    pub name: String,
    pub inputTypes: Vec<TypeId>,
    pub stateType: TypeId,
    pub returnType: TypeId,
    /// Name of the state transition function called for each input row.
    pub sfuncName: String,
    /// Name of the optional function called after all rows are processed
    /// to produce the final aggregate result.
    pub finalfuncName: Option<String>,
    /// Name of the optional function that merges two partial aggregate
    /// states for parallel aggregation.
    pub combinefuncName: Option<String>,
    /// Optional initial value for the aggregate state, as a string literal.
    pub initcond: Option<String>,
}

/// In-memory registry of user-defined aggregates keyed by name.
pub struct UdaRegistry {
    aggregates: scc::HashMap<String, Arc<UserDefinedAggregate>>,
}

impl UdaRegistry {
    /// Creates an empty aggregate registry.
    pub fn new() -> Self {
        Self {
            aggregates: scc::HashMap::new(),
        }
    }

    /// Registers a new user-defined aggregate. Returns an error if an
    /// aggregate with the same name already exists.
    pub fn register(&self, uda: UserDefinedAggregate) -> Result<()> {
        let name = uda.name.clone();
        let arc = Arc::new(uda);
        let entry = self.aggregates.entry_sync(name.clone());
        match entry {
            scc::hash_map::Entry::Occupied(_) => Err(ZyronError::FunctionAlreadyExists(name)),
            scc::hash_map::Entry::Vacant(vac) => {
                vac.insert_entry(arc);
                Ok(())
            }
        }
    }

    /// Removes an aggregate by name. Returns an error if not found.
    pub fn dropAggregate(&self, name: &str) -> Result<()> {
        match self.aggregates.remove_sync(&name.to_string()) {
            Some(_) => Ok(()),
            None => Err(ZyronError::FunctionNotFound(name.to_string())),
        }
    }

    /// Resolves an aggregate by name. Returns None if not found.
    pub fn resolve(&self, name: &str) -> Option<Arc<UserDefinedAggregate>> {
        let mut result = None;
        self.aggregates.read_sync(&name.to_string(), |_k, v| {
            result = Some(Arc::clone(v));
        });
        result
    }

    /// Returns true if the given name is a registered aggregate.
    pub fn isAggregate(&self, name: &str) -> bool {
        let mut found = false;
        self.aggregates.read_sync(&name.to_string(), |_k, _v| {
            found = true;
        });
        found
    }

    /// Returns the total number of registered aggregates.
    pub fn aggregateCount(&self) -> usize {
        let mut count = 0;
        self.aggregates.iter_sync(|_k, _v| {
            count += 1;
            true
        });
        count
    }
}

impl Default for UdaRegistry {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn makeUda(name: &str, id: u32) -> UserDefinedAggregate {
        UserDefinedAggregate {
            id: AggregateId(id),
            name: name.to_string(),
            inputTypes: vec![TypeId::Int64],
            stateType: TypeId::Int64,
            returnType: TypeId::Int64,
            sfuncName: format!("{name}_sfunc"),
            finalfuncName: None,
            combinefuncName: None,
            initcond: Some("0".to_string()),
        }
    }

    #[test]
    fn test_register_and_resolve() {
        let reg = UdaRegistry::new();
        let uda = makeUda("my_sum", 1);
        reg.register(uda).expect("register");

        let resolved = reg.resolve("my_sum");
        assert!(resolved.is_some());
        let agg = resolved.expect("resolved");
        assert_eq!(agg.name, "my_sum");
        assert_eq!(agg.sfuncName, "my_sum_sfunc");
    }

    #[test]
    fn test_duplicate_rejected() {
        let reg = UdaRegistry::new();
        reg.register(makeUda("dup_agg", 1)).expect("first");
        let err = reg.register(makeUda("dup_agg", 2)).unwrap_err();
        assert!(matches!(err, ZyronError::FunctionAlreadyExists(_)));
    }

    #[test]
    fn test_drop_aggregate() {
        let reg = UdaRegistry::new();
        reg.register(makeUda("temp_agg", 1)).expect("register");
        reg.dropAggregate("temp_agg").expect("drop");
        assert!(reg.resolve("temp_agg").is_none());
    }

    #[test]
    fn test_drop_nonexistent() {
        let reg = UdaRegistry::new();
        let err = reg.dropAggregate("missing").unwrap_err();
        assert!(matches!(err, ZyronError::FunctionNotFound(_)));
    }

    #[test]
    fn test_is_aggregate() {
        let reg = UdaRegistry::new();
        reg.register(makeUda("count_distinct", 1))
            .expect("register");
        assert!(reg.isAggregate("count_distinct"));
        assert!(!reg.isAggregate("nonexistent"));
    }

    #[test]
    fn test_aggregate_count() {
        let reg = UdaRegistry::new();
        assert_eq!(reg.aggregateCount(), 0);

        reg.register(makeUda("a1", 1)).expect("register");
        reg.register(makeUda("a2", 2)).expect("register");
        assert_eq!(reg.aggregateCount(), 2);

        reg.dropAggregate("a1").expect("drop");
        assert_eq!(reg.aggregateCount(), 1);
    }

    #[test]
    fn test_full_aggregate_definition() {
        let reg = UdaRegistry::new();
        let uda = UserDefinedAggregate {
            id: AggregateId(10),
            name: "weighted_avg".to_string(),
            inputTypes: vec![TypeId::Float64, TypeId::Float64],
            stateType: TypeId::Bytea,
            returnType: TypeId::Float64,
            sfuncName: "wavg_sfunc".to_string(),
            finalfuncName: Some("wavg_final".to_string()),
            combinefuncName: Some("wavg_combine".to_string()),
            initcond: Some("{\"sum\": 0.0, \"weight\": 0.0}".to_string()),
        };
        reg.register(uda).expect("register");

        let resolved = reg.resolve("weighted_avg").expect("resolved");
        assert_eq!(resolved.inputTypes.len(), 2);
        assert_eq!(resolved.finalfuncName.as_deref(), Some("wavg_final"));
        assert_eq!(resolved.combinefuncName.as_deref(), Some("wavg_combine"));
        assert!(resolved.initcond.is_some());
    }

    #[test]
    fn test_resolve_nonexistent_returns_none() {
        let reg = UdaRegistry::new();
        assert!(reg.resolve("no_such_agg").is_none());
    }

    #[test]
    fn test_register_after_drop_succeeds() {
        let reg = UdaRegistry::new();
        reg.register(makeUda("recycled", 1))
            .expect("first register");
        reg.dropAggregate("recycled").expect("drop");
        reg.register(makeUda("recycled", 2)).expect("re-register");

        let resolved = reg.resolve("recycled").expect("resolved");
        assert_eq!(resolved.id, AggregateId(2));
    }
}
