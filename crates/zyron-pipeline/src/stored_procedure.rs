//! Stored procedure registry and interpreter framework for ZyronDB.
//!
//! Provides a registry for stored procedures with security mode support.
//! Procedures are stored by name and resolved at call time. The registry
//! manages metadata only. Execution is handled by the executor layer.

use crate::ids::ProcedureId;
use std::sync::Arc;
use zyron_common::{Result, TypeId, ZyronError};

/// Controls whose security context the procedure body executes under.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
#[repr(u8)]
pub enum SecurityMode {
    /// Execute with the privileges of the procedure owner.
    Definer = 0,
    /// Execute with the privileges of the calling user.
    Invoker = 1,
}

/// A single parameter in a stored procedure's signature.
#[derive(Clone, Debug)]
pub struct ProcedureParam {
    pub name: String,
    pub typeId: TypeId,
}

/// A registered stored procedure definition.
#[derive(Clone, Debug)]
pub struct Procedure {
    pub id: ProcedureId,
    pub name: String,
    pub params: Vec<ProcedureParam>,
    pub bodySql: String,
    pub security: SecurityMode,
    pub ownerId: u32,
}

/// In-memory registry of stored procedures keyed by name.
pub struct ProcedureRegistry {
    procedures: scc::HashMap<String, Arc<Procedure>>,
}

impl ProcedureRegistry {
    /// Creates an empty procedure registry.
    pub fn new() -> Self {
        Self {
            procedures: scc::HashMap::new(),
        }
    }

    /// Registers a new stored procedure. Returns an error if a procedure
    /// with the same name already exists.
    pub fn register(&self, proc: Procedure) -> Result<()> {
        let name = proc.name.clone();
        let arc = Arc::new(proc);
        let entry = self.procedures.entry_sync(name.clone());
        match entry {
            scc::hash_map::Entry::Occupied(_) => Err(ZyronError::ProcedureNotFound(format!(
                "procedure already exists: {name}"
            ))),
            scc::hash_map::Entry::Vacant(vac) => {
                vac.insert_entry(arc);
                Ok(())
            }
        }
    }

    /// Removes a stored procedure by name. Returns an error if the
    /// procedure is not found.
    pub fn dropProcedure(&self, name: &str) -> Result<()> {
        match self.procedures.remove_sync(&name.to_string()) {
            Some(_) => Ok(()),
            None => Err(ZyronError::ProcedureNotFound(name.to_string())),
        }
    }

    /// Resolves a stored procedure by name. Returns None if not found.
    pub fn resolve(&self, name: &str) -> Option<Arc<Procedure>> {
        let mut result = None;
        self.procedures.read_sync(&name.to_string(), |_k, v| {
            result = Some(Arc::clone(v));
        });
        result
    }

    /// Returns the total number of registered procedures.
    pub fn procedureCount(&self) -> usize {
        let mut count = 0;
        self.procedures.iter_sync(|_k, _v| {
            count += 1;
            true
        });
        count
    }
}

impl Default for ProcedureRegistry {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn makeProc(name: &str, id: u32, security: SecurityMode) -> Procedure {
        Procedure {
            id: ProcedureId(id),
            name: name.to_string(),
            params: vec![ProcedureParam {
                name: "input_id".to_string(),
                typeId: TypeId::Int32,
            }],
            bodySql: "BEGIN SELECT 1; END".to_string(),
            security,
            ownerId: 1,
        }
    }

    #[test]
    fn test_register_and_resolve() {
        let reg = ProcedureRegistry::new();
        let proc = makeProc("process_order", 1, SecurityMode::Definer);
        reg.register(proc).expect("register");

        let resolved = reg.resolve("process_order");
        assert!(resolved.is_some());
        let p = resolved.expect("resolved");
        assert_eq!(p.name, "process_order");
        assert_eq!(p.security, SecurityMode::Definer);
    }

    #[test]
    fn test_duplicate_rejected() {
        let reg = ProcedureRegistry::new();
        reg.register(makeProc("dup_proc", 1, SecurityMode::Invoker))
            .expect("first");
        let err = reg
            .register(makeProc("dup_proc", 2, SecurityMode::Invoker))
            .unwrap_err();
        assert!(matches!(err, ZyronError::ProcedureNotFound(_)));
    }

    #[test]
    fn test_drop_procedure() {
        let reg = ProcedureRegistry::new();
        reg.register(makeProc("temp_proc", 1, SecurityMode::Definer))
            .expect("register");
        reg.dropProcedure("temp_proc").expect("drop");
        assert!(reg.resolve("temp_proc").is_none());
    }

    #[test]
    fn test_drop_nonexistent() {
        let reg = ProcedureRegistry::new();
        let err = reg.dropProcedure("missing").unwrap_err();
        assert!(matches!(err, ZyronError::ProcedureNotFound(_)));
    }

    #[test]
    fn test_resolve_nonexistent_returns_none() {
        let reg = ProcedureRegistry::new();
        assert!(reg.resolve("no_such_proc").is_none());
    }

    #[test]
    fn test_procedure_count() {
        let reg = ProcedureRegistry::new();
        assert_eq!(reg.procedureCount(), 0);

        reg.register(makeProc("p1", 1, SecurityMode::Definer))
            .expect("register p1");
        reg.register(makeProc("p2", 2, SecurityMode::Invoker))
            .expect("register p2");
        assert_eq!(reg.procedureCount(), 2);

        reg.dropProcedure("p1").expect("drop");
        assert_eq!(reg.procedureCount(), 1);
    }

    #[test]
    fn test_security_modes() {
        let reg = ProcedureRegistry::new();
        reg.register(makeProc("definer_proc", 1, SecurityMode::Definer))
            .expect("register");
        reg.register(makeProc("invoker_proc", 2, SecurityMode::Invoker))
            .expect("register");

        let definer = reg.resolve("definer_proc").expect("resolved");
        let invoker = reg.resolve("invoker_proc").expect("resolved");

        assert_eq!(definer.security, SecurityMode::Definer);
        assert_eq!(invoker.security, SecurityMode::Invoker);
    }

    #[test]
    fn test_procedure_with_multiple_params() {
        let reg = ProcedureRegistry::new();
        let proc = Procedure {
            id: ProcedureId(10),
            name: "multi_param".to_string(),
            params: vec![
                ProcedureParam {
                    name: "id".to_string(),
                    typeId: TypeId::Int32,
                },
                ProcedureParam {
                    name: "name".to_string(),
                    typeId: TypeId::Varchar,
                },
                ProcedureParam {
                    name: "amount".to_string(),
                    typeId: TypeId::Float64,
                },
            ],
            bodySql: "BEGIN INSERT INTO orders(id, name, amount) VALUES ($1, $2, $3); END"
                .to_string(),
            security: SecurityMode::Definer,
            ownerId: 5,
        };
        reg.register(proc).expect("register");

        let resolved = reg.resolve("multi_param").expect("resolved");
        assert_eq!(resolved.params.len(), 3);
        assert_eq!(resolved.params[0].name, "id");
        assert_eq!(resolved.params[1].typeId, TypeId::Varchar);
        assert_eq!(resolved.params[2].typeId, TypeId::Float64);
        assert_eq!(resolved.ownerId, 5);
    }

    #[test]
    fn test_register_after_drop_succeeds() {
        let reg = ProcedureRegistry::new();
        reg.register(makeProc("recycled", 1, SecurityMode::Definer))
            .expect("first register");
        reg.dropProcedure("recycled").expect("drop");
        reg.register(makeProc("recycled", 2, SecurityMode::Invoker))
            .expect("re-register");

        let resolved = reg.resolve("recycled").expect("resolved");
        assert_eq!(resolved.id, ProcedureId(2));
        assert_eq!(resolved.security, SecurityMode::Invoker);
    }

    #[test]
    fn test_no_params_procedure() {
        let reg = ProcedureRegistry::new();
        let proc = Procedure {
            id: ProcedureId(20),
            name: "cleanup".to_string(),
            params: Vec::new(),
            bodySql: "BEGIN DELETE FROM temp_data; END".to_string(),
            security: SecurityMode::Invoker,
            ownerId: 1,
        };
        reg.register(proc).expect("register");

        let resolved = reg.resolve("cleanup").expect("resolved");
        assert!(resolved.params.is_empty());
    }
}
