//! User-defined function registry with overloading, inlining, and hot-swap.
//!
//! Supports SQL scalar/table functions and Rust scalar/vectorized functions.
//! Functions are registered by name with overloading on parameter types.
//! Immutable SQL scalar functions can be inlined at bind time.
//! Rust UDFs support atomic hot-swap of the function pointer.

use crate::ids::FunctionId;
use std::sync::Arc;
use std::sync::atomic::{AtomicPtr, Ordering};
use zyron_common::{Result, TypeId, ZyronError};

/// Volatility classification for functions. Determines whether the
/// planner can cache or inline function results.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
#[repr(u8)]
pub enum Volatility {
    /// Returns the same result for the same inputs across all calls.
    Immutable = 0,
    /// Returns the same result within a single statement execution.
    Stable = 1,
    /// May return different results on each call.
    Volatile = 2,
}

/// The return type of a user-defined function.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum FunctionReturnType {
    /// Returns a single scalar value.
    Scalar(TypeId),
    /// Returns a table with named typed columns.
    Table(Vec<(String, TypeId)>),
    /// Returns a set of values of the given type.
    SetOf(TypeId),
}

/// Describes the function's name, parameters, return type, and volatility.
#[derive(Clone, Debug)]
pub struct FunctionSignature {
    pub name: String,
    pub params: Vec<(String, TypeId)>,
    pub returnType: FunctionReturnType,
    pub volatility: Volatility,
}

/// A registered user-defined function definition. Each variant
/// holds the signature and the body or pointer to the implementation.
pub enum UdfDefinition {
    /// SQL scalar function with a SQL expression body.
    SqlScalar {
        id: FunctionId,
        signature: FunctionSignature,
        bodySql: String,
    },
    /// SQL table-valued function with a SQL query body.
    SqlTable {
        id: FunctionId,
        signature: FunctionSignature,
        bodySql: String,
    },
    /// Rust scalar function loaded from a shared library.
    RustScalar {
        id: FunctionId,
        signature: FunctionSignature,
        libraryPath: String,
        symbolName: String,
        funcPtr: AtomicPtr<()>,
    },
    /// Rust vectorized function for batch columnar execution.
    RustVectorized {
        id: FunctionId,
        signature: FunctionSignature,
        libraryPath: String,
        symbolName: String,
        funcPtr: AtomicPtr<()>,
    },
}

impl UdfDefinition {
    /// Returns a reference to the function signature.
    pub fn signature(&self) -> &FunctionSignature {
        match self {
            UdfDefinition::SqlScalar { signature, .. } => signature,
            UdfDefinition::SqlTable { signature, .. } => signature,
            UdfDefinition::RustScalar { signature, .. } => signature,
            UdfDefinition::RustVectorized { signature, .. } => signature,
        }
    }

    /// Returns the function name.
    pub fn name(&self) -> &str {
        &self.signature().name
    }

    /// Returns the parameter type list for this definition.
    pub fn paramTypes(&self) -> Vec<TypeId> {
        self.signature().params.iter().map(|(_, t)| *t).collect()
    }
}

/// Key for looking up a specific function overload.
/// Two keys are equal when they have the same name and parameter types.
#[derive(Clone, Debug, Hash, PartialEq, Eq)]
pub struct UdfRegistryKey {
    pub name: String,
    pub paramTypes: Vec<TypeId>,
}

/// In-memory registry of user-defined functions.
/// Functions are stored by name, with each name mapping to a list
/// of overloads distinguished by parameter types.
pub struct UdfRegistry {
    functions: scc::HashMap<String, Vec<Arc<UdfDefinition>>>,
}

impl UdfRegistry {
    /// Creates an empty function registry.
    pub fn new() -> Self {
        Self {
            functions: scc::HashMap::new(),
        }
    }

    /// Registers a new function definition. Returns an error if an
    /// overload with the same name and parameter types already exists.
    pub fn register(&self, def: UdfDefinition) -> Result<()> {
        let name = def.name().to_string();
        let paramTypes = def.paramTypes();

        let entry = self.functions.entry_sync(name.clone());
        match entry {
            scc::hash_map::Entry::Occupied(mut occ) => {
                let overloads = occ.get_mut();
                // Check for duplicate overload.
                for existing in overloads.iter() {
                    if existing.paramTypes() == paramTypes {
                        return Err(ZyronError::FunctionAlreadyExists(name));
                    }
                }
                overloads.push(Arc::new(def));
            }
            scc::hash_map::Entry::Vacant(vac) => {
                vac.insert_entry(vec![Arc::new(def)]);
            }
        }
        Ok(())
    }

    /// Drops a function. If paramTypes is Some, removes only the matching
    /// overload. If None, removes all overloads with that name.
    /// Returns an error if no matching function is found.
    pub fn dropFunction(&self, name: &str, paramTypes: Option<&[TypeId]>) -> Result<()> {
        let mut found = false;
        let entry = self.functions.entry_sync(name.to_string());
        match entry {
            scc::hash_map::Entry::Occupied(mut occ) => match paramTypes {
                Some(types) => {
                    let overloads = occ.get_mut();
                    let before = overloads.len();
                    overloads.retain(|d| d.paramTypes() != types);
                    found = overloads.len() < before;
                    if overloads.is_empty() {
                        let _ = occ.remove_entry();
                    }
                }
                None => {
                    found = true;
                    let _ = occ.remove_entry();
                }
            },
            scc::hash_map::Entry::Vacant(_) => {}
        }
        if !found {
            return Err(ZyronError::FunctionNotFound(name.to_string()));
        }
        Ok(())
    }

    /// Resolves a function by name and argument types.
    /// Tries exact match first, then checks type compatibility.
    pub fn resolve(&self, name: &str, argTypes: &[TypeId]) -> Option<Arc<UdfDefinition>> {
        let mut result = None;
        self.functions
            .read_sync(&name.to_string(), |_k, overloads| {
                // Exact match first.
                for def in overloads.iter() {
                    if def.paramTypes() == argTypes {
                        result = Some(Arc::clone(def));
                        return;
                    }
                }
                // Fallback: match by parameter count with compatible types.
                for def in overloads.iter() {
                    let defTypes = def.paramTypes();
                    if defTypes.len() == argTypes.len() && typesCompatible(&defTypes, argTypes) {
                        result = Some(Arc::clone(def));
                        return;
                    }
                }
            });
        result
    }

    /// Returns true if the resolved function for the given name and
    /// argument types is an IMMUTABLE SQL scalar function.
    pub fn isImmutableSql(&self, name: &str, argTypes: &[TypeId]) -> bool {
        match self.resolve(name, argTypes) {
            Some(def) => {
                matches!(&*def, UdfDefinition::SqlScalar { .. })
                    && def.signature().volatility == Volatility::Immutable
            }
            None => false,
        }
    }

    /// Atomically swaps the function pointer for a Rust UDF.
    /// Loads the new library path and symbol name into the definition.
    /// The old function pointer is replaced with a null pointer to
    /// signal that the caller should reload from the new library.
    pub fn hotSwap(
        &self,
        name: &str,
        paramTypes: &[TypeId],
        newLibrary: &str,
        newSymbol: &str,
    ) -> Result<()> {
        let mut found = false;
        let entry = self.functions.entry_sync(name.to_string());
        match entry {
            scc::hash_map::Entry::Occupied(mut occ) => {
                let overloads = occ.get_mut();
                for def in overloads.iter() {
                    if def.paramTypes() == paramTypes {
                        match &**def {
                            UdfDefinition::RustScalar { funcPtr, .. } => {
                                // Store a null pointer to signal reload needed.
                                // The actual library loading is handled by the
                                // executor when it detects a null function pointer.
                                funcPtr.store(std::ptr::null_mut(), Ordering::Release);
                                found = true;
                            }
                            UdfDefinition::RustVectorized { funcPtr, .. } => {
                                funcPtr.store(std::ptr::null_mut(), Ordering::Release);
                                found = true;
                            }
                            _ => {
                                return Err(ZyronError::UdfExecutionError(format!(
                                    "hot-swap only supported for Rust UDFs, not SQL functions: {name}"
                                )));
                            }
                        }
                        break;
                    }
                }
                if found {
                    // Replace the definition with updated library/symbol info.
                    let paramTypesVec: Vec<TypeId> = paramTypes.to_vec();
                    for i in 0..overloads.len() {
                        if overloads[i].paramTypes() == paramTypesVec {
                            let oldDef = &*overloads[i];
                            let newDef = match oldDef {
                                UdfDefinition::RustScalar { id, signature, .. } => {
                                    UdfDefinition::RustScalar {
                                        id: *id,
                                        signature: signature.clone(),
                                        libraryPath: newLibrary.to_string(),
                                        symbolName: newSymbol.to_string(),
                                        funcPtr: AtomicPtr::new(std::ptr::null_mut()),
                                    }
                                }
                                UdfDefinition::RustVectorized { id, signature, .. } => {
                                    UdfDefinition::RustVectorized {
                                        id: *id,
                                        signature: signature.clone(),
                                        libraryPath: newLibrary.to_string(),
                                        symbolName: newSymbol.to_string(),
                                        funcPtr: AtomicPtr::new(std::ptr::null_mut()),
                                    }
                                }
                                _ => break,
                            };
                            overloads[i] = Arc::new(newDef);
                            break;
                        }
                    }
                }
            }
            scc::hash_map::Entry::Vacant(_) => {}
        }
        if !found {
            return Err(ZyronError::FunctionNotFound(name.to_string()));
        }
        Ok(())
    }

    /// Returns the total number of function overloads in the registry.
    pub fn functionCount(&self) -> usize {
        let mut count = 0;
        self.functions.iter_sync(|_k, v| {
            count += v.len();
            true
        });
        count
    }
}

impl Default for UdfRegistry {
    fn default() -> Self {
        Self::new()
    }
}

/// Checks whether the argument types are compatible with the parameter types.
/// Two types are compatible if they are equal, or if an implicit widening
/// cast exists (e.g. Int32 -> Int64, Float32 -> Float64).
fn typesCompatible(paramTypes: &[TypeId], argTypes: &[TypeId]) -> bool {
    if paramTypes.len() != argTypes.len() {
        return false;
    }
    for (param, arg) in paramTypes.iter().zip(argTypes.iter()) {
        if param == arg {
            continue;
        }
        if !canImplicitCast(*arg, *param) {
            return false;
        }
    }
    true
}

/// Returns true if an implicit cast from source to target is allowed.
/// Supports standard numeric widening casts.
fn canImplicitCast(source: TypeId, target: TypeId) -> bool {
    use TypeId::*;
    matches!(
        (source, target),
        (Int8, Int16)
            | (Int8, Int32)
            | (Int8, Int64)
            | (Int8, Int128)
            | (Int16, Int32)
            | (Int16, Int64)
            | (Int16, Int128)
            | (Int32, Int64)
            | (Int32, Int128)
            | (Int64, Int128)
            | (UInt8, UInt16)
            | (UInt8, UInt32)
            | (UInt8, UInt64)
            | (UInt8, UInt128)
            | (UInt16, UInt32)
            | (UInt16, UInt64)
            | (UInt16, UInt128)
            | (UInt32, UInt64)
            | (UInt32, UInt128)
            | (UInt64, UInt128)
            | (Float32, Float64)
            | (Int8, Float64)
            | (Int16, Float64)
            | (Int32, Float64)
            | (Varchar, Text)
            | (Char, Varchar)
            | (Char, Text)
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn makeSqlScalar(
        name: &str,
        params: Vec<(&str, TypeId)>,
        retType: TypeId,
        vol: Volatility,
    ) -> UdfDefinition {
        UdfDefinition::SqlScalar {
            id: FunctionId(1),
            signature: FunctionSignature {
                name: name.to_string(),
                params: params
                    .into_iter()
                    .map(|(n, t)| (n.to_string(), t))
                    .collect(),
                returnType: FunctionReturnType::Scalar(retType),
                volatility: vol,
            },
            bodySql: "SELECT 1".to_string(),
        }
    }

    #[test]
    fn test_register_and_resolve() {
        let reg = UdfRegistry::new();
        let def = makeSqlScalar(
            "add_one",
            vec![("x", TypeId::Int32)],
            TypeId::Int32,
            Volatility::Immutable,
        );
        reg.register(def).expect("register");

        let resolved = reg.resolve("add_one", &[TypeId::Int32]);
        assert!(resolved.is_some());
        assert_eq!(resolved.as_ref().expect("resolved").name(), "add_one");
    }

    #[test]
    fn test_duplicate_overload_rejected() {
        let reg = UdfRegistry::new();
        let d1 = makeSqlScalar(
            "f",
            vec![("x", TypeId::Int32)],
            TypeId::Int32,
            Volatility::Stable,
        );
        let d2 = makeSqlScalar(
            "f",
            vec![("y", TypeId::Int32)],
            TypeId::Int64,
            Volatility::Volatile,
        );
        reg.register(d1).expect("first");
        let err = reg.register(d2).unwrap_err();
        assert!(matches!(err, ZyronError::FunctionAlreadyExists(_)));
    }

    #[test]
    fn test_overloading_different_params() {
        let reg = UdfRegistry::new();
        let d1 = makeSqlScalar(
            "f",
            vec![("x", TypeId::Int32)],
            TypeId::Int32,
            Volatility::Immutable,
        );
        let d2 = makeSqlScalar(
            "f",
            vec![("x", TypeId::Int64)],
            TypeId::Int64,
            Volatility::Immutable,
        );
        reg.register(d1).expect("first overload");
        reg.register(d2).expect("second overload");

        let r1 = reg.resolve("f", &[TypeId::Int32]);
        let r2 = reg.resolve("f", &[TypeId::Int64]);
        assert!(r1.is_some());
        assert!(r2.is_some());
        // Verify they resolve to different return types.
        assert_ne!(
            r1.expect("r1").signature().returnType,
            r2.expect("r2").signature().returnType
        );
    }

    #[test]
    fn test_resolve_not_found() {
        let reg = UdfRegistry::new();
        assert!(reg.resolve("nonexistent", &[]).is_none());
    }

    #[test]
    fn test_resolve_with_implicit_cast() {
        let reg = UdfRegistry::new();
        let def = makeSqlScalar(
            "wide",
            vec![("x", TypeId::Int64)],
            TypeId::Int64,
            Volatility::Stable,
        );
        reg.register(def).expect("register");

        // Int32 should implicitly cast to Int64.
        let resolved = reg.resolve("wide", &[TypeId::Int32]);
        assert!(resolved.is_some());
    }

    #[test]
    fn test_drop_specific_overload() {
        let reg = UdfRegistry::new();
        let d1 = makeSqlScalar(
            "f",
            vec![("x", TypeId::Int32)],
            TypeId::Int32,
            Volatility::Immutable,
        );
        let d2 = makeSqlScalar(
            "f",
            vec![("x", TypeId::Text)],
            TypeId::Text,
            Volatility::Immutable,
        );
        reg.register(d1).expect("register d1");
        reg.register(d2).expect("register d2");

        reg.dropFunction("f", Some(&[TypeId::Int32]))
            .expect("drop overload");
        // Int32 overload removed, no implicit cast from Int32 to Text.
        assert!(reg.resolve("f", &[TypeId::Int32]).is_none());
        assert!(reg.resolve("f", &[TypeId::Text]).is_some());
    }

    #[test]
    fn test_drop_all_overloads() {
        let reg = UdfRegistry::new();
        let d1 = makeSqlScalar(
            "f",
            vec![("x", TypeId::Int32)],
            TypeId::Int32,
            Volatility::Immutable,
        );
        let d2 = makeSqlScalar(
            "f",
            vec![("x", TypeId::Int64)],
            TypeId::Int64,
            Volatility::Immutable,
        );
        reg.register(d1).expect("register d1");
        reg.register(d2).expect("register d2");

        reg.dropFunction("f", None).expect("drop all");
        assert!(reg.resolve("f", &[TypeId::Int32]).is_none());
        assert!(reg.resolve("f", &[TypeId::Int64]).is_none());
    }

    #[test]
    fn test_drop_nonexistent() {
        let reg = UdfRegistry::new();
        let err = reg.dropFunction("nope", None).unwrap_err();
        assert!(matches!(err, ZyronError::FunctionNotFound(_)));
    }

    #[test]
    fn test_is_immutable_sql() {
        let reg = UdfRegistry::new();
        let immut = makeSqlScalar(
            "pure",
            vec![("x", TypeId::Int32)],
            TypeId::Int32,
            Volatility::Immutable,
        );
        let vol = makeSqlScalar("random", vec![], TypeId::Float64, Volatility::Volatile);
        reg.register(immut).expect("register");
        reg.register(vol).expect("register");

        assert!(reg.isImmutableSql("pure", &[TypeId::Int32]));
        assert!(!reg.isImmutableSql("random", &[]));
        assert!(!reg.isImmutableSql("nonexistent", &[]));
    }

    #[test]
    fn test_hot_swap_rust_udf() {
        let reg = UdfRegistry::new();
        let def = UdfDefinition::RustScalar {
            id: FunctionId(10),
            signature: FunctionSignature {
                name: "fast_hash".to_string(),
                params: vec![("data".to_string(), TypeId::Bytea)],
                returnType: FunctionReturnType::Scalar(TypeId::Int64),
                volatility: Volatility::Immutable,
            },
            libraryPath: "/old/lib.so".to_string(),
            symbolName: "fast_hash_v1".to_string(),
            funcPtr: AtomicPtr::new(0x1234 as *mut ()),
        };
        reg.register(def).expect("register");

        reg.hotSwap("fast_hash", &[TypeId::Bytea], "/new/lib.so", "fast_hash_v2")
            .expect("hot swap");

        let resolved = reg
            .resolve("fast_hash", &[TypeId::Bytea])
            .expect("resolved");
        match &*resolved {
            UdfDefinition::RustScalar {
                libraryPath,
                symbolName,
                funcPtr,
                ..
            } => {
                assert_eq!(libraryPath, "/new/lib.so");
                assert_eq!(symbolName, "fast_hash_v2");
                assert!(funcPtr.load(Ordering::Acquire).is_null());
            }
            _ => panic!("expected RustScalar"),
        }
    }

    #[test]
    fn test_hot_swap_sql_function_rejected() {
        let reg = UdfRegistry::new();
        let def = makeSqlScalar("sql_fn", vec![], TypeId::Int32, Volatility::Immutable);
        reg.register(def).expect("register");

        let err = reg.hotSwap("sql_fn", &[], "/lib.so", "sym").unwrap_err();
        assert!(matches!(err, ZyronError::UdfExecutionError(_)));
    }

    #[test]
    fn test_hot_swap_not_found() {
        let reg = UdfRegistry::new();
        let err = reg.hotSwap("nope", &[], "/lib.so", "sym").unwrap_err();
        assert!(matches!(err, ZyronError::FunctionNotFound(_)));
    }

    #[test]
    fn test_function_count() {
        let reg = UdfRegistry::new();
        assert_eq!(reg.functionCount(), 0);

        reg.register(makeSqlScalar(
            "a",
            vec![],
            TypeId::Int32,
            Volatility::Immutable,
        ))
        .expect("register");
        reg.register(makeSqlScalar(
            "b",
            vec![("x", TypeId::Int32)],
            TypeId::Int32,
            Volatility::Stable,
        ))
        .expect("register");
        assert_eq!(reg.functionCount(), 2);
    }

    #[test]
    fn test_table_return_type() {
        let reg = UdfRegistry::new();
        let def = UdfDefinition::SqlTable {
            id: FunctionId(5),
            signature: FunctionSignature {
                name: "get_users".to_string(),
                params: vec![("dept".to_string(), TypeId::Varchar)],
                returnType: FunctionReturnType::Table(vec![
                    ("id".to_string(), TypeId::Int32),
                    ("name".to_string(), TypeId::Varchar),
                ]),
                volatility: Volatility::Stable,
            },
            bodySql: "SELECT id, name FROM users WHERE department = $1".to_string(),
        };
        reg.register(def).expect("register");

        let resolved = reg
            .resolve("get_users", &[TypeId::Varchar])
            .expect("resolved");
        match &resolved.signature().returnType {
            FunctionReturnType::Table(cols) => {
                assert_eq!(cols.len(), 2);
                assert_eq!(cols[0].0, "id");
                assert_eq!(cols[1].0, "name");
            }
            _ => panic!("expected Table return type"),
        }
    }

    #[test]
    fn test_implicit_cast_compatibility() {
        assert!(canImplicitCast(TypeId::Int32, TypeId::Int64));
        assert!(canImplicitCast(TypeId::Float32, TypeId::Float64));
        assert!(canImplicitCast(TypeId::Char, TypeId::Text));
        assert!(!canImplicitCast(TypeId::Int64, TypeId::Int32));
        assert!(!canImplicitCast(TypeId::Text, TypeId::Int32));
    }
}
