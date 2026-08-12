//! Every scalar function name declared in the zyron-types registry must be
//! dispatched by the executor. Dispatched means the call either succeeds or
//! fails with a semantic error (argument count, argument type, requires an
//! OVER clause). The one outcome this test forbids is the executor not
//! recognizing the name at all.
//!
//! The name list is extracted from the registry source itself, so a name
//! added to the registry without an executor arm fails here immediately.

use zyron_common::TypeId;
use zyron_executor::batch::DataBatch;
use zyron_executor::column::{Column, ColumnData, ScalarValue};
use zyron_executor::expr::evaluate;
use zyron_planner::binder::BoundExpr;

/// Names declared in the scalar section of the registry match
fn declared_scalar_names() -> Vec<String> {
    let src = include_str!("../../zyron-types/src/registry.rs");
    let start = src
        .find("fn infer_types_scalar_return_type")
        .expect("registry scalar fn present");
    let end = src
        .find("fn infer_types_aggregate_return_type")
        .expect("registry aggregate fn present");
    let scalar = &src[start..end];
    let mut names = Vec::new();
    let mut rest = scalar;
    while let Some(open) = rest.find('"') {
        rest = &rest[open + 1..];
        let Some(close) = rest.find('"') else { break };
        let token = &rest[..close];
        rest = &rest[close + 1..];
        if !token.is_empty()
            && token
                .chars()
                .all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || c == '_')
        {
            names.push(token.to_string());
        }
    }
    names.sort();
    names.dedup();
    names
}

fn one_row_batch() -> DataBatch {
    DataBatch::new(vec![Column::new(ColumnData::Int64(vec![0]), TypeId::Int64)])
}

fn call_scalar(name: &str, args: Vec<BoundExpr>) -> Result<Column, String> {
    let expr = BoundExpr::Function {
        name: name.to_string(),
        args,
        return_type: TypeId::Null,
        distinct: false,
    };
    evaluate(&expr, &one_row_batch(), &[], &[]).map_err(|e| e.to_string())
}

#[test]
fn every_declared_scalar_name_is_dispatched() {
    let names = declared_scalar_names();
    assert!(
        names.len() > 300,
        "registry extraction looks broken, found only {} names",
        names.len()
    );
    let mut unknown: Vec<String> = Vec::new();
    for name in &names {
        // a zero arg call is enough to separate a dispatched name from an
        // unrecognized one, arm errors mention arity or types, the
        // fallthrough error alone says unknown function
        match call_scalar(name, vec![]) {
            Ok(_) => {}
            Err(msg) => {
                if msg.contains("unknown function") {
                    unknown.push(name.clone());
                }
            }
        }
    }
    assert!(
        unknown.is_empty(),
        "{} registry scalar names are not dispatched by the executor:\n{}",
        unknown.len(),
        unknown.join("\n")
    );
}
