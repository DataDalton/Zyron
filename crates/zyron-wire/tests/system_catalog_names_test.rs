//! Every canonical system name parses and dispatches.

use zyron_catalog::system_catalog::{SYSTEM_OBJECTS, SystemObjectKind};

#[test]
fn every_canonical_name_parses_as_a_relation() {
    let mut failures = Vec::new();
    for object in SYSTEM_OBJECTS {
        if object.kind != SystemObjectKind::View {
            continue;
        }
        let name = object.canonical_name();
        let sql = format!("SELECT * FROM {}", name);
        match zyron_parser::parse(&sql) {
            Ok(stmts) => match stmts.into_iter().next() {
                Some(zyron_parser::ast::Statement::Select(sel)) => match sel.from.first() {
                    Some(zyron_parser::TableRef::Table { name: parsed, .. }) => {
                        if parsed != &name {
                            failures.push(format!("{} parsed as {}", name, parsed));
                        }
                    }
                    other => failures.push(format!("{} became {:?}", name, other)),
                },
                other => failures.push(format!("{} became {:?}", name, other)),
            },
            Err(e) => failures.push(format!("{} did not parse: {}", name, e)),
        }
    }
    assert!(failures.is_empty(), "{:#?}", failures);
}

#[test]
fn every_table_function_parses_as_a_call() {
    let mut failures = Vec::new();
    for object in SYSTEM_OBJECTS {
        if object.kind != SystemObjectKind::TableFunction {
            continue;
        }
        let name = object.canonical_name();
        let sql = format!("SELECT * FROM {}('x')", name);
        match zyron_parser::parse(&sql) {
            Ok(stmts) => match stmts.into_iter().next() {
                Some(zyron_parser::ast::Statement::Select(sel)) => match sel.from.first() {
                    Some(zyron_parser::TableRef::TableFunction(call)) => {
                        if call.name != name {
                            failures.push(format!("{} parsed as {}", name, call.name));
                        }
                    }
                    other => failures.push(format!("{} became {:?}", name, other)),
                },
                other => failures.push(format!("{} became {:?}", name, other)),
            },
            Err(e) => failures.push(format!("{} did not parse: {}", name, e)),
        }
    }
    assert!(failures.is_empty(), "{:#?}", failures);
}
