//! The user-authored objects an upgrade classifies and rewrites, read from
//! and written back to the catalog.
//!
//! Eight kinds carry SQL the catalog stores as text: views, materialized
//! views, functions, procedures, pipelines, schedules, streaming jobs, and
//! endpoints. The rest of `ObjectKind` has no catalog representation yet
//! and so nothing to rewrite.
//!
//! ## Writing back
//!
//! The rewriter produces a statement tree and the parser's unparser renders
//! it as SQL. Before that text replaces an object's SQL it is parsed again,
//! and only a rendering that comes back as one statement is written. A
//! statement the unparser has no spelling for renders as its tree, fails
//! that check, and is reported as computed but not written with the diff
//! for a person to apply. The check is what keeps a rewrite from ever
//! replacing an object's SQL with text no binary reads

use zyron_catalog::Catalog;
use zyron_common::Result;
use zyron_common::format::rewrite::{ObjectKind, RewriteStatus};
use zyron_common::format::{RewriteRecord, UpgradeBoard};
use zyron_parser::rewriter;

use super::compat_gate::UserObject;

/// What writing one object's rewritten SQL did
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum WriteBack {
    /// The catalog holds the rewritten SQL
    Written,
    /// The rewritten statement does not render as SQL this binary parses,
    /// so the object was left as it was
    NotRenderable { reason: String },
    /// The catalog has no write path for this kind, so the object was left
    /// as it was
    NoWriteApi { reason: String },
    /// No object of that name and kind exists any more
    Missing,
}

/// The qualified name an object is known by on the board
fn qualified(catalog: &Catalog, schema_id: zyron_catalog::SchemaId, name: &str) -> String {
    match catalog.get_schema_by_id(schema_id) {
        Ok(schema) => format!("{}.{}", schema.name, name),
        Err(_) => name.to_string(),
    }
}

/// Every object with stored SQL, in the shape the gate reads
pub fn collect_user_objects(catalog: &Catalog) -> Vec<UserObject> {
    let mut objects = Vec::new();
    for view in catalog.list_views() {
        objects.push(UserObject {
            name: qualified(catalog, view.schema_id, &view.name),
            kind: ObjectKind::View,
            sql: view.definition_sql.clone(),
        });
    }
    for mview in catalog.list_mviews() {
        objects.push(UserObject {
            name: qualified(catalog, mview.schema_id, &mview.name),
            kind: ObjectKind::MaterializedView,
            sql: mview.definition_sql.clone(),
        });
    }
    for function in catalog.list_functions() {
        objects.push(UserObject {
            name: qualified(catalog, function.schema_id, &function.name),
            kind: ObjectKind::Function,
            sql: function.body_sql.clone(),
        });
    }
    for procedure in catalog.list_procedures() {
        objects.push(UserObject {
            name: qualified(catalog, procedure.schema_id, &procedure.name),
            kind: ObjectKind::Procedure,
            sql: procedure.body_sql.clone(),
        });
    }
    for pipeline in catalog.list_pipelines() {
        objects.push(UserObject {
            name: qualified(catalog, pipeline.schema_id, &pipeline.name),
            kind: ObjectKind::Pipeline,
            sql: pipeline.definition_sql.clone(),
        });
    }
    for schedule in catalog.list_schedules() {
        objects.push(UserObject {
            name: qualified(catalog, schedule.schema_id, &schedule.name),
            kind: ObjectKind::Schedule,
            sql: schedule.body_sql.clone(),
        });
    }
    for job in catalog.list_streaming_jobs() {
        objects.push(UserObject {
            name: qualified(catalog, job.source_schema_id, &job.name),
            kind: ObjectKind::StreamingJob,
            sql: job.select_sql.clone(),
        });
    }
    for endpoint in catalog.list_endpoints() {
        objects.push(UserObject {
            name: qualified(catalog, endpoint.schema_id, &endpoint.name),
            kind: ObjectKind::Endpoint,
            sql: endpoint.sql_body.clone(),
        });
    }
    objects.sort_by(|a, b| a.name.cmp(&b.name).then_with(|| a.kind.cmp(&b.kind)));
    objects
}

/// Whether rewritten text is SQL this binary parses as exactly one
/// statement, which is the condition for it to replace an object's SQL
pub fn renderable(sql: &str) -> std::result::Result<(), String> {
    match zyron_parser::parse(sql) {
        Ok(statements) if statements.len() == 1 => Ok(()),
        Ok(statements) => Err(format!(
            "the rewritten text parses as {} statements rather than one",
            statements.len()
        )),
        Err(e) => Err(format!("the rewritten text does not parse as SQL, {e}")),
    }
}

/// Puts rewritten SQL in the catalog for one object, when it can be
pub async fn write_back(
    catalog: &Catalog,
    name: &str,
    kind: ObjectKind,
    sql: &str,
) -> Result<WriteBack> {
    if let Err(reason) = renderable(sql) {
        return Ok(WriteBack::NotRenderable { reason });
    }
    match kind {
        ObjectKind::View => {
            let Some(view) = catalog
                .list_views()
                .into_iter()
                .find(|v| qualified(catalog, v.schema_id, &v.name) == name)
            else {
                return Ok(WriteBack::Missing);
            };
            let mut entry = (*view).clone();
            entry.definition_sql = sql.to_string();
            catalog.create_view(entry, true).await?;
            Ok(WriteBack::Written)
        }
        ObjectKind::MaterializedView => Ok(WriteBack::NoWriteApi {
            reason: "a materialized view's definition is fixed at creation and owns its backing \
                     table, drop and recreate it"
                .into(),
        }),
        ObjectKind::Function => {
            let Some(function) = catalog
                .list_functions()
                .into_iter()
                .find(|f| qualified(catalog, f.schema_id, &f.name) == name)
            else {
                return Ok(WriteBack::Missing);
            };
            let mut entry = (*function).clone();
            entry.body_sql = sql.to_string();
            catalog.create_function(entry, true).await?;
            Ok(WriteBack::Written)
        }
        ObjectKind::Procedure => {
            let Some(procedure) = catalog
                .list_procedures()
                .into_iter()
                .find(|p| qualified(catalog, p.schema_id, &p.name) == name)
            else {
                return Ok(WriteBack::Missing);
            };
            let mut entry = (*procedure).clone();
            entry.body_sql = sql.to_string();
            catalog.create_procedure(entry, true).await?;
            Ok(WriteBack::Written)
        }
        ObjectKind::Pipeline => {
            let Some(pipeline) = catalog
                .list_pipelines()
                .into_iter()
                .find(|p| qualified(catalog, p.schema_id, &p.name) == name)
            else {
                return Ok(WriteBack::Missing);
            };
            let mut entry = (*pipeline).clone();
            entry.definition_sql = sql.to_string();
            catalog.update_pipeline(entry).await?;
            Ok(WriteBack::Written)
        }
        ObjectKind::Schedule => {
            let Some(schedule) = catalog
                .list_schedules()
                .into_iter()
                .find(|s| qualified(catalog, s.schema_id, &s.name) == name)
            else {
                return Ok(WriteBack::Missing);
            };
            let mut entry = (*schedule).clone();
            entry.body_sql = sql.to_string();
            catalog.update_schedule(entry).await?;
            Ok(WriteBack::Written)
        }
        ObjectKind::Endpoint => {
            let Some(endpoint) = catalog
                .list_endpoints()
                .into_iter()
                .find(|e| qualified(catalog, e.schema_id, &e.name) == name)
            else {
                return Ok(WriteBack::Missing);
            };
            let mut entry = (*endpoint).clone();
            entry.sql_body = sql.to_string();
            catalog.update_endpoint(entry).await?;
            Ok(WriteBack::Written)
        }
        ObjectKind::StreamingJob => Ok(WriteBack::NoWriteApi {
            reason: "a streaming job's query is fixed at creation, drop and recreate it".into(),
        }),
        other => Ok(WriteBack::NoWriteApi {
            reason: format!("{other} objects have no stored SQL to rewrite"),
        }),
    }
}

/// What applying a set of rewrites did
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct AppliedToCatalog {
    pub written: u32,
    pub not_written: u32,
}

/// Writes the SQL the rewrite pass produced for each object, and marks the
/// board's records with what happened to them
pub async fn write_rewritten(
    catalog: &Catalog,
    board: &UpgradeBoard,
    rewritten: &[(String, String)],
    objects: &[UserObject],
    now_secs: u64,
) -> AppliedToCatalog {
    let mut outcome = AppliedToCatalog::default();
    let mut records = board.rewrites();
    for (name, sql) in rewritten {
        let Some(kind) = objects.iter().find(|o| &o.name == name).map(|o| o.kind) else {
            continue;
        };
        let result = match write_back(catalog, name, kind, sql).await {
            Ok(result) => result,
            Err(e) => WriteBack::NoWriteApi {
                reason: e.to_string(),
            },
        };
        settle(&mut records, name, &result, &mut outcome, now_secs);
    }
    board.set_rewrites(records);
    outcome
}

/// Applies every acknowledged rewrite: the rewriter named on the record is
/// run over the object again and the result written back through the gate
pub async fn apply_acknowledged(
    catalog: &Catalog,
    board: &UpgradeBoard,
    now_secs: u64,
) -> AppliedToCatalog {
    let mut outcome = AppliedToCatalog::default();
    let objects = collect_user_objects(catalog);
    let mut records = board.rewrites();
    let acknowledged: Vec<(String, ObjectKind, String)> = records
        .iter()
        .filter(|r| r.status == RewriteStatus::Acknowledged)
        .map(|r| {
            (
                r.object_name.clone(),
                r.object_kind,
                r.rewriter_name.clone(),
            )
        })
        .collect();
    for (name, kind, rewriter_name) in acknowledged {
        let result = match objects.iter().find(|o| o.name == name && o.kind == kind) {
            Some(object) => match rewrite_with(&object.sql, kind, &rewriter_name) {
                Ok(sql) => match write_back(catalog, &name, kind, &sql).await {
                    Ok(result) => result,
                    Err(e) => WriteBack::NoWriteApi {
                        reason: e.to_string(),
                    },
                },
                Err(reason) => WriteBack::NotRenderable { reason },
            },
            None => WriteBack::Missing,
        };
        settle(&mut records, &name, &result, &mut outcome, now_secs);
    }
    board.set_rewrites(records);
    outcome
}

/// Runs one named rewriter over an object's SQL and renders the result
fn rewrite_with(
    sql: &str,
    kind: ObjectKind,
    rewriter_name: &str,
) -> std::result::Result<String, String> {
    let mut statements = zyron_parser::parse(sql).map_err(|e| e.to_string())?;
    if statements.len() != 1 {
        return Err(format!("the object holds {} statements", statements.len()));
    }
    let mut statement = statements.remove(0);
    let rewrite = rewriter::for_kind(kind)
        .into_iter()
        .find(|r| r.name == rewriter_name)
        .ok_or_else(|| format!("no rewriter called {rewriter_name} is registered"))?;
    (rewrite.rewriter)(&mut statement);
    Ok(rewriter::render(&statement))
}

/// Marks the records for one object with what its write-back did
fn settle(
    records: &mut [RewriteRecord],
    name: &str,
    result: &WriteBack,
    outcome: &mut AppliedToCatalog,
    now_secs: u64,
) {
    for record in records.iter_mut().filter(|r| r.object_name == name) {
        match result {
            WriteBack::Written => {
                record.status = RewriteStatus::Applied;
                outcome.written += 1;
            }
            WriteBack::NotRenderable { reason } | WriteBack::NoWriteApi { reason } => {
                record.status = RewriteStatus::Failed;
                record.diff = format!(
                    "computed but not written, {reason}. Apply by hand:\n{}",
                    record.diff
                );
                outcome.not_written += 1;
            }
            WriteBack::Missing => {
                record.status = RewriteStatus::Failed;
                record.diff = format!("the object no longer exists\n{}", record.diff);
                outcome.not_written += 1;
            }
        }
        record.updated_at_secs = now_secs;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rendered_sql_passes_the_gate_and_a_tree_dump_does_not() {
        assert!(renderable("SELECT 1").is_ok());
        let statements = zyron_parser::parse("CREATE VIEW s.v AS SELECT a FROM s.t WHERE a > 1")
            .expect("parses");
        let rendered = rewriter::render(&statements[0]);
        assert!(renderable(&rendered).is_ok(), "{rendered}");
        assert_eq!(
            zyron_parser::parse(&rendered).expect("parses again"),
            statements,
            "the rendering reads back as the same statement"
        );
        // A statement kind with no SQL spelling renders as its tree
        let schema = zyron_parser::parse("CREATE SCHEMA s").expect("parses");
        let dump = rewriter::render(&schema[0]);
        let err = renderable(&dump).expect_err("a tree dump is not SQL");
        assert!(err.contains("does not parse"), "{err}");
        let err = renderable("SELECT 1; SELECT 2").expect_err("two statements");
        assert!(err.contains("2 statements"), "{err}");
    }

    #[test]
    fn test_settling_marks_every_record_of_the_object() {
        let record = |name: &str| RewriteRecord {
            object_name: name.to_string(),
            object_kind: ObjectKind::View,
            rewriter_name: "r".into(),
            category: zyron_common::format::rewrite::RewriteCategory::Safe,
            status: RewriteStatus::Applied,
            before_hash: 1,
            after_hash: 2,
            acknowledged_by: String::new(),
            updated_at_secs: 0,
            diff: "-a\n+b".into(),
        };
        let mut records = vec![record("s.v1"), record("s.v1"), record("s.v2")];
        let mut outcome = AppliedToCatalog::default();
        settle(
            &mut records,
            "s.v1",
            &WriteBack::NotRenderable {
                reason: "no unparser".into(),
            },
            &mut outcome,
            77,
        );
        assert_eq!(outcome.not_written, 2);
        assert_eq!(records[0].status, RewriteStatus::Failed);
        assert!(records[0].diff.contains("Apply by hand"));
        assert_eq!(records[0].updated_at_secs, 77);
        assert_eq!(
            records[2].status,
            RewriteStatus::Applied,
            "another object is untouched"
        );
        settle(&mut records, "s.v2", &WriteBack::Written, &mut outcome, 78);
        assert_eq!(outcome.written, 1);
    }

    #[test]
    fn test_an_unknown_rewriter_is_refused_by_name() {
        let err =
            rewrite_with("SELECT 1", ObjectKind::View, "nothing_registered").expect_err("refused");
        assert!(err.contains("nothing_registered"), "{err}");
    }
}
