//! Pipelines that run on change data.
//!
//! A pipeline created with `ON CHANGE DATA FROM stream [MIN ROWS n] [MAX
//! WAIT d]` runs when the stream holds at least `n` pending changes, or when
//! `d` has passed with at least one pending. The pending count is read off
//! the feeds' counters the way `zyron_sys.cdc.change_streams` reads it, so
//! deciding never touches the stream's position. The run itself is RUN
//! PIPELINE, whose stages move the position in their own commits

use std::collections::HashMap;
use std::sync::Arc;

use zyron_catalog::PipelineEntry;
use zyron_cdc::change_stream::ChangeStreamRuntime;
use zyron_parser::ast::ChangeDataTrigger;

use crate::connection::ServerState;
use crate::session::Session;

/// How long each pipeline has had changes pending, for MAX WAIT, and the
/// trigger each pipeline's definition carries
#[derive(Default)]
pub struct TriggerWatch {
    /// The instant, in microseconds, at which changes were first seen
    /// pending for each pipeline, cleared when a run fires or the stream
    /// drains
    waiting_since: HashMap<u32, i64>,
    /// The trigger parsed out of each pipeline's definition, beside the
    /// catalog entry it was parsed from. A redefinition replaces the entry,
    /// so it is parsed again, and an unchanged pipeline costs one pointer
    /// comparison per tick rather than a parse of its SQL
    triggers: HashMap<u32, (Arc<PipelineEntry>, Option<ChangeDataTrigger>)>,
}

impl TriggerWatch {
    pub fn new() -> Self {
        Self::default()
    }
}

/// The pipelines whose trigger fires now.
///
/// A pipeline that is disabled, whose definition no longer parses, or whose
/// stream is gone fires nothing. A stream with nothing pending resets the
/// pipeline's wait, so MAX WAIT counts from the first change that arrives
/// after a drain
pub fn due_pipelines(
    server: &Arc<ServerState>,
    watch: &mut TriggerWatch,
    now_micros: i64,
) -> Vec<Arc<PipelineEntry>> {
    let Some(feeds) = server.cdc_registry.as_ref() else {
        return Vec::new();
    };
    let runtime = ChangeStreamRuntime::new(Arc::clone(feeds));
    let mut due = Vec::new();
    let pipelines = server.catalog.list_pipelines();
    watch
        .triggers
        .retain(|id, _| pipelines.iter().any(|p| p.id == *id));
    for pipeline in pipelines {
        if !pipeline.enabled {
            watch.waiting_since.remove(&pipeline.id);
            continue;
        }
        let parsed = watch
            .triggers
            .entry(pipeline.id)
            .or_insert_with(|| (Arc::clone(&pipeline), trigger_of(&pipeline)));
        if !Arc::ptr_eq(&parsed.0, &pipeline) {
            *parsed = (Arc::clone(&pipeline), trigger_of(&pipeline));
        }
        let Some(trigger) = parsed.1.as_ref() else {
            watch.waiting_since.remove(&pipeline.id);
            continue;
        };
        let Some(stream) = stream_of(server, &pipeline, &trigger.stream) else {
            watch.waiting_since.remove(&pipeline.id);
            continue;
        };
        let pending = runtime.status(&stream, now_micros).pending_rows;
        if pending == 0 {
            watch.waiting_since.remove(&pipeline.id);
            continue;
        }
        let since = *watch.waiting_since.entry(pipeline.id).or_insert(now_micros);
        let waited_out = trigger
            .max_wait
            .as_ref()
            .map(|wait| now_micros.saturating_sub(since) >= duration_micros(wait))
            .unwrap_or(false);
        if pending >= trigger.min_rows || waited_out {
            watch.waiting_since.remove(&pipeline.id);
            due.push(pipeline);
        }
    }
    due
}

/// Runs every pipeline whose trigger fires now, answering with each run's
/// name and outcome.
///
/// A run that fails leaves the position where it was, the way a run a
/// person started would, and the failure is on the pipeline's own status
/// for the operator views. The next pass sees the changes still pending
/// and runs it again
pub async fn run_due_pipelines(
    server: &Arc<ServerState>,
    watch: &mut TriggerWatch,
    now_micros: i64,
) -> Vec<(String, Result<(), String>)> {
    let mut outcomes = Vec::new();
    for pipeline in due_pipelines(server, watch, now_micros) {
        let outcome = run_pipeline(server, &pipeline).await;
        outcomes.push((pipeline.name.clone(), outcome));
    }
    outcomes
}

/// Runs one pipeline the way RUN PIPELINE does, under a session in the
/// pipeline's own schema
async fn run_pipeline(server: &Arc<ServerState>, pipeline: &PipelineEntry) -> Result<(), String> {
    let schema = server
        .catalog
        .get_schema_by_id(pipeline.schema_id)
        .map_err(|e| e.to_string())?;
    let database = server
        .catalog
        .list_databases()
        .into_iter()
        .find(|db| db.id == schema.database_id)
        .map(|db| db.name.clone())
        .ok_or_else(|| format!("schema '{}' belongs to no database", schema.name))?;
    let mut session = Some(Session::new(
        "zyron".to_string(),
        database,
        schema.database_id,
    ));
    if let Some(session) = session.as_mut() {
        session.search_path = vec![schema.name.clone()];
    }
    let sql = format!("RUN PIPELINE {}", pipeline.name);
    let statement = zyron_parser::parse(&sql)
        .map_err(|e| e.to_string())?
        .into_iter()
        .next()
        .ok_or_else(|| "RUN PIPELINE parsed to nothing".to_string())?;
    let mut txn = None;
    let mut branch = None;
    match crate::ddl_dispatch::try_handle_ddl_utility(
        &statement,
        server,
        &mut session,
        &mut txn,
        &mut branch,
        &sql,
    )
    .await
    {
        Some(Ok(_)) => Ok(()),
        Some(Err(e)) => Err(e.to_string()),
        None => Err("RUN PIPELINE was not carried out by the dispatcher".to_string()),
    }
}

/// The trigger a pipeline's stored definition carries, when it carries one
pub fn trigger_of(pipeline: &PipelineEntry) -> Option<ChangeDataTrigger> {
    let parsed = zyron_parser::parse(&pipeline.definition_sql).ok()?;
    match parsed.into_iter().next()? {
        zyron_parser::Statement::CreatePipeline(create) => create.trigger,
        _ => None,
    }
}

/// The stream a pipeline's trigger names, resolved in the pipeline's own
/// schema first and by qualified name otherwise
fn stream_of(
    server: &Arc<ServerState>,
    pipeline: &PipelineEntry,
    name: &str,
) -> Option<Arc<zyron_catalog::ChangeStreamEntry>> {
    if let Some(stream) = server.catalog.get_change_stream(pipeline.schema_id, name) {
        return Some(stream);
    }
    let schema = server.catalog.get_schema_by_id(pipeline.schema_id).ok()?;
    server
        .catalog
        .resolve_change_stream(schema.database_id, name)
        .ok()
}

/// A wait as microseconds
fn duration_micros(duration: &zyron_parser::ast::TtlDuration) -> i64 {
    use zyron_parser::ast::TtlUnit;
    let unit_secs: i64 = match duration.unit {
        TtlUnit::Seconds => 1,
        TtlUnit::Minutes => 60,
        TtlUnit::Hours => 3600,
        TtlUnit::Days => 86400,
    };
    duration
        .value
        .saturating_mul(unit_secs)
        .saturating_mul(1_000_000)
}
