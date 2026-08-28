//! The `zyron_sys.streaming.*` views.
//!
//! Four of them read the process-wide streaming metrics registry, which the
//! job runners publish into once per poll cycle. The fifth reads the
//! catalog's streaming job entries, which is the definition rather than the
//! runtime: a job that is not running still has a row there, and that is the
//! point of separating it from `zyron_sys.stat.streaming_jobs`.

use zyron_common::ZyronError;
use zyron_streaming::metrics::global_metrics;

use crate::connection::ServerState;
use crate::system_views::{ViewRows, make_field};
use crate::types::{PG_INT4_OID, PG_INT8_OID, PG_TEXT_OID};

fn cell(value: impl ToString) -> Option<Vec<u8>> {
    Some(value.to_string().into_bytes())
}

/// Dispatches the `zyron_sys.streaming.*` entities. Reached only for a name
/// the registry recognized.
pub fn build(object: &str, server: &ServerState) -> Result<ViewRows, ZyronError> {
    Ok(match object {
        "watermarks" => build_watermarks(),
        "checkpoint_history" => build_checkpoint_history(),
        "backpressure" => build_backpressure(),
        "operator_metrics" => build_operator_metrics(),
        "jobs" => build_jobs(server),
        other => {
            return Err(ZyronError::Internal(format!(
                "`zyron_sys.streaming.{}` is registered but has no builder",
                other
            )));
        }
    })
}

/// Builds zyron_sys.streaming.watermarks.
/// Columns: source_id, current_watermark_ms, last_updated_ms,
///          global_watermark_ms.
///
/// The global watermark is repeated on every row rather than split into its
/// own view: a source's watermark only means something next to the global
/// one it is holding back or keeping up with.
fn build_watermarks() -> ViewRows {
    let fields = vec![
        make_field("source_id", PG_INT4_OID, 4),
        make_field("current_watermark_ms", PG_INT8_OID, 8),
        make_field("last_updated_ms", PG_INT8_OID, 8),
        make_field("global_watermark_ms", PG_INT8_OID, 8),
    ];
    let registry = global_metrics();
    let global = registry
        .global_watermark_ms
        .load(std::sync::atomic::Ordering::Relaxed);
    let mut views = registry.watermark_views();
    views.sort_by_key(|v| v.source_id);
    let rows = views
        .into_iter()
        .map(|v| {
            vec![
                cell(v.source_id),
                cell(v.current_watermark_ms),
                cell(v.last_updated_ms),
                // i64::MIN is the unset marker the registry starts at, which
                // is not a time and is reported as absent rather than as the
                // year it would print as
                if global == i64::MIN {
                    None
                } else {
                    cell(global)
                },
            ]
        })
        .collect();
    (fields, rows)
}

/// Builds zyron_sys.streaming.checkpoint_history.
/// Columns: job_name, checkpoint_id, checkpoint_time_ms, duration_ms,
///          size_bytes, status.
///
/// Oldest first, bounded to the window the registry keeps.
fn build_checkpoint_history() -> ViewRows {
    let fields = vec![
        make_field("job_name", PG_TEXT_OID, -1),
        make_field("checkpoint_id", PG_INT8_OID, 8),
        make_field("checkpoint_time_ms", PG_INT8_OID, 8),
        make_field("duration_ms", PG_INT8_OID, 8),
        make_field("size_bytes", PG_INT8_OID, 8),
        make_field("status", PG_TEXT_OID, -1),
    ];
    let rows = global_metrics()
        .checkpoint_history()
        .into_iter()
        .map(|c| {
            vec![
                cell(&c.job_name),
                cell(c.checkpoint_id),
                cell(c.checkpoint_time_ms),
                cell(c.duration_ms),
                cell(c.size_bytes),
                cell(c.status),
            ]
        })
        .collect();
    (fields, rows)
}

/// Builds zyron_sys.streaming.backpressure.
/// Columns: job_name, operator_name, operator_id, queue_usage,
///          queue_capacity, ratio.
///
/// A ratio at 1.0 means the last cycle filled its read budget, which is the
/// observable form of a source producing faster than the pipeline drains it.
fn build_backpressure() -> ViewRows {
    let fields = vec![
        make_field("job_name", PG_TEXT_OID, -1),
        make_field("operator_name", PG_TEXT_OID, -1),
        make_field("operator_id", PG_INT4_OID, 4),
        make_field("queue_usage", PG_INT8_OID, 8),
        make_field("queue_capacity", PG_INT8_OID, 8),
        make_field("ratio", PG_TEXT_OID, -1),
    ];
    let mut views = global_metrics().backpressure_views();
    views.sort_by(|a, b| (&a.job_name, a.operator_id).cmp(&(&b.job_name, b.operator_id)));
    let rows = views
        .into_iter()
        .map(|v| {
            vec![
                cell(&v.job_name),
                cell(&v.operator_name),
                cell(v.operator_id),
                cell(v.queue_usage),
                cell(v.queue_capacity),
                cell(format!("{:.4}", v.ratio)),
            ]
        })
        .collect();
    (fields, rows)
}

/// Builds zyron_sys.streaming.operator_metrics.
/// Columns: job_name, operator_name, operator_id, input_records_total,
///          output_records_total, processing_time_ns_total,
///          avg_processing_time_ns, current_watermark_ms.
fn build_operator_metrics() -> ViewRows {
    let fields = vec![
        make_field("job_name", PG_TEXT_OID, -1),
        make_field("operator_name", PG_TEXT_OID, -1),
        make_field("operator_id", PG_INT4_OID, 4),
        make_field("input_records_total", PG_INT8_OID, 8),
        make_field("output_records_total", PG_INT8_OID, 8),
        make_field("processing_time_ns_total", PG_INT8_OID, 8),
        make_field("avg_processing_time_ns", PG_INT8_OID, 8),
        make_field("current_watermark_ms", PG_INT8_OID, 8),
    ];
    let mut views = global_metrics().operator_metrics_views();
    views.sort_by(|a, b| (&a.job_name, a.operator_id).cmp(&(&b.job_name, b.operator_id)));
    let rows = views
        .into_iter()
        .map(|v| {
            let avg = v.avg_processing_time_ns();
            vec![
                cell(&v.job_name),
                cell(&v.operator_name),
                cell(v.operator_id),
                cell(v.input_records_total),
                cell(v.output_records_total),
                cell(v.processing_time_ns_total),
                cell(avg),
                if v.current_watermark_ms == i64::MIN {
                    None
                } else {
                    cell(v.current_watermark_ms)
                },
            ]
        })
        .collect();
    (fields, rows)
}

/// Builds zyron_sys.streaming.jobs.
/// Columns: job_id, job_name, source_table_id, source_table, target_table_id,
///          target_table, write_mode, status, created_at, last_error,
///          select_sql.
///
/// The catalog's definition of every declared job. Distinct from
/// zyron_sys.stat.streaming_jobs, which reports what the job manager is
/// running right now: a job can be defined and stopped, or running and
/// mid-restart, and reading one view for both would hide which.
fn build_jobs(server: &ServerState) -> ViewRows {
    let fields = vec![
        make_field("job_id", PG_INT4_OID, 4),
        make_field("job_name", PG_TEXT_OID, -1),
        make_field("source_table_id", PG_INT4_OID, 4),
        make_field("source_table", PG_TEXT_OID, -1),
        make_field("target_table_id", PG_INT4_OID, 4),
        make_field("target_table", PG_TEXT_OID, -1),
        make_field("write_mode", PG_TEXT_OID, -1),
        make_field("status", PG_TEXT_OID, -1),
        make_field("created_at", PG_INT8_OID, 8),
        make_field("last_error", PG_TEXT_OID, -1),
        make_field("select_sql", PG_TEXT_OID, -1),
    ];
    let name_of = |id: zyron_catalog::TableId| -> String {
        server
            .catalog
            .get_table_by_id(id)
            .map(|t| t.name.clone())
            .unwrap_or_else(|_| format!("table_{}", id.0))
    };
    let mut jobs = server.catalog.list_streaming_jobs();
    jobs.sort_by_key(|j| j.id.0);
    let rows = jobs
        .into_iter()
        .map(|job| {
            vec![
                cell(job.id.0),
                cell(&job.name),
                cell(job.source_table_id.0),
                cell(name_of(job.source_table_id)),
                cell(job.target_table_id.0),
                cell(name_of(job.target_table_id)),
                cell(format!("{:?}", job.write_mode)),
                cell(format!("{:?}", job.status)),
                cell(job.created_at),
                job.last_error.as_ref().map(|e| e.as_bytes().to_vec()),
                cell(&job.select_sql),
            ]
        })
        .collect();
    (fields, rows)
}
