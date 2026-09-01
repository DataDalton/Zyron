//! The `zyron_sys.retention.*` dashboard views.
//!
//! Four views over the retention machinery: row age distribution per table,
//! the next actions each policy will take, what purging now would reclaim,
//! and how the last executed jobs went. Age distributions run real
//! aggregation queries over each table's lifecycle age column at read time,
//! byte figures multiply row counts by the ANALYZE average row size, or a
//! schema derived estimate when the table was never analyzed.

use std::sync::Arc;

use zyron_common::ZyronError;
use zyron_executor::column::ScalarValue;

use crate::connection::ServerState;
use crate::system_views::{ViewRows, make_field};
use crate::types::{PG_BOOL_OID, PG_INT4_OID, PG_INT8_OID, PG_TEXT_OID};

fn cell(value: impl ToString) -> Option<Vec<u8>> {
    Some(value.to_string().into_bytes())
}

fn null_cell() -> Option<Vec<u8>> {
    None
}

/// Dispatches one `zyron_sys.retention` view to its builder.
pub async fn build(object: &str, server: &ServerState) -> Result<ViewRows, ZyronError> {
    match object {
        "storage_by_age" => build_storage_by_age(server).await,
        "upcoming_actions" => build_upcoming_actions(server).await,
        "savings_estimate" => build_savings_estimate(server).await,
        "compliance_summary" => build_compliance_summary(server).await,
        other => Err(ZyronError::Internal(format!(
            "zyron_sys.retention.{other} is registered but has no builder"
        ))),
    }
}

const MICROS_PER_DAY: i64 = 86_400 * 1_000_000;

/// Age bucket upper bounds in days, oldest bucket is open ended
const BUCKET_DAYS: [i64; 4] = [1, 7, 30, 90];
const BUCKET_LABELS: [&str; 5] = ["0d-1d", "1d-7d", "7d-30d", "30d-90d", "90d+"];

fn now_micros() -> i64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_micros() as i64)
        .unwrap_or(0)
}

/// A table the retention views report on, with its resolved age column
struct AgeTarget {
    table_id: u32,
    table_name: String,
    schema_id: zyron_catalog::SchemaId,
    /// Column holding the age timestamp in microseconds, when one is set
    age_column: Option<String>,
    /// Comparison cutoff semantics: true when the column holds an expiry
    /// time compared against now, false when it holds a creation time
    /// compared against now minus the ttl
    column_is_expiry: bool,
    ttl_seconds: i64,
}

/// Resolves the age column for a table the way the retention reaper does:
/// an explicit retention column holds expiry timestamps, otherwise the ttl
/// column holds creation timestamps aged against ttl_seconds
fn age_target(table: &zyron_catalog::TableEntry) -> AgeTarget {
    let lc = &table.lifecycle;
    let (age_column, column_is_expiry, ttl_seconds) =
        if zyron_catalog::schema::LifecycleConfig::column_is_set(lc.retention_column_id) {
            let name = table
                .columns
                .iter()
                .find(|c| c.id.0 as u32 == lc.retention_column_id)
                .map(|c| c.name.clone());
            (name, true, 0)
        } else if zyron_catalog::schema::LifecycleConfig::column_is_set(lc.ttl_column_id)
            && lc.ttl_seconds > 0
        {
            let name = table
                .columns
                .iter()
                .find(|c| c.id.0 as u32 == lc.ttl_column_id)
                .map(|c| c.name.clone());
            (name, false, lc.ttl_seconds)
        } else {
            (None, false, 0)
        };
    AgeTarget {
        table_id: table.id.0,
        table_name: table.name.clone(),
        schema_id: table.schema_id,
        age_column,
        column_is_expiry,
        ttl_seconds,
    }
}

/// Tables the retention views cover: every table with a retention policy
/// plus every table with a lifecycle age column configured
async fn collect_targets(server: &ServerState) -> Result<Vec<AgeTarget>, ZyronError> {
    let policies = server.catalog.load_retention_policies().await?;
    let mut targets: Vec<AgeTarget> = Vec::new();
    let mut seen: std::collections::HashSet<u32> = std::collections::HashSet::new();
    for policy in &policies {
        if let Ok(table) = server
            .catalog
            .get_table_by_id(zyron_catalog::TableId(policy.table_id))
            && seen.insert(table.id.0)
        {
            targets.push(age_target(&table));
        }
    }
    for table in server.catalog.list_all_tables() {
        let target = age_target(&table);
        if target.age_column.is_some() && seen.insert(table.id.0) {
            targets.push(target);
        }
    }
    targets.sort_by_key(|t| t.table_id);
    Ok(targets)
}

/// Average bytes per row: the ANALYZE figure when present, otherwise the
/// sum of fixed column widths with 16 bytes assumed per varlen column
fn avg_row_bytes(server: &ServerState, table_id: u32) -> u64 {
    if let Some(stats) = server.catalog.get_stats(zyron_catalog::TableId(table_id))
        && stats.0.avg_row_size > 0
    {
        return stats.0.avg_row_size as u64;
    }
    match server
        .catalog
        .get_table_by_id(zyron_catalog::TableId(table_id))
    {
        Ok(table) => table
            .columns
            .iter()
            .map(|c| c.physical_type_id().fixed_size().unwrap_or(16) as u64)
            .sum::<u64>()
            .max(1),
        Err(_) => 1,
    }
}

fn namespace_for_schema(
    server: &ServerState,
    schema_id: zyron_catalog::SchemaId,
) -> (zyron_catalog::DatabaseId, Vec<String>) {
    match server.catalog.get_schema_by_id(schema_id) {
        Ok(s) => (s.database_id, vec![s.name.clone()]),
        Err(_) => (
            zyron_catalog::DatabaseId(1),
            zyron_catalog::default_search_path(),
        ),
    }
}

/// Runs one read only statement against a table's own schema namespace and
/// returns the result batches. The transaction is aborted afterwards, a
/// read never holds anything
async fn run_select(
    server: &ServerState,
    ns: (zyron_catalog::DatabaseId, Vec<String>),
    sql: &str,
) -> Result<Vec<zyron_executor::batch::DataBatch>, ZyronError> {
    let stmts = zyron_parser::parse(sql)?;
    let stmt = stmts
        .into_iter()
        .next()
        .ok_or_else(|| ZyronError::Internal("empty retention view query".into()))?;
    let plan = zyron_planner::plan(
        &server.catalog,
        ns.0,
        ns.1,
        stmt,
        Some(&server.peer_facts()),
    )
    .await?;
    let mut txn = server
        .txn_manager
        .begin(zyron_storage::txn::IsolationLevel::ReadCommitted)?;
    let snapshot = txn.snapshot.clone();
    let txn_id = txn.txn_id;
    let ctx = zyron_executor::context::ExecutionContext::new(
        Arc::clone(&server.catalog),
        Arc::clone(&server.wal),
        Arc::clone(&server.buffer_pool),
        Arc::clone(&server.disk_manager),
        txn_id,
        snapshot,
    );
    let ctx = Arc::new(ctx);
    let result = zyron_executor::execute(plan, &ctx).await;
    let _ = server.txn_manager.abort(&mut txn);
    result
}

fn scalar_i64(value: ScalarValue) -> Option<i64> {
    match value {
        ScalarValue::Int8(v) => Some(v as i64),
        ScalarValue::Int16(v) => Some(v as i64),
        ScalarValue::Int32(v) => Some(v as i64),
        ScalarValue::Int64(v) => Some(v),
        ScalarValue::Int128(v) => i64::try_from(v).ok(),
        ScalarValue::UInt8(v) => Some(v as i64),
        ScalarValue::UInt16(v) => Some(v as i64),
        ScalarValue::UInt32(v) => Some(v as i64),
        ScalarValue::UInt64(v) => i64::try_from(v).ok(),
        ScalarValue::Float32(v) => Some(v as i64),
        ScalarValue::Float64(v) => Some(v as i64),
        _ => None,
    }
}

/// First row of a result as i64 values, one per selected column
fn first_row_i64(batches: &[zyron_executor::batch::DataBatch], columns: usize) -> Vec<Option<i64>> {
    for batch in batches {
        if batch.num_rows > 0 {
            return (0..columns)
                .map(|c| {
                    if c < batch.columns.len() {
                        scalar_i64(batch.column(c).get_scalar(0))
                    } else {
                        None
                    }
                })
                .collect();
        }
    }
    vec![None; columns]
}

fn action_name(action: u8) -> &'static str {
    match action {
        0 => "delete",
        1 => "archive",
        2 => "anonymize",
        _ => "unknown",
    }
}

fn kind_name(kind: u8) -> &'static str {
    match kind {
        0 => "ttl",
        1 => "cold_tiering",
        2 => "archive",
        _ => "unknown",
    }
}

async fn build_storage_by_age(server: &ServerState) -> Result<ViewRows, ZyronError> {
    let fields = vec![
        make_field("table_id", PG_INT4_OID, 4),
        make_field("table_name", PG_TEXT_OID, -1),
        make_field("age_bucket", PG_TEXT_OID, -1),
        make_field("byte_count", PG_INT8_OID, 8),
        make_field("row_count", PG_INT8_OID, 8),
    ];
    let now = now_micros();
    let mut rows: Vec<Vec<Option<Vec<u8>>>> = Vec::new();
    for target in collect_targets(server).await? {
        let per_row = avg_row_bytes(server, target.table_id);
        let Some(col) = target.age_column.as_deref() else {
            // No age column configured, report the whole table as unknown age
            let sql = format!("SELECT COUNT(*) FROM \"{}\"", target.table_name);
            let ns = namespace_for_schema(server, target.schema_id);
            let count = match run_select(server, ns, &sql).await {
                Ok(batches) => first_row_i64(&batches, 1)[0].unwrap_or(0),
                Err(_) => 0,
            };
            rows.push(vec![
                cell(target.table_id),
                cell(&target.table_name),
                cell("unknown"),
                cell(count.max(0) as u64 * per_row),
                cell(count.max(0)),
            ]);
            continue;
        };
        // Age buckets relative to now over the configured timestamp column
        let bounds: Vec<i64> = BUCKET_DAYS
            .iter()
            .map(|d| now - d * MICROS_PER_DAY)
            .collect();
        let mut selects: Vec<String> = Vec::with_capacity(BUCKET_LABELS.len() + 1);
        selects.push(format!(
            "SUM(CASE WHEN \"{col}\" >= {} THEN 1 ELSE 0 END)",
            bounds[0]
        ));
        for i in 1..BUCKET_DAYS.len() {
            selects.push(format!(
                "SUM(CASE WHEN \"{col}\" < {} AND \"{col}\" >= {} THEN 1 ELSE 0 END)",
                bounds[i - 1],
                bounds[i]
            ));
        }
        selects.push(format!(
            "SUM(CASE WHEN \"{col}\" < {} THEN 1 ELSE 0 END)",
            bounds[BUCKET_DAYS.len() - 1]
        ));
        selects.push(format!(
            "SUM(CASE WHEN \"{col}\" IS NULL THEN 1 ELSE 0 END)"
        ));
        let sql = format!(
            "SELECT {} FROM \"{}\"",
            selects.join(", "),
            target.table_name
        );
        let ns = namespace_for_schema(server, target.schema_id);
        let counts = match run_select(server, ns, &sql).await {
            Ok(batches) => first_row_i64(&batches, BUCKET_LABELS.len() + 1),
            Err(_) => vec![None; BUCKET_LABELS.len() + 1],
        };
        for (i, label) in BUCKET_LABELS.iter().enumerate() {
            let count = counts.get(i).copied().flatten().unwrap_or(0).max(0);
            rows.push(vec![
                cell(target.table_id),
                cell(&target.table_name),
                cell(label),
                cell(count as u64 * per_row),
                cell(count),
            ]);
        }
        let unknown = counts
            .get(BUCKET_LABELS.len())
            .copied()
            .flatten()
            .unwrap_or(0)
            .max(0);
        if unknown > 0 {
            rows.push(vec![
                cell(target.table_id),
                cell(&target.table_name),
                cell("unknown"),
                cell(unknown as u64 * per_row),
                cell(unknown),
            ]);
        }
    }
    Ok((fields, rows))
}

/// The expired row count and earliest upcoming expiry for one TTL target
async fn expiry_figures(server: &ServerState, target: &AgeTarget) -> (Option<i64>, Option<i64>) {
    let Some(col) = target.age_column.as_deref() else {
        return (None, None);
    };
    let now = now_micros();
    let cutoff = if target.column_is_expiry {
        now
    } else {
        now - target.ttl_seconds.saturating_mul(1_000_000)
    };
    let sql = format!(
        "SELECT SUM(CASE WHEN \"{col}\" < {cutoff} THEN 1 ELSE 0 END), MIN(CASE WHEN \"{col}\" >= {cutoff} THEN \"{col}\" ELSE NULL END) FROM \"{}\"",
        target.table_name
    );
    let ns = namespace_for_schema(server, target.schema_id);
    match run_select(server, ns, &sql).await {
        Ok(batches) => {
            let values = first_row_i64(&batches, 2);
            let expired = values[0];
            let next_expiry = values[1].map(|oldest_live| {
                if target.column_is_expiry {
                    oldest_live
                } else {
                    oldest_live + target.ttl_seconds.saturating_mul(1_000_000)
                }
            });
            (expired, next_expiry)
        }
        Err(_) => (None, None),
    }
}

async fn build_upcoming_actions(server: &ServerState) -> Result<ViewRows, ZyronError> {
    let fields = vec![
        make_field("table_id", PG_INT4_OID, 4),
        make_field("table_name", PG_TEXT_OID, -1),
        make_field("policy_kind", PG_TEXT_OID, -1),
        make_field("action", PG_TEXT_OID, -1),
        make_field("next_expiry_at", PG_INT8_OID, 8),
        make_field("row_count_impact_estimate", PG_INT8_OID, 8),
    ];
    let policies = server.catalog.load_retention_policies().await?;
    let mut rows: Vec<Vec<Option<Vec<u8>>>> = Vec::new();
    for policy in policies {
        let Ok(table) = server
            .catalog
            .get_table_by_id(zyron_catalog::TableId(policy.table_id))
        else {
            continue;
        };
        let target = age_target(&table);
        let (expired, next_expiry) = if policy.kind == 0 {
            expiry_figures(server, &target).await
        } else {
            (None, None)
        };
        rows.push(vec![
            cell(policy.table_id),
            cell(&table.name),
            cell(kind_name(policy.kind)),
            cell(action_name(policy.action)),
            next_expiry.map_or_else(null_cell, cell),
            expired.map_or_else(null_cell, cell),
        ]);
    }
    rows.sort_by(|a, b| a[0].cmp(&b[0]));
    Ok((fields, rows))
}

async fn build_savings_estimate(server: &ServerState) -> Result<ViewRows, ZyronError> {
    let fields = vec![
        make_field("table_id", PG_INT4_OID, 4),
        make_field("table_name", PG_TEXT_OID, -1),
        make_field("if_purged_now_bytes", PG_INT8_OID, 8),
        make_field("if_purged_now_rows", PG_INT8_OID, 8),
    ];
    let policies = server.catalog.load_retention_policies().await?;
    let mut rows: Vec<Vec<Option<Vec<u8>>>> = Vec::new();
    for policy in policies.iter().filter(|p| p.kind == 0) {
        let Ok(table) = server
            .catalog
            .get_table_by_id(zyron_catalog::TableId(policy.table_id))
        else {
            continue;
        };
        let target = age_target(&table);
        let (expired, _) = expiry_figures(server, &target).await;
        let expired_rows = expired.unwrap_or(0).max(0);
        let per_row = avg_row_bytes(server, policy.table_id);
        rows.push(vec![
            cell(policy.table_id),
            cell(&table.name),
            cell(expired_rows as u64 * per_row),
            cell(expired_rows),
        ]);
    }
    rows.sort_by(|a, b| a[0].cmp(&b[0]));
    Ok((fields, rows))
}

async fn build_compliance_summary(server: &ServerState) -> Result<ViewRows, ZyronError> {
    let fields = vec![
        make_field("table_id", PG_INT4_OID, 4),
        make_field("table_name", PG_TEXT_OID, -1),
        make_field("policy_name", PG_TEXT_OID, -1),
        make_field("last_run", PG_INT8_OID, 8),
        make_field("last_purged_bytes", PG_INT8_OID, 8),
        make_field("ok", PG_BOOL_OID, 1),
        make_field("message", PG_TEXT_OID, -1),
    ];
    let policies = server.catalog.load_retention_policies().await?;
    let jobs = server.catalog.load_retention_jobs().await?;
    let mut rows: Vec<Vec<Option<Vec<u8>>>> = Vec::new();
    for policy in policies {
        let table_name = server
            .catalog
            .get_table_by_id(zyron_catalog::TableId(policy.table_id))
            .map(|t| t.name.clone())
            .unwrap_or_else(|_| format!("table_{}", policy.table_id));
        let policy_name = format!("{}_{}", kind_name(policy.kind), action_name(policy.action));
        let last_job = jobs
            .iter()
            .filter(|j| j.table_id == policy.table_id)
            .max_by_key(|j| j.finished_at);
        let per_row = avg_row_bytes(server, policy.table_id);
        match last_job {
            Some(job) => {
                let ok = job.status == 2;
                let message = if job.detail.is_empty() {
                    format!("last job finished with status {}", job.status)
                } else {
                    job.detail.clone()
                };
                rows.push(vec![
                    cell(policy.table_id),
                    cell(&table_name),
                    cell(&policy_name),
                    cell(job.finished_at),
                    cell(job.rows_affected * per_row),
                    cell(ok),
                    cell(message),
                ]);
            }
            None => {
                rows.push(vec![
                    cell(policy.table_id),
                    cell(&table_name),
                    cell(&policy_name),
                    null_cell(),
                    cell(0u64),
                    cell(true),
                    cell("no retention jobs recorded yet"),
                ]);
            }
        }
    }
    rows.sort_by(|a, b| a[0].cmp(&b[0]));
    Ok((fields, rows))
}
