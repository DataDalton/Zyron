//! The `zyron_sys.compliance.report(kind)` table function.
//!
//! One shape for four reports, because an operator asking "is this node
//! compliant" wants one result set to read rather than four with different
//! columns. The `kind` argument picks which evidence the rows describe:
//! retention policies against the regulatory floors, legal holds, the
//! tamper-evident audit chain, or the raw compliance events.

use std::sync::Arc;

use zyron_common::ZyronError;
use zyron_lifecycle::compliance::{RetentionRequirement, event, validate_retention};

use crate::connection::ServerState;
use crate::system_views::{ViewRows, make_field};
use crate::types::{PG_INT8_OID, PG_TEXT_OID};

/// The reports `kind` can name, in the order the help text lists them.
const KINDS: &[&str] = &["retention", "legal_hold", "audit", "events"];

fn cell(value: impl ToString) -> Option<Vec<u8>> {
    Some(value.to_string().into_bytes())
}

/// Runs `zyron_sys.compliance.report(kind)`.
///
/// Columns: kind, subject, table_id, status, detail, event_time.
///
/// Status is one of PASS, FAIL, or INFO. A report with no FAIL row is the
/// node saying it found nothing to flag under that heading, which is a
/// different statement from a report that could not be produced: an
/// unrecognized kind is an error, never an empty PASS.
pub async fn call(args: &[String], server: &ServerState) -> Result<ViewRows, ZyronError> {
    if args.len() != 1 {
        return Err(ZyronError::PlanError(format!(
            "zyron_sys.compliance.report(kind) takes one argument, one of {}",
            KINDS.join(", ")
        )));
    }
    let kind = args[0].trim().to_ascii_lowercase();

    let fields = vec![
        make_field("kind", PG_TEXT_OID, -1),
        make_field("subject", PG_TEXT_OID, -1),
        make_field("table_id", PG_INT8_OID, 8),
        make_field("status", PG_TEXT_OID, -1),
        make_field("detail", PG_TEXT_OID, -1),
        make_field("event_time", PG_INT8_OID, 8),
    ];

    let rows = match kind.as_str() {
        "retention" => retention_rows(server).await?,
        "legal_hold" => legal_hold_rows(server).await?,
        "audit" => audit_rows(server).await?,
        "events" => event_rows(server).await?,
        other => {
            return Err(ZyronError::PlanError(format!(
                "zyron_sys.compliance.report: `{}` is not a report kind, expected one of {}",
                other,
                KINDS.join(", ")
            )));
        }
    };
    Ok((fields, rows))
}

type Row = Vec<Option<Vec<u8>>>;

/// The regulatory floors and ceilings every retention policy is checked
/// against. Shipped requirements rather than configured ones: a node cannot
/// opt out of SOX by not declaring it.
fn requirements() -> Vec<RetentionRequirement> {
    vec![
        RetentionRequirement::sox_financial(),
        RetentionRequirement::gdpr_pii(),
    ]
}

/// Each retention policy checked against the requirements for the category
/// its table's classification puts it in.
async fn retention_rows(server: &ServerState) -> Result<Vec<Row>, ZyronError> {
    let policies = server.catalog.load_retention_policies().await?;
    let reqs = requirements();
    let mut rows = Vec::with_capacity(policies.len());
    for policy in policies {
        let table = server
            .catalog
            .get_table_by_id(zyron_catalog::TableId(policy.table_id))
            .ok();
        let table_name = table
            .as_ref()
            .map(|t| t.name.clone())
            .unwrap_or_else(|| format!("table_{}", policy.table_id));
        // The category a policy is judged under comes from the strongest
        // classification any of the table's columns carries. A table with no
        // classified column is reported, not silently passed: an
        // unclassified table is exactly the case a review is looking for
        let category = table_category(server, policy.table_id);
        let (status, detail) = if category == "unclassified" {
            (
                "INFO",
                format!(
                    "{} retention of {}s is not checked: the table declares no data classification",
                    retention_kind(policy.kind),
                    policy.interval_seconds
                ),
            )
        } else {
            match validate_retention(category, policy.interval_seconds, &reqs) {
                Ok(()) => (
                    "PASS",
                    format!(
                        "{} retention of {}s satisfies every `{}` requirement",
                        retention_kind(policy.kind),
                        policy.interval_seconds,
                        category
                    ),
                ),
                Err(e) => ("FAIL", e.to_string()),
            }
        };
        rows.push(vec![
            cell("retention"),
            cell(table_name),
            cell(policy.table_id),
            cell(status),
            cell(detail),
            None,
        ]);
    }
    Ok(rows)
}

/// Every legal hold, active or released. A released hold is INFO rather than
/// absent, because the record of one having existed is itself the artifact.
async fn legal_hold_rows(server: &ServerState) -> Result<Vec<Row>, ZyronError> {
    let holds = server.catalog.load_legal_holds().await?;
    Ok(holds
        .into_iter()
        .map(|hold| {
            let active = hold.released_at == 0;
            let scope = if hold.predicate_sql.is_empty() {
                "the whole table".to_string()
            } else {
                format!("rows matching {}", hold.predicate_sql)
            };
            vec![
                cell("legal_hold"),
                cell(&hold.name),
                cell(hold.table_id),
                cell(if active { "PASS" } else { "INFO" }),
                cell(format!(
                    "{} on {}: {}",
                    if active { "held" } else { "released" },
                    scope,
                    hold.reason
                )),
                cell(if active {
                    hold.created_at
                } else {
                    hold.released_at
                }),
            ]
        })
        .collect())
}

/// The commit chain over the compliance log, walked end to end.
///
/// The same walk `VERIFY TABLE zyron_sys.compliance.log` runs, through the
/// same code, so this report and that statement answer with one result
async fn audit_rows(server: &ServerState) -> Result<Vec<Row>, ZyronError> {
    let Some(table) = compliance_log_table(server) else {
        return Ok(vec![vec![
            cell("audit"),
            cell("compliance_log"),
            None,
            cell("INFO"),
            cell("the compliance log is not open on this node, so its chain cannot be walked"),
            None,
        ]]);
    };
    let cancelled: Arc<dyn Fn() -> bool + Send + Sync> = Arc::new(|| false);
    let outcome = crate::verify_dispatch::run_verify(
        server,
        crate::verify_dispatch::VerifyRequest {
            table_id: table.id.0,
            from_version: None,
            to_version: None,
            mode: zyron_lifecycle::verify::RowMode::Sampled,
            sample: crate::verify_dispatch::DEFAULT_SAMPLE,
        },
        0,
        "compliance report",
        cancelled,
    )
    .await?;
    Ok(vec![vec![
        cell("audit"),
        cell(&table.name),
        cell(table.id.0),
        cell(if outcome.intact { "PASS" } else { "FAIL" }),
        cell(outcome.summary()),
        None,
    ]])
}

/// The compliance log's catalog entry, which the system catalog registers
/// under its own schema
fn compliance_log_table(server: &ServerState) -> Option<Arc<zyron_catalog::TableEntry>> {
    server
        .catalog
        .list_all_tables()
        .into_iter()
        .find(|table| table.name.eq_ignore_ascii_case(COMPLIANCE_LOG_TABLE))
}

/// The name the compliance log's rows are held under
pub const COMPLIANCE_LOG_TABLE: &str = "compliance_log";

/// The compliance log itself, one row per recorded event.
async fn event_rows(server: &ServerState) -> Result<Vec<Row>, ZyronError> {
    let log = server.catalog.load_compliance_log().await?;
    Ok(log
        .into_iter()
        .map(|entry| {
            vec![
                cell("events"),
                cell(&entry.subject),
                cell(entry.table_id),
                cell("INFO"),
                cell(format!(
                    "{}: {}",
                    event_label(entry.event_type),
                    entry.detail
                )),
                cell(entry.ts),
            ]
        })
        .collect())
}

/// The data category a table falls under, taken from the strongest
/// classification any of its columns carries.
///
/// Confidential and Restricted are the two levels PII detection assigns, so
/// a table holding either is judged under the `pii` requirements. Public and
/// Internal columns say nothing about regulated content, and a table with
/// only those is reported as unclassified rather than passed: a retention
/// policy on data nobody has classified has not been checked, and saying so
/// is more useful than a PASS that means nothing.
fn table_category(server: &ServerState, table_id: u32) -> &'static str {
    let Some(security) = server.security_manager.as_ref() else {
        return "unclassified";
    };
    let strongest = security
        .classification_store
        .classifications_for_table(table_id)
        .into_iter()
        .map(|c| c.level)
        .max();
    match strongest {
        Some(zyron_auth::ClassificationLevel::Confidential)
        | Some(zyron_auth::ClassificationLevel::Restricted) => "pii",
        _ => "unclassified",
    }
}

/// What a retention policy's kind code means.
fn retention_kind(kind: u8) -> &'static str {
    match kind {
        0 => "TTL",
        1 => "cold-tier",
        2 => "archive",
        _ => "unknown",
    }
}

/// What a compliance event's type code means.
fn event_label(code: u8) -> &'static str {
    match code {
        event::TTL => "ttl",
        event::ARCHIVE => "archive",
        event::RESTORE => "restore",
        event::LEGAL_HOLD => "legal_hold",
        event::FORGET_USER => "forget_user",
        event::EXPORT_USER => "export_user",
        event::CLASSIFICATION => "classification",
        event::TIER_MOVE => "tier_move",
        event::RETENTION_LOCK => "retention_lock",
        event::CRYPTO_SHRED => "crypto_shred",
        event::PURGE => "purge",
        event::UNDROP => "undrop",
        event::CHAIN_ANCHORED => "chain_anchored",
        event::TABLE_VERIFICATION_ENABLED => "table_verification_enabled",
        event::VERIFY_RUN => "verify_run",
        _ => "unknown",
    }
}
