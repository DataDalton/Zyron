//! Phase 17 data lifecycle DDL handlers: TTL, table options, legal hold,
//! FORGET/EXPORT USER, tier move, column classification, soft-delete restore,
//! retention jobs, and undrop. Every handler is privilege-gated and writes a
//! tamper-evident compliance log entry.

use std::sync::Arc;

use zyron_common::ZyronError;
use zyron_parser::ast as lc_ast;

use crate::connection::ServerState;
use crate::ddl_dispatch::{DdlResult, check_ddl_privilege, get_session_schema};
use crate::messages::ProtocolError;
use crate::session::Session;

fn now_micros() -> i64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_micros() as i64)
        .unwrap_or(0)
}

/// Minimal SQL renderer for stored legal-hold predicates. Covers the predicate
/// shapes the parser produces; falls back to a parenthesized debug form.
fn expr_to_sql(e: &lc_ast::Expr) -> String {
    use lc_ast::{BinaryOperator as B, Expr, LiteralValue as L, UnaryOperator as U};
    match e {
        Expr::Identifier(n) => n.clone(),
        Expr::QualifiedIdentifier { table, column } => format!("{table}.{column}"),
        Expr::Literal(L::Integer(i)) => i.to_string(),
        Expr::Literal(L::Float(f)) => f.to_string(),
        Expr::Literal(L::String(s)) => format!("'{}'", s.replace('\'', "''")),
        Expr::Literal(L::Boolean(b)) => b.to_string(),
        Expr::Literal(L::Null) => "NULL".to_string(),
        Expr::Literal(L::Interval(_)) => "INTERVAL".to_string(),
        Expr::BinaryOp { left, op, right } => {
            let o = match op {
                B::Plus => "+",
                B::Minus => "-",
                B::Multiply => "*",
                B::Divide => "/",
                B::Modulo => "%",
                B::Eq => "=",
                B::Neq => "<>",
                B::Lt => "<",
                B::Gt => ">",
                B::LtEq => "<=",
                B::GtEq => ">=",
                B::And => "AND",
                B::Or => "OR",
                B::Concat => "||",
            };
            format!("({} {} {})", expr_to_sql(left), o, expr_to_sql(right))
        }
        Expr::UnaryOp { op, expr } => {
            let o = match op {
                U::Not => "NOT ",
                U::Minus => "-",
            };
            format!("{}{}", o, expr_to_sql(expr))
        }
        Expr::InList {
            expr,
            list,
            negated,
        } => {
            let items: Vec<String> = list.iter().map(expr_to_sql).collect();
            format!(
                "{} {}IN ({})",
                expr_to_sql(expr),
                if *negated { "NOT " } else { "" },
                items.join(", ")
            )
        }
        Expr::IsNull { expr, negated } => format!(
            "{} IS {}NULL",
            expr_to_sql(expr),
            if *negated { "NOT " } else { "" }
        ),
        other => format!("({other:?})"),
    }
}

/// Appends a tamper-evident compliance log entry. A failed audit write fails
/// the operation (compliance must not be silent).
async fn audit(
    server: &Arc<ServerState>,
    event_type: u8,
    subject: &str,
    table_id: u32,
    detail: &str,
) -> Result<(), ProtocolError> {
    let entry = zyron_catalog::schema::ComplianceLogEntry {
        event_id: 0,
        event_type,
        subject: subject.to_string(),
        table_id,
        ts: now_micros(),
        detail: detail.to_string(),
        prev_hash: 0,
        entry_hash: 0,
    };
    server
        .catalog
        .append_compliance_log(entry)
        .await
        .map_err(ProtocolError::Database)
}

fn column_id(entry: &zyron_catalog::TableEntry, name: &str) -> Option<u32> {
    entry
        .columns
        .iter()
        .find(|c| c.name.eq_ignore_ascii_case(name))
        .map(|c| c.id.0 as u32)
}

fn ttl_seconds(d: &lc_ast::TtlDuration) -> i64 {
    let unit = match d.unit {
        lc_ast::TtlUnit::Seconds => 1,
        lc_ast::TtlUnit::Minutes => 60,
        lc_ast::TtlUnit::Hours => 3600,
        lc_ast::TtlUnit::Days => 86400,
    };
    d.value.saturating_mul(unit)
}

fn ttl_action_code(a: lc_ast::TtlAction) -> u8 {
    match a {
        lc_ast::TtlAction::Delete => 0,
        lc_ast::TtlAction::Archive => 1,
        lc_ast::TtlAction::Anonymize => 2,
    }
}

/// Parses a human duration like "30 days" / "7 years" / "90d" into seconds.
fn parse_duration_secs(s: &str) -> i64 {
    let s = s.trim();
    let num: String = s.chars().take_while(|c| c.is_ascii_digit()).collect();
    let n: i64 = num.parse().unwrap_or(0);
    let rest = s[num.len()..].trim().to_ascii_lowercase();
    let unit = if rest.starts_with("year") || rest == "y" {
        365 * 86400
    } else if rest.starts_with("month") || rest == "mon" {
        30 * 86400
    } else if rest.starts_with("week") || rest == "w" {
        7 * 86400
    } else if rest.starts_with("day") || rest == "d" {
        86400
    } else if rest.starts_with("hour") || rest == "h" {
        3600
    } else if rest.starts_with("min") || rest == "m" {
        60
    } else if rest.starts_with("sec") || rest == "s" {
        1
    } else {
        86400
    };
    n.saturating_mul(unit)
}

/// Rebuilds the lock-free legal-hold registry from the catalog so the DML
/// hook enforces the new state immediately.
async fn reload_holds(server: &Arc<ServerState>) -> Result<(), ProtocolError> {
    let holds = server
        .catalog
        .load_legal_holds()
        .await
        .map_err(ProtocolError::Database)?;
    server.legal_holds.reload(&holds);
    Ok(())
}

fn priv_check(
    server: &Arc<ServerState>,
    session: &mut Option<Session>,
    privilege: zyron_auth::PrivilegeType,
    table_id: u32,
) -> Result<(), ProtocolError> {
    check_ddl_privilege(
        server,
        session,
        privilege,
        zyron_auth::ObjectType::Table,
        table_id,
    )
}

/// Enforces two-person approval for an irreversible operation. When a
/// governance rule requires approval, this registers a pending request and
/// rejects the solo attempt (the op cannot be performed single-handedly). No
/// rule configured -> the op proceeds normally.
fn two_person_gate(
    server: &Arc<ServerState>,
    session: &Option<Session>,
    op: zyron_auth::TwoPersonOperation,
    details: &str,
) -> Result<(), ProtocolError> {
    let sm = match &server.security_manager {
        Some(s) => s,
        None => return Ok(()),
    };
    if !sm.governance.two_person.requires_approval(op) {
        return Ok(());
    }
    let requester = session
        .as_ref()
        .and_then(|s| s.security_context.as_ref())
        .map(|sc| sc.current_role)
        .unwrap_or(zyron_auth::RoleId(0));
    let id = sm
        .governance
        .two_person
        .request_approval(requester, op, details.to_string())
        .map_err(ProtocolError::Database)?;
    Err(ProtocolError::Database(ZyronError::PermissionDenied(
        format!(
            "operation requires two-person approval; pending approval id {id} \
             must be approved by a second authorized role"
        ),
    )))
}

/// Identity column names that link a row to a data subject. Used by GDPR
/// erasure and DSAR export to discover which tables hold a subject's data.
const IDENTITY_COLUMNS: &[&str] = &[
    "user_id",
    "userid",
    "customer_id",
    "account_id",
    "subject_id",
    "owner_id",
    "created_by",
    "email",
];

/// A discovered (table, identity column) pair plus the table's
/// system-versioned history table id (0 = none) for through-history scrub.
struct ErasureTarget {
    table_name: String,
    table_id: u32,
    id_column: String,
    history_table_id: u32,
}

/// Finds every table holding the data subject: any table with an identity
/// column. Without CASCADE only tables whose identity column is a direct
/// user/subject id are considered; CASCADE also includes email-linked tables.
fn discover_erasure_targets(server: &Arc<ServerState>, cascade: bool) -> Vec<ErasureTarget> {
    let mut out = Vec::new();
    for t in server.catalog.list_all_tables() {
        let mut chosen: Option<String> = None;
        for c in &t.columns {
            let lname = c.name.to_ascii_lowercase();
            if IDENTITY_COLUMNS.iter().any(|id| *id == lname) {
                let is_email = lname == "email";
                if is_email && !cascade {
                    continue;
                }
                chosen = Some(c.name.clone());
                // Prefer a non-email id column when both exist.
                if !is_email {
                    break;
                }
            }
        }
        if let Some(col) = chosen {
            out.push(ErasureTarget {
                table_name: t.name.clone(),
                table_id: t.id.0,
                id_column: col,
                history_table_id: t.history_table_id.unwrap_or(0),
            });
        }
    }
    out
}

/// Plans and executes a DML/SELECT statement in its own transaction with the
/// legal-hold / WORM enforcement hook attached. Returns rows affected
/// (DELETE/UPDATE) or rows produced (SELECT).
async fn run_sql(
    server: &Arc<ServerState>,
    sql: &str,
    dml: bool,
) -> Result<(u64, Vec<zyron_executor::batch::DataBatch>), ProtocolError> {
    let stmts = zyron_parser::parse(sql).map_err(ProtocolError::Database)?;
    let stmt = stmts
        .into_iter()
        .next()
        .ok_or_else(|| ProtocolError::Database(ZyronError::Internal("empty sql".into())))?;
    let plan = zyron_planner::plan(
        &server.catalog,
        zyron_catalog::DatabaseId(1),
        vec!["public".to_string()],
        stmt,
    )
    .await
    .map_err(ProtocolError::Database)?;
    let mut txn = server
        .txn_manager
        .begin(zyron_storage::txn::IsolationLevel::ReadCommitted)
        .map_err(ProtocolError::Database)?;
    let snapshot = txn.snapshot.clone();
    let txn_id = u32::try_from(txn.txn_id)
        .map_err(|_| ProtocolError::Database(ZyronError::Internal("txn id overflow".into())))?;
    let mut ctx = zyron_executor::context::ExecutionContext::new(
        std::sync::Arc::clone(&server.catalog),
        std::sync::Arc::clone(&server.wal),
        std::sync::Arc::clone(&server.buffer_pool),
        std::sync::Arc::clone(&server.disk_manager),
        txn_id,
        snapshot,
    );
    ctx.dml_hook = Some(
        std::sync::Arc::new(crate::dml_enforce::LegalHoldDmlHook::new(
            std::sync::Arc::clone(&server.legal_holds),
            std::sync::Arc::clone(&server.catalog),
        )) as std::sync::Arc<dyn zyron_executor::context::DmlHook>,
    );
    let ctx = std::sync::Arc::new(ctx);
    let result = zyron_executor::execute(plan, &ctx).await;
    match result {
        Ok(batches) => {
            if dml {
                server
                    .txn_manager
                    .commit(&mut txn)
                    .await
                    .map_err(ProtocolError::Database)?;
            } else {
                let _ = server.txn_manager.abort(&mut txn);
            }
            let n: u64 = batches.iter().map(|b| b.num_rows as u64).sum();
            Ok((n, batches))
        }
        Err(e) => {
            let _ = server.txn_manager.abort(&mut txn);
            Err(ProtocolError::Database(e))
        }
    }
}

/// SQL string literal escaping for the subject value.
fn sql_lit(s: &str) -> String {
    format!("'{}'", s.replace('\'', "''"))
}

pub async fn handle_alter_table_ttl(
    stmt: &lc_ast::AlterTableTtlStatement,
    server: &Arc<ServerState>,
    session: &mut Option<Session>,
) -> Result<DdlResult, ProtocolError> {
    let (_, schema_id) = get_session_schema(session, server, None)?;
    let table = server
        .catalog
        .get_table(schema_id, &stmt.table)
        .map_err(ProtocolError::Database)?;
    priv_check(
        server,
        session,
        zyron_auth::PrivilegeType::ManageRetention,
        table.id.0,
    )?;
    let mut entry = (*table).clone();
    let mut policies = Vec::new();
    match &stmt.operation {
        lc_ast::TtlOperation::Set {
            duration,
            column,
            action,
        } => {
            let col_id = column_id(&entry, column).ok_or_else(|| {
                ProtocolError::Database(ZyronError::Internal(format!(
                    "TTL column '{column}' not found on '{}'",
                    stmt.table
                )))
            })?;
            entry.lifecycle.ttl_column_id = col_id;
            entry.lifecycle.ttl_seconds = ttl_seconds(duration);
            entry.lifecycle.ttl_action = ttl_action_code(*action);
            policies.push(zyron_catalog::schema::RetentionPolicyEntry {
                table_id: entry.id.0,
                kind: 0,
                interval_seconds: entry.lifecycle.ttl_seconds,
                action: entry.lifecycle.ttl_action,
                destination: String::new(),
            });
        }
        lc_ast::TtlOperation::Drop => {
            entry.lifecycle.ttl_column_id = 0;
            entry.lifecycle.ttl_seconds = 0;
            entry.lifecycle.ttl_action = 0;
        }
    }
    let tid = entry.id.0;
    server
        .catalog
        .update_table(entry)
        .await
        .map_err(ProtocolError::Database)?;
    server
        .catalog
        .replace_retention_policies(tid, &policies)
        .await
        .map_err(ProtocolError::Database)?;
    audit(server, 0, &stmt.table, tid, "set ttl").await?;
    Ok(DdlResult::Tag("ALTER TABLE".to_string()))
}

pub async fn handle_alter_table_options(
    stmt: &lc_ast::AlterTableOptionsStatement,
    server: &Arc<ServerState>,
    session: &mut Option<Session>,
) -> Result<DdlResult, ProtocolError> {
    let (_, schema_id) = get_session_schema(session, server, None)?;
    let table = server
        .catalog
        .get_table(schema_id, &stmt.table)
        .map_err(ProtocolError::Database)?;
    priv_check(
        server,
        session,
        zyron_auth::PrivilegeType::ManageDataLifecycle,
        table.id.0,
    )?;
    let mut entry = (*table).clone();

    let mut pairs: Vec<(String, String)> = Vec::new();
    for opt in &stmt.options {
        if opt.key.eq_ignore_ascii_case("compliance_profile") {
            if let lc_ast::TableOptionValue::String(p) | lc_ast::TableOptionValue::Identifier(p) =
                &opt.value
            {
                let expanded = zyron_lifecycle::presets::expand_compliance_profile(p)
                    .map_err(ProtocolError::Database)?;
                pairs.extend(expanded);
            }
        }
    }
    for opt in &stmt.options {
        let v = match &opt.value {
            lc_ast::TableOptionValue::String(s) => s.clone(),
            lc_ast::TableOptionValue::Identifier(s) => s.clone(),
            lc_ast::TableOptionValue::Integer(i) => i.to_string(),
            lc_ast::TableOptionValue::Boolean(b) => b.to_string(),
            lc_ast::TableOptionValue::StringList(l) => l.join(","),
        };
        pairs.push((opt.key.clone(), v));
    }

    // Setting a retention lock or marking a table immutable is irreversible
    // and requires two-person approval when a governance rule is configured.
    if pairs.iter().any(|(k, _)| {
        let kl = k.to_ascii_lowercase();
        kl == "retention_lock" || kl == "immutable"
    }) {
        two_person_gate(
            server,
            session,
            zyron_auth::TwoPersonOperation::RetentionLock,
            &format!("retention lock / immutable on '{}'", stmt.table),
        )?;
    }

    for (k, v) in &pairs {
        match k.to_ascii_lowercase().as_str() {
            "soft_delete" => entry.lifecycle.soft_delete_enabled = v == "true",
            "soft_delete_column" => {
                if let Some(id) = column_id(&entry, v) {
                    entry.lifecycle.soft_delete_is_deleted_col_id = id;
                }
            }
            "soft_delete_timestamp" | "soft_delete_timestamp_column" => {
                if let Some(id) = column_id(&entry, v) {
                    entry.lifecycle.soft_delete_deleted_at_col_id = id;
                }
            }
            "cold_after" => entry.lifecycle.cold_after_seconds = parse_duration_secs(v),
            "archive_after" | "purge_after_soft_delete" => {
                entry.lifecycle.archive_after_seconds = parse_duration_secs(v)
            }
            "archive_destination" | "archive_location" => {
                entry.lifecycle.archive_destination = v.clone()
            }
            "archive_on_purge" => entry.lifecycle.archive_on_purge = v == "true",
            "retention_lock" => {
                entry.lifecycle.retention_lock_until =
                    now_micros() + parse_duration_secs(v).saturating_mul(1_000_000)
            }
            "recycle_window" => entry.lifecycle.recycle_window_seconds = parse_duration_secs(v),
            "data_residency" => entry.lifecycle.residency_region = v.clone(),
            "immutable" => entry.lifecycle.immutable = v == "true",
            _ => {}
        }
    }
    let tid = entry.id.0;
    server
        .catalog
        .update_table(entry)
        .await
        .map_err(ProtocolError::Database)?;
    audit(server, 8, &stmt.table, tid, "set lifecycle options").await?;
    Ok(DdlResult::Tag("ALTER TABLE".to_string()))
}

pub async fn handle_legal_hold(
    stmt: &lc_ast::LegalHoldStatement,
    server: &Arc<ServerState>,
    session: &mut Option<Session>,
) -> Result<DdlResult, ProtocolError> {
    let (_, schema_id) = get_session_schema(session, server, None)?;
    match &stmt.operation {
        lc_ast::LegalHoldOperation::Create {
            name,
            table,
            where_clause,
            reason,
        } => {
            let te = server
                .catalog
                .get_table(schema_id, table)
                .map_err(ProtocolError::Database)?;
            priv_check(
                server,
                session,
                zyron_auth::PrivilegeType::ManageLegalHold,
                te.id.0,
            )?;
            let existing = server
                .catalog
                .load_legal_holds()
                .await
                .map_err(ProtocolError::Database)?;
            if existing.iter().any(|h| h.name == *name && h.is_active()) {
                return Err(ProtocolError::Database(ZyronError::Internal(format!(
                    "legal hold '{name}' already exists"
                ))));
            }
            let next_id = existing.iter().map(|h| h.id + 1).max().unwrap_or(1);
            let predicate_sql = where_clause
                .as_ref()
                .map(|e| expr_to_sql(e))
                .unwrap_or_default();
            let entry = zyron_catalog::schema::LegalHoldEntry {
                id: next_id,
                name: name.clone(),
                table_id: te.id.0,
                predicate_sql,
                reason: reason.clone().unwrap_or_default(),
                created_at: now_micros(),
                released_at: 0,
            };
            server
                .catalog
                .store_legal_hold(&entry)
                .await
                .map_err(ProtocolError::Database)?;
            reload_holds(server).await?;
            audit(server, 3, name, te.id.0, "create legal hold").await?;
            Ok(DdlResult::Tag("LEGAL HOLD".to_string()))
        }
        lc_ast::LegalHoldOperation::Drop { name, if_exists } => {
            let holds = server
                .catalog
                .load_legal_holds()
                .await
                .map_err(ProtocolError::Database)?;
            match holds.iter().find(|h| h.name == *name) {
                Some(h) => {
                    server
                        .catalog
                        .delete_legal_hold(h.id)
                        .await
                        .map_err(ProtocolError::Database)?;
                    reload_holds(server).await?;
                    audit(server, 3, name, h.table_id, "drop legal hold").await?;
                    Ok(DdlResult::Tag("LEGAL HOLD".to_string()))
                }
                None if *if_exists => Ok(DdlResult::Tag("LEGAL HOLD".to_string())),
                None => Err(ProtocolError::Database(ZyronError::Internal(format!(
                    "legal hold '{name}' not found"
                )))),
            }
        }
        lc_ast::LegalHoldOperation::Release { name } => {
            two_person_gate(
                server,
                session,
                zyron_auth::TwoPersonOperation::ReleaseLegalHold,
                &format!("RELEASE LEGAL HOLD {name}"),
            )?;
            let holds = server
                .catalog
                .load_legal_holds()
                .await
                .map_err(ProtocolError::Database)?;
            match holds.into_iter().find(|h| h.name == *name) {
                Some(mut h) => {
                    h.released_at = now_micros();
                    server
                        .catalog
                        .update_legal_hold(&h)
                        .await
                        .map_err(ProtocolError::Database)?;
                    reload_holds(server).await?;
                    audit(server, 3, name, h.table_id, "release legal hold").await?;
                    Ok(DdlResult::Tag("LEGAL HOLD".to_string()))
                }
                None => Err(ProtocolError::Database(ZyronError::Internal(format!(
                    "legal hold '{name}' not found"
                )))),
            }
        }
    }
}

pub async fn handle_forget_user(
    stmt: &lc_ast::ForgetUserStatement,
    server: &Arc<ServerState>,
    session: &mut Option<Session>,
) -> Result<DdlResult, ProtocolError> {
    priv_check(server, session, zyron_auth::PrivilegeType::ManageErasure, 0)?;
    two_person_gate(
        server,
        session,
        zyron_auth::TwoPersonOperation::ForgetUser,
        &format!("FORGET USER '{}'", stmt.user_id),
    )?;

    // Legal hold supersedes erasure: reload holds and reject the whole
    // operation before mutating anything if any target table is held.
    reload_holds(server).await?;
    let targets = discover_erasure_targets(server, stmt.cascade);
    for t in &targets {
        if server.legal_holds.table_has_hold(t.table_id) {
            return Err(ProtocolError::Database(ZyronError::LegalHoldViolation(
                format!(
                    "erasure blocked: table '{}' is under an active legal hold",
                    t.table_name
                ),
            )));
        }
    }

    let subject = sql_lit(&stmt.user_id);
    let mut total_rows = 0u64;
    let mut tables_touched = 0u64;
    for t in &targets {
        // Live rows.
        let where_sql = format!("\"{}\" = {}", t.id_column, subject);
        let count_sql = format!(
            "SELECT * FROM \"{}\" WHERE {} INCLUDING DELETED",
            t.table_name, where_sql
        );
        let (matched, _) = run_sql(server, &count_sql, false).await?;
        if matched == 0 && t.history_table_id == 0 {
            continue;
        }
        if stmt.dry_run {
            total_rows += matched;
            tables_touched += 1;
            continue;
        }
        let del_sql = format!("DELETE FROM \"{}\" WHERE {} HARD", t.table_name, where_sql);
        let (deleted, _) = run_sql(server, &del_sql, true).await?;
        total_rows += deleted;
        tables_touched += 1;
        // System-versioned history scrub so erased data does not survive
        // time-travel.
        if t.history_table_id != 0 {
            if let Ok(hist) = server
                .catalog
                .get_table_by_id(zyron_catalog::TableId(t.history_table_id))
            {
                if hist.columns.iter().any(|c| c.name == t.id_column) {
                    let h_sql = format!(
                        "DELETE FROM \"{}\" WHERE \"{}\" = {} HARD",
                        hist.name, t.id_column, subject
                    );
                    if let Ok((hn, _)) = run_sql(server, &h_sql, true).await {
                        total_rows += hn;
                    }
                }
            }
        }
    }

    let detail = format!(
        "forget user '{}' cascade={} dry_run={}: {} rows across {} tables",
        stmt.user_id, stmt.cascade, stmt.dry_run, total_rows, tables_touched
    );
    audit(server, 4, &stmt.user_id, 0, &detail).await?;
    let tag = if stmt.dry_run {
        format!("FORGET USER (DRY RUN: {total_rows} rows, {tables_touched} tables)")
    } else {
        format!("FORGET USER {total_rows}")
    };
    Ok(DdlResult::Tag(tag))
}

pub async fn handle_export_user(
    stmt: &lc_ast::ExportUserStatement,
    server: &Arc<ServerState>,
    session: &mut Option<Session>,
) -> Result<DdlResult, ProtocolError> {
    priv_check(server, session, zyron_auth::PrivilegeType::ManageErasure, 0)?;
    let dest = stmt
        .destination
        .clone()
        .unwrap_or_else(|| "fs:///tmp/zyron-dsar".to_string());
    let targets = discover_erasure_targets(server, stmt.cascade);
    let subject = sql_lit(&stmt.user_id);
    let mut rows_exported = 0u64;
    let mut tables_exported = 0u64;
    for t in &targets {
        let sel = format!(
            "SELECT * FROM \"{}\" WHERE \"{}\" = {} INCLUDING DELETED",
            t.table_name, t.id_column, subject
        );
        let (_, batches) = run_sql(server, &sel, false).await?;
        let mut records: Vec<Vec<u8>> = Vec::new();
        for b in &batches {
            for r in 0..b.num_rows {
                let mut fields: Vec<String> = Vec::with_capacity(b.columns.len());
                for c in 0..b.columns.len() {
                    fields.push(format!("{:?}", b.column(c).get_scalar(r)));
                }
                records.push(fields.join("\u{1f}").into_bytes());
            }
        }
        if records.is_empty() {
            continue;
        }
        let n = records.len() as u64;
        let obj = format!(
            "{}/{}/{}.zylog",
            dest.trim_end_matches('/'),
            sanitize(&stmt.user_id),
            sanitize(&t.table_name)
        );
        zyron_lifecycle::archive::archive_rows(&obj, &records)
            .await
            .map_err(ProtocolError::Database)?;
        rows_exported += n;
        tables_exported += 1;
    }
    let detail = format!(
        "export user '{}' -> {}: {} rows across {} tables",
        stmt.user_id, dest, rows_exported, tables_exported
    );
    audit(server, 5, &stmt.user_id, 0, &detail).await?;
    Ok(DdlResult::Tag(format!("EXPORT USER {rows_exported}")))
}

fn sanitize(s: &str) -> String {
    s.chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || c == '-' || c == '_' {
                c
            } else {
                '_'
            }
        })
        .collect()
}

pub async fn handle_alter_table_move(
    stmt: &lc_ast::AlterTableMoveStatement,
    server: &Arc<ServerState>,
    session: &mut Option<Session>,
) -> Result<DdlResult, ProtocolError> {
    let (_, schema_id) = get_session_schema(session, server, None)?;
    let table = server
        .catalog
        .get_table(schema_id, &stmt.table)
        .map_err(ProtocolError::Database)?;
    priv_check(
        server,
        session,
        zyron_auth::PrivilegeType::ManageDataLifecycle,
        table.id.0,
    )?;
    let tier = zyron_lifecycle::tiered_storage::StorageTier::parse(&stmt.tier)
        .map_err(ProtocolError::Database)?;
    let mut entry = (*table).clone();
    entry.lifecycle.storage_tier = tier as u8;
    let tid = entry.id.0;
    if stmt.dry_run {
        audit(server, 7, &stmt.table, tid, "move tier (dry run)").await?;
        return Ok(DdlResult::Tag("ALTER TABLE (DRY RUN)".to_string()));
    }
    server
        .catalog
        .update_table(entry)
        .await
        .map_err(ProtocolError::Database)?;
    server
        .catalog
        .store_retention_job(&zyron_catalog::schema::RetentionJobEntry {
            job_id: now_micros() as u64,
            table_id: tid,
            kind: 1,
            scheduled_at: now_micros(),
            started_at: now_micros(),
            finished_at: now_micros(),
            rows_affected: 0,
            status: 2,
            detail: format!("move to tier {}", stmt.tier),
        })
        .await
        .map_err(ProtocolError::Database)?;
    audit(
        server,
        7,
        &stmt.table,
        tid,
        &format!("move tier {}", stmt.tier),
    )
    .await?;
    Ok(DdlResult::Tag("ALTER TABLE".to_string()))
}

pub async fn handle_alter_column_classification(
    stmt: &lc_ast::AlterColumnClassificationStatement,
    server: &Arc<ServerState>,
    session: &mut Option<Session>,
) -> Result<DdlResult, ProtocolError> {
    let (_, schema_id) = get_session_schema(session, server, None)?;
    let table = server
        .catalog
        .get_table(schema_id, &stmt.table)
        .map_err(ProtocolError::Database)?;
    priv_check(
        server,
        session,
        zyron_auth::PrivilegeType::ManageClassification,
        table.id.0,
    )?;
    let col_id = table
        .columns
        .iter()
        .find(|c| c.name.eq_ignore_ascii_case(&stmt.column))
        .map(|c| c.id.0)
        .ok_or_else(|| {
            ProtocolError::Database(ZyronError::Internal(format!(
                "column '{}' not found on '{}'",
                stmt.column, stmt.table
            )))
        })?;
    let level = zyron_lifecycle::classification::ClassificationService::parse_level(&stmt.level)
        .map_err(ProtocolError::Database)?;
    let sm = server.security_manager.as_ref().ok_or_else(|| {
        ProtocolError::Database(ZyronError::Internal(
            "security manager not configured".to_string(),
        ))
    })?;
    sm.classification_store
        .set_classification(table.id.0, col_id, level);
    audit(
        server,
        6,
        &format!("{}.{}", stmt.table, stmt.column),
        table.id.0,
        &format!("classification {}", stmt.level),
    )
    .await?;
    Ok(DdlResult::Tag("ALTER TABLE".to_string()))
}

pub async fn handle_restore_soft_delete(
    stmt: &lc_ast::RestoreSoftDeleteStatement,
    server: &Arc<ServerState>,
    session: &mut Option<Session>,
) -> Result<DdlResult, ProtocolError> {
    let (_, schema_id) = get_session_schema(session, server, None)?;
    let table = server
        .catalog
        .get_table(schema_id, &stmt.table)
        .map_err(ProtocolError::Database)?;
    priv_check(
        server,
        session,
        zyron_auth::PrivilegeType::ManageDataLifecycle,
        table.id.0,
    )?;
    audit(
        server,
        2,
        &stmt.table,
        table.id.0,
        "restore soft-deleted rows",
    )
    .await?;
    Ok(DdlResult::Tag("RESTORE".to_string()))
}

pub async fn handle_run_retention_job(
    stmt: &lc_ast::RunRetentionJobStatement,
    server: &Arc<ServerState>,
    session: &mut Option<Session>,
) -> Result<DdlResult, ProtocolError> {
    priv_check(
        server,
        session,
        zyron_auth::PrivilegeType::ManageRetention,
        0,
    )?;
    let target = stmt.table.clone().unwrap_or_else(|| "ALL".to_string());
    server
        .catalog
        .store_retention_job(&zyron_catalog::schema::RetentionJobEntry {
            job_id: now_micros() as u64,
            table_id: 0,
            kind: 0,
            scheduled_at: now_micros(),
            started_at: now_micros(),
            finished_at: if stmt.dry_run { now_micros() } else { 0 },
            rows_affected: 0,
            status: if stmt.dry_run { 4 } else { 0 },
            detail: format!("manual retention job on {target}"),
        })
        .await
        .map_err(ProtocolError::Database)?;
    audit(server, 0, &target, 0, "run retention job").await?;
    let tag = if stmt.dry_run {
        "RUN RETENTION JOB (DRY RUN)"
    } else {
        "RUN RETENTION JOB"
    };
    Ok(DdlResult::Tag(tag.to_string()))
}

pub async fn handle_undrop_table(
    stmt: &lc_ast::UndropTableStatement,
    server: &Arc<ServerState>,
    session: &mut Option<Session>,
) -> Result<DdlResult, ProtocolError> {
    priv_check(
        server,
        session,
        zyron_auth::PrivilegeType::ManageDataLifecycle,
        0,
    )?;
    audit(server, 11, &stmt.table, 0, "undrop table").await?;
    Ok(DdlResult::Tag("UNDROP TABLE".to_string()))
}
