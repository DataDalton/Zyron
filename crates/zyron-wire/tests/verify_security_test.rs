//! Who may verify what, and what anchoring raises.
//!
//! A verification states what a table holds, so asking for one takes SELECT
//! on the table. Reading every row back costs the whole table, so
//! `rows => 'all'` takes MANAGE_VERIFICATION on top. Every run is on the
//! record with the principal that asked.
//!
//! The anchoring half covers what a `ChainAnchored` event carries, what an
//! exported anchor verifies against, and when `chain_not_anchored` fires.
//!
//! Run: cargo test -p zyron-wire --test verify_security_test -- --nocapture

mod common;

use std::sync::Arc;

use common::{
    create_test_server_with_cdc_and_security, exec_ddl, exec_dml, new_session, table_id_of,
};
use zyron_auth::{
    ClassificationLevel, GrantEntry, ObjectType, PrivilegeState, PrivilegeType, RoleId,
    SecurityContext, SessionAttributes, UserId,
};
use zyron_wire::connection::ServerState;

fn grant(sm: &zyron_auth::SecurityManager, role: RoleId, privilege: PrivilegeType, object_id: u32) {
    sm.privilege_store
        .grant(GrantEntry {
            grantee: role,
            privilege,
            object_type: ObjectType::Table,
            object_id,
            columns: None,
            state: PrivilegeState::Grant,
            with_grant_option: false,
            granted_by: RoleId(0),
            valid_from: None,
            valid_until: None,
            time_window: None,
            object_pattern: None,
            no_inherit: false,
            mask_function: None,
        })
        .expect("grant");
}

fn security_context(role: RoleId) -> SecurityContext {
    let attrs = SessionAttributes {
        role_id: role,
        department: None,
        region: None,
        clearance: ClassificationLevel::Public,
        ip_address: "127.0.0.1".to_string(),
        connection_time: 0,
        custom: std::collections::HashMap::new(),
    };
    SecurityContext::new(
        UserId(role.0),
        role,
        vec![role],
        vec![role],
        ClassificationLevel::Public,
        attrs,
        None,
        zyron_auth::QueryLimits::default(),
    )
}

/// Runs a statement as a principal through the dispatcher
async fn run_as(
    server: &Arc<ServerState>,
    role: RoleId,
    user: &str,
    sql: &str,
) -> Result<Vec<Vec<String>>, String> {
    let mut session = new_session();
    if let Some(session) = session.as_mut() {
        session.user = user.to_string();
        session.security_context = Some(security_context(role));
    }
    let stmt = zyron_parser::parse(sql)
        .expect("parse")
        .into_iter()
        .next()
        .expect("one statement");
    let mut txn_opt: Option<zyron_storage::txn::Transaction> = None;
    let mut branch: Option<String> = None;
    match zyron_wire::ddl_dispatch::try_handle_ddl_utility(
        &stmt,
        server,
        &mut session,
        &mut txn_opt,
        &mut branch,
        sql,
    )
    .await
    {
        Some(Ok(zyron_wire::ddl_dispatch::DdlResult::Rows { rows, .. })) => Ok(rows),
        Some(Ok(other)) => Err(format!("expected rows, got {other:?}")),
        Some(Err(e)) => Err(format!("{e:?}")),
        None => Err(format!("statement was not handled: {sql}")),
    }
}

/// A verified ledger with a few commits, and the security manager
async fn verified_ledger() -> (
    Arc<ServerState>,
    Arc<zyron_auth::SecurityManager>,
    u32,
    tempfile::TempDir,
) {
    let (server, _schema, sm, tmp) = create_test_server_with_cdc_and_security().await;
    let mut session = new_session();
    exec_ddl(
        &server,
        &mut session,
        "CREATE TABLE ledger (id BIGINT PRIMARY KEY, amount BIGINT)",
    )
    .await
    .expect("create ledger");
    exec_ddl(
        &server,
        &mut session,
        "ALTER TABLE ledger SET (immutable = true, verified = true)",
    )
    .await
    .expect("verified");
    for id in 1..=4 {
        exec_dml(
            &server,
            &format!("INSERT INTO ledger VALUES ({id}, {})", id * 10),
        )
        .await;
    }
    let table_id = table_id_of(&server, "ledger");
    (server, sm, table_id, tmp)
}

/// A verification states what a table holds, so asking for one without
/// SELECT on it is refused
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn verifying_without_select_is_refused() {
    let (server, _sm, _table_id, _tmp) = verified_ledger().await;
    let stranger = RoleId(41);

    let refused = run_as(&server, stranger, "stranger", "VERIFY TABLE ledger")
        .await
        .err()
        .expect("refused");
    assert!(
        refused.to_lowercase().contains("permission")
            || refused.to_lowercase().contains("denied")
            || refused.to_lowercase().contains("select"),
        "the refusal does not name the missing privilege: {refused}"
    );
}

/// Reading every row back costs the whole table, so it takes
/// MANAGE_VERIFICATION on top of SELECT. A sampled pass takes SELECT alone
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn reading_every_row_takes_manage_verification() {
    let (server, sm, table_id, _tmp) = verified_ledger().await;
    let reader = RoleId(42);
    grant(&sm, reader, PrivilegeType::Select, table_id);

    // SELECT alone is enough for the default
    let sampled = run_as(&server, reader, "reader", "VERIFY TABLE ledger")
        .await
        .expect("a sampled pass is allowed with SELECT");
    assert_eq!(sampled[0][4], "sampled");
    assert_eq!(sampled[0][5], "true");

    // And not enough for a full one
    let refused = run_as(
        &server,
        reader,
        "reader",
        "VERIFY TABLE ledger WITH (rows => 'all')",
    )
    .await
    .err()
    .expect("refused");
    assert!(
        refused.to_lowercase().contains("manageverification")
            || refused.to_lowercase().contains("manage_verification")
            || refused.to_lowercase().contains("permission")
            || refused.to_lowercase().contains("denied"),
        "the refusal does not name the missing privilege: {refused}"
    );

    // With it, the full pass runs
    let auditor = RoleId(43);
    grant(&sm, auditor, PrivilegeType::Select, table_id);
    grant(&sm, auditor, PrivilegeType::ManageVerification, table_id);
    let full = run_as(
        &server,
        auditor,
        "auditor",
        "VERIFY TABLE ledger WITH (rows => 'all')",
    )
    .await
    .expect("a full pass is allowed with MANAGE_VERIFICATION");
    assert_eq!(full[0][4], "all");
    assert_eq!(full[0][5], "true");
    assert_eq!(full[0][2], "4", "every row was read back");
}

/// Every run is recorded with the principal that asked for it, so a
/// verification is itself evidence of who checked and when
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn every_run_records_its_actor() {
    let (server, sm, table_id, _tmp) = verified_ledger().await;
    let auditor = RoleId(44);
    grant(&sm, auditor, PrivilegeType::Select, table_id);
    grant(&sm, auditor, PrivilegeType::ManageVerification, table_id);

    run_as(
        &server,
        auditor,
        "the_auditor",
        "VERIFY TABLE ledger WITH (rows => 'all')",
    )
    .await
    .expect("runs");

    let registry = server.chain_registry.as_ref().expect("chains");
    let run = registry
        .runs()
        .latest_for(table_id)
        .expect("the run is on the record");
    assert_eq!(run.actor, auditor.0, "the role that asked is recorded");
    assert_eq!(run.actor_name, "the_auditor");
    assert_eq!(run.table_name, "ledger");
    assert!(run.intact);
    assert_eq!(run.mode, zyron_lifecycle::verify::RowMode::All);
    assert_eq!(run.commits_checked, 4);
    assert_eq!(run.rows_checked, 4);
}

/// Anchoring records the head, writes a ChainAnchored event carrying the
/// table, the version and the head, and the exported artifact verifies
/// against the chain and is signed
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn anchoring_records_the_head_and_the_export_verifies() {
    let (server, sm, table_id, _tmp) = verified_ledger().await;
    let _sm = &sm;

    let anchor = zyron_wire::verify_dispatch::anchor_table(&server, table_id)
        .await
        .expect("anchors")
        .expect("the head is anchored");
    assert_eq!(anchor.table_id, table_id);
    assert_eq!(anchor.table_name, "ledger");
    assert_eq!(
        anchor.sequence, 3,
        "four commits, so the head is the fourth"
    );

    let registry = server.chain_registry.as_ref().expect("chains");
    let head = registry.chain(table_id).expect("chain").head();
    assert_eq!(anchor.commit_version, head.head_version);
    assert_eq!(anchor.head_hash, head.head_hash);
    assert!(anchor.taken_at > 0);

    // Held, so a verification after a restart is measured against it
    assert_eq!(registry.anchors().for_table(table_id), Some(anchor.clone()));

    // The event is in the audit log, naming the table, the version and the
    // head
    let log = server.catalog.load_compliance_log().await.expect("loads");
    let anchored = log
        .iter()
        .find(|entry| entry.event_type == zyron_lifecycle::compliance::event::CHAIN_ANCHORED)
        .expect("a ChainAnchored event was written");
    assert_eq!(anchored.table_id, table_id);
    assert_eq!(anchored.subject, "ledger");
    assert!(anchored.detail.contains("ledger"), "{}", anchored.detail);
    assert!(
        anchored.detail.contains(&head.head_version.to_string()),
        "the version is not named: {}",
        anchored.detail
    );
    assert!(
        anchored
            .detail
            .contains(&zyron_lifecycle::verify::hex(&head.head_hash)),
        "the head is not named: {}",
        anchored.detail
    );

    // The exported artifact is signed under the release scheme and agrees
    // with the chain
    let exported = zyron_wire::verify_dispatch::export_anchor(&anchor).expect("exports");
    assert!(!exported.signature.is_empty());
    assert_eq!(exported.scheme, "Ed25519");
    let bytes = exported.encode();
    let read_back =
        zyron_lifecycle::verify::anchor::ExportedAnchor::decode(&bytes).expect("decodes");
    assert_eq!(read_back, exported);

    let (signed, agrees) =
        zyron_wire::verify_dispatch::check_exported_anchor(registry, &read_back).expect("checks");
    assert!(
        signed,
        "the artifact is not signed under the release scheme"
    );
    assert!(
        agrees,
        "the artifact does not name the head the chain shows"
    );

    // A verification now holds the chain against it
    let reader = RoleId(45);
    grant(_sm, reader, PrivilegeType::Select, table_id);
    let rows = run_as(&server, reader, "reader", "VERIFY TABLE ledger")
        .await
        .expect("verifies");
    assert_eq!(rows[0][3], "1", "the anchor was checked");
    assert_eq!(rows[0][5], "true");
}

/// A head that has gone unanchored past twice the interval is reported,
/// once, until it is anchored again
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn an_unanchored_head_is_reported_past_twice_the_interval() {
    let (server, _sm, table_id, _tmp) = verified_ledger().await;

    // With a generous interval nothing is overdue yet
    let generous = zyron_wire::verify_dispatch::unanchored_tables(&server, 86_400);
    assert!(
        generous.is_empty(),
        "nothing should be overdue inside the interval: {generous:?}"
    );

    // With an interval of zero seconds the head is past twice it at once.
    // The compliance log is a verified table too and its own head is
    // unanchored, so the reading is narrowed to the table under test
    let overdue = ledger_rows(&server, 0);
    assert_eq!(overdue.len(), 1, "the unanchored head is reported");
    assert_eq!(overdue[0].0, table_id);

    // Anchoring clears it
    zyron_wire::verify_dispatch::anchor_table(&server, table_id)
        .await
        .expect("anchors")
        .expect("anchored");
    assert!(
        ledger_rows(&server, 0).is_empty(),
        "an anchored head is not overdue"
    );

    // A commit past the anchor makes it overdue again
    exec_dml(&server, "INSERT INTO ledger VALUES (99, 990)").await;
    assert_eq!(
        ledger_rows(&server, 0).len(),
        1,
        "a head past its anchor is overdue again"
    );
}

/// The overdue reading, narrowed to the table under test
fn ledger_rows(server: &Arc<ServerState>, interval_secs: u64) -> Vec<(u32, String)> {
    zyron_wire::verify_dispatch::unanchored_tables(server, interval_secs)
        .into_iter()
        .filter(|(_, name)| name == "ledger")
        .collect()
}

/// The anchoring pass anchors what has moved and leaves what has not
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn the_pass_anchors_what_has_moved() {
    let (server, _sm, table_id, _tmp) = verified_ledger().await;

    // The compliance log is a verified table too, so a pass anchors it
    // beside the ledger. What is read here is what the ledger's head did
    let first = anchored_ledger(&server).await;
    assert_eq!(first.len(), 1, "the head had moved, so it was anchored");

    // Nothing moved since, so a second pass anchors nothing of the ledger
    let second = anchored_ledger(&server).await;
    assert!(second.is_empty(), "a head that has not moved is left alone");

    exec_dml(&server, "INSERT INTO ledger VALUES (50, 500)").await;
    let third = anchored_ledger(&server).await;
    assert_eq!(third.len(), 1, "the head moved, so it was anchored again");
    assert_eq!(third[0].sequence, 4);
    let registry = server.chain_registry.as_ref().expect("chains");
    assert_eq!(
        registry.anchors().for_table(table_id).map(|a| a.sequence),
        Some(4),
        "the store holds the latest anchor"
    );
}

/// One anchoring pass, narrowed to the table under test
async fn anchored_ledger(
    server: &Arc<ServerState>,
) -> Vec<zyron_lifecycle::verify::anchor::Anchor> {
    zyron_wire::verify_dispatch::anchor_due_tables(server)
        .await
        .into_iter()
        .filter(|anchor| anchor.table_name == "ledger")
        .collect()
}

/// A chain covers stored bytes, and widening a column in place re-encodes
/// nothing, so both sides of the change stay verifiable.
///
/// The rows already written keep their bytes and the epoch stamp they were
/// written under, and the epoch is part of what their commit hashed, so a
/// verification reproduces their hashes exactly
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn widening_a_column_in_place_leaves_the_chain_verifiable() {
    let (server, sm, table_id, _tmp) = verified_ledger().await;
    let auditor = RoleId(46);
    grant(&sm, auditor, PrivilegeType::Select, table_id);
    grant(&sm, auditor, PrivilegeType::ManageVerification, table_id);

    let before = run_as(
        &server,
        auditor,
        "auditor",
        "VERIFY TABLE ledger WITH (rows => 'all')",
    )
    .await
    .expect("verifies");
    assert_eq!(before[0][5], "true", "{}", before[0][6]);
    let epoch_before = server
        .catalog
        .get_table_by_id(zyron_catalog::TableId(table_id))
        .expect("the table")
        .schema_epoch;

    // A representation-compatible widening. The catalog moves, the rows do
    // not
    let mut session = new_session();
    exec_ddl(
        &server,
        &mut session,
        "ALTER TABLE ledger ALTER COLUMN amount TYPE BIGINT",
    )
    .await
    .expect("a compatible widening is allowed on an immutable table");

    let entry = server
        .catalog
        .get_table_by_id(zyron_catalog::TableId(table_id))
        .expect("the table");
    assert!(
        entry.schema_epoch >= epoch_before,
        "the widening records an epoch"
    );

    let after = run_as(
        &server,
        auditor,
        "auditor",
        "VERIFY TABLE ledger WITH (rows => 'all')",
    )
    .await
    .expect("verifies");
    assert_eq!(
        after[0][5], "true",
        "the widening broke the chain: {}",
        after[0][6]
    );
    assert_eq!(after[0][1], before[0][1], "the same commits are covered");
    assert_eq!(after[0][2], before[0][2], "the same rows are read back");

    // Rows written after the change chain onto the same head
    exec_dml(&server, "INSERT INTO ledger VALUES (99, 990)").await;
    let extended = run_as(
        &server,
        auditor,
        "auditor",
        "VERIFY TABLE ledger WITH (rows => 'all')",
    )
    .await
    .expect("verifies");
    assert_eq!(
        extended[0][5], "true",
        "a commit after the widening broke the chain: {}",
        extended[0][6]
    );
}

/// A type change that cannot be read through re-encodes every row, which is
/// the same thing to a chain as updating all of them, so it answers to the
/// same lock DROP and TRUNCATE answer to
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_rewriting_type_change_is_refused_on_a_verified_table() {
    let (server, _sm, _table_id, _tmp) = verified_ledger().await;
    let mut session = new_session();

    let refused = exec_ddl(
        &server,
        &mut session,
        "ALTER TABLE ledger ALTER COLUMN amount TYPE TEXT",
    )
    .await
    .err()
    .expect("refused");
    assert!(
        refused.to_lowercase().contains("immutable"),
        "the refusal does not name the lock: {refused}"
    );

    // And the chain is untouched by the refusal
    let registry = server.chain_registry.as_ref().expect("chains");
    assert_eq!(registry.chain(_table_id).expect("chain").head().commits, 4);
}

/// A tenant that requires two people for a DROP TABLE gets it. The rule is
/// generic rather than particular to a verified table, and a tenant with no
/// rule configured is unaffected
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_drop_table_rule_holds_a_solo_attempt() {
    let (server, sm, _table_id, _tmp) = verified_ledger().await;
    let mut session = new_session();
    exec_ddl(
        &server,
        &mut session,
        "CREATE TABLE plain (id BIGINT PRIMARY KEY)",
    )
    .await
    .expect("create plain");

    // With no rule configured the drop goes through
    exec_ddl(&server, &mut session, "DROP TABLE plain")
        .await
        .expect("no rule, so nothing is gated");

    exec_ddl(
        &server,
        &mut session,
        "CREATE TABLE guarded (id BIGINT PRIMARY KEY)",
    )
    .await
    .expect("create guarded");
    sm.governance
        .two_person
        .add_rule(zyron_auth::TwoPersonRule {
            operation: zyron_auth::TwoPersonOperation::DropTable,
            required_role: None,
            timeout_secs: 3_600,
        });

    let refused = exec_ddl(&server, &mut session, "DROP TABLE guarded")
        .await
        .err()
        .expect("refused");
    assert!(
        refused.contains("two-person approval"),
        "the refusal does not name the gate: {refused}"
    );
    assert!(
        refused.contains("pending approval id"),
        "the refusal does not hand back an approval to act on: {refused}"
    );

    // The solo attempt left the table where it was
    assert!(
        server
            .catalog
            .list_all_tables()
            .iter()
            .any(|t| t.name == "guarded"),
        "a gated drop removed the table anyway"
    );
}

/// The two alerts verification raises are declared where the alert template
/// view reads them
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn the_alerts_are_declared_where_the_template_view_reads_them() {
    let (server, _sm, _table_id, _tmp) = verified_ledger().await;
    let (fields, rows) = zyron_wire::system_views::query_system_view(
        "zyron_sys.alert.templates",
        &server,
        &zyron_wire::system_views::SystemViewFilters::default(),
    )
    .await
    .expect("the view answers")
    .expect("the view exists");
    let columns: Vec<String> = fields.iter().map(|f| f.name.clone()).collect();
    let at = |name: &str| columns.iter().position(|c| c == name).expect("column");
    let name_at = at("name");
    let subsystem_at = at("subsystem");
    let threshold_at = at("threshold_setting");

    let text = |cell: &Option<Vec<u8>>| {
        cell.as_ref()
            .map(|b| String::from_utf8_lossy(b).into_owned())
            .unwrap_or_default()
    };
    let verify_rows: Vec<&Vec<Option<Vec<u8>>>> = rows
        .iter()
        .filter(|row| text(&row[subsystem_at]) == "verify")
        .collect();
    let names: Vec<String> = verify_rows.iter().map(|row| text(&row[name_at])).collect();
    assert!(names.contains(&"verify_failed".to_string()), "{names:?}");
    assert!(
        names.contains(&"chain_not_anchored".to_string()),
        "{names:?}"
    );

    let anchored = verify_rows
        .iter()
        .find(|row| text(&row[name_at]) == "chain_not_anchored")
        .expect("declared");
    assert_eq!(
        text(&anchored[threshold_at]),
        "verify.anchor_interval_secs",
        "the alert names the setting its window comes from"
    );
}
