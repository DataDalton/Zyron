//! A change stream never widens what its reader could read of the table.
//!
//! Row security is evaluated per reader over the change rows, so two
//! principals reading one stream see different rows while the position
//! moves once. Masking applies to the change rows the way it applies to the
//! table's rows. Creating a stream needs SELECT on the source, and a reader
//! holding PEEK but not SELECT can look at what is pending without moving
//! another consumer's position.
//!
//! Run: cargo test -p zyron-wire --test change_stream_security_test -- --nocapture

mod common;

use std::sync::Arc;

use common::{
    Storage, create_test_server_with_cdc_and_security, end_statement, exec_ddl, exec_dml,
    new_session,
};
use zyron_auth::{
    ClassificationLevel, GrantEntry, ObjectType, PolicyType, PrivilegeState, PrivilegeType,
    RlsCommand, RlsPolicy, RoleId, SecurityContext, SessionAttributes, UserId,
};
use zyron_executor::column::ScalarValue;
use zyron_executor::context::ExecutionContext;
use zyron_wire::connection::ServerState;

fn grant(
    sm: &zyron_auth::SecurityManager,
    role: RoleId,
    privilege: PrivilegeType,
    object_type: ObjectType,
    object_id: u32,
) {
    sm.privilege_store
        .grant(GrantEntry {
            grantee: role,
            privilege,
            object_type,
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

/// Runs one statement as a principal, planned under the principal's row
/// security, executed with the principal's security context, and ended the
/// way a connection ends it so a stream read advances at commit
async fn run_as(
    server: &Arc<ServerState>,
    sm: &Arc<zyron_auth::SecurityManager>,
    role: RoleId,
    sql: &str,
) -> Result<Vec<Vec<ScalarValue>>, zyron_common::ZyronError> {
    let stmt = zyron_parser::parse(sql)
        .expect("parse")
        .into_iter()
        .next()
        .expect("one statement");
    let sc = security_context(role);
    let provider: Arc<dyn zyron_planner::RowSecurityProvider> = Arc::new(
        zyron_wire::row_security::SmRowSecurityProvider::new(Arc::clone(sm), &sc),
    );
    let plan = zyron_planner::plan_with_security(
        &server.catalog,
        zyron_catalog::DatabaseId(1),
        vec!["zyron_test".into()],
        stmt,
        Some(provider),
        None,
    )
    .await?;
    let mut txn = server
        .txn_manager
        .begin(zyron_storage::txn::IsolationLevel::ReadCommitted)?;
    let snapshot = server.txn_manager.refresh_snapshot(&txn);
    let mut ctx = ExecutionContext::new(
        server.catalog.clone(),
        server.wal.clone(),
        server.buffer_pool.clone(),
        server.disk_manager.clone(),
        txn.txn_id,
        snapshot,
    );
    ctx.row_locks = Some(Arc::clone(server.txn_manager.lock_table()));
    ctx.intent_locks = Some(Arc::clone(server.txn_manager.intent_locks()));
    ctx.heap_files = Some(Arc::clone(&server.heap_files));
    ctx.btree_indexes = Some(Arc::clone(&server.btree_indexes));
    ctx.doc_registry = Some(Arc::clone(&server.doc_registry));
    ctx.set_media_store(Arc::clone(&server.media_store));
    ctx.set_key_store(Arc::clone(&server.key_store));
    ctx.security_manager = Some(Arc::clone(sm));
    ctx.security_context = Some(Arc::new(sc));
    common::install_change_reads(server, &mut ctx);
    let ctx = Arc::new(ctx);
    let result = zyron_executor::execute(plan, &ctx).await;
    match result {
        Ok(batches) => {
            end_statement(server, &ctx, &mut txn).await?;
            Ok(batches
                .iter()
                .flat_map(|b| {
                    (0..b.num_rows)
                        .map(|r| b.columns.iter().map(|c| c.get_scalar(r)).collect())
                        .collect::<Vec<Vec<ScalarValue>>>()
                })
                .collect())
        }
        Err(e) => {
            let _ = server.txn_manager.abort(&mut txn);
            Err(e)
        }
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn two_principals_see_their_own_rows_and_the_position_moves_once() {
    for storage in Storage::BOTH {
        two_principals_see_their_own_rows_and_the_position_moves_once_on(storage).await;
    }
}

async fn two_principals_see_their_own_rows_and_the_position_moves_once_on(storage: Storage) {
    println!("{storage}");
    let (server, schema, sm, _tmp) = create_test_server_with_cdc_and_security().await;
    let mut session = new_session();
    exec_ddl(
        &server,
        &mut session,
        &storage.create("CREATE TABLE orders (id BIGINT PRIMARY KEY, region TEXT, total BIGINT)"),
    )
    .await
    .expect("create");
    exec_ddl(
        &server,
        &mut session,
        "ALTER TABLE orders SET (change_data_feed = true)",
    )
    .await
    .expect("feed on");
    exec_ddl(
        &server,
        &mut session,
        "CREATE CHANGE STREAM s ON TABLE orders",
    )
    .await
    .expect("create the stream");
    let table_id = server
        .catalog
        .get_table(schema, "orders")
        .expect("table")
        .id
        .0;
    let stream_id = server
        .catalog
        .resolve_change_stream(zyron_catalog::DatabaseId(1), "s")
        .expect("stream")
        .id;

    // Two principals, each allowed one region's rows
    let eu = RoleId(11);
    let us = RoleId(12);
    for (role, region) in [(eu, "eu"), (us, "us")] {
        grant(
            &sm,
            role,
            PrivilegeType::Select,
            ObjectType::Table,
            table_id,
        );
        grant(
            &sm,
            role,
            PrivilegeType::Select,
            ObjectType::ChangeStream,
            stream_id,
        );
        grant(
            &sm,
            role,
            PrivilegeType::Peek,
            ObjectType::ChangeStream,
            stream_id,
        );
        sm.rls_store
            .add_policy(RlsPolicy {
                id: role.0,
                name: format!("only_{region}"),
                table_id,
                command: RlsCommand::Select,
                policy_type: PolicyType::Permissive,
                roles: vec![role],
                using_expr: Some(format!("region = '{region}'")),
                check_expr: None,
                enabled: true,
            })
            .expect("policy");
    }
    exec_dml(
        &server,
        "INSERT INTO orders VALUES (1, 'eu', 10), (2, 'us', 20), (3, 'eu', 30)",
    )
    .await;

    // Each reader peeks and sees only its region
    let eu_rows = run_as(
        &server,
        &sm,
        eu,
        "SELECT id FROM s WITH (peek => true) ORDER BY id",
    )
    .await
    .expect("eu peeks");
    assert_eq!(
        eu_rows,
        vec![vec![ScalarValue::Int64(1)], vec![ScalarValue::Int64(3)]]
    );
    let us_rows = run_as(
        &server,
        &sm,
        us,
        "SELECT id FROM s WITH (peek => true) ORDER BY id",
    )
    .await
    .expect("us peeks");
    assert_eq!(us_rows, vec![vec![ScalarValue::Int64(2)]]);

    // A consume by one reader moves the position once, for everyone
    let consumed = run_as(&server, &sm, eu, "SELECT id FROM s ORDER BY id")
        .await
        .expect("eu consumes");
    assert_eq!(consumed.len(), 2);
    let after = run_as(&server, &sm, us, "SELECT id FROM s WITH (peek => true)")
        .await
        .expect("us peeks again");
    assert!(
        after.is_empty(),
        "the position moved past every change, not only the reader's rows"
    );

    // The same predicates narrow a table_changes read
    exec_dml(&server, "INSERT INTO orders VALUES (4, 'us', 40)").await;
    let eu_changes = run_as(
        &server,
        &sm,
        eu,
        "SELECT id FROM table_changes(orders, 0, LATEST) ORDER BY id",
    )
    .await
    .expect("eu reads changes");
    assert_eq!(
        eu_changes,
        vec![vec![ScalarValue::Int64(1)], vec![ScalarValue::Int64(3)]]
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn masking_applies_to_change_rows() {
    for storage in Storage::BOTH {
        masking_applies_to_change_rows_on(storage).await;
    }
}

async fn masking_applies_to_change_rows_on(storage: Storage) {
    println!("{storage}");
    let (server, schema, sm, _tmp) = create_test_server_with_cdc_and_security().await;
    let mut session = new_session();
    exec_ddl(
        &server,
        &mut session,
        &storage.create("CREATE TABLE people (id BIGINT PRIMARY KEY, email TEXT)"),
    )
    .await
    .expect("create");
    exec_ddl(
        &server,
        &mut session,
        "ALTER TABLE people SET (change_data_feed = true)",
    )
    .await
    .expect("feed on");
    let table_id = server
        .catalog
        .get_table(schema, "people")
        .expect("table")
        .id
        .0;
    let email_id = server
        .catalog
        .get_table(schema, "people")
        .expect("table")
        .live_columns()
        .find(|c| c.name == "email")
        .expect("column")
        .id
        .0;
    let reader = RoleId(21);
    grant(
        &sm,
        reader,
        PrivilegeType::Select,
        ObjectType::Table,
        table_id,
    );
    sm.masking_policy_store
        .add_policy(zyron_auth::MaskingPolicy {
            id: 1,
            name: "hide_email".to_string(),
            table_id,
            column_id: email_id,
            function: zyron_auth::MaskFunction::Email,
            exempt_roles: Vec::new(),
            enabled: true,
        })
        .expect("masking policy");
    exec_dml(&server, "INSERT INTO people VALUES (1, 'ann@example.com')").await;

    let plain = run_as(&server, &sm, reader, "SELECT email FROM people")
        .await
        .expect("reads");
    let changed = run_as(
        &server,
        &sm,
        reader,
        "SELECT email FROM table_changes(people, 0, LATEST)",
    )
    .await
    .expect("reads changes");
    assert_eq!(
        plain, changed,
        "the change row is masked exactly as the table row is"
    );
    assert_ne!(
        changed[0][0],
        ScalarValue::Utf8("ann@example.com".to_string()),
        "the address is not readable through the feed"
    );
}

/// A NULL mask withholds the cell. The policy's function answers no text,
/// which is what the row shows, rather than the value it was written to hide
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_null_mask_withholds_the_cell() {
    let (server, schema, sm, _tmp) = create_test_server_with_cdc_and_security().await;
    let mut session = new_session();
    exec_ddl(
        &server,
        &mut session,
        "CREATE TABLE staff (id BIGINT PRIMARY KEY, salary BIGINT, note TEXT)",
    )
    .await
    .expect("create");
    let table = server.catalog.get_table(schema, "staff").expect("table");
    let column_id = |name: &str| {
        table
            .live_columns()
            .find(|c| c.name == name)
            .expect("column")
            .id
            .0
    };
    let reader = RoleId(22);
    grant(
        &sm,
        reader,
        PrivilegeType::Select,
        ObjectType::Table,
        table.id.0,
    );
    sm.masking_policy_store
        .add_policy(zyron_auth::MaskingPolicy {
            id: 2,
            name: "hide_note".to_string(),
            table_id: table.id.0,
            column_id: column_id("note"),
            function: zyron_auth::MaskFunction::Null,
            exempt_roles: Vec::new(),
            enabled: true,
        })
        .expect("masking policy");
    // A mask on a column that holds no text withholds nothing from a
    // cleared reader, since the function has no text to mask
    sm.masking_policy_store
        .add_policy(zyron_auth::MaskingPolicy {
            id: 3,
            name: "hash_salary".to_string(),
            table_id: table.id.0,
            column_id: column_id("salary"),
            function: zyron_auth::MaskFunction::Hash,
            exempt_roles: Vec::new(),
            enabled: true,
        })
        .expect("masking policy");
    exec_dml(&server, "INSERT INTO staff VALUES (1, 4200, 'quiet')").await;

    let rows = run_as(&server, &sm, reader, "SELECT id, salary, note FROM staff")
        .await
        .expect("reads");
    assert_eq!(rows.len(), 1);
    assert_eq!(rows[0][0], ScalarValue::Int64(1));
    assert_eq!(rows[0][1], ScalarValue::Int64(4200));
    assert_eq!(rows[0][2], ScalarValue::Null, "the note is withheld");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn creating_a_stream_needs_select_on_the_source() {
    for storage in Storage::BOTH {
        creating_a_stream_needs_select_on_the_source_on(storage).await;
    }
}

async fn creating_a_stream_needs_select_on_the_source_on(storage: Storage) {
    println!("{storage}");
    let (server, schema, sm, _tmp) = create_test_server_with_cdc_and_security().await;
    let mut session = new_session();
    exec_ddl(
        &server,
        &mut session,
        &storage.create("CREATE TABLE guarded (id BIGINT PRIMARY KEY, v BIGINT)"),
    )
    .await
    .expect("create");
    exec_ddl(
        &server,
        &mut session,
        "ALTER TABLE guarded SET (change_data_feed = true)",
    )
    .await
    .expect("feed on");
    let table_id = server
        .catalog
        .get_table(schema, "guarded")
        .expect("table")
        .id
        .0;

    let role = RoleId(31);
    grant(
        &sm,
        role,
        PrivilegeType::ManageChangeStream,
        ObjectType::Schema,
        schema.0,
    );
    session.as_mut().expect("session").security_context = Some(security_context(role));
    let refused = exec_ddl(
        &server,
        &mut session,
        "CREATE CHANGE STREAM s ON TABLE guarded",
    )
    .await
    .expect_err("no SELECT on the source");
    assert!(
        refused.to_lowercase().contains("select") || refused.contains("permission"),
        "{refused}"
    );
    assert!(
        server
            .catalog
            .resolve_change_stream(zyron_catalog::DatabaseId(1), "s")
            .is_err(),
        "the refused stream leaves no entry"
    );

    grant(
        &sm,
        role,
        PrivilegeType::Select,
        ObjectType::Table,
        table_id,
    );
    exec_ddl(
        &server,
        &mut session,
        "CREATE CHANGE STREAM s ON TABLE guarded",
    )
    .await
    .expect("with SELECT it is created");
}
