//! Inline external endpoints on a streaming job are anonymous external
//! sources and sinks, so they require the privilege creating the named
//! object requires. Without that gate, any role holding the schema-level
//! streaming privilege could read or write any URI the server process can
//! reach by inlining FILE '...' instead of naming an admin-created object,
//! which the named form never allows: a named endpoint is created under
//! CreateExternalSource or CreateExternalSink and delegated through Usage.

mod common;

use std::sync::Arc;

use common::{create_test_server_with_security, exec_ddl, new_session};
use zyron_auth::{
    ClassificationLevel, GrantEntry, ObjectType, PrivilegeState, PrivilegeType, QueryLimits,
    RoleId, SecurityContext, SessionAttributes, UserId,
};

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
        UserId(1),
        role,
        vec![role],
        vec![role],
        ClassificationLevel::Public,
        attrs,
        None,
        QueryLimits::default(),
    )
}

#[tokio::test]
async fn an_inline_file_source_requires_the_external_source_privilege() {
    let (server, schema, sm, tmp) = create_test_server_with_security().await;
    let mut session = new_session();
    exec_ddl(
        &server,
        &mut session,
        "CREATE TABLE stream_tgt (a BIGINT, b TEXT)",
    )
    .await
    .expect("create target");
    let tgt_id = server
        .catalog
        .get_table(schema, "stream_tgt")
        .expect("table")
        .id;

    // The role holds everything the job needs except the external-source
    // privilege the inline endpoint stands in for
    let role = RoleId(5);
    grant(
        &sm,
        role,
        PrivilegeType::CreateStreamingJob,
        ObjectType::Schema,
        schema.0,
    );
    grant(&sm, role, PrivilegeType::Insert, ObjectType::Table, tgt_id.0);
    session.as_mut().expect("session").security_context = Some(security_context(role));

    let csv = tmp.path().join("rows.csv");
    std::fs::write(&csv, "1,one\n2,two\n").expect("write csv");
    let sql = format!(
        "CREATE STREAMING JOB leak AS SELECT a, b FROM FILE 'file://{}' FORMAT CSV \
         COLUMNS (a BIGINT, b TEXT) INTO stream_tgt",
        csv.display()
    );
    let err = exec_ddl(&server, &mut session, &sql)
        .await
        .expect_err("the inline file source must be refused");
    assert!(
        err.contains("CreateExternalSource"),
        "the refusal names the missing privilege: {err}"
    );
    assert!(
        server.catalog.get_streaming_job(schema, "leak").is_none(),
        "the refused job leaves no catalog entry"
    );

    // With the privilege the named form requires, the gate opens. Creation
    // then stops at the runtime dependency this harness does not configure,
    // which proves the refusal above was the privilege and nothing else
    grant(
        &sm,
        role,
        PrivilegeType::CreateExternalSource,
        ObjectType::Schema,
        schema.0,
    );
    let err = exec_ddl(&server, &mut session, &sql)
        .await
        .expect_err("the harness configures no CDC registry");
    assert!(
        !err.contains("permission denied"),
        "the gate opens once the privilege is granted: {err}"
    );
    assert!(
        err.contains("CDC registry"),
        "creation proceeds to the runtime dependency: {err}"
    );
}

#[tokio::test]
async fn an_inline_file_sink_requires_the_external_sink_privilege() {
    let (server, schema, sm, tmp) = create_test_server_with_security().await;
    let mut session = new_session();
    exec_ddl(
        &server,
        &mut session,
        "CREATE TABLE stream_src (a BIGINT, b TEXT)",
    )
    .await
    .expect("create source");
    // A Zyron-table source streams through CDC, which the binder requires
    // to be enabled before it accepts the job. The DDL toggle needs a CDC
    // registry this harness does not configure, so the catalog flag is set
    // directly, the subject here is the privilege gate alone
    {
        let entry = server.catalog.get_table(schema, "stream_src").expect("table");
        let mut entry = (*entry).clone();
        entry.cdf_enabled = true;
        server
            .catalog
            .update_table(entry)
            .await
            .expect("enable change data feed");
    }
    let src_id = server
        .catalog
        .get_table(schema, "stream_src")
        .expect("table")
        .id;

    let role = RoleId(6);
    grant(
        &sm,
        role,
        PrivilegeType::CreateStreamingJob,
        ObjectType::Schema,
        schema.0,
    );
    grant(&sm, role, PrivilegeType::Select, ObjectType::Table, src_id.0);
    session.as_mut().expect("session").security_context = Some(security_context(role));

    let out = tmp.path().join("out.csv");
    let sql = format!(
        "CREATE STREAMING JOB spill AS SELECT a, b FROM stream_src \
         INTO FILE 'file://{}' FORMAT CSV",
        out.display()
    );
    let err = exec_ddl(&server, &mut session, &sql)
        .await
        .expect_err("the inline file sink must be refused");
    assert!(
        err.contains("CreateExternalSink"),
        "the refusal names the missing privilege: {err}"
    );
    assert!(
        server.catalog.get_streaming_job(schema, "spill").is_none(),
        "the refused job leaves no catalog entry"
    );

    grant(
        &sm,
        role,
        PrivilegeType::CreateExternalSink,
        ObjectType::Schema,
        schema.0,
    );
    let err = exec_ddl(&server, &mut session, &sql)
        .await
        .expect_err("the harness configures no CDC registry");
    assert!(
        !err.contains("permission denied"),
        "the gate opens once the privilege is granted: {err}"
    );
}
