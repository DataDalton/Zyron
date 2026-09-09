//! A privilege change survives the node that made it.
//!
//! `SecurityManager` holds two things for every grant: a row in auth storage
//! and an entry in the in-memory privilege store. Startup reads the rows and
//! fills the store from them, so a grant written only to the store decides
//! every check until the process stops and is gone when it starts again. The
//! statement reports success either way, and nothing a running node can be
//! asked would tell the two apart.
//!
//! What each case here does is read the rows back through a second storage
//! handle over the same heap, which is what a restart does.

mod common;

use std::sync::Arc;

use common::{create_test_server_with_security, exec_ddl, new_session};
use zyron_auth::{ObjectType, PrivilegeType, SecurityManager};

/// A second manager over the storage the server writes to, which is what a
/// restart builds.
///
/// The storage handle is reused rather than reopened, so what this reads is
/// exactly what was written to the auth heaps and nothing the first manager
/// held in memory reaches it. A manager loads every store from storage as it
/// starts, so anything a statement wrote only to memory is absent here
async fn reload(sm: &Arc<SecurityManager>) -> SecurityManager {
    SecurityManager::new(Arc::clone(&sm.auth_storage))
        .await
        .expect("a manager loads what storage holds")
}

/// The privileges a role holds after a reload, rendered so a failure names
/// what was found rather than only that a count differed
fn held(sm: &SecurityManager, role: &str) -> Vec<String> {
    let Some(r) = sm.lookup_role(role) else {
        return Vec::new();
    };
    let mut out: Vec<String> = sm
        .privilege_store
        .grants_for_role(r.id)
        .iter()
        .map(|g| format!("{:?}:{:?}:{}", g.privilege, g.object_type, g.object_id))
        .collect();
    out.sort();
    out
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_granted_privilege_is_still_held_after_a_reload() {
    let (server, schema, sm, _tmp) = create_test_server_with_security().await;
    let mut s = new_session();

    exec_ddl(
        &server,
        &mut s,
        "CREATE TABLE dur_t (id BIGINT PRIMARY KEY)",
    )
    .await
    .expect("create the table");
    exec_ddl(&server, &mut s, "CREATE ROLE dur_reader")
        .await
        .expect("create the role");
    exec_ddl(&server, &mut s, "GRANT SELECT ON dur_t TO dur_reader")
        .await
        .expect("grant on the table");

    let table = server
        .catalog
        .get_table(schema, "dur_t")
        .expect("the table is in the catalog")
        .id
        .0;

    let reloaded = reload(&sm).await;
    assert!(
        reloaded.lookup_role("dur_reader").is_some(),
        "the role was not read back"
    );
    let after = held(&reloaded, "dur_reader");
    assert!(
        after
            .iter()
            .any(|g| g.starts_with(&format!("{:?}:", PrivilegeType::Select))
                && g.ends_with(&format!(":{table}"))),
        "the grant did not survive the reload, only these did: {after:?}"
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_revoked_privilege_does_not_come_back_after_a_reload() {
    let (server, _schema, sm, _tmp) = create_test_server_with_security().await;
    let mut s = new_session();

    exec_ddl(
        &server,
        &mut s,
        "CREATE TABLE dur_rv_t (id BIGINT PRIMARY KEY)",
    )
    .await
    .expect("create the table");
    exec_ddl(&server, &mut s, "CREATE ROLE dur_rv_keeps")
        .await
        .expect("create the role that keeps its grant");
    exec_ddl(&server, &mut s, "CREATE ROLE dur_rv_loses")
        .await
        .expect("create the role that loses its grant");
    exec_ddl(&server, &mut s, "GRANT SELECT ON dur_rv_t TO dur_rv_keeps")
        .await
        .expect("grant to the first role");
    exec_ddl(&server, &mut s, "GRANT SELECT ON dur_rv_t TO dur_rv_loses")
        .await
        .expect("grant to the second role");
    exec_ddl(
        &server,
        &mut s,
        "REVOKE SELECT ON dur_rv_t FROM dur_rv_loses",
    )
    .await
    .expect("revoke from the second role");

    let reloaded = reload(&sm).await;
    assert!(
        held(&reloaded, "dur_rv_loses").is_empty(),
        "the revoked grant came back: {:?}",
        held(&reloaded, "dur_rv_loses")
    );
    // A revoke that cleared more than it named would pass the check above
    assert!(
        !held(&reloaded, "dur_rv_keeps").is_empty(),
        "the revoke took the grant the other role held as well"
    );
}

/// A dropped table takes its grants with it.
///
/// A grant names its object by kind and id. The catalog's id allocator
/// recovers by scanning the live rows for the highest id it finds, so an id
/// stops being taken the moment the row naming it is gone, and the next
/// object created after a restart can be numbered what the dropped one was.
/// A grant left behind then decides access to a table its grantee was never
/// given anything on.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn dropping_a_table_takes_its_grants_with_it() {
    let (server, schema, sm, _tmp) = create_test_server_with_security().await;
    let mut s = new_session();

    exec_ddl(
        &server,
        &mut s,
        "CREATE TABLE gone_t (id BIGINT PRIMARY KEY)",
    )
    .await
    .expect("create the table");
    exec_ddl(&server, &mut s, "CREATE ROLE gone_reader")
        .await
        .expect("create the role");
    exec_ddl(&server, &mut s, "GRANT SELECT ON gone_t TO gone_reader")
        .await
        .expect("grant on the table");

    let table_id = server
        .catalog
        .get_table(schema, "gone_t")
        .expect("the table is in the catalog")
        .id
        .0;
    assert!(
        !sm.privilege_store
            .grants_for_object(ObjectType::Table, table_id)
            .is_empty(),
        "the grant was not recorded, so the case would pass for the wrong reason"
    );

    exec_ddl(&server, &mut s, "DROP TABLE gone_t")
        .await
        .expect("drop the table");

    assert!(
        sm.privilege_store
            .grants_for_object(ObjectType::Table, table_id)
            .is_empty(),
        "the dropped table's grants are still recorded against id {table_id}"
    );
    // And not on disk either, or the next start would read them back
    let reloaded = reload(&sm).await;
    assert!(
        reloaded
            .privilege_store
            .grants_for_object(ObjectType::Table, table_id)
            .is_empty(),
        "the dropped table's grants came back from storage against id {table_id}"
    );
}

/// Dropping one object leaves every other object's grants alone.
///
/// The grants for an object are removed in one pass over the store rather
/// than one scan per grant, and a pass that matched too broadly would take
/// unrelated grants with it and report success either way
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn dropping_one_table_leaves_another_tables_grants_alone() {
    let (server, schema, sm, _tmp) = create_test_server_with_security().await;
    let mut s = new_session();

    exec_ddl(
        &server,
        &mut s,
        "CREATE TABLE keep_t (id BIGINT PRIMARY KEY)",
    )
    .await
    .expect("create the table that stays");
    exec_ddl(&server, &mut s, "CREATE TABLE go_t (id BIGINT PRIMARY KEY)")
        .await
        .expect("create the table that goes");
    exec_ddl(&server, &mut s, "CREATE ROLE multi_reader")
        .await
        .expect("create the role");
    exec_ddl(&server, &mut s, "GRANT SELECT ON keep_t TO multi_reader")
        .await
        .expect("grant on the table that stays");
    exec_ddl(&server, &mut s, "GRANT INSERT ON keep_t TO multi_reader")
        .await
        .expect("second grant on the table that stays");
    exec_ddl(&server, &mut s, "GRANT SELECT ON go_t TO multi_reader")
        .await
        .expect("grant on the table that goes");

    let keep_id = server
        .catalog
        .get_table(schema, "keep_t")
        .expect("the table that stays is in the catalog")
        .id
        .0;
    let before = sm
        .privilege_store
        .grants_for_object(ObjectType::Table, keep_id)
        .len();
    assert!(before >= 2, "both grants were recorded, found {before}");

    exec_ddl(&server, &mut s, "DROP TABLE go_t")
        .await
        .expect("drop the table that goes");

    let after = sm
        .privilege_store
        .grants_for_object(ObjectType::Table, keep_id)
        .len();
    assert_eq!(
        after,
        before,
        "dropping one table took {} of another table's grants with it",
        before - after
    );

    let reloaded = reload(&sm).await;
    assert_eq!(
        reloaded
            .privilege_store
            .grants_for_object(ObjectType::Table, keep_id)
            .len(),
        before,
        "the surviving table's grants did not come back from storage"
    );
}
