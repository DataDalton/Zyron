//! Every statement, run on a real group, checked on every member.
//!
//! `replication_class` decides how a statement reaches the rest of a
//! consensus group. Until this suite existed that decision was a claim: the
//! only statements any test had ever run against a group were CREATE TABLE,
//! CREATE PROCEDURE, MERGE INTO and ALTER CLUSTER. Everything else was
//! classified by reading the handler and writing a comment, and a wildcard at
//! the end of the match meant a statement nobody had thought about was
//! refused on every cluster and worked on every single node, which is the one
//! shape of bug that never shows up in development.
//!
//! What a case proves here is stronger than "it did not error". The probe is
//! read on the leader before and after, so a statement that quietly did
//! nothing fails rather than passing, and it is read on every member
//! afterwards, so a statement whose effects stopped at the leader fails too.
//!
//! Run: `cargo test -p zyron-server --test replication_conformance_test`

mod common;

use std::sync::Arc;
use std::time::Duration;

use common::{Group, Node, WireClient};

/// How a statement is shown to reach the group.
///
/// Either way the reading has to change on the leader and then agree on every
/// member. A statement that changed nothing fails as loudly as one whose
/// effects stopped where it ran
enum Proof {
    /// Read out of the node's own catalog
    Catalog(fn(&Node) -> String),
    /// Count the rows a table holds, through the node's own storage. Catalog
    /// state can agree while the rows behind it do not, so a statement that
    /// moves rows is checked on the rows
    Rows(&'static str),
}

/// Reads whichever proof a case carries, on one node
async fn read_proof(node: &Node, proof: &Proof) -> String {
    match proof {
        Proof::Catalog(probe) => probe(node),
        Proof::Rows(table) => format!("{table}:{}", node.count(table).await),
    }
}

/// One statement and what proves it replicated.
struct Case {
    /// What the statement under test is called in a failure report
    name: &'static str,
    /// Run on the leader first, so the statement under test is legal. Not
    /// itself under test, though a failure here still fails the case, because
    /// a case that cannot be set up proves nothing
    setup: &'static [&'static str],
    /// The statement under test
    sql: &'static str,
    proof: Proof,
}

// ---------------------------------------------------------------------------
// Probes
//
// Each reads one kind of object out of a node's own catalog and renders it as
// a string. Comparing the strings across members is what turns "the leader
// says it worked" into "every member holds the same thing"
// ---------------------------------------------------------------------------

fn render(mut items: Vec<String>) -> String {
    items.sort();
    items.join(",")
}

fn analyzers(node: &Node) -> String {
    render(
        node.catalog
            .list_analyzers()
            .iter()
            .map(|a| format!("{}:{}", a.name, a.tokenizer))
            .collect(),
    )
}

fn synonym_dictionaries(node: &Node) -> String {
    render(
        node.catalog
            .list_synonym_dictionaries()
            .iter()
            .map(|d| format!("{}:{}", d.name, d.rules.len()))
            .collect(),
    )
}

fn user_types(node: &Node) -> String {
    render(
        node.catalog
            .list_user_types()
            .iter()
            .map(|t| format!("{}:{}", t.name, t.storage_type_id))
            .collect(),
    )
}

fn collations(node: &Node) -> String {
    render(
        node.catalog
            .list_collations()
            .iter()
            .map(|c| format!("{}:{}", c.name, c.locale))
            .collect(),
    )
}

fn resilience_policies(node: &Node) -> String {
    render(
        node.catalog
            .list_resilience_policies()
            .iter()
            .map(|p| format!("{}:{:?}:{}", p.name, p.kind, p.max_concurrent))
            .collect(),
    )
}

fn endpoints(node: &Node) -> String {
    render(
        node.catalog
            .list_endpoints()
            .iter()
            .map(|e| format!("{}:{}", e.name, e.path))
            .collect(),
    )
}

fn external_sinks(node: &Node) -> String {
    render(
        node.catalog
            .list_external_sinks()
            .iter()
            .map(|s| {
                let credentials = match (s.credential_provider, s.credential_key_id.is_some()) {
                    (Some(kind), _) => kind.catalog_name(),
                    (None, true) => "SEALED",
                    (None, false) => "NONE",
                };
                format!("{}:{}:{}", s.name, s.uri, credentials)
            })
            .collect(),
    )
}

fn external_sources(node: &Node) -> String {
    render(
        node.catalog
            .list_external_sources()
            .iter()
            .map(|s| {
                let credentials = match (s.credential_provider, s.credential_key_id.is_some()) {
                    (Some(kind), _) => kind.catalog_name(),
                    (None, true) => "SEALED",
                    (None, false) => "NONE",
                };
                format!(
                    "{}:{}:{}:paused={}:{}",
                    s.name,
                    s.uri,
                    s.columns.len(),
                    s.paused,
                    credentials
                )
            })
            .collect(),
    )
}

/// Row counts the planner prices against, on this node.
///
/// Read out of the node's own statistics rather than by counting, because the
/// question is whether this member measured the table at all
fn analyzed_rows(node: &Node) -> String {
    let Ok(entry) = node.catalog.get_table(node.schema, "conf_stats") else {
        return "conf_stats:absent".to_string();
    };
    match node.catalog.get_stats(entry.id) {
        Some(stats) => format!("conf_stats:{}", stats.0.row_count),
        None => "conf_stats:unanalyzed".to_string(),
    }
}

/// Every function a node holds, with the body that decides what it returns.
///
/// The body is in the rendering because a REPLACE rewrites it and a member
/// that kept the old one would agree on the name and compute something else
fn functions(node: &Node) -> String {
    render(
        node.catalog
            .list_functions()
            .iter()
            .map(|f| format!("{}/{}/{}", f.name, f.param_types.len(), f.body_sql))
            .collect(),
    )
}

/// Every aggregate a node holds, with the functions it folds through
fn aggregates(node: &Node) -> String {
    render(
        node.catalog
            .list_aggregates()
            .iter()
            .map(|a| {
                format!(
                    "{}/{}/{}",
                    a.name,
                    a.sfunc_name,
                    a.finalfunc_name.as_deref().unwrap_or("none")
                )
            })
            .collect(),
    )
}

fn tables(node: &Node) -> String {
    render(
        node.catalog
            .list_all_tables()
            .iter()
            .map(|t| format!("{}:{}", t.name, t.columns.len()))
            .collect(),
    )
}

/// The security manager a node holds.
///
/// Roles, users and grants are not catalog objects, so the probes over them
/// read here rather than through the node's catalog. Every node in this
/// harness carries one, which is what makes a privilege statement legal
fn security(node: &Node) -> &zyron_auth::SecurityManager {
    node._server
        .security_manager
        .as_deref()
        .expect("every node in this harness carries a security manager")
}

/// The roles and users the privilege cases name.
///
/// A fixed list rather than everything the store holds, because the harness
/// creates principals of its own and a probe over all of them would report a
/// difference no case caused
const PRINCIPALS: &[&str] = &[
    "conf_role",
    "conf_role_rn",
    "conf_role_rn2",
    "conf_role_drop",
    "conf_role_grant",
    "conf_role_grant_cs",
    "conf_role_revoke",
    "conf_role_exec",
    "conf_role_exec_rv",
    "conf_user",
    "conf_user_rn",
    "conf_user_rn2",
    "conf_user_drop",
];

/// Which of those principals this node holds.
///
/// Read by name rather than by id, because an id is allocated by whichever
/// node runs the statement and two members holding the same principal can
/// number it differently while agreeing on everything a grant is written
/// against. A user carries a companion role, so both kinds are rendered and a
/// member that created one without the other reads differently here
fn principals(node: &Node) -> String {
    let sm = security(node);
    let mut items = Vec::new();
    for name in PRINCIPALS {
        if sm.lookup_role(name).is_some() {
            items.push(format!("role:{name}"));
        }
        if sm.lookup_user(name).is_some() {
            items.push(format!("user:{name}"));
        }
    }
    render(items)
}

/// Every privilege those principals hold on this node.
///
/// The object kind and id are in the rendering because a grant that landed on
/// a different object than the leader recorded would agree on the privilege
/// and open a different table. The state is in it because a DENY and a GRANT
/// are the same entry with opposite meanings
fn grants(node: &Node) -> String {
    let sm = security(node);
    let mut items = Vec::new();
    for name in PRINCIPALS {
        let Some(role) = sm.lookup_role(name) else {
            continue;
        };
        for g in sm.privilege_store.grants_for_role(role.id) {
            items.push(format!(
                "{name}:{:?}:{:?}:{}:{:?}",
                g.privilege, g.object_type, g.object_id, g.state
            ));
        }
    }
    render(items)
}

/// Every index a node holds.
///
/// The column list and the uniqueness are in the rendering because an index
/// that arrived naming different columns would agree on the name and answer
/// different rows, and a search index carries the kind it was built as
fn indexes(node: &Node) -> String {
    render(
        node.catalog
            .list_all_indexes()
            .iter()
            .map(|i| {
                let cols: Vec<String> = i
                    .columns
                    .iter()
                    .map(|c| c.column_id.0.to_string())
                    .collect();
                format!("{}:{}:{}", i.name, cols.join("+"), i.unique)
            })
            .collect(),
    )
}

/// Every view, with the query it stands for. A member holding the name and a
/// different definition would answer a different table
fn views(node: &Node) -> String {
    render(
        node.catalog
            .list_views()
            .iter()
            .map(|v| format!("{}:{}", v.name, v.definition_sql.len()))
            .collect(),
    )
}

fn mviews(node: &Node) -> String {
    render(
        node.catalog
            .list_mviews()
            .iter()
            .map(|v| format!("{}:{}", v.name, v.definition_sql.len()))
            .collect(),
    )
}

/// Every sequence with the bounds that decide what it hands out next
fn sequences(node: &Node) -> String {
    render(
        node.catalog
            .list_sequences()
            .iter()
            .map(|s| format!("{}:{}:{}:{}", s.name, s.increment, s.min_value, s.max_value))
            .collect(),
    )
}

/// The schemas the schema cases name.
///
/// Named rather than listed, because a node registers the `zyron_sys` schemas
/// its own system catalog needs as it starts and on first use, so two members
/// list different system schemas without any statement having run
fn schemas(node: &Node) -> String {
    const NAMED: &[&str] = &["conf_schema", "conf_schema_drop"];
    let held = node.catalog.list_schemas();
    render(
        NAMED
            .iter()
            .filter(|name| held.iter().any(|s| s.name == **name))
            .map(|name| name.to_string())
            .collect(),
    )
}

/// Every procedure with its parameter count and body length, so a member that
/// stored a different body fails rather than agreeing on the name
fn procedures(node: &Node) -> String {
    render(
        node.catalog
            .list_procedures()
            .iter()
            .map(|p| format!("{}:{}:{}", p.name, p.param_names.len(), p.body_sql.len()))
            .collect(),
    )
}

/// Every trigger with when it fires and what it fires on
fn triggers(node: &Node) -> String {
    render(
        node.catalog
            .list_triggers()
            .iter()
            .map(|t| format!("{}:{}:{}:{}", t.name, t.table_id, t.timing, t.events))
            .collect(),
    )
}

/// Every comment, which is the object it is on and the text
fn comments(node: &Node) -> String {
    render(
        node.catalog
            .list_comments()
            .iter()
            .map(|c| {
                format!(
                    "{}:{}:{}:{}",
                    c.object_type, c.object_name, c.column_name, c.comment
                )
            })
            .collect(),
    )
}

fn publications(node: &Node) -> String {
    render(
        node.catalog
            .list_publications()
            .iter()
            .map(|p| format!("{}:{}", p.name, p.id))
            .collect(),
    )
}

/// Which tables each publication carries.
///
/// Membership lives in rows of its own, so the publication row is unchanged
/// by an ADD TABLE and reading it alone would call the statement a no-op
fn publication_tables(node: &Node) -> String {
    render(
        node.catalog
            .list_publications()
            .iter()
            .map(|p| {
                let mut tables: Vec<String> = node
                    .catalog
                    .list_publication_tables(p.id)
                    .iter()
                    .map(|t| t.table_id.0.to_string())
                    .collect();
                tables.sort();
                format!("{}:[{}]", p.name, tables.join(","))
            })
            .collect(),
    )
}

/// The tags each publication carries, which is all TAG and UNTAG change
fn publication_tags(node: &Node) -> String {
    render(
        node.catalog
            .list_publications()
            .iter()
            .map(|p| {
                let mut tags = p.tags.clone();
                tags.sort();
                format!("{}:[{}]", p.name, tags.join(","))
            })
            .collect(),
    )
}

/// The expectations each table carries, which is what an ADD or DROP
/// EXPECTATION changes and nothing else about the table does
fn expectations(node: &Node) -> String {
    render(
        node.catalog
            .list_all_tables()
            .iter()
            .flat_map(|t| {
                t.expectations
                    .iter()
                    .map(|e| format!("{}:{}", t.name, e.name))
                    .collect::<Vec<_>>()
            })
            .collect(),
    )
}

/// The change feed flags a table carries, which is what ENABLE and DISABLE
/// FEATURE move and nothing else on the table does
fn table_features(node: &Node) -> String {
    render(
        node.catalog
            .list_all_tables()
            .iter()
            .map(|t| {
                format!(
                    "{}:{}:{}:{}",
                    t.name, t.cdf_enabled, t.versioning_enabled, t.lifecycle.soft_delete_enabled
                )
            })
            .collect(),
    )
}

/// The classification each column carries.
///
/// Read out of the security manager rather than the column row, because that
/// is where a classification is recorded and a probe over the catalog would
/// call the statement a no-op
fn column_classifications(node: &Node) -> String {
    let sm = security(node);
    render(
        node.catalog
            .list_all_tables()
            .iter()
            .flat_map(|t| {
                sm.classification_store
                    .classifications_for_table(t.id.0)
                    .iter()
                    .map(|c| format!("{}:{}:{:?}", t.name, c.column_id, c.level))
                    .collect::<Vec<_>>()
            })
            .collect(),
    )
}

/// Which tables are foreign, and where they point. A foreign table is a
/// catalog row like any other, so what proves it is the target it names
fn foreign_tables(node: &Node) -> String {
    render(
        node.catalog
            .list_all_tables()
            .iter()
            .filter(|t| t.foreign.is_foreign())
            .map(|t| format!("{}:{}:{}", t.name, t.foreign.peer, t.foreign.table))
            .collect(),
    )
}

/// The branches a node holds. A branch is not a catalog object, so this
/// reads the branch manager the server was built with
fn branches(node: &Node) -> String {
    let Some(mgr) = node._server.branch_manager.as_ref() else {
        return String::new();
    };
    render(mgr.list_branches().iter().map(|b| b.name.clone()).collect())
}

/// The table options a SET clause writes, which live on the lifecycle and
/// retention fields rather than on the column list
fn table_options(node: &Node) -> String {
    render(
        node.catalog
            .list_all_tables()
            .iter()
            .map(|t| {
                format!(
                    "{}:{}:{}",
                    t.name, t.time_travel_retention_secs, t.cdf_retention_days
                )
            })
            .collect(),
    )
}

/// Whether a table is verified and what its chain links with, which is what
/// `ALTER TABLE SET (immutable, verified)` writes
fn table_verification(node: &Node) -> String {
    render(
        node.catalog
            .list_all_tables()
            .iter()
            .map(|t| {
                format!(
                    "{}:{}:{}:{}",
                    t.name,
                    t.lifecycle.immutable,
                    t.lifecycle.verified,
                    t.lifecycle.chain_algorithm
                )
            })
            .collect(),
    )
}

/// The row TTL a table carries, which is the column it reads and how long
fn table_ttl(node: &Node) -> String {
    render(
        node.catalog
            .list_all_tables()
            .iter()
            .map(|t| {
                format!(
                    "{}:{}:{}:{}",
                    t.name,
                    t.lifecycle.ttl_column_id,
                    t.lifecycle.ttl_seconds,
                    t.lifecycle.ttl_action
                )
            })
            .collect(),
    )
}

/// How a table is clustered, which is the keys and the mode a CLUSTER BY sets
fn table_clustering(node: &Node) -> String {
    render(
        node.catalog
            .list_all_tables()
            .iter()
            .map(|t| {
                let keys: Vec<String> = t
                    .cluster
                    .keys
                    .iter()
                    .map(|k| k.column_id.to_string())
                    .collect();
                format!(
                    "{}:{}:{}:[{}]",
                    t.name,
                    t.cluster.mode,
                    t.cluster.schedule,
                    keys.join("+")
                )
            })
            .collect(),
    )
}

fn pipelines(node: &Node) -> String {
    render(
        node.catalog
            .list_pipelines()
            .iter()
            .map(|p| p.name.clone())
            .collect(),
    )
}

/// Every schedule with whether it is paused, which is what PAUSE and RESUME
/// change and nothing else about the row does
fn schedules(node: &Node) -> String {
    render(
        node.catalog
            .list_schedules()
            .iter()
            .map(|s| format!("{}:{}", s.name, s.paused))
            .collect(),
    )
}

fn event_handlers(node: &Node) -> String {
    render(
        node.catalog
            .list_event_handlers()
            .iter()
            .map(|h| h.name.clone())
            .collect(),
    )
}

fn version_tags(node: &Node) -> String {
    render(
        node.catalog
            .list_version_tags()
            .iter()
            .map(|v| v.name.clone())
            .collect(),
    )
}

fn security_maps(node: &Node) -> String {
    render(
        node.catalog
            .list_security_maps()
            .iter()
            .map(|m| format!("{:?}:{}:{}", m.kind, m.key, m.role_id))
            .collect(),
    )
}

fn streaming_jobs(node: &Node) -> String {
    render(
        node.catalog
            .list_streaming_jobs()
            .iter()
            .map(|j| format!("{}:{}", j.name, j.select_sql.len()))
            .collect(),
    )
}

/// Change streams with the count each has consumed per source, which is
/// the form of a position that reads the same on every member, plus the
/// columns and the flags a reset or a column change moves
fn change_streams(node: &Node) -> String {
    render(
        node.catalog
            .list_change_streams()
            .iter()
            .map(|s| {
                let consumed: Vec<String> = s
                    .position
                    .iter()
                    .map(|p| format!("{}={}", p.table_id, p.consumed))
                    .collect();
                format!(
                    "{}:{}:{:?}:{}:{}",
                    s.name,
                    consumed.join("+"),
                    s.columns.as_ref().map(|c| c.len()),
                    s.stale,
                    s.needs_attention
                )
            })
            .collect(),
    )
}

/// Tables with the feed settings a SET (cdf_...) moves
fn feed_settings(node: &Node) -> String {
    render(
        node.catalog
            .list_all_tables()
            .iter()
            .filter(|t| t.cdf_enabled)
            .map(|t| {
                format!(
                    "{}:{}:{}:{}",
                    t.name,
                    t.cdf.retention_micros,
                    t.cdf.before_image,
                    t.cdf.recorded_columns().len()
                )
            })
            .collect(),
    )
}

/// Tables with the columns they carry, so an ALTER that changed a column
/// reads differently from one that changed nothing.
///
/// Live columns only. A dropped column keeps its row and its ordinal so the
/// rows written before the drop still decode, and a probe over every column
/// would read a drop as having changed nothing
fn table_columns(node: &Node) -> String {
    render(
        node.catalog
            .list_all_tables()
            .iter()
            .map(|t| {
                let cols: Vec<String> = t
                    .live_columns()
                    .map(|c| format!("{}:{}", c.name, c.type_id as u8))
                    .collect();
                format!("{}[{}]", t.name, cols.join(","))
            })
            .collect(),
    )
}

// ---------------------------------------------------------------------------
// The cases
// ---------------------------------------------------------------------------

/// Statements this release says reach every member of a group.
///
/// A statement classified `Statement` or `Rows` belongs here. One that is
/// refused does not, and the refusal is covered by
/// `a_refused_statement_says_so_rather_than_half_running` below
const CASES: &[Case] = &[
    Case {
        name: "CREATE ANALYZER",
        setup: &[],
        sql: "CREATE ANALYZER conf_an AS (tokenizer = 'standard')",
        proof: Proof::Catalog(analyzers),
    },
    Case {
        name: "ALTER ANALYZER",
        setup: &["CREATE ANALYZER conf_an_alt AS (tokenizer = 'standard')"],
        sql: "ALTER ANALYZER conf_an_alt SET (tokenizer = 'ngram(3, 5)')",
        proof: Proof::Catalog(analyzers),
    },
    Case {
        name: "DROP ANALYZER",
        setup: &["CREATE ANALYZER conf_an_drop AS (tokenizer = 'standard')"],
        sql: "DROP ANALYZER conf_an_drop",
        proof: Proof::Catalog(analyzers),
    },
    Case {
        name: "CREATE SYNONYM DICTIONARY",
        setup: &[],
        sql: "CREATE SYNONYM DICTIONARY conf_syn (('car', 'automobile'))",
        proof: Proof::Catalog(synonym_dictionaries),
    },
    Case {
        name: "ALTER SYNONYM DICTIONARY",
        setup: &["CREATE SYNONYM DICTIONARY conf_syn_alt (('car', 'automobile'))"],
        sql: "ALTER SYNONYM DICTIONARY conf_syn_alt ADD ('bike', 'bicycle')",
        proof: Proof::Catalog(synonym_dictionaries),
    },
    Case {
        name: "DROP SYNONYM DICTIONARY",
        setup: &["CREATE SYNONYM DICTIONARY conf_syn_drop (('car', 'automobile'))"],
        sql: "DROP SYNONYM DICTIONARY conf_syn_drop",
        proof: Proof::Catalog(synonym_dictionaries),
    },
    Case {
        name: "CREATE TYPE",
        setup: &[],
        sql: "CREATE TYPE conf_zip AS (storage = TEXT)",
        proof: Proof::Catalog(user_types),
    },
    Case {
        name: "DROP TYPE",
        setup: &["CREATE TYPE conf_zip_drop AS (storage = TEXT)"],
        sql: "DROP TYPE conf_zip_drop",
        proof: Proof::Catalog(user_types),
    },
    Case {
        name: "CREATE COLLATION",
        setup: &[],
        sql: "CREATE COLLATION conf_col (locale = 'de_DE')",
        proof: Proof::Catalog(collations),
    },
    Case {
        name: "DROP COLLATION",
        setup: &["CREATE COLLATION conf_col_drop (locale = 'de_DE')"],
        sql: "DROP COLLATION conf_col_drop",
        proof: Proof::Catalog(collations),
    },
    Case {
        name: "CREATE BULKHEAD",
        setup: &[],
        sql: "CREATE BULKHEAD conf_bh (max_concurrent = 10)",
        proof: Proof::Catalog(resilience_policies),
    },
    Case {
        name: "DROP BULKHEAD",
        setup: &["CREATE BULKHEAD conf_bh_drop (max_concurrent = 4)"],
        sql: "DROP BULKHEAD conf_bh_drop",
        proof: Proof::Catalog(resilience_policies),
    },
    Case {
        name: "CREATE RETRY POLICY",
        setup: &[],
        sql: "CREATE RETRY POLICY conf_rp (max_attempts = 3)",
        proof: Proof::Catalog(resilience_policies),
    },
    Case {
        name: "DROP RETRY POLICY",
        setup: &["CREATE RETRY POLICY conf_rp_drop (max_attempts = 2)"],
        sql: "DROP RETRY POLICY conf_rp_drop",
        proof: Proof::Catalog(resilience_policies),
    },
    Case {
        name: "CREATE ENDPOINT",
        setup: &[],
        sql: "CREATE ENDPOINT conf_ep ON PATH '/conf/ep' METHOD GET USING 'SELECT 1' AUTH NONE",
        proof: Proof::Catalog(endpoints),
    },
    Case {
        name: "DROP ENDPOINT",
        setup: &[
            "CREATE ENDPOINT conf_ep_drop ON PATH '/conf/ep_drop' METHOD GET USING 'SELECT 1' AUTH NONE",
        ],
        sql: "DROP ENDPOINT conf_ep_drop",
        proof: Proof::Catalog(endpoints),
    },
    Case {
        name: "CREATE EXTERNAL SINK",
        setup: &[],
        sql: "CREATE EXTERNAL SINK conf_sink TYPE FILE URI '/tmp/conf_sink' FORMAT CSV",
        proof: Proof::Catalog(external_sinks),
    },
    Case {
        name: "DROP EXTERNAL SINK",
        setup: &["CREATE EXTERNAL SINK conf_sink_drop TYPE FILE URI '/tmp/conf_drop' FORMAT CSV"],
        sql: "DROP EXTERNAL SINK conf_sink_drop",
        proof: Proof::Catalog(external_sinks),
    },
    // A provider's configuration is sealed by whichever node writes the row,
    // so each member holds it under its own key. What has to agree is which
    // provider the sink asks
    Case {
        name: "CREATE EXTERNAL SINK with a credential provider",
        setup: &[],
        sql: "CREATE EXTERNAL SINK conf_sink_prov TYPE S3 URI 's3://b/out' FORMAT JSONLINES \
              CREDENTIAL_PROVIDER (type = 'aws_secrets_manager', region = 'us-west-2', \
              secret_id = 'prod/out')",
        proof: Proof::Catalog(external_sinks),
    },
    Case {
        name: "ALTER EXTERNAL SINK SET CREDENTIAL_PROVIDER",
        setup: &[
            "CREATE EXTERNAL SINK conf_sink_alter TYPE S3 URI 's3://b/alter' FORMAT JSONLINES",
        ],
        sql: "ALTER EXTERNAL SINK conf_sink_alter SET CREDENTIAL_PROVIDER \
              (type = 'vault', url = 'https://vault:8200', path = 'secret/data/out', \
              token = 's.token')",
        proof: Proof::Catalog(external_sinks),
    },
    // The slot a provider's configuration lives in is the slot a static list
    // lives in, so setting a list has to clear the provider
    Case {
        name: "ALTER EXTERNAL SINK SET CREDENTIALS clears the provider",
        setup: &[
            "CREATE EXTERNAL SINK conf_sink_clear TYPE S3 URI 's3://b/clear' FORMAT JSONLINES \
             CREDENTIAL_PROVIDER (type = 'aws_secrets_manager', region = 'us-east-1', \
             secret_id = 'prod/clear')",
        ],
        sql: "ALTER EXTERNAL SINK conf_sink_clear SET CREDENTIALS (access_key = 'AKIA')",
        proof: Proof::Catalog(external_sinks),
    },
    Case {
        name: "CREATE EXTERNAL SOURCE",
        setup: &[],
        sql: "CREATE EXTERNAL SOURCE conf_src TYPE FILE URI '/tmp/conf_src' FORMAT CSV \
              COLUMNS (id BIGINT, name VARCHAR)",
        proof: Proof::Catalog(external_sources),
    },
    Case {
        name: "DROP EXTERNAL SOURCE",
        setup: &[
            "CREATE EXTERNAL SOURCE conf_src_drop TYPE FILE URI '/tmp/conf_sd' FORMAT CSV \
             COLUMNS (id BIGINT)",
        ],
        sql: "DROP EXTERNAL SOURCE conf_src_drop",
        proof: Proof::Catalog(external_sources),
    },
    // A held source yields nothing to a streaming job, a COPY, or a
    // subscription, so the hold has to be held by every member. A pause that
    // stopped at the leader would leave the other members ingesting
    Case {
        name: "ALTER EXTERNAL SOURCE PAUSE",
        setup: &[
            "CREATE EXTERNAL SOURCE conf_src_pause TYPE FILE URI '/tmp/conf_sp' FORMAT CSV \
             COLUMNS (id BIGINT)",
        ],
        sql: "ALTER EXTERNAL SOURCE conf_src_pause PAUSE",
        proof: Proof::Catalog(external_sources),
    },
    Case {
        name: "ALTER EXTERNAL SOURCE RESUME",
        setup: &[
            "CREATE EXTERNAL SOURCE conf_src_resume TYPE FILE URI '/tmp/conf_sr' FORMAT CSV \
             COLUMNS (id BIGINT)",
            "ALTER EXTERNAL SOURCE conf_src_resume PAUSE",
        ],
        sql: "ALTER EXTERNAL SOURCE conf_src_resume RESUME",
        proof: Proof::Catalog(external_sources),
    },
    Case {
        name: "ALTER EXTERNAL SOURCE SET COLUMNS",
        setup: &[
            "CREATE EXTERNAL SOURCE conf_src_cols TYPE FILE URI '/tmp/conf_sc' FORMAT CSV \
             COLUMNS (id BIGINT)",
        ],
        sql: "ALTER EXTERNAL SOURCE conf_src_cols SET COLUMNS (id BIGINT, name VARCHAR)",
        proof: Proof::Catalog(external_sources),
    },
    // The provider's own configuration is sealed by whichever node writes the
    // row, so each member holds it under its own key. What has to agree is
    // which provider the source asks
    Case {
        name: "ALTER EXTERNAL SOURCE SET CREDENTIAL_PROVIDER",
        setup: &[
            "CREATE EXTERNAL SOURCE conf_src_prov TYPE S3 URI 's3://b/k' FORMAT CSV \
             COLUMNS (id BIGINT)",
        ],
        sql: "ALTER EXTERNAL SOURCE conf_src_prov SET CREDENTIAL_PROVIDER \
              (type = 'aws_secrets_manager', region = 'us-west-2', secret_id = 'prod/etl')",
        proof: Proof::Catalog(external_sources),
    },
    Case {
        name: "REFRESH MATERIALIZED VIEW",
        setup: &[
            "CREATE TABLE conf_mv_src (id BIGINT PRIMARY KEY, v BIGINT)",
            "INSERT INTO conf_mv_src (id, v) VALUES (1, 10), (2, 20)",
            "CREATE MATERIALIZED VIEW conf_mv AS SELECT id, v FROM conf_mv_src",
            "INSERT INTO conf_mv_src (id, v) VALUES (3, 30)",
        ],
        sql: "REFRESH MATERIALIZED VIEW conf_mv",
        proof: Proof::Rows("conf_mv"),
    },
    Case {
        name: "RUN RETENTION JOB",
        setup: &[
            "CREATE TABLE conf_ret (id BIGINT PRIMARY KEY, created_at TIMESTAMP)",
            "INSERT INTO conf_ret (id, created_at) VALUES (1, '2000-01-01 00:00:00')",
            "INSERT INTO conf_ret (id, created_at) VALUES (2, '2000-01-02 00:00:00')",
            "ALTER TABLE conf_ret SET TTL 30 DAYS ON created_at",
        ],
        sql: "RUN RETENTION JOB ON conf_ret",
        proof: Proof::Rows("conf_ret"),
    },
    Case {
        name: "RESTORE FROM (undo soft delete)",
        setup: &[
            "CREATE TABLE conf_soft (id BIGINT PRIMARY KEY, v BIGINT, is_deleted BOOLEAN, deleted_at TIMESTAMP)",
            "ALTER TABLE conf_soft ENABLE soft_delete",
            "INSERT INTO conf_soft (id, v, is_deleted, deleted_at) VALUES (1, 10, false, '2000-01-01 00:00:00'), (2, 20, false, '2000-01-01 00:00:00')",
            "DELETE FROM conf_soft WHERE id = 2",
        ],
        sql: "RESTORE FROM conf_soft WHERE id = 2",
        proof: Proof::Rows("conf_soft"),
    },
    Case {
        name: "UNDROP TABLE",
        setup: &[
            "CREATE TABLE conf_undrop (id BIGINT PRIMARY KEY)",
            "ALTER TABLE conf_undrop SET (recycle_window = '7 days')",
            "DROP TABLE conf_undrop",
        ],
        sql: "UNDROP TABLE conf_undrop",
        proof: Proof::Catalog(tables),
    },
    Case {
        name: "ARCHIVE TABLE",
        setup: &[
            "CREATE TABLE conf_arch (id BIGINT PRIMARY KEY, v BIGINT)",
            "INSERT INTO conf_arch (id, v) VALUES (1, 10), (2, 20), (3, 30)",
        ],
        sql: "ARCHIVE TABLE conf_arch WHERE id < 3 TO '/tmp/conf_archive'",
        proof: Proof::Rows("conf_arch"),
    },
    Case {
        name: "RUN PIPELINE",
        setup: &[
            "CREATE TABLE conf_pipe_src (id BIGINT PRIMARY KEY, v BIGINT)",
            "CREATE TABLE conf_pipe_tgt (id BIGINT PRIMARY KEY, v BIGINT)",
            "INSERT INTO conf_pipe_src (id, v) VALUES (1, 10), (2, 20)",
            "CREATE PIPELINE conf_pipe AS (STAGE load (SOURCE conf_pipe_src, TARGET conf_pipe_tgt))",
        ],
        sql: "RUN PIPELINE conf_pipe",
        proof: Proof::Rows("conf_pipe_tgt"),
    },
    Case {
        name: "RESTORE TABLE",
        setup: &[
            "CREATE TABLE conf_rest (id BIGINT PRIMARY KEY, v BIGINT)",
            "INSERT INTO conf_rest (id, v) VALUES (1, 10), (2, 20)",
            "ARCHIVE TABLE conf_rest WHERE id <= 2 TO '/tmp/conf_restore_src'",
        ],
        sql: "RESTORE TABLE conf_rest FROM '/tmp/conf_restore_src'",
        proof: Proof::Rows("conf_rest"),
    },
    Case {
        name: "RESTORE TABLE TO VERSION",
        setup: &[
            "CREATE TABLE conf_lake (id BIGINT NOT NULL) USING ZYRONLAKE",
            "INSERT INTO conf_lake (id) VALUES (1)",
            "INSERT INTO conf_lake (id) VALUES (2)",
        ],
        sql: "RESTORE TABLE conf_lake TO VERSION 1",
        proof: Proof::Rows("conf_lake"),
    },
    Case {
        name: "FORGET USER",
        setup: &[
            "CREATE TABLE conf_erase (id BIGINT PRIMARY KEY, user_id VARCHAR)",
            "INSERT INTO conf_erase (id, user_id) VALUES (1, 'subject-a'), (2, 'subject-b')",
        ],
        sql: "FORGET USER 'subject-a'",
        proof: Proof::Rows("conf_erase"),
    },
    Case {
        // A row written without naming the soft delete marker carries no
        // marker, and was never deleted. It stayed invisible, and deleting
        // beside it built a batch whose columns were not the same length
        name: "DELETE on a soft delete table after a partial column insert",
        setup: &[
            "CREATE TABLE repro_sd (id BIGINT PRIMARY KEY, v BIGINT, is_deleted BOOLEAN, deleted_at TIMESTAMP)",
            "ALTER TABLE repro_sd ENABLE soft_delete",
            "INSERT INTO repro_sd (id, v) VALUES (1, 10), (2, 20)",
        ],
        sql: "DELETE FROM repro_sd WHERE id = 2",
        proof: Proof::Rows("repro_sd"),
    },
    Case {
        name: "ANALYZE",
        setup: &[
            "CREATE TABLE conf_stats (id BIGINT PRIMARY KEY, v BIGINT)",
            "INSERT INTO conf_stats (id, v) VALUES (1, 10), (2, 20), (3, 30)",
        ],
        sql: "ANALYZE conf_stats",
        proof: Proof::Catalog(analyzed_rows),
    },
    Case {
        name: "CREATE FUNCTION",
        setup: &[],
        sql: "CREATE FUNCTION conf_fn(x BIGINT) RETURNS BIGINT AS 'x * 2' LANGUAGE SQL",
        proof: Proof::Catalog(functions),
    },
    Case {
        name: "DROP FUNCTION",
        setup: &["CREATE FUNCTION conf_fn_drop(x BIGINT) RETURNS BIGINT AS 'x + 1' LANGUAGE SQL"],
        sql: "DROP FUNCTION conf_fn_drop",
        proof: Proof::Catalog(functions),
    },
    Case {
        name: "CREATE AGGREGATE",
        setup: &[
            "CREATE FUNCTION conf_agg_add(acc INT, val INT) RETURNS INT AS 'acc + val'              LANGUAGE SQL",
        ],
        sql: "CREATE AGGREGATE conf_agg(val INT) (SFUNC = conf_agg_add, STYPE = INT,               INITCOND = '0')",
        proof: Proof::Catalog(aggregates),
    },
    Case {
        name: "DROP AGGREGATE",
        setup: &[
            "CREATE FUNCTION conf_agg_add_d(acc INT, val INT) RETURNS INT AS 'acc + val'              LANGUAGE SQL",
            "CREATE AGGREGATE conf_agg_drop(val INT) (SFUNC = conf_agg_add_d, STYPE = INT,              INITCOND = '0')",
        ],
        sql: "DROP AGGREGATE conf_agg_drop",
        proof: Proof::Catalog(aggregates),
    },
    Case {
        name: "CREATE TABLE",
        setup: &[],
        sql: "CREATE TABLE conf_tbl (id BIGINT PRIMARY KEY, v BIGINT)",
        proof: Proof::Catalog(tables),
    },
    Case {
        name: "DROP TABLE",
        setup: &["CREATE TABLE conf_tbl_drop (id BIGINT PRIMARY KEY)"],
        sql: "DROP TABLE conf_tbl_drop",
        proof: Proof::Catalog(tables),
    },
    // Roles, users and grants.
    //
    // These reach a member the same way every other statement does and are
    // read back out of the security manager rather than the catalog. A member
    // that agreed the statement and never ran it holds no role, so a login
    // that the leader refuses would be let through there
    Case {
        name: "CREATE ROLE",
        setup: &[],
        sql: "CREATE ROLE conf_role",
        proof: Proof::Catalog(principals),
    },
    Case {
        name: "ALTER ROLE",
        setup: &["CREATE ROLE conf_role_rn"],
        sql: "ALTER ROLE conf_role_rn RENAME TO conf_role_rn2",
        proof: Proof::Catalog(principals),
    },
    Case {
        name: "DROP ROLE",
        setup: &["CREATE ROLE conf_role_drop"],
        sql: "DROP ROLE conf_role_drop",
        proof: Proof::Catalog(principals),
    },
    Case {
        name: "CREATE USER",
        setup: &[],
        sql: "CREATE USER conf_user WITH PASSWORD 'conf_pw_create'",
        proof: Proof::Catalog(principals),
    },
    Case {
        name: "ALTER USER",
        setup: &["CREATE USER conf_user_rn WITH PASSWORD 'conf_pw_rename'"],
        sql: "ALTER USER conf_user_rn RENAME TO conf_user_rn2",
        proof: Proof::Catalog(principals),
    },
    Case {
        name: "DROP USER",
        setup: &["CREATE USER conf_user_drop WITH PASSWORD 'conf_pw_drop'"],
        sql: "DROP USER conf_user_drop",
        proof: Proof::Catalog(principals),
    },
    Case {
        name: "GRANT",
        setup: &[
            "CREATE ROLE conf_role_grant",
            "CREATE TABLE conf_grant_t (id BIGINT PRIMARY KEY)",
        ],
        sql: "GRANT SELECT ON conf_grant_t TO conf_role_grant",
        proof: Proof::Catalog(grants),
    },
    Case {
        name: "REVOKE",
        setup: &[
            "CREATE ROLE conf_role_revoke",
            "CREATE TABLE conf_revoke_t (id BIGINT PRIMARY KEY)",
            "GRANT SELECT ON conf_revoke_t TO conf_role_revoke",
        ],
        sql: "REVOKE SELECT ON conf_revoke_t FROM conf_role_revoke",
        proof: Proof::Catalog(grants),
    },
    // Indexes. A member holding the catalog row and no built index answers
    // the same rows more slowly, so what these prove is the row; the build
    // itself is the online DDL suite's subject
    Case {
        name: "CREATE INDEX",
        setup: &["CREATE TABLE conf_ix_t (id BIGINT PRIMARY KEY, v BIGINT)"],
        sql: "CREATE INDEX conf_ix ON conf_ix_t (v)",
        proof: Proof::Catalog(indexes),
    },
    Case {
        name: "DROP INDEX",
        setup: &[
            "CREATE TABLE conf_ix_d_t (id BIGINT PRIMARY KEY, v BIGINT)",
            "CREATE INDEX conf_ix_d ON conf_ix_d_t (v)",
        ],
        sql: "DROP INDEX conf_ix_d",
        proof: Proof::Catalog(indexes),
    },
    Case {
        name: "CREATE FULLTEXT INDEX",
        setup: &["CREATE TABLE conf_ft_t (id BIGINT PRIMARY KEY, body TEXT)"],
        sql: "CREATE FULLTEXT INDEX conf_ft ON conf_ft_t (body)",
        proof: Proof::Catalog(indexes),
    },
    Case {
        name: "CREATE VECTOR INDEX",
        setup: &["CREATE TABLE conf_vec_t (id BIGINT PRIMARY KEY, embedding VECTOR(4))"],
        sql: "CREATE VECTOR INDEX conf_vec ON conf_vec_t (embedding) WITH (metric = 'cosine')",
        proof: Proof::Catalog(indexes),
    },
    Case {
        name: "CREATE SPATIAL INDEX",
        setup: &["CREATE TABLE conf_geo_t (id BIGINT PRIMARY KEY, p GEOMETRY)"],
        sql: "CREATE SPATIAL INDEX conf_geo ON conf_geo_t (p)",
        proof: Proof::Catalog(indexes),
    },
    // Views
    Case {
        name: "CREATE VIEW",
        setup: &["CREATE TABLE conf_v_src (id BIGINT PRIMARY KEY, v BIGINT)"],
        sql: "CREATE VIEW conf_v AS SELECT id, v FROM conf_v_src",
        proof: Proof::Catalog(views),
    },
    Case {
        name: "DROP VIEW",
        setup: &[
            "CREATE TABLE conf_v_d_src (id BIGINT PRIMARY KEY, v BIGINT)",
            "CREATE VIEW conf_v_d AS SELECT id FROM conf_v_d_src",
        ],
        sql: "DROP VIEW conf_v_d",
        proof: Proof::Catalog(views),
    },
    Case {
        name: "CREATE MATERIALIZED VIEW",
        setup: &["CREATE TABLE conf_mvc_src (id BIGINT PRIMARY KEY, v BIGINT)"],
        sql: "CREATE MATERIALIZED VIEW conf_mvc AS SELECT id, v FROM conf_mvc_src",
        proof: Proof::Catalog(mviews),
    },
    Case {
        name: "DROP MATERIALIZED VIEW",
        setup: &[
            "CREATE TABLE conf_mv_d_src (id BIGINT PRIMARY KEY, v BIGINT)",
            "CREATE MATERIALIZED VIEW conf_mv_d AS SELECT id FROM conf_mv_d_src",
        ],
        sql: "DROP MATERIALIZED VIEW conf_mv_d",
        proof: Proof::Catalog(mviews),
    },
    // Sequences
    Case {
        name: "CREATE SEQUENCE",
        setup: &[],
        sql: "CREATE SEQUENCE conf_seq",
        proof: Proof::Catalog(sequences),
    },
    Case {
        name: "ALTER SEQUENCE",
        setup: &["CREATE SEQUENCE conf_seq_alt"],
        sql: "ALTER SEQUENCE conf_seq_alt MAXVALUE 500",
        proof: Proof::Catalog(sequences),
    },
    Case {
        name: "DROP SEQUENCE",
        setup: &["CREATE SEQUENCE conf_seq_drop"],
        sql: "DROP SEQUENCE conf_seq_drop",
        proof: Proof::Catalog(sequences),
    },
    // Schemas
    Case {
        name: "CREATE SCHEMA",
        setup: &[],
        sql: "CREATE SCHEMA conf_schema",
        proof: Proof::Catalog(schemas),
    },
    Case {
        name: "DROP SCHEMA",
        setup: &["CREATE SCHEMA conf_schema_drop"],
        sql: "DROP SCHEMA conf_schema_drop",
        proof: Proof::Catalog(schemas),
    },
    // Procedures
    Case {
        name: "CREATE PROCEDURE",
        setup: &["CREATE TABLE conf_proc_t (id BIGINT PRIMARY KEY)"],
        sql: "CREATE PROCEDURE conf_proc(v BIGINT) AS \
              'INSERT INTO zyron_test.conf_proc_t (id) VALUES ($1)' LANGUAGE SQL",
        proof: Proof::Catalog(procedures),
    },
    Case {
        name: "DROP PROCEDURE",
        setup: &[
            "CREATE TABLE conf_proc_d_t (id BIGINT PRIMARY KEY)",
            "CREATE PROCEDURE conf_proc_d(v BIGINT) AS \
             'INSERT INTO zyron_test.conf_proc_d_t (id) VALUES ($1)' LANGUAGE SQL",
        ],
        sql: "DROP PROCEDURE conf_proc_d",
        proof: Proof::Catalog(procedures),
    },
    // Comments
    Case {
        name: "COMMENT ON",
        setup: &["CREATE TABLE conf_cmt_t (id BIGINT PRIMARY KEY, v BIGINT)"],
        sql: "COMMENT ON TABLE conf_cmt_t IS 'what this table is for'",
        proof: Proof::Catalog(comments),
    },
    // Publications
    Case {
        name: "CREATE PUBLICATION",
        setup: &["CREATE TABLE conf_pub_t (id BIGINT PRIMARY KEY)"],
        sql: "CREATE PUBLICATION conf_pub FOR TABLE conf_pub_t",
        proof: Proof::Catalog(publications),
    },
    Case {
        name: "DROP PUBLICATION",
        setup: &[
            "CREATE TABLE conf_pub_d_t (id BIGINT PRIMARY KEY)",
            "CREATE PUBLICATION conf_pub_d FOR TABLE conf_pub_d_t",
        ],
        sql: "DROP PUBLICATION conf_pub_d",
        proof: Proof::Catalog(publications),
    },
    // Table shape. `tables` counts columns and `table_columns` names them, so
    // an ALTER that renamed one is not read as an ALTER that did nothing
    Case {
        name: "ALTER TABLE ADD COLUMN",
        setup: &["CREATE TABLE conf_at_add (id BIGINT PRIMARY KEY)"],
        sql: "ALTER TABLE conf_at_add ADD COLUMN extra BIGINT",
        proof: Proof::Catalog(table_columns),
    },
    Case {
        name: "ALTER TABLE DROP COLUMN",
        setup: &["CREATE TABLE conf_at_drop (id BIGINT PRIMARY KEY, gone BIGINT)"],
        sql: "ALTER TABLE conf_at_drop DROP COLUMN gone",
        proof: Proof::Catalog(table_columns),
    },
    Case {
        name: "ALTER TABLE RENAME COLUMN",
        setup: &["CREATE TABLE conf_at_ren (id BIGINT PRIMARY KEY, before_name BIGINT)"],
        sql: "ALTER TABLE conf_at_ren RENAME COLUMN before_name TO after_name",
        proof: Proof::Catalog(table_columns),
    },
    Case {
        name: "ALTER INDEX",
        setup: &[
            "CREATE TABLE conf_ix_r_t (id BIGINT PRIMARY KEY, v BIGINT)",
            "CREATE INDEX conf_ix_before ON conf_ix_r_t (v)",
        ],
        sql: "ALTER INDEX conf_ix_before RENAME TO conf_ix_after",
        proof: Proof::Catalog(indexes),
    },
    Case {
        name: "ALTER VIEW",
        setup: &[
            "CREATE TABLE conf_v_r_src (id BIGINT PRIMARY KEY)",
            "CREATE VIEW conf_v_before AS SELECT id FROM conf_v_r_src",
        ],
        sql: "ALTER VIEW conf_v_before RENAME TO conf_v_after",
        proof: Proof::Catalog(views),
    },
    // Triggers
    // A trigger and an event handler both run a procedure rather than a
    // function, so the thing they name is created with CREATE PROCEDURE
    Case {
        name: "CREATE TRIGGER",
        setup: &[
            "CREATE TABLE conf_trg_t (id BIGINT PRIMARY KEY, v BIGINT)",
            "CREATE PROCEDURE conf_trg_fn() AS \
             'INSERT INTO zyron_test.conf_trg_t (id) VALUES (99)' LANGUAGE SQL",
        ],
        sql: "CREATE TRIGGER conf_trg BEFORE INSERT ON conf_trg_t \
              FOR EACH ROW EXECUTE FUNCTION conf_trg_fn",
        proof: Proof::Catalog(triggers),
    },
    Case {
        name: "DROP TRIGGER",
        setup: &[
            "CREATE TABLE conf_trg_d_t (id BIGINT PRIMARY KEY, v BIGINT)",
            "CREATE PROCEDURE conf_trg_d_fn() AS \
             'INSERT INTO zyron_test.conf_trg_d_t (id) VALUES (99)' LANGUAGE SQL",
            "CREATE TRIGGER conf_trg_d BEFORE INSERT ON conf_trg_d_t \
             FOR EACH ROW EXECUTE FUNCTION conf_trg_d_fn",
        ],
        sql: "DROP TRIGGER conf_trg_d ON conf_trg_d_t",
        proof: Proof::Catalog(triggers),
    },
    // Schedules. PAUSE and RESUME change one field and nothing else, which is
    // why the probe renders it
    Case {
        name: "CREATE SCHEDULE",
        setup: &["CREATE TABLE conf_sch_t (id BIGINT PRIMARY KEY)"],
        sql: "CREATE SCHEDULE conf_sch EVERY 5 MINUTES \
              DO INSERT INTO conf_sch_t (id) VALUES (1)",
        proof: Proof::Catalog(schedules),
    },
    Case {
        name: "PAUSE SCHEDULE",
        setup: &[
            "CREATE TABLE conf_sch_p_t (id BIGINT PRIMARY KEY)",
            "CREATE SCHEDULE conf_sch_p EVERY 5 MINUTES \
             DO INSERT INTO conf_sch_p_t (id) VALUES (1)",
        ],
        sql: "PAUSE SCHEDULE conf_sch_p",
        proof: Proof::Catalog(schedules),
    },
    Case {
        name: "RESUME SCHEDULE",
        setup: &[
            "CREATE TABLE conf_sch_r_t (id BIGINT PRIMARY KEY)",
            "CREATE SCHEDULE conf_sch_r EVERY 5 MINUTES \
             DO INSERT INTO conf_sch_r_t (id) VALUES (1)",
            "PAUSE SCHEDULE conf_sch_r",
        ],
        sql: "RESUME SCHEDULE conf_sch_r",
        proof: Proof::Catalog(schedules),
    },
    Case {
        name: "DROP SCHEDULE",
        setup: &[
            "CREATE TABLE conf_sch_d_t (id BIGINT PRIMARY KEY)",
            "CREATE SCHEDULE conf_sch_d EVERY 5 MINUTES \
             DO INSERT INTO conf_sch_d_t (id) VALUES (1)",
        ],
        sql: "DROP SCHEDULE conf_sch_d",
        proof: Proof::Catalog(schedules),
    },
    // Pipelines
    Case {
        name: "CREATE PIPELINE",
        setup: &[
            "CREATE TABLE conf_pipec_src (id BIGINT PRIMARY KEY, v BIGINT)",
            "CREATE TABLE conf_pipec_tgt (id BIGINT PRIMARY KEY, v BIGINT)",
        ],
        sql: "CREATE PIPELINE conf_pipec AS \
              (STAGE load (SOURCE conf_pipec_src, TARGET conf_pipec_tgt))",
        proof: Proof::Catalog(pipelines),
    },
    Case {
        name: "DROP PIPELINE",
        setup: &[
            "CREATE TABLE conf_piped_src (id BIGINT PRIMARY KEY, v BIGINT)",
            "CREATE TABLE conf_piped_tgt (id BIGINT PRIMARY KEY, v BIGINT)",
            "CREATE PIPELINE conf_piped AS \
             (STAGE load (SOURCE conf_piped_src, TARGET conf_piped_tgt))",
        ],
        sql: "DROP PIPELINE conf_piped",
        proof: Proof::Catalog(pipelines),
    },
    // Event handlers
    Case {
        name: "CREATE EVENT HANDLER",
        setup: &[
            "CREATE TABLE conf_eh_t (id BIGINT PRIMARY KEY)",
            "CREATE PROCEDURE conf_eh_fn() AS \
             'INSERT INTO zyron_test.conf_eh_t (id) VALUES (1)' LANGUAGE SQL",
        ],
        sql: "CREATE EVENT HANDLER conf_eh WHEN PipelineCompleted EXECUTE FUNCTION conf_eh_fn",
        proof: Proof::Catalog(event_handlers),
    },
    Case {
        name: "DROP EVENT HANDLER",
        setup: &[
            "CREATE TABLE conf_eh_d_t (id BIGINT PRIMARY KEY)",
            "CREATE PROCEDURE conf_eh_d_fn() AS \
             'INSERT INTO zyron_test.conf_eh_d_t (id) VALUES (1)' LANGUAGE SQL",
            "CREATE EVENT HANDLER conf_eh_d WHEN PipelineCompleted \
             EXECUTE FUNCTION conf_eh_d_fn",
        ],
        sql: "DROP EVENT HANDLER conf_eh_d",
        proof: Proof::Catalog(event_handlers),
    },
    // Version tags
    Case {
        name: "CREATE VERSION",
        setup: &["CREATE TABLE conf_ver_t (id BIGINT PRIMARY KEY)"],
        sql: "CREATE VERSION conf_ver ON conf_ver_t",
        proof: Proof::Catalog(version_tags),
    },
    Case {
        name: "DROP VERSION",
        setup: &[
            "CREATE TABLE conf_ver_d_t (id BIGINT PRIMARY KEY)",
            "CREATE VERSION conf_ver_d ON conf_ver_d_t",
        ],
        sql: "DROP VERSION conf_ver_d",
        proof: Proof::Catalog(version_tags),
    },
    // Publications beyond create and drop
    Case {
        name: "ALTER PUBLICATION",
        setup: &[
            "CREATE TABLE conf_pub_a_t (id BIGINT PRIMARY KEY)",
            "CREATE TABLE conf_pub_a_t2 (id BIGINT PRIMARY KEY)",
            "CREATE PUBLICATION conf_pub_a FOR TABLE conf_pub_a_t",
        ],
        sql: "ALTER PUBLICATION conf_pub_a ADD TABLE conf_pub_a_t2",
        proof: Proof::Catalog(publication_tables),
    },
    Case {
        name: "TAG PUBLICATION",
        setup: &[
            "CREATE TABLE conf_pub_tag_t (id BIGINT PRIMARY KEY)",
            "CREATE PUBLICATION conf_pub_tag FOR TABLE conf_pub_tag_t",
        ],
        sql: "TAG PUBLICATION conf_pub_tag WITH '#conf'",
        proof: Proof::Catalog(publication_tags),
    },
    Case {
        name: "UNTAG PUBLICATION",
        setup: &[
            "CREATE TABLE conf_pub_untag_t (id BIGINT PRIMARY KEY)",
            "CREATE PUBLICATION conf_pub_untag FOR TABLE conf_pub_untag_t",
            "TAG PUBLICATION conf_pub_untag WITH '#conf_gone'",
        ],
        sql: "UNTAG PUBLICATION conf_pub_untag '#conf_gone'",
        proof: Proof::Catalog(publication_tags),
    },
    // Security maps
    Case {
        name: "ALTER SECURITY MAP",
        setup: &["CREATE ROLE conf_map_role"],
        sql: "ALTER SECURITY MAP JWT ISSUER 'https://conf' SUBJECT 'conf_subject' \
              TO ROLE 'conf_map_role'",
        proof: Proof::Catalog(security_maps),
    },
    Case {
        name: "DROP SECURITY MAP",
        setup: &[
            "CREATE ROLE conf_map_d_role",
            "ALTER SECURITY MAP JWT ISSUER 'https://conf_d' SUBJECT 'conf_subject_d' \
             TO ROLE 'conf_map_d_role'",
        ],
        sql: "DROP SECURITY MAP JWT ISSUER 'https://conf_d' SUBJECT 'conf_subject_d'",
        proof: Proof::Catalog(security_maps),
    },
    // Table options and shape beyond columns
    Case {
        name: "TRUNCATE",
        setup: &[
            "CREATE TABLE conf_trunc (id BIGINT PRIMARY KEY, v BIGINT)",
            "INSERT INTO conf_trunc (id, v) VALUES (1, 10), (2, 20)",
        ],
        sql: "TRUNCATE TABLE conf_trunc",
        proof: Proof::Rows("conf_trunc"),
    },
    Case {
        name: "ALTER TABLE SET options",
        setup: &["CREATE TABLE conf_opt (id BIGINT PRIMARY KEY)"],
        sql: "ALTER TABLE conf_opt SET (time_travel_retention = '7 days')",
        proof: Proof::Catalog(table_options),
    },
    Case {
        name: "ALTER TABLE SET (immutable, verified)",
        setup: &["CREATE TABLE conf_verified (id BIGINT PRIMARY KEY, amount BIGINT)"],
        sql: "ALTER TABLE conf_verified SET (immutable = true, verified = true)",
        proof: Proof::Catalog(table_verification),
    },
    Case {
        name: "ALTER TABLE SET TTL",
        setup: &["CREATE TABLE conf_ttl (id BIGINT PRIMARY KEY, expires_at TIMESTAMP)"],
        sql: "ALTER TABLE conf_ttl SET TTL 15 MINUTES ON expires_at",
        proof: Proof::Catalog(table_ttl),
    },
    Case {
        name: "ALTER TABLE CLUSTER BY",
        setup: &["CREATE TABLE conf_clus (id BIGINT PRIMARY KEY, v BIGINT)"],
        sql: "ALTER TABLE conf_clus CLUSTER BY (v) AUTO",
        proof: Proof::Catalog(table_clustering),
    },
    Case {
        name: "ALTER TABLE ADD EXPECTATION",
        setup: &["CREATE TABLE conf_exp (id BIGINT PRIMARY KEY, amount BIGINT)"],
        sql: "ALTER TABLE conf_exp ADD EXPECTATION conf_exp_pos \
              EXPECT amount > 0 ON VIOLATION WARN",
        proof: Proof::Catalog(expectations),
    },
    Case {
        name: "ALTER TABLE DROP EXPECTATION",
        setup: &[
            "CREATE TABLE conf_exp_d (id BIGINT PRIMARY KEY, amount BIGINT)",
            "ALTER TABLE conf_exp_d ADD EXPECTATION conf_exp_gone \
             EXPECT amount > 0 ON VIOLATION WARN",
        ],
        sql: "ALTER TABLE conf_exp_d DROP EXPECTATION conf_exp_gone",
        proof: Proof::Catalog(expectations),
    },
    // Branches. Not a catalog object, so the probe reads the branch manager
    Case {
        name: "CREATE BRANCH",
        setup: &[],
        sql: "CREATE BRANCH conf_branch",
        proof: Proof::Catalog(branches),
    },
    Case {
        name: "DROP BRANCH",
        setup: &["CREATE BRANCH conf_branch_drop"],
        sql: "DROP BRANCH conf_branch_drop",
        proof: Proof::Catalog(branches),
    },
    // Streaming jobs. The source has to be producing a change feed before a
    // job can read one, which is what the ENABLE in the setup does
    Case {
        name: "CREATE STREAMING JOB",
        setup: &[
            "CREATE TABLE conf_sj_src (id BIGINT PRIMARY KEY, v BIGINT)",
            "CREATE TABLE conf_sj_tgt (id BIGINT PRIMARY KEY, v BIGINT)",
            "ALTER TABLE conf_sj_src ENABLE change_data_feed",
        ],
        sql: "CREATE STREAMING JOB conf_sj AS SELECT id, v FROM conf_sj_src INTO conf_sj_tgt",
        proof: Proof::Catalog(streaming_jobs),
    },
    Case {
        name: "DROP STREAMING JOB",
        setup: &[
            "CREATE TABLE conf_sj_d_src (id BIGINT PRIMARY KEY, v BIGINT)",
            "CREATE TABLE conf_sj_d_tgt (id BIGINT PRIMARY KEY, v BIGINT)",
            "ALTER TABLE conf_sj_d_src ENABLE change_data_feed",
            "CREATE STREAMING JOB conf_sj_d AS SELECT id, v FROM conf_sj_d_src \
             INTO conf_sj_d_tgt",
        ],
        sql: "DROP STREAMING JOB conf_sj_d",
        proof: Proof::Catalog(streaming_jobs),
    },
    // Change feed flags
    Case {
        name: "ALTER TABLE ENABLE FEATURE",
        setup: &["CREATE TABLE conf_feat (id BIGINT PRIMARY KEY, v BIGINT)"],
        sql: "ALTER TABLE conf_feat ENABLE change_data_feed",
        proof: Proof::Catalog(table_features),
    },
    Case {
        name: "ALTER TABLE DISABLE FEATURE",
        setup: &[
            "CREATE TABLE conf_feat_d (id BIGINT PRIMARY KEY, v BIGINT)",
            "ALTER TABLE conf_feat_d ENABLE change_data_feed",
        ],
        sql: "ALTER TABLE conf_feat_d DISABLE change_data_feed",
        proof: Proof::Catalog(table_features),
    },
    // Change streams. The position replicates as the count each stream has
    // consumed, so a consume on the leader reads as the same count on every
    // member, and a stage or an apply that reads a stream moves it in the
    // commit that carried its rows
    Case {
        name: "CREATE CHANGE STREAM",
        setup: &[
            "CREATE TABLE conf_cs_src (id BIGINT PRIMARY KEY, v BIGINT)",
            "ALTER TABLE conf_cs_src SET (change_data_feed = true)",
            "INSERT INTO conf_cs_src (id, v) VALUES (1, 10), (2, 20)",
        ],
        sql: "CREATE CHANGE STREAM conf_cs ON TABLE conf_cs_src",
        proof: Proof::Catalog(change_streams),
    },
    Case {
        name: "CREATE CHANGE STREAM ON TABLES",
        setup: &[
            "CREATE TABLE conf_cs_a (id BIGINT PRIMARY KEY, v BIGINT)",
            "CREATE TABLE conf_cs_b (id BIGINT PRIMARY KEY, w TEXT)",
            "ALTER TABLE conf_cs_a SET (change_data_feed = true)",
            "ALTER TABLE conf_cs_b SET (change_data_feed = true)",
        ],
        sql: "CREATE CHANGE STREAM conf_cs_ab ON TABLES (conf_cs_a, conf_cs_b) AT VERSION 0",
        proof: Proof::Catalog(change_streams),
    },
    Case {
        name: "ALTER CHANGE STREAM RESET TO POSITION",
        setup: &[
            "CREATE TABLE conf_cs_r_src (id BIGINT PRIMARY KEY, v BIGINT)",
            "ALTER TABLE conf_cs_r_src SET (change_data_feed = true)",
            "INSERT INTO conf_cs_r_src (id, v) VALUES (1, 10), (2, 20), (3, 30)",
            "CREATE CHANGE STREAM conf_cs_r ON TABLE conf_cs_r_src",
        ],
        sql: "ALTER CHANGE STREAM conf_cs_r RESET TO POSITION 1",
        proof: Proof::Catalog(change_streams),
    },
    Case {
        name: "ALTER CHANGE STREAM RESET TO VERSION",
        setup: &[
            "CREATE TABLE conf_cs_v_src (id BIGINT PRIMARY KEY, v BIGINT)",
            "ALTER TABLE conf_cs_v_src SET (change_data_feed = true)",
            "INSERT INTO conf_cs_v_src (id, v) VALUES (1, 10), (2, 20)",
            "CREATE CHANGE STREAM conf_cs_v ON TABLE conf_cs_v_src",
        ],
        sql: "ALTER CHANGE STREAM conf_cs_v RESET TO VERSION 0",
        proof: Proof::Catalog(change_streams),
    },
    Case {
        name: "ALTER CHANGE STREAM SET COLUMNS",
        setup: &[
            "CREATE TABLE conf_cs_c_src (id BIGINT PRIMARY KEY, v BIGINT, w BIGINT)",
            "ALTER TABLE conf_cs_c_src SET (change_data_feed = true)",
            "CREATE CHANGE STREAM conf_cs_c ON TABLE conf_cs_c_src",
        ],
        sql: "ALTER CHANGE STREAM conf_cs_c SET COLUMNS (id, v)",
        proof: Proof::Catalog(change_streams),
    },
    Case {
        name: "DROP CHANGE STREAM",
        setup: &[
            "CREATE TABLE conf_cs_d_src (id BIGINT PRIMARY KEY, v BIGINT)",
            "ALTER TABLE conf_cs_d_src SET (change_data_feed = true)",
            "CREATE CHANGE STREAM conf_cs_d ON TABLE conf_cs_d_src",
        ],
        sql: "DROP CHANGE STREAM conf_cs_d",
        proof: Proof::Catalog(change_streams),
    },
    Case {
        name: "GRANT ON CHANGE STREAM",
        setup: &[
            "CREATE ROLE conf_role_grant_cs",
            "CREATE TABLE conf_cs_g_src (id BIGINT PRIMARY KEY, v BIGINT)",
            "ALTER TABLE conf_cs_g_src SET (change_data_feed = true)",
            "CREATE CHANGE STREAM conf_cs_g ON TABLE conf_cs_g_src",
        ],
        sql: "GRANT SELECT ON CHANGE STREAM conf_cs_g TO conf_role_grant_cs",
        proof: Proof::Catalog(grants),
    },
    Case {
        name: "ALTER TABLE SET (cdf options)",
        setup: &[
            "CREATE TABLE conf_cdf_opt (id BIGINT PRIMARY KEY, v BIGINT, w BIGINT)",
            "ALTER TABLE conf_cdf_opt SET (change_data_feed = true)",
        ],
        sql: "ALTER TABLE conf_cdf_opt SET (cdf_retention = '2 days', cdf_before_image = false, cdf_columns = 'v')",
        proof: Proof::Catalog(feed_settings),
    },
    Case {
        name: "ALTER TABLE ALTER COLUMN TYPE ACKNOWLEDGE STREAM BREAK",
        setup: &[
            "CREATE TABLE conf_cs_brk (id BIGINT PRIMARY KEY, v BIGINT)",
            "ALTER TABLE conf_cs_brk SET (change_data_feed = true)",
            "CREATE CHANGE STREAM conf_cs_brk_s ON TABLE conf_cs_brk",
        ],
        sql: "ALTER TABLE conf_cs_brk ALTER COLUMN v TYPE INT ACKNOWLEDGE STREAM BREAK",
        proof: Proof::Catalog(change_streams),
    },
    // A consume moves the position in the commit that carried its rows, and
    // the count it consumed is what every member reads
    Case {
        name: "consume a change stream (position)",
        setup: &[
            "CREATE TABLE conf_cs_k_src (id BIGINT PRIMARY KEY, v BIGINT)",
            "CREATE TABLE conf_cs_k_tgt (id BIGINT PRIMARY KEY, v BIGINT)",
            "ALTER TABLE conf_cs_k_src SET (change_data_feed = true)",
            "CREATE CHANGE STREAM conf_cs_k ON TABLE conf_cs_k_src",
            "INSERT INTO conf_cs_k_src (id, v) VALUES (1, 10), (2, 20), (3, 30)",
        ],
        sql: "INSERT INTO conf_cs_k_tgt (id, v) SELECT id, v FROM conf_cs_k",
        proof: Proof::Catalog(change_streams),
    },
    Case {
        name: "consume a change stream (rows)",
        setup: &[
            "CREATE TABLE conf_cs_w_src (id BIGINT PRIMARY KEY, v BIGINT)",
            "CREATE TABLE conf_cs_w_tgt (id BIGINT PRIMARY KEY, v BIGINT)",
            "ALTER TABLE conf_cs_w_src SET (change_data_feed = true)",
            "CREATE CHANGE STREAM conf_cs_w ON TABLE conf_cs_w_src",
            "INSERT INTO conf_cs_w_src (id, v) VALUES (1, 10), (2, 20)",
        ],
        sql: "INSERT INTO conf_cs_w_tgt (id, v) SELECT id, v FROM conf_cs_w",
        proof: Proof::Rows("conf_cs_w_tgt"),
    },
    Case {
        name: "APPLY CHANGES",
        setup: &[
            "CREATE TABLE conf_ap_src (id BIGINT PRIMARY KEY, v BIGINT)",
            "CREATE TABLE conf_ap_tgt (id BIGINT PRIMARY KEY, v BIGINT)",
            "ALTER TABLE conf_ap_src SET (change_data_feed = true)",
            "CREATE CHANGE STREAM conf_ap ON TABLE conf_ap_src",
            "INSERT INTO conf_ap_src (id, v) VALUES (1, 10), (2, 20)",
            "UPDATE conf_ap_src SET v = 21 WHERE id = 2",
        ],
        sql: "APPLY CHANGES INTO conf_ap_tgt FROM conf_ap KEYS (id) SEQUENCE BY _commit_version",
        proof: Proof::Rows("conf_ap_tgt"),
    },
    Case {
        name: "CREATE PIPELINE (change stream stages)",
        setup: &[
            "CREATE TABLE conf_pl_src (id BIGINT PRIMARY KEY, v BIGINT)",
            "ALTER TABLE conf_pl_src SET (change_data_feed = true)",
            "CREATE CHANGE STREAM conf_pl_cs ON TABLE conf_pl_src",
        ],
        sql: "CREATE PIPELINE conf_pl_cdc ON CHANGE DATA FROM conf_pl_cs MIN ROWS 100 AS (STAGE land (CONSUME CHANGES FROM conf_pl_cs INTO conf_pl_bronze))",
        proof: Proof::Catalog(pipelines),
    },
    Case {
        name: "RUN PIPELINE (consume stage)",
        setup: &[
            "CREATE TABLE conf_pl_r_src (id BIGINT PRIMARY KEY, v BIGINT)",
            "CREATE TABLE conf_pl_r_tgt (id BIGINT PRIMARY KEY, v BIGINT)",
            "ALTER TABLE conf_pl_r_src SET (change_data_feed = true)",
            "CREATE CHANGE STREAM conf_pl_r_cs ON TABLE conf_pl_r_src",
            "CREATE PIPELINE conf_pl_r AS (STAGE load (CONSUME CHANGES FROM conf_pl_r_cs AS (INSERT INTO conf_pl_r_tgt SELECT id, v FROM changes)))",
            "INSERT INTO conf_pl_r_src (id, v) VALUES (1, 10), (2, 20)",
        ],
        sql: "RUN PIPELINE conf_pl_r",
        proof: Proof::Rows("conf_pl_r_tgt"),
    },
    Case {
        name: "CREATE CDC STREAM (implicit change stream)",
        setup: &[
            "CREATE TABLE conf_ob_src (id BIGINT PRIMARY KEY, v BIGINT)",
            "ALTER TABLE conf_ob_src SET (change_data_feed = true)",
        ],
        sql: "CREATE CDC STREAM conf_ob ON TABLE conf_ob_src TO webhook WITH (url = 'http://127.0.0.1:9/changes')",
        proof: Proof::Catalog(change_streams),
    },
    Case {
        name: "DROP CDC STREAM (implicit change stream)",
        setup: &[
            "CREATE TABLE conf_ob_d_src (id BIGINT PRIMARY KEY, v BIGINT)",
            "ALTER TABLE conf_ob_d_src SET (change_data_feed = true)",
            "CREATE CDC STREAM conf_ob_d ON TABLE conf_ob_d_src TO webhook WITH (url = 'http://127.0.0.1:9/changes')",
        ],
        sql: "DROP CDC STREAM conf_ob_d",
        proof: Proof::Catalog(change_streams),
    },
    Case {
        name: "ALTER TABLE ALTER COLUMN SET CLASSIFICATION",
        setup: &["CREATE TABLE conf_class (id BIGINT PRIMARY KEY, ssn TEXT)"],
        sql: "ALTER TABLE conf_class ALTER COLUMN ssn SET CLASSIFICATION restricted",
        proof: Proof::Catalog(column_classifications),
    },
    // Graph schemas
    Case {
        name: "CREATE GRAPH SCHEMA",
        setup: &[],
        sql: "CREATE GRAPH SCHEMA conf_graph (NODE Person (id INT))",
        proof: Proof::Catalog(tables),
    },
    Case {
        name: "DROP GRAPH SCHEMA",
        setup: &["CREATE GRAPH SCHEMA conf_graph_d (NODE Thing (id INT))"],
        sql: "DROP GRAPH SCHEMA conf_graph_d",
        proof: Proof::Catalog(tables),
    },
];

// ---------------------------------------------------------------------------
// The run
// ---------------------------------------------------------------------------

/// What went wrong with one case, in the words a fix would need.
struct Failure {
    case: String,
    detail: String,
}

/// Every statement this release claims replicates, run on a three node group.
///
/// Driven over the wire rather than through the replication handle directly,
/// because `replication_class` sits on the connection path and nowhere else.
/// A case driven through `Node::ddl` would replicate whatever it was handed
/// and prove the plumbing works while saying nothing about whether the
/// statement was classified correctly, which is the thing under test.
///
/// One group for the whole table: starting a group per case would spend the
/// run electing leaders rather than exercising statements, and the cases are
/// written so no case depends on another having run
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn every_statement_that_replicates_reaches_every_member() {
    let group = Group::start(3).await;
    let leader = group.leader(Duration::from_secs(10)).await;
    let addr = group.nodes[leader].serve_wire().await;
    let mut client = WireClient::connect(addr).await;
    let (_, errors) = client.query("SET search_path = zyron_test").await;
    assert!(
        errors.is_empty(),
        "could not set the search path: {errors:?}"
    );

    let mut failures: Vec<Failure> = Vec::new();

    let total = CASES.len();
    for case in CASES {
        let setup: Vec<&str> = case.setup.to_vec();
        run_case(
            &group,
            leader,
            &mut client,
            case.name,
            &setup,
            case.sql,
            &case.proof,
            &mut failures,
        )
        .await;
    }

    client.terminate().await;
    group.shutdown().await;

    // A pass here means nothing unless a refusal would have been caught, and
    // `a_statement_this_release_refuses_says_so` is what shows it would

    assert!(
        failures.is_empty(),
        "{} of {} statements did not reach every member:\n{}",
        failures.len(),
        total,
        failures
            .iter()
            .map(|f| format!("  {}: {}", f.case, f.detail))
            .collect::<Vec<_>>()
            .join("\n")
    );
}

/// Runs one statement on the leader and checks that it changed something and
/// that every member agrees on what.
///
/// Written once because a case whose statement is a literal and one whose
/// statement carries a compiled module have to be held to the same standard,
/// and two copies of the check is how one of them ends up weaker
#[allow(clippy::too_many_arguments)]
async fn run_case(
    group: &Group,
    leader: usize,
    client: &mut WireClient,
    name: &str,
    setup: &[&str],
    sql: &str,
    proof: &Proof,
    failures: &mut Vec<Failure>,
) {
    for statement in setup {
        let (_, errors) = client.query(statement).await;
        if !errors.is_empty() {
            failures.push(Failure {
                case: name.to_string(),
                detail: format!("setup `{statement}` failed: {errors:?}"),
            });
            return;
        }
    }
    group.settle(leader, Duration::from_secs(20)).await;

    let before = read_proof(&group.nodes[leader], proof).await;
    let (tags, errors) = client.query(sql).await;
    if !errors.is_empty() {
        failures.push(Failure {
            case: name.to_string(),
            detail: format!("refused on the leader: {errors:?}"),
        });
        return;
    }
    group.settle(leader, Duration::from_secs(20)).await;

    // A statement that reported success and changed nothing is a silent
    // no-op, which reads as a pass to any check that only compares members
    let after = read_proof(&group.nodes[leader], proof).await;
    if before == after {
        failures.push(Failure {
            case: name.to_string(),
            detail: format!("answered {tags:?} and changed nothing, still `{after}`"),
        });
        return;
    }

    for node in &group.nodes {
        let seen = read_proof(node, proof).await;
        if seen != after {
            failures.push(Failure {
                case: name.to_string(),
                detail: format!(
                    "the leader holds `{after}` and {} holds `{seen}`",
                    node.name
                ),
            });
        }
    }
}

/// Nothing is refused on a node in a group any more.
///
/// `Unsupported` was a bug list rather than a design category: a statement
/// landed there because its handler wrote rows through a path replication
/// could not see, not because of anything about the statement. Those paths are
/// wired now, so the list is empty.
///
/// This is also what keeps the suite above worth anything. Every case there
/// passes by not erroring and then converging, so a refusal has to be
/// something this harness would notice
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn no_statement_is_refused_on_a_node_in_a_group() {
    let group = Group::start(3).await;
    let leader = group.leader(Duration::from_secs(10)).await;
    let addr = group.nodes[leader].serve_wire().await;
    let mut client = WireClient::connect(addr).await;
    let (_, errors) = client.query("SET search_path = zyron_test").await;
    assert!(
        errors.is_empty(),
        "could not set the search path: {errors:?}"
    );

    // Every one of these once came back refused for being unreplicable. They
    // are run against objects that do not exist, so what comes back now is the
    // engine's own complaint about the object, never a refusal about the group
    let mut refused: Vec<String> = Vec::new();
    for sql in [
        "ARCHIVE TABLE conf_absent TO '/tmp/conf_absent'",
        "EXPORT USER 'conf_absent'",
        "FORGET USER 'conf_absent'",
        "RUN RETENTION JOB ON conf_absent",
        "UNDROP TABLE conf_absent",
        "MERGE BRANCH conf_absent INTO main",
        "RESTORE FROM conf_absent WHERE id = 1",
        "REFRESH MATERIALIZED VIEW conf_absent",
        "RESTORE TABLE conf_absent FROM '/tmp/conf_absent'",
        "RESTORE TABLE conf_absent TO VERSION 1",
        "RUN PIPELINE conf_absent",
        "ROTATE SERVICE PRINCIPAL KEY conf_sp",
    ] {
        let (_, errors) = client.query(sql).await;
        if errors
            .iter()
            .any(|e| e.contains("cannot run on a node in a consensus group"))
        {
            refused.push(format!("  {sql}: {errors:?}"));
        }
    }

    client.terminate().await;
    group.shutdown().await;

    assert!(
        refused.is_empty(),
        "statements are still refused for being unreplicable:
{}",
        refused.join(
            "
"
        )
    );
}

/// Writes a Parquet file carrying a known layout, so a source pointed at it
/// can be checked against something rather than merely being non-empty
fn write_parquet_fixture(dir: &std::path::Path) {
    use arrow::array::{Int64Array, StringArray};
    use arrow::datatypes::{DataType, Field, Schema};
    use arrow::record_batch::RecordBatch;
    use parquet::arrow::ArrowWriter;

    std::fs::create_dir_all(dir).expect("fixture directory");
    let schema = Arc::new(Schema::new(vec![
        Field::new("order_id", DataType::Int64, false),
        Field::new("customer", DataType::Utf8, true),
    ]));
    let batch = RecordBatch::try_new(
        Arc::clone(&schema),
        vec![
            Arc::new(Int64Array::from(vec![1i64, 2])),
            Arc::new(StringArray::from(vec!["ada", "grace"])),
        ],
    )
    .expect("record batch");
    let file = std::fs::File::create(dir.join("part-0.parquet")).expect("create fixture");
    let mut writer = ArrowWriter::try_new(file, schema, None).expect("parquet writer");
    writer.write(&batch).expect("write batch");
    writer.close().expect("close parquet");
}

/// A Parquet source written the ordinary way, with no column list.
///
/// The layout comes from the file, so every member has to end up holding the
/// columns the file declares. The node the statement arrives at reads the file
/// once and hands the group the resolved statement, which is what stops two
/// members reading the store separately and disagreeing.
///
/// Outside a group this is simply how the statement works, so a cluster
/// requiring a COLUMNS clause was the engine behaving differently depending on
/// its deployment
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_parquet_source_with_no_column_list_reaches_every_member() {
    let group = Group::start(3).await;
    let leader = group.leader(Duration::from_secs(10)).await;

    let dir = group.nodes[leader]._tmp.path().join("orders");
    write_parquet_fixture(&dir);
    let uri = dir
        .display()
        .to_string()
        .replace(std::path::MAIN_SEPARATOR, "/");

    let addr = group.nodes[leader].serve_wire().await;
    let mut client = WireClient::connect(addr).await;
    let (_, errors) = client.query("SET search_path = zyron_test").await;
    assert!(
        errors.is_empty(),
        "could not set the search path: {errors:?}"
    );

    let sql = format!(
        "CREATE EXTERNAL SOURCE orders_src TYPE FILE URI '/' FORMAT PARQUET          OPTIONS (root = '{uri}')"
    );
    let (tags, errors) = client.query(&sql).await;
    assert!(
        errors.is_empty(),
        "a source with no column list was refused: {errors:?}"
    );
    assert!(!tags.is_empty(), "the statement reported nothing");
    group.settle(leader, Duration::from_secs(20)).await;

    let expected = "orders_src:order_id,customer";
    for node in &group.nodes {
        let held = node
            .catalog
            .list_external_sources()
            .iter()
            .find(|s| s.name == "orders_src")
            .map(|s| {
                format!(
                    "{}:{}",
                    s.name,
                    s.columns
                        .iter()
                        .map(|(n, _)| n.as_str())
                        .collect::<Vec<_>>()
                        .join(",")
                )
            })
            .unwrap_or_else(|| "missing".to_string());
        assert_eq!(
            held, expected,
            "{} did not take the layout the file declares",
            node.name
        );
    }

    client.terminate().await;
    group.shutdown().await;
}

/// Writes a Parquet fixture whose layout differs from `write_parquet_fixture`,
/// so a refresh that reads it produces a layout the source did not already
/// hold and cannot pass by leaving the columns alone
fn write_widened_parquet_fixture(dir: &std::path::Path) {
    use arrow::array::{Int64Array, StringArray};
    use arrow::datatypes::{DataType, Field, Schema};
    use arrow::record_batch::RecordBatch;
    use parquet::arrow::ArrowWriter;

    std::fs::create_dir_all(dir).expect("fixture directory");
    let schema = Arc::new(Schema::new(vec![
        Field::new("order_id", DataType::Int64, false),
        Field::new("customer", DataType::Utf8, true),
        Field::new("region", DataType::Utf8, true),
    ]));
    let batch = RecordBatch::try_new(
        Arc::clone(&schema),
        vec![
            Arc::new(Int64Array::from(vec![3i64])),
            Arc::new(StringArray::from(vec!["alan"])),
            Arc::new(StringArray::from(vec!["emea"])),
        ],
    )
    .expect("record batch");
    let file = std::fs::File::create(dir.join("part-0.parquet")).expect("create fixture");
    let mut writer = ArrowWriter::try_new(file, schema, None).expect("parquet writer");
    writer.write(&batch).expect("write batch");
    writer.close().expect("close parquet");
}

/// REFRESH SCHEMA reads the store once and hands the group a column list.
///
/// The statement is replicated as itself, and it is an instruction to go and
/// read an external store, which is the one thing every member must not do
/// separately: two members reading an object store at two moments can see two
/// different files and settle on two different layouts. The node the
/// statement arrives at resolves it into SET COLUMNS before the group sees
/// it, and this is what shows that. The fixture is rewritten with an extra
/// column between the create and the refresh, so a refresh that quietly did
/// nothing fails rather than passing
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_schema_refresh_reaches_every_member_as_a_column_list() {
    let group = Group::start(3).await;
    let leader = group.leader(Duration::from_secs(10)).await;

    let dir = group.nodes[leader]._tmp.path().join("refreshed");
    write_parquet_fixture(&dir);
    let uri = dir
        .display()
        .to_string()
        .replace(std::path::MAIN_SEPARATOR, "/");

    let addr = group.nodes[leader].serve_wire().await;
    let mut client = WireClient::connect(addr).await;
    let (_, errors) = client.query("SET search_path = zyron_test").await;
    assert!(
        errors.is_empty(),
        "could not set the search path: {errors:?}"
    );

    let sql = format!(
        "CREATE EXTERNAL SOURCE refreshed_src TYPE FILE URI '/' FORMAT PARQUET          OPTIONS (root = '{uri}')"
    );
    let (_, errors) = client.query(&sql).await;
    assert!(errors.is_empty(), "the source was refused: {errors:?}");
    group.settle(leader, Duration::from_secs(20)).await;

    // The store changes under the source, which is what a refresh is for
    write_widened_parquet_fixture(&dir);

    let (tags, errors) = client
        .query("ALTER EXTERNAL SOURCE refreshed_src REFRESH SCHEMA")
        .await;
    assert!(errors.is_empty(), "the refresh was refused: {errors:?}");
    assert!(!tags.is_empty(), "the statement reported nothing");
    group.settle(leader, Duration::from_secs(20)).await;

    let expected = "refreshed_src:order_id,customer,region";
    for node in &group.nodes {
        let held = node
            .catalog
            .list_external_sources()
            .iter()
            .find(|s| s.name == "refreshed_src")
            .map(|s| {
                format!(
                    "{}:{}",
                    s.name,
                    s.columns
                        .iter()
                        .map(|(n, _)| n.as_str())
                        .collect::<Vec<_>>()
                        .join(",")
                )
            })
            .unwrap_or_else(|| "missing".to_string());
        assert_eq!(
            held, expected,
            "{} did not take the layout the refreshed file declares",
            node.name
        );
    }

    client.terminate().await;
    group.shutdown().await;
}

/// A source with no log position refuses RESET LSN rather than reporting a
/// reset it did not do.
///
/// An object store tracks ingest as a set of acknowledged object keys, which
/// no sequence number addresses. Accepting the statement there would report
/// success for a position that moved nowhere
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn resetting_the_position_of_a_source_that_keeps_none_is_refused() {
    let group = Group::start(3).await;
    let leader = group.leader(Duration::from_secs(10)).await;
    let addr = group.nodes[leader].serve_wire().await;
    let mut client = WireClient::connect(addr).await;
    let (_, errors) = client.query("SET search_path = zyron_test").await;
    assert!(
        errors.is_empty(),
        "could not set the search path: {errors:?}"
    );

    let (_, errors) = client
        .query(
            "CREATE EXTERNAL SOURCE no_lsn_src TYPE FILE URI '/tmp/no_lsn' FORMAT CSV \
             COLUMNS (id BIGINT)",
        )
        .await;
    assert!(errors.is_empty(), "the source was refused: {errors:?}");
    group.settle(leader, Duration::from_secs(20)).await;

    let (_, errors) = client
        .query("ALTER EXTERNAL SOURCE no_lsn_src RESET LSN TO 'earliest'")
        .await;
    let joined = errors.join(" ");
    assert!(
        joined.contains("no LSN to reset"),
        "a file-backed source should say it keeps no log position, got {errors:?}"
    );

    client.terminate().await;
    group.shutdown().await;
}

/// Merging a branch into main lands its rows on every member.
///
/// A test of its own rather than a case in the table, because the work has to
/// happen on a session pointed at the branch and the merge then consumes that
/// branch. Leaving the shared connection pointed at a branch that no longer
/// exists would fail every case that ran after it
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_branch_merged_into_main_reaches_every_member() {
    let group = Group::start(3).await;
    let leader = group.leader(Duration::from_secs(10)).await;
    let addr = group.nodes[leader].serve_wire().await;

    let mut client = WireClient::connect(addr).await;
    for sql in [
        "SET search_path = zyron_test",
        "CREATE TABLE conf_branch (id BIGINT PRIMARY KEY, v BIGINT)",
        "INSERT INTO conf_branch (id, v) VALUES (1, 10)",
        "CREATE BRANCH conf_work",
        "USE BRANCH conf_work",
        "INSERT INTO conf_branch (id, v) VALUES (2, 20)",
    ] {
        let (_, errors) = client.query(sql).await;
        assert!(errors.is_empty(), "`{sql}` failed: {errors:?}");
    }
    group.settle(leader, Duration::from_secs(20)).await;

    // Main has not seen the branch's row yet, which is what makes the merge
    // below something to measure rather than a no-op
    assert_eq!(group.nodes[leader].count("conf_branch").await, 1);

    let (_, errors) = client.query("MERGE BRANCH conf_work INTO main").await;
    assert!(errors.is_empty(), "the merge was refused: {errors:?}");
    group.settle(leader, Duration::from_secs(20)).await;

    for node in &group.nodes {
        assert_eq!(
            node.count("conf_branch").await,
            2,
            "{} did not get the row the branch held",
            node.name
        );
    }

    client.terminate().await;
    group.shutdown().await;
}

/// A verified table's chain reaches every member byte for byte.
///
/// The leader links each entry and it travels in the changeset, so every
/// member writes the same bytes in the log's order. That is what makes
/// `VERIFY TABLE` answerable on any member and an anchor meaningful for all
/// of them: two members computing a link from their own clocks would produce
/// different entries for one commit and no anchor would describe both.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn every_member_computes_the_same_chain_head_for_the_same_version() {
    let group = Group::start(3).await;
    let leader = group.leader(Duration::from_secs(10)).await;
    let addr = group.nodes[leader].serve_wire().await;

    let mut client = WireClient::connect(addr).await;
    for sql in [
        "SET search_path = zyron_test",
        "CREATE TABLE conf_chain (id BIGINT PRIMARY KEY, amount BIGINT)",
        "ALTER TABLE conf_chain SET (immutable = true, verified = true)",
    ] {
        let (_, errors) = client.query(sql).await;
        assert!(errors.is_empty(), "`{sql}` failed: {errors:?}");
    }
    group.settle(leader, Duration::from_secs(20)).await;

    let table_id = group.nodes[leader]
        .catalog
        .list_all_tables()
        .iter()
        .find(|t| t.name == "conf_chain")
        .map(|t| t.id.0)
        .expect("the table is in the catalog");

    // Every member agrees the table is verified before a row is written
    for node in &group.nodes {
        let entry = node
            .catalog
            .get_table_by_id(zyron_catalog::TableId(table_id))
            .expect("the table reached this member");
        assert!(
            entry.lifecycle.verified && entry.lifecycle.immutable,
            "{} does not hold the table as verified",
            node.name
        );
    }

    for id in 1..=6 {
        let sql = format!(
            "INSERT INTO conf_chain (id, amount) VALUES ({id}, {})",
            id * 10
        );
        let (_, errors) = client.query(&sql).await;
        assert!(errors.is_empty(), "`{sql}` failed: {errors:?}");
    }
    group.settle(leader, Duration::from_secs(20)).await;

    // Each member's chain, read from its own files
    let mut heads = Vec::new();
    for node in &group.nodes {
        let registry = node
            ._server
            .chain_registry
            .as_ref()
            .expect("every member holds chains");
        let chain = registry.chain(table_id).expect("the chain opens");
        let head = chain.head();
        assert_eq!(
            head.commits, 6,
            "{} chained {} of the 6 commits",
            node.name, head.commits
        );
        // Every entry links to the one before it, on this member's own copy
        let entries = chain.read_range(0, head.commits - 1).expect("reads");
        let mut prev = zyron_lifecycle::verify::NO_PREVIOUS;
        for entry in &entries {
            assert_eq!(
                entry.prev_hash, prev,
                "{} holds a broken link at {}",
                node.name, entry.sequence
            );
            assert_eq!(
                entry.compute_entry_hash(),
                entry.entry_hash,
                "{} holds an entry that does not hash to its link",
                node.name
            );
            prev = entry.entry_hash;
        }
        heads.push((
            node.name.clone(),
            head.head_version,
            zyron_lifecycle::verify::hex(&head.head_hash),
            entries
                .iter()
                .map(|e| zyron_lifecycle::verify::hex(&e.entry_hash))
                .collect::<Vec<_>>(),
        ));
    }

    let (first_name, first_version, first_head, first_entries) = &heads[0];
    for (name, version, head, entries) in &heads[1..] {
        assert_eq!(
            version, first_version,
            "{name} stands at a different version from {first_name}"
        );
        assert_eq!(
            head, first_head,
            "{name} computed a different head from {first_name} for the same version"
        );
        assert_eq!(
            entries, first_entries,
            "{name} holds different entries from {first_name}"
        );
    }

    // VERIFY TABLE answers on the leader, which reads this member's own
    // chain and its own rows
    let (rows, errors) = client
        .query_rows("VERIFY TABLE conf_chain WITH (rows => 'all')")
        .await;
    assert!(
        errors.is_empty(),
        "the verification was refused: {errors:?}"
    );
    let answered = rows
        .iter()
        .flat_map(|row| row.iter().map(|cell| cell.clone().unwrap_or_default()))
        .collect::<Vec<_>>()
        .join(" | ");
    assert!(answered.contains("conf_chain"), "{answered}");
    assert!(answered.contains("true"), "not intact: {answered}");
    assert!(
        answered.contains("all"),
        "the mode is not stated: {answered}"
    );
    assert!(
        answered.contains(" 6 "),
        "six commits were not checked: {answered}"
    );

    // Every member reads its own rows back by the stamps its own
    // transactions put on them, and each one's chain covers exactly what it
    // holds, which is what proves a member linked what it applied rather
    // than what it was told
    for node in &group.nodes {
        let outcome = zyron_wire::verify_dispatch::run_verify(
            &node._server,
            zyron_wire::verify_dispatch::VerifyRequest {
                table_id,
                from_version: None,
                to_version: None,
                mode: zyron_lifecycle::verify::RowMode::All,
                sample: 0,
            },
            0,
            "conformance",
            Arc::new(|| false),
        )
        .await
        .unwrap_or_else(|e| panic!("{} could not verify its copy: {e}", node.name));
        assert!(
            outcome.intact,
            "{} holds rows its chain does not cover: {:?}",
            node.name, outcome.failure
        );
        assert_eq!(outcome.commits_checked, 6, "{}", node.name);
        assert_eq!(outcome.commits_rehashed, 6, "{}", node.name);
        assert_eq!(outcome.rows_checked, 6, "{}", node.name);
    }

    client.terminate().await;
    group.shutdown().await;
}

/// Writers committing together to one verified table on a group link in
/// the log's order on every member.
///
/// The connections hash their own rows and the group decides the order the
/// commits apply in, so the order two commits are proposed in and the order
/// they link in are one order, whichever connection reached its commit
/// first. Every member's chain holds every commit, links clean, agrees with
/// every other member's, and covers the rows the member holds
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn concurrent_writers_link_in_the_log_order_on_every_member() {
    let group = Group::start(3).await;
    let leader = group.leader(Duration::from_secs(10)).await;
    let addr = group.nodes[leader].serve_wire().await;

    let mut admin = WireClient::connect(addr).await;
    for sql in [
        "SET search_path = zyron_test",
        "CREATE TABLE conf_race (id BIGINT PRIMARY KEY, writer BIGINT)",
        "ALTER TABLE conf_race SET (immutable = true, verified = true)",
    ] {
        let (_, errors) = admin.query(sql).await;
        assert!(errors.is_empty(), "`{sql}` failed: {errors:?}");
    }
    group.settle(leader, Duration::from_secs(20)).await;

    let writers = 8u64;
    let per_writer = 12u64;
    let mut tasks = Vec::with_capacity(writers as usize);
    for writer in 0..writers {
        tasks.push(tokio::spawn(async move {
            let mut client = WireClient::connect(addr).await;
            let (_, errors) = client.query("SET search_path = zyron_test").await;
            assert!(errors.is_empty(), "{errors:?}");
            for run in 0..per_writer {
                let id = writer * per_writer + run + 1;
                let sql = format!("INSERT INTO conf_race (id, writer) VALUES ({id}, {writer})");
                let (_, errors) = client.query(&sql).await;
                assert!(errors.is_empty(), "`{sql}` failed: {errors:?}");
            }
            client.terminate().await;
        }));
    }
    for task in tasks {
        task.await.expect("the writer finished");
    }
    group.settle(leader, Duration::from_secs(30)).await;

    let table_id = group.nodes[leader]
        .catalog
        .list_all_tables()
        .iter()
        .find(|t| t.name == "conf_race")
        .map(|t| t.id.0)
        .expect("the table is in the catalog");
    let expected = writers * per_writer;

    let mut heads = Vec::new();
    for node in &group.nodes {
        let registry = node
            ._server
            .chain_registry
            .as_ref()
            .expect("every member holds chains");
        let chain = registry.chain(table_id).expect("the chain opens");
        let head = chain.head();
        assert_eq!(
            head.commits, expected,
            "{} chained {} of the {expected} commits",
            node.name, head.commits
        );
        assert_eq!(
            chain.pending_commits(),
            0,
            "{} holds entries whose commit never settled",
            node.name
        );
        let outcome = zyron_wire::verify_dispatch::run_verify(
            &node._server,
            zyron_wire::verify_dispatch::VerifyRequest {
                table_id,
                from_version: None,
                to_version: None,
                mode: zyron_lifecycle::verify::RowMode::All,
                sample: 0,
            },
            0,
            "conformance",
            Arc::new(|| false),
        )
        .await
        .unwrap_or_else(|e| panic!("{} could not verify its copy: {e}", node.name));
        assert!(
            outcome.intact,
            "{} holds rows its chain does not cover: {:?}",
            node.name, outcome.failure
        );
        assert_eq!(outcome.commits_checked, expected, "{}", node.name);
        assert_eq!(outcome.rows_checked, expected, "{}", node.name);
        heads.push((
            node.name.clone(),
            zyron_lifecycle::verify::hex(&head.head_hash),
        ));
    }
    let (first_name, first_head) = &heads[0];
    for (name, head) in &heads[1..] {
        assert_eq!(
            head, first_head,
            "{name} computed a different head from {first_name}"
        );
    }

    admin.terminate().await;
    group.shutdown().await;
}

/// A temporary table created on the leader is absent on every follower.
///
/// The only case here whose proof is an absence, and it is the point of the
/// feature rather than a gap in it: a temporary table lives in the session
/// that created it, on the node that session is connected to, so a statement
/// about one classifies Local and reaches no other member. What this checks
/// is that none of it leaks: not the definition, not the rows, and not a
/// catalog entry on any node including the one that ran it.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_temporary_table_stays_on_the_node_that_created_it() {
    let group = Group::start(3).await;
    let leader = group.leader(Duration::from_secs(10)).await;
    let addr = group.nodes[leader].serve_wire().await;

    let mut client = WireClient::connect(addr).await;
    for sql in [
        "SET search_path = zyron_test",
        // A permanent table beside it, so the group is proven to be
        // replicating at all rather than quietly doing nothing
        "CREATE TABLE conf_permanent (id BIGINT PRIMARY KEY)",
        "INSERT INTO conf_permanent (id) VALUES (1)",
        "CREATE TEMP TABLE conf_scratch (id BIGINT)",
        "INSERT INTO conf_scratch (id) VALUES (1), (2), (3)",
    ] {
        let (_, errors) = client.query(sql).await;
        assert!(errors.is_empty(), "`{sql}` failed: {errors:?}");
    }
    group.settle(leader, Duration::from_secs(20)).await;

    // The permanent table reached every member, so the group is working
    for node in &group.nodes {
        assert_eq!(
            node.count("conf_permanent").await,
            1,
            "{} did not get the permanent table's row",
            node.name
        );
    }

    // The temporary table reached none of them, the leader included: its
    // definition is in the session, not in any catalog
    for node in &group.nodes {
        assert!(
            !node
                .catalog
                .list_all_tables()
                .iter()
                .any(|t| t.name == "conf_scratch"),
            "{} has a catalog entry for a temporary table",
            node.name
        );
        assert!(
            node.catalog.get_table(node.schema, "conf_scratch").is_err(),
            "{} resolves a temporary table by schema and name",
            node.name
        );
    }

    // A follower cannot read it under any name
    let follower = (leader + 1) % group.nodes.len();
    let follower_addr = group.nodes[follower].serve_wire().await;
    let mut other = WireClient::connect(follower_addr).await;
    let (_, errors) = other.query("SET search_path = zyron_test").await;
    assert!(
        errors.is_empty(),
        "could not set the search path: {errors:?}"
    );
    let (_, errors) = other.query("SELECT id FROM conf_scratch").await;
    assert!(
        !errors.is_empty(),
        "a follower answered a query against a temporary table it should not have"
    );
    other.terminate().await;

    // A second session on the leader does not see it either, because the
    // namespace belongs to the connection rather than to the node
    let mut second = WireClient::connect(addr).await;
    let (_, errors) = second.query("SET search_path = zyron_test").await;
    assert!(
        errors.is_empty(),
        "could not set the search path: {errors:?}"
    );
    let (_, errors) = second.query("SELECT id FROM conf_scratch").await;
    assert!(
        !errors.is_empty(),
        "another session on the same node saw a temporary table that is not its own"
    );
    second.terminate().await;

    // Dropping it is Local too, so it is not proposed to the group either
    let (_, errors) = client.query("DROP TABLE conf_scratch").await;
    assert!(errors.is_empty(), "the drop was refused: {errors:?}");
    group.settle(leader, Duration::from_secs(20)).await;
    for node in &group.nodes {
        assert_eq!(
            node.count("conf_permanent").await,
            1,
            "{} lost rows while a temporary table was dropped",
            node.name
        );
    }

    client.terminate().await;
    group.shutdown().await;
}

/// A notification issued on one member reaches a listener on another.
///
/// A test of its own because what it proves is delivery rather than catalog
/// or row state. The listener subscribes on a follower's own channel registry,
/// which is exactly what a client running LISTEN against that node waits on.
/// Answered only where it was typed, which is what happened before, a client
/// listening anywhere else never heard a thing
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_notification_reaches_a_listener_on_another_member() {
    let group = Group::start(3).await;
    let leader = group.leader(Duration::from_secs(10)).await;
    let follower = (0..group.nodes.len())
        .find(|i| *i != leader)
        .expect("a follower");

    // Subscribed before the notification is sent, the way a client that ran
    // LISTEN is already waiting
    let mut listener = group.nodes[follower]
        ._server
        .notification_channels
        .as_ref()
        .expect("notification channels")
        .listen("conf_channel");

    let mut sender = WireClient::connect(group.nodes[leader].serve_wire().await).await;
    let (_, errors) = sender.query("NOTIFY conf_channel, 'from the leader'").await;
    assert!(
        errors.is_empty(),
        "the notification was refused: {errors:?}"
    );
    group.settle(leader, Duration::from_secs(20)).await;

    let heard = tokio::time::timeout(Duration::from_secs(10), listener.recv())
        .await
        .expect("the follower's listener heard nothing before the deadline")
        .expect("the channel closed");
    assert_eq!(heard.channel, "conf_channel");
    assert_eq!(heard.payload, "from the leader");

    sender.terminate().await;
    group.shutdown().await;
}
