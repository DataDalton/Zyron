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

fn tables(node: &Node) -> String {
    render(
        node.catalog
            .list_all_tables()
            .iter()
            .map(|t| format!("{}:{}", t.name, t.columns.len()))
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
];

// ---------------------------------------------------------------------------
// The run
// ---------------------------------------------------------------------------

/// What went wrong with one case, in the words a fix would need.
struct Failure {
    case: &'static str,
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

    for case in CASES {
        let mut setup_failed = None;
        for sql in case.setup {
            let (_, errors) = client.query(sql).await;
            if !errors.is_empty() {
                setup_failed = Some(format!("setup `{sql}` failed: {errors:?}"));
                break;
            }
        }
        if let Some(detail) = setup_failed {
            failures.push(Failure {
                case: case.name,
                detail,
            });
            continue;
        }
        group.settle(leader, Duration::from_secs(20)).await;

        let before = read_proof(&group.nodes[leader], &case.proof).await;
        let (tags, errors) = client.query(case.sql).await;
        if !errors.is_empty() {
            failures.push(Failure {
                case: case.name,
                detail: format!("refused on the leader: {errors:?}"),
            });
            continue;
        }
        group.settle(leader, Duration::from_secs(20)).await;

        // A statement that reported success and changed nothing is a silent
        // no-op, which reads as a pass to any check that only compares members
        let after = read_proof(&group.nodes[leader], &case.proof).await;
        if before == after {
            failures.push(Failure {
                case: case.name,
                detail: format!("answered {tags:?} and changed nothing, still `{after}`"),
            });
            continue;
        }

        for node in &group.nodes {
            let seen = read_proof(node, &case.proof).await;
            if seen != after {
                failures.push(Failure {
                    case: case.name,
                    detail: format!(
                        "the leader holds `{after}` and {} holds `{seen}`",
                        node.name
                    ),
                });
            }
        }
    }

    client.terminate().await;
    group.shutdown().await;

    // A pass here means nothing unless a refusal would have been caught, and
    // `a_statement_this_release_refuses_says_so` is what shows it would

    assert!(
        failures.is_empty(),
        "{} of {} statements did not reach every member:\n{}",
        failures.len(),
        CASES.len(),
        failures
            .iter()
            .map(|f| format!("  {}: {}", f.case, f.detail))
            .collect::<Vec<_>>()
            .join("\n")
    );
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
