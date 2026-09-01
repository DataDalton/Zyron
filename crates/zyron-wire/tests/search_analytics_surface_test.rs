//! The SQL surface harvest: search DDL (analyzers, synonym dictionaries,
//! hybrid indexes, phonetic matching), resilience policies and their
//! callable forms, expectation metrics, the analytics table function
//! families (anomalies, clustering, causal inference, feature encoding,
//! statistical tests), vector math, money conversion over the writable
//! currency rates table, retention dashboard views, PII masking, semver,
//! and bloom filter estimation.
//!
//! Run: cargo test -p zyron-wire --test search_analytics_surface_test

mod common;

use std::sync::Arc;

use common::{
    create_test_server, exec_ddl, exec_dml, new_session, query_error, query_values,
    try_query_values, wire_query,
};
use zyron_executor::column::ScalarValue;
use zyron_wire::connection::ServerState;

// ---------------------------------------------------------------------------
// Value readers
// ---------------------------------------------------------------------------

fn as_f64(v: &ScalarValue) -> f64 {
    match v {
        ScalarValue::Int8(x) => *x as f64,
        ScalarValue::Int16(x) => *x as f64,
        ScalarValue::Int32(x) => *x as f64,
        ScalarValue::Int64(x) => *x as f64,
        ScalarValue::UInt8(x) => *x as f64,
        ScalarValue::UInt16(x) => *x as f64,
        ScalarValue::UInt32(x) => *x as f64,
        ScalarValue::UInt64(x) => *x as f64,
        ScalarValue::Float32(x) => *x as f64,
        ScalarValue::Float64(x) => *x,
        other => panic!("expected a number, got {other:?}"),
    }
}

fn as_i64(v: &ScalarValue) -> i64 {
    match v {
        ScalarValue::Int8(x) => *x as i64,
        ScalarValue::Int16(x) => *x as i64,
        ScalarValue::Int32(x) => *x as i64,
        ScalarValue::Int64(x) => *x,
        ScalarValue::UInt32(x) => *x as i64,
        ScalarValue::UInt64(x) => *x as i64,
        other => panic!("expected an integer, got {other:?}"),
    }
}

fn as_str(v: &ScalarValue) -> String {
    match v {
        ScalarValue::Utf8(s) => s.clone(),
        other => panic!("expected text, got {other:?}"),
    }
}

fn as_bool(v: &ScalarValue) -> bool {
    match v {
        ScalarValue::Boolean(b) => *b,
        other => panic!("expected a boolean, got {other:?}"),
    }
}

/// First column of every row as sorted integers
fn first_ints_sorted(rows: &[Vec<ScalarValue>]) -> Vec<i64> {
    let mut out: Vec<i64> = rows.iter().map(|r| as_i64(&r[0])).collect();
    out.sort_unstable();
    out
}

// ---------------------------------------------------------------------------
// System entity reader, the same path a client SELECT of zyron_sys takes
// ---------------------------------------------------------------------------

async fn read_system(
    server: &Arc<ServerState>,
    sql: &str,
) -> Result<(Vec<String>, Vec<Vec<String>>), String> {
    let stmt = zyron_parser::parse(sql)
        .map_err(|e| format!("parse: {e}"))?
        .into_iter()
        .next()
        .ok_or_else(|| "no statement".to_string())?;
    let sel = match stmt {
        zyron_parser::Statement::Select(sel) => sel,
        other => return Err(format!("not a select: {other:?}")),
    };

    // A call with arguments goes down the function path, a bare name down
    // the relation path, the same split the wire dispatch makes
    if let Some(parsed) = zyron_wire::system_views::parse_system_function(&sel) {
        let call = parsed.map_err(|e| e.to_string())?;
        let name = call.object.canonical_name();
        let filters = zyron_wire::system_views::parse_system_view_query(&name, &sel)
            .map_err(|e| e.to_string())?;
        let built = zyron_wire::system_views::query_system_function(&call, server, &filters)
            .await
            .map_err(|e| e.to_string())?;
        return Ok(render_system(built));
    }

    let name = match &sel.from[0] {
        zyron_parser::TableRef::Table { name, .. } => name.clone(),
        other => return Err(format!("not a plain table ref: {other:?}")),
    };
    if !zyron_wire::system_views::is_system_view(&name) {
        return Err(format!("{name} is not a registered system view"));
    }
    let filters = zyron_wire::system_views::parse_system_view_query(&name, &sel)
        .map_err(|e| e.to_string())?;
    let built = zyron_wire::system_views::query_system_view(&name, server, &filters)
        .await
        .map_err(|e| e.to_string())?
        .ok_or_else(|| format!("{name} is registered but built nothing"))?;
    Ok(render_system(built))
}

fn render_system(
    built: (
        Vec<zyron_wire::messages::backend::FieldDescription>,
        Vec<Vec<Option<Vec<u8>>>>,
    ),
) -> (Vec<String>, Vec<Vec<String>>) {
    let (fields, rows) = built;
    let names = fields.iter().map(|f| f.name.clone()).collect();
    let rows = rows
        .into_iter()
        .map(|row| {
            row.into_iter()
                .map(|cell| {
                    cell.map(|b| String::from_utf8_lossy(&b).into_owned())
                        .unwrap_or_default()
                })
                .collect()
        })
        .collect();
    (names, rows)
}

// ---------------------------------------------------------------------------
// Task 1, CREATE ANALYZER
// ---------------------------------------------------------------------------

/// Analyzer DDL round trips, refuses a duplicate, and a fulltext index
/// created with the analyzer actually analyzes through it, proven by a
/// stemmed query term matching an inflected document term
#[tokio::test]
async fn test_create_analyzer_ddl_and_fts_index_using_it() {
    let (server, _schema, _tmp) = create_test_server().await;
    let mut session = new_session();

    exec_ddl(
        &server,
        &mut session,
        "CREATE ANALYZER fancy_an AS (
            TOKENIZER = 'ngram(3, 5)',
            CHAR_FILTERS = ARRAY['html_strip', 'lowercase'],
            TOKEN_FILTERS = ARRAY['stop', 'stem', 'phonetic:metaphone']
        )",
    )
    .await
    .expect("create analyzer");

    let dup = exec_ddl(
        &server,
        &mut session,
        "CREATE ANALYZER fancy_an AS (TOKENIZER = 'standard')",
    )
    .await
    .expect_err("duplicate analyzer must be refused");
    assert!(dup.contains("already exists"), "unexpected error: {dup}");

    exec_ddl(
        &server,
        &mut session,
        "ALTER ANALYZER fancy_an SET (tokenizer = 'standard')",
    )
    .await
    .expect("alter analyzer");

    exec_ddl(&server, &mut session, "DROP ANALYZER fancy_an")
        .await
        .expect("drop analyzer");
    let missing = exec_ddl(&server, &mut session, "DROP ANALYZER fancy_an")
        .await
        .expect_err("dropping a dropped analyzer must fail");
    assert!(
        missing.contains("does not exist"),
        "unexpected error: {missing}"
    );
    exec_ddl(&server, &mut session, "DROP ANALYZER IF EXISTS fancy_an")
        .await
        .expect("IF EXISTS tolerates the missing analyzer");

    // A stemming analyzer bound to an index, exercised through MATCH.
    // The builtin simple pipeline does not stem, so 'sleeping' finding
    // 'sleeps' proves the index runs the configured analyzer
    exec_ddl(
        &server,
        &mut session,
        "CREATE ANALYZER body_an AS (TOKENIZER = 'standard', TOKEN_FILTERS = ARRAY['lowercase', 'stem'])",
    )
    .await
    .expect("create body analyzer");
    exec_ddl(
        &server,
        &mut session,
        "CREATE TABLE an_docs (id INT, body TEXT)",
    )
    .await
    .expect("create table");
    exec_ddl(
        &server,
        &mut session,
        "CREATE FULLTEXT INDEX an_docs_fts ON an_docs (body) WITH (analyzer = 'body_an')",
    )
    .await
    .expect("create fulltext index with analyzer");
    exec_dml(
        &server,
        "INSERT INTO an_docs VALUES (1, 'The Quick Brown Fox'), (2, 'lazy dog sleeps'), (3, 'quick silver')",
    )
    .await;

    let quick = query_values(
        &server,
        "SELECT id FROM an_docs WHERE MATCH(body) AGAINST('quick')",
    )
    .await;
    assert_eq!(
        first_ints_sorted(&quick),
        vec![1, 3],
        "lowercase filter matches Quick and quick"
    );

    let stemmed = query_values(
        &server,
        "SELECT id FROM an_docs WHERE MATCH(body) AGAINST('sleeping')",
    )
    .await;
    assert_eq!(
        first_ints_sorted(&stemmed),
        vec![2],
        "the stem filter reduces sleeping and sleeps to one term"
    );

    // An index naming an unknown analyzer is refused at creation
    let bad = exec_ddl(
        &server,
        &mut session,
        "CREATE FULLTEXT INDEX bad_fts ON an_docs (body) WITH (analyzer = 'no_such_an')",
    )
    .await
    .expect_err("unknown analyzer must fail the index create");
    assert!(bad.contains("does not exist"), "unexpected error: {bad}");
}

// ---------------------------------------------------------------------------
// Task 2, CREATE SYNONYM DICTIONARY
// ---------------------------------------------------------------------------

/// A dictionary attached to a fulltext index expands terms both ways,
/// ALTER ADD reaches live queries, and a referenced dictionary refuses to
/// drop until the index goes first
#[tokio::test]
async fn test_synonym_dictionary_expands_matches() {
    let (server, _schema, _tmp) = create_test_server().await;
    let mut session = new_session();

    exec_ddl(
        &server,
        &mut session,
        "CREATE SYNONYM DICTIONARY syn_dict (('car', 'automobile', 'vehicle'), ('nyc' => 'new york city'))",
    )
    .await
    .expect("create dictionary");

    exec_ddl(
        &server,
        &mut session,
        "CREATE TABLE syn_docs (id INT, body TEXT)",
    )
    .await
    .expect("create table");
    exec_ddl(
        &server,
        &mut session,
        "CREATE FULLTEXT INDEX syn_docs_fts ON syn_docs (body) WITH (synonyms = 'syn_dict')",
    )
    .await
    .expect("create index with dictionary");
    exec_dml(
        &server,
        "INSERT INTO syn_docs VALUES (1, 'my automobile is fast'), (2, 'the weather is nice'), (3, 'bicycle lane ahead')",
    )
    .await;

    let car = query_values(
        &server,
        "SELECT id FROM syn_docs WHERE MATCH(body) AGAINST('car')",
    )
    .await;
    assert_eq!(
        first_ints_sorted(&car),
        vec![1],
        "car finds the automobile document through the synonym group"
    );

    // A rule added after indexing reaches new queries through the
    // refreshed analyzer
    exec_ddl(
        &server,
        &mut session,
        "ALTER SYNONYM DICTIONARY syn_dict ADD ('bike', 'bicycle')",
    )
    .await
    .expect("alter dictionary add");
    let bike = query_values(
        &server,
        "SELECT id FROM syn_docs WHERE MATCH(body) AGAINST('bike')",
    )
    .await;
    assert_eq!(
        first_ints_sorted(&bike),
        vec![3],
        "the added bike group expands the query to bicycle"
    );

    // Dropping a term prunes it from the rules
    exec_ddl(
        &server,
        &mut session,
        "ALTER SYNONYM DICTIONARY syn_dict DROP 'nyc'",
    )
    .await
    .expect("alter dictionary drop term");

    let refused = exec_ddl(&server, &mut session, "DROP SYNONYM DICTIONARY syn_dict")
        .await
        .expect_err("a referenced dictionary must refuse to drop");
    assert!(
        refused.contains("syn_docs_fts"),
        "the refusal names the referencing index: {refused}"
    );

    exec_ddl(&server, &mut session, "DROP INDEX syn_docs_fts")
        .await
        .expect("drop index");
    exec_ddl(&server, &mut session, "DROP SYNONYM DICTIONARY syn_dict")
        .await
        .expect("drop dictionary once unreferenced");
}

// ---------------------------------------------------------------------------
// Task 3, CREATE HYBRID INDEX
// ---------------------------------------------------------------------------

/// Hybrid index creation lands one catalog entry of the Hybrid type,
/// refuses bad configuration, and DROP INDEX removes it
#[tokio::test]
async fn test_create_and_drop_hybrid_index() {
    let (server, schema, _tmp) = create_test_server().await;
    let mut session = new_session();

    exec_ddl(
        &server,
        &mut session,
        "CREATE TABLE hy_t (id INT, body TEXT, emb VECTOR(3))",
    )
    .await
    .expect("create table");
    exec_ddl(
        &server,
        &mut session,
        "CREATE HYBRID INDEX hy_idx ON hy_t (body, emb) WITH (
            fulltext_analyzer = 'standard',
            vector_distance = 'cosine',
            fusion_method = 'rrf',
            rrf_k = 60
        )",
    )
    .await
    .expect("create hybrid index");

    let table = server.catalog.get_table(schema, "hy_t").expect("table");
    let indexes = server.catalog.get_indexes_for_table(table.id);
    assert_eq!(indexes.len(), 1, "one index on the table");
    assert_eq!(indexes[0].name, "hy_idx");
    assert_eq!(indexes[0].index_type, zyron_catalog::IndexType::Hybrid);

    let bad_fusion = exec_ddl(
        &server,
        &mut session,
        "CREATE HYBRID INDEX hy_bad ON hy_t (body, emb) WITH (fusion_method = 'bogus')",
    )
    .await
    .expect_err("an unknown fusion method must be refused");
    assert!(
        bad_fusion.contains("fusion_method must be rrf or linear"),
        "unexpected error: {bad_fusion}"
    );

    let bad_column = exec_ddl(
        &server,
        &mut session,
        "CREATE HYBRID INDEX hy_bad2 ON hy_t (body, id)",
    )
    .await
    .expect_err("a non vector second column must be refused");
    assert!(
        bad_column.contains("must be a VECTOR column"),
        "unexpected error: {bad_column}"
    );

    exec_ddl(&server, &mut session, "DROP INDEX hy_idx")
        .await
        .expect("drop hybrid index");
    let table = server.catalog.get_table(schema, "hy_t").expect("table");
    assert!(
        server.catalog.get_indexes_for_table(table.id).is_empty(),
        "the hybrid index is gone from the catalog"
    );
}

// ---------------------------------------------------------------------------
// Task 4, phonetic FTS
// ---------------------------------------------------------------------------

/// An index whose analyzer encodes terms phonetically answers a
/// PHONETIC MODE match across spellings, and the phonetic scalar
/// functions agree
#[tokio::test]
async fn test_phonetic_fts_and_scalar_functions() {
    let (server, _schema, _tmp) = create_test_server().await;
    let mut session = new_session();

    exec_ddl(
        &server,
        &mut session,
        "CREATE ANALYZER ph_an AS (TOKENIZER = 'standard', TOKEN_FILTERS = ARRAY['lowercase', 'phonetic:metaphone'])",
    )
    .await
    .expect("create phonetic analyzer");
    exec_ddl(
        &server,
        &mut session,
        "CREATE TABLE ph_people (id INT, name TEXT)",
    )
    .await
    .expect("create table");
    exec_ddl(
        &server,
        &mut session,
        "CREATE FULLTEXT INDEX ph_idx ON ph_people (name) WITH (analyzer = 'ph_an')",
    )
    .await
    .expect("create index");
    exec_dml(
        &server,
        "INSERT INTO ph_people VALUES (1, 'Smith'), (2, 'Smyth'), (3, 'Jones')",
    )
    .await;

    let matched = query_values(
        &server,
        "SELECT id FROM ph_people WHERE MATCH(name) AGAINST ('smith' IN PHONETIC MODE)",
    )
    .await;
    assert_eq!(
        first_ints_sorted(&matched),
        vec![1, 2],
        "Smith and Smyth share a metaphone code, Jones does not"
    );

    let rows = query_values(
        &server,
        "SELECT phonetic_match('Smith', 'Smyth', 'metaphone'), \
                phonetic_match('Smith', 'Jones', 'metaphone'), \
                phonetic_score('Robert', 'Rupert', 'soundex')",
    )
    .await;
    assert!(as_bool(&rows[0][0]), "Smith and Smyth match phonetically");
    assert!(!as_bool(&rows[0][1]), "Smith and Jones do not match");
    assert_eq!(
        as_f64(&rows[0][2]),
        1.0,
        "Robert and Rupert share a soundex code exactly"
    );
}

// ---------------------------------------------------------------------------
// Task 5, HYBRID_SEARCH table function
// ---------------------------------------------------------------------------

/// The table function fuses both halves of a hybrid index and returns
/// ranked rows with the per side raw figures
#[tokio::test]
async fn test_hybrid_search_table_function() {
    let (server, _schema, _tmp) = create_test_server().await;
    let mut session = new_session();

    exec_ddl(
        &server,
        &mut session,
        "CREATE TABLE hs_t (id INT, body TEXT, emb VECTOR(3))",
    )
    .await
    .expect("create table");
    exec_ddl(
        &server,
        &mut session,
        "CREATE HYBRID INDEX hs_idx ON hs_t (body, emb) WITH (fusion_method = 'rrf', rrf_k = 60)",
    )
    .await
    .expect("create hybrid index");
    exec_dml(
        &server,
        "INSERT INTO hs_t VALUES \
         (1, 'red apple pie', ARRAY[1.0, 0.0, 0.0]), \
         (2, 'blue sky today', ARRAY[0.0, 1.0, 0.0]), \
         (3, 'green grass field', ARRAY[0.0, 0.0, 1.0])",
    )
    .await;

    let rows = query_values(
        &server,
        "SELECT * FROM hybrid_search('hs_idx', 'apple', '[1.0, 0.0, 0.0]', k => 2)",
    )
    .await;
    assert_eq!(rows.len(), 2, "k caps the fused result");
    let top_score = as_f64(&rows[0][1]);
    let second_score = as_f64(&rows[1][1]);
    assert!(
        top_score > second_score,
        "rows come back ranked, got {top_score} then {second_score}"
    );
    // Only the apple document matches the text query, so the top fused row
    // carrying a full text score identifies it
    assert!(
        !matches!(rows[0][2], ScalarValue::Null),
        "the top row matched the full text half"
    );
    assert!(
        !matches!(rows[0][3], ScalarValue::Null),
        "the top row was found by the vector half"
    );
    let distance = as_f64(&rows[0][3]);
    assert!(
        distance.abs() < 1e-5,
        "the query vector equals the top row's vector, distance {distance}"
    );

    let missing = query_error(
        &server,
        "SELECT * FROM hybrid_search('no_such_idx', 'apple', '[1.0, 0.0, 0.0]')",
    )
    .await;
    assert!(
        missing.contains("does not exist"),
        "unexpected error: {missing}"
    );
}

// ---------------------------------------------------------------------------
// Task 6, zyron_sys.query.recommend_indexes
// ---------------------------------------------------------------------------

/// Filtered scans of an unindexed selective column on a large analyzed
/// table surface as a recommendation with a benefit score and a runnable
/// sample query
#[tokio::test]
async fn test_recommend_indexes_reports_scanned_column() {
    let (server, _schema, _tmp) = create_test_server().await;
    let mut session = new_session();

    exec_ddl(
        &server,
        &mut session,
        "CREATE TABLE reco_probe (id INT, v INT)",
    )
    .await
    .expect("create table");

    // The advisor only recommends for tables of at least ten thousand rows
    for chunk in 0..4 {
        let mut sql = String::from("INSERT INTO reco_probe VALUES ");
        for i in 0..3000 {
            let id = chunk * 3000 + i;
            if i > 0 {
                sql.push_str(", ");
            }
            sql.push_str(&format!("({id}, {})", id % 40));
        }
        exec_dml(&server, &sql).await;
    }

    // ANALYZE through the production connection loop fills the stats the
    // advisor reads row counts and distinct counts from
    let outcomes = wire_query(&server, &["ANALYZE reco_probe"]).await;
    assert!(
        outcomes[0].errors.is_empty(),
        "ANALYZE failed: {:?}",
        outcomes[0].errors
    );

    for _ in 0..3 {
        let rows = query_values(&server, "SELECT * FROM reco_probe WHERE v = 7").await;
        assert_eq!(rows.len(), 300, "40 distinct values over 12000 rows");
    }

    let (columns, rows) = read_system(&server, "SELECT * FROM zyron_sys.query.recommend_indexes()")
        .await
        .expect("recommendations");
    let at = |name: &str| columns.iter().position(|c| c == name).expect(name);
    let ours: Vec<&Vec<String>> = rows
        .iter()
        .filter(|r| r[at("table_name")] == "reco_probe")
        .collect();
    assert_eq!(
        ours.len(),
        1,
        "one recommendation for the scanned column, got {rows:?}"
    );
    let rec = ours[0];
    assert_eq!(rec[at("column_names")], "v");
    let scans: u64 = rec[at("scan_count")].parse().expect("scan_count");
    assert!(
        scans >= 3,
        "three filtered scans were recorded, got {scans}"
    );
    let benefit: f64 = rec[at("estimated_benefit_score")]
        .parse()
        .expect("benefit parses");
    assert!(benefit > 0.0, "a selective scanned column has benefit");
    assert_eq!(
        rec[at("sample_query")],
        "SELECT * FROM reco_probe WHERE v = $1"
    );
    assert_eq!(
        rec[at("create_statement")],
        "CREATE INDEX idx_reco_probe_v ON reco_probe (v)"
    );
}

// ---------------------------------------------------------------------------
// Task 7, session.prepared_statements and sql.triggers views
// ---------------------------------------------------------------------------

/// A wire PREPARE publishes a row for the session view while the
/// connection lives, and the view clears when it closes
#[tokio::test]
async fn test_prepared_statements_view_follows_the_connection() {
    let (server, _schema, _tmp) = create_test_server().await;

    let outcomes = wire_query(
        &server,
        &[
            "PREPARE p_probe AS SELECT 1",
            "SELECT * FROM zyron_sys.session.prepared_statements",
        ],
    )
    .await;
    assert!(
        outcomes[0].errors.is_empty(),
        "PREPARE failed: {:?}",
        outcomes[0].errors
    );
    assert!(
        outcomes[0].tags.iter().any(|t| t == "PREPARE"),
        "PREPARE completes: {:?}",
        outcomes[0].tags
    );
    assert_eq!(
        outcomes[1].tags,
        vec!["SELECT 1".to_string()],
        "the live connection's one prepared statement is the one row"
    );

    // After the connection closed its published rows are gone
    let (columns, rows) = read_system(
        &server,
        "SELECT * FROM zyron_sys.session.prepared_statements",
    )
    .await
    .expect("view answers");
    assert_eq!(
        columns,
        vec!["pid", "name", "query", "param_count", "planned"]
    );
    assert!(
        rows.is_empty(),
        "a closed connection leaves no rows, got {rows:?}"
    );
}

/// A created trigger appears in zyron_sys.sql.triggers with its timing,
/// events, granularity, and function
#[tokio::test]
async fn test_triggers_view_lists_a_created_trigger() {
    let (server, _schema, _tmp) = create_test_server().await;
    let mut session = new_session();

    exec_ddl(&server, &mut session, "CREATE TABLE trg_t (id INT, v INT)")
        .await
        .expect("create table");
    exec_ddl(&server, &mut session, "CREATE TABLE trg_audit (tid INT)")
        .await
        .expect("create audit table");
    exec_ddl(
        &server,
        &mut session,
        "CREATE PROCEDURE trg_log() AS 'INSERT INTO zyron_test.trg_audit (tid) VALUES ($1)' LANGUAGE SQL",
    )
    .await
    .expect("create procedure");
    exec_ddl(
        &server,
        &mut session,
        "CREATE TRIGGER trg_probe AFTER INSERT ON trg_t FOR EACH ROW EXECUTE FUNCTION zyron_test.trg_log",
    )
    .await
    .expect("create trigger");

    let (columns, rows) = read_system(&server, "SELECT * FROM zyron_sys.sql.triggers")
        .await
        .expect("triggers view");
    let at = |name: &str| columns.iter().position(|c| c == name).expect(name);
    let ours: Vec<&Vec<String>> = rows
        .iter()
        .filter(|r| r[at("trigger_name")] == "trg_probe")
        .collect();
    assert_eq!(ours.len(), 1, "the created trigger has one row");
    let row = ours[0];
    assert_eq!(row[at("table_name")], "trg_t");
    assert_eq!(row[at("timing")], "after");
    assert_eq!(row[at("events")], "insert");
    assert_eq!(row[at("for_each")], "row");
    assert!(
        row[at("execute_function")].contains("trg_log"),
        "the function column names the procedure: {}",
        row[at("execute_function")]
    );
    assert_eq!(row[at("enabled")], "true");
}

// ---------------------------------------------------------------------------
// Task 8, resilience patterns
// ---------------------------------------------------------------------------

/// Bulkhead and retry policy DDL round trips, refuses duplicates, bad
/// options, and kind mismatched drops
#[tokio::test]
async fn test_bulkhead_and_retry_policy_ddl() {
    let (server, _schema, _tmp) = create_test_server().await;
    let mut session = new_session();

    exec_ddl(
        &server,
        &mut session,
        "CREATE BULKHEAD my_bh (max_concurrent = 10, max_wait = '5s', queue_size = 100)",
    )
    .await
    .expect("create bulkhead");
    let dup = exec_ddl(
        &server,
        &mut session,
        "CREATE BULKHEAD my_bh (max_concurrent = 1)",
    )
    .await
    .expect_err("duplicate bulkhead refused");
    assert!(dup.contains("already exists"), "unexpected error: {dup}");

    exec_ddl(
        &server,
        &mut session,
        "CREATE RETRY POLICY my_rp (max_attempts = 3, backoff = 'exponential', \
         base_delay = '100ms', max_delay = '10s', jitter = 0.2, \
         retryable_errors = ARRAY['TransientError'])",
    )
    .await
    .expect("create retry policy");

    let bad_jitter = exec_ddl(
        &server,
        &mut session,
        "CREATE RETRY POLICY bad_rp (max_attempts = 3, jitter = 1.5)",
    )
    .await
    .expect_err("jitter beyond 1.0 refused");
    assert!(
        bad_jitter.contains("jitter must be between 0.0 and 1.0"),
        "unexpected error: {bad_jitter}"
    );

    let wrong_kind = exec_ddl(&server, &mut session, "DROP RETRY POLICY my_bh")
        .await
        .expect_err("dropping a bulkhead as a retry policy refused");
    assert!(
        wrong_kind.contains("is a bulkhead, not a retry policy"),
        "unexpected error: {wrong_kind}"
    );

    exec_ddl(&server, &mut session, "DROP BULKHEAD my_bh")
        .await
        .expect("drop bulkhead");
    exec_ddl(&server, &mut session, "DROP RETRY POLICY my_rp")
        .await
        .expect("drop retry policy");
    exec_ddl(&server, &mut session, "DROP BULKHEAD IF EXISTS my_bh")
        .await
        .expect("IF EXISTS tolerates the missing bulkhead");
    let missing = exec_ddl(&server, &mut session, "DROP BULKHEAD my_bh")
        .await
        .expect_err("dropping a dropped bulkhead refused");
    assert!(
        missing.contains("does not exist"),
        "unexpected error: {missing}"
    );
}

/// The callable resilience forms evaluate: an admitted bulkhead call
/// answers its operation, a retry answers, a fallback chain skips an
/// erroring operation, and cache aside serves the stored value without
/// re evaluating the fetch
#[tokio::test]
async fn test_resilience_callable_functions() {
    let (server, _schema, _tmp) = create_test_server().await;
    let mut session = new_session();

    exec_ddl(
        &server,
        &mut session,
        "CREATE BULKHEAD bh_call (max_concurrent = 4)",
    )
    .await
    .expect("create bulkhead");
    exec_ddl(
        &server,
        &mut session,
        "CREATE RETRY POLICY rp_call (max_attempts = 2, base_delay = '1ms', max_delay = '10ms')",
    )
    .await
    .expect("create retry policy");
    exec_ddl(&server, &mut session, "CREATE TABLE fb_t (v INT)")
        .await
        .expect("create table");
    exec_dml(&server, "INSERT INTO fb_t VALUES (10)").await;

    let rows = query_values(&server, "SELECT bulkhead_call('bh_call', v + 32) FROM fb_t").await;
    assert_eq!(
        as_i64(&rows[0][0]),
        42,
        "an admitted call runs the operation"
    );

    let rows = query_values(&server, "SELECT with_retry('rp_call', v * 4 + 2) FROM fb_t").await;
    assert_eq!(
        as_i64(&rows[0][0]),
        42,
        "a clean operation answers on the first try"
    );

    // The first operation divides by zero at evaluation time, so the
    // chain answers from the second
    let rows = query_values(
        &server,
        "SELECT fallback_chain(10 / (v - v), v + 90) FROM fb_t",
    )
    .await;
    assert_eq!(
        as_i64(&rows[0][0]),
        100,
        "the chain skips the failing operation"
    );

    // A fresh cache entry answers later calls without the fetch, so a
    // changed fetch expression cannot change the answer inside the ttl
    let rows = query_values(
        &server,
        "SELECT cache_aside('surface_probe_key', v + 1, '60s', '120s') FROM fb_t",
    )
    .await;
    assert_eq!(as_i64(&rows[0][0]), 11, "the miss evaluates the fetch");
    let rows = query_values(
        &server,
        "SELECT cache_aside('surface_probe_key', v + 500, '60s', '120s') FROM fb_t",
    )
    .await;
    assert_eq!(
        as_i64(&rows[0][0]),
        11,
        "the fresh hit serves the stored value, not the new fetch"
    );
}

// ---------------------------------------------------------------------------
// Task 9, expectation metrics
// ---------------------------------------------------------------------------

/// Metric expectations evaluate per statement and every evaluation lands
/// a row in zyron_sys.expectation.results with its verdict
#[tokio::test]
async fn test_expectation_metrics_record_results() {
    let (server, _schema, _tmp) = create_test_server().await;
    let mut session = new_session();

    exec_ddl(
        &server,
        &mut session,
        "CREATE TABLE exp_t (v INT, ts TIMESTAMP)",
    )
    .await
    .expect("create table");
    for ddl in [
        "ALTER TABLE exp_t ADD EXPECTATION exp_nr EXPECT NULL_RATE(v, 0.5) ON VIOLATION WARN",
        "ALTER TABLE exp_t ADD EXPECTATION exp_dr EXPECT DISTINCT_RATE(v, 0.9) ON VIOLATION WARN",
        "ALTER TABLE exp_t ADD EXPECTATION exp_fr EXPECT FRESHNESS(ts, '1h') ON VIOLATION WARN",
        "ALTER TABLE exp_t ADD EXPECTATION exp_rc_tight EXPECT ROW_COUNT_CHANGE(50.0) ON VIOLATION WARN",
        "ALTER TABLE exp_t ADD EXPECTATION exp_rc_loose EXPECT ROW_COUNT_CHANGE(500.0) ON VIOLATION WARN",
    ] {
        exec_ddl(&server, &mut session, ddl).await.expect(ddl);
    }

    // Two of three values NULL fails the null rate, the single distinct
    // non NULL value passes the distinct rate, the fresh timestamp keeps
    // freshness green, and three rows against an unanalyzed table trip the
    // tight row count change but not the loose one
    exec_dml(
        &server,
        "INSERT INTO exp_t (v, ts) VALUES (NULL, NOW()), (NULL, NOW()), (7, NOW())",
    )
    .await;
    // A batch whose newest timestamp is years old fails freshness
    exec_dml(
        &server,
        "INSERT INTO exp_t (v, ts) VALUES (8, TIMESTAMP '2020-01-01 00:00:00')",
    )
    .await;

    let (columns, rows) = read_system(&server, "SELECT * FROM zyron_sys.expectation.results")
        .await
        .expect("results view");
    let at = |name: &str| columns.iter().position(|c| c == name).expect(name);
    let verdicts = |exp_name: &str| -> Vec<(String, String)> {
        rows.iter()
            .filter(|r| r[at("expectation_name")] == exp_name)
            .map(|r| (r[at("passed")].clone(), r[at("details")].clone()))
            .collect()
    };

    let nr = verdicts("exp_nr");
    assert!(
        nr.iter()
            .any(|(passed, details)| passed == "false" && details.contains("\"violations\":3")),
        "the first insert fails the null rate over all three rows: {nr:?}"
    );
    assert!(
        nr.iter().any(|(passed, _)| passed == "true"),
        "the second insert has no nulls and passes: {nr:?}"
    );

    let dr = verdicts("exp_dr");
    assert!(
        dr.iter().all(|(passed, _)| passed == "true"),
        "every batch had fully distinct non null values: {dr:?}"
    );

    let fr = verdicts("exp_fr");
    assert!(
        fr.iter().any(|(passed, _)| passed == "true"),
        "the current timestamp batch is fresh: {fr:?}"
    );
    assert!(
        fr.iter().any(|(passed, _)| passed == "false"),
        "the stale batch fails freshness: {fr:?}"
    );

    let tight = verdicts("exp_rc_tight");
    assert!(
        tight.iter().any(|(passed, _)| passed == "false"),
        "three rows against an empty table exceed fifty percent: {tight:?}"
    );
    let loose = verdicts("exp_rc_loose");
    assert!(
        loose.iter().any(|(passed, _)| passed == "true"),
        "three hundred percent stays under five hundred: {loose:?}"
    );
}

// ---------------------------------------------------------------------------
// Task 10, DETECT_ANOMALIES
// ---------------------------------------------------------------------------

/// A clear outlier in an otherwise flat series is the one flagged row
#[tokio::test]
async fn test_detect_anomalies_flags_the_outlier() {
    let (server, _schema, _tmp) = create_test_server().await;
    let mut session = new_session();

    exec_ddl(
        &server,
        &mut session,
        "CREATE TABLE anom_t (v DOUBLE PRECISION)",
    )
    .await
    .expect("create table");
    exec_dml(
        &server,
        "INSERT INTO anom_t VALUES (10), (11), (9), (10), (10), (11), (9), (10), (100)",
    )
    .await;

    let rows = query_values(
        &server,
        "SELECT * FROM detect_anomalies('anom_t', 'v', method => 'zscore', threshold => 2.0)",
    )
    .await;
    assert_eq!(rows.len(), 9, "one verdict per row");
    for row in &rows {
        let idx = as_i64(&row[0]);
        let flagged = as_bool(&row[1]);
        let score = as_f64(&row[2]);
        if idx == 8 {
            assert!(flagged, "the 100 at index 8 is the anomaly");
            assert!(
                score > 2.0,
                "its z score exceeds the threshold, got {score}"
            );
        } else {
            assert!(!flagged, "index {idx} is ordinary, score {score}");
        }
    }

    let bad = query_error(
        &server,
        "SELECT * FROM detect_anomalies('anom_t', 'v', method => 'bogus')",
    )
    .await;
    assert!(
        bad.contains("bogus") || bad.contains("method"),
        "the unknown method is refused loudly: {bad}"
    );
}

// ---------------------------------------------------------------------------
// Task 11, vector math scalars
// ---------------------------------------------------------------------------

/// The vector space functions compute known values over JSON array
/// literals and refuse mismatched dimensions
#[tokio::test]
async fn test_vector_math_scalar_functions() {
    let (server, _schema, _tmp) = create_test_server().await;

    let rows = query_values(
        &server,
        "SELECT vector_dot('[1, 2, 3]', '[4, 5, 6]'), \
                vector_norm('[3, 4]'), \
                vector_norm(vector_normalize('[3, 4]')), \
                vector_dot(vector_cross('[1, 0, 0]', '[0, 1, 0]'), '[0, 0, 1]'), \
                vector_angle('[1, 0]', '[0, 1]')",
    )
    .await;
    assert_eq!(as_f64(&rows[0][0]), 32.0, "1*4 + 2*5 + 3*6");
    assert_eq!(as_f64(&rows[0][1]), 5.0, "the 3 4 5 triangle");
    assert!(
        (as_f64(&rows[0][2]) - 1.0).abs() < 1e-6,
        "a normalized vector has unit length"
    );
    assert!((as_f64(&rows[0][3]) - 1.0).abs() < 1e-6, "x cross y is z");
    assert!(
        (as_f64(&rows[0][4]) - std::f64::consts::FRAC_PI_2).abs() < 1e-9,
        "orthogonal vectors sit at a right angle"
    );

    let mismatch = query_error(&server, "SELECT vector_dot('[1, 2]', '[1]')").await;
    assert!(
        mismatch.contains("dimensions differ"),
        "unexpected error: {mismatch}"
    );
}

// ---------------------------------------------------------------------------
// Task 12, currency rates table and money conversion
// ---------------------------------------------------------------------------

/// A rate written through the system table converts money, the view reads
/// it back, and a delete empties both the view and the conversion path
#[tokio::test]
async fn test_currency_rates_write_read_convert_delete() {
    let (server, _schema, _tmp) = create_test_server().await;
    let mut session = new_session();

    exec_ddl(
        &server,
        &mut session,
        "INSERT INTO zyron_sys.cost.currency_rates VALUES ('USD', 'EUR', '2026-08-01', 0.5)",
    )
    .await
    .expect("insert rate");

    let (columns, rows) = read_system(&server, "SELECT * FROM zyron_sys.cost.currency_rates")
        .await
        .expect("rates view");
    assert_eq!(
        columns,
        vec!["from_currency", "to_currency", "rate_date", "rate"]
    );
    assert_eq!(
        rows,
        vec![vec![
            "USD".to_string(),
            "EUR".to_string(),
            "2026-08-01".to_string(),
            "0.5".to_string()
        ]]
    );

    let converted = query_values(
        &server,
        "SELECT money_currency_code(convert_currency(money_create(100.0, 'USD'), 'USD', 'EUR')), \
                money_format(convert_currency(money_create(100.0, 'USD'), 'USD', 'EUR'))",
    )
    .await;
    assert_eq!(as_str(&converted[0][0]), "EUR");
    let formatted = as_str(&converted[0][1]);
    assert!(
        formatted.contains("50.00"),
        "100 USD at 0.5 formats as 50.00 EUR, got {formatted}"
    );

    exec_ddl(
        &server,
        &mut session,
        "DELETE FROM zyron_sys.cost.currency_rates",
    )
    .await
    .expect("delete rates");
    let (_, rows) = read_system(&server, "SELECT * FROM zyron_sys.cost.currency_rates")
        .await
        .expect("rates view after delete");
    assert!(rows.is_empty(), "the delete cleared every rate");

    let refused = try_query_values(
        &server,
        "SELECT convert_currency(money_create(100.0, 'USD'), 'USD', 'EUR')",
    )
    .await
    .expect_err("conversion without a rate must fail")
    .to_string();
    assert!(
        refused.contains("rate"),
        "the error names the missing rates: {refused}"
    );
}

// ---------------------------------------------------------------------------
// Task 13, retention dashboard views
// ---------------------------------------------------------------------------

/// A TTL table with one fresh and one long expired row shapes all four
/// retention views
#[tokio::test]
async fn test_retention_dashboard_views() {
    let (server, _schema, _tmp) = create_test_server().await;
    let mut session = new_session();

    exec_ddl(
        &server,
        &mut session,
        "CREATE TABLE ret_events (id INT, created_at TIMESTAMP)",
    )
    .await
    .expect("create table");
    exec_ddl(
        &server,
        &mut session,
        "ALTER TABLE ret_events SET TTL 30 DAYS ON created_at",
    )
    .await
    .expect("set ttl");
    exec_dml(
        &server,
        "INSERT INTO ret_events VALUES (1, TIMESTAMP '2020-01-01 00:00:00'), (2, NOW())",
    )
    .await;

    let (columns, rows) = read_system(&server, "SELECT * FROM zyron_sys.retention.storage_by_age")
        .await
        .expect("storage_by_age");
    assert_eq!(
        columns,
        vec![
            "table_id",
            "table_name",
            "age_bucket",
            "byte_count",
            "row_count"
        ]
    );
    let at = |cols: &[String], name: &str| cols.iter().position(|c| c == name).expect(name);
    let bucket_rows: Vec<(String, String)> = rows
        .iter()
        .filter(|r| r[at(&columns, "table_name")] == "ret_events")
        .map(|r| {
            (
                r[at(&columns, "age_bucket")].clone(),
                r[at(&columns, "row_count")].clone(),
            )
        })
        .collect();
    assert!(
        bucket_rows.contains(&("0d-1d".to_string(), "1".to_string())),
        "the fresh row lands in the youngest bucket: {bucket_rows:?}"
    );
    assert!(
        bucket_rows.contains(&("90d+".to_string(), "1".to_string())),
        "the 2020 row lands in the oldest bucket: {bucket_rows:?}"
    );

    let (columns, rows) = read_system(
        &server,
        "SELECT * FROM zyron_sys.retention.upcoming_actions",
    )
    .await
    .expect("upcoming_actions");
    let ours: Vec<&Vec<String>> = rows
        .iter()
        .filter(|r| r[at(&columns, "table_name")] == "ret_events")
        .collect();
    assert_eq!(ours.len(), 1, "one policy row for the table");
    assert_eq!(ours[0][at(&columns, "policy_kind")], "ttl");
    assert_eq!(ours[0][at(&columns, "action")], "delete");
    assert_eq!(
        ours[0][at(&columns, "row_count_impact_estimate")],
        "1",
        "the expired 2020 row is the impact estimate"
    );

    let (columns, rows) = read_system(
        &server,
        "SELECT * FROM zyron_sys.retention.savings_estimate",
    )
    .await
    .expect("savings_estimate");
    let ours: Vec<&Vec<String>> = rows
        .iter()
        .filter(|r| r[at(&columns, "table_name")] == "ret_events")
        .collect();
    assert_eq!(ours.len(), 1);
    assert_eq!(ours[0][at(&columns, "if_purged_now_rows")], "1");
    let bytes: u64 = ours[0][at(&columns, "if_purged_now_bytes")]
        .parse()
        .expect("bytes parse");
    assert!(bytes > 0, "purging one row reclaims bytes");

    let (columns, rows) = read_system(
        &server,
        "SELECT * FROM zyron_sys.retention.compliance_summary",
    )
    .await
    .expect("compliance_summary");
    let ours: Vec<&Vec<String>> = rows
        .iter()
        .filter(|r| r[at(&columns, "table_name")] == "ret_events")
        .collect();
    assert_eq!(ours.len(), 1);
    assert_eq!(ours[0][at(&columns, "policy_name")], "ttl_delete");
    assert_eq!(ours[0][at(&columns, "ok")], "true");
    assert_eq!(
        ours[0][at(&columns, "message")],
        "no retention jobs recorded yet"
    );
}

// ---------------------------------------------------------------------------
// Task 14, PII masking
// ---------------------------------------------------------------------------

/// The five masking functions answer under their canonical zyron_sys
/// names with deterministic outputs
#[tokio::test]
async fn test_masking_functions() {
    let (server, _schema, _tmp) = create_test_server().await;

    let rows = query_values(
        &server,
        "SELECT zyron_sys.security.masking_email('user@example.com'), \
                zyron_sys.security.masking_ip('192.168.1.100', 24), \
                zyron_sys.security.masking_phone('+1-555-867-5309', true), \
                zyron_sys.security.masking_ssn('123-45-6789'), \
                zyron_sys.security.masking_name('John Smith')",
    )
    .await;

    let email = as_str(&rows[0][0]);
    assert_eq!(
        email,
        zyron_types::masking::masking_email("user@example.com").expect("mask"),
        "the SQL surface matches the library function"
    );
    assert!(email.ends_with("@example.com"), "the domain survives");
    assert!(!email.starts_with("user@"), "the local part is hashed away");

    assert_eq!(as_str(&rows[0][1]), "192.168.1.0");
    assert_eq!(as_str(&rows[0][2]), "+1-***-***-****");

    let ssn = as_str(&rows[0][3]);
    assert!(ssn.ends_with("-6789"), "the last four digits survive");
    assert!(!ssn.contains("123"), "the leading digits are gone");

    assert_eq!(as_str(&rows[0][4]), "J.S.");

    // Determinism keeps masked columns joinable
    let again = query_values(
        &server,
        "SELECT zyron_sys.security.masking_email('user@example.com')",
    )
    .await;
    assert_eq!(as_str(&again[0][0]), email);

    let invalid = query_error(
        &server,
        "SELECT zyron_sys.security.masking_email('not-an-email')",
    )
    .await;
    assert!(
        invalid.contains("not-an-email") || invalid.to_lowercase().contains("email"),
        "an unmaskable value errors loudly: {invalid}"
    );
}

// ---------------------------------------------------------------------------
// Task 15, KMEANS
// ---------------------------------------------------------------------------

/// Two well separated point clouds split cleanly into two clusters, and
/// the fitted centroids sit inside their groups
#[tokio::test]
async fn test_kmeans_separates_two_groups() {
    let (server, _schema, _tmp) = create_test_server().await;
    let mut session = new_session();

    exec_ddl(
        &server,
        &mut session,
        "CREATE TABLE km_t (x DOUBLE PRECISION, y DOUBLE PRECISION)",
    )
    .await
    .expect("create table");
    exec_dml(
        &server,
        "INSERT INTO km_t VALUES (0, 0), (1, 0), (0, 1), (100, 100), (101, 100), (100, 101)",
    )
    .await;

    let rows = query_values(
        &server,
        "SELECT * FROM kmeans_cluster('km_t', 'x', 'y', k => 2, seed => 7)",
    )
    .await;
    assert_eq!(rows.len(), 6, "one assignment per row");
    let cluster_of: Vec<i64> = rows.iter().map(|r| as_i64(&r[1])).collect();
    assert_eq!(cluster_of[0], cluster_of[1]);
    assert_eq!(cluster_of[0], cluster_of[2]);
    assert_eq!(cluster_of[3], cluster_of[4]);
    assert_eq!(cluster_of[3], cluster_of[5]);
    assert_ne!(
        cluster_of[0], cluster_of[3],
        "the two clouds land in different clusters"
    );
    for row in &rows {
        let distance = as_f64(&row[2]);
        assert!(
            distance < 2.0,
            "every point is close to its own centroid, got {distance}"
        );
    }

    let centroids = query_values(
        &server,
        "SELECT * FROM kmeans_centroids('km_t', 'x', 'y', k => 2, seed => 7)",
    )
    .await;
    assert_eq!(centroids.len(), 4, "two clusters times two features");
    let mut xs: Vec<f64> = centroids
        .iter()
        .filter(|r| as_str(&r[1]) == "x")
        .map(|r| as_f64(&r[2]))
        .collect();
    xs.sort_by(|a, b| a.partial_cmp(b).expect("finite"));
    assert!(
        (xs[0] - 1.0 / 3.0).abs() < 0.1,
        "the low cluster's x centroid is a third, got {}",
        xs[0]
    );
    assert!(
        (xs[1] - 100.0 - 1.0 / 3.0).abs() < 0.1,
        "the high cluster's x centroid is a hundred and a third, got {}",
        xs[1]
    );
}

// ---------------------------------------------------------------------------
// Task 16, causal inference
// ---------------------------------------------------------------------------

/// With treated and control rows paired on identical features, the
/// counterfactual effect equals the constructed treatment lift, and
/// propensity matching pairs every treated row
#[tokio::test]
async fn test_causal_inference_recovers_the_constructed_effect() {
    let (server, _schema, _tmp) = create_test_server().await;
    let mut session = new_session();

    exec_ddl(
        &server,
        &mut session,
        "CREATE TABLE cf_t (outcome DOUBLE PRECISION, treated INT, x DOUBLE PRECISION)",
    )
    .await
    .expect("create table");
    // Each treated row has a control twin at the same x with outcome
    // exactly ten lower, so every nearest neighbor effect is ten
    exec_dml(
        &server,
        "INSERT INTO cf_t VALUES \
         (10, 0, 1), (20, 0, 2), (30, 0, 3), \
         (20, 1, 1), (30, 1, 2), (40, 1, 3)",
    )
    .await;

    let rows = query_values(
        &server,
        "SELECT * FROM counterfactual('cf_t', 'outcome', 'treated', 'x')",
    )
    .await;
    assert_eq!(rows.len(), 6, "one counterfactual per row");
    for row in &rows {
        let effect = as_f64(&row[3]);
        assert!(
            (effect - 10.0).abs() < 1e-9,
            "the constructed lift is ten, got {effect}"
        );
    }

    let pairs = query_values(
        &server,
        "SELECT * FROM propensity_match('cf_t', 'treated', 'x', caliper => 1.0)",
    )
    .await;
    assert_eq!(pairs.len(), 3, "every treated row finds a control match");
    for pair in &pairs {
        let diff = as_f64(&pair[2]);
        assert!(
            diff < 0.5,
            "identical covariate distributions keep score differences small, got {diff}"
        );
    }
}

// ---------------------------------------------------------------------------
// Task 17, feature engineering
// ---------------------------------------------------------------------------

/// ONE_HOT_ENCODE and TFIDF return exact long form rows for a tiny input
#[tokio::test]
async fn test_feature_encoding_exact_rows() {
    let (server, _schema, _tmp) = create_test_server().await;
    let mut session = new_session();

    exec_ddl(&server, &mut session, "CREATE TABLE cat_t (c TEXT)")
        .await
        .expect("create table");
    exec_dml(&server, "INSERT INTO cat_t VALUES ('red'), ('blue')").await;

    let rows = query_values(&server, "SELECT * FROM one_hot_encode('cat_t', 'c')").await;
    let got: Vec<(i64, String, i64)> = rows
        .iter()
        .map(|r| (as_i64(&r[0]), as_str(&r[1]), as_i64(&r[2])))
        .collect();
    assert_eq!(
        got,
        vec![
            (0, "blue".to_string(), 0),
            (0, "red".to_string(), 1),
            (1, "blue".to_string(), 1),
            (1, "red".to_string(), 0),
        ],
        "one indicator row per input row and sorted level"
    );

    exec_ddl(&server, &mut session, "CREATE TABLE doc_t (body TEXT)")
        .await
        .expect("create table");
    exec_dml(
        &server,
        "INSERT INTO doc_t VALUES ('apple banana'), ('apple cherry')",
    )
    .await;

    let rows = query_values(&server, "SELECT * FROM tfidf('doc_t', 'body')").await;
    let got: Vec<(i64, String, f64)> = rows
        .iter()
        .map(|r| (as_i64(&r[0]), as_str(&r[1]), as_f64(&r[2])))
        .collect();
    // Smoothed idf: a term in both documents scores tf * 1.0, a term in
    // one document scores tf * (ln(3/2) + 1)
    let rare = 0.5 * ((3.0f64 / 2.0).ln() + 1.0);
    let expected = vec![
        (0i64, "apple".to_string(), 0.5f64),
        (0, "banana".to_string(), rare),
        (1, "apple".to_string(), 0.5),
        (1, "cherry".to_string(), rare),
    ];
    assert_eq!(got.len(), expected.len(), "rows: {got:?}");
    for ((gd, gt, gs), (ed, et, es)) in got.iter().zip(expected.iter()) {
        assert_eq!(gd, ed);
        assert_eq!(gt, et);
        assert!(
            (gs - es).abs() < 1e-9,
            "term {gt} scored {gs}, expected {es}"
        );
    }
}

// ---------------------------------------------------------------------------
// Task 18, statistical tests
// ---------------------------------------------------------------------------

/// Identical samples do not drift and clearly shifted ones do, with the
/// p values on the right sides, and Shapiro-Wilk rates a symmetric sample
/// as plausibly normal
#[tokio::test]
async fn test_statistical_tests_p_value_ranges() {
    let (server, _schema, _tmp) = create_test_server().await;
    let mut session = new_session();

    for name in ["drift_a", "drift_b", "drift_c"] {
        exec_ddl(
            &server,
            &mut session,
            &format!("CREATE TABLE {name} (v DOUBLE PRECISION)"),
        )
        .await
        .expect("create table");
    }
    let base: Vec<String> = (1..=30).map(|i| format!("({i})")).collect();
    exec_dml(
        &server,
        &format!("INSERT INTO drift_a VALUES {}", base.join(", ")),
    )
    .await;
    exec_dml(
        &server,
        &format!("INSERT INTO drift_b VALUES {}", base.join(", ")),
    )
    .await;
    let shifted: Vec<String> = (101..=130).map(|i| format!("({i})")).collect();
    exec_dml(
        &server,
        &format!("INSERT INTO drift_c VALUES {}", shifted.join(", ")),
    )
    .await;

    let same = query_values(
        &server,
        "SELECT * FROM detect_drift('drift_a', 'v', 'drift_b', 'v', method => 'ks_test')",
    )
    .await;
    assert_eq!(as_str(&same[0][0]), "ks_test");
    assert_eq!(as_f64(&same[0][1]), 0.0, "identical samples have zero D");
    assert!(
        as_f64(&same[0][2]) > 0.9,
        "identical samples give a high p value, got {}",
        as_f64(&same[0][2])
    );
    assert!(!as_bool(&same[0][3]), "identical samples do not drift");

    let apart = query_values(
        &server,
        "SELECT * FROM detect_drift('drift_a', 'v', 'drift_c', 'v', method => 'ks_test')",
    )
    .await;
    assert_eq!(
        as_f64(&apart[0][1]),
        1.0,
        "disjoint ranges have the maximal D"
    );
    assert!(
        as_f64(&apart[0][2]) < 0.01,
        "disjoint samples give a tiny p value, got {}",
        as_f64(&apart[0][2])
    );
    assert!(as_bool(&apart[0][3]), "disjoint samples drift");

    exec_ddl(
        &server,
        &mut session,
        "CREATE TABLE norm_t (v DOUBLE PRECISION)",
    )
    .await
    .expect("create table");
    exec_dml(
        &server,
        "INSERT INTO norm_t VALUES (-2.1), (-1.6), (-1.2), (-0.9), (-0.7), (-0.5), (-0.3), (-0.1), \
         (0.1), (0.3), (0.5), (0.7), (0.9), (1.2), (1.6), (2.1)",
    )
    .await;
    let sw = query_values(&server, "SELECT * FROM shapiro_wilk('norm_t', 'v')").await;
    let w = as_f64(&sw[0][0]);
    let p = as_f64(&sw[0][1]);
    assert!(
        w > 0.9 && w <= 1.0,
        "a symmetric bell shaped sample has W near one, got {w}"
    );
    assert!(
        p > 0.05 && p <= 1.0,
        "normality is not rejected for a symmetric sample, got p {p}"
    );
}

// ---------------------------------------------------------------------------
// Task 19, semver and bloom estimation
// ---------------------------------------------------------------------------

/// The semver family parses, formats, compares, checks constraints, and
/// bumps versions, and a bloom filter built over eight distinct values
/// estimates close to eight
#[tokio::test]
async fn test_semver_and_bloom_estimate() {
    let (server, _schema, _tmp) = create_test_server().await;

    let rows = query_values(
        &server,
        "SELECT semver_compare('1.2.0', '1.10.0'), \
                semver_compare('2.0.0', '2.0.0'), \
                semver_compare('1.10.0', '1.2.0'), \
                semver_format(semver_parse('4.5.6')), \
                semver_format(semver_parse('1.2.3-beta.1')), \
                semver_satisfies(semver_parse('1.5.0'), '^1.0.0'), \
                semver_satisfies(semver_parse('1.5.0'), '^2.0.0'), \
                semver_major(semver_parse('4.5.6')), \
                semver_minor(semver_parse('4.5.6')), \
                semver_patch(semver_parse('4.5.6')), \
                semver_format(semver_increment_minor(semver_parse('1.2.3'))), \
                semver_prerelease('1.2.3-rc.1')",
    )
    .await;
    assert_eq!(as_i64(&rows[0][0]), -1, "1.2.0 is below 1.10.0 numerically");
    assert_eq!(as_i64(&rows[0][1]), 0);
    assert_eq!(as_i64(&rows[0][2]), 1);
    assert_eq!(
        as_str(&rows[0][3]),
        "4.5.6",
        "a release version round trips"
    );
    // The packed form keeps only a prerelease flag, so formatting renders
    // the generic pre marker and the identifier text comes from
    // semver_prerelease over the original string
    assert_eq!(as_str(&rows[0][4]), "1.2.3-pre");
    assert!(as_bool(&rows[0][5]), "1.5.0 satisfies caret 1.0.0");
    assert!(!as_bool(&rows[0][6]), "1.5.0 does not satisfy caret 2.0.0");
    assert_eq!(as_i64(&rows[0][7]), 4);
    assert_eq!(as_i64(&rows[0][8]), 5);
    assert_eq!(as_i64(&rows[0][9]), 6);
    assert_eq!(
        as_str(&rows[0][10]),
        "1.3.0",
        "a minor bump clears the patch"
    );
    assert_eq!(as_str(&rows[0][11]), "rc.1");

    // Eight distinct values through nested adds, then the fill based
    // cardinality estimate
    let filter = "bloom_add(bloom_add(bloom_add(bloom_add(bloom_add(bloom_add(bloom_add(bloom_add(\
                  bloom_create(100, 0.01), 'v1'), 'v2'), 'v3'), 'v4'), 'v5'), 'v6'), 'v7'), 'v8')";
    let rows = query_values(
        &server,
        &format!(
            "SELECT bloom_filter_estimate_count({filter}), \
                    bloom_contains({filter}, 'v3'), \
                    bloom_contains({filter}, 'absent_value'), \
                    bloom_false_positive_rate({filter})"
        ),
    )
    .await;
    let estimate = as_i64(&rows[0][0]);
    assert!(
        (6..=10).contains(&estimate),
        "eight distinct values estimate within tolerance, got {estimate}"
    );
    assert!(as_bool(&rows[0][1]), "an added value is contained");
    assert!(
        !as_bool(&rows[0][2]),
        "an absent value is not reported at this sizing"
    );
    let fpr = as_f64(&rows[0][3]);
    assert!(
        fpr >= 0.0 && fpr < 0.05,
        "a lightly filled filter keeps a low false positive rate, got {fpr}"
    );
}
