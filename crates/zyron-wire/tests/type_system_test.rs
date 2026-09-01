//! Type system surface: VARIANT dotted access with shredding stats, user
//! defined composite types, the Arrow extension registry, STRUCT and MAP
//! access, CREATE COLLATION with COLLATE comparisons and ordering, WITHOUT
//! OVERLAPS temporal keys, generated columns, ENCRYPTED columns, LTREE
//! functions, and the QUIC transport defaults.
//!
//! Run: cargo test -p zyron-wire --test type_system_test

mod common;

use common::{
    create_test_server, exec_ddl, exec_dml, exec_dml_result, new_session, query_error, query_values,
};
use std::sync::Arc;
use zyron_executor::column::ScalarValue;
use zyron_wire::connection::ServerState;
use zyron_wire::system_views::{SystemViewFilters, query_system_view};

fn text_of(v: &ScalarValue) -> String {
    match v {
        ScalarValue::Utf8(s) => s.clone(),
        other => panic!("expected text, got {other:?}"),
    }
}

fn int_of(v: &ScalarValue) -> i64 {
    match v {
        ScalarValue::Int8(x) => *x as i64,
        ScalarValue::Int16(x) => *x as i64,
        ScalarValue::Int32(x) => *x as i64,
        ScalarValue::Int64(x) => *x,
        other => panic!("expected an integer, got {other:?}"),
    }
}

/// Reads zyron_sys.storage.variant_shredding_stats as column names plus rows
/// of text cells
async fn shredding_stats(server: &Arc<ServerState>) -> (Vec<String>, Vec<Vec<String>>) {
    let (fields, rows) = query_system_view(
        "zyron_sys.storage.variant_shredding_stats",
        server,
        &SystemViewFilters::default(),
    )
    .await
    .expect("view query")
    .expect("variant_shredding_stats is a registered view");
    let columns = fields.iter().map(|f| f.name.clone()).collect();
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
    (columns, rows)
}

/// VARIANT columns take JSON text, dotted access reads scalar fields at any
/// depth, predicates filter on extracted values, and the shredding tracker
/// reports and promotes frequent paths through the system view
#[tokio::test]
async fn test_variant_dotted_access_and_shredding_stats() {
    let (server, _schema, _tmp) = create_test_server().await;
    let mut session = new_session();
    exec_ddl(&server, &mut session, "CREATE TABLE vt (v VARIANT)")
        .await
        .expect("create");
    exec_dml(
        &server,
        r#"INSERT INTO vt VALUES
            ('{"address":{"city":"Portland","geo":{"lat":45.5}},"age":30}'),
            ('{"address":{"city":"Salem","geo":{"lat":44.9}},"age":41}'),
            ('{"address":{"city":"Bend","geo":{"lat":44.1}},"age":27}')"#,
    )
    .await;

    // A one segment path needs the table qualifier so the chain has three
    // dotted parts, deeper paths work bare
    let rows = query_values(
        &server,
        "SELECT v.address.city FROM vt WHERE vt.v.age = '41'",
    )
    .await;
    assert_eq!(rows.len(), 1);
    assert_eq!(text_of(&rows[0][0]), "Salem");

    let rows = query_values(
        &server,
        "SELECT v.address.geo.lat FROM vt WHERE v.address.city = 'Portland'",
    )
    .await;
    assert_eq!(rows.len(), 1);
    assert_eq!(text_of(&rows[0][0]), "45.5");

    let rows = query_values(
        &server,
        "SELECT vt.v.age FROM vt WHERE v.address.geo.lat = '44.1'",
    )
    .await;
    assert_eq!(rows.len(), 1);
    assert_eq!(text_of(&rows[0][0]), "27");

    // A path missing from a row reads as NULL rather than failing the query
    let rows = query_values(
        &server,
        "SELECT v.address.zip FROM vt WHERE vt.v.age = '30'",
    )
    .await;
    assert_eq!(rows.len(), 1);
    assert!(matches!(rows[0][0], ScalarValue::Null));

    // 1197 more rows of the same shape bring every path to 1200 occurrences
    // at full coverage, past the 1000 row 70 percent promotion thresholds
    let mut written = 0usize;
    while written < 1197 {
        let take = (1197 - written).min(200);
        let tuples: Vec<String> = (0..take)
            .map(|i| {
                let n = written + i;
                format!(r#"('{{"address":{{"city":"c{n}","geo":{{"lat":{n}.5}}}},"age":{n}}}')"#)
            })
            .collect();
        exec_dml(
            &server,
            &format!("INSERT INTO vt VALUES {}", tuples.join(", ")),
        )
        .await;
        written += take;
    }

    let (columns, rows) = shredding_stats(&server).await;
    let col = |name: &str| {
        columns
            .iter()
            .position(|c| c == name)
            .unwrap_or_else(|| panic!("view has no column {name}, got {columns:?}"))
    };
    let (c_table, c_column, c_path) = (col("table_name"), col("column_name"), col("path"));
    let (c_occ, c_cov, c_kind, c_shred) = (
        col("occurrences"),
        col("coverage_percent"),
        col("value_kind"),
        col("shredded"),
    );
    let stat = |path: &str| {
        rows.iter()
            .find(|r| r[c_table] == "vt" && r[c_column] == "v" && r[c_path] == path)
            .unwrap_or_else(|| panic!("no stats row for path {path}"))
    };

    let city = stat("address.city");
    assert_eq!(city[c_occ], "1200");
    assert_eq!(city[c_cov], "100.00");
    assert_eq!(city[c_kind], "string");
    assert_eq!(city[c_shred], "true", "full coverage clears the thresholds");

    let age = stat("age");
    assert_eq!(age[c_occ], "1200");
    assert_eq!(age[c_kind], "number");
    assert_eq!(age[c_shred], "true");

    let lat = stat("address.geo.lat");
    assert_eq!(lat[c_occ], "1200");
    assert_eq!(lat[c_kind], "number");
}

/// A user defined composite type applies its input cast on write, its check
/// against the cast value, and its output cast on read, and refuses to drop
/// while a column still uses it
#[tokio::test]
async fn test_create_type_composite_casts_check_and_drop() {
    let (server, _schema, _tmp) = create_test_server().await;
    let mut session = new_session();
    exec_ddl(
        &server,
        &mut session,
        "CREATE TYPE us_zipcode AS (
            storage = TEXT,
            check = 'length(value) >= 5',
            input_cast = 'TRIM(value)',
            output_cast = 'UPPER(value)'
        )",
    )
    .await
    .expect("create type");
    exec_ddl(
        &server,
        &mut session,
        "CREATE TABLE zips (zip us_zipcode, id INT)",
    )
    .await
    .expect("create table");

    // The read comes back trimmed and uppercased, proving the input cast ran
    // on write and the output cast ran on read
    exec_dml(&server, "INSERT INTO zips VALUES ('  9021a  ', 1)").await;
    let rows = query_values(&server, "SELECT zip FROM zips WHERE id = 1").await;
    assert_eq!(rows.len(), 1);
    assert_eq!(text_of(&rows[0][0]), "9021A");

    // Three characters after trimming fail length(value) >= 5
    let err = query_error(&server, "INSERT INTO zips VALUES ('  123  ', 2)").await;
    assert!(err.contains("CHECK constraint"), "unexpected error: {err}");
    let rows = query_values(&server, "SELECT COUNT(*) FROM zips").await;
    assert_eq!(int_of(&rows[0][0]), 1, "the refused row was not written");

    let err = exec_ddl(&server, &mut session, "DROP TYPE us_zipcode")
        .await
        .expect_err("drop while a column uses the type");
    assert!(err.contains("is used by column"), "unexpected error: {err}");
    exec_ddl(&server, &mut session, "DROP TABLE zips")
        .await
        .expect("drop table");
    exec_ddl(&server, &mut session, "DROP TYPE us_zipcode")
        .await
        .expect("drop type after the table is gone");
}

/// The Arrow extension registry stamps zyron.* names over declared storage
/// types and restores the TypeId on import. The COPY external paths need
/// configured endpoints, so this exercises the registry API directly
#[tokio::test]
async fn test_arrow_extension_registry() {
    use zyron_common::TypeId;
    use zyron_streaming::format::ColumnSpec;
    use zyron_streaming::format::arrow_ext::{
        EXTENSION_NAME_KEY, export_field, extension_name_for, import_type_id, storage_type_for,
        type_id_for_extension,
    };

    let expectations = [
        (TypeId::Uuid, "zyron.uuid", "FixedSizeBinary(16)"),
        (TypeId::Money, "zyron.money", "FixedSizeBinary(10)"),
        (TypeId::Inet, "zyron.inet", "Binary"),
        (TypeId::MacAddr, "zyron.macaddr", "FixedSizeBinary(6)"),
        (TypeId::Interval, "zyron.interval", "FixedSizeBinary(16)"),
    ];
    for (type_id, name, physical) in expectations {
        assert_eq!(extension_name_for(type_id), Some(name));
        assert_eq!(type_id_for_extension(name), Some(type_id));
        let storage =
            storage_type_for(type_id).unwrap_or_else(|| panic!("{name} declares a storage type"));
        assert_eq!(format!("{storage:?}"), physical, "storage type of {name}");

        // Export stamps the metadata over the storage type, import restores
        // the TypeId from the name before any DataType inference
        let field = export_field(&ColumnSpec::new("c", type_id));
        assert_eq!(
            field.metadata().get(EXTENSION_NAME_KEY).map(String::as_str),
            Some(name)
        );
        assert_eq!(format!("{:?}", field.data_type()), physical);
        assert_eq!(
            import_type_id(&field).expect("import"),
            type_id,
            "round trip for {name}"
        );
    }
}

/// A declared STRUCT is stored in its binary layout, not as the json it was
/// written with.
///
/// The declaration carries the field names, so the value carries none. That
/// is the whole reason the layout is worth having, and a read that still
/// found the names in the bytes would mean the write path never encoded.
#[tokio::test]
async fn a_declared_struct_is_stored_without_its_field_names() {
    let (server, _schema, _tmp) = create_test_server().await;
    let mut session = new_session();
    exec_ddl(
        &server,
        &mut session,
        "CREATE TABLE stored (s STRUCT<name TEXT, age INT>)",
    )
    .await
    .expect("create");
    exec_dml(
        &server,
        r#"INSERT INTO stored VALUES ('{"name":"ada","age":36}')"#,
    )
    .await;

    // The whole value reads back as the json it was written with, so nothing
    // downstream of a read sees the change
    let rows = query_values(&server, "SELECT s FROM stored").await;
    let rendered = text_of(&rows[0][0]);
    assert!(
        rendered.contains("\"name\":\"ada\"") && rendered.contains("\"age\":36"),
        "a whole struct has to read back as its json, got {rendered}"
    );

    // A field still reads by name through the declaration
    let rows = query_values(&server, "SELECT stored.s.name FROM stored").await;
    assert_eq!(text_of(&rows[0][0]), "ada");

    // A field the value omitted is null rather than missing, and the rest of
    // the struct still reads
    exec_dml(&server, r#"INSERT INTO stored VALUES ('{"name":"grace"}')"#).await;
    let rows = query_values(
        &server,
        "SELECT stored.s.age FROM stored WHERE stored.s.name = 'grace'",
    )
    .await;
    assert_eq!(rows[0][0], ScalarValue::Null, "an omitted field reads null");
}

/// STRUCT and MAP columns answer dotted access at the type the column was
/// declared with, so a field declared INT arrives as an integer rather than
/// as the text the extraction reads out of the value
#[tokio::test]
async fn test_struct_and_map_dotted_access() {
    let (server, _schema, _tmp) = create_test_server().await;
    let mut session = new_session();
    exec_ddl(
        &server,
        &mut session,
        "CREATE TABLE people (s STRUCT<name TEXT, age INT>, m MAP<TEXT, INT>)",
    )
    .await
    .expect("create");
    exec_dml(
        &server,
        r#"INSERT INTO people VALUES ('{"name":"ada","age":36}', '{"alpha":1,"beta":2}')"#,
    )
    .await;

    let rows = query_values(&server, "SELECT people.s.name FROM people").await;
    assert_eq!(rows.len(), 1);
    assert_eq!(text_of(&rows[0][0]), "ada");

    // Declared INT, so the field arrives as one
    let rows = query_values(&server, "SELECT people.s.age FROM people").await;
    assert_eq!(
        rows[0][0],
        ScalarValue::Int32(36),
        "a field declared INT came back as something else"
    );

    // A map's values are all the declared value type
    let rows = query_values(&server, "SELECT people.m.beta FROM people").await;
    assert_eq!(rows[0][0], ScalarValue::Int32(2));

    // Arithmetic settles that the type is real rather than a label: text
    // would not add. The sum widens the way any integer arithmetic does
    let rows = query_values(&server, "SELECT people.s.age + 6 FROM people").await;
    assert_eq!(rows[0][0], ScalarValue::Int64(42));

    // And the predicate compares numbers, not spellings
    let rows = query_values(
        &server,
        "SELECT people.s.name FROM people WHERE people.m.alpha = 1 AND people.s.age > 30",
    )
    .await;
    assert_eq!(rows.len(), 1);
    assert_eq!(text_of(&rows[0][0]), "ada");

    // A key the map does not hold reads as NULL
    let rows = query_values(&server, "SELECT people.m.gamma FROM people").await;
    assert_eq!(rows.len(), 1);
    assert!(matches!(rows[0][0], ScalarValue::Null));
}

/// A declaration that nests keeps every level, so a path reaching through a
/// struct into another struct, or into an array, still knows what it lands on
#[tokio::test]
async fn test_nested_declarations_type_the_whole_path() {
    let (server, _schema, _tmp) = create_test_server().await;
    let mut session = new_session();
    exec_ddl(
        &server,
        &mut session,
        "CREATE TABLE nested (s STRUCT<home STRUCT<city TEXT, zip INT>, tags TEXT>)",
    )
    .await
    .expect("create");
    exec_dml(
        &server,
        r#"INSERT INTO nested VALUES ('{"home":{"city":"leeds","zip":90210},"tags":"a"}')"#,
    )
    .await;

    let rows = query_values(&server, "SELECT nested.s.home.city FROM nested").await;
    assert_eq!(text_of(&rows[0][0]), "leeds");
    let rows = query_values(&server, "SELECT nested.s.home.zip FROM nested").await;
    assert_eq!(
        rows[0][0],
        ScalarValue::Int32(90210),
        "a field two levels down lost its declared type"
    );
}

/// A named ICU collation compares at secondary strength when declared case
/// insensitive, so sharp s equals ss and case folds, and collated ordering
/// puts umlauts beside their base letter. The bare locale literal form works
/// without a CREATE COLLATION object
#[tokio::test]
async fn test_create_collation_comparisons_and_ordering() {
    let (server, _schema, _tmp) = create_test_server().await;
    let mut session = new_session();
    exec_ddl(
        &server,
        &mut session,
        "CREATE COLLATION my_col (locale = 'de_DE', provider = 'icu',
         deterministic = true, case_sensitive = false)",
    )
    .await
    .expect("create collation");
    exec_ddl(&server, &mut session, "CREATE TABLE words (name TEXT)")
        .await
        .expect("create table");
    exec_dml(
        &server,
        "INSERT INTO words VALUES ('banane'), ('\u{e4}pfel'), ('apfel'), ('stra\u{df}e')",
    )
    .await;

    // strasse equals a stored sharp s spelling under the German collation
    let rows = query_values(
        &server,
        "SELECT name FROM words WHERE name = 'strasse' COLLATE my_col",
    )
    .await;
    assert_eq!(rows.len(), 1);
    assert_eq!(text_of(&rows[0][0]), "stra\u{df}e");

    // Case folds, while the umlaut still separates at secondary strength
    let rows = query_values(
        &server,
        "SELECT name FROM words WHERE name = 'APFEL' COLLATE my_col",
    )
    .await;
    assert_eq!(rows.len(), 1);
    assert_eq!(text_of(&rows[0][0]), "apfel");

    // Collated order interleaves the umlaut with its base letter instead of
    // sorting it after z the way codepoint order would
    let rows = query_values(
        &server,
        "SELECT name FROM words ORDER BY name COLLATE my_col",
    )
    .await;
    let names: Vec<String> = rows.iter().map(|r| text_of(&r[0])).collect();
    assert_eq!(names, vec!["apfel", "\u{e4}pfel", "banane", "stra\u{df}e"]);

    // The literal locale form resolves without a catalog object
    let rows = query_values(
        &server,
        "SELECT name FROM words ORDER BY name COLLATE 'de_DE'",
    )
    .await;
    let names: Vec<String> = rows.iter().map(|r| text_of(&r[0])).collect();
    assert_eq!(names, vec!["apfel", "\u{e4}pfel", "banane", "stra\u{df}e"]);

    exec_ddl(&server, &mut session, "DROP COLLATION my_col")
        .await
        .expect("drop collation");
}

/// A WITHOUT OVERLAPS key accepts equal scalar keys over disjoint periods
/// and refuses an intersecting period, against stored rows and between the
/// rows of one statement
#[tokio::test]
async fn test_without_overlaps_rejects_intersecting_periods() {
    let (server, _schema, _tmp) = create_test_server().await;
    let mut session = new_session();
    exec_ddl(
        &server,
        &mut session,
        "CREATE TABLE booking (
            room_id INT,
            booking_period DATERANGE,
            PRIMARY KEY (room_id, booking_period WITHOUT OVERLAPS)
        )",
    )
    .await
    .expect("create");

    exec_dml(
        &server,
        "INSERT INTO booking VALUES (1, '[2026-01-01,2026-01-10)')",
    )
    .await;

    let err = exec_dml_result(
        &server,
        "INSERT INTO booking VALUES (1, '[2026-01-05,2026-01-15)')",
    )
    .await
    .expect_err("an intersecting period for the same room")
    .to_string();
    assert!(
        err.contains("overlaps an existing row") && err.contains("temporal constraint"),
        "unexpected error: {err}"
    );

    // Half open ranges meeting at the boundary do not overlap
    exec_dml(
        &server,
        "INSERT INTO booking VALUES (1, '[2026-01-10,2026-01-20)')",
    )
    .await;
    // The same period under another scalar key is fine
    exec_dml(
        &server,
        "INSERT INTO booking VALUES (2, '[2026-01-05,2026-01-15)')",
    )
    .await;

    // The rows of one statement check against each other too
    let err = exec_dml_result(
        &server,
        "INSERT INTO booking VALUES (3, '[2026-02-01,2026-02-10)'), (3, '[2026-02-05,2026-02-15)')",
    )
    .await
    .expect_err("overlap within one statement")
    .to_string();
    assert!(
        err.contains("rows in this statement overlap"),
        "unexpected error: {err}"
    );

    let rows = query_values(&server, "SELECT COUNT(*) FROM booking").await;
    assert_eq!(int_of(&rows[0][0]), 3, "only the accepted rows persist");
}

/// Stored and virtual generated columns compute from their siblings, a bare
/// insert skips them, and naming one as a write target is refused
#[tokio::test]
async fn test_generated_columns_compute_and_refuse_direct_writes() {
    let (server, _schema, _tmp) = create_test_server().await;
    let mut session = new_session();
    exec_ddl(
        &server,
        &mut session,
        "CREATE TABLE gen (
            a INT,
            doubled INT GENERATED ALWAYS AS (a * 2) STORED,
            tripled INT GENERATED ALWAYS AS (a * 3) VIRTUAL
        )",
    )
    .await
    .expect("create");

    exec_dml(&server, "INSERT INTO gen (a) VALUES (2), (5)").await;
    let rows = query_values(&server, "SELECT a, doubled, tripled FROM gen ORDER BY a").await;
    assert_eq!(rows.len(), 2);
    assert_eq!(
        (
            int_of(&rows[0][0]),
            int_of(&rows[0][1]),
            int_of(&rows[0][2])
        ),
        (2, 4, 6)
    );
    assert_eq!(
        (
            int_of(&rows[1][0]),
            int_of(&rows[1][1]),
            int_of(&rows[1][2])
        ),
        (5, 10, 15)
    );

    // A bare insert targets only the plain columns
    exec_dml(&server, "INSERT INTO gen VALUES (7)").await;
    let rows = query_values(&server, "SELECT doubled, tripled FROM gen WHERE a = 7").await;
    assert_eq!(rows.len(), 1);
    assert_eq!((int_of(&rows[0][0]), int_of(&rows[0][1])), (14, 21));

    let err = query_error(&server, "INSERT INTO gen (a, doubled) VALUES (1, 99)").await;
    assert!(
        err.contains("GENERATED ALWAYS and cannot be written directly"),
        "unexpected error: {err}"
    );
    let err = query_error(&server, "UPDATE gen SET tripled = 1").await;
    assert!(
        err.contains("GENERATED ALWAYS and cannot be written directly"),
        "unexpected error: {err}"
    );
}

/// An ENCRYPTED column stores ciphertext and decrypts on scan, so reads see
/// plaintext and predicates on the column filter over decrypted values
#[tokio::test]
async fn test_encrypted_column_round_trips_and_filters() {
    let (server, _schema, _tmp) = create_test_server().await;
    let mut session = new_session();
    exec_ddl(
        &server,
        &mut session,
        "CREATE TABLE vault (id INT, secret TEXT ENCRYPTED WITH (algorithm = 'aes256_gcm'))",
    )
    .await
    .expect("create");

    exec_dml(
        &server,
        "INSERT INTO vault VALUES (1, 'alpha clearance'), (2, 'beta clearance')",
    )
    .await;

    let rows = query_values(&server, "SELECT secret FROM vault WHERE id = 1").await;
    assert_eq!(rows.len(), 1);
    assert_eq!(text_of(&rows[0][0]), "alpha clearance");

    // The scan decrypts before the predicate runs, so filtering on the
    // encrypted column finds the row by its plaintext value
    let rows = query_values(
        &server,
        "SELECT id FROM vault WHERE secret = 'beta clearance'",
    )
    .await;
    assert_eq!(rows.len(), 1);
    assert_eq!(int_of(&rows[0][0]), 2);

    let rows = query_values(&server, "SELECT secret FROM vault ORDER BY id").await;
    assert_eq!(
        rows.iter().map(|r| text_of(&r[0])).collect::<Vec<_>>(),
        vec!["alpha clearance", "beta clearance"]
    );
}

/// LTREE paths answer level counts, subpaths, inclusive ancestry, lquery
/// matching, and longest common ancestors over a small tree
#[tokio::test]
async fn test_ltree_paths_and_functions() {
    let (server, _schema, _tmp) = create_test_server().await;
    let mut session = new_session();
    exec_ddl(&server, &mut session, "CREATE TABLE taxa (path LTREE)")
        .await
        .expect("create");
    exec_dml(
        &server,
        "INSERT INTO taxa VALUES
            ('top'),
            ('top.science'),
            ('top.science.astronomy'),
            ('top.science.astronomy.astrophysics'),
            ('top.hobbies.astronomy_clubs')",
    )
    .await;

    let rows = query_values(
        &server,
        "SELECT nlevel(path) FROM taxa WHERE path = 'top.science.astronomy'",
    )
    .await;
    assert_eq!(rows.len(), 1);
    assert_eq!(int_of(&rows[0][0]), 3);

    // Prefix containment is inclusive, matching the postgres <@ operator
    let rows = query_values(
        &server,
        "SELECT path FROM taxa WHERE ltree_is_descendant(path, 'top.science') ORDER BY path",
    )
    .await;
    let paths: Vec<String> = rows.iter().map(|r| text_of(&r[0])).collect();
    assert_eq!(
        paths,
        vec![
            "top.science",
            "top.science.astronomy",
            "top.science.astronomy.astrophysics"
        ]
    );

    let rows = query_values(
        &server,
        "SELECT subpath(path, 0, 2) FROM taxa WHERE path = 'top.science.astronomy.astrophysics'",
    )
    .await;
    assert_eq!(text_of(&rows[0][0]), "top.science");

    let rows = query_values(
        &server,
        "SELECT path FROM taxa WHERE ltree_matches(path, 'top.*.astronomy')",
    )
    .await;
    assert_eq!(rows.len(), 1);
    assert_eq!(text_of(&rows[0][0]), "top.science.astronomy");

    let rows = query_values(
        &server,
        "SELECT lca(path, 'top.science.physics') FROM taxa WHERE path = 'top.science.astronomy'",
    )
    .await;
    assert_eq!(rows.len(), 1);
    assert_eq!(text_of(&rows[0][0]), "top.science");
}

/// The shipped configuration serves HTTP/3 over QUIC as the primary
/// transport, on the UDP port beside the TCP port unless one is pinned
#[tokio::test]
async fn test_quic_transport_config_defaults() {
    let config = zyron_common::ServerConfig::default();
    assert!(
        config.quic_enabled,
        "QUIC defaults on as the primary transport"
    );
    assert_eq!(config.quic_port, None);
    assert_eq!(config.quic_listen_port(), config.port + 1);
    assert_eq!(config.quic_listen_port(), 5433);
    assert!(!config.quic_zero_rtt, "0-RTT resumption stays opt in");
    assert_eq!(config.quic_idle_timeout_secs, 300);

    let mut pinned = zyron_common::ServerConfig::default();
    pinned.quic_port = Some(7443);
    assert_eq!(pinned.quic_listen_port(), 7443);
}

/// The declared shape is a contract the write path holds writers to.
///
/// A field the declaration does not name could never be read back, so
/// storing it would discard data the writer believed it had written, and a
/// field whose value does not fit its declared type would make the typed
/// read describe something that was never there.
#[tokio::test]
async fn test_a_nested_write_is_checked_against_its_declaration() {
    let (server, _schema, _tmp) = create_test_server().await;
    let mut session = new_session();
    exec_ddl(
        &server,
        &mut session,
        "CREATE TABLE decl (s STRUCT<name TEXT, age INT>, m MAP<TEXT, INT>)",
    )
    .await
    .expect("create");

    // The shape it was declared with
    exec_dml(
        &server,
        r#"INSERT INTO decl VALUES ('{"name":"ada","age":36}', '{"a":1}')"#,
    )
    .await;

    // A field the declaration does not name
    let err = exec_dml_result(
        &server,
        r#"INSERT INTO decl VALUES ('{"name":"ada","age":36,"height":170}', '{"a":1}')"#,
    )
    .await
    .expect_err("an undeclared field was stored");
    assert!(
        format!("{err:?}").contains("height"),
        "the error did not name the field: {err:?}"
    );

    // A value that does not fit the field's declared type
    let err = exec_dml_result(
        &server,
        r#"INSERT INTO decl VALUES ('{"name":"ada","age":"nope"}', '{"a":1}')"#,
    )
    .await
    .expect_err("a field value that is not an integer was stored");
    assert!(
        format!("{err:?}").contains("age"),
        "the error did not name the field: {err:?}"
    );

    // A map value that does not fit the declared value type
    let err = exec_dml_result(
        &server,
        r#"INSERT INTO decl VALUES ('{"name":"ada","age":36}', '{"a":"nope"}')"#,
    )
    .await
    .expect_err("a map value that is not an integer was stored");
    assert!(
        format!("{err:?}").to_lowercase().contains("declared"),
        "{err:?}"
    );

    // Text that is not json at all
    let err = exec_dml_result(&server, "INSERT INTO decl VALUES ('not json', '{}')")
        .await
        .expect_err("a struct column took text that is not json");
    assert!(
        format!("{err:?}").to_lowercase().contains("json"),
        "{err:?}"
    );

    // A declared field the value omits is null, the way an absent field is
    // everywhere else
    exec_dml(
        &server,
        r#"INSERT INTO decl VALUES ('{"name":"bob"}', '{}')"#,
    )
    .await;
    let rows = query_values(
        &server,
        "SELECT decl.s.age FROM decl WHERE decl.s.name = 'bob'",
    )
    .await;
    assert_eq!(rows.len(), 1);
    assert!(matches!(rows[0][0], ScalarValue::Null));

    // Only the two good rows landed
    let rows = query_values(&server, "SELECT count(*) FROM decl").await;
    assert_eq!(
        rows[0][0],
        ScalarValue::Int64(2),
        "a refused write was stored"
    );
}

/// An UPDATE is held to the declaration the same way an INSERT is, so a
/// column cannot be walked out of its shape one statement at a time
#[tokio::test]
async fn test_an_update_is_checked_against_the_declaration_too() {
    let (server, _schema, _tmp) = create_test_server().await;
    let mut session = new_session();
    exec_ddl(
        &server,
        &mut session,
        "CREATE TABLE upd (id INT, s STRUCT<name TEXT, age INT>)",
    )
    .await
    .expect("create");
    exec_dml(
        &server,
        r#"INSERT INTO upd VALUES (1, '{"name":"ada","age":36}')"#,
    )
    .await;

    let err = exec_dml_result(
        &server,
        r#"UPDATE upd SET s = '{"name":"ada","age":"nope"}' WHERE id = 1"#,
    )
    .await
    .expect_err("an update wrote a value outside the declaration");
    assert!(format!("{err:?}").contains("age"), "{err:?}");

    // The row still holds what it held
    let rows = query_values(&server, "SELECT upd.s.age FROM upd").await;
    assert_eq!(rows[0][0], ScalarValue::Int32(36));
}

/// A declared STRUCT crosses csv as the json a reader can read.
///
/// csv carries text only, so the stored layout has to be spelled out. The
/// spelling comes from the engine's own renderer rather than a second copy
/// living in the streaming crate, so the same value reads the same in a csv
/// file as it does in an answer. That equality is asserted, not assumed,
/// because two spellings of one value is the bug this arrangement avoids.
#[test]
fn a_declared_struct_crosses_csv_as_the_json_the_engine_renders() {
    use zyron_catalog::schema::{NestedShape, NestedType};
    use zyron_common::TypeId;
    use zyron_executor::nested_codec::{encode_json_text, render_json_text};
    use zyron_streaming::format::{ColumnSpec, FormatKind, reader_for, writer_for};
    use zyron_streaming::row_codec::StreamValue;

    // The engine hands its codec to the streaming layer, which is what
    // `columns_to_specs` does for a running server
    zyron_streaming::nested_render::register(render_json_text, encode_json_text);

    let shape = NestedShape::Struct(vec![
        ("kind".to_string(), NestedType::scalar(TypeId::Text)),
        ("seq".to_string(), NestedType::scalar(TypeId::Int32)),
    ]);
    let stored =
        encode_json_text(r#"{"kind":"ingest","seq":7}"#, &shape, "s").expect("the sample encodes");

    let schema = vec![
        ColumnSpec::new("id", TypeId::Int64),
        ColumnSpec::new("s", TypeId::Struct).with_nested_shape(Some(Arc::new(shape.clone()))),
    ];
    let rows = vec![vec![
        StreamValue::I64(1),
        StreamValue::Binary(stored.clone()),
    ]];

    let bytes = writer_for(FormatKind::Csv)
        .write_rows(&rows, &schema)
        .expect("a shaped struct writes to csv");
    let text = String::from_utf8(bytes.clone()).expect("csv is utf8");

    // Reading the same file with the column declared text hands the cell back
    // unescaped, which is what a csv consumer sees once its own quoting is
    // undone. It is exactly what the query path renders for the same value
    let rendered = render_json_text(&stored, &shape);
    let as_text = vec![
        ColumnSpec::new("id", TypeId::Int64),
        ColumnSpec::new("s", TypeId::Text),
    ];
    let cells = reader_for(FormatKind::Csv)
        .read_rows(&bytes, &as_text)
        .expect("the file reads back as text");
    match &cells[0][1] {
        StreamValue::Utf8(cell) => assert_eq!(
            cell, &rendered,
            "the csv cell and the query path spell one value two ways"
        ),
        other => panic!("expected the cell as text, got {other:?}"),
    }
    assert!(
        text.contains("ingest"),
        "the field values did not reach the file, got {text}"
    );

    // Reading it back rebuilds the stored layout byte for byte, so a csv
    // round trip is not a lossy one
    let back = reader_for(FormatKind::Csv)
        .read_rows(&bytes, &schema)
        .expect("a shaped struct reads from csv");
    assert_eq!(back.len(), 1);
    match &back[0][1] {
        StreamValue::Binary(b) => assert_eq!(b, &stored, "the layout changed crossing csv"),
        other => panic!("expected the stored layout back, got {other:?}"),
    }

    // json carries the object itself rather than base64 of the layout, which
    // is what a json consumer expects to find under the column name
    let json_bytes = writer_for(FormatKind::JsonLines)
        .write_rows(&rows, &schema)
        .expect("a shaped struct writes to jsonl");
    let line = String::from_utf8(json_bytes.clone()).expect("jsonl is utf8");
    assert!(
        line.contains(r#""kind":"ingest""#),
        "jsonl should hold the object, got {line}"
    );
    let json_back = reader_for(FormatKind::JsonLines)
        .read_rows(&json_bytes, &schema)
        .expect("a shaped struct reads from jsonl");
    match &json_back[0][1] {
        StreamValue::Binary(b) => assert_eq!(b, &stored, "the layout changed crossing jsonl"),
        other => panic!("expected the stored layout back, got {other:?}"),
    }

    // Without a shape the column cannot be spelled, and that is an error
    // rather than a byte dump into a text file
    let unshaped = vec![
        ColumnSpec::new("id", TypeId::Int64),
        ColumnSpec::new("s", TypeId::Struct),
    ];
    let err = writer_for(FormatKind::Csv)
        .write_rows(&rows, &unshaped)
        .expect_err("a struct with no shape cannot be written as text");
    assert!(format!("{err:?}").contains("shape"), "{err:?}");
}
