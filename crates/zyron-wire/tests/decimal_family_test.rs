//! DECIMAL correctness across joins, aggregates, casts, computed schemas
//! and every write path.
//!
//! A decimal is an i128 holding the value times ten to the column's scale,
//! and the scale lives on the column. Every operator that combines two
//! decimal sources, converts one to another type, or writes one into a
//! column has to account for the scale explicitly. Each test here pins one
//! place that read or wrote the raw scaled integer as if it were the value.
//!
//! Run: cargo test -p zyron-wire --test decimal_family_test

mod common;

use common::{create_test_server, exec_ddl, exec_dml, new_session, query_values};
use zyron_executor::column::ScalarValue;

async fn setup(
) -> (
    std::sync::Arc<zyron_wire::connection::ServerState>,
    Option<zyron_wire::session::Session>,
    tempfile::TempDir,
) {
    let (server, _schema, tmp) = create_test_server().await;
    let session = new_session();
    (server, session, tmp)
}

fn fmt_decimals(rows: &[Vec<ScalarValue>], scale: u8) -> Vec<String> {
    rows.iter()
        .map(|row| match row.first() {
            Some(ScalarValue::Int128(v)) => zyron_common::format_decimal(*v, scale),
            Some(ScalarValue::Null) => "NULL".to_string(),
            other => panic!("expected a decimal, got {other:?}"),
        })
        .collect()
}

/// Equal numbers on different scales are still equal join keys.
#[tokio::test]
async fn test_cross_scale_decimal_equi_join_matches() {
    let (server, mut session, _tmp) = setup().await;
    exec_ddl(&server, &mut session, "CREATE TABLE a (v DECIMAL(10,2))")
        .await
        .expect("create a");
    exec_ddl(&server, &mut session, "CREATE TABLE b (w DECIMAL(10,3))")
        .await
        .expect("create b");
    exec_dml(&server, "INSERT INTO a VALUES (10.50), (7.25)").await;
    exec_dml(&server, "INSERT INTO b VALUES (10.500), (7.251)").await;

    let rows = query_values(
        &server,
        "SELECT a.v FROM a JOIN b ON a.v = b.w",
    )
    .await;
    assert_eq!(
        fmt_decimals(&rows, 2),
        vec!["10.50"],
        "10.50 equals 10.500 and 7.25 does not equal 7.251"
    );
}

/// An integer key equals a decimal key when the numbers are equal, not when
/// the raw stored integers are.
#[tokio::test]
async fn test_int_decimal_equi_join_compares_values() {
    let (server, mut session, _tmp) = setup().await;
    exec_ddl(&server, &mut session, "CREATE TABLE i (n BIGINT NOT NULL)")
        .await
        .expect("create i");
    exec_ddl(&server, &mut session, "CREATE TABLE d (v DECIMAL(10,2))")
        .await
        .expect("create d");
    // 1050 as an integer must not match 10.50, whose raw storage is 1050.
    // 10 must match 10.00, whose raw storage is 1000.
    exec_dml(&server, "INSERT INTO i VALUES (10), (1050)").await;
    exec_dml(&server, "INSERT INTO d VALUES (10.50), (10.00)").await;

    let rows = query_values(
        &server,
        "SELECT i.n FROM i JOIN d ON i.n = d.v",
    )
    .await;
    assert_eq!(rows.len(), 1, "exactly one numeric match exists");
    assert_eq!(rows[0][0], ScalarValue::Int64(10));
}

/// The partitioned parallel join path must co-locate equal cross-scale keys.
#[tokio::test]
async fn test_cross_scale_decimal_join_partitioned_path() {
    let (server, mut session, _tmp) = setup().await;
    exec_ddl(&server, &mut session, "CREATE TABLE pa (v DECIMAL(12,2))")
        .await
        .expect("create pa");
    exec_ddl(&server, &mut session, "CREATE TABLE pb (w DECIMAL(12,3))")
        .await
        .expect("create pb");
    // Enough rows on both sides that the join partitions instead of running
    // the single serial fallback
    for chunk in 0..5 {
        let values: Vec<String> = (0..1000)
            .map(|i| format!("({}.25)", chunk * 1000 + i))
            .collect();
        exec_dml(&server, &format!("INSERT INTO pa VALUES {}", values.join(","))).await;
        let values: Vec<String> = (0..1000)
            .map(|i| format!("({}.250)", chunk * 1000 + i))
            .collect();
        exec_dml(&server, &format!("INSERT INTO pb VALUES {}", values.join(","))).await;
    }

    let rows = query_values(
        &server,
        "SELECT COUNT(*) FROM pa JOIN pb ON pa.v = pb.w",
    )
    .await;
    assert_eq!(
        rows[0][0],
        ScalarValue::Int64(5000),
        "every key matches exactly once across the scales"
    );
}

/// AVG divides the scaled sum back onto the value scale.
#[tokio::test]
async fn test_avg_of_decimal_is_the_value_average() {
    let (server, mut session, _tmp) = setup().await;
    exec_ddl(&server, &mut session, "CREATE TABLE d (v DECIMAL(10,2))")
        .await
        .expect("create");
    exec_dml(&server, "INSERT INTO d VALUES (1.00), (2.00)").await;

    let rows = query_values(&server, "SELECT AVG(v) FROM d").await;
    assert_eq!(
        rows[0][0],
        ScalarValue::Float64(1.5),
        "the average of 1.00 and 2.00 is 1.5, not 150"
    );
}

/// STDDEV and VARIANCE accept decimal input instead of silently returning
/// NULL, and they compute on the value scale.
#[tokio::test]
async fn test_stddev_and_variance_of_decimal_compute_on_values() {
    let (server, mut session, _tmp) = setup().await;
    exec_ddl(&server, &mut session, "CREATE TABLE d (v DECIMAL(10,2))")
        .await
        .expect("create");
    exec_dml(&server, "INSERT INTO d VALUES (1.00), (3.00)").await;

    let rows = query_values(&server, "SELECT VARIANCE_AGG(v) FROM d").await;
    assert_eq!(
        rows[0][0],
        ScalarValue::Float64(2.0),
        "sample variance of 1 and 3 is 2"
    );

    let rows = query_values(&server, "SELECT STDDEV_AGG(v) FROM d").await;
    match rows[0][0] {
        ScalarValue::Float64(s) => assert!(
            (s - 2f64.sqrt()).abs() < 1e-12,
            "sample stddev of 1 and 3 is sqrt(2), got {s}"
        ),
        ref other => panic!("expected a float, got {other:?}"),
    }
}

/// CAST out of a decimal converts the value, not the raw scaled integer.
#[tokio::test]
async fn test_cast_from_decimal_converts_the_value() {
    let (server, mut session, _tmp) = setup().await;
    exec_ddl(&server, &mut session, "CREATE TABLE d (v DECIMAL(10,2))")
        .await
        .expect("create");
    exec_dml(&server, "INSERT INTO d VALUES (10.50)").await;

    let rows = query_values(&server, "SELECT CAST(v AS BIGINT) FROM d").await;
    assert_eq!(
        rows[0][0],
        ScalarValue::Int64(11),
        "10.50 to BIGINT rounds half away from zero, not 1050"
    );

    let rows = query_values(&server, "SELECT CAST(v AS DOUBLE PRECISION) FROM d").await;
    assert_eq!(rows[0][0], ScalarValue::Float64(10.5));

    let rows = query_values(&server, "SELECT CAST(v AS TEXT) FROM d").await;
    assert_eq!(rows[0][0], ScalarValue::Utf8("10.50".to_string()));
}

/// HAVING compares an aggregated decimal on the value scale.
#[tokio::test]
async fn test_having_compares_aggregated_decimals_as_values() {
    let (server, mut session, _tmp) = setup().await;
    exec_ddl(
        &server,
        &mut session,
        "CREATE TABLE d (g INT, v DECIMAL(10,2))",
    )
    .await
    .expect("create");
    exec_dml(
        &server,
        "INSERT INTO d VALUES (1, 10.50), (1, 20.25), (1, 5.00)",
    )
    .await;

    let rows = query_values(
        &server,
        "SELECT g FROM d GROUP BY g HAVING SUM(v) > 100",
    )
    .await;
    assert!(
        rows.is_empty(),
        "35.75 is not greater than 100, raw 3575 is the bug"
    );

    let rows = query_values(
        &server,
        "SELECT g FROM d GROUP BY g HAVING SUM(v) > 30",
    )
    .await;
    assert_eq!(rows.len(), 1, "35.75 is greater than 30");
}

/// A computed decimal in a derived table keeps its scale for the outer query.
#[tokio::test]
async fn test_computed_decimal_keeps_scale_through_derived_table() {
    let (server, mut session, _tmp) = setup().await;
    exec_ddl(
        &server,
        &mut session,
        "CREATE TABLE d (a DECIMAL(12,2), b DECIMAL(12,4))",
    )
    .await
    .expect("create");
    exec_dml(&server, "INSERT INTO d VALUES (1.25, 0.0625)").await;

    let rows = query_values(
        &server,
        "SELECT s FROM (SELECT a + b AS s FROM d) t WHERE s < 2",
    )
    .await;
    assert_eq!(
        rows.len(),
        1,
        "1.3125 is less than 2, its raw 13125 is not"
    );
}

/// INSERT from a SELECT across scales stores the value on the target scale.
#[tokio::test]
async fn test_insert_select_rescales_to_the_target_column() {
    let (server, mut session, _tmp) = setup().await;
    exec_ddl(&server, &mut session, "CREATE TABLE src (v DECIMAL(10,3))")
        .await
        .expect("create src");
    exec_ddl(&server, &mut session, "CREATE TABLE dst (v DECIMAL(10,2))")
        .await
        .expect("create dst");
    exec_dml(&server, "INSERT INTO src VALUES (1.234), (2.235)").await;
    exec_dml(&server, "INSERT INTO dst SELECT v FROM src").await;

    let rows = query_values(&server, "SELECT v FROM dst ORDER BY v").await;
    assert_eq!(
        fmt_decimals(&rows, 2),
        vec!["1.23", "2.24"],
        "scale three rescales onto scale two with rounding"
    );
}

/// ALTER TABLE ADD COLUMN keeps a decimal declaration's scale.
#[tokio::test]
async fn test_alter_add_column_keeps_decimal_scale() {
    let (server, mut session, _tmp) = setup().await;
    exec_ddl(&server, &mut session, "CREATE TABLE d (id INT)")
        .await
        .expect("create");
    exec_dml(&server, "INSERT INTO d VALUES (1)").await;
    exec_ddl(
        &server,
        &mut session,
        "ALTER TABLE d ADD COLUMN v DECIMAL(10,2) DEFAULT 1.50",
    )
    .await
    .expect("alter");

    let rows = query_values(&server, "SELECT v FROM d").await;
    assert_eq!(fmt_decimals(&rows, 2), vec!["1.50"]);

    exec_dml(&server, "INSERT INTO d VALUES (2, 10.50)").await;
    let rows = query_values(&server, "SELECT v FROM d ORDER BY id").await;
    assert_eq!(fmt_decimals(&rows, 2), vec!["1.50", "10.50"]);
}

/// ALTER TABLE ALTER COLUMN SET TYPE to a decimal converts existing values
/// onto the declared scale.
#[tokio::test]
async fn test_alter_set_type_to_decimal_scales_existing_rows() {
    let (server, mut session, _tmp) = setup().await;
    exec_ddl(&server, &mut session, "CREATE TABLE d (id INT, v BIGINT)")
        .await
        .expect("create");
    exec_dml(&server, "INSERT INTO d VALUES (1, 3)").await;
    exec_ddl(
        &server,
        &mut session,
        "ALTER TABLE d ALTER COLUMN v TYPE DECIMAL(10,2)",
    )
    .await
    .expect("alter");

    let rows = query_values(&server, "SELECT v FROM d").await;
    assert_eq!(fmt_decimals(&rows, 2), vec!["3.00"]);
}

/// A lake table's UPDATE stores the assigned decimal, not zero.
#[tokio::test]
async fn test_lake_decimal_update_keeps_the_value() {
    let (server, mut session, _tmp) = setup().await;
    exec_ddl(
        &server,
        &mut session,
        "CREATE TABLE lk (id BIGINT NOT NULL, v DECIMAL(10,2)) USING ZYRONLAKE",
    )
    .await
    .expect("create");
    exec_dml(&server, "INSERT INTO lk VALUES (1, 10.50)").await;
    exec_dml(&server, "UPDATE lk SET v = 42.75 WHERE id = 1").await;

    let rows = query_values(&server, "SELECT v FROM lk").await;
    assert_eq!(fmt_decimals(&rows, 2), vec!["42.75"]);
}

/// DECIMAL(38,20): a scale past twelve survives the catalog round trip, so
/// the table still opens after a restart.
#[tokio::test]
async fn test_wide_decimal_scale_survives_catalog_roundtrip() {
    use zyron_catalog::schema::ColumnEntry;
    let entry = ColumnEntry {
        id: zyron_catalog::ids::ColumnId(1),
        table_id: zyron_catalog::ids::TableId(1),
        name: "v".to_string(),
        type_id: zyron_common::TypeId::Decimal,
        ordinal: 0,
        nullable: true,
        default_expr: None,
        max_length: Some(38),
        fractional_digits: Some(20),
        tz_offset_secs: None,
        element_type: None,
    };
    let bytes = entry.to_bytes();
    let back = ColumnEntry::from_bytes(&bytes).expect("a stored scale of 20 must read back");
    assert_eq!(back.fractional_digits, Some(20));
}

/// COPY TO renders a decimal as fixed point, not as its raw scaled integer,
/// so the output round-trips back through COPY FROM.
#[tokio::test]
async fn test_copy_to_renders_decimals_as_fixed_point() {
    use zyron_executor::batch::DataBatch;
    use zyron_executor::column::{Column, ColumnData, NullBitmap};
    use zyron_wire::copy::{CopyFormat, CopyOutHandler};

    let col = Column::with_nulls_ts(
        ColumnData::Int128(vec![1050, -325]),
        NullBitmap::none(2),
        zyron_common::TypeId::Decimal,
        Some(2),
    );
    let batch = DataBatch::new(vec![col]);
    let schema = vec![zyron_planner::logical::LogicalColumn {
        table_idx: Some(0),
        column_id: zyron_catalog::ids::ColumnId(0),
        name: "v".to_string(),
        type_id: zyron_common::TypeId::Decimal,
        nullable: true,
        fractional_digits: Some(2),
    }];
    let handler = CopyOutHandler::new(schema, CopyFormat::Text);
    let lines: Vec<String> = handler
        .format_batch(&batch)
        .into_iter()
        .map(|m| match m {
            zyron_wire::messages::backend::BackendMessage::CopyData(bytes) => {
                String::from_utf8(bytes).expect("utf8")
            }
            other => panic!("expected CopyData, got {other:?}"),
        })
        .collect();
    assert_eq!(lines, vec!["10.50\n".to_string(), "-3.25\n".to_string()]);
}

/// The declared precision and scale are validated at parse time instead of
/// being truncated modulo 256.
#[tokio::test]
async fn test_out_of_range_decimal_declarations_are_refused() {
    assert!(
        zyron_parser::parse("CREATE TABLE d (v DECIMAL(300,2))").is_err(),
        "precision 300 does not fit an i128 and must be refused, not truncated"
    );
    assert!(
        zyron_parser::parse("CREATE TABLE d2 (v DECIMAL(10,20))").is_err(),
        "scale greater than precision must be refused"
    );
    assert!(
        zyron_parser::parse("CREATE TABLE d3 (v DECIMAL(38,12))").is_ok(),
        "the widest valid declaration still parses"
    );
}

#[tokio::test]
async fn division_rounds_half_away_like_multiplication() {
    let (server, mut session, _tmp) = setup().await;
    common::exec_ddl(
        &server,
        &mut session,
        "CREATE TABLE dv (a DECIMAL(10,2) NOT NULL, b DECIMAL(10,2) NOT NULL)",
    )
    .await
    .expect("create");
    exec_dml(&server, "INSERT INTO dv VALUES (2.00, 3.00), (-2.00, 3.00)").await;

    // 2 / 3 is 0.666..., the last kept digit rounds away from zero on both
    // signs, matching how multiplication already rounds
    assert_eq!(
        common::query_rows(
            &server,
            "SELECT a FROM dv WHERE a / b = CAST(0.67 AS DECIMAL(10,2))"
        )
        .await,
        1,
        "positive quotient rounds 0.666 to 0.67"
    );
    assert_eq!(
        common::query_rows(
            &server,
            "SELECT a FROM dv WHERE a / b = CAST(-0.67 AS DECIMAL(10,2))"
        )
        .await,
        1,
        "negative quotient rounds -0.666 to -0.67"
    );
}

#[tokio::test]
async fn sum_overflow_is_an_error_not_a_wrap() {
    let (server, mut session, _tmp) = setup().await;
    common::exec_ddl(
        &server,
        &mut session,
        "CREATE TABLE so (v DECIMAL(38,0) NOT NULL)",
    )
    .await
    .expect("create");
    exec_dml(
        &server,
        "INSERT INTO so VALUES (99999999999999999999999999999999999999), \
         (99999999999999999999999999999999999999)",
    )
    .await;

    let err = common::query_error(&server, "SELECT SUM(v) FROM so").await;
    assert!(
        err.to_lowercase().contains("overflow"),
        "a sum past the 128-bit accumulator must fail loudly, got: {err}"
    );
}

/// A decimal literal wider than an f64 holds must land exactly. The lexer
/// read every fixed-point literal through f64, which is exact only to 17
/// significant digits, so a wider literal silently changed value between
/// the statement and the row while the insert reported success.
#[tokio::test]
async fn test_wide_decimal_literals_land_exactly() {
    let (server, mut session, _tmp) = setup().await;
    exec_ddl(&server, &mut session, "CREATE TABLE wide (v DECIMAL(38, 9))")
        .await
        .expect("create");
    exec_dml(
        &server,
        "INSERT INTO wide VALUES (123456789012345678.123456789), (-0.000000001)",
    )
    .await;
    let rows = query_values(&server, "SELECT v FROM wide ORDER BY v").await;
    assert_eq!(
        fmt_decimals(&rows, 9),
        vec![
            "-0.000000001".to_string(),
            "123456789012345678.123456789".to_string(),
        ],
        "27 significant digits survive the literal path"
    );

    // The same literal as a predicate selects exactly the row it names
    let hit = query_values(
        &server,
        "SELECT v FROM wide WHERE v = 123456789012345678.123456789",
    )
    .await;
    assert_eq!(
        fmt_decimals(&hit, 9),
        vec!["123456789012345678.123456789".to_string()],
        "a wide literal compares exactly against the stored value"
    );
}
