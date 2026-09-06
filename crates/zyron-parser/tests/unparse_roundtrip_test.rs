//! The unparser is the inverse of the parser.
//!
//! For every statement here, parsing, rendering, and parsing again has to
//! produce the tree the first parse produced. That is the property the
//! upgrade rewriter relies on: it changes a tree and writes the rendering
//! into the catalog, and the next binary parses that text. A rendering that
//! parsed to something else would change an object's meaning silently.
//!
//! Run: cargo test -p zyron-parser --test unparse_roundtrip_test

use zyron_parser::{parse, statement_to_sql};

/// Parses, renders, parses again, and requires the same tree. Returns the
/// rendering so a test can look at it
fn round_trip(sql: &str) -> String {
    let first = parse(sql).unwrap_or_else(|e| panic!("`{sql}` did not parse, {e}"));
    assert_eq!(first.len(), 1, "`{sql}` is one statement");
    let rendered =
        statement_to_sql(&first[0]).unwrap_or_else(|e| panic!("`{sql}` did not render, {e}"));
    let second = parse(&rendered)
        .unwrap_or_else(|e| panic!("the rendering of `{sql}` did not parse, {e}\n  {rendered}"));
    assert_eq!(
        second, first,
        "`{sql}` rendered as `{rendered}` reads back differently"
    );
    // Rendering the re-parsed tree gives the same text, so the text is a
    // fixed point and not something that drifts on each pass
    let again = statement_to_sql(&second[0]).expect("renders again");
    assert_eq!(again, rendered);
    rendered
}

#[test]
fn plain_selects_round_trip() {
    for sql in [
        "SELECT 1",
        "SELECT a, b AS bee, t.c, t.* FROM t",
        "SELECT * FROM s.t AS x WHERE x.a = 1 AND (x.b > 2 OR x.c IS NOT NULL)",
        "SELECT DISTINCT a FROM t ORDER BY a DESC NULLS LAST LIMIT 10 OFFSET 5",
        "SELECT DISTINCT ON (a) a, b FROM t ORDER BY a ASC NULLS FIRST, b",
        "SELECT count(*), sum(DISTINCT x), max(y) FROM t GROUP BY z HAVING count(*) > 1",
        "SELECT a FROM t GROUP BY ROLLUP (a, b)",
        "SELECT a FROM t GROUP BY CUBE (a)",
        "SELECT a FROM t GROUP BY GROUPING SETS ((a), (a, b), ())",
        "SELECT a FROM t QUALIFY row_number() OVER (PARTITION BY a ORDER BY b) = 1",
        "SELECT a FROM t FETCH FIRST 5 ROWS ONLY",
        "SELECT a FROM t ORDER BY a FETCH FIRST 10 PERCENT ROWS WITH TIES",
        "SELECT a FROM t FOR UPDATE OF t NOWAIT",
        "SELECT a FROM t FOR SHARE SKIP LOCKED",
        "SELECT a FROM t FOR NO KEY UPDATE",
        "SELECT a FROM t FOR KEY SHARE",
        "SELECT a FROM t INCLUDING DELETED",
        "SELECT a FROM t WHERE b = 1 ONLY DELETED",
        "SELECT MixedCase, \"Quoted Name\", \"select\" FROM \"Table Name\"",
    ] {
        round_trip(sql);
    }
}

#[test]
fn from_clauses_round_trip() {
    for sql in [
        "SELECT * FROM a INNER JOIN b ON a.id = b.id LEFT JOIN c USING (id) CROSS JOIN d",
        "SELECT * FROM a NATURAL JOIN b RIGHT JOIN c ON c.x = a.x FULL JOIN d ON d.y = a.y",
        "SELECT * FROM (SELECT 1 AS x) AS sub",
        "SELECT * FROM t, LATERAL (SELECT * FROM u WHERE u.id = t.id) AS l",
        "SELECT * FROM generate_series(1, 10) AS g",
        "SELECT * FROM s.f(a => 1, 2) AS g",
        "WITH RECURSIVE r (n) AS (SELECT 1 UNION ALL SELECT n + 1 FROM r WHERE n < 5) SELECT n FROM r",
        "WITH a AS (SELECT 1 AS x), b AS (SELECT x FROM a) SELECT * FROM b",
        "SELECT a FROM t UNION SELECT a FROM u INTERSECT SELECT a FROM v EXCEPT ALL SELECT a FROM w ORDER BY a",
        "SELECT * FROM t AS OF TIMESTAMP '2024-01-01'",
        "SELECT * FROM t VERSION AS OF 3 AS x",
        "SELECT * FROM t IN BRANCH 'dev' AS x",
        "SELECT * FROM t FOR PORTION OF valid FROM 1 TO 2",
    ] {
        round_trip(sql);
    }
}

#[test]
fn expressions_round_trip() {
    for sql in [
        "SELECT CASE WHEN a > 1 THEN 'x' ELSE 'y' END, CASE a WHEN 1 THEN 2 END FROM t",
        "SELECT CAST(a AS BIGINT), a::TEXT FROM t",
        "SELECT CAST(a AS DECIMAL(10, 2)), CAST(b AS TIMESTAMP(3)), CAST(c AS INT[]) FROM t",
        "SELECT a IN (1, 2, 3), b NOT IN (SELECT c FROM u), d BETWEEN 1 AND 2, e NOT BETWEEN 1 AND 2 FROM t",
        "SELECT f LIKE 'x%', g NOT ILIKE 'y', h IS NULL, i NOT LIKE 'z', j ILIKE 'w' FROM t",
        "SELECT EXISTS (SELECT 1 FROM u), NOT EXISTS (SELECT 1 FROM v) FROM t",
        "SELECT -a, NOT b, a + b * c - d / e % f, a || b, a <> b, a <= b, a >= b, a < b FROM t",
        "SELECT NOT a = b AND c OR d FROM t",
        "SELECT (a + b) * c, -(a + b), ((a)) FROM t",
        "SELECT 1.5, 1e21, 0.25, -3, TRUE, FALSE, NULL, 'it''s', 'two\nlines' FROM t",
        "SELECT 170141183460469231731687303715884105727, -170141183460469231731687303715884105727 FROM t",
        "SELECT 12345678901234567890.123456789, -12345678901234567890.123456789 FROM t",
        "SELECT INTERVAL '1 day 2 hours', INTERVAL '3 months', DATE '2024-01-01', TIMESTAMP '2024-01-01 00:00:00', TIME '12:00:00', TIMESTAMPTZ '2024-01-01 00:00:00' FROM t",
        "SELECT ARRAY[1, 2][1], ARRAY[] FROM t",
        "SELECT j -> 'a', j ->> 'b', j #> '{a,b}', j #>> '{a}', j @> '{}', j <@ '{}', j ? 'k' FROM t",
        "SELECT v.address.city, v.a.b.c.d FROM t AS v",
        "SELECT a <=> b, a <-> b, a <#> b FROM t",
        "SELECT MATCH (title, body) AGAINST ('query' IN boolean mode) FROM t",
        "SELECT MATCH (title) AGAINST ('query') FROM t",
        "SELECT $1 + $2",
        "SELECT EXTRACT(year FROM d), substring(s FROM 1 FOR 2), substring(s, 1, 2) FROM t",
        "SELECT a COLLATE 'de_DE' FROM t ORDER BY b COLLATE 'C'",
        "SELECT sum(x) OVER (PARTITION BY a ORDER BY b ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) FROM t",
        "SELECT sum(x) OVER (ORDER BY b RANGE BETWEEN INTERVAL '1 hour' PRECEDING AND CURRENT ROW) FROM t",
        "SELECT sum(x) OVER (ROWS 3 PRECEDING), rank() OVER () FROM t",
        "SELECT sum(x) OVER (ORDER BY b ROWS BETWEEN 2 PRECEDING AND 1 FOLLOWING) FROM t",
        "SELECT f(a => 1, b => 2), g(), count(DISTINCT *) FROM t",
        "SELECT row_diff(t.x AS OF TIMESTAMP '2024-01-01', t.x VERSION AS OF 2, t.x IN BRANCH 'dev') FROM t",
        "SELECT (SELECT max(b) FROM u) AS m FROM t",
        "SELECT a FROM t WHERE b = (SELECT c FROM u WHERE u.id = t.id)",
    ] {
        round_trip(sql);
    }
}

#[test]
fn dml_round_trips() {
    for sql in [
        "INSERT INTO s.t (a, b) VALUES (1, 'x'), (2, 'y') ON CONFLICT (a) DO UPDATE SET b = 'z' RETURNING a",
        "INSERT INTO t SELECT * FROM u ON CONFLICT DO NOTHING",
        "INSERT INTO t VALUES (1)",
        "UPDATE t SET a = 1, b = b + 1 WHERE c = 2 RETURNING *",
        "UPDATE s.t SET a = NULL",
        "DELETE FROM t WHERE a = 1 RETURNING a HARD",
        "DELETE FROM s.t",
        "MERGE INTO t USING u ON t.id = u.id WHEN MATCHED AND u.x > 1 THEN UPDATE SET a = u.a WHEN MATCHED THEN DELETE WHEN NOT MATCHED THEN INSERT (id, a) VALUES (u.id, u.a)",
        "VALUES (1, 'a'), (2, 'b')",
    ] {
        round_trip(sql);
    }
}

#[test]
fn user_object_definitions_round_trip() {
    for sql in [
        "CREATE OR REPLACE VIEW s.v (a, b) AS SELECT 1, 2",
        "CREATE VIEW v AS SELECT a FROM s.t WHERE a > 1 ORDER BY a",
        "CREATE MATERIALIZED VIEW IF NOT EXISTS mv AS SELECT a, count(*) FROM t GROUP BY a",
        "CREATE FUNCTION f(a INT, b TEXT DEFAULT 'x') RETURNS BIGINT AS 'SELECT 1' LANGUAGE SQL IMMUTABLE",
        "CREATE OR REPLACE FUNCTION g() RETURNS TABLE(x INT, y TEXT) AS 'SELECT 1, ''a''' LANGUAGE RUST VOLATILE LIBRARY 'lib.so' SYMBOL 'sym'",
        "CREATE FUNCTION h() RETURNS SETOF INT AS 'x' LANGUAGE RUST_VECTORIZED STABLE",
        "CREATE PROCEDURE p(a INT) AS 'CALL q(1)' LANGUAGE PLSQL SECURITY DEFINER",
        "CREATE OR REPLACE PROCEDURE s.p() AS 'SELECT 1' LANGUAGE SQL",
        "CALL s.p(1, 'a')",
        "CALL p()",
        "CREATE SCHEDULE nightly EVERY 1 DAYS DO DELETE FROM t WHERE old",
        "CREATE SCHEDULE hourly EVERY 2 HOURS DO CALL p()",
        "CREATE SCHEDULE cron_job CRON '0 0 * * *' DO INSERT INTO t SELECT * FROM u",
        "CREATE PIPELINE pl AS (STAGE bronze (SOURCE raw, TARGET clean, MODE append, TRANSFORM AS (SELECT * FROM raw), EXPECT a > 0, EXPECT b IS NOT NULL), STAGE silver (SOURCE clean, TARGET gold))",
        "CREATE STREAMING JOB IF NOT EXISTS sj AS SELECT a, b FROM src INTO dst WRITE MODE UPSERT MODE SCHEDULED EVERY '5 minutes' WATERMARK FOR ts AS ts - INTERVAL '10 seconds' WITH LATE DATA POLICY SIDE OUTPUT",
        "CREATE STREAMING JOB j2 AS SELECT * FROM a INNER JOIN b ON a.k = b.k WITHIN INTERVAL '5' minute INTO FILE '/out' FORMAT JSONLINES OPTIONS (x = '1') WRITE MODE APPEND MODE ONESHOT",
        "CREATE STREAMING JOB j3 AS SELECT * FROM FILE '/data/x.csv' FORMAT CSV OPTIONS (delimiter = ',') COLUMNS (a INT, b TEXT) AS f INTO dst MODE WATCH",
        "CREATE STREAMING JOB j4 AS SELECT * FROM a LEFT JOIN b AS OF ts ON a.k = b.k INTO dst MODE SCHEDULED CRON '* * * * *'",
        "CREATE ENDPOINT IF NOT EXISTS ep ON PATH '/api/x' METHOD GET, POST USING 'SELECT 1' AUTH JWT REQUIRE SCOPE 'read', 'write' RATE LIMIT 100 / minute PER IP OUTPUT FORMAT CSV CORS ORIGINS 'https://a' CACHE 30 SECONDS TIMEOUT 5 SECONDS MAX REQUEST BODY 64 KB",
        "CREATE ENDPOINT ep2 ON PATH '/y' METHOD GET USING 'SELECT $1' AUTH NONE",
        "CREATE ENDPOINT ep3 ON PATH '/z' METHOD PATCH USING 'DELETE FROM t' AUTH API_KEY RATE LIMIT 5 / second",
    ] {
        round_trip(sql);
    }
}

#[test]
fn the_rendering_is_one_line_of_readable_sql() {
    let rendered = round_trip("SELECT a,\n  b\nFROM t\nWHERE a = 1");
    assert_eq!(rendered, "SELECT a, b FROM t WHERE a = 1");
    let rendered = round_trip("create view v as select x from y");
    assert_eq!(rendered, "CREATE VIEW v AS SELECT x FROM y");
}

#[test]
fn a_statement_with_no_spelling_is_refused_rather_than_approximated() {
    for sql in [
        "CREATE TABLE t (a INT)",
        "CREATE SCHEMA s",
        "EXPLAIN SELECT 1",
        "BEGIN",
    ] {
        let statement = parse(sql).expect("parses").remove(0);
        let err = statement_to_sql(&statement).expect_err(sql);
        assert!(err.to_string().contains("no SQL spelling"), "{err}");
    }
}
