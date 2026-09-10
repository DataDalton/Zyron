//! Every statement that names a schema-scoped object reads `schema.name`.
//!
//! A table, view, index or sequence lives in a schema, so every statement
//! naming one has to accept the qualified form. The parser holds one helper
//! that reads a dotted name and another that stops at the first dot, and a
//! statement wired to the second silently truncates the name: the object part
//! is taken as the whole name and the rest fails to parse as a new statement,
//! which reports a position past the dot rather than naming the real problem.
//!
//! This gate lists the forms rather than deriving them, because there is
//! nothing in the AST that marks a field as holding an object name. A
//! statement added without its entry here is not caught, so add one with the
//! statement.
//!
//! Run: cargo test -p zyron-parser --test qualified_name_test

use zyron_parser::parse;

/// One statement per form that names a schema-scoped object, written with the
/// schema spelled out.
const QUALIFIED_FORMS: &[&str] = &[
    // Reading and writing rows
    "SELECT a FROM s.t",
    "INSERT INTO s.t VALUES (1)",
    "UPDATE s.t SET a = 1",
    "DELETE FROM s.t",
    "MERGE INTO s.t USING s.u ON s.t.a = s.u.a WHEN MATCHED THEN DELETE",
    "COPY s.t FROM 'f.csv'",
    "COPY s.t TO 'f.csv'",
    "TRUNCATE TABLE s.t",
    // Tables
    "CREATE TABLE s.t (a INT)",
    "CREATE TABLE s.t CLONE OF s.u",
    "DROP TABLE s.t",
    "UNDROP TABLE s.t",
    "ALTER TABLE s.t ADD COLUMN b INT",
    "ALTER TABLE s.t DROP COLUMN b",
    "ALTER TABLE s.t RENAME TO u",
    "ALTER TABLE s.t ADD EXPECTATION e EXPECT a > 0 ON VIOLATION WARN",
    "ALTER TABLE s.t DROP EXPECTATION e",
    "CREATE FOREIGN TABLE s.f (a INT) SERVER sv",
    "DROP FOREIGN TABLE s.f",
    // A foreign key names the table it points at
    "CREATE TABLE t (a INT REFERENCES s.u (b))",
    "CREATE TABLE t (a INT, FOREIGN KEY (a) REFERENCES s.u (b))",
    // Indexes
    "CREATE INDEX i ON s.t (a)",
    "CREATE UNIQUE INDEX i ON s.t (a)",
    "DROP INDEX s.i",
    "ALTER INDEX s.i RENAME TO j",
    "REINDEX TABLE s.t",
    "REINDEX INDEX s.i",
    "CREATE FULLTEXT INDEX f ON s.t (a)",
    "CREATE VECTOR INDEX v ON s.t (a)",
    "CREATE SPATIAL INDEX p ON s.t (a)",
    "CREATE HYBRID INDEX h ON s.t (a, b)",
    // Views
    "CREATE VIEW s.v AS SELECT 1",
    "DROP VIEW s.v",
    "ALTER VIEW s.v RENAME TO w",
    "CREATE MATERIALIZED VIEW s.m AS SELECT 1",
    "DROP MATERIALIZED VIEW s.m",
    "REFRESH MATERIALIZED VIEW s.m",
    // Sequences
    "CREATE SEQUENCE s.q",
    "DROP SEQUENCE s.q",
    "ALTER SEQUENCE s.q RESTART",
    // Triggers name a table
    "CREATE TRIGGER g BEFORE INSERT ON s.t FOR EACH ROW EXECUTE FUNCTION f",
    "DROP TRIGGER g ON s.t",
    // Maintenance
    "ANALYZE s.t",
    "VACUUM s.t",
    "OPTIMIZE TABLE s.t",
    // Comments and grants
    "COMMENT ON TABLE s.t IS 'x'",
    "COMMENT ON COLUMN s.t.a IS 'x'",
    "COMMENT ON INDEX s.i IS 'x'",
    "COMMENT ON SEQUENCE s.q IS 'x'",
    "COMMENT ON VIEW s.v IS 'x'",
    "GRANT SELECT ON s.t TO r",
    "REVOKE SELECT ON s.t FROM r",
    // Statements whose subject is a table they attach something to
    "CREATE PUBLICATION p FOR TABLE s.t",
    "CREATE CDC STREAM c ON s.t TO sk",
    "CREATE ABAC POLICY pol ON TABLE s.t WHERE a > 0",
    "ARCHIVE TABLE s.t TO 'dest'",
];

#[test]
fn every_statement_naming_an_object_reads_a_qualified_name() {
    let mut rejected: Vec<String> = Vec::new();
    for form in QUALIFIED_FORMS {
        if let Err(e) = parse(form) {
            rejected.push(format!("{form}\n      {e}"));
        }
    }
    assert!(
        rejected.is_empty(),
        "{} of {} statements reject a schema-qualified name, so an object in a \
         schema cannot be named by these:\n  {}",
        rejected.len(),
        QUALIFIED_FORMS.len(),
        rejected.join("\n  ")
    );
}

#[test]
fn a_qualified_name_reaches_the_statement_whole() {
    // Parsing is not enough on its own. A statement that consumed only the
    // schema part would still parse when the rest happened to read as another
    // statement, so the name the statement carries is read back here
    let cases: &[(&str, &str)] = &[
        ("DROP TABLE s.t", "s.t"),
        ("TRUNCATE TABLE s.t", "s.t"),
        ("UNDROP TABLE s.t", "s.t"),
        ("OPTIMIZE TABLE s.t", "s.t"),
        ("DROP VIEW s.v", "s.v"),
        ("DROP MATERIALIZED VIEW s.m", "s.m"),
        ("REFRESH MATERIALIZED VIEW s.m", "s.m"),
        ("DROP SEQUENCE s.q", "s.q"),
        ("DROP INDEX s.i", "s.i"),
        ("DROP FOREIGN TABLE s.f", "s.f"),
    ];
    for (sql, expected) in cases {
        let parsed = parse(sql).unwrap_or_else(|e| panic!("{sql} does not parse: {e}"));
        let rendered = format!("{:?}", parsed);
        assert!(
            rendered.contains(expected),
            "{sql} parsed without carrying {expected}, so the name was truncated"
        );
        assert_eq!(
            parsed.len(),
            1,
            "{sql} parsed as {} statements, so part of the name was read as \
             another statement",
            parsed.len()
        );
    }
}

#[test]
fn a_three_part_name_is_read_whole() {
    // The resolver reads catalog.schema.object, so the parser has to deliver
    // all three rather than stopping at the second dot
    let parsed = parse("DROP TABLE c.s.t").expect("a three part name parses");
    assert_eq!(parsed.len(), 1);
    assert!(
        format!("{:?}", parsed).contains("c.s.t"),
        "the catalog part was dropped"
    );
}

#[test]
fn a_column_comment_keeps_its_column_separate_from_a_qualified_table() {
    // COMMENT ON COLUMN ends in the column, so a greedy read of the dotted
    // name would take the column as part of the table
    for (sql, table, column) in [
        ("COMMENT ON COLUMN t.a IS 'x'", "t", "a"),
        ("COMMENT ON COLUMN s.t.a IS 'x'", "s.t", "a"),
        ("COMMENT ON COLUMN c.s.t.a IS 'x'", "c.s.t", "a"),
    ] {
        let parsed = parse(sql).unwrap_or_else(|e| panic!("{sql} does not parse: {e}"));
        let zyron_parser::Statement::CommentOn(stmt) = &parsed[0] else {
            panic!("{sql} did not parse as COMMENT ON");
        };
        assert_eq!(stmt.name, table, "{sql} read the wrong table");
        assert_eq!(
            stmt.column.as_deref(),
            Some(column),
            "{sql} read the wrong column"
        );
    }
}

#[test]
fn a_qualified_name_survives_the_unparser() {
    // A DDL statement replicated to a group is replayed from text, so a name
    // the unparser drops the schema from would be replayed against the wrong
    // schema on every other member. Forms the unparser does not write are
    // skipped, the same way the grammar coverage gate skips them
    let mut checked = 0usize;
    for form in QUALIFIED_FORMS {
        let parsed = parse(form).unwrap_or_else(|e| panic!("{form} does not parse: {e}"));
        let Ok(rendered) = zyron_parser::statement_to_sql(&parsed[0]) else {
            continue;
        };
        let reparsed = parse(&rendered).unwrap_or_else(|e| {
            panic!(
                "{form} does not parse after unparsing: {e}
  {rendered}"
            )
        });
        assert_eq!(
            format!("{:?}", reparsed[0]),
            format!("{:?}", parsed[0]),
            "{form} changed when written back as
  {rendered}"
        );
        checked += 1;
    }
    assert!(
        checked > 0,
        "the unparser wrote none of these back, so this gate checked nothing"
    );
    println!("{checked} of {} round-tripped", QUALIFIED_FORMS.len());
}
