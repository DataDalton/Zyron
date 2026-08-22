//! Parse tests for the lake format DDL surface, USING and CLUSTER BY.

use zyron_parser::ast::{ClusterMode, ClusteringSchedule, Statement, TableFormat};
use zyron_parser::parse;

fn create_table(sql: &str) -> zyron_parser::ast::CreateTableStatement {
    let mut stmts = parse(sql).expect("parses");
    assert_eq!(stmts.len(), 1);
    match stmts.remove(0) {
        Statement::CreateTable(ct) => *ct,
        other => panic!("expected CREATE TABLE, got {:?}", other),
    }
}

#[test]
fn test_using_format_variants() {
    let lake = create_table("CREATE TABLE t (id BIGINT) USING ZYRONLAKE");
    assert_eq!(lake.using, Some(TableFormat::ZyronLake));
    assert!(lake.cluster_by.is_none());

    let heap = create_table("CREATE TABLE t (id BIGINT) USING HEAP");
    assert_eq!(heap.using, Some(TableFormat::Heap));

    let plain = create_table("CREATE TABLE t (id BIGINT)");
    assert_eq!(plain.using, None);

    let err = parse("CREATE TABLE t (id BIGINT) USING PARQUET");
    assert!(err.is_err());
}

#[test]
fn test_cluster_by_forms() {
    // A bare key list seeds the layout without pinning it, so the mode is
    // Auto and measurement may move the keys on once it has evidence
    let seeded = create_table("CREATE TABLE t (a INT, b INT) USING ZYRONLAKE CLUSTER BY (a, b)");
    let clause = seeded.cluster_by.expect("clause");
    assert_eq!(clause.mode, ClusterMode::Auto);
    assert_eq!(clause.keys.len(), 2);
    assert_eq!(clause.keys[0].column_name(), Some("a"));
    assert!(clause.keys[0].strategy.is_none());

    // FORCE is what pins a declared key set against measurement
    let pinned =
        create_table("CREATE TABLE t (a INT, b INT) USING ZYRONLAKE CLUSTER BY (a, b) FORCE");
    let clause = pinned.cluster_by.expect("clause");
    assert_eq!(clause.mode, ClusterMode::Force);
    assert_eq!(clause.keys.len(), 2);

    let auto = create_table("CREATE TABLE t (a INT) USING ZYRONLAKE CLUSTER BY AUTO");
    let clause = auto.cluster_by.expect("clause");
    assert_eq!(clause.mode, ClusterMode::Auto);
    assert!(clause.keys.is_empty());

    let hybrid = create_table("CREATE TABLE t (a INT, b INT) USING ZYRONLAKE CLUSTER BY (a) AUTO");
    let clause = hybrid.cluster_by.expect("clause");
    assert_eq!(clause.mode, ClusterMode::Hybrid);
    assert_eq!(clause.keys.len(), 1);

    let strategies = create_table(
        "CREATE TABLE t (a INT, b INT) USING ZYRONLAKE CLUSTER BY (a USING SpaceFilling, b)",
    );
    let clause = strategies.cluster_by.expect("clause");
    assert_eq!(clause.keys[0].strategy.as_deref(), Some("SpaceFilling"));
    assert!(clause.keys[1].strategy.is_none());

    let none = create_table("CREATE TABLE t (a INT) USING ZYRONLAKE CLUSTER BY () FORCE");
    let clause = none.cluster_by.expect("clause");
    assert_eq!(clause.mode, ClusterMode::Force);
    assert!(clause.keys.is_empty());
}

#[test]
fn test_clauses_compose_with_ttl_and_with() {
    let full = create_table(
        "CREATE TABLE t (id BIGINT, ts TIMESTAMP) USING ZYRONLAKE CLUSTER BY (ts) \
         WITH (target_file_size = '256MB')",
    );
    assert_eq!(full.using, Some(TableFormat::ZyronLake));
    assert!(full.cluster_by.is_some());
    assert_eq!(full.options.len(), 1);
    assert_eq!(full.options[0].key, "target_file_size");
}

#[test]
fn test_alter_table_lake_statements() {
    let mut set_using = parse("ALTER TABLE t SET USING ZYRONLAKE").expect("parses");
    match set_using.remove(0) {
        Statement::AlterTableSetUsing(s) => {
            assert_eq!(s.table, "t");
            assert_eq!(s.format, TableFormat::ZyronLake);
            assert!(s.options.is_empty());
        }
        other => panic!("unexpected {:?}", other),
    }

    let mut with_opts =
        parse("ALTER TABLE t SET USING HEAP WITH (drop_history = true)").expect("parses");
    match with_opts.remove(0) {
        Statement::AlterTableSetUsing(s) => {
            assert_eq!(s.format, TableFormat::Heap);
            assert_eq!(s.options.len(), 1);
            assert_eq!(s.options[0].key, "drop_history");
        }
        other => panic!("unexpected {:?}", other),
    }

    let mut cluster = parse("ALTER TABLE t CLUSTER BY (a, b) AUTO").expect("parses");
    match cluster.remove(0) {
        Statement::AlterTableClusterBy(s) => {
            assert_eq!(s.table, "t");
            assert_eq!(s.clause.mode, ClusterMode::Hybrid);
            assert_eq!(s.clause.keys.len(), 2);
        }
        other => panic!("unexpected {:?}", other),
    }

    let mut sched = parse("ALTER TABLE t SET CLUSTERING SCHEDULE = CONTINUOUS").expect("parses");
    match sched.remove(0) {
        Statement::AlterTableClusteringSchedule(s) => {
            assert_eq!(s.schedule, ClusteringSchedule::Continuous);
        }
        other => panic!("unexpected {:?}", other),
    }

    // The plain options path still works beside the new SET forms
    let mut opts = parse("ALTER TABLE t SET (fillfactor = 70)").expect("parses");
    assert!(matches!(opts.remove(0), Statement::AlterTableOptions(_)));
}

#[test]
fn test_new_keywords_still_work_as_identifiers() {
    // Every soft keyword stays usable as a table or column name
    let t = create_table("CREATE TABLE cluster (auto INT, force TEXT, zorder BIGINT)");
    assert_eq!(t.name, "cluster");
    assert_eq!(t.columns.len(), 3);
    assert_eq!(t.columns[0].name, "auto");
    assert_eq!(t.columns[1].name, "force");
    assert_eq!(t.columns[2].name, "zorder");

    let q = parse("SELECT auto, force FROM clustering WHERE incremental = 1").expect("parses");
    assert!(matches!(q.first(), Some(Statement::Select(_))));

    // CLONE, added for CREATE TABLE ... CLONE OF
    let t = create_table("CREATE TABLE clone (clone INT)");
    assert_eq!(t.name, "clone");
    assert_eq!(t.columns[0].name, "clone");
    assert!(parse("SELECT clone FROM t").is_ok());
}

/// A clone takes the source's shape rather than declaring one, so it has
/// no column list at all. The version is optional and means the source's
/// head when absent
#[test]
fn test_create_table_clone_of_parses_with_and_without_a_version() {
    let head = create_table("CREATE TABLE backup CLONE OF orders");
    assert_eq!(head.name, "backup");
    assert!(
        head.columns.is_empty(),
        "a clone declares no columns, it takes the source's"
    );
    let source = head.clone_of.expect("a clone source");
    assert_eq!(source.table, "orders");
    assert_eq!(
        source.at_version, None,
        "no version named means the source as it stands now"
    );

    let pinned = create_table("CREATE TABLE audit CLONE OF orders AT VERSION 42");
    let source = pinned.clone_of.expect("a clone source");
    assert_eq!(source.table, "orders");
    assert_eq!(source.at_version, Some(42));

    // IF NOT EXISTS still applies, because a clone is a CREATE TABLE
    let guarded = create_table("CREATE TABLE IF NOT EXISTS backup CLONE OF orders");
    assert!(guarded.if_not_exists);
    assert!(guarded.clone_of.is_some());
}

/// A version has to be a whole number. Accepting anything else would defer
/// the failure to the point where the source is opened, which is a worse
/// place to find out
#[test]
fn test_a_clone_version_that_is_not_a_number_is_refused() {
    assert!(parse("CREATE TABLE backup CLONE OF orders AT VERSION latest").is_err());
    assert!(parse("CREATE TABLE backup CLONE OF orders AT VERSION -1").is_err());
    assert!(
        parse("CREATE TABLE backup CLONE OF orders AT 42").is_err(),
        "the version keyword is not optional, or AT would be ambiguous with a time travel form"
    );
}

/// A foreign table names a peer and the shape it expects there
#[test]
fn test_create_foreign_table_names_a_server_and_a_remote() {
    let mut plain =
        parse("CREATE FOREIGN TABLE orders (id BIGINT, total DOUBLE PRECISION) SERVER west")
            .expect("parses");
    match plain.remove(0) {
        Statement::CreateForeignTable(s) => {
            assert_eq!(s.name, "orders");
            assert_eq!(s.server, "west");
            assert_eq!(
                s.remote_table, None,
                "an unnamed remote is the local name, not a missing one"
            );
            assert_eq!(s.columns.len(), 2);
            assert_eq!(s.columns[0].name, "id");
            assert_eq!(s.columns[1].name, "total");
            assert!(!s.if_not_exists);
        }
        other => panic!("unexpected {:?}", other),
    }

    let mut renamed =
        parse("CREATE FOREIGN TABLE IF NOT EXISTS o (id BIGINT) SERVER west TABLE orders")
            .expect("parses");
    match renamed.remove(0) {
        Statement::CreateForeignTable(s) => {
            assert_eq!(s.name, "o");
            assert_eq!(s.server, "west");
            assert_eq!(s.remote_table.as_deref(), Some("orders"));
            assert!(s.if_not_exists);
        }
        other => panic!("unexpected {:?}", other),
    }
}

/// The peer owns the rows, so it owns their constraints. Accepting one
/// here would report enforcement this node never performs
#[test]
fn test_a_foreign_column_may_not_carry_a_constraint() {
    assert!(parse("CREATE FOREIGN TABLE t (id BIGINT NOT NULL) SERVER west").is_err());
    assert!(parse("CREATE FOREIGN TABLE t (id BIGINT PRIMARY KEY) SERVER west").is_err());
    assert!(parse("CREATE FOREIGN TABLE t (id BIGINT DEFAULT 1) SERVER west").is_err());
    // A table with no column could never be read from
    assert!(parse("CREATE FOREIGN TABLE t () SERVER west").is_err());
    // SERVER is the only way to name the peer
    assert!(parse("CREATE FOREIGN TABLE t (id BIGINT)").is_err());
}

#[test]
fn test_drop_foreign_table() {
    let mut plain = parse("DROP FOREIGN TABLE orders").expect("parses");
    match plain.remove(0) {
        Statement::DropForeignTable(s) => {
            assert_eq!(s.name, "orders");
            assert!(!s.if_exists);
        }
        other => panic!("unexpected {:?}", other),
    }

    let mut guarded = parse("DROP FOREIGN TABLE IF EXISTS orders").expect("parses");
    match guarded.remove(0) {
        Statement::DropForeignTable(s) => {
            assert_eq!(s.name, "orders");
            assert!(s.if_exists);
        }
        other => panic!("unexpected {:?}", other),
    }
}

/// SERVER joins the soft keywords: still usable wherever a name goes
#[test]
fn test_server_stays_usable_as_an_identifier() {
    let t = create_table("CREATE TABLE server (server INT)");
    assert_eq!(t.name, "server");
    assert_eq!(t.columns[0].name, "server");

    let q = parse("SELECT server FROM server WHERE server = 1").expect("parses");
    assert!(matches!(q.first(), Some(Statement::Select(_))));
}

/// Expression keys and column keys interleave in one key list, and a bare
/// identifier still means the column rather than an expression over it
#[test]
fn test_cluster_by_accepts_expressions_beside_columns() {
    use zyron_parser::ast::{ClusterKeyTarget, Expr};

    let t = create_table(
        "CREATE TABLE t (ts TIMESTAMP, country TEXT, region TEXT) USING ZYRONLAKE          CLUSTER BY (date_trunc('day', ts), region, upper(country) USING BitInterleave)",
    );
    let clause = t.cluster_by.expect("clause");
    assert_eq!(clause.keys.len(), 3);

    assert!(matches!(
        clause.keys[0].target,
        ClusterKeyTarget::Expression(Expr::Function { .. })
    ));
    assert_eq!(clause.keys[0].column_name(), None);

    assert_eq!(clause.keys[1].column_name(), Some("region"));

    assert!(matches!(
        clause.keys[2].target,
        ClusterKeyTarget::Expression(Expr::Function { .. })
    ));
    assert_eq!(
        clause.keys[2].strategy.as_deref(),
        Some("BitInterleave"),
        "a strategy still binds to the key it follows"
    );
}

/// ADD DERIVED COLUMN names an expression column so it can be referred to
#[test]
fn test_add_derived_column_parses() {
    use zyron_parser::ast::{AlterTableOperation, Expr};

    let mut stmts =
        parse("ALTER TABLE t ADD DERIVED COLUMN day AS date_trunc('day', ts)").expect("parses");
    match stmts.remove(0) {
        Statement::AlterTable(a) => match a.operation {
            AlterTableOperation::AddDerivedColumn { name, expr } => {
                assert_eq!(name, "day");
                assert!(matches!(expr, Expr::Function { .. }));
            }
            other => panic!("unexpected operation {:?}", other),
        },
        other => panic!("unexpected {:?}", other),
    }

    // COLUMN is optional, matching ADD [COLUMN]
    assert!(parse("ALTER TABLE t ADD DERIVED day AS upper(country)").is_ok());
    // A plain ADD COLUMN is unaffected
    assert!(parse("ALTER TABLE t ADD COLUMN c INT").is_ok());
    // DERIVED stays usable as an identifier
    assert!(parse("SELECT derived FROM t").is_ok());
}

/// Rolling a table back to a version it held is a different statement from
/// reading a flat snapshot out of an archive, and the grammar keeps them
/// apart: TO names a version this table had, FROM names a file it did not
#[test]
fn test_restore_table_to_version_and_to_timestamp_parse() {
    let mut parsed = parse("RESTORE TABLE orders TO VERSION 7").expect("parses");
    match parsed.remove(0) {
        Statement::RestoreTableVersion(s) => {
            assert_eq!(s.table, "orders");
            assert_eq!(
                s.target,
                zyron_parser::ast::RestoreVersionTarget::Version(7)
            );
        }
        other => panic!("expected a version restore, got {other:?}"),
    }

    let mut parsed =
        parse("RESTORE TABLE orders TO TIMESTAMP '2026-08-19 12:00:00'").expect("parses");
    match parsed.remove(0) {
        Statement::RestoreTableVersion(s) => {
            assert_eq!(
                s.target,
                zyron_parser::ast::RestoreVersionTarget::Timestamp("2026-08-19 12:00:00".into())
            );
        }
        other => panic!("expected a timestamp restore, got {other:?}"),
    }

    // The forms that were already there still parse as themselves
    assert!(matches!(
        parse("RESTORE TABLE orders").expect("parses").remove(0),
        Statement::UndropTable(_)
    ));
    assert!(matches!(
        parse("RESTORE TABLE orders FROM 's3://bucket/snap'")
            .expect("parses")
            .remove(0),
        Statement::RestoreTable(_)
    ));
}

/// A restore target has to be one of the two things a table has, or the
/// statement would defer its failure to whatever it guessed
#[test]
fn test_a_restore_target_that_is_neither_a_version_nor_a_timestamp_is_refused() {
    assert!(parse("RESTORE TABLE orders TO 7").is_err());
    assert!(parse("RESTORE TABLE orders TO VERSION yesterday").is_err());
    assert!(parse("RESTORE TABLE orders TO TIMESTAMP 7").is_err());
}
