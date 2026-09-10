//! The SQL surface this phase adds: UNNEST, FLATTEN, temporary tables,
//! PIVOT, UNPIVOT and ASOF JOIN.
//!
//! Every construct is checked twice: that it parses into the tree the binder
//! reads, and that the unparser writes it back as written, so stored SQL
//! round-trips through a catalog and a later binary reads the same statement.
//!
//! Run: cargo test -p zyron-parser --test sql_surface_test -- --nocapture

use zyron_parser::ast::*;
use zyron_parser::grammar::{GRAMMAR, GrammarPosition, entries_matching, entry_for_word};
use zyron_parser::{parse, statement_to_sql};

fn one(sql: &str) -> Statement {
    let mut stmts = parse(sql).unwrap_or_else(|e| panic!("`{sql}` did not parse, {e}"));
    assert_eq!(stmts.len(), 1, "`{sql}` is one statement");
    stmts.remove(0)
}

fn select_of(sql: &str) -> Box<SelectStatement> {
    match one(sql) {
        Statement::Select(s) => s,
        other => panic!("`{sql}` is not a SELECT, it is {other:?}"),
    }
}

/// Parses, renders, parses again, and requires the same tree, then requires
/// the second rendering to equal the first so the text is a fixed point.
fn round_trip(sql: &str) -> String {
    let first = parse(sql).unwrap_or_else(|e| panic!("`{sql}` did not parse, {e}"));
    let rendered =
        statement_to_sql(&first[0]).unwrap_or_else(|e| panic!("`{sql}` did not render, {e}"));
    let second = parse(&rendered)
        .unwrap_or_else(|e| panic!("the rendering of `{sql}` did not parse, {e}\n  {rendered}"));
    assert_eq!(
        second, first,
        "`{sql}` rendered as `{rendered}` reads back differently"
    );
    let again = statement_to_sql(&second[0]).expect("renders again");
    assert_eq!(again, rendered, "the rendering is not a fixed point");
    rendered
}

// ---------------------------------------------------------------------------
// UNNEST
// ---------------------------------------------------------------------------

#[test]
fn unnest_parses_one_array_and_several() {
    let stmt = select_of("SELECT x FROM UNNEST(ARRAY[1, 2, 3]) AS t (x)");
    let TableRef::Unnest(unnest) = &stmt.from[0] else {
        panic!("expected UNNEST, got {:?}", stmt.from[0]);
    };
    assert_eq!(unnest.arrays.len(), 1);
    assert!(!unnest.with_ordinality);
    assert_eq!(unnest.alias.as_deref(), Some("t"));
    assert_eq!(unnest.column_aliases, vec!["x".to_string()]);

    let stmt = select_of("SELECT a, b FROM UNNEST(p, q) AS t (a, b)");
    let TableRef::Unnest(unnest) = &stmt.from[0] else {
        panic!("expected UNNEST");
    };
    assert_eq!(unnest.arrays.len(), 2, "two arrays zip to two columns");
    assert_eq!(
        unnest.column_aliases,
        vec!["a".to_string(), "b".to_string()]
    );
}

#[test]
fn unnest_with_ordinality_is_recorded() {
    let stmt = select_of("SELECT v, n FROM UNNEST(ARRAY[10, 20]) WITH ORDINALITY AS t (v, n)");
    let TableRef::Unnest(unnest) = &stmt.from[0] else {
        panic!("expected UNNEST");
    };
    assert!(unnest.with_ordinality);
    assert_eq!(unnest.column_aliases.len(), 2, "the position is a column");
}

#[test]
fn unnest_under_lateral_wraps_rather_than_replaces() {
    let stmt = select_of("SELECT o.id, e FROM orders o, LATERAL UNNEST(o.items) AS u (e)");
    let TableRef::Lateral { subquery } = &stmt.from[1] else {
        panic!("expected LATERAL, got {:?}", stmt.from[1]);
    };
    assert!(matches!(subquery.as_ref(), TableRef::Unnest(_)));
}

#[test]
fn unnest_round_trips_as_written() {
    for sql in [
        "SELECT x FROM UNNEST(ARRAY[1, 2, 3]) AS t (x)",
        "SELECT a, b FROM UNNEST(p, q) AS t (a, b)",
        "SELECT v, n FROM UNNEST(ARRAY[10, 20]) WITH ORDINALITY AS t (v, n)",
        "SELECT o.id, e FROM orders AS o, LATERAL UNNEST(o.items) AS u (e)",
    ] {
        round_trip(sql);
    }
}

#[test]
fn unnest_still_reads_as_a_relation_name_without_parentheses() {
    // The word opens the construct only when '(' follows it, so a table
    // actually named `unnest` still parses
    let stmt = select_of("SELECT * FROM unnest");
    assert!(matches!(&stmt.from[0], TableRef::Table { name, .. } if name == "unnest"));
}

// ---------------------------------------------------------------------------
// FLATTEN
// ---------------------------------------------------------------------------

#[test]
fn flatten_reads_its_named_arguments() {
    let stmt = select_of(
        "SELECT f.path, f.value FROM FLATTEN(doc, path => 'a.b[*]', outer => TRUE, recursive => TRUE) AS f",
    );
    let TableRef::Flatten(flatten) = &stmt.from[0] else {
        panic!("expected FLATTEN, got {:?}", stmt.from[0]);
    };
    assert_eq!(flatten.path.as_deref(), Some("a.b[*]"));
    assert!(flatten.outer);
    assert!(flatten.recursive);
    assert_eq!(flatten.alias.as_deref(), Some("f"));
}

#[test]
fn flatten_defaults_are_off() {
    let stmt = select_of("SELECT * FROM FLATTEN(doc)");
    let TableRef::Flatten(flatten) = &stmt.from[0] else {
        panic!("expected FLATTEN");
    };
    assert!(flatten.path.is_none());
    assert!(!flatten.outer);
    assert!(!flatten.recursive);
}

#[test]
fn flatten_refuses_an_argument_it_does_not_have() {
    let err = parse("SELECT * FROM FLATTEN(doc, depth => 2)").expect_err("refused");
    let text = err.to_string();
    assert!(
        text.contains("depth") && text.contains("path"),
        "the refusal names the argument and what FLATTEN takes, got {text}"
    );
}

#[test]
fn flatten_round_trips_as_written() {
    for sql in [
        "SELECT * FROM FLATTEN(doc)",
        "SELECT * FROM FLATTEN(doc, path => 'a.b[*]') AS f",
        "SELECT * FROM FLATTEN(doc, outer => TRUE, recursive => TRUE) AS f (seq, key, path, index, value, this)",
        "SELECT o.id, f.value FROM orders AS o, LATERAL FLATTEN(o.doc) AS f",
    ] {
        round_trip(sql);
    }
}

// ---------------------------------------------------------------------------
// Temporary tables
// ---------------------------------------------------------------------------

#[test]
fn temporary_and_temp_both_open_a_temporary_table() {
    for sql in [
        "CREATE TEMPORARY TABLE t (a INT)",
        "CREATE TEMP TABLE t (a INT)",
    ] {
        let Statement::CreateTable(create) = one(sql) else {
            panic!("`{sql}` is not a CREATE TABLE");
        };
        assert!(create.temporary, "`{sql}` did not set temporary");
        assert_eq!(
            create.on_commit,
            Some(OnCommitAction::PreserveRows),
            "the default keeps the rows"
        );
    }
}

#[test]
fn on_commit_actions_parse() {
    for (sql, want) in [
        (
            "CREATE TEMP TABLE t (a INT) ON COMMIT PRESERVE ROWS",
            OnCommitAction::PreserveRows,
        ),
        (
            "CREATE TEMP TABLE t (a INT) ON COMMIT DELETE ROWS",
            OnCommitAction::DeleteRows,
        ),
        (
            "CREATE TEMP TABLE t (a INT) ON COMMIT DROP",
            OnCommitAction::Drop,
        ),
    ] {
        let Statement::CreateTable(create) = one(sql) else {
            panic!("`{sql}` is not a CREATE TABLE");
        };
        assert_eq!(create.on_commit, Some(want), "`{sql}`");
    }
}

#[test]
fn a_permanent_table_carries_no_commit_action_and_refuses_the_clause() {
    let Statement::CreateTable(create) = one("CREATE TABLE t (a INT)") else {
        panic!("not a CREATE TABLE");
    };
    assert!(!create.temporary);
    assert!(create.on_commit.is_none());

    let err = parse("CREATE TABLE t (a INT) ON COMMIT DELETE ROWS").expect_err("refused");
    assert!(
        err.to_string().contains("temporary"),
        "the refusal says the clause reads only on a temporary table, got {err}"
    );
}

#[test]
fn a_temporary_table_refuses_a_schema_qualifier() {
    for sql in [
        "CREATE TEMP TABLE app.t (a INT)",
        "SELECT a INTO TEMP app.t FROM src",
    ] {
        let err = parse(sql).expect_err("refused");
        let text = err.to_string();
        assert!(
            text.contains("bare name") && text.contains("qualified"),
            "`{sql}` should say a temporary table takes a bare name, got {text}"
        );
    }
}

#[test]
fn create_temp_table_as_select_and_select_into_temp_both_carry_the_query() {
    let Statement::CreateTable(create) = one("CREATE TEMP TABLE t AS SELECT a, b FROM src") else {
        panic!("not a CREATE TABLE");
    };
    assert!(create.temporary);
    assert!(create.columns.is_empty(), "the layout comes from the query");
    assert!(create.as_query.is_some());

    let Statement::CreateTable(create) = one("SELECT a, b INTO TEMP t FROM src") else {
        panic!("SELECT ... INTO TEMP did not become a CREATE TABLE");
    };
    assert!(create.temporary);
    assert_eq!(create.name, "t");
    let query = create.as_query.expect("carries the query");
    assert_eq!(query.projections.len(), 2);
    assert!(
        query.into_target.is_none(),
        "the target is lifted out of the query, not left on it"
    );
}

#[test]
fn or_replace_reaches_a_temporary_table() {
    let Statement::CreateTable(create) = one("CREATE OR REPLACE TEMP TABLE t (a INT)") else {
        panic!("not a CREATE TABLE");
    };
    assert!(create.or_replace);
    assert!(create.temporary);
}

#[test]
fn temp_still_reads_as_a_relation_name() {
    let stmt = select_of("SELECT * FROM temp");
    assert!(matches!(&stmt.from[0], TableRef::Table { name, .. } if name == "temp"));
    assert!(parse("SELECT temporary FROM t").is_ok());
}

// ---------------------------------------------------------------------------
// PIVOT and UNPIVOT
// ---------------------------------------------------------------------------

#[test]
fn pivot_reads_its_aggregates_and_static_value_list() {
    let stmt = select_of(
        "SELECT * FROM sales PIVOT (SUM(amount) AS total FOR quarter IN ('Q1' AS q1, 'Q2' AS q2)) AS p",
    );
    let TableRef::Pivot(pivot) = &stmt.from[0] else {
        panic!("expected PIVOT, got {:?}", stmt.from[0]);
    };
    assert_eq!(pivot.aggregates.len(), 1);
    assert_eq!(pivot.aggregates[0].function, "SUM");
    assert_eq!(pivot.aggregates[0].alias.as_deref(), Some("total"));
    assert_eq!(pivot.values.len(), 2);
    assert_eq!(pivot.values[0].alias.as_deref(), Some("q1"));
    assert!(!pivot.value_subquery);
    assert_eq!(pivot.alias.as_deref(), Some("p"));
}

#[test]
fn pivot_records_a_subquery_value_list_for_the_binder_to_refuse() {
    let stmt = select_of(
        "SELECT * FROM sales PIVOT (SUM(amount) FOR quarter IN (SELECT DISTINCT quarter FROM sales))",
    );
    let TableRef::Pivot(pivot) = &stmt.from[0] else {
        panic!("expected PIVOT");
    };
    assert!(
        pivot.value_subquery,
        "the parser records the subquery so the binder can name the two statements"
    );
    assert!(pivot.values.is_empty());
}

#[test]
fn two_pivot_aggregates_are_kept_in_order() {
    let stmt = select_of(
        "SELECT * FROM sales PIVOT (SUM(amount) AS total, COUNT(amount) AS n FOR region IN ('east', 'west'))",
    );
    let TableRef::Pivot(pivot) = &stmt.from[0] else {
        panic!("expected PIVOT");
    };
    assert_eq!(pivot.aggregates.len(), 2);
    assert_eq!(pivot.aggregates[1].function, "COUNT");
    assert_eq!(
        pivot.aggregates.len() * pivot.values.len(),
        4,
        "two aggregates over two values make four output columns"
    );
}

#[test]
fn unpivot_reads_its_null_handling_and_groups() {
    let stmt = select_of(
        "SELECT * FROM t UNPIVOT EXCLUDE NULLS (amount FOR month IN (jan AS 'Jan', feb AS 'Feb')) AS u",
    );
    let TableRef::Unpivot(unpivot) = &stmt.from[0] else {
        panic!("expected UNPIVOT, got {:?}", stmt.from[0]);
    };
    assert!(!unpivot.include_nulls);
    assert_eq!(unpivot.value_columns, vec!["amount".to_string()]);
    assert_eq!(unpivot.name_column, "month");
    assert_eq!(unpivot.items.len(), 2);
    assert_eq!(
        unpivot.items[0].label,
        Some(LiteralValue::String("Jan".to_string()))
    );

    let stmt = select_of("SELECT * FROM t UNPIVOT INCLUDE NULLS (amount FOR month IN (jan, feb))");
    let TableRef::Unpivot(unpivot) = &stmt.from[0] else {
        panic!("expected UNPIVOT");
    };
    assert!(unpivot.include_nulls);
    assert!(unpivot.items[0].label.is_none());
}

#[test]
fn a_two_value_column_unpivot_takes_tuple_groups() {
    let stmt = select_of(
        "SELECT * FROM t UNPIVOT ((amount, tax) FOR month IN ((jan_amt, jan_tax) AS 'Jan', (feb_amt, feb_tax) AS 'Feb'))",
    );
    let TableRef::Unpivot(unpivot) = &stmt.from[0] else {
        panic!("expected UNPIVOT");
    };
    assert_eq!(unpivot.value_columns.len(), 2);
    assert_eq!(unpivot.items.len(), 2);
    assert_eq!(unpivot.items[0].columns.len(), 2);
}

#[test]
fn an_unpivot_group_of_the_wrong_arity_is_refused() {
    let err =
        parse("SELECT * FROM t UNPIVOT ((amount, tax) FOR month IN ((jan_amt, jan_tax), feb_amt))")
            .expect_err("refused");
    let text = err.to_string();
    assert!(
        text.contains("1 column(s)") && text.contains("declares 2"),
        "the refusal states both counts, got {text}"
    );
}

#[test]
fn pivot_and_unpivot_round_trip_as_written() {
    for sql in [
        "SELECT * FROM sales PIVOT (SUM(amount) AS total FOR quarter IN ('Q1' AS q1, 'Q2' AS q2)) AS p",
        "SELECT * FROM sales PIVOT (SUM(amount) AS total, COUNT(amount) AS n FOR region IN ('east', 'west'))",
        "SELECT * FROM t UNPIVOT EXCLUDE NULLS (amount FOR month IN (jan AS 'Jan', feb AS 'Feb')) AS u",
        "SELECT * FROM t UNPIVOT INCLUDE NULLS (amount FOR month IN (jan, feb))",
        "SELECT * FROM t UNPIVOT ((amount, tax) FOR month IN ((jan_amt, jan_tax) AS 'Jan', (feb_amt, feb_tax) AS 'Feb'))",
    ] {
        let rendered = round_trip(sql);
        assert!(
            rendered.contains("PIVOT"),
            "the rendering keeps the construct as written rather than the rewrite, got {rendered}"
        );
    }
}

// ---------------------------------------------------------------------------
// ASOF JOIN
// ---------------------------------------------------------------------------

#[test]
fn asof_join_reads_its_match_condition_and_optional_on() {
    let stmt = select_of(
        "SELECT * FROM trades ASOF JOIN quotes MATCH_CONDITION (trades.ts >= quotes.ts) ON trades.symbol = quotes.symbol",
    );
    let TableRef::Join(join) = &stmt.from[0] else {
        panic!("expected a join, got {:?}", stmt.from[0]);
    };
    assert_eq!(join.join_type, JoinType::Inner);
    let asof = join.asof.as_ref().expect("carries a match condition");
    assert!(matches!(asof.condition, Expr::BinaryOp { .. }));
    assert!(matches!(join.condition, JoinCondition::On(_)));
}

#[test]
fn asof_left_join_is_a_left_join() {
    let stmt = select_of(
        "SELECT * FROM trades ASOF LEFT JOIN quotes MATCH_CONDITION (trades.ts >= quotes.ts)",
    );
    let TableRef::Join(join) = &stmt.from[0] else {
        panic!("expected a join");
    };
    assert_eq!(join.join_type, JoinType::Left);
    assert!(join.asof.is_some());
    assert_eq!(join.condition, JoinCondition::None, "ON is optional");
}

#[test]
fn a_tolerance_rides_inside_the_match_condition() {
    let stmt = select_of(
        "SELECT * FROM trades ASOF JOIN quotes MATCH_CONDITION (trades.ts >= quotes.ts AND trades.ts - quotes.ts <= INTERVAL '5 minutes')",
    );
    let TableRef::Join(join) = &stmt.from[0] else {
        panic!("expected a join");
    };
    let asof = join.asof.as_ref().expect("carries a match condition");
    assert!(
        matches!(&asof.condition, Expr::BinaryOp { op, .. } if *op == BinaryOperator::And),
        "the tolerance is the second conjunct of one condition"
    );
}

#[test]
fn asof_join_requires_a_match_condition() {
    let err = parse("SELECT * FROM trades ASOF JOIN quotes ON trades.symbol = quotes.symbol")
        .expect_err("refused");
    assert!(
        err.to_string().to_uppercase().contains("MATCH_CONDITION"),
        "the refusal names the missing clause, got {err}"
    );
}

#[test]
fn asof_still_reads_as_a_relation_name() {
    let stmt = select_of("SELECT * FROM asof");
    assert!(matches!(&stmt.from[0], TableRef::Table { name, .. } if name == "asof"));
}

#[test]
fn asof_join_round_trips_as_written() {
    for sql in [
        "SELECT * FROM trades ASOF JOIN quotes MATCH_CONDITION (trades.ts >= quotes.ts) ON trades.symbol = quotes.symbol",
        "SELECT * FROM trades ASOF LEFT JOIN quotes MATCH_CONDITION (trades.ts > quotes.ts)",
        "SELECT * FROM trades ASOF JOIN quotes MATCH_CONDITION (trades.ts <= quotes.ts)",
        "SELECT * FROM trades ASOF JOIN quotes MATCH_CONDITION (trades.ts >= quotes.ts AND trades.ts - quotes.ts <= INTERVAL '5 minutes') ON trades.symbol = quotes.symbol",
    ] {
        let rendered = round_trip(sql);
        assert!(
            rendered.contains("ASOF") && rendered.contains("MATCH_CONDITION"),
            "the rendering keeps the ASOF spelling, got {rendered}"
        );
    }
}

// ---------------------------------------------------------------------------
// One file holding all four, and the grammar description
// ---------------------------------------------------------------------------

#[test]
fn a_file_holding_every_construct_round_trips() {
    let file = concat!(
        "SELECT x FROM UNNEST(ARRAY[1, 2, 3]) WITH ORDINALITY AS t (x, n); ",
        "SELECT f.value FROM FLATTEN(doc, path => 'a.b[*]', recursive => TRUE) AS f; ",
        "SELECT * FROM sales PIVOT (SUM(amount) FOR quarter IN ('Q1', 'Q2')) AS p; ",
        "SELECT * FROM t UNPIVOT EXCLUDE NULLS (amount FOR month IN (jan, feb)) AS u; ",
        "SELECT * FROM trades ASOF LEFT JOIN quotes MATCH_CONDITION (trades.ts >= quotes.ts) ON trades.symbol = quotes.symbol",
    );
    let first = parse(file).expect("the file parses");
    assert_eq!(first.len(), 5);
    let rendered: Vec<String> = first
        .iter()
        .map(|s| statement_to_sql(s).expect("renders"))
        .collect();
    let second = parse(&rendered.join("; ")).expect("the rendering parses");
    assert_eq!(second, first, "the file does not survive a round trip");
}

#[test]
fn the_grammar_description_covers_every_construct_this_phase_added() {
    for word in ["UNNEST", "FLATTEN", "PIVOT", "UNPIVOT", "MATCH_CONDITION"] {
        let entry = entry_for_word(word)
            .unwrap_or_else(|| panic!("{word} is not in the grammar description"));
        assert!(!entry.syntax.is_empty(), "{word} has no syntax to hover");
        assert!(!entry.summary.is_empty(), "{word} has no summary to hover");
    }
    let asof = entry_for_word("ASOF").expect("ASOF resolves through its opening word");
    assert_eq!(asof.name, "ASOF JOIN");
    assert_eq!(asof.position, GrammarPosition::Join);

    let temp = entry_for_word("CREATE TEMPORARY TABLE").expect("the temporary form is described");
    assert_eq!(temp.position, GrammarPosition::Statement);

    // Completion reaches each of them by prefix
    assert!(entries_matching("UNN").any(|e| e.name == "UNNEST"));
    assert!(entries_matching("FLA").any(|e| e.name == "FLATTEN"));
    assert!(entries_matching("ASOF").any(|e| e.name == "ASOF JOIN"));
    assert!(
        GRAMMAR.len() >= 20,
        "the description should carry the array functions too"
    );
}
