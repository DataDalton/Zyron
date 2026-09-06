//! IN and NOT IN over a list, on both paths that answer it.
//!
//! Run: cargo test -p zyron-wire --test in_list_test
//!
//! A short list is answered by one comparison pass per item, a long one by
//! membership in a set. Which path runs is a performance decision and must
//! not be a semantic one, so every case here is asked twice: once with a
//! list short enough to take the passes, and once with the same values
//! padded out with non-matching filler so the same question crosses into
//! the set path. The two answers have to match.
//!
//! Three-valued logic is where a set path is easiest to get wrong. A miss
//! beside a NULL in the list is unknown rather than false, because the
//! value could have been the one that was not known, and NOT IN over a
//! list holding NULL therefore returns nothing at all.

use std::sync::Arc;

use zyron_executor::column::ScalarValue;
use zyron_wire::connection::ServerState;

mod common;
use common::{create_test_server, exec_ddl, exec_dml, new_session, query_values};

/// Filler values no row holds, enough of them to push a list past the
/// length at which membership is answered from a set
const FILLER: usize = 24;

async fn seeded() -> (Arc<ServerState>, tempfile::TempDir) {
    let (server, _schema_id, tmp) = create_test_server().await;
    let mut session = new_session();
    exec_ddl(
        &server,
        &mut session,
        "CREATE TABLE probe (id BIGINT NOT NULL, code INT, label TEXT)",
    )
    .await
    .expect("create the probe table");
    exec_dml(
        &server,
        "INSERT INTO probe VALUES \
         (1, 10, 'alpha'), (2, 20, 'beta'), (3, 30, 'gamma'), \
         (4, NULL, NULL), (5, 50, 'epsilon')",
    )
    .await;
    (server, tmp)
}

/// The same list twice: as written, and padded past the set threshold with
/// values no row holds. Padding cannot change an answer, so any difference
/// between the two is the set path disagreeing with the passes
fn both_lengths(items: &str, filler_from: i64) -> (String, String) {
    let padded: Vec<String> = (0..FILLER)
        .map(|i| (filler_from + i as i64).to_string())
        .collect();
    (items.to_string(), format!("{items}, {}", padded.join(", ")))
}

fn render(rows: &[Vec<ScalarValue>]) -> String {
    format!("{rows:?}")
}

/// Runs one predicate on both list lengths and returns the answer they
/// agree on, failing if they do not
async fn agreed(server: &Arc<ServerState>, shape: &str, short: &str, long: &str) -> String {
    let short_sql = shape.replace("{}", short);
    let long_sql = shape.replace("{}", long);
    let short_answer = render(&query_values(server, &short_sql).await);
    let long_answer = render(&query_values(server, &long_sql).await);
    assert_eq!(
        short_answer, long_answer,
        "the two paths disagree\n  passes: {short_sql}\n  set:    {long_sql}"
    );
    short_answer
}

#[tokio::test]
async fn in_a_list_of_integers_agrees_on_both_paths() {
    let (server, _tmp) = seeded().await;
    let (short, long) = both_lengths("1, 3", 900);
    let answer = agreed(
        &server,
        "SELECT id FROM probe WHERE id IN ({}) ORDER BY id",
        &short,
        &long,
    )
    .await;
    assert_eq!(
        answer,
        render(&[vec![ScalarValue::Int64(1)], vec![ScalarValue::Int64(3)]])
    );
}

#[tokio::test]
async fn not_in_a_list_of_integers_agrees_on_both_paths() {
    let (server, _tmp) = seeded().await;
    let (short, long) = both_lengths("1, 3", 900);
    let answer = agreed(
        &server,
        "SELECT id FROM probe WHERE id NOT IN ({}) ORDER BY id",
        &short,
        &long,
    )
    .await;
    assert_eq!(
        answer,
        render(&[
            vec![ScalarValue::Int64(2)],
            vec![ScalarValue::Int64(4)],
            vec![ScalarValue::Int64(5)]
        ])
    );
}

/// A NULL in the list turns every miss into unknown, so IN returns only
/// the rows that actually matched and NOT IN returns nothing at all
#[tokio::test]
async fn a_null_in_the_list_makes_a_miss_unknown_on_both_paths() {
    let (server, _tmp) = seeded().await;
    let (short, long) = both_lengths("1, 3, NULL", 900);

    let in_answer = agreed(
        &server,
        "SELECT id FROM probe WHERE id IN ({}) ORDER BY id",
        &short,
        &long,
    )
    .await;
    assert_eq!(
        in_answer,
        render(&[vec![ScalarValue::Int64(1)], vec![ScalarValue::Int64(3)]]),
        "a hit stays a hit beside a NULL"
    );

    let not_in_answer = agreed(
        &server,
        "SELECT id FROM probe WHERE id NOT IN ({}) ORDER BY id",
        &short,
        &long,
    )
    .await;
    assert_eq!(
        not_in_answer, "[]",
        "NOT IN over a list holding NULL is unknown for every row"
    );
}

/// A NULL probe is unknown whatever the list holds
#[tokio::test]
async fn a_null_probe_is_unknown_on_both_paths() {
    let (server, _tmp) = seeded().await;
    let (short, long) = both_lengths("10, 30", 900);

    let in_answer = agreed(
        &server,
        "SELECT id FROM probe WHERE code IN ({}) ORDER BY id",
        &short,
        &long,
    )
    .await;
    assert_eq!(
        in_answer,
        render(&[vec![ScalarValue::Int64(1)], vec![ScalarValue::Int64(3)]])
    );

    // Row 4 holds a NULL code and is absent from both, rather than being
    // returned by the negation
    let not_in_answer = agreed(
        &server,
        "SELECT id FROM probe WHERE code NOT IN ({}) ORDER BY id",
        &short,
        &long,
    )
    .await;
    assert_eq!(
        not_in_answer,
        render(&[vec![ScalarValue::Int64(2)], vec![ScalarValue::Int64(5)]])
    );
}

/// A narrower column against wider literals. The set widens every value to
/// 128 bits, which is the same answer the passes reach by coercing both
/// sides to a common type
#[tokio::test]
async fn a_narrow_column_against_wide_literals_agrees_on_both_paths() {
    let (server, _tmp) = seeded().await;
    let (short, long) = both_lengths("10, 50", 4_000_000_000);
    let answer = agreed(
        &server,
        "SELECT id FROM probe WHERE code IN ({}) ORDER BY id",
        &short,
        &long,
    )
    .await;
    assert_eq!(
        answer,
        render(&[vec![ScalarValue::Int64(1)], vec![ScalarValue::Int64(5)]]),
        "a literal too wide for the column matches nothing and excludes nothing"
    );
}

#[tokio::test]
async fn in_a_list_of_strings_agrees_on_both_paths() {
    let (server, _tmp) = seeded().await;
    let filler: Vec<String> = (0..FILLER).map(|i| format!("'filler-{i}'")).collect();
    let short = "'alpha', 'gamma'".to_string();
    let long = format!("{short}, {}", filler.join(", "));

    let answer = agreed(
        &server,
        "SELECT id FROM probe WHERE label IN ({}) ORDER BY id",
        &short,
        &long,
    )
    .await;
    assert_eq!(
        answer,
        render(&[vec![ScalarValue::Int64(1)], vec![ScalarValue::Int64(3)]])
    );

    // The NULL label in row 4 is unknown against any list, so the negation
    // leaves it out
    let not_answer = agreed(
        &server,
        "SELECT id FROM probe WHERE label NOT IN ({}) ORDER BY id",
        &short,
        &long,
    )
    .await;
    assert_eq!(
        not_answer,
        render(&[vec![ScalarValue::Int64(2)], vec![ScalarValue::Int64(5)]])
    );
}

/// Duplicates in the list say nothing new. A set collapses them and the
/// passes evaluate each one, so this is where the two could differ on a
/// count if either mistook membership for multiplicity
#[tokio::test]
async fn duplicates_in_the_list_change_nothing_on_both_paths() {
    let (server, _tmp) = seeded().await;
    let repeated = "1, 1, 1, 3, 3".to_string();
    let (short, long) = both_lengths(&repeated, 900);
    let answer = agreed(
        &server,
        "SELECT COUNT(*) FROM probe WHERE id IN ({})",
        &short,
        &long,
    )
    .await;
    assert_eq!(answer, render(&[vec![ScalarValue::Int64(2)]]));
}
