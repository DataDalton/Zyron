//! Verifiable Tables Benchmark
//!
//! What a commit chain costs a write, what its serial section costs a
//! commit, what a verification costs and what it costs the writers running
//! beside it, and what an entry costs on disk.
//!
//! Performance Targets:
//! | Test                                                  | Metric              | Target        |
//! |-------------------------------------------------------|---------------------|---------------|
//! | INSERT, verified vs immutable-only, 1M rows one commit | delta               | ≤ 5%          |
//! | INSERT, verified, 16 parallel writers, small commits  | delta vs immutable  | ≤ 10%         |
//! | Chain serial section per commit                       | latency             | 5us           |
//! | SHA-256 over commit rows                              | throughput          | 1 GB/s        |
//! | VERIFY sampled, large table, many commits             | latency             | 60s           |
//! | VERIFY full                                           | throughput          | 500 MB/s      |
//! | Writer throughput during a full VERIFY                | delta vs no verify  | ≤ 10%         |
//! | Anchor write                                          | latency             | 5ms           |
//! | Chain storage per commit                              | bytes               | 96            |
//!
//! Validation Requirements:
//! - Each measured benchmark runs 5 iterations over one prepared data set
//! - Results averaged across all 5 runs
//! - Pass/fail determined by average performance
//! - Individual runs logged for variance analysis
//!
//! Run: cargo test --release -p zyron-wire --test verify_bench -- --nocapture

use std::sync::Arc;
use std::time::Instant;

use zyron_bench_harness::*;
use zyron_lifecycle::verify::{self, CommitChain, RowsHasher};
use zyron_wire::connection::ServerState;
use zyron_wire::session::Session;

mod common;
use common::*;

const VALIDATION_RUNS: usize = 5;

/// The most a chain may add to a bulk load of one commit
const BULK_DELTA_TARGET: f64 = 5.0;
/// The most a chain may add to sixteen writers taking small commits
const PARALLEL_DELTA_TARGET: f64 = 10.0;
/// Microseconds the serial section of one commit may take
const SERIAL_SECTION_US_TARGET: f64 = 5.0;
/// Bytes a second the row hash must reach
const ROW_HASH_BYTES_PER_SEC_TARGET: f64 = 1_000_000_000.0;
/// Seconds a sampled verification of the prepared table may take
const SAMPLED_VERIFY_SECS_TARGET: f64 = 60.0;
/// Bytes a second a full verification must reach
const FULL_VERIFY_BYTES_PER_SEC_TARGET: f64 = 500_000_000.0;
/// The most a running verification may cost a writer
const WRITER_DELTA_TARGET: f64 = 10.0;
/// Milliseconds one anchor may take
const ANCHOR_MS_TARGET: f64 = 5.0;
/// Bytes one chain entry takes
const CHAIN_BYTES_PER_COMMIT_TARGET: f64 = 96.0;

/// Rows the bulk commit writes. Large enough that the hash dominates the
/// chain's fixed cost, small enough to prepare five times
const BULK_ROWS: usize = 1_000_000;
/// Commits the sixteen writers take between them
const PARALLEL_COMMITS: usize = 16 * 64;
/// Commits the verification table holds
const VERIFY_COMMITS: usize = 2_000;
/// Rows each of those commits wrote
const VERIFY_ROWS_PER_COMMIT: usize = 50;

async fn ddl(server: &Arc<ServerState>, session: &mut Option<Session>, sql: &str) {
    exec_ddl(server, session, sql)
        .await
        .unwrap_or_else(|e| panic!("`{sql}` failed: {e}"));
}

/// One statement inserting `rows` rows, which is one commit
fn bulk_insert(table: &str, rows: usize, base: usize) -> String {
    let mut sql = String::with_capacity(rows * 24);
    sql.push_str("INSERT INTO ");
    sql.push_str(table);
    sql.push_str(" VALUES ");
    for n in 0..rows {
        if n > 0 {
            sql.push_str(", ");
        }
        let id = base + n;
        sql.push_str(&format!("({id}, {})", id * 7));
    }
    sql
}

// ---------------------------------------------------------------------------
// What the chain costs a write
// ---------------------------------------------------------------------------

/// A bulk load of one commit pays one hash pass over its rows and one link.
///
/// Measured against the same load into an immutable table with no chain, so
/// what the figure states is the chain's own cost rather than the engine's
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn test_insert_throughput_verified_vs_immutable() {
    zyron_bench_harness::init("verify");
    tprintln!("\n=== INSERT, verified vs immutable-only, one commit of {BULK_ROWS} rows ===");

    let mut plain_runs = Vec::with_capacity(VALIDATION_RUNS);
    let mut chained_runs = Vec::with_capacity(VALIDATION_RUNS);

    for run in 0..VALIDATION_RUNS {
        let (server, _schema, _tmp) = create_test_server_with_cdc().await;
        let mut session = new_session();
        ddl(
            &server,
            &mut session,
            "CREATE TABLE plain (id BIGINT, amount BIGINT)",
        )
        .await;
        ddl(
            &server,
            &mut session,
            "ALTER TABLE plain SET (immutable = true)",
        )
        .await;
        ddl(
            &server,
            &mut session,
            "CREATE TABLE chained (id BIGINT, amount BIGINT)",
        )
        .await;
        ddl(
            &server,
            &mut session,
            "ALTER TABLE chained SET (immutable = true, verified = true)",
        )
        .await;

        // The statement text is built before either is timed, so what is
        // measured is the write rather than the formatting
        let plain_sql = bulk_insert("plain", BULK_ROWS, run * BULK_ROWS * 2);
        let chained_sql = bulk_insert("chained", BULK_ROWS, run * BULK_ROWS * 2 + BULK_ROWS);

        // The order alternates per run, so whichever goes first does not
        // carry the cost of warming the pool for the other
        let (plain_secs, chained_secs) = if run % 2 == 0 {
            let started = Instant::now();
            exec_dml(&server, &plain_sql).await;
            let plain = started.elapsed().as_secs_f64();
            let started = Instant::now();
            exec_dml(&server, &chained_sql).await;
            (plain, started.elapsed().as_secs_f64())
        } else {
            let started = Instant::now();
            exec_dml(&server, &chained_sql).await;
            let chained = started.elapsed().as_secs_f64();
            let started = Instant::now();
            exec_dml(&server, &plain_sql).await;
            (started.elapsed().as_secs_f64(), chained)
        };
        plain_runs.push(plain_secs);
        chained_runs.push(chained_secs);

        assert_eq!(
            server
                .chain_registry
                .as_ref()
                .expect("chains")
                .chain(table_id_of(&server, "chained"))
                .expect("chain")
                .head()
                .commits,
            1,
            "the bulk load is one commit and one chain entry"
        );
    }

    let plain = plain_runs.iter().sum::<f64>() / plain_runs.len() as f64;
    let chained = chained_runs.iter().sum::<f64>() / chained_runs.len() as f64;
    let delta = ((chained - plain) / plain) * 100.0;
    record_metric("INSERT one commit", "immutable only", " s", plain_runs);
    record_metric("INSERT one commit", "verified", " s", chained_runs);
    tprintln!("  Verified adds {delta:.2}% over immutable only (target {BULK_DELTA_TARGET:.0}%)");
    assert!(
        check_performance(
            "INSERT one commit",
            "verified vs immutable-only delta",
            delta,
            BULK_DELTA_TARGET,
            false,
        ),
        "the chain adds {delta:.2}% to a bulk load, over the {BULK_DELTA_TARGET:.0}% target"
    );
}

/// Sixteen writers taking small commits queue only for the link.
///
/// Sixteen writers' worth of single-row commits through the whole write
/// path, rows into the heap and a commit record into the log, into an
/// immutable table and then into one whose commits are chained, so what the
/// delta states is what the chain costs sixteen writers contending for one
/// table's link. The serial section itself, the head read, the link and the
/// publish, is timed on its own with nothing contending for it, which is
/// the time the chain lock is held per commit
#[tokio::test(flavor = "multi_thread", worker_threads = 8)]
async fn test_parallel_writers_queue_only_for_the_link() {
    zyron_bench_harness::init("verify");
    tprintln!("\n=== 16 parallel writers, {PARALLEL_COMMITS} single-row commits ===");

    let mut plain_runs = Vec::with_capacity(VALIDATION_RUNS);
    let mut chained_runs = Vec::with_capacity(VALIDATION_RUNS);
    let mut serial_us_runs = Vec::with_capacity(VALIDATION_RUNS);

    for run in 0..VALIDATION_RUNS {
        let (server, _schema, _tmp) = create_test_server_with_cdc().await;
        let mut session = new_session();
        ddl(
            &server,
            &mut session,
            "CREATE TABLE plain (id BIGINT, writer BIGINT)",
        )
        .await;
        ddl(
            &server,
            &mut session,
            "ALTER TABLE plain SET (immutable = true)",
        )
        .await;
        ddl(
            &server,
            &mut session,
            "CREATE TABLE chained (id BIGINT, writer BIGINT)",
        )
        .await;
        ddl(
            &server,
            &mut session,
            "ALTER TABLE chained SET (immutable = true, verified = true)",
        )
        .await;

        // One commit into each table first, so the writers measure the
        // steady state rather than the first commit's creation of the
        // chain file, which a table pays once
        exec_dml(&server, "INSERT INTO plain VALUES (-1, -1)").await;
        exec_dml(&server, "INSERT INTO chained VALUES (-1, -1)").await;

        // The two tables take turns going first, so neither side carries
        // the cost of warming the engine on every run
        if run % 2 == 0 {
            plain_runs.push(sixteen_writers(&server, "plain").await);
            chained_runs.push(sixteen_writers(&server, "chained").await);
        } else {
            chained_runs.push(sixteen_writers(&server, "chained").await);
            plain_runs.push(sixteen_writers(&server, "plain").await);
        }
        let chain = server
            .chain_registry
            .as_ref()
            .expect("chains")
            .chain(table_id_of(&server, "chained"))
            .expect("chain");
        assert_eq!(
            chain.head().commits,
            PARALLEL_COMMITS as u64 + 1,
            "every writer's commit is on the chain"
        );

        // The serial section on its own: one thread and nothing contending,
        // so what is timed is the lock's held time, the head read, the
        // link, and the copy of the record into the file's buffer. The
        // first commit creates the file, which a table pays once, and is
        // timed on its own
        let dir = tempfile::TempDir::new().expect("temp dir");
        let alone = CommitChain::open(dir.path(), 1).expect("opens");
        let mut hasher = RowsHasher::new();
        hasher.row(1, b"one row of a realistic width, as stored");
        let count = hasher.rows();
        let rows_hash = hasher.finish();
        let link = |txn_id: u64| {
            alone
                .append(
                    verify::CommitFields {
                        txn_id,
                        rows_hash,
                        row_count: count,
                        commit_ts: txn_id as i64,
                        algorithm_id: 4,
                        genesis: false,
                    },
                    || txn_id,
                )
                .expect("links");
            alone.publish(txn_id).expect("publishes");
        };
        let creating = Instant::now();
        link(1);
        let created_us = creating.elapsed().as_secs_f64() * 1_000_000.0;
        // Every commit timed on its own as well, so a commit that paid for
        // the buffer reaching the file stands apart from the ones that
        // only copied into it
        let mut slowest_us = 0f64;
        let mut over_a_millisecond = 0usize;
        let mut in_the_slow_ones_us = 0f64;
        let started = Instant::now();
        for txn_id in 2..=PARALLEL_COMMITS as u64 + 1 {
            let one = Instant::now();
            link(txn_id);
            let took_us = one.elapsed().as_secs_f64() * 1_000_000.0;
            slowest_us = slowest_us.max(took_us);
            if took_us > 1_000.0 {
                over_a_millisecond += 1;
                in_the_slow_ones_us += took_us;
            }
        }
        let per_commit = started.elapsed().as_secs_f64() * 1_000_000.0 / PARALLEL_COMMITS as f64;
        tprintln!(
            "  serial section: {per_commit:.2} us per commit, the first commit's file creation \
             {created_us:.0} us on its own, slowest commit {slowest_us:.0} us, \
             {over_a_millisecond} commit(s) over a millisecond holding {in_the_slow_ones_us:.0} us \
             between them"
        );
        serial_us_runs.push(per_commit);
        alone.sync().expect("flushes");

        // The link's hash on its own, three ways over the same bytes, so
        // what the link costs is told apart from what a SHA-256 costs
        let sample = alone.read_range(0, 0).expect("reads").remove(0);
        let rounds = 10_000u32;
        let linking = Instant::now();
        for _ in 0..rounds {
            std::hint::black_box(sample.compute_entry_hash());
        }
        let link_ns = linking.elapsed().as_nanos() as f64 / rounds as f64;
        let mut flat = Vec::with_capacity(96);
        flat.extend_from_slice(&sample.prev_hash);
        flat.extend_from_slice(&sample.table_id.to_le_bytes());
        flat.extend_from_slice(&sample.commit_version.to_le_bytes());
        flat.extend_from_slice(&sample.rows_hash);
        flat.extend_from_slice(&sample.row_count.to_le_bytes());
        flat.extend_from_slice(&sample.commit_ts.to_le_bytes());
        flat.extend_from_slice(&sample.algorithm_id.to_le_bytes());
        flat.push(0);
        let one_call = Instant::now();
        for _ in 0..rounds {
            use sha2::Digest;
            std::hint::black_box(sha2::Sha256::digest(&flat));
        }
        let one_call_ns = one_call.elapsed().as_nanos() as f64 / rounds as f64;
        let row_style = Instant::now();
        for _ in 0..rounds {
            let mut hasher = RowsHasher::new();
            hasher.row(1, &flat);
            std::hint::black_box(hasher.finish());
        }
        let row_style_ns = row_style.elapsed().as_nanos() as f64 / rounds as f64;
        tprintln!(
            "  one link hash: {link_ns:.0} ns as the chain computes it, {one_call_ns:.0} ns as one \
             SHA-256 call over the same bytes, {row_style_ns:.0} ns through the row hasher"
        );
    }

    let plain = plain_runs.iter().sum::<f64>() / plain_runs.len() as f64;
    let chained = chained_runs.iter().sum::<f64>() / chained_runs.len() as f64;
    let delta = ((chained - plain) / plain) * 100.0;
    record_metric(
        "Parallel writers",
        "immutable only",
        " commits/s",
        plain_runs
            .iter()
            .map(|secs| PARALLEL_COMMITS as f64 / secs)
            .collect(),
    );
    record_metric(
        "Parallel writers",
        "verified",
        " commits/s",
        chained_runs
            .iter()
            .map(|secs| PARALLEL_COMMITS as f64 / secs)
            .collect(),
    );
    tprintln!(
        "  The chain adds {delta:.2}% to sixteen writers' commits (target \
         {PARALLEL_DELTA_TARGET:.0}%)"
    );
    // Both gates are evaluated and printed before either fails the test,
    // so a miss on one still reports what the other measured
    let delta_passed = check_performance(
        "Parallel writers",
        "verified vs immutable-only delta",
        delta,
        PARALLEL_DELTA_TARGET,
        false,
    );
    let serial = serial_us_runs.iter().sum::<f64>() / serial_us_runs.len() as f64;
    let serial_passed = validate_metric_with_unit(
        "Parallel writers",
        "chain serial section per commit",
        " us",
        serial_us_runs,
        SERIAL_SECTION_US_TARGET,
        false,
    )
    .passed;
    assert!(
        delta_passed,
        "sixteen writers pay {delta:.2}% for the chain, over the {PARALLEL_DELTA_TARGET:.0}% target"
    );
    assert!(
        serial_passed,
        "the serial section takes {serial:.2}us per commit, over the \
         {SERIAL_SECTION_US_TARGET:.0}us target"
    );
}

/// Sixteen writers each taking their share of the commits into one table,
/// each commit a single-row insert through the whole write path, at the
/// same time. Answers with the seconds the sixteen took between them
async fn sixteen_writers(server: &Arc<ServerState>, table: &str) -> f64 {
    let started = Instant::now();
    let mut writers = Vec::with_capacity(16);
    for writer in 0..16u64 {
        let server = Arc::clone(server);
        let table = table.to_string();
        writers.push(tokio::spawn(async move {
            for run in 0..(PARALLEL_COMMITS / 16) as u64 {
                let id = writer * 10_000 + run;
                exec_dml(
                    &server,
                    &format!("INSERT INTO {table} VALUES ({id}, {writer})"),
                )
                .await;
            }
        }));
    }
    for writer in writers {
        writer.await.expect("the writer finished");
    }
    started.elapsed().as_secs_f64()
}

/// The row hash is one SHA-256 pass, which is what the whole design rests
/// on being fast
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_row_hash_throughput() {
    zyron_bench_harness::init("verify");
    tprintln!("\n=== SHA-256 over commit rows ===");

    let row = vec![7u8; 256];
    let rows = 400_000usize;
    let bytes = (rows * row.len()) as f64;
    let mut runs = Vec::with_capacity(VALIDATION_RUNS);
    for _ in 0..VALIDATION_RUNS {
        let started = Instant::now();
        let mut hasher = RowsHasher::new();
        for _ in 0..rows {
            hasher.row(1, &row);
        }
        std::hint::black_box(hasher.finish());
        runs.push(bytes / started.elapsed().as_secs_f64());
    }
    assert!(
        validate_metric_with_unit(
            "Row hash",
            "throughput",
            " B/s",
            runs,
            ROW_HASH_BYTES_PER_SEC_TARGET,
            true,
        )
        .passed,
        "the row hash is below the {ROW_HASH_BYTES_PER_SEC_TARGET:.0} B/s target"
    );
}

// ---------------------------------------------------------------------------
// What a verification costs
// ---------------------------------------------------------------------------

/// A verified table with the prepared shape, built once per run
async fn prepared(server: &Arc<ServerState>, session: &mut Option<Session>) -> Arc<CommitChain> {
    ddl(
        server,
        session,
        "CREATE TABLE ledger (id BIGINT, amount BIGINT)",
    )
    .await;
    ddl(
        server,
        session,
        "ALTER TABLE ledger SET (immutable = true, verified = true)",
    )
    .await;
    for commit in 0..VERIFY_COMMITS {
        let values: Vec<String> = (0..VERIFY_ROWS_PER_COMMIT)
            .map(|n| {
                let id = commit * VERIFY_ROWS_PER_COMMIT + n;
                format!("({id}, {})", id * 7)
            })
            .collect();
        exec_dml(
            server,
            &format!("INSERT INTO ledger VALUES {}", values.join(", ")),
        )
        .await;
    }
    let chain = server
        .chain_registry
        .as_ref()
        .expect("chains")
        .chain(table_id_of(server, "ledger"))
        .expect("chain");
    assert_eq!(chain.head().commits, VERIFY_COMMITS as u64);
    chain
}

/// A sampled verification walks the whole chain and reads back the rows of
/// a sample, in one pass over the table
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn test_sampled_and_full_verification() {
    zyron_bench_harness::init("verify");
    tprintln!("\n=== VERIFY over {VERIFY_COMMITS} commits of {VERIFY_ROWS_PER_COMMIT} rows ===");

    let (server, _schema, _tmp) = create_test_server_with_cdc().await;
    let mut session = new_session();
    let chain = prepared(&server, &mut session).await;
    let table_id = table_id_of(&server, "ledger");
    let entries = chain
        .read_range(0, chain.head().commits - 1)
        .expect("reads");
    let bytes = entries
        .iter()
        .map(|entry| entry.row_count * 32)
        .sum::<u64>() as f64;

    // The two halves of the chain phase on their own, so a run states
    // whether the file read or the link recomputation is what it costs
    for _ in 0..3 {
        let reading = Instant::now();
        let read = chain
            .read_range(0, chain.head().commits - 1)
            .expect("reads");
        let read_us = reading.elapsed().as_micros();
        let linking = Instant::now();
        let mut agreed = 0usize;
        for entry in &read {
            if entry.compute_entry_hash() == entry.entry_hash {
                agreed += 1;
            }
        }
        let link_us = linking.elapsed().as_micros();
        tprintln!(
            "  chain phase apart: reading {} entries {read_us} us, recomputing {agreed} links \
             {link_us} us",
            read.len()
        );
    }

    let cancelled: Arc<dyn Fn() -> bool + Send + Sync> = Arc::new(|| false);
    let mut sampled_runs = Vec::with_capacity(VALIDATION_RUNS);
    let mut full_runs = Vec::with_capacity(VALIDATION_RUNS);
    for _ in 0..VALIDATION_RUNS {
        let started = Instant::now();
        let outcome = zyron_wire::verify_dispatch::run_verify(
            &server,
            zyron_wire::verify_dispatch::VerifyRequest {
                table_id,
                from_version: None,
                to_version: None,
                mode: verify::RowMode::Sampled,
                sample: 64,
            },
            0,
            "bench",
            Arc::clone(&cancelled),
        )
        .await
        .expect("verifies");
        let elapsed = started.elapsed();
        sampled_runs.push(elapsed.as_secs_f64());
        assert!(outcome.intact);
        assert_eq!(outcome.commits_checked, VERIFY_COMMITS as u64);
        tprintln!(
            "  sampled: {} us total, {} us in the chain, {} us in the rows, {} us around them",
            elapsed.as_micros(),
            outcome.chain_micros,
            outcome.rows_micros,
            (elapsed.as_micros() as u64).saturating_sub(outcome.chain_micros + outcome.rows_micros)
        );

        let started = Instant::now();
        let outcome = zyron_wire::verify_dispatch::run_verify(
            &server,
            zyron_wire::verify_dispatch::VerifyRequest {
                table_id,
                from_version: None,
                to_version: None,
                mode: verify::RowMode::All,
                sample: 0,
            },
            0,
            "bench",
            Arc::clone(&cancelled),
        )
        .await
        .expect("verifies");
        let elapsed = started.elapsed();
        assert!(outcome.intact);
        assert_eq!(
            outcome.rows_checked,
            (VERIFY_COMMITS * VERIFY_ROWS_PER_COMMIT) as u64
        );
        tprintln!(
            "  full: {} us total, {} us in the chain, {} us in the rows, {} us around them",
            elapsed.as_micros(),
            outcome.chain_micros,
            outcome.rows_micros,
            (elapsed.as_micros() as u64).saturating_sub(outcome.chain_micros + outcome.rows_micros)
        );
        full_runs.push(bytes / elapsed.as_secs_f64());
    }

    assert!(
        validate_metric_with_unit(
            "VERIFY sampled",
            "latency",
            " s",
            sampled_runs,
            SAMPLED_VERIFY_SECS_TARGET,
            false,
        )
        .passed,
        "a sampled verification is over the {SAMPLED_VERIFY_SECS_TARGET:.0}s target"
    );
    assert!(
        validate_metric_with_unit(
            "VERIFY full",
            "throughput",
            " B/s",
            full_runs,
            FULL_VERIFY_BYTES_PER_SEC_TARGET,
            true,
        )
        .passed,
        "a full verification is below the {FULL_VERIFY_BYTES_PER_SEC_TARGET:.0} B/s target"
    );
}

/// A verification takes no locks and runs at background priority, so a
/// writer keeps its own throughput while one runs
#[tokio::test(flavor = "multi_thread", worker_threads = 8)]
async fn test_writer_throughput_during_a_verification() {
    zyron_bench_harness::init("verify");
    tprintln!("\n=== Writer throughput during a full VERIFY ===");

    let (server, _schema, _tmp) = create_test_server_with_cdc().await;
    let mut session = new_session();
    let _chain = prepared(&server, &mut session).await;
    let table_id = table_id_of(&server, "ledger");
    ddl(
        &server,
        &mut session,
        "CREATE TABLE busy (id BIGINT, amount BIGINT)",
    )
    .await;

    let writes = 200usize;
    let write_batch = |at: usize| {
        let values: Vec<String> = (0..32)
            .map(|n| {
                let id = at * 32 + n;
                format!("({id}, {})", id * 3)
            })
            .collect();
        format!("INSERT INTO busy VALUES {}", values.join(", "))
    };

    let mut alone_runs = Vec::with_capacity(VALIDATION_RUNS);
    let mut during_runs = Vec::with_capacity(VALIDATION_RUNS);
    let cancelled: Arc<dyn Fn() -> bool + Send + Sync> = Arc::new(|| false);

    for run in 0..VALIDATION_RUNS {
        let started = Instant::now();
        for at in 0..writes {
            exec_dml(&server, &write_batch(run * writes * 2 + at)).await;
        }
        alone_runs.push(writes as f64 / started.elapsed().as_secs_f64());

        // The same writes with a full verification running beside them
        let verifying = {
            let server = Arc::clone(&server);
            let cancelled = Arc::clone(&cancelled);
            tokio::spawn(async move {
                zyron_wire::verify_dispatch::run_verify(
                    &server,
                    zyron_wire::verify_dispatch::VerifyRequest {
                        table_id,
                        from_version: None,
                        to_version: None,
                        mode: verify::RowMode::All,
                        sample: 0,
                    },
                    0,
                    "bench",
                    cancelled,
                )
                .await
            })
        };
        let started = Instant::now();
        for at in 0..writes {
            exec_dml(&server, &write_batch(run * writes * 2 + writes + at)).await;
        }
        during_runs.push(writes as f64 / started.elapsed().as_secs_f64());
        let outcome = verifying.await.expect("the task ran").expect("verifies");
        assert!(outcome.intact);
    }

    let alone = alone_runs.iter().sum::<f64>() / alone_runs.len() as f64;
    let during = during_runs.iter().sum::<f64>() / during_runs.len() as f64;
    let delta = ((alone - during) / alone) * 100.0;
    record_metric("Writer during VERIFY", "alone", " commits/s", alone_runs);
    record_metric(
        "Writer during VERIFY",
        "with a verification running",
        " commits/s",
        during_runs,
    );
    tprintln!(
        "  A running verification costs the writer {delta:.2}% (target {WRITER_DELTA_TARGET:.0}%)"
    );
    assert!(
        check_performance(
            "Writer during VERIFY",
            "throughput delta",
            delta,
            WRITER_DELTA_TARGET,
            false,
        ),
        "a running verification costs the writer {delta:.2}%, over the \
         {WRITER_DELTA_TARGET:.0}% target"
    );
}

// ---------------------------------------------------------------------------
// Anchoring and storage
// ---------------------------------------------------------------------------

/// One anchor is a sync of the chain, a record of the head and one audit
/// append
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn test_anchor_write_latency() {
    zyron_bench_harness::init("verify");
    tprintln!("\n=== Anchor write ===");

    let (server, _schema, _tmp) = create_test_server_with_cdc().await;
    let mut session = new_session();
    ddl(
        &server,
        &mut session,
        "CREATE TABLE ledger (id BIGINT, amount BIGINT)",
    )
    .await;
    ddl(
        &server,
        &mut session,
        "ALTER TABLE ledger SET (immutable = true, verified = true)",
    )
    .await;
    let table_id = table_id_of(&server, "ledger");

    let mut runs = Vec::with_capacity(VALIDATION_RUNS);
    for run in 0..VALIDATION_RUNS {
        // A commit between anchors, so each anchor has a head that moved
        exec_dml(
            &server,
            &format!("INSERT INTO ledger VALUES ({run}, {run})"),
        )
        .await;
        let started = Instant::now();
        zyron_wire::verify_dispatch::anchor_table(&server, table_id)
            .await
            .expect("anchors")
            .expect("anchored");
        runs.push(started.elapsed().as_secs_f64() * 1_000.0);
    }
    assert!(
        validate_metric_with_unit(
            "Anchor write",
            "latency",
            " ms",
            runs,
            ANCHOR_MS_TARGET,
            false,
        )
        .passed,
        "an anchor is over the {ANCHOR_MS_TARGET:.0}ms target"
    );
}

/// A chain of n commits is exactly n fixed records long, which is what the
/// documented per-commit figure rests on
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_chain_storage_per_commit() {
    zyron_bench_harness::init("verify");
    tprintln!("\n=== Chain storage per commit ===");

    let mut runs = Vec::with_capacity(VALIDATION_RUNS);
    for run in 0..VALIDATION_RUNS {
        let dir = tempfile::TempDir::new().expect("temp dir");
        let chain = CommitChain::open(dir.path(), 1).expect("opens");
        let commits = 500u64 + run as u64;
        let mut hasher = RowsHasher::new();
        hasher.row(1, b"one row");
        let rows_hash = hasher.finish();
        for n in 1..=commits {
            chain
                .append(
                    verify::CommitFields {
                        txn_id: n,
                        rows_hash,
                        row_count: 1,
                        commit_ts: n as i64,
                        algorithm_id: 4,
                        genesis: false,
                    },
                    || n,
                )
                .expect("links");
            chain.publish(n).expect("publishes");
        }
        chain.sync().expect("flushes");

        // The envelope header is one-time, so what a commit costs is read
        // off the body rather than off the whole file
        let length = std::fs::metadata(chain.path()).expect("measures").len();
        let body = length - 20;
        runs.push(body as f64 / commits as f64);
        assert_eq!(
            chain.head().bytes(),
            commits * verify::CHAIN_RECORD_LEN as u64,
            "the view reports the same figure the file holds"
        );
    }

    let result = validate_metric_with_unit(
        "Chain storage",
        "bytes per commit",
        " B",
        runs,
        CHAIN_BYTES_PER_COMMIT_TARGET,
        false,
    );
    assert!(
        result.passed,
        "a chain entry is not the documented {CHAIN_BYTES_PER_COMMIT_TARGET:.0} bytes"
    );
    assert_eq!(
        verify::CHAIN_RECORD_LEN as f64,
        CHAIN_BYTES_PER_COMMIT_TARGET,
        "the record the code writes is the figure the documentation states"
    );
}
