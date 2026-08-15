#![allow(non_snake_case)]
//! TPC-H and TPC-C benchmark drivers.
//!
//! The schema, the data and the workload come from `zyron-tpc`, which is a
//! library so the same definitions can be driven in process by a test. What
//! lives here is the driving: loading over the wire, timing, and reporting.
//!
//! A query the server refuses is reported by name and the run continues. A
//! driver that stopped at the first refusal would report a partial result as
//! a complete one, and a driver that skipped the query would report a number
//! for a workload nobody asked for.

use std::time::Instant;

use super::remote::RemoteClient;
use zyron_tpc::{QueryOutcome, Rng, tpcc, tpch};

/// Rows per INSERT during a load. Large enough that the round trip is
/// amortized, small enough that one statement stays a reasonable size.
const LOAD_BATCH_ROWS: usize = 500;

/// Loads a schema and streams generated data through the client.
fn load(
    client: &mut RemoteClient,
    schema: Vec<&'static str>,
    generate: impl FnOnce(&mut dyn FnMut(&str) -> Result<(), String>) -> Result<(), String>,
) -> Result<(u64, f64), String> {
    for ddl in schema {
        client
            .execute(ddl)
            .map_err(|e| format!("schema creation failed: {e}"))?;
    }

    let started = Instant::now();
    let mut statements: u64 = 0;
    let mut sink = |sql: &str| -> Result<(), String> {
        client
            .execute(sql)
            .map(|_| ())
            .map_err(|e| format!("load failed after {statements} statements: {e}"))?;
        statements += 1;
        if statements % 200 == 0 {
            println!("  loaded {statements} batches...");
        }
        Ok(())
    };
    generate(&mut sink)?;
    Ok((statements, started.elapsed().as_secs_f64()))
}

/// Runs one statement and times it, keeping a refusal rather than raising it.
fn timed(client: &mut RemoteClient, name: &str, sql: &str) -> QueryOutcome {
    let started = Instant::now();
    match client.execute(sql) {
        Ok(result) => QueryOutcome {
            name: name.to_string(),
            elapsed_micros: started.elapsed().as_micros(),
            rows: result.rows.len(),
            error: None,
        },
        Err(e) => QueryOutcome {
            name: name.to_string(),
            elapsed_micros: started.elapsed().as_micros(),
            rows: 0,
            error: Some(e),
        },
    }
}

fn reportOutcomes(outcomes: &[QueryOutcome]) -> Result<(), String> {
    println!();
    println!(
        "  {:<8} {:>14} {:>10}  {}",
        "query", "elapsed (ms)", "rows", "status"
    );
    println!("  {}", "-".repeat(52));
    let mut refused = 0usize;
    let mut total_micros: u128 = 0;
    for o in outcomes {
        total_micros += o.elapsed_micros;
        match &o.error {
            None => println!(
                "  {:<8} {:>14.3} {:>10}  ok",
                o.name,
                o.elapsed_micros as f64 / 1000.0,
                o.rows
            ),
            Some(e) => {
                refused += 1;
                println!(
                    "  {:<8} {:>14.3} {:>10}  REFUSED: {}",
                    o.name,
                    o.elapsed_micros as f64 / 1000.0,
                    0,
                    e
                );
            }
        }
    }
    println!("  {}", "-".repeat(52));
    println!(
        "  {} of {} completed in {:.3} ms total",
        outcomes.len() - refused,
        outcomes.len(),
        total_micros as f64 / 1000.0
    );
    if refused > 0 {
        return Err(format!(
            "{refused} of {} statements were refused, see the table above",
            outcomes.len()
        ));
    }
    Ok(())
}

/// Creates the TPC-H schema, loads a dataset at the given scale factor, and
/// runs all 22 queries.
pub fn runTpch(client: &mut RemoteClient, scale: f64) -> Result<(), String> {
    println!("TPC-H benchmark at scale factor {scale}");
    println!("Rows to load:");
    for (table, rows) in tpch::row_counts(scale) {
        println!("  {table:<10} {rows:>12}");
    }

    println!("Loading...");
    let (statements, seconds) = load(client, tpch::schema(), |sink| {
        tpch::generate(scale, LOAD_BATCH_ROWS, sink)
    })?;
    println!("  {statements} batches in {seconds:.2}s");

    println!("Running the 22 queries...");
    let outcomes: Vec<QueryOutcome> = tpch::queries()
        .into_iter()
        .map(|(name, sql)| timed(client, name, sql))
        .collect();
    reportOutcomes(&outcomes)
}

/// Creates the TPC-C schema, loads the warehouses the scale factor asks for,
/// and runs the five-transaction mix.
///
/// The reported rate counts New-Order transactions per minute, which is the
/// benchmark's metric, measured over the whole mix rather than over
/// New-Order alone.
pub fn runTpcc(client: &mut RemoteClient, scale: f64) -> Result<(), String> {
    let warehouses = tpcc::warehouses_for(scale);
    println!("TPC-C benchmark at {warehouses} warehouse(s)");
    println!("Rows to load:");
    for (table, rows) in tpcc::row_counts(warehouses) {
        println!("  {table:<12} {rows:>12}");
    }

    println!("Loading...");
    let (statements, seconds) = load(client, tpcc::schema(), |sink| {
        tpcc::generate(scale, LOAD_BATCH_ROWS, sink)
    })?;
    println!("  {statements} batches in {seconds:.2}s");

    // Enough transactions to cover the mix's rarest member many times over
    const TRANSACTIONS: usize = 1_000;
    println!("Running {TRANSACTIONS} transactions...");

    let mut rng = Rng::new(0x5450_4343);
    let mut per_kind: Vec<(tpcc::Transaction, u64, u128, u64)> = vec![
        (tpcc::Transaction::NewOrder, 0, 0, 0),
        (tpcc::Transaction::Payment, 0, 0, 0),
        (tpcc::Transaction::OrderStatus, 0, 0, 0),
        (tpcc::Transaction::Delivery, 0, 0, 0),
        (tpcc::Transaction::StockLevel, 0, 0, 0),
    ];
    let started = Instant::now();
    let mut firstError: Option<String> = None;

    for _ in 0..TRANSACTIONS {
        let kind = tpcc::next_transaction(&mut rng);
        let statements = tpcc::statements(kind, warehouses, &mut rng);
        let txnStart = Instant::now();
        // The specification requires a transaction's statements to be one
        // atomic unit, so measuring them autocommitted would measure
        // something else
        let mut failed = None;
        if let Err(e) = client.execute("BEGIN") {
            failed = Some(e);
        } else {
            for sql in &statements {
                if let Err(e) = client.execute(sql) {
                    failed = Some(e);
                    break;
                }
            }
        }
        let outcome = match failed {
            None => client.execute("COMMIT").map(|_| ()).map_err(|e| e),
            Some(e) => {
                let _ = client.execute("ROLLBACK");
                Err(e)
            }
        };
        let elapsed = txnStart.elapsed().as_micros();

        let slot = per_kind
            .iter_mut()
            .find(|(k, _, _, _)| *k == kind)
            .ok_or_else(|| format!("no counter for {}", kind.name()))?;
        slot.1 += 1;
        slot.2 += elapsed;
        if let Err(e) = outcome {
            slot.3 += 1;
            if firstError.is_none() {
                firstError = Some(format!("{}: {e}", kind.name()));
            }
        }
    }
    let seconds = started.elapsed().as_secs_f64();

    println!();
    println!(
        "  {:<14} {:>8} {:>14} {:>10}",
        "transaction", "count", "avg (ms)", "failed"
    );
    println!("  {}", "-".repeat(50));
    let mut newOrders = 0u64;
    let mut failures = 0u64;
    for (kind, count, micros, failed) in &per_kind {
        if *count == 0 {
            continue;
        }
        println!(
            "  {:<14} {:>8} {:>14.3} {:>10}",
            kind.name(),
            count,
            (*micros as f64 / *count as f64) / 1000.0,
            failed
        );
        if *kind == tpcc::Transaction::NewOrder {
            newOrders = count - failed;
        }
        failures += failed;
    }
    println!("  {}", "-".repeat(50));
    let tpmc = if seconds > 0.0 {
        newOrders as f64 * 60.0 / seconds
    } else {
        0.0
    };
    println!("  {TRANSACTIONS} transactions in {seconds:.2}s");
    println!("  {tpmc:.1} tpmC (New-Order transactions per minute)");

    if failures > 0 {
        return Err(format!(
            "{failures} of {TRANSACTIONS} transactions failed, first: {}",
            firstError.unwrap_or_else(|| "unknown".to_string())
        ));
    }
    Ok(())
}
