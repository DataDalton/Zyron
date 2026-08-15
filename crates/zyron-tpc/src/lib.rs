//! TPC-H and TPC-C schemas, data generation and workloads.
//!
//! The generators stream: a caller receives batched SQL and never holds the
//! dataset, so a scale factor is bounded by the server's storage rather than
//! by the generator's memory. Every value is derived from a seeded counter,
//! so the same scale factor produces the same database on every run and on
//! every machine, which is what makes two measurements comparable.
//!
//! This lives in a library rather than in the CLI so the same schema, data
//! and queries can be driven in process by a test, which is what proves the
//! queries execute rather than merely parse.

pub mod tpcc;
pub mod tpch;

/// Deterministic pseudo-random source.
///
/// A 64-bit linear congruential generator with the constants Knuth lists for
/// MMIX, taking the high bits, which are the well-distributed ones. Seeded
/// per stream so each column's values are reproducible independently of how
/// many rows another column drew.
#[derive(Clone)]
pub struct Rng {
    state: u64,
}

impl Rng {
    pub fn new(seed: u64) -> Self {
        // A zero seed would leave an all-zero state, so it is displaced
        Self {
            state: seed.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1),
        }
    }

    #[inline]
    pub fn next_u64(&mut self) -> u64 {
        self.state = self
            .state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        // The high 32 bits carry the period, the low bits cycle short
        (self.state >> 32) ^ (self.state << 16)
    }

    /// Uniform in `[low, high]`, inclusive at both ends
    #[inline]
    pub fn range(&mut self, low: i64, high: i64) -> i64 {
        if high <= low {
            return low;
        }
        let span = (high - low + 1) as u64;
        low + (self.next_u64() % span) as i64
    }

    /// Uniform over a slice, which every fixed vocabulary in the specs uses
    #[inline]
    pub fn pick<'a, T>(&mut self, items: &'a [T]) -> &'a T {
        &items[(self.next_u64() as usize) % items.len()]
    }

    /// A fixed-point value in `[low, high]` with two decimal places, the
    /// shape every money column in both specs takes
    pub fn money(&mut self, low: i64, high: i64) -> String {
        let cents = self.range(low * 100, high * 100);
        format_cents(cents)
    }
}

/// Renders a signed cent count as a decimal string with two places, without
/// going through a float, so a value round-trips exactly.
pub fn format_cents(cents: i64) -> String {
    let sign = if cents < 0 { "-" } else { "" };
    let abs = cents.unsigned_abs();
    format!("{}{}.{:02}", sign, abs / 100, abs % 100)
}

/// Escapes a string for a single-quoted SQL literal.
pub fn quote(s: &str) -> String {
    if s.contains('\'') {
        format!("'{}'", s.replace('\'', "''"))
    } else {
        format!("'{s}'")
    }
}

/// Accumulates rows into multi-row INSERT statements and hands each finished
/// statement to the sink.
///
/// Batching is what keeps loading from paying one round trip per row. The
/// batch is flushed by row count rather than by byte length so a caller can
/// reason about the statement size from the row width it knows.
pub struct InsertBatcher<'a> {
    prefix: String,
    buf: String,
    rows: usize,
    rows_per_statement: usize,
    sink: &'a mut dyn FnMut(&str) -> Result<(), String>,
}

impl<'a> InsertBatcher<'a> {
    pub fn new(
        table: &str,
        rows_per_statement: usize,
        sink: &'a mut dyn FnMut(&str) -> Result<(), String>,
    ) -> Self {
        Self {
            prefix: format!("INSERT INTO {table} VALUES "),
            buf: String::with_capacity(64 * 1024),
            rows: 0,
            rows_per_statement: rows_per_statement.max(1),
            sink,
        }
    }

    /// Adds one row, already rendered as `(v1, v2, ...)` without the comma
    pub fn push(&mut self, row: &str) -> Result<(), String> {
        if self.rows == 0 {
            self.buf.push_str(&self.prefix);
        } else {
            self.buf.push_str(", ");
        }
        self.buf.push_str(row);
        self.rows += 1;
        if self.rows >= self.rows_per_statement {
            self.flush()?;
        }
        Ok(())
    }

    /// Emits whatever is buffered. Safe to call when nothing is
    pub fn flush(&mut self) -> Result<(), String> {
        if self.rows == 0 {
            return Ok(());
        }
        (self.sink)(&self.buf)?;
        self.buf.clear();
        self.rows = 0;
        Ok(())
    }
}

/// Days elapsed from 1970-01-01 to a proleptic Gregorian date, so generated
/// dates can be produced by arithmetic and rendered back without a calendar
/// dependency.
pub fn days_from_civil(year: i64, month: u32, day: u32) -> i64 {
    let y = if month <= 2 { year - 1 } else { year };
    let era = if y >= 0 { y } else { y - 399 } / 400;
    let yoe = y - era * 400;
    let m = month as i64;
    let d = day as i64;
    let doy = (153 * (if m > 2 { m - 3 } else { m + 9 }) + 2) / 5 + d - 1;
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    era * 146_097 + doe - 719_468
}

/// The inverse of `days_from_civil`, rendered as `YYYY-MM-DD`
pub fn civil_from_days(days: i64) -> String {
    let z = days + 719_468;
    let era = if z >= 0 { z } else { z - 146_096 } / 146_097;
    let doe = z - era * 146_097;
    let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146_096) / 365;
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    let year = if m <= 2 { y + 1 } else { y };
    format!("{year:04}-{m:02}-{d:02}")
}

/// One query's outcome, so a driver can report a failure without abandoning
/// the rest of the run. A query the engine refuses is a finding about the
/// SQL surface, not a reason to stop measuring the others.
pub struct QueryOutcome {
    pub name: String,
    pub elapsed_micros: u128,
    pub rows: usize,
    pub error: Option<String>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_a_seeded_stream_repeats_exactly() {
        let a: Vec<u64> = Rng::new(42).into_iter_values(16);
        let b: Vec<u64> = Rng::new(42).into_iter_values(16);
        assert_eq!(a, b, "the same seed produces the same stream");
        let c: Vec<u64> = Rng::new(43).into_iter_values(16);
        assert_ne!(a, c, "a different seed produces a different stream");
    }

    #[test]
    fn test_range_stays_inside_its_bounds_and_covers_them() {
        let mut rng = Rng::new(7);
        let mut low_seen = false;
        let mut high_seen = false;
        for _ in 0..10_000 {
            let v = rng.range(1, 5);
            assert!((1..=5).contains(&v), "range escaped its bounds: {v}");
            low_seen |= v == 1;
            high_seen |= v == 5;
        }
        assert!(
            low_seen && high_seen,
            "both ends of the range are reachable"
        );
    }

    #[test]
    fn test_cents_render_without_a_float() {
        assert_eq!(format_cents(0), "0.00");
        assert_eq!(format_cents(5), "0.05");
        assert_eq!(format_cents(1234), "12.34");
        assert_eq!(format_cents(-1234), "-12.34");
        assert_eq!(format_cents(100_000), "1000.00");
    }

    #[test]
    fn test_dates_round_trip_through_the_epoch_day_count() {
        for (y, m, d) in [
            (1970, 1, 1),
            (1992, 1, 1),
            (1998, 12, 1),
            (2000, 2, 29),
            (2026, 8, 13),
        ] {
            let days = days_from_civil(y, m, d);
            assert_eq!(civil_from_days(days), format!("{y:04}-{m:02}-{d:02}"));
        }
    }

    #[test]
    fn test_quote_escapes_an_embedded_apostrophe() {
        assert_eq!(quote("plain"), "'plain'");
        assert_eq!(quote("it's"), "'it''s'");
    }

    #[test]
    fn test_the_batcher_splits_at_the_row_count_and_flushes_the_remainder() {
        let mut statements: Vec<String> = Vec::new();
        {
            let mut sink = |sql: &str| {
                statements.push(sql.to_string());
                Ok(())
            };
            let mut b = InsertBatcher::new("t", 2, &mut sink);
            for i in 0..5 {
                b.push(&format!("({i})")).unwrap();
            }
            b.flush().unwrap();
        }
        assert_eq!(statements.len(), 3, "five rows at two per statement");
        assert_eq!(statements[0], "INSERT INTO t VALUES (0), (1)");
        assert_eq!(
            statements[2], "INSERT INTO t VALUES (4)",
            "remainder flushed"
        );
    }

    impl Rng {
        fn into_iter_values(mut self, n: usize) -> Vec<u64> {
            (0..n).map(|_| self.next_u64()).collect()
        }
    }
}
