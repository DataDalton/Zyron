//! The writable `zyron_sys.cost.currency_rates` system table.
//!
//! Rates live in a checksummed file under the data directory and in the
//! process wide rate store `CONVERT_CURRENCY` reads. INSERT INTO the
//! canonical three part name upserts rows, DELETE without a predicate
//! clears the table, and the registered view reads it back. Every write
//! rewrites the file with an fsync before it answers, so a crash never
//! loses an acknowledged rate.

use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use zyron_common::ZyronError;
use zyron_parser::ast::{Expr, InsertSource, InsertStatement, LiteralValue};
use zyron_types::money::{CurrencyRate, currency_rate_store};

use crate::connection::ServerState;
use crate::ddl_dispatch::DdlResult;
use crate::messages::ProtocolError;
use crate::system_views::{ViewRows, make_field};
use crate::types::{PG_FLOAT8_OID, PG_TEXT_OID};

pub const CURRENCY_RATES_TABLE: &str = "zyron_sys.cost.currency_rates";

const FILE_MAGIC: &[u8; 8] = b"ZYCRATE\0";
const FILE_VERSION: u32 = 1;

fn rates_path(data_dir: &Path) -> PathBuf {
    data_dir.join("currency_rates.zyrates")
}

/// Serialized row layout inside the file body
#[derive(serde::Serialize, serde::Deserialize)]
struct StoredRate {
    from: String,
    to: String,
    rate_date_days: i32,
    rate: f64,
}

fn read_rates_file(data_dir: &Path) -> Result<Vec<CurrencyRate>, ZyronError> {
    let path = rates_path(data_dir);
    let bytes = match std::fs::read(&path) {
        Ok(b) => b,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(Vec::new()),
        Err(e) => {
            return Err(ZyronError::Internal(format!(
                "currency rates file {} unreadable: {e}",
                path.display()
            )));
        }
    };
    if bytes.len() < 16 || &bytes[0..8] != FILE_MAGIC {
        return Err(ZyronError::Internal(format!(
            "currency rates file {} has a bad header",
            path.display()
        )));
    }
    let version = u32::from_le_bytes([bytes[8], bytes[9], bytes[10], bytes[11]]);
    if version != FILE_VERSION {
        return Err(ZyronError::Internal(format!(
            "currency rates file version {version} is not readable by this build"
        )));
    }
    let stored_crc = u32::from_le_bytes([bytes[12], bytes[13], bytes[14], bytes[15]]);
    let body = &bytes[16..];
    if crc32fast::hash(body) != stored_crc {
        return Err(ZyronError::Internal(format!(
            "currency rates file {} failed its checksum",
            path.display()
        )));
    }
    let rows: Vec<StoredRate> = serde_json::from_slice(body)
        .map_err(|e| ZyronError::Internal(format!("currency rates file did not decode: {e}")))?;
    Ok(rows
        .into_iter()
        .map(|r| CurrencyRate {
            from: r.from,
            to: r.to,
            rate_date_days: r.rate_date_days,
            rate: r.rate,
        })
        .collect())
}

fn write_rates_file(data_dir: &Path, rates: &[CurrencyRate]) -> Result<(), ZyronError> {
    let rows: Vec<StoredRate> = rates
        .iter()
        .map(|r| StoredRate {
            from: r.from.clone(),
            to: r.to.clone(),
            rate_date_days: r.rate_date_days,
            rate: r.rate,
        })
        .collect();
    let body = serde_json::to_vec(&rows)
        .map_err(|e| ZyronError::Internal(format!("currency rates did not encode: {e}")))?;
    let mut bytes = Vec::with_capacity(16 + body.len());
    bytes.extend_from_slice(FILE_MAGIC);
    bytes.extend_from_slice(&FILE_VERSION.to_le_bytes());
    bytes.extend_from_slice(&crc32fast::hash(&body).to_le_bytes());
    bytes.extend_from_slice(&body);
    let path = rates_path(data_dir);
    let tmp = path.with_extension("zyrates.tmp");
    let io_err =
        |e: std::io::Error| ZyronError::Internal(format!("currency rates file write failed: {e}"));
    {
        let mut file = std::fs::File::create(&tmp).map_err(io_err)?;
        file.write_all(&bytes).map_err(io_err)?;
        file.sync_all().map_err(io_err)?;
    }
    std::fs::rename(&tmp, &path).map_err(io_err)?;
    Ok(())
}

/// Loads the persisted rates into the process store. Called at startup
pub fn load_into_store(data_dir: &Path) -> Result<usize, ZyronError> {
    let rates = read_rates_file(data_dir)?;
    let count = rates.len();
    currency_rate_store().replace_all(rates);
    Ok(count)
}

/// Whether a statement addresses the currency rates table. Only the
/// canonical three part name reaches it, matching the rest of zyron_sys
pub fn targets_currency_rates(table: &str) -> bool {
    table.eq_ignore_ascii_case(CURRENCY_RATES_TABLE)
}

fn literal_of(expr: &Expr) -> Result<&LiteralValue, ProtocolError> {
    match expr {
        Expr::Literal(v) => Ok(v),
        other => Err(ProtocolError::Database(ZyronError::ExecutionError(
            format!("currency rate values must be literals, got {other:?}"),
        ))),
    }
}

fn text_of(value: &LiteralValue, what: &str) -> Result<String, ProtocolError> {
    match value {
        LiteralValue::String(s) => Ok(s.clone()),
        other => Err(ProtocolError::Database(ZyronError::ExecutionError(
            format!("{what} must be a string, got {other:?}"),
        ))),
    }
}

fn number_of(value: &LiteralValue, what: &str) -> Result<f64, ProtocolError> {
    match value {
        LiteralValue::Float(f) => Ok(*f),
        LiteralValue::Integer(n) => Ok(*n as f64),
        other => Err(ProtocolError::Database(ZyronError::ExecutionError(
            format!("{what} must be a number, got {other:?}"),
        ))),
    }
}

/// Civil date to days since the epoch, Howard Hinnant's algorithm
fn days_from_civil(y: i32, m: u32, d: u32) -> i32 {
    let y = y - if m <= 2 { 1 } else { 0 };
    let era = if y >= 0 { y } else { y - 399 } / 400;
    let yoe = y - era * 400;
    let mp = (m as i32 + 9) % 12;
    let doy = (153 * mp + 2) / 5 + d as i32 - 1;
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    era * 146_097 + doe - 719_468
}

fn civil_from_days(z: i32) -> (i32, u32, u32) {
    let z = z + 719_468;
    let era = if z >= 0 { z } else { z - 146_096 } / 146_097;
    let doe = z - era * 146_097;
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = (doy - (153 * mp + 2) / 5 + 1) as u32;
    let m = if mp < 10 { mp + 3 } else { mp - 9 } as u32;
    (y + if m <= 2 { 1 } else { 0 }, m, d)
}

fn parse_date_days(text: &str) -> Result<i32, ProtocolError> {
    let mut parts = text.trim().splitn(3, '-');
    let bad = || {
        ProtocolError::Database(ZyronError::ExecutionError(format!(
            "rate_date must be YYYY-MM-DD, got {text}"
        )))
    };
    let y: i32 = parts.next().and_then(|p| p.parse().ok()).ok_or_else(bad)?;
    let m: u32 = parts.next().and_then(|p| p.parse().ok()).ok_or_else(bad)?;
    let d: u32 = parts.next().and_then(|p| p.parse().ok()).ok_or_else(bad)?;
    if !(1..=12).contains(&m) || !(1..=31).contains(&d) {
        return Err(bad());
    }
    Ok(days_from_civil(y, m, d))
}

const COLUMNS: [&str; 4] = ["from_currency", "to_currency", "rate_date", "rate"];

/// INSERT INTO zyron_sys.cost.currency_rates: upserts (from, to, date)
/// rows, persists the file, and refreshes the process store
pub async fn handle_insert(
    stmt: &InsertStatement,
    server: &Arc<ServerState>,
) -> Result<DdlResult, ProtocolError> {
    let InsertSource::Values(rows) = &stmt.source else {
        return Err(ProtocolError::Database(ZyronError::ExecutionError(
            "zyron_sys.cost.currency_rates accepts INSERT ... VALUES only".to_string(),
        )));
    };
    // Column order defaults to (from_currency, to_currency, rate_date, rate)
    let order: Vec<usize> = if stmt.columns.is_empty() {
        (0..4).collect()
    } else {
        let mut order = Vec::with_capacity(4);
        for wanted in COLUMNS {
            let idx = stmt
                .columns
                .iter()
                .position(|c| c.eq_ignore_ascii_case(wanted))
                .ok_or_else(|| {
                    ProtocolError::Database(ZyronError::ExecutionError(format!(
                        "currency rates INSERT must name all of {}",
                        COLUMNS.join(", ")
                    )))
                })?;
            order.push(idx);
        }
        order
    };

    let mut incoming = Vec::with_capacity(rows.len());
    for row in rows {
        if row.len() != 4 {
            return Err(ProtocolError::Database(ZyronError::ExecutionError(
                "each currency rate row has four values: from_currency, to_currency, rate_date, rate".to_string(),
            )));
        }
        let from = text_of(literal_of(&row[order[0]])?, "from_currency")?.to_uppercase();
        let to = text_of(literal_of(&row[order[1]])?, "to_currency")?.to_uppercase();
        let date_days = parse_date_days(&text_of(literal_of(&row[order[2]])?, "rate_date")?)?;
        let rate = number_of(literal_of(&row[order[3]])?, "rate")?;
        if from.len() != 3 || to.len() != 3 {
            return Err(ProtocolError::Database(ZyronError::ExecutionError(
                "currency codes are three letters".to_string(),
            )));
        }
        if !(rate.is_finite() && rate > 0.0) {
            return Err(ProtocolError::Database(ZyronError::ExecutionError(
                "rate must be a positive number".to_string(),
            )));
        }
        incoming.push(CurrencyRate {
            from,
            to,
            rate_date_days: date_days,
            rate,
        });
    }

    let mut all = read_rates_file(&server.data_dir).map_err(ProtocolError::Database)?;
    let inserted = incoming.len();
    for new_rate in incoming {
        match all.iter_mut().find(|r| {
            r.from == new_rate.from
                && r.to == new_rate.to
                && r.rate_date_days == new_rate.rate_date_days
        }) {
            Some(existing) => existing.rate = new_rate.rate,
            None => all.push(new_rate),
        }
    }
    write_rates_file(&server.data_dir, &all).map_err(ProtocolError::Database)?;
    currency_rate_store().replace_all(all);
    Ok(DdlResult::Tag(format!("INSERT 0 {inserted}")))
}

/// DELETE FROM zyron_sys.cost.currency_rates clears every rate. A
/// predicate is refused so a partial delete never silently half applies
pub async fn handle_delete(
    where_clause: &Option<Box<Expr>>,
    server: &Arc<ServerState>,
) -> Result<DdlResult, ProtocolError> {
    if where_clause.is_some() {
        return Err(ProtocolError::Database(ZyronError::ExecutionError(
            "zyron_sys.cost.currency_rates supports DELETE of the whole table only".to_string(),
        )));
    }
    let existing = read_rates_file(&server.data_dir).map_err(ProtocolError::Database)?;
    let removed = existing.len();
    write_rates_file(&server.data_dir, &[]).map_err(ProtocolError::Database)?;
    currency_rate_store().replace_all(Vec::new());
    Ok(DdlResult::Tag(format!("DELETE {removed}")))
}

/// The registered view over the persisted rates
pub fn build_view(server: &ServerState) -> Result<ViewRows, ZyronError> {
    let fields = vec![
        make_field("from_currency", PG_TEXT_OID, -1),
        make_field("to_currency", PG_TEXT_OID, -1),
        make_field("rate_date", PG_TEXT_OID, -1),
        make_field("rate", PG_FLOAT8_OID, 8),
    ];
    let mut rates = read_rates_file(&server.data_dir)?;
    rates.sort_by(|a, b| {
        (&a.from, &a.to, a.rate_date_days).cmp(&(&b.from, &b.to, b.rate_date_days))
    });
    let rows = rates
        .into_iter()
        .map(|r| {
            let (y, m, d) = civil_from_days(r.rate_date_days);
            vec![
                Some(r.from.into_bytes()),
                Some(r.to.into_bytes()),
                Some(format!("{y:04}-{m:02}-{d:02}").into_bytes()),
                Some(format!("{}", r.rate).into_bytes()),
            ]
        })
        .collect();
    Ok((fields, rows))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_civil_date_round_trip() {
        for (y, m, d) in [(1970, 1, 1), (2026, 8, 30), (2000, 2, 29), (1969, 12, 31)] {
            let days = days_from_civil(y, m, d);
            assert_eq!(civil_from_days(days), (y, m, d));
        }
        assert_eq!(days_from_civil(1970, 1, 1), 0);
    }

    #[test]
    fn test_rates_file_round_trip_and_checksum() {
        let dir = std::env::temp_dir().join(format!("zyron_rates_{}", std::process::id()));
        let _ = std::fs::create_dir_all(&dir);
        let rates = vec![CurrencyRate {
            from: "USD".to_string(),
            to: "EUR".to_string(),
            rate_date_days: 20_000,
            rate: 0.91,
        }];
        write_rates_file(&dir, &rates).expect("write");
        let back = read_rates_file(&dir).expect("read");
        assert_eq!(back.len(), 1);
        assert_eq!(back[0].from, "USD");
        // A flipped byte in the body fails the checksum
        let path = rates_path(&dir);
        let mut bytes = std::fs::read(&path).expect("raw");
        let last = bytes.len() - 1;
        bytes[last] ^= 0xff;
        std::fs::write(&path, bytes).expect("tamper");
        assert!(read_rates_file(&dir).is_err());
        let _ = std::fs::remove_dir_all(&dir);
    }
}
