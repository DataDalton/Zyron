//! Cron, range, and business-time function dispatch for the types bridge
//!
//! Cron expressions cross the boundary as cron text or as the 19 byte cell
//! produced by cron_parse (minute u64 BE, hour u32 BE, day_of_month u32 BE,
//! month u16 BE, day_of_week u8). CronExpr has no to_bytes or Serialize form
//! of its own so that fixed layout is defined here and accepted back on input
//!
//! Range cells use the zyron_types::range byte layout with a fixed 8 byte
//! element size. Integer bounds are encoded big-endian with the sign bit
//! flipped so byte comparison matches numeric order
//!
//! Dates are i32 days since 1970-01-01, timestamps are epoch microseconds,
//! holiday lists are JSON integer arrays per the Array cell encoding

use crate::column::{Column, ColumnData, NullBitmap};
use zyron_common::{Interval, Result, TypeId, ZyronError};
use zyron_types::cron::CronExpr;

pub(super) fn dispatch(name: &str, args: &[Column], num_rows: usize) -> Option<Result<Column>> {
    // every function in this family takes at least one argument
    let _ = num_rows;
    Some(match name {
        // ---------- cron ----------
        "cron_parse" => cron_parse_impl(args),
        "cron_next" => cron_next_impl(args),
        "cron_prev" => cron_prev_impl(args),
        "cron_matches" => cron_matches_impl(args),
        "cron_between" => cron_between_impl(args),
        "cron_human_readable" => cron_human_readable_impl(args),
        "cron_list" => cron_list_impl(args),

        // ---------- range ----------
        "range_create" => range_create_impl(args),
        "range_is_empty" => range_flag_impl(args, "range_is_empty(range)", |r| {
            zyron_types::range::range_is_empty(r)
        }),
        "range_lower" => range_bound_impl(args, "range_lower(range)", |r| {
            zyron_types::range::range_lower(r, RANGE_ELEM)
        }),
        "range_upper" => range_bound_impl(args, "range_upper(range)", |r| {
            zyron_types::range::range_upper(r, RANGE_ELEM)
        }),
        "range_lower_inclusive" => range_flag_impl(args, "range_lower_inclusive(range)", |r| {
            zyron_types::range::range_lower_inclusive(r)
        }),
        "range_upper_inclusive" => range_flag_impl(args, "range_upper_inclusive(range)", |r| {
            zyron_types::range::range_upper_inclusive(r)
        }),
        "range_contains_value" => range_contains_value_impl(args),
        "range_contains_range" => {
            range_pair_bool_impl(args, "range_contains_range(outer, inner)", |a, b| {
                zyron_types::range::range_contains_range(a, b, RANGE_ELEM)
            })
        }
        "range_overlaps" => range_pair_bool_impl(args, "range_overlaps(a, b)", |a, b| {
            zyron_types::range::range_overlaps(a, b, RANGE_ELEM)
        }),
        "range_adjacent" => range_pair_bool_impl(args, "range_adjacent(a, b)", |a, b| {
            zyron_types::range::range_adjacent(a, b, RANGE_ELEM)
        }),
        "range_union" => range_pair_range_impl(args, "range_union(a, b)", |a, b| {
            zyron_types::range::range_union(a, b, RANGE_ELEM)
        }),
        "range_intersection" => range_pair_range_impl(args, "range_intersection(a, b)", |a, b| {
            zyron_types::range::range_intersection(a, b, RANGE_ELEM)
        }),

        // ---------- business_time ----------
        "day_of_week" => day_of_week_impl(args),
        "fiscal_quarter" => fiscal_impl(args, "fiscal_quarter(date, fy_start_month)", |d, m| {
            zyron_types::business_time::fiscal_quarter(d, m) as i32
        }),
        "fiscal_year" => fiscal_impl(args, "fiscal_year(date, fy_start_month)", |d, m| {
            zyron_types::business_time::fiscal_year(d, m)
        }),
        "week_of_fiscal_year" => {
            fiscal_impl(args, "week_of_fiscal_year(date, fy_start_month)", |d, m| {
                zyron_types::business_time::week_of_fiscal_year(d, m) as i32
            })
        }
        "is_business_day" => is_business_day_impl(args),
        "next_business_day" => next_business_day_impl(args),
        "add_business_days" => add_business_days_impl(args),
        "business_days_between" => business_days_between_impl(args),
        "parse_natural_date" => parse_natural_date_impl(args),
        "parse_natural_duration" => parse_natural_duration_impl(args),

        _ => return None,
    })
}

// ---------------------------------------------------------------------------
// Argument access helpers, per row and null aware
// ---------------------------------------------------------------------------

fn check_args(args: &[Column], expected: usize, sig: &str) -> Result<()> {
    if args.len() != expected {
        return Err(ZyronError::ExecutionError(format!(
            "{} takes exactly {} arguments, got {}",
            sig,
            expected,
            args.len()
        )));
    }
    Ok(())
}

fn check_args_between(args: &[Column], min: usize, max: usize, sig: &str) -> Result<()> {
    if args.len() < min || args.len() > max {
        return Err(ZyronError::ExecutionError(format!(
            "{} takes {} to {} arguments, got {}",
            sig,
            min,
            max,
            args.len()
        )));
    }
    Ok(())
}

fn row_count(args: &[Column]) -> usize {
    args.iter().map(|c| c.data.len()).max().unwrap_or(0)
}

// out of range indexes count as null so mismatched column lengths degrade to NULL
fn is_null_at(col: &Column, i: usize) -> bool {
    i >= col.nulls.len() || col.nulls.is_null(i)
}

fn all_null(col: &Column, n: usize) -> bool {
    (0..n).all(|i| is_null_at(col, i))
}

// NULL literal columns arrive with non integer storage and every row null,
// they read as all None instead of a type error
fn opt_int_rows(col: &Column, n: usize, sig: &str) -> Result<Vec<Option<i64>>> {
    let vals = match super::column_ints(col) {
        Ok(v) => v,
        Err(_) if all_null(col, n) => return Ok(vec![None; n]),
        Err(_) => {
            return Err(ZyronError::ExecutionError(format!(
                "{} expects an integer argument",
                sig
            )));
        }
    };
    Ok((0..n)
        .map(|i| {
            if is_null_at(col, i) {
                None
            } else {
                vals.get(i).copied()
            }
        })
        .collect())
}

fn opt_bool_rows(col: &Column, n: usize, sig: &str) -> Result<Vec<Option<bool>>> {
    match &col.data {
        ColumnData::Boolean(v) => Ok((0..n)
            .map(|i| {
                if is_null_at(col, i) {
                    None
                } else {
                    v.get(i).copied()
                }
            })
            .collect()),
        _ if all_null(col, n) => Ok(vec![None; n]),
        _ => Err(ZyronError::ExecutionError(format!(
            "{} expects a boolean argument",
            sig
        ))),
    }
}

fn opt_str_rows<'a>(col: &'a Column, n: usize, sig: &str) -> Result<Vec<Option<&'a str>>> {
    match &col.data {
        ColumnData::Utf8(v) => Ok((0..n)
            .map(|i| {
                if is_null_at(col, i) {
                    None
                } else {
                    v.get(i).map(|s| s.as_str())
                }
            })
            .collect()),
        _ if all_null(col, n) => Ok(vec![None; n]),
        _ => Err(ZyronError::ExecutionError(format!(
            "{} expects a text argument",
            sig
        ))),
    }
}

fn opt_bytes_rows<'a>(col: &'a Column, n: usize, sig: &str) -> Result<Vec<Option<&'a [u8]>>> {
    match &col.data {
        ColumnData::Binary(v) => Ok((0..n)
            .map(|i| {
                if is_null_at(col, i) {
                    None
                } else {
                    v.get(i).map(|b| b.as_slice())
                }
            })
            .collect()),
        ColumnData::Utf8(v) => Ok((0..n)
            .map(|i| {
                if is_null_at(col, i) {
                    None
                } else {
                    v.get(i).map(|s| s.as_bytes())
                }
            })
            .collect()),
        _ if all_null(col, n) => Ok(vec![None; n]),
        _ => Err(ZyronError::ExecutionError(format!(
            "{} expects a binary argument",
            sig
        ))),
    }
}

// ---------------------------------------------------------------------------
// Output builders, None rows become SQL NULL
// ---------------------------------------------------------------------------

fn out_bool(rows: Vec<Option<bool>>) -> Result<Column> {
    let mut data = Vec::with_capacity(rows.len());
    let mut nulls = NullBitmap::none(rows.len());
    for (i, r) in rows.into_iter().enumerate() {
        match r {
            Some(v) => data.push(v),
            None => {
                data.push(false);
                nulls.set_null(i);
            }
        }
    }
    Ok(Column::with_nulls(
        ColumnData::Boolean(data),
        nulls,
        TypeId::Boolean,
    ))
}

fn out_i32(rows: Vec<Option<i32>>, type_id: TypeId) -> Result<Column> {
    let mut data = Vec::with_capacity(rows.len());
    let mut nulls = NullBitmap::none(rows.len());
    for (i, r) in rows.into_iter().enumerate() {
        match r {
            Some(v) => data.push(v),
            None => {
                data.push(0);
                nulls.set_null(i);
            }
        }
    }
    Ok(Column::with_nulls(ColumnData::Int32(data), nulls, type_id))
}

fn out_i64(rows: Vec<Option<i64>>, type_id: TypeId) -> Result<Column> {
    let mut data = Vec::with_capacity(rows.len());
    let mut nulls = NullBitmap::none(rows.len());
    for (i, r) in rows.into_iter().enumerate() {
        match r {
            Some(v) => data.push(v),
            None => {
                data.push(0);
                nulls.set_null(i);
            }
        }
    }
    Ok(Column::with_nulls(ColumnData::Int64(data), nulls, type_id))
}

fn out_utf8(rows: Vec<Option<String>>) -> Result<Column> {
    let mut data = Vec::with_capacity(rows.len());
    let mut nulls = NullBitmap::none(rows.len());
    for (i, r) in rows.into_iter().enumerate() {
        match r {
            Some(v) => data.push(v),
            None => {
                data.push(String::new());
                nulls.set_null(i);
            }
        }
    }
    Ok(Column::with_nulls(
        ColumnData::Utf8(data),
        nulls,
        TypeId::Varchar,
    ))
}

fn out_binary(rows: Vec<Option<Vec<u8>>>, type_id: TypeId) -> Result<Column> {
    let mut data = Vec::with_capacity(rows.len());
    let mut nulls = NullBitmap::none(rows.len());
    for (i, r) in rows.into_iter().enumerate() {
        match r {
            Some(v) => data.push(v),
            None => {
                data.push(Vec::new());
                nulls.set_null(i);
            }
        }
    }
    Ok(Column::with_nulls(ColumnData::Binary(data), nulls, type_id))
}

fn out_interval(rows: Vec<Option<Interval>>) -> Result<Column> {
    let mut data = Vec::with_capacity(rows.len());
    let mut nulls = NullBitmap::none(rows.len());
    for (i, r) in rows.into_iter().enumerate() {
        match r {
            Some(v) => data.push(v),
            None => {
                data.push(Interval::ZERO);
                nulls.set_null(i);
            }
        }
    }
    Ok(Column::with_nulls(
        ColumnData::Interval(data),
        nulls,
        TypeId::Interval,
    ))
}

// ---------------------------------------------------------------------------
// Cron
// ---------------------------------------------------------------------------

// fixed cell layout for CronExpr, 19 bytes total
const CRON_BYTES_LEN: usize = 19;

fn encode_cron(e: &CronExpr) -> Vec<u8> {
    let mut out = Vec::with_capacity(CRON_BYTES_LEN);
    out.extend_from_slice(&e.minute.to_be_bytes());
    out.extend_from_slice(&e.hour.to_be_bytes());
    out.extend_from_slice(&e.day_of_month.to_be_bytes());
    out.extend_from_slice(&e.month.to_be_bytes());
    out.push(e.day_of_week);
    out
}

fn decode_cron(b: &[u8]) -> Option<CronExpr> {
    if b.len() != CRON_BYTES_LEN {
        return None;
    }
    let minute = u64::from_be_bytes(b[0..8].try_into().ok()?);
    let hour = u32::from_be_bytes(b[8..12].try_into().ok()?);
    let day_of_month = u32::from_be_bytes(b[12..16].try_into().ok()?);
    let month = u16::from_be_bytes(b[16..18].try_into().ok()?);
    let day_of_week = b[18];
    Some(CronExpr {
        minute,
        hour,
        day_of_month,
        month,
        day_of_week,
    })
}

// accepts cron text or the encoded cell, unparseable rows read as None
fn opt_cron_rows(col: &Column, n: usize, sig: &str) -> Result<Vec<Option<CronExpr>>> {
    match &col.data {
        ColumnData::Utf8(v) => Ok((0..n)
            .map(|i| {
                if is_null_at(col, i) {
                    None
                } else {
                    v.get(i).and_then(|s| zyron_types::cron::cron_parse(s).ok())
                }
            })
            .collect()),
        ColumnData::Binary(v) => Ok((0..n)
            .map(|i| {
                if is_null_at(col, i) {
                    None
                } else {
                    v.get(i).and_then(|b| decode_cron(b))
                }
            })
            .collect()),
        _ if all_null(col, n) => Ok(vec![None; n]),
        _ => Err(ZyronError::ExecutionError(format!(
            "{} expects a cron expression as text or cron_parse bytes",
            sig
        ))),
    }
}

// parser style fn, invalid expressions yield NULL for the row
fn cron_parse_impl(args: &[Column]) -> Result<Column> {
    check_args(args, 1, "cron_parse(text)")?;
    let n = row_count(args);
    let exprs = opt_cron_rows(&args[0], n, "cron_parse")?;
    out_binary(
        exprs
            .into_iter()
            .map(|e| e.map(|x| encode_cron(&x)))
            .collect(),
        TypeId::Bytea,
    )
}

// unmatchable expressions yield NULL for the row instead of failing the batch
fn cron_next_impl(args: &[Column]) -> Result<Column> {
    check_args(args, 2, "cron_next(cron, after)")?;
    let n = row_count(args);
    let exprs = opt_cron_rows(&args[0], n, "cron_next")?;
    let after = opt_int_rows(&args[1], n, "cron_next")?;
    let rows = (0..n)
        .map(|i| match (&exprs[i], after[i]) {
            (Some(e), Some(t)) => zyron_types::cron::cron_next(e, t).ok(),
            _ => None,
        })
        .collect();
    out_i64(rows, TypeId::TimestampTz)
}

fn cron_prev_impl(args: &[Column]) -> Result<Column> {
    check_args(args, 2, "cron_prev(cron, before)")?;
    let n = row_count(args);
    let exprs = opt_cron_rows(&args[0], n, "cron_prev")?;
    let before = opt_int_rows(&args[1], n, "cron_prev")?;
    let rows = (0..n)
        .map(|i| match (&exprs[i], before[i]) {
            (Some(e), Some(t)) => zyron_types::cron::cron_prev(e, t).ok(),
            _ => None,
        })
        .collect();
    out_i64(rows, TypeId::TimestampTz)
}

fn cron_matches_impl(args: &[Column]) -> Result<Column> {
    check_args(args, 2, "cron_matches(cron, timestamp)")?;
    let n = row_count(args);
    let exprs = opt_cron_rows(&args[0], n, "cron_matches")?;
    let ts = opt_int_rows(&args[1], n, "cron_matches")?;
    let rows = (0..n)
        .map(|i| match (&exprs[i], ts[i]) {
            (Some(e), Some(t)) => Some(zyron_types::cron::cron_matches(e, t)),
            _ => None,
        })
        .collect();
    out_bool(rows)
}

// Array result, JSON array of epoch microsecond fire times inside a Binary cell
fn cron_between_impl(args: &[Column]) -> Result<Column> {
    check_args(args, 3, "cron_between(cron, start, end)")?;
    let n = row_count(args);
    let exprs = opt_cron_rows(&args[0], n, "cron_between")?;
    let start = opt_int_rows(&args[1], n, "cron_between")?;
    let end = opt_int_rows(&args[2], n, "cron_between")?;
    let rows = (0..n)
        .map(|i| match (&exprs[i], start[i], end[i]) {
            (Some(e), Some(s), Some(t)) => zyron_types::cron::cron_between(e, s, t)
                .ok()
                .and_then(|v| serde_json::to_string(&v).ok())
                .map(|s| s.into_bytes()),
            _ => None,
        })
        .collect();
    out_binary(rows, TypeId::Array)
}

// cron_list(cron, after_micros, count) -> JSON array of the next count
// firing timestamps after the given instant, count capped at 1000
fn cron_list_impl(args: &[Column]) -> Result<Column> {
    check_args(args, 3, "cron_list(cron, after, count)")?;
    let n = row_count(args);
    let exprs = opt_cron_rows(&args[0], n, "cron_list")?;
    let after = opt_int_rows(&args[1], n, "cron_list")?;
    let count = opt_int_rows(&args[2], n, "cron_list")?;
    let rows = (0..n)
        .map(|i| match (&exprs[i], after[i], count[i]) {
            (Some(e), Some(mut cursor), Some(c)) => {
                let cap = c.clamp(0, 1000) as usize;
                let mut firings: Vec<i64> = Vec::with_capacity(cap);
                for _ in 0..cap {
                    match zyron_types::cron::cron_next(e, cursor) {
                        Ok(next) => {
                            firings.push(next);
                            cursor = next;
                        }
                        Err(_) => break,
                    }
                }
                serde_json::to_string(&firings).ok().map(|s| s.into_bytes())
            }
            _ => None,
        })
        .collect();
    out_binary(rows, TypeId::Array)
}

fn cron_human_readable_impl(args: &[Column]) -> Result<Column> {
    check_args(args, 1, "cron_human_readable(cron)")?;
    let n = row_count(args);
    let exprs = opt_cron_rows(&args[0], n, "cron_human_readable")?;
    out_utf8(
        exprs
            .into_iter()
            .map(|e| e.map(|x| zyron_types::cron::cron_human_readable(&x)))
            .collect(),
    )
}

// ---------------------------------------------------------------------------
// Range
// ---------------------------------------------------------------------------

// SQL level ranges always use 8 byte elements
const RANGE_ELEM: usize = 8;

// sign flipped big-endian keeps byte order equal to numeric order
fn order_key_i64(v: i64) -> [u8; 8] {
    ((v as u64) ^ (1u64 << 63)).to_be_bytes()
}

// NULL bounds map to unbounded sides matching range_create Option bounds,
// NULL inclusivity flags null the whole row
fn range_create_impl(args: &[Column]) -> Result<Column> {
    check_args(
        args,
        4,
        "range_create(lower, upper, lower_inclusive, upper_inclusive)",
    )?;
    let n = row_count(args);
    let lower = opt_int_rows(&args[0], n, "range_create")?;
    let upper = opt_int_rows(&args[1], n, "range_create")?;
    let linc = opt_bool_rows(&args[2], n, "range_create")?;
    let uinc = opt_bool_rows(&args[3], n, "range_create")?;
    let mut rows = Vec::with_capacity(n);
    for i in 0..n {
        let (Some(li), Some(ui)) = (linc[i], uinc[i]) else {
            rows.push(None);
            continue;
        };
        let lb = lower[i].map(order_key_i64);
        let ub = upper[i].map(order_key_i64);
        let cell = zyron_types::range::range_create(
            lb.as_ref().map(|b| b.as_slice()),
            ub.as_ref().map(|b| b.as_slice()),
            li,
            ui,
            RANGE_ELEM,
        )?;
        rows.push(Some(cell));
    }
    out_binary(rows, TypeId::Range)
}

fn range_flag_impl<F: Fn(&[u8]) -> bool>(args: &[Column], sig: &str, f: F) -> Result<Column> {
    check_args(args, 1, sig)?;
    let n = row_count(args);
    let ranges = opt_bytes_rows(&args[0], n, sig)?;
    out_bool(ranges.into_iter().map(|r| r.map(&f)).collect())
}

// infinite or empty bounds surface as NULL matching the module's Option result
fn range_bound_impl<F: Fn(&[u8]) -> Option<Vec<u8>>>(
    args: &[Column],
    sig: &str,
    f: F,
) -> Result<Column> {
    check_args(args, 1, sig)?;
    let n = row_count(args);
    let ranges = opt_bytes_rows(&args[0], n, sig)?;
    out_binary(
        ranges.into_iter().map(|r| r.and_then(&f)).collect(),
        TypeId::Bytea,
    )
}

fn range_contains_value_impl(args: &[Column]) -> Result<Column> {
    check_args(args, 2, "range_contains_value(range, value)")?;
    let n = row_count(args);
    let ranges = opt_bytes_rows(&args[0], n, "range_contains_value")?;
    let values = opt_int_rows(&args[1], n, "range_contains_value")?;
    let rows = (0..n)
        .map(|i| match (ranges[i], values[i]) {
            (Some(r), Some(v)) => Some(zyron_types::range::range_contains_value(
                r,
                &order_key_i64(v),
                RANGE_ELEM,
            )),
            _ => None,
        })
        .collect();
    out_bool(rows)
}

fn range_pair_bool_impl<F: Fn(&[u8], &[u8]) -> bool>(
    args: &[Column],
    sig: &str,
    f: F,
) -> Result<Column> {
    check_args(args, 2, sig)?;
    let n = row_count(args);
    let a = opt_bytes_rows(&args[0], n, sig)?;
    let b = opt_bytes_rows(&args[1], n, sig)?;
    let rows = (0..n)
        .map(|i| match (a[i], b[i]) {
            (Some(x), Some(y)) => Some(f(x, y)),
            _ => None,
        })
        .collect();
    out_bool(rows)
}

// union of disjoint non adjacent ranges propagates the module error,
// intersection only errors on structurally impossible bound sizes
fn range_pair_range_impl<F: Fn(&[u8], &[u8]) -> Result<Vec<u8>>>(
    args: &[Column],
    sig: &str,
    f: F,
) -> Result<Column> {
    check_args(args, 2, sig)?;
    let n = row_count(args);
    let a = opt_bytes_rows(&args[0], n, sig)?;
    let b = opt_bytes_rows(&args[1], n, sig)?;
    let mut rows = Vec::with_capacity(n);
    for i in 0..n {
        rows.push(match (a[i], b[i]) {
            (Some(x), Some(y)) => Some(f(x, y)?),
            _ => None,
        });
    }
    out_binary(rows, TypeId::Range)
}

// ---------------------------------------------------------------------------
// Business time
// ---------------------------------------------------------------------------

fn day_of_week_impl(args: &[Column]) -> Result<Column> {
    check_args(args, 1, "day_of_week(date)")?;
    let n = row_count(args);
    let dates = opt_int_rows(&args[0], n, "day_of_week")?;
    let rows = dates
        .into_iter()
        .map(|o| {
            o.and_then(|v| i32::try_from(v).ok())
                .map(|d| zyron_types::business_time::day_of_week(d) as i32)
        })
        .collect();
    out_i32(rows, TypeId::Int32)
}

// fy_start_month outside 1 to 12 nulls the row rather than failing the batch
fn fiscal_impl<F: Fn(i32, u8) -> i32>(args: &[Column], sig: &str, f: F) -> Result<Column> {
    check_args(args, 2, sig)?;
    let n = row_count(args);
    let dates = opt_int_rows(&args[0], n, sig)?;
    let months = opt_int_rows(&args[1], n, sig)?;
    let rows = (0..n)
        .map(|i| match (dates[i], months[i]) {
            (Some(d), Some(m)) if (1..=12).contains(&m) => {
                i32::try_from(d).ok().map(|dd| f(dd, m as u8))
            }
            _ => None,
        })
        .collect();
    out_i32(rows, TypeId::Int32)
}

fn parse_holidays(b: &[u8]) -> Option<Vec<i32>> {
    let v: serde_json::Value = serde_json::from_slice(b).ok()?;
    let arr = v.as_array()?;
    let mut out = Vec::with_capacity(arr.len());
    for e in arr {
        out.push(i32::try_from(e.as_i64()?).ok()?);
    }
    Some(out)
}

// absent holiday argument means an empty holiday list,
// malformed holiday JSON nulls the row
fn holidays_rows(col: Option<&Column>, n: usize, sig: &str) -> Result<Vec<Option<Vec<i32>>>> {
    match col {
        None => Ok(vec![Some(Vec::new()); n]),
        Some(c) => {
            let cells = opt_bytes_rows(c, n, sig)?;
            Ok(cells
                .into_iter()
                .map(|o| o.and_then(parse_holidays))
                .collect())
        }
    }
}

fn is_business_day_impl(args: &[Column]) -> Result<Column> {
    check_args_between(args, 1, 2, "is_business_day(date [, holidays])")?;
    let n = row_count(args);
    let dates = opt_int_rows(&args[0], n, "is_business_day")?;
    let holidays = holidays_rows(args.get(1), n, "is_business_day")?;
    let rows = (0..n)
        .map(|i| match (dates[i], holidays[i].as_ref()) {
            (Some(d), Some(h)) => i32::try_from(d)
                .ok()
                .map(|dd| zyron_types::business_time::is_business_day(dd, h)),
            _ => None,
        })
        .collect();
    out_bool(rows)
}

fn next_business_day_impl(args: &[Column]) -> Result<Column> {
    check_args_between(args, 1, 2, "next_business_day(date [, holidays])")?;
    let n = row_count(args);
    let dates = opt_int_rows(&args[0], n, "next_business_day")?;
    let holidays = holidays_rows(args.get(1), n, "next_business_day")?;
    let rows = (0..n)
        .map(|i| match (dates[i], holidays[i].as_ref()) {
            (Some(d), Some(h)) => i32::try_from(d)
                .ok()
                .map(|dd| zyron_types::business_time::next_business_day(dd, h)),
            _ => None,
        })
        .collect();
    out_i32(rows, TypeId::Date)
}

fn add_business_days_impl(args: &[Column]) -> Result<Column> {
    check_args_between(args, 2, 3, "add_business_days(date, n [, holidays])")?;
    let n = row_count(args);
    let dates = opt_int_rows(&args[0], n, "add_business_days")?;
    let counts = opt_int_rows(&args[1], n, "add_business_days")?;
    let holidays = holidays_rows(args.get(2), n, "add_business_days")?;
    let rows = (0..n)
        .map(|i| match (dates[i], counts[i], holidays[i].as_ref()) {
            (Some(d), Some(c), Some(h)) => match (i32::try_from(d), i32::try_from(c)) {
                (Ok(dd), Ok(cc)) => Some(zyron_types::business_time::add_business_days(dd, cc, h)),
                _ => None,
            },
            _ => None,
        })
        .collect();
    out_i32(rows, TypeId::Date)
}

fn business_days_between_impl(args: &[Column]) -> Result<Column> {
    check_args_between(args, 2, 3, "business_days_between(start, end [, holidays])")?;
    let n = row_count(args);
    let start = opt_int_rows(&args[0], n, "business_days_between")?;
    let end = opt_int_rows(&args[1], n, "business_days_between")?;
    let holidays = holidays_rows(args.get(2), n, "business_days_between")?;
    let rows = (0..n)
        .map(|i| match (start[i], end[i], holidays[i].as_ref()) {
            (Some(s), Some(e), Some(h)) => match (i32::try_from(s), i32::try_from(e)) {
                (Ok(ss), Ok(ee)) => {
                    Some(zyron_types::business_time::business_days_between(ss, ee, h))
                }
                _ => None,
            },
            _ => None,
        })
        .collect();
    out_i32(rows, TypeId::Int32)
}

// parser style fn, unparseable text yields NULL for the row
fn parse_natural_date_impl(args: &[Column]) -> Result<Column> {
    check_args(args, 2, "parse_natural_date(text, reference_date)")?;
    let n = row_count(args);
    let texts = opt_str_rows(&args[0], n, "parse_natural_date")?;
    let refs = opt_int_rows(&args[1], n, "parse_natural_date")?;
    let rows = (0..n)
        .map(|i| match (texts[i], refs[i]) {
            (Some(t), Some(r)) => i32::try_from(r)
                .ok()
                .and_then(|rr| zyron_types::business_time::parse_natural_date(t, rr).ok()),
            _ => None,
        })
        .collect();
    out_i32(rows, TypeId::Date)
}

// parser style fn, unparseable text yields NULL for the row
fn parse_natural_duration_impl(args: &[Column]) -> Result<Column> {
    check_args(args, 1, "parse_natural_duration(text)")?;
    let n = row_count(args);
    let texts = opt_str_rows(&args[0], n, "parse_natural_duration")?;
    let rows = texts
        .into_iter()
        .map(|o| o.and_then(|t| zyron_types::business_time::parse_natural_duration(t).ok()))
        .collect();
    out_interval(rows)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn text_col(vals: &[&str]) -> Column {
        Column::new(
            ColumnData::Utf8(vals.iter().map(|s| s.to_string()).collect()),
            TypeId::Text,
        )
    }

    fn i64_col(vals: &[i64]) -> Column {
        Column::new(ColumnData::Int64(vals.to_vec()), TypeId::Int64)
    }

    fn i32_col(vals: &[i32]) -> Column {
        Column::new(ColumnData::Int32(vals.to_vec()), TypeId::Date)
    }

    fn bool_col(vals: &[bool]) -> Column {
        Column::new(ColumnData::Boolean(vals.to_vec()), TypeId::Boolean)
    }

    // 2024-01-01 00:00:00 UTC
    const JAN1_2024_MICROS: i64 = 1_704_067_200_000_000;
    // 2024-01-01 as days since epoch, a Monday
    const JAN1_2024_DAYS: i32 = 19723;

    #[test]
    fn cron_parse_roundtrips_through_cron_matches() {
        let parsed = dispatch("cron_parse", &[text_col(&["0 0 * * *"])], 1)
            .unwrap()
            .unwrap();
        assert_eq!(parsed.type_id, TypeId::Bytea);
        let ts = i64_col(&[JAN1_2024_MICROS]);
        let matched = dispatch("cron_matches", &[parsed, ts], 1).unwrap().unwrap();
        match &matched.data {
            ColumnData::Boolean(v) => assert!(v[0]),
            other => panic!("expected boolean column, got {:?}", other),
        }
        assert!(!matched.is_null(0));
    }

    #[test]
    fn cron_next_finds_next_midnight() {
        let out = dispatch(
            "cron_next",
            &[text_col(&["0 0 * * *"]), i64_col(&[JAN1_2024_MICROS])],
            1,
        )
        .unwrap()
        .unwrap();
        assert_eq!(out.type_id, TypeId::TimestampTz);
        match &out.data {
            ColumnData::Int64(v) => assert_eq!(v[0], JAN1_2024_MICROS + 86400 * 1_000_000),
            other => panic!("expected int64 column, got {:?}", other),
        }
    }

    #[test]
    fn cron_next_null_input_yields_null_row() {
        let mut nulls = NullBitmap::none(2);
        nulls.set_null(1);
        let after = Column::with_nulls(
            ColumnData::Int64(vec![JAN1_2024_MICROS, 0]),
            nulls,
            TypeId::Int64,
        );
        let out = dispatch(
            "cron_next",
            &[text_col(&["0 0 * * *", "0 0 * * *"]), after],
            2,
        )
        .unwrap()
        .unwrap();
        assert!(!out.is_null(0));
        assert!(out.is_null(1));
    }

    #[test]
    fn cron_between_returns_json_match_list() {
        let end = JAN1_2024_MICROS + 86400 * 1_000_000;
        let out = dispatch(
            "cron_between",
            &[
                text_col(&["0 * * * *"]),
                i64_col(&[JAN1_2024_MICROS]),
                i64_col(&[end]),
            ],
            1,
        )
        .unwrap()
        .unwrap();
        assert_eq!(out.type_id, TypeId::Array);
        match &out.data {
            ColumnData::Binary(v) => {
                let parsed: Vec<i64> = serde_json::from_slice(&v[0]).unwrap();
                assert_eq!(parsed.len(), 24);
                assert_eq!(parsed[0], JAN1_2024_MICROS);
            }
            other => panic!("expected binary column, got {:?}", other),
        }
    }

    #[test]
    fn range_create_and_contains_value_respect_bounds() {
        let r = dispatch(
            "range_create",
            &[
                i64_col(&[1]),
                i64_col(&[10]),
                bool_col(&[true]),
                bool_col(&[false]),
            ],
            1,
        )
        .unwrap()
        .unwrap();
        assert_eq!(r.type_id, TypeId::Range);
        let inside = dispatch("range_contains_value", &[r.clone(), i64_col(&[5])], 1)
            .unwrap()
            .unwrap();
        let at_lower = dispatch("range_contains_value", &[r.clone(), i64_col(&[1])], 1)
            .unwrap()
            .unwrap();
        let at_upper = dispatch("range_contains_value", &[r, i64_col(&[10])], 1)
            .unwrap()
            .unwrap();
        match (&inside.data, &at_lower.data, &at_upper.data) {
            (ColumnData::Boolean(a), ColumnData::Boolean(b), ColumnData::Boolean(c)) => {
                assert!(a[0]);
                assert!(b[0]);
                assert!(!c[0]);
            }
            other => panic!("expected boolean columns, got {:?}", other),
        }
    }

    #[test]
    fn range_null_lower_bound_is_unbounded() {
        let mut nulls = NullBitmap::none(1);
        nulls.set_null(0);
        let null_lower = Column::with_nulls(ColumnData::Int64(vec![0]), nulls, TypeId::Int64);
        let r = dispatch(
            "range_create",
            &[
                null_lower,
                i64_col(&[10]),
                bool_col(&[true]),
                bool_col(&[true]),
            ],
            1,
        )
        .unwrap()
        .unwrap();
        let contains = dispatch("range_contains_value", &[r.clone(), i64_col(&[-999])], 1)
            .unwrap()
            .unwrap();
        match &contains.data {
            ColumnData::Boolean(v) => assert!(v[0]),
            other => panic!("expected boolean column, got {:?}", other),
        }
        let lower = dispatch("range_lower", &[r], 1).unwrap().unwrap();
        assert!(lower.is_null(0));
    }

    #[test]
    fn business_day_functions_skip_weekends_and_holidays() {
        // Friday 2024-01-05 plus one business day lands on Monday 2024-01-08
        let out = dispatch(
            "add_business_days",
            &[i32_col(&[JAN1_2024_DAYS + 4]), i64_col(&[1])],
            1,
        )
        .unwrap()
        .unwrap();
        assert_eq!(out.type_id, TypeId::Date);
        match &out.data {
            ColumnData::Int32(v) => assert_eq!(v[0], JAN1_2024_DAYS + 7),
            other => panic!("expected int32 column, got {:?}", other),
        }
        // Monday declared a holiday is not a business day
        let holidays = text_col(&["[19723]"]);
        let biz = dispatch(
            "is_business_day",
            &[i32_col(&[JAN1_2024_DAYS]), holidays],
            1,
        )
        .unwrap()
        .unwrap();
        match &biz.data {
            ColumnData::Boolean(v) => assert!(!v[0]),
            other => panic!("expected boolean column, got {:?}", other),
        }
        // Monday through Sunday holds five business days
        let between = dispatch(
            "business_days_between",
            &[i32_col(&[JAN1_2024_DAYS]), i32_col(&[JAN1_2024_DAYS + 6])],
            1,
        )
        .unwrap()
        .unwrap();
        match &between.data {
            ColumnData::Int32(v) => assert_eq!(v[0], 5),
            other => panic!("expected int32 column, got {:?}", other),
        }
    }

    #[test]
    fn natural_date_and_duration_parse_with_null_on_failure() {
        let date = dispatch(
            "parse_natural_date",
            &[text_col(&["3 days ago"]), i32_col(&[JAN1_2024_DAYS])],
            1,
        )
        .unwrap()
        .unwrap();
        match &date.data {
            ColumnData::Int32(v) => assert_eq!(v[0], JAN1_2024_DAYS - 3),
            other => panic!("expected int32 column, got {:?}", other),
        }
        let dur = dispatch(
            "parse_natural_duration",
            &[text_col(&["1 week", "not a duration"])],
            2,
        )
        .unwrap()
        .unwrap();
        assert_eq!(dur.type_id, TypeId::Interval);
        match &dur.data {
            ColumnData::Interval(v) => assert_eq!(v[0].days, 7),
            other => panic!("expected interval column, got {:?}", other),
        }
        assert!(!dur.is_null(0));
        assert!(dur.is_null(1));
    }

    #[test]
    fn unknown_name_returns_none_and_bad_arity_errors() {
        assert!(dispatch("not_a_function", &[], 1).is_none());
        let err = dispatch("cron_next", &[text_col(&["* * * * *"])], 1)
            .unwrap()
            .unwrap_err();
        assert!(err.to_string().contains("cron_next"));
    }
}
