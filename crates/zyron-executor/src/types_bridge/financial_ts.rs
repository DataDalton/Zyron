//! Financial and calendar time-series dispatch arms for the types bridge
//!
//! Array typed arguments and results are JSON number arrays carried as text
//! bytes inside Binary cells, Utf8 JSON literals are accepted on input
//!
//! Per-row domain failures (IRR non convergence, bond parameter violations,
//! malformed array cells, out of range periods) yield NULL for that row,
//! structural misuse (wrong arg count, non numeric column) returns
//! ExecutionError with the function signature in the message

use crate::column::{Column, ColumnData, NullBitmap};
use zyron_common::{Interval, Result, TypeId, ZyronError};

// plain time_bucket, time_bucket_gapfill, locf, interpolate live in expr.rs
pub(super) fn dispatch(name: &str, args: &[Column], _num_rows: usize) -> Option<Result<Column>> {
    Some(match name {
        "npv" => npv_impl(args),
        "irr" => irr_impl(args),
        "xnpv" => xnpv_impl(args),
        "xirr" => xirr_impl(args),
        "pmt" => map_floats("pmt(rate, nper, pv)", args, 3, |v| {
            zyron_types::financial::pmt(v[0], v[1], v[2])
        }),
        "fv" => map_floats("fv(rate, nper, pmt, pv)", args, 4, |v| {
            zyron_types::financial::fv(v[0], v[1], v[2], v[3])
        }),
        "pv" => map_floats("pv(rate, nper, pmt, fv)", args, 4, |v| {
            zyron_types::financial::pv(v[0], v[1], v[2], v[3])
        }),
        "compound_interest" => {
            map_floats("compound_interest(principal, rate, n, t)", args, 4, |v| {
                zyron_types::financial::compound_interest(v[0], v[1], v[2], v[3])
            })
        }
        "depreciation_sl" => map_floats("depreciation_sl(cost, salvage, life)", args, 3, |v| {
            zyron_types::financial::depreciation_sl(v[0], v[1], v[2])
        }),
        "depreciation_db" => map_floats(
            "depreciation_db(cost, salvage, life, period)",
            args,
            4,
            |v| zyron_types::financial::depreciation_db(v[0], v[1], v[2], v[3]),
        ),
        "depreciation_syd" => map_floats(
            "depreciation_syd(cost, salvage, life, period)",
            args,
            4,
            |v| zyron_types::financial::depreciation_syd(v[0], v[1], v[2], v[3]),
        ),
        "bond_price" => bond_price_impl(args),
        "bond_yield" => bond_yield_impl(args),
        "amortization_schedule" => amortization_schedule_impl(args),
        "lttb" => lttb_impl(args),
        "time_bucket_calendar" => time_bucket_calendar_impl(args),
        "time_bucket_gapfill_calendar" => time_bucket_gapfill_calendar_impl(args),
        _ => return None,
    })
}

// ---------------------------------------------------------------------------
// shared readers
// ---------------------------------------------------------------------------

// tolerant numeric reader, accepts any float or int column width
fn numeric_column(sig: &str, col: &Column) -> Result<Vec<f64>> {
    super::column_floats(col)
        .or_else(|_| super::column_ints(col).map(|v| v.into_iter().map(|x| x as f64).collect()))
        .map_err(|_| ZyronError::ExecutionError(format!("{sig} expects numeric arguments")))
}

// array cells arrive as Binary holding JSON text bytes, Utf8 JSON also accepted
fn check_array_column(sig: &str, col: &Column) -> Result<()> {
    match &col.data {
        ColumnData::Binary(_) | ColumnData::Utf8(_) => Ok(()),
        _ => Err(ZyronError::ExecutionError(format!(
            "{sig} expects a JSON array argument"
        ))),
    }
}

// interval args arrive as Interval columns, integer widths are whole
// microseconds matching the plain time_bucket contract
fn column_intervals(sig: &str, col: &Column) -> Result<Vec<Interval>> {
    match &col.data {
        ColumnData::Interval(v) => Ok(v.clone()),
        ColumnData::Int64(v) => Ok(v
            .iter()
            .map(|&m| Interval::from_nanoseconds(m.saturating_mul(1000)))
            .collect()),
        ColumnData::Int32(v) => Ok(v
            .iter()
            .map(|&m| Interval::from_nanoseconds((m as i64).saturating_mul(1000)))
            .collect()),
        _ => Err(ZyronError::ExecutionError(format!(
            "{sig} expects an interval first argument"
        ))),
    }
}

fn cell_text(col: &Column, row: usize) -> Option<&str> {
    match &col.data {
        ColumnData::Binary(v) => std::str::from_utf8(v.get(row)?.as_slice()).ok(),
        ColumnData::Utf8(v) => v.get(row).map(|s| s.as_str()),
        _ => None,
    }
}

// parses a JSON number array cell, None on malformed content
fn parse_f64_array(col: &Column, row: usize) -> Option<Vec<f64>> {
    let parsed: serde_json::Value = serde_json::from_str(cell_text(col, row)?).ok()?;
    let arr = parsed.as_array()?;
    let mut out = Vec::with_capacity(arr.len());
    for v in arr {
        out.push(v.as_f64()?);
    }
    Some(out)
}

// parses a JSON integer array cell, floats are truncated toward zero
fn parse_i64_array(col: &Column, row: usize) -> Option<Vec<i64>> {
    let parsed: serde_json::Value = serde_json::from_str(cell_text(col, row)?).ok()?;
    let arr = parsed.as_array()?;
    let mut out = Vec::with_capacity(arr.len());
    for v in arr {
        out.push(v.as_i64().or_else(|| v.as_f64().map(|f| f as i64))?);
    }
    Some(out)
}

// bounds a periods argument, guards against runaway loops and allocation
fn periods_u32(sig: &str, v: f64) -> Result<u32> {
    if !v.is_finite() || v < 0.0 || v > 1_000_000.0 {
        return Err(ZyronError::ExecutionError(format!(
            "{sig} periods out of range"
        )));
    }
    Ok(v as u32)
}

// ---------------------------------------------------------------------------
// row-wise float mappers
// ---------------------------------------------------------------------------

// applies an n-ary float function row-wise, NULL in any input yields NULL
fn map_floats(
    sig: &str,
    args: &[Column],
    arity: usize,
    f: impl Fn(&[f64]) -> f64,
) -> Result<Column> {
    map_floats_fallible(sig, args, arity, |v| Ok(f(v)))
}

// fallible variant, a per-row Err yields NULL for that row
fn map_floats_fallible(
    sig: &str,
    args: &[Column],
    arity: usize,
    f: impl Fn(&[f64]) -> Result<f64>,
) -> Result<Column> {
    if args.len() != arity {
        return Err(ZyronError::ExecutionError(format!(
            "{sig} takes exactly {arity} arguments"
        )));
    }
    let cols: Vec<Vec<f64>> = args
        .iter()
        .map(|c| numeric_column(sig, c))
        .collect::<Result<Vec<_>>>()?;
    let n = cols.iter().map(|v| v.len()).min().unwrap_or(0);
    let mut data = Vec::with_capacity(n);
    let mut nulls = NullBitmap::none(n);
    let mut vals = vec![0.0f64; arity];
    for i in 0..n {
        if args.iter().any(|c| c.nulls.is_null(i)) {
            data.push(0.0);
            nulls.set_null(i);
            continue;
        }
        for (k, col) in cols.iter().enumerate() {
            vals[k] = col[i];
        }
        match f(&vals) {
            Ok(v) => data.push(v),
            Err(_) => {
                data.push(0.0);
                nulls.set_null(i);
            }
        }
    }
    Ok(Column::with_nulls(
        ColumnData::Float64(data),
        nulls,
        TypeId::Float64,
    ))
}

// ---------------------------------------------------------------------------
// financial
// ---------------------------------------------------------------------------

fn bond_price_impl(args: &[Column]) -> Result<Column> {
    const SIG: &str = "bond_price(face, coupon_rate, yield_rate, periods)";
    map_floats_fallible(SIG, args, 4, |v| {
        let periods = periods_u32(SIG, v[3])?;
        Ok(zyron_types::financial::bond_price(
            v[0], v[1], v[2], periods,
        ))
    })
}

fn bond_yield_impl(args: &[Column]) -> Result<Column> {
    const SIG: &str = "bond_yield(face, coupon_rate, price, periods)";
    map_floats_fallible(SIG, args, 4, |v| {
        let periods = periods_u32(SIG, v[3])?;
        zyron_types::financial::bond_yield(v[0], v[1], v[2], periods)
    })
}

// npv(rate, cashflows) with cashflows as a JSON number array cell
fn npv_impl(args: &[Column]) -> Result<Column> {
    const SIG: &str = "npv(rate, cashflows)";
    if args.len() != 2 {
        return Err(ZyronError::ExecutionError(format!(
            "{SIG} takes exactly 2 arguments"
        )));
    }
    let rates = numeric_column(SIG, &args[0])?;
    check_array_column(SIG, &args[1])?;
    let n = rates.len().min(args[1].data.len());
    let mut data = Vec::with_capacity(n);
    let mut nulls = NullBitmap::none(n);
    for i in 0..n {
        if args[0].nulls.is_null(i) || args[1].nulls.is_null(i) {
            data.push(0.0);
            nulls.set_null(i);
            continue;
        }
        match parse_f64_array(&args[1], i) {
            Some(flows) => data.push(zyron_types::financial::npv(rates[i], &flows)),
            None => {
                data.push(0.0);
                nulls.set_null(i);
            }
        }
    }
    Ok(Column::with_nulls(
        ColumnData::Float64(data),
        nulls,
        TypeId::Float64,
    ))
}

// irr(cashflows), rows where IRR cannot be computed yield NULL
fn irr_impl(args: &[Column]) -> Result<Column> {
    const SIG: &str = "irr(cashflows)";
    if args.len() != 1 {
        return Err(ZyronError::ExecutionError(format!(
            "{SIG} takes exactly 1 argument"
        )));
    }
    check_array_column(SIG, &args[0])?;
    let n = args[0].data.len();
    let mut data = Vec::with_capacity(n);
    let mut nulls = NullBitmap::none(n);
    for i in 0..n {
        if args[0].nulls.is_null(i) {
            data.push(0.0);
            nulls.set_null(i);
            continue;
        }
        let rate =
            parse_f64_array(&args[0], i).and_then(|flows| zyron_types::financial::irr(&flows).ok());
        match rate {
            Some(r) => data.push(r),
            None => {
                data.push(0.0);
                nulls.set_null(i);
            }
        }
    }
    Ok(Column::with_nulls(
        ColumnData::Float64(data),
        nulls,
        TypeId::Float64,
    ))
}

// xnpv(rate, dates, cashflows) with dates as epoch day integers
fn xnpv_impl(args: &[Column]) -> Result<Column> {
    const SIG: &str = "xnpv(rate, dates, cashflows)";
    if args.len() != 3 {
        return Err(ZyronError::ExecutionError(format!(
            "{SIG} takes exactly 3 arguments"
        )));
    }
    let rates = numeric_column(SIG, &args[0])?;
    check_array_column(SIG, &args[1])?;
    check_array_column(SIG, &args[2])?;
    let n = rates.len().min(args[1].data.len()).min(args[2].data.len());
    let mut data = Vec::with_capacity(n);
    let mut nulls = NullBitmap::none(n);
    for i in 0..n {
        if args.iter().any(|c| c.nulls.is_null(i)) {
            data.push(0.0);
            nulls.set_null(i);
            continue;
        }
        let dates = parse_i64_array(&args[1], i);
        let flows = parse_f64_array(&args[2], i);
        match (dates, flows) {
            (Some(dates), Some(flows)) => {
                let day_vals: Vec<i32> = dates.iter().map(|&d| d as i32).collect();
                data.push(zyron_types::financial::xnpv(rates[i], &day_vals, &flows));
            }
            _ => {
                data.push(0.0);
                nulls.set_null(i);
            }
        }
    }
    Ok(Column::with_nulls(
        ColumnData::Float64(data),
        nulls,
        TypeId::Float64,
    ))
}

// xirr(dates, cashflows), rows where XIRR cannot be computed yield NULL
fn xirr_impl(args: &[Column]) -> Result<Column> {
    const SIG: &str = "xirr(dates, cashflows)";
    if args.len() != 2 {
        return Err(ZyronError::ExecutionError(format!(
            "{SIG} takes exactly 2 arguments"
        )));
    }
    check_array_column(SIG, &args[0])?;
    check_array_column(SIG, &args[1])?;
    let n = args[0].data.len().min(args[1].data.len());
    let mut data = Vec::with_capacity(n);
    let mut nulls = NullBitmap::none(n);
    for i in 0..n {
        if args[0].nulls.is_null(i) || args[1].nulls.is_null(i) {
            data.push(0.0);
            nulls.set_null(i);
            continue;
        }
        let dates = parse_i64_array(&args[0], i);
        let flows = parse_f64_array(&args[1], i);
        let rate = match (dates, flows) {
            (Some(dates), Some(flows)) => {
                let day_vals: Vec<i32> = dates.iter().map(|&d| d as i32).collect();
                zyron_types::financial::xirr(&day_vals, &flows).ok()
            }
            _ => None,
        };
        match rate {
            Some(r) => data.push(r),
            None => {
                data.push(0.0);
                nulls.set_null(i);
            }
        }
    }
    Ok(Column::with_nulls(
        ColumnData::Float64(data),
        nulls,
        TypeId::Float64,
    ))
}

// amortization_schedule(principal, rate, periods) -> Array of
// {payment, principal, interest, balance} objects as JSON text bytes
fn amortization_schedule_impl(args: &[Column]) -> Result<Column> {
    const SIG: &str = "amortization_schedule(principal, rate, periods)";
    if args.len() != 3 {
        return Err(ZyronError::ExecutionError(format!(
            "{SIG} takes exactly 3 arguments"
        )));
    }
    let principals = numeric_column(SIG, &args[0])?;
    let rates = numeric_column(SIG, &args[1])?;
    let periods = numeric_column(SIG, &args[2])?;
    let n = principals.len().min(rates.len()).min(periods.len());
    let mut data: Vec<Vec<u8>> = Vec::with_capacity(n);
    let mut nulls = NullBitmap::none(n);
    for i in 0..n {
        if args.iter().any(|c| c.nulls.is_null(i)) {
            data.push(Vec::new());
            nulls.set_null(i);
            continue;
        }
        let p = match periods_u32(SIG, periods[i]) {
            Ok(p) => p,
            Err(_) => {
                data.push(Vec::new());
                nulls.set_null(i);
                continue;
            }
        };
        let schedule = zyron_types::financial::amortization_schedule(principals[i], rates[i], p);
        let items: Vec<serde_json::Value> = schedule
            .iter()
            .map(|&(payment, principal_paid, interest, balance)| {
                serde_json::json!({
                    "payment": payment,
                    "principal": principal_paid,
                    "interest": interest,
                    "balance": balance,
                })
            })
            .collect();
        data.push(serde_json::Value::Array(items).to_string().into_bytes());
    }
    Ok(Column::with_nulls(
        ColumnData::Binary(data),
        nulls,
        TypeId::Array,
    ))
}

// ---------------------------------------------------------------------------
// timeseries
// ---------------------------------------------------------------------------

// lttb(timestamps, values, threshold) -> Array of retained point indices
// mismatched array lengths yield NULL for that row
fn lttb_impl(args: &[Column]) -> Result<Column> {
    const SIG: &str = "lttb(timestamps, values, threshold)";
    if args.len() != 3 {
        return Err(ZyronError::ExecutionError(format!(
            "{SIG} takes exactly 3 arguments"
        )));
    }
    check_array_column(SIG, &args[0])?;
    check_array_column(SIG, &args[1])?;
    let thresholds = numeric_column(SIG, &args[2])?;
    let n = args[0]
        .data
        .len()
        .min(args[1].data.len())
        .min(thresholds.len());
    let mut data: Vec<Vec<u8>> = Vec::with_capacity(n);
    let mut nulls = NullBitmap::none(n);
    for i in 0..n {
        if args.iter().any(|c| c.nulls.is_null(i)) {
            data.push(Vec::new());
            nulls.set_null(i);
            continue;
        }
        let ts = parse_f64_array(&args[0], i);
        let vals = parse_f64_array(&args[1], i);
        match (ts, vals) {
            (Some(ts), Some(vals)) if ts.len() == vals.len() => {
                let threshold = thresholds[i].max(0.0) as usize;
                let indices = zyron_types::timeseries::lttb(&ts, &vals, threshold);
                let arr: Vec<serde_json::Value> = indices
                    .iter()
                    .map(|&ix| serde_json::Value::from(ix as u64))
                    .collect();
                data.push(serde_json::Value::Array(arr).to_string().into_bytes());
            }
            _ => {
                data.push(Vec::new());
                nulls.set_null(i);
            }
        }
    }
    Ok(Column::with_nulls(
        ColumnData::Binary(data),
        nulls,
        TypeId::Array,
    ))
}

// time_bucket_calendar(interval, ts) floors ts to a calendar aligned boundary
fn time_bucket_calendar_impl(args: &[Column]) -> Result<Column> {
    const SIG: &str = "time_bucket_calendar(interval, timestamp)";
    if args.len() != 2 {
        return Err(ZyronError::ExecutionError(format!(
            "{SIG} takes exactly 2 arguments"
        )));
    }
    let intervals = column_intervals(SIG, &args[0])?;
    let timestamps = super::column_ints(&args[1]).map_err(|_| {
        ZyronError::ExecutionError(format!("{SIG} expects a timestamp second argument"))
    })?;
    let n = intervals.len().min(timestamps.len());
    let mut data = Vec::with_capacity(n);
    let mut nulls = NullBitmap::none(n);
    for i in 0..n {
        if args[0].nulls.is_null(i) || args[1].nulls.is_null(i) {
            data.push(0);
            nulls.set_null(i);
            continue;
        }
        data.push(zyron_types::timeseries::time_bucket_calendar(
            intervals[i],
            timestamps[i],
        ));
    }
    Ok(Column::with_nulls(
        ColumnData::Int64(data),
        nulls,
        TypeId::Timestamp,
    ))
}

// time_bucket_gapfill_calendar(interval, start, end) -> Array of bucket
// boundary micros in [start, end) as JSON text bytes
fn time_bucket_gapfill_calendar_impl(args: &[Column]) -> Result<Column> {
    const SIG: &str = "time_bucket_gapfill_calendar(interval, start, end)";
    if args.len() != 3 {
        return Err(ZyronError::ExecutionError(format!(
            "{SIG} takes exactly 3 arguments"
        )));
    }
    let intervals = column_intervals(SIG, &args[0])?;
    let starts = super::column_ints(&args[1]).map_err(|_| {
        ZyronError::ExecutionError(format!("{SIG} expects timestamp range arguments"))
    })?;
    let ends = super::column_ints(&args[2]).map_err(|_| {
        ZyronError::ExecutionError(format!("{SIG} expects timestamp range arguments"))
    })?;
    let n = intervals.len().min(starts.len()).min(ends.len());
    let mut data: Vec<Vec<u8>> = Vec::with_capacity(n);
    let mut nulls = NullBitmap::none(n);
    for i in 0..n {
        if args.iter().any(|c| c.nulls.is_null(i)) {
            data.push(Vec::new());
            nulls.set_null(i);
            continue;
        }
        let buckets =
            zyron_types::timeseries::time_bucket_gapfill_calendar(intervals[i], starts[i], ends[i]);
        let arr: Vec<serde_json::Value> = buckets
            .iter()
            .map(|&b| serde_json::Value::from(b))
            .collect();
        data.push(serde_json::Value::Array(arr).to_string().into_bytes());
    }
    Ok(Column::with_nulls(
        ColumnData::Binary(data),
        nulls,
        TypeId::Array,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn f64_col(vals: Vec<f64>) -> Column {
        Column::new(ColumnData::Float64(vals), TypeId::Float64)
    }

    fn json_col(cells: Vec<&str>) -> Column {
        Column::new(
            ColumnData::Binary(cells.into_iter().map(|s| s.as_bytes().to_vec()).collect()),
            TypeId::Array,
        )
    }

    fn float_values(col: &Column) -> Vec<f64> {
        match &col.data {
            ColumnData::Float64(v) => v.clone(),
            other => panic!("expected Float64, got {other:?}"),
        }
    }

    fn binary_cell_json(col: &Column, row: usize) -> serde_json::Value {
        match &col.data {
            ColumnData::Binary(v) => serde_json::from_slice(&v[row]).unwrap(),
            other => panic!("expected Binary, got {other:?}"),
        }
    }

    fn ymd_micros(y: i32, m: u32, d: u32) -> i64 {
        (zyron_common::days_from_ymd(y, m, d) as i64) * 86_400_000_000
    }

    #[test]
    fn pmt_zero_rate() {
        let out = dispatch(
            "pmt",
            &[
                f64_col(vec![0.0]),
                f64_col(vec![10.0]),
                f64_col(vec![1000.0]),
            ],
            1,
        )
        .unwrap()
        .unwrap();
        assert!((float_values(&out)[0] - (-100.0)).abs() < 1e-10);
        assert!(!out.nulls.is_null(0));
    }

    #[test]
    fn npv_json_cashflows() {
        let out = dispatch(
            "npv",
            &[f64_col(vec![0.1]), json_col(vec!["[-100, 60, 60]"])],
            1,
        )
        .unwrap()
        .unwrap();
        assert!((float_values(&out)[0] - 4.132).abs() < 0.01);
    }

    #[test]
    fn irr_domain_failure_yields_null() {
        let out = dispatch(
            "irr",
            &[json_col(vec!["[-100, 110]", "[100, 200]", "not json"])],
            3,
        )
        .unwrap()
        .unwrap();
        let vals = float_values(&out);
        assert!((vals[0] - 0.10).abs() < 1e-6);
        assert!(out.nulls.is_null(1));
        assert!(out.nulls.is_null(2));
    }

    #[test]
    fn null_input_row_propagates() {
        let mut nulls = NullBitmap::none(2);
        nulls.set_null(0);
        let rate = Column::with_nulls(ColumnData::Float64(vec![0.0, 0.0]), nulls, TypeId::Float64);
        let out = dispatch(
            "pmt",
            &[
                rate,
                f64_col(vec![10.0, 10.0]),
                f64_col(vec![1000.0, 1000.0]),
            ],
            2,
        )
        .unwrap()
        .unwrap();
        assert!(out.nulls.is_null(0));
        assert!(!out.nulls.is_null(1));
        assert!((float_values(&out)[1] - (-100.0)).abs() < 1e-10);
    }

    #[test]
    fn amortization_schedule_json_shape() {
        let out = dispatch(
            "amortization_schedule",
            &[
                f64_col(vec![1000.0]),
                f64_col(vec![0.0]),
                f64_col(vec![4.0]),
            ],
            1,
        )
        .unwrap()
        .unwrap();
        let parsed = binary_cell_json(&out, 0);
        let arr = parsed.as_array().unwrap();
        assert_eq!(arr.len(), 4);
        assert!((arr[0]["payment"].as_f64().unwrap() - 250.0).abs() < 1e-10);
        assert!((arr[3]["balance"].as_f64().unwrap()).abs() < 1e-6);
    }

    #[test]
    fn lttb_below_threshold_returns_all_indices() {
        let out = dispatch(
            "lttb",
            &[
                json_col(vec!["[0, 1, 2]"]),
                json_col(vec!["[1.0, 2.0, 3.0]"]),
                f64_col(vec![10.0]),
            ],
            1,
        )
        .unwrap()
        .unwrap();
        let parsed = binary_cell_json(&out, 0);
        assert_eq!(parsed, serde_json::json!([0, 1, 2]));
    }

    #[test]
    fn time_bucket_calendar_month_alignment() {
        let iv = Column::new(
            ColumnData::Interval(vec![Interval::from_months(1)]),
            TypeId::Interval,
        );
        let ts = Column::new(
            ColumnData::Int64(vec![ymd_micros(2024, 3, 15) + 5 * 3_600_000_000]),
            TypeId::Timestamp,
        );
        let out = dispatch("time_bucket_calendar", &[iv, ts], 1)
            .unwrap()
            .unwrap();
        match &out.data {
            ColumnData::Int64(v) => assert_eq!(v[0], ymd_micros(2024, 3, 1)),
            other => panic!("expected Int64, got {other:?}"),
        }
        assert_eq!(out.type_id, TypeId::Timestamp);
    }

    #[test]
    fn gapfill_calendar_month_boundaries() {
        let iv = Column::new(
            ColumnData::Interval(vec![Interval::from_months(1)]),
            TypeId::Interval,
        );
        let start = Column::new(
            ColumnData::Int64(vec![ymd_micros(2024, 1, 15)]),
            TypeId::Timestamp,
        );
        let end = Column::new(
            ColumnData::Int64(vec![ymd_micros(2024, 4, 1)]),
            TypeId::Timestamp,
        );
        let out = dispatch("time_bucket_gapfill_calendar", &[iv, start, end], 1)
            .unwrap()
            .unwrap();
        let parsed = binary_cell_json(&out, 0);
        assert_eq!(
            parsed,
            serde_json::json!([ymd_micros(2024, 2, 1), ymd_micros(2024, 3, 1)])
        );
    }

    #[test]
    fn unknown_name_returns_none() {
        assert!(dispatch("not_a_financial_fn", &[], 1).is_none());
    }
}
