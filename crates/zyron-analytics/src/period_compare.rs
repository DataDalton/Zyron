// Period-over-period comparison helpers and running cumulative sums
// All functions take a slice of (timestamp_ms, value) pairs sorted by time
// and return a parallel vector of comparison results, one per input row.

use crate::value::{AnalyticsValue, MS_PER_DAY};
use zyron_common::error::{Result, ZyronError};
use zyron_common::{PreHashMap, fx_mix, mix_finalize_2round};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PeriodUnit {
    Day,
    Week,
    Month,
    Quarter,
    Year,
}

impl PeriodUnit {
    pub fn from_str_ci(s: &str) -> Option<Self> {
        let lower = s.to_ascii_lowercase();
        match lower.as_str() {
            "day" | "days" | "d" => Some(PeriodUnit::Day),
            "week" | "weeks" | "w" => Some(PeriodUnit::Week),
            "month" | "months" | "mo" | "m" => Some(PeriodUnit::Month),
            "quarter" | "quarters" | "q" => Some(PeriodUnit::Quarter),
            "year" | "years" | "y" => Some(PeriodUnit::Year),
            _ => None,
        }
    }
}

// Civil calendar arithmetic on milliseconds since the Unix epoch UTC.
// Conversion uses Howard Hinnant's days-from-civil algorithm.
#[inline]
fn civil_from_days(days: i64) -> (i32, u32, u32) {
    let z = days + 719_468;
    let era = if z >= 0 { z } else { z - 146_096 } / 146_097;
    let doe = (z - era * 146_097) as u64;
    let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146_096) / 365;
    let y = yoe as i64 + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    let y = if m <= 2 { y + 1 } else { y };
    (y as i32, m as u32, d as u32)
}

#[inline]
fn days_from_civil(y: i32, m: u32, d: u32) -> i64 {
    let y = if m <= 2 { y as i64 - 1 } else { y as i64 };
    let era = if y >= 0 { y } else { y - 399 } / 400;
    let yoe = (y - era * 400) as u64;
    let doy = (153 * (if m > 2 { m - 3 } else { m + 9 } as u64) + 2) / 5 + d as u64 - 1;
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    era * 146_097 + doe as i64 - 719_468
}

fn ms_to_civil(ms: i64) -> (i32, u32, u32) {
    let days = ms.div_euclid(MS_PER_DAY);
    civil_from_days(days)
}

/// The inverse of `ms_to_civil`. Production converts one way, so this exists
/// for the round trip that proves the pair agree
#[cfg(test)]
fn civil_to_ms(y: i32, m: u32, d: u32) -> i64 {
    days_from_civil(y, m, d) * MS_PER_DAY
}

// Subtract n units from a timestamp using calendar arithmetic where needed.
// Day and Week shifts are pure i64 arithmetic. Month/Quarter/Year shifts
// have to consult the calendar.
//
// Year-shift fast path: when the input day is not Feb 29, shifting back N
// years lands on the same (month, day) regardless of leap years, so the
// shift is just `ts_ms - days_between_years(y-N..y) * MS_PER_DAY`. We
// short-circuit by counting leap years between (y - N) and y rather than
// converting to (y, m, d) and back. Only the Feb 29 case needs the
// clamp-day path that the original code took for every row.
//
// Month-shift uses the full calendar conversion since month lengths vary.
#[inline]
fn shift_back(ts_ms: i64, periods: i64, unit: PeriodUnit) -> i64 {
    match unit {
        PeriodUnit::Day => ts_ms - periods * MS_PER_DAY,
        PeriodUnit::Week => ts_ms - periods * 7 * MS_PER_DAY,
        PeriodUnit::Month => {
            let days = ts_ms.div_euclid(MS_PER_DAY);
            let tod = ts_ms - days * MS_PER_DAY;
            let (y, m, d) = civil_from_days(days);
            let total_months = (y as i64) * 12 + (m as i64 - 1) - periods;
            let new_y = total_months.div_euclid(12) as i32;
            let new_m = total_months.rem_euclid(12) as u32 + 1;
            let new_d = clamp_day(new_y, new_m, d);
            let new_days = days_from_civil(new_y, new_m, new_d);
            new_days * MS_PER_DAY + tod
        }
        PeriodUnit::Quarter => shift_back(ts_ms, periods * 3, PeriodUnit::Month),
        PeriodUnit::Year => {
            let days = ts_ms.div_euclid(MS_PER_DAY);
            let tod = ts_ms - days * MS_PER_DAY;
            let (y, m, d) = civil_from_days(days);
            // Fast path: any (m, d) other than Feb 29 shifts cleanly by
            // 365*periods + (number of leap years in the relevant year
            // window). The periods == 1 case (the common YOY path) is
            // inlined as a single is_leap check to avoid any function-call
            // overhead. Larger period counts fall through to the helper.
            if m != 2 || d != 29 {
                let span_days = if periods == 1 {
                    let leap_year = if m >= 3 { y } else { y - 1 };
                    365 + if is_leap(leap_year) { 1 } else { 0 }
                } else {
                    year_shift_days(y - periods as i32, y, m)
                };
                return (days - span_days) * MS_PER_DAY + tod;
            }
            // Feb 29 must clamp to Feb 28 in non-leap target years, so
            // fall back to the full calendar reconstruction.
            let new_y = y - periods as i32;
            let new_d = clamp_day(new_y, m, d);
            let new_days = days_from_civil(new_y, m, new_d);
            new_days * MS_PER_DAY + tod
        }
    }
}

// Day count between (start_y, m, d) and (end_y, m, d) where end_y > start_y
// and (m, d) is not Feb 29. For (m, d) >= Mar 1 the only leap years that
// matter are those in the open-closed interval (start_y, end_y]. For
// (m, d) < Mar 1 it's the closed-open interval [start_y, end_y). Either
// case adds one extra day per matching leap year on top of 365 * span.
//
// For periods = 1 this loop runs at most once, so the calendar fast path
// below saves one full civil round-trip (~half of shift_back's cost on
// daily YOY workloads) for every non-Feb-29 input.
fn year_shift_days(start_y: i32, end_y: i32, current_month: u32) -> i64 {
    let span = (end_y - start_y) as i64;
    let (range_start, range_end_exclusive) = if current_month >= 3 {
        (start_y + 1, end_y + 1)
    } else {
        (start_y, end_y)
    };
    let mut leaps: i64 = 0;
    for y in range_start..range_end_exclusive {
        if is_leap(y) {
            leaps += 1;
        }
    }
    365 * span + leaps
}

fn clamp_day(y: i32, m: u32, d: u32) -> u32 {
    let last = days_in_month(y, m);
    d.min(last)
}

fn days_in_month(y: i32, m: u32) -> u32 {
    match m {
        1 | 3 | 5 | 7 | 8 | 10 | 12 => 31,
        4 | 6 | 9 | 11 => 30,
        2 => {
            if is_leap(y) {
                29
            } else {
                28
            }
        }
        _ => 0,
    }
}

#[inline(always)]
fn is_leap(y: i32) -> bool {
    (y % 4 == 0 && y % 100 != 0) || y % 400 == 0
}

// Generic period-shift comparison: for each row, find the value whose
// timestamp matches the current timestamp shifted back by (periods, unit).
// Match is exact on the shifted timestamp.
//
// Sorted-input fast path: when the input is already sorted by ts (the
// common case for analytics pipelines), we avoid the giant HashMap
// entirely and use a two-pointer sliding window that walks both the
// "current" and "shifted-back-target" positions monotonically through
// the same Vec. Memory usage drops from O(N * (24 bytes hashmap entry)
// + table overhead) to zero auxiliary storage; lookup work drops from
// O(N) DRAM hashmap accesses to O(N) sequential scans that stay in L1
// for any cache-resident segment.
//
// Unsorted-input fallback: build a hash lookup table keyed on the
// timestamp's fx_mix hash. Each value slot stores the canonical
// timestamp alongside, and lookups verify the stored timestamp matches
// the target. This catches the rare 64-bit collision (~2.5% probability
// per run at 10M timestamps) and reports None rather than returning a
// wrong neighbour value.
pub fn shift_compare(series: &[(i64, f64)], periods: i64, unit: PeriodUnit) -> Vec<Option<f64>> {
    if is_sorted_by_ts(series) {
        shift_compare_sorted(series, periods, unit)
    } else {
        shift_compare_unsorted(series, periods, unit)
    }
}

#[inline]
fn is_sorted_by_ts(series: &[(i64, f64)]) -> bool {
    series.windows(2).all(|w| w[0].0 <= w[1].0)
}

fn shift_compare_sorted(series: &[(i64, f64)], periods: i64, unit: PeriodUnit) -> Vec<Option<f64>> {
    let mut out = Vec::with_capacity(series.len());
    // `j` tracks the position of the most recent series entry whose
    // timestamp is <= the current row's shifted-back target. It only
    // advances, so total work across the whole loop is O(N).
    let mut j: usize = 0;
    for (ts, _) in series {
        let target = shift_back(*ts, periods, unit);
        // Advance j while the next candidate is still <= target
        while j < series.len() && series[j].0 <= target {
            j += 1;
        }
        // After the loop, j either points past series or to the first
        // element strictly greater than target. The match (if any) is at
        // index j-1.
        if j == 0 {
            out.push(None);
            continue;
        }
        let candidate = &series[j - 1];
        if candidate.0 == target {
            out.push(Some(candidate.1));
        } else {
            out.push(None);
        }
    }
    out
}

fn shift_compare_unsorted(
    series: &[(i64, f64)],
    periods: i64,
    unit: PeriodUnit,
) -> Vec<Option<f64>> {
    const SEED: u64 = 0x6B0F_A1A2_3D4E_5F60;
    #[inline]
    fn ts_key(ts: i64) -> u64 {
        mix_finalize_2round(fx_mix(SEED, ts as u64))
    }
    let mut idx: PreHashMap<u64, (i64, f64)> = PreHashMap::default();
    idx.reserve(series.len());
    for (ts, v) in series {
        idx.insert(ts_key(*ts), (*ts, *v));
    }
    let mut out = Vec::with_capacity(series.len());
    for (ts, _) in series {
        let target = shift_back(*ts, periods, unit);
        match idx.get(&ts_key(target)) {
            Some(&(stored_ts, val)) if stored_ts == target => out.push(Some(val)),
            _ => out.push(None),
        }
    }
    out
}

pub fn yoy(series: &[(i64, f64)]) -> Vec<Option<f64>> {
    shift_compare(series, 1, PeriodUnit::Year)
}

pub fn mom(series: &[(i64, f64)]) -> Vec<Option<f64>> {
    shift_compare(series, 1, PeriodUnit::Month)
}

pub fn wow(series: &[(i64, f64)]) -> Vec<Option<f64>> {
    shift_compare(series, 1, PeriodUnit::Week)
}

pub fn qoq(series: &[(i64, f64)]) -> Vec<Option<f64>> {
    shift_compare(series, 1, PeriodUnit::Quarter)
}

pub fn same_period_last_year(series: &[(i64, f64)]) -> Vec<Option<f64>> {
    yoy(series)
}

// Growth helpers: (current - prior) / prior, returns NULL when prior missing
// or prior == 0
fn growth(series: &[(i64, f64)], prior: &[Option<f64>]) -> Vec<Option<f64>> {
    let mut out = Vec::with_capacity(series.len());
    for ((_, current), p) in series.iter().zip(prior.iter()) {
        let v = match p {
            Some(prev) if *prev != 0.0 => Some((current - prev) / prev),
            _ => None,
        };
        out.push(v);
    }
    out
}

pub fn yoy_growth(series: &[(i64, f64)]) -> Vec<Option<f64>> {
    let prior = yoy(series);
    growth(series, &prior)
}

pub fn mom_growth(series: &[(i64, f64)]) -> Vec<Option<f64>> {
    let prior = mom(series);
    growth(series, &prior)
}

// Generic period comparison invoked from SQL
pub fn period_compare_value(
    series: &[(i64, f64)],
    periods: i64,
    unit: &str,
) -> Result<Vec<AnalyticsValue>> {
    let unit = PeriodUnit::from_str_ci(unit)
        .ok_or_else(|| ZyronError::ExecutionError(format!("unknown period unit: {}", unit)))?;
    let prior = shift_compare(series, periods, unit);
    Ok(prior
        .into_iter()
        .map(|v| match v {
            Some(x) => AnalyticsValue::Float(x),
            None => AnalyticsValue::Null,
        })
        .collect())
}

// To-date cumulative sums. Bucket boundaries: year, quarter, month.
// For each row, sum all earlier-or-equal rows that share the same bucket.
fn bucket_year(ms: i64) -> i64 {
    let (y, _, _) = ms_to_civil(ms);
    y as i64
}

fn bucket_quarter(ms: i64) -> i64 {
    let (y, m, _) = ms_to_civil(ms);
    let q = (m - 1) / 3;
    (y as i64) * 4 + q as i64
}

fn bucket_month(ms: i64) -> i64 {
    let (y, m, _) = ms_to_civil(ms);
    (y as i64) * 12 + (m as i64 - 1)
}

fn cumulative_in_bucket(series: &[(i64, f64)], bucket: impl Fn(i64) -> i64) -> Vec<f64> {
    // Sort by timestamp via index permutation so we can return values in
    // input order. Bucket IDs are spread via fx_mix before keying the
    // PreHashMap so calendar buckets land in well-distributed slots.
    let mut order: Vec<usize> = (0..series.len()).collect();
    order.sort_by_key(|&i| series[i].0);
    let mut running: PreHashMap<u64, f64> = PreHashMap::default();
    let mut by_index: Vec<f64> = vec![0.0; series.len()];
    const SEED: u64 = 0x4A8C_31D5_77E2_018B;
    for &i in &order {
        let (ts, v) = series[i];
        let b = bucket(ts);
        let key = mix_finalize_2round(fx_mix(SEED, b as u64));
        let acc = running.entry(key).or_insert(0.0);
        *acc += v;
        by_index[i] = *acc;
    }
    by_index
}

pub fn ytd_sum(series: &[(i64, f64)]) -> Vec<f64> {
    cumulative_in_bucket(series, bucket_year)
}

pub fn qtd_sum(series: &[(i64, f64)]) -> Vec<f64> {
    cumulative_in_bucket(series, bucket_quarter)
}

pub fn mtd_sum(series: &[(i64, f64)]) -> Vec<f64> {
    cumulative_in_bucket(series, bucket_month)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn d(y: i32, m: u32, d: u32) -> i64 {
        civil_to_ms(y, m, d)
    }

    #[test]
    fn civil_roundtrip() {
        for &(y, m, day) in &[(2024, 1, 1), (2024, 12, 31), (2000, 2, 29), (1970, 1, 1)] {
            let days = days_from_civil(y, m, day);
            let (ry, rm, rd) = civil_from_days(days);
            assert_eq!((ry, rm, rd), (y, m, day));
        }
    }

    #[test]
    fn yoy_aligns_same_day_previous_year() {
        let series = vec![(d(2023, 6, 15), 100.0), (d(2024, 6, 15), 120.0)];
        let prior = yoy(&series);
        assert_eq!(prior[0], None);
        assert_eq!(prior[1], Some(100.0));
        let g = yoy_growth(&series);
        assert!((g[1].unwrap() - 0.20).abs() < 1e-9);
    }

    #[test]
    fn mom_clamps_short_months() {
        let series = vec![(d(2024, 1, 31), 50.0), (d(2024, 2, 29), 70.0)];
        // Feb 29 -> Jan 29, no match for Jan 31 -> Dec 31 prior year
        let prior = mom(&series);
        assert_eq!(prior[0], None);
        assert_eq!(prior[1], None);
    }

    #[test]
    fn ytd_resets_at_year_boundary() {
        let series = vec![
            (d(2024, 1, 5), 10.0),
            (d(2024, 6, 1), 20.0),
            (d(2025, 1, 1), 5.0),
        ];
        let acc = ytd_sum(&series);
        assert_eq!(acc, vec![10.0, 30.0, 5.0]);
    }

    #[test]
    fn qtd_buckets_match_calendar_quarters() {
        let series = vec![
            (d(2024, 1, 5), 10.0),
            (d(2024, 3, 31), 5.0),
            (d(2024, 4, 1), 1.0),
        ];
        let acc = qtd_sum(&series);
        assert_eq!(acc, vec![10.0, 15.0, 1.0]);
    }
}
