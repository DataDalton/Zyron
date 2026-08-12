//! id_gen, identifier, rating, and semver dispatch arms
//! bridges zyron-types generator, identifier extractor, competitive rating,
//! and packed semver functions into Column-based evaluation

use crate::column::{Column, ColumnData, NullBitmap};
use std::sync::atomic::AtomicU64;
use zyron_common::{Result, TypeId, ZyronError};

// process wide sequence state so snowflake ids stay unique and monotonic across calls
static SNOWFLAKE_STATE: AtomicU64 = AtomicU64::new(0);

pub(super) fn dispatch(name: &str, args: &[Column], num_rows: usize) -> Option<Result<Column>> {
    Some(match name {
        // id_gen names lacking arms in mod.rs, the gen_* spellings there win first
        "nanoid" => nanoid_gen(args, num_rows),
        "snowflake" | "gen_snowflake" => snowflake_gen(args, num_rows),
        "uuid_to_string" => uuid_to_string_impl(args),

        // identifier extractors, per row parse failure yields NULL
        "iban_country" => one_string_fallible_to_string(
            "iban_country(iban)",
            args,
            zyron_types::identifier::iban_country,
        ),
        "iban_bban" => one_string_fallible_to_string(
            "iban_bban(iban)",
            args,
            zyron_types::identifier::iban_bban,
        ),
        "isbn_to_13" => one_string_fallible_to_string(
            "isbn_to_13(isbn)",
            args,
            zyron_types::identifier::isbn_to_13,
        ),
        "isbn_format" => isbn_format_impl(args),
        "vin_country" => one_string_fallible_to_string(
            "vin_country(vin)",
            args,
            zyron_types::identifier::vin_country,
        ),
        "vin_manufacturer" => one_string_fallible_to_string(
            "vin_manufacturer(vin)",
            args,
            zyron_types::identifier::vin_manufacturer,
        ),
        "vin_year" => one_string_fallible_to_int32("vin_year(vin)", args, |s| {
            zyron_types::identifier::vin_year(s).map(|y| y as i32)
        }),

        // rating functions, counts read as floats then clamped to u64
        "elo_expected" => n_floats_to_float("elo_expected(rating_a, rating_b)", args, 2, |r| {
            Ok(zyron_types::rating::elo_expected(r[0], r[1]))
        }),
        "elo_update" => n_floats_to_float(
            "elo_update(rating, expected, actual, k_factor)",
            args,
            4,
            |r| Ok(zyron_types::rating::elo_update(r[0], r[1], r[2], r[3])),
        ),
        "bayesian_average" => n_floats_to_float(
            "bayesian_average(item_avg, item_count, global_avg, min_votes)",
            args,
            4,
            |r| {
                Ok(zyron_types::rating::bayesian_average(
                    r[0],
                    r[1].max(0.0) as u64,
                    r[2],
                    r[3].max(0.0) as u64,
                ))
            },
        ),
        "win_rate" => n_floats_to_float("win_rate(wins, total)", args, 2, |r| {
            Ok(zyron_types::rating::win_rate(
                r[0].max(0.0) as u64,
                r[1].max(0.0) as u64,
            ))
        }),
        // invalid confidence is call misuse not bad data, the error propagates
        "wilson_score" => {
            n_floats_to_float("wilson_score(positive, total, confidence)", args, 3, |r| {
                zyron_types::rating::wilson_score(r[0].max(0.0) as u64, r[1].max(0.0) as u64, r[2])
            })
        }
        "glicko2_update" => glicko2_impl(args),
        "trueskill_update" => trueskill_impl(args),

        // semver, packed u64 in UInt64 cells, semver_compare has an arm in mod.rs
        "semver_parse" => semver_parse_impl(args),
        "semver_format" => one_u64_to_string(
            "semver_format(version)",
            args,
            zyron_types::semver::semver_format,
        ),
        "semver_major" => one_u64_to_int32("semver_major(version)", args, |p| {
            zyron_types::semver::semver_major(p) as i32
        }),
        "semver_minor" => one_u64_to_int32("semver_minor(version)", args, |p| {
            zyron_types::semver::semver_minor(p) as i32
        }),
        "semver_patch" => one_u64_to_int32("semver_patch(version)", args, |p| {
            zyron_types::semver::semver_patch(p) as i32
        }),
        "semver_is_prerelease" => one_u64_to_bool(
            "semver_is_prerelease(version)",
            args,
            zyron_types::semver::semver_is_prerelease,
        ),
        "semver_satisfies" => semver_satisfies_impl(args),
        "semver_increment_major" => one_u64_to_semver(
            "semver_increment_major(version)",
            args,
            zyron_types::semver::semver_increment_major,
        ),
        "semver_increment_minor" => one_u64_to_semver(
            "semver_increment_minor(version)",
            args,
            zyron_types::semver::semver_increment_minor,
        ),
        "semver_increment_patch" => one_u64_to_semver(
            "semver_increment_patch(version)",
            args,
            zyron_types::semver::semver_increment_patch,
        ),
        _ => return None,
    })
}

// ---------------------------------------------------------------------------
// shared helpers
// ---------------------------------------------------------------------------

fn arity(sig: &str, args: &[Column], expected: usize) -> Result<()> {
    if args.len() != expected {
        let unit = if expected == 1 {
            "argument"
        } else {
            "arguments"
        };
        return Err(ZyronError::ExecutionError(format!(
            "{} takes exactly {} {}",
            sig, expected, unit
        )));
    }
    Ok(())
}

fn strings_arg<'a>(sig: &str, col: &'a Column) -> Result<Vec<&'a str>> {
    super::column_strings(col)
        .map_err(|_| ZyronError::ExecutionError(format!("{} expects a text argument", sig)))
}

fn ints_arg(sig: &str, col: &Column) -> Result<Vec<i64>> {
    super::column_ints(col)
        .map_err(|_| ZyronError::ExecutionError(format!("{} expects an integer argument", sig)))
}

fn floats_arg(sig: &str, col: &Column) -> Result<Vec<f64>> {
    super::column_floats(col)
        .map_err(|_| ZyronError::ExecutionError(format!("{} expects a numeric argument", sig)))
}

fn bytes_arg<'a>(sig: &str, col: &'a Column) -> Result<Vec<&'a [u8]>> {
    super::column_bytes(col).map_err(|_| {
        ZyronError::ExecutionError(format!("{} expects a binary or text argument", sig))
    })
}

// packed semver values live in UInt64 cells, integer widths accepted with bit preserving casts
fn u64s_arg(sig: &str, col: &Column) -> Result<Vec<u64>> {
    match &col.data {
        ColumnData::UInt64(v) => Ok(v.clone()),
        ColumnData::Int64(v) => Ok(v.iter().map(|&x| x as u64).collect()),
        ColumnData::UInt32(v) => Ok(v.iter().map(|&x| x as u64).collect()),
        ColumnData::Int32(v) => Ok(v.iter().map(|&x| x as u64).collect()),
        _ => Err(ZyronError::ExecutionError(format!(
            "{} expects a semver argument",
            sig
        ))),
    }
}

// resolves the source row for value i with length 1 broadcast
// None when the row is NULL or out of range
fn broadcast_row(col: &Column, i: usize) -> Option<usize> {
    let len = col.data.len();
    let j = if len == 1 { 0 } else { i };
    if j >= len || col.nulls.is_null(j) {
        None
    } else {
        Some(j)
    }
}

// parser style, a per row Err from f yields NULL for that row
fn one_string_fallible_to_string(
    sig: &str,
    args: &[Column],
    f: impl Fn(&str) -> Result<String>,
) -> Result<Column> {
    arity(sig, args, 1)?;
    let vals = strings_arg(sig, &args[0])?;
    let mut data = Vec::with_capacity(vals.len());
    let mut nulls = NullBitmap::none(vals.len());
    for (i, s) in vals.iter().enumerate() {
        if args[0].nulls.is_null(i) {
            data.push(String::new());
            nulls.set_null(i);
            continue;
        }
        match f(s) {
            Ok(v) => data.push(v),
            Err(_) => {
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

fn one_string_fallible_to_int32(
    sig: &str,
    args: &[Column],
    f: impl Fn(&str) -> Result<i32>,
) -> Result<Column> {
    arity(sig, args, 1)?;
    let vals = strings_arg(sig, &args[0])?;
    let mut data = Vec::with_capacity(vals.len());
    let mut nulls = NullBitmap::none(vals.len());
    for (i, s) in vals.iter().enumerate() {
        if args[0].nulls.is_null(i) {
            data.push(0);
            nulls.set_null(i);
            continue;
        }
        match f(s) {
            Ok(v) => data.push(v),
            Err(_) => {
                data.push(0);
                nulls.set_null(i);
            }
        }
    }
    Ok(Column::with_nulls(
        ColumnData::Int32(data),
        nulls,
        TypeId::Int32,
    ))
}

// row wise float application over N equal shaped columns with NULL union
// closure errors propagate, used for domain misuse like invalid confidence
fn n_floats_to_float(
    sig: &str,
    args: &[Column],
    expected: usize,
    f: impl Fn(&[f64]) -> Result<f64>,
) -> Result<Column> {
    arity(sig, args, expected)?;
    let mut cols = Vec::with_capacity(args.len());
    for c in args {
        cols.push(floats_arg(sig, c)?);
    }
    let n = cols.iter().map(|v| v.len()).max().unwrap_or(0);
    let mut row = vec![0.0f64; cols.len()];
    let mut data = Vec::with_capacity(n);
    let mut nulls = NullBitmap::none(n);
    'rows: for i in 0..n {
        for (k, vals) in cols.iter().enumerate() {
            match broadcast_row(&args[k], i) {
                Some(j) => row[k] = vals[j],
                None => {
                    data.push(0.0);
                    nulls.set_null(i);
                    continue 'rows;
                }
            }
        }
        data.push(f(&row)?);
    }
    Ok(Column::with_nulls(
        ColumnData::Float64(data),
        nulls,
        TypeId::Float64,
    ))
}

fn one_u64_to_string(sig: &str, args: &[Column], f: impl Fn(u64) -> String) -> Result<Column> {
    arity(sig, args, 1)?;
    let vals = u64s_arg(sig, &args[0])?;
    let data: Vec<String> = vals.iter().map(|&v| f(v)).collect();
    Ok(Column::with_nulls(
        ColumnData::Utf8(data),
        args[0].nulls.clone(),
        TypeId::Varchar,
    ))
}

fn one_u64_to_int32(sig: &str, args: &[Column], f: impl Fn(u64) -> i32) -> Result<Column> {
    arity(sig, args, 1)?;
    let vals = u64s_arg(sig, &args[0])?;
    let data: Vec<i32> = vals.iter().map(|&v| f(v)).collect();
    Ok(Column::with_nulls(
        ColumnData::Int32(data),
        args[0].nulls.clone(),
        TypeId::Int32,
    ))
}

fn one_u64_to_bool(sig: &str, args: &[Column], f: impl Fn(u64) -> bool) -> Result<Column> {
    arity(sig, args, 1)?;
    let vals = u64s_arg(sig, &args[0])?;
    let data: Vec<bool> = vals.iter().map(|&v| f(v)).collect();
    Ok(Column::with_nulls(
        ColumnData::Boolean(data),
        args[0].nulls.clone(),
        TypeId::Boolean,
    ))
}

fn one_u64_to_semver(sig: &str, args: &[Column], f: impl Fn(u64) -> u64) -> Result<Column> {
    arity(sig, args, 1)?;
    let vals = u64s_arg(sig, &args[0])?;
    let data: Vec<u64> = vals.iter().map(|&v| f(v)).collect();
    Ok(Column::with_nulls(
        ColumnData::UInt64(data),
        args[0].nulls.clone(),
        TypeId::SemVer,
    ))
}

// ---------------------------------------------------------------------------
// id_gen
// ---------------------------------------------------------------------------

fn nanoid_gen(args: &[Column], num_rows: usize) -> Result<Column> {
    if args.len() > 1 {
        return Err(ZyronError::ExecutionError(
            "nanoid(length) takes at most 1 argument".to_string(),
        ));
    }
    // negative lengths clamp to 0 which selects the default length 21
    let len = args
        .first()
        .and_then(super::column_first_int)
        .unwrap_or(21)
        .max(0) as usize;
    let n = num_rows.max(1);
    let data: Vec<String> = (0..n).map(|_| zyron_types::id_gen::nanoid(len)).collect();
    Ok(Column::new(ColumnData::Utf8(data), TypeId::Varchar))
}

fn snowflake_gen(args: &[Column], num_rows: usize) -> Result<Column> {
    if args.len() > 1 {
        return Err(ZyronError::ExecutionError(
            "snowflake(machine_id) takes at most 1 argument".to_string(),
        ));
    }
    // optional machine id masked to 10 bits, defaults to 0
    let machine = args.first().and_then(super::column_first_int).unwrap_or(0);
    let machine_id = (machine as u16) & 0x03FF;
    let n = num_rows.max(1);
    let data: Vec<i64> = (0..n)
        .map(|_| zyron_types::id_gen::snowflake(machine_id, &SNOWFLAKE_STATE))
        .collect();
    Ok(Column::new(ColumnData::Int64(data), TypeId::Int64))
}

fn uuid_to_string_impl(args: &[Column]) -> Result<Column> {
    let sig = "uuid_to_string(uuid)";
    arity(sig, args, 1)?;
    let col = &args[0];
    let n = col.data.len();
    let mut data = Vec::with_capacity(n);
    let mut nulls = NullBitmap::none(n);
    match &col.data {
        ColumnData::FixedBinary16(v) => {
            for (i, bytes) in v.iter().enumerate() {
                if col.nulls.is_null(i) {
                    data.push(String::new());
                    nulls.set_null(i);
                } else {
                    data.push(zyron_types::id_gen::uuid_to_string(bytes));
                }
            }
        }
        // tolerant binary path, a cell that is not 16 bytes yields NULL
        ColumnData::Binary(v) => {
            for (i, cell) in v.iter().enumerate() {
                if col.nulls.is_null(i) {
                    data.push(String::new());
                    nulls.set_null(i);
                    continue;
                }
                match <[u8; 16]>::try_from(cell.as_slice()) {
                    Ok(b) => data.push(zyron_types::id_gen::uuid_to_string(&b)),
                    Err(_) => {
                        data.push(String::new());
                        nulls.set_null(i);
                    }
                }
            }
        }
        _ => {
            return Err(ZyronError::ExecutionError(format!(
                "{} expects a uuid argument",
                sig
            )));
        }
    }
    Ok(Column::with_nulls(
        ColumnData::Utf8(data),
        nulls,
        TypeId::Varchar,
    ))
}

// ---------------------------------------------------------------------------
// identifier
// ---------------------------------------------------------------------------

fn isbn_format_impl(args: &[Column]) -> Result<Column> {
    let sig = "isbn_format(isbn, version)";
    arity(sig, args, 2)?;
    let texts = strings_arg(sig, &args[0])?;
    let versions = ints_arg(sig, &args[1])?;
    let n = texts.len().max(versions.len());
    let mut data = Vec::with_capacity(n);
    let mut nulls = NullBitmap::none(n);
    for i in 0..n {
        let (Some(jt), Some(jv)) = (broadcast_row(&args[0], i), broadcast_row(&args[1], i)) else {
            data.push(String::new());
            nulls.set_null(i);
            continue;
        };
        // out of range version maps to 0 which the domain fn rejects, yielding NULL
        let version = u8::try_from(versions[jv]).unwrap_or(0);
        match zyron_types::identifier::isbn_format(texts[jt], version) {
            Ok(v) => data.push(v),
            Err(_) => {
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

// ---------------------------------------------------------------------------
// rating aggregate style updates over Array encoded args
// ---------------------------------------------------------------------------

// opponents arrive as JSON text bytes in a Binary cell, [[rating, rd, score], ..]
// output cell is JSON [new_rating, new_rd, new_volatility]
// malformed opponent json is a data error, the row yields NULL
fn glicko2_impl(args: &[Column]) -> Result<Column> {
    let sig = "glicko2_update(rating, rd, volatility, opponents)";
    arity(sig, args, 4)?;
    let ratings = floats_arg(sig, &args[0])?;
    let rds = floats_arg(sig, &args[1])?;
    let vols = floats_arg(sig, &args[2])?;
    let opps = bytes_arg(sig, &args[3])?;
    let n = [ratings.len(), rds.len(), vols.len(), opps.len()]
        .into_iter()
        .max()
        .unwrap_or(0);
    let mut data: Vec<Vec<u8>> = Vec::with_capacity(n);
    let mut nulls = NullBitmap::none(n);
    for i in 0..n {
        let (Some(jr), Some(jd), Some(jv), Some(jo)) = (
            broadcast_row(&args[0], i),
            broadcast_row(&args[1], i),
            broadcast_row(&args[2], i),
            broadcast_row(&args[3], i),
        ) else {
            data.push(Vec::new());
            nulls.set_null(i);
            continue;
        };
        let opponents: Vec<(f64, f64, f64)> = match serde_json::from_slice(opps[jo]) {
            Ok(v) => v,
            Err(_) => {
                data.push(Vec::new());
                nulls.set_null(i);
                continue;
            }
        };
        let encoded =
            zyron_types::rating::glicko2_update(ratings[jr], rds[jd], vols[jv], &opponents)
                .ok()
                .and_then(|(nr, nrd, nv)| serde_json::to_vec(&[nr, nrd, nv]).ok());
        match encoded {
            Some(bytes) => data.push(bytes),
            None => {
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

// team ratings arrive as JSON [[mu, sigma], ..] and ranks as JSON [0, 1, ..]
// output cell is JSON [[new_mu, new_sigma], ..] in rating order
// malformed json or mismatched lengths are data errors, the row yields NULL
fn trueskill_impl(args: &[Column]) -> Result<Column> {
    let sig = "trueskill_update(team_ratings, ranks)";
    arity(sig, args, 2)?;
    let teams = bytes_arg(sig, &args[0])?;
    let ranks = bytes_arg(sig, &args[1])?;
    let n = teams.len().max(ranks.len());
    let mut data: Vec<Vec<u8>> = Vec::with_capacity(n);
    let mut nulls = NullBitmap::none(n);
    for i in 0..n {
        let (Some(jt), Some(jr)) = (broadcast_row(&args[0], i), broadcast_row(&args[1], i)) else {
            data.push(Vec::new());
            nulls.set_null(i);
            continue;
        };
        let team_ratings: Vec<(f64, f64)> = match serde_json::from_slice(teams[jt]) {
            Ok(v) => v,
            Err(_) => {
                data.push(Vec::new());
                nulls.set_null(i);
                continue;
            }
        };
        let rank_vals: Vec<u32> = match serde_json::from_slice(ranks[jr]) {
            Ok(v) => v,
            Err(_) => {
                data.push(Vec::new());
                nulls.set_null(i);
                continue;
            }
        };
        let encoded = zyron_types::rating::trueskill_update(&team_ratings, &rank_vals)
            .ok()
            .and_then(|updated| serde_json::to_vec(&updated).ok());
        match encoded {
            Some(bytes) => data.push(bytes),
            None => {
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

// ---------------------------------------------------------------------------
// semver
// ---------------------------------------------------------------------------

// parser style, malformed version text yields NULL
fn semver_parse_impl(args: &[Column]) -> Result<Column> {
    let sig = "semver_parse(text)";
    arity(sig, args, 1)?;
    let vals = strings_arg(sig, &args[0])?;
    let mut data = Vec::with_capacity(vals.len());
    let mut nulls = NullBitmap::none(vals.len());
    for (i, s) in vals.iter().enumerate() {
        if args[0].nulls.is_null(i) {
            data.push(0);
            nulls.set_null(i);
            continue;
        }
        match zyron_types::semver::semver_parse(s) {
            Ok(p) => data.push(p),
            Err(_) => {
                data.push(0);
                nulls.set_null(i);
            }
        }
    }
    Ok(Column::with_nulls(
        ColumnData::UInt64(data),
        nulls,
        TypeId::SemVer,
    ))
}

// a malformed constraint is call misuse not bad data, the error propagates
fn semver_satisfies_impl(args: &[Column]) -> Result<Column> {
    let sig = "semver_satisfies(version, constraint)";
    arity(sig, args, 2)?;
    let versions = u64s_arg(sig, &args[0])?;
    let constraints = strings_arg(sig, &args[1])?;
    let n = versions.len().max(constraints.len());
    let mut data = Vec::with_capacity(n);
    let mut nulls = NullBitmap::none(n);
    for i in 0..n {
        let (Some(jv), Some(jc)) = (broadcast_row(&args[0], i), broadcast_row(&args[1], i)) else {
            data.push(false);
            nulls.set_null(i);
            continue;
        };
        data.push(zyron_types::semver::semver_satisfies(
            versions[jv],
            constraints[jc],
        )?);
    }
    Ok(Column::with_nulls(
        ColumnData::Boolean(data),
        nulls,
        TypeId::Boolean,
    ))
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

    fn float_col(vals: &[f64]) -> Column {
        Column::new(ColumnData::Float64(vals.to_vec()), TypeId::Float64)
    }

    #[test]
    fn semver_parse_format_and_invalid_null() {
        let input = text_col(&["1.2.3", "garbage"]);
        let parsed = dispatch("semver_parse", &[input], 2).unwrap().unwrap();
        assert!(!parsed.nulls.is_null(0));
        assert!(parsed.nulls.is_null(1));

        let formatted = dispatch("semver_format", &[parsed], 2).unwrap().unwrap();
        match &formatted.data {
            ColumnData::Utf8(v) => assert_eq!(v[0], "1.2.3"),
            other => panic!("expected Utf8, got {:?}", other),
        }
        assert!(formatted.nulls.is_null(1));
    }

    #[test]
    fn semver_satisfies_caret_constraint() {
        let packed = zyron_types::semver::semver_parse("1.5.0").unwrap();
        let version = Column::new(ColumnData::UInt64(vec![packed, packed]), TypeId::SemVer);
        let constraint = text_col(&["^1.2.0", "^2.0.0"]);
        let out = dispatch("semver_satisfies", &[version, constraint], 2)
            .unwrap()
            .unwrap();
        match &out.data {
            ColumnData::Boolean(v) => {
                assert!(v[0]);
                assert!(!v[1]);
            }
            other => panic!("expected Boolean, got {:?}", other),
        }
    }

    #[test]
    fn iban_country_propagates_null_input() {
        let mut nulls = NullBitmap::none(2);
        nulls.set_null(1);
        let input = Column::with_nulls(
            ColumnData::Utf8(vec!["DE89370400440532013000".to_string(), String::new()]),
            nulls,
            TypeId::Text,
        );
        let out = dispatch("iban_country", &[input], 2).unwrap().unwrap();
        match &out.data {
            ColumnData::Utf8(v) => assert_eq!(v[0], "DE"),
            other => panic!("expected Utf8, got {:?}", other),
        }
        assert!(!out.nulls.is_null(0));
        assert!(out.nulls.is_null(1));
    }

    #[test]
    fn vin_year_invalid_yields_null() {
        let input = text_col(&["11111111111111111", "short"]);
        let out = dispatch("vin_year", &[input], 2).unwrap().unwrap();
        match &out.data {
            ColumnData::Int32(v) => assert_eq!(v[0], 2001),
            other => panic!("expected Int32, got {:?}", other),
        }
        assert!(!out.nulls.is_null(0));
        assert!(out.nulls.is_null(1));
    }

    #[test]
    fn elo_expected_even_match_is_half() {
        let a = float_col(&[1500.0]);
        let b = float_col(&[1500.0]);
        let out = dispatch("elo_expected", &[a, b], 1).unwrap().unwrap();
        match &out.data {
            ColumnData::Float64(v) => assert!((v[0] - 0.5).abs() < 1e-10),
            other => panic!("expected Float64, got {:?}", other),
        }
    }

    #[test]
    fn wilson_score_invalid_confidence_errors() {
        let positive = float_col(&[70.0]);
        let total = float_col(&[100.0]);
        let confidence = float_col(&[0.0]);
        let out = dispatch("wilson_score", &[positive, total, confidence], 1).unwrap();
        assert!(out.is_err());
    }

    #[test]
    fn glicko2_update_win_raises_rating() {
        let rating = float_col(&[1500.0]);
        let rd = float_col(&[200.0]);
        let vol = float_col(&[0.06]);
        let opponents = Column::new(
            ColumnData::Binary(vec![b"[[1400.0,30.0,1.0]]".to_vec()]),
            TypeId::Array,
        );
        let out = dispatch("glicko2_update", &[rating, rd, vol, opponents], 1)
            .unwrap()
            .unwrap();
        match &out.data {
            ColumnData::Binary(v) => {
                let triple: Vec<f64> = serde_json::from_slice(&v[0]).unwrap();
                assert_eq!(triple.len(), 3);
                assert!(triple[0] > 1500.0);
                assert!(triple[1] < 200.0);
            }
            other => panic!("expected Binary, got {:?}", other),
        }
    }

    #[test]
    fn snowflake_unique_ids_and_min_one_row() {
        let out = dispatch("snowflake", &[], 4).unwrap().unwrap();
        match &out.data {
            ColumnData::Int64(v) => {
                assert_eq!(v.len(), 4);
                let mut sorted = v.clone();
                sorted.sort_unstable();
                sorted.dedup();
                assert_eq!(sorted.len(), 4);
            }
            other => panic!("expected Int64, got {:?}", other),
        }
        let single = dispatch("gen_snowflake", &[], 0).unwrap().unwrap();
        assert_eq!(single.data.len(), 1);
    }
}
