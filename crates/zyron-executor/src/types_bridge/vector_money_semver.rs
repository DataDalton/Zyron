//! Dispatch arms for phonetic comparison, vector space math, money
//! rounding, locale parsing and store backed conversion, semver precedence,
//! and the bloom count estimate
//!
//! Vector cells are Binary payloads of packed little endian f32 with no
//! header, matching the VECTOR column codec. JSON number arrays in Binary
//! or Utf8 cells are accepted as vector input as well
//! Money cells reuse the 10 byte layout shared with the money arms
//! Array results are JSON text bytes inside Binary cells

use crate::column::{Column, ColumnData, NullBitmap};
use zyron_common::{Result, TypeId, ZyronError};

use super::money_quantity::{bidx, decode_money, encode_money, out_len};

pub(super) fn dispatch(name: &str, args: &[Column], _num_rows: usize) -> Option<Result<Column>> {
    Some(match name {
        "phonetic_match" => phonetic_match_impl(args),
        "phonetic_score" => phonetic_score_impl(args),
        "vector_dot" => vector_f64_2(
            "vector_dot(vector, vector)",
            args,
            zyron_types::vector_math::vector_dot,
        ),
        "vector_cross" => vector_vec_2(
            "vector_cross(vector, vector)",
            args,
            zyron_types::vector_math::vector_cross,
        ),
        "vector_norm" => vector_norm_impl(args),
        "vector_normalize" => vector_normalize_impl(args),
        "vector_angle" => vector_f64_2(
            "vector_angle(vector, vector)",
            args,
            zyron_types::vector_math::vector_angle,
        ),
        "money_round" => money_round_impl(args),
        "parse_money" => parse_money_impl(args),
        "convert_currency" => convert_currency_impl(args),
        "semver_prerelease" => semver_prerelease_impl(args),
        "semver_sort" => semver_sort_impl(args),
        "bloom_filter_estimate_count" => bloom_estimate_impl(args),
        _ => return None,
    })
}

// ---------------------------------------------------------------------------
// argument readers
// ---------------------------------------------------------------------------

fn arity_err(sig: &str, arity: &str) -> ZyronError {
    ZyronError::ExecutionError(format!("{} takes {}", sig, arity))
}

fn strings_arg<'a>(sig: &str, col: &'a Column) -> Result<Vec<&'a str>> {
    super::column_strings(col)
        .map_err(|_| ZyronError::ExecutionError(format!("{}: expected a string argument", sig)))
}

fn ints_arg(sig: &str, col: &Column) -> Result<Vec<i64>> {
    super::column_ints(col)
        .map_err(|_| ZyronError::ExecutionError(format!("{}: expected an integer argument", sig)))
}

fn binary_cells<'a>(sig: &str, col: &'a Column) -> Result<&'a Vec<Vec<u8>>> {
    match &col.data {
        ColumnData::Binary(v) => Ok(v),
        _ => Err(ZyronError::ExecutionError(format!(
            "{}: expected a binary encoded argument",
            sig
        ))),
    }
}

// ---------------------------------------------------------------------------
// vector cell codec
// ---------------------------------------------------------------------------

/// Packs values as the little endian f32 sequence a vector column stores
fn encode_vector(values: &[f32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(values.len() * 4);
    for v in values {
        out.extend_from_slice(&v.to_le_bytes());
    }
    out
}

fn decode_packed_vector(sig: &str, cell: &[u8]) -> Result<Vec<f32>> {
    if cell.len() % 4 != 0 {
        return Err(ZyronError::ExecutionError(format!(
            "{}: vector payload length {} is not a multiple of 4",
            sig,
            cell.len()
        )));
    }
    Ok(cell
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect())
}

fn json_f32_array(sig: &str, bytes: &[u8]) -> Result<Vec<f32>> {
    let parsed: serde_json::Value = serde_json::from_slice(bytes)
        .map_err(|_| ZyronError::ExecutionError(format!("{} expects a JSON number array", sig)))?;
    let arr = parsed.as_array().ok_or_else(|| {
        ZyronError::ExecutionError(format!("{} expects a JSON number array", sig))
    })?;
    arr.iter()
        .map(|v| {
            v.as_f64().map(|f| f as f32).ok_or_else(|| {
                ZyronError::ExecutionError(format!("{} expects numeric array elements", sig))
            })
        })
        .collect()
}

/// Reads one vector per row. VECTOR tagged cells hold packed f32, other
/// binary and text cells hold JSON number arrays. Null rows yield an empty
/// placeholder the caller masks out
fn vector_rows(sig: &str, col: &Column) -> Result<Vec<Vec<f32>>> {
    match &col.data {
        ColumnData::Binary(cells) => {
            let mut out = Vec::with_capacity(cells.len());
            for (i, cell) in cells.iter().enumerate() {
                if col.nulls.is_null(i) {
                    out.push(Vec::new());
                } else if col.type_id == TypeId::Vector {
                    out.push(decode_packed_vector(sig, cell)?);
                } else {
                    out.push(json_f32_array(sig, cell)?);
                }
            }
            Ok(out)
        }
        ColumnData::Utf8(texts) => {
            let mut out = Vec::with_capacity(texts.len());
            for (i, text) in texts.iter().enumerate() {
                if col.nulls.is_null(i) {
                    out.push(Vec::new());
                } else {
                    out.push(json_f32_array(sig, text.as_bytes())?);
                }
            }
            Ok(out)
        }
        _ => Err(ZyronError::ExecutionError(format!(
            "{}: expected a vector argument",
            sig
        ))),
    }
}

// ---------------------------------------------------------------------------
// phonetic
// ---------------------------------------------------------------------------

// row loop shared by both phonetic arms, per row values feed the sink
fn phonetic_rows<T: Default>(
    sig: &str,
    args: &[Column],
    f: impl Fn(&str, &str, &str) -> Result<T>,
) -> Result<(Vec<T>, NullBitmap)> {
    if args.len() != 3 {
        return Err(arity_err(sig, "exactly 3 arguments"));
    }
    let a = strings_arg(sig, &args[0])?;
    let b = strings_arg(sig, &args[1])?;
    let algs = strings_arg(sig, &args[2])?;
    let n = out_len(sig, &[a.len(), b.len(), algs.len()])?;
    let mut data = Vec::with_capacity(n);
    let mut nulls = NullBitmap::none(n);
    for i in 0..n {
        let ai = bidx(a.len(), i);
        let bi = bidx(b.len(), i);
        let gi = bidx(algs.len(), i);
        if args[0].nulls.is_null(ai) || args[1].nulls.is_null(bi) || args[2].nulls.is_null(gi) {
            data.push(T::default());
            nulls.set_null(i);
            continue;
        }
        data.push(f(a[ai], b[bi], algs[gi])?);
    }
    Ok((data, nulls))
}

fn phonetic_match_impl(args: &[Column]) -> Result<Column> {
    let (data, nulls) = phonetic_rows(
        "phonetic_match(text, text, algorithm)",
        args,
        zyron_types::fuzzy::phonetic_match,
    )?;
    Ok(Column::with_nulls(
        ColumnData::Boolean(data),
        nulls,
        TypeId::Boolean,
    ))
}

fn phonetic_score_impl(args: &[Column]) -> Result<Column> {
    let (data, nulls) = phonetic_rows(
        "phonetic_score(text, text, algorithm)",
        args,
        zyron_types::fuzzy::phonetic_score,
    )?;
    Ok(Column::with_nulls(
        ColumnData::Float32(data),
        nulls,
        TypeId::Float32,
    ))
}

// ---------------------------------------------------------------------------
// vector math
// ---------------------------------------------------------------------------

fn vector_f64_2(
    sig: &str,
    args: &[Column],
    f: impl Fn(&[f32], &[f32]) -> Result<f64>,
) -> Result<Column> {
    if args.len() != 2 {
        return Err(arity_err(sig, "exactly 2 arguments"));
    }
    let a = vector_rows(sig, &args[0])?;
    let b = vector_rows(sig, &args[1])?;
    let n = out_len(sig, &[a.len(), b.len()])?;
    let mut data = Vec::with_capacity(n);
    let mut nulls = NullBitmap::none(n);
    for i in 0..n {
        let ai = bidx(a.len(), i);
        let bi = bidx(b.len(), i);
        if args[0].nulls.is_null(ai) || args[1].nulls.is_null(bi) {
            data.push(0.0);
            nulls.set_null(i);
            continue;
        }
        data.push(f(&a[ai], &b[bi])?);
    }
    Ok(Column::with_nulls(
        ColumnData::Float64(data),
        nulls,
        TypeId::Float64,
    ))
}

fn vector_vec_2(
    sig: &str,
    args: &[Column],
    f: impl Fn(&[f32], &[f32]) -> Result<Vec<f32>>,
) -> Result<Column> {
    if args.len() != 2 {
        return Err(arity_err(sig, "exactly 2 arguments"));
    }
    let a = vector_rows(sig, &args[0])?;
    let b = vector_rows(sig, &args[1])?;
    let n = out_len(sig, &[a.len(), b.len()])?;
    let mut data = Vec::with_capacity(n);
    let mut nulls = NullBitmap::none(n);
    for i in 0..n {
        let ai = bidx(a.len(), i);
        let bi = bidx(b.len(), i);
        if args[0].nulls.is_null(ai) || args[1].nulls.is_null(bi) {
            data.push(Vec::new());
            nulls.set_null(i);
            continue;
        }
        data.push(encode_vector(&f(&a[ai], &b[bi])?));
    }
    Ok(Column::with_nulls(
        ColumnData::Binary(data),
        nulls,
        TypeId::Vector,
    ))
}

fn vector_norm_impl(args: &[Column]) -> Result<Column> {
    let sig = "vector_norm(vector)";
    if args.len() != 1 {
        return Err(arity_err(sig, "exactly 1 argument"));
    }
    let rows = vector_rows(sig, &args[0])?;
    let n = rows.len();
    let mut data = Vec::with_capacity(n);
    let mut nulls = NullBitmap::none(n);
    for (i, row) in rows.iter().enumerate() {
        if args[0].nulls.is_null(i) {
            data.push(0.0);
            nulls.set_null(i);
            continue;
        }
        data.push(zyron_types::vector_math::vector_norm(row));
    }
    Ok(Column::with_nulls(
        ColumnData::Float64(data),
        nulls,
        TypeId::Float64,
    ))
}

fn vector_normalize_impl(args: &[Column]) -> Result<Column> {
    let sig = "vector_normalize(vector)";
    if args.len() != 1 {
        return Err(arity_err(sig, "exactly 1 argument"));
    }
    let rows = vector_rows(sig, &args[0])?;
    let n = rows.len();
    let mut data = Vec::with_capacity(n);
    let mut nulls = NullBitmap::none(n);
    for (i, row) in rows.iter().enumerate() {
        if args[0].nulls.is_null(i) {
            data.push(Vec::new());
            nulls.set_null(i);
            continue;
        }
        data.push(encode_vector(&zyron_types::vector_math::vector_normalize(
            row,
        )?));
    }
    Ok(Column::with_nulls(
        ColumnData::Binary(data),
        nulls,
        TypeId::Vector,
    ))
}

// ---------------------------------------------------------------------------
// money
// ---------------------------------------------------------------------------

// decimal_places is optional and defaults to 2
fn money_round_impl(args: &[Column]) -> Result<Column> {
    let sig = "money_round(money [, decimal_places])";
    if args.is_empty() || args.len() > 2 {
        return Err(arity_err(sig, "1 or 2 arguments"));
    }
    let cells = binary_cells("money_round", &args[0])?;
    let places = if args.len() == 2 {
        Some(ints_arg("money_round", &args[1])?)
    } else {
        None
    };
    let n = out_len(
        "money_round",
        &[cells.len(), places.as_ref().map_or(1, |p| p.len())],
    )?;
    let mut data = Vec::with_capacity(n);
    let mut nulls = NullBitmap::none(n);
    for i in 0..n {
        let ci = bidx(cells.len(), i);
        let place_null = places
            .as_ref()
            .is_some_and(|p| args[1].nulls.is_null(bidx(p.len(), i)));
        if args[0].nulls.is_null(ci) || place_null {
            data.push(Vec::new());
            nulls.set_null(i);
            continue;
        }
        let raw_places = places.as_ref().map_or(2, |p| p[bidx(p.len(), i)]);
        let dp = i32::try_from(raw_places).map_err(|_| ZyronError::InvalidParameter {
            name: "decimal_places".to_string(),
            value: raw_places.to_string(),
        })?;
        let (val, cur) = decode_money("money_round", &cells[ci])?;
        let (rv, rc) = zyron_types::money::money_round(val, cur, dp)?;
        data.push(encode_money(rv, rc));
    }
    Ok(Column::with_nulls(
        ColumnData::Binary(data),
        nulls,
        TypeId::Money,
    ))
}

// locale is optional and defaults to en_US
fn parse_money_impl(args: &[Column]) -> Result<Column> {
    let sig = "parse_money(text [, locale])";
    if args.is_empty() || args.len() > 2 {
        return Err(arity_err(sig, "1 or 2 arguments"));
    }
    let texts = strings_arg("parse_money", &args[0])?;
    let locales = if args.len() == 2 {
        Some(strings_arg("parse_money", &args[1])?)
    } else {
        None
    };
    let n = out_len(
        "parse_money",
        &[texts.len(), locales.as_ref().map_or(1, |l| l.len())],
    )?;
    let mut data = Vec::with_capacity(n);
    let mut nulls = NullBitmap::none(n);
    for i in 0..n {
        let ti = bidx(texts.len(), i);
        let locale_null = locales
            .as_ref()
            .is_some_and(|l| args[1].nulls.is_null(bidx(l.len(), i)));
        if args[0].nulls.is_null(ti) || locale_null {
            data.push(Vec::new());
            nulls.set_null(i);
            continue;
        }
        let locale = locales.as_ref().map_or("en_US", |l| l[bidx(l.len(), i)]);
        let (val, cur) = zyron_types::money::parse_money(texts[ti], locale)?;
        data.push(encode_money(val, cur));
    }
    Ok(Column::with_nulls(
        ColumnData::Binary(data),
        nulls,
        TypeId::Money,
    ))
}

/// Days since 1970-01-01 for a civil date, valid across the i32 day range
fn days_from_ymd(y: i32, m: u32, d: u32) -> i32 {
    let y = if m <= 2 { y - 1 } else { y };
    let era = if y >= 0 { y } else { y - 399 } / 400;
    let yoe = (y - era * 400) as i64;
    let mp = ((m + 9) % 12) as i64;
    let doy = (153 * mp + 2) / 5 + (d as i64 - 1);
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    (era as i64 * 146097 + doe - 719468) as i32
}

fn parse_iso_date(sig: &str, text: &str) -> Result<i32> {
    let bad = || {
        ZyronError::ExecutionError(format!(
            "{}: expected a date in YYYY-MM-DD form, got '{}'",
            sig, text
        ))
    };
    let parts: Vec<&str> = text.trim().split('-').collect();
    if parts.len() != 3 {
        return Err(bad());
    }
    let y: i32 = parts[0].parse().map_err(|_| bad())?;
    let m: u32 = parts[1].parse().map_err(|_| bad())?;
    let d: u32 = parts[2].parse().map_err(|_| bad())?;
    if !(1..=12).contains(&m) || !(1..=31).contains(&d) {
        return Err(bad());
    }
    Ok(days_from_ymd(y, m, d))
}

// reads one rate date per row, accepts integer days since epoch or a text
// date, null rows yield a 0 placeholder the caller masks out
fn date_days_rows(sig: &str, col: &Column) -> Result<Vec<i32>> {
    match &col.data {
        ColumnData::Int32(v) => Ok(v.clone()),
        ColumnData::Int64(v) => v
            .iter()
            .map(|&x| {
                i32::try_from(x).map_err(|_| {
                    ZyronError::ExecutionError(format!("{}: date value {} out of range", sig, x))
                })
            })
            .collect(),
        ColumnData::Utf8(texts) => {
            let mut out = Vec::with_capacity(texts.len());
            for (i, text) in texts.iter().enumerate() {
                if col.nulls.is_null(i) {
                    out.push(0);
                } else {
                    out.push(parse_iso_date(sig, text)?);
                }
            }
            Ok(out)
        }
        _ => Err(ZyronError::ExecutionError(format!(
            "{}: expected a date argument",
            sig
        ))),
    }
}

// the 3 argument form uses the latest stored rate, the 4 argument form uses
// the rate at the given date falling back to the latest on record
fn convert_currency_impl(args: &[Column]) -> Result<Column> {
    let sig = "convert_currency(money, from_currency, to_currency [, rate_date])";
    if args.len() < 3 || args.len() > 4 {
        return Err(arity_err(sig, "3 or 4 arguments"));
    }
    let cells = binary_cells("convert_currency", &args[0])?;
    let from = strings_arg("convert_currency", &args[1])?;
    let to = strings_arg("convert_currency", &args[2])?;
    let dates = if args.len() == 4 {
        Some(date_days_rows("convert_currency", &args[3])?)
    } else {
        None
    };
    let n = out_len(
        "convert_currency",
        &[
            cells.len(),
            from.len(),
            to.len(),
            dates.as_ref().map_or(1, |d| d.len()),
        ],
    )?;
    let mut data = Vec::with_capacity(n);
    let mut nulls = NullBitmap::none(n);
    for i in 0..n {
        let ci = bidx(cells.len(), i);
        let fi = bidx(from.len(), i);
        let ti = bidx(to.len(), i);
        let date_null = dates
            .as_ref()
            .is_some_and(|d| args[3].nulls.is_null(bidx(d.len(), i)));
        if args[0].nulls.is_null(ci)
            || args[1].nulls.is_null(fi)
            || args[2].nulls.is_null(ti)
            || date_null
        {
            data.push(Vec::new());
            nulls.set_null(i);
            continue;
        }
        let date = dates.as_ref().map(|d| d[bidx(d.len(), i)]);
        let (val, cur) = decode_money("convert_currency", &cells[ci])?;
        let (rv, rc) = zyron_types::money::convert_currency(val, cur, from[fi], to[ti], date)?;
        data.push(encode_money(rv, rc));
    }
    Ok(Column::with_nulls(
        ColumnData::Binary(data),
        nulls,
        TypeId::Money,
    ))
}

// ---------------------------------------------------------------------------
// semver
// ---------------------------------------------------------------------------

// a release version yields NULL because it has no prerelease tag
fn semver_prerelease_impl(args: &[Column]) -> Result<Column> {
    let sig = "semver_prerelease(version)";
    if args.len() != 1 {
        return Err(arity_err(sig, "exactly 1 argument"));
    }
    let texts = strings_arg("semver_prerelease", &args[0])?;
    let n = texts.len();
    let mut data = Vec::with_capacity(n);
    let mut nulls = NullBitmap::none(n);
    for (i, text) in texts.iter().enumerate() {
        if args[0].nulls.is_null(i) {
            data.push(String::new());
            nulls.set_null(i);
            continue;
        }
        match zyron_types::semver::semver_prerelease(text)? {
            Some(tag) => data.push(tag),
            None => {
                data.push(String::new());
                nulls.set_null(i);
            }
        }
    }
    Ok(Column::with_nulls(
        ColumnData::Utf8(data),
        nulls,
        TypeId::Text,
    ))
}

fn json_string_array(sig: &str, bytes: &[u8]) -> Result<Vec<String>> {
    let parsed: serde_json::Value = serde_json::from_slice(bytes)
        .map_err(|_| ZyronError::ExecutionError(format!("{} expects a JSON string array", sig)))?;
    let arr = parsed.as_array().ok_or_else(|| {
        ZyronError::ExecutionError(format!("{} expects a JSON string array", sig))
    })?;
    arr.iter()
        .map(|v| {
            v.as_str().map(|s| s.to_string()).ok_or_else(|| {
                ZyronError::ExecutionError(format!("{} expects string array elements", sig))
            })
        })
        .collect()
}

fn semver_sort_impl(args: &[Column]) -> Result<Column> {
    let sig = "semver_sort(versions)";
    if args.len() != 1 {
        return Err(arity_err(sig, "exactly 1 argument"));
    }
    let cells = super::column_bytes(&args[0])
        .map_err(|_| ZyronError::ExecutionError(format!("{}: expected an array argument", sig)))?;
    let n = cells.len();
    let mut data = Vec::with_capacity(n);
    let mut nulls = NullBitmap::none(n);
    for (i, cell) in cells.iter().enumerate() {
        if args[0].nulls.is_null(i) {
            data.push(Vec::new());
            nulls.set_null(i);
            continue;
        }
        let versions = json_string_array(sig, cell)?;
        let borrowed: Vec<&str> = versions.iter().map(|s| s.as_str()).collect();
        let sorted = zyron_types::semver::semver_sort(&borrowed)?;
        let rendered =
            serde_json::Value::Array(sorted.into_iter().map(serde_json::Value::String).collect());
        data.push(rendered.to_string().into_bytes());
    }
    Ok(Column::with_nulls(
        ColumnData::Binary(data),
        nulls,
        TypeId::Array,
    ))
}

// ---------------------------------------------------------------------------
// probabilistic
// ---------------------------------------------------------------------------

fn bloom_estimate_impl(args: &[Column]) -> Result<Column> {
    let sig = "bloom_filter_estimate_count(bloom)";
    if args.len() != 1 {
        return Err(arity_err(sig, "exactly 1 argument"));
    }
    let cells = binary_cells("bloom_filter_estimate_count", &args[0])?;
    let n = cells.len();
    let mut data = Vec::with_capacity(n);
    let mut nulls = NullBitmap::none(n);
    for (i, cell) in cells.iter().enumerate() {
        if args[0].nulls.is_null(i) {
            data.push(0);
            nulls.set_null(i);
            continue;
        }
        data.push(zyron_types::probabilistic::bloom_filter_estimate_count(
            cell,
        )?);
    }
    Ok(Column::with_nulls(
        ColumnData::Int64(data),
        nulls,
        TypeId::Int64,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn utf8_col(values: &[&str]) -> Column {
        Column::new(
            ColumnData::Utf8(values.iter().map(|s| s.to_string()).collect()),
            TypeId::Text,
        )
    }

    fn i64_col(values: &[i64]) -> Column {
        Column::new(ColumnData::Int64(values.to_vec()), TypeId::Int64)
    }

    fn vector_col(rows: &[&[f32]]) -> Column {
        Column::new(
            ColumnData::Binary(rows.iter().map(|r| encode_vector(r)).collect()),
            TypeId::Vector,
        )
    }

    fn money_col(cells: &[(i64, u16)]) -> Column {
        Column::new(
            ColumnData::Binary(cells.iter().map(|&(v, c)| encode_money(v, c)).collect()),
            TypeId::Money,
        )
    }

    #[test]
    fn phonetic_match_soundex_and_score() {
        let out = dispatch(
            "phonetic_match",
            &[
                utf8_col(&["Robert"]),
                utf8_col(&["Rupert"]),
                utf8_col(&["soundex"]),
            ],
            1,
        )
        .unwrap()
        .unwrap();
        match &out.data {
            ColumnData::Boolean(v) => assert!(v[0]),
            other => panic!("expected Boolean, got {:?}", other),
        }
        let score = dispatch(
            "phonetic_score",
            &[
                utf8_col(&["Robert"]),
                utf8_col(&["Rupert"]),
                utf8_col(&["soundex"]),
            ],
            1,
        )
        .unwrap()
        .unwrap();
        assert_eq!(score.type_id, TypeId::Float32);
        match &score.data {
            ColumnData::Float32(v) => assert_eq!(v[0], 1.0),
            other => panic!("expected Float32, got {:?}", other),
        }
    }

    #[test]
    fn phonetic_unknown_algorithm_errors() {
        let out = dispatch(
            "phonetic_match",
            &[utf8_col(&["a"]), utf8_col(&["b"]), utf8_col(&["bogus"])],
            1,
        )
        .unwrap();
        assert!(out.is_err());
    }

    #[test]
    fn vector_dot_on_packed_cells() {
        let out = dispatch(
            "vector_dot",
            &[
                vector_col(&[&[1.0, 2.0, 3.0]]),
                vector_col(&[&[4.0, 5.0, 6.0]]),
            ],
            1,
        )
        .unwrap()
        .unwrap();
        match &out.data {
            ColumnData::Float64(v) => assert_eq!(v[0], 32.0),
            other => panic!("expected Float64, got {:?}", other),
        }
    }

    #[test]
    fn vector_cross_accepts_json_arrays() {
        let a = Column::new(ColumnData::Binary(vec![b"[1,0,0]".to_vec()]), TypeId::Array);
        let b = Column::new(ColumnData::Binary(vec![b"[0,1,0]".to_vec()]), TypeId::Array);
        let out = dispatch("vector_cross", &[a, b], 1).unwrap().unwrap();
        assert_eq!(out.type_id, TypeId::Vector);
        match &out.data {
            ColumnData::Binary(v) => {
                assert_eq!(v[0], encode_vector(&[0.0, 0.0, 1.0]));
            }
            other => panic!("expected Binary, got {:?}", other),
        }
    }

    #[test]
    fn vector_normalize_zero_vector_errors() {
        let out = dispatch("vector_normalize", &[vector_col(&[&[0.0, 0.0]])], 1).unwrap();
        assert!(out.is_err());
    }

    #[test]
    fn vector_norm_and_angle() {
        let norm = dispatch("vector_norm", &[vector_col(&[&[3.0, 4.0]])], 1)
            .unwrap()
            .unwrap();
        match &norm.data {
            ColumnData::Float64(v) => assert_eq!(v[0], 5.0),
            other => panic!("expected Float64, got {:?}", other),
        }
        let angle = dispatch(
            "vector_angle",
            &[vector_col(&[&[1.0, 0.0]]), vector_col(&[&[0.0, 1.0]])],
            1,
        )
        .unwrap()
        .unwrap();
        match &angle.data {
            ColumnData::Float64(v) => {
                assert!((v[0] - std::f64::consts::FRAC_PI_2).abs() < 1e-9)
            }
            other => panic!("expected Float64, got {:?}", other),
        }
    }

    #[test]
    fn money_round_defaults_to_two_places() {
        // BHD carries three minor digits so the default rounds the last one
        let out = dispatch("money_round", &[money_col(&[(12345, 48)])], 1)
            .unwrap()
            .unwrap();
        match &out.data {
            ColumnData::Binary(v) => {
                assert_eq!(decode_money("test", &v[0]).unwrap(), (12350, 48));
            }
            other => panic!("expected Binary, got {:?}", other),
        }
    }

    #[test]
    fn money_round_explicit_places() {
        let out = dispatch(
            "money_round",
            &[money_col(&[(1950, 840)]), i64_col(&[0])],
            1,
        )
        .unwrap()
        .unwrap();
        match &out.data {
            ColumnData::Binary(v) => {
                assert_eq!(decode_money("test", &v[0]).unwrap(), (2000, 840));
            }
            other => panic!("expected Binary, got {:?}", other),
        }
    }

    #[test]
    fn parse_money_defaults_to_en_us() {
        let out = dispatch("parse_money", &[utf8_col(&["$1,234.56"])], 1)
            .unwrap()
            .unwrap();
        assert_eq!(out.type_id, TypeId::Money);
        match &out.data {
            ColumnData::Binary(v) => {
                assert_eq!(decode_money("test", &v[0]).unwrap(), (123456, 840));
            }
            other => panic!("expected Binary, got {:?}", other),
        }
    }

    #[test]
    fn parse_money_locale_argument() {
        let out = dispatch(
            "parse_money",
            &[utf8_col(&["1.234,56"]), utf8_col(&["de_DE"])],
            1,
        )
        .unwrap()
        .unwrap();
        match &out.data {
            ColumnData::Binary(v) => {
                assert_eq!(decode_money("test", &v[0]).unwrap(), (123456, 978));
            }
            other => panic!("expected Binary, got {:?}", other),
        }
    }

    #[test]
    fn convert_currency_store_flow() {
        // one test covers identity, direct, and dated lookups so the
        // process global store is only populated from a single place
        zyron_types::money::currency_rate_store().replace_all(vec![
            zyron_types::money::CurrencyRate {
                from: "USD".to_string(),
                to: "EUR".to_string(),
                rate_date_days: 20000,
                rate: 0.8,
            },
            zyron_types::money::CurrencyRate {
                from: "USD".to_string(),
                to: "EUR".to_string(),
                rate_date_days: 20100,
                rate: 0.9,
            },
        ]);
        let identity = dispatch(
            "convert_currency",
            &[
                money_col(&[(1000, 840)]),
                utf8_col(&["USD"]),
                utf8_col(&["USD"]),
            ],
            1,
        )
        .unwrap()
        .unwrap();
        match &identity.data {
            ColumnData::Binary(v) => {
                assert_eq!(decode_money("test", &v[0]).unwrap(), (1000, 840));
            }
            other => panic!("expected Binary, got {:?}", other),
        }
        let latest = dispatch(
            "convert_currency",
            &[
                money_col(&[(10000, 840)]),
                utf8_col(&["USD"]),
                utf8_col(&["EUR"]),
            ],
            1,
        )
        .unwrap()
        .unwrap();
        match &latest.data {
            ColumnData::Binary(v) => {
                assert_eq!(decode_money("test", &v[0]).unwrap(), (9000, 978));
            }
            other => panic!("expected Binary, got {:?}", other),
        }
        // 2024-10-10 is day 20006, inside the first rate's window
        let dated = dispatch(
            "convert_currency",
            &[
                money_col(&[(10000, 840)]),
                utf8_col(&["USD"]),
                utf8_col(&["EUR"]),
                utf8_col(&["2024-10-10"]),
            ],
            1,
        )
        .unwrap()
        .unwrap();
        match &dated.data {
            ColumnData::Binary(v) => {
                assert_eq!(decode_money("test", &v[0]).unwrap(), (8000, 978));
            }
            other => panic!("expected Binary, got {:?}", other),
        }
    }

    #[test]
    fn semver_prerelease_null_for_release() {
        let out = dispatch(
            "semver_prerelease",
            &[utf8_col(&["1.0.0-rc.1", "1.0.0"])],
            2,
        )
        .unwrap()
        .unwrap();
        assert!(!out.nulls.is_null(0));
        assert!(out.nulls.is_null(1));
        match &out.data {
            ColumnData::Utf8(v) => assert_eq!(v[0], "rc.1"),
            other => panic!("expected Utf8, got {:?}", other),
        }
    }

    #[test]
    fn semver_sort_orders_prereleases() {
        let input = Column::new(
            ColumnData::Binary(vec![
                br#"["1.0.0","1.0.0-alpha","1.0.0-alpha.1","1.0.0-beta"]"#.to_vec(),
            ]),
            TypeId::Array,
        );
        let out = dispatch("semver_sort", &[input], 1).unwrap().unwrap();
        assert_eq!(out.type_id, TypeId::Array);
        match &out.data {
            ColumnData::Binary(v) => {
                let json = String::from_utf8(v[0].clone()).unwrap();
                assert_eq!(
                    json,
                    r#"["1.0.0-alpha","1.0.0-alpha.1","1.0.0-beta","1.0.0"]"#
                );
            }
            other => panic!("expected Binary, got {:?}", other),
        }
    }

    #[test]
    fn semver_sort_invalid_version_errors() {
        let input = Column::new(
            ColumnData::Binary(vec![br#"["1.0.0","nope"]"#.to_vec()]),
            TypeId::Array,
        );
        assert!(dispatch("semver_sort", &[input], 1).unwrap().is_err());
    }

    #[test]
    fn bloom_estimate_counts_inserts() {
        let mut filter = zyron_types::probabilistic::bloom_create(500, 0.01).unwrap();
        for i in 0..100 {
            zyron_types::probabilistic::bloom_add(&mut filter, format!("k{}", i).as_bytes())
                .unwrap();
        }
        let col = Column::new(ColumnData::Binary(vec![filter]), TypeId::BloomFilter);
        let out = dispatch("bloom_filter_estimate_count", &[col], 1)
            .unwrap()
            .unwrap();
        match &out.data {
            ColumnData::Int64(v) => {
                assert!(
                    (v[0] - 100).abs() <= 10,
                    "estimate {} too far from 100",
                    v[0]
                );
            }
            other => panic!("expected Int64, got {:?}", other),
        }
    }

    #[test]
    fn arity_errors_do_not_fall_through() {
        for name in [
            "phonetic_match",
            "phonetic_score",
            "vector_dot",
            "vector_cross",
            "vector_norm",
            "vector_normalize",
            "vector_angle",
            "money_round",
            "parse_money",
            "convert_currency",
            "semver_prerelease",
            "semver_sort",
            "bloom_filter_estimate_count",
        ] {
            let out = dispatch(name, &[], 0);
            assert!(out.is_some(), "{} not dispatched", name);
            assert!(out.unwrap().is_err(), "{} accepted zero arguments", name);
        }
    }
}
