//! Dispatch arms for color, fingerprint, fuzzy distance, set and vector
//! similarity, and entity resolution functions
//!
//! Array typed cells hold JSON text bytes inside Binary columns, minhash
//! signatures travel as little endian u64 bytes in Bytea cells

use crate::column::{Column, ColumnData, NullBitmap};
use zyron_common::{Result, TypeId, ZyronError};

pub(super) fn dispatch(name: &str, args: &[Column], num_rows: usize) -> Option<Result<Column>> {
    // num_rows only matters for zero arg generators, none in this family
    let _ = num_rows;
    Some(match name {
        // color, color_rgb and color_hex live in mod.rs
        "color_from_rgba" | "color_rgba" => color_from_rgba_cols(args),
        "color_from_hsl" | "color_hsl" => color_from_hsl_cols(args),
        "color_blend" => color_blend_cols(args),
        "color_lighten" => color_adjust_cols(
            args,
            "color_lighten(color, double)",
            zyron_types::color::color_lighten,
        ),
        "color_darken" => color_adjust_cols(
            args,
            "color_darken(color, double)",
            zyron_types::color::color_darken,
        ),
        "color_to_hsl" => color_to_hsl_cols(args),
        "color_palette" => color_palette_cols(args),
        "wcag_contrast_ratio" => wcag_contrast_ratio_cols(args),
        "wcag_compliant" => wcag_compliant_cols(args),

        // fingerprint
        "minhash_signature" => minhash_signature_cols(args),
        "minhash_encode" => minhash_encode_cols(args),
        "minhash_decode" => minhash_decode_cols(args),
        "minhash_similarity" => minhash_similarity_cols(args),
        "simhash" => simhash_cols(args),
        "simhash_distance" => simhash_distance_cols(args),
        "simhash_similar" => simhash_similar_cols(args),
        "shingle" => shingle_cols(
            args,
            "shingle(text, int)",
            zyron_types::fingerprint::shingle,
        ),
        "word_shingle" => shingle_cols(
            args,
            "word_shingle(text, int)",
            zyron_types::fingerprint::word_shingle,
        ),

        // fuzzy
        "double_metaphone" => double_metaphone_cols(args),
        "hamming" => hamming_cols(args),

        // similarity
        "cosine_similarity" => cosine_similarity_cols(args),
        "jaccard_similarity" => token_similarity_cols(
            args,
            "jaccard_similarity(array, array)",
            zyron_types::similarity::jaccard_similarity,
        ),
        "sorensen_dice" => token_similarity_cols(
            args,
            "sorensen_dice(array, array)",
            zyron_types::similarity::sorensen_dice,
        ),
        "overlap_coefficient" => token_similarity_cols(
            args,
            "overlap_coefficient(array, array)",
            zyron_types::similarity::overlap_coefficient,
        ),
        "ngram_similarity" => ngram_similarity_cols(args),
        "qgram_distance" => qgram_distance_cols(args),

        // entity resolution
        "address_similarity" => two_string_similarity_cols(
            args,
            "address_similarity(text, text)",
            zyron_types::entity_resolution::address_similarity,
        ),
        "company_similarity" => two_string_similarity_cols(
            args,
            "company_similarity(text, text)",
            zyron_types::entity_resolution::company_similarity,
        ),
        "name_similarity" => two_string_similarity_cols(
            args,
            "name_similarity(text, text)",
            zyron_types::entity_resolution::name_similarity,
        ),

        _ => return None,
    })
}

// ---------------------------------------------------------------------------
// shared readers and writers
// ---------------------------------------------------------------------------

// batch row count, min across args guards literal length skew
fn row_count(args: &[Column]) -> usize {
    args.iter().map(|c| c.data.len()).min().unwrap_or(0)
}

// union of input null bits, a NULL in any input row makes the output row NULL
fn combined_nulls(args: &[Column], n: usize) -> NullBitmap {
    let mut nulls = NullBitmap::none(n);
    for col in args {
        for i in 0..n.min(col.nulls.len()) {
            if col.nulls.is_null(i) {
                nulls.set_null(i);
            }
        }
    }
    nulls
}

fn expect_args(sig: &str, args: &[Column], expected: usize) -> Result<()> {
    if args.len() != expected {
        return Err(ZyronError::ExecutionError(format!(
            "{} takes exactly {} argument{}, got {}",
            sig,
            expected,
            if expected == 1 { "" } else { "s" },
            args.len()
        )));
    }
    Ok(())
}

// packed rgba colors arrive as UInt32 from the color constructors or as any
// int width from literals
fn color_values(col: &Column) -> Result<Vec<u32>> {
    Ok(super::column_ints(col)?.iter().map(|&v| v as u32).collect())
}

// clamps an int channel value into the byte range
fn clamp_component(v: i64) -> u8 {
    v.clamp(0, 255) as u8
}

// per row text regardless of physical Utf8 or Binary storage, Array cells
// arrive as JSON text bytes inside Binary
fn column_texts(col: &Column) -> Result<Vec<String>> {
    match &col.data {
        ColumnData::Utf8(v) => Ok(v.clone()),
        ColumnData::Binary(v) => Ok(v
            .iter()
            .map(|b| String::from_utf8_lossy(b).into_owned())
            .collect()),
        _ => Err(ZyronError::ExecutionError(
            "expected string or binary column".into(),
        )),
    }
}

// token list per row, JSON array of strings when the cell parses as one,
// whitespace split otherwise
fn token_list(cell: &str) -> Vec<String> {
    if let Ok(serde_json::Value::Array(items)) = serde_json::from_str(cell.trim()) {
        return items
            .iter()
            .map(|v| match v {
                serde_json::Value::String(s) => s.clone(),
                other => other.to_string(),
            })
            .collect();
    }
    cell.split_whitespace().map(str::to_string).collect()
}

// minhash signature per row, JSON array of u64 for Array cells, little
// endian u64 bytes for Bytea cells
fn parse_signature(raw: &[u8]) -> Option<Vec<u64>> {
    let first = raw.iter().find(|b| !b.is_ascii_whitespace());
    if first == Some(&b'[') {
        serde_json::from_slice::<Vec<u64>>(raw).ok()
    } else {
        zyron_types::fingerprint::minhash_decode(raw).ok()
    }
}

// numeric vector per row as a JSON array of numbers
fn parse_vector(raw: &[u8]) -> Option<Vec<f64>> {
    serde_json::from_slice::<Vec<f64>>(raw).ok()
}

fn json_string_array(items: &[String]) -> Vec<u8> {
    serde_json::Value::Array(
        items
            .iter()
            .map(|s| serde_json::Value::String(s.clone()))
            .collect(),
    )
    .to_string()
    .into_bytes()
}

fn json_u64_array(items: &[u64]) -> Vec<u8> {
    serde_json::Value::Array(
        items
            .iter()
            .map(|&v| serde_json::Value::Number(serde_json::Number::from(v)))
            .collect(),
    )
    .to_string()
    .into_bytes()
}

// non finite floats serialize as JSON null
fn json_f64_array(items: &[f64]) -> Vec<u8> {
    serde_json::Value::Array(
        items
            .iter()
            .map(|&v| {
                serde_json::Number::from_f64(v)
                    .map(serde_json::Value::Number)
                    .unwrap_or(serde_json::Value::Null)
            })
            .collect(),
    )
    .to_string()
    .into_bytes()
}

// ---------------------------------------------------------------------------
// color
// ---------------------------------------------------------------------------

fn color_from_rgba_cols(args: &[Column]) -> Result<Column> {
    expect_args("color_from_rgba(int, int, int, int)", args, 4)?;
    let r = super::column_ints(&args[0])?;
    let g = super::column_ints(&args[1])?;
    let b = super::column_ints(&args[2])?;
    let a = super::column_ints(&args[3])?;
    let n = row_count(args);
    let data: Vec<u32> = (0..n)
        .map(|i| {
            zyron_types::color::color_from_rgba(
                clamp_component(r[i]),
                clamp_component(g[i]),
                clamp_component(b[i]),
                clamp_component(a[i]),
            )
        })
        .collect();
    Ok(Column::with_nulls(
        ColumnData::UInt32(data),
        combined_nulls(args, n),
        TypeId::Color,
    ))
}

fn color_from_hsl_cols(args: &[Column]) -> Result<Column> {
    expect_args("color_from_hsl(double, double, double)", args, 3)?;
    let h = super::column_floats(&args[0])?;
    let s = super::column_floats(&args[1])?;
    let l = super::column_floats(&args[2])?;
    let n = row_count(args);
    let data: Vec<u32> = (0..n)
        .map(|i| zyron_types::color::color_from_hsl(h[i], s[i], l[i]))
        .collect();
    Ok(Column::with_nulls(
        ColumnData::UInt32(data),
        combined_nulls(args, n),
        TypeId::Color,
    ))
}

fn color_blend_cols(args: &[Column]) -> Result<Column> {
    expect_args("color_blend(color, color, double)", args, 3)?;
    let a = color_values(&args[0])?;
    let b = color_values(&args[1])?;
    let ratio = super::column_floats(&args[2])?;
    let n = row_count(args);
    let data: Vec<u32> = (0..n)
        .map(|i| zyron_types::color::color_blend(a[i], b[i], ratio[i]))
        .collect();
    Ok(Column::with_nulls(
        ColumnData::UInt32(data),
        combined_nulls(args, n),
        TypeId::Color,
    ))
}

fn color_adjust_cols(args: &[Column], sig: &str, f: fn(u32, f64) -> u32) -> Result<Column> {
    expect_args(sig, args, 2)?;
    let colors = color_values(&args[0])?;
    let amounts = super::column_floats(&args[1])?;
    let n = row_count(args);
    let data: Vec<u32> = (0..n).map(|i| f(colors[i], amounts[i])).collect();
    Ok(Column::with_nulls(
        ColumnData::UInt32(data),
        combined_nulls(args, n),
        TypeId::Color,
    ))
}

fn color_to_hsl_cols(args: &[Column]) -> Result<Column> {
    expect_args("color_to_hsl(color)", args, 1)?;
    let colors = color_values(&args[0])?;
    let n = row_count(args);
    let data: Vec<Vec<u8>> = (0..n)
        .map(|i| {
            let (h, s, l) = zyron_types::color::color_to_hsl(colors[i]);
            json_f64_array(&[h, s, l])
        })
        .collect();
    Ok(Column::with_nulls(
        ColumnData::Binary(data),
        combined_nulls(args, n),
        TypeId::Array,
    ))
}

fn color_palette_cols(args: &[Column]) -> Result<Column> {
    expect_args("color_palette(color, text)", args, 2)?;
    let base = color_values(&args[0])?;
    let schemes = super::column_strings(&args[1])?;
    let n = row_count(args);
    let nulls = combined_nulls(args, n);
    let mut data = Vec::with_capacity(n);
    for i in 0..n {
        if nulls.is_null(i) {
            data.push(Vec::new());
            continue;
        }
        // an unknown scheme is a call error, not a row NULL
        let colors = zyron_types::color::color_palette(base[i], schemes[i])
            .map_err(|e| ZyronError::ExecutionError(format!("color_palette: {}", e)))?;
        // palette cells are JSON arrays of hex strings, round trip through color_from_hex
        let hex: Vec<String> = colors
            .iter()
            .map(|&c| zyron_types::color::color_to_hex(c))
            .collect();
        data.push(json_string_array(&hex));
    }
    Ok(Column::with_nulls(
        ColumnData::Binary(data),
        nulls,
        TypeId::Array,
    ))
}

fn wcag_contrast_ratio_cols(args: &[Column]) -> Result<Column> {
    expect_args("wcag_contrast_ratio(color, color)", args, 2)?;
    let fg = color_values(&args[0])?;
    let bg = color_values(&args[1])?;
    let n = row_count(args);
    let data: Vec<f64> = (0..n)
        .map(|i| zyron_types::color::wcag_contrast_ratio(fg[i], bg[i]))
        .collect();
    Ok(Column::with_nulls(
        ColumnData::Float64(data),
        combined_nulls(args, n),
        TypeId::Float64,
    ))
}

fn wcag_compliant_cols(args: &[Column]) -> Result<Column> {
    expect_args("wcag_compliant(color, color, text)", args, 3)?;
    let fg = color_values(&args[0])?;
    let bg = color_values(&args[1])?;
    let levels = super::column_strings(&args[2])?;
    let n = row_count(args);
    let nulls = combined_nulls(args, n);
    let mut data = Vec::with_capacity(n);
    for i in 0..n {
        if nulls.is_null(i) {
            data.push(false);
            continue;
        }
        // an unknown level is a call error, not a row NULL
        let ok = zyron_types::color::wcag_compliant(fg[i], bg[i], levels[i])
            .map_err(|e| ZyronError::ExecutionError(format!("wcag_compliant: {}", e)))?;
        data.push(ok);
    }
    Ok(Column::with_nulls(
        ColumnData::Boolean(data),
        nulls,
        TypeId::Boolean,
    ))
}

// ---------------------------------------------------------------------------
// fingerprint
// ---------------------------------------------------------------------------

fn minhash_signature_cols(args: &[Column]) -> Result<Column> {
    expect_args("minhash_signature(array, int)", args, 2)?;
    let texts = column_texts(&args[0])?;
    let hashes = super::column_ints(&args[1])?;
    let n = row_count(args);
    let nulls = combined_nulls(args, n);
    let mut data = Vec::with_capacity(n);
    for i in 0..n {
        if nulls.is_null(i) {
            data.push(Vec::new());
            continue;
        }
        if !(0..=65_536).contains(&hashes[i]) {
            return Err(ZyronError::ExecutionError(format!(
                "minhash_signature num_hashes must be in 0..=65536, got {}",
                hashes[i]
            )));
        }
        let tokens = token_list(&texts[i]);
        let refs: Vec<&str> = tokens.iter().map(String::as_str).collect();
        let sig = zyron_types::fingerprint::minhash_signature(&refs, hashes[i] as u32);
        data.push(zyron_types::fingerprint::minhash_encode(&sig));
    }
    Ok(Column::with_nulls(
        ColumnData::Binary(data),
        nulls,
        TypeId::Bytea,
    ))
}

fn minhash_encode_cols(args: &[Column]) -> Result<Column> {
    expect_args("minhash_encode(array)", args, 1)?;
    let cells = super::column_bytes(&args[0])?;
    let n = row_count(args);
    let mut nulls = combined_nulls(args, n);
    let mut data = Vec::with_capacity(n);
    for i in 0..n {
        if nulls.is_null(i) {
            data.push(Vec::new());
            continue;
        }
        // unparsable signature yields a NULL row
        match parse_signature(cells[i]) {
            Some(sig) => data.push(zyron_types::fingerprint::minhash_encode(&sig)),
            None => {
                data.push(Vec::new());
                nulls.set_null(i);
            }
        }
    }
    Ok(Column::with_nulls(
        ColumnData::Binary(data),
        nulls,
        TypeId::Bytea,
    ))
}

fn minhash_decode_cols(args: &[Column]) -> Result<Column> {
    expect_args("minhash_decode(bytea)", args, 1)?;
    let cells = super::column_bytes(&args[0])?;
    let n = row_count(args);
    let mut nulls = combined_nulls(args, n);
    let mut data = Vec::with_capacity(n);
    for i in 0..n {
        if nulls.is_null(i) {
            data.push(Vec::new());
            continue;
        }
        // byte length not a multiple of 8 yields a NULL row
        match parse_signature(cells[i]) {
            Some(sig) => data.push(json_u64_array(&sig)),
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

fn minhash_similarity_cols(args: &[Column]) -> Result<Column> {
    expect_args("minhash_similarity(bytea, bytea)", args, 2)?;
    let a = super::column_bytes(&args[0])?;
    let b = super::column_bytes(&args[1])?;
    let n = row_count(args);
    let mut nulls = combined_nulls(args, n);
    let mut data = Vec::with_capacity(n);
    for i in 0..n {
        if nulls.is_null(i) {
            data.push(0.0);
            continue;
        }
        // parse failure or signature length mismatch is data dependent,
        // the row is NULL rather than failing the batch
        let sim = parse_signature(a[i]).and_then(|sa| {
            parse_signature(b[i])
                .and_then(|sb| zyron_types::fingerprint::minhash_similarity(&sa, &sb).ok())
        });
        match sim {
            Some(v) => data.push(v),
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

fn simhash_cols(args: &[Column]) -> Result<Column> {
    expect_args("simhash(text)", args, 1)?;
    let texts = super::column_strings(&args[0])?;
    let n = row_count(args);
    // 64 bit fingerprint bit cast into the Int64 physical slot
    let data: Vec<i64> = (0..n)
        .map(|i| zyron_types::fingerprint::simhash(texts[i]) as i64)
        .collect();
    Ok(Column::with_nulls(
        ColumnData::Int64(data),
        combined_nulls(args, n),
        TypeId::Int64,
    ))
}

fn simhash_distance_cols(args: &[Column]) -> Result<Column> {
    expect_args("simhash_distance(bigint, bigint)", args, 2)?;
    let a = super::column_ints(&args[0])?;
    let b = super::column_ints(&args[1])?;
    let n = row_count(args);
    let data: Vec<i32> = (0..n)
        .map(|i| zyron_types::fingerprint::simhash_distance(a[i] as u64, b[i] as u64) as i32)
        .collect();
    Ok(Column::with_nulls(
        ColumnData::Int32(data),
        combined_nulls(args, n),
        TypeId::Int32,
    ))
}

fn simhash_similar_cols(args: &[Column]) -> Result<Column> {
    expect_args("simhash_similar(bigint, bigint, int)", args, 3)?;
    let a = super::column_ints(&args[0])?;
    let b = super::column_ints(&args[1])?;
    let t = super::column_ints(&args[2])?;
    let n = row_count(args);
    // hamming distance over 64 bits never exceeds 64 so the threshold clamps there
    let data: Vec<bool> = (0..n)
        .map(|i| {
            zyron_types::fingerprint::simhash_similar(
                a[i] as u64,
                b[i] as u64,
                t[i].clamp(0, 64) as u32,
            )
        })
        .collect();
    Ok(Column::with_nulls(
        ColumnData::Boolean(data),
        combined_nulls(args, n),
        TypeId::Boolean,
    ))
}

fn shingle_cols(args: &[Column], sig: &str, f: fn(&str, usize) -> Vec<String>) -> Result<Column> {
    expect_args(sig, args, 2)?;
    let texts = super::column_strings(&args[0])?;
    let k = super::column_ints(&args[1])?;
    let n = row_count(args);
    let data: Vec<Vec<u8>> = (0..n)
        .map(|i| json_string_array(&f(texts[i], k[i].max(0) as usize)))
        .collect();
    Ok(Column::with_nulls(
        ColumnData::Binary(data),
        combined_nulls(args, n),
        TypeId::Array,
    ))
}

// ---------------------------------------------------------------------------
// fuzzy
// ---------------------------------------------------------------------------

fn double_metaphone_cols(args: &[Column]) -> Result<Column> {
    expect_args("double_metaphone(text)", args, 1)?;
    let texts = super::column_strings(&args[0])?;
    let n = row_count(args);
    let data: Vec<Vec<u8>> = (0..n)
        .map(|i| {
            let (primary, alternate) = zyron_types::fuzzy::double_metaphone(texts[i]);
            json_string_array(&[primary, alternate])
        })
        .collect();
    Ok(Column::with_nulls(
        ColumnData::Binary(data),
        combined_nulls(args, n),
        TypeId::Array,
    ))
}

fn hamming_cols(args: &[Column]) -> Result<Column> {
    expect_args("hamming(text, text)", args, 2)?;
    let a = super::column_strings(&args[0])?;
    let b = super::column_strings(&args[1])?;
    let n = row_count(args);
    let mut nulls = combined_nulls(args, n);
    let mut data = Vec::with_capacity(n);
    for i in 0..n {
        if nulls.is_null(i) {
            data.push(0);
            continue;
        }
        // length mismatch is data dependent, the row is NULL rather than
        // failing the batch
        match zyron_types::fuzzy::hamming(a[i], b[i]) {
            Ok(d) => data.push(d.min(i32::MAX as usize) as i32),
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

// ---------------------------------------------------------------------------
// similarity
// ---------------------------------------------------------------------------

fn cosine_similarity_cols(args: &[Column]) -> Result<Column> {
    expect_args("cosine_similarity(array, array)", args, 2)?;
    let a = super::column_bytes(&args[0])?;
    let b = super::column_bytes(&args[1])?;
    let n = row_count(args);
    let mut nulls = combined_nulls(args, n);
    let mut data = Vec::with_capacity(n);
    for i in 0..n {
        if nulls.is_null(i) {
            data.push(0.0);
            continue;
        }
        // parse failure or vector length mismatch is data dependent,
        // the row is NULL rather than failing the batch
        let sim = parse_vector(a[i]).and_then(|va| {
            parse_vector(b[i])
                .and_then(|vb| zyron_types::similarity::cosine_similarity(&va, &vb).ok())
        });
        match sim {
            Some(v) => data.push(v),
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

fn token_similarity_cols(
    args: &[Column],
    sig: &str,
    f: fn(&[&str], &[&str]) -> f64,
) -> Result<Column> {
    expect_args(sig, args, 2)?;
    let a = column_texts(&args[0])?;
    let b = column_texts(&args[1])?;
    let n = row_count(args);
    let data: Vec<f64> = (0..n)
        .map(|i| {
            let ta = token_list(&a[i]);
            let tb = token_list(&b[i]);
            let ra: Vec<&str> = ta.iter().map(String::as_str).collect();
            let rb: Vec<&str> = tb.iter().map(String::as_str).collect();
            f(&ra, &rb)
        })
        .collect();
    Ok(Column::with_nulls(
        ColumnData::Float64(data),
        combined_nulls(args, n),
        TypeId::Float64,
    ))
}

fn ngram_similarity_cols(args: &[Column]) -> Result<Column> {
    expect_args("ngram_similarity(text, text, int)", args, 3)?;
    let a = super::column_strings(&args[0])?;
    let b = super::column_strings(&args[1])?;
    let sizes = super::column_ints(&args[2])?;
    let n = row_count(args);
    let data: Vec<f64> = (0..n)
        .map(|i| zyron_types::similarity::ngram_similarity(a[i], b[i], sizes[i].max(0) as usize))
        .collect();
    Ok(Column::with_nulls(
        ColumnData::Float64(data),
        combined_nulls(args, n),
        TypeId::Float64,
    ))
}

fn qgram_distance_cols(args: &[Column]) -> Result<Column> {
    expect_args("qgram_distance(text, text, int)", args, 3)?;
    let a = super::column_strings(&args[0])?;
    let b = super::column_strings(&args[1])?;
    let sizes = super::column_ints(&args[2])?;
    let n = row_count(args);
    let data: Vec<i32> = (0..n)
        .map(|i| {
            zyron_types::similarity::qgram_distance(a[i], b[i], sizes[i].max(0) as usize)
                .min(i32::MAX as usize) as i32
        })
        .collect();
    Ok(Column::with_nulls(
        ColumnData::Int32(data),
        combined_nulls(args, n),
        TypeId::Int32,
    ))
}

// ---------------------------------------------------------------------------
// entity resolution
// ---------------------------------------------------------------------------

fn two_string_similarity_cols(
    args: &[Column],
    sig: &str,
    f: fn(&str, &str) -> f64,
) -> Result<Column> {
    expect_args(sig, args, 2)?;
    let a = super::column_strings(&args[0])?;
    let b = super::column_strings(&args[1])?;
    let n = row_count(args);
    let data: Vec<f64> = (0..n).map(|i| f(a[i], b[i])).collect();
    Ok(Column::with_nulls(
        ColumnData::Float64(data),
        combined_nulls(args, n),
        TypeId::Float64,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn utf8_col(vals: &[&str]) -> Column {
        Column::new(
            ColumnData::Utf8(vals.iter().map(|s| s.to_string()).collect()),
            TypeId::Text,
        )
    }

    fn int_col(vals: &[i64]) -> Column {
        Column::new(ColumnData::Int64(vals.to_vec()), TypeId::Int64)
    }

    #[test]
    fn color_rgba_packs_components() {
        let cols = [
            int_col(&[10]),
            int_col(&[20]),
            int_col(&[30]),
            int_col(&[40]),
        ];
        let out = dispatch("color_rgba", &cols, 1).unwrap().unwrap();
        assert_eq!(out.type_id, TypeId::Color);
        match out.data {
            ColumnData::UInt32(v) => {
                assert_eq!(v, vec![(10u32 << 24) | (20 << 16) | (30 << 8) | 40])
            }
            other => panic!("expected UInt32, got {:?}", other),
        }
    }

    #[test]
    fn color_to_hsl_emits_json_and_propagates_null() {
        let mut nulls = NullBitmap::none(2);
        nulls.set_null(1);
        let col = Column::with_nulls(
            ColumnData::UInt32(vec![0xFF0000FF, 0]),
            nulls,
            TypeId::Color,
        );
        let out = dispatch("color_to_hsl", &[col], 2).unwrap().unwrap();
        assert!(!out.nulls.is_null(0));
        assert!(out.nulls.is_null(1));
        match &out.data {
            ColumnData::Binary(cells) => {
                let hsl: Vec<f64> = serde_json::from_slice(&cells[0]).unwrap();
                assert_eq!(hsl.len(), 3);
                assert!(hsl[0].abs() < 1e-9);
                assert!((hsl[1] - 1.0).abs() < 1e-9);
                assert!((hsl[2] - 0.5).abs() < 1e-9);
            }
            other => panic!("expected Binary, got {:?}", other),
        }
    }

    #[test]
    fn minhash_signature_roundtrips_to_full_similarity() {
        let tokens = utf8_col(&[r#"["a","b","c"]"#]);
        let hashes = int_col(&[64]);
        let sig = dispatch("minhash_signature", &[tokens, hashes], 1)
            .unwrap()
            .unwrap();
        match &sig.data {
            ColumnData::Binary(cells) => assert_eq!(cells[0].len(), 64 * 8),
            other => panic!("expected Binary, got {:?}", other),
        }
        let sim = dispatch("minhash_similarity", &[sig.clone(), sig], 1)
            .unwrap()
            .unwrap();
        match sim.data {
            ColumnData::Float64(v) => assert!((v[0] - 1.0).abs() < 1e-9),
            other => panic!("expected Float64, got {:?}", other),
        }
    }

    #[test]
    fn hamming_mismatched_length_row_is_null() {
        let a = utf8_col(&["karolin", "abc"]);
        let b = utf8_col(&["kathrin", "ab"]);
        let out = dispatch("hamming", &[a, b], 2).unwrap().unwrap();
        match &out.data {
            ColumnData::Int32(v) => assert_eq!(v[0], 3),
            other => panic!("expected Int32, got {:?}", other),
        }
        assert!(!out.nulls.is_null(0));
        assert!(out.nulls.is_null(1));
    }

    #[test]
    fn jaccard_reads_json_token_arrays() {
        let a = utf8_col(&[r#"["a","b","c"]"#]);
        let b = utf8_col(&[r#"["b","c","d"]"#]);
        let out = dispatch("jaccard_similarity", &[a, b], 1).unwrap().unwrap();
        match out.data {
            ColumnData::Float64(v) => assert!((v[0] - 0.5).abs() < 1e-9),
            other => panic!("expected Float64, got {:?}", other),
        }
    }

    #[test]
    fn wcag_compliant_passes_and_rejects_bad_level() {
        let black = Column::new(ColumnData::UInt32(vec![0x000000FF]), TypeId::Color);
        let white = Column::new(ColumnData::UInt32(vec![0xFFFFFFFF]), TypeId::Color);
        let out = dispatch(
            "wcag_compliant",
            &[black.clone(), white.clone(), utf8_col(&["AA"])],
            1,
        )
        .unwrap()
        .unwrap();
        match out.data {
            ColumnData::Boolean(v) => assert!(v[0]),
            other => panic!("expected Boolean, got {:?}", other),
        }
        let err = dispatch("wcag_compliant", &[black, white, utf8_col(&["ZZ"])], 1).unwrap();
        assert!(err.is_err());
    }

    #[test]
    fn color_palette_emits_hex_string_cells() {
        let red = Column::new(ColumnData::UInt32(vec![0xFF0000FF]), TypeId::Color);
        let out = dispatch("color_palette", &[red, utf8_col(&["triadic"])], 1)
            .unwrap()
            .unwrap();
        match &out.data {
            ColumnData::Binary(cells) => {
                let hex: Vec<String> = serde_json::from_slice(&cells[0]).unwrap();
                assert_eq!(hex.len(), 3);
                assert!(hex.iter().all(|h| h.starts_with('#')));
            }
            other => panic!("expected Binary, got {:?}", other),
        }
    }

    #[test]
    fn names_owned_elsewhere_return_none() {
        assert!(dispatch("color_rgb", &[], 1).is_none());
        assert!(dispatch("color_hex", &[], 1).is_none());
        assert!(dispatch("not_a_function", &[], 1).is_none());
    }
}
